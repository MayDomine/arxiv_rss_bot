# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-01 11:51:36 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [HAPMoE: Heterogeneity-Aware Automatic Parallelism Planning for Mixture-of-Experts Models Training](https://arxiv.org/abs/2609.39350)

**Authors**: Mengyuan Fan, Peizhuang Cong, Zixiao Huang, Si Xu, Tong Qiao, Yanghao Li, Jing Yang, Tong Yang, Quanlu Zhang, Yu Wang  
**Category**: cs.DC  
**Published**: 2026-10-01  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2609.39350v1  

#### Abstract
As model sizes continue to scale, distributed training has become inevitable. Automatic parallelization techniques can derive efficient training parallelism strategies at low cost while achieving superior performance. The difficulty of this problem is jointly determined by the complexity of the mode...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HAPMoE: Heterogeneity-Aware Automatic Parallelism Planning for Mixture-of-Experts Models Training  
**——核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着模型规模持续扩大，分布式训练已成为必然选择。然而，当前自动并行化系统面临两大挑战：
- **MoE 模型的复杂性**：Mixture-of-Experts（MoE）架构引入动态 token 路由、All-to-All 通信以及专家负载不均衡等问题。
- **硬件异构性**：现代训练集群常混合不同厂商、代际的加速器（如 H800、MI300X、910B），导致计算能力、内存容量和网络带宽差异显著。

现有方法通常只针对其中一个方面进行优化，**缺乏对“MoE + 异构集群”双重挑战的联合建模与搜索能力**，从而难以实现高效训练。

---

### 🚀 提出的新方法：HAPMoE
HAPMoE 是一个**面向 MoE 模型、感知硬件异构性的自动并行规划系统**，其核心设计包括：

#### （1）六维并行空间联合搜索（6D Parallel Search）
支持完整的并行维度组合：
> `(DP, PP, TP, CP, EP, TPE)`  
其中 EP（Expert Parallelism）和 TPE（Tensor-Parallel Experts）是专为 MoE 设计的关键维度。

#### （2）非均匀流水线与数据并行划分（Non-uniform PP/DP Partitioning）
- 不强制每个 stage 分配相同层数或设备数；
- 根据各设备类型的算力（`Ph`）、内存（`Ch`）和带宽（`Bh`）动态分配层和设备；
- 显著缓解因硬件异构导致的 pipeline bubble 和负载失衡。

#### （3）轻量级 MoE 感知代价模型（MoE-Aware Cost Model）
构建基于 profiling 的性能预测模型，涵盖：
- 稀疏专家执行时间
- 路由开销（`troute ∝ S·B`）
- All-to-All 通信体积与延迟
- 阶段级内存占用（含 `Mmoe-extra` 缓冲区）
- 负载不平衡因子（`Pimb`）对关键路径的影响

#### （4）剪枝增强的动态规划算法（Pruning-enhanced DP）
通过三项剪枝策略将搜索空间从 $O(PP \times N^{PP} \times H^{PP})$ 压缩至 $O(PP \times (N/PP)^{PP} \times H)$，确保在 **<1分钟内完成搜索**。

---

### 🔍 相比现有方法的优势
| 维度 | HAPMoE | 典型基线（如 Metis、DeepSpeed-MoE） |
|------|--------|-------------------------------|
| 支持 MoE | ✅ 完整建模 EP/TPE、路由、All-to-All | ❌ 多假设为 dense model |
| 感知异构性 | ✅ 联合建模设备类型差异 | ⚠️ 部分支持，但忽略 MoE 特性 |
| 并行策略灵活性 | ✅ 非均匀 PP/DP，全局搜索 | ❌ 固定划分或局部优化 |
| 搜索效率 | ✅ <1分钟完成 | ⚠️ 可能需小时级模拟或人工调参 |

> ✅ HAPMoE 实现了 **MoE-aware + Heterogeneity-aware** 的统一框架，在真实生产环境中具备高实用性。

---

## 2. 核心实验方法和设置

### 🧪 数据集与模型
未使用传统 NLP 数据集，而是聚焦于 **大规模语言模型训练场景**，采用以下模型配置：

| 模型 | 类型 | 层数 | 参数量 | MoE 设置 |
|------|------|-------|---------|-----------|
| LLaMA-2 7B / 13B | Dense | 32 / 40 | ~7B / 13B | — |
| Mixtral-S (M1) | MoE | 24 | ~47B | 8 experts, top-2 routing |
| Mixtral-L (M2) | MoE | 48 | ~122B | 8 experts, top-2 routing, All-to-All dispatcher |

所有模型均使用 BF16 混合精度训练，ZeRO-1 优化器；微批次大小固定为 1。

---

### 💻 实验集群设置（共 13 种配置）
结合三种主流加速器构建异构与同构集群：
- **NVIDIA H800**
- **AMD MI300X**
- **Ascend 910B**

| 规模 | 示例配置 | 类型 |
|------|----------|------|
| 16卡 | H800(1×8) + MI300X(1×8) | 异构 |
| 24卡 | H800(1×8) + MI300X(2×8) | 异构 |
| 32卡 | H800(2×8) + MI300X(2×8) | 异构 |
| 同构对照组 | 全 H800 / MI300X / 910B | Homogeneous |

详见原文 Table 1。

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Throughput** | TFLOP/s/device，衡量整体计算吞吐 |
| **MFU (Model FLOPs Utilization)** | 加权浮点利用率，反映硬件效率 |
| **Latency** | 单次迭代耗时（ms），直接体现训练速度 |
| **Estimation Error** | 预测延迟与实测误差率，验证模型准确性 |
| **Search Time** | 策略搜索耗时，评估自动化效率 |

---

### 🆚 基线方法对比
根据不同场景选取代表性 baseline：

| 场景 | 对比方法 |
|------|----------|
| Dense + Heterogeneous | Megatron-Infinigence (MI), Alpa, Metis |
| MoE + Homogeneous | DeepSpeed-MoE, Tutel, MI |
| MoE + Heterogeneous |  
&nbsp;&nbsp;- **Metis-style**：固定 EP/TPE，仅搜 PP/DP/TP/CP  
&nbsp;&nbsp;- **HeterMoE-style**：MoE 层级调度，无全局 6D 搜索 |

所有方法保持相同超参、精度、batch size 和测量协议。

---

## 3. 主要实验结果和性能指标

### 📈 性能提升汇总（见 Table 2）

| 模型 | 方法 | Throughput↑ | MFU↑ | Latency↓ |
|------|------|-------------|------|---------|
| Mixtral-S | MI | 1.00× | 1.00× | 1.00× |
|           | HeterMoE-style | 1.40× | 1.46× | 0.72× |
|           | Metis-style | 1.45× | 1.50× | 0.69× |
|           | **HAPMoE** | **1.67×** | **1.72×** | **0.56×** |
| Mixtral-L | MI | 1.00× | 1.00× | 1.00× |
|           | HeterMoE-style | 1.47× | 1.50× | 0.73× |
|           | Metis-style | 1.56× | 1.53× | 0.69× |
|           | **HAPMoE** | **1.78×** | **1.73×** | **0.58×** |

> ✅ **端到端训练吞吐最高提升达 3.2×**（跨多种异构集群）

---

### 🔬 消融实验结果

#### （1）禁用非均匀 PP/DP 的影响（Table 3）
| 集群类型 | 延迟增加 | 吞吐下降 |
|----------|----------|----------|
| 同构（16-h） | 10–18% | ~4–7% |
| 异构（16-2） | **2.10×** | **0.69×** |
| 异构大集群（32-2） | **2.30×** | **0.56×** |

> 🔍 发现：在异构环境下，**强制均匀划分会导致最慢 stage 成为瓶颈**，严重制约整体效率；非均匀划分可带来 **额外 4%–78% 的吞吐增益**。

#### （2）搜索效率与准确性（Figure 5）
- **搜索时间**：在所有配置下均 **<60秒**，平均约 30–50 秒；
- **预测误差**：
  - 延迟估计误差：<15%
  - VRAM 占用误差：<10%
- **小规模 profile 扩展性**：4节点 profile 可较准确预测 16节点表现（误差 ~10%），支持低成本预估。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MoE 与异构硬件必须联合建模**  
   单独处理 MoE 或异构性都无法取得最优性能；HAPMoE 通过统一框架实现了两者的协同优化。

2. **非均匀 PP/DP 在异构场景中至关重要**  
   允许不同 stage 拥有不同层数和设备组合，能有效匹配硬件差异，避免 pipeline stall。

3. **轻量级 profiling + 精准建模即可实现高效搜索**  
   无需复杂仿真或强化学习，仅靠少量 warm-up 迭代即可获得可靠代价模型。

4. **实际部署友好**  
   输出可直接集成进 Megatron-LM，无需修改代码，支持“script-in, script-out”流程。

---

### ⚠️ 局限性
1. **静态规划，缺乏在线适应能力**  
   当前方案基于训练初期的 profile 结果生成计划，若训练过程中出现路由漂移、设备故障或资源抢占，无法动态调整。

2. **依赖 Megatron-LM 架构**  
   当前集成仅适用于 Megatron-LM 风格的 Transformer 实现，尚未扩展至其他框架（如 PyTorch FSDP、JAX）。

3. **未考虑更复杂的拓扑干扰**  
   如大规模部署中的网络拥塞、多租户竞争等未被显式建模。

---

### 🔮 未来工作方向
1. **支持在线 re-planning 机制**  
   在训练过程中定期采集统计信息（如 router 分布变化），触发重新搜索以维持最优策略。

2. **拓展至 Geo-distributed 训练场景**  
   支持跨数据中心的异构集群，结合 WAN 通信优化。

3. **引入学习-based 搜索加速器**  
   利用历史搜索经验训练 surrogate model，进一步缩短搜索时间。

4. **开放更多底层控制接口**  
   与 NCCL、CUDA Stream 等深度集成，实现更细粒度的通信-计算重叠优化。

---

## ✅ 总结
HAPMoE 是首个同时解决 **MoE 模型特性** 与 **硬件异构性** 的自动并行系统。它通过：
- 构建 MoE-aware 的轻量代价模型，
- 实现非均匀 PP/DP 划分，
- 采用剪枝增强的动态规划算法，

在 <1 分钟内生成高性能并行策略，**在真实异构集群上实现最高 3.2× 的端到端吞吐提升**，具有极强的工程实用价值，为未来大规模 MoE 模型在多样化硬件上的高效训练提供了可靠解决方案。

</details>

---

### 2. [AdaKerNet: Neural Kernel Decoding for Task-Adaptive Prediction with Multimodal Large Models](https://arxiv.org/abs/2609.36368)

**Authors**: Konstantinos D. Polyzos, Eleni Oikonomou, Tara Javidi  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2609.36368v1  

#### Abstract
Large foundation models have been introduced with the promise of efficient adaptation to downstream tasks. Yet, under limited supervision, MLLMs, an important class of large foundation models, remain challenging to adapt to various downstream tasks. Adaptation typically relies either on MLLM paramet...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **AdaKerNet: Neural Kernel Decoding for Task-Adaptive Prediction with Multimodal Large Models**  
**核心结论与实验结果总结**

---

## 1. **论文的主要贡献和创新点**

### **解决的问题**
在下游任务监督数据稀缺（scarce-label regime）的情况下，如何高效地适配 **Multimodal Large Language Models (MLLMs)** 进行预测任务。传统方法如 fine-tuning 或参数微调（如 LoRA）面临两大挑战：
- 在标签稀少时容易过拟合，统计效率低；
- 需要访问模型内部参数，对闭源模型（如通过 API 提供的模型）不可行。

因此，本文聚焦于一种更通用、更实用的范式：**保持 MLLM 冻结作为特征提取器，仅训练一个轻量级解码器（decoder）进行下游预测**。

### **提出的新方法：AdaKerNet**
提出 **AdaKerNet** —— 一种全新的、可学习的、任务自适应的神经核解码器（neural kernel decoder），其核心思想是将 **kernel 表示学习** 与 **非线性神经预测** 统一在一个联合优化框架中。

#### **三大核心组件**：
1. **Lipschitz-controlled 特征提取模块**  
   - 使用谱归一化（spectrally normalized）的浅层神经网络，将 MLLM 输出的嵌入 $ e $ 映射为几何受控的中间表示 $ z $。
   - 保证语义距离不会被过度放大，便于后续 kernel 对齐。

2. **参考核（reference kernel）提供软结构先验**  
   - 引入一个固定的正定核函数 $ k_0 $（如 RBF 或 Matérn），作为对样本间相似性的初始假设（structural prior）。
   - 不强制该核完美匹配任务，而是作为“起点”。

3. **任务自适应的神经核变形（adaptive kernel deformation）**  
   - 学习一个显式的非线性特征映射 $ \phi_w(z) $，定义新的可学习核 $ k_b(z,z') = \phi_w(z)^T\phi_w(z') $。
   - 该核被建模为参考核的“变形”：$ k_b = k_0 + \Delta_k $，其中 $ \Delta_k $ 可正可负。
   - 通过联合优化目标，使核既能保留参考结构，又能根据监督信号自适应调整。

4. **联合优化目标**：
$$
\mathcal{L}(w,v) = \lambda \cdot \mathcal{L}_{\text{pred}} + \frac{1}{N^2}\sum_{i,j}[k_b(z_i,z_j) - k_0(z_i,z_j)]
$$
- 第一项：监督预测损失（如 MSE）；
- 第二项：核重建损失，约束学习到的核不能偏离参考核太远；
- 超参数 $ \lambda $ 控制任务适配与结构保留之间的权衡。

### **相比现有方法的优势**
| 方法 | 局限性 | AdaKerNet 如何改进 |
|------|--------|------------------|
| **Fine-tuning / LoRA** | 需要修改 MLLM 参数，不适用于闭源模型；小样本下易过拟合 | 完全冻结 MLLM，仅训练轻量 decoder，适用于任何 embedding API |
| **直接 MLP / Transformer 解码器** | 缺乏结构归纳偏置，在小样本下难以捕捉有效关系 | 引入参考核作为软先验，提供额外无标签的成对相似性信号 |
| **固定核方法（如 KRR）** | 核选择敏感，若选错则性能差 | 可自适应地“修正”参考核以匹配任务需求 |
| **Deep Kernel Learning** | 多结合 GP，推理慢；核形式受限 | 使用显式特征映射 + MLP 预测头，灵活且高效 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
共 **6 个真实世界多模态基准**，涵盖医疗、电商、情感计算、语言教育等领域：

| 数据集 | 模态 | 任务类型 | 标签池大小 |
|-------|------|---------|----------|
| **PAD-UFES-20-AGE** | 图像、文本、表格 | 年龄回归 | 1,612 |
| **PAD-UFES-20-MOLE-LESION** | 图像、文本、表格 | 病变直径对数回归 | 1,043 |
| **AMAZON-FASHION** | 图像、文本、表格 | 商品价格对数回归 | 3,486 |
| **QUECHUA-VALENCE** | 音频、转录文本 | 情感值回归 | 2,800 |
| **SPEECHOCEAN762-FLUENCY** | 音频、转录文本 | 流利度评分回归 | 2,000 |
| **SPEECHOCEAN762-PROSODIC** | 音频、转录文本 | 语调评分回归 | 2,000 |

所有任务均为连续值预测（regression）。

### **MLLM 表示来源**
使用 **4 种不同 MLLM** 提取冻结嵌入：
- **BLIP-2**, **LLaVA-1.5**, **Qwen2.5-VL**（开源）
- **Gemini Embedding 2**（闭源，仅提供 embedding API）

### **实验设置**
- **稀疏标签设定**：训练样本量 $ N \in \{100, 200, 300, 500, 1000\} $
- **评估指标**：测试集上的 **R² 系数**（越高越好），因其尺度不变性适合跨数据集比较
- **重复次数**：10 次独立运行取平均
- **消融研究**：验证各模块贡献、超参数敏感性等

### **基线方法（Baselines）**
| 编号 | 方法 | 描述 |
|-----|------|------|
| **B-I** | MLP Decoder | 三层 MLP，ReLU 激活 |
| **B-II** | Transformer Decoder | 两层 Transformer + mean pooling |
| **B-III** | Autoencoder + Linear Head | 编码器降维后接线性预测头 |
| **B-IV** | Kernel Ridge Regression (KRR) | RBF 核，超参从训练数据选出 |
| **B-V** | SNGP Head | Spectral-normalized Neural Gaussian Process，更灵活的 GP 变体 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
- 在 **60 个实验配置**（6 数据集 × 4 MLLM × 5 标签预算）中：
  - **AdaKerNet 在 58 个配置中取得最高 R²**
  - 剩余 2 个中排名第二
- **平均测试 MSE 下降高达 41%**（相对于所有基线平均）

#### **代表性结果（部分摘录）**
| 数据集/MLLM | N=100 R² | N=500 R² | 最佳基线 | AdaKerNet 提升 |
|------------|---------|---------|----------|---------------|
| PAD-UFES-20-AGE / Qwen2.5-VL | 0.2771 vs -0.0157 (MLP) | 0.3690 vs 0.1150 | +41.4% MSE↓ |
| QUECHUA-VALENCE / Gemini 2 | 0.0620 vs -0.0545 (MLP) | 0.2719 vs 0.0361 | 所有 N 下均为正 R²，而多数基线为负 |
| SPEECHOCEAN762-FLUENCY / Gemini 2 | 0.2033 vs 0.2092 (MLP) | 0.4204 vs 0.3611 | 尽管略低于 MLP 在 N=100，但在更大预算下显著领先 |

> ✅ **特别亮点**：在多个任务上，当基线表现极差甚至不如均值预测时（R² < 0），AdaKerNet 仍能稳定获得正 R²。

### **与基线方法的对比结果**
- **全面超越所有五类基线**：
  - 超越 MLP 和 Transformer（B-I/B-II）在 **58/60** 场景
  - 超越 KRR（B-IV）和 SNGP（B-V）在 **全部 60** 场景
- 即使在 **非稀疏标签场景**（N > 1000）也保持优势，在 16 个大样本配置中 **12 个最优**

### **消融实验结果**
#### **(1) Lipschitz-controlled 特征的有效性**
- 使用 $ z $ 替代原始嵌入 $ e $ 后：
  - **MLP 解码器**：33/35 设置下提升，最大 MSE↓达 **28.5%**
  - **KRR 解码器**：34/35 设置下提升，最大 MSE↓达 **38.7%**
- 说明该特征提取模块本身就能大幅提升下游性能。

#### **(2) 联合学习核与预测器的价值**
- 相比直接在 $ z $ 上训练 MLP：
  - AdaKerNet 在 **54/60** 配置中更优，最大相对 R² 提升 **25.2%**
- 相比更深的 MLP（移除核重建损失）：
  - 仍能在 **53/60** 中胜出，表明增益来自 **结构正则化** 而非单纯增加深度

#### **(3) 自适应核的优越性**
- 将 AdaKerNet 学得的核 $ k_b $ 用于 KRR（替换原预测头）：
  - 在 **217/240** 配置中优于原始参考核 $ k_0 $
  - 证明学到的核本身更具任务相关性

#### **(4) 对参考核选择的鲁棒性**
- 使用 **RBF、Matérn-1/2、3/2、5/2** 四种核作为 $ k_0 $：
  - AdaKerNet 在 **80/80** 配置中优于 B-I/B-II
  - 表明其性能不依赖特定核的选择

#### **(5) 与 Tuned Random Features (TRF) 对比**
- TRF 是另一种 kernel 学习方法（基于谱密度）
- AdaKerNet 在 **26/30** 配置中更优，尤其在 **N=100** 时增益最大（高达 **137.7%** R² 提升）

---

## 4. **关键结论和发现**

### **主要发现**
1. **冻结 MLLM + 任务自适应解码器是高效且实用的范式**，尤其适用于闭源模型和小样本场景。
2. **引入参考核作为软先验**，并通过可学习变形机制进行任务适配，能有效平衡 **结构引导** 与 **数据驱动灵活性**。
3. **Lipschitz-controlled 特征提取** 显著改善了嵌入空间的几何性质，有利于后续 kernel 学习。
4. **联合优化核表示与神经预测器** 比单独优化任一部分更有效，体现了“双路协同”的优势。
5. **AdaKerNet 在多种 MLLM、模态组合、标签预算下均表现出强鲁棒性和一致性优势**。

### **方法的局限性**
- 当标签极度稀少（如 N=100）且任务与参考核严重不匹配时，初期 Lipschitz 特征学习可能无法有效增强结构。
- 当前实现依赖于手动选择参考核类型和超参数 $ \lambda $，尚未完全自动化。
- 主要验证于回归任务，分类任务扩展需进一步探索。

### **未来工作方向**
- 将自适应核变形机制扩展至 **分类、排序、生成等其他任务**。
- 探索 **动态参考核选择机制** 或 **多核融合策略**。
- 结合不确定性估计（如 SNGP）构建更可靠的预测系统。
- 探索在 **联邦学习** 或 **终身学习** 场景下的应用潜力。

---

> 🔚 **总结**：AdaKerNet 提出了一种新颖且强大的解码框架，成功实现了在 **冻结 MLLM 表示上进行高效、鲁棒的小样本多模态预测**。其实验设计严谨，结果显著，为 MLLM 的轻量化下游适配提供了重要新思路。

</details>

---

### 3. [BASE: Batch-Aware Selection of Experts Using Predicted Removal Error for Efficient MoE Decoding](https://arxiv.org/abs/2609.36222)

**Authors**: Ali Abbasi, Justin Shi, Soheil Kolouri  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2609.36222v1  

#### Abstract
Large language models are increasingly expensive to serve. In large-scale serving systems, autoregressive decoding is often bottlenecked by transferring model weights from accelerator high-bandwidth memory into on-chip SRAM. Mixture-of-experts (MoE) models reduce computation by activating only a sma...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**BASE: Batch-Aware Selection of Experts Using Predicted Removal Error for Efficient MoE Decoding**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在大规模语言模型（LLM）推理中，**Mixture-of-Experts (MoE)** 架构通过每 token 激活少量专家来减少计算量。然而，在**批处理（batched decoding）场景下**，不同请求可能激活不同的专家，导致整个 batch 需要加载大量不重叠的专家权重。

这带来了严重的**内存带宽瓶颈**：尽管单个 token 是稀疏的，但 batch 级别的“活跃专家集合”可能非常大，从而抵消了 MoE 的效率优势。现有方法大多基于路由得分（router scores）独立选择专家，未能有效控制 batch 层面的专家负载。

---

### ✅ 提出的新方法：BASE

本文提出 **BASE**（**Batch-Aware Selection of Experts**），一种面向高效 MoE 推理的批感知专家选择机制，其核心思想是：

> **以最小化 MoE 层输出变化为目标，预测并保留对当前 batch 影响最大的专家。**

#### 创新点如下：

1. **从输出误差角度定义专家重要性**  
   不再依赖 router 权重或静态统计量，而是提出使用“**移除专家后对 MoE 输出的平方误差变化**”作为专家重要性的度量标准（removal error）。

2. **忽略交叉项仍能保持高质量选择**  
   虽然精确误差包含专家之间的交叉项（cross terms），但作者证明这些项在实际中影响极小（<3.2% 误差增加），因此可安全忽略，实现**独立打分 + 贪心选择**。

3. **轻量级线性预测器估计专家输出能量**  
   在离线校准阶段训练一个 per-expert 的线性模型，用输入 token 表示 $ z_t $ 预测该专家的输出能量 $ \|E_u(z)\| $，进而估算其 removal cost。

4. **端到端 GPU 内核优化支持高效实现**  
   开发定制化的 Triton GPU kernels，集成预测、聚合、选择和 backfill 流程，仅引入约 3–7% 的额外延迟。

---

### ✅ 相比现有方法的优势

| 方面 | BASE | 现有方法（如 SERE/OEA/ExFold） |
|------|------|-------------------------------|
| **选择依据** | 基于预测的输出误差影响 | 基于 router 权重或固定校准的输出范数 |
| **批处理协调** | 显式优化 batch 级专家集合 | 多为 token 独立决策的联合 |
| **动态适应性** | 每个 token 动态预测输出能量 | 使用静态统计，无法反映上下文差异 |
| **替换策略** | 使用 backfill（次优偏好专家） | 使用相似专家 substitution（效果差） |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集

在 **8 个生成式基准任务**上进行全面评估：

- **CMMLU**: 中文多任务理解
- **BBH** (BIG-Bench Hard)
- **MATH**, **GSM8K**, **MATH-401**: 数学推理
- **BoolQ**: 是非问答
- **MBPP**: Python 编程生成
- **HumanEval**: 函数级代码生成

最终报告 **平均得分（Avg.）** 和 **吞吐量（tok/s）**。

---

### ✅ 实验设置

| 参数 | 设置 |
|------|------|
| **模型** | Qwen3-30B-A3B, Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite |
| **硬件** | 单张 NVIDIA H100 80GB GPU |
| **Batch Size** | 16（主实验），也测试 8–128 范围 |
| **Prefill** | 所有方法均使用 dense 模型执行（避免复杂化） |
| **Decoding** | 仅在此阶段应用专家裁剪 |
| **生成长度** | 固定为 384 tokens，禁用 EOS 提前终止 |
| **精度** | bf16 |
| **框架** | vLLM 0.9.2（V0 引擎） |

---

### ✅ 基线方法对比

与以下四种 batch-aware 方法比较：

- **SERE**: 基于 top-k 专家 union，并用相似专家替换缺失者
- **OEA**: 类似 union 策略，采用 backfill
- **Lynx**: 结合 router confidence 与 batch 内流行度
- **ExFold**: 使用校准期间测量的专家输出范数加权，固定预算

所有方法调整超参以匹配 BASE 的解码速度，确保公平比较。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Table 1）

#### 🔹 在 **Qwen3-30B-A3B** 上（受限预算 M=10 vs. S=1/ko=1）：

| 方法 | 平均准确率（Avg. Acc.） | 吞吐量（tok/s） |
|------|------------------------|----------------|
| Dense（全量） | 82.50 | 845.3 |
| OEA (ko=1) | 43.01 | 1372.5 |
| SERE (S=1) | 37.54 | 1452.9 |
| **BASE (M=10)** | **76.38** | **1463.5** |

➡️ **提升高达 29.5 分（vs 最强 baseline）**，且吞吐更高！

#### 🔹 放松预算下（M=16）表现：

| 方法 | 平均准确率 | 吞吐量 |
|------|----------|--------|
| Dense | 82.50 | 845.3 |
| OEA (ko=2) | 70.15 | 1190.7 |
| **BASE (M=16)** | **82.47** | **1350.2** |

➡️ **接近 dense 性能（仅差 0.03 分）**，同时**快 60%**！

---

### ✅ 综合性能优势

- 在三个 MoE 架构上，**BASE 在相同速度下平均提升 3.0～29.5 分**。
- 在高预算设置下：
  - 吞吐量比 dense 高 **22%～60%**
  - 准确率损失 < **0.4 分**
- 在最极端压缩下（M=10）：
  - 比 dense 快 **73%**
  - 准确率下降 6.1 分 → 性价比极高

---

### ✅ 消融实验结果

#### 📊 **Selection Rule Ablation（Table 2）**

在 Qwen3-30B-A3B 上固定活跃专家数量，比较不同排序策略：

| 排序方式 | 保留的能量比例（Retained Energy %） | 平均准确率 |
|---------|-------------------------------|------------|
| Top-1 Union | 85.91 | 58.68 |
| Router Weight Sum | 82.61 | 42.87 |
| Squared Router Weight | 90.98 | 55.50 |
| Static Expert Energy | 92.65 | 64.46 |
| **BASE (Linear Predictor)** | **96.47** | **73.50** |
| Oracle（理想上限） | 100.00 | 73.58 |

✅ **结论**：结合预测输出能量显著优于仅靠 router 权重的方法。

---

#### 📊 **Fill Rule Ablation（Table 3）**

比较三种处理未加载专家的方式（M=10）：

| 策略 | 平均准确率 | tok/s |
|------|-----------|-------|
| Drop（直接丢弃） | 74.20 | 1473.4 |
| **Backfill（用 token 自身次优专家填充）** | **76.45** | 1479.3 |
| Substitution（用相似专家替代） | 27.49 | 1477.0 |

✅ **结论**：co-routed 专家输出近似正交（见 Fig 1a），substitution 效果极差；**backfill 是最优选择**。

---

#### 📊 **Batch Size 影响分析（Fig 3）**

- BASE 在各种 batch size（8–128）下始终优于 OEA/SERE
- 尤其在小 batch 下优势更明显（因专家多样性更高）
- 性能随 batch 增大更稳定，说明方法具有良好的扩展性

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **MoE 的 token 级稀疏 ≠ batch 级高效**  
   批处理中专家集合膨胀严重，成为内存瓶颈主因。

2. **专家选择应基于对输出的影响而非路由分数**  
   removal error 是更合理的优化目标。

3. **交叉项可忽略，独立评分足够有效**  
   实验证明 additive approximation 导致的误差 < 3.2%，不影响选择质量。

4. **专家输出方向高度正交**  
   图 1(a) 显示 co-routed 专家输出余弦相似度集中在 0 附近（均值 ~0.04），意味着不能用“相似专家”替代。

5. **轻量预测器即可实现高性能压缩**  
   线性模型已足够捕捉上下文相关的输出能量变化。

6. **Backfill > Substitution**  
   利用 token 自身的次优专家比跨 token 替代更可靠。

---

### ⚠️ 方法的局限性

- **需要离线校准（calibration）**  
  需要在部署前收集数据并训练每个专家的预测器，增加了准备成本。
- **校准数据需具代表性**  
  若校准数据分布与真实推理数据偏差较大，预测准确性可能下降。
- **目前仅用于 decoding 阶段**  
  prefill 阶段仍使用 dense，未来可探索统一加速。

---

### 🔮 未来工作方向

1. **在线自适应校准机制**  
   动态更新预测器以适应输入分布漂移。

2. **将 BASE 应用于 prefill 阶段**  
   设计适用于长序列并行处理的批优化策略。

3. **与其他系统技术结合**  
   如与 **expert offloading / prefetching / sharding** 联合优化，进一步降低显存压力。

4. **扩展至 multimodal MoE 模型**  
   探索视觉-语言等混合专家架构中的通用性。

5. **理论分析 removal error approximation 的边界条件**  
   更深入理解何时 additive 近似会失效。

---

## ✅ 总结

**BASE 是一项针对 MoE 批处理推理瓶颈的重要改进**。它通过：

- **以输出误差为核心目标**
- **轻量预测 + 批感知选择**
- **高效 GPU 实现**

实现了在极低专家预算下的高质量推理，在多个 MoE 模型上大幅超越现有方法。其设计原则——**关注实际影响而非中间信号**——为后续高效推理研究提供了新范式。

</details>

---

### 4. [AIMS: An Agentic AI Framework for Sim-to-Real Multi-Modal ISAC](https://arxiv.org/abs/2609.39964)

**Authors**: Yijie Bian, Kai Zhang, Wei Guo, Zixin Wang, Shenghui Song, Jun Zhang, Khaled B. Letaief  
**Category**: cs.AI  
**Published**: 2026-10-01  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.39964v1  

#### Abstract
Multi-modal integrated sensing and communication (ISAC) enables environmental perception and reliable connectivity for intelligent wireless networks. Data-driven multi-modal ISAC models depend heavily on annotated real-world data to learn relationships across sensing and wireless observations, there...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**AIMS: An Agentic AI Framework for Sim-to-Real Multi-Modal ISAC**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
多模态集成感知与通信（multi-modal ISAC）依赖大量标注的真实世界数据进行训练，而真实数据采集成本高、部署场景特定性强，严重制约了模型的可扩展性。虽然仿真数据（synthetic data）可以缓解数据需求，但**仿真到现实（sim-to-real）的迁移效果受限于场景、感知、无线配置和学习模块之间的物理不一致性和耦合依赖关系**。

现有方法在适应新部署时缺乏对这些跨域依赖的系统性建模，导致配置冲突、样本错位和迁移失败。

---

### 🚀 提出的新方法：AIMS 框架

提出了一种**基于 Agentic AI 的 sim-to-real 多模态 ISAC 框架——AIMS**（An Agentic AI Framework for Sim-to-Real Multi-Modal ISAC），其核心是：

- **自然语言驱动**：接受自然语言形式的部署请求（如任务类型、传感器配置、真实数据预算等）。
- **双智能体协同架构**：
  - **Scene Construction Agent**：负责生成地理锚定、物理一致的同步感知与无线数据。
  - **Scene Understanding Agent**：负责配置任务相关的模态选择、MoE 学习结构及零样本/少样本迁移策略。
- **共享实验状态（Shared Experiment State）**：维护部署配置、中间产物、验证证据和依赖关系，支持动态决策更新。
- **结构化领域知识引导**：利用任务、能力、依赖三类知识实现需求解析、依赖推理和反馈驱动重规划。

---

### 🔍 相比现有方法的优势

| 方面 | AIMS 的优势 |
|------|-------------|
| **配置一致性** | 显式建模物理状态 `(g, s)` 的共享依赖，确保感知与无线数据在时空和几何上严格对齐。 |
| **灵活性与自动化** | 支持从自然语言请求自动推导出完整的 sim-to-real 配置流程，无需人工干预。 |
| **可复用性与效率** | 利用验证过的中间产品（如静态场景、轨迹），仅重新生成受影响部分，提升执行效率。 |
| **鲁棒性** | 通过 validation evidence 实现反馈驱动的局部重规划，应对部署条件变化或配置冲突。 |

> 💡 **核心创新**：将传统“固定流水线”式的 sim-to-real 流程升级为**具备认知推理、依赖管理与自我修正能力的 agentic 工作流**。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 主要使用真实世界的 **DeepSense 6G** 多模态 ISAC 数据集作为目标域评估基准。
- 覆盖三个典型场景（Scenario 3, 4, 9）：
  - 包含 RGB 图像、LiDAR、GPS、毫米波无线信道测量等多模态数据。
  - 支持车辆检测（sensing task）和 beam prediction（communication task）。

---

### ⚙️ 实验设置

#### （1）仿真环境重建
- 使用 **OpenStreetMap + Overture Maps** 进行地理重建。
- 使用 **CARLA** 模拟动态交通与传感器观测（RGB/LiDAR/GPS）。
- 使用 **Blender + Sionna RT ray-tracing engine** 进行电磁传播仿真，生成信道与 beam label。

#### （2）学习模型
- 采用 **Mixture-of-Experts (MoE)** 架构：
  - RGB → ResNet-18
  - LiDAR → PointNet
  - GPS → Position Encoder
  - Gating Network 实现上下文感知的模态融合
- 训练方式：
  - 先在合成数据上预训练（synthetic pretraining）
  - 在 `n_real > 0` 时进行有限参数微调（few-shot adaptation），仅更新指定参数组 `UT`

#### （3）评估任务
| 任务 | 输入模态 | 输出 | 指标 |
|------|--------|------|------|
| **Vehicle Detection** | RGB + GPS | 车辆边界框与类别 | AP, AP50 |
| **Beam Prediction** | RGB + LiDAR + GPS | 最优 beam index | Top-3 Accuracy, Top-5 Accuracy |

#### （4）对比基线方法
| 类型 | 基线名称 | 描述 |
|------|--------|------|
| **Sim-to-Real Baselines** | Scenario-agnostic Simulation | 通用场景生成，无地理重建 |
| | Map-derived Simulation | 仅用 OSM 建模建筑体积，固定轨迹 |
| | Codebook-agnostic Simulation | 使用 DFT codebook 替代实际设备 codebook |
| **Fusion Baselines** | Feature Concatenation | 特征拼接代替 MoE 融合 |
| | Multi-modal Transformer | 使用 Transformer 做跨模态注意力 |
| **Upper Bound** | Real-data Training Benchmark | 在完整真实训练集上训练 MoE 模型 |

#### （5）Orchestration Benchmark
- 设计两个测试层级共 **140 条自然语言请求**（对应 60 个标准案例）：
  - **Normal Tier**：明确可行的任务请求
  - **Challenging Tier**：隐含输出、间接描述、依赖冲突、未支持目标等复杂情况
- 评估指标：
  - Plan-TSR（任务成功率）
  - CSR（约束满足率）
  - Dependency-F1 / Capability-F1
  - First-Try Success, Avg. Attempts

---

## 3. 主要实验结果和性能指标

### 📊 Vehicle Detection 性能（Scenarios 3 & 9）

| 方法 | AP ↑ | AP50 ↑ |
|------|-------|--------|
| Scenario-agnostic Simulation | ~0.3–0.4 | ~0.5–0.6 |
| Map-derived Simulation | ~0.45 | ~0.65 |
| **Proposed (Zero-shot)** | **~0.55–0.63** | **~0.71–0.76** |
| **Proposed (Few-shot, 20 samples)** | **~0.65–0.73** | **~0.80–0.85** |
| Real-data Training | ~0.80 | ~0.90 |

> ✅ **结论**：即使零样本迁移，AIMS 显著优于所有基线；加入 20 个真实样本后进一步逼近全量训练性能。

---

### 📊 Beam Prediction 性能（Scenarios 3 & 4）

| 方法 | Top-3 Acc ↑ | Top-5 Acc ↑ |
|------|------------|------------|
| Scenario-agnostic Simulation | ~0.45–0.55 | ~0.60–0.70 |
| Codebook-agnostic Simulation | ~0.50–0.58 | ~0.65–0.72 |
| Map-derived Simulation | ~0.60–0.65 | ~0.70–0.75 |
| **Proposed (Zero-shot)** | **~0.70–0.78** | **~0.80–0.88** |
| **Proposed (Few-shot, 20 samples)** | **~0.80–0.88** | **~0.88–0.93** |
| Real-data Training | ~0.90–0.95 | ~0.95–0.98 |

> ✅ **关键发现**：
> - 地理重建 + 动态轨迹模拟显著提升性能（vs Scenario-agnostic）
> - 实际 codebook 对齐至关重要（vs Codebook-agnostic）
> - MoE 融合机制优于特征拼接与 Transformer

---

### 🤖 Agentic Orchestration 性能对比

| Tier | Method | Plan-TSR (%) | CSR (%) | Dependency-F1 (%) | Capability-F1 (%) | First-Try (%) |
|------|--------|---------------|----------|--------------------|-------------------|----------------|
| Normal | Direct LLM | 60.0 | 100.0 | 88.1 | 82.2 | 60.0 |
| | Tool-only AIMS | 72.9 | 98.6 | 87.0 | 77.4 | 52.9 |
| | **AIMS** | **100.0** | **100.0** | **100.0** | **100.0** | **78.6** |
| Challenging | Direct LLM | 48.6 | 97.1 | 63.6 | 61.3 | 48.6 |
| | Tool-only AIMS | 60.0 | 98.6 | 72.5 | 65.3 | 45.7 |
| | **AIMS** | **88.6** | **100.0** | **92.4** | **90.3** | **68.6** |

> ✅ **消融分析结论**：
> - **结构化领域知识** 是性能跃升的关键（vs Tool-only AIMS）。
> - **验证反馈机制** 支持高效重规划，平均尝试次数更低（1.23 vs 1.29）。
> - 在挑战性请求下仍保持高鲁棒性（88.6% 成功率）。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **物理一致性是 sim-to-real 成功的前提**  
   - 场景、感知、无线、学习四者必须在共享物理状态 `(g, s)` 下协同构建，否则无法保证跨域对齐。

2. **Agentic 架构显著提升配置正确性与适应性**  
   - 结构化知识 + 反馈驱动机制使系统能在部署变更时自动识别并修复受影响组件。

3. **MoE 融合机制更适合多模态 ISAC 任务**  
   - 相比简单拼接或 Transformer，MoE 更灵活地融合异构模态，在零样本和少样本下表现更优。

4. **少量真实数据即可显著缩小 sim-to-real 鸿沟**  
   - 仅需 20 个标注样本进行 few-shot adaptation，即可大幅提升检测与 beam prediction 性能。

---

### ⚠️ 局限性

1. **依赖高质量仿真工具链**  
   - 当前框架依赖 CARLA、Sionna RT、Blender 等外部工具，限制了端到端优化空间。

2. **LLM 推理延迟影响实时性**  
   - 当前使用 Qwen3-14B 进行规划，单次推理耗时较高，难以用于低延迟在线部署。

3. **尚未覆盖极端环境或罕见事件**  
   - 如恶劣天气、突发遮挡等复杂场景的泛化能力有待验证。

---

### 🔮 未来工作方向

1. **闭环性能反馈机制**  
   - 将下游任务性能（如 beam prediction 准确率）作为 reward，驱动 AIMS 自动优化仿真配置。

2. **轻量化 agent 设计**  
   - 探索 small models + tool augmentation 路径，降低推理开销，提升部署效率。

3. **跨城市/跨场景迁移能力增强**  
   - 引入 domain-invariant 表示学习，减少对每个新地点重复重建的需求。

4. **支持更多 ISAC 任务类型**  
   - 扩展至手势识别、人体姿态估计、语义分割等高级感知任务。

---

> ✅ **总体评价**：  
> AIMS 是首个将 **Agentic AI** 系统性应用于 **multi-modal ISAC sim-to-real** 的框架，不仅在性能上超越多种基线，更重要的是提出了一个**可解释、可验证、可演进的智能部署范式**，为未来 6G 智能网络的自动化部署提供了重要技术路径。

</details>

---

### 5. [Efficient Expert-Parallel Communication on PCIe-Connected Consumer GPUs](https://arxiv.org/abs/2609.40093)

**Authors**: Jaehwan Lee, Sangmin Lee, Chaewon Kim, Junsik Shin, Jaejin Lee  
**Category**: cs.DC  
**Published**: 2026-10-01  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.40093v1  

#### Abstract
Expert parallelism (EP) enables inference of large Mixture-of-Experts (MoE) models by placing their experts across multiple GPUs, but requires substantial communication between GPUs at every MoE layer. As contemporary MoE models activate more experts per token, this communication accounts for a grow...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Efficient Expert-Parallel Communication on PCIe-Connected Consumer GPUs*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代 **Mixture-of-Experts (MoE)** 模型在推理时广泛采用 **Expert Parallelism (EP)** 将专家分布到多个 GPU 上。然而，在消费级 GPU（如 RTX 40/50 系列）组成的 PCIe 系统中，由于缺乏 **NVLink** 或 **GPUDirect P2P** 支持，所有 GPU 间通信必须通过 **CPU 内存中的 bounce buffer** 中转，导致以下三大瓶颈：

- **PCIe 链路竞争**：传统 ring 算法造成冗余的多次 PCIe 数据传输。
- **SM 资源竞争**：NCCL 使用 SM 执行通信内核，与专家计算（如 GEMM）争抢资源，限制了通信-计算重叠。
- **高延迟同步**：完成标志（completion flag）需放在 CPU 内存，轮询操作引发频繁 PCIe 往返。

现有 EP 专用通信库（如 DeepEP、pplx-kernels、NCCL EP）均依赖 P2P 连接，无法在消费级系统运行，因此主流框架（如 vLLM）只能退而使用 NCCL，性能受限。

---

### 提出的新方法：ThunderEP
作者提出 **ThunderEP** —— 一种专为无 P2P 的 PCIe 消费级 GPU 系统设计的高效 EP 通信方案，其核心创新包括：

#### （1）**单步集体通信算法（Single-step Collective Algorithm）**
- 替代 NCCL 的多跳 **ring 算法**，采用 **All-Gather + Reduce-Scatter** 架构。
- 在 dispatch 阶段，每个 GPU **一次性写入 host memory**，其余 GPU 直接读取，避免中间 relay。
- 分析表明：PCIe 传输量从 `2(N−1)S` 降至 `NS`，理论节省达 **2×**（当 GPU 数 > 2）。

#### （2）**基于 DMA 引擎的数据传输设计**
- 利用 GPU 的 **DMA engine** 执行数据搬移，**不占用 SM 资源**，从而实现与专家 GEMM 的真正并行。
- 采用双流（dual-stream）流水线，利用 PCIe 全双工特性，实现 **D2H 和 H2D 重叠**。

#### （3）**轻量级同步机制（Lightweight Synchronization）**
- 将 completion flag 与数据分离存储，减少内存占用。
- 使用少量线程轮询 flag，并引入 **per-sender waiting policy**，允许接收方在任意发送方完成后立即开始接收，避免全局等待。
- 减少 PCIe 同步事务次数，显著降低小消息延迟。

#### （4）**细粒度通信-计算重叠**
- 借助上述同步机制，实现 **按 sender 粒度触发计算**，而非等待整个 All-Gather 完成。
- 在 prefill 阶段有效隐藏通信延迟。

---

### 相比现有方法的优势
| 维度 | NCCL（Baseline） | ThunderEP |
|------|------------------|-----------|
| 通信路径 | 多跳 relay（ring） | 单步广播 |
| 传输引擎 | SM kernel | DMA engine |
| SM 占用 | 是 | 否（仅 reduce 阶段） |
| 同步机制 | Inline flag + 多线程轮询 | 分离 flag + 少量线程 + per-sender 触发 |
| 通信-计算重叠 | 有限（需等全部完成） | 细粒度（可逐 shard 开始） |
| P2P 依赖 | 不需要，但性能差 | 完全无需 |

---

## 2. 核心实验方法和设置

### 使用的模型
- **Qwen3-30B-A3B**（BF16）
- **GPT-OSS-20B** 和 **GPT-OSS-120B**（MXFP4）

### 实验平台
| 系统 | RTX 4090 系统 | RTX 5090 系统 |
|------|---------------|---------------|
| GPU 数量 | 6 × RTX 4090 | 6 × RTX 5090 |
| 互连 | PCIe 4.0 ×16 | PCIe 5.0 ×16 |
| P2P 支持 | ❌（驱动禁用） | ❌ |
| 主机内存 | 8× DDR5-4800 32GB | 同左 |
| NUMA 结构 | 单节点 | 单节点 |

### 评估指标
- **通信层**：
  - 小消息（1KB–1MB）：**延迟（Latency）**
  - 大消息（1MB–1GB）：**带宽（Bandwidth）**
- **端到端推理**：
  - **Prefill**：TTFT（Time to First Token），单位 ms
  - **Decode**：TPOT（Time Per Output Token），单位 ms/token
  - **吞吐提升倍数（Speedup）**

### 基线方法对比
- **vLLM (v0.27.1)**：默认使用 NCCL All-Gather + Reduce-Scatter
- **SGLang (v0.5.18)** 和 **Megatron-Core (v0.19.2)**：作为端到端推理框架对比
- **NCCL (v2.30.7)**：用于底层通信原语对比

> 注：所有 EP 专用库（如 DeepEP、pplx-kernels）因依赖 P2P，**无法在测试系统运行**。

---

## 3. 主要实验结果和性能指标

### 通信层性能（6-GPU 单节点）

#### ✅ Dispatch（All-Gather）
| 指标 | RTX 4090 | RTX 5090 |
|------|----------|----------|
| **平均延迟加速比** | **2.30×** | **2.65×** |
| **1GB 带宽** | 26.5 GB/s vs 20.8 GB/s (**1.27×**) | 35.3 GB/s vs 23.0 GB/s (**1.53×**) |

> ThunderEP 显著降低小消息延迟，大消息带宽更高，得益于更少 PCIe 传输和双向流水。

#### ✅ Combine（Reduce-Scatter）
| 指标 | RTX 4090 | RTX 5090 |
|------|----------|----------|
| **平均延迟加速比** | **1.91×** | **2.14×** |
| **1GB 带宽** | 接近持平（21.0 vs 21.3 GB/s） |

> Combine 通信量相同，优势主要来自 **更短的同步链** 和 **避免 ring 中继**。

---

### 端到端推理性能（RTX 5090 系统）

#### Prefill 性能（batch=24, seq_len=256~4K）
| 模型 | ThunderEP vs vLLM | vs SGLang | vs Megatron-Core |
|------|--------------------|------------|-------------------|
| Qwen3-30B-A3B | **1.42×** | 1.58× | 1.59× |
| GPT-OSS-20B | 1.33× | — | — |
| GPT-OSS-120B | 1.22×（最高 **1.66×**） | — | —（OOM） |

> Megatron-Core 在 GPT-OSS-120B 上因无 MXFP4 支持而 OOM。

#### Decode 性能（seq_len=4K, batch=6~96）
| 模型 | ThunderEP vs vLLM | vs SGLang | vs Megatron-Core |
|------|--------------------|------------|-------------------|
| 平均吞吐提升 | **1.16×** | **1.28×** | **1.90×** |

> Decode 提升较小，因 token 数少，难以形成有效 pipeline，但低延迟通信仍带来收益。

---

### 消融实验（Ablation Study，Qwen3-30B-A3B on RTX 5090）
| 配置 | Prefill Speedup | Decode Speedup |
|------|------------------|----------------|
| Baseline (vLLM + NCCL) | 1.00× | 1.00× |
| + Algorithm & Sync (S4.2 & S4.4) | 1.15× | 1.15× |
| + DMA Transfer (S4.3) | 1.23× | 1.15× |
| + DMA-based Overlap (S4.5) | **1.47×** | 1.15× |

> 结论：
- 单步算法和同步优化对 decode 提升显著（小消息主导）。
- DMA + 重叠是 prefill 性能飞跃的关键。

---

## 4. 关键结论和发现

### 主要发现
1. **消费级 GPU 上的 EP 通信瓶颈严重**：传统 NCCL 在无 P2P 场景下性能低下，主要受限于：
   - 冗余 PCIe 传输（ring relay）
   - SM 资源竞争
   - 高开销同步（host memory polling）

2. **ThunderEP 显著提升通信效率**：
   - Dispatch 通信延迟降低 **2.00×**
   - Combine 通信延迟降低 **1.53×**
   - 端到端推理最高提速 **1.66×（prefill）** 和 **1.26×（decode）**

3. **DMA + 细粒度同步是关键**：
   - 使用 DMA 引擎释放 SM 资源，支持真正重叠。
   - per-sender waiting 和分离 flag 设计有效降低同步开销。

4. **真实场景验证有效**：
   - 在 ShareGPT 和 LMSYS-Chat-1M 等真实对话数据集上，性能增益与合成数据一致（Prefill ~1.38×, Decode ~1.13–1.14×）。

---

### 局限性
1. **未实现 routing-aware All-to-All**：
   - 当前仍使用 All-Gather + Reduce-Scatter，存在冗余通信。
   - 因 routing 元数据在 GPU 生成，而 DMA 需 CPU 发起，协调开销大。

2. **Decode 阶段重叠有限**：
   - 小 batch 下难以构建有效 pipeline，通信-计算重叠主要在 prefill 阶段生效。

3. **依赖固定 CUDA graph 的 decode 优化受限**：
   - 为兼容 vLLM 的 graph replay，decode 通信仍由 SM 执行，未能完全发挥 DMA 优势。

---

### 未来工作方向
1. **实现高效的 routing-aware All-to-All** 通信，进一步减少传输量。
2. **探索 CPU-GPU 协同调度机制**，支持动态 routing 下的 DMA 传输。
3. **扩展至多节点系统**，结合 RDMA 或 NVMe-oF，支持更大规模 MoE 推理。
4. **支持更多量化格式和稀疏模式**，适配未来 MoE 架构演进。

---

> **总结**：ThunderEP 是首个专为 **无 P2P 的消费级 GPU** 设计的高效 EP 通信方案，通过 **单步算法 + DMA 传输 + 轻量同步**，显著提升了 MoE 模型在低成本硬件上的推理效率，推动了大模型平民化部署。

</details>

---

### 6. [Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for Speculative Decoding](https://arxiv.org/abs/2609.36173)

**Authors**: Haohui Zhang, Keyu Chen, Haocheng Sun, Weibo Gu, Ruizhi Qiao, Xing Sun, Bo Jiang  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.36173v1  

#### Abstract
Parallel speculative drafting generates multiple candidates in one backbone pass, but independent token selection can produce inconsistent continuations that shorten the accepted prefix. Existing methods mostly leave conditional decoding to a lightweight module after the backbone, which limits the f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for Speculative Decoding*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Speculative Decoding** 中，**Parallel Drafting** 方法（如 DFlash）通过一次前向传播并行生成多个候选 token，显著降低了 drafting 的延迟。然而，由于所有位置独立预测，缺乏对已生成前驱 token 的依赖建模，导致生成的候选序列内部不一致，从而被 target model 验证时提前截断，**acceptance length 较短**，限制了加速效果。

现有方法（如 Domino、DSpark、DFlash2）通常将条件依赖建模放在 backbone 之后的轻量模块中，无法让前驱信息在深层网络中逐步影响后续 token 的生成。

---

### 🚀 提出的新方法：**DSpine**

DSpine 提出了一种全新的 **“深度内因果注入”** 架构，其核心思想是：

> **在 backbone 的每一层，显式地将每个位置的前驱 token 的预测特征注入到其后继位置中，实现“依赖链随网络深度展开”的并行条件建模。**

#### 主要创新点：

1. **Adjacent Causal Injection（相邻因果注入）**
   - 在每个 Transformer 层后，通过一个 **gated injection 模块**，将 `position t-1` 的预测特征写入 `position t` 的残差流中。
   - 注入从第一层开始，使得前驱信息尽早参与后继表示的构建。
   - 所有位置仍保持并行更新，不引入额外串行开销。

2. **Unified Transfer Space（统一传递空间）**
   - 基于 target model 的 output embeddings 构建一个共享的低维空间（经 PCA 白化处理），用于统一表示：
     - 骨干网络中未确定的“预测特征”
     - 已确定的“真实 token embedding”
   - 实现了 **injection 与 decoding 路径的一致性**。

3. **Layer-wise Output-Embedding Supervision**
   - 对每一层的预测特征施加 **cosine alignment loss**，使其对齐目标 token 的 embedding。
   - 促进浅层即可形成高质量的预测特征，支持早期注入的有效性。

4. **Transition Cache 加速解码**
   - 预计算所有相邻候选对的条件得分，将原本串行的 conditional scoring 转为并行预计算 + 串行选择，提升推理效率。

---

### 🔍 相比现有方法的优势

| 方面 | DSpine 优势 |
|------|-------------|
| **依赖建模粒度** | 在骨干网络**每一层**注入前驱信息，而非仅在最后 |
| **信息传递形式** | 传递的是**预测特征**（可组合、可演进），而非最终 token |
| **时间性** | 前驱信息**尽早进入**网络，影响更深层表示形成 |
| **训练一致性** | 统一空间使训练时的“预测注入”与推理时的“token 注入”路径一致 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Math**: GSM8K, MATH-500
- **Code**: HumanEval, MBPP, LiveCodeBench (LCB)
- **Chat**: MT-Bench, Arena-Hard  
共 **7 个 benchmark**，覆盖主流任务类型。

### ⚙️ 实验设置
- **模型基础**: Qwen3-4B 和 Qwen3-8B
- **draft block size**: 16（1 anchor + 15 candidates）
- **drafter 结构**: 5 层轻量 Transformer，共享 target 的 input embedding 和 LM head
- **温度设置**: 温度 0（greedy）和 温度 1（sampling）
- **评估平台**: SGLang，支持 CUDA graph 和 fused kernels

### 📊 评估指标
1. **Average Acceptance Length (t)**  
   每轮 spec decode 成功接受的 token 数量（越高越好）
2. **Throughput (tokens/s)**  
   实际服务吞吐量，考虑 prefill、调度等端到端开销
3. **Per-round Latency Breakdown**  
   分析 drafting 和 verification 各阶段耗时

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **DFlash** | 并行无依赖 | 各位置独立预测 |
| **Domino** | 后处理依赖 | 引入 causal correction branch |
| **DSpark** | 半自回归 | sequential head 建模依赖 |
| **DFlash2** | 路径选择 | top-k 路径打分与选择 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Qwen3-8B, temp=0）

| 方法 | 平均 Acceptance Length (↑) | 相比 DFlash 提升 |
|------|----------------------------|----------------|
| DFlash | 3.77 | — |
| DSpine (**ours**) | **4.82** | **+27.8%** |

> 在 **7 个 benchmark 上全面领先**，平均 acceptance length 提升显著。

#### 各基准表现（vs 最强 baseline）：
- GSM8K: +12.8%
- MATH-500: +15.2%
- HumanEval: +7.7%
- MBPP: +12.7%
- MT-Bench: +11.5%
- Arena-Hard: +13.6%

---

### 🔄 与最强基线对比（Qwen3-8B）

| 方法 | vs DFlash2 (Acceptance) | vs DFlash2 (Throughput) |
|------|--------------------------|--------------------------|
| DSpine | **+10.0% ~ +14.1%** | **+11.1% 平均吞吐提升** |

> 在 Qwen3-4B 上也取得 **+5.4% ~ +10.1%** 的领先。

---

### ⚙️ 消融实验结果

#### 表：消融组件的影响（MATH-500, MBPP, MT-Bench）

| 变体 | Acceptance Length 下降 |
|------|------------------------|
| 完整 DSpine | 5.49 / 4.98 / 4.11 |
| 移除 last-write refinement | ↓ 6.6–7.1% |
| 进一步移除 parallel injection | ↓ 8.8–10.7% |

✅ **证明两个核心机制互补且必要**。

#### 表：监督方式对比

| 监督方式 | 效果 |
|--------|------|
| 无监督（DFlash） | 3.75 (MATH-500) |
| Per-layer Token CE | 3.67 ↓ |
| **Embedding Cosine Alignment** | **3.85 ↑** |

✅ **中间层直接对齐 output embedding 更有效**，避免干扰上下文建模。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **浅层已有可恢复的前缀预测信息**  
   → 支持在浅层就开始注入前驱信息。

2. **前驱信息越早进入，对后继预测帮助越大**  
   → 注入时机比内容本身更重要。

3. **相邻前驱提供不可替代的预测线索**  
   → 即使历史正确，错误的 immediate predecessor 也会大幅降低 successor 准确率（↓31.8pp）。

4. **DSpine 实现了“深度内依赖建模”与“并行高效 drafting”的统一**  
   → 在单次并行前向中完成依赖链的逐层构建。

---

### ⚠️ 方法的局限性

1. **依赖固定 block size**  
   当前设计基于固定长度 block（如 16），难以动态扩展。

2. **仅建模相邻依赖**  
   当前 injection 仅限 `t-1 → t`，未考虑更长距离依赖（如 t-2, t-3）。

3. **transition cache 内存开销**  
   预计算 $K^2$ 转移分数带来额外内存占用（但 $K=16$ 时可接受）。

4. **需重新训练 drafter**  
   不像某些方法可直接部署，DSpine 需专门训练以配合 injection 和 supervision。

---

### 🔮 未来工作方向

1. **Flexible Communication Range**  
   探索非相邻位置的信息注入（如跳跃连接、attention-based routing）。

2. **Dynamic Block Adaptation**  
   支持变长 drafting block，适应不同任务需求。

3. **通用 In-layer Transfer Framework**  
   将 unified transfer space 思想推广至其他并行 decoding 架构。

4. **Hardware-aware Optimization**  
   进一步优化 fused kernel 和 memory layout，最大化 GPU 利用率。

---

## ✅ 总结

**DSpine** 通过 **“深度内相邻因果注入”** 和 **“统一传递空间”** 的设计，在保持并行 drafting 高效性的同时，首次实现了 **前驱信息在骨干网络中的逐层传播与利用**。实验证明其在 **acceptance length** 和 **SGLang serving throughput** 上均显著优于当前最先进的 parallel drafters，为 speculative decoding 提供了一个新的范式：**Draft in Parallel, Condition Through Depth**。

</details>

---

### 7. [AutoLoCo: Communication Efficient Distributed LLM Training via Adaptive Synchronization](https://arxiv.org/abs/2609.36662)

**Authors**: Pengyu He, Yan Zhang, Ruien Li, Guangwen Yang  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.36662v1  

#### Abstract
The pre-training of Large Language Models (LLMs) is increasingly conducted across multiple data centers. As training scales to a larger number of accelerators, the fraction of time spent on computation decreases, while the fraction spent on communication increases. Therefore, frequent synchronizatio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AutoLoCo: Communication Efficient Distributed LLM Training via Adaptive Synchronization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在大规模语言模型（LLM）的预训练中，随着分布式计算节点数量的增加，**通信开销逐渐成为训练瓶颈**。传统的同步方法（如 DDP）需要频繁同步梯度，而现有的局部更新方法（如 DiLoCo）虽然减少了通信频率，但其**通信间隔是固定的**，无法适应训练过程中动态变化的优化状态。

固定间隔存在以下问题：
- 早期阶段：模型参数变化剧烈，短间隔更安全，长间隔可能导致发散。
- 后期阶段：学习率衰减，参数更新变小，可以容忍更长的本地训练周期以减少通信。

因此，如何**动态调整通信间隔并保持训练稳定性与收敛性**，是一个关键挑战。

---

### **提出的新方法与新思路**
本文提出了 **AutoLoCo** —— 一种**自适应同步框架**，通过两个核心机制实现通信效率与训练性能的平衡：

#### ✅ **(1) 自适应通信间隔选择（Adaptive Interval Selection）**
- 基于当前训练状态（如 worker drift 和 aggregation coherence）在线决定下一个通信间隔。
- 使用 **token-aligned horizon** 控制本地计算量，避免因 batch packing 或 padding 导致的步数不一致。
- 映射为实际执行的 `local steps`，提升调度灵活性。

#### ✅ **(2) 外层优化器校正（Outer Optimizer Correction）**
当通信间隔改变时，每个 outer update 所代表的“本地优化量”也随之变化，导致：
- 伪梯度（pseudo-gradient）尺度变化
- 动量累积的时间单位失配

为此，AutoLoCo 引入了基于 **accumulated inner learning rate** 的校正机制：
- **伪梯度归一化**：将平均后的伪梯度按比例缩放至基准时间单位。
- **动量保留调整**：根据区间长度指数级调整动量系数 $ \mu_t = \mu_{\text{base}}^{p_t} $
- **外层学习率缩放**：控制更新步长，防止过大跳跃。

> 🔑 核心思想：**将通信调度与外层优化视为耦合设计问题**，而非独立模块。

---

### **相比现有方法的优势**
| 方法 | 是否自适应 | 是否校正外层优化 | 通信节省 | 性能保持 |
|------|------------|------------------|----------|-----------|
| DDP | ❌ | ❌ | 基准 | 最优 |
| DiLoCo | ❌（固定间隔） | ❌ | 有限 | 下降明显 |
| QSR / Linear Schedule | ✅（预定义规则） | ❌ | 中等 | 下降 |
| **AutoLoCo** | ✅（反馈驱动） | ✅（完整校正） | **↑27%** | **优于 DiLoCo** |

> AutoLoCo 是首个将**动态间隔选择**与**外层优化语义对齐**结合的方法，在降低通信的同时**反而提升了最终性能**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **预训练任务**：`C4` 数据集（英文文本）
- **微调任务**：`UltraChat-200k`（高质量指令对话数据）

---

### **实验设置**

| 设置项 | 预训练（C4） | 微调（UltraChat） |
|--------|---------------|--------------------|
| 模型 | LLaMA-style (215M) | Llama-3.1-8B |
| 全局 batch size | 1,024（8 workers × 128） | 32 |
| 序列长度 | 1,024 | ≤2,048 |
| 内部优化器 | AdamW | AdamW |
| 学习率调度 | Cosine | Cosine |
| 精度 | BF16 | BF16 |
| 总训练步数 | 50,000 steps | 1 epoch (~6,490 steps) |

> 所有方法共享相同的初始化、数据顺序和训练预算，确保公平比较。

---

### **评估指标**
- **训练质量**：
  - Training loss（最后 1,000 步均值）
  - Validation loss（最终检查点）
- **下游能力**（微调后）：
  - MMLU-Pro、BBH、HumanEval+、IFEval 等 benchmark
- **通信成本**：
  - 同步次数（#Syncs）
  - 逻辑通信量（Logical payload, GB）
  - 实际带宽占用（Wall time）

---

### **基线方法对比**
- **DDP**：每步同步，通信密集，性能最优。
- **DiLoCo**：固定间隔（H=500），代表低频通信 baseline。
- **Linear Interval**：线性增长间隔（10→990），共 100 次同步。
- **QSR (Quadratic Synchronization Rule)**：基于学习率平方反比设定间隔。
- **AutoLoCo**：本文方法，动态选择 + 外层校正。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（预训练，C4）**

| Method | Training Loss | #Syncs | Validation Loss |
|--------|----------------|--------|------------------|
| DDP | 2.7227 | 50,000 | 2.7233 |
| DiLoCo | 2.8004 | 100 | 2.8020 |
| Linear Interval | 2.8706 | 100 | 2.8694 |
| QSR | 2.8914 | 41 | 2.8793 |
| **AutoLoCo** | **2.7843** | **73** | **2.7842** |

> ✅ AutoLoCo 在仅需 **73 次同步**的情况下，**训练损失低于 DiLoCo**，且通信量减少 **27%**（100 → 73）。

---

### **微调结果（Llama-3.1-8B on UltraChat）**

| Method | Test NLL | #Syncs | Logical Payload (Total) |
|--------|----------|--------|--------------------------|
| DiLoCo | 0.9112 | 65 | 1,043.934 GB |
| **AutoLoCo** | **0.8766** | **49** | **786.966 GB** |

> ✅ 通信减少 **24.6%**，同时 **Test NLL 显著下降**，说明训练质量更高。

#### **下游任务表现（8 workers）**
| Benchmark | DiLoCo | AutoLoCo | Δ |
|-----------|--------|---------|----|
| MMLU-Pro | 31.21 | **36.25** | ↑5.04 |
| BBH CoT EM | 56.03 | **61.10** | ↑5.07 |
| HumanEval+ | 15.24 | **22.56** | ↑7.32 |

> AutoLoCo 接近 DDP 表现，显著优于 DiLoCo，尤其在代码生成任务上提升巨大。

---

### **消融实验结果（Ablation Study）**

| Variant | Training Loss | #Syncs | Notes |
|--------|----------------|--------|-------|
| DiLoCo | 2.8251 | 100 | 固定间隔 |
| Adaptive Intervals | 2.8310 | 73 | 仅改间隔，无校正 |
| + Momentum Adjustment | 2.8484 | 73 | 动量未缩放，性能恶化 |
| **AutoLoCo (Full)** | **2.8116** | 73 | 完整校正，性能最佳 |

> 🔍 发现：
> - 单独使用自适应间隔会轻微损害性能（因未校正外层优化）。
> - 必须**同时调整动量保留和学习率缩放**才能恢复甚至超越原性能。
> - 证明了“**外层优化器必须感知本地优化量**”的设计必要性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **通信间隔应随训练进程动态调整**：
   - 早期需保守（短间隔），后期可激进（长间隔）。
   - 基于 `local-drift energy` 和 `aggregation coherence` 的反馈机制有效。

2. ✅ **外层优化器必须适配不同长度的本地轨迹**：
   - 不同长度的 local horizon 导致 pseudo-gradient 尺度和动量持续时间失配。
   - 使用 **accumulated inner learning rate** 作为“优化工作量”的度量，进行归一化和校正是关键。

3. ✅ **AutoLoCo 实现双赢**：
   - 通信减少 **27%（预训练） / 24.6%（微调）**
   - **训练损失更低、验证性能更强**
   - 是目前唯一能在减少通信的同时**提升性能**的方法。

---

### **局限性**
- 当前控制器依赖手工设计的阈值（如 `z_max` 判断标准），未来可探索强化学习策略。
- 主要在 BF16 和 INT8 场景验证，极端低精度（如 INT4）下的稳定性待研究。
- 控制逻辑引入少量额外开销（约 0.3% 训练时间），虽可忽略但在超大规模场景仍需关注。

---

### **未来工作方向**
1. **扩展到更长通信间隔**：探索分钟级甚至小时级同步的可能性。
2. **异构网络环境适配**：结合链路延迟预测动态调整策略。
3. **与压缩技术联合优化**：如与 QSGD、PowerSGD、Error Feedback 结合，进一步降低总通信成本。
4. **自动化控制器设计**：用轻量级 ML 模型替代规则引擎，实现端到端优化。

---

> 📌 **总结一句话**：  
> **AutoLoCo 通过“自适应同步 + 外层优化校正”，首次实现了在大幅降低通信频率的同时，反而提升 LLM 训练质量的目标，为跨数据中心高效训练提供了新范式。**

</details>

---

### 8. [$S^3$: Spectral Null-Space Swap Makes Reasoning Models Efficient](https://arxiv.org/abs/2609.37976)

**Authors**: Hongbo Ma, Sansheng Cao, Jiajun Fan, Bangji Yang, Ge Liu  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.37976v1  

#### Abstract
LLMs trained with Chain-of-thought excel in reasoning capability, but often come with excessive token cost. We find that the core of reasoning capacity lies in the Thinking model's weight component within the null space of a projection defined by the corresponding Non-thinking model's dominant singu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《$S^3$: Spectral Null-Space Swap Makes Reasoning Models Efficient》核心总结

---

## 1. 主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在经过 **Chain-of-thought (CoT)** 后训练后推理能力显著提升，但其生成过程往往产生过长的推理链，导致**推理时的 token 开销巨大**，影响效率。如何在不牺牲推理准确率的前提下，显著降低推理成本，是当前高效推理研究的核心挑战。

现有方法大多关注于在模型的主导子空间（dominant subspace）内进行优化（如权重插值、任务算术等），而忽略了模型差异中功能上更重要的部分。

### 提出的新方法与新思路
本文提出了一种全新的训练无关（training-free）模型组合方法——**Spectral Null-Space Swap ($S^3$)**，其核心思想基于一个关键发现：

> **推理能力的关键并非存在于 Non-thinking 模型已占据的主导奇异方向（dominant singular directions）中，而是集中在其正交的零空间（null space）中。**

具体而言：
- 将 Thinking 模型与 Non-thinking 模型的权重差 $\Delta W = W_t - W_o$ 分解为两个分量：
  - **对齐分量（Aligned Component）**：位于 $W_o$ 的主导奇异子空间 $S_\rho$ 内。
  - **互补分量（Complementary/Null-Space Component）**：位于该子空间的正交补空间（即零空间）中。
- 实验发现，尽管对齐分量在参数空间（parameter space）中能量更高，但它引起的**函数空间变化（functional impact）却很弱**；相反，零空间分量虽然参数能量较低，却是驱动模型功能变化（即增强推理能力）的主要来源。

因此，$S^3$ 方法通过以下方式构建新模型：
$$
W_\rho = \mathcal{P}_\rho(W_o) + (\mathbf{I} - \mathcal{P}_\rho)(W_t)
$$
其中 $\mathcal{P}_\rho$ 是投影到 $W_o$ 前 $\rho$ 个主导奇异向量构成的子空间的算子。该操作保留了 Non-thinking 模型在保护子空间内的结构，并从 Thinking 模型中引入了其零空间分量。

### 相比现有方法的优势
- **无需额外训练**：完全基于已有 checkpoint 进行权重重组，无任何微调或强化学习开销。
- **精准定位有效成分**：首次揭示并利用了“零空间”在推理能力中的关键作用，而非在主导子空间内进行低效调整。
- **帕累托最优**：在准确率-效率权衡上，$S^3$ 在多个基准上达到了新的经验帕累托前沿（empirical Pareto Frontier），实现了**精度提升的同时大幅降低 token 成本**。
- **通用性强**：适用于 Dense 和 Mixture-of-Experts (MoE) 架构，以及文本、视觉语言、音频等多种模态。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了 28 个评估环境，涵盖三大类推理任务：
- **数学推理**：AIME24/25, HMMT25, CMIMC25, Olympiad-Bench, AMC23, MATH-500, GSM8K
- **视觉语言推理**：MathVista-testmini, MMMU, MMLU
- **音频推理**：MMAR, MMSU

### 实验设置和评估指标
- **模型架构**：在 2B-30B 规模的 Dense 和 MoE 模型上验证，包括：
  - `Qwen3-4B`, `Qwen3-30B-A3B` (Dense & MoE)
  - `Qwen3-VL-2B`, `Qwen3-VL-4B` (Vision-Language)
  - `Qwen3-Omni-30B-A3B` (Omni: Text + Audio)
- **评估指标**：
  - **准确率（Accuracy）**：Pass@1, Avg@4
  - **效率（Efficiency）**：平均每题生成的 token 数（Generated Tokens）
  - **综合表现**：准确率 vs. Token 开销的帕累托前沿分析
- **生成配置**：统一使用 `top_p=0.95`, `temperature=0.7` 等设置，确保公平比较。

### 基线方法对比
- **Non-thinking (Instruct)**：基础指令模型，token 成本低但推理能力弱。
- **Thinking**：全量推理模型，能力强但 token 成本高。
- **MI-0.8**：直接的 Instruct-Thinking 权重插值（0.8 比例）。
- **TIES-Merging**：一种先进的模型合并方法，通过裁剪和符号选举解决冲突。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **平均 token 减少 27.4%**：相比全量 Thinking 模型，$S^3$ 平均减少了 27.4% 的推理 token 开销。
- **平均准确率提升 1.0 个百分点**：在降低开销的同时，整体任务准确率反而提升了 1.0%。
- **多项任务实现突破性表现**：
  - **HMMT25**：准确率 **+8.3%**，同时 token 减少 **33.0%**。
  - **MathVista-testmini**：准确率 **78.05%**，token 减少 **31.9%**。
  - **AIME25**：在 `Qwen3-4B` 上达到 73.3% 准确率，仅用 15,004 tokens，优于 Thinking 模型（20,704 tokens）。

### 与基线方法的对比结果
| 模型 | 相比 Thinking 的 Token 变化 | 相比 Thinking 的 Acc 变化 |
|------|----------------------------|--------------------------|
| **$S^3$** | **平均 -27.4%** | **平均 +1.0 pp** |
| MI-0.8 | ~ -20% | ~ ±0 |
| TIES | ~ -20% | ~ ±0 |

在多个任务上，$S^3$ 同时优于 MI-0.8 和 TIES，实现了更低的 token 成本和更高的准确率，确立了新的帕累托最优边界。

### 消融实验结果
#### (1) 保护子空间比例 $\rho$ 的影响
- $\rho$ 控制保留 Non-thinking 模型主导方向的比例。
- **$\rho=0.8$ 被选为默认值**：在准确率和效率之间提供了最佳平衡。
- 更小的 $\rho$ 引入更多 Thinking 模型的零空间分量，可能提升准确率但增加 token 长度。
- 更大的 $\rho$ 更接近 Non-thinking 模型，token 最短但推理能力下降。

#### (2) 子空间与零空间分量的独立作用（Probe Family）
通过构造四个变体模型进行消融：
- **Base** ($W_o$)
- **Sub** ($W_o + \Delta W_{\parallel}$)
- **Null** ($W_o + \Delta W_{\perp}$)
- **Full** ($W_t$)

**关键发现**：
- **Sub 模型**：性能接近 Base，说明对齐分量几乎不贡献推理能力。
- **Null 模型**：性能接近 Full，说明**零空间分量是推理能力的核心来源**。
- **Null 模型生成更短的推理链**，证明其推理更高效。

---

## 4. 关键结论和发现

### 主要发现
1. **功能不对称性**：Thinking 模型与 Non-thinking 模型的差异中，参数能量高的对齐分量功能影响弱，而能量低的零空间分量才是功能变化的驱动力。
2. **零空间是效率关键**：通过 $S^3$ 选择性地引入零空间分量，可以在保持甚至提升准确率的同时，显著减少推理 token。
3. **注意力熵解释**：$S^3$ 模型的注意力熵（attention entropy）更低，表明其注意力更集中，认知负荷更低，从而推理更高效。理论分析表明，零空间扰动倾向于降低注意力熵，而子空间扰动则会增加它。

### 方法的局限性
- **依赖成对模型**：需要同一预训练模型下配对的 Instruct 和 Thinking checkpoint。
- **超参数 $\rho$ 需调优**：不同任务可能需要不同的 $\rho$ 值以达到最佳效果。
- **理论假设简化**：关于注意力熵局部最优性的分析是一个理想化假设，实际模型可能更复杂。

### 未来工作方向
- 探索其他类型的子空间分解（如任务特定子空间）用于更细粒度的能力迁移。
- 将 $S^3$ 思想应用于模型压缩、知识蒸馏等场景。
- 研究如何自动确定最优的保护比例 $\rho$。
- 扩展至更多模态和更复杂的多步推理任务。

--- 

> **总结**：$S^3$ 通过揭示并利用权重空间中被忽视的“零空间”，提供了一种简单、高效且训练无关的方法，实现了推理模型在准确率和效率上的双重提升，为高效推理研究开辟了新的几何视角。

</details>

---

### 9. [LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization](https://arxiv.org/abs/2609.38166)

**Authors**: Yi Pan, Haocheng Xi, Kan Zhu, Xingyang Li, Yibo Wu, Mayank Mishra, Hongtao Zhang, William X. Zheng, Baris Kasikci, Song Han, Kurt Keutzer, Rishabh Iyer, Ion Stoica  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.38166v1  

#### Abstract
Recent LLMs increasingly adopt hybrid designs that replace standard attention with linear attention, such as Gated DeltaNet (GDN) and Kimi Delta Attention (KDA). Although they compress the context into a fixed-size recurrent state and substantially reduce the cost of long-context processing, repeate...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代大型语言模型（LLMs）越来越多地采用**混合架构**，用**线性注意力（linear attention）** 替代标准注意力机制（如 Gated DeltaNet, GDN 和 Kimi Delta Attention, KDA）。这类方法通过将上下文压缩为一个固定大小的**循环状态（recurrent state）** 来降低长上下文处理的计算开销。

然而，在推理过程中，该循环状态需要在每一步生成 token 时从 HBM 中读取、更新并写回，导致：
- **内存带宽瓶颈**：频繁的状态传输成为解码吞吐量的主要限制。
- **内存占用高**：尤其在启用前缀缓存（prefix caching）时，多个请求的独立状态显著增加 GPU 内存消耗。

直接对循环状态进行低比特量化（如 INT8）虽可减少内存流量和占用，但会因以下两个原因严重损害模型精度：
1. **误差累积（Error Accumulation）**：每次更新都基于已量化的状态，舍入误差随时间逐步积累。
2. **离群值问题（Outlier Rows/Columns）**：状态中少数极大值扩大了量化范围，迫使其他小值落入粗糙的量化级别。

---

### **提出了什么新方法或新思路**
本文提出 **LeapQuant**，一种无需训练的（training-free）、近无损的循环状态量化方法，核心思想是：

#### ✅ **Per-Window Quantization（窗口级量化）**
- 不再每个 token 后重新量化整个状态，而是**每隔一个窗口（window）才量化一次**。
- 在窗口内，保持低精度的边界状态不变，**缓冲高精度的 token 更新**。
- 只有在窗口结束时，才重构完整状态并进行一次量化。
- **效果**：大幅减缓量化误差的累积速度。

#### ✅ **Compensator Tokens（补偿令牌）**
- 将状态中的最大离群值提取出来，表示为若干个**高精度的“补偿令牌”**（rank-one 外积形式）。
- 这些补偿令牌与真实 token 共享相同的更新路径，不需额外维护状态。
- 剩余残差被平滑后更容易量化。
- **效果**：显著降低单次量化的误差。

#### ✅ **Residual Smoothing（残差平滑）**
- 在量化前对残差矩阵按行进行缩放（channel-wise scaling），平衡各 key row 的幅值分布。
- 缓解因个别通道过大而导致的整体量化失真。

---

### **相比现有方法的优势**
| 维度 | LeapQuant | 现有方法（如 INT8/BF16 per-step） |
|------|-----------|-------------------------------|
| **精度保持** | 接近 FP32 基线，几乎无损 | 显著下降，尤其在长序列上崩溃 |
| **量化频率** | 每 `p` 个 token 一次（如 p=16） | 每个 token 都重量化 |
| **离群值处理** | 显式建模为 Compensator Tokens | 忽略或简单分组处理 |
| **是否需要训练/校准** | ❌ 完全无需训练或数据校准 | 多数需要校准数据（如 SmoothQuant） |
| **适用性** | 支持多种线性注意力变体（GDN/KDA/GLA 等） | 往往针对特定结构设计 |

---

## **2. 核心实验方法和设置**

### **使用的模型**
在五个主流混合架构 LLM 上验证：
- **Qwen 系列**：Qwen3.5-9B, Qwen3.5-35B-A3B, Qwen3.8-Flash（均使用 GDN）
- **Kimi 系列**：Kimi-Linear-48B-A3B-Instruct（使用 KDA）
- **GLM 系列**：GLM-5.3-Flash（使用 KDA）

部分模型仅运行 8 层以测量效率（保留解码成本特性）。

---

### **数据集**
用于下游任务评估：
- **AIME 2026**：数学推理
- **GPQA-Diamond**：研究生级别问答
- **MMLU-Pro**：多任务理解
- **LiveCodeBench v6**：代码生成
- **GSM8K**：小学数学题

所有结果取三次随机种子平均。

---

### **评估指标**
| 类别 | 指标 |
|------|------|
| **准确性** | 下游任务准确率（Accuracy %） |
| **效率** | - 单层 kernel 吞吐加速比<br>- 端到端 decode 步骤吞吐加速比<br>- 内存流量减少倍数<br>- 端到端内存占用降低比例 |
| **消融研究** | 分析 window size (`p`) 和 Compensator Token 数量 (`r`) 的影响 |

---

### **基线方法对比**
#### **基础格式**
- FP32（浮点32位）
- BF16（脑浮点16位）

#### **8-bit 方法**
- FP8 / INT8（逐张量/逐通道/逐组）
- KVQuant（KV 缓存离群分离）
- QuaRot（旋转去相关）
- TurboQuant（在线向量量化）

#### **6-bit 与 4-bit 方法**
- NVFP6/NVFP4, MXFP6/MXFP4, INT6/INT4
- 对应低比特版本的上述适配方法

> 注：由于缺乏专门针对循环状态的量化方法，作者将原本用于 KV Cache 或激活量化的技术迁移到本场景作为 baseline。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 指标 | 结果 |
|------|------|
| **精度表现（8-bit）** | LeapQuant 在 12 个 model-task 对上达到与 FP32 相当的准确率 |
| **内存流量减少** | 状态内存流量减少 **3.4×** |
| **端到端内存占用降低** | 最高达 **56%**（Kimi-48B） |
| **kernel 级加速** | 平均 **2.05–3.70×**（B200, RTX PRO 6000, RTX 5090） |
| **端到端推理加速** | 平均 **1.47×** |

---

### **与基线方法的对比结果**

#### 🔹 **8-bit 对比**
- **LeapQuant vs BF16**：
  - Qwen3.5-9B 在 AIME 上：BF16 准确率从 87.9% ↓ 至 72.1%，而 LeapQuant 保持 **87.9%**
- **LeapQuant vs INT8/FP8**：
  - INT8 导致 AIME 准确率暴跌至 **7.1%**
  - FP8 仅为 **14.6%**
  - LeapQuant 达到 **87.9%**

> 表明传统 per-step 量化在长序列下完全失效。

#### 🔹 **6-bit 与 4-bit 对比**
| 方法 | 6-bit 平均准确率 | 4-bit 平均准确率 |
|------|------------------|------------------|
| NVFP6 / MXFP6 | ~29.6% | — |
| TurboQuant | ~31.9% | ~22.9% |
| MXFP4 | — | **9.7%** |
| **LeapQuant** | **72.4%** | **60.4%** |

> 即使在极端 4-bit 场景下，LeapQuant 仍能维持远超 baselines 的性能，且在 Kimi 模型上所有任务差距 <1.1%。

---

### **消融实验结果**

#### ✅ **组件有效性分析（Table 2）**
在 Qwen3.5-9B 上逐步添加模块：

| 方法 | AIME (%) | LiveCodeBench (%) | Kernel Speedup |
|------|----------|--------------------|----------------|
| INT8 per-step | 7.1 | 9.2 | — |
| + Per-window | 82.4 | 60.6 | ↑ |
| + Compensator Tokens | 86.6 | 61.5 | ↑ |
| + Residual Smoothing | **87.9** | **64.1** | **2.52×** |

> 所有三个组件均有显著增益，最终实现 FP32 精度 + 超 2.5× 加速。

#### ✅ **窗口长度 ablation（Table 3 左）**
- `p=16` 是最优选择：
  - 更短 → 量化太频，误差大
  - 更长 → buffer 过大，共享内存不足，性能下降
- `p=16` 时 kernel speedup 达 **2.52×**

#### ✅ **Compensator Token 数量 ablation（Table 3 右）**
- `r=4` 即饱和：
  - 准确率不再提升
  - `r≥8` 开始出现性能倒退（power iteration 开销暴露）
- 使用 `r=4` FP16 tokens 成本极低且完全隐藏于内存读取中

---

## **4. 关键结论和发现**

### **主要发现**
1. **循环状态是线性注意力推理的关键瓶颈**，其频繁访问导致严重的 HBM 带宽压力和内存占用。
2. **简单的低比特量化会导致灾难性精度损失**，主因是误差累积和离群值干扰。
3. **LeapQuant 通过 per-window 量化 + compensator tokens + smoothing 三管齐下，实现了近乎无损的 8-bit 量化**。
4. **该方法完全无需训练或校准数据**，易于部署，兼容性强。
5. **在真实硬件（B200, RTX PRO 6000, RTX 5090）上验证有效**，带来显著的端到端加速和内存节省。

---

### **方法的局限性**
1. **依赖窗口机制**：虽然 `p=16` 效果良好，但在极短序列或流式输出场景中可能收益较小。
2. **Compensator Tokens 引入额外计算**：尽管被隐藏，若 `r` 设置过大（如 >8）会影响性能。
3. **目前仅适用于支持 delta-rule 更新的线性注意力家族**（如 GDN/KDA），不直接推广至所有 SSM 架构（如 Mamba）。
4. **未探索训练时联合优化的可能性**：当前为纯推理期方法，未来可通过训练进一步压缩。

---

### **未来工作方向**
1. **扩展至更多 SSM 架构**：适配 Mamba、RWKV 等具有不同状态更新规则的模型。
2. **动态调整 window size 和 `r`**：根据输入复杂度自适应配置资源。
3. **结合权重/激活量化**：构建完整的全模型低比特推理方案。
4. **探索更高效的 compensator token 拟合算法**：替代 power iteration，降低延迟。
5. **应用于训练阶段的状态缓存压缩**：进一步降低训练成本。

---

> 💡 **总结一句话**：  
> **LeapQuant 是首个实现近无损、无需训练的循环状态量化方案，通过“跳过式窗口量化”与“补偿令牌”机制，破解了线性注意力推理中的精度-效率困境，在真实硬件上实现最高 3.7× kernel 加速和 56% 内存压缩，有望成为下一代混合 LLM 推理的标准组件。**

</details>

---

### 10. [DEdit: Iterative Draft Editing for Speculative Decoding](https://arxiv.org/abs/2609.38510)

**Authors**: Longxuan Yu, Bingsen Chen, Peng Shi, Dongkyu Lee, Yi Xiang, Hideo Kobayashi, Sheng Zhang, Shuaichen Chang, Xing Niu, Zhuoyan Xu, Greg Ver Steeg, Jiarong Jiang  
**Category**: cs.CL  
**Published**: 2026-10-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.38510v1  

#### Abstract
Speculative decoding accelerates autoregressive LLMs by having a lightweight drafter propose tokens that the target model verifies in parallel. Diffusion-based drafters further reduce drafting latency by proposing multiple tokens at once. However, these tokens are predicted independently, so a singl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DEdit: Iterative Draft Editing for Speculative Decoding

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **Speculative Decoding** 中，轻量级的 **drafter** 模型并行生成多个候选 token，由目标模型（target model）批量验证以加速推理。然而，现有的 **diffusion-based drafters** 虽然能并行预测多个 token，但由于各位置独立预测，早期出现错误会导致后续所有候选被前缀验证（prefix verification）机制丢弃，即使后面的预测可能是正确的。

这一问题导致：
- 接受率（token acceptance）随位置衰减；
- 浪费了大量潜在有用的“未来”预测。

### 提出的新方法
论文提出了 **DEdit**，一种基于扩散模型的 **迭代草案编辑器（iterative draft editor）**，其核心创新在于：

- **双向编辑（Bidirectional Editing）**：DEdit 不仅能像传统方法一样生成初始草案（initial proposal），还能通过多轮迭代，利用 **bidirectional attention** 对整个草案进行并行编辑。这使得后期预测可以作为上下文，用于修复早期错误。
- **迭代编辑机制**：在验证前，DEdit 执行 $K$ 轮编辑（第一轮为生成，后 $K-1$ 轮为编辑），每轮都基于上一轮的完整草案进行更新，最终只将最后一轮的草案提交给目标模型验证。
- **PROPOSALMIX 训练策略**：提出了一种新颖的训练算法，通过混合首次预测与真实标签（ground-truth）来构造训练输入，使模型学会在保留正确预测的同时修正错误。

### 相比现有方法的优势
- **更高的 token 接受率**：通过编辑修复早期错误，显著延长了可接受的前缀长度。
- **更好的扩展性**：随着草案窗口（proposal window）增大，DEdit 的收益远超基线方法，因为更多未来 token 可作为编辑上下文。
- **避免有害编辑**：PROPOSALMIX 显著减少了会缩短接受前缀的“有害编辑”。

---

## 2. 核心实验方法和设置

### 数据集
在 **七个基准任务** 上进行评估，涵盖数学、代码和对话领域：
- **MATH**: GSM8K, MATH-500, AIME25
- **CoDE**: HumanEval, MBPP, LiveCodeBench (LCB)
- **CHAT**: MT-Bench

训练数据使用约 80万条与目标对齐的语料（Nemotron 和 CodeAlpaca prompts），消融实验使用单独的 10万条语料。

### 实验设置和评估指标
- **目标模型**：Qwen3-4B 和 Qwen3-8B
- **草案窗口宽度（W）**：主实验中 DEdit 使用 W=32，基线使用 W=16
- **编辑轮数（K）**：DEdit 默认使用 3 轮（P3，即 1 次生成 + 2 次编辑）
- **评估协议**：
  - **Token Acceptance (T)**：每轮平均接受的 token 数（含额外生成的一个 target token）。
  - **End-to-end Speedup**：相对于自回归生成的整体加速比。
  - 解码方式：greedy ($T_p=0$) 和 stochastic ($T_p=1$)
- **执行环境**：单张 NVIDIA H100 GPU，batch size=1，drafting 使用 CUDA Graph，target verification 使用 eager mode。

### 基线方法对比
- **DFlash**：单次并行扩散草案生成（baseline 架构相同）
- **Domino**：引入轻量级因果校正（causal correction）
- **DSpark**：引入顺序模块和置信度调度验证

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **greedy decoding** 下，DEdit 实现了当前最高的加速比：

| Model        | Method   | Macro-Avg T | Speedup |
|--------------|----------|-------------|---------|
| Qwen3-4B     | DEdit P3 | **7.58**    | **5.72×** |
| Qwen3-8B     | DEdit P3 | **7.75**    | **5.97×** |

相比最强基线 DSpark，token acceptance 提升 **9.7% (4B)** 和 **7.6% (8B)**。

在 **stochastic decoding** 下，DEdit 仍保持领先：
- Qwen3-4B: 6.22 vs DSpark 5.86 (+6.1%)
- Qwen3-8B: 6.09 vs DSpark 5.83 (+4.5%)

### 与基线方法的对比结果
- 在 **MATH 类任务** 上提升最显著（+12–18%），说明复杂推理中未来上下文对纠错帮助大。
- 在 **HumanEval/MBPP** 上与 DSpark 接近，部分组合下 DSpark 更优。
- 在 **MT-Bench** 上也有明显优势（+5–13%）。

### 消融实验结果
#### （1）窗口大小与编辑轮数（Table 2a）
| Method     | W=16 | W=32 | ΔT |
|------------|------|------|-----|
| DEdit (P1) | 5.82 | 5.93 | +0.11 |
| DEdit (P3) | 6.87 | 7.58 | **+0.71** |

- **无编辑时**，扩大窗口仅带来微弱提升（+2.1%）；
- **有编辑时**，从 W=16 到 W=32，T 提升 **10.3%**，证明宽窗口的价值主要体现在提供编辑上下文。

#### （2）训练策略消融（Table 2b）
| Variant           | T     | ΔT   |
|--------------------|-------|------|
| Full (Joint + ProposalMix) | 5.499 | —    |
| w/o Joint          | 5.414 | -0.085 |
| w/o ProposalMix    | 5.355 | -0.144 |
| w/o Both           | 5.242 | -0.257 |

- **PROPOSALMIX 贡献更大**，说明其在训练中教会模型“何时保留、何时修改”至关重要。
- **Joint training** 也有助益，表明共享参数有助于学习更鲁棒的编辑行为。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **未来上下文是关键**：DEdit 的成功依赖于 **future proposal context**。实验证明，当禁用未来 token 注意力（改为 causal attention）时，接受率显著下降，尤其在可预测任务（如推理、摘要、复制类任务）上差距最大。
2. ✅ **编辑质量优于频率**：PROPOSALMIX 并未增加编辑次数，而是**减少有害编辑**（使接受前缀变短的编辑）达 **50%**，同时提升大修成功率。
3. ✅ **宽窗口 + 编辑 = 协同增益**：更大的草案窗口不仅提供更多候选，更重要的是为编辑提供了丰富的上下文，其价值远超单纯增加候选数量。
4. ✅ **双向编辑优于因果校正**：相比 Domino 等仅依赖前缀上下文的因果校正方法，DEdit 能利用整个草案进行双向修复，更具灵活性。

### 方法的局限性
- **计算开销**：每轮编辑增加一次 drafter 的前向传播，在某些执行后端可能无法完全掩盖开销。
- **单请求场景**：实验在单请求、CUDA Graph 优化环境下进行，未验证在 **continuous batching** 生产场景下的表现。
- **约 2/3 的编辑无效**：分析显示多数编辑并未改变接受前缀，存在计算浪费。
- **仅测试至 8B 模型**：未扩展到更大规模模型。

### 未来工作方向
- **动态跳过编辑**：根据置信度或收敛性预测是否执行编辑，减少冗余计算。
- **联合算法与调度设计**：在生产系统中结合算法优化与请求调度，最大化吞吐。
- **跨请求的 drafting/verification 交错**：探索多请求间的资源协同。
- **扩展至更大模型**：验证 DEdit 在 70B+ 规模下的有效性。

---

> **总结**：DEdit 通过引入 **迭代双向编辑** 和 **PROPOSALMIX 训练范式**，有效利用了传统 speculative decoding 中被浪费的“未来预测”，实现了当前最高的 token 接受率和端到端加速比。其核心洞见是：**未来的 token 可以用来修复过去的错误**，这一思想为高效推理开辟了新路径。

</details>

---

### 11. [Spike-driven Vision-Language-Action Model](https://arxiv.org/abs/2609.39514)

**Authors**: Shuai Wang, Malu Zhang, Mingquan Liu, Weihui Dai, Dehao Zhang, Jieyuan Zhang, Yimeng Shan, Zijian Zhou, Yang Yang  
**Category**: cs.CL  
**Published**: 2026-10-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.39514v1  

#### Abstract
Vision-language-action (VLA) models bridge multimodal understanding and robotic control, advancing the dominant paradigm for embodied intelligence. However, most existing models rely on large Transformers, whose latency and energy costs hinder deployment on resource-constrained platforms. Through sp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **Spike-driven Vision-Language-Action Model** 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Vision-Language-Action (VLA)** 模型大多基于大型 Transformer 架构，虽然在机器人控制任务中表现出色，但其高计算开销（FLOPs）、内存占用和推理延迟严重限制了在资源受限边缘平台（如嵌入式机器人系统）上的部署。

此外，现有轻量化 VLA 模型（如 TinyVLA、SmolVLA）仍依赖密集的乘累加（MAC）操作，无法充分利用事件驱动计算的潜力。

---

### 🚀 提出的新方法与创新思路
本文提出了首个支持**端到端直接训练**的 **Spike-driven VLA** 框架，将脉冲神经网络（SNN）引入 VLA 领域，实现高效能、低功耗的具身智能决策。其三大核心组件为：

#### （1）**Spiking Visual/Instruction Encoders (SVE/SIE)**
- 将视觉观测和语言指令分别编码为稀疏的脉冲表示。
- SVE 基于预训练的 Spiking Transformer 提取多尺度视觉特征。
- SIE 基于冻结的 Spiking BERT（SmoothSpike）提取上下文语义，并通过可学习投影适配至共享空间。

#### （2）**Multi-Winner Spike Fusion (MWSF)**
- 引入双向 Top-k Winner-Take-All（WTA）路由机制，在跨模态融合时抑制背景干扰。
- 利用共享亲和矩阵进行稀疏 token 聚合，显著减少无效计算。
- 实现“指令引导”的场景理解，增强任务相关性。

#### （3）**Spike Action Chunking Transformer (SpikeACT)**
- 结合融合后的记忆 `M_n` 和当前机器人状态 `S_n`，并行预测连续的动作块（action chunk）。
- 使用 spiking cross-attention 机制，仅在有脉冲输入时触发计算，极大提升能效。
- 支持 chunked control，摊薄每步控制的计算成本。

---

### ⚖️ 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **能效性** | 利用 SNN 的稀疏事件驱动特性，将 MAC 操作替换为低能耗的 AC 操作（理论能耗仅为 1/5–1/10），显著降低推理能量消耗。 |
| **参数量小** | 仅含 **0.15B 参数**，远小于多数 ANN-based VLA（如 OpenVLA: 7.5B, To: 3.2B）。 |
| **计算效率高** | 推理 FLOPs 极低（LIBERO 上仅 **15.68G**），适合边缘部署。 |
| **端到端可训练** | 不依赖 ANN-to-SNN 转换，而是采用直接训练（direct training），允许任务特定架构设计和 spike 域内优化。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **Meta-World MT50**  
  包含 50 种桌面操作任务（如 push, pick-place），由 Sawyer 机械臂执行。使用 SmolVLA 发布的版本，共 2,500 条演示轨迹。
  
- **LIBERO (Spatial, Object, Goal, Long)**  
  四个任务套件，共 40 个语言条件操控任务。使用 OpenVLA 提供的 no-op RLDS 数据集，动作以 12 步 chunk 形式输出。

- **LIBERO-Plus (零样本鲁棒性测试)**  
  在原始 LIBERO 基础上引入七类分布偏移（object layout, camera viewpoint, lighting, noise 等），共 10,030 个扰动实例，用于评估模型泛化能力。

---

### 🧪 实验设置与评估指标
| 项目 | 设置 |
|------|------|
| **训练方式** | 行为克隆（Behavior Cloning），使用 masked L1 loss |
| **批量大小** | 有效 batch size 为 256（8 GPU × 32） |
| **训练步数** | LIBERO: 80K 步（含 10K 预热）；MT50: 类似设置 |
| **评估方式** | 每任务运行 50 次 rollout，报告平均成功率（Success Rate %） |
| **推理精度** | FP32，EMA 权重，open-loop 执行每个 action chunk 12 步 |
| **能效估算** | 基于 45nm ASIC 模型，AC ≈ 0.9 pJ，MAC ≈ 4.6 pJ，估算单次前向传播能耗 |

---

### 🔁 对比的基线方法
- **TinyVLA (0.42B)** – 轻量级 VLA
- **SmolVLA (0.45B / 2.25B)** – 高效紧凑架构
- **TurboVLA (0.22B)** – 实时 VLA
- **OpenVLA (7.5B)** – 开源通用 VLA
- **To / To.5 (3.2B / 3.4B)** – 流水线式 VLA 模型
- **DreamVLA (0.7B)** – 多世界知识蒸馏模型
- **CogVLA (8.3B)** – 大规模认知对齐模型

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ **Meta-World MT50 性能对比（表1）**

| 方法 | 参数量 (B) | FLOPs (G) | 能耗估计 (mJ) | 平均成功率 (%) |
|------|------------|-----------|----------------|------------------|
| SmolVLA (2.25B) | 2.25 | 4239.5 | 9751.1 | 68.2 |
| **Spike-driven VLA (Ours)** | **0.15** | **10.5** | **11.6** | **72.4** |

> 💡 **结论**：尽管参数量仅为 SmolVLA(2.25B) 的 **1/15**，FLOPs 减少超过 **400倍**，能耗降低 **800倍以上**，但性能反而高出 **4.2个百分点**，尤其在 Medium 和 Hard 任务上表现更优。

---

#### ✅ **LIBERO 性能对比（表2）**

| 方法 | 参数量 (B) | FLOPs (G) | 能耗 (mJ) | 平均成功率 (%) |
|------|------------|-----------|------------|------------------|
| OpenVLA | 7.5 | 4158.3 | 9564 | 76.5 |
| SmolVLA | 2.3 | 521.1 | 1199 | 88.8 |
| DreamVLA | 0.7 | 995.2 | 2289 | 92.6 |
| **Spike-driven VLA (Ours)** | **0.15** | **15.68** | **20.9** | **92.4** |

> 💡 **结论**：
- 成功率接近 DreamVLA（92.4 vs 92.6），超越所有其他轻量级模型；
- 参数量减少 **95%+**，FLOPs 下降两个数量级，能效优势极为显著。

---

#### ✅ **零样本鲁棒性：LIBERO-Plus（表3 & 表10）**
在未经过任何微调的情况下直接测试：

| 方法 | 平均成功率 (%) |
|------|----------------|
| OpenVLA | 16.1 |
| To | 55.9 |
| **Spike-driven VLA (Ours)** | **54.9** |

> 💡 **结论**：
- 尽管参数极小，但在多种分布偏移下仍保持强健的零样本泛化能力；
- 特别是在光照变化（84.2%）和物体布局变化（67.5%）中表现优异；
- 对传感器噪声（34.5%）和视角变化（41.2%）仍有改进空间。

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）**编码器消融（表5）**
| 变体 | 视觉编码器 | 语言编码器 | LIBERO Avg SR (%) | 能耗 (mJ) |
|------|-------------|--------------|--------------------|-----------|
| B0 | DINOv3-B (ANN) | BERT (ANN) | 96.3 | 179.7 |
| B1 | DINOv3-B | SIE (SNN) | 96.1 | 173.0 |
| B2 | SVE (SNN) | BERT | 92.6 | 28.8 |
| B3 | SVE | SIE | **92.4** | **20.9** |

> 🔎 发现：
- SIE 替代 BERT 几乎无损（↓0.2%），说明脉冲语言编码足够表达指令语义；
- SVE 替代 DINOv3-B 导致下降约 3.7%，表明视觉感知是性能瓶颈，需更强先验。

#### （2）**MWSF 中 WTA 模块作用（表4 & 图4）**

| 设置 | LIBERO Overall SR (%) | 提升 |
|------|------------------------|------|
| w/o WTA | 90.2 | — |
| w/ WTA | **92.4** | **+2.2** |

> 🔎 发现：
- WTA 显著提升 Spatial（+4.8%）和 Long（+3.8%）任务表现；
- 注意力热图显示，WTA 能更好聚焦于目标区域（如“red mug”、“plate”），抑制无关背景。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **SNN 可用于构建高性能、低功耗的端到端 VLA 模型**，无需 ANN-to-SNN 转换即可实现直接训练。
2. **Spike-driven VLA 在多个基准上达到甚至超越大规模 ANN 模型的表现**，同时参数量、计算量和能耗大幅降低。
3. **Multi-Winner Spike Fusion 有效增强了指令引导的视觉接地能力**，通过稀疏路由提升鲁棒性和效率。
4. **脉冲驱动的动作生成模块（SpikeACT）实现了高效的 chunked 控制输出**，适用于闭环机器人控制。

---

### ⚠️ 局限性
1. **视觉编码器容量有限**：当前 SVE 基于较小规模的 Spiking Transformer，相比 DINOv3 等大模型在特征表达上有差距。
2. **对极端分布偏移敏感**：在传感器噪声和剧烈视角变化下性能下降明显。
3. **尚未在真实硬件上部署验证能效**：目前能效为理论估算，缺乏 Neuromorphic Chip（如 Loihi、Speck）的实际运行数据。
4. **时间步长固定（T=4）可能影响动态建模能力**：缺乏自适应时序处理机制。

---

### 🔮 未来工作方向
1. **扩展至更大规模的 Spiking Backbone**（如 Spiking ViT-Huge）以提升感知能力。
2. **集成到类脑芯片平台进行实机部署与能效实测**。
3. **探索自适应时序建模机制**（如动态脉冲发放、可变仿真步长）。
4. **拓展至多模态传感输入**（触觉、声音）与复杂长期规划任务。
5. **研究 spike-aware 的强化学习训练范式**，进一步提升策略学习效率。

---

## ✅ 总结
本论文开创性地将 **Spiking Neural Networks** 引入 **Vision-Language-Action Modeling**，提出首个支持**端到端直接训练**的脉冲驱动 VLA 框架——**Spike-driven VLA**。该方法在保持高性能的同时，实现了前所未有的**参数精简与能效优化**，为资源受限平台上的具身智能提供了全新路径，标志着向“神经形态具身代理”（neuromorphic embodied agent）迈出了关键一步。

</details>

---

### 12. [Efficient and Scalable Physics-Guided Fully Convolutional Spatiotemporal Learning for 3D Microstructure Evolution Prediction](https://arxiv.org/abs/2609.36504)

**Authors**: Michael Trimboli, Wenxi Liu, Xianqi Li  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.36504v1  

#### Abstract
Accurate prediction of three-dimensional (3D) microstructure evolution remains computationally demanding because high-fidelity phase-field simulations require repeated numerical integration over large volumetric domains and long temporal horizons. This study develops an efficient and scalable physic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient and Scalable Physics-Guided Fully Convolutional Spatiotemporal Learning for 3D Microstructure Evolution Prediction

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **高计算成本的3D微结构演化预测**：传统的phase-field模拟（如基于Cahn-Hilliard方程）在高分辨率3D空间中进行长时间演化时，需要大量数值积分，计算开销巨大，难以用于多查询场景（如参数优化、不确定性分析等）。
- **现有深度学习模型的局限性**：许多基于RNN或latent-space recurrence的方法存在推理效率低、长期预测不稳定、物理一致性差等问题。

### 提出的新方法与创新点
1. **非递归的全卷积时空学习框架（Fully Convolutional Spatiotemporal Framework）**
   - 采用 **encoder-translator-decoder** 架构，直接从输入序列映射到多个连续未来的3D场（multi-frame prediction），避免逐帧隐藏状态传播。
   - 支持并行化推理，显著提升速度。

2. **因子化解码器（Factorized Latent Translator）**
   - 将时空动态分解为三个独立操作：
     - **Temporal Mixing**：沿时间轴的一维depthwise卷积，建模短期演化模式。
     - **Spatial Mixing**：局部3D卷积，捕捉体素间的空间相互作用。
     - **Channel Mixing**：MLP形式的通道混合，实现特征解耦与非线性融合。
   - 避免了昂贵的4D卷积或全局注意力机制，在保持高效的同时捕获复杂耦合动力学。

3. **物理引导训练（Physics-Guided Training）**
   - 引入离散化的 **Cahn-Hilliard (CH) residual** 作为正则项，约束预测轨迹符合物理规律。
   - **仅在训练阶段引入物理损失，不影响推理路径**，因此不增加部署成本。

4. **可扩展性设计**
   - 全卷积结构支持跨分辨率迁移（无需重新训练即可应用于更高分辨率数据）。
   - 内存和计算成本随体积增长可控。

### 相比现有方法的优势
| 特性 | 本文方法 | 传统RNN/ConvLSTM | 图神经网络（Graph-based） | Physics-Informed NN |
|------|----------|-------------------|----------------------------|---------------------|
| 推理效率 | ⭐⭐⭐⭐⭐（>30倍加速） | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 多帧预测能力 | 直接输出整块未来序列 | 逐帧生成 | 通常单步 | 可能单步 |
| 物理一致性 | 显式CH残差正则化 | 无保证 | 依赖图结构 | 软约束 |
| 3D保真度 | 原生3D卷积，保留拓扑连接 | 容易失真 | 抽象表示可能丢失细节 | 视实现而定 |

---

## 2. 核心实验方法和设置

### 数据集
- **数据来源**：使用开源工具 `SpectralETD` 模拟 **spinodal decomposition** 过程。
- **控制方程**：3D Cahn-Hilliard 方程，双阱自由能函数 $ f(c) = Wc^2(1-c)^2 $。
- **参数设置**：
  - 网格尺寸：$128 \times 128 \times 128$
  - 时间步长：$\Delta t = 0.01$，每10个单位保存一帧，共201帧（对应2000模拟时间）
  - 初始条件：浓度均值0.5 + 高斯噪声
  - 参数固定：$W=1.0, M=1.0, K=0.01$
- **数据划分**：
  - 训练：80条轨迹 → 2960个样本
  - 验证：10条轨迹 → 370个样本
  - 测试：10条轨迹 → 多种任务下测试（370或270样本）

### 实验设置
- **任务类型**：
  1. **Nominal Forecasting (10→10)**：用前10帧预测后10帧
  2. **Long-Horizon Forecasting (10→40)**：迭代rollout预测40帧
  3. **Reduced Context (5→15, 1→19)**：输入帧数减少，考察模型鲁棒性
- **输入格式**：每个样本为形状 $(T, D, H, W) = (20, 128, 128, 128)$ 的滑动窗口，其中前10帧为输入，后10帧为目标。
- **零填充处理**：当可用输入少于10帧时，前面补零以维持输入维度一致。

### 评估指标
| 指标 | 描述 |
|------|------|
| **RMSE** | 均方根误差，衡量像素级差异 |
| **3D SSIM** | 三维结构相似性指数，反映形态保真度 |
| **PSNR** | 峰值信噪比，评价图像质量 |
| **Interface Curvature Distribution (ICD)** | 界面曲率分布比较，评估几何演化合理性 |
| **Total Variation Distance (TVD)** | 衡量预测与真实ICD之间的统计距离，越小越好 |

### 基线方法对比
- **Data-driven baseline**：相同架构但无physics loss（即 $\lambda_{phy}=0$）
- **Physics-guided variant**：加入CH残差正则项（$\lambda_{phy}=10^{-8}$）
- 所有比较均在同一网络结构和训练条件下进行，确保公平。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 实验任务 | 模型 | 平均3D SSIM | RMSE @ 最终帧 | TVD @ 最终帧 | 推理时间（40帧） |
|--------|-------|-------------|---------------|--------------|------------------|
| 10→10 | Proposed | **>0.97** | ~0.007 | — | 0.3682s |
| 10→40 | Physics-guided | ~0.75 | ~0.21 | **0.135** | 同上 |
|        | Baseline   | ~0.75 | ~0.19 | 0.244 | 同上 |
| 5→15  | Physics-guided | ~0.90 | ~0.065 | **0.088** | 同上 |
|        | Baseline   | ~0.90 | ~0.065 | 0.199 | 同上 |
| 1→19  | Physics-guided | **~0.80** | ~0.65 | **0.190** | 同上 |
|        | Baseline   | ~0.70 | ~0.65 | 0.338 | 同上 |

> 注：所有deep learning模型推理时间相同；SpectralETD需约11.896秒 → **~32.3× wall-clock speedup**

### 与基线方法对比结果
- 在标准视觉指标（RMSE, SSIM, PSNR）上，**physics-guided模型并未显著优于baseline**，尤其在短期预测中两者接近。
- 但在**界面曲率分布（ICD）匹配度**方面，physics-guided模型明显更优：
  - 在1→19任务中，TV distance降低超过40%（0.338 vs 0.190）
  - 表明其更好地保留了物理上有意义的界面演化行为。
- 在**长期预测稳定性**方面，physics-guided模型表现出更强鲁棒性，尤其是在输入信息受限时。

### 消融实验结果
- **物理正则项的有效性验证**：
  - 当输入上下文充足（如10→10）时，physics loss贡献较小，数据驱动已足够准确。
  - 当输入减少至1帧时，physics guidance成为关键约束，防止模型发散。
- **训练损失分析**（Fig. 7）显示：
  - Physics loss（weighted $L_{CH}$）虽小，但稳定下降，说明有效参与优化过程。
  - 总损失收敛良好，未出现训练不稳定现象。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **全卷积非递归架构适用于3D microstructure evolution预测**  
   - 成功实现了高效、可扩展的端到端3D时空预测，无需循环机制。
   - 在128³分辨率下实现 **>30倍于spectral phase-field solver的速度提升**。

2. 🔍 **物理引导的价值取决于上下文信息丰富程度**  
   - 当输入序列完整时，纯数据驱动模型已能取得高精度。
   - **随着输入帧数减少，physics-guided模型优势愈发明显**，特别是在形态稳定性和界面演化保真度方面。

3. 📏 **应结合多种评估指标判断模型性能**
   - 单纯依赖RMSE/SSIM可能误导——两个模型可能有相近像素误差，但一个严重偏离物理规律。
   - **Interface Curvature Distribution + TVD 是评估“物理合理性”的重要补充指标**。

4. ⚙️ **物理正则项是“软约束”，不影响推理效率**
   - CH residual仅在训练中使用，**推理路径完全不变**，适合实际部署。

### 方法的局限性
- ❌ **当前仅限于二元系统且参数固定**：未考虑不同材料参数、多组分系统或外部场影响。
- ❌ **训练数据全部来自仿真，尚未验证于实验测量的3D microstructure**。
- ❌ **输入长度固定，采用zero-padding处理短序列，可能导致边界伪影**。
- ❌ **CH residual为软正则项，并不能严格保证质量守恒或能量耗散**。

### 未来工作方向
- ✅ 开展 **parameter-conditioned modeling**，使模型泛化至不同 $W, K, M$ 组合。
- ✅ 引入 **variable-length input机制**，替代zero-padding，提高灵活性。
- ✅ 结合 **experimental tomography data**（如APT、EBSD重建）进行真实世界验证。
- ✅ 发展 **uncertainty-aware prediction** 与 **data assimilation框架**，支持在线校正。
- ✅ 扩展至其他phase-field模型（如Allen-Cahn, multiphase-field）及更复杂的microstructure processes（如grain growth with nucleation）。

---

> **总结一句话**：  
> 该研究提出了一种**高效、可扩展、物理引导的全卷积3D时空学习框架**，能够在保持极快推理速度的同时，通过CH方程残差正则化增强预测的物理一致性，特别适用于**低输入上下文、长时程、高通量的3D microstructure evolution surrogate modeling**。

</details>

---

### 13. [Replay the Curvature: Accurate and Scalable NVFP4 Quantization for Large Language Model Inference](https://arxiv.org/abs/2609.36654)

**Authors**: Ruiyi Ding, Jie Li, Kang He, Ziyan Liu, Chengru Song, Yuedong Xu, Yuan Cheng  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.36654v1  

#### Abstract
Large language models make weight storage and memory traffic major inference costs, motivating low-precision formats that represent each weight with only a few bits. Such formats use a scale to map floating-point values into a small codebook; NVFP4 improves local range utilization by letting every 1...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Replay the Curvature: Accurate and Scalable NVFP4 Quantization for Large Language Model Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大语言模型（LLM）在推理时面临巨大的权重存储和内存带宽开销。为此，低精度量化（如 **NVFP4**）被广泛采用以减少每个权重的比特数。然而，现有方法存在两个关键挑战：

1. **算法层面**：在基于 GPTQ 的量化过程中，由于列间顺序依赖性，独立评估一个 block 的 scale 会错误估计其最终重建误差（reconstruction error），导致次优的 scale 选择。
2. **系统层面**：对于超大规模模型（如 397B 参数），全精度权重、校准激活和二阶梯度状态无法同时驻留在单个加速器内存中；而传统的按层分配策略使得大型层的量化过程成为串行瓶颈。

### 提出的新方法
本文提出了 **Schur Replay** 算法与可扩展的执行框架，分别从算法和系统两个维度解决问题。

#### （1）Schur Replay（算法创新）
- **核心思想**：在 GPTQ 的序列化更新上下文中评估每一个候选 scale，而非孤立地评估 block。
- **关键技术**：
  - 引入 **conditional Schur objective**，通过 Schur 补（Schur complement）衡量在考虑未量化列补偿能力后的不可逆误差。
  - 设计 **recurrence-exact replay** 机制，在组内重放 GPTQ 更新流程，精确模拟每个 scale 引发的状态变化。
  - 先通过闭式解（closed-form proposal）快速定位 promising 区域，再在小窗口内进行精确回放搜索，提升效率。

#### （2）可扩展执行基础设施（系统创新）
- **Active-layer residency**：仅将当前层的权重保留在设备上，其余层卸载至主机内存或磁盘。
- **Tiered activation storage**：激活值按 GPU → Host → SSD → HDFS 分级存储，突破内存容量限制。
- **Output-row parallelism**：将同一层的不同输出行分配给不同的 tensor-parallel rank 并行处理，充分利用多卡资源。
- **Recoverable streaming export**：支持断点续传和逐层持久化导出，避免重复计算。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **准确性** | 显著优于 SOAR、ScaleSweep、MR-GPTQ 等 NVFP4 方法，在多个基准上恢复超过 99% 的 BF16 性能。 |
| **可扩展性** | 支持千亿参数级别模型（如 Qwen3.5-397B）的完整量化，峰值显存仅需 35.0 GB/GPU。 |
| **效率** | 在 397B 模型上，每层量化时间比 ModelOpt 快 15.17×，比 LLM Compressor 快 23.14×。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准数据集（Calibration Set）**：
  - 包含 255 个多轮对话文档，来自三个领域：
    - 科学（SciQ）
    - 常识（CommonsenseQA 和 PIQA）
    - 数学（NuminaMath-CoT）
  - 文档长度约 11k 字符，最大 token 长度为 16,384。
  - 经过去污染处理，确保与测试集无重叠。

- **评估数据集（Evaluation Benchmarks）**：
  - **MMLU-Pro**（科学常识）
  - **GSM8K**（小学数学题）
  - **CMath**（中文数学）
  - **LiveCodeBench**（代码生成）
  - **English MGSM**（英文多语言数学）
  - **HumanEval**（Python 函数生成）
  - **GPQA-Diamond**（研究生级问答）

共 15,461 个问题，用于综合评估。

### 实验设置
- **模型**：
  - **Qwen3.5-397B-A17B**（MoE 架构，TP8/EP8）
  - **Llama-3.3-70B-Instruct**（稠密架构，TP1 量化 / TP2 推理）
- **量化格式**：W4A4 NVFP4（权重和激活均为 4-bit）
- **评估协议**：
  - 单样本贪婪解码（greedy pass@1），温度 T=0，top-p=1，最多生成 8,192 新 token。
  - 报告 **question-weighted recovery rate**（相对于 BF16 基线）作为主指标：
    $$
    R_w = 100 \times \frac{\sum_d n_d \cdot \text{acc}_d^{\text{quant}}}{\sum_d n_d \cdot \text{acc}_d^{\text{BF16}}}
    $$

### 基线方法对比
| 方法 | 类型 | 是否使用相同 backend |
|------|------|---------------------|
| GPTQ | 基础量化方法 | 是 |
| SmoothQuant + GPTQ | 激活重标定 | 是 |
| AWQ + GPTQ | 权重重标定 | 是 |
| Four Over Six + GPTQ | 自适应 E2M1 范围 | 是 |
| ScaleSweep | Block scale 搜索 | 是 |
| SOAR + GPTQ | Scale 优化 | 是 |
| ARCQuant | 残差通道增强 | 否（专用流程） |
| MR-GPTQ | 微旋转 + reorder | 否（专用流程） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 方法 | Question-weighted Recovery (%) |
|------|------|-------------------------------|
| Qwen3.5-397B-A17B | **Schur Replay (ours)** | **99.35%** |
| | ScaleSweep | 97.28% |
| | MR-GPTQ | 93.86% |
| | BF16（上限） | 100.00% |
| Llama-3.3-70B-Instruct | **Schur Replay (ours)** | **100.84%** |
| | SOAR + GPTQ | 100.20% |
| | ScaleSweep | 99.03% |
| | BF16（上限） | 100.00% |

> 注：Llama 上超过 100% 是因在有限测试集上个别任务表现略优。

### 与基线方法的对比结果
- 在 **Qwen3.5-397B** 上，Schur Replay 比最强基线（SOAR）高出 **1.07 个百分点**。
- 在 **Llama-70B** 上，达到 **100.84%**，是唯一全面超越 BF16 的量化方法。
- 所有任务中均取得最优或接近最优表现，尤其在 **MMLU-Pro、GSM8K、GPQA** 上优势明显。

### 消融实验结果
作者进行了完整的消融研究（见 Appendix B），验证了各组件的有效性：

| 变体 | Macro Recovery (%) | 相对 BF16 正确题数差 |
|------|------------------|--------------------|
| **Ours (replay × Schur)** | 98.92% | -79 |
| static × Schur（静态残差） | 98.70% | -94 |
| replay × Euclidean（欧氏损失） | 98.00% | -169 |
| static × Euclidean（最弱） | 97.90% | -194 |

- **主要发现**：
  - **Schur scoring** 贡献最大，移除后损失 90 题。
  - **Recurrence replay** 贡献次之，移除后损失 15 题。
  - 两者结合有正向交互作用，说明二者协同有效。

此外，窗口大小 $ h=4 $ 已足够：
- $ h=4 $ 时，83.3% 的行能找到全局最优 scale；
- $ h=8 $ 可达 99.96%，但 recovery 不再提升，表明当前设计已达性价比最优。

---

## 4. 关键结论和发现

### 主要发现
1. **Scale 选择必须考虑 GPTQ 序列动态**：传统独立 block 评估忽略列间补偿效应，导致次优决策；Schur Replay 通过“重放”机制捕捉真实误差传播路径，显著提升重建质量。
2. **条件 Schur 损失是更合理的评估目标**：它衡量的是“在允许未来列最优补偿后仍无法消除的误差”，比原始 MSE 更贴近最终效果。
3. **系统级优化至关重要**：仅靠算法改进无法支撑千亿模型量化；active-layer residency + row-parallelism 实现了真正的可扩展性。
4. **高保真量化无需重新训练**：Schur Replay 在 PTQ 场景下即可实现近乎 BF16 的性能，证明了无需 QAT 或蒸馏也能达成高质量压缩。

### 方法的局限性
- **依赖 GPTQ 框架**：目前仅适用于 GPTQ 类二阶补偿量化器，是否可推广至其他范式（如 AWQ、SmoothQuant）尚待验证。
- **计算开销增加**：尽管使用 proposal + 小窗口搜索，但仍比简单 scale 选择更耗时（但远低于端到端收益）。
- **硬件兼容性约束**：scale 必须属于 E4M3 可表示集合，搜索空间受限于部署要求。

### 未来工作方向
- 探索 Schur Replay 是否适用于其他微缩放格式（microscaling formats），如 INT4 或 NF4。
- 扩展至更多模型家族（如 Mistral、Phi）、任务类型和校准策略。
- 将该方法集成进训练阶段（QAT 或 distillation），进一步逼近理论极限。
- 开源实现并推动工业级部署，降低大模型服务成本。

---

> ✅ **一句话总结**：  
> 本文提出 **Schur Replay** ——一种在真实 GPTQ 动态中评估 block scale 的新算法，并配合高效的分布式执行框架，首次实现了 **397B 级别模型的高保真、可扩展 NVFP4 量化**，在七项基准上恢复 **99.35%~100.84%** 的 BF16 性能，显著优于现有方法。

</details>

---

### 14. [LongSpark: Efficient speculative decoding with a fixed-cost parallel drafter](https://arxiv.org/abs/2609.37029)

**Authors**: Hao-Yuan He, Peng-Fei Liu, Si Shen, Ming Li  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.37029v1  

#### Abstract
Speculative decoding accelerates autoregressive inference by verifying multiple draft tokens in a single target forward pass. However, as the context grows, existing state-of-the-art drafters become increasingly expensive, eroding the very efficiency advantage they are designed to provide. We argue ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LongSpark: Efficient Speculative Decoding with a Fixed-Cost Parallel Drafter**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前主流的 **speculative decoding** 方法虽然能通过轻量级 **drafter** 模型并行生成多个候选 token 来加速 **autoregressive inference**，但其 **drafter 的计算和内存开销随上下文长度（prefix length）增长而线性增加**。这导致在长上下文场景下，drafter 自身成为性能瓶颈，削弱了其本应带来的效率优势。

具体而言：
- **Autoregressive drafters** 维护随上下文增长的 KV cache。
- **Block-diffusion drafters**（如 DFlash、DSpark）虽能并行预测多个 token，但仍需从目标模型（target）读取整个前缀的特征，导致注意力计算复杂度为 $O(t)$。

### **提出的新方法与核心思想**
论文提出了 **LONGSPARK**，一种基于 **block-diffusion** 架构的 **fixed-cost drafter**，其核心创新是 **fixed-cost drafting** 范式：

> **由于 target 模型会通过 rejection sampling 验证所有 draft token 并纠正错误，因此 drafter 不需要完整、精确地建模整个上下文。它只需提供“足够好”的提案，且其成本可以完全独立于上下文长度。**

为此，LONGSPARK 设计了一个 **固定大小、多尺度的上下文接口（fixed-cost context interface）**，仅从 target 的验证过程中提取三种固定尺寸的视图：
1. **Boundary State**: 解码边界的单个隐藏状态，作为生成起点。
2. **Recent KV Window**: 最近 $W$ 个 token 的 KV 缓存，保留局部细节。
3. **Global Context Summary**: 一个由可学习查询（learned summary queries）生成的全局摘要，将整个前缀压缩为固定数量的条目，并支持增量更新。

所有这些操作的计算和存储成本均为 $O(1)$，与上下文长度 $t$ 无关。

### **相比现有方法的优势**
- **效率更高**：drafter 的每轮提案时间几乎恒定，不随上下文增长而变慢。
- **内存更小**：drafter 的上下文状态（context state）比现有方法（如 DSpark）小几个数量级（在 128K 上下文时减少 406 倍）。
- **端到端吞吐量更高**：尤其在高并发和长上下文场景下，优势显著。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **通用任务基准**（用于评估平均性能）：
  - 数学推理：`GSM8K`, `MATH-500`, `AIME25`
  - 代码生成：`MBPP`, `HumanEval`, `LiveCodeBench (LCB)`
  - 对话：`MT-Bench`, `Alpaca`
- **长上下文专项测试集**（用于评估扩展性）：
  - `LongSpec 32K`: 32K 上下文的延续任务。
  - `CodeSpan 64K`: 64K 上下文的源代码延续。
  - `LongSWE-Bench (LSWE) 128K`: 128K 上下文的仓库级 Bug 修复。

训练数据：`Open-PerfectBlend`（约 1.42M 样本），使用对应 target 模型重新生成响应以确保分布对齐。

### **实验设置**
- **Target 模型**：`Qwen3-4B`, `Qwen3-8B`, `Qwen3-14B`
- **Drafter 模型**：轻量级 decoder，5 层，预测 7 个 token/block。
- **硬件配置**：采用 TD（Target-Drafter）分离架构，额外添加一块 drafter GPU。
- **并发设置**：测试了不同并发级别（8, 16, 32, 128）下的性能。
- **生成参数**：温度 $T=1$ 或 $T=0$（贪婪解码），最大生成 2048 或 8192 token。

### **评估指标**
- **Throughput Speedup**：相对于自回归解码的吞吐量提升倍数（tokens/s）。
- **Accepted Length ($\bar{r}$)**：每轮验证中被接受的 token 数量的期望值。
- **Time Per Output Token (TPOT)**：每个输出 token 的平均耗时（ms/token），衡量延迟。
- **Drafting Overhead**：提案阶段的时间和内存占用。

### **基线方法对比**
- **Vanilla Autoregressive**: 基准。
- **EAGLE-3**: 基于隐藏状态复用的 autoregressive drafter。
- **DFlash**: 单步 block-diffusion drafter。
- **DSpark**: 当前最先进的 block-diffusion drafter，具有半自回归修正和动态提案长度。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **端到端吞吐量提升（Table 1 & Table 3）**
在多种模型规模下，LONGSPARK 均实现了最高的平均吞吐量提升：
- `Qwen3-4B`: **1.88×** 加速
- `Qwen3-8B`: **1.99×** 加速
- `Qwen3-14B`: **2.13×** 加速

即使在贪婪解码（$T=0$）下，优势依然存在，证明其为结构性优势。

#### **长上下文性能（Table 2 & Table 4）**
在 32K–128K 长上下文任务上，LONGSPARK 在所有三个 benchmark 上均取得最低 TPOT 和最高 TPS：
- **CodeSpan 64K**：TPS 达 **1571.8 tok/s**，比 DSpark 高 **29.1%**；TPOT 低 **23.1%**。
- **LSWE 128K**：TPS 达 **212.7 tok/s**，TPOT 降至 **74.3 ms/token**。

#### **内存与计算开销（Figure 1）**
- **Drafter Context State**：
  - DSpark 在 128K 上下文时达 **2560 MiB**。
  - LONGSPARK 恒定为 **6.3 MiB**，减少 **406×**。
- **Drafting Time**：
  - DSpark 从 32K 的 6.3ms 增至 128K 的 13.8ms。
  - LONGSPARK 始终稳定在 **~3.5ms**。

### **与基线方法的对比结果**
- **优于 DSpark**：尽管 DSpark 的 accepted length 更长，但因其 drafting 开销大，在高并发和长上下文下吞吐量反被 LONGSPARK 超越。
- **高并发优势放大**（Figure 4 & Table 6）：随着并发数从 8 增至 128，LONGSPARK 的吞吐优势持续扩大，因其固定成本避免了内存带宽瓶颈。
- **优于 EAGLE-3 和 DFlash**：在所有指标上全面领先。

### **消融实验结果（Figure 5 & Table 5）**
移除任一上下文组件均导致性能下降：
- **移除 Recent KV Window**：平均 accepted length 从 4.69 降至 3.71，影响最大 → 表明局部 token 细节最关键。
- **移除 Global Context Summary**：accepted length 降至 4.35，训练损失出现剧烈波动 → 表明全局摘要对训练稳定性重要。
- **移除 Boundary State**：性能轻微下降至 4.28。
- **结论**：三者协同作用，其中 recent KV window 是提案质量的主要驱动力。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Drafter 不必随上下文线性扩展**：得益于 target 的验证机制，drafter 可以使用有损、压缩的上下文视图，实现 **fixed-cost drafting**。
2. **低开销比高质量更重要**：在 speculative decoding 中，**单位提案开销下获得的 accepted token 数** 是核心优化目标。LONGSPARK 通过极低的固定开销，实现了更高的端到端效率，即使其 accepted length 略低于 DSpark。
3. **优势随上下文和并发增长而放大**：LONGSPARK 的 $O(1)$ 成本特性使其在长文本和高负载服务场景中表现尤为出色。

### **方法的局限性**
- **依赖 target 的中间状态**：需要从 target 模型获取边界状态、KV cache 和隐藏层特征，可能限制其在某些部署环境中的灵活性。
- **压缩表示的理论极限**：目前尚不清楚在极端长上下文（如 1M tokens）下，固定大小的 summary 是否仍能提供足够的语义信息。
- **训练复杂性**：需要联合训练 summary queries 和 drafter 网络，增加了训练难度。

### **未来工作方向**
- **更大规模和更长上下文的验证**：在更大的 target 模型和百万级上下文窗口中评估 fixed-cost 接口的有效性。
- **替代接口设计**：探索其他形式的 fixed-size context summarization 方法。
- **树形或多 drafter 验证**：结合 tree-structured 或 multi-drafter 机制，在保持 $O(1)$ 提案成本的同时进一步提高每轮接受的 token 数。
- **强化学习应用**：在 RL rollout 等长轨迹生成场景中应用，利用恒定开销累积显著的吞吐增益。

--- 

> **总结**：LONGSPARK 通过引入 **fixed-cost drafting** 范式，从根本上改变了 speculative decoding 的设计哲学——从“尽可能准确地建模上下文”转向“以最小固定成本提供有用提案”。其实验结果充分证明，在真实的服务负载下，**恒定的低开销比边际的提案质量提升更能决定最终的系统性能**。

</details>

---

### 15. [Probe-Space Preconditioning for Fast and Stable Zero-Order Training](https://arxiv.org/abs/2609.38095)

**Authors**: Francois Chaubard, Mykel J. Kochenderfer, Chris R\'e  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.38095v1  

#### Abstract
Backpropagation (BP) dominates deep learning but imposes a massive memory tax. For example, training OPT-30B with Adam requires $\approx$ 600GB of GPU memory (assuming batch size 8 and sequence length 2048). Alternatively, zero-order optimization (ZOO) trains in inference-mode (requiring only $\appr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Probe-Space Preconditioning for Fast and Stable Zero-Order Training**

---

## 1. **论文的主要贡献和创新点**

### **解决的问题**
当前深度学习训练严重依赖 **Backpropagation (BP)** 和自适应优化器（如 **Adam**），但这带来了巨大的 **GPU 内存开销**，尤其是在训练大模型（如 OPT-30B）时，需要高达 ~600GB 的内存。这限制了在单个加速器上的训练能力。

此外，神经网络的损失曲面通常具有高度病态的 Hessian 矩阵（即条件数高），导致训练不稳定，需要复杂的优化策略。

本文旨在设计一种满足以下三个目标的新求解器：
1. **推理模式内存使用**（Inference-Mode Memory Use）：不存储激活值、梯度或动量状态。
2. **对病态损失鲁棒**（Robust to Ill-Conditioned Loss）：能在高曲率方向上稳定收敛。
3. **高效的训练计算**（Efficient Training Compute）：以更少的浮点运算实现更高的性能提升。

### **提出的新方法：1.5-SPSA**
作者基于 **Zero-Order Optimization (ZOO)** 框架，提出了 **1.5-SPSA**，其核心思想是：
- 在 **1SPSA** 的基础上，每一步增加一个“干净”的前向传播（clean forward-pass），用于估计 **探针空间中的方向曲率（directional curvature）**。
- 利用该曲率构建一个 **廉价的对角预处理器（diagonal preconditioner）**，在更新中对高曲率方向进行降权，从而提高稳定性并允许更大的步长。

### **相比现有方法的优势**
| 方法 | 内存 | 步数 | 准确率 | 特点 |
|------|------|------|--------|------|
| **BP + Adam** | ~600GB | 数万步 | 中等 | 高内存，标准方法 |
| **MeZO** | ~60GB | 100,000步 | 91.4% (SST-2) | ZOO，低内存，但慢 |
| **1.5-SPSA** | ~60GB | **仅70步** | **94.5% (SST-2)** | **低内存 + 快速 + 更高准确率** |

- **内存降低10倍**：从 ~600GB 降至 ~60GB，可在消费级 GPU（如 A100）上原位训练 OPT-30B。
- **训练速度大幅提升**：相比 MeZO 的 100,000 步，1.5-SPSA 仅需 **70 步** 即可超越其性能。
- **性能超越 BP 和 MeZO**：在多个任务上达到 **SOTA ZOO 性能**，甚至优于 BP。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **GLUE/SuperGLUE 任务**：
  - **SST-2**（情感分类）
  - **RTE**, **BoolQ**, **WSC**, **WiC**（自然语言理解）
- **Stable ToolBench**：用于测试工具调用能力（Tool-use）
- **合成任务**：
  - **Stiff Paraboloid**：控制条件数的凸优化问题
  - **DNC Overfitting**：非凸、难训练的递归模型压力测试

### **实验设置**
- **模型**：
  - **OPT-13B / OPT-30B**
  - **Qwen3-1.7B / Qwen3-8B**
- **评估指标**：
  - **测试准确率（Test Accuracy）**
  - **收敛所需步数（Steps to Convergence）**
  - **总前向传播次数（Forward-passes）**
  - **Wall-clock 时间**
- **超参数设置**：
  - 学习率 $ \lambda = \epsilon $（绑定设置）
  - 曲率饱和指数 $ \alpha = 0.1 $
  - 扰动数量 $ n_{\text{pert}} \in \{40, 60, 100\} $
  - 批大小（有效）$ \in \{128, 256\} $

### **基线方法对比**
| 基线方法 | 类型 | 是否使用梯度 | 内存开销 | 步数 |
|---------|------|--------------|----------|------|
| **BP + Adam** | 一阶 | 是 | 高 (~600GB) | 多 |
| **MeZO** | ZOO | 否 | 低 (~60GB) | 极多 (100K+) |
| **1SPSA** | ZOO | 否 | 低 | 较少 |
| **1.5-SPSA** | ZOO | 否 | 低 | **极少 (70步)** |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **OPT-13B 上的 SST-2 结果**
| 方法 | 准确率 | 优化步数 | 前向传播数 |
|------|--------|----------|-------------|
| MeZO | 91.4% | 100,000 | ~200k |
| BP + Adam | 92.0% | 多 | 多 |
| **1.5-SPSA** | **94.5%** | **70** | **179k** |

> **+3.1% 超越 MeZO，+2.5% 超越 BP，且仅用 70 步！**

#### ✅ **OPT-30B 结果（Table 2）**
- **1.5-SPSA 在所有任务上均优于 MeZO 和 1SPSA**
- 证明方法可扩展至超大规模模型

#### ✅ **Qwen3-8B（Table 3）**
| 方法 | SST-2 | RTE | BoolQ | WSC | WiC |
|------|-------|-----|-------|-----|-----|
| 1SPSA | 94.5 | 88.5 | 85.7 | 75.0 | 64.6 |
| **1.5-SPSA** | **94.7** | **88.0** | **86.1** | **80.8** | **71.2** |

> 在现代架构上依然有效。

#### ✅ **Qwen3-1.7B on Stable ToolBench（Table 4）**
- **1.5-SPSA 在 lr=1e-3 下仍能收敛（24步）**
- MeZO 和 1SPSA 在 lr=5e-4 时已发散
> 表明 **1.5-SPSA 支持更大学习率，收敛更快**

---

### **消融实验结果**

#### 🔬 **α（曲率饱和指数）消融（Table 5）**
| α | 最佳准确率 | 状态 |
|----|------------|------|
| 0.01 | 93.6% | 发散风险高 |
| **0.1** | **94.5%** | **最优，稳定** |
| 0.5 | 89.7% | 过度降权，性能下降 |
| 1.0 | 89.2% | 接近牛顿法，不稳定 |

> **α=0.1 是最佳选择**：轻微降权即可稳定训练。

#### 🔬 **ε（扰动大小）消融（Table 6）**
| ε | 最佳准确率 | 状态 |
|----|------------|------|
| 1e-4 | **94.5%** | **最佳（与 lr 绑定）** |
| <1e-5 或 >1e-3 | <80.5% | 不收敛 |

> **ε = λ = 1e-4 是最稳定配置**

#### 🔬 **系统优化（Table 7）**
| 方法 | 扰动生成时间 | 总步耗时 | 加速比 |
|------|----------------|-----------|--------|
| PyTorch | 499.5 ms/pert | 59.9 s | 1.0× |
| Triton + Bit-Packed | **102.1 ms/pert** | **21.7 s** | **2.76×** |

> 通过 **8-bit 打包随机生成器 + Triton 融合内核**，显著提升速度。

---

## 4. **关键结论和发现**

### **主要发现**
1. **大批次 + 多扰动 + 少步骤** 比 “小批次 + 少扰动 + 多步骤” 更高效。
2. **探针空间中的曲率估计** 可用于构建有效的预处理器，无需额外内存。
3. **1.5-SPSA 显著提升 ZOO 的收敛速度和稳定性**，在多个任务上达到 SOTA。
4. **性能增益与 Hessian 条件数正相关**：病态越严重，1.5-SPSA 优势越大（Figure 3）。
5. **系统优化至关重要**：Bit-packing + Triton + 分布式并行使大规模 ZOO 训练可行。

### **方法的局限性**
- **未全面调优 BP 基线**：作者承认未在相同极端计算分配下重新优化 BP，因此不能断言 1.5-SPSA 在所有场景下都优于 BP。
- **依赖曲率估计质量**：当曲率接近零或极大时，估计可能不稳定，需通过 $ \alpha $-saturation 和正则化缓解。
- **并行性要求高**：需要足够 GPU 并行处理多个扰动。

### **未来工作方向**
- **批归一化的曲率归一化**：探索跨批次或扰动的相对曲率估计，进一步稳定更新。
- **更智能的学习率调度**：替代暴力搜索，采用线性/二次拟合等自动调参方法。
- **扩展到 RL 和 MoE 模型**：验证在强化学习和稀疏模型中的有效性。
- **理论分析**：建立 1.5-SPSA 在非凸、随机设定下的收敛保证。

---

> **一句话总结**：  
> **1.5-SPSA 通过在探针空间中引入廉价的曲率感知预处理，在保持 ZOO 低内存优势的同时，实现了比 MeZO 快数百倍、比 BP 更高的性能，是迈向高效、稳定、可扩展零阶训练的重要一步。**

</details>

---

### 16. [Self-Evolving Algorithm-Design Agents: Escaping In-Context Evolutionary Stagnation via Population-Curated Policy Optimization](https://arxiv.org/abs/2609.38757)

**Authors**: Chen Lu, Ke Xue, Siyuan Xu, Mingxuan Yuan, Chao Qian  
**Category**: cs.AI  
**Published**: 2026-10-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.38757v1  

#### Abstract
Large language models are increasingly participating in complex real-world tasks in the form of algorithm-design agents, designing and refining algorithms. Many successful algorithm-design agents adopt pure in-context evolutionary frameworks, but they may quickly plateau in domains that require spec...

---

### 17. [Reinforcement Learning-Guided Graph Transformations for SpTRSV Optimization](https://arxiv.org/abs/2609.40159)

**Authors**: Buse Y{\i}lmaz  
**Category**: cs.DC  
**Published**: 2026-10-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.40159v1  

#### Abstract
Sparse triangular solve (SpTRSV) is a fundamental kernel in numerous scientific and engineering applications. However, the data dependencies inherent in sparse triangular matrices significantly limit the available parallelism and make efficient workload distribution challenging. Recent graph transfo...

---

### 18. [ThinQuant: Scalable Rotation Learning for Weight and Activation Quantization of LLMs](https://arxiv.org/abs/2609.36120)

**Authors**: Mehdi Makni, Ryan Lucas, Rahul Mazumder  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.36120v1  

#### Abstract
Learned rotations play an important role in enabling low-bit weight and activation quantization of large language models by smoothing outliers in the activation distribution. State-of-the-art approaches include gradient-based procedures such as SpinQuant and computationally friendlier gradient-free ...

---

### 19. [Unlocking the Critic: Reward-Free Policy Optimization for LLM Post-Training](https://arxiv.org/abs/2609.37119)

**Authors**: Hongyang Li, Xiao Li, Caesar Wu, Said Mammar, Gr\'egoire Danoy, Pascal Bouvry  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.37119v1  

#### Abstract
Recent approaches to reinforcement learning (RL) post-training for large language models increasingly remove the critic to reduce training instability and memory overhead. Even where a critic is trained, it is discarded once training ends, although it has learned to predict outcomes. We revisit this...

---

### 20. [Decode-Latency Feedback Prefill: A Model-Free Controller and Its Generalization Limits](https://arxiv.org/abs/2609.38386)

**Authors**: Gaurav Agarwal, Ashish Garg, Isha Singhal  
**Category**: cs.AI  
**Published**: 2026-10-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.38386v1  

#### Abstract
Concurrent autoregressive inference creates a fundamental interference problem: prefilling a newly arrived long prompt can delay tokens for requests that are already decoding. Fixed prefill chunks reduce this interference, but the best chunk size depends on the model, hardware, load, and latency obj...

---

### 21. [Koa-action: Fast and Consistent Structured Decision Making with Generative LLMs](https://arxiv.org/abs/2609.36115)

**Authors**: Shenghong Dai, Shiva Kumar Pentyala, Yingchi Liu, Shubham Mehrotra, Suman Banerjee, James Zhu, Bin Bi, Sitaram Asur, Phil Mui  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.36115v1  

#### Abstract
Industry applications often demand low-latency classification, yet current large language model (LLM) approaches remain poorly suited for latency-critical applications. Existing prompting and constrained decoding produce verbose, multi-token outputs that require expensive token-by-token generation, ...

---

### 22. [Message Passing Does More with Less for In-Context Learning on Graphs](https://arxiv.org/abs/2609.37057)

**Authors**: Dooho Lee, Jinmo Lee, Minho Jeong, Kijung Shin, Jaemin Yoo  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.37057v1  

#### Abstract
Achieving strong performance with graph neural networks (GNNs) typically requires training and hyperparameter tuning for each dataset, incurring repeated costs and effort. Graph in-context learning (ICL) avoids this by using a single pretrained model to predict unknown node labels directly from labe...

---

### 23. [GLASS: Global Latent Aggregation with Slot-based Set Decoding for Scalable All-Atom Crystal Generation](https://arxiv.org/abs/2609.37158)

**Authors**: Hendrik Kra{\ss}, Seyed Mohamad Moosavi, Mathias Niepert  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.37158v1  

#### Abstract
Generative models for crystals enable the discovery of novel structures, but scaling all-atom generation to larger systems such as metal--organic frameworks remains challenging. We connect this difficulty to the correspondence problem of particle-space generation. Even on a single fixed target set, ...

---

### 24. [SkillFM: Generating Skills for LLM Agents via Latent Flow Matching](https://arxiv.org/abs/2609.39382)

**Authors**: Zuming Zhang, Jie He, Yizhe Zhang, Jeff Z. Pan  
**Category**: cs.AI  
**Published**: 2026-10-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.39382v1  

#### Abstract
Textual skills provide reusable guidance for large language model agents, but existing approaches often rely on manually curated skill banks or reinforcement learning with indirect and delayed feedback. We introduce SkillFM (Skill Flow Matching), a generative framework that synthesizes task-conditio...

---

### 25. [UBTree: Parallel Tree Drafting via Unigram and Bigram Models for Speculative Decoding](https://arxiv.org/abs/2609.39972)

**Authors**: Chumeng Liang, Linxuan Wang, Xinyu Peng, Huabin Liu, Yuxin Chen, Ge Liu, Guang Lin, Qifan Song, Jianguo Li  
**Category**: cs.CL  
**Published**: 2026-10-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.39972v1  

#### Abstract
Speculative decoding accelerates language model inference by verifying multiple draft tokens in a single target-model pass. Recent parallel drafters have achieved breakthrough performance in frontier production models, but their effectiveness deteriorates as the entropy of target distributions incre...

---

### 26. [Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling](https://arxiv.org/abs/2609.36529)

**Authors**: Oliver Sieberling, Bharat Runwal, David Jin, Ryan Chin, Rameswar Panda, Yoon Kim  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.36529v1  

#### Abstract
Recurrent neural networks (RNNs) compress the historical context into a memory state of fixed size, thus allowing for constant-time inference. The memory state size is a crucial factor in their performance, as exemplified by the strong performance and resurgence of linear attention, which extends th...

---

### 27. [On-Policy Parameter Update Direction Underlies Generalization in LLM Post-Training](https://arxiv.org/abs/2609.36659)

**Authors**: Shufan Shen, Zhongni Hou, Junshu Sun, Yufei Zhang, Wei Lin, Guojun Yin, Qingming Huang, Shuhui Wang  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.36659v1  

#### Abstract
The strong generalization performance of on-policy post-training paradigms has motivated studies of their parameter update behaviors. However, these studies treat the observed behaviors only as byproducts in on-policy training, overlooking their potential to serve as optimization principles for impr...

---

### 28. [Architecture Alignment With Sparse Priors in Tabular Foundation Models](https://arxiv.org/abs/2609.36883)

**Authors**: Tianqi Zhao, Tianyi Zhuang, Shuo Duan, Guanyang Wang, Yan Shuo Tan, Qiong Zhang  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.36883v1  

#### Abstract
Tabular foundation models (TFMs) are increasingly popular because they deliver strong predictions on new datasets through in-context learning, without task-specific training or extensive tuning. Yet released TFMs differ simultaneously in their pretraining priors, architectures, and objectives, obscu...

---

### 29. [Privy to the Foil: Recasting Value Estimation with a Self-Privileged Critic for RLVR](https://arxiv.org/abs/2609.37825)

**Authors**: Kun Liang, Chenming Tang, Clive Bai, Weijie Liu, Zeyuan Liu, Qingyang Zhang, Saiyong Yang, Yunfang Wu  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.37825v1  

#### Abstract
Assigning credit to intermediate steps remains a central challenge in training Large Language Models (LLMs) on multi-step reasoning tasks with sparse terminal rewards, and actor-critic methods such as PPO address this by learning value functions to construct token-level advantages. Their effectivene...

---

### 30. [Scaling Zero-Order Pretraining through Model Sharding](https://arxiv.org/abs/2609.37899)

**Authors**: Francois Chaubard, Mykel J. Kochenderfer, Chris R\'e  
**Category**: cs.LG  
**Published**: 2026-10-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.37899v1  

#### Abstract
Zero-order optimization (ZO) trains without backpropagation, making it relevant to forward-only hardware and non-differentiable loss, but its gradient variance grows with perturbed dimension, inhibiting large-model training. Sharded Optimization Mixture of Assemblies (SOMA) trains LSTM experts indep...

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

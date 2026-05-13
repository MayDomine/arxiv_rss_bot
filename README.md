# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-13 08:24:28 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models](https://arxiv.org/abs/2605.11854)

**Authors**: Kecheng Chen, Ziru Liu, Xijia Tao, Hui Liu, Yibing Liu, Xinyu Fu, Shi Wu, Suiyun Zhang, Dandan Tu, Lingpeng Kong, Rui Liu, Haoliang Li  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.11854v1  

#### Abstract
Diffusion Language Models (DLMs) have recently emerged as a promising alternative to autoregressive language models, offering stronger global awareness and highly parallel generation. However, post-training DLMs with standard Negative Evidence Lower Bound (NELBO)-based supervised fine-tuning remains...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Diffusion Language Models (DLMs) 在推理时采用**confidence-guided、multi-step 的由易到难（easy-to-hard）去噪路径**，而训练阶段通常使用标准的 **Negative Evidence Lower Bound (NELBO)** 目标函数，通过**随机均匀掩码**进行单步重建。这种“**训练-推理不一致**”（training-inference discrepancy）导致模型难以充分吸收推理轨迹中的结构性知识。

尽管已有工作（如 dInfer、T3D）利用 self-distilled 轨迹进行后训练，但其目标主要是**加速推理**（如减少采样步数），而非提升模型能力本身，甚至在全步长解码下可能导致性能下降。

### 提出了什么新方法或新思路
本文提出 **Trajectory-Aligned optimization via Boltzmann Modeling (TABOM)**，一种基于 self-distilled 轨迹的新型后训练框架，旨在将推理轨迹用于**真正的知识获取**，而不仅仅是效率优化。

核心思想包括：
- **理论建模**：将推理过程中的 unmasking 偏好建模为一个 **Boltzmann 分布**，其能量项为 token 的理想预测熵（predictive entropy）。这形式化地捕捉了“由易到难”的归纳偏置。
- **可学习目标**：由于直接优化 KL 散度不可行（涉及不可计算的配分函数），提出一个**基于成对排序（pairwise ranking）的可处理代理目标**，强制模型预测的熵顺序与实际解码轨迹一致。
- **局部窗口机制**：在固定大小的局部时间窗口内（如 $W=32$）应用排序损失，避免跨阶段比较带来的噪声。

### 相比现有方法的优势
| 方法 | 主要目标 | 是否提升生成质量 | 是否缓解灾难性遗忘 | 是否支持并行解码 |
|------|----------|------------------|--------------------|------------------|
| SFT-GT | 提升领域性能 | ✅（有限） | ❌（严重遗忘） | ❌ |
| SFT-SD / dInfer / T3D | 推理加速 | ❌ 或有限 | ✅ | ✅ |
| **TABOM (Ours)** | **对齐训练-推理分布，提升能力** | ✅✅（显著） | ✅✅（完全缓解） | ✅ |

- **首次证明**：self-distilled 轨迹可用于**实质性性能提升**，而不仅是压缩。
- **实现双赢**：既获得接近 SFT-GT 的 in-domain 性能增益，又保留 SFT-SD 的 out-of-distribution (OOD) 稳定性，**彻底缓解灾难性遗忘**。
- **增强鲁棒性**：在多 token 并行解码下表现更稳定。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **数学推理**：`MixChain-Z-PRM12K`（12K queries）
- **代码生成**：`Ling-Coder-SFT`（18K queries）
- **自蒸馏数据生成**：使用 base model（Dream 或 LLaDA）在其默认解码策略下生成约 3.8K–5.1K 条有效 self-distilled 轨迹。

### 实验设置和评估指标
#### 模型
- **Base Models**：
  - `Dream-7B-Instruct`
  - `LLaDA-8B-Instruct`

#### 训练细节
- 使用 **LoRA** 进行参数高效微调（rank=16, α=16）
- 优化器：AdamW，学习率 2e-5，cosine 衰减，warm-up 50 步
- 批大小：每设备 4，共 8 GPUs
- 训练轮数：5 epochs
- TABOM 参数：窗口大小 $W=32$，margin $\gamma \in \{0.1,0.2,0.3\}$，ranking loss weight $\lambda \in \{1,2\}$

#### 评估任务
| 类别 | 任务 | 描述 |
|------|------|------|
| 数学推理 | GSM8K, MATH500 | 多步数学问题求解 |
| 代码生成 | HumanEval, MBPP | 代码补全与执行正确率 |
| 指令遵循 | IFEval | 验证指令遵循能力 |

> 对每个模型同时评估 **in-domain** 和 **out-of-distribution (OOD)** 性能。

### 基线方法对比
| 基线 | 简介 |
|------|------|
| **No-SFT** | 原始 DLM，无任何微调 |
| **SFT-GT** | 使用离线真实标签（offline ground-truth）进行标准 SFT |
| **SFT-SD** | 使用 self-distilled 轨迹进行标准 SFT |
| **dInfer** | 学习从后期状态跳跃到早期状态的压缩转换 |
| **T3D** | 结合路径一致性加权的判别性优化 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

#### 在 Dream-7B-Instruct 上的表现（部分摘录）

| Method | HumanEval ↑ | GSM8K ↑ | MATH500 ↑ | IFEval ↑ |
|--------|-------------|---------|-----------|----------|
| No-SFT | 52.66 | 81.41 | 39.80 | 56.56 |
| SFT-GT | **61.55 (+8.89)** | 52.33 (-29.08) | 32.40 (-7.40) | 46.21 (-10.35) |
| SFT-SD | 53.66 (+1.00) | 81.81 (+0.40) | 41.60 (+1.80) | 57.10 (+0.54) |
| **TABOM (Ours)** | **60.36 (+7.70)** | **81.73 (+0.32)** | **42.40 (+2.60)** | 55.45 (-1.11) |

> ✅ TABOM 在 **in-domain (HumanEval)** 接近最优，同时 **完全避免 OOD 性能崩溃**

#### 在 LLaDA-8B-Instruct 上的表现（数学推理）

| Method | GSM8K ↑ | HumanEval ↑ | MATH500 ↑ |
|--------|---------|-------------|-----------|
| No-SFT | 76.12 | 36.01 | 36.20 |
| SFT-GT | 74.29 (-1.83) | 31.09 (-4.92) | 35.50 (-0.70) |
| SFT-SD | 75.96 (-0.16) | 36.58 (+0.57) | 35.70 (-0.50) |
| **TABOM (Ours)** | **78.62 (+2.50)** | **40.30 (+4.29)** | **36.80 (+0.60)** |

> ✅ TABOM 实现 **全面正向增益**，在所有 in-domain 和 OOD 指标上均优于其他方法。

### 与基线方法的对比结果
- **vs SFT-GT**：TABOM 避免了灾难性遗忘，在 OOD 任务上平均提升超过 **+15 pts**。
- **vs SFT-SD**：TABOM 显著提升了 in-domain 性能（如 HumanEval +7.70 vs +1.00），说明其**真正实现了知识迁移**。
- **vs dInfer/T3D**：TABOM 在大多数任务上取得更高分数，表明其**对齐机制比单纯压缩更有效**。

### 消融实验结果（Table 5）
| 设置 | GSM8K | MATH500 | HumanEval |
|------|-------|---------|-----------|
| SFT-SD (Base) | 81.95 | 39.80 | 57.92 |
| + Global Ranking | 83.10 | 40.20 | 57.50 |
| **+ Local Ranking (W=32, TABOM)** | **84.31** | **41.10** | **58.54** |

> 🔍 发现：
> - 单纯使用 trajectory-aware masking 提升有限；
> - 加入 pairwise ranking 后性能显著上升；
> - **局部窗口（W=32）优于全局窗口**，过大窗口（如 W=256）会引入噪声导致性能骤降（GSM8K ↓至 75.36）。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **self-distilled 轨迹具有更低的优化壁垒**（lower optimization barrier），但**直接用 NELBO 微调只能带来边际收益**。
2. **训练-推理不一致是根本瓶颈**：标准 SFT 引入“均匀重建偏置”，破坏了推理所需的“由易到难”结构。
3. **TABOM 成功对齐了这一结构**：通过 Boltzmann 建模 + pairwise ranking，使模型在解码时能更好地区分“简单”和“困难”token。
4. **引入 Trajectory Discrimination Score (TDS)** 作为诊断工具，定量验证 TABOM 构建了更具区分性的熵景观（entropy landscape）：
   - 在 Dream 上，MBPP 的 TDS 从 SFT-SD 的 0.138 提升至 TABOM 的 **0.929**（↑646%）。
5. **TABOM 实现了“鱼与熊掌兼得”**：
   - in-domain 性能接近甚至超越 SFT-GT，
   - 完全避免灾难性遗忘，
   - 支持高效并行解码。

### 方法的局限性
- 当前方法依赖于高质量的 self-distilled 轨迹，若 base model 初始能力弱，可能限制性能上限。
- 排序损失的设计假设 token 难度差异在局部窗口内可比，极端长程依赖场景可能仍受限。
- 虽然未聚焦加速，但当前实现未进一步压缩步数，**推理速度未优于 dInfer/T3D**。

### 未来工作方向
- 将 TABOM 与 step compression 方法结合，实现**性能与效率双提升**。
- 探索更复杂的能量函数设计，以建模 token 间的交互关系。
- 将该范式扩展至其他生成模型（如 AR + Diffusion hybrid）。
- 研究如何动态调整 ranking margin 或 window size 以适应不同任务复杂度。

---

> 💡 **一句话总结**：  
> TABOM 首次系统性地将 self-distilled 轨迹从“推理捷径”转变为“知识载体”，通过 **Boltzmann Modeling + Pairwise Ranking** 实现训练与推理的动态对齐，在不牺牲泛化能力的前提下显著提升 DLM 的生成质量，为下一代高效且强大的扩散语言模型训练提供了新范式。

</details>

---

### 2. [AB-Sparse: Sparse Attention with Adaptive Block Size for Accurate and Efficient Long-Context Inference](https://arxiv.org/abs/2605.12110)

**Authors**: Di Liu, Ruitian Wang, Chen Chen, Mingliang Gong, Yongjie Yuan, Han Zhao, Yu Feng, Quan Chen, Minyi Guo  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.12110v1  

#### Abstract
As large language models scale to longer contexts, loading the growing KV cache during attention computation becomes a critical bottleneck. Previous work has shown that attention computation is dominated by a small subset of tokens. This motivates block sparse attention methods that partition the KV...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AB-Sparse: Sparse Attention with Adaptive Block Size for Accurate and Efficient Long-Context Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Long-Context Inference** 场景下，随着上下文长度增长，KV Cache 的加载成为推理过程中的内存瓶颈。现有的 **block sparse attention** 方法通过将 KV Cache 分块并仅加载重要性高的 Top-K 块来缓解该问题。然而，这些方法对所有 **attention head** 使用统一的固定块大小（uniform block size），忽略了不同 attention head 对块粒度敏感性的显著差异，导致精度损失。

### 🚀 提出的新方法：AB-Sparse
AB-Sparse 是一个**无需训练**、算法与系统协同设计的框架，提出以下三项核心技术：

1. **Adaptive Block Size Allocation（自适应块大小分配）**  
   - 观察到不同 attention head 对块大小的敏感性具有输入无关性和稳定性。
   - 通过轻量级校准（calibration-driven profiling）在部署前为每个 head 分配最优块大小：敏感 head 使用更小块以保留细粒度信息，不敏感 head 使用更大块以减少开销。

2. **Lossless Centroid Quantization（无损质心量化）**  
   - 发现 block centroids 仅用于排序选择，对精度不敏感。
   - 采用 **INT4 asymmetric per-channel quantization** 对 centroids 进行压缩，在几乎无精度损失的前提下大幅降低内存占用。

3. **Custom GPU Kernels（定制化 GPU 内核）**  
   - 设计三个专用内核解决异构块大小带来的执行难题：
     - **Fused query-centroid estimation**：支持变长 centroid 批处理，避免填充。
     - **Batched Top-K selection**：按 head 动态选择 Top-K 块。
     - **Heterogeneous paged attention**：利用逻辑块到物理页的 stride 映射机制，兼容标准 paged KV cache 管理。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **Accuracy** | 显著优于 Quest 和 ArkVale 等固定块大小方法，最高提升达 **5.43%** |
| **Throughput** | 不牺牲吞吐量，甚至因 INT4 量化在长序列上延迟更低 |
| **Compatibility** | 兼容现有 paged attention 架构，无需重构 KV layout |
| **Plug-and-Play** | 可作为 drop-in replacement 应用于多种 block sparse 方法（如 Quest / ArkVale） |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **RULER [28]**：合成基准测试，涵盖检索、多跳推理、聚合、问答等 13 项任务，评估模型在 **16K–96K 上下文长度**下的能力。
- **LongBench [30]**：真实世界长文本理解任务，覆盖单文档 QA、多文档 QA、摘要、少样本学习、代码补全等六类任务。
- **Calibration Set**：使用 **Wikipedia [27]** 子集（50 个样本）进行离线校准以确定各 head 的块大小。

### ⚙️ 实验设置
- **模型**：
  - Llama-3.1-8B
  - Qwen3-8B
  - Qwen3-32B
- **硬件平台**：
  - NVIDIA A100-80GB
  - NVIDIA H800-80GB
- **上下文长度范围**：从 64K 到 256K tokens
- **KV 缓存预算**：固定为总 KV 的 **4%**
- **平均块大小**：保持为 **32**，确保与 baseline 公平比较

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 在 RULER 和 LongBench 上的平均得分 |
| **Decoding Latency (ms)** | 解码阶段 attention 计算耗时 |
| **Throughput (tokens/s)** | 单位时间内生成的 token 数量 |
| **Top-K Page Recall** | 衡量被选中的 Top-K 块中是否包含高注意力分数 token 的比例 |
| **Pass@4** | 长生成任务中四次采样至少一次成功的概率 |

### 🆚 基线方法
- **Full Attention**：完整 attention 作为上限参考
- **Quest [16]**：基于 min-max pooling 的 block importance 估计
- **ArkVale [17]**：使用 bounding-volume centroids 提升块表示质量
- AB-Sparse 分别构建于 Quest 和 ArkVale 之上（即 AB-Sparse-Quest / AB-Sparse-ArkVale）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

#### 在 RULER 上的表现（Llama-3.1-8B）
| Method | Avg. Accuracy (%) |
|--------|-------------------|
| Full Attention | 88.71 |
| Quest | 78.23 |
| **AB-Sparse-Quest** | **81.74 (+3.51)** |
| ArkVale | 80.60 |
| **AB-Sparse-ArkVale** | **84.40 (+3.80)** |

> → 最高提升 **5.43%**（Qwen3-32B 上）

#### 在 LongBench 上的表现（Qwen3-32B）
| Method | Avg. Accuracy (%) |
|--------|-------------------|
| Quest | 27.12 |
| **AB-Sparse-Quest** | **29.74 (+2.62)** |
| ArkVale | 28.25 |
| **AB-Sparse-ArkVale** | **29.77 (+1.52)** |

> → 一致优于所有 baseline

---

### 🔁 长生成任务表现（Table 2）
在 **AIME24**, **AMC23**, **MATH500** 上使用 Qwen3-8B 测试长输出生成（max gen len = 32K）：

| Method | Pass@4 (%) |
|--------|------------|
| Quest | 47.2 |
| **AB-Sparse-Quest** | **53.1 (+5.9)** |

> → 验证了方法在 **long-output** 场景同样有效

---

### 📉 效率表现（Figure 10）
- **Latency**：AB-Sparse 与 Quest 相当或更优，尤其在长上下文（256K）时由于 INT4 量化减少 memory traffic，延迟更低。
- **Throughput Scaling**（Figure 11）：
  - Batch size=1：吞吐相近
  - Batch size=4：AB-Sparse 达到 Quest 的 **1.59× 吞吐**

---

### 🔍 消融实验结果（Ablation Study）

#### （1）Centroid Quantization 影响（Figure 13）
| Precision | RULER Accuracy |
|----------|----------------|
| BF16（未量化） | ~84.4 |
| INT8 | ≈ BF16 |
| **INT4** | **≈ BF16（无明显下降）** |
| INT2 | 明显下降 |

> ✅ 结论：**INT4 asymmetric per-channel quantization 实现近乎无损压缩**

#### （2）Custom Kernels 效果（Figure 14）
| Kernel | Speedup vs Naive |
|-------|------------------|
| Estimation | up to **5.6×** |
| Top-K Selection | up to **9.4×** |
| Attention | up to **3.1×** |

> ✅ 定制内核显著提升执行效率，尤其在异构块大小场景下

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Attention heads 对 block granularity 的敏感性高度异质（heterogeneous）**  
   - 有些 head 必须用小块才能保留关键 token（敏感 head）
   - 有些 head 即使大块也能保持高 recall（不敏感 head）
   - 固定块大小无法兼顾两者，造成精度-效率权衡失衡

2. **Per-head block size sensitivity is stable across inputs**  
   - 校准只需一次即可泛化到不同任务（Figure 6 显示在 RULER 多任务上均有效）

3. **Adaptive allocation + INT4 quantization + custom kernels = win-win**  
   - 在不增加 KV budget 的前提下，实现精度提升且维持甚至优化吞吐

4. **AB-Sparse is plug-and-play and generalizable**  
   - 可无缝集成至 Quest / ArkVale 等主流 block sparse 方法
   - 支持不同架构（Llama / Qwen）、不同规模（8B / 32B）模型

---

### ⚠️ 局限性
- **依赖 calibration set**：虽然仅需少量样本，但仍需额外步骤确定 block size 分配。
- **假设 centroids 可量化**：若 future 方法将 centroids 直接用于 attention computation，则可能不再适用。
- **当前仅支持静态分配**：运行时动态调整 block size 尚未支持。

---

### 🔮 未来工作方向
1. **Dynamic Adaptive Block Size**：探索在推理过程中根据输入内容动态调整 block size。
2. **Cross-layer Coordination**：联合优化多个 layer 的 block size 分配策略。
3. **Integration with Other Sparsity Paradigms**：结合 token-based 或 semantic-based 方法进一步提升稀疏效率。
4. **Extension to Training Phase**：研究如何在训练中引入 adaptive block structure 以更好适配 inference。

---

## 总结
AB-Sparse 通过揭示并利用 **attention head 级别的 block size 敏感性异质性**，提出了首个支持 **adaptive block size** 的高效 sparse attention 框架。其三大组件——轻量校准、无损量化、定制内核——共同实现了 **精度显著提升（最高 +5.43%）而不牺牲吞吐**，为 long-context LLM 推理提供了实用且通用的解决方案。

</details>

---

### 3. [Generalization Bounds of Emergent Communications for Agentic AI Networking](https://arxiv.org/abs/2605.08613)

**Authors**: Yong Xiao, Jingxuan Chai, Guangming Shi, Ping Zhang  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.08613v1  

#### Abstract
The evolution of 6G networking toward agentic AI networking (AgentNet) systems requires a shift from traditional data pipelines to task-aware, agentic AI-native communication solutions. Emergent communication, a novel communication paradigm in which autonomous agents learn their own signaling protoc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Generalization Bounds of Emergent Communications for Agentic AI Networking

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统通信网络依赖**预定义、刚性的协议架构**（如固定帧结构和分层设计），难以适应 Agentic AI Networking（AgentNet）系统中动态、多模态、任务驱动的智能体协作需求。现有 **emergent communication**（EC）框架虽然能自主学习通信协议，但普遍存在以下问题：
- 忽视物理层约束（如带宽、计算复杂度）
- 缺乏信息论基础，理论保障不足
- 决策模块与通信模块分离训练，导致冗余和不一致性
- 泛化能力弱，在未见环境状态下的表现不稳定

### 🚀 提出的新方法与创新思路
本文提出了一种**基于分布式信息瓶颈**（Distributed Information Bottleneck, DIB）的新型 emergent communication 框架，其核心创新包括：

1. **联合优化损失函数**（Joint Loss Function）  
   首次将 agent 的 **decision-making 函数** 和 **emergent communication signaling** 统一到一个端到端可微的优化目标中，避免了模块化设计带来的协调开销。

2. **基于 Multi-Agent Multi-Task DIB 的信息论建模**  
   引入两个关键互信息项作为正则项：
   - $ I(Y_k; C_{-k,k}) $：最大化通信信号中的**任务相关性信息**
   - $ I(S_k; C_{k,-k}) $：最小化消息长度（即 **Minimum Description Length, MDL**），控制表示复杂度

3. **理论泛化界分析**（Generalization Bounds）  
   推导了在去中心化推理场景下，emergent communication 协议的**泛化误差上界**，基于 Rényi divergence 和 sub-Gaussian 假设，为协议稳定性提供数学保证。

4. **理论与实践结合验证**  
   在真实硬件原型（UE + gNodeB + 5GC）上实现并测试，验证了框架在实际网络环境中的有效性。

### 🔍 相比现有方法的优势
| 方面 | 本文方法 | 现有主流方法（如 EC-SOTA） |
|------|----------|-----------------------------|
| 架构设计 | 联合训练决策与通信模型 | 模块化设计（先学表示再通信） |
| 理论支撑 | 基于 DIB 的信息论基础，具备泛化界 | 多为启发式、实验驱动 |
| 通信效率 | 自动压缩任务无关信息，降低 MDL | 可能耗费更多带宽传输冗余特征 |
| 泛化能力 | 显著更优，收敛快且误差低 | 容易过拟合训练噪声 |
| 实用性 | 支持资源受限 AgentNet 系统 | 忽视物理层限制 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **应用层数据**：采用公开的真实智能手机流量数据集 [15]，涵盖五类典型移动应用：
  - Live Streaming（直播）
  - Video Conferencing（视频会议）
  - Mobile Gaming（手游）
  - Web Browsing
  - Social Media
- **物理层输入**：将上述应用流量注入自研的软硬件融合 5G RAN + Core 网络原型，生成真实的无线信道反馈与资源配置需求。

### ⚙️ 实验设置
- **Agents 设计**：
  - **Application-layer Agent**：观察高层语义流量模式，预测 QoE 需求，并发送 emergent signal。
  - **Physical-layer Agent**：接收信号后动态调整波形、调制编码策略、资源块分配等。
- **通信机制**：通过深度神经网络学习 emergent communication protocol（无固定语法）。
- **训练方式**：去中心化训练，每个 agent 仅基于本地观测 $ s_k $ 和接收到的消息 $ c_{-k,k} $ 做决策。
- **评估阶段**：在**未见过的应用流量模式和信道状态组合**下进行 inference，测试泛化性能。

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 应用层 agent 正确识别任务意图的比例 |
| **Generalization Error** | 训练损失 vs. 推理损失之间的差距（越小越好） |
| **Convergence Speed** | 达到稳定性能所需的迭代次数 |
| **Error Floor** | 收敛后的最低泛化误差水平 |

### 🆚 基线方法
- **EC-SOTA** [16]：当前最先进的 emergent communication 基线，使用 autoencoder 学习 latent representation 并独立训练通信模块。
  - 特点：两阶段训练，通信与任务解耦。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Fig. 2 与 Fig. 3）

#### （1）准确率对比（Fig. 2）
- 在约 10,000 次迭代后：
  - **本文方法**：应用层 agent 准确率达到 **~88%**
  - **EC-SOTA**：准确率约为 **~75%**
  → 提升超过 **13个百分点**

#### （2）泛化误差对比（Fig. 3）
| 方法 | 泛化误差峰值 | 收敛速度 | 最终误差 floor |
|------|---------------|-----------|------------------|
| EC-SOTA | ~28%         | 缓慢（>8k iter） | ~18%            |
| **本文方法** | **~15%**     | **快速（<4k iter）** | **~6%**         |

- **优势明显**：不仅初始泛化误差更低，而且更快收敛至稳定的低误差平台。
- 尤其在高带宽、低延迟敏感业务（如 Live Streaming 和 Mobile Gaming）中表现更为稳健。

#### （3）消融实验（隐含分析，文中未明确列出表格但可推断）
从理论分析和实验趋势可知：
- 若移除 **MDL 正则项**（$ I(S_k; C_{k,-k}) $）：会导致信息过载，泛化误差上升，出现“informational collapse”现象。
- 若移除 **task-relevance term**（$ I(Y_k; C_{-k,k}) $）：agent 无法聚焦关键语义，accuracy 下降显著。
- 联合损失函数的设计是性能提升的关键驱动力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **emergent communication 可以被信息论严格建模**  
   利用 **multi-agent multi-task DIB** 成功量化了任务相关性与通信复杂度之间的根本权衡。

2. **联合优化显著提升泛化能力**  
   将 decision-making 与 communication signaling 融合在一个 loss 中，有效缓解了去中心化学习中的非平稳性（non-stationarity）问题。

3. **MDL 正则化增强了鲁棒性**  
   最小描述长度原则抑制了对环境噪声的过拟合，使协议更具普适性和抗干扰能力。

4. **理论泛化界具有指导意义**  
   推导出的 generalization bound 形式合理，且与实证结果一致，证明了该框架具备良好的结构性稳定性。

### ⚠️ 局限性
- 当前实验集中在双 agent 场景（应用层 + 物理层），扩展至大规模 multi-agent 网络仍需进一步研究。
- 所提 bound 依赖于 sub-Gaussian 和独立同分布假设，在高度动态或 adversarial 环境中可能不够紧致。
- 硬件原型尚未支持 full-stack 自主演化，部分参数仍需人工初始化。

### 🔮 未来工作方向
1. 扩展至 **multi-hop AgentNet topology**，研究路由与 emergent protocol 的协同演化。
2. 结合 **LLMs as Agents**，探索自然语言风格 emergent interface 的可控生成。
3. 引入 **causal representation learning** 进一步提升跨域迁移能力。
4. 推动标准化进程，构建面向 6G 的 **AI-native protocol stack**。

---

> **总结一句话**：  
> 本论文首次建立了 **具备信息论基础和泛化保证的 emergent communication 框架**，为构建高效、自适应、可扩展的 Agentic AI Networking 系统提供了坚实的理论与实践支撑，是迈向 6G AI-native 网络的重要一步。

</details>

---

### 4. [BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion](https://arxiv.org/abs/2605.11577)

**Authors**: Shaobin Zhuang, Yuang Ai, Jiaming Han, Xiaohui Li, Huaibo Huang, Xiangyu Yue, Xuefeng Hu, Kun Xu, Yali Wang, Hao Chen  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.11577v1  

#### Abstract
Autoregressive language models generate text one token at a time, yet natural language is inherently structured in multi-token units, including phrases, n-grams, and collocations that carry meaning jointly. This one-token bottleneck limits both the expressiveness of the model during pre-training and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统自回归语言模型（Autoregressive LLMs）采用“逐个token生成”的范式，即通过 `softmax` 输出层对词汇表中的下一个 token 进行分类预测。这种范式存在两个核心瓶颈：
- **表达能力受限**：将语言建模为一系列独立的类别选择，忽略了短语、n-gram等多词组合的联合语义结构。
- **推理效率低下**：生成过程本质上是串行的，限制了吞吐量。

尽管已有工作尝试通过 speculative decoding、multi-token prediction 或非自回归方法加速，但大多仍依赖于词汇级的 `softmax` 接口，未能从根本上改变输出空间的几何结构。

---

### **提出了什么新方法或新思路**
论文提出 **BitLM**，一种全新的语言模型架构，其核心思想是：
> **将 token 生成从“大词汇表上的分类任务”重构为“固定长度二进制码空间中的连续扩散去噪任务”。**

具体创新包括：

- **二进制 token 表示（Binary Token Interface）**  
  每个 token ID 被映射为一个固定长度（如 18-bit）的二进制码 $\phi(y_i) \in \{-1, +1\}^B$，所有 token 构成一个超立方体顶点集合。

- **基于扩散头的联合块生成（Diffusion Head for Joint Block Realization）**  
  使用轻量级 diffusion head 对未来多个 token 的二进制码进行并行去噪，实现 **multi-token joint realization**。

- **块因果注意力机制（Block-Causal Attention）**  
  在 backbone Transformer 中使用 block-causal mask，允许块内 token 全连接，跨块保持因果性，支持块级并行生成。

- **分离式架构设计（Dual-Path Architecture）**  
  - **Backbone**：负责上下文推理（contextual reasoning），保持标准的因果 Transformer 结构。
  - **Diffusion Head**：负责符号实现（symbolic realization），在二进制空间中迭代去噪出下一组 token。

这使得模型既能保留 autoregressive 模型的可靠性，又能天然支持并行生成。

---

### **相比现有方法的优势**
| 维度 | BitLM | 传统方法 |
|------|-------|----------|
| **输出接口** | 二进制空间去噪 | 大词汇表 softmax 分类 |
| **生成方式** | 块内并行去噪 | 单 token 顺序采样 |
| **解码机制** | 内生并行（native parallelism） | 后处理加速（post-hoc tricks） |
| **几何视角** | 连续二进制空间优化 | 离散 simplex 上决策 |

> ✅ **优势总结**：
> - 打破了“one-token-at-a-time”并非必要前提的认知；
> - 将生成视为结构化对象的逐步结晶（iterative commitment），而非孤立分类；
> - 实现更高效的训练与显著更快的推理速度；
> - 不牺牲 autoregressive 的因果归纳偏置。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **预训练（Pretraining）**：FineWeb 数据集的一个子集，共 **350B tokens**。
- **微调与评估（Fine-tuning & Evaluation）**：XSum 新闻摘要数据集（验证集用于测试）。

---

### **实验设置**
- **模型架构基础**：
  - Backbone：基于 **Qwen-3** 架构（类似 LLaMA 的 decoder-only Transformer）。
  - Diffusion Head：借鉴 **BitDance** 的轻量扩散网络结构。
- **关键参数**：
  - 二进制码长度 $B = 18$（覆盖常见 tokenizer 的 vocab size）。
  - 块大小 $m = 4$（每次生成 4 个 token）。
  - 预训练序列长度：16384 tokens（多样本打包）。
- **训练细节**：
  - 优化器：AdamW，lr=1e-4，$\beta_1=0.9, \beta_2=0.95$。
  - 训练时长：1 epoch。
- **推理设置**：
  - 去噪步数 $K = 15$。
  - 使用 ODE solver 和 **Classifier-Free Guidance (CFG)**，guidance scale = 9.0。

---

### **评估指标**
- 主要指标：**ROUGE-1, ROUGE-2, ROUGE-L**（标准化后报告）。
- 消融研究：分析不同去噪步数 $K$ 和 CFG 值的影响。

---

### **基线方法对比**
- **经典基线**：
  - ILead-3（取前3句）
  - PTGEN / PTGEN+COV（Pointer-Generator Network）
- **自身变体对比**：
  - BitLM w/ LM Head（即用 softmax 替换 diffusion head）
  - BitLM w/ Diffusion Head（本文方法）
  - 区分预训练（PT）与微调后（FT）版本

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（XSum 测试集）**

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|---------|---------|---------|
| ILead-3 | 16.30 | 1.60 | 11.95 |
| PTGEN (See et al., 2017) | 29.70 | 9.21 | 23.24 |
| BitLM 8B w/ LM Head (FT) | 23.20 | 4.45 | 18.04 |
| **BitLM 8B w/ Diff. Head (FT)** | **26.05** | **6.44** | **20.12** |

> 🔍 微调后的 BitLM 在 ROUGE 指标上明显优于其 softmax 版本，且接近但尚未超越最强指针网络基线。

---

### **与基线方法的对比结果**
- BitLM（diffusion head）比同等结构下使用 softmax head 的版本提升显著：
  - ROUGE-1 提升约 **+2.85 pts**
  - ROUGE-2 提升约 **+1.99 pts**
- 表明 **binary diffusion 输出层本身具有更强的生成潜力**，尤其在捕捉局部结构方面。
- 虽未超越 SOTA 摘要模型（如 PTGEN），但证明了该新范式的可行性。

---

### **消融实验结果**
#### （1）去噪步数 $K$ 与 CFG 影响（Fig. 4）
- 当 $K=15$, CFG=9 时达到最佳性能。
- 更少的去噪步导致质量下降；过多步数收益饱和。
- CFG 值过低则控制力弱，过高易产生重复或失真。

#### （2）模型可扩展性（Fig. 3）
- 成功训练了 **0.6B, 1.7B, 4B, 8B** 四种规模的 BitLM。
- 随着模型增大，**预训练 loss 持续下降**，显示良好的 scalability。
- 说明该架构可在大规模下稳定训练。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **“逐 token 分类”不是语言建模的唯一路径**  
   使用二进制码 + 扩散去噪可以有效替代传统的 `softmax` 输出层。

2. ✅ **block-causal 并行生成可以成为一类原生模式（first-class generation mode）**  
   并非对 autoregressive 的近似，而是由输出空间几何决定的新范式。

3. ✅ **分离“推理”与“实现”模块是可行的设计原则**  
   Backbone 专注上下文建模，diffusion head 专注符号落地，职责分明。

4. ✅ **bitwise diffusion 支持高效且灵活的生成控制**  
   如通过调整 block size $m$ 控制并行度，适应不同延迟需求。

---

### **方法的局限性**
- 当前在 XSum 上的表现仍 **低于最先进的摘要模型**（如 PTGEN），表明在精确词汇实现和长程一致性上仍有差距。
- 二进制码为固定映射，未学习语义编码，可能限制表示能力。
- 去噪过程引入额外计算开销，需权衡速度与质量。
- 实验集中在英文文本，多语言和多模态扩展尚待验证。

---

### **未来工作方向**
- 探索 **learned binary codes** 替代固定编码，增强语义表达。
- 设计 **adaptive block size** 机制，动态调整并行粒度。
- 引入 **hybrid softmax-binary 架构**，结合两者优势。
- 开发更适合 diffusion-based generation 的 **fine-tuning 策略**（如强化学习对齐）。
- 扩展至 **multimodal generation**（已受 BitDance 和 UniWeTok 启发）。

---

## **总结**
BitLM 提出了一种根本性的语言建模新视角：  
> **将 token 生成从“高维分类”转变为“低维连续空间中的结构化去噪”**。

它不仅挑战了 `softmax` 作为默认输出层的历史地位，也为下一代高效、并行、可控的语言模型架构提供了新的设计维度——**symbolic output space 的几何结构**。

虽然当前性能尚未全面超越主流 LLM，但其实验验证了这一路径的可行性与潜力，有望启发更多关于“如何更好地连接连续隐空间与离散语言符号”的研究。

</details>

---

### 5. [Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference](https://arxiv.org/abs/2605.11581)

**Authors**: Wenxin Dong, Mingqing Hu, Guanghui Yu, Qiang Fu, Peng Xu, Hui Xu, Yue Xing, Xuewu Jiao, Shuanglong Li, Lin Liu  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.11581v1  

#### Abstract
When large language models (LLMs) serve real-time inference in commercial online advertising systems, end-to-end latency must be strictly bounded to the millisecond range. Yet every token generated during the decode phase triggers thousands of kernel launches, and kernel launch overhead alone can ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在商业在线广告系统等**低延迟、高并发**场景中，大语言模型（LLM）推理面临两大瓶颈：
- **Kernel launch overhead**：解码阶段每生成一个token会触发数千次 kernel 启动，启动开销占端到端延迟的 **14.6%**。
- **HBM round-trip latency**：传统算子间通过全局内存（HBM）交换中间结果，造成频繁访存。

现有 MegaKernel 方案存在矛盾：
- 手工调优方案（如 Stanford MegaKernel）性能强但**缺乏可移植性**，依赖特定架构（如 Hopper/Blackwell），难以部署到资源受限的 **NVIDIA Ada 架构（如 L20 GPU）**。
- 自动编译方案（如 Mirage MPK）引入运行时分支判断，导致指令发射效率下降，在超低延迟场景不可接受。

### 提出的新方法与创新思路
作者提出 **Ada-MK**，一种面向 Ada 架构优化的自适应 MegaKernel 框架，其三大核心贡献如下：

#### （1）自适应共享内存管理（Adaptive Shared Memory Management）
- 建立三维约束模型：综合考虑 **硬件规格、模型架构、动态工作负载** 进行共享内存分配。
- 引入 **K-dimension 细粒度分裂**：将计算块沿 K 维度切分为子块，每次仅加载所需权重子块，**峰值共享内存需求降低 50%**。
- 实现跨算子页面复用：
  - **Activation-weight page reuse**：激活数据加载至寄存器后，释放共享内存用于权重预取。
  - **Activation-output page reuse**：输出写回前复用激活页存储 MMA 结果。

> ✅ 显著缓解 Ada 架构仅 128KB 共享内存（H100 为 227KB）带来的优化空间压缩问题。

#### （2）基于 MLIR 的细粒度 DAG 离线搜索（Fine-grained DAG-based Automatic Search）
- 利用 **MLIR Lowering** 将高级 IR 分解为 PTX 级细粒度依赖图（DAG），精确建模数据流与资源竞争。
- 在离线阶段进行 **DAG 遍历与路径搜索**，确定最优执行轨迹，并将其固化为静态代码。
- 完全消除运行时动态决策（如 if-else 分支），避免指令流水线气泡。

> ✅ 相比 Ansor、Pruner 等传统 autotuning 框架，能捕捉更细粒度并行机会，尤其适用于不规则算子链。

#### （3）异构混合推理引擎（Heterogeneous Hybrid Inference Engine）
- 将 MegaKernel 作为插件嵌入 **TensorRT-LLM**，构建“双模”推理流程：
  - **Prefill 阶段**：使用 TensorRT-LLM 原生融合算子，发挥其大规模并行优势。
  - **Decode 阶段**：切换至 MegaKernel 引擎，实现单 kernel 持久化执行，消除 launch 开销。
- 支持零成本复用 TensorRT-LLM 已有功能（如 prefix-tree decoding、generation-discrimination）。

> ✅ 首次实现 MegaKernel 在工业级生产系统的落地应用。

### 相比现有方法的优势
| 方法 | 可移植性 | 运行时开销 | 支持模型广度 | 是否支持 Prefill |
|------|--------|------------|----------------|------------------|
| Stanford MegaKernel | ❌ 极差（硬编码） | ✅ 无分支 | ❌ 仅 Llama-1B | ❌ 不支持 |
| Mirage MPK | ✅ 自动化 | ❌ 引入 if-else 分支 | ✅ 较好 | ❌ 有限支持 |
| **Ada-MK (本文)** | ✅ 跨模型通用 | ✅ **完全静态路径** | ✅ 支持 Qwen 系列 | ✅ 支持 |

---

## 2. 核心实验方法和设置

### 数据集
- **固定短序列任务**：`input=64`, `output=12`（模拟低延迟短文本生成）
- **真实任务数据集**：
  - **CSL dataset**：中文科学文献摘要，上下文长度 ~200–1000 tokens
  - **Human-eval dataset**：代码生成任务，评估在编程场景下的表现

### 实验设置
- **硬件平台**：单台服务器，配备 **NVIDIA L20 GPU**（Ada 架构，48GB GDDR6，SMEM=128KB）
- **软件环境**：
  - CUDA 12.2, Driver 535.161.07
  - Docker 隔离运行，确保公平比较
- **测试模型**：
  - Qwen3-1.7B-GPTQ-W4A16
  - Qwen2.5-1.5B-GPTQ-W4A16
- **批处理大小（Batch Size）**：1, 2, 4, 8, 16
- **评估模式**：离线批量推理（offline batch mode），控制并发一致性

### 评估指标
- **Generation Throughput (tokens/s)**：越高越好
- 对比基线：
  - **vLLM**（v0.19.0）：高效 KV Cache 管理，高吞吐代表
  - **SGLang**（v0.5.10）：结构化生成高性能服务框架
  - **vanilla TensorRT-LLM**（v1.1.0rc5）：NVIDIA 官方基准
  - **Ada-MK**（本文方法）：TRT-LLM + MegaKernel 插件

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 场景 | 模型 | Batch Size | Ada-MK 吞吐 (tokens/s) | 提升幅度（vs TRT-LLM） |
|------|------|------------|--------------------------|-------------------------|
| 固定短序列 | Qwen3-1.7B | 1 | ~3.5K | **+23.6%** |
| 固定短序列 | Qwen2.5-1.5B | 1 | ~2.5K | +15.6% |
| CSL 数据集 | Qwen3-1.7B | 1–8 | 最高 | +4.0% ~ +19.5% |
| CSL 数据集 | Qwen3-1.7B | 16 | 第二 | 落后 vLLM 3.5% |
| Human-eval | Qwen3-1.7B | 16 | **最高** | **+19.5% vs TRT-LLM** |

### 与基线方法的对比结果
- **相比 vanilla TensorRT-LLM**：
  - 所有场景下均取得正向增益，提升范围 **4.0% ~ 23.6%**
  - 在小批量（BS=1）短序列场景中收益最大（达 23.6%）
- **相比 vLLM 和 SGLang**：
  - 在 **小批量 + 短序列** 场景中优势显著：
    - BS=1 下对 vLLM 提升高达 **50.2%**
    - 对 SGLang 提升达 **71.9%**
  - 在 **大批量 + 长上下文** 场景中优势收窄：
    - CSL @ BS=16：vLLM 表现最佳，Ada-MK 落后 3.5%
    - Human-eval @ BS=16：Ada-MK 仍保持领先（+1.6% vs vLLM）

### 消融实验与分析（隐含于文中）
虽然未明确列出消融表，但从设计分析中可推断各模块贡献：
- **K-dimension splitting + page reuse**：共享内存峰值降低 50%，使 pipeline stage 从 2 提升至 4，有效隐藏访存延迟。
- **DAG-based search**：相比原始 MegaKernel 实现，在 Decode 阶段带来约 **30% 性能提升**，主要来自：
  - 消除伪依赖（pseudo-dependency）
  - 实现跨路径流式 Reduce（如 SwiGLU 中 Up/Gate 并行）
  - 更优的 warp 角色分配（Consumer warps 从 16 减至 8，减少角色间吞吐失配）
- **Hybrid Engine 设计**：兼顾 Prefill 高吞吐与 Decode 低延迟，避免全 MegaKernel 化带来的工程迁移成本。

---

## 4. 关键结论和发现

### 主要发现
1. **MegaKernel 可成功适配资源受限的 Ada 架构**：
   - 通过自适应共享内存管理和细粒度 DAG 搜索，克服了 Ada 架构缺乏 TMA 支持、共享内存小等限制。
2. **“离线搜索 + 运行时固化”是低延迟场景的关键范式**：
   - 在部署配置固定的前提下，最优执行路径唯一，应将动态决策提前至编译期完成。
3. **Ada-MK 在小批量、短序列推理中具有绝对优势**：
   - 特别适合 **在线广告推荐、实时对话、交互式生成** 等毫秒级响应需求场景。
4. **异构混合架构更具实用价值**：
   - 不追求全栈 MegaKernel 化，而是结合 TensorRT-LLM 成熟生态，实现平滑集成与快速落地。

### 方法的局限性
- **对 Prefill 阶段优化有限**：因 Prefill 是 compute-bound，MegaKernel 的 I/O 重叠优势不明显。
- **在高并发长上下文场景下系统级调度劣势显现**：vLLM/SGLang 在 KV Cache 管理、请求调度方面具备更强的系统级扩展能力。
- **当前仅验证于中小规模模型（1.5B–1.7B）**：是否可扩展至百亿参数以上模型尚待验证。

### 未来工作方向
- 探索 Ada-MK 向更大模型规模的迁移与适配。
- 向下一代 **Blackwell 架构** 移植，利用其更强硬件支持进一步释放性能潜力。
- 扩展支持更多模型家族（Beyond Qwen/Llama）与量化格式（如 FP8、W2A16）。

---

> ✅ **总结一句话**：  
> **Ada-MK 是首个在工业级商业系统中成功部署的 MegaKernel 方案，它通过“三维共享内存建模 + DAG 级离线搜索 + 异构混合执行”，实现了在资源受限 Ada GPU 上的低延迟、高效率 LLM 推理，尤其在小批量短序列场景中大幅超越主流框架。**

</details>

---

### 6. [Training-Inference Consistent Segmented Execution for Long-Context LLMs](https://arxiv.org/abs/2605.11744)

**Authors**: Xianpeng Shang, Jiang Li, Zehua Duo, Qianyi Cai, Xiangdong Su  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.11744v1  

#### Abstract
Transformer-based large language models face severe scalability challenges in long-context generation due to the computational and memory costs of full-context attention. Under practical computation and memory constraints, many inference-efficient long-context methods improve efficiency by adopting ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Training-Inference Consistent Segmented Execution for Long-Context LLMs*

---

## 1. 论文的主要贡献和创新点

### **解决的问题**
当前基于 Transformer 的大语言模型（LLMs）在处理长上下文（long-context）生成任务时面临严重的可扩展性挑战，其根本原因在于 **full-context attention** 的计算和内存开销随上下文长度呈二次增长。为缓解这一问题，许多推理阶段高效的方法（如 bounded-context、chunked attention）仅在 **inference 阶段**引入受限执行策略，而训练仍采用 full-context attention，导致 **training-inference mismatch**。

这种不一致性带来了两个关键问题：
- **执行语义不一致**：训练时模型可以访问完整历史，而推理时只能看到有限上下文。
- **状态转移语义不一致**：跨段状态传播机制在训练和推理中不同，影响模型稳定性与泛化能力。

### **提出的新方法**
本文提出一种 **训练-推理一致的分段执行框架（Training-Inference Consistent Segmented Execution）**，核心思想是将分段执行作为共享建模假设，而非仅限于推理优化。

#### 主要设计：
- 将输入序列划分为非重叠的 segments。
- 定义两种跨段输入接口：
  - **Carried KV state**：从上一段保留的固定大小 KV 缓冲区，是**唯一可微分的跨段状态**，用于训练和推理中的局部连续性建模。
  - **Retrieved KV prefix**：通过检索机制从前序段获取的历史 KV，以 **forward-only 方式** 提供远距离上下文支持，**不参与梯度传播**。

#### 关键技术机制：
- **Truncated Backpropagation Through Time (TBPTT)**：限制梯度仅沿最多 $ K $ 个 segment 的 carried KV 链回传，确保训练目标与推理行为对齐。
- **Head- and Layer-Sparse Long-Range Retrieval**：仅在部分 attention heads 和 layers 中启用 long-range retrieval，控制计算开销。

### **相比现有方法的优势**
| 维度 | 传统方法（如 MInference, StreamingLLM） | 本文方法 |
|------|----------------------------------------|---------|
| **Training-Inference Alignment** | ❌ 不一致（训练用 full attention，推理用受限模式） | ✅ 严格一致 |
| **梯度传播合理性** | 可能依赖推理时不可见的信息 | 仅学习推理时可用的状态路径 |
| **可扩展性** | 在极长上下文下内存仍高 | 显著降低 peak memory（如 128K 下约 6× 降低） |
| **理论保证** | 多为经验设计 | TBPTT 可精确计算 inference-consistent 目标的梯度 |

---

## 2. 核心实验方法和设置

### **使用的数据集**
- **PG19**：用于评估语言建模 perplexity 随上下文长度变化的趋势。
- **LongBench**（Bai et al., 2024）及其子集 **LongBench-E**：多任务长上下文理解基准，涵盖问答、摘要、代码、合成任务等。
- **RULER**（Hsieh et al., 2024）：专门用于测试长度外推能力的任务集，包括：
  - **CWE**（Common Words Extraction）
  - **FWE**（Frequent Words Extraction）

### **实验设置**
- **模型基础**：
  - 主要实验基于 **LLaMA2-7B-32K** 和 **LLaMA2-7B-80K**。
  - 补充实验验证于 **LLaMA3.1-8B-Instruct**。
- **分段参数**：
  - Segment 长度 $ S = 4096 $
  - Carried KV 长度 $ M = 512 $
  - Retrieved KV 长度 $ R = 512 $
  - TBPTT 截断深度 $ K = 1 $（默认），对应最大训练上下文 8K
- **训练配置**：
  - 微调语言建模目标，在 SlimPajama 数据集上训练 1000 步。
  - 所有 alignment-based 方法统一训练设置。

### **评估指标**
| 指标 | 含义 |
|------|------|
| **Perplexity (PPL)** | 语言建模质量，越低越好 |
| **Average Score on LongBench-E** | 多任务综合性能，越高越好 |
| **Prefill Latency (TTFT)** | 首 token 时间，衡量推理延迟 |
| **Peak GPU Memory** | prefill 阶段峰值显存占用，越低越好 |
| **Recall Accuracy on RULER** | 长度外推能力，越高越好 |

### **基线方法对比**
| 方法 | 类型 | 是否训练-推理一致 |
|------|------|------------------|
| **Vanilla Self-Attention** | 全注意力 | ❌ |
| **MInference** | 推理稀疏注意力 | ❌ |
| **StreamingLLM** | 滑动窗口 + sink tokens | ❌ |
| **DuoAttention** | 分离 streaming 与 retrieval heads | ❌ |
| **CCA (Core Context Aware)** | 压缩上下文训练 | ✅ |
| **Ours** | 分段执行 + TBPTT + retrieval | ✅ |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **语言建模 Perplexity（图4）**
- 在 PG19 上，随着 context length 增加到 64K（超出训练长度），本文方法表现出更平滑的增长趋势，无明显 spike。
- 相比之下，其他方法（尤其是 CCA）在超长上下文中出现显著波动或崩溃。

#### ✅ **下游任务性能（表1 & 表4）**
| 方法 | LongBench-E @32K (Avg) | LongBench-E @80K (Avg) | Standard LongBench @80K (Avg) |
|------|------------------------|------------------------|-------------------------------|
| Vanilla | 23.13 | 23.38 | 24.20 |
| StreamingLLM | 21.90 | 21.56 | 22.21 |
| CCA | 21.12 | 21.98 | — |
| DuoAttention | 23.00 | 22.94 | 23.53 |
| **Ours** | **23.24** | **24.17** | **25.36** |

> 📌 **结论**：本文方法在多个 backbone 和任务集上均取得最优平均得分，尤其在 summarization 和 multi-document QA 上提升显著。

#### ✅ **长度外推能力（表2, RULER）**
| 方法 | CWE Avg* (4K–32K) | FWE Avg* | CWE @64K | FWE @64K |
|------|--------------------|----------|----------|----------|
| Vanilla | 32.94 | 41.33 | – | – |
| StreamingLLM | 27.78 | 41.37 | – | – |
| CCA | 24.90 | 31.96 | – | – |
| **Ours** | **46.39** | **43.88** | **2.00** | **34.17** |

> 📌 **结论**：即使在 64K（远超训练长度）下，本文方法仍保持非零准确率，展现出更强的鲁棒性和外推能力。

#### ✅ **效率表现（图5 & 图6）**
- **Prefill Memory**：在 128K 上，相比 full attention + FlashAttention，**peak memory 降低约 6×**。
- **Latency-Memory Trade-off**：在 64K context 下，本文方法在较低内存下实现适中延迟，优于多数 baseline（如 MInference 内存高，StreamingLLM 延迟低但性能弱）。

---

### **消融实验结果（表3 & 附录G）**

#### 🔍 **训练-推理一致性的影响（表3）**
| 方法 | Avg Score |
|------|-----------|
| Aligned (TBPTT=1) | 24.17 |
| **Misaligned** | **11.91** ↓↓ |
| Aligned (TBPTT=2) | 24.07 |

> 💡 移除一致性导致性能腰斩，证明 alignment 至关重要；$ K=1 $ 已足够，更深截断无益。

#### 🔍 **局部状态容量影响（表7 & 表8）**
- 增加 carried KV 大小（0 → 512 → 1024）可小幅降低 PPL 并提升下游性能，但收益递减。
- 即使 $ M=0 $，只要保留 retrieval 通道，仍有一定性能，说明两者互补。

#### 🔍 **长程模块位置影响（表9 & 表10）**
- 插入更多 long-range layers（如 4 层 vs 0 层）显著提升 LongBench 性能（22.63 → 24.17），但对 PPL 几乎无影响。
> 💡 支持“long-range retrieval 主要增强 cross-segment reasoning”而非基础语言建模。

#### 🔍 **head grouping 策略（表11）**
- **Prior-based grouping**（基于先验选择 retrieval-prone heads）效果最好，优于 contiguous 或 interleaved 分组。
> 💡 表明 head 功能异质性存在，合理分配 long-range capacity 更有效。

---

## 4. 关键结论和发现

### **主要发现**
1. ✅ **训练-推理一致性至关重要**：不一致会导致严重性能退化，尤其是在超长上下文场景。
2. ✅ **TBPTT 可精确优化 inference-consistent 目标**：在严格受限的跨段递归下，截断反向传播不再是近似，而是精确梯度。
3. ✅ **分离“可微局部状态”与“前向远距检索”是高效且鲁棒的设计**：
   - Carried KV 支持稳定的状态延续；
   - Retrieved KV 提供额外证据而不破坏训练动态。
4. ✅ **方法具备强可扩展性**：在 128K context 下实现约 6× 内存压缩，显著优于 full attention。

### **方法的局限性**
- **依赖预定义 segment 划分**：未考虑动态或语义边界分割。
- **retrieval overhead 随历史增长**：虽然 retrieval prefix 固定，但检索池线性增长，可能成为瓶颈。
- **未探索更复杂的 memory management**（如压缩、淘汰），长期运行需进一步优化。
- **目前仅验证于 decoder-only 架构**，是否适用于 encoder-decoder 或 multimodal 模型尚待研究。

### **未来工作方向**
- 结合 **semantic-aware segmentation**（如按句号、主题切分）替代固定长度分段。
- 引入 **learned compression** 或 **eviction policy** 以控制 retrieval pool 规模。
- 探索 **adaptive K in TBPTT**，根据不同任务自动调整梯度回溯深度。
- 扩展至 **多模态长上下文建模**（如视频、文档理解）。
- 研究如何 **自动化识别 long-range heads**，减少人工先验依赖。

--- 

> 🏁 **总体评价**：该论文提出了一个理论上严谨、实践中高效的 long-context 建模范式，强调“execution semantics alignment”应作为系统设计的第一原则，为构建真正可扩展的 LLMs 提供了新视角。

</details>

---

### 7. [SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization](https://arxiv.org/abs/2605.12245)

**Authors**: Chengzhu Bao, Xianglong Yan, Zhiteng Li, Guangshuo Qin, Guanghua Yu, Yulun Zhang  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.12245v1  

#### Abstract
NVFP4 has recently emerged as an efficient 4-bit microscaling format for large language models (LLMs), offering superior numerical fidelity with native hardware support. However, existing methods often yield suboptimal performance due to inflexible scale selection and the coupled treatment of quanti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **NVFP4** 量化方法在处理大语言模型（LLMs）时存在两个关键瓶颈：
- **Scale Selection 不灵活**：当前方法通常采用启发式规则（如基于最大值的缩放）或有限的离散搜索来确定全局和块级（block-wise）scale，难以精确拟合LLM中复杂且不规则的权重分布。
- **Quantization 与 Dequantization Scale 耦合**：硬件限制要求块级scale以低精度（FP8, E4M3）存储，并同时用于量化和反量化过程。这种耦合导致scale本身的量化误差会传播到重建过程中，影响最终精度。

### 提出了什么新方法或新思路
作者提出 **SOAR**（Scale Optimization for Accurate Reconstruction），一种新的后训练量化（PTQ）框架，包含两大核心技术：

#### ✅ Closed-form Joint Scale Optimization (CJSO)
- **思想**：将全局scale（α）和块级scale（Δ）联合优化，目标是最小化重构误差 $ \|W - \hat{W}\|^2 $。
- **实现**：通过固定FP4量化分配 $ Q $，推导出关于 α 和 Δ 的闭式解析解（closed-form updates），实现高效的迭代优化。
- **优势**：相比启发式方法，能更准确地捕捉权重分布特性，提升重建质量。

#### ✅ Decoupled Scale Search (DSS)
- **思想**：打破传统中量化与反量化使用同一scale的限制，引入两个独立的scale：
  - **高精度量化scale（Δ_q）**：仅用于决定FP4映射（即rounding过程），无需存储。
  - **硬件约束反量化scale（Δ_a）**：必须为FP8(E4M3)格式，用于推理时的反量化。
- **实现**：对每个块进行局部联合搜索（joint discrete search），在候选空间中寻找使重构误差最小的一对 $(\Delta_q, \Delta_a)$。
- **优势**：缓解了因scale被强制量化为E4M3带来的精度损失，提升了量化灵活性。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **精度更高** | 在多个LLM上一致超越现有NVFP4方法（如4over6、RaZeR），接近甚至缩小与FP16的差距。 |
| **无额外开销** | 不增加模型大小或推理延迟——Δ_q 仅在训练时使用，不需存储；Δ_a 仍用标准FP8表示。 |
| **通用性强** | 可与GPTQ等校准方法结合，进一步提升性能；也可推广至MXFP4等其他微缩放格式。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准数据（Calibration Data）**：
  - WikiText-2
  - C4
  - 序列长度：2048 tokens
- **评估任务（Zero-shot Evaluation）**：
  - **常识推理**：WinoGrande, PIQA, HellaSwag
  - **科学问答**：ARC-Easy / ARC-Challenge
  - **综合知识**：MMLU
  - **数学推理**：GSM8K

### 实验设置和评估指标
| 设置项 | 内容 |
|--------|------|
| **量化配置** | W4A4（权重和激活均为4bit）、W4A16（仅权重量化） |
| **硬件模拟** | 使用PyTorch + HuggingFace Transformers，在NVIDIA A800 GPU上实现NVFP4行为 |
| **评估指标** | 
  - 零样本准确率（Zero-shot Accuracy）
  - 平均准确率（Avg.）
  - 困惑度（Perplexity ↓ on WikiText2/C4）
  - 重构MSE（Reconstruction MSE ↓）

### 基线方法对比
- **基础NVFP4**：原始NVFP4量化方案
- **4over6** (Cook et al., 2025)：自适应选择block scale范围（6或4）
- **RaZeR** (Chen et al., 2025)：利用冗余零位扩展数值覆盖范围
- **GPTQ**：经典PTQ方法，用于验证SOAR与其兼容性

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### 📊 在 Qwen3-8B 上的表现（W4A4）
| Method | Wino | PIQA | Hella | Arc-E | Arc-C | **Avg** |
|--------|------|------|-------|-------|-------|--------|
| FP16   | 67.64 | 77.58 | 74.86 | 80.89 | 56.66 | **71.53** |
| NVFP4  | 66.54 | 75.63 | 73.27 | 73.27 | 55.03 | 68.75 |
| 4over6 | 64.09 | 75.68 | 73.24 | 79.25 | 53.50 | 69.15 |
| RaZeR  | 67.72 | 76.22 | 73.58 | 78.41 | 54.69 | **70.12** |
| **SOAR** | **68.35** | **77.26** | **73.19** | **79.42** | **55.20** | **70.68** |

> 🔺 **SOAR 较 SOTA (RaZeR) 提升 +0.56 pts，较基础NVFP4提升 +1.93 pts**

#### 🧠 推理能力测试（MMLU/GSM8K）
| Model | Method | MMLU | GSM8K | Avg |
|-------|--------|------|-------|-----|
| Qwen3-8B | NVFP4 | 70.70 | 79.30 | 75.00 |
| Qwen3-8B | RaZeR | 71.31 | 82.41 | 76.86 |
| Qwen3-8B | **SOAR** | **71.47** | **82.56** | **77.02** |

> 🔺 在数学推理（GSM8K）上显著提升 **+0.15 pts**

### 与基线方法的对比结果
- **全面优于所有NVFP4 baseline**：在几乎所有模型（LLaMA3.1/3.2系列、Qwen3系列）和任务上，SOAR均取得最高平均准确率。
- **优于GPTQ组合版本**：当与GPTQ结合时，SOAR仍能带来额外增益（见Table 3），说明其优化是正交且互补的。
- **内存开销相同**：如图5所示，SOAR与标准NVFP4具有完全相同的模型大小，远小于FP16。

### 消融实验结果（Table 4）
在 LLaMA-3.2-3B-Instruct 上的消融研究：

| Method | Wiki2 PPL ↓ | C4 PPL ↓ | Avg Acc ↑ |
|--------|-------------|----------|-----------|
| NVFP4 | 11.98 | 15.53 | 65.02 |
| +CJSO | 12.04 | 15.53 | 65.64 |
| +DSS | 11.96 | 15.45 | 65.50 |
| **SOAR (CJSO+DSS)** | **11.88** | **15.44** | **66.00** |

> ✅ **CJSO 和 DSS 各有贡献，联合使用效果最佳**  
> ✅ CJSO 主要提升下游任务表现，DSS 进一步改善重建质量和PPL

### 其他重要实验
- **DSS 在 MXFP4 上也有效**（Table 5）：表明该思想可泛化到其他微缩放格式。
- **迭代次数影响**（Table 7）：随着迭代数增加，性能稳步上升，15次达到最优。
- **运行时间合理**（Table 8）：最大模型（Qwen-8B）量化耗时约36分钟，具备实用性。

---

## 4. 关键结论和发现

### 主要发现
1. **Scale Optimization 是 NVFP4 性能的关键瓶颈**：现有方法因僵化的scale选择机制和scale耦合问题，未能充分发挥NVFP4潜力。
2. **CJSO 提供高效且精准的scale初始化路径**：通过解析解快速逼近最优scale，避免暴力搜索。
3. **DSS 显著缓解 scale quantization error**：解耦设计允许在高精度空间中优化量化行为，而保持硬件兼容性。
4. **SOAR 是轻量但强大的增强模块**：不增加部署成本，却能系统性提升量化质量。

### 方法的局限性
- 当前为 **calibration-free** 框架，未显式建模激活分布的影响。
- 尽管支持与GPTQ集成，但未深入探索与其他变换类方法（如rotation）的协同效应。
- 所有实验基于模拟量化，尚未在真实Blackwell GPU上验证端到端吞吐量。

### 未来工作方向
- **引入 activation-aware objective**：利用少量校准数据优化layer-wise输出失真 $ \min \| (W - \hat{W})X \|_F $，实现数据感知的scale优化。
- **扩展至动态量化场景**：探索在不同输入下自适应调整scale的可能性。
- **应用于训练中量化（QAT）**：将CJSO/DSS思想融入训练流程，构建端到端的低比特训练方案。

---

> 🔗 **代码与模型开源地址**：https://github.com/steven-bao1/SOAR

> 💡 **一句话总结**：SOAR 通过 **闭式联合优化** 与 **解耦尺度搜索**，实现了在零硬件开销下的高保真 NVFP4 量化，是当前最先进的微缩放量化框架之一。

</details>

---

### 8. [GraphFlash: Enabling Fast and Elastic Graph Processing on Serverless Infrastructure](https://arxiv.org/abs/2605.11631)

**Authors**: Chen Zhao, Parsa Poorsistani, Mohammad Goudarzi, Tawfiq Islam, Adel N. Toosi  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.11631v1  

#### Abstract
Graph processing systems are essential for analyzing large-scale data with complex relationships, yet most existing frameworks rely on statically provisioned clusters, resulting in poor elasticity and inefficient resource utilization under dynamic workloads. Serverless computing offers automatic sca...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GraphFlash: Enabling Fast and Elastic Graph Processing on Serverless Infrastructure

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统分布式图处理系统（如 Giraph、GraphScope）依赖静态配置的集群资源，导致在动态、突发或短时负载下**弹性差、资源利用率低、成本高**。而现有的 serverless 图处理框架（如 Graphless、FaaSGraph）存在以下问题：
- **Graphless**：基于 vertex-centric 模型，通信开销大，性能低下；
- **FaaSGraph**：依赖共享内存和共置容器，违反 serverless 的无状态、隔离原则，无法真正部署于纯 serverless 平台。

因此，如何在保持 serverless 弹性优势的同时实现高性能图计算是一个关键挑战。

### 提出的新方法与创新思路
作者提出 **GraphFlash** —— 一个完全运行在 serverless 基础设施上的高效、弹性的图处理框架，其核心创新包括：

#### （1）**子图中心化（subgraph-centric）编程模型**
- 采用 GRAPE 框架的思想，以 subgraph 为单位进行并行处理，减少跨分区消息数量，提升局部性与计算效率。
- 支持标准接口 `PEval` 和 `IncVal`，便于算法开发。

#### （2）**双执行模式适应不同资源环境**
- **Rotating Mode（旋转模式）**：适用于资源受限场景，少量函数轮流处理多个分区，节省资源。
- **Pinned Mode（固定模式）**：当资源充足时，每个分区绑定到专属函数，避免重复加载数据，消除冷启动延迟，显著提升性能。

#### （3）**针对 serverless 瓶颈的系统级优化**
- **Partition-aware key aggregation**：将细粒度的顶点级通信聚合为分区级通信，将每 worker 的 key 访问从 O(v) 降至 O(p)，大幅降低 I/O 开销。
- **Intra-function partition co-location**：单个函数内并发处理多个分区，共享边界顶点存储，减少内存占用，并通过 bitmap 掩码交集机制聚合消息，避免冗余传输。
- **Superstep-aware activation**：仅在后期 superstep 启用活跃顶点检测，避免早期不必要的检查开销。
- 其他优化：二进制序列化 + zstd 压缩、前缀压缩 + varint 编码、消息批处理、Cilium CNI 提升网络吞吐等。

### 相比现有方法的优势
| 维度 | GraphFlash | Graphless | FaaSGraph |
|------|------------|----------|---------|
| **Serverless 兼容性** | ✅ 完全兼容 | ✅ | ❌（依赖共享内存） |
| **性能** | 高 | 低 | 高（但非真 serverless） |
| **成本效率** | 极高（最高降本 99.97%） | 低 | 不适用 |
| **弹性与可扩展性** | 强 | 中等 | 弱 |

GraphFlash 在保证 serverless 原则的前提下，实现了接近甚至超越传统分布式系统的性能，同时具备极强的成本效益和弹性能力。

---

## 2. 核心实验方法和设置

### 使用的数据集
涵盖真实与合成图，覆盖多种规模与结构特征：
- **真实图**：
  - `dota-league (DL)`：61.1K 节点，50.9M 边
  - `com-friendster (CF)`：65.6M 节点，1.81B 边
- **合成图（Graph500 系列）**：
  - `G3 (2^23)` ~ `G8 (2^28)`：节点数从 4.6M 到 121.2M，边数达 4.23B
- **其他合成图**：
  - `ZF`：434.9M 节点，1.04B 边（顶点密集型）

### 实验设置
- **平台**：
  - 自建 Knative 集群（4 节点，AMD EPYC 9474F，128GB RAM，25Gbps 网络）
  - AWS Lambda（x86 架构，按 GB-second 计费）
- **MaaS 存储层**：
  - Knative：Dragonfly（元数据）+ MinIO（数据）
  - AWS：S3 替代 MinIO
- **资源配置**：默认每函数 1 core / 2GB 内存（Knative），Lambda 按比例分配 vCPU

### 评估指标
- **执行时间（Execution Time）**
- **资源消耗**：Core·seconds、GB·seconds
- **金钱成本（AWS 上的实际费用）**
- **并发函数数变化趋势**
- **消融实验中的性能增益**

### 基线方法对比
| 方法 | 类型 | 是否 serverless 可部署 |
|------|------|------------------------|
| **GraphFlash (pinned/rotating)** | 本文方法 | ✅ |
| **Graphless [13]** | Serverless 图框架 | ✅ |
| **FaaSGraph [14]** | 类 serverless，需共置容器 | ❌（非真 serverless） |
| **GraphScope [8]** | 分布式图系统（多引擎） | ❌ |
| **Giraph [5]** | 经典 BSP 框架（Hadoop 生态） | ❌ |

> 注：FaaSGraph 因不支持 CDLP 算法且不能部署于纯 serverless 环境，在部分实验中被排除。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）**总体性能对比（Fig. 7）**
- 在小中型图（DL、G3、G5）上，GraphFlash 显著优于所有基线：
  - 比 Graphless 快 **12× ~ 127×**（CDLP 最高达 127×）
  - 比 Giraph 快数倍
  - 与 GraphScope 相当甚至更优（尤其在小图上免去编译开销）
- 在大型图（G7）上，性能与 GraphScope 持平，优于 Giraph 和 FaaSGraph。

#### （2）**成本效率（Fig. 8, Fig. 11）**
- 在 AWS Lambda 上运行 DL 数据集：
  - **速度提升最高达 48×**
  - **成本降低高达 99.97%**
  - 单函数运行时资源消耗减少 **96.8% ~ 98.8%**
- 在 Knative 上：
  - 所有配置均比 Graphless 更快且使用更少并发函数。

#### （3）**可扩展性测试（Table II）**
| Dataset | BFS | PageRank | CDLP | WCC |
|--------|-----|----------|------|-----|
| CF (64p) | 102.95s | 154.14s | 281.94s | 126.02s |
| ZF (256p) | 226.46s | 328.18s | 321.48s | 241.83s |
| G8 (256p) | 167.07s | 311.61s | 321.48s | 199.87s |

→ 表明 GraphFlash 能稳定扩展至超大规模图（如 G8 的 121M 节点、4.23B 边），无明显性能退化。

#### （4）**分区数对性能的影响（Fig. 9）**
- 不同算法最优分区数不同：
  - PageRank 最佳为 18 分区
  - CDLP 因计算密集，最佳为 42 分区
- 成本（core·sec）随分区增加单调上升 → 过度划分会因 edge cut 增加而导致通信开销上升。

### 消融实验结果（Section VI-B5）

#### （1）**Partition-aware key aggregation（表 III）**
- 对 CDLP 和 WCC 提升最大：
  - G5-WCC 上提速 **5.42×**
  - G5-CDLP 上提速 **4.06×**
- 小图也有明显收益（DL-Pagerank 提速 2.4×）

#### （2）**Intra-function partition co-location（表 IV & V）**
- 时间提速：G3 上 BFS 提速 1.32×，WCC 提速 1.31×
- **内存节省超过 50%**：
  - G3：543MB → 254MB（↓53.2%）
  - G6：52.3GB → 23.2GB（↓55.6%）

#### （3）**Superstep-aware activation（表 VI）**
- 性能提升虽小但一致：
  - G6-WCC 上执行时间从 21.8s 降至 17.6s（↓19%）
  - G3-WCC 下降 25%

---

## 4. 关键结论和发现

### 主要发现
1. **Serverless 图处理可以既快又省**：
   - GraphFlash 证明了在严格遵守 serverless 原则（无状态、自动伸缩、按需计费）的前提下，仍可实现媲美传统分布式系统的性能。
   
2. **subgraph-centric + 多优化组合是突破口**：
   - 子图为中心的模型天然适合降低通信；
   - 结合 key 聚合、分区共置、激活感知等优化，有效缓解了 serverless 的 I/O 和冷启动瓶颈。

3. **灵活执行模式增强实用性**：
   - Rotating 模式适合低成本、资源紧张场景；
   - Pinned 模式释放全部性能潜力，适用于高性能需求。

4. **成本优势极其显著**：
   - 最高实现 **99.97% 的成本削减**，特别适合间歇性、突发性的图分析任务。

### 方法的局限性
- **严重依赖外部存储（MaaS）性能**：虽然 S3/Dragonfly 可扩展，但仍是潜在瓶颈。
- **当前仍基于 BSP 模型**：不适合异步或流式图计算。
- **缺乏对 UDF 外部依赖的良好支持**：函数镜像需打包算法逻辑，灵活性有限。
- **未解决函数间直接通信问题**：仍需通过 MaaS 中转消息，引入额外延迟。

### 未来工作方向
- 设计支持 **function-to-function direct communication** 的 serverless 平台，绕过 MaaS 中继，进一步降低通信延迟。
- 扩展支持 **streaming graph processing** 和 **heterogeneous workloads**。
- 探索 **AI-driven auto-scaling 与 partition tuning** 策略，实现全自动优化。
- 集成 **checkpointing 与 fault tolerance** 机制，提升生产可用性。

---

> ✅ **总结一句话**：  
> **GraphFlash 是首个在纯 serverless 环境下实现高性能、高弹性、低成本图处理的实用框架，通过 subgraph-centric 模型与多项系统优化，在性能上超越已有 serverless 方案多达 127×，成本降低近 100%，推动 serverless 图计算走向现实应用。**

</details>

---

### 9. [Agent-X: Full Pipeline Acceleration of On-device AI Agents](https://arxiv.org/abs/2605.10380)

**Authors**: Jinha Chung, Byeongjun Shin, Jiin Kim, Minsoo Rhu  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.10380v1  

#### Abstract
LLM-based agents deliver state-of-the-art performance across tasks but incur high end-to-end latency on edge devices. We introduce Agent-X, a software-only, accuracy-preserving framework that accelerates both the prefill and decode stages of on-device agent workloads. Agent-X's two key components re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Agent-X: Full Pipeline Acceleration of On-device AI Agents

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

- **问题背景**：基于 LLM 的 AI Agent 在边缘设备（on-device）上运行时面临显著的端到端延迟问题，影响用户体验。
- **核心瓶颈**：传统研究认为云端 LLM 的性能瓶颈主要在 **decode 阶段**，但本文通过系统级分析发现，在资源受限的 on-device 场景下，**prefill 和 decode 阶段均构成关键延迟瓶颈**。
- **根本原因**：
  - Prefill 阶段因输入序列长、计算密集而变慢；
  - Decode 阶段受内存带宽限制，且输出高度模板化却仍依赖大模型生成。

> 🔍 这是**首次对 on-device AI Agent 进行系统级性能剖析**，揭示了全流水线优化的必要性。

---

### 🚀 提出的新方法与创新思路

提出 **Agent-X** —— 一种纯软件、不损失准确率的 on-device Agent 加速框架，包含两个核心组件：

#### （1）PromptWeaver：加速 Prefill 阶段
- **目标**：最大化 **prefix caching** 的利用率。
- **挑战**：原始 prompt 中动态插入的 `tool descriptions` 和 `tool-use examples` 导致早期 token 不匹配，KV cache 复用率低。
- **解决方案**：
  - 将所有工具描述和指南设为静态前缀（static prefix），消除早期动态性；
  - 利用 **tool co-activation locality**（工具共激活局部性）进行聚类；
  - 设计 **cluster combination selection 算法**，预计算高频 cluster 组合的 KV cache 并存储于 SSD；
  - 在线阶段按需加载，实现高效 cache reuse。

> 💡 创新点：从语义层面重构 prompt 结构，使动态内容也能被部分缓存，突破传统 prefix caching 对“完全前缀匹配”的依赖。

#### （2）ExSpec：加速 Decode 阶段
- **目标**：实现轻量级、无开销的 speculative decoding。
- **挑战**：
  - 传统 speculative decoding 依赖小型 draft LLM，但在边缘设备上存在 **multi-token tax**（验证多 token 时效率下降）和 **latency overhead**；
  - draft model 越小，acceptance rate 越低；越大则自身推理耗时高。
- **解决方案**：
  - 引入 **n-gram lookup table (LUT)** 作为零训练成本的 draft model；
  - LUT 构建自 few-shot examples 和 user query，确保 draft token 与 prompt 高度相关；
  - 设计 **selective decoding** 机制：若当前上下文不在 LUT 中，则直接回退到 autoregressive 生成，避免无效 speculative 开销。

> 💡 创新点：用极轻量的统计模型替代 LLM 作为 draft model，规避了 draft LLM 的训练与部署成本，并有效规避 multi-token tax。

---

### ⚖️ 相比现有方法的优势

| 方面 | Agent-X | 现有方法 |
|------|--------|---------|
| **适用场景** | On-device Agent 全流程优化 | 多针对通用 LLM 或仅 decode 优化 |
| **是否需要额外模型** | 否（ExSpec 使用 LUT） | 是（需训练 draft LLM） |
| **是否损失精度** | 否（iso-accuracy） | 可能（如低 bit quantization） |
| **是否可集成** | 是（纯软件方案，兼容 MLX-LM 等） | 多需硬件支持或复杂改造 |
| **KV cache 复用率** | 显著提升（利用语义相似性） | 仅限 exact prefix match |

---

## 2. 核心实验方法和设置

### 📚 数据集

- **TinyAgent-dataset** [68]：包含 1,022 个用于 fine-tuning 和测试的用户查询任务；
- 查询涵盖最多 16 种不同工具调用组合；
- 用于训练 PromptWeaver 的 clustering 与 combination selection 模块；
- 测试集用于评估 end-to-end latency 与 Planner accuracy。

---

### 🧪 实验设置

- **平台**：Apple Mac mini (M4 Pro)，64GB RAM，512GB SSD；
- **框架**：基于 Apple 的 **MLX-LM** 和 **MLX-engine** 实现；
- **后端模型**：**TinyAgent-7B**（基于 WizardLM-2-7B 微调）；
- **集成系统**：TinyAgent [19] + ToolRAG [69]；
- **对比基线**：
  - Baseline：原始 TinyAgent 流程；
  - Static caching：仅缓存固定 prompt 前缀；
  - SpecDec：使用 Llama-3.2-1B-Instruct 作为 draft LLM 的 speculative decoding；
  - Agent-X（PW）、Agent-X（ES）、Agent-X（PW+ES）：分别启用 PromptWeaver、ExSpec 或两者联合。

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **End-to-end latency** | 完整 Agent 执行任务所需时间 |
| **Prefill / Decode latency** | 分阶段延迟拆解 |
| **Speedup** | 相对于 baseline 的加速比 |
| **Planner accuracy** | 输出计划 DAG 与 ground truth 匹配程度 |
| **KV cache reuse rate** | 缓存命中的 token 比例 |
| **Draft token accuracy** | speculative decoding 中被接受的 draft token 比例 |
| **Storage overhead** | 预计算 KV cache 占用的 SSD 空间 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 模块 | 性能提升 |
|------|--------|
| **PromptWeaver** | Prefill 阶段平均加速 **1.97×** |
| **ExSpec** | Decode 阶段平均加速 **1.73×** |
| **Agent-X（完整）** | 端到端平均加速 **1.61×** |
| **Planner uncacheable tokens** | 从 1,711 ↓ 至 519（减少 **70%**） |
| **KV cache storage overhead** | 6.26 GB（其中 74.4% tool-use example 被覆盖） |

> ✅ **无精度损失**：Planner accuracy 达 **0.841**（baseline 为 0.836），略有提升。

---

### 🔁 与基线方法对比

| 方法 | End-to-end Speedup | 是否可行 | 备注 |
|------|------------------|----------|------|
| Baseline | 1.00× | ✅ | 原始流程 |
| Static caching | ~1.01× | ✅ | 改善有限，因早期动态性 |
| SpecDec (Llama-3.2-1B) | **<1.0×（变慢）** | ❌ | 受 multi-token tax 和 tokenizer 差异拖累 |
| **Agent-X (PW+ES)** | **1.61×** | ✅ | 显著加速且稳定 |

> 📉 表明：**传统 speculative decoding 在 on-device 场景下可能适得其反**，而 ExSpec 成功规避此问题。

---

### 🔍 消融实验结果

#### （1）PromptWeaver：附加 tool-use examples 数量的影响（K 值）
- K=0 → accuracy 下降；
- **K=1 → accuracy 最高（0.841）**；
- K>1 → accuracy 下降，且 uncacheable tokens 增加；
- ✅ **结论**：只需添加 1 个额外 example 即可恢复并超越原精度。

#### （2）ExSpec：n-gram 模型阶数选择
- **n=2（bigram）**：draft token accuracy = 0.10 → 过低；
- **n=3（trigram）**：accuracy = 0.25，速度最快 → **最优选择**；
- **n=4（quadgram）**：accuracy ↑ 至 0.31，但生成 draft 更保守，fallback 更频繁 → **总 decode 时间更长**。

#### （3）KV cache budget 影响
- cluster budget = 15 → 覆盖 74.4% 的 tool-use example；
- 继续增加收益递减；
- ✅ 小预算即可获得高覆盖率。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **On-device Agent 的瓶颈是双向的**：
   - 不同于云端 LLM，**prefill 和 decode 都是主要延迟来源**；
   - 必须进行 **full-pipeline acceleration**。

2. **Agent 特有的语义规律可被利用**：
   - 工具具有强 **co-activation locality**；
   - 输出高度依赖 few-shot examples → 可用简单模型预测。

3. **轻量化设计优于复杂模型堆叠**：
   - ExSpec 使用 **n-gram LUT** 替代 draft LLM，实现更高效率；
   - **selective decoding** 避免无效 speculative 开销。

4. **纯软件方案即可实现显著加速**：
   - Agent-X 无需修改硬件或模型架构；
   - 可无缝集成到现有 on-device Agent 系统中（如 TinyAgent）。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖 SSD 存储 KV cache** | 增加约 6GB 存储开销，对存储敏感设备可能受限 |
| **适用于 fine-tuned agent system** | PromptWeaver 依赖训练数据构建 cluster，泛化性待验证 |
| **主要面向文本型 Agent** | 对视觉或多模态 Agent 支持有限 |
| **n-gram LUT 效果依赖 prompt 质量** | 若 few-shot examples 质量差，ExSpec 性能下降 |

---

### 🔮 未来工作方向

1. **扩展至多模态 Agent**：结合 vision encoder 输出构建 multimodal LUT；
2. **动态更新 KV cache**：应对工具集合变化或用户习惯漂移；
3. **跨设备协同缓存**：在家庭/办公网络中共享常用 KV cache；
4. **探索更复杂的轻量 draft model**：如 FSM-based generator 或 rule-enhanced LUT；
5. **支持更多 backend 框架**：除 MLX 外，适配 Android NNAPI、Qualcomm SNPE 等。

---

## ✅ 总结

> **Agent-X 是首个针对 on-device AI Agent 全流程延迟瓶颈进行系统优化的工作**。它通过 **PromptWeaver** 和 **ExSpec** 分别解决 prefill 和 decode 阶段的效率问题，在不牺牲准确率的前提下，实现了 **1.61× 的端到端加速**。其纯软件、轻量化、易集成的设计理念，为推动私密、快速、本地化的 AI Agent 落地提供了重要实践路径。

</details>

---

### 10. [Efficient LLM-based Advertising via Model Compression and Parallel Verification](https://arxiv.org/abs/2605.11582)

**Authors**: Wenxin Dong, Chang Gao, Guanghui Yu, Xuewu Jiao, Mingqing Hu, Qiang Fu, Peng Xu, Penghui Wei, Hui Xu, Yue Xing, Shuanglong Li, Lin Liu  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.11582v1  

#### Abstract
Large language models (LLMs) have shown remarkable potential in advertising scenarios such as ad creative generation and targeted advertising. However, deploying LLMs in real-time advertising systems poses significant challenges due to their high inference latency and computational cost. In this pap...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Efficient LLM-based Advertising via Model Compression and Parallel Verification*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**大语言模型（LLM）在在线广告场景中推理延迟高、计算成本大**的问题，提出了一套高效的生成式广告投放框架。传统方法虽然在推荐精度上表现优异，但在实时性要求极高的广告系统中面临严重的**可扩展性-效率困境（scalability-efficiency dilemma）**。

具体挑战包括：
- LLM 参数量大导致推理延迟高；
- 传统压缩技术牺牲精度；
- 自回归解码过程缓慢，难以满足实时广告请求。

---

### 提出的新方法与创新思路

作者提出了一个集成化的高效 LLM 推理框架，包含两大核心技术模块：

#### ✅ **Model Compression（模型压缩）**
1. **Adaptive Group-Wise Quantization（自适应分组量化）**  
   - 将模型层按参数敏感度划分为敏感层与非敏感层；
   - 对敏感层采用细粒度（更多组）量化，非敏感层粗粒度处理；
   - 实现从 FP16 到 INT4 的压缩，在保持精度的同时显著降低计算开销。

2. **Layer-wise Semi-Structured Sparsity（逐层半结构化稀疏化）**  
   - 基于重要性分析进行 N:M 剪枝（如 2:4 或 1:4）；
   - 关键层保留更高密度（2:4），次要层更激进剪枝（1:4）；
   - 针对 GEMV 工作负载优化，提升矩阵运算效率。

3. **Index-Compressed Data Structure（索引压缩结构）**  
   - 提出 **2bit-CSR** 结构，将原始 CSR 的索引和权重大小压缩至原来的 30%；
   - 显著减少内存带宽消耗。

4. **Custom SparseGemv Kernel（定制稀疏加速核）**  
   - 自主开发支持 INT4 权重仅量化（weight-only）和混合稀疏性的 SparseGemv 内核；
   - 弥补了 NVIDIA cuSparse/cuSparseLT 在 GEMV 加速上的空白。

#### ✅ **Prefix Tree-based Parallel Verification（前缀树并行验证）**
1. **Hierarchical Clustering 构建 Prefix Tree（Trie）**  
   - 使用 DSI 算法对广告实体（如“淘宝”、“百度”）聚类构建语义结构化的 Trie；
   - Trie 上宽下窄，利于逐步缩小候选空间。

2. **Dynamic Parallel Verification Trigger（动态触发机制）**  
   - 动态判断何时启动并行验证：基于生成剩余 token 所需时间 vs 并行验证开销的时间差；
   - 实现最优切换点选择，避免过早或过晚触发带来的资源浪费。

3. **Tree-Based Parallel Decoding + Beam Search**  
   - 将 speculative decoding 思想与 Trie 约束结合；
   - 多路径并行解码 + 树掩码（tree mask）过滤无效路径；
   - 使用 BScore（节点得分）进行 beam search，最终回溯生成合法序列。

---

### 相比现有方法的优势

| 维度 | 本文方法优势 |
|------|--------------|
| **效率** | 推理速度提升 >78%，生产环境中实现 **>1.8× speedup**；消融显示最高达 **×1.89 加速比** |
| **精度保留** | Recall 和 BLEU/Meteor 指标下降极小，尤其在混合稀疏 + 量化配置下达到最佳权衡 |
| **工程落地性** | 定制内核适配真实硬件（A10/A30 GPU），已在百度广告平台部署 |
| **方法新颖性** | 据作者所知，首次将 **prefix tree + beam search + speculative decoding** 融合应用于广告生成任务 |

---

## 2. 核心实验方法和设置

### 使用的数据集

| 场景 | 数据集 | 描述 |
|------|--------|------|
| **Targeted Advertising（定向广告）** | 公司内部商业流量数据（未公开） | 包含用户查询与对应广告匹配记录 |
| **Ad Creative Generation（广告创意生成）** | **CSL**（Chinese Scientific Literature Dataset） | 包含约 39.6 万篇中文论文元数据，用于创意改写与关键词摘要任务 |

---

### 实验设置

- **基础模型**：ERNIE 1.5B（15亿参数）
- **框架**：PaddlePaddle
- **硬件环境**：
  - 广告生成任务：NVIDIA A10 GPU，beam size = 1
  - 定向广告任务：NVIDIA A30 GPU，beam size = 20（因需输出多个候选）
- **输入长度**：平均约 17–18 tokens

---

### 评估指标

| 任务 | 主要指标 |
|------|---------|
| **Targeted Advertising** | Per-token latency（每 token 延迟）、Recall（召回率） |
| **Ad Creative Generation** | BLEU、Meteor（文本生成质量）、AVGLen（平均长度）、latency |
| **综合评估** | Speedup（相对于 FP16 基线的加速比） |

---

### 基线方法对比

- **Baseline (FP16)**：全精度模型，无任何压缩或优化
- **Quantization Only**：仅应用 INT4 分组量化
- **Sparsity Only**：分别测试 2:4 和 1:4 的 N:M 剪枝
- **Sparse + Quant**：稀疏化 + 量化组合
- **+PTPV**：进一步加入 Prefix Tree Parallel Verification

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Figure 2 & Table 1）

#### 🔹 定向广告场景（Targeted Advertising）

| 方法 | Per-token Latency (ms) | Recall | Speedup |
|------|------------------------|--------|--------|
| Baseline (FP16) | ~1.9 | 60.08% | ×1.00 |
| Quantization | ~1.4 | 60.08% | — |
| Sparse + Quant | ~1.2 | 60.08% | — |
| **Full Framework (Sparse+Quant+PTPV)** | **~1.0** | **60.08%** | **>1.8×** |

✅ **结论**：在不损失 Recall 的前提下，实现 **>1.8× 实际推理加速**。

---

#### 🔹 广告创意生成（Ad Creative Generation）

| 方法 | BLEU | Meteor | AVGLen | Latency (ms/token) | Speedup |
|------|------|--------|--------|--------------------|--------|
| Baseline (FP16) | 0.4247 | 0.6345 | 17.5 | 6.6 | ×1.00 |
| Quantization | 0.4178 | 0.6283 | 17.6 | 4.8 | ×1.37 |
| Sparsity (2:4) | 0.4161 | 0.6260 | 17.5 | 5.3 | ×1.25 |
| Sparsity (1:4) | 0.3476 | 0.5549 | — | 4.6 | ×1.43 |
| **Sparse(2:4)+Quant** | 0.4103 | 0.6195 | — | **4.0** | **×1.65** |
| **Sparse(1:4)+Quant** | 0.3369 | 0.5446 | — | 3.5 | ×1.89 |
| **Sparse(Mix)+Quant** | 0.4038 | 0.6127 | — | **3.7** | **×1.78** |

> 注：“Mix” 表示根据不同层的重要性混合使用 2:4 与 1:4 稀疏策略。

✅ **关键发现**：
- 单独量化即可带来 **37% 速度提升**，且质量几乎无损；
- 混合稀疏 + 量化方案在质量和效率之间取得**最佳平衡**（×1.78 加速，BLEU 仅降 ~2%）；
- 激进稀疏（1:4）虽快，但质量下降明显，不适合高精度业务。

---

### 消融实验（Ablation Study）

| 组件 | 贡献说明 |
|------|--------|
| **Quantization** | 最有效单一项，显著降低延迟，维持高生成质量 |
| **Sparsity (2:4)** | 效率增益有限，但对关键层保护较好 |
| **Sparsity (1:4)** | 速度快，但 Meteor 下降超 10%，不可接受 |
| **Sparse + Quant** | 协同效应强，压缩效果叠加 |
| **+PTPV** | 进一步释放并行潜力，是达成 >1.8× 的关键一环 |

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **模型压缩与解码优化必须协同设计**：单独使用量化或稀疏无法满足工业级实时需求，必须结合定制 kernel 与并行验证才能突破瓶颈。
2. ✅ **自适应策略优于固定规则**：无论是量化分组还是稀疏比例，基于层重要性的动态调整能更好兼顾效率与精度。
3. ✅ **Prefix Tree 可有效约束搜索空间**：通过语义聚类构建 Trie，使 beam search 更聚焦于高概率路径，减少冗余计算。
4. ✅ **动态触发 parallel verification 是关键**：静态开启并行会引入额外开销，动态决策确保收益最大化。
5. ✅ **已成功落地生产环境**：该系统已在 **Baidu 广告平台上线运行**，服务于大规模实时流量。

---

### 方法的局限性

1. **领域依赖性强**：当前优化策略（尤其是稀疏化和 Trie 构建）高度针对**商业广告数据分布**，通用性有待验证；
2. **Trie 构建依赖先验知识**：需要高质量的广告 ID 库和聚类算法支持，冷启动场景可能表现不佳；
3. **对长序列生成支持有限**：实验集中在短文本生成（avg len < 20），是否适用于更复杂文案仍需验证；
4. **缺乏跨模型泛化测试**：仅在 ERNIE 1.5B 上验证，其他 LLM（如 Llama、ChatGLM）适配性未知。

---

### 未来工作方向

1. **引入 Adaptive Algorithms + Reinforcement Learning**  
   - 动态调节量化粒度、稀疏比例、并行触发时机，实现更智能的运行时优化；
   
2. **拓展到多模态广告生成场景**  
   - 支持图文联合生成、视频广告脚本等更复杂的 creative 输出；

3. **探索全自动 Trie 构建机制**  
   - 减少人工干预，实现端到端语义结构学习；

4. **研究跨领域迁移能力**  
   - 将本框架推广至新闻推荐、电商搜索等其他生成式推荐任务。

---

> 📌 **一句话总结**：  
> 本文提出了一套面向工业级 LLM 广告系统的高效推理框架，通过 **adaptive quantization + layer-wise sparsity + custom kernel + prefix tree parallel verification** 四重优化，在保持精度的前提下实现 **>1.8× 实际加速**，并已成功部署于百度广告平台，为生成式推荐系统的落地提供了重要实践范例。

</details>

---

### 11. [PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory for Long-Horizon Agents](https://arxiv.org/abs/2605.12260)

**Authors**: Jingyi Peng, Zhongwei Wan, Weiting Liu, Qiuzhuang Sun  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.12260v1  

#### Abstract
Long-horizon language agents accumulate conversation history far faster than any fixed context window can hold, making memory management critical to both answer accuracy and serving cost. Existing approaches either expand the context window without addressing what is retrieved, perform heavy ingesti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory for Long-Horizon Agents

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在**long-horizon language agents**（长周期语言智能体）中，对话历史迅速增长，远超任何固定长度的上下文窗口（context window）。这导致两个核心挑战：
- **答案准确性**：如何从海量记忆中检索出真正相关的信息；
- **服务成本**：如何在有限的上下文预算下高效检索，避免高昂的token开销。

现有方法存在以下不足：
- 单纯扩展上下文窗口（如使用Long-context LLMs）成本极高，且模型对长输入中的关键证据注意力下降；
- 依赖启发式图遍历（heuristic graph traversal）的方法效率低、精度不稳定；
- 重写时提取事实（ingestion-time fact extraction）的方法token消耗大，灵活性差。

PRISM旨在解决这一**准确率-上下文成本权衡**（accuracy-context-cost trade-off）问题，尤其是在高准确率、低上下文成本的“空白区域”。

---

### 提出的新方法和思路
PRISM是一个**无需训练的检索侧框架**（training-free retrieval-side framework），将长期记忆管理建模为一个**联合检索与压缩问题**（joint retrieval-and-compression problem），运行在一个**图结构化记忆**（graph-structured memory）之上。

其核心由四个正交的推理时组件构成：

| 组件 | 功能 |
|------|------|
| **N1: Hierarchical Bundle Search** | 在**带类型的关联路径**（typed relation paths）上进行分层束搜索，超越表面相似性，支持多跳推理 |
| **N2: Query-Sensitive Edge Costing** | 根据检测到的查询意图动态调整边的遍历成本，使检索更符合语义意图（如“when”偏好temporal边，“why”偏好causal边） |
| **N3: Evidence Compression** | 使用单次LLM调用对候选证据进行重排序和压缩，生成紧凑的上下文，显著减少token数 |
| **N4: Adaptive Intent Routing** | 通过三级级联路由（关键词匹配 → 原型嵌入 → LLM分类）决定是否调用LLM，避免每次查询都使用LLM判断意图 |

> ✅ **关键创新**：首次在不引入训练策略的前提下，将**基于路径模板的最小成本检索**与**LLM端证据压缩**相结合，实现高精度、低成本的检索。

---

### 相比现有方法的优势
- **无需微调或修改上游流水线**：完全插件式设计，适用于任何已构建的图结构记忆系统。
- **极高的上下文效率**：相比全上下文基线，**减少约13倍token**，同时**提升35个百分点的准确率**。
- **帕累托最优**：在准确率-上下文成本前沿占据此前未被填充的“高准确率/低成本”角落。
- **模块化与可解释性**：每个组件职责明确，便于分析与优化。

---

## 2. 核心实验方法和设置

### 数据集
- **LoCoMo** [14]：一个专为评估**长周期对话记忆能力**设计的基准，包含10场多轮对话，共1,540个问答对。
- 覆盖四类任务：
  - Single-hop（单跳）
  - Multi-hop（多跳）
  - Temporal（时间推理）
  - Open-domain（开放域）

> ❗排除第5类（对抗性拒绝测试），因不涉及检索质量评估。

---

### 实验设置与评估指标

#### 主要设置
- **Answer Model & Judge Model**：`gpt-4o-mini`（temperature=0.0）
- **统一协议**（same-protocol）：所有对比方法使用相同的提示词、tokenizer、token计数方式，确保公平比较。
- **上下文预算**：目标是尽可能少地传递token给answer model。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **LLM-judge score** | 由LLM judge判断生成答案是否正确，计算 `CORRECT / (CORRECT + WRONG)` |
| **Context tokens per query** | 平均每条查询传给answer model的token数，衡量检索成本 |
| **Per-1K Efficiency** | 每千token带来的judge得分，衡量单位成本效益 |
| **Evidence Recall@K (ER@K)** | 前K个检索结果中包含黄金证据的比例（用于消融分析） |

---

### 基线方法对比
| 基线 | 类型 | 特点 |
|------|------|------|
| **Full Context** | 长上下文基线 | 将全部~26K tokens输入模型 |
| **MAGMA** [7] | 图结构记忆 | 使用启发式beam search遍历四种边图 |
| **Mem0 / Mem09** [3] | 结构化记忆提取 | 在写入时提取结构化事实，查询时检索 |
| **M-Flow** [6] | 商业级流程参考 | 不同协议下的更强系统（使用gpt-5-mini） |
| **Mem0 platform** | 商业平台 | 受控流水线，更高预算 |

> PRISM与前四者在同一协议下复现对比；后两者作为不同协议参考。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1）

| 方法 | Judge Score | Context Tokens/Query | Per-1K Efficiency |
|------|------------|------------------------|-------------------|
| Full Context | 0.481 | 26,031 | 0.018 |
| MAGMA | 0.688 | 3,370 | 0.204 |
| Mem0 | 0.669 | 1,764 | 0.379 |
| Mem09 | 0.684 | 3,616 | 0.189 |
| **PRISM (ours)** | **0.831** | **2,023** | **0.411** |
| PRISM (gpt-5.5) | 0.891 | 2,023 | 0.440 |
| Mem0 platform | 0.916 | ~7,000 | 0.131 |

---

### 与基线方法的对比结果
- **PRISM以2K token实现0.831准确率**，**击败所有相同协议基线**：
  - 比**Mem09**高出 **+14.2 pp**
  - 比**MAGMA**高出 **+14.0 pp**
  - 比**Full Context**高出 **+35.2 pp**
- **上下文成本仅为Full Context的1/13**（~26K → ~2K），却大幅提升准确率。
- **效率最高**：每千token带来0.411分，优于Mem0的0.379。

> 🔍 **重要发现**：**检索质量 > 原始上下文大小**  
> 即使模型能处理完整对话（26K tokens），直接喂入仍仅得0.481分，低于所有检索方法。说明“**检索什么**”比“**检索多少**”更重要。

---

### 消融实验结果（Ablation Study）

#### 主要发现（Table 2 & E.1）

| 变体 | Judge Score | ER@5 | Context Tokens |
|------|-------------|-------|---------------|
| PRISM (完整) | **0.831** | **0.694** | **2,023** |
| -N1 (无关系路径) | 0.831 | 0.694 | 2,024 |
| -N2 (无边成本调整) | 0.831 | 0.694 | 2,020 |
| **-N3 (无LLM重排)** | **0.825** | **0.627↓** | **4,108↑** |
| +N4 (启用自适应路由) | 0.833 | 0.694 | 2,023 |

#### 关键结论：
- **N3（Evidence Compression）是主导因素**：
  - 移除后，**上下文膨胀至4,108 tokens**（+2K），证明其为**最主要的压缩杠杆**；
  - ER@5下降6.8pp，说明它有效过滤了结构便宜但主题无关的内容；
  - 准确率轻微下降，表明其作用主要是**提升证据排序质量**而非最终答案生成。
- **N1 和 N2 在LoCoMo上效果不显著**：
  - 因为LoCoMo中**72%的多跳问题是anchor-discoverable**（可通过命名实体或关键词直接命中）；
  - 真正需要“桥接”（bridge-style）推理的问题仅占约3%，不足以体现优势。
  - 预期在MuSiQue、HotpotQA等更难的多跳数据集上会更有效。
- **N4（自适应意图路由）节省LLM调用**：
  - **42.3%的查询无需LLM即可判断意图**（通过关键词或原型匹配）；
  - 其中**82.6%的时间类查询**可通过关键词触发（如“when”、“before”）；
  - 准确率无统计显著差异（△=+0.26pp, p=0.71），说明**零成本替换部分LLM调用是安全的**。

---

## 4. 关键结论和发现

### 主要发现
1. **高准确率与低上下文成本可以兼得**：PRISM成功占据了准确率-成本前沿的“空白角落”，证明二者并非强权衡关系。
2. **检索应视为“最小成本路径选择”问题**：通过定义**带类型的关系路径模板**并最小化路径成本，可系统化支持复杂推理。
3. **LLM-side压缩是关键压缩机制**：相比结构化检索，**内容感知的LLM重排序**才是降低上下文token的核心。
4. **意图感知可显著提升检索效率**：通过动态调整边权重，使检索更贴合查询语义。
5. **无需训练也能实现高性能**：PRISM全程无需fine-tuning或policy learning，具备良好的通用性和部署友好性。

---

### 方法的局限性（Appendix A）
- 当前聚焦于**对话记忆**，尚未扩展到包含动作、工具调用、反馈等更丰富的agent轨迹。
- **SEMANTIC边默认关闭**：在LoCoMo上相似度诱导的语义链接噪声较多，未来需更精细控制。
- 对**非anchor-discoverable的多跳问题**依赖较强，但在当前主流benchmark中占比不高。

---

### 未来工作方向
- 扩展至**action-aware memory graphs**，支持完整agent轨迹管理。
- 探索在**MuSiQue、HotpotQA**等更具挑战性的多跳QA数据集上的表现。
- 重新激活并优化**SEMANTIC边**，探索跨文档概念链接的价值。
- 进一步优化**Adaptive Intent Routing**，减少对LLM fallback的依赖。
- 与KV-cache压缩、long-context LLMs结合，形成端到端高效推理系统。

---

> ✅ **总结一句话**：  
> **PRISM通过“意图感知的图路径检索 + LLM端证据压缩”的免训练组合，在不牺牲准确率的前提下，将上下文成本降低一个数量级，为长周期agent的记忆管理提供了新的帕累托最优解。**

</details>

---

### 12. [LEAP: Unlocking dLLM Parallelism via Lookahead Early-Convergence Token Detection](https://arxiv.org/abs/2605.10980)

**Authors**: Haohui Zhang, Zhiye Wang, Xiaoying Gan, Xinbing Wang, Bo Jiang  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.10980v1  

#### Abstract
Diffusion Language Models (dLLMs) have garnered significant attention for their potential in highly parallel processing. The parallel capabilities of existing dLLMs stem from the assumption of conditional independence at high confidence levels, which ensures negligible discrepancy between the margin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LEAP: Unlocking dLLM Parallelism via Lookahead Early-Convergence Token Detection

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Diffusion Language Models (dLLMs)** 虽然具备并行生成潜力，但其实际并行度受限于**基于高置信度（high-confidence）的解码策略**。这类方法仅在预测概率超过某一严格阈值时才解码 token，导致：
- 大量已正确收敛但置信度中等的 token 被延迟解码；
- 并行候选 token 数量稀少，限制了解码效率；
- 高置信 token 信息熵低，不利于后续上下文推理。

因此，**依赖高置信先验成为 dLLM 并行化的瓶颈**。

### 🚀 提出的新方法：LEAP
作者提出 **LEAP (Lookahead Early-Convergence Token Detection)** ——一种无需训练、即插即用的加速解码策略，核心思想是：
> **检测“早期收敛”（early-converged）token**，即使它们尚未达到高置信度标准。

#### 创新机制：
1. **未来上下文剪枝（Future Context Candidate Pruning）**  
   在每一步 $t-1$ 后，保留所有置信度高于一个宽松阈值 $\eta$ 的候选 token，作为潜在的未来上下文。

2. **多序列叠加一致性检测（Multi-Sequence Superimposed Consistency Detection）**  
   将原始序列与候选 token 和 mask token 的副本拼接成一个“叠加序列”（superposed context），通过一次前向传播同时计算：
   - 当前上下文下的预测；
   - 受扰动（含未来信息）上下文下的预测。

3. **一致性判据进行早解码**  
   若某 token 在两种上下文中预测一致且满足最低置信阈值 $\tau$，则判定为“早期收敛”，可提前解码。

### 🔍 相比现有方法的优势
| 特性 | LEAP | Confidence-Based | KLASS | LoPA |
|------|------|------------------|--------|-------|
| 是否需训练 | ❌（training-free） | ❌ | ❌ | ❌ |
| 利用中等置信 token | ✅ | ❌ | ⚠️（需高置信+低KL） | ⚠️（基于分支置信） |
| 检测早期收敛 | ✅（显式） | ❌ | ❌ | ⚠️（间接） |
| 并行度提升幅度 | **显著更高** | 中等 | 中等 | 高 |
| 准确率保持能力 | ✅（持平或略优） | ✅ | ✅ | ⚠️可能下降 |

> **核心优势**：突破对“高置信”的强依赖，在不牺牲精度的前提下大幅提升并行度和信息增益。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
涵盖多个领域任务，验证泛化性：
- **数学推理**：GSM8K（4-shot）、MATH（4-shot）
- **代码生成**：HumanEval（0-shot）、MBPP（3-shot）
- **综合问答**：GPQA（graduate-level QA）

### ⚙️ 实验设置
- **基础模型**：
  - `LLaDA-8B-Instruct`（纯扩散架构）
  - `Dream-7B-Instruct`（基于 AR 权重初始化的 dLLM）
- **生成策略**：semi-autoregressive，block size = 32
- **硬件平台**：单张 NVIDIA 5090（32GB）GPU
- **超参数**：默认 $\eta = 0.2$, $\tau = 0.7$

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **Acc** | 任务准确率（如 Pass@1） |
| **TPS (Tokens Per Second)** | 每秒解码 token 数量 |
| **Steps** | 平均去噪步数（denoising steps） |
| **Spd(Lat.)** | 推理延迟相对于全步长 baseline 的加速比 |
| **TFLOPs / TFOPs** | 总计算量（Token Forward Operations），用于衡量能效 |

### 🆚 基线方法对比
1. **Baseline**：完整步数解码（无并行）
2. **Conf-Based**：传统置信度阈值法（$\text{threshold}=0.9$）
3. **KLASS**：结合 KL 散度稳定性的高置信解码
4. **LoPA**：基于未来分支置信的并行采样

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2）

| 方法 | GSM8K (LLaDA) | HumanEval (LLaDA) | MBPP (LLaDA) | MATH (LLaDA) | GPQA (LLaDA) |
|------|---------------|--------------------|--------------|-------------|-------------|
| Baseline | 1.0× (256 steps) | 1.0× (512 steps) | 1.0× (512 steps) | 1.0× (256 steps) | 1.0× (256 steps) |
| Conf-Based | **3.2×**, 79 steps | **2.8×**, 176 steps | **11.3×**, 45 steps | **2.5×**, 101 steps | **9.1×**, 28 steps |
| **LEAP** | **4.0×**, **58 steps** | **3.3×**, **125 steps** | **20.3×**, **22 steps** | **3.1×**, **73 steps** | **13.3×**, **17 steps** |

> ✅ **平均减少约 30% 的去噪步数**，最高达 **50%+ 步数压缩**

#### 在 Dream 模型上的表现更优：
- 平均加速比达 **10.6×**（vs Conf-Based 的 8.3×）
- 如 GPQA 上实现 **62.0× 延迟加速**

### 🔁 与 dParallel 结合的结果（Table 3）
| 方法 | GSM8K TPF | Score |
|------|-----------|--------|
| dParallel | 4.7 | 75.4% |
| **dParallel + LEAP** | **7.2** | **75.1%** |

> 💡 **TPF（Tokens Per Forward pass）提升 53%**，证明 LEAP 与模型级优化正交且可叠加。

### 📉 消融分析（Hyperparameter Sensitivity）

#### 影响阈值 $\tau$（图6a-b）：
- $\tau < 0.6$：准确率明显下降 → 存在噪声解码风险
- $\tau = 0.7$：在 GSM8K 上达到最优平衡 → 设为默认值

#### 影响阈值 $\eta$（图6c-d）：
- 即使 $\eta = 0.1$（更多候选），总 TFOPs 仍比 Conf-Based 低 **20–30%**
- 因步数大幅减少，抵消了每步开销增加 → 验证了“全局加速 > 局部开销”

### ⏱️ 每步开销分析（Figure 7）
| 场景 | 增加 token 数 | 延迟增加 |
|------|----------------|----------|
| GSM8K ($\eta=0.5$) | +1.4% | +5.2% |
| GSM8K ($\eta=0.1$) | +3.6% | +8.9% |

> ✅ **每步开销可控，全局收益远大于成本**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **大量 token 实际上“早熟收敛”但未被利用**  
   统计显示许多 medium-confidence token 已正确预测并在后续步骤中保持稳定（见 Figure 2），说明 confidence-based 方法过于保守。

2. **早期收敛 ≠ 高置信，但可可靠识别**  
   通过引入“未来上下文扰动 + 一致性检测”，可在不访问未来真实状态的情况下判断 token 是否已收敛。

3. **LEAP 显著提升并行度而不损失精度**  
   - 平均减少 **~30% 解码步数**
   - TPS 提升至 **30+ tokens/second**
   - 在部分任务上 **准确率略有上升**（如 LLaDA 上从 47.4% → 47.8%）

4. **方法具有正交性和通用性**  
   成功集成到 dParallel 等模型级加速技术中，进一步提升性能（TPF 从 4.7→7.2），表明其适用于多种 dLLM 架构。

### ⚠️ 方法的局限性
- **依赖模型输出稳定性假设**：若模型对未来上下文敏感度过高，可能导致一致性误判。
- **叠加序列带来内存压力**：虽然实验证明开销可控，但在极长序列或资源受限场景下可能受限。
- **目前仅支持 greedy decoding**：未探索采样（sampling）模式下的扩展。

### 🔮 未来工作方向
1. 扩展至 **non-greedy 解码策略**（如 nucleus sampling）；
2. 探索 **动态调整 $\eta$ 和 $\tau$** 的自适应机制；
3. 应用于 **语音、图像等其他离散扩散模型**；
4. 结合 **KV-Cache 优化** 进一步降低叠加序列的计算负担。

---

## ✅ 总结
**LEAP 是一项突破性的 dLLM 加速技术**，它通过识别“早期收敛但非高置信”的 token，打破了长期以来对高置信解码的依赖。其实现简单、无需训练、即插即用，并在多个基准上实现了 **高达 3–6 倍的推理加速**，同时保持甚至略微提升了模型准确性。该工作为 **dLLM 的高效并行推理开辟了新范式**，有望推动 diffusion-based LLM 在实际应用中的广泛部署。

</details>

---

### 13. [U-STS-LLM A Unified Spatio-Temporal Steered Large Language Model for Traffic Prediction and Imputation](https://arxiv.org/abs/2605.11735)

**Authors**: Yichen Zhang, Jun Li  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.11735v1  

#### Abstract
The efficient operation of modern cellular networks hinges on the accurate analysis of spatio-temporal traffic data. Mastering these patterns is essential for core network functions, chiefly forecasting future load to pre-empt congestion and imputing missing values caused by sensor failures or trans...

---

### 14. [Arcane: An Assertion Reduction Framework through Semantic Clustering and MCTS-Guided Rule Exploring](https://arxiv.org/abs/2605.10107)

**Authors**: Hongqin Lyu, Yonghao Wang, Zhiteng Chao, Tiancheng Wang, Huawei Li  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.10107v1  

#### Abstract
Assertion-based Verification (ABV) is essential for ensuring that hardware designs conform to their intended specifications. However, existing automated assertion-generation approaches, such as LLM-based frameworks, often generate large numbers of redundant assertions, which significantly degrade si...

---

### 15. [SOMA: Efficient Multi-turn LLM Serving via Small Language Model](https://arxiv.org/abs/2605.11317)

**Authors**: Xueqi Cheng, Qiong Wu, Zhengyi Zhou, Xugui Zhou, Tyler Derr, Yushun Dong  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.11317v1  

#### Abstract
Large Language Models (LLMs) are increasingly deployed in multi-turn dialogue settings where preserving conversational context across turns is essential. A standard serving practice concatenates the full dialogue history at every turn, which reliably maintains coherence but incurs substantial cost i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SOMA: Efficient Multi-turn LLM Serving via Small Language Model

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Large Language Model (LLM)** 在多轮对话（multi-turn dialogue）场景中通常需要将完整的对话历史作为输入传递给模型，以保持上下文连贯性。这种做法虽然有效，但带来了显著的计算开销：
- **高延迟**（latency）
- **高内存消耗**
- **高昂的API成本**（尤其是调用大型闭源模型时）

现有方法在效率与响应质量之间难以平衡，例如：
- 单模型压缩历史的方法可能丢失关键信息；
- 多模型路由策略（如简单查询用小模型，复杂查询用大模型）存在切换开销且小模型泛化能力差。

---

### 提出的新方法：SOMA（Soft-prompts for lOcal Manifold Approximation）

SOMA 是一种高效的多轮 LLM 服务框架，其核心思想是利用早期对话轮次来构建一个“局部推理流形”（local reasoning manifold），并让一个小语言模型（surrogate SLM）在此局部区域内逼近大模型的行为。

#### 三阶段流程：
1. **Soft Prompt Tuning（软提示挖掘）**
   - 在早期对话阶段，固定大模型 $ F $ 和小模型 $ G $，仅优化一个可学习的 soft prompt $ P $。
   - 设计目标函数最大化 $ G $ 与 $ F $ 的语义分歧（semantic divergence），从而暴露小模型在当前对话上下文中最薄弱的方向。
   - 引入 **anti-degeneration regularizer** 防止提示坍缩为重复输出。

2. **Localized Fine-tuning（局部微调）**
   - 将挖掘出的 hardest cases（即最大分歧样本）用于对小模型进行轻量级 LoRA 微调。
   - 微调后的小模型不再依赖 soft prompts，实现 prompt-free 推理。

3. **Efficiency Inference（高效推理 + 回滚机制）**
   - 引入一个 **cosine gate** 判断是否满足切换条件（输出相似性和上下文接近度）。
   - 成功切换后，后续轮次由微调后的小模型处理，并使用压缩上下文（summary + 最近几轮）减少输入长度。
   - 实时监控语义漂移（drift detection），一旦检测到话题跳跃，则回滚至原始大模型重新初始化。

---

### 相比现有方法的优势
| 方面 | SOMA | 传统方法 |
|------|------|--------|
| 效率 | 后期使用小型模型 + 压缩上下文，大幅降低 token 数和延迟 | 每轮都需完整历史，成本随轮次增长 |
| 质量保留 | 局部适应确保小模型模仿大模型行为，而非通用微调 | 小模型直接替换易偏离原风格和逻辑 |
| 动态性 | 支持自动回滚，应对话题跳变 | 多数方法无动态恢复机制 |
| 理论支撑 | 提供 switching 条件、prompt 覆盖率、suboptimality 上界等理论分析 | 多为经验设计 |

---

## 2. 核心实验方法和设置

### 数据集
在六个真实世界的多轮对话基准上评估：
- **ShareGPT**：开放域人类-LLM 对话
- **ReMeDi**：医生-病人医疗咨询
- **Craigslist**：买卖双方谈判对话
- **Multi-Char**：多角色扮演对话
- **MATH**：数学解题任务（测试推理一致性）
- **MT-Bench**：多任务质量评测

> 所有数据均经过过滤，仅保留 **context-dependent** 类型对话（通过多个 LLM judge 投票确认）

---

### 实验设置
#### 模型配置
- **LLaMA 家族**：$ F = \text{LLaMA-3.1-70B}, G = \text{LLaMA-2-7B} $
- **Qwen 家族**：$ F = \text{Qwen-3-8B}, G = \text{Qwen-3-0.6B} $

#### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Original** | 每轮调用大模型 + 完整历史 |
| **Surrogate** | 每轮调用小模型 + 完整历史 |
| **History-Prefix** | 小模型接收大模型的历史输出，不微调 |
| **History-FT** | 在相同上下文-响应对上对小模型做 LoRA 微调 |
| **LLMLingua-2** | 使用压缩算法缩短历史再输入小模型 |
| **RouteLLM** | 学习路由器决定每轮使用哪个模型 |
| **Random-FT** | 使用与 SOMA 相同数量的数据，但随机采样训练样本（用于验证 soft prompt 选择的有效性） |

---

### 评估指标
| 指标 | 说明 |
|------|------|
| **Response Similarity** | 使用多个 LLM judge（GPT-OSS、DeepSeek-V3、Gemma2-27B）评估生成结果与 Original 的相似度（0–1 分） |
| **Exact Match (EM) Accuracy** | 在 MATH 数据集上衡量最终答案正确率 |
| **Token Usage** | 总消耗 token 数（反映 API 成本） |
| **Throughput & Latency** | 端到端吞吐量和响应时间 |
| **Drift Detection Rate** | 是否能及时识别话题变化并触发回滚 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 表1：LLaMA 家族上的平均响应相似度（↑越好）
| 方法 | Avg. Similarity |
|------|----------------|
| Surrogate | 75.1 ± 5.98 |
| History-Prefix | 84.8 ± 2.94 |
| History-FT | 90.8 ± 2.18 |
| RouteLLM | 92.2 ± 1.78 |
| **SOMA** | **93.1 ± 1.99** ✅ |

> SOMA 显著优于所有基线，在所有六项任务中均取得最高分。

#### 表2：MATH 数据集上的 Exact Match 准确率
| 方法 | LLaMA 家族 | Qwen 家族 |
|------|------------|-----------|
| Original | 48.34 ± 0.32 | 36.48 ± 0.41 |
| Surrogate | 19.20 ± 0.78 | 11.73 ± 0.64 |
| History-FT | 31.46 ± 0.87 | 22.57 ± 0.83 |
| RouteLLM | 33.88 ± 0.79 | 25.08 ± 0.71 |
| **SOMA** | **41.62 ± 0.66** ✅ | **31.14 ± 0.74** ✅ |

> SOMA 接近原始大模型表现，远超其他小模型方案，表明其不仅表面相似，还能保留深层推理能力。

---

### 与基线方法的对比结果
- **相比 Surrogate / History-Prefix**：SOMA 平均提升 **+18%** 相似度；
- **相比 History-FT**：尽管训练数据量相当，SOMA 仍高出约 **2.3%**，证明 soft prompt 成功挖掘了更具挑战性的训练样本；
- **相比 RouteLLM**：SOMA 更稳定，避免频繁切换带来的抖动；
- **效率方面**：在长对话中（>12轮），SOMA 可节省 **高达 37.2% 的 token 开销**（见 Table 3）。

---

### 消融实验结果（Ablation Study）

#### 图2b：组件消融（LLaMA 家族）
| 变体 | 性能下降幅度 |
|------|-------------|
| Full SOMA | 基准 |
| w/o Anti-degeneration Loss | ↓ 明显，尤其在复杂任务 |
| w/o Expectation-weighted Term | ↓ 中等 |
| w/o Both Components | ↓↓ 最大（特别是在 MATH 和 Multi-Char） |

> 结果表明：**anti-degeneration 正则项** 和 **expectation-weighted divergence** 对稳定 prompt 搜索和捕捉深层语义差异至关重要。

#### 图8：Warm-start 长度影响
- 简单任务（如 ShareGPT、Craigslist）只需 3–5 轮即可完成有效适配；
- 复杂推理或多角色任务（如 MATH、Multi-Char）需要更长 warm-start（8+轮）才能收敛；
- 过早切换会损害质量，过晚则浪费效率优势 —— 支持动态 switch 决策。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **多轮对话存在“长尾模式”**：前几轮承载大量上下文信息，后续轮次较短但高度依赖前期设定的状态。
2. ✅ **局部流形近似可行**：在一个 session 内，大模型的行为可以在局部被小模型有效拟合。
3. ✅ **soft prompt 是有效的探针工具**：它能系统性地揭示小模型在特定上下文下的弱点，指导高质量训练数据的选择。
4. ✅ **SOMA 实现效率与质量的双赢**：在中长对话中，可在几乎不损失质量的前提下显著降低成本。
5. ✅ **drift-aware rollback 机制有效**：面对突发话题转移，系统能快速检测并回滚，防止持续劣化解。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **小模型容量限制** | 当 small-large 模型差距过大时，某些高级能力无法完全复现 |
| **需访问 embedding 层** | soft prompt tuning 要求能操作小模型的嵌入空间，难以应用于纯黑盒 API |
| **依赖局部一致性假设** | 若对话频繁跳跃主题或 paraphrase 差异剧烈，局部适配效果下降 |
| **一次性 warm-start 成本** | 极短对话（<5轮）可能无法摊销初始化开销，得不偿失 |

---

### 未来工作方向
1. **改进 drift detection**：引入更强的话题建模或记忆机制，提高切换鲁棒性；
2. **降低部署门槛**：开发无需内部访问的 approximate mining 方法（如基于 API 输出反推）；
3. **支持多区域适应**：允许在同一会话中维护多个局部适配模块，适应多主题对话；
4. **跨会话知识迁移**：将在相似任务中学到的 soft prompt 或 LoRA 权重迁移到新会话，加速 warm-start；
5. **隐私保护增强**：结合 differential privacy 或联邦学习，在敏感领域安全使用；
6. **扩展至多模态场景**：探索视觉-语言或多模态对话中的局部代理服务。

---

> 🔗 **开源代码地址**：[https://github.com/LabRAI/SOMA](https://github.com/LabRAI/SOMA)

</details>

---

### 16. [Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding](https://arxiv.org/abs/2605.12185)

**Authors**: Yigeng Zhou, Wu Li, Yifan Lu, Yequan Wang, Xuebo Liu, Wenya Wang, Jun Yu, Min Zhang, Jing Li  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.12185v1  

#### Abstract
Large language models accumulate extensive parametric knowledge through pre-training. However, knowledge conflicts occur when outdated or incorrect parametric knowledge conflicts with external knowledge in the context. Existing methods address knowledge conflicts through contrastive decoding, but in...

---

### 17. [ReCoVer: Resilient LLM Pre-Training System via Fault-Tolerant Collective and Versatile Workload](https://arxiv.org/abs/2605.11215)

**Authors**: Ziyue Liu, Zhengyang Wang, Ruijie Zhang, Avinash Maurya, Hui Zhou, Paul Hovland, Sheng Di, Franck Cappello, Bogdan Nicolae, Zheng Zhang  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.11215v1  

#### Abstract
Pre-training large language models on massive GPU clusters has made hardware faults routine rather than rare, driving the need for resilient training systems. Yet existing frameworks either focus on specific parallelism schemes or risk drifting away from a failure-free training trajectory. We propos...

---

### 18. [MLCommons Chakra: Advancing Performance Benchmarking and Co-design using Standardized Execution Traces](https://arxiv.org/abs/2605.11333)

**Authors**: Srinivas Sridharan, Andy Balogh, Bradford M. Beckmann, Brian Coutinho, Louis Feng, Sheng Fu, Sanshan Gao, Mehryar Garakani, Taekyung Heo, David Kanter, Josh Ladd, Ziwei Li, Winston Liu, Changhai Man, Dan Mihailescu, Spandan More, Joongun Park, Ashwin Ramachandran, Vinay Ramakrishnaiah, Saeed Rashidi, Vijay Janapa Reddi, Puneet Sharma, Phio Tian, William Won, Hanjiang Wu, Huan Xu, Jinsun Yoo, Tushar Krishna  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.11333v1  

#### Abstract
The fast pace of artificial intelligence~(AI) innovation demands an agile methodology for observation, reproduction and optimization of distributed machine learning~(ML) workload behavior in production AI systems and enables efficient software-hardware~(SW-HW) co-design for future systems. We presen...

---

### 19. [CATS: Cascaded Adaptive Tree Speculation for Memory-Limited LLM Inference Acceleration](https://arxiv.org/abs/2605.11186)

**Authors**: Yuning Han, Yangchenchen Jin, Dylan Zhao, Jingwei Sun  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.11186v1  

#### Abstract
Auto-regressive decoding in Large Language Models (LLMs) is inherently memory-bound: every generation step requires loading the model weights and intermediate results from memory (e.g., High-Bandwidth Memory (HBM) for GPU servers), making throughput bottlenecked by memory bandwidth rather than compu...

---

### 20. [Learning, Fast and Slow: Towards LLMs That Adapt Continually](https://arxiv.org/abs/2605.12484)

**Authors**: Rishabh Tiwari, Kusha Sareen, Lakshya A Agrawal, Joseph E. Gonzalez, Matei Zaharia, Kurt Keutzer, Inderjit S Dhillon, Rishabh Agarwal, Devvrit Khatri  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.12484v1  

#### Abstract
Large language models (LLMs) are trained for downstream tasks by updating their parameters (e.g., via RL). However, updating parameters forces them to absorb task-specific information, which can result in catastrophic forgetting and loss of plasticity. In contrast, in-context learning with fixed LLM...

---

### 21. [WindINR: Latent-State INR for Fast Local Wind Query and Correction in Complex Terrain](https://arxiv.org/abs/2605.09511)

**Authors**: Yi Xiao, Qilong Jia, Hang Fan, Pascal Fua, Robert Jenssen, Xiaosong Ma, Wei Xue  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.09511v1  

#### Abstract
Many downstream decisions in complex terrain require fast wind estimates at a small number of user-specified locations and heights for a given forecast valid time, rather than another dense forecast field on a fixed grid. We present WindINR, a latent-state implicit neural representation framework fo...

---

### 22. [GriNNder: Breaking the Memory Capacity Wall in Full-Graph GNN Training with Storage Offloading](https://arxiv.org/abs/2605.11517)

**Authors**: Jaeyong Song, Seongyeon Park, Hongsun Jang, Jaewon Jung, Hunseong Lim, Junguk Hong, Jinho Lee  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.11517v1  

#### Abstract
Full-graph training of graph neural networks (GNNs) is widely used as it enables direct validation of algorithmic improvements by preserving complete neighborhood information. However, it typically requires multiple GPUs or servers, incurring substantial hardware and inter-device communication costs...

---

### 23. [The Illusion of Power Capping in LLM Decode: A Phase-Aware Energy Characterisation Across Attention Architectures](https://arxiv.org/abs/2605.11999)

**Authors**: Bole Ma, Ayesha Afzal, Jan Eitzinger, Gerhard Wellein  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.11999v1  

#### Abstract
Power capping is the standard GPU energy lever in LLM serving, and it appears to work: throughput drops, power readings fall, and energy budgets are met. We show the appearance is illusory for the phase that dominates production serving: autoregressive decode. Across four attention paradigms -- GQA,...

---

### 24. [Efficient LLM Reasoning via Variational Posterior Guidance with Efficiency Awareness](https://arxiv.org/abs/2605.11019)

**Authors**: Zizhao Chen, Yuying Li, Siting Lin, Lianxi Wang  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.11019v1  

#### Abstract
Although large language models rely on chain-of-thought for complex reasoning, the overthinking phenomenon severely degrades inference efficiency. Existing reinforcement learning methods compress reasoning chains by designing elaborate reward functions, which renders high-quality samples extremely s...

---

### 25. [gym-invmgmt: An Open Benchmarking Framework for Inventory Management Methods](https://arxiv.org/abs/2605.11355)

**Authors**: Reza Barati, Qinmin Vivian Hu  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.11355v1  

#### Abstract
Inventory-policy comparisons are often difficult to interpret because performance depends on the evaluation contract as much as on the policy itself. Differences in topology, demand regime, information access, feasibility constraints, shortage treatment, and Key Performance Indicator (KPI) definitio...

---

### 26. [Fast MoE Inference via Predictive Prefetching and Expert Replication](https://arxiv.org/abs/2605.11537)

**Authors**: Ankit Jyothish, Ali Jannesari, Aishwarya Sarkar, Joseph Zuber  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.11537v1  

#### Abstract
The Mixture of Experts (MoE) architecture has become a fundamental building block in state-of-the-art large language models (LLMs), improving domain-specific expertise in LLMs and scaling model capacity without proportionally increasing their computational overhead. However, MoE inference often suff...

---

### 27. [Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria](https://arxiv.org/abs/2605.08354)

**Authors**: Juanxi Tian, Fengyuan Liu, Jiaming Han, Yilei Jiang, Yongliang Wu, Yesheng Liu, Haodong Li, Furong Xu, Wanhua Li  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.08354v1  

#### Abstract
Aligning multimodal generative models with human preferences demands reward signals that respect the compositional, multi-dimensional structure of human judgment. Prevailing RLHF approaches reduce this structure to scalar or pairwise labels, collapsing nuanced preferences into opaque parametric prox...

---

### 28. [C2L-Net: A Data-Driven Model for State-of-Charge Estimation of Lithium-Ion Batteries During Discharge](https://arxiv.org/abs/2605.08653)

**Authors**: Khoa Tran, T. Nguyen-Thoi, Vin Nguyen-Thai, Duong Tran Anh, Hung-Cuong Trinh, Tri Le  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.08653v1  

#### Abstract
Accurate state-of-charge (SOC) estimation is critical for the safe and efficient operation of lithium-ion batteries in battery management systems (BMS). Although data-driven approaches can effectively capture nonlinear battery dynamics, many existing methods rely on long historical input sequences, ...

---

### 29. [AHD Agent: Agentic Reinforcement Learning for Automatic Heuristic Design](https://arxiv.org/abs/2605.08756)

**Authors**: Haoze Lv, Ning Lu, Ziang Zhou, Shengcai Liu  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.08756v1  

#### Abstract
Automatic heuristic design (AHD) has emerged as a promising paradigm for solving NP-hard combinatorial optimization problems (COPs). Recent works show that large language models (LLMs), when integrated into well-designed frameworks (i.e., LLM-AHD), can autonomously discover high-performing heuristic...

---

### 30. [Forge: Quality-Aware Reinforcement Learning for NP-Hard Optimization in LLMs](https://arxiv.org/abs/2605.08905)

**Authors**: Xiaozhe Li, Xinyu Fang, Shengyuan Ding, Yang Li, Linyang Li, Haodong Duan, Qingwen Liu, Kai Chen  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.08905v1  

#### Abstract
Large Language Models (LLMs) have achieved remarkable success on reasoning benchmarks through Reinforcement Learning with Verifiable Rewards (RLVR), excelling at tasks such as math, coding, logic, and puzzles. However, existing benchmarks evaluate only correctness, while overlooking optimality, name...

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

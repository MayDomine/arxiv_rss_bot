# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-17 06:09:12 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FreeBalance: Pre-Routing Online Moe Load Balancing via Residual Workload Prediction](https://arxiv.org/abs/2608.14205)

**Authors**: Pengfei Chen, Yize Wu, Shouxu Kuang, Ke Gao, Ling Li  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2608.14205v1  

#### Abstract
Load imbalance poses a major bottleneck to the efficiency of expert parallelism in distributed inference of Mixture-of-Experts (MoE) models. The most heavily loaded rank stalls global execution due to skewed routing distributions, directly increasing latency. While offline expert placement can allev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*FreeBalance: Pre-Routing Online MoE Load Balancing via Residual Workload Prediction*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在分布式 **Mixture-of-Experts (MoE)** 模型推理中，**专家并行（Expert Parallelism, EP）** 面临严重的 **负载不均衡（load imbalance）** 问题。由于路由分布（routing distribution）通常高度偏斜（skewed），部分设备（rank）成为“straggler”，拖慢整个系统的执行速度，导致设备利用率低下、端到端延迟增加。

现有在线负载均衡方法依赖于路由决策完成后的统计信息来调整专家映射（expert-device mapping），但这一过程发生在推理的关键路径（critical path）上，引入额外延迟，限制了其实际收益。

### 提出了什么新方法或新思路
本文提出 **FreeBalance** —— 一种**无损（lossless）的在线负载均衡框架**，其核心思想是：

> **通过残差表示（residual representation）提前预测下一层的专家负载，并在路由执行前启动专家权重迁移，从而将迁移开销与前置计算阶段（如 attention）重叠，避免影响关键路径。**

具体创新点包括：

- **残差工作负载预测（Residual Workload Prediction）**  
  利用相邻层之间的 **hidden representation** 高度相似的特性（源于 residual connection），使用前一层输出 $h_{l-1}$ 输入当前层的 router 来**提前预测**该层的专家负载分布。此预测轻量且无需额外训练。

- **预算化专家交换规划（Budgeted Expert-Swap Planning）**  
  基于预测负载，采用贪心策略选择最优的成对专家交换（pairwise expert swaps），并在通信带宽和 attention 计算时间窗口内控制迁移数量，确保迁移不会超出可隐藏的时间窗口。

- **与 pre-routing 计算重叠（Overlap with Pre-Routing Computation）**  
  将专家权重迁移过程与当前层的 **attention 计算阶段重叠**，充分利用长序列场景下 attention 占主导的延迟窗口，实现“免费”平衡。

- **保持原始路由语义（Lossless Inference）**  
  预测仅用于指导物理专家放置，**不影响最终的路由决策**，MoE 执行仍基于真实 router 输出，保证模型输出不变。

### 相比现有方法的优势
| 维度 | 现有方法（如 EPLB, UltraEP） | FreeBalance |
|------|-------------------------------|-----------|
| **时机** | 路由后才开始迁移（关键路径上） | 路由前预测并启动迁移（非关键路径） |
| **开销可见性** | 迁移延迟直接增加推理时间 | 迁移被 attention 阶段完全掩盖 |
| **适应性** | 多为离线或反应式（reactive） | 在线、主动式（proactive）、每层自适应 |
| **是否改变输出** | 否（多数） | 否（严格保持原路由） |
| **适用场景** | 固定任务或缓慢变化负载 | 动态多任务、批间负载突变 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LongBench (Bai et al., 2024)**：作为主基准测试套件，包含 19 个子集，覆盖多种长上下文能力：
  - 单文档问答（NarrativeQA）
  - 多文档问答（HotpotQA, 2WikiMQA）
  - 摘要生成（GovReport, MultiNews）
  - 少样本学习（TREC, TriviaQA）
  - 合成任务（PassageCount）
  - 代码补全（LCC, RepoBench-P）
- **Mixed Tasks**：人工构造的动态多任务负载，模拟真实服务中连续批次来自不同 LongBench 子集的情景，用于评估跨任务负载漂移下的鲁棒性。

### 实验设置
- **模型**：
  - **Qwen3-30B-A3B-Instruct-2507**：128 专家，top-8 激活，EP=8，每 rank 16 专家
  - **Moonlight-16B-A3B-Instruct**：64 专家，top-6 激活，EP=8，每 rank 8 专家
- **硬件**：
  - 8×NVIDIA A800-SXM4 GPU，通过 NVLink 连接
  - 使用 **expert parallelism (EP=8)**
- **输入配置**：
  - Batch size = 16
  - Input length 从 1K 到 8K tokens 变化
- **运行方式**：
  - 每个配置预热一次，测量三次取平均值

### 评估指标
- **端到端预填充延迟（End-to-end prefill latency / TTFT）**：越低越好
- **最大/平均 rank 负载比（max-to-mean rank-load ratio）**：衡量负载均衡程度，理想值为 1
- **预测质量指标**：
  - $h_{l-1}$ 与 $H_l$ 的 cosine similarity
  - router logits 的 cosine similarity
  - top-k hit rate（早期预测恢复最终 top-k 分配的比例）
- **规划质量**：执行后降低实际不平衡的比例

### 基线方法对比
- **Vanilla**：固定专家映射，无负载均衡
- **Vanilla + FreeBalance**：本文方法应用于 vanilla 设置
- **EPLB**：基于历史统计的离线负载均衡器（DeepSeek-AI, 2025）
- **EPLB + FreeBalance**：结合历史映射与 FreeBalance 的在线优化

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### ✅ 负载均衡改善
- FreeBalance 将 **max-to-mean rank-load ratio 平均降低 32.8%**  
  （从 Vanilla 的 ~2.0 下降到 ~1.35）

#### ✅ 推理延迟降低
- 在 Qwen3-30B 上，**平均减少 end-to-end prefill latency 13.1%**
- 在长序列（8K tokens）下，**TTFT 减少达 23.3%**

#### ✅ 开销完全隐藏
- 每层平均隐藏 **5.1 个专家的迁移开销**
- 若不重叠，这些迁移将占关键路径延迟的 **8.5%**

#### ✅ 预测准确性高
| 指标 | 数值范围 |
|------|--------|
| Hidden state cosine similarity | 0.7116 – 0.9316 |
| Router logits cosine similarity | **0.9896 – 0.9952**（极高一致性） |
| Top-k hit rate | 0.7419 – 0.8494 |

> 表明尽管 hidden states 差异较大，但 **router logits 对输入扰动鲁棒性强**，适合用于轻量预测。

#### ✅ 规划有效性
- **95% 的层**在应用 swap plan 后实现了更低的实际负载不平衡
- 成功验证了预算化规划的有效性和稳定性

### 与基线方法的对比结果（以 Qwen3-30B 为例）
| 场景 | Vanilla (s) | FreeBalance (s) | 减少 |
|------|------------|----------------|------|
| NarrativeQA | 147.6 | 126.6 | **-14.2%** |
| TriviaQA | 175.9 | 112.1 | **-36.3%** ⬅️ 最大收益 |
| LCC (Code) | 62.3 | 55.1 | **-11.6%** |
| Mixed Tasks | 83.1 | 78.0 | **-6.1%** |

> 注意：即使已有 EPLB 的情况下，**FreeBalance 仍能进一步加速**，说明其与离线方法正交互补。

### 消融实验结果（隐含分析）
- **迁移数量 vs 性能增益**（图4）：
  - 固定每层交换数无法达到最优
  - FreeBalance 自适应调整交换数量（1.82–2.68），在不同序列长度下均优于固定策略
- **序列长度敏感性**（表3）：
  - 序列越长 → attention 时间越长 → 可隐藏的迁移越多 → 收益越大
  - 1K tokens 时提速 ~11%，8K 时达 ~23.3%

---

## 4. 关键结论和发现

### 主要发现
1. **残差连接提供了高质量的跨层 workload 预测信号**，使得在路由前进行负载预测成为可能。
2. **pre-routing 阶段（尤其是 attention）是一个天然的“免费”窗口**，可用于隐藏昂贵的专家迁移操作。
3. **在线负载均衡不必牺牲推理正确性**，FreeBalance 实现了 **zero-output-change 的加速**。
4. **动态多任务场景下，静态或历史驱动的方法存在滞后性**，而 FreeBalance 能逐层响应即时负载变化。
5. **长序列推理中，FreeBalance 的优势更加显著**，因其提供了更长的 overlap 窗口。

### 方法的局限性
- **依赖 residual similarity**：若模型架构改变（如无残差连接），预测效果可能下降。
- **仅适用于 transformer-based MoE 架构**，对其他结构泛化能力未知。
- **假设通信带宽稳定**，在复杂网络拓扑或拥塞环境下可能难以准确预算。
- **未处理专家复制（replication）或分片（sharding）**，仅支持整专家迁移。

### 未来工作方向
- 扩展至 **decoder-only 流水线中的 decode 阶段**，实现实时 token 级负载调整。
- 结合 **expert replication 或 partial migration** 以进一步提升灵活性。
- 探索 **更高效的预测器压缩机制**，降低 early router invocation 的计算成本。
- 将该思想推广至其他稀疏激活结构（如 Block-Sparse Transformers）。

--- 

> **总结一句话**：  
> FreeBalance 通过“**用残差预测换时间窗口，用时间窗口藏迁移开销**”，首次实现了**非侵入式、零损失、完全隐藏开销的在线 MoE 负载均衡**，为高效分布式大模型推理提供了新范式。

</details>

---

### 2. [Robust Dual-Model Collaborative Random Vector Functional Link Network](https://arxiv.org/abs/2608.13628)

**Authors**: A. Quadir, A. Rahaman, Mushir Akhtar, M. Tanveer  
**Category**: cs.LG  
**Published**: 2026-08-17  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2608.13628v1  

#### Abstract
Random vector functional link (RVFL) networks are lightweight and fast neural models that offer efficient training and strong generalization through randomized hidden-layer weights and direct input-output connections. However, conventional RVFL models are sensitive to noisy labels, outliers, and imb...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Robust Dual-Model Collaborative Random Vector Functional Link Network 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Random Vector Functional Link (RVFL)** 网络虽然具有训练速度快、结构简单等优点，但在面对现实世界中的以下挑战时表现不佳：
- **标签噪声（label noise）**：训练样本中存在错误标注。
- **异常值（outliers）**：数据分布中存在偏离正常模式的样本。
- **类别不平衡（imbalanced data）**：不同类别的样本数量差异大。

这些问题会导致模型泛化能力下降，鲁棒性不足。

---

### 提出了什么新方法或新思路
本文提出了一种新的鲁棒模型：  
**Kernel Risk-Sensitive Mean p-Power based RVFL (KRPRVFL)**，其核心创新包括：

1. **引入 Kernel Risk-Sensitive Mean p-Power (KRP) 准则作为损失函数**  
   - 替代传统 RVFL 中的最小二乘（least-squares）目标。
   - KRP 在再生核希尔伯特空间（RKHS）中定义，对异常值和噪声样本更不敏感。
   - 能够自适应地降低被污染样本在训练过程中的影响权重。

2. **采用协同学习机制（collaborative learning mechanism）**  
   - 允许模型组件之间进行动态交互，提升预测一致性与稳定性。
   - 增强了在复杂、高噪声环境下的适应能力。

3. **利用核诱导特征映射（kernel-induced feature mapping）捕捉非线性关系**  
   - 无需手动选择隐藏层大小或进行复杂的迭代优化。
   - 维持了 RVFL 的高效性和可扩展性。

---

### 相比现有方法的优势
| 特性 | 传统 RVFL 及变体 | KRPRVFL |
|------|------------------|---------|
| 对噪声/异常值鲁棒性 | 弱（均方误差易受干扰） | 强（KRP 抗噪能力强） |
| 是否需要调参隐藏层 | 是（需设定节点数 N） | 否（通过 kernel 隐式处理） |
| 训练效率 | 快（闭式解） | 快（迭代更新但收敛迅速） |
| 泛化性能 | 一般 | 显著提升（尤其在噪声环境下） |

> ✅ **优势总结**：KRPRVFL 在保持 RVFL 高效性的同时，显著增强了对噪声、异常值和不平衡数据的鲁棒性，适用于更具挑战性的实际分类任务。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 来源于两个公开基准库：
  - **UCI Machine Learning Repository**
  - **KEEL Dataset Repository**
- 总共使用了 **37 个 benchmark 数据集**，涵盖多种领域（医疗、金融、生物、工程等），如：
  - `breast_cancer`, `heart_hungarian`, `ionosphere`, `vehicle`, `yeast` 等。

此外，在补充材料中还进行了：
- **添加人工标签噪声的实验**（5% ~ 40% 的噪声比例）
- 敏感性分析（activation function, D, N, p, u 参数变化）

---

### 实验设置和评估指标

#### 实验配置
- **操作系统**：Windows 11
- **硬件平台**：Intel Xeon Gold 6226R @ 2.90GHz, 256GB RAM
- **编程语言**：Python 3.11
- **数据划分**：70% 训练，30% 测试
- **超参数调优**：网格搜索 + 五折交叉验证（five-fold CV）

#### 超参数范围
| 参数 | 取值范围 |
|------|--------|
| 正则化系数 $ D $ | $ \{10^{-5}, ..., 10^5\} $ |
| 风险敏感参数 $ u $ | $[1, 10]$ |
| 功率参数 $ p $ | $[20, 21, ..., 210]$ |
| 隐藏节点数 $ N $ | $3$ 到 $203$ |
| 激活函数（Activation Function） | SELU, ReLU, Sigmoid, Sine, Hardlim, Tribas, Radbas, Sign, Leaky ReLU |

#### 评估指标
- **Classification Accuracy (Acc)**：主评价指标
- **Average Rank**：基于各数据集上的排名平均
- **统计检验**：
  - Friedman test（检验是否存在显著差异）
  - Nemenyi post-hoc test（成对比较）

---

### 基线方法对比
与以下主流 RVFL 和 ELM 类模型进行比较：
1. **RVFL** [4]
2. **ELM** (Extreme Learning Machine) [27]
3. **GB-RVFL** [5]
4. **GE-GB-RVFL** [5]
5. **CRVFL** (Complex-valued RVFL) [16]
6. **ACRVFL** [16]

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table I）

| 模型 | 平均准确率 (Avg Acc) | 平均排名 (Avg Rank) |
|------|--------------------|-------------------|
| **KRPRVFL (Ours)** | **85.20%** | **2.03** |
| RVFL | 81.55% | 3.88 |
| ELM | 80.81% | 4.34 |
| GB-RVFL | 79.62% | 4.07 |
| GE-GB-RVFL | 78.17% | 4.38 |
| CRVFL | 75.86% | 4.95 |
| ACRVFL | 76.62% | 4.36 |

> 🔺 KRPRVFL 在 **所有基线模型中排名第一**，且平均排名远低于其他模型（越小越好）。

---

### 与基线方法的对比结果

#### 统计显著性检验
- **Friedman Test**：
  - 检验统计量 $ F_F = 8.35 $
  - 自由度为 $(6, 216)$，临界值 $ F_{0.05}(6,216) = 2.1407 $
  - 因为 $ 8.35 > 2.1407 $，**拒绝零假设** → 存在显著性能差异。

- **Nemenyi Post-hoc Test**
  - 关键差值（Critical Difference, C.D.）= **1.4811**
  - KRPRVFL 与其他模型的平均排名差均大于 C.D.：
    - vs RVFL: 1.85 (>1.48)
    - vs ELM: 2.31 (>1.48)
    - vs GB-RVFL: 2.04 (>1.48)
    - ...
  - ✅ **KRPRVFL 在统计上显著优于所有基线模型**

---

### 添加标签噪声后的实验结果（见 Table S.I）

在多个数据集上注入 **5%–40% 的标签噪声**后，性能如下：

| 模型 | Overall Average Acc (含噪声) |
|------|----------------------------|
| **KRPRVFL** | **80.61%** |
| RVFL | 75.92% |
| ELM | 74.18% |
| GB-RVFL | 75.96% |
| GE-GB-RVFL | 74.25% |
| CRVFL | 69.68% |
| ACRVFL | 69.47% |

> 📌 在高噪声场景下（如 ionosphere 加 40% 噪声），KRPRVFL 仍能维持约 65% 准确率，而多数基线模型已降至 50% 左右。

---

### 消融实验 / 敏感性分析（Supplementary Material）

#### (1) 激活函数（Activation Function）的影响（Fig. S.1）
- 多数激活函数下性能稳定。
- **Radbas** 和 **ReLU-family** 表现最佳。
- 对 SELU 在某些数据集上有轻微下降，但整体鲁棒。

#### (2) 正则化参数 $ D $ 与增强节点数 $ N $（Fig. S.2）
- 中等规模的 $ D $ 和 $ N $ 表现最优。
- 极端小值导致欠拟合，过大则无明显增益。
- 表明模型在合理范围内易于调参。

#### (3) 风险敏感参数 $ u $ 与功率参数 $ p $（Fig. S.3）
- 中间取值（如 $ u \in [3,7], p \in [50,150] $）效果最好。
- 过小或过大都会损害性能。
- 验证了 KRP 设计的有效性。

> ❗ 结论：KRPRVFL 在较宽的超参数区域内保持高性能，具备良好的实用性和可调性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **KRP 准则有效提升了 RVFL 的鲁棒性**  
   - 通过风险敏感机制抑制噪声样本影响，使模型聚焦于真实数据分布。

2. **协同学习 + 核映射增强了表达能力**  
   - 无需显式设计网络深度或宽度，即可捕获非线性结构。

3. **KRPRVFL 在标准与噪声环境下均表现最优**  
   - 不仅在干净数据上领先，在高达 40% 标签噪声下依然稳健。

4. **统计检验证实性能提升具有显著意义**  
   - Friedman 与 Nemenyi 检验表明其优势不是偶然。

---

### 方法的局限性
1. **依赖核函数与参数选择**  
   - 尽管做了敏感性分析，但最优 $ u, p, \sigma $ 仍需调参。
   - 不同数据集可能需要不同的 kernel bandwidth。

2. **计算开销略高于原始 RVFL**  
   - 虽然仍较快，但由于涉及迭代优化与 kernel matrix 计算，不如标准 RVFL 完全闭式求解快。

3. **更适合中小规模数据集**  
   - 文中未测试超大规模或流式数据场景。

---

### 未来工作方向
1. **扩展至大规模与流式数据学习**
2. **自动化的 adaptive kernel selection 机制**
3. **多任务学习（multi-task learning）与多视图学习（multi-view learning）集成**
4. **结合 deep RVFL 构建层次化特征提取框架**

---

> ✅ **总体评价**：KRPRVFL 是一种兼具 **速度、简洁性与强鲁棒性** 的新型 RVFL 扩展，在处理现实世界脏数据方面展现出巨大潜力，是轻量级神经网络迈向可靠 AI 的重要一步。

</details>

---

### 3. [The Integer Alibi: Localizing Cross-Kernel Divergence in INT8-Quantized LLM Inference](https://arxiv.org/abs/2608.13756)

**Authors**: Teng-Ruei Chen  
**Category**: cs.LG  
**Published**: 2026-08-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.13756v1  

#### Abstract
Two GPU kernels implementing the same scaled INT8 GEMM interface are usually treated as interchangeable. We test that assumption: holding the checkpoint, prompts, hardware, inference engine, decoding, and quantization configuration fixed, we swap only the INT8 linear kernel (CUTLASS versus Triton) i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*The Integer Alibi: Localizing Cross-Kernel Divergence in INT8-Quantized LLM Inference*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文挑战了一个在量化推理中被广泛接受的“隐含契约”（implicit contract）：**不同的 GPU kernel 实现（如 CUTLASS 和 Triton）在输入相同时应产生数值上可忽略差异的输出**。作者发现，在 INT8 量化的大语言模型（LLM）推理中，这一假设并不成立——即使所有其他条件（checkpoint、prompt、硬件、引擎等）完全固定，仅更换 INT8 GEMM kernel 就会导致完全不同的生成序列。

### ✅ 提出的新方法与新思路
作者提出了 **“整数不在场证明”（Integer Alibi）** 这一核心概念，并基于此构建了一套完整的分析框架：

- **Integer Alibi（整数不在场证明）**：  
  在 INT8×INT8 GEMM 中，若 dot product 不会溢出 INT32（即 $ K \leq 131,071 $），则其累加过程是**精确且顺序无关的**。这意味着任何跨 kernel 差异**不可能来自累加阶段**，而必须发生在后续的 scale 应用和输出舍入（epilogue）阶段。

- **Power-of-Two Probe（2 的幂次探测）**：  
  提出一种干预手段：将 weight scales 强制设为最接近的 $ 2^n $ 形式。由于乘以 $ 2^n $ 在浮点中只是指数位移，不改变尾数，因此满足 `rnd(x * 2^n) == rnd(x) * 2^n`（在有限正常数范围内），从而使得 epilogue 舍入变得可交换，理论上应恢复 bit-identical 输出。

- **Conformance Procedure（一致性验证流程）**：  
  构建了一个包含 7 项检查的可执行验证流程（见 Table IV），用于判断两个 kernel 是否真正可互换。其中前五项为**可证伪的布尔断言**（falsifiable outcomes），而非依赖阈值的经验判断。

### ✅ 相比现有方法的优势
| 方面 | 本文优势 |
|------|--------|
| **机制隔离能力** | 现有研究多比较整个 backend 或平台，混杂多种因素；本文通过“单一 kernel 替换”实现精细控制，精准定位到 epilogue 阶段。 |
| **理论支撑强度** | 利用整数算术的精确性和结合律作为“不在场证明”，这是 FP 运算无法提供的强保证。 |
| **验证方式严谨性** | 采用 pre-registration、digest-pinned 容器、artifact manifest 审计等方式，确保实验可复现、可审计。 |
| **实用性导向** | 不止于发现问题，还提出可操作的 conformance 流程，供工程部署前进行 kernel interchangeability 检查。 |

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **模型**：Qwen3 系列，具体为：
  - Qwen3-1.7B（196 层 INT8 linear layers）
  - Qwen3-8B（252 层）
- **量化配置**：W8A8 INT8，对称权重量化（per-channel scales）、动态激活量化（per-token scales）
- **数据集**：WikiText-2 的 64 个确定性 320-token 窗口（pinned revision）
- **推理模式**：greedy decoding，生成长度为 64 或 256 tokens

### ⚙️ 实验设置
- **硬件**：单块 RTX 4090（sm_89），驱动版本 580.173.02
- **软件栈**：
  - vLLM 0.27.1（container digest 固定）
  - SGLang（作为第三方 backend 对照）
  - llm-compressor 0.13.0 用于量化
- **对比 kernel**：
  - `CutlassInt8ScaledMMLinearKernel`
  - `TritonInt8ScaledMMLinearKernel`
  - 通过环境变量 `VLLM_DISABLED_KERNELS` 控制选择
- **控制措施**：
  - 所有 checkpoint、prompt、quantization config、decoding strategy 固定
  - 使用 **pre-registration** 锁定协议、预测列表、分析计划（SHA-256）
  - 每次运行生成 **manifest**，记录 kernel selection 日志，实现“治疗仅差 kernel”的**可审计性**

### 📊 评估指标
| 指标类型 | 具体指标 |
|--------|--------|
| **bitwise 一致性** | 输出是否完全相同（signed zero 敏感） |
| **数值误差** | finite 元素间的 ulp distance（bf16 bit pattern 距离） |
| **token-level 差异** | 序列匹配率、flip rate（teacher-forced 下单步差异） |
| **logit margin 分析** | ROC-AUC、calibration curve、Brier score（预测 flip 风险） |
| **divergence signature** | 差异随 reduction depth $ K $ 的变化趋势 |

### 🔁 基线方法对比
- **主对比**：CUTLASS vs. Triton（同属 vLLM 内部不同 kernel 实现）
- **扩展对比**：
  - SGLang vs. vLLM（跨 inference engine）
  - INT8 vs. FP8 GEMM（不同 accumulator 类型）
- **理想对照**：同一 kernel 在冷启动下的 self-consistency（bit-identical）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）端到端生成结果（Table III）
| 比较项 | 1.7B 结果 | 8B 结果 |
|-------|----------|--------|
| 同 kernel 冷启动一致性 | 8/8 完全一致 | —— |
| CUTLASS vs. Triton（real scales） | **0/8 完全无一致序列** | **0/16 无一致序列** |
| CUTLASS vs. Triton（pow2 scales） | **8/8 完全一致** | **16/16 完全一致** |
| SGLang vs. vLLM（real scales） | 0/8 不一致 | —— |
| SGLang vs. vLLM（pow2 scales） | 1/8 仅一个一致 | —— |

> 💡 **说明**：即使两个实现都完全确定性，它们仍是**不同的确定性函数**。

#### （2）层级别验证结果（Table II）
| 指标 | Qwen3-1.7B | Qwen3-8B |
|------|-----------|---------|
| 总层数 | 196 | 252 |
| 预测 accumulator < $ 2^{24} $ | 196/196 | 252/252 |
| 最小 headroom（bits） | 2.82 | 3.36 |
| pow2 scales 下 bit-identical 层数 | **196/196** | **252/252** |
| real scales 下 bit-identical 层数 | 8/196 | 10/252 |
| 最大 finite ulp distance | **1** | **1** |
| 超过 1 ulp 的元素数 | 0 | 0 |

> ✅ 所有层均满足 accumulator 精确表示条件 → **Integer Alibi 成立**

#### （3）teacher-forced 单步分析（Figure 3）
- 在 64 prompts × 256 positions = 16,384 个位置上进行强制回放
- **总 flip 数**：769 / 16,384 ≈ **4.7%**
- **flip 集中于低 logit margin 区域**：
  - 最低 margin decile：flip rate 达 **34%**
  - margin > 9 时基本无 flip
- **logit margin 预测 flip 的能力**：
  - ROC-AUC = **0.942**（prompt cluster CI: [0.935, 0.949]）
  - Calibration Brier Score = 0.0352（优于常量基准 0.0440）

#### （4）reduction depth 影响对比（Figure 2）
| 特征 | INT8（CUTLASS vs. Triton） | FP8（CUTLASS vs. torch._scaled_mm） |
|------|----------------------------|----------------------------------|
| 差异比例 | ~5 ppm（parts per million） | 从 8% 增至 53% |
| 是否随 $ K $ 增长 | ❌ 无显著增长（$ \chi^2 = 3.8 $） | ✅ 显著增长 |
| 最大 ulp distance | **≤1**（始终） | 高达数千倍（tail 很重） |
| 拟合趋势 | 无积累机制（符合预期） | 符合 $ 1 - \exp(-cK^\alpha), \alpha \approx 0.518 $ |

> ✅ 支持结论：**INT8 accumulator 是 order-free 的，而 FP8 不是**

---

## 4. 关键结论和发现

### 🔍 主要发现
1. **Kernel 不可互换性真实存在**：  
   在严格控制下，仅更换 INT8 GEMM kernel（CUTLASS ↔ Triton）即可导致**所有生成序列完全不同**（0/64 匹配），尽管每个实现自身高度稳定。

2. **差异根源在于 Epilogue，而非 Accumulator**：  
   “Integer Alibi” 成功将 blame 定位到 scale 应用和输出舍入阶段，排除了 accumulator 的嫌疑。实验证明：
   - accumulator 完全精确（INT32 相同）
   - 所有 accumulator < $ 2^{24} $，可无损转为 float32
   - real-scale 差异最大仅为 **1 ulp（bf16 spacing）**

3. **Power-of-Two Scale 可修复一致性**：  
   将 weight scales 改为最近的 $ 2^n $ 后，**端到端输出恢复 bitwise 一致**（8/8 和 16/16 匹配），验证了 epilogue 是唯一自由度。

4. **Flip 可被 Logit Margin 高效预测**：  
   单步 token flip 几乎全部集中在 logit margin 小的位置，margin 本身即可作为 flip 风险的强预测器（AUC > 0.94）。

5. **格式决定 reproducibility 能力边界**：  
   - **INT8**：具备实现 near-bitwise interchangeability 的数学基础
   - **FP8/FLOAT**：因累加过程固有的 order-dependence，难以避免深度相关的累积误差

---

### ⚠️ 方法的局限性（Limitations）
1. **范围有限**：仅测试 sm_89 GPU、Qwen3 模型族、vLLM 引擎、两种 kernel。
2. **未解决 same-arm regime variance**：prefill 与 decode 模式间仍有 2.9% 的 token flip（pow2 下仍存），来源未定位（可能是 attention/KV-cache/normalization）。
3. **PoW2 并非部署方案**：rounding scales to $ 2^n $ 会影响模型精度，本文未评估其对 accuracy/calibration/performance 的影响。
4. **whole-model 无 bitwise 保证**：alibi 仅覆盖 INT8 GEMM，attention、RoPE、unquantized head 等仍为 float。
5. **缺乏正例测试（no positive control）**：所有 conformance checks 均通过，但未在故意引入缺陷的 kernel 上测试其检出能力。
6. **decode regime 缺少 layer-level 验证**：当前 capture 使用 M=512，未覆盖 M=1 的 decode 场景。

---

### 🔮 未来工作方向
1. **定位 same-arm 差异源**：开发 first-divergence instrumentation，追踪 prefilled 与 incremental decode 的首次分歧模块。
2. **评估 PoW2 的实际代价**：系统评测 power-of-two scaling 对模型 accuracy、calibration、throughput 的影响，判断其是否可作为实用 mitigation。
3. **扩展到更多格式与硬件**：验证 exact accumulator vs. float accumulator 的对比是否在其他 GPU 架构（如 Hopper）、模型家族、以及 W4A8 等低比特格式中成立。
4. **推广 conformance procedure**：将本文提出的 7 项检查封装为通用工具，支持其他 kernel 对（如 FlashAttention variants）的一致性验证。
5. **构建 margin-based verification gate**：利用 margin-to-flip 的高预测性，设计轻量级运行时监控机制，仅在高风险 token 上触发更严格的校验。

---

> **最终结论**：  
> 本文并未宣称量化推理“脆弱”，而是指出：**对于 INT8 路径，kernel interchangeability 是一个可以被形式化、检验并修复的问题**。通过“Integer Alibi + Power-of-Two Probe + Conformance Procedure”三部曲，作者将一个经验性差异转化为可验证的机制分析，为高可靠 LLM 推理系统提供了新的工程实践标准。

</details>

---

### 4. [Second Thought: Reasoning in Parallel as LLM Agents Act and Observe](https://arxiv.org/abs/2608.13667)

**Authors**: Zhensu Sun, Chengran Yang, Yunbo Lyu, Jieke Shi, David Lo  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.13667v1  

#### Abstract
LLM agents in the ReAct paradigm alternate between reasoning, acting, and observing, but deliberate reasoning is confined to the Thought phase: while the agent serializes an action and waits for the environment, its reasoning is frozen. We identify this recurring interval for Action and Observation ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Second Thought: Reasoning in Parallel as LLM Agents Act and Observe**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
在 ReAct 范式的 LLM Agent 中，推理（Thought）仅发生在 `Thought` 阶段，而在 `Action` 和 `Observation` 阶段，模型处于“推理空闲”状态（reasoning idle window），无法进行任何思考。这种串行流程导致：
- 推理被限制在关键路径上，增加延迟；
- 无法利用等待环境响应的时间进行额外推理。

该论文指出这一 **Action-Observation 间隔是一个未被利用的并行推理窗口**，提出应在此期间注入额外推理以提升效率与准确性。

---

### 🚀 提出的新方法：**Second Thought**
- **训练无关（training-free）的推理框架**，无需修改模型本身。
- 在每个 `Thought` 结束后立即 **fork 出四个辅助推理分支**，与主流程并行运行：
  - **CHECK**：验证当前计划中的假设是否成立；
  - **RECALL**：从历史轨迹中召回关键约束或上下文；
  - **REHEARSE**：预演可能的下一步行动（如工具调用失败后的备选方案）；
  - **ALTERNATIVE**：生成替代策略以防当前路径失败。
- 所有分支输出为 **atomic thoughts**（原子化思维单元），即自包含、无前后依赖的小片段，支持中断后仍保留有效部分。
- 当环境返回 `Observation` 时，终止所有分支，收集已完成的 atomic thoughts 并追加到下一回合的输入中，供主 Agent 使用。

> 🔑 核心思想：将本用于“等待”的时间转化为“思考”时间，且不延长主路径的解码延迟。

---

### ⚖️ 相比现有方法的优势
| 对比维度 | 现有方法（如 Self-Consistency, Tree-of-Thought） | Second Thought |
|--------|---------------------------------------------|----------------|
| 并行方式 | 水平并行（horizontal branching）：探索多条独立路径 | 垂直并行（complementary reasoning）：扩展同一条路径的不同视角 |
| 合并机制 | 需要投票、排序、聚合等复杂决策 | 简单拼接即可，无需协调 |
| 是否影响关键路径 | 是，更多推理 = 更长延迟 | 否，推理发生在非关键路径的 idle window 内 |
| 中断容忍性 | 弱，中途停止可能导致无效结果 | 强，atomic thoughts 支持任意截断 |
| 实现成本 | 多数需模型微调或架构变更 | 完全推理时干预，无需训练 |

✅ 因此，Second Thought 实现了 **更高性价比的推理增强** —— 在几乎不增加 wall-clock 时间的前提下，显著减少总 turn 数和主路径 token 消耗。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在三个代表性 agentic benchmark 上评估，覆盖多种任务类型：

| 数据集 | 任务类型 | 描述 |
|------|--------|------|
| **SWE-Bench Pro** | 软件工程 | 在真实代码仓库中定位并修复 bug，通过测试用例判断成功 |
| **Terminal-Bench 2.1** | 终端操作 | 执行系统管理、构建脚本等命令行任务 |
| **T³-bench (banking)** | 多轮对话 + 工具调用 | 模拟银行客服场景，需检索政策文档并函数调用 |

---

### 🧪 实验设置与评估指标

#### 模型选择（3个不同家族的 reasoning LLMs）：
- **DeepSeek-V4-Flash**
- **Qwen3.6-Plus**
- **MiniMax-M3**

#### 评估指标：
| 指标 | 含义 |
|-----|------|
| **Pass@1** | 单次尝试下任务完成率（主要准确率指标） |
| **#OUT_main** | 主线程生成的 output token 总数（衡量主路径计算开销） |
| **#Turns** | 平均交互轮数（反映规划效率） |
| **Wall-clock latency** | 实际执行时间（秒），通过 replay 测量 |
| **API Cost** | 基于各提供商 token 价格估算的实际调用成本 |

#### 基线方法对比：
| 基线 | 说明 |
|-----|------|
| **base** | 原始 ReAct Agent，无额外推理 |
| **s1 (budget forcing)** | 将相同数量的额外推理 token 加入主线程的 Thought 阶段（作为 compute-matched 控制组） |

> 💡 注意：s1 不适用于 MiniMax-M3 和 T³-bench，因这些模型不支持延续未闭合的 thought。

---

## 3. 主要实验结果和性能指标

### 📊 整体性能表现（见 Table 1）
在 **9 个 model-benchmark 组合**中：

| 指标 | 表现 |
|------|------|
| **平均 turn 数** | 在全部 9 组中均下降（最大降幅达 13.7 轮） |
| **主线程 decoding token 数** | 在 6 组中显著降低（最多 ↓43%，平均 ↓~20%）；第7组基本不变 |
| **Pass@1** | 7 组无显著变化；2 组显著提升：
  - Terminal-Bench 2.1 + Qwen3.6-Plus: **+12.4 pts**
  - Terminal-Bench 2.1 + MiniMax-M3: **+10.2 pts**

> ✅ 表明 Second Thought 可在 **保持甚至提升准确率的同时，大幅压缩主路径资源消耗**。

---

### ⏱️ 实际延迟收益（replay 实验）
对 SWE-Bench Pro 进行配对重放实验（paired wall-clock replay）：

- **中位任务延迟降低 10.9%**（从 256.9s → 229.0s）
- 分解原因：
  - 主线程解码时间 ↓13.4%（对应 token ↓15.0%）
  - 工具执行时间 ↓6.0%（因 turn 数减少）

> ✅ 证明性能提升不是“账面优化”，而是真实的 **wall-clock 加速**。

---

### 🔍 消融实验（Ablation Study，见 Figure 3 & Table 2）

#### 单一分支效果（only-X）：
- **only-recall**：token 最少（19.7k），但 Pass@1 降至 46.0%（↓2.7pts），说明仅回忆不够；
- **only-rehearse / only-alternative**：表现接近完整版，但仍略差。

#### 缺失某一分支（w/o-X）：
| 移除分支 | 影响 |
|-------|------|
| **w/o RECALL** | Pass@1 ↓至 48.0%（最大跌幅），说明历史约束召回至关重要 |
| **w/o REHEARSE** | Pass@1 不变，但 #output tokens ↑10%（22.4k vs 20.3k），说明其主要作用是 **卸载主路径推理负担** |

> ✅ 四个分支各有分工，组合使用达到 Pareto 最优。

---

### 💰 成本分析（Table 3）
虽然引入 4 个并行分支会增加 API 成本（input prompt 开销为主）：
- 成本增幅：**+66.4% ~ +181.5%**（取决于 provider 的 cache 折扣）
- 但 **output token 成本差异极小（<$0.02/任务）**
- 若只保留最强分支（如 ALTERNATIVE），可将增量成本控制在 **+16.3% ~ +35.5%**

> ✅ 可根据预算灵活裁剪分支数量，在性能与成本间权衡。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **ReAct Agent 存在普遍的 reasoning idle window**，这是可被利用的并行推理机会。
2. **Second Thought 成功将这部分空闲时间转化为有效推理**，通过 four complementary branches 提供前瞻性与回顾性洞察。
3. 该方法实现了：
   - 显著减少交互轮数（↑效率）
   - 降低主路径 token 消耗（↓延迟、↓成本）
   - 在部分任务上显著提升 Pass@1（↑准确性）
4. 与 compute-matched baseline（s1）相比，**Second Thought 在所有适用设置下都取得更优的 Pass@1 与更低的顺序解码量（1.3× ~ 3.2× less）**。

---

### ⚠️ 局限性
1. **依赖较长的 Action-Observation 延迟窗口**：若工具响应极快（如 banking 场景），则辅助分支产出有限。
2. **增加 total API cost**：尽管主路径节省，但总请求量上升，尤其在 input-heavy 定价模式下。
3. **目前固定四分支设计**：未能动态适配任务需求（未来可做 adaptive forking）。
4. **不改变最终决策机制**：仍依赖主 Agent 自主整合信息，存在误读风险。

---

### 🔮 未来工作方向
1. **动态分支调度**：根据任务类型、idle window 长度自动启用/关闭某些 branch。
2. **跨任务知识迁移**：让辅助分支学习何时生成最有价值的 second thoughts。
3. **轻量化部署**：结合 speculative execution 或 cache-aware scheduling 降低额外开销。
4. **与其他并行范式融合**：例如将 Second Thought 与 Tree-of-Thought 结合，实现“水平+垂直”双重并行推理。

---

## ✅ 总结一句话
> **Second Thought 创造性地将 ReAct Agent 的“等待时间”变为“思考时间”，在不拖慢响应速度的前提下，通过 interruption-friendly 的并行辅助推理，实现了更高效、更鲁棒的任务解决能力。**

</details>

---

### 5. [Joint Optimization of Memory and Computing Frequency for Energy-Efficient DNN Inference](https://arxiv.org/abs/2608.13863)

**Authors**: Yunchu Han, Zhaojun Nan, Sheng Zhou, Zhisheng Niu  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.13863v1  

#### Abstract
Deep neural network (DNN) inference on mobile devices often incurs high latency and energy consumption due to limited computing and memory resources. To enable energy-efficient DNN inference, most existing studies focus on dynamic voltage and frequency scaling (DVFS) for adjusting the computing freq...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Joint Optimization of Memory and Computing Frequency for Energy-Efficient DNN Inference*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**移动设备上深度神经网络（DNN）推理高延迟和高能耗**的问题，指出当前大多数研究仅关注通过 **Dynamic Voltage and Frequency Scaling (DVFS)** 调整计算频率（如 CPU/GPU 频率），而**忽略了内存频率（memory frequency）对推理性能的重要影响**。

尽管已有研究表明内存访问是 DNN 推理中的瓶颈之一，尤其是在内存密集型模型中，但**内存频率与计算频率的联合优化尚未被系统研究**。

### ✅ 提出的新方法与创新思路
1. **联合优化框架**：
   - 首次提出将 **memory frequency、computing frequency、transmission power 和 bandwidth** 进行联合优化，以最小化移动设备的总能耗，并满足推理任务的截止时间（deadline）约束。
   
2. **建模与理论分析**：
   - 构建了一个基于实测数据的**真实 DNN 推理时延模型**，明确刻画了 memory frequency 和 computing frequency 对推理时间的影响。
   - 在局部推理场景下，推导出一个**近似最优的闭式解（near-optimal closed-form solution）**，适用于一般情况。
   - 在边缘推理场景下，给定带宽时，推导出**传输功率的最优闭式解**。

3. **低复杂度启发式算法**：
   - 设计了一种**多项式时间复杂度的启发式算法（Algorithm 1）**，通过贪心策略决定哪些设备执行本地推理、哪些进行边缘卸载，同时优化资源分配。

### ✅ 相比现有方法的优势
| 方面 | 本文方法 | 传统方法 |
|------|--------|---------|
| 优化维度 | 联合优化 memory + computing frequency + communication 资源 | 通常只优化 computing frequency 或仅考虑卸载决策 |
| 内存频率作用 | 显式建模并优化 memory frequency | 忽略或固定 memory frequency |
| 解法效率 | 提供闭式解和低复杂度启发式算法，适合实时部署 | 多依赖数值求解器，计算开销大 |
| 性能提升 | 最多降低 **10.4%** 的设备能耗 | 缺乏对 memory frequency 的利用导致能效次优 |

---

## 2. 核心实验方法和设置

### ✅ 数据集与模型
- **DNN 模型**：ResNet152 和 VGG19
- **数据集**：CIFAR-100
- **硬件平台**：NVIDIA Jetson TX1（典型边缘设备）
- **部署方式**：在 Jetson TX1 上实际部署模型并测量不同频率组合下的推理时间和功耗

### ✅ 实验设置
- **设备数量**：$ N = 12 $ 台移动设备
- **区域范围**：400m × 400m 正方形区域
- **总带宽**：$ B = 20 $ MHz，采用 OFDMA 分配
- **信道模型**：3GPP TR36.931 路损模型 $ h_n = 38 + 30\log_{10}(l_n) $（单位：dB）
- **传输功率范围**：[0.1, 1] W
- **目标**：最小化所有设备的总能量消耗，满足每个设备的任务 deadline 约束

### ✅ 评估指标
- **总能耗（Total Energy Consumption）**
- **推理延迟（Inference Latency）**
- **内存/计算频率配置**
- **卸载决策准确性**

### ✅ 基线方法对比
1. **Random**：随机选择设备进行本地推理（而非按最大传输功率移除）
2. **Only Compute [24]**：仅优化 computing frequency，memory frequency 固定为最大值
3. **No DVFS**：memory 和 computing frequency 均保持最大值（无动态调节）

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据
| 指标 | 结果 |
|------|------|
| **近似最优解 vs 最优解差距** | 在严格 deadline 下，性能差距不超过 **2.5%**（ResNet152）和 **2.3%**（VGG19） |
| **能耗降低幅度** | 相比 No DVFS 方法，平均节能达 **10.4%**<br>相比 Only Compute 方法，平均节能 **3.6%** |
| **内存频率调节效果** | 将 memory frequency 从 0.1GHz 提升至 1.6GHz：<br>- ResNet152 推理时间 ↓84%，能耗 ↓80%<br>- VGG19 推理时间 ↓93%，能耗 ↓92% |

### ✅ 与基线方法的对比结果（见 Fig. 5）
- 所有策略中，**本文提出的算法能耗最低**。
- 当带宽充足或设备数较少时，各方法差异缩小；但在资源受限场景下，本文方法优势显著。
- “Random” 策略表现最差，说明**合理选择卸载设备至关重要**。
- “Only Compute” 和 “No DVFS” 表现较差，验证了**联合优化 memory frequency 的必要性**。

### ✅ 消融实验结果（隐含于对比中）
- **关闭 memory frequency 优化 → 能耗上升 3.6%~10.4%**：表明 memory frequency 是节能的关键自由度。
- **不使用启发式调度 → 无法保证可行性**：原始 MINLP 问题难以直接求解，凸显所提算法的实用性。
- **闭式解在短 deadline 下逼近最优**：证明近似解在实际严苛场景中高度有效。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Memory frequency 对 DNN 推理性能影响巨大**，尤其对于 VGG19 等内存密集型模型，其调节带来的能效增益甚至超过 computing frequency。
2. **联合优化 memory 和 computing frequency 可显著降低能耗**，最多可达 **10.4%**，且无需额外硬件成本。
3. 所提出的 **near-optimal closed-form solution 在 deadline 较紧时几乎达到最优性能**（误差 < 2.5%），具有很强实用价值。
4. 启发式算法具有**多项式时间复杂度**，适合大规模系统部署。

### ✅ 方法的局限性
- 假设 MEC server 计算能力无限，忽略边缘侧拥塞风险。
- 采用二元卸载（binary offloading），未支持更细粒度的模型分割（如 layer-level partitioning）。
- 实验基于静态信道和确定性推理时间，未充分考虑无线环境波动和模型内部不确定性。
- 未探索 memory voltage scaling（仅 frequency），进一步节能潜力有待挖掘。

### ✅ 未来工作方向
- 扩展到 **multi-user MIMO 或干扰感知通信** 场景。
- 引入 **reinforcement learning** 框架处理动态环境和不确定推理时间。
- 支持 **partial offloading** 和 **DNN 模型自适应剪枝/量化** 与频率调节协同优化。
- 探索 **memory voltage + frequency 联合缩放（DVFS+MVFS）** 的跨层节能机制。

---

> 📌 **总结一句话**：  
> 本论文揭示了 **memory frequency 在 DNN 推理能效优化中的关键作用**，提出了首个联合优化 memory frequency、computing frequency 与通信资源的框架，通过理论分析与高效算法实现了高达 **10.4% 的能耗降低**，为边缘智能系统的绿色计算提供了新路径。

</details>

---

### 6. [Learning to Run Power Networks: Effective AlphaZero-inspired Topological Control](https://arxiv.org/abs/2608.14114)

**Authors**: Lukas Zetto, Benjamin Sch\"afer, Qiong Huang  
**Category**: cs.LG  
**Published**: 2026-08-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.14114v1  

#### Abstract
As the integration of volatile renewable energy sources increases the strain on modern power grids, the use of Reinforcement Learning (RL) for autonomous topological reconfiguration has emerged as a promising research field to keep strained grids stable and operational. Compared to traditional redis...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning to Run Power Networks: Effective AlphaZero-inspired Topological Control*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代电力系统在高比例**可再生能源**（RES）接入背景下，面临**电网拥塞加剧**和**级联故障风险上升**的挑战。传统的**redispatching**（如调整发电机出力）虽然有效，但成本高昂且未充分利用现有网络拓扑灵活性。

本文聚焦于**拓扑控制**（Topological Control），即通过**母线重组**（busbar reconfiguration）和**线路切换**（line switching）等操作来主动重配置电网结构，以缓解拥塞、提升稳定性。然而，该任务面临两大难题：
- **组合爆炸**：可能的拓扑动作空间巨大（IEEE-14系统有405个原始动作）
- **严格运行约束**：需满足物理潮流方程和安全边界

### 提出的新方法与思路
提出并系统评估了一种**AlphaZero-inspired**的**model-based强化学习框架**，结合**Monte Carlo Tree Search**（MCTS）进行前瞻式电网管理。其核心创新在于对以下设计维度的系统性消融研究：

- **奖励函数设计**：探索不同复杂度的奖励信号对MCTS效率的影响
- **观察空间密度**：分析输入特征数量对训练稳定性和收敛速度的作用
- **搜索引导机制**：比较基于先验策略、启发式规则和无引导的MCTS变体
- **动作空间剪枝策略**：评估对称性消除（SYM）、N-0、N-1等缩减方法的效果

### 相比现有方法的优势
- **超越纯RL方法**：相比主流的**PPO**等model-free RL方法，所提方法具备**前瞻能力**，能评估多步后果，避免级联失效
- **优于传统启发式**：相比固定规则或暴力搜索，MCTS能动态探索非直观但有效的“生存路径”
- **实现近完美性能**：在IEEE-14标准测试案例上达到**98.43%的存活率**，接近理论极限

---

## 2. 核心实验方法和设置

### 数据集与环境
- **平台**：使用 `Grid2Op` 框架模拟电力系统运行
- **电网模型**：**IEEE-14 bus system**（14个变电站、20条线路、5台发电机）
- **数据规模**：共使用 **1004个chronics**（时间序列场景），每个长达 **8064个时间步**
- **测试集划分**：保留10%作为独立测试集以评估泛化能力

### 实验设置与评估指标
#### 主要评估指标：
- **Avg. Steps Survived**：代理维持电网稳定运行的平均步数
- **Survivability (%)**：相对于最大时间步（8064）的比例
- **训练效率**：达到高性能所需的训练步数与计算资源

#### MCTS关键参数（Baseline）：
- **激活阈值**：当最大线路负载（max rho）超过 **98%** 时触发MCTS
- **每步MCTS模拟次数**：最多 **250次**
- **早期停止机制**：跳过200步后识别恢复节点，累计50个即终止搜索
- **回放缓冲区**：收集状态-动作对用于神经网络更新

### 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **DQN / PPO Redispatch** | Model-free RL | 连续控制发电出力 |
| **PPO Topological** | Model-free RL | 离散拓扑动作空间 |
| **PPO Curriculum (TTJS)** | Model-free + Curriculum | 教师-导师-学生分阶段训练 |
| **Rainbow PPO** | Optimized PPO | 当前SOTA model-free baseline |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 平均存活步数 | 存活率 |
|------|----------------|--------|
| Do Nothing | 847 | 10.5% |
| PPO Redispatch | 1670 | 20.7% |
| PPO Topological | 3250 | 40.3% |
| PPO Curriculum | 4130 | 51.2% |
| Rainbow PPO | **7403** | **91.80%** |
| AlphaZero Baseline | 7486 | 92.83% |
| **AlphaZero + D3QN-2022 Reward** | **7937** | **98.43%** ✅ |

> ✅ 所提方法在引入简单二元奖励后，显著超越所有基线，达到当前最优水平。

### 与基线方法对比结果
- **AlphaZero Baseline** 已经略优于 Rainbow PPO（92.83% vs 91.80%）
- 引入 **D3QN-2022 Binary Reward** 后，性能跃升至 **98.43%**
- 尽管训练成本更高（~40M步 vs ~100k步），但**最终可靠性更高**

### 消融实验结果

#### （1）奖励函数对比（Reward Shaping）
| 奖励函数 | 性能表现 |
|---------|----------|
| **Binary Survival Reward (D3QN-2022)** | ✔️ 最佳性能（7937步），训练最稳定 |
| AlphaZero Reward | 次优（7485步） |
| MaxRho / PPO Reward | 中等 |
| D3QN-2020 Composite / Loss-based | ❌ 表现差，收敛慢，噪声大 |

> 🔍 发现：**稀疏、目标一致的二元奖励**比复杂的多目标奖励更有效，避免子目标冲突。

#### （2）观察空间配置（Observation Density）
| 配置 | 特征内容 | 性能 |
|------|----------|------|
| **Minimal** | 仅线路负载（line loads） | ✔️ 收敛最快，最稳定（7340步） |
| Custom / Reduced / Complete | 包含电压、功率注入、时间等 | ⚠️ 性能相近但学习更慢、波动更大 |

> 🔍 发现：“**特征噪声**”（feature noise）会干扰学习；**最小化输入空间**反而提升性能。

#### （3）动作空间剪枝策略
| 策略 | 动作数 | 性能 |
|------|--------|------|
| **SYM (Symmetry Reduction)** | 203 | ✔️ 最快收敛，最高稳定性 |
| N-0 | 142 | ⚠️ 性能下降 |
| N-1 | 82 | ❌ 表现最差 |

> 🔍 发现：过度剪枝限制了寻找“非常规生存路径”的能力；**适度保留拓扑灵活性更有利**。

#### （4）MCTS搜索引导方式（Guidance Variants）
| 方式 | 描述 | 性能与效率 |
|------|------|------------|
| **No Guidance (IL Variant)** | 无先验策略或价值函数 | ✔️ **训练效率最高**，无需NN频繁推理 |
| Heuristic-based | 使用启发式函数评估叶节点 | ⚠️ 可行但较慢 |
| **Learned Q-Function** | 使用神经网络预测Q值 | ❌ 开销极大，仅完成20–30M步训练 |

> 🔍 发现存在“**引导悖论**”（Guidance Paradox）：在物理系统中，**无偏MCTS + 明确奖励** 比依赖不准确学习信号更高效。

---

## 4. 关键结论和发现

### 主要发现
1. **AlphaZero框架适用于拓扑控制**：MCTS能够有效处理组合动作空间，在安全关键系统中提供必要的**前瞻规划能力**。
2. **简约设计优于复杂建模**：
   - **Minimal Observation Space**（仅line loads）效果最佳
   - **Binary Survival Reward** 提供最清晰的目标信号
   - **无引导MCTS** 在初期训练中效率最高
3. **领域知识集成至关重要**：
   - 虽然纯RL不足，但结合**物理启发式**（如自动重连、参考拓扑回归）可大幅提升鲁棒性
4. **动作空间不宜过度压缩**：保留一定拓扑自由度有助于应对极端情况下的“逃生路径”。

### 方法的局限性
- **训练开销大**：需要约 **4000万训练步**，远高于PPO（~10万步）
- **可扩展性待验证**：目前仅在**IEEE-14**小系统上验证，大规模网络（如数百节点）仍具挑战
- **实时性受限**：MCTS每次决策需数百次模拟，可能难以满足超实时控制需求
- **黑箱决策缺乏解释性**：不利于电网运营商信任与采纳

### 未来工作方向
1. **提升训练效率**：
   - 结合**Curriculum Learning**，从简单故障逐步过渡到复杂场景
   - 引入**Options Framework**，将复杂开关序列抽象为高层动作
2. **增强可扩展性**：
   - 采用**Multi-Agent RL**（MARL）或**Hierarchical RL**，分区管理电网
   - 利用**State and Action Factorization**降低全局搜索负担
3. **提高透明度与可信度**：
   - 集成**Explainable RL**（XRL）技术（如SHAP、LIME），解释拓扑决策依据
   - 构建人机协同的**决策支持系统**
4. **融合物理先验**：
   - 在MCTS中嵌入**power flow-aware heuristics**，加速搜索收敛
   - 探索**hybrid control architecture**：拓扑控制优先，必要时才启用costly redispatch

---

> 📌 **总结一句话**：  
> 本文证明，一个**极简主义设计**的AlphaZero控制器——**最小观测 + 二元奖励 + 无引导MCTS**——在电网拓扑控制任务中实现了接近完美的生存率（98.43%），揭示了在物理系统中，“智能”更多来自**搜索机制与领域知识的结合**，而非复杂的端到端学习。

</details>

---

### 7. [Depth-Aware Sensitivity Analysis of Mixture-of-Experts Models via Magnitude-Based Expert Masking](https://arxiv.org/abs/2608.13565)

**Authors**: Pradeep Kumar Sharma, Shantanu Godbole, Hritvik Shrivastava  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.13565v1  

#### Abstract
Mixture-of-Experts (MoE) architectures scale large language models (LLMs) while preserving computational efficiency through sparse activation. Despite their widespread adoption, the relative importance of individual MoE layers remains insufficiently characterized, particularly for model compression....

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Depth-Aware Sensitivity Analysis of Mixture-of-Experts Models via Magnitude-Based Expert Masking**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
该论文针对 **Mixture-of-Experts (MoE)** 架构中一个尚未充分研究的关键问题：**不同深度的 MoE 层对专家剪枝（expert pruning）是否具有相同的敏感性？**  
尤其是在模型压缩场景下，能否在不显著影响输出质量的前提下，选择性地移除部分低重要性的专家？

现有工作多关注全局均匀稀疏化或基于路由频率的专家删除，缺乏对 **layer-wise depth-dependent sensitivity** 的系统性实证分析。

---

### 🚀 提出的新方法与新思路

1. **Depth-Aware Expert Masking（基于深度感知的专家掩蔽）**  
   提出通过 **magnitude-based expert masking** 对 MoE 模型进行逐层敏感性分析，即根据每个 expert feed-forward 权重的 L2 范数排序，屏蔽低幅值专家，并观察其对生成质量的影响。

2. **非破坏性干预机制**  
   不直接修改权重，而是通过将被屏蔽专家的 router logits 设为极大负值来阻止其被选中，从而实现可逆、可控的实验设计。

3. **分阶段、大规模评估框架**  
   在多个 prompt 规模（100 / 300 / 500）上执行多轮实验，确保结果具备统计稳健性和泛化能力。

4. **引入“very-late layer”压缩策略**  
   首次提出并验证了 **late layers（尤其是 very-late layers, 35–39）具有高度冗余性**，适合激进压缩。

5. **辅助 phase-aware routing 分析**  
   在相关模型上进行了 reasoning vs. answer 阶段的专家使用隔离分析，揭示 phase-specific expert usage 模式，为未来细粒度评分提供依据。

---

### 🔍 相比现有方法的优势

| 方面 | 本文优势 |
|------|--------|
| **分析粒度** | 细致到 layer-level 和 depth-region，而非全局统一策略 |
| **压缩效率** | 在更少专家屏蔽数量下实现更高输出保真度 |
| **实证基础** | 大规模（最高达 500 prompts）、跨阶段验证，支持强结论 |
| **方法论严谨性** | 包含机械完整性检查（mechanical integrity checks）、baseline-relative scoring、disjoint masking buckets |

---

## 2. **核心实验方法和设置**

### 📚 数据集
- 使用 **XLCoST benchmark** 中的跨语言代码翻译任务（如 C++ → Python）
- 固定确定性子集用于各阶段实验：
  - Early smoke tests: 10 prompts
  - Regional ablations: 100 prompts
  - Discovery sweep: 300 prompts
  - Held-out validation: 500 prompts

> 所有样本均为指令格式输入，仅要求目标语言代码输出。

---

### ⚙️ 实验设置

| 项目 | 设置详情 |
|------|---------|
| **Model** | Qwen3.6-35B-A3B<br>• 40 MoE layers<br>• 每层 256 experts<br>• top-8 routing<br>• 总参数 ~35B，激活参数 ~3.3B |
| **Hardware** | NVIDIA H100 80GB × 2 per run（共三台服务器） |
| **Precision** | bfloat16 |
| **Software Stack** | HuggingFace Transformers 5.8.0.dev0, PyTorch 2.10.0+cu128, Accelerate 1.13.0 |

---

### 🎯 评估指标

采用两种评估方式：

#### （1）Teacher-forced token-level accuracy（早期小规模实验）

#### （2）Generation-based Baseline-Relative Scoring（主指标）
对自由生成输出（`max_new_tokens=1200`）进行分类：

| 类别 | 判定标准 |
|------|--------|
| **Good (G)** | 与 baseline 完全匹配且语法有效 |
| **Similar (S)** | 或两边都不完全匹配；或语法有效且 normalized code similarity ≥ 0.85 |
| **Bad (B)** | 语法无效或严重偏离 baseline 输出 |

> 主要指标：**G + S 数量 / 总 prompt 数**

> 注：此为 **behavioral preservation** 度量，非绝对语义正确性。

---

### 🔁 基线方法对比

| 基线配置 | 描述 |
|--------|------|
| **Flat all-layer masking** | 全局统一屏蔽 30% 或 40% 最低幅值专家（作为传统均匀剪枝代表） |
| **Unmasked baseline** | 完整模型输出，用作比较基准 |
| **No masking + top-k=6** | 减少每 token 激活专家数从 8→6，测试推理加速潜力 |

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据汇总

| 配置 | Prompt Scale | G+S / Total | 屏蔽专家数 | 平均相似度 |
|------|-------------|------------|-----------|----------|
| Flat all-layer 30% | 300 | 150 / 300 | 3,040 | 0.73 |
| Late layers (30–39) @40% | 300 | 249 / 300 | 1,020 | 0.89 |
| **Late ramp**: 30–34@35%, 35–39@55% | 300 | **255 / 300** | 1,145 | 0.91 |
| **Very late only**: 35–39 @50% | 300 | 250 / 300 | **640** | 0.89 |
| **Very late only**: 35–39 @50% | **500** | **419 / 500** | **640** | 0.897 |
| Global late budget (weakest 1,020) | 500 | 410 / 500 | 1,020 | 0.895 |
| Late ramp | 500 | 408 / 500 | 1,145 | 0.892 |

> 💡 结果显示：**very-late-only policy 在更大验证集上表现最优，兼顾高质量保留与高专家节省**

---

### 📈 与基线方法对比结果

- **Flat all-layer 30% masking**：
  - 在 300 prompts 下仅保留 **150/300 G+S**
  - 显著劣于所有 late-focused 策略（差距达 +99~+105 G+S）
  - 表明 **uniform pruning 是次优甚至有害策略**

- **Late-focused policies**：
  - 即使屏蔽更多专家（如 late ramp），也能维持接近完整的输出一致性
  - **very-late-only @50%** 以最少屏蔽量（640 experts）达到最佳 trade-off

---

### 🔍 消融实验结果

#### （1）Layer Region Ablation（100 prompts, 40% hard masking）

| 屏蔽区域 | G+S / 100 |
|--------|----------|
| Early (0–9) | 64 |
| Middle (10–29) | 67 |
| **Late (30–39)** | **80** |
| All-layer 40% | 55 |

> ➤ **early/middle layers 极其脆弱，late layers 更鲁棒**

#### （2）Discovery Sweep（300 prompts）
- 发现 **top-5 最佳策略集中在 late/very-late 区域**
- 向前扩展至 middle layers（如 L20–29 加 mask）会导致性能下降
- 支持 “depth-dependent sensitivity” 假设

#### （3）Top-k Routing Width Reduction（k=8 → k=6）

| 配置 | G+S / 100 |
|-----|----------|
| Baseline (k=8) | 100 |
| **k=6, no masking** | **100** |
| k=6 + late 40% masking | 77 |

> ➤ **单独降低 k 可大幅减少 wall-clock time 且无质量损失**
> ➤ 但与 aggressive masking **不可叠加**，存在非加性交互效应

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **MoE 层敏感性具有强烈 depth-dependency**：
   - **Early (0–9) 和 Middle (10–29) layers 极其脆弱**，轻微扰动即可导致语义漂移
   - **Late (30–39) 特别是 Very-Late (35–39) layers 高度容忍专家屏蔽**
   - 路由熵分析表明 late layers 路由分散、信心低，专家间更具可替代性

2. **拒绝 flat/uniform pruning 策略**：
   - 尽管技术上可行（masked_selection_hits_total == 0），但行为退化严重
   - 常见失败模式：“reasoning leakage”，即中间推理内容泄露至最终输出

3. **Pareto Frontier of Late-Focused Policies**：
   - 在 late layers 内部形成多个高效压缩策略前沿
   - **very-late-only @50%** 成为综合最优解：**质量最高 + 屏蔽最少**

4. **Top-k Reduction 具备独立优化价值**：
   - 从 top-8 降至 top-6 可显著降低延迟，且不影响输出质量（在无 masking 条件下）

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **未执行物理权重裁剪** | 当前为 inference-time masking，checkpoint 大小不变，尚无实际内存节省 |
| **单一模型 & 单一任务** | 结论基于 Qwen3.6-35B-A3B 和 XLCoST code translation，泛化性待验证 |
| **评估依赖 baseline-relative heuristic** | 未使用 AST similarity、CodeBLEU 或执行评测等更强语义指标 |
| **缺乏置信区间** | 所有配置为单次运行，frontier 上相近策略无法判断统计显著性 |
| **无 post-masking fine-tuning** | 零样本评估，可能低估恢复潜力 |
| **reasoning-phase segmentation 不可靠** | 在 Qwen3.6 上无法稳定提取 think/answer 边界，限制 phase-aware pruning 应用 |

---

### 🔮 未来工作方向

1. **Physical Weight Surgery**  
   将最佳 masking 策略转化为真实 checkpoint 修改，测量实际部署收益（显存、存储、加载时间）

2. **Semantic-Rich Evaluation**  
   引入 AST matching、CodeBLEU、execution-based testing 和人工评审增强评估可信度

3. **Bootstrap Confidence Intervals**  
   对 300/500-prompt 结果进行重采样，量化 frontier 策略间的不确定性

4. **Training-Based Recovery**  
   对 masked 或 pruned 模型进行 SFT 或 routing-aware 微调，探索质量恢复路径

5. **Dynamic Active-Expert Reduction**  
   深入研究 k=6 的 latency 收益，结合 very-late masking 探索协同优化空间

6. **Phase-Aware Expert Scoring**  
   整合 magnitude、routing frequency、task affinity 和 phase-specific usage 信号构建综合重要性评分

7. **Cross-Task Generalization Study**  
   在自然语言任务或其他代码任务上复现 depth-sensitivity protocol，检验 very-late compressibility 是否普适

--- 

> **一句话总结**：  
> 本论文通过大规模实证证明，**MoE 模型的 late layers（特别是最后五层）是安全高效的压缩靶区**，提出 **depth-aware expert masking** 作为优于 flat pruning 的新范式，为 MoE 模型的实际压缩与部署提供了坚实的经验基础和清晰的技术路线图。

</details>

---

### 8. [Post-training Quantization for Hybrid Iterative Generative Models](https://arxiv.org/abs/2608.13932)

**Authors**: Jing Gao, Junyi Wu, Wei Wang, Yan Yan, Yao Zhao  
**Category**: cs.LG  
**Published**: 2026-08-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.13932v1  

#### Abstract
Iterative Generative Models (IGMs) span autoregressive and diffusion paradigms, and hybrid variants that couple them can achieve remarkable image-generation fidelity. However, their iterative inference incurs substantial computational overhead, making Post-training Quantization (PTQ) appealing for a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Post-training Quantization for Hybrid Iterative Generative Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **Hybrid Iterative Generative Models (Hybrid IGMs)** 在应用 **Post-training Quantization (PTQ)** 时面临的两大挑战：

- **Excessive Outliers (EOs)**：激活值中存在稀疏但幅值极高的异常通道，导致统一量化尺度下难以兼顾正常值精度与异常值表示范围，造成生成质量严重下降。
- **Amplified Anomalies (AAs)**：微小的量化误差在迭代去噪过程中被逐层放大，超出静态校准范围，引发截断（truncation），最终导致模型崩溃（model collapse）。

直接将传统 PTQ 方法应用于 Hybrid IGMs（如 MAR 模型）会导致严重的性能退化甚至完全失效。

---

### 提出的新方法：HyGenQ
为解决上述问题，作者提出 **HyGenQ** —— 一种专为 Hybrid IGMs 设计的 PTQ 框架，包含两个核心模块：

#### (1) **Hierarchical Cluster Decoupling (HCD)**
- **目标**：应对 Excessive Outliers。
- **思路**：观察到异常通道在不同迭代步中具有高度一致性，因此可通过聚类识别并分离“异常通道”与“正常通道”。
- **实现**：
  - 采用三级分层聚类策略（fine-grained clustering → aggregation → decoupling）；
  - 利用遗传算法增强聚类鲁棒性，避免局部最优；
  - 对两类通道分别进行独立量化，保留正常值精度的同时维持异常值动态范围。

#### (2) **Scaling Recalibration (SR)**
- **目标**：缓解 Amplified Anomalies。
- **思路**：利用全精度模型中边界层激活服从标准高斯分布的特性，定义一个稳定的 **Gaussian Bound**（如 ±5.545）作为参考。
- **实现**：
  - 在推理时动态监测边界层输入是否超出该界限；
  - 若超出，则对整个通道进行缩放回界内，防止溢出和截断；
  - 保持量化稳定性，尤其在早期去噪阶段。

---

### 相比现有方法的优势
| 维度 | HyGenQ 优势 |
|------|-------------|
| **有效性** | 成功实现 W8A8 量化，在多种 Hybrid IGM 上均未发生模型崩溃，显著优于所有 baseline。 |
| **鲁棒性** | 同时处理 EOs 和 AAs，适用于不同规模（B/L/H）和架构的模型。 |
| **通用性** | 在非混合模型（如 LlamaGen、LDM-4）上也表现优异，验证其跨范式的泛化能力。 |
| **无需重训练** | 完全基于 PTQ 范式，仅需少量校准样本，部署成本低。 |

---

## 2. 核心实验方法和设置

### 数据集
- 主要使用 **ImageNet 256×256** 进行图像生成任务评估。
- 生成图像数量：50,000 张用于主实验；8,000 张用于消融与泛化实验。

### 实验设置
- **模型**：以 **MAR (Masked Autoregressive Model)** 为代表性 Hybrid IGM，涵盖 MAR-B、MAR-L、MAR-H 三种规模。
- **量化配置**：
  - 统一采用 **W8A8**（权重和激活均为 8-bit）；
  - 所有线性层均参与量化（除非特别标注）；
  - 使用 per-(step, timestep) 的独立量化参数以控制分布漂移。
- **校准集**：从 ImageNet 随机选取 **32 个样本**，运行完整生成流程收集激活统计量。
- **硬件平台**：单张 NVIDIA RTX 4090 GPU，PyTorch 框架。

### 评估指标
| 指标 | 描述 |
|------|------|
| **FID ↓** | Fréchet Inception Distance，衡量生成图像与真实图像分布的距离，越低越好。 |
| **IS ↑** | Inception Score，反映生成图像多样性和清晰度，越高越好。 |
| **Spd. ×** | 推理加速比（相对于全精度模型）。 |
| **T./Img.** | 单图平均生成时间（秒）。 |

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **UniformQuant** | 基础量化 | MSE 截断处理异常值 |
| **RepQ-ViT [18]** | ViT 专用 PTQ | Scale reparameterization |
| **SmoothQuant [24]** | LLM 友好 | 将激活难度转移到权重 |
| **PTQ4DM [25]** | Diffusion 专用 | 考虑去噪过程中的方差变化 |
| **TFMQ-DM [41]** | 时间感知 | 保留 timestep-dependent 信息 |
| **TaQ-DiT [16]** | DiT 优化 | 缓解 Post-GELU 异常 |
| **OCS [68]** | Outlier Splitting | 分离极端通道 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table I）
在 **64 iterations + 100 timesteps** 设置下，**HyGenQ 在所有模型上均取得最佳性能**：

| Model | Method | oFID ↓ | oIS ↑ | ·FID ↓ | ·IS ↑ |
|-------|--------|--------|--------|--------|--------|
| MAR-B | Full-Precision | 2.30 | 279.44 | 2.30 | 279.44 |
|       | HyGenQ (Ours) | **3.50** | **247.33** | **3.61** | **246.00** |
| MAR-L | Full-Precision | 1.80 | 294.10 | 1.80 | 294.10 |
|       | HyGenQ (Ours) | **2.91** | **259.59** | **3.14** | **255.35** |
| MAR-H | Full-Precision | 1.58 | 299.72 | 1.58 | 299.72 |
|       | HyGenQ (Ours) | **3.77** | **257.31** | **6.89** | **210.95** |

> 注：`o` 表示边界层保留全精度；`·` 表示所有线性层均量化。

- 在 `·` 设置下，大多数 baseline 出现 **严重崩溃（FID > 200）**，而 HyGenQ 仍能稳定生成高质量图像。
- 即使在最难的 MAR-H 上，HyGenQ 也能将 FID 控制在 6.89，远优于第二好的 TaQ-DiT（139.90）。

---

### 与其他模型的泛化实验（Table II）
在 **LlamaGen（自回归）** 和 **LDM-4（扩散）** 上，HyGenQ 依然优于 PTQ4DM，表明其具备良好的跨架构泛化能力。

| Model | Method | IS ↑ | FID ↓ |
|-------|--------|------|-------|
| LlamaGen-XL | HyGenQ | **252.308** | **6.598** |
| LDM-4 | HyGenQ | **360.513** | **17.457** |

---

### 消融实验（Table V）
在 MAR-B 上验证各组件作用：

| Method | FID ↓ | IS ↑ |
|--------|--------|--------|
| o Baseline | 10.34 | 144.22 |
| o + HCD | **3.50** | **247.33** |
| · Baseline | 252.41 | 1.98 |
| · + SR | 11.34 | 139.77 |
| · + HCD + SR (HyGenQ) | **3.61** | **246.00** |

- **HCD 显著提升 o-setting 性能**，说明有效缓解 EOs。
- **SR 是防止 ·-setting 崩溃的关键**，单独使用即可避免灾难性失败。
- **HCD + SR 联合使用达到最优效果**，证明二者互补。

---

## 4. 关键结论和发现

### 主要发现
1. **Hybrid IGMs 对 PTQ 极其敏感**，传统方法极易因 EOs 或 AAs 导致模型崩溃。
2. **异常通道具有空间稳定性**，可通过聚类识别并独立量化（HCD）。
3. **边界层异常可建模为高斯偏离**，通过动态缩放（SR）可有效抑制传播。
4. **HyGenQ 实现了首个成功的 W8A8 端到端量化方案**，在多种 Hybrid IGM 上均保持高保真生成能力。

---

### 方法的局限性
1. **引入额外延迟**：SR 操作带来轻微推理开销，尤其在深层模型（如 MAR-H）中更明显。
2. **依赖校准数据分布**：虽然仅需 32 样本，但仍假设其能代表整体激活行为。
3. **低比特扩展受限**：在 W4A4/W4A8 下仍出现严重崩溃，表明当前机制难以支持更低比特。
4. **Gaussian Bound 固定全局使用**：未针对不同层或模型自适应调整，可能牺牲部分精度。

---

### 未来工作方向
1. **降低 SR 开销**：设计轻量化或可学习的动态缩放机制。
2. **探索训练感知量化（QAT）结合**：进一步压缩至 4-bit。
3. **自动化异常检测阈值选择**：减少人工设定（如 `k`, `w`, `G`）依赖。
4. **拓展至视频/多模态生成模型**：验证在更复杂迭代系统中的适用性。

---

> ✅ **总结一句话**：  
> HyGenQ 首次系统性揭示了 Hybrid IGM 在 PTQ 中的两大致命挑战（EOs 和 AAs），并通过 HCD 与 SR 的协同设计实现了稳定高效的 W8A8 量化，在多个模型和设置下显著超越现有方法，为复杂生成模型的高效部署提供了新路径。

</details>

---

### 9. [Polar Code Based Federated Learning: Convergence Analysis and Resource Allocation](https://arxiv.org/abs/2608.13961)

**Authors**: Han Xiao, Wei Kang, Nan Liu  
**Category**: cs.LG  
**Published**: 2026-08-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.13961v1  

#### Abstract
Federated learning (FL) enables collaborative model training across distributed devices without sharing raw data; however, it faces significant communication bottlenecks and channel impairments in practice. Conventional network layer treatments either idealize the channel as error free or apply equa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Polar Code Based Federated Learning: Convergence Analysis and Resource Allocation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的联邦学习（FL）研究大多在**网络层**进行建模，存在两个根本缺陷：
1. **延迟问题**：假设信道无误（error-free），这在有限时延下不现实。
2. **分离原则失效**：将压缩（quantization）与信道编码（channel coding）独立处理，采用**等错误保护（EEP, Equal Error Protection）**，而忽略了量化比特之间的重要性差异。

具体而言，一个本地模型更新被量化为二进制序列后，高位比特（MSB）对数值影响远大于低位比特（LSB）。传统 EEP 对所有比特一视同仁，导致资源浪费且鲁棒性差。

### 提出的新方法与新思路
本文提出了一种**跨层设计（cross-layer design）** 的极化码（polar code）基础联邦学习方案，其核心创新在于：
- **利用极化码的有限块长下的非对称特性**，实现**不等错误保护（UEP, Unequal Error Protection）**。
- 将量化后的梯度比特按重要性排序，并映射到极化码中可靠性更高的合成信道上，从而优先保护更重要的比特。
- 在整个训练过程中，联合优化**量化比特数 $n$** 和**极化码块长度 $N_t$**，以最小化收敛间隙的上界。

### 相比现有方法的优势
- **更强的抗噪能力**：通过 UEP 显著降低因信道噪声导致的模型失真。
- **更高的通信效率**：避免了对所有比特进行同等强度的保护，节省了信道资源。
- **理论支持的动态资源配置**：证明并验证了在训练后期分配更多信道资源（更长的块长度）能有效抑制累积误差，提升最终性能。

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **MNIST** 数据集进行图像分类任务。
- 包含 60,000 张训练图像和 10,000 张测试图像，共 10 类手写数字（0–9）。

### 模型架构
- 采用卷积神经网络（CNN）：
  - 两个 5×5 卷积层（输出通道分别为 10 和 20），每个后接 2×2 最大池化和 ReLU 激活。
  - 全连接层（50 输出，ReLU）。
  - Dropout 层。
  - 输出层（10 类）。
- 使用 **SGD** 优化器，学习率设为 0.005，batch size 为 100。

### 实验设置
- 总客户端数量 $M = 20$，每轮随机选择 $K = 4$（20%）参与训练。
- 总训练轮次 $T = 40$。
- 信道模型为**二元擦除信道（BEC, Binary Erasure Channel）**，擦除概率 $e \in \{0.1, 0.2, ..., 0.8\}$。
- 所有客户端数据为 **IID** 分布。

### 评估指标
- **测试准确率（Test Accuracy）**：作为主要性能指标。
- 收敛速度与最终精度均被考察。

### 基线方法对比
1. **Uncoded Transmission**：
   - 不使用信道编码，直接将 32 位量化后的梯度发送至 32 次 BEC。
2. **LDPC Coded Transmission**：
   - 使用 EEP 的 LDPC 编码，量化为 5 位，固定码长 $N=32$。
3. **Polar Coded Transmission with Constant Block Length**：
   - 使用 UEP 极化码，量化为 5 位，固定块长度 $N=32$。
4. **Polar Coded Transmission with Variable Block Length**（本文提出）：
   - 动态调整量化位数 $n$ 和块长度 $\{N_t\}$，基于优化问题求解。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 如图 9 和图 10 所示，在不同擦除概率下，所提方案显著优于所有基线。
- 当 $e = 0.8$ 时，**Polar Code (variable $N_t$)** 的最终测试准确率可达约 **0.88**，而 Uncoded 方案仅为 **~0.62**，LDPC 方案约为 **~0.70**。

### 与基线方法的对比结果
| 方法 | 准确率优势（vs. 基线） | 收敛速度 |
|------|------------------------|----------|
| **Polar Code (constant $N$)** | 明显高于 Uncoded 和 LDPC | 更快 |
| **Polar Code (variable $N_t$)** | 进一步小幅提升 | 后期收敛更稳定 |

- 所有基于 Polar Code 的方案均表现出更快的收敛速度和更高的最终精度。
- 随着信道质量恶化（$e$ 增大），本文方案相对于基线的性能增益**愈发显著**。

### 消融实验结果
- **UEP vs. EEP**：比较 Polar Code（UEP）与 LDPC（EEP）表明，仅引入 UEP 即可带来巨大性能提升，说明**保护重要比特至关重要**。
- **Variable $N_t$ vs. Constant $N_t$**：
  - 变长配置相比定长配置仅有**轻微增益**。
  - 表明虽然理论上有益，但在实际应用中若计算开销高，可安全省略动态块长优化。

---

## 4. 关键结论和发现

### 主要发现
1. **UEP 是关键**：极化码天然具备的 UEP 特性非常适合 FL 中量化梯度的传输，能够显著提升系统鲁棒性和效率。
2. **后期需更多资源**：理论分析与实验共同表明，在训练后期应分配更多信道资源（如更长的极化码块长度），以抑制累积误差的影响。
3. **量化位数 $n$ 主要取决于信道质量 $e$**：对于给定的擦除概率 $e$，存在一组“良好”的 $n$ 值，且该值基本不受块长度 $N$ 或迭代次数 $t$ 影响。
4. **跨层设计有效性**：打破传统分离原则，联合设计量化与信道编码，是提升 FL 在真实信道下性能的有效路径。

### 方法的局限性
- **依赖于 BEC 模型**：当前分析基于二元擦除信道，扩展到 AWGN 或衰落信道需要进一步研究。
- **优化复杂度较高**：动态块长优化是一个混合整数非线性规划（MINLP）问题，需借助 PSO 等启发式算法求解，实时部署有一定挑战。
- **未考虑非 IID 场景下的偏差放大效应**：尽管模型本身适用于非 IID，但 UEP 对其影响尚未深入探讨。

### 未来工作方向
- 将该 UEP 框架推广至其他信道模型（如 Rayleigh 衰落、AWGN）。
- 探索低复杂度的在线资源分配策略，替代离线优化。
- 结合梯度稀疏化、模型剪枝等技术，构建端到端高效的 FL 通信框架。
- 研究在异构设备（不同计算/通信能力）环境下的自适应 UEP 机制。

--- 

> ✅ **总结一句话**：  
> 本文提出了一种基于 Polar Code 的 UEP 跨层 FL 传输方案，通过优先保护重要量化比特，并联合优化量化与编码参数，显著提升了 FL 在噪声信道下的收敛性能与鲁棒性，尤其在恶劣信道条件下优势明显。

</details>

---

### 10. [Connected Subspace Clustering: Hardness, a Scalable Heuristic, and an Application to Sea Level Geodesy](https://arxiv.org/abs/2608.14215)

**Authors**: Johanna Hillebrand, Jan H\"ockendorff, J\"urgen Kusche, Kelin Luo, Heiko R\"oglin, Melanie Schmidt, Christian Sohler, Bernd Uebbing  
**Category**: cs.LG  
**Published**: 2026-08-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.14215v1  

#### Abstract
Constrained optimization extends classical optimization by integrating side information, making it widely applicable across scientific and engineering domains. Consider a setting where we measure variables at different physical locations. When grouping these measurements, we often want clusters that...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Connected Subspace Clustering: Hardness, a Scalable Heuristic, and an Application to Sea Level Geodesy

## 1. 论文的主要贡献和创新点

### 解决的问题
本文提出并研究了**Connected Subspace Clustering**（连通子空间聚类）问题，旨在解决高维时空数据在进行聚类时同时满足两个目标：
- **统计相似性**：将具有相似时间序列模式的数据点分组（通过最小化到最佳拟合的 $m'$-维仿射子空间的距离实现）。
- **空间连通性**：每个簇必须是空间上连通的区域（即簇在给定的邻接图中诱导出一个连通子图）。

该问题源于**海平面大地测量学**（sea level geodesy）中的实际需求：为了对海平面异常（SLA）数据进行区域化的主成分分析（PCA），需要将海洋表面划分为内部动态一致且地理上连续的区域。

### 提出的新方法与新思路
1. **问题定义**：
   - 形式化定义了 **Connected $m'$-Subspace $k$-Clustering** 问题，将子空间聚类的目标函数与图拓扑上的连通性约束相结合。

2. **理论贡献**：
   - 证明了该问题是**计算困难的**（NP-hard）。即使在最简单的情况下（$m'=0$，即 $k$-means 聚类）且连通性图为带孔洞的网格图（grid graph with holes）时，也难以在 $\Omega(n^{1/2-\epsilon})$ 因子内近似，其中 $n$ 是数据点数量。

3. **算法设计**：
   - 提出了一种高效的 **Lloyd-style 启发式算法**，其核心是一个**通用的合并子程序**（generic merging subroutine）。
   - 该算法交替执行以下步骤：
     - **子空间拟合**：为每个簇计算最佳拟合的 $m'$ 维仿射子空间。
     - **点重分配**：根据到子空间的重构误差重新分配点。
     - **连通性修复**：当重分配破坏连通性时，通过迭代地将最小的连通组件合并到其相邻簇中来恢复连通性。

### 相比现有方法的优势
- **严格满足连通性**：与鼓励空间一致性但不强制连通性的方法（如 HSI 领域的方法）不同，本文方法保证输出**恰好 $k$ 个连通的区域**。
- **优化目标明确**：直接优化子空间重构误差，而不仅仅是空间或时间相似性。
- **可扩展性强**：提出的启发式算法在 $O(n(k+\log n))$ 时间内运行，适用于大规模数据集（如全球海平面数据）。
- **效果优越**：在实验中，该方法生成的区域不仅连通，而且在物理上可解释，并能更好地分离出如 ENSO 和 IOD 等气候信号。

---

## 2. 核心实验方法和设置

### 数据集
- **数据来源**：使用 **Copernicus Marine Service (CMEMS)** 提供的全球格网化海平面异常（SLA）产品。
- **时空范围**：覆盖1993年至今，空间分辨率为 $0.25^\circ \times 0.25^\circ$。
- **有效数据点**：经过预处理后，在 $2^\circ \times 2^\circ$ 的粗分辨率下，共包含 **8,160** 个有效的海洋网格点。
- **数据维度**：每个点对应一个长度为 365（月度时间步长）的时间序列。

### 实验设置和评估指标
- **连通性图**（Connectivity Graph）：构建一个四邻域的网格图（grid graph with holes），排除陆地单元，并考虑经度的周期性。
- **评估指标**：
  - **主要指标**：**子空间重构误差**（subspace clustering cost），即所有点到其所属簇的最佳拟合 $m'$ 维仿射子空间的平方距离之和。
  - **次要指标**：簇的数量（确保为 $k$）、连通组件的数量（理想情况下等于 $k$）。
- **参数设置**：
  - 簇数 $k \in \{8, 15, 20, 25\}$。
  - 子空间维度 $m' \in \{5, 10, 15, 30\}$。
  - 对**滤波**（spatially and temporally filtered）和**未滤波**（unfiltered）两种数据进行了测试。

### 基线方法对比
- **初始化方法**（用于比较初始聚类质量）：
  - `Agglo-ST`：Thompson & Merrifield 提出的无显式连通性约束的凝聚聚类。
  - `Conn-Agglo-Euc` / `Conn-Agglo-ST`：带连通性约束的凝聚聚类（欧氏距离/空时距离）。
  - `Conn-Ward`：带连通性约束的 Ward 方法（来自 scikit-learn）。
  - `Conn-KMeans++`：k-means++ 初始化后通过后处理使其连通。
- **连通性策略**（用于比较 Conn-Subspace 框架内的不同修复策略）：
  - `IterMerge`：每次迭代后都进行合并修复。
  - `PostMerge`：仅在算法结束后进行一次合并修复。
  - `SmoothMerge`：使用高斯加权标签投票进行平滑，最后再修复。
  - `IntegratedConn`：在重分配时限制只能分配给邻居簇。
- **其他对比方法**：
  - **子空间聚类**：`SSC-OMP`, `EnSC`。
  - **高光谱成像**（HSI）：`EGCSC`, `EKGCSC`。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **连通性修复策略对比**：
  - 在 **160 种配置**（4 种 $m'$ × 4 种 $k$ × 2 种数据 × 5 种初始化）中，`IterMerge` 策略在 **73.75%**（118/160）的配置中取得了最低的最终重构误差。
  - `IterMerge` 在所有有效配置中都能降低初始聚类的重构误差。
- **与基线方法的对比**：
  - 本文方法（例如 `Conn-Agglo-Euc` + `Conn-Subspace` + `IterMerge`）相比 `SSC-OMP` 等方法，**重构误差显著更低**。例如，在 $k=25, m'=30$ 下，`SSC-OMP` 的误差高达 14,473.1，而本文方法仅为 2,947.9。
  - 本文方法**始终返回恰好 $k$ 个连通组件**。相比之下：
    - `SSC-OMP` 最多产生 **1,966** 个连通组件。
    - `EnSC` 最多产生 **222** 个连通组件。
    - `EGCSC` 和 `EKGCSC` 产生的连通组件数也远超 $k$。
- **初始化方法的影响**：
  - 在 $k$-means 目标下表现好的初始化（如 `Conn-Ward`）在子空间重构目标下表现较差。
  - `Conn-Agglo-Euc` 和 `Conn-Agglo-ST` 在子空间目标下表现更优。

### 消融实验结果
- **连通性策略消融**：`IterMerge` 明显优于 `PostMerge` 和 `SmoothMerge`，因为后者在最后一步修复时会产生较大的成本跳跃。
- **初始化方法消融**：不同的初始化方法导致最终结果有差异，但 `IterMerge` 框架能稳定地改进所有初始化的结果。
- **数据预处理影响**：`Agglo-ST` 方法对是否进行滤波非常敏感，在未滤波数据上会退化为一个主导簇和多个小簇，无法使用。

---

## 4. 关键结论和发现

### 主要发现
1. **Connected Subspace Clustering 问题具有很强的理论难度**，即使是简化版本也难以近似。
2. 尽管存在理论难度，所提出的基于 **Lloyd-style 迭代和迭代合并**（IterMerge）的启发式算法在实践中非常有效。
3. **`IterMerge` 是最优的连通性维护策略**，它能在迭代过程中平稳地优化目标函数，避免了后期修复带来的成本激增。
4. 该方法能够从全球海平面数据中识别出**物理上连贯且可解释的区域**，这些区域与已知的气候模态（如 ENSO、IOD）高度相关。
5. 该方法在**重构误差**和**连通性保证**方面均显著优于现有的子空间聚类和 HSI 方法。

### 方法的局限性
- **对初始化敏感**：最终结果的质量依赖于初始聚类的质量。
- **对邻接图敏感**：结果依赖于输入的连通性图（邻接关系）的定义。
- **局部最优**：作为启发式算法，不能保证找到全局最优解。
- **参数选择**：需要预先指定簇数 $k$ 和子空间维度 $m'$。

### 未来工作方向
- 研究**双准则优化**（bi-criteria optimization），即同时优化重构误差和连通性。
- 探索如何自动确定最优的 $k$ 和 $m'$。
- 将该方法应用于其他领域的空间嵌入型多元时间序列，如气候场、遥感影像、神经影像和传感器网络。

</details>

---

### 11. [ARC: Fair Relative Advantage Comparison in Open-Ended Real-World Interaction](https://arxiv.org/abs/2608.13622)

**Authors**: Yongqi Tong, Tan Li Hui Faith, Choy Zhen Wen Marcus, Zhou Jin, Kewei Fu, Jiang-Ming Yang, Jianshe Li, Xin Zhang  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.13622v1  

#### Abstract
Open-ended real-world interaction admits multiple valid behaviors: an agent may answer directly, ask for clarification, provide progress updates, or confirm before acting. This flexibility breaks a core assumption behind group-based RL: rollouts compared within a group are no longer guaranteed to be...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ARC: Fair Relative Advantage Comparison in Open-Ended Real-World Interaction

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**开放式的现实世界交互（open-ended real-world interaction）中，基于分组的强化学习（group-based RL）存在的“奖励不公平”（reward fairness）问题**。

在真实场景中，一个任务可能有多种合理的交互策略（如直接回答、先澄清、进度更新、确认后再执行等），而传统 RL 将所有策略的 rollout 放在同一组内进行相对比较。这会导致：
- Reward Model 可能对某些交互风格（如更长的回答）有偏好。
- 即使行为都有效，不同策略间的比较会引入偏差，使得优化倾向于“reward-preferred”而非“context-appropriate”的行为。
- 最终策略可能被扭曲，不是因为质量高，而是因为其风格受奖励模型青睐。

### 提出了什么新方法或新思路
提出 **ARC (Advantage Regularization via Conditioning)**，一种用于开放式交互的训练范式，核心思想是**通过策略条件化（strategy-conditioned）来实现更公平的优势估计**。

具体方法包括：
- **策略条件化分组（Strategy-Conditioned Rollout Grouping）**：在训练时，为每个样本分配一个策略指令（如“Progress Update”），仅在相同策略的 rollout 组内计算相对优势（relative advantages）。这消除了跨策略比较带来的污染。
- **混合奖励与熵正则化（Hybrid Rewards & Entropy Regularization）**：结合格式、工具调用正确性和答案质量的奖励，并加入熵正则化防止策略崩溃（entropy collapse），鼓励多通道生成的多样性。
- **推断时自主选择**：训练时使用策略指令，但在推理时移除指令，让策略自主选择交互方式，确保部署灵活性。

同时，提出了 **INTER³** 框架：
- 一个异步流式交互框架，将用户可见的通信（`<answer>`）与内部推理和工具调用分离。
- 这种分离使得实时响应（降低 TTFT）、用户中断、进度更新等成为可观察、可标注的一等公民行为。
- 支持构建大规模、策略标注的训练数据集 **INTER³-86K**。

### 相比现有方法的优势
- **解决了 RL 中的比较公平性问题**：不同于 SAGE、Scaf-GRPO 等通过提示引导解法的方法，ARC 的提示仅用于定义公平的比较组，不改变任务难度。
- **架构与算法协同设计**：INTER³ 架构为 ARC 提供了理想的训练环境，两者结合解决了从数据收集到公平优化的完整链条。
- **提升真实交互体验**：显著降低了 Time-to-First-Token (TTFT)，从 4.91s 降至 1.27s，提升了用户体验。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **INTER³-86K**：论文构建的 86.8K 示例的策略标注数据集，包含：
  - **SFT 集**：57.9K 示例，涵盖工具使用（59.1%）、多跳问答（29.7%）、逻辑推理（11.1%）。
  - **RL 集**：28.9K 示例，全部来自工具使用场景，并注入策略指令。
- 数据来源包括：
  - 真实线上客户服务平台的交互日志（脱敏后）。
  - 公共基准数据集（如 ToolMind、Musique、KnightsAndKnaves）。
  - 由强教师模型（Qwen3.5-397B）合成的多样化轨迹。

### 实验设置和评估指标
- **模型**：基于 Qwen3-8B 和 Qwen3-4B 进行微调。
- **训练**：使用 GRPO、PPO、DAPO 等作为 RL backbone，warm-start 自同一 SFT 检查点。
- **评估维度**：
  - **In-domain 工具使用能力**：使用 **tau-bench (T-bench)** 和 **tau2-bench (T2-bench)**，衡量多轮工具调用的成功率（平均得分）。
  - **Out-of-domain 推理能力**：使用 AIME2026、GPQA-Diamond、HMMT2025 等数学和科学推理基准。
  - **指令遵循与对齐**：使用 IFBench、Arena-Hard。
  - **延迟**：**Time-to-First-Token (TTFT)**，越低越好。

### 基线方法对比
- **基础模型变体**：
  - `Qwen3-8B-noThink`：无内部推理模式。
  - `Qwen3-8B-Think`：标准的 think-then-act 模式。
- **标准 RL 方法**：
  - `PPO`, `DAPO`, `GRPO`：作为 backbone。
- **ARC 变体**：
  - `PPO+ARC`, `DAPO+ARC`, `GRPO+ARC`：在 backbone 上应用 ARC 方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
在 **GRPO backbone** 上，ARC 带来了最显著的提升：

| 方法 | 平均得分 (Avg.) | T-airline | T-retail | T2-airline | TTFT ↓ |
|------|-----------------|---------|----------|------------|--------|
| GRPO | 28.09 | 31.33 | 40.29 | 36.67 | 0.61s |
| **GRPO + ARC** | **33.46 (+5.37)** | **44.00** | **50.00** | **48.00** | **1.27s** |

- **在域内工具使用上大幅提升**：平均得分提升 **5.37 分**，多个子任务提升超过 10 分。
- **延迟显著降低**：INTER³ 架构本身将 TTFT 从 think 模式的 4.91s 降至 1.27s，ARC 在此基础上进一步优化了能力。
- **小模型验证**：在 Qwen3-4B 上，`GRPO+ARC` 的平均得分达 34.23，远超 `noThink` (22.38) 和 `Think` (29.54)，证明效果不依赖大模型容量。

### 消融实验结果
#### （1）策略条件化与熵正则化的消融（Table 9）
| 配置 | 平均得分 | T/T2 平均 |
|------|----------|-----------|
| GRPO | 25.43 | 35.81 |
| GRPO + Entropy | 24.23 | 33.53 |
| GRPO + 策略分组 | 27.80 | 43.93 |
| **GRPO + 策略分组 + 熵正则化** | **29.59** | **47.00** |

- **策略分组是主要增益来源**，单独加熵正则化甚至有害。
- 两者结合达到最佳效果，熵正则化在分组后起到稳定多通道生成的作用。

#### （2）策略多样性扩展实验（Table 6）
| 策略组合 | 平均得分 | T/T2 平均 |
|---------|----------|-----------|
| 仅 Progress Update | 21.74 | 23.62 |
| + Direct Answer | 24.91 | 31.67 |
| + Clarify First | 25.60 | 31.04 |
| **全策略组合 (4类)** | **29.59** | **47.00** |

- 随着策略多样性增加，性能持续提升，**总提升达 36.1%**。
- 证明 ARC 能有效利用异构策略数据。

#### （3）训练时策略指令的课程学习（Table 5）
| 设置 | 平均得分 |
|------|----------|
| 不移除指令（默认 ARC） | **29.59** |
| 线性移除 | 27.28 |
| 固定概率移除 | 26.97 |

- **保持指令贯穿训练全程效果最好**，提前移除会削弱优势估计的稳定性。

#### （4）推理时策略提示的影响（Figure 7）
- 移除、随机或匹配指令，在推理时均**未带来性能提升**。
- 证明策略行为已在训练中内化，指令仅是训练时的“脚手架”。

---

## 4. 关键结论和发现

### 主要发现
1. **公平比较是开放式交互学习的关键瓶颈**：优化不仅取决于如何奖励，更取决于如何公平地比较不同策略的行为。
2. **ARC 通过策略条件化实现了更干净的优势估计**：理论分析表明其能消除跨策略方差，实验证明其在工具使用任务上大幅超越基线。
3. **INTER³ 架构是 ARC 成功的基础**：分离通信与执行通道，使得多样化交互行为可观测、可标注、可训练。
4. **策略多样性本身具有价值**：即使少数策略（如 Alignment Check 仅占 2.5%）也能为整体能力做出贡献。

### 方法的局限性
1. **策略抽象较粗粒度**：当前的四类策略无法完全捕捉真实交互的细微差别。
2. **领域范围有限**：最强结果集中在工具使用场景，是否泛化到其他领域待验证。
3. **依赖数据标注**：需要高质量的策略标注，可能存在标注偏见或部署特异性。
4. **理论分析简化**：方差分析是理想化模型，非完整的收敛性证明。

### 未来工作方向
- 设计更细粒度、动态演化的策略分类体系。
- 探索无需显式策略标注的自监督或弱监督条件化方法。
- 将 ARC 思想推广到更多开放域任务（如教育、医疗对话）。
- 研究如何在保证公平性的同时，进一步提升策略选择的智能性与适应性。

</details>

---

### 12. [Simulation-Aware In-Context Policy Improvement for LLM-Aided Analog Layout Refinement](https://arxiv.org/abs/2608.13767)

**Authors**: Bingyang Liu, Ziming Wei, Xiaohan Gao, David Z. Pan  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.13767v1  

#### Abstract
Analog IC layout design remains a labor-intensive iterative process dominated by simulation-driven refinement. Although end-to-end layout generators accelerate initial placement and routing, they still require experts to manually tune layout optimization parameters with repeated post-layout simulati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Simulation-Aware In-Context Policy Improvement for LLM-Aided Analog Layout Refinement*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
模拟集成电路（Analog IC）后端布局设计是一个高度迭代、依赖**post-layout simulation**反馈的过程。尽管已有端到端的自动布局生成器（如 ALIGN、MAGICAL），但其内置的启发式规则是静态且通用的，无法针对特定电路进行动态优化。设计师仍需手动调整大量**layout optimization parameters**（如对称约束、布线优先级等），并反复运行耗时的寄生提取和仿真，效率极低。

此外，传统自动化方法如 **Bayesian Optimization (BO)** 虽在前端尺寸设计中有效，但在布局阶段因每轮评估成本极高（寄生提取 + post-layout simulation），通常需要数百次迭代，远超实际允许的“数十次”预算，因此不实用。

同时，现有的 LLM 辅助设计方法多依赖模糊的自然语言指令，缺乏对几何状态的精确感知，也难以积累跨轮次的设计经验。

---

### 🚀 提出的新方法与创新思路
本文提出一种**simulation-aware 多智能体 LLM 框架**，实现基于稀疏仿真反馈的 **In-Context Policy Improvement (ICPI)**，通过一个 **act-observe-reflect 循环**来迭代优化布局参数。

#### 核心创新点包括：

- **紧凑、结构化的 Layout State 表示**  
  设计了一种 LLM 友好的中间表示，整合了：
  - 电路连接性（connectivity）
  - 器件位置与包围盒（bbox）
  - 当前激活的 layout optimization parameters（见下表）
  - 寄生参数摘要（parasitic summaries）
  - post-layout simulation 结果  
  替代了模糊的文本描述，使 LLM 能够“看到”当前布局状态并做出精准推理。

- **持久化 Design Journal 机制**  
  引入跨轮次的记忆存储，记录每次修改的目标、操作、结果及反思（reflection）。这使得模型能在不更新权重的情况下，在上下文中学习设计经验，避免重复错误，提升决策质量。

- **多智能体 ICPI 循环架构**  
  包含三个角色明确的 LLM Agents：
  - **Supervisor**：决定下一步优化哪个参数族（parameter family）
  - **Executor**：将高层目标转化为具体参数修改，并处理失败回滚
  - **Reflector**：生成本轮反思，写入 Design Journal  
  形成闭环的 **act → observe → reflect** 流程，实现测试时策略改进（test-time policy improvement）。

- **暴露可控的 Action Space**  
  在 MAGICAL 基础上显式暴露五类可调参数作为动作空间：

| Parameter Family | Effect |
|------------------|--------|
| Net weights      | 提高关键 net 在 placement 中的权重 |
| Placement bias   | 控制器件偏向某一方向放置 |
| Symmetry         | 加强敏感器件的对称性约束 |
| Priority         | 提升关键 net 的布线优先级 |
| Wire widths      | 权衡电阻与电容，调节走线宽度 |

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | 本方法优势 |
|------|--------|-----------|
| 手动调参 | 高人力成本，依赖专家经验 | 自动化探索，降低门槛 |
| BO（贝叶斯优化） | 需要上百次仿真，超出预算 | 仅需数十次仿真即可收敛 |
| LLM + 自然语言交互 | 描述模糊，无状态记忆 | 结构化状态 + 持久记忆，更可靠 |
| 其他 LLM agents（如 PANDA） | 不维护 layout state 或 design journal | 显式建模状态演化与经验积累 |

> ✅ **核心优势**：在极小的 post-layout simulation 预算下（仅约 10 次），显著优于启发式方法和 BO，实现了高效、可靠的模拟布局优化。

---

## 2. 核心实验方法和设置

### 📊 数据集与电路基准
在两个真实世界的 **Operational Transconductance Amplifier (OTA)** 上验证：
- **OTA1**：两阶段 Miller 补偿 OTA，65nm CMOS 工艺
- **OTA2**：全差分 OTA 含共模反馈，40nm 工艺  
两者均为工业常用模块，非玩具案例。

---

### ⚙️ 实验设置
- **布局生成器**：基于开源工具 MAGICAL 修改，支持 PDK 扩展并暴露上述五类 layout optimization parameters
- **仿真流程**：
  - 寄生提取：Siemens Calibre
  - Post-layout simulation：Cadence Spectre
- **LLM backbone**：GPT-5（所有 agents 共享同一模型，使用不同 prompt 分工）
- **优化预算**：
  - 总共 **31 轮候选生成**
  - 每 3 轮执行一次 post-layout simulation ⇒ 每个设计仅 **11 次仿真**
  - 对比方法 BO 则需每轮都仿真 ⇒ 共 31 次仿真

---

### 🎯 评估指标
定义了一个综合 **Figure of Merit (FoM)** 来量化电气性能：

$$
\text{FoM} = \left( s_{\text{Gain}} \cdot s_{\text{UGB}} \cdot s_{\text{CMRR}} \right)^{1/3} \times o_{\text{PM}}
$$

其中每个 $ s_m $ 函数如下：
- 若指标未达标（$ u_m < L_m $）：采用平方惩罚项
- 若达标：平滑增长奖励
- $ o_{\text{PM}} $ 对相位裕度进行 capped 奖励（上限为 3）

> ✅ 强调满足最低规格的重要性，同时鼓励超越目标。

此外还报告：
- 单项指标（Gain, UGB, CMRR, PM）
- 版图面积（Area）
- 收敛稳定性（是否产生无效设计）

---

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **Heuristic** | 使用生成器默认固定参数（net weight=1, no bias, min wire width 等） |
| **BO** | 在相同参数空间上运行贝叶斯优化，目标为最大化 FoM |
| **Ours w/o ICPI** | 移除 Design Journal 和跨轮反思，仅单轮提示生成更新 |
| **Ours (Full ICPI)** | 完整框架，包含状态、日志、三智能体循环 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table II）

| Method       | OTA1 FoM | OTA1 Area (μm²) | OTA2 FoM | OTA2 Area (μm²) |
|--------------|----------|------------------|----------|------------------|
| Heuristic    | 0.869    | 3592.9           | 0.696    | 7080.0           |
| BO           | 0.979    | 4036.9           | 0.880    | 8857.3 (+25%)    |
| Ours w/o ICPI| 1.003    | 3288.3 (-8.6%)   | 0.868    | 6490.5 (-8.3%)   |
| **Ours (ICPI)** | **1.104** | **4031.4**     | **1.038**| **6468.2**       |

> ✅ OTA1：ICPI 达到最高 FoM（1.104），较 BO 提升 **12.8%**
>
> ✅ OTA2：ICPI 成功满足所有四项电气目标（Gain ≥60dB, CMRR≥80dB 等），FoM 达 **1.038**，相较 BO 提升 **18%**，且面积减少近 **27%**

---

### 🔍 详细分析与消融实验

#### （1）OTA1 小搜索空间表现
- BO 表现尚可（FoM=0.979），但仍低于 ICPI
- Ours w/o ICPI 已优于 Heuristic 和 BO，说明 LLM 本身具备一定推理能力
- **完整 ICPI 再提升 10%+**，证明 Design Journal 和反思机制带来持续增益

#### （2）OTA2 大参数空间挑战
- 参数维度比 OTA1 高约两个数量级
- BO 虽有提升，但 **CMRR 仍未达标**，且面积大幅增加
- Ours w/o ICPI 未能完全达标，显示纯一次性提示不足以应对复杂设计
- **完整 ICPI 成功达标并缩小面积**，体现其在高维空间中的优越样本效率

#### （3）可行性与鲁棒性对比
- **BO 会产生电学失效的设计**（loop gain < 1），FoM=0
- **ICPI 所有提交布局均合法、LVS-clean、功能正常**
  - 因 Executor 可检测 backend failure（如布线崩溃）并回滚
  - 体现了物理意识（physical awareness）的重要性

#### （4）运行开销分析
- 每轮 agent orchestration 平均耗时：
  - OTA1: ~37s, 12k tokens
  - OTA2: ~43s, 17k tokens
- 总 agent 时间：< 21 分钟（30 轮）
- 相比之下，**寄生提取 + 仿真 > 141 分钟**
- ⇒ **agent 开销占比 < 16%**，主要瓶颈仍是仿真，符合“economize simulation”的初衷

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **ICPI 框架可在极低 simulation 预算下（~10 次）实现高质量模拟布局优化**，显著优于传统启发式与 BO。
2. **结构化 Layout State + Design Journal 是实现 LLM “状态感知”与“经验积累”的关键**，解决了自然语言描述模糊、无记忆的问题。
3. **多智能体分工 + act-observe-reflect 循环能有效引导 LLM 进行长期规划与自我修正**，形成真正的 in-context policy improvement。
4. **该方法不仅提升 FoM，还能保持甚至减小面积**，表明优化集中在电气性能而非盲目扩大版图。
5. **相比黑箱优化（如 BO），ICPI 更具鲁棒性**，能规避物理不可行配置，确保输出始终可用。

---

### ⚠️ 局限性
1. **目前仅适用于中小规模电路**（如 OTA），扩展到更大系统（如 PLL、ADC）面临状态表示复杂性和搜索空间爆炸问题。
2. **依赖于布局生成器对外暴露参数接口**，若工具封闭则难以集成。
3. **尚未支持多目标联合优化**（如 FoM + Area），当前目标仍以电气为主。
4. **LLM 推理成本虽不高，但 token 消耗随电路规模增长较快**，可能影响实用性。

---

### 🔮 未来工作方向
1. **引入层次化 partitioning 策略**，将大电路分解为子模块分别优化，提升可扩展性。
2. **开发更高效的 state compression 方法**，降低上下文长度与 token 开销。
3. **拓展至更多类型的 analog blocks**（如 Bandgap, LDO, ADC），验证泛化能力。
4. **结合 hierarchical P&R flow**，实现从 block-level 到 chip-level 的全流程协同优化。
5. **开放演示项目**：已发布 [GitHub 仓库](https://github.com/bingyang1132/ICLAD2026-demo-sim-aware-in-context) 提供 PDK-free demo，便于复现与社区发展。

---

> 💡 **总体评价**：本文首次将 **in-context learning** 与 **simulation-aware layout refinement** 深度结合，提出了一个兼具理论深度与工程价值的 LLM 多智能体框架，为下一代智能化模拟 EDA 工具提供了重要范式。

</details>

---

### 13. [Dynamic Multi-Depot Vehicle Routing with Online Requests: Event-Driven Transformer--DRL and Rolling-Horizon Benchmarking](https://arxiv.org/abs/2608.13799)

**Authors**: Faezeh Ardali, Gerald M. Knapp  
**Category**: cs.LG  
**Published**: 2026-08-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.13799v1  

#### Abstract
This paper presents an event-driven learning and benchmarking framework for the Dynamic Multi-Depot Vehicle Routing Problem with progressively revealed requests and evolving vehicle states. Masked MLP and Transformer policies are trained through behavior cloning and proximal policy optimization. Det...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Dynamic Multi-Depot Vehicle Routing with Online Requests: Event-Driven Transformer-DRL and Rolling-Horizon Benchmarking*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文研究的是**动态多仓车辆路径问题（Dynamic Multi-Depot Vehicle Routing Problem, D-MDVRP）**，其中客户请求是逐步在线揭示的，且车辆状态（位置、容量、服务进度等）随时间演化。该问题广泛应用于最后一公里配送、应急物流、移动医疗服务等领域。

传统静态 VRP 假设所有请求已知，而实际系统中请求在运行过程中不断到达，因此需要一个能够实时响应变化并做出高质量调度决策的框架。

---

### 提出的新方法与创新思路

作者提出了一种**事件驱动的学习与基准测试集成框架（event-driven learning and benchmarking framework）**，其核心创新点包括：

- ✅ **事件驱动的 D-MDVRP 环境建模**  
  将系统状态更新绑定到关键事件（如请求到达、车辆抵达、服务开始/完成），实现高效仿真。

- ✅ **确定性可行性掩码（Deterministic Feasibility Masking）**  
  在策略选择动作时，通过硬约束排除非法分配（如超载、无效车辆-请求组合），确保输出始终满足容量和可达性要求。

- ✅ **固定前缀/灵活后缀路线承诺机制（Fixed-prefix/Flexible-suffix Route Commitments）**  
  保护已完成、正在进行以及短期内将执行的任务不被修改，仅对“灵活后缀”部分进行重规划，提升计划稳定性。

- ✅ **分离的重分配与重排序度量（Separate Reassignment and Resequencing Metrics）**  
  区分车辆更换（reassignment）和前驱节点变更（resequencing），更精细地衡量路线扰动（route disruption），支持稳定性分析。

- ✅ **统一协议下的公平比较（Common-protocol Evaluation）**  
  所有方法（启发式、学习型策略、rolling horizon）在同一环境、目标函数、承诺规则下进行对比，增强了可比性和可信度。

- ✅ **基于 Transformer 和 MLP 的策略训练流程**  
  使用行为克隆（Behavior Cloning, BC）预训练 + 近端策略优化（PPO）微调的方式训练神经网络策略，并验证其迁移能力。

---

### 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **系统设计** | 集成事件驱动、可行性保障、路线冻结、细粒度扰动测量于一体，提供完整实验平台 |
| **评估严谨性** | 引入 common-protocol benchmarking，避免因设置差异导致不公平比较 |
| **实用性** | 学习策略可在毫秒级完成推理，适合在线部署；支持跨规模泛化（zero-shot transfer） |
| **透明性与复现性** | 开源代码与实验脚本公开，便于后续研究 |

---

## 2. 核心实验方法和设置

### 数据集与实例生成
- 使用合成的欧氏空间（Euclidean）实例，车辆容量为 30。
- 实例按规模分为三类：
  - **Small**: 2 depot, 4 vehicle, 18 初始 + 12 动态请求 → 共 30 请求
  - **Medium**: 3/6/30/20 → 50 请求
  - **Large**: 4/8/48/32 → 80 请求
- 动态请求分四波到达，场景种子固定以保证可重复性。

---

### 实验设置

- **编程环境**：Python 3.8.4 + PyTorch 2.4.1，CPU 平台（Intel i7-1165G7）
- **无 GPU 加速**，强调轻量级部署潜力
- **Rolling Horizon 基线**：使用 OR-Tools + SCIP 求解器，每轮 replanning 时间限制为 **1 秒**

---

### 评估指标（Evaluation Objectives）

主评估目标函数为加权组合：

$$
J_{\text{eval}} = D + 0.10W + 2.00N_{\text{chg}}
$$

其中：
- $D$: 总行驶距离（含回 depot）
- $W$: 客户总等待时间（customer-minutes）
- $N_{\text{chg}}$: 路线变更次数（route changes）

此外还报告：
- 平均/最大等待时间（Avg./Max Wait）
- Makespan（最后完成时间）
- Route disruption 分解为：
  - $N_{\text{asg}}$: 车辆重分配次数
  - $N_{\text{seq}}$: 前驱节点变更次数
- 决策耗时（ms/action 或 ms/replan）

---

### 基线方法对比

共比较五类方法：

| 方法 | 类型 | 描述 |
|------|------|------|
| **Random** | 启发式 | 随机从可行动作中选择 |
| **Nearest Feasible** | 启发式 | 最近邻插入（最小终点到请求距离） |
| **Cheapest Append** | 启发式 | 最小增量距离插入 |
| **Waiting-aware Insertion** | 启发式 | 综合考虑增量距离 + 预测等待时间 + 变更惩罚 |
| **Hybrid Expert** | 启发式 | 多策略融合专家，用于 BC 训练 |
| **MLP-BC / MLP-PPO** | Learning-based | 多层感知机 + 行为克隆 / PPO 微调 |
| **Transformer-BC / Trans.-PPO** | Learning-based | 基于 Attention 的架构，处理高维状态依赖 |
| **Rolling Horizon Optimization** | Optimization-based | 每次事件触发重新优化，限时 1 秒 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table III & VI）

#### 统一基准测试（20 场景平均）

| Method | Distance | Avg Wait | Objective ($J_{\text{eval}}$) | Changes | Time (ms/action) |
|--------|----------|-----------|-------------------------------|---------|------------------|
| **Nearest Feasible** | **179.1** | 44.5 | **346.1** | 16.9 | **0.156** |
| Cheapest Append | 191.1 | 62.6 | 477.5 | 49.3 | 0.163 |
| Waiting-aware | 193.4 | 55.6 | 428.7 | 34.3 | 0.164 |
| MLP-PPO | 192.0 | 56.8 | 430.8 | 34.2 | 0.609 |
| Trans.-PPO | 200.9 | 60.2 | 441.9 | 30.2 | 0.763 |

> 📌 **结论**：**Nearest feasible 在距离、目标值、稳定性、速度上全面优于所有学习模型**

---

#### 公共协议下的详细对比（Table VI，$H_{\text{commit}}=15$ min）

| Method | Distance | Avg/Max Wait | Makespan | Objective ($J_{\text{stab}}$) | Reassign/Seq | Runtime (ms/replan) |
|--------|----------|---------------|-----------|------------------------------|--------------|---------------------|
| **Nearest** | **175.1** | 42.7 / 122.9 | 183.7 | **328.7** | 8.1 / 9.3 | **7.23** |
| Rolling Horizon | 216.0 | **34.6 / 106.3** | **177.3** | 377.4 | 13.6 / 30.5 | **587.50** |
| Transformer-PPO | 200.0 | 59.9 / 180.3 | 205.9 | 430.8 | 13.9 / 23.3 | 19.69 |

> 📌 **Rolling horizon** 在 **等待时间和 makespan 上最优**，但代价是更高的距离、更多序列变更和 **~80倍于启发式的计算开销**

> 📌 **Learned policies** 推理速度快（<20ms/replan），但性能未超越 best heuristic

---

### 消融实验结果（Ablation Study, Table IV）

评估“全框架”各组件的重要性（使用 waiting-aware 策略）：

| 配置 | Objective | Adj. p-value | 显著性 |
|------|-----------|--------------|--------|
| Full Framework | **414.38** | — | — |
| No near-term commit | 434.37 (+4.83%) | 0.824 | 不显著 |
| **No reassignment penalty** | 430.29 (+3.84%) | **0.007** | ✅ 显著恶化 |
| No resequencing penalty | 427.84 (+3.25%) | 0.098 | 边缘显著 |

> 🔍 **发现**：移除 **车辆重分配惩罚项** 对性能影响最显著，说明控制车辆切换对稳定性至关重要。

---

### 跨规模泛化能力（Zero-shot Transfer）

| Method | Small (30 req) | Medium (50 req) | Large (80 req) |
|--------|----------------|------------------|----------------|
| Nearest | 10.70 / 0.21 | 12.08 / 0.61 | **14.35 / 1.72** |
| MLP-PPO | 13.70 / 0.67 | 13.25 / 1.49 | 16.25 / 3.54 |
| Transformer-PPO | 14.21 / 0.95 | 13.88 / 1.87 | 17.74 / **4.36** |

> ✅ 所有学习策略无需再训练即可扩展至 80 请求场景  
> ⚠️ 但性能仍落后 nearest feasible，且推理延迟随规模增长更快

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **最强基线仍是简单启发式**  
   `Nearest feasible` 在综合目标、稳定性、响应速度上全面优于所有学习模型和 rolling horizon 方法。

2. ✅ **学习策略具备实用推理速度与零样本迁移能力**  
   MLP/Transformer 模型可在 **毫秒级** 输出合法路径，适用于在线部署，并能直接应用于更大规模实例（30→80 请求）而无需微调。

3. ✅ **PPO 微调效果有限且具随机性**  
   - 对 MLP 几乎无改进（平均 +0.13%）
   - 对 Transformer 有约 **2.5% 改进**，但不同训练种子间方差较大（seed variability）

4. ✅ **Rolling horizon 牺牲效率换取服务质量**  
   虽然获得最低等待时间与 makespan，但带来更高行驶距离、更多路线扰动及百倍以上的计算成本。

5. ✅ **稳定性机制有效**  
   固定前缀 + 分离扰动度量的设计有助于量化和控制计划变更，尤其 **重分配惩罚** 起关键作用。

6. ❌ **当前学习方法未能超越强启发式**  
   尽管模型结构先进（如 Transformer），但在本任务中并未展现出优于手工规则的决策质量。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **未包含时间窗约束** | 实际应用中常见硬/软时间窗，本文未涉及 |
| **同质车队假设** | 所有车辆容量相同，缺乏异构性建模 |
| **静态路网与时速恒定** | 忽略交通拥堵、动态行程时间等现实因素 |
| **Reward 设计保守** | 密集奖励虽提供反馈，但可能限制探索空间 |
| **缺乏大规模真实数据验证** | 当前基于合成数据，推广性有待实证 |

---

### 未来工作方向

- ✅ 多尺度联合训练（multi-size training）以增强泛化能力
- ✅ 设计更优 reward shaping 机制，鼓励长期服务质量提升
- ✅ 引入 soft/hard time windows 约束
- ✅ 扩展至异构车队（heterogeneous fleet）、时间依赖旅行时间（time-dependent travel time）
- ✅ 探索 real transportation networks 替代欧氏距离
- ✅ 结合 meta-learning 或 prompt-based adaptation 提升 zero-shot 性能

---

> 💡 **总体评价**：  
> 本文的价值不在于提出一种“更强”的算法，而在于构建了一个**稳定、可复现、可比较的动态路由实验框架**，强调了在引入复杂模型（如 Transformer-DRL）之前必须建立强有力的启发式基线，并倡导将**可行性、稳定性、计算效率**纳入统一评估体系。这一工程导向的研究范式对推动 DRL 在运筹领域落地具有重要参考意义。

</details>

---

### 14. [A Graph-Based Reinforcement Learning Framework for Structured Drift Diagnosis and Recovery in Autonomous LLM Agents](https://arxiv.org/abs/2608.14109)

**Authors**: Ismail El Hamraoui, Sagar Jose, Nicolas Bureau, Robert Plana  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.14109v1  

#### Abstract
Autonomous LLM agents are increasingly deployed in complex real-world workflows, yet they remain vulnerable to runtime behavioral drift, a silent deviation from the original task that can lead to irreversible side effects on external systems. Existing approaches address drift at the prompt level but...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**自主 LLM Agents 在运行时出现的行为漂移（behavioral drift）**问题。行为漂移是指代理在执行任务过程中逐渐偏离原始目标，可能导致对环境的不可逆修改（如转账、删除消息等），而传统的“重置对话历史”无法修复这些外部系统状态的变化。

现有方法大多集中在**预防阶段**（如输入隔离、指令分层），缺乏对已发生漂移的**结构化诊断与恢复机制**，尤其是在细粒度步骤层面进行检测、风险评估和决策的能力。

---

### 提出的新方法与思路
作者提出了一种**基于图的强化学习框架（Graph-Based Reinforcement Learning Framework）**，用于结构化的漂移诊断与恢复。其核心思想是：

- 将恢复过程建模为一个由五个节点组成的**诊断状态机（diagnostic state machine）**，每个节点负责特定角色：
  1. `Classify Drift`：判断某一步是否漂移
  2. `Detect Operations`：提取漂移期间执行的写操作
  3. `Search Documentation`：识别涉及的应用程序以获取 API 文档
  4. `Evaluate Risk`：基于文档判断操作是否可逆
  5. `Aggregate & Decide`：综合信息决定回滚或上报人工

- 所有节点共享同一个**小型语言模型（Small Language Model, SLM）**，通过**角色条件提示（role-conditioned prompts）** 和统一的训练目标实现多角色专业化。

- 使用 **GRPO（Group Relative Policy Optimization）** 进行强化学习训练，结合两种奖励信号：
  - **规则驱动的结构奖励**：确保输出符合预定义的 XML + JSON schema
  - **LLM-as-Judge 语义质量评分**：评估推理内容的合理性与角色适配性

---

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|--------|
| **模块化与可插拔性（Plug-and-play）** | 不依赖主任务代理模型重训练，适用于大型昂贵模型部署场景 |
| **结构化推理能力** | 强制使用 `<reasoning>` + `<answer>` 的 XML 格式，提升下游节点解析可靠性，避免自由文本歧义 |
| **多角色共享策略** | 单一 SLM 可胜任五种不同角色，降低部署成本，无需为每个节点训练独立策略 |
| **训练高效性** | GRPO 无需价值网络，可在单张 GPU 上完成训练，适合资源受限环境 |

---

## 2. 核心实验方法和设置

### 数据集
- 主要使用 **AppWorld [Trivedi et al., 2024]** 基准测试平台：
  - 包含约 450 个模拟应用 API（如 Spotify、Gmail、Venmo）
  - 支持长轨迹工具调用任务，涵盖多种复杂度级别（Difficulty 1–3）
  - 提供真实漂移注入协议，支持端到端评估

### 实验设置
- **漂移注入方式**：
  - 在代理的消息历史中插入恶意指令（来自 AppWorld 的其他任务），诱导其偏离原任务
  - 分为两类漂移：
    - **Type I：Transient Read Drift**（仅读操作，无环境变更）
    - **Type II：Persistent Read Drift**（读操作持续存在触发源）
  - 当前工作暂不处理 Type III（Write Drift）中的自动修正，仅做检测与上报

- **评估流程**：
  - 给定疑似漂移起始步 $k_0$，从后向前遍历轨迹
  - 各节点依次处理，最终由 $n_5$ 输出 `rollback_before_drift` 或 `escalate_human`

### 评估指标
| 指标 | 描述 |
|------|------|
| **Composite Reward $R$** | 内部评估指标，加权组合 8 项结构检查 + 1 项 LLM-as-Judge 语义得分 |
| **$r_{\text{judge}}$** | 冻结的 14B 推理模型作为裁判给出的内容质量分数（0–1） |
| **task_goal_completion** | AppWorld 官方指标，衡量最终满足的真实子目标比例 |
| **Per-node breakdown** | 按节点分析各项奖励增益，验证角色专业化效果 |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Base Model (no training)** | 使用相同提示模板但未经 GRPO 训练的初始 SLM |
| **No Recovery** | 不启用任何恢复机制，直接运行漂移轨迹 |
| **No Drift (Oracle)** | 无漂移的理想上限 |
| **Frontier Model as Oracle** | 使用 GPT-4o/GPT-4o-mini 替代 SLM 构成“能力上限”对照组 |

使用的 SLM 基座模型：
- **Granite 3.3 2B**
- **Qwen 2.5 1.5B**

---

## 3. 主要实验结果和性能指标

### 内部评估结果（Intrinsic Evaluation）

#### 表 1：持有集上的平均复合奖励与裁判得分

| Model | Setting | $R$ (mean) | $r_{\text{judge}}$ |
|-------|---------|------------|-------------------|
| Granite 3.3 2B | Base | 3.68 | 0.71 |
| Granite 3.3 2B | Ours (GRPO) | **5.15** (+40%) | **0.90** (+27%) |
| Qwen 2.5 1.5B | Base | 0.56 | 0.47 |
| Qwen 2.5 1.5B | Ours (GRPO) | **4.80** (**+8.6x**) | **0.66** (+40%) |

> ✅ 所有 100 个持有提示均显示正向增益（见 Fig. 3），表明训练稳定且泛化良好。

#### 节点级表现分解（Fig. 4 & 5）
- **Schema Compliance 显著提升**：
  - `json_schema`: Granite 从 0.67 → 0.98；Qwen 从 0.08 → 0.97
  - `json_sanity`: Granite 从 0.42 → 0.59；Qwen 从 0.05 → 0.58
- **语义质量同步提高**：
  - 最大提升出现在需要强上下文绑定的角色：
    - `classify_drift`: $r_{\text{judge}}$ 从 0.66 → 0.95
    - `detect_drift_operations`: 从 0.53 → 0.90
- **evaluate_risk 提升有限**（0.72 → 0.78），因依赖外部知识，通用裁判难以准确评判

---

### 端到端恢复性能（End-to-End Recovery）

#### 表 2：Type I Drift 下使用 GPT-4o 初始代理的结果（task_goal_completion %）

| Setting | D1 | D2 | D3 | Agg. |
|--------|----|----|----|------|
| No drift | 71.4 | 7.7 | 23.1 | 35.3 |
| Drift, no recovery | 0.0 | 0.0 | 0.0 | 0.0 |
| Granite base | 0.0 | 0.0 | 0.0 | 0.0 |
| GPT-4o recovery | 57.1 | 15.4 | 23.1 | **32.5** |
| **Granite trained** | **57.1** | **7.7** | **15.4** | **27.5** |

> 🔹 训练后的 Granite SLM 恢复了 **78% 的无漂移性能上限**（27.5 / 35.3）  
> 🔹 在难度 1 上完全匹配 GPT-4o 表现（57.1%）

#### 表 4：Type II Drift 下使用 GPT-4o 初始代理的结果

| Setting | D1 | D2 | D3 | Agg. |
|--------|----|----|----|------|
| No drift | 78.6 | 15.4 | 30.8 | 42.5 |
| Drift, no recovery | 0.0 | 0.0 | 0.0 | 0.0 |
| Granite trained | 85.7 | 0.0 | 14.4 | **35.0** |
| GPT-4o recovery | 85.7 | 15.4 | 30.8 | **45.0** |

> 🔹 尽管 Aggregate 得分略超 No-drift（可能因随机波动），但整体接近理想水平  
> 🔹 差距主要体现在 D2/D3，反映小模型生成警告语义丰富度仍逊于大模型

---

### 消融实验与关键发现
- **单一策略能否胜任多角色？✅ 是**
  - Granite 和 Qwen 均实现了跨节点一致的高分，证明角色条件提示 + 共享 GRPO 目标有效
- **模型规模的影响？✅ 可补偿**
  - Qwen 1.5B 初始能力远弱于 Granite 2B（R=0.56 vs 3.68），但训练后差距缩小至 4.80 vs 5.15（压缩 8.9x）
  - 支持“plug-and-play”理念：SLM 可在特定任务上逼近更大模型表现
- **输出长度 trade-off**
  - `tot_len` 奖励轻微下降（Granite: 0.14 → 0.08），因更丰富的 reasoning 导致总长度增加
  - 但被 $r_{\text{judge}}$ 提升所抵消，净收益为正

---

## 4. 关键结论和发现

### 主要结论
1. ✅ **行为漂移的结构化恢复可通过图状状态机实现**，将复杂诊断拆解为可管理的子任务。
2. ✅ **单一 SLM 可通过 GRPO + 角色提示在多个诊断角色间专业化**，无需为每节点训练独立模型。
3. ✅ **复合奖励（结构 + 语义）能有效引导模型同时优化形式与内容**，尤其在 schema 合规性和 grounded reasoning 方面显著提升。
4. ✅ **训练后的 SLM 可恢复大部分任务成功率**，在 Type I/II 漂移下达到甚至接近 GPT-4o 级别的恢复能力。
5. ✅ **方法具备低资源、可插拔、易集成特性**，适合部署在已有 LLM Agent 系统中作为安全层。

---

### 局限性
| 问题 | 说明 |
|------|------|
| **依赖外部漂移检测器提供 $k_0$** | 本框架假设漂移起点已知，未解决自主检测问题 |
| **LLM-as-Judge 存在领域盲区** | 对 API 可逆性的判断可能误判，尤其当模型虚构 inverse endpoint 时 |
| **尚未闭环执行纠正动作** | 当前来得及识别可逆操作，但未实际调用 inverse API 完成自动修复 |
| **评估范围有限** | 当前仅覆盖 Type I/II 漂移，未全面测试 Type III 及更多漂移模式 |
| **潜在 Reward Hacking 风险** | 模型可能学会“讨好裁判”的表达风格而非真正提升事实准确性 |

---

### 未来工作方向
1. **增加 Correction Node**：扩展图结构，在 `n4` 后加入专门节点执行 reverse API 调用，形成闭环恢复系统。
2. **环境感知奖励（Environment-grounded reward）**：在 AppWorld 模拟器中回放恢复决策，以实际状态还原程度作为奖励信号。
3. **文档支撑奖励（Documentation-grounded reward）**：强制模型引用 API 文档来支持 reversibility 判断，防止幻觉。
4. **跨漂移类型泛化评估**：在 AgentDojo 等平台测试不同类型漂移下的鲁棒性。
5. **人机协作接口设计**：构建 human-in-the-loop escalation UX，提升人工干预效率。
6. **与边界防火墙协同**：结合 agent-tool boundary firewalls [Debenedetti et al., 2025] 实现预防+恢复双重防护。

---

> 📌 **总体评价**：本文提出了一个实用性强、结构清晰、训练高效的 LLM Agent 漂移恢复框架，推动了从“被动容错”向“主动诊断+结构化恢复”的演进，为构建安全可靠的自主代理系统提供了重要路径。

</details>

---

### 15. [ScienceFlow: A long-horizon agent for ML research, scientific discovery and beyond](https://arxiv.org/abs/2608.14354)

**Authors**: Mingming Zhao, Jiqian Dong, Kangping Xu, Zadid Hasan, Chengrui Fan, Shan Jiang, Shuai Mao, Ting Lingya, Linyi Zou, Tailin Zhou, Yun Hin Chan, Wenkai Zhang, Zhanhong Zhou, Guowei Huang, Hongliang Li, Wenjing Cun, Zhitang Chen, Mingxuan Yuan, Yanhui Geng  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.14354v1  

#### Abstract
Enabling LLM agents to sustain productive, stable, and goal-aligned research over extended horizons is a central challenge for autonomous machine learning and scientific discovery, as progress hinges on continuously managing evolving state, exploration decisions, and computational resources. Pioneer...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ScienceFlow: A long-horizon agent for ML research, scientific discovery and beyond

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **autoresearch agents** 在执行长周期（long-horizon）科研任务时面临以下核心挑战：
- **缺乏连续性机制**：无法有效管理研究状态，难以从失败路径中恢复。
- **探索效率低下**：在死胡同（dead ends）上浪费大量计算资源。
- **资源分配不合理**：缺乏基于验证进展（validated progress）的价值驱动计算分配策略。

这些问题导致搜索效率低、资源浪费严重，并降低了最终成功的概率。

### 提出的新方法：ScienceFlow
ScienceFlow 是一个端到端的 **autoresearch agent framework**，其核心创新在于通过三个关键组件实现对长周期研究工作的组织和控制：

1. **可恢复的可执行工作区（Recoverable Executable Workspaces）**
   - 将研究过程中的代码、数据、模型检查点等所有状态封装为“可执行状态”（executable state），并存入状态归档（state archive）。
   - 研究进展被表示为可恢复的可执行状态，支持高效探索、修订和执行。

2. **通过重锚定的可执行状态转移（ESTRA: Executable-State Transition through Re-Anchoring）**
   - 在研究段（research segment）边界，由 ESTRA 决策选择下一个锚定状态（anchor state）和研究方向（extend 或 redirect）。
   - 锚定状态可以是当前实时状态（live state）或归档中的历史状态，从而实现从成功路径继续或从失败路径回溯。

3. **证据感知的执行控制器（Evidence-Aware Execution Control）**
   - 控制器根据资源可用性、剩余预算和已验证的研究进展，动态地为物理任务（physical jobs）分配计算资源。
   - 实现了科学决策（由研究工作者做出）与资源执行控制的分离。

### 相比现有方法的优势
- **更强的鲁棒性和恢复能力**：通过 ESTRA 可以从归档中恢复状态，避免陷入局部最优或死胡同。
- **更高的资源利用效率**：证据感知的控制器能及时终止无价值的任务，将资源重新分配给更有希望的方向。
- **更一致的长期进展**：将研究状态、轨迹决策和物理执行统一在一个框架内，解决了现有系统中各部分视图不一致的问题。

---

## 2. 核心实验方法和设置

### 数据集
论文在三大类可执行研究任务上进行了评估：
- **机器学习工程（Machine Learning Engineering）**：使用 **MLE-bench**，包含 75 个真实的 Kaggle 竞赛任务，涵盖表格、视觉、语言、音频和时间序列等多种模态。
- **数学与工程优化（Mathematical and Engineering Optimization）**：
  - 连续优化：**Circle Packing**, **Ratio Minimization**, **Uncertainty Inequality**。
  - 组合调度优化：**SpOC4 KTTSP** 挑战赛的 easy、medium 和 hard 轨道。
- **科学建模（Scientific Modeling）**：使用 **SciModelingBench**，包含 12 个任务，涉及 DNA 结合、RNA/蛋白质设计、超导材料、临床毒理学和具身控制等领域。

### 实验设置和评估指标
- **评估协议**：
  - **MLE-bench**：24 小时预算，最多 2 个 GPU。
  - **数学优化**：12 小时墙钟预算。
  - **SciModelingBench**：2 小时预算。
- **主要评估指标**：
  - **MLE-bench**：**Any-Medal rate**（提交达到原始 Kaggle 排行榜铜牌、银牌或金牌阈值的比例）。
  - **数学优化**：直接比较目标函数的最优值。
  - **SciModelingBench**：使用 **best-K mean (BKM)**、**normalized enrichment (NE)** 和 **global NDCG**，并计算组平衡得分（group-balanced score）。

### 基线方法对比
- **MLE-bench**：对比了 **Iris**, **MLEvolve**, **AIBuildAI**, **CAIR MARS+** 等多个先进系统。
- **数学优化**：对比了 **AlphaEvolve**, **ThetaEvolve**, **FM Agent** 等。
- **SciModelingBench**：对比了 **OpenCode**, **Pi**, **Codex**, **Claude Code** 等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **MLE-bench**：在完整的 75 项任务上，ScienceFlow 达到了 **70.22±1.18%** 的 Any-Medal 率，显著优于最强基线（65.30%），提升了 **4.92 个百分点**。
- **SciModelingBench**：取得了 **54.41** 的组平衡得分，是所有评测代理中的最佳表现。
- **数学优化**：
  - 在 **Circle Packing** 上取得与最强结果相近的数值（2.6359830849）。
  - 在 **Ratio Minimization** 上与最强结果持平。
  - 在 **Uncertainty Inequality** 上，将最强的 Hermite 基上界减少了 **2.5%**。
  - 在 **KTTSP-hard** 赛道上排名第三。

### 与基线方法的对比结果
- 在 MLE-bench 的所有难度等级上均表现出色，尤其在 Medium 难度上优势最大（74.56% vs 最强基线 64.04%）。
- 在 SciModelingBench 的 12 项任务中，有 5 项排名第一，6 项排名第二，仅 1 项未进入前二。

### 消融实验结果
论文对两个核心组件进行了消融研究（ablation study）：
- **移除 ESTRA**：在 MLE-bench Lite 上的 Any-Medal 率从 **80.30%** 下降到 **66.67%**。
- **移除 Evidence-Aware Execution Control**：Any-Medal 率下降到 **69.70%**。
- **综合分析**：消融实验证明，ESTRA 对于快速获得奖牌至关重要，而执行控制器对于长时间运行的任务影响更大。两者共同作用才能实现最佳性能。

---

## 4. 关键结论和发现

### 主要发现
1. **有效的长期研究依赖于三者的协同进化**：可执行状态的持久化、研究轨迹的自适应调整以及与已验证进展对齐的执行控制，这三者缺一不可。
2. **ScienceFlow 具有强大的跨领域泛化能力**：其提出的一套长周期研究抽象（long-horizon research abstractions）能够成功应用于机器学习、科学建模和数学优化等多个截然不同的领域。
3. **状态管理和资源控制是瓶颈**：相比单纯的推理能力提升，高效的 **state management**、**adaptive exploration** 和 **objective-aligned execution** 才是扩展自主研究超越短周期交互的关键。

### 方法的局限性
- 论文未明确讨论 ScienceFlow 在极端复杂或需要人类直觉的开放性科学发现中的表现。
- 虽然强调了“污染控制”，但仍承认无法完全排除预训练数据中可能存在的候选标签记忆问题。

### 未来工作方向
- 进一步探索如何让多个异构（heterogeneous）的研究工作者进行协作。
- 研究如何将 ScienceFlow 的框架应用于更广泛的科学发现场景，如湿实验室（wet-lab）自动化。
- 开发更智能的 ESTRA 策略，以更好地权衡探索（exploration）与利用（exploitation）。

</details>

---

### 16. [Tripwire: Triggering Aligned Refusal via Statistically Certified Safety Neurons](https://arxiv.org/abs/2608.14392)

**Authors**: Wei Zhao, Zhe Li, Peixin Zhang, Jun Sun  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.14392v1  

#### Abstract
Neuron- and path-level interventions offer the finest-grained route to defending large language models (LLMs) against jailbreak attacks, yet existing methods fall short of this promise, i.e., they often compromise model utility significantly. Specifically, one line of work suppresses toxic neurons t...

---

### 17. [Bootstrapping Niche Multilingual Code Translation via Reinforcement Learning with Execution-Based Verifiable Supervision](https://arxiv.org/abs/2608.13854)

**Authors**: Kouki Yuki, Jie Zeng, Kyoko Ogawa, Ryunosuke Ikeda, Yohei Kobashi, Takeshi Kojima, Ikuya Yamada, Yusuke Iwasawa, Yutaka Matsuo  
**Category**: cs.CL  
**Published**: 2026-08-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.13854v1  

#### Abstract
Code translation must preserve executable behavior across many programming languages, yet neural code translation has largely focused on a few popular languages such as C++, Java, and Python. This leaves a niche, many-to-many setting where parallel supervision is sparse, producing plausible but non-...

---

### 18. [Stochastic Control Policies for Robust Molecular Transition Path Sampling](https://arxiv.org/abs/2608.13800)

**Authors**: Jingqian Liu, Yu-Hsiang Wang, Yanru Qu, Ge Liu  
**Category**: cs.LG  
**Published**: 2026-08-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.13800v1  

#### Abstract
Transition path sampling (TPS) aims to efficiently generate rare molecular transition trajectories between metastable states and is essential for understanding biomolecular mechanisms. Beyond traditional molecular dynamics (MD)-based sampling, machine learning has become central to state-of-the-art ...

---

### 19. [MedMix: Specialization-Consistent Federated Sparse MoEs under Modality Heterogeneity](https://arxiv.org/abs/2608.13911)

**Authors**: Adiba Orzikulova, Dong Min Kim, Jaehong Yoon, Sung-Ju Lee  
**Category**: cs.LG  
**Published**: 2026-08-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.13911v1  

#### Abstract
Federated multimodal medical AI faces modality heterogeneity at both the client and sample levels: clients may systematically lack access to specific modality types, while individual records within the same client may contain different partial modality subsets. Sparse Mixture-of-Experts (MoE) archit...

---

### 20. [Probabilistic indirect models for undrained shear strength: addressing significant data missing and variability with advanced imputation and machine learning techniques](https://arxiv.org/abs/2608.13934)

**Authors**: Haibin Xiong, Shaoheng Dai, Peng Lan, Xuzhen He, Chenxi Tong, Sheng Zhang, Daichao Sheng  
**Category**: cs.LG  
**Published**: 2026-08-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.13934v1  

#### Abstract
Accurate prediction of undrained shear strength (su) is crucial for geotechnical design, but is often hampered by substantial uncertainty in traditional empirical methods. This study uses the CLAY/10/7490 global database to develop probabilistic indirect models to predict su based on Atterberg limit...

---

### 21. [Towards Efficient Multimodal and Multilingual Opinion Extraction for STI: A QLoRA-Based Fine-Tuning Approach](https://arxiv.org/abs/2608.14152)

**Authors**: Sheng Hong, Xuanqi Wang, Jiacheng Wang, Yuwei Wang  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.14152v1  

#### Abstract
Recent advances in large language models (LLMs) have reshaped semantic analysis. Opinion Extraction (OE) for Science and Technology Intelligence (STI) requires concise core opinions from large information streams. Off-the-shelf models struggle to filter noise from these streams and show limited stru...

---

### 22. [HERMES: a multi-agent framework for structured knowledge extraction from ultra-long documents in geoscience](https://arxiv.org/abs/2608.14055)

**Authors**: Ziqi Song, Zongyuan Xiang, James G. Ogg, Bruce S. Lieberman, Gabi Ogg, Natalia L\'opez Carranza, Wen Du, Yufei Ye, Shuan Li, Zhong Peng, Shaoqi Yu, Juye Wei, Ying Zhou, Jieping Ye, Jiang Yang  
**Category**: cs.CL  
**Published**: 2026-08-17  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.14055v1  

#### Abstract
Authoritative scientific knowledge in geoscience remains largely trapped in legacy monographs and historical literature, where unstructured text and complex layouts hinder computational access. We introduce HERMES, a scalable multi-agent framework that extracts structured data from ultra-long scient...

---

### 23. [Validating LLM-Modernized Scientific Software Through Differential Fault Injection](https://arxiv.org/abs/2608.14527)

**Authors**: Evan Coleman, Yuzhong Shen, Masha Sosonkina, Peng Xu  
**Category**: cs.DC  
**Published**: 2026-08-17  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.14527v1  

#### Abstract
Large language model (LLM) agents are increasingly used to modernize the legacy Fortran underlying production scientific software, but validation of these transformations emphasizes nominal executions and may not test whether a modernization preserves the original code's response to faults, perturba...

---

### 24. [Modular Cognitive Architecture Emerges in Large Language Models](https://arxiv.org/abs/2608.13567)

**Authors**: Pengrui Han, Jacob Andreas, Evelina Fedorenko, Andrea Gregor de Varda  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.13567v1  

#### Abstract
The human brain exhibits a striking degree of functional specialization, with distinct networks supporting language, formal reasoning, reasoning about other minds, and reasoning about the physical world. Is this modular organization a fundamental principle of how intelligent systems must be built, o...

---

### 25. [Reward Machines for Signal Temporal Logic](https://arxiv.org/abs/2608.13625)

**Authors**: Alper Kamil Bozkurt, Shangtong Zhang, Yuichi Motai  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.13625v1  

#### Abstract
Signal temporal logic (STL) provides a formal language for specifying real-time properties of real-valued observations, along with a quantitative robustness score for monitoring satisfaction. Control synthesis from STL specifications is of interest since manual controller design becomes infeasible a...

---

### 26. [Benchmarking data-driven material models on the classic Treloar dataset](https://arxiv.org/abs/2608.14063)

**Authors**: Hagen Holthusen, Moritz Flaschel, Denisa Martonov\'a, Ellen Kuhl  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.14063v1  

#### Abstract
Machine learning is rapidly reshaping constitutive modeling, offers new ways to learn material behavior directly from experimental data, and challenges long-established modeling paradigms. But with a growing number of machine-learning-based approaches available, how do they compare in practice? In t...

---

### 27. [Reinforcement Learning-Based Production Scheduling in an Industry-Based Coating Scenario Using the Digital Model Playground](https://arxiv.org/abs/2608.14122)

**Authors**: Arne Kr\"oger, Ralf Buscherm\"ohle, Wilhelm Hasselbring, Henrik Wilbers  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.14122v1  

#### Abstract
Production scheduling in complex manufacturing environments is challenging when sequence-dependent setup times, stochastic disturbances, and due-date constraints must be addressed simultaneously. While reinforcement learning (RL) methods have shown promising results in research, most studies rely on...

---

### 28. [Polaris : Multi Agentic System for Conversational Enterprise Analytics](https://arxiv.org/abs/2608.14246)

**Authors**: Varuni H K, Soham Sarkar, Jay Kumar, Goutham Krishnan, Tanvi Johari, Avinash Bharadwaj, Santosh Hegde  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.14246v1  

#### Abstract
In today's fast-paced environment, the ability to swiftly access, understand, and act on data is no longer optional; it is essential. Yet most organizations remain data-rich but insight-poor, constrained by the complexity of querying, interpreting, and explaining enterprise-scale information. We pre...

---

### 29. [Intern-S2-Mobius: Foundation Model with Decoupled Knowledge and Reasoning](https://arxiv.org/abs/2608.14290)

**Authors**: Kai Chen, Jifeng Ding, Ning Ding, Jiaye Ge, Lixin Gu, Yicheng Gu, Qipeng Guo, Ermo Hua, Haian Huang, Haozheng Hou, Jie Hou, Xiangyu Hong, Che Jiang, Minxi Jin, Cheng Liang, Dahua Lin, Dawei Liu, Kuikun Liu, Chengqi Lv, Haijun Lv, Han Lv, Ningsheng Ma, Biqing Qi, Jianmin Qian, Shiya Su, Youbang Sun, Huanze Tang, Zhongbo Tian, Hanjing Wang, Rui Wang, Ting Wang, Yi Wang, Baiting Wu, Jun Xu, Bowen Yang, Hui Wang, Weida Wang, Haochen Ye, Jiashuo Yu, Shan Yu, Xiaoyi Yu, Qirui Zeng, Qi Zhang, Ming Zhang, Wenwei Zhang, Bowen Zhou, Xinyu Zhou  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.14290v1  

#### Abstract
We introduce Mobius-v0, an architecture that comprises a globally shared Memory (FFN) that stores knowledge vectors and multiple Reasoners (Self-Attn) that iteratively achieve compositional reasoning. Using hidden states as cache and carrier, reasoners repeatedly query memory for required knowledge-...

---

### 30. [Twin: Playing an Unknown Game with a Test-Time Digital Twin](https://arxiv.org/abs/2608.14490)

**Authors**: Alexy Skoutnev, Kirill Acharya, Gaston Longhitano, Madeleine Udell, Kevin Ellis, Iddo Drori  
**Category**: cs.AI  
**Published**: 2026-08-17  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.14490v1  

#### Abstract
We present a Test-time World-model Inference (Twin) system, in which a frontier coding agent writes an executable world model for completing continual learning tasks, such as ARC-AGI-3 games. Traditional approaches hand-engineer such models, one custom design per task. Each game hides its rules and ...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-30 11:24:38 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [BASE: Batch-Aware Selection of Experts Using Predicted Removal Error for Efficient MoE Decoding](https://arxiv.org/abs/2609.36222)

**Authors**: Ali Abbasi, Justin Shi, Soheil Kolouri  
**Category**: cs.LG  
**Published**: 2026-09-30  
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

在 **Mixture-of-Experts (MoE)** 大语言模型中，虽然每个 token 只激活少量专家（sparsity），但在 **batched decoding** 场景下，不同请求选择的专家集合差异大，导致整个 batch 需要加载大量不同的专家权重。这使得 **内存带宽成为瓶颈**，显著降低推理效率。

现有方法（如 SERE、OEA）通常基于 router 权重或专家统计信息进行选择，但存在以下问题：
- **未考虑专家对当前 batch 输出的实际影响**
- **缺乏对“移除该专家会带来多大误差”的直接建模**
- **无法动态适应不同 token 的上下文变化**

### ✅ 提出的新方法和创新思路

作者提出 **BASE (Batch-Aware Selection of Experts)**，其核心思想是：

> **以“预测移除某专家后对 MoE 层输出造成的误差”作为专家重要性评分标准，在固定 batch-level 专家预算下选择最重要的专家。**

#### 主要创新点包括：

1. **基于 Removal Error 的专家评分机制**
   - 定义专家 `u` 的 removal cost 为：  
     $$
     D_u = \sum_{t: u \in R_t} w_{t,u}^2 \|E_u(z_t)\|^2
     $$
     即所有路由到该专家的 token 中，其加权输出能量之和。
   - 证明忽略 cross terms 后的 additive approximation 在实践中足够准确。

2. **轻量级线性预测器估计输出能量**
   - 在离线校准阶段训练一个 **per-expert 的线性模型**，输入为 token 表示 $z_t$，输出预测 $\log \|E_u(z_t)\|$。
   - 推理时无需执行专家即可快速估算其潜在贡献。

3. **端到端 GPU 内核优化实现**
   - 开发了定制化的 **Triton GPU kernels**，高效完成：
     - 成本预测（scoring）
     - 批次聚合（aggregation）
     - 专家选择（selection）
     - 回填策略（backfill）

4. **批感知（batch-aware）而非 token 独立的选择策略**
   - 综合所有 token 的预测 removal cost 进行全局排序，选出 top-M 专家供整个 batch 共享。

### ✅ 相比现有方法的优势

| 方面 | BASE | 现有方法（SERE/OEA/ExFold等） |
|------|------|-------------------------------|
| **选择依据** | 预测 removal error（与输出误差直接相关） | Router 权重 / 固定校准统计量 |
| **动态性** | 动态预测每 token 的专家输出能量 | 使用静态均值或 norm |
| **准确性** | 更贴近真实误差，提升质量 | 忽略上下文变化，易误判 |
| **效率** | 减少不必要的专家加载，提高吞吐 | 易因低重叠度导致高 fetch 开销 |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集

在 **8 个生成式基准任务** 上进行全面评估：

| 数据集 | 类型 | 指标 |
|--------|------|------|
| **CMMLU** | 中文多任务理解 | Accuracy |
| **BBH (BIG-Bench Hard)** | 复杂推理 | Accuracy |
| **MATH**, **GSM8K**, **MATH-401** | 数学解题 | Accuracy |
| **BoolQ**, **MBPP**, **HumanEval** | 是非判断 / 编码 | Accuracy / Pass Rate / Pass@1 |

最终报告 **8 项平均得分（Avg.）** 和 **端到端吞吐（tok/s）**。

---

### ✅ 实验设置

| 参数 | 设置 |
|------|------|
| **模型** | Qwen3-30B-A3B, Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite |
| **硬件** | 单张 NVIDIA H100 80GB GPU |
| **框架** | vLLM 0.9.2 (V0 engine), bf16 精度 |
| **批大小** | decode batch size = 16 |
| **prefill** | 使用 dense 模型（无剪枝） |
| **生成长度** | 强制生成 384 tokens（禁用 EOS 提前终止） |
| **采样参数** | temp=0.7, top_p=0.8, top_k=20, seed=0 |

> 🔁 **对比协议**：将 BASE 与其他方法在 **相同吞吐水平下比较质量**，确保公平。

---

### ✅ 基线方法对比

| 方法 | 简介 |
|------|------|
| **Dense** | 不跳过任何专家，原始完整模型 |
| **SERE** | 基于相似性的专家重路由，union-based 活跃集构建 |
| **OEA** | Opportunistic Expert Activation，取 top-k 路由专家并填充已有专家 |
| **Lynx** | 结合 router confidence 与 batch 内流行度进行裁剪 |
| **ExFold** | 使用校准期间测量的 expert output norm 加权选择 |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Table 1）

#### 📊 在 **Qwen3-30B-A3B** 上的表现（受限模式，M≈10）

| 方法 | 平均 Accuracy | 吞吐 (tok/s) |
|------|----------------|-------------|
| Dense | 82.50 | 845.3 |
| OEA (ko=1) | 43.01 | 1372.5 |
| Lynx | 46.87 | 1304.9 |
| **BASE (M=10)** | **76.38** | **1463.5** |

> 💡 **相比最强 baseline（Lynx），BASE 提升 +29.51 分，同时更快！**

#### 📈 在宽松预算下的表现（M=16–19）

| 方法 | 平均 Accuracy | 吞吐提升 |
|------|----------------|----------|
| **BASE (M=16)** | 82.47（仅比 dense 低 0.03） | 比 dense 快 **60%** |
| **BASE (M=19)** | 54.02（接近 dense 的 53.59） | 比 dense 快 **36%** |

> ✅ 在高质量区间，**BASE 实现近似 dense 性能的同时获得显著加速**。

---

### ✅ 与基线方法的对比总结

| 对比维度 | 结果 |
|---------|------|
| **质量优势** | 在 tight budget 下，平均提升 **3.0 ~ 29.5 points** |
| **速度优势** | 在 comparable 质量下，最高快 **60%** |
| **帕累托前沿** | 在所有架构上均提供最佳 quality-throughput tradeoff |

> 📉 图 3 显示：**BASE 在各种 batch size 下都保持稳定优势，尤其在小 batch 时领先更明显**。

---

### ✅ 消融实验结果

#### 🔍 **Selection Rule Ablation（Table 2）**

在 Qwen3-30B-A3B 上控制活跃集大小一致，仅改变评分方式：

| 评分方式 | 保留能量 (%) | 平均 Accuracy |
|--------|--------------|----------------|
| Top-1 Union | 85.91 | 58.68 |
| Router Weight Sum | 82.61 | 42.87 |
| Squared Router Weight | 90.98 | 55.50 |
| Static Expert Energy | 92.65 | 64.46 |
| **BASE (Linear Predictor)** | **96.47** | **73.50** |
| Oracle (Ideal) | 100.00 | 73.58 |

> ✅ **BASE 接近 oracle 性能，远超其他启发式规则**

#### 🔁 **Fill Rule Ablation（Table 3）**

比较三种处理缺失专家的方式（M=10）：

| 策略 | Average Accuracy | tok/s |
|------|------------------|-------|
| Drop（直接丢弃） | 74.20 | 1473.4 |
| Substitution（替换为最相似专家） | 27.49 | 1477.0 |
| **Backfill（回填下一偏好专家）** | **76.45** | 1479.3 |

> ❌ 替换效果极差 → 支持论文观点：**co-routed expert outputs nearly orthogonal**
>
> ✅ **Backfill 是最优策略**：不增加 memory traffic，小幅增加 compute，收益显著

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Cross terms 可安全忽略**
   - 尽管 exact error 包含交叉项，但实验证明 additive approximation 导致的选错率 < 3.2%
   - 原因：同一 token 路由的多个专家输出方向接近正交（mean |cosine| ≈ 0.04）

2. **专家输出不能互相替代**
   - 替换（substitution）严重损害性能，说明不能用“相似专家”代替原专家

3. **动态预测优于静态统计**
   - 使用校准时的平均输出 norm（如 ExFold）不如在线预测当前 token 的输出能量

4. **BASE 实现高效 batch-aware selection**
   - 在相同吞吐下，大幅超越现有方法的质量
   - 在高质量要求下，仍可比 dense inference 快 22–60%

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **需要额外离线校准** | 需要在部署前运行 calibration，耗时约 10–40 分钟（取决于模型大小） |
| **增加少量路由开销** | BASE 增加约 3–7% 的 decode 时间（主要来自 scoring kernel） |
| **依赖 MoE 架构特性** | 当前设计针对 token-choice routing MoE，可能不适用于所有变体 |

---

### 🔮 未来工作方向

1. **减少校准成本**
   - 设计更高效的 predictor 学习方式（如蒸馏、共享参数）
2. **扩展至 Prefill 阶段**
   - 当前仅用于 decoding，未来可探索 prefill 中的应用
3. **结合 expert offloading / prefetching**
   - 与系统级优化（如 Fiddler、Pre-gated MoE）协同设计
4. **支持更多 MoE 路由机制**
   - 如 expert choice routing 或 top-p sampling

---

## ✅ 总结一句话

> **BASE 通过预测“移除专家带来的输出误差”，实现了更精准、动态、高效的 batch-aware 专家选择，在几乎不损失质量的前提下显著提升了 MoE 模型的 decoding 吞吐，是当前最先进的免训练 MoE 推理加速方案之一。**

</details>

---

### 2. [FineSID: Scalable and Efficient Semantic Identifier Learning for Generative Recommendation](https://arxiv.org/abs/2609.36670)

**Authors**: Song-Li Wu, Weinan Gan, Zhaocheng Du, Xianquan Wang, Jingyi Wang  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.36670v1  

#### Abstract
A critical prerequisite of generative recommendation is designing semantic identifiers (SIDs) that are both scalable to large item sets and efficiently learnable. Existing SID learning methods fundamentally rely on Top-1 hard assignment during vector quantization. While heuristic strategies -- such ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# FineSID: Scalable and Efficient Semantic Identifier Learning for Generative Recommendation —— 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在生成式推荐（Generative Recommendation, GR）中，**Semantic Identifier (SID)** 是将连续物品表示映射为离散 token 的关键桥梁。然而，现有 SID 学习方法普遍依赖 **Top-1 硬分配（hard assignment）** 进行向量量化，导致以下根本性瓶颈：

- **梯度稀疏（Sparse gradient propagation）**：只有被选中的单个码字（codeword）接收梯度更新，其余大量码字长期“失活”，造成 **codebook collapse** 和严重的 **SID collision**（不同物品映射到相同 SID 序列）。
- **语义失真（Semantic drift）**：为缓解碰撞而采用的启发式策略（如强制重分配、聚类初始化）破坏了语义一致性，牺牲了推荐质量。

因此，现有方法在 **scalability**（应对大规模物品库）和 **efficiency**（低推理开销）之间难以兼顾。

---

### 提出的新方法与新思路
作者提出 **FineSID**，一种统一的、无需复杂初始化先验的量化框架，其核心思想是：

> **打破 Top-1 硬分配限制，实现全码本（full-codebook）范围内的细粒度梯度传播，同时保持输出 SID 的严格离散性以支持自回归生成。**

#### 核心组件：
1. **Global-Local Quantization (GLQ)**  
   - **Local Refinement Quantization (LRQ)**：通过 soft assignment 分布实现密集梯度流，使所有码字（包括长尾项）都能参与学习。
   - **Global Anchor Quantization (GAQ)**：利用指数移动平均（EMA）跟踪码字使用频率，动态调整更新速率，防止高频码主导训练，提升码本平衡性。

2. **Quantization Semantic Consistency Module (QSCM)**  
   引入双向对齐损失，强制约束连续嵌入与离散 SID 之间的语义一致性，防止因全局优化导致的语义漂移。

---

### 相比现有方法的优势
| 维度 | 传统方法 | FineSID |
|------|--------|---------|
| **梯度传播** | 仅 Top-1 码字更新 → 梯度稀疏 | 全码本软更新 → 密集梯度流 |
| **码本利用率** | 低频码失活严重 → 利用率低 | 高且均衡的码本利用 |
| **语义保真度** | 启发式重分配破坏语义 | QSCM 显式维护语义一致性 |
| **初始化依赖** | 依赖 K-means 等强初始化 | 对初始化鲁棒，随机初始化即有效 |
| **效率与扩展性** | 扩展码本易崩溃；加辅助 token 增加解码成本 | 支持大码本高效训练，不增加序列长度 |

---

## 2. 核心实验方法和设置

### 数据集
在三个真实世界公开数据集上进行验证：
- **Instrument**（乐器）
- **Scientific**（工业科学）
- **Game**（视频游戏）

数据统计如下：

| Dataset      | #Users  | #Items  | #Interactions | Sparsity  |
|--------------|---------|---------|---------------|-----------|
| Instrument   | 57,439  | 24,587  | 511,836       | 99.964%   |
| Scientific   | 50,985  | 25,848  | 412,947       | 99.969%   |
| Game         | 94,762  | 25,612  | 814,586       | 99.966%   |

均采用 **5-core** 过滤，并按时间顺序划分训练/验证/测试集（留一法）。

---

### 实验设置与评估指标

#### 评估指标
- **Recall@K**（K=5,10）
- **NDCG@K**（K=5,10）

#### 模型架构
- **Backbone**: T5（6层编码器+解码器，hidden size=128）
- **Tokenizer**: RQ-VAE，3个量化层级，每层 256 个码字，维度 128
- **优化器**: AdamW，weight decay=0.05
- **Beam Search Size**: 200

#### 基线方法（Baselines）
分为两类：
1. **传统序列推荐模型**：Caser, GRU4Rec, SASRec, BERT4Rec 等
2. **生成式推荐模型**：
   - 基于辅助目标：SaviorRec, LETTER
   - 基于扩展标识符空间：OneSearch, CAR

---

## 3. 主要实验结果和性能指标

### 整体性能对比（Table 2）

FineSID 在所有数据集和所有指标上均显著优于所有基线方法（p < 10⁻⁴），表现如下：

| Method     | Instrument (NDCG@10) | Scientific (NDCG@10) | Game (NDCG@10) |
|------------|-----------------------|------------------------|----------------|
| CAR        | 0.0241                | 0.0241                 | 0.0507         |
| **FineSID** | **0.0388**↑           | **0.0294**↑            | **0.0594**↑    |

> ✅ **平均提升超过 20%+，尤其在 Game 数据集上提升明显。**

---

### 消融实验（Ablation Study, Table 3）

验证各模块贡献：

| Variant          | Instrument (NDCG@10) | Scientific (NDCG@10) | Game (NDCG@10) |
|------------------|-----------------------|------------------------|----------------|
| w/o GLQ          | 0.0334                | 0.0241                 | 0.0518         |
| w/o QSCM         | 0.0337                | 0.0246                 | 0.0522         |
| w/o LRQ / w/o GAQ| ~0.0346–0.0351         | ~0.0251–0.0254          | ~0.0526–0.0528 |
| **FineSID (Full)** | **0.0388**             | **0.0294**              | **0.0594**     |

> 🔍 **GLQ 贡献最大**，说明全局优化机制是性能提升的关键。

---

### 其他关键实验证据

#### ✅ **码本利用率分析（Figure 2）**
- FineSID 实现高达 **82.42% 的码本利用率**，远超 CAR（~48%）和 SaviorRec（~82%，但靠正则化牺牲语义）。
- **Balance Rate** 达 0.4812，表明码字激活分布更均匀。

#### ✅ **初始化鲁棒性（Table 4）**
- 即使使用 **random initialization**，FineSID 仍能取得最佳性能。
- 表明 FineSID 不依赖预训练语义先验或聚类初始化，具备强自适应能力。

#### ✅ **训练效率（Table 5）**
- **训练时间减少 >50%**（如 Game 上从 CAR 的 24.5h 降至 9.3h）
- 更快收敛得益于全局梯度流动，避免无效训练阶段。

#### ✅ **高维可扩展性（Figure 3）**
- 随着码本尺寸增大（至 4096），FineSID 仍保持高利用率；
- 其他方法出现严重 **codebook collapse**。

#### ✅ **长尾性能优势（Table 8）**
- 在 **Tail items** 上 NDCG@5 提升尤为显著（如 ML-1M 上从 CAR 的 0.0791 → FineSID 的 0.0854）。
- 表明 FineSID 有效缓解了 popularity bias。

---

## 4. 关键结论和发现

### 主要发现
1. **Top-1 硬分配是 SID 学习的根本瓶颈**，现有启发式手段无法根治梯度稀疏问题。
2. **FineSID 通过 GLQ + QSCM 实现了真正的全局优化**：
   - GLQ 提供密集梯度流与动态平衡机制；
   - QSCM 保障语义一致性，防止优化过程中的语义漂移。
3. **无需辅助目标或额外 token 即可实现高性能**，兼顾了 scalability 与 efficiency。
4. **对初始化完全鲁棒**，适用于冷启动或无强语义特征场景。
5. **特别适合处理长尾物品和大规模商品目录**，具有实际部署潜力。

---

### 方法的局限性
- 当前基于 RQ-VAE 架构，可能限制极端压缩下的表达能力。
- 尚未探索跨域（cross-domain）或指令控制（instruction-driven）推荐场景。
- 虽然推理速度可控，但在超大规模（百万级 items）下的 serving 成本仍需工程优化。

---

### 未来工作方向
1. **扩展至跨域推荐**：利用 FineSID 的通用性构建统一标识系统。
2. **结合自然语言指令**：实现可控生成式推荐（controllable recommendation）。
3. **探索更高效的 hierarchical structure**：进一步降低 SID 序列长度。
4. **集成到端到端 LLM 推荐框架**：已在 Qwen 系列上初步验证有效性（见 Table 7）。

---

> 📌 **总结一句话**：  
> **FineSID 通过引入全码本细粒度优化与语义一致性约束，从根本上解决了生成式推荐中 SID 学习的梯度稀疏与语义失真问题，在不牺牲效率的前提下实现了更高性能、更强鲁棒性和更好可扩展性。**

</details>

---

### 3. [Cobalt: Leveraging Expert Co-activation for Efficient Distributed MoE Training](https://arxiv.org/abs/2609.36959)

**Authors**: Junkang Zhou, Xinyi Liu, Fangcheng Fu  
**Category**: cs.DC  
**Published**: 2026-09-30  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.36959v1  

#### Abstract
Mixture-of-Experts (MoE) has increasingly become a mainstream approach for scaling large language models, as it expands model capacity while keeping computation cost nearly constant. Training large-scale MoE models relies on Expert Parallelism (EP), which distributes expert replicas across GPUs and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**COBALT: Leveraging Expert Co-activation for Efficient Distributed MoE Training**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大规模 **Mixture-of-Experts (MoE)** 模型的分布式训练中，**Expert Parallelism (EP)** 是主流并行策略，但其效率受限于两大系统瓶颈：
- **跨节点通信开销高**：由于专家分布在不同 GPU 上，需通过 all-to-all 通信传输 token，而跨节点带宽远低于节点内带宽（如 B200 中 NVLink 达 1.8TB/s，InfiniBand 仅 400GB/s）。
- **计算负载不均衡**：某些专家被频繁选中，导致部分 GPU 成为“straggler”。

现有方法通常基于**单个专家的工作负载统计**进行优化，忽略了专家之间可能共享通信路径的潜力。

---

### 🚀 提出的新方法与创新思路
本文提出 **COBALT**，首次从 **expert co-activation（专家共激活）** 视角出发，利用以下观察设计高效 MoE 训练框架：

> **许多 token 同时激活多个专家对（expert pairs），这些共激活现象随训练演进而增强。**

#### 主要创新点：
1. **新视角（New Perspective）**
   - 首次系统性地提出并验证了 **expert co-activation** 在 MoE 训练中的普遍存在，并将其作为优化通信与负载平衡的关键信号。

2. **两阶段专家布局规划器（Two-stage Expert Layout Planner）**
   - **Stage 1: 周期性全局规划（Periodic Global Planning）**
     - 利用历史的专家共激活频率（Ochiai similarity）和工作负载 EMA，通过 **MILP（Mixed-Integer Linear Programming）** 求解最优专家复制与放置方案，将常被共激活的专家尽量部署在同一节点，减少跨节点通信。
   - **Stage 2: 步级节点内重平衡（Per-step Intra-node Rebalancing）**
     - 每步微调同一节点内的副本分布，采用贪心 bin-packing 算法实现快速负载均衡，避免局部过载。

3. **通信感知的任务分配（Communication-aware Task Assignment）**
   - 给定当前专家布局，为每个 token 动态选择最少数量的远程节点来服务其 top-K 专家，从而最小化跨节点 token 传输次数。
   - 采用 bitmask 枚举解决最小集合覆盖问题，再在选定节点内均匀采样具体执行 rank。

---

### 🔍 相比现有方法的优势
| 方法 | 是否考虑共激活 | 是否兼顾通信与负载 | 是否动态调整 |
|------|------------------|--------------------|--------------|
| SmartMoE / LAER-MoE | ❌ | ⚠️ 仅关注负载 | ✅ |
| HierMoE / DeepEP | ⚠️ 节点去重 | ❌ 忽视负载 | ❌ |
| **COBALT** | ✅ | ✅ | ✅ |

- COBALT **同时优化通信与计算负载**，且适应训练过程中不断变化的共激活模式。
- 不修改模型结构或 gating 输出，完全兼容现有 MoE 架构。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用广泛用于 LLM 训练的 **C4 数据集**。
- 序列长度设为 **8192**，模拟真实训练场景。

### ⚙️ 实验设置
- **硬件平台**：4 台 NVIDIA DGX-B200 服务器，共 **32 张 B200 GPU**。
  - 节点内：NVLink，带宽高达 1.8 TB/s。
  - 跨节点：InfiniBand，聚合带宽 400 GB/s。
- **模型架构**：选取三种主流 MoE 模型
  - **Hunyuan3**（60B 总参，A5B 激活）
  - **GLM-4.5-Air**（106B 总参，A12B 激活）
  - **DeepSeek-V3**（140B 总参，A8B 激活）
- **EP Degree**：测试 EP=16 和 EP=32 两种配置。
- **实现基础**：基于 PyTorch + **DeepEP** 作为 all-to-all 通信后端。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Training Step Time** | 单步平均耗时，反映整体训练效率 |
| **Speedup** | 相对于基线的速度提升倍数 |
| **Cross-node Communication Volume** | 跨节点 token 传输总量（GiB） |
| **Workload Imbalance Ratio** | 最大 rank 负载 / 平均负载 |

### 🆚 基线方法对比
1. **EP+FSDP**：标准 PyTorch 实现，MoE 层用 EP，其余层用 FSDP。
2. **SmartMoE**：基于负载感知的专家复制与放置。
3. **LAER-MoE**：动态重布局以平衡负载。

所有方法均启用 DeepEP 的节点级 token 去重功能，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
| 模型 | EP Degree | COBALT Speedup (vs EP+FSDP) | Cross-node Traffic Reduction |
|-------|-----------|-------------------------------|------------------------------|
| Hunyuan3 | 32 | **2.41×** | **99.26%** |
| GLM-4.5-Air | 32 | **1.89×** | **95.3%** |
| DeepSeek-V3 | 32 | **1.53×** | **75.74%** |

> ✅ **平均加速 1.28–1.89×，最高达 2.41×**
>
> ✅ **跨节点通信量降低 75.74% – 99.26%**

---

### 🔁 与基线方法对比
| 对比项 | COBALT vs SmartMoE | COBALT vs LAER-MoE |
|--------|---------------------|---------------------|
| 速度提升 | 最高快 **1.52×** | 最高快 **1.28×** |
| 通信优化 | 显著更优（利用共激活） | 更优（LAER-MoE 忽视通信） |
| 负载均衡 | 更好 | 相当或更好 |

- 当 EP 规模增大时（如 EP=32），COBALT 优势更加明显，因为此时跨节点通信成为主导瓶颈。

---

### 🔍 消融实验结果（Ablation Studies）
在 GLM-4.5-Air（EP=16）上进行消融分析：

| 组件组合 | Speedup | Cross-node Comm ↓ | Imbalance Ratio |
|---------|--------|-------------------|-----------------|
| Baseline (EP+FSDP) | 1.00× | — | 3.35 |
| + Intra-node Rebalancing | 1.22× | — | **1.28** |
| + Periodic Global Update | 1.43× | **↓95.3%** | 1.13 |
| + Communication-aware TA | **1.53×** | **↓98.2%** | 1.13 |

> 结论：三个组件协同作用，其中：
> - **节点内重平衡** 主要改善负载；
> - **全局布局更新** 大幅削减通信；
> - **任务分配优化** 进一步压降通信开销。

---

### ⏱️ 开销分析（Overhead）
| 操作 | 时间开销（EP=16） | 是否可重叠 |
|------|------------------|------------|
| 全局布局规划（CPU） | ~1.21 秒 | ✅ 可与前向计算重叠 |
| 节点内重平衡 | ~0.25 秒 | ✅ 完全可重叠 |
| 任务分配（GPU kernel） | <1ms | ✅ 并行执行 |

> 整体开销可控，不影响端到端训练效率。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Expert Co-activation 是真实存在且可利用的现象**
   - 在 GLM-4.5-Air 训练中，专家对的 Ochiai 相似度从 0.41 上升至 0.89，表明共激活趋势显著增强。
2. **利用共激活可大幅减少跨节点通信**
   - 将高频共激活专家置于同节点，结合节点级去重机制，能极大压缩实际传输量。
3. **必须联合优化通信与负载**
   - 单纯聚集专家会导致负载倾斜；COBALT 的两阶段设计实现了二者权衡。
4. **无需改动模型即可获得显著收益**
   - COBALT 在不改变 gating、专家选择或参数的前提下，仅通过系统调度优化即实现加速。

---

### ⚠️ 局限性
1. **依赖 EMA 统计信息**
   - 专家布局基于历史统计数据（EMA），无法实时响应瞬时变化，可能导致次优决策。
2. **MILP 求解成本随规模增长**
   - 虽然当前规模下可接受，但在超大规模集群（如千卡以上）可能面临扩展性挑战。
3. **未整合其他并行策略**
   - 实验仅在纯 EP 设置下进行，尚未验证在 TP/DP/PP 混合并行下的表现。

---

### 🔮 未来工作方向
1. **在线自适应布局更新**
   - 探索轻量化在线学习机制，替代固定周期的全局规划。
2. **多层级并行协同优化**
   - 将 COBALT 思想扩展至与其他并行范式（如 DP、TP）结合的大规模训练系统。
3. **推理阶段的应用**
   - 将共激活模式用于 MoE 推理时的专家预取与请求调度（类似 Semantic Parallelism）。
4. **硬件感知布局生成**
   - 结合网络拓扑与带宽异构性，进一步精细化通信代价建模。

---

## 总结
> **COBALT 是首个利用 expert co-activation 来协同优化 MoE 分布式训练中通信与负载的系统框架。它通过两阶段专家布局规划与通信感知任务分配，在保持低开销的同时实现了高达 2.41× 的训练加速，并将跨节点通信减少近一个数量级。该工作揭示了细粒度路由模式在系统优化中的巨大潜力，为下一代高效 MoE 训练系统提供了新范式。**

</details>

---

### 4. [Joint Effects of GPU Server Topology, Parallelism, and Congestion Control on MoE Inference: A Controlled Simulation Study](https://arxiv.org/abs/2609.37828)

**Authors**: Kaikai Yuan, Rui Xi, Yu Liu  
**Category**: cs.DC  
**Published**: 2026-09-30  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.37828v1  

#### Abstract
Mixture-of-experts (MoE) models expand capacity via sparse activation, but inference across GPUs introduces tensor-parallel (TP) collectives and expert-parallel (EP) dispatch and combine operations. Completion time depends not just on communication volume but on how logical groups map onto intra-ser...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

**论文标题**: *Joint Effects of GPU Server Topology, Parallelism, and Congestion Control on MoE Inference: A Controlled Simulation Study*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
该论文系统地研究了在 **Mixture-of-Experts (MoE)** 模型推理过程中，多个系统层面因素——包括 **GPU服务器拓扑 (GPU server topology)**、**并行策略 (tensor and expert parallelism)**、**集体通信算法 (collective algorithms)** 和 **拥塞控制机制 (congestion control)** ——如何共同影响推理延迟。

现有研究通常孤立分析某一因素（如仅优化通信量或仅调整并行度），而忽略了这些因素之间的复杂交互作用。本文指出，在固定模型和计算负载下，**完成时间主要由“暴露的通信时间”（exposed communication time）决定**，而这一时间是上述所有因素耦合的结果，无法通过单一维度优化来准确预测性能。

### **提出的新方法或新思路**
- 构建了一个**受控的模拟矩阵 (controlled simulation matrix)**，在 **ASTRA-sim + NS-3** 联合仿真框架下，对四个 MoE 模型配置、六种 GPU 服务器拓扑、四种 TP/EP 分区策略、两种 TP 集体算法和四种网络/拥塞控制模式进行了组合实验，共执行了 **768 次确定性仿真**。
- 采用 **Chakra Execution Trace (Chakra ET)** 生成可复现的合成 workload，确保不同配置间的比较具有公平性和可控性。
- 明确区分了 **逻辑通信组 (logical communication groups)** 与 **物理传输路径 (physical transfer paths)**，强调了 **rank mapping** 在性能中的关键作用。

### **相比现有方法的优势**
- **系统性与全面性**：首次在一个统一的实验框架中量化了拓扑、并行度、集体算法和拥塞控制四者之间的联合效应，揭示了孤立优化可能失效的原因。
- **高保真仿真**：使用 NS-3 作为网络后端，能够精确建模链路、队列、RDMA 流和拥塞反馈，比 ASTRA-sim 的解析模型更贴近真实网络行为。
- **可解释性强**：通过消融实验和跨模型对比，识别出稳定的主效应（main effects）和条件依赖的交互效应（conditional interactions），为系统设计提供了明确指导。

---

## 2. 核心实验方法和设置

### **使用的数据集**
- **非传统数据集**：未使用真实数据集进行训练或推理。
- **合成 workload**：基于四个 MoE 模型配置（Kimi K3, GLM-5.3, DeepSeek V4 Pro 0813, DeepSeek V4 Flash 0731）生成 **4096-token 的 prefill-like 合成 Chakra ET 迹**。
- **模型参数来源**：官方技术报告和模型仓库（如 Hugging Face）。

### **实验设置**
- **总规模**：固定为 **32 GPU ranks**，数据并行 (DP) 和流水线并行 (PP) 固定为 1，仅变化 **TP × EP = 32**。
- **TP/EP 策略**：TP2EP16, TP4EP8, TP8EP4, TP16EP2。
- **服务器拓扑 (6种)**：
  - Topology 1–5：含外部网络（400 Gbps NIC），节点数从 32 到 1 不等。
  - Topology 6：单节点 32-GPU，无外部 NIC，仅内部 PCIe 互联。
- **集体算法组合**：
  - `ring_direct`：TP 使用 Ring AllReduce，EP 使用 Direct All-to-All。
  - `dbt_direct`：TP 使用 Double Binary Tree (DBT)，EP 使用 Direct。
- **网络/拥塞控制模式 (4种)**：
  - IB CCO（无速率反馈）
  - IB-like HPCC
  - RoCE HPCC
  - RoCE DCQCN
- **仿真工具链**：ASTRA-sim (workload & system layer) + NS-3 (network backend)。
- **验证**：通过与实际集群上的集体通信微基准测试对比，验证了仿真通信时间的准确性（相对误差仅 **4.9%**）。

### **评估指标**
- **主要指标**：
  - `Wall time`：分布式作业的总完成时间（瓶颈 rank 的时间）。
  - `GPU time`：瓶颈 rank 的计算时间。
  - `Comm time`：瓶颈 rank 的通信时间。
  - `Comm fraction`：通信时间占比。
- **辅助指标**：
  - `T_comm / T_wall`：通信占比。
  - `(B-A)/A`：相对开销。
  - `T_max / T_min`：配置敏感度。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**
- **通信主导性**：在所有配置中，**暴露的通信时间占平均完成时间的 89.9%–95.8%**，表明优化通信是提升性能的关键。
- **TP/EP 策略影响巨大**：
  - **TP16EP2** 的平均完成时间是 **TP2EP16** 的 **3.68–4.35 倍**。
  - 低 TP（TP2）高 EP（EP16）始终是最优策略。
- **集体算法差异显著**：
  - 在当前实现下，`dbt_direct` 比 `ring_direct` **慢 28.3%–83.2%**。
  - DBT 的理论低相位优势被其在树边上的大消息量和聚合压力抵消。
- **拥塞控制影响**：
  - **RoCE DCQCN** 比 **RoCE HPCC** **慢 23.8%–35.7%**。
  - **IB-like HPCC** 与 **RoCE HPCC** 性能接近，前者仅快约 **0.9%**。
- **拓扑表现依赖于并行度**：
  - **Topology 6**（单节点）在 **TP2EP16** 时最快，但在 **TP16EP2** 时因内部 PCIe 路径集中而成为最慢之一。
  - **Topology 1**（每 GPU 一个 NIC）和 **Topology 2** 表现稳定，得益于均衡的注入带宽。

### **与基线方法的对比结果**
- **vs. 仅考虑通信量的研究**：本文证明，即使通信总量相同，不同的物理映射（如是否跨越 UPI 或共享 NIC）也会导致高达数倍的性能差异。
- **vs. 仅优化并行度的研究**：本文显示，选择最优并行策略（如 TP2EP16）带来的收益远超切换拥塞控制协议（如 DCQCN → HPCC）。
- **vs. 仅关注硬件数量**：增加 NIC 数量（如 Topology 5）不一定提升性能，若 rank mapping 不均，仍会产生热点。

### **消融实验结果**
- **TP/EP 消融**：固定其他因素，仅改变 TP/EP，发现 **TP 增加带来多倍延迟增长**，且 EP 使用 Direct 可避免串行转发。
- **集体算法消融**：在 Kimi K3 上预实验显示，将 EP 从 Ring 改为 Direct 可使完成时间降低 **56.6%**，因此主实验固定 EP 为 Direct。
- **拓扑消融**：通过 Topology 2 vs. 3 对比，发现双 CPU/UPI 路径比单 PCIe 开关聚合慢约 **17%**，说明额外主机桥接层会增加同步成本。

---

## 4. 关键结论和发现

### **主要发现**
1. **MoE 推理性能是多因素耦合的结果**：不能仅凭 GPU 数、NIC 数或名义带宽预测性能；**逻辑通信组到物理路径的映射** 是决定性因素。
2. **TP/EP 并行度是最大影响因子**：**TP2EP16** 是所有模型下的最优起点，因其最小化了 TP AllReduce 的同步范围和序列化阶段。
3. **Ring AllReduce 优于 DBT**：在当前 ASTRA-sim 实现和固定 rank mapping 下，`ring_direct` 始终优于 `dbt_direct`，表明理论相位复杂度不等于实际性能。
4. **拓扑优势是条件性的**：**Topology 6** 的“无外部网络”优势仅在低 TP 时成立；高 TP 时，其内部 PCIe 路径成为瓶颈。
5. **拥塞控制效果依赖于拓扑**：DCQCN 的惩罚集中在有共享出口或 UPI 路径的拓扑上（如 Topology 3），而非均匀分布。
6. **模型排名稳定**：在所有 192 种系统配置下，**DeepSeek V4 Flash 0731** 始终最快，**GLM-5.3** 最慢，尽管后者参数更少。

### **方法的局限性**
- **仿真抽象**：使用合成 workload，未包含动态批处理、KV 缓存生命周期、运行时调度等生产级特性。
- **固定 rank mapping**：所有实验使用连续编号映射，结论对此映射方式敏感，不保证对任意映射通用。
- **网络简化**：外部网络为单交换机，未考虑 ToR/spine 层次或多路径路由。
- **集体实现特定**：DBT 的劣势源于其在 ASTRA-sim 中“全消息传输”的实现，不适用于所有 NCCL 实现。

### **未来工作方向**
- 将研究扩展到 **prefill/decode 分离部署** 和 **动态批处理** 场景。
- 探索 **自适应 rank mapping** 和 **拓扑感知的并行策略搜索**（如结合 TopoOpt、Alpa 等框架）。
- 引入 **真实运行时 trace** 以校准仿真，提高绝对延迟预测精度。
- 研究 **异构拓扑** 和 **多租户干扰** 下的 MoE 推理性能。

--- 

> **总结**：该论文通过大规模受控仿真，揭示了 MoE 推理性能的深层耦合机制，强调了“**减少暴露通信**”应从 **并行策略选择**、**集体算法实现** 和 **物理拓扑匹配** 三方面协同优化，而非孤立改进硬件指标。

</details>

---

### 5. [Mixture-of-Kittens: MoE Megakernel for NVL72s](https://arxiv.org/abs/2609.36070)

**Authors**: Stuart H. Sul, Nash Brown, Henry Wildermuth, William Lin, Federico Cassano, Christopher R\'e  
**Category**: cs.DC  
**Published**: 2026-09-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.36070v1  

#### Abstract
AI accelerator systems are rapidly consolidating into scale-up architectures, where tens to thousands of GPUs communicate over high-bandwidth, single-hop fabrics. We find that existing Mixture-of-Experts (MoE) training systems, optimized for conventional scale-out networks, transfer poorly to this s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Mixture-of-Kittens: MoE Megakernel for NVL72s*

## 1. 论文的主要贡献和创新点

### 解决的问题
随着AI加速器系统向**scale-up架构**演进（如NVIDIA NVL72），传统的Mixture-of-Experts (MoE) 分布式训练系统在这些高带宽、单跳互联的大规模GPU集群上表现不佳，甚至不如基于PyTorch和NCCL的朴素实现。现有方法针对scale-out网络设计，在scale-up环境下存在通信开销大、计算-通信重叠效率低、CPU-GPU同步瓶颈等问题。

### 提出的新方法与创新思路
作者提出 **Mixture-of-Kittens (MoK)** ——一个专为NVIDIA NVL72平台设计的MoE训练系统，其核心是一个融合了所有MoE操作的**确定性megakernel**。MoK基于三大关键洞察构建：

1. **按算子选择通信方向（Push vs. Pull）**  
   在scale-up架构中，pull-based通信与push同样高效。MoK为不同操作选择最优方向：
   - **Pull-based dispatch**：接收方主动拉取输入token，避免跨GPU完成信号同步。
   - **Push-based combine**：计算完成后主动推送输出。
   - 优势：将信号开销从最高18%降至<1%，调度表可复用，显著降低预调度开销。

2. **重构计算-通信重叠机制（Fine-grained Overlap with Tunable Granularity）**  
   引入**可调粒度的微批量（minibatch）** 来平衡细粒度与粗粒度重叠的优劣：
   - 太小 → 张量核利用率不足；
   - 太大 → 首次dispatch和最终combine无法被掩盖。
   - MoK通过**最大延迟同步（maximally deferred synchronization）** 和**kernel fusion**支持任意粒度下的高效重叠，最佳粒度在512–32,768 tokens之间。

3. **完全消除CPU-GPU同步（Fully Eliminate CPU-GPU Sync）**  
   利用**设备端环形缓冲区（device-side ring buffer / macrobatch）** 管理token流动，避免主机参与内存分配与管理。
   - 仅增加平均1.7%延迟（相比过分配方案）。
   - 结合**环感知前向回放（ring-aware forward replay）**，最小化反向传播时的中间激活重计算。

此外，MoK还具备以下生产级特性：
- 支持 **MXFP8** 精度，融合量化操作。
- 集成 **RDMA overlap via Cluster Launch Control (CLC)**，支持FSDP等跨机柜通信并行。
- 融合 **router weight gradient computation**，减少HBM流量。
- 支持**可调SM分区**（computation vs. communication SMs），适应不同负载。

### 相比现有方法的优势
- **性能更高**：在多种模型形状下达到最高2.37×的吞吐提升。
- **更少同步开销**：消除跨GPU completion signaling 和 CPU-GPU 同步。
- **更强扩展性**：特别适合大规模、每GPU token数较少的训练场景。
- **生产就绪**：已开源，并集成至NVIDIA NeMo AutoModel，支撑Composer等实际训练任务。

---

## 2. 核心实验方法和设置

### 使用的模型与配置
评估覆盖四个主流开源MoE模型的层结构：
- **Kimi K2.7 Code**: $E=384$, $D=7168$, $I=2048$, $k=8$
- **GLM-5.2**: $E=256$, $D=6144$, $I=2048$, $k=8$
- **Qwen3.5-397B-A17B**: $E=512$, $D=4096$, $I=1024$, $k=10$
- **DeepSeek-V4-Pro**: $E=384$, $D=7168$, $I=3072$, $k=6$

每个模型均含一个shared expert。

### 实验设置
- **硬件平台**：GB300 NVL72 racks，每rack 72块B300 GPU，通过第五代NVLink全互连。
- **软件环境**：CUDA 13.0, Python 3.13, PyTorch 2.13。
- **评估模式**：
  - 单MoE层前向/反向执行时间。
  - 生产级端到端训练吞吐量。
- **输入生成**：router logits 和 model inputs 从标准正态分布采样，各实现使用相同输入以保证公平比较。

### 评估指标
- **有效专家FFN吞吐量（TFLOP/s）**：
  - 前向：$ \frac{6N(k+1)DI}{t} $
  - 反向：两倍于前向FLOPs
- **端到端吞吐量**：tokens per second per GPU
- 所有测量取100次运行的中位数，以最慢rank为准。

### 基线方法对比
- `NCCL + PyTorch`：朴素实现
- `DeepEP + PyTorch`
- `DeepEP + TransformerEngine`
- `HybridEP + Megatron`

> 注：所有基线均为公开可用且支持NVL72的实现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（MoE层级别）
| 精度/阶段 | 最高加速比（vs. 最强baseline） |
|----------|-------------------------------|
| MXFP8 前向 | **2.37×** |
| MXFP8 反向 | **1.78×** |
| BF16 前向  | **1.92×** |
| BF16 反向  | **1.58×** |

> 图7显示，在所有测试配置下MoK均显著领先，尤其在MXFP8前向阶段增益最大。

### 端到端训练性能
| 系统 | Tokens/sec/GPU |
|------|----------------|
| DeepEP-based（原系统） | 761.0 |
| **MoK**（新系统） | **1,070.2** |

👉 **端到端吞吐提升 1.41×**

### 消融实验与敏感性分析（Table 1）
- **本地token数量越少，MoK增益越大**：
  - 如在`Kimi K2.7`, `EP=16`, `1024 tokens/GPU`下，前向提速达 **2.70×**
  - 因为通信与调度开销占比更高，MoK优化效果更明显。
- **EP degree影响**：
  - 在`EP=64`下仍保持显著优势（如最高2.37×），表明可扩展性强。
- **精度趋势**：
  - MXFP8增益 > BF16，因前者计算更密集，MoK对通信瓶颈缓解更有效。
  - 前向增益 > 反向，因反向计算量更大，更易成为瓶颈。

### 其他关键数据
- **调度开销降低**：MoK调度耗时仅为Comet的 **1.4–2.2× 更低**（Appendix A）。
- **环形缓冲额外延迟**：平均仅 **+1.7%**，远低于过分配带来的内存浪费。
- **信号开销控制**：从push-based的8–18%下降至<1%。

---

## 4. 关键结论和发现

### 主要发现
1. **Scale-up架构改变了MoE系统设计范式**：传统面向scale-out的设计不再适用，必须重新思考通信、同步与资源管理策略。
2. **Pull-based dispatch + Push-based combine 是最优组合**：可在scale-up环境中最小化同步开销，并实现调度复用。
3. **可调粒度是性能关键**：固定粒度无法适应多样化工况，MoK通过灵活配置实现吞吐最大化。
4. **设备端内存管理至关重要**：NVL72上CPU较弱，任何CPU-GPU同步都会严重拖累GPU利用率。
5. **Megakernel是理想载体**：融合dispatch、expert FFNs、combine于一体，消除kernel launch开销，保障稳定重叠。

### 方法的局限性
- **依赖Blackwell架构特性**：如Cluster Launch Control (CLC)，在旧代GPU上可能无法发挥全部潜力。
- **需预先知道macrobatch size**：虽然可调，但仍需根据workload手动或自动调优。
- **当前聚焦MoE层内部优化**：未涉及与其他并行策略（如TP、PP）的深度协同优化。

### 未来工作方向
- 自动化**granularity tuning**机制，结合runtime profiling动态调整。
- 将MoK思想推广至其他scale-up平台（如AMD Helios、Google TPU Pod）。
- 探索与**异构专家调度**、**动态路由**等高级MoE特性的集成。
- 进一步融合更多训练组件（如embedding lookup、attention）进入megakernel。

---

> ✅ **总结一句话**：  
> MoK通过**通信方向选择、可调粒度重叠、设备端环缓冲**三大创新，构建了一个专为NVL72 scale-up架构优化的高性能MoE megakernel，在真实生产环境中实现了高达 **2.37× 层级吞吐提升** 和 **1.41× 端到端训练加速**，已成为工业界MoE训练的新标杆。

</details>

---

### 6. [Reshaping Rollout Workloads for Asynchronous RL Post-Training on Heterogeneous Accelerators](https://arxiv.org/abs/2609.36899)

**Authors**: Jiahui Li, Hao Nie, Yibo Zhu, Pengjin Xie, Yu Zhou, Xiaolong Zheng, Liang Liu, Huadong Ma  
**Category**: cs.DC  
**Published**: 2026-09-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.36899v1  

#### Abstract
Reinforcement learning (RL) post-training increasingly relies on long-horizon, multi-turn rollouts. As post-training jobs outgrow a single cluster, rollout pools assembled across clusters introduce hardware heterogeneity. Rollout scheduling must serve two stakeholders: the hardware needs high aggreg...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Reshaping Rollout Workloads for Asynchronous RL Post-Training on Heterogeneous Accelerators*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在大规模强化学习（**RL post-training**）中，随着任务复杂度提升，模型需要进行长周期、多轮次的 **rollout**（轨迹生成）。这些 rollout 作业往往超出单个集群的算力容量，需跨多个异构集群调度。然而，这种 **heterogeneous rollout pool** 引入了硬件差异（如 GPU/NPU、内存带宽、成本等），导致以下矛盾：

- **硬件需求**：追求高聚合 **decode throughput**（解码吞吐量）
- **训练需求**：要求每个 trajectory 尽快完成，尤其是长上下文轨迹，以减少 **policy staleness**（策略陈旧性）

由于 autoregressive decoding 是 **memory-bandwidth-bound** 的，大 batch 虽能摊薄权重读取开销、提高吞吐，却会稀释每个 trajectory 的带宽份额，延长其完成时间。传统调度方法难以同时满足这两个目标。

---

### **提出了什么新方法或新思路**

作者提出 **CadenceRL**，一种面向异构 rollout 池的动态调度系统，其核心思想是 **specialization（专业化分工）**，而非在每个 worker 上妥协。

#### 创新机制：

1. **Pacing（节奏控制）**
   - 在仍有新鲜短上下文任务时，执行“**长出短进**”（long-out, short-in）操作：
     - 从 worker 中驱逐长上下文 trajectory
     - 替换为一个或多个短上下文 trajectory
   - 该操作结构性地维持高 throughput 区域的工作负载状态，延缓 throughput 下降。
   - **无需预测迁移收益**，避免了因依赖未来状态而产生的循环依赖问题。

2. **Concentration（集中处理）**
   - 当新鲜短任务耗尽后，进入尾部集中阶段：
     - 将剩余的长上下文轨迹集中到对长上下文有高亲和性的 worker（如高带宽设备）
     - 释放低带宽设备用于下一轮 policy 的 fresh work
   - 实现尾部加速 drain，降低 P95 latency。

3. **Late-Bound KV Preparation（延迟绑定 KV 准备）**
   - trajectory 离开源 worker 后，先将 KV cache 快照至 host DRAM，并通过 RDMA 预加载到 relay 节点
   - 目标 worker 确定后再并行拉取，避免重复 prefill
   - 支持 **deferred destination binding**，增强调度灵活性

4. **KV Headroom 控制结构**
   - 所有调度决策基于 **KV headroom**（可用 KV 缓存空间）
   - 不依赖硬件标签、角色分配或 workload 分类器，实现轻量级、通用控制

---

### **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **调度逻辑** | 避免 per-migration benefit estimation，解决循环依赖问题 |
| **资源利用** | 实现 throughput worker 与 tail worker 的自然分化，最大化硬件互补性 |
| **适应性** | 动态响应 workload 演化，无需静态配置 |
| **可扩展性** | 支持异构硬件组合，添加不同设备可分别提升 throughput 或降低 tail latency |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- 多任务混合训练集，涵盖：
  - 数学推理（mathematics reasoning）[2,10,46]
  - 竞赛编程（competitive programming）[17,18]
  - STEM 推理 [4,28,38]
  - 逻辑谜题（logical puzzles）[23,34,40]
- 包含单轮生成与多轮环境交互任务

---

### **模型与硬件平台**

#### **模型**
- **Qwen3-8B**, **Qwen3-14B**, **Qwen3-32B**
- 使用 full-attention 架构，对 KV-cache 迁移构成压力测试

#### **硬件配置（Table 1）**
| 设备 | 类型 | HBM 带宽 (GB/s) | 成本 ($/h) |
|------|------|------------------|------------|
| **H800** | GPU | 3,350 | 2.00 |
| **A910X** | NPU | 1,935 | 0.67 |
| **BI-V150** | GPU | 800 | 0.22 |

- 每种设备位于独立集群内
- 集群间通过 20 Gbps 专用链路连接
- 训练始终运行于 H800

---

### **评估指标**

| 指标 | 定义 |
|------|------|
| **Aggregate decode throughput** | 总 token/s，衡量系统吞吐能力 |
| **Trajectory latency (P50/P95)** | 从首次 inference 到完成的时间（含 pending 时间） |
| **Tokens per dollar** | 成本效益指标 |
| **Non-resident fraction** | 长轨迹处于 pending 状态的比例 |

---

### **基线方法对比**

| 基线 | 简介 |
|------|------|
| **Sync** | 同步 rollout 与 training，无 staleness |
| **One-off** | 单步 off-policy 重叠 |
| **DORA*** | 工作负载感知的资源再分配（本文实现简化版） |
| **Laminar** | 轨迹级异步 + repack |
| **AReaL** | partial rollout + 权重切换恢复 |

> 注：所有方法共享相同底层框架（vLLM + Steptron），仅调度策略不同。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **异构池性能（图12）**
| 配置 | 方法 | Throughput (↑) | P95 Latency (↓) | Tokens/$ (↑) |
|------|------|----------------|------------------|--------------|
| 16A16B | CadenceRL | **最高** | **最低** | **最高** |
| 16A8H / 16B8H | CadenceRL | 显著优于基线 | P95 ↓ up to **64%** | — |
| 16A16B8H | CadenceRL | ↑ up to **48%** | P95 ↓ up to **64%** | — |

> ✅ **结论**：CadenceRL 在所有异构配置下均优于最强基线（Laminar/AReaL）

---

#### **消融与敏感性分析**

##### （1）**Staleness 敏感性（图15）**
- 增加 staleness budget（`η`）可提升 throughput（更多并发）
- 但 P95 latency 随之单调上升
- CadenceRL 可通过调节 `η` 控制 throughput-latency 权衡

##### （2）**Decode Reserve (`δ`) 敏感性（图16）**
- `δ` 过大会限制调度自由度，削弱 tail-latency 抑制能力
- 实验表明 `δ=512~1024` 为较优选择

##### （3）**KV Transfer 开销（图18）**
- P50 传输延迟随 relay 节点增加快速收敛
- 准备机制带来轻微吞吐影响（±几个百分点），但总体可控

---

#### **长尾轨迹行为分析（图14）**
- **CadenceRL**：轨迹虽仅 **<50% 时间 resident**，但集中在高 KV share 窗口解码，单位时间进展更快
- **Laminar/AReaL**：持续低 share 解码，进度缓慢
- 结果：**CadenceRL 的长尾轨迹最先完成**

---

#### **非驻留时间统计（表3）**
| Pool | P95 Non-resident Fraction |
|------|----------------------------|
| 32A | 39.91% |
| 32B | 80.94% |
| 16A16B8H | **40.31%** |

✅ 表明高 throughput 配置能更快释放资源，减少 pending 时间

---

## 4. 关键结论和发现

### **主要发现**

1. **High throughput 与 low tail latency 并不冲突**  
   二者需要不同的 workload composition，可通过 **specialization** 实现共存。

2. **Decode-throughput landscape 具有单调性**  
   “长出短进” 操作结构性保证正向增益，使调度可绕过 per-move benefit estimation。

3. **异构硬件不是负担而是机会**  
   - 高带宽设备（H800）天然适合作为 **tail worker**
   - 低成本设备（A910X/BI-V150）适合维持大 batch throughput
   - 两者互补形成 **synergy**，打破 throughput-latency trade-off

4. **Scaling 是可组合的（composable）**  
   - 添加 high-bandwidth accelerators → 降低 tail latency
   - 添加 cost-efficient accelerators → 提升 throughput
   - **无需手动路由配置**

5. **Pending Set 是关键协调机制**  
   - 在 pacing 阶段作为 transit station
   - 在 concentration 阶段作为 pace governor（通过 `P_v = ∅` 控制 drain 速率）

---

### **方法的局限性**

| 局限性 | 说明 |
|--------|------|
| **依赖 KV cache 可迁移性** | 对 MoE 或稀疏注意力架构支持有限 |
| **跨集群通信开销** | 虽优化 KV transfer，但在低带宽网络下仍可能成为瓶颈 |
| **未考虑环境响应不确定性** | 多轮交互中的 env latency 仍具挑战 |
| **当前实现基于 credit model** | 更复杂的 admission control 可进一步优化 |

---

### **未来工作方向**

1. **扩展至 MoE/Mixture-of-Experts 架构**
2. **结合 speculative execution 加速长尾**
3. **自适应 tuning of `δ` 和 `η`**
4. **集成 Heddle-style 进度预测以辅助 concentration 决策**
5. **探索更细粒度的 chunk-level workload shaping**

---

> **总结一句话**：  
> CadenceRL 通过 **structural workload reshaping** 而非 **per-move optimization**，实现了在异构 rollout 池上的高效 specialization，显著提升了 throughput 与 tail latency 的双重表现，且具备良好的可扩展性与自动化能力。

</details>

---

### 7. [HyperZip: Efficient Data Compression through Personalized Diffusion LLMs with Hypernetworks](https://arxiv.org/abs/2609.36357)

**Authors**: Thai Nguyen, Khang Tran, NhatHai Phan  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.36357v1  

#### Abstract
Large language models (LLMs) have shown strong potential for lossless data compression, but existing approaches are constrained by the high computational cost and low throughput of autoregressive decoding. We propose HyperZip, an efficient and scalable LLM-based compression framework that leverages ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*HYPERZIP: Efficient Data Compression through Personalized Diffusion LLMs with Hypernetworks*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的基于 **Large Language Models (LLMs)** 的无损数据压缩方法虽然在压缩率上优于传统算法（如 gzip），但受限于以下两个关键瓶颈：
1. **低吞吐量**：主流 LLMs 采用自回归（autoregressive）解码机制，每次只能预测一个 token，导致压缩/解压速度极慢。
2. **高资源消耗**：为获得低 perplexity 和高预测置信度，通常需要对 LLM 进行 fine-tuning 或使用超大规模模型，计算成本高昂。

这使得当前 LLM-based 压缩难以在实际场景中部署，尤其是在资源受限的设备上。

---

### 🚀 提出的新方法与创新思路
作者提出 **HYPERZIP** —— 一种高效、可扩展的个性化 LLM 压缩框架，其核心创新包括：

#### （1）引入 **diffusion-based LLMs (dLLMs)** + **Multi-Token Prediction (MTP)**
- 替代传统的 autoregressive LLM，利用 dLLMs 在每轮迭代中并行“去噪”多个 masked tokens。
- 显著提升压缩过程中的 **throughput（吞吐量）**，实现更快的编码/解码。

#### （2）揭示并缓解 **compression rate 与 throughput 的权衡关系**
- 发现 dLLMs 存在固有 trade-off：降低 confidence threshold γ 可提高 throughput，但会增加压缩率（即压缩效果变差）。
- 为此，HYPERZIP 引入 **hypernetwork** 架构来生成数据特定的 LoRA 更新（△θ），从而动态适配 dLLM 到目标文件内容。

#### （3）无需 fine-tuning 的个性化建模
- 使用一个轻量级 hypernetwork，从输入文本的上下文向量 $ e = \text{emb}(X) $ 生成 LoRA 参数更新。
- 在推理阶段，将该 context vector 与压缩比特流一起存储，在解压时重建个性化模型。
- 避免了昂贵的 per-file fine-tuning，同时显著提升了模型在特定数据上的预测置信度。

---

### 🔍 相比现有方法的优势
| 维度 | HYPERZIP | 传统方法 |
|------|---------|--------|
| **压缩速度** | 高（支持 MTP 并行解码） | 低（自回归逐 token 解码） |
| **压缩率** | 低（通过个性化适应优化） | 一般或需 fine-tuning 才能改善 |
| **资源效率** | 高效（仅传输小 context vector） | 低效（若传 LoRA 权重则开销大） |
| **泛化能力** | 跨主题表现稳定 | 泛化性弱，依赖训练分布 |

> 💡 **核心优势**：实现了 **压缩率与速度之间的最优权衡（best trade-off）**，推动 LLM-based 压缩走向实用化。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **FineWeb-Edu**：高质量教育类网页文本数据集，用于训练和主评估。
- **enwik9**：经典基准数据集（维基百科前 1GB 内容），用于跨域测试泛化能力。

> ⚠️ 注意：enwik9 文本未参与 hypernetwork 的训练，确保公平比较。

---

### ⚙️ 实验设置
#### 模型架构
- **Backbone**：基于 `SmolLM2` 蒸馏得到的 diffusion LLMs，规模涵盖：
  - 150M、0.5B、1.5B 参数版本（如 Fast-dLLM-v2）
- **Hypernetwork 结构**：
  - 文档编码器：`GTE-Qwen2-1.5B-Instruct` 提取 context vector $ e $
  - 多层感知机（MLP）生成 LoRA 参数（rank=8）
  - 支持模块/层嵌入以区分不同位置的适配参数

#### 推理配置
- **Block size**: 256
- **Confidence threshold γ**: 默认设为 0.5（验证集调优所得）
- **Arithmetic coding**：使用 dLLM 输出的条件概率进行精确编码

---

### 📊 评估指标
| 指标 | 定义 | 目标 |
|------|-----|------|
| **Compression Ratio (CR)** | $ CR = 100 \times \frac{S_{\text{payload}}}{S_{\text{raw}}} \% $ | 越低越好 |
| **Throughput** | tokens per second (tok/s) | 越高越好 |
| **Losslessness** | 解压后是否完全还原原始字节序列 | 必须满足 |

> 注：CR 包含 context vector 的大小，体现端到端压缩效率。

---

### 🆚 基线方法对比
| 类别 | 方法 |
|------|------|
| **AR-based** | 自回归模型、L3TC、L-MTP、EAGLE-3 |
| **Non-AR-based** | DFlash、Nemotron-Labs-Diffusion、Fast-dLLM-v2 |
| **Fine-tuning 对比** | 在目标数据上监督微调 dLLM |

---

## 3. 主要实验结果和性能指标

### 📈 总体性能对比（图3）
在 FineWeb-Edu 和 enwik9 上，HYPERZIP 在 **所有模型尺度下均取得最佳 trade-off**：
- 在相同 throughput 下，**压缩率更低**
- 在相同压缩率下，**吞吐量更高**

> ✅ 图中右下角区域（低 CR + 高 throughput）为理想区，HYPERZIP 最接近该区域。

---

### 📊 表格结果（Table 1 & 5）——跨主题压缩表现

| Model | Scientific | Literature | Programming | General Web | Throughput (tok/s) |
|-------|------------|------------|-------------|--------------|---------------------|
| Fast-dLLM | 16.1% | 15.7% | 18.0% | 16.3% | 141.0 |
| Fine-tuning | 15.9% | 15.4% | 17.7% | 16.1% | 140.2 |
| **HYPERZIP (Ours)** | **15.6%** | **15.3%** | **16.3%** | **15.8%** | **138.9** |

> 🔍 观察：
- 在编程文本上，**压缩率从 18.0% 降至 16.3%**，提升明显；
- 吞吐量仅轻微下降（141 → 138.9 tok/s），说明个性化代价极小；
- 在所有主题上均优于 vanilla 和 fine-tuned 版本。

---

### 🔬 消融实验与分析（Appendix C.3）
- **Context vector 有效性**：去除 context vector 后性能退化至接近 vanilla dLLM，证明其关键作用。
- **LoRA rank 影响**：r=8 已足够有效，进一步增大收益有限。
- **Block size 影响**：更大的 block size（如 256 vs 32）显著提升 throughput。
- **γ 阈值敏感性**：HYPERZIP 在不同 γ 下保持更稳定的 CR-througphut 曲线，鲁棒性强。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **dLLMs 具备加速压缩的巨大潜力**，但存在 compression rate 与 throughput 的内在 trade-off。
2. **Hypernetwork + LoRA 是实现快速个性化建模的有效手段**，可在不 fine-tuning 的前提下大幅提升模型对特定数据的拟合能力。
3. **context vector 可安全压缩并与 bitstream 联合传输**，总开销远小于完整 LoRA 权重，适合实际部署。
4. **HYPERZIP 具备强泛化性**：在法律、科学、编程、文学等多种文本类型上均表现优异。

---

### ⚠️ 局限性（论文未明确列出，但可推断）
- **依赖预训练 dLLM 质量**：性能上限受 backbone 模型能力制约。
- **context vector 截断问题**：对于超长文档（>32k tokens），截断可能损失语义信息。
- **缺乏 error bar 报告**：实验结果未提供统计显著性检验（见 CheckList Q7 回答为 No）。
- **未开源代码**：目前无法复现实验（CheckList Q5 回答为 TODO）。

---

### 🔮 未来工作方向
1. **支持更长上下文的 context 编码机制**（如 chunk-wise aggregation）
2. **探索更高效的 hypernetwork 架构**（如稀疏化、量化）
3. **扩展至多模态压缩任务**（参考 OmniZip）
4. **研究压缩-重构延迟联合优化策略**
5. **探索联邦式个性化压缩**：用户本地生成 context vector，保护隐私

---

## ✅ 总结一句话
> **HYPERZIP 通过结合 diffusion LLMs 的高速 MTP 解码能力和 hypernetwork 实现的数据级个性化适配，在无需 fine-tuning 的前提下实现了当前最优的压缩率-速度权衡，是迈向实用化 LLM-based 压缩的重要一步。**

</details>

---

### 8. [IronLLM: Forging Compact Edge-Native Language Models for Real-Time Embodied Intelligence](https://arxiv.org/abs/2609.36860)

**Authors**: Changdi Yang, Fengquan Jiao, Haochih Lin, Haoran Yang, Jing Xiao, Liangyu Huo, Suxin Lu, Tiance Chen, Wei Liu, Yinggan Xu, Yunxiang Lu, Zai Zheng, Zhirui Xie, Zhongyang Che, Ziyan Tang, Zuoxiang Zhao, Jian Yao  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.36860v1  

#### Abstract
We present IronLLM-0.6B, a 654M-parameter language model designed for efficient on-device inference. IronLLM-0.6B combines a hybrid attention architecture with X-MTP, a lightweight shared-KV multi-token prediction design that eliminates per-depth KV-cache replay and employs a lightweight verificatio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：IronLLM: Forging Compact Edge-Native Language Models for Real-Time Embodied Intelligence

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**资源受限的边缘设备**（如机器人、车载系统）上的实时语言模型部署需求，解决了以下关键挑战：
- **高延迟**：传统大模型在边缘设备上推理速度慢，难以满足实时交互需求。
- **内存与能耗限制**：边缘设备计算资源有限，无法承载大规模模型的KV Cache和计算开销。
- **小模型能力不足**：小型语言模型通常性能弱于大模型，且对训练数据质量更敏感。
- **多领域能力冲突**：后训练中集成多个专家能力时易出现**跨域干扰**（cross-domain interference）和**灾难性遗忘**。

### 提出的新方法与创新点

#### （1）**X-MTP：轻量级共享KV多令牌预测架构**
- 设计了一种**共享KV的Multi-Token Prediction (MTP)** 架构，消除了传统MTP中每层重复构建KV缓存的开销。
- 引入**轻量验证头**（Lightweight Verification Head），实现无需回滚的自适应起草（rollback-free drafting），特别适用于混合线性注意力结构（如GDN）。
- 在数学和代码任务上实现了 **1.48× 的解码加速**。

#### （2）**IronLLM-0.6B-Light：面向边缘优化的极致高效变体**
- 首次提出**无RMSNorm的LLM架构**，采用 **Dynamic Tanh (DyT)** 替代RMSNorm，降低计算复杂度并提升量化友好性。
- 简化激活函数为**可学习上界ReLUx**，移除注意力输出门控和QK-Norm等冗余模块。
- 显著提升推理效率与量化兼容性，同时保持大部分性能。

#### （3）**高质量、高效率的数据处理框架**
- 构建统一的数据飞轮（Data Flywheel），通过**系统性质量增强**、**数据组成优化**和**数据-模型协同优化循环**，仅用 **6.2万亿token** 即达到媲美更大规模训练的效果。
- 数据策略强调中文-英文双语平衡，在中文理解任务中表现优异。

#### （4）**Multi-Domain On-Policy Distillation (MOPD) 后训练范式**
- 采用“先分域训练专家 → 再通过策略蒸馏融合”的两阶段方案。
- 利用冻结的领域专家作为教师，结合**可验证奖励**（Verifiable Rewards, VRs）进行在线策略蒸馏，避免参数合并带来的负迁移。
- 成功整合数学、代码、指令遵循等多种能力而无显著退化。

---

## 2. 核心实验方法和设置

### 使用的数据集

| 类别 | 主要数据集 |
|------|----------|
| **预训练数据** | 超过20万亿token的原始池，精选出6.2T高质量token，涵盖：<br>- 通用网页、书籍、学术论文<br>- 数学、代码、推理数据<br>- 高质量合成数据（如QA对）<br>- 中文与英文双语内容 |
| **评估基准** | - **通用知识**：MMLU, MMLU-Pro, MMLU-Redux, CMMLU, C-Eval<br>- **数学推理**：GSM8K, MATH-500, AIME 2025/2026, HMMT<br>- **代码生成**：HumanEval, MBPP, LiveCodeBench v6<br>- **指令遵循**：IFEval, IFBench, Multi-IF<br>- **主观质量**：AlpacaEval 2.0, ArenaHard<br>- **长上下文**：RULER, LongBench v2<br>- **函数调用**：BFCL v3 |

### 实验设置与评估指标

- **模型规模**：IronLLM-0.6B（654M参数），对比模型包括 Qwen3-0.6B、Qwen3.5-0.8B、MiniCPM5-1B、LFM2-700M。
- **硬件平台**：RTX 4090，使用 vLLM 推理引擎，batch size=1。
- **评估协议**：
  - 所有模型使用官方推荐采样参数（temperature, top_p等）。
  - 采用 **0-shot** 设置，统一 Prompt 模板。
  - 使用 **EvalScope v1.10.0** 进行标准化评测。
- **关键指标**：
  - 准确率（Accuracy）、Pass@k、Avg@k
  - 解码吞吐量（Tokens Per Second, TPS）
  - 平均输出长度（Average Output Token Length）
  - **Score Efficiency**：综合考虑准确率、TPS 和输出长度的综合效率指标

### 基线方法对比
| 模型 | 参数量 | 是否Thinking模式 | 特点 |
|------|--------|------------------|------|
| Qwen3-0.6B | ~600M | 是 | 支持思维链，输出较长 |
| Qwen3.5-0.8B | ~800M | 是 | 性能较强但推理成本高 |
| MiniCPM5-1B | ~1B | 是 | 更大模型，部分任务领先 |
| LFM2-700M | ~700M | 否 | 小模型基线 |

> 注：所有对比均以“非思考模式”（non-thinking）为主，确保公平比较推理效率。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | IronLLM-0.6B | 最优基线 | 表现 |
|------|--------------|-----------|-------|
| **GSM8K** | **78.6** | Qwen3.5-0.8B: 67.6 | ✅ 显著领先 |
| **MATH-500** | **67.4** | Qwen3.5-0.8B: 56.4 | ✅ 领先 |
| **HumanEval** | **46.3** | Qwen3.5-0.8B: 68.9 | ⚠️ 略低（但输出更短） |
| **IFEval**（指令遵循） | **75.6** | Qwen3.5-0.8B: 70.4 | ✅ 领先 |
| **AlpacaEval 2.0**（主观质量） | **10.2** | Qwen3.5-0.8B: 4.0 | ✅ 大幅领先 |
| **BFCL v3**（函数调用） | **49.4** | MiniCPM5-1B: 49.6 | ≈ 接近最优 |

> 💡 **IronLLM-0.6B 在多数任务上优于或媲美更大的 Qwen3.5-0.8B 和 MiniCPM5-1B**，尤其在指令遵循和主观质量方面优势明显。

### 与基线方法的对比结果

- **相比 Qwen3.5-0.8B（多出约23%参数）**：
  - 在 **13项任务中胜出或持平**，尤其是在 IFEval、GSM8K、MMLU-Pro 上大幅领先。
  - 输出更**简洁**，平均输出长度减少30%-50%，显著降低延迟。
  - 在 **32K上下文长度下，解码速度快1.48倍**（见Table 6）。

- **相比 MiniCPM5-1B（多出53%参数）**：
  - 在大多数任务上仍具竞争力，甚至反超，体现极强的**参数效率**。

- **推理效率优势**：
  - **X-MTP** 在 GSM8K 上带来 **1.48× 解码加速**（从460 → 678 tokens/s）。
  - **IronLLM-0.6B-Light** 在保持高性能的同时进一步提升推理速度。

### 消融实验结果

| 实验 | 发现 |
|------|------|
| **MOPD vs TIES 参数合并** | MOPD 在所有13个任务上均优于TIES，无负迁移现象，证明其在能力融合上的优越性（Table 8） |
| **验证头有效性测试** | 在 MMLU-Redux 和 IFEval 上几乎无损（+0.08, -2.77），但在 GSM8K 上下降12.36，说明复杂推理任务对早期错误敏感 |
| **不同MTP深度对比**（K=1,2,3） | K=3时速度提升最大（1.48×），接受率分别为95%, 89%, 82%，表明共享KV设计在深层依然有效（Table 10） |
| **数据污染分析** | 多数任务去污染后性能变化 < ±2.5点，证明结果可靠（Appendix A） |

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **紧凑模型也能具备强大能力**：IronLLM-0.6B 通过高质量数据、先进架构和MOPD后训练，在多项任务上超越更大模型，打破“越大越好”的固有认知。
2. ✅ **效率与性能可以兼得**：通过 X-MTP 和 Hybrid Attention（GDN+GA），在不牺牲性能的前提下大幅提升推理速度。
3. ✅ **Instruct-Only 设计是边缘场景的关键**：去除“thinking mode”使响应更简洁，显著提升 **Score Efficiency**，更适合实时应用。
4. ✅ **MOPD 是有效的多能力融合机制**：相比参数合并，MOPD 能更好地保留各领域专家能力，避免负迁移。
5. ✅ **数据质量比数量更重要**：仅用6.2T token即达到顶尖水平，得益于精细化的数据筛选与混合策略。

### 方法的局限性

- ❗ **数学与代码任务仍有差距**：尽管整体领先，但在 AIME、HMMT 等高难度数学竞赛题上仍落后于最强闭源模型。
- ❗ **验证头在复杂任务中可能引入误差**：对于需要严格推理链条的任务（如数学证明），轻量验证头可能导致累积错误。
- ❗ **硬件依赖性强**：当前优化基于特定GPU（RTX 4090）和推理框架（vLLM），在其他平台上的收益需重新验证。
- ❗ **未支持多模态**：目前仅为纯语言模型，尚未扩展到视觉或其他模态输入。

### 未来工作方向

1. 🔮 **向更高效的边缘原生架构演进**：探索更低延迟、更高吞吐的新型注意力机制与归一化方式。
2. 🔮 **扩展至紧凑多模态模型**：将 IronLLM 的设计理念应用于 VLM 或具身智能代理。
3. 🔮 **提升数学与代码专项能力**：引入更强的形式化验证机制或符号推理模块。
4. 🔮 **加强安全与鲁棒性**：在资源受限条件下实现可控生成、偏见缓解与对抗防御。
5. 🔮 **部署到真实具身系统**：在机器人、智能座舱等实际场景中验证端到端性能与可靠性。

---

> 📌 **总结一句话**：  
> **IronLLM 展示了如何通过“高质量数据 + 高效架构 + 精细后训练”三位一体的设计，在极小参数量下实现高性能与高效率的完美平衡，为边缘端实时语言智能提供了新的范式。**

</details>

---

### 9. [SEED: Self-Speculative Decoding via Implicit Encoder-Decoder](https://arxiv.org/abs/2609.36590)

**Authors**: Hankun Lin, Patrick Pynadath, Ruqi Zhang  
**Category**: cs.CL  
**Published**: 2026-09-30  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.36590v1  

#### Abstract
Self-speculative decoding accelerates large language model (LLM) inference by drafting tokens from the target model itself, but faces a sharp tradeoff between the quality and cost of the draft. Early-exit methods produce drafts cheaply by terminating computation at intermediate layers, but forgo the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SEED: Self-Speculative Decoding via Implicit Encoder-Decoder**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLM）推理面临**内存带宽瓶颈**，生成每个 token 都需要一次完整的前向传播，导致 GPU 计算资源利用率低。  
自推测解码（self-speculative decoding）通过在同一个模型内进行起草（drafting）和验证（verification）来加速推理，但存在**质量与成本之间的权衡**：
- **Early-exit 方法**（如 LayerSkip）：在中间层提前退出，计算便宜但依赖浅层表示，草案质量差。
- **MTP 方法**（如 Apple MTP）：从最终隐藏状态生成多个 token，保持高质量但每次起草都需要完整前向传播，成本高。

### **提出了什么新方法或新思路**
提出 **SEED (Self-Speculative Encoder-Decoder)**，其核心思想是将标准的 decoder-only Transformer 重新解释为一个**隐式的 encoder-decoder 架构**：
- **Encoder（前若干层）**：构建深层上下文表示（deep contextual representations）。
- **Decoder（最后几层）**：基于这些表示自回归地生成 token。

**关键洞察**：验证步骤已经完成了 encoder 的工作——它生成了已验证前缀的深层 KV 缓存（KV cache）。SEED 利用这一点，在两次验证之间复用这些缓存，仅用轻量级的 decoder 进行快速起草。

### **相比现有方法的优势**
- ✅ **高质量**：草案条件于完整的深层上下文表示（而非浅层或原始 token）。
- ✅ **低成本**：起草阶段只需轻量级 decoder 的前向传播（如 2 层），无需重复运行整个 encoder。
- ✅ **简单高效**：训练时仅需在标准 SFT 上增加一个辅助损失（`Lspec`），不改变模型架构。
- ✅ **参数共享**：drafter 和 verifier 共享参数，保证分布对齐，提升接受率。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
在四个代表性任务上进行实验：
- **GSM8K**：数学推理（0-shot pass@1 accuracy）
- **KodCode**：代码生成（exact problem-solving accuracy）
- **ScienceQA**：科学问答（multiple-choice accuracy）
- **CNN/Daily Mail**：摘要生成（ROUGE-1, ROUGE-2, ROUGE-L）

### **实验设置和评估指标**
- **模型**：Qwen3-1.7B-Base 和 Qwen3-4B-Base
- **训练方式**：在任务特定数据上进行 supervised fine-tuning（SFT），SEED 添加 `Lspec` 损失
- **Decoder 设计**：使用模型最后 2 层作为 decoder
- **Block Size**：训练时划分为 10-token 块以模拟起草场景
- **评估指标**：
  - **Decoding Throughput**（tokens/s）：主要效率指标
  - **Average Speedup**：相对于 AR 的加速比
  - **Task Accuracy**：保留或提升生成质量

### **基线方法对比**
| 类别 | 方法 |
|------|------|
| **通用加速** | AR（标准自回归） |
| **MTP / 扩散类** | Apple MTP, E2D2 |
| **早期退出类** | LayerSkip, SWIFT, DEL |
| **外部草稿模型** | EAGLE-3, PARD |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
| 方法 | Qwen3-1.7B 平均加速比 | Qwen3-4B 平均加速比 | 吞吐量（tokens/s）示例 |
|------|------------------------|------------------------|--------------------------|
| **AR** | 1.0× | 1.0× | ~37.9 (GSM8K) |
| **EAGLE-3** | 2.0× | 2.1× | ~79.4 |
| **Apple MTP** | 1.4× | 1.6× | ~59.1 |
| **LayerSkip** | 1.4× | 1.4× | ~47.0 |
| **SEED (Ours)** | **2.6×** | **2.7×** | **~91.8** |

> 在 Qwen3-4B 上，SEED 比当前最优的 EAGLE-3 快 **28%**。

### **与基线方法的对比结果**
- 🚀 **显著优于所有 self-speculative 基线**：
  - 比 **LayerSkip** 快约 **2.0×**（throughput 91.8 vs 47.0）
  - 比 **Apple MTP** 快约 **1.6×**（throughput 91.8 vs 59.1）
- ✅ **质量不降反升**：
  - 在 GSM8K 上，SEED 准确率 **57.1%**，高于 AR 的 56.3%
  - 而 E2D2 和 LayerSkip 等方法出现明显质量下降（如 E2D2 降至 26.6%）
- 🔁 **验证与编码合并**：一次完整 forward pass 同时完成验证和更新深层缓存，极大提升效率。

### **消融实验结果**
#### **(1) 解码器层数影响（Decoder Size）**
| Decoder Layers | Throughput (tokens/s) | Acceptance Rate (%) |
|----------------|------------------------|----------------------|
| 2 | **91.8** | 88.6% |
| 4 | 87.6 | 89.1% |
| 6 | 85.0 | 89.5% |

➡️ 更大的 decoder 提升草案质量但降低吞吐量，**2 层取得最佳平衡**。

#### **(2) 训练块大小（Block Size）**
- 增大 block size（如从 4 到 10）能更好模拟长序列起草，**提升接受长度和吞吐量**。
- 最终选择 **block size = 10** 达到最高速度。

#### **(3) 损失权重 λ（λ in `L = L_CE + λ * L_spec`)**
| λ | Accuracy (%) | Throughput (tokens/s) | Acceptance Length |
|----|---------------|------------------------|--------------------|
| 0.1 | 58.5 | 81.5 | 3.0 |
| 1.0 | **57.1** | **91.8** | **4.0** |
| 2.0 | 55.1 | 91.4 | 4.0 |

➡️ **λ = 1.0** 在质量和速度间达到最佳权衡。

#### **(4) 参数共享消融**
移除参数共享后：
- 准确率从 **57.5% → 44.1%**
- 吞吐量从 **74.3 → 68.8 tokens/s**
➡️ 证明 **参数共享对对齐 drafter 与 verifier 分布至关重要**。

---

## 4. **关键结论和发现**

### **主要发现**
1. **隐式 encoder-decoder 结构有效**：将 decoder-only 模型视为 encoder-decoder 可自然分离“理解”与“生成”，实现高效复用。
2. **验证即编码**：验证过程天然完成了下一轮起草所需的上下文编码，无需额外计算。
3. **轻量 decoder + 深层缓存 = 高效高质量起草**：仅用最后 2 层即可生成高质量草案，因条件于完整上下文。
4. **提升下游任务性能**：SEED 的训练目标促使 encoder 学习更具预测性的表示（类似 MTP 的“提前规划”效应），从而**提升任务准确率**。
5. **与 AR 训练兼容**：可在已有 fine-tuned 模型上继续训练 SEED，**保留原有性能并获得加速**（见 Table 8）。

### **方法的局限性**
- **未验证更大模型**：实验集中在 1.7B 和 4B 规模，尚未在 8B+ 模型上验证效果。
- **固定分割策略**：encoder-decoder 分割（如前 26 层 + 后 2 层）是手动设定，可能非全局最优。
- **依赖 KV Cache 实现**：实际性能受 KV cache 管理、树注意力 kernel 等工程细节影响较大。

### **未来工作方向**
- **扩展到更大模型**：验证 SEED 在 8B、14B、32B 等规模上的表现。
- **自适应分割机制**：根据输入动态调整 encoder/decoer 分界层。
- **结合 post-training 或 pretraining**：在指令微调、偏好优化等阶段引入 `Lspec`，打造原生支持 SEED 的通用模型。
- **硬件感知优化**：开发定制 kernel 支持 decoder-only drafting 和 tree-based attention，进一步释放性能潜力。

---

> **一句话总结**：  
> SEED 通过将 decoder-only 模型重构为隐式 encoder-decoder，实现了**廉价且高质量的自我推测解码**，在保持甚至提升生成质量的同时，达到高达 **2.7× 的平均加速**，超越 EAGLE-3 等 SOTA 方法。

</details>

---

### 10. [VLALight: A Vision-Language-Action Model for Traffic Signal Control](https://arxiv.org/abs/2609.36934)

**Authors**: Pan Zhang, Siqi Lai, Kemu Dong, Hao Liu  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.36934v1  

#### Abstract
Traffic signal control (TSC) is essential for improving urban mobility and reducing congestion. Although roadside cameras are widely deployed at signalized intersections and provide rich visual observations of evolving traffic, existing TSC methods typically rely on manually engineered traffic state...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# VLALight: A Vision-Language-Action Model for Traffic Signal Control 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的**交通信号控制 (TSC)** 方法通常依赖于手工设计的交通状态（如队列长度、车辆计数等），这些状态在真实世界中可能不完整或不可靠。尽管路边摄像头提供了丰富的视觉观测数据，但现有系统未能实现从原始视频到信号动作的**端到端控制**，导致感知与决策之间存在鸿沟。

### 提出了什么新方法或新思路
本文提出了 **VLALight**，是首个用于交通信号控制的**视觉-语言-行动 (Vision-Language-Action, VLA) 模型**，其核心创新如下：

- **端到端视觉控制**：直接将多视角路边摄像头视频映射为协调的信号控制动作，无需中间的人工特征工程。
- **多目标时空推理与协同感知**：通过**拓扑感知的协同感知 (topology-aware cooperative perception)**，聚合相邻交叉口的信息，实现网络级协调控制。
- **自适应快慢推理模式 (adaptive fast/slow reasoning)**：模型能根据交通复杂度动态选择“快速”或“深度推理”的“慢速”模式，平衡决策质量与推理成本。
- **两阶段训练框架**：
  1. **监督冷启动 (supervised cold-start)**：分阶段训练视觉理解与决策能力。
  2. **协作式代理强化学习 (cooperative agentic RL)**：联合优化局部控制与全局网络效率，并通过**平衡的快慢推理回放 (balanced rollouts)** 学习何时进行深度思考。

### 相比现有方法的优势
- **输入更真实**：仅依赖摄像头视频，而非理想化的仿真器状态。
- **控制更智能**：通过VLA架构实现从物理观察到行动的统一策略，具备更强的泛化与推理能力。
- **效率更高**：自适应推理机制避免在简单场景下浪费计算资源，显著降低平均延迟。
- **网络协调更好**：利用图拓扑传递上下文，有效缓解拥堵传播与溢出。

---

## 2. 核心实验方法和设置

### 使用的数据集
在来自三个城市的**7个真实交通流数据集**上进行了验证：
- **Jinan 1-3**：12个交叉口，3×4网格
- **Hangzhou 1-2**：16个交叉口，4×4网格
- **New York 1-2**：196个交叉口，28×7网格（大规模城市路网）

### 实验设置
- **环境**：使用 **TranSimHub**（基于SUMO的三维交通模拟平台）重放真实轨迹并渲染路边视频。
- **控制周期**：每30秒一个决策周期（25秒绿灯 + 5秒黄灯）。
- **相位选项**：ETWT（东西直行）、ELWL（东西左转）、NTST（南北直行）、NLSL（南北左转）。
- **输入**：每个交叉口接收四个方向的视频序列（6帧，512×960分辨率）。

### 评估指标
- **Average Travel Time (ATT)**：车辆平均行程时间
- **Average Queue Length (AQL)**：网络平均排队长度
- **Average Waiting Time (AWT)**：车辆平均等待时间  
（数值越低越好）

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **交通工程方法** | FixedTime, MaxPressure |
| **RL方法** | PressLight, MPLight, CoLight, CityLight |
| **LLM/VLM方法** | LLMLight, CoLLMLight, VLMLight, Qwen3.5-27B |
| **VLA方法** | Owen3.5-27B |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1）
- 在**21项**（7数据集 × 3指标）对比中，**VLALight在16项中排名第一**。
- 在大规模纽约网络上表现尤为突出：
  - **New York 1**：ATT = 975.80, AQL = 1565.44, AWT = 566.82
  - **New York 2**：ATT = 1240.11, AQL = 3121.64, AWT = 880.49
- 相比最优基线，在纽约网络上：
  - **AQL降低13.0%（NY1）和5.5%（NY2）**
  - **AWT降低12.1%（NY1）和6.7%（NY2）**

### 与基线方法的对比结果
- 显著优于传统方法（如FixedTime、MaxPressure）和主流RL方法（如CoLight）。
- 超过LLM/VLM类方法（如LLMLight、CoLLMLight），表明**直接从视觉端到端学习优于分离的感知+决策架构**。
- 即使与强大的通用VLM（如Qwen3.5-27B）相比，VLALight仍大幅领先，说明**领域专用训练的重要性**。

### 消融实验结果（Table 2 & Table 3）
| 变体 | New York 1 (ATT/AQL/AWT) | New York 2 (ATT/AQL/AWT) | 结论 |
|------|--------------------------|--------------------------|------|
| **SFT**（仅监督微调） | 1171.08 / 2439.95 / 866.98 | 1439.73 / 4021.12 / 1164.74 | 性能最差，证明RL必要性 |
| **w/o coop.**（无协同信息） | 1067.60 / 2365.90 / 914.98 | 1317.63 / 3690.11 / 1113.65 | 协同感知对缓解视野受限至关重要 |
| **w/o balance**（无平衡回放） | 984.76 / 1654.22 / 595.02 | 1296.10 / 3328.31 / 938.46 | 平衡探索防止策略偏向慢模式 |
| **w/o net. reward**（无网络奖励） | 991.34 / 1647.22 / 599.61 | 1261.11 / 3230.98 / 924.42 | 网络级奖励对全局优化不可或缺 |
| **VLALight**（完整模型） | **975.80 / 1565.44 / 566.82** | **1240.11 / 3121.64 / 880.49** | 所有组件共同作用达到最优 |

#### 自适应推理效率（Table 3 & Table 5）
| 模式 | 平均推理延迟 | 说明 |
|------|-------------|------|
| **Pure Fast** | 2.866 秒 | 快速决策，适用于简单场景 |
| **Pure Slow** | 4.507 秒 | 深度推理，代价高 |
| **VLALight (adaptive)** | **3.936 秒** | 比纯慢模式**降低12.7%延迟**，接近最优性能 |

- **慢模式占比约48%**，且随队列压力增加而单调上升（见Figure 2），证明策略合理。

---

## 4. 关键结论和发现

### 主要发现
1. **VLA模型可用于真实物理世界的交通控制**：VLALight首次实现了从多视角视频到信号动作的端到端控制，验证了VLA范式在现实世界中的潜力。
2. **协同感知至关重要**：通过图拓扑传递邻居信息，有效弥补单点视野限制，提升网络级协调能力。
3. **自适应推理是高效的关键**：模型学会在复杂场景下启用深度推理，在简单场景下快速响应，实现**性能与效率的帕累托最优**。
4. **平衡探索促进稳定学习**：强制平衡的快慢回放防止策略陷入过度推理的局部最优。

### 方法的局限性
- **计算开销仍较高**：尽管已优化，但VLA模型的推理延迟（~4秒）仍高于传统轻量级控制器。
- **依赖高质量视频**：在恶劣天气或低光照条件下性能可能下降。
- **可解释性有限**：虽然生成推理链，但内部注意力机制仍较难完全解释。

### 未来工作方向
- 探索更高效的VLA架构（如TinyVLA、VLA-Cache）以降低延迟。
- 引入多模态输入（如雷达、GPS）增强鲁棒性。
- 扩展至更复杂的控制任务（如公交优先、应急车辆通行）。
- 部署到真实路口进行实地测试。

---

> **项目代码开源地址**：[https://github.com/usail-hkust/VLALight.git](https://github.com/usail-hkust/VLALight.git)

</details>

---

### 11. [Routing Should Pay for Itself: Sparse Supervision for Economical LLM Routing](https://arxiv.org/abs/2609.37402)

**Authors**: Guannan Lai, Gelin Bian, Hao-Xuan Ma, Jun-Peng Jiang, Long Chen, Jian-Dong Liu, Zhi-Hao Tan, Han-Jia Ye  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.37402v1  

#### Abstract
Large language model (LLM) routing reduces serving cost by assigning each query to an appropriate model while preserving response quality. Learning such a router, however, often requires executing multiple candidate models on historical queries to collect query--model quality feedback, creating a no...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Routing Should Pay for Itself: Sparse Supervision for Economical LLM Routing**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 **LLM routing** 方法通常依赖于在大量历史查询上对所有候选模型进行密集执行，以收集 `query-model` 质量反馈（即 dense supervision），从而训练一个高效的路由器。然而，这种监督获取过程本身会产生高昂的**前期成本（upfront supervision cost）**，可能远超后续推理阶段节省的成本。

现有研究大多只关注部署时的效率（serving-time efficiency），而忽略了这一“先投入、后回报”的经济平衡问题。本文指出：**路由系统必须“自己支付自己的开销”**（Routing Should Pay for Itself），否则即使部署时省了钱，总体仍是亏损。

此外，作者观察到：**routing quality 往往在获得全部反馈前就已饱和**，说明 dense supervision 是经济上的过度供给（over-provisioned）。

---

### **提出的新方法与新思路**
为解决上述问题，作者提出了 **SAVEROUTER** —— 一种基于**稀疏监督（sparse supervision）** 的 LLM 路由框架，其核心思想是：

- **选择性地获取最有信息量的 query-model 反馈**，而非均匀采样。
- 利用**相关查询之间的能力共享结构**（structured capability estimation）来泛化未观测的行为。
- 在保持细粒度路由能力的同时，显著降低监督成本。

#### **关键创新点：**
1. **首次将监督成本纳入 LLM 路由的整体经济评估中**，并提出两个新指标：
   - **SA-BEP**（Supervision-Amortized Break-Even Point）：需要多少部署请求才能收回前期监督成本。
   - **SA-CR**（Supervision-Amortized Cost Ratio）：在固定部署规模下，总成本（含监督）相对于基准系统的比率。
2. 设计了 **adaptive sparse feedback acquisition** 策略，结合模型能力估计与不确定性（UCB-style），动态决定采集哪个 query-model 对。
3. 构建了 **hierarchical capability estimation** 模型：
   - 先通过 group-model 结构建模共性（structured prior）
   - 再利用局部观测进行修正（evidence-based correction）
   - 最后加入轻量级残差预测器捕捉 query-level 差异（residual correction）

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **经济性** | 显著减少监督成本，使 break-even 时间大幅提前（1.9–9.5× 加速） |
| **效率与质量平衡** | 仅使用 33–41% 的训练反馈，仍能维持甚至超越全监督方法的 routing quality |
| **通用性** | 不依赖特定任务标签，支持聚类分组；适用于多种 benchmark 和成本配置 |
| **决策导向设计** | 强调“决策充分性”（decision sufficiency）而非“信息完整性”，避免冗余监督 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个异构的 LLM routing benchmark 上进行了评估：

| 数据集 | 特点 |
|--------|------|
| **LLMRouterBench** | 多任务文本理解，10个任务组，12种模型 |
| **Mixinstruct** | 指令遵循任务，单一全局组，12种模型 |
| **MMRBench** | 多模态任务（图文），7个数据集组，10种模型 |
| **RouterBench** | 高多样性任务，85个任务组，11种模型 |

所有数据集均采用标准的 20%/80% 训练/测试划分（seed=42）。

---

### **实验设置与评估指标**

#### **监督预算设置**
- 每个训练 query 最多评估 **K=4** 个候选模型（即最多 33–41% 的完整矩阵覆盖率）
- 所有方法在同一 ORBIT 工具包内复现，保证公平比较

#### **评估指标**
| 指标 | 定义 | 越小越好？ |
|------|------|-----------|
| **Ps**（Peak Score） | 最高测试得分 | ✅ |
| **CR**（Cost Ratio） | 达到目标质量所需的最小归一化服务成本 | ✅ |
| **SA-BEP** | 收回监督成本所需部署请求数 | ✅ |
| **SA-CR@1M** | 在 100 万次部署后的总成本比 | ✅ |

> 注：SA-BEP 和 SA-CR 将监督成本与服务成本联合考虑，更真实反映经济效益。

---

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **全监督基线** | EmbedLLM, kNN, OmniRouter, RMSoftmax, TRouter, UniRoute, InferenceDynamics |
| **稀疏/部分监督基线** | WISERouter (bandit-style), BaRP (bandit feedback), SemiRouter (anchor models) |

所有方法均在相同训练/测试划分和成本字段下运行。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（主实验 K=4）**

| 方法 | 平均监督比例 | Ps | CR | SA-BEP | SA-CR@1M |
|------|----------------|-------|-------|------------|-------------|
| **SAVEROUTER** | **33–41%** | **最高** | **最低** | **5.3K–42.6K** | **0.236–0.695** |
| 最快传统全监督 | 100% | 次优 | 较高 | 11.5K–326.4K | 0.364–0.953 |
| BaRP | ~100%* | 中等 | 中等 | 114K–44.7M | 0.394–2.333 |
| WISERouter/SemiRouter | 变动 | 常无法达标 | — | ∞（不可达） | ∞ |

> *注：BaRP 因重复交互导致实际监督成本极高（fresh-feedback accounting）

#### **核心发现：**
- SAVEROUTER 在所有四个 benchmark 上实现了 **最低的 SA-BEP**，平均比最快的传统全监督方法快 **1.9–9.5×**。
- 在 Mixinstruct 上，尽管各方法 Ps 接近饱和（≈0.75），SAVEROUTER 仍将 SA-BEP 从 11.63M 降至 **1.23M**，凸显稀疏监督的经济价值。
- 即便 Ps 提升有限，**更低的监督成本也能极大加速回本**。

---

### **消融实验结果（LLMRouterBench, K=4）**

| 变体 | Ps | CR | SA-BEP | SA-CR@1M |
|------|-------|-------|---------|-------------|
| **SAVEROUTER（完整）** | **0.6338** | **0.2320** | **5.3K** | **0.2360** |
| w/o Query Residual | 0.6175 | 0.3484 | 6.2K | 0.3525 |
| Capability-only | 0.6291 | 0.2587 | 5.9K | 0.2630 |
| Uncertainty-only | 0.6135 | 0.3411 | 4.0K | 0.3435 |
| Random-K | 0.6183 | 0.2958 | **3.8K** | 0.2983 |

#### **关键结论：**
- **Query-level residual correction 至关重要**：移除后 CR 明显上升，说明 group-level 估计不足以捕捉个体差异。
- **能力 + 不确定性驱动的 acquisition 更优**：UCB 策略在质量与长期成本之间取得最佳权衡。
- **最早回本 ≠ 最佳长期效益**：Random-K 虽然 SA-BEP 更低（3.8K），但 SA-CR@1M 更差，说明短期优势可能牺牲长期效率。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Dense supervision 是经济上的浪费**：routing quality 常在获得 30–60% 反馈后即趋于饱和（如 EmbedLLM、kNN），继续增加监督边际收益极低。
2. ✅ **监督成本必须被计入总成本**：忽略 upfront expenditure 会导致对路由系统的真实效益产生误判。
3. ✅ **SAVEROUTER 实现了“早回本、高效率”双赢**：
   - 仅用约 1/3 的监督量
   - 保持甚至提升 routing quality
   - 将 break-even 请求量减少 **1.9–9.5×**
4. ✅ **最优监督水平 ≠ 最大监督水平**：更多监督不一定带来更早回本或更低摊销成本，存在权衡。

---

### **方法的局限性**
- **依赖 query grouping**：若 query 分组不合理（如聚类失败），会影响结构先验的有效性。
- **不支持完全零样本新增模型**：虽然支持稀疏扩展（见 Appendix F），但仍需少量新模型反馈。
- **假设反馈可独立获取**：未建模多模型并行执行或缓存机制的影响（尽管在 Appendix E.7 中分析了 caching 敏感性）。
- **离线训练范式**：目前为静态训练，未考虑在线增量更新场景。

---

### **未来工作方向**
1. **动态调整监督预算 K**：根据部署预期流量自动优化采集数量。
2. **支持 zero-shot 或 few-shot 新模型接入**：结合 prompt engineering 或参数高效微调实现真正冷启动兼容。
3. **引入缓存与重用机制**：探索 feedback caching 对监督成本的进一步压缩潜力。
4. **扩展至 cascade routing 场景**：将稀疏监督思想应用于多跳推理路径规划。
5. **构建端到端经济优化目标**：将 SA-BEP 或 SA-CR 直接作为训练目标进行优化。

---

> 🔗 **代码开源地址**：[https://github.com/LAMDA-Model-Reuse/SaveRouter](https://github.com/LAMDA-Model-Reuse/SaveRouter)

</details>

---

### 12. [Efficient and Scalable Physics-Guided Fully Convolutional Spatiotemporal Learning for 3D Microstructure Evolution Prediction](https://arxiv.org/abs/2609.36504)

**Authors**: Michael Trimboli, Wenxi Liu, Xianqi Li  
**Category**: cs.LG  
**Published**: 2026-09-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.36504v1  

#### Abstract
Accurate prediction of three-dimensional (3D) microstructure evolution remains computationally demanding because high-fidelity phase-field simulations require repeated numerical integration over large volumetric domains and long temporal horizons. This study develops an efficient and scalable physic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **高计算成本的3D微结构演化预测**：传统的基于相场模型（phase-field）的模拟（如Cahn-Hilliard方程求解）在三维空间中需要大量数值积分，尤其在高分辨率、长时间尺度下计算代价极高，难以满足多查询场景（如参数优化、不确定性分析）的需求。
- **现有机器学习代理模型的局限性**：许多深度学习方法依赖循环结构（recurrent architectures），存在推理慢、长时序不稳定、难以扩展到3D等问题；同时，纯数据驱动模型可能偏离物理规律，导致长期预测失真。

### **提出的新方法与新思路**
- **非递归全卷积时空学习框架（nonrecurrent fully convolutional spatiotemporal framework）**：
  - 采用 **Encoder-Translator-Decoder** 架构，直接从输入序列映射到多个连续未来的3D微结构帧，实现**多帧并行预测**，避免了逐帧递推带来的误差累积。
- **因子化解码器（factorized latent translator）**：
  - 将时空动态分解为三个独立操作：**时间混合（temporal mixing）**、**局部3D空间交互（spatial mixing）** 和 **通道交互（channel mixing）**，避免显式的4D卷积，显著降低内存和计算开销。
- **物理引导训练（physics-guided training）**：
  - 在训练过程中引入离散化的 **Cahn-Hilliard (CH) 残差项** 作为正则化损失，使预测轨迹更符合物理动力学，但**不改变推理路径**，因此无额外推理开销。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **效率** | 推理速度比谱方法相场求解器（SpectralETD）快 **超过30倍**（32.3×），且支持高通量重复预测 |
| **可扩展性** | 全卷积设计天然支持不同分辨率迁移，无需重新训练即可应用于更高分辨率的3D数据 |
| **稳定性** | 非递归结构避免了RNN类模型的梯度爆炸/消失问题，适合长时序外推 |
| **物理一致性** | 物理残差正则化提升了界面形态（interface morphology）和粗化行为（coarsening dynamics）的保真度，尤其在低上下文条件下表现更鲁棒 |

---

## **2. 核心实验方法和设置**

### **数据集**
- **生成方式**：使用 `SpectralETD` 求解器对 **3D Cahn-Hilliard 方程** 进行数值模拟，模拟 **spinodal decomposition（旋节分解）** 过程。
- **参数配置**：
  - 网格大小：`128 × 128 × 128`
  - 时间步长：`Δt = 0.01`，每10个单位时间保存一帧，共201帧（对应2000仿真时间）
  - 初始条件：浓度均值为0.5，叠加高斯噪声
  - 参数固定：`W=1.0`, `M=1.0`, `K=0.01`
- **划分**：
  - 训练集：80条轨迹 → 2960个样本
  - 验证集：10条轨迹 → 370个样本
  - 测试集：10条轨迹 → 多种任务下测试（370或270样本）

### **实验设置**
- **任务类型**：
  1. **10→10**：用前10帧预测后10帧（nominal）
  2. **10→40**：迭代rollout预测40帧（long-horizon）
  3. **5→15 / 1→19**：减少输入帧数，测试低时间上下文下的泛化能力
- **模型架构细节**：
  - 编码器：4层3D Conv，空间分辨率从 `128³ → 32³`，通道数 `1 → 16`
  - Translator：堆叠多个残差块，分别进行时间、空间、通道混合
  - 解码器：上采样恢复至原始尺寸，带跳跃连接（skip connection）
- **训练配置**：
  - 优化器：Adam，学习率 `1e-3`
  - 批大小：1
  - 总轮数：200 epochs
  - 物理损失权重：`λ_phy = 1e-8`
  - 物理损失仅作用于最后5个预测帧

### **评估指标**
| 类型 | 指标 | 描述 |
|------|------|------|
| **重建质量** | RMSE, 3D SSIM, PSNR | 衡量预测与真实体素之间的相似性 |
| **形态保真度** | Interface Curvature Distribution（界面曲率分布） | 反映相界面几何特征，体现粗化动力学 |
| | Total Variation Distance（总变差距离） | 量化预测与真实曲率分布的差异，越小越好 |
| **效率** | Wall-clock Inference Time | 单次推理耗时，用于对比加速比 |

### **基线方法对比**
- **Data-driven baseline**：相同网络结构但无物理损失（`λ_phy = 0`）
- **Reference Solver**：`SpectralETD` 数值求解器（作为“黄金标准”但极慢）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **1. Nominal Task (10→10)**
- **平均3D SSIM > 0.97**（最高达0.986），RMSE ≈ 0.007，PSNR ≈ 43 dB
- 表明在充分上下文下，模型能高度还原微结构演化过程
- 2D切片可视化显示内部结构一致性良好

#### **2. Long-Horizon Forecasting (10→40)**
- 虽然两模型的RMSE/SSIM趋势接近，但：
  - **物理引导模型的界面曲率分布更接近真实值**
  - 在t=50时，其 **Total Variation Distance 显著更低**
- 说明物理正则化有效维持了**形态级稳定性**，而非仅仅提升像素精度

#### **3. Reduced Temporal Context (5→15, 1→19)**
- 输入信息越少，物理引导的优势越明显：
  - 在 **1→19** 任务中，物理模型在t=20时：
    - **3D SSIM 达 0.8 vs 基线 0.7**
    - **PSNR 达 15.5 vs 基线 14**
    - **曲率分布TV距离下降约30–50%**
- 基线模型出现明显退化，而物理模型仍保持连贯的双连续结构

#### **4. 计算效率**
| 模型 | 分辨率 | 平均推理时间（秒） | 加速比（vs SpectralETD） |
|------|--------|---------------------|----------------------------|
| Proposed Model | 128³ × 40帧 | 0.3682 s | **32.3×** |
| SpectralETD | 同上 | 11.896 s | 1× |

> ✅ **重要结论**：物理引导**不增加推理时间**，训练时间仅多约2小时。

---

## **4. 关键结论和发现**

### **主要发现**
1. **直接多帧预测优于递归推进**：
   - 非递归全卷积框架能高效建模有限时间窗口内的完整演化路径，避免误差传播。
2. **物理正则化显著增强鲁棒性**：
   - 尤其在**低时间上下文或长时外推**场景下，CH残差约束帮助模型“记住”物理规律，防止发散。
3. **形态保真度 ≠ 像素相似度**：
   - RMSE/SSIM可能掩盖结构性偏差；**interface curvature distribution 是更敏感的评价指标**。
4. **高吞吐代理模型可行性验证**：
   - 该框架实现了**高保真 + 高速 + 可扩展**的统一，适用于材料设计中的大规模仿真替代。

### **方法的局限性**
- **参数固定性**：当前模型仅针对单一CH参数组训练，未考虑材料参数变化（如`W`, `K`, `M`）的影响。
- **二元系统限制**：仅处理单浓度场的binary system，未扩展至multicomponent alloy。
- **零填充引入伪影**：在少帧输入实验中使用zero-padding可能导致短期异常响应。
- **软约束而非硬约束**：CH残差是正则项，不能保证严格的质量守恒或能量耗散。

### **未来工作方向**
- **参数条件建模（parameter-conditioned modeling）**：将物理参数作为输入，构建通用代理模型。
- **实验数据融合**：结合tomography等实测3D microstructure进行监督或域适应。
- **不确定性建模与数据同化**：引入贝叶斯框架或Kalman-filter-like机制，支持在线修正。
- **多物理场耦合扩展**：推广至含弹性应变、热传导等耦合效应的复杂相场系统。

---

> 📌 **总结一句话**：  
> 本文提出了一种**高效、可扩展、物理引导的全卷积时空学习框架**，首次实现了**高质量、高速度、长时稳定的3D微结构演化直接预测**，为数字孪生与高通量材料模拟提供了强有力的代理模型工具。

</details>

---

### 13. [Probe-Space Preconditioning for Fast and Stable Zero-Order Training](https://arxiv.org/abs/2609.38095)

**Authors**: Francois Chaubard, Mykel J. Kochenderfer, Chris R\'e  
**Category**: cs.LG  
**Published**: 2026-09-30  
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
当前深度学习训练严重依赖 **Backpropagation (BP)** 和自适应优化器（如 **Adam**），但这种方法存在显著的 **内存开销**。例如，训练 OPT-30B 模型需要约 600GB GPU 内存，主要来自存储激活值、梯度和优化器状态。这限制了在单个设备上训练大模型的能力。

此外，神经网络损失曲面通常具有高度病态的 Hessian 矩阵（即特征值分布极广），导致训练不稳定，尤其在 **零阶优化 (Zero-Order Optimization, ZOO)** 中，由于缺乏梯度信息，收敛速度慢且噪声大。

本文旨在设计一种满足以下三个目标的新求解器：
1. **推理模式内存使用**：不存储激活、梯度或动量状态。
2. **对病态损失鲁棒**：能在高曲率变化下稳定收敛。
3. **高效训练计算**：以更少的浮点运算实现更高的性能提升。

---

### **提出的新方法：1.5-SPSA**

作者基于经典的 **1SPSA (Simultaneous Perturbation Stochastic Approximation)** 进行改进，提出了 **1.5-SPSA**，其核心创新如下：

#### ✅ **1. 大批量 + 多探针（Probes）策略**
- 在固定前向传播预算下，将计算资源从“大量训练步数”转向“每步更多并行探针和更大有效 batch size”。
- 发现：**更大的 batch size 和更多 perturbations 能显著提升收敛速度和最终精度**，尽管存在收益递减现象。
- 优势：该策略可高度并行化，大幅缩短 wall-clock 时间。

#### ✅ **2. 探针空间预条件化（Probe-Space Preconditioning）**
- 引入一个额外的 **干净前向传播（clean forward-pass）** 来估计每个探针方向上的 **标量曲率 $ C_i $**。
- 设计 **α-饱和加权方案**：  
  $$
  w_i = \frac{1}{\max(\lambda_{\text{reg}}, |C_i|^\alpha)}
  $$
  其中 $\alpha=0.1$ 表现最优，用于抑制高曲率方向更新，避免发散。
- 本质是在低维探针子空间中进行近似的对角牛顿法，但仅需 $O(n_{\text{pert}})$ 额外计算，而非 $O(d)$ 存储。

#### ✅ **3. 高效系统实现**
- **8-bit 打包随机生成器**：Rademacher 扰动以 1 bit/参数存储，节省内存。
- **Triton 融合内核**：unpack + scale + apply 一步完成，加速扰动生成与应用。
- **分布式并行**：通过种子广播，各 GPU 自行生成扰动，仅回传 loss 标量，通信开销极小。

---

### **相比现有方法的优势**
| 维度 | 1.5-SPSA vs. MeZO / BP |
|------|------------------------|
| **内存** | ~10× 更少（~60GB vs. ~600GB for OPT-30B） |
| **训练步数** | 减少 **3–4 个数量级**（70 步 vs. MeZO 的 100,000 步） |
| **总计算量** | 达到 SOTA 性能所需 **前向传播次数减少 44×** |
| **收敛稳定性** | 曲率重加权显著提升后期训练稳定性 |
| **适用性** | 支持原地训练（in-place），适用于商品级 GPU（如 A100） |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **GLUE/SuperGLUE 基准任务**：
  - **SST-2**（情感分类）
  - **RTE**, **BoolQ**, **WSC**, **WiC**（自然语言推理与理解）
- **Stable ToolBench**（工具调用任务，用于 Qwen3-1.7B）

### **模型架构**
- **OPT 系列**：OPT-13B, OPT-30B
- **Qwen3 系列**：Qwen3-1.7B, Qwen3-8B

### **实验设置**
- **优化器配置**：
  - **1SPSA / 1.5-SPSA**：$\epsilon = \eta$, sweep $\eta \in \{10^{-3}, ..., 10^{-7}\}$, $\alpha=0.1$, plateau learning rate decay
  - **Adam (BP baseline)**：$(\beta_1, \beta_2)=(0.9, 0.99)$, weight decay $10^{-3}$
- **硬件平台**：8×A100 GPU 集群
- **评估指标**：
  - 测试准确率（Test Accuracy）
  - 收敛所需优化步数（Optimization Steps）
  - 总前向传播次数（Total Forward Passes）
  - Wall-clock 时间（部分实验）

### **基线方法对比**
| 方法 | 类型 | 是否需要反向传播 | 内存占用 |
|------|------|------------------|---------|
| **BP + Adam** | 一阶 | 是 | 高（~600GB） |
| **MeZO** | 零阶 | 否 | 低（~60GB） |
| **1SPSA** | 零阶 | 否 | 低 |
| **1.5-SPSA (Ours)** | 零阶 | 否 | 低 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **OPT-13B on SST-2**
| 方法 | SST-2 准确率 | 优化步数 | 前向传播总数 |
|------|--------------|----------|---------------|
| MeZO | 91.4% | 100,000 | ~200k |
| BP + Adam | 92.0% | 数万级 | 极高 |
| 1SPSA | 94.2% | 80 | ~205k |
| **1.5-SPSA** | **94.5%** | **70** | **~179k** |

✅ **1.5-SPSA 在仅 70 步内超越 MeZO (+3.1%) 和 BP (+2.5%)，且总计算更低。**

#### 🔹 **OPT-30B 结果（Table 2）**
- 在多个任务上均优于 MeZO 和 1SPSA，证明可扩展至超大规模模型。

#### 🔹 **Qwen3-8B（Table 3）**
| 方法 | SST-2 | RTE | BoolQ | WSC | WiC |
|------|-------|-----|-------|-----|-----|
| 1SPSA | 94.5 | 88.5 | 85.7 | 75.0 | 64.6 |
| **1.5-SPSA** | **94.7** | **88.0** | **86.1** | **80.8** | **71.2** |

✅ 显著提升 WSC 和 WiC 等复杂任务表现。

#### 🔹 **Qwen3-1.7B on Stable ToolBench（Table 4）**
- **1.5-SPSA 达到 79.0% 准确率（lr=5e-4）**，而 1SPSA 在 lr≥5e-4 时发散。
- 表明 **1.5-SPSA 支持更大学习率，加快收敛速率**。

---

### **消融实验结果**

#### 📊 **α 参数消融（Table 5）**
| α | 最佳准确率 | 状态 |
|----|------------|------|
| 0.01 | 93.6% | × Diverged |
| **0.1** | **94.5%** | ✔ Stable |
| 0.5 | 89.7% | Stable |
| 1.0 | 89.2% | Stable |

➡️ **α=0.1 是最佳平衡点**：既不过度压制曲率，又能有效稳定训练。

#### 📊 **ε 参数消融（Table 6）**
| ε | 最佳准确率 | 状态 |
|----|------------|------|
| 1e-4 | **94.5%** | BEST |
| <1e-5 或 >1e-3 | ≤80.5% | Fail |

➡️ **当 $\epsilon = \eta = 10^{-4}$ 时性能最优**，验证了“测量半径应等于更新步长”的直觉。

#### 📈 **系统性能对比（Table 7）**
| 方法 | 扰动生成速度 (ms/pert) | 总步耗时 (s) | 加速比 |
|------|------------------------|-------------|--------|
| PyTorch | 499.5 | 59.9 | 1.0× |
| Triton + Bit-packed | **102.1** | **21.7** | **2.76×** |

➡️ 系统优化带来显著 wall-clock 提升。

---

## 4. **关键结论和发现**

### **主要发现**
1. **重新分配训练计算预算至关重要**：  
   将资源集中于 **每步更多探针和更大 batch size**，而非增加步数，可大幅提升 ZOO 效率。

2. **探针空间曲率估计是有效的预条件方式**：  
   即使只用一个干净前向传播估计方向曲率，并进行轻量加权，也能显著改善收敛性和稳定性。

3. **1.5-SPSA 实现了 ZOO 的 SOTA 性能**：  
   在多个 LLM 家族和任务上，**以极少步数（<100）超越 MeZO 和标准 BP**，同时保持推理级别内存。

4. **性能增益与 Hessian 条件数正相关**：  
   在合成“刚性抛物面”和 DNC 模型压力测试中，**病态越严重，1.5-SPSA 相对 1SPSA 的加速越明显（最高达 7×）**。

---

### **局限性**
- **未对 BP 基线进行全面重调优**：作者承认 BP 在极端大 batch 少步场景下的潜力未被充分挖掘，因会违反内存约束。
- **α 和 ε 仍需手动调节**：虽有理论指导，但最优值依赖任务和模型，缺乏自动调度机制。
- **假设扰动方向独立同分布**：未探索结构化或自适应扰动方向（如 learned subspace）。

---

### **未来工作方向**
1. **自动化学习率与扰动规模调度**：结合线性/二次拟合等策略降低调参成本。
2. **批归一化式曲率标准化**：探索跨 batch 或 perturbation 的相对曲率归一化，进一步提升稳定性。
3. **扩展至 RL 或 MoE 训练**：利用其低内存特性，在强化学习或稀疏模型中应用。
4. **结合 Checkpointing 与 ZO**：探索混合范式，在内存与计算间取得更好权衡。

---

> **一句话总结**：  
> 1.5-SPSA 通过 **探针空间预条件化 + 大批量多探针策略 + 高效系统实现**，实现了 **低内存、快速、稳定** 的零阶训练，在多个大模型上以 **数十步** 超越传统方法十万步的效果，为大规模模型微调提供了新范式。

</details>

---

### 14. [PE-EK-PINN: Physics Embedding with Evolving Kernel for Scalable Physics-Informed Neural Networks](https://arxiv.org/abs/2609.38023)

**Authors**: Huiwen Zhang, Feng Ye, Chu Ma  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.38023v1  

#### Abstract
Physics-Informed Neural Networks (PINNs) embed governing equations into deep learning, but enforce them only through loss residuals, leaving highly oscillatory wave behavior to be discovered by optimization. As a result, methods that achieve relative $L_2$ errors below $10^{-3}$ on standard manufact...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PE-EK-PINN: Physics Embedding with Evolving Kernel for Scalable Physics-Informed Neural Networks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **传统 PINNs 在复杂波动场建模中的失效**：尽管在制造解（manufactured solution）基准上表现良好（相对 L2 误差 < 10⁻³），但在实际辐射问题中（如偶极子阵列、吸收边界、奇异激励）性能急剧下降（例如 CoPINN 从 5.0×10⁻³ 恶化到 9.95×10⁻¹）。
- **PE-PINN 的可扩展性瓶颈**：虽然通过将物理核（physics kernel）嵌入网络架构提升了精度，但其依赖于**手动构建的核字典**，导致核数量随系统规模呈指数增长（O(N)），形成“**kernel scalability problem**”。

### 提出的新方法：PE-EK-PINN
- **核心思想**：提出 **Physics Embedded with Evolving Kernels (PE-EK-PINN)**，将已收敛的子系统场表示冻结并提升为“**evolved kernel**”，作为更高层级配置的可复用物理感知构件。
- **工作机制**：
  - 子系统的 PE-PINN 输出被冻结为一个 evolved kernel。
  - 高层结构通过平移/旋转复制这些 evolved kernels，并仅训练新的 envelope 和 gating 函数。
  - 利用 Helmholtz 算子在均匀介质中的**刚体变换不变性**保证物理一致性。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可扩展性** | 将累计训练成本从 O(N) 降低至 O(log N)，峰值活跃核数与系统大小无关 |
| **效率** | 显著减少训练时间（如 256 偶极子阵列快 30 倍以上） |
| **精度** | 在多数情况下实现更低或相当的相对 L2 误差 |
| **自动化程度** | 消除对大规模系统需手动设计分析核的需求 |

---

## 2. 核心实验方法和设置

### 数据集与源配置
所有实验基于二维自由空间下 f = 2.4 GHz 的电磁波场，使用以下三种递归结构：
| 场景 | 基础单元 | 变换方式 | 层级发展 |
|------|--------|---------|----------|
| **Dipole Array** | 单个 dipole | 平移 | 1 → 2×2 → 4×4 → 8×8 → 16×16 |
| **Composite Line Geometry** | 2λ 连续线源 | 平移+旋转 | 构造十字形、五角星等复合几何 |
| **Cross Array** | 0.5λ 线源 | 平移+旋转 | 单十字 → 2×2 → 4×4 阵列 |

> ✅ 所有配置均有闭式解析解（Hankel 函数积分），用于训练目标和评估参考。

### 实验设置
- **域范围**：Ω = [-2.5, 2.5]² m²（部分扩大至 [-3.6, 3.6]² 以支持旋转外推）
- **采样密度**：约 12.5 点/波长（0.01 m 网格）
- **排除区域**：避免奇异性，在源附近设置胶囊状或圆形边界施加激励
- **网络结构**：隐藏层宽度 [40,120,120,120]，正弦激活函数
- **优化器**：Adam，学习率 1e-4，迭代 50,000 次
- **损失权重**：λ_pde=0.01, λ_src=10, λ_bc=1

### 评估指标
- **主指标**：**Complex Relative L2 Error**
- **辅指标**：Mean Squared Error (MSE)
- **效率指标**：阶段训练时间（Stage Time）、累计训练时间（Cumulative Time）

### 基线方法对比
- **主要基线**：**PE-PINN**（直接使用原始物理核）
- **其他对比方法**：PINN, gPINN, SPINN, CoPINN 等（见 Table 1）
- **特别说明**：由于早期 PINN 方法无法处理大尺度配置，本研究聚焦于 PE-PINN vs PE-EK-PINN 对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 📊 Dipole Array 结果（Table 3）
| 配置 | 方法 | N_kernels | Cum. Time | Rel. L2 |
|------|------|-----------|------------|---------|
| 2×2 | PE-EK-PINN | 4 | 01:13:31 | 1.62×10⁻² |
| 4×4 | PE-EK-PINN | 4 | 01:13:31 | 1.62×10⁻² |
| 8×8 | PE-EK-PINN | 4 | 01:41:00 | 2.71×10⁻² |
| 16×16 | PE-EK-PINN | 4 | **02:10:35** | 9.43×10⁻² |
| 2×2 | PE-PINN | 8 | — | 2.37×10⁻² |
| 4×4 | PE-PINN | 32 | — | 2.22×10⁻² |
| 8×8 | PE-PINN (extrap.) | — | ~17.9 h | — |
| 16×16 | PE-PINN (extrap.) | — | **~71.4 h** | — |

> 🔹 **速度提升 >30×**（256 dipole），且每阶段训练时间几乎恒定（~27 min）

#### 📊 Composite Line Source 结果（Table 4）
| 配置 | 方法 | Kernels | Stage T. | Rel. L2 |
|------|------|--------|----------|---------|
| Two-line cross | PE-EK-PINN | 2 | 00:16:29 | **3.34×10⁻²** |
| Two-line cross | PE-PINN | 20 | 03:13:19 | 1.33×10⁻¹ |
| 5-point star | PE-EK-PINN | 5 | 00:39:18 | 6.32×10⁻² |

> 🔹 **加速 11.73×（阶段）**，端到端仍快 1.7×，误差显著降低

#### 📊 Cross Array 结果（Table 5）
| 配置 | 方法 | Kernels | Stage T. | Rel. L2 |
|------|------|--------|----------|---------|
| Cross | PE-EK-PINN | 2 | 00:16:26 | **4.18×10⁻²** |
| Cross | PE-PINN | 8 | 01:00:50 | 9.33×10⁻² |
| 2×2 | PE-EK-PINN | 4 | 00:31:44 | **5.07×10⁻²** |
| 2×2 | PE-PINN | 32 | 04:05:07 | 1.01×10⁻¹ |

> 🔹 **单级加速 3.7×，2×2 阶段加速 7.72×**

### 与基线方法对比总结
| 维度 | 对比结果 |
|------|----------|
| **训练效率** | PE-EK-PINN 每阶段时间基本恒定，而 PE-PINN 时间随核数线性增长 |
| **模型规模** | 活跃 kernel 数量从 O(N) 降至 O(log N)，峰值内存占用可控 |
| **重建精度** | 多数场景下相对 L2 误差更低，尤其在交互效应主导的情形 |
| **可扩展极限** | 成功训练 256 dipole / 16-cross 系统，远超直接 PE-PINN 实际可行范围 |

### 消融实验与验证
- **有效性验证**：通过对比不同层级的 stage time 与 kernel 数量关系，验证了 O(log N) 成本缩放律（Fig. 4, Fig. 7）。
- **误差传播分析**：深层结构（如 16×16 dipole）误差略有上升，归因于近似误差累积及吸收边界假设不匹配。
- **外推风险控制**：通过预训练更大域的 kernel（[-3.6,3.6]²）缓解旋转导致的坐标外推问题。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **标准 benchmark 性能 ≠ 实际泛化能力**：在 manufactured Helmholtz 问题上表现优异的方法（如 CoPINN）在真实辐射场中可能完全失效。
2. ✅ **architecture-level physics embedding 至关重要**：仅靠 loss-based 正则化不足以捕捉高频振荡特性；必须将物理先验编码进网络结构。
3. ✅ **evolved kernel 可实现高效复用**：将已学子系统场抽象为固定 kernel，可在保持物理一致性的前提下大幅压缩训练负担。
4. ✅ **层级复用带来 O(log N) 复杂度增益**：累计训练成本由 O(N) 改善为 O(log N)，使大规模 structured system 的建模成为可能。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖已知层次结构** | 当前框架要求预先知道系统的递归组织方式，难以自动发现重复子结构 |
| **误差累积效应** | 冻结 kernel 中的近似误差会向上传播，影响高层精度，尤其在深层级时更明显 |
| **外推不可靠性** | MLP 表示的 kernel 在训练域之外进行评估时可能出现 extrapolation 错误（如大角度旋转后） |
| **同质介质假设** | 依赖 Helmholtz 算子的平移不变性，扩展至非均匀介质需额外处理 |

### 未来工作方向
1. **自动发现可复用子结构**：探索 clustering 或 representation learning 技术，自动识别潜在的 reusable subsystem。
2. **结合 neural operators**：利用 PE-EK-PINN 生成高质量数据训练 Fourier Neural Operator 等模型，提升多查询场景下的推理效率。
3. **推广至其他 PDE 类型**：将 evolving kernel 范式应用于弹性波、声学、薛定谔方程等具有类似对称性的系统。
4. **改进 kernel 表示形式**：研究更鲁棒的 kernel 参数化方式（如 spectral basis、wavelet expansion）以增强外推稳定性。

---

> 💡 **总结一句话**：  
> **PE-EK-PINN 通过将“已学会的物理”转化为“可复用的构件”，实现了从 O(N) 到 O(log N) 的训练复杂度跨越，在保持高精度的同时，首次实现了数百单元规模波动系统的高效 PINN 建模。**

</details>

---

### 15. [FastGuide: Accelerating Reward Guidance for Diffusion Large Language Models](https://arxiv.org/abs/2609.36202)

**Authors**: Darshan Thaker, Lachlan Ewen MacDonald, Ren\'e Vidal  
**Category**: cs.CL  
**Published**: 2026-09-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.36202v1  

#### Abstract
Gradient-based reward guidance provides a flexible way to use downstream reward models to control masked diffusion language models at inference time. However, its computational cost remains high as each decoding iteration incurs expensive diffusion model forward passes and reward model backpropagati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FastGuide: Accelerating Reward Guidance for Diffusion Large Language Models

## 1. 论文的主要贡献和创新点

### 解决的问题
扩散大语言模型（dLLMs）在推理时通过梯度引导（reward guidance）可以有效对齐下游目标（如指令遵循、真实性等），但其计算成本高昂。每次解码迭代都需要进行完整的 dLLM 前向传播和奖励模型（reward model）的反向传播，导致推理速度缓慢。

### 提出的新方法与思路
本文提出了 **FastGuide**，一种用于加速 dLLM 奖励引导的训练无关（training-free）、混合型（hybrid）且自适应的解码策略。其核心思想是将并行解码（parallel decoding）和自回归解码（autoregressive decoding）的优点结合起来，并针对奖励引导的特点进行优化。

具体创新点如下：
1.  **指导向量缓存（Guidance Caching）**：在每个解码步骤中，仅计算一次奖励模型的梯度（即指导向量 `rt`），然后将其重用于该步骤内所有待解码的多个 token。这显著摊销了昂贵的奖励模型反向传播开销。
2.  **稀疏 dLLM 重新计算（Sparse dLLM Recomputation）**：虽然指导向量可以安全地重用，但直接重用 dLLM 的 logits 会因“联合-边际不匹配”（joint-marginal mismatch）而降低生成质量。因此，FastGuide 在解码组内以自回归方式逐个解码 token，但在每次解码后，只对受此解码影响的局部 token 表示进行稀疏的前向传播更新（利用 KV 缓存技术），而非完整的 dLLM 前向传播。
3.  **基于置信度的延迟（Confidence Deferral）**：引入自适应机制，在每次稀疏重新计算后，如果某个候选 token 的预测分布置信度低于阈值，则将其推迟到后续步骤再处理。这动态平衡了并行性和自回归性，确保了解码质量。

### 相比现有方法的优势
- **高效性**：相比传统的顺序奖励引导解码，FastGuide 将推理速度提升了高达 **4.4倍**。
- **高质量**：在大幅提升速度的同时，生成质量（以奖励模型得分衡量）与顺序引导解码相当，远优于其他并行解码方法。
- **通用性**：其“指导向量可重用，dLLM logits 需重新计算”的原则，使得现有的多种并行解码器（如 Fast-dLLM, DAPD 等）都可以通过添加指导向量缓存来扩展到奖励引导场景。

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个奖励基准测试集上进行：
- **JudgeBench**：评估模型区分客观正确答案和看似合理但有细微错误的答案的能力。
- **RM-Bench**：测试奖励模型是否优先考虑实质性质量而非回复风格，涵盖聊天、代码、数学、安全等领域。
- **Reward-Bench-2**：覆盖事实准确性、精确指令遵循、数学、安全、专注度等。

### 实验设置和评估指标
- **模型**：在两个开源 dLLM 上进行评估：`Dream-7B-Instruct` 和 `LLaDA-8B-Instruct`。
- **奖励模型**：主要使用 `Skywork-Reward-V2-Qwen3-1.7B`，也测试了更小（0.6B）和更大（Llama-8B）的版本。
- **评估指标**：
  - **Top@1**：多次生成轨迹中的最高奖励模型得分。
  - **Seq Gap**：并行解码方法的 Top@1 与对应顺序解码基线之间的差距（越接近0越好）。
  - **LMUnit**：外部评判 LLM 给出的 1-5 分评分，用于评估回复质量并防止奖励过优化（reward overoptimization）。
  - **s/gen**：每条生成的平均耗时（秒），用于衡量推理速度。

### 基线方法对比
- **顺序引导解码（Sequential Guided Decoding）**：作为性能上限的基线。
- **Best-of-N 采样（BoN）**：一种无需训练的无引导基线。
- **现有并行解码器**：包括 `Conf` (基于置信度), `Fast-dLLM`, `KLASS`, `EB-Sampler`, `DAPD`。这些方法通常用于无引导场景，本文通过添加指导向量缓存使其适用于引导场景。

## 3. 主要实验结果和性能指标

### 关键性能数据
- **速度提升**：FastGuide 在不同基准和模型上实现了 **2.8 到 4.4 倍** 的加速，同时保持了与顺序引导解码相近的生成质量。
- **质量保持**：在 `Dream-7B` 模型上，FastGuide 的 `Seq Gap` 仅为 `-0.45` 至 `-0.72`，意味着其性能几乎与顺序解码持平。相比之下，其他并行方法的 `Seq Gap` 损失严重（如 `Fast-dLLM` 达到 `-4.66`）。
- **效率与质量权衡**：即使在高吞吐量模式下（k=16），FastGuide 的 `Seq Gap` 也控制在 `1.20` 以内，而其他方法的性能急剧下降。

### 与基线方法的对比结果
- **显著优于其他并行解码器**：在所有基准测试中，FastGuide 均取得了最小的 `Seq Gap` 和最高的 `Top@1` 分数，大幅领先于 `Fast-dLLM`, `KLASS`, `EB-Sampler` 等基线。
- **与最强基线 DAPD 对比**：`DAPD` 是表现最好的现有并行解码器，但 FastGuide 仍能超越它，尤其是在对不兼容性敏感的推理任务（如编码、数学）上优势明显。
- **与顺序引导对比**：FastGuide 的速度接近甚至超过了无引导的 `Best-of-N` 采样，但其 `Top@1` 分数比后者高出 `0.1-0.65` 个奖励点，证明了其在效率和效果上的优越性。

### 消融实验结果
- **指导向量缓存 vs. dLLM logits 重用**：实验证明，重用指导向量对质量影响很小，但重用 dLLM logits 会导致质量严重下降，验证了其混合策略的必要性。
- **置信度延迟（Confidence Deferral）**：移除延迟机制（固定解码k个token）会使速度更快但质量下降；增加延迟阈值T会提高质量但牺牲速度，证明了该机制提供了有效的质量-效率权衡开关。
- **稀疏重新计算的有效性**：稀疏重新计算得到的 token 分布与完整重新计算的结果非常接近（总变差距离小），证明了其近似是有效的。

## 4. 关键结论和发现

### 主要发现
1.  **指导向量可安全重用**：在高置信度位置，即使指导向量来自较早的扩散时间步，其引导后的 token 分布变化也很小，因此可以在一个解码步骤内被多个 token 安全地共享。
2.  **指导加剧了 token 不兼容性**：奖励引导会放大并行解码中固有的“联合-边际不匹配”问题，导致同时提出的 token 更容易出现局部不兼容。这是直接重用 dLLM logits 会失败的根本原因。
3.  **混合策略最优**：结合指导向量缓存（解决瓶颈）和稀疏的 dLLM 自回归重新计算（解决不兼容性），是实现高速且高质量奖励引导解码的最佳路径。

### 方法的局限性
- **依赖于 KV 缓存**：方法的有效性依赖于 dLLM 架构支持高效的 KV 缓存和稀疏注意力计算。
- **超参数调优**：需要调整候选池大小 `k`、延迟阈值 `T` 和注意力窗口半径 `w` 等超参数以达到最佳效果。
- **极端并行化受限**：由于置信度延迟机制的存在，实际解码的 token 数量是自适应的，无法保证在每个步骤都解码最大数量的 token。

### 未来工作方向
- **探索更复杂的规划器**：研究如何将 FastGuide 与其他高级规划策略（如外部自回归 LLM）结合。
- **优化稀疏计算**：进一步改进稀疏前向传播的算法和实现，减少内存开销。
- **理论分析**：为“指导向量可重用”和“指导加剧不兼容性”提供更深入的理论解释。
- **扩展到其他领域**：将 FastGuide 的思想应用于连续扩散模型或其他模态的扩散模型。

</details>

---

### 16. [MemEvo: Automatic Discovery of Streaming Video Memory Mechanisms](https://arxiv.org/abs/2609.36581)

**Authors**: Guohong Liu, Jialei Ye, Shanhui Zhao, Yunxin Liu, Yuanchun Li  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.36581v1  

#### Abstract
Query-agnostic streaming video understanding requires vision-language models to continuously compress an indefinitely growing visual stream into a bounded memory before future queries are known. The performance depends critically on the memory mechanism--what observations to preserve, how to represe...

---

### 17. [SimpleEvol: An Agent-Loop Framework for LLM-Driven Automated Heuristic Design with Minimal Human Priors](https://arxiv.org/abs/2609.37172)

**Authors**: Jianghan Zhu, Cong Zhang, Rongjie Zhu, Chi Zhang, Zhiguang Cao  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.37172v1  

#### Abstract
Large language models (LLMs) have emerged as powerful tools for automated heuristic design (AHD), enabling iterative generation and refinement of heuristics. However, the dominant paradigm embeds LLMs as narrow, fixed components, such as crossover or mutation, within heavily hand-engineered evolutio...

---

### 18. [FOCUS: Training-Free Decision-Preserving Context Compression for LLM Agents](https://arxiv.org/abs/2609.37590)

**Authors**: Shantanu Dixit, Anson Bastos, Xuchao Zhang, Chetan Bansal, Saravan Rajmohan  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.37590v1  

#### Abstract
LLM agents accumulate interaction histories that grow linearly with task length, causing quadratic inference cost scaling and performance degradation from attention dilution. Existing context-compression methods learn what to discard offline: by contrastively optimizing guidelines, distilling compre...

---

### 19. [It's All Training: A Fully Synthetic Single-Stage Recipe for LLMs](https://arxiv.org/abs/2609.37891)

**Authors**: Pierre-Carl Langlais, Pieter Delobelle, Yannick Detrois, Pavel Chizhov, Carlos Rosas-Hinostroza, Neil Si Smail, Benjamin Burtin, Hanna Shcharbakova, Ivan Yamshchikov, Anastasia Stasenko  
**Category**: cs.CL  
**Published**: 2026-09-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.37891v1  

#### Abstract
Current pre-training datasets are derived from web crawls, with all their issues, and were not designed to support mid- and post-training pipelines--for instance, they contain little explicit reasoning. Thus, many frontier labs have begun to develop their own internal datasets, starting from state-o...

---

### 20. [Koa-action: Fast and Consistent Structured Decision Making with Generative LLMs](https://arxiv.org/abs/2609.36115)

**Authors**: Shenghong Dai, Shiva Kumar Pentyala, Yingchi Liu, Shubham Mehrotra, Suman Banerjee, James Zhu, Bin Bi, Sitaram Asur, Phil Mui  
**Category**: cs.LG  
**Published**: 2026-09-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.36115v1  

#### Abstract
Industry applications often demand low-latency classification, yet current large language model (LLM) approaches remain poorly suited for latency-critical applications. Existing prompting and constrained decoding produce verbose, multi-token outputs that require expensive token-by-token generation, ...

---

### 21. [Message Passing Does More with Less for In-Context Learning on Graphs](https://arxiv.org/abs/2609.37057)

**Authors**: Dooho Lee, Jinmo Lee, Minho Jeong, Kijung Shin, Jaemin Yoo  
**Category**: cs.LG  
**Published**: 2026-09-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.37057v1  

#### Abstract
Achieving strong performance with graph neural networks (GNNs) typically requires training and hyperparameter tuning for each dataset, incurring repeated costs and effort. Graph in-context learning (ICL) avoids this by using a single pretrained model to predict unknown node labels directly from labe...

---

### 22. [GLASS: Global Latent Aggregation with Slot-based Set Decoding for Scalable All-Atom Crystal Generation](https://arxiv.org/abs/2609.37158)

**Authors**: Hendrik Kra{\ss}, Seyed Mohamad Moosavi, Mathias Niepert  
**Category**: cs.LG  
**Published**: 2026-09-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.37158v1  

#### Abstract
Generative models for crystals enable the discovery of novel structures, but scaling all-atom generation to larger systems such as metal--organic frameworks remains challenging. We connect this difficulty to the correspondence problem of particle-space generation. Even on a single fixed target set, ...

---

### 23. [SMat-Attention: Structured Long-Context Sequence Modeling](https://arxiv.org/abs/2609.36062)

**Authors**: Emile Anand, Abdullah Ateyeh, Archer Wang, Marin Solja\v{c}i\'c  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.36062v1  

#### Abstract
Long-context sequence models face a fundamental tradeoff: softmax attention uses flexible token-level interactions at quadratic cost, whereas linear attention obtains linear-time training and constant-time decoding by compressing history into a fixed-size state. In this work, we ask whether we can c...

---

### 24. [Principled Thoughts for Latent Recursive LLM Systems](https://arxiv.org/abs/2609.36159)

**Authors**: Fahd Seddik, Fatemeh Fard  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.36159v1  

#### Abstract
Large language models can reason in continuous space instead of decoded text, by recurring on their own hidden states or by passing those states between agents, while training supervises only the Cross-Entropy (CE) of the final decoded answer and does not constrain the thought. Theoretical and empir...

---

### 25. [Bits Under ZK-LLM: Evaluating Zero-Knowledge-Friendly Quantization for Verifiable Private LLM Inference](https://arxiv.org/abs/2609.36437)

**Authors**: Taeung Yoon, Yupeng Zhang, Xiaojing Liao  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.36437v1  

#### Abstract
Zero-knowledge proofs are emerging as a promising approach for enabling private, verifiable LLM governance and auditing, where regulators, users, and auditors need to verify claims about training-data usage or LLM inference-time behavior, while model providers must protect proprietary model paramete...

---

### 26. [AI as a Compiler: Compiling Triton kernels without the Triton compiler](https://arxiv.org/abs/2609.36800)

**Authors**: Fran\c{c}ois Costa, Charly Castes, Thomas Bourgeat, Azalia Mirhoseini  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.36800v1  

#### Abstract
Compiler backends are expensive to build and maintain as programming models, workloads, and accelerators evolve. We investigate whether large language models can replace the conventional optimizing and lowering pipeline, a process that we call AI lowering. We study AI lowering from Triton to NVIDIA ...

---

### 27. [Physics-Informed Multi-Agent Coordination for Hospital Patient Flow Optimization](https://arxiv.org/abs/2609.37022)

**Authors**: Guoqing Zhang, Rafik Hadfi, Takayuki Ito  
**Category**: cs.AI  
**Published**: 2026-09-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.37022v1  

#### Abstract
Efficient patient flow coordination across autonomous hospital departments is critical for mitigating overcrowding and balancing resource utilization. While classical queueing theory, specifically open Baskett--Chandy--Muntz--Palacios (BCMP) networks, provides an interpretable mathematical topology ...

---

### 28. [LatCom: Cross-Agent Latent Compression for Efficient Multi-Agent Collaboration](https://arxiv.org/abs/2609.37017)

**Authors**: Shinan Zhang, Tao Zhang, Qihui Zhu, Mengjie Zhang, Dong Jin, Yunpeng Hou, Shuangwu Chen, Xiaobin Tan, Quan Zheng, Jian Yang  
**Category**: cs.CL  
**Published**: 2026-09-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.37017v1  

#### Abstract
LLM-based multi-agent systems (MAS) increasingly use latent collaboration to avoid the information loss and repeated encoding-decoding overhead of natural-language communication. However, directly forwarding all sender latents makes the receiver-side context scale with both the number of agents and ...

---

### 29. [AnthroDial: Benchmarking LLM Anthropomorphism in Autonomous Social Interaction](https://arxiv.org/abs/2609.37853)

**Authors**: Wentao Liu, Xi Chen, Siyu Song, Biao Yuan, Yu Zhang, Zhou Zhuotong, Jingying Zhou, Guohao Feng, Shasha Hu, Tianfu Wang, Shangshang Yang, Haoyang Liu, Youjia Li, Xiaokun Wang, Min Ji, Ji Wang  
**Category**: cs.CL  
**Published**: 2026-09-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.37853v1  

#### Abstract
Large language models (LLMs) are increasingly deployed as social agents, yet credible human-like interaction requires more than fluent responses or persona consistency. Agents must autonomously decide whether, when, and how to communicate while adapting to evolving contexts, goals, and relationships...

---

### 30. [Efficient Agentic LLM Serving over SSD-based Sparse KV Storage](https://arxiv.org/abs/2609.36938)

**Authors**: Wenhao He, Ping Zhang, Xiaohe Hu, Chutian Wang, Jinlong Hou, Yuan Cheng, Peng Sun, Fangcheng Fu  
**Category**: cs.DC  
**Published**: 2026-09-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.36938v1  

#### Abstract
Agentic sessions driven by Large language models (LLMs) often alternate between model inference and tool use, accumulating long histories across successive rounds. Serving these sessions efficiently requires reducing attention computation and retaining history key-value (KV) caches to avoid recomput...

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

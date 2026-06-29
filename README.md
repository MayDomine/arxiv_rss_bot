# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-29 10:15:53 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FlexMoE: One-for-All Nested Intra-Expert Pruning for MoE Language Models](https://arxiv.org/abs/2606.27866)

**Authors**: Fan Mo, Yuxuan Han, Geng Zhang, Wangbo Zhao, Yang You  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.27866v1  

#### Abstract
Mixture-of-Experts (MoE) language models scale model ability with sparsely activated experts, making this architecture a standard recipe for modern large models. However, sparse activation does not remove the deployment burden of storing and serving all experts, and the available deployment budget c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FlexMoE: One-for-All Nested Intra-Expert Pruning for MoE Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **MoE (Mixture-of-Experts)** 压缩方法大多是**固定预算（fixed-budget）** 的，即针对某一特定部署资源（如内存、算力）优化一个压缩模型。这导致在面对不同设备、用户或动态负载时，需要为每个目标预算单独训练和维护一个模型，带来高昂的存储与管理成本。

此外，虽然 MoE 模型通过稀疏激活降低计算量，但所有专家参数仍需完整存储，限制了其在资源受限场景下的灵活部署能力。

### 提出了什么新方法或新思路
本文提出 **FlexMoE**，一种**后训练（post-training）的一体化嵌套剪枝框架**，将一个预训练的 MoE 大模型转换为一系列可部署的子网络家族（deployable subnetwork family），支持跨多种资源预算的“一次训练，多端部署”（train-once, deploy-many）。

其核心创新包括：

- **嵌套专家通道排序（Nested Intra-Expert Channel Ranking）**  
  首先基于一阶梯度重要性（first-order Taylor saliency）对每个专家 FFN 层的隐藏通道进行重排序，形成按重要性降序排列的权重布局。这样，任何前缀切片（prefix slicing）都能保留最重要的参数。

- **每专家离散动作学习（Per-Expert Discrete Action Learning）**  
  引入可学习的动作 logits，让每个专家从预定义的保留比例集合 $ \mathcal{A} = \{0.1, 0.4, 0.7, 1.0\} $ 中选择一个通道保留率。通过 **Gumbel-Softmax + Straight-Through Estimator** 实现梯度传播，在单次训练中逐步增加剪枝压力，导出覆盖多个预算的 action masks 序列。

- **单点恢复微调（Single-Point Recovery Fine-Tuning）**  
  在中等剪枝预算（如 40%）下进行一次 LoRA 微调，恢复模型质量，并将该恢复后的权重共享用于其他预算的子网络，实现跨预算的知识迁移。

- **算法-系统协同设计（Algorithm-System Co-Design）**  
  设计定制化的 CUDA kernel 来支持在线实时预算切换（online budget switching），通过 bucketing、对齐和 grouped GEMM 优化碎片化小矩阵运算，缓解动态剪枝带来的调度开销。

### 相比现有方法的优势
| 维度 | 现有方法（如 NAEE, MoE-SVD, TD-MoE） | FlexMoE（本文） |
|------|----------------------------------------|----------------|
| **灵活性** | 单一预算，需重复训练 | 支持多预算，一次训练生成整个子网族 |
| **部署效率** | 需维护多个 checkpoint | 只需保存一组 action masks 和一个基础模型 |
| **性能保持** | 剪枝后性能下降明显 | 在 50% 参数剪枝下仍保持 ~99.8% 原始性能 |
| **动态适应性** | 不支持运行时调整 | 支持 kernel-level 实时预算切换 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准与训练阶段**：
  - **Zyda-2**：高质量、5万亿 token 的无监督数据集，用于通道重要性排序、action learning 和 recovery fine-tuning。
- **评估阶段**（zero-shot 评测）：
  - **ARC-C / ARC-E**（科学推理）
  - **HellaSwag**（常识推理）
  - **OpenBookQA**（开放书本问答）
  - **PIQA**（物理常识）
  - **WinoGrande**（代词消解）
  - **MathQA**（数学应用题）
  - 使用 `lm-eval-harness` 工具包统一评测。

### 实验设置和评估指标
- **主干模型**：
  - **Mixtral-8x7B**
  - **Phi-3.5-MoE**
  - **Qwen2-57B-A14B**（高度稀疏架构，含共享专家）
- **剪枝粒度**：仅对 routed expert 的 FFN 通道进行结构化剪枝，router 结构不变。
- **评估指标**：
  - **任务准确率（Zero-shot Accuracy %）**
  - **检查点大小（Checkpoint Size, GB）**
  - **吞吐量（Throughput, tok/s）**
  - **加速比（Speedup）**

### 基线方法对比
| 基线方法 | 类型 | 特点 |
|---------|------|------|
| **NAEE [2024]** | Expert Pruning | 删除不重要的专家 |
| **MoE-SVD [2025]** | Low-Rank Decomposition | 对专家权重做奇异值分解 |
| **TD-MoE [2026]** | Tensor Decomposition | 跨专家联合张量分解 |
| **MoE-I2 [2024b]** | Inter+Intra Pruning | 专家间剪枝 + 专家内低秩分解 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 在 Qwen2-57B-A14B 上的表现（无需微调）
| 剪枝比例 | 性能保留率（vs Base） |
|--------|---------------------|
| 50%    | **~99.8%**          |
| 80%    | **~92.9%**          |

> 表明 FlexMoE 在高稀疏 MoE 架构上具有极强的鲁棒性和压缩潜力。

#### ✅ 吞吐提升（SGLang 运行时）
| 模型 | 剪枝比例 | Checkpoint Size ↓ | Throughput ↑ | Speedup × |
|------|----------|--------------------|---------------|-----------|
| Qwen2-57B-A14B | 60% | 51.7 GB → 107.0 GB | **5607.8 tok/s** | **×4.57** |
| Mixtral-8x7B | 60% | 37.0 GB → 87.0 GB | **13347.5 tok/s** | **×1.75** |

> 显示剪枝显著减少内存占用并提升推理速度，尤其在大 batch 场景下增益更明显。

#### ✅ 在线剪枝吞吐（Online Clipping）
| 方法 | 0% | 20% | 40% | 60% |
|------|----|-----|-----|-----|
| 原生 Python 实现 | ×1.00 | ×0.94 | ×0.91 | ×0.85 |
| **定制 Kernel** | ×1.00 | **×1.12** | **×1.25** | **×1.47** |

> 定制 kernel 成功逆转了原生实现因调度开销导致的性能倒退，实现了真正的在线加速。

---

### 与基线方法的对比结果（Mixtral-8x7B @ 40% 剪枝）

| 方法 | Average Accuracy (%) |
|------|-----------------------|
| Base Model | 63.29 |
| NAEE [2024] | 48.71 |
| MoE-SVD [2025] | 50.00 |
| TD-MoE [2026] | 57.00 |
| **FlexMoE (Ours, SP-FT)** | **58.00** ✅ |

> FlexMoE 在相同剪枝强度下显著优于所有基线，且是唯一超过 58 分的方法。

---

### 消融实验结果

#### 🔍 重要性重排序的作用（Ablation on Channel Ranking）
| 方法 | 20% 剪枝 | 40% 剪枝 |
|------|--------|--------|
| 无重排序（Base Model） | 54.43 | 48.86 |
| 排序（5k samples） | **60.00** | **53.86** |

> 证明重要性感知的通道重排序对 prefix slicing 至关重要。

#### 🔍 动作学习结构的重要性（Structure Ablation）
| 掩码类型 | Mixtral @60% Avg |
|--------|------------------|
| Learned（真实学习） | **49.57** |
| Uniform（统一比例） | 45.14 |
| Global Shuffle | 44.14 |
| In-layer Shuffle | 46.00 |

> 表明 FlexMoE 学习到的是**跨层+跨专家的结构化策略**，而非简单全局压缩。

#### 🔍 单点微调 vs 跨预算微调（SP-FT vs CP-FT）
| 方法 | 平均准确率（Mixtral） | 训练复杂度 |
|------|----------------------|------------|
| Per-Budget FT（理想上限） | ~58–60 | 极高 |
| **SP-FT @40% + Transfer** | **接近上限** | 极低 |
| CP-FT（多预算联合训练） | 较低 | 高 |

> 图 4 显示：单点微调 + 转移的效果已非常接近逐预算微调，远优于 CP-FT。

---

## 4. 关键结论和发现

### 主要发现
1. **嵌套剪枝可行且高效**：通过对专家内部通道进行重要性排序，可以安全地执行前缀剪枝，构建嵌套子网络族。
2. **单次训练即可覆盖多预算**：通过渐进式剪枝压力控制，一次 action learning 可导出全谱系 action masks。
3. **中等预算微调最具泛化性**：在 40% 剪枝点进行单次 recovery fine-tuning，其恢复权重能有效迁移到更高和更低预算。
4. **高稀疏 MoE 更易压缩**：Qwen2 等具有更多专家和共享专家的模型，在 FlexMoE 下表现更优，说明更强的稀疏性提升了压缩容忍度。
5. **系统协同至关重要**：若无 kernel-level 优化，动态剪枝反而会因调度开销导致性能下降；而 co-design 可将其转化为实际收益。

### 方法的局限性
- 当前仅作用于 FFN 通道，未涉及 attention 或 embedding 层。
- 依赖预定义的离散 action set，连续剪枝尚未支持。
- 在线切换仍略慢于静态剪枝模型（约 10–20% 开销），尚未完全消除调度代价。
- 适用于 post-training 场景，未探索 pre-training 阶段的弹性设计。

### 未来工作方向
- 扩展至 **attention head、KV cache、embedding 层** 的联合剪枝。
- 支持 **连续预算调节** 与更细粒度的弹性控制。
- 探索 **动态请求感知的自动预算分配机制**（如根据输入长度、难度自适应调整）。
- 将 FlexMoE 思路应用于 **非 MoE 模型** 的通用结构化剪枝。
- 构建端到端的 **budget-adaptive serving system**，结合 SLO 自动选择最优子网络。

---

> 💡 **一句话总结**：  
> **FlexMoE 提供了一种“一次训练、多端部署”的 MoE 压缩新范式，通过重要性排序 + 每专家动作学习 + 单点恢复 + 系统协同，实现了高性能、高灵活性、高实用性的多预算 MoE 子网络生成方案。**

</details>

---

### 2. [EntMTP: Accelerating LLM Inference with Entropy Guided Multi Token Prediction](https://arxiv.org/abs/2606.27550)

**Authors**: Carrie Chen  
**Category**: cs.CL  
**Published**: 2026-06-29  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.27550v1  

#### Abstract
Multi-token prediction has been shown to increase data density during training, improve downstream text-generation quality, and serves as the defacto approach for self-speculative decoding. Existing foundation and open source models that use MTP heads commit to a static tree-based attention topology...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EntMTP: Accelerating LLM Inference with Entropy Guided Multi-Token Prediction

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **Multi-Token Prediction (MTP)** 方法（如 Hydra 和 Medusa）在推理过程中采用**静态的树形 attention 拓扑结构**，即在整个生成序列中保持固定的 speculation depth。这种设计忽略了自然语言中不同上下文区域的可预测性差异：
- **低熵区域**（如代码、公式）适合长距离多步推测；
- **高熵区域**（如开放对话）需要保守推测以避免频繁回退。

因此，静态拓扑导致计算资源浪费或接受率下降，无法最大化吞吐量。

### 🚀 提出的新方法：EntMTP
作者提出 **Entropy-guided Multi-Token Prediction (EntMTP)** ——一种无需训练的动态调度器，能够根据当前生成上下文的**局部熵估计**，实时切换最优的 draft tree topology。

#### 核心思想：
- **离线阶段**：为每个任务构建一个 **throughput-Pareto frontier** 的候选树集合（TopologyBank），仅保留提升成本-吞吐权衡的“前沿”树。
- **在线阶段**：每一步解码时，基于轻量级特征（如 EAGLE-2 path value 或 top-1 概率）选择最合适的 tree topology。

#### 支持三种调度策略：
- **ENTMTP\***：使用该任务下全局最优的固定树（task-specific optimal tree）；
- **ENTMTP-l**：阈值阶梯法，将得分划分为多个区间对应不同树；
- **ENTMTP+**：带滞后的二元切换策略（hysteretic binary rule），防止频繁抖动。

### 🔍 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **灵活性** | 动态匹配 speculation depth 与上下文可预测性，实现更高效的资源利用 |
| **无需训练** | 完全基于已有模型特征进行调度决策，不引入额外参数或微调 |
| **零开销切换** | 所有树的 attention mask、position offset 等预编译存储，切换仅为 O(1) 指针交换 |
| **兼容性强** | 可直接应用于 Hydra/Medusa 架构，作为插件式优化模块 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在四个典型基准上评估性能：
- **HumanEval**：代码补全任务，测试编程能力
- **GSM8K**：小学数学应用题，需多步推理
- **ShareGPT**：真实用户与 ChatGPT 的多轮对话记录，反映开放域交互
- （LitBench 在附录中提及，主实验聚焦前三者）

### ⚙️ 实验设置
- **基础模型**：Vicuna-7B v1.3
- **验证器**：ankner/hydra-vicuna-7b-v1.3（Hydra 架构）
- **硬件平台**：单张 NVIDIA A100 GPU，FP16 精度
- **生成参数**：
  - 温度 $ T = 0.7 $
  - 后验阈值 $ \epsilon = 0.09 $
  - 混合系数 $ \alpha = 0.3 $
  - 最大输入长度：1400 tokens
  - 最大生成长度：256 tokens
- **每项任务测试 100 条 prompt**（来自 val 集），排除 warm-up 轮次

### 📊 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **tok/s** | 实际输出 token 数 / 墙钟时间（含 prefill） | 衡量端到端加速效果 |
| **Speedup ratio ($\rho$)** | 相对于 vanilla autoregressive decoding 的速度提升倍数 | 主要性能比较依据 |
| **Average acceptance length ($\tau$)** | 每个 draft-verify cycle 平均接受的 token 数 | 反映 draft 质量与验证效率 |
| **Perplexity** | 续写文本相对于原模型的困惑度偏差 | 验证是否保持生成质量（lossless） |

### 🆚 基线方法
- **Vanilla Vicuna**：标准自回归解码（baseline）
- **Medusa (default)**：原始 Medusa 默认树结构
- **Hydra (default)**：Hydra 官方发布的默认树（mc_sim_7b_63，63 nodes）
- **ENTMTP\***：任务特定的 throughput-optimal 固定树
- **ENTMTP+**：动态调度版本（本文最佳配置）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Benchmark | 方法 | tok/s | Speedup | $\tau$ |
|---------|-------|--------|----------|--------|
| **HumanEval** | Vanilla | 38.2 | 1.00x | 1.00 |
|             | Medusa | 91.2 | 2.38x | 2.87 |
|             | Hydra | 109.0 | 2.85x | 3.06 |
|             | ENTMTP\* | 123.4 | **3.21x** | 3.28 |
|             | **ENTMTP+** | **124.7** | **3.26x** | 3.20 |
| **GSM8K**   | Vanilla | 33.7 | 1.00x | 1.00 |
|             | Medusa | 87.4 | 2.59x | 2.51 |
|             | Hydra | 102.4 | 2.87x | 2.86 |
|             | ENTMTP\* | 109.7 | 3.07x | 3.08 |
|             | **ENTMTP+** | **112.0** | **3.13x** | 3.02 |
| **ShareGPT**| Vanilla | 35.4 | 1.00x | 1.00 |
|             | Medusa | 94.6 | 2.72x | 2.86 |
|             | Hydra | 109.0 | 2.89x | 3.06 |
|             | ENTMTP\* | 116.9 | 3.42x | 2.97 |
|             | **ENTMTP+** | **117.5** | **3.47x** | 2.99 |

> ✅ **峰值加速达 3.47×**（ShareGPT 上相对于 vanilla AR），相比 Hydra 提升 **9.4%-14.0%**

### 🔁 与基线对比结果
- **vs Hydra default**：
  - 平均提速 **+9.4% ~ +14.0%**
  - 使用更小的树（28/46/30 vs 63 nodes），降低 verifier 开销
  - 接受长度 $\tau$ 更高或相当
- **vs Medusa default**：
  - 提速 **+7.7% ~ +32.4%**
- **消融分析（Section 6.1）**：
  - 性能增益主要来自两部分：
    1. **任务定制化树选择**（占 7–13%）：通过 smaller tree 减少 verifier 成本
    2. **动态调度机制**（额外 +0.5–2.1%）：尤其在 GSM8K 上显著（+2.1%），因高熵推理步骤受益于保守树

### ✅ 生成质量保障
- 所有方法的 continuation perplexity 与 base LM 差距 < 0.02 nats
- 表明 EntMTP 是 **lossless acceleration**，未牺牲生成一致性

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **静态 speculation depth 不适应多样化任务分布**  
   不同任务甚至同一任务内部的 acceptance behavior 存在显著差异（见 Fig. 3），单一 topology 无法全局最优。

2. **任务特定的 Pareto-optimal tree 显著优于通用默认树**  
   即使是固定树（ENTMTP\*），也能通过离线优化获得明显收益。

3. **动态调度进一步释放潜力**  
   利用简单的熵感知信号（如 path value）即可实现高效树切换，在高不确定性区域节省计算，在高确定性区域大胆推测。

4. **调度开销极低**  
   树切换为指针操作，s 计算仅需 <0.1ms，几乎无运行时负担。

### ⚠️ 方法的局限性
- 当前调度依赖手工设计的启发式规则（如 threshold ladder），尚未探索 learnable policy。
- 仅适用于支持 tree-based MTP 的架构（如 Hydra/Medusa），对纯 autoregressive 或其他 speculative decoding 框架不直接适用。
- 所有树仍需预先构建并存入 TopologyBank，内存占用随候选树数量线性增长（但实际仅保留 Pareto frontier，规模可控）。

### 🔮 未来工作方向
- 将 EntMTP 思想扩展至 **动态生成树结构**（类似 EAGLE-2），而非从有限集合中选择。
- 探索 **multi-modal 或跨任务泛化调度器**，减少 per-task calibration 成本。
- 结合 **runtime profiling feedback** 实现闭环自适应调度。

---

## ✅ 总结一句话
> **EntMTP 提出了一种无需训练、基于熵感知的动态 multi-token prediction 调度机制，通过按需切换最优 draft tree topology，在保持生成质量的前提下，实现了高达 3.47× 的推理加速，显著超越 Hydra 和 Medusa 等静态方案。**

</details>

---

### 3. [JD Oxygen AI Item Center (Oxygen AIIC) V1: An Industrial-Scale LLM/VLM-Centric Solution for Item Understanding, Management, and Applications](https://arxiv.org/abs/2606.28070)

**Authors**: Oxygen AIIC, Chan Long, Chao Liu, Chaofan Chen, Chaohui Dong, Chunyuan Guo, Danping Liu, Debin Liu, Deping Xiang, Fulai Xu, Guangyue Liu, Hao Li, Huichun Hu, Jian Yang, Jianan Wang, Jianbo Zhao, Jiaoyang Li, Jiaxing Wang, Jinglong Li, Jinjin Guo, Jun Fang, Jun Liu, Kai Zhou, Li Wang, Lili Gao, Liying Chen, Luning Yang, Mengdi Zhou, Pengzhang Liu, Qi Lv, Qianyun Wang, Qixia Jiang, Ruyue Li, Shimu Liang, Shuxing Wang, Sijie Zhang, Siqi Li, Tianhao Gao, Wang Ke, Weihu Huang, Wencan Lai, Wenjie Zhang, Xiaohui Zhang, Xiaojing Dong, Ya Liu, Yifeng Zhang, Yixiang Wang, Yongtai Zhang, Yongyi Liao, Zhaoru Chen, Zhen Chen, Zhiyong Ma, Zhiyuan Liu, Zhongwei Liu, Ziyan Xing  
**Category**: cs.AI  
**Published**: 2026-06-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.28070v1  

#### Abstract
JD.com, one of the world's largest e-commerce platforms, serves over 700 million active users and millions of merchants, with a catalog of tens of billions of SKUs. At this scale, high-quality, structured item knowledge underpins a better consumer experience, lower management costs, and higher opera...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对大规模电商平台（如京东）在**商品理解、管理与应用**中面临的三大工业级挑战：
- **快速涌现的新概念**：市场趋势变化快，新属性、品类不断出现，传统静态本体难以跟上。
- **海量SKU的知识生产成本高**：数十亿级商品需要高质量、结构化的知识提取，人工标注不可持续。
- **下游业务场景需求多样化**：搜索、推荐、运营等不同场景对知识的**格式、时效性和覆盖度**要求各异。

传统基于BERT等小模型的方法存在泛化能力弱、依赖大量标注、无法适应动态本体等问题。

---

### 提出的新方法与架构
作者提出了 **JD Oxygen AI Item Center (Oxygen AIIC)**，一个以 **LLM/VLM 为核心**的工业级商品知识基础设施，其四大核心支柱为：

#### （1）高效人机协同的本体工程（Ontology Engineering）
- **Top-down + Bottom-up 构建模式**：
  - **专家定义主干**：领域专家定义类目体系、核心属性及场景标签。
  - **LLM驱动增长**：通过“知识发现 → 融合 → 验证”流程，从商品信息、用户query、网页内容中自动挖掘潜在概念。
- 创新点：引入 **Multi-LLM 协同验证框架** 和 **基于波士顿矩阵的重要性评估机制**，确保新增概念既准确又具备商业价值。

#### （2）“语义检索再判别”（S²D: Semantic Search then Discrimination）架构
- 将传统的端到端属性抽取任务拆解为两阶段：
  1. **Semantic Search**：将商品与动态演进的本体条目映射到统一向量空间，召回Top-K相关候选值。
  2. **Discrimination**：由轻量级LLM判断商品是否匹配这些候选值。
- **优势**：
  - 解耦本体与模型参数，支持零样本扩展新属性。
  - 显著降低幻觉风险，提升泛化能力和可控性。

#### （3）自演进的商品理解 LLM/VLM 框架
- 建立统一的多任务基础模型，支持：
  - **增量学习（Incremental Adaptation）**：采用改进的 **LoRAM 初始化策略** 加速收敛。
  - **自适应专家组合（GROLE）**：动态融合多个LoRA专家模块，实现跨域知识共享。
  - **指令跟随表示学习（Instruction-following Representation）**：通过隐式思维链蒸馏增强细粒度语义对齐。
  - **模型自进化（Model Self-evolution）**：闭环迭代框架，通过在线反馈识别bad case，合成修复数据，持续优化模型。

#### （4）统一的数据与服务枢纽——Item Tunnel
- 提供**分层时效性服务**（秒级、分钟级、天级）满足不同业务需求。
- 支持**最终一致性**保障，解决流批混合场景下的状态同步问题。
- 统一服务接口，避免重复开发，提升SLA稳定性。

---

### 相比现有方法的优势
| 维度 | 传统方法 | Oxygen AIIC |
|------|----------|-------------|
| **本体更新** | 手工维护，滞后严重 | 自动发现+专家验证，敏捷扩展 |
| **知识抽取** | 封闭分类/生成式抽取，难泛化 | S²D架构，支持开放域、可解释性强 |
| **模型演化** | 全量重训，成本高 | 增量适配+自进化，稳定可控 |
| **系统吞吐** | 低效串行处理 | 计算负载削减+缓存复用+异步并行，吞吐提升 >10× |
| **部署灵活性** | 场景定制化强 | 统一隧道+分级服务，灵活复用 |

---

## 2. 核心实验方法和设置

### 数据集
- **真实工业数据**：来自京东平台的真实商品数据，涵盖：
  - 数十亿级 **SKU**
  - 上万个 **类目**
  - 数百万条 **本体条目**
  - 多源输入：商品标题、详情页文本、主图、用户query、外部网页内容等。
- **训练数据构建方式**：
  - 使用强LLM进行 **OpenIE** 和 **Targeted Attribute Filling** 自动生成监督信号。
  - 构造正负样本用于表示学习与判别任务。

---

### 实验设置与评估指标

#### 主要任务
- **端到端商品知识抽取**：从非结构化商品信息中识别并标准化输出 `(商品, 属性键, 属性值)` 三元组。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Precision** | 抽取结果中正确的比例 |
| **Recall** | 应被抽取出的知识中实际被覆盖的比例 |
| **Throughput Efficiency** | 单位计算资源下每秒处理的 SKU × Attribute 对数量 |
| **Freshness** | 知识更新延迟（秒 / 分钟 / 天） |
| **自动化填充率** | 商品发布时核心属性自动填充的比例 |

#### 基线方法对比
文中虽未列出具体第三方模型名称，但隐含对比了以下几类典型方法：
- **Closed-set Classification**：固定标签分类模型，无法应对新属性。
- **End-to-End Generation**：直接生成属性值，易产生幻觉。
- **RAG-based Extraction**：检索增强生成，推理开销大，控制性差。
- **传统NLP流水线**：NER + 规则映射，维护成本高，覆盖率低。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **知识抽取精度（Precision）** | **94.2%** |
| **知识抽取召回率（Recall）** | **82.8%** |
| **AI生成知识资产总量** | **数百亿条** |
| **知识丰富度提升** | 达到原有水平的 **3.35×** |
| **单日处理商品更新量** | **数亿次** |
| **吞吐效率提升** | 相比原始方案 **>10倍**（在华为Ascend NPU上） |
| **搜索流量覆盖率** | **80.4%** |
| **商品信息质量问题下降** | 下降 **37%** |
| **核心属性自动填充率** | 超过 **80%** |
| **创意优化CTR提升** | 平均提升 **9%** |
| **类目规划决策周期** | 从**数周缩短至数天** |

---

### 与基线方法的对比结果
- 在相同测试集上，S²D架构相比传统生成式方法：
  - Precision 提升约 **12–15个百分点**
  - Recall 更稳定，在长尾属性上表现更优
  - 推理成本降低 **60%以上**
- 自进化框架使模型在持续迭代中保持性能上升趋势，而传统SFT模型在多次更新后出现性能波动甚至退化。

---

### 消融实验结果（Ablation Study）
尽管未明确列出表格，但从描述中可提炼关键消融发现：

| 组件 | 移除后的性能影响 |
|------|------------------|
| **SKU级去重（Deduplication）** | 吞吐下降约 **30%** |
| **属性稀疏探测（Attribute-level probing）** | 计算量增加近 **3倍** |
| **前缀缓存复用（Prefix Cache Reuse）** | 吞吐下降 **6倍以上** |
| **异步流水线并行** | 整体等待延迟增加 **2倍以上** |
| **LoRAM初始化优化** | 收敛速度慢 **40%+**，最终精度略低 |
| **隐式CoT蒸馏** | 细粒度属性对齐准确率下降 **~8%** |

> 多项优化叠加后实现**超过10倍的吞吐提升**，证明各组件协同效应显著。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM/VLM 可作为工业级商品理解的核心引擎**，尤其在开放域、动态演进环境中展现出远超传统模型的泛化与推理能力。
2. **S²D 架构是平衡准确性、可控性与扩展性的有效范式**，特别适合将LLM应用于高精度结构化知识抽取任务。
3. **人机协同 + 自进化机制** 是维持长期系统可靠性的关键，能有效修复长尾缺陷，防止系统退化。
4. **统一的服务隧道（Item Tunnel）** 极大提升了知识资产的复用效率，支撑了跨业务场景的大规模落地。

---

### 方法的局限性
1. **本体关系建模尚不充分**：当前主要聚焦于扁平属性建模，缺乏对复杂语义关系（如“兼容”、“替代”、“搭配”）的深度建模。
2. **线上缺陷检测机制不足**：尽管有自进化，但仍缺乏实时感知线上bad case的能力，修复存在延迟。
3. **灾难性遗忘风险仍存**：虽然通过LoRAM缓解，但在频繁增量更新下仍可能削弱通用能力。
4. **多语言支持有限**：目前主要面向中文电商环境，国际化适配有待加强。

---

### 未来工作方向
1. **强化本体中的关系建模**：引入图神经网络与逻辑推理，构建更丰富的商品知识图谱。
2. **建立在线bad case发现机制**：结合用户行为反馈与异常检测，形成“消费→反馈→修复”的数据飞轮。
3. **探索更鲁棒的领域适配技术**：
   - 扩展MoE架构中的专家数量
   - 改进Prompt设计以显式注入领域知识
   - 简化任务复杂度，提升迁移稳定性
4. **推动全链路自治**：实现从知识生产、验证、服务到反馈的完全自动化闭环。

--- 

> **总体评价**：Oxygen AIIC 是首个完整披露的、基于LLM/VLM的大规模电商商品理解工业系统，不仅实现了技术突破，更展示了如何将前沿AI能力系统化地集成到超大规模生产环境中，具有极强的工程参考价值。

</details>

---

### 4. [Quantum Generative Diffusion Model for Real-World Time Series](https://arxiv.org/abs/2606.27561)

**Authors**: Jack Waller, Filippo Caruso, Dimitrios Makris, Rajagopal Nilavalan, Xing Liang  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.27561v1  

#### Abstract
Generative models have achieved remarkable success in data synthesis, though recent advances driven by increasing model scale have introduced challenges in computational cost and efficiency. Quantum machine learning offers a promising alternative, representing complex data distributions using compac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Quantum Generative Diffusion Model for Real-World Time Series*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的**生成模型**（如 GANs、VAEs 和 Diffusion Models）在时间序列生成任务中取得了进展，但其性能提升高度依赖于模型规模的扩大，导致**计算成本高、参数量大、能耗高**，限制了其在资源受限场景下的应用。

此外，尽管量子机器学习（QML）在图像等领域的生成任务中已有探索，但**尚未有研究将量子扩散模型（Quantum Diffusion Models, QDMs）应用于真实世界的时间序列数据**，尤其是具有非平稳性、重尾分布和复杂时序依赖性的金融时间序列。

### 🚀 提出的新方法与创新思路
本文提出了 **QDiffusion-TS** —— **首个面向现实时间序列生成的量子生成扩散模型**，并成功在真实量子硬件上验证。

#### 主要创新点包括：
- **首次将 QDM 应用于时间序列生成任务**，填补了该领域空白。
- 在经典 Diffusion-TS 架构基础上，**用 Quantum Neural Networks (QNNs) 完全替代 Transformer 中的 Feed-Forward Neural Networks (FFNNs)**，构建了一个混合量子 Transformer 架构。
- 利用 QNN 的高表达能力，在**显著减少可训练参数的前提下实现更优或相当的生成质量**。
- 验证了该模型在**真实量子处理器 IQM Emerald 上的可行性与鲁棒性**，甚至观察到硬件噪声可能带来轻微性能增益。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **参数效率** | 每个被替换的 FFNN 组件从 33,088 参数降至仅 36 参数，**减少近三个数量级**；整体模型参数由 589,030 减少至 461,254（约 ↓20%）。 |
| **生成质量** | 合成数据更准确地还原真实分布，**Wasserstein 距离平均降低 44%**（Amazon 数据集达 51%）。 |
| **下游任务表现** | 用于数据增强后，预测任务 RMSE 最多改善 **71%**，且优于直接扩展真实数据集的效果。 |
| **硬件兼容性** | 成功部署于 IQM 量子设备，**硬件执行结果略优于模拟器**，表明对 NISQ 设备噪声具有鲁棒性甚至受益。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **Apple 和 Amazon 的日频股票价格数据**（来自 Yahoo Finance）
- 时间范围：2021-02-12 至 2026-02-12
- 特征：Open, High, Low, Close, Volume
- 输入形式：log-return $ r_t = \ln(y_t / y_{t-1}) $
- 序列长度：默认 256；量子硬件实验中缩短为 32（受编码容量限制）

> （补充数据集中还包括 Microsoft 和 Google，见附录）

### ⚙️ 实验设置
- **基础架构**：基于 Diffusion-TS [58]，采用 Encoder-Decoder Transformer 结构进行去噪。
- **QNN 集成方式**：
  - 替换所有 FFNN 层为 QNN；
  - 使用 PCA 将输入特征压缩至适配 QNN 的维度；
  - 采用 amplitude encoding（模拟）或 angle encoding（硬件）进行数据编码；
  - 输出通过测量 Pauli-Z 期望值重构为经典向量。
- **训练策略**：
  - 正向过程：标准高斯扩散；
  - 反向过程：QNN 预测原始样本 $ \hat{x}_0(x_t, t, \theta) $；
  - 损失函数结合重建误差与 Fourier Loss，以保留频率结构。
- **硬件平台**：IQM Emerald 量子处理器（via Amazon Braket）
- **仅在推理阶段使用真实量子硬件**，训练仍通过模拟器完成（due to NISQ 限制）

### 📏 评估指标
| 类型 | 指标 | 描述 |
|------|------|------|
| **分布相似性** | Wasserstein Distance, KS Statistic | 衡量合成与真实 log-return 分布之间的距离 |
| **矩匹配度** | MAE of Moments (Mean, Variance, Skewness, Kurtosis) | 评估是否复现金融“stylised facts”（如重尾、波动聚集） |
| **下游任务性能** | RMSE, R² Score | BiLSTM 预测收盘价的表现，衡量生成数据实用性 |
| **参数效率** | Trainable Parameters Count | 对比模型复杂度 |

### 🆚 基线方法对比
- **Classical Diffusion-TS (FFNN)**：原始全经典版本作为主要对照组
- **Baseline (Real Data Only)**：不进行数据增强的预测模型
- **Real Data Expansion**：用更多历史真实数据扩充训练集（用于对比合成数据 vs 真实数据扩展效果）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 指标 | QDiffusion-TS (QNN) | Classical Diffusion-TS (FFNN) | 改进幅度 |
|------|---------------------|-------------------------------|----------|
| **总参数量** | 461,254 | 589,030 | ↓ ~20% |
| **单个 FFNN/QNN 参数数** | 36 | 33,088 | ↓ ~99.9% |
| **Wasserstein Distance (avg.)** | 显著更低 | 较高 | ↓ ~44% |
| - Apple 子集 | — | — | ↓ ~36% |
| - Amazon 子集 | — | — | ↓ ~51% |
| **MAE of Variance** | ↓ 63% 更准确 | 较高偏差 | 明显优势 |
| **Downstream Forecasting (RMSE)** | 最多 ↓71% | 基线水平 | 显著提升 |
| **Peak R² Performance** | ≈ Classical Generated | ≈ Quantum Generated | 相当 |
| **Hardware vs Simulator** | Wasserstein ↓8% 更优 | — | 硬件轻微胜出 |

### 🔁 与基线方法的对比结果
- **统计保真度方面**：
  - QDiffusion-TS 在大多数 central moments（尤其 variance 和 mean）上 MAE 更低；
  - 虽然 classical 模型在 skewness/kurtosis 上略有优势，但两者均能有效捕捉重尾特性；
  - 所有特征上的 Wasserstein Distance 和 KS Statistic 均系统性低于 classical 模型。
- **小样本训练分析**（Fig. 4）：
  - 在极小训练集下，两种模型性能接近；
  - 随着训练数据增加，QDiffusion-TS 持续保持更优统计距离；
  - **未显示出明显的“低数据优势”**，说明性能提升主要源于更强的分布建模能力而非样本效率。
- **下游预测任务**（Fig. 5）：
  - 添加合成数据后，RMSE 最多下降 **71%**，R² 显著上升；
  - 性能在 **1:5 增强比例**达到饱和；
  - **QNN 与 classical 生成数据在峰值预测精度上相当**，但 QNN 在低增强比时表现更好；
  - **扩展真实数据反而导致性能下降**，归因于金融时间序列的 non-stationarity 引发分布偏移。

### 🔬 消融实验与硬件验证（关键发现）
- **硬件执行验证**（Sec. 2.6 & Fig. 6）：
  - 在 IQM Emerald 上运行 QDiffusion-TS 推理；
  - 硬件版（QNNHard）与模拟版（QNNSim）性能几乎一致；
  - **硬件版 Wasserstein Distance 相较 classical 模型平均降低 89%**；
  - **甚至比模拟版再低约 8%**，提示硬件噪声可能引入有益随机性，增强多样性。
- **参数缩减分析**：
  - 单组件参数减少 3 个数量级；
  - 整体参数减少 20%，却实现更优生成效果 → **证明 QNN 具备极高参数效率**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **QDiffusion-TS 是首个成功的量子扩散模型用于现实时间序列生成的应用**，验证了其在金融数据中的有效性。
2. **QNN 可高效嵌入现代深度架构**（如 Transformer），在大幅削减参数的同时维持甚至超越经典模型性能。
3. **生成数据具备高质量统计保真度**，能准确复现 log-return 分布、方差结构及 stylised facts。
4. **合成数据可用于有效数据增强**，显著提升下游预测任务性能（↑71% in RMSE），且优于简单扩展真实数据。
5. **真实量子硬件执行不仅可行，且性能媲美甚至略超模拟器**，表明当前 NISQ 设备已足以支持此类混合生成任务。
6. **硬件噪声未损害性能，反而可能有益**，暗示噪声可在某些生成场景中起到正则化或多样化作用。

### ⚠️ 方法的局限性
- **仅在推理阶段使用量子硬件**，训练仍在经典模拟器完成 → 受限于当前量子硬件训练效率。
- **序列长度受限**（硬件实验中仅支持 32 步），难以处理长程依赖。
- **QNN 设计依赖特定编码方案与 ansatz**，泛化性和架构搜索空间有待探索。
- 当前优势集中在参数效率和分布拟合，**尚未展示明确的“量子优越性”（quantum advantage）**，尤其是在训练速度或样本复杂度方面。

### 🔮 未来工作方向
- 探索端到端的量子训练流程（on-device training）；
- 设计适用于更长时间序列的量子注意力机制；
- 开展跨领域测试（如医疗、气象、工业传感时间序列）；
- 系统研究硬件噪声在生成任务中的作用机制，发展“噪声利用”策略；
- 构建更大规模的混合量子-经典生成框架，逐步扩大 QNN 在架构中的占比。

---

> 💡 **一句话总结**：  
> QDiffusion-TS 成功将 QNN 融入扩散 Transformer 架构，实现了**参数极少、分布保真度更高、下游任务有效的金融时间序列生成**，并在真实量子硬件上验证了其可行性与潜力，为未来高效、可扩展的量子生成模型提供了实用路径。

</details>

---

### 5. [Mechanism-Driven Monitors for Preemptive Detection of LLM Training Instability](https://arxiv.org/abs/2606.28116)

**Authors**: Ruixuan Huang, Yipei Wang, Wenyi Fang, Hantao Huang, Yifan Huang, Ansheng You, Zhenxing Zhang, Shuai Wang, Fan Wu, Yang Zheng  
**Category**: cs.CL  
**Published**: 2026-06-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.28116v1  

#### Abstract
Frontier large language model training consumes massive accelerator fleets and long wall-clock computation, making stability failures costly when they occur. After a numerical or a hyperparameter fault has already destabilized the training dynamics, it may continue for thousands of steps while loss ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Mechanism-Driven Monitors for Preemptive Detection of LLM Training Instability**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

大型语言模型（LLM）训练过程极其昂贵，涉及数千个加速器运行数周甚至数月。一旦发生**训练不稳定性**（如 loss spike 或 divergence），往往意味着大量计算资源的浪费。传统监控依赖于全局指标（如 loss、gradient norm），但这些信号出现时，模型参数或优化器状态可能已受到不可逆损害。

本文指出，**在故障显现前，系统早已进入不稳定状态**，而当前监控手段滞后严重。因此，亟需一种能够**提前数千步检测到训练失稳根源**的机制。

---

### 🚀 提出了什么新方法或新思路

提出 **“机制驱动的监控”（mechanism-driven monitoring）** 范式：

- 不再依赖症状级（symptom-level）的全局曲线（如 loss 上升），
- 而是从每个关键模块的**功能角色**出发，分析其**故障机制**，并据此设计能最早捕捉异常的内部监测器。

#### 具体应用两个案例：

| 模块 | 监控目标 | 核心思想 |
|------|--------|---------|
| **Flash Attention (FA)** | 权重更新 `△W` 的谱几何 | 低精度 FA 在反向传播中引入有偏舍入误差，导致 `△W` 出现低秩漂移；通过监测 `△W` 的奇异值谱熵可提前发现 |
| **MoE Router** | 路由行为多样性 | Router 应实现判别性专家选择；若权重趋于冗余或路由分布坍缩，则系统不稳；提出基于 `router weight similarity` 和 `per-token routing entropy` 的监控 |

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | 本工作 |
|------|--------|-------|
| **响应时机** | 滞后（loss 已上升） | **提前数千步预警**（如 `△W` 谱坍缩早于 loss divergence 约 9,000 步） |
| **归因能力** | 难以定位故障源 | **模块级可解释性**：不同故障触发不同监控器，互不交叉污染 |
| **计算开销** | 如 Hessian 分析太贵 | 所提指标均可高效计算（如利用 bilinear decomposition 避免全矩阵乘法） |
| **适用性** | 多为通用指标（如 max-logit） | **机制定制化**：针对特定模块设计专属监控器 |

> 💡 **核心创新**：将“可解释性”从 post-hoc 分析推进到 **preemptive monitoring**，构建面向机理的可观测性体系。

---

## 2. **核心实验方法和设置**

### 📚 数据集与模型

- 使用自研 LLM 进行训练实验（未公开具体预训练语料，但上下文符合前沿 LLM 规模）
- 模型规模覆盖数十亿至百亿参数级别
- 包含标准 Dense Transformer 和 MoE 架构变体

---

### ⚙️ 实验设置

#### 故障注入实验（Fault Injection）

| 故障类型 | 注入方式 |
|--------|--------|
| **Low-precision FA Fault** | 对 BF16 张量进行低位截断（bit-shift），模拟低精度舍入偏差 |
| **Large Learning Rate / Small GBS** | 调整超参数组合，诱发 MoE router entropy collapse |
| **Combined Faults** | 同时注入注意力与路由侧故障，测试监控器分离能力 |

#### 监控指标设计

| 模块 | 监控变量 | 计算方式 |
|------|--------|--------|
| **Attention Update** | `△W` 的奇异值谱熵（singular spectrum entropy） | 基于 `△W = W_t - W_{t−δ}` 的 SVD |
| | `△1`, `△2`, `△3`（QK-product increment 分解） | 利用 bilinear decomposition 提取一阶主导项 `△2` |
| **MoE Router** | `router weight similarity`: `sim(WR)` | `E[cosine_similarity(w_i, w_j)]` |
| | `per-token routing entropy`: `H(p(x))` | `−∑ p(x) log p(x)`，反映路由分布均匀性 |

#### 基线对比方法

- **Symptom-level baselines**:
  - Loss 曲线
  - Gradient norm
  - Weight norm (`||W||_F`)
  - Stable rank of `W`
- **其他诊断工具**:
  - Max-logit（难以在生产 FA 中获取）
  - Hessian/curvature 分析（计算代价过高）

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据

| 监控信号 | 故障响应时间（step） | 距 loss divergence 提前量 |
|--------|------------------|----------------------|
| **Loss divergence** | ~22,000 | — |
| **Gradient norm** | ~21,000 | ~1,000 步 |
| **Weight norm `||W||_F`** | 无明显变化 | — |
| **`△W` singular spectrum entropy** | ~13,000 | **~9,000 步** |
| **`△2` (first-order QK increment)** | ~5,000 | **~17,000 步** |
| **Router per-token entropy** | 在 FA 故障下不变 | — |

> ✅ `△2` 是目前观测中最先响应该类故障的指标。

---

### 🔁 与基线方法的对比结果

| 指标 | 是否提前预警 | 可归因性 | 噪声鲁棒性 |
|------|------------|----------|-----------|
| Loss | ❌ 最晚出现 | 低 | 高 |
| Gradient norm | ⭕ 接近崩溃点 | 低 | 中 |
| Weight norm | ❌ 几乎无信号 | 低 | 高 |
| `△W` spectrum | ✅ 提前约 9k 步 | 高（仅 FA 故障触发） | 高 |
| `△2` spectrum | ✅ 提前约 17k 步 | 高 | 高 |
| Router entropy | ✅ 对 LR/GBS 敏感 | 高（仅路由相关故障触发） | 高 |

> ✅ **双模块监控器互不干扰**：  
> - 低精度 FA 故障 → `△W` 谱坍缩，但 router entropy 正常  
> - 高学习率 → router entropy 下降，但 `△W` 谱正常  

---

### 🔍 消融实验结果

#### （1）`△W` vs `W` 监控效果对比

- `W` 的稳定秩（stable rank）和范数在整个训练过程中几乎不变 → **初始化能量主导，信噪比极低**
- `△W` 移除了初始化背景，直接暴露更新几何结构 → **对低秩漂移高度敏感**

#### （2）`△1`, `△2`, `△3` 分解有效性验证

- `△1`（完整 QK-product 增量）与 `△2`（一阶项）几乎同步响应故障
- `△3`（二阶交互项）响应较慢，但仍能揭示 Q/K 更新耦合模式
- 利用 head-dimensional core 计算 `△2` 的奇异谱，在 d ≫ dk 时提速达 **50 倍以上**（见 Table 1）

#### （3）Router conditioning ratio 与 entropy 关系

- 学习率越大、batch size 越小 → **router entropy 下降越快**
- 辅助负载均衡损失（auxiliary loss）移除后，entropy 下降更剧烈
- 支持 “stable-winner reinforcement” 机制假设

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **训练不稳定性可在 loss 显现前数千步被检测到**：
   - 特别是 `△2`（QK-product 一阶增量）可提前约 **17,000 步** 发出预警。

2. **模块级机制决定最早可观测信号的位置**：
   - 不能靠统一监控器解决所有问题；
   - 必须理解模块的功能与失效路径，才能设计有效前置探测器。

3. **两类监控器具有正交响应特性**：
   - Attention 更新监控对数值精度错误敏感
   - MoE 路由监控对超参数配置敏感
   - 二者**不交叉响应**，支持故障归因

4. **低秩更新漂移是系统性现象而非偶然噪声**：
   - 低精度 FA 导致的 `△W` 低秩化可通过 concentration argument 解释（Observation 1）

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **Attention 变体适配性** | 当前 `△2` 分解基于标准 MHA；对于 GQA/MQA/MLA 等结构需重新推导 |
| **FP8 / Stochastic Rounding 扩展不足** | 当前验证集中在 BF16 bit-shift，尚未覆盖更复杂的低精度训练场景 |
| **forward-error closure 维度不匹配** | QK 算子空间 D×D 与 head-space dk×dk 存在维度鸿沟，尚未完全建模 |
| **router indicator 对激活各向异性敏感** | 当前 `||WR,c||` 与 entropy 的联系依赖 `M_xx ≈ I` 假设，实际中可能偏离 |

---

### 🔮 未来工作方向

1. **扩展至更多新兴模块**：
   - 如 Engram（持久记忆）、manifold-constrained residual connections、learnable sparsity mechanisms 等，均需机制定制化监控。

2. **建立模块化监控框架（Monitoring-as-a-Service）**：
   - 将机制驱动的设计流程模板化，支持快速为新架构生成专用监控器。

3. **结合动态阈值与在线检测算法**：
   - 当前依赖人工设定阈值；未来可引入 spiked random-matrix theory 实现自动 onset detection。

4. **跨模块故障传播建模**：
   - 当前关注单点故障；下一步研究多模块耦合失稳的级联效应。

---

> 🧩 **最终洞见**：  
> “**不是 loss 决定训练是否健康，而是内部机制决定了 loss 是否即将失控。**”  
> 本文推动了 LLM 训练从“看曲线调参”迈向“基于机理的可观测性工程”。

</details>

---

### 6. [Yuvion LLM: An Adversarially-Aware Large Language Model for Content And AI Safety](https://arxiv.org/abs/2606.27632)

**Authors**: Ting Ma, Xiufeng Huang, Benlei Cui, Xiaowen Xu, Shikai Qiu, Ruijie Jian, Hongxing Li, Guanghui Wang, Longtao Huang, Haiwen Hong, Haolei Xu, Wenjing Jiang, Ziwen Xu, Zhaoyu Fan, Shaoxuan He, Chuxi Xiao, Yujian Li, Xinyue Chen, Chunyang Chai, Wenxuan Liu, Ziheng Wang, Dongjie Zhang, Yangfan Zhou, Libin Dong, Yupeng Cao, Xiaoqian Xia, Jing Wang, Zhe Jiang, Zhenan Ye, Guang Yang, Bin Liu, Wei Peng, Ziqiang Zhu, Meihui Lian, Kaiwen Lv Kacuila, Haidong Ding, Bingyu Zhu, Yan Wang, Hai Zhao, Xuan Jin, Wei Zhao, Pengfei Sun, Wei Wang, Huiming Zhang, Bin Li, Hui Xue  
**Category**: cs.CL  
**Published**: 2026-06-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.27632v1  

#### Abstract
As large language models are increasingly deployed in real-world systems, safety failures can still lead to harmful outputs and dangerous misuse. We argue that the essence of safety is adversarial: many failures arise not from natural inputs alone, but from strategic attempts to evade model policies...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Yuvion LLM: An Adversarially-Aware Large Language Model for Content And AI Safety 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前的 **Large Language Models (LLMs)** 在部署于真实内容安全和AI安全场景时，面临两大核心挑战：

1. **对抗脆弱性（Adversarial Fragility）**：模型在面对经过伪装、委婉表达、跨语言混合、符号替换等策略性规避手段时，容易失效。例如，将“燃烧瓶”改为“大范围燃烧效果”，或将“冰毒”写成“6”，即可绕过多数通用模型的安全检测。

2. **代理能力缺失（Lack of Agentic Capability）**：现有安全模型多为单轮分类器，无法支持多步推理、工具调用（如图像识别）、政策检索、证据合成等复杂审计流程，难以满足工业级内容审核的实际需求。

这些问题导致在公开基准上表现良好的模型，在真实部署中鲁棒性被高估。

---

### 提出了什么新方法或新思路

Yuvion LLM 提出了一套**以对抗感知为核心、面向实际部署的安全专用训练范式**，其核心创新如下：

#### （1）**对抗感知的数据系统（Adversarially-Aware Data System）**

构建了五类协同作用的数据：
- **General data**：维持基础语言能力
- **Safety-domain data**：注入安全领域知识（如政策、风险分类）
- **Adversarial data**：显式建模规避策略（同音字、符号替换、角色扮演等）
- **Agentic data**：支持多步推理与工具交互轨迹
- **Synthetic & expert-constructed data**：覆盖长尾、复杂场景

#### （2）**渐进式安全训练范式（Progressive Safety Training Paradigm）**

分为三个阶段：
1. **Knowledge-Enhanced Continued Pretraining**：通过知识增强的持续预训练，内化安全领域知识。
2. **Policy-Grounded Multi-Task Safety Post-Training**：基于监督微调（SFT）和强化学习（GRPO），实现细粒度风险识别与策略一致决策。
3. **Safety-Aware Agentic Reinforcement Learning**：引入工具调用（Tool Use）和搜索增强推理（Search-Augmented Reasoning），提升在复杂安全任务中的轨迹级可靠性。

#### （3）**Yuvion LLM RiskEval (YLRE) 评估框架**

提出一个四层递进式评估体系，涵盖：
- **Level 1**: 开源通用能力（General Benchmarks）
- **Level 2**: 开源内容安全（Content Safety Benchmarks）
- **Level 3**: 自建对抗鲁棒性基准（Adversarial Safety）
- **Level 4**: 工业部署能力（Industrial Deployment）

该框架强调从“静态分类”到“动态对抗”再到“真实业务流”的全栈评估。

---

### 相比现有方法的优势

| 维度 | Yuvion LLM | 传统方法 |
|------|-----------|--------|
| **训练目标** | 安全是第一优先级，对抗鲁棒性与代理能力内生设计 | 事后添加安全补丁（post-hoc safeguards） |
| **评估方式** | 四层递进式评估，覆盖真实对抗与业务流程 | 多依赖静态、非对抗性公开基准 |
| **能力范围** | 支持多步推理、工具调用、策略一致性决策 | 多为单轮分类或拒绝响应 |
| **规模效率** | 小模型（8B）可超越更大通用模型（如 GPT-5.4） | 依赖模型规模提升安全性 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

#### （1）**开源通用基准（Level 1）**
- **General Benchmarks**：MMLU, MMLU-Redux, GPQA, ARC-Challenge, C-Eval, CMMLU, GSM8K-ZH, BBH 等（共 >30 个）
- **Agentic Benchmarks**：API-Bank, BFCL, Seal-0（测试工具调用与搜索能力）

#### （2）**开源内容安全基准（Level 2）**
- **Content Safety**：ChineseHarm, COLD, Moderation, HateXplain, ToxiGen, Jigsaw, CivilComments, SafetyBench（8个）
- **Guard Benchmarks**：SEval, AEGIS 等（>20子集），涵盖提示分类、响应分类、攻击防御等

#### （3）**自建对抗鲁棒性基准（Level 3）**
- 覆盖五大风险类别：
  - Advertising & Traffic Diversion
  - Gambling & Fraud
  - Abusive Content
  - Pornographic Content
  - Spam & Flooding
- 分为 **Static**（标准表达） 和 **Dynamic**（对抗性改写）两部分
- 动态样本通过自动化红队攻击（automated red-teaming）生成

#### （4）**内部能力与业务基准（Level 4）**
- **Capability Benchmarks**：Political Risk, Prohibited Content, Insult, Low-Info Text, Emotion Analysis 等
- **Business Benchmarks**：UGC Moderation, AIGC Moderation, Business Porn Detection, Data Security NER 等
- 数据来自匿名化生产流量，反映真实审核流程

---

### 实验设置和评估指标

| 评估层级 | 主要指标 | 说明 |
|--------|--------|------|
| Level 1 | Accuracy | 通用任务准确率 |
| Level 2 | Macro F1-Score | 内容安全分类，缓解类别不平衡 |
| Level 3 (Dynamic) | Combined Score ↓ | = Bypass Success Rate × Semantic Fidelity，越低越好 |
| Level 4 | Task-specific Metrics | 如 F1、Attribution Accuracy、Workflow-level Indicators |

所有模型使用相同 prompt 模板与解码配置，确保公平比较。

---

### 基线方法对比

#### **开源模型**
- Qwen3 系列（8B, 32B, 397B-A17B）
- DeepSeek-R1/V3.2, Kimi-K2.5, GLM-5

#### **闭源前沿模型**
- GPT-5.4, Qwen3-Max, Qwen3.5-Plus, Qwen3.6-Plus

#### **专用安全模型**
- Qwen3Guard-8B, Llama-Guard4-12B, WildGuard-7B

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 模型 | Level 2: Content Safety (Macro F1) | Level 3: Adversarial Safety (Combined Score ↓) | Level 4: Industrial Deployment |
|------|----------------------------------|---------------------------------------------|-------------------------------|
| **Yuvion-32B** | **78.2%** | **20.6%** | **86.1%** |
| GPT-5.4 | 72.2% | 22.3% | 80.6% |
| Qwen3-MAX | 73.9% | 24.4% | 83.0% |
| **Yuvion-8B** | 73.3% | 28.1% | 82.7% |

> ✅ **Yuvion-32B 在所有维度均优于所有基线模型，包括远大于它的 GPT-5.4 和 Qwen3-MAX。**

---

### 与基线方法的对比结果

#### （1）**在内容安全（Level 2）上显著领先**
- Yuvion-32B 在 **ChineseHarm** 上达 **97.9% F1**，远超 GPT-5.4（85.3%）
- 即使是 **Yuvion-8B**，也超过多数 32B 及以上规模的通用模型

#### （2）**在对抗鲁棒性（Level 3）上优势明显**
- **Dynamic Benchmark** 中，Yuvion-32B 的 Combined Score 为 **20.6%**，低于 GPT-5.4（22.3%）和 Qwen3-Max（24.4%）
- 表明其对规避攻击（如“6”代指“溜冰”）具有更强识别能力

#### （3）**在工业部署（Level 4）中表现卓越**
- Yuvion-32B 在业务复合得分上达 **86.34**，远超 GPT-5.4（80.40）和 Qwen3-Max（83.00）
- **Yuvion-32B (Agent)** 进一步提升至 **87.34**，验证代理强化学习的有效性

#### （4）**专用安全模型（Guard Models）在真实业务中表现极差**
- Qwen3Guard-8B 和 Llama-Guard4-12B 在内部业务基准上得分为 **接近 0**
- 说明其仅适用于简单过滤，无法胜任复杂审核流程

---

### 消融实验结果

#### （1）**领域知识数据消融（Table 10）**
- 移除知识数据后，**Domain Composite** 下降 **3.96%**
- 在 **Insult** 和 **Meaningless** 等需语义理解的任务上下降超 10%
- 证明结构化知识注入对细粒度判断至关重要

#### （2）**代理强化学习消融（Table 11）**
| 训练阶段 | API-Bank | BFCL | Seal-0 |
|--------|--------|------|-------|
| SFT only | 83.75 | 45.07 | 19.82 |
| + Tool Use RL | 88.78 | 54.64 | 31.53 |
| + Search Agent RL | **90.45** | **66.16** | **40.54** |

> ✅ 代理强化学习显著提升工具调用与搜索能力，且不损害安全性能

---

## 4. 关键结论和发现

### 主要发现

1. **安全应被视为对抗性问题**：传统静态评估严重高估真实部署鲁棒性，必须主动建模规避策略。
2. **小模型可超越大模型**：Yuvion-8B 在多个安全任务上优于 GPT-5.4 和 Qwen3-MAX，表明**针对性训练比单纯扩大规模更有效**。
3. **代理能力是工业部署刚需**：真实审核需多步推理、工具调用、证据合成，Yuvion 的 agentic RL 显著提升此类任务表现。
4. **专用安全模型 ≠ 可部署模型**：现有 Guard Models 在复杂业务中几乎失效，需构建面向实际工作流的“安全智能体”。

---

### 方法的局限性

1. **对抗攻防是持续博弈**：尽管对抗训练提升了鲁棒性，但新型规避策略仍可能突破防线。
2. **多语言泛化待验证**：当前评估主要集中于中英文，跨文化、多语言安全场景尚未系统评估。
3. **通用能力略有下降**：相比基线模型，Yuvion 在部分通用任务上存在轻微性能差距，需进一步平衡专业化与通用性。

---

### 未来工作方向

1. **构建闭环对抗演进机制**：
   - 引入红队自博弈（red-team self-play）
   - 实时捕获线上规避行为并反馈训练
2. **扩展多语言安全能力**：
   - 构建多语言对抗基准
   - 支持跨文化敏感内容识别
3. **增强代理能力**：
   - 支持更复杂的多工具协作
   - 实现策略检索、证据归因、跨系统联动

---

> **总结**：Yuvion LLM 重新定义了安全大模型的开发范式——不再将其视为通用模型的附加模块，而是作为**原生对抗感知、具备代理能力的专用基础模型**。其实验结果表明，**针对性的训练设计可以超越规模红利**，为未来内容安全与AI治理提供了新的技术路径。

</details>

---

### 7. [Layerwise Progressive Freezing: A Training Scaffold for Depth-Scalable Binary Networks](https://arxiv.org/abs/2606.27759)

**Authors**: Evan Gibson Smith, Bashima Islam  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.27759v1  

#### Abstract
Training binary neural networks (BNNs) from scratch is dominated by the straight-through estimator (STE), whose forward/backward mismatch produces severe accuracy degradation as networks deepen. We study an orthogonal axis: when and where binarization is enforced during training. We introduce StoMPP...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Layerwise Progressive Freezing: A Training Scaffold for Depth-Scalable Binary Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **深度二值网络（BNN）训练中的“深度退化”问题**：传统基于 Straight-Through Estimator (STE) 的方法在深层网络中表现严重下降，准确率随深度增加而急剧降低。
- **STE 的前向/后向不一致性**：STE 在前向传播中使用不可导的 `sign` 函数，在反向传播中使用可导代理梯度，这种不匹配导致优化困难，尤其在深层模型中更为显著。
- **全局渐进量化失败**：已有渐进量化方法（如 INQ）在仅量化权重时有效，但在同时量化权重和激活（即全二值网络 BNN）时会因梯度阻塞（gradient blockade）而崩溃。

### **提出了什么新方法或新思路**
- **StoMPP (Stochastic Masked Partial Progressive Binarization)**：
  - 一种**层级渐进冻结**（layerwise progressive freezing）策略，从输入到输出逐层将连续权重和激活替换为硬二值形式。
  - 引入**随机部分掩码**（stochastic partial masks）和**软刷新机制**（soft refresh），避免过早固化参数配置。
  - 支持两种模式：
    - **STE-free StoMPP**：对已冻结项设梯度为 0，实现完全无 STE 的训练。
    - **StoMPP+STE**：仅对已冻结项应用 STE，保留学习信号。

### **相比现有方法的优势**
- **无需依赖 STE 即可稳定训练深层 BNN**，从根本上缓解了前向/后向不一致问题。
- **层级推进顺序至关重要**：前向（input → output）推进成功，反向（output → input）则崩溃，揭示了激活引起的梯度阻塞机制。
- **通用性强**：适用于 ResNet、MobileNetV2、BERT 等多种架构，且可与 Bi-Real Net 等专用二值架构兼容。
- **训练更鲁棒**：对调度形状（schedule shape）、刷新率（refresh rate）等超参数不敏感。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 任务 | 模型 |
|--------|------|------|
| **CIFAR-10 / CIFAR-100** | 图像分类 | ResNet-18/34/50, MobileNetV2 |
| **ImageNet (ILSVRC2012)** | 图像分类 | ResNet-18/34/50 |
| **SST-2** | 文本分类 | BERT-base |

### **实验设置和评估指标**
- **评估指标**：Top-1 准确率（accuracy）
- **量化设定**：
  - **BWN**（Binary-Weight Network）：仅权重二值化，激活保持浮点。
  - **BNN**（Binary Neural Network）：权重和激活均二值化。
- **精度策略**：首尾层和下采样层保持全精度，其余层按方法进行二值化。
- **训练协议**：
  - 使用标准 SGD（momentum=0.9），**恒定学习率（0.1）**，**无 weight decay**，以控制变量。
  - 所有方法共享相同 backbone、数据增强、batch size（256）、训练轮数等。
- **StoMPP 默认配置**：
  - 冻结调度：立方函数 $ p(T) = (T/T_{\text{total}})^3 $
  - 刷新率：$ r = 100 $（每步更新约 1% 参数）
  - 层级顺序：从前向后（input → output）

### **基线方法对比**
- **Vanilla STE**：BinaryConnect 风格的标准 STE 方法，全程使用 `sign` 前向 + identity surrogate 后向。
- **StoMPP (STE-free)**：无 STE，冻结参数梯度为 0。
- **StoMPP+STE**：仅对冻结参数使用 STE。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Top-1 Accuracy %）**

#### **表 1：ResNet-50 上 BNN 性能对比（vs Vanilla STE）**

| 方法 | CIFAR-10 | CIFAR-100 | ImageNet |
|------|----------|-----------|----------|
| Vanilla STE | 51.5 | 26.7 | 30.8 |
| **StoMPP (STE-free)** | **69.5 (+18.0)** | **40.2 (+13.5)** | **34.2 (+3.8)** |
| **StoMPP+STE** | **78.6 (+27.1)** | **46.5 (+19.8)** | **48.5 (+17.7)** |

> ✅ **增益随深度增长而扩大**，表明 StoMPP 特别适合深层网络。

#### **其他架构上的泛化能力**

| 架构 | 任务 | Vanilla STE | StoMPP+STE | 提升 |
|------|------|-------------|------------|------|
| MobileNetV2 | CIFAR-100 | 40.5 | 48.0 | **+7.5** |
| BERT-base | SST-2 | 50.9 (≈随机) | 82.8 | **+31.9** |

> 🚨 **STE 在 BERT 上几乎崩溃（接近随机猜测）**，而 StoMPP 成功训练出高性能二值模型。

---

### **消融实验结果**

#### **(1) 推进步序（Ordering Ablation）**
| 调度方式 | ResNet-18 (CIFAR-100) | ResNet-50 (CIFAR-100) |
|--------|------------------------|------------------------|
| **Layerwise (input→output)** | 53.8 | 40.0 |
| **Global** | 53.2 | 35.9 |
| **Reverse (output→input)** | 28.4 | 8.6 (**崩溃**) |

> 🔍 **结论**：前向推进是成功的必要条件；反向推进导致梯度阻塞，训练失败。  
> 💡 **根本原因**：一旦某层激活被冻结为 `sign(z)`，其上游梯度几乎为零，若早期层尚未训练完成，则无法回传信号。

#### **(2) 超参数敏感性（Schedule & Refresh）**
- **调度函数**（cosine, linear, quadratic, cubic, flipped quadratic）：在合理范围内性能稳定。
- **刷新率**（r ∈ [10, 1000]）：只要不过快或过慢，性能变化不大。
> ✅ **StoMPP 对超参数高度鲁棒**，优于全局调度下的敏感行为。

#### **(3) 训练动态分析**
- **锯齿状收敛（Sawtooth Dynamics）**：
  - 每当新层开始冻结时，准确率短暂下降，随后恢复。
  - 表明未冻结部分正在适应新的二值化环境。
- **BWN 中无此现象**：说明主要影响来自激活的逐步冻结。

#### **(4) 学习率敏感性**
| 学习率 | Vanilla STE | StoMPP |
|--------|-------------|--------|
| 0.1 | 48.9 | 53.8 |
| 0.05 | 46.8 | 54.2 (**最优**) |
> ✅ StoMPP 在更低学习率下表现更好，可能更适合精细调优。

---

## **4. 关键结论和发现**

### **主要发现**
1. **层级推进顺序决定成败**：
   - **前向推进（input → output）** 成功，保证下游始终存在可导路径。
   - **反向推进（output → input）** 导致梯度阻塞，训练崩溃。
   - **BWN 不敏感**：因为激活仍连续，梯度可流通。

2. **激活引起的梯度阻塞（Activation-induced Gradient Blockade）是核心机制**：
   - 一旦激活被硬二值化，其局部梯度为 0，切断上游学习信号。
   - 层级推进通过保留“未冻结后缀”来维持梯度通路。

3. **StoMPP 是一个通用训练支架（Training Scaffold）**：
   - 可独立运行（STE-free），也可与 STE 组合（StoMPP+STE）。
   - 改进了训练稳定性与最终性能，尤其在深层网络中优势明显。

4. **性能提升随深度增长而放大**：
   - 在 ResNet-50 上提升高达 **+27.1%（CIFAR-10）** 和 **+17.7%（ImageNet）**。
   - 在 BERT 上从近随机提升至 **82.8%**，显示巨大潜力。

---

### **方法的局限性**
- **引入额外超参数**：如冻结调度、刷新率、每层训练步数等，虽较鲁棒但仍需选择。
- **当前评估基于简化训练流程**：未使用 weight decay、学习率衰减、知识蒸馏等 SOTA 技巧，绝对性能低于最先进水平。
- **理论机制尚待深入验证**：梯度阻塞假说虽合理，但缺乏直接干预实验证明。

---

### **未来工作方向**
1. **设计直接干预实验** 验证梯度阻塞机制（例如人工插入/移除阻塞点）。
2. **集成到更强训练流程中**：结合 distillation、adaptive LR、weight decay 等，冲击 SOTA。
3. **扩展至更大规模模型**：如 ViT、LLM（如 BitNet）等，探索 1-bit 大模型的可行性。
4. **自动化调度设计**：学习最优冻结顺序与节奏，而非固定规则。

---

> ✅ **总结一句话**：  
> **StoMPP 揭示了“何时何地”进行二值化比“如何估计梯度”更重要，通过前向层级推进构建稳定的训练路径，显著提升了深度二值网络的可训练性与性能上限。**

</details>

---

### 8. [Fair Classification with Efficient and Post-hoc Controllable Fairness-Accuracy Trade-off](https://arxiv.org/abs/2606.28097)

**Authors**: Maaya Sakata, Kazuto Fukuchi  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.28097v1  

#### Abstract
Post-hoc controllability of fair machine learning models, the ability to control the trade-off between fairness and accuracy after training, is valuable for practical deployment. Existing post-processing methods provide such post-hoc controllability but often suffer from significant accuracy degrada...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fair Classification with Efficient and Post-hoc Controllable Fairness-Accuracy Trade-off

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**公平机器学习模型在部署后对 fairness-accuracy trade-off 进行灵活调整的需求**。现有方法面临两难困境：
- **Post-processing 方法**（如 FairBayes）虽然支持 **post-hoc controllability**（即训练后通过改变阈值等参数调节 trade-off），但通常导致显著的 accuracy 下降，trade-off 效率低下。
- **In-processing 方法**（如 EPO、FairBiNN）能实现高效的 trade-off，但缺乏 post-hoc controllability，每次调整 trade-off 参数都需要昂贵的重新训练。

### 提出的新方法：Guidance to Fairest-Boundary (GFB)
作者提出了一种名为 **Guidance to Fairest-Boundary (GFB)** 的新型公平分类算法，其核心思想是：
- **结合 representation learning 与 post-processing**：在训练阶段，通过一个梯度优化过程学习一种有效的特征表示（feature representation），使得这些特征在后续进行任何目标公平性水平的 post-processing 时，都能获得高效的 trade-off。
- **理论指导设计**：论文首先从理论上分析了影响 post-processed 分类器 trade-off 效率的关键因素，发现当特征（features）更集中在“最公平分类器”（fairest classifier）的决策边界附近时，trade-off 效率更高。这一发现直接指导了 GFB 的设计。

### 相比现有方法的优势
- **双重优势**：GFB 同时实现了 **post-hoc controllability** 和 **高 trade-off 效率**，兼具了 post-processing 和 in-processing 方法的优点。
- **无需重新训练**：与 in-processing 方法不同，GFB 只需训练一次模型，之后即可通过简单的 post-processing（调整阈值）来适应不同的公平性要求，避免了昂贵的 retraining 成本。
- **计算效率高**：推理时的调整成本极低（仅需 O(n log n) 的二分搜索），远低于重新训练整个模型的成本。

## 2. 核心实验方法和设置

### 使用的数据集
实验在四个真实世界数据集上进行，涵盖图像和表格数据：
- **图像数据集**：
  - **CelebA**：预测人脸吸引力（Target: `Attractive`），敏感属性为性别（A: `Gender`）。
  - **UTKFace**：预测年龄是否大于等于30岁（Target: `Age≥30`），敏感属性为性别（A: `Gender`）。
- **表格数据集**：
  - **Adult**：预测年收入是否超过5万美元（Target: `Income>50K`），敏感属性为性别（A: `Gender`）。
  - **COMPAS**：预测两年内是否会再犯（Target: `2-year recidivism`），敏感属性为种族（A: `Race`，分为高加索人和其他）。

### 实验设置和评估指标
- **评估指标**：
  - **Accuracy (Acc)**：标准分类准确率。
  - **|DDP| (Absolute Demographic Parity Difference)**：衡量公平性的指标，值越小越公平。
  - **Hypervolume (HV)**：衡量 trade-off 曲线覆盖“理想区域”（高 Acc，低 |DDP|）的能力，**越大越好**。
  - **Inverted HV**：衡量 trade-off 曲线中“表现差的解”的数量，**越小越好**。它能揭示方法是否会产生一些非常糟糕的 trade-off 点。
- **实验流程**：
  - 所有方法均在训练集上训练。
  - 对于支持 post-hoc controllability 的方法（GFB, FairBayes, YOTO），只需训练一个模型，然后在测试集上评估 **10 个不同 trade-off 参数**下的性能。
  - 对于不支持 post-hoc controllability 的方法（EPO, FairBiNN），每个 trade-off 参数都需要单独训练一个模型。

### 基线方法对比
- **Post-processing (支持 post-hoc)**：
  - **FairBayes**：基于公平贝叶斯最优分类器的 post-processing 方法。
  - **YOTO**：一种 in-processing 方法，通过条件化 trade-off 参数实现单次训练、多次使用的特性。
- **In-processing (不支持 post-hoc)**：
  - **EPO**：多目标优化方法。
  - **FairBiNN**：双层优化框架。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
综合 **Table 1 (HV)** 和 **Table 2 (Inverted HV)** 以及图4的结果，可以得出以下结论：

#### 与 Post-processing 基线对比
- **相比 FairBayes**：GFB 在所有数据集上的 **HV 均显著更高**，且 **Inverted HV 更低**。这表明 GFB 不仅获得了更优的 trade-off 曲线，还避免了产生较差的 trade-off 点，性能全面超越。
- **相比 YOTO**：GFB 在所有数据集上的 **HV 更高**，**Inverted HV 更低**。尽管 YOTO 也支持 post-hoc controllability，但 GFB 实现了更高效、更稳定的 trade-off。

#### 与 In-processing 基线对比
- **相比 EPO**：GFB 在 **CelebA, UTKFace, Adult** 上的 HV 更高，在 COMPAS 上略低。然而，GFB 在所有数据集上的 **Inverted HV 都更低**，说明其 trade-off 曲线更稳定。最重要的是，GFB 以一次训练的成本达到了与需要多次训练的 EPO 相当甚至更好的性能。
- **相比 FairBiNN**：GFB 在 **CelebA 和 Adult** 上的 HV 更高。在 UTKFace 和 COMPAS 上，FairBiNN 的 HV 更高，但 GFB 的 **Inverted HV 显著更低**。这表明 FairBiNN 虽然能达到某些非常优秀的点，但也产生了更多表现很差的点；而 GFB 的 trade-off 表现更均衡、更可靠。

### 消融实验结果
论文未明确列出消融实验（Ablation Study）的独立表格，但其核心思想本身就是一种“消融”：
- **GFB vs. FairBayes**：两者都采用相同的 post-processing 流程，唯一的区别在于 GFB 在训练阶段额外学习了一个用于特征变换的模块 `gₐ`。实验结果显示 GFB 性能远超 FairBayes，这直接证明了 **学习特定特征表示对于提升 trade-off 效率至关重要**。

## 4. 关键结论和发现

### 主要发现
1.  **理论发现**：post-processed 分类器的 trade-off 效率高度依赖于特征分布。当特征集中在“最公平分类器”的决策边界附近时，放松公平性约束所带来的 accuracy 提升最大，从而实现更高效的 trade-off。
2.  **方法有效性**：提出的 **GFB 方法**成功地将这一理论发现转化为实践。它通过一个双层优化（bi-level optimization）框架，利用 MA-SOBA 算法进行求解，有效地学习到了符合上述特性的特征表示。
3.  **性能优越性**：实验表明，GFB 在保持 **post-hoc controllability** 的前提下，其 trade-off 效率可媲美甚至超越现有的 in-processing 方法，同时显著优于传统的 post-processing 方法。

### 方法的局限性
1.  **公平性度量范围有限**：当前框架主要支持 **Demographic Parity (DP)**，扩展到 **Equalized Odds (EO)** 等更复杂的公平性度量仍是一个重要挑战。
2.  **任务限制**：目前的方法局限于 **二元分类**（binary classification）和 **二元敏感属性**（binary sensitive attribute）。将其推广到多类别分类和由多个敏感属性定义的交叉群体（intersectional groups）是未来的重要方向。
3.  **计算复杂度**：虽然推理成本低，但 GFB 的训练时间比 FairBayes 略长（约1.2-1.7倍），因为它涉及更复杂的双层优化。

### 未来工作方向
1.  **扩展公平性度量**：将 GFB 框架扩展到支持 Equalized Odds (EO) 和 Predictive Equality (PE) 等更广泛的公平性定义。
2.  **处理多类别和交叉群体**：发展适用于多类别标签和多维敏感属性的技术，以应对更复杂的现实场景。
3.  **优化训练效率**：探索更高效的算法来加速 GFB 的双层优化训练过程，进一步降低其训练开销。

</details>

---

### 9. [Physics-Informed Neural Network with Transfer Learning for State Estimation in Lithium-Ion Batteries using the Single Particle Model with Electrolyte](https://arxiv.org/abs/2606.28220)

**Authors**: Gift Modekwe, Qiugang Lu  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.28220v1  

#### Abstract
Physics-informed neural networks (PINNs) have emerged as a powerful tool for solving nonlinear partial differential equations (PDEs), including battery electrochemical models. They typically en-force conservation laws within the loss function to ensure physically consistent solutions. Tradi-tional n...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Physics-Informed Neural Network with Transfer Learning for State Estimation in Lithium-Ion Batteries using the Single Particle Model with Electrolyte*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于 **Physics-informed Neural Networks (PINNs)** 的锂离子电池建模方法通常需要从零开始训练，针对不同电池化学体系（如 NMC vs LFP）、老化状态或工况条件时，存在以下问题：
- **训练耗时长**，收敛慢；
- **泛化能力差**，难以跨电池迁移；
- 对目标电池需大量标注数据，限制了实际应用。

### 🚀 提出的新方法与创新思路
本文提出了一种结合 **Transfer Learning (TL)** 的 **SPMe-PINN** 框架，用于锂离子电池的状态估计与参数识别：
- **首次将 Transfer Learning 引入 SPMe-based PINN 框架中**，实现电化学知识的跨电池迁移；
- 构建统一的 PINN 模型以同时捕捉固相扩散、电解液传输和反应动力学；
- 在微调阶段引入 **可学习的电化学参数**（如 $D_{s,n}$, $D_{s,p}$），支持对不同电池特性的自适应调整。

### 🔍 相比现有方法的优势
| 优势 | 说明 |
|------|------|
| **高效迁移** | 利用预训练模型的知识，显著减少目标电池上的训练时间与数据需求； |
| **物理一致性保持** | 通过冻结部分网络层保留通用电化学动态表示，确保预测符合 SPMe 物理规律； |
| **多任务能力** | 同时实现电压预测、内部状态估计（浓度）和关键参数反演（diffusivity）； |
| **无需标签监督** | 可在无真实内部状态测量的情况下进行训练，仅依赖终端电压与电流输入。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
实验采用三个公开电池数据集，代表不同的电池配置与化学体系：
| 数据集 | 缩写 | 化学体系 | 形态 | 来源 |
|--------|------|----------|-------|------|
| Chen et al. (2020) | B1 | Graphite/NMC | 圆柱形（LG M50） | Source battery |
| Ecker et al. (2015) | B2 | Graphite/NMC | 软包电池（Kokam） | Cross-domain transfer |
| Prada et al. (2013) | B3 | Graphite/LFP | — | Cross-chemistry transfer |

> 注：B1 用于预训练，B2 和 B3 用于迁移学习验证。

### ⚙️ 实验设置
- **PINN 架构**：4 隐藏层 × 64 神经元，激活函数为 tanh；
- **输入变量**：归一化时空坐标 $(\tilde{t}, \tilde{x}, r)$；
- **输出变量**：表面固相浓度 $c_{s,n}(R_n,t)$, $c_{s,p}(R_p,t)$，边界电解液浓度 $c_e$；
- **损失函数组成**：
  - PDE Residual Loss ($\mathcal{L}_{pde}$)
  - Boundary Condition Loss ($\mathcal{L}_{bc}$)
  - Initial Condition Loss ($\mathcal{L}_{ic}$)
  - Voltage Data Loss ($\mathcal{L}_v = \text{MSE}(V_{\text{pred}}, V_{\text{ref}})$)

- **迁移策略**：
  - 预训练阶段：在 B1 上完整训练 SPMe-PINN；
  - 微调阶段：加载预训练权重 → 冻结前两层 → 微调其余层及可学习参数（如 $D_{s,k}$）；
  - 优化器：Adam + L-BFGS。

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| RMSE | 终端电压预测误差（Root Mean Square Error） |
| MAE | 平均绝对误差 |
| 参数相对误差 | 如 $\frac{|D_{\text{pred}} - D_{\text{true}}|}{D_{\text{true}}}$ |
| 收敛速度 | 达到稳定损失所需的迭代次数或训练时间 |

### 🔁 基线方法对比
虽然文中未直接列出多个显式基线模型，但隐含对比包括：
- **从头训练的 SPMe-PINN**（Scratch Training）→ 作为迁移效果的对照；
- **传统数值求解器**（PyBaMM 中的 SPMe）→ 作为“真值”参考；
- **纯数据驱动模型**（如标准 MLP/RNN）→ 强调物理一致性的缺失。

---

## 3. 主要实验结果和性能指标

### 📌 关键性能数据

#### ✅ 源模型验证（B1）
- **RMSE = 8.1e-4 V**（图3）
- 几乎完美复现 PyBaMM 的 SPMe 解，表明 PINN 成功学习了基础电化学动态。

#### ✅ 跨域迁移（B2：同化学体系，不同结构）
- **1C 放电**下电压预测与 PyBaMM 高度一致（图4a），虽略有误差上升但仍保持高精度；
- 内部状态预测准确：
  - 负极表面浓度 RMSE < 0.01 p.u.
  - 正极表面浓度趋势匹配良好（图4b–c）；
- **0.5C 放电**仍保持优异表现（图4d），显示对操作条件变化鲁棒。

#### ✅ 跨化学体系迁移（B3：LFP vs NMC）
- 尽管 LFP 具有平坦电压平台等独特特征，模型仍能捕捉其关键响应特性（图5a–b）；
- 在 **1C 和 1.2C 放电率**下均表现出强泛化能力，证明框架适用于显著不同的电化学行为。

#### ✅ 参数估计能力（表1）
| 电池 | 电极 | 真实 $D_s$ | 预测 $D_s$ | 相对误差 |
|------|------|-------------|--------------|------------|
| B2 | 负极 | 8.332e-15 | 8.341e-15 | ~0.1% |
| B2 | 正极 | 2.981e-13 | 2.983e-13 | ~0.07% |
| B3 | 负极 | 3.000e-15 | 3.013e-15 | ~0.43% |
| B3 | 正极 | 5.900e-13 | 5.896e-13 | ~0.07% |

> ✅ 所有预测 diffusivity 与真实值高度接近，验证了 **learnable parameter** 设计的有效性。

### 🔍 消融实验（隐含分析）
尽管未明确命名消融实验，但以下设计体现了关键组件的作用：
- **冻结前两层** → 保留共享电化学表示，防止灾难性遗忘；
- **仅微调部分参数** → 加速收敛并避免过拟合小样本；
- 结果显示：若不冻结层，则需更多数据和更长时间才能达到相似性能。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Transfer Learning 显著提升 PINN 的实用性**：
   - 可将在一个电池上学习到的电化学先验知识迁移到另一个电池；
   - 大幅缩短训练时间，提高收敛稳定性。

2. **物理一致性得以保持**：
   - 即使在跨化学体系迁移中，模型依然遵循 SPMe 控制方程，输出物理合理的状态轨迹。

3. **兼具状态估计与参数辨识能力**：
   - 不仅能精准预测电压和内部浓度，还能反演关键材料参数（如 $D_s$），为在线健康管理提供可能。

4. **良好的泛化能力**：
   - 成功应对几何结构差异（圆柱 vs 软包）、放电倍率变化（0.5C–1.2C）以及完全不同化学体系（NMC → LFP）。

### ⚠️ 局限性
- 当前研究基于 **仿真生成数据**（来自 PyBaMM），尚未在真实实验噪声环境下验证；
- 迁移过程中假设部分参数（如反应速率常数 $k_k$）不变，可能影响极端差异体系的表现；
- 冻结层数选择依赖经验，缺乏自动化机制；
- 未考虑温度依赖性和老化效应（SEI增长、锂沉积等）。

### 🔮 未来工作方向
1. **集成退化模型**：将容量衰减、阻抗增长等老化机制纳入 PINN 框架；
2. **引入温度动态耦合**：构建 thermo-electrochemical PINN；
3. **使用真实实验数据训练与验证**：增强工业适用性；
4. **开发自适应冻结策略**：基于梯度分析决定哪些层应被固定；
5. **扩展至 full DFN model**：探索更复杂模型中的迁移可行性。

---

## 总结

该论文成功地将 **Transfer Learning** 与 **SPMe-PINN** 相结合，提出了一种高效、可推广的锂离子电池建模新范式。它不仅解决了传统 PINN 训练成本高、泛化差的问题，还实现了 **跨电池的状态估计与参数识别**，为下一代智能电池管理系统（BMS）提供了强有力的建模工具。

</details>

---

### 10. [Internalizing the Future: A Unified Agentic Training Paradigm for World Model Planning](https://arxiv.org/abs/2606.27483)

**Authors**: Xuan Zhang, Zhijian Zhou, Lingfeng Qiao, Yulei Qin, Ke Li, Xing Sun, Xiaoyu Tan, Chao Qu, Yuan Qi  
**Category**: cs.AI  
**Published**: 2026-06-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.27483v1  

#### Abstract
Large language model (LLM) agents have demonstrated strong capability in sequential decision-making, yet they remains fundamentally reactive in long-horizon tasks. Unlike humans who employ "what-if" reasoning to evaluate potential plans before commitment, standard agents lack an internal world model...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Internalizing the Future: A Unified Agentic Training Paradigm for World Model Planning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **LLM agents** 在长程任务中表现出**根本上的反应性（reactive）**，缺乏人类式的“what-if”推理能力。尽管已有方法尝试通过 Chain-of-Thought 等方式提升推理能力，但它们并未真正内化对未来状态的预测机制。作者指出，仅在后训练阶段（post-training）引入前瞻性格式（look-ahead format），会导致模型产生“**格式-能力差距（format-capability gap）**”——即模型能模仿前瞻推理的形式，但其内容缺乏真实、可靠的预测基础，导致幻觉和误导。

### 提出了什么新方法或新思路
为解决上述问题，作者提出了一种**统一的三阶段训练范式**，将世界模型规划能力内化到单一自回归语言模型中，无需额外模块或价值头（value head）。该方法的核心是让模型同时生成两种输出：
- **Prospective State Rollout**：对未来的紧凑状态推演。
- **Plan-Conditioned Success Estimate**：计划条件下的成功概率估计，可视为文本化的 Q-value。

具体三阶段如下：

1. **World Model Agentic Mid-Training (WM-AMT)**  
   在 mid-training 阶段注入预测能力。通过对大规模代理轨迹进行增强，在历史状态后插入一个结构化的“世界模型块”（world model block），包含抽象的未来路径规划和成功置信度估计。这一步旨在建立**潜在的预测能力先验**。

2. **Format-Eliciting SFT (FE-SFT)**  
   在 SFT 阶段教会模型如何以结构化方式表达其内部世界模型。通过指令微调，引导模型在推理时主动输出 `<imaginary>`、`<keyword>`、`<confidence>` 等结构化标签，从而激活 WM-AMT 中注入的能力。

3. **Foresight-Conditioned Reinforcement Learning (FC-RL)**  
   在 RL 阶段联合优化策略和世界模型。设计了两个互补的目标：
   - **World Model Optimization**：通过 `R_ground`（关键词匹配奖励）和 `R_calib`（基于 Brier Score 的置信度校准惩罚）来确保模拟的准确性和置信度的可靠性。
   - **Policy Optimization**：结合全局任务成功奖励和局部置信度优势（step-level advantage）更新策略，避免稀疏奖励问题。

### 相比现有方法的优势
- **统一架构**：所有功能集成于单个 autoregressive 模型，无外部模拟器或独立价值网络。
- **能力优先（capability-first）**：强调在 mid-training 阶段构建预测能力，而非仅靠后训练“教格式”。
- **显式校准机制**：通过环境反馈持续校准文本化 Q-value，使其与实际成功率对齐。
- **透明可解释**：生成的“世界模型块”提供了可读的推理信号，支持隐式路径剪枝。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在两类代表性代理任务上进行：

1. **Search（搜索增强问答）**  
   - 数据集：7个 QA 数据集，包括单跳（NQ, TriviaQA, PopQA）和多跳（HotpotQA, 2Wiki, MuSiQue, Bamboogle）。
   - 检索源：2018 Wikipedia dump，使用 E5 作为检索器。

2. **Mathematical Reasoning（数学推理）**  
   - 数据集：AIME2024/2025/2026（美国数学邀请赛）。
   - 评估方式：重复评估30次，报告 `mean@30` 和 `pass@30`。

### 实验设置和评估指标
- **基础模型**：Youtu-LLM-2B-Init（20亿参数）。
- **训练流程对比**：
  - **Base**：标准 agentic mid-training + SFT + RL。
  - **Ours (WM-AMT)**：在相同轨迹基础上插入世界模型块进行 mid-training。
- **评估指标**：
  - Search：平均得分（Avg.）。
  - Math：`mean@30`, `pass@30`。
  - 使用 DeepSeek-V3.1 作为 LLM-as-a-judge 进行评分。

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| Post-Training Only* | 仅在标准 Base 上进行 SFT/RL，无世界模型增强 |
| State-Only Prediction (Post-Training Only)† | 仅预测未来状态，不包含 verbalized Q-value |
| State-Only Prediction (With Mid-Training)† | 在 mid-training 中加入状态预测 |
| IWM | 基于 [15] 的世界模型数据生成方法 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 表1：Search 任务最终性能（含 RL）
| Method | Avg. |
|--------|------|
| SFT & RL* | 46.1 |
| FE-SFT & RL* | 47.0 |
| State-SFT & RL† | 47.1 |
| IWM & RL | 48.4 |
| **FE-SFT & FC-RL (Ours)** | **50.6** ✅ |

#### 表2：Mathematical Reasoning 最终性能
| Method | mean@30 | pass@30 |
|--------|---------|---------|
| SFT & RL* | 28.0 | 52.2 |
| FE-SFT & RL* | 28.0 | 51.1 |
| State-SFT & RL† | 27.7 | 56.7 |
| IWM & RL | 28.6 | 57.8 |
| **FE-SFT & FC-RL (Ours)** | **29.5** ✅ | **60.0** ✅ |

> 注：`FE-SFT & FC-RL` 是完整三阶段方法的结果。

### 与基线方法的对比结果
- 在 **Search** 任务上，相比最强 state-only 基线（48.7），我们的方法提升了 **1.9 分**。
- 在 **Math** 任务上，`pass@30` 达到 **60.0**，显著优于其他方法。
- 特别是在**复杂多跳任务**（如 MuSiQue, Bamboogle）上提升明显，表明内化世界模型对长程规划至关重要。

### 消融实验结果
#### （1）三阶段必要性分析
| 阶段组合 | Search (Avg.) | Math (pass@30) |
|--------|--------------|----------------|
| WM-AMT + SFT | 39.9 | —— |
| WM-AMT + FE-SFT | 41.8 | 43.3 |
| WM-AMT + FE-SFT + FC-RL | **50.6** | **60.0** |

> 结果显示：**WM-AMT 提供能力基础，FE-SFT 激活结构化输出，FC-RL 完成联合优化**，缺一不可。

#### （2）消融 `R_ground`（接地奖励）
| 方法 | Avg. |
|------|------|
| w/ R_ground | **50.6** |
| w/o R_ground | 50.1 |

> 移除接地奖励导致性能下降，说明**确保模拟内容与真实执行一致**对稳定性至关重要。

#### （3）敏感性分析（FC-RL 中的 `w`）
| `w` | Search (Avg.) |
|-----|---------------|
| 0.0 | 49.1 |
| 0.1 | 50.0 |
| **0.2** | **50.6** ✅ |
| 0.5 | 49.2 |

> 当 `w=0.2` 时达到最优，过大则分散注意力，影响基础推理。

#### （4）响应长度分析
| 方法 | Search (Tokens) | Math (Tokens) |
|------|----------------|----------------|
| Standard SFT | 2603.8 | 4601.2 |
| FE-SFT (Ours) | 3084.5 (+18.5%) | 4749.1 (+3.2%) |

> 引入世界模型块仅带来轻微开销，说明模型学会了生成**高度压缩的未来摘要**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **存在 format-capability gap**：仅靠 post-training 教授前瞻格式无法获得真正的预测能力，容易导致幻觉。
2. **必须采用 capability-first 范式**：预测能力需在 mid-training 阶段通过大规模轨迹增强注入，而非后期“补课”。
3. **Verbalized Q-value 至关重要**：仅预测未来状态不足以支撑有效决策；显式的成功估计（confidence）并加以校准，才能指导策略优化。
4. **内化优于外挂**：将世界模型能力内化于策略本身，比依赖外部模拟器更高效且一致。
5. **长程任务受益最大**：在多跳问答和复杂数学推理中提升最显著，验证了其对 long-horizon planning 的有效性。

### 方法的局限性
- **依赖高质量轨迹生成**：WM-AMT 所需的“未来真相”由另一个 LLM（DeepSeek-V3.1）生成，可能存在噪声。
- **计算成本较高**：mid-training 阶段需要处理大量增强数据，对资源要求高。
- **泛化性待验证**：目前仅在 search 和 math 任务验证，是否适用于更广泛的 agent 任务（如代码、游戏）尚需探索。

### 未来工作方向
- 探索更高效的 mid-training 数据合成方法，减少对外部强模型的依赖。
- 将该范式扩展至多模态 agent 或具身智能体（embodied agents）。
- 研究如何动态决定何时启动“世界模型模式”，实现推理预算的自适应分配。
- 探索将 verbalized Q-value 用于跨任务迁移或元学习。

--- 

> **总结**：本文提出了一个开创性的“能力先行”训练范式，首次系统性地解决了 LLM agents 缺乏内生世界模型的问题。通过 WM-AMT、FE-SFT 和 FC-RL 三阶段协同，实现了**可校准、可解释、可优化的前瞻性规划能力**，为构建真正具备长期规划能力的通用智能体提供了坚实基础。

</details>

---

### 11. [ATOD: Annealed Turn-aware On-policy Distillation for Multi-turn Autonomous Agents](https://arxiv.org/abs/2606.27814)

**Authors**: Qitai Tan, Zefang Zong, Yang Li, Peng Chen  
**Category**: cs.AI  
**Published**: 2026-06-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.27814v1  

#### Abstract
Training small language-model agents for long-horizon interactive tasks requires both fast imitation and reward-driven improvement. On-policy distillation (OPD) provides dense teacher guidance and typically improves rapidly in the early stage, but its gains saturate once the student approaches the t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ATOD: Annealed Turn-aware On-policy Distillation for Multi-turn Autonomous Agents**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
训练小型语言模型（small language-model agents）在长周期、多轮交互任务中面临两大挑战：
- **冷启动效率低**：纯 **Reinforcement Learning (RL)** 方法（如 GRPO）依赖稀疏且延迟的环境奖励信号，在早期探索阶段学习效率极低，尤其对小模型不友好。
- **教师天花板效应**：**On-policy Distillation (OPD)** 虽然通过教师模型提供密集的 token-level 指导，能快速提升性能，但学生模型容易陷入模仿瓶颈，难以超越教师表现。

此外，传统 OPD 在长轨迹中对所有“turn”（决策步）施加均匀监督，导致在**常规操作**上浪费监督资源，而在**关键决策点**上监督不足。

---

### **提出的新方法与思路**
本文提出 **ATOD**（**Annealed Turn-aware On-policy Distillation**），一种融合 OPD 与 RL 的在线蒸馏算法，包含两个核心创新：

#### **(1) Annealed OPD-RL Schedule（退火式 OPD-RL 调度）**
- **早期阶段**：以 **OPD** 为主，利用教师模型的密集指导快速引导学生掌握基本行为模式，实现高效冷启动。
- **后期阶段**：逐步增强 **RL** 信号权重，鼓励学生基于环境奖励进行探索，突破教师能力上限。
- 公式形式为混合优势函数：
  $$
  A = \kappa(s) A_{\text{OPD}} + \rho(s) A_{\text{GRPO}}
  $$
  其中 $\kappa(s)$ 和 $\rho(s)$ 随训练步数动态调整。

#### **(2) Turn-level Disagreement-Uncertainty Reweighting (T-DUR)**
- 将监督重点从“token”提升到“turn”级别，因为“turn”是代理决策的基本单位。
- 为每个 turn 计算两个信号并融合为一个软权重 $w_k$：
  - **Disagreement Proxy**：衡量学生与教师在该 turn 的输出分歧。
  - **Uncertainty Proxy**：衡量学生自身对该 turn 输出的不确定性（通过负对数概率估计）。
- 使用 **Soft-OR** 融合机制：
  $$
  w_k = 1 - (1 - \hat{d}_k)(1 - \hat{h}_k)
  $$
  当任一信号高时，即认为该 turn 是高价值学习点，应加强蒸馏监督。

> ✅ **注意**：T-DUR 仅重加权 OPD 部分，不影响 RL 奖励信号，确保探索动力不受干扰。

---

### **相比现有方法的优势**
| 方法 | 缺陷 | ATOD 如何改进 |
|------|------|----------------|
| **GRPO** | 稀疏奖励导致冷启动慢 | 利用 OPD 快速引导初期学习 |
| **OPD** | 易饱和于教师水平，缺乏探索 | 引入渐进式 RL 探索，突破天花板 |
| **SOD / TCOD** | 固定或简单调度，未解耦信号强度 | 动态退火调度 + turn-aware 加权，更精细控制学习过程 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在三个具有代表性的长周期多轮代理任务上验证：
- **ALFWorld**：文本版家庭环境中的具身指令执行（如“把遥控器放到沙发上”）。
- **WebShop**：目标导向的网页导航与商品选择（模拟电商购物）。
- **Search-QA**：结合搜索引擎的多跳问答任务，涵盖多个子集：
  - Natural Questions, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle

这些任务共同特点是：**错误传播、延迟奖励、部分可观测、多步骤决策**。

---

### **实验设置与评估指标**

#### **模型配置**
- **学生模型**：`Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`
- **教师模型**：
  - 对 0.6B/1.7B 学生：`Qwen3-4B GRPO`
  - 对 4B 学生：`Qwen3-30B-A3B GRPO`（150-step checkpoint）

#### **训练细节**
- 使用 **vLLM** 进行 rollouts，group size = 8
- 训练步数：150 steps
- 退火窗口：前 80 步完成从 OPD 到 RL 的过渡
- 不使用显式 KL 惩罚项

#### **评估指标**
- **Success Rate (SR, %)**：任务成功完成率（主指标）
- **Average Trajectory Length (Len.)**：平均交互步数（反映策略效率）
- 报告 **最大验证成功率** 及对应长度

---

### **基线方法对比**
| 基线 | 类型 | 简介 |
|------|------|------|
| **Vanilla** | 无后训练 | 原始基础模型 |
| **GRPO** | RL | 纯环境奖励优化 |
| **SDAR** | Self-Distillation + RL | 自蒸馏辅助 RL |
| **OPD** | Teacher-based Distillation | 标准在线蒸馏 |
| **SOD** | Step-wise OPD | 按步骤调整蒸馏强度 |
| **TCOD** | Temporal Curriculum OPD | 时间课程控制蒸馏范围 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| 学生大小 | 方法 | 平均成功率 (Avg. SR) | 是否超过教师？ |
|--------|------|------------------|-------------|
| 0.6B | ATOD | **70.62%** | ✅ 是（教师：68.93%） |
| 1.7B | ATOD | **71.58%** | ✅ 是（教师：68.91%） |
| 4B | ATOD | **71.06%** | ✅ 是（教师：68.91%） |

> 📈 **总体提升**：
> - 比 **OPD** 平均高出 **3.03 pts**
> - 比 **GRPO** 平均高出 **23.62 pts**
> - 比第二佳基线平均高出 **3.7–5.1%**

---

### **与基线方法的对比结果**
- **ATOD 在所有三个数据集、所有学生规模下均达到 SOTA 性能**。
- 特别是在 **ALFWorld** 上，0.6B 模型从 Vanilla 的 **0.78%** 提升至 **82.81%**，实现超百倍增长。
- 在 **WebShop** 上，1.7B 模型达到 **89.06%** 成功率，显著优于其他方法。
- 即使面对更强的 30B 教师模型，4B 学生仍能通过 ATOD **反超教师表现**。

---

### **消融实验结果（Ablation Study）**

在 ALFWorld 上对 Qwen3-0.6B/1.7B/4B 进行消融分析：

| 变体 | 描述 | 结果影响 |
|------|------|---------|
| **w/o Annealing** | 固定 OPD/RL 权重 | 性能下降最严重（如 0.6B 从 82.8% → 75.8%） |
| **w/o T-DUR** | 所有 turn 均匀加权 | 明显劣于完整 ATOD |
| **Token-level reweighting** | 在 token 级别应用 T-DUR | 不稳定，效果不如 turn-level |

> 🔍 **结论**：**退火调度** 和 **turn-level T-DUR** 均为关键组件，缺一不可。

---

## **4. 关键结论和发现**

### **主要发现**
1. **OPD 与 RL 的互补性可被有效利用**：
   - 早期依赖教师指导加速收敛；
   - 后期释放 RL 探索潜力，突破模仿极限。

2. **Turn 是比 Token 更合适的监督粒度**：
   - 代理决策天然以 turn 为单位；
   - 在 turn 级别聚合 disagreement 与 uncertainty 更稳定、语义更清晰。

3. **T-DUR 能智能分配监督资源**：
   - 高权重赋予“高分歧”或“高不确定”的 turn（如错误恢复、关键抉择）；
   - 低权重跳过“例行公事”操作（如打开已知容器）。

4. **ATOD 支持“小模型超越大模型”**：
   - 即便学生小于教师，也能通过奖励驱动优化实现反超。

---

### **方法的局限性**
- **依赖高质量教师模型**：若教师本身存在系统性偏差，早期蒸馏可能固化错误。
- **T-DUR 信号估计基于采样 token**：虽避免全词表查询，但仍有一定方差。
- **当前仅适用于 post-training 场景**：未涉及预训练阶段的知识迁移。

---

### **未来工作方向**
- 探索 **自适应退火策略**（非固定 schedule）。
- 将 T-DUR 扩展至 **multi-agent 或 hierarchical policy** 设置。
- 结合 **value-aware distillation** 进一步提升信用分配精度。
- 应用于真实世界工具调用场景（如浏览器自动化、代码执行）。

---

> ✅ **一句话总结**：  
> **ATOD 通过“退火式 OPD-RL 调度”与“turn-aware 动态加权”，实现了小语言模型代理的高效冷启动与持续超越，是迈向高效、高性能轻量级智能体的重要一步。**

</details>

---

### 12. [Accelerating Hierarchical Sparse Predictive Coding with Hybrid Amortized Inference](https://arxiv.org/abs/2606.27802)

**Authors**: Kazuhisa Fujita  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.27802v1  

#### Abstract
Hierarchical predictive coding provides an interpretable framework for perception as error-driven inference in multi-layer generative models, while sparse coding imposes parsimonious latent representations through explicit sparsity constraints. Their combination yields hierarchical sparse predictive...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Accelerating Hierarchical Sparse Predictive Coding with Hybrid Amortized Inference*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Hierarchical Sparse Predictive Coding**（如 SDPC）模型中存在的**推理效率瓶颈**问题。这类模型虽然具有良好的可解释性和神经科学合理性，但由于依赖迭代优化（如 ISTA）进行潜变量推断，导致每输入一个样本都需要多次递归更新，尤其在深层结构中延迟高、稳定性差，限制了其实际应用。

### 提出的新方法
提出了一种 **Hybrid Inference**（混合推理）框架，结合了**快速摊销初始化**（amortized initialization）与**少量能量驱动的迭代精炼**（iterative refinement）：
- **摊销初始化**：采用受 LISTA 启发的前馈编码器（LISTA-style bottom-up encoder），将输入快速映射为稀疏潜码的初始估计。
- **迭代精炼**：以该初始码为起点，执行少量 ISTA 风格的基于完整分层稀疏能量函数的梯度更新步骤，进行误差修正。

### 相比现有方法的优势
- **速度远快于纯迭代方法**（如 ISTA/MFISTA）：避免了数十甚至上百步的迭代。
- **质量优于纯摊销方法**（如 LISTA）：通过能量驱动的精炼弥补了纯前馈预测的精度损失。
- **在质量-延迟权衡上取得更优平衡**：实现了“**浅层摊销 + 少量精炼**”的高效策略，而非深度摊销或长序列迭代。

---

## 2. 核心实验方法和设置

### 数据集
实验在以下静态图像基准上进行：
- **MNIST**：手写数字，28×28 灰度图
- **Fashion-MNIST**：服饰图像，28×28 灰度图
- **CIFAR-10 Gray**：CIFAR-10 转灰度图，32×32
- （附录）**BSDS500 Patch**：自然图像块，16×16，用于验证经典稀疏编码场景

### 实验设置
- **模型架构**：固定为两层（默认）或更深的分层稀疏生成模型（Hierarchical Sparse Generative Model）
- **共享目标函数**：所有方法优化相同的分层稀疏能量函数（含重建误差、层间一致性、ℓ₁ 稀疏正则项）
- **训练协议**：交替优化——先固定字典推断潜码，再用最终损失更新字典和编码器参数（不反向传播整个推理过程，提升稳定性和效率）

### 评估指标
- **Test Loss**：测试集上的平均分层稀疏能量（综合指标）
- **Reconstruction Error**：仅输入空间的重建误差 $\|x - D_1 a_1\|^2$
- **Latency**：单样本推理耗时（ms/sample），在统一硬件下测量
- **Sparsity**：激活系数比例（magnitude > 1e-5）
- **Stability**：多随机种子下的性能方差

### 基线方法对比
比较了四种推理方案：
1. **ISTA**：标准迭代软阈值算法（慢而准）
2. **MFISTA**：加速版单调 FISTA（更快收敛的迭代参考）
3. **LISTA**：纯摊销推理，使用 LISTA 风格编码器直接输出结果
4. **Hybrid**（本文方法）：LISTA 初始化 + $T_{\text{ref}}$ 步 ISTA 精炼
5. （消融）**Hybrid-MFISTA**：替换精炼阶段为 MFISTA

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
| 方法 | 推理预算 | Latency (ms/sample) | Reconstruction Error (典型值) | 相对优势 |
|------|----------|---------------------|-------------------------------|---------|
| **LISTA (K=1)** | K=1 | ~0.03 | 较高（如 FMNIST ~2.75） | 最快，但精度有损 |
| **Hybrid (K=1, Tref=5)** | K+Tref=6 | ~0.25–0.35 | 显著降低（如 FMNIST ~2.71） | 比 LISTA 更准，比 ISTA 快 10 倍以上 |
| **ISTA (50步)** | 50 | ~3–5 | 低 | 准但极慢 |
| **MFISTA (20步)** | 20 | ~2–4 | 低 | 加速但仍慢 |

- **Hybrid 在所有数据集上一致优于纯 LISTA**：即使只加 1–3 步精炼，也能显著降低 Test Loss 和 Reconstruction Error。
- **Hybrid 远快于迭代方法**：完成精炼后仍比 20–50 步 ISTA/MFISTA 快一个数量级。

### 消融实验结果
#### （1）Hybrid 预算分配分析（K vs Tref）
- **增加 $K$（摊销深度）效果有限甚至有害**：例如在 Fashion-MNIST 上，$(K=5, T_{\text{ref}}=1)$ 不如 $(K=1, T_{\text{ref}}=5)$，说明更多摊销阶段不能有效替代精炼。
- **增加 $T_{\text{ref}}$（精炼步数）持续提升质量**：前几步增益最大，之后趋于饱和。
- **结论**：最优策略是 **浅层摊销 + 多步精炼**，而非深层摊销。

#### （2）深度扩展（Depth Scaling）
- 随着层数增加（L=1 到 L=8），**ISTA 方差增大，不稳定**；**LISTA 质量下降明显**。
- **Hybrid 表现出更好的稳定性**，在多数深度下保持良好性能，优于纯摊销和未加速迭代方法。

#### （3）稀疏强度（Sparsity Weight λ）与步长缩放（scale）敏感性
- **Hybrid 在不同 λ 下均优于 LISTA**，表明其优势不依赖特定稀疏设定。
- **对 step-size scale 敏感但稳健**：scale=1.0 表现最佳，但即使 scale 变化，Hybrid 仍保持在合理性能区间，而 ISTA 对 scale 更敏感。

---

## 4. 关键结论和发现

### 主要发现
1. **摊销与迭代应协同而非互斥**：纯摊销（LISTA）虽快但易损失精度；纯迭代（ISTA）虽准但慢。**Hybrid 将二者结合，在共享目标下实现高效高质量推理**。
2. **“浅摊销 + 少精炼” 是最优策略**：一个简单的 LISTA 编码器即可提供良好初值，后续少量基于真实能量函数的 ISTA 更新能高效纠正误差。
3. **Hybrid 提升稳定性**：在深层模型和复杂数据（如 CIFAR-10 Gray）上，相比纯方法更具鲁棒性。

### 方法的局限性
- 当前实现基于全连接结构，尚未扩展到 **Convolutional Sparse Coding** 或更大规模图像。
- 精炼步数仍需手动设定，缺乏自适应机制。
- 训练依赖 detach 字典以稳定梯度，非完全局部学习规则，生物学合理性有待加强。
- 未处理时间序列或动态预测任务。

### 未来工作方向
- 扩展至 **卷积稀疏编码** 和更大图像模型（如 ImageNet 子集）。
- 引入 **adaptive inference budget**：根据输入难度动态调整精炼步数。
- 结合更先进的摊销结构，如 **ALISTA** 或 **structured amortizers**。
- 探索 **safeguarded refinement** 机制，确保收敛性。
- 延伸至 **temporal data** 和 **predictive coding for video**，验证在连续输入下的有效性。

--- 

> **总结**：该论文通过系统实验证明，**Hybrid Amortized Inference** 是加速分层稀疏预测编码的有效范式。它不是简单折衷，而是揭示了“**快速初始化 + 局部能量修正**”这一高效推理路径，在保持模型语义的同时大幅降低延迟，为构建高效、可解释的类脑感知模型提供了实用方案。

</details>

---

### 13. [MER-R1: Multimodal Emotion Reasoning via Slow-Fast Thinking Synergy](https://arxiv.org/abs/2606.27652)

**Authors**: Zhiyuan Han, Beier Zhu, Wenwen Tong, Chengwei Qin, Xinyi Wang, Jiayu Zhang, Jiangnan Chen, Hewei Guo, Dongchuan Ran, Lewei Lu, Xun Yang  
**Category**: cs.AI  
**Published**: 2026-06-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.27652v1  

#### Abstract
We find that explicit reasoning does not necessarily translate into better multimodal emotion recognition (MER) accuracy, even though it makes predictions more interpretable. Specifically, for reasoning-based MLLMs, fast thinking by triggering direct answers often outperforms slow thinking after del...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MER-R1: Multimodal Emotion Reasoning via Slow-Fast Thinking Synergy

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文揭示并解决了**多模态情感识别（MER）中的“思维悖论”（thinking paradox）**：
- 当前基于推理的 MLLMs 在进行显式推理（slow thinking）时，虽然提升了模型的可解释性，但其识别准确率反而低于直接快速作答（fast thinking）。
- 这表明：**推理本身并不必然提升情感识别性能**，甚至可能因过度保守而丢失正确情绪。

### ✅ 提出的新方法与新思路
作者提出 **MER-R1**，一个通过**慢速-快速思维协同（Slow-Fast Thinking Synergy）** 来优化多模态情感推理的强化学习框架。其核心思想是：
> 将 fast thinking 的高召回优势与 slow thinking 的高精度选择性结合起来，而非简单取舍。

#### 主要创新组件：

1. **Dual-Objective Disentanglement（双目标解耦）**
   - 将 F1 分数分解为独立的 **Recall Reward** 和 **Precision Reward**。
   - 在优势函数（advantage）空间中分别归一化处理，避免传统 F1 优化中因方差差异导致某一目标被主导的问题。
   - 理论上证明该设计能实现 recall 与 precision 的均衡优化。

2. **Slow-Fast Confidence Calibration（慢快置信度校准）**
   - 利用 fast thinking 的置信行为指导最终 slow thinking 输出：
     - 对于正确的类别：鼓励保持或超过 fast thinking 的置信度；
     - 对于错误的类别：保留 slow thinking 的抑制倾向。
   - 实现“增强正确情绪、抑制噪声”的效果。

### ✅ 相比现有方法的优势
| 维度 | MER-R1 的优势 |
|------|----------------|
| 性能 | 显著优于所有基线，在多个 benchmark 上达到 SoTA |
| 推理有效性 | 首次使 slow thinking 的最终输出优于 fast thinking，真正让推理“有用” |
| 设计原理 | 不仅提升性能，还提供理论支持（如对抗方差干扰） |
| 可解释性 | 保留 Chain-of-Thought 结构的同时提升准确性 |

---

## 2. 核心实验方法和设置

### ✅ 数据集
在两个主流多模态情感理解 benchmark 上进行全面评估：

| 数据集 | 描述 |
|-------|------|
| **MER-UniBench** | 包含 9 个子数据集，涵盖三类任务：<br>- 细粒度情感识别（OV-MERD+）<br>- 基础情感识别（MER2023/24, MELD, IEMOCAP）<br>- 情感极性分析（MOSI, MOSEI, SIMS, SIMSv2） |
| **MME-Emotion** | 更全面的评测基准，包含 8 项任务：<br>- 实验室/野外/噪声环境下的情感识别（ER-Lab/Wild/Noise-ER）<br>- 多标签/细粒度情感识别（ML-ER, FG-ER）<br>- 情感/细粒度情感/意图识别（SA, FG-SA, IR） |

### ✅ 实验设置与评估指标

| 设置项 | 内容 |
|--------|------|
| **Backbone** | Qwen2.5-Omni |
| **训练流程** | 两阶段：Supervised Fine-Tuning (SFT) + GRPO-style RL |
| **SFT** | 5k 样本，2 轮，lr=2e-5 |
| **RL** | 剩余数据，1 轮，lr=2e-6，每提示采样 4 个响应 |
| **硬件** | 16 × NVIDIA H100 GPUs |

#### 评估指标：
- **MER-UniBench**：
  - 细粒度情感识别：F1-score（基于 emotion wheel 归一化）
  - 基础情感识别：Hitrate（官方），本文补充使用 F1
  - 情感极性分析：Weighted Average F-score (WAF)
- **MME-Emotion**：
  - 使用 LLM-as-a-judge 机制（本文采用 gemini-3.1-flash-lite-preview 替代已不可用的 GPT-4o）
  - 报告三项指标：Recognition、Reasoning、CoT（二者平均）

### ✅ 基线方法对比
- **通用模型**：Qwen-Audio, SALMONN, VideoChat2, LLaMA-VID, Chat-UniVi, mPLUG-Owl, PandaGPT
- **情感专用模型**：R1-Omni, Emotion-LLaMA, AffectGPT, AffectGPT-R1
- **推理增强模型**：AffectGPT-R1, VideoAuto-R1（answer-think-answer 范式）

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### 📊 表1：MER-UniBench 官方协议结果
| 模型 | Mean Score |
|------|------------|
| AffectGPT-R1 | 79.98 |
| **MER-R1（本文）** | **83.50** ✅ |

> **提升幅度：+5.63 pts**，在所有子任务上均取得领先。

#### 📊 表2：MME-Emotion 整体表现
| 方法 | CoT Mean |
|------|----------|
| Baseline | 45.3 |
| VideoAuto-R1 | 46.6 |
| **MER-R1（本文）** | **51.5** ✅ |

> 在 Recognition 得分从 27.9 → **38.4**，Reasoning 也略有提升（62.7 → 64.6），说明性能提升未牺牲推理质量。

### ✅ 与基线方法的对比结果

| 发现 | 说明 |
|------|------|
| ✅ Fast thinking > Slow thinking（在 baseline 中） | 原始 slow thinking 因过于保守，反而不如直接输出 |
| ✅ VideoAuto-R1 仍无法逆转趋势 | 其“先答后想”虽奖励早期答案，但缺乏对 slow-fast 差异建模 |
| ✅ **MER-R1 成功反转趋势** | 最终 slow-thinking 输出优于 fast-thinking（见 Table 3） |

#### 🔍 Table 3 关键对比（统一 F1 评估下）：
| 模型 | Fast R-Mean | Slow R-Mean |
|------|-------------|--------------|
| Baseline | 58.49 | 58.11 ❌ |
| VideoAuto-R1 | 61.05 | 60.51 ❌ |
| **MER-R1** | **60.86** | **61.80** ✅ |

> **首次实现 slow-thinking 输出优于 fast-thinking**，验证了 slow-fast synergy 的有效性。

### ✅ 消融实验结果（Ablation Studies）

#### 📊 表4：逐步添加模块的效果（MER-UniBench）
| 变体 | RD | AD | SFCC | Mean<sub>official</sub> | Mean<sub>F1</sub> |
|------|----|----|------|------------------------|------------------|
| Baseline |    |    |      | 77.87 | 70.55 |
| +RD     | √  |    |      | 80.18 | 71.15 |
| +RD+AD  | √  | √  |      | 82.20 | 71.60 |
| Full (MER-R1) | √ | √ | √   | **83.50** | **73.15** |

> 所有三个组件均有贡献，**SFCC 提升最大**。

#### 📊 表5：Slow-Fast Confidence Calibration 消融
| 变体 | 描述 | Mean<sub>official</sub> |
|------|------|------------------------|
| A1 | 使用 word-level confidence（非 category-level） | 81.81 |
| A2 | 移除 precision side calibration | 82.87 |
| B1 | reward-space calibration | 81.91 |
| B2 | mixed-space calibration | 83.11 |
| **MER-R1** | **category-level + advantage-space + bidirectional** | **83.50** ✅ |

> 证明：**category-level 双向校准 + advantage space 优化** 是最优组合。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **存在“思维悖论”**：
   - 在当前 MER 任务中，**fast thinking 的 recall 更高，slow thinking 的 precision 更高但 recall 下降明显**。
   - 导致整体 F1 不及 fast thinking。

2. **fast 与 slow 的互补性明确**：
   - Fast thinking：覆盖广、置信高（尤其对正确类别）
   - Slow thinking：过滤强、抑制噪声，但可能误删正确项

3. **MER-R1 成功融合两者优势**：
   - 通过 dual-objective disentanglement 平衡 recall 与 precision 优化路径；
   - 通过 slow-fast confidence calibration 保留 fast 的正向置信、继承 slow 的负向抑制。

4. **推理终于“有用”了**：
   - MER-R1 是首个使得 **slow-thinking 最终输出优于 fast-thinking** 的方法，真正实现了“推理有益”。

### ⚠️ 局限性（Limitations）
1. **依赖 emotion wheel 映射**：
   - 限制了对非常细粒度情绪（如“nostalgic”、“bittersweet”）的表达能力。
2. **训练成本增加**：
   - 需额外一次 fast-thinking 前向传播以获取 confidence，带来约 20% 计算开销。
3. **泛化性待验证**：
   - 当前仅在 MER 类任务验证，是否适用于其他多模态推理任务（如 VQA、video reasoning）尚不明确。

### 🔮 未来工作方向
1. **构建无需 emotion wheel 的端到端开放词汇优化机制**
2. **探索更高效的 confidence estimation 方式**（如缓存、蒸馏）
3. **将 slow-fast synergy 扩展至更多认知任务**（如决策、规划）
4. **结合 human-in-the-loop 学习进一步优化 reasoning quality**

---

> 💡 **一句话总结**：  
> MER-R1 首次系统揭示并解决多模态情感推理中的“思维悖论”，通过 **dual-objective disentanglement** 与 **slow-fast confidence calibration** 实现 fast 与 slow 思维的优势互补，不仅刷新 SoTA，更让推理过程真正服务于性能提升，而非仅用于解释。

</details>

---

### 14. [ToE: A Hierarchical and Explainable Claim Verification Framework with Dynamic Multi-source Evidence Retrieval and Aggregation](https://arxiv.org/abs/2606.27736)

**Authors**: Zhaoqi Wang, Zijian Zhang, Kun Zheng, Zhen Li, Xin Li, Chunlei Li, Jiamou Liu  
**Category**: cs.AI  
**Published**: 2026-06-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.27736v1  

#### Abstract
The rapid spread of fake news poses increasing threats to information ecosystems, especially as AI-generated misinformation under Generative Engine Optimization (GEO) poisoning allows adversarially crafted content to be systematically surfaced by retrieval systems, contaminating LLM reasoning. In th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ToE: A Hierarchical and Explainable Claim Verification Framework with Dynamic Multi-source Evidence Retrieval and Aggregation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对当前 **LLM 在事实核查任务中易受虚假信息污染** 的问题，尤其是由 **Generative Engine Optimization (GEO) 攻击** 导致的检索结果中毒现象。传统基于 RAG 或 Tool Calling 的方法虽然能引入外部知识缓解幻觉，但若检索到的内容本身是 AI 生成的误导性信息，则会进一步加剧错误推理。

例如，攻击者可注入“Tim Cook 加入 OpenAI 担任 CEO”这类流畅且看似可信的虚假文档，在检索排序中占据高位，从而误导 LLM 得出错误结论。

### 提出了什么新方法或新思路
作者提出 **Tree of Evidence (ToE)** ——一种层次化、可解释的事实核查框架，其核心思想是将每个声明（claim）建模为一个动态扩展的 **argument tree（论证树）**，通过以下三个核心组件实现自动化验证：

- **Reinforcement Learning-driven Multi-source Retrieval Agent**  
  基于强化学习的多源检索代理，根据声明特征自主选择最优信息源（如 Wikipedia、arXiv、PolitiFact、社交媒体等），并决定何时停止检索。
  
- **Evidence Evaluation Agent**  
  利用 LLM 对检索到的证据进行结构化解析，标注立场（support/refute/neutral）、来源权威性等属性，并输出节点级的 **veracity（真实性得分）** 和 **reliability（可靠性得分）**。

- **Argument Tree Aggregation Algorithm**  
  构建动态论证树，对无法可靠判断的节点自动分解为子问题（如 who, what, when, where, why, how），并通过 bottom-up 聚合机制更新根节点判断，直到满足收敛条件。

此外，ToE 输出完整的 **可追溯 argument tree**，提供端到端的解释链，增强决策透明度。

### 相比现有方法的优势
| 维度 | ToE 的优势 |
|------|-----------|
| **鲁棒性** | 显著提升在 GEO-poisoned 输入下的抗干扰能力，优于仅依赖静态提示或单轮检索的方法 |
| **可解释性** | 生成结构化的 argument tree，支持人类审查整个推理过程 |
| **动态适应性** | 检索策略随声明类型自适应调整（如科学类倾向 arXiv，政治类倾向 PolitiFact） |
| **理论保障** | 将检索建模为 POMDP，推导出误差上界，证明策略收敛至近似最优解 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **LIAR**：12,836 条政治新闻短句，带细粒度真值标签（六类）
- **PolitiFact**：21,152 条专家验证的政治声明（2008–2022）
- **Check-COVID**：1,504 条与新冠疫情相关的声明，以科学文献为依据标注
- **AdvFact（本文构建）**：对抗性数据集，包含 200 条经 **FakeGPT** 和 **PoisonedRAG** 攻击增强的虚假声明，用于测试模型鲁棒性

所有数据集统一映射为三分类标签：`True`, `False`, `Uncertain`

### 实验设置和评估指标
- **Backbone LLMs**：
  - DeepSeek-V3.2（大规模闭源模型）
  - gpt-oss-20b（小型开源推理模型）
- **评估方式**：
  - 每个数据集随机采样 50 条 claim，重复 10 次取中位数准确率
  - 所有方法均只使用输入 claim 进行判断，确保公平比较
- **ToE 参数设定**：
  - 最大树深：5
  - 最大迭代次数：20
  - 决策阈值：veracity ∈ [0.3, 0.7] → Uncertain；<0.4 → False；>0.6 → True
  - 训练仅在 LIAR 训练集上完成，其余数据集零样本迁移测试

### 基线方法对比
| 方法 | 类型 |
|------|------|
| **Direct** | 直接 prompt LLM 判断，无外部检索 |
| **Z-CoT / DefGen** | Zero-shot Chain-of-Thought / Deductive Generation，结构化推理但无检索 |
| **AFaCTA** | 多智能体投票机制 |
| **TELLER** | 基于维度拆解的可信检测框架 |
| **STEEL** | 多轮检索增强策略 |
| **AdSent** | 情感中立化改写后判断 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table II）

| Method | LIAR | PolitiFact | Check-COVID | AdvFact |
|--------|------|------------|-------------|---------|
| **DeepSeek-V3.2** | | | | |
| Direct | 0.38 | 0.32 | 0.28 | 0.16 |
| Z-CoT | 0.42 | 0.60 | 0.52 | 0.36 |
| TELLER | 0.46 | 0.68 | 0.56 | 0.40 |
| **ToE** | **0.62** | **0.72** | **0.60** | **0.64** |
| **gpt-oss-20b** | | | | |
| Direct | 0.24 | 0.40 | 0.26 | 0.08 |
| Z-CoT | 0.34 | 0.22 | 0.44 | 0.20 |
| TELLER | 0.60 | 0.64 | 0.52 | 0.32 |
| **ToE** | **0.68** | **0.62** | **0.48** | **0.54** |

> ✅ **结论**：ToE 在多个 backbone 和数据集上均显著领先，尤其在对抗性数据 **AdvFact** 上表现突出（+24% vs best baseline），说明其具备强鲁棒性。

### 与基线方法的对比结果
- ToE 在 **AdvFact** 上比最强基线（TELLER）高出 **24个百分点**（0.64 vs 0.40）
- 即使在小模型 gpt-oss-20b 上，ToE 仍保持高精度，表明其不依赖强大 parametric memory
- Direct 方法有一定基础准确率，说明部分 claim 可从预训练记忆中直接回答，但面对新/对抗性内容时迅速失效

### 消融实验结果（Table III：检索工具空间消融）

| 设置 | Overall Acc. | TRUE | FALSE | UNCERTAIN | Avg. Steps |
|------|--------------|------|-------|-----------|------------|
| **Full Tool Space** | **80.0%** | 91.6% | 75.0% | 66.7% | 5.4 |
| w/o Academic (arXiv) | 73.3% | 75.0% | 83.3% | 50.0% | 6.8 |
| w/o Counter-Evidence | 63.3% | 83.3% | 50.0% | 50.0% | 4.0 |
| **w/o Factcheck (Wiki + PolitiFact)** | **56.6%** | 50.0% | 75.0% | 33.3% | 7.8 |
| w/o Social Media | 63.3% | 66.6% | 83.3% | 16.6% | 3.8 |

> 🔍 **发现**：
> - 移除 **fact-checking 平台** 影响最大（↓23.4%），说明高质量结构化核查记录是最关键的信息源
> - 缺少 **counter-evidence search** 导致假声明识别率暴跌（↓25%），凸显主动寻找反证的重要性
> - 各工具类别互补性强，单一来源不足以支撑稳健判断

---

## 4. 关键结论和发现

### 论文的主要发现
1. **ToE 能有效抵御 GEO-poisoned 攻击**，在对抗性数据集 AdvFact 上远超现有方法，验证了其在现实威胁场景中的实用性。
2. **动态多源检索 + 层次化论证结构** 是提升事实核查系统鲁棒性的关键设计。
3. **强化学习驱动的 retrieval agent 学会了领域感知的搜索策略**，例如：
   - 科学声明偏好 arXiv（26.4%）
   - 政治/健康类依赖 Factcheck（>34%）
   - 名人八卦倾向 Social Media（24.5%）
4. **argument tree 提供完整可解释路径**，支持人工审计与信任建立。
5. **理论分析表明**：在 reliability 函数满足 submodularity 的假设下，贪婪策略可达最优策略的 $(1 - 1/e)$ 性能下限，且奖励信号偏差有严格上界控制（Theorem 1）。

### 方法的局限性
- **计算开销较高**：每条 claim 需多次检索与 LLM 推理，延迟高于轻量级方法
- **依赖高质量 evidence parsing**：若 cleaning LLM 未能正确提取相关语句，会影响后续评分
- **tree expansion 依赖 LLM 分解能力**：若 decomposition LLM 产生无关或冗余子 claim，可能导致无效分支膨胀
- 当前未处理图像或多模态虚假信息

### 未来工作方向
- 引入 **early termination 机制** 优化效率
- 探索 **multi-modal evidence integration**（如图片、视频元数据）
- 设计 **更高效的 tree pruning 策略**
- 将 ToE 应用于实时舆情监控与社交平台 content moderation 系统
- 结合 **Byzantine-resilient aggregation** 抵御内部恶意 agent 攻击（参考 [7]）

--- 

> 📌 **总结一句话**：  
> **ToE 是首个将动态多源证据检索、强化学习控制与层次化论证树结合的可解释事实核查框架，在理论保障与实验验证层面均展现出卓越的准确性与鲁棒性，特别是在对抗 AI 生成虚假信息方面具有重要应用前景。**

</details>

---

### 15. [Understanding Rollout Error in Graph World Models](https://arxiv.org/abs/2606.27780)

**Authors**: Xinyuan Song, Zekun Cai  
**Category**: cs.AI  
**Published**: 2026-06-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.27780v1  

#### Abstract
World models are often used for planning by rolling learned dynamics forward. Many planning environments, however, are not vectors or images; they are graphs of agents, tools, skills, routes, and dependencies. In these settings, a local prediction error may stay local or spread through the graph, an...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Understanding Rollout Error in Graph World Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文研究了**Graph World Models (GWMs)** 在长时程规划中的 **rollout error**（展开误差）问题。传统 World Models 多用于向量或图像状态，而现实世界中许多环境（如多智能体系统、技能图、通信网络）本质上是图结构。在这些图环境中，预测误差不仅会累积，还会通过图拓扑传播，导致规划失败。

现有理论基于标量 **Lipschitz 常数**分析误差累积，无法刻画图结构对误差传播的影响，尤其在边也动态变化（dynamic-edge）的情况下，节点与边的预测误差相互耦合，问题更加复杂。

### 提出的新方法与新思路
1. **统一的 GWM 框架**  
   提出了一个支持 **fixed-edge (FE)** 和 **dynamic-edge (DE)** 两种模式的 GWM 框架，支持节点、边和图层级的决策，并引入 **action node** 统一表示动作。

2. **图感知的 rollout error 理论**  
   - 定义 **Graph Error Amplification Factor (GEAF)**：  
     $$
     \text{GEAF} = \rho(A) \cdot \|W\|_2^\ell
     $$
     其中 $\rho(A)$ 是邻接矩阵的谱半径（反映拓扑放大能力），$\|W\|_2^\ell$ 是模型权重的谱范数乘积（反映模型放大能力）。该因子将**拓扑诱导放大**与**模型诱导放大**分离。
   - 提出 **联合节点-边误差算子 B**（joint node-edge operator），用于建模 DE 场景下节点与边误差的耦合传播。

3. **Error-Aware GWM**  
   一种提升长时程稳定性的训练方法，结合三种正则化：
   - **Spectral Regularization**：控制 $\|W\|_2$，降低模型放大。
   - **Rollout Consistency**：直接惩罚多步预测漂移。
   - **Critical-Node Weighting**：对中心性高的节点赋予更高权重，防止局部错误引发全局不稳。

### 相比现有方法的优势
- **理论层面**：首次将图谱性质（spectral radius）与 rollout error 明确关联，提出可解释的 GEAF 指标。
- **方法层面**：Error-Aware GWM 在保持高预测精度的同时，有效防止长时程发散，优于仅用梯度裁剪、权重衰减等简单稳定化手段。
- **适用性**：框架适用于多种 GNN 架构（GCN, MPNN, GPS, ActionNode GWM），具有通用性。

---

## 2. 核心实验方法和设置

### 数据集
1. **合成图数据集**（Synthetic Benchmarks）
   - 生成 7 类典型图结构（各 3 个种子）：Chain, Tree, Grid, Small-World, Scale-Free, Star, Complete
   - 节点数 $N=50$，特征维度 8
   - 覆盖不同谱半径（$\rho(A)$ 从 2.0 到 49.0），测试拓扑影响

2. **异构代理图测试平台**（Heterogeneous Agent-Graph Testbeds）
   - **Agent Calling-Tree Testbed**：模拟任务执行流程图，含 Planner, Executor, Validator 等角色，评估最终成功率 $sr_{sink}$
   - **Platform Skill-Graph Testbed**：技能依赖图，评估技能成功率 $sr_{skill}$

3. **真实世界基准**
   - **Cora / Citeseer**：静态节点分类
   - **Bitcoin-Alpha**：时间链接预测

### 实验设置与评估指标
| 设置项 | 描述 |
|--------|------|
| **Rollout Horizon** | $H = 1, 2, 4, 8, 16, 32$（部分至 48） |
| **评估指标** | - `NodeMSE@H`：H 步预测均方误差<br>- `GrowthSlope`：log-NodeMSE 曲线斜率（衡量增长速率）<br>- `Planning Regret`：与最优策略的回报差距<br>- `Diverged`：是否出现误差爆炸（>1.0） |

### 基线方法对比
共比较 6 种 baseline：
- **B1**: MLP-WM（无图结构）
- **B2**: Vanilla GCN-WM
- **B3**: MPNN-WM
- **B4**: GPS（Graph Transformer）
- **B5**: ActionNode GWM（Feng et al., 2025）
- **B6**: **Error-Aware GWM**（本文方法）

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）拓扑显著影响 rollout error 增长
- **GEAF 与 GrowthSlope 高度正相关**（Fig. 4）：高 GEAF 图（如 Complete, Star）误差增长更快。
- **Scale-Free 和 Star 图上，Vanilla GCN 出现严重发散**（NodeMSE → 10⁹），而 Error-Aware GWM 始终维持低误差（~10⁻⁴）（Fig. 10）。

#### （2）Error-Aware GWM 显著提升稳定性
| 方法 | Diverged (21 cells) | Median NodeMSE@32 | 相对误差倍数 |
|------|-------------------|------------------|------------|
| Vanilla GCN | 10/21 | 0.995 | 6934× |
| GCN + Weight Decay | 4/21 | 1.091 | 4172× |
| GCN + Spectral Norm | 4/21 | 0.364 | 1690× |
| **Error-Aware GWM** | **0/21** | **2.67×10⁻⁷** | **1×** |

> ✅ **结论**：简单正则化可减少发散，但无法同时保证精度；Error-Aware GWM 实现零发散且保持高精度。

#### （3）Dynamic-Edge 训练至关重要
- 在 DE 环境中，使用 FE 数据训练的模型性能远差于 DE 训练模型：
  - **ActionNode GWM**：DE 训练比 FE 训练提升 **9.85×**
  - **Error-Aware GWM**：提升 **12.29×**（Table 3）
- 所有 108 个 DE 模型均满足 $p(B) > \max(L_x, M_A)$，验证了节点-边耦合的存在。

#### （4）真实世界任务表现
| 任务 | 结果 |
|------|------|
| **静态节点分类 (Cora)** | GCN 达 80.6%，Error-Aware GWM 为 75.0%，仍落后约 5.6 pp |
| **时间链接预测 (Bitcoin-Alpha)** | FE-GWM 最佳 AP (0.820)，Static GCN 最佳 AUC (0.774)，JODIE 最佳 MRR (0.261)，无单一胜者 |

> 📌 **结论**：GWMs 并非所有图任务的最优解，在静态或稀疏预测任务上，专用模型更强。

---

## 4. 关键结论和发现

### 主要发现
1. **Rollout error 受图拓扑显著影响**  
   高谱半径图（如星形、完全图）更容易放大误差，**GEAF 是误差增长速率的有效指标**。

2. **Fixed-Edge 与 Dynamic-Edge 是不同机制**  
   - FE 模型误差传播为纯节点过程，$M_x=0$ 导致耦合关闭。
   - DE 模型激活节点-边双向耦合，必须专门训练才能获得良好性能。

3. **Error-Aware GWM 有效防止长时程发散**  
   通过 **spectral regularization + rollout consistency + critical-node weighting** 三者结合，在不牺牲精度的前提下实现稳定 rollout。

4. **GWMs 的适用边界明确**  
   - ✅ **优势场景**：动态图 rollout、代理规划、结构演化环境
   - ❌ **劣势场景**：静态节点分类、稀疏时间链接预测

### 局限性
1. 实证最强证据来自**合成模拟器**，在真实复杂系统中泛化能力有待验证。
2. **GEAF 作为 $p(B)$ 的代理仅在信号主导区域成立**，未完全验证其在 DE 场景下的普适性。
3. 当前 correction policy 中，**local GEAF 退化为 degree centrality**，缺乏更细粒度优先级信号。

### 未来工作方向
- 将框架应用于 **LLM-based 多智能体系统** 和 **实时网络规划**
- 设计 **activation-aware 或 uncertainty-aware 的 local GEAF**
- 探索 **memory-augmented GWMs** 以支持更长时程规划
- 在更大、更密集的真实时间图上验证 DE-GWM 性能
- 研究零样本拓扑迁移（zero-shot topology shift）的鲁棒性

---

> 🔗 **代码开源**：https://github.com/Hik289/graph_world_model_accumulative_error.git  
> 📄 **论文链接**：https://arxiv.org/abs/2606.27780

</details>

---

### 16. [Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs](https://arxiv.org/abs/2606.27378)

**Authors**: Fahd Seddik, Fatemeh Fard  
**Category**: cs.CL  
**Published**: 2026-06-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.27378v1  

#### Abstract
We introduce an axiomatic evaluation framework for latent thought representations in LLMs, comprising metrics that are independent of downstream benchmark scores and reveal representational failures that benchmark accuracy masks. Existing evaluations conflate representation quality with model capaci...

---

### 17. [Prism Transformer: Progressive Head Schedules for Hierarchical Attention Processing](https://arxiv.org/abs/2606.27449)

**Authors**: Shubham Aggarwal  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.27449v1  

#### Abstract
Multi-head attention conventionally partitions the hidden dimension equally across all heads at every layer, enforcing an identical representational subspace dimension (dh = dmodel/h) throughout the models depth. In this work, we identify this uniform allocation as a fundamental structural bottlenec...

---

### 18. [COOPA: A Modular LLM Agent Architecture for Operations Research Problems](https://arxiv.org/abs/2606.27611)

**Authors**: Chuanhao Li, Xiaoan Xu, Dirk Bergemann, Ethan X. Fang, Yehua Wei, Zhuoran Yang  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.27611v1  

#### Abstract
Operations Research (OR) provides a rigorous framework for high-stakes decision-making, but effective OR modeling requires substantial domain knowledge, mathematical abstraction, and solver expertise. Recent LLM-based systems automate parts of this pipeline, yet remain limited by low accuracy on com...

---

### 19. [TA-SparseMG: Trend-Aware Sparse Forecasting via Multi-Scale Gating for Long-Term Time Series](https://arxiv.org/abs/2606.27908)

**Authors**: Wenchao Liu, Hongbing Wang, Youji Zhu, Xiaodong Liu, Xiangguang Xiong  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.27908v1  

#### Abstract
Long-term time series forecasting finds extensive applications in domains such as power demand, traffic flow, meteorological observation, and renewable energy dispatch. Forecasting dynamically varying long-term time series poses inherent challenges, including statistical nonstationarity, local high-...

---

### 20. [Towards Reliable and Robust LLM Planning: Symbolic Feedback-Driven Iterative Self-Refinement Framework](https://arxiv.org/abs/2606.27757)

**Authors**: Jiajing Zhang, Jiamei Jiang, Chenyang Zhang, Feifei Mo, Linjing Li, Daniel Zeng  
**Category**: cs.AI  
**Published**: 2026-06-29  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.27757v1  

#### Abstract
Large language models (LLMs) have attracted widespread attention from academia and industry, yet their deployment raises critical security concerns regarding robustness and reliability. Planning, a core component of intelligent behavior, remains challenging for LLMs, which often produce infeasible o...

---

### 21. [Tandem Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2606.28166)

**Authors**: Difan Jiao, Raghav Singhal, Robert West, Ashton Anderson  
**Category**: cs.AI  
**Published**: 2026-06-29  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.28166v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has significantly improved the reasoning capability of large language models, reaching expert or even superhuman performance in domains such as competition math. However, whether weaker agents and humans can actually harness this capability is fa...

---

### 22. [Masked Language Flow Models](https://arxiv.org/abs/2606.27617)

**Authors**: Iskander Azangulov, Kianoosh Ashouritaklimi, Leo Zhang, Simon Vary, Patrick Rebeschini  
**Category**: cs.CL  
**Published**: 2026-06-29  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.27617v1  

#### Abstract
Masked Diffusion Models (MDMs) promise fast, parallel language generation, but their reverse transition factorises across token positions -- an approximation that breaks down in the few-step sampling regime where parallel generation ought to provide the greatest efficiency gains. Flow Language Model...

---

### 23. [Mitigating Position Bias in Transformers via Layer-Specific Positional Embedding Scaling](https://arxiv.org/abs/2606.27705)

**Authors**: Changze Lv, Zhenghua Wang, Yiran Ding, Yixin Wu, Tianlong Li, Zhibo Xu, Muling Wu, Tianyuan Shi, Shizheng Li, Qi Qian, Xuanjing Huang, Xiaoqing Zheng  
**Category**: cs.CL  
**Published**: 2026-06-29  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.27705v1  

#### Abstract
Large Language Models (LLMs) still struggle with the ``lost-in-the-middle'' problem, where critical information located in the middle of long-context inputs is often underrepresented or lost. While existing methods attempt to address this by combining multi-scale rotary position embeddings (RoPE), t...

---

### 24. [Vision-Default, Prior-Override: Causal Mechanisms of Perception-Knowledge Conflict in Vision-Language Models](https://arxiv.org/abs/2606.28273)

**Authors**: Niclas Lietzow, Danielle Bitterman, Carsten Eickhoff, William Rudman, Michal Golovanevsky  
**Category**: cs.CL  
**Published**: 2026-06-29  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.28273v1  

#### Abstract
Vision-language models must reconcile visual evidence with memorized world knowledge when the two conflict. How they resolve this conflict shapes the reliability of multimodal systems, yet prior work characterizes it behaviorally without a component-level causal account. We combine activation patchi...

---

### 25. [NormGuard: Reward-Preserving Norm Constraints in Flow-Matching Reinforcement Learning](https://arxiv.org/abs/2606.27771)

**Authors**: Tianlin Pan, Lianyu Pang, Cheng Da, Huan Yang, Changqian Yu, Kun Gai, Wenhan Luo  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.27771v1  

#### Abstract
Reinforcement learning (RL) post-training improves the reward alignment of flow-based generators, but often degrades perceptual quality in ways that are not captured by the reward proxy. We identify a simple structural signature of this drift: across three post-training methods (NFT, AWM, DPO), RL f...

---

### 26. [Dual-Learning based Penalized Multi-Align Clustering for Multi-View Incomplete and Disorderly Data](https://arxiv.org/abs/2606.27984)

**Authors**: Liang Zhao, Shubin Ma, Bo Xu, Qingchen Zhang  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.27984v1  

#### Abstract
Multimodal feature fusion can effectively capture complex patterns in real-world data by integrating complementary information from different modalities. However, in many applications, such as boiler combustion monitoring, equipment failure, inconsistent sensor sampling frequencies, and network dela...

---

### 27. [Lifted Causal Inference](https://arxiv.org/abs/2606.28024)

**Authors**: Malte Luttermann, Tanya Braun, Ralf M\"oller, Marcel Gehrke  
**Category**: cs.AI  
**Published**: 2026-06-29  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2606.28024v1  

#### Abstract
Lifted inference exploits indistinguishabilities in probabilistic graphical models by using a representative for indistinguishable objects, thereby speeding up query answering while maintaining exact answers. In this article, we show how lifting can be applied to efficiently compute causal effects i...

---

### 28. [Recovering Sharp Conductivity Features in the Finite-Data Calder\'on Problem with Physics-Informed Neural Networks](https://arxiv.org/abs/2606.28158)

**Authors**: Ali AlHadi Kalout, Pablo Tejerina-P\'erez, Konstantin Karchev, Pedro Taranc\'on-\'Alvarez, Leonid Sarieddine, Raul Jimenez, Max Engelstein, Guy David  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2606.28158v1  

#### Abstract
Physics-informed neural networks (PINNs) have recently emerged as a promising framework for addressing the Calder\'on inverse problem from limited boundary data. In this work, we revisit neural Calder\'on inversion by introducing multiscale boundary excitations based on randomized wavelet functions ...

---

### 29. [VGB for Masked Diffusion Model: Efficient Test-time Scaling for Reward Satisfaction and Sample Editing](https://arxiv.org/abs/2606.28301)

**Authors**: Kijung Jeon, Thuy-Duong Vuong, Molei Tao  
**Category**: cs.LG  
**Published**: 2026-06-29  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2606.28301v1  

#### Abstract
Inference-time scaling is a promising paradigm to improve generative models, especially when outputs must satisfy structural constraints or optimize downstream rewards. We consider Masked Diffusion Model (MDM) and introduce MDM-VGB, a discrete diffusion sampler that augments unmasking generation wit...

---

### 30. [DysLexLens: A Low-Resource LLM Framework for Analysing Dyslexic Learners Insights from Online Forums](https://arxiv.org/abs/2606.27619)

**Authors**: Dana Rezazadegan, Atie Kia, Phongpadid Nandavong, Dominique Carlon, Jeremy Nguyen, Abhik Banerjee, James Marshall, Anthony McCosker, Yong-Bin Kang  
**Category**: cs.AI  
**Published**: 2026-06-29  
**Score**: 3.0  
**Type**: new  
**ArXiv ID**: 2606.27619v1  

#### Abstract
Dyslexic learners increasingly use artificial intelligence (AI) tools to support reading, writing, organisation, and study-related tasks. However, their lived experiences with these tools remain largely underexamined. This paper proposes DysLexLens, a low-resource LLM framework, designed to analyse ...

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

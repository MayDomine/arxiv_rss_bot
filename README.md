# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-16 07:43:33 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts](https://arxiv.org/abs/2607.12696)

**Authors**: Jincheng Xie, Runheng Liu, Heyan Huang, Yawen Ling, Hanbin Dai, Yu Zheng, Wen Hu  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2607.12696v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) models have become an important approach for scaling Large Language Models (LLMs), but their inference efficiency depends strongly on expert activation patterns. Speculative decoding (SD) accelerates autoregressive generation by verifying multiple draft tokens in para...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在大规模 **Sparse Mixture-of-Experts (MoE)** 模型中，传统的 **Speculative Decoding (SD)** 虽然能通过并行验证多个 draft token 来加速推理，但其 draft selection 策略通常只优化 **acceptance likelihood（接受率）**，而忽略了 MoE 架构特有的 **expert activation cost**。

作者观察到一个关键现象：**confidence-driven draft selection** 可能导致 **expert scattering（专家散射）** —— 即高置信度的 draft token 路由到互不重叠的专家集合，从而在一次 speculative step 中激活大量不同的专家，显著增加 **HBM（High Bandwidth Memory）访问开销** 和验证延迟。

> 💡 **核心矛盾**：更高的 acceptance likelihood ≠ 更高的 end-to-end 速度，因为验证成本随激活专家数量线性增长。

---

### 🚀 提出的新方法：**EcoSPEC**

提出 **EcoSPEC** —— 一种 **cost-aware speculative decoding** 框架，专门针对 MoE 模型的内存瓶颈进行优化。

#### 创新设计包括：

1. **轻量级 Expert Predictor (II₀)**  
   - 使用一个小模型预测每个 draft token 在目标 MoE 模型中可能激活的专家集合。
   - 不依赖实时查询目标模型，避免引入额外计算开销。

2. **动态 Expert Buffer (B)**  
   - 维护当前已选 draft tokens 所覆盖的专家集合。
   - 用于动态计算新增候选路径带来的 **边际专家成本（marginal expert cost）**。

3. **Cost-Aware Draft Selection 算法**  
   - 在选择验证节点时，综合考虑两个因素：
     - **累计 draft 概率（P(t)）**：保持高接受率
     - **边际专家成本 ΔCost(t|B)**：鼓励复用已有专家
   - 得分函数为：
     $$
     S(t) = \frac{P(t)}{\Delta\text{Cost}(t|B) + \epsilon}
     $$
   - 该策略在不改变标准 lossless verification 规则的前提下，实现更高效的 draft 树剪枝。

---

### 🔍 相比现有方法的优势

| 方面 | 传统 SD 方法（如 EAGLE/MTP） | EcoSPEC |
|------|-------------------------------|---------|
| **目标** | 最大化 acceptance length | 平衡 acceptance 与 expert cost |
| **成本感知** | ❌ 忽略 expert memory traffic | ✅ 显式建模边际专家成本 |
| **验证效率** | 高接受率但可能高内存开销 | 接受率相近，但验证更快 |
| **兼容性** | 通用 | 专为 MoE 设计，可与运行时优化（如 prefetching）结合 |

> ✅ **关键优势**：在几乎不牺牲 acceptance length 的前提下，显著降低每步激活的专家数量，提升端到端解码速度。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

涵盖多种任务类型，共 **7 个基准**：

| 数据集 | 类型 | 描述 |
|--------|------|------|
| **GSM8K** | 数学推理 | 小学数学应用题，5-shot CoT |
| **AIME-25** | 数学竞赛 | AIME 级别难题，zero-shot |
| **MATH500** | 数学 | MATH 数据集子集，zero-shot |
| **AMC22-24** | 数学 | AMC 竞赛题，zero-shot |
| **HumanEval** | 编程 | Python 函数生成与测试 |
| **MT-Bench** | 对话 | 多轮指令遵循 |
| **MMStar** | 文本生成 | 原为 VLM，本文仅用文本部分 |

> 所有实验均在相同 prompt 和 decoding 设置下进行，确保公平比较。

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| **目标模型** | DeepSeek-V3.1 (671B), Qwen3-235B-A22B, GPT-OSS-120B |
| **MoE 配置** | Top-8 或 Top-4 路由，每层 128–256 专家 |
| **硬件平台** | 8× NVIDIA H200 GPUs |
| **批大小（batch size）** | 默认 1（消融实验扩展至 2/4/8） |
| **验证预算（verification budget γ）** | 默认 4（含 bonus token） |
| **温度设置** | Greedy (T=0) 和 Sampling (T=1) 均测试 |

---

### 📊 评估指标

| 指标 | 含义 |
|------|------|
| **Speedup (Spd↑)** | 相对于 autoregressive decoding 的端到端吞吐加速比 |
| **Mean Accept Length (α)** | 每次 speculative step 成功接受的 token 数量 |
| **Avg. Active Experts (E↓)** | 每 MoE 层平均激活的独特专家数（越低越好） |
| **HBM Read Traffic** | 验证阶段估计的 HBM 读取总量（GB） |
| **Latency Breakdown** | 分解 T_pred（预测）、T_draft（生成）、T_verify（验证）耗时 |

---

### 🆚 基线方法对比

| 模型 | 基线方法 |
|------|----------|
| DeepSeek-V3.1 | MTP（Multi-Token Prediction） |
| Qwen3-235B-A22B & GPT-OSS-120B | EAGLE-3 |
| 额外对比 | GTO（Group Tree Optimization） |

> 所有 baseline 与 EcoSPEC 共享相同的 draft generation 流程，仅在 **draft selection 阶段** 引入差异，保证可控比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（T=0, greedy decoding）

| 模型 | 方法 | Speedup | Accept Len | Avg. Experts (E) |
|------|------|---------|------------|------------------|
| **Qwen3-235B-A22B** | EAGLE-3 | 1.22× | 2.41 | 23.7 |
| | **EcoSPEC** | **1.36×** | 2.32 | **20.5** |
| | **↑ +0.14×** | ≈ | **↓ -3.2** |
| **GPT-OSS-120B** | EAGLE-3 | 1.14× | 1.88 | 11.6 |
| | **EcoSPEC** | **1.31×** | 1.86 | **10.6** |
| | **↑ +0.17×** | ≈ | **↓ -1.0** |
| **DeepSeek-V3.1** | MTP | 1.10× | 2.85 | 31.4 |
| | **EcoSPEC** | **1.15×** | 2.79 | **31.2** |
| | **↑ +0.05×** | ≈ | **↓ -0.2** |

> ✅ **最高加速达 1.62×（Qwen3-235B 上 MT-Bench）**

---

### 💾 HBM 内存流量减少（Table 2）

| 模型 | Baseline HBM | EcoSPEC HBM | 减少量 |
|------|--------------|-------------|--------|
| Qwen3-235B-A22B | 99.3 GB | 88.1 GB | ↓ **11.2 GB (-11.3%)** |
| GPT-OSS-120B | 6.0 GB | 5.5 GB | ↓ **0.5 GB (-8.0%)** |
| DeepSeek-V3.1 | 97.3 GB | 96.8 GB | ↓ **0.5 GB (-1.0%)** |

> 即使专家减少不多（如 DeepSeek），由于单个专家体积大（44MB），仍带来可观节省。

---

### ⏱️ 延迟分解（Table 3）

| 模型 | 方法 | T_verify | T_total | ΔT_verify |
|------|------|--------|--------|----------|
| Qwen3-235B | EAGLE-3 | 0.832s | 0.840s | — |
| | EcoSPEC | **0.730s** | **0.742s** | ↓ **0.102s** |
| GPT-OSS-120B | EAGLE-3 | 0.113s | 0.148s | — |
| | EcoSPEC | **0.090s** | **0.129s** | ↓ **0.023s** |

> ✅ **验证时间显著下降**，而 predictor 开销仅约 **4ms**，可忽略。

---

### 🔬 消融实验结果（Ablation Study）

#### ✅ 边际成本 vs 全局成本（Table 4）

| 模型 | 策略 | Speedup | Accept Len | E |
|------|------|---------|-----------|----|
| Qwen3-235B | Global Cost | 1.28× | 2.21 | 21.5 |
| | **Marginal Cost (EcoSPEC)** | **1.39×** | **2.54** | **21.0** |

> 动态更新 buffer 的 marginal cost 更有效，能更好识别可复用专家的路径。

#### ✅ 预测器准确性影响（Table 9）

| 模型 | Predictor Acc | Speedup (MTBench) |
|------|---------------|-------------------|
| Qwen3-235B | ~50% | 1.20× |
| | ~60% | 1.45× |
| | **82% (converged)** | **1.62×** |

> 预测器质量直接影响性能，但即使低精度也能提供正向增益。

#### ✅ Oracle 分析（Table 10）

使用真实 expert 路由作为 oracle 输入，结果与训练好的 predictor 几乎一致，说明当前 predictor 已足够准确。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Expert scattering 是 MoE speculative decoding 的关键瓶颈**  
   - 高置信度 ≠ 高效率，需考虑路由局部性。

2. **显式建模 marginal expert cost 可显著提升效率**  
   - 通过轻量预测 + 动态缓冲机制，在不修改验证逻辑的情况下实现更优选择。

3. **EcoSPEC 在多种 MoE 模型上一致有效**  
   - 在 DeepSeek、Qwen、GPT-OSS 上均取得加速，最高达 **1.62×**。

4. **性能提升主要来自降低验证成本，而非提高接受率**  
   - Acceptance length 基本持平，但 **T_verify 显著下降**。

5. **对 batch size 扩展友好**  
   - 在大 batch 下，EcoSPEC 仍能维持更低的 active expert footprint，延缓性能衰减。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖 expert predictor 训练** | 需离线收集路由轨迹并微调 predictor，增加部署复杂度 |
| **收益受 MoE 路由模式影响** | 如 DeepSeek-V3.1 路由较均衡，reuse 空间小，增益有限 |
| **未改变验证时的专家加载策略** | 与 MoE-Spec 等 budgeting 方法正交，但本身不 prefetch 或 offload |

---

### 🔮 未来工作方向

1. **联合优化 draft generation 与 cost-aware selection**  
   - 当前 EcoSPEC 作用于 selection 阶段，未来可将 cost signal 注入 draft model 训练过程。

2. **探索更高效的 predictor 架构**  
   - 如共享 embedding、蒸馏压缩等，进一步降低 predictor 开销。

3. **与 MoE 运行时系统深度集成**  
   - 结合 expert caching、prefetching、offloading 等技术，形成完整优化栈。

4. **扩展至其他稀疏架构**  
   - 如 Blockwise MoE、Hierarchical MoE 等，验证 cost-aware 思路的普适性。

---

## ✅ 总结

**EcoSPEC** 是首个明确提出并解决 **MoE 模型中 speculative decoding 的 expert scattering 问题** 的工作。它通过引入 **轻量专家预测器** 和 **动态成本感知选择机制**，实现了在几乎不损失接受率的前提下，显著减少每步激活的专家数量，从而降低 HBM 流量与验证延迟，最终提升端到端解码速度。

> 🌟 **一句话总结**：  
> _“少激活专家，更快解码” —— EcoSPEC 将 MoE 的硬件特性纳入 speculative decoding 的决策过程，走出了一条面向实际部署的高效推理新路径。_

</details>

---

### 2. [Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning](https://arxiv.org/abs/2607.12395)

**Authors**: Xinyu Tang, Gangqiang Cao, Yurou Liu, Yuliang Zhan, Xiaochong Lan, Yifan Li, Yuchen Yan, Han Peng, Zican Dong, Zhenduo Zhang, Tianshu Wang, Xinyu Kong, Zujie Wen, Wayne Xin Zhao, Zhiqiang Zhang, Jun Zhou  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.12395v1  

#### Abstract
Reinforcement learning with verifiable rewards without human-annotated data, often referred to as zero RL, has emerged as a powerful paradigm for eliciting chain-of-thought reasoning. However, due to computational constraints, existing studies are largely restricted to small models, leaving the trai...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
该论文旨在探索**零监督强化学习（Zero RL）在超大规模模型（万亿参数级别）上的训练动态和涌现能力**。现有研究受限于计算资源，多集中于小规模模型（如百亿参数），导致对大规模下 Zero RL 的行为、稳定性及认知策略演化缺乏理解。

具体挑战包括：
- **可读性差**：生成的 Chain-of-Thought（CoT）推理链冗长、无结构。
- **长度偏差（Length Bias）**：标准算法（如 GRPO）隐含鼓励更长输出，导致 token 冗余。
- **固定推理深度**：单一响应模式无法适应不同难度任务。
- **训练不稳定**：训练引擎（Megatron）与推理引擎（SGLang）间的数值差异引发崩溃。

### **提出了什么新方法或新思路**
作者提出了一套**稳定且高效的万亿参数自迭代训练框架 Ring-Zero**，其核心是“极简主义”设计原则——仅引入必要优化以实现高质量推理涌现。

#### 主要方法创新：
- **Clipped Importance Sampling + Ratio Correction**  
  修正训练-推理引擎间因浮点精度差异导致的 logit 不一致，防止梯度爆炸。
- **Training-Inference Ratio Correction**  
  在重要性采样比中使用训练引擎的 logits 作为分子，消除系统级数值鸿沟。
- **Mixed-Precision Control**  
  将 Attention Softmax 和 LM Head 保持为 FP32，其余主体用 BF16，提升数值稳定性。
- **Multi-Stage Training Pipeline**  
  四阶段流程：
  1. **第一阶段 RL**：Token-level loss 初步激发推理行为。
  2. **Self-Distillation**：压缩并清理冗余推理路径，重置训练-推理差距。
  3. **第二阶段 RL**：切换至 Sample-level loss 防止长度膨胀。
  4. **第三阶段 RL**：Tier-based training 实现自适应推理深度。

- **Emergent Cognitive Behaviors**  
  模型自发涌现出无需人工设计的高级思维机制，验证了“苦涩教训”（bitter lesson）。

### **相比现有方法的优势**
| 维度 | Ring-Zero | 先前方法（如 DeepSeek-R1） |
|------|-----------|-----------------------------|
| 模型规模 | **1T 参数 MoE** | 多为 <100B |
| 是否依赖 SFT 数据 | ❌ 完全从预训练模型启动 | ✅ 通常需 CoT 微调 |
| 推理质量评估维度 | ✅ 提出 Comprehensibility, Reproducibility, Efficiency 三维框架 | ❌ 仅关注最终答案准确率 |
| 自主认知能力 | ✅ 自发出现结构化格式、自我验证等 | ⚠️ 需人工奖励引导 |
| 可扩展性 | ✅ 极简设计支持高效万亿级训练 | ⚠️ 复杂工程阻碍规模化 |

---

## 2. 核心实验方法和设置

### **使用的数据集**
- 所有训练基于**未标注的数学问题集合**，不使用任何人工标注的 CoT 数据。
- 评估采用七个具有挑战性的数学基准：
  - **AIME 2024–2026**
  - **HMMT February & November 2025–2026**
  - **IMOAnswerBench**

### **实验设置**
- **基础模型**：
  - `Ling-2.5-1T-Base`：1 万亿参数 MoE 模型（激活 63B）
  - `Ling-2.5-flash-Base`：104B 参数 MoE 模型（激活 7.4B）
- **硬件平台**：320 × H200 GPUs
- **训练架构**：
  - 训练引擎：Megatron
  - 推理引擎：SGLang
  - 框架：Areal（分布式 RL 协调系统）
- **超参数**：
  - Batch size: 512 → 256
  - Learning rate: 2×10⁻⁶
  - Rollout 数量 G=16，温度=1.0
  - 上界 Clipping ε=5.0，无下界
  - KL Penalty β=1e-4（仅第一阶段）

### **评估指标**
#### （1）传统指标
- **Pass@1 Accuracy (%)**：64 次采样取最优一次正确率平均值。

#### （2）提出的 CoT 质量三维评估框架
| 维度 | 定义 | 测量方式 |
|------|------|---------|
| **Comprehensibility**（可理解性） | 人类是否能轻松跟随逻辑流 | LLM-as-a-Judge 成对比较（逻辑连贯性、因果显式性、无幻觉） |
| **Reproducibility**（可复现性） | 弱模型能否通过蒸馏学会相同策略 | 在 Qwen2.5-32B / Llama3.3-70B 上进行知识蒸馏后性能增益 |
| **Efficiency**（效率） | 是否简洁有效，避免冗余 | 正确解法下的平均 token 数量 |

### **基线方法对比**
- **前沿闭源模型**：
  - GLM-5.1, Qwen3.7-Plus, Kimi K2.6, Claude Opus 4.8, Gemini 3.1 Pro, GPT-5.5
- **开源 Zero RL 模型**：
  - DeepSeek-R1-Zero, DeepSeek-R1
- **消融对象**：
  - 不同 RL 算法（GRPO, DAPO, CISPO, GSPO）
  - 是否使用 KL Penalty / Ratio Correction
  - 不同格式奖励（Format A vs B）
  - 不同上下文窗口大小（16k vs 32k）

---

## 3. 主要实验结果和性能指标

### **关键性能数据**
| 模型 | AIME 2026 | HMMT Feb 2026 | IMOAnswerBench |
|------|----------|---------------|----------------|
| **Ring-2.5-1T-Zero (Second Stage RL)** | **92.5%** | **78.1%** | **72.7%** |
| GPT-5.5 | 98.3% | 96.7% | 91.4% |
| Gemini 3.1 Pro | 98.2% | 87.3% | 81.0% |
| Ring-2.5-flash-Zero | 65.3% | 50.3% | — |
| DeepSeek-R1-Zero | — | — | — |

> 注：尽管未达最先进闭源模型水平，但在完全无监督条件下表现极具竞争力。

### **与基线方法的对比结果**
- **Zero RL 从头实现强推理能力**：
  - Ring-2.5-1T-Zero（第一阶段）在 AIME 2026 达到 **84.2%**，证明无需 SFT 即可激发复杂推理。
- **多阶段训练持续提效**：
  - 第二阶段 RL 后，AIME 2026 提升至 **92.5%**，验证 pipeline 有效性。
- **自适应推理模式提供灵活权衡**：
  - High 模式（128k）达到峰值性能；
  - Low 模式（4k）仍保持 68.8% AIME 准确率，显著降低延迟与成本。

### **消融实验结果**
#### （1）RL 算法比较（Flash 模型）
- **CISPO / DAPO**：初期学习快，但熵坍缩严重，易训练失败。
- **GSPO**：维持高熵，但无法激励长序列生成。
- **GRPO + Ratio Correction + KL Penalty**：平衡速度与稳定性，最优选择。

#### （2）KL Penalty 影响
- 移除 KL 导致 log-prob 差异发散、熵归零、reward 崩溃 → **必须保留**。

#### （3）Ratio Correction 效果
- 基线（仅 SGLang logits）：800 步内崩溃。
- IcePop 方法：延缓崩溃但仍失败。
- **本文方法（Megatron/SGLang 分子分母分离）**：全程稳定训练。

#### （4）Format Reward 设计
- Format A（仅 `<think>` 开标签）：长度无限增长，reward 不升 → 存在 exploitable loophole。
- **Format B（双闭合标签 + EOS 要求）**：强制终止，防止垃圾文本膨胀。

#### （5）Context Window 影响
- 32k 窗口比 16k 平均长度翻倍，但 reward 提升微弱 → **存在严重 token 冗余**。
- 支持“长度惯性”（Length Inertia）现象：模型盲目填满可用空间。

#### （6）超参数敏感性分析
- 学习率（1e-6 ~ 3e-6）、Rollout 数量（8~32）影响不大。
- **Token-level loss** 显著促进长度增长；**Sample-level loss** 抑制长度膨胀。

---

## 4. 关键结论和发现

### **主要发现**
1. ✅ **“苦涩教训”的实证验证**（Empirical Validation of the Bitter Lesson）  
   > “简单而可扩展的方法终将超越复杂的人工启发式。”  
   - 当模型达到 **1T 规模**，原本需要精心设计的推理技巧（如结构化输出、自我验证）会**自发涌现**，无需额外奖励或模板。
   - 表明：**scale 是解锁高级认知行为的关键杠杆**。

2. 🔁 **两阶段学习过程：“发现 vs 锐化”（Discovery vs Sharpening）**
   - **Discovery Phase**：早期 pass@1024 快速上升，表明模型正在探索新的解题路径。
   - **Sharpening Phase**：后期 pass@1 持续上升而 pass@1024 饱和，说明模型在已有策略上精细化输出分布。

3. 🌀 **自发涌现五大高级认知行为**
   - **Anthropomorphism**：模拟情绪状态（“brain fart”, “genius idea”）
   - **Structured Formatting**：自动编号步骤（Step 1… Step 2…）
   - **Self-Verification**：主动回查假设、交叉验证公式
   - **Parallel Reasoning**：尝试多种解法并择优
   - **Context Anxiety**：临近 token 上限时强行猜测以防格式错误

4. 📊 **CoT 质量全面领先**
   - **Comprehensibility**：LLM-as-a-Judge 判断胜率远超 GLM、Kimi、MiniMax 等。
   - **Reproducibility**：仅用 100K 数据蒸馏，学生模型性能优于 DeepSeek-R1（使用 800K）。
   - **Efficiency**：解决同一问题平均仅需 **6,368 tokens**，不足其他模型一半。

### **方法的局限性**
- **High Mode 性能下降**：第三阶段联合训练三个 tier 导致负迁移，High mode 能力被拉低。
- **缺乏高质量超长数据**：限制了 128k 模式的上限。
- **Pretrained Priors 限制**：若预训练中缺失某些数学概念，RL 无法凭空创造。
- **Long-Tail 数据效率低**：现实世界多数问题是简单的，过度拟合这些无助于能力提升。

### **未来工作方向**
- 扩展 **context window 至 1M+**，解锁更深层数学推导。
- 构建 **动态课程学习机制**，按模型能力实时调整问题难度。
- 探索 **统一目标函数**，同时优化准确性与推理效率。
- 进一步研究 **emergent behavior 的可控性与安全性**，防止有害类人倾向扩散。

---

> 💡 **总结一句话**：  
> **Ring-Zero 证明，在足够大的 scale 下，简单的 Zero RL 加上极简优化，就能让模型自己“想明白”如何更好地思考——这不仅是技术突破，更是对 AI 发展范式的深刻启示。**

</details>

---

### 3. [FastCentNN: Accelerating Centroid Neural Network with Entropy Proxy](https://arxiv.org/abs/2607.13613)

**Authors**: Le-Anh Tran  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.13613v1  

#### Abstract
Centroid neural network (CentNN) is an unsupervised competitive learning algorithm in which centroid splitting is triggered only after strict local stabilization, often leading to prolonged low-movement training phases before model expansion. This report proposes FastCentNN, an accelerated variant t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《FastCentNN: Accelerating Centroid Neural Network with Entropy Proxy》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
标准 **Centroid Neural Network (CentNN)** 是一种基于竞争学习的无监督聚类算法，其模型通过**严格局部稳定化（strict local stabilization）**后才触发质心分裂（splitting），从而实现自适应增长。然而，这种机制导致在每次分裂前存在长时间的“低运动阶段”（low-movement phase），即质心几乎不再移动但仍需持续训练多个 epoch，造成显著的计算资源浪费。

该问题表现为：
- 大量冗余训练 epoch
- 收敛速度慢
- 实际运行效率低下

---

### ✅ 提出了什么新方法或新思路  
提出 **FastCentNN** —— 一种加速版 CentNN，其核心创新是引入 **“早期分裂策略”（early splitting strategy）**，将全局质心运动总量 $ M_t $ 作为**训练熵代理（entropy proxy）** 来衡量当前聚类不确定性。

#### 关键设计：
- 定义每轮 epoch 的总质心移动量为：
  $$
  M_t = \sum_{j=1}^{k_t} \| w_j^{(t)} - w_j^{(t-1)} \|^2
  $$
- 当 $ M_t $ 连续 $ P $ 轮低于设定阈值 $ T $ 时，即触发分裂操作（无需等待完全收敛）
- 支持两种阈值模式：
  - **绝对模式（absolute）**: $ T_t = T $
  - **阶段相对模式（stage-relative）**: $ T_t = T \cdot B_r $，其中 $ B_r $ 是上一次分裂后的基准运动水平

此策略将分裂时机从“严格收敛事件”转变为“可配置的早触发决策”，有效跳过无效训练周期。

---

### ✅ 相比现有方法的优势  
| 维度 | 优势说明 |
|------|---------|
| **效率提升** | 显著减少不必要的 reassignment epochs，缩短整体训练时间 |
| **保持性能** | 完全保留原始 CentNN 的 winner-loser 学习动态和在线自适应能力 |
| **灵活性高** | 可配置的分裂阈值支持跨数据集和训练阶段的一致行为 |
| **即插即用** | 可作为 CentNN 的直接替代方案，无需修改网络结构或更新规则 |

---

## 2. 核心实验方法和设置

### ✅ 使用了哪些数据集  

实验涵盖两类典型聚类任务：

#### （1）合成二维数据集（Synthetic 2D Datasets）
- A1, A2
- S1, S2
- R15
- Aggregation  
→ 共6个广泛使用的基准数据集，用于验证基础聚类效果与收敛行为

#### （2）高维图像数据集（High-dimensional Datasets）
- **MNIST**（手写数字图像）
- **Fashion-MNIST**（服装图像）  
→ 各取 10,000 样本，目标聚类数 $ K=10 $

---

### ✅ 实验设置和评估指标  

#### 参数设置
- 所有实验中 FastCentNN 与原始 CentNN 使用**相同参数配置**
- 最大 epoch 数、分裂尺度 $ \epsilon $、初始质心等一致
- FastCentNN 引入额外参数：
  - 移动阈值 $ T $
  - 耐心窗口 $ P $（连续低运动 epoch 数）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Runtime (s)** | 总运行时间（秒） |
| **Epochs** | 达到收敛所需的训练轮数 |
| **ΔMSE (%)** | 聚类质量差异：<br>$ \Delta\text{MSE} = \frac{\text{MSE}_{\text{FastCentNN}} - \text{MSE}_{\text{CentNN}}}{\text{MSE}_{\text{CentNN}}} \times 100\% $ |
| **Speed Up (%)** | 加速比：<br>$ \text{SpeedUp} = \frac{T_{\text{CentNN}} - T_{\text{FastCentNN}}}{T_{\text{CentNN}}} \times 100\% $ |

---

### ✅ 基线方法对比  
- **Baseline**: 原始 CentNN
- **Proposed**: FastCentNN（本文方法）
- 对比方式：相同条件下并列比较 runtime、epochs 和 MSE 表现

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Tables 1 & 2）

#### 📊 在合成 2D 数据集上的表现（Table 1）

| Dataset | Speed Up | ΔMSE (%) | Epoch Reduction |
|--------|----------|-----------|----------------|
| A1     | **16.72%** | +2.47×10⁻⁴ | 155 → 128 |
| A2     | 9.37%      | +5.18×10⁻⁴ | 274 → 246 |
| S1     | 5.54%      | 0         | 64 → 61 |
| S2     | 6.90%      | +7.53×10⁻⁴ | 110 → 102 |
| R15    | 4.18%      | 0         | 34 → 33 |
| Aggregation | 14.80% | -2.83×10⁻² | 40 → 34 |

> ⬆️ 平均加速约 **9.6%**，最大提速达 **16.72%**

---

#### 📈 在高维数据集上的表现（Table 2）

| Dataset | Runtime (CentNN/FastCentNN) | Speed Up | Epochs | ΔMSE (%) |
|--------|------------------------------|----------|--------|----------|
| MNIST | 8.344s → **7.924s** | **5.03%** | 141 → 132 | +1.53×10⁻³ |
| Fashion-MNIST | 8.220s → **7.775s** | **5.41%** | 127 → 119 | -6.37×10⁻³ |

> ⬆️ 在真实图像数据上仍能稳定提速 **~5%**，且 MSE 差异可忽略

---

### ✅ 与基线方法的对比结果  
- **所有数据集上均实现 runtime 和 epochs 下降**
- **聚类质量（MSE）几乎无损**，ΔMSE 均在 ±10⁻²% 以内，部分甚至略有改善
- **视觉对比图（Fig. 2）显示聚类结果高度一致**

---

### ✅ 消融实验结果（隐含分析）  
虽然未明确列出消融实验表格，但从以下方面体现设计有效性：
- 不同数据集（简单/复杂、低维/高维）下均有效 → 验证泛化性
- 绝对与相对阈值模式均可工作 → 验证策略鲁棒性
- 提前分裂未破坏最终结构 → 验证 early trigger 的合理性

---

## 4. 关键结论和发现

### ✅ 论文的主要发现  
1. **CentNN 中的“低运动期”确实存在大量冗余计算**
2. **以总质心移动量 $ M_t $ 作为 entropy proxy 是合理且有效的启发式指标**
3. **引入耐心窗口的早期分裂机制可在不牺牲聚类质量的前提下显著提升训练效率**
4. **FastCentNN 是一个实用、高效、即插即用的 CentNN 替代方案**

> 💡 “The proposed early splitting strategy effectively eliminates unnecessary low-movement epochs.”

---

### ⚠️ 方法的局限性  
1. **依赖经验性阈值选择**：分裂阈值 $ T $ 和耐心 $ P $ 需手动调节，缺乏理论最优解
2. **未改变渐近复杂度**：仅优化常数因子，无法突破 CentNN 的本质计算瓶颈
3. **尚未测试流式/增量场景下的长期稳定性**
4. **未与其他现代聚类器（如 DBSCAN、UMAP+KMeans）横向比较性能**

---

### 🔮 未来工作方向  
1. **自动化阈值调优机制**：结合自适应控制或强化学习动态调整 $ T $ 和 $ P $
2. **扩展至 streaming setting**：研究在持续到来的数据流中如何维持稳定性与响应性
3. **与其他原型模型融合**：探索与 SOM、Neural Gas 等结合的可能性
4. **理论分析 early-split 的收敛性质**：建立更坚实的数学基础

---

## ✅ 总结一句话  
**FastCentNN 通过引入基于质心运动的熵代理与早期分裂机制，在几乎不损失聚类精度的前提下，实现了对 CentNN 最多 **16%** 的加速，是一种简洁、高效、可解释性强的改进方案。**

</details>

---

### 4. [Lighthouse RL: Sample-Efficient Circuit Optimization via Strategic Reset Points](https://arxiv.org/abs/2607.14008)

**Authors**: Mustafa Emre G\"ursoy, Stefan Uhlich, Ryoga Matsuo, Ya\u{g}{\i}z Gen\c{c}er, Arun Venkitaraman, Chia-Yu Hsieh, Andrea Bonetti, Eisaku Ohbuchi, Lorenzo Servadei  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.14008v1  

#### Abstract
In this paper, we introduce Lighthouse RL, a sample-efficient reinforcement learning (RL) approach for analog circuit sizing. Traditional methods lack generalization across different performance targets, while standard RL approaches waste resources exploring unpromising regions. Our method addresses...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Lighthouse RL: Sample-Efficient Circuit Optimization via Strategic Reset Points

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统模拟电路 sizing 方法（如 Bayesian Optimization, BO 或 Evolutionary Strategies, ES）在面对不同设计目标时缺乏泛化能力，每次目标变化都需要从头重新优化，导致**样本效率低下**。而标准 Reinforcement Learning (RL) 虽具备一定适应性，但在探索过程中仍会浪费大量资源在无前途的参数区域。

### 提出的新方法：Lighthouse RL
提出 **Lighthouse RL** ——一种基于**战略性重置策略**（strategic reset strategy）的高效 RL 方法，用于解决黑盒形式的模拟电路优化问题。

#### 核心思想：
- 在训练过程中维护一组高性能配置（称为“**lighthouses**”），这些是接近目标但尚未完全满足所有约束的优质中间解。
- 将这些 lighthouse 状态作为后续 episode 的起始点，从而引导 agent 从更优的位置开始探索，避免重复低效搜索。

#### 方法流程分为两个阶段：
1. **Phase 1: Exploration – Finding Lighthouse States**  
   初始阶段采用随机或近优配置启动 episodes，持续收集接近目标的高质量参数，并存入优先队列 $P$ 和成功集合 $S$。
2. **Phase 2: Exploitation – Using Lighthouse States**  
   当积累足够多的 lighthouse 后，固定使用它们作为 reset 点进行 fine-tuning，加速收敛。

### 相比现有方法的优势
| 优势维度 | 描述 |
|--------|------|
| **Sample Efficiency** | 显著减少达到目标所需的仿真次数（最多提升 1.72×） |
| **Generalization** | 对未见目标（尤其是外推任务）具有更强适应能力（外推成功率最高达 75%，远超其他方法） |
| **Plug-and-Play 特性** | 可无缝集成到任何基于 RL 的优化框架中，无需修改底层算法 |
| **Objective Maximization** | 在极限性能挖掘任务中表现优异，能有效逼近甚至超越人工设计 |

---

## 2. 核心实验方法和设置

### 使用的数据集与电路
实验在以下三种任务上展开：
1. **2D Benchmark Problem**  
   多目标可行性问题：最大化两个球函数 $f_1(x), f_2(x)$，存在多个局部最优区域，用于验证方法通用性。
2. **Two-Stage OpAmp**（两阶段运放）  
   包含 8 个晶体管、1 个补偿电容和 1 个电阻，共 **11 个独立参数**。
3. **Multistage Amplifier**（多级放大器）[21]  
   更复杂的基准电路，含 **29 个独立参数**，代表当前 RL-based sizing 中最大规模之一。

> 所有电路仿真基于 **Skywater SKY130 PDK** + **Ngspice**，工艺角为 TT（Typical-Typical）。

### 实验设置
- **训练预算**：
  - Two-Stage OpAmp：6,000 次 SPICE 仿真
  - Multistage Amplifier：30,000 次 SPICE 仿真
- **推理阶段**：每个目标最多允许 30 步 inference，测试 50 个随机采样的目标
- **随机种子**：5 个不同 seed 运行以保证可复现性
- **目标范围**：训练期间目标从指定区间采样；外推任务则使用超出训练区间的更高要求目标

### 评估指标
| 指标 | 定义 |
|-----|------|
| **Training Sample Efficiency (T.S.E.)** | 达成 50 次成功 episode 所需仿真的相对效率（归一化至最差方法） |
| **Success Rate (SR%)** | 推理阶段成功满足目标的比例（50 个目标下平均） |
| **Inference Steps** | 成功找到解所需的平均仿真步数（失败按最大长度计） |
| **Objective Maximization** | 在不可达目标训练下所能达到的最大性能值（增益、带宽等） |

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Bayesian Optimization (BO)** | 黑箱优化，无训练 | 不依赖模型，但需为目标单独优化 |
| **RL Backbone** | 基础 RL 框架 | 固定起点或随机起点，无 reset 策略 |
| **RoSeOpt [5]** | BO + RL 结合 | 使用 BO 初始化起点，再用 RL 微调，SOTA 方法 |
| **RoSeOpt (fine-grained)** | 改进版 RoSeOpt | 缩小动作空间步长以提高精度 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

#### ✅ Two-Stage OpAmp
| Method | T.S.E. (×) | SR (%) | Inf. Steps |
|--------|------------|--------|-----------|
| BO | N/A | 37.6±4.8 | 123.1±6.2 |
| RoSeOpt | 1.00× | 58.2±15.4 | 18.0±4.2 |
| RL Backbone | 1.10× | **100.0±0.0** | 15.7±2.1 |
| **Lighthouse RL** | **1.55×** | **100.0±0.0** | **3.8±1.4** |

> ⭐ Lighthouse RL 实现 **100% 成功率**，且仅需 **3.8 步**完成推理，比 RL Backbone 快 **4.1×**

#### ✅ Multistage Amplifier
| Method | T.S.E. (×) | SR (%) | Inf. Steps |
|--------|------------|--------|-----------|
| BO | N/A | 26.4±6.2 | 560.8±19.7 |
| RoSeOpt | 1.02× | 13.2±26.4 | 28.0±4.0 |
| RL Backbone | 1.00× | 0.0±0.0 | 30.0±0.0 |
| **Lighthouse RL** | **1.72×** | **87.2±18.0** | **7.9±4.8** |

> ⭐ 在高维复杂电路中仍保持 **87.2% 成功率**，训练效率领先 RoSeOpt **1.72×**

### 外推任务（Extrapolation）表现（Table 3）
| Task | Method | SR (%) |
|------|--------|--------|
| Two-Stage (外推) | BO | 8.0±4.4 |
| | RoSeOpt | 5.6±4.0 |
| | RL Backbone | 50.4±41.6 |
| | **Lighthouse RL** | **75.2±26.8** |
| Multistage (外推) | BO | 10.0±1.8 |
| | **Lighthouse RL** | **29.2±37.0** |

> 💡 Lighthouse RL 展现出卓越的**外推能力**，尤其在 multistage 上是唯一取得显著成功的 RL 方法

### Objective Maximization（表 4）
针对 multistage 放大器进行极限性能挖掘：

| Method | Gain (dB) | BW (MHz) | PM (°) | GM (dB) |
|-------|----------|---------|-------|--------|
| Human Design | 91 | 1.2 | 86.3 | 57.8 |
| RL Backbone | 52.5 | 13.1 | 59.0 | 15.9 |
| **Lighthouse RL** | **118.1** | **24.5** | **89.7** | **59.6** |

> 🏆 Lighthouse RL 在所有指标上全面超越人类设计，尤其在 **Gain (+30dB)** 和 **Bandwidth (+20MHz)** 上实现巨大突破

### 消融实验分析（隐含于实验设计）
虽然未明确列出消融实验表格，但从以下对比可看出关键组件作用：
- **固定 reset vs. 随机 reset vs. Lighthouse reset**（图 2b）显示：lighthouse 初始化显著加快收敛速度
- **Phase 1 + Phase 2 设计** 是性能提升的关键：先探索后聚焦，避免过早陷入局部最优
- **Priority Queue P 与 Successful Set S 分离机制** 提升了训练稳定性与实用性

---

## 4. 关键结论和发现

### 主要发现
1. **Strategic Reset 是提升 RL 样本效率的有效手段**  
   利用历史发现的 high-performing states 作为起始点，极大减少了无效探索。
   
2. **Lighthouse RL 具备强泛化与外推能力**  
   即使面对训练分布之外的目标，也能高效收敛，解决了传统方法“只能优化已知目标”的瓶颈。

3. **适用于高维复杂电路优化**  
   在 29 参数的 multistage amplifier 上依然保持高成功率，证明其可扩展性。

4. **支持 Objective Maximization 场景**  
   通过设置不可达目标并持续更新 lighthouse，系统可自动逼近性能极限，辅助工程师突破设计边界。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **Lighthouse 多样性依赖训练动态** | 当前选择机制未显式考虑多样性，可能导致覆盖不足 |
| **Phase 切换阈值需手动设定** | 如何时从 exploration 转向 exploitation（由 $m$ 控制）需要经验调整 |
| **对初始探索质量敏感** | 若早期未能发现优质候选，可能影响后期性能 |

### 未来工作方向
1. 引入**显式的多样性度量机制**（diversity-aware selection）来丰富 lighthouse 集合
2. 探索**并行化训练**以加速 high-performance configuration 的发现
3. 将该 reset 策略推广至更多类型的 black-box optimization 任务（如机器人控制、芯片布局等）
4. 研究自适应 phase transition 机制，实现全自动切换

---

> ✅ **总结一句话**：  
> **Lighthouse RL 通过引入“战略重置点”机制，在不改变底层 RL 算法的前提下，实现了样本效率、泛化能力和性能上限的全面提升，是面向实际模拟电路设计场景的一项重要进展。**

</details>

---

### 5. [Leveraging unlabelled data for generalizable neural population decoding](https://arxiv.org/abs/2607.14086)

**Authors**: Ximeng Mao, Nanda H. Krishna, Avery Hee-Woon Ryoo, Matthew G. Perich, Guillaume Lajoie  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.14086v1  

#### Abstract
Robust and accurate neural decoders are integral to neurotechnologies such as brain-computer interfaces and closed-loop experiments. Recent work has shown that tokenizing neural data at the spike level facilitates multi-session pretraining and delivers state-of-the-art decoding performance. However,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Leveraging unlabelled data for generalizable neural population decoding

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于**spike-tokenizing**的神经解码模型（如POYO系列）严重依赖**监督学习（Supervised Learning, SL）**，只能利用带有行为标签的配对数据进行训练。这极大地限制了其可扩展性，因为大量已有的神经记录数据是**无标签（unlabelled）**的。该论文旨在解决如何有效利用这些海量无标签数据来提升模型泛化能力的问题。

### 提出的新方法：MOJO
作者提出了 **MOJO (Masked autOencoder-based JOint training)**，一种用于spike-tokenizing模型的联合训练框架。其核心思想是将**自监督学习（Self-Supervised Learning, SSL）** 与现有的监督学习目标相结合。

- **SSL目标**：采用**掩码自编码器（Masked Autoencoder, MAE）**策略，在**隐变量（latent）层面**对时间片段进行掩码，并要求模型重建被掩码部分的**spike rates**。
- **SL目标**：保持原有的行为解码任务（如手部速度预测、视觉刺激分类等）。
- **联合优化**：通过一个加权损失函数 $L_{MOJO} = \alpha_{SSL} L_{SSL} + \alpha_{SL} L_{SL}$ 同时优化两个目标。

### 相比现有方法的优势
1.  **突破数据瓶颈**：首次成功地将SSL引入到以单个spike为token的模型中，使其能够利用无标签数据进行预训练，显著扩大了可用数据集的规模和多样性。
2.  **性能全面提升**：在多种任务、物种和模态上，MOJO均优于纯SL训练的同类模型（如POYO, POGRU, POMAMBA），尤其是在**少样本（few-shot）** 和**低标签数据**场景下优势更为明显。
3.  **更优的表征学习**：学习到的**unit embeddings**具有更强的可解释性，能自然地区分不同的脑区、神经元放电特性（如firing rate, CV等），而无需显式优化这些任务。
4.  **跨物种迁移能力**：证明了在猴子运动皮层数据上预训练的模型，可以正向迁移到小鼠视觉皮层的任务上，展示了构建通用**Neuro-Foundation Models (NFMs)** 的潜力。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了多个物种、模态和任务，验证了方法的普适性：
- **猴子运动任务 (Monkey Reaching Tasks)**：来自多个实验室的5个公开数据集（Perich et al., O'Doherty et al.等），记录了非人灵长类动物在执行中心向外（center-out）、随机目标（random target）和迷宫导航任务时的**motor cortex** spiking活动，目标是解码二维手部速度。
- **小鼠视觉任务 (Mouse Vision Tasks)**：使用Allen Institute的**visual coding Neuropixels dataset**，记录了小鼠在观看自然场景（Natural Scenes）和漂移光栅（Drifting Gratings）时，跨越视觉皮层、丘脑等多个脑区的spiking活动，目标是进行视觉刺激分类。
- **小鼠决策任务 (Mouse Decision Tasks)**：使用IBL Reproducible Electrophysiology dataset，记录了小鼠在执行视觉驱动的决策任务时的多区域spiking活动，目标是解码选择（choice）、区块（block）、轮速（wheel）和胡须运动（whisker）。
- **人类语音任务 (Human Speech Tasks)**：使用人类**ECoG**（皮层脑电图）数据集，记录了参与者朗读辅音-元音音节时的信号，目标是解码音节、辅音和元音。

### 实验设置和评估指标
- **骨干网络 (Backbones)**：在POYO家族的两种主流架构上进行了测试：**Transformer-based (POYO)** 和 **SSM-based (POSSM, 如POGRU, POMAMBA)**。
- **微调策略 (Finetuning Strategies)**：
  - **UI (Unit Identification)**：仅更新新会话中的unit和session embeddings，其余参数冻结，模拟快速适应。
  - **FT (Full Finetuning)**：解冻所有参数进行端到端微调。
- **评估指标**：
  - 连续变量（如手部速度、轮速）：使用 **R²**。
  - 分类任务（如视觉刺激、选择）：使用 **Accuracy** 或 **Balanced Accuracy**。
  - 少样本学习：在仅有少量（如2, 4, 8...）带标签试验的情况下评估性能。

### 基线方法对比
- **纯监督的spike-tokenizing模型**：POYO, POGRU, POMAMBA (UI and FT)。
- **基于binning的SSL模型**：NDT-2, NDT-3, NEDS，这些是处理连续信号（如binned spike counts）的先进SSL解码器。
- **其他先进模型**：MLP, GRU (传统方法)；Du-IN (针对ECoG的最新foundation model)。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与基线对比
1.  **猴子运动任务 (Table 1)**：
    - 在“新动物”（New animal）的挑战性场景下，MOJO-POGRU (FT) 在T-RT任务上的R²达到 **0.7776**，显著优于最好的纯SL基线POGRU (FT) 的0.7575。
    - 即使只进行UI微调，MOJO-POSSM（<18K参数）的性能也能媲美全参数微调（FT）的纯SL-POSSM（>7.6M参数），展现了极高的参数效率。

2.  **小鼠视觉任务 (Table 2)**：
    - 在自然场景（NS）分类任务上，MOJO-POYO-L(J) (FT) 达到了 **95.48%** 的准确率，优于NEDS的93.96%。
    - 在漂移光栅（DG）分类上，多个MOJO变体达到了接近完美的 **100%** 准确率。

3.  **人类语音任务 (Figure 3c)**：
    - MOJO在音节（syllable）分类上的准确率相比纯SL的POYO有**47%的相对提升**。
    - 其性能与专门为ECoG设计的SOTA模型Du-IN相当，证明了MOJO从spike数据设计出发，却能**泛化到连续信号模态**的强大能力。

4.  **少样本微调 (Figure 2a)**：
    - 在仅有**2个带标签试验**的情况下，MOJO结合额外的无标签数据，即可达到完全监督性能的**60%以上**，而纯SL模型在此极端低数据条件下表现极差。

### 消融实验结果
- **路径整合 (Pathway Integration)**：共享输入交叉注意力（input cross-attention）和骨干网络参数的集成方式，虽然略微降低了UI性能，但显著提升了FT性能，并且计算开销增加很小（约2-10%参数）。
- **掩码策略 (Masking Strategy)**：实验证明MOJO对掩码比例（从0.3到0.7）不敏感，表明其鲁棒性。
- **替代SSL方法 (Alternative SSL methods)**：与在输入token层面进行空间掩码（masking random neurons/regions）或使用对比预测编码（CPC）的方法相比，MOJO的**掩码自编码器（MAE）在隐变量层面**的策略效果最好。

---

## 4. 关键结论和发现

### 主要发现
1.  **SSL赋能spike-tokenizing模型**：MOJO的成功证明了将SSL（特别是MAE）与spike-tokenizing模型结合是可行且高效的，为利用海量无标签神经数据铺平了道路。
2.  **性能与数据效率双提升**：联合SSL-SL训练不仅能提升最终解码性能，还能在**标签数据极度稀缺**的场景下实现稳健的few-shot学习，这对临床应用（如需要频繁校准的BCI）至关重要。
3.  **学习到的表征更具生物学意义**：MOJO学习到的unit embeddings不仅是解码的副产品，其本身蕴含了丰富的神经生物学信息（如脑区身份、放电统计特性），并且其几何结构反映了功能连接性（如将远距离但功能相关的皮层-丘脑神经元聚在一起）。
4.  **跨模态与跨物种的通用性**：MOJO不仅适用于spiking data，还能直接应用于ECoG等连续信号，并取得了与专用模型相当的性能。同时，跨物种的联合预训练也显示出积极的迁移效应，为构建统一的**Neuro-Foundation Models**提供了有力证据。

### 方法的局限性
1.  **数据饥渴 (Data-hungry)**：MOJO需要大量的数据来学习有效的神经动力学，如果数据量不足，弱的SSL目标甚至可能损害SL性能。
2.  **仅限于时间掩码**：当前的MAE策略只进行时间维度的掩码，缺乏对空间（未见过的神经元或脑区）的显式建模。
3.  **微调时需重学embeddings**：当迁移到新会话时，unit embeddings需要从头开始学习，这继承了POYO家族的缺点，在SSL背景下显得尤为浪费。
4.  **模态特定设计**：为了适应ECoG，仍需添加value embeddings等模态特定的设计，限制了不同模态数据的无缝联合预训练。

### 未来工作方向
1.  **探索多模态联合训练 (Joint multi-modal training)**：将spiking, ECoG, calcium imaging等多种模态的数据联合起来进行训练，以获得更全面的神经动态理解。
2.  **改进单元嵌入复用**：研究**amortized methods**或**discrete codes**，使得学习到的单元知识可以在不同会话间高效复用，避免每次微调都从零开始。
3.  **开发更强大的空间SSL目标**：设计能够推断未见神经元或脑区特性的SSL任务，进一步增强模型的泛化能力。

</details>

---

### 6. [Transforming LLMs into Efficient Cross-Encoders via Knowledge Distillation for RAG Reranking](https://arxiv.org/abs/2607.11933)

**Authors**: Shreeya Dasa Lakshminath, Shubhan S  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.11933v1  

#### Abstract
Cross-encoders achieve high reranking accuracy in Retrieval-Augmented Generation (RAG) pipelines but impose quadratic inference costs that limit real-time deployment. We address this by fine-tuning LLaMA 3 (8B) as a drop-in reranker using a two-stage pipeline: supervised fine-tuning on a custom quer...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Transforming LLMs into Efficient Cross-Encoders via Knowledge Distillation for RAG Reranking

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **Cross-encoder** 在 Retrieval-Augmented Generation (RAG) 中虽能提供高精度重排序（reranking），但其推理复杂度为 $O(n)$（每个查询-文档对需一次前向传播），导致在大规模或实时场景下计算开销过高。
- 同时，大型语言模型（LLM）本身参数量大，部署成本高。

### 🚀 提出的新方法与思路
提出一种两阶段 fine-tuning + quantization 流程，将 **LLaMA 3 (8B)** 改造为高效的 **drop-in reranker**，替代传统 BERT-based cross-encoder：
1. **Supervised fine-tuning** 使用自建的 (query, document, relevance) 数据集，基于 **Unsloth 框架** 和 **LoRA** 进行参数高效微调；
2. 微调后进行 **4-bit quantization**（GGUF 格式），显著降低推理资源需求。

该方法并非严格意义上的 *knowledge distillation*，而是通过指令微调让 LLM 学习 cross-encoder 的排序行为，实现“功能蒸馏”。

### 🔍 相比现有方法的优势
| 方法 | 缺陷 | 本工作的改进 |
|------|------|---------------|
| Cross-encoder (e.g., BERT) | 推理慢、扩展性差 | 减少冗余计算，支持更高效部署 |
| Zero-shot LLM reranking (e.g., RankGPT) | 成本高、延迟大 | 经过 fine-tuning + quantization，推理效率更高 |
| Full fine-tuning of LLM | 资源消耗巨大 | 使用 LoRA 实现参数高效训练 |

> ✅ **核心优势**：在保持甚至超越 cross-encoder 排序质量的同时，通过量化大幅降低部署门槛，适用于生产环境。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- 自定义领域特定问答数据集，涵盖：
  - 学术项目详情
  - 教学政策
  - 课程要求等
- 文档来源：机构发布的 PDF、DOCX、TXT 文件
- 查询与答案由人工标注，确保高质量监督信号

### ⚙️ 实验设置
- **RAG Pipeline 架构**：
  1. **Dual Retriever**：
     - Dense retrieval：使用 OpenAI `text-embedding-ada-002` + Chroma 向量数据库
     - Sparse retrieval：BM25
     - 结果合并形成候选文档集合（top-k，k=5）
  2. **Reranking Stage**：
     - 对比两种 reranker：
       - Baseline：BERT-based cross-encoder
       - Ours：fine-tuned LLaMA 3 8B with LoRA + 4-bit quantization
  3. **Generation Stage**：
     - 使用 **GPT-4o** 生成最终回答，上下文为重排序后的 top 文档

- **Fine-tuning 配置**：
  - 模型初始化：`llama-3-8b-bnb-4bit`
  - LoRA 设置：
    - 应用于 `q_proj`, `k_proj`, `v_proj`, `o_proj` 层
    - rank $r=16$, scaling $\alpha=32$
    - 使用 AdamW (8-bit)、学习率 $2\times10^{-5}$、weight decay=0.01
  - 训练格式：listwise ranking prompt（如 “Rank the following documents…”）
  - 精度：混合精度训练（fp16/bf16）

- **推理优化**：
  - LoRA adapter 合并后导出为 **4-bit GGUF** 格式，便于轻量级部署

### 📊 评估指标（使用 RAGAS 框架）
| 指标 | 描述 |
|------|------|
| **Answer Relevancy** | 回答是否贴合查询意图 |
| **Context Precision** | 检索到的上下文中有多少是真正相关的 |
| **Answer Similarity** | 生成答案与真实答案之间的语义相似度 |
| **Answer Correctness** | 回答的事实准确性（相对于 ground truth） |

> 所有指标均为 reference-free，自动化评估，无需人工评分。

### ↔️ 基线方法对比
- **Baseline**：标准 BERT-based cross-encoder（典型双塔后接 [CLS] 分类头）
- **Experimental Condition**：除 reranker 外，其余模块完全一致（相同 retriever、embedding model、generator）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table I）

| Metric | Cross-Encoder | LLaMA 3 (Ours) | 提升幅度 |
|--------|----------------|----------------|----------|
| **Answer Relevancy** | 0.78 | **0.89** | ↑14% |
| **Context Precision** | 0.75 | **0.87** | ↑16% |
| **Answer Similarity** | 0.74 | **0.88** | ↑19% |
| **Answer Correctness** | 0.70 | **0.85** | ↑21% |

> ✅ 全面优于 cross-encoder 基线，在所有四项 RAGAS 指标上均有显著提升。

### 🔍 性能分析要点
- 最大增益出现在 **Answer Correctness (+21%)**，说明更好的文档排序直接提升了生成内容的事实准确性。
- **Context Precision 提升 16%** 表明模型能更精准识别相关段落，减少噪声干扰。
- 尽管未报告 latency 或 throughput 数据，但 **4-bit quantization** 显著降低了内存占用，使 8B 模型可在消费级硬件运行。

### ❌ 消融实验（Ablation Study）
- 论文中 **未包含明确的消融实验**（如移除 LoRA、关闭量化、比较不同 rank 等）
- 作者指出未来可进一步探索更小模型（如 LLaMA 3 1B/3B）的影响

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Instruction-tuned LLM 完全可以胜任 reranking 任务**，且经过 fine-tuning 后性能超过传统 cross-encoder。
2. 利用 **LoRA + 4-bit quantization** 可以构建一个高性能、低部署成本的 reranker，适合集成进实际 RAG 系统。
3. 更强的世界知识（world knowledge）可能帮助 LLM 区分“表面相关”与“事实有用”的文档，从而提升 context quality 和 generation accuracy。
4. 提供了可复用的 **LoRA adapter checkpoint**，允许他人直接加载使用，无需重新训练。

### ⚠️ 方法的局限性
1. **数据集局限性强**：仅在一个小型、领域特定的学术文档 QA 数据集上验证，泛化能力未知。
2. **缺乏开放域测试**：未在 MS-MARCO、BEIR 等标准 IR benchmark 上评估。
3. **缺少横向对比**：未与 monoT5、RankGPT 等 LLM-based reranker 方法对比。
4. **无延迟/吞吐量测量**：虽然声称“高效”，但没有量化推理速度或 QPS 数据支撑。

### 🔮 未来工作方向
1. 在 **MS-MARCO、BEIR** 等公开基准上评估泛化能力；
2. 开展 **latency benchmarking**，量化推理效率增益；
3. 与 **listwise rerankers（如 RankGPT）** 进行公平比较；
4. 探索更小 base model（如 **LLaMA 3 1B / 3B**）以进一步压缩成本；
5. 引入真正的 **knowledge distillation** 机制，从 cross-encoder 输出中学习 soft labels。

---

## ✅ 总结一句话
> 本文证明了通过 **LoRA 微调 + 4-bit 量化**，可将 LLaMA 3 8B 高效转化为优于传统 cross-encoder 的 reranker，在提升 RAG 生成质量的同时兼顾部署可行性，为构建高性能低成本 RAG 系统提供了实用路径。

</details>

---

### 7. [Concurrent Image Understanding and Generation: Self-Correcting Coupled Markov Jump Processes](https://arxiv.org/abs/2607.13188)

**Authors**: Minh-Quan Le, Armand Comas, Alexandros Lattas, Stylianos Moschoglou, Pedro V\'elez, Amit Raj, Aaron Germuth, Thabo Beeler, Dimitris Samaras, Di Qiu  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.13188v1  

#### Abstract
Human cognition does not separate understanding and generation. A teacher at a whiteboard speaks and draws $\textit{together}$, each modality reshapes the other. In this paper, we bring this coupled loop to artificial systems. Masked Diffusion Models (MDMs) are ideally suited to this task, yet exist...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Concurrent Image Understanding and Generation: Self-Correcting Coupled Markov Jump Processes**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有 **Masked Diffusion Models (MDMs)** 在多模态生成任务中存在两个关键缺陷：
- **模态漂移 (modality drift)**：文本和图像分支在生成过程中独立更新，仅依赖上一步的联合状态，缺乏**跨模态实时反馈**，导致生成内容不一致（如文本描述一个对象，图像却未体现）。
- **无法修正错误**：标准 MDM 只能单向“解掩码”（unmask），一旦某个 token 被生成就无法撤销，导致早期错误无法纠正。

这些问题使得当前方法难以实现真正意义上的**并发理解与生成**，即语言和视觉信号在生成过程中持续相互塑造。

---

### **提出的新方法与新思路**
论文提出了 **Self-Correcting Coupled Markov Jump Processes (SC-CMJP)** 框架，并设计了训练免费的采样器 **CO2Jump (Self-COrrecting COupled Jump)**，其核心创新如下：

#### **(1) 耦合跳变过程 (Coupled Markov Jump Processes)**
- 将文本和图像视为统一序列上的两个子集，在每个去噪步骤中，**一个模态的转移速率（transition rate）是另一个模态置信度的函数**。
- 利用 **cross-modal attention** 将文本对某个区域的理解传递给图像分支，动态调整图像 token 的生成优先级。

#### **(2) 自我修正机制（Self-Correction via Remasking Jump）**
- 引入 **remasking jump**，允许已生成的 token 被重新掩码（retract commitment），当跨模态证据冲突时可主动撤回错误决策。
- 这种“生-死”（birth-death）机制实现了真正的双向纠错能力。

#### **(3) 非对称评分规则（Asymmetric Scoring Rule）**
- 文本分支基于自身置信度（self-confidence）进行评分；
- 图像分支则采用混合评分：  
  $$
  \text{Score}_{\text{image}} = (1-\lambda)\cdot\text{Rank(SelfConf)} + \lambda\cdot\text{Rank(CrossSignal)}
  $$
  其中 $\lambda$ 是由图像和文本熵动态控制的门控变量，实现“不确定性高的一方信任另一方”。

---

### **相比现有方法的优势**
| 特性 | MDM / MMaDA-Parallel | ReMDM | CO2Jump (Ours) |
|------|------------------------|--------|----------------|
| 并发生成 | ✅（但无耦合） | ✅ | ✅（强耦合） |
| 跨模态反馈 | ❌ | ❌ | ✅（注意力驱动） |
| 可 remask | ❌ | ✅（单模态） | ✅（跨模态触发） |
| 单次前向传播 | ✅ | ✅ | ✅ |
| 无需额外训练 | ✅ | ✅ | ✅ |

> ✅ **CO2Jump 是首个实现“训练免费 + 单步前传 + 跨模态耦合 + 自我修正”的并发多模态生成框架。**

---

## **2. 核心实验方法和设置**

### **使用的数据集**
为支持联合多模态生成任务，作者构建并计划发布三个大规模数据集：

| 数据集 | 任务 | 规模 | 内容 |
|-------|------|-----|------|
| **JEDIT-1M** | 图像编辑与理解 | 1M 样本 | 包含源图、目标图、编辑指令、场景图分析、推理链 |
| **JMAZE-200K** | 迷宫求解 | 200K 样本 | 输入迷宫地图，输出路径坐标序列和可视化路径 |
| **JNoNO-200K** | 数织谜题（Nonogram） | 200K 样本 | 行列线索输入，输出填色结果和逻辑推理 |

所有数据集均提供 **in-distribution (ID)** 和 **out-of-distribution (OOD)** 测试集以评估泛化能力。

---

### **实验设置与评估指标**

#### **模型基础**
- 主干模型：**Lumina-DiMOO**（统一文本-图像词表的 MDM）
- 所有方法共享相同训练配置（64× H100, batch size 512, lr=2e-5）

#### **评估任务**
1. **图像编辑与理解（JEDIT-1M）**
   - **文本质量**：GPT-2 Large 的 PPL（Perplexity） + Token Entropy
   - **图像质量**：ImgEditBench Oracle Score（Gemini 3 Flash）
   - **图像理解能力**：COCO-style **mAP@0.5:0.95**，基于模型自生成的目标图像进行伪标注（pseudo-grounding）

2. **视觉推理任务（JMAZE & JNoNO）**
   - **联合准确率（Joint Accuracy）**：只有当文本答案和图像输出都正确时才算成功
   - 分别报告 ID / OOD 子集表现

#### **基线方法对比**
| 方法 | 类型 | 是否 remask | 是否耦合 |
|------|------|-------------|----------|
| **MDM** (Sahoo et al., 2024) | 独立并行 | ❌ | ❌ |
| **ReMDM** (Wang et al., 2025) | 独立并行 | ✅（单模态） | ❌ |
| **MMaDA-Parallel** (Tian et al., 2026) | 交错更新 | ❌ | ❌ |
| **CO2Jump (Ours)** | 耦合跳变 | ✅（跨模态） | ✅ |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 图像编辑任务（Table 1）**
| 方法 | ImgEditBench ↑ | Target mAP ↑ | Overall mAP ↑ |
|------|------------------|---------------|----------------|
| MDM | 1.78 | 0.322 | 0.354 |
| ReMDM | 1.73 | 0.318 | 0.353 |
| MMaDA-Parallel | 1.44 | 0.304 | 0.335 |
| **CO2Jump (Ours)** | **1.93** | **0.346** | **0.369** |

> 🔹 CO2Jump 在图像质量和理解能力上全面领先，且显著优于强参考模型 **Qwen3-VL-8B**（target mAP: 0.346 vs 0.330）

#### **(2) 视觉推理任务（Table 2）**
| 方法 | JMAZE Total Acc | JNoNO Total Acc |
|------|------------------|------------------|
| MDM | 0.424 | 0.100 |
| ReMDM | 0.180 | 0.002 |
| MMaDA-Parallel | 0.390 | 0.138 |
| **CO2Jump (Ours)** | **0.432** | **0.168** |

> 🔹 在非ogram 上，CO2Jump 的 OOD 性能提升达 **+0.062**（0.175 vs 0.113），显示强大泛化能力。

---

### **消融实验结果（Table 3）**

| 消融项 | ImgEditBench | Target mAP | Overall mAP |
|--------|--------------|------------|-------------|
| Full CO2Jump | 1.93 | 0.346 | 0.369 |
| -Shared Percentile Rank | 1.87 | 0.315 | 0.344 |
| -Entropy-Based Gating ($\lambda$) | 1.91 | 0.316 | 0.354 |
| -Self-Correction (no remask) | 1.92 | 0.310 | 0.337 |

> 🔸 移除 **Self-Correction** 导致理解能力下降最大（-0.032 mAP），说明 **remasking 是维持一致性关键**。  
> 🔸 移除 **Percentile Rank** 显著影响生成质量，表明跨模态分数需归一化才可融合。

---

### **其他重要发现**
- **性能随 NFE 单调上升**（Figure 5）：
  - CO2Jump 是唯一在增加去噪步数（NFE）后性能持续提升的方法。
  - 其他方法在高 NFE 下出现退化，而 CO2Jump 因耦合效应积累优势。
- **定性分析**（Figure 6）：
  - 基线方法常出现跨模态矛盾（如文本路径正确但图像走错路）。
  - CO2Jump 始终保持双通道一致，满足所有约束条件。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **并发理解与生成可通过耦合跳变过程建模**：将文本与图像视为同一 Markov Jump Process 的两个部分，实现实时交互。
2. ✅ **跨模态置信度可用于指导生成调度**：通过 cross-modal attention 传递语义置信，使图像生成更受文本引导。
3. ✅ **remasking 应由跨模态矛盾触发**：传统的单模态 remasking 不足以修复语义不一致，必须引入“耦合驱动的撤销机制”。
4. ✅ **性能可随去噪步数单调增长**：证明了跨模态耦合具有**复合增益效应（compound benefit）**，而非饱和。

---

### **方法的局限性**
- 当前实现仍基于 **frozen backbone**，未探索端到端训练下的潜力。
- 所有实验集中在 **text-image** 对，尚未扩展至视频、音频等其他模态。
- 对 extremely long-range dependencies（如超大 nonogram）仍有挑战。

---

### **未来工作方向**
- 将 SC-CMJP 框架推广至 **video, audio, structured output** 等多模态组合。
- 探索 **joint training + coupled sampling** 的协同优化。
- 构建更复杂的 **multi-turn interactive editing** 场景，支持人机协作白板式解释。

---

> 📌 **一句话总结**：  
> 本文首次将人类认知中的“边说边画、互为修正”机制形式化为 **Self-Correcting Coupled Markov Jump Processes**，并通过 **CO2Jump** 实现了无需训练、单步前传、持续耦合、自我修正的并发图文生成，推动 AI 向更接近人类认知的多模态智能迈进。

</details>

---

### 8. [Where Should RL Post-Training Compute Go? Model Size, Search, Learning, and Feedback](https://arxiv.org/abs/2607.13389)

**Authors**: Patrick Wilhelm, Odej Kao  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.13389v1  

#### Abstract
Reinforcement Learning (RL) post-training is increasingly used to adapt foundation models for reasoning, planning, and feedback-driven robot-learning pipelines, but constrained post-training resources are often summarized by a single total FLOP budget. We study the fixed-budget decision problem behi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Where Should RL Post-Training Compute Go? Model Size, Search, Learning, and Feedback*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**强化学习（RL）后训练（post-training）中的计算资源分配问题**。在给定固定总 FLOP 预算的前提下，研究者需要决策如何分配有限的计算资源：
- 是使用更大的模型？
- 还是用更小的模型但训练更久？
- 或是生成更多 rollout 搜索？
- 抑或是投入更多计算到奖励/反馈模型中？

当前大多数 RL 后训练工作仅报告“总 FLOPs”，忽略了这些内部资源分配机制，导致实验不可复现、比较不公平。

### 🚀 提出的新方法与新思路

#### （1）提出 **FLOP 分解框架**
将 RL 后训练的总计算 $ C_{\text{total}} $ 显式分解为三个部分：
$$
C_{\text{total}} = C_{\text{search}} + C_{\text{learning}} + C_{\text{reward}}
$$
- $ C_{\text{search}} $：autoregressive rollout 生成（推理开销）
- $ C_{\text{learning}} $：policy update / gradient 更新（训练开销）
- $ C_{\text{reward}} $：reward model 或 verifier 的推理成本

这一分解揭示了不同配置下看似相同的总 FLOP 实际上可能代表完全不同的资源利用模式。

#### （2）引入 **更新分数（update fraction）$ p $** 作为核心调控变量
定义：
$$
p = \frac{C_{\text{learning}}}{C_{\text{total}}}
$$
- 小 $ p $：偏向搜索或奖励计算（search-heavy / reward-heavy）
- 大 $ p $：偏向策略更新（update-heavy）

通过调节 $ p $，系统地探索在相同总预算下的最优资源配置前沿（allocation frontier）。

#### （3）提出诊断协议 **RACE（Reward-Aware Compute Allocation）**
一种用于从小规模 pilot grid 中识别有效资源分配范式的诊断工具：
- 在有限预算下运行一组 IsoFLOP 实验
- 拟合局部响应曲线以估计最佳 $ p $
- 推荐验证应集中在哪个 regime（高更新 vs 低更新）

> ⚠️ 注意：RACE 不保证提升最终性能，而是作为一种**资源优先级排序工具**，帮助减少昂贵的全量验证次数。

### 🔍 相比现有方法的优势
| 方面 | 传统做法 | 本文改进 |
|------|--------|---------|
| 资源描述 | 只报告 total FLOPs | 明确拆解为 search/learning/reward 三部分 |
| 比较公平性 | 忽略 reward model 成本 | 统一计入 $ C_{\text{reward}} $，实现机制可比 |
| 决策支持 | 缺乏指导原则 | 提供 RACE 协议辅助 early-stage 决策 |
| 结果解释 | “某方法更强” | 揭示“强”的原因是因目标不同、模型大小不同等条件变化所致 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 主要任务：**数学推理（mathematical reasoning）**
- 训练数据：**Polaris-53K**（instruction-following 数学题数据集）
- 下游验证集：
  - **GSM8K**
  - **MATH-500**

> 使用数学推理作为机器人学习中规划与反馈模块的代理任务（proxy task），便于控制变量并进行精细 FLOP 分析。

### ⚙️ 实验设置
- **模型架构**：LoRA-adapted Qwen2.5 系列语言模型
  - 规模：1.5B、3B、7B 参数
- **训练算法**：GRPO（Generalized Reward Policy Optimization）
- **采样设置**：
  - 每个 prompt 采样 $ K=2 $ 条 completion
  - 最大长度 $ L=2048 $
- **LoRA rank 变化范围**：从 8 到 256，覆盖不同训练强度
- **IsoFLOP 控制**：调整 update steps 和 LoRA rank，使各实验落在近似相等的总 FLOP 区间内（如 $ \log_{10} C \approx 16.0–16.4 $）

### 🎯 评估指标
| 类型 | 指标名称 | 说明 |
|------|--------|------|
| **训练期指标** | Native training reward | 各 reward system 自身优化的目标函数 |
| | Normalized reward | 跨 reward system 比较时使用的 min-max 归一化版本 |
| **下游泛化指标** | Final-answer accuracy | 精确答案匹配率 |
| | Symbolic equivalence accuracy | 表达式等价判断（pass@1） |
| | Common judge score | 共享的过程质量评分器对中间推理打分 |

### 🔁 奖励系统设计（关键变量）
共测试五类 reward 设计，形成一个“反馈丰富度谱”：
| 类型 | 特点 | 是否消耗额外 compute |
|------|------|------------------|
| Sparse | 仅最终答案正确性 | 否 |
| Structured | 加入格式与输出结构约束 | 否 |
| Dense | 提供数值接近度的部分信用 | 否 |
| Proxy PRM | 使用轻量过程监督信号模拟 PRM | 是（少量） |
| Real PRM (Qwen2.5-Math-PRM-7B) | 使用专用 7B 过程奖励模型打分 | 是（显著） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据与发现

#### ✅ RQ1: 固定预算下是否存在结构性资源分配行为？
- **存在明显的 allocation frontier**
  - Search-heavy（探索不足）和 update-heavy（探索受限）都会导致性能下降
  - 存在一个“甜蜜点”（sweet spot），表现为 reward 随 $ p $ 变化的非线性响应曲面
  - 图1显示，在相同总 FLOP 下，不同 $ p $ 设置对应明显不同的训练 reward

#### ✅ RQ2: 分配前沿是否受模型大小和奖励系统影响？（Conditional Frontiers）
| 条件 | 最佳 observed $ p $ |
|------|--------------------|
| Rule-based rewards（sparse/dense） | ~0.72（update-heavy） |
| PRM-style rewards（proxy & real） | ~0.44–0.45（update-lighter） |

> 原因：PRM 引入了额外的 $ C_{\text{reward}} $，压缩了可用于 rollout 和 update 的空间 → 即便总 FLOP 相同，实际能执行的 policy update 更少。

#### ✅ RQ3: 不同评估目标是否偏好不同资源配置？
| 评估目标 | 偏好 setting |
|--------|-------------|
| Native training reward | PRM 偏好较低 $ p $（~0.45） |
| Downstream pass@1 accuracy | 普遍偏好 **a180**（即高 update setting） |
| Common judge（过程质量） | 偏好 **a100** setting（中等更新频率） |

📌 **重要发现**：没有单一“最优”配置；选择取决于你关心的目标！

#### ✅ RQ4: RACE 是否能有效诊断分配范式？
- 在 **in-grid 和 leave-rank-out 验证中，regime 识别准确率达 1.0**
- 但 **无法稳定提升 held-out native reward**
- ➜ 支持其作为**诊断工具**的有效性，不支持其作为全局优化器

---

### 🔍 消融实验与补充分析（Appendix）

#### A. 模型大小的影响（Same-FLOP 对比）
- 更大的模型（如 7B）每 token 消耗更多 compute
- → 相同 FLOP 预算下获得的 rollout tokens 和 update steps 更少
- ➜ **模型选择与训练分配是耦合的**（not separable）

> 图2展示：在相同 pilot budget 下，1.5B 能完成最多训练步数，而 7B 明显受限

#### B. RACE 拟合模型消融（Appendix G）
测试多种响应函数形式（constant, linear, quadratic in $ p $ / $ \log p $）：
- **常数基线 RMSE 最低** → 表明 pooled response 噪声大
- **log-linear 模型表现最好**（非平凡模型中）
- ➜ 当前 quadratic 拟合主要用于边界检测，而非精确预测

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **RL 后训练不是简单的“越多越好”问题，而是条件性的资源分配问题**
   - 最优资源配置（search vs learning vs reward）依赖于：
     - 模型大小（model size）
     - 奖励系统类型（reward design）
     - 下游评估目标（evaluation target）

2. **报告 total FLOPs 已经不够了**
   - 必须同时报告：
     - Rollout FLOPs
     - Update FLOPs
     - Reward-evaluation FLOPs
     - 更新分数 $ p $
   - 否则无法判断两个“相同计算量”的实验是否真正可比

3. **PRM 并非天然需要更少更新，而是因其自身计算开销改变了资源分布**
   - 若忽略 $ C_{\text{reward}} $，会误以为 PRM 更高效
   - 实际上它只是把预算转移到 reward model 上

4. **不同目标之间存在权衡（trade-off）**
   - 最大化训练 reward ≠ 最大化 downstream accuracy
   - 最大化最终答案正确率 ≠ 最大化过程质量得分
   - ➜ 必须明确“我们想优化什么？”

5. **RACE 是有效的诊断工具，但不是性能提升引擎**
   - 可靠识别 high-/low-update regime
   - 有助于减少验证成本
   - 不能替代 downstream evaluation

---

### ⚠️ 局限性

| 局限 | 说明 |
|------|------|
| 任务领域限制 | 使用数学推理作为 proxy，未涉及真实机器人控制任务 |
| 模型家族单一 | 仅基于 Qwen2.5 + LoRA + GRPO，结论外推需谨慎 |
| 种子与随机性 | 实验未充分重复，可能存在方差影响 |
| 硬件效应忽略 | FLOP 估算未建模内存带宽、通信延迟等硬件瓶颈 |
| RACE 不保优 | 仅为 regime 识别工具，不承诺提升最终性能 |

---

### 🔮 未来工作方向

1. **扩展至多模态与具身智能场景**
   - 将 FLOP 分解框架应用于 Vision-Language-Action（VLA）模型的 RL post-training
2. **动态资源调度机制**
   - 开发可根据训练进程自动调整 $ p $ 的 adaptive allocator
3. **跨任务通用 allocation law 探索**
   - 是否存在某些 reward 类型普遍偏好特定 $ p $？
4. **结合 test-time compute 分析**
   - 统一考虑 training-time 与 inference-time 的 compute trade-offs
5. **开源 RACE 工具链**
   - 提供自动化 pilot grid 设计与 regime 分析 pipeline

---

## 💡 总结一句话
> **“Where should RL post-training compute go?” 没有唯一答案 —— 它取决于你的模型、你的奖励、以及你想被谁评价。**  
> 本文呼吁社区从“只报 total FLOPs”转向“report allocation-aware compute”，推动 RL post-training 成为一门可量化、可复现、可比较的工程科学。

</details>

---

### 9. [Task-Oriented Sensing and Covert Transmissions for Collaborative Multi-AUV Systems](https://arxiv.org/abs/2607.13880)

**Authors**: Xueyao Zhang, Chenyang Yan, Bo Yang, Xuelin Cao, Zhiwen Yu, Bin Guo, George C. Alexandropoulos, Merouane Debbah, Chau Yuen  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.13880v1  

#### Abstract
In underwater covert cooperative missions, autonomous underwater vehicles (AUVs) often cannot rely on active sonar to continuously obtain complete information, since active sensing and frequent communications increase the risk of exposure. As a result, AUVs primarily rely on passive observation, an ...

---

### 10. [PiVoT: A Variational Solution for Real-time Large-scale Multi-object Detection and Tracking under Heavy Clutter](https://arxiv.org/abs/2607.13891)

**Authors**: Runze Gan, Qing Li, Simon J. Godsill, Mike E. Davies, James R. Hopgood  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.13891v1  

#### Abstract
Multi-object detection and tracking from noisy point clouds remain challenging in many data-scarce radar applications. Current Bayesian trackers based on Poisson measurement models offer a training-free solution but struggle to achieve accuracy and efficiency under severe clutter, large object popul...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PiVoT: A Variational Solution for Real-time Large-scale Multi-object Detection and Tracking under Heavy Clutter

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**在高杂波（heavy clutter）、大规模目标数量以及全分辨率 Doppler 点云**条件下，基于雷达等传感器的多目标检测与跟踪（multi-object detection and tracking, MODT）难题。传统方法面临以下挑战：
- **数据稀缺性**：雷达应用中标注数据获取成本高昂，限制了深度学习模型的应用。
- **效率与准确性权衡**：现有的基于贝叶斯推理的方法（如 PMBM、SPA）在复杂场景下难以兼顾实时性和精度。
- **Doppler 信息利用不足**：许多现有方法未能有效整合 Doppler 测量信息。

### 提出的新方法和创新点
作者提出了 **PiVoT (Poisson Measurements-based Variational Multi-object Detection and Tracking)**，一种快速、无需训练、对杂波鲁棒的大规模多目标联合检测与跟踪框架。其核心创新在于一个**新颖的两阶段变分推断（two-stage variational inference）框架**，具体包括：

1. **两阶段变分推断设计**：
   - **Stage 1: Detection and Tracking**：通过变分推断近似边缘后验分布 $p(X, A, P, \theta|Y)$，联合估计目标状态（kinematic states）、形状（shapes）、泊松率（Poisson rates）和数据关联（data association），同时进行检测和跟踪。
   - **Stage 2: Existence Evaluation**：独立评估每个目标的存在性（existence）和可检测性（detectability）概率，解决了标准均场假设（mean-field assumption）在处理不确定存在性时的根本缺陷。

2. **三项关键技术革新**：
   - **自适应出生移除（Adaptive Birth Removal）**：在 Stage 1 的 CAVI 迭代过程中，理论保证地早期移除无效的“出生”目标（即无法关联到任何测量的目标），显著加速了推理过程。
   - **精确全局优化器（Exact Global Optimizer）**：在 Stage 2 中，将计算复杂度从二次方降低到线性时间，使得大规模场景下的存在性评估变得高效。
   - **Doppler 增强的 NHPP 模型（Doppler-augmented NHPP Model）**：首次在 NHPP 框架下有效整合 Doppler 信息，同时保持了单目标似然函数的线性高斯形式，从而保留了高效的闭式更新结构。

### 相比现有方法的优势
- **无需训练（Training-free）**：完全基于物理模型，不依赖标注数据。
- **高效率与实时性**：在标准笔记本电脑上，每秒可处理超过1000个目标，满足实时处理要求。
- **强鲁棒性**：在视觉上难以区分目标与杂波的极端杂波环境中仍能可靠工作。
- **端到端联合检测与跟踪**：无需外部聚类或检测器，直接从原始点云进行处理。
- **性能媲美深度学习**：在真实汽车雷达数据集上的表现与深度学习基准（如 RadarGNN）相当。

---

## 2. 核心实验方法和设置

### 数据集
实验使用了两类数据集：
1. **模拟数据集（Simulated Datasets）**：构建了6个复杂度递增的模拟场景（DS1-DS6），用于与现有贝叶斯方法进行公平比较。这些场景基于四种轨迹集（T1-T4），包含不同大小、数量和密度的目标，并引入了严重的杂波。
2. **真实世界数据集（Real-world Dataset）**：
   - **RadarScenes**：一个广泛使用的现代汽车雷达点云数据集，包含典型的雷达杂波和多样化的道路使用者，用于评估 PiVoT 在真实环境中的性能。

### 实验设置和评估指标
- **评估指标**：
  - **GOSPA (Generalized Optimal Sub-Pattern Assignment)**：衡量多目标跟踪的整体误差，分解为定位误差、漏检和虚警。
  - **F1 Score, Precision, Recall, AP (Average Precision)**：用于评估检测和跟踪的准确性。
  - **运行时间（Runtime）**：以每帧或每秒雷达数据的处理时间为单位，评估算法效率。
- **基线方法对比**：
  - **PMBM (Poisson Multi-Bernoulli Mixture)** 和 **PMBMc**：先进的基于随机有限集（RFS）的贝叶斯跟踪器。
  - **SPA (Sum-Product Algorithm)**：一种基于信念传播的可扩展跟踪器。
  - **RadarGNN**：一个强大的基于图神经网络（GNN）的深度学习检测基准，作为性能上限参考。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1. **在模拟数据集上的结果（Table I）**：
   - **精度更高**：PiVoT 在所有6个数据集上均取得了最低的 GOSPA 分数，表明其综合性能最优。
   - **速度极快**：PiVoT 的运行时间远低于其他方法。例如，在 DS6 上，PiVoT 耗时 0.3008 秒，而次优的 PMBMc 耗时 39.983 秒，速度快了约 **130倍**。
   - **优势随场景复杂度增加而扩大**：在杂波严重（DS4-DS6）或目标数量众多的场景下，PiVoT 的优势尤为明显。

2. **大规模跟踪演示（Section VIII-B）**：
   - 成功实现了对 **1000个以上目标**的同时检测与跟踪。
   - 平均每步耗时仅 **0.67秒**（在 Apple M4 Pro 上），展示了卓越的可扩展性。

3. **在 RadarScenes 数据集上的结果（Table II & III）**：
   - **性能媲美深度学习**：PiVoT 的 F1 分数（Val: 0.8016, Test: 0.7723）和 GOSPA 错误（Val: 2.8114, Test: 3.6252）与经过大量数据训练的 RadarGNNp 相当甚至更优。
   - **更高的精度**：PiVoT 的虚警（False）更低，表明其在杂波抑制方面更强。
   - **更快的速度**：PiVoT 处理每秒雷达数据平均耗时 **0.83秒**，而 RadarGNN 超过 4.8 秒，速度快了 **5倍以上**，且 PiVoT 满足实时性要求。

### 消融实验结果
论文虽未明确列出消融实验表格，但通过理论分析和设计选择体现了关键组件的重要性：
- **自适应出生移除**：通过图3展示了出生目标数量从初始的1312个迅速减少到收敛时的47个，证明了该机制对提升效率至关重要。
- **两阶段框架**：Stage 2 的存在性评估能够准确区分真实目标和由杂波引起的虚假聚类（如图3(c)所示）。

---

## 4. 关键结论和发现

### 主要发现
1. **PiVoT 是一种高效且强大的解决方案**：它成功地将变分推断应用于大规模、高杂波环境下的多目标检测与跟踪，解决了现有贝叶斯方法的效率瓶颈。
2. **无需训练也能达到顶尖性能**：PiVoT 证明了基于物理模型的纯推理方法可以在真实世界任务上达到与数据驱动的深度学习模型相媲美的性能。
3. **Doppler 信息的有效整合**：提出的 Doppler-augmented NHPP 模型为在模型驱动框架中利用 Doppler 信息提供了优雅且高效的途径。
4. **实时性与可扩展性兼备**：PiVoT 不仅能在标准硬件上实现实时处理，还能轻松扩展到上千个目标的场景。

### 方法的局限性
- **模型假设**：Doppler 增强模型假设刚体目标作直线运动，对于剧烈转弯或非刚体目标（如行人）可能产生模型失配。
- **后处理依赖**：为了报告移动目标，需要额外的确定性后处理步骤来筛选结果，这增加了系统复杂性。
- **参数敏感性**：虽然设计上追求鲁棒性，但某些超参数（如出生移除阈值 `L`）的选择仍需谨慎，不当设置可能导致有效目标被错误移除。

### 未来工作方向
- **更完善的移动目标指示**：对移动目标的判断进行正式的贝叶斯建模，而非依赖启发式后处理。
- **4D 雷达建模**：将高度维度纳入模型，以支持 4D 雷达。
- **融合更多信息**：整合雷达截面积（RCS）信息和进行目标分类。
- **扩展至更广领域**：探索去中心化传感器融合（decentralised sensor fusion）和非线性高斯测量模型。

</details>

---

### 11. [Beyond Parallel Tracking: Interactive Multi-Feature Fusion Drives Semantic Reconstruction from Non-invasive Brain Recordings](https://arxiv.org/abs/2607.12071)

**Authors**: Boda Xiao, Xiran Xu, Songyi Li, Yujie Yan, Xihong Wu, Heping Cheng, Jing Chen  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.12071v1  

#### Abstract
Continuous semantic reconstruction from non-invasive neural recordings remains limited by the representational mismatch between semantic feature spaces and neural coding patterns, which severely impedes cross-modal alignment between high-noise neural signals and target semantic features. Prior seman...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究针对**非侵入式脑信号（如 MEG 和 EEG）在连续语义重建中的代表性失配问题**（representational mismatch）。传统方法通常仅依赖单一语义特征（静态词嵌入如 Word2Vec 或动态上下文表示如 GPT），导致信息丢失，难以实现高噪声神经信号与复杂语言表征之间的准确跨模态对齐。

### 提出的新方法与新思路
提出了一种**交互式多特征融合框架**（Interactive Multi-Feature Fusion Framework），通过整合：
- **静态词汇属性**（Static Lexical Attributes，使用 W2V 表示）
- **动态上下文信息**（Dynamic Contextual Information，使用 GPT 表示）

并设计两种融合策略进行系统比较：
1. **Naive Concatenation**：线性拼接，作为基线。
2. **Multi-Head Cross-Attention**：非线性交互融合，模拟大脑中词汇记忆与上下文整合路径的协同机制。

该框架引入**交互门控机制**（interactive gating mechanism），使两类特征相互调制，生成统一的目标语义空间 `Efused`，从而更贴近人类语言处理的神经机制。

### 相比现有方法的优势
- ✅ **避免信息损失**：同时建模稳定词义与动态语境，克服单特征方法的信息瓶颈。
- ✅ **提升跨模态对齐精度**：非线性交叉注意力显著优于线性拼接，验证了“协作式”特征整合的有效性。
- ✅ **适用于低信噪比信号**：通过构建结构化语义目标空间，增强了模型对非侵入式高噪声信号的鲁棒性。
- ✅ **支持端到端文本生成**：不仅可用于语义重构，还可驱动 pre-trained LM 进行自然语言生成。

---

## 2. 核心实验方法和设置

### 数据集
使用 **SEM4Lang 数据集**：
- 包含 **60.7 小时**的 MEG 记录
- 来自 **12 名母语为中文的受试者**
- 听取 **60 段不同主题的音频故事**（每段 4–7 分钟）
- MEG 采样率：1,000 Hz，共 306 通道
- 预处理后保留 204 个平面梯度计，降采至 **40 Hz**

### 实验设置
#### 特征提取
- **静态特征**：使用 CBOW Word2Vec（300 维）提取每个 token 的固定向量 `e_w2v`
- **动态特征**：使用预训练中文 GPT 模型 *mengzi-GPT-Neo-base*，提取第 9 层 Transformer 的隐藏状态 `e_GPT`（768 维），上下文窗口为 5 个 token
- 所有特征时间维度与 MEG 对齐（线性插值至 40 Hz）

#### 融合模型架构
- **Concatenation Fusion**：直接拼接 `[e_w2v; e_GPT]` → 线性投影至 768 维
- **Cross-Attention Fusion**：
  - 先分别投影并归一化
  - 通过 Multi-Head Cross-Attention 实现双向交互：
    - `Attngpr = MultiheadAttn(H_gpt, H_w2v, H_w2v)`
    - `Attnw2v = MultiheadAttn(H_w2v, H_gpt, H_gpt)`
  - 拼接输出并通过 MLP 得到最终融合表示 `Efused`

#### 解码流程（两阶段框架）
1. **Stage 1: 语义特征重建 + 对比学习**
   - 使用 Brain Network（如 ConvConcatNet、BrainMagicNet）将 MEG 映射为预测的 `E_pred`
   - 构建 batch-wise 的 InfoNCE 损失函数，最大化真实 `(E_pred, Efused)` 对的相似性
   - 相似度基于时间步上的 Pearson 相关系数计算

2. **Stage 2: 文本生成**
   - 基于历史前缀和 Stage 1 输出的 `E_pred`，使用 beam search（宽度=100）从 *mengzi-GPT-Neo* 中生成候选词
   - 每个候选词经 Fusion Model 编码为 `E(w)`，与当前时间段的 `E_pred` 计算生理相关性得分 `S_brain(w)`
   - 按 `S_brain(w)` 排序并剪枝，选择最优路径

### 评估指标
#### 语义重构任务
- **Top-K Rank Accuracy**（K=1 和 10）：衡量从测试集中正确检索出对应语义向量的能力

#### 文本生成任务
- **BLEU-1**：字符级精确匹配
- **METEOR**：综合考虑精确率与召回率的结构对齐指标
- **BERTScore**：基于 BERT 隐藏层的上下文语义相似度

#### 基线对比
- 单一特征方法：
  - W2V-only
  - GPT-only
- 多特征融合方法：
  - Concatenation（线性拼接）
- Null Baseline：用随机分数替代 `S_brain(w)`，检验是否由语言模型主导而非神经信号引导

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 语义重构性能（Top-1 / Top-10 准确率）
| 方法 | 3s 窗口（Top-1） | 5s 窗口（Top-1） | 10s 窗口（Top-1） |
|------|------------------|------------------|--------------------|
| W2V | ~10% | ~15% | ~20% |
| GPT | ~20% | ~25% | ~30% |
| Concat | ~25% | ~30% | ~35% |
| **Cross-Att** | **~30%** | **~38%** | **~45%** |

> 注：具体数值因模型略有差异，但趋势一致。Cross-Att 在所有时间窗下均显著优于其他方法（*p < 0.05 至 ***p < 0.001）

#### 文本生成性能（以 ConvConcatNet + Cross-Att 最优组合为例）
| 指标 | Cross-Att | Concat | GPT | W2V | Null Baseline |
|------|----------|--------|-----|-----|---------------|
| **BLEU-1** | **15.58±1.16** | 14.38±1.49 | 13.47±1.40 | 11.76±1.40 | 10.77 |
| **METEOR** | **9.23±0.89** | 8.40±1.49 | 7.76±0.98 | 6.85±1.01 | 5.89 |
| **BERTScore** | **53.62±0.45** | 53.03±0.77 | 52.68±0.56 | 52.33±0.75 | 51.50 |

> 所有指标均显著高于 Null Baseline，说明生成结果是由神经信号驱动，而非语言模型幻觉。

### 与基线方法的对比结果
- **性能层级明确**：  
  `Cross-Att > Concat > GPT > W2V > Null`
- **非线性融合优势显著**：Cross-Attention 方法在 Top-K 检索和文本生成中全面领先，尤其在长时窗（10s）表现突出。
- **GPT 优于 W2V**：表明上下文信息比静态词义更重要，但仍受限于边界模糊问题。
- **Concat 优于单独特征**：证明多特征融合本身有效，但仍有提升空间。

### 消融实验结果
- **移除 Cross-Attention 改为 Concat**：性能下降明显，验证其关键作用。
- **不同时间窗长度影响**：随着 `seglen` 从 3s 增加到 10s，Top-10 准确率大幅提升，说明**更长时间序列有助于平滑噪声、增强语义稳定性**。
- **不同 Brain Encoder 表现**：
  - **ConvConcatNet** 和 **BrainMagicNet** 表现最佳
  - 特别是 BrainMagicNet + Cross-Att 在 5s/10s 任务中达到最高天花板

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **人类语言理解依赖双通路协同**：大脑同时利用稳定词汇记忆和动态上下文整合来消除歧义（N400 时间窗内交互）。单一特征无法充分建模这一过程。
2. ✅ **非线性交互融合显著优于线性拼接**：Multi-Head Cross-Attention 成功模拟了神经层面的“协作式加工”，实现了更高精度的跨模态对齐。
3. ✅ **多特征融合提供正则化效应**：通过约束解码目标在结构化的语义空间内搜索，提升了对非平稳、高噪声非侵入信号的鲁棒性。
4. ✅ **可实现高质量非侵入式脑到文本生成**：首次展示了基于 MEG 的流畅自然语言生成能力，且结果由神经信号主导（非 LM 幻觉）。

### 方法的局限性
- 🚫 当前仅在**听觉叙事任务**中验证，尚未扩展至主动语言产生（如自由说话或想象）。
- 🚫 所有实验基于**特定受试者群体**（12 名中国人），个体间 variability 较大，缺乏跨被试泛化能力。
- 🚫 文本生成仍处于**初步可读水平**（BLEU-1 ≈ 15.6），远未达到实用化标准。
- 🚫 模型依赖大量预训练语言模型（GPT、Word2Vec），存在**计算开销大、解释性弱**的问题。

### 未来工作方向
1. **Cross-Subject Generalization**：引入对抗域适应（adversarial domain adaptation）减少个体差异，降低新用户校准成本。
2. **Open-Source Benchmarking**：计划开源完整的编码器、预处理模块和生成系统，建立标准化 Python 库，推动 BCI 社区协作发展。
3. 扩展至其他模态（如 EEG）和任务场景（如视觉刺激、内部言语）。
4. 探索更轻量化、可部署的实时解码架构。

---

> **总结一句话**：  
> 本文通过引入 **Interactive Multi-Feature Fusion + Cross-Attention** 框架，成功模拟了人脑语言处理的双通路协同机制，在非侵入式 MEG 数据上实现了当前最优的语义重建与文本生成性能，为未来无创 Brain-Computer Interfaces 提供了新的技术路径。

</details>

---

### 12. [Fine-Tuned Multi-Agent Framework for Detecting OCEAN in Life Narratives](https://arxiv.org/abs/2607.12215)

**Authors**: Rasiq Hussain, Darshil Italiya, Joshua Oltmanns, Mehak Gupta  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.12215v1  

#### Abstract
Accurately assessing personality from text is challenging because traits are latent, context-dependent, and often subtly expressed across long narratives. Large language models (LLMs) offer new opportunities by processing extensive textual contexts, but pretraining of these models can induce latent ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fine-Tuned Multi-Agent Framework for Detecting OCEAN in Life Narratives

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
准确从长文本（如人生叙事）中识别 **OCEAN**（Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism）人格特质具有挑战性，原因包括：
- 人格是**潜变量**（latent），表达隐含且上下文依赖；
- 单一 **LLM** 在推理时可能因预训练引入“类人格”偏见，导致不一致预测；
- 简单关键词匹配或单一模型提示（prompting）难以捕捉复杂、分布式的语言线索。

### 提出了什么新方法或新思路
提出了一种**细粒度多智能体框架（fine-tuned multi-agent framework）**，其核心设计如下：
- **子智能体（sub-agents）专业化**：为每个 OCEAN 特质构建三个子智能体，分别专注于 **HIGH、NEUTRAL、LOW** 三种倾向；
- **心理测量学引导（psychometric supervision）**：利用 **IPIP-NEO** 量表中的**facet keys**（如“喜欢冒险”对应高开放性）作为提示输入，使子智能体聚焦于理论支持的行为信号；
- **掩码语言建模微调（MLM fine-tuning）**：通过在情感词和非情感词上进行掩码预测任务，使子智能体内化与特定人格水平相关的语言模式；
- **法官智能体聚合（judge agent aggregation）**：由一个独立的 judge LLM 综合三个子智能体的输出（含证据和置信度），做出最终分类决策。

### 相比现有方法的优势
- ✅ **减少偏见**：多视角推理缓解了单一模型的固有偏差；
- ✅ **提升可解释性**：每个子智能体提供基于 facet 的证据支持，judge 进行推理整合；
- ✅ **增强鲁棒性**：通过分解与聚合机制，能更好处理模糊或矛盾线索；
- ✅ **理论对齐**：结合 IPIP-NEO 理论框架，确保推理过程符合心理学定义。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **SPAN Life Narrative Interviews**（St. Louis Personality and Aging Network）
  - 包含 1,535 条人生叙事转录文本；
  - 涵盖家庭、工作、成长经历等主题；
  - 每条记录标注连续型 OCEAN 分数，并通过三分位法（tercile binning）划分为 HIGH / NEUTRAL / LOW 三类（见 Table 1）；
  - 数据按 50% 微调 + 50% 测试划分。

### 实验设置和评估指标
- **主干模型**：
  - 子智能体：`Mistral-7B-Instruct`（共15个，每特质3个）；
  - 法官智能体：`Qwen-7B-Instruct`（更大上下文窗口，适合聚合）；
- **微调方式**：
  - 使用 **LoRA**（Low-Rank Adaptation），仅更新 `q_proj`, `k_proj`, `v_proj`, `o_proj` 层；
  - 掩码策略：30% token 被掩码（15% 情感词来自 SenticNet，15% 非情感词）；
  - LoRA 参数：rank=4，α=8，dropout=0.05，约 4.2M 可训练参数/agent；
- **评估指标**：
  - 主要：**Macro-F1**（平衡各类别重要性）；
  - 辅助：Accuracy（按类别）、Perplexity、Cosine Similarity（语义对齐分析）；
- **推理流程**：
  - 子智能体依次运行 → 输出 JSON 格式的二元判断、证据、置信度；
  - GPU 内存逐次清理以节省资源；
  - Judge 模型常驻内存，接收所有输出并生成最终标签。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **判别式模型** | RoBERTa 微调（multi-label 分类） |
| **单智能体 LLM** | Mistral/Qwen + Chain-of-Thought（CoT）直接预测 |
| **多智能体变体** | All-Mistral、All-Qwen、Mistral-sub + Qwen-judge（本文方法） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Macro-F1）
| Model | O | C | E | A | N | Avg |
|-------|----|----|----|----|----|-----|
| RoBERTa | 0.256 | 0.272 | 0.248 | 0.261 | 0.239 | 0.255 |
| Single-Agent (Qwen-CoT) | 0.381 | 0.414 | 0.383 | 0.386 | 0.353 | 0.383 |
| **Proposed (Mistral-sub + Qwen-judge)** | **0.424** | **0.366** | **0.416** | **0.415** | **0.458** | **0.416** |

👉 **平均 Macro-F1 提升约 8.6%**，显著优于最强单智能体基线。

### 与基线方法的对比结果
- 本文方法在 **Openness、Extraversion、Agreeableness、Neuroticism** 上全面领先；
- **Conscientiousness** 表现略低于单智能体 Qwen，表明该特质可能更适合整体推理而非分解；
- 使用不同模型分工（Mistral 提取 + Qwen 聚合）效果最佳，说明**证据提取与综合任务需不同能力模型**。

### 消融实验结果（Ablation Study）
| Ablation | Avg Macro-F1 | Δ vs Full |
|--------|---------------|----------|
| Full Multi-Agent | **0.416** | — |
| No IPIP-NEO keys | 0.351 | ↓15.6% |
| No MLM Fine-tuning | 0.369 | ↓11.3% |
| No Trait-level Cues | 0.372 | ↓10.6% |
| No Sub-agents | 0.383 | ↓7.9% |

👉 **IPIP-NEO facet keys 最关键**，移除后性能下降最大，说明心理测量学引导至关重要。

### MLM 微调有效性验证
| Agent Type | Avg Test Acc | Acc@5 | Perplexity ↓ |
|----------|--------------|--------|-------------|
| Pretrained Baseline | ~9.5% | ~13% | ~8.3 |
| Fine-tuned (High/Low/Neutral) | **52.7%** | **80.2%** | **~7.0** |

👉 MLM 微调极大提升了子智能体对人格相关语言模式的理解能力。

### 定向语义对齐分析（Directional Semantic Alignment）
- 使用 Sentence-BERT 计算子智能体输出证据与 IPIP-NEO keys 的 cosine similarity；
- 结果显示完整模型下，**HIGH 智能体更接近 High keys，LOW 更接近 Low keys**；
- 移除 IPIP keys 后部分 agent 出现负 margin（即反向对齐），进一步证明其必要性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **多智能体架构优于单模型推理**：通过引入 HIGH/LOW/NEUTRAL 多视角，系统能有效识别并调和冲突线索，避免过拟合表面特征（如“换工作多”≠高开放性）；
2. **心理测量学引导显著提升准确性与可解释性**：IPIP-NEO facet keys 是实现理论对齐的关键，防止 LLM 陷入泛化解释；
3. **MLM 微调有助于内化人格语言模式**：即使无显式标签监督，也能通过掩码重建任务学习情绪与结构特征；
4. **角色分工模型更优**：小模型（Mistral）适合作为专家提取局部证据，大模型（Qwen）擅长整合全局信息；
5. **NEUTRAL 类最难预测**：中间态表达模糊，缺乏强烈行为信号，准确率普遍偏低（见 Table 3）。

### 方法的局限性
- ❌ **NEUTRAL 分类性能弱**：中等水平人格表达不够鲜明，当前框架仍难精准识别；
- ❌ **依赖高质量转录文本**：数据来自临床访谈，结构清晰；在社交媒体等噪声文本中泛化能力未知；
- ❌ **计算成本高**：需运行15个子智能体 + 1个 judge，推理延迟远高于单模型；
- ❌ **语言与文化限制**：目前仅适用于英语 IPIP-NEO 框架，跨语言或多文化适应未探索。

### 未来工作方向
- 引入动态投票机制或加权融合策略优化 judge 判断；
- 探索轻量化子智能体部署方案（如蒸馏）降低开销；
- 扩展至其他心理构念（如价值观、动机）检测；
- 开发跨文化版本，适配非西方人格理论体系；
- 将框架应用于临床辅助诊断或个性化教育场景（需严格伦理审查）。

--- 

> **总结一句话**：  
> 本论文提出一种结合 **MLM 微调 + IPIP-NEO 引导 + 多智能体推理** 的新型人格识别框架，在长文本中实现了更准确、可解释、理论对齐的 OCEAN 特质检测，为 LLM 在心理计算领域的可信应用提供了新范式。

</details>

---

### 13. [Evaluating Health Misinformation in Low-Resource Languages: Integrating Small Language Models with a Culturally-Sensitive Responsible NLP Framework (Bangla as a Case Study)](https://arxiv.org/abs/2607.12336)

**Authors**: Farnaz Farid, Raihan Alam, Al Al-Areqi, Farhad Ahamed, Muhammad Hassan Khan, Sadia Hossain, Irena Veljanova, Anika Tabassum Binte Hossain  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.12336v1  

#### Abstract
Artificial Intelligence (AI) technologies, while serving as a foundational enabler for modern social media and digital health services, exert a bivalent effect by simultaneously acting as a combatant against and a spread vector for misinformation. A prevalent challenge in mitigating this issue arise...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究聚焦于**低资源语言（Low-Resource Languages）** 中健康虚假信息（Health Misinformation）检测的严峻挑战，特别是针对**文化与语言多样性（CALD）社区**。当前主流的 **Large Language Models (LLMs)** 在非英语语境下表现不佳，主要原因包括：
- 缺乏高质量、标注良好的健康虚假信息数据集；
- 忽视语言细微差别和文化背景；
- 高昂的微调成本限制其在资源受限环境中的部署。

### 提出的新方法与创新
论文提出了以下四项主要贡献：

1. **构建首个专家验证的孟加拉语（Bangla）健康虚假信息数据集**  
   将英文数据集 *MHMisinfo: Video-based Mental Health Misinformation* 翻译为 Bangla，并由母语领域专家进行语言准确性、语义保真度和医学正确性审查，填补了该领域的空白。

2. **系统评估多种 Small Language Models (SLMs) 在 Bangla 虚假信息检测中的性能**  
   对比了 Phi-4、Qwen3 系列、Llama-3.1、Gemma-3 和 Ministral-8B 等多个 SLM 在跨语言虚假信息检测任务上的表现。

3. **提出一种基于负责任自然语言处理（Responsible NLP, RNLP）的多维评估框架**  
   超越传统的技术指标（如 Precision、Recall），引入六个维度综合评估：
   - **Content Accuracy (CA)**：科学准确性
   - **Misinformation Detection (M)**：误导性识别
   - **Harm and Risk Assessment (HR)**：潜在危害
   - **Cultural and Linguistic Sensitivity (CLS)**：文化语言敏感性
   - **Communication Quality (CQ)**：沟通质量
   - **Performance Accuracy (PA)**：模型性能可靠性

4. **设计并实现一个原型应用 MALAK**  
   一个面向 CALD 社区用户的移动工具，支持用户输入健康声明，自动分析其真实性、风险等级和可信来源，是所提框架的实际落地尝试。

### 相比现有方法的优势
- **更适用于低资源场景**：SLMs 比 LLMs 更轻量、成本更低，适合本地部署；
- **更强的文化适应性**：通过专家参与和多维框架，避免“西方中心主义”偏见；
- **更全面的评估体系**：不仅看准确率，还考虑社会影响与传播效果；
- **可操作性强**：提供可复现的数据集、代码和原型系统。

---

## 2. 核心实验方法和设置

### 数据集
- **源数据集**：[MHMisinfo](https://doi.org/10.1609/icwsm.v19i1.35875)，包含 739 个视频（YouTube 和 BitChute）及 135,372 条评论。
- **目标数据集**：将清洗后的 286 个视频文本翻译为 **Bangla**，采用“人机协同”方式（Azure + 双母语专家校对），确保语言与文化适配性。

### 实验设置
- **任务**：从 Bangla 视频文本中提取健康相关主张（Claim Extraction），并判断是否构成虚假信息（Misinformation Detection）。
- **模型**：测试多个 SLM，包括：
  - Phi-4 (14B)
  - Qwen3-4B / Qwen3-8B / Qwen3-14B
  - Llama-3.1-8B-Instruct
  - Gemma-3-12B-IT
  - Ministral-8B-2512
- **训练方式**：未进行微调（no fine-tuning），仅使用 **Prompt Engineering** 进行零样本或少样本推理。

### 评估指标
使用 **micro-averaged** 和 **macro-averaged** 的：
- **Precision**
- **Recall**
- **F1 Score**

此外，在提出的多维框架中，采用 **Entropy-TOPSIS** 多准则决策方法整合六维评分，生成最终的“接近系数”（Closeness Coefficient, CCi）用于风险分类。

### 基线对比
- 主要与人类专家标注的 **ground-truth** 数据集对比；
- 同时比较不同 SLM 之间的性能差异；
- 间接对比传统深度学习模型（如 CNN）在类似任务中的表现（引用文献 [39]）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Bangla 数据集）

| Model | Prec.micro | Rec.micro | **F1micro** | Prec.macro | Rec.macro | **F1macro** |
|-------|------------|-----------|-------------|------------|-----------|-------------|
| **Phi-4** | 80.86 | 52.65 | **63.77** | 12.70 | 24.90 | **15.49** |
| Qwen3-8B | 85.65 | 41.24 | 55.68 | 12.37 | 25.44 | 15.36 |
| Qwen3-4B | 85.60 | 32.60 | 47.23 | 12.50 | 24.60 | 15.50 |
| Llama-3.1-8B | 50.24 | 15.24 | 23.39 | 8.78 | 13.45 | 9.71 |
| Gemma-3-12B | 40.67 | 9.15 | 14.94 | 5.40 | 10.33 | 6.66 |
| Ministral-8B | 36.36 | 11.13 | 17.04 | 5.45 | 13.22 | 7.23 |

> ✅ **Phi-4 表现最优**，在 F1micro 上达到 **63.77**，显著优于其他模型。

### 与基线方法的对比结果
- **Phi-4 vs Qwen3-8B**：
  - Qwen3-8B 精度更高（85.65 vs 80.86），但召回率明显偏低（41.24 vs 52.65），说明其策略保守，漏检较多；
  - Phi-4 在 **Precision 与 Recall 之间取得了最佳平衡**。
- **多语言预训练优势**：
  - Phi 和 Qwen 系列因更强的多语言编码能力，在 Bangla 上表现更好；
  - Llama、Gemma、Ministral 因缺乏足够的非英语训练数据，表现较差。
- **与传统模型对比**：
  - 引用文献 [39] 显示 CNN 模型在虚假信息检测中准确率为 58.6%，而 Phi-4 达到 63.77% F1，表明 SLM 已具备竞争力。

### 消融实验与观察（隐含分析）
尽管未明确列出消融实验，但文中进行了深入分析：
- **高 Precision、低 Recall 是普遍现象**：所有模型均倾向于“宁可错杀一千，不可放过一个”，导致大量 **False Negatives**；
- **Macro-F1 极低**（最高仅 ~15.5）：表明模型在不同视频间表现极不稳定，存在严重的 **video-level inconsistency**；
- **跨语言性能下降明显**：同一模型在英文数据上表现优于 Bangla，凸显了跨语言迁移的挑战。

---

## 4. 关键结论和发现

### 主要发现
1. **Phi-4 是目前最适合 Bangla 健康虚假信息检测的 SLM**，因其在精度与召回之间实现了理想平衡。
2. **SLMs 具备成为 LLM 替代方案的潜力**，尤其适合资源受限、需本地部署的 CALD 场景。
3. **仅依赖技术指标不足以评估虚假信息系统**：必须结合文化敏感性、传播质量和潜在危害等维度。
4. **字面机器翻译可能加剧信息鸿沟**：将复杂医学术语直接翻译成 Bangla 可能导致“术语墙”，反而阻碍信息获取。
5. **模型对正式学术语言有偏好偏差**：即使内容具有高度误导性和危害性，只要语言形式规范，仍可能被误判为可靠（如疫苗自闭症阴谋论视频仅被评为“中等风险”）。

### 方法的局限性
- **未进行微调**：仅依赖 Prompting，性能仍有提升空间；
- **翻译过程可能存在语义漂移**：虽经专家审核，但仍无法完全消除文化转译误差；
- **数据来源有限**：仅来自 YouTube 和 BitChute，未覆盖 WhatsApp、Facebook 等 CALD 社区常用平台；
- **框架尚未大规模验证**：Entropy-TOPSIS 框架仅在小样本（6 个视频）上测试，需进一步实证；
- **缺乏实时动态更新机制**：无法应对快速演变的虚假信息模式。

### 未来工作方向
1. **对 SLMs 进行领域微调（fine-tuning）和超参数优化**，以提升在 Bangla 医疗语境下的表现；
2. **构建更多低资源语言的专家标注数据集**，扩展至阿拉伯语、越南语等；
3. **开展 Content Validity Index 和 Inter-rater Reliability 测试**，验证多维框架的有效性；
4. **开发支持多模态输入（音频、视频）的检测系统**；
5. **推动 MALAK 类工具的实际部署与用户研究**，检验其在真实社区中的接受度与影响力；
6. **探索不对称惩罚规则**：当内容同时具备高危害（HR=3）和严重误导（M=0）时，即使模型提取稳定也应触发高风险警报。

---

> 📌 **总结一句话**：  
> 本文开创性地将 **Small Language Models** 与 **Responsible NLP 多维框架** 结合，为低资源语言环境下的健康虚假信息检测提供了兼具技术可行性与社会伦理考量的完整解决方案，标志着从“纯算法驱动”向“人本智能治理”的重要迈进。

</details>

---

### 14. [PQFA: Parallel Quantum Feature Augmentation of Fused Representations for Multimodal Classification](https://arxiv.org/abs/2607.13466)

**Authors**: Mingzhu Wang, Yun Shang  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.13466v1  

#### Abstract
Most multimodal learning methods improve how heterogeneous representations are aligned and fused, while post-fusion enhancement remains less explored. We propose Parallel Quantum Feature Augmentation (PQFA), a hybrid quantum-classical framework that applies multiple shallow variational quantum circu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PQFA: Parallel Quantum Feature Augmentation of Fused Representations for Multimodal Classification

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大多数多模态学习方法集中在如何对齐和融合异构表示上，而**后融合增强（post-fusion augmentation）** 阶段的研究相对不足。此外，现有量子或多模态量子混合模型往往未能清晰分离量子组件的独立作用，其性能提升可能源于更强的编码器、更大的参数量或更复杂的融合结构，而非量子机制本身。

本文旨在解决以下问题：
- 如何在不替换强大经典融合主干的前提下，有效增强已融合的多模态表示？
- 量子模块是否能作为轻量级、高效的后处理增强工具，而非端到端替代方案？
- 性能增益是否真正来自量子特征变换，而非简单的宽度扩展或参数增加？

### 提出的新方法：PQFA
提出 **Parallel Quantum Feature Augmentation (PQFA)** ——一种混合量子-经典框架，将多个浅层变分量子电路（PQC）并行应用于**已经融合后的多模态表示**之上。

#### 核心设计思想：
- **冻结预训练编码器**（RoBERTa 和 ViT），提取文本与图像特征。
- 经过双向交叉注意力（bidirectional cross-attention）、注意力池化（attentive pooling）和自适应门控融合（adaptive gated fusion）生成融合表示 $ z_c $。
- 将 $ z_c $ 进行**振幅编码**（amplitude encoding）输入多个并行的浅层 PQC。
- 各量子分支输出测量期望值（如 Pauli-Z），拼接为量子增强向量 $ q $。
- 最终表示为 $ h = [z_c; q] $，送入分类头进行预测。

### 相比现有方法的优势
| 方面 | PQFA 的优势 |
|------|-------------|
| **架构定位** | 不改变主干融合路径，仅在后融合阶段引入量子增强，职责明确 |
| **参数效率** | 仅需约 **2.2K 可训练参数**用于增强分支，远低于宽度匹配的 MLP-Aug（24.0K） |
| **可控比较** | 明确对比无量子基线（NoQ）和宽度匹配的经典 MLP 增强（MLP-Aug），确保公平性 |
| **鲁棒性增强** | 在缺失模态场景下表现更优，尤其当主导模态（如文本）严重退化时 |
| **非线性表达能力** | 利用量子态的高维希尔伯特空间实现结构化非线性映射，超越随机或纯经典宽度过扩 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型 | 任务 | 样本数 | 类别数 | 特点 |
|--------|------|-------|--------|--------|------|
| **MM-IMDb** | 图像 + 文本 | 多标签分类 | 3,894（测试集） | 23 个电影类型 | 包含海报与剧情摘要，标签不平衡 |
| **N24News** | 图像 + 文本 | 单标签分类 | 完整划分未详述 | 24 新闻主题 | 来自《纽约时报》，图文配对新闻 |

### 实验设置
- **编码器固定**：RoBERTa（文本）、ViT（图像）参数冻结。
- **投影维度**：统一映射至 128 维共享空间。
- **量子设置**：
  - 使用 **7 个量子比特**（$ 2^7 = 128 $，适配振幅编码）
  - 每个量子分支输出 7 个 Pauli-Z 测量值
  - 并行分支数 $ K = 8 $（通过验证选择）
  - 总量子增强维度：$ 8 \times 7 = 56 $
- **MLP-Aug 基线**：构造相同输出维度（56）的前馈网络，保证宽度一致。
- **优化器**：Adam，经典部分学习率 $ 1\times10^{-4} $，量子部分 $ 1\times10^{-5} $
- **重复次数**：所有实验基于 **5 个随机种子**取平均

### 评估指标
| 任务类型 | 主要指标 | 公式说明 |
|---------|----------|----------|
| 多标签（MM-IMDb） | **Micro-F1**, **Macro-F1** | Micro 聚焦整体决策质量，Macro 更关注小类均衡 |
| 单标签（N24News） | **Accuracy (%)** | 正确分类样本比例 |

### 基线方法对比
| 类型 | 对比方法 |
|------|----------|
| **参考基准**（跨不同设置） | LRMF, MFM, MBT, UniS-MMC, SDDA, M3CoL 等 |
| **控制变量基线**（相同设置） |  
| - 单模态 | Text-only, Image-only |
| - 融合方式 | Average fusion, Concat fusion |
| - 主干模型 | NoQ（无增强） |
| - 经典增强 | MLP-Aug（宽度匹配） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（表3）

| 方法 | MM-IMDb Micro-F1 | MM-IMDb Macro-F1 | N24News Acc (%) |
|------|------------------|------------------|------------------|
| NoQ（无增强） | 67.57 | 60.98 | 84.70 |
| MLP-Aug（宽度匹配） | 67.62 | 61.22 | 84.23 |
| **PQFA（本文）** | **68.28** | **61.85** | **85.35** |

> ✅ PQFA 在两个数据集上均显著优于所有基线，且仅使用 **2.2K 增强参数** vs MLP-Aug 的 **24.0K**

### 与基线方法的对比结果
- **vs NoQ**：PQFA 在 MM-IMDb 上提升 +0.71 Micro-F1，+0.87 Macro-F1；在 N24News 上提升 +0.65%
- **vs MLP-Aug**：
  - MM-IMDb：+0.66 Micro-F1，+0.63 Macro-F1
  - N24News：+1.12%，且 MLP-Aug 反而略低于 NoQ，表明盲目加宽不一定有益
- **参数效率**：PQFA 增强分支参数仅为 MLP-Aug 的 **~9.2%**，却带来更大收益

### 消融实验结果
#### （1）融合策略消融
- 固定编码器下，**gated fusion > concat > average**，证明自适应门控有效
- 所有后续比较均建立在此最强主干之上

#### （2）增强机制消融
| 变体 | Micro-F1 | Macro-F1 | 发现 |
|------|----------|----------|------|
| PQFA（完整） | 68.28 | 61.85 | — |
| PQFA-NoEnt（移除纠缠） | 67.50 | 61.35 | 性能下降 → 缠结操作重要 |
| PQFA-FrozenQ（冻结量子参数） | 67.70 | 60.23 | 需训练 → 任务驱动优化关键 |
| RFF-Aug（随机傅里叶特征） | 67.78 | 60.47 | 非结构化映射无效 |
| MLP-Aug-2x（双倍宽度） | 67.98 | 60.95 | 加宽仍不如 PQFA |

> 🔍 结论：PQFA 的优势**不能归因于宽度扩展、随机映射或未训练量子变换**

#### （3）分支数量敏感性（K=1~9）
- 最佳性能出现在 $ K=8 $，进一步增加导致性能下降
- 表明存在容量与泛化的权衡，并非越多越好

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **PQFA 是有效的后融合增强策略**：
   - 在保持强大经典融合主干的同时，通过少量可训练量子参数即可持续提升性能。
   
2. ✅ **量子增强具有参数高效性**：
   - 仅用 **2.2K 参数**即超越参数量超10倍的 MLP-Aug，显示量子特征映射的信息密度更高。

3. ✅ **具备更强的缺失模态鲁棒性**：
   - 当文本缺失率达 90% 时，PQFA 在 MM-IMDb 上仍达 **36.96 Micro-F1**，显著优于 NoQ（27.78）和 MLP-Aug（34.51）
   - 表明量子分支能在主导模态失效时提供有价值的补充表示

4. ✅ **决策层面改善明显**：
   - 配对错误转移分析显示，PQFA 相比 MLP-Aug **多纠正 181 个错误决策**（McNemar p < 0.001）
   - 改进集中在 MLP-Aug 存在可修正错误的类别（如 Film-Noir、Animation）

5. ✅ **特征空间更具结构性**：
   - PCA 分析显示 PQFA 增强特征既不过度集中也不过度分散，形成“紧凑且任务相关”的非线性表示
   - 优于 RFF-Aug（太散）、PQFA-NoEnt（太集中）

6. ✅ **量子状态稳定且多样化**：
   - 噪声模拟显示，在 depolarizing、bit-flip 等噪声下 Micro-F1 下降 < 0.2 pts，读出稳定
   - 并行量子分支产生不同的 entanglement entropy 分布，体现结构多样性

7. ✅ **门控融合行为合理**：
   - Gate weight 分析显示模型采用**特征维度级路由**，部分维度偏好文本，部分偏好图像
   - 整体偏向图像侧（mean gate ~0.4），但关键语义仍依赖文本输入

### 方法的局限性
- **完全基于经典模拟**：未在真实量子硬件上运行，忽略 shot noise、readout error、编译开销等实际限制
- **振幅编码成本高**：将经典数据加载为量子态在 NISQ 设备上仍具挑战
- **规模受限**：受限于 qubit 数量（仅 7 qubits），难以处理更高维表示
- **任务范围有限**：目前仅验证于图像-文本分类，尚未拓展至生成、检索等任务

### 未来工作方向
- 探索 **hardware-aware implementation**：结合真实设备噪声模型、有限采样、误差缓解技术
- 研究 **finite-shot evaluation** 与梯度估计的影响
- 扩展至更多 **multimodal tasks**：如视觉问答、跨模态检索、视频理解
- 探索 **circuit scaling** 与 **branch scaling** 的协同效应
- 引入 **dropout-like classical baselines** 以更好区分噪声鲁棒性来源

---

> 📌 **总体结论**：  
> PQFA 成功展示了**浅层并行量子电路**可以作为一种**参数高效、结构化、鲁棒性强的后融合增强模块**，在不取代经典主干的前提下，为多模态分类注入额外判别能力。该工作强调了在混合量子-经典系统中进行**受控实验设计**的重要性，为未来量子机器学习在现实应用中的角色提供了新视角。

</details>

---

### 15. [RF Spectrogram Anomaly Detection with Quantum Kitchen Sinks: Architecture, Representation, and Hardware Validation](https://arxiv.org/abs/2607.13897)

**Authors**: Abdallah Aaraba, Alexis Vieloszynski, Remon Polus, Ola Ahmad, Soumaya Cherkaoui  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.13897v1  

#### Abstract
The broadcast nature of wireless channels exposes radio-frequency (RF) networks to anomalous and malicious transmissions, making anomaly detection a fundamental requirement for secure spectrum management. Quantum Kitchen Sinks (QKS) offer a lightweight hybrid quantum feature map suitable for near-te...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RF Spectrogram Anomaly Detection with Quantum Kitchen Sinks: Architecture, Representation, and Hardware Validation — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
无线频谱的开放性和广播特性使其容易受到**异常或恶意传输**（如 jamming、spoofing、chirp 干扰等）的影响。传统的 RF anomaly detection 方法在动态、高维信号环境中面临计算复杂度高、泛化能力差等问题。本文聚焦于利用 **Near-term 量子设备**（NISQ）实现高效且实用的 RF 异常检测，尤其是在基于 spectrogram 的结构化信号上。

### 🚀 提出的新方法与创新
1. **扩展 Quantum Kitchen Sinks (QKS) 架构**：
   - 在标准浅层 QKS 模板基础上引入 **multi-depth 数据重上传（data re-uploading）** 和 **ring entanglement**，增强特征映射的表达能力。
   - 保持轻量级经典读出头（classical readout），适合 NISQ 设备部署。

2. **提出五阶段消融协议（five-stage ablation protocol）**：
   - 验证锁定（validation-locked）、无数据泄露（leakage-free）的设计，系统性分离以下因素影响：
     - 浅层架构（shallow architecture）
     - 重上传深度（re-uploading depth）
     - episode 预算分配
     - 输入表示（input representation）
     - 经典读出模型（classical readout）

3. **真实数据 + 真实硬件双重验证**：
   - 使用 **实测 sub-6 GHz 蜂窝信号**（来自 LTE 网络）构建 spectrogram 数据集。
   - 在 **IBM 的 ibm_quebec QPU** 上进行真实量子硬件验证，评估噪声环境下的鲁棒性。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **效率** | QKS 是随机特征生成器，无需训练整个变分电路，训练简单、收敛快。 |
| **可解释性** | 通过结构化消融实验明确识别关键影响因子（如 DCT 表示优于 PCA）。 |
| **实用性** | 支持真实测量数据 + 真实 QPU 执行，推动 QML 向实际无线监控系统落地。 |
| **性能提升** | 在多个 representation-readout 组合下均超越匹配的经典直接读出基线（direct-readout baseline）。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **来源**：基于公开数据集 [22]，包含从真实 LTE 发射中采集的 **sub-6 GHz IQ 信号**。
- **处理流程**：
  1. 原始 IQ 信号 → 添加合成异常信号（Python 仿真注入）。
  2. 使用 **STFT（Short-Time Fourier Transform）** 将时域信号转换为 **spectrogram**（分辨率 400×400）。
- **异常类型**（共三类）：
  - **Chirp signal**：未经授权的频率扫描传输。
  - **Barrage Jamming**：宽带密集干扰。
  - **Frequency-Hopping Jamming**：跳频噪声干扰。
- **信噪比控制**：使用 **JSR（Jamming-to-Signal Ratio）** 控制干扰强度（范围：-10 dB 到 5 dB，步长 2 dB）。
- **数据划分**：
  - 训练集：21,600 个样本（含正常与异常对）
  - 测试集：8,124 个独立样本

### ⚙️ 实验设置
- **输入表示方式对比三种**：
  1. `raw`：原始 spectrogram flatten 成向量（160000 维）
  2. `DCT`：2D-DCT 变换后保留低频系数块（如 64×64, 128×128）
  3. `PCA`：主成分分析降维（维度 m ∈ {32,128,…,2048}）

- **QKS 特征提取器设计**：
  - 多层数据重上传（depth D ∈ {1,2,4,6,8,10}）
  - 支持 ring entanglement（CZ gates 连接相邻 qubits）
  - 每 episode 输出量子特征块 via 测量所有单体与双体 Pauli 观测量（On）

- **读出模型（Readout）对比五种**：
  1. Linear SVM
  2. Logistic Regression
  3. RFF + SVM / Logistic
  4. Nystrom + Logistic
  > 所有读出模型仅作用于最终拼接的 QKS 特征向量

- **评估指标**：
  - **AUROC**（Area Under ROC Curve）
  - **F1 Score**（阈值固定为 0.5）

- **基线方法**：
  - **Matched direct-readout baseline**：相同输入表示 + 相同读出模型，但跳过 QKS 映射，直接输入经典分类器。
  - 对比确保公平性：仅改变是否使用 QKS 特征映射。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据
| 指标 | 最佳结果 |
|------|--------|
| **Test AUROC** | **0.8778**（dct64x64 + LinearSVM） |
| **Test F1** | **0.7995**（dct64x64 + LinearLogistic） |

> 注：论文结论部分提到“best test AUROC of 0.8773”应为笔误，正文中 Table III 显示最高为 **0.8778**

### 🔁 与基线方法对比结果
- 在全部 **15 种 representation-readout 组合** 中：
  - **QKS 均优于对应的 direct-readout 基线**
  - AUROC 提升范围：**+0.0952 ~ +0.1839**
  - F1 提升范围：**+0.1104 ~ +0.1451**

#### 典型增益示例（见 Table III）：
| Representation | Readout | Direct AUROC | QKS AUROC | ΔAUROC |
|----------------|---------|-------------|-----------|--------|
| dct64x64 | LinearSVM | 0.6939 | **0.8778** | **+0.1839** |
| dct128x128 | LinearSVM | 0.6945 | 0.8763 | +0.1818 |
| dct256x256 | LinearLogistic | 0.7312 | 0.8264 | +0.0952 |

> 图6 显示所有组合中 QKS 性能均高于基线，尤其在线性读出器上增益最大。

### 🔍 消融实验结果

#### （1）Stage 1: 浅层架构搜索
- 最优配置：**10 qubits, 256 episodes, entanglement on, D=1**
- 结论：适度规模即可捕获判别结构；过大配置可能引入冗余。

#### （2）Stage 2: 深度扫描（Depth Sweep）
- 最佳深度出现在 **D=4**（而非最大 D=10）
- 示例：10 qubit + 256 episode 配置下：
  - D=1 → AUROC=0.828
  - D=4 → AUROC=0.850 (+2.2%)
- 超过 D=4 后性能下降 → 存在“适度深度”最优区间。

#### （3）Stage 3: 深度 vs Episode 权衡
- 固定预算 $ D \times E = 256 $
- 最优组合：**(D=4, E=64)** > (D=2, E=128) > (D=1, E=256) ≈ (D=8, E=32)
- 结论：需平衡 **每 episode 表达力** 与 **跨 episode 多样性**。

#### （4）Stage 4: 输入表示比较
- **DCT 家族显著胜出**：
  - 前三名均为 DCT 变体（dct128x128, dct64x64, dct256x256）
  - 最佳 PCA（pca32）仅达 0.5262 AUROC，远低于 DCT
- 原因分析：
  - DCT 保留时间-频率平面的局部结构，利于捕捉 chirp/jamming 模式。
  - PCA 基于全局方差，易混合无关区域，破坏判别特征。

#### （5）Stage 5: 全数据最终测试
- 所有 QKS 配置均优于 direct-readout
- **线性读出器增益最大** → QKS 特征已具备良好线性可分性
- 非线性读出（RFF/Nystrom）也有增益，但边际效应递减

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **QKS 不是通用增强器，而是依赖表示的选择**：
   - 只有与 **DCT 类输入表示结合** 时才能发挥最大效用。
   - “Representation-enhancer” 而非“Universal classifier”。

2. **中等深度 + 缠绕结构效果最佳**：
   - **moderate-depth entangled QKS**（如 D=4, ring entanglement）构成最强操作模式。
   - 单纯增加深度或 episode 数不会持续提升性能。

3. **DCT 表示具有决定性作用**：
   - 低频 DCT 系数有效浓缩异常的时间-频谱结构信息。
   - raw 和 PCA 输入无法提供同等质量的初始表示。

4. **QKS 提升主要体现在线性可分性上**：
   - 最大增益出现在 Linear SVM / Logistic 上。
   - 表明 QKS 成功将数据映射到更利于线性分类的空间。

5. **真实硬件验证成功**：
   - 在 **ibm_quebec QPU** 上运行，尽管资源受限（4 qubits, 32 episodes, reduced observable set）
   - 与模拟结果相比，**AUROC 差异 < 0.013**
   - 排名趋势一致（NystroemLogistic 最优），说明特征几何结构在噪声下仍稳定。

### ⚠️ 局限性
- **任务特定性**：数据为受控合成异常，未测试真实世界未知干扰类型。
- **随机初始化稳定性未量化**：仅使用单一运行，未报告多 seed 实验结果。
- **读出模型有限**：未考虑更强的经典模型（如 deep neural networks），可能导致低估经典基线上限。
- **无量子优势声明**：作者明确指出不主张“quantum advantage”，仅为探索可行 hybrid pipeline。

### 🔮 未来工作方向
1. 扩展至更多 RF 异常家族（如 replay attack, spoofing 等）。
2. 在更广泛的硬件约束下测试（不同 QPU、噪声模型、编译优化）。
3. 探索自适应 DCT/QKS 联合学习机制。
4. 将该框架集成进 real-time spectrum monitoring 系统原型。

---

> **一句话总结**：  
> 本论文展示了 **DCT-preprocessed RF spectrograms + moderate-depth entangled QKS** 构成一个高效、可复现的 anomaly detection pipeline，在真实数据与真实量子硬件上均表现出稳健性能，为 NISQ 时代 QML 在无线安全中的应用提供了重要实践路径。

</details>

---

### 16. [SAFETY SENTRY: Context-Aware Human Intervention via EXECUTE-ASK-REFUSE Routing](https://arxiv.org/abs/2607.13594)

**Authors**: Tianyu Chen, Chujia Hu, Wenjie Wang  
**Category**: cs.AI  
**Published**: 2026-07-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.13594v1  

#### Abstract
LLM agents act on real-world environments through tool calls, and a single misjudged action can cause irreversible harm. The standard safeguard is a guard model that labels each proposed action as safe or unsafe, but this binary view conflates two distinct decisions: whether the action is harmful in...

---

### 17. [A Distributed Framework for Compiling and Reasoning with d-DNNF](https://arxiv.org/abs/2607.13642)

**Authors**: Zhenghang Xu, Minghao Yin, jianan Wang, Jean-Marie Lagniez  
**Category**: cs.DC  
**Published**: 2026-07-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.13642v1  

#### Abstract
Knowledge Compilation (KC) is a powerful paradigm that enables efficient reasoning by transforming propositional formulas into tractable target languages, such as Deterministic, Decomposable Negation Normal Form (d-DNNF). However, as real-world problem instances grow in complexity, the offline compi...

---

### 18. [Self-Improving is Often Sudden: Enlightenment-style Finetuning for Large-Scale Models](https://arxiv.org/abs/2607.13395)

**Authors**: Jing-Xiao Liao, Tianwei Zhang, Yu-Hao Jiang, Feifei Zhang, Hang-Cheng Dong, Feng-Lei Fan  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.13395v1  

#### Abstract
The pursuit of autonomously self-improving models has attracted growing interest in the era of large-scale foundation models. Drawing inspiration from the concept of "enlightenment" or "aha moment" in human brain, we hypothesize that large models exhibit an analogous enlightenment phenomenon-a laten...

---

### 19. [EXPLORE: Exploration with Guided Search for Analog Topology Generation using Language Models](https://arxiv.org/abs/2607.13416)

**Authors**: Guanglei Zhou, Chen-Chia Chang, Yikang Shen, Jonathan Ku, Isaac Jacobson, Jingyu Pan, Yiran Chen, Xin Zhang  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.13416v1  

#### Abstract
Automating analog circuit topology design is essential to reduce the extensive manual effort required to meet increasingly diverse and customized application demands. Recent advances have applied sequence-to-sequence fine-tuning on pretrained language models to directly generate circuit topologies f...

---

### 20. [A VAE-Driven Multi-Task Satellite-Aided Semantic Communication Framework for 6G-Enabled Connected Autonomous Vehicles](https://arxiv.org/abs/2607.13494)

**Authors**: S. M. Abtahiul Alam, Niloy Das, Apurba Adhikary, Yu Qiao, Zhu Han, Choong Seon Hong  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.13494v1  

#### Abstract
The development of smart transportation systems and the introduction of 6G wireless communication technologies have significantly changed vehicle network topologies. Future connected autonomous vehicle (CAV) networks require bandwidth-efficient, reliable, and low-latency communication for safety-cri...

---

### 21. [The Capacity of Thought: Benchmarking Llama 3.2 in Semantic fMRI Neural Language Decoding and Improving the Huth Encoding-Model Baseline](https://arxiv.org/abs/2607.12079)

**Authors**: Milos Suvakovic, Dom Marhoefer, Glenn Grant-Richards, Aidan Pinero  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.12079v1  

#### Abstract
Decoding continuous language from fMRI signals remains a core challenge in non-invasive brain-computer interface research. We present two complementary investigations. First, we improve the Huth et al. ridge regression encoding pipeline through expanded voxel selection (10K->15K), substitution of GP...

---

### 22. [Translation as a Computationally Efficient Bridge: Feasibility of English BERT for Low-Resource Languages](https://arxiv.org/abs/2607.12612)

**Authors**: Hielke Muizelaar, Giulia Rivetti, Marco Spruit, Marcel Haas  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.12612v1  

#### Abstract
BERT models have revolutionised Natural Language Processing (NLP) through their ability to process unstructured text across diverse domains. However, developing high-quality BERT models for non-English languages remains challenging due to limited annotated data and high computational demands. Transl...

---

### 23. [Automatic Differentiation from Scratch: How PyTorch Computes Gradients in Physics-Informed Neural Networks](https://arxiv.org/abs/2607.13042)

**Authors**: Abdeladhim Tahimi  
**Category**: cs.LG  
**Published**: 2026-07-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.13042v1  

#### Abstract
This paper traces, with explicit numerical values, how PyTorch's automatic differentiation (AD) engine computes gradients for Physics-Informed Neural Network (PINN) training -- a setting that requires two levels of differentiation: computing the physics derivative $\hat{y}'(t)=d\hat{y}/dt$ through t...

---

### 24. [Experience Memory Graph: One-Shot Error Correction for Agents](https://arxiv.org/abs/2607.13884)

**Authors**: Wenjun Wang, Yuchen Fang, Fengrui Liu, Zibo Liang, Kai Zheng  
**Category**: cs.AI  
**Published**: 2026-07-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.13884v1  

#### Abstract
Large Language Model (LLM) agents have shown remarkable capabilities in autonomous decision-making by generating sequential trajectories of states, actions, and observations. However, in complex, long-horizon tasks, these agents frequently suffer from compounding errors and struggle to recover from ...

---

### 25. [Scaling Point-in-Time Language Models](https://arxiv.org/abs/2607.11889)

**Authors**: Bryan Kelly, Semyon Malamud, Johannes Schwab, Teng Andrea Xu  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.11889v1  

#### Abstract
Large language models trained on unrestricted internet corpora inevitably embed information from the future, introducing lookahead bias that compromises the validity of backtests and causal inference in finance and the social sciences. Point-in-time language models--trained exclusively on text avail...

---

### 26. [Graph-Based Detection of Disinformation Narrative Diffusion between Russian and Ukrainian Telegram Channels](https://arxiv.org/abs/2607.11894)

**Authors**: Yuliia Vistak, Viktoriia Makovska, Vera Schmitt, Veronika Solopova  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.11894v1  

#### Abstract
Detecting disinformation narratives on social media is challenging due to the scale of amplification, rapid evolution, and linguistic variability of online content. We propose a graph-based framework for identifying and analyzing disinformation narratives in Telegram ecosystems by combining weak sup...

---

### 27. [TAKE: Trajectory-Aware Knowledge Estimation for Text Dataset Distillation](https://arxiv.org/abs/2607.11898)

**Authors**: Tri-Nhan Vo, Dang Nguyen, Sunil Gupta  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.11898v1  

#### Abstract
Large-scale text corpora have become a quiet bottleneck in modern NLP, not just in storage, but in the accumulated cost of training, fine-tuning, and continual learning. We propose a text dataset distillation framework that reduces corpora to as little as 0.1% of their original size while preserving...

---

### 28. [CityBehavEx: A Scalable and Empirically Validated LLM-Assisted Urban Simulation Platform](https://arxiv.org/abs/2607.12086)

**Authors**: Gustavo H. Santos, Aline Viana, Thiago H Silva  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.12086v1  

#### Abstract
Recent LLM-based multi-agent urban simulators can generate semantically rich city routines, but they remain costly to scale and are often weakly validated against empirical mobility patterns. We present CityBehavEx, an interactive LLM-assisted urban simulation platform that scales to city-size popul...

---

### 29. [FinResearchBench II: A Deep Research Benchmark with Consensus-Derived Gold Rubrics for Distinguishing Financial Report Quality](https://arxiv.org/abs/2607.12252)

**Authors**: Beidi Luan, Rui Sun, Sinuo Wang, Yan Gu, Chao Li, Zhenliang Xiong, Jing Li, Zuo Bai  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.12252v2  

#### Abstract
Deep research agents are increasingly used to produce long-form financial reports, yet large-scale evaluation remains bottlenecked by the need for human experts to define and execute high-quality rubrics. We address this problem by proposing a scalable pipeline for generating high-quality rubrics wi...

---

### 30. [Beyond Binary Detection: A Multi-Dimensional Taxonomy of Cancer Misinformation on Reddit](https://arxiv.org/abs/2607.12383)

**Authors**: Aria Pessianzadeh, Pooriya Jamie, Naima Sultana, Georgia Himmelstein, Yuliya Zektser, Patricia Ganz, Homa Hosseinmardi, Amir Ghasemian, Rezvaneh Rezapour  
**Category**: cs.CL  
**Published**: 2026-07-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.12383v1  

#### Abstract
Cancer-related discussions on social media provide an important space for information exchange and peer support, but also facilitate the spread of misinformation that may influence prevention, screening, and treatment decisions. Existing research on cancer misinformation often relies on narrow defin...

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

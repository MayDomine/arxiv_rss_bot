# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-11 09:58:20 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Beyond Fully Random Masking: Attention-Guided Denoising and Optimization for Diffusion Language Models](https://arxiv.org/abs/2606.12273)

**Authors**: Jia Deng, Junyi Li, Wayne Xin Zhao, Jinpeng Wang, Hongyu Lu, Ji-Rong Wen  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.12273v1  

#### Abstract
Diffusion large language models (dLLMs) offer an efficient alternative to autoregressive models through parallel decoding, yet existing post-training methods largely rely on random masking strategies that overlook intrinsic token dependencies. In this work, we present an empirical analysis of attent...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond Fully Random Masking: Attention-Guided Denoising and Optimization for Diffusion Language Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现有的 **Diffusion Large Language Models (dLLMs)** 在 post-training 阶段普遍采用 **fully random masking** 或固定的半自回归掩码策略（如 blockwise SFT），这些方法忽略了文本生成过程中 token 之间的内在依赖关系。这种训练与推理过程的不一致（training-inference mismatch）限制了模型在复杂推理任务（如数学和代码生成）上的表现。

此外，尽管 dLLMs 具备 bidirectional attention 能力，能够动态建立 token 间的依赖，但现有方法并未有效利用这一特性。

---

### 🚀 提出了什么新方法或新思路

本文提出 **AGDO (Attention-Guided Denoising and Optimization)**，一个统一的 post-training 框架，其核心思想是：  
> **将 dLLMs 的去噪顺序和优化目标显式地对齐到注意力机制所揭示的 token 依赖结构上**。

具体包括两个关键组件：

1. **Attention-Guided Denoising Order (AGDO-SFT)**  
   - 基于最终层的 attention 分布计算每个 token 对已解码上下文的关注程度（valid attention score $ S $）。
   - 按照 $ S $ 排序决定去噪顺序：优先恢复那些更依赖已有上下文且上下文已充分暴露的 token。
   - 这使得训练时的去噪路径更贴近模型实际推理时的语义依赖流。

2. **Attention-Guided Policy Optimization (AGDO-RL)**  
   - 在强化学习阶段（如 GRPO），引入 attention hub tokens 的影响权重。
   - 定义 token 影响力 $ I_k $ 为其他 token 对它的总注意力值，并用于调整优势函数（advantage）：
     $$
     A'_k = A_k + \text{sign}(A_k)\cdot\delta\cdot I_k
     $$
   - 强化学习更新更关注“注意力中心”token，提升全局一致性。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | AGDO |
|------|--------|------|
| **Masking 策略** | 随机或固定顺序 | 动态、基于 attention 依赖 |
| **训练-推理一致性** | 存在 mismatch | 显著增强 alignment |
| **推理稳定性** | 忽视概率漂移 | 利用 valid attention 提升稳定 |
| **优化重点** | 平等对待所有 token | 强调 attention-hub tokens |
| **适用性** | 通用但低效 | 更适合复杂推理任务 |

> ✅ **核心优势**：首次系统性分析并利用 dLLMs 中 attention 的结构性与稳定性，实现“以模型内部认知驱动训练”的范式转变。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### 数学推理任务：
- **MATH-500**：高难度数学题基准，覆盖多个领域。
- **GSM8K**：小学级数学应用题，强调多步推理。
- **Minerva**：大规模科学与数学问题集合。

#### 编程任务：
- **LiveBench**：真实世界编程挑战，避免污染。
- **LiveCodeBench-V2**：执行验证的代码生成评测。

#### 泛化能力测试：
- **HellaSwag**：常识填空任务。
- **CommonsenseQA**：常识问答。

---

### ⚙️ 实验设置和评估指标

| 设置项 | 描述 |
|-------|------|
| **主模型** | Dream-v0-Instruct-7B, LLaDA-8B-Instruct |
| **训练方式** | SFT + RL（两阶段） |
| **评估指标** | 准确率（Accuracy） |
| **推理配置** | 静态解码，每次去噪 1 个 token，最大长度 1024，温度 0.1 |
| **重复次数** | 每个实验重复 8 次取平均 |
| **公平性控制** | 控制前向传播次数相近（通过多 mask 增强数据） |

---

### 🆚 基线方法对比

#### SFT 阶段基线：
- **Standard SFT**：完全随机掩码。
- **Blockwise SFT**：分块自回归式去噪。

#### RL 阶段基线：
- **diff-GRPO**：基于 group normalization 的 policy optimization。
- **Coupled RL**：联合训练框架。
- **TraceRL**：追踪生成轨迹进行优化。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 2）

在 **Dream-v0-Instruct-7B** 上的整体平均准确率：

| 方法 | Average Score |
|------|----------------|
| Dream-v0-Instruct-7B (原始) | 28.3% |
| SFT (baseline) | 33.9% |
| Blockwise SFT | 34.4% |
| **AGDO-SFT (Ours)** | **36.0%** |
| diff-GRPO | 35.0% |
| TraceRL | 36.5% |
| **AGDO-RL (Ours)** | **38.1%** |
| **AGDO (SFT+RL)** | **38.8%** |

> 💡 **AGDO 整体提升达 +10.5% 绝对增益**，显著优于所有基线。

---

### 🔬 详细任务表现亮点

| 任务 | AGDO 表现 | 超越最强基线 |
|------|----------|-------------|
| **LiveBench (code)** | 18.3% | > diff-GRPO (15.2%) |
| **Minerva (math)** | 17.0% | > blockwise SFT (+2.7%) |
| **MATH500 (math)** | 56.2% | > TraceRL (52.8%) |

---

### 🔍 消融实验结果（Ablation Study）

#### ✅ **Ablation on γ 和 δ（加权系数）**
- 当 $ \gamma = 100 $（SFT 加权强度）时效果最佳。
- 即使 $ \gamma = 0 $（无加权），仅靠 attention-guided order 已超越 blockwise SFT。
- $ \delta < 10 $ 有助于 RL 性能，过大（如 20）导致梯度震荡，违反 PPO 信任域假设。

#### ✅ **不同 block size 下的表现（L=512）**
| 方法 | MATH500 平均准确率 |
|------|--------------------|
| Standard SFT | 49.1% |
| Blockwise SFT | 45.8% ↓ |
| **AGDO-SFT** | **49.6%** ↑ |
| **AGDO-RL** | **51.6%** （最高 53.7% @ bs=64）|

> ❗说明 AGDO 在上下文受限下仍保持鲁棒性，而 blockwise SFT 明显退化。

#### ✅ 层与头的选择（Layer & Head Ablation）
| 设置 | MATH500 Acc |
|------|-------------|
| First Layer | 48.8% |
| Middle Layer | 52.5% |
| **Last Layer (Ours)** | **53.7%** ✅ |
| Local-focused heads | 53.0% |
| Global-focused heads | 52.2% |
| **All heads (Ours)** | **53.7%** ✅ |

> 表明高层语义注意力更重要，且局部与全局模式都应保留。

#### ✅ Order vs. Weighting 消融（Table 7）
| 方法 | MATH500 Acc |
|------|--------------|
| Blockwise SFT (baseline) | 51.7% |
| Order Only ($ \gamma=0 $) | 52.7% (+1.0%) |
| Weight Only (random order) | 52.4% |
| Random Weight | 51.9% |
| **Order + Weight (Ours)** | **53.7%** |

> 结论：**两者协同作用最大，gain 来源于 attention 信号本身而非随机正则化**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Attention in dLLMs 是稀疏且时间稳定的**  
   - 注意力图呈现明显对角线（自注意+邻近）和亮列（hub tokens）。
   - 同一 token 在不同去噪步中关注对象高度一致 → 可预测依赖结构。

2. **Valid Attention 与生成稳定性强相关**  
   - token 若更多关注已去噪上下文（$ S $ 高），其生成概率变化小（$ \Delta P $ 小）→ 更稳定。
   - 利用 $ S $ 构建去噪顺序可显著提升准确率（见 Table 1）。

3. **AGDO 显著提升推理能力而不损害泛化**
   - 在非推理任务（HellaSwag, CommonsenseQA）也全面领先（Table 9）：
     - AGDO-SFT 在 CSQA 达 78.1%，超 baseline +5.2%
     - AGDO 整体达 78.3%

4. **训练过程更稳定高效**
   - 图 3 显示 AGDO-RL 在训练/测试集上收敛更快、波动更小。
   - 计算开销极低：attention analysis 仅占 rollout 时间 **3%**（Table 8）。

---

### ⚠️ 方法的局限性

- **仅适用于 full-attention dLLMs**  
  不适用于 block attention 或稀疏 attention 结构的变体（作者明确指出此限制）。
- **依赖最后一层 attention**  
  虽然实验证明 last layer 最优，但可能忽略中间层的动态演化信息。
- **online attention analysis 开销虽小但仍存在**  
  在极端低延迟场景下可能成为瓶颈。

---

### 🔮 未来工作方向

1. **扩展至 block-based dLLMs**  
   设计适配局部注意力结构的 guided denoising 策略。
2. **动态 layer selection**  
   自适应选择最具代表性的 attention layer 进行引导。
3. **结合 remasking 与 AGDO**  
   在 inference-time 进行 attention-aware remasking 优化。
4. **探索 AGDO 在 vision-language 模型中的应用**  
   将 attention-guided 思想推广到多模态扩散模型。

---

## ✅ 总结

| 维度 | 内容 |
|------|------|
| **核心思想** | 利用 dLLMs 的 attention 结构指导去噪顺序与优化重点 |
| **方法名称** | AGDO（Attention-Guided Denoising and Optimization） |
| **关键技术** | Valid attention score、influence score、attention-aligned loss |
| **实验成果** | 在 math/code 多项任务上 SOTA，平均提升超 10% |
| **理论意义** | 揭示了 attention 可作为 dLLMs 内部推理逻辑的代理信号 |
| **实践价值** | 提供了一种轻量、通用、高效的 post-training 新范式 |

> 🌟 **一句话总结**：  
> **AGDO 通过“让模型按照自己关注的方式去思考”，实现了 dLLMs 推理能力的本质跃迁。**

</details>

---

### 2. [Re-evaluating Confidence Remasking in Masked Diffusion Language Models](https://arxiv.org/abs/2606.12232)

**Authors**: Stipe Frkovic, Metod Jazbec, Dan Zhang, Christian A. Naesseth, Ilija Bogunovic, Eric Nalisnick  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.12232v1  

#### Abstract
Masked diffusion language models (dLLMs) have recently emerged as a competitive alternative to autoregressive language models, with the promise of faster inference via parallel token generation. A notable limitation of the masked formulation, however, is that once a token has been unmasked it can no...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Re-evaluating Confidence Remasking in Masked Diffusion Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文重新审视了**post-hoc confidence-based remasking**（后处理基于置信度的重掩码）在 **masked diffusion language models (dLLMs)** 中的实际有效性。尽管近期如 **WINO** 等方法声称通过训练无关的方式实现自纠正（self-correction），从而提升生成质量，但其宣称的性能增益缺乏严谨、全面的评估。

本文指出，现有研究存在以下不足：
- 与弱基线（如随机或高置信度采样）比较，而非强基线（如 Fast-dLLM）；
- 在非标准设置下（如大 block length）进行评估；
- 忽略了 remasking 所带来的额外计算开销（latency 和 FLOPs）；
- 缺乏对非贪婪解码（non-greedy decoding）等多样化生成场景的研究。

### 提出了什么新方法或新思路
本论文**并未提出新的 remasking 方法**，而是提出了一个更全面、更严格的**评估框架**，用于系统性地分析当前 post-hoc remasking 方法（以 WINO 为代表）的真实价值。

其核心思路是：
- 将 remasking 的收益与强大的 confidence-based unmasking（如 Fast-dLLM）进行公平比较；
- 考察不同解码设置（block length、sampling temperature、unmasking 策略）下的表现；
- 引入 flip-flop 频率等诊断指标，深入分析 remasking 失败的原因。

### 相比现有方法的优势
- **批判性视角**：挑战了“remasking 总是有益”的直觉，揭示其收益高度依赖于具体设置；
- **全面评估**：覆盖了标准与非标准 block length、贪婪与非贪婪解码、确定性与随机 unmasking 策略；
- **实用导向**：不仅关注 accuracy，还考虑了实际推理效率（throughput、latency）；
- **机制解释**：通过 flip-flop 分析和消融实验，揭示了模型无法生成更好替代 token 是根本瓶颈。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在四个标准 benchmark 上进行：
- **GSM8k**：小学数学应用题
- **MATH-500**：高等数学问题
- **HumanEval**：代码生成任务
- **MBPP**：面向编程的自然语言到代码任务

### 实验设置和评估指标
#### 模型
- **LLaDA-8B-Instruct** [Nie et al., 2025]
- **Dream-v0-Instruct-7B** [Ye et al., 2025]

#### 解码策略与变量
- **Block Length (BL)**：32（标准设置） vs. 128（WINO 原文设置）
- **Unmasking Threshold (λ₁)**：{0.5, 0.6, 0.7, 0.8, 0.9}
- **Remasking Threshold (λ₂)**：固定为 0.8（WINO 设置）
- **Sampling Temperature (T)**：0（greedy） vs. 0.8 / 1.5（non-greedy）
- **Unmasking Strategy**：Fast-dLLM（确定性） vs. dUltra（基于 Bernoulli policy 的随机）

#### 评估指标
- **Accuracy / Pass@k**：Pass@1、Pass@64 等，衡量至少一个生成正确解的概率
- **Network Function Evaluations (NFEs)**：前向传播次数，衡量计算成本
- **Wall-clock time (throughput)**：实际推理延迟
- **Flip-flop frequency**：remask 后又预测回原 token 的比例，反映纠错能力

### 基线方法对比
- **Fast-dLLM** [Wu et al., 2025]：confidence-thresholding unmasking，作为主要强基线
- **WINO** [Hong et al., 2026]：Fast-dLLM + shadow-token-based remasking
- **dUltra + WINO**：学习型 Bernoulli unmasking + remasking
- **Saber** [Dong et al., 2025]：其他 post-hoc remasking 方法，用于泛化性验证

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 在标准设置下（BL=32, T=0）：
- **WINO vs Fast-dLLM**：accuracy 提升极小且不一致
  - 平均仅提升约 **0.4% (LLaDA)** 和 **0.5% (Dream)**
  - 在 **HumanEval** 上略有提升（~1.2–1.8%），但在 **MATH-500** 上甚至下降
- **效率劣势明显**：
  - 由于 shadow block 增加序列长度，**throughput 显著低于 Fast-dLLM**
  - 在相同 wall-clock time 下，Fast-dLLM 表现更优（见 Figure 7）

#### ⚠️ 在大 block length 下（BL=128）：
- WINO 相对 Fast-dLLM 提升更大（平均 **+1.3%~1.5%**）
- 但这主要是因为 **Fast-dLLM 在大 block 下性能严重退化**，而 remasking 起到了“修复”作用
- **绝对性能仍低于 Fast-dLLM @ BL=32**，说明 remasking 并未真正提升模型上限

#### 🔁 Flip-flop 分析（关键诊断）：
- **高达 75–95% 的 remasked positions 最终恢复为原 token**
  - LLaDA: ~75–90%
  - Dream: ~85–95%
- 表明：**模型能识别出“可疑”位置，但无法生成更好的替代 token**

#### 🌡️ 在非贪婪解码下（T > 0）：
- **Pass@1 提升明显**：T=0.8 时平均 +2.6%
- **但多样性受损**：
  - Pass@64 仅提升 +0.7%，表明 remasking 抑制了探索
  - 支持“diversity collapse”现象被进一步加剧

#### 🎲 在随机 unmasking 下（dUltra + Bernoulli policy）：
- **WINO 带来显著增益**：平均 accuracy 提升 **+3.2%**
- NFE 仅增加 ~2
- 表明：remasking 更适合纠正**由随机性引入的错误**

### 消融实验结果
- **Shadow token approximation 质量高**：
  - 与 oracle leave-one-out confidence 性能相当，但节省 ~14× NFE
- **不同 remasking criteria 影响不大**：
  - 使用 consistency-based 或 threshold-based 效果相近
- **Loop-guard 变体无改善**
- **扩展 remasking 范围（邻居 token）无效甚至有害**：
  - 无论是空间邻域（S=1,2）还是时间邻域（T=1,2），均不能降低 flip-flop 率或提升 accuracy
  - 说明问题不在上下文依赖，而在模型自身预测分布

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Post-hoc remasking 的收益被高估**：
   - 在标准设置（BL=32, greedy decoding）下，**WINO 相比 Fast-dLLM 几乎没有实质性提升**，且代价更高。
   
2. **Remasking 的有效性高度依赖于解码设置**：
   - 在大 block length 下的增益主要源于“修复 Fast-dLLM 的缺陷”，而非提升模型本质能力。
   - 在非贪婪或随机 unmasking 场景下，remasking 更有价值，因其可纠正随机性带来的错误。

3. **根本瓶颈是模型无法生成更好 token**：
   - 高 flip-flop 率表明：**模型能发现问题，但提不出更好答案**。
   - 即使移除潜在的上下文依赖（如 remask 邻居），模型仍会重复原预测。

4. **Remasking 可能损害生成多样性**：
   - 在非贪婪解码中，remasking 提升 Pass@1 但抑制 Pass@k（k 大时），加剧了 confidence-based 方法固有的“diversity collapse”。

### 方法的局限性
- **评估对象有限**：主要聚焦于 WINO，虽补充 Saber 结果，但仍需更多 post-hoc 方法验证。
- **未涉及训练时 remasking**：如 fine-tuning 或 RL-based 方法可能更有效，但不在本研究范围内。
- **flip-flop 是现象而非解决方案**：指出了问题，但未提供如何改进模型预测分布的方法。

### 未来工作方向
- **建立标准化评估框架**：
  - 应统一在标准 block length（如 BL=32）、合理计算成本下比较 remasking 方法。
- **探索更有效的 remasking 机制**：
  - 当前基于 confidence 的方法受限于模型自身表达能力，可能需要引入外部知识或更强的生成机制。
- **结合训练时干预**：
  - 纯 post-hoc 方法可能已达瓶颈，未来应探索与 fine-tuning、RL 或 uniform diffusion 的结合。
- **研究 remasking 对多样性和创造力的影响**：
  - 特别是在非贪婪、多样本生成场景中，避免过度收敛到单一解。

--- 

> **一句话总结**：  
> 本文揭示了当前 post-hoc confidence-based remasking（如 WINO）在标准设置下对 masked dLLMs 的收益极为有限，其微弱增益常被计算成本所抵消；真正的瓶颈在于模型无法在 remasking 后生成更优 token，未来需更全面的评估体系与更深层次的建模改进。

</details>

---

### 3. [Energy-Efficient On-Device RAG on a Mobile NPU: System Design and Benchmark on Snapdragon X Elite](https://arxiv.org/abs/2606.11257)

**Authors**: Zhiyuan Cheng, Longying Lai  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11257v1  

#### Abstract
Retrieval-Augmented Generation (RAG) pipelines are compute-intensive, combining embedding, retrieval, reranking, and large language model (LLM) generation. Running them entirely on-device benefits privacy, latency, and offline use, but the energy cost of CPU inference is a major barrier. We present ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Energy-Efficient On-Device RAG on a Mobile NPU: System Design and Benchmark on Snapdragon X Elite*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Retrieval-Augmented Generation (RAG)** 系统计算密集，通常依赖云端部署以满足性能需求。然而，这带来了**隐私泄露、高延迟和无法离线使用**等问题。虽然在设备端运行 RAG 可解决上述问题，但传统基于 CPU 的推理能耗过高，难以在移动或笔记本平台持续运行。

此外，尽管现代 SoC 集成了专用的 **Neural Processing Unit (NPU)**，但此前尚无研究实现并评测一个完整的 RAG 流程（包括 embedding、reranking 和 LLM 生成）全部运行于移动 NPU 上。

### 提出了什么新方法或新思路
本文提出了首个端到端的、全神经阶段均在 **Qualcomm Hexagon NPU** 上执行的 on-device RAG 系统，部署于搭载 **Snapdragon X Elite** 平台的设备上。其核心创新包括：

- **完整 NPU 加速 RAG 架构设计**：将 embedding（EmbeddingGemma）、reranking（Jina Reranker v2）和 LLM 生成（Qwen3-4B-Instruct）三个神经组件全部迁移至 Hexagon NPU，通过 **Qualcomm AI Runtime (QAIRT/QNN)** SDK 进行静态图编译与调度。
- **统一接口抽象**：为 NPU、CPU 和 GPU 后端提供一致的推理接口，支持灵活切换后端进行公平比较。
- **工程挑战应对方案**：系统性地解决了 NPU 上多模型加载顺序限制、固定 context length 限制以及 Windows ARM64 生态不成熟等实际部署难题。

### 相比现有方法的优势
- **能效显著提升**：相比纯 CPU 推理，系统总能量消耗降低 **4–12.3×**。
- **延迟大幅下降**：端到端查询延迟减少 **4.0×**，尤其在 LLM prefilling 阶段提速达 **18.1×**。
- **质量无损**：使用 GPT-4.1 作为 LLM-as-judge 评估，答案质量与 CPU/GPU 基线相当，在评估噪声范围内保持一致（86.7% 查询得分相同）。
- **唯一可行的片上加速器**：集成 GPU（Adreno X1-85）在此任务中表现差于 CPU，且功耗更高，表明 NPU 是当前最合适的 on-chip 加速选择。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Indexing Corpus**：10 家公司的 SEC 10-K 财报文件（如 AAPL, ABBV），共提取出 **9,324 个文本块**（约 204 万 tokens）。
- **Query Workloads**：
  - **wiki_minirag**：从 `rag-mini-wikipedia` 数据集中选取 120 个通用知识问答对，用于主性能与能效评测。
  - **FinDER**：来自金融领域 QA 基准的 120 个问题，测试小规模专业语料下的“正确拒绝”能力。

### 实验设置
- **硬件平台**：Dell XPS 13 9345 笔记本，配备：
  - **CPU**：Snapdragon X Elite X1E80100（12 核）
  - **GPU**：Qualcomm Adreno X1-85
  - **NPU**：Hexagon NPU（INT8 性能达 45 TOPS）
  - **内存**：64GB LPDDR5x
  - **操作系统**：Windows 11 ARM64
- **模型配置**：
  | 组件 | 模型 | 参数量 |
  |------|------|--------|
  | Embedding | EmbeddingGemma 300M | 300M |
  | Reranker | Jina Reranker v2 Base Multilingual | 278M |
  | LLM | Qwen3-4B-Instruct | 4B |

### 评估指标
- **性能指标**：
  - 各阶段 wall-clock 时间（如 parsing、embedding、prefill、decode）
  - **Tokens/s**（embedding throughput、prefilling speed、decoding speed）
  - **端到端延迟**（Total query latency）与尾部延迟（P95）
- **能效指标**：
  - 系统平均功率（Average system power, W）
  - 总系统能耗（Total system energy, J/kJ），通过 HWiNFO64 共享内存采集传感器数据（500ms 采样粒度）
- **质量评估**：
  - 使用 **GPT-4.1** 作为 LLM-as-judge，采用 1–10 分制评分标准（≥7 表示基本正确，10 表示完美匹配）
  - 报告平均分、失败率（s=1）、正确率（s≥7）、完美率（s=10）
  - 执行配对 Wilcoxon 检验分析差异显著性

### 基线方法对比
所有实验在同一硬件平台上运行，仅改变 LLM 与 embedder 的执行后端（reranker 固定在 NPU 上）：
- **NPU**：所有神经模块运行于 Hexagon NPU（使用 QAIRT/QNN 编译的静态图模型）
- **CPU**：LLM 与 embedder 使用 llama.cpp + GGUF 量化模型（Q4_K_M / BF16），n_gpu_layers=0
- **GPU**：LLM 与 embedder 使用 llama.cpp + OpenCL 卸载至 Adreno GPU，n_gpu_layers=999

此设计确保了除计算后端外其他变量完全一致，保证了可比性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）Indexing 阶段性能（表 III）
| 指标 | NPU | CPU | 提升倍数 |
|------|-----|-----|---------|
| Embedding 吞吐量 | 3,325 tokens/s | 367 tokens/s | **9.1×** |
| 总 pipeline 时间 | 710.7 s | 5,648.2 s | **7.9×** |
| 平均系统功耗 | 27.6 W | 42.7 W | ↓35% |
| **总系统能耗** | **19,592 J** | **241,020 J** | **↓12.3×** |

> ✅ **说明**：embedding 是瓶颈，NPU 显著缓解该问题；同时因 CPU 利用率降至 12.6%，系统整体功耗更低，带来超线性的节能效果。

#### （2）Query Processing 性能（wiki_minirag，表 IV）
| 指标 | NPU | CPU | GPU | NPU vs CPU | GPU vs CPU |
|------|-----|-----|-----|------------|-----------|
| LLM TTFT（首 token 延迟） | 1.30 s | 24.77 s | 42.24 s | **↓19.1×** | ↑1.7× |
| LLM Prefilling Speed | 786.7 tok/s | 43.4 tok/s | 25.2 tok/s | **↑18.1×** | ↓0.58× |
| Decoding Speed | 14.19 tok/s | 8.17 tok/s | 4.65 tok/s | ↑1.74× | ↓0.57× |
| **总查询延迟** | **9.48 s** | **37.98 s** | **63.61 s** | **↓4.0×** | ↑1.7× |
| **尾延迟（P95）** | **17.9 s** | **69.6 s** | **107.4 s** | **↓3.9×** | ↑1.5× |
| **总系统能耗（120 queries）** | **37.83 kJ** | **150.12 kJ** | **246.14 kJ** | **↓4.0×** | ↑6.5× |

> ✅ **关键发现**：
> - NPU 在 prefilling 阶段优势巨大（密集矩阵运算友好）
> - decoding 阶段也有明显提升（↑1.74×），但仍受限于内存带宽
> - GPU 表现最差：速度慢 1.7×，能耗高 6.5×，验证其不适合作为此类负载的加速目标

#### （3）Per-Component 能耗分解（表 V）
- NPU 模式下，**CPU cluster 功耗仅为 4.1–4.6W**（空闲水平），而 CPU 基线中高达 5.3–6.5W。
- GPU 模式下，Adreno 自身消耗 **16.4 kJ**（占总量 6.7%），远高于 NPU 模式的 GPU 能耗（仅 0.1 kJ），说明 OpenCL 确实激活了 GPU，但效率极低。

#### （4）消融实验与配置影响
- **Chunk size 限制**：由于 NPU 静态图限制，chunk size 从 2,500 字符降为 1,000 字符，context 中保留 chunk 数从 10 减至 7。
- **预期担忧**：更少上下文可能导致质量下降。
- **实证结果**：GPT-4.1 评判显示 NPU 质量反而略优（均分 9.32 vs 8.95），且 86.7% 查询得分完全一致 → **证明质量未受损，甚至可能受益于更聚焦的输入**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **NPU 是实现高效 on-device RAG 的理想平台**：
   - 在 Snapdragon X Elite 上，NPU 可实现 **端到端 RAG 全流程神经推理加速**，无需依赖云服务。
   - 相比 CPU，获得 **4.0× 延迟降低** 和 **4.0–12.3× 能源节省**，且不影响输出质量。
2. ❌ **集成 GPU 不是有效替代方案**：
   - 当前 Adreno X1-85 在此类 4B 模型上的 OpenCL 支持虽功能可用，但性能不如 CPU，能耗更是高出 **6.5×**，不应作为默认加速路径。
3. 🔋 **节能机制具有“双重效应”**：
   - 不仅因为速度快，还因为卸载后 CPU 进入近似空闲状态，系统基础功耗下降，形成**超线性节能**（尤其在 indexing 场景）。
4. 📊 **可持续边缘智能的新范式**：
   - 每日千次查询场景下，单设备年节能可达 **94.9 kWh**，对大规模设备集群有显著碳减排意义，契合 Green AI 发展方向。

### 方法的局限性
- **静态计算图限制**：必须预先编译模型，context length 固定，无法动态扩展，限制了 chunk size 和 context window。
- **模型加载顺序约束**：需按参数量降序加载模型（LLM → Embedder → Reranker），否则会内存分配失败，增加工程复杂度。
- **生态不完善**：Windows ARM64 下许多 Python 包缺乏预编译轮子，需手动构建，开发门槛较高。
- **单一硬件平台验证**：目前仅在 Snapdragon X Elite 上测试，尚未覆盖 Apple Neural Engine、Intel NPU 或 MediaTek APU。
- **单个 LLM-as-judge 评估风险**：虽进行了人工抽查佐证，但仍存在 judge bias 或敏感性干扰判断的风险。

### 未来工作方向
- 支持 **dynamic shape** 的下一代 NPU runtime，解除 context length 限制。
- 更灵活的 **内存管理策略**，消除模型加载顺序依赖。
- 扩展 **NPU 兼容模型生态系统**，支持更多 embedding/reranker/LLM 架构。
- 跨平台评估：在 iPhone（Apple Neural Engine）、Intel Lunar Lake 设备等上复现实验，增强普适性。
- 多 evaluator + human annotation 联合评估，进一步夯实质量一致性结论。

---

> 💡 **总结一句话**：  
> 本文首次实现了全链路 on-device RAG 在移动 NPU 上的高效运行，证明了 **NPU 是当前最节能、最低延迟的片上加速选择**，为绿色边缘智能提供了切实可行的技术路径。

</details>

---

### 4. [Accurate and Resource-Efficient Federated Continual Learning](https://arxiv.org/abs/2606.11480)

**Authors**: Jebacyril Arockiaraj, Dhruv Parikh, Jayashree Adivarahan, Rajgopal Kannan, Viktor Prasanna  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11480v1  

#### Abstract
Federated continual learning (FCL) must learn from distributed task streams under limited resources, such as communication, computation, memory, and label availability. Existing FCL methods often rely on repeated local optimization, replay, and full supervision. Analytic alternatives avoid iterative...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Accurate and Resource-Efficient Federated Continual Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**联邦持续学习（Federated Continual Learning, FCL）** 中的多重资源约束问题展开研究。现实场景中的FCL面临以下挑战：
- **通信开销大**：传统基于梯度的方法需要多轮模型更新，上传大量参数。
- **计算成本高**：客户端需进行多次反向传播训练。
- **内存受限**：无法存储历史数据用于回放（replay）。
- **标签稀缺**：实际部署中仅有少量样本有真实标签。

现有方法通常依赖迭代优化、数据回放或全监督机制，在资源受限环境下表现不佳。

---

### **提出的新方法：FedRAN**
作者提出了 **FedRAN**（Federated Random-feature ANalytic），一种**资源感知的解析式联邦持续学习框架**，其核心思想是：
- **摒弃迭代训练**，转而采用**前向统计量构建 + 闭式求解分类器**。
- 利用**随机特征（random features）** 提升表示能力，并通过**低秩截断SVD摘要**压缩二阶特征统计（Gram矩阵），避免传输完整的 $ M \times M $ 矩阵。
- 在服务器端执行**两级OR-SVD子空间合并**（spatial-temporal merge），实现跨客户端和跨任务的知识融合。
- 引入**原型伪标签（prototype-based pseudo-labeling）** 扩展至半监督场景（FedRAN-SSL），利用无标签数据提升性能。

---

### **相比现有方法的优势**
| 维度 | FedRAN优势 |
|------|------------|
| **通信效率** | 将上传量从 $ O(M^2) $ 降至 $ O(Mr) $，显著降低带宽需求 |
| **计算效率** | 客户端仅需前向推理和SVD摘要，无需反向传播；平均比梯度法快 **190.3×** |
| **稳定性** | 冻结主干网络，避免非IID数据导致的**表征漂移（feature drift）** |
| **准确性** | 保留主导特征方向，优于仅估计一阶统计的方法（如STSA） |
| **标签利用率** | 支持伪标签机制，在仅20%标签下仍能有效学习 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在三个主流持续学习基准上进行了评估：

| 数据集 | 类型 | 总类数 | 任务数 |
|-------|------|--------|--------|
| **CIFAR-100** | 图像分类 | 100 | 5 / 10 |
| **ImageNet-R** | OOD泛化（艺术风格图像） | 200 | 10 |
| **VTAB** | 多领域视觉任务（自然/医学/遥感等） | 50 | 5 |

---

### **实验设置**
- **客户端数量**：$ K = 5 $
- **非IID划分**：使用Dirichlet分布（浓度参数 $\beta \in \{0.1, 0.5, 1\}$）模拟类别偏斜程度，$\beta$ 越小越不均衡。
- **骨干网络**：ResNet-18 和 ViT-B/16（均预训练）
- **随机投影维度**：$ M = 8192 $（ResNet）、$ M = 2048 $（ViT）
- **保留秩（rank）**：$ r = 2048 $ 或 $ 512 $
- **伪标签阈值**：余弦相似度 ≥ 0.5 才接受

---

### **评估指标**

#### **准确性指标**
- **最终准确率（Final Accuracy, $A_T$）**：最后一轮对所有已见类别的测试准确率
- **平均准确率（Average Accuracy, $A_{avg}$）**：各轮准确率的平均值，衡量遗忘控制能力

#### **资源效率指标**
- **通信成本**：单个客户端每任务最大上传字节数（MB）
- **运行时间**：完成一个任务所需的总墙钟时间（秒），含客户端与服务器计算

---

### **基线方法对比**
| 类别 | 方法 |
|------|------|
| **优化型FCL** | Finetune, FedLwF, FedEWC, FediCaRL, TARGET |
| **提示/适配器型FCL**（基于预训练模型） | DualPrompt, CodaPrompt, Fed-CPrompt, PiLoRA |
| **解析式FCL** | STSA（Spatial-Temporal Statistics Aggregation） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 指标 | FedRAN vs 最强基线（STSA） |
|------|-----------------------------|
| **平均准确率提升** | ↑ **4.8 pp**（CIFAR-100）<br>↑ **4.24 pp**（ImageNet-R）<br>↑ **2.68 pp**（VTAB） |
| **通信减少** | ↓ **30.6–121.8×** 每客户端通信量 |
| **训练加速** | 平均 **190.3× 更快**（最高达246.9×） |
| **伪标签增益（20%标签）** | ↑ **6.61 pp**（ImageNet-R） |

> 注：pp = percentage points（百分点）

---

### **与基线方法的对比结果**

#### ✅ 准确性全面领先
- 在所有数据集和 $\beta$ 设置下，FedRAN 均取得最高的 $A_{avg}$ 和 $A_T$。
- 相较于最强解析基线 **STSA**，FedRAN 在 CIFAR-100 上平均准确率高出 **+2.29~4.80 pp**。
- 优于所有基于梯度的方法（如Finetune、TARGET），尤其在长期任务流中表现出更强的抗遗忘能力。

#### ✅ 通信开销极低
| 方法 | CIFAR-100 (MB) | ImageNet-R (MB) | VTAB (MB) |
|------|----------------|------------------|-----------|
| **TARGET** | 4283.82 | 4306.32 | 4276.96 |
| **STSA** | 194.59 | 389.18 | 97.29 |
| **FedRAN** | **134.27** | **140.52** | **35.13** |

- FedRAN 通信量仅为 TARGET 的 **1/30 至 1/120**，且比 STSA 进一步降低 **1.45–2.77×**。

#### ✅ 计算速度显著提升
| 方法 | CIFAR-100 (s) | ImageNet-R (s) | VTAB (s) |
|------|---------------|----------------|----------|
| **Finetune** | 322.28 | 589.72 | 229.37 |
| **STSA** | 1.75 | 2.13 | 1.01 |
| **FedRAN** | **3.54** | **2.48** | **0.98** |

- FedRAN 虽略慢于 STSA（因额外SVD操作），但仍为**秒级响应**，远超数百秒的梯度方法。
- **平均速度快190.3倍**，适合边缘设备实时更新。

---

### **消融实验结果**

#### 🔹 **组件消融（Ablation Study）**
在 ViT 骨干上的逐步添加实验表明：
| 组件 | 平均准确率 |
|------|-----------|
| 原始ViT特征（baseline） | 87.71% |
| + 随机投影（Random Projection） | 90.21% |
| + ReLU非线性 | 93.95% |
| + 低秩SVD摘要（完整FedRAN） | **93.96%** |

✅ 结论：**非线性随机特征 + 低秩摘要** 是性能提升的关键，且引入SVD几乎无精度损失。

#### 🔹 **投影维度 $M$ 与秩 $r$ 影响**
- 增大 $r$ 和 $M$ 可提升准确率，但存在饱和效应。
- 例如当 $M=8192$，$r=1024$ 时已达大部分收益，继续增大 $r$ 提升有限但通信翻倍。
- 表明**适度的 $r$ 即可捕获关键谱信息**，实现高效-准确平衡。

#### 🔹 **伪标签有效性**
- 在仅 **20%标签可用** 时：
  - CIFAR-100：$A_{avg}$ ↑ **3.26 pp**
  - ImageNet-R：↑ **6.61 pp**
  - VTAB：↑ **5.76 pp**
- 随着标签率上升，增益减小，说明伪标签在**低标签率下尤为有效**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **解析式方法可以同时实现高性能与高效率**：FedRAN 证明了无需迭代训练也能达到甚至超越传统FCL方法的准确率。
2. **低秩谱摘要优于统计估计**：直接传输Gram矩阵的主导方向（via SVD）比通过一阶统计重构更稳定、误差可控。
3. **冻结主干 + 分析式更新可缓解表征漂移**：解决了非IID客户端更新引发的特征空间不稳定问题。
4. **伪标签能有效利用无标签数据**：在标签稀缺场景下显著提升性能，且不增加训练负担。

---

### **方法的局限性**
- **依赖预训练主干**：未探索端到端训练，假设主干已具备良好表示能力。
- **SVD计算开销**：虽然比反向传播轻量，但在大规模客户端或极高维特征下仍可能成为瓶颈。
- **理论分析基于确定性边界**：未建模伪标签噪声的影响，实际中可能存在错误累积风险。
- **当前聚焦于class-incremental setting**：尚未扩展至其他持续学习范式（如domain/task-incremental）。

---

### **未来工作方向**
1. **隐私保护增强**：结合 secure aggregation 对上传的统计量进行加密聚合。
2. **扩展至更多任务类型**：将FedRAN推广至 domain-incremental、task-aware 等设定。
3. **纳入伪标签噪声建模**：在理论层面分析错误伪标签对分类器稳定性的影响。
4. **动态秩调整机制**：根据任务复杂度自适应选择保留秩 $r$，进一步优化资源-精度权衡。

---

> 📌 **一句话总结**：  
> **FedRAN 通过“冻结主干 + 随机特征 + 低秩谱摘要 + 闭式求解”的设计，在保证高准确率的同时，实现了通信、计算和标签资源的极致节省，为资源受限下的联邦持续学习提供了新的高效范式。**

</details>

---

### 5. [Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching](https://arxiv.org/abs/2606.11583)

**Authors**: Zhuoyi Peng, Hanlin Gu, Lixin Fan, Yi Yang  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11583v1  

#### Abstract
Text-attributed graphs (TAGs) underlie real-world applications such as citation networks, social media, and e-commerce. Few-shot graph learning on TAGs is hard: with only a handful of labels per class and the rest of the graph unannotated, neither GNNs nor LLMs can learn well on their own. GNNs read...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在**文本属性图（Text-attributed Graphs, TAGs）**上的**少样本图学习（few-shot graph learning）**任务中，传统方法面临严重挑战：
- **GNNs** 依赖图拓扑结构，在度数低的“冷节点”（cold nodes）上表现差（邻居信息不足）。
- **LLMs** 依赖文本内容，在文本简短或语义模糊的节点上表现差。
- 现有 **LLM-GNN 融合方法**普遍采用“**黄金教师范式**”（golden-teacher assumption），即固定一个模型为权威教师，另一个作为学生单向学习。但在标签稀疏的少样本场景下，**任一模型都无法可靠地充当“黄金教师”**，导致其盲区被传递给学生。

### 提出的新方法与新思路
提出 **LLM-GNN Co-Teaching** 框架，**摒弃“黄金教师”假设**，实现双向协同教学：

#### （1）**双向共教机制（Bidirectional Co-Teaching）**
- GNN 和 LLM 在每一轮中**互相交换最自信的伪标签**（pseudo-labels），基于架构特定的“小损失准则”（small-loss criterion）选择：
  - GNN：交叉熵损失最低的预测。
  - LLM：生成 token 的最小对数概率最高的预测。
- 双方模型**每轮都更新**，而非冻结一方，实现动态互惠。

#### （2）**基于轮次的伪标签偏好优化（RPL-PO）**
- 利用训练轨迹中的**跨轮一致性变化**生成监督信号：
  - 当某个节点在第 `t` 轮时两模型预测**矛盾**，而在第 `t+1` 轮**达成一致**时，LLM 在该节点上的两次回答构成一个偏好对（preference pair）。
  - 将 `t+1` 轮的“同伴认可”答案作为 **chosen**，将 `t` 轮的“自我矛盾”答案作为 **rejected**。
- 使用 **Direct Preference Optimization (DPO)** 对 LLM 进行训练，无需人工标注、奖励模型或外部评判。

### 相比现有方法的优势
- **无权威假设**：不指定任一模型为“黄金教师”，避免盲区传播。
- **自监督信号挖掘**：从模型交互的**时间轨迹**中自动提取高质量监督信号（RPL-PO）。
- **互补性利用**：充分发挥 GNN（结构鲁棒）和 LLM（语义理解）的互补归纳偏置。
- **适用于异构架构**：首次在 GNN 与 LLM 这类异构模型间实现迭代共教。

---

## 2. 核心实验方法和设置

### 数据集
在六个标准 **Text-attributed Graphs** 上进行评估，涵盖不同规模与领域：
| 数据集 | 类别数 | 节点数 | 边数 | 领域 |
|--------|-------|--------|------|------|
| **Cora** | 7 | 2,708 | 10,858 | 学术引用 |
| **Citeseer** | 6 | 3,186 | 4,277 | 学术引用 |
| **PubMed** | 3 | 19,717 | 88,670 | 生物医学引用 |
| **WikiCS** | 10 | 11,701 | 431,726 | 维基百科超链接 |
| **ogbn-arxiv** | 40 | 169,343 | 1,166,243 | arXiv 计算机科学论文 |
| **ogbn-products** | 47 | 54,025 | 72,319 | 亚马逊商品共购 |

### 实验设置
- **任务**：少样本半监督节点分类（few-shot semi-supervised node classification）
- **标签预算**：每类仅 `k=3`, `5`, `10` 个标签用于训练。
- **评估指标**：测试集节点分类准确率（Accuracy %）。
- **划分**：每个数据集固定验证集（500 或官方划分），其余未标记节点用于伪标签生成。

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **经典 GNN** | GCN, GAT, GraphSAGE |
| **零样本 LLM** | Zero-shot, Graph-CoT, Neighbor-Augmented Prompting |
| **LLM-as-Enhancer** | TAPE, GLEM |
| **LLM-as-Predictor** | LLM-GNN, LLaGA, GraphGPT |
| **GNN-as-Judge** | GNN-as-Judge (GAJ) — 当前最优基线 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（3-shot 准确率 %）

| Method | Cora | Citeseer | PubMed | WikiCS | ogbn-arxiv | ogbn-products |
|--------|------|----------|--------|--------|------------|--------------|
| **GNN-as-Judge (GAJ)** | 77.89 | 73.59 | 87.12 | 74.80 | 62.21 | 81.02 |
| **LLM-GNN Co-Teaching (Ours)** | **85.75** | **77.12** | **91.32** | **74.80** | **69.94** | **82.82** |
| **绝对提升** | **+7.86** | **+3.53** | **+4.20** | **+0.00** | **+7.73** | **+1.80** |

> ✅ **平均绝对增益**：在 3-shot 下相比最强基线 **GNN-as-Judge** 平均提升 **+5.40%**。

### 与基线方法的对比结果
- **全面超越所有基线**：在全部 6 个数据集、所有标签预算（3/5/10-shot）下均取得 SOTA 性能。
- **在困难任务上优势显著**：
  - ogbn-arxiv（40类）和 ogbn-products（47类）等大类别空间任务中，传统方法性能急剧下降（如 LLaGA 降至 ~30%），而本方法仍保持高精度。
  - 表明方法在**高难度、细粒度分类**任务中更具鲁棒性。
- **双向优于单向**：所有单向监督方法（无论谁当教师）均被双向共教框架超越，验证了“去中心化教学”的有效性。

### 消融实验结果（Ablation Study）

| 变体 | Cora | ogbn-arxiv | 说明 |
|------|------|------------|------|
| **完整模型** | 85.75 | 69.94 | — |
| **移除双向教学** | 78.66 | 65.50 | 性能大幅下降，接近 GAJ 基线，表明**互教机制至关重要**。 |
| **移除 RPL-PO** | 83.03 | 66.77 | 性能下降，证明**轨迹偏好优化提供额外增益**。 |
| **固定选择比例**（R=0.5） | 83.20 | 67.10 | 动态退火策略更优。 |
| **仅选一致节点** | 82.52 | 68.47 | “小损失”选择优于“一致性过滤”。 |
| **移除邻居信息** | 85.08 | 68.76 | 结构上下文对 LLM 至关重要。 |

---

## 4. 关键结论和发现

### 主要发现
1. **“黄金教师”假设在少样本下失效**：在标签稀缺时，任一模型都不足以成为可靠教师，单向监督会固化错误。
2. **双向共教可实现协同进化**：GNN 和 LLM 通过交换高置信伪标签，能在各自弱项区域相互补强，逐步提升。
3. **训练轨迹蕴含自监督信号**：RPL-PO 成功从“矛盾→一致”的状态转移中提取偏好对，实现了**无标注、无奖励模型的高效对齐**。
4. **误差结构分析证实互补性**：
   - GNN 在低度节点上错误集中，LLM 错误分布均匀。
   - 共教后，GNN 在低度节点的错误率显著降低，表明 LLM 的语义信号有效弥补了结构信息缺失。

### 方法的局限性
- **计算开销较高**：多轮迭代和 DPO 更新导致训练时间较长（约是单次方法的 2–3 倍），尤其在大规模图上（如 ogbn-arxiv 需 ~4.7 小时）。
- **依赖 LLM 基础能力**：若 LLM 本身能力较弱（如换用 Vicuna-7B），性能显著下降，表明 LLM 是性能瓶颈之一。
- **文本质量依赖性强**：当前方法假设节点文本具有描述性。在文本噪声大、稀疏或缺失的图（如分子图、金融交易图）上尚未验证。

### 未来工作方向
- 探索在**非文本图**或**多模态图**上的扩展。
- 设计更高效的共教调度策略以减少计算成本。
- 将共教范式推广至其他 LLM-与其他模型（如 Diffusion Model）的协作学习任务。
- 研究如何缓解 LLM 伪标签中的**社会偏见放大**问题。

---

> **代码地址**：https://github.com/llmgnncoteaching/LLM-GNN-Coteaching

</details>

---

### 6. [Efficient Time Series Clustering from Multiscale Reservoir Dynamics with Granular-Ball Anchoring Graph Optimization](https://arxiv.org/abs/2606.12077)

**Authors**: Yifan Wang, Lifeng Shen, Shuyin Xia, Yi Wang  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.12077v1  

#### Abstract
Time-series clustering remains challenging due to the inherent trade-off between clustering effectiveness and computational efficiency. Similarity-based methods often suffer from quadratic complexity caused by pairwise distance computations, while deep learning-based approaches typically rely on cos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Time Series Clustering from Multiscale Reservoir Dynamics with Granular-Ball Anchoring Graph Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
时间序列聚类面临两大挑战：
- **计算效率低**：基于相似性的方法（如 DTW）因成对距离计算导致 $O(N^2)$ 复杂度，难以扩展到大规模数据。
- **表示能力与训练成本的权衡**：深度学习方法虽能学习强表达力的特征，但依赖迭代训练和大量参数，计算开销大。

此外，现有方法在建模多尺度动态特性时通常需要重复训练或复杂集成，且传统聚类（如 k-means）对簇几何形状假设过强。

### **提出的新方法与新思路**
本文提出了 **MSRGC-Net**（Multi-Scale Reservoir Granular-ball Consensus Network），一种高效的时间序列聚类框架，融合了以下三个关键技术：

1. **Training-free Multiscale Reservoir Encoding**
   - 利用多个具有不同谱半径（spectral radius）的固定 **Echo State Network (ESN)** 构建多尺度储层，无需反向传播即可从原始时间序列中提取互补的时序表征。
   - 不同谱半径控制记忆容量：小值捕获短期动态，接近1的值保留长期依赖。

2. **Granular-Ball Anchoring Graph Construction**
   - 引入 **granular-ball computing (GBC)** 自适应地将高维状态空间划分为密度一致的区域（即“粒球”），每个粒球作为锚点代表局部结构。
   - 构建样本到锚点的稀疏图（anchoring graph），避免全样本间亲和力建模，显著降低复杂度。

3. **Consensus-based Graph Optimization**
   - 设计一个统一的优化目标，通过加权聚合多个视图下的锚定图，实现跨尺度信息融合。
   - 引入自适应权重机制，使对齐质量更高的视图获得更大贡献。

### **相比现有方法的优势**
| 维度 | MSRGC-Net | 传统方法 |
|------|-----------|---------|
| **训练方式** | Training-free（无反向传播） | 需要端到端训练（如 DEC, TimeSURL） |
| **复杂度** | 近线性 $O(N)$ | 二次方 $O(N^2)$ 或更高 |
| **可扩展性** | 支持百万级样本 | 受限于内存与计算资源 |
| **鲁棒性** | 区域级建模抑制噪声影响 | 点对点关系易受异常值干扰 |
| **多尺度整合** | 显式建模并优化融合 | 多数为单尺度或简单拼接 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在 **UCR/UEA 时间序列档案** 中选取共 **10个基准数据集**，涵盖：
- **Multivariate（5个）**：CharacterTrajectories (CT), JapaneseVowels (JV), BasicMotions (BM), Cricket (Cric), SelfRegulationSCP1 (SCP1)
- **Univariate（5个）**：BCCrop, EPG-R, EPG-S, Wafer, PigArtPressure
- 数据长度从几十到上千不等，类别数从2到20，覆盖动作识别、医疗、工业等多个领域。

### **实验设置**
- **Reservoir 设置**：
  - 储层数量 $V=3$，谱半径取不同值（体现多尺度）
  - 固定参数：储层大小 $R=400$，连接率 $\beta=0.25$，输入缩放 $w=0.15$
- **Granular-ball 参数**：
  - 锚点数量 $m \ll N$，通过分裂准则自动确定区域粒度
- **优化参数**：
  - 对齐系数 $\gamma = 10^3$，正则化项 $\lambda = 10^{-1}$，网格搜索调参
- **硬件环境**：Intel i7-12700 CPU + 64GB RAM，每组实验重复10次取均值

### **评估指标**
采用三种标准无监督聚类评价指标：
- **Normalized Mutual Information (NMI)**
- **Adjusted Rand Index (ARI)**
- **Rand Index (RI)**

### **基线方法对比**
分为三类共 **11种代表性方法**：
1. **Raw-data Based**  
   - `k-Shape`, `Fuzzy-kShape`, `TCK`
2. **Representation Learning Based**  
   - `Modular-RC`, `GRAIL`, `Time2Feat`, `DEC`, `TimeSURL`, `TFMCC`
3. **Multi-view Clustering Methods**  
   - `GB-SMKKM`, `MV-CAGAF`

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Multivariate 数据集平均表现）**

| 模型 | NMI ↑ | ARI ↑ | RI ↑ |
|------|-------|-------|-----|
| **MSRGC-Net (Ours)** | **0.689** | **0.562** | **0.878** |
| GRAIL | 0.608 | 0.506 | 0.869 |
| Time2Feat | 0.604 | 0.488 | 0.861 |
| GB-SMKKM | 0.612 | 0.458 | 0.859 |
| DEC | 0.565 | 0.362 | 0.838 |

> ✅ 在 **15项评估结果中，MSRGC-Net 获得12项最优，2项第二**

#### **亮点案例**
- **JapaneseVowels (JV)**：  
  - ARI 达 **0.750**，比第二名 GRAIL（0.581）提升 **16.9%**
- **Cricket (Cric)**（超长序列 T=1197）：  
  - RI 达 **0.983**，远超其他方法（最高仅0.966）
- **CharacterTrajectories (CT)**（20类）：  
  - NMI 达 **0.789**，显著优于单视图 rm-ESN（0.476）

### **与基线方法的对比结果**
- 相比 **raw-data 方法**（如 k-Shape）：
  - 利用 reservoir 提取动态特征，在复杂模式下性能更优（如 JV 上 ARI 提升超6倍）
- 相比 **deep learning 方法**（如 DEC, TimeSURL）：
  - 无需训练却达到甚至超越其性能，且运行速度更快
- 相比 **multi-view 方法**（如 GB-SMKKM）：
  - 在保持高效的同时实现更强的跨尺度融合能力

### **消融实验结果（Ablation Study）**
使用 **RI 指标** 进行组件分析（见 Table 2）：

| 变体 | Multiscale | Granular-ball | Optimization | 平均 RI |
|------|------------|---------------|--------------|--------|
| w/o Multiscale | × | √ | √ | 0.844 |
| w/o Granular-ball | √ | × | √ | 0.832 |
| w/o Optimization | √ | √ | × | 0.811 |
| **Full MSRGC-Net** | √ | √ | √ | **0.883** |

> 🔍 发现：
- **多尺度机制贡献最大**：移除后性能下降最明显
- **粒球锚定提升结构感知能力**：相比 k-means 更鲁棒
- **共识优化增强融合效果**：简单平均融合会削弱多视图优势

---

## **4. 关键结论和发现**

### **主要发现**
1. **Training-free reservoir computing 可有效替代深度模型进行特征提取**  
   - 固定权重的 ESN 能稳定捕捉非线性动态，无需昂贵训练过程。

2. **Granular-ball 实现了从“点”到“区域”的跃迁**  
   - 以密度一致的粒球作为锚点，既能压缩表示又能保留局部拓扑结构，提升聚类鲁棒性。

3. **Consensus graph optimization 是多尺度融合的关键**  
   - 自适应加权机制让高质量视图主导融合过程，优于简单拼接或平均。

4. **MSRGC-Net 实现了精度与效率的帕累托前沿**  
   - 如 Figure 4 所示，其位于 **Pareto frontier 上方左侧**，兼具高准确率与低耗时。

5. **具备良好的可扩展性**  
   - 在含 **180万样本** 的 Pedestrian 数据集上测试，runtime 呈近线性增长，RI 仍达 **0.947**，显著优于 k-Shape（0.882）

### **方法的局限性**
- **依赖 reservoir 设计经验**：虽然无需训练，但 reservoir 规模、谱半径组合等仍需合理设定。
- **适用于中等维度时间序列**：极高维或多模态异构信号可能需额外预处理。
- **对周期性极弱或高度随机的数据泛化能力待验证**。

### **未来工作方向**
1. 探索 **自动化 reservoir 配置搜索策略**（如贝叶斯优化）
2. 将框架拓展至 **time series anomaly detection** 和 **classification** 任务
3. 结合 **causal discovery** 分析多变量时间序列间的驱动关系
4. 开发 **online/streaming 版本** 以支持实时聚类应用

---

> 📌 **一句话总结**：  
> MSRGC-Net 通过 **training-free 多尺度储层编码 + granular-ball 锚定图 + 共识优化**，实现了高效、鲁棒、可扩展的时间序列聚类，在多个 benchmark 上超越 SOTA 方法，同时将运行时间降低一个数量级以上。

</details>

---

### 7. [Teaching Diffusion to Speculate Left-to-Right](https://arxiv.org/abs/2606.11552)

**Authors**: Lexington Whalen, Yuki Ito, Ryo Sakamoto  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.11552v1  

#### Abstract
Large language models (LLMs) achieve remarkable performance across a wide range of tasks, but their autoregressive decoding process incurs substantial inference costs due to inherently sequential token generation. Speculative decoding addresses this bottleneck by employing a lightweight draft model ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Teaching Diffusion to Speculate Left-to-Right*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **speculative decoding** 中的一个关键训练-推理不一致问题（training-verification gap）：
- 当前基于 **block-diffusion** 的 draft model（如 DFlash）在训练时采用 **bidirectional attention**，即每个 token 可以看到块内所有其他位置的信息。
- 但在推理时，target model 以 **left-to-right autoregressive 方式验证** draft tokens，并通过 rejection sampling 接受最长的因果一致前缀。
- 这导致一个矛盾：即使后续 token 预测正确，只要前面某个位置出错，整个后缀都会被丢弃。而标准训练目标（如 position-uniform cross-entropy）并未考虑这种“前缀截断”效应，造成训练信号与实际吞吐量目标脱节。

### 提出的新方法
作者提出三种**正交且可叠加**的训练时干预策略，以对齐训练目标与推理时的实际奖励机制：

| 方法 | 核心思想 |
|------|--------|
| **Position-wise Loss Decay (权重衰减)** | 对越靠前的位置赋予更高的损失权重（指数衰减），因为早期错误会截断更多后续 token。 |
| **First-Error Focal Loss (首错焦点损失)** | 引入辅助损失项，仅作用于每个 block 中第一个预测错误的位置（chain breaker），直接优化最关键的“断链点”。 |
| **Chain Reward (链式奖励)** | 使用一个可微的代理函数来近似期望接受长度（expected accepted length），并对整个联合前缀概率进行优化，使梯度自然偏向长连续正确的序列。 |

这三种方法分别从 **position 轴、block-conditional 首错轴、joint prefix 轴** 上重塑训练信号，互不冲突，可加性组合。

### 相比现有方法的优势
- **无需额外 forward pass**：所有改进均复用已有计算图，不增加推理开销。
- **保持 exactness 合同**：不改变 speculative decoding 的 rejection sampling 规则，保证生成样本统计等价于原模型。
- **与 test-time 方法正交**：可与 ddTree、SpecDiff-2 等推理阶段优化技术结合使用。
- **显著提升 accepted draft length**：在多个 benchmark 上平均提升 21%-76%，最高达 109.8%（结合 target-aligned 数据）。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主训练数据**：`ShareGPT`，包含多轮用户-助手对话，格式化为 Llama-3 模板。
- **目标对齐数据（ablation）**：`Nemotron-V2 + CodeAlpaca`，经 target model（Llama-3-8B-Instruct）重生成，实现 target-aligned 训练。
- **评估基准（6个）**：
  - 数学推理：GSM8K, AIME
  - 编程能力：HumanEval, MBPP, LiveCodeBench
  - 开放对话：MT-Bench

### 实验设置
- **Target Model**：Meta-Llama-3-8B-Instruct（及其他变体如 Llama-3.2-3B, Qwen-3-4B/8B）
- **Drafter Model**：基于 DFlash 架构的 4 层 block-diffusion 模型，block size $ K=16 $
- **特征输入**：使用 target 模型第 {0,10,20,30} 层隐藏状态作为条件
- **训练配置**：
  - AdamW 优化器，cosine schedule，warm-up 1.5%
  - batch size 32（8×H100），3 epochs
  - 峰值学习率调优为 $1e^{-3}$

### 评估指标
- **Average Accepted Length ($\bar{T}$)**：每轮 speculative decoding 平均成功接受的 token 数（含 bonus token），是衡量 throughput 的核心指标。
- **Throughput (TPS)**：tokens per second，综合反映速度增益。
- **消融分析**：逐步添加三种干预，观察 $\bar{T}$ 提升情况。

### 基线方法对比
- **Baseline**：原始 DFlash，使用 uniform CE loss（无任何对齐机制）
- **对比对象**：
  - 单独应用三种方法 vs. baseline
  - 组合使用 vs. 单独使用
  - 与 test-time 方法（ddTree）结合效果
  - 与同期工作 SpecDiff-2 结合潜力

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 和 Table 3）

| 配置 | Avg. $\bar{T}$ | 提升幅度 |
|------|----------------|----------|
| Position-uniform baseline | 2.376 | — |
| + Loss Decay ($\gamma=10$) | 2.583 | +8.7% |
| + Focal Loss ($\alpha_f=0.3$) | 2.892 | +21.7% |
| + Chain Loss ($\alpha_c=40$) | **3.420** | **+43.9%** |

> 在 Llama-3-8B-Instruct 上，六项任务平均接受长度从 2.376 提升至 3.420。

#### 分任务表现（vs. baseline row b）
- **HumanEval**：2.803 → **4.336**（+54.7%）
- **AIME**：2.705 → **4.757**（+75.9%）
- 显示在 **reasoning 和 code 类任务**上提升最大，因这些任务更依赖长连续正确预测。

#### 跨模型泛化（Table 3）
在不同 target model 上均取得一致增益：
- Llama-3.2-3B：+27.5%
- Qwen-3-4B：+20.9%
- Qwen-3-8B：+24.7%

#### 与其他技术结合（Table 7）
| 技术 | TPS | $\bar{T}$ |
|------|-----|-----------|
| Baseline (no ddTree) | 126.75 | 2.376 |
| Fully stacked (no ddTree) | 225.12 | 3.420 |
| Baseline + ddTree | 188.68 | 3.791 |
| **Fully stacked + ddTree** | **294.68** | **4.609** |

> 表明训练时对齐与推理时搜索策略（ddTree）**完全正交且可叠加**，联合使用带来高达 **+132.5% TPS 提升**。

#### 与 SpecDiff-2 兼容性（Table 8）
在 SpecDiff-2 训练的 draft model 上继续应用本文方法：
- 基线（streak-distillation only）：$\bar{T}=2.339$
- 完整堆叠（+decay+focal+chain）：$\bar{T}=4.071$（+74.0%）

> 说明本文方法可作为 **通用增强模块**，进一步提升已有先进训练范式的性能。

---

## 4. 关键结论和发现

### 主要发现
1. **训练-验证不对称是 block-diffusion drafters 的瓶颈**：标准 bidirectional 训练导致大量正确预测 token 因上游错误被丢弃（Table 1 显示约 47% 正确 token 被浪费）。
2. **三种干预策略有效且正交**：
   - **Loss decay** 最简单，提供基础增益；
   - **Focal loss** 更智能地聚焦“断链点”；
   - **Chain reward** 效果最强，直接逼近最终目标。
3. **组合效果显著优于单一方法**：三者叠加带来接近线性的增益，表明它们捕捉了不同的优化维度。
4. **与 test-time 方法兼容**：可无缝集成到 ddTree、SpecDiff-2 等系统中，形成“训练+推理”双重加速。

### 方法的局限性
- 所有方法仍基于 **teacher-forcing**，未引入强化学习或策略梯度。
- **chain reward 的 trade-off**：过强的 chain loss（如 $\alpha_c=50$）会导致性能下降，需仔细调参。
- 当前 block size 固定为 16，扩展到更大 K 时性能下降（Table 6），提示训练与推理 block size 应匹配。

### 未来工作方向
- 将 chain reward 与 **reinforcement learning** 结合，端到端优化 expected speedup。
- 探索动态 block size 或 adaptive drafting 策略。
- 将类似思想应用于 **non-diffusion non-autoregressive models**。
- 研究如何将 position decay 和 focal loss **自动化**（如通过 meta-learning）而非手动设定。

---

> ✅ **总结一句话**：  
> 本文揭示了 diffusion-based speculative decoding 中的训练-验证不对称问题，并提出了三种轻量、正交、可叠加的训练干预方法，在不改变推理流程的前提下，显著提升了 draft token 的利用率和系统吞吐量，为高效 LLM inference 提供了一套实用且通用的优化框架。

</details>

---

### 8. [Verifiable Environments Are LEGO Bricks: Recursive Composition for Reasoning Generalization](https://arxiv.org/abs/2606.12373)

**Authors**: Hao Xiang, Qiaoyu Tang, Le Yu, Yaojie Lu, Xianpei Han, Ben He, Le Sun, Bowen Yu, Peng Wang, Hongyu Lin, Dayiheng Liu  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.12373v1  

#### Abstract
Reinforcement Learning (RL) with verifiable environments has emerged as a powerful approach for enhancing the reasoning capabilities of Large Language Models (LLMs). While prior research demonstrates that scaling environment quantity improves RL performance, existing manual or individual constructio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Verifiable Environments Are LEGO Bricks: Recursive Composition for Reasoning Generalization*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Reinforcement Learning with Verifiable Rewards (RLVR)** 的推理能力提升方法依赖大量人工或自动生成的 **verifiable environments**（如代码、数学题、逻辑谜题等）。然而，现有方法通常以**独立方式**构建环境，导致环境池的扩展呈**线性增长**，难以在有限成本下实现足够的多样性，从而限制了模型的**reasoning generalization**（推理泛化）能力。

### 提出了什么新方法或新思路
本文提出 **RACES**（**Recursive Automated Composition for Environment Scaling**），其核心思想是将可验证环境视为可组合的“**LEGO积木**”，通过递归组合的方式生成新的复合环境。  
关键洞察是：当一个环境的输出类型（codomain）与另一个环境的输入类型（domain）匹配时，二者可以自动融合为一个新的、仍可验证的环境。

RACES 定义了四种**composition operators**：
- **SEQUENTIAL**：链式执行，要求模型计算中间输出。
- **PARALLEL**：并行处理多个独立任务。
- **SORT**：打乱顺序，要求模型恢复正确执行序列。
- **SELECT**：从候选集中选择并排序正确的子集来构成有效路径。

这种方法实现了环境空间的**指数级扩展**，而非线性增长。

### 相比现有方法的优势
- **高效利用已有环境**：仅用 50 个基础环境即可达到训练于 300 个独立环境上的性能水平。
- **诱导多样化推理模式**：不同 composition operator 引导出多样的推理行为（如状态追踪、顺序推断、干扰项排除等）。
- **支持无限扩展**：理论上可通过递归组合构造任意复杂度的环境。
- **无需人工设计适配器**：不同于问题级组合（problem-level composition），RACES 在程序层面直接兼容，避免了繁琐的手工对齐。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **训练环境池**：由 300 个 verifiable environments 构成，来源包括：
  - 标准算法题数据集（如 LeetCode 风格）
  - 使用 Claude-Sonnet-4.5 自动生成的环境
  - 手动编写环境
- **评估基准（evaluation benchmarks）**（全部未参与训练）：
  - **LiveCodeBench (LCBench)**：代码生成
  - **AIME 2024/2025**：数学推理
  - **Enigmata**：合成逻辑推理谜题
  - **IFEval**：指令遵循能力
  - **LongBench-v2**：长上下文理解

### 实验设置和评估指标
- **模型架构**：
  - 主要实验：`DeepSeek-R1-Distill-Qwen-14B` 和 `Qwen3-14B`
  - 分析实验：`Qwen3-4B-Instruct-2507`
- **训练框架**：基于 VERL 实现，使用 GRPO 算法进行 RL 优化。
- **训练参数**：
  - 总步数：300 步（主实验），200 步（分析实验）
  - 每步采样 128 个问题，共约 12,800 个训练实例
  - 最大上下文长度达 32K tokens
- **评估方式**：
  - 多次运行取平均（AIME 测试 32 次，其余 4 次）
  - 报告各 benchmark 上的准确率及平均得分（Avg.）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Base** | 未经 RL 微调的原始模型 |
| **RL_individual** | 在 300 个独立环境中进行 RL 训练 |
| **RL_RACES** | 使用 RACES 框架生成复合环境进行 RL 训练 |

此外还进行了消融实验，比较不同数量的基础环境（50 vs 300）、不同 composition size（长度 2–6）的影响。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | LCBench | Enigmata | LBench-V2 | IFEval | AIME | **Avg.** |
|-------|--------|----------|-----------|--------|------|---------|
| DeepSeek-R1-Distill-Qwen-14B (Base) | 47.2 | 32.3 | 32.5 | 70.6 | 58.5 | **48.2** |
| + RL_individual | 46.9 | 34.2 | 33.7 | 69.3 | 59.8 | **48.8** |
| + RL_RACES | **48.8** | **35.4** | **36.0** | **74.6** | **61.7** | **51.3** |
| | | | | | | |
| Qwen3-14B (Base) | 55.0 | 47.4 | 32.5 | 74.8 | 58.8 | — |
| + RL_individual | 56.3 | 48.2 | 34.1 | 84.5 | 76.0 | **60.1** |
| + RL_RACES | **57.0** | **49.2** | **35.5** | **85.7** | **77.0** | **61.1** |

> ✅ **结论**：RACES 显著提升两个模型家族在所有 benchmark 上的表现。

### 与基线方法的对比结果
- 对 `DeepSeek-R1-Distill-Qwen-14B`：
  - 平均分从 **48.2 → 51.3**（↑3.1 pts）
  - 特别在 IFEval（+4.0）和 LongBench-v2（+3.5）上提升显著
- 对 `Qwen3-14B`：
  - 平均分从 **60.1 → 61.1**（↑1.0 pt），且在所有子任务均有增益

> ⚠️ 注意：`RL_individual` 收益极小甚至负向（如 LCBench 下降），说明单纯增加独立环境效果有限。

### 消融实验结果（Table 2）

| 设置 | Avg. Score |
|------|------------|
| RL_individual (300 envs) | 50.4 |
| **RL_RACES (50 envs)** | **50.8** |
| RL_RACES (300 envs) | **51.9** |

> ✅ **关键发现**：仅用 **50 个基础环境** 经 RACES 组合后，性能已超过在 **300 个独立环境** 上训练的结果，证明其**环境利用率极高**。

#### Composition Size 影响（Figure 3 & Table 3）
| Composition Size | Average Score |
|------------------|---------------|
| 2 | 50.8 |
| 3 | 50.7 |
| 4 | 51.0 |
| 5 | **51.2** |
| 6 | 50.7 |

> 🔁 存在非单调关系：中等深度（size=5）最优；过深（size=6）反而下降，因优化难度过大导致 reward 稀疏。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **复合环境比独立环境更有效**：RACES 生成的复合任务能引导模型学习更深层次的推理模式（如中间状态维护、操作顺序推断、干扰过滤），从而显著提升跨领域泛化能力。
2. **组合带来质变而非量变**：相比简单堆砌更多环境，**结构化的递归组合**才是提升 reasoning generalization 的关键。
3. **环境效率大幅提升**：**50 个 base environments + RACES ≈ 300 个 individual environments**，极大降低对大规模环境池的依赖。
4. **composition size 是可控的教学变量**：可通过调节组合长度实现“课程学习”式的训练设计，在可训练性和推理深度之间取得平衡。

### 方法的局限性
1. **operator 设计尚未完备**：目前仅支持 SEQUENTIAL、PARALLEL、SORT、SELECT 四种结构，缺乏条件分支（conditional branching）和循环（looping）等更复杂的控制流。
2. **依赖较强的基础模型**：对于 reasoning 能力较弱的小模型，复合任务可能导致 reward 过于稀疏，难以收敛。
3. **需要大上下文窗口**：复合任务产生长推理轨迹，需至少 **32K token context length** 才能完整训练。
4. **运行时异常风险**：尽管有质量过滤机制，部分 domain-compatible 的组合仍可能因边界情况失败。

### 未来工作方向
- 探索更丰富的 composition operators（如 IF/ELSE、WHILE 等）
- 结合 curriculum learning 动态调整 composition size
- 将 RACES 应用于其他模态（如视觉-语言任务）
- 开发轻量化版本以适应中小规模模型
- 研究如何自动化发现高价值的 composition 路径（而非随机搜索）

---

> 🧩 **总结一句话**：  
> RACES 将 verifiable environments 视为可编程的 LEGO 积木，通过递归组合打破环境扩展的线性瓶颈，在更少资源下实现了更强的推理泛化能力，为 RLVR 提供了一条高效、可控、可扩展的新路径。

</details>

---

### 9. [APEX: A Network-Native Time-Series Foundation Model for Forecasting and Anomaly Detection for Wireless Edge Operations](https://arxiv.org/abs/2606.11553)

**Authors**: Swadhin Pradhan, Niloo Bahadori, Peiman Amini  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11553v1  

#### Abstract
Generic time-series foundation models transfer poorly to wireless network telemetry whose signals are bursty, zero-inflated, and coupled across protocol layers. We present APEX, a network-native, decoder-only transformer for forecasting enterprise AP telemetry, and evaluate it on DHCP degradation as...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：APEX: A Network-Native Time-Series Foundation Model for Forecasting and Anomaly Detection for Wireless Edge Operations

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
无线网络中的 **Access Point (AP)** 故障通常在用户感知后才被发现（如 DHCP 超时、连接中断），而现有的通用 **Time-Series Foundation Models (TSFMs)** 在应用于企业级无线网络遥测数据时表现不佳。原因在于网络信号具有以下特性：
- **Bursty**（突发性强）
- **Zero-inflated**（正常状态下大量为零）
- **Cross-layer dependencies**（跨协议层耦合）
- **Topology-dependent dynamics**（拓扑依赖动态）

此外，将原始遥测数据上传至云端进行分析会带来高带宽开销、隐私风险和延迟。

### 提出了什么新方法或新思路
提出 **APEX** —— 一种专为无线网络设计的 **network-native、decoder-only Transformer** 架构，支持在边缘设备（AP）上运行，实现：
- **Forecasting**（预测）
- **Anomaly Detection**（异常检测）
- **Edge deployment**（边缘部署）

#### 主要创新点：
1. **Network-native pretraining**  
   首次在大规模真实企业无线网络遥测数据上预训练 TSFM，编码协议层先验知识。
   
2. **Unified forecasting and anomaly detection**  
   利用 **MC-dropout** 从同一个 checkpoint 同时生成预测和不确定性区间，实现统一模型完成两项任务，无需独立的检测模块。

3. **Edge-deployable lightweight model (APEX-Edge)**  
   设计轻量级版本 APEX-Edge (10.5M 参数)，可在 AP 级 ARM 硬件（如 Raspberry Pi 5）上实现 **sub-second inference**，且不需上传原始数据。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 数据来自约 **4,500 个生产级无线网络**，涵盖 **~100K AP 时间序列**
- 每个 AP 包含 **34 个 metric**，采样粒度为 **30 分钟**（每日 48 个观测点）
- 输入特征为 **10-channel multivariate telemetry**，包括：
  - **5 个目标变量**：client count, offer rate, ACK ratio, success rate, latency
  - **5 个外生变量**：server count, VLAN count, timeout rate MAX, latency MAX, latency STD
- 数据经过两级聚合以保留分布信息（AVG/MAX/STD）

### 实验设置和评估指标

#### 预测任务（Forecasting）
- **预测步长**：192 步（即 4 天）
- **评估指标**：
  - MAE（Mean Absolute Error）
  - RMSE（Root Mean Squared Error）
  - MAPE（Mean Absolute Percentage Error）
- **训练/测试划分**：每个 AP 的最后 192 步作为测试集

#### 异常检测任务（Anomaly Detection）
- 使用 **consensus ground truth**：若超过 3 种独立方法标记某时刻为异常，则视为真异常
- **评估指标**：
  - Precision
  - Recall
  - F1 Score

#### 推理效率测试
- 平台：Raspberry Pi 5（ARM Cortex-A76，类比现网 AP SoC）
- 测试指标：inference latency、memory usage、model size

### 基线方法对比

| 类别 | 基线模型 |
|------|---------|
| **统计模型** | SARIMA, VAR |
| **通用 TSFM** | TimesFM (200M), Toto (151M), Chronos-2 (120M) |
| **其他检测方法** | Z-Score, Isolation Forest, VAR-Mahalanobis, SARIMAX CI |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 预测性能（Table 2）
| Model | MAE ↓ | RMSE ↓ | MAPE ↓ |
|-------|--------|--------|--------|
| SARIMA | 4.82 | 7.31 | 5.1% |
| Toto (best general TSFM) | 3.64 | 5.52 | 3.8% |
| **APEX-Large (multi)** | **2.98** | **4.61** | **3.1%** |

> APEX-Large 相比最强通用模型 **Toto**，MAE 下降 **18%**；相比 SARIMA 下降 **38%**

#### ✅ 异常检测性能（Table 3）
| Method | F1 ↑ |
|--------|------|
| VAR-Mahalanobis | 0.94 |
| **APEX-Large MC-dropout** | **0.93** |
| Toto P5/P95 | 0.85 |
| APEX-Edge MC-dropout | 0.89 |

> APEX-Large 异常检测 F1 达到 **0.93**，接近最优统计方法，且能捕捉非线性故障模式

#### ✅ 边缘部署性能（Table 4）
| Model | Params | Latency (96-step) | Cloud? |
|-------|--------|-------------------|--------|
| TimesFM / Toto / Chronos-2 | >100M | — | Yes |
| **APEX-Edge (multi)** | **10.5M** | **~202ms** | **No** |

> APEX-Edge 可在 **202ms 内完成一次预测**，峰值内存仅 428MB，适合部署于资源受限 AP

### 消融实验结果（Ablation Study）

| Variant | MAE | 说明 |
|--------|-----|------|
| APEX-Large (1D) | 3.21 | 单变量输入，性能下降 |
| APEX-Large (multi) | **2.98** | 多变量输入显著提升 |
| APEX-Edge (1D) | 4.78 | 接近 SARIMA，性能差 |
| APEX-Edge (multi) | **3.87** | 接近 Toto（3.64），但参数少 26× |

> 结论：**multivariate causal-chain 输入结构是性能关键**，可补偿 96% 参数减少带来的损失

---

## 4. 关键结论和发现

### 主要发现
1. **通用 TSFM 无法有效迁移至网络领域**  
   尽管 TimesFM、Toto 等在金融、气象等领域表现优异，但在网络遥测上仍落后于 domain-specific 模型，说明 **gap 来自数据而非架构**。

2. **Network-native pretraining 是关键**  
   在真实网络数据上预训练可显著提升预测精度（↓18% MAE），证明领域适配的重要性。

3. **多变量因果链输入结构至关重要**  
   显式建模 DHCP 协议因果路径（client → server response）使小模型（APEX-Edge）也能媲美大模型性能。

4. **MC-dropout 支持统一模型双任务输出**  
   仅用一个 checkpoint 即可同时提供高质量预测与校准后的异常检测，简化部署流程。

5. **边缘部署完全可行**  
   APEX-Edge 可在 AP 级硬件实现亚秒级推理，满足实时性要求，并保障 **zero cloud dependency** 和 **data privacy**。

### 方法的局限性
- **Anomaly labels 为共识伪标签**，非人工标注，可能存在偏差
- 当前评估集中在 **DHCP degradation** 场景，尚未扩展到 RF、roaming 等其他网络问题
- 边缘延迟基于 Raspberry Pi 5 测量，虽代表当前 AP SoC 水平，但仍为代理平台

### 未来工作方向
- 将 causal-chain 输入结构推广至 **RF telemetry** 和 **client roaming behavior**
- 开发跨域统一的 multivariate forecasting benchmark for edge networking
- 探索 INT8 量化、神经加速器集成以进一步降低延迟
- 扩展至更多网络协议（如 DNS、TCP 性能退化）

---

> **总结一句话**：  
> APEX 通过 **network-native pretraining + multivariate causal-chain modeling + edge-optimized design**，实现了高性能、低延迟、隐私友好的无线网络预测与异常检测一体化方案，为 AIOps 在企业边缘的落地提供了实用基础模型范式。

</details>

---

### 10. [Holding the FP8 Quality Ceiling at 8-Bit Weights and Activations: INT8 and GGUF Post-Training Quantization of Ideogram 4.0 for Consumer GPUs](https://arxiv.org/abs/2606.12280)

**Authors**: Deep Gandhi, Ali Asaria, Tony Salomone  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.12280v1  

#### Abstract
Post-training quantization lets large text-to-image diffusion transformers run on consumer GPUs, yet the hardware-specific trade-offs are seldom measured directly. We quantize Ideogram 4.0 - a 9.3B flow-matching diffusion transformer (DiT), shipped as two separate-weight copies of a single-stream 34...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Holding the FP8 Quality Ceiling at 8-Bit Weights and Activations: INT8 and GGUF Post-Training Quantization of Ideogram 4.0 for Consumer GPUs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大型文本到图像扩散模型（如 **Ideogram 4.0**）的原始权重无法在消费级 GPU 上高效运行。虽然已有 **FP8** 和 **NF4** 等量化版本发布，但在缺乏 FP8 张量核心的硬件（如 **Ampere 架构的 RTX 3090**）上，FP8 实际以 dequantize 到 bf16 运行，效率不高；而 NF4 虽然节省显存，却带来明显的生成质量下降，尤其在 **text rendering（文本渲染）** 方面。

本文旨在解决以下两个核心问题：
- 如何在 **无 FP8 硬件支持的消费级 GPU** 上实现高质量、低精度推理？
- 如何在保持 FP8 级别生成质量的同时，提升量化方案的 **实际可用性和效率潜力**？

### 提出的新方法与创新思路
作者提出了一套完整的 **INT8 W8A8 post-training quantization (PTQ)** 流程，并引入了 **GGUF Q4_K** 权重格式，主要创新如下：

#### ✅ 主要贡献
1. **INT8 W8A8 量化方案**  
   - 采用 **per-channel weights + per-token dynamic activations**，结合 **SmoothQuant (α=0.5)** 进行 outlier migration。
   - 引入 **mixed-precision protection**：保护高脆弱性的模块（主要是 FFN down-projections），其余层使用 INT8 W8A8。
   - 成功在 **8-bit weights and activations** 下“hold the FP8 quality ceiling”。

2. **首次对 diffusion transformer 进行 per-module fragility profiling**  
   - 通过分析各 layer 的 **max-abs, std, kurtosis**，构建 fragility score。
   - 发现 **FFN down-projections (feed_forward.w2)** 是最脆弱的组件，仅需保护前 17 层即可恢复几乎全部质量。

3. **GGUF k-quants 应用于 diffusion transformer**  
   - 首次将 **GGUF Q4_K**（4.5 bpw）应用于 text-to-image DiT 模型。
   - 自研 NumPy Q4K quantizer 与 Torch dequantization kernel，确保 bit-exact。

4. **引入 per-category OCR evaluation**  
   - 在 text rendering benchmark 上使用 **EasyOCR** 进行 **exact-match accuracy** 和 **normalized edit distance (NED)** 评估。
   - 这是该类模型中 **首次系统报告 OCR 表现**，揭示了文本保真度的真实情况。

### 相比现有方法的优势
| 维度 | 本方法优势 |
|------|-----------|
| **质量 vs. NF4** | INT8 显著优于 NF4：**+1.9 CLIP**（95% CI 不包含零），且 OCR 更优（NED ↓） |
| **质量 vs. FP8** | INT8 与 FP8 差异无统计显著性（paired bootstrap CI 包含零），即“hold the ceiling” |
| **存储 vs. NF4** | GGUF Q4_K 与 NF4 同等磁盘大小（~10.4 GB），但 **质量更高（+3.57 CLIP）**，为 **Pareto 最优解** |
| **实用性** | 所有量化模型已开源发布于 Hugging Face，支持 consumer GPU 部署 |

---

## 2. 核心实验方法和设置

### 使用的数据集与提示集
所有实验基于 **PartiPrompts** 构建三个互斥集合：
- **Calibration set**: n=128（用于 SmoothQuant 缩放因子校准）
- **Quality benchmark**: n=200（类别分层采样，不含文本提示）
- **Text-rendering benchmark**: n=100（其中 63 个含可验证 OCR 内容）

> 所有变体均使用相同 `(prompt, seed=1000, steps=48, resolution=1024×1024)` 设置生成图像。

### 实验设置
- **模型架构**：**Ideogram 4.0**，单流 34 层 DiT，flow-matching，双分支（conditional & unconditional），共 211×2 个 Linear 层。
- **量化目标**：仅量化 DiT 主干，**Qwen3-VL-8B encoder 和 VAE 保持原精度**。
- **参考基准**：**FP8 checkpoint**（dequantized to bf16），作为最高保真输出（因无公开 BF16 版本）。

### 评估指标
| 类型 | 指标 |
|------|------|
| **Standalone Quality** | PickScore, CLIPScore（自实现，避免跨论文偏差） |
| **Text Legibility** | OCR exact-match accuracy, Normalized Edit Distance (NED, 越低越好) |
| **Reference Fidelity** | PSNR, SSIM, LPIPS (AlexNet) 对比 FP8 输出 |
| **Efficiency** | end-to-end latency (s/image)，peak VRAM usage，on-disk size |

### 基线方法对比
- **FP8 (reference)**：官方发布的 FP8 版本（实际运行时 dequantize 到 bf16）
- **NF4**：官方发布的 NF4 权重（4-bit weight-only）
- **Q4_K (ours)**：自研 GGUF Q4_K 量化版本（4.5 bpw）
- **INT8 (ours)**：提出的 W8A8 + protection + SmoothQuant 方案

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Variant | Pick | CLIP | OCR NED | s/image | Size (GB) | Weights |
|--------|------|------|---------|--------|------------|----------|
| FP8 (ref) | 18.97 | 17.54 | 0.715 | 172.9 | 18.6 | 8-bit (deq.) |
| NF4 | 18.50 | 15.53 | 0.760 | 164.5 | ~10.4 | 4-bit W |
| **INT8 (ours)** | **18.96** | **17.42** | **0.704** | **184–185** | **18.6** | **8-bit W+A** |
| **Q4_K (ours)** | **19.46** | **19.10** | **0.62–0.73** | **203.3** | **10.4** | **4.5-bit W** |

> 注：Q4_K 的高质量源于其输出风格差异被 CLIP/Pick 偏好，而非更接近 FP8（见 reference fidelity）。

### 与基线对比结果
- **INT8 vs. FP8**：
  - Paired bootstrap 95% CI 在 Pick 和 CLIP 上均 **包含零** → 统计不可区分，成功“hold the ceiling”。
- **INT8 vs. NF4**：
  - CLIP ↑ **+1.90**（CI: [+1.21, +2.64]，排除零）→ 显著优于 NF4。
  - OCR NED ↓（0.704 vs. 0.760）→ 文本更清晰。
- **Q4_K vs. NF4**（同尺寸）：
  - CLIP ↑ **+3.57**，Pick ↑ +0.96，CI 均排除零 → **Pareto 最优**（同等 size，更高 quality）。
  - OCR NED 更低（0.62–0.73），表明文本保真更好。

### Reference Fidelity（vs. FP8 输出）
| Variant | PSNR | SSIM | LPIPS |
|--------|------|------|-------|
| INT8 | 21.42 | 0.722 | 0.306 |
| NF4 | 21.91 | 0.705 | 0.296 |
| Q4_K (slice) | 19.96 | 0.646 | 0.388 |

- **INT8 和 NF4 处于相似感知距离带**：NF4 略优 LPIPS，INT8 更高 SSIM。
- 两者都接近 PSNR > 21 的“匹配 16-bit”阈值。
- **关键洞察**：像素相似 ≠ 质量一致。NF4 偏离 FP8 的方式导致质量下降，而 INT8 偏离但仍保持高质量。

### 消融实验结果（Table 3）
在 50-prompt 子集上的 CLIP 对比：

| Configuration | CLIP |
|---------------|------|
| Naive W8A8 (no smoothing, no protection) | 14.30 |
| + SmoothQuant only (α=0.5) | 16.56 |
| + Protection only (top-17) | 16.81 |
| **+ Both (final recipe)** | **17.54** |
| Protection ladder N=8 | 16.01 |
| Protection ladder N=17 | 17.42 |

#### 发现：
- **Protection 是主导因素**：贡献约 **78–92% 的恢复能力**。
- **Sharp recovery knee**：从 N=8 到 N=17，CLIP 提升 **+1.41**，说明存在明显质变点。
- **Sub-additive effect**：SmoothQuant 与 protection 效果非线性叠加。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **INT8 W8A8 可以 hold 住 FP8 的质量天花板**，在 consumer GPU 上实现与 FP8 无统计差异的生成质量。
2. 🔍 **FFN down-projections 是量化中最脆弱的模块**，仅需保护约 8% 的层（17/211）即可恢复绝大部分质量。
3. 📊 **pixel-level fidelity 与 standalone quality 是两个独立轴**：
   - NF4 更接近 FP8 像素但质量更低；
   - INT8 像素略有偏离但质量持平。
4. 💾 **GGUF Q4_K 是当前最优权衡方案**：在与 NF4 相同磁盘占用下，提供更高生成质量，是真正的 **Pareto winner**。
5. ✍️ **OCR 是检测量化损伤的关键探针**：文本是最先退化的部分，必须单独评估。

### 方法的局限性
- **INT8 当前无计算加速**：由于缺少 fused INT8 GEMM kernel，INT8 推理速度反而比 NF4 慢 ~12%（184s vs. 164s）。
- **依赖特定硬件假设**：研究基于 RTX 3090（Ampere），不适用于拥有 FP8 支持的 newer architectures（如 Hopper）。
- **未覆盖多 seed 分析**：所有 headline 结果基于单一 seed（1000），缺乏跨 seed 泛化性验证。
- **Q4_K 推理较慢**：Torch dequant path 未优化，latency 达 203s/image。

### 未来工作方向
1. **开发 fused Ampere INT8 GEMM kernel**（如 CUTLASS/ViDiT-Q 风格）——这是将 INT8 从“质量 substrate”转化为“真实加速”的关键路径。
2. **扩展 step-caching 技术**（如 TACache）以进一步降低 latency。
3. **完整评估 Q8_0 的 reference fidelity** 并补充误差区间。
4. **量化 Qwen3-VL text encoder**，实现端到端低精度 pipeline。
5. **引入更强的文本评估器**（如 ImageReward/HPSv2）替代 CLIP-family scorer。

---

> **总结一句话**：  
> 本文证明，在 consumer GPU 上，通过 **INT8 W8A8 + selective bf16 protection** 可完美保留 Ideogram 4.0 的 FP8 生成质量，同时提出 **GGUF Q4_K** 作为更优的存储-质量折衷方案，为大模型本地部署提供了实用且高性能的量化路径。

</details>

---

### 11. [Automated Mediator for Human Negotiation: Pre-Mediation via a Structured LLM Pipeline](https://arxiv.org/abs/2606.11379)

**Authors**: Jamie Bergen, Sarit Kraus  
**Category**: cs.AI  
**Published**: 2026-06-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.11379v1  

#### Abstract
Pre-mediation, the preparatory phase preceding direct human negotiation, plays a critical role in achieving mutually beneficial agreements, yet is often omitted due to cost, time, and limited access to trained mediators. We introduce an automated mediator for human negotiation, implemented as a stru...

---

### 12. [SwiftCTS: Fast Cross-Design Prediction and Pareto Optimization of Clock Tree Metrics via Few-Shot Calibration](https://arxiv.org/abs/2606.11348)

**Authors**: Barsat Khadka, Kawsher Roxy, Md Rubel Ahmed  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.11348v1  

#### Abstract
Clock Tree Synthesis (CTS) is a computationally expensive stage in the physical design flow, requiring iterative EDA tool invocations to navigate a vast configuration space for optimal power, wirelength, and timing skew. Existing machine learning approaches require computationally expensive retraini...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SwiftCTS 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
Clock Tree Synthesis (CTS) 是物理设计流程中计算开销巨大的阶段，需要反复调用 EDA 工具在庞大的配置空间中搜索最优解，以优化 **power**、**wirelength** 和 **timing skew**。现有机器学习方法存在以下问题：
- **泛化能力差**：深度神经网络在未见过的 macro 架构上表现不佳，需昂贵的重训练或微调。
- **推理速度慢**：无法支持百万级组合搜索所需的快速评估。

### 提出的新方法与创新点
作者提出 **SwiftCTS**，一个基于物理先验的轻量级代理模型（surrogate framework），具备以下三大核心创新：

#### ● Agile Surrogate Architecture（敏捷代理架构）
- 采用 **gradient-boosted decision trees**（XGBoost/LightGBM）替代重型 DNN，实现：
  - **<5 秒 CPU 训练时间**
  - **<1 毫秒单次推理延迟**（无需 GPU）
- 支持从零开始快速重训练，避免依赖 GPU 资源。

#### ● K-Shot Multiplicative Calibration（K-shot 乘法校准机制）
- 引入一种新颖的 **few-shot 校准技术**，仅需 **1–2 次物理运行**（K=1）即可将预测锚定到新设计域。
- 通过几何平均计算全局缩放因子 $ k_{cal} $，消除跨设计偏移，显著提升 OOD 泛化能力。

#### ● Search-Based Pareto Optimization（基于搜索的帕累托优化）
- 将高速预测引擎集成至 **NSGA-II 多目标进化优化器**。
- 在 **<10 秒内评估 100,000 个 CTS 配置**，生成高质量 Pareto 前沿。
- 实现主动设计空间探索（DSE），而非被动预测。

### 相比现有方法的优势
| 维度 | 现有方法（如 GAN-CTS） | SwiftCTS |
|------|------------------------|---------|
| 推理速度 | 高延迟，难以大规模采样 | 子毫秒级，支持 10^5+ 规模搜索 |
| 泛化能力 | 严重依赖训练分布，OOD 性能骤降 | K-shot 校准实现强 OOD 泛化 |
| 可靠性 | 存在“False Success”风险 | 全面搜索确保找到真正更优解 |
| 硬件需求 | 依赖 GPU 加速 | 纯 CPU 运行，部署成本低 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **CTS-Bench** 生成共 **5,520 个 CTS 评估样本**，涵盖 6 个开源 IP 核：
  - **训练集（4 个）**：AES, PicoRV32, SHA-256, ETHMAC（共 5,400 runs）
  - **OOD 测试集（2 个）**：
    - JPEG Encoder（比最大训练设计大 5×）
    - ZipDiv Core（比最小训练设计小 21×）
- 所有实验基于 **Sky130 PDK** 和 **OpenROAD** 流程。

### 实验设置
#### 两阶段评估协议：
1. **Stage 1: Leave-One-Design-Out (LODO)**  
   - 每轮训练使用 3 个设计，测试第 4 个，验证跨设计泛化基础能力。
2. **Stage 2: Out-of-Distribution (OOD)**  
   - 在完整训练集上训练后，在完全未见的 JPEG 和 ZipDiv 上进行零样本测试。

#### CTS Knobs（4 个可调参数）：
- `cd`: Sink Max Diameter (35–70 μm)
- `cs`: Cluster Size (12–30 sinks)
- `mw`: Max Wire Length (130–280 μm)
- `bd`: Buffer Distance (70–150 μm)

### 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **MAPE** | Mean Absolute Percentage Error | Power & Wirelength（归一化尺度差异） |
| **MAE** | Mean Absolute Error (ns) | Skew（绝对时序误差） |
| **Pareto Frontier Quality** | 与默认工具设置对比的支配关系 | 最终 QoR 提升验证 |

### 基线方法对比
- **Random Search**
- **Sobol Sequence Sampling**
- **NSGA-II + SwiftCTS**（本文方法）
- 文献对比见 Table V：包括 [4] ANN, [5] CNN, [6] GAN-CTS

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table III）

#### Zero-Shot (K=0) 表现：
| 设计类别 | Power MAPE (%) | Wirelength MAPE (%) | Skew MAE (ns) |
|--------|----------------|---------------------|---------------|
| LODO 平均 | 24.5% | 10.3% | — |
| OOD JPEG 平均 | 31.2% | 9.4% | — |
| OOD ZipDiv 平均 | 8.4% | 56.6% | — |

> 注：ZipDiv 密度极高，导致 wirelength 零样本误差巨大。

#### K-Shot Calibration 效果（K=1）：
| 指标 | 改进效果 |
|------|--------|
| **Power MAPE ↓** | 从 **24.5% → 3.3%**（LODO） |
| **Wirelength MAPE ↓** | 从 **56.6% → 0.7%**（ZipDiv） |
| **Skew MAE ↓** | 控制在 **~0.11 ns**（base）或 **<5 ps**（ZipDiv） |

> ✅ **仅一次物理运行即实现近物理精度预测**

### 与基线方法对比（Table IV）

| Search Algorithm | Runtime (100K evals) | Convergence Profile |
|------------------|-----------------------|----------------------|
| Random Search | 604.3 s | Exhaustive Coverage |
| Sobol Sequence | 600.0 s | Exhaustive Coverage |
| **NSGA-II (SwiftCTS)** | **8.9 s** | Evolutionary (Fast) |

> ⚡️ **比传统方法快两个数量级**

### 消融实验结果（Table VI）
测试“跨 placement 校准”是否可行（即在一个 placement 上 calibrate 后广播到同 design 的其他 placement）：

| Calibration Level | Power MAPE (K=1) | Wirelength MAPE (K=1) |
|-------------------|------------------|------------------------|
| Per-Placement（本文） | 3.3% | 0.6% |
| Per-Architecture（跨 placement） | 13.6% ±8.2% | 11.4% ±6.4% |

> ❗ 结果表明：**clock tree metrics 高度依赖 floorplan 物理布局**，必须进行 placement-level 校准。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **SwiftCTS 实现了超高速、高精度的 CTS 指标预测**：
   - 训练 <5 秒，推理 <1ms，适合工业级 DSE。
2. ✅ **K-shot calibration 显著提升 OOD 泛化能力**：
   - 单次物理运行即可将预测误差降低一个数量级以上。
3. ✅ **集成进化算法可高效探索千万级配置空间**：
   - 100K 配置搜索仅需 ~9 秒，远超传统方法。
4. ✅ **闭环验证确认预测可靠性**：
   - OpenROAD 实际布线结果显示：
     - Power/WL 预测误差 **<0.5%**
     - Skew 预测误差 **<5 ps**
   - 所有设计均优于默认工具设置（Fig 4）。

### 方法的局限性
- 当前特征空间未考虑工业复杂约束，例如：
  - `dont-touch nets`
  - 局部 hold-only 优化
- 实验仅在 **OpenROAD + Sky130** 上验证，尚未覆盖先进工艺节点（如 7nm/5nm）及商业工具（Innovus/ICC2）。
- 虽然支持多目标优化，但未显式建模电压域（multi-Vt）、非均匀电源网格等现代设计要素。

### 未来工作方向
- 扩展至 **advanced technology nodes** 和 **commercial EDA suites**。
- 引入对 **multi-Vt design**、**power gating**、**localized optimization** 的建模能力。
- 探索将 SwiftCTS 集成至全流程自动化 Agent（如 LLM-based EDA agent）中，构建端到端自主设计系统。

> 🔗 开源代码地址：[https://github.com/BarsatKhadka/SwiftCTS](https://github.com/BarsatKhadka/SwiftCTS)

</details>

---

### 13. [Organize then Retrieve: Hierarchical Memory Navigation for Efficient Agents](https://arxiv.org/abs/2606.11680)

**Authors**: Hao-Lun Hsu, Nikki Lijing Kuang, Boyi Liu, Zhewei Yao, Yuxiong He  
**Category**: cs.AI  
**Published**: 2026-06-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.11680v1  

#### Abstract
Large language model (LLM) agents struggle with long-horizon tasks due to their inherent statelessness, requiring all task-relevant information to be encoded in growing input contexts. The resulting degraded reasoning quality, increased inference cost, and higher latency necessitate efficient workin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Organize then Retrieve: Hierarchical Memory Navigation for Efficient Agents 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）代理在处理长时程任务时面临“状态无记忆”（stateless）的根本缺陷。所有相关任务信息都必须编码在不断增长的输入上下文中，这导致了三个核心问题：
*   **推理质量下降**：过长的上下文会稀释关键信号，导致模型“迷失在中间”（lost-in-the-middle）。
*   **推理成本高昂**：处理超长序列需要巨大的计算资源和时间。
*   **信息丢失风险**：现有的压缩机制（如摘要）是不可逆的，会永久丢失下游推理所需的细粒度信息。

### 提出的新方法和新思路
本文提出了 **HORMA (Hierarchical Organize-and-Retrieve Memory Agent)**，其核心创新在于**将工作记忆（working memory）解耦为两个独立且功能不同的模块**：

1.  **分层组织（Structured Memory Construction）**：
    *   将原始交互轨迹（raw trajectories）组织成一个类似文件系统的**分层结构**。
    *   通过 `mkdir`, `nano` 等 Bash 工具创建结构化的笔记（structured notes），这些笔记包含摘要实体，并链接到对应的原始轨迹，确保信息不丢失。
    *   该过程被视为一种持续的**技能获取（skill acquisition）**，通过对比成功与失败的轨迹来迭代优化记忆管理策略。

2.  **基于导航的检索（Navigation-based Retrieval）**：
    *   引入一个轻量级的、经过强化学习（RL）训练的**检索代理（Retrieval Agent）**。
    *   该代理通过执行 `ls`, `grep`, `cd`, `cat` 等 Bash 命令，在分层内存树中主动**导航**，以找到最相关的上下文。
    *   为了优化检索，引入了一个**证据导向的奖励信号（evidence-grounded retrieval reward）**，该信号基于检索到的上下文与真实证据的重叠度（Jaccard Index）来提供直接反馈，从而解耦了检索质量与下游推理性能的关联。

### 相比现有方法的优势
*   **效率与性能的帕累托最优**：HORMA 在任务成功率和上下文消耗之间取得了更好的平衡，实现了更高的效率-性能权衡。
*   **保留细粒度信息**：通过将摘要与原始轨迹链接，避免了传统压缩方法的信息丢失问题。
*   **更准确的检索**：利用显式的结构信号（目录层次、时间顺序、来源追踪）进行导航，相比基于语义相似性的检索，能获得更符合时间逻辑和因果依赖的信息。
*   **更强的泛化能力**：学习到的轻量级检索代理展现出强大的跨域零样本迁移能力（out-of-distribution generalization）。

## 2. 核心实验方法和设置

### 使用的数据集
论文在三个具有挑战性的长时程基准上进行了评估：
*   **ALFWorld**: 一个基于文本的具身交互环境，用于测试物理世界中的规划和操作任务。
*   **LoCoMo**: 一个长对话基准，包含10个长篇多轮对话，用于测试从扩展对话历史中构建记忆的能力。
*   **LongMemEval**: 另一个长对话基准，特别设计用于评估聊天助手在长期互动中的记忆能力，强调跨领域的零样本泛化。

### 实验设置和评估指标
*   **上下文限制**：在 ALFWorld 上设置了两个上下文窗口大小（1950 和 2200 tokens）；在 LoCoMo 和 LongMemEval 上分别设置了 10K 和 50K 的 token 上下文预算，以模拟严格的内存限制。
*   **评估指标**：
    *   **ALFWorld**：任务成功率（Success Rate）、平均交互步数（Average Steps per Task）、每步平均输入 token 数（Average Input Tokens per Step）。
    *   **LoCoMo & LongMemEval**：LLM-as-a-judge (L-J) 评分（使用 Claude Sonnet 4.5 作为裁判）、总输入 token 消耗量（Total Input Token Usage）。

### 基线方法对比
与以下代表性方法进行了比较：
*   **静态内存方法**：截断（Truncation）、滑动窗口（Slide Window）、基于嵌入的相似性检索（Embedding）。
*   **动态内存方法**：ReSum（摘要压缩）、Acon（上下文优化）、Fold（上下文折叠）。
*   **外部动态内存系统**：A-MEM、Mem0。

## 3. 主要实验结果和性能指标

### 关键性能数据
*   **ALFWorld**：在小上下文（1950 tokens）和大上下文（2200 tokens）下，HORMA 的任务成功率分别达到 **56.7%** 和 **73.9%**，均优于所有基线方法。
*   **长对话任务**：在 LoCoMo 和 LongMemEval 上，HORMA 的 token 消耗量仅为不同基线方法的 **3.07%-22.17%** 和 **1.24%-16.19%**，极大地提升了效率。

### 与基线方法的对比结果
*   **效率-性能权衡**：如图 2a 所示，HORMA 在 ALFWorld 上始终位于帕累托前沿，即在更低的 token 消耗下实现了更高的成功率。
*   **上下文消耗**：如图 2b 所示，所有 HORMA 变体在长对话任务中都能将每查询的 token 消耗控制在 1000 以内，而许多基线方法的消耗远超此限。
*   **零样本泛化**：在未见过的 LongMemEval 任务上，HORMA 表现最佳，证明了其强大的泛化能力。

### 消融实验结果
*   **技能演进（Skill Evolution）**：移除技能演进后，性能有所下降，证明了通过对比分析迭代优化记忆管理策略的有效性。
*   **强化学习检索（RL-based Retrieval）**：使用经过 RL 训练的轻量级检索器（基于 Qwen 3.5 4B）显著提升了性能和检索效率，减少了不必要的 LLM 调用次数。
*   **跨骨干模型性能**：当使用较小的 LLM（如 Qwen 3.5 4B）作为记忆管理器时，即使使用强大的检索器（Claude Sonnet 4.5），性能提升也有限，这表明**高质量的记忆组织是有效检索的前提**。

## 4. 关键结论和发现

### 主要发现
1.  **解耦是关键**：将记忆的**组织**（异步、长期影响）和**检索**（同步、直接影响推理）解耦，可以显著改善信用分配（credit assignment），并实现更高效的优化。
2.  **分层结构优于扁平存储**：将经验组织成文件系统般的分层结构，能够更好地捕捉时间层级和因果依赖，从而支持更精确的检索。
3.  **导航优于匹配**：让检索代理通过可执行命令主动导航内存树，比被动地进行语义相似性匹配更有效，尤其是在处理长时程任务时。
4.  **非参数技能演进**：通过对比分析来迭代增强记忆管理提示（prompt），是一种有效的、无需更新主模型参数即可持续适应新任务的方法。

### 方法的局限性
*   **对记忆管理者的要求高**：实验表明，如果记忆管理者本身不具备强大的语义抽象和分层推理能力，整个系统的性能上限会受到严重制约。
*   **依赖工具执行**：方法的成功依赖于代理正确执行 Bash 命令的能力，命令执行错误可能导致检索失败。

### 未来工作方向
*   将当前基于证据的检索训练框架扩展到完全在线的、由交互驱动的学习模式。
*   探索如何进一步自动化和优化技能库的构建过程。
*   将 HORMA 的模块化设计应用于更广泛的智能体任务中。

</details>

---

### 14. [MODF-SIR: A Multi-agent Omni-modal Distilled Framework for Social Intelligence Reasoning](https://arxiv.org/abs/2606.12018)

**Authors**: Shang Ma, Jisheng Dang, Wencan Zhang, Yifan Zhang, Bimei Wang, Hong Peng, Bin Hu, Qi Tian, Tat-Seng Chua  
**Category**: cs.AI  
**Published**: 2026-06-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.12018v1  

#### Abstract
We propose a multi-agent collaborative framework built upon a lightweight Multimodal Large Language Model (MLLM), specifically designed for social intelligence reasoning. A key feature of our approach is that both the training and inference phases are augmented via knowledge distillation. Within thi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MODF-SIR: A Multi-agent Omni-modal Distilled Framework for Social Intelligence Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**社会智能推理（Social Intelligence Reasoning）**这一复杂任务，旨在让多模态大语言模型（MLLM）能够理解人类意图、情绪、人际关系及隐含的社会规范。这类任务面临以下挑战：
- 多模态信号（语音、表情、动作等）高度耦合且动态演化；
- 关键信息常以**长尾事件（long-tail events）**形式出现（如微表情、语气变化），易被主流事件掩盖；
- 传统端到端黑箱推理易产生“认知过载”和错误级联（error cascade）。

### 🚀 提出的新方法与创新思路
作者提出 **MODF-SIR**（Multi-agent Omni-modal Distilled Framework for Social Intelligence Reasoning），一个基于轻量级 MLLM 的**多智能体协作框架**，其核心创新包括：

#### （1）**双阶段检索 + 显式文本化策略**
- 第一阶段由 **ELT Retriever Agent** 快速扫描输入，提取潜在长尾线索；
- 第二阶段由 **OMLT Reasoner Agent** 进行精细分析；
- 所有中间结果均**序列化为自然语言文本**，防止关键长尾信息在 tokenization 中丢失。

#### （2）**动态路由机制（AKD Router Agent）**
- 受“双系统理论”（Dual-Process Theory）启发，引入 **Asymmetric Knowledge Distilled (AKD) Router Agent**；
- 根据查询显隐性和事件分布（head vs. long-tail），动态选择是否执行时空定位（temporal grounding）；
- 实现计算资源的最优分配。

#### （3）**精准局部化模块（GRPO Grounder Agent）**
- 针对长尾事件难以全局搜索的问题，设计 **GRPO Grounder Agent** 对视频片段进行高精度时序定位；
- 使用 **GRPO 算法**优化 IoU 指标，提升定位准确性。

#### （4）**测试时自适应（Test-Time Adaptation, TTA）与自我修正机制**
- 引入 **TTA Reviser Agent** 构建闭环反馈系统；
- 利用 LLM “评价能力 > 生成能力”的特性（generation-evaluation gap），通过外部教师模型打分并驱动 **LoRA 微调**；
- 支持迭代式反思与重生成，显著减少幻觉。

#### （5）**知识蒸馏增强训练**
- 在训练和推理中全面应用知识蒸馏；
- 使用大规模教师模型（如 Qwen3-Omni-30B）生成高质量伪标签，训练小型学生模型（如 Qwen2.5-Omni-7B），实现认知压缩。

### 🔍 相比现有方法的优势
| 维度 | MODF-SIR 的优势 |
|------|------------------|
| **架构设计** | 多智能体协同，打破黑箱推理，提高可解释性 |
| **长尾事件处理** | 文本化 + 局部化 + 动态路由，有效捕捉细微信号 |
| **推理鲁棒性** | TTA + 自我反思机制抑制错误传播 |
| **效率与规模平衡** | 轻量模型 + LoRA + 知识蒸馏，实现高性能低开销 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
论文在三个公开基准上进行了广泛评估：
| 数据集 | 描述 |
|-------|------|
| **Daily-Omni [10]** | 包含日常场景下的音视频问答，强调跨模态时间对齐与上下文理解 |
| **IntentBench [11]** | 专注于人类意图识别，涵盖 Why / How / When / Who 等问题类型 |
| **WorldSense [9]** | 测试世界知识与领域理解能力，覆盖科技、文化、体育等多个类别 |

### ⚙️ 实验设置与评估指标
- **模型大小统一为 7B 参数级别**，确保公平比较；
- 主要采用 **Accuracy (%)** 作为主指标；
- 在 IntentBench 上按问题类型细分评估（Why, How, When, Who/Which, Other）；
- 使用 **LoRA** 进行参数高效微调，仅更新低秩矩阵；
- 推理过程中启用 **TTA 最大迭代次数控制**（max iterations）；
- 所有 LoRA 权重在推理后丢弃，避免灾难性遗忘（catastrophic forgetting）。

### 🆚 基线方法对比
#### 开源模型（Open-Source Video-Audio MLLMs）：
- Unified-IO-2 [45]
- VideoLLaMA2 [46]
- Qwen2.5-Omni [31]
- Ola [47]
- MiniCPM-o [49]
- HumanOmniV2 [11]（强基线）

#### 商业闭源模型（Proprietary MLLMs）：
- GPT-4o [12]
- GPT-o1 [48]
- Gemini-2.5-Pro (think) [13]
- Claude 3.5 Sonnet [51]

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### 表 I：Daily-Omni [10] 结果对比（平均 Accuracy）
| 方法 | 模型大小 | 平均得分 |
|------|----------|---------|
| Gemini 2.0 Flash | — | 67.8 |
| HumanOmniV2 [11] | 7B | 58.5 |
| **MODF-SIR (Ours)** | **7B** | **64.9** |

> ✅ **超越所有开源模型**，接近商业模型 Gemini 2.0 Flash。

#### 表 II：IntentBench [11] 结果对比（平均 Accuracy）
| 方法 | 模型大小 | 平均得分 |
|------|----------|---------|
| GPT-4o | — | 60.0 |
| GPT-o1 | — | 66.7 |
| Gemini-2.5-Pro (think) | — | 67.2 |
| HumanOmniV2 [11] | 7B | 69.3 |
| **MODF-SIR (Ours)** | **7B** | **70.3** |

> ✅ **达到甚至略微超越最强开源基线**，逼近 GPT-o1 和 Gemini。

#### 表 III：WorldSense [9] 结果对比（平均 Accuracy）
| 方法 | 模型大小 | 平均得分 |
|------|----------|---------|
| GPT-4o | — | 42.6 |
| Gemini 1.5 Pro | — | 48.0 |
| HumanOmniV2 [11] | 7B | 47.1 |
| **MODF-SIR (Ours)** | **7B** | **51.5** |

> ✅ **显著领先开源模型**，超过 Gemini 1.5 Pro，接近顶尖水平。

---

### 🔬 消融实验结果（Ablation Study）

#### 表 IV：各模块消融效果（Daily-Omni 上的 Avg Score）

| 方法配置 | 平均得分 |
|--------|--------|
| HumanOmniV2 (Baseline) | 58.5 |
| + TTA Reviser + OMLT Reasoner | 64.0 |
| + GRPO Grounder（无 Router） | 58.6 ↓ |
| + GRPO Grounder + AKD Router | 59.4 |
| 完整 MODF-SIR | **64.9** |

#### 关键发现：
1. **TTA Reviser 贡献最大**：单独加入即可从 58.5 → 64.0，验证了测试时自适应的有效性；
2. **AKD Router 至关重要**：盲目使用 Grounder 反而降低性能（58.6），说明动态路由能避免噪声干扰；
3. **ELT Retriever 是抽象基础**：缺少它会导致后续模块输入质量下降；
4. **GRPO Grounder 优于传统方法**：相比 VideoMind Grounder 提升明显（58.6 vs 57.3）；

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **多智能体协作显著提升社会智能推理能力**：将感知、路由、定位、推理、反思解耦，使每个模块专注特定子任务，大幅提升整体表现。
2. **长尾事件必须显式建模**：通过文本化和局部化手段，成功放大微弱但关键的社会信号（如眼神、语气），避免被主导事件淹没。
3. **TTA + LoRA 是有效的推理优化范式**：利用 generation-evaluation gap 实现自我纠正，是缓解幻觉和逻辑不一致的重要路径。
4. **知识蒸馏可用于构建高质量路由决策器**：即使小模型也能通过蒸馏获得接近大模型的战略规划能力。

### ⚠️ 方法的局限性
1. **依赖高质量教师模型生成伪标签**：若教师模型存在偏见或错误，会传导至学生模型；
2. **推理延迟较高**：由于多轮 TTA 和 Agent 协作，实时性不如单次前向推理模型；
3. **当前主要面向视频+音频模态**：对更多模态（如触觉、生理信号）的支持尚未探索；
4. **GRPO 训练需要标注边界数据**：限制了其在完全无监督场景中的应用。

### 🔮 未来工作方向
1. **构建专用视频镜头分割 Agent**：结合 CV 中的 shot boundary detection 技术，实现更细粒度的时间语义划分；
2. **跨镜头因果推理**：利用 LLM 分析不同镜头间的语义关联，建立长期因果链；
3. **扩展至真实人机交互场景**：部署于机器人、虚拟助手等实际系统中；
4. **探索完全自监督的路由学习机制**：减少对人工设计 prompt 和教师模型的依赖。

---

> 🔗 **代码与资源**  
> - GitHub: [https://github.com/eeee-sys/MODF-SIR](https://github.com/eeee-sys/MODF-SIR)  
> - Demo: [https://huggingface.co/spaces/Harry-1234/MODF-SIR](https://huggingface.co/spaces/Harry-1234/MODF-SIR)  
> - LoRA 权重: [https://huggingface.co/Harry-1234/MODF-SIR](https://huggingface.co/Harry-1234/MODF-SIR)  
> - 路由器训练数据: [https://huggingface.co/datasets/Harry-1234/IntentRouterTrain](https://huggingface.co/datasets/Harry-1234/IntentRouterTrain)

</details>

---

### 15. [GraspLLM: Towards Zero-Shot Generalization on Text-Attributed Graphs with LLMs](https://arxiv.org/abs/2606.11898)

**Authors**: Hengyi Feng, Zeang Sheng, Meiyi Qiang, Meiyi Qiang, Wentao Zhang  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.11898v1  

#### Abstract
Research on Text-Attributed Graphs (TAGs) has gained significant attention recently due to its broad applications across various real-world data scenarios, such as citation networks, e-commerce platforms, social media, and web pages. Inspired by the remarkable semantic understanding ability of Large...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# GraspLLM: Towards Zero-Shot Generalization on Text-Attributed Graphs with LLMs —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

当前将 **Large Language Models (LLMs)** 应用于 **Text-Attributed Graphs (TAGs)** 的方法在**跨数据集**（cross-dataset）和**跨任务**（cross-task）的零样本泛化能力上存在显著瓶颈，主要体现在两个方面：

- **C1: 特征泛化受限**（Limited generalizability on features）  
  不同TAG数据集的节点文本特征分布差异大（如学术 vs 电商），导致模型难以迁移。

- **C2: 结构理解不足**（Limited generalizability on graph structures）  
  现有方法对图结构的理解浅层（如仅用1-hop邻居），缺乏对复杂拓扑模式（如motifs）的建模，限制了LLM的推理能力。

---

### 🚀 **提出了什么新方法或新思路**

作者提出 **GraspLLM**，一个**骨干无关**（backbone-agnostic）的框架，通过以下三个核心设计实现强零样本泛化：

#### （1）统一语义编码（Unified Semantic Encoding）
- 使用冻结的 **Qwen3-Embedding-8B** 将所有图的节点文本映射到统一的语义空间，缓解跨域特征分布偏移。

#### （2）**Motif-Aware 自监督学习**（Structural Information Extraction）
- 在多个 **motif-induced 邻接矩阵**（edge, triangle, 4-cycle, 4-clique）上进行对比学习，提取**数据集无关**（dataset-agnostic）的结构先验。
- GNN作为“结构提取器”，不依赖任务特定训练头。

#### （3）**最优上下文子图采样 + 对齐调优**（Optimal Contextual Subgraph Sampling & Alignment Tuning）
- 提出 **Contextual Relevance** 函数，指导贪婪算法采样最具语义与结构相关性的子图。
- 设计 **alignment projector**（MLP + cross-attention pooling）将子图嵌入对齐至LLM的token空间。
- 冻结LLM，仅优化projector，提升泛化性。

---

### 🔍 **相比现有方法的优势**

| 维度 | 传统方法缺陷 | GraspLLM优势 |
|------|--------------|-------------|
| **泛化能力** | 依赖任务微调，跨域/跨任务性能骤降 | 在**零样本**下实现SOTA，尤其在跨域场景 |
| **结构理解** | 局部邻域或单节点表示，忽略全局模式 | 利用motif捕捉高阶结构，支持远距离推理（可达10跳） |
| **可扩展性** | 难以处理百万级节点 | 工程优化支持OGBN-Products（245万节点） |
| **鲁棒性** | 对prompt敏感 | 在不同prompt格式下仍保持竞争力 |

---

## 2. 核心实验方法和设置

### 📚 **使用的数据集**

共 **14个真实世界TAG数据集**，覆盖5大领域：

| 类别 | 数据集 |
|------|--------|
| **Citation** | Cora, Citeseer, Pubmed, Arxiv |
| **E-commerce** | Books-History, Ele-Computer, Ele-Photo |
| **Social** | Reddit, Instagram |
| **Web pages** | Cornell, Texas, Washington, Wisconsin |
| **Wikipedia** | WikiCS |

> 数据规模从187到169,343节点，平均度数2.65–36.94，聚类系数跨度达两个数量级，体现高度结构异质性。

---

### ⚙️ **实验设置与评估指标**

#### **任务类型**
- **Node Classification**（零样本 & 全监督）
- **Link Prediction**（跨任务零样本）

#### **评估范式**
- **In-domain Zero-shot**：在某领域训练，在同一领域其他数据集测试。
- **Cross-domain Zero-shot**：在Arxiv/Computer/Reddit训练，迁移到其他领域测试。
- **Cross-task Zero-shot**：在分类任务上训练，直接用于链接预测。

#### **评估指标**
- **Node Classification**：Accuracy
- **Link Prediction**：AUC

#### **LLM Backbone**
- 主干模型：**Vicuna-7B-v1.5**, Mistral-7B, LLaMA-3.1-8B, Qwen3-8B
- LLM冻结，仅训练alignment projector。

---

### 🆚 **基线方法对比**

涵盖7类共20+基线：

| 类别 | 代表方法 |
|------|--------|
| **MLP/GNN** | GCN, GAT, GraphSAGE |
| **Graph Transformer** | NodeFormer, DIFFormer |
| **Small LMs** | BERT, RoBERTa, E5, Sentence-BERT |
| **纯LLM** | Qwen2-7B, LLaMA-2-7B |
| **LLM as Enhancer** | OFA, ZeroG, GraphCLIP, LLM-BP |
| **LLM as Predictor** | GraphGPT, LLaGA, TEA-GLM, GOFA |

---

## 3. 主要实验结果和性能指标

### 📊 **关键性能数据**

#### ✅ **In-domain Zero-shot Node Classification**（Table II）
- GraspLLM在10个未见数据集上平均准确率达 **74.0%**（Vicuna-7B），显著优于最佳基线TEA-GLM（62.1%）。
- 在History、WikiCS、Webpage等数据集上领先超 **0.1**。

#### ✅ **Cross-domain Zero-shot**（Table III）
- 平均性能下降小于0.15，部分转移甚至**反超**in-domain结果（如Pubmed ← Reddit）。
- 在最困难的跨域迁移中（如Arxiv → History），仍保持 **0.640** 准确率，而TEA-GLM仅 **0.205**。

#### ✅ **Cross-task Zero-shot Link Prediction**（Table IV）
- 在10个数据集上AUC平均提升 **>0.1**，最高达 **0.751**（Qwen3-8B on Pubmed）。
- 显著优于GraphGPT、LLaGA等专为图设计的方法。

#### ✅ **Supervised Performance**（Table VI）
- 在全监督设置下仍达到SOTA，平均准确率 **83.5%**（Qwen3-8B），证明其**兼顾零样本与监督学习**的双重优势。

---

### 🔪 **消融实验结果**（Table V）

| 变体 | Node Classification ↓ | Link Prediction ↓ |
|------|------------------------|--------------------|
| **w/o Structural Extractor** | 最多↓0.14（WikiCS） | 最多↓0.13（Instagram） |
| **w/o Alignment Projector** | 平均↓0.05 | 平均↓0.03 |

> 结论：**motif-aware GNN** 和 **alignment projector** 均为关键组件，前者贡献更大。

---

## 4. 关键结论和发现

### 🧠 **主要发现**

1. **GraspLLM实现了真正的零样本泛化**  
   - 在**跨域、跨任务**设置下均显著优于现有方法，验证了其强大的迁移能力。

2. **Motif-Aware 学习有效捕获通用结构模式**  
   - 通过motif对比学习，GNN能提取与具体任务无关的结构先验，成为LLM的“结构感知器”。

3. **最优上下文子图是LLM图推理的关键输入**  
   - 相比随机游走或固定邻域，该策略能筛选出语义与结构双相关的节点，提升LLM推理质量。

4. **无需微调LLM即可获得强大性能**  
   - 冻结LLM + 轻量projector的设计避免过拟合，增强泛化。

---

### ⚠️ **方法的局限性**

1. **对结构分布差异敏感**  
   - 若源域与目标域拓扑统计差异过大（如社交图→引文图），性能仍有明显下降。

2. **标签-结构弱相关场景效果受限**  
   - 如Instagram二分类任务中，邻居信息与标签关联弱，子图增益较小。

3. **当前仅支持节点/边级任务**  
   - 未拓展至图分类、社区检测等更复杂任务。

4. **仍依赖线性化子图序列**  
   - 子图通过pooling压缩为序列，存在信息损失风险。

---

### 🔮 **未来工作方向**

1. **更智能的图Tokenization**  
   - 设计**结构感知的tokenization机制**，减少线性化过程中的信息丢失。

2. **从静态输入到动态探索**  
   - 构建**LLM-agent可交互的图轨迹**（如goal-conditioned walks），支持多步推理。

3. **拓展至异构图、时序图、符号图**  
   - 当前仅限于同构、静态、单文本属性图。

4. **构建图-语言联合预训练框架**  
   - 探索类似“Graph-CLIP”的统一预训练范式，进一步提升泛化边界。

---

> ✅ **总结一句话**：  
> **GraspLLM通过“统一编码 + motif结构提取 + 上下文子图对齐”三重设计，首次实现了LLM在Text-Attributed Graph上的强零样本泛化，兼具高性能、高效率与高可扩展性，为图与语言模型融合提供了新范式。**

</details>

---

### 16. [Harnessing Routing Foresight for Micro-step-level MoE load balancing in RL Post-training](https://arxiv.org/abs/2606.11867)

**Authors**: Yuming Zhou, Haoyang Li, Sheng Lin, Yanfeng Zhao, Tong Zhao, Xupeng Miao, Jie Jiang, Fangcheng Fu, Bin Cui  
**Category**: cs.DC  
**Published**: 2026-06-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.11867v1  

#### Abstract
Mixture-of-Experts (MoE) and reinforcement learning (RL) post-training now dominate large language model (LLM) development, yet expert load imbalance remains a critical challenge. Existing load-balancing systems target pre-training by relying on historical step-level statistics. However, these metho...

---

### 17. [Range-Aware Bayesian Optimization for Discovering Diverse Designs within Target Property Windows](https://arxiv.org/abs/2606.11574)

**Authors**: Shengli Jiang, Jason Wu, Charles M. Schroeder, Michael A. Webb  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.11574v1  

#### Abstract
In many materials and product design problems, desirable candidates exhibit properties that fall within an acceptable range rather than achieve a single optimum. Recovering multiple, distinct solutions that satisfy such specifications is also practically valuable, as some candidates may be preferred...

---

### 18. [DeMix: Debugging Training Data with Mixed Data Error Types by Investigating Influence Vectors](https://arxiv.org/abs/2606.11616)

**Authors**: Jiale Deng, Yanyan Shen, Xiaogang Shi, Chai Junjun  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.11616v1  

#### Abstract
High-quality training data is essential for the success of machine learning models. However, real-world datasets often contain mixed types of errors arising from systematic flaws in data preparation pipelines, including label errors, feature errors, and spurious correlations. Effective debugging of ...

---

### 19. [SVoT: State-aware Visualization-of-Thought for Spatial Reasoning via Reinforcement Learning](https://arxiv.org/abs/2606.11770)

**Authors**: Chao Lei, Yanbei Jiang, Markus Hiller, Zhijian Zhou, Xunye Tian, Krista A. Ehinger, Nir Lipovetzky  
**Category**: cs.AI  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.11770v1  

#### Abstract
Spatial reasoning remains a challenge for Multimodal Large Language Models (MLLMs), as it requires reliable multi-hop inference over both intermediate states and state transitions. Current studies often leave intermediate states unverified and treat state transitions as implicit processes, which lim...

---

### 20. [One Jailbreak, Many Tongues: Learning Language-Insensitive Intention Representations for Multilingual Jailbreak Detection](https://arxiv.org/abs/2606.11202)

**Authors**: Shuyu Jiang, Kaiyu Xu, Xingshu Chen, Hao Ren, Rui Tang, Yi Zhang, Tianwei Zhang, Hongwei Li  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.11202v1  

#### Abstract
Large language models (LLMs) are increasingly deployed in applications for global multilingual users, yet safety training remains concentrated in dominant languages and has not progressed in parallel with multilingual capability, creating exploitable gaps for jailbreak attacks. Current jailbreak def...

---

### 21. [Benchmarking Large Language Models for Safety Data Extraction](https://arxiv.org/abs/2606.11204)

**Authors**: Jonas Grill, Thomas Bayer, S\"oren Berlinger  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.11204v1  

#### Abstract
Accurate extraction of structured information from Safety Data Sheets (SDS) remains challenging in industrial safety due to heterogeneous document formats and the limitations of traditional rule-based methods. This study benchmarks state-of-the-art Large Language Models (LLMs) for automated SDS data...

---

### 22. [uva-irlab-conv at SemEval-2026 Task 8: Multi-Turn RAG with Learned Sparse Retrieval and Listwise Reranking](https://arxiv.org/abs/2606.11945)

**Authors**: Simon Lupart, Kidist Amde Mekonnen, Zahra Abbasiantaeb, Mohammad Aliannejadi  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.11945v1  

#### Abstract
This report describes our participation in SemEval-2026 Task 8 on multi-turn retrieval and question answering. The task evaluates conversational systems across four domains (finance, cloud documentation, government, Wikipedia), and includes unanswerable queries where the available collection does no...

---

### 23. [Agreement in Representation Space for Open-Ended Self-Consistency](https://arxiv.org/abs/2606.12003)

**Authors**: Paula Ontalvilla, Gorka Azkune, Aitor Ormazabal  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.12003v1  

#### Abstract
Self-consistency improves LLM reasoning by sampling multiple outputs and selecting the most consistent answer, but existing formulations largely rely on exact matching and therefore remain limited to tasks with categorical outputs. In this work, we study self-consistency in open-ended generation tas...

---

### 24. [A Controlled Study of Decoding-Time Truthfulness Methods on Instruction-Tuned LLMs](https://arxiv.org/abs/2606.12160)

**Authors**: Ao Sun  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.12160v1  

#### Abstract
In this work, we introduce CHAIR (Classifier of Hallucination As ImproveR), a supervised framework for detecting hallucinations by analyzing internal logits from each layer of every token. Our method extracts a compact set of features such as maximum, minimum, mean, standard deviation, and slope-fro...

---

### 25. [Adaptive Multi-Resolution Procedural Knowledge Compression for Large Language Models](https://arxiv.org/abs/2606.12203)

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yichen Tang, Runzhong Qiao, Xuancheng Li, Min Zhang, Yiqun Liu  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.12203v1  

#### Abstract
Large language models (LLMs) are widely used to tackle complex tasks with autonomous workflows. Recently, reusable natural language skills have emerged as a popular paradigm to inject procedural knowledge into LLM applications. Since popular skills are often invoked repeatedly, placing their full te...

---

### 26. [Can News Predict the Market? Limits of Zero-Shot Financial NLP and the Role of Explainable AI](https://arxiv.org/abs/2606.12210)

**Authors**: Ali M Karaoglu, Shreyank N Gowda  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.12210v1  

#### Abstract
Can financial news reliably predict short-term stock movements? Despite advances in large language models, this question remains unresolved. We revisit this problem using a zero-shot natural language processing framework, investigating whether models can extract actionable signals from financial new...

---

### 27. [Context-Driven Incremental Compression for Multi-Turn Dialogue Generation](https://arxiv.org/abs/2606.12411)

**Authors**: Yeongseo Jung, Jaehyeok Kim, Eunseo Jung, Jiachuan Wang, Yongqi Zhang, Ka Chun Cheung, Simon See, Lei Chen  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.12411v1  

#### Abstract
Modern conversational agents condition on an ever-growing dialogue history at each turn, incurring redundant attention and encoding costs that grow with conversation length. Naive truncation or summarization degrades fidelity, while existing context compressors lack cross-turn memory sharing or revi...

---

### 28. [GLACIER: A Multimodal Student-Teacher Foundation Model for Molecular Property Prediction](https://arxiv.org/abs/2606.11382)

**Authors**: Emily Nguyen, Yongchan Hong, Harsh Toshniwal, Yan Liu, Andreas Luttens  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.11382v1  

#### Abstract
Deep learning models facilitate the discovery of molecules with tailored properties among billions of candidate compounds. However, the computational burden to develop and deploy state-of-the-art models continuously increases, limiting their scalability. Most large-scale models are unimodal in natur...

---

### 29. [Attention by Synchronization in Coupled Oscillator Networks](https://arxiv.org/abs/2606.12059)

**Authors**: Fabio Pasqualetti, Taosha Guo  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.12059v1  

#### Abstract
We address transformer attention on energy-constrained physical substrates. Softmax attention requires exponentiation and global reduction, operations with high energy cost on von Neumann hardware and no natural physical analog. We show that Kuramoto synchronization dynamics (which arise in electric...

---

### 30. [A Riemannian Approach to Low-Rank Optimal Transport](https://arxiv.org/abs/2606.12120)

**Authors**: Pratik Jawanpuria, Bamdev Mishra  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.12120v1  

#### Abstract
Low-rank optimal transport (OT) mitigates the quadratic scaling of classical solvers, yet existing approaches rely heavily on first-order mirror-descent updates that require careful hyperparameter tuning and ignore the optimization landscape's curvature. To address these limitations, we propose a un...

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

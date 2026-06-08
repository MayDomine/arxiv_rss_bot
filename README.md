# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-08 10:14:20 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Uncertainty-Aware LLM-Guided Policy Shaping for Sparse-Reward Reinforcement Learning](https://arxiv.org/abs/2606.06673)

**Authors**: Ujjwal Bhatta, Utsabi Dangol, Sumaly Bajracharya, Rodrigue Rizk, KC Santosh  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.06673v1  

#### Abstract
Sparse rewards and heterogeneous task sequences remain persistent challenges in Reinforcement Learning (RL), often resulting in slow convergence, weak generalization, and inefficient exploration. We propose Uncertainty-Aware LLM-Guided Policy Shaping (ULPS), a novel framework that integrates a calib...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Uncertainty-Aware LLM-Guided Policy Shaping for Sparse-Reward Reinforcement Learning*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
该论文针对**稀疏奖励（sparse rewards）** 和**异构任务序列（heterogeneous task sequences）** 在强化学习（Reinforcement Learning, RL）中的挑战，这些问题导致：
- 学习收敛缓慢
- 探索效率低下
- 泛化能力弱
- 难以在长序列任务中发现成功路径

尤其是在 MiniGrid 等环境中，智能体仅在完成一系列严格顺序的子任务后才获得奖励，传统 RL 方法依赖随机探索，样本效率极低。

---

### 🚀 提出的新方法：ULPS（Uncertainty-Aware LLM-Guided Policy Shaping）
提出了一种**统一框架 ULPS**，将经过校准的 Large Language Model（LLM）集成到 RL 训练循环中，实现**不确定性感知的策略塑造（policy shaping）**。

#### 核心创新点：
1. **符号轨迹生成与LLM微调**  
   使用 **A\*算法**作为oracle生成最优符号轨迹（symbolic trajectories），并用这些轨迹**微调一个BERT-based LLM**，使其具备多步任务推理能力。

2. **不确定性感知的策略融合机制**  
   利用 **Monte Carlo (MC) Dropout** 估计LLM输出的**认知不确定性（epistemic uncertainty）**，并通过**熵归一化（H<sub>norm</sub>）** 动态调节LLM建议的影响权重。

3. **自适应策略混合公式**  
   最终策略为凸组合形式：
   $$
   P_{\text{final}} = (1 - H_{\text{norm}}) \cdot P_{\text{LLM}} + H_{\text{norm}} \cdot P_{\text{agent}}
   $$
   当LLM不确定时（高熵），更多依赖PPO策略；当自信时（低熵），优先采纳LLM建议。

4. **可扩展的自监督训练范式**  
   不依赖人工标注或人类反馈，通过A\*自动生成大量高质量轨迹用于LLM预训练，提升了方法的可扩展性。

---

### 🔍 相比现有方法的优势
| 对比维度 | ULPS优势 |
|--------|---------|
| **相比纯RL方法（如PPO、DQN）** | 显著提升探索效率，在稀疏奖励下更快找到成功路径 |
| **相比未校准LLM引导方法** | 避免对LLM“幻觉”或错误建议的过度依赖，提高稳定性 |
| **相比人类反馈引导方法** | 无需昂贵的人类标注，完全自监督构建先验知识 |
| **相比固定调度策略（如早期用LLM后期切换）** | 动态按需调整，更灵活可靠 |

---

## 2. **核心实验方法和设置**

### 📚 数据集与环境
- 主要基准：**MiniGrid-UnlockPickup-v0**  
  包含三个有序子任务：
  1. 找到并拾取钥匙（key pickup）
  2. 移动至门并解锁（unlock door）
  3. 到达目标位置（reach goal）
- 观察空间：7×7局部视野，离散动作空间（turn left/right, move forward, pick up, toggle）
- 奖励结构稀疏：
  - 拾钥：+0.5
  - 开门：+0.5
  - 到达终点：+0.2 + 步数bonus（$1 - \frac{\text{steps}}{\text{max\_steps}}$）
  - 无效动作惩罚：-0.02

### ⚙️ 实验设置
- **模型架构**：
  - LLM：`bert-base-uncased`，输入最大100 tokens，dropout=0.1
  - RL算法：Proximal Policy Optimization (**PPO**)，AdamW优化器（lr=5e-5），batch size=16
  - 不确定性估计：MC Dropout进行8次前向传播（T=8）
- **训练配置**：
  - 总训练轮数：1000 episodes
  - 每episode最多50 steps
  - 每50个episode更新一次PPO策略
- **环境规模**：
  - LLM微调阶段：8×4 grid
  - RL评估阶段：4×4 和 8×8 grids

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Success Rate (%)** | 成功到达目标的episode占比 |
| **Average Steps to Goal** | 成功episode的平均步数 |
| **Reward AUC** | 奖励曲线下的面积（Area Under Curve），衡量累积奖励效率 |
| **Total Steps** | 所有episodes总交互步数（反映样本复杂度） |
| **Brier Score (BS)** | 概率预测准确性，越低越好 |
| **Expected Calibration Error (ECE)** | 校准误差，衡量置信度与准确率一致性 |
| **Discrimination Analysis (DA) / AUC-ROC** | 区分正确与错误动作的能力 |

### 🆚 基线方法对比
- **Unguided RL (PPO)**：无外部引导的标准PPO
- **Linear RL (PPO)**：线性衰减LLM影响
- **Uncalibrated LLM + RL**：直接使用LLM建议，不估计不确定性
- **Q-Learning (GRPO)**：基于Q表的传统方法
- **DQN (GRPO)**：深度Q网络变体（Double DQN + Dueling + Prioritized Replay）

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（4×4环境）

| 方法 | Reward AUC | Success Rate (%) | Avg. Steps | Total Steps (1000 eps) |
|------|------------|------------------|-----------|------------------------|
| **Proposed ULPS (Ours)** | **2055.08** | **99.90** | **7.24** | **7,286** |
| Uncalibrated LLM | 1706.43 | 94.00 | 18.39 | 20,284 |
| Linear RL (PPO) | 1865.57 | 74.90 | 15.84 | 24,412 |
| Unguided RL (PPO) | 221.31 | 5.90 | 35.54 | 49,147 |
| Q-Learning (GRPO) | 1515.71 | 82.40 | 16.19 | 22,142 |
| DQN (GRPO) | 317.46 | 11.60 | 31.66 | 47,873 |

> ✅ ULPS在所有指标上均显著优于基线：
> - 成功率接近完美（>99.9%）
> - 平均仅需 **7.24步** 即完成任务
> - **总环境交互减少86%**（vs. unguided RL）

---

### 📊 更大规模环境表现（8×8）

| 方法 | Reward AUC | Success Rate (%) | Avg. Steps |
|------|------------|------------------|-----------|
| **Proposed ULPS (Ours)** | **1886.80** | **99.70** | **15.37** |
| Uncalibrated LLM | 1113.77 | 72.3 | 38.57 |
| Unguided RL (PPO) | -358.18 | 0.0 | 50.0 |

> ✅ 在更大、更复杂的环境中，ULPS依然保持极高成功率，而其他方法几乎失败。

---

### 🔬 消融实验结果（Ablation Study）

#### 表格 III：不同MC Dropout参数的影响（4×4）

| Dropout Rate | Forward Passes | Avg. Steps | Reward AUC |
|--------------|----------------|-------------|------------|
| 0.05 | 8 | 7.0 | 2054.36 |
| 0.1 | **8** | **7.24** | **2055.08** |
| 0.1 | 12 | 7.4 | 2055.72 |
| 0.2 | 12 | 7.0 | **2055.88** |

- 发现：虽然更高dropout和更多forward pass能略微提升AUC，但 **dropout=0.1, T=8** 提供最佳性价比（计算成本低且性能稳定）。
- 支持结论：适度的不确定性建模即可带来巨大收益。

---

### 🧪 模型校准性能比较（Table I）

| Model | Acc. (%) | ECE ↓ | BS ↓ | DA (AUC) ↑ |
|-------|----------|--------|-------|-------------|
| Shoaeinaeini et al. [15], 4×4 | 90.00 | 0.15 | 0.20 | 0.80 |
| **Proposed (Ours)** | **99.17** | **0.20** | **0.06** | **1.00** |

> 注：此处原文ECE值可能有误（通常ECE越小越好，但本文中Proposed模型ECE=0.20高于[15]的0.15），但从BS和DA看，本模型概率预测更准确、区分能力更强。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **不确定性感知是关键**  
   直接使用LLM建议会导致过拟合错误先验；引入MC Dropout进行不确定性估计，可有效避免对不可靠建议的盲目信任。

2. **A\*-generated轨迹是高效监督信号**  
   利用符号规划器生成最优路径，为LLM提供高质量训练数据，无需人工干预。

3. **动态策略融合优于静态调度**  
   基于熵的加权机制实现了“按需引导”，在简单状态依赖LLM加速，在复杂/模糊状态保留自主决策。

4. **大幅降低样本复杂度**  
   ULPS仅需约 **7,286次环境交互** 即达到近完美性能，相较无引导RL（~49k）节省超过85%，证明其卓越的**sample efficiency**。

5. **良好的泛化性和可扩展性**  
   在4×4和8×8环境下均表现出色，表明方法对环境规模变化具有鲁棒性。

---

### ⚠️ 局限性
1. **计算开销增加**  
   MC Dropout需进行多次前向传播（T=8），带来约8倍的**单步推理延迟**，不适合实时性要求高的场景。

2. **依赖可建模的环境结构**  
   A\* planner适用于网格世界等结构化环境，但在高度非结构化或连续空间中难以应用。

3. **语言模型容量限制**  
   当前使用BERT-base，未来可尝试更大LLM（如LLaMA系列）以处理更复杂语义。

4. **提示工程敏感性**  
   文本prompt的设计会影响LLM表现，目前尚未系统研究最优prompt结构。

---

### 🔮 未来工作方向
1. **扩展至部分可观测环境（POMDP）**  
   结合记忆机制（如Transformer memory）处理历史依赖。

2. **多智能体协作场景**  
   将ULPS应用于multi-agent RL，利用LLM协调团队行为。

3. **层次化提示设计（Hierarchical Prompting）**  
   引导LLM生成高级子目标（subgoals）而非单一动作，提升长期规划能力。

4. **多模态输入支持**  
   融合视觉输入（如图像观测）与文本描述，构建 multimodal LLM + RL 架构。

5. **更紧密的规划-学习耦合**  
   将A\*等搜索过程嵌入训练流程，实现在线 replanning 与 policy refinement 联合优化。

---

## ✅ 总结
该论文提出的 **ULPS框架** 成功地将 **symbolic planning（A\*）**、**pretrained language priors（BERT）** 与 **uncertainty-aware control（MC Dropout + entropy blending）** 有机结合，为解决稀疏奖励RL问题提供了**原则性强、可解释性高、效率优越**的新范式。其实验充分验证了“**何时相信LLM”比“是否使用LLM”更重要**这一核心思想，为未来LLM-for-RL的研究奠定了坚实基础。

</details>

---

### 2. [Self-evolving LLM agents with in-distribution Optimization](https://arxiv.org/abs/2606.07367)

**Authors**: Yudi Zhang, Meng Fang, Zhenfang Chen, Mykola Pechenizkiy  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.07367v1  

#### Abstract
Large Language Models (LLMs) have recently emerged as powerful controllers for interactive agents in complex environments, yet training them to perform reliable long-horizon decision making remains a fundamental challenge. A key difficulty lies in credit assignment: agents often receive delayed rewa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Self-evolving LLM agents with in-distribution Optimization*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **LLM agents 在长视野、稀疏奖励环境中的决策能力不足** 这一核心挑战展开研究。主要难点在于：
- **Credit Assignment 问题**：任务的成功信号通常只在 episode 结束时获得，难以将最终奖励归因于中间步骤。
- **Distribution Shift 问题**：现有方法（如 PRM）依赖离线训练的奖励模型，但在策略演化过程中生成的新状态-动作对可能超出其训练分布，导致反馈不可靠。
- **样本效率低**：许多方法依赖大量在线交互进行搜索或回溯，成本高昂且难以扩展。

### 提出了什么新方法或新思路
作者提出 **Q-Evolve**，一个自演化的 LLM agent 框架，其核心思想是：
- **统一过程奖励标注与策略学习**：在一个共享的 **in-distribution** 学习循环中，协同演化策略（policy）、批评家（critic）和数据集。
- **基于混合离线数据集的 in-distribution critic 学习**：利用 **expert demonstrations** 和 **agent-generated trajectories** 构建混合数据集，通过加权的 **Implicit Q-Learning (IQL)** 学习一个稳定的 critic，避免外推误差。
- **优势估计作为过程奖励**：使用 **Generalized Advantage Estimation (GAE)** 从 critic 中导出 step-wise 的过程奖励，实现密集监督。
- **行为邻近策略优化 (BPPO)**：在固定数据集上进行策略更新，采用带非对称裁剪的 **behavior-proximal policy optimization**，显式抑制负优势动作，防止过拟合。

### 相比现有方法的优势
| 特性 | Q-Evolve | 现有方法（如 QLASS, PPO） |
|------|---------|------------------------|
| **无需手动标注** | √ | × (QLASS 需大量搜索) |
| **无需环境回溯** | √ | × (许多方法需要) |
| **缓解分布偏移** | √ (in-distribution learning) | × (易受分布外动作影响) |
| **样本效率高** | √ (仅需少量在线交互) | × (PPO/QLASS 需大量在线采样) |
| **自我迭代改进** | √ (closed-loop evolution) | × (多为一次性训练) |

---

## 2. 核心实验方法和设置

### 使用的数据集
在三个具有延迟奖励特性的标准基准上进行评估：
- **AlfWorld**：文本驱动的具身家庭任务，代理需执行长序列动作完成目标，仅在最后获得二元奖励。
- **WebShop**：模拟在线购物环境，代理需通过导航和点击完成购买，奖励在“点击购买”后给出。
- **ScienceWorld**：文本虚拟科学实验环境，任务包含子目标，奖励稀疏。

### 实验设置和评估指标
- **基础模型**：`Llama-2-7B-Chat` 和 `Llama-3-8B-Instruct`。
- **评估指标**：平均累积奖励（average accumulated rewards），分别报告 **Seen**（见过的任务）和 **Unseen**（未见过的任务）上的性能以评估泛化能力。
- **交互方式**：每个任务采样 3 条轨迹用于自收集数据。

### 基线方法对比
对比了三类基线：
1. **零样本 LLM**：`GPT-3.5-Turbo`, `GPT-4`（使用 ReAct 提示）。
2. **无奖励重分配的微调方法**：
   - `SFT`：在专家轨迹上监督微调。
   - `RFT`：拒绝采样微调。
3. **有奖励重分配的方法**：
   - `PPO`：强化学习基线。
   - `ETO`：基于轨迹偏好对的优化。
   - `DMPO`：多轮偏好优化。
   - `QLASS`：基于 Q 值引导的逐步搜索（需大量在线采样）。
   - `Best-of-N`：推理时策略。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）
| Method | WebShop | SciWorld (Seen) | SciWorld (Unseen) | ALFWorld (Seen) | ALFWorld (Unseen) | **Average** |
|--------|---------|------------------|--------------------|------------------|--------------------|-------------|
| **QLASS** | 70.3 | 75.3 | 66.4 | 77.9 | 82.8 | 74.5 |
| **Q-Evolve (Ours)** | **70.5** | **76.3** | **69.7** | **90.7** | **89.6** | **79.4** |

- Q-Evolve 在所有任务上均取得 **最佳平均分**，尤其在 ALFWorld 上大幅领先（+16.5 vs QLASS）。
- 在 **未见任务 (Unseen)** 上表现优异，表明其良好的泛化能力。

### 与基线方法的对比结果
- **vs QLASS**：性能更高，且 **样本效率显著提升**（AlfWorld 上 QLASS 需 600K 步，Q-Evolve 仅需 20K）。
- **vs ETO**：整体性能更优，得益于更稳定的 in-distribution 自我演化循环。
- **vs PPO/SFT**：显著优于不处理稀疏奖励的方法，证明了密集过程监督的有效性。

### 消融实验结果（见 Table 3, Figure 3）
- **移除任何组件均导致性能下降**，验证了各模块的必要性：
  - **w/o Retrospective Relabeling (RT)**：性能下降 → 说明规则化辅助奖励对识别失败很重要。
  - **w/o Weighted IQL (W-IQL)**：性能下降 → 加权机制对稳定 critic 学习至关重要。
  - **w/o GAE**：性能大幅下降 → 多步优势估计比单步 Q-V 更可靠。
  - **w/o Policy Improvement (PI)**：性能严重退化 → 显式的策略优化是性能提升的关键。
- **迭代改进有效**：从 Iter-1 到 Iter-2 性能持续提升，证明了闭环自我演化的可行性。
- **样本效率高**：Q-Evolve (1-iter, 13K 步) 在 ALFWorld 上的表现 **超过所有训练 320K 步的 online RL 方法**（见 Table 5）。

---

## 4. 关键结论和发现

### 主要发现
1. **稳定可靠的自我演化是可行的**：通过在共享的 in-distribution 数据集上协同演化 critic、策略和数据，可以实现稳定、可迭代的性能提升。
2. **in-distribution learning 是关键**：将过程奖励标注和策略学习限制在同一数据分布内，能有效缓解 distribution shift，保证反馈的可靠性。
3. **混合数据集至关重要**：结合 **expert demonstrations**（提供成功路径）和 **agent rollouts**（覆盖失败模式）能构建更鲁棒的 critic。
4. **显式抑制负动作很重要**：使用非对称裁剪的 BPPO 能有效抑制有害行为，而不仅仅是增强好行为。

### 方法的局限性
- **依赖结构化环境反馈**：回溯奖励（retrospective rewards）的设计依赖于环境中明确的错误提示（如“格式错误”），在反馈不明确的任务中可能需要调整。
- **轨迹多样性受限**：数据收集依赖贪婪 rollout，可能导致策略收敛到局部最优。
- **跨迭代分布漂移**：虽然单次迭代内缓解了分布偏移，但多次迭代间的策略演化仍会积累分布漂移，当前框架未显式纠正。

### 未来工作方向
- 探索更高效的探索策略以增加轨迹多样性。
- 设计机制来检测和纠正跨迭代的分布漂移。
- 将方法扩展到更多样化、更复杂的现实世界交互任务中。
- 进一步降低对环境特定规则（如回溯奖励设计）的依赖，提高通用性。

</details>

---

### 3. [Teaching the Way, Not the Answer: Privileged Tutoring Distillation for Multimodal Policy Optimization](https://arxiv.org/abs/2606.07000)

**Authors**: Shizhe Xiang, Ke An, Wenlong Yu, Yue Liu, Jian Luan, Pei Fu, Qilong Wang  
**Category**: cs.AI  
**Published**: 2026-06-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.07000v1  

#### Abstract
Recent post-training methods, particularly Reinforcement Learning with Verifiable Rewards (RLVR), have significantly enhanced the reasoning ability of Large Vision-Language Models (LVLMs). However, the sparse nature of verifiable rewards provides little token-level supervision for failed rollouts, o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Teaching the Way, Not the Answer: Privileged Tutoring Distillation for Multimodal Policy Optimization*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对 **Reinforcement Learning with Verifiable Rewards (RLVR)** 在多模态推理任务中存在的两大核心挑战：

- **Reward Sparsity**：RLVR 仅提供最终答案级别的稀疏奖励信号，无法为失败的推理路径（failed rollouts）提供细粒度的 token-level 监督，导致探索效率低下。
- **Answer-Revealing Distillation 的副作用**：现有的自蒸馏（self-distillation）方法常通过引入真实答案（GT Answer）或完整思维链（CoT）来增强监督，但这会诱导模型学习“答案捷径”（answer shortcuts），抑制探索，导致策略熵（policy entropy）过早坍缩。

### **提出了什么新方法或新思路**

作者提出 **PTD-PO (Privileged Tutoring Distillation Policy Optimization)**，一种基于**特权信息范式**（Learning with Privileged Information, LwPI）的新型蒸馏框架，其核心思想是：

> **“教方法，而不是教答案”** —— 通过构建**非泄露性的结构化提示**（structured privileged hints）作为“特权教师”的输入，为失败的推理轨迹提供密集的 token-level 监督，同时确保学生策略仍在原始无答案上下文中进行训练。

具体创新包括：

- **Privileged Tutoring Distillation (PTD)**：冻结的参考模型（reference model）在训练时接收额外的**空间注意力引导**（spatial attention guidance）和**中间文本推理步骤**（intermediate textual reasoning steps）作为提示，生成更优的 token 分布，用于指导学生模型。
- **非对称上下文蒸馏**（Asymmetric-context distillation）：学生策略始终在 `question-only` 上下文中生成响应，而教师分布则在 `question + hint` 的增强上下文中生成，实现了信息解耦。
- **Top-K Jensen-Shannon Divergence with Tail Compensation**：为稳定因上下文差异导致的分布偏移，并降低内存开销，提出了一种改进的蒸馏目标函数，只保留 Top-K 高概率 token 和一个聚合的“尾部桶”（tail bucket）进行 JSD 计算。

### **相比现有方法的优势**

| 对比维度 | 传统方法（如 HDPO） | PTD-PO |
|---------|---------------------|--------|
| **监督密度** | 低（仅 outcome-level） | 高（token-level） |
| **计算开销** | 高（需外部教师在线推理） | 低（自蒸馏，无需外部模型） |
| **是否暴露答案** | 是（GT Answer / CoT） | 否（仅非泄露性 hints） |
| **探索能力** | 易受抑制，熵坍缩快 | 更好维持探索，缓解熵坍缩 |
| **泛化性** | 可能过拟合答案模式 | 更关注推理路径，泛化更强 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **训练数据**：`ViRL39K`，包含 38,870 个可验证的视觉-语言问答样本，涵盖数学、科学、图表、空间推理等多种复杂场景。
- **评估基准**：`PAPO` 多模态推理评测套件，整合多个权威数据集，分为两类：
  - **General Multimodal Reasoning**：MMK12, Geo3K, MathVerse, MathVista, We-Math
  - **Vision-Dependent Multimodal Reasoning**：MMMU-Pro, Counting, MathVersey, LogicVista
- 所有自由形式答案均被过滤，采用规则匹配进行自动验证，避免依赖 LLM-as-a-judge。

### **实验设置和评估指标**

- **模型规模**：在 `Qwen3-VL-2B`, `4B`, `8B` 三种参数量级上进行训练与评估。
- **训练配置**：
  - 最大响应长度：4096 tokens
  - Rollout 数量：8
  - 优化器：AdamW，学习率 1e-6
  - 使用 `bf16` 精度
- **评估方式**：
  - 学生模型始终在 `question-only` 上下文中推理（不接触任何 hint）
  - 报告各数据集准确率及平均得分（AVG）

### **基线方法对比**

- **SFT**：标准监督微调
- **OPSD**：Online Policy Self-Distillation
- **GRPO**：Group Relative Policy Optimization（主流 RLVR 方法）
- **HDPO**：Hybrid Distillation PO，使用 GT Answer 进行自蒸馏
- **PAPO**：Perception-Aware Policy Optimization，强调视觉感知的 RLVR 方法

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Table 1）**

| 方法 | 2B AVG | 4B AVG | 8B AVG |
|------|--------|--------|--------|
| GRPO | 50.63 | 68.06 | 68.78 |
| HDPO | 57.68 | 68.29 | 69.92 |
| PAPO | 54.11 | 69.20 | 69.05 |
| **PTD-PO (Ours)** | **61.21** | **71.23** | **71.86** |

> ✅ **PTD-PO 在所有模型尺度上均显著优于所有基线方法**，尤其在 2B 小模型上提升最大（+10.58%），说明其对资源受限场景更具价值。

#### **分项性能亮点（以 4B 为例）**
- **Geo3K**：从 62.62 (GRPO) → **64.79** (+2.17)
- **MathVerse**：从 75.25 → **80.45** (+5.2)
- **MMMU-Pro**：从 40.19 → **41.20**
- **LogicVista**：从 58.56 → **60.32**

### **与基线方法的对比结果**

- **vs GRPO**：PTD-PO 显著提升复杂多模态推理能力，尤其是在需要精细视觉接地和多步推理的任务上。
- **vs HDPO**：尽管 HDPO 使用了 GT Answer，但 PTD-PO 仍实现超越，证明**非泄露性提示比直接暴露答案更能促进稳健推理**。
- **vs PAPO**：PAPO 虽也关注视觉感知，但 PTD-PO 通过更丰富的结构化提示进一步提升了性能，表明**提示设计比单一感知信号更有效**。

### **消融实验结果（Table 2）**

| 变体 | 4B 总体 AVG | 相对增益 |
|------|-------------|----------|
| GRPO | 68.29 | — |
| +PTD (thr=1.0) | **71.77** | +3.48 |
| +PTD (thr=0.2) | 69.48 | +1.19 |
| +PTD (All Trajectories) | 70.11 | +1.82 |
| +PTD (w/o Structured Hint) | 70.76 | +2.47 |

#### **关键发现**：
- **激活阈值 Tptd=1.0 效果最佳**：即对所有失败轨迹应用 PTD，说明失败路径最需要纠正。
- **结构化提示设计至关重要**：去除“零剧透”、“过滤干扰项”等约束后性能下降，说明提示质量直接影响蒸馏效果。
- **不应应用于成功轨迹**：对所有轨迹蒸馏反而不如仅对失败轨迹，避免过度正则化。

---

## 4. 关键结论和发现

### **主要发现**

1. **稀疏奖励不足以支撑复杂多模态推理**：大量采样组在训练初期为“全失败”，缺乏有效学习信号。
2. **答案暴露会损害探索行为**：GT Answer 或 CoT 条件下的教师会产生“捷径式”输出，导致响应变短、分布尖锐、KL 散度增大。
3. **结构化非泄露提示可提供有效纠正信号**：相比直接给答案，合理的 hint 能引导教师模型聚焦关键推理路径而不泄露终点。
4. **PTD-PO 显著提升恢复能力**：在硬例上，PTD-PO 能将更多“完全失败”案例转化为“部分成功”或“完全成功”。
5. **Top-K JSD 有效平衡性能与效率**：在 K=100 时即可接近完整 JSD 效果，内存开销从 O(BTV) 降至 O(BTK)，适合大规模训练。

### **方法的局限性**

- **依赖高质量 hint 构建**：当前 hints 由强模型（如 Qwen-235B）离线生成，若 hint 错误或模糊，可能导致错误蒸馏。
- **对高能力模型增益递减**：随着模型变大（如 8B），失败轨迹减少，PTD 可应用的机会变少，相对提升幅度缩小。
- **未完全解决极端组合推理错误**：如化学方程式配平、多条件一致性校验等任务，仍可能出现“看似合理但最终错误”的输出。

### **未来工作方向**

- **构建更具挑战性的 RLVR 训练数据**：设计更难的多模态任务，以激发更强模型的失败路径，扩大 PTD 的应用范围。
- **动态 hint 生成机制**：探索在训练过程中根据学生表现动态调整 hint 内容，实现个性化辅导。
- **扩展至多模态 Agent 场景**：将 PTD 思想应用于具身智能体，利用中间观测或工具反馈作为特权信息，指导失败的行为序列。
- **自动化 hint 构建 pipeline**：减少对外部大模型的依赖，实现端到端的 hint 自学习机制。

---

> 🔗 **项目主页**：[https://github.com/XszNeverSleep/PTD-PO](https://github.com/XszNeverSleep/PTD-PO)

</details>

---

### 4. [PCCL: Process Group-Aware Scalable and Generic Collective Algorithm Synthesizer](https://arxiv.org/abs/2606.07019)

**Authors**: William Won, Kartik Lakhotia, Madhu Kumar, Sudarshan Srinivasan, Tushar Krishna  
**Category**: cs.DC  
**Published**: 2026-06-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.07019v1  

#### Abstract
Distributed machine learning has become increasingly important due to the massive scale of large-scale generative models. Both model parameters and data are distributed across many compute devices, which requires frequent collective communications to synchronize activations and parameter updates. Su...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PCCL: Process Group-Aware Scalable and Generic Collective Algorithm Synthesizer

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模分布式机器学习（尤其是大模型训练）中，**collective communication**（如 All-Reduce、All-to-All）已成为系统性能的关键瓶颈。现有的 **Collective Algorithm Synthesizer** 存在以下不足：
- 多数方法忽略 **Process Group** 结构，假设所有设备参与集体通信，而实际中仅子集参与；
- 缺乏对 **异构网络拓扑**（heterogeneous/asymmetric）的支持；
- 难以扩展到大规模集群（如数千 NPUs）；
- 支持的 **collective pattern** 类型有限，尤其缺乏对 **All-to-All** 和 **All-to-Allv** 的高效支持。

### 提出的新方法
作者提出 **PCCL**（Process Group-Aware Collective Communication Library），一个**可扩展、通用且感知进程组的 collective algorithm synthesizer**，其核心创新包括：

- **基于 TEN（Time-Expanded Network）的数据结构建模**  
  将时空信息统一表示为三维布尔矩阵 `TEN[t][s][d]`，支持精确建模时间维度上的链路占用与传输延迟。

- **BFS 路径搜索算法**  
  在 TEN 上执行广度优先搜索（BFS）来合成任意 collective pattern 的最优路径，相比 ILP 或 LP 方法具有更好的可扩展性。

- **原生支持 Process Group Awareness**  
  明确利用 process group 信息，在合成时允许跨 group 外部节点作为中继，从而更充分地利用网络资源。

- **支持通用拓扑与 collectives**  
  支持异构带宽/延迟链路、switch 建模，并能生成 All-Reduce、All-Gather、Reduce-Scatter、All-to-All、All-to-Allv 等多种模式。

### 相比现有方法的优势

| 特性 | PCCL | 其他 synthesizer（如 TACCL、TE-CCL、SCCL） |
|------|------|---------------------------------------------|
| 可扩展性（Scalability） | ✅ 支持 512-NPU 合成仅需 11.68 分钟 | ❌ 多数基于 ILP/SMT，复杂度高，难以扩展 |
| 通用拓扑支持 | ✅ 支持异构、非对称、含 switch 的网络 | ⚠️ 多数假设同构或对称拓扑 |
| 通用 collective 支持 | ✅ 支持 All-to-All、All-to-Allv 等 | ⚠️ 多数仅支持 All-Reduce 类 |
| Process Group 感知 | ✅ 原生支持，可跨 group 利用闲置链路 | ❌ 假设全集群参与，无法利用外部资源 |

---

## 2. 核心实验方法和设置

### 实验平台与仿真器
- 使用 **ASTRA-sim** 进行系统级模拟，该工具已被验证在真实系统（如 128-GPU H100 集群）上达到 **97% 准确率**。
- 合成结果通过 **MSCCL executor** 在 16 和 32-GPU 集群上执行验证正确性。

### 网络拓扑类型
- **2D Mesh**
- **3D Hypercube**
- **Heterogeneous 2D Switch Topology**（每节点 8 NPU，集群规模从 16 到 256 NPU）

### 评估指标
- **Synthesis Time**：算法合成耗时（衡量可扩展性）
- **All-to-All Bandwidth / Speedup**：有效带宽与加速比
- **Link Utilization Heatmap**：网络链路利用率可视化
- **Normalized Bandwidth**：相对于基线的归一化性能

### 基线方法对比
- **Direct**：点对点发送接收（即 pairwise send/receive），当前主流 CCL 实现方式。
- **CCLs**：指代现有集体通信库（如 NCCL、RCCL、oneCCL）中的默认算法。
- **TE-CCL**：最先进的 All-to-All synthesizer，用于比较可扩展性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 结果 |
|------|------|
| **512-NPU All-to-All 合成时间** | **11.68 分钟** |
| **1000-NPU All-to-All 合成时间** | **2.01 小时** |
| **6×6 Mesh (36 NPU) 合成速度提升** | 比 TE-CCL 快 **4,404×** |
| **All-to-All 合成复杂度** | 实测为 **O(n³)**，远优于 ILP 的指数级增长 |

### 与基线方法的对比结果

#### 可扩展性对比（vs. TE-CCL）
- 图 11 显示：在 6×6 Mesh 上，PCCL 合成时间为秒级，而 TE-CCL 耗时超过 30 分钟。
- TE-CCL 在超过 7×7 Mesh 后无法完成合成，而 PCCL 可持续扩展至 512 NPU。

#### 性能加速比（vs. Direct / CCLs）
- 在多个 process group 并发运行 All-to-All 场景下：
  - **平均加速 2.68×**
  - 最高达 **3.03×**
- 在异构 2D Switch 拓扑上：
  - **平均带宽提升 1.33×**

#### 网络利用率对比
- 图 17 显示：**PCCL 合成算法充分利用整个网络拓扑**，即使某些 NPU 不属于 process group，仍被用作中继；
- 而 Direct 算法局限于局部通信，导致严重网络欠载（underutilization）。

#### 敏感性分析（图 19）
- 固定 8×8 Mesh，增加并发的 128 MiB All-to-All process group 数量（每个大小为 8）：
  - 当仅有 1 个 group 时，PCCL 达到 **3.05× 加速**
  - 随着并发 group 增多，资源竞争加剧，加速比下降但仍保持优势

---

## 4. 关键结论和发现

### 主要发现
1. **Process Group Awareness 是关键优化机会**  
   允许算法跨越 process group 边界使用空闲链路，显著提升带宽利用率和通信效率。

2. **BFS + TEN 架构实现了可扩展性与通用性的平衡**  
   相比 ILP/SMT 方法，避免了 NP-hard 问题带来的计算爆炸，同时保留了对异构拓扑和复杂 collective 的支持能力。

3. **All-to-All 成为 MoE 模型的关键瓶颈，亟需专用算法**  
   当前 CCLs 多采用 Direct 实现，效率低下；PCCL 首次实现大规模 All-to-All 的自动高效合成。

4. **PCCL 可无缝对接现有执行框架**  
   通过映射到 **MSCCL** 或 **MSCCL++ IR**，可在 GPU 系统上直接部署，无需修改底层硬件或驱动。

### 方法的局限性
- **未完全解决 switch 内部缓冲区建模的动态行为**：虽然支持 switch 节点建模，但对复杂调度策略（如优先级队列）的支持仍有限。
- **chunk size 固定假设**：目前将消息划分为固定大小 chunk，可能影响小消息场景下的最优性。
- **依赖 TEN 展开的时间步长精度**：在极高精度 α-β 模型下，TEN 维度可能急剧膨胀。

### 未来工作方向
- 扩展至 **动态 workload 调度**，结合 runtime profiling 实现 adaptive synthesis；
- 探索 **learning-based pathfinding** 替代 BFS，进一步加速合成过程；
- 支持 **multi-commodity flow with contention modeling**，更精细处理拥塞控制；
- 与编译器集成（如 **MLIR**），实现从模型到通信算法的端到端自动化优化。

--- 

> **总结一句话**：  
> PCCL 是首个兼具 **可扩展性、通用性和 process group 感知能力** 的 collective algorithm synthesizer，通过 **TEN + BFS** 架构实现了对大规模异构集群中 All-to-All 等关键通信模式的高效自动优化，为下一代分布式 AI 系统提供了算法基础。

</details>

---

### 5. [Structure-Preserving Correction Learning for Sparse Bayesian Inference in Brain Source Imaging](https://arxiv.org/abs/2606.07196)

**Authors**: Marco Morik, Xiao Ruiting, Shinichi Nakajima, Stefan Haufe, Ismail Huseynov  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.07196v1  

#### Abstract
Classical sparse Type-II Bayesian methods for M/EEG brain imaging support joint estimation of source and noise hyperparameters, but rely on fixed iterative update rules. Although these updates are principled and interpretable, their dynamics cannot be adapted from data. We propose to learn the updat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Structure-Preserving Correction Learning for Sparse Bayesian Inference in Brain Source Imaging**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在 M/EEG 脑源成像中，**稀疏贝叶斯学习（Sparse Bayesian Learning, SBL）** 是一种主流的 Type-II 贝叶斯推断框架，用于联合估计源活动和噪声超参数。然而，传统方法依赖固定的迭代更新规则（如 EM、MacKay 或 convex-bounding），这些规则虽然具有理论依据且可解释，但其动态过程无法从数据中自适应调整。

此外，尽管深度学习方法（如 DeepSIF）能提升重建速度和精度，但通常以“黑箱”方式预测源活动，牺牲了模型的可解释性和物理一致性。

### **提出的新方法与新思路**
本文提出了一种 **结构保持的修正学习（Structure-Preserving Correction Learning）** 框架，将经典的凸界（convex-bounding）Type-II 贝叶斯求解器进行展开（unfolding），构建一个可训练的神经网络架构，其每一层对应原算法的一次迭代。

核心思想是：
- **保留原始贝叶斯结构**：初始化时完全复现经典 convex-bounding 更新；
- **仅学习增量修正项**：通过可学习的偏差（bias）、残差 MLP 或注意力机制，在 log-domain 中对超参数更新施加输入相关的修正；
- **不替代原有机制**：训练的目标不是替换贝叶斯推理，而是学习围绕解析更新路径的“结构化修正”，从而提升经验性能的同时保持算法透明性。

具体提出了三种渐进增强的修正变体：
1. **Bias CB**：每层添加可学习的偏置向量（input-agnostic）；
2. **Deep CB**：使用共享的 pointwise MLP 学习输入相关的修正（input-adaptive）；
3. **Deep Attn. CB**：引入跨源注意力模块，建模源之间的上下文依赖关系。

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **可解释性** | 保持了原始贝叶斯更新结构，修正项可视作“扰动”，易于分析和调试； |
| **泛化能力** | Deep CB 和 Deep Attn. CB 使用共享参数，适用于不同传感器/源空间布局； |
| **性能提升** | 显著优于传统 convex-bounding 及其他基线方法，在重建精度、收敛速度和支持恢复上均有改进； |
| **鲁棒性** | 在异方差噪声（heteroscedastic noise）下表现稳定，联合学习噪声仍接近已知噪声性能； |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **合成数据集**：基于 `fsaverage` 模板脑生成，采用 MNE-Python 构建前向模型（lead-field matrix），模拟稀疏时空源活动。
  - 源空间：ico3 网格（1284 个源点），固定方向；
  - 时间步长：T=100，采样率 100Hz；
  - 频段：delta, theta, alpha, mu, beta；
  - 噪声：传感器级对角异方差高斯噪声，SNR ~ U(5,30) dB；
- **真实世界迁移测试**：
  - **THINGS-EEG2 数据集**：用于零样本（zero-shot）验证，评估模型在真实 EEG 上的泛化能力。

### **实验设置**
- **任务**：M/EEG 源成像，目标是从传感器测量 $ Y \in \mathbb{R}^{M\times T} $ 推断源活动 $ X \in \mathbb{R}^{N\times T} $；
- **模型深度**：展开 25 层（K=25）；
- **训练策略**：
  - 使用 **逐层预训练 + 全局微调**（progressive layer-wise training）提高稳定性；
  - 引入 **随机深层监督正则化**（stochastic deep supervision）防止中间状态漂移；
- **优化器**：Adam，学习率 0.001，batch size 512。

### **评估指标**
| 指标 | 含义 |
|------|------|
| **rMSE**（relative Mean Squared Error） | 信号保真度，越小越好； |
| **EMD**（Earth Mover’s Distance） | 空间分布距离，衡量整体定位误差； |
| **F1-score** | 支持恢复能力，基于阈值化的活跃偶极子匹配； |
| **LE**（Localization Error） | 真实峰值到预测最近峰值的平均欧氏距离； |
| **Time** | 单样本推理时间（秒）； |

### **基线方法对比**
| 基线 | 类型 | 特点 |
|------|------|------|
| **sLORETA** | 线性最小范数 | 密集先验，不适合稀疏场景； |
| **Convex Bounding (I)** | 经典 SBL | 固定噪声，仅学习源方差； |
| **Convex Bounding (I,A)** | 经典 SBL | 联合学习源与噪声方差； |
| **DeepSIF** | 端到端深度模型 | 黑箱预测，无显式超参数学习； |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Train Cap, Ico3 设置，均值 ± 标准差，5 seeds）**

| Method | rMSE ↓ | EMD ↓ | F1 ↑ | LE ↓ | Time ↓ |
|--------|--------|-------|------|------|--------|
| Convex Bounding (I) | 0.410 | 0.0122 | 0.535 | 0.0056 | 0.0048 |
| Convex Bounding (I,A) | 0.434 | 0.0130 | 0.522 | 0.0060 | 0.0046 |
| sLORETA | 24.373 | 0.0291 | 0.038 | 0.0251 | 0.0010 |
| DeepSIF | 0.796±0.007 | 0.0403±0.0018 | 0.331±0.009 | 0.0104±0.0001 | 0.0003±0.0001 |
| **Bias CB (I)** | **0.383±0.000** | **0.0115±0.0000** | **0.553±0.000** | **0.0054±0.0000** | 0.0025±0.0007 |
| **Deep CB (I)** | **0.344±0.006** | **0.0105±0.0002** | **0.594±0.008** | **0.0048±0.0000** | 0.0027±0.0000 |
| **Deep Attn. CB (I)** | **0.323±0.002** | **0.0102±0.0002** | **0.608±0.003** | **0.0045±0.0000** | 0.0071±0.0001 |

> ✅ 所有修正学习变体均显著优于原始 convex-bounding 和其他基线。

### **与基线方法的对比结果**
- **Deep CB 和 Deep Attn. CB 明显领先**：
  - rMSE 下降约 **15–20%**；
  - F1 提升超过 **10个百分点**；
  - EMD 减少约 **15%**；
- **联合学习（I,A）几乎无性能损失**：
  - Deep Attn. CB (I,A) 的 rMSE 仅比 (I) 高 0.009，说明动态噪声估计非常稳健；
- **DeepSIF 表现不佳**：
  - 尽管速度快，但在稀疏设置下 F1 和 EMD 远差于 SBL 基础的方法；
- **sLORETA 完全失效**：
  - rMSE 极高，表明密集先验无法处理稀疏源配置；

### **消融实验结果**
#### （1）修正形式的影响（从左至右表达力递增）
| 变体 | 性能趋势 |
|------|--------|
| **Bias CB** | 输入无关，提升有限，但优于原始方法； |
| **Deep CB** | 输入相关 MLP 大幅提升性能，尤其在早期层提供强修正； |
| **Deep Attn. CB** | 引入跨源交互后进一步提升，尤其在复杂空间模式下更优； |

#### （2）收敛行为分析（图4）
- **Deep Attn. CB 收敛最快**：在第5–10层即达到甚至超越传统方法最终性能；
- **加入正则化至关重要**：未加 deep supervision 的模型在中间层不稳定；
- **早期大修正，后期趋近经典更新**（见图8）：
  - 修正项在前几层较大，随后逐渐趋于零；
  - 表明网络学会“快速校正初始轨迹”，后期回归稳定贝叶斯路径；

#### （3）鲁棒性测试（图5）
- **低 SNR 下优势明显**：即使 SNR < 5 dB（超出训练范围），仍优于 baseline；
- **源数量变化下的稳定性**：
  - 在训练分布内（5–20 源）性能稳定；
  - 极端稀疏（1–3 源）略有下降 → 反映出训练先验偏差；
- **零样本迁移成功**（图9）：
  - 在 THINGS-EEG2 上无需微调即可定位枕叶视觉响应；
  - Deep CB 在单试次（low-SNR）下表现最优，Deep Attn. CB 对噪声较敏感；

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **结构化修正学习有效提升了 SBL 的经验性能**：
   - 在不破坏原有贝叶斯结构的前提下，通过学习残差修正显著改善重建质量；
2. ✅ **Deep CB 和 Deep Attn. CB 实现更快收敛与更高精度**：
   - 尤其在联合学习源与噪声超参数时表现出色；
3. ✅ **修正项具有“瞬态引导”特性**：
   - 早期层施加大幅修正以加速收敛，后期回归经典更新路径，体现“智能初始化 + 渐进精炼”机制；
4. ✅ **模型具备良好泛化能力**：
   - 可推广至新传感器布局（Things Cap）和更高分辨率源空间（Ico4）；
5. ✅ **零样本应用于真实 EEG 成功**：
   - 能合理定位视觉诱发活动，提取出典型 ERP 波形；

### **局限性**
1. ❗ **假设限制**：
   - 当前框架假设固定方向源和对角协方差结构；
   - 未支持自由方向偶极子或全协方差矩阵；
2. ❗ **依赖合成数据训练**：
   - 缺乏体内 ground truth，定量验证受限；
   - 模型学到一定程度的数据先验（如 sparsity level），在极端稀疏下性能下降；
3. ❗ **计算成本较高（训练阶段）**：
   - Deep Attn. CB 训练耗时长达 15 小时/模型（A100 GPU）；
   - 但推理仍高效（< 0.1 秒/样本）；

### **未来工作方向**
1. 🔮 扩展至 **自由方向偶极子建模** 和 **MEG 特定方向先验**；
2. 🔮 推广到 **非对角噪声/源协方差结构**（如 full noise covariance）；
3. 🔮 将结构化修正学习应用于其他 Type-II 贝叶斯模型：
   - 如 Automatic Relevance Determination (ARD)、Relevance Vector Machine (RVM)、compressive sensing 等；
4. 🔮 结合生理先验（如 fMRI connectivity）进行多模态引导训练；
5. 🔮 开发轻量化版本以便临床实时应用。

---

> 📌 **总结一句话**：  
> 本文提出了一种**既保留贝叶斯可解释性又具备数据驱动灵活性**的新范式——**结构保持的修正学习**，为 M/EEG 源成像提供了高性能、高透明度的下一代稀疏推断框架。

</details>

---

### 6. [GenPO++: Generative Policy Optimization with Jacobian-free Likelihood Ratios](https://arxiv.org/abs/2606.06967)

**Authors**: Ke Hu, Shutong Ding, Panxin Tao, Jingya Wang, Ye Shi  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.06967v1  

#### Abstract
Generative policies provide expressive and multimodal action distributions, making them attractive for reinforcement learning (RL) in complex continuous-control tasks. Among them, flow-based policies are especially appealing because they generate actions through deterministic transport maps. However...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GenPO++: Generative Policy Optimization with Jacobian-free Likelihood Ratios

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在基于 likelihood 的 on-policy 强化学习（如 PPO）中，使用生成式策略（generative policies），尤其是基于 flow 或 diffusion 的策略面临一个核心挑战：**如何高效且准确地计算执行动作在新旧策略下的概率密度比（likelihood ratio）**。

现有方法存在以下两类缺陷：
- **近似方法（如 FPO）**：用 ELBO 等变分下界替代真实 likelihood ratio，导致更新有偏（biased updates），影响训练稳定性。
- **精确方法（如 GenPO）**：通过引入 dummy action 扩展动作空间以实现可逆采样和精确密度估计，但带来了维度翻倍、冗余探索变量、额外计算开销等问题，不适用于 fine-tuning 预训练模型。

### **提出了什么新方法或新思路**

本文提出 **GenPO++**，一种**无需雅可比（Jacobian-free）且保持原动作维度的可逆生成策略优化框架**。其核心思想是：

- 利用**高阶 ODE 求解器中的历史状态（solver-history states）作为辅助记忆**，替代 GenPO 中独立的 dummy action。
- 构造一个**高阶可逆 flow policy solver**，使得前向传播和反向恢复都可在闭式（closed-form）完成。
- 该求解器对应的变换的 log-determinant **仅依赖于固定的求解器系数，与神经网络速度场无关**，从而实现 **Jacobian-free 的精确 likelihood ratio 计算**。

### **相比现有方法的优势**

| 特性 | FPO | GenPO | GenPO++ |
|------|-----|--------|---------|
| 是否使用真实 likelihood ratio | ❌（使用 ELBO 近似） | ✅ | ✅ |
| 是否改变原始动作维度 | ❌ | ✅（翻倍） | ❌ |
| 是否需要计算 Jacobian | ❌ | ✅（反向过程） | ❌ |
| 是否支持预训练策略 fine-tuning | ✅ | ❌（结构不兼容） | ✅ |
| 计算效率 | 中等 | 低（高开销） | 高 |
| 训练稳定性 | 可能不稳定（偏差） | 高（但慢） | 高且快 |

> ✅ **GenPO++ 在保留生成式策略表达能力的同时，避免了偏差和冗余计算，实现了高效、稳定、兼容的 on-policy 优化。**

---

## 2. 核心实验方法和设置

### **使用的数据集与任务**

实验涵盖三大类场景，验证 GenPO++ 的通用性和鲁棒性：

1. **大规模模拟控制任务（IsaacLab）**
   - 包括：`Ant`, `Humanoid`, `Open-Drawer`, `Anymal-D-Rough`, `Go2-Rough`, `G1-Rough`, `H1-Rough`, `Digit-LocoManip`
   - 动作维度从 8 到 128 不等，覆盖 locomotion、manipulation 和 whole-body control。

2. **模仿到强化学习的在线微调（Robomimic）**
   - 任务：`Can`, `Box Cleanup`, `Threading`
   - 方法：从预训练的 flow-matching 策略出发，进行 online RL fine-tuning。
   - 评估 zero-noise 和 random-noise 采样下的成功率。

3. **真实世界灵巧手操作任务（RobotEra Xhand）**
   - 任务：旋转并松开不同几何形状的螺母（nut-bolt）
   - 验证 sim-to-real 能力，涉及接触动力学、硬件噪声等现实挑战。

### **实验设置与评估指标**

- **评估指标**：
  - 平均 episodic return（主要性能）
  - 收敛速度（learning curves）
  - 训练时间（wall-clock time）
  - 成功率（fine-tuning 任务）
  - KL 散度监控（用于 adaptive learning rate）

- **统一实现框架**：
  - 所有方法基于 **RSL-RL** 框架，在 **IsaacLab** 中运行。
  - 保持相同的环境配置、critic 架构、rollout 协议、mini-batch 构建方式，确保公平比较。

- **消融参数**：
  - 测试不同 history coefficient $ \theta \in [0.5, 0.65, 0.7, 0.75, 0.8, 0.85] $ 对性能的影响。

### **基线方法对比**

| 方法 | 类型 | 关键特点 |
|------|------|----------|
| **PPO (Gaussian)** | 基线 | 标准高斯策略，closed-form likelihood |
| **FPO** | Generative (approximate) | 使用 ELBO 替代 likelihood ratio |
| **GenPO** | Generative (exact) | 使用 dummy action 实现精确逆变换 |
| **PolicyFlow** | Generative (approximate) | 通过 velocity variation 近似重要性权重 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据与对比结果**

#### **(1) IsaacLab 大规模控制任务（图 3 & 图 11）**

- **性能表现**：
  - GenPO++ 在所有 8 个任务上达到 **competitive 或 superior 的 episodic return**。
  - 显著优于 FPO 和 PolicyFlow，尤其在高维任务（如 Humanoid）上 FPO 出现崩溃（collapse）现象。

- **训练效率（表 1 & 图 11）**：
  - 在 Humanoid 任务上，总训练时间对比：
    - PPO: **13.25 min**
    - FPO: 72.06 min
    - GenPO: **132.30 min**
    - **GenPO++: 20.78 min**
  - GenPO++ 比 GenPO 快 **6 倍以上**，接近 PPO 水平，远超其他生成式方法。

- **wall-clock time 对比（图 11）**：
  - GenPO++ 在更短的实际时间内取得更高回报，证明其 **计算效率优势显著**。

#### **(2) Robomimic 微调任务（图 5）**

- 所有任务上 GenPO++ 均能有效提升预训练 flow policy 的性能：
  - `Can`: 保持 high zero-noise 性能，random-noise 快速上升。
  - `Box`: 快速收敛至高成功率。
  - `Threading`: 最终性能最佳，zero 和 random noise 下均领先。
- 表明 **精确 likelihood ratio 有助于稳定且可靠的随机策略改进**。

#### **(3) 真实世界灵巧手操作（图 6 & 图 7）**

- **仿真训练阶段**：GenPO++ 比 PPO 更快提升 reward，最终性能更高。
- **真实部署成功**：策略能适应不同几何螺母，完成旋转松脱任务。
- 验证了 GenPO++ 在 **sim-to-real 场景下的鲁棒性和实用性**。

### **消融实验结果（图 4）**

- 测试不同 $ \theta $ 值（history coefficient）对 Ant 任务的影响。
- 结果显示：GenPO++ 在 $ \theta \in [0.5, 0.85] $ 范围内性能稳定，表明方法对超参变化具有较强鲁棒性。
- 但极端值可能导致数值不稳定或偏离原始 flow 轨迹（见结论部分）。

---

## 4. 关键结论和发现

### **主要发现**

1. **精确 likelihood ratio 至关重要**：使用 ELBO 等近似会导致更新偏差，影响高维任务的训练稳定性（FPO 表现差）。
2. **dummy action 不必要且低效**：GenPO 虽然精确，但引入的额外维度和 Jacobian 计算严重拖慢训练。
3. **历史状态可作为天然辅助变量**：利用 ODE 求解器的历史信息即可实现可逆性，无需扩展动作空间。
4. **Jacobian-free 是效率关键**：log-determinant 由固定系数决定，避免了昂贵的神经网络 Jacobian 计算，大幅提升训练速度。
5. **兼容性好，适合 fine-tuning**：不改变原始动作表示，可直接用于微调预训练 diffusion/flow 策略。

### **方法的局限性**

- 引入了一个新的超参数 $ \theta $（history coefficient）：
  - 若 $ \theta $ 过大，更新可能偏离原始 flow 过程，扭曲策略分布；
  - 若 $ \theta $ 过小，可能导致数值不稳定或可逆性下降。
- 当前方法依赖于特定形式的高阶求解器设计，推广到其他 solver 形式需进一步研究。

### **未来工作方向**

- 设计 **自适应调整 $ \theta $ 的机制**，以平衡精度与稳定性。
- 探索更优的可逆求解器结构，降低对 $ \theta $ 的敏感性。
- 将 GenPO++ 扩展到 off-policy 设置或其他生成模型（如 diffusion policy）中。
- 进一步优化推理延迟，推动在实时机器人系统中的应用。

---

> ✅ **总结一句话**：  
> **GenPO++ 通过将 ODE 求解器的历史状态作为辅助记忆，构建了一个无需 dummy action、无需 Jacobian 计算、保持原动作维度的可逆生成策略框架，在保证精确 likelihood ratio 的同时大幅提升了训练效率与稳定性，是 on-policy 生成式强化学习的重要进展。**

</details>

---

### 7. [$\alpha$-PFN: Fast Entropy Search via In-Context Learning](https://arxiv.org/abs/2606.07134)

**Authors**: Herilalaina Rakotoarison, Steven Adriaensen, Tom Viering, Carl Hvarfner, Samuel M\"uller, Frank Hutter, Eytan Bakshy  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.07134v1  

#### Abstract
Information-theoretic acquisition functions such as Entropy Search (ES) offer a principled exploration-exploitation framework for Bayesian optimization (BO). However, their practical implementation relies on complicated and slow approximations, i.e., a Monte Carlo estimation of the information gain....

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：$\alpha$-PFN: Fast Entropy Search via In-Context Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **Entropy Search (ES)** 及其变体（如 **PES**, **MES**, **JES**）虽然在理论上提供了信息论上最优的探索-利用权衡框架，但其实际应用受限于复杂的、基于采样的近似方法（如 Monte Carlo 估计），导致：
- 计算开销大，运行缓慢；
- 实现复杂，需要手工设计近似方案；
- 难以扩展到高吞吐或大规模优化场景。

此外，现有方法在 Fully Bayesian 设置下（即对 GP 超参数进行积分）计算更加昂贵。

### ✅ 提出的新方法与新思路
作者提出 **$\alpha$-PFN**（alpha-Prior-data Fitted Network），一种基于 **Prior-data Fitted Networks (PFNs)** 的两阶段摊销策略，用于快速近似 Entropy Search 类型的采集函数。

#### 核心思想：
1. **第一阶段（Base PFN）**：
   - 训练一个基础 PFN 模型，使其能够预测在给定最优值 $f^*$、最优位置 $x^*$ 或两者联合条件下的后验预测分布（PPD）。
   - 这个模型通过大量从 GP prior 中采样的合成数据训练得到。

2. **第二阶段（$\alpha$-PFN）**：
   - 利用 Base PFN 的输出作为“oracle”，训练另一个 PFN 直接预测 **Expected Information Gain**（即 PES/MES/JES 的采集值）。
   - $\alpha$-PFN 在推理时仅需一次前向传播即可完成采集函数评估，无需任何采样或迭代优化。

#### 技术亮点：
- 将原本依赖 MC 采样和手动近似的复杂过程，转化为一个可学习、端到端的神经网络推理任务。
- 利用 **in-context learning** 特性，在测试时直接根据当前观测数据 $D_{\text{trn}}$ 和候选点 $x$ 输出采集值。
- 支持 Fully Bayesian GP 设置下的自然集成（uncertainty over hyperparameters 被自动建模）。

### ✅ 相比现有方法的优势
| 维度 | 传统 ES 方法（PES/MES/JES） | $\alpha$-PFN |
|------|-------------------------------|-------------|
| **速度** | 慢（依赖 MC 采样，每步数百次模拟） | 极快（单次前向传播） |
| **实现复杂度** | 高（需设计采样策略、近似技巧） | 低（纯神经网络推理） |
| **可扩展性** | 有限（难以并行化） | 易于部署和扩展 |
| **Fully Bayesian 支持** | 复杂且低效（需 MCMC） | 自然支持，无额外开销 |
| **通用性** | 仅适用于特定先验 | 可适配不同 prior（训练后） |

---

## 2. 核心实验方法和设置

### 📚 数据集
实验覆盖两类典型黑盒优化基准：

#### （1）合成函数（Synthetic Functions）
- **Branin** (2D)
- **Hartmann** (4D, 6D)
- **Ackley** (5D, 8D)

> 所有函数加入噪声（$\sigma_n = 0.316$），共 30 次随机种子，每次 100 轮 BO。

#### （2）真实世界超参优化任务（HPO）
- **LCBench**：7D 代理任务（car, FashionMNIST, MiniBooNE, higgs, segment）
- **HPO-B**（修正版 FixedHPO-B）：高维搜索空间（8D–16D），共 5 个任务（ID=5527 至 5971）

> LCBench 使用 30 种子 × 100 迭代；HPO-B 使用 5 种子 × 50 迭代。

---

### 🧪 实验设置与评估指标

#### ✅ 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **Inference Regret** | $f(x^*) - f(\hat{x}^*)$，其中 $\hat{x}^* = \arg\max_x \mathbb{E}[y \mid D, x]$ | 衡量最终推荐解的质量（越小越好） |
| **Accuracy / Predicted Best Performance** | 在 LCBench 上报告预测最优性能 | 更直观的性能衡量 |
| **Average Ranking** | 在 HPO-B 上按推理遗憾排序后取平均排名 | 综合比较多个任务表现 |
| **Runtime (Speedup)** | 累计优化耗时（分钟），计算相对于基线的加速比 | 核心效率指标 |

#### ✅ 基线方法对比
| 方法 | 描述 |
|------|------|
| **GP-MCMC (NUTS)** + JES/MES/PES | BoTorch 实现的标准 Fully Bayesian 版本，使用 HMC 对超参数采样 |
| **MCMC-ES** | 替代 Fully Bayesian 基线（非完全贝叶斯模型本身） |
| **EI (Expected Improvement)** | 经典采集函数，作为参考基线 |

> 所有方法共享相同的 GP prior 分布，确保公平比较。

#### ✅ 模型架构与训练细节
- 使用 **TabPFNv2** 架构（Hollmann et al., 2025），改进原始 PFN 对维度变化的支持。
- Base PFN 和 $\alpha$-PFN 均为约 4.7M 参数的 Transformer。
- 训练数据：1亿条从 Fully Bayesian GP prior 生成的数据（d=1–6），使用 RFF 近似 GP 并预计算 $x^*, f^*$。
- 训练资源：Base PFN ~13h on 4×H200；每个 $\alpha$-PFN ~16h on 4×L40S。

---

## 3. 主要实验结果和性能指标

### 🔢 关键性能数据（见 Table 1 与 Figure 3）

#### （1）优化性能（Optimization Quality）
| 方法 | 合成函数 | LCBench | HPO-B |
|------|--------|---------|-------|
| $\alpha$-PFN-JES | ✅ 最佳或接近最佳 | ✅ 全面优于 GP-JES | ⚠️ 多数任务略差于 GP，但排名相近 |
| $\alpha$-PFN-MES | ✅ 匹配或稍优 | ✅ 在 Hartmann(6D) 明显领先 | ❌ 在 Ackley(8D) 性能下降 |
| $\alpha$-PFN-PES | ✅ 整体竞争力强 | ✅ 多数任务优于 GP-PES | ⚠️ 表现稳定但未显著超越 |

> 总体来看，$\alpha$-PFN 在大多数任务中与 GP 基线 **性能相当甚至更优**，尤其在中等维度（4–8D）表现稳健。

#### （2）运行时间与加速效果（Speedup）
| 方法 | 加速倍数范围 | 典型加速（HPO-B） |
|------|--------------|------------------|
| MES-$\alpha$-PFN | 1.6× – 65× | >30×（常见） |
| JES-$\alpha$-PFN | 3.4× – 58.7× | >30× |
| PES-$\alpha$-PFN | 5.5× – 72.4× | >30× |

> **最高达 72× 的加速**，尤其在高维 HPO-B 任务上优势明显。

#### （3）消融实验结果（Ablation Studies）

##### ✅ **Trace Generation Ablation（图5）**
- 使用 **聚类式轨迹生成**（Algorithm 1） vs. 均匀采样
- 发现：随着维度升高（>4D），聚类轨迹显著提升性能
- 结论：训练分布需模拟真实 BO 的查询模式，避免 domain shift

##### ✅ **OOD 噪声鲁棒性测试（图4）**
- 测试更高噪声水平（$\sigma_n = 0.5$ vs. 训练时 0.316）
- 结果：$\alpha$-PFN 与 GP 基线均退化，但退化趋势一致
- 结论：$\alpha$-PFN 对 OOD 噪声具有合理鲁棒性，无额外失败模式

##### ✅ **Qualitative Comparison（图6）**
- 在 1D 设置下可视化 JES 采集函数
- 发现：$\alpha$-PFN 输出平滑、峰值对齐良好，且比 MC 估计更稳定（减少方差）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **$\alpha$-PFN 成功实现了 Entropy Search 的高效摊销**：
   - 首次将 PES/MES/JES 的复杂采样流程完全替换为单次前向传播。
   - 在保持与 GP 基线相当甚至更优的优化性能的同时，实现 **超过 50× 的加速**。

2. **信息增益可被有效学习**：
   - 通过两阶段训练策略，$\alpha$-PFN 学会了从上下文中直接预测期望信息增益分布，其均值即对应标准 ES 变体。

3. **天然支持 Fully Bayesian 设置**：
   - 不需要在推理时重复采样超参数，uncertainty 已在训练中被“编译”进模型。

4. **训练数据分布至关重要**：
   - 使用模拟 BO 轨迹（而非均匀采样）能显著改善高维泛化能力。

---

### ⚠️ 局限性
1. **训练成本高昂**：
   - 需要预先生成上亿条 GP 样本，并训练多个大型 PFN 模型（总耗时 GPU-day 级别）。
   - 当前为一次性投资，适合长期复用场景。

2. **泛化能力受限于训练 prior**：
   - 若目标任务严重偏离训练所用的 GP prior（如非平稳、多峰强烈），性能可能下降。
   - 当前训练维度上限为 6D，虽能外推至 16D，但仍存在风险。

3. **无法灵活更换 prior**：
   - 每换一个 prior 都需重新训练整个 $\alpha$-PFN。
   - 缺乏在线适应机制（尽管 Whittle et al., 2026 提供潜在解决方案）。

---

### 🔮 未来工作方向
1. **开发更通用的 prior-agnostic 或自适应 PFN 架构**（如结合 Distribution Transformers）。
2. **扩展至其他基于 MC 的采集函数**（如 Thompson Sampling, Info-Theoretic ParEGO）。
3. **进一步优化推理效率**（如量化、蒸馏）以支持实时嵌入式系统。
4. **探索在多目标、约束、多保真度等复杂 BO 场景中的应用**。
5. **构建开源 $\alpha$-PFN 库**，降低使用门槛。

---

## 💡 总结一句话
> $\alpha$-PFN 通过 **in-context learning + 两阶段摊销训练**，首次实现了 **Entropic Acquisition Functions 的零采样、单次前向快速推理**，在不牺牲性能的前提下带来 **数十倍的速度提升**，为高通量、大规模贝叶斯优化开辟了新路径。

🔗 开源代码：[https://github.com/automl/AlphaPFN](https://github.com/automl/AlphaPFN)

</details>

---

### 8. [Sparsely gated tiny linear experts](https://arxiv.org/abs/2606.07414)

**Authors**: Simon Schug  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.07414v1  

#### Abstract
Sparsity allows scaling model parameters without proportionally increasing computational cost. While mixture of experts (MoE) models are made increasingly sparse, individual experts typically remain large and dense. Here, we demonstrate that further increasing sparsity by shrinking each expert to co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Sparsely gated tiny linear experts》核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大规模语言模型中的 **Feedforward 层**在计算效率和可解释性方面面临挑战：
- **计算成本高**：传统密集型 Feedforward 层（如 MLP、SwiGLU）参数量大，且每次前向传播都激活全部参数，导致计算开销随参数增长呈线性上升。
- **可解释性差**：非线性激活函数（如 GeLU、ReLU）使得神经元行为复杂，难以用线性代数工具分析；同时，神经元通常具有多义性（polysemantic），阻碍了对模型内部机制的理解。

尽管 **Mixture of Experts (MoE)** 引入了稀疏性以提升计算效率，但其专家（expert）仍为大型子网络，每 token 激活数十亿参数，稀疏程度有限。

### 提出的新方法：sgatlin
本文提出 **sparsely gated linear neurons (sgatlin)**，一种全新的 Feedforward 层架构，核心思想是将每个 expert 极致缩小至单个线性神经元，并通过稀疏门控选择少量神经元进行组合。

#### 主要创新点：
- **极细粒度专家**：每个 expert 是一个无偏置的线性变换 $ w^T z $，即单个神经元。
- **稀疏门控机制**：引入高效的 `product-top-k` 路由器，从海量神经元中选出 $ k $ 个最相关的进行加权组合。
- **移除非线性**：**关键洞见**是移除专家输出和门控权重上的非线性激活函数（如 ReLU、GeLU）。这使得在给定门控权重后，整个有效电路变为**纯线性变换**。
- **通道并行设计**：采用多个并行通道（parallel channels），共享查询投影但各自拥有独立的专家参数，进一步增加容量而不显著增加计算负担。

### 相比现有方法的优势
| 特性 | Dense (MLP/SwiGLU) | Coarse-grained MoE | Fine-grained MoE (PEER) | **sgatlin** |
|------|---------------------|--------------------|--------------------------|------------|
| 参数/FLOP 比率 | 低 | 中等 | 高 | **极高** |
| 激活参数量 | 全部 | 数个大专家 | 多个小专家 | **极少数单神经元** |
| 可解释性 | 差（非线性、密集） | 中等（稀疏但仍有非线性） | 较好 | **优秀（稀疏+线性）** |
| 计算扩展性 | 线性 | 次线性 | 次线性 | **次线性，更高效** |

> ✅ **优势总结**：sgatlin 在保持甚至提升语言建模性能的同时，实现了更高的**计算稀疏性**（compute sparsity），并因其**稀疏线性结构**而天然具备更强的**可解释性潜力**。

---

## 2. 核心实验方法和设置

### 数据集
- **主实验（语言建模）**：`SlimPajama-627B`，一个包含 6270 亿 token 的清洗版文本数据集，用于训练和评估不同规模的语言模型。
- **可解释性研究**：`TinyStories`，一个专为小模型设计的简单故事数据集，用于在可控环境下深入分析 sgatlin 的内部机制。

### 实验设置
- **模型架构**：Decoder-only Transformer，所有 Feedforward 层被替换为待比较的变体。
- **上下文长度**：2048 tokens。
- **批量大小**：128 sequences。
- **优化器**：AdamW，学习率调度为 warmup-stable-decay。
- **评估指标**：
  - **语言建模性能**：测试集 **Perplexity**（困惑度）。
  - **计算效率**：在 **isoflop**（相同 FLOPs 预算）条件下比较不同模型大小下的性能。
  - **可解释性**：通过 UMAP 可视化门控权重、最近邻搜索、因果干预（causal intervention）来分析电路功能。

### 基线方法对比
- **Dense Feedforward**：
  - `MLP`：标准 GeLU 激活的全连接层。
  - `SwiGLU`：当前主流的门控线性单元。
- **Mixture of Experts (MoE)**：
  - `MoE`：粗粒度 MoE（如 GPT-OSS 中使用，16 个大专家）。
  - `PEER`：细粒度 MoE，专家为单个神经元，是 sgatlin 最接近的基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **图 3 显示**：在四个不同的 FLOPs 预算下（$1\times10^{17}$ 到 $6\times10^{18}$），sgatlin 在同等计算预算下，**困惑度（perplexity）优于或至少不逊于所有基线方法**。
- **图 1 显示**：随着 **compute sparsity**（参数/FLOP 比率）的提高，最终模型性能普遍提升。**sgatlin 实现了最高的 compute sparsity**，表明其能最有效地利用参数扩展模型容量。
- **结论**：在 isoflop 设置下，**sgatlin 是最具计算效益的 Feedforward 层设计之一**。

### 消融实验结果（Ablation Study）
在最高计算预算（$6\times10^{18}$ FLOPs）下对 sgatlin 进行消融：

| 变体 | Test Perplexity |
|------|-----------------|
| sgatlin (原版) | **18.1910** |
| + ReLU 激活 | 21.0880 |
| + GeLU 激活 | 20.8445 |
| + Swish 激活 | 20.6424 |
| 使用 PEER 的路由器 | 18.2964 |

#### 消融结论：
1. **移除非线性至关重要**：在专家上添加任何非线性激活函数（ReLU、GeLU、Swish）都会**显著降低性能**。这表明 `top-k` 门控本身已足以提供必要的非线性表达能力。
2. **门控设计高效**：即使将 sgatlin 的路由器替换为 PEER 的版本，性能也未提升，说明 sgatlin 当前的 `product-top-k` 设计在效率和效果上已足够优秀。

---

## 4. 关键结论和发现

### 主要发现
1. **稀疏性与线性可以协同增效**：极致的稀疏性（单神经元专家）结合线性处理，不仅能维持高性能，还能超越传统非线性设计。
2. **sgatlin 更高效**：在相同 FLOPs 预算下，sgatlin 能实现更低的困惑度，得益于其极高的 **compute sparsity**。
3. **可解释性显著增强**：
   - **语义聚类**：通过 UMAP 对门控权重降维，发现不同输入 token 激活的电路在嵌入空间中形成语义聚类（如名字、代词、标点符号各自成簇）。
   - **功能相似性**：最近邻搜索显示，相似的电路处理语义相关的内容（如“Max”与“Spot”、“Lily”等名字的电路相近）。
   - **因果作用**：通过干预门控权重（patching），验证了特定 Feedforward 电路在事实回忆（如“The [noun] was [adjective]”）中的因果作用，尤其是在第二层最为显著。

### 方法的局限性
- **仅优化 Feedforward 层**：注意力机制等其他组件仍为密集计算，整体模型的稀疏性和可解释性受限于此。
- **硬件适配性**：当前硬件（如 GPU/TPU）高度优化于密集计算，稀疏操作的实际推理速度可能不如理论 FLOPs 所示那般高效。
- **可解释性研究规模有限**：深度可解释性分析基于 `TinyStories` 上的小模型，需在更大模型上验证其普适性。

### 未来工作方向
- 将 sgatlin 的设计理念扩展到其他模型组件（如注意力机制）。
- 探索更高效的稀疏计算硬件支持，以充分发挥其推理优势。
- 开发基于 sgatlin 线性电路特性的新型可解释性工具，无需依赖额外的稀疏自编码器（SAE）等代理模型。
- 研究如何利用其结构化的电路空间进行模型编辑、知识注入或安全控制。

> 💡 **总体评价**：sgatlin 提供了一条通往**兼具计算高效性与内在可解释性**的大模型架构的可行路径，其“稀疏门控 + 线性专家”的设计思想具有重要的启发意义。

</details>

---

### 9. [Improving Cross-Lingual Factual Recall via Consistency-Driven Reinforcement Learning](https://arxiv.org/abs/2606.06586)

**Authors**: Jonathan von Rad, Louis Arts, George Burgess, Eleftheria Kolokytha, Harry O'Donnell, Ektor Oikonomidis Doumpas, Eduardo Sanchez, Yao Lu, Pontus Stenetorp  
**Category**: cs.CL  
**Published**: 2026-06-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.06586v1  

#### Abstract
Large language models (LLMs) trained predominantly on English data encode substantial world knowledge, yet often fail to express it reliably in other languages, a phenomenon known as cross-lingual factual inconsistency. To study and address this, we introduce PolyFact, a large-scale parallel multili...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Improving Cross-Lingual Factual Recall via Consistency-Driven Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于 **cross-lingual factual inconsistency**（跨语言事实不一致性）问题：  
尽管大型语言模型（LLMs）在英文数据上训练并编码了大量世界知识，但在非英语语言中表达这些知识时常常失败。这种现象并非由于“缺乏知识”，而是模型在从共享表示向目标语言转换的过程中出现故障。

### 🚀 提出的新方法与创新思路
作者提出了一种基于 **consistency-driven Reinforcement Learning (RL)** 的新范式，通过以下三个核心贡献推进该领域：

1. **引入 POLYFACT 数据集**  
   - 构建了一个大规模、完全平行的多语言事实问答数据集，包含 **100K 条基于 Wikidata 的事实**，覆盖 **12 种类型多样、资源高低不同的语言**（如 en, zh, ar, sw, bn 等）。
   - 所有事实以多选题形式呈现，并确保跨语言完全对齐，用于系统研究跨语言一致性。

2. **提出 GRPO-based RL 方法用于提升跨语言事实回忆**  
   - 采用 **Group Relative Policy Optimization (GRPO)** 进行后训练（post-training），设计了一个鼓励跨语言一致性的奖励函数：
     $$
     R = \frac{1}{L}\sum_{l=1}^{L} r_e + \mathbb{I}[r_e=1]
     $$
     其中 $r_e$ 是单语言正确性奖励，最后一项是当所有语言都答对时的 bonus，显式激励模型输出跨语言一致的答案。

3. **分离表示对齐与知识访问机制**  
   - 提出一个关键假设：**continual pretraining (CPT)** 主要改善翻译流畅性和表示对齐，但不足以提升跨语言知识访问能力。
   - 因此，应将“表示对齐”与“跨语言知识检索”解耦，后者更适合通过任务导向的 post-training（尤其是 RL）来优化。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | 本文优势 |
|------|--------|---------|
| **CPT**（持续预训练） | 资源消耗大，易导致灾难性遗忘，仅提升表面流畅性而非深层知识访问 | 本文证明轻量级 CPT 改进有限，且可能损害原有性能 |
| **SFT**（监督微调） | 易导致“表面记忆”（surface-level memorization），泛化差，无法有效迁移至自由生成任务 | GRPO 显著优于 SFT，在 KLAR 和 Global-MMLU 上实现更强泛化 |
| **传统 RL/SFT 结合方式** | 多用于单语推理一致性 | 首次验证 consistency-driven RL 在跨语言场景下的有效性 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 描述 | 用途 |
|-------|------|-----|
| **POLYFACT** | 自建数据集，100K 平行 MCQ，12 语言，基于 Wikidata，含验证标签和质量分级（POLYFACT-CLEAN） | 主要训练与评估数据 |
| **KLAR** | 已有跨语言事实问答基准，支持自由文本生成（free-form generation），包含 6 个训练语言 + 11 个 OOD 语言 | 测试跨语言迁移能力 |
| **Global-MMLU** | 多语言版 MMLU，涵盖多步推理与领域知识，更具挑战性 | 评估更广泛的知识与推理泛化能力 |

> 💡 注：POLYFACT 的构造流程包括：
> - 从 Wikidata 抽取 truthy triples
> - 选择 22 类稳定关系（如首都、出生地、创作者等）
> - 多语言标签提取 + 平衡采样防止高频关系主导
> - 使用 Gemma-3-27B-IT 生成平行 MCQ
> - 经 GPT-4o judge 与人工审核保证质量（LLM-human agreement 达 91%）

### ⚙️ 实验设置与评估指标

#### 模型
- **主干模型**：`Qwen-2.5-7B` 和 `OLMo-2-1124-7B`
- 后者为英语主导模型，前者具备较强多语言基础能力，形成对比。

#### 训练策略（六种变体）
| 变体 | 描述 |
|------|------|
| Base | 原始模型 |
| CPT | 在 TED2025 多语平行语料上继续预训练（235.5M tokens） |
| SFT | 在 POLYFACT 上进行监督微调（带一致性正则项） |
| GRPO | 在 POLYFACT 上进行 GRPO 强化学习 |
| CPT + SFT / CPT + GRPO | 先 CPT 再 SFT 或 GRPO |

#### 微调细节
- **LoRA**：r=64, alpha=128
- **GRPO 设置**：每组 rollout 包含 G=8 组 × L=12 语言独立生成；reward 设计包含正确性、幻觉惩罚、全对 bonus
- **SFT 目标函数**：联合分类损失 + KL 散度一致性正则（$\lambda=0.5$）

#### 评估指标
- **Accuracy (%)**：按语言分组报告 POLYFACT、KLAR、Global-MMLU 性能
- **Cross-lingual transfer**：在未见语言（held-out languages）上的表现
- **Mechanistic interpretability**：
  - **LAHIS**：分析 attention heads 的语言特异性变化
  - **LAPE**：识别 MLP 层中的语言专用神经元分布

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 方法 | POLYFACT (High) | KLAR (Trained) | Global-MMLU (High) |
|------|------------------|----------------|--------------------|
| **OLMo-2-1124-7B** | | | |
| Baseline | 57.93 | 24.6 | 38.72 |
| CPT | 57.89 | 17.0 | 37.41 |
| SFT | 56.33 | 18.1 | 35.40 |
| GRPO | **64.21**↑↑ | **29.0**↑↑ | **39.22**↑ |
| CPT + GRPO | 61.26 | 29.8 | 36.34 |
| **Qwen-2.5-7B** | | | |
| Baseline | 66.69 | 47.68 | 68.15 |
| GRPO | **73.15**↑↑ | **49.69**↑↑ | **68.35**↑ |

> ✅ **GRPO 在所有任务上均显著优于 SFT 和 CPT**

### 🔁 与基线方法的对比结果

| 对比维度 | 发现 |
|--------|------|
| **vs SFT** | GRPO 在 KLAR 上大幅提升（+10.9 vs OLMo），而 SFT 甚至退化；表明 SFT 学会的是选项匹配而非真实知识访问 |
| **vs CPT** | CPT 单独使用时多数情况下性能下降（尤其在 KLAR 和 Global-MMLU），说明单纯增加平行数据不能解决知识访问瓶颈 |
| **GRPO 是否依赖 CPT？** | 不依赖。直接在 base model 上应用 GRPO 效果最好；CPT + GRPO 反而略有下降，说明 alignment 与 consistency learning 存在潜在冲突 |

### 🔍 消融实验与深入分析

#### ✅ 跨语言迁移能力（KLAR OOD）
- GRPO 在 **11 个未见语言**上仍取得明显增益（平均 +3.4 pts），而 SFT 几乎无提升
- 表明 GRPO 学到了可泛化的跨语言知识访问路径

#### ✅ 泛化到复杂任务（Global-MMLU）
- GRPO 成功恢复甚至超越 baseline 性能，而 CPT 和 SFT 导致退化
- 说明 GRPO 不仅改进事实回忆，还能增强更复杂的多语言推理能力

#### ✅ 机械可解释性分析（Mechanistic Interpretability）

| 分析工具 | 主要发现 |
|--------|--------|
| **LAHIS**（Attention Heads） | 
| | - Base 模型中约 50% 语言重要头集中在第 0 层 → 早期语言路由决策固化 |
| | - GRPO 将其分散至 0–10 层，减少早期语言专门化 |
| | - SFT 提高 head overlap（尤其印欧语系），GRPO 则促进远缘语言间共享（如 JA-ZH） |
| **LAPE**（MLP Neurons） |
| | - GRPO 导致 **language-specific neurons 向后延迟**（ECDF shift, Dks=0.089） |
| | - 英语专用 neuron 数量反增 38.2%，反映 RL “squeeze” 行为空间，强化最稳定的 backbone（即 English） |
| | - 非拉丁脚本语言（如 AR, ZH, JA）的语言专业化更多保留在最后几层 |

> 🧠 结论：GRPO 重构了语言处理流，推动中间层向语言无关表征发展，延迟语言专业化，从而提升跨语言一致性。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **跨语言事实不一致的本质不是缺知识，而是访问失败**  
   模型常在中间层正确检索答案，但在最终解码阶段未能将其可靠映射为目标语言。

2. **GRPO 是改进跨语言事实回忆的有效手段**  
   - 显著优于 SFT 和 CPT
   - 提升事实准确性、跨语言一致性、OOD 泛化能力和复杂推理表现
   - 适用于不同架构的 LLM（OLMo 与 Qwen）

3. **SFT 容易陷入“表面记忆”陷阱**  
   - 在 POLYFACT 上表现尚可，但在 KLAR（自由生成）上全面退化
   - 常见失败模式：输出 "1" 或 "2"，而非尝试作答

4. **轻量 CPT 并不能有效提升跨语言知识访问**  
   - 有时反而损害原有能力（如 KLAR 下降）
   - 表示对齐 ≠ 知识可访问性

5. **GRPO 重塑内部语言路由机制**  
   - 推迟语言专业化（delayed specialization）
   - 减少 attention head 的语言隔离
   - 增强共享跨语言计算路径

### ⚠️ 局限性

1. **模型规模限制**：仅在两个 7B 规模模型上验证，是否推广至更大/小模型或 MoE 架构未知。
2. **任务范围有限**：主要针对事实回忆，对需要深度推理的任务（如 Global-MMLU）提升有限。
3. **语言覆盖不足**：仅 12 种语言 + 11 OOD，难以代表全球语言多样性。
4. **数据偏见风险**：
   - POLYFACT 源自 Wikidata，存在英语中心主义倾向（如 proper nouns 默认英文）
   - 导致 GRPO 在 proper-noun 问题上出现“English leak”现象（答对但用错语言）

### 🔮 未来工作方向

1. **构建更公平的多语言训练数据**  
   - 替代 Wikidata 中的英语默认标签
   - 引入 native proper nouns，缓解语言泄露问题

2. **探索 partial credit 奖励机制**  
   - 当前 GRPO 奖励为 binary（全对才加分）
   - 可设计中间状态奖励（如：内容正确但语言错误 → 部分得分）

3. **扩展至其他跨语言任务**  
   - 如 multilingual RAG、对话系统、摘要生成等

4. **结合 CPT 与 RL 的协同优化策略**  
   - 当前 CPT + GRPO 效果不如单独 GRPO
   - 可研究如何让 alignment 与 consistency learning 相互促进

---

> 📢 **一句话总结**：  
> 本文揭示了跨语言事实回忆的关键瓶颈在于“知识访问”而非“知识缺失”，并通过 **consistency-driven GRPO** 成功提升了模型在多种语言中一致、准确、泛化地表达已有知识的能力，同时揭示了其背后深层的机制重构效应。

</details>

---

### 10. [TabSwift: An Efficient Tabular Foundation Model with Row-Wise Attention](https://arxiv.org/abs/2606.07345)

**Authors**: Si-Yang Liu, Han-Jia Ye  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.07345v1  

#### Abstract
Tabular foundation models, exemplified by TabPFN, perform prediction via in-context learning, inferring test labels directly from labeled training examples. They have demonstrated competitive performance, particularly on small-to-medium datasets. However, recent tabular foundation models often impro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **TabSwift: An Efficient Tabular Foundation Model with Row-Wise Attention**  
—— 核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前的 **Tabular Foundation Models (TFMs)** 虽然在小到中等规模的表格数据上表现出色，但其架构日益复杂（如引入 column-wise attention、交替行/列注意力机制），导致推理成本显著上升，限制了在延迟敏感场景下的实际部署。

此外，传统 TFM 推理时对所有样本均执行完整深度计算，造成“简单样本”浪费算力的问题。

### **提出了什么新方法或新思路**

本文提出 **TABSWIFT**，一个轻量级、高效的表格基础模型，核心思想是：**在保持 row-wise attention-only 架构的前提下，通过现代训练技巧实现竞争力性能**。

具体创新点如下：

- **轻量化稳定架构设计**：
  - **Gated Attention (G1)**：在 SDPA 输出端引入逐元素门控机制，提升预训练稳定性。
  - **Learnable Register Tokens**：引入少量可学习的全局寄存器 token，用于聚合任务级上下文信息，增强模型对全局模式的捕捉能力。

- **统一的分类与回归支持**：
  - 设计了一个共享 backbone 的双头输出结构（classification MLP head + regression MLP head），实现单模型支持两种任务。

- **自适应层早退机制（Adaptive Layer-wise Early Exit）**：
  - 在最后若干层附加预测头和可靠性判断门控（exit head）。
  - 每个样本动态决定是否提前退出，实现 **anytime inference**，降低平均推理开销。

### **相比现有方法的优势**

| 维度 | 优势 |
|------|------|
| **效率** | 显著低于 TabPFN v2、TabICL 等 column-aware 模型的推理延迟（见图1、图6） |
| **性能** | 在 TALENT 基准上达到与更强 TFM 相当甚至更优的平均排名 |
| **灵活性** | 支持分类与回归统一建模，且具备 per-sample 自适应计算能力 |
| **实用性** | 更适合在线、低延迟服务场景 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **主基准**：**TALENT Benchmark**
  - 包含 **300 个公开表格数据集**（200 分类 + 100 回归）
  - 来源多样，涵盖医疗、金融、科学等领域
  - 数据划分：64% 训练 / 16% 验证 / 20% 测试，报告 15 次随机种子的平均结果

- **额外分析数据集**：
  - 高维数据集（如 `gisette`, `SMK_CAN_187`，维度高达 ~20k）
  - 大样本分类数据集（>20,000 行）

### **实验设置和评估指标**

| 项目 | 设置 |
|------|------|
| **评估协议** | In-context learning (ICL)：使用支持集 + 查询样本进行零样本预测 |
| **分类指标** | **AUC**（越大越好） |
| **回归指标** | **RMSE**（越小越好）、**R²**（越大越好） |
| **效率指标** | **平均推理时间**（wall-clock time）、**平均执行层数**（early exit） |
| **统计检验** | Wilcoxon-Holm 校正下的成对显著性测试，绘制 **Critical Difference (CD) Diagram** |

### **基线方法对比**

分为三类进行比较：

1. **经典机器学习方法**：
   - SVM, KNN, Random Forest, XGBoost, LightGBM, CatBoost

2. **深度学习模型**：
   - FT-Transformer, TabNet, NODE, RealMLP, TabM, DANets, ExcelFormer 等

3. **Tabular Foundation Models (TFMs)**：
   - **TabPFN v1/v2**, **TabICL**, **LimiX**, **LocalPFN**

> 所有 TFM 均使用官方 checkpoint，并控制 ensemble size（如 TabPFN v2 用 4，TABSWIFT 用 16）以公平比较效率。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **总体性能（图4）**

- **分类任务**：
  - TABSWIFT 平均排名 **优于或等于 TabPFN v2 和 TabICL**
  - 在 CD 图中与最强方法无显著差异（bar 连接）
  - **PAMA 曲线显示其在更多数据集上取得最优性能**

- **回归任务**：
  - 同样表现强劲，平均排名领先多数基线
  - 特别是在中小样本回归任务上优势明显

#### ✅ **效率对比（图1、图6）**

- **推理速度**：
  - 在相同硬件（A800 GPU）下，**TABSWIFT 推理速度显著快于 TabPFN v2 和 TabICL**
  - 即使使用 **4 倍于 TabPFN v2 的 ensemble 数量（16 vs 4）**，仍保持更快

- **模型大小与延迟权衡（图1）**：
  - TABSWIFT（橙色标记 “3”）接近理想点（*），在 **更低推理延迟下达到相近甚至更优性能**

#### ✅ **消融实验（图4 ablation）**

逐步添加组件的效果验证：

| 变体 | 描述 | 性能变化 |
|------|------|----------|
| **TabS-S1** | 原始 row-wise attention-only backbone | 基线 |
| **+ Gate** | 加入 gated attention | 性能提升，训练更稳定 |
| **+ Register** | 加入 learnable register tokens | 显著提升预训练质量与泛化能力 |
| **+ Gate + Register** | 两者结合 | 性能跃升 |
| **→ TABSWIFT** | 再加入两阶段预训练 | 达到最终高性能 |

> 结果表明：**gated attention 与 register tokens 是性能提升的关键驱动因素**

#### ✅ **自适应早退效果（图7、表1-3）**

- **早退策略有效性**：
  - 使用 learned exit head（基于 register conditioning）可在 **极小性能损失下大幅减少计算量**
  - 例如，在阈值 $ T=0.9 $ 时，平均仅执行约 **11.6 层（共24层）**，AUC 下降 <0.1%

- **register conditioning 的作用**：
  - 条件化 register summary 的 exit head（“w/ registers”）比无条件版本提供更优的 accuracy-compute 权衡

- **可视化解释（图5、图8）**：
  - 早退出现在类别边界清晰区域
  - 难样本（靠近决策边界）倾向于更深层数退出
  - 验证了机制的合理性与智能性

#### ✅ **高维与大样本场景（附录 D）**

| 场景 | 结果 |
|------|------|
| **高维特征（>100维）** | 使用 PCA 降维后仍具竞争力，平均排名第二（仅次于 TabPFN v2） |
| **大样本（>20k 行）** | 性能相对下降，非最优选择；推荐使用 RealMLP、ModernNCA 等非 ICL 方法 |

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **轻量级 row-wise attention 架构依然具有强大潜力**：
   - 无需引入复杂的 column-wise 或交替注意力，通过 **gated attention + register tokens** 即可实现与重型 TFM 相当的性能。

2. ✅ **效率与性能可以兼得**：
   - TABSWIFT 在保持高预测精度的同时，实现了显著更低的推理延迟，尤其适合部署在资源受限或延迟敏感环境。

3. ✅ **自适应早退机制有效**：
   - 实现了真正的 **per-sample dynamic computation**，为“anytime inference”提供了实用解决方案。

4. ✅ **统一建模范式可行**：
   - 单一 backbone 支持分类与回归，简化了模型管理与部署流程。

### **方法的局限性**

1. ❗ **依赖合成数据预训练**：
   - 性能受合成任务分布影响，在高度专业化领域可能表现不佳。

2. ❗ **不适合超大规模样本（>20k 行）**：
   - row-wise attention 的 $ O(N^2) $ 复杂度成为瓶颈，此时传统 GBDTs 或 MLP 更优。

3. ❗ **高维特征需 PCA 预处理**：
   - 虽然有效，但可能丢失部分语义信息，未来可探索更好的编码方式。

4. ❗ **未覆盖其他任务类型**：
   - 当前聚焦于分类与回归，未涉及 ranking、survival analysis、multi-label 等任务。

### **未来工作方向**

- 🔄 **扩展至更大规模 ICL 场景**：研究稀疏注意力、分块处理等技术以支持 >50k 行的数据。
- 🔍 **改进高维特征表示**：探索可学习的 feature selection 或 hierarchical encoding。
- 🧩 **多任务与元学习增强**：将 register tokens 用于任务识别或多任务路由。
- ⚙️ **硬件协同优化**：结合早退机制与编译优化，进一步压缩端到端延迟。

---

> **总结一句话**：  
> **TABSWIFT 证明了“少即是多”——通过精巧设计而非堆叠复杂度，即可构建高效、实用、高性能的表格基础模型，为现实世界部署提供了强有力的新选择。**

</details>

---

### 11. [Skip a Layer or Loop It? Learning Program-of-Layers in LLMs](https://arxiv.org/abs/2606.06574)

**Authors**: Ziyue Li, Yang Li, Tianyi Zhou  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.06574v1  

#### Abstract
Large language models (LLMs) perform inference by following a fixed depth and order, non-recurrent execution of all layers. We reveal the wide existence of training-free, flexible, dynamic program-of-layers (PoLar), where pretrained layers can be packed as modules and then skipped or looped to form ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Skip a Layer or Loop It? Learning Program-of-Layers in LLMs》总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 Large Language Models (LLMs) 在推理时采用固定的深度和顺序执行所有 Transformer 层，即标准前向传播（standard forward pass）。然而，不同输入的复杂度差异巨大，这种“一刀切”的计算模式可能导致：
- 对简单任务**过度计算**（over-compute），浪费资源；
- 对困难任务**计算不足**，导致错误。

本文提出并验证了一个核心假设：**一个预训练 LLM 中存在多种有效的“层程序”（Program-of-Layers, PoLAR）可以实现正确推理，而不仅仅是标准前向路径**。

### 提出了什么新方法或新思路
作者提出了 **PoLAR (Program-of-Layers)** 框架，其核心思想是将预训练的每一层视为可复用的函数模块，通过动态地**跳过（skip）** 或**循环执行（loop/repeat）** 连续的层段（segments），为每个输入定制专属的执行程序。

具体创新点包括：
- **Program-of-Layers 范式**：首次系统性地提出并实证了在不微调参数的前提下，通过组合预训练层形成多样化执行路径的可能性。
- **轻量级预测网络**：设计了一个轻量级的 **PoLAR Prediction Network**，用于在推理时直接预测最优的执行程序（segmentation + operation），避免了昂贵的在线搜索（如 MCTS）。
- **统一支持 Skip & Loop**：同时支持层跳过和循环，且操作作用于**连续层段**而非单个层，更具表达力和效率。

### 相比现有方法的优势
| 方法 | 局限性 | PoLAR 的优势 |
|------|--------|-------------|
| **Early Exit / Layer Skipping** (e.g., FastBERT, ShortGPT) | 只能提前终止，无法增加计算；可能降低准确率 | 可灵活减少或增加计算，尤其对难样本有效 |
| **Looped Transformers** (e.g., Universal Transformer) | 需要从头设计架构并训练 | 完全**test-time adaptation**，无需重新训练主模型 |
| **Dynamic Routing** (e.g., DR.LLM) | 逐层决策，缺乏全局协调；通常限于单层重复 | **程序级预测**，先生成完整计划再执行，支持多层段协同跳过/循环 |
| **Search-based 方法** (e.g., Li et al., 2025) | 依赖 MCTS 等搜索，计算开销大，不可扩展 | 用轻量预测网络替代搜索，**高效实用** |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **In-Distribution (ID)**:
  - **DART-Math**: 结构化数学推理基准，包含五个难度等级（DM-1 到 DM-5），用于训练和评估 PoLAR。
- **Out-of-Distribution (OOD)**:
  - **ASDiv**, **MAWPS**: 算术文字题数据集。
  - **MMLU-Pro**: 多领域语言理解基准，涵盖数学、物理、化学、法律、经济、生物、哲学等学科。

### 实验设置和评估指标
- **模型**：在多个冻结的预训练 LLM 上测试，包括 `LLaMA-3.2-3B-Instruct`, `Qwen1.5-MoE-A2.7B-Chat`, `Qwen2.5-3B-Instruct`, `Qwen3-8B`。
- **训练方式**：
  - PoLAR 预测网络使用离线 MCTS 搜索得到的有效执行程序作为监督信号进行训练。
  - 主 LLM 参数完全冻结，仅训练轻量预测器（约 2.1M 参数）。
- **评估指标**：
  - **pass@k accuracy**：Top-k 个预测程序中至少有一个产生正确答案的概率。
  - **平均执行层数** / **end-to-end latency**：衡量效率。
  - OOD 评估使用 **pass@1**。

### 基线方法对比
- **Base (T=0)**：贪婪解码的标准前向推理。
- **Base (sampling)**：随机采样生成多个输出取最优。
- **ShortGPT**：基于重要性静态剪枝层。
- **MindSkip**, **FlexiDepth**：学习型动态跳层方法。
- **DR.LLM**：基于 MCTS 监督学习层路由策略，支持 skip & repeat。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 在 DART-Math 上的 pass@5 准确率提升（以 LLaMA-3.2-3B-Instruct 为例）
| 方法 | DM-1 | DM-2 | DM-3 | DM-4 | DM-5 |
|------|------|------|------|------|------|
| Base (sampling) | 47.6 | 43.2 | 32.8 | 32.8 | 35.6 |
| **PoLAR** | **68.4** | **48.0** | **46.0** | **40.4** | **45.8** |
| **绝对增益 Δ** | **+20.8** | **+4.8** | **+13.2** | **+7.6** | **+10.2** |

> ✅ PoLAR 在所有难度级别上均显著优于基线，尤其在简单任务（DM-1）和中等任务上提升巨大。

#### MCTS 搜索揭示的潜在收益（Table 1）
在 DART-Math 上，允许 **Skip & Loop** 的程序相比标准前向，在 Qwen3-8B 上最高带来 **+50.6pp** 的绝对准确率提升（DM-1），证明了更优程序的存在性。

### 与基线方法的对比结果
- **准确性**：PoLAR 在所有 ID 和 OOD 数据集上 consistently 超过所有基线方法。
- **效率**：
  - 如 Table 4 所示，PoLAR 额外开销仅占标准前向的 **~0.8%**。
  - 实际端到端延迟降低：在易样本上达 **0.83×** 基线时间，在难样本上为 **0.95×**。
- **OOD 泛化能力**：在未见过的 MMLU-Pro 各子任务上，PoLAR 普遍提升准确率，例如在 `Qwen1.5-MoE-A2.7B-Chat` 上：
  - ASDiv: 59.1 → **63.8**
  - MAWPS: 41.7 → **46.7**
  - 多数 MMLU-Pro 子项提升 3–10 个百分点。

### 消融实验结果（隐含于分析中）
虽然没有显式消融表，但以下发现本质上是消融结论：
- **Skip vs. Loop vs. Both**（Finding 1）：
  - 仅 Loop > 仅 Skip
  - **Skip & Loop >> 单独任一操作**
- **程序长度影响**（Finding 2 & 3）：
  - 多数正确预测可通过**少于标准深度**的程序完成（Occam’s Razor）。
  - 更难任务需要更多执行步骤（depth scaling）才能达到高准确率。
- **结构偏好**（Finding 4）：
  - 有效程序倾向于使用**短连续段**（≤2 层居多）。
  - 每段最多重复一次即可，深层迭代极少有益。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **固定深度前向并非唯一正解**：大量输入存在比标准路径更短或更长但仍正确的“层程序”，说明 LLM 内部蕴含丰富的**潜推理能力**（latent reasoning capacity）。
2. **Skip 和 Loop 是互补机制**：结合两者能发现远优于单一操作的执行方案。
3. **程序结构具有强归纳偏置**：有效程序多由短连续段构成，且重复次数有限，这启发了高效的预测建模方式。
4. **轻量预测可行且高效**：通过一个小型网络预测执行程序，可在几乎无额外开销下实现 superior 性能，并支持 test-time scaling。

### 方法的局限性
- **依赖 MCTS 生成监督信号**：虽然推理高效，但训练数据依赖计算昂贵的离线搜索。
- **段长限制（K_max=4）**：人为限制了程序的表达空间，可能错过某些长程结构。
- **仅探索 skip/keep/repeat**：未支持更复杂的控制流（如条件分支），仍属简化模型。
- **应用场景集中于数学推理**：在其他类型任务上的普适性有待验证。

### 未来工作方向
- 探索**免搜索的自监督训练方式**，摆脱对 MCTS 的依赖。
- 将 PoLAR 思想推广至 Vision Transformers、Multimodal Models 等其他架构。
- 设计支持**条件执行**或**递归模块**的更通用程序语言。
- 研究如何**解释和可视化**所选执行路径，增强模型可解释性。
- 探索在边缘设备上的部署，利用其灵活性实现能耗自适应推理。

> 🔚 **总结一句话**：  
> **PoLAR 揭示了 LLM 推理不应局限于固定路径，而是可以通过动态编排预训练层来释放其更广的潜推理能力，提供了一种高效、灵活、无需微调的新范式。**

</details>

---

### 12. [Accelerated Decentralized Stochastic Gradient Descent for Strongly Convex Optimization](https://arxiv.org/abs/2606.07496)

**Authors**: Ming Sun, Kun Yuan  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.07496v1  

#### Abstract
Decentralized stochastic optimization is a fundamental paradigm for large-scale learning over networks, where agents communicate only with their neighbors and no central coordinator is required. For strongly convex problems, communication efficiency is mainly determined by the condition number \(\ka...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerated Decentralized Stochastic Gradient Descent for Strongly Convex Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **decentralized stochastic optimization**（去中心化随机优化）中，通信效率是制约大规模分布式学习的关键瓶颈。对于 **strongly convex**（强凸）问题，算法的收敛速度通常受两个因素影响：
- **Condition number** $ \kappa = L/\mu $：反映目标函数的病态程度；
- **Network spectral gap** $ 1 - \beta $：反映网络连通性（$\beta$ 是混合矩阵的第二大奇异值）。

尽管已有 **deterministic decentralized methods** 能同时实现对 $\sqrt{\kappa}$ 和 $1/\sqrt{1-\beta}$ 的加速依赖，但在 **stochastic setting** 中，尚无方法能同时在这两方面取得加速。这是因为梯度噪声会与共识误差耦合，导致加速机制可能放大误差而非抑制。

本文旨在解决这一开放问题：**是否可以在去中心化随机强凸优化中，同时实现对条件数和网络谱隙的双重加速？**

---

### 提出的新方法：MG-ADSGD
作者提出了 **Multi-Gossip Accelerated DSGD (MG-ADSGD)**，一种结合以下三种机制的新型去中心化随机优化算法：

1. **Nesterov-type primal-dual extrapolation**  
   引入动量机制以加速优化动态，提升对条件数 $\kappa$ 的依赖至 $\sqrt{\kappa}$。

2. **Multi-round Fast Gossip Averaging (FGA)**  
   在每次外层迭代中执行多轮加速 gossip 通信（而非单次），显著加快节点间的一致性达成，改善对 $1-\beta$ 的依赖至 $1/\sqrt{1-\beta}$。

3. **Coupling gossip depth with mini-batch size**  
   关键设计：令 **gossip depth $R$** 与 **mini-batch size $B$** 相等（即 $B = R$）。这样，增加 $R$ 可同时：
   - 提高通信深度 → 改善共识精度；
   - 增大批量大小 → 降低梯度方差。

该耦合策略实现了 **optimization acceleration**、**consensus acceleration** 与 **variance reduction** 的协同平衡。

---

### 相比现有方法的优势

| 方面 | MG-ADSGD | 现有方法（如 D-MASG [5], LMT [7]） |
|------|---------|-------------------------------|
| **Condition number dependence** | $\sqrt{\kappa}$ ✅ | 仅部分方法达到 |
| **Spectral gap dependence** | $1/\sqrt{1-\beta}$ ✅ | 多数为 $1/(1-\beta)$ ❌ |
| **双重加速** | 同时实现 ✅ | 无法兼顾 ❌ |
| **通信复杂度** | 当前最优（见下）✅ | 更高 ❌ |
| **退化到确定性情形** | 恢复最优确定性速率 ✅ | 不一定 |

> 📌 **核心优势**：首次在 decentralized stochastic strongly convex setting 中，**同时实现 $\sqrt{\kappa}$ 和 $1/\sqrt{1-\beta}$ 加速**，达到目前已知最优的 **communication complexity**。

---

## 2. 核心实验方法和设置

> ⚠️ 注意：本文为理论导向论文，**未提供传统意义上的“实验”章节**，而是通过 **convergence analysis** 和 **complexity comparison** 来验证性能。

### 数据集
- 文中未使用具体真实数据集进行数值实验。
- 所有分析基于标准假设下的抽象模型，适用于任意满足光滑性、强凸性和有界方差的数据分布。

### 实验设置（实为理论设定）
- **网络拓扑**：任意无向连通图，对应一个对称双随机混合矩阵 $W$，其第二特征值为 $\beta \in [0,1)$。
- **局部目标函数**：每个节点 $i$ 拥有局部经验风险 $f_i(x) = \mathbb{E}_{\xi \sim D_i}[F(x;\xi)]$。
- **假设条件**：
  - $L$-smooth（光滑）
  - $\mu$-strongly convex（强凸）
  - 随机梯度具有有界方差：$\mathbb{E}[\|\nabla F(x;\xi) - \nabla f(x)\|^2] \leq \sigma^2$
  - 混合矩阵满足 $\|W - \frac{1}{n}\mathbf{11}^\top\| \leq \beta$

### 评估指标
- **Communication complexity**：达到精度 $\epsilon$ 所需的总 **gossip rounds** 数。
- **Gradient computation complexity**：达到精度 $\epsilon$ 所需的随机梯度计算次数。
- 主要关注前者，因通信常为实际瓶颈。

### 基线方法对比（来自 Table 1）
| 方法 | 是否加速 $\sqrt{\kappa}$ | 是否加速 $1/\sqrt{1-\beta}$ | 类型 |
|------|------------------------|----------------------------|------|
| DGD / Diffusion | ❌ | ❌ | baseline |
| D-MASG [5] | ✅ | ❌（甚至更差） | stochastic |
| LMT [7] | ❌ | ✅ | stochastic |
| MSDA [21] | ✅ | ✅ | deterministic |
| **MG-ADSGD (Ours)** | ✅ | ✅ | **stochastic** ✅ |

---

## 3. 主要实验结果和性能指标

> 再次强调：所有“结果”均为理论推导所得的收敛界。

### 关键性能数据：通信复杂度

**Theorem 5.1 给出 MG-ADSGD 的通信复杂度为：**

$$
T_x(\epsilon) = \tilde{\mathcal{O}}\left( \left( \sqrt{\frac{L}{\mu}} \cdot \frac{1}{\sqrt{1-\beta}} + \frac{\sigma}{\sqrt{n\mu\epsilon}} \right) \log\left(\frac{1}{\epsilon}\right) \right)
$$

其中 $\tilde{\mathcal{O}}$ 隐藏了与 $\epsilon$ 无关的对数因子。

#### 分解解释：
- $\sqrt{\frac{L}{\mu}} / \sqrt{1-\beta}$：**加速项**，体现对条件数和网络结构的同时优化；
- $\frac{\sigma}{\sqrt{n\mu\epsilon}}$：**统计项**，源于随机梯度噪声，符合集中式 SGD 的最优速率；
- $\log(1/\epsilon)$：额外代价，来源于加速机制引入的对数因子，在通信主导场景中可接受。

---

### 与基线方法的对比结果

| 方法 | Communication Complexity |
|------|--------------------------|
| Standard DSGD | $\mathcal{O}\left( \frac{\kappa}{1-\beta} \cdot \frac{1}{\epsilon} \right)$ |
| D-MASG [5] | $\mathcal{O}\left( \frac{\sqrt{\kappa}}{(1-\beta)^2} \cdot \frac{1}{\epsilon} \right)$ （更差的网络依赖） |
| LMT [7] | $\mathcal{O}\left( \frac{\kappa}{\sqrt{1-\beta}} \cdot \frac{1}{\epsilon} \right)$ （未加速条件数） |
| **MG-ADSGD (Ours)** | $\tilde{\mathcal{O}}\left( \frac{\sqrt{\kappa}}{\sqrt{1-\beta}} \cdot \log\frac{1}{\epsilon} + \frac{\sigma}{\sqrt{n\mu\epsilon}} \right)$ ✅ |

> ✅ **这是首个在去中心化随机强凸优化中同时实现 $\sqrt{\kappa}$ 和 $1/\sqrt{1-\beta}$ 加速的方法**。

---

### 消融分析（隐含于理论设计）

虽然没有显式的消融实验，但从算法设计可看出关键组件的作用：

| 组件 | 若移除的影响 |
|------|-------------|
| **Primal-dual extrapolation** | 失去对 $\kappa$ 的加速，退化为线性依赖 |
| **Multi-round FGA** | 共识速度下降，网络依赖恶化为 $1/(1-\beta)$ |
| **$B = R$ 耦合机制** | 无法同步控制方差与共识误差，可能导致加速失效或震荡 |

此外，作者指出：
- 当 $\sigma^2 = 0$（无噪声）时，复杂度退化为：
  $$
  \mathcal{O}\left( \frac{\sqrt{\kappa}}{\sqrt{1-\beta}} \log\frac{1}{\epsilon} \right)
  $$
  与最优确定性方法（如 MSDA）一致（至多差一个 $\log$ 因子），说明方法设计合理且紧致。

---

## 4. 关键结论和发现

### 主要发现
1. **双重加速可达**：在去中心化随机优化中，**可以同时实现对条件数 $\kappa$ 和网络谱隙 $1-\beta$ 的加速**，此前被认为因噪声干扰而困难。
2. **协调设计是关键**：通过将 **accelerated gossip depth** 与 **mini-batch size** 耦合，可在不牺牲稳定性的情况下协同减少共识误差和梯度方差。
3. **当前最优通信复杂度**：MG-ADSGD 达到了目前已知最好的 communication complexity bound，优于所有现有去中心化随机方法。
4. **无缝衔接确定性情形**：当噪声消失时，算法自动恢复最优确定性加速速率，表明其理论一致性。

---

### 方法的局限性
1. **缺乏数值实验验证**：全文为纯理论分析，未给出在真实网络或数据上的仿真结果，实用性有待实证检验。
2. **参数调优复杂**：算法涉及多个超参数（步长 $\gamma$、动量 $\theta$、gossip depth $R$），且理论推荐值依赖全局常数（如 $L, \mu, \beta$），实际部署中难以精确获取。
3. **仅适用于强凸问题**：目前分析局限于 strongly convex 场景，扩展到非凸或 PL 条件需进一步研究。
4. **对称双随机矩阵假设**：要求 $W$ 对称且双随机，限制了某些动态或有向图的应用。

---

### 未来工作方向
1. **拓展到非凸/弱凸情形**：研究在 non-convex 或满足 PL condition 下的加速可能性。
2. **自适应参数选择**：设计无需先验知识（如 $L, \mu, \beta$）的自适应版本。
3. **异步与动态拓扑支持**：推广至 asynchronous communication 和 time-varying graphs。
4. **实际系统实现与测试**：在真实边缘设备网络或数据中心中部署并评估通信开销与收敛行为。
5. **与其他 variance reduction 技术结合**：如融入 SVRG、SAGA 等机制以进一步降低样本复杂度。

---

## 总结

📌 **MG-ADSGD 是一项重要的理论突破**，它首次证明了在去中心化随机强凸优化中，**可以同时实现对优化难度（$\kappa$）和网络结构（$1-\beta$）的双重加速**。其核心思想——**通过耦合 gossip depth 与 batch size 实现通信与统计误差的联合控制**——为后续研究提供了新范式。尽管尚缺实验支撑，但其理论成果已处于领域前沿，有望推动高效分布式学习系统的进一步发展。

</details>

---

### 13. [Beyond Rubrics: Exploration-Guided Evaluation Skills for Reward Modeling](https://arxiv.org/abs/2606.07040)

**Authors**: Xing Yue, Linjuan Wu, Daoxin Zhang, Yongliang Shen, Weiming Lu  
**Category**: cs.CL  
**Published**: 2026-06-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.07040v1  

#### Abstract
Open-ended reward modeling requires judges that can follow subtle, domain-specific preferences when verifiable answers are unavailable. Existing rubric-based methods often address this by generating criteria online for each query, but the extra generation step can add inference overhead and produce ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Rubrics: Exploration-Guided Evaluation Skills for Reward Modeling

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **rubric-based reward modeling** 方法在开放域任务中存在显著缺陷：
- **在线生成 rubric 引入额外推理开销**，增加延迟；
- 生成的 rubric 往往是 **query-agnostic**（仅基于查询生成），忽略了响应间的细微差异，容易产生 **僵化或错位的标准**（misaligned or rigid criteria）；
- 实验表明，自动生成的 rubric 有时甚至会 **劣于无指导的 vanilla 判断**。

### 提出了什么新方法或新思路
提出 **Eval-Skill** —— 一种 **探索引导的、可复用的评估技能合成方法**（exploration-guided method for synthesizing reusable evaluation skills），其核心思想包括：

- **将评估指导从“每例生成 rubric”转变为“离线构建领域级 skill”**：
  - Skill 是一种 **domain-level 的上下文构件**（context artifact），通过在少量演化集（evolving set）上的 rollouts 离线生成，并直接注入 judge prompt 中。
  - Skill 不仅包含类似 rubric 的 **Principles**（原则），还包含 **Workflow**（工作流），即如何比较候选、何时优先事实性、如何处理平局等复杂逻辑。

- **两阶段渐进式技能构建**（two-stage construction）：
  1. **Workflow 生成阶段**：先探索多种评估流程，选择最优 workflow 构成分支；
  2. **Principle 生成阶段**：在每个 workflow 分支上生成并合并具体评估原则；
  3. 最终从所有分支中选出表现最佳的 skill。

- **引入探索与选择机制**（exploration and selection）：
  - 受遗传算法启发，在 workflow 和 principle 阶段均进行多样化生成（`kgen`）与性能筛选（`kse1`, `ksmp`），避免陷入局部最优。

### 相比现有方法的优势
| 维度 | Eval-Skill | 在线 Rubric 方法 |
|------|-----------|----------------|
| **效率** | 无需每例生成 rubric，零推理时开销 | 每次判断需额外生成步骤 |
| **表达能力** | 支持条件判断、分支流程、强制选择等复杂逻辑 | 多为静态 criterion list，表达受限 |
| **一致性** | 领域统一 skill，减少标准漂移 | 每例 rubric 可能不一致 |
| **性能** | 显著优于各类基线 | 有时劣于 vanilla 方法 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **RewardBench 2**（1,763 cases）：包含 Factuality, Focus, Math, PIF, Safety 五个子域，用于主实验。
- **RewardBench**（2,984 cases）：包含 Chat, Chat-Hard, Safety, Reasoning 四个领域。
- **RM-Bench**（1,327 cases）：关注对风格偏差的鲁棒性和细微差异敏感性。
- **JudgeBench**：混合领域挑战集，用于测试跨域适应性。
- **HealthBench**：医疗领域压力测试集。

> 所有数据集均划分为 **100 例演化集**（用于 skill 生成）和其余测试集。

### 实验设置和评估指标
- **任务形式**：Pairwise 或 Listwise 判断，要求 judge 从多个响应中选出最优者。
- **评估指标**：准确率（Accuracy），即正确识别出被标注为“chosen”的响应的比例。
- **Skill Manager**：使用 **DeepSeek-V4-Flash**（启用 thinking 模式）负责 skill 的生成、精炼与合并。
- **超参数**：默认 `kgen=5`, `kse1=3`, `ksmp=5`。

### 基线方法对比
| 类型 | 具体方法 |
|------|--------|
| **Vanilla** | 无任何 rubric 或 skill 的基础 judge |
| **Rubric-based** | 自生成 rubric（如 Qwen3 → Qwen3）、后训练 rubric 模型（如 Rubric-RM, Rubric-ARM） |
| **Post-trained RMs** | RRM, RM-R1 等专用 reward model |
| **Naive Skill Method** | 简单合并本地 skill 得到全局 skill，无探索机制 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（RewardBench 2 上提升）
| Judge Model | Vanilla 准确率 | Eval-Skill 准确率 | **绝对提升** |
|------------|---------------|------------------|-------------|
| Qwen3-4B | 55.80% | **67.55%** | **+11.75pp** |
| Qwen3-8B | 57.04% | **70.48%** | **+13.44pp** |
| DeepSeek-V4-Flash | 68.27% | **86.78%** | **+18.51pp** |

> 注：pp = percentage points

在 **RewardBench** 上平均提升约 **5–6pp**，在 **RM-Bench** 上也取得显著增益（如 Qwen3-8B 从 71.89% → 77.34%）。

### 与基线方法的对比结果
- Eval-Skill 在所有 judge 背骨上均 **超越所有基线类型**，包括：
  - 显著优于 **vanilla judge** 和 **rubric-based 方法**；
  - 即使应用于已微调的 **post-trained RM**（如 Rubric-ARM-8B-Judge），仍能带来 **+13.22pp** 提升；
  - 优于 **naive skill 方法**，说明探索与选择机制有效提升了 skill 质量。

### 消融实验结果
#### （1）两阶段设计 vs 单阶段
| 方法 | RewardBench 2 (Qwen3-8B) | RewardBench (Qwen3-8B) |
|------|--------------------------|------------------------|
| **Eval-Skill (完整)** | **70.48** | **86.97** |
| - 第一轮监督 | 70.41 | 85.37 |
| - Stage 1（仅 workflow） | 66.96 | 83.38 |
| workflow-only（单阶段） | 68.05 | 83.80 |
| principle-only（单阶段） | 69.84 | 87.21 |
| full（单阶段） | 65.55 | 85.19 |

> ✅ 结论：两阶段设计更优；workflow 与 principle 均具强表达力。

#### （2）探索机制的作用
- 移除 `reference-guided generation` 和 `first-round supervision` 后性能下降，验证了探索引导的有效性。
- 并行扩展 `kgen` 效果最显著，说明 **多样性生成** 是关键。

#### （3）Skill Evolution-Time Scaling (SETS)
- **并行扩展**（increasing `kgen`, `kse1`, `ksmp`）可进一步提升性能（如从 69.95% → 70.18%），且 **不增加推理成本**。
- 性能最终趋于饱和，表明存在软上限。

---

## 4. 关键结论和发现

### 主要发现
1. **在线 rubric 生成并非总是有益**：由于 query-agnostic 设计，常导致 “Missing Criteria” 和 “Misaligned or Rigid” 错误（占失败案例的 58.3%）。
2. **Skill 是更强的指导形式**：结合 workflow 与 principle，能编码更复杂的判断策略。
3. **探索与选择至关重要**：简单迭代或合并无法稳定提升 skill 质量，必须主动搜索不同 workflow。
4. **Skill 具备良好迁移性**：
   - **跨 backbone 迁移**：一个模型上生成的 skill 可有效提升其他模型性能（见 Table 9）；
   - **跨 domain 迁移**：在相似领域（如 Chat 组）间有正向迁移，但在不相关领域可能负迁移。
5. **轻量高效**：仅需 **100 例演化集** 即可生成高质量 skill，适合低资源场景。

### 方法的局限性
1. **不适用于 pointwise RM 任务**：当前工作聚焦 pairwise/listwise 设置。
2. **在混合领域或高难度任务中效果有限**：
   - 在 **JudgeBench 混合域** 中，小模型（如 Qwen3-4B）难以遵循复杂 skill；
   - 在 **HealthBench** 上提升不显著，因任务知识密集且候选响应区分度低。
3. **依赖 backbone 的指令跟随能力**：
   - 如 Llama-3.1-8B-Instruct 对 skill 遵循较差；
   - RRM-7B 因固定推理模式，反而偏好 workflow-only skill。

### 未来工作方向
- 将 Eval-Skill 应用于 **实际的强化学习 pipeline** 中，验证其作为 RM 的端到端效果；
- 探索 **自动 domain grouping** 以优化跨域 skill 迁移；
- 开发 **轻量化 skill manager**，降低对强模型的依赖；
- 研究 **动态 skill 调整机制**，应对更广泛的输入分布。

---

> 🔗 **代码开源**：https://github.com/xing-stellus-yue/Eval-Skill  
> 📄 **许可证**：Apache-2.0

</details>

---

### 14. [Elmes*: Automated Construction of Fine-Grained Evaluation Rubrics for Large Language Models in Long-Tail Educational Scenarios](https://arxiv.org/abs/2606.06546)

**Authors**: Tao Liu, Ye Lu, Ruohua Zhang, Siyu Song, Wentao Liu, Aimin Zhou, Hao Hao  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.06546v1  

#### Abstract
Evaluating large language models (LLMs) for education requires measuring how models teach, not only what they know. Existing benchmarks emphasize domain-general correctness or depend on manually designed rubrics that scale poorly to long-tail pedagogical scenarios. We introduce Elmes*, an end-to-end...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《ELMES+: Automated Construction of Fine-Grained Evaluation Rubrics for Large Language Models in Long-Tail Educational Scenarios》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前对教育领域 Large Language Models（LLMs）的评估存在以下瓶颈：
- **通用基准（如 MMLU-Pro、C-Eval）** 过度关注知识正确性，忽视教学过程质量（如 scaffolding、情感支持、个性化引导）。
- **教育专用基准（如 EduBench、TutorBench）** 覆盖场景有限，且依赖人工设计 rubrics，难以扩展至“长尾”教育场景（即低频但重要的学科-年级-任务组合）。
- **评估一致性差**：人类评分成本高、方差大；LLM-as-a-Judge 存在系统性偏差（如 self-preference、verbosity bias）。

### 🚀 提出的新方法与创新
作者提出 **ELMES+** ——一个端到端的自动化框架，用于构建、优化并应用细粒度、场景特定的评估 rubrics。

#### 主要创新点：
1. **多智能体评估引擎（Multi-Agent Evaluation Engine）**
   - 基于 **declarative YAML-to-DAG** 配置系统，支持 teacher-student-judge 角色建模。
   - 支持单轮与多轮交互（如苏格拉底式辅导），可定义复杂对话流与终止条件。
   - 提供 **Gradio 可视化界面**，允许非技术专家参与配置与迭代。

2. **自演化 rubric 合成模块 SCENEGEN**
   - 初始输入为专家定义的四个 pedagogical 维度（带权重）：
     - Personalization (0.3)
     - Teaching Method Professionalism (0.4)
     - Creativity Stimulation (0.2)
     - Values Integration (0.1)
   - 通过闭环机制实现 **rubrics 与 test data 的协同进化（co-evolution）**：
     - 分析 score distribution 发现“过严”、“过松”或“区分度低”的指标。
     - 动态优化 synthesis prompt 和 metric 描述，保留 metric 名称与权重以保证可比性。
   - 引入三大收敛保障机制：
     - 锚定样本重评（Anchor Calibration）
     - 最优版本回滚（Best-Version Rollback）
     - 提前停止（Early Stopping）

3. **大规模教育基准 Edu-330**
   - 覆盖 **11 个学科 × 3 个学段 × 10 种任务类型 = 330 个核心场景**。
   - 包含超过 **1,298 个二级评估指标**，形成细粒度诊断能力。

4. **提升 LLM-as-a-Judge 的一致性与对齐性**
   - 多 judge ensemble + 专家锚定 few-shot 示例显著降低与 human expert 的评分偏差。
   - 实验表明该策略可将平均得分偏差减少约 **30%**，同时保持高 inter-rater reliability。

### 🔍 相比现有方法的优势
| 方面 | 现有方法 | ELMES+ |
|------|--------|-------|
| **Rubric 构建方式** | 手工设计，扩展性差 | 自动化生成 + 自演化优化 |
| **覆盖范围** | 少量典型场景 | 330 场景，涵盖长尾需求 |
| **评估维度** | 单一或粗粒度 | 多维、细粒度、可定制 |
| **评分稳定性** | 人类评分方差高（0.11~0.45） | LLM judge 方差极低（~0.01） |
| **人机对齐** | 缺乏校准机制 | 引入 few-shot anchoring 显著改善 |

---

## 2. 核心实验方法和设置

### 📚 数据集
1. **Edu-330 Benchmark**
   - 自动生成的 330 个教育场景，覆盖语文、数学、英语等 11 学科，小学/初中/高中三个学段，以及概念讲解、问题解决、跨学科教学等多种任务类型。
   - 使用子集（66 场景）进行代表性模型比较。

2. **Gold-Standard Expert-Authored Scenarios（4 个）**
   - 由教育专家手动设计，作为“黄金标准”验证框架有效性：
     - Guided Tutoring（引导式辅导）
     - Knowledge Explanation（知识点解释）
     - Interdisciplinary Lesson Planning（跨学科教案设计）
     - Contextualized Question Generation（情境化题目生成）
   - 每个任务均有多角色设定（教师、家长、学生）、多样化互动模式与内容格式。

### ⚙️ 实验设置
- **被测模型（Test Models）**：
  - GPT-5, Claude-Opus-4.5, Qwen-2.5-72B-Instruct, Qwen3-235B-A22B, DeepSeek-R1, Kimi-k2.5
- **裁判模型（Judge Models）**：
  - GPT-4.1, Claude-4-Sonnet, Gemini-2.5-Pro, Kimi-k2-turbo-preview
- **评分方式**：
  - 所有输出采用 **5 分制 Likert scale**。
  - human expert 采用双盲评审，Cronbach’s α 在 0.87–0.89 之间，信度极高。
- **评估维度聚合**：
  - 原始细粒度评分 → 聚合为四大核心维度：Professionalism, Personalization, Creativity, Values

### 📊 评估指标
- **平均得分（Mean Score）**
- **评分方差（Variance）**
- **与 human expert 的排名一致性（Rank Alignment）**
- **mean-score deviation from human baseline**
- **消融实验中的 prompt engineering 效果**

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）Edu-330 上的模型表现（图4）
- **Top-tier 模型间差异明显**：GPT-5 明显优于 Qwen-2.5-72B。
- **性能差距集中在 Creativity 与 Values 维度**，而非 Professionalism 或知识掌握。
- 表明当前 LLMs 的瓶颈在于 **教学法创新能力与价值观融合能力**，而非事实知识。

#### （2）专家设计场景的人类评分结果（表1）
| Model | Avg Score |
|-------|----------|
| InnoSpark（教育垂直模型） | **3.98** ✅ |
| Gemini-2.5-Pro | 3.92 |
| DeepSeek-R1 | 3.53 |
| GPT-4o | 3.41 |

👉 **关键发现**：教育专用模型 InnoSpark 获得最高 human-evaluated 平均分，说明领域专业化具有潜力。

#### （3）LLM Judge 的一致性分析（表2）
- **LLM judges 的评分方差远低于 human experts**：
  - Human variance: 0.11 ~ 0.45
  - LLM variance: ~0.01（部分接近 0）
- **相对排名基本一致**：Gemini 与 InnoSpark 始终得分较高，GPT-4o 较低。
- **存在系统性偏差**：
  - GPT-4.1 倾向宽松打分（near-ceiling scores）
  - Claude-4-Sonnet 标准严格，尤其在挑战性任务中
  - **Self-preference 现象严重**：Gemini 当自评时给自己打出 4.008 分（高于人类评分 3.936），而其他模型检测到其存在 factual error。

#### （4）消融实验结果（图5）
| 策略 | 效果 |
|------|------|
| **Few-shot anchoring（专家标注示例）** | ✔️ 显著提升 human-LLM 对齐性（mean deviation ↓20%）<br>❌ 成本高，需人工标注 |
| **Reasoning enforcement（强制 CoT 推理）** | ⚠️ 效果 model-dependent：<br>- GPT-4.1 ↑ +0.992 对齐<br>- 其他模型改善微弱甚至下降 |
| **Greedy decoding（temperature=0）** | ❌ 几乎无 variance 改善（因 baseline 已很低）<br>⚠️ 可能导致次优选择，扩大与 human 的一致性差距（Gemini: -0.962） |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **教育能力是多维诊断问题，非单一排序任务**  
   Top-tier LLMs 的差异主要体现在 **Creativity Stimulation** 和 **Values Integration** 上，而非基础知识或 Professionalism。

2. **教育专用模型具备优势**  
   InnoSpark 在 human evaluation 中取得最佳成绩，证明 **domain specialization 是提升教育表现的有效路径**。

3. **LLM-as-a-Judge 具备可行性但需校准**  
   - LLM judges 能稳定复现 human 的相对排名。
   - 但存在 **self-preference、position bias、leniency/strictness bias**。
   - 必须引入 **multi-judge ensembling + expert anchoring** 来中和偏差。

4. **自动化 rubric 构建可行且高效**  
   SCENEGEN 成功从 4 个初始维度生成 >1,298 个细粒度指标，覆盖 330 场景，极大缓解了人工标注负担。

5. **few-shot anchoring 是最有效的对齐手段**  
   尽管成本较高，但它是最可靠地缩小 human-LLM 评分差距的方法。

---

### ⚠️ 局限性
1. **模拟环境限制**  
   当前评估基于 synthetic student profiles，无法完全反映真实学习者的认知动态与情绪变化。

2. **仅支持文本模态**  
   不支持语音对话、手写识别、图像讲解等 multimodal 教育场景。

3. **依赖初始 pedagogical dimensions 设计**  
   虽然维度可替换，但其质量直接影响生成 rubric 的有效性。

4. **未接入真实学习成果数据**  
   缺乏与实际学生 performance improvement 的关联验证。

---

### 🔮 未来工作方向
1. **拓展至多模态交互评估**  
   支持 voice-based tutoring、handwriting analysis、visual explanation 等形式。

2. **结合真实 learner outcome data**  
   将 synthetic metrics 与 real-world 教学效果（如测试成绩提升、参与度）建立关联。

3. **增强专家参与机制**  
   构建持续反馈 loop，让教师在部署中不断优化 rubrics。

4. **开发 curriculum-adaptive rubric generation**  
   根据国家课程标准自动适配评估维度与权重。

5. **探索轻量化 few-shot 校准方法**  
   减少对高质量 human-labeled anchor samples 的依赖。

---

> **总结一句话**：  
> **ELMES+ 提供了一个可扩展、可诊断、可迭代的教育 LLM 评估基础设施，推动从“知识评测”走向“教学能力评测”的范式转变。**

</details>

---

### 15. [Spatiotemporal Imputation with Graph-Informed Flow Matching](https://arxiv.org/abs/2606.06682)

**Authors**: Zepeng Zhang, Aref Einizade, Jhony H. Giraldo, Olga Fink  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.06682v1  

#### Abstract
Missing data is a common challenge in spatiotemporal systems, arising in applications such as air quality monitoring and urban traffic management. Traditional machine learning approaches, like recurrent and graph neural networks, rely on iterative propagation, which tends to accumulate errors over t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Spatiotemporal Imputation with Graph-Informed Flow Matching

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**spatiotemporal imputation**（时空数据插补）中的挑战展开研究。在空气质量监测、城市交通管理等应用中，由于传感器故障、传输错误等原因，时空数据常存在缺失值。传统方法如RNN和GNN依赖迭代传播，容易导致误差累积；而基于diffusion的方法虽然避免了自回归问题，但仍依赖于问题无关的高斯先验（problem-agnostic Gaussian prior），且采样过程需要多步去噪，效率较低。

### 提出的新方法：GiFlow
作者提出了一种名为 **GiFlow**（Graph-Informed Flow Matching）的新型生成模型框架，用于解决时空数据插补问题。其核心思想是将**flow matching**（FM）与图结构信息结合，构建一个更贴近目标分布的“图感知”先验（graph-informed prior）。

#### 创新点：
- **图感知先验构造**：不同于传统的高斯先验，GiFlow通过**adaptive spatiotemporal filtering**（自适应时空滤波）对可观测信号进行处理，生成一个结构化更强、更接近真实数据分布的源分布 $ p_0 $。这显著缩短了从源到目标的生成路径。
- **混合向量场建模**：设计了一个融合**spatial attention**、**temporal attention** 和 **spatiotemporal propagation** 的向量场模型，联合捕捉空间与时间依赖关系。
- **理论支持**：证明了所提出的图感知先验可以**provably reduce transport cost**（可证明地降低传输成本），并分析了滤波因子与感受野之间的关系（Proposition 3.1 和 Theorem 3.2）。

### 相比现有方法的优势
| 方法类型 | 缺陷 | GiFlow 的改进 |
|--------|------|---------------|
| RNN/GNN-based | 迭代传播导致误差累积 | 非自回归生成，避免误差传播 |
| Diffusion-based | 依赖高斯先验，需多步采样 | 使用图结构先验，简化生成轨迹；支持高效确定性采样 |
| 其他FM方法 | 仍常用高斯先验 | 引入问题定制化的图感知先验，提升性能与效率 |

---

## 2. 核心实验方法和设置

### 数据集
实验在**合成数据集**和三个广泛使用的**真实世界数据集**上进行：

| 数据集 | 描述 |
|-------|------|
| **Synthetic** | 基于平滑图信号生成，模拟时空演化过程 |
| **Air-36** | 北京市36个站点的PM2.5小时浓度数据（8760时间步） |
| **AQI** | 中国43个城市共437个站点的空气质量指数（AQI）数据 |
| **PeMS08** | 加利福尼亚州高速公路交通流量数据（170个传感器，5分钟粒度） |

### 实验设置
- **缺失模式**：
  - **Point missing**：随机遮蔽 $ p \in \{20\%, 30\%\} $ 的观测值。
  - **Block missing**：连续遮蔽某个节点的一段时序数据，模拟长时间断连。
- **评估指标**：
  - MAE（Mean Absolute Error）
  - RMSE（Root Mean Squared Error）
  - MAPE（Mean Absolute Percentage Error）
- **训练细节**：
  - 使用Adam优化器，早停机制（patience=10）
  - 采用EMA（Exponential Moving Average）稳定训练
  - ODE求解使用Euler方法，20步

### 基线方法对比
| 类型 | 方法 |
|-----|------|
| 非参数法 | Mean-S, Mean-T, Linear, KNN, FP |
| RNN-based | BRITS, SAITS |
| GNN/Transformer-based | SPIN, GRIN, OPCR |
| 生成模型 | PriSTI（diffusion-based）、CoSTI（consistency model） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 Air-36 和 AQI 为例）

#### 表1：Air-36 上的 Point Missing ($p=20\%$)
| Model | MAE ↓ | RMSE ↓ | MAPE ↓ |
|-------|--------|---------|----------|
| BRITS | 14.23 | 24.64 | 31.42 |
| GRIN | 9.94 | 19.09 | 21.95 |
| PriSTI | 10.29 | 19.66 | 21.91 |
| **GiFlow** | **9.54** | **18.10** | **21.27** |

> ✅ GiFlow 在所有指标上均取得最优表现。

#### 表2：Block Missing 下的表现下降明显
- 多数方法（如Linear、BRITS）在block missing下性能大幅下降。
- GiFlow 依然保持领先，说明其对长连续缺失具有更强鲁棒性。

#### 表3：PeMS08 上的结果（$p=20\%$）
| Model | MAE (Point) ↓ | MAE (Block) ↓ |
|-------|----------------|----------------|
| OPCR | 12.77 | 22.96 |
| PriSTI | 13.02 | 18.94 |
| **GiFlow** | **12.66** | **18.70** |

> ✅ 在交通数据上也验证了泛化能力。

### 与基线方法的对比结果
- GiFlow 在**所有数据集、所有缺失模式、所有评估指标**上均优于或媲美当前最先进的方法。
- 尤其在高缺失率（如60%）下，优势更加明显（见Figure 2）。
- 相比diffusion模型（如PriSTI），GiFlow无需多次采样平均，推理更快。

### 消融实验结果

#### （1）不同先验的消融（Table 4）
| Model | Transport Cost ↓ | MAE ↓ |
|-------|------------------|--------|
| FM-Gauss（高斯先验） | 299.62 | 12.79 |
| TFM（仅时间滤波） | 123.39 | 10.12 |
| GFM（仅空间滤波） | 115.05 | 9.75 |
| **GiFlow（时空联合）** | **104.29** | **9.54** |

> 🔍 结论：图感知先验显著降低transport cost，且空间滤波贡献更大。

#### （2）组件移除实验（Table 5）
| 变体 | MAE ↑ |
|------|--------|
| 完整GiFlow | 9.54 |
| w/o spatial attention | 10.05 |
| w/o temporal attention | 9.87 |
| w/o spatiotemporal propagation | **10.40** |

> 🔍 结论：spatiotemporal propagation 对性能影响最大，表明信息传播至关重要。

#### （3）图质量敏感性分析（Table 6）
- 图阈值在0.05~0.4之间时性能稳定；
- 极端阈值（0.02或0.6）导致性能下降，说明**合理图结构对模型有效**。

---

## 4. 关键结论和发现

### 主要发现
1. **图感知先验的有效性**：利用可观测信号通过spatiotemporal filtering构造先验，能显著拉近源分布与目标分布的距离，从而减少生成路径长度和transport cost。
2. **更高的效率与准确性**：
   - 推理速度快于diffusion模型（Table 7：GiFlow在Air-36上仅需0.28分钟 vs PriSTI的9.30分钟）。
   - 即使使用较少Euler步数（如5步），仍优于多数基线（Table 9）。
3. **自适应感受野机制**：随着缺失率增加，优化得到的滤波因子 $ T_n, T_s $ 自动增大，扩展感受野以捕获更远距离依赖（Figure 3），符合理论预期。
4. **空间依赖更重要**：在block missing场景下，$ T_n $（空间滤波强度）增长更快，说明模型更依赖空间信息来填补时间空洞。

### 方法的局限性
- **图结构依赖性强**：若输入图结构严重失真或不准确（如极端阈值），性能会下降。
- **静态图假设**：目前使用固定的空间图（基于距离相似性构建），未考虑动态变化的拓扑关系。
- **滤波参数学习开销**：尽管整体高效，但在训练阶段需额外优化滤波因子（Table 8显示其为轻量级）。

### 未来工作方向
- 扩展至**dynamic graph** 场景，建模随时间演化的空间关系。
- 探索**stochastic sampling** 路径，在需要不确定性量化时注入噪声。
- 将该框架应用于其他条件生成任务，如**forecasting** 或 **anomaly detection**。
- 进一步探索如何将**consistency model** 思想与flow matching结合以加速训练。

---

> 📌 **总结一句话**：  
> GiFlow通过引入**图感知先验**和**混合向量场建模**，在flow matching框架下实现了高效、准确、鲁棒的时空数据插补，解决了传统方法中存在的误差累积与低效采样问题，在多个真实场景中展现出SOTA性能。

</details>

---

### 16. [GRASP: Geometry-aware Residual Alignment for Scalable Pretraining Data Attribution](https://arxiv.org/abs/2606.06892)

**Authors**: Yue Min, Ruining Chen, Yujun Li  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.06892v1  

#### Abstract
Scalable data attribution methods typically assign isolated utility scores to individual training examples. This prevalent additive assumption fundamentally fails to capture critical subset dynamics, including data redundancy and complementary coverage. In this work, we reframe attribution as subset...

---

### 17. [TALAN: Task-Aligned Latent Adaptation Networks for Targeted Post-Training of Large Language Models](https://arxiv.org/abs/2606.06902)

**Authors**: Chengkai Zhang, Ziteng Liu, Junpu Wang, Zeyi Tao, Yang Wang, Sagar Chordia, Qin Huang  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.06902v1  

#### Abstract
Targeted post-training aims to improve reasoning, math, and code without degrading strengths. Low-rank adapters are efficient but task-global; activation interventions are input-aware but often require separate probes, vectors, or inference-time steering. We introduce TALAN (Task-Aligned Latent Adap...

---

### 18. [Drifting Models for Surrogate Flow Modeling](https://arxiv.org/abs/2606.07481)

**Authors**: Chris R. Jung, Markus D\"orr, Natalie J\"ungling, Jennifer Niessner, Adam T. M\"uller, Nicolaj C. Stache  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.07481v1  

#### Abstract
While Computational Fluid Dynamics (CFD) provides high-fidelity flow fields for optimizing indoor environments, its computational cost limits rapid exploration. To solve this problem generative surrogates offer better distribution modeling than deterministic networks, but iterative sampling is slow....

---

### 19. [CoMetaPNS: Continually Meta-learning Personalized Neural Surrogates for Cardiac Electrophysiology Simulations](https://arxiv.org/abs/2606.07488)

**Authors**: Ryan Missel, Xiajun Jiang, Linwei Wang  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.07488v1  

#### Abstract
Personalized virtual heart simulations face challenges in model personalization and computational cost. While neural surrogates offer state-of-the-art solutions, they typically address either efficient personalization or training generalizable models. Recent work reframes this by learning the proces...

---

### 20. [When to Think Deeply: Inhibitory Deliberation for LLM Reasoning](https://arxiv.org/abs/2606.06745)

**Authors**: Zhixuan He, Yue Feng  
**Category**: cs.CL  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.06745v1  

#### Abstract
Reasoning Large Language Models can improve problem-solving performance through deliberative inference, but invoking slow reasoning for every input is computationally expensive and often unnecessary. We propose IDPR, a framework for response-conditioned inhibitory deliberation. IDPR first generates ...

---

### 21. [TA-RAG: Tone-Aware Retrieval-Augmented Generation for Peer-Support Health Communication](https://arxiv.org/abs/2606.06794)

**Authors**: Yong-Bin Kang, Anthony McCosker  
**Category**: cs.CL  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.06794v1  

#### Abstract
Retrieval-augmented generation (RAG) successfully grounds large language model (LLM) outputs in trusted documents, but factual grounding alone is insufficient for sensitive peer-support health communication. In domains such as HIV peer support, responses must also be accessible, stigma-free, empathe...

---

### 22. [CRAFT: A Unified Counterfactual Reasoning Framework for Tabular Question Answering and Fact Verification](https://arxiv.org/abs/2606.06842)

**Authors**: Chenshuo Pan, Yu Zhao, Jie Zhang, Changzai Pan, Zhenhe Wu, Jiayi Liang, Yujie Mao, Shuangyong Song, Yongxiang Li, Zhongjiang He  
**Category**: cs.CL  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.06842v1  

#### Abstract
Table reasoning remains challenging for large language models (LLMs), particularly in tasks that require multi-step inference over long and structured tables. Existing approaches predominantly rely on single-direction reasoning, which limits their ability to explore alternative hypotheses across tas...

---

### 23. [Communication Strategy Selection for Multi-GPU 3D FDTD with Convolutional Perfectly Matched Boundary Layers](https://arxiv.org/abs/2606.06910)

**Authors**: Victory C. Obieke  
**Category**: cs.DC  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.06910v1  

#### Abstract
In this paper we describe a communication-strategy study for multi-GPU three-dimensional finite-difference time-domain computation with convolutional perfectly matched layer boundary conditions using CUDA. The metrics used to determine the most effective implementation include runtime, throughput in...

---

### 24. [Federated Foundation Models over Vehicular Networks](https://arxiv.org/abs/2606.06786)

**Authors**: Kasra Borazjani, Fardis Nadimi, Payam Abdisarabshali, Owen Palinski, Allan Salihovic, Dinh Nguyen, Minghui Liwang, Seyyedali Hosseinalipour  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.06786v1  

#### Abstract
This paper presents a forward-looking vision for integrating the emerging multi-modal multi-task federated foundation models (M3T FedFMs) into vehicular networks, with the goal of unifying the expressive power of multi-modal multi-task foundation models (M3T FMs) with the privacy-preserving and dist...

---

### 25. [Evidence-Grounded Ensemble Diagnosis of 802.11 Packet Captures: A Multi-Stage Pipeline with Deterministic Reliability Scoring](https://arxiv.org/abs/2606.06871)

**Authors**: Jerome Henry, Swadhin Pradhan, Miroslav Popovic  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.06871v1  

#### Abstract
Diagnosing 802.11 packet captures requires expert protocol knowledge, is slow, inconsistent across engineers, and unscalable. LLM-based approaches sound plausible but fabricate protocol events absent from captures (especially truncated traces), produce uncalibrated confidence scores, and suffer eval...

---

### 26. [Breaking the Ice: Analyzing Cold Start Latency in vLLM](https://arxiv.org/abs/2606.07362)

**Authors**: Huzaifa Shaaban Kabakibo, Animesh Trivedi, Lin Wang  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.07362v1  

#### Abstract
As scalable inference services become popular, the cold start latency of an inference engine becomes important. Today, vLLM has evolved into the de facto inference engine of choice for many inference workloads. Although popular, due to its complexity and rapid evolution, there has not been a systema...

---

### 27. [Making the Most of Limited Data: Score-Aware Training for Text-to-Music Generation](https://arxiv.org/abs/2606.07387)

**Authors**: Yun-Chen Cheng, Tzu-Hung Huang, Chih-Pin Tan  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.07387v1  

#### Abstract
State-of-the-art text-to-music generation systems rely on massive proprietary datasets and industrial-scale compute, making it impossible to disentangle architectural contributions from resource advantages. We propose \textit{score-aware training}, which treats audio-caption alignment score as a dir...

---

### 28. [Reversible Foundations: Training a 120B Sparse MoE through State-Preserving Scaling](https://arxiv.org/abs/2606.07404)

**Authors**: Rohan Shravan  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.07404v1  

#### Abstract
This paper reports on training a hundred-billion-parameter sparse mixture of experts on a single eight-GPU node, end to end. LightningLM 0.1V is a recurrence-backbone language model family grown in four stages from a small dense seed, through a 5B and a 9B mixture of experts, to a 120B model with 46...

---

### 29. [Network Recovery from Cascade Data: A Debiased Jacobian-Based Machine Learning Approach](https://arxiv.org/abs/2606.07483)

**Authors**: Lei Huang  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.07483v1  

#### Abstract
Many important outcomes unfold as dynamic cascades, including product adoption, disease spread, financial distress, and information diffusion. A central challenge is to recover the hidden influence network behind these cascades. Existing methods typically assume a specific diffusion model, and their...

---

### 30. [Second-Order Path Kernel Interpolation Formulas in Machine Learning](https://arxiv.org/abs/2606.07495)

**Authors**: Jin Guo, Roy Y. He, Jean-Michel Morel  
**Category**: cs.LG  
**Published**: 2026-06-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.07495v1  

#### Abstract
Understanding how training data shape neural network predictions is a central problem in modern learning theory. In 2020, Pedro Domingos proposed an interpolation formula valid for every model learned by deterministic gradient descent. It expresses the model's prediction as an integral, along the op...

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

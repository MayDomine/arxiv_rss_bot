# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-24 10:34:14 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models](https://arxiv.org/abs/2609.27373)

**Authors**: Ke Wan, Chen Chen  
**Category**: cs.CL  
**Published**: 2026-09-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.27373v1  

#### Abstract
Recurrent language models repeatedly apply shared network blocks to refine latent representations, but standard inference recomputes global attention at every recurrent step. We study attention dynamics across recurrent depth and find that attention support and distributions stabilize substantially ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
当前的 **recurrent language models**（如 Huginn、Recurrent Transformers）在推理时通过多轮重复应用共享网络块来逐步精炼隐状态（hidden representations），从而提升多步推理能力。然而，标准实现中每一轮都执行 **full global attention**，即对整个上下文重新计算注意力路由，即使模型“已经知道该关注哪里”。这种重复的全局计算带来了显著的 **O(n²)** 注意力开销，尤其在长上下文场景下效率低下。

本文提出：**是否可以在不牺牲性能的前提下，减少后期 recurrent 步骤中的冗余全局 attention 计算？**

---

### **提出了什么新方法或新思路**
作者提出了一个训练无关（**training-free**）的推理优化方法 —— **WISE (Working-set Inference with Support Exploitation)**，其核心思想是：

> **Discover globally early, reuse sparsely late.**

具体分为两个阶段：
1. **早期阶段（Discovery Phase）**：前 $ t_a $ 轮使用完整的 global attention，动态地发现一个稀疏的、稳定的 **working set**（即关键上下文块集合）。
2. **后期阶段（Reuse Phase）**：后续步骤固定使用该 working set 的 **block-structured support**，仅在此子集上进行 attention 计算，而隐藏状态、Q/K/V 和内部 attention 权重仍保持动态更新。

关键洞察是：**attention routing（路由结构）比 representation refinement（表示精炼）更早稳定**，因此可以复用“去哪里找”的路径，而不冻结“如何处理”的过程。

---

### **相比现有方法的优势**
| 维度 | WISE | 其他方法 |
|------|------|--------|
| **是否需重训练** | ❌ 否（training-free） | 多数需要微调或特定训练 |
| **是否保留 recurrent depth** | ✅ 是 | 如 early-exit 方法会提前终止 |
| **是否动态更新表示** | ✅ 是（within-support 动态） | 如 attention reuse 冻结权重 |
| **是否利用输入自适应结构** | ✅ 是（data-dependent working set） | 如固定稀疏模式无法适配输入 |
| **是否兼容高效 kernel** | ✅ 是（block-sparse 可用 Triton 实现加速） | 多数理论优化难落地 |

此外，WISE 与 **FlashAttention** 等优化互补，可进一步提升实际速度。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **HotpotQA**：多跳问答，要求模型整合多个句子进行推理。
- **GSM8K**：数学应用题，测试复杂逻辑与数值推理。
- **2WikiMultiHopQA**：另一个多跳 QA 数据集，用于跨任务验证。
- 上下文长度从 **512 到 4K** 不等，用于测试 scalability。

---

### **实验设置和评估指标**

#### **模型**
- 主要模型：**Huginn**（publicly available recurrent LM, T=32）
- 对比模型：**Recurrent-Llama-T32**（Llama-3.2 改造为 recurrent 架构）

#### **WISE 配置**
- Block size $ B = 32 $
- Discovery depth $ t_a = 12 $
- Working set = union of top-95% mass blocks from steps 9–12
- 支持直接在 block 空间构建（而非 token-level 四舍五入）

#### **评估指标**
| 类型 | 指标 |
|------|------|
| **下游质量** | Token-level F1, EM |
| **行为一致性** | Answer changes vs. Full model, △F1 |
| **路由结构分析** | Block density, Future Full-attention mass |
| **效率** | Latency (ms), Speedup × |
| **机制诊断** | Attention support stability, Hidden state convergence |

---

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Full** | 标准 full-attention recurrent inference |
| **Truncate@12** | 在第 12 步后停止 recurrence |
| **Static-95** | 第一步就冻结 top-95% 支持 |
| **MassMatched** | 匹配 WISE 的 attention mass，但基于 step-1 排序 |
| **SizeMatched** | 完全匹配 WISE 的 block 数量 |
| **Freeze-A@12** | 使用相同支持，但冻结 step-12 的 attention 分布 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### **下游任务表现（F1）**
| Method | HotpotQA | 2WikiMultiHopQA |
|--------|----------|------------------|
| Full | 0.2263 | 0.2516 |
| WISE | **0.2317** | **0.2497** |
| SizeMatched | 0.1836 | 0.2446 |
| Freeze-A@12 | 0.2164 | 0.2001 |

✅ **WISE 不仅未降质，反而轻微提升性能**，且显著优于其他静态控制。

#### **答案变化率（vs. Full）**
- WISE: 仅 **24/100** 个样本答案改变（最低之一）
- Static-95: 87/100 → 显示早期静态策略破坏性强

#### **路由复用效率**
- **Working set 密度**：随 context 增加而下降
  - 512 context: ~57%
  - 4K context: **~37%**
- **Retained future attention mass**: 保持 **96–98%** 的原始 attention 质量

---

### **与基线方法的对比结果**

| 方法 | F1 下降 | Answer Changes | Future Mass Retained |
|------|--------|----------------|-----------------------|
| Truncate@12 | ↓ 显著 | ↑↑ | — |
| Static-95 | ↓↓↓ 最大 | ↑↑↑ | 88% |
| SizeMatched | ↓ 中等 | ↑↑ | 94.7% |
| Freeze-A@12 | ↓ 较小 | ↑↑ | 96.7% |
| **WISE** | ↔️ 或略升 | **最少之一** | **96.7%** |

👉 结论：**只复用 routing support（位置），不冻结 attention weights（权重）是最优选择**。

---

### **消融实验结果**

#### **不同 mass threshold $\eta$**
| $\eta$ | Density | Future Mass | F1 (HotpotQA) |
|--------|--------|-------------|---------------|
| 0.90 | 30.2% | 93.3% | 0.2102 |
| **0.95** | **45.0%** | **96.7%** | **0.2317** |
| 0.99 | 77.8% | 99.4% | 0.2212 |

➡️ $\eta=0.95$ 在稀疏性和质量间取得最佳平衡。

#### **不同 block size $B$**
| $B$ | Density (2K) | Speedup (2K) | F1 差异 |
|-----|--------------|-------------|---------|
| 16 | 31.12% | 1.141× | ≈ |
| **32** | **37.51%** | **1.656×** | ≈ |
| 64 | 44.71% | 1.440× | ≈ |
| 128 | 54.44% | 1.236× | ≈ |

➡️ **B=32 是系统层面的“knee point”**：虽非最稀疏，但执行效率最高。

#### **不同 discovery depth $t_a$**
- $t_a=8$: 性能略降（未充分探索）
- $t_a=16$: 无明显增益（过晚切换）
➡️ $t_a=12$ 是合理折衷。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **Routing-Stabilization Hypothesis Confirmed**：
   - Attention support 和 attention distribution **显著早于** hidden states 和 attention outputs 收敛。
   - 存在明确的两阶段结构：先确定“看哪”，再精炼“怎么用”。

2. ✅ **Recurrent Discovery Matters**：
   - 早期静态支持（如 step-1）无法捕捉动态演变的注意力路径。
   - 延迟发现（delayed discovery）在 routing 持续演化的模型中尤为重要。

3. ✅ **Support-Only Reuse 是最优策略**：
   - 冻结 entire attention 分布（Freeze-A@12）损害严重。
   - 只复用 routing support，允许内部动态更新，能最好保留模型行为。

4. ✅ **实际 GPU 加速可达 1.76×**：
   - 在 4K context 下，**late-stage attention 速度提升 1.76×**
   - 整体 T=32 流程仍快 **1.36×**
   - 使用定制 Triton kernel 实现 block-sparse 执行

5. ✅ **质量-效率权衡有利**：
   - 至 2K context，**无统计显著 F1 损失**
   - 4K context 出现约 3.3 point 下降，但仍优于多数压缩方法

---

### **方法的局限性**
1. **依赖 routing 尽早稳定**：
   - 若模型早期 routing 不够 predictive（如 Huginn），则 WISE 更有效；若已很稳定（如 Recurrent-Llama），收益较小。

2. **block size 设计需权衡**：
   - 过细（B=16）稀疏但执行效率低；过粗（B=128）失去稀疏优势。

3. **当前实现为 workload-specialized kernel**：
   - 尚未与 FlashAttention 级别的通用 kernel 深度集成，仍有 co-design 空间。

4. **仅适用于 recurrent 架构**：
   - 不直接推广到 standard autoregressive decoding。

---

### **未来工作方向**
1. **Kernel Co-design**：
   - 开发原生支持 block-sparse recurrence 的 attention kernel，进一步提升效率。

2. **Adaptive Switching Depth**：
   - 动态判断 routing 稳定时机，而非固定 $t_a$。

3. **扩展至其他模态**：
   - 视觉、语音等序列模型中的 recurrent refinement 场景。

4. **结合 KV Cache Compression**：
   - 与 H2O、SnapKV 等方法联合使用，实现端到端长上下文优化。

5. **理论分析深化**：
   - 形式化证明 routing 支持有限时间可识别性，并指导参数设计。

---

> 🔚 **总结一句话**：  
> WISE 揭示了 recurrent 模型中 **“路由先稳，表示后熟”** 的内在机制，并据此提出一种无需训练、高效实用的推理加速方案，在几乎不损失性能的前提下实现高达 **1.76× 的 attention 速度提升**，为长上下文下的 latent reasoning 提供了新的优化范式。

</details>

---

### 2. [SkillGym: Internalizing Human Skills into LLMs for Real-World Problem Solving](https://arxiv.org/abs/2609.27717)

**Authors**: Zhilong Ge, Yuting Shao, Yutao Yang, Yuxuan Cai, Jie Zhou, Kai Chen, Bo Zhang, Qin Chen, Liang He  
**Category**: cs.CL  
**Published**: 2026-09-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.27717v1  

#### Abstract
Human-written agent skills encode rich workflows for real-world problem solving, but are typically used as external inference-time instructions rather than internalized as reusable model capabilities. We introduce \texttt{SkillGym}, a framework that transforms these skills into executable, verifiabl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SkillGym: Internalizing Human Skills into LLMs for Real-World Problem Solving**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前大语言模型（LLMs）在推理、指令遵循和工具使用方面取得了显著进展，但在**真实世界复杂任务中的可靠执行能力仍不足**。尽管人类编写的 **agent skills**（如工作流脚本、操作指南）蕴含丰富的程序性知识，但它们通常仅作为**推理时的外部参考**，而非被模型内部化为可复用的能力。这导致：
- 模型表现依赖检索质量与上下文管理；
- 反复调用技能无法转化为持久的程序性能力；
- 缺乏对完整工作流执行过程的训练支持。

### **提出的新方法与思路**
作者提出了 **SkillGym** 框架，将人类编写的 agent skills 转化为**可执行、可验证的训练环境**，使 LLM 通过交互实践来“内化”这些技能，从而获得可复用的程序性能力。

其核心流程包括：
1. **Skill-to-Task Pipeline**：将抽象的技能文档实例化为具体的任务环境，配备输入资产、交互接口和代码化结果验证器（code-based outcome verifier）。
2. **对比式技能依赖评估**（Contrastive Skill-Dependency Test）：通过对比有无目标技能时的执行成功率，识别出真正依赖该技能的任务，确保训练数据的有效性。
3. **构建大规模训练资源**：发布 2,756 个跨 12 类别的环境和 8,364 条成功轨迹，支持监督微调（SFT）与基于结果奖励的强化学习（RL）。

### **相比现有方法的优势**
| 维度 | 传统方法 | SkillGym |
|------|--------|---------|
| **技能使用方式** | 外部检索调用 | 内部化为模型能力 |
| **训练信号来源** | 工具调用序列、API 示例 | 完整工作流执行 + 结果验证 |
| **任务构造依据** | 手动标注或自动采样 | 基于真实 human-written skills 构建 |
| **验证机制** | 输出格式匹配 | 代码化 outcome verification + 抗捷径检查 |
| **能力泛化性** | 依赖提示工程 | 学习到可迁移的 procedural competence |

> ✅ **核心创新**：首次系统性地将“人类技能”从**推理时的知识源**转变为**训练时的经验源**，实现从“看说明书做事”到“学会怎么做”的转变。

---

## **2. 核心实验方法和设置**

### **使用的数据集与资源**
- **SkillGym 环境库**：
  - 包含 **2,756 个任务环境**，覆盖 12 个主要类别（如 `development`, `business`, `data-ai` 等），63 个子类。
  - 平均每个任务需 **4.5 小时人工构建时间**。
- **轨迹数据集**：
  - 收集 **8,364 条成功轨迹**，来自多个 teacher models 和 harnesses 的组合。
  - 每条轨迹平均包含 **49 次 tool calls**、**63.4k 文本 tokens** 和 **35.2 步交互**，最长达 350 步。
- **公开资源地址**：
  - Envs & Datasets: [HuggingFace](https://huggingface.co/datasets/ecnu-icalk/SkillGym)
  - Model: [HuggingFace](https://huggingface.co/datasets/ecnu-icalk/SkillGym-Agent)
  - Code: [GitHub](https://github.com/ECNU-ICALK/SkillGym)

### **实验设置与评估指标**
#### **评估基准（Benchmarks）**
| 基准 | 任务类型 | 评估方式 |
|------|----------|----------|
| **GDPval-AA v2** | 专业产出（文档、表格等） | Elo rating（基于盲评配对比较） |
| **Terminal-Bench 2.1** | 命令行复杂任务（调试、安全修复） | 成功率 (%) |
| **SkillsBench v1.1** | 技能辅助任务求解 | 成功率 (%)，分 w/ Skills 与 w/o Skills 两种模式 |

#### **基线模型**
- **同规模基线**：
  - `Qwen3.5-35B-A3B`（起点模型）
  - `TerminalTraj-32B`, `OpenThinkerAgent-32B`, `Nemotron-Terminal-32B`, `Agents-A1`
- **前沿公开模型对比**：
  - `Claude Sonnet 4.6`, `GPT-5.4 Mini`, `Gemini 3.1 Pro`, `DeepSeek V4 Pro (Preview)` 等

#### **训练方法**
- 对 `Qwen3.5-35B-A3B` 进行**长上下文监督微调**（SFT），使用 SkillGym 中的 8,364 条成功轨迹。
- 使用 **Megatron backend** 在 16 张 H200 GPU 上训练。
- 推理时采用不同 harness（Codex vs Claude Code）进行公平对比。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | GDPval-AA v2 (Elo) | Terminal-Bench 2.1 (%) | SkillsBench v1.1 (w/ Skills) | SkillsBench v1.1 (w/o Skills) |
|------|---------------------|-------------------------|-------------------------------|--------------------------------|
| Qwen3.5-35B-A3B (Base, Claude Code) | 974 | 39.33 | 23.34 | 12.13 |
| **SkillGym-Agent (Ours)** | **1173 (+199)** | **58.43 (+19.10)** | **51.47 (+28.13)** | **24.51 (+12.38)** |

> 🔺 在 Claude Code harness 下，所有指标均取得显著提升。

### **与基线方法的对比结果**
- **超越同规模 agent 模型**：
  - 相比最强同规模基线 `Agents-A1`，在 GDPval-AA v2 上高出 **189 Elo**。
  - 在 Terminal-Bench 2.1 上高出 **14.61 pp**。
- **优于部分前沿闭源模型**：
  - 在 **skill-assisted SkillsBench** 上达到 **51.47%**，超过：
    - `Claude Sonnet 4.6`（47.2%）
    - `GPT-5.4 Mini`（41.4%）
    - `DeepSeek V4 Pro (Preview)`（50.1%）
  - 在 Terminal-Bench 2.1 上接近 `Claude Sonnet 4.6`（58.43% vs 58.5%）

> 🚀 特别值得注意的是：即使**不提供 inference-time skills**，SkillGym-Agent 依然超过了其 base model 在有技能情况下的表现（24.51% > 23.34%），表明其已**内化了程序性能力**。

### **消融实验结果**
#### **教师模型的影响（Teacher Mixing）**
| 教师组合 | Terminal-Bench ↑ | SkillsBench (w/) ↑ | SkillsBench (w/o) ↑ | GDPval-AA ↓ |
|--------|------------------|--------------------|----------------------|------------|
| GPT-5.4 + Nex-N2-Pro | +6.74 | +5.91 | +0.98 | -98 Elo |
| DeepSeek V4 Pro + GLM-5.2 | +2.24 | +1.83 | +3.31 | -51 Elo |
| **All Teachers** | 最高终端/技能得分 | | | 单一教师更优 |

> 💡 发现：混合教师有助于提升执行类任务表现，但可能损害创意类任务（如 GDPval）；不同能力间存在权衡。

#### **Harness 的影响（Cross-Harness Pooling）**
- 将 Codex 与 Claude Code 下采集的轨迹合并训练，进一步提升了性能：
  - Codex harness 下增益更大（尤其在无技能场景下 +20.39pp）。
  - 表明跨 harness 数据具有互补性，支持**多样化训练配置的价值**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **技能可以被“内化”为模型能力**  
   通过在 SkillGym 环境中进行监督训练，LLM 不再只是“读技能”，而是真正“学会了如何做”，实现了**可复用的 procedural competence**。

2. ✅ **verified workflow experience 显著提升综合能力**  
   基于真实技能构建的长程、多步、可验证轨迹，有效提升了模型在专业产出、命令行操作和技能任务上的表现。

3. ✅ **无需推理时技能也能超越基线**  
   SkillGym-Agent 在 **without skills** 场景下的表现甚至超过 base model 在 **with skills** 时的表现，证明其已具备独立完成任务的能力。

4. ✅ **对比式技能依赖检测有效筛选高质量任务**  
   通过 `(with_skill=success, without_skill=fail)` 的对比策略，识别出真正体现技能价值的任务，保障了训练数据的质量。

### **方法的局限性**
- **构建成本较高**：每个任务平均需 4.5 小时人工设计，难以完全自动化扩展。
- **依赖高质量技能文档**：若原始 skill 描述模糊或错误，会影响任务构建质量。
- **未探索 RL 训练**：目前仅验证了 SFT 效果，尚未在 SkillGym 环境中开展完整的强化学习实验。
- **评估 harness 差异影响结果可比性**：不同模型使用的 harness 和系统提示不同，限制了严格横向对比。

### **未来工作方向**
- 开展基于 outcome reward 的 **Reinforcement Learning**，进一步优化长期规划与容错能力。
- 探索 **自动化的 skill-to-task 转换 pipeline**，降低人工干预成本。
- 构建 **动态难度调节机制**，实现自适应挑战训练。
- 扩展至更多领域（如医疗、法律）并支持多模态交互。

---

> **一句话总结**：  
> SkillGym 成功将“人类技能”从**外部工具书**变为**内在经验值**，让 LLM 在真实世界任务中不仅“知道怎么做”，更能“真的会做”。

</details>

---

### 3. [Resource-Efficient Distributed Recursive Gaussian Processes](https://arxiv.org/abs/2609.26979)

**Authors**: Josephine King, Ali Emre Balci, Raj Thilak Rajan  
**Category**: cs.LG  
**Published**: 2026-09-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.26979v1  

#### Abstract
Gaussian processes (GPs) provide a flexible framework for learning unknown functions from noisy measurements while quantifying predictive uncertainty, making them well suited for estimation in multi-agent systems. However, when measurements are collected by multiple agents, maintaining a unified GP ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Resource-Efficient Distributed Recursive Gaussian Processes

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**多智能体系统中的分布式高斯过程（Gaussian Processes, GPs）在线回归**中存在的两大挑战：
- **通信开销大**：现有分布式 GP 方法（如 Consensus-RGP）依赖大量通信轮次（communication rounds）以达成网络共识，限制了在资源受限设备（如无人机、无线传感器网络）上的应用。
- **计算复杂度高与实时性差**：标准 GP 回归具有 $O(N^3)$ 的时间复杂度，难以处理流式数据。

尽管已有工作解决了核超参数训练的分布式问题，但对**在线、递归式、多输出**场景下的通信效率研究仍不足。

---

### 提出的新方法与新思路
作者提出了两种新型的**通信高效的分布式递归高斯过程算法**：
- **ADMM-RGP**：基于交替方向乘子法（Alternating Direction Method of Multipliers, ADMM），将全局信息向量与信息矩阵的融合建模为带图拉普拉斯约束的优化问题，并采用通信高效变体进行求解。
- **PDMM-RGP**：基于原始-对偶乘子法（Primal-Dual Method of Multipliers, PDMM），通过引入有向边上的辅助变量，在稀疏图上实现更快收敛。

两种方法均运行在**信息形式（information form）** 下，便于分布式更新，并结合了 RGP 的稀疏诱导点近似机制，支持在线学习。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **通信效率** | 显著减少达到相同共识水平所需的通信轮数（K）。例如，在某些拓扑下，ADMM-RGP 用 $K=7$ 即可达到 Consensus-RGP 在 $K=10$ 的性能，节省 30% 通信；PDMM-RGP 在稀疏图上表现更优。 |
| **收敛速度** | 提供理论分析并给出参数选择策略（如 ADMM 中的 $\alpha$, $\tau$ 和 PDMM 中的 $c$），可加速收敛，降低瞬态误差。 |
| **适用性广** | 不绑定特定 kernel 或 basis 构造方式，兼容 inducing points 和 random Fourier features 等表示；适用于任意连通图结构。 |
| **理论保障** | 对两种算法进行了稳定性与收敛性分析，证明其在合理参数条件下 Schur 稳定，确保收敛。 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用来自 **Copernicus Climate Data Store** 的 “ERA5 post-processed daily statistics” 数据集。
- 包含全球范围内 **10米高度的风速 u/v 分量**（multi-output），按 GPS 位置索引。
- 实验聚焦于重建二维风场函数。

---

### 实验设置
- **智能体数量**：$N = 10$
- **通信图结构**：两种不同连接性的图（见 Figure 1），分别代表较密集与较稀疏拓扑。
- **测量模型**：每个 agent 在其位置周围采样输入点 $x \sim \mathcal{N}(p_n, \sigma_s^2 I)$，观测值 $y = f(x) + v, v \sim \mathcal{N}(0, R), R = \text{diag}(0.01, 0.01)$
- **时间步长**：$T = 50$，每步收集 20 个样本，共约 10,000 条训练数据。
- **基础位置（basis locations）**：$P = 400$，构成 $20\times20$ 网格。
- **预测位置**：$2{,}500$ 个测试点，形成 $50\times50$ 网格用于评估。
- **Latent functions**：$Q = 2$，分别预训练于 u 和 v 风速分量。
- **Laplacian weighting**：比较无权重图拉普拉斯与“最优”谱范数最小化权重。

---

### 评估指标
| 指标 | 定义与用途 |
|------|-----------|
| **RMSE**（Root Mean Square Error） | 衡量预测准确性：<br>$\text{RMSE} = \sqrt{\frac{1}{PD'}(\mu_{*,t} - f)^T(\mu_{*,t} - f)}$ |
| **MVOP**（Mean Variance of Predictions） | 衡量 agent 间的一致性（consensus quality）：<br>$\text{MVOP} = \frac{1}{PD'} \sum_{i=1}^{PD'} \text{Var}_n([\mu_{n,t}]_i)$，越低越好 |

---

### 基线方法对比
- **Consensus-RGP** [8]：当前最先进的分布式递归多输出 GP 方法，采用 K 轮平均共识（average consensus）。
- **Centralized RGP**：集中式版本，作为性能上限参考。
- **Centralized GP**：非递归集中式 GP，作为理想基准。

所有方法在同一设置下进行 **100 次 Monte Carlo 仿真**取平均结果。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | 平均 RMSE（95% CI） | 通信效率提升 |
|------|---------------------|--------------|
| All Methods (Consensus-RGP, ADMM-RGP, PDMM-RGP, Centralized RGP) | **0.0862** [0.0860, 0.0864] | 所有方法精度相当 |

> ✅ **关键发现：三种分布式方法在预测精度上与集中式方法几乎一致，且彼此之间无显著差异。**

---

### 与基线方法的对比结果（MVOP vs K）
#### 在 Communication Graph 1（较密）
- **ADMM-RGP 表现最佳**：
  - 当 $K=7$ 时，MVOP 已接近 Consensus-RGP 在 $K=10$ 的水平 → **通信减少 30%**
- **PDMM-RGP 略逊于 ADMM-RGP**

#### 在 Communication Graph 2（较稀疏）
- **PDMM-RGP 显著领先**：
  - MVOP 比其他方法低 **多达三个数量级**
  - $K=5$ 即可达 Consensus-RGP 在 $K=10$ 的共识水平 → **通信减少 50%**
- **ADMM-RGP 同样优于 Consensus-RGP**

> 📊 图 4 显示：**PDMM-RGP 在代数连通性（algebraic connectivity）较低（即图稀疏）时优势明显；而 ADMM-RGP 在高连通图中表现更好。**

---

### 计算开销分析（Figure 5）
- **单轮计算时间随邻居数线性增长**
- **Consensus-RGP 最轻量**
- **PDMM-RGP 斜率最大**（每轮计算成本最高）
- **但因所需通信轮次少，总资源消耗更低**

> ⚖️ 权衡：虽然 PDMM-RGP 每轮更重，但由于收敛快，总体仍更 resource-efficient。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **ADMM-RGP 和 PDMM-RGP 可在不牺牲估计精度的前提下，显著降低通信负担**。
2. ✅ **两种方法在网络共识质量（MVOP）上优于或等于 Consensus-RGP，尤其在有限通信预算下**。
3. ✅ **PDMM-RGP 特别适合稀疏通信图环境，是目前最通信高效的方案**。
4. ✅ **ADMM-RGP 在密集图中收敛最快，适合高连通网络**。
5. ✅ **理论指导的参数选择（如 $\tau < 0$）可进一步加快 ADMM 收敛速度**。

---

### 方法的局限性
- ❗ **依赖同步通信假设**：所有 agent 在每个时间步需同步执行 K 轮通信迭代，未考虑异步或延迟场景。
- ❗ **固定 basis locations 和 kernel 参数**：未联合优化 inducing points 或自适应调整 kernel 超参数。
- ❗ **广播策略虽减通信量，但仍要求邻居间可靠传输**，在动态拓扑中可能受限。
- ❗ **PDMM-RGP 每轮计算开销较大**，对计算能力弱的节点可能不友好。

---

### 未来工作方向
- 🔁 **扩展至异步通信机制**：允许 agent 异步更新，提高鲁棒性。
- 🔄 **支持时变函数与动态图拓扑**：适应移动 agent 场景。
- 🧠 **集成 kernel 自适应或 multiple-model 方法**：增强对非平稳过程的建模能力。
- 🛰️ **应用于 active sensing 与协同控制任务**：如多机器人路径规划、目标跟踪等。
- 🧩 **探索其他 GP 表示形式**：如 Random Feature Approximation 或 Deep GP 的分布式实现。

--- 

> 💡 **总结一句话**：  
> 本论文提出的 **ADMM-RGP** 与 **PDMM-RGP** 是面向资源受限多智能体系统的高效分布式递归 GP 框架，在保持与集中式方法相当精度的同时，大幅降低了通信需求，为边缘智能中的在线贝叶斯学习提供了实用解决方案。

</details>

---

### 4. [When Parallel Drafter Meets Parallel Speculative Decoding](https://arxiv.org/abs/2609.27396)

**Authors**: Fuliang Liu, Xue Li, Kun Qian, Zhibin Wang, Wanchun Dou, Wenyuan Yu, Chen Tian  
**Category**: cs.CL  
**Published**: 2026-09-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.27396v1  

#### Abstract
DSpark-style parallel drafters have made speculative decoding highly effective, yet their draft phase remains serialized on the critical path of every round. Parallel speculative decoding (PSD) overlaps drafting with verification, yet existing methods must guess the accepted prefix and bonus token i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：When Parallel Drafter Meets Parallel Speculative Decoding

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Parallel Speculative Decoding (PSD)** 方法（如 SSD）虽然试图通过将 drafting 与 verification 并行化来加速推理，但由于必须**提前预测 accepted prefix 和 bonus token**，一旦预测错误，整个 batch 就会回退到串行 drafting 模式。这种“**probabilistic fallback**”机制在 batch size 增大时失败率急剧上升（在 batch=16 时高达 96%），严重削弱了并行化的收益。

此外，即使使用高效的 parallel drafter（如 DSpark），其 drafting 阶段仍处于每轮解码的关键路径上，占端到端延迟的 24%-32%，成为性能瓶颈。

### 提出的新方法：DPara
本文提出了 **DPara**，一种全新的 PSD 框架，其核心思想是：

- **分离计算依赖**：将必须等待 verification 结果的计算（bonus token conditioning）与可以独立进行的计算（draft representation 预计算）解耦。
- **预计算所有可能边界**：在 verification 进行的同时，DPara 的 backbone（M-DFlash）**预先为所有可能的 acceptance boundary**（即接受 0 到 d 个 draft token 的所有情况）**并行计算 draft representations**，且不指定 bonus token。
- **轻量级头实时生成**：当 verification 完成后，一个轻量级的 autoregressive (AR) head 立即结合真实的 verification 结果（accepted length `r` 和 bonus token `b`）和对应的预计算表示，几乎瞬时生成下一轮的 draft tokens。

### 相比现有方法的优势
- **完全消除串行回退**：由于 DPara 覆盖了所有可能的 acceptance 边界，无论 verification 结果如何，总有一个预计算分支可用，因此**永远不会发生因预测失败而导致的串行回退**。
- **最大化并行度**：昂贵的 backbone forward pass 完全隐藏在 verification 时间窗口内，只有轻量级的 AR head 处于关键路径上，实现了骨干网络与验证的完全重叠。
- **可扩展性强**：该优势对每个请求独立成立，因此**不会随着 batch size 增大而恶化**，解决了 SSD 等方法的根本缺陷。

---

## 2. 核心实验方法和设置

### 数据集
在 **Qwen3-8B** 和 **Qwen3-14B** 两个模型上，评估了以下七大数据集，覆盖三大类任务：
- **Math**: GSM8K, MATH-500
- **Coding**: HumanEval, MBPP, CodeAlpaca
- **Chat**: MT-Bench, Alpaca

### 实验设置和评估指标
- **硬件**：主实验在 NVIDIA H800 GPU 上进行，PSD 方法额外使用一块 GPU 用于 drafting。
- **批处理大小 (batch size)**：从 1 到 16 进行测试。
- **评估指标**：
  - **Speedup over AR**：相对于标准自回归解码 (autoregressive decoding) 的速度提升倍数。
  - **Average acceptance length (T)**：平均接受长度，衡量 draft token 的有效性。
  - **Throughput (tok/s)**：吞吐量（tokens per second）。

### 基线方法对比
- **Serial SD Baselines**：
  - **EAGLE3**：自回归 drafter。
  - **DFlash / DSpark**：先进的 parallel drafter，但 drafting 串行执行。
- **Parallel SD (PSD) Baselines**：
  - **PEARL**, **SSD**：现有的并行 speculative decoding 方法。
  - **SSD-E3**：使用 EAGLE3 作为 draft model 的 SSD 变体。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **Qwen3-8B** 上，DPara 相对于 AR 的平均速度提升达到 **3.21×**。
- 在 **Qwen3-14B** 上，平均速度提升进一步提高到 **3.52×**，最高单任务提升达 **4.53×**。

### 与基线方法的对比结果
- **超越最强串行基线**：相比最强的串行 drafting 方法 **DSpark**，DPara 在 Qwen3-8B 和 Qwen3-14B 上分别取得了 **11.1%** 和 **5.1%** 的平均速度提升。
- **大幅超越并行基线**：相比并行基线 **PEARL** 和 **SSD**，DPara 的平均速度提升分别高出 **31%** 和 **64%**。
- **批处理下的稳定性**：DPara 的优势在所有 batch size 下均保持稳定。例如，在 batch=16 时，DPara 仍能保持超过 **2×** 的 AR 吞吐量，而 SSD 等方法因频繁回退导致吞吐量急剧下降。

### 消融实验结果
- **M-DFlash 微调的有效性**：仅微调 backbone 部分，就能使平均 acceptance length (T) 提升 **30.7%**，证明了适应多锚点、无特征输入范式的必要性。
- **资源受限场景**：
  - **DPara-A10**：在弱 GPU (A10) 上进行 drafting，仍能保留 DPara **98%** 的加速效果。
  - **DPara-single**：在单 GPU 上实现 co-location，在小 batch (1-4) 下能保留 **90%-95%** 的加速效果。

---

## 4. 关键结论和发现

### 主要发现
1. **Outcome-Independent Precomputation 是关键**：通过预计算所有可能的 acceptance 边界，DPara 成功地将 backbone 执行与不确定的 verification 结果解耦，从而**彻底消除了 PSD 中的串行回退问题**。
2. **高性能与高效率兼得**：DPara 不仅实现了 state-of-the-art 的推理速度，而且其优势在各种 batch size 和不同任务类型下都具有鲁棒性。
3. **架构复用性强**：DPara 成功地将已有的高效 parallel drafter (DSpark) 架构改造为适用于 PSD 的形式，展示了强大的兼容性和实用性。

### 方法的局限性
- **信息损失**：由于 backbone 必须在没有完整 target features 的情况下运行（feature-less anchors），其预测能力受到一定限制，导致 acceptance length (T) 略低于串行的 DSpark（4.41 vs 5.36）。但实验证明，消除关键路径上的 backbone 开销所带来的收益远超于此。
- **需要微调**：标准的 DFlash 模型无法直接满足 DPara 的需求，必须通过微调得到 M-DFlash backbone。

### 未来工作方向
- 探索更高效的多锚点预测架构，以进一步减少信息损失，提升 acceptance length。
- 将 DPara 的思想应用于其他类型的生成模型或更复杂的推理任务中。
- 研究在动态调整 draft length 或更复杂调度策略下的 DPara 变体。

</details>

---

### 5. [Towards Efficient Reasoning: Learning Causal Shortcuts for Diffusion Language Models](https://arxiv.org/abs/2609.28272)

**Authors**: Dian Jin, Kairong Han, Baohong Li, Xinpeng Dong, Zijing Hu, Nuanqiao Shan, Fei Wu, Kun Kuang  
**Category**: cs.CL  
**Published**: 2026-09-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.28272v1  

#### Abstract
Diffusion Language Models (DLMs) have attracted significant attention for their strong reasoning ability. However, under a bidirectional attention mechanism, DLMs operate over an exponentially large exploration space compared to autoregressive models (ARMs), making it challenging to focus on reasoni...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Towards Efficient Reasoning: Learning Causal Shortcuts for Diffusion Language Models —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **问题背景**：Diffusion Language Models (DLMs) 虽然具备强大的推理能力，但由于采用双向注意力机制和随机掩码策略，其生成轨迹空间呈指数级增长，导致模型难以聚焦于对推理有指导意义的关键 token（reasoning-guiding tokens）。
- **核心挑战**：在如此庞大的探索空间中，模型容易优先学习高频但非关键的 token，而忽略真正引导正确推理路径的 token，从而影响推理效率和准确性。

### ✅ 提出的新方法与新思路
作者提出了 **Causal Shortcut Learning (CSL)** 框架，其核心思想是：
- 定义 **因果捷径（causal shortcuts）**：即能够覆盖整个序列并提供明确推理引导的 token 链条。
- 引入 **条件互信息（Conditional Mutual Information, CMI）** 作为衡量 token 推理重要性的指标。CMI 衡量的是揭示某个 token 后，其余被掩码 token 的不确定性（熵）下降程度，反映该 token 对整体推理的因果影响力。
- 设计 **逐步提取（step-by-step extraction）算法** 来构建因果捷径集合，避免一次性提取导致的 token 聚集问题。
- 在训练中应用 **并行优先掩码（parallel prioritized masking）**，使模型在训练阶段更关注这些高 CMI 的因果捷径 token。

### ✅ 相比现有方法的优势
| 方法 | 局限性 | CSL 的优势 |
|------|--------|-----------|
| Random Masking | 均匀掩码，无法区分 token 重要性 | 显式识别并强化关键推理 token |
| Entropy/Loss-aware 方法（如 MGDM, GIFT） | 将“难度”或“不确定性”等同于“重要性”，易误判远距离简单 token 为重要 | 从**因果信息流角度**定义重要性，更具语义合理性 |
| Blockwise / Semi-autoregressive 方法 | 引入局部顺序约束，牺牲部分并行性 | 保持完全并行生成能力，仅通过掩码策略引导 |

> ✅ **关键优势**：CSL 不改变 DLM 的架构或解码方式，仅通过改进训练中的 token 选择与掩码策略，即可显著提升推理效率与准确率。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **数学推理任务**（7个基准）：
  - GSM8K, MATH-500, SAT, Sudoku, GPQA, MMLU-STEM, ARC-C
- **代码生成任务**（2个基准）：
  - HumanEval, MBPP
- **训练数据**：
  - 数学领域：Math-CoT
  - 代码领域：OPC-SFT-Stage2

### ⚙️ 实验设置
- **基础模型**：
  - `LLaDA-8B-Instruct`
  - `LLaDA-1.5B`
- **训练配置**：
  - 使用 LoRA 进行高效微调（rank=8, lr=2e-4）
  - 数学任务：上下文长度 2048，训练 4–8 轮
  - 代码任务：上下文长度 1024，训练 4 轮
  - 批大小：基于 4×A100 GPU
- **推理设置**：
  - 多种生成长度测试（如 256, 512）
  - 0-shot 设置，固定随机种子（seed=42）

### 📊 评估指标
- 主要指标：**Accuracy (%)**
- 辅助分析指标：
  - 平均收敛时间步（Entropy decay step）
  - 累积熵（Cumulative entropy）
  - 数字与操作符的生成顺序与置信度（Entropy & Generation Order）

### 🔁 基线方法对比
| 基线 | 方法类型 |
|------|---------|
| SFT-only | 标准监督微调 |
| DiBT (DiffusionBert) | 基于熵调整掩码概率 |
| MGDM | 基于损失重加权 |
| Blockwise | 分块半自回归生成 |
| DSFT | 数值感知 + Curriculum Masking |
| GIFT | 重要性感知微调 |

---

## 3. 主要实验结果和性能指标

### 📈 总体性能表现（见 Table 2 & 3）

#### ✅ 数学推理任务（平均提升）
| 模型 | 平均准确率提升（vs SFT） |
|------|------------------------|
| LLaDA-8B-Instruct (CSL) | **+1.92%** |
| LLaDA-1.5B (CSL) | **+1.58%** |

> 🔥 在 **MATH-500 (256长度)** 上取得最大增益：**+4.20%**

#### ✅ 代码生成任务（Table 3）
| 模型 | HumanEval ↑ | MBPP ↑ | 平均提升 |
|------|-------------|--------|--------|
| CSL | 35.37% (+1.22%) | 44.36% (+3.89%) | **+2.30%** |

> 💡 特别是在长序列生成（512）下，CSL 显著优于其他方法，表明其有效缓解了错误累积问题。

### 🔍 消融实验结果（Table 4）

| 提取策略 | K=0.1L | K=0.2L | K=0.3L | 观察结论 |
|--------|--------|--------|--------|----------|
| Random | ↓ 性能下降 | ↓↓ 严重下降 | ↓↓↓ 崩溃 | 随机选择无效 |
| One-step | 中等提升 | 轻微提升 | 下降 | 存在 token 聚集问题 |
| **Step-by-step** | 提升 | ✅ 最优 | 微降 | **K=0.2L 效果最佳** |

> ✅ **关键发现**：
> - 因果捷径数量需适中（约 20% 序列长度），太少则不足以引导，太多则破坏依赖关系。
> - 逐步提取（step-by-step）明显优于一步提取（one-step），解决了 token 聚类问题。

### 📉 推理动态分析（Figure 5 & Table 5–6）

| 指标 | SFT | CSL | 差距 |
|------|-----|-----|------|
| 平均熵降至 1.0 所需时间步（GSM8K） | 0.901 | **0.814** | ↓ 8.7% |
| 累积熵（MATH-512） | 138.4 | **121.1** | ↓ 17.26 |

> ✅ **结论**：CSL 实现更快的熵衰减与更低的累积不确定性，说明其推理过程更稳定、收敛更早。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **因果捷径能显著提升 DLM 的推理效率与准确性**：
   - 通过 CMI 可有效识别对推理具有强引导作用的 token。
   - 将这些 token 构建成“因果捷径”并在训练中加以强调，可形成更短、更可靠的推理路径。

2. **逐步提取优于单步提取**：
   - 单步提取会导致高 CMI token 聚集在 prompt 附近，无法覆盖完整推理链。
   - 逐步提取通过迭代更新 CMI 分布，实现跨区域 token 选择，形成连贯的推理轨迹。

3. **并行优先掩码策略安全且高效**：
   - 因果捷径 token 间依赖较弱，适合并行预测。
   - 该策略提升了梯度更新效率，同时未破坏语义一致性。

4. **CSL 缓解了长序列中的错误累积问题**：
   - 在 HumanEval 和 MBPP 上，CSL 在长生成长度下仍保持领先，而多数基线出现性能退化。

### ⚠️ 方法的局限性
- **预处理开销大**：CMI 计算需要大量前向传播，虽引入评分模型（score model）缓解，但仍增加训练前成本。
- **依赖高质量 CoT 数据**：当前方法假设存在清晰的推理链条，对于无明确步骤的任务可能效果受限。
- **固定比例 K=0.2L 不普适**：不同难度任务最优 K 不同，未来需动态调整机制（如基于熵阈值）。

### 🔮 未来工作方向
1. **探索内部表示作为代理信号**：
   - 利用隐藏状态或注意力模式直接估计 token 重要性，减少对外部计算的依赖。
   
2. **扩展至更大规模模型**：
   - 当前实验限于 8B 级别，预期 CSL 在更大 DLM 上收益更显著。

3. **结合强化学习进一步优化路径**：
   - 将因果捷径视为潜在动作空间，用 RL 优化推理策略。

4. **动态因果捷径选择**：
   - 根据输入复杂度自适应决定 K 值或提取终止条件。

---

## ✅ 总结

**Causal Shortcut Learning (CSL)** 是一种简洁而高效的 DLM 推理增强框架。它通过 **CMI 指标识别关键推理 token**，利用 **逐步提取构建因果捷径**，并通过 **并行优先掩码** 在训练中强化这些路径。实验证明，CSL 在多个数学与代码基准上**持续超越 SFT 及多种先进变体**，尤其在复杂任务（如 MATH-500）上达到 **+4.20%** 的绝对提升，并展现出更快的收敛速度与更强的稳定性。

> 🌟 **一句话总结**：  
> CSL 揭示了 DLM 推理的本质在于“找到正确的因果路径”，并通过学习这些“因果捷径”实现了高效、准确的语言生成。

</details>

---

### 6. [CerebroSim: Scalable Whole-Brain Simulator at 100-Trillion-Synapse Scale on the LineShine Supercomputer](https://arxiv.org/abs/2609.27482)

**Authors**: Guangnan Feng, Tianxiang Lyu, Hao Huang, Honghui Liang, Jingjing Li, Zhiguang Chen, Yutong Lu  
**Category**: cs.DC  
**Published**: 2026-09-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.27482v1  

#### Abstract
Building executable brain models is essential for moving neuroscience from description to mechanism and prediction. Human-brain-scale spiking simulation is constrained by highly irregular communication, multithreaded spike delivery, and the memory cost of sparse connectivity. We present CerebroSim, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*CerebroSim: Scalable Whole-Brain Simulator at 100-Trillion-Synapse Scale on the LineShine Supercomputer*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
人类全脑尺度的 spiking neural network（SNN）模拟面临三大核心瓶颈：
- **不规则通信**：神经元发放的 spikes 具有高度稀疏性和异构延迟，导致分布式系统中通信效率低下。
- **多线程竞争**：在单节点内，多个线程并发更新突触后神经元状态时易引发 race condition，依赖锁或原子操作带来显著同步开销。
- **内存压力巨大**：100万亿级 synapses 的连接索引和权重存储占用海量内存，限制了模型规模。

现有框架如 NEST、Fugaku 上的模拟器等虽能实现大规模模拟，但在通信优化、内存压缩和并行效率方面仍受限于传统设计。

---

### 提出的新方法与创新
CerebroSim 提出了三项协同优化技术，构成一个端到端可扩展的 whole-brain simulation 框架：

#### ✅ **Delay-aware Spike Broadcast (DSB)**
- 将全局不规则 spike 广播重构为基于虚拟拓扑（virtual topology）的 hop-by-hop 聚合转发。
- 利用生物延迟窗口作为“调度松弛”（schedulable slack），实现 **communication-computation overlap**。
- 引入轻量级传输语义（Verbs send/recv），减少协议开销，避免 MPI 中复杂的 tag 匹配与缓冲区管理。

> **优势**：显著降低小包注入频率，提升网络带宽利用率，并隐藏大部分通信延迟。

#### ✅ **Race-free Synaptic Dynamics Computation (RSDC)**
- 通过静态数据划分与确定性调度，在初始化阶段规避写冲突，彻底消除运行时的 mutex 或 atomic 操作。
- 结合 HBM-aware 软件预取策略（Inter-iteration 和 Inter-loop prefetching），缓解内存访问瓶颈。

> **优势**：实现无锁多线程 spike delivery，提高缓存局部性和吞吐量，尤其适用于现代 NUMA 架构。

#### ✅ **Sparse Synapse Storage Compression (3SC)**
包含两个关键技术：
- **Synapse-Aware Compressed Index (SACI)**：仅对本地存在突触目标的 pre-synaptic neurons 存储索引条目，去除空项浪费。
- **FlySyn：on-the-fly synapse regeneration**  
  基于伪随机数生成机制，在运行时按需重建 synapse 连接（post-synaptic ID、weight、delay），无需显式存储。

> **优势**：将 synapse 存储从 $O(N + E)$ 压缩至接近 $O(\alpha N)$，其中 $\alpha$ 是跨体素连接系数（远小于1），极大节省内存。

---

### 相比现有方法的优势
| 维度 | 现有方法（如 NEST、Fugaku 模拟） | CerebroSim |
|------|-------------------------------|------------|
| 通信模式 | 依赖 MPI_Allgatherv / Alltoall，产生大量冗余消息 | DSB 实现定向聚合广播，减少无效流量 |
| 多线程同步 | 使用 atomic 或锁保护共享状态 | RSDC 完全消除运行时竞争 |
| 内存使用 | 显式存储所有 synapse 参数 | FlySyn 实现零显式存储，仅保留再生种子 |
| 可移植性 | 高度耦合底层架构 | 抽象为独立库（DSB Lib, SDT Lib, FlySyn Lib），支持迁移 |

---

## 2. 核心实验方法和设置

### 数据集与模型构建
- **数据来源**：基于 **T1-weighted MRI** 和 **diffusion-weighted MRI** 构建人脑结构连接图谱。
- **神经元分配**：
  - 总计 **86 billion neurons**，分布在约 20,000 个体素（voxel）中。
  - 皮层（cortex）占 ~16 billion，皮层下结构（subcortex, cerebellum 等）占 ~70 billion。
- **连接建模**：
  - 使用 row-normalized connectivity matrix 表示体素间投射概率。
  - 突触参数（delay, weight）服从正态分布；无 synaptic plasticity（因缺乏全脑尺度参数支持）。
- **神经元模型**：Leaky Integrate-and-Fire (LIF)，驱动输入为 post-synaptic current + Poisson background noise。

---

### 实验平台
- **超级计算机**：**LineShine Supercomputer**
  - 节点数：22,680
  - 每节点配置：
    - 2× LX2 处理器（共 608 cores）
    - HBM: 64GB @ ~4TB/s，DDR5: 512GB
    - 8× 200Gbps NIC，总带宽 1.6Tbps/node
    - 互联拓扑：2-plane × 4-rail fat-tree，单跳延迟 1.07μs
- **实际使用规模**：**18,432 nodes**, 跨越 **11.2 million cores**

---

### 评估指标
| 类别 | 指标 |
|------|------|
| **可扩展性** | Weak Scaling Efficiency, Strong Scaling Efficiency |
| **性能** | Sustained Performance (PFlop/s), Runtime Breakdown |
| **通信效率** | Message Size Distribution, Non-overlapped Communication Time |
| **内存效率** | Memory Footprint Reduction (%) |
| **正确性验证** | Raster Plots 对比癫痫 vs 正常状态动态 |

---

### 基线方法对比
- **通信基线**：MPI collectives（MPI_Allgatherv, MPI_Alltoallv）
- **存储基线**：传统 CSC（Compressed Sparse Column）格式
- **同步基线**：Atomic-based spike delivery

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 数值 |
|------|------|
| 模拟规模 | 86 billion neurons, **100 trillion synapses** |
| 使用资源 | 18,432 nodes (~11.2M cores) |
| 持续性能 | **24.44 PFlop/s** |
| 弱扩展效率（排除 I/O） | **91%** |
| 强扩展效率（whole-brain case） | **94%** |
| 单核 RNG 吞吐（FlySyn） | **112 GOPS/core** |

---

### 与基线方法对比结果

#### 🔹 DSB vs. MPI Collectives（图9a）
- 在相同规模下，DSB 相比 MPI_Allgatherv 实现 **59%–262% 的端到端加速**。
- 原因：
  - DSB 减少了 **spike filtering overhead**（只收所需数据）。
  - 更高的通信-计算重叠率，非重叠通信时间极低。
- 不同虚拟拓扑比较显示：**3D-HyperX** 表现最优，优于 Dragonfly 和其他维度配置。

#### 🔹 RSDC 多线程优化效果（图10a）
- 在 144–576 节点范围内，相比 baseline（含 atomic 操作）：
  - Spike delivery 时间下降 **~30–40%**
  - “Other” 开销（同步、负载均衡）也明显降低
- HBM + Prefetching（图10b）进一步提速：
  - 将 synapse array 和 index 放入 HBM 提升约 20%
  - Inter-iteration prefetching（stride=1）再提速 ~5%，Inter-loop prefetching 额外增益 ~1%

#### 🔹 内存压缩效果（图11）
在 144 节点子集上测试（覆盖 ~0.78% 脑区）：
| 方案 | 内存占用（相对 baseline） |
|------|------------------------|
| Baseline（传统 CSC） | 100% |
| + SACI | **65.0%**（↓35%） |
| + SACI + FlySyn（RNG再生） | **21.8%**（↓78.2%） |

> 表明：**on-the-fly regeneration 是突破内存墙的关键手段**。

---

### 消融实验结果
- **DSB ablation**：关闭 delay-aware scheduling 或使用 MPI fallback 后，通信开销上升 2–3 倍，扩展性下降。
- **RSDC ablation**：启用 atomic 更新后，多线程扩展性显著退化，尤其在高并发场景。
- **3SC ablation**：恢复显式 synapse 存储后，内存需求超出可用容量，无法完成全脑模拟。

---

## 4. 关键结论和发现

### 主要发现
1. **极端规模全脑模拟已成为现实**：
   - CerebroSim 成功实现了 **86B neurons + 100T synapses** 的 human-brain-scale 模拟，是目前公开报道中最大规模之一。
2. **通信不再是不可逾越的障碍**：
   - DSB 证明可通过虚拟拓扑聚合 + 延迟感知调度，将不规则通信转化为高效结构化流程。
3. **内存瓶颈可通过“以算换存”解决**：
   - FlySyn 展示了 **deterministic regeneration** 在大规模 SNN 中的巨大潜力，为未来 neuromorphic 和 in-memory computing 提供新思路。
4. **同步开销可以被完全规避**：
   - RSDC 表明，通过前期静态规划而非运行时控制，可在保证正确性的前提下实现 lock-free 并行。

---

### 方法的局限性
1. **依赖高质量结构连接先验知识**：
   - 当前模型基于 DWI/MRI 推断连接，尚未整合功能连接或细胞级精度数据。
2. **未包含 synaptic plasticity**：
   - 因缺乏全脑尺度可参数化的 STDP 或 Hebbian 规则，当前模型为静态连接。
3. **FlySyn 依赖伪随机一致性**：
   - 若不同进程/时间步种子不一致，可能导致再生错误，需严格控制执行环境。
4. **硬件依赖较强**：
   - 最佳性能依赖 HBM、SVE/SME 指令集和高性能 RDMA 网络，通用性受限。

---

### 未来工作方向
1. **融合动态可塑性机制**：
   - 结合实验数据发展 scalable plasticity models，支持 learning and memory 模拟。
2. **集成 neurovascular coupling 模型**：
   - 将 spiking activity 映射为 fMRI/BOLD 信号，桥接微观模拟与宏观观测。
3. **支持 real-time closed-loop stimulation**：
   - 用于虚拟药物筛选（virtual drug screening）和干预假说检验（intervention hypothesis testing）。
4. **向 neuromorphic hardware 移植**：
   - 利用 CerebroSim 的算法设计指导类脑芯片架构开发。
5. **开放框架生态建设**：
   - 推动 DSB、RSDC、FlySyn 库在其他稀疏图计算任务中的应用（如 sparse AI、agent-based modeling）。

---

> ✅ **总结一句话**：  
> CerebroSim 不仅刷新了 whole-brain simulation 的性能纪录，更重要的是提出了一套面向 **sparse, irregular, delay-constrained workload** 的 co-design 范式，为下一代 brain-inspired computing 提供了坚实基础。

</details>

---

### 7. [Do We Need Complex Topology Control? Distinct-Peer Random Routing Improves Cost-Efficiency in Sparse Multi-Agent Debate](https://arxiv.org/abs/2609.27150)

**Authors**: Boxuan Wang, Zhuoyun Li, Xiaowei Huang, Yi Dong  
**Category**: cs.AI  
**Published**: 2026-09-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.27150v1  

#### Abstract
Multi-agent debate (MAD) has emerged as a promising paradigm for improving the reasoning accuracy of large language models (LLMs) through iterative peer interaction. Communication topology plays a central role in this process, motivating increasingly sophisticated mechanisms that learn, adapt, or dy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Do We Need Complex Topology Control? Distinct-Peer Random Routing Improves Cost-Efficiency in Sparse Multi-Agent Debate*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文探讨了一个关键问题：**在稀疏多智能体辩论（Sparse Multi-Agent Debate, MAD）中，是否真的需要复杂的拓扑控制机制来提升推理准确性和效率？**  
尽管已有研究提出通过学习、自适应或动态重构通信图（如 AgentPrune、G-Designer）来优化信息流动，但这些方法增加了系统复杂性和计算开销。作者质疑：**更简单的路由策略是否足以实现甚至超越这些复杂方法的表现？**

### 🚀 提出的新方法与新思路
1. **Random-NoDup 路由机制**  
   - 每轮中每个 agent 随机选择两个**不同的新对等体（distinct peers）**进行交互，且允许后续轮次再次访问之前接触过的 agent。
   - 这是一种**无状态、无需学习、无需不确定性估计**的简单随机路由策略。

2. **双轴分析框架（Dual-Axis Evaluation Framework）**  
   - 将 MAD 的性能影响因素解耦为两个独立维度：
     - **Routing（路由）**：决定谁与谁通信
     - **Stopping（停止）**：决定通信持续多少轮
   - 该框架有助于分离“拓扑设计”与“时序控制”的影响，提供更清晰的归因分析。

3. **轻量级停止机制（Lightweight Stopping）**
   - 引入基于多数稳定性（majority-stability）或不确定性门控（uncertainty-gated）的自适应停止规则，提前终止无益的讨论以节省 token 成本。

### ⚖️ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **简洁性** | Random-NoDup 不依赖额外模型、奖励信号或不确定性建模，易于复现和部署 |
| **成本效益** | 在相似甚至更低 token 消耗下达到更高或相当的准确率 |
| **鲁棒性** | 在多个任务和不同 LLM 上表现稳定，泛化能力强 |
| **基准价值** | 为未来 topology control 方法提供了强有力的 baseline，强调应先与简单方法比较再论证其必要性 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验在五个具有挑战性的基准上进行：
- **ARC-Challenge**
- **ScienceQA-text**（纯文本子集）
- **GSM8K**（数学推理）
- **MMLU-Pro**（多任务理解）
- **GPQA-Diamond**（研究生级别问答，高难度）

### ⚙️ 实验设置
- **Agents 数量**：8 个同构 agent（使用相同 LLM）
- **每轮接收消息数**：k = 2
- **最大辩论轮数**：T = 14（完整辩论），短版本在第 3 轮结束（即两次更新后）
- **模型主干**：主要使用 `GPT-4o-mini`，并在消融实验中扩展至 `GPT-4.1-mini`, `GPT-4.1`, `GPT-5.6Luna`
- **生成参数一致**：温度、prompt 等保持不变以确保公平比较
- **重复次数**：5 次随机种子运行，报告均值 ± 标准差

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 最终答案正确率（macro average across tasks） |
| **Tokens per question (k)** | 输入 + 输出总 token 数（衡量推理成本） |
| **Accuracy-Cost Trade-off** | 综合考虑准确率与 token 开销的性价比 |

### 🔁 基线方法对比
| 类别 | 方法 |
|------|------|
| **Reference Routing** | Ring / Sparse MAD, Small-world-inspired |
| **Structured/Adaptive Routing** | Stage-based switching, Random switching, Uncertainty-guided routing |
| **Learned Topology Control** | AgentPrune-style temporal-edge adaptation（基于奖励学习并剪枝边） |
| **Stopping Methods** | Fixed round=3, Majority-stability stopping, Uncertainty-gated stopping |
| **Joint Control** | Uncertainty-guided routing + adaptive stopping |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 方法 | Macro Acc (%) | Tokens per Q (k) |
|------|----------------|------------------|
| **Fully Connected MAD** | ~76.0 | >80 |
| **Ring / Sparse MAD** | 76.27 | 57.25 |
| **Small-world-inspired** | **77.07** | 57.52 |
| **Random-NoDup (full rounds)** | 76.87 | 57.38 |
| **Random-NoDup (round 3)** | **77.33** | **12.36** |
| **Uncertainty-gated Stopping** | 77.20 | 18.60 |
| **AgentPrune-style (learned)** | 76.27 | 57.59 |

> ✅ **关键观察**：
> - **Random-NoDup（full）** 准确率优于 Ring 和 AgentPrune，在多数任务上持平或领先。
> - **Short Random-NoDup（仅3轮）** 达到最高 macro accuracy（77.33%），同时 token 消耗仅为全轮次的约 **1/5**。
> - 所有复杂 routing 控制（stage-based、uncertainty-guided）均未在整体 trade-off 上超越 Random-NoDup。

### 🔍 与基线方法的对比结果
| 对比项 | 结果 |
|--------|------|
| **vs. Ring / Sparse MAD** | Random-NoDup 在 ARC-Challenge、MMLU-Pro、GPQA-Diamond 上准确率更高，token 成本相近 |
| **vs. Small-world-inspired** | 准确率略低（77.07 vs 76.87），但结构更简单 |
| **vs. Uncertainty-guided routing** | 后者在个别任务（如 GSM8K）有优势，但整体无显著增益，且需额外计算不确定性 |
| **vs. AgentPrune-style learned adaptation** | 准确率相当（76.27 vs 76.87），但 Random-NoDup 无需训练、校准或奖励建模 |
| **vs. Fully Connected MAD** | 以 <20% 的 token 成本达到相近甚至更高的准确率 |

### 🔧 消融实验结果（Ablation Studies）

#### （1）跨模型验证（Table 2）
在四种不同 LLM 上重复实验：
- **Routing ablation**：在全部 12 个 model-task 对比中，**没有任何一种 structured 或 learned routing 显著优于 Random-NoDup（full）**
- **Stopping ablation**：
  - **Short Random-NoDup（round 3）** 在所有模型上 token 成本最低
  - 宏观准确率在 3/4 模型上最高
  - 唯一例外是 `GSM8K` 上某模型因早期收敛慢导致精度下降 → 表明 stopping 效果依赖任务特性

#### （2）回合数影响分析（Figure 3）
- 宏观准确率在 **round 2–5** 达到平台期，之后增长停滞
- 从 round 3 到 round 14，累计 token 增加超过 5 倍，但平均准确率变化不超过 ±2%
- 支持“早停有效”的结论

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Random-NoDup 是一个极强的 baseline**
   - 即使是最简单的随机非重复对等路由，在稀疏 MAD 中也能实现与先进 topology control 方法相当甚至更好的准确率。
   - 其成功可能源于避免了局部固化信息流，促进了全局信息探索。

2. **复杂拓扑控制未必带来收益**
   - 包括 stage-based switching、uncertainty-guided routing 和 learned edge adaptation 在内的多种高级方法，**未能系统性地改善 accuracy-cost trade-off**。
   - 特定任务上的微弱优势不足以证明其增加的复杂性合理。

3. **轻量级 stopping 是提升效率的关键**
   - 将辩论限制在前 3 轮（仅两次通信更新），即可获得接近最优的准确率，同时将 token 消耗降低 **75–85%**。
   - 自适应 stopping（如 uncertainty-gated）有一定作用，但固定短轮次已足够高效。

4. **应优先评估简单 baseline**
   - 新提出的 topology control 方法应在相同的协议下与 Random-NoDup 和 short debate 进行比较，否则难以判断其真实价值。

### ⚠️ 方法的局限性
- **假设同构 agent**：所有 agent 使用相同 LLM，未考虑异构或多角色场景
- **固定通信度（degree=2）**：未探索更广的连接模式
- **任务依赖性**：short debate 在部分数学任务（如 GSM8K）上可能导致性能下降
- **缺乏理论解释**：为何 Random-NoDup 如此有效仍需进一步研究

### 🔮 未来工作方向
1. 探索 Random-NoDup 在异构 agent、角色分工、外部工具调用等更复杂 MAD 场景中的表现
2. 研究是否存在比“轮数”更优的 stopping signal（如语义收敛、逻辑完备性）
3. 设计兼具 simplicity 与 adaptivity 的 hybrid routing-stop 策略
4. 将本工作提出的双轴框架推广为 MAD 方法的标准评估范式

---

> 💡 **一句话总结**：  
> 本文表明，在稀疏多智能体辩论中，**一个极其简单的“随机选择两个不同对等体 + 提前停止”策略，就能在大幅降低成本的同时取得媲美甚至超越复杂拓扑控制方法的效果**，从而挑战了“必须用学习或自适应机制优化通信图”的主流假设，并呼吁社区重视强而简单的 baseline。

</details>

---

### 8. [Hunyuan-A13B Technical Report](https://arxiv.org/abs/2609.27284)

**Authors**: Tencent Hunyuan Team, Ao Liu, Botong Zhou, Can Xu, Chayse Zhou, ChenChen Zhang, Chengcheng Xu, Chenhao Wang, Decheng Wu, Dengpeng Wu, Dian Jiao, Dong Du, Dong Wang, Feng Zhang, Fengzong Lian, Guanghui Xu, Guanwei Zhang, Hai Wang, Haipeng Luo, Han Hu, Huilin Xu, Jiajia Wu, Jianchen Zhu, Jianfeng Yan, Jiaqi Zhu, Jihong Zhang, Jinbao Xue, Jun Xia, Junqiang Zheng, Kai Liu, Kai Zhang, Kai Zheng, Kejiao Li, Keyao Wang, Lan Jiang, Lixin Liu, Lulu Wu, Mengyuan Huang, Peijie Yu, Peiqi Wang, Qian Wang, Qianbiao Xiang, Qibin Liu, Qingfeng Sun, Richard Guo, Ruobing Xie, Saiyong Yang, Shaohua Chen, Shihui Hu, Shuai Li, Shuaipeng Li, Shuang Chen, Suncong Zheng, Tao Yang, Tian Zhang, Tinghao Yu, Weidong Han, Weijie Liu, Weijin Zhou, Weikang Wang, Wesleye Chen, Xiao Feng, Xiaoqin Ren, Xingwu Sun, Xiong Kuang, Xuemeng Huang, Xun Cao, Yanfeng Chen, Yang Du, Zhen Yang, Yangyu Tao, Yaping Deng, Yi Shen, Yigeng Hong, Yiqi Chen  
**Category**: cs.AI  
**Published**: 2026-09-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.27284v1  

#### Abstract
We present Hunyuan-A13B, an open-source large language model based on a Mixture-of-Experts architecture. It contains 80 billion total parameters but activates only 13 billion during inference, balancing model capability, computational efficiency, and deployment cost. The model is pretrained on a rig...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Hunyuan-A13B Technical Report 核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的大型语言模型（LLMs）虽然在能力上接近**Artificial General Intelligence (AGI)**，但其部署通常需要巨大的计算资源，导致高推理延迟和硬件成本，限制了广泛应用。Hunyuan-A13B 旨在解决这一矛盾——如何在保持强大模型性能的同时，显著降低推理开销。

### 提出的新方法与创新点
1. **Sparse Mixture-of-Experts (MoE) 架构设计**  
   - 总参数量为 **80B**，但在推理时仅激活 **13B 参数**，通过专家选择机制实现高效计算。
   - 采用细粒度 MoE 结构：1 个共享专家 + 64 个专用专家，每步激活 8 个非共享专家。
   - 实验证明单个共享专家即可有效提升性能，增加更多共享专家收益递减。

2. **高质量、STEM 强化的预训练语料构建**  
   - 使用超过 **20T tokens** 的严格过滤数据进行预训练。
   - 特别强化 STEM 领域的数据采集与清洗流程，提取出 **250B 高质量 STEM tokens**，显著增强数学与科学推理能力。

3. **双模式 Chain-of-Thought (Dual-Mode CoT) 推理框架**  
   - 支持两种推理模式：
     - **Fast-thinking mode**（`/no_think`）：用于简单任务，输出简洁，低延迟。
     - **Slow-thinking mode**（`/think`）：支持多步复杂推理，适用于高难度问题。
   - 用户可按需切换，灵活平衡效率与准确性。

4. **系统化后训练流程（Post-training Pipeline）**  
   - 分两个阶段优化：
     - **Reasoning-oriented SFT & RL**：专注数学、编程、逻辑、科学等领域的复杂推理能力。
     - **All-Scenarios SFT & RL**：覆盖创意写作、多轮对话、角色扮演、工具调用等通用场景。
   - 在 RL 阶段使用 **Group Relative Policy Optimization (GRPO)** 和多种 reward 模型（如 outcome reward、sandbox execution）进行强化学习。

5. **长上下文扩展至 256K**  
   - 通过 NTK-aware positional encoding 技术逐步将 context length 扩展到 **256K tokens**。
   - 支持超长文本理解与跨文档推理。

---

### 相比现有方法的优势
| 维度 | Hunyuan-A13B 优势 |
|------|------------------|
| **计算效率** | 激活参数仅为 13B，远低于多数 dense 或 MoE 模型（如 Qwen3-A22B 激活 22B），推理吞吐更高。 |
| **推理灵活性** | 双 CoT 模式允许动态调整推理深度，适应不同应用场景。 |
| **STEM 能力** | 数据层面强化 STEM 内容，在数学与科学任务中表现优异。 |
| **Agent 能力** | 在工具调用、任务规划等 agent 场景中超越更大模型。 |
| **开源开放性** | 完全开源，支持主流推理框架（vLLM, SGLang, TensorRT-LLM），便于部署。 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### 预训练数据
- 总规模：**>20T tokens**
- 来源：多样化领域文本，重点加强 STEM 数据（教科书、竞赛题、科研文献）
- 处理流程：
  - 数据去重、低质过滤、denoising
  - 自动提取 + 语义级 deduplication
  - 多维难度分级体系辅助筛选

#### 后训练数据
| 类型 | 数据来源 | 关键处理方式 |
|------|--------|-------------|
| **数学推理** | 教材、标考、奥赛题（AIME/MATH） | LLM 自动生成 CoT + 自动验证 + 人工校验 |
| **代码推理** | GitHub 开源项目 | 生成指令-代码对 + 沙箱执行测试 |
| **逻辑推理** | 谜题集合 + ZebraLogic 合成数据 | 自动合成 + critic model + 人工审核 |
| **科学推理** | 物理/化学/生物试题 | LLM 判定难度 + 科学符号一致性检查 |
| **多模态 Agent** | 多角色合成引擎（User/Planner/Tool/Agent/Checker） | 工具协议集成（MCPs）、格式泛化 |

---

### 实验设置与评估指标

#### 模型配置
| 参数 | 数值 |
|------|-----|
| Architecture | MoE |
| Total Parameters | 80B |
| Activated Parameters | 13B |
| Shared Experts | 1 |
| Specialized Experts | 64 |
| Activated per Token | 8 |
| Hidden Size | 4096 |
| FFN Hidden Size | 3072 |
| Layers | 32 |
| Attention Heads | 32 |
| KV Heads | 8 |
| Context Length | 256K |
| Activation Function | SwiGLU |
| Tokenizer | 128K vocab（同 Hunyuan-Large） |
| Attention | GQA |

#### 推理优化技术
- 支持 **Auto Prefix Caching**, **Chunk Prefill**
- 支持量化格式：Weight Only INT8, W8A8, KV Cache FP8
- 兼容 TP, EP, FusedMoE 并行策略

#### 评估基准（Benchmark）

##### 主要公开基准
| 类别 | Benchmark |
|------|----------|
| **General Tasks** | MMLU, MMLU-Pro, MMLU-Redux, BBH, SuperGPQA |
| **Math & STEM** | MATH, CMATH, GSM8K, GPQA, GPQA-Diamond, OlympiadBench |
| **Coding** | EvalPlus, MultiPL-E, MBPP, LiveCodeBench, FullstackBench, McEval, ArtifactsBench（新） |
| **Reasoning** | BBH, DROP (F1), ZebraLogic |
| **Instruction Following** | IFEval, SysBench |
| **NLU** | ComplexNLU（内部）, Word-Task（内部） |
| **Agents** | BFCL v3, T-Bench, ComplexFuncBench, C3-Bench |
| **Long Context** | PenguinScrolls, LongBench-v2, FRAMES, RULER |

##### 评估模式
- **Slow-thinking mode**：启用 `/think`，允许详细 CoT 输出
- **Fast-thinking mode**：启用 `/no_think`，直接输出答案

---

### 基线方法对比
| 对比模型 | 类型 | 激活参数 | 总参数 | 是否开源 |
|--------|------|---------|-------|--------|
| Hunyuan-Large | MoE | 52B | 389B | 是 |
| Qwen2.5-72B | Dense | 72B | 72B | 是 |
| Qwen3-A22B | MoE | 22B | 235B | 是 |
| DeepSeek-R1 | MoE | ~22B | ~235B | 是 |
| OpenAI-o1 | Closed | - | - | 否 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Tables 2–6）

#### ✅ Pre-trained Model 表现（Table 2）
| Benchmark | Hunyuan-A13B | 最佳基线 | 表现 |
|----------|--------------|-----------|------|
| MMLU | **88.17** | 88.40 (Hunyuan-Large) | 接近最优 |
| MMLU-Pro | **67.23** | 68.18 (Qwen3-A22B) | 第二 |
| BBH | **87.56** | 88.87 (Qwen3-A22B) | 第二 |
| MATH | **72.35** | 71.84 (Qwen3-A22B) | **SOTA** |
| GPQA | **49.12** | 47.47 (Qwen3-A22B) | **SOTA** |
| EvalPlus | **78.64** | 77.60 (Qwen3-A22B) | **SOTA** |

> 💡 尽管总参数和激活参数均小于其他模型，Hunyuan-A13B 在多数任务上达到甚至超越更大模型的表现。

---

#### ✅ Post-trained Model 表现（Table 3 & 4）

##### Slow-thinking Mode（复杂推理）
| Benchmark | Hunyuan-A13B | 最佳 | 表现 |
|----------|---------------|------|------|
| AIME2024 | **87.3** | — | **第一** |
| AIME2025 | 76.8 | 79.8 (Deepseek-R1) | 第二 |
| MATH | 94.3 | 96.4 (OpenAI-o1) | 接近闭源 SOTA |
| BBH | **89.1** | 88.9 (Qwen3-A22B) | **第一** |
| ZebraLogic | **84.7** | 81.0 (OpenAI-o1) | **大幅领先** |
| BFCLv3 (Agent) | **78.3** | 70.8 (Qwen3-A22B) | **显著领先** |

> 🔥 在逻辑推理（ZebraLogic）和 agent 工具调用（BFCLv3）方面表现尤为突出。

##### Fast-thinking Mode（快速响应）
| Benchmark | Hunyuan-A13B | 表现亮点 |
|----------|---------------|---------|
| BFCLv3 | 65.9 | 仍优于部分大模型 |
| ComplexFuncBench | **74.0** | 远超 Qwen3-A22B (38.1) |
| C3-Bench | **65.4** | 显著领先 |

> ⚡ 即使在 fast 模式下，agent 能力依然强劲，说明其具备良好的泛化性和鲁棒性。

---

#### ✅ Long-context Evaluation（Tables 5 & 6）

| Benchmark | Hunyuan-A13B | 表现 |
|----------|---------------|------|
| PenguinScrolls | 87.7 | 接近 Gemini 2.5 Pro (88.3) |
| LongBench-v2 | **55.0** | 仅次于 Gemini 2.5 Pro (60.9)，优于 DeepSeek-R1/Qwen3 |
| FRAMES | 81.1 | 超越 Gemini 2.5 Pro (80.1)，略逊于 DeepSeek-R1/Qwen3 |
| RULER (Avg.) | **76.7** | 第二，仅次于 Gemini 2.5 Pro (81.7) |
| RULER (64K–128K) | **73.9** | 显著优于 DeepSeek-R1 (65.6) 和 Qwen3-A22B (66.6) |

> 📏 在极长上下文（>64K）下性能衰减更缓慢，体现优秀的 long-range consistency。

---

#### ✅ Inference Efficiency（Table 7）
在 A16W16C16 精度下，使用批量推理测试吞吐：

| Batch Size | Input Len | Output Len | Throughput (tokens/s) |
|------------|-----------|------------|------------------------|
| 1 | 2048 | 14336 | 190.84 |
| 16 | 2048 | 14336 | 1246.54 |
| 32 | 2048 | 14336 | **1981.99** |
| 32 | 2048 | 22528 | 1725.95 |

> ⚙️ 支持主流推理框架，实测高吞吐，适合低延迟服务部署。

---

### 消融实验（隐含分析）
虽然未单独列出消融表，但从文中可推断以下关键因素影响：
- **STEM 数据增强** → 显著提升 MATH/GPQA 成绩
- **Dual-CoT 设计** → 实现 fast/slow 模式的灵活切换，兼顾效率与精度
- **GRPO + 多 reward signal** → 提升 RL 训练稳定性与泛化性
- **NTK-aware RoPE** → 成功扩展 context 到 256K 且保持性能稳定

---

## 4. 关键结论和发现

### 主要发现
1. **小激活参数 ≠ 弱性能**  
   Hunyuan-A13B 以仅 **13B 激活参数** 实现媲美甚至超越数十倍参数模型的能力，证明了 MoE 架构在性价比上的巨大潜力。

2. **STEM 数据质量决定推理上限**  
   高质量 STEM 数据预训练是提升数学与科学推理能力的关键瓶颈，针对性优化带来显著增益。

3. **Dual-Mode CoT 提供实用灵活性**  
   “快思”与“慢想”双模式满足不同场景需求，尤其适合生产环境中对延迟敏感的应用。

4. **Agent 能力全面领先**  
   在 BFCLv3、ComplexFuncBench 等复杂工具调用任务中表现最佳，表明其在真实世界任务中的强适应性。

5. **长上下文稳定性优秀**  
   在 RULER 测试中，随着 context 增加至 128K，性能下降最平缓之一，适合处理书籍、财报、法律文书等长文本。

---

### 方法的局限性
1. **闭源模型仍有差距**  
   在某些任务（如 MATH）上仍略逊于 OpenAI-o1，说明顶级闭源系统在推理链完整性和知识密度上仍有优势。

2. **RAG 场景有待加强**  
   在 FRAMES 上得分低于 DeepSeek-R1 和 Qwen3-A22B，反映其在检索增强生成方面的整合能力尚有改进空间。

3. **Fast-thinking 模式性能受限**  
   在 fast 模式下，部分复杂任务（如 OlympiadBench）表现明显下降，依赖用户正确选择模式。

4. **依赖高质量标注与沙箱系统**  
   后训练依赖大量人工干预与安全沙箱环境，构建成本较高，难以完全自动化复制。

---

### 未来工作方向
1. **进一步压缩激活参数比例**  
   探索更稀疏的 MoE 结构，在保持性能前提下进一步降低推理成本。

2. **自动模式切换机制**  
   当前需手动指定 `/think` 或 `/no_think`，未来可研究基于输入复杂度的自适应推理路径选择。

3. **增强 RAG 与外部工具协同能力**  
   提升在多跳问答、跨源信息融合等任务中的表现。

4. **构建全自动数据合成流水线**  
   减少对人工标注的依赖，实现高质量 SFT/RL 数据的持续生成。

5. **探索多模态扩展可能性**  
   当前为纯文本模型，未来有望结合图像、音频等模态打造统一智能体。

---

> ✅ **总体评价**：Hunyuan-A13B 是一个兼具高性能、高效率与强实用性的开源 MoE 模型，代表了当前 LLM 发展中“**以巧取胜**”的重要方向，特别适合部署于资源受限但要求高推理质量的场景。

</details>

---

### 9. [Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks for Climate-Aware Digital Twins](https://arxiv.org/abs/2609.27290)

**Authors**: Tannaz Goodarzvand Chegini, Elyas Shivanian, Behzad Karimi, Faraz Dadgostari  
**Category**: cs.AI  
**Published**: 2026-09-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.27290v1  

#### Abstract
Short-horizon forecasts of atmospheric temperature are needed to support climate-aware digital-twin systems, but such forecasts must be produced where thermal observations are incomplete. This study evaluates a physics-informed neural network for potential-temperature forecasting, constrained by a p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks for Climate-Aware Digital Twins

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该研究针对**气候感知数字孪生系统**（climate-aware digital twins）中的一个关键挑战：在**大气温度观测稀疏**（sparse observations）的情况下，如何实现高精度的短时地表热场预测。  
传统数据驱动模型（如神经网络）在观测不足时表现受限，而纯物理模型又难以灵活适应局部变化。本文旨在探索一种结合两者优势的方法。

### ✅ 提出的新方法
提出了一种基于**Physics-Informed Neural Networks**（PINN）的大气潜在温度（potential temperature）短期预测框架，其核心创新包括：

- **物理约束建模**：将压力坐标系下的热力学平流-源方程（thermodynamic advection-source equation）作为PINN的PDE残差项，强制网络输出符合基本大气动力学规律。
- **经验性二热源闭合**（Empirical diabatic-source closure）：利用前12小时ERA5数据拟合一个低维参数化的$ Q_\theta $项，捕捉潜热、辐射等未解析过程，并在后续预测阶段冻结这些参数，避免对未来目标“作弊”。
- **硬初始条件嵌入**：通过构造函数形式确保神经网络在初始时刻精确匹配观测重建场，而非依赖软损失项逼近。
- **分阶段训练策略**（Staged training）：先学习历史源项，再逐步扩展未来时间窗口进行物理一致性优化，提升训练稳定性。

### ✅ 相比现有方法的优势
- 在观测极度稀疏（低至5%站点密度）下仍能保持显著性能优势；
- 随着预测步长增加（1→3小时），相对基线的改进持续增强，表明物理约束对长期演化更具价值；
- 方法具有跨区域可迁移性，在阿拉巴马州热浪事件中验证成功；
- 明确区分了“使用未来气象强迫”与“物理方程约束”的作用，证明性能增益来自物理机制本身，而非额外输入信息。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用 **ERA5 reanalysis** 数据，涵盖以下变量：
  - 温度 $ T $
  - 风速（纬向 $ u $、经向 $ v $）
  - 垂直速度（pressure vertical velocity $ w $）
  - 比湿 $ q $
  - 地面气压 $ p_s $
- 时间分辨率：每小时
- 垂直层次：固定选取 **700, 850, 925 hPa** 三个压力层
- 空间处理：转换为局地笛卡尔坐标（x, y），以千米为单位

### ⚙️ 实验设置
- **任务类型**：条件回溯预报（conditional hindcast），即允许使用未来的非温度气象强迫（u, v, w, q），但**不使用未来真实温度**；
- **预测目标**：潜在温度 $ \theta $，后转回空气温度用于评估；
- **预测步长**：+1h, +2h, +3h；
- **训练方式**：
  - 分两阶段训练：先用过去12小时数据联合学习神经网络和源项系数；
  - 冻结源项后，逐步推进未来时间域（curriculum learning）；
- **稀疏观测设计**：
  - 构造“虚拟观测站”网络，仅保留部分网格点的温度数据；
  - 使用 farthest-point sampling 实现空间均匀采样；
  - 测试不同密度：100%（密集）、25%、10%、5%

### 🎯 评估指标
- 主要指标：
  - **RMSE**（Kelvin）
  - **Relative improvement (%)**：相对于最强基线的RMSE下降百分比
- 辅助诊断：
  - MAE, Correlation
- 胜率统计：
  - PINN win rate：RMSE低于所有基线中最优者的比例
- 显著性判断标准（预设决策准则）：
  - 平均改进 ≥5%
  - 至少67%案例优于最强基线
  - 3小时预测中至少半数胜出

### 🔁 基线方法对比
| 基线名称 | 描述 |
|--------|------|
| **Persistence** | 初始状态不变，强于极短时预测 |
| **Recent-trend extrapolation** | 基于最近4小时趋势外推 |
| **Coordinate-only NN** | 同架构NN，仅从历史温度学习，无物理约束 |
| **Forcing-aware NN** | 接收相同未来气象强迫（u, v, w, q, Q_latent等），但无PDE残差项 |
| **Oracle envelope** | 取上述四个基线中RMSE最低者作为上界，用于保守比较 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ **Oklahoma 开发与复现实验（密集观测）**

| 实验阶段 | PINN RMSE (K) | 最强基线 RMSE (K) | 改进 (%) | 总体胜率 (%) | +3h 胜率 (%) |
|---------|----------------|--------------------|----------|---------------|----------------|
| 开发期 | 0.526 | 0.636 | **17.3%** | 70.8% | 62.5% |
| 复现期 | 0.641 | 0.781 | **18.0%** | 75.0% | 75.0% |

> ✔️ 表明方法具备时间上的可重复性和稳健性

#### ✅ **Oklahoma 观测密度扫描实验**

| 观测密度 | +1h 改进 (%) | +2h 改进 (%) | **+3h 改进 (%)** | 平均改进 (%) | +3h 胜率 (%) |
|--------|--------------|--------------|------------------|----------------|----------------|
| 100%   | 8.1          | 14.7         | **23.8**         | 18.0           | 75.0%          |
| 25%    | 2.6          | 5.5          | **16.9**         | 10.1           | 87.5%          |
| 10%    | 1.8          | 6.3          | **15.1**         | 9.3            | 87.5%          |
| 5%     | 1.4          | 9.1          | **14.6**         | 9.8            | 87.5%          |

> 🔍 发现：
> - 尽管短时预测增益较小，但在**+3小时预测中始终保持14.6–23.8%的显著提升**
> - 即使观测减少到**仅5%**，性能优势依然稳定存在
> - 无证据显示更低观测密度反而有利 → 排除“过拟合密集数据”假设

#### ✅ **Alabama 跨区域迁移实验（10%稀疏）**

| 预测步长 | PINN RMSE (K) | 最强基线 RMSE (K) | 改进 (%) | 起点胜率 |
|--------|----------------|--------------------|----------|-----------|
| +1h    | 0.373          | 0.391              | 4.6%     | 37.5%     |
| +2h    | 0.573          | 0.646              | 11.3%    | 75.0%     |
| **+3h** | **0.691**      | **0.860**          | **19.7%**| **75.0%** |

> ✅ 满足预设标准（≥5%改进且≥3个起点胜出）

#### ✅ **Alabama 不同观测布局鲁棒性测试（+3h）**

| 布局 | 改进 (%) | 更优起点数量（共4） |
|-----|------------|-----------------------|
| Layout 1 | 19.7%      | 3                     |
| Layout 2 | 23.6%      | 4                     |
| Layout 3 | 24.4%      | 4                     |

> ✔️ 结果不受具体传感器位置影响，说明方法具备空间布局鲁棒性

#### ❌ **Montana 地形压力测试（10%稀疏）**

| 预测步长 | PINN RMSE (K) | 最强基线 RMSE (K) | 改进 (%) |
|--------|----------------|--------------------|----------|
| +1h    | 3.201          | 2.954              | **-8.4%** |
| +2h    | 3.363          | 2.951              | **-14.0%** |
| **+3h** | **3.458**      | **2.942**          | **-17.5%** |

> ⚠️ PINN全面劣于基线，尤其在+3h差距扩大

> 原因分析：
> - 固定压力层（特别是925 hPa）在复杂地形区大量失效（图3a显示925 hPa完全无效）
> - 导致垂直结构表达失真，物理模型不再适用

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **物理约束的价值随预测时长远化而增强**：
   - 在+1h时改进有限（约8%），但在+3h可达**15–24%**；
   - 表明PINN通过物理机制更好地模拟了温度场的动态演化。

2. **方法对观测稀疏高度鲁棒**：
   - 即使只有**5%的观测站点**，+3h预测仍保持**14.6–16.9%** 的RMSE优势；
   - 适用于偏远地区或传感器部署受限场景。

3. **跨区域可迁移性强**：
   - 在阿拉巴马州热浪事件中无需调参即可复现正向收益；
   - 对不同观测布局也表现出良好鲁棒性。

4. **性能增益源于物理机制而非输入优势**：
   - 与 forcing-aware NN 的对比表明，即使两者都获得相同的未来气象强迫，PINN仍更优；
   - 证实了PDE约束提供了独立的信息增益。

### ⚠️ 局限性
1. **固定压力层表示在复杂地形下失效**：
   - 在蒙大拿州实验中，由于925 hPa层贴近地面且被地形切断，导致模型退化；
   - 暴露了当前公式在山地地区的适用边界。

2. **依赖高质量再分析强迫数据**：
   - 使用了ERA5提供的未来u, v, w, q等作为输入，属于“条件回溯”设定；
   - 若应用于实际业务预报，需耦合数值天气预报（NWP）输出，可能引入误差传播。

3. **源项采用经验闭合，泛化能力待验证**：
   - $ Q_\theta $ 是从历史数据中拟合并冻结的，未考虑季节或天气类型的变化；
   - 在极端天气或不同气候带中可能需要重新校准。

4. **未使用独立实测数据验证**：
   - 所有实验基于ERA5内部抽样，属于“重建ERA5”，尚未在真实探空或地面站数据上测试。

### 🔮 未来工作方向
1. **引入地形自适应垂直坐标**：
   - 如 terrain-following coordinates 或 hybrid sigma-pressure coordinates，提升在山区的表现；
   - 参考文献[21][22]已有成熟方案可供集成。

2. **拓展至更多地理区域和气候类型**：
   - 在沿海、高原、城市热岛等多种环境中测试迁移能力；
   - 增加样本量以支持更严格的统计检验。

3. **改进源项建模方式**：
   - 引入可学习的时空变化源项模块；
   - 或结合物理过程参数化方案（如辐射、边界层模型）替代纯经验拟合。

4. **向操作型系统演进**：
   - 替换ERA5未来强迫为实时NWP输出；
   - 集成到数字孪生平台（如DigiCARES项目）中进行端到端验证。

5. **不确定性量化与置信区间估计**：
   - 当前为确定性预测，未来可结合贝叶斯PINN或集成方法提供概率输出。

---

> 💡 **总体评价**：  
本论文系统地验证了PINN在稀疏观测下进行短时大气热场预测的有效性，明确了其优势边界——**在平坦或缓坡区域、中长期预测、低观测密度条件下表现卓越**，但也揭示了**固定垂直坐标的地形敏感性这一关键限制**。研究成果对构建面向气候感知的数字孪生系统具有重要指导意义。  

🔗 **代码开源地址**：[https://github.com/Tannaz-Chegini/pinn-atmospheric-heat-transfer](https://github.com/Tannaz-Chegini/pinn-atmospheric-heat-transfer)

</details>

---

### 10. [The KV Cache Working Set: Online Capacity Planning for LLM Inference Systems](https://arxiv.org/abs/2609.27746)

**Authors**: Luchang Li, Shuaishuai Wang, Zhao Ruan, Dongfang Li, Bozhao Gong  
**Category**: cs.DC  
**Published**: 2026-09-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.27746v1  

#### Abstract
Prefix caching is critical for efficient large language model (LLM) serving, particularly for agentic workloads that repeatedly invoke the model with a growing conversation and tool-use history. By reusing the key-value (KV) states of previously processed prefixes, prefix caching avoids redundant pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：The KV Cache Working Set: Online Capacity Planning for LLM Inference Systems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大型语言模型（LLM）推理服务中，**prefix caching** 是提升吞吐量、降低计算开销的关键技术。通过重用历史请求中的 key-value (KV) states，可以避免重复执行昂贵的 prefill 阶段计算。然而，KV cache 的容量规划面临两难：
- 容量太小 → 缓存命中率（hit rate）低，失去缓存优势；
- 容量太大 → 存储成本高昂，且边际收益递减。

因此，如何**在线准确估计达到目标 hit rate 所需的最小 KV cache 容量**（即“working set”），是高效系统设计的核心挑战。

传统方法如独立模拟多个 cache 容量配置（capacity-by-capacity simulation）计算开销大，难以用于实时系统。

---

### 🚀 提出的新方法：KVSET
作者提出 **KVSET** —— 一种高效的在线分析框架，用于估计 LLM 推理 workload 的 KV cache working set。

#### 核心思路：
- 基于经典的 **Mattson stack algorithm**，利用 LRU 替换策略下的栈距离（stack distance）来判断任意容量下某页是否命中。
- 引入 **Fenwick Tree** 高效维护访问顺序，快速计算每个 KV page 的 LRU stack distance。
- 在单次 trace 回放中，同时评估**所有候选 cache 容量**的命中情况，无需为每个容量单独运行模拟。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（如 kvcache-simulator） | KVSET |
|------|-------------------------------|--------|
| 模拟方式 | 对每个 cache 容量独立模拟 | 单次回放评估所有容量 |
| 时间复杂度 | $O(C \times N)$，$C$: 容量数，$N$: 请求量 | $O(N \log N)$ |
| 内存开销 | 需维护多个 cache 实例状态 | 仅需维护 Fenwick Tree 和访问记录 |
| 是否支持在线分析 | 否（离线为主） | ✅ 支持实时流式处理 |
| 准确性 | 高（但代价高） | 高，且接近真实部署测量值 |

> ✅ **核心优势**：KVSET 将原本需要数十次重复模拟的任务压缩为一次高效分析，显著降低了计算与内存开销，使**在线容量规划成为可能**。

此外，KVSET 还能根据目标 hit rate 反推出所需的最小 cache 容量，并支持分层存储（hierarchical storage）的容量建议。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用来自金山云内部生产环境的真实 **coding-agent workload trace**。
- 包含：
  - **10,000 个请求**用于 accuracy 评估；
  - **24,000 个连续请求**用于 online 分析能力验证。

该 workload 具有典型的 agentic 特征：多轮对话、工具调用历史不断增长，prefix reuse 显著。

---

### ⚙️ 实验设置
| 项目 | 设置 |
|------|------|
| 硬件平台 | NVIDIA H20 GPU servers |
| 模型 | GLM-5.2（W4A8 量化） |
| 推理引擎 | SGLang + Mooncake（支持多级 KV cache） |
| KV cache 大小 | ~60 KB per token（FP8 量化后） |
| 替换策略 | LRU（主流系统默认策略） |
| MTP | Disabled |

---

### 🎯 评估指标
1. **Hit Rate vs. Cache Capacity 曲线**：比较 KVSET 预测值与实际部署系统的测量值。
2. **Required KV Cache Capacity**：在不同 coverage target 下（如 95%, 99%, 99.9% 的请求保留其理论最大 hit rate）所需存储总量。
3. **Online 动态趋势跟踪**：随着请求流入，所需容量的变化趋势。

---

### 🆚 基线方法对比
- **kvcache-simulator [19]**：主流开源工具，采用 capacity-by-capacity 模拟。
- **直接部署实测**：在 Mooncake 上部署多种物理 cache 容量进行实测（资源密集型，仅选少数点测试）。

KVSET 不依赖这些方法，而是提供更轻量、等效甚至更实用的替代方案。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）预测准确性高（Figure 3）
- KVSET 的 hit rate 预测曲线与真实部署系统测量结果**高度吻合**。
- 在从几百 GiB 到数 TiB 的广泛容量范围内，误差极小。
- 表明：**无需实际部署即可精确建模容量-命中率关系**。

#### （2）容量收益呈现边际递减（Figure 3）
- 在低容量区，增加 cache 显著提升 hit rate；
- 超过一定阈值后，每单位存储带来的 hit rate 增益急剧下降。
- 说明：盲目扩容不经济，应基于明确目标优化。

#### （3）不同 coverage target 所需容量差异巨大（Figure 4）
| Coverage Target | Required KV Storage |
|------------------|--------------------|
| 99.9%            | 持续增长，未收敛     |
| 99%              | ~5 TiB             |
| 95%              | ~1 TiB             |

> 💡 仅放宽 4% 的覆盖要求（99% → 95%），可节省约 80% 的存储需求！

#### （4）online 分析可行性验证
- 在 24,000 请求序列上实现了实时容量估算；
- 曲线随活跃用户数变化动态波动（例如在 ~21k 请求处因并发减少而下降），体现现实 workload 特性；
- 支持动态调整 cache provision 策略。

---

### ❌ 无显式消融实验
论文未提供 formal ablation study，但通过以下方式间接验证有效性：
- 对比 Fenwick Tree 与显式栈实现的效率差异（隐含说明数据结构选择的重要性）；
- 展示不同 coverage 下容量变化趋势，反映算法对参数敏感性的合理响应。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **KV cache working set 可被高效在线估计**
   - 利用 Mattson stack + Fenwick Tree，可在单遍扫描中完成全容量范围的 hit rate 分析。
   - 实现了**零额外模拟开销**下的精确容量建模。

2. **最优 cache 容量取决于明确的 performance-cost 权衡**
   - 并非“越大越好”，而应设定合理的 **coverage target**（如 95% 或 99% 的请求达到理论 hit rate）。
   - 可据此指导分层存储设计（如 CPU memory + disk tier）。

3. **支持分层 KV cache 的容量规划**
   - 示例：若 CPU memory 提供 1 TiB → 覆盖 95% 请求；disk tier 提供额外 4 TiB → 总共覆盖 99%，形成 cost-effective 架构。

4. **open-source 工具已发布**
   - 开源地址：[https://github.com/llc-kc/kv_cache_capacity_estimator](https://github.com/llc-kc/kv_cache_capacity_estimator)
   - 支持 online request processing 与 offline trace replay，具备强实用性。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|-------|------|
| 仅适用于 LRU eviction | 当前主流系统多用 LRU，但若使用 LFU、TTL 或其他策略则不适用 |
| 仅建模 full-attention layers | 对 hybrid architectures（如 Mamba、Sliding Window Attention）需额外预留空间 |
| 假设 page-granular caching | 基于 paged KV cache 设计（如 vLLM），对连续内存管理模型适配性待验证 |

---

### 🔮 未来工作方向

1. **扩展至非 LRU 替换策略**  
   如 LFU、ARC、基于访问频率或语义重要性的定制策略。

2. **支持 hybrid attention 架构**  
   结合 Mamba、SWA 等新型架构的状态存储需求，统一建模 total cache footprint。

3. **集成到自动弹性调度系统**  
   将 KVSET 输出作为 feedback signal，驱动 runtime cache resizing 或 tier migration。

4. **跨用户 / session 的 working set 分析**  
   探索用户行为模式对共享 cache 效果的影响，优化 global cache sharing 策略。

---

## ✅ 总结

KVSET 提出了一种**高效、准确、可落地**的 KV cache 容量规划方法，解决了 LLM serving 中关键的资源平衡难题。它不仅在技术上突破了传统模拟方法的性能瓶颈，还在实践中提供了清晰的 **performance-cost trade-off 分析工具**，为构建经济高效的分层 KV cache 系统奠定了基础。其开源实现进一步推动了社区在 LLM inference optimization 方向的发展。

</details>

---

### 11. [Scaling of Capability and Efficiency at Inference Time in Large Reasoning Models](https://arxiv.org/abs/2609.27166)

**Authors**: Moritz Laber, Zohair Shafi, Germans Savcisens, Brennan Klein, Matteo Chinazzi, Samuel V. Scarpino, Albert-L\'aszl\'o Barab\'asi, Tina Eliassi-Rad  
**Category**: cs.LG  
**Published**: 2026-09-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.27166v1  

#### Abstract
Capability and efficiency are two key dimensions of reasoning in large language models (LLMs). Capability refers to the ability to solve a given problem correctly, whereas efficiency refers to the ability to do so with limited resources. When LLMs use Chain-of-Thought (CoT) reasoning to solve proble...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Scaling of Capability and Efficiency at Inference Time in Large Reasoning Models*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文系统研究了**Large Reasoning Models (LRMs)** 在推理时（inference time）的两个核心维度——**Capability（能力）** 和 **Efficiency（效率）** 如何随模型规模（parameter count）扩展的问题。

具体而言，作者关注：
- **Capability**：模型正确解决复杂问题的能力，如何随模型大小和问题难度变化。
- **Efficiency**：模型在解决问题时的资源消耗（以输出 token 数量衡量），是否也随着模型变大而变得更高效。

该问题的重要性在于：当前 AI 社区普遍依赖“**scaling hypothesis**”（即增大模型参数可提升性能），但这种策略是否在推理阶段同样有效且高效，尚缺乏系统性实证分析。

### 提出了什么新方法或新思路
- **提出将 Capability 和 Efficiency 视为“潜变量”（latent traits）**，并通过**分层贝叶斯建模（Hierarchical Bayesian Modeling）** 来从观测数据中推断其扩展规律。
- 设计了一套系统的评估框架，结合 **Chain-of-Thought (CoT)** 推理，量化模型在不同任务、不同实例大小下的表现。
- 引入了两个关键的统计模型：
  - **Variable Asymptote Model (VAM)** 与 **Fixed Asymptote Model (FAM)**：用于建模能力随实例大小的指数衰减，并分析其衰减尺度 $v_{N,T}$ 如何随模型大小 $N$ 扩展。
  - **Prefactor Exponent Model (PEM)** 与 **Prefactor Model (PM)**：用于建模输出长度（效率代理）随实例大小的幂律增长，并分析其参数是否随模型大小变化。

### 相比现有方法的优势
- 超越了传统的“准确率 vs 参数量”简单比较，深入到**问题难度（instance size）** 和 **推理成本（token length）** 的细粒度分析。
- 使用贝叶斯方法自然地处理不确定性，并通过 **leave-one-out expected log predictive density (elpd)** 进行模型选择，增强了结论的稳健性。
- 实验覆盖多个任务、温度和模型尺寸，提供了更具普适性的洞见。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
论文并未使用传统 benchmark，而是构建了**四个可控难度的合成推理任务（synthetic reasoning tasks）**，每个任务可通过调整 **instance size $n$** 控制难度：

| Task | 描述 | 示例 |
|------|------|------|
| **Addition** | 对 $n$ 个整数求和 | `Add the following numbers: 1,31,42,10.` |
| **Brackets** | 判断长度为 $n$ 的括号序列是否合法嵌套 | `Check whether the brackets are properly nested: {([]())}` |
| **Parity** | 判断长度为 $n$ 的二进制串中 1 的个数是否为偶数 | `Determine the parity of the following binary string: 0110` |
| **Index** | 返回长度为 $n$ 的列表中第 $i$ 个元素 | `What is the element at position 3 in the sequence 1,10,20,5?` |

这些任务均具有多项式时间算法，确保理论上可被高效解决。

### 实验设置和评估指标

#### 模型
使用 **DeepSeek-R1-Distill** 模型族，共五个尺寸：
- `1.5B`, `7B`, `14B`, `32B`, `70B` parameters

#### 设置
- 温度采样：$T \in \{0.4, 0.6, 0.8\}$，主分析使用 $T=0.6$
- 每个任务、每个 $n$ 生成 $R=100$ 个独立实例
- 使用 **Chain-of-Thought (CoT)** 提示，要求模型逐步推理并用 `\boxed{}` 包裹最终答案
- 使用 **SGLang** 库进行高效批处理

#### 评估指标
1. **Capability**：
   - 正确解出的实例数量：$y_{N,n,T} = |\mathcal{C}_{N,n,T}|$
   - 建模为：$y_{N,n,T} \sim \text{Binomial}(R, p_{N,T}(n))$，其中 $p_{N,T}(n)$ 为成功概率
   - 假设 $p_{N,T}(n)$ 随 $n$ **指数衰减**：  
     $$
     p_{N,T}(n) = p^\infty + (p^0 - p^\infty)\exp(-n / v_{N,T})
     $$
   - 关注 $v_{N,T}$（衰减尺度）如何随 $N$ 变化：$\log v_{N,T} = \log C^{(v)} + \beta^{(v)} \log(N/\bar{N})$

2. **Efficiency**：
   - 正确响应的平均输出长度：$l_{N,n,T}$
   - 建模为：$\log l_{N,n,T} \sim \mathcal{N}(\log A_{N,T} + \alpha_{N,T} \log(n/\bar{n}), w_{N,n,T})$
   - 关注 $A_{N,T}$（prefactor）和 $\alpha_{N,T}$（exponent）是否随 $N$ 变化

### 基线方法对比
本文未直接对比其他训练或推理方法，而是通过**统计模型比较**（如 FAM vs VAM, PM vs PEM）来判断哪种扩展规律更符合数据。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### Capability Scaling（能力扩展）
- 成功概率 $p_{N,T}(n)$ 随实例大小 $n$ **近似指数衰减**。
- 衰减尺度 $v_{N,T}$ 随模型大小 $N$ **次线性增长**（sublinear scaling）：
  - 幂律指数 $\beta^{(v)}$ 的后验均值在 **0.5–0.7** 之间（见 Fig. 3）
  - 例如，在 Addition 任务中，$E[\beta^{(v)}|D] \approx 0.69$，意味着**模型参数翻倍，能解决的问题规模仅增加约 1.61 倍**（$2^{0.69} \approx 1.61$）
  - 从 $1.5B$ 到 $70B$，$v_{N,T}$ 从 ~26 增至 ~363（见 Table A2）

#### Efficiency Scaling（效率扩展）
- 正确响应的输出长度 $l_{N,n,T}$ 随 $n$ **近似幂律增长**。
- 但幂律的 **prefactor $A_{N,T}$** 和 **exponent $\alpha_{N,T}$** 几乎不随模型大小 $N$ 变化：
  - $A_{N,T}$ 的扩展指数 $\beta^{(A)}$ 后验均值接近 0（见 Fig. 5, Table A3）
  - 例如，在 Index 任务中，$E[\beta^{(A)}|D] \approx -0.03$，95% 可信区间包含 0
- 表明：**更大的模型并未产生更短的推理链（CoT）**

### 与基线方法的对比结果
- **FAM vs VAM**：
  - Addition 和 Index 任务中 FAM 更优（更简单的共享渐近线假设即可拟合）
  - Brackets 和 Parity 任务中 VAM 更优（不同大小模型的极限性能不同）
- **PM vs PEM**：
  - 大多数情况下两模型差异不大
  - PEM（允许 exponent 也扩展）在 Addition 任务中略优，但在其他任务中 PM 更简洁

### 消融实验结果
- **温度敏感性测试**（$T=0.4, 0.6, 0.8$）：
  - 结论高度一致：能力次线性扩展，效率无显著改进
  - 表明结果对采样温度不敏感，增强了鲁棒性

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **能力（Capability）随模型规模提升，但收益递减**：
   - 更大的模型能解决更大、更难的问题，但提升是**次线性的**（diminishing returns）。
   - 支持“scaling works, but not infinitely”的观点。

2. ❌ **效率（Efficiency）未随模型规模提升**：
   - 更大的模型在生成正确答案时，**并未使用更少的 tokens**。
   - 输出长度的增长模式（幂律）几乎不受模型大小影响。
   - 表明“更大即更高效”的假设在当前训练范式下不成立。

3. 🧠 **对 AI 发展策略的启示**：
   - 单纯依赖“naive scaling”（盲目扩大参数）可能不是最优路径。
   - 当前 post-training（如 RLVR）虽提升了能力，但未优化 token 效率。
   - 应探索显式奖励“token efficiency”的训练目标。

### 方法的局限性
- **模型家族限制**：结论基于 **DeepSeek-R1-Distill** 系列，是否推广到其他架构（如 Llama, Mistral）未知。
- **任务范围有限**：仅涵盖四种算术/算法任务，复杂逻辑、数学证明等任务未涉及。
- **distillation 影响**：模型由大蒸馏而来，可能引入偏差。
- **静态评估**：未考虑动态推理控制（如 early stopping, adaptive CoT length）。

### 未来工作方向
- 研究 **Capability-Efficiency Trade-off**：是否存在权衡？能否设计 Pareto-optimal 模型？
- 分析扩展规律在整个 **model lifecycle**（预训练、SFT、RL）中的来源。
- 探索显式优化 **token efficiency** 的训练方法（如 RL with token cost penalty）。
- 将框架扩展到更多任务类型（如数学定理证明、程序合成）。

---

> **一句话总结**：  
> 本文发现，尽管更大的 LRM 能解决更难的问题（能力提升），但这种提升存在**收益递减**；更重要的是，它们**并未变得更高效**——生成的答案长度并未缩短。这挑战了“越大越好”的 scaling 范式，呼吁更精细的资源-性能权衡设计。

</details>

---

### 12. [Resource-Adaptive Stochastic Gradient Descent for Online Linear Programming without Re-solving](https://arxiv.org/abs/2609.28263)

**Authors**: Jiameng Lyu  
**Category**: cs.LG  
**Published**: 2026-09-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.28263v1  

#### Abstract
The growth of large language model (LLM) inference and search services increases the scale of online linear programming problems, motivating computationally efficient algorithms. We develop resource-adaptive stochastic gradient descent (RASGD) for stochastic online linear programming. The algorithm ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Resource-Adaptive Stochastic Gradient Descent for Online Linear Programming without Re-solving

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文研究的是**随机在线线性规划（stochastic Online Linear Programming, OLP）**中的资源分配问题。在大规模场景下（如大语言模型推理服务、搜索引擎等），传统基于重复求解线性规划（LP re-solving）的方法计算开销巨大，难以满足实时性要求。因此，如何设计一种**计算高效、无需反复求解LP**，同时又能保证高质量资源分配的算法，成为亟待解决的问题。

### 提出了什么新方法或新思路
作者提出了 **Resource-Adaptive Stochastic Gradient Descent (RASGD)** 框架，其核心思想是将“基于当前资源状态的定价逻辑”通过一个**一阶随机梯度下降（SGD）更新**来实现，而无需显式地重新优化。

- **核心机制**：
  - 每次请求到来时，仅使用当前请求和剩余库存信息，对资源价格进行一次SGD更新。
  - 采用**动态步长（adaptive stepsize）**：早期步长递减以支持学习，后期步长递增以匹配库存快速变化的动态。
  - 更新目标函数为基于当前剩余库存的对偶目标（current-resource dual objective），实现了**每到达一次的资源反馈（per-arrival resource feedback）**。

### 相比现有方法的优势
- **最优后悔界（Optimal Regret）**：在标准非退化条件（standard non-degeneracy）下，达到 $O(\log T)$ 的期望后悔（expected regret），与已知分布且无限算力的策略的理论下界匹配。
- **完全可行性（Exact Feasibility）**：在所有样本路径上都保持资源约束可行（prefix feasibility）。
- **极低计算复杂度**：每次到达仅需 $O(m)$ 时间和内存（$m$ 为资源数），不涉及任何LP求解或样本平均优化。
- **无需重启或主动集识别**：相比epoch-based方法，RASGD连续更新，无周期重启，更简洁高效。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验采用**合成数据集**，模拟两类典型OLP场景：

- **B1: 单资源问题（Single Resource）**  
  多秘书问题（multisecretary model）：单位消耗 $a_t=1$，奖励 $r_t \sim \text{Unif}[0,1]$，容量 $B_1 = pT$，$p$ 为覆盖率。

- **B2: 多资源问题（Multiple Resources）**  
  $m=10$ 或 $m=50$ 资源，奖励 $r_t \sim \text{Unif}[0,2]$，各维度消耗独立同分布于 $\text{Unif}[0,2]$。容量 $B_i = T d_i$，分两种配置：
  - **Homogeneous**：所有资源具有相同覆盖率 $p$。
  - **Mixed**：前5个资源 $p=0.50$（紧约束），后5个 $p=0.90$（松约束）。

### 实验设置和评估指标
- **时间范围**：$T = 1,000$ 至 $20,000$。
- **覆盖率扫描**：从 $0.10$ 到 $1.25$。
- **评估指标**：
  - **Hindsight Regret**：与已知全部请求序列后的分数最优解（fractional hindsight optimum）之间的期望收益差距。
  - **Online Runtime**：单次运行的在线处理时间（毫秒）。
- **重复次数**：30条独立请求流用于测试，10条用于超参数验证。

### 基线方法对比
共比较7种策略：

| 方法 | 类型 | 是否求解LP |
|------|------|-----------|
| **RASGD (本文)** | First-order, resource-adaptive | ❌ |
| Gao A3 | First-order, decoupled learning & decision | ❌ |
| Ma A5 | First-order, epoch-based resource target | ❌ |
| LSY A1 | First-order, fixed target | ❌ |
| DMD A1 | Dual Mirror Descent (entropic) | ❌ |
| Li-Ye A2 | Empirical LP, geometric checkpoints | ✅ |
| Li-Ye A3 | Empirical LP, per-arrival re-solving | ✅ |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）后悔值（Regret）
- 在 $T=5,000$ 下，RASGD 的后悔值接近最先进的 **Li-Ye A3**（每到达重解LP）：
  - **B1, 覆盖率0.90**：RASGD 为 **0.57**，优于 Li-Ye A3 的 0.80。
  - **B2, 覆盖率0.50**：RASGD 为 **38.40**，显著优于其他一阶方法（如 Gao A3: 53.32, Ma A5: 70.96）。
- 在中高覆盖率下，RASGD 表现尤为出色，几乎追平 Li-Ye A3。

#### （2）运行时间（Runtime）
- 所有一阶方法（包括RASGD）的运行时间均在 **1ms以下**。
- **Li-Ye A3** 因每步求解LP，耗时高达 **18,184 ms**（约18秒），比RASGD慢**5万倍以上**。
- RASGD 与 Gao A3、Ma A5 等一阶方法运行时间相当，具备同等计算效率。

| 方法 | B2 ($m=10$) 平均时间 (ms) |
|------|--------------------------|
| RASGD | 0.262 ± 0.039 |
| Ma A5 | 0.189 ± 0.033 |
| Li-Ye A3 | 18,184.344 ± 5,307.279 |

### 与基线方法的对比结果
- **vs. 一阶方法**：RASGD 在大多数设置下（尤其是资源紧张或混合覆盖）**显著优于所有其他一阶基线**（Gao A3, Ma A5, LSY A1, DMD A1）。
- **vs. LP重解方法**：RASGD 的后悔值**接近甚至在某些情况下优于** Li-Ye A2 和 A3，但计算成本**低几个数量级**。
- **综合表现**：RASGD 实现了“**接近LP重解的质量 + 一阶方法的速度**”，在性价比上占据绝对优势。

### 消融实验结果
进行了两组消融实验，验证两个核心组件的作用：

| 变体 | $T=2,000$ | $T=10,000$ |
|------|-----------|------------|
| Full RASGD | 20.51 | 34.10 |
| No resource feedback | 31.32 (+52%) | 73.63 (+116%) |
| No late-horizon increase | 27.83 (+36%) | 62.46 (+83%) |
| Neither | 32.49 (+58%) | 74.85 (+120%) |

- **移除任一组件都会导致后悔显著上升**，证明：
  - **资源反馈** 对长期稳定性至关重要。
  - **后期步长递增** 是应对库存敏感性的关键设计。
- 即使对变体重新调参，也无法弥补性能损失，说明这两个设计是本质性改进。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **RASGD 实现了理论与实践的双重突破**：
   - 理论上达到 $O(\log T)$ 最优后悔界，且无需知道分布或频繁重解LP。
   - 实践中在多种OLP场景下，后悔值接近最先进的LP重解方法，同时保持一阶方法的计算效率。
2. **资源自适应 + 步长匹配库存动态** 是成功的关键：
   - 将库存变化速率纳入步长设计，使得价格能及时响应资源稀缺性。
   - 避免了传统方法中因步长固定或过早衰减导致的“调整滞后”问题。
3. **无需复杂机制即可实现高质量控制**：
   - 不需要epoch划分、主动集识别、虚拟队列或探索-利用切换。

### 方法的局限性
- **依赖标准非退化假设**（standard non-degeneracy）：如严格互补性（strict complementarity）、正定协方差矩阵等。在退化或边界情形下性能可能下降。
- **需要提供保守曲率下界**（conservative curvature bound $\eta_0$）作为输入，虽不影响渐近阶，但影响常数项。
- 当前分析基于i.i.d.请求假设，对非平稳需求的鲁棒性有待验证。

### 未来工作方向
- 扩展至**随机资源补充**（stochastic replenishment）场景。
- 处理**时变需求**（time-varying demand）和非平稳环境。
- 应用于更复杂的AI服务调度问题，如**联合准入控制与任务调度**（joint admission and scheduling）。
- 探索在**组合OLP**或**非线性收益**下的推广。

---

> **总结**：RASGD 为大规模OLP问题提供了一种**高效、简洁、高性能**的新范式，特别适用于LLM推理、广告投放、云计算等对延迟敏感的应用场景，是连接理论最优性与工程实用性的优秀桥梁。

</details>

---

### 13. [ZOCheck: CPU-Shadow Checkpointing for Zeroth-Order LLM Fine-Tuning](https://arxiv.org/abs/2609.27189)

**Authors**: Minqiu Sun, Xin Huang, Luanzheng Guo, Nathan R. Tallent, Kento Sato, Dong Dai  
**Category**: cs.DC  
**Published**: 2026-09-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.27189v1  

#### Abstract
Zeroth-order (ZO) optimization is an attractive option for memory-efficient LLM fine-tuning, but its fault tolerance remains underexplored. Unlike first-order training, ZO progress can be represented by lightweight seed-and-scalar step logs, yet naive log-only recovery still incurs replay cost that ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ZOCHECK: CPU-Shadow Checkpointing for Zeroth-Order LLM Fine-Tuning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模 **LLM fine-tuning** 中，**Zeroth-Order (ZO) optimization** 因其内存效率高（接近推理级别）而受到关注。然而，现有的 **fault tolerance** 机制（如 checkpoint/restart）并未充分利用 ZO 训练的独特语义特性。

传统 **first-order (FO)** 训练依赖反向传播，checkpoint 需保存完整的模型参数和 optimizer state，开销大。而 ZO 训练每一步仅由一个 **PRNG seed** 和一个 **scalar projected-gradient** 决定，理论上可完全通过“回放”恢复训练状态。但直接采用“log-only replay”会导致恢复时间随训练步数线性增长，且由于浮点运算顺序的影响，**shortcut replay**（跳过扰动序列）会偏离原始数值轨迹，导致恢复不精确。

因此，核心问题是：  
> 如何设计一种既能利用 ZO 的轻量级 step log 特性，又能实现**快速、精确、低开销**的容错恢复机制？

---

### 提出的新方法：ZOCHECK
作者提出 **ZOCHECK** —— 一种基于 **CPU shadow process** 的非阻塞 checkpointing 系统，其核心思想是将 checkpoint 的构建从 GPU 关键路径中移出。

#### 创新架构：CPU Shadow Process
- **GPU** 正常执行 ZO 训练，每完成一步后发布一个极小的元数据日志 `(seed, g_proj, lr, eps)`（仅几十字节）。
- **CPU shadow process** 持续消费该日志，在 CPU 主存中维护一个训练状态的副本。
- CPU shadow 严格按照原始 ZO 的 in-place 扰动序列（`+ez`, `-2ez`, `+ez`, descent）进行回放，确保**浮点轨迹完全一致**。
- 定期在 CPU 上对 shadow 状态进行 snapshot，并异步持久化到 SSD。
- 故障恢复时：
  - 若为软件故障（GPU 进程崩溃），直接从内存中的最新 snapshot 恢复。
  - 若为主机故障，则加载最新的 durable snapshot，并仅需回放其后的少量日志后缀。

#### 关键优势
| 维度 | 传统方法（Full-State Checkpointing） | ZOCHECK |
|------|----------------------------------------|---------|
| **Checkpoint 开销** | 高（需复制模型大小的状态） | 极低（仅传输日志，回放在 CPU） |
| **恢复速度** | 慢（需加载完整 checkpoint） | 快（从近实时状态恢复） |
| **恢复精度** | 高（直接加载） | **比特级精确**（重放完整浮点轨迹） |
| **GPU 路径影响** | 阻塞或占用带宽 | **完全非阻塞** |

---

### 相比现有方法的优势
- **vs. Log-only replay**：避免了恢复时间随训练步数线性增长的问题。
- **vs. Periodic full-state checkpointing**：消除了 GPU 上的大规模同步数据拷贝，显著降低训练开销。
- **vs. Asynchronous checkpointing**：虽然异步化减少了阻塞，但仍需在 GPU 上执行 D2H 传输；ZOCHECK 将整个 checkpoint 构建过程移至 CPU，进一步解耦。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验中使用的数据集主要用于 fine-tuning 任务，包括：
- **SST-2**（默认配置）
- **BoolQ**, **WSC**, **WiC**
- **SQuAD**, **MultiRC**, **DROP**

其中 SQuAD 和 DROP 是生成任务，其余为分类任务。

---

### 实验设置
| 项目 | 配置 |
|------|------|
| **硬件平台** | <ul><li>NVIDIA L40S (48GB GPU mem) + AMD EPYC 9554</li><li>NVIDIA A100 (40GB) + AMD EPYC 7763</li></ul> |
| **模型家族** | Qwen3 (0.6B, 1.7B, 4B, 8B), Llama3 (3B, 8B), OPT-6.7B |
| **训练配置** | FP32, batch size=16, seq length=512, full-parameter fine-tuning |
| **Optimizer** | 默认 ZO-SGD (`lr=1e-7`)，部分实验使用 ZO-Adam (`lr=1e-5`) |
| **失败注入** | 在第 16,000 步注入一次故障，运行 20,000 步 |

---

### 评估指标
1. **Recovery Latency**：故障后恢复所需时间。
2. **Checkpoint Overhead**：训练过程中 checkpoint 引入的时间开销。
3. **End-to-End Wasted Time**：考虑失败和恢复的整体浪费时间。
4. **Exactness**：恢复后的模型参数是否与未中断训练**比特级一致**（通过哈希验证）。
5. **Resource Overhead**：CPU 利用率、DRAM 占用等。

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **MeZO (Log-only)** | 仅记录每步日志，恢复时从初始模型重新回放所有步骤 |
| **Sync ckpt (K)** | 每 K 步同步保存完整 checkpoint，阻塞 GPU |
| **Async ckpt (K)** | 每 K 步异步保存 checkpoint（D2H 后台进行），但仍阻塞 GPU 直到 D2H 完成 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **Checkpoint 开销降低** | 最高达 **219.7×**（vs. 异步全状态 checkpointing） |
| **平均恢复延迟降低** | **1.55× 更快**（vs. 异步 checkpointing） |
| **恢复延迟 vs. Log-only** | 平均快 **586.38×** |
| **端到端浪费时间减少** | 最多降低 **21.3×**（跨不同 MTBF） |
| **恢复精度** | **比特级精确**（single & triple failure 下均无差异） |

---

### 与基线方法的对比结果
- **图 3a (恢复延迟)**：
  - ZOCHECK 在软件故障下几乎“瞬时恢复”（从 CPU 内存 snapshot 加载）。
  - 即使在主机故障下（需从磁盘加载），恢复也远快于所有基线。
  - 异步 checkpointing 虽然降低了训练开销，但恢复延迟与同步方式相同。

- **图 3b (训练开销)**：
  - ZOCHECK 的 checkpoint 开销近乎为零（仅累计 0.48 秒日志开销）。
  - 全状态 checkpointing 即使异步化，仍因 D2H 传输带来明显开销。

- **图 4 (端到端浪费时间)**：
  - 在 MTBF 为 3–15 小时时，ZOCHECK 将浪费时间减少 **14.0–21.3×**（vs. 最优异步基线）。
  - 即使在磁盘恢复场景下，仍减少 **6.5–10.0×**。

---

### 消融实验与敏感性分析
| 实验 | 发现 |
|------|------|
| **多故障测试（图 5）** | 三次注入故障后，恢复的训练 loss 曲线与无故障运行**完全重合**，证明鲁棒性。 |
| **Bitwise Exactness** | 哈希验证显示，单次和三次故障恢复后，参数张量**最大差值为 0**，loss 差也为 0。 |
| **Sparse-MeZO 支持** | ZOCHECK 可扩展至稀疏 ZO 方法，恢复曲线同样重合，验证通用性。 |
| **CPU/GPU pacing 分析（表 2）** | 在所有主测模型上，CPU replay time < GPU step time，shadow 能保持同步。 |
| **边界情况（表 3–5）** | <ul><li>**低精度（FP16/BF16）**：GPU 加速更多，导致 CPU 落后，恢复变慢。</li><li>**小 batch / 短序列**：GPU 步骤变短，可能进入 lag regime。</li><li>**ZO-Adam**：额外的 moment 更新增加 CPU replay 负担，恢复时间显著上升。</li></ul> |

---

## 4. 关键结论和发现

### 主要发现
1. **ZO 的 replayable semantics 可被系统性地用于 fault tolerance**：通过 seed-driven 回放，可以构建精确的恢复机制。
2. **CPU shadow 是解耦 checkpoint 与训练的关键**：将 replay 和 snapshot 构建移至 CPU，实现了**非阻塞 checkpointing**。
3. **ZOCHECK 显著优化了恢复延迟与训练开销的权衡**：在多数场景下，同时实现**近零训练开销**和**快速恢复**。
4. **恢复必须重放完整 in-place 序列**：任何 shortcut replay（如直接累加更新）都会因浮点舍入误差累积而导致轨迹偏移（图 1）。
5. **方法具有良好的通用性**：支持 ZO-SGD、ZO-Adam、Sparse-MeZO 等多种 ZO 变体。

---

### 方法的局限性
1. **依赖 CPU replay 速度**：当 GPU 训练极快（如低精度、小 batch）时，CPU 可能无法跟上，导致 shadow lag，恢复时间变长。
2. **增加 CPU 资源消耗**：需要大量 CPU 核心并行执行 replay（峰值利用率超 12,000%），且占用额外 DRAM（最高达 6× model size）。
3. **不适用于非 replayable 的 ZO 变体**：若更新无法仅由 `(seed, scalar, opt_state)` 决定（如某些自适应或混合 ZO/FO 方法），则需额外扩展。
4. **当前为单机设计**：尚未扩展到分布式 ZO 训练场景。

---

### 未来工作方向
1. **扩展至分布式 ZO training**：支持多 GPU/多节点下的 fault tolerance。
2. **支持更广泛的 ZO optimizers**：如 DeepZero、MUZO 等复杂自适应方法。
3. **优化 runtime policy**：联合优化 replay lag、snapshot 频率、存储层级。
4. **系统级优化**：
   - NUMA-aware placement
   - Thread pinning
   - 持久化路径上的 contention 优化
   - Crash consistency for asynchronous logging
5. **专用 replay kernel**：开发低精度（BF16/FP16）CPU replay 内核，提升 replay 吞吐，应对更快 GPU 场景。

--- 

> ✅ **总结一句话**：  
> **ZOCHECK 通过 CPU shadow replay 架构，首次实现了对 ZO fine-tuning 的高效、精确、非阻塞容错，显著优于传统 checkpointing 方法，为大规模 LLM 的稳定训练提供了新范式。**

</details>

---

### 14. [LayerCheck: Adaptive Layer-wise Checkpointing for Large Language Model Post-training](https://arxiv.org/abs/2609.27193)

**Authors**: Minqiu Sun, Xin Huang, Luanzheng Guo, Nathan R. Tallent, Kento Sato, Dong Dai  
**Category**: cs.DC  
**Published**: 2026-09-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.27193v1  

#### Abstract
With the rising computational and monetary costs of training large language models (LLMs), checkpointing---periodically storing model states for recovery---becomes essential for fault tolerance. Conventional checkpointing entails a severe trade-off between checkpoint frequency (I/O overhead) and com...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LayerCheck: Adaptive Layer-wise Checkpointing for Large Language Model Post-training**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
在大规模语言模型（LLM）的 post-training 阶段，**传统 checkpointing 方法面临严重的 I/O 开销、存储成本和训练中断问题**。固定频率的全模型快照导致：
- 写入大量未显著更新的参数；
- I/O 瓶颈限制训练吞吐；
- 故障恢复时需回滚大量计算。

尽管已有方法如 GEMINI（内存快照）、LowDiff（差分压缩）等尝试优化，但它们**未利用 LLM 训练中层间更新异步性的内在特性**，仍存在冗余写入或高重建开销。

---

### ✅ **提出了什么新方法或新思路**
本文提出 **LayerCheck**，一种基于**层级别动态变化感知的自适应检查点机制**，其核心思想是：

> **并非所有 Transformer 层在每一步都发生显著更新；只有当某一层的累积参数漂移超过阈值时才进行持久化。**

#### 主要设计原则：
- **Layer-wise Selectivity**：以模块为单位（如每个 Transformer block），跟踪各层的归一化参数更新幅度（drift）。
- **Adaptive Thresholding**：使用学习率感知的动态阈值 $ K(t) = K_0 \cdot s(\text{lr}(t)) $，避免因 LR 调度导致误判。
- **Bounded Staleness Guard**：设置最大陈旧度 $ S_{\text{max}} $，强制长期未更新的层定期保存，防止任意延迟。
- **Composite Recovery**：故障后从不同时间戳中组合最新可用的每层状态，形成混合时间戳检查点，无需重放。

---

### ✅ **相比现有方法的优势**
| 维度 | LayerCheck | 传统方法（如 DeepSpeed 默认） | 差分方法（如 LowDiff, Amber） |
|------|-----------|-------------------------------|------------------------------|
| **写入量** | 显著减少（仅写变化大的层） | 固定全量写入 | 写增量但需维护元数据 |
| **I/O 模式** | 分散、平滑，无突发高峰 | 周期性 I/O 爆发 | 可能仍需同步 |
| **恢复速度** | 快速组装，无需重算 | 快（但依赖完整快照） | 慢（需 delta replay） |
| **系统开销** | 极低（复用 optimizer 输出） | 高（序列化全部） | 中高（追踪 diff） |
| **理论保障** | 收敛性分析支持有界扰动下稳定 | 不适用 | 缺乏 |

> 🔑 **关键优势**：**首次将 LLM 层级更新非均匀性用于 checkpointing 优化，在不牺牲模型质量的前提下大幅降低 I/O 和端到端训练时间。**

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集与模型**
- **模型**：
  - `Llama-3.2-1B`（18 层）
  - `Qwen-2.5-3B`（38 层）
  - `Qwen-2.5-7B`（31 层）
- **任务类型**：Supervised Fine-Tuning (SFT)
- **数据集**：
  - **MedQA**：医学领域问答
  - **OpenThoughts**：推理能力微调数据集

---

### ⚙️ **实验设置**
- **硬件环境**：
  - 单节点 8× NVIDIA A100 40GB GPU
  - AMD EPYC CPU + 2TB RAM
  - 100TB Lustre 并行文件系统
- **框架集成**：
  - 基于 PyTorch + DeepSpeed ZeRO-3 实现
  - 参数与优化器状态分片存储
- **训练配置**：
  - 序列长度：2048
  - Micro-batch size：1
  - Gradient accumulation：2 steps
  - Optimizer：AdamW
  - 初始 LR：$10^{-4}$，带 warmup/decay
- **Checkpoint 频率**：每步一次（high-frequency regime），用于压力测试

---

### 🎯 **评估指标**
| 类别 | 指标 |
|------|------|
| **系统效率** | - 总训练时间（end-to-end time）<br>- Checkpoint 时间占比<br>- I/O 体积（total checkpoint size）<br>- 平均每次 checkpoint 大小 |
| **恢复性能** | - 恢复时间 $ t_{\text{recovery}} = t_{\text{load}} + t_{\text{retrain}} $<br>- 实际 rollback 步数 |
| **模型保真度** | - 重启后的 loss 轨迹偏移<br>- 最终下游 benchmark score（ARC-easy, HellaSwag, MMLU-Med 等） |
| **可扩展性** | - 在模拟多故障场景下的 Effective Training Time Ratio (ETTR) |

---

### 🔁 **基线方法对比**
| 方法 | 类型 | 特点 |
|------|------|------|
| **DeepSpeed (Default)** | 全量定期 checkpoint | 基准线，高 I/O 成本 |
| **CheckFreq** | 流水线 checkpoint | 尝试重叠 I/O 与计算 |
| **GEMINI** | 内存快照 + 异步落盘 | 减少阻塞，但仍写全量 |
| **LowDiff** | 差分 checkpoint | 只存梯度差异，节省空间但需 replay |
| **Amber** | 增量参数选择 | 位图标记变更参数，细粒度但元数据开销大 |

> 所有方法在同一平台、相同配置下运行，确保公平比较。

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据汇总**

| 指标 | LayerCheck 表现 | 对比提升 |
|------|------------------|----------|
| **总 checkpoint 大小减少** | 最高达 **22.6×**（平均 ~17.1×） | vs. LowDiff |
| **平均单次 checkpoint 大小减少** | 最高达 **6.6×**（平均 ~5.0×） | 显著缓解 I/O 突发 |
| **端到端训练时间缩短** | 最快 **1.31×**（vs. LowDiff）<br>最高达 **3.76×**（vs. GEMINI on Qwen-7B） | 更高效利用 GPU |
| **恢复时间减少** | **3.4× 更快**（vs. LowDiff on Qwen-7B） | 无需重放，加载更少数据 |
| **有效训练利用率 (ETTR)** | 在 MTBF=0.5h 下达 **92.3%** | 显著优于 GEMINI (84.5%) 和 LowDiff (88.8%) |

---

### 📈 **详细实验结果**

#### （1）**Checkpoint I/O 与存储效率（Table 1 & 2）**
- 在 `Qwen-2.5-7B` 上，传统方法每轮写入约 **180TB** 数据，而 LayerCheck 仅写入 **1.2TB** → **150倍减少**。
- 平均每次 checkpoint 大小从 106.6GB（Default）降至 **4.35GB**。
- 使用不同阈值 $ K $ 控制精度-效率权衡：
  - $ K=10^{-2} $：最少写入，最大陈旧度
  - $ K=10^{-4} $：频繁刷新，接近全量行为
  - $ K=10^{-3} $：推荐默认值，平衡良好

#### （2）**恢复保真度（Figure 5）**
- **Loss 轨迹几乎完全重合**：重启后 loss 偏移最大仅 **0.54%**（绝对偏差 < 0.0024）。
- 多次连续失败（cascade recovery）下仍保持稳定，无误差累积。
- 不同 $ K $ 设置下均能快速收敛，验证鲁棒性。

#### （3）**最终模型质量（Table 3）**
| 模型 | 任务 | 无故障得分 | LayerCheck 恢复得分 | 差异 |
|------|------|------------|---------------------|------|
| Qwen-2.5-7B | MedMCQA | 60.39 | 60.41 | +0.02 |
| Qwen-2.5-7B | MMLU-Med | 92.00 | 93.00 | +1.00 |
| Llama-3.2-1B | PubMedQA | 55.80 | 55.80 | 0.00 |

> ✅ **所有任务上恢复模型性能与无故障训练基本一致，部分甚至略有提升（可能源于正则化效应）。**

#### （4）**消融实验（Ablation Study）**
- **是否启用 $ S_{\text{max}} $ 安全机制？**
  - 否则某些层（如 normalization）可能永远不被保存，导致严重陈旧。
  - 加入后保证最坏情况可控，且实际触发频率极低。
- **是否使用 LR 自适应阈值？**
  - 固定阈值在 warmup/decay 阶段表现不稳定。
  - LR-aware 设计使决策更合理，减少误写/漏写。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **Transformer 层更新具有显著异步性**：部分层更新剧烈，部分长期静止 → 为 selective checkpointing 提供基础。
2. **Layer-wise selective persistence 是高效的**：通过只保存“有意义”的层，可**成数量级减少 I/O 和存储开销**。
3. **Mixed-timestamp recovery 是安全可行的**：即使各层来自不同迭代，只要满足 bounded staleness，**不会破坏收敛性和最终性能**。
4. **系统收益可叠加**：更低的 checkpoint 开销 + 更快的恢复 → 在高频故障场景下显著提升整体训练效率（ETTR ↑）。

---

### ⚠️ **局限性**
1. **依赖存储后端特性**：在本地高速 SSD 或 NVMe 上，传统方法也可能较快，优势缩小；但在共享并行文件系统（如 Lustre）中优势明显。
2. **未公开比较 reconstruction time**：由于部分 baseline（如 GEMINI）未开源恢复逻辑，无法统一评测 $ t_{\text{reconstruct}} $。
3. **当前实现聚焦 AdamW**：对 SGD 或其他 optimizer 的适配需重新校准阈值 $ K_0 $。
4. **Embedding 层特殊处理**：因其更新较小，默认使用更高阈值，需手动调整。

---

### 🔮 **未来工作方向**
1. **与流水线 checkpointing 结合**：将 LayerCheck 的轻量写入与 GEMINI 等的 overlap 能力融合，进一步隐藏 I/O 延迟。
2. **自动阈值调优**：引入在线学习机制动态调整 $ K(t) $ 和 $ S_{\text{max}} $，适应不同训练阶段。
3. **跨节点一致性优化**：在超大规模分布式训练中，协调全局 layer persistence 决策。
4. **扩展至 Pre-training 场景**：验证在更长周期、更大规模预训练中的有效性。
5. **结合量化/压缩技术**：在 selective persistence 基础上进一步压缩已选层的表示。

---

> 💡 **一句话总结**：  
> **LayerCheck 利用 LLM 层级更新的非均匀性，提出了一种高效、低扰动、理论可证的自适应检查点机制，在不影响模型质量的前提下实现了高达 22.6× 的 checkpoint 存储压缩和 1.31× 的训练加速，为大规模 LLM fault-tolerant training 提供了新的系统级解决方案。**

🔗 **代码开源地址**：[https://github.com/DIR-LAB/LayerCheck](https://github.com/DIR-LAB/LayerCheck)

</details>

---

### 15. [Repurposing Pre-trained LLMs as High Fidelity Continuous Text Autoencoders](https://arxiv.org/abs/2609.27248)

**Authors**: Arkanath Pathak, Unnat Jain, Alexander C. Berg  
**Category**: cs.LG  
**Published**: 2026-09-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.27248v1  

#### Abstract
Next-token prediction has enabled highly fluent autoregressive language models, but it represents global structure only indirectly through sequential factorization. In contrast, high-fidelity autoencoders have become a standard primitive in image generation, enabling generative models to operate ove...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Repurposing Pre-trained LLMs as High Fidelity Continuous Text Autoencoders*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前主流的 **autoregressive LLMs** 虽然在文本生成上表现出色，但其逐 token 预测的方式导致：
- 缺乏对全局结构的建模能力；
- 下游任务微调成本高且易引发 **catastrophic forgetting**；
- 文本缺乏类似图像领域的 **high-fidelity 连续潜在表示**，限制了非自回归（NAR）生成和扩散模型等应用。

此外，现有的文本 autoencoder（如 ICAE、COSMOS）存在重建质量差、参数量大或无法处理长序列等问题。

### 🚀 提出的新方法：LLMAE
提出 **LLMAE**（LLM AutoEncoder），一种将预训练的 decoder-only LLM 改造为高质量连续文本自编码器的方法，核心思想包括：

- **固定长度连续潜在瓶颈**：通过在 LLM 中间层提取激活值作为 latent 表示 $ z \in \mathbb{R}^{K \times D} $，实现从变长文本到固定维度连续空间的映射。
- **结构化注意力掩码（Structured Attention Masks）**：将单个 LLM 分割为 Encoder 和 Decoder 功能块，确保解码过程必须经过 latent bottleneck，防止信息“绕道”泄露。
- **轻量化适配策略**：
  - 使用 **LoRA** 进行参数高效微调（仅训练 10.7M 参数）；
  - 引入 **additive codec**（单层 Transformer）对 latent 进行精细化调整；
  - 采用 **KL 正则化** 使 latent 分布接近标准正态分布，提升平滑性和可生成性。

### 🔍 相比现有方法的优势
| 方面 | LLMAE | ICAE / COSMOS |
|------|-------|--------------|
| 模型大小 | 仅 278.8M 参数 | ICAE 使用 7.46B 参数 |
| 重建质量 | 近完美重建（BLEU > 0.99） | 明显低于 LLMAE |
| 序列长度支持 | 支持长达 1024 tokens | COSMOS 限于 512 tokens |
| 架构简洁性 | 单一 LLM 完成编解码 | 多模块拼接（如 BERT + Perceiver） |
| 下游适用性 | 可直接用于 latent diffusion | latent 不够平滑，难用于生成 |

> ✅ **核心创新**：首次成功利用单一小型 LLM（Gemma-270M）构建出高保真、连续、可用于复杂下游任务（如 diffusion）的文本 autoencoder。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练数据**（共 1.4M 样本）：
  - `Pile-uncopyrighted`：1M 样本
  - `C4-realnewslike`：400K 样本
- **测试数据**（held-out 测试集）：
  - `C4-News-Stratified`：新闻类文本，分中长（≤512）、长（≤1024）两档
  - `OpenWebText`：通用网页文本，用于跨分布评估
  - `CreationMMBench`：多模态长文本生成基准，含文学、功能写作等四类任务

> ⚠️ 特别设计：采用 **stratified sampling** 和 **curriculum learning**（按长度排序训练），以增强对不同长度文本的泛化能力。

### 🧪 实验设置与评估指标

#### 主要任务
1. **Autoencoding 重建质量**
2. **Latent 空间分布分析**
3. **下游应用验证**：基于 LLMAE latents 训练 **latent text diffusion model** 用于 detailed image captioning

#### 评估指标
| 指标 | 含义 |
|------|------|
| **BLEU-4** | 衡量 n-gram 匹配程度，越高越好 |
| **Perplexity (PPL)** | 使用 GPT-2-large 计算，越低越好 |
| **BERTScore F1** | 基于 BERT 的语义相似度，越高越好 |
| **VLM Judge Score (0–6)** | GPT-4o 对 caption 覆盖度、一致性、幻觉打分 |
| **CapArena-Auto** | 自动化偏好评分，基于 GPT-4o 的 Elo 排名代理 |
| **RefCLIPScore** | 图像-文本对齐度（CLIP 相似性） |
| **DkL** | latent 与标准正态分布的 KL 散度，衡量平滑性 |

#### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **ICAE [27]** | Soft prompt-based AE | 使用 Mistral-7B 微调，压缩比 4× |
| **COSMOS [21]** | BERT + Perceiver Resampler | 固定 512 latent tokens，无压缩 |
| **LLMAE (Ours)** | 中间层激活 + 注意力掩码 | 278.8M 参数，K=256，D=640 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 方法 | 输入长度 | BLEU↑ | PPL↓ | BERTScore↑ |
|------|----------|--------|-------|-------------|
| **Source Text** | — | — | 30.8 / 24.1 | — |
| **ICAE** | ≤512 | 0.831 | 36.8 | 0.981 |
| **COSMOS** | ≤512 | 0.783 | 66.9 | 0.979 |
| **LLMAE (Ours)** | ≤512 | **0.999** | **30.9** | **0.999** |
| **LLMAE (Ours)** | ≤1024 | **0.995** | **24.4** | **0.999** |

> ✅ **结论**：LLMAE 在所有指标上显著优于基线，尤其在长文本（1024 tokens）下仍保持近完美重建。

### 🔁 与基线方法对比结果
- **相比 ICAE**：
  - 尽管参数量小一个数量级（278M vs 7.46B），重建质量更高；
  - ICAE 在长文本上性能下降明显（BLEU 从 0.831 → 0.710），而 LLMAE 更鲁棒。
- **相比 COSMOS**：
  - COSMOS 的 PPL 高达 66.9，表明重建文本流畅性差；
  - LLMAE 的 latent 更紧凑（256×640 vs 512×768），更适合下游建模。

### 🔍 消融实验结果

#### （1）Latent 层深度影响（Table 2）
| Depth | BLEU↑ | PPL↓ | BERT↑ |
|-------|--------|-------|--------|
| 2     | 0.001  | 1.1   | 0.720  |
| 5     | 0.589  | 20.8  | 0.960  |
| 11    | **0.978** | **25.5** | **0.997** |
| 14    | 0.222  | 5.7   | 0.815  |

> ✅ 发现 “**entropy valley**” 现象：中间深层（第11层）是最佳 bottleneck 位置。

#### （2）Latent token 数量（Table 3）
| K | BLEU↑ | PPL↓ | BERT↑ |
|----|--------|-------|--------|
| 64 | 0.376 | 4e6 | 0.925 |
| 128 | 0.216 | 9.64 | 0.885 |
| **256** | **0.978** | **25.5** | **0.997** |
| 512 | 0.958 | 22.8 | 0.992 |
| 1024 | 0.916 | 32.9 | 0.982 |

> ✅ 最佳压缩比为 4×（1024 → 256），过宽反而降低性能。

#### （3）Additive Codec 消融（Table 1）
| 设置 | BLEU↑ | PPL↓ | BERT↑ |
|------|--------|-------|--------|
| w/o Codec | 0.966 | 26.9 | 0.993 |
| **with Codec** | **0.995** | **24.4** | **0.998** |

> ✅ Additive codec 提升显著，弥补了 LoRA 的表达能力限制。

#### （4）KL 正则化与 latent 平滑性（Table 4）
| Ablation | DkL↓ | BLEU↑ |
|---------|--------|--------|
| NTP Only | 38556 | 0.978 |
| +KL | 35.36 | 0.967 |
| +KL + Codec | **11.69** | **0.995** |

> ✅ KL 正则化极大提升了 latent 空间的平滑性和稳定性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **中间层激活可作为高质量 latent 表示**：
   - Transformer LLM 内部存在 “**entropy valley**”，中间深层（如第11层）最适合提取抽象语义。
2. **单一 LLM 可同时承担 Encoder 和 Decoder 角色**：
   - 无需额外架构（如 BERT + GPT），即可实现高保真重建。
3. **LLMAE 的 latent 具备强下游适应性**：
   - 成功应用于 **latent diffusion for image captioning**，仅训练 111M 参数的 score network 即可生成高质量描述。
4. **轻量高效且可扩展**：
   - 总训练参数仅 10.7M，完整系统仅 278.8M，远小于主流 VLMs。

### 🖼️ 下游任务表现（Table 5）
| 方法 | #Params | VLM Judge↑ | CapArena-Auto↑ | PPL↓ |
|------|--------|------------|------------------|-------|
| Gemini-1.5-Pro | — | 5.90 | 56.17 | 19.8 |
| LLaVA-1.5-7B | 7.1B | 2.89 | -94.00 | 10.9 |
| **LLMAE + Diffusion (Ours)** | **819M** | **3.44** | **-62.00** | **26.0** |
| w/ COSMOS | 943M | 2.24 | -88.33 | 48.5 |

> ✅ 使用 LLMAE latent 的 diffusion 模型显著优于使用 COSMOS 的版本，证明其 latent 更适合生成。

### ⚠️ 方法的局限性
1. **依赖特定 backbone**：目前仅在 Gemma-270M 上验证，需探索其他 LLM 家族。
2. **解码仍为 autoregressive**：虽编码高效，但解码长文本仍有顺序延迟。
3. **最大支持 1024 tokens**：未测试更长序列（如 2K+）的表现。
4. **符号理解弱**：在 OCR、计数、品牌识别等任务上易产生 **symbolic hallucination**（见 Appendix A）。
5. **latent 密度过高**：追求高保真导致 latent 接近“无损压缩”，可能不利于弱条件生成。

### 🔮 未来工作方向
- 扩展至更大规模 LLM（如 Gemma-9B 或 Llama 系列）；
- 结合并行解码技术（如 speculative decoding）加速推理；
- 探索 latent compression 与语义抽象之间的平衡；
- 引入视觉监督信号增强 symbolic grounding 能力；
- 将 LLMAE 推广至其他多模态任务（如 text-to-video、dialogue planning）。

---

## 总结

📌 **LLMAE 是一项开创性工作**，它证明了：
> 即使是小型 decoder-only LLM，也能通过巧妙的架构改造（attention masking + LoRA + codec），成为一个**高保真、连续、可微分的文本 autoencoder**，不仅重建质量卓越，还能有效支撑复杂的生成任务（如 latent diffusion）。

🎯 其意义在于为 **text diffusion、non-autoregressive generation、multimodal integration** 提供了一个**轻量、高效、高质量的潜在空间基础**，有望推动文本生成范式的结构性演进。

</details>

---

### 16. [Reinforcement Learning with Decomposed Subtasks](https://arxiv.org/abs/2609.27035)

**Authors**: Mattie Terzolo, Mikolaj Sacha, Ayan Sinha, Andrew Rabinovich  
**Category**: cs.AI  
**Published**: 2026-09-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.27035v1  

#### Abstract
Group Relative Policy Optimization (GRPO) and related policy-gradient methods for training language model agents collapse an entire multi-turn rollout into a single scalar trajectory reward before it enters the policy update. When the task composes distinct skills, especially under sparse and delaye...

---

### 17. [TinyUDE: Solver-Free Universal Differential Equations on Microcontrollers via Lie-Taylor Jet Matching](https://arxiv.org/abs/2609.26972)

**Authors**: Pranavanath Balamurali, Hrishi Kamireddy  
**Category**: cs.LG  
**Published**: 2026-09-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.26972v1  

#### Abstract
Training Universal Differential Equations (UDEs) traditionally relies on backpropagating through numerical ODE solvers, creating memory footprints far exceeding the capabilities of edge microcontrollers. We present Lie-Taylor jet matching, a solver-free training framework that fits a hybrid vector f...

---

### 18. [Probabilistic and Geometry Aware Neural Surrogate of Scrape Off Layer Plasma Simulations](https://arxiv.org/abs/2609.28116)

**Authors**: Gabriele Gianuzzo, Stefan Dasbach, Fleur Hendriks, Sven Wiesen, Vlado Menkovski  
**Category**: cs.LG  
**Published**: 2026-09-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.28116v1  

#### Abstract
Fast surrogates for tokamak boundary-plasma simulation are typically deterministic regressors mapping a global operating point to a flattened vector of cell values. Near the divertor detachment transition the steady state is not reliably single-valued. A point estimate must average over qualitativel...

---

### 19. [Escaping Python Dependency Hell: A Hybrid Replay-and-Repair Pipeline for Python Dependency Resolution](https://arxiv.org/abs/2609.26952)

**Authors**: Veronica Poweska, Ariana Oyanguren, Jessica Pourleyli, Sourena Khanzadeh, Manar Alalfi  
**Category**: cs.AI  
**Published**: 2026-09-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.26952v1  

#### Abstract
Dependency conflicts in Python ecosystems arise from incompatible version constraints, missing packages, and undocumented compatibility relationships, causing many real-world code snippets to fail at execution. This paper presents PLLM+, a hybrid dependency-repair pipeline evaluated on the HG2.9K be...

---

### 20. [Stable Geometry with Divergent Task Evidence for Efficient Long-Horizon Agent Compression](https://arxiv.org/abs/2609.27332)

**Authors**: Mingxuan Wang, Fei Luo, Bo Wang, Guorun Yao, Yinglong Guo, Chao Ning, Hongyue Chen, Yanbiao Ma, Jungong Han  
**Category**: cs.AI  
**Published**: 2026-09-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.27332v1  

#### Abstract
Long horizon agents accumulate growing interaction histories that increase context and inference costs. We find that geometric redundancy alone is an insufficient criterion for safe compression. Although agent histories exhibit strong low dimensional structure, similar global geometry can preserve v...

---

### 21. [Data-driven discrete-time deep recurrent neural network-based modeling for dissipative systems](https://arxiv.org/abs/2609.27186)

**Authors**: Tuan Luong, Hyungpil Moon  
**Category**: cs.LG  
**Published**: 2026-09-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.27186v1  

#### Abstract
Physical AI has gained increasing attention for its role in developing AI systems that better understand, predict, and control real-world dynamics. Achieving this requires AI models that not only achieve high prediction accuracy but also preserve fundamental physical properties of dynamical systems....

---

### 22. [NGN: Learning Neural Network Size as a Differentiable Count](https://arxiv.org/abs/2609.27291)

**Authors**: Lixing Li  
**Category**: cs.LG  
**Published**: 2026-09-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.27291v1  

#### Abstract
Neural network size is usually chosen before training, separating architecture selection from weight optimization. We introduce the Neurogenesis Network (NGN), a differentiable parameterization for learning how many ordered structural components a model should use. For each ordered component group, ...

---

### 23. [Limiting-Kernel Q($\lambda$): Bridging Short and Long Horizons](https://arxiv.org/abs/2609.27741)

**Authors**: Tolga Ok, Arman Sharifi Kolarijani, Peyman Mohajerin Esfahani, Mohamad Amin Sharifi Kolarijani  
**Category**: cs.LG  
**Published**: 2026-09-24  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.27741v1  

#### Abstract
In value-based reinforcement learning, improving the accuracy of policy evaluation has been shown to improve downstream policy optimization performance. The widely adopted family of approximations relying on $n$-step truncation yields computationally efficient value estimators but is inherently limi...

---

### 24. [XLOG: A CUDA-Native Engine for Neurosymbolic Integration](https://arxiv.org/abs/2609.27203)

**Authors**: Levi Dubrovin, Nikita Pospelov, Kirill Sabitov  
**Category**: cs.AI  
**Published**: 2026-09-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.27203v1  

#### Abstract
xlog is a CUDA-native logic programming engine integrating neural perception with deterministic Datalog, probabilistic inference, and epistemic world views through a typed frontend and provider-owned CUDA runtime. Its reasoning modes share device data planes, but their execution boundaries differ: o...

---

### 25. [Learn How to Act from Your Own Interactions: On-Policy Self-Distillation for GUI Agents](https://arxiv.org/abs/2609.27307)

**Authors**: Yan Zhang, Daiqing Wu, Huawen Shen, Liang Li, Gang Cao, Zhi Gong, Wei Dai, Xiaode Zhang, Can Ma, Yu Zhou  
**Category**: cs.AI  
**Published**: 2026-09-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.27307v1  

#### Abstract
Graphical User Interface (GUI) agents enable the fulfillment of complex user instructions through multi-turn interactions with software environments, requiring step-wise reasoning and long-horizon memory to guide actions and retain task-relevant information, respectively. Recent on-policy self-disti...

---

### 26. [Finite-Sample Probabilistic Safety Certification for AI-Based Grid-Edge Coordination](https://arxiv.org/abs/2609.28182)

**Authors**: Yihong Zhou, Hanbin Yang, Thomas Morstyn  
**Category**: cs.AI  
**Published**: 2026-09-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.28182v1  

#### Abstract
Coordinating large population of flexible grid-edge devices can alleviate the need for time-consuming and capital-intensive network upgrades, and AI-based control methods such as multi-agent reinforcement learning or imitation learning are promising in their real-time decision scalability. However, ...

---

### 27. [LEGO: Synergizing Expert GraphRAG and Expert Chain-of-Thought for Legal Reasoning](https://arxiv.org/abs/2609.27009)

**Authors**: Qingjing Chen, Junkai Zhang, Shaochun Wang, Jiahao Ding, Siyuan Zheng, Yukun Yan, Zhi Zheng, Antonino Rotolo, Yun Liu, Weixing Shen  
**Category**: cs.CL  
**Published**: 2026-09-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.27009v1  

#### Abstract
Large language models are increasingly applied to high-risk domains such as law, yet complex legal reasoning remains limited by two structural challenges. First, existing RAG and GraphRAG methods emphasize lexical or semantic similarity while overlooking normative relations among legal provisions. S...

---

### 28. [Six Layers Less: Encoder Pruning for Whisper with Label-Free Recovery](https://arxiv.org/abs/2609.27980)

**Authors**: Rasmus Aagaard, Nicki Skafte Detlefsen  
**Category**: cs.CL  
**Published**: 2026-09-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.27980v1  

#### Abstract
Pruning large pre-trained transformer-based ASR models such as OpenAI's Whisper has seen great adoption, as pruning the decoder led to significant end-to-end transcription speedups. For instance, the {\tt whisper-large-v3-turbo} variant reduced the decoder from 32 to 4 layers, while Distill-Whisper ...

---

### 29. [Evaluating Open-Weight LLMs for Turkish Domain Documents Under Retrieval and Hardware Constraints](https://arxiv.org/abs/2609.28007)

**Authors**: Imtiaz Ul Hassan, \"Oyk\"u Akbulut, Onur Kaya, Ardhendu Behera, Swagat Kumar, Peter Matthew, Yonghuai Liu  
**Category**: cs.CL  
**Published**: 2026-09-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.28007v1  

#### Abstract
Most Turkish-capable large language models (LLMs) are evaluated using general-purpose benchmarks rather than long, structurally complex domain documents. This paper evaluates five open-weight 7B-8B models for Turkish document question answering under a resource-constrained local deployment setting. ...

---

### 30. [CORE-STACK+: Meta-Learning for Deep Stacked Generalization](https://arxiv.org/abs/2609.26905)

**Authors**: Noor Islam S. Mohammad  
**Category**: cs.LG  
**Published**: 2026-09-24  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.26905v1  

#### Abstract
Stacking heterogeneous vision backbones (CNNs, ViTs, and hybrids) is the de facto recipe for accuracy, calibration, and robustness, yet two coupled pathologies limit its returns. Prediction-space multicollinearity ill-conditions the meta-learner's Gram matrix, inflating weight variance and producing...

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

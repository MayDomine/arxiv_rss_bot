# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-30 08:50:33 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [When LLMs Develop Languages: Symbolic Communication for Efficient Multi-Agent Reasoning](https://arxiv.org/abs/2606.29354)

**Authors**: Zhengqi Pei, Qingming Huang, Shuhui Wang  
**Category**: cs.AI  
**Published**: 2026-06-30  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.29354v1  

#### Abstract
Chain-of-Thought (CoT) improves large language models (LLMs) on difficult reasoning tasks, but it often incurs long natural-language rationales that are poorly aligned with efficient machine reasoning. We propose Communicative Language Symbolism Routing (CLSR), a test-time framework in which multipl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：When LLMs Develop Languages: Symbolic Communication for Efficient Multi-Agent Reasoning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **Chain-of-Thought (CoT)** 虽然提升了 LLM 在复杂任务上的推理能力，但其依赖冗长的自然语言推理链，导致生成 token 数量多、延迟高，且这些文本并非为高效机器推理而设计。这在资源受限或对延迟敏感的应用中成为瓶颈。

本文提出一个核心问题：  
> **能否让 LLM 自主发明一种紧凑的符号语言（symbolic language），以更高效地进行内部推理？**

### 提出的新方法：CLSR
作者提出了 **Communicative Language Symbolism Routing (CLSR)** —— 一种在测试时（test-time）运行的多智能体框架，其核心是让多个 LLM 智能体自主地：
- **发明**（invent）
- **演化**（evolve）
- **共享**（share）

一种名为 **Language Symbolism Framework (LSF)** 的紧凑符号化通信协议。

#### CLSR 的三大核心机制：
1. **LSF（语言符号主义框架）**  
   - 不是自然语言指令，也不是可执行代码，而是一种**可复用的符号协议**，包含：
     - 符号命名（lexicon）
     - 语法（grammar）
     - 使用规则（constraints）
   - LSF 是从训练样本中由 LLM 自主生成并演化的，无需人工设计。

2. **进化式自举（Evolutionary Bootstrapping）**  
   - 多个 LLM 智能体通过迭代的“提出 → 批评 → 变异 → 选择”循环，逐步优化 LSF 池。
   - 选择标准基于 **准确率** 和 **token 成本** 的帕累托最优。

3. **路由机制（Latent-free Router）**  
   - 一个由 LLM 自身担任的 **router**，根据查询难度动态决定：
     - 使用单个 LSF（Single）
     - 集成多个 LSF（Aggregate）
     - 组合多轮 LSF 协议（Compose）
   - 实现**按需分配计算资源**，平衡 accuracy 与 token 开销。

### 相比现有方法的优势
| 方法类别 | 代表 | 局限性 | CLSR 的优势 |
|--------|-----|-------|-----------|
| **Prompt Optimization** | CoD, SoT, CCoT | 仅优化表面指令，未改变中间表示 | 引入**可复用的符号系统**，提升信息密度 |
| **Program-based Methods** | PoT, PAL | 依赖外部执行器（如 Python 解释器） | **纯 LLM 内部执行**，无需工具依赖 |
| **Compression Methods** | 压缩 CoT | 仍基于自然语言，压缩有限 | 发明**离散符号语言**，实现结构性压缩 |
| **RL-based Methods** | RL + verifier | 训练成本高，不稳定 | **无训练开销**，全为测试时优化 |

> ✅ **核心优势**：CLSR 在不牺牲准确率的前提下，显著降低生成 token 数量（减少 3~6x），实现了更优的 **accuracy-token trade-off**。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖七类具有挑战性的推理任务：
- **知识密集型问答**：MMLU-Pro, GPQA
- **数学推理**：GSM8K, MATH-500, AIME21-24
- **科学问答**：ScienceQA
- **多跳问答**：HotpotQA

### 实验设置
- **骨干模型（Backbone LLMs）**：
  - Qwen3-8B / Qwen3-32B
  - LLaMA3-8B
  - DeepSeek-R1 (distilled)
- **测试时框架**：所有模型权重冻结，仅通过 CLSR 进行推理优化。
- **LSF 演化过程**：
  - 使用训练集进行多轮进化（通常 5 轮）
  - 每轮选择“高杠杆”（正确且 token 少）的推理轨迹用于生成下一代 LSF
- **在线推理流程**：
  1. 输入查询 → Router 规划协议
  2. 执行 LSF 推理（单轮或多轮）
  3. 聚合输出最终答案

### 评估指标
- **Accuracy (%)**：任务准确率
- **Generated Tokens per Problem**：每个问题生成的 token 数量（作为延迟代理指标）
- **Cache-aware Token-equivalent Cost**：考虑前缀缓存后的综合成本（见附录）

### 对比的基线方法
| 类别 | 方法 |
|------|------|
| **Token Reduction** | Raw CoT, CoD, CCoT, SoT |
| **Program Execution** | PoT, PAL |
| **Prompt Optimization** | Plan-to-Solve (P2S), PromptBreeder (PBrd) |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

| Backbone | Method | MMLU-Pro (Acc/Tkn) | GPQA (Acc/Tkn) | GSM8K (Acc/Tkn) | MATH-500 (Acc/Tkn) |
|---------|--------|------------------|-------------|--------------|----------------|
| Qwen3-8B | Raw CoT | 60.2 / 276 | 49.1 / 1085 | 90.9 / 243 | 87.2 / 878 |
| Qwen3-8B | **CLSR (Ours)** | **60.4 / 96** | **47.7 / 228** | **91.2 / 89** | **86.8 / 257** |

> 📌 **观察**：CLSR 在保持甚至略微提升准确率的同时，**token 使用量减少约 60–75%**。

#### 与 Program-based 方法对比（Table 2）
| Method | GSM8K (Acc/Tkn) | MATH-500 (Acc/Tkn) |
|--------|----------------|------------------|
| PoT | 92.1 / 113 | 88.1 / 375 |
| PAL | 92.5 / 148 | 88.6 / 422 |
| **CLSR (T=3)** | **94.8 / 214** | **89.7 / 417** |

> ✅ CLSR 在不依赖外部执行器的情况下，**达到甚至超越 PoT/PAL 的性能**，且总部署成本更低（无需维护工具链）。

### 消融实验结果（Ablation Studies）

#### （1）进化深度（Evolution Depth）
- 随着进化代数增加，LSF 池质量提升，准确率上升，token 下降。
- 效果在约 5 代后趋于饱和，说明**适度演化即可获得收益**。

#### （2）智能体数量（Agent Population）
- 更多智能体参与演化 → 更多样化的 LSF 提案 → 更强的选择压力 → 更鲁棒的 LSF。
- 图 3 显示：增加 agent 数量可提升下游任务准确率。

#### （3）多轮推理（Multi-round T）的影响（Table 3）
| Method | GSM8K (Acc/Tkn) | MATH-500 (Acc/Tkn) | AIME (Acc/Tkn) |
|--------|----------------|------------------|---------------|
| CLSR (T=1) | 92.1 / 83 | 86.2 / 134 | 38.4 / 1047 |
| CLSR (T=3) | 94.8 / 214 | 89.7 / 417 | 46.8 / 3314 |
| **CLSR (Adaptive T)** | **92.8 / 90** | **86.8 / 257** | **44.7 / 2361** |

> 🔍 **关键发现**：**自适应停止策略**（adaptive T）能在大多数简单问题上早停（T=1），仅在难题上调用多轮，实现**最优的 accuracy-token 权衡**。

#### （4）跨模型 LSF 迁移（Cross-model Transfer）
- 最佳效果出现在 **LSF 生成器与推理模型一致** 时。
- 若生成器弱于推理模型，性能下降明显，说明 LSF 存在“方言错配”（dialect mismatch）风险。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 可以自发发展出高效的符号语言**  
   在准确性与 token 成本的双重压力下，LLM 能够演化出类似“机器方言”的 LSF，显著提升推理效率。

2. ✅ **符号化推理优于自然语言推理**  
   LSF 将冗余叙述压缩为紧凑的操作符（如变量绑定、子目标标记、验证标签），提升每 token 的信息密度（information-per-token）。

3. ✅ **CLSR 实现了帕累托前沿的突破**  
   在多个任务和模型上，CLSR 显著改善了 accuracy-token 曲线，优于所有基线方法。

4. ✅ **多轮 LSF 协议可模拟程序执行**  
   在满足“解释器可实现性”（interpreter realizability）前提下，CLSR 的多轮组合协议可条件性地**泛化 program-execution pipelines**（如 PoT/PAL）。

5. ✅ **路由机制实现智能计算分配**  
   Router 能根据问题难度自动选择：
   - 单一 LSF（快速响应）
   - 多 LSF 投票（增强鲁棒性）
   - 多轮组合（复杂问题分解）

### 方法的局限性
- **离线演化成本高**：LSF 池的构建需要大量计算资源（数天 GPU 时间），适用于可复用场景。
- **依赖高质量训练样本**：需要带推理链的训练数据来引导 LSF 合成。
- **符号可读性差**：LSF 输出对人类不友好，不利于审计和解释。
- **跨领域迁移有限**：当前 LSF 主要在同领域内有效，跨域泛化能力有待加强。

### 未来工作方向
- 研究 LSF 在多模态、工具调用和多智能体协作中的迁移能力。
- 探索如何将 LSF 与人类可读解释相结合（例如：内部用 LSF 推理，输出用自然语言解释）。
- 设计更高效的 LSF 搜索与压缩算法，降低离线成本。
- 研究 LSF 在持续学习和终身学习场景下的演化动力学。

---

> **总结一句话**：  
> CLSR 展示了 LLM 不仅可以作为推理引擎，还可以作为**语言创造者与文化演化参与者**，通过社会性符号系统的涌现，实现更高效、可控、模块化的智能推理。

</details>

---

### 2. [Importance-Aware Resource Allocation for Collaborative Task-Oriented Semantic Communication](https://arxiv.org/abs/2606.29052)

**Authors**: Kaiyi Lei, Yuanzhe Peng, Letian Zhang, Jie Xu  
**Category**: cs.DC  
**Published**: 2026-06-30  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.29052v1  

#### Abstract
Task-oriented semantic communication must allocate scarce radio resources to semantic features under fast fading wireless conditions and strict end-to-end latency budgets. Existing solutions are either optimization-heavy, leading to prohibitive computational overhead during online operation, or rely...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Importance-Aware Resource Allocation for Collaborative Task-Oriented Semantic Communication**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对**协作式任务导向语义通信**（collaborative task-oriented semantic communication）中的资源分配难题，解决以下关键挑战：
- 在多设备系统中，如何在**动态无线信道**（fast-varying channels）和**严格端到端延迟约束**下，高效分配有限的无线资源（如 RBs 和比特预算）。
- 如何在带宽受限的情况下，决定**哪些语义嵌入维度**（semantic embedding dimensions）应被传输、以何种精度（quantization level）传输，从而最大化下游任务性能（如分类准确率）。

传统方法要么依赖复杂的在线优化导致高计算开销，要么假设信道缓慢变化且需重新训练模型，难以适应实时、分布式的边缘场景。

---

### **提出了什么新方法或新思路**
作者提出 **iCoTASC**（importance-aware Collaborative TAsk-oriented Semantic Communication），一种**混合离线-在线框架**，其核心创新包括：

1. **基于 Integrated Gradients 的重要性感知机制**  
   利用 **Integrated Gradients (IG)** 对每个嵌入维度进行归因分析，量化其对下游任务输出的贡献，生成**维度级重要性图谱**（dimension-level importance map）。该图谱作为运行时通信控制信号，指导资源优先分配给“更重要的”语义特征。

2. **数据驱动的量化效用建模**  
   引入一个**修正的 Weibull 累积分布函数**作为量化效用函数 $ u(q) $，刻画“量化位数增加 → 任务准确率提升”的**边际收益递减规律**，更真实反映实际性能曲线。

3. **离线-在线协同的轻量级调度架构**  
   - **离线阶段**：预计算每个发射机在不同比特预算下的最大可实现效用 $ U_k(x) $，并构建查找表（lookup table）。
   - **在线阶段**：仅通过查表和贪心策略完成资源块（RB）分配与嵌入选择，实现低延迟响应。

4. **联合优化问题分解**  
   将原 NP-hard 的联合优化问题（embedding selection + quantization + RB allocation）解耦为可高效求解的形式，支持实时调度。

---

### **相比现有方法的优势**
| 维度 | iCoTASC | 现有方法 |
|------|--------|---------|
| **计算效率** | 在线阶段复杂度为 $ O(B \log K) $，适合实时应用 | 多数需每时隙求解复杂优化或端到端重训练 |
| **适应性** | 支持快速时变信道下的自适应资源分配 | 假设信道稳定或需缓慢变化 |
| **无需重训练** | 不改变底层推理模型，完全解耦 | 部分方法依赖联合训练或微调 |
| **细粒度控制** | 实现维度级选择与量化控制 | 多为整体压缩或均匀分配 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **CIFAR-10**：RGB 图像（32×32），10 类，用于三发射机（K=3）设置，每编码器输出 256 维嵌入。
- **Fashion-MNIST**：灰度图像（28×28），10 类服饰类别，测试两种配置：
  - K=2 编码器，每编码器输出 256 维；
  - K=4 编码器，每编码器输出 128 维。

所有嵌入由接收端拼接后送入分类器进行预测。

---

### **实验设置和评估指标**

#### **无线传输模型**
- 采用 OFDMA，总资源块（RB）数量为 $ B $，每个 RB 容量为 256 bits。
- 每个时隙开始时，各发射机观测其瞬时信道增益 $ c_k \in [0.1, 0.3] $，决定每 RB 可承载比特数。
- 带宽约束：$ \sum_{k} b_k \leq B $，且 $ \sum_{m \in S_k} q_{k,m} \leq b_k c_k $。

#### **评估指标**
- **Accuracy**：下游任务（图像分类）的 Top-1 准确率，随 RB 预算变化绘制曲线。
- **Computing Time**：仅统计**在线调度阶段**的运行时间（不包括训练），单位为毫秒（ms）。

---

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Baseline 1** | 使用 CVX 工具箱直接求解原始优化问题（4），然后进行整数化处理。无离线预计算，完全在线优化。 |
| **Baseline 2** | 信道感知但无视重要性：每时隙选择信道增益最高的发射机，随机在其维度上分配 2-bit chunks。 |
| **Baseline 3** | 信道感知 + 广度优先：同 Baseline 2，但优先将比特分配给当前量化级别较低的维度（鼓励覆盖更多维度）。 |

> 注：Baseline 2 和 3 忽略语义重要性，而 iCoTASC 显式利用 IG 得到的重要性权重。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **准确率表现（Accuracy）**
- **Fashion-MNIST（K=2）**：
  - 当 $ RB=4 $：iCoTASC 达到 ~0.59，显著高于 Baseline 2/3（~0.18）
  - 当 $ RB=32 $：iCoTASC 达到 **0.853**，优于 Baseline 1（0.814）
- **CIFAR-10（K=3）**：
  - 当 $ RB=32 $：iCoTASC 达到 **0.870**，优于 Baseline 1（0.885）接近但略低，但在低 RB 区域明显领先。

> ✅ **结论**：在**低带宽预算区域**（RB < 16），iCoTASC 显著优于所有基线，尤其在资源极度紧张时优势最大。

#### **计算延迟（Online Latency）**
| 方法 | CIFAR-10 @ RB=32 耗时 |
|------|------------------|
| Baseline 1 (CVX) | 15.900 ± 0.410 ms |
| Baseline 2 | 1.954 ± 0.091 ms |
| Baseline 3 | 1.991 ± 0.040 ms |
| **iCoTASC (Ours)** | **1.554 ± 0.183 ms** |

> ⚡️ iCoTASC 的在线延迟远低于 Baseline 1（减少约 90%），甚至略低于 Baseline 2/3，说明其调度算法高度轻量化。

---

### **与基线方法的对比结果**
| 对比项 | 结果 |
|-------|------|
| vs. Baseline 1 | 在低 RB 下准确率更高，且运行速度快一个数量级以上；虽在高 RB 下略有差距，但综合性价比更优。 |
| vs. Baseline 2 & 3 | 准确率全面碾压，在 RB=4 时可达其 3 倍以上；证明**引入语义重要性**是关键增益来源。 |

> 📈 图 3 和图 4 显示，iCoTASC 在 **4–32 RB 范围内增长最快**，验证了“少量高重要性维度精细量化即可大幅提升性能”的设计思想。

---

### **消融实验结果（隐含分析）**
虽然未明确列出消融实验表格，但从设计逻辑和结果可推断出以下关键发现：
- 若移除 IG 重要性（退化为 Baseline 2/3），性能急剧下降 → 表明**重要性感知机制至关重要**。
- 若不使用 Weibull 效用建模而采用线性假设，则无法捕捉边际收益递减特性，导致资源浪费。
- 离线预计算有效剥离了昂贵操作，使在线阶段仅需查表和贪心更新，实现**实时性保障**。

此外，Table I 中比较了不同 $ K $（发射机数量）的影响：
- 在小 RB 预算下，K=2 比 K=4 表现更好（gap ≈ 0.05），表明系统倾向于**优先保证部分设备高质量传输**而非平均分配。
- 随着 RB 增加，差距缩小，说明从“广度覆盖”转向“深度精修”。

---

## **4. 关键结论和发现**

### **主要发现**
1. **语义维度具有显著异质性**：并非所有嵌入维度对任务同等重要，**高重要性维度的小幅量化提升能带来巨大性能增益**。
2. **重要性感知资源分配优于信道感知或均匀分配**：结合 IG 归因与效用建模，可在极低带宽下维持较高任务准确率。
3. **离线-在线分离架构可行且高效**：将复杂建模与优化前置，实现实时调度，适用于移动边缘环境。
4. **存在“有效量化阈值”**：超过一定量化位数后，准确率提升趋于饱和，符合 Weibull 模型预测。

---

### **方法的局限性**
1. **依赖预训练模型稳定性**：若任务模型频繁变更，需重新执行离线阶段（如 IG 分析、效用拟合）。
2. **IG 计算成本较高**：尽管在离线阶段完成，但对于超大模型仍可能成为瓶颈。
3. **假设信道在时隙内恒定**：未考虑快衰落或多径效应带来的符号间干扰。
4. **未探索跨设备重要性协同优化**：目前重要性独立计算，尚未建模设备间的语义互补性。

---

### **未来工作方向**
1. **动态重要性更新机制**：研究能否在运行时增量更新重要性图谱，以应对输入分布漂移（distribution shift）。
2. **联合重要性学习与模型训练**：将 IG 或其他 attribution 方法融入训练过程，端到端优化重要性可解释性。
3. **扩展至非正交多址接入**（NOMA）等更复杂物理层协议。
4. **应用于更多任务类型**：如目标检测、语音识别、联邦学习中的梯度传输等。
5. **硬件部署验证**：在真实无线平台上测试 iCoTASC 的端到端延迟与能耗表现。

---

> 🔚 **总结一句话**：  
> iCoTASC 成功将 **explainable AI** 与 **semantic communication** 相结合，提出了一种**高效、可扩展、实时响应**的多设备语义资源分配框架，在严苛带宽条件下实现了优于传统方法的任务性能，为分布式智能系统的轻量化通信提供了新范式。

</details>

---

### 3. [The Mirage of Optimizing Training Policies: Monotonic Inference Policies as the Real Objective for LLM Reinforcement Learning](https://arxiv.org/abs/2606.29526)

**Authors**: Jing Liang, Hongyao Tang, Yi Ma, Yancheng He, Weixun Wang, Xiaoyang Li, Ju Huang, Wenbo Su, Jinyi Liu, Yan Zheng, Jianye Hao, Bo Zheng  
**Category**: cs.LG  
**Published**: 2026-06-30  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.29526v1  

#### Abstract
Reinforcement learning (RL) has gained growing attention in large language model (LLM) post-training, yet RL training remains fragile and can suffer from instability or collapse. One vital cause is training-inference mismatch: LLM adopts separate inference and training engines for generation efficie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：The Mirage of Optimizing Training Policies: Monotonic Inference Policies as the Real Objective for LLM Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现代大语言模型（LLM）在强化学习（RL）后训练中广泛采用**分离的训练引擎（training engine）和推理引擎（inference engine）**，以兼顾生成效率与训练精度。然而，即使模型参数同步，由于实现层面的差异（如数值精度、解码策略等），**训练策略（training policy）** 和 **推理策略（inference policy）** 对相同轨迹会分配不同的概率，导致 **training-inference mismatch**。

这一现象引发了一个被忽视的关键问题：**目标错位（objective misalignment）** —— 即使一个更新在训练侧提升了训练策略的性能，也无法保证其同步到推理引擎后能提升实际部署所用的推理策略的表现。传统方法仅关注缓解 mismatch 本身，而忽略了其对优化目标的根本影响。

---

### 🚀 提出了什么新方法或新思路

论文提出两个核心创新：

#### （1）**Monotonic Inference Policy Improvement (MIPI)**  
一种新的策略优化原则：**优化目标应是推理策略的单调提升**，而非训练策略的改进。即，我们关心的是同步更新后，推理策略 $ \mu_{k+1} $ 是否比 $ \mu_k $ 更好，而不是 $ \pi_{k+1} $ 是否比 $ \pi_k $ 更好。

该目标通过以下分解形式化：
$$
J(\mu_{k+1}) - J(\mu_k) = \underbrace{J(\mu_{k+1}) - J(\pi_{k+1})}_{\text{① 后同步差距}} + \underbrace{J(\pi_{k+1}) - J(\pi_k)}_{\text{② 训练侧更新}} + \underbrace{J(\pi_k) - J(\mu_k)}_{\text{③ 前同步差距}}
$$

#### （2）**Monotonic Inference Policy Update (MIPU)**  
一个两步框架，用于实现 MIPI 原则：

- **Step 1: Sampler-Referenced Policy Update**  
  在训练中引入 **截断的 sampler-referenced correction（TIS 风格）**，使得候选更新 $ \pi_{k+1} $ 相对于推理策略 $ \mu_k $ 构建，从而优化项 ②+③，确保 $ J(\pi_{k+1}) \geq J(\mu_k) $。

- **Step 2: Inference-Gap-Aware Acceptance**  
  将候选模型同步至推理引擎得到 $ \mu_{k+1} $，并估计 **后同步推理差距（post-update inference gap）**：
  $$
  T_{\text{post}} = J(\mu_{k+1}) - J(\pi_{k+1})
  $$
  若 $ T_{\text{post}} \geq -c $（$ c $ 为容忍噪声的阈值），则接受更新；否则回滚。这一步确保项 ① 非负或可控。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（如 GRPO, MIS, LR-decay） | MIPU |
|------|-------------------------------|------|
| **目标视角** | 优化训练策略 $ \pi $ | 优化推理策略 $ \mu $（真正部署所用） |
| **处理 mismatch 方式** | 减少 mismatch（算法或系统级修正） | 承认 mismatch 存在，从目标层面应对 |
| **更新决策依据** | 训练侧收益 | 推理侧一致性验证 |
| **稳定性机制** | 过滤样本、降低 LR | 动态回滚不可靠更新 |
| **理论基础** | 未考虑目标错位 | 显式分解推理策略增益 |

> ✅ MIPU 不是对现有方法的替代，而是**正交增强**，可与其他 mismatch 缓解技术结合。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **训练数据**：
  - `DAPO-Math-17`：数学推理 RL 数据集
  - `DeepMath-103K`：大规模、去污染的数学问题数据集
  - 经过滤，保留基础模型非饱和成功率的问题，确保有意义的 reward 变化

- **评估数据集（5个数学推理基准）**：
  - `MATH-500`
  - `AIME24`
  - `AMC23`
  - `Minerva`
  - `OlympiadBench`

---

### ⚙️ 实验设置

- **模型规模**：
  - `Qwen3-4B`
  - `Qwen3-1.7B`

- **高 mismatch 设置**：
  - 使用 **FP8-quantized rollout** 进行推理采样，显著放大 training-inference mismatch

- **训练配置**：
  - 训练引擎：Megatron
  - 推理引擎：vLLM
  - Batch size: 64，梯度累积步数 32
  - 学习率：1e-6
  - 优势估计：Group Relative Advantage（GRPO 风格）

- **评估指标**：
  - 主要指标：**Pass@1 准确率**
  - 辅助诊断指标：
    - `Mismatch-K3-KL`：推理与训练策略之间的 KL 散度
    - `T_post`：后同步推理差距代理
    - 回滚率（rollback rate）
    - 训练曲线稳定性

---

### 🆚 基线方法对比

| 方法 | 类型 | 说明 |
|------|------|------|
| **Baseline (GRPO)** | 标准 RL 方法 | Vanilla GRPO + dual-clipped loss |
| **MIS** | 优化侧稳定化 | 过滤 mismatch 过大的 token |
| **LR-decay** | 优化侧稳定化 | 按固定间隔衰减学习率 |
| **TIS** | Step 1 变体 | 仅使用 sampler-referenced 更新 |
| **Step 2 only** | Step 2 变体 | 仅使用推理差距回滚机制 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Table 1）

| Model | Method | MATH | AIME | Olympiad | Minerva | AMC23 | **Avg.** | Stable? |
|-------|--------|------|------|----------|---------|-------|--------|--------|
| Qwen3-4B | Baseline | 89.34 | 42.00 | 64.89 | 43.39 | 82.50 | 64.42 | ❌ |
| Qwen3-4B | MIS | 90.95 | 38.44 | 62.50 | 44.12 | 81.09 | 63.42 | ❌ |
| Qwen3-4B | LR-decay | 90.34 | 44.00 | 67.26 | 43.75 | 82.97 | 65.66 | ❌ |
| Qwen3-4B | **Ours (MIPU)** | **91.15** | **43.56** | **67.86** | **45.96** | **85.00** | **66.71** | ✅ |
| Qwen3-1.7B | Baseline | 83.10 | 25.33 | 56.55 | 31.68 | 57.66 | 50.86 | ❌ |
| Qwen3-1.7B | MIS | 81.29 | 24.67 | 58.33 | 34.19 | 60.16 | 51.73 | ❌ |
| Qwen3-1.7B | LR-decay | 82.09 | 26.00 | 58.93 | 28.68 | 65.47 | 52.23 | ❌ |
| Qwen3-1.7B | **Ours (MIPU)** | **86.52** | 24.67 | **59.52** | **33.82** | **65.31** | **53.97** | ✅ |

> ✅ MIPU 在两个模型上均取得**最高平均准确率**，且训练过程**无崩溃或严重退化**。

---

### 🔬 消融实验结果（Table 2 & Figure 3）

| Method | Avg. Score | Key Observation |
|--------|------------|----------------|
| Baseline | 64.42 | 易受 mismatch 影响，后期性能下降 |
| + Step 1 (TIS) | 65.36 | 改善候选质量，但无法防止 mismatch 积累 |
| + Step 2 only | 62.81 | 能防崩溃，但难以持续提升（依赖差候选） |
| **Ours (Step 1 + Step 2)** | **66.71** | **互补作用明显：Step 1 提升方向，Step 2 控制风险** |

#### 关键发现：
- **Step 1**：提升更新方向正确性，减少训练初期 KL 差距
- **Step 2**：有效识别并拒绝“训练收益无法在推理端复现”的更新
- **两者结合**：实现更强性能与更稳训练轨迹

---

### 🧪 Step 2 有效性分析（Figure 4b）

- **随机回滚控制实验**：设置 67% 回滚率（高于 MIPU 的 53.5%）
  - 结果：仍发生崩溃
  - 结论：**回滚数量不是关键，关键是基于 $ T_{\text{post}} $ 的信号条件判断**

> ✅ Step 2 是**智能筛选机制**，而非简单稀疏化更新。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Training-inference mismatch 导致目标错位**：训练策略的提升 ≠ 推理策略的提升，这是 RL 训练不稳定的根本原因之一。
2. **MIPI 是更合理的优化目标**：应直接优化部署所用的推理策略性能。
3. **MIPU 有效实现 MIPI**：
   - Step 1 构建更优候选
   - Step 2 基于推理侧信号进行风险控制
4. **两步机制互补**：仅靠一方无法同时实现高性能与高稳定性。
5. **在高 mismatch 场景下优势显著**：FP8 量化 rollout 中，MIPU 显著优于现有稳定化方法。

---

### ⚠️ 方法的局限性

1. **计算开销增加**：
   - 需额外 validation rollout 来估计 $ T_{\text{post}} $
   - 回滚机制带来重复训练成本
2. **当前实验规模有限**：
   - 仅在中等规模模型（1.7B / 4B）上验证
   - 更大模型或不同任务（如 coding）中的表现待验证
3. **Step 2 是灵活设计空间**：
   - 当前使用基于 rollout 的 proxy，未来可探索更高效估计器
4. **容忍参数 $ c $ 需调优**：
   - 虽然采用动态 annealing，但仍需经验设定

---

### 🔮 未来工作方向

1. **扩展到更大模型和更多任务场景**（如 agent、code generation）
2. **改进 $ T_{\text{post}} $ 估计效率**：
   - 设计低方差、无需完整 rollout 的 proxy
3. **将 inference-gap 信号融入优化过程**：
   - 而不仅是作为 acceptance filter
4. **结合系统级优化**（如 FP16 rollout）与 MIPU，形成端到端鲁棒 RL 流程
5. **探索在线自适应容忍阈值 $ c $**

---

## 总结

> **MIPU 揭示了 LLM 强化学习中的一个“海市蜃楼”：优化训练策略看似合理，实则可能偏离真实目标。真正的目标应是推理策略的单调提升。**

通过提出 **MIPI 原则** 和 **MIPU 框架**，论文实现了从“训练视角”到“部署视角”的范式转变，在高 mismatch 场景下显著提升了 LLM RL 的**性能上限**与**训练稳定性**，为构建可靠的大模型 RL 系统提供了新思路。

</details>

---

### 4. [DAIN: Dynamic Agent-Based Interaction Network for Efficient and Collaborative Multimodal Reasoning](https://arxiv.org/abs/2606.30189)

**Authors**: Xinxin Chen, Yuchen Li, Zihan Wang, Haoyu Zhang, Ruixin Liu, Mingyuan Zhao  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.30189v1  

#### Abstract
Current multimodal fusion approaches, particularly those based on static Mixture-of-Experts (MoE) architectures, often struggle to provide the adaptive and efficient collaborative reasoning required by complex real-world applications. We introduce the Dynamic Agent-based Interaction Network (DAIN), ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DAIN: Dynamic Agent-Based Interaction Network for Efficient and Collaborative Multimodal Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的多模态融合方法（尤其是基于静态 **Mixture-of-Experts (MoE)** 架构）存在以下局限：
- **缺乏动态适应性**：专家之间的协作是预定义且固定的，无法根据输入样本上下文动态调整。
- **计算效率低**：通常采用密集交互（如 full cross-attention），导致高计算开销。
- **难以建模多样化交互模式**：未能显式区分并利用 **synergy**（协同）、**uniqueness**（独特性）和 **redundancy**（冗余）等不同类型的跨模态信息交互。

这些问题限制了模型在复杂真实场景中的高效、鲁棒推理能力。

---

### 🚀 提出的新方法：DAIN（Dynamic Agent-based Interaction Network）
作者提出了一种全新的范式——将多模态融合视为一个由多个**交互代理（interaction agents）** 动态协作完成的决策过程。其三大核心创新如下：

#### （1）**Context-Aware Meta-Controller 实现稀疏调度**
- 引入一个轻量级的 **Meta-Controller**，根据输入的多模态上下文向量 $ e_c $，生成稀疏门控向量 $ g \in \mathbb{R}^K $。
- 仅激活最相关的子集代理（top-$T$ agents），实现 **sample-wise sparse activation**，提升效率。

#### （2）**结构化压缩通信机制（Compressed Inter-Agent Communication）**
- Meta-Controller 同时预测一个软连接图 $ \tilde{G} \in \mathbb{R}^{K\times K} $，控制活跃代理间的通信强度。
- 消息通过低秩瓶颈（bottleneck）进行压缩传递，避免全连接通信带来的冗余。

#### （3）**多目标优化学习框架**
联合优化三个目标：
- **任务准确率**（`L_task`）
- **代理专业化程度**（`L_spec`，鼓励各 agent 对特定模态组合敏感）
- **运行效率正则项**（`R_eff`，对 gate 向量和通信图施加 L1 和 Frobenius 范数惩罚）

> 💡 整体设计受 **Partial Information Decomposition (PID)** 理论启发，明确划分 Synergy、Uniqueness、Redundancy 三类 agent。

---

### ⚖️ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **性能** | 显著优于所有静态融合与 MoE 变体，在多个基准上达到 SOTA |
| **效率** | 有效参数更少（平均仅激活 ~3–5 个 agent），前向传播成本更低 |
| **可解释性** | 提供 agent 激活路径与通信图，揭示“为何”做出该决策 |
| **灵活性** | Backbone-agnostic，适用于不同编码器架构 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集（涵盖医疗与通用领域）
| 数据集 | 模态 | 任务类型 | 应用背景 |
|--------|-------|-----------|----------|
| **ADNI** | MRI, PET, Genetics, Cognitive Assessments | 分类（阿尔茨海默病诊断） | 医疗健康 |
| **MIMIC-IV** | 临床笔记、生命体征、实验室结果 | 分类（死亡率预测） | 医疗健康 |
| **MM-IMDB** | 海报图像 + 剧情文本 | 多标签分类（电影类型识别） | 视觉-语言 |
| **CMU-MOSI** | 视频、音频、文本 | 回归（情感强度预测，MSE） | 情感分析 |
| **ENRICO** | 移动UI截图 + 元数据 | 分类（设计主题识别） | 人机交互 |

---

### 🔬 实验设置
- **编码器**：ResNet-50（图像）、BERT-base（文本）、Transformer（序列数据）
- **嵌入维度**：统一为 512
- **Agent 数量**：$ K = 8 $（3 Synergy + 3 Uniqueness + 2 Redundancy）
- **训练配置**：
  - 优化器：Adam ($\beta_1=0.9$, $\beta_2=0.999$)
  - 学习率：$1 \times 10^{-4}$
  - Batch Size：32
  - 早停策略：patience=10
  - 重复次数：5次随机种子（42, 123, ..., 1024）

---

### 🆚 基线方法对比
| 方法 | 类型 |
|------|------|
| Early Fusion (Concat.) | 简单拼接 |
| Late Fusion (Attn.) | 注意力聚合输出 |
| MoE [25] | 经典 MoE |
| Sparse MoE [48] | 稀疏门控 MoE |
| MMoE [77] | 多模态交互专家 MoE（专门建模交互） |

> 所有 baseline 使用相同 backbone 与 embedding dimension，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 表1：主实验结果（Table 1）

| Method | ADNI (%) | MIMIC-IV (%) | MM-IMDB (%) | MOSI (MSE↓) | ENRICO (%) |
|--------|----------|--------------|-------------|--------------|------------|
| Early Fusion | 69.4 | 82.7 | 48.5 | 2.49 | 78.0 |
| Late Fusion | 70.9 | 81.9 | 48.7 | 2.50 | 78.4 |
| MoE | 69.8 | 82.3 | 49.0 | 2.53 | 78.9 |
| Sparse MoE | 68.9 | 82.3 | 49.8 | 2.52 | 78.7 |
| MMoE | 70.2 | 82.5 | 49.6 | 2.48 | 79.1 |
| **DAIN (Ours)** | **73.5*** | **84.2*** | **51.7*** | **2.41*** | **80.4*** |
| ↑Improvement | (+2.6) | (+1.7) | (+2.1) | (-0.07) | (+1.3) |

> ✅ 在所有五个数据集上均取得显著提升（p < 0.001），尤其在医学任务 ADNI 上提升达 **+2.6% accuracy**。

---

### ⚙️ 参数效率分析（Table 2）

| Method | Total Params | Effective Params |
|--------|---------------|------------------|
| Early Fusion | 8.2M | 8.2M |
| MoE (K=8) | 12.4M | 12.4M |
| Sparse MoE | 12.4M | 6.2M |
| **DAIN (Ours)** | **12.8M** | **5.8M** |

> 尽管总参数略高（因引入 Meta-Controller 和通信模块），但由于 **sparse activation**，每样本实际参与计算的有效参数仅为 **5.8M**，远低于完整 MoE。

---

### 🔍 消融实验（Ablation Study, Table 4）

| Variant | ADNI (%) | MM-IMDB (%) |
|--------|----------|-------------|
| Static MoE (K=8) | 70.1 | 49.5 |
| DAIN-Static（固定激活全部 agents） | 71.8 | 50.6 |
| DAIN-NoComm（无 agent 间通信） | 72.1 | 50.9 |
| **DAIN (Full)** | **73.5** | **51.7** |

> 结果表明：
- **动态调度** 贡献约 +1.7%，说明 context-aware agent selection 至关重要；
- **代理间通信** 贡献约 +1.4%，验证了协作共识机制的有效性；
- 二者结合带来显著增益。

---

### 📈 效率-精度权衡分析（Table 5 & Figure 5）
通过调节最大允许激活 agent 数 $ T $ 进行测试：

| $ T $ | MIMIC-IV Acc (%) | Avg #agents | MOSI MSE |
|-------|------------------|-------------|---------|
| 1 | 82.1 | 1.0 | 2.58 |
| 2 | 83.3 | 1.8 | 2.49 |
| **4** | **84.2** | **3.4** | **2.41** |
| 6 | 84.0 | 5.1 | 2.42 |
| 8 | 83.9 | 8.0 | 2.43 |

> ✅ 最优性能出现在 $ T=4 $，即仅需一半 agent 即可达到峰值表现，进一步证明 **动态稀疏选择** 的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **动态代理协作优于静态融合**  
   将多模态融合重构为“智能体协作推理”过程，能更好地捕捉复杂的、依赖上下文的交互模式。

2. **稀疏激活 + 压缩通信 = 高效且高性能**  
   DAIN 在降低计算负担的同时提升了泛化能力和最终性能，打破了“高效 vs 高性能”的传统权衡。

3. **具备强可解释性**  
   agent 激活模式随输入变化而动态调整，例如：
   - 在 ADNI 中，认知正常患者主要激活 **Uniqueness agents**；
   - MCI/Dementia 患者更多依赖 **Synergy agents**，反映疾病诊断需要整合多种模态信息。

4. **Meta-Controller 是关键组件**  
   它不仅决定谁参与，还调控如何交流，实现了真正的 context-driven reasoning。

---

### ⚠️ 局限性
1. **Agent 分配需手动设定**  
   当前固定为 3 Synergy + 3 Uniqueness + 2 Redundancy，未来可探索自动分配或自适应增长机制。

2. **Meta-Controller 开销不可忽略**  
   对极轻量级 backbone 可能不划算，需进一步优化控制器结构。

3. **尚未拓展至生成式任务**  
   当前评估集中于分类与回归，未验证在 image captioning、VQA 等生成任务上的潜力。

4. **解释性非形式化证明**  
   agent 激活提供直观洞察，但尚不能作为高风险应用（如临床决策）的正式解释依据，需结合 post-hoc 方法。

---

### 🔮 未来工作方向
- 设计 **learned agent allocation policy**
- 探索 **hierarchical agent 组织结构**
- 扩展至 **generative multimodal tasks**
- 构建 **end-to-end 可验证的 reasoning graph**
- 在更大规模模型（如 MLLMs）中集成 DAIN 架构

---

## ✅ 总结
**DAIN** 成功地将 **Partial Information Decomposition** 理论与 **dynamic Mixture-of-Experts** 实践相结合，提出了一个新颖、高效、可解释的多模态推理框架。其实验充分、设计合理，在多个领域展现出强大性能与灵活性，为未来的 **agent-based multimodal AI** 提供了重要范式参考。

</details>

---

### 5. [Depth Exploration for LLM Decoding](https://arxiv.org/abs/2606.29223)

**Authors**: Weisi Yang, Zipeng Sun, Stephen Xia  
**Category**: cs.LG  
**Published**: 2026-06-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.29223v1  

#### Abstract
Autoregressive LLM decoding evaluates every generated token through the full layer stack, even though many tokens become predictable at intermediate depths. Existing lossless depth-adaptive methods exploit this redundancy by choosing a single non-final exit depth and verifying its prediction with th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Depth Exploration for LLM Decoding**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代 Large Language Models (LLMs) 在 **autoregressive (AR) decoding** 过程中，每个生成的 token 都必须通过完整的模型层堆栈（full layer stack），导致解码延迟高、计算成本大。尽管许多 token 在中间层（intermediate depths）就已可预测，传统方法仍强制执行全深度前向传播。

现有基于 **lossless depth-adaptive decoding** 的方法（如 LayerSkip、AdaDecode、DEL）采用“选择-验证”范式（draft-and-verify），即在某个选定的非最终层退出并生成候选 token，再由最终层验证其正确性。然而，这类方法存在 **selection bottleneck**：
- 若选择过晚 → 浪费计算资源；
- 若选择过早 → 触发 fallback，丢弃依赖该 token 的后续草案。

因此，如何更稳健地利用深度冗余（depth redundancy）成为关键挑战。

---

### **提出的新方法：DEX (Depth Exploration Decoding)**

作者提出 **DEX (Depth Exploration Decoding)**，一种全新的 **lossless decoding algorithm**，其核心思想是将传统的单深度选择（single-depth selection）替换为 **多候选深度的并行探索（parallel depth exploration）**。

#### **核心机制：expand-commit-collapse**
1. **Expand（扩展）**：  
   在多个候选深度（candidate depths）上并行生成 token 分支，形成一个 depth-position lattice。
2. **Commit（提交）**：  
   使用最终层输出作为参考（final-depth reference），验证所有分支，并提交与参考一致的 token。
3. **Collapse（坍缩）**：  
   保留最早匹配的分支状态，清除不一致的分支，确保等价于标准 AR decoding。

#### **关键创新**
- 引入 **Earliest Available Depth (EAD)** 概念，量化 token 在深度维度上的“就绪时间”。
- 将 exit depth 决策从“单次押注”变为“多路径覆盖”，降低因错误选择带来的风险。
- 错误容忍度由 **exploration resolution（△(X)）** 控制，而非 exit predictor 的准确性。

---

### **相比现有方法的优势**
| 维度 | 传统 depth-selection 方法 | DEX |
|------|--------------------------|-----|
| 决策方式 | 单一 exit depth 选择 | 多深度并行探索 |
| 错误代价 | 早期退出失败 → 全深度 fallback + 回滚开销 | 只要有一个更深分支匹配即可复用 |
| 性能上限 | 受限于 exit policy 质量 | 接近 EAD oracle 上限 |
| 扩展性 | 固定策略，难以随硬件扩展 | 更多 explorers ⇒ 更细粒度 ⇒ 更高速度 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **GSM8K**：数学推理任务
- **HumanEval**：代码生成任务
- **XSum**：摘要生成任务

### **模型**
- **Early-exit-trained models**：
  - `LS-CodeLlama-34B`
  - `LS-Llama-2-70B`（经 LayerSkip 训练）
- **Standard LLMs**（通用指令模型）：
  - `CodeLlama-34B-Instruct`
  - `Llama-2-70B-Instruct`
  - `Qwen3-32B`

### **评估指标**
- **端到端吞吐量（end-to-end throughput）**：单位时间内生成的 token 数量（tokens/sec）
- **理论加速比（depth-side speedup）**：基于理想化 layer-work accounting 的加速上限
- **实际 walltime 加速**：真实运行时间测量

### **基线方法对比**
分为两类：

#### **(1) 深度轴对比（depth-axis baselines）**
- **LayerSkip Self-Speculative Decoding (LSSD)**
- **AdaDecode**
- **DEL**

#### **(2) 端到端加速对比（end-to-end baselines）**
- **Autoregressive (AR) decoding**
- **Tensor Parallelism (TP)**（张量并行）
- **Lookahead Decoding**
- **PEARL**（并行 speculative decoding）
- **EAGLE-2 / EAGLE-3**（基于 draft tree 的 speculative decoding）

### **实验配置**
- 硬件：NVIDIA H100 GPUs
- DEX 配置：`DEX-1/K` 表示使用 K 个 depth explorers，均匀划分模型深度
  - 如 `DEX-1/3`：3 个 explorer，各负责 1/3 模型深度
- 所有方法在相同 GPU 资源下比较（如 3 GPU vs 3 GPU）
- 解码策略：greedy decoding 或 temperature sampling（T=1.0）
- 最大生成长度：512（Llama 系列），1024（Qwen3）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **在 early-exit-trained 模型上的表现（Fig. 3）**
- DEX 在 `LS-CodeLlama-34B` 和 `LS-Llama-2-70B` 上均显著优于所有 depth-selection 基线。
- **相对基线提升**：
  - 平均比 LSSD 快 **1.3–1.6x**
  - 比 AdaDecode 和 DEL 快 **1.2–1.5x**
- 接近 **EAD oracle**（理论最优）的性能，剩余 gap 明显小于其他方法。

#### ✅ **端到端吞吐量对比（Fig. 5）**
| 模型 | 最佳 DEX 配置 | 吞吐量优势 |
|------|---------------|-----------|
| `CodeLlama-34B` | DEX-1/3 | ≈ 匹配 3-GPU PEARL，优于 TP 和 EAGLE-3 |
| `Llama-2-70B` | DEX-1/8 | 达到最高吞吐，超越所有 baseline |
| `Qwen3-32B` | DEX-1/4 | 显著优于 AR、TP、PEARL、EAGLE-3 |

> 💡 **结论**：DEX 在匹配 GPU 资源下已具备竞争力，且随着 K 增加持续提升。

---

### **消融实验结果（Ablation Study, Fig. 4）**

#### 🔹 **探索分辨率的影响（Exploration Resolution）**
- 更细粒度（increasing K）带来稳定增益。
- 在 `LS-Llama-2-70B` 上，从 `DEX-1/2` 到 `DEX-1/8`，吞吐持续上升，逼近 EAD oracle。
- 支持理论分析：**finer resolution ⇒ smaller rounding error ⇒ higher speedup**

#### 🔹 **Single-Exit vs Full Exploration**
- 构造一个仅允许首层和末层 exit 的变体（DEX-single-exit）进行对比。
- 结果显示，在粗粒度时两者接近，但在细粒度下 **full exploration 明显胜出**。
- 说明收益不仅来自并行执行，更来自 **多候选路径的选择灵活性**。

#### 🔹 **DECS（Depth-Coupled Sampling）的作用**
- DECS 通过对浅层已提议 token 进行 masking，鼓励深层探索不同候选。
- 实验表明：
  - 在 greedy 和 sampling 模式下均带来 **稳定吞吐提升（约 5–10%）**
  - 不改变最终 token 分布，保持 lossless 特性

---

## **4. 关键结论和发现**

### **主要发现**
1. **Selection Bottleneck 是现有 depth-adaptive 方法的根本限制**：
   - 单一 exit depth 决策易受不确定性影响，导致浪费或回滚。
2. **Parallel Depth Exploration 是更优替代方案**：
   - 将不确定性转化为可控的向上取整误差（rounding error），由 resolution 决定。
3. **DEX 实现了接近 EAD oracle 的效率**：
   - 在多种模型和任务上均优于代表性的 depth-selection 和 speculative decoding 方法。
4. **可扩展性强**：
   - 增加 depth explorers 数量（K）可系统性提升性能，尤其适用于多 GPU 场景。

---

### **方法的局限性**
1. **硬件依赖性强**：
   - 当前实现需要多个 GPU（K GPUs for DEX-1/K），不适合单卡部署。
2. **Lattice Expansion 开销随 K 指数增长**：
   - 每轮最多产生 $2^K - 1$ 个探索 token，内存和通信压力增大。
   - 当前实验最大测试到 K=8，更大规模可能面临瓶颈。
3. **对标准 LLM 效果有限（无 early-exit 训练）**：
   - 中间层输出与最终层不一致 → EAD 集中在深层。
   - 需借助 **inducing adapter** 微调中间层对齐能力（增加训练成本）。

---

### **未来工作方向**
1. **结合 token-axis 与 depth-axis 加速**：
   - 将 DEX 与 speculative decoding（如 EAGLE）结合，实现双轴加速。
   - 挑战在于协调 token tree verification 与 depth lattice collapse。
2. **稀疏化探索 lattice**：
   - 引入 pruning 策略，剔除低价值分支，降低 expansion 成本。
3. **动态 exploration resolution**：
   - 根据上下文自适应调整探索深度集合 X。
4. **单设备上的轻量级实现**：
   - 探索在单 GPU 上通过流水线模拟 depth exploration 的可行性。

---

## **总结**
DEX 提出了一种全新的视角来理解 LLM 解码过程中的 **depth redundancy**，并通过 **parallel depth exploration** 和 **expand-commit-collapse** 机制，有效规避了传统方法的 selection bottleneck。实验证明，DEX 不仅在理论层面更接近最优，也在实际吞吐上达到了与主流 speculative 和 distributed decoding 方法相媲美甚至领先的水平，为高效 LLM inference 提供了一个 **可扩展、lossless、硬件友好的新范式**。

</details>

---

### 6. [SMART-MIG: A Learning Framework for Scalable and Energy-Efficient GPU Scheduling](https://arxiv.org/abs/2606.29775)

**Authors**: Wenqing Yu, Neel Karia, Tanvi Hisaria, Clifford Stein, Olivier Tardieu, Asser Tantawi  
**Category**: cs.DC  
**Published**: 2026-06-30  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.29775v1  

#### Abstract
The emergence of Multi-Instance GPU (MIG) technology enables us to run smaller machine learning models on partitions of a GPU rather than the entire device, thus improving utilization and reducing energy consumption, albeit with potential performance trade-offs. Meanwhile, the growing energy demands...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SMART-MIG: A Learning Framework for Scalable and Energy-Efficient GPU Scheduling》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着 AI 和机器学习（尤其是大语言模型）的广泛应用，GPU 集群在数据中心中的能耗急剧上升。传统 GPU 调度方法存在以下问题：
- 忽略 **Multi-Instance GPU (MIG)** 技术的动态资源分割能力；
- 多数调度策略仅关注吞吐量、公平性或 SLO 合规，**缺乏对能效与延迟（tardiness）的联合优化**；
- 在大规模场景下，调度决策面临 **NP-hard 问题**，算法复杂度高，难以实时求解。

本文聚焦于如何在大规模、异构、动态变化的负载环境中，实现 **可扩展、节能且低延迟的 MIG 调度系统**。

---

### 🚀 提出的新方法与创新思路

作者提出 **SMART-MIG** —— 一种结合 **机器学习（ML）与运筹学（OR）** 的智能调度框架，其核心创新如下：

#### （1）**Mean-Field Multi-Agent Reinforcement Learning (MF-MARL) 用于动态 MIG Repartitioning**
- 利用 **MF-MARL** 将传统的多智能体强化学习（MARL）从指数级状态空间压缩为常数维输入输出空间。
- 所有 MIG 设备被视为**可互换、不可区分的个体**，控制器只关心“每种配置有多少个”，而非“哪个具体设备”。
- 引入 **Top-k Sampling** 提升训练稳定性，避免极端动作选择。

> ✅ 优势：**算法复杂度与 GPU 数量无关**，实现了真正的可扩展性（scalability），适用于工业级规模部署。

#### （2）**定制化启发式调度算法（如 CEDF）用于任务分配**
- 在固定 MIG 分区后，采用基于 **Earliest Deadline First (EDF)** 的调度策略，并结合 job throughput 特性和 MIG 的 **concave power curve** 进行优化。
- 提出 **Categorical EDF (CEDF)**：将 jobs 按 sublinearity 分类，优先匹配适合 slice size 的任务，减少资源浪费。

#### （3）**理论下界（Lower Bounds）构建用于公平评估**
- 推导了 **energy consumption** 和 **tardiness** 的理论下界：
  - 能耗下界通过将 sublinear jobs 线性化并转化为最小 makespan 问题；
  - 延迟下界通过时间索引的混合整数规划（Time-indexed MIP）建模。
- 这些下界为不同调度策略提供了统一、客观的性能基准，减少了因 workload 差异导致的评估偏差。

---

### 🔍 相比现有方法的优势

| 维度 | SMART-MIG | 现有方法（如静态分区、DVFS、简单启发式） |
|------|-----------|-------------------------------|
| 可扩展性 | ✔️ 常数复杂度，支持大规模集群 | ❌ 复杂度随设备/任务增长而爆炸 |
| 动态适应性 | ✔️ 支持在线 repartitioning，响应负载波动 | ❌ 多为静态配置或有限重配 |
| 能效-延迟权衡 | ✔️ 显式联合优化 ET 目标函数 | ❌ 多侧重单一目标（如吞吐或功耗） |
| 评估严谨性 | ✔️ 提供理论下界作为参考 | ❌ 缺乏统一基准，结果难比较 |

---

## 2. 核心实验方法和设置

### 📊 数据集与工作负载生成

由于真实数据中心 trace 中缺少完整的 deadline 信息，作者采用合成 workload 模拟：

- **Job 类型**：分为 training 和 inference 两类。
- **到达过程**：使用 **Poisson 过程**模拟 job 到达，基础速率随时间变化（见图5），乘以倍率得到不同负载强度（如 12x, 16x, 20x）。
- **执行时间建模**：
  - Training jobs：处理时长服从 **lognormal 分布**；
  - Inference jobs：处理时长服从 **exponential 分布**；
  - Slice-wise throughput 基于 ResNet-50 和 BERT-Base 实测数据 [25][6]。
- **Deadline 设置**：$ d_j \sim \text{Unif}(1, 1.5) \times p_{j,7g} $，即基于最大 slice 下完成时间的随机扩展。

> 💡 总共模拟超过 **500万 jobs**（相当于 5000 天的真实运行）。

---

### ⚙️ 实验设置

- **硬件平台**：NVIDIA A100-40GB GPU，支持最多 7 个 MIG slices。
- **MIG 配置**：共 12 种有效非冗余配置（见 Fig. 2），涵盖不同大小的 slices（1g, 2g, ..., 7g）。
- **调度频率**：每当有 job 到达或完成时触发调度；**每 5 个事件调用一次中央控制器进行 repartitioning**，以降低开销。
- **Repartitioning 开销**：计入 4 秒切换延迟（来自 [5]），并通过提前扣除 deadline 来建模。

---

### 📏 评估指标

定义综合目标函数 **ET Value**（Energy-Tardiness Tradeoff）：

$$
\text{ET} = \frac{\bar{e} + \alpha \bar{t}}{1 + \alpha}, \quad \alpha = 1
$$

其中：
- $\bar{e}$：平均总能耗（kJ）
- $\bar{t}$：平均 tardiness（完成时间超出 deadline 的部分）
- $\alpha$ 为权衡参数，设为 1 表示同等重视 energy 与 tardiness

其他指标：
- 平均 tardiness
- 总 energy consumption
- 配置选择分布（configuration proportions）

---

### 🆚 基线方法对比

| 方法 | 描述 |
|------|------|
| **No Partition (Baseline)** | 不启用 MIG，整个 GPU 作为一个单元使用 |
| **Static MIG (EDF / MET / CEDF)** | 固定 MIG 配置（如 [1,1,2,3,3,5,5,10]），仅做任务调度 |
| **SMART-MIG (CEDF + MF-MARL)** | 动态 repartitioning + CEDF 调度，本文提出的方法 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table I 与 Fig. 12）

| 指标 | SMART-MIG | No Partition | 提升幅度 |
|------|----------|-------------|---------|
| **Tardiness @ 20x 到达率** | 1.0037 | 96.67 | ↓ **99%** |
| **Energy @ 20x 到达率** | 2.5M kJ | 3.1M kJ | ↓ **19.4%** |
| **Tardiness @ 12x 到达率** | 0.18 | 0.34 | ↓ **47%** |

> 🔥 在高负载下，non-MIG GPU 几乎崩溃（tardiness 高达百倍），而 MIG 显著缓解 backlog。

---

### 🆚 与静态 MIG 方法对比（Fig. 12）

| 对比项 | SMART-MIG vs Static CEDF |
|--------|--------------------------|
| **Average Tardiness** | ↓ **25%** |
| **Energy Consumption** | ↓ **1.2%** |
| **Overall ET Metric** | ↑ **18% 更优** |

> ✅ 即使 CEDF 已是优秀的静态调度器，加入 **动态 repartitioning** 仍带来显著提升。

| SMART-MIG vs Static EDF |
|-------------------------|
| Tardiness ↓ 40% |
| Energy ↓ 7% |
| ET ↑ 32% |

---

### 🔬 消融实验与行为分析（Fig. 13）

- **Repartitioning 策略自适应性强**：
  - 当 job sublinearity 较高（更非线性）时，系统倾向于选择含更多 small slices 的配置（如 config 8–12）；
  - 当 sublinearity 较低时，偏好 large slices（如 config 1–3）；
- **最常用配置**：1 (7g), 2 (4g+3g), 3 (4g+2g+1g), 7 (2g+3g)，合计占比 >80%。
- **small slices 使用比例约 5%**，说明需平衡并行性与效率。

> ✅ 表明 SMART-MIG 能根据 workload 特征智能调整资源配置。

---

### 📉 与理论下界的差距分析

| 方法 | 能耗相对于理论下界 |
|------|--------------------|
| SMART-MIG (RL) | **仅高出 27%** |
| Static CEDF | 高出 ~29% |
| 理论下界本身因 job 线性化而偏松（not tight） |

> ✅ 表明 SMART-MIG 已非常接近理论上最优水平，尤其在多目标优化中表现优异。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **MIG 技术对现代 AI 负载至关重要**  
   在高到达率下，non-MIG GPU 的 tardiness 是 MIG 的 **60–100 倍**，几乎无法满足 SLO。

2. **动态 repartitioning 显著优于静态分区**  
   结合 MF-MARL 的 SMART-MIG 在 ET 效率上比最佳静态方案提升 **18%**，且在各种 arrival rate 下保持鲁棒。

3. **MF-MARL 实现真正可扩展的强化学习控制**  
   输入输出维度独立于 GPU 数量，使得 RL 可应用于数千台设备的大规模集群。

4. **理论下界为调度研究提供新标准**  
   首次为 MIG 调度问题建立 energy 与 tardiness 的下界，推动领域向更科学评估演进。

---

### ⚠️ 局限性

1. **Job 线性化假设影响下界紧致性**  
   当前下界通过对 sublinear jobs 线性化获得，可能低估真实最优值，存在一定误差。

2. **未考虑跨 GPU 的分布式训练**  
   MIG 不支持 P2P 通信，因此不适用于需要跨多个 MIG slices 的分布式训练任务（如 large model training）。

3. **Repartitioning 开销建模简化**  
   4 秒切换时间是经验值，实际中可能受固件、驱动等因素影响。

4. **调度器与控制器耦合较强**  
   当前 SMART-MIG 使用 CEDF 作为底层调度器，虽可替换，但整体性能依赖其质量。

---

### 🔮 未来工作方向（原文 VII 节）

1. **将 RL 应用于 job scheduling 本身**  
   当前调度仍用启发式算法，未来可尝试端到端 RL 控制任务分配。

2. **进一步收紧理论下界**  
   避免 job 线性化，直接建模 sublinear throughput，提高下界精度。

3. **扩展至 heterogeneous GPU 环境**  
   支持多种型号 GPU（如 A100 + H100）混合部署下的协同调度。

4. **探索几何级 job 对齐以降低散热需求**  
   通过物理布局优化减少热点，从而降低 cooling energy。

---

## ✅ 总结

| 维度 | 内容 |
|------|------|
| **核心思想** | 利用 MF-MARL 实现可扩展的动态 MIG 分区 + 启发式调度联合优化 |
| **关键技术** | Mean-Field MARL, Top-k Sampling, CEDF, Energy/Tardiness Lower Bounds |
| **核心成果** | ET 效率提升 18%，tardiness 最多降低 99%，能耗降低近 20% |
| **理论价值** | 建立首个 MIG 调度问题的能量与延迟下界 |
| **实践意义** | 为大规模绿色 AI 数据中心提供可行的智能调度解决方案 |

> 🌟 **一句话总结**：  
> SMART-MIG 通过 **Mean-Field MARL + 定制调度 + 理论基准**，首次实现了在大规模 MIG 系统中高效、节能、低延迟的在线调度，在性能与可扩展性之间取得突破性平衡。

</details>

---

### 7. [Beyond Uniform Experts: Cost-Aware Expert Execution for Efficient Multi-Device MoE Inference](https://arxiv.org/abs/2606.29982)

**Authors**: Hui Zang, Pengfei Xia, Hong Liu, Jiajia Chu, Tuo Hao, Minghao Chen, Rui Zhang, Ziyang Zhang  
**Category**: cs.DC  
**Published**: 2026-06-30  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.29982v1  

#### Abstract
Mixture-of-Experts (MoE) architectures enable language models to achieve unprecedented scale via sparse activation. However, their inference performance is often limited by data movement bottlenecks. Two coupled challenges exacerbate this limtation: (1) Importance-Agnostic Cost: Low-contribution exp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Uniform Experts: Cost-Aware Expert Execution for Efficient Multi-Device MoE Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对 **Mixture-of-Experts (MoE)** 架构在多设备系统中推理时面临的两大核心瓶颈：

- **C1: Importance-Agnostic Cost（重要性无关的成本）**  
  所有被激活的专家无论其对输出的贡献大小（即路由分数 G(t)e 高低），都会触发相同的内存访问或 Host-to-Device (H2D) 参数传输开销，导致低贡献专家浪费大量带宽资源。

- **C2: System-Level Imbalance（系统级不平衡）**  
  在多设备部署中，端到端延迟由最慢的设备决定（straggler effect）。即使在一个轻负载设备上减少成本，若未优化瓶颈设备，则无法提升整体性能。

### 🚀 提出的新方法：Cost-Aware Expert Execution (CAEE)
CAEE 是一个硬件感知的运行时框架，通过联合建模 token 级专家重要性和系统级执行成本，实现高效的 MoE 推理优化。

#### 主要创新点：
1. **轻量级多设备成本建模（Lightweight Multi-Device Cost Modeling）**  
   - 建立基于离线简短 profiling 的成本模型，估算每个专家在特定设备上的数据移动成本 $ C(e,d) $。
   - 引入 **max-aggregation 层成本函数 $ F_{\text{cost}} $**，显式捕捉跨设备的 **straggler 效应**，以系统中最慢设备的代价为优化目标。

2. **成本感知的剪枝策略（Cost-Aware Pruning）**  
   将专家选择形式化为约束优化问题：
   - 目标：最小化系统最大延迟 $ F_{\text{cost}} $
   - 约束：限制移除的专家总重要性不超过阈值 $ \delta $
   - 动态划分强制保留集（Amust）与候选剪枝集（Acand），仅当加入某专家不增加关键路径延迟时才保留。

3. **低开销补偿机制（Low-overhead Compensation）**  
   - 对被剪枝专家的贡献进行重定向，仅分配给已激活或低成本的专家。
   - 无需额外参数加载或通信，完全通过本地 mask 和 Top-k 重选完成，避免引入新的数据移动。

### 🔍 相比现有方法的优势
| 方法类别 | 典型代表 | 缺陷 | CAEE 的优势 |
|--------|--------|------|-------------|
| 静态剪枝/压缩 | TSEP, MoE-Pruner | 忽略动态 token 重要性，降低模型容量 | 动态感知 token 级重要性，保持高精度 |
| 动态路由 | DA-MoE, DynMoE | 忽视 per-expert 硬件成本和系统级不平衡 | 显式建模硬件成本与 straggler 影响 |
| 专家卸载系统 | Mixtral-Offloading, Fiddler | 多为单设备设计，缺乏跨场景鲁棒性 | 支持 on-device 与 offloading 统一优化 |

> ✅ **核心优势总结**：CAEE 实现了“在哪里省成本真正有效”的精准判断——不是简单删掉低分专家，而是只在它们成为系统瓶颈时才规避。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
用于评估模型准确性的公开基准测试集，涵盖三大领域：

- **知识理解**：MMLU-shot5、CEval-shot5
- **数学推理**：GSM8K-shot5
- **代码生成**：HumanEval

### ⚙️ 实验设置

| 项目 | 设置详情 |
|------|----------|
| **模型** | DeepSeek-R1 671B（W8量化），含 256 个 MoE 专家，每 token 激活 8 个专家 |
| **部署模式** | 分两种场景：<br>• **Expert Offloading**：8 xPU，HBM 缓存有限，其余专家驻留主机内存<br>• **On-device Inference**：16 xPU，所有专家驻留 HBM |
| **硬件配置** | 使用 Huawei 自研系统，支持 PCIe/CXL 等互连协议 |
| **评估指标** |<br>• **TTFT**（Time to First Token）<br>• **TPOT / TOPT**（Time Per Output Token）<br>• 准确率变化（Accuracy Drop）<br>• 设备间负载分布（Load Imbalance） |
| **基线方法** | 原始 MoE 路由（无剪枝）、随机补偿策略等 |

### 🔁 关键变量控制
- **Offloading 场景**：设定不同的重要性保留阈值 $ \theta_{\text{imp}} \in \{0.9, 0.8, 0.7\} $
- **On-device 场景**：调整专家得分阈值 $ \theta \in [0.3, 1.2] $
- **批量大小**：从 1 到 128 不等，覆盖典型推理负载

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### （1）**专家卸载场景（Expert Offloading）**

| Batch Size | TTFT (s) Baseline → CAEE ($\theta_{\text{imp}}=0.8$) | Gain | TOPT (s) Baseline → CAEE | Gain |
|------------|-----------------------------------------------------|------|---------------------------|------|
| 1          | 6.171 → 5.887                                       | 4.60%↓ | 0.819 → 0.704             | 14.04%↓ |
| 4          | 16.638 → 15.971                                     | 4.01%↓ | 0.957 → 0.824             | 13.90%↓ |
| 8          | 27.961 → 26.863                                     | 3.93%↓ | 1.024 → 0.849             | 17.09%↓ |
| 16         | 46.273 → 44.703                                     | 3.39%↓ | 1.377 → 1.143             | 16.99%↓ |

> ✅ 最大 **TOPT 下降达 23.24%**（$\theta_{\text{imp}}=0.7$, BS=16）

#### （2）**On-device 推理场景**

| Batch Size | TPOT (ms) Baseline → CAEE ($\theta=0.8$) | Gain | Accuracy 变化（GSM8K, BS=128） |
|------------|-----------------------------------------|------|-------------------------------|
| 64         | 59.29 → 52.30                            | 11.78%↓ | 0.8992 → 0.9121 (+1.44%)       |
| 128        | 68.78 → 62.18                            | 9.59%↓  | 0.8992 → 0.9121 (+1.44%)       |

> ✅ 在并发 128 时仍可获得 **>9.5% 延迟下降且精度略有上升**

### 🆚 与基线方法对比
- **相比原始 MoE 路由**：
  - 平均 **端到端延迟降低 8%–18%**
  - **准确率损失 <1%**（多数任务在 0.5% 以内）
- **在高批处理下收益更显著**：因数据移动压力更大，CAEE 的剪枝效果更明显

### 🔍 消融实验结果
#### （1）补偿机制对比（见 Fig. 9）
- 对比 **随机重定向** vs **CAEE 的低开销补偿**
- 结果显示：在 GSM8K 上，CAEE 补偿始终优于随机策略，尤其在较低 $\theta_{\text{imp}}$ 时差距扩大
- ➤ 证明：**智能重定向能更好维持语义一致性**

#### （2）负载均衡观察（见 Fig. 10）
- 可视化多个 MoE 层中各设备的“缓存未命中专家数”
- 基线存在严重不均衡（如 Layer 30 中 Device 0 达 9 次，其他仅 2–4 次）
- CAEE 成功将峰值设备的转移次数降至 3，显著缓解 straggler 问题
- ➤ 证明：**CAEE 真正作用于系统瓶颈而非局部优化**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MoE 推理是典型的 memory-bound 和 transfer-bound 过程**，单纯计算稀疏化不足以释放性能潜力。
2. **专家的重要性与其执行成本高度解耦**，统一对待所有激活专家会造成严重的资源浪费。
3. **系统级性能受最慢设备支配**，必须以 `max-aggregation` 方式建模成本才能实现有效优化。
4. **CAEE 能在几乎无损精度的前提下（<1% accuracy drop），实现 8%–18% 的端到端延迟下降**，适用于多种部署形态（offloading / on-device）。

### ⚠️ 方法的局限性
- **依赖离线 profiling** 来校准带宽和延迟参数，在异构性极强的生产环境中可能需要频繁更新。
- **补偿机制局限于已有活跃专家集合内重分配**，若强制剪枝过多可能导致表达能力受限。
- 当前实现集中于推理阶段，未涉及训练时的协同优化。

### 🔮 未来工作方向
- 扩展至 **异构设备混合集群**（如 GPU+NPUs+NVMes）
- 结合 **prefetching 机制**，进一步隐藏 H2D 开销
- 探索 **动态自适应阈值调节**，根据实时负载自动调整 $ \theta $
- 将 CAEE 思想推广至其他稀疏架构（如 Sparse Attention, Block-Sparsity）

---

> 💡 **一句话总结**：  
> CAEE 通过“**按需加载、精准避堵**”的理念，首次实现了 MoE 推理中 **重要性—成本—系统瓶颈** 的三维协同优化，在不牺牲精度的前提下大幅提升多设备系统的实际吞吐效率。

</details>

---

### 8. [Predict, Reuse, and Repair: Accelerating Dynamic Sparse Attention for Long-Context LLM Decoding](https://arxiv.org/abs/2606.30389)

**Authors**: Tianyu Wang, Gourav Rattihalli, Aditya Dhakal, Junbo Li, Zhiwei Ren, Dejan Milojicic, Longfei Shangguan  
**Category**: cs.LG  
**Published**: 2026-06-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.30389v1  

#### Abstract
Dynamic sparse attention (DSA) accelerates long-context LLM decoding by attending to only the top-K KV blocks relevant to each query, but it introduces a serialized selection-to-attention dependency that emerges as a new latency bottleneck. We present PRR, a speculate-reuse-repair runtime that explo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Predict, Reuse, and Repair: Accelerating Dynamic Sparse Attention for Long-Context LLM Decoding*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
动态稀疏注意力（**Dynamic Sparse Attention, DSA**）通过仅关注与当前查询最相关的 top-K KV 块来加速长上下文 LLM 解码，显著减少了计算量。然而，DSA 引入了一个新的瓶颈：**selection-to-attention 依赖关系**——必须先完成 top-K 块的选择，才能开始注意力计算，这导致了严重的串行化延迟。

随着上下文长度增长，**selection 阶段可占单个 token 解码延迟的高达 41%**，成为解码关键路径上的主要瓶颈。

### 提出了什么新方法或新思路
本文提出 **PRR**（Predict, Reuse, and Repair），一种用于 DSA 的推测-重用-修复运行时系统，旨在打破 selection 与 attention 之间的串行依赖。

其核心思想是利用 DSA 中 top-K 块选择在时间上的局部性（temporal locality），对未来的 top-K 块进行预测，并**并行执行推测性注意力计算**，同时等待实际选择结果。一旦真实选择完成，只需“修复”未命中块（missed blocks），而无需重新计算全部注意力。

PRR 的三大关键技术组件：

1. **轻量级 EMA-based Predictor**  
   - 使用指数移动平均（Exponential Moving Average, EMA）模型跟踪每个 KV 块的重要性分数轨迹，预测下一时刻的 top-K 块集合。
   - 在 **prefill 阶段通过网格搜索自动校准超参数**（α, β, γ），实现提示自适应（prompt-adaptive）预测，无需额外训练。

2. **关键路径感知的推测预算（Critical-Path-Aware Speculation Budget）**  
   - 动态调整推测集合大小 `P`，确保推测性注意力计算不会超过 selection 阶段的时间窗口，从而避免延长关键路径。
   - 通过离线分析不同上下文长度下的 selection 和 speculative attention 延迟，构建查找表指导在线决策。

3. **基于 Online-Softmax 的增量修复内核（Incremental Repair Kernel）**  
   - 实现了一个定制的 CUDA 内核，基于 FlashAttention 的 online-softmax 机制，支持将未命中的块增量地合并到已有的部分注意力输出中。
   - 修复成本仅与 `|A\P|`（未命中块数）成正比，而非重新计算整个 `|A|`。

### 相比现有方法的优势
- **打破串行瓶颈**：首次系统性地识别并解决了 DSA 中 selection-to-attention 的关键路径依赖。
- **高精度预测 + 高效修复**：EMA 预测器达到约 **98% 的 top-K 覆盖率**，结合增量修复机制，避免了因预测失败而导致的全量重算开销。
- **零精度损失**：最终注意力输出始终覆盖真实的 DSA 选中块集，保持与原始 DSA 完全一致的下游任务准确性。
- **通用性强**：作为运行时优化，适用于多种 DSA 方法（如 Quest、InfLLM-v2），无需修改模型结构。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在五个代表性的长上下文基准上进行：
- **LongBench**：多语言、多任务长文本理解
- **InfiniteBench**：超长序列评估（>100K tokens）
- **RULER**：测试模型真实上下文长度能力
- **AIME**：数学推理任务（2024 年 AIME 竞赛题）
- **MATH500**：高质量数学问题集

### 实验设置和评估指标
- **模型**：GLM-4-9B、GLM-Z1-9B、DeepSeek-R1-8B、Llama3-8B-1M、Qwen3-14B、Qwen3-32B
- **DSA 方法**：Quest、InfLLM-v2
- **硬件平台**：NVIDIA H100 GPU，CUDA 12.8，张量并行度为 2
- **评估指标**：
  - **端到端解码延迟（end-to-end decoding latency）**
  - **加速比（speedup）**：相对于标准串行 DSA 执行的每 token 解码速度提升
  - **top-K 块预测命中率（overlap rate）**
  - **GPU 资源利用率**（SM、L2 BW、DRAM BW）

### 基线方法对比
- **Serial DSA**：标准的串行执行流程（selection → attention → FFN）
- **S1（Baseline Speculation）**：直接复用上一步的 top-K 块进行推测
- **S2（+ Incremental Repair）**：引入 PRR 的增量修复内核
- **S3（+ EMA Predictor）**：完整 PRR 架构

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| DSA 方法 | 平均加速比（Across Models & Benchmarks） |
|---------|----------------------------------------|
| Quest   | **1.42×**                                |
| InfLLM-v2 | **1.56×**                              |

在多个模型和任务上，PRR 最高可将每 token 解码延迟降低 **40%**。

#### 具体模型表现（Table 1）
| Model       | Quest (Avg.) | InfLLM-v2 (Avg.) |
|-------------|--------------|------------------|
| GLM-4       | 1.42×        | **1.56×**        |
| Qwen3-14B   | 1.40×        | 1.51×            |
| Llama3      | 1.38×        | 1.43×            |

> 注：InfLLM-v2 上收益更高，因其 selection 阶段更耗时，留给 speculation 的空间更大。

### 与基线方法的对比结果
| Benchmark     | Serial DSA | S1 (+Reuse) | S2 (+Repair) | S3 (PRR, Full) |
|---------------|------------|-------------|--------------|----------------|
| LongBench     | 1.00×      | 1.02×       | 1.34×        | **1.64×**      |
| InfiniteBench | 1.00×      | 1.00×       | 1.19×        | **1.38×**      |
| AIME          | 1.00×      | 1.02×       | 1.34×        | **1.62×**      |
| **Average**   | 1.00×      | 1.01×       | 1.30×        | **1.56×**      |

- **S1 几乎无增益**：因预测不准确触发全量重算，抵消了 speculation 收益。
- **S2 显著提升至 1.30×**：证明 **incremental repair 是关键突破**，使部分预测也能带来收益。
- **S3 达到 1.56×**：EMA 预测器将 top-K 覆盖率从 ~68%（S1）提升至 **~98%**，极大减少修复开销。

### 消融实验结果
- **EMA 预测器命中率**（Table 4）：
  - 在 GLM-4 上，Quest 达到 **98.05%**，InfLLM-v2 达到 **97.52%** 的 top-K 重叠率。
- **内核性能对比**（Table 3）：
  - 自定义稀疏注意力内核相比 FlashInfer 的 BlockSparseAttention，平均快 **2.22×**，最高达 **3.71×**（8K token 预算下）。
- **资源利用率低**（Figure 4）：
  - 即使在 512K 上下文下，SM、L2、DRAM 利用率均低于 40%，表明有充足资源支持 speculation。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Selection-to-attention 依赖是 DSA 的新瓶颈**：随着上下文增长，selection 成为关键路径主导因素。
2. **Top-K 选择具有强时间局部性**：连续解码步间平均有 **68% 的块被重复选中**，为 speculation 提供基础。
3. **轻量级 EMA 预测器高效且准确**：无需训练即可实现近 98% 的 top-K 覆盖率。
4. **Incremental Repair 是 speculation 可行的关键**：若无此机制，任何 miss 都会导致全量重算，使 speculation 失效。
5. **PRR 实现显著加速且无精度损失**：在多个 LLM 和 DSA 方法上稳定提速 **1.4–1.6×**，同时保持原始 DSA 的语义一致性。

### 方法的局限性
- 当前仅评估了 **非可训练型 DSA 方法**（如 Quest、InfLLM-v2），未包含需从头训练的机制（如 NSA）。
- 优化针对 **NVIDIA Hopper 架构**（H100）和半精度（FP16/BF16），未扩展至其他架构（如 Blackwell）或更低比特（如 FP8）。
- 实验限于 **batch size = 1**，未探索批量请求间的协调优化。
- 页面限制下主文仅展示 GLM-4 结果，其余模型结果见附录。

### 未来工作方向
- 将 PRR 扩展至 **可训练稀疏注意力机制**。
- 适配更多 **GPU 架构** 和 **低比特精度** 推理场景。
- 探索 **batch-level coordination**，在批量解码中进一步优化各阶段调度。
- 结合其他 KV cache 优化技术（如 FlexiCache），实现多层次加速。

---

> **GitHub 开源地址**：https://github.com/Tianyu9748/Incremental_FlashAttention

</details>

---

### 9. [One-Step Gradient Delay is Not a Barrier for Large-Scale Asynchronous Pipeline Parallel LLM Pretraining](https://arxiv.org/abs/2606.30634)

**Authors**: Philip Zmushko, Egor Petrov, Nursultan Abdullaev, Mikhail Khrushchev, Samuel Horv\'ath  
**Category**: cs.LG  
**Published**: 2026-06-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.30634v1  

#### Abstract
Modern large-scale LLM pretraining benefits from utilizing Pipeline Parallelism; however, synchronous implementations leave GPUs idle during pipeline bubbles, wasting computational resources. Asynchronous Pipeline Parallelism eliminates these bubbles, maximizing throughput at the cost of gradient st...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：One-Step Gradient Delay is Not a Barrier for Large-Scale Asynchronous Pipeline Parallel LLM Pretraining

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大规模 LLM 预训练广泛采用 **Pipeline Parallelism (PP)** 来突破单卡内存限制。然而，**同步 PP**（Synchronous PP）存在“pipeline bubbles”——由于各阶段需等待彼此完成前向/后向传播，导致大量 GPU 处于空闲状态，严重浪费计算资源。

**异步 PP**（Async PP）通过消除同步机制来完全去除 bubbles，提升吞吐量，但引入了 **梯度延迟**（gradient staleness），即参数更新基于过时的梯度。传统观点认为这种延迟会导致训练不稳定甚至发散，因此 Async PP 在实践中受限。

本文挑战这一共识，系统研究了在 **固定一步延迟**（one-step gradient delay）下，如何实现稳定高效的异步预训练。

---

### 🚀 提出的新方法与新思路

#### （1）提出优化器选择是关键
论文首次指出：**梯度延迟带来的性能下降并非不可避免，而是高度依赖于所使用的 optimizer**。  
- **AdamW**（当前主流）在延迟下表现极差（验证损失显著上升）。  
- **现代优化器如 Muon** 则表现出强鲁棒性，在相同超参下几乎无损。

#### （2）引入 Error Feedback 启发的修正机制
受 **Error Feedback (EF)** 思想启发，作者设计了一种**优化器无关**的更新级校正方法：
```python
x_{t+1} = x_t - 2 * u_{t-1}(g_{t-1}) + u_{t-2}(g_{t-2})
```
该方法将上一轮“本应执行”的更新与实际执行的更新之间的差异作为补偿项加入当前步骤，有效缓解延迟影响。

#### （3）理论支持
首次为 **LMO 类算法**（如 Muon）在梯度延迟下的收敛性提供了理论分析，证明其在有/无 EF 修正的情况下均能收敛。

#### （4）采用 PipeDream-2BW 调度策略
使用 **PipeDream-2BW** 而非原始 PipeDream，确保所有 stage 的梯度延迟恒定为一步，避免因延迟不一致（variable delay）导致的额外不稳定性。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法（如 PipeDream） | 本文方法 |
|------|--------------------------|--------|
| **延迟模式** | Variable delay（随 stage 变化） | Fixed one-step delay（统一） |
| **优化器兼容性** | 仅对特定 optimizer（如 NAdam）部分有效 | 支持多种现代 optimizer（Muon, SOAP, Adan 等） |
| **性能差距** | 与同步训练差距大（>0.2 loss） | 几乎无差距（+EF 下完全匹配） |
| **理论保障** | 缺乏针对 LMO 的延迟分析 | 提供 LMO 在延迟下的收敛证明 |
| **实用性** | 需复杂调参 | 使用默认超参即可达到高性能 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **FineWeb-Edu**：用于中小规模实验（135M / 360M 模型）
- **FineWeb**：用于大规模 MoE 模型训练（2B / 10B 参数）

### ⚙️ 实验设置
| 项目 | 设置说明 |
|------|---------|
| **模型架构** | - SmolLM-2（135M, 360M）<br>- 自定义 MoE 架构（2B, 10B） |
| **任务** | Decoder-only LLM 的 next-token prediction |
| **Token-to-Parameter Ratio** | 20:1（遵循 Chinchilla 最优比例） |
| **Batch Size** | - 中小模型：约 180K–360K tokens<br>- 大模型：1M–4M tokens（略高于理论最优以提高 GPU 利用率） |
| **学习率调度** | Cosine decay，最低为峰值的 10%，warmup 占 10% 训练步数 |
| **其他设置** | Gradient clipping at 1.0，weight decay = 0.1 |

### 📊 评估指标
- **主指标**：Training loss 和 Validation loss
- **下游能力验证**：MMLU、ARC、HellaSwag、PIQA 等多项基准测试
- **对比维度**：
  - 同步 vs 异步（标准延迟）
  - 是否启用 Error Feedback
  - 不同 optimizer 表现
  - 不同 pipeline 深度下的稳定性

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **Synchronous PP** | 标准同步训练，作为性能上限 |
| **Async PP (no EF)** | 使用 PipeDream-2BW 的一步延迟异步训练 |
| **Async PP + EF** | 上述基础上加入 Error Feedback 修正 |
| **Sync Start** | 先同步训练再切换到异步（已有稳定技巧） |
| **Original PipeDream** | 对比 variable delay 调度的影响 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 3 & Figure 1）

#### 🔹 10B MoE 模型（200B tokens）
| 方法 | 验证损失（Validation Loss） |
|------|----------------------------|
| Synchronous | **1.906** |
| Async (no EF) | 1.911 |
| **Async + EF** | **1.906** ✅ |

✅ **结论**：**异步 + EF 完全追平同步训练性能**，且使用**完全相同的超参数**。

#### 🔹 360M 模型（不同 optimizer 对比，Table 1）
| Optimizer | Sync Loss | Async Loss | 差距 Δ |
|----------|---------|-----------|-------|
| **Muon** | 2.578 | 2.590 | **+0.012** |
| **Adan** | 2.641 | 2.651 | **+0.010** |
| SOAP | 2.581 | 2.608 | +0.027 |
| **AdamW** | 2.612 | 2.890 | **+0.278** ❌ |

➡️ **发现**：**AdamW 极度敏感于延迟**，而 Muon、Adan 等现代 optimizer 鲁棒性强。

---

### 🔍 消融实验结果

#### （1）Error Feedback 效果（Table 2）
| Optimizer | Async Loss (no EF) | Async + EF | 改善幅度 |
|----------|--------------------|------------|----------|
| Muon | 2.590 | **2.583** | ↓ 71% gap |
| AdamW | 2.890 | **2.640** | ↓ 90% gap |
| SOAP | 2.608 | **2.590** | ↓ 67% gap |

➡️ EF 显著缩小同步-异步差距，尤其对原本退化严重的 optimizer 更明显。

#### （2）Momentum 影响（Figure 4）
- **动量衰减系数**（momentum decay β 或 μ）越高 → 延迟容忍度越强。
- 解释：高动量使优化轨迹更依赖历史信息，降低对当前延迟梯度的敏感性。

#### （3）Batch Size 影响（Appendix A.2）
- 小 batch size 可减小 sync-async gap，但会牺牲绝对性能。
- 大 batch size 虽加剧 gap，但更利于硬件利用率。
- 本文选择 near-optimal batch size 进行公平比较。

#### （4）PipeDream vs PipeDream-2BW（Figure 6）
- 当 pipeline stage 数增加（P=16）时，原始 PipeDream（variable delay）性能急剧下降。
- PipeDream-2BW（fixed delay）保持稳定，表明**固定延迟至关重要**。

#### （5）下游任务表现（Table 4）
| Setup | MMLU | HellaSwag | PIQA | ... | 平均表现 |
|-------|------|-----------|------|-----|--------|
| Sync | 0.411 | 0.670 | 0.775 | ... | **等效** |
| Async + EF | 0.407 | 0.673 | 0.778 | ... | **无统计差异** |

➡️ 验证损失一致 ⇒ 下游能力一致，证明方法有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **一步梯度延迟不是根本障碍**：  
   在合适的 optimizer 和调度策略下，异步 PP 可以实现与同步训练**完全相同的最终性能**。

2. **optimizer 选择决定成败**：  
   - **AdamW 对延迟极度敏感**，不应直接用于异步训练。  
   - **Muon、Adan、SOAP 等现代 optimizer 天然鲁棒**，适合异步场景。

3. **Error Feedback 是有效的通用修正机制**：  
   轻量级、优化器无关，可进一步缩小甚至消除 sync-async gap。

4. **固定延迟优于可变延迟**：  
   **PipeDream-2BW** 提供的恒定一步延迟比原始 PipeDream 更稳定，尤其在深 pipeline 场景下优势明显。

5. **大规模可行性已验证**：  
   成功在 **10B 参数 MoE 模型**上实现无损异步训练，是目前最大规模的成功案例。

---

### ⚠️ 局限性

1. **缺乏对 AdamW 退化机制的完整解释**：  
   虽观察到第一矩（first-moment）更新是关键，但为何其在延迟下失配尚无完整理论。

2. **batch size 与 lr 的网格搜索有限**：  
   主要在 135M 模型上进行，更大模型的超参敏感性未充分探索。

3. **未涵盖所有调度策略**：  
   如 **WPipe** 虽被讨论，但未作为主实验展开。

4. **仅限于 pretraining**：  
   fine-tuning 或 instruction tuning 场景尚未验证。

---

### 🔮 未来工作方向

1. **深入理解 optimizer 动力学**：  
   探索为何某些 optimizer（如 Muon）天然抗延迟，建立更系统的理论框架。

2. **扩展至其他并行范式**：  
   将 EF 思想应用于 Data Parallel 或 Tensor Parallel 中的通信延迟补偿。

3. **结合 WPipe 等更优调度**：  
   将 PipeDream-2BW 替换为 WPipe，可能进一步减少延迟影响范围。

4. **探索自适应 EF 系数**：  
   动态调整 EF 的强度 λ，而非固定为 1。

5. **端到端吞吐量实测**：  
   当前仅提供理论 bubble 分析，未来需真实集群上的 wall-clock 时间测量。

---

## ✅ 总结一句话

> **通过选用鲁棒的现代 optimizer（如 Muon）并辅以 Error Feedback 修正，一步梯度延迟不再是异步流水线并行 LLM 预训练的性能瓶颈，可在 10B 规模上实现与同步训练完全等效的结果。**

</details>

---

### 10. [PromptGNN-sim: Deep Fusion and Alignment of GNN and LLMs for Text-Attributed Graph Learning](https://arxiv.org/abs/2606.30291)

**Authors**: Zhifei Hu, Alexandra I. Cristea  
**Category**: cs.AI  
**Published**: 2026-06-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.30291v1  

#### Abstract
Text-Attributed Graphs (TAGs) combine textual semantics with graph structure and are central to many graph learning tasks. However, existing fusion methods often treat text and structure as separate inputs in a shallow, one-way pipeline, which limits deep interaction between modalities and weakens p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PromptGNN-sim: Deep Fusion and Alignment of GNN and LLMs for Text-Attributed Graph Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前在 **Text-Attributed Graphs (TAGs)** 上的图学习方法存在以下关键问题：
- **浅层融合（shallow fusion）**：大多数方法将文本模态（textual modality）和结构模态（structural modality）作为独立输入，通过简单的拼接（concatenation）或后期融合（late fusion）处理，缺乏深层次的交互。
- **单向信息流**：信息通常从结构到文本或反之，无法实现双向协同优化。
- **邻域聚合忽略语义相似性**：传统 GNN 仅基于拓扑连接选择邻居，忽略了语义相关性，导致在稀疏图中表现不佳。
- **鲁棒性和泛化能力差**：面对噪声、稀疏连接、跨领域迁移等现实挑战时，模型性能显著下降。

### **提出的新方法与创新思路**
作者提出了 **PromptGNN-sim**，一种全新的 **双向结构-语义融合框架**，实现了 GNN 与 LLM 的深度协作。其核心创新包括：

#### ✅ **动态提示机制（Dynamic Prompting Mechanism）**
- 利用 **GAT** 提取结构注意力，并结合 **文本相似度（cosine similarity）** 进行语义感知的邻居筛选。
- 动态生成富含上下文的自然语言提示（prompt），包含：
  - 节点摘要（title + abstract）
  - 邻居关键词（representative keywords）
  - 图结构信息（degree, key neighbors）
  - 任务指令（如分类类别）
- 提示内容根据节点文本长度和连接度自适应调整（如短文本更依赖结构，长文本侧重内容）。

#### ✅ **双向跨模态注意力（Bi-directional Cross-modal Attention）**
- 设计双路注意力模块：
  - **Text-guided Graph Attention**：以文本为 Query，聚合图结构信息。
  - **Graph-guided Text Attention**：以图结构为 Query，引导文本特征提取。
- 实现模态间的信息互导，增强表示学习。

#### ✅ **对比学习对齐（Contrastive Learning for Prompt-Text Alignment）**
- 引入多视角对比损失（multi-view contrastive objective），对齐原始文本（raw text）与 LLM 生成的结构化摘要（structured summary）。
- 使用 **memory queues** 扩展负样本池，提升语义一致性与鲁棒性。

#### ✅ **联合优化训练策略**
- 整体框架端到端可训练，通过联合优化 GNN 和 LLM 组件，实现模态间的协同进化。

### **相比现有方法的优势**
| 方面 | 传统方法 | PromptGNN-sim |
|------|--------|-------------|
| 融合方式 | 浅层、单向 | 深层、双向 |
| 邻居选择 | 仅结构连接 | 结构 + 语义相似性 |
| 信息交互 | 弱或无 | 显式跨模态注意力 |
| 表示对齐 | 无显式约束 | 对比学习强制对齐 |
| 泛化与鲁棒性 | 差 | 显著更强 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在 **6 个公开的真实世界 TAG 数据集**上进行实验：

| 数据集 | 类型 | 节点数 | 边数 | 类别数 | 描述 |
|-------|------|--------|--------|--------|------|
| **Cora** | Citation | 2,708 | 5,429 | 7 | 机器学习论文，含标题和摘要 |
| **Citeseer** | Citation | 3,186 | 4,277 | 6 | 计算机科学文献 |
| **PubMed** | Citation | 19,717 | 44,338 | 3 | 医学研究论文 |
| **WikiCS** | Knowledge Graph | 11,701 | 215,863 | 10 | Wikipedia 计算机科学文章 |
| **History** | E-commerce | 41,551 | 358,574 | 12 | Amazon 图书共购网络 |
| **Photo** | E-commerce | 48,362 | 500,928 | 12 | Amazon 电子产品评论网络 |

### **实验设置与评估指标**

#### **任务**
- **Node Classification**：准确率（Accuracy）、Macro-F1
- **Link Prediction**：AUC、AP（Average Precision）、F1、Accuracy

#### **训练细节**
- 使用 **PyTorch + PyG** 实现
- 主要硬件：NVIDIA A100 80GB / RTX 4070 Ti SUPER
- LLM 后端：Llama3.1-8B / Llama3B / GPT-4o（用于 prompt 生成）
- 优化器：AdamW
- 超参数见附录 Table 10（如 lr=1e-5, batch_size=128, temperature=0.07）

### **基线方法对比**
分为三类进行比较：

#### **GNN 基线**
- GCN, GAT, GraphSAGE, NodeFormer, GLNN, GraphCL, Graphormer

#### **纯 LLM / LM 基线**
- BERT, RoBERTa, Sentence-BERT

#### **SOTA GNN-LLM 融合方法**
- ENGINE, ULTRATAG-S, OFA, PromptGFM (Flan-T5 / Llama3), GraphPrompter

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1 & 2）**

#### 📊 **Node Classification Accuracy (%)**

| Model | Cora | PubMed | Citeseer | WikiCS | History | Photo |
|-------|------|--------|----------|--------|---------|-------|
| GCN | 85.23 | 82.98 | 73.51 | 80.52 | 77.70 | 77.39 |
| BERT | 72.27 | 90.25 | 75.65 | 80.35 | 83.45 | 73.93 |
| PromptGFM (Llama3) | 92.42 | 94.65 | 85.32 | 84.66 | 86.72 | 86.61 |
| **PromptGNN-sim (Ours)** | **90.59** | **94.12** | **82.76** | **85.82** | **84.97** | **87.20** |

> 💡 在 **WikiCS** 和 **Photo** 上大幅领先所有基线；在 Cora 上接近最优且更稳定。

#### 🔗 **Link Prediction Accuracy (%)**

| Model | Cora | PubMed | Citeseer | WikiCS | History | Photo |
|-------|------|--------|----------|--------|---------|-------|
| GCN | 77.65 | 76.08 | 77.87 | 78.40 | 80.54 | 79.13 |
| BERT | 60.53 | 89.54 | 93.19 | 89.32 | 92.98 | 89.37 |
| **PromptGNN-sim (Llama3B)** | **87.54** | **88.30** | **86.27** | **89.93** | **89.57** | **89.55** |

> ✅ 在所有数据集上取得 **SOTA 性能**，尤其在 Cora 上比最强 GNN 提升 **近 10%**。

---

### **与基线方法的对比结果**
- **优于经典 GNNs**（如 GCN、GAT）：得益于 LLM 强大的语义理解能力。
- **优于纯 LLM 方法**（如 BERT）：因引入了图结构信息，在稀疏图中更具鲁棒性。
- **优于近期 SOTA 融合方法**（如 PromptGFM、GraphPrompter）：
  - 在 **WikiCS** 和 **Photo** 上明显胜出；
  - 在跨任务/跨领域场景下泛化更强。

---

### **消融实验结果（Ablation Study）**

#### 🔍 **组件有效性分析（Table 5 & 16）**

| 模型变体 | Cora Acc | PubMed Acc | Citeseer Acc | WikiCS Acc |
|---------|----------|------------|---------------|-------------|
| Full Model (Ours) | 90.59 | 94.12 | 82.76 | 85.82 |
| w/o Cross Attention | 89.48 | 94.02 | 80.41 | 85.65 |
| w/o Contrastive Learning | 89.11 | 93.74 | 81.35 | 84.96 |
| w/o Warmup | 88.01 | 93.76 | 79.31 | 86.16 |
| w/o TF-IDF | 88.93 | 93.03 | 79.94 | 85.69 |

> ⚠️ 移除 **cross-attention** 或 **contrastive learning** 导致最大性能下降，说明二者是实现深度对齐的关键。

#### 🔬 **提示设计影响（Table 9）**
在 Citeseer 上测试不同 prompt 格式：
- “Abstract First” vs “Title First” vs “Our Prompt”
- **Our Prompt（含结构+邻居信息）** 达到最高 Acc (82.76) 和 F1 (77.92)，验证了丰富上下文的重要性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **深层双向融合优于浅层单向融合**：PromptGNN-sim 通过动态 prompt + cross-attention + contrastive learning 实现了 GNN 与 LLM 的真正协同。
2. **结构与语义应共同指导邻居选择**：结合 GAT 注意力与文本相似度，提升了邻域质量，尤其在稀疏图中效果显著。
3. **提示工程极大增强模型表现**：精心设计的 prompt 可有效引导 LLM 利用图上下文，提升分类与链接预测能力。
4. **高鲁棒性与强泛化能力**：
   - 在 **边/节点/文本删除** 的扰动下仍保持高性能（Fig. 2–3, Tables 12–15）；
   - 支持 **zero-shot / few-shot 跨领域迁移**（Table 4），表明学到的表示具有通用性。

### **方法的局限性**
- **计算开销较大**：依赖大参数量 LLM（如 Llama3B），推理成本高于轻量级 GNN。
- **依赖高质量 prompt 设计**：性能受 prompt 模板影响，需人工调优。
- **目前为 transductive 设置**：尚未支持完全归纳式学习（inductive learning）。
- **对极端稀疏图仍有挑战**：当图极度稀疏且文本质量低时，性能可能受限。

### **未来工作方向**
1. **扩展至归纳学习场景**：支持新图上的零样本推理。
2. **支持多语言与多模态输入**：融合图像、表格等异构信息。
3. **轻量化与高效推理**：探索蒸馏、适配器（adapter）等技术降低部署成本。
4. **构建统一的 TAG Foundation Model**：推动 PromptGNN-sim 成为通用图学习基础架构。

---

> ✅ **总结一句话**：  
> **PromptGNN-sim 通过“动态提示 + 双向注意力 + 对比对齐”三重机制，首次实现了 GNN 与 LLM 在 TAG 上的深度、双向、可训练融合，在准确性、鲁棒性与泛化性方面全面超越现有方法，为未来多模态图学习提供了新范式。**

</details>

---

### 11. [Morphing into Hybrid Attention Models](https://arxiv.org/abs/2606.30562)

**Authors**: Disen Lan, Jianbin Zheng, Yuxi Ren, Xin Xia, Xuanda Wang, Xuefeng Xiao, Xipeng Qiu, Yu Cheng  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.30562v1  

#### Abstract
Hybrid attention models improve long-context efficiency by retaining only a subset of full-attention layers and replacing the remaining layers with linear attention. However, the effectiveness of Transformer-to-hybrid conversion critically depends on which layers preserve full attention. Existing hy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Morphing into Hybrid Attention Models

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于 **Transformer-to-hybrid Conversion**（将预训练的 Transformer 模型转换为混合注意力模型）中的一个核心挑战：如何在有限的 **full-attention 层预算**下，选择最优的层来保留 full attention，其余层则替换为更高效的 **linear attention**。

现有方法通常采用启发式策略，如：
- **Uniform Interleaving**：固定间隔地保留 full-attention 层。
- **Layerwise Scoring**：逐层评估重要性并打分（如 KL-LS, HALO）。

这些方法存在两大缺陷：
1. **孤立评估**（Isolated Evaluation）：忽略了层与层之间的相互依赖关系（inter-layer dependency），即多个层共同作用时的效果无法通过单个层的重要性简单叠加得出。
2. **效率低下**：需要反复进行单层扰动、恢复或微调，计算开销巨大。

### 提出的新方法：FlashMorph
作者提出了 **FlashMorph**（Fast LAyer Selection for Hybrid MORPHing），一种高效、可扩展且效果优越的层选择方法。其核心思想是将混合层选择问题重新定义为一个**带预算约束的联合优化问题**（budget-constrained joint optimization problem）。

#### 创新点
1. **构建可变形模型（Morphable Model）**：
   - 为每个预训练的 full-attention 层配备一个对应的、已通过 **hidden-state alignment** 训练好的 linear-attention 分支。
   - 这样，每一层都可以看作是一个“可变形”模块，能够在 full 和 linear attention 之间平滑切换。

2. **联合优化层门控（Joint Optimization of Layerwise Gates）**：
   - 引入一个可学习的标量门控 `α(l) ∈ [0,1]` 来控制第 `l` 层对 full-attention 分支的依赖程度。
   - 在层选择阶段，**冻结所有原始模型权重和 linear-attention 分支的权重**，只优化这 `L` 个门控值 `α`。
   - 优化目标由两部分组成：
     - **对齐损失（Alignment Loss, `L_align`）**：确保变形后模型的隐藏状态与原始 full-attention 教师模型对齐。
     - **线性化正则项（Linearization Regularization, `L_reg`）**：鼓励门控值 `α` 尽可能小（即倾向于使用更高效的 linear attention），以提升模型效率。

3. **全局配置下的非孤立选择**：
   - 所有门控值是**同时联合优化**的，这使得 FlashMorph 能够捕捉到层间的互补性（complementarity）和冗余性（redundancy），从而选出在全局混合配置下最有效的 full-attention 层组合。

### 相比现有方法的优势
- **有效性更高**：通过联合优化，选出的层组合能更好地平衡性能与效率。
- **效率极高**：层选择成本（Selection Cost）远低于现有方法。例如，在 Qwen3-1.7B 上，FlashMorph 仅需 **20M tokens** 和 **2.1 GPU hours**，而 KL-LS 需要 20B tokens 和 1071.8 GPU hours，PostNAS 更高达 50B tokens 和 2561.3 GPU hours。
- **可扩展性强**：其轻量级的优化过程使其能够轻松扩展到更大的模型。

---

## 2. 核心实验方法和设置

### 数据集
1. **合成检索数据集（Synthetic Retrieval Dataset）**：
   - 用于 **层选择阶段**（Layer Selection）。
   - 基于 DCLM 语料库构建长上下文文档，并在其中插入随机生成的 **passkeys**。
   - 模型任务是在文档末尾准确回忆出所有 passkeys。
   - 此设计专门用于测试模型的 **长距离信息访问能力**，为层选择提供强监督信号。

2. **通用语言建模数据集**：
   - 用于后续的 **logits distillation** 和 **long-context finetuning** 阶段。
   - 同样基于 DCLM 语料库。

### 评估指标
1. **NIAH (Needle-in-a-Haystack)**：
   - 衡量模型在超长上下文中精确检索关键信息的能力。
   - 报告不同上下文长度（32K, 64K, ..., 256K）下的召回率。

2. **零样本推理任务**：
   - **常识推理**：ARC-e, ARC-c, PIQA, HellaSwag, WinoGrande。
   - **真实世界回忆密集型任务**：SQuAD, FDA, SWDE。
   - 报告各任务的准确率（acc）及平均得分。

3. **效率指标**：
   - **推理延迟**（Latency Time）：Prefill 和 Decode 阶段的时间。
   - **GPU 内存占用**（Peak GPU Memory）。
   - **层选择成本**：所需 tokens 数量、FLOPs 和 GPU hours。

### 基线方法对比
- **Uniform**：均匀交错模式。
- **KL-LS**：基于 KL 散度的逐层重要性评分。
- **HALO**：基于逐层替换和评估的层选择方法。
- **PostNAS**：基于超网搜索的方法（作为高成本的上界参考）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### NIAH 检索性能（Table 2）
- 在 **Qwen3-1.7B** 模型上，FlashMorph 在 **NIAH-Single-1** 任务上实现了 **100% 的完美召回率**，即使在 256K 的超长上下文中也保持不变。
- 在更具挑战性的 **NIAH-Single-2** 和 **NIAH-Single-3** 任务上，FlashMorph 显著优于其他方法，尤其是在长上下文场景下。

#### 零样本任务性能（Table 3）
- **常识推理**：FlashMorph 保持了与原始 Qwen3 模型相当甚至更好的性能，平均得分接近或达到最佳水平。
- **回忆密集型任务**：优势更为明显。例如，在 1.7B 模型上，FlashMorph 在 GDN 架构下的回忆平均得分达到了 **70.2**，超过了 PostNAS 的 69.6。

#### 效率结果（Table 4, Figure 2, Figure 3）
- **层选择成本**：
  | 方法 | Tokens | GPU Hours |
  |---|---|---|
  | PostNAS | 50B | 2561.3 |
  | KL-LS | 20B | 1071.8 |
  | HALO | 234M | 15.4 |
  | **FlashMorph (Ours)** | **20M** | **2.1** |

- **推理效率**（Figure 2）：
  - **Prefill**：在 256K 上下文长度下，FlashMorph 达到 **2.81×** 的速度提升。
  - **Decode**：在 512K 解码长度下，达到 **2.07×** 的速度提升，并且内存占用显著降低，避免了 Out-of-Memory (OOM) 错误。

### 与基线方法的对比结果
- **性能更强**：在 NIAH 和回忆密集型任务上，FlashMorph 普遍优于或持平于所有基线方法。
- **成本极低**：其层选择成本比 KL-LS 低 **1000倍**，比 PostNAS 低 **2500倍**。
- **效率更高**：推理速度更快，内存占用更少。

### 消融实验结果（Figure 4）
1. **不同混合比例下的鲁棒性**：
   - 在 6:1, 3:1, 1:1 (linear:full) 等多种混合比例下，FlashMorph 均表现优异。
   - 特别是在 **6:1**（即 full-attention 层极少）的极端情况下，其优势最为突出，证明了其在稀疏预算下选择关键层的强大能力。

2. **监督信号的影响**：
   - 使用 **合成检索数据**（FlashMorph w/ syn）作为监督信号，比使用 **通用语言建模数据**（FlashMorph w/ lm）效果更好。
   - 例如，在 GDN 架构上，RULER 得分从 61.6 提升至 64.7，验证了**面向检索的监督信号**对于识别关键 full-attention 层的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. **联合优化优于孤立评估**：FlashMorph 证明了将层选择视为一个全局的联合优化问题，而非孤立的逐层打分，能够发现更有效、更鲁棒的混合架构。
2. **效率与性能可以兼得**：通过精心设计的层选择，可以在大幅降低计算和内存开销的同时，几乎无损地保留原始 Transformer 模型在长上下文检索和复杂推理上的强大能力。
3. **合成任务的有效性**：专为长上下文检索设计的合成数据（如 passkey retrieval）是指导层选择的理想监督信号。

### 方法的局限性
- **依赖于高质量的 linear-attention 替代分支**：FlashMorph 的效果建立在第一步成功训练出能模仿 full-attention 表现的 linear-attention 分支的基础上。如果这一步失败，整个流程会受影响。
- **门控离散化的近似性**：连续的门控优化后需要通过 Top-K 离散化得到最终的二元选择（full 或 linear），这个过程本身是一种近似，可能会丢失一些细微的优化信息。

### 未来工作方向
- 探索更复杂的门控机制或端到端的联合训练框架。
- 将 FlashMorph 的思想应用于其他类型的模型压缩或架构搜索任务。
- 研究如何动态调整混合配置以适应不同的输入长度或任务需求。

</details>

---

### 12. [Understanding Evaluation Illusion in Diffusion Large Language Models](https://arxiv.org/abs/2606.29228)

**Authors**: Hengxiang Zhang, Jiaxi Ren, Hongxin Wei  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.29228v1  

#### Abstract
Despite the capability of parallel decoding, diffusion large language models (dLLMs) require many denoising steps to maintain generation quality, motivating recent research on efficient decoding strategies. However, existing studies have reported inconsistent evaluation results even under seemingly ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Understanding Evaluation Illusion in Diffusion Large Language Models》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文揭示了当前在 **diffusion large language models (dLLMs)** 领域中广泛存在的 **evaluation illusion（评估错觉）** 问题。尽管已有研究声称某些并行解码方法（parallel decoding methods）可以在不牺牲生成质量的前提下显著提升推理效率，但这些结论在不同实验设置下存在高度不一致。

作者指出，这种不一致性源于对 **prompt template** 的敏感性以及对其他评估设置（如硬件平台、few-shot 设置等）的忽视，导致单一模板下的评估结果可能产生误导性的“性能提升”假象。

### **提出了什么新方法或新思路**
本论文并未提出新的解码算法，而是从**评估方法论**角度出发，系统性地分析了 dLLMs 解码方法评估中的可靠性问题，并提出了以下关键洞见和建议：

- **识别出“评估错觉”的根源**：并行解码方法对 prompt template 的微小变化极为敏感，导致其性能排名不稳定。
- **强调多模板评估的重要性**：主张必须在多个 prompt templates 下进行评估，以获得可靠结论。
- **提出可复现评估的实践指南**：包括跨硬件平台的一致性控制、详细报告评估配置等。

### **相比现有方法的优势**
- **方法论上的突破**：不同于以往专注于改进解码效率的研究，本文聚焦于**评估本身的可信度**，填补了当前 dLLM 研究中被忽视的关键环节。
- **实证驱动的批判性视角**：通过大量严谨实验挑战了“并行解码优于单 token 解码”的主流观点，促使社区重新审视已有成果。
- **普适性强的指导原则**：所提出的评估规范不仅适用于 dLLMs，也可推广至其他 LLM 解码策略的评估中。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **数学推理任务**：
  - `GSM8K`
  - `MATH500`
- **代码生成任务**：
  - `HumanEval`
  - `MBPP`

所有实验均基于两个主流 dLLM 模型：
- `LLaDA-8B-Instruct`
- `LLaDA-1.5`

### **实验设置和评估指标**
#### **评估指标**
- 主要指标为 **accuracy**（准确率），用于衡量模型输出是否正确解决数学题或通过代码测试用例。
- 使用 **Kendall's tau correlation** 衡量不同 prompt templates 下方法排名的一致性，量化“评估不一致性”。

#### **关键变量控制**
- 统一 generation length 为 128。
- 批大小（batch size）设为 1，避免批处理引入偏差。
- 在单台机器上运行以消除系统环境差异。
- 对比多种 prompt templates（共 8 种），涵盖近似相同语义但措辞不同的版本。

#### **评估的解码方法**
| 方法 | 类型 | 简介 |
|------|------|------|
| **Vanilla (Low-confidence)** | 单 token 解码基线 | 每步仅解码置信度最高的一个 token |
| **Fast-dLLM** | 并行 + KV Cache | 块级 KV 缓存 + 高置信度 token 并行解码 |
| **AdaBlock-dLLM** | 自适应块大小 | 动态调整每次解码的 token 数量 |
| **AdaBlock-Fast** | 结合 Fast-dLLM 与 AdaBlock | 融合两种加速机制 |
| **dKV-Cache-Decode** | 条件延迟缓存 | 控制 KV 缓存重用时机 |
| **Elastic-Cache** | 自适应缓存刷新 | 根据注意力漂移决定何时刷新缓存 |
| **WINO** | Draft-and-Verify 机制 | 先宽泛解码再验证保留有效 token |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **表1 & 表2：数学任务上的性能对比（GSM8K / MATH500）**
- 在 `LLaDA-8B-Instruct` 上：
  - Vanilla 基线平均准确率为 **77.64% (GSM8K)** 和 **65.43% (MATH500)**。
  - 多数并行方法在多数模板下表现**低于基线**，例如：
    - `Top-K (k=4)`：平均仅 **66.02%**，远低于 Vanilla。
    - `dKV-Cache-Decode`：平均下降约 **3.73%**。
- 在 `LLaDA-1.5` 上也观察到类似趋势，表明结果具有模型泛化性。

#### **Kendall's tau 排名一致性分析**
- 不同 prompt templates 之间的 Kendall's tau 平均值仅为：
  - `LLaDA-8B-Instruct`: **0.539**
  - `LLaDA-1.5`: **0.495**
- 表明方法排名在不同模板间**高度不稳定**，支持“评估错觉”假设。

### **与基线方法的对比结果**
- **并行解码方法普遍未能超越 Vanilla 基线**：
  - 尽管部分方法在特定模板上有轻微提升（如 Elastic-Cache 在 Template 3 中略胜），但在整体平均上始终落后。
  - 特别是依赖 KV Cache 的方法（如 Fast-dLLM）常因缓存误差导致性能下降。
- **结论颠覆主流认知**：当前所谓“高效无损”的并行解码方案实际上**未能真正克服速度-质量权衡（speed-quality trade-off）**。

### **消融实验结果**
#### **(1) Prompt Template 敏感性分析（Table 3）**
- 并行解码方法的标准差（Std）和范围（Range）显著高于 Vanilla：
  - `Top-K (k=4)`：Std = **1.142**, Range = **3.79**
  - `Vanilla`：Std = **0.355**, Range = **1.07**
- 说明并行解码更容易受 prompt 微小变动影响，稳定性差。

#### **(2) Denoising Steps vs Prompt Design 影响力比较（Figure 4）**
- 一个设计良好的 prompt template（如 Template H）即使使用低阈值（γ=0.6）、仅 25 步去噪，也能达到 **74% 准确率**。
- 而较差的 template（如 G）即使使用完整 128 步，准确率仍只有 **68.16%**。
- **结论**：prompt design 的影响力 > 增加 denoising steps 的边际收益。

#### **(3) Few-shot 设置的影响（Table 4）**
- 并行方法在 4-shot 设置下表现相对更好，可能造成“性能优越”的错觉。
- 但在 0-shot 或 2-shot 下，其性能普遍劣于 Vanilla。
- 表明 few-shot 设置会放大并行方法的虚假优势。

#### **(4) 硬件平台差异（Table 8 & 9）**
- 在 BF16 精度下，不同 GPU（L40, A100, RTX 4090）之间出现高达 **1.06%** 的绝对准确率偏差。
- 改用 FP32 后，差异几乎消失（最大偏差 0.00%），证明**数值精度是评估不可靠的重要来源**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **评估错觉普遍存在**：dLLM 解码方法的性能排名在不同 prompt templates 下剧烈波动，单一模板评估极易产生误导性结论。
2. ❌ **并行解码未真正突破瓶颈**：当前主流并行解码方法在综合评估下**持续劣于单 token 解码基线（Vanilla）**，未能实现真正的“高效无损”。
3. 🔍 **prompt template 是主导因素**：其设计对最终性能的影响远超增加 denoising steps 或优化解码策略本身。
4. ⚠️ **其他设置亦不可忽略**：
   - **few-shot 示例数量**会影响并行方法的表现评估；
   - **generation length** 并非越长越好，需匹配 prompt 设计；
   - **硬件平台与数值精度**会导致非预期的结果差异。

### **方法的局限性**
- 本研究聚焦于评估方法论，**未提出新的高性能解码算法**。
- 实验集中在公开可用的 dLLM 模型（LLaDA系列），尚未覆盖所有潜在架构。
- prompt templates 虽多样，但仍属人工构造，未来可探索更大规模自动化采样。

### **未来工作方向**
1. **建立标准化评估框架**：推动社区采用多 prompt、多 seed、统一硬件的评估协议。
2. **开发更鲁棒的解码方法**：设计对 prompt 变化不敏感的并行解码策略。
3. **研究 prompt engineering 对 dLLMs 的特殊影响机制**：为何微小改动会引起巨大性能波动？
4. **构建专用 benchmark suite**：专门用于测试 dLLM 解码方法的稳定性和公平性。

---

> 📌 **一句话总结**：  
> 本文揭示了 dLLMs 解码研究中的“皇帝的新衣”现象——许多看似高效的并行解码方法其实只是在特定 prompt 下的幻象；唯有通过多维度、透明化的评估，才能看清真实性能边界。

</details>

---

### 13. [MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training](https://arxiv.org/abs/2606.30406)

**Authors**: Wenhan Ma, Jianyu Wei, Liang Zhao, Hailin Zhang, Bangjun Xiao, Lei Li, Qibin Yang, Bofei Gao, Yudong Wang, Rang Li, Jinhao Dong, Zhifang Sui, Fuli Luo  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.30406v1  

#### Abstract
Modern large language models (LLMs) rely on reinforcement learning during post-training to push specific capabilities, yet integrating multiple capabilities into one model remains hard. Existing methods, such as Off-Policy Finetune and Mix-RL, are either inefficient or lose performance. In this work...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代大语言模型（LLMs）在**post-training**阶段广泛使用强化学习（RL）来提升特定能力（如数学推理、代码生成、指令遵循等）。然而，如何将多个领域专家模型（domain-specific teachers）的能力有效整合到一个统一的学生模型中，仍然是一个开放挑战。

现有方法存在以下问题：
- **Mix-RL**：联合训练时存在跨域干扰（see-saw effect），导致某些领域性能下降。
- **Cascade RL**：顺序训练易引发灾难性遗忘，且训练周期长、不稳定。
- **Off-Policy Finetune**：基于教师模型的离线轨迹进行SFT，引入**exposure bias**。
- **Param-Merge**：权重空间融合常导致模型不稳定，难以同时继承所有教师的能力。

### 提出了什么新方法或新思路
本文提出 **MOPD (Multi-Teacher On-Policy Distillation)**，一种新的多教师在线策略蒸馏范式，用于LLM后训练中的能力集成。

其核心思想是：
1. **Stage 1**: 先对基础模型进行通用SFT，得到共享起点。
2. **Stage 2**: 在各任务域上独立并行地进行RL训练，获得多个**domain-specialized RL teachers**。
3. **Stage 3**: 将学生模型从SFT检查点初始化，并通过**on-policy distillation**方式，利用每个prompt对应的领域教师在其自身rollout上的token-level输出进行监督学习。

关键机制：
- 学生生成序列 → 路由至对应领域的冻结教师 → 教师prefill该序列并提供每token的log-probabilities → 最小化学生与教师之间的**per-token reverse KL**损失。

### 相比现有方法的优势
| 特性 | MOPD | Mix-RL | Cascade RL | Off-Policy FT | Param-Merge |
|------|------|--------|------------|----------------|--------------|
| ✅ Dense optimization | ✔️ | ❌ | ❌ | ❌ | ❌ |
| ✅ On-policy training | ✔️ | ❌ | ❌ | ❌ | ❌ |
| ✅ Parallelizable pipeline | ✔️ | ❌ | ❌ | ✔️ | ✔️ |
| ✅ 无暴露偏差（No exposure bias） | ✔️ | ❌ | ❌ | ❌ | ❌ |
| ✅ 高样本效率 | ✔️ | ❌ | ❌ | ❌ | ❌ |

MOPD 是唯一同时满足 **dense optimization**、**on-policy training** 和 **parallelisable pipeline** 三个关键工程属性的方法。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖三大任务领域：

| 领域 | SFT 数据来源 | RL 数据来源 | 评估基准 |
|------|-------------|------------|----------|
| **Math** | Mixture-of-Thoughts 数学子集 | BigMath, ORZ | AIME25, AIME26 |
| **Instruction Following (IF)** | IFBench 构造 + gpt-oss-120b 蒸馏 | IFBench 合成数据 | IFBench, IFEval |
| **Software Engineering (SWE)** | R2E-Gym 蒸馏 | R2E-Gym-Lite | SWE-bench Verified |

最大序列长度分别为 32K（Math/IF）、65K（SWE），支持长上下文建模。

### 实验设置和评估指标

#### 模型架构
- 主要实验：`Qwen3-30B-A3B`
- 工业级验证：`MiMo-V2-Flash`（309B参数）

#### 评估指标：**Normalized Score**
由于不同领域绝对性能上限差异大，采用归一化得分：
$$
\text{Norm. Score}_d = \frac{s_d - s_{\text{SFT}}}{s_{\text{Teacher},d} - s_{\text{SFT}}}
$$
其中 $s_d$ 是某方法在域 $d$ 上的表现。最终报告的是各域平均值 $\bar{s} = \frac{1}{|D|}\sum_d \text{Norm. Score}_d$

- 若 >1：超越教师表现
- 若 <0：退化回SFT以下

#### 基线方法对比
共比较五种capability integration范式：
1. **Mix-RL**：混合所有领域数据进行联合RL训练
2. **Cascade RL**：按顺序依次训练（IF → Math → SWE）
3. **Off-Policy Finetune**：用教师rollouts做SFT
4. **Param-Merge (Avg./Task Arith.)**：权重平均或任务向量加减
5. **MOPD (Ours)**

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen3-30B-A3B）

| Method | AIME25 | AIME26 | IFBench | IFEval | SWE-bench V. | **Norm. Score** |
|--------|--------|--------|---------|--------|---------------|------------------|
| SFT Only | 45.42 | 54.48 | 42.69 | 84.17 | 35.80 | 0.000 |
| RL Teacher | 54.79 | 63.65 | 78.40 | 95.50 | 51.20 | 1.000 |
| Mix-RL | 52.71 | 63.75 | 75.00 | 94.58 | 48.80 | 0.882 |
| Cascade RL | 48.54 | 61.88 | 77.11 | 95.80 | 47.80 | 0.775 |
| Off-Policy FT | 51.56 | 63.44 | 80.95 | 93.35 | 45.80 | 0.824 |
| Param-Merge (Task Arith.) | 49.38 | 63.96 | 78.23 | 95.81 | 48.80 | 0.857 |
| **MOPD (Ours)** | **51.46** | **65.31** | **77.89** | **93.84** | **50.40** | **0.937** |

✅ **MOPD以0.937的归一化得分显著领先第二名Mix-RL（0.882），领先达+5.5个百分点**

### 与基线方法的对比结果
- **优于Mix-RL**：尽管更平衡，但MOPD仍全面胜出，尤其在Math领域实现反超。
- **优于Cascade RL**：避免了“先训好后忘”的问题，在Math阶段未出现明显衰退。
- **优于Off-Policy FT**：虽在IF上略优，但在SWE上严重不足（仅65% headroom closed）。
- **远优于Param-Merge**：线性平均失败严重（0.328），任务算术依赖调参，不可靠。

### 消融实验结果

#### （1）Top-k vs Policy Gradient 实现形式
| Loss Variant | Norm. Score |
|--------------|-------------|
| Policy Gradient | 0.937 |
| Top-k (k=64) | 0.909 |

👉 两者性能接近，说明在same-origin teacher设定下，policy gradient已足够稳定。

#### （2）Same-Origin Teachers 的重要性
替换Math教师为更强但分布不同的 `Qwen3-235B-A22B`：
- Policy Gradient：Norm. Score降至0.600
- Top-k：训练崩溃，Score为-1.19 ❌

📌 初始KL divergence从~0.04升至~0.19，证明**distributional mismatch**会导致优化不稳定。

#### （3）Multi-round Evolution 实验
进行两轮MOPD迭代：
- Iter 1 MOPD → 得到新Student
- 以此Student为起点重训Math/IF教师 → 再次MOPD

结果：
| Round | Norm. Score |
|-------|-------------|
| Iter 1 MOPD | 0.937 |
| Iter 2 MOPD | 0.986 (+0.049) |

✅ 表明MOPD可支持**持续迭代增强**，吸收更强教师的知识。

### 工业规模验证（MiMo-V2-Flash, 309B）

| Benchmark | Student | Teacher | MOPD | Δ(MOPD-Teacher) |
|---------|--------|--------|------|----------------|
| AIME25 | 89.3 | 93.9 | 94.1 | +0.2 |
| HMMT25 | 76.9 | 82.6 | 84.4 | +1.8 |
| LCB | 77.5 | 82.6 | 83.2 | +0.6 |
| IFBench | 55.4 | 68.9 | 66.7 | -2.2 |
| SWE-Bench V. | 67.8 | 74.2 | 73.4 | -0.8 |
| T2-Bench | 75.9 | 79.6 | 80.3 | +0.7 |
| 2-Telecom | 92.7 | 95.0 | 95.3 | +0.3 |

✅ 多数指标持平或超过教师，仅有两个轻微回退，验证了MOPD在工业前沿模型上的可行性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **MOPD实现了高效、稳定的多领域能力集成**，在Qwen3-30B上达到0.937归一化得分，显著优于主流基线。
2. ✅ **on-policy distillation + same-origin teachers** 是稳定训练的关键，能有效消除exposure bias并保持低初始KL。
3. ✅ **模块化并行开发流程** 支持团队独立迭代各领域RL策略，提升研发吞吐量。
4. ✅ **支持multi-round evolution**，可通过反复蒸馏进一步逼近甚至超越教师性能。
5. ✅ 在**工业级309B模型MiMo-V2-Flash上成功部署**，具备实际应用价值。

### 方法的局限性
- 依赖高质量、同源（same-origin）的domain teachers，若教师来自异构架构或训练流程，可能导致训练不稳定。
- 需要额外部署teacher prefill服务，增加系统复杂度（尽管作者称延迟可忽略）。
- 当前路由逻辑基于prompt domain标签，尚未实现自动识别或动态路由。

### 未来工作方向
- 探索**automatic prompt routing**机制，减少人工标注依赖。
- 扩展至更多任务类型（如vision-language、agent planning等）。
- 研究**asynchronous teacher updates**与在线MOPD结合的可能性。
- 进一步优化top-k distillation在large divergence场景下的鲁棒性。

---

> 📌 **总结一句话**：  
> **MOPD通过“分域训练 + 在线策略蒸馏”的三阶段范式，解决了LLM多能力集成中的效率、稳定性与工程可扩展性难题，是当前最接近理想capability integration方案的工作之一。**

</details>

---

### 14. [Speculative Pre-Positioning: Decoding Stateful Sessions to the Next Decision Point Off the Critical Path](https://arxiv.org/abs/2606.29565)

**Authors**: Victor Norgren  
**Category**: cs.LG  
**Published**: 2026-06-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.29565v1  

#### Abstract
A stateless inference server (vLLM, SGLang, TensorRT-LLM) idles between requests while the accelerator waits; a stateful session reclaims that idle time. Speculative pre-positioning decodes the session forward to its next decision point with the target model's own forward pass and no draft model, mo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Speculative Pre-Positioning: Decoding Stateful Sessions to the Next Decision Point Off the Critical Path*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **stateless inference**（无状态推理）系统中，如 vLLM、SGLang 或 TensorRT-LLM，加速器在请求之间处于空闲状态，造成资源浪费。而 **stateful session**（有状态会话）虽然保持上下文状态以支持流式数据或 agent 工具调用等场景，但其“空窗期”——即数据更新后到下一个查询到来前，或工具调用发出后等待返回结果的时间——仍被闲置。

本文提出利用这些**跨请求的空闲时间窗口**（idle window），提前为下一次请求做准备，从而将关键路径上的延迟转移到非关键路径上进行摊销。

---

### 提出了什么新方法或新思路
作者提出了 **Speculative Pre-Positioning**（推测性预定位）机制，其核心思想是：

- 在 session 处于 idle window 时，使用目标模型（target model）自身执行一次 forward pass，解码至“下一个决策点”（next decision point），并缓存输出分布（output distribution）。
- “决策点”定义为生成恢复的位置：
  - 流式会话中：查询前缀 + assistant header 后的位置；
  - Agent 会话中：tool call 返回后的 envelope 开始处。
- 缓存的内容不仅是 KV Cache，还包括该位置的 **ready distribution**（就绪分布，即 logits 输出）。

引入两个优化路径：
1. **Fast Path（快速路径）**：当一个置信门控（confidence gate）触发时（top-1 与 top-2 logit gap 超过阈值且预测 token 属于 domain fast-answer set），直接从缓存中返回单个 token，无需任何 decode。
2. **Fall-Through Path（回退路径）**：若未触发 fast path，则已预填充 entry，只需处理增量部分（delta），节省 prefill 成本。

> ⚠️ 注意：此方法不依赖 draft model，也不需要验证步骤，而是基于 selective prediction 思想，通过校准的 confidence gate 控制风险。

---

### 相比现有方法的优势
| 方法 | 是否需要 draft model | 是否需要验证 | 是否移动 decode 到 idle 时间 | 是否缓存 output distribution |
|------|------------------------|---------------|-------------------------------|-------------------------------|
| Speculative Decoding [Leviathan et al., 2023] | ✅ 是 | ✅ 是 | ❌ 否（在 decode 内部 speculation） | ❌ 否 |
| Prompt Lookup Decoding [Saxena, 2023] | ❌ 否 | ✅ 是 | ❌ 否 | ❌ 否 |
| Prefix / KV Cache [Kwon et al., 2023] | ❌ 否 | ❌ 否 | ✅ 是（prefill 阶段） | ❌ 否（仅缓存 KV） |
| **Speculative Pre-Positioning（本文）** | ❌ 否 | ❌ 否 | ✅ 是（整个 entry decode 移出关键路径） | ✅ 是 |

**优势总结**：
- 将 **entry prefill 和 entry decode** 完全移出关键路径；
- 实现真正的 **zero-decode fast path**，仅需一次 vocabulary scan；
- 不增加正确性风险（invalidation discipline 保证一致性）；
- 支持多种 session 类型（streaming & agentic）；
- 能量成本可控，失败代价仅为 energy 浪费而非延迟上升。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
论文未使用公开 benchmark 数据集，而是构建了一个 **domain-specific binary market-trend task**（二元市场趋势判断任务），用于评估模型对流式金融数据的理解能力。

- 输入：连续时间序列数据块；
- 查询：询问当前趋势（上涨/下跌）；
- 输出：单 token 回答（属于预定义的 fast-answer set，共 31 个候选 token）；
- 场景模拟真实 streaming 与 agent tool-calling 的 idle window 特性。

---

### 实验设置和评估指标

#### 硬件平台
- 单张 **H100 GPU**
- 目标模型：**70B-class LLM @ 4-bit precision**（TP=1）
- 对比模型：**8B model @ BF16**

#### 主要评估指标
| 指标 | 描述 |
|------|------|
| `first-token latency`（P50/P99） | 首 token 延迟，衡量响应速度 |
| `L_cold` | 基线延迟：`T_prefill(L_entry) + T_decode` |
| `L_fast` | 快速路径延迟：`T_scan(V)`（词汇表扫描） |
| `L_fall` | 回退路径延迟：`T_restore + T_prefill(Δ) + T_decode` |
| `hit rate (h)` | 预定位命中率（pre-position 被消费的比例） |
| `gate coverage (c(T))` | confidence gate 触发比例 |
| `false-accept rate r(T)` | fast path 错误接受率（返回错误 token 的比例） |
| `energy per pre-position` | 每次预定位的能量消耗 |

#### 基线方法对比
- **Baseline (Cold Path)**：标准 stateful 推理，每次请求重新处理 entry；
- **Prefix/KV Cache**：缓存 KV 状态，避免重复 prefill，但仍需运行一次 decode 获取 first token；
- **本文方法（Speculative Pre-Positioning）**：缓存 ready distribution，实现 zero-decode fast path。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Section 5.2）

| 指标 | 数值 |
|------|------|
| `T_prefill per token` | 0.843 ms/token |
| `T_decode`（单步 decode） | 39.12 ms |
| `T_restore`（KV 状态恢复） | 14.47 ms |
| `L_fast`（fast path 扫描开销） | **0.664 ms（边际）→ 实际端到端 1.01 ms（P50）** |
| `L_cold`（cold path） | **53.1 ms（P50）** |
| `fall-through latency` | 53.0 ms（P50），接近 cold path（因 entry 较短） |
| `fast path P99` | 1.35 ms |
| `mean fast path latency` | 1.04 ± 0.52 ms（128 次触发） |

> ✅ **实测端到端首 token 延迟降低超过 50 倍**（53.1ms → 1.01ms）

---

### 与基线方法的对比结果

| 对比维度 | 结果 |
|--------|------|
| vs Cold Path | **>50× 加速**（53.1ms → 1.01ms） |
| vs Prefix Cache | **~40× 加速**（因 prefix cache 仍需 39.12ms 的 decode 步骤） |
| hit rate (`h`) | 在不同 query-to-update ratio 下为 0.13–0.78，随 `p = λ_q / λ_u` 增大而升高 |
| gate coverage (`g`) | 在阈值 T=2.0 时达 **94.5%** |
| false-accept rate (`r(T)`) | **13.2%**（Wilson 95% CI: [8.3%, 20.4%]） |
| 模型能力影响 | 70B 模型能触发 gate；8B 模型 logit gap ≈ 0.65，始终无法触发 → fast path 无效 |

> 🔍 **关键发现**：fast path 的收益高度依赖模型在任务上的置信度。只有当模型“读得懂”任务时，gap 才足够大以触发 gate。

---

### 消融实验结果（隐含分析）

虽然没有显式的消融表格，但文中通过多组对照揭示了关键因素的影响：

| 变量 | 影响 |
|------|------|
| **模型规模** | 70B 模型可触发 gate（gap 最高达 20），8B 模型 gap≈0.65，方向不稳定 → 说明 fast path 仅适用于 high-capability models |
| **entry length** | fast path 加速比随 `L_entry` 增加而提升（见 Figure 4），因为 `L_cold` 上升而 `L_fast` 不变 |
| **hit rate** | 当 `p = λ_q / λ_u` 较高（查询密集）时，hit rate 接近 1，能量浪费少；反之则浪费严重 |
| **confidence threshold T** | 提高 T 可降低 false-accept rate，但牺牲 coverage（trade-off 曲线见 Figure 7） |

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Idle Time 是可利用资源**：stateful session 中的 idle window 可用于执行下一轮请求的 entry decode，实现 wall-clock-free 的计算摊销。
2. **Ready-Distribution Cache 是核心创新**：不仅缓存 KV，还缓存 logits 分布，使得某些请求可以完全跳过 decode。
3. **Confidence-Gated Fast Path 可行**：通过 calibrated confidence gate（logit gap + fast-answer set）可在低风险下实现 zero-decode 响应。
4. **性能增益显著**：实测首 token 延迟从 **53.1ms 降至 1.01ms（P50）**，提速超 **50 倍**。
5. **效果依赖模型能力**：只有当目标模型在任务上具有高置信度时，gate 才会频繁触发，否则机制退化为普通 fallback。

---

### 方法的局限性
1. **依赖 calibrated confidence gate**：无验证机制，安全性完全取决于 gate 的校准质量。
2. **需要 genuine idle window**：若系统饱和，无法执行 pre-positioning，则无收益。
3. **fast path 适用范围有限**：仅适用于可归约为单 token 回答的任务，并受限于 fast-answer set。
4. **模型能力门槛高**：小模型（如 8B）无法触发 gate，收益为零。
5. **精度差异**：70B 模型使用 4-bit，8B 使用 BF16，非严格同精度比较。
6. **仅适用于 stateful setting**：无法应用于传统 stateless request-response 架构。

---

### 未来工作方向
- 探索不同模型尺度下 confidence gap 的变化规律（sharp threshold or gradual？）
- 扩展 fast path 至 multi-token 输出（e.g., early-exit 多层 head）
- 动态调整 gate threshold 以适应负载变化
- 在多 GPU 场景下优化通信开销
- 将 speculative pre-positioning 应用于更多 domain-specific 场景（如代码补全、对话摘要等）

---

> 📌 **一句话总结**：  
> *Speculative Pre-Positioning* 利用 stateful session 的空闲时间，用目标模型自身提前解码到“决策点”，并通过缓存 output distribution 和 confidence-gated fast path，实现了 **首 token 延迟下降 50 倍以上** 的突破性优化，尤其适用于大模型在高置信任务中的低延迟服务场景。

</details>

---

### 15. [MuonSSM: Orthogonalizing State Space Models for Sequence Modeling](https://arxiv.org/abs/2606.30461)

**Authors**: Thai-Khanh Nguyen, Ngoc-Bich-Uyen Vo, Thieu N. Vo, Tan M. Nguyen, Cuong Pham  
**Category**: cs.LG  
**Published**: 2026-06-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.30461v1  

#### Abstract
State space models (SSMs) have emerged as efficient linear-time alternatives to attention for long-sequence modeling. However, existing SSMs often suffer from instability and memory degradation over extended horizons due to poorly conditioned first-order updates and unbalanced update geometry. We in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**MuonSSM: Orthogonalizing State Space Models for Sequence Modeling**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现有的 **State Space Models (SSMs)** 虽然在长序列建模中具有线性时间复杂度优势，但在长期依赖和深层堆叠时面临以下挑战：

- **训练不稳定**：由于状态转移矩阵的谱条件差（poorly conditioned），导致梯度消失或爆炸。
- **记忆退化**（memory degradation）：重复的秩-1 更新导致奇异值分布不均（spectral anisotropy），信息被覆盖。
- **更新几何失衡**：输入相关的仿射更新本质上是**一阶动态**，缺乏对更新方向的惯性控制。

这些问题限制了 SSM 在超长上下文、鲁棒性和泛化能力上的表现。

---

### ✅ 提出的新方法与核心思想

作者提出 **MuonSSM**，一个通用框架，通过显式地**正交化（orthogonalizing）内存更新的几何结构**来稳定 SSM 训练。

#### 核心创新点：

1. **动量增强路径**（Momentum-Augmented Dynamics）
   - 引入辅助动量矩阵 $ M_t $，累积跨时间步的更新方向。
   - 类似于优化中的动量机制，提升长期信息传播能力。

2. **轻量级 Newton-Schulz 变换**
   - 对低秩输入注入（low-rank input injection）应用单步 **Newton-Schulz (NS)** 正则化。
   - 作用：将输入注入的奇异值约束在合理范围内（$\sigma_{\text{max}} \leq 1.2$），防止谱放大。

3. **保持并行扫描效率**
   - 整体架构仍满足**块仿射递推**（block-affine recurrence），支持高效的 **parallel associative scan**，维持 $O(L)$ 总计算量和 $O(\log L)$ 并行深度。

> 🔑 **核心洞见**：与其约束递归转移矩阵本身，不如直接调控**输入相关内存更新的几何性质**。

---

### ✅ 相比现有方法的优势

| 方面 | MuonSSM 的优势 |
|------|----------------|
| **稳定性** | 显著缓解梯度衰减，改善长期信用分配（long-range credit assignment） |
| **表达能力** | 动量积累促进高有效秩（effective rank）状态，减少信息干扰 |
| **泛化性** | 在超出训练长度的上下文中仍保持良好性能（length generalization） |
| **兼容性** | 可作为插件集成到多种 SSM 骨干（如 Mamba, LongHorn, Gated DeltaNet） |
| **效率** | 不增加渐近复杂度，仅引入常数因子开销 |

---

## 2. 核心实验方法和设置

### ✅ 数据集

实验覆盖三大模态，验证方法的普适性：

| 模态 | 数据集 | 任务 |
|-------|--------|------|
| **语言建模** | FineWeb-Edu 10B, Alpaca-52K | 预训练 + 指令微调 |
| **视觉建模** | ImageNet-1K, COCO, ADE20K | 图像分类、目标检测、语义分割 |
| **时间序列** | MuWiGes, UESTC-MMEA-CL, MMAct | 人类活动识别（HAR） |

---

### ✅ 实验设置与评估指标

#### 共同原则：
- 控制变量法：所有模型参数量、训练预算、超参一致。
- 使用相同的骨干网络结构，仅替换 SSM 层为 MuonSSM。
- 所有实验在 **4×NVIDIA H100 GPU** 上进行。

#### 关键评估指标：

| 模态 | 主要指标 |
|------|----------|
| 语言 | Perplexity, S-NIAH 准确率（PassKey, Number, UUID） |
| 视觉 | Top-1/Top-5 Accuracy, mCE（Mean Corruption Error）, AP (Box/Mask) |
| 时间序列 | Accuracy, Precision, Recall, F1-score |

#### 特别设计：
- **S-NIAH**（Single Needle in a Haystack）：测试模型在 8K 上下文下的检索能力，而训练仅限于 2K。
- **Robustness Evaluation**：在 ImageNet-C/R/A 上测试对抗扰动、风格迁移等分布外鲁棒性。

---

### ✅ 基线方法对比

| 基线模型 | 简介 |
|---------|------|
| **Mamba** | 当前主流选择性 SSM 架构，硬件高效 |
| **LongHorn** | 基于在线学习视角的 SSM，强调记忆压缩 |
| **Gated DeltaNet** | 改进 Delta Rule 的门控变体 |
| **2×dstate Baseline** | 扩展状态维度以排除容量效应 |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据汇总

#### 📊 表格 2 & 3：语言建模与长上下文检索

| 模型 | PPL ↓ | S-NIAH-PK @8K ↑ |
|------|--------|------------------|
| Mamba | 89.17 | 11.5 |
| Mamba + Muon | **83.47** | **15.8** |
| LongHorn | 80.98 | 39.3 |
| LongHorn + Muon | **74.65** | **44.5** |

> ⬆️ 在 **8K 上下文**下，MuonSSM 显著优于原生模型，说明其具备更强的**长度外推能力**。

---

#### 📊 表格 4 & 5：视觉任务性能

| 模型 | IN-1K Top-1 ↑ | mCE ↓ | COCO APb ↑ | ADE20K mIoU ↑ |
|------|---------------|--------|------------|--------------|
| Mamba | 81.08 | 112.84 | 50.8 | 43.9 |
| Mamba + Muon | **81.19** | **112.52** | **51.1** | **45.2** |
| LongHorn | 81.63 | 111.68 | 50.6 | 44.2 |
| LongHorn + Muon | **82.01** | **111.24** | **51.0** | **45.7** |

> ✅ 在标准任务和鲁棒性上均有提升，尤其在 **IN-C 和 IN-A** 上误差降低明显。

---

#### 📊 表格 6：时间序列 HAR 性能（MMAct）

| 模型 | Accuracy ↑ | F1-score ↑ |
|------|-----------|-------------|
| LongHorn | 72.47 | 73.76 |
| LongHorn + Muon | **74.40** | **76.43** |
| GatedDeltaNet | 66.39 | 67.75 |
| GatedDeltaNet + Muon | **66.61** | **68.73** |

> 💡 即使在非极长序列场景，也表现出更优的时间建模能力，表明其增益不仅来自“延长记忆”，而是**更新动态本身的稳定性提升**。

---

### ✅ 消融实验结果

#### 🔍 容量 vs. 几何调节（Table 7）

| 模型 | Accuracy (%) |
|------|--------------|
| LongHorn | 72.47 |
| LongHorn 2×dstate | 72.88 (+0.41) |
| MuonLongHorn | **74.40** (+1.93) |

> ❗ 结论：性能提升主要来自**几何正则化机制**，而非简单扩大状态空间。

#### 🔍 Newton-Schulz 迭代次数消融（Figure 4）

- **1次迭代**：最佳平衡点，收敛快且性能最优。
- **5次迭代**：过强约束反而损害性能，可能抑制必要非正交相关性。

#### 🔍 动量与归一化组合（Table 12）

| 变体 | Val Acc ↑ | Effective Rank ↑ |
|------|-----------|--------------------|
| Momentum Only | 72.04 | 12.98 |
| + Frobenius Norm | 72.53 | 13.34 |
| + Newton-Schulz | **74.67** | **16.62** |

> ✅ NS 归一化不仅能控制幅度，还能改变反向传播的**雅可比几何**，鼓励更正交的写入方向。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **几何正交化是稳定 SSM 的关键路径**
   - 显式调节内存更新的谱特性比修改转移矩阵更有效。
   - Newton-Schulz 归一化提供了廉价但有效的谱预处理。

2. **动量路径显著改善梯度传播**
   - 提供额外的梯度通道，缓解因输入依赖衰减导致的指数级衰减。
   - 实证显示梯度范数在长序列中更均匀（Figure 7）。

3. **提升表示丰富性**
   - 动量积累 + NS 归一化共同促进高有效秩状态，增强记忆多样性。

4. **广泛适用性**
   - 在语言、视觉、时间序列任务中一致提升性能，证明其通用价值。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **新增超参数** | $\gamma$（动量衰减）、$T$（缩放因子）需调节，尽管敏感性分析表明默认值已足够稳健（Figure 6） |
| **轻微计算开销** | NS 操作带来约 1.3× 常数倍延迟，但更快收敛抵消总成本 |
| **理论假设简化** | 分析基于线性动态，实际模型含非线性与门控，存在偏差 |

---

### 🔮 未来工作方向

1. **自适应几何调节策略**
   - 动态调整 $T$ 或迭代次数，根据序列内容变化。

2. **大规模预训练验证**
   - 当前实验集中在中小规模，未来可在百亿级以上模型验证效果。

3. **混合架构探索**
   - 结合 Attention 与 MuonSSM，构建 hybrid 模型，在局部精细建模与全局高效推理间取得平衡。

4. **理论深化**
   - 探索 NS 操作与最优控制、微分方程之间的联系，建立更坚实的数学基础。

---

## ✅ 总结

**MuonSSM** 是一种简单而强大的 SSM 改进框架，它通过引入**动量路径**和**轻量级 Newton-Schulz 正则化**，从根本上改善了内存更新的几何结构。实验证明，该方法在多个领域、多种骨干上均能带来**一致性提升**，特别是在**长上下文建模、鲁棒性和泛化能力**方面表现突出。其核心思想——“**正交化更新而非约束转移**”——为未来稳定、可扩展的序列建模提供了新的设计范式。

</details>

---

### 16. [Discovering Collaboration from Novelty: Random Network Distillation for Clustered Federated Learning](https://arxiv.org/abs/2606.30499)

**Authors**: Davide Domini, Gianluca Aguzzi, Ivana Dusparic, Danilo Pianini, Mirko Viroli  
**Category**: cs.LG  
**Published**: 2026-06-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.30499v1  

#### Abstract
Federated Learning often suffers under non-independently and identically distributed data, where a single global model may fail to represent the diversity of client distributions. Clustered Federated Learning mitigates this issue by training specialized models for groups of similar clients, but exis...

---

### 17. [BV-Blend: Uncertainty-Weighted Historical Baselines for Stable Critic-Free RL with Verifiable Rewards](https://arxiv.org/abs/2606.28707)

**Authors**: Yupeng Chang, Yuan Wu, Yi Chang  
**Category**: cs.AI  
**Published**: 2026-06-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.28707v1  

#### Abstract
Critic-free reinforcement learning with verifiable rewards (RLVR), exemplified by Group Relative Policy Optimization (GRPO), avoids training a value function (critic) and reduces memory and compute overhead relative to critic-based PPO pipelines for aligning large language models. However, GRPO-styl...

---

### 18. [Evolution Fine-Tuning: Learning to Discover Across 371 Optimization Tasks](https://arxiv.org/abs/2606.29082)

**Authors**: Young-Jun Lee, Seungone Kim, Minki Kang, Alistair Cheong Liang Chuen, Zerui Chen, Seungho Han, Taehee Jung, Dongyeop Kang  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.29082v1  

#### Abstract
Would experience designing faster GPU kernels also help close in on a long-standing open mathematical conjecture? Large Language Models (LLMs) integrated into evolutionary search have recently produced state-of-the-art solutions on optimization tasks, including open mathematical conjectures, GPU ker...

---

### 19. [DistilledGemma: Balanced Efficiency-Accuracy for Person-Place Relation Extraction from Multilingual Historical Articles](https://arxiv.org/abs/2606.29130)

**Authors**: Youssef Aboelwafa, Ahmed Samir, Nagwa Elmakky, Marwan Torki  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.29130v1  

#### Abstract
We present DistilledGemma, an efficient and accurate system for the HIPE-2026 shared task on person-place relation extraction from multilingual historical newspaper articles in English, German, and French. Our approach adopts a three-stage knowledge distillation pipeline designed to balance classifi...

---

### 20. [Are Humans Evolved Instruction Followers? An Underlying Inductive Bias Enables Rapid Instructed Task Learning](https://arxiv.org/abs/2606.29792)

**Authors**: Anjishnu Kumar  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.29792v1  

#### Abstract
Human adults can often perform a novel task correctly on the first attempt after only receiving verbal or written instructions. This rapid instructed task learning (RITL) is a hallmark of human cognitive flexibility, yet its mechanisms and parallels in artificial systems remain under-explored across...

---

### 21. [Efficient Retrieval-Augmented Generation via Token Co-occurrence Graphs](https://arxiv.org/abs/2606.30093)

**Authors**: Gianluca Bonifazi, Christopher Buratti, Michele Marchetti, Federica Parlapiano, Giulia Quaglieri, Davide Traini, Domenico Ursino, Luca Virgili  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.30093v1  

#### Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Language Models (LLMs) by grounding the generation process on external knowledge. However, standard RAG approaches struggle with multi-hop reasoning. While recent graph-based RAG methods improve the retrieval of interconnected ch...

---

### 22. [KernelFlume: Elastic Core-Attention Scaling for Agentic Long-Context Decoding](https://arxiv.org/abs/2606.29207)

**Authors**: Guangyu Xiang, Xueze Kang, Lin Zhang, Wenxiang Lin, Shaohuai Shi, Yuxin Wang, Xiaowen Chu  
**Category**: cs.DC  
**Published**: 2026-06-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.29207v1  

#### Abstract
LLM serving is increasingly dominated by long and dynamic decode workloads from agents, reasoning models, and extended conversations. When bursty long-context demand exceeds deployed capacity, existing serving systems typically scale out by launching additional serving instances with model replicas....

---

### 23. [NI-ORCA: A Parallel Algorithm for Counting the Orbits of Non-Induced Graphlets up to K4](https://arxiv.org/abs/2606.29651)

**Authors**: Syed Ibtisam Tauhidi, Arindam Karmakar, Thai Son Mai, Hans Vandierendonck  
**Category**: cs.DC  
**Published**: 2026-06-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.29651v1  

#### Abstract
Counting the orbits of graphlets in a network is a vital tool for understanding the structural roles of vertices in various graph analytics tasks. While existing algorithms efficiently compute orbits of induced graphlets, many real-world applications require non-induced orbit counts. However, no cur...

---

### 24. [Atompack: A Storage and Distribution Layer for Read-Heavy Atomistic ML Training Datasets](https://arxiv.org/abs/2606.29975)

**Authors**: Ali Ramlaoui, Daniel T. Speckhard, Sagar Pal, Fragkiskos D. Malliaros, Alexandre Duval, Victor Schmidt  
**Category**: cs.LG  
**Published**: 2026-06-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.29975v1  

#### Abstract
Atomistic machine learning datasets are increasingly used for training: large immutable snapshots are read repeatedly, shuffled across epochs, staged across clusters' storage systems, and republished as reusable scientific artifacts. This workload differs from interactive scientific curation, where ...

---

### 25. [Flow Reasoning Models: Scaling Reasoning Through Iterative Self-Refinement](https://arxiv.org/abs/2606.29150)

**Authors**: Alec Helbling, Andrey Bryutkin, Mauro Martino, Nima Dehmamy, Hendrik Strobelt  
**Category**: cs.AI  
**Published**: 2026-06-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.29150v1  

#### Abstract
Discrete flow models have recently shown promising performance on few-step text generation; however, when naively applied to structured reasoning tasks such as Sudoku and Zebra puzzles, they converge confidently to incorrect answers (solving only $\sim$36% of Sudoku puzzles). We introduce Flow Reaso...

---

### 26. [Toward Secure and Reliable PDDL Formalization of Large Language Models with Planner-in-the-Loop Feedback](https://arxiv.org/abs/2606.29700)

**Authors**: Jiamei Jiang, Jiajing Zhang, Feifei Mo, Linjing Li, Daniel Zeng  
**Category**: cs.AI  
**Published**: 2026-06-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.29700v1  

#### Abstract
Planning often requires symbolic specifications that are both executable and verifiable. For large language models deployed in autonomous or decision-support systems, failures in such formalization may lead to unverifiable decisions, execution failures, or unsafe downstream behavior. We present NL-P...

---

### 27. [Structure-Preserving Document Translation via Multi-Stage LLM Pipeline: A Case Study in Marathi](https://arxiv.org/abs/2606.28796)

**Authors**: Manasi Waghe, Danish Chandargi, Mohammad Aamir Rayyan, Raviraj Joshi, A. R. Deshpande  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.28796v1  

#### Abstract
Government documents in India are predominantly issued in regional languages such as Marathi, creating substantial accessibility barriers for non-native readers, interstate administrative bodies, and policy analysts. Although recent advances in neural machine translation have improved sentence-level...

---

### 28. [EVLA: An Electro-Aware Multimodal Assistant for Physically-Grounded Driving Reasoning and Control](https://arxiv.org/abs/2606.28938)

**Authors**: Yuxin Liu, Zihan Chen, Haoyu Wang, Mingxuan Zhang, Ruijie Lin, Siyuan Zhao  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.28938v1  

#### Abstract
Modern vision-language models (VLMs) for driving assistants typically treat vehicle dynamics as a black box, resulting in decisions that lack awareness of the vehicle's real-time electro-mechanical state. To bridge this gap, we introduce the Electro-Visual-Language Assistant (EVLA) -- a novel framew...

---

### 29. [Travel-Oriented Reasoning Large Language Model via Domain-Specific Knowledge Graphs](https://arxiv.org/abs/2606.29254)

**Authors**: Vignesh Ram Nithin Kappagantula, Shayan Hassantabar, Samuel Simpson, Golnaz Moallem  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.29254v1  

#### Abstract
Large language models (LLMs) demonstrate broad reasoning abilities but struggle with accuracy and reliability in specialized domains such as travel, where reasoning depends on precise definitions, rules, and expert-defined conceptual frameworks, and where confident but unfounded outputs arise from a...

---

### 30. [REAR: Test-time Preference Realignment through Reward Decomposition](https://arxiv.org/abs/2606.30339)

**Authors**: Fuxiang Zhang, Pengcheng Wang, Chenran Li, Yi-Chen Li, Yuxin Chen, Lang Feng, Chenfeng Xu, Masayoshi Tomizuka, Bo An  
**Category**: cs.CL  
**Published**: 2026-06-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.30339v1  

#### Abstract
Aligning large language models (LLMs) with diverse user preferences is a critical yet challenging task. While post-training methods can adapt models to specific needs, they often require costly data curation and additional training. Test-time scaling (TTS) presents an efficient, training-free altern...

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

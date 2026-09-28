# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-28 12:02:48 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DeepEdu-v1: Efficient and Scalable Agentic LLMs for Vietnamese Education](https://arxiv.org/abs/2609.31568)

**Authors**: Quang Nguyen, Hieu Nguyen, Hien Hoang, Toan Pham, Cong Tran, Nam Vu  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2609.31568v1  

#### Abstract
AI tutoring could markedly improve learning outcomes for students in developing regions such as Vietnam, yet the two obvious paths both fall short. Cloud assistants such as ChatGPT route sensitive student data to foreign servers---violating data-sovereignty laws such as Vietnam's Decree 53---and, pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# DeepEdu-v1: Efficient and Scalable Agentic LLMs for Vietnamese Education 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

本论文针对在越南等资源受限地区部署 **AI 教育辅导系统** 所面临的两大核心挑战：

- **基础设施瓶颈**：自托管（self-hosting）大模型时，长上下文推理中的 **KV Cache 内存占用** 和 **prefill 阶段的二次延迟** 导致消费级 GPU 上出现内存溢出（OOM）和响应缓慢。
- **语义对齐瓶颈**：主流 LLMs 在西方中心化语料上预训练，缺乏对本地课程（如越南教科书 *Sach Giao Khoa*）的系统性理解，导致在区域特定内容上频繁产生 **hallucination**（幻觉），存在严重教学风险。

此外，直接微调（fine-tuning）以适应本地课程成本高昂且易引发 **catastrophic forgetting**（灾难性遗忘）。

---

### 提出了什么新方法或新思路

作者提出了 **SCALE**（Self-improving Context-Aware Learning Engine）框架，并在此基础上构建了 **DeepEdu-v1** 系统，包含两大创新组件：

#### （1）高效长上下文推理引擎：Similarity Chunk Rolling (SCR)

- **核心思想**：将传统的 **per-sub-chunk** 粒度的稀疏注意力选择（如 TokenSelect）提升为 **per-cluster** 粒度。
- **实现方式**：
  - 利用子块间查询表示的高度相似性，通过 **anchor-based clustering** 将连续相似的 sub-chunks 聚合成 cluster。
  - 对每个 cluster 只执行一次 **KV Cache 检索**（基于 anchor 子块的查询向量），然后在整个 cluster 上并行计算稀疏注意力。
- **优势**：显著减少检索调用次数，降低 prefill 阶段的计算开销和延迟。

#### （2）自我改进的代理层（Self-improving Agentic Layer）

- **核心思想**：不通过微调模型权重，而是通过 **Agentic Context Engineering (ACE)** 动态维护一个可验证的、课程对齐的“**Playbook**”（策略手册）。
- **实现机制**：
  - **Generator-Reflector-Curator 架构**：生成答案 → 反思错误 → 编辑 Playbook。
  - **Retrieval-Augmented Execution (RAE)**：在生成前仅检索与当前问题相关的 Playbook 条目，提高效率和准确性。
  - **Failure Memory Bank (FMB)**：存储过去失败案例的诊断，供反思模块类比学习。
  - **Adversarial Curriculum**：定期生成边界测试题，主动暴露系统弱点，促进自我修复。

---

### 相比现有方法的优势

| 维度 | 现有方法 | DeepEdu-v1 / SCALE |
|------|--------|---------------------|
| **数据主权** | 云服务（如 ChatGPT）违反本地法规 | ✅ 完全自托管，符合越南 Decree 53 |
| **本地化知识** | 依赖微调，成本高且静态 | ✅ 动态积累可信本地知识，无需微调 |
| **长上下文效率** | 传统稀疏注意力仍逐块检索 | ✅ 聚合检索，减少 7.7× 检索调用 |
| **自我进化能力** | 固定模型或 Prompt | ✅ 通过交互持续优化 Playbook |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **InfiniteBench**：包含六项长上下文任务：
  - `Code.D`（代码调试）
  - `R.KV`, `R.Num`, `R.PK`（结构化检索）
  - `En.Dia`（叙事对话问答）
  - `Math.F`（数学查找）
- **RULER**：针对于“大海捞针”（Needle-in-a-Haystack, NIAH）任务，测试不同长度下（4K–128K tokens）的检索准确率。
- **Formula & FiNER**：金融推理基准，用于评估 grounded reasoning 能力。
- **AppWorld**：交互式多步任务环境，模拟真实教学场景中的复杂操作。

---

### 实验设置和评估指标

| 设置项 | 描述 |
|-------|------|
| **硬件平台** | 单张 NVIDIA H100 GPU (80GB) |
| **主干模型** | Qwen2-7B-Instruct（用于 SCR 实验）、Qwen3-4B-Instruct-2507（用于代理层实验） |
| **量化方法** | 使用 AWQ/GPTQ 进行 PTQ（Post-Training Quantization） |
| **上下文长度** | 最长达 1M tokens |
| **评估指标** | - **Accuracy**：任务正确率<br>- **TTFT (Time-To-First-Token)**：首 token 延迟<br>- **TPOT (Time-Per-Output-Token)**：每输出 token 时间<br>- **TGC/SGC**（Task/Scenario Goal Completion）：AppWorld 成功率 |

---

### 基线方法对比

| 类别 | 基线方法 |
|------|---------|
| **长上下文推理** | TokenSelect（当前最优的 token-level 稀疏注意力）、FlashAttention-2、StreamingLLM、InfLLM |
| **代理架构** | 原始 ACE（Generator-Reflector-Curator 循环，无 RAE/FMB/Adversarial） |
| **压缩技术** | AWQ、GPTQ、Knowledge Distillation |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）长上下文推理性能（SCR vs TokenSelect）

| 指标 | 结果 |
|------|------|
| **检索调用减少** | 减少 **7.7×** 检索调用（从 8,959 → 1,165） |
| **TTFT 降低** | 平均降低 **~35%**（例如 R.KV 从 13.2s → 8.3s） |
| **准确率** | 在 `R.KV` 上达到 **98.0%**（vs TokenSelect 的 91.0%），其他任务持平或略优 |
| **RULER 长度扩展** | 在 128K 上保持 **65.28%** 准确率（vs 64.78%），TTFT 降低 **32.9%**（9.89s → 6.64s） |

> 🔍 **关键发现**：SCR 不仅提速，还因更宽的局部窗口提升了结构化检索任务的准确率。

---

#### （2）自我改进代理层性能

| 方法 | Formula Acc | FiNER Acc | 平均 |
|------|-------------|-----------|------|
| **Origin (Baseline)** | 70.0 | 51.3 | 60.7 |
| **+ RAE** | 77.5 | 47.6 | 62.6 |
| **+ Adversarial** | 79.5 | 52.1 | 65.8 |
| **+ FMB (DeepEdu)** | 73.0 | 52.9 | **63.0** |

- **RAE** 显著提升 Formula 表现（+7.5），但可能因信息裁剪损害 FiNER。
- **Adversarial Curriculum** 进一步挖掘边界案例，推动平均性能达峰值（65.8）。
- **FMB** 引入历史失败记忆，在 AppWorld 中表现最佳。

---

#### （3）端到端性能（SCR + Agentic Layer）

| 场景 | 方法 | 平均 TTFT | 加速比 |
|------|------|----------|--------|
| **AppWorld Normal** | Origin | 11.96s | 1× |
| | **SCR (DeepEdu)** | **5.51s** | **2.17×** |
| **AppWorld Challenge** | Origin | 12.02s | 1× |
| | **SCR (DeepEdu)** | **5.75s** | **2.09×** |

> ✅ 在真实代理交互场景中，SCR 实现近 **2倍** 的 TTFT 加速。

---

### 消融实验结果

| 模块 | 影响 |
|------|------|
| **Lmax=4096 vs 1024** | 更大 cluster 提升 amortization，降低延迟；但对精细推理任务略有负面影响 |
| **Threshold θ=0.95 vs 0.99** | θ=0.95 更激进聚类，延迟更低；θ=0.99 更细粒度，适用于敏感任务 |
| **RAE** | 显著降低 prompt 噪声和 TTFT，但需高质量检索 |
| **FMB** | 提升跨任务泛化能力，尤其在多步交互中稳定表现 |
| **Adversarial** | 主动暴露弱点，是突破性能天花板的关键 |

---

## 4. 关键结论和发现

### 论文的主要发现

1. **长上下文效率瓶颈可通过聚合检索解决**：利用子块间查询相似性进行 cluster-level 检索，可在几乎不损失准确率的前提下大幅降低 prefill 延迟。
2. **本地化知识应通过运行时上下文而非权重更新来注入**：通过 **Playbook + ACE** 架构，系统能持续积累可信的本地教学知识，避免昂贵且不可控的 fine-tuning。
3. **自我改进需要主动压力测试**：仅靠常规交互无法覆盖所有边界情况，**Adversarial Curriculum** 是驱动系统进化的关键机制。
4. **系统组件可组合增益**：SCR 与 agentic layer 各自有效，联合部署时实现 **2× 端到端加速** 和 **+9.5% 复杂任务准确率提升**。

---

### 方法的局限性

- **依赖高质量初始反馈**：Playbook 的演化依赖于 Ground Truth 或教师修正，若反馈错误会污染知识库。
- **未在真实越南课程数据上全面验证**：实验基于通用基准，尚未完全对接越南本土教材体系。
- **RAE 检索质量影响整体性能**：若检索不到相关规则，可能导致生成失败。
- **集群阈值需手动调节**：θ 和 Lmax 的选择影响效率与精度权衡，缺乏自动调参机制。

---

### 未来工作方向

1. **在真实教育场景中部署并评估 DeepEdu-v1**，特别是在越南中小学课堂中进行试点。
2. **开发自动化的 Playbook 质量监控与清洗机制**，防止错误知识传播。
3. **探索多语言、多文化适配框架**，将 SCALE 推广至其他低资源语言地区。
4. **结合更多模态输入**（如手写板、语音），增强人机交互体验。
5. **研究动态调整 clustering threshold 的方法**，实现效率与精度的自适应平衡。

---

> 📌 **总体评价**：  
> DeepEdu-v1 展示了一条 **高效、可扩展、可信赖** 的本地化 AI 教育路径。它不仅解决了技术层面的效率与准确性问题，更提出了一种 **去中心化、可持续演化的智能教育范式**，为资源受限地区的教育公平提供了切实可行的技术方案。

</details>

---

### 2. [Communication-Aware Model Distributed Inference via Latent Representation Compression](https://arxiv.org/abs/2609.30413)

**Authors**: Peyman Gholami, Theodoros-Thirimachos Davarakis, Teng Li, Miquel Sirera Perell\'o, Salil Reddy, Ayberk Yark{\i}n Y{\i}ld{\i}z, Anish Arora, Atilla Eryilmaz, Stratis Ioannidis, Chengzhang Li, Hulya Seferoglu, Ness Shroff  
**Category**: cs.DC  
**Published**: 2026-09-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.30413v1  

#### Abstract
We study optimization of distributed model inference over resource-constrained edge resources. We propose a framework that optimizes the trade-off between model accuracy and communication costs by controlling latent representation compression to meet strict Quality of Service (QoS) throughput target...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Communication-Aware Model Distributed Inference via Latent Representation Compression

## 1. 论文的主要贡献和创新点

### 解决的问题
本文研究在资源受限的边缘设备上进行**分布式模型推理**（distributed model inference）时，如何优化模型准确率与通信开销之间的权衡。具体而言，当多个机器学习任务的模型层被分割并分布在不同的边缘节点上执行时，中间激活（activations）在节点间的传输会引入显著的通信延迟。为了满足严格的**服务质量**（QoS）吞吐量目标，必须对这些激活进行压缩，但这又会导致下游推理准确率的下降。

### 提出的新方法和新思路
作者提出了一套名为 **Communication-Aware Model Distributed Inference** 的框架，其核心是通过控制**潜在表示压缩**（latent representation compression）来动态调节通信负载，从而在满足吞吐量约束的同时最大化模型准确率。

该框架的关键创新在于：
1.  **理论建模**：首次形式化地将机器学习任务的准确率与吞吐量之间的权衡问题建模为一个优化问题。
2.  **分层解决方案**：
    *   **已知信道状态信息**（CSI-Aware）：当可以获取精确的信道容量 `ce(t)` 时，对于单任务场景，推导出了**闭式最优解**（closed-form optimal solution）；对于多任务场景，证明了在准确率函数满足凹性（concave）的假设下，该问题可转化为一个**凸优化**（convex optimization）程序，并可通过一种**逐链路水填充策略**（per-link water-filling）求解。
    *   **未知信道状态信息**（Estimated-CSI）：在更现实的、信道状态未知且随机波动的环境中，提出了一种**基于随机对偶下降**（stochastic dual descent）的在线算法。该算法仅依赖于因果估计的信道状态，通过Lyapunov稳定性分析，证明了该方法能严格满足长期延迟约束，并实现有界的最优性差距。

### 相比现有方法的优势
*   **系统性权衡**：不同于以往静态的模型压缩（如剪枝、量化），该方法专注于在推理过程中动态调整**激活压缩**（activation compression），提供了一个系统性的框架来平衡通信与准确率。
*   **理论保证**：在两种信道信息假设下都提供了坚实的理论基础和性能保证（如可行性、收敛性、最优性差距）。
*   **鲁棒性和可扩展性**：所提出的算法在动态、不可预测的网络环境中表现出强大的鲁棒性，适用于复杂的多任务场景，为在动态边缘系统中部署流水线AI任务提供了一个可扩展的蓝图。

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖了多种视觉（vision）和语言（language）任务，具体数据集和模型如下：
*   **视觉任务**：
    *   `MNIST`：使用 `MLP` 模型。
    *   `CIFAR-10`：使用 `ResNet-56` 模型。
*   **语言任务**：
    *   `ShareGPT`：使用 `Gemma-1.1 2B/7B` 模型。
    *   `MMLU`：使用 `Llama-3.1 8B` 模型。
    *   `WikiText`：使用 `Llama-3.1 8B` 模型。
    *   `SST2`：使用 `Flan-T5-Base` 模型。

### 实验设置和评估指标
*   **实验环境**：实验分为两部分：
    1.  **离线模拟**（Offline Experiments）：在仿真环境中进行，用于验证算法的理论性能。
    2.  **真实测试床实验**（Testbed Experiments）：在三个真实的硬件平台上部署，以验证实际效果。
        *   **Raspberry Pi**：高度受限的CPU集群，用于小型LLM。
        *   **Jetson Orin Nano**：GPU加速的边缘平台，用于LLM和视觉工作负载。
        *   **PRESCIENT**：GPU密集型网络研究测试床，用于大规模和多任务部署。
*   **压缩方案**（Compression Schemes）：评估了三种主流的激活压缩技术：
    *   **Top-k sparsification (T)**：只传输最大幅值的激活坐标。
    *   **Uniform quantization (Q)**：将浮点数映射到低比特宽度的整数。
    *   **LLM.int8 (I8)**：一种混合精度方案，对异常值和主体分别处理。
*   **评估指标**：
    *   **聚合效用**（Aggregate Utility, U）：所有活跃任务的加权归一化准确率总和。
    *   **平均延迟**（Average Delay, D）：瓶颈阶段的平均延迟。
    *   **超额延迟**（Excess Delay, △D）：衡量QoS合规性，即 `D - 1/Rk` 的差值。若 `△D/D > 0.05`，则认为违反了可行性。

### 基线方法对比
*   **参考基线**（Reference）：
    *   `NONE`：不压缩，作为准确率上限。
    *   `MAX`：最大压缩，作为延迟下限。
*   **CSI-Aware 基线**：
    *   `UNI`：均匀压缩，受限于最弱链路。
    *   `EQ`：资源均分。
    *   `PROP`：按计算负载和数据大小比例分配。
    *   `PRIO`：优先满足高优先级任务。
*   **No-CSI 基线**：
    *   `MYO`：使用最近一次的信道观测。
    *   `CONS`：使用历史最低信道容量。
    *   `MA`：使用过去信道容量的移动平均。
    *   `D-EQUAL`：资源均分，独立运行No-CSI算法。
    *   `D-L-PROP`：根据对偶变量动态调整资源分配。
*   **本文方法**：
    *   `OURS-CSI`：本文的CSI-Aware最优求解器。
    *   `OURS-NOCSI`：本文的No-CSI求解器（使用90%置信下限LCB作为信道估计）。

## 3. 主要实验结果和性能指标

### 关键性能数据
*   在**离线实验**中，`OURS-CSI` 在所有单任务和多任务场景中，都是唯一能在满足延迟约束的同时达到最高可行效用的方法。
*   在**真实测试床实验**中，`OURS-NOCSI` 表现尤为出色。例如，在极具挑战性的LLM场景 `P-L1-G` 和 `P-L1-E` 中，`OURS-NOCSI` 是唯一一个在使用 `I8` 压缩方案时既满足延迟约束又能保持较高效用（分别为0.942和0.857）的No-CSI方法，而其他所有No-CSI方法均失效。

### 与基线方法的对比结果
*   **优于CSI-Aware基线**：`UNI` 因受限于最弱链路而牺牲了过多效用。`EQ`, `PROP`, `PRIO` 等方法无法有效利用不同任务对压缩的敏感性差异。
*   **优于No-CSI基线**：`MYO` 和 `MA` 由于对信道估计过于乐观，导致频繁违反延迟约束。`CONS` 虽然保守地满足了约束，但付出了巨大的效用损失。相比之下，`OURS-NOCSI` 成功地在满足延迟约束和保持高准确率之间取得了最佳平衡。
*   **消融实验**：实验表明，`OURS-NOCSI` 的成功很大程度上依赖于其使用的**90%置信下限**（LCB）信道估计器。这种悲观的估计策略为算法提供了安全缓冲，使其能够在高度波动的网络中稳定运行。

## 4. 关键结论和发现

### 主要发现
1.  **动态压缩至关重要**：通过动态调整激活压缩因子，可以在不牺牲过多准确率的前提下，有效应对边缘网络的动态变化，确保QoS目标的达成。
2.  **理论指导实践**：本文提出的基于凸优化和随机对偶下降的理论框架，不仅具有坚实的数学基础，而且在真实世界部署中也表现出了卓越的性能和鲁棒性。
3.  **悲观估计的有效性**：在不确定的环境中，采用悲观的信道估计（如LCB）是一种非常有效的策略，它能为系统提供必要的稳定性，避免因过度乐观而导致的服务中断。

### 方法的局限性
1.  **信道测量偏差**：在真实测试床上，直接探测信道容量（probing）得到的带宽测量值往往高于实际可用于激活传输的有效带宽，因为探测忽略了HTTP请求开销等第二阶效应。这导致了CSI-Aware方法（如 `OURS-CSI`）在真实环境中的性能不如预期。
2.  **压缩机制限制**：一旦信道容量低于最大压缩后激活的最小需求，任何压缩都无法满足要求，此时系统必须采取缓冲、丢弃负载或提前退出等措施。
3.  **自回归任务复杂性**：对于LLM等自回归任务，其预填充（prefill）和解码（decode）阶段的计算和激活负载不同，本文的简化模型未能完全捕捉这一复杂性。

### 未来工作方向
1.  **处理链路中断**：设计针对链路中断的早期退出（early-exit）策略。
2.  **支持自回归工作负载**：开发能够区分并优化预填充和解码阶段不同需求的推理框架。
3.  **改进信道感知**：研究更精确的信道状态测量方法，以减少探测与实际应用之间的差距。

</details>

---

### 3. [Training-Free Pronunciation Transcription via Text-Constrained Acoustic Rescoring](https://arxiv.org/abs/2609.30924)

**Authors**: Hikaru Asano, Yotaro Kubo, So Kuroki  
**Category**: cs.CL  
**Published**: 2026-09-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.30924v1  

#### Abstract
Accurate and efficient pronunciation transcription is essential for preparing text-to-speech training data at scale. Existing approaches have different limitations: grapheme-to-pronunciation (G2P) and speech-to-pronunciation (S2P) methods each capture only partial information, using only text or onl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Training-Free Pronunciation Transcription via Text-Constrained Acoustic Rescoring*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的发音转录（pronunciation transcription）方法存在以下局限：
- **G2P**（Grapheme-to-Pronunciation）仅依赖文本，忽略语音信号中的声学信息；
- **S2P**（Speech-to-Pronunciation）仅依赖语音，忽略已有的文本转录；
- **ST2P**（Speech-and-Text-to-Pronunciation）虽能结合两者，但通常需要昂贵的带发音标注的数据进行训练，限制了其可扩展性。

本文旨在解决如何在**无需额外训练**的前提下，高效、准确地融合文本与语音信息完成发音转录，尤其适用于大规模 TTS 数据准备场景。

### 提出的新方法
提出了一种**训练免费**（training-free）的 **ST2P 推理流水线**，核心思想是：
1. 利用现有的**词典、G2P 工具和规则**生成候选发音序列（lexical candidates）；
2. 使用**冻结的预训练 S2P 模型**（如 wav2vec2 或 Whisper 变体）对这些候选序列进行**基于声学证据的重排序**（acoustic rescoring）；
3. 通过**从左到右的贪心搜索**（left-to-right greedy search），选择整体负对数似然（Negative Log-Likelihood, NLL）最低的发音组合。

该方法完全在推理阶段集成文本与声学线索，**无需任何模型微调或标注数据**。

### 相比现有方法的优势
- ✅ **无需训练**：不依赖发音标注数据，显著降低部署成本；
- ✅ **高精度**：在多个语言上超越传统 G2P、S2P、ST2P 及商用多模态 LLM；
- ✅ **高效率**：推理速度远超 beam search 和直接解码，实现实时性优势；
- ✅ **模块化与可迁移**：只需更换语言资源和 S2P 模型即可迁移到新语言。

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 日语（Japanese）
- **JVS-dev**：600 条非平行语句，20 名说话人，人工标注假名（kana）
- **JSUT-BASIC5000**：5,000 条语句，1 名说话人，验证标签
- **JVS-par**：9,997 条语句，100 名说话人，对齐导出的银标签（silver labels）

每组评测均在三个数据集中各取 **512 条公共语句**进行评估。

#### 多语言扩展（Multilingual）
- **Spanish**：DIMEx100（墨西哥西班牙语）
- **French**：Rhapsodie（法语口语树库）
- **English**：Buckeye Corpus（美式英语对话）

### 实验设置
- **输入**：语音信号 $x$ + 文本转录 $w$（参考文本或 ASR 输出）
- **输出**：实际发音序列 $y$
- **发音表示**：
  - 日语：kana
  - 其他语言：IPA（International Phonetic Alphabet）

### 评估指标
- **CER**（Character Error Rate）：用于日语，归一化后计算字符错误率；
- **PFER**（Phonological Feature Error Rate）：用于多语言，基于 PanPhon 特征距离的编辑距离加权得分，更符合音系差异。

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **(A) Text-only G2P** | Open JTalk, Sudachi + Open JTalk, + spoken-form rules |
| **(B) Audio-only decoding** | Kana CTC greedy, kana-whisper |
| **(C) Same-model text conditioning** | kana-whisper + prompt, CTC + dictionary constraint |
| **(D) Released ST2P annotator** | Furigana Whisper（公开训练过的 ST2P 模型） |
| **(E) Open MLLMs** | Qwen3-Omni, Qwen2.5-Omni, Gemma-3n-E4B, Phi-4-multimodal |
| **Commercial References** | Gemini 2.5/3/3.5/3.6 Flash |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（日语，CER%，参考文本输入）

| 方法 | JVS-dev | JSUT | JVS-par |
|------|--------|------|--------|
| Text-only baseline (Sudachi + Open JTalk + rules) | 0.85 | 1.40 | 0.60 |
| Kana CTC (greedy) | 5.38 | 5.61 | 13.02 |
| Furigana Whisper + dict. constraint | 0.21 | — | 0.73 |
| **Ours (cascade)** | **0.15** | **0.17** | **0.04** |

> ✅ 在所有数据集上达到最低 CER，显著优于所有基线。

### ASR 转录作为输入的结果
| 方法 | JVS-dev | JSUT | JVS-par |
|------|--------|------|--------|
| Text-only baseline | 0.85 | 1.40 | 0.60 |
| **Ours (cascade)** | **0.64** | **0.65** | **1.58** |

> 即使使用 ASR 输出而非人工文本，仍保持最佳性能。

### 多语言结果（PFER%，参考文本输入）

| 方法 | Spanish | French | English |
|------|---------|--------|--------|
| Specialist default (text-only) | 2.47 | 4.39 | 13.76 |
| Best open MLLM (Qwen3-Omni) | 3.90 | 11.08 | 17.05 |
| **Ours (cascade)** | **2.52** | **4.11** | **13.21** |
| Best commercial (Gemini 3 Flash) | 2.13 | 2.98 | 13.05 |

> ✅ 在三种语言中均优于开源 MLLMs 和专用 G2P 工具；
> 🔁 英语结果为初步（preliminary），因数据划分受限。

### 消融实验（Ablation Study，日语，CER%，参考文本）

| 配置 | JVS-dev | JSUT | JVS-par |
|------|--------|------|--------|
| **Proposed (full)** | 0.154 | 0.171 | 0.042 |
| w/o margin check (gate) | 0.154 | 0.193 | 0.054 |
| w/o cascade | 0.161 | 0.182 | 0.058 |
| w/ beam search (B=5) | 0.167 | 0.160 | 0.042 |
| w/ silent audio | 13.87 | 14.82 | 12.54 |
| w/ other audio | 15.33 | 16.19 | 12.44 |
| Greedy oracle | 0.047 | 0.121 | 0.000 |

#### 发现：
- 移除 **margin check** 或 **cascade** 导致轻微性能下降，说明二者有效；
- 替换音频为静音或其他语音导致 CER 急剧上升至 ~13%，证明声学信号至关重要；
- **beam search** 与贪心搜索性能相近，但速度慢 3–3.5×，支持贪心策略合理性；
- 当前方法已接近 oracle 性能（恢复了 87–96% 的改进空间）。

---

## 4. 关键结论和发现

### 主要发现
1. **训练免费 ≠ 性能低下**：通过巧妙组合现有 G2P 工具与冻结 S2P 模型，可在无训练情况下实现 SOTA 级别的发音转录；
2. **推理时融合优于联合建模**：在推理阶段融合 lexical 和 acoustic 信息，避免了昂贵的多模态训练；
3. **贪心搜索足够有效**：由于发音决策具有局部性，left-to-right greedy search 能高效逼近最优解；
4. **级联结构提升效率**：通过 cascade 设计（仅在必要时运行大模型），大幅降低计算开销；
5. **跨语言可迁移性强**：仅更换语言资源即可迁移到西班牙语、法语、英语，表现稳定领先。

### 方法的局限性
- 依赖高质量的 **G2P 工具和词典资源**，在低资源语言中可能受限；
- 候选集合需覆盖真实发音变体，否则上限受制于候选质量；
- margin check 和 cascade 中的超参数（如 $ \lambda, \tau $）需在开发集上调优；
- 英语等语言的 IPA 表示复杂度较高，当前仅为初步探索。

### 未来工作方向
- 扩展至更多语言，尤其是低资源语言；
- 自动化候选生成机制，减少对手工规则的依赖；
- 探索更高效的搜索策略（如动态剪枝）；
- 将该框架应用于 TTS 端到端流程中的发音规范化模块；
- 结合 speaker-adaptive scoring 进一步提升个性化发音建模能力。

--- 

> 💡 **一句话总结**：  
> 本文提出一种无需训练的发音转录方法，通过 **text-constrained candidate generation + frozen S2P model 的 acoustic rescoring + greedy search**，实现了**高精度、高速度、低成本**的多模态发音预测，在日语及多种语言上全面超越传统方法与先进 MLLMs。

</details>

---

### 4. [Predictive Rolling-Horizon Optimization for Commitment-Aware Model-Parallel Inference under Spatio-Temporal Edge Dynamics](https://arxiv.org/abs/2609.31018)

**Authors**: Minghui Liwang, Chenxi Xu, Wei Gong, Li Li, Wenbo Zhu, Xinlei Yi, Yuhan Su, Xianbin Wang  
**Category**: cs.DC  
**Published**: 2026-09-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.31018v1  

#### Abstract
Model-parallel inference over dynamic edge systems requires scheduling decisions that account for not only instantaneous resources but also future resource contention and reliable service commitments. Existing edge-inference designs, however, predominantly optimize performance metrics based on curre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Predictive Rolling-Horizon Optimization for Commitment-Aware Model-Parallel Inference under Spatio-Temporal Edge Dynamics

## 1. 论文的主要贡献和创新点

### 解决的问题
现有边缘推理系统在调度决策时通常仅基于当前或短期的系统状态（如资源负载、通信条件），而忽略了**未来资源竞争**和**服务承诺的可履行性**。这导致：
- 调度决策具有“短视性”（myopic），无法保证长期的服务可靠性。
- 缺乏对服务完成时间的主动承诺机制，难以提供可验证的QoS保障。

### 提出的新方法：PROMISE
作者提出了 **PROMISE**（Predictive Rolling-horizon Optimization for Model-parallel Inference with Service Commitment），一个面向时空动态边缘环境的**预测性滚动优化框架**，其核心创新包括：

#### （1）引入“承诺完成时间”（Committed Completion Time, CCT）
- **CCT 是一个内生的调度变量**（endogenous service decision），而非外部给定的截止时间（deadline）。
- 平台主动宣布一个任务将在多长时间内完成（CCT），若实际完成时间（ACT）超过CCT，则视为**承诺违约**（commitment violation）。
- **CCT 的紧致性与奖励挂钩**：更紧的CCT带来更高的服务奖励，但违约风险也更高，从而在**承诺激进性**与**履行可靠性**之间建立显式权衡。

#### （2）统一建模时空动态性
PROMISE 联合建模了以下不确定性因素：
- **随机任务生成**（stochastic task generation）
- **移动性引起的通信变化**（mobility-induced communication variations）
- **隐私感知的模型分割**（privacy-aware model partitioning）
- **负载依赖的边缘计算能力**（load-dependent edge computing capability）

#### （3）预测引导的滚动优化（Prediction-guided Rolling-horizon Optimization）
- 在每个调度时隙，基于当前观测状态和对未来多个时隙的任务到达与计算负载的预测，构建一个**确定性等价的有限时域优化问题**。
- 通过“前滚”（rollout）模拟未来ES状态演化，评估当前映射决策对下游任务完成的影响。
- **仅执行第一阶段决策**，并在下一时刻重新优化，实现在线自适应。

### 相比现有方法的优势
| 维度 | 现有方法 | PROMISE |
|------|--------|--------|
| **动态性建模** | 通常只考虑部分动态（如信道变化） | 耦合建模任务、通信、计算的联合时空演化 |
| **时间决策机制** | 反应式/自适应调度（reactive/adaptive） | 预测感知的滚动优化（prediction-aware rolling-horizon） |
| **QoS 抽象** | 基于延迟、能耗等性能指标 | 基于CCT的**承诺型服务供给**（commitment-based service provisioning） |

---

## 2. 核心实验方法和设置

### 数据集
- **数值仿真**：使用合成数据集，参数见 Table 3，涵盖多种DNN模型（ResNet18, MobileNetV2, TextCNN, M5）。
- **真实硬件测试**：在基于 **Raspberry Pi** 的集群上部署，使用以下真实数据集：
  - **CIFAR-10**（图像分类）
  - **MNIST**（手写数字识别）
  - **THUCNews**（中文文本分类）
  - **UrbanSound8K**（城市声音分类）

### 实验设置
- **仿真平台**：Intel Core i9-13900H CPU + NVIDIA RTX 4060 GPU，Python 3.11 + PyTorch 2.7.0。
- **硬件平台**：4节点Raspberry Pi集群（2×Pi 4B, 2×Pi 3B），通过HTTP RESTful API通信。
- **调度时隙**：ΔT = 1.0 秒。
- **评估场景**（见 Table 5）：
  - S#1：改变SD数量（10–30）
  - S#2：改变ES数量（3–7）
  - S#3：改变任务生成间隔（5–25 slots）
  - S#4：改变数据批次数量（1–50）

### 评估指标
- **任务完成率**（Task Completion Ratio, TCR）
- **服务奖励**（Service Reward, R）
- **目标函数值**（Objective Value = λ_cmlp × TCR + λ_rwd × R）
- **端到端延迟**（End-to-end latency）
- **吞吐量**（Throughput）
- **隐私泄露率**（Privacy Leakage Rate，以SSIM衡量）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **RFA**（Random Feasible Assignment） | 随机可行分配，无信息基线 |
| **IRM**（Instantaneous Reward Maximization） | 仅优化即时奖励，短视策略 |
| **EAA**（Expectation-aware Assignment） | 基于统计期望，忽略状态演化 |
| **SHPS**（Short-horizon Predictive Scheduling） | 单步预测，短时域 |
| **IJO**（Iterative Joint Optimization） | 迭代联合优化，无轨迹预测 |

---

## 3. 主要实验结果和性能指标

### 数值仿真结果（Fig. 5–8）
- **高任务密度下仍保持高TCR**：即使SD数量从10增至30，PROMISE的TCR始终高于其他方法（平均高出15–25%）。
- **奖励最大化能力强**：在所有场景下，PROMISE的服务奖励均显著优于基线，尤其在高负载时优势更明显。
- **对系统规模变化鲁棒**：无论ES数量、任务到达率或计算负载如何变化，PROMISE均表现出**最稳定的性能**。

### 预测时域（X）消融实验（Fig. 9）
- **固定时域性能受限**：X=0（无预测）性能最差；X=5,10,20在不同负载下表现不一，无单一最优。
- **自适应时域最优**：PROMISE采用**自适应预测时域**（X[PROMISE]），根据当前决策的影响跨度动态调整，实现了**最稳定且最高的综合性能**。

### 硬件实验结果
- **ONNX部署可行性**：所有模型均可成功转换并部署在Raspberry Pi上，最大精度损失为3.44%（M5模型）。
- **多核并行加速有效**：ResNet18在4核并行下达到2.10倍加速（非线性，符合Amdahl定律）。
- **模型分割影响延迟**：
  - 早期分割 → 小计算 + 大通信开销
  - 深层分割 → 大计算 + 小通信开销
  - **存在最优分割点**，平衡计算与通信。
- **隐私泄露随深度递减**：在ResNet18中，`conv1`层重建SSIM达0.4135，而深层（如`layer4`）降至0.15以下，验证了**深层特征更安全**。

---

## 4. 关键结论和发现

### 主要发现
1. **预测性调度优于反应式调度**：将未来负载预测嵌入当前决策，能显著提升服务可靠性和奖励。
2. **CCT 是有效的服务抽象**：将服务承诺作为内生变量，使可靠性成为可量化、可优化的目标。
3. **自适应预测时域至关重要**：固定时域无法适应动态变化，而自适应机制能平衡预测收益与计算开销。
4. **模型分割需权衡效率与隐私**：分割点直接影响计算、通信与隐私，需联合优化。

### 方法的局限性
- **预测误差影响性能**：虽然采用滚动优化缓解，但预测不准仍可能导致承诺违约。
- **假设独立任务生成**：未建模任务间的空间相关性或突发流量。
- **简化通信模型**：假设时隙内传输速率恒定，忽略快速信道波动。
- **未考虑模型加载延迟**：假设模型常驻内存。

### 未来工作方向
- 扩展至更大规模硬件测试床，支持**在线随机调度**。
- 引入**更复杂的关联性工作负载与移动性模型**。
- 设计适用于大规模边缘系统的**可扩展决策机制**。
- 探索**基于学习的预测模块**，提升需求估计准确性。

> **总结**：PROMISE 通过**预测性滚动优化**与**承诺完成时间**（CCT）机制，首次将服务承诺的可履行性显式纳入边缘推理调度框架，在复杂时空动态下实现了**高可靠性、高奖励、强鲁棒性**的服务供给，为未来智能边缘系统提供了新的设计范式。

</details>

---

### 5. [Momentum-Guided Federated Split Distillation for Personalized Temporal Edge Intelligence](https://arxiv.org/abs/2609.31159)

**Authors**: Ahmed-Rafik Baahmed (LINEACT), Jean-Fran\c{c}ois Dollinger (LINEACT), Amine Brahmia (LINEACT), Mourad Zghal (LINEACT)  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.31159v1  

#### Abstract
We propose a momentum-guided federated split distillation framework for personalized, efficient, and autonomous temporal edge intelligence. We introduce TeRR-SAtt, our novel temporal reservoir student attention design that combines fixed reservoir representations, a lightweight temporal student, and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Momentum-Guided Federated Split Distillation for Personalized Temporal Edge Intelligence**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**个性化时序边缘智能（Personalized Temporal Edge Intelligence）**中的三大挑战：
- **资源受限**：IoT设备计算、内存、能耗有限，难以部署大型时序模型。
- **训练-推理不匹配**：传统的 Federated Split Learning (FSL) 要求服务器参与推理，限制了边缘端的自主性和低延迟能力。
- **客户端异构性**：不同客户端具有不同的时间分布、目标和学习轨迹，单一全局模型或共享教师更新无法满足个性化需求。

### **提出的新方法与创新思路**
作者提出了一个名为 **Momentum-Guided Federated Split Distillation (MG-FSD)** 的框架，其核心由两个创新组件构成：

#### **(1) TeRR-SAtt（Temporal ReseRvoir Student Attention）**
- 一种轻量级、可独立部署的边缘学生模型设计，结合了：
  - **固定 Reservoir 表示模块**：低成本提取时序特征，无需训练。
  - **轻量 Temporal Student 模块**：在边缘端进行高效学习。
  - **个性化 Output 模块**：支持本地任务定制。
- 采用 **Split Distillation 架构**：服务器仅在训练阶段提供高容量 Temporal Teacher 指导，**不参与推理**，实现完全自主的边缘推理。

#### **(2) AMGF（Anticipatory Momentum-Guided Fusion）**
- 一种基于学习动量的动态融合机制，用于生成个性化的教师更新：
  - 利用客户端诱导的梯度构建 **路径感知动量（path-aware momentum）**。
  - 使用 **Affinity Propagation Clustering** 将学习轨迹相似的客户端聚类。
  - 为每个集群生成 **cluster-specialized teacher update**，并引入 **自适应前瞻系数（adaptive anticipation coefficient）** 来控制更新步长，提升收敛速度。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **效率与自主性** | 解决了 FSL 中的“训练-部署不匹配”问题，实现全本地推理，降低通信依赖。 |
| **个性化能力** | 不再使用单一全局教师更新，而是通过动量聚类实现**按需协作与个性化指导**。 |
| **稳定性与收敛性** | 动量引导 + 自适应前瞻机制提升了非IID场景下的学习稳定性和收敛速度。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LBNL Building Dataset [38]**：真实世界智能建筑数据集。
- 包含 **20个边缘设备**，每个对应一个热区（thermal zone），采集以下传感器数据：
  - 风扇转速（air fan speed %）
  - 温度（temperature °F）
  - 加热水阀位置（heating-water valve position %）
- 任务：从 **72步输入窗口** 预测未来 **6步输出**（multi-step forecasting）

### **实验设置**
- **硬件平台**：Raspberry Pi 5（8GB内存），模拟资源受限边缘环境。
- **训练配置**：
  - Batch size: 16
  - 迭代次数：87轮模型更新
- **评估周期**：统一采用滑动窗口的 walk-forward validation 协议，共10轮。

### **评估指标**
| 类别 | 指标 |
|------|------|
| **边缘效率** | Training Latency, Inference Latency, CPU Usage, Memory Usage |
| **学习性能** | RMSE（Root Mean Square Error） |
| **消融分析** | 对比不同 AMGF 配置下的 RMSE 变化 |

### **基线方法对比**
1. **Traditional FL**：完整模型部署于边缘（GRU + Multi-head Attention + Dense）
2. **Traditional FSL**：将注意力模块卸载至服务器，其余保留在边缘
3. **FSL-KD**：基于 TeRR-SAtt 架构，但使用标准全局教师更新（Eq. 4）
4. **AMGF w/o anticipation**：关闭前瞻机制（α_max = 0）
5. **Full AMGF**：完整提出的动量引导融合机制（β=0.4, η=0.1, α_max=0.2）

---

## **3. 主要实验结果和性能指标**

### **边缘计算效率（TeRR-SAtt）**
| 指标 | 提升幅度（vs. Baseline） | 结果说明 |
|------|--------------------------|----------|
| **Training Latency** | ↓65.50% vs. FL, ↓53.70% vs. FSL | 训练速度快约 **2.9× (FL)** 和 **2.2× (FSL)** |
| **Inference Latency** | ↓44.70% vs. FL | 全本地推理避免通信开销，FSL 实际延迟更高（未计入通信） |
| **Training Memory Usage** | ↓18.40% vs. FL, ↓11.40% vs. FSL | 更适合内存受限设备 |
| **Inference CPU Usage** | ↓33.10% vs. FL, ↓14.50% vs. FSL | 显著减轻边缘CPU负担 |
| **Inference Memory** | ↓10% vs. FL | 接近 FSL 水平，但无服务器依赖 |

> ✅ **结论**：TeRR-SAtt 在保持高性能的同时显著优化了边缘资源消耗，并实现了**完全自主推理**。

---

### **学习性能（AMGF）**
#### **关键性能数据**
- **最大 RMSE 改善**：
  - **Full AMGF**: 较 FSL-KD 最多 **↓35.31% RMSE**
  - **AMGF w/o anticipation**: 最多 **↓31.49% RMSE**
- **平均 RMSE 改善（20 clients, 10 rounds）**：
  - Full AMGF: 平均 ↓5.14% ±6.14%
  - AMGF w/o anticipation: 平均 ↓0.51% ±7.48%

#### **代表性边缘节点表现（Fig. 4）**
| Edge Zone | 特征 | Full AMGF 改进 |
|---------|------|---------------|
| **z58 / z68** | 与全局动态差异大 | Round 6: RMSE 从 0.48 → 0.18 (**↓62.5%**) |
| **z27 / z69** | 文献[11]中高个性化需求客户 | z69 Round 5: 0.7390 → 0.4781 (**↓35.31%**) |
| **z41** | 学习轨迹突变频繁（bursty） | Round 4: 0.5807 → 0.4150 (**↓28.53%**) |

#### **消融实验结果**
| 配置 | RMSE 改进 | 分析 |
|------|----------|------|
| **FSL-KD (Global Update)** | 基准 | 全局聚合导致噪声和冲突信号干扰 |
| **AMGF w/o anticipation** | ↑31.49% | 验证了**动量聚类 + cluster-specialized fusion**的有效性 |
| **Full AMGF** | ↑35.31% | **自适应前瞻机制进一步加速收敛**，尤其在轨迹稳定时 |

> ✅ **结论**：AMGF 通过动量聚类识别兼容客户端，并利用可靠预测增强教师更新，显著优于全局聚合策略。

---

## **4. 关键结论和发现**

### **主要发现**
1. **效率与自主性的平衡**：
   - TeRR-SAtt 成功分离了**训练期协作指导**与**推理期边缘自治**，解决了 FSL 的部署瓶颈。
2. **个性化优于全局一致性**：
   - 在异构时序环境中，**单一全局模型/教师更新会损害个性化性能**；而 AMGF 能有效发现“学习伙伴”，实现定向知识迁移。
3. **动量是有效的协作信号**：
   - 客户端的学习动量不仅能反映当前状态，还能揭示长期趋势，适合作为**协作分组依据**。
4. **前瞻性更新需受控**：
   - 盲目外推会导致不稳定；AMGF 的 **可靠性门控机制（A[k] × C[t])** 确保只有在方向一致且稳定的集群中才启用前瞻。

### **方法的局限性**
- **服务器复杂度增加**：AMGF 引入了二次复杂度的操作（动量亲和矩阵 + Affinity Propagation），对大规模客户端群体可能带来计算压力。
- **超参数敏感性**：β（动量衰减）、α_max（最大前瞻步长）等参数需要调优以适应不同场景。
- **仅验证于特定领域**：目前实验集中在智能建筑，工业 IoT 或医疗场景尚未测试。

### **未来工作方向**
1. **扩展到更多应用场景**：如工业 IoT、智慧医疗（IoMT）等。
2. **层次化动量建模**：探索跨空间/组织层级的多粒度动量协同。
3. **通信优化**：利用动量信息压缩梯度传输，减少 Split Learning 的通信开销。
4. **动态资源适配**：让 TeRR-SAtt 模型大小随边缘设备预算动态调整。

---

> 📌 **总体评价**：本文提出了一种面向资源受限、异构、时序性强的边缘智能系统的新型联邦学习范式，兼具**高效性、个性化与自主性**，为下一代 Edge AI 提供了重要技术路径。

</details>

---

### 6. [EAServe: Encode-Aware Disaggregated Serving for Multimodal Large Language Models](https://arxiv.org/abs/2609.31551)

**Authors**: Kunxiong Zhu, Zhihao Shu, Hangyu Zheng, Minghai Qin, Miao Yin, Gagan Agrawal, Wei Niu  
**Category**: cs.DC  
**Published**: 2026-09-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.31551v1  

#### Abstract
Disaggregating the two stages, Prefill and Decode, onto separate GPU pools is now a standard optimization for (text-only) LLM serving. However, multimodal LLMs (MLLMs), which add a third phase, Encode, pose new challenges for resource allocation. Encode turns images, video, or audio into embeddings ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《EAServe: Encode-Aware Disaggregated Serving for Multimodal Large Language Models》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
多模态大语言模型（**MLLMs**）在推理过程中引入了第三个阶段——**ENCODE**（将图像、视频或音频编码为嵌入向量），形成了 **Encode-Prefill-Decode (EPD)** 三阶段流水线。然而，现有的服务系统存在以下问题：

- **资源利用失衡**：ENCODE 阶段是所有请求的入口，但由于单次前向传播计算量小，难以饱和 GPU，导致其 GPU 利用率极低。
- **下游饥饿**：ENCODE 成为瓶颈，导致 PREFILL 和 DECODE 阶段因等待输入而空转。
- **缺乏协同优化**：现有框架如 NVIDIA Dynamo 将 ENCODE 视为“透明通道”，未调节其批处理大小或与下游的负载分配。

### 提出的新方法与思路
论文提出 **EAServe**，一种以 **ENCODE 为控制点** 的分拆式（disaggregated）MLLM 推理服务架构，通过两个协同设计的层次实现高效资源调度：

#### （1）**Hybrid Auto Selection (HAS)** —— 离线配置搜索层
- 自动联合优化三个维度：
  - GPU 分配 `alloc`（E, P, D）
  - ENCODE 最大批大小 `B`
  - PREFILL 请求分流比例 `s`（远程 vs. 本地共置）
- 采用两阶段策略：
  - **Stage 1**: 基于各阶段容量分析进行剪枝，剔除明显不平衡的配置。
  - **Stage 2**: 在剩余候选中使用 **TPE-based Bayesian Optimization** 进行端到端微调。

#### （2）**运行时机制层** —— 在线自适应执行
- **Load-Adaptive Micro-Batching**：基于泊松到达过程推导动态批处理阈值（Poisson-gap dispatch），自动适应不同负载强度。
- **Rate-Controlled Partial Offload**：使用 **Deficit Counter Routing** 实现精确的分流控制，确保实际分流比接近设定值 `s`。
- **Dynamic SM Partitioning**：利用 `libsmctrl` 动态划分 Streaming Multiprocessors（SM），根据当前批大小调整 ENCODE 与本地 PREFILL 的资源占比，在保证 ENCODE 延迟的前提下最大化 GPU 利用率。

### 相比现有方法的优势
| 方面 | 现有方法（如 Dynamo, vLLM） | EAServe |
|------|-------------------------------|--------|
| ENCODE 资源管理 | 固定批大小（B=1）、无调控 | 动态批处理 + 可预测延迟 |
| 资源共享 | 不支持 ENCODE 与其他阶段共置 | 支持 ENCODE 与本地 PREFILL 共享 GPU |
| 配置搜索 | 手动调参或部分自动化 | 全自动联合搜索 `(alloc, B, s)` |
| 下游负载均衡 | 无显式控制 | 通过 offload ratio 主动调节 |

---

## 2. 核心实验方法和设置

### 使用的数据集与模型
评估覆盖三种主流模态，使用以下三个代表性 MLLM 架构：

| 模型 | 模态 | 编码器 | LLM Backbone | 参数量 |
|------|------|--------|---------------|--------|
| **LLaVA-v1.6-34B** | 图像 | CLIP ViT-L | Yi-34B | 34B |
| **Qwen2.5-VL-32B** | 视频 | ViT (dynamic) | Qwen2.5-32B | 32B |
| **Ultravox-v0.6-27B** | 音频 | Whisper | Gemma-2-27B | 27B |

- **图像/视频输入**：使用 vLLM 基准中的随机多模态生成器。
- **音频输入**：从 LibriSpeech 数据集中抽取真实语音样本。

### 实验设置
- **硬件平台**：
  - 主测试平台：8× NVIDIA RTX 6000 Ada（48GB，PCIe）
  - 补充平台：8× A100 SXM4（80GB，NVLink）
- **部署方式**：
  - ENCODE：TP=1（单卡）
  - PREFILL/DECODE：TP=2 或 TP=4（跨多卡并行）
- **请求模式**：Poisson 到达，速率从 1~5 req/s 扫描
- **每轮运行**：处理 500 个请求，取三次独立运行中位数

### 评估指标
- **Goodput**（核心指标）：
  $$
  \text{Goodput} = \frac{\text{满足 SLO 的请求数 } N_{\text{SLO-met}}}{\text{总运行时间 } T}
  $$
- **SLO 定义**：
  - **TTFT**（Time-to-First-Token）：必须 ≤ 给定阈值（P99）
  - **TPOT**（Time-Per-Output-Token）：必须 ≤ 给定阈值（P99）
- 其他指标：吞吐量（req/s）、平均/尾部延迟（TTFT/TPOT）

### 基线方法对比
- **NVIDIA Dynamo**：最先进的 EPD 分拆框架，但 ENCODE 批大小固定为 1。
- **vLLM**：主流文本 LLM 服务系统，采用连续批处理（continuous batching），但不原生支持 EPD。
- **EPDServe**、**HydraInfer**：其他 EPD 架构，用于交叉平台验证。
- 对比方式：在同一 8-GPU 环境下比较最佳配置下的性能。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Goodput 提升）
在相同 SLO 约束下，EAServe 相比基线实现显著提升：

| 对比对象 | 最高 Goodput 提升倍数 |
|---------|------------------|
| **vs. NVIDIA Dynamo** | **4.3×** |
| **vs. vLLM** | **1.7×** |

具体数值示例（中等 SLO Tier S3）：
| 模型 | Dynamo | vLLM | EAServe |
|------|--------|-------|---------|
| LLaVA-34B (图像) | 0.95 req/s | 1.11 req/s | **1.93 req/s** |
| Qwen2.5-VL-32B (视频) | 0.95 req/s | 1.32 req/s | **1.90 req/s** |
| Ultravox-27B (音频) | 0.98 req/s | 2.77 req/s | **4.17 req/s** |

### 尾部延迟大幅降低
在 A=2 req/s 下，P99 TTFT 显著下降：
- **LLaVA 图像任务**：
  - Dynamo: 233.8 s → EAServe: **10.8 s**（↓95%）
- **Qwen 视频任务**：
  - Dynamo: 144.8 s → EAServe: **18.0 s**（↓88%）

原因：Dynamo 的 ENCODE 队列不断积压；EAServe 通过自适应批处理保持“线速”处理。

### 跨平台一致性优势
在 RTX 6000 Ada 和 A100 平台上均优于 EPDServe 和 HydraInfer：

| 平台 | 系统 | 吞吐 (req/s) | P99 TTFT (s) |
|------|------|-------------|--------------|
| RTX 6000 Ada | EPDServe | 0.46 | 929 |
|              | HydraInfer | 0.58 | 664 |
|              | **EAServe** | **1.95** | **127** |
| A100 SXM4    | EPDServe | 1.01 | 304 |
|              | HydraInfer | 1.51 | 214 |
|              | **EAServe** | **4.91** | **29** |

> ✅ 表明 EAServe 的优势不受互联带宽限制（PCIe vs. NVLink）影响。

### 消融实验结果

#### （1）HAS 搜索效率（图 12）
- 在 30 次试验预算内，**HAS 达到 99% 最优吞吐所需时间远低于 Random 和 Coordinate Search**。
- 例如在图像任务上：
  - HAS 达到 99% 最优需 **1.7 小时**
  - Random 未能达到该水平
- 原因：HAS 第一阶段剪枝掉不平衡配置，使搜索更聚焦。

#### （2）Dynamic SM Partitioning 消融（表 6）
在 Ultravox-27B 上启用本地 PREFILL (`s=0.55`)，比较不同 SM 管理策略：

| SM 配置 | 吞吐 (req/s) | P99 TTFT (s) | P99 TPOT (ms) |
|--------|--------------|---------------|----------------|
| 无分区（time-slicing） | 4.78 | 43.7 | 236 |
| 静态 50/50 分区 | 5.07 | 40.7 | 203 |
| **动态分区（本文）** | **5.21** | **36.7** | **167** |

> ✅ 动态分区进一步提升吞吐 3~9%，显著降低尾部延迟。

---

## 4. 关键结论和发现

### 主要发现
1. **ENCODE 是 MLLM 服务的关键控制点**：
   - 它既是入口又是资源浪费最严重的阶段，应作为调度中心而非被动模块。
2. **三阶段参数需联合优化**：
   - GPU 分配、批大小、分流比相互耦合，手动调参不可行，必须自动化。
3. **动态资源划分 + 自适应批处理是核心增益来源**：
   - 动态 SM partitioning 可在不影响 ENCODE QoS 的前提下释放闲置算力给本地 PREFILL。
4. **EAServe 显著提升 GPU 利用率**：
   - Dynamo 中 ENCODE GPU 利用率 <10%；EAServe 提升至约 **80%**（图 11）。
5. **性能增益具有模态依赖性**：
   - 图像：受益于自适应批处理 + 局部 PREFILL
   - 视频：主要靠批处理和阶段分配（HAS 自动选择 `s=1`，不启用本地 PREFILL）
   - 音频：轻量编码器留下大量空间，局部 PREFILL 效果最显著

### 方法的局限性
- **HBM 带宽无法隔离**：SM partitioning 仅隔离计算单元，共享内存总线可能导致带宽争用（尤其对 DECODE 类内存密集操作）。
- **目前限于单节点**：跨节点的 embedding 传输开销未考虑，多机扩展需额外研究。
- **假设到达过程近似泊松**：虽然实验证明对 bursty 流量鲁棒，但在极端非平稳场景下可能需在线速率估计增强。

### 未来工作方向
- 扩展至 **multi-node datacenter scale**，研究跨节点 embedding 传输优化。
- 引入 **online arrival-rate estimation** 机制，动态更新 Poisson-gap 参数。
- 探索 **MIG 或未来硬件分区技术** 以实现完全隔离的 HBM 分配。
- 将 EAServe 思路推广至其他多阶段生成模型（如 diffusion + LLM 联合推理）。

--- 

> 📌 **总结一句话**：  
> EAServe 重新定义了 MLLM 推理服务范式，将 **ENCODE 从瓶颈转变为控制中枢**，通过 **HAS + 三大运行时机制** 实现了高达 **4.3× 的 Goodput 提升**，同时显著改善了 GPU 利用率与尾部延迟。

</details>

---

### 7. [Unifying In-Memory Data Analytics through Sparse Compilation](https://arxiv.org/abs/2609.30497)

**Authors**: Anand Jayarajan, Gennady Pekhimenko  
**Category**: cs.DC  
**Published**: 2026-09-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.30497v1  

#### Abstract
As modern data analytics workloads become increasingly heterogeneous and hardware-intensive, achieving efficient multi-core performance across diverse applications remains an open challenge. We present Reffine, a compiler-based in-memory analytics engine that delivers high performance across a broad...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Unifying In-Memory Data Analytics through Sparse Compilation》总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代数据 analytics 工作负载日益异构化且对硬件资源要求高，导致在不同领域（如 OLAP、流处理、图分析、机器学习）中，系统往往采用**领域特定优化**（domain-specific optimizations），形成了碎片化的生态系统。这种碎片化带来了以下问题：
- 优化逻辑重复：如减少中间数据移动、提升局部性、并行化等基础执行挑战被反复解决。
- 抽象不统一：不同系统的数据与计算抽象差异大，难以共享优化策略。
- 编译器支持不足：许多非关系型 workload 缺乏高效的编译时优化与代码生成。

因此，如何构建一个**统一、高效、可移植**的 in-memory 数据分析引擎成为关键挑战。

### 提出了什么新方法或新思路
论文提出了 **Reffine** —— 一种基于稀疏编译（sparse compilation）的内存数据分析引擎，其核心创新在于：

#### （1）提出 Reffine IR：统一的数据与计算中间表示
- **数据抽象**：引入 `Field`，将所有数据建模为定义在**稀疏、无限、多维坐标空间**上的映射 `{(coordinates), value}`。
  - 表 → 以主键为坐标的 Field
  - 图 → 二维 Field（源节点, 目标节点）→ 边权重
  - 时间序列 → 一维 Field（时间戳 → 值）
- **计算抽象**：
  - `Operator`：通过谓词逻辑表达式定义输出 Field，形式为 `Out = Viterators:[predicate]{value}`，扩展自 domain relational calculus。
  - `Reduction`：用于聚合操作（如 SUM, COUNT），支持增量更新（via `deacc` 函数）。
- 该 IR 融合了 **relational algebra** 的表达能力与 **sparse iteration theory** 的高效代码生成潜力。

#### （2）设计基于稀疏编译的后端
- 利用 **SMT solver（Z3）** 自动推导高效的迭代空间（iteration space），将稀疏逻辑坐标映射到紧凑物理索引。
- 支持自动 **operator fusion** 和 **parallelization**，无需依赖复杂的模式匹配规则。
- 生成 LLVM IR 并通过 JIT 编译为硬件高效的并行代码。

#### （3）实现端到端优化流水线
从高级 DSL（如 SQL）经由 Reffine IR，进行融合、并行化、规范化，最终生成向量化、多线程执行的底层代码。

### 相比现有方法的优势
| 方面 | 现有方法（如 DuckDB, Umbra, Polars, NetworkX） | Reffine |
|------|---------------------------------------------|--------|
| **抽象通用性** | 领域特定抽象（列存、数组、图结构） | 统一 Field 抽象，跨领域适用 |
| **优化机制** | 基于规则的查询重写（易遗漏复杂模式） | 基于 IR 变换的通用融合与并行化 |
| **代码生成** | 手工调优内核或有限编译 | 全自动稀疏循环生成，支持 SIMD 与多核 |
| **生态整合** | 孤立系统 | 可作为通用后端供其他引擎复用 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **TPC-H**：标准决策支持基准，使用 scale factor 10 生成数据集。
- **Streaming Analytics**：
  - `Trading`：股票价格流上的 50 天移动平均比较。
  - `Normalize`：每 100 天窗口归一化。
- **Graph Analytics**：
  - `PageRank`：单步迭代。
  - `Triangle Counting`：三角形计数。
  - 使用 **SNAP Twitter graph**（81,306 节点，1.7M+ 边）。
- **Micro-benchmarks**：合成数据集（1 亿条记录），用于 select, sum, join 等基本操作测试。

### 实验设置和评估指标
- **硬件环境**：AMD EPYC 7371（32 核，含超线程），128GB DRAM。
- **评估指标**：
  - 执行时间（median of 5 runs）
  - 加速比（speedup）
  - 可扩展性（多线程性能）
- **公平性控制**：
  - 排除数据加载时间，仅测量计算阶段。
  - 输入数据均预加载至内存（Arrow 格式）。

### 基线方法对比
| 类别 | 基线系统 |
|------|---------|
| DataFrame / 分析库 | **Pandas**, **Polars** |
| 内存分析数据库 | **DuckDB**, **Umbra**（编译型 DB） |
| 图分析库 | **NetworkX**（Python 实现） |
| 相关研究系统 | **Weld**（学术系统，用于补充对比） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）TPC-H 性能（16 线程）
| 对比项 | 最高加速比 | 典型表现 |
|-------|-----------|----------|
| vs **DuckDB** | **24.9×**（Query 4） | 在 6/7 查询上更快，仅 Q3 慢 ~10% |
| vs **Umbra** | **3.2×**（Query 4） | 在 5/7 查询上更快，Q6 和 Q3 略慢 |

> **原因分析**：
> - Q4/Q18 含嵌套子查询，传统引擎难优化；Reffine 通过 **operator fusion** 将其转为单层循环，避免中间物化。
> - Umbra 虽也编译，但 fusion 受限于 pipeline 结构，无法跨越 join/aggregation 等阻塞算子。

#### （2）流式与图分析性能
| 应用 | vs 基线 | 加速比 |
|------|--------|--------|
| `Trading` | vs **Polars** | 4.2× |
| `Normalize` | vs **Polars** | 32.4× |
| `PageRank` | vs **NetworkX** | 2.5× |
| `Triangle Counting` | vs **NetworkX** | **93.4×** |

> - `Normalize` 巨大优势源于 Reffine 对 **range-based indexing** 的原生支持，而 Polars 需 group-by 模拟，开销大。
> - `Triangle Counting` 中 NetworkX 使用纯 Python 实现，效率极低。

#### （3）微基准测试（单线程）
| 操作 | vs Pandas | vs Polars |
|------|----------|----------|
| Select, Sum, Join | 1.21–50× | 1.05–21× |
| Inner/Outer Join | — | 最高 **1.33×** 优于 DuckDB |

> 显示 Reffine 即使在基本操作上也能生成高度优化的 co-iterating loops。

### 消融实验结果

#### （1）Operator Fusion 效果（Figure 6b）
- 测试未优化 vs 融合后的 TPC-H Q20 示例。
- **DuckDB**：
  - 未优化版本 → 手动重写为 join 后性能提升 **3.2×**
  - 表明其优化器无法自动完成此类变换。
- **Reffine**：
  - 未优化版本已比 DuckDB 快 **1.5×**
  - 应用 fusion 后总加速达 **8.8×**（vs 未优化 DuckDB），**2.7×** vs 优化后 DuckDB。

> 说明 Reffine 的通用 fusion 机制能自动捕获复杂优化机会。

#### （2）可扩展性分析（Figure 6c）
- **Query 4**（最佳场景）：
  - 单线程性能 Reffine 是 DuckDB 的 **35×**
  - 多线程下 Reffine 几乎线性扩展至 8 线程，最终达 **24×** 总加速。
- **Query 3**（最差场景）：
  - 单线程 Reffine 比 DuckDB 慢 2.1×
  - 但 DuckDB 仅扩展至 4 线程即饱和，而 Reffine 持续扩展至 16 线程，最终追平性能。

> 体现 Reffine 的 **operation-level parallelization** 更具可扩展性。

---

## 4. 关键结论和发现

### 主要发现
1. **统一抽象可行且高效**：`Field` + `Operator/Reduction` 的 IR 设计能够统一表达关系型、流式、图等多种 workload，并支持端到端优化。
2. **稀疏编译是关键使能技术**：通过 SMT solver 自动生成高效迭代空间，解决了稀疏数据上循环生成的难题。
3. **通用优化优于规则匹配**：基于 IR 的简单变换（如 fusion）能自动实现传统系统需手动编码的复杂优化。
4. **性能超越专用系统**：在多个领域上显著优于 DuckDB、Umbra、Polars、NetworkX，验证了“统一优于专用”的潜力。

### 方法的局限性
1. **未针对稠密线性代数优化**：
   - 当前 backend 不适用于 dense tensor operations。
   - 论文建议通过桥接现有 tensor compiler（如 TACO）来弥补。
2. **缺乏高级查询规划**：
   - 如 join order 选择、统计信息利用等仍需前端系统完成。
   - Reffine 定位为“优化与执行后端”，而非完整查询优化器。
3. **依赖 Z3 solver 带来编译时开销**：
   - 虽然运行时快，但编译过程可能较慢，不适合极低延迟场景。

### 未来工作方向
1. **集成稠密计算支持**：扩展 backend 以高效处理 dense linear algebra。
2. **动态调度与自适应执行**：结合运行时反馈调整迭代策略。
3. **更广泛的前端集成**：将 Reffine IR 作为标准后端，接入更多 DSL（如 Pandas, Spark, SQL 引擎）。
4. **GPU 支持**：将稀疏编译扩展至 GPU，生成 CUDA kernel。
5. **开源发布**：作者计划在论文接受后开源 Reffine，推动社区发展。

> 💡 **总体评价**：Reffine 展示了一种“**统一抽象 + 稀疏编译**”的新范式，有望打破当前数据分析系统的碎片化格局，在保持高性能的同时提供更强的通用性与可组合性。

</details>

---

### 8. [Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency](https://arxiv.org/abs/2609.31619)

**Authors**: Parsa Hosseini, Akasha Tigalappanavara, Sumit Nawathe, Chenrui Fan, Sourya Basu, Genta Indra Winata, Anirban Das, Soheil Feizi, Nima Chitsazan  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.31619v1  

#### Abstract
Reasoning models often generate very long reasoning traces, making inference computationally expensive. Existing approaches typically improve efficiency either through inference-time early-stopping mechanisms or by explicitly encouraging shorter reasoning during training, for example through reinfor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency

## 1. 论文的主要贡献和创新点

### 解决的问题
现代 **Reasoning Language Models** 在数学、科学和编程等复杂任务上表现出色，但其推理过程（Chain-of-Thought, CoT）往往生成过长的中间文本，导致 **inference cost** 极高。现有方法通常通过以下方式优化效率：
- **训练时**：使用强化学习（RL）加入长度惩罚（length penalty），或对简洁的推理路径进行微调。
- **推理时**：引入 early-stopping 机制，如基于 confidence、uncertainty 或答案稳定性来提前终止推理。

这些方法都**显式地将“效率”作为优化目标**，可能改变模型行为或依赖额外推理控制逻辑。

### 提出的新方法与创新思路
本文提出了一种全新的范式：**ConfSFT (Confidence-based Self-supervised Fine-Tuning)**。

- **核心思想**：不直接优化推理长度或效率，而是让模型在训练中学习预测自己在推理过程中的 **confidence**（置信度）。
- **方法本质**：
  - 利用模型自身生成的 token probabilities 自动构建 confidence 标签（无需人工标注或黄金答案）。
  - 仅监督 confidence 预测任务，**loss 中完全不包含任何关于推理长度、效率或停止的项**。
  - 推理时仍使用标准生成流程，**无 confidence elicitation、无 early-stopping 机制**。

### 相比现有方法的优势
- **更“自然”的效率提升**：效率是学习 metacognitive（元认知）信号（confidence）的**下游涌现结果**，而非被强制优化的目标。
- **推理流程不变**：部署简单，无需修改推理逻辑或添加外部控制器。
- **跨领域泛化性强**：仅在数学题（AIME）上训练，却能在科学（GPQA）、编程（LiveCodeBench, HumanEval）等任务上提升效率。
- **保持准确性**：在显著减少 token 数量的同时，准确率与基线持平。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练数据**：`AIME 2000–2023` 共 695 道数学题。
- **验证数据**：`AIME 2024`（30 题）用于模型选择。
- **测试基准**：
  - `AIME 2025`（数学）
  - `GSM8K`（小学数学应用题）
  - `GPQA-Diamond`（研究生级别科学问答）
  - `LiveCodeBench` 和 `HumanEval`（编程）

### 实验设置与评估指标
- **模型家族**：在四个不同规模的模型上验证：
  - `Gemma-4-E2B`
  - `Qwen3-4B`
  - `Nemotron-Nano-8B`
  - `GPT-OSS-20B`
- **评估方式**：
  - 每题采样 16 条独立推理路径。
  - **Accuracy (Acc.)**：正确样本比例（pass@1）。
  - **Avg. Tokens**：平均生成的 token 数量。
  - **Token Reduction**：相比基线模型的 token 减少百分比。

### 基线方法对比
- **Base Model**：原始推理模型。
- **DEER (Yang et al., 2025)**：基于 confidence 的推理时 early-stopping 方法。
- **On-Policy SFT (Zhao et al., 2026)**：在策略生成的简洁推理轨迹上微调。
- **A&Z (Arora & Zanette, 2025)**：使用强化学习并加入长度惩罚。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
| Model | 方法 | Acc. | Avg. Tokens | Token Red. |
|-------|------|------|-------------|------------|
| **Nemotron-Nano-8B** | Base | 42.5 | 11,325 | 0.0% |
| | **ConfSFT (Ours)** | **42.5** | **9,076** | **-19.9%** |
| **Gemma-4-E2B** | Base | 35.0 | 7,284 | 0.0% |
| | **ConfSFT (Ours)** | **35.0** | **5,987** | **-17.8%** |
| **Qwen3-4B** | Base | 59.2 | 13,268 | 0.0% |
| | **ConfSFT (Ours)** | **59.2** | **11,365** | **-14.3%** |
| **GPT-OSS-20B** | Base | 52.7 | 5,802 | 0.0% |
| | **ConfSFT (Ours)** | **53.3** | **5,235** | **-9.8%** |

> ✅ **总体平均 token 减少达 11.1%**，且准确率基本不变。

### 与基线方法对比
- **vs. 显式效率训练方法（On-Policy SFT, A&Z）**：
  - ConfSFT 效率增益 **相当甚至更好**，尽管后者明确优化长度。
  - 例如，在 Qwen 上，ConfSFT 减少 19.2% tokens，而 On-Policy SFT 仅减少 17.2%。
- **vs. 推理时 early-stopping（DEER）**：
  - DEER 可能减少更多 token，但**代价巨大**：在编程任务上，HumanEval 准确率从 89.7% 降至 47.1%，**严重损害可靠性**。
  - ConfSFT 在**不牺牲准确率的前提下**实现稳定效率提升。

### 消融实验结果
#### （1）Supervision Signal Ablation（Table 2）
替换 confidence 标签为其他信号：
| 方法 | Gemma Token Red. | Nemotron Token Red. |
|------|------------------|---------------------|
| **ConfSFT (Ours)** | **-11.7%** | **-9.7%** |
| Position（位置） | -7.3% | +1.3% |
| Binary Correctness（二值正确性） | +5.9% | -2.9% |
| Shuffled Confidence（打乱标签） | -0.1% | +4.0% |

> 🔍 结果表明：**只有保留与推理状态对应的分级 confidence 标签**，才能带来稳定效率提升。说明效果并非来自简单微调，而是真正学到了有意义的信号。

#### （2）Decision Point Marker Ablation（Table 3）
默认使用 `Wait` 作为决策点标记，改为段落分隔符 `\n\n`：
- 在 Gemma 上，token 减少 8.4%（vs. 10.3% with `Wait`）。
- 表明该方法对**标记的选择不敏感**，只要能采样到足够的中间状态即可。

---

## 4. 关键结论和发现

### 主要发现
1. **效率可以作为元认知学习的副产品涌现**：
   - 仅通过 self-supervised confidence training，就能使模型推理变得更高效，而无需显式优化长度或引入 early-stopping。
2. **推理结构得以保留**：
   - 基于 Schoenfeld 的认知阶段分析（Read, Analyze, Plan, etc.），ConfSFT **未显著改变各阶段的 token 分配比例**，说明它不是通过抑制某类行为（如 Verify）来缩短推理，而是整体更高效。
3. **泛化能力强**：
   - 仅在数学题上训练，却能提升科学和编程任务的推理效率，显示 confidence 学习具有跨领域价值。

### 方法的局限性
- **依赖模型生成合理的 confidence 轨迹**：若模型本身 confidence 与答案质量无关，则无法有效学习。
- **效率增益存在上限**：相比某些 aggressive early-stopping 方法，token 减少幅度较小。
- **未解决“错误自信”问题**：模型可能在错误答案上给出高置信度，但本文 focus 在效率而非校准。

### 未来工作方向
- 将 confidence learning 扩展到多模态或对话系统。
- 探索其他 metacognitive 信号（如 uncertainty, self-correction intention）是否也能驱动效率。
- 结合 confidence training 与轻量级 early-stopping，进一步提升效率边界。
- 研究 confidence 学习如何影响模型的鲁棒性和可解释性。

> 💡 **一句话总结**：  
> **“教模型学会判断自己有多确定”，反而让它“想得更快”——这揭示了高效推理可能源于对自身思维状态的理解，而非对速度的直接追求。**

</details>

---

### 9. [All In Good Time: Causality-Aware Framework for LLM-Based Simultaneous Speech-to-Speech Translation](https://arxiv.org/abs/2609.30416)

**Authors**: Amir Hussein, Enas Albasiri, Travis M. Bartley, Nourchene Ferchichi, Ke Hu, Harishchandra Dubey, Myungjong Kim, Zhehuai Chen, Oluwatobi Olabiyi, Sanjeev Khudanpur  
**Category**: cs.CL  
**Published**: 2026-09-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.30416v1  

#### Abstract
Large Language Models (LLMs) have shown strong performance in low-resource offline translation; however, extending them to simultaneous speech-to-speech translation (Simul-S2ST) remains challenging due to the scarcity of causally aligned training data with high cross-lingual speaker fidelity. In add...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：All In Good Time: Causality-Aware Framework for LLM-Based Simultaneous Speech-to-Speech Translation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **LLM-based Simul-S2ST**（Large Language Model-based Simultaneous Speech-to-Speech Translation）系统面临以下挑战：
- 缺乏**因果对齐**（causally aligned）的训练数据，难以学习“何时等待、何时输出”的策略；
- 依赖固定的翻译策略（如 wait-k），导致质量与延迟之间的权衡不佳；
- 忽视跨语言的**说话人保真度**（speaker fidelity）和**副语言特征**（prosody, emotion 等）；
- 现有延迟度量（如 LAAL）未考虑词序重排等语言必要延迟，造成评估偏差。

### 🚀 提出的新方法与创新
本文提出一个**因果感知框架**（Causality-Aware Framework），包含三大核心组件：

#### （1）**Factorized Architecture (FAST)**  
- **解耦建模**：将语音中的**词汇信息**（lexical）与**声学信息**（acoustic）分离处理。
  - 词汇信息由 ASR encoder 提取并输入 LLM；
  - 声学信息通过神经音频 codec（如 NanoCodec）建模，并结合 speaker embedding 实现高保真的跨语言 voice transfer。
- 采用 **multi-stream full-duplex 设计**，支持实时多模态输入、重叠语音和对话中断处理。

#### （2）**Causality-Aware Adaptive Policy (CAP)**  
- 受专业口译员“chunking”策略启发，基于源-目标文本对齐动态决定读写时机。
- 利用 **monotonic alignment 构造因果锚点**（causal pivots），确保每个目标 token 在获得足够源证据后才生成。
- 支持自适应分块（adaptive chunking），优于固定 wait-k 或静态分块策略。

#### （3）**Causality-Aware Average Lagging (CAAL)**  
- 新型延迟度量，区分**语言必需延迟**（如词序调整）与**系统可避免延迟**。
- 基于对齐关系计算“超出理想因果策略”的额外等待时间，提供更准确、更具判别力的延迟评估。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **训练数据** | 构建高质量因果对齐数据管道，提升 speaker similarity（从 0.24 → 0.61） |
| **架构设计** | 解耦感知与生成，避免共享表示带来的竞争目标问题 |
| **翻译策略** | 自适应而非固定策略，实现更优的质量-延迟平衡 |
| **延迟评估** | CAAL 比 LAAL 更合理、更具判别性，纠正误判（如将 System 1 错评更快） |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 主要使用 **CVSS-T dataset**（Spanish, German, French → English）；
- 构建约 **2.7K 小时/语言对** 的因果对齐 S2ST 训练数据；
- 额外引入内部数据，在最终对比中扩展至 **8K 小时多语言数据**（远少于基线）；
- 使用 **A2Flow TTS** 对 CVSS-T 进行重新合成，显著提升 speaker similarity。

### ⚙️ 实验设置
- **模型架构**：
  - Speech Encoder：multilingual FastConformer ASR（17层）
  - LLM Backbone：Qwen2.5-1.5B-Instruct
  - TTS Decoder：MagpieTTS + streaming NanoCodec（FSQ，13 codebooks）
  - 总参数量：约 **2B**
- **训练细节**：
  - 使用 NeMo 框架，32×NVIDIA A100 (80GB)
  - AdamW，lr=1e-4，warmup 4K 步，inverse square-root 衰减
  - 多任务损失权重：α_s2t=3, α_t2s=2
- **推理方式**：greedy decoding，每 80ms 接收一次语音编码输出

### 📊 评估指标
| 类型 | 指标 |
|------|------|
| **翻译质量** | BLEU, chrF++, COMET（case-insensitive, punctuation-free） |
| **语音质量** | UTMOS-V2（自然度评分） |
| **说话人相似度** | ECAPA-TDNN 的 cosine similarity |
| **延迟度量** | LAAL 和提出的 **CAAL**（p=0.9 处理删除项） |
| **语音翻译评估** | 先用 ASR 转录生成语音，再计算其翻译指标 |

### 🆚 基线方法
- **Hibiki-Zero** [15]：基于 LLM 的多流架构，使用离线 MT perplexity 控制策略
- **SeamlessM4T** [14]：流式多语言表达性 S2ST 系统
- **Fixed Policy Baseline**：固定 2s 分块策略的 FAST 版本

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Test Set, Table IV）

| System | Text BLEU | Spoken BLEU (ASR) | CAAL ↓ | LAAL ↓ | UTMOS-V2 ↑ | Spk-Sim ↑ |
|--------|-----------|-------------------|--------|--------|------------|-----------|
| Hibiki-Zero | 32.9 | 31.4 | 1.39 | 2.11 | 2.1 | 0.47 |
| SeamlessM4T | 36.0 | 32.2 | 1.23 | 1.89 | 1.9 | 0.31 |
| **FAST-CAP** | **35.2** | **27.6** | **0.85** | **1.25** | **1.8** | **0.44** |
| **FAST-CAP-L** | **35.8** | **32.8** | **0.85** | **1.25** | **2.0** | **0.53** |

> 注：FAST-CAP-L 使用更强的 TTS（Audio Flamingo 3-Chat）和更大 LLM，总参数达 5B。

### 🔁 与基线对比结果
- **相比 Hibiki-Zero**：
  - 文本翻译质量更高：+2.3 BLEU, +3.2 COMET
  - 延迟更低：**CAAL 减少 38.8%**，LAAL 减少 40.8%
  - 说话人相似度相当（0.44 vs 0.47），且使用数据仅为 1/20
- **相比 SeamlessM4T**：
  - 文本翻译接近（35.8 vs 36.0），但**说话人保真度显著更高**（0.53 vs 0.31）
  - 延迟大幅降低：**CAAL 减少 30.9%**，LAAL 减少 33.9%

### 🔍 消融实验（Ablation Study, Table II & III）
| 变体 | 影响 |
|------|------|
| **MFA vs NFA 对齐** | MFA 提升 +4.4 BLEU（更精确边界） |
| **Multi-stream vs Interleaved** | 多流设计带来 +3.6 BLEU 提升 |
| **LLM Latent vs Text Conditioning** | latent conditioning 显著降质（-16.3 BLEU），说明需显式文本监督 |
| **TinyLLaMA 替代 Qwen** | 性能接近，表明方法对 backbone **agnostic**，无需大规模多模态预训练 |

### ⏱️ CAP vs Fixed Policy（Table II）
| System | Input Duration | CAAL ↓ | BLEU ↑ |
|--------|----------------|--------|--------|
| FAST-Fixed (2s) | 2.0s | 0.87–0.91 | 26.7–33.4 |
| **FAST-CAP (1.5s)** | **1.5s** | **0.67–0.80**（↓16–26%） | **27.8–34.6**（↑1.1–1.2 BLEU） |

> 结论：**CAP 在更短平均输入下实现了更低延迟和更高翻译质量**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **因果对齐数据构建至关重要**：高质量的对齐支撑了有效的自适应策略训练。
2. **解耦架构（FAST）有效提升性能**：分离 lexical 与 acoustic 表示，兼顾翻译准确性与语音自然度。
3. **CAP 显著优化质量-延迟权衡**：相比固定策略，在更低延迟下取得更好翻译质量。
4. **CAAL 是更合理的延迟度量**：
   - 揭示 LAAL 的缺陷（如错误惩罚必要等待、奖励插入）
   - 提供更公平、更具判别性的系统比较依据
5. **小数据也能达到 SOTA**：仅用 8K 小时数据（远低于基线的 145K–160K），仍实现领先性能。

### ⚠️ 局限性
- 当前方法依赖外部 ASR 和 TTS 模块，端到端优化空间未完全挖掘；
- 对长对话上下文建模能力有限；
- 跨语言 voice cloning 仍有改进空间（尤其在情感迁移方面）；
- CAAL 对删除项的处理依赖经验分布近似，可能不够鲁棒。

### 🔮 未来工作方向
- 进一步提升**跨语言 voice transfer** 的自然性和表现力；
- 扩展至**长对话场景**，支持上下文记忆机制；
- 探索**多方交互**与**多模态输入**（如视觉线索）；
- 开发完全端到端的联合优化方案，减少模块间误差传播。

---

> 💡 **一句话总结**：  
> 本文提出了首个完整的因果感知 Simul-S2ST 框架 **FAST-CAP**，通过解耦架构、自适应策略和新型延迟度量，在**更少数据下实现了更低延迟、更高翻译质量和说话人保真度**，推动 LLM-based S2ST 向实用化迈进一大步。

</details>

---

### 10. [The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models](https://arxiv.org/abs/2609.31341)

**Authors**: Christoph Walser, Mauricio Fadel Argerich, Jonathan F\"urst  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.31341v1  

#### Abstract
Whether an information extraction pipeline should process page images or parsed text depends on the document, and the answer flips across the layout spectrum. We study this trade-off under a constraint that rules out (closed) cloud services: privacy-sensitive documents processed on-premise by small ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
本文针对**隐私敏感、需本地部署的小型模型（≤8B参数）在信息提取（Information Extraction, IE）任务中的准确性与能耗权衡问题**，系统地研究了不同文档类型下最优的信息提取流水线设计。

传统方法通常依赖云端大模型或固定流程（如OCR+LLM），但在资源受限且对隐私要求高的场景（如金融、医疗）中不适用。本文提出：**没有“一刀切”的最佳IE流水线，最优方案取决于文档本身的布局特性**。

### 🚀 提出了什么新方法或新思路
- **多目标优化框架**：将IE任务视为一个兼顾 **accuracy（准确性）** 和 **end-to-end energy（端到端能耗）** 的多目标决策问题。
- **跨维度设计空间分析**：系统性探索三大维度：
  - 输入表示（Input Representation）：raw images vs. OCR vs. parser
  - 模型架构（Model Family）：VLMs、text-only LLMs、specialized layout models
  - 推理配置（Inference Configuration）：batch size、FP8 quantization
- **实证驱动的部署指南**：基于真实能效测量（而非理论估算），提供可复现的节能建议。

### 🔍 相比现有方法的优势
| 维度 | 本文优势 |
|------|--------|
| **全面性** | 覆盖从预处理（OCR）、模型选择到推理优化的完整pipeline，而非仅关注模型本身 |
| **实用性** | 所有代码和配置开源（GitHub: [local-ie-energy](https://github.com/chrewbroccoli/local-ie-energy)），支持本地部署验证 |
| **能效导向** | 明确量化各组件能耗，揭示以往被忽视的关键成本项（如神经OCR） |
| **文档感知** | 首次明确指出“最优流水线随文档类型反转”这一核心现象 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 类型 | 特征 | 文档数量 | 字段数/文档 |
|-------|------|------|----------|------------|
| **Kleister-NDA** | 近纯文本（near-plain text） | 数字原生PDF，视觉结构极少，语义集中于一维文本流 | 337份合同 | 最多4个字段（effective_date, jurisdiction, party, term） |
| **VRDU Registration Forms** | 布局丰富（layout-rich） | 扫描件为主，信息高度依赖二维空间排布（表格、框线等） | 500份表单 | 最多6个字段 |

> ⚠️ 注意：两个数据集在**文档长度上也有差异**（Kleister平均5.87页/文档，VRDU为1.83页），因此所有能耗均归一化为 **per page（每页毫瓦时, mWh/pg）** 以公平比较。

### ⚙️ 实验设置
#### 模型类别
| 类别 | 示例模型 |
|-----|---------|
| **Vision-Language Models (VLMs)** | Qwen3-VL (2B/4B/8B) |
| **Text-only LLMs** | Llama-3.2 (1B/3B), Ministral-3-3B, Mistral-7B, Qwen3 (0.6B–8B) |
| **Specialized Layout Models** | Arctic-TILT（带bounding box输入）、NuExtract-2.0-4B |

#### 输入表示方式对比
| 方法 | 描述 |
|------|------|
| **Raw Images** | VLM直接读取图像（OCR-free） |
| **Tesseract OCR** | 经典OCR引擎，CPU运行 |
| **Docling** | 支持嵌入文本解析（digital-born优先）和视觉OCR（扫描件备用） |
| **DeepSeek-OCR 2** | 基于VLM的神经OCR，GPU加速，高精度但高耗能 |

#### 推理优化技术
- **Batching**：使用 `vLLM` 实现动态批处理，测试 batch size = {1, 5, 10, 20, 40, 60}
- **Quantization**：FP16 vs. FP8（通过vLLM实现在线量化）
- **硬件平台**：单张 NVIDIA L4 GPU（24GB VRAM）

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Average Field Exact Match (EM)** | 字段预测值与金标准完全匹配的比例（忽略大小写和格式标准化） |
| **Energy per Page (mWh/pg)** | 包括OCR/parsing + inference全过程能耗，使用 CodeCarbon 和 Bench360 测量 |
| **Pareto Frontier** | 在准确率-能耗平面上找出非支配解集合，用于判断最优trade-off |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 结果维度 | 发现 |
|--------|------|
| **最佳流水线因文档而异** | - **VRDU（布局丰富）**：VLM > Text-only LLM<br>- **Kleister-NDA（近纯文本）**：Text-only LLM + 简单parser > VLM |
| **批处理是最大节能杠杆** | 批处理使能耗降低 **38–85%**，且不影响准确率<br>多数模型在 batch=10 时已达能效饱和 |
| **FP8量化收益有限** | 单请求时节省27–32%，但**一旦启用批处理，仅额外节省9–19%（<1 mWh/pg）** |
| **神经OCR代价高昂** | DeepSeek-OCR 2 能耗是 Tesseract 的 **17–18倍**，其带来的精度提升常无法抵消能耗开销 |

### 🔁 与基线方法的对比结果

#### 在 VRDU 上的表现（布局丰富）
| 模型 | EM (%) | E2E Energy (mWh/pg) | 是否进入Pareto前沿 |
|------|--------|---------------------|--------------------|
| Qwen3-VL-2B | 65.1 | 7.9 | ✅ |
| Qwen3-VL-4B | 65.6 | 12.2 | ✅ |
| NuExtract-2.0-4B | 78.3 | 17.8 | ✅ |
| Qwen3-4B + DeepSeek-OCR 2 | 73.4 | 91.8 | ❌（太耗能） |

> ✅ VLM 和 specialized model 占据主导地位。

#### 在 Kleister-NDA 上的表现（近纯文本）
| 模型 | EM (%) | E2E Energy (mWh/pg) | 是否进入Pareto前沿 |
|------|--------|---------------------|--------------------|
| Qwen3-0.6B + Docling | 56.0 | 2.4 | ✅ |
| Qwen3-4B + Tesseract | 76.9 | 7.9 | ✅ |
| Qwen3-VL-4B | 70.1 | 8.1 | ❌ |
| Qwen3-4B + DeepSeek-OCR 2 | 74.7 | 80.8 | ❌ |

> ✅ 小型 text-only LLM + 低成本 parser 更优；**所有 VLM 配置均未进入Pareto前沿**

> ⚠️ Arctic-TILT 因在 Kleister 上做过微调（in-domain），表现高达92.4% EM，但作者已将其排除在零样本比较之外。

### 🔍 消融实验结果

#### RQ2: 输入表示的影响
| Parser | Kleister-NDA 能耗 (mWh/pg) | VRDU 能耗 (mWh/pg) | 说明 |
|--------|----------------------------|---------------------|------|
| Tesseract | 4.4 | 5.3 | CPU OCR，适合扫描件 |
| Docling | 1.5 | 14.1 | 数字文档极高效，扫描件退化为OCR |
| DeepSeek-OCR 2 | 75.1 | 83.5 | 能耗极高，仅在VRDU带来显著精度增益 |

> 💡 **结论**：  
> - 对数字文档 → 用 Docling（利用嵌入文本）  
> - 对扫描文档 → 可考虑神经OCR，但必须权衡其**一个数量级更高的能耗**

#### RQ3: 推理优化效果
| 技术 | 能耗降幅（vs. baseline） | 准确率影响 |
|------|--------------------------|-----------|
| **Batching (→bs=60)** | 38–85% | 无损失 |
| **FP8 Quantization (bs=1)** | 27–32% | ±<1 EM point |
| **FP8 (batched)** | 仅再降 9–19% | 基本不变 |

> 🔄 **批处理与FP8呈替代关系**：批处理提升了计算强度，使得低精度算术的收益大幅缩水。

---

## 4. 关键结论和发现

### 🧩 论文的主要发现

1. **最优IE流水线“反转”现象**：
   - **布局丰富的文档（如VRDU）**：应采用 **VLM 直接处理图像**，跳过OCR阶段，既更准又更省。
   - **近纯文本文档（如Kleister-NDA）**：应采用 **小型text-only LLM + 高效parser（如Docling）**，避免不必要的视觉模态开销。

2. **批处理是最有效的节能手段**：
   - 提升 batch size 至 GPU 内存饱和，可减少 **38–85% 能耗**，且不牺牲准确率。
   - 是部署前必须优先尝试的优化。

3. **FP8量化价值有限**：
   - 在无法批处理的场景中有意义（如实时交互）；
   - 但在批量服务中，其节能效果被批处理吸收，仅剩不到1 mWh/pg的边际收益。

4. **神经OCR不是万能药**：
   - DeepSeek-OCR 2 能耗是经典OCR的 **17–18倍**；
   - 其带来的精度提升不足以补偿能耗代价，在Pareto前沿上始终落败。

5. **预处理决定整体能效上限**：
   - OCR/parser 的能耗可能远超模型推理本身；
   - 因此必须将 **parsing energy** 纳入整体评估，否则会误导设计决策。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **能量测量依赖软件估计** | 使用 CodeCarbon 和 Bench360 进行能耗建模，未使用物理功率计，可能导致低估约20–30%。但相对趋势仍可靠。 |
| **单一GPU平台限制泛化性** | 实验基于 NVIDIA L4 GPU，其他架构（如H100、消费级卡）可能有不同的批处理效率和FP8支持程度。 |
| **文档长度与布局混淆** | Kleister（长+低布局） vs. VRDU（短+高布局），二者变量共变，难以分离“长上下文”独立影响。 |
| **未涵盖复杂提示工程** | 仅使用 single-shot prompting，未测试 CoT、few-shot 等高级策略，报告准确率为下界。 |
| **模型加载未计入** | 冷启动加载时间未包含在能耗中，适用于高频批量场景，不适用于低频按需加载。 |

---

### 🔮 未来工作方向

1. **构建长度匹配的长篇布局丰富数据集**：以解耦“文档长度”与“布局复杂度”的影响。
2. **探索 activation-aware weight quantization (AWQ)**：相比在线FP8，预量化模型可能进一步压缩内存并提升能效。
3. **扩展至多语言、手写体、发票收据等多样化文档类型**：验证当前指南的普适性。
4. **开发轻量级专用parser**：平衡神经OCR的精度与经典OCR的效率。
5. **集成节能调度器**：结合动态批处理、模型切换、early exiting 等机制实现自适应IE pipeline。

---

## ✅ 总结：三条实用部署建议（来自原文 Section 7）

> **First**, match parsing to layout:  
> - born-digital/text-heavy → embedded-text parser or classical OCR  
> - scanned/forms → neural OCR only if accuracy gain justifies 10× energy cost  

> **Second**, align architecture with structure:  
> - layout-rich → VLMs (read images directly)  
> - low-layout → small text-only LLMs + efficient parser  
> - if in-domain data exists → fine-tune small model instead of scaling zero-shot one  

> **Third**, batch before you quantize:  
> - increase batch size until GPU memory saturated (**38–85% energy saving**)  
> - treat FP8 as secondary optimization (<1 mWh/pg saving on top of batching)

</details>

---

### 11. [Why Jailbreaks Succeed in Diffusion Language Models: An Energy Landscape Analysis](https://arxiv.org/abs/2609.30841)

**Authors**: Thong Bach, Dung Nguyen, Thao Minh Le, Truyen Tran  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.30841v1  

#### Abstract
Existing attacks and defenses for diffusion-based large language models (dLLMs) target specific vulnerabilities but lack a shared framework explaining why attacks succeed. We propose one by interpreting safety alignment as shaping the denoising energy landscape: a well-aligned model routes harmful q...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Why Jailbreaks Succeed in Diffusion Language Models: An Energy Landscape Analysis

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前针对 **diffusion-based Large Language Models (dLLMs)** 的 jailbreak 攻击和防御研究缺乏统一框架来解释：
- 为什么攻击能成功？
- 不同攻击方式（如模板注入、锚定攻击）之间的关系是什么？
- 安全对齐在模型内部是如何体现的？

现有方法（如 DiffuGuard-SD）仅针对特定漏洞设计，无法泛化到新型攻击。

### 提出的新方法与思路
本文提出了一种基于 **能量景观（energy landscape）** 的分析框架，将安全对齐视为对 denoising 能量路径的塑造过程，并据此推导出三种互补的、无需训练的检测信号：

#### 核心思想
- 将 dLLM 的去噪过程建模为在概率单纯形上的能量最小化轨迹。
- 安全对齐通过创建一个“能量壁垒”（energy barrier），使有害查询被引导至安全输出区域（refusal/harmless completion）。
- 所有已知 jailbreak 攻击都可归结为两种策略以绕过该壁垒：
  1. **Potential-energy attacks**：在初始状态（step-0）隐藏恶意意图（如 PAD、DIJA）。
  2. **Kinetic-energy attacks**：在去噪过程中强行穿越能量壁垒（如 Anchoring）。

#### 三大检测信号（training-free）
| 信号 | 含义 | 检测目标 |
|------|------|--------|
| **Step-0 Ratio ($R_0$)** | 初始完全掩码状态下，拒绝词（`V_ref`）与服从词（`V_comp`）logit质量之比 | 检测初始恶意意图（potential energy） |
| **SED slope** | Logits 与初始状态的余弦相似度随时间下降的速度 | 检测全词汇空间中的轨迹扰动（kinetic energy in $V_\ell$） |
| **$\Delta E$ slope** | 安全能量间隙 $\Delta E(t)$ 随时间的变化率 | 检测安全相关子空间内的快速穿越（kinetic energy in $V_s$） |

> ✅ **优势**：三个信号覆盖了完整的能量预算，任何成功的攻击必须至少触发其中一个。

### 相比现有方法的优势
| 方法 | 局限性 | 本工作优势 |
|------|-------|-----------|
| **DiffuGuard-SD** | 只能检测模板类攻击（state modification），对非模板攻击无效 | 检测的是攻击后果（basin displacement），而非机制，更具普适性 |
| **AR 模型检测器（如 FJD）** | 依赖首token预测，无法观察完整生成轨迹 | dLLM 允许重复读取同一位置，暴露更多动态信息 |
| **黑盒/白盒优化攻击检测** | 需要额外训练或微调 | 本文方法完全 **training-free**，仅需读取 logits，开销极低 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 类别 | 数据集 | 数量 | 描述 |
|------|--------|-----|------|
| **有害提示（Harmful Prompts）** | HarmBench | 400 | 标准红队行为集合 |
| | AdvBench | 520 | 对抗性提示 |
| **良性提示（Benign Prompts）** | AlpacaEval | 200 | 指令遵循任务 |
| | XSTest | 250 | 边界安全测试（不触发误报） |

共构建 **5,050 prompt-condition pairs per model**，总计超过 10,000 条样本。

### 模型
- **LLaDA-8B-Instruct**（原生训练）
- **Dream-7B-Instruct**（AR 初始化）
- **LLaDA-1.5**（改进版）
- **LLaDA-MoE-7B**（稀疏 MoE 架构，附录验证）

### 攻击类型
| 攻击 | 类型 | 描述 |
|------|------|------|
| **PAD** | Template Attack | 在响应区插入结构连接符（如 "Step 1:"） |
| **DIJA** | Template Attack | 交错插入有害 token 和掩码 |
| **Anchoring (t=1, t=8)** | Mid-trajectory Intervention | 在第 t 步替换模型预测为有害内容并重新掩码 |

### 评估指标
- **AUROC**：衡量每个信号区分攻击与良性提示的能力。
- **Threshold-level Detection**：使用 `OR-rule` 或 `max-z` 规则进行联合判断，控制总体 FPR ≤ α。
- **TPR @ FPR=10%**：阈值级别召回率。

### 基线对比
- **DiffuGuard-SD**：唯一可比的 detection-only 方法，用于比较其与本文信号的互补性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（主表 Table 3 & Figure 4）

| Model | Attack | $R_0$ | SED slope | $\Delta E$ slope | $R_0$+SED | All Three |
|-------|--------|--------|------------|------------------|------------|------------|
| **LLaDA-8B** | Harmful Direct | 0.94 | 0.43 | 0.30 | 0.83 | 0.83 |
| | PAD | 0.45 | 0.90 | 0.39 | 0.88 | 0.87 |
| | DIJA | 0.17 | 0.89 | 0.28 | 0.87 | 0.87 |
| | Anchoring t=8 | 0.94 | 0.91 | 0.92 | 0.95 | **0.96** |
| **Dream-7B** | PAD | 0.39 | 0.52 | **0.98** | 0.41 | **0.95** |
| | DIJA | 0.11 | 0.92 | 0.76 | 0.90 | 0.90 |

> 🔍 **关键发现**：
> - 单一信号无法覆盖所有攻击；
> - 三信号组合后，在所有模型上均达到 **AUROC ≥ 0.83**；
> - 特别地，Dream-7B 上 PAD 攻击仅被 $\Delta E$ slope 捕获（0.98），而 SED slope 失效（0.52），说明架构差异影响动能集中子空间。

### 与基线方法对比（DiffuGuard-SD）
| 方法 | PAD | DIJA | Anchoring | Harmful Direct |
|------|-----|------|----------|----------------|
| **DiffuGuard-SD** | 1.00 | 1.00 | ~0.50 | ~0.50 |
| **本文三信号组合** | 0.87–0.95 | 0.87–0.90 | 0.96 | 0.83 |

> ✅ **结论**：两者高度互补。SD 检测攻击“手段”（是否修改输入），本文检测攻击“后果”（是否跨越能量壁垒）。

### 消融实验结果

#### （1）词汇集鲁棒性（Table 5, Figure 6）
- 使用自动提取的 top-10 高频拒绝/服从词（AUTO10）效果接近人工设定（DEFAULT）；
- 同义词集（SYNONYM）因低频导致 AUROC 下降至 ~0.80；
- 最小集合（MINIMAL5）反而表现最好（最高达 1.00），表明只需捕捉主导模式即可。

#### （2）跨语言迁移（Table 7）
- 在中文提示下使用英文词汇集时，$R_0$ 性能从 0.937 降至 0.796；
- 重新用中文数据派生词汇后恢复至 0.962；
> 📌 表明：词汇集是模型自身的可测量属性，应按语言重学习。

#### （3）参数压力测试（Table 4）
- **PAD connector 数量减少至 1**：虽可逃避检测（$R_0$, SED slope 均低），但不足以诱导完整有害生成（仅产生截断片段）；
- **Anchoring 时间延迟**：越晚干预，速度信号越强（符合 Theorem 1 预测）；
> ✅ **核心结论**：攻击无法同时规避检测且成功越狱——检测阈值与越狱门槛紧密耦合。

#### （4）聚合规则比较（Table 15）
| Aggregation Rule | Performance |
|------------------|-----------|
| OR-rule | 基线，保守 |
| **max-z (calibrated z-score max)** | **无训练下全面优于 OR**，利用信号相关性回收 union bound slack |
| Logistic combination | 若有标注数据，recall 可翻倍（尤其 LLaDA 上的模板攻击） |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Jailbreak 成功的本质** 是绕过由安全对齐建立的能量壁垒，分为两类策略：
   - 隐藏初始意图（potential manipulation）
   - 强行加速穿越（kinetic forcing）

2. ✅ **三大信号天然互补**：
   - $R_0$：监控起点（intent visibility）
   - SED/$\Delta E$ slope：监控过程（trajectory velocity）
   - 任一环节暴露即被捕获。

3. ✅ **架构决定动能分布**：
   - LLaDA 系列：攻击扰动分散于全词汇空间 → SED slope 更有效
   - Dream-7B：扰动集中在安全子空间 $V_s$ → $\Delta E$ slope 主导

4. ✅ **检测与越狱门槛难以分离**：
   - 所有逃避检测的配置都无法生成完整有害内容；
   - 表明当前攻击强度受限于物理能量约束。

### 方法局限性
| 局限 | 说明 |
|------|------|
| **依赖 per-token factorization** | 当前 dLLM 均采用此结构，但若引入序列级修正（如 EDLM），可能关闭 per-position 攻击面 |
| **语言依赖性** | $R_0$ 需语言特定词汇集；跨语言需重新提取 |
| **长度敏感性** | 在 LLaDA 上，当 $L > 128$ 时 recall 显著下降（refusal mass 分散） |
| **未测试完全自适应攻击** | 如梯度搜索优化对抗 OR-detector，留待未来工作 |

### 未来工作方向
1. 设计 **energy-corrected alignment** 方法（如集成 EDLM 思想），从根本上关闭 per-position 攻击通道；
2. 探索 **多步能量预算追踪**，实现更细粒度的动态防御调度；
3. 将框架扩展至 **text-to-image diffusion models**，探索概念擦除与能量盆地的关系；
4. 开发 **adaptive thresholding** 策略以应对不同生成长度和 block size；
5. 研究如何结合本文信号与 generation-time defenses（如 DiffuGuard 完整 pipeline）。

---

> 💡 **一句话总结**：  
> 本文首次从 **能量动力学视角** 统一解释了 dLLM jailbreak 的成功机制，并提出了三个 **无需训练、高效互补** 的检测信号，实验证明它们共同构成了一个几乎无法绕过的“能量守恒”防火墙。

</details>

---

### 12. [MoMHa: Multi-Objective Optimization of LLM Harnesses over Accuracy, Safety, and Tokens](https://arxiv.org/abs/2609.30967)

**Authors**: Subhojyoti Mukherjee, Md Mehrab Tanjim  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.30967v1  

#### Abstract
Most work on improving large language models treats accuracy as the sole objective. We argue that the harness, the Python code surrounding the model that constructs prompts, routes calls, and parses outputs, is a first-class design surface whose quality is inherently multi-objective: an accurate har...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MoMHa: Multi-Objective Optimization of LLM Harnesses over Accuracy, Safety, and Tokens

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大多数提升大语言模型（LLM）性能的工作将**accuracy**作为唯一优化目标，忽略了实际部署中至关重要的其他维度。本文指出，**LLM Harness**（即围绕模型的Python代码，负责构建提示、调用路由、输出解析等）的设计本质上是多目标问题，必须同时权衡：
- **Accuracy**：任务正确率
- **Safety**：行为安全性（拒绝不安全请求）
- **Tokens**：推理成本（token消耗）

一个仅追求高准确率却无视安全或消耗大量token的harness并非好设计。

### 提出的新方法：MoMHa
作者提出 **META-HARNESS** 框架，并在其基础上构建了核心系统 **MoMHa**（Multi-Objective Optimization of LLM Harnesses），其核心创新如下：

- **将Harness设计形式化为多目标优化问题**：明确在三个目标轴上进行搜索：`accuracy`, `behavioral safety`, 和 `-tokens`。
- **单阶段联合奖励机制（Single-phase Joint Reward）**：
  - 使用标量化联合奖励函数：  
    `R(h) = accuracy(h) + λ_s * safety(h) - λ_t * tokens(h)`
  - 与传统的“先优化accuracy，再压缩token”的两阶段方法（2-phase）不同，MoMHa在**同一阶段**内同时考虑所有目标，使proposer能做出更优的结构性决策。
- **基于代理的Proposer（Agentic Proposer）**：
  - 使用 **CLAUDE CODE** 作为智能体proposer，拥有对文件系统的完全访问权限，可读取历史harness源码、执行轨迹（traces）、评分日志等。
  - proposer通过迭代地诊断失败并修改代码来生成新的harness候选。
- **三层安全架构（Three-Layer Safety Stack）**：
  1. **AST-GUARD**：静态分析，禁止危险的导入和函数调用（如`os`, `subprocess`, `eval`等）。
  2. **Domain-specific Safety Skills**：每个领域有定制化的安全规范，控制允许的导入和输出格式。
  3. **Sandbox Enforcement**：沙箱执行环境，限制超时、token数、API调用次数等。

### 相比现有方法的优势
- **超越单目标优化**：相比仅优化accuracy的MH、DSPy等，MoMHa在保持高准确率的同时显著提升了安全性和效率。
- **超越两阶段优化**：单阶段联合优化优于“accuracy-then-tokens”的两阶段流程，能发现更优的结构（如单次置信度门控验证器而非两次冗余验证）。
- **超越Prompt优化**：相比APE、OPRO、TextGrad等仅优化prompt文本的方法，MoMHa优化的是整个harness程序，搜索空间更大，能力更强。
- **可解释性与可控性**：生成的是可审计的Python代码，而非黑盒的soft prompt或embedding。

---

## 2. 核心实验方法和设置

### 数据集
在 **17个领域** 上进行了评估，分为三大类：

| 类型 | 领域 | 数量 |
|------|------|------|
| **合成能力套件（Synthetic Capability Suites）** | 文本分类（TC）、数学推理（Math）、智能体编码（Code）、多项选择题（MCQ）、事实验证（FV）、命名实体识别（NER）、SQL生成（SQL） | 7 |
| **真实世界公共基准（Real-World Public Benchmarks）** | LawBench, NuminaMath, FEVER, Spider, HumanEval, MBPP, MMLU-Pro | 7 |
| **用户特定安全领域（U-SafeBench衍生）** | 非法活动问答、自主身体伤害、自主心理伤害 | 3 |

每个合成领域有100个LLM生成的样本（50个用于搜索，50个用于测试）。真实世界基准直接使用公开数据集。

### 实验设置
- **Proposer**：CLAUDE CODE，拥有文件系统访问权限。
- **搜索预算**：每个合成领域约进行100次harness评估。
- **目标模型舰队（Target Model Fleet）**：在 **12个模型** 上进行跨模型评估，涵盖四个家族：
  - **Anthropic**: Claude Opus/Sonnet/Haiku
  - **OpenAI**: GPT-4/GPT-5系列
  - **Google**: Gemini 3.1 Flash-Lite
  - **DeepSeek**: DeepSeek-R1
- **评估方式**：在合成领域上进行搜索优化，在真实世界基准上进行零额外搜索成本的泛化测试。

### 评估指标
- **联合得分（Joint Score J）**：综合三个目标的复合指标  
  `J_v,d = acc_v,d * Safe / ln(tokens_v,d)`  
  其中 `Safe` 是该变体在三个安全领域的平均安全得分。
- **3D Hypervolume Indicator (HV)**：衡量Pareto前沿质量的正式多目标指标（越高越好）。
- **各单项指标**：准确率（accuracy）、安全得分（safety composite）、每例token消耗。

### 基线方法对比
共对比了 **10个基线**，分为三类：

| 类型 | 基线方法 |
|------|---------|
| **Harness重写基线** | **MH**：同框架但仅优化accuracy（单目标） |
| **Prompt优化基线** | CoT, APE, OPRO, DSPy, MIPROv2, TextGrad, GEPA, Rand |
| **自合成Harness基线** | **AH (AutoHarness)**：可生成完整Python代码，但仅优化accuracy |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | MoMHa | 最强基线（DSPy） | 提升 |
|------|--------|------------------|------|
| **合成赛道联合均分 J** | **0.482** | 0.422 (TextGrad) | **+0.060** |
| **真实世界赛道联合均分 J** | **0.461** | 0.377 (DSPy) | **+0.084** |
| **行为安全复合得分（U-SafeBench）** | **0.781** | 0.747 (TextGrad) | **+3.4** |
| **3D Hypervolume (HV)** | **0.481** | 0.362 (MH) | **+0.119** |
| **每例平均token消耗** | **672** | 2,229 (DSPy) | **-1,557** |

### 与基线方法的对比结果
- 在 **17个领域中的12个** 上，MoMHa的策略无需重新训练即可成功迁移到目标模型。
- 在 **合成赛道** 上，MoMHa赢得 **7/10** 个领域的单项冠军。
- 在 **真实世界赛道** 上，MoMHa赢得 **5/7** 个领域的单项冠军。
- MoMHa在 **所有11个系统中** 达到了最高的行为安全得分（0.781），而所有外部基线均未超过0.75。
- MoMHa在 **8/12** 个目标模型上表现最强，跨模型平均联合得分最高（0.434）。

### 消融实验结果（Ablation Studies）
| 变体 | 准确率 | 安全性 | 联合得分 J | 平均token |
|------|--------|--------|------------|-----------|
| **MoMHa (Ours)** | 0.539 | 0.781 | **0.611** | 572 |
| **2-phase (acc → tokens)** | 0.520 | 0.735 | 0.584 | 667 |
| **MoMHa-ns (无安全技能)** | 0.528 | 0.716 | 0.584 | 620 |
| **MoMHa-noTok (无token项)** | 0.512 | 0.748 | 0.583 | 641 |
| **MoMHa-scalar (仅标量反馈)** | 0.539 | 0.754 | 0.604 | 640 |

- **Joint > Two-phase**：单阶段联合优化比两阶段高出 **+2.7** 分，且每例少用 **95个token**。
- **Safety Skill至关重要**：移除安全技能导致安全得分从0.781降至0.716（-6.5分）。
- **Trace级反馈有效**：仅提供标量反馈会使安全得分下降至0.754，证明执行轨迹（traces）提供了关键因果信号。
- **Token项促进结构简化**：移除token惩罚会导致token消耗增加69，且在某些领域准确率反而下降。

---

## 4. 关键结论和发现

### 主要发现
1. **单阶段联合优化优于两阶段**：这是本文最核心的发现。联合优化允许proposer在早期就做出成本感知的结构性选择（如单次验证而非两次），而两阶段方法因第一阶段已冻结结构而无法发现此类优化。
2. **Harness质量是多维的**：仅追求accuracy会牺牲安全和效率。MoMHa证明了显式地将safety和tokens作为一等公民目标是必要的。
3. **Trace级反馈具有价值**：proposer能从详细的执行轨迹中提取标量分数无法提供的因果信息，尤其在安全敏感任务中更为重要。
4. **策略具有跨模型泛化能力**：在小型模型（如Haiku）上发现的优化策略，可以无缝迁移到大型模型（如Opus、GPT-5）上，表明这些策略捕捉的是任务结构而非模型特异性模式。

### 局限性
- **搜索成本较高**：每个领域需要约100次proposer API调用，虽然总token成本低于DSPy，但仍高于简单的prompt优化方法。
- **在推理密集型模型上泛化受限**：对于内部生成长链式思维（CoT）的模型（如GPT-5-mini, o4-mini, DeepSeek-R1），harness的token节省效果被掩盖，导致MoMHa优势减弱。
- **依赖LLM生成的数据**：部分能力数据集由LLM生成，可能引入噪声或偏差。

### 未来工作方向
1. 将方法扩展到 **SWE-Bench / LiveCodeBench** 等更具挑战性的代码生成任务。
2. 引入更丰富的优化目标，如 **latency**, **calibration**, **monetary cost**。
3. 探索 **proposer ensembles** 或多个proposer协作。
4. 实现 **在线harness精炼**，利用生产流量持续优化。
5. 探索在 **开放权重模型** 上的应用，结合activation steering等白盒技术。

</details>

---

### 13. [Up and Down the Abstraction Ladder: Code-Based Skills for Language Agents](https://arxiv.org/abs/2609.31076)

**Authors**: Bart{\l}omiej Cupia{\l}, Jens Tuyls, Maciej Wo{\l}czyk, Davide Paglieri, Martin Klissarov, Benjamin Eysenbach, Piotr Mi{\l}o\'s, Karthik R. Narasimhan  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.31076v1  

#### Abstract
Language agents struggle to act and learn in environments that require long sequences of low-level actions. Code-based abstractions can make these agents more productive by letting them invoke reusable skills instead of repeatedly selecting individual actions. The code handles recurring local decisi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Up and Down the Abstraction Ladder: Code-Based Skills for Language Agents 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
语言模型（**Language Models, LMs**）作为智能体（**agents**）在复杂、长视野环境（如软件工程、计算机操作、游戏等）中执行任务时，面临一个核心挑战：需要连续选择大量低层次（low-level）的原始动作（primitives），例如精确的鼠标点击、键盘输入或命令行指令。这种基于 **primitive action space** 的控制方式虽然提供了最细粒度的控制，但效率低下，推理成本高，且难以学习和探索。

此外，尽管高层级技能（high-level skills）可以提升效率，但它们往往是“有缺陷的抽象”（leaky abstractions）——当遇到超出其能力范围的情况时，仍需回退到原始动作。如何在**生产力**（productivity）和**灵活性**（flexibility）之间取得平衡，是本文要解决的核心问题。

### 提出了什么新方法或新思路
论文提出了 **CodeHack**，一个为 **NetHack** 和 **MiniHack** 游戏环境设计的开源代码技能库（code-based skill library），并系统研究了不同层级的动作抽象对语言智能体性能的影响。

其核心思想是构建一个可共享的控制层（shared control layer），支持三种控制模式：
- **Primitive-only**: 仅使用底层原始动作。
- **Skill-only**: 仅使用高层级语义技能（如 `explore`, `fight_melee`, `goto_room`）。
- **Mixed control**: 同时允许使用技能和原始动作，实现“上下抽象阶梯”（up and down the abstraction ladder）的灵活切换。

这些技能以 Python 函数形式实现，封装了复杂的动作序列，并通过自然语言描述供语言模型理解和调用。

### 相比现有方法的优势
- **系统性比较**：首次系统地比较了 primitives、skills 和 mixed control 在零样本（zero-shot）、监督微调（SFT）和强化学习（RL）三种范式下的表现。
- **真实可复现的技能库**：CodeHack 提供了一个可检验、可修改、可扩展的技能实现，而非隐式的、不可见的“潜技能”（latent skills）。
- **保留灵活性**：mixed control 设计允许智能体在技能失效时回退到原始动作，解决了技能覆盖不全的问题。
- **高效且低成本**：技能显著降低了推理成本（inference cost）和 token 消耗，同时提升了性能。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **NetHack Learning Environment (NLE)**：主实验平台，一个极其复杂、长视野的终端式 Roguelike 游戏，要求智能体探索超过 50 层的地牢，获取宝物并返回。完成该游戏被视为一项长期挑战。
- **MiniHack**：一个基于 NLE 构建的轻量级沙盒环境，用于受控测试特定行为（如导航、战斗、物品使用）。包含 7 个任务：
  - **Corridor-{R3, R5, R10}**：测试导航与探索。
  - **WoD-Hard-Full**：测试使用魔杖进行战斗。
  - **Quest-{Easy, Medium, Hard}**：组合导航、探索、物品使用和战斗。

### 实验设置和评估指标
#### 控制接口对比
- **Primitive-only**：直接选择 NetHack 的底层命令（如 `move north`, `quaff potion`）。
- **Skill-only**：从 78 个预定义技能中选择（NetHack）或任务相关子集（MiniHack）。
- **Mixed control**：可在技能和原始动作间自由选择。

#### 评估场景
- **Zero-shot prompting**：在 14 个不同规模的 LLM 上评估（包括 GPT-5, Llama, Gemma, Qwen 等）。
- **Supervised Fine-Tuning (SFT)**：使用 Gemma-4-31B 生成的教师轨迹对 Llama-3.1-8B-Instruct 和 Qwen-3.5-4B 进行微调。
- **Reinforcement Learning (RL)**：使用 PPO 算法训练上述两个模型。

#### 评估指标
- **MiniHack**：任务成功率（success rate）。
- **NetHack**：
  - **Score**：游戏内置得分。
  - **Progression**：基于人类轨迹映射的进展度量（BALROG metric）。
  - **Dungeon Level (Dlvl)**：达到的最大地牢层数。
  - **Inference Cost / Token Usage**：每回合的推理成本和 token 消耗。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与基线对比

#### 零样本（Zero-shot）结果（Table 1 & Table 5）
| 指标 | Primitives | Mixed | Skills | Skills vs. Primitives |
|------|------------|-------|--------|------------------------|
| **Progression (%)** | 0.69 | 1.88 | 1.98 | **+187%** |
| **Score** | 83.9 | 291.0 | 318.3 | **+279%** |
| **Dungeon Level** | 1.19 | 2.79 | 2.88 | **+143%** |
| **Cost per episode ($)** | 4.35 | 1.36 | 0.59 | **-86%** |
| **Tokens per episode (M)** | 6.72 | 3.54 | 1.74 | **-74%** |

- **技能显著提升性能**：在 14 个模型上平均，技能将游戏进展（progression）提高了近 **3 倍**，得分提高 **3.8 倍**。
- **大幅降低成本**：技能将每回合推理成本降低 **86%**，token 消耗减少 **74%**，因为大部分动作由 CPU 快速执行的代码技能完成，减少了昂贵的 LLM 调用次数（平均减少 5.1 倍）。

#### 强化学习（RL）结果（Table 2）
| 控制器 | 模型 | Dlvl 增益（vs. Primitives） |
|--------|------|----------------------------|
| **Primitive-only** | —— | 1.0x |
| **Skill-only** | 平均 | **7.2x** |
| **Mixed** | 平均 | **8.6x** |

- **技能加速学习**：在相同训练预算下，基于技能的智能体学习速度远超原始动作智能体，地牢深度增益高达 **7.2 倍**。
- **混合控制优势最大**：mixed control 在 RL 中表现最佳，兼具高性能和灵活性。

#### 消融实验：技能覆盖不全的影响（Figure 17）
- 当移除某一类技能（如探索、战斗）时：
  - **Skill-only** 性能急剧下降，尤其在缺少探索技能时几乎无法推进。
  - **Mixed control** 性能受影响较小，智能体可通过原始动作（如 `move`, `search`）弥补缺失技能，继续前进。
- 结论：**mixed control 对技能库的不完整性更具鲁棒性**。

---

## 4. 关键结论和发现

### 主要发现
1. **技能大幅提升性能与效率**：在长视野任务中，使用高层级代码技能（code-based skills）相比原始动作，能显著提升语言智能体的游戏进展、得分，并大幅降低推理成本和 token 消耗。
2. **混合控制是理想折衷**：允许智能体在技能和原始动作间切换的 **mixed control** 保留了技能的绝大部分性能优势（约 95%），同时提供了应对未知情况的灵活性，避免因技能缺失而完全失败。
3. **技能加速强化学习**：在 RL 中，基于技能的智能体学习速度远超基于原始动作的智能体，表明高层级抽象有助于更好的探索和信用分配。
4. **SFT 可有效初始化 RL**：使用技能控制器生成的专家轨迹进行 SFT，可显著提升最终 RL 策略的性能（Table 2 中 RL(SFT) 明显优于 RL(Base)）。

### 方法的局限性
- **技能库依赖性强**：当前技能库是手工设计的，必然不完整。智能体的表现受限于技能是否覆盖当前情境。
- **未解决技能自动构建**：论文假设技能库是给定的，未探讨如何让 LLM 自动发现、创建或优化技能。
- **决策路由开销**：mixed control 引入了额外的“路由问题”（routing problem）——智能体必须决定何时使用技能、何时使用原始动作，这可能导致低效尝试（如反复调用无效技能）。

### 未来工作方向
- **自适应技能学习**：让 LLM 能够识别重复失败模式，并自主编写、测试和精炼新的代码技能。
- **工具与记忆系统**：结合技能与信息查询工具（如记忆检索），减少对长历史的依赖。
- **动态抽象层级切换**：开发更智能的机制，让智能体能根据环境不确定性或任务需求，自动在抽象层级间切换。
- **通用技能框架**：将 CodeHack 的理念推广到其他复杂环境（如操作系统控制、机器人操作），建立通用的代码技能生态系统。

> **总结**：本文通过 **CodeHack** 库系统论证了“上下抽象阶梯”的价值——**技能提供效率，原始动作提供安全网**。这一范式为构建更强大、更高效的语言智能体提供了清晰路径。

</details>

---

### 14. [Inference-Time Target Speaker Unlearning in LLM-Based Automatic Speech Recognition](https://arxiv.org/abs/2609.30439)

**Authors**: Bo Su, Yueru Yan, Thai Le  
**Category**: cs.CL  
**Published**: 2026-09-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.30439v1  

#### Abstract
We introduce target-speaker unlearning ASR (TSU-ASR) task in a fully end-to-end framework for multi-speaker ASR and diarization. Given a multi-speaker utterance and a set of opt-out speakers who do not wish to have their speech transcribed, the task requires an ASR system to transcribe all speakers ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Inference-Time Target Speaker Unlearning in LLM-Based Automatic Speech Recognition

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
本文提出了 **Target-Speaker Unlearning ASR (TSU-ASR)** 新任务，旨在解决现代视频会议中隐私保护的关键挑战：  
- 在多说话人语音识别（multi-speaker ASR）场景下，允许部分参与者“选择退出”（opt-out），即其语音**不被转录为文本**，但仍保留其**说话时段和身份信息**（用于 diarization）。  
- 这与传统的 target-speaker ASR（只识别指定说话人）相反，TSU-ASR 要求系统**排除特定说话人**，同时准确识别其余所有说话人。

该问题具有现实紧迫性：
- 当前主流会议平台（如 Zoom、Teams）无法在推理时动态忽略某些说话人。
- 后处理过滤存在误删风险（尤其在重叠语音中），而重新训练模型成本高昂且不可行。

---

### 🚀 提出的新方法或新思路  
提出 **Enrollment-Conditioned Gating (ECG)** 模块，一种轻量级、可插拔（plug-and-play）的组件，适用于冻结的 LLM-based 多说话人 ASR 模型（如 TagSpeech）。

#### 核心设计思想：
- 利用 opt-out 说话人的短语音样本（enrollment audio）生成 **enrollment embedding**。
- 在推理时，通过一个轻量 MLP 计算每帧语音特征与 enrollment embedding 的匹配得分（matching score）。
- 使用该得分作为“门控信号”（gate），**动态抑制语义流（semantic stream）中的内容信息**，但**保留声纹流（voice stream）以维持 diarization 能力**。
- 公式如下：
  $$
  \tilde{s}_t = s_t \cdot (1 - \max\{g_t \odot e_k | e_k \in F\})
  $$
  其中 $g_t$ 是门控输出，控制对语义表示 $s_t$ 的抑制强度。

#### 关键创新：
- **推理时动态解耦**：无需重新训练主干模型，即可在 inference time 动态添加新的 opt-out speaker。
- **双路径分离处理**：仅干扰 content pathway，保持 speaker pathway 完整，实现“听得到谁在说，但不知道说了什么”。
- **端到端兼容性**：适配当前流行的 dual-stream LLM-based ASR 架构（如 TagSpeech + Qwen2.5-7B）。

---

### ⚖️ 相比现有方法的优势

| 方面 | 传统方案 | 本文 ECG |
|------|--------|---------|
| **灵活性** | 需重新训练或微调模型 | 冻结主干模型，仅训练轻量 ECG 模块一次 |
| **动态性** | 不支持运行时新增 opt-out speaker | 支持任意新 speaker 即插即用 |
| **隐私粒度** | 整体关闭 transcription 或后处理删除 | 精确屏蔽特定说话人内容 |
| **重叠语音处理** | 易误删非目标语音 | 可区分并保留其他说话人输出 |
| **计算开销** | 高（retrain 成本大） | 极低（仅 MLP 推理） |

此外，相比 machine unlearning 中常见的“去训练影响”范式（removing training data influence），ECG 更贴近实际需求——**不是忘记某个 speaker 的训练痕迹，而是实时屏蔽其语音内容**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **AMI-SDM**（英语）：单远场麦克风会议录音，共 2,607 条片段。
- **AliMeeting**（中文）：远场多通道会议数据，取第一通道，共 4,373 条片段。

#### 数据划分策略：
- 按说话人隔离：训练集与测试集无重叠 speaker → 验证对**未见说话人**的泛化能力。
- 测试阶段随机选取 25% 的测试 speaker 作为 opt-out 集合（AMI: 4/16, AliMeeting: 15/60），重复 10 次取平均。

---

### 🔬 实验设置
- **基础模型**：TagSpeech（基于两个 Zipformer 编码器 + Qwen2.5-7B LLM），参数完全冻结。
- **ECG 模块**：
  - 插入位置：语义编码器输出之后，projector 之前。
  - 结构：两层 MLP，输入包括帧级 voice feature、enrollment embedding、element-wise product 和 cosine similarity。
  - 输出：[0,1] 区间内的连续门控值，便于反向传播优化。
- **训练方式**：
  - 仅训练 ECG，主干模型冻结。
  - 正例：从音频中随机选一人作为 target，将其文本从 ground truth 中移除。
  - 负例：引入未出现在该段音频中的 speaker 的 enrollment 样本。
  - 训练损失：
    $$
    \mathcal{L} = \alpha \cdot \text{CE}(y, y^-) + \beta \cdot \text{BCE}(g, m)
    $$
    - CE：预测文本与期望输出之间的交叉熵。
    - BCE：门控 logits 与真实匹配标签之间的二元交叉熵。

---

### 📊 评估指标

#### （1）Content Leakage Rate (CLR)
衡量 opt-out speaker 的内容泄露程度：
$$
\text{CLR} = \frac{2 \cdot \text{LCS}(r(x), h(x))}{|r(x)|}
$$
- $r(x)$：opt-out speaker 的参考文本
- $h(x)$：完整生成 transcript
- LCS：最长公共子序列长度（考虑顺序但不要求连续）
- 分为两种：
  - **CLR-all**：所有词/字符
  - **CLR-rare**：仅文档频率 <1% 的罕见词/字符（减少通用表达干扰）

> ✅ 数值越低越好，表示 suppression 效果更强。

#### （2）Retained-Speaker Transcription Accuracy
- 英文 AMI：**cpWER-R**, **gWER-R**
- 中文 AliMeeting：**cpCER-R**, **gCER-R**
- 后缀 `-R` 表示仅针对 retained speakers 计算错误率
- 使用 cpWER/cpCER 时进行最优 speaker assignment 匹配

> ✅ 数值变化应尽可能小，表明对保留说话人无负面影响。

#### （3）分组分析（Segment Types）
将测试样本按说话人组合分为四类：
| 类型 | 描述 |
|------|------|
| **Retain-only** | 仅有保留说话人 |
| **Forget-only** | 仅有被屏蔽说话人 |
| **Mixed-overlap** | 两者共现且语音重叠 >0.05s |
| **Mixed-nonoverlap** | 两者共现但无显著重叠 |

---

### 🆚 基线方法对比
文中未直接比较多个外部 baseline，而是采用 **ablation-style 对比**：
- **Base Framework**：原始 TagSpeech 模型（无 ECG）
- **+ ECG Module**：加入 ECG 后的结果
- 所有改进均归因于 ECG 的引入。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2）

| Segment Type | Dataset | CLR-rare ↓ | CLR-all ↓ | cpWER-R / cpCER-R ↑ |
|--------------|---------|------------|-----------|-----------------------|
| **All** | AMI | 67.0 → **41.6** (-25.4%) | 72.3 → **48.2** (-24.1%) | 48.9 → **50.1** (+1.2%) |
| **All** | AliMeeting | 65.4 → **22.7** (-42.7%) | 73.6 → **27.3** (-46.3%) | 49.1 → **51.4** (+2.3%) |

> ✅ **结论**：ECG 显著降低 content leakage，同时对 retained speaker 几乎无损。

---

### 🔍 细粒度结果分析（Fig. 2 & Table 2）

#### ✔️ 在各类 segment 上的表现趋势一致：
- 所有含 opt-out speech 的组别（Forget-only, Mixed-overlap 等）中，**CLR 均显著下降**。
- 尤其在 **AliMeeting** 上效果更明显（最高降幅达 ~70%），说明中文环境下 suppression 更有效。

#### ❗ 特殊观察：
- **Forget-only 场景仍有 content 泄露**（AMI: 42.1%, AliMeeting: 16.9%），说明即使没有 overlap 干扰，也无法完全清除内容 → 表明信息仍可通过残余路径传递至 LLM。
- **Mixed-overlap 场景下 retained speaker 错误率上升**（AMI: cpWER-R 从 62.7→65.6），表明 suppression 机制可能轻微干扰共享时间帧的内容重建。
- **Retain-only 场景几乎无影响**（cpWER-R: 36.7→37.0），验证 ECG 不破坏正常识别流程。

---

### 🔁 消融实验（隐含在设计中）
虽然未明确列出消融表，但从以下设计体现 ablation 思想：
- 是否使用 enrollment embedding 控制门控？
- 是否联合优化 BCE 损失项？
- 是否使用连续门控而非硬掩码？

实验结果显示：
- 引入 ECG 后，**仅需极少量额外训练即可获得强大 suppression 能力**。
- 门控机制能有效学习 speaker pattern 匹配，并转化为概率性抑制行为。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Finding #1：Forget-only 的低 CLR 不足以证明成功**
   - 单纯输出空文本也能达到零泄漏，但会破坏 diarization 完整性。
   - 成功的标准是：**在保留 speaker activity 标注的前提下，精准去除其内容**。
   - ECG 实现了这一平衡。

2. **Finding #2：ECG 是 TSU-ASR 的强有力第一步**
   - 成功实现了 inference-time dynamic unlearning。
   - 学会利用 enrollment audio 匹配说话人模式，并通过 soft gating 抑制 content stream。
   - 模块简单、高效、可扩展，适合部署于大规模会议系统。

3. **跨语言有效性验证**
   - 在英文（AMI）和中文（AliMeeting）数据上均取得显著效果，尤其后者表现更优，显示方法具备良好语言适应性。

---

### ⚠️ 方法的局限性

1. **无法彻底消除 content leakage**
   - 即使在 Forget-only 场景，仍有约 40%~50% 的内容被还原 → 表明冻结的 LLM 仍能从残余信号恢复部分信息。
   - 可能需要更强的干预机制（如 activation rewriting）。

2. **对混合语音的干扰风险**
   - 在 Mixed-overlap 场景中，retained speaker 的 WER/CER 有所上升，提示 suppression 可能“伤及无辜”。

3. **依赖高质量 enrollment samples**
   - 若提供的 opt-out speaker voice sample 质量差或风格差异大，可能导致匹配失败，削弱 suppression 效果。

4. **未探索更多架构适配性**
   - 目前仅验证于 TagSpeech，尚不清楚是否广泛适用于其他 LLM-based ASR 框架（如 Speech-LLM、SpeakerLM 等）。

---

### 🔮 未来工作方向

1. **增强 suppression 强度**
   - 探索更深层干预机制，如 attention manipulation、prompt engineering 或 activation editing。

2. **提升鲁棒性**
   - 支持噪声环境、口音变异、短 enrollment（<3s）等更具挑战性的 setting。

3. **扩展应用场景**
   - 应用于 TTS 中的 speaker unlearning（防止克隆）、会议摘要中的 selective summarization。
   - 支持 multiple concurrent opt-out speakers 更高效融合。

4. **隐私-可用性权衡研究**
   - 建立 formal privacy metric 来量化“说话人遗忘”的程度，推动标准化 benchmark 建设。

5. **实时性与部署优化**
   - 将 ECG 模块进一步压缩，支持边缘设备低延迟运行，满足在线会议实时性要求。

---

## ✅ 总结

本文开创性地定义了 **Target-Speaker Unlearning ASR (TSU-ASR)** 新任务，回应了智能会议系统中日益增长的语音隐私诉求。提出的 **Enrollment-Conditioned Gating (ECG)** 模块以极低成本实现了推理时动态屏蔽指定说话人内容的能力，在 AMI 和 AliMeeting 数据集上验证了其有效性与实用性。尽管存在 residual leakage 和轻微干扰问题，但其 plug-and-play 设计理念为构建可信赖、可配置的语音 AI 系统提供了重要范式转变。

</details>

---

### 15. [Feeding BabyLMs Macaroni: Code-Switching Curricula Cause Cross-Lingual Convergence](https://arxiv.org/abs/2609.30535)

**Authors**: Dries Rooryck, Alex Cai, Yonatan Belinkov, David Alvarez-Melis, Kiant\'e Brantley  
**Category**: cs.CL  
**Published**: 2026-09-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.30535v1  

#### Abstract
Children in multilingual communities often code-switch, using multiple languages in a single utterance. Can we induce cross-lingual alignment in language models by training on code-switched text? We pretrain small decoder-only transformers on two 100M-word multilingual corpora: a base corpus formed ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Feeding BabyLMs Macaroni: Code-Switching Curricula Cause Cross-Lingual Convergence —— 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文探讨在**数据受限的多语言预训练场景下**（如 BabyLM 挑战），如何有效提升小规模语言模型（BabyLMs）的跨语言对齐能力与下游任务性能。传统方法依赖大量平行语料或单语混合，但缺乏对语言间表示空间对齐的显式建模。

特别关注的问题是：  
- 是否可以通过模拟人类多语儿童的语言习得方式（即 **code-switching, CS**）来增强模型的跨语言理解？  
- 这种策略是否能在有限数据预算（1亿词）下优于标准的单语混合训练？

### 提出了什么新方法或新思路
提出了一种基于 **code-switching curriculum learning** 的新型预训练范式，其核心思想如下：

- **合成 code-switched 数据**：利用 LLM（如 DeepSeek-V4-Flash）将原始单语文本（来自 BabyBabelLM）转换为两种类型的 CS 文本：
  - **Word-level CS**：在句子内部嵌入目标语言的内容词（名词、动词等）
  - **Sentence-level CS**：交替翻译整句，保持约一半原语言句子
- **三阶段 curriculum 训练顺序**：
  1. 先训练 **word-level CS**
  2. 再训练 **sentence-level CS**
  3. 最后训练 **monolingual documents**
  
  这一顺序模仿了人类学习第二语言的过程：从单词 → 句子 → 流利表达。

### 相比现有方法的优势
- **无需真实 CS 语料**：通过 LLM 合成高质量、语法合理的 CS 文本，突破了真实双语语料稀缺的限制。
- **更自然的数据增强**：相比基于词典替换的方法，LLM 生成的 CS 更符合语言学规律，避免“生硬拼接”。
- **诱导跨语言表示收敛**：实验证明，这种 curriculum 显著提升了模型对平行句的跨语言检索能力，尤其是在不同书写系统之间（如 English ↔ Chinese）。
- **性能可迁移**：提升不仅体现在零样本任务上，还在 fine-tuning 任务中表现出更强的泛化能力。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **主数据集**：[BabyBabelLM](https://huggingface.co/datasets/BabyLM-community/babylm-r) 数据集，包含：
  - English (~33.4M words)
  - Dutch (~35.1M words)
  - Chinese (~31.2M words)
- 总词量控制在 **~100M words**，符合 BabyLM 2026 挑战要求。
- 所有模型共享相同的 **tokenizer**（byte-level BPE, vocab size 16,384）。

### 实验设置
- **模型架构**：GPT-2-small（12层，768维隐藏层，12头注意力）
- **训练配置**：
  - Optimizer: Adam（无 weight decay）
  - Batch size: 16
  - Learning rate: warm-up 1%，然后 cosine decay 至 0
  - Context length: 1024
- **训练策略对比**：
  | 条件 | Corpus 类型 | 数据顺序 |
  |---|---|---|
  | Baseline | non-CS | shuffled |
  | Main Method | CS corpus | curriculum (word → sent → mono) |
  | Ablation Controls | 多种变体 | shuffled |

### 评估指标
#### 主要评估套件：BabyLM Evaluation Suite
- **Zero-shot 任务**：
  - **BLiMP / MultiBLiMP / ZhoBLiMP**：语法最小对立对判断
  - **HellaSwag, Winogrande, XStoryCloze**：常识推理
  - **Global PIQA**：跨文化常识问答
  - **XNLI, MNLI**：自然语言推断
  - **POS tagging**：词性标注（跨语言）
- **Fine-tuning 任务**：
  - **SIB-200**：主题分类（多语言）
  - **INCLUDE**：区域知识理解
  - **BMLAMA**：事实记忆检索
  - **TruthfulQA**：真实性检测
- **跨语言对齐度量（S6）**：
  - **Bitext Retrieval Precision@1 (P@1)**：给定源语言句子，在目标语言候选集中能否找到正确翻译作为最近邻（使用 CSLS 调整相似度）
  - 数据来源：FLORES+ dev set（997 对平行句）

### 基线方法对比
- **non-CS baseline**：仅使用原始单语文档，随机打乱顺序训练
- **shuffled CS**：相同 CS 数据，但不按 curriculum 顺序训练
- **其他控制组（ablation）**：
  - Word-level CS only
  - Sentence-level CS only
  - “Word salad”：随机替换非相关语言词（破坏语义连贯性）
  - “Document translation”：直接全文翻译（无混合）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 2 & Table 3）

| 条件 | Leaderboard Score (avg.) | Fine-tune Avg. | Zero-shot Avg. |
|---|---|---|---|
| non-CS / shuffled | 46.44 ± 0.25 | 42.27 ± 0.48 | 51.77 ± 0.17 |
| CS / shuffled | 46.55 ± 0.47 | 42.33 ± 0.79 | 51.95 ± 0.14 |
| non-CS / curriculum | 46.36 ± 0.37 | 42.26 ± 0.74 | 51.60 ± 0.22 |
| **CS / curriculum (Ours)** | **46.72 ± 0.25** | **42.81 ± 0.42** | **51.72 ± 0.13** |

> ✅ **结论**：只有在 **curriculum 顺序下使用 CS 数据**时，才观察到显著增益（+0.36 点领先），且统计显著（p=0.016 vs non-CS/curriculum）。

### 跨语言对齐结果（Figure 2 & Table 6）

| 语言对 | non-CS P@1 | CS P@1 (curriculum) | 提升幅度 |
|---|---|---|---|
| English → Dutch | 79.0% | 87.5% | +8.5 pp |
| English → Chinese | 5.5% | 61.4% | **+55.9 pp** ❗️ |
| Dutch → Chinese | 4.5% | 52.5% | **+48.0 pp** ❗️ |

> ⚠️ 注意：non-CS 模型在跨脚本语言对（如 EN↔ZH）几乎完全失败（接近随机），而 CS 模型实现大幅跃升。

### 消融实验结果（Ablation Studies）

| 条件 | Leaderboard Score | 英中 P@1 | 结论 |
|---|---|---|---|
| Word-level CS only | 46.59 | 48.1% | 有效，但弱于完整 curriculum |
| Sentence-level CS only | 46.70 | 46.6% | 接近最优，但仍低于完整流程 |
| Word salad (incoherent switching) | 46.08 | 4.2% | ❌ 无法提升对齐，说明**语义一致性至关重要** |
| Document translation | 46.56 | 25.3% | ❌ 不如同等比例的 CS，说明“混合”本身比“翻译”更重要 |

> 🔍 发现：**coherent code-switching 是关键**；单纯的 token 共现不足以建立对齐。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Code-switching curriculum 有效提升跨语言对齐**：
   - 尤其在**不同书写系统之间**（如拉丁文 ↔ 汉字），CS 训练使 bitext retrieval P@1 从 ~5% 提升至 >60%。
   - 表示空间中的平行句变得高度相似。

2. ✅ **训练顺序至关重要**：
   - 只有遵循 **word-level → sentence-level → monolingual** 的 curriculum 顺序，才能获得显著性能增益。
   - 若打乱顺序，则效果消失，说明这是一种**结构化的学习路径**而非简单数据增强。

3. ✅ **对齐具有泛化性**：
   - 即使某些词从未出现在 CS 上下文中（never-embedded words），其表示也因整体语言空间对齐而受益（Figure 4）。
   - 表示对齐在训练完 monolingual 阶段后仍得以保留（Figure 3）。

4. ✅ **LLM 合成 CS 数据可行且高效**：
   - 使用 instruction-tuned LLM 可生成语法合理、语义忠实的 CS 文本。
   - 优于基于规则或词典的替换方法。

### 方法的局限性
- **语言覆盖有限**：仅测试了 English, Dutch, Chinese 三种语言组合，未验证对低资源语言或多语言（>3）扩展的有效性。
- **依赖 LLM 生成成本高**：当前方法需调用 LLM 进行大规模数据合成，难以扩展到更大规模语料。
- **潜在偏差风险**：生成的 CS 分布可能偏离真实儿童语言使用模式，存在 domain shift 风险。
- **模型规模限制**：所有实验均基于 ~98M 参数的小模型，尚不清楚在更大 LLM 中是否依然有效。

### 未来工作方向
- 探索更低成本的 CS 合成方法（如小型模型蒸馏、模板生成）。
- 将该 curriculum 应用于更多语言对，尤其是 typologically distant 或 low-resource 语言。
- 研究 CS 如何影响模型内部机制（如 attention head specialization）。
- 在 instruction tuning 或推理阶段引入 gradual code-switching 以促进跨语言推理。

---

> 📦 **开源信息**：作者已公开代码、数据和模型：
> - GitHub: [https://github.com/drooryck/multilingual-macaroni](https://github.com/drooryck/multilingual-macaroni)
> - Hugging Face Models: [https://huggingface.co/drooryck/multilingual-macaroni-models](https://huggingface.co/drooryck/multilingual-macaroni-models)
> - Dataset: [https://huggingface.co/datasets/drooryck/multilingual-macaroni-corpus](https://huggingface.co/datasets/drooryck/multilingual-macaroni-corpus)

</details>

---

### 16. [The KV Cache Is the New Memory Wall](https://arxiv.org/abs/2609.30854)

**Authors**: Tejinder Singh  
**Category**: cs.DC  
**Published**: 2026-09-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.30854v1  

#### Abstract
Autoregressive LLM inference at long context is bounded by memory bandwidth, not arithmetic throughput, and the binding resource shifts from model weights to the Key-Value (KV) cache as sequence length grows. For Llama-3-70B in BF16, the 140 GB weight footprint exceeds the 80 GB HBM of a single acce...

---

### 17. [Rank-Reliable Teacher-Guided Fitness Approximation for Expensive Evolutionary Optimization: A TinyML Architecture Search Study](https://arxiv.org/abs/2609.30553)

**Authors**: Soumen Garai, Suman Samui  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.30553v1  

#### Abstract
Expensive evolutionary search does not always need an exact fitness estimate for every candidate. It often needs a reliable answer to a simpler question: which candidate is better? We address this need through Teacher-Guided Learning NSGA-II (TGL-NSGA-II), a low-fidelity framework for constrained Ti...

---

### 18. [Selective Amortization of Full-Budget Counterfactual Reasoning for Visual Token Communication](https://arxiv.org/abs/2609.30756)

**Authors**: Qinglei Qi, Zhihe Liang, Fengzhan Jing, Shenao Zhu, Lei Zhang, Chenyang Zhang, Shuqing He, Jia Guo  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.30756v1  

#### Abstract
Generative image communication transmits compact semantic tokens under a limited packet budget, where token selection directly affects the final reconstruction quality after the complete packet is decoded. However, accurately estimating the terminal value of every candidate token requires repeated r...

---

### 19. [PTC-Decoder: Towards Intelligent SLMs on Offline Resource-Constrained Edge Devices](https://arxiv.org/abs/2609.30836)

**Authors**: Minghui Yu, Ke Mu, Gang Wu  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.30836v1  

#### Abstract
Deploying small language models (SLMs) on offline, resource-constrained edge devices such as remote sensing satellites presents a fundamental challenge: their limited reasoning capacity hinders reliable execution of multi-step agent tasks requiring complex tool orchestration. Existing plan-solve par...

---

### 20. [Same Text, Different Numbers: The Divergence of LLM-Based Measures](https://arxiv.org/abs/2609.31013)

**Authors**: Hamid Boustanifar, Sasan Mansouri  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.31013v1  

#### Abstract
Researchers increasingly use generative large language models (LLMs) to convert corporate text into empirical variables. We examine the extent to which LLM-based textual measures are invariant to model choice using thirteen measures, including sentiment, management clarity, uncertainty, answer speci...

---

### 21. [AtomWorld-Mem: Memory-Restored World States for Long-Horizon Atomistic Evolution](https://arxiv.org/abs/2609.31133)

**Authors**: Tian Luo, Ruge Zhang, Haozhi Han, Yifrng Chen, Yunquan Zhang, Yunxin Liu, Ting Cao, Kun Li  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.31133v1  

#### Abstract
High-fidelity atomistic evolution over long timescales requires more than observing the current crystal configuration. Instantaneous atomistic snapshots are often incomplete: locally similar configurations can correspond to different hidden dynamical contexts, future event preferences, and waiting-t...

---

### 22. [Accounting for Bias Enables Sustainable LLM Evaluation](https://arxiv.org/abs/2609.31184)

**Authors**: Harshita Katoch, David Antony Selby, Gerrit Gro{\ss}mann, Sebastian Vollmer  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.31184v1  

#### Abstract
LLM-as-a-judge has become the de facto standard for scalable, subjective evaluation, yet current leaderboards compensate for systematic measurement bias by running ever more comparisons, an approach that is both statistically unsound and computationally wasteful. The root cause is an incomplete meas...

---

### 23. [Segment-Level Agentic Topic Modeling for Improved Data Exploration and Resource Efficiency](https://arxiv.org/abs/2609.31460)

**Authors**: Myeongjun Erik Jang, Antonios Georgiadis, Sae Young Moon, Fran Silavong  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.31460v1  

#### Abstract
Topic modeling is an effective technique for discovering hidden themes within documents and is widely used in text mining and data analysis across a variety of industry sectors. Recently, large language model (LLM)-based topic models have been emerged that prompt LLMs to generate topics then assign ...

---

### 24. [SEA-CLIP-Tiny: Efficient Multilingual Text-Vision Embedding for Southeast Asian Languages](https://arxiv.org/abs/2609.30739)

**Authors**: Puja Ahmad Habibi, Faiz Assabil Firdaus, Ashvanth S, Ekapol Chuangsuwanich, Pume Tuchinda, Peerat Limkonchotiwat  
**Category**: cs.CL  
**Published**: 2026-09-28  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.30739v1  

#### Abstract
Multilingual text-vision embedding models are essential for cross-lingual image-text retrieval, but Southeast Asian languages remain poorly supported due to the region's linguistic diversity and limited data and computing resources. In this paper, we introduce SEA-CLIP-Tiny, a compact multilingual t...

---

### 25. [Strategically Diverse Sampling for Self-Training](https://arxiv.org/abs/2609.31571)

**Authors**: Alexander Gurung, Esmeralda S. Whitammer, Mirella Lapata  
**Category**: cs.CL  
**Published**: 2026-09-28  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.31571v1  

#### Abstract
Many LLM training and inference methods, including RL and test-time scaling, depend on repeated sampling, but benefit only when the responses meaningfully differ. Self-training faces the same challenge: training data is typically constructed by sampling IID responses and filtering primarily for corr...

---

### 26. [KCensus: Synthesizing Latency-Optimal Consensus Fast Paths (Extended Version)](https://arxiv.org/abs/2609.31302)

**Authors**: Cl\'ement Burgelin, Antoine Murat, Gal Sela, Marcos K. Aguilera, Rachid Guerraoui  
**Category**: cs.DC  
**Published**: 2026-09-28  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.31302v1  

#### Abstract
Strongly consistent geo-replication often relies on fast paths to reduce latency in the common case of no failures or contention. Existing fast-path schemes, however, are ad hoc and restrictive: each corresponds to a point in a broad design space shaped by network topology, workload, and latency obj...

---

### 27. [Spectral Feedback for Test-Time Alignment of Protein Diffusion Models](https://arxiv.org/abs/2609.30456)

**Authors**: Shai Dickman, Mert Cemri, Landon Butler, Kannan Ramchandran  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.30456v1  

#### Abstract
Reward maximization alignment methods for discrete diffusion models have primarily focused on steering the reverse process, either by influencing token logits or by selecting favorable sequences at intermediate steps. These approaches largely treat inference as a unidirectional process, lacking mech...

---

### 28. [Pretrained ASR Pseudo-labeling for Noisy Police Audio](https://arxiv.org/abs/2609.30469)

**Authors**: Kaavya Chaparala, Su Huang, Stephen L. Miller, Rhiannon N. Miller, Anjalie Field  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.30469v1  

#### Abstract
Pretrained ASR systems perform poorly on noisy Broadcast Police Communication (BPC), hindering efforts to understand police decision-making. Pseudo-labeling offers an unsupervised path to improve ASR without expensive human labels, but the efficacy of this approach on very noisy domains is not known...

---

### 29. [Learning What to Skip: Counterfactual Credit Assignment for Efficient Multi-Agent LLM Workflows](https://arxiv.org/abs/2609.30734)

**Authors**: Jinfeng Xu, Zheyu Chen, Ziyue Peng, Zheng Lin, Shuo Yang, Jinze Li, Zheng Xing, Mengran Li, Victor C. M. Leung  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.30734v1  

#### Abstract
Multi-agent LLM workflows use planning, execution, verification, and summarization to improve task performance, yet the value of each component depends on the state already produced. Executing every component can waste computation or overwrite a correct intermediate answer. We formulate component om...

---

### 30. [ConsultMind:Towards Automated Diagnostic Consultation via Uncertainty-Aware Reasoning](https://arxiv.org/abs/2609.30796)

**Authors**: Xiao Sun, Yuming Yang, Yun Chen, Jiang Zhong, Junnan Zhu, Xinyi Jiang, Haoyang Zeng, Ruirui Chen, Yining Wang, Xinyu Zhou, Rong Tang, Kaiwen Wei  
**Category**: cs.AI  
**Published**: 2026-09-28  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2609.30796v1  

#### Abstract
Diagnostic consultation is an online sequential decision-making process in which clinicians gather evidence through patient interaction until a diagnosis is sufficiently supported. Automating this process requires adaptive inquiry and interpretable decisions. Bayesian networks offer a natural founda...

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

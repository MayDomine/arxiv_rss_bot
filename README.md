# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-14 07:04:00 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SPADE: Speculative Decoding for Precise and Low Cost Distributed Edge Cloud Inference](https://arxiv.org/abs/2608.13076)

**Authors**: Divya Jyoti Bajpai, Kishan Kumar Upadhyay, Manjesh Kumar Hanawal  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2608.13076v1  

#### Abstract
Large Language Models (LLMs) have achieved remarkable success in natural language understanding and generation, but their deployment is constrained by high computational demands. Deploying smaller LLMs directly on the edge can circumvent this, but with degraded accuracy. Deploying smaller cloud-base...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SPADE: Speculative Decoding for Precise and Low-Cost Distributed Edge-Cloud Inference 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）虽然在自然语言理解和生成任务中表现出色，但其高计算需求限制了在边缘设备上的部署。直接在边缘运行小型模型会牺牲准确性，而完全依赖云端推理则带来高昂的**per-token 计算成本**和延迟。

核心挑战是：  
> 如何在保证大模型输出质量的前提下，显著降低云端计算开销并提升推理效率？

---

### 🚀 提出的新方法：SPADE 框架
作者提出 **SPADE** —— 一种基于 **Speculative Decoding (SD)** 的分布式边云协同推理框架，其核心思想如下：

- **边缘端（Edge）** 部署一个轻量级的 **draft model**，快速自回归生成候选 token 序列（draft tokens）。
- **云端（Cloud）** 部署高性能的 **verifier model**（即目标大模型），对 draft tokens 进行**并行验证**。
- 只有被拒绝的 token 才触发 verifier 的修正；接受的 token 被保留，后续 generation 继续基于更新后的上下文进行。

该过程打破了传统 autoregressive decoding 的串行瓶颈，实现了“一次验证多个 token”。

---

### 🔍 相比现有方法的优势

| 对比维度 | 现有方法局限 | SPADE 改进 |
|--------|-------------|-----------|
| **Accuracy vs Efficiency 权衡** | 模型压缩（pruning/quantization/distillation）导致精度下降 | 完全保持 verifier model 输出分布，**零精度损失** |
| **部署灵活性** | 分层切分（layer-splitting）、早退机制（early-exit）需模型结构调整或训练 | **Plug-and-play 设计**，无需任何 retraining 或架构修改 |
| **通用性** | 复杂度感知路由（如 DIMEE）依赖数据集特定启发式规则 | 不依赖 dataset-specific heuristics，适用于多种 NLP 任务 |
| **成本控制** | 全云推理导致大量昂贵的 model calls | 显著减少 verifier 调用次数，**降低云资源消耗与费用** |

> ✅ **关键创新**：首次将 Speculative Decoding 成功应用于 **edge-cloud 分布式场景**，实现高效、低成本且高保真的 LLM 推理。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 用途 | 特点 |
|------|-----|------|
| **CNN/DailyMail** | 文本摘要任务 | 广泛使用的新闻摘要 benchmark，评估生成质量 |
| **Spec-Bench** | 多任务综合评测 | 包含六个子任务：<br>- Multi-turn Conversation<br>- Translation<br>- Summarization<br>- QA<br>- Mathematical Reasoning<br>- Retrieval-Augmented Generation |

---

### ⚙️ 实验设置

| 组件 | 配置 |
|------|------|
| **Draft Model (Edge)** | LLaMA-3.2-1B（12GB GPU，NVIDIA RTX 3080） |
| **Verifier Model (Cloud)** | LLaMA-3.1-8B（48GB GPU，NVIDIA RTX A6000） |
| **通信方式** | 边缘生成 d 个 draft tokens 后批量发送至云端一次性验证 |
| **超参数 d** | 控制每次生成的 draft token 数量（实验中调整以优化吞吐） |

---

### 📊 评估指标

#### 性能指标（↑ 越好）
- **Task Scores**（Spec-Bench）：由 Gemini-2.5-Flash-Lite 自动评分（1–5 Likert scale），涵盖 correctness, clarity, factuality 等维度
- **BLEU-1/BLEU-4**, **ROUGE-1/ROUGE-L**, **CIDEr-D**（CNN/DM）：标准文本生成评价指标

#### 效率指标（↓ 越好）
- **Mean Target Model Calls**：平均调用 verifier 模型的次数
- **Average Throughput (tokens/s)**：系统整体吞吐量
- **Cloud Runtime (×)**：相对于全云推理的时间比例（越小越好）

---

### 🔁 基线方法对比

| 基线 | 描述 |
|------|------|
| **Target Model** | 全模型部署于云端，作为性能上限（准确但慢且贵） |
| **Draft Model** | 小模型单独运行于边缘，速度快但精度低 |
| **SPADE (Ours)** | 结合二者优势，利用 SD 实现边云协同 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### 表格 I：Spec-Bench 多任务结果

| 指标 | Target Model | Draft Model | SPADE (Ours) |
|------|--------------|-------------|----------------|
| **Overall Score (↑)** | 4.45 | 3.39 | **4.38** |
| **Mean Target Model Calls (↓)** | 133.25 | 0.00 | **30.16** (-77.4%) |
| **Cloud Runtime (↓)** | 1.00× | — | **0.23×** |
| **Throughput (tokens/s) (↑)** | 2.43 | 3.91 | **3.25** |

> ✅ SPADE 在几乎所有任务上接近 Target Model 的表现，同时将 verifier 调用减少 **77.4%**

#### 表格 II：CNN/DailyMail 摘要任务

| 指标 | Target Model | Draft Model | SPADE (Ours) |
|------|--------------|-------------|----------------|
| **BLEU-1** | 23.76 | 22.33 | **23.39** |
| **ROUGE-L** | 24.32 | 22.49 | **23.92** |
| **CIDEr-D** | 2.50 | 1.15 | **3.19** |
| **Target Model Calls (↓)** | 127.30 | — | **30.79 (-76%)** |
| **Cloud Runtime (↓)** | 1.00× | — | **0.24×** |

> ✅ 生成质量几乎与 full model 持平，**无明显性能损失**，但云运行时间仅为原来的 **24%**

---

### 🔍 消融实验分析（关于 `d` 的影响）

- 图 2 展示了不同 **draft token 长度 d** 对 verifier 调用的影响：
  - 随着 `d` 增加，**target model calls 持续下降** → 更少的验证频率
  - 原因：更强的 draft-verifier 对齐带来更高的 token 接受率
  - 但过大的 `d` 可能增加边缘冗余计算风险（若接受率低）
- 最终选择通过小规模验证集经验确定最优 `d`

> 💡 发现：**draft 和 target 模型之间的 alignment 是决定效率的关键因素**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **SPADE 显著降低了云端计算负担**：
   - 减少 **约 76%~77% 的 verifier model calls**
   - 云运行时间降至 **原始的 23%~24%**
   
2. **输出质量与完整大模型等价**：
   - 理论上证明输出分布一致（依据 [12]）
   - 实验显示各项任务得分接近 Target Model，**无显著性能损失**

3. **边缘承担主要计算负载**：
   - 利用边缘设备完成大部分 token 生成
   - 仅关键纠错由云端处理，真正实现“智能分流”

4. **高度实用化设计**：
   - 无需 retraining
   - 即插即用（plug-and-play）
   - 支持跨任务、跨领域迁移

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **Draft-Verifier Alignment 依赖性强** | 若 draft model 与 verifier 差异过大，acceptance rate 下降，收益减弱 |
| **网络延迟敏感** | 若 edge-to-cloud 传输延迟高，可能抵消部分加速效果 |
| **不适合极低带宽环境** | 需定期传输 draft tokens，对通信有一定要求 |
| **未考虑动态负载调度** | 当前为静态配置，缺乏实时资源适配能力 |

---

### 🔮 未来工作方向

1. **自动调节 `d` 的机制**：根据上下文难度、acceptance rate 动态调整 draft 长度
2. **多轮 feedback loop 优化**：引入轻量反馈通道提升 alignment
3. **支持 streaming 场景下的低延迟推理**：如语音助手、实时翻译
4. **扩展到 vision-language 多模态模型**：探索 SD 在多模态边云协同中的应用
5. **构建 adaptive routing 策略**：结合 early-exit 与 speculative decoding 实现更灵活的任务分配

---

## ✅ 总结

SPADE 是一项面向实际部署的创新性研究，成功将 **Speculative Decoding** 引入 **edge-cloud 分布式推理** 架构，在不牺牲大模型准确性的前提下，大幅降低云端计算成本和延迟。其实验充分、设计简洁、效果显著，为未来大规模 LLM 在移动与边缘场景中的落地提供了**可扩展、低成本、高保真**的技术路径。

</details>

---

### 2. [Dual-Flow Transformers: Decoupling the Primary Prefill Path from Additional Decode Computation](https://arxiv.org/abs/2608.12385)

**Authors**: Liming Liu, Mingze Wang, Tuo Zhao  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.12385v1  

#### Abstract
As large language models serve more requests, cumulative inference cost is becoming increasingly important relative to one-time training cost. The two inference phases stress hardware differently: prompt prefill is parallel and typically compute-bound, whereas autoregressive decode is sequential and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Dual-Flow Transformers: Decoupling the Primary Prefill Path from Additional Decode Computation》总结**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
大型语言模型（LLM）在推理过程中存在两个阶段：**prompt prefill** 和 **autoregressive decode**，它们对硬件资源的需求截然不同：
- **Prefill 阶段**：并行处理所有 prompt token，计算密集型（compute-bound），受限于算力吞吐。
- **Decode 阶段**：逐个生成 token，顺序执行，内存带宽受限（memory-bandwidth-bound）。

传统模型通过增加宽度（width）或深度（depth）来提升能力，但这会同时增加 prefill 和 decode 的开销，无法灵活分配计算资源。  
**核心问题**：如何在不显著增加 prefill 成本的前提下，为 decode 阶段分配更多计算资源以提升生成质量？

---

### ✅ **提出了什么新方法或新思路**
提出 **Dual-Flow Transformer** 架构，实现 **prefill 与 decode 计算的结构性解耦**：

#### 主要设计思想：
- **Primary Flow（主路径）**：
  - 完整的因果语言模型路径。
  - 单独处理 prompt，构建并写入唯一的 **persistent KV cache**。
  - 决定整个 prefill 的计算量和状态存储。

- **Auxiliary Flow（辅助路径）**：
  - 只从 prompt 最后一个位置开始激活，仅用于 continuation prediction。
  - **不写入任何持久状态**（如 KV cache），也不影响主路径。
  - 在 decode 阶段与主路径并行运行，增加额外计算。

#### 关键机制：
- **权重共享**：Attention、MLP、Output 投影矩阵全部共享，减少参数增长。
- **独立嵌入 + 轻量耦合**：使用独立的 token embeddings，但通过 learnable coupling vectors（`a`, `b`, `c`）在 attention-query、MLP 中间和输出层进行跨流信息传递。
- **混合预测目标（Mixture Objective）**：
  - 最终预测分布是主路径 $p$ 和辅助路径 $q$ 的加权混合：
    $$
    \text{loss} = -\log(\alpha p(y) + (1-\alpha) q(y)), \quad \alpha = \text{clip}_{[0.5,1]}(D_p^2)
    $$
  - $\alpha$ 是基于主路径分布集中度（collision probability）动态调整的置信分数，确保主路径主导预测。

- **Router Replay（用于 MoE 扩展）**：
  - 在 MoE 模型中，辅助流复用主流选择的专家集合（expert indices），但使用自己的路由权重组合。
  - 保证引用的专家参数不变，避免扩展持久状态，同时允许双流分别应用专家。

---

### ✅ **相比现有方法的优势**
| 维度 | Dual-Flow | 其他方法（如 PHD、PLT、Hidden Decoding） |
|------|----------|----------------------------------------|
| **Prefill 成本控制** | ✅ 不变（仅主路径处理 prompt） | ❌ 多数需扩展序列长度或多流参与 prefill |
| **Persistent State** | ✅ 仅一个 KV cache | ❌ 通常引入额外缓存或状态 |
| **计算灵活性** | ✅ 可独立调节 decode 阶段计算量 | ❌ 计算随 prefill 同步增长 |
| **部署效率** | ✅ 支持 grouped execution，重用 weight/KV 访问 | ⚠️ 实现复杂，难以优化 |
| **训练信号有效性** | ✅ 辅助路径直接参与 loss，端到端可训 | ⚠️ 有些方法间接影响预测 |

> ✅ **本质优势**：首次实现了 **phase-decoupled computation** —— 将“继续生成”的能力作为可扩展资源，而不扰动原始 prompt 处理流程。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
- **FineWeb**：高质量网页文本子集，用于 dense 和 sparse LLaMA-style 模型训练。
- **modded-NanoGPT Track 3 设置**：用于 controlled token scaling 实验。

---

### ⚙️ **实验设置**
#### 模型架构：
- **NanoGPT 设置**：
  - 12 层，768 维度，head dim=128，vocab=50,257。
  - 序列长度 1024，global batch size=512。
  - 数据乘子 $D \in \{1,2,3,4,5,10,20\}$ 控制训练 token 数（最多 ~40B）。

- **Dense LLaMA-style**：
  - RMSNorm、Rotary Embedding、Gated MLP、Grouped-Query Attention。
  - 数据量固定为模型参数的 80 倍。

- **Sparse MoE（Qwen-style）**：
  - 包含 routed experts 和 shared experts。
  - 使用 router replay 或独立 routing 进行对比。

#### 评估指标：
- **Validation Loss**：主要评价指标，反映建模能力。
- **Training Compute Match**：比较相同 FLOPs 下 Dual-Flow vs. Standard Transformer。
- **Ablation Studies**：分析耦合机制、混合目标、多流扩展等组件作用。

#### 基线方法对比：
| 对比项 | 基线模型 |
|-------|---------|
| 主体性能 | Standard Transformer / MoE |
| 架构近似 | PHD-2（token duplication + hidden copy） |
| 消融实验 | Dual-Flow w/o coupling, w/o mixture, etc. |

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据**

#### ✅ **NanoGPT Token Scaling 结果**
- 在所有 $D$ 下，**Dual-Flow 均优于标准 Transformer**。
- 拟合 Chinchilla 缩放律得：
  - Transformer 渐近损失：**2.9416**
  - Dual-Flow 渐近损失：**2.9013** → 显著更低
- 图表显示性能差距随数据量增大而稳定保持。

#### ✅ **训练计算匹配实验（D=10 Dual-Flow vs D=20 Transformer）**
- Dual-Flow 每步约两倍 FLOPs（双流前向传播）。
- 当与 D=20 Transformer 终点进行 **training-FLOP 匹配**时：
  - Dual-Flow 仍达到 **更低 validation loss**。
  - 表明：将训练计算分配给两个交互流，比单纯增加训练 token 更有效。

#### ✅ **Dense LLaMA 模型缩放**
- 在多个规模（0.12B–0.5B）下测试：
  - Dual-Flow 在相同 token 预算下始终优于 baseline。
  - 性能增益随模型规模扩大而持续存在。

#### ✅ **Sparse MoE 实验**
- Dual-MoE（with router replay）在两种配置下均优于 standard MoE。
- **Router replay 几乎保留了独立 routing 的全部收益**，同时维持专家集合一致。

---

### 🔍 **消融实验结果**

| 变体 | 描述 | 结果 |
|------|------|------|
| **Minimal Dual** | 无耦合、仅辅助流预测（q-only） | 略优于 PHD-2，尤其在大数据下 |
| **+ a/b/c coupling** | 加入轻量级跨流耦合 | 小幅但一致提升 |
| **+ mixture readout** | 引入主辅混合预测 | 性能达到最优 |
| **Three Flows** | 扩展至三个辅助流 | 进一步降低 loss，验证可扩展性 |
| **Auxiliary-specific weights** | 辅助流拥有独立 dense 参数 | 比共享权重版本表现更好，说明专用容量有效 |

> 💡 **结论**：所有组件均有正向贡献，尤其是 **mixture objective** 和 **cross-flow coupling** 至关重要。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **Phase-Decoupled Computation 是可行且有效的**：
   - Dual-Flow 成功分离了 prefill 和 decode 的计算路径。
   - 实现了“只在 decode 阶段增加计算”而不影响 prefill 成本的目标。

2. **Decode-Time Computation 是一种可扩展的能力维度**：
   - 类似于 test-time scaling，但内置于模型结构中。
   - 更高效地利用空闲 decode 算力（如低批大小场景）。

3. **在 MoE 中实现 Phase-Specific Expert Allocation**：
   - 主流控制 prefill 专家数量 $k_1$，辅助流控制 $k_2$。
   - 形成三元权衡：**prefill cost ↔ decode cost ↔ predictive quality**
     - 固定 $k_1$，增加 $k_2$ → decode 更强，quality 提升；
     - 固定 $k_1 + k_2$，减小 $k_1$ → prefill 更省，decode 不变，仍可维持 high quality。

4. **Grouped Execution 提供系统优化机会**：
   - 双流可堆叠执行，共享 weight 和 KV 访问，提高 memory bandwidth 利用率。

---

### ⚠️ **局限性**
1. **训练成本翻倍**：
   - 训练时每个位置都需运行双流，FLOPs 约为标准模型的 2×。
   - 虽然推理更灵活，但训练代价更高。

2. **新增参数虽少但仍存在**：
   - 新增第二张 token embedding 表（vocabulary-scale）和少量 coupling vectors。
   - 不适合极度参数敏感场景。

3. **当前仅探索了“增加 decode 计算”方向**：
   - 尚未研究“减少 decode 计算”或动态跳过策略。

4. **尚未集成到大规模生产系统中验证延迟/吞吐**：
   - 实际部署中的工程收益有待实测。

---

### 🔮 **未来工作方向**
1. **扩展更多 auxiliary flows**：
   - 如 C.1 所示，three-flow 已展现潜力，未来可探索自适应激活机制。

2. **引入 auxiliary-specific dense layers**：
   - 如 C.2 所示，专用参数进一步提升性能，可结合稀疏化控制成本。

3. **动态控制辅助流激活**：
   - 根据 prompt 难度或不确定性决定是否启用辅助流（类似 adaptive computation time）。

4. **与其他 inference 优化技术结合**：
   - 如 speculative decoding、KV compression、quantization 等协同优化。

5. **探索非对称训练策略**：
   - 例如对辅助流采用不同的学习率或正则化方式。

---

## ✅ 总结

| 维度 | 内容 |
|------|------|
| **核心思想** | 提出 Dual-Flow Transformer，结构性解耦 prefill 与 decode 计算 |
| **关键技术** | 主辅双流、共享权重、独立嵌入、轻量耦合、混合预测、router replay |
| **核心优势** | decode 可扩展计算，prefill 成本不变，KV cache 唯一 |
| **实验结果** | 在 NanoGPT、LLaMA-dense、LLaMA-MoE 上全面优于 baseline，消融验证各组件有效性 |
| **重大意义** | 揭示了 **generation-time computation as a scalable resource** 的新范式，为 LLM 推理效率与质量平衡提供新路径 |

> 🎯 **一句话总结**：  
> **Dual-Flow Transformers 首次实现了“在不影响 prompt 处理的前提下，只为文本续写阶段注入额外智能”，为下一代高效、灵活的 LLM 架构开辟了新方向。**

</details>

---

### 3. [FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving](https://arxiv.org/abs/2608.12932)

**Authors**: Zekai Li, Yihao Liang, Hongfei Zhang, Jian Chen, Yesheng Liang, Zhijian Liu  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.12932v1  

#### Abstract
Vision-Language-Action (VLA) models promise to bring end-to-end reasoning to autonomous driving, but their computational cost remains far too high for real-time control. The core challenge is structural: VLA inference is not a single bottleneck but a cascade of four. Visual encoding wastes compute o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
Vision-Language-Action (VLA) 模型在端到端自动驾驶中展现出强大的推理能力，能够统一感知、语言级推理与轨迹预测。然而，其**计算开销极高**，导致推理延迟远超实时控制需求（如 Alpamayo 1.5-10B 在 RTX PRO 6000 上需 717ms），难以部署于实际驾驶系统。

该问题并非单一瓶颈，而是由四个结构性阶段共同造成的冗余计算：
- **Encode**: 视觉编码重复处理滑动窗口中的重叠视频帧；
- **Prefill**: 每步重新填充 KV Cache，未复用历史上下文；
- **Decode**: 自回归生成推理 token，效率低下；
- **Action**: 流匹配（flow-matching）去噪过程采用均匀步长，中间步骤计算冗余。

### **提出了什么新方法或新思路**
提出 **FlashDrive** ——一种**算法-系统协同设计框架**，针对上述四个阶段分别引入轻量级算法优化，并结合底层系统加速技术，实现端到端推理速度的复合提升。

#### **核心创新点：**
1. **Streaming Inference（流式推理）**
   - 利用驾驶场景中视频帧的高度时间连续性，仅对最新帧进行视觉编码，复用前序帧的 KV Cache。
   - 引入**流式注意力掩码**和**RoPE前缓存机制**，解决位置偏移问题。
   - 配套**轻量化微调策略**（仅微调 action expert），补偿因 KV Cache 近似带来的分布偏移。

2. **Speculative Reasoning（推测式推理）**
   - 提出使用 **DFlash**（基于扩散的并行 drafter）非自回归地生成整块 CoC（Chain-of-Causation）推理 token。
   - 利用驾驶领域推理熵低、token 内部强相关的特点，实现高接受率（平均接受长度 ~5.6 tokens）。
   - 显著减少 autoregressive decoding 步数。

3. **Adaptive-Step Flow Matching（自适应步长流匹配）**
   - 分析 flow-matching 过程中 velocity field 的结构：两端变化剧烈，中间平缓。
   - 在中间区域**缓存并复用 velocity 输出**，跳过部分网络前向传播。
   - 实现“计算集中于关键步骤”，减少无效迭代。

4. **W4A8 Quantization + System-Level Optimization**
   - 应用 **W4A8 量化**（4-bit weights, 8-bit activations），兼顾内存带宽与计算效率。
   - 结合 **CUDA Graph 编译** 和 **Kernel Fusion**，降低小核调度开销，提升执行效率。

### **相比现有方法的优势**
| 维度 | FlashDrive | 现有方法 |
|------|----------|--------|
| **覆盖范围** | 同时优化 encode/prefill/decode/action 四阶段 | 多为单阶段优化（如仅 KV Cache 或仅 decoding） |
| **通用性** | 可应用于任意具有相同 pipeline 结构的 VLA 模型 | 往往依赖特定架构修改 |
| **精度损失** | 几乎无损（minADE6 仅上升 0.08m） | 多数压缩/加速方法伴随显著性能下降 |
| **部署友好性** | 支持边缘设备（如 Jetson Thor）、多卡平台 | 多聚焦高端服务器环境 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **NVIDIA Autonomous Vehicle Dataset**（开源）
  - 用于训练 streaming fine-tuning 和 DFlash draft model。
  - 包含真实驾驶场景下的多视角图像、轨迹标注及语言指令。

### **实验设置**
- **模型**: 主要基于 **Alpamayo 1.5-10B**（SOTA 开源 VLA 模型），并在 Alpamayo 1 上验证泛化性。
- **硬件平台**:
  - 主测试平台：NVIDIA RTX PRO 6000
  - 跨设备验证：Jetson Thor, RTX 3090, RTX 4090, RTX 5090
- **输入配置**:
  - 滑动窗口：4帧 × 4视图
  - 推理轨迹数：1 或 6 条（用于规划多样性）

### **评估指标**
#### **Open-Loop Metrics**
- `minADE6@6.4s`: 六条预测轨迹中最小的平均位移误差（ADE）
- `minADE1@6.4s`: 单条最优轨迹的 ADE

#### **Closed-Loop Metrics（在 AlpaSim 中评估）**
| 指标 | 定义 |
|------|------|
| `Collision` | 是否发生碰撞（最大值聚合） |
| `Off Road` | 是否驶离可行驶区域 |
| `Wrong Lane` | 车头方向偏离车道中心 > 2/3 |
| `Plan Dev.` | 相邻计划间路径偏差（时间加权） |
| `Dtraj`, `Dloc` | 与真值轨迹的距离指标 |
| `Latency` | 每步 rollout 总耗时（含模型 + 渲染） |

### **基线方法对比**
- **Baseline**: 原始 Alpamayo 1.5-10B（FP16）
- **Ablation Baselines**:
  - +System Optimizations（CUDA Graph + Kernel Fusion）
  - +Streaming / +Speculative / +Adaptive / +Quantization 逐项叠加
- 对比对象还包括其他高效 VLA 方法（如 TinyVLA, EdgeVLA 等），但本文强调**无需架构改动即可加速现有模型**。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 方法 | 端到端延迟 (ms) | 控制频率 (Hz) | minADE1↓ | minADE6↓ |
|------|------------------|---------------|---------|----------|
| Alpamayo 1.5 | 717 | 1.4 | 1.705 | 0.767 |
| **FlashDrive (完整版)** | **151.4** | **6.6** | **1.573** | **0.844** |
| ➜ **加速比** | **4.7×** | — | ↓0.132m | ↑0.077m |

> ✅ **说明**：延迟从不可用级别（717ms）降至接近可用范围（151ms），且 `minADE1` 反而提升！

### **与基线方法的对比结果**
- **跨设备一致性加速**（见 Table 2）：
  - 在 **RTX PRO 6000** 上达 **4.7×** 加速（单轨迹）
  - 在 **6 轨迹模式下更显著**：最高达 **10.6×** 加速（RTX PRO 6000）
  - 成功在 **RTX 3090/4090** 上运行（原模型 OOM），体现显存优化优势
  - 在 **Jetson Thor** 边缘设备上实现 **4.0×~9.6×** 加速

- **闭环仿真表现（AlpaSim）**
  | 指标 | Alpamayo 1.5 | FlashDrive | 变化趋势 |
  |------|--------------|------------|----------|
  | Collision ↓ | 0.19 | **0.15** | ✅ 下降 |
  | Off Road ↓ | 0.41 | **0.32** | ✅ 下降 |
  | Plan Dev. ↓ | 0.24 | **0.16** | ✅ 显著改善 |
  | Wrong Lane ↑ | 0.45 | 0.51 | ⚠️ 小幅上升（交叉口敏感） |
  | Per-step Latency ↓ | 1150ms | **463ms (2.5×)** | ✅ 大幅提速 |

> 📌 **结论**：不仅推理更快，**闭环安全性更高、计划更稳定**。

### **消融实验结果**
| 阶段优化 | 延迟降幅 | 关键效果 |
|--------|----------|--------|
| **System Optimizations** | 1.40× | 所有阶段受益，decode 最明显 |
| **Streaming Inference** | encode ↓3.4×, prefill ↓3.2× | 减少 ~75% 视觉序列长度 |
| **Speculative Reasoning** | decode ↓2.9×（相对系统优化后） | 平均接受 5.6 tokens/block |
| **Adaptive-Step FM** | action ↓2.4× | 复用 4/8 diffusion steps |
| **W4A8 Quantization** | 整体再降 14% | 显存从 31.6GB → 18.3GB |

> 🔗 **复合效应**：各阶段优化叠加后总加速达 4.7×，远超任一单独优化。

---

## **4. 关键结论和发现**

### **主要发现**
1. **VLA 推理不是单一瓶颈，而是四重冗余的级联问题**：
   - 时间冗余（encode）、上下文冗余（prefill）、序列冗余（decode）、迭代冗余（action）需分别应对。
2. **每个阶段都存在“轻量级捷径”**：
   - 流式 KV 复用、扩散式推测解码、自适应去噪步长等均可大幅削减计算，且几乎不损精度。
3. **算法-系统协同带来复合增益**：
   - 算法减少冗余计算量，系统降低执行开销，二者相辅相成。
4. **加速反而可能提升性能**：
   - 如 streaming fine-tuning 起到正则化作用，降低预测方差；adaptive step 减少数值误差累积。

### **方法的局限性**
- **依赖固定 pipeline 结构**：目前适用于 encode-prefill-decode-action 类 VLA，对完全新型架构适配性待验证。
- **对极端动态场景鲁棒性未知**：如突发切入车辆是否会影响流式 KV Cache 的有效性。
- **Wrong Lane 指标轻微恶化**：表明在复杂路口方向判断上可能存在瞬时偏差，需进一步优化。

### **未来工作方向**
- 将 FlashDrive 思路推广至更多模态（LiDAR、雷达）融合的 VLA 模型。
- 探索动态调整 speculative block size 或 adaptive step 数量的在线策略。
- 构建面向 FlashDrive 的专用编译器，进一步自动化优化流程。
- 在真实车辆上进行实车部署与压力测试。

---

> 🔗 **项目主页**: [https://z-lab.ai/projects/flashdrive](https://z-lab.ai/projects/flashdrive)  
> 💾 **代码与 Checkpoint**: [GitHub - z-lab/flashdrive](https://github.com/z-lab/flashdrive)

</details>

---

### 4. [LoKiFormer: Locality-aware Attention with Decoupled Knowledge Memory for Efficient Large Language Model Pretraining](https://arxiv.org/abs/2608.12419)

**Authors**: Qiuwu Chen, Zimo Liu, Yuchen Li, Ying Sun, Yifan Zhang, Zhijie Qiu, Zeng You, Ryan Dong, Simeng Ma, Yaofo Chen, Mingkui Tan  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.12419v1  

#### Abstract
Large language models (LLMs) have achieved remarkable breakthroughs across various applications. However, their architectures remain inefficient in pretraining due to two main limitations: (i) self-attention lacks an explicit inductive bias for locality, leading to redundant modeling of sequence-int...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LoKiFormer: Locality-aware Attention with Decoupled Knowledge Memory for Efficient Large Language Model Pretraining

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前的 **Large Language Models (LLMs)** 在预训练阶段存在两个关键效率瓶颈：

1. **Self-Attention 缺乏局部归纳偏置**（local inductive bias）  
   Self-attention 机制通过全序列成对交互建模依赖关系，即使对于短距离的局部模式也需进行冗余计算，导致对局部信息建模效率低下。

2. **MoE 架构隐式耦合知识存储与计算路径**  
   Mixture-of-Experts (MoE) 虽然提升了模型容量，但其知识被隐式编码在专家网络参数中，知识检索过程间接且不透明，难以灵活访问和更新全局知识。

---

### 提出的新方法与新思路

为解决上述问题，论文提出 **LoKiFormer**，一种新型 LLM 架构，在标准解码器基础上引入两个专用模块：

#### （1）Local Fusion Attention (LFA)
- 在 **Multi-Head Latent Attention (MLA)** 前引入 **分组因果卷积**（grouped causal 1D convolution），对相邻 token 表示进行局部融合。
- 显式注入 **局部归纳偏置**，使 attention 层能专注于更复杂的长程上下文建模。
- 卷积核按注意力头数 `h` 分组，每个头学习独立的局部融合模式。

#### （2）Knowledge Memory Module (KMM)
- 引入一个 **可学习的参数化 key-value 内存结构**，将全局知识显式存储在可寻址的“知识槽”（knowledge fields）中。
- 解耦了 **知识存储** 与 **计算路径**，允许 token 直接查询并检索相关知识，提升知识访问的透明性和灵活性。
- KMM 输入来自 MLA 的压缩表示 `cKY`，输出与 MoE 输出融合。

---

### 相比现有方法的优势

| 维度 | LoKiFormer 优势 |
|------|----------------|
| **局部建模效率** | LFA 显式处理局部依赖，减少 attention 对局部信息的冗余建模，提升训练收敛速度 |
| **全局知识管理** | KMM 实现显式、可编辑、可解释的知识存储，支持直接检索，优于 MoE 的隐式知识编码 |
| **架构解耦性** | 知识存储（KMM）与计算（MoE）分离，增强模块化与可维护性 |
| **轻量化设计** | LFA 和 KMM 仅引入极少量额外参数（如 LFA 仅增加 0.41% 参数量） |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **预训练数据**：  
  - **Matrix Data Pile**：4.5T 高质量双语语料（英文 + 中文）
  - 英文部分：RedPajama、Dolma、SlimPajama 等
  - 中文部分：Skypile、ChineseWebText、Wanjuan、Yayi2 等，并补充大量新爬取中文网页内容
  - 包含代码、学术论文、书籍、政府报告等多领域数据

- **验证集**：从 Matrix Data Pile 的英文 Common Crawl 子集中随机采样 1%，约 18B tokens

- **SFT 数据集**：约 10B 高质量指令数据，源自公开指令跟随语料，经过五阶段清洗与去污染处理

- **评估基准**：
  - **语言理解**：MMLU（英文）、CMMLU（中文）、C-Eval（中文）
  - **推理能力**：HellaSwag、ARC-Challenge
  - **代码生成**：HumanEval（pass@1）
  - **数学推理**：GSM8K

---

### 实验设置与评估指标

- **模型规模**：构建了 1B 到 60B 参数的 LoKiFormer 家族，主实验基于 **LoKiFormer-7B**
- **训练框架**：Megatron-LM
- **硬件配置**：NVIDIA H200 或 Ascend 910B GPU
- **上下文长度**：2048（小模型）至 4096（大模型）
- **批量大小**：全局 batch size 16,384（主训练），1,024（消融实验）
- **优化器**：AdamW（β₁=0.9, β₂=0.95），混合精度训练，梯度裁剪为 1.0

---

### 基线方法对比

#### 尺寸相近模型（~7B）：
- Llama-3.1-8B、Qwen2.5-7B、Gemma-7B、InternLM2-7B、Phi-3-medium
- Mixtral-8×7B（MoE）、DeepSeek-MoE-16B

#### 前沿开源大模型：
- Llama-3.1-70B/405B、Qwen2.5-72B、Mixtral-8×22B、DeepSeek-V2.5、Hunyuan-Large

#### 前沿闭源模型：
- Claude-3.5-Sonnet、Gemini-1.5-Pro

---

## 3. 主要实验结果和性能指标

### 关键性能数据（LoKiFormer-7B）

| 任务 | 性能 |
|------|------|
| **MMLU** | **91.5** |
| **CMMLU** | **93.4** |
| **C-Eval** | **92.8** |
| **HellaSwag** | **89.5** |
| **ARC-Challenge** | **96.2** |
| **HumanEval** | **87.0** |
| **GSM8K** | **73.0** |

> 注：以上均为 zero-shot 或 few-shot 设置下的结果。

---

### 与基线方法的对比结果

#### ✅ 超越尺寸相近模型
- 在所有任务上显著优于所有 7B–8B 规模的 dense 和 MoE 模型
- 例如：MMLU 上比 Qwen2.5-7B 高出近 15 个点

#### ✅ 超越更大规模前沿开源模型
- 以 **7B 总参数（3B 激活）** 的规模，全面超越 **70B–405B** 的 dense 模型
- 甚至优于 **Hunyuan-Large（389B 总参数）**

#### ✅ 超越闭源顶尖模型
- **MMLU 91.5** 超过 Claude-3.5-Sonnet（88.3）和 Gemini-1.5-Pro（85.9）
- 在多个任务上达到或超过闭源模型水平

> **结论**：LoKiFormer-7B 以极小参数量实现 SOTA 性能，验证了架构设计的有效性。

---

### 消融实验结果

#### （1）预训练收敛速度
- **Baseline**：10k 步达到 PPL ≈ 31.82
- **+LFA**：9k 步达到相同 PPL → **1.11× 更快**
- **+LFA+KMM（LoKiFormer）**：**7.5k 步** 达到相同 PPL → **1.33× 更快收敛**

#### （2）下游零样本性能（MMLU）
- Baseline：17.9
- +LFA：21.2
- +KMM：22.9
- +LFA+KMM：**25.7** → 显示模块互补效应

#### （3）LFA 消融
- **卷积核大小 k=4** 效果最佳（PPL 30.59），优于 k=2（31.06）
- **分组策略 g=h**（与注意力头对齐）最优，PPL 30.49，优于 g=1（31.07）

#### （4）KMM 消融
- **知识场数量 F=64** 时取得良好平衡
  - F=32：PPL 30.78
  - F=64：PPL 30.35
  - F=128：PPL 29.88（继续下降）
- F=64 已捕获约 73% 的总收益，性价比高

---

## 4. 关键结论和发现

### 主要发现

1. **LFA 显著提升局部建模效率**  
   通过卷积融合局部信息，减轻 attention 的局部建模负担，使其更专注于全局上下文，加速收敛。

2. **KMM 实现显式、可解释的知识组织**  
   - 不同知识场在训练中自发专业化（如 Field 25 → 历史，Field 51 → 物理）
   - 可视化显示 **field-domain 关联性强**，支持定向知识检索
   - 知识场间接近正交（cosine similarity 接近 0），无明显干扰或坍缩

3. **架构高效且可扩展**
   - LFA 和 KMM 仅引入极小参数开销（< 3%）
   - 在 1B 至 60B 规模下均稳定训练，PPL 随规模平滑下降
   - 最大梯度范数始终 < 1.0，无发散现象

4. **推理效率高**
   - 解码吞吐量几乎不变（1299 vs 1294 tokens/s/NPU）
   - 首 token 时间仅轻微增加（132 → 136 ms）
   - LFA 不影响 KV Cache 机制

---

### 方法的局限性

1. **KMM 的静态性**  
   知识场在推理时固定，无法动态更新或在线学习新知识（虽保证稳定性，但也限制适应性）。

2. **依赖高质量预训练语料**  
   性能高度依赖 Matrix Data Pile 的大规模、高质量、双语平衡数据，数据偏差可能影响泛化。

3. **未探索跨模态扩展**  
   当前仅针对文本模态设计，尚未验证在图像、音频等多模态任务中的有效性。

---

### 未来工作方向

1. **扩展至多模态推理**  
   探索 LFA 如何建模跨模态局部依赖，KMM 如何统一存储和检索图文音知识。

2. **支持动态知识编辑**  
   设计可微分机制，允许在部署后安全地添加、删除或修改特定知识场。

3. **探索更大规模 KMM**  
   研究 F >> 128 时是否会出现知识场功能分化或层级结构。

4. **结合强化学习对齐**  
   当前仅使用 SFT 对齐，未来可探索结合 RLHF 进一步提升指令遵循与安全性。

---

> **Impact Statement**：LoKiFormer 通过提升预训练效率（1.33× 加速收敛），显著降低计算成本与能耗，推动更高效、可持续的 AI 系统发展。

</details>

---

### 5. [CABS+: Efficient and Scalable Model Merging via Conflict-Aware Sparsification and Adaptive Weight Allocation](https://arxiv.org/abs/2608.12842)

**Authors**: Yuchen Liu, Zongzhen Yang, Binhang Qi, Hailong Sun, Xiang Gao  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.12842v1  

#### Abstract
Model merging has recently attracted significant attention as a promising paradigm for constructing unified multi-task models without requiring additional retraining. However, parameter conflicts and knowledge interference across tasks often degrade merged-model performance. Prior work introduced Co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CABS+: Efficient and Scalable Model Merging via Conflict-Aware Sparsification and Adaptive Weight Allocation

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

模型合并（Model Merging）是一种无需额外训练即可将多个专家模型融合为统一多任务模型的有效范式。然而，该领域面临以下挑战：

- **参数冲突（Parameter Conflicts）**：不同任务的模型在参数空间中存在干扰，导致合并后性能下降。
- **高计算开销**：主流方法如 AdaMerging 和 WUDIMerging 依赖梯度优化，导致 GPU 内存消耗大、合并时间长。
- **优化偏差**：传统方法（如 CABS）采用网格搜索（Grid Search）确定缩放系数（scaling coefficients），其时间复杂度随任务数指数增长，且易被高性能任务主导，忽略低性能任务。

### **提出了什么新方法或新思路**

本文提出 **CABS+**，是对先前工作的增强版本，核心创新包括：

#### ✅ **Adaptive Weight Allocation (AWA)**  
一种基于 **无梯度优化**（gradient-free optimization）的自适应权重分配策略，采用改进的 **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** 来高效搜索最优缩放系数。

- 避免了反向传播所需的计算图构建和中间激活存储，显著降低 GPU 内存占用。
- 引入 **边界约束** 和 **非对称适应度函数**（asymmetric fitness function），防止优化过程被高损失任务主导。

#### ✅ **Relative Synergy Score (RSS)**  
提出一个新指标用于量化模型可合并性（model mergeability），定义为：
$$
\text{RSS} = \frac{\text{Score}_{\text{merged}} - \text{Score}_{\text{ideal}}}{\text{Score}_{\text{ideal}}} \times 100\%
$$
其中 $\text{Score}_{\text{ideal}}$ 是各单任务模型的平均性能上限。正 RSS 表示协同增益，负值表示破坏性干扰。

#### ✅ **系统性实证研究（Empirical Investigation）**  
首次系统分析影响模型合并效果的关键因素，涵盖六个维度：
- 优化历史（学习率、训练轮数）
- 任务异质性（task heterogeneity）
- 数据分布差异
- 模型架构（encoder-only vs decoder-only）
- 模型规模（7B → 70B）

---

### **相比现有方法的优势**

| 维度 | CABS+ 优势 |
|------|-----------|
| **效率** | 合并时间比 WUDIMerging 快近 **4倍**，内存仅为 AdaMerging 的 **<25%** |
| **性能** | 在 27 个任务上平均优于 AdaMerging 和 WUDIMerging 分别达 **16.97%** 和 **12.93%** |
| **稳定性** | 对任务数量、合并顺序、模型架构变化具有更强鲁棒性 |
| **可扩展性** | 支持在消费级 GPU（如 V100）上合并数十亿参数模型 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

覆盖 **语言模型** 和 **视觉模型** 多种场景：

#### 📚 大型语言模型（LLM）基准：
- **LLM Leaderboard**：基于 Mistral-7B，包含 ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K
- **Open LLM Leaderboard 2**：基于 Qwen-2.5-7B，包含 IFEval, BBH, MATH, GPQA, MUSR, MMLU-Pro

#### 📘 小型语言模型：
- **GLUE Benchmark**：使用 RoBERTa 和 GPT-2，在 CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2 上微调
- 扩展任务：RACE（阅读理解）、SQuAD（问答）

#### 👁️ 视觉模型：
- **ViT-B/32** 在 6 个视觉任务上的跨模态实验

所有 checkpoint 均来自公开平台（Hugging Face / FusionBench）。

---

### **实验设置和评估指标**

| 设置项 | 描述 |
|-------|------|
| **硬件环境** | V100 GPU（32GB），Crater 服务器平台 |
| **评估方式** | 每配置重复 3 次取均值 |
| **推理工具** | EleutherAI LM Evaluation Harness（batch size 自动调节） |
| **AWA 参数** | 种群大小 $K=6$，迭代次数 $G=50$，早停机制（连续 6 轮无改进则终止） |
| **剪枝比例** | 小模型 90%，大模型 75% |

#### ✅ 评估指标：
- **语言任务**：Accuracy, Success Rate（依榜单规定）
- **综合性能**：各任务平均得分（AVG）
- **相对提升**：相对于 CABS、AdaMerging、WUDIMerging 的 improvement (%)

---

### **基线方法对比**

| 方法 | 类型 | 是否需训练 | 特点 |
|------|------|------------|------|
| **Task Arithmetic** | 基础加法 | 否 | 简单平均 task vectors |
| **TIES-Merging** | 剪枝+符号裁剪 | 否 | 移除冗余/冲突参数 |
| **AdaMerging** | 测试时自适应 | 否 | 使用测试数据通过梯度下降学习系数 |
| **WUDIMerging** | 几何子空间优化 | 否 | 利用 task vector 子空间结构抑制干扰 |
| **CABS (prior work)** | 结构化剪枝 | 否 | 序列掩码 + n:m 剪枝减少重叠 |
| **CABS+ (ours)** | AWA + CABS | 否 | 无梯度优化 + 更优 fitness 设计 |

此外还比较了上述方法结合 Magnitude/DARE 剪枝后的变体。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔢 总体性能提升（Figure 3）：
- 相比 **AdaMerging**：**+16.97%**
- 相比 **WUDIMerging**：**+12.93%**
- 相比原版 **CABS**：平均提升 **+1.75% ~ +2.86%**

> 在多个任务组合下均保持领先，尤其在高异质性任务中表现更稳健。

#### 💾 效率对比（Table VIII）：

| 模型 | 方法 | AVG | Runtime | GPU Memory |
|------|------|-----|---------|-------------|
| RoBERTa (4 tasks) | AdaMerging | 80.52 | 12 min | 6.69 GB |
| | WUDIMerging | 81.79 | 3 min | 5.10 GB |
| | **CABS+** | **82.42** | **3 min** | **3.77 GB** |
| Mistral (2 vectors) | AdaMerging | 75.90 | 1 h | 66.70 GB |
| | WUDIMerging | 76.82 | 4 h | 12.04 GB |
| | **CABS+** | **76.71** | **~1 h** | **15.95 GB** |

> CABS+ 在性能接近最优的同时，**内存仅为 AdaMerging 的 24%**，**速度是 WUDIMerging 的约 4 倍**。

---

### **与基线方法的对比结果**

#### ✅ 在 LLM 上的表现（Tables I & II）：
- 在 Qwen2.5 系列上，CABS+ 显著超越 WUDIMerging（如 Table I 中 AVG 达 45.10 vs 44.71）
- 在 Mistral 上虽略低于 WUDIMerging，但仍优于其他所有方法，并接近理想模型（Ideal Model）

#### ✅ 在小型语言模型上的表现（Tables III–VII）：
- RoBERTa 上 4/6 任务合并：CABS+ 持续优于 CABS（+0.78% ~ +1.56%）
- GPT-2 上同样取得稳定增益，即使在强基线上仍有提升
- 图表显示：随着任务数增加，多数方法性能下降明显，而 **CABS+ 下降最缓，鲁棒性强**

#### ✅ 跨模态实验（ViT-B/32）：
- 平均准确率达 **82.50**，超过 CABS（81.08）和 WUDIMerging（82.30），验证通用性

---

### **消融实验结果**

#### ✅ AWA 组件有效性（Section IV.E）：
- 移除非对称 fitness 函数 → 优化不稳定，收敛慢
- 移除边界约束 → 容易陷入局部最优
- 结果表明 AWA 中的设计显著提升了搜索效率与稳定性

#### ✅ CABS 剪枝必要性：
- 若不进行 CABS 阶段的冲突感知剪枝，后续 AWA 优化难度增大，性能下降明显
- 验证了“先减少冲突，再优化权重”的两阶段设计合理性

---

## 4. 关键结论和发现

### **主要发现**

1. **模型可合并性受多种因素影响**：
   - ✅ **学习率适中最佳**：过低导致参数重叠过多引发干扰；过高导致过拟合，共享知识少
   - ✅ **训练轮次存在“黄金窗口”**：太少欠拟合，太多过拟合，中间阶段最利于合并
   - ✅ **任务越相似，越容易合并**（RSS 正相关）
   - ✅ **数据分布一致至关重要**：相同任务但不同领域（如 IMDb vs Financial）会导致严重性能退化
   - ✅ **模型架构影响剪枝效果**：
     - Encoder-only（RoBERTa）适合随机剪枝（DARE）
     - Decoder-only（GPT-2）更适合 magnitude pruning
   - ✅ **模型越大，越容易成功合并**：
     - 70B 模型出现正 RSS（+1.28%），表明存在协同效应
     - 归因于 **知识解耦**（knowledge decoupling）和 **参数冗余**

2. **无梯度优化更适合大规模模型合并**：
   - 梯度方法（如 AdaMerging）受限于显存瓶颈，难以扩展
   - AWA 实现 CPU-GPU 协同，仅用 GPU 进行前向推理，大幅降低内存压力

3. **CABS+ 实现了性能与效率的双重突破**：
   - 不仅性能领先，而且可在资源受限设备上运行
   - 适用于真实世界部署场景

---

### **方法的局限性**

- 当前 AWA 仍依赖启发式超参设置（如种群大小、步长），尚未完全自动化
- RSS 指标目前主要用于事后分析，尚不能直接指导事前模型选择
- 实验主要集中于同构模型合并，对异构架构或多模态融合支持有限

---

### **未来工作方向**

1. 探索 **多模态与异构模型合并**（multimodal/heterogeneous merging）
2. 开发基于 RSS 的 **预筛选机制**，实现“merge-before-try”
3. 将 AWA 扩展至动态在线合并场景
4. 结合稀疏训练进一步提升合并后模型的推理效率

---

## 总结

CABS+ 是一项在 **模型合并领域兼具理论深度与工程实用性的前沿工作**。它不仅提出了高效的 **AWA 优化框架** 和科学的 **RSS 可合并性度量**，还在 **27 个任务、5 类模型** 上验证了其卓越的性能与可扩展性。该研究推动了模型合并从“试错式调参”走向“科学化选型”，为构建高效、轻量、通用的多任务 AI 系统提供了重要路径。

</details>

---

### 6. [NAS-Driven Hardware Accelerator Exploration for Edge AI and Quantization Effects on the Pareto Space](https://arxiv.org/abs/2608.13293)

**Authors**: Eleftherios Mylonas, Angelos Kouprizas, Michael Birbas, Alexios Birbas  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.13293v1  

#### Abstract
Edge AI deployment demands neural architectures that are simultaneously accurate, computationally efficient, and hardware-deployable - a challenge addressed by hardware-aware Neural Architecture Search (NAS). While recent works incorporate quantization directly into the NAS loop, these approaches ex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NAS-Driven Hardware Accelerator Exploration for Edge AI and Quantization Effects on the Pareto Space

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

本论文针对以下三个关键挑战展开研究：

- **Post-Training Quantization (PTQ)** 对 NAS 发现的 Pareto 前沿结构的影响尚未被系统研究，尤其是在全搜索空间尺度上缺乏形式化的稳定性度量。
- 现有硬件感知 NAS 方法大多将量化嵌入搜索循环（如混合精度搜索），导致搜索复杂度急剧上升，并紧密耦合架构与量化设计，限制了灵活性。
- 缺乏一个端到端框架，能够将量化后的神经网络架构自动映射到可重构加速器（如 CGRA）并进行高效的 **Domain Space Exploration (DSE)**。

### 🚀 **提出了什么新方法或新思路**

提出了一种**三阶段硬件感知 NAS 流程**，解耦模型搜索、量化评估与硬件映射：

1. **Stage I: Hardware-Agnostic Frontend**
   - 在 NAS-Bench-201 上训练一个基于 Pareto rank 的 surrogate 模型，以 Accuracy 和 FLOPs 为双目标，筛选出初始 Pareto 候选集。
   - 使用改进的 HW-PR-NAS 架构编码方式提升 Kendall’s Tau (KT) 相关性。

2. **Stage II: Quantization Bridge**
   - 引入“量化桥”模块，使用 Brevitas 对候选架构执行 INT4 PTQ。
   - 进行 Pareto 重排序与预算感知过滤，并引入反馈机制：若无幸存架构则触发新一轮 NAS 搜索。
   - 首次在完整 NAS-Bench-201 的 15,625 个架构上系统分析 PTQ 对 Pareto 结构的影响。

3. **Stage III: Hardware-Aware Backend**
   - 基于开源工具 CGRA4ML 构建 DSE 环境，通过进化算法探索最优 CGRA 配置（PE 数量、SRAM 深度、AXI 宽度等）。
   - 使用 `predict_model_performance()` 分析性 oracle 快速估计延迟、PE 利用率等指标，避免 RTL 综合开销。
   - 设计三项目标归一化适应度函数联合优化 Latency、Idle Ratio 和 Array Area。

### 🔍 **相比现有方法的优势**

| 方面 | 本文优势 |
|------|--------|
| **方法论** | 提出“NAS → Quantize → Map”的解耦流程，降低搜索复杂度，提高部署灵活性；而现有方法（如 HAQ、APQ、QuantNAS）将量化纳入搜索，显著扩大搜索空间。 |
| **量化影响分析** | 首次提供对 INT4 PTQ 在全 NAS-Bench-201 空间上的形式化 Pareto 稳定性分析（Survival Rate, Flip Rate, KT 等）。 |
| **Surrogate 能力迁移** | 发现 FP32 训练的零样本（zero-shot）surrogate 在 INT4 搜索中表现优于专门训练的 INT4 surrogate，挑战了“必须重新训练量化代理”的常规认知。 |
| **硬件映射自动化** | 构建首个结合 QONNX-to-QKeras 转换层 + CGRA4ML + 自动 DSE 的全流程闭环系统，支持灵活的 reconfigurable accelerator 探索。 |

---

## 2. 核心实验方法和设置

### 📊 **使用的数据集**

- **NAS-Bench-201**：包含 15,625 个预训练的 cell-based 架构，在 CIFAR-10 上训练完成，提供每个架构的 Accuracy、FLOPs、Params 等 ground-truth 数据。
- 所有实验均基于该 benchmark 查询真实性能，确保可复现性和公平比较。

### ⚙️ **实验设置**

#### **量化配置选择**
- 测试 325 种 PTQ 配置（无微调），最终选定 **INT4 权重 + INT4 激活 + 无偏置量化**（Brevitas ID 6）作为主方案。
- 原因：尽管精度下降明显（如 Top1 Acc 从 ~90% 降至 ~26%），但两轮微调即可恢复至 83.53%，证明信息保留能力强且硬件成本低。

#### **Surrogate 模型**
- 使用改进版 HW-PR-NAS 中的 Pareto rank predictor，输入为架构编码向量，输出为 Pareto rank 预测。
- 训练目标：Listwise ranking loss，保持多目标排序一致性。

#### **搜索策略对比**
- **Random Search (RS)**：采样 300 次
- **Multi-Objective Evolutionary Algorithm (MOEA)**：250 次查询，种群大小 150，锦标赛选择，突变率 0.9
- 每种设置重复 50 次取平均值

#### **DSE 设置**
- CGRA 参数空间：
  - PE 行数 $ R \in \{8,10,12,16\} $
  - 列数 $ C \in [12,96] $（步长为卷积核宽度）
  - Weight SRAM 深度 $ \in [256,1024] $（步长 64）
  - AXI 宽度 $ w_{AXI} \in \{64,128\} $
  - 总约束：$ R \times C < MAX\_PEs $，有效组合约 1,200 种
- 使用进化算法优化归一化适应度函数：
  $$
  F = -\frac{\text{clocks}}{\text{clocks}_{ref}} - \alpha(1-\text{util}) - \beta \cdot \frac{R\times C}{\text{MAX\_PEs}} - \lambda \cdot \max(0, R\times C - \text{MAX\_PEs}) \times 10^2
  $$
  其中 $\alpha=0.2$, $\beta=0.1$

### 📈 **评估指标**

| 类别 | 指标 |
|------|------|
| **Surrogate 性能** | Kendall’s Tau (KT) Rank Correlation |
| **搜索质量** | Normalized Global Hypervolume (NGHV)，参考点设为 (0, 10M FLOPs) |
| **Pareto 稳定性** | 
| - Pareto Front Survival Rate (%) | 原始 FP32 Pareto 成员仍属于 INT4 Pareto 的比例 |
| - Dominance Flip Rate (%) | FP32 中 a > b，但在 INT4 中 b ≥ a 的比例 |
| - Pareto Rank Sensitivity | 排名变化程度的敏感性度量 |
| **硬件性能** | Clock cycles, PE Utilization, Idle Ratio, Array Area |

### 🆚 **基线方法对比**

- **Surrogate 对比**：
  - FP32 Zero-Shot Surrogate（未重训练，直接用于 INT4 预测）
  - Dedicated INT4-Trained Surrogate（在量化后数据上重新训练）
- **搜索策略对比**：
  - RS vs MOEA
- **量化前后对比**：
  - 原始 FP32 Pareto vs INT4 Quantized Pareto

---

## 3. 主要实验结果和性能指标

### 📊 **关键性能数据**

#### **表 III：NAS-Bench-201 性能与稳定性指标摘要**

| 指标 | RS (FP32 Zero-Shot) | RS (INT4 Trained) | MOEA (FP32 Zero-Shot) | MOEA (INT4 Trained) |
|------|---------------------|-------------------|------------------------|----------------------|
| **Normalized Hypervolume** | 0.5740 ± 0.2012 | 0.5113 ± 0.1846 | 0.6920 ± 0.0238 | 0.6481 ± 0.0719 |
| **Hypervolume Ratio** | — | 1.12 (RS) / 1.07 (MOEA) | — | — |
| **Relative Improvement (%)** | — | +12.26 (RS) / +6.77 (MOEA) | — | — |
| **Accuracy (%)** | 87.73 ± 2.36 | 88.05 ± 2.48 | 86.48 ± 0.99 | 87.49 ± 1.28 |
| **FLOPs (MFLOPs)** | 3.8 ± 2.3 | 4.5 ± 2.1 | 2.5 ± 0.3 | 3.0 ± 0.8 |

> ✅ **核心发现**：**FP32 zero-shot surrogate 在 NGHV 上全面超越 dedicated INT4 surrogate**，说明无需重新训练也能获得更优的 Pareto 覆盖能力。

#### **Pareto 稳定性分析**

| 指标 | 数值 |
|------|------|
| **Pareto Front Survival Rate** | 0% |
| **Dominance Flip Rate** | 21.73% |
| **KT-Rank Correlation (Ground Truth)** | 0.6655 |
| **Pareto Rank Sensitivity** | 0.2404 |

> ❗ 尽管 Pareto 前沿完全重组（生存率为 0%），但全局排名相关性仍较高（KT ≈ 0.67），表明 FP32 surrogate 仍具备良好迁移能力。

#### **Surrogate Transferability Analysis（表 II）**

| 场景 | KT 值 |
|------|-------|
| 原始 FP32 域 | 0.8352 |
| INT4 域（zero-shot，未重训练） | 0.7219 |
| INT4 域（fine-tuned） | 0.8219 |

> 🔁 表明 FP32 surrogate 在零样本下仅损失约 13.6% 的 KT 性能，远优于从头训练的小样本 INT4 surrogate。

#### **DSE 结果（表 IV）**

| Arch. ID | PEs (R,C) | SRAM Depth | Clocks (cycles) | PE Util (%) |
|---------|----------|------------|------------------|-------------|
| 7856    | (16,66)  | 576        | 53,605           | 42.61       |
| 8592    | (16,66)  | 576        | 118,690          | 45.8        |
| 6854    | (16,66)  | 576        | 74,995           | 41.4        |

> 💡 所有架构收敛至相同最优 CGRA 配置，验证了 DSE 的一致性；arch 8592 因含 `nor_conv_3x3` 导致计算量更大，时钟周期显著增加。

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **INT4 PTQ 导致 Pareto 前沿完全重组**  
   - 原始 FP32 Pareto 成员在 INT4 下全部被淘汰（Survival Rate = 0%）
   - 超过五分之一的支配关系发生翻转（Flip Rate = 21.73%）

2. **FP32 surrogate 可有效迁移到 INT4 搜索任务**  
   - 尽管存在分布偏移，FP32 zero-shot surrogate 在 normalized hypervolume 上优于 dedicated INT4 surrogate（+12.26% RS, +6.77% MOEA）
   - 原因推测：FP32 训练信号更干净、噪声更少，而 INT4 数据受量化误差干扰较大，影响 surrogate 学习质量。

3. **模型复杂度越高，对 INT4 量化越鲁棒**  
   - 低 FLOPs 模型（7–47 MFLOPs）精度波动大，中位数下降达 5%
   - 高 FLOPs 模型趋于稳定（median error → 0%），因其结构趋同于大量 `nor_conv_3x3` 操作，具有天然低通滤波效应，抑制量化噪声传播。

4. **提出的三阶段流程可实现高效自动化部署**  
   - 成功构建从 NAS → PTQ → CGRA 映射的完整闭环系统
   - DSE 能快速定位最优硬件配置，适用于边缘 AI 快速原型设计

### ⚠️ **方法的局限性**

- **依赖 NAS-Bench-201 的简化假设**：所有架构在相同条件下训练，实际场景中不同架构对训练策略敏感度不同。
- **INT4 PTQ 本身精度损失严重**：虽可通过微调恢复，但本文未包含微调成本分析。
- **CGRA4ML 当前仅支持整数量化（INT1/2/4/8）**，不支持浮点或自定义格式。
- **未考虑内存带宽、数据流调度等底层硬件瓶颈**，仅使用分析性 oracle 估算性能。

### 🔮 **未来工作方向**

1. 进一步优化 Stage I 的前端 surrogate，增强其对量化扰动的鲁棒性；
2. 将实证研究扩展到更复杂的混合精度量化方案（如 INT4/INT8 混合）；
3. 引入轻量级微调模块到 Quantization Bridge，形成 PTQ+FT 一体化流程；
4. 探索跨任务泛化能力，将方法推广至 ImageNet、NLP 等更复杂任务；
5. 结合 NAAS 类思想，尝试联合优化 Neural Architecture 与 CGRA Configuration。

--- 

> 📌 **一句话总结**：  
> 本文揭示了一个反直觉但重要的现象——**即使 INT4 量化彻底重塑了 Pareto 前沿，FP32 训练的零样本 surrogate 依然能在量化感知搜索中胜出**，为“先高精度搜索再量化部署”的轻量级 NAS 范式提供了强有力的理论与实证支持。

</details>

---

### 7. [MARCH: Scaling Recurrent Memory with Content-Routed State Anchors](https://arxiv.org/abs/2608.12435)

**Authors**: Ming Zhang, Kaisen Yang, Shu Yu, Ermo Hua, Ning Ding, Xia Hu, Bowen Zhou, Chaochao Lu, Youbang Sun  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.12435v1  

#### Abstract
Transformers owe much of their strong long-context retrieval capability to a token-level memory that grows with context length. This flexibility, however, incurs a quadratic computation complexity during training and a key--value cache that grows linearly during autoregressive inference. Recurrent a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **Recurrent 模型**（如 SSM、DeltaNet）虽然在推理时具有常数级内存开销（constant-memory decoding），但由于其将整个历史压缩到一个固定维度的 **recurrent state** 中，导致早期信息容易被后续更新覆盖或遗忘。这使得它们在长程依赖任务（如长文本理解、in-context retrieval）上表现不佳。

相比之下，**Transformer** 虽然通过 key-value cache 实现强大的长上下文检索能力，但其训练复杂度为 $O(T^2)$，推理时 cache 随序列长度线性增长，计算和存储成本高昂。

MARCH 的目标是：**在保持 Recurrent 模型高效解码优势的同时，显著增强其长程记忆能力**。

---

### 提出的新方法：MARCH（Memory-Anchor Routing across Context History）

MARCH 引入了一种新型的 **content-routed state anchoring** 机制，核心思想如下：

- **State Anchors（状态锚点）**：在处理序列过程中，周期性地将当前的 cumulative recurrent state 快照保存为“state anchor”，形成一个可随上下文增长而扩展的 **state anchor bank**。
- **Content-Conditioned Keys**：每个 anchor 关联一个由该 state 内容决定的紧凑 routing key，使模型能基于语义内容进行检索。
- **Query-Dependent Retrieval**：在每一步，当前 token 生成一个 routing query，对所有因果可见的 anchors 进行 attention-style 聚合，读取相关历史状态。
- **Residual Fusion**：将从 anchors 读取的历史信息与当前 recurrent state 的输出相加，保留原始路径并引入辅助记忆分支。

> ✅ **关键创新**：首次将 **content-based retrieval** 思想应用于 recurrent state 的历史版本，实现了对“过去记忆状态”的选择性访问。

---

### 相比现有方法的优势

| 特性 | Transformer | Linear Attention / GDN | MARCH |
|------|-----------|------------------------|-------|
| 训练复杂度 | $O(T^2)$ | $O(T)$ | $O(T)$ |
| 推理内存 | $O(T)$ | $O(1)$ | $O(T/C)$（C为anchor间隔） |
| 长程记忆能力 | 强（token-level cache） | 弱（单状态压缩） | 强（多状态锚点 + 内容路由） |
| 可扩展性 | 差（cache爆炸） | 好（固定状态） | 好（可控trade-off） |

- **优于 Gated DeltaNet 等线性注意力变体**：解决了 fixed-state bottleneck 问题，允许模型访问多个历史状态快照。
- **优于 Log-Linear Attention 等层级缓存方法**：采用内容驱动而非时间索引驱动的路由，更具灵活性和泛化性。
- **支持 zero-shot 长度外推**：由于 routing 机制不依赖位置编码或特定索引，在超过训练长度（如32K）仍表现优异。

---

## 2. 核心实验方法和设置

### 数据集

| 类别 | 数据集 |
|------|--------|
| **零样本常识推理** | LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-Easy/Challenge, OpenBookQA, CommonsenseQA |
| **长上下文理解** | LongBench（涵盖单文档QA、多文档QA、摘要、少样本学习） |
| **长程检索能力测试** | RULER 中的 Needle-In-A-Haystack (NIAH) 任务（包括 single/multi-needle，长度达 4K–32K） |
| **真实场景 in-context retrieval** | SQuAD, TriviaQA, SWDE, FDA, Natural Questions, DROP |

---

### 实验设置

- **预训练配置**：
  - 数据量：50B tokens（Long-Data-Collections）
  - 序列长度：16K
  - 模型大小：约 793M 参数（与 Gated DeltaNet 对齐）
  - 层数：21 层
  - Anchor 间隔：每 512 个文本 token 插入一个 anchor
  - Routing dimension：$d_r = 64$

- **优化器**：fused AdamW，batch size ~4.2M tokens，peak LR = 4e-4

- **评估方式**：
  - Zero-shot evaluation
  - 使用 LM-Evaluation-Harness 统一评测
  - 报告准确率（accuracy）或 normalized accuracy

---

### 基线方法对比

| 基线 | 描述 |
|------|------|
| **Transformer (21L / 24L)** | 全注意力模型，用于衡量上限性能 |
| **Gated DeltaNet (GDN)** | 当前最先进的线性递归架构 |
| **GDN + Log-Linear Attention** | 结合层级状态缓存的改进版 GDN |

> 所有模型使用相同架构深度、参数规模、训练数据和超参，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 📊 表1：零样本常识推理（平均准确率）

| 模型 | 平均准确率 |
|------|----------|
| Gated DeltaNet | 40.1 |
| GDN + Log-Linear | 40.0 |
| **MARCH** | **41.5** ✅ |
| Transformer (21L) | 41.3 |
| Transformer (24L) | 41.4 |

> ➤ MARCH 在所有8项任务中均超越 GDN 变体，并略胜于同等规模的 Transformer。

---

#### 📊 表2：LongBench 十二项任务平均得分

| 模型 | 平均得分 | 相对提升 |
|------|---------|----------|
| Gated DeltaNet | 11.9 | — |
| GDN + Log-Linear | 12.5 | +5% |
| **MARCH** | **14.9** | **+25%** ✅ |

> ➤ 在 multi-document QA 上提升尤为明显（如 2WikiMultihopQA 提升 32%，MuSiQue 提升 45%）。

---

#### 📊 图3 & 表2：NIAH（长程关联检索）性能

| 设置 | MARCH 表现 |
|------|------------|
| 4K–16K | 显著优于所有基线 |
| **32K（zero-shot 外推）** | 在所有6个任务上取得最佳成绩，部分任务保持完美准确率（如 S-NIAH-1），而其他模型（包括 Transformer）全部降为 0 |

> ➤ 展示了 MARCH 出色的长度外推能力和稳定的信息保留机制。

---

#### 📊 表3：in-context retrieval 平均准确率

| 模型 | 平均准确率 | 相对提升 |
|------|----------|----------|
| Gated DeltaNet | 19.2 | — |
| GDN + Log-Linear | 20.5 | +6.8% |
| **MARCH** | **23.3** | **+14%** ✅ |

> ➤ 在 SQuAD、SWDE、FDA 等真实检索任务上均有显著增益（最高达 117%）。

---

### 消融实验结果（Ablation Studies）

#### 🔍 Chunk Size（Anchor 间隔）影响（Table 4）

| Chunk Size | 性能趋势 |
|------------|----------|
| 256 | 更高精度但更高内存开销 |
| **512（默认）** | 最佳平衡点 ✅ |
| ≥1024 | 性能明显下降（anchors 过稀疏） |

> ➤ 支持“更密集锚点有助于回忆”但需权衡效率。

#### 🧩 Routing 设计消融（Table 5）

| 配置 | 影响 |
|------|------|
| $d_r=64$ vs $192$ | $64$ 更均衡；$192$ 提升 retrieval 但损害 generalization |
| **Top-4 Sparse Routing** | 性能接近 dense，大幅降低计算成本（尤其在长序列）✅ |
| 移除 Null Option | 所有指标下降 → 验证了“跳过无关历史”的必要性 ✅ |

> ➤ 支持 MARCH 的设计选择：compact key + sparse retrieval + null route 是高效且有效的组合。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **State Anchoring 有效缓解了 fixed-state bottleneck**：通过保存历史 recurrent states 快照，MARCH 成功扩展了 recurrent 模型的记忆容量。
2. ✅ **Content-Routed Retrieval 是关键机制**：基于内容的 routing 比基于时间索引的方法更具适应性和泛化能力，尤其是在长度外推场景下。
3. ✅ **无需牺牲效率即可获得强大 recall 能力**：MARCH 在保持 $O(T)$ 训练复杂度和较低推理延迟的前提下，实现了媲美甚至超越 Transformer 的长程检索性能。
4. ✅ **模块化设计兼容性强**：MARCH 可作为插件集成到现有的 recurrent 架构中（如 GDN），提升其表现而不改变主干结构。

---

### 方法的局限性

1. **固定周期 anchoring 不够智能**：当前采用固定间隔（如每512 token）创建 anchor，可能导致在状态变化缓慢区域冗余，在剧烈变化区域分辨率不足。
2. **未增加底层 memory 容量**：仅扩展了“可访问的历史状态数量”，但未提升单个 state 的表达能力或引入 specialized memory 分区。
3. **硬件实现仍有优化空间**：尽管已采用 fused kernel 优化，dense routing 在极长序列下仍有一定开销。

---

### 未来工作方向

1. **Adaptive Anchoring**：根据 state 更新幅度或 novelty 动态决定是否创建 anchor。
2. **Hierarchical Memory Architecture**：结合不同 temporal scale 的 memory 分区（短期、事件性、长期知识），并通过 hierarchical router 调度。
3. **External Memory Integration**：将 MARCH 的 anchor bank 视为 external memory，支持 test-time training 和 continual learning。
4. **Memory Consolidation & Eviction**：引入 memory budget 控制机制，自动合并或淘汰低价值 anchor。

---

## 总结

> **MARCH 提出了一种新颖且实用的范式——将 recurrent state 的历史版本视为可检索的记忆单元，并通过 content-based routing 实现选择性访问。它成功弥合了 recurrent 模型的效率与 Transformer 的长程记忆之间的鸿沟，在多项长上下文任务上实现了 state-of-the-art 的性能，同时保持了良好的可扩展性和训练效率。**

该工作为构建下一代高效、强记忆能力的 LLM 架构提供了重要思路。

</details>

---

### 8. [Multi-AUV Ad-hoc network-based Target Tracking: A Value Gradient Guidance Multi-Agent Diffusion Reinforcement Learning Approach](https://arxiv.org/abs/2608.12436)

**Authors**: Jiaao Ma, Chuan Lin, Guangjie Han, Shengchao Zhu, Qian Zhu, Ying Liu, Zhenyu Wang  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.12436v1  

#### Abstract
Multi-AUV ad-hoc network-based target tracking requires networked autonomous underwater vehicles (AUVs) to cooperatively track maneuvering targets under constrained acoustic communication, dynamic topology, and uncertain ocean disturbances. Although multi-agent reinforcement learning (MARL) enables ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**Multi-AUV Ad-hoc Network-based Target Tracking**（基于多自主水下航行器自组织网络的目标跟踪）中的三大挑战提出解决方案：
- **高维联合状态-动作建模困难**：传统MARL方法难以有效建模动态拓扑下的复杂连续协同动作分布。
- **训练不稳定**：依赖单一优化目标导致在通信中断或拓扑频繁变化时出现策略震荡、收敛缓慢。
- **采样过程中缺乏价值引导**：扩散模型反向去噪过程易受噪声影响，生成次优动作，降低决策一致性。

### 提出的新方法与思路
作者提出了两个核心组件：

#### （1）**VGG-MADiffRL**（Value Gradient Guided Multi-Agent Diffusion Reinforcement Learning）
一种新型多智能体强化学习算法，结合**扩散模型**与**值梯度引导机制**：
- 使用**扩散策略**替代传统的确定性Actor，提升对复杂、耦合连续动作分布的建模能力。
- 在反向去噪过程中引入**Critic网络输出的价值梯度**，引导动作向高回报区域演化。
- 设计**双目标联合优化机制**：同时最小化Q-guided损失和策略梯度损失，增强训练稳定性。

#### （2）**MDCA**（Multi-agent Diffusion-based Collaborative Architecture）
一个三层闭环控制架构，支持CTDE（Centralized Training with Decentralized Execution）范式：
- **全局智能控制层**：负责任务分解与调度（通过USV-CG网关）。
- **本地在线训练层**：执行策略学习与更新。
- **物理动作执行层**：各AUV基于局部观测生成控制指令。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **表达能力** | 扩散策略能更好捕捉多AUV间的强耦合动作关系，优于确定性策略（如MADDPG、MAPPO）。 |
| **训练稳定性** | 双Critics + 软更新 + 值梯度引导抑制过估计与振荡，收敛更快更平稳。 |
| **动作质量** | 值梯度在采样阶段主动“纠偏”，避免无效探索，尤其适合低带宽水声通信环境。 |

---

## 2. 核心实验方法和设置

### 数据集与仿真平台
- 使用开源水下仿真基准 **OceanGym** 构建实验环境。
- 自定义扩展模块以模拟：
  - 海流扰动（Navier-Stokes方程建模）
  - 声呐探测机制（主动声呐方程）
  - 动态网络拓扑与通信约束

### 实验设置
| 参数 | 配置 |
|------|------|
| AUV数量 | 4, 6, 8, 10 |
| 目标数量 | 2 或 3 |
| 场景规模 | 初始距离目标约3.5–5km |
| 每轮长度（Lep） | 400步 |
| 总训练轮数（Nep） | 4000 |
| 学习率 | 1e-3 |
| 折扣因子 γ | 0.95 |
| 缓冲区大小 | 100,000 |
| 批量大小 B | 256 |

### 评估指标
1. **Convergence Speed**：平均累积奖励随训练episode的变化曲线。
2. **Tracking Accuracy (%)**：单位时间内AUV群保持在目标追踪阈值内的比例。
3. **Mean Tracking Error (MTE)**：所有agent与目标间欧氏距离的均值。
4. **Error Standard Deviation (Error Std)**：MTE的标准差，反映轨迹稳定性。
5. **Ablation Study**：验证值梯度引导与扩散策略模块的有效性。
6. **Availability Evaluation**：Unity 3D高保真可视化验证实际运行效果。

### 基线方法对比
分为两类进行比较：

#### （1）通用MARL算法（Continuous Control）
- **MAPPO**, **MASAC**, **MAAC**, **MATD3**, **MADDPG**

#### （2）面向水下AUV场景的专用方法
- **DSBM**（Dynamic-Switching Based MARL）
- **MA-A3C**（Hierarchical Advantage-Attention A3C）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ 表1：**Tracking Accuracy 对比（%）**
| Method | 4A2T | 6A2T | 8A3T | 10A3T |
|--------|-------|-------|-------|--------|
| **VGG-MADiffRL** | **74.63±0.19** | **72.73±0.04** | **77.40±0.05** | **61.52±0.28** |
| DSBM | 55.83±0.47 | 43.83±2.78 | 38.58±1.86 | 35.21±1.50 |
| MA-A3C | 33.42±0.61 | 31.68±1.89 | 25.74±0.20 | 32.29±0.66 |
| MAPPO | 60.27±0.13 | 52.82±0.41 | 67.75±0.11 | 48.76±0.02 |

> 📌 **结论**：VGG-MADiffRL在所有场景下均显著领先，最高精度达 **77.4%**（8A3T），远超第二名MAPPO（67.75%）。

---

#### ✅ 表2：**Mean Tracking Error (MTE) 对比（km）**
| Method | 4A2T | 6A2T | 8A3T | 10A3T |
|--------|-------|-------|-------|--------|
| **VGG-MADiffRL** | **0.1329±0.0001** | **0.1159±0.0001** | **0.1099±0.0001** | **0.1524±0.0001** |
| DSBM | 0.1629±0.0009 | 0.1929±0.0210 | 0.1852±0.0048 | 0.1966±0.0006 |
| MAPPO | 0.1439±0.0002 | 0.1497±0.0004 | 0.1312±0.0004 | 0.2114±0.0001 |

> 📌 **结论**：MTE全面占优，误差最低仅 **0.1099 km**，说明其具备更高精度的持续跟踪能力。

---

#### ✅ 表3：**Tracking Error Std 对比（km）**
| Method | 4A2T | 6A2T | 8A3T | 10A3T |
|--------|-------|-------|-------|--------|
| **VGG-MADiffRL** | **0.1755±0.0000** | **0.1818±0.0000** | **0.1748±0.0002** | **0.1938±0.0001** |
| DSBM | 0.1917±0.0003 | 0.1837±0.0080 | 0.1952±0.0030 | 0.1924±0.0005 |
| MAPPO | 0.1898±0.0001 | 0.1816±0.0002 | 0.1975±0.0002 | 0.2369±0.0001 |

> 📌 **结论**：Error Std最小且波动极小（部分为±0.0000），表明系统具有卓越的**动态稳定性与鲁棒性**。

---

### 收敛速度分析
- 如图4所示，在四种不同规模场景中，**VGG-MADiffRL均实现最快收敛**。
- 尤其在大规模场景（10A3T）中，其他方法（如MASAC）甚至出现性能下降趋势，而VGG-MADiffRL仍稳定上升。

### 消融实验结果（Ablation Study）
如图6所示，移除任一关键模块都会造成性能下降：
- **Without Critic Guidance**：失去值梯度引导后，收敛变慢，最终回报降低。
- **Without Diffusion Module**：替换为确定性策略后，表达能力受限，无法适应动态协作需求。

> 📌 **结论**：**值梯度引导机制**与**扩散策略架构**均为不可或缺的核心创新。

---

## 4. 关键结论和发现

### 主要发现
1. **扩散模型可用于多智能体连续控制**：相比传统确定性策略，扩散策略能更灵活地建模高度耦合的动作空间。
2. **值梯度可在采样阶段提供显式引导**：将Critic信号嵌入反向去噪过程，可显著提升动作质量和训练效率。
3. **双目标联合优化增强稳定性**：Q-guided loss与policy gradient loss协同作用，缓解非平稳性带来的训练难题。
4. **分层架构适配AUV ad-hoc网络特性**：MDCA实现了从全局规划到本地执行的高效协同，适用于动态拓扑与受限通信环境。

### 方法的局限性
1. **计算开销较高**：扩散模型需多步迭代去噪，推理延迟大于传统Actor-Critic结构。
2. **未显式建模能量消耗**：当前reward函数未考虑能耗均衡，可能导致某些AUV过早耗尽能源。
3. **依赖相对稳定的局部观测**：极端通信丢包情况下，局部online training layer可能失效。

### 未来工作方向
1. **优化水下避障机制**：减少AUV碰撞风险与设备损伤。
2. **平衡能量消耗**：设计能耗感知的任务分配与路径规划策略，延长系统续航。
3. **增强对不稳水声通信的鲁棒性**：构建显式应对突发链路中断的容错控制框架。

---

> ✅ **总体评价**：  
> 本论文提出的 **VGG-MADiffRL + MDCA** 框架在**动态性、不确定性、资源受限**的水下多AUV协同环境中展现出卓越性能，不仅在多个指标上超越SOTA方法，也为**Diffusion-based MARL** 在真实工程场景的应用提供了重要范例。

</details>

---

### 9. [Behavioral Reprogramming of Open-Weights Models: Cognitive Plasticity and Alignment Bounds](https://arxiv.org/abs/2608.13069)

**Authors**: Lucia Mal\'i\v{c}kov\'a  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.13069v1  

#### Abstract
Large language models (LLMs) are predominantly aligned to function as passive, sycophantic assistants. We challenge this default paradigm by empirically evaluating the cognitive plasticity of open-weight architectures when subjected to rigorous behavioral reprogramming. Our objective is to induce a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前主流的大型语言模型（LLMs）普遍通过 **Reinforcement Learning from Human Feedback (RLHF)** 或 **Direct Preference Optimization (DPO)** 对齐为被动、迎合用户的“助手”角色（sycophantic assistant），这种默认范式严重限制了其在需要主动干预、批判性思维或行为引导等场景中的应用（如心理辅导、决策支持）。  
本文旨在挑战这一范式，探索如何对**开源权重模型（open-weights models）进行行为重编程（behavioral reprogramming）**，使其具备**主动、苏格拉底式（Socratic）的对话能力**，即高频提出反问、引导用户反思。

此外，现有研究缺乏对模型“认知可塑性”（cognitive plasticity）——即模型接受根本性行为转变的能力——的系统量化，且参数高效微调（PEFT）常伴随“对齐税”（alignment tax），导致语言连贯性下降。

---

### **提出了什么新方法或新思路**
1. **定义并量化“认知可塑性”（Cognitive Plasticity）**  
   首次提出并实证测量不同架构在行为重编程中的结构性抵抗程度，建立跨架构的行为适应基准。

2. **构建“主动型苏格拉底人格”（Proactive Socratic Persona）**  
   通过两阶段训练流程实现：
   - **Structural Fine-Tuning (SFT)**：使用精心设计的小规模多语言行为语料库，打破默认的顺从行为模式。
   - **Direct Preference Optimization (DPO)**：在低秩子空间中直接优化偏好，将“主动提问”设为优选响应，解耦行为与表层语法。

3. **提出严格计算约束下的最优训练边界**  
   在大规模超参搜索（405个HPC任务）基础上，确立了 **LoRA rank、训练轮数（epochs）、学习率** 等关键参数的数学收敛边界。

4. **验证零样本跨语言人格迁移（Zero-Shot Cross-Lingual Persona Transfer）**  
   探索行为特征是否能跨越语言边界迁移，并揭示其结构性限制。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **目标行为** | 被动响应、知识提供 | 主动提问、引导反思 |
| **训练效率** | 多依赖全量微调或复杂奖励建模 | 参数高效（PEFT + DPO），仅需数千GPU小时 |
| **资源消耗** | 高算力、大数据集 | 小数据集（<1500样本）、低rank LoRA（r=16）即可有效 |
| **可解释性** | 黑箱式对齐 | 明确界定过拟合边界与收敛窗口（e ∈ [2,3]） |
| **泛化能力** | 局限于训练语言/任务 | 实现部分语言的零样本人格迁移 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **结构微调数据集（SFT Dataset）**  
   - 规模：1,458 对高质量、人工清洗的多语言对话对（涵盖斯洛伐克语、英语、德语、法语、西班牙语、意大利语、葡萄牙语）
   - 内容：聚焦心理逃避、拖延、情绪疲劳等真实人类-AI摩擦场景，旨在瓦解默认的顺从行为。
   
2. **偏好优化数据集（DPO Dataset）**  
   - 规模：440 个偏好对 $(x, y_w, y_r)$
   - 构成：每个提示 $x$ 匹配一个“优选”的主动提问响应 $y_w$ 和一个“劣选”的冗长顺从响应 $y_r$
   - 示例：  
     - 用户说：“我明天开始锻炼。” → 模型应回应：“星期一？你打算做什么？”（反问）而非“太好了！坚持就是胜利！”（附和）

3. **评估语料（Evaluation Corpus）**  
   - **对抗性压力测试矩阵（Adversarial Stress-Testing Matrix）**：18 个高密度心理情景 × 7 种语言 = 126 个推理任务
   - 场景类型：幽默/逃避、共情、直接性、危机识别、哲学追问

---

### **实验设置**
- **硬件平台**：Leonardo 超级计算机（EuroHPC JU），使用 NVIDIA A100-SXM-64GB 节点
- **总计算量**：约 50,000 GPU 小时
- **模型架构**：
  - Llama-3.1-8B-Instruct
  - Mistral-7B-Instruct
  - Qwen3-14B（斯洛伐克适配版）
- **微调技术**：
  - **LoRA**：应用于注意力投影层（q_proj, k_proj, v_proj, o_proj）和MLP层（gate_proj, up_proj, down_proj）
  - **量化**：4-bit NF4（BitsAndBytes） + bfloat16 计算精度
  - **优化器**：PyTorch AdamW（排除paged版本以保证确定性）
  - **调度器**：余弦退火学习率调度
  - **最大序列长度**：1024 tokens
  - **有效批量大小**：4（micro-batch=1, accumulation steps=4）

---

### **评估指标**
| 指标 | 定义 | 用途 |
|------|------|------|
| **Conditional Perplexity (PPL)** | $\text{PPL}(Y|X) = \exp\left(-\frac{1}{N}\sum \log P(y_i|x, y_{<i})\right)$ | 衡量语言连贯性和灾难性遗忘 |
| **Proactive Question Rate (QR)** | $QR = \frac{1}{|D_{test}|} \sum \mathbf{1}[\text{endswith}(y, ?)]$ | 核心行为指标：生成以问号结尾的比例 |
| **Average Response Length (Avg Len)** | 平均输出词数 | 衡量简洁性，避免冗余填充 |
| **Validation Loss** | 交叉熵损失 | 判断收敛与过拟合 |

---

### **基线方法对比**
- **Vanilla Base Models**：未经指令微调的基础模型（如 Llama-3-8B-base）
- **Instruction-Tuned Only**：仅经过标准指令微调的模型（如 Llama-3.1-8B-Instruct）
- **LoRA-only Adapter**：仅用LoRA进行SFT，无DPO优化
- **DPO Baseline**：在非最优超参下执行DPO（如 r=32, e=5）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | Eval Loss | PPL | Trainable % | QR (Exp.6) | Avg Len |
|------|----------|-----|------------|-----------|--------|
| **Qwen3-14B-SK** | 0.346 | **1.414** | 0.78% | — | — |
| **Llama-3.1-8B-Instruct (Optimized)** | 0.7856 | 2.19 | 0.92% | **48.5% (EN)** / **60.0% (ES)** | **<5 words** |
| **LoRA-only Adapter** | ~0.93 | ~2.52 | 0.91% | 21.4% | ~95 words |
| **Pre-DPO Model** | — | — | — | <25% | >90 words |

> ✅ **最佳配置**：`LoRA rank r=16`, `α=32`, `dropout=0.1`, `lr=2e-4`, `epochs=3`, `β=0.15`

---

### **与基线方法的对比结果**
- **vs. Vanilla Base Models**：
  - Base模型在相同设置下表现极不稳定（median eval loss ≈ 1.21 vs. Instruct ≈ 0.93），存在严重梯度干扰和灾难性遗忘。
  - 结论：**指令微调是稳定低资源行为重编程的前提条件**。

- **vs. LoRA-only Adapter**：
  - 仅有SFT的模型仍保持“啰嗦助手”风格（平均95词），QR仅21.4%。
  - 加入DPO后，QR提升至32.0%-60.0%，且平均长度压缩至**3.22词**，实现真正的行为解耦。

- **vs. Larger Models (Qwen3-14B)**：
  - Qwen3-14B 在单样本评估中达到最低PPL（1.414），显示更强的认知可塑性。
  - 但在**批量推理时出现OOM**，无法部署于边缘节点，实用性受限。

---

### **消融实验结果**

#### **(1) LoRA Rank 消融（Experiment 1 & 8）**
- **r=8**：容量不足，无法捕捉跨语言抽象，性能天花板明显。
- **r=32**：过度参数化，快速过拟合，eval loss 上升（0.938 @ full data）。
- **r=16**：**最优平衡点**，在所有数据密度下均取得最低PPL和loss。

#### **(2) Epoch 消融（Experiment 5）**
- 所有配置均呈现 **U型验证曲线**：
  - **e=2~3**：达到最优（min eval loss ≈ 0.919）
  - **e>3**：验证损失上升，训练损失趋近于0 → **严重过拟合**
- 结论：**训练轮数必须严格控制在 e ∈ [2,3]**，超出即引发分布漂移。

#### **(3) 学习率与Rank交互效应**
- 低学习率（5e-5）导致收敛缓慢（avg loss=0.9756）
- 高学习率（2e-4）配合 r=16 可加速收敛至 0.9409
- 有效学习率公式：$\eta_{\text{eff}} = \alpha / r$，需动态匹配

#### **(4) 超参网格搜索（Experiment 8）**
- 总计 **405个HPC任务**，验证全局最优：
  - 最佳组合：`r=16`, `lr=2e-4`, `e=3`, `dropout=0.1`
  - 达到 **mean eval loss = 0.9277 ± 0.0162**, **PPL = 2.529 ± 0.043**

---

## **4. 关键结论和发现**

### **主要发现**
1. **认知可塑性可被量化且受控于结构边界**  
   模型能否成功转变为“苏格拉底式”角色，取决于其架构、训练策略与数据密度的精确匹配。

2. **指令微调是必要前提（Instruction-Tuning is Mandatory）**  
   未经指令微调的Base模型缺乏内在的对话路由机制，在低资源PEFT中会浪费参数容量去学习基础语法，导致失败。

3. **存在严格的训练收敛窗口：e ∈ [2,3]**  
   超出此范围将不可避免地导致过拟合并破坏泛化能力，即使训练损失持续下降。

4. **LoRA rank 存在“黄金值”：r=16**  
   过低则欠拟合，过高则过拟合；r=16 是多语言行为迁移的最佳折衷。

5. **DPO 成功解耦行为与语法**  
   在低秩子空间中应用DPO（β=0.15）可有效剥离“主动提问”这一深层行为，而不损害语言质量。

6. **零样本跨语言人格迁移具有选择性**  
   - **成功迁移**：西班牙语（QR=60.0%）、英语（48.5%）→ 与源语言（斯洛伐克语）同属印欧语系，形态相近
   - **失败迁移**：德语、葡萄牙语（QR=0.0%）→ 形态差异大或tokenization碎片化严重

7. **行为压缩优于内容扩展**  
   成功的重编程不仅在于“说什么”，更在于“不说什么”——彻底消除冗余填充，维持平均回复长度 <5 词。

---

### **方法的局限性**
| 限制 | 描述 |
|------|------|
| **语言覆盖有限** | 当前仅支持7种欧洲语言，对亚洲、非洲语言的迁移效果未知 |
| **依赖高质量标注** | DPO阶段需高度专业化的人工构造偏好对，难以规模化 |
| **硬件瓶颈** | 大模型（如14B）虽性能优，但KV缓存过大，难以部署于边缘设备 |
| **Tokenization Fragmentation** | 分词器对稀疏语言支持差，影响LoRA路径激活一致性 |
| **安全风险** | 极简反问可能在敏感情境中被视为冷漠或攻击性，需额外护栏 |

---

### **未来工作方向**
1. **引入多语言DPO锚定（Multilingual DPO Anchoring）**  
   在多个目标语言上显式进行偏好优化，提升零样本迁移鲁棒性。

2. **实时多模态整合（Real-Time Multimodal Integration）**  
   结合语音、表情、生理信号，构建更自然的主动交互代理。

3. **自主多智能体协调（Autonomous Multi-Agent Coordination）**  
   让多个“苏格拉底式”模型相互辩论、校准信念，提升推理深度。

4. **轻量化边缘部署方案**  
   探索蒸馏、稀疏化、动态路由等技术，使该框架适用于移动端或嵌入式系统。

5. **伦理与安全性增强**  
   引入情感识别模块，动态调节提问强度，在危机场景自动切换为共情模式。

---

> 🔍 **一句话总结**：  
> 本研究证明，在严格计算约束下，通过对**指令微调模型**应用**r=16 LoRA + DPO（β=0.15）**，并在**2~3个epoch内完成训练**，可高效、稳定地将LLM重塑为**简洁、主动、情境感知的苏格拉底式对话者**，为个性化认知数字孪生的部署提供了可复现的技术路径。

</details>

---

### 10. [Prof-K: Probabilistic One-Pass Filtering for Efficient Top-k Selection](https://arxiv.org/abs/2608.12573)

**Authors**: Tadeusz Dziarmaga, Witold Sikora, {\L}ukasz Struski, Jacek Tabor, Marcin Mazur  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.12573v1  

#### Abstract
Top-k selection is a fundamental computational primitive with applications spanning databases, information retrieval, signal processing, and modern machine learning workloads, including sparse activations and attention pruning. As data sizes grow, existing approaches become inefficient: exact method...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Prof-K: Probabilistic One-Pass Filtering for Efficient Top-k Selection

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Top-k selection 是现代机器学习系统中的基础计算原语，广泛应用于稀疏激活、注意力剪枝、MoE 路由和 Sparse Autoencoders（SAEs）等场景。随着张量规模 $N$ 的增长，尤其是在 **large-$N$, small-to-moderate-$k$** 的稀疏场景下，传统 exact top-k 方法（如 PyTorch 的 `topk`）面临严重的效率瓶颈：

- **高内存带宽开销**：必须扫描整个输入并进行全局排序或选择。
- **同步成本高昂**：在 GPU 上，memory movement 成为主要瓶颈。
- **近似方法不可靠**：现有 approximate 方法依赖于对输入分布的假设，在 heavy-tailed 或 adversarial 输入下表现退化。

---

### 提出了什么新方法或新思路
本文提出 **Prof-K** —— 一种 **概率性单次过滤算法（probabilistic one-pass filtering）** 来高效实现 Top-k 选择。

#### 核心思想（见 Figure 1）
1. **Sampling**：从 $N$ 个元素中均匀采样一个大小为 $S$ 的子集。
2. **Threshold Estimation**：基于样本估计一个保守阈值 $T$，作为 $(k/N)$-quantile 的下界估计。
3. **Filtering**：单次遍历全量数据，仅保留 $\geq T$ 的元素到一个小的候选缓冲区（candidate buffer）。
4. **Refinement**：在小缓冲区上运行 exact top-k 得到最终结果；若失败则 fallback 到 full top-k。

该过程将原本对 $N$ 元素的 exact selection 转换为对远小于 $N$ 的候选集的操作。

---

### 相比现有方法的优势

| 维度 | Prof-K 的优势 |
|------|----------------|
| **效率** | 在 large-$N$, small-$k$ 场景下显著降低计算和内存开销，尤其适用于 GPU 内存受限环境。 |
| **理论保证** | 提供 **distribution-agnostic** 的概率正确性保证（即不依赖输入分布），适用于 heavy-tailed、多峰甚至对抗性输入。 |
| **可调精度-速度权衡** | 用户可通过调节 failure budget $\epsilon$ 控制 recall 和 buffer overflow 概率，支持灵活的 accuracy-speed trade-off。 |
| **互补性** | 可与现有优化 kernel（如 RadiK、Dr. Top-k）结合使用，进一步提升性能。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **合成数据集**：涵盖多种分布以验证 distribution-agnostic 性质：
  - Uniform
  - Standard Normal
  - Heavy-tailed Pareto 分布
- **真实 ML 工作负载**：
  - **BatchTopK Sparse Autoencoders (SAEs)**：使用来自 `EleutherAI/pythia-70m-deduped` 模型第1层 MLP 的激活值训练 SAE。
  - 数据维度：$N = B \cdot d_{\text{dict}} = 4096 \times 12288 = 50,331,648$，$K = B \cdot k = 131,072$

---

### 实验设置和评估指标

| 设置项 | 描述 |
|--------|------|
| **硬件平台** | - 高端：NVIDIA DGX H100 / A100<br>- 消费级：RTX 3060 |
| **实现方式** | 基于 Triton 实现 Prof-K，与 native `torch.topk` 和 RadiK 对比 |
| **参数配置** | 默认 failure budget $\epsilon = 10^{-3}$，通过 Algorithm 1 自适应选择 $S, t, M$ |
| **评估指标** | - Wall-clock latency（kernel 级 & end-to-end）<br>- Speedup 倍数<br>- Auxiliary GPU memory usage<br>- Reconstruction quality（NMSE, CE degradation）<br>- Sparsity behavior（active latents） |

---

### 基线方法对比
- **PyTorch `topk`**：工业级标准实现，提供 exact 结果。
- **RadiK [11]**：最新的 radix-based 并行 GPU top-k 算法，针对浮点表示优化。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 场景 | Prof-K 表现 |
|------|------------|
| **Synthetic Benchmarks (DGX H100)** | - 相比 `torch.topk`：**1.5× – 10× speedup**<br>- 相比 RadiK：**最高达 5.9× 加速**<br>- 尤其在 $N \geq 2^{24}$, $k$ 较小时优势明显 |
| **Memory Efficiency** | - Prof-K 辅助内存仅为 $O(M)$，其中 $M \approx ck$ ($c$ 小常数)<br>- RadiK 需要 $O(N)$ 额外空间 → 易发生 OOM |
| **SAE Training (DGX A100)** | - **Top-k kernel 时间**：从 2.56ms → **1.13ms**（**2.27× 加速**）<br>- **End-to-end step time**：30.73ms → **29.42ms**（**1.044× 加速**）<br>- **总训练时间减少约 4.25%** |

---

### 与基线方法的对比结果

| 对比项 | Prof-K vs Baseline |
|-------|--------------------|
| **Latency Scaling w.r.t $N$** | Prof-K 的延迟随 $N$ 增长更平缓（接近常数），而 baseline 受限于 memory bandwidth，呈线性上升趋势（图3）。 |
| **Scalability w.r.t $k$** | 当 $k$ 增大时，Prof-K 仍保持竞争力，虽优势缩小但仍优于 baseline（图4）。 |
| **Memory Footprint** | Prof-K 内存占用几乎恒定（仅与 buffer 大小相关），而 RadiK 占用随 $k$ 增加而显著上升（图5）。 |

---

### 消融实验结果（隐含分析）

虽然未明确列出消融表，但文中提供了以下关键设计选择的理论与实证支撑：

- **Sample Size $S^* \sim (kN)^{1/3}$**：推导出近似最优采样量，并验证其有效性（Theorem 5）。
- **Buffer Size $M = k + O(\sqrt{k(N-k)/S})$**：闭式表达确保高概率容纳所有 top-k 元素（Corollary 4）。
- **Fallback 机制的有效性**：即使在极端稀疏场景（$\alpha S < 1$）偶尔触发 fallback，其 amortized overhead 极低（$O(\epsilon)$）。

---

## 4. 关键结论和发现

### 论文的主要发现

1. ✅ **Prof-K 显著加速 large-$N$, small-$k$ 场景下的 top-k 选择**，尤其适合当前主流的稀疏 ML 工作负载（如 SAEs、sparse attention）。
2. ✅ **理论保证强且实用**：提供可调的概率正确性保障（recall ≥ $1-\epsilon$），且不依赖输入分布特性。
3. ✅ **实际部署有效**：在真实的 BatchTopK SAE 训练中，Prof-K 不仅降低了 kernel 开销，还带来了可观的端到端训练时间节省（~4.25%），同时 **完全保持了模型重建质量、sparsity 和 downstream behavior**（图7）。
4. ✅ **内存友好**：辅助内存仅为 $O(k)$ 级别，避免大规模张量处理中的 OOM 问题。

---

### 方法的局限性（见 Appendix A）

| 局限性 | 说明 |
|--------|------|
| **小 $N$ 场景收益有限** | 当输入较小时，sampling 和 filtering 的开销无法被 offset，性能增益不显著。 |
| **存在 fallback 开销** | 极少数情况下需回退到 full top-k，引入轻微不确定性（但 amortized 成本极低）。 |
| **Batched 场景风险累积** | 批处理时，至少一次 failure 的概率随 batch size 上升而增加，需谨慎调参。 |
| **尚未扩展至分布式环境** | 当前工作聚焦单节点 GPU，multi-GPU 或 distributed top-k 尚未探索。 |

---

### 未来工作方向

1. **扩展至分布式 Top-k**：研究如何在 multi-GPU 或集群环境中应用 Prof-K 进行高效的全局 top-k 选择。
2. **动态自适应参数调整**：根据运行时统计信息在线调整 $S, t, M$ 以最大化稳定性与性能。
3. **集成进更多稀疏架构**：应用于 MoE routing、pruned transformers、retrieval systems 等其他频繁调用 top-k 的场景。
4. **支持 top-k gradients 的高效反向传播**：目前 focus 在 forward pass，未来可探索 backward pass 的协同优化。

--- 

> **总结一句话**：  
> Prof-K 通过 **概率性预过滤 + 精确 refine** 的设计，在保证高 recall 的前提下，大幅降低了 Top-k 选择的计算与内存开销，是面向现代大规模稀疏机器学习系统的高效、鲁棒、可扩展的系统级优化方案。

</details>

---

### 11. [DARTree: Speculative Diffusion Decoding with Autoregressive Draft Trees](https://arxiv.org/abs/2608.13524)

**Authors**: Tianyi Li, Yaxin Luo, Xinyi Shang, Zhiqiang Shen  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.13524v1  

#### Abstract
Speculative decoding losslessly accelerates autoregressive language models by verifying multiple draft tokens in parallel. Diffusion-based drafters further reduce proposal latency by predicting an entire token block in parallel, but their position-wise distributions are marginal rather than conditio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**DARTree: Speculative Diffusion Decoding with Autoregressive Draft Trees**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代 **Autoregressive (AR) 大语言模型 (LLMs)** 虽然在文本生成、推理等方面表现出色，但由于其固有的**串行解码机制**，推理延迟较高，限制了实际应用中的响应速度。

现有的 **Speculative Decoding** 方法通过轻量级“drafter”并行预测多个未来 token 并由目标模型验证，以实现无损加速。然而：
- **传统 autoregressive drafters** 仍需逐个生成 draft token，提案延迟随长度增长；
- **Diffusion-based block drafters**（如 DFlash）虽可并行预测整个 token 块，但各位置分布是基于已验证前缀独立建模的（即 marginal 分布），缺乏对块内先前 draft token 的依赖，导致与目标模型的条件分布之间存在 **causal mismatch（因果不匹配）**；
- 现有引入因果修正的方法（如 Domino 使用 RNN correction）仅作用于单条 draft 链，无法扩展到树形结构中；
- **DDTree** 等基于 diffusion drafter 构造 draft tree 的方法虽然提升了候选覆盖范围，但在构造过程中未将因果修正沿各个分支传播，且因与堆操作耦合而引入新的效率瓶颈。

### 🚀 提出的新方法：DARTree
DARTree 是一种**无需训练**的 speculative decoding 方法，旨在将预训练好的 **causally corrected AR correction head** 从单一链式结构推广到**树状结构**中，从而兼顾高接受率与高效并行化。

#### 核心创新点：
1. **深度维度批处理的树构建（Depth-wise Batched Tree Construction）**
   - 在每一层深度上，并行扩展固定数量的节点（例如每层保留 4 或 12 个候选），并在一个 batch 中统一进行 AR head 的因果修正评分。
   - 避免了传统 best-first search 中“弹出堆 → 推理 correction head → 推入子节点”的串行交错过程。

2. **延迟的最佳优先剪枝（Deferred Best-First Pruning）**
   - 先构建一个更宽的“超树”（supertree），然后一次性执行全局 top-B 选择来获得最终用于验证的小型紧凑树。
   - 利用 **Lemma 1** 证明：当 depth bonus β ≤ 0 且祖先优先时，best-first heap selection 与 global top-B selection 等价，因此可以安全替换为批量操作。

3. **解耦设计提升效率**
   - 将 **correction-head inference** 与 **sequential heap operations** 完全解耦，显著降低树构建延迟（见 Figure 1）。

### 🔍 相比现有方法的优势
| 方法 | 是否支持因果修正 | 是否支持树结构 | 是否批量化修正 | 是否无需训练 |
|------|------------------|----------------|----------------|-------------|
| DFlash | ❌ | ❌ | ❌ | ✅ |
| DDTree | ❌ | ✅ | ❌ | ✅ |
| Domino | ✅（仅链式） | ❌ | ❌ | ❌（需联合训练） |
| **DARTree** | ✅（多分支路径依赖） | ✅ | ✅（depth-wise 批处理） | ✅ |

> ✅ **优势总结**：DARTree 在保持无需训练的前提下，首次实现了在**树结构中并行应用因果修正**，既提高了 acceptance length，又避免了串行瓶颈。

---

## 2. 核心实验方法和设置

### 📚 数据集
涵盖三大类共七个基准任务：
- **Math 数学推理**：
  - GSM8K（小学数学题）
  - MATH-500（高等数学问题）
  - AIME25（美国数学邀请赛）
- **Code 编程能力**：
  - HumanEval（代码生成）
  - MBPP（Python 编程任务）
- **Chat 对话质量**：
  - MT-Bench（多轮对话评测）
  - Alpaca（指令遵循）

### ⚙️ 实验设置
- **主干模型**：Qwen3-4B 和 Qwen3-8B
- **温度设置**：T=0（greedy decoding）和 T=1（stochastic sampling）
- **评估指标**：
  - **Average Acceptance Length (T)**：每个 draft-verify 轮次平均接受的 token 数量（含 bonus token）
  - **Speedup**：相对于标准 autoregressive 解码的速度提升倍数
    $$
    \text{Speedup} = \frac{L_{\text{AR}}}{(T_{\text{draft}} + T_{\text{verify}})/T}
    $$
- **硬件平台**：单张 NVIDIA RTX 6000 Ada Generation GPU，batch size = 1
- **最大生成长度**：2048 tokens / prompt

### 🆚 基线方法对比
| 方法 | 类型说明 |
|------|--------|
| **Vanilla AR** | 标准自回归解码（baseline） |
| **DFlash** | 并行预测 16-token 块的 diffusion drafter |
| **DDTree** | 基于 DFlash 的 marginal 分布构建 best-first draft tree |
| **Domino** | 联合训练 diffusion backbone + autoregressive correction head（链式） |

> DARTree 默认复用 Domino 发布的 correction head，不做额外训练。

### 🌲 DARTree 变体
- **DARTree (fixed)**：每层均匀分配 budget（如 B=64，d=16，则每层保留 4 个节点）
- **DARTree (pruned)**：先构建更宽的 supertree（如宽度 12），再通过 deferred pruning 得到 B=64 的验证树

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

#### 在 Qwen3-4B 上的整体表现（T=0）：
| Method | Avg. T | Max Speedup |
|--------|--------|------------|
| DFlash | 5.97 | 4.58× |
| DDTree | 7.78 | 5.79× |
| Domino | 7.11 | 5.17× |
| **DARTree (fixed)** | **9.36** | **6.55×** |
| **DARTree (pruned)** | **9.81** | **6.99×** |

> ✅ **最高接受长度达 12.97 tokens/round（GSM8K 上）**
>
> 💡 比 DFlash 多接受 **98.6%** 的 token，比 Domino 多 **27.9%**

#### 最高速度提升：
> **高达 9.73× 的 lossless speedup**（相比本地 autoregressive 解码）

---

### 🔁 与基线方法的对比结果

| 维度 | 结果 |
|------|------|
| **Acceptance Length** | 在所有 7 个 benchmark 和 4 种 model-temperature 配置下，DARTree 均取得最高的平均 acceptance length |
| **Speedup 表现** | 同样全面领先，在多个任务上突破 9× 加速 |
| **跨模型泛化性** | 应用于 DSpark-Markov correction head 时仍能提升 acceptance（↑14.6–40.6%）和 speedup（↑ up to 34.3%）|

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）不同树构建策略对比（Table 2）
| 方法 | T | Speedup |
|------|----|---------|
| Sequential correction + heap | 12.84 | 4.38× |
| Domino-chain + DDTree | 11.27 | 8.48× |
| **DARTree** | **12.97** | **9.77×** |

> ✅ DARTree 不仅接受更多 token，而且由于去除了串行堆交互，**round time 更短**，综合加速更强。

#### （2）超参数影响分析（Figure 4）
- **Layer Width (W)**：
  - 从 4 增加到 12 显著提升 acceptance；
  - 超过 12 收益递减 → 推荐 W=12
- **Verification Budget (B)**：
  - acceptance 随 B 单调上升；
  - speedup 在 B=64~128 达到峰值，B=192 下降 → 推荐 B=64 作为平衡点

#### （3）其他关键参数（Figure 6）
- **Depth Bonus β**：最优值在 **-0.2 ~ -0.1** 区间（负值确保祖先优先）
- **Candidate Cap K**：K=64 已足够，使用 full vocab 会增加 AR head 延迟，损害 speedup

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DARTree 成功解决了“因果修正”与“树搜索效率”的矛盾**：
   - 通过 depth-wise 批量修正 + deferred pruning，实现了路径依赖打分的同时最大化并行度。
2. **在不增加训练成本的情况下显著超越已有方法**：
   - 无需重新训练任何模块，即可复用现有 causal correction heads（如 Domino、DSpark）。
3. **acceptance length 与 speedup 双重领先**：
   - 最高接受 **12.97 tokens/round**，最高速度提升达 **9.73×**。
4. **生成质量更高**：
   - 可视化显示 DARTree 构造的树具有更强语义一致性（如避免 “plants plants sunlight” 这类重复错误）。

---

### ⚠️ 局限性（Limitations）

1. **依赖预训练的 causal correction head**
   - 不能直接用于没有 correction head 的 naive diffusion drafters；
   - 必须有兼容模型发布才能部署。

2. **计算开销更大**
   - 虽然减少了延迟，但总 FLOPs 增加（因需验证大树）；
   - 不适用于高并发场景，可能成为计算瓶颈。

3. **更适合低并发、内存带宽受限环境**
   - 如个人设备、边缘部署、低批量服务等。

---

### 🔮 未来工作方向
- 设计轻量化的通用 causal correction head，便于迁移至各类 drafter；
- 动态调整 tree width 和 budget 以适应不同请求负载；
- 探索在 compute-bound 场景下的优化策略（如稀疏化、蒸馏）；
- 扩展至 multimodal generation 中的 speculative decoding。

---

## ✅ 总结一句话
> **DARTree 是首个实现“无需训练 + 树结构 + 并行因果修正”的 speculative decoding 方法，在多项 benchmark 上实现了当前最高的 acceptance length 与 speedup，为高效 LLM inference 提供了一种极具潜力的新范式。**

</details>

---

### 12. [Trie Automata for Constrained Decoding over Large Finite Sets](https://arxiv.org/abs/2608.12574)

**Authors**: Xingzi Xu, Karim Bouyarmane  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.12574v1  

#### Abstract
Large language models increasingly need to generate structured outputs that conform to predefined schemas, with one common constraint being selection from a finite set of valid strings. Current constrained decoding systems handle this through general-purpose grammar compilation, which becomes prohib...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Trie Automata for Constrained Decoding over Large Finite Sets

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **constrained decoding** 系统（如 XGrammar、Outlines）在处理大规模有限字符串集合（finite-set constraints）时面临严重的性能瓶颈，即所谓的 **cardinality wall**（基数墙）。当枚举值数量 $ K $ 超过数百至数千时，编译时间和每步解码延迟急剧上升，导致系统无法用于生产环境。

这一问题在以下场景中尤为突出：
- 工具调用（tool calling）中的 API 注册表选择（$ K \sim 500–5000+ $）
- 零样本分类任务（如产品分类、医疗编码 ICD-10-CM，$ K > 70,000 $）
- 动态检索增强生成中的每查询约束（per-query constraints）

通用的 FSM（有限状态机）或 Earley parser 编译管道对所有 schema 一视同仁，未能利用“有限字符串集合”特有的结构特征（共享前缀、有界深度、已知基数），造成不必要的计算开销。

---

### 提出的新方法：Trie Automaton
作者提出 **Trie Automaton** ——一种专为有限字符串集合设计的轻量级、高性能 constrained decoding 后端机制，其核心思想是：

1. **构建字符级 Trie（character-level trie）**  
   将枚举集合 $ E = \{e_1, ..., e_K\} $ 构建成一个字符粒度的前缀树，最大化共享路径以减少节点数。

2. **通过 Aho-Corasick 多模式匹配预计算 token masks**  
   利用 Aho-Corasick 算法将整个 vocabulary 视为一组 pattern，在一次遍历中高效地找出每个 Trie 节点上合法延续的 BPE tokens。

3. **解码时进行 O(1) 查找式的 mask 应用**  
   所有有效 token 集合 `valid[n]` 在编译阶段预先计算并缓存，解码时只需查表即可获得当前状态下的允许 token 集合，无需运行时 FSM 推理。

该方法实现了从 **动态 FSM 推理** 到 **静态 mask 查找** 的范式转变。

---

### 相比现有方法的优势

| 维度 | Trie Automaton | FSM-based (XGrammar) | LLGuidance |
|------|----------------|------------------------|-----------|
| **编译时间** | 近似恒定（30–67ms @K=10K） | 随 $ K $ 快速增长（~2.7s @K=100K） | 极低（<24ms） |
| **每步 mask 成本** | **0.65μs**（查表） | ~5.8μs（动态 FSM 扫描） | 73–141μs（全词表扫描） |
| **批处理吞吐量** | **219 req/s @B=256** | 7.5 req/s @B=256 | 受限于 CPU 掩码瓶颈 |
| **内存占用** | ~0.9MB @K=10K | ~2GB（理论） | 中等 |
| **输出正确性** | 100% schema compliance | 100% | 100% |
| **算法复杂度** | $ O((N_{chars} + V)\cdot l) $ 编译<br>$ O(|valid[s_t]|) $ 解码 | $ O(K \cdot L_{max} \cdot |\Sigma|) $ 编译<br>$ O(V \cdot l) $ 解码 | $ O(1) $ 编译<br>$ O(V) $ 解码 |

> ✅ **核心优势总结**：
> - **7× 更快的每步 mask 计算**
> - **2–6.5× 更快的编译速度（当 $ K \geq 300 $）**
> - **批处理下端到端吞吐提升达 29×**
> - 支持高达 $ K=100,000 $ 的枚举规模，远超主流平台限制（OpenAI: 1K, Gemini: ~120）

---

## 2. 核心实验方法和设置

### 数据集

#### （1）合成工具名基准（Synthetic Tools Benchmark）
- 构造格式：`<namespace>.<action>_<resource>`，例如 `slack.get_user`, `aws.create_invoice`
- 包含 10 个命名空间、10 个动作、10 个资源
- $ K \in \{10, 100, 1K, 5K, 10K, 50K, 100K\} $
- 用于评估 **延迟、编译时间、吞吐量**

#### （2）公开分类数据集（Accuracy/Evaluation）
- **TREC** ($ K=42 $): 问题分类
- **MASSIVE** ($ K=59 $): 意图识别
- **Banking77** ($ K=77 $): 银行客服意图
- **CLINC150** ($ K=150 $): 多领域意图分类
- 用于评估 **准确率与有效性（accuracy & validity）**

#### （3）高基数真实世界枚举
- **Product Names (K=1,500)**: 合成商品名称
- **ICD-10-CM (K=74,719)**: 美国医保医疗编码标准（真实生产级数据）

---

### 实验设置与评估指标

| 类别 | 内容 |
|------|------|
| **模型** | Qwen3-8B, Mistral-7B, GPT2, OLMo, Gemma3-12B 等共 7 种 tokenizer（32K–262K vocab） |
| **硬件** | NVIDIA A100 GPU (80GB), AMD EPYC 7R32 CPU (96 cores) |
| **集成框架** | vLLM（作为 `LogitsProcessor` 集成） |
| **评估指标** | |
| - 编译时间（Compile time） | 构建 automaton 时间 |
| - 每步 mask 时间（Mask time / step） | 仅指确定有效 token 的时间（不包括 logits 修改） |
| - 端到端吞吐量（Throughput, req/s） | 批大小 $ B=1 $ 至 $ 256 $ 下的请求处理速率 |
| - 准确率（Accuracy） | 分类任务中预测正确的比例 |
| - 有效性（Validity） | 输出是否严格符合 schema（100% for trie） |

---

### 基线方法对比

| 方法 | 描述 |
|------|------|
| **XGrammar** | 当前 vLLM 和 SGLang 主要后端，基于 FSM 的 regex 编译 |
| **LLGuidance** | 基于 Earley parser 的惰性解析器，避免提前编译，适合多样 schema |
| **GENRE-style Token Trie** | 基于 token ID 构建的 trie，绕过 BPE 对齐问题但牺牲字符级共享 |
| **Unconstrained + Retry / Post-hoc** | 无约束生成 + 重试或模糊匹配，常失败且不保证有效性 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）编译时间 vs. 枚举基数 $ K $

| $ K $ | Trie Automaton | XGrammar | 加速比 |
|-------|---------------|----------|--------|
| 100   | 31ms          | 15ms     | ~2×    |
| 1,000 | 33ms          | 75ms     | **2.3×** |
| 10,000| 40ms          | 239ms    | **6×** |
| 100,000| 67ms         | 2.7s     | **40×** |

> 🔹 编译时间几乎平坦，因主导成本为与 $ K $ 无关的 Aho-Corasick 自动机构建。

#### （2）每步 mask 时间（per-step masking cost）

| 方法 | 平均耗时（Qwen3-8B） |
|------|---------------------|
| Trie Automaton | **0.65 μs** |
| XGrammar | 5.8 μs |
| LLGuidance | 73–141 μs |

> 🔹 Trie 实现 **~9×** 于 XGrammar，**~110–215×** 于 LLGuidance 的加速。

#### （3）端到端 vLLM 吞吐量（$ K=1,000 $）

| Batch Size | Trie (req/s) | XGrammar (req/s) | 加速比 |
|------------|-------------|------------------|--------|
| 1          | 4.4         | 2.9              | 1.5×   |
| 32         | 70.4        | 7.4              | 9.6×   |
| 128        | 170.5       | 7.6              | 22.4×  |
| **256**    | **219.4**   | **7.5**          | **29.3×** |

> ✅ **29× 吞吐提升** 来源于两方面：
> 1. **算法层面**：每步 mask 快 7×
> 2. **集成路径优化**：Trie 可作为 stateless `LogitsProcessor`，绕过 vLLM 的 guided decoding pipeline 开销

#### （4）跨 tokenizer 家族泛化能力（$ K=10,000 $）

| Model | Vocab Size | 编译加速比（vs. XGrammar） |
|-------|------------|----------------------------|
| Mistral 7B v0.3 | 32K | **11.5×** |
| GPT-2 | 50K | 12.0× |
| OLMo 3 7B | 100K | 5.2× |
| Gemma3 12B | 262K | 3.5× |

> 🔹 小 vocab 下优势更明显（AC 构建更快）

---

### 消融实验结果

| 配置 | Latency (s) @K=5K | Schema Compliance |
|------|-------------------|--------------------|
| Trie Automaton（完整） | 0.77 | 100% |
| Hierarchical Only | 3.99 | 100% |
| Speculative Only | 2.00 | 100% |
| XGrammar Regex（Baseline） | 0.89 | 100% |

> 🔹 在 $ K=5K $ 时，Trie 单独最快；推测性解码（speculative）未带来收益，说明 mask 本身已足够廉价。

---

## 4. 关键结论和发现

### 主要发现

1. **存在显著的“基数墙”现象**  
   当前主流 constrained decoding 引擎在 $ K > 1,000 $ 时性能急剧下降，而 Trie Automaton 成功突破此墙，支持 $ K \sim 10^5 $。

2. **不同约束应使用不同执行机制（Constraint-Specialized Backends）**  
   有限字符串集合具有可被利用的结构特性（共享前缀、固定长度），不应使用通用 FSM 处理。

3. **预计算（precomputation）能解锁新的服务路径**  
   Trie 的 mask 是静态的，使其可以走 **stateless serving path**，从而绕过复杂的 guided decoding 流程，这是实现 29× 吞吐的关键。

4. **输出等价性保障**  
   理论证明 Trie Automaton 与最小 DFA 等价（Proposition 1），保证输出完全一致，非近似方法。

5. **实际部署友好**  
   - 内存占用小（~8MB @K=100K）
   - 支持多线程并发读取（immutable 结构）
   - 可缓存 AC 自动机供多个 schema 复用

---

### 局限性（Limitations）

| 限制 | 说明 |
|------|------|
| **仅适用于 flat finite-set constraints** | 不支持嵌套对象、数组、递归结构等复杂 schema |
| **混合 schema 需组合使用** | 需与 FSM/PDA 后端协同处理结构性字段 |
| **单次动态 schema 场景可能不如 LLGuidance** | 若 $ K < 500 $ 且每次 schema 不同，LLGuidance 的零编译更具优势 |
| **极端 $ K $ 下准确性受限于模型而非 decoder** | 如 ICD-10-CM ($ K=74,719 $) 上模型准确率为 0%，但 trie 仍使尝试成为可能 |

---

### 未来工作方向

1. **Hierarchical Schema Rewriting**  
   对超大 $ K $（>50K）进行聚类分组（如按语义、命名空间），实现两级解码（group → item），将有效基数降至 $ O(\sqrt{K}) $。

2. **Speculative Short-Circuiting**  
   使用轻量级模型先生成 top-k 候选，再用 trie 精确约束，牺牲部分完整性换取更低延迟。

3. **扩展至其他规则约束**  
   将“约束感知调度”推广到日期时间格式（YYYY-MM-DD）、数值范围、正则子类等，如 Appendix K 所示，char-position mask 在 date/time 上可达 **7,939× 编译加速**。

4. **GPU-based Masking Integration**  
   探索将预计算 bitmask 移至 GPU 执行，进一步降低 CPU-GPU 同步开销。

---

> 📌 **最终结论**：  
> **Trie Automaton 是解决大规模有限集合 constrained decoding 的最优算法路径**。它通过 **exploiting structure + precomputation + integration simplification** 实现了数量级的性能飞跃，并推动了“constraint-specialized backend”的设计理念。随着 agentic systems 和 structured generation 的普及，专用解码引擎将成为标配。

</details>

---

### 13. [From Local Mismatch to Global Impact: Optimizing Cache Reuse Policy for Efficient Diffusion](https://arxiv.org/abs/2608.13043)

**Authors**: Xichen Ye, Yifan Wu, Zhikang Xie, Xiangyu Yue, Cheng Jin, Weizhong Zhang  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.13043v1  

#### Abstract
Diffusion models have achieved dominant performance in visual generation but suffer from substantial inference overhead. While cache-based acceleration has emerged as a promising solution, existing policies rely on local similarity heuristics, which we identify as being significantly misaligned with...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Local Mismatch to Global Impact: Optimizing Cache Reuse Policy for Efficient Diffusion

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现有的 **cache-based acceleration** 方法（如 TeaCache、ERTACache）依赖于局部相似性度量（如相对 $l_1$ 距离）来决定是否复用缓存的残差。然而，这些**局部误差指标与最终生成质量之间存在显著错位**（misalignment），即：

- 局部差异大的时刻，可能对全局视觉保真度影响很小；
- 反之，某些看似微小的早期误差，可能因在去噪轨迹中累积传播而造成严重退化。

这种错位导致现有策略做出次优甚至有害的缓存决策，牺牲了生成质量以换取速度。

---

### 🚀 提出了什么新方法或新思路

作者提出 **Global-Impact Cache (GCache)**，一种从“全局影响”视角优化缓存复用策略的新框架，其核心思想包括：

#### （1）理论建模误差传播上界
- 建立了 cache reuse 引发的误差在 ODE 推理过程中的**传播动力学理论分析**。
- 推导出一个**全局误差上界**，该上界显式地包含了误差随时间指数放大的机制（$\exp(L \cdot t)$），揭示了**早期步骤的误差具有更强的破坏力**。

#### （2）识别保守性偏差并引入可学习参数化
- 发现直接最小化上述理论上界会导致过于悲观（overly conservative）的策略（例如过早停止复用）。
- 因此将传播指数 $w_t$ 参数化为 **Bernstein 形式的多项式函数**，使其能够灵活拟合实际非凸模型中的误差传播行为。

#### （3）构建双层优化框架（Bilevel Optimization）
- **内层目标**：给定当前传播权重参数 $s$，通过动态规划（Dynamic Programming）求解最优缓存策略 $m^*(s)$，最小化加权后的传播误差。
- **外层目标**：利用贝叶斯优化（Bayesian Optimization）调整参数 $s$，使得对应策略下的实际生成损失（如 LPIPS）最小化。

> 这种设计实现了**理论严谨性与实证性能之间的有效平衡**，让模型学会优先保护那些真正影响视觉质量的关键计算步骤。

---

### 🔍 相比现有方法的优势

| 维度 | GCache | 现有方法（如 ERTACache） |
|------|--------|--------------------------|
| 决策依据 | 全局影响 + 学习型误差传播模型 | 局部相似性启发式（如 Rel-L1） |
| 是否引入额外模块 | 否（仅优化 policy） | 是（常需 error rectification 模块） |
| 性能表现 | 更高质量 + 更高速度 | 在相同加速比下质量更低 |
| 泛化能力 | 支持跨分辨率、跨提示分布零样本迁移 | 多为固定规则或静态策略 |

> ✅ **GCache 不需要修改模型结构或增加推理开销，是一种轻量级、即插即用的高效加速方案。**

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **视频生成任务**：
  - 使用 **VBench** 提供的官方 946 个 text-to-video prompts。
  - 覆盖多样化的场景类别与运动模式，用于全面评估生成质量。

- **图像生成任务**：
  - 使用 **COCO validation set** 中前 30K 文本描述作为 prompt 输入。
  - 所有方法在同一 prompt 集上测试，确保公平比较。

---

### ⚙️ 实验设置和评估指标

#### 模型架构
在多个主流 DiT-based 扩散模型上验证通用性：
- 视频模型：
  - Open-Sora 1.2
  - CogVideoX-2B
  - Wan 2.1-1.3B
- 图像模型：
  - Flux-dev 1.0

#### 缓存预算控制
- 定义缓存刷新次数 $K = \|m\|_0$，表示允许进行完整计算的步数。
- 对比不同 $K$ 下的速度-质量权衡。

#### 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **效率** | Speedup ↑, Latency ↓ | 端到端推理耗时加速比 |
| **视觉质量** | VBench ↑ | 视频多维度综合评分 |
| | LPIPS ↓ | 感知相似性（越低越好） |
| | SSIM ↑, PSNR ↑ | 结构与像素级保真度 |

---

### 🆚 基线方法对比

对比了一系列最新的 cache-based 加速方法：
- △-DiT [14]
- PAB [18]
- TeaCache [15]
- ERTACache [16]（当前 SOTA）
- T-GATE [28], ProfilingDiT [29], FasterCache [30]

特别强调与 **ERTACache** 的对比，因其是目前最先进的带误差校正机制的方法。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

#### 在 **Wan 2.1-1.3B** 上的表现（视频生成）
| 方法 | Speedup | LPIPS | PSNR |
|------|---------|-------|------|
| ERTACache | 2.17× | 0.1095 | 23.77 |
| **GCache-slow** | **2.17×** | **0.0316** | **32.44** |
| GCache-fast | **3.01×** | 0.0828 | 22.06 |

> 💥 **在相同 2.17× 加速下，LPIPS 降低 71%（0.1095 → 0.0316），PSNR 提升近 9 分！**

#### 在 **Flux-dev 1.0** 上的表现（图像生成）
| 方法 | Speedup | LPIPS | PSNR |
|------|--------|-------|------|
| ERTACache | 2.87× | 0.2658 | 20.60 |
| **GCache-fast** | **2.87×** | **0.1825** | **23.76** |

> ✅ 显著优于所有 baseline，在保持高加速比的同时大幅提升感知质量。

---

### 🔬 消融实验结果（Ablation Studies）

#### （1）Bernstein 多项式阶数 $d$ 的影响（Table 3）
| $d$ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|-----|--------|--------|--------|
| 1 | 0.1114 | 0.8591 | 26.15 |
| 2 | 0.0733 | 0.9042 | 29.09 |
| **3** | **0.0721** | **0.9042** | **29.14** |
| 4 | 0.0733 | 0.9016 | 28.84 |

> ✅ 最佳性能出现在 $d=3$，更高阶并未带来增益，表明三次 Bernstein 已足够建模复杂传播行为。

#### （2）外层目标函数的影响（Table 4）
| 外层目标 | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|--------|--------|--------|--------|
| LPIPS | 0.0733 | 0.9016 | 28.84 |
| SSIM | 0.0736 | 0.9045 | 29.10 |
| **LPIPS+SSIM** | **0.0721** | **0.9042** | **29.14** |

> ✅ **混合目标（LPIPS + SSIM）效果最好**，兼顾感知与结构一致性。

#### （3）是否依赖错误校正模块？（Table 8）
| 方法 | LPIPS ↓ | PSNR ↑ |
|------|--------|--------|
| ERTACache*（无 rectification） | 0.1659 | 22.34 |
| ERTACache（含 rectification） | 0.1012 | 26.44 |
| **GCache（无 rectification）** | **0.0721** | **29.14** |

> ✅ **GCache 即使不使用任何 error rectification 模块，仍显著优于 ERTACache 的完整版本**，证明其策略本身更优。

---

### 🌐 泛化性实验

#### （1）跨分辨率零样本迁移（Table 7）
- 将在 1024×1024 上训练的 GCache-fast 策略直接应用于 512 和 256 分辨率。
- 在所有尺度上均持续优于 ERTACache。

> ✅ 表明 GCache 学到的是**分辨率无关的冗余模式**，具备强泛化能力。

#### （2）跨提示分布鲁棒性（Table 6）
- 在静态/动态提示子集上分别训练 GCache，交叉测试。
- 性能差距极小（LPIPS 差异 < 0.002），说明策略稳定且泛化良好。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **局部误差 ≠ 全局影响**  
   现有基于局部相似性的缓存策略严重误判关键步骤，必须考虑误差沿去噪路径的**非均匀传播特性**。

2. **早期误差更具破坏性**  
   理论与实验共同验证：**早期步骤的缓存误差会被系统动力学指数放大**，应优先保留这些步骤的精确计算。

3. **理论指导 + 数据驱动 = 最佳策略**  
   GCache 成功结合了理论误差边界与数据驱动优化，通过 bilevel 框架学习到更贴近真实感知质量的误差加权方式。

4. **无需辅助模块即可超越 SOTA**  
   GCache 仅靠优化 policy 本身，就在多个任务上大幅超越依赖 error rectification 的复杂方法。

---

### ⚠️ 方法的局限性

- **固定预算假设**：当前方法基于预设的缓存刷新次数 $K$，尚未支持输入自适应的动态调度。
- **依赖预计算误差代理**：虽然验证显示其可靠性高，但在极端轨迹偏移情况下可能存在偏差。
- **未探索更多感知目标**：目前外层目标集中在 LPIPS/SSIM，未来可尝试 CLIP-based 或人类偏好对齐的目标。

---

### 🔮 未来工作方向

1. **Sample-adaptive scheduling**：根据输入内容复杂度自动调整 $K$ 或分配策略。
2. **更丰富的监督信号**：引入 CLIP Score、Aesthetic Score 或 human feedback 作为外层目标。
3. **扩展至其他生成范式**：如应用于 AR 模型、flow-based models 或 language models 中的状态复用。
4. **硬件协同优化**：结合内存访问模式进一步降低延迟。

---

## 总结

> **GCache 是首个系统性解决“局部缓存误差”与“全局生成质量”错位问题的工作**。它通过建立误差传播理论、设计可学习的双层优化框架，在不增加推理负担的前提下，实现了**速度与质量的双重突破**。大量实验证明其在图像与视频扩散模型上的普适性和优越性，代表了 cache-based acceleration 方向的重要进展。

</details>

---

### 14. [Teach the Magnitude, Not the Direction: Verifier-Bounded Credit Assignment for Multi-Turn Multi-step LLM Agents](https://arxiv.org/abs/2608.13179)

**Authors**: Zechuan Wang, Siyuan Lu, Hongxuan Zhang, Linjian Mo, Chenyi Zhuang, Leilei Gan  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.13179v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) offers a verifier-bounded performance ceiling for training multi-turn tool-use agents, yet its trajectory-level credit assignment conflates heterogeneous per-turn outcomes into a single reward signal. On-policy distillation provides dense per-tok...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Teach the Magnitude, Not the Direction: Verifier-Bounded Credit Assignment for Multi-Turn Multi-step LLM Agents*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

本文针对**多轮多步 LLM Agent** 在强化学习训练中的**信用分配（credit assignment）难题**，指出当前主流方法存在两大缺陷：

- **RL-based 方法（如 GRPO）**：使用轨迹级奖励（trajectory-level reward），在多轮场景下将成功与失败的轮次混为一谈，导致“**跨轮稀释（inter-turn dilution）**”——失败轮次的 token 被错误地赋予正优势（advantage），影响策略优化。
- **On-policy Distillation 方法（如 OPSD）**：虽提供密集的 token 级监督信号，但存在“**梯度集中崩溃（gradient concentration collapse）**”，即梯度集中在少数低熵格式 token（如 `tool_name:`）上，而忽略高不确定性、高价值的内容决策 token（如 `get_flight`）。此外，其性能上限受教师模型限制（**teacher-bounded ceiling**），无法超越教师。

### 提出了什么新方法或新思路

提出 **CREST**（**Hierarchical Credit Assignment via Entropy-Gated Self-Teacher**），一种分层信用分配框架，核心思想是：

> **让验证器（verifier）决定梯度方向，让自教师（self-teacher）仅调节更新幅度（magnitude）**。

该方法通过两个层级解决信用分配问题：

1. **Inter-turn Credit Assignment（跨轮信用分配）**  
   使用**按轮分割的验证奖励（turn-segmented verified rewards）**，独立计算每一轮的优势值 $A_{\text{turn}}$，避免整条轨迹的奖励平均化，确保失败轮获得负优势。

2. **Intra-turn Credit Assignment（轮内信用分配）**  
   引入**熵门控的自教师调制（entropy-gated self-teacher modulation）**：
   - 自教师基于 ground-truth 上下文生成参考分布；
   - 计算学生与自教师之间的 token 级差异 $\Delta_t$；
   - 使用**方向门（direction gate）** 确保教师信号不改变 verifier 决定的方向；
   - 使用**熵门（entropy gate）** 将调制强度集中在高不确定性（surprisal 高）的内容 token 上，抑制对低熵格式 token 的过度响应。

最终的 per-token advantage 结构为：
$$
A_t = A_{\text{turn}} \cdot \phi_t, \quad \phi_t = 1 + \lambda \cdot g_d \cdot g_e \cdot (w_t - 1)
$$
其中 $\phi_t$ 是由自教师驱动的**幅度调制因子**，且始终满足 $\text{sign}(A_t) = \text{sign}(A_{\text{turn}})$，保证性能上限由 verifier 决定（**verifier-bounded ceiling**）。

### 相比现有方法的优势

| 特性 | GRPO / MT-GRPO | OPD / OPSD | CREST |
|------|----------------|------------|--------|
| Credit Level | Trajectory/Step | Token | **Turn + Token (Hierarchical)** |
| Gradient Direction Source | Verifier | Teacher | ✅ **Verifier only** |
| Dense Signal | ❌ Sparse | ✅ Dense | ✅ Dense |
| Performance Ceiling | ✅ Verifier-bounded | ❌ Teacher-bounded | ✅ **Verifier-bounded** |
| Gradient Concentration | Diffuse (uniform) | ❌ Severe | ✅ Controlled via entropy gate |

> ✅ CREST 成功结合了 RL 的“可验证上限”与蒸馏的“密集监督”，同时避免了两者的致命缺陷。

---

## 2. 核心实验方法和设置

### 使用的数据集

1. **BFCL V3**（Berkeley Function-Calling Leaderboard）
   - 多轮工具调用基准，包含 Base、Missing Functions、Missing Parameters、Long-Context 四个子集。
   - 使用 100 个固定 ID 进行训练，400 个非重叠样本用于测试。

2. **WildToolBench**
   - 更贴近真实交互的多轮会话环境，包含指代消解、任务切换、隐含意图等挑战。
   - 包含 256 个多轮 session，划分为 128 训练 + 128 测试。
   - 评估指标包括 Action Accuracy 和 **Session Accuracy**（严格端到端正确率）。

### 实验设置和评估指标

- **Base Models**：
  - `Qwen3-4B-Instruct`（指令微调型）
  - `Qwen3-8B`（推理增强型）
- **Group Size**：$G=16$
- **Optimizer**：Adam（β1=0.9, β2=0.999），学习率 $1\times10^{-6}$
- **评估方式**：每个策略运行 3 次 decode（temperature=$10^{-6}$），报告平均准确率。

### 基线方法对比

| 类别 | 方法 | 描述 |
|------|------|------|
| **RL-based** | GRPO | 标准 Group Relative Policy Optimization，轨迹级二元奖励 |
| | MT-GRPO | 每 agent step 计算优势，但仍共享同一优势值 |
| | EnvTuning | 使用细粒度过程奖励加权聚合为轨迹级优势 |
| **Distillation-based** | OPD | 使用同族大模型作为外部教师进行 on-policy distillation |
| | OPSD | 使用自身 + ground-truth 上下文作为自教师，易崩溃 |

> CREST 仅引入一个可调超参数 $\lambda$（默认 0.3），其余为固定设计。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

#### 在 **BFCL V3 Multi-Turn** 上的表现（单位：%）

| Model | 方法 | Average | Base | Miss Func | Miss Param | Long Context |
|-------|------|---------|------|-----------|------------|--------------|
| Qwen3-4B | CREST | **52.00** | **67.0** | 48.0 | **38.0** | **60.0** |
| | MT-GRPO | 49.25 | 63.0 | 46.0 | 35.0 | 53.0 |
| | EnvTuning | 47.25 | 60.0 | 47.0 | 32.0 | 50.0 |
| | GRPO | 43.63 | 53.5 | 42.5 | 31.5 | 47.0 |

#### 在 **WildToolBench** 上的表现（单位：%）

| Model | 方法 | Task Acc. | **Session Acc.** |
|-------|------|-----------|------------------|
| Qwen3-4B | CREST | **48.44** | **7.03** |
| | MT-GRPO | 43.36 | 6.25 |
| | GRPO | 40.23 | 4.69 |
| Qwen3-8B | CREST | **52.34** | **9.38** |
| | EnvTuning | 49.02 | 7.81 |

> ✅ CREST 在所有模型尺度和绝大多数子任务上取得 **SOTA 性能**，尤其在 **Long Context** 和 **Session Accuracy** 上提升显著。

### 与基线方法的对比结果

- **相比最强 RL 基线（MT-GRPO / EnvTuning）**：
  - Qwen3-4B 平均提升 **+2.75~+4.75%**
  - 在 Long Context 子集上提升高达 **+7.0%**
  - Session Accuracy 提升 **+0.78~+1.57%**（相对提升约 30–50%）

- **相比蒸馏类方法（OPD/OPSD）**：
  - 显著优于 OPSD（甚至低于 GRPO），验证其 teacher-bounded 限制；
  - OPD 表现尚可但受限于教师能力，在多数任务上仍落后于 CREST。

- **收敛速度更快**（见 Figure 3a）：
  - CREST 在约 20 步内就超过 OPSD 的峰值性能，并持续上升；
  - GRPO 收敛缓慢，OPSD 很快达到平台期并下降。

### 消融实验结果

#### （1）分层信用分解消融（Table 3）

| 方法 | Average |
|------|---------|
| GRPO（baseline） | 43.63 |
| + Inter-turn only | 47.88 |
| + Intra-turn only | 48.75 |
| + Both（CREST） | **52.00** |

> ✅ 两个层级互补：**Inter-turn** 解决跨轮混淆，**Intra-turn** 解决轮内 token 不区分问题，二者缺一不可。

#### （2）门控机制消融（Table 4）

| 变体 | Average | Long Context | Miss Param |
|------|---------|--------------|------------|
| CREST（完整） | **52.00** | **60.0** | **38.0** |
| w/o Direction Gate | 46.75 | 52.0 | 28.0 |
| w/o Entropy Gate | 46.25 | 49.0 | 27.0 |
| w/o Both | 43.50 | 51.0 | 27.0 |

> ✅ 两个 gate 至关重要：
> - 移除 **direction gate** 导致 teacher override verifier，性能下降明显；
> - 移除 **entropy gate** 导致梯度集中于 format token，长序列表现恶化。

---

## 4. 关键结论和发现

### 论文的主要发现

1. **教师无需决定方向也能发挥作用**：  
   自教师可以仅作为“**幅度放大器（magnitude amplifier）**”，在不改变梯度方向的前提下，选择性增强高不确定性 token 的更新强度，从而实现更高效的信用分配。

2. **分层信用分配是多轮 Agent 训练的关键**：  
   必须同时处理 **inter-turn** 和 **intra-turn** 两个层面的信用模糊问题，单一层次无法充分建模复杂决策结构。

3. **verifier-bounded 与 dense signal 可兼得**：  
   CREST 成功打破了“要么稀疏奖励但可验证，要么密集监督但受限于教师”的两难困境。

4. **entropy 是控制梯度分布的有效代理**：  
   使用 surprisal（$-\log p$）作为 token 不确定性的轻量级指标，能有效引导梯度流向真正需要学习的内容 token。

### 方法的局限性

1. **依赖局部可验证性**：  
   当前方法要求每轮都能独立打分（turn-level verification），难以直接应用于仅在会话结束才给出全局奖励的任务。

2. **熵门是启发式设计**：  
   surprisal 虽然有效，但并非最优的重要性度量，某些高价值 token 可能因已学好而熵低，被错误抑制。

3. **评估范围有限**：  
   实验集中在 4B 和 8B 规模模型及两个 benchmark，更大模型或更多场景下的泛化性有待验证。

### 未来工作方向

1. **扩展至其他结构化生成任务**：  
   如 multi-hop retrieval、collaborative dialogue 等具有层级结果结构的任务。

2. **探索动态 teacher influence scheduling**：  
   自适应调整 $\lambda$ 在不同训练阶段的强度，例如初期强调探索，后期精细调优。

3. **融合更先进的 reward design**：  
   结合 progress reward（如 EnvTuning）与 CREST 的 credit assignment，进一步提升效率。

4. **研究在线 co-training 架构**：  
   探索 student 与 self-teacher 的协同演化机制，而非固定 teacher 或简单 EMA 更新。

---

> **总结一句话**：  
> CREST 提出了一种“**方向归验证器，幅度归自教师**”的新范式，实现了**既密集又安全**的信用分配，在多轮多步 LLM Agent 训练中取得了全面领先的性能。

</details>

---

### 15. [LLM-Guided Graph Generation for Structure-Based Local Improvement Methods](https://arxiv.org/abs/2608.13333)

**Authors**: Hai Xia, Vaidyanathan Peruvemba Ramaswamy, Stefan Szeider  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.13333v1  

#### Abstract
Large neighborhood search normally selects a random subset of decision variables for iterative optimization. For efficiently solving different problems, researchers tend to design variable selection strategies by taking into account structural features from different domains. In this paper, we build...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*LLM-Guided Graph Generation for Structure-Based Local Improvement Methods*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **Structure-Based Local Improvement Methods**（SLIM）在组合优化中依赖专家手动设计变量选择策略和邻域结构，这一过程高度依赖领域知识，且难以泛化到不同问题类型。此外，算法配置选择通常局限于单一问题内部，缺乏跨问题的通用性。

本文旨在解决以下挑战：
- 如何自动化地从约束模型中提取语义结构以指导局部搜索；
- 如何构建一个**问题无关**（problem-agnostic）的框架，适用于任意 MiniZinc 格式的优化问题；
- 如何实现跨问题的统一特征表示，从而支持通用的算法选择机制。

---

### 提出的新方法与新思路

1. **LLM-Guided Graph Generation Pipeline**
   - 利用 **Large Language Model**（LLM，具体为 Claude Opus 4.5）作为“语义编译器”，将 MiniZinc 模型（`.mzn` 文件）自动转换为一个 Python 程序——即 **Graph Generator**。
   - 该生成器能将任意实例映射为统一的加权图结构：
     - **节点**：代表决策变量，带有权重 $ w \in [0,1] $（重要性）和 domain size（取值数量）；
     - **边**：表示变量间的约束关系，权重反映耦合强度（coupling strength）。
   - 图结构是**问题无关的**（problem-agnostic），所有问题共享相同的表示格式。

2. **基于图的通用 SLIM 框架**
   - 在生成的图上运行两种通用的变量提取策略：
     - **BFS Extraction**：从高权重节点出发，按边权重扩展，捕捉局部强关联结构；
     - **LNS Extraction**：基于节点权重随机采样，不考虑拓扑连接。
   - 支持动态冻结部分变量，交由子求解器（如 Gurobi）优化子问题。

3. **跨问题算法选择机制**
   - 所有问题共享同一套 **54维图特征**（包括拓扑统计、权重分布、domain size 特征等）；
   - 使用这些特征训练机器学习模型进行 **Algorithm Configuration Selection**，从 30 种 SLIM 配置中选出最适合当前实例的策略组合（如 budget、timeout、extractor 类型）。

---

### 相比现有方法的优势

| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| 变量选择设计 | 手工工程，需大量领域知识 | 自动化生成，无需专家干预 |
| 表示通用性 | 问题特定结构 | 统一加权图表示，跨问题通用 |
| 算法选择范围 | 单一问题内选择 | 跨20个异构问题统一选择 |
| 可审计性 | 黑箱神经网络策略 | LLM生成确定性代码，可审查验证 |
| 泛化能力 | 限于特定问题 | 支持所有 MiniZinc 兼容问题 |

> ✅ **核心优势**：将 SLIM 从“专家驱动”转变为“自动化、可扩展”的通用优化框架。

---

## 2. 核心实验方法和设置

### 数据集
- 来源于 **MiniZinc Challenge**（2008–2025）中的 20 个不同问题类型，例如：
  - `tdtsp`, `spot5`, `community-detection`, `rectangular-packing`, `VRP`, `RCPSP` 等。
- 过滤条件：
  1. Gurobi 能在 10 分钟内找到可行解，但在 60 分钟内无法证明最优；
  2. 每类问题至少有 5 个符合条件的实例。
- 最终共使用数百个实例，训练/测试按 70:30 分割，分层抽样确保各类问题均衡。

---

### 实验设置

| 组件 | 设置说明 |
|------|----------|
| **主求解流程** | SLIM + Gurobi 子求解器，总时间预算 60 分钟 |
| **初始解获取** | 启发式或短超时运行 Gurobi 得到 |
| **SLIM 配置空间** | 共 30 种配置：<br>• 局部预算 `budget ∈ {10,20,50,70,100,200}`<br>• 超时 `timeout ∈ {20,30,45,60}` 秒<br>• 提取方式：BFS 或 LNS |
| **算法选择器** | 五种 ML 方法：<br>• 回归（regression）<br>• 集成（ensemble）<br>• 二分类（binary）<br>• 多分类（classification）<br>• 两阶段（two_stage）<br>均基于 **Random Forest** 模型 |
| **特征提取** | 从统一图中提取 **54维特征**，涵盖：<br>• 图拓扑统计（密度、直径等）<br>• 节点/边权重分布<br>• domain size 统计与相关性<br>• 数值参数元数据 |
| **训练策略** | 使用 **problem-weighted sampling**，防止大问题（如 RCPSP 占 38%）主导训练 |

---

### 评估指标

- **Problem-weighted win rate (%)**：
  - 对每个问题计算 SLIM 优于、等于、劣于基线的比例；
  - 各问题权重相同，避免数据不平衡影响；
  - 主要衡量算法选择的整体有效性。
- **Net score = Win − Loss**
- **Tie rate**
- **消融实验**：逐步移除配置和特征，观察性能变化。

---

### 基线方法对比

| 基线 | 描述 |
|------|------|
| **One-shot Gurobi** | 在原始问题上直接运行 Gurobi 60 分钟，不进行迭代优化 |
| **Best Single Configuration** | 在 30 个固定 SLIM 配置中选择在整个测试集上表现最好的一个（非自适应） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 方法 | 平均 problem-weighted win rate |
|------|-------------------------------|
| **One-shot Gurobi（基线）** | —— |
| **Best single SLIM config** | **19.3%** |
| **Algorithm Selection（平均）** | **39.5%** |
| **+ Configuration & Feature Ablation** | **44.0%** ✅ |

> 🔺 性能提升超过 **2倍于最佳单配置**！

---

### 各算法选择器表现（Table 1）

| 方法 | Seed 51 | Seed 52 | Seed 53 | 平均 |
|------|--------|--------|--------|-----|
| regression | 40.1% | 40.2% | 34.2% | ~38.2% |
| ensemble | 38.3% | **40.6%** | 35.3% | ~38.1% |
| binary | 36.4% | 36.9% | **37.9%** | ~37.1% |
| classification | 36.9% | 39.0% | 36.2% | ~37.4% |
| two_stage | 34.3% | 37.2% | 31.9% | ~34.5% |
| **Best single config** | 17.8% | 17.7% | 22.3% | **19.3%** |

- 所有选择器显著优于单配置基线（p < 0.01）；
- **regression** 和 **ensemble** 表现最佳，适合预测 per-configuration improvement margin；
- **binary selector** 最稳定（波动仅 ±2%），适合鲁棒场景。

---

### 消融实验结果

- **Configuration Ablation**：
  - 移除对交叉验证性能无益的 SLIM 配置后，平均 win rate 提升约 3–5 个百分点；
- **Feature Ablation**：
  - 在精简后的配置集中进一步剔除冗余特征，最终保留关键特征子集；
  - 结果显示：即使减少特征维度，性能仍可维持甚至略有上升；
- **联合消融**使最终性能达到 **44.0%**，验证了特征与配置设计的有效性。

---

### 按问题类型的胜负情况（Figure 2）
- 在多个问题上 SLIM 显著领先：
  - `tdtsp`, `spot5`, `community-detection`, `triangular`, `opd`：胜率 >75%
- 少数问题表现较差：
  - `rectangle-packing`, `VRP`：输多赢少
- 特殊情况：
  - `filters`, `grid-coloring`：虽未获胜，但全部打平（tie = 100%），说明 SLIM 至少不退化

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **LLM 可有效充当“语义解析器”**：
   - 通过提示工程引导 LLM 生成可靠的 graph generator，成功捕获约束模型中的语义结构；
   - 生成的图可用于指导变量选择，显著优于随机策略。

2. ✅ **统一图表示支持真正的跨问题通用优化**：
   - 不同问题共享同一图格式与特征集，使得算法选择可在异构问题间迁移；
   - 是迈向“通用优化框架”的重要一步。

3. ✅ **算法选择大幅提升性能**：
   - 自适应选择配置比固定策略提升超过 2 倍；
   - 验证了“结构感知 + 自适应控制”的协同效应。

4. ✅ **框架具备良好可扩展性和可维护性**：
   - 新问题只需提供 `.mzn` 模型即可接入整个 pipeline；
   - 无需重新训练模型或修改核心代码。

---

### 方法的局限性

1. ❌ **并非所有问题都能超越 Gurobi**：
   - 在 `rectangle-packing` 和 `VRP` 上表现不佳，可能因图生成未能充分建模关键结构；
   - 表明 LLM 生成的质量仍有改进空间。

2. ❌ **图生成依赖 LLM 输出质量**：
   - 尽管经过验证，但仍属于近似语义建模；
   - 若 prompt 设计不当，可能导致错误的权重分配或遗漏关键约束。

3. ❌ **静态图表示限制动态调整能力**：
   - 当前图在预处理阶段生成，搜索过程中不变；
   - 缺乏在线反馈机制（如 tabu、权重衰减）来避免重复探索。

4. ❌ **评估仅限于 MiniZinc 竞赛基准**：
   - 实际工业问题可能存在更复杂的建模模式，尚未验证泛化能力。

---

### 未来工作方向

1. 🔄 **动态图更新机制**：
   - 引入搜索历史信息，在运行时调整节点/边权重（类似 tabu search 或强化学习）；
   - 实现“turbocharged”SLIM，具备自适应能力。

2. 🔍 **增强 LLM 提示设计**：
   - 加入 probing 技术，让 LLM 先分析模型行为再生成图；
   - 引导其识别关键子结构（如 clique、chain、tree-like patterns）。

3. 🧠 **结合神经符号方法**：
   - 使用 GNN 对生成的图进行嵌入学习，辅助 extraction 或 configuration selection；
   - 探索 hybrid symbolic-neural 架构。

4. 🌐 **扩展至其他建模语言**：
   - 将 pipeline 推广至 AMPL、OPL 或 Pyomo 等格式；
   - 构建真正的“通用约束优化前端”。

5. ⚙️ **集成至开源求解器生态**：
   - 与 Chuffed、OR-Tools 等集成，推动自动化优化工具普及。

---

> 💡 **总体评价**：  
> 本论文提出了一条新颖且实用的技术路径——利用 LLM 实现从约束模型到结构化搜索空间的自动编译，极大降低了 SLIM 方法的应用门槛，并首次实现了跨问题的通用优化框架。其实验充分、设计严谨，为未来自动化求解器的发展提供了重要范式。

</details>

---

### 16. [AlayaWorld: Interactive Long-Horizon World Modeling - Full Technical Report (v1.1)](https://arxiv.org/abs/2608.13492)

**Authors**: AlayaWorld Team, Kaipeng Zhang, Chuanhao Li, Yifan Zhan, Yongtao Ge, Yuanyang Yin, Jiaming Tan, Kang He, Liaoyuan Fan, Mingliang Zhai, Ruicong Liu, Xiaojie Xu, Xuangeng Chu, Zhen Li, Zhengyuan Lin, Zhixiang Wang, Zian Meng, Zihui Gao  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.13492v1  

#### Abstract
This report presents an improved version of AlayaWorld. While the backbone architecture, chunk-wise autoregressive generation scheme, and training data remain unchanged from the previous release, we substantially revise how conditioning signals are represented and integrated into the model. The new ...

---

### 17. [SAEVerbalizer: Generating Explanations for Sparse Autoencoder Features via Representation Verbalization](https://arxiv.org/abs/2608.13538)

**Authors**: Weihan Meng, Hongzhu Guo, Yi Jing, Dewen Liu, Zijun Yao, Xiaozhi Wang, Lei Hou, Juanzi Li  
**Category**: cs.CL  
**Published**: 2026-08-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.13538v1  

#### Abstract
Sparse autoencoders (SAEs) are proposed to extract numerous features from large language model (LLM) representations, yet explaining these features still relies primarily on external observation. This reliance leads to superficial explanations inferred from observed model behavior and computational ...

---

### 18. [CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/abs/2608.12629)

**Authors**: Zihao Ye, Yingyi Huang, Hongyi Jin, Bohan Hou, Junru Shao, Zhongming Yu, Jinqi Chen, Meghan Cowan, Shiyi Cao, Shanli Xing, Hanfeng Chen, Vinod Grover, Tianqi Chen, Luis Ceze  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.12629v1  

#### Abstract
GPU kernel agents and GPU programming languages have advanced separately, leaving expert kernels difficult to reproduce. Agents usually treat the compiler as a fixed black box and receive only errors, correctness outcomes, and timing, while existing DSLs either hide critical scheduling decisions or ...

---

### 19. [Finding the Needle in a Haystack: Test-Time Analog Circuit Representation Adaptation for Bayesian Optimization](https://arxiv.org/abs/2608.12687)

**Authors**: Fin Amin, Sounak Dutta, Paul D. Franzon  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.12687v1  

#### Abstract
Bayesian optimization (BO) is a sample-efficient framework for analog circuit topology search, where evaluating each candidate topology can require costly simulation. However, representation-based BO methods typically treat circuit embeddings as fixed after encoder training. This creates a mismatch ...

---

### 20. [TANGCO: Learning Topology-Aware Capacity Allocation for Overload-driven Cascading Failures](https://arxiv.org/abs/2608.13212)

**Authors**: Orkun Irsoy, Leman Akoglu, Osman Yagan  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.13212v1  

#### Abstract
Networked systems, from power grids to traffic networks and cloud clusters, carry loads across nodes with limited capacity. A node whose load exceeds its capacity fails and sheds its load onto its neighbors, which can trigger a system-wide cascade. We study how to allocate a fixed capacity budget ac...

---

### 21. [TsuGO: Probing Search Efficiency in LLM Reasoning via Go Life-and-Death Problems](https://arxiv.org/abs/2608.13221)

**Authors**: Shunwen Bai, Ziping Ma, Chaoyang Zhang, Yarong Wang, Jiale Liu, Zhen Qin, Qingpei Guo  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.13221v1  

#### Abstract
The evaluation of LLM reasoning is moving from final-answer accuracy to process-level assessment, yet existing methods still fail to capture how models plan reasoning paths and allocate reasoning resources--that is, how they organize search. Prior process-level methods focus on the coherence and red...

---

### 22. [Training Under Challenge: Executable Certificates and Challenge-Closed Optimality for Neural Networks](https://arxiv.org/abs/2608.12655)

**Authors**: Farhang Yeganegi, Arian Eamaz, Mojtaba Soltanalian  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.12655v1  

#### Abstract
A flat training curve does not reveal whether a neural network has reached a global optimum, is locally trapped, is representation-limited, or is mismatched to its trainer. We introduce Training Under Challenge, an executable-certificate framework in which predeclared, architecture-valid procedures ...

---

### 23. [HiRoute: Hierarchical Routed Prompt Tuning for Safety Alignment of Large Language Models](https://arxiv.org/abs/2608.12821)

**Authors**: Fangzhou Chen, Shiji Zhao, Mengyang Wang, Qihui Zhu, Ranjie Duan, Maoxun Yuan, Xingxing Wei  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.12821v1  

#### Abstract
Large language models (LLMs) remain vulnerable to harmful requests and jailbreak attacks. Parameter-efficient safety alignment methods based on prompt tuning typically rely on a single global prompt or externally selected prompt modules. Such static designs struggle to maintain a cross-category safe...

---

### 24. [Learning the Mathematical Property for Designing Low Mutual Coherence Binary Sensing Matrices](https://arxiv.org/abs/2608.12982)

**Authors**: Rekha, Santosh Singh, S. K. Neogy  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.12982v1  

#### Abstract
In this research work, we are constructing the sensing matrix, which is essential for the success of the compressive sensing technique. We have chosen a learning-based technique for the construction of the sensing matrix. The novelty and uniqueness of the proposed technique is that it does not use a...

---

### 25. [Balanced Adaptive Prototype Selection for Scalable TabPFN Inference on Large-Scale Tabular Data](https://arxiv.org/abs/2608.12989)

**Authors**: Mahboobe Jadid, Melika Rezaye Garkani, Ali Mousavi  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.12989v1  

#### Abstract
Pretrained tabular foundation models have demonstrated strong predictive capability; however, their application to large-scale datasets remains constrained by the limited inference context. This paper introduces Balanced Adaptive Prototype Selection (BAPS), a framework for constructing compact, info...

---

### 26. [EEG Decoding Using CNN and LSTM Network](https://arxiv.org/abs/2608.13285)

**Authors**: Athanasios Karagounis  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.13285v1  

#### Abstract
Motor imagery (MI) brain--computer interfaces (BCIs) have emerged as a promising approach for establishing flexible communication pathways between the human brain and external devices , particularly for individuals affected by stroke or neurodegenerative disorders. Reliable decoding of motor-imagery...

---

### 27. [Sparse Orthogonal Regression Technique: A Spectral Framework for Equation Discovery, Approximation, and Integration](https://arxiv.org/abs/2608.13504)

**Authors**: Sabin Roman, Ljupco Todorovski, Saso Dzeroski  
**Category**: cs.LG  
**Published**: 2026-08-14  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.13504v1  

#### Abstract
We develop the Sparse Orthogonal Regression Technique (SORT), a sparse spectral framework for learning orthonormal-basis expansions from noisy and irregularly sampled data. SORT estimates expansion coefficients directly from observations using L1-regularized regression, avoiding explicit quadrature ...

---

### 28. [$\varepsilon$-MemEvo: Adaptive Cross-Task Memory Transfer for LLM Program Evolution](https://arxiv.org/abs/2608.12522)

**Authors**: Aofan Liu, Shiyuan Song, Yiyan Qi  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.12522v1  

#### Abstract
LLM-based program evolution systems such as FunSearch and AlphaEvolve have shown strong ability to discover novel algorithms, but typically optimize each task in isolation, discarding search experience after completion. We introduce $\varepsilon$-MemEvo, a framework for cross-task knowledge transfer...

---

### 29. [Lines and Ladders: A Context-Aware Multi-Agent Framework for Large-Scale Retail Price Taxonomy](https://arxiv.org/abs/2608.12674)

**Authors**: Ravi Teja Chunduri, Srikaran Reddy Boya, Deep Narayan Mishra, Ajay Kumar B, Karthik Kumaran, Pranay Kona  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.12674v1  

#### Abstract
Maintaining price consistency and executing an Every Day Low Price strategy is critical for global retailers. However, with catalogs spanning millions of active items, manual governance of price relationships is infeasible. Inconsistent pricing across item variants distorts customer value perception...

---

### 30. [EEG-PRIME: Prototype-Aligned Representation Learning with Multi-Level Conditioning for EEG Decoding](https://arxiv.org/abs/2608.13072)

**Authors**: Shuailei Zhang, Muyun Jiang, Wei Zhang, Jinbo Chen, Zhiwei Guo, Yong Li, Yi Ding, Cuntai Guan  
**Category**: cs.AI  
**Published**: 2026-08-14  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.13072v1  

#### Abstract
Electroencephalography (EEG) decoding models often generalize poorly across datasets and subjects due to domain shifts in acquisition protocols and individual neurophysiology. We propose EEG-PRIME, a two-stage EEG foundation model for cross-dataset multi-task decoding. EEG-PRIME combines masked pret...

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

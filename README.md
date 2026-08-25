# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-25 06:08:41 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SAEM: Stage-Aware Expert Management for Memory-Efficient MoE Inference in Chain-of-Thought Reasoning](https://arxiv.org/abs/2608.21614)

**Authors**: Yujie Zhang, Bin Gao, Tulika Mitra  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2608.21614v1  

#### Abstract
Chain-of-thought (CoT) prompting improves LLM reasoning by decomposing complex problems into intermediate steps, but its sequential nature increases decoding latency and memory usage. Mixture-of-Experts (MoE) models scale capacity through sparse expert activation, yet their full expert weights often...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SAEM: Stage-Aware Expert Management for Memory-Efficient MoE Inference in Chain-of-Thought Reasoning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **CoT 推理中的内存与延迟瓶颈**：Chain-of-Thought (CoT) 推理通过分步推理提升 LLM 的准确性，但其长序列生成导致高解码延迟和内存压力。
- **MoE 模型在资源受限下的低效执行**：Mixture-of-Experts (MoE) 模型虽能扩展容量，但全量专家权重常超出 GPU 显存，需频繁进行 GPU-CPU 数据传输。现有运行时系统（如基于 token 级别的缓存）忽略 CoT 中的语义结构，造成不必要的数据移动和缓存污染。

### 提出的新方法与思路
SAEM 是一种**面向推理阶段感知的 MoE 推理运行时系统**，其核心思想是利用 CoT 推理过程中“连续推理阶段具有稳定且可预测的专家激活模式”这一结构性特征，提出三项协同机制：

1. **Stage-Aware Caching（阶段感知缓存）**  
   - 利用轻量级模式匹配检测 CoT 中的**阶段边界**（如 “Alternatively”, “On second thought” 等话语线索）。
   - 在阶段边界处聚合该阶段内各专家的激活频率，并据此更新 GPU 缓存，仅在语义转变时迁移专家，显著减少数据传输。

2. **Expert-Aligned Token Repacking（专家对齐的 Token 重组）**  
   - 将属于同一专家的 tokens 在内存中重新打包为连续批次，消除传统 `gather-compute-scatter` 流程中的碎片化访问。
   - 减少 kernel launch 开销，提高 GPU 利用率，尤其在 top-k 路由较宽（如 Qwen3 使用 top-8）时效果更明显。

3. **In-Situ CPU Execution（原位 CPU 执行）**  
   - 对于未驻留于 GPU 且仅被少量 token 使用的专家，直接在 CPU 上执行，避免高昂的 PCIe 传输开销。
   - 防止低频专家“污染”GPU 缓存，同时将 CPU 从被动存储设备转变为**主动计算单元**。

### 相比现有方法的优势
| 维度 | 传统方法（如 Mixtral-Offloading, DAOP） | SAEM |
|------|----------------------------------------|------|
| 管理粒度 | Token-level 或 sequence-level | **Stage-level**，契合 CoT 结构 |
| 缓存策略 | 基于 LRU 或静态统计 | 动态响应**语义变化**，保留高频专家 |
| 数据传输 | 高频迁移导致大量 PCIe 通信 | 仅在阶段边界更新，大幅降低传输次数 |
| 内存效率 | 存在缓存抖动与碎片 | 更高的 cache hit ratio 和 GPU 利用率 |
| 整体性能 | 受限于调度开销与 kernel 分裂 | 实现更高吞吐量 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MATH-500**：包含 500 道奥数级别数学题，作为主基准测试集。
- **AIME 2024**：30 道竞赛级数学问题，涵盖代数、几何、组合等，用于评估挑战性场景。
- **GPQA-Diamond**：198 道研究生水平多选题，覆盖物理、化学、生物领域，检验跨学科泛化能力。

### 实验设置
- **模型**：
  - `Qwen3-30B-A3B`：48 层 MoE，每层 128 个专家，top-8 路由。
  - `ERNIE-4.5-21B-A3B`：28 层 MoE，每层 64 个专家，top-6 路由。
- **硬件平台**：
  - 单节点：NVIDIA A100 GPU (80GB HBM)，Intel Xeon Gold 6326 CPU (16核)，PCIe 4.0×16 连接。
  - 主机内存 512GB DDR4。
- **评估指标**：
  - **End-to-end throughput**：以 tokens/sec 衡量。
  - **Cache Hit Ratio (CHR)**：衡量缓存有效性。
  - **Expert Cache Ratio (ECR)**：GPU 驻留专家占比（控制变量），范围从 3.125% 到 50%。
- **实现框架**：基于 Hugging Face Transformers + PyTorch 实现。

### 基线方法对比
| 基线方法 | 特点 |
|---------|------|
| **MoE-OnDemand** | GPU 保留非 MoE 层和主导专家，其余按需加载 |
| **Mixtral-Offloading** | LRU 驱动的 token-level 缓存机制 |
| **Fiddler** | 在 CPU 上执行非驻留专家，减少传输 |
| **DAOP** | 基于序列级激活预测，在 CPU 预计算专家输出 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **MATH-500** 上平均实现 **1.60× (Qwen3)** 和 **1.47× (ERNIE-4.5)** 吞吐提升。
- 在所有任务和配置下，相比最强 baseline 平均达到 **1.33× 吞吐加速**；当校准数据与工作负载匹配时可达 **1.54×**。
- 在极端低 ECR 条件下（如 3.125%），SAEM 相比 Mixtral-Offloading 在 batch size=8 时仍取得 **2.10× 吞吐增益**。

### 与基线方法的对比结果
| 方法 | 相对性能（MATH-500 平均） |
|------|--------------------------|
| SAEM vs. Mixtral-Offloading | **+60% ~ 110%** |
| SAEM vs. DAOP | **+170%**（平均 2.70×） |
| SAEM vs. Fiddler | **+60%**（平均 1.60×） |

> 图表显示 SAEM 在不同 batch size 和 ECR 下均优于所有 baseline，尤其在小 batch 和低 ECR 场景优势更为显著。

### 消融实验结果（Ablation Study）
在 Qwen3 上、ECR=12.5% 条件下的消融分析（Table IV）：

| 技术组合 | Batch=1 吞吐 (tokens/s) | Speedup | Batch=8 吞吐 (tokens/s) | Speedup |
|--------|------------------------|--------|------------------------|--------|
| 最佳 baseline | 3.45 | — | 8.51 | — |
| 仅 Stage-Aware Cache Update | 4.81 | 1.39× | 10.44 | 1.23× |
| 仅 In-Situ CPU Execution | 2.73 | 0.79× | 8.51 | 1.00× |
| 仅 Token Repacking | 5.02 | 1.46× | 9.09 | 1.07× |
| **全部组件联合使用** | **6.10** | **1.77×** | **13.82** | **1.62×** |

> ✅ **结论**：三个组件互补，协同作用带来最大收益；其中 **token repacking** 和 **stage-aware caching** 是主要驱动力，而 **in-situ execution** 是使能机制（防止缓存污染）。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **CoT 推理存在强阶段级专家激活一致性**：相邻推理阶段间专家激活模式相似度高达 **~89.3%**（Temporal Coherence Score），表明可利用历史阶段指导未来缓存决策。
2. ✅ **细粒度管理不适用于结构化推理**：token-level 缓存无法捕捉语义连续性，导致过度迁移和资源浪费。
3. ✅ **阶段感知调度显著降低数据传输开销**：SAEM 通过仅在阶段边界更新缓存，有效减少了 PCIe 数据移动。
4. ✅ **混合 CPU-GPU 执行可行且高效**：合理利用 CPU 计算资源可避免冗余传输，提升整体利用率。

### 方法的局限性
- **依赖显式话语线索**：当前阶段边界检测依赖于如 “Alternatively”、“Instead” 等文本提示，若模型生成风格不含此类标记，则检测精度下降。
- **初始化依赖校准数据**：transition cue 集合和初始缓存依赖离线 calibration，可能影响跨领域迁移表现。
- **尚未支持超长上下文**：目前设计针对数百至数千 token 的 CoT，对万级长度的扩展性有待验证。

### 未来工作方向
- 🔄 **隐式边界检测机制**：探索基于专家激活熵变、注意力分布突变等信号自动识别推理阶段转换。
- 🌐 **跨模型/任务自适应机制**：构建无需人工标注 cue 的通用 stage detection 模块。
- 🔮 **更精准的专家预测器**：结合 cross-layer gate 或 meta-controller 实现多步 ahead 的专家需求预测。
- 💾 **持久化缓存与预热机制**：支持跨请求的专家缓存复用，进一步降低冷启动开销。

---

> **总结一句话**：SAEM 通过**感知 CoT 推理的阶段性结构**，实现了**局部性驱动的 MoE 推理优化**，在有限 GPU 内存下显著提升了推理吞吐，为复杂推理任务提供了高效的 MoE 部署方案。

</details>

---

### 2. [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308)

**Authors**: Zeyun Zhong, Joya Chen, Manuel Martin, Frederik Diederichs, Juergen Gall, Juergen Beyerer  
**Category**: cs.LG  
**Published**: 2026-08-25  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2608.21308v1  

#### Abstract
Test-Time Training (TTT) enables long-context processing via continuous weight updates during inference, but current methods struggle to balance the expressivity of per-token update dynamics with the hardware efficiency of chunk-wise approximations. We propose E$^2$-TTT (Expressive and Efficient TTT...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Rethinking Expressivity and Efficiency in Test-Time Training

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Test-Time Training (TTT)** 方法在**表达力（expressivity）** 和 **硬件效率（efficiency）** 之间存在根本性权衡：
- **Token-wise TTT**：逐token更新，具有强大的建模能力，但因序列依赖导致训练缓慢，硬件利用率低。
- **Chunk-wise TTT**（如 LaCT）：以chunk为单位更新，提升计算效率，但通过简化更新规则（如对动量和衰减因子取平均）牺牲了token级别的动态变化，丢失了时间结构。

### 提出的新方法：E²-TTT (Expressive and Efficient TTT)
本文提出 **E²-TTT**，旨在**同时实现高表达力和高效率**，其核心创新是：
- **闭式并行化标量核（Closed-form Parallel Scalar Kernel）**：在标准近似（梯度在chunk起始权重处计算）下，推导出一个**精确的闭式状态转移方程**，该方程能够**完全复现**逐token递归更新在chunk结束时的快速权重（fast-weight）和动量状态。
- **保留时间结构**：该方法在chunk级别实现了完全并行化训练，同时**完整保留了逐token更新规则中的时间结构**（如学习率、动量、衰减的token级变化），而这是之前chunk-wise方法所丢弃的。

### 相比现有方法的优势
- **表达力不降**：性能媲美甚至超越基于全注意力或混合架构的基线模型，尤其在长上下文任务中表现优异。
- **效率不损**：训练吞吐量与高效的chunk-wise方法（如LaCT）相当，解决了TTT的序列瓶颈问题。
- **长度外推性强**：在远超训练长度的序列上仍能保持高性能，显著优于现有方法。

---

## 2. 核心实验方法和设置

### 数据集
- **预训练数据**：使用 `HuggingFace FineWeb-Edu` 数据集，共训练 **15B tokens**。
- **评估任务**：
  - **通用语言建模**：WikiText, LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-e/c。
  - **上下文检索**：FDA, SWDE, SQuAD。
  - **长度外推测试**：
    - 合成任务：`S-NIAH-1` (Passkey Retrieval) 和 `S-NIAH-2` (Numerical Needle in Haystack)，测试长度最高达 **16K tokens**（8倍于训练长度）。
    - 真实世界长上下文：`LongBench`，包含14个真实世界的长文本任务。
  - **多模态扩展**：在 `LLaVA-Video-178K` 子集上进行视频理解微调，并在 `VideoMMMU` 和 `LongVideoBench` 上评估。

### 实验设置和评估指标
- **模型规模**：训练了 **340M** 和 **1.3B** 参数的模型。
- **序列长度**：训练序列长度为 **2K tokens**（1.3B模型为2.24K）。
- **Chunk大小**：统一设置为 **512**。
- **评估指标**：
  - 语言建模：**Perplexity (↓)**。
  - 推理与检索：**Zero-shot Accuracy (↑)**。
  - 长度外推：在不同长度下的准确率。

### 基线方法对比
- **纯Attention**：`Transformer++` (Llama风格)。
- **线性Attention**：`DeltaNet`, `Mamba2`。
- **混合架构**：
  - `HQLT`：DeltaNet + Sliding Window Attention (SWA)。
  - `LaCT`：Chunk-wise TTT + SWA（最强基线之一）。
- **其他TTT变体**：`Titans`。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
#### 通用语言建模
- 在 **1.3B** 模型上，E²-TTTMLP 取得了最低的困惑度（**Wiki: 19.5**, **LAMBADA: 15.3**），零样本平均准确率达到 **54.5%**，优于所有基线。

#### 上下文检索能力
- 在真实检索任务（FDA, SWDE, SQuAD）上，**E²-TTTswiGLU** 在1.3B模型上的平均准确率达到 **43.6%**，大幅领先于 `HQLT` (35.5%) 和 `LaCT` (36.7%)。

#### 长度外推能力（核心优势）
- **S-NIAH-1 (Passkey Retrieval)**：
  - 在 **16K tokens**（8倍训练长度）时，**E²-TTTswiGLU** 仍保持 **93.6%** 的准确率。
  - 对比基线：`LaCT` 几乎崩溃至 **3.0%**，`HQLT` 下降至 **25.2%**。
- **S-NIAH-2**：
  - 在 **8K tokens** 时，E²-TTTswiGLU 达到 **40.6%**，而 `HQLT` 仅为 **15.4%**，`LaCT` 为 **5.6%**。
- **LongBench**：
  - E²-TTTswiGLU 平均得分为 **14.1%**，显著高于 `HQLT` (12.1%)、`Mamba2` (10.3%) 和 `LaCT` (7.7%)。

#### 训练效率
- **训练吞吐量**：E²-TTT 的训练吞吐量与高度优化的 `LaCT` 非常接近，在1.3B模型上仅慢 **3.3%**，证明了其高效性。
- **推理成本**：作为循环模型，其解码延迟和内存占用不随序列长度增长而增加，优于Attention模型。

### 消融实验结果
- **更新规则消融**：
  - 将token级的动量和衰减因子“坍缩”为chunk级标量（`Chunk-averaged`），在S-NIAH-1@8K的准确率从 **95.4%** 骤降至 **6.8%**，证明了保留token级动态的重要性。
  - `LaCT-Matched`（将LaCT规则移植到本框架）在训练长度内尚可，但在16K时也崩溃至 **0.0%**。
- **组件消融**：
  - 移除 `Sliding Window Attention` 分支，模型在长距离检索上完全失效（0.0%）。
  - 移除 `TTT` 分支，模型无法处理长距离依赖。
  - 两者结合的混合架构是成功的关键。
- **鲁棒性**：方法对基础超参数（如 `η_base`, `α_base`）在较宽范围内表现稳定。

---

## 4. 关键结论和发现

### 主要发现
1. **表达力与效率可以兼得**：E²-TTT 成功地在chunk-wise的计算效率下，实现了token-wise更新的高表达力，打破了TTT领域的经典权衡。
2. **精确的时间结构至关重要**：在长上下文任务，尤其是长度外推中，**保留token级的学习率、动量和衰减的动态变化**是取得优越性能的关键。简单的chunk平均会严重损害模型能力。
3. **混合架构的有效性**：将E²-TTT与 `Sliding Window Attention` 结合，能同时捕捉局部精细依赖和全局长距离依赖，是处理长序列任务的理想方案。
4. **强大的长度外推性**：E²-TTT 在远超训练长度的序列上表现出惊人的鲁棒性，为构建真正无限上下文的模型提供了可行路径。

### 方法的局限性
1. **Chunk内的“盲点”**：由于输出步骤使用的是前一个chunk结束时的权重（`O[r] = f_{W[r-1]}(Q[r])`），当前chunk内的token无法通过TTT路径影响彼此，形成了一个“盲点”。虽然通过SWA分支缓解，但这仍是chunk-wise方法的固有缺陷。
2. **模型规模限制**：当前实验集中在 **1.3B** 及以下规模，将其扩展到更大规模（如10B+）的可行性有待验证。
3. **输出步骤非并行**：尽管训练过程完全并行，但推理时的输出生成仍是自回归的。

### 未来工作方向
- 设计一种**高效的token-wise输出机制**，以解决chunk内的“盲点”问题。
- 将E²-TTT扩展到**更大规模的模型**和更复杂的任务（如长视频理解、复杂Agent任务）。
- 探索将此闭式并行化思想应用于其他具有复杂递归动态的序列模型。

</details>

---

### 3. [PowerSlider: Exploiting Phase Asymmetry for LLM Serving under Demand Response](https://arxiv.org/abs/2608.21719)

**Authors**: Yueying Li, Jiayang Chen, Yuanfan Chen, Leo Han, Haoran Qiu, Esha Choukse, Rodrigo Fonseca, Udit Gupta  
**Category**: cs.DC  
**Published**: 2026-08-25  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2608.21719v1  

#### Abstract
AI inference clusters are increasingly constrained by instantaneous power, not just energy: grid operators condition new capacity on demand response, imposing time-varying power caps. Existing LLM serving systems optimize a static energy objective or shed fixed priority tiers under load; either way,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**PowerSlider: Exploiting Phase Asymmetry for LLM Serving under Demand Response**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代生成式 AI 推理集群面临电网运营商提出的 **Demand Response（需求响应）** 要求，即在电网压力期间必须遵守动态变化的瞬时功率上限 `P_max(t)`。传统系统通常优化静态能耗目标或采用固定优先级负载削减策略，在面对时间变化的功率限制时，**goodput（有效吞吐量）急剧下降**。

此外，大型语言模型（LLM）推理管道具有显著的阶段异构性：
- **Prefill 阶段**：计算密集型（compute-bound），对 GPU 频率敏感；
- **Answer Decode 阶段**：内存带宽受限（memory-bandwidth-bound），可在低频下维持高吞吐；
- **Reasoning 模型的 Thinking 阶段**：KV-cache 占用大，影响调度和并发能力。

现有系统无法有效利用这些差异来应对动态功率约束。

---

### 提出的新方法与创新思路

**PowerSlider** 是首个为在线 LLM 推理服务设计的支持动态功率上限的系统，其核心创新包括：

#### （1）**Flex SLO 合同机制**
- 引入灵活的服务等级目标（Flexible SLO），允许部分请求（如 Flex 类）在需求响应窗口内接受有界延迟退化。
- 定义 `(α, p)` 合同：最多 `p` 比例时间内延迟不超过 `α` 倍基准延迟。
- 将用户可容忍的“延迟松弛”转化为优化中的**凸约束**，实现资源分配与服务质量之间的量化权衡。

#### （2）**PTA Disaggregation 架构**
- 扩展传统的 Prefill-Decode（PD）分离架构，提出 **Prefill-Think-Answer（PTA）三阶段解耦**。
- 为每个阶段设立独立的 GPU 池，并支持 per-stage 的频率控制（DVFS）和 KV-cache 分区管理。
- 隔离 Thinking 阶段可避免长 KV-chain 对 Answer Decode 的干扰，提升批处理效率。

#### （3）**基于 KKT 条件的在线优化求解器（PSOpt）**
- 将联合资源配置问题建模为一个带约束的凸优化问题，目标是最小化用户感知影响（impact）和 GPU 重分配开销。
- 利用 **Karush-Kuhn-Tucker (KKT)** 条件推导出闭式解，实现在 **7.7ms 内完成重新求解**，远快于通用求解器。
- 动态决定如何在不同阶段、SLO 类别之间分配有限的功率预算，优先降低“每瓦性能损失最小”的组件频率。

#### （4）**双层运行时控制系统（PSSched + PSRoute）**
- **PSSched**：以分钟级粒度调整各阶段 GPU 数量，采用 drain-before-reassign 协议确保无状态丢失。
- **PSRoute**：基于观察到的实际输出长度进行自适应准入控制，而非依赖易错的预测路由。

---

### 相比现有方法的优势

| 维度 | PowerSlider | 现有方法（如 POLCA、SplitWise、DynamoLLM） |
|------|------------|----------------------------------------|
| **动态功率支持** | ✅ 支持实时变化的 `P_max(t)` | ❌ 多数仅支持静态功率预算 |
| **阶段感知控制** | ✅ per-stage DVFS + KV 控制 | ❌ 统一频率或粗粒度控制 |
| **SLO 灵活性** | ✅ 可配置的 Flex SLO 合同 | ❌ 固定优先级或硬 SLO |
| **推理工作负载支持** | ✅ 显式处理 Thinking 阶段 | ❌ 忽视 KV-cache 积累效应 |
| **响应速度** | ✅ 7.7ms 在线求解，适应快速变化 | ❌ 依赖离线配置或慢速迭代 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **真实生产流量轨迹**：
  - Azure LLM inference traces（代码生成与聊天）
  - Magpie-Reasoning 和 S1K reasoning traces
- 请求特征包含输入/输出 token 长度分布、到达模式（bursty gamma process）、CoV（变异系数）等。

### 实验设置
- **硬件平台**：
  - 物理测试床：DGX-A100 和 GH200 服务器
  - 模拟扩展至 64–512 GPUs，使用修改版 SplitwiseSim 进行事件驱动模拟
- **模型配置**：
  - Llama-70B, CodeLlama-34B, Qwen-32B, DeepSeek-R1 等
  - Tensor Parallelism (TP) 设置为 4
- **功率模型**：
  - GPU 功率建模为频率的立方函数：`P(f) = c₂f³ + c₁f + c₀`
  - 基于 SGLang profiling 数据拟合，R² > 0.95

### 评估指标
| 指标 | 描述 |
|------|------|
| **Goodput** | 成功满足 SLO 的请求占比（分 LC/Flex/BE 类别） |
| **P90/P99 TTFAT** | Time To First Answer Token 的尾部延迟 |
| **TTLT** | Time To Last Token，端到端完成延迟 |
| **BE Throughput** | Best-effort 类请求的归一化吞吐 |
| **Power Tracking Accuracy** | 实际功耗是否始终低于 `P_max(t)` |

### 基线方法对比
| 基线名称 | 架构 | DVFS 策略 | 调度器 | 是否支持多 SLO |
|--------|------|-----------|--------|----------------|
| **B1 Uniform** | PD disag | 全局统一频率 | Overlap-KV JSQ | 单一类 |
| **B2 POLCA** | Collocated | 优先级感知 DVFS | Priority-aware Mixed-JSQ | HP/LP 二元分级 |
| **B3 SplitWise+** | PD disag | 阶段感知比例调节 | Overlap-KV JSQ | +Priority |
| **B4 DynamoLLM+** | Collocated | 统一 DVFS | Prediction-based routing | SLO-aware |
| **B5 SLOs-Serve+** | Collocated | 统一 DVFS | ProMax | LC/BE 优先级 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）静态功率削减下的 Goodput 表现
- 在 **非推理工作负载** 下：
  - PowerSlider 在 **60% 功率削减** 下仍能保持 **100% online goodput**。
  - 其他基线在 40% 削减后均跌至 15% 以下。
- 在 **高负载推理工作负载**（128-GPU，QPS=60）下：
  - **30% 功率削减** 时：
    - PowerSlider：**78.3% online goodput**, **54% BE goodput**
    - 最佳基线（SLOs-Serve+）：47.6% online, 0% BE
    - ➜ **提升 1.64× online goodput**

#### （2）尾部延迟表现（P90）
- 在 60% 功率削减下：
  - **LC 类 TTFAT**：仅增加 **1.3×**（从 42s → 54s）
  - **Flex 类 TTFAT**：仅增加 **1.3×**
  - 基线中最佳者（POLCA）延迟达 **2.3–6×**，最差超过 **12×**

#### （3）真实电网应急日回放测试（CAISO EEA-3 日）
- 功率下限低至 **0.41× nominal**
- PowerSlider：
  - 全天平均 **≥98% online goodput**
  - 波谷时刻仍维持 **92% goodput**
- 所有基线在波谷处崩溃至 **<7% goodput**（最低为 0%）

#### （4）能量效率
- 在 60% 功率削减下：
  - PowerSlider 能耗为 **0.42 J/token**，优于未受限制时的 **0.44 J/token**
  - 基线因大量请求超时或失败，**有效能耗上升约 3 倍**

---

### 消融实验结果（Ablation Study）

通过移除关键组件验证其作用：

| 组件移除 | 30% 功率削减下 LC TTLT 增幅 | 影响说明 |
|---------|----------------------------|----------|
| **-KKT Solver** | ↑74% | 缺乏最优功率再分配，导致 prefill 性能受损 |
| **-PTA Disaggregation** | ↑100% | Thinking 干扰 decode，引发严重排队 |
| **-Flex Contract** | ↑59% | 无法利用弹性请求释放功率空间 |

✅ 结论：三大组件缺一不可，协同作用才能实现鲁棒性能。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 推理管道存在显著的阶段不对称性**：
   - Prefill 对频率高度敏感（线性吞吐下降）
   - Answer Decode 在低频下吞吐稳定（可达 0.57× nominal）
   - Thinking 阶段引入 KV-cache 容量瓶颈，需隔离处理

2. **Phase Asymmetry 是应对动态功率的关键杠杆**：
   - PowerSlider 通过将功率从“昂贵”阶段（如 prefill）转移到“便宜”阶段（如 answer decode），最大化每瓦性能收益。

3. **Flex SLO 合同是可行且高效的接口**：
   - 商业 API 已提供类似机制（如 Anthropic Fast/Standard、Google Gemini Flex）
   - 合同参数 `(α, p)` 不需精细调优即可获得大部分增益。

4. **在线优化是必要的**：
   - 配置空间巨大（~10¹⁸ 种组合），离线剖面不适用。
   - KKT 求解器实现了 **亚毫秒级决策延迟**，足以跟踪每 5 分钟更新的电网信号。

5. **系统具备经济可行性**：
   - 在 ERCOT 等市场定价下，PowerSlider 在所有深度下均为净正收益。
   - 基线系统在 27–34% 削减深度即转为亏损。

---

### 方法的局限性
1. **静态功耗限制了极致节能**：
   - 当频率降至 `f_min` 后，进一步节能需依赖 consolidation 或 power-gating。
   - 当前系统在极深 cap 下仍受限于静态功耗（约占总功耗 40%）。

2. **依赖硬件 DVFS 接口延迟较高**：
   - NVIDIA `nvmlDeviceSetGpuLockedClocks` 调用耗时数十毫秒，迫使使用 vote-commit 批处理协议。

3. **KV-transfer 开销虽被掩盖但仍存在**：
   - PTA 架构引入额外 think→answer KV 传输，尽管通过持久会话池和 RDMA 优化，仍有一定开销。

---

### 未来工作方向
1. **更细粒度的频率域控制**：
   - 支持 per-SM 或 per-partition 的 DVFS，实现微秒级响应，适配 AGC 等高频调节场景。

2. **硬件级功率封顶与计量支持**：
   - 需要在芯片层面集成可信功率封顶机制，提供可审计的合同履约保证。

3. **支持更多类型的弹性 SLO**：
   - 如精度缩放（precision scaling）、早期退出（early exit）等，进一步扩大功率调节空间。

4. **跨数据中心协同调度**：
   - 将 PowerSlider 扩展至集群间层级，结合碳感知调度（carbon-aware scheduling）实现全局优化。

---

> ✅ **总结一句话**：  
> **PowerSlider 通过揭示并利用 LLM 推理中“阶段不对称性”与“SLO 弹性”的双重自由度，首次实现了在动态功率约束下接近无损的在线服务性能，为 AI 集群参与电网需求响应提供了实用化路径。**

</details>

---

### 4. [BF1: A Causal Dyadic Sparse-Attention Retrofit for Efficient Long-Context Transformers](https://arxiv.org/abs/2608.20427)

**Authors**: Hina Dixit  
**Category**: cs.LG  
**Published**: 2026-08-25  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2608.20427v1  

#### Abstract
Dense causal attention remains expensive at long context even when implemented with highly optimized exact kernels. We study BF1, a deterministic block-aligned dyadic sparse-attention route that combines a small exact local neighborhood, a global first block, and logarithmically spaced historical bl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BF1: A Causal Dyadic Sparse-Attention Retrofit for Efficient Long-Context Transformers

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **Transformer 自注意力机制** 在长上下文场景下存在严重的效率瓶颈：其计算和内存开销随序列长度呈 **二次方增长 $O(n^2)$**，即使使用 FlashAttention 等优化内核也难以根本解决。本文聚焦于如何在不破坏预训练模型能力的前提下，通过稀疏化注意力模式来提升长上下文推理的系统效率。

### 提出的新方法：BF1
BF1 是一种**确定性的、块对齐的因果多尺度稀疏注意力路由（causal dyadic sparse-attention route）**，其核心设计如下：

- **Block-aligned 结构**：将输入序列划分为固定宽度的 block（如 64 tokens），在 block 粒度上定义注意力连接。
- **混合读取策略**：每个 query block 可以访问：
  - 自身 block 内的 token（token-level causal）
  - 若干个最近邻的局部前驱 block（local neighborhood）
  - 全局第一个 block（global first block）
  - 按照 dyadic（即 $2^k$ 距离）指数间隔的历史 block（logarithmically spaced historical blocks）

该结构结合了局部性、全局可及性和多尺度历史感知，在保证信息传播路径的同时大幅减少 token 对之间的交互数量。

### 创新点与优势
| 维度 | 创新点 |
|------|--------|
| **Operator 设计** | 不是提出全新的稀疏模式，而是构建了一个**可复现、数据无关、块对齐的公共路由规范**，便于部署和比较。 |
| **Pretrained Model Retrofit** | 首次实现**正确性保障下的 BF16 精度预训练模型微调适配**，支持梯度反向传播与数值一致性验证。 |
| **系统级分析** | 将每层稀疏度与整模型延迟（whole-model latency）关联，提供从 kernel 到 system 的完整性能画像。 |
| **通信深度理论分析** | 证明 BF1 图结构具有 $O(\log n)$ 的最短路径通信深度，优于随机图或滑动窗口。 |
| **匹配控制实验** | 在参数量、训练步数、token 序列顺序等完全一致条件下，与其他稀疏拓扑进行公平对比。 |

相比现有方法（如 Longformer、BigBird、LogSparse、Random Sparse），BF1 的优势在于：
- **确定性路由** → 更易编译优化和硬件调度；
- **结构化非局部连接** → 支持高效的信息传递；
- **保留关键历史节点**（首块 + 指数回溯）→ 更适合语言建模中的长期依赖捕捉。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练数据**：基于 ORCA-style 的 49,000 行问答语料（Mukherjee et al., 2023），经清洗后按内容哈希分割为 train / select / report 三部分。
- **评估数据**：固定的 174 条 packed sequences，共包含 356,178 个预测 tokens，用于最终报告 perplexity 测评。

### 实验设置
| 项目 | 设置详情 |
|------|----------|
| **模型架构** | Qwen3-0.6B-Base：28 层 decoder，16 query heads，8 KV heads，head dim=128 |
| **稀疏化范围** | 仅替换中间 8 层 attention 层为 BF1，其余 20 层保持 dense global attention |
| **Block 宽度** | $b = 64$ |
| **硬件平台** | NVIDIA RTX PRO 6000 Blackwell GPU（~95 GiB 显存，compute cap 12.0） |
| **软件栈** | PyTorch 2.13.0 + CUDA 13.0，使用 FlexAttention 编译自定义 mask，BF16 精度 |
| **训练协议** | Stage A Adaptation：<br>- 总训练 token 数：16.384M<br>- 优化器步数：1,000 steps<br>- 目标函数：next-token loss + teacher logit distillation + hidden state matching<br>- 学生与教师均初始化自同一 checkpoint |

### 评估指标
| 类型 | 指标 |
|------|------|
| **系统性能** | - per-layer prefill latency<br>- whole-model Time To First Token (TTFT)<br>- kernel 执行时间 vs planning 开销分解 |
| **模型质量** | - Report Perplexity（越低越好）<br>- paired bootstrap confidence interval 分析 |
| **理论性质** | - Selected interaction 复杂度 $O(n \log n)$<br>- Communication depth（最大最短路径 hop 数） |

### 基线方法对比
所有稀疏方法在 selected pair 数量上严格对齐（equal-pair-count），确保物理计算负载相近：

| 方法 | 描述 |
|------|------|
| **Dense-CT** | Dense Continued Training：相同层可训练，但稀疏混合系数为 0（即仍为 dense attention） |
| **Sliding (Local)** | 固定大小的连续局部上下文窗口（equal-budget local sliding） |
| **Static-Random Nonlocal Graph** | 随机选择非局部 block 连接，seed=17 固定以隔离变量 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 系统效率（Prefill 加速）
| 上下文长度 | 相对于 Dense SDPA 的加速比 | Interaction Reduction（交互减少倍数） |
|------------|-------------------------------|----------------------------------------|
| 8K         | 2.84×                         | —                                      |
| 16K        | 5.50×                         | —                                      |
| **32K**    | **10.91×**                    | **26.98×**                             |

> ⚠️ 注：实际加速未达交互减少比例，说明 metadata 开销和 kernel 启动成本仍有优化空间。

#### ✅ 整体模型 TTFT 提升（Warm 模式）
在真实八层 retrofit 下，warm TTFT 显著下降：

| 上下文长度 | 原始 Dense 路径 | BF1 Retrofit | 加速比 | TTFT 降低幅度 |
|------------|------------------|---------------|--------|----------------|
| 8K         | 69.0 ms          | 63.7 ms       | 1.08×  | **7.7%**       |
| 16K        | 190.1 ms         | 168.6 ms      | 1.13×  | **11.3%**      |
| **32K**    | **589.1 ms**     | **499.1 ms**  | **1.18×** | **15.3%**      |

> 🔺 若假设所有 28 层均为 BF1（timing-only 模拟），32K 下可达 **2.14×** 加速，表明进一步扩展覆盖层可带来更大收益。

#### ✅ 通信深度优势
在相同 selected pair 预算下，不同拓扑的最大最短路径 hop 数对比：

| 方法 | 32K 上下文下的最大 hop 数 |
|------|----------------------------|
| **BF1** | **8**                      |
| Static-Random Graph 17 | 14             |
| Matched Sliding | 59                   |

👉 BF1 具有更优的信息传播效率，尤其显著优于局部滑动窗口。

---

### 与基线方法的对比结果（Language Modeling）

#### 报告困惑度（Report Perplexity）——越低越好

| 方法 | Seed 1234 | Seed 2026 | Seed 3407 | **Mean** |
|------|-----------|-----------|-----------|----------|
| **BF1 (dyadic)** | 1.68634 | 1.68622 | 1.68660 | **1.68639** |
| Static-Random (graph 17) | 1.69145 | 1.69128 | 1.69188 | 1.69154 |
| Dense-CT (blend=0) | 1.69241 | 1.69226 | 1.69308 | 1.69258 |
| Matched Sliding | 1.81512 | 1.81486 | 1.81517 | 1.81505 |

✅ **BF1 在全部三个训练种子下均排名第一**。

#### 相对 BF1 的性能差距（paired interval，seed=1234）
| 对比项 | 平均相对 PPL 上升 | 95% 置信区间 |
|--------|--------------------|--------------|
| Static-Random Graph 17 | +0.3030% | [+0.2441%, +0.3642%] |
| Dense-CT | +0.3599% | [+0.3169%, +0.4055%] |
| Matched Sliding | +7.6367% | [+7.3325%, +7.9554%] |

📌 发现：
- **Non-local connectivity 是关键**：即使是随机非局部连接，也远好于局部滑动窗口（+7.6% 差距）。
- **BF1 拓扑优于随机图**：尽管差距较小（约 0.3%），但在三次重复中稳定胜出。
- **大部分增益来自继续训练**：Dense-CT 已恢复约 92.7% 的绝对改进，说明 domain adaptation 是主因，BF1 提供额外增量。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **BF1 是一种有效的稀疏注意力原语（primitive）**：
   - 在长上下文（>4K）下超越 dense attention；
   - 实现高达 **10.91× per-layer prefill speedup @32K**；
   - 在部分层替换下仍能带来 **15.3% whole-model TTFT 降低**。

2. ✅ **非局部稀疏连接具有“承载作用”（load-bearing）**：
   - 在相同计算预算下，**non-local sparse topology 显著优于 local-only sliding**；
   - 即使是静态随机非局部图，也能取得接近 BF1 的表现。

3. ✅ **BF1 拓扑结构具有优势**：
   - 在控制训练随机性的前提下，**BF1 一致优于 static-random graph**；
   - 其 dyadic + global-first 设计提供了更优的通信深度和信息流特性。

4. ✅ **系统瓶颈正在转移**：
   - 在 selected-page decode 中，**planning 开销成为低 batch 场景下的主导成本**；
   - 当前实现尚未实现端到端 decode 加速，需解决 **persistent plan reuse** 问题。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| ❌ **非全模型亚二次复杂度** | 仅 8/28 层被替换，剩余 dense 层使整体仍为 $\Omega(n^2)$；KV Cache 仍为 $O(n)$ |
| ❌ **未压缩 KV 存储** | BF1 是 sparse-read 方法，不是 memory-compressed 方法 |
| ❌ **缺乏 capability-level 评估** | 未测试 retrieval、multi-hop reasoning、state tracking 等高级能力是否保留 |
| ❌ **图结构鲁棒性待验证** | 仅测试一个 random graph seed（17），无法推广至分布级结论 |
| ❌ **decode 未实现端到端加速** | planning 成本过高，低 batch 下反而变慢 |

---

### 未来工作方向
1. **扩大 retrofit 覆盖范围**：尝试更多层甚至全模型替换，并重新训练以探索真正的 $O(n \log n)$ 模型可行性。
2. **Persistent Plan Reuse**：开发可在多个 token 解码中复用的 attention planning 机制，降低 decode 开销。
3. **Graph Draw Robustness Study**：在多个 random graph seeds 上重复实验，判断 BF1 是否普遍优于随机拓扑。
4. **跨模型/跨任务迁移**：验证 BF1 在更大规模模型（如 Qwen3-7B）、其他语言或领域上的泛化能力。
5. **capability preservation analysis**：专门设计 benchmark 来评估 long-range retrieval、aggregation 和 state tracking 能力。

---

## 总结一句话
> **BF1 是一种经过系统验证的、可用于预训练模型 retrofit 的因果稀疏注意力操作符，在长上下文 prefill 场景中实现了显著加速，并在匹配控制下取得了最优的语言建模性能，标志着稀疏注意力从理论设计走向实用系统的一步重要进展。**

</details>

---

### 5. [KVBoost: Chunk-Level Key-Value Cache Reuse with Deviation-Guided Recomputation for Efficient Large Language Model Inference](https://arxiv.org/abs/2608.21362)

**Authors**: Srihari Unnikrishnan  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.21362v1  

#### Abstract
Transformer-based large language models (LLMs) incur high prefill latency because key-value (KV) tensors must be recomputed for each request. Existing prefix-caching systems reduce this cost but require prompts to share a leading contiguous prefix, limiting effectiveness when shared content appears ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# KVBoost 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **prefix caching** 方法（如 vLLM、SGLang）仅在多个请求共享一个**连续前缀**时才能复用 Key-Value（KV）缓存，这在现实场景中存在严重限制。例如：
- 共享内容可能出现在提示的任意位置（如检索到的文档块、系统提示后接个性化查询）；
- 多个非连续共享段落无法被有效利用。

因此，这类方法在实际部署中的 **cache hit rate** 很低。

KVBoost 提出了一种更灵活的缓存机制，旨在解决以下核心挑战：
- 如何实现**任意位置**的共享文本 KV 缓存复用？
- 如何处理因拼接独立缓存块导致的 **seam error**（注意力边界错误）？
- 如何兼容 RoPE（Rotary Positional Embeddings）的位置编码？

---

### 提出的新方法与创新点

KVBoost 是一个面向 HuggingFace 兼容解码器模型的 **chunk-level KV cache reuse** 系统，其主要创新包括：

#### （1）双哈希键控机制（Dual-Hash Keying）
- **Prefix Hash**：基于上下文链的哈希，保证完全相同的前置上下文 + 内容，支持**精确复用**。
- **Content Hash**：仅基于 token 内容的哈希，允许跨位置匹配，支持**近似复用**（需修复）。
> ✅ 实现了“内容出现即缓存”的理念，不再依赖于是否为前缀。

#### （2）两种缝合误差修复策略（Seam Repair）
- **SelectiveRecompute**：对每个 chunk 边界前后固定窗口（默认 16 tokens）重新计算，空间局部修复。
- **CacheBlendRecompute**（推荐）：通过一次前向传播探测 KV 张量的 **cosine deviation**，仅重算偏差最大的 ~15% tokens。
> 🔍 动态识别高偏差 token，显著降低修复开销且能捕捉长程依赖影响。

#### （3）RoPE 位置纠偏
通过注入正确的 `position_ids` 并结合双哈希设计，解决了 KV 张量在不同绝对位置下 RoPE 旋转不一致的问题。

#### （4）其他增强功能
| 特性 | 描述 |
|------|------|
| **Asymmetric KV Quantization** | 采用 KIVI 风格量化（int8/int4），key 按 channel、value 按 token 量化，节省内存 |
| **Adaptive Chunk Splitting** | 自动将 chunk 边界调整至自然语言断点（如句末标点），提升语义连贯性 |
| **Overlap & Sink Tokens** | 缓存时加入重叠 token 和 attention sink token，提高边界 token 质量 |
| **Importance-weighted LRU Eviction** | 使用每 chunk KV 张量的 ℓ² 范数作为重要性权重进行淘汰 |
| **Two-tier Storage** | 支持内存 + 可选 mmap 磁盘溢出层，扩展缓存容量 |

#### （5）完整开源实现
- 开源地址：[https://github.com/pythongiant/kvboost](https://github.com/pythongiant/kvboost)
- 兼容所有 RoPE-based HuggingFace 模型（如 LLaMA、Qwen、Mistral 等）

---

### 相比现有方法的优势

| 维度 | vLLM / SGLang（Prefix Caching） | KVBoost（Chunk-level Reuse） |
|------|-------------------------------|------------------------------|
| 缓存粒度 | Page-level（连续前缀） | Chunk-level（任意位置） |
| 匹配条件 | 必须从 position 0 开始共享 | 任意位置的内容匹配即可 |
| 缓存命中率 | 在非前缀共享场景下极低 | 显著更高，尤其适用于 RAG、多轮对话等混合输入 |
| 修复机制 | 无（仅限精确前缀） | 支持偏差引导修复，保障输出质量 |
| 实际加速效果 | 中等 | 更优（见实验部分） |

> 🚀 核心思想：**Where content appears should not determine whether it can be cached.**

---

## 2. 核心实验方法和设置

### 数据集与任务
- **数据集**：自建 **bug-localization benchmark**，共 1,000 个样本。
- **任务形式**：每个样本包含一段共享代码上下文（163–3,400+ tokens）+ 一道四选一的选择题（A-D）。
- **请求模式**：
  - **Cold 请求**：首次访问某段代码上下文；
  - **Warm 请求**：后续对该上下文提问，可复用缓存。

> 💡 场景典型性：模拟 RAG 或 IDE 插件中“一次加载代码，多次提问”的真实负载。

---

### 实验设置

| 参数 | 设置 |
|------|------|
| **模型** | Qwen/Qwen2.5-3B（3B 参数，RoPE） |
| **精度** | float16 |
| **硬件** | 单卡 NVIDIA RTX 4060（8GB VRAM） |
| **软件栈** | PyTorch + HuggingFace Transformers（无 TensorRT/DeepSpeed 加速） |
| **Chunk Size** | 默认 128 tokens |
| **Recompute Ratio (p)** | CacheBlendRecompute 使用 ~15% tokens |
| **Memory Budget** | 最大缓存字节数受限，支持磁盘溢出 |

---

### 评估指标

| 指标 | 定义 |
|------|------|
| **TTFT**（Time-to-First-Token） | 首 token 生成时间，衡量 prefill 阶段延迟 |
| **Exact Match Accuracy** | 输出选项 A/B/C/D 是否完全正确 |
| **Cache Reuse Ratio** | 从缓存服务的 token 占比 |
| **Peak GPU Memory** | 推理过程最大显存占用 |
| **Speedup** | 相对于 baseline 的 TTFT 加速比 |

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Baseline (Full Recompute)** | 不使用任何缓存，每次完整 prefill |
| **vLLM Prefix Cache** | 使用 PagedAttention 的前缀缓存 |
| **KVBoost** | 本文方法，启用 CacheBlendRecompute + dual-hash |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | Baseline | vLLM Prefix | KVBoost |
|------|----------|-------------|---------|
| **Mean TTFT (ms)** | 639.1 | 165.5 | **142.4** |
| **Median TTFT (ms)** | 440.3 | 80.0 | **76.1** |
| **p95 TTFT (ms)** | 1702.0 | 651.4 | **502.6** |
| **Speedup vs Baseline (Mean)** | 1× | 3.86× | **4.49×** |
| **Speedup vs vLLM (Mean)** | — | 1× | **1.16×** |
| **Exact Match Accuracy** | 99.1% | 99.1% | **99.2%** |
| **Peak GPU Memory (MB)** | 6,140.6 | — | 6,125.8 (-14.8MB) |
| **Avg Cache Reuse Ratio** | 0.0% | 39.5% | 36.4% |

> ✅ **KVBoost 在保持准确率不变的前提下，平均 TTFT 比 baseline 快 4.49×，比 vLLM 快 16%。**

---

### 分桶性能分析（按上下文长度）

| 上下文长度 | BL (ms) | vLLM (ms) | KVBoost (ms) | KvB/BL × | KvB/vL × |
|-----------|--------|----------|------------|----------|----------|
| 0–500     | 162.8  | 74.9     | **48.8**   | **3.34×** | **1.53×** |
| 500–1K    | 285.2  | 100.5    | **73.7**   | **3.87×** | **1.36×** |
| 1K–2K     | 480.2  | 138.8    | **118.9**  | **4.04×** | **1.17×** |
| 2K+       | 1211.3 | 271.2    | **250.2**  | **4.84×** | **1.08×** |

> 🔺 **短上下文优势最明显**：在 0–500 token 段，KVBoost 比 vLLM 快 1.53×，因为此时共享内容常不在开头，vLLM 无法命中。

---

### 缓存复用分布（图7）
- **KVBoost 与 vLLM 均呈双峰分布**：
  - Cold 请求：接近 0% 复用；
  - Warm 请求：峰值在 30–50%，KVBoost 最高可达 80%+。
- 尽管 **vLLM 平均复用率略高（39.5% vs 36.4%）**，但 KVBoost 实际 TTFT 更低，说明其：
  - 缝合修复效率更高；
  - 减少了 PagedAttention 的页管理开销。

---

### 消融实验与关键发现（Section 7）

#### （1）CacheBlendRecompute 的必要性
- 所有 content-hash 匹配必须强制使用 CacheBlendRecompute；
- 否则会引入 RoPE 错位和上下文缺失，导致输出偏差；
- 实验验证：启用后准确率维持在 99.2%，无退化。

#### （2）adaptive splitting 提升语义一致性
- 将 chunk 边界对齐到标点符号附近，减少跨句断裂；
- 提高了边界 token 的 attention fidelity。

#### （3）quantization 影响微小
- int8 量化实现约 2× 内存压缩；
- int4 达到 4× 压缩，质量损失可忽略（accuracy 不变）；
- 对于显存受限场景极具价值。

---

## 4. 关键结论和发现

### 主要结论
1. ✅ **chunk-level KV reuse 可显著提升缓存利用率**，尤其在共享内容分散或非前缀的现实场景中。
2. ✅ **deviation-guided recomputation（CacheBlendRecompute）是一种高效且精准的缝合修复方式**，仅需重算 ~15% tokens 即可恢复全上下文质量。
3. ✅ **双哈希机制成功解耦内容身份与位置身份**，解决了 RoPE 下的位置冲突问题。
4. ✅ **KVBoost 在真实任务上实现了 4.49× 的 TTFT 加速，优于 vLLM 前缀缓存 16%**，同时保持 99.2% 的准确率。
5. ✅ 整体系统模块化、轻量、无需模型修改，**可直接集成进现有 HuggingFace 流水线**。

---

### 局限性
| 限制 | 说明 |
|------|------|
| **仅支持 RoPE 模型** | 不兼容 ALiBi（Falcon/MPT）、绝对位置编码（GPT-2）或滑动窗口注意力 |
| **单 GPU 设计** | 尚未支持 tensor parallelism 下的分布式缓存同步 |
| **chunk size 敏感** | 过小（如 32）导致过多 seam；过大（如 512）降低命中率；默认 128 是经验平衡点 |
| **probe pass 开销** | 当缓存过长（>8K）时，CacheBlend 的 probe pass 本身耗时较高，建议设置阈值跳过低命中请求 |
| **单一任务验证** | 当前仅在 bug-localization 上测试，缺乏长文本生成、摘要等任务的验证 |

---

### 未来工作方向
1. **扩展至 ALiBi 和其他位置编码方案**；
2. **支持多 GPU 缓存分片与同步机制**；
3. **动态调整 chunk size 与 recomputation ratio**，根据内容复杂度自适应优化；
4. **结合 prompt compression 技术**（如 LLMLingua）进一步减少输入长度；
5. **在更多任务上评估**：如多文档摘要、代码补全、RAG QA 等；
6. **探索更高效的 probe-free deviation estimation 方法**，避免额外前向传播。

---

## 总结
KVBoost 提出了一种**真正面向现实部署需求的 KV 缓存复用范式**。它突破了传统 prefix caching 的结构性限制，通过 **dual-hash + deviation-guided repair** 的组合，在不牺牲输出质量的前提下，实现了高达 **4.49× 的 TTFT 加速**，并具备良好的通用性和工程落地能力。该工作为 LLM serving 系统提供了重要的基础设施级优化路径。

</details>

---

### 6. [LLM4LLM: Bridging Kernel Benchmarks and Real Deployment via Closed-Loop Agentic Optimization](https://arxiv.org/abs/2608.21836)

**Authors**: Hui Zeng, Pengfei Yang, Yanxin Chen, Fusong Ju, Xinran Wei  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.21836v1  

#### Abstract
Large language models have become increasingly capable agents for low-level code and kernel optimization, but isolated kernel benchmarks provide only a proxy for the deployment behavior that matters in language-model inference. We identify a benchmark-to-deployment gap: candidate kernels that appear...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM4LLM: Bridging Kernel Benchmarks and Real Deployment via Closed-Loop Agentic Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

论文指出当前基于 **LLM 的 kernel 优化**存在一个关键问题：**benchmark-to-deployment gap**（基准测试到实际部署的差距）。  
即：在孤立的 microbenchmark 中表现优异的候选 kernel，在集成到真实语言模型推理流程后，可能因以下原因导致性能下降甚至运行失败：

- **Cache 状态差异**：microbenchmark 使用冷缓存（cold cache），而真实模型中前序层输出为热缓存（warm cache），内存局部性不同。
- **运行时安全性问题**：孤立测试中越界访问可能不触发错误，但在真实内存布局下会引发 CUDA 错误。
- **阶段行为不一致**：如 `prefill` 和 `decode` 阶段的输入形状、KV cache 状态不同，导致专为某一阶段优化的 kernel 在另一阶段失效。

因此，仅依赖 isolated kernel benchmark 的优化无法保证端到端性能提升。

---

### 🚀 提出的新方法与创新思路

提出 **LLM4LLM** —— 一种 **deployment-aware closed-loop agentic optimization framework**，其核心思想是将真实部署上下文纳入优化闭环。

#### 主要创新点：

1. **Phase-aware Task Extraction**  
   从用户提供的推理脚本出发，通过 profiling 提取对端到端延迟影响最大的模块，并根据 `prefill` 和 `decode` 不同阶段生成独立的优化任务。

2. **Experience-Guided Episodic Search**  
   引入“有经验引导的片段式搜索”机制：
   - 每个 episode 包含 generate → verify → decide 流程；
   - 将验证轨迹压缩为紧凑的经验记录（如有效 tiling、边界掩码、失败模式）；
   - 下一轮搜索基于原始任务 + 经验重启，避免陷入局部修复陷阱。

3. **In-Model Validation & Deployment-Time Acceptance**  
   候选 kernel 必须通过 **在完整模型上下文中进行正确性和延迟验证** 才能被接受，确保其在真实执行环境中有效。

4. **Closed-Loop Optimization Pipeline**  
   构建了一个完整的闭环流程：  
   `Profiling → Task Extraction → Agent Search → In-Model Validation → Patching`  
   实现了从代码生成到部署验收的全链路一致性。

---

### 🔍 相比现有方法的优势

| 方面 | 现有方法（如 KernelBench） | LLM4LLM |
|------|----------------------------|---------|
| 评估方式 | 孤立 kernel 测试（standalone harness） | 在真实模型中验证（in-model validation） |
| 上下文感知 | 忽略 cache、phase、dispatch 开销 | 显式建模 phase、cache state、shape guard |
| 搜索策略 | 单一迭代路径易陷入局部最优 | 经验引导 + 重启机制促进探索多样性 |
| 安全性保障 | 无运行时安全检查 | 编译与运行时安全双重校验 |
| 实际收益 | 可能出现“benchmark 赢、部署输” | 确保端到端延迟真实降低 |

> ✅ **核心优势**：将优化目标从“kernel 级加速”转向“end-to-end 模型级加速”，真正解决部署可用性问题。

---

## 2. 核心实验方法和设置

### 📊 数据集与模型家族

在多个主流语言模型家族上进行评估，涵盖不同架构类型：

- **Transformer-family**: Qwen3-4B/32B, Llama-2-7B/13B
- **State-Space Models**: Mamba(130M/2.8B), Mamba2(130M/2.7B)
- **Recurrent Models**: RecurrentGemma-2B/9B

所有实验均基于真实的推理脚本构建 workload。

---

### ⚙️ 实验设置

| 项目 | 设置说明 |
|------|----------|
| **硬件平台** | NVIDIA A100 (80GB) 和 H100 (80GB HBM3) |
| **起始路径** | PyTorch eager execution |
| **优化粒度** | 模块级别（module-level）hotspot 替换 |
| **任务提取依据** | 基于 profiling 的 workload-weighted hotspot score：<br>$ s(m) = \sum_{p \in P} w_p \frac{t(m,p)}{T(p)} $ |
| **验证流程** | 候选 kernel 先通过 isolated task 测试，再插入完整模型进行 in-context 正确性与延迟测量 |

---

### 📈 评估指标

- **主指标**：**End-to-End Latency**（端到端延迟）
- **辅助指标**：
  - Geometric Mean Speedup（几何平均加速比）
  - Kernel-level latency reduction（针对 attention/mixer 模块）
  - Pass Rate, Fast1/Fast2（KernelBench 标准指标）

---

### 🆚 基线方法对比

| 基线 | 类型 | 说明 |
|------|------|------|
| **Eager Execution** | 基准 | PyTorch 默认 eager 模式 |
| **FlashAttention / FlashAttention-2** | 专家手工优化 kernel | 用于 attention 模块的强 baseline |
| **mamba_ssm / causal-conv1d** | Mamba 官方 fast path | state-space mixer 的高性能实现 |
| **torch.compile** | 编译优化 | PyTorch 自带图优化工具 |
| **KernelBench SOTA 方法** | LLM-based kernel gen | 如 KernelSkill, STARK, QiMeng-Kernel 等 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Model | A100 Speedup | H100 Speedup |
|-------|--------------|--------------|
| Qwen3-4B | **3.03×** | **3.78×** |
| Qwen3-32B | 2.19× | 2.55× |
| Llama-2-7B | 1.86× | 2.22× |
| Mamba(130M) | **22.37×** | **32.79×** |
| Mamba2(2.7B) | 15.23× | 14.04× |
| RecurrentGemma-9B | 1.24× → **7.55×** (H100) |

> ✅ **总体表现**：
> - 在 **A100** 上实现 **3.91× 几何平均加速**
> - 在 **H100** 上实现 **6.98× 几何平均加速**

---

### 🔬 Kernel-Level 性能（KernelBench Level 2 对比）

| 方法 | A100 GeoMean | H100 GeoMean |
|------|---------------|---------------|
| CUDA-L1 | 3.55× | 6.64× |
| KernelSkill | 2.82× | — |
| STARK | 2.69× | 2.592× |
| **LLM4LLM (ours)** | **2.745×** | **2.628×** |

> 💡 尽管 KernelBench 不是主要目标，LLM4LLM 仍达到 SOTA 水平，且更注重 **correctness + deployability**。

---

### 📉 Scope-Matched 对比（Figures 5 & 6）

- **Attention-only latency**（Figure 5）：  
  LLM4LLM 在多种 prompt length 和 decode 场景下接近 FlashAttention 表现，部分场景反超。

- **Mixer-only latency**（Figure 6）：  
  在 Mamba-family 模型中，LLM4LLM 生成的 Triton kernel 在多个配置下 **优于官方 mamba_ssm 实现**，因其能融合更多操作（如 proj + conv + scan）。

---

### 🔍 消融实验（Ablation Study, Table 3）

在 GPT-5.4、Claude Sonnet 4.6、GLM-5 上对比不同策略：

| 方法 | GeoMean Speedup (vs Eager) |
|------|-----------------------------|
| Sample-1 | ~1.0–1.3× |
| Sample-10 | ~2.1× |
| Iter-10 | Pass 率高但 GeoMean 较低（易陷入局部修复） |
| **LLM4LLM (w/ restart, 15 trials)** | **2.153× (GPT), 2.546× (Claude), 2.745× (GLM)** |

> ✅ **关键发现**：
> - **Restart 机制显著提升性能**：相比不重启（w/o restart），GeoMean 平均提升 >20%
> - **增加 trial 数量（10→15）持续带来增益**
> - **Sampling + Feedback + Restart** 三者结合效果最佳

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Benchmark-to-Deployment Gap 真实存在且不可忽视**  
   很多在 microbenchmark 中“更快”的 kernel 在真实模型中反而变慢，甚至崩溃。

2. **Deployment Context 是优化的关键约束**  
   必须考虑：
   - phase semantics（prefill vs decode）
   - cache residency（warm vs cold）
   - memory layout & allocator state
   - dispatch overhead & shape guards

3. **LLM4LLM 实现了跨模型家族的端到端加速**  
   不仅适用于 Transformer，也显著提升了 Mamba 和 RecurrentGemma 等新兴架构的性能。

4. **Profiling-First + Phase-Aware 是高效优化的前提**  
   固定优化 attention 的策略在非 attention 主导模型中效率低下；LLM4LLM 动态识别热点模块，资源分配更合理。

5. **Episodic Restart 提升搜索效率与泛化能力**  
   避免搜索过程被冗长的失败轨迹锁定，保留有用约束的同时鼓励结构创新。

---

### ⚠️ 局限性（Limitations）

- 当前仅支持 **单 GPU 推理场景**，未覆盖 tensor/pipeline parallelism。
- 假设存在代表性 inference script，若实际部署的 prompt distribution 差异大，需重新 profiling。
- 优化成本较高，适合长期复用的部署场景（如服务上线前优化）。
- 依赖底层 LLM 的 coding 能力，对复杂 kernel（如稀疏算子）仍有挑战。

---

### 🔮 未来工作方向

- 扩展至 **分布式 serving 系统**，支持通信开销建模与 multi-tenant 干扰分析。
- 支持 **continuous batching** 与 **dynamic shape** 更复杂的调度环境。
- 引入 **multi-agent collaboration** 进行模块间协同优化。
- 结合 **reinforcement learning** 进一步自动化搜索策略选择。
- 探索 **cross-model knowledge transfer**，使在一个模型上学到的经验可用于相似结构的其他模型。

---

## ✅ 总结一句话

> **LLM4LLM 成功弥合了 LLM 自动生成 kernel 与真实部署之间的鸿沟，通过构建“以真实模型为中心”的闭环优化框架，首次实现了从 isolated benchmark 胜利到 end-to-end inference 加速的可靠转化。**

</details>

---

### 7. [SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning](https://arxiv.org/abs/2608.23493)

**Authors**: Jialong Liu, Yuling Shi, Ning Yang, Xiaodong Gu, Zuchao Li  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.23493v1  

#### Abstract
Self-reflection is a powerful mechanism for credit assignment in human learning, converting sparse outcome feedback into actionable guidance. However, its potential for post-training Large Language Models (LLMs) remains underexplored. We propose Self-Reflective Policy Optimization (SRPO), a framewor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **post-training** 方法（如 **PPO** 和 **GRPO**）在处理**长程推理任务**（long-horizon reasoning）时面临严重挑战，主要原因如下：
- **信用分配问题**（credit assignment）：仅依赖稀疏的终端奖励信号（如成功/失败），导致梯度方差高、样本效率低。
- **熵崩溃**（entropy collapse）：策略在训练中逐渐丧失探索能力，陷入局部最优。
- **缺乏中间监督**：无法对长序列中的具体错误步骤进行有效纠正。

这些问题在数学推理、交互式代理任务（如 WebShop、ALFWorld）等需要多步连贯决策的任务中尤为突出。

---

### 提出的新方法：SRPO（Self-Reflective Policy Optimization）
SRPO 是一种将**自反思机制**（self-reflection）内化为**密集监督信号生成器**的强化学习框架，其核心思想是：
> **让模型自己成为自己的老师**（self-as-teacher），通过反思已完成的轨迹，生成“反思补丁”（reflection patch），并利用该补丁构建一个更强的“教师分布”，再通过**on-policy distillation** 将知识蒸馏回原始策略。

#### 核心流程分为两个阶段：
1. **Stage 1: Reflection-Guided State Augmentation**
   - 模型执行初始轨迹 $ y \sim \pi_\theta(\cdot|x) $，获得稀疏结果 $ o $。
   - 基于 $(x, T, o)$ 生成简洁的 **reflection patch** $ p $，诊断错误并提供可操作指导。
   - 采用 **reset-with-memory** 机制：将 $ p $ **前置**到原始提示 $ x $ 上，形成增强提示 $ \tilde{x} = [p; x] $，避免上下文漂移。

2. **Stage 2: On-Policy Self-Distillation**
   - 学生策略 $ \pi_\theta(\cdot|x) $ 在原始提示下生成 on-policy 轨迹。
   - 教师策略 $ \pi_{\text{teacher}}(\cdot|\tilde{x}) = \pi_\theta(\cdot|[p;x]) $ 在增强提示下，通过 **teacher-forcing** 对学生每一步的 token 打分。
   - 最小化学生与教师之间的 **reverse KL divergence**，实现密集的 token-level 监督。

---

### 相比现有方法的优势
| 特性 | SRPO | 传统 RL (PPO/GRPO) | 外部教师蒸馏 (OPD) | 推理时反思 (Reflexion) |
|------|------|---------------------|----------------------|------------------------|
| 监督密度 | **O(T)** (token-level) | O(1) (episode-level) | O(T) | O(1) |
| 是否需要外部模型 | ❌ 否 | ❌ 否 | ✅ 是 | ❌ 否 |
| 训练时依赖反思 | ✅ 是 | ❌ 否 | ❌ 否 | ❌ 否 |
| **推理时依赖反思** | ❌ **否** | ❌ 否 | ❌ 否 | ✅ 是 |
| 数据效率 | **极高** | 低 | 中 | 低 |
| 计算开销 | **~3.8× 更少 FLOPs** | 高 | 高（需大模型） | 高（推理时翻倍计算） |

> ✅ **关键优势**：SRPO 将稀疏反馈转化为**密集的 hindsight-guided 监督**，无需外部批评者、奖励模型或更大教师模型，且**推理时无额外开销**。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 类别 | 数据集 | 描述 |
|------|-------|------|
| **数学推理** | AIME'24 (30题) | 竞赛级数学题，高难度 |
| | MATH-500 | 500道涵盖多个领域的数学题 |
| | GSM8K | 小学数学应用题 |
| | DeepScaleR | 11,200道跨领域难题，用于测试泛化能力 |
| **长程代理任务** | WebShop | 12k在线购物任务，稀疏成功信号 |
| | ALFWorld | 134个家庭任务（如找物品、加热食物） |
| | SWE-Bench-Lite | 300个真实 GitHub 编程修复任务 |

---

### 实验设置
- **基础模型**：Qwen3-1.5B, Qwen3-8B, Qwen3-32B, Llama-3.1-8B-Instruct
- **训练细节**：
  - 批大小：256 rollouts
  - 学习率：1e-5，余弦退火
  - Clip ratio: 0.2
  - 使用 8x H100 GPU
- **评估指标**：
  - 数学：Pass@1 准确率
  - 代理任务：Success Rate
  - 效率：平均 episode 步数、训练 FLOPs

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **SFT** | 监督微调，使用专家轨迹 |
| **PPO** | 标准强化学习，带价值函数 |
| **GRPO** | 组相对策略优化，无显式价值函数 |
| **OPD** | On-Policy Distillation，使用更大的外部教师（如 Qwen3-32B → 8B） |
| **Reflexion**, **Self-Refine** | 推理时自我反思方法 |
| **SCoRe**, **RISE**, **R3L** | 训练时引入反思的基线方法 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen3-8B 模型）

| 方法 | AIME'24 | WebShop | ALFWorld | SWE-Bench-Lite | 训练 FLOPs |
|------|---------|---------|----------|----------------|------------|
| **GRPO** | 68.0% | 51.2% | 62.7% | 22.1% | 1.0× |
| **OPD (32B teacher)** | 70.0% | 57.3% | 69.2% | 26.4% | 4.0× |
| **OPD (72B teacher)** | 72.5% | 61.8% | 72.4% | 28.6% | 9.0× |
| **SRPO (Ours)** | **73.3%** | **64.7%** | **76.8%** | **31.2%** | **0.26×** |

> ✅ **SRPO 以仅 8% 的训练 FLOPs 超越所有基线**，包括使用 72B 教师模型的蒸馏方法。

---

### 与基线方法的对比结果
- **数学推理**：
  - 在 AIME'24 上比 GRPO 高 **+5.3%**，比最大教师蒸馏高 **+0.8%**。
  - 在 DeepScaleR 上泛化性能提升 **+5.9%**，表明反思有助于识别领域特定失败模式。
- **代理任务**：
  - WebShop 成功率达 **64.7%**（+7.9% vs SFT）。
  - ALFWorld 达 **76.8%**（+5.6% vs Reflexion）。
  - SWE-Bench-Lite 达 **31.2%**（+4.4% vs Reflexion）。
- **效率**：
  - 平均 episode 步数最短（**10.2 步**），说明策略更高效，非盲目试错。

---

### 消融实验结果（AIME'24, Qwen3-8B）

| 配置 | AIME'24 | WebShop | 分析 |
|------|--------|---------|------|
| **SRPO (Full)** | 73.3% | 64.7% | 完整方法 |
| w/o reflection (直接重试) | 65.8% | 54.2% | 反思提供显著指导 |
| w/ verbose reflection (>10点) | 70.0% | 60.3% | 过长反思引入噪声 |
| Forward KL 替代 Reverse KL | 69.4% | 58.6% | **Reverse KL 更优**（mode-seeking） |
| Off-policy 蒸馏（使用教师轨迹） | 68.0% | 55.9% | **On-policy 更稳定** |
| Append reflection（非前置） | 68.5% | 57.4% | **Prepend + Reset 更好** |
| No state reset（迭代式） | 66.3% | 52.8% | 积累上下文有害 |

> ✅ **关键设计选择验证**：
> - **简洁反思**（2–5条）优于冗长反思。
> - **Reverse KL** 比 Forward KL 更适合行为克隆。
> - **reset-with-memory** 显著优于上下文追加。

---

## 4. 关键结论和发现

### 主要发现
1. **自反思可作为高效的监督信号生成器**：模型能可靠生成高质量反思（67% 被 GPT-4 评为 ≥4/5），且反思质量与性能提升强相关（r=0.72）。
2. **自我蒸馏优于外部教师蒸馏**：SRPO 使用自身作为教师，在 AIME'24 上超越 Qwen3-72B 教师蒸馏，且节省 9× FLOPs。
3. **极高的数据和计算效率**：达到 GRPO 相同性能所需 FLOPs 不足 1/10。
4. **缓解灾难性遗忘**：在持续学习场景中，SRPO 在学习新技能后保留旧技能的能力更强（数学能力保留 **95.2%** vs GRPO 的 87.2%）。
5. **提升推理时扩展性**：SRPO 训练的模型在结合 Self-Refine 时，每轮迭代提升 **+1.8%**，优于 GRPO 的 +1.2%。

---

### 方法的局限性
- **依赖反思质量**：若模型本身缺乏基本知识，反思可能无效（23% 失败案例属于“超出能力范围”）。
- **诊断错误风险**：约 35% 的失败反思存在错误归因。
- **通用性待验证**：目前主要在有明确结果反馈的任务上验证，对开放域任务适用性未知。

---

### 未来工作方向
- 引入**外部验证信号**（external verification）提升反思准确性。
- 结合**检索增强反思**（retrieval-augmented reflection）补充知识缺口。
- 扩展至**多模态推理**和**更长工具使用轨迹**。
- 探索在**无明确结果反馈**任务上的应用（如创意写作、对话）。

---

> 🔗 **代码开源**：https://github.com/Galleons2029/SRPO  
> 📚 **论文链接**：https://arxiv.org/abs/2608.23493

</details>

---

### 8. [Industrial-Instruction: An End-to-End Framework for Building Instruction-Tuning and Benchmark Datasets from Industrial Technical Reports](https://arxiv.org/abs/2608.22817)

**Authors**: Parsa Bakhtiari, Hassan Bashiri, Alireza Khalilipour, Masoud Nasiripour, Moharram Challenger  
**Category**: cs.CL  
**Published**: 2026-08-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.22817v1  

#### Abstract
Industrial technical reports contain high-value knowledge for maintenance, troubleshooting, and product engineering, but their heterogeneous structure (dense prose, specifications, tables) makes them difficult to index and reason over with standard retrieval and QA pipelines, and no public instructi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
工业技术报告（如维护手册、产品规格书）包含大量高价值的专业知识，但由于其**异构性强**（密集文本、表格、图表混合）、结构复杂且分散，难以被标准的检索与问答系统有效利用。目前**缺乏基于真实工业文档构建的公开指令微调（Instruction-Tuning）和基准测试（Benchmark）数据集**，导致大语言模型（LLMs）在工业场景下的表现受限。

### **提出了什么新方法或新思路**
本文提出 **Industrial-Instruction**，一个端到端框架，用于从工业技术报告中构建指令微调和基准数据集。其核心创新包括：

- **端到端自动化流水线**：结合布局感知的文本提取（layout-aware extraction）、语义索引构建（semantic indexing）和基于检索增强生成（RAG）的多选题（multiple-choice QA）合成，实现从原始PDF到高质量QA对的全流程自动化。
- **建模五种现实查询-文档关系**：明确设计并生成以下五种场景的样本，以训练和评估模型对检索噪声和多步推理的鲁棒性：
  1. `r0`: Irrelevant retrieval（无关文档）
  2. `r1`: Single-document support（单文档提供线索）
  3. `r2`: Multi-document support（多文档提供线索）
  4. `r3`: Single-document answer（单文档直接给出答案）
  5. `r4`: Multi-document answer（多文档共同构成答案）
- **双版本数据集发布**：使用同一套流水线，分别用**开源模型 Qwen3-30B-A3B-Instruct** 和**闭源前沿模型 Claude-Opus-4.6** 生成两套平行数据集，首次实现了对“开源 vs. 前沿模型”作为数据生成器的直接比较。

### **相比现有方法的优势**
- **实用性与可复现性**：提供了一条可复制、可扩展的路径，将真实世界的企业文档转化为可用于训练和评估小规模LLMs的资源。
- **关注小模型**：专注于参数量小于10B的小型开放LLMs，更符合工业部署的实际需求（计算成本低、易于部署）。
- **数据质量可控**：通过精心设计的提示（prompt）和自动化过滤流程，确保生成数据的质量和多样性。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **原始文档**：906份公开的松下（Panasonic）技术文档，共7,525页。
- **生成的数据集**：
  - **Panasonic Dataset (Qwen版)**：由 Qwen3-30B-A3B-Instruct 生成，经过过滤后得到约13.6k个QA对。
  - **Panasonic Dataset (Claude版)**：由 Claude-Opus-4.6 生成，经过过滤后得到约25.3k个QA对。
- **外部基准**：`FailureSensorIQ`，一个已有的工业领域多选题QA基准。

### **实验设置和评估指标**
- **模型**：
  - **基础模型**：Qwen-4B-Instruct, Phi-3-mini-4k-Instruct。
  - **对比模型**：RAG-Instruct-Llama3-8B。
- **微调方法**：采用**全量微调（full fine-tuning）** 和 **LoRA** 进行对比。
- **检索组件**：使用 `EmbeddingGemma` 作为嵌入模型，`FAISS` 作为检索引擎。
- **评估指标**（针对多选题）：
  - **Set-Match Accuracy**：预测选项集合与真实答案集合完全匹配的比例（忽略顺序），是本研究的主要指标。
  - **F1-Score**：多标签分类的F1分数。
  - **Jaccard Similarity**：预测集与真实集的交并比。
  - **MMLU**：用于评估微调前后模型通用知识的保留情况，防止“灾难性遗忘”（catastrophic forgetting）。

### **基线方法对比**
- **零样本（Zero-shot）**：未微调的基础模型。
- **RAG模式**：使用基础模型配合检索，不进行微调。
- **不同数据生成器**：比较在Qwen生成数据和Claude生成数据上微调的效果差异。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 在 **Qwen-4B-Instruct** 模型上进行全量微调的结果：
  - **Set-Match Accuracy** 从 28.5% 提升至 **42.0%**。
  - **F1-Score** 从 46.6% 提升至 **63.5%**。
  - **Jaccard Similarity** 从 41.6% 提升至 **58.0%**。
- 在 **Claude-4.6生成的数据** 上微调，提升幅度更大：
  - Set-Match Accuracy 从 40.9% 提升至 **56.4%**（绝对提升15.5个百分点）。
- **RAG效果**：无论是否使用RAG，全量微调都带来了显著且一致的性能增益。

### **与基线方法的对比结果**
- **vs. RAG-Instruct-Llama3-8B**：该模型在 `FailureSensorIQ` 上表现尚可（AccPerIBM=33%），但在自建的 Panasonic 基准上几乎失效（Set-Match Accuracy ≈ 1%），表明其知识库与目标领域不匹配。
- **vs. 其他小模型**：Phi-3-mini-4k-Instruct 的基础性能（17.5% Set-Match）远低于 Qwen-4B-Instruct（28.5%）。

### **消融实验结果**
- **LoRA vs. Full Fine-tuning**：LoRA 微调在该任务上**几乎没有带来任何性能提升**（各项指标基本不变），而全量微调则大幅提升性能。这表明对于这种需要深度学习新领域知识的任务，参数高效的微调方法（如LoRA）效果有限。
- **数据生成器的影响**：
  - **数据质量**：Claude-Opus-4.6 生成的原始数据质量更高，过滤率仅为0.5%，而Qwen生成的数据过滤率高达43%。
  - **微调收益**：在Claude数据上微调获得的下游性能增益更大。
  - **成本**：Qwen本地生成成本约 **$3.2**，而Claude API调用成本约 **$330**，相差两个数量级。
  - **通用知识保留**：在Claude数据上微调的模型，其MMLU得分仅下降0.05分；而在Qwen数据上微调的模型，MMLU得分下降1.26分，显示出轻微的“灾难性遗忘”。

---

## **4. 关键结论和发现**

### **主要发现**
1. **工业文档可转化为有效训练数据**：提出的 Industrial-Instruction 框架能够成功地将复杂的工业PDF文档转化为高质量的指令微调数据集。
2. **全量微调对小模型至关重要**：对于小型LLMs（<10B），在特定领域数据上进行**全量微调**是提升其工业问答能力最有效的手段，LoRA等PEFT方法在此场景下无效。
3. **数据生成器的质量直接影响最终效果**：使用更强大的模型（如Claude-Opus-4.6）作为数据生成器，能产出更干净、更高质量的合成数据，从而带来更大的下游性能提升，并更好地保留通用知识。
4. **成本与性能的权衡**：虽然前沿API模型生成的数据质量更高，但开源模型提供了极具成本效益的替代方案，尤其适合预算有限的研究和应用。

### **方法的局限性**
- **对问题重述（perturbation）极度脆弱**：所有在本研究中评估的模型（无论是否微调），在 `FailureSensorIQ` 的重述问题（AccPerIBM）上准确率均为 **0%**。这表明当前的微调数据未能解决模型对问题表述变化的敏感性问题。
- **未处理多模态信息**：当前框架移除了所有图像，仅处理文本和表格，忽略了文档中的视觉信息（如示意图、流程图）。

### **未来工作方向**
- **扩大数据集规模和来源**：整合来自不同行业的技术文档，构建一个大规模、综合性的工业基准。
- **探索高级RAG架构**：超越简单的检索机制，研究更复杂的RAG范式。
- **处理多模态工业文档**：开发能够同时理解和利用文本、表格和图像信息的模型。
- **提高对问题重述的鲁棒性**：在数据生成过程中显式地加入同义改写或对抗性重述的变体，以增强模型的泛化能力。

</details>

---

### 9. [NeuroPrefetcher: Storage-Aware Sparse LLM Inference via Delta Prefetching](https://arxiv.org/abs/2608.22643)

**Authors**: Nobel Dhar, Md Romyull Islam, Xuechen Zhang, Gongjin Sun, Sahidul Islam, Bobin Deng, Kun Suo  
**Category**: cs.DC  
**Published**: 2026-08-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.22643v1  

#### Abstract
Deploying large language models on edge devices is increasingly limited by a widening gap between model size and available memory. Existing approaches such as quantization, smaller models, and offloading can raise the effective memory limit, but they still assume that the model can be compressed or ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# NeuroPrefetcher: Storage-Aware Sparse LLM Inference via Delta Prefetching —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前在边缘设备（edge devices）上部署 **Large Language Models (LLMs)** 面临一个根本性挑战：**模型尺寸增长远超边缘设备内存容量的增长**。传统方法如量化（quantization）、小型化模型设计、权重卸载（offloading）等，都假设模型最终可以被压缩或分片以适应内存预算。

然而，本文聚焦于更困难的场景：**模型始终大于可用内存（model-exceeds-memory）**，此时存储（storage）不再是被动的溢出层，而是推理路径上的主动权重供给源。在这种情况下，**权重从 NVMe 存储加载的延迟成为性能瓶颈**，而现有系统（如 `llama.cpp`）依赖操作系统级的按需分页（demand paging），导致大量 I/O 等待和 GPU 利用率极低。

---

### 🚀 提出的新方法与创新思路

作者提出 **NeuroPrefetcher**，一种面向存储感知的稀疏 LLM 推理系统，其核心是 **Predictive Delta Prefetching（预测性增量预取）**，包含以下三大创新：

#### （1）**早期激活预测器（Early Activation Predictor）**
- 在 **Layer 0 执行后**，使用一个轻量级 GPU-resident 的共享预测器（shared predictor），基于输入 token 的语义特征（token embedding、attention sketch 等），一次性预测所有后续 MLP 层的稀疏激活模式（通过聚类后的 centroid ID 表示）。
- 该预测器仅占基础模型参数的 **2.86%**（约 207M 参数），且无需任务微调即可保持高准确率。

#### （2）**增量式权重预取（Delta Prefetching）**
- 利用 **MLP 激活在连续 token 间具有强时间局部性** 的观察（82–85% 激活神经元持续存在），只预取“新增”所需的权重行（delta rows），而非每次重新加载全部稀疏权重。
- 运行时维护每层的 resident buffer 和 slot map，通过集合差计算 `Δ = A_t \ A_{t-1}` 来确定需从 NVMe 加载的增量部分。

#### （3）**显式的 NVMe-to-GPU 稀疏行传输路径**
- 构建一条应用级调度的 I/O 路径，绕过操作系统的 reactive demand paging。
- 包括：
  - **神经元中心布局（neuron-centric layout）**：将 gate/up/down 投影合并为连续记录，实现单次读取完整 neuron 权重。
  - **异步 I/O 与 CUDA Scatter**：使用 `io_uring` 异步提交请求，通过 pinned staging buffer 和 fused `deinterleave_scatter` 内核高效填充 GPU 缓冲区。
  - **统一内存一致性处理**：在 Jetson 平台上注册 mapped host memory 保证 CPU 写入对 GPU 可见。

#### （4）**内存自适应的稠密/稀疏分区（Memory-Adaptive Dense/Sparse Split）**
- 将前 `d` 层设为稠密执行（fully resident），其余层稀疏执行。
- 自动调节 `d` 以最大化吞吐，在内存紧张时优先保留稀疏缓冲区，避免退化到整层加载。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | NeuroPrefetcher 的改进 |
|------|--------|------------------------|
| `llama.cpp`, `FlexGen` | 依赖 OS demand paging，反应式加载，I/O 密集 | 主动预测 + 应用级预取，减少 90%+ I/O |
| `DejaVu` | 每层独立预测，太晚无法用于 lookahead prefetching | 单次预测暴露全层 I/O 计划，支持提前调度 |
| `PowerInfer` | 假设 CPU DRAM 是额外容量层（不适用于统一内存设备） | 明确支持统一内存架构下的 NVMe 权重供给 |
| `LLM in a Flash`, `Neuralink` | 优化稀疏访问顺序或缓存历史激活，无预测机制 | 引入预测驱动的增量更新机制 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **WikiText-2**：用于评估生成吞吐量和模型质量（perplexity）
- **HellaSwag**：10-shot 下的归一化准确率，评估任务泛化能力

### ⚙️ 实验平台
- **硬件**：NVIDIA Jetson AGX Orin（32GB 统一内存，Arm CPU + Ampere GPU）
- **存储**：Samsung 990 PRO NVMe SSD（7.4 GB/s 顺序读）
- **软件环境**：Linux + CUDA，使用 `io_uring` 实现异步 I/O

### 🧪 模型与稀疏配置
- **主模型**：Mistral-7B-v0.1（FP16，13.49 GiB，32 层 SwiGLU MLP）
- **对比模型**：Llama-3-8B（验证通用性）
- **稀疏度控制**：通过 `K_max` 控制每层最大活跃神经元数：
  - `K_max=5,500` → 62% 稀疏度
  - `K_max=7,168` → 50% 稀疏度
  - `K_max=9,865` → 31% 稀疏度

### 📊 评估指标
- **吞吐量（throughput）**：tokens per second (tok/s)
- **延迟分解（latency breakdown）**：I/O / Compute / Predictor 时间占比
- **I/O 体积**：每 token 从 NVMe 读取的数据量
- **模型质量**：
  - Perplexity（越低越好）
  - HellaSwag 准确率保留率（relative to dense baseline）

### 🆚 基线方法
- `llama.cpp`（FP16 + mmap demand paging）
- `FlexGen`（all-disk 配置）
- `Ollama`（封装 llama.cpp）
- 其他 17 个主流系统（如 vLLM, TensorRT-LLM, PowerInfer 等）进行兼容性分析

---

## 3. 主要实验结果和性能指标

### 📈 吞吐量提升（vs. llama.cpp）

| 内存条件 | NeuroPrefetcher | llama.cpp | 加速比 |
|---------|------------------|------------|--------|
| mem=14G (~11.9G CUDA可用) | **3.22 tok/s** | 0.34 tok/s | **9.5×** |
| 整体内存范围（11–17G） | 1.7 – 3.7 tok/s | 0.2 – 0.3 tok/s | **7.9–12.0×** |
| crossover 后（mem≥13.9G） | ~5.9–7.7 tok/s | 快速上升 | 1.1–1.2×（优势缩小） |

> 💡 **说明**：当模型可完全驻留时，NeuroPrefetcher 优势减弱；但在 **model-exceeds-memory 场景下优势显著**。

---

### 🔁 I/O 优化效果

#### （1）**Delta Reuse Ratio**
- **82–85% 的稀疏行在相邻 token 间复用**，仅需加载 15–18% 的新增行。
- 图 8(a) 显示几乎所有稀疏需求都是 buffer hit。

#### （2）**NVMe 读取量大幅下降**
- 每 token 读取量从接近整模型大小（13.49 GiB）降至：
  - **14.8 GiB 内存时：103 MiB**
  - **9.0 GiB 内存时：~1 GiB**
- 相比整层加载减少 **两个数量级以上**。

---

### ⏱️ 延迟分解（图 9）
- **NVMe I/O 占据每 token 延迟的 80–87%**，即使经过优化仍是主导因素。
- GPU compute 和 predictor inference 占比较小，非瓶颈。
- 随着内存增加，更多层转为稠密执行，延迟从 **593ms/token（9.0G）降至 113ms/token（14.8G）**。

---

### 🧪 消融实验与配置影响

#### （1）稠密/稀疏分割点 `d` 的影响（图 6）
- 存在明显最优值 `d*`：
  - 62% 稀疏度：`d*=14`
  - 50% 稀疏度：`d*=8`
  - 31% 稀疏度：`d*=2`
- 超过 `d*` 后，稀疏层失去 buffer，退化为整层加载，性能急剧下降（"falling cliff"）。

#### （2）不同稀疏度的影响
- 更高稀疏度（62%）带来更高吞吐，但可能轻微影响质量。
- 提供 **内存-吞吐-质量三者间的可控权衡**。

---

### ✅ 模型质量保留（图 10）

| 模型 | 配置 | Perplexity | HellaSwag Accuracy Retention |
|------|------|------------|-------------------------------|
| Mistral-7B | d=20, 62% sparsity | < 7（接近 dense） | **95–96%** |
| Llama-3-8B | d=20, 62% sparsity | 略升 | **92–93%** |

> ✔️ 所有配置均无需任务微调，表明预测器具有良好的跨任务泛化能力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MLP 激活在自回归解码中具有强时间局部性**（82–85% 持续激活），这是实现高效增量预取的基础。
2. **传统 offloading 系统在统一内存边缘设备上表现不佳**：20 个生产级系统中，仅 4 个能运行，其余因内存分配失败、架构不匹配或缺乏 ARM 支持而崩溃。
3. **NeuroPrefetcher 是唯一专为“稀疏+存储后端”设计的系统**，在 model-exceeds-memory 场景下达到最高速度。
4. **预测 + 显式 I/O 调度 > 反应式 demand paging**：将权重移动从被动响应转变为主动规划，是突破 I/O 瓶颈的关键。

---

### ⚠️ 方法的局限性
1. **仍受限于 NVMe I/O 延迟**：尽管优化了数据量，但 I/O 仍占延迟的 80%+，是主要瓶颈。
2. **依赖激活稀疏性**：对 SwiGLU 类模型有效，若模型稀疏性弱则收益降低。
3. **离线准备开销**：需要预先进行激活收集、阈值化和聚类，增加部署复杂度。
4. **统一内存平台特定设计**：staging buffer 的 coherence 处理针对 Jetson 设计，其他平台需适配。

---

### 🔮 未来工作方向
1. **加速 I/O 路径**：
   - 更深的预取流水线（deeper prefetch pipeline）
   - 更强的读取合并（read coalescing）
   - 量化稀疏行格式（quantized sparse-row formats）
2. **提升 overlap 效率**：进一步并行化 I/O 与 compute，减少空闲等待。
3. **扩展至 Attention 层**：目前仅对 MLP 层稀疏化，未来可探索 KV-cache 或 attention 权重的类似机制。
4. **动态稀疏度调整**：根据输入复杂度或内存压力动态调整 `K_max`。

---

## 总结

> **NeuroPrefetcher 的核心洞见是：在模型超出内存的边缘推理场景中，不能仅靠“压缩模型”，而必须“智能管理权重流动”。**

它通过 **早期预测 + 增量预取 + 显式 I/O 调度**，将存储从被动溢出层转变为可编程的权重供给通道，在真实边缘硬件上实现了 **7.9–12.0× 的端到端加速**，是首个专门为 **sparse, storage-backed, memory-constrained edge LLM inference** 设计的有效解决方案。

</details>

---

### 10. [SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality](https://arxiv.org/abs/2608.21952)

**Authors**: Hyunwoo Kim, Byoungchan Ko, Minseok Kang, Minwoo Kim, Dongjin Lee, Jaehoon Lee, Sungroh Yoon, Dahuin Jung  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.21952v1  

#### Abstract
Recent advances in sequence modeling have highlighted Mamba as a state space architecture offering efficient long-range dependency modeling and providing a viable alternative to Transformers. Building upon this, Mamba-2 introduces the Structured State Space Duality (SSD), which integrates recurrent ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Mamba-2 引入了 **Structured State Space Duality (SSD)** 架构，在保持长序列建模能力的同时提升了硬件利用率。然而，SSD 的计算模式具有以下特点，导致传统量化方法（如 GPTQ、Hadamard Rotation）在应用时出现严重精度下降：
- **维度解耦**：模型维度被划分为头数 $H$ 和每头维度 $P$，二者统计特性差异大；
- **激活复用频繁**：$B$, $C$ 等 channel-varying 激活在多个模块中重复使用，导致大量 DRAM 访问；
- **逐元素乘法（Element-wise Multiplication）与矩阵乘法交织**：破坏了低比特 GEMM 执行路径。

因此，如何为 SSD 设计一个**高效且准确的后训练量化（PTQ）框架**成为关键挑战。

### 提出了什么新方法或新思路
本文提出 **SSDi8**，是首个专为 Mamba-2 的 SSD 架构设计的、支持**持久 INT8 路径**（Persistent INT8 Representation Path）的 PTQ 框架。其核心创新包括：

#### （1）稀疏感知的计算重构（Sparse-aware Reformulation）
将原始的 `ChunkState` 模块中的计算顺序从：
$$
\text{State} = X \times (B \odot \text{LUT}_{\text{state}})
$$
重构为：
$$
\text{State}_{\text{INT32}} = Q(X_{\text{scaled}}) \times Q(B), \quad \text{其中 } X_{\text{scaled}} = \text{LUT}_{\text{state}} \odot X
$$
该重构使得：
- 可以对 $X_{\text{scaled}}$ 进行 INT8 量化并直接参与 GEMM；
- 维持全程 INT8 执行路径，避免混合精度开销；
- 利用 $X_{\text{scaled}}$ 的高稀疏性（见 Fig. 3），即使存在离群值也能实现较低量化误差（理论证明见 Appendix A）。

#### （2）通道感知的自适应量化（Channel-aware Quantization）
- 对 $B$, $C$ 在 **group 维度 $G$ 上提前量化**，再广播至 $H$，显著降低量化开销（仅增加约 3% 延迟）；
- 对 $X$, $\text{State}$ 等激活采用 **per-(H, P)** 量化策略，以应对不同 head 间的异质分布；
- 避免对参与后续 GEMM 的状态维度 $N$ 单独量化，防止误差累积。

#### （3）基于通道误差均值的校正机制（Mean Correction）
引入一个轻量级的 **per-channel error correction term** $c$，定义为：
$$
c^* = \arg\min_c \|Y - (Y' + c)\|^2 \Rightarrow c_p = \frac{1}{N}\sum_i (Y - Y')_{i,p}
$$
该修正项通过层间顺序更新算法（Algorithm 1）估计，并仅应用于输出投影层（out-proj），以最小化额外延迟（仅增加 ~1–2%）。

---

### 相比现有方法的优势

| 特性 | Quamba / Quamba2 | SSDi8 |
|------|------------------|--------|
| 是否支持 SSD 内部完整 INT8 路径 | ❌（仅输入/权重） | ✅（全路径） |
| 是否处理 Element-wise × GEMM 交织 | ❌ | ✅（通过 Reformulation） |
| 是否利用 SSD 内在维度分解 | ❌ | ✅（per-H/P/G/N 分别处理） |
| 是否引入误差补偿机制 | ❌ | ✅（Mean Correction） |
| 实际加速效果 | 有限 | 显著（最高达 1.47×） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **零样本任务评估**：
  - LAMBADA (LA)
  - WinoGrande (WG)
  - PIQA
  - HellaSwag (HS)
  - ARC-Easy (Arc-E)
  - ARC-Challenge (Arc-C)
- **语言建模能力评估**：
  - WikiText2（困惑度 PPL）
  - The Pile（长上下文 PPL）
- **校准数据集**：The Pile 中采样 512 条文本用于量化参数校准。

### 实验设置和评估指标
- **模型规模**：Mamba-2 1.3B, 2.7B, 8B
- **量化配置**：
  - W8A8：权重 8-bit，激活 8-bit
  - W4A8：权重 4-bit（GPTQ），激活 8-bit
  - 排除 W4A4 因硬件效率下降（参考 Lin et al.）
- **评估平台**：
  - 主要：NVIDIA A5000 GPU
  - 边缘设备验证：NVIDIA Orin NX 16G
- **评估指标**：
  - 准确率（Zero-shot Accuracy）
  - 困惑度（Perplexity ↓）
  - 推理延迟（Latency ↓）
  - 吞吐量（Throughput ↑）
  - 内存占用

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **FP16** | 全精度基准 |
| **HAD (HadMamba2)** | 应用 Hadamard 变换 + GPTQ 权重量化 |
| **Quamba** | 面向 Mamba-1 的 PTQ 方法 |
| **Quamba2** | 支持 Mamba-2 的 PTQ 方法，但未深入优化 SSD 内部 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）零样本准确率（Tab. 2）
| 模型 | 方法 | Bitwidth | 平均准确率 |
|------|------|---------|------------|
| Mamba-2 2.7B | FP16 | FP16 | 63.8% |
| Mamba-2 2.7B | Quamba2 | W4A8 | 62.1% |
| Mamba-2 2.7B | **SSDi8 (Ours)** | **W4A8** | **62.6%** |
| Mamba-2 8B | FP16 | FP16 | 70.7% |
| Mamba-2 8B | Quamba2 | W8A8 | 69.1% |
| Mamba-2 8B | **SSDi8 (Ours)** | **W8A8** | **69.6%** |

> ✅ **SSDi8 在所有配置下均优于 Quamba2，接近 FP16 性能**

#### （2）困惑度表现（Tab. 3 & 9）
| 模型 | 方法 | Bitwidth | WikiText2 PPL | Pile PPL |
|------|------|---------|--------------|----------|
| Mamba-2 8B | FP16 | FP16 | 7.25 | 444.47 |
| Mamba-2 8B | Quamba2 | W8A8 | 7.79 | 309.18 |
| Mamba-2 8B | **SSDi8** | **W8A8** | **7.49** | **9.04** |

> ✅ **SSDi8 显著缩小与 FP16 的差距，尤其在长上下文稳定性上远超基线**

#### （3）推理速度提升（Fig. 4 & Tab. 4）
在 Mamba-2 2.7B 上（B=32, L=2048）：
- **整体延迟降低**：相比 FP16 提速 **1.47×**
- **相比 Quamba2 提速 1.38×**
- **ChunkScan 模块提速高达 1.77×**

在边缘设备 **Orin NX 16G** 上（L=2048）：
- W4A8 下延迟从 262.90ms（Quamba2）降至 **240.54ms**
- W8A8 下从 249.29ms 降至 **217.69ms**

> ✅ **SSDi8 不仅在高端 GPU 有效，在资源受限设备也具备实用价值**

#### （4）内存占用（Tab. 12）
| 模型 | 方法 | W8A8 内存 |
|------|------|-----------|
| Mamba-2 2.7B | Quamba2 | 2.948 GB |
| Mamba-2 2.7B | **SSDi8** | **2.953 GB**（+0.17%）|

> ✅ **内存开销几乎无增加，仅因存储量化 scale 导致微小上升**

---

### 消融实验结果（Tab. 5 & 6）

#### （1）各组件消融（W4A8, 2.7B, PPL）
| 配置 | Latency (ms) | PPL |
|------|-------------|-----|
| Baseline (FP16 in SSD) | 8.63 | 9.34 |
| + Q(X) only | 8.58 | 9.35 |
| + Sparse Reformulation | 8.05 | 9.37 |
| + Persistent INT8 | 7.60 | 9.39 |
| + ChunkBMM Quant | **6.53** | **9.43** |

> 🔍 **Sparse Reformulation 和 Persistent INT8 是延迟下降主因；PPL 仅轻微上升 <0.1**

#### （2）Mean Correction 消融（LAMBADA, 2.7B）
| 方法 | Accuracy |
|------|----------|
| FP16 | 69.5% |
| SSDi8 w/o correction | 67.2% |
| **SSDi8 w/ correction** | **67.4%** |

> 🔍 **Mean Correction 在极低代价下带来稳定增益**

---

## 4. 关键结论和发现

### 主要发现
1. **SSD 架构对传统量化高度敏感**，源于其独特的维度划分、激活复用和运算交织结构。
2. **通过稀疏感知的计算重构**，可以打破 element-wise 操作对 INT8 GEMM 的阻碍，建立端到端的 INT8 执行路径。
3. **通道感知量化 + 层间误差校正** 能有效缓解量化误差累积，使 W4A8/W8A8 下性能逼近 FP16。
4. **SSDi8 在多种部署场景下均有效**：从数据中心大 batch 推理到边缘设备低功耗运行。

### 方法的局限性
- 当前仅针对 SSD 模块进行优化，未扩展至整个 Mamba-2 模型其他部分（如 MLP）；
- 依赖特定 Triton/CUDA 实现，通用性受限；
- 对极端短序列（如 L=256）加速不明显，因计算强度不足。

### 未来工作方向
- 将 SSDi8 扩展至 **W4A4 或更低比特**（需解决硬件瓶颈）；
- 结合 **Quantization-Aware Training (QAT)** 进一步压缩；
- 探索 **动态量化策略** 以适应不同输入分布；
- 应用于更多基于 SSD 的多模态模型（如 ML-Mamba, Nemotron-H）。

---

> 📌 **总结一句话**：  
> **SSDi8 是首个成功构建 Mamba-2 SSD 持久 INT8 路径的量化框架，通过稀疏感知重构、通道自适应量化与轻量误差校正，在几乎无损精度的前提下实现了高达 1.47× 的推理加速，并展现出良好的跨平台鲁棒性。**

</details>

---

### 11. [CAI-DLLM: Convergence Aware Inference for Diffusion Language Models](https://arxiv.org/abs/2608.22646)

**Authors**: Farhana Amin, Sabiha Afroz, Dimitrios S. Nikolopoulos  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.22646v1  

#### Abstract
Diffusion language models can generate many tokens in parallel, but they still require repeated denoising steps during inference. This makes generation costly, especially when the model continues to recompute tokens that are already stable. To address these limitations, we propose CAI-DLLM, a traini...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CAI-DLLM: Convergence Aware Inference for Diffusion Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **diffusion language models (DLLMs)** 虽然支持并行生成多个 token，但在推理过程中仍需进行多轮去噪（denoising）步骤。这些步骤中，许多已经稳定的 token 会被重复计算，而困难 token 却得不到额外优化，导致**计算资源浪费、推理延迟高、能耗大**。

现有方法如 KV caching 或 early-skipping 仅减少每步开销，但无法动态调整每个 token 的去噪预算。

### 🚀 提出的新方法：CAI-DLLM
提出了一种**无需训练的推理加速方法**——**Convergence Aware Inference for DLLMs (CAI-DLLM)**，其核心思想是利用**第一步去噪时的 confidence 信号**作为 token 难度预测器，并据此实现以下机制：

#### 创新点：
1. **First-step confidence 作为 token-level 控制信号**
   - 使用第 0 步 forward pass 中自然产生的 `c_i(0)`（即最大预测概率）来估计 token 收敛难度。
   - 高 confidence token（>0.5）通常在 5 步内稳定；低 confidence token（<0.25）可能需要 30–50 步。
   - 该信号**零成本获取**，无需额外模型或前向传播。

2. **Block-level 自适应阈值调度（Block Adaptive Scheduling）**
   - 后续输出 block 因有更多已提交上下文，收敛更快。
   - 设计递减的结束阈值 `θ_e(b)`：`0.70 → 0.60 → 0.50 → 0.40` 对应 block 1–4。
   - 更早放松 commit 条件，提升后期块效率。

3. **Token-level 动态预算分配与“磨合阶段”检测**
   - 根据 `c_i(0)` 和位置 `pos(i)` 分配不同最大去噪步数 `B_i ∈ {8, 32, 64}`。
   - 引入 **grinding phase detector**：当连续 K=4 步中新提交 token 数低于 1.5%，则触发提前退出。
   - 避免后期低效迭代。

4. **Position-aware commit rule**
   - 前部 token 收敛更快，因此对后部 token 设置更高的 commit 阈值增益因子（up to 1.1×）。

### 🔍 相比现有方法的优势
| 特性 | CAI-DLLM | DualCache / Fast-dLLM | ES-dLLM |
|------|----------|------------------------|---------|
| 是否需 retrain/fine-tune | ❌ 否 | ❌ 否 | ❌ 否 |
| 是否引入额外模型 | ❌ 否 | ❌ 否 | ❌ 否 |
| 是否修改权重 | ❌ 否 | ❌ 否 | ❌ 否 |
| 是否 per-token 动态控制 | ✅ 是 | ❌ 固定阈值 | ❌ 层级跳过 |
| 是否兼容其他优化 | ✅ 可叠加 KV cache | ✅ | ✅ |
| 是否 plug-and-play | ✅ | ✅ | ✅ |

> ✅ **优势总结**：完全无需训练、无额外参数、不改变模型结构，却能实现细粒度的 adaptive inference。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
涵盖数学、代码、常识推理、长文本等任务：
- **Math**: GSM8K, MathQA
- **Code**: HumanEval, MBPP
- **Reasoning**: BBH (Big-Bench Hard)
- **Commonsense**: PIQA, Winogrande
- **Long-context**: LongBench（Dream-7B 上测试）

### ⚙️ 实验设置
| 参数 | 设置 |
|------|------|
| 模型 | LLaDA-8B-Instruct, Dream-7B-Instruct |
| 硬件 | 单张 NVIDIA H200 SXM5 GPU (141GB HBM3e) |
| 批大小 | 8 |
| 温度 | 0（确定性解码） |
| 输出长度 | 256（reasoning/commonsense），512（code/long-context） |
| Block size | 64（LongBench 用 32） |
| KV cache 更新频率 | LLaDA: 16步，Dream: 8步（沿用 ES-dLLM） |

### 📊 评估指标
| 指标 | 用途 |
|------|------|
| **TPS (Tokens/sec)** | 吞吐量 |
| **Speedup (×)** | 相对于 no-cache 的 wall-clock 时间加速比 |
| **Accuracy** | Exact Match (GSM8K, BBH), Pass@1 (HumanEval, MBPP), F1/ROUGE-L (LongBench) |
| **Energy (Wh)** | 能耗 = 平均功率 × 推理时间（仅 GPU） |
| **Std. Err.** | 报告标准误差以衡量稳定性 |

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **No-cache** | 原始 DLLM 解码，无任何优化 |
| **DualCache (Fast-dLLM)** | KV caching + confidence-based parallel decoding |
| **ES-dLLM** | 早期层 token skipping，降低单步计算量 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 1 & 2）
| 任务 | 模型 | 方法 | Speedup | 准确率 | 能耗 (Wh) |
|------|------|-------|--------|--------|----------|
| GSM8K | LLaDA | **CAI-DLLM** | **18.2×** | **77.41% ↑** | **169 ↓ (−95.3%)** |
| HumanEval | Dream | **CAI-DLLM** | **13.1×** | **48.17% ↑** | **24 ↓ (−93.7%)** |
| BBH | LLaDA | **CAI-DLLM** | **44.8×** | 52.33% (↓4.4pt) | 579 ↓ (−98.0%) |
| MBPP | Dream | **CAI-DLLM** | **20.0×** | 56.60% | 34 ↓ (−96.1%) |
| LongBench | Dream | **CAI-DLLM** | **19.0×** | 23.11 (vs 24.63) | — |

> 💡 **亮点**：
> - 在 **LLaDA GSM8K** 上不仅提速 18.2×，还**提升了准确率**（77.41% vs 76.27%）
> - 在 **Dream HumanEval** 上提速 13.1× 且 **pass@1 更高**（48.17% vs 46.95%）
> - 最高可达 **44.8× wall-clock 加速**（LLaDA BBH）
> - 能耗最多下降 **95.3%**

### 🔬 消融实验结果（Ablation Study）

#### HumanEval 消融（Table 3）
| 组件 | LLaDA TPS | Dream TPS | Dream Pass@1 |
|------|-----------|-----------|--------------|
| ES-dLLM baseline | 139.8 | 56.7 | 41.46% |
| + APD (Adaptive Parallel Decoding) | 424.3 | — | — |
| + Block schedule | 434.2 | — | — |
| + Token budgets ★ | **435.3** | 614.7 | 40.34% |
| Full CAI (gate off) ★ | — | **614.6** | **46.34%** |

> ✅ 发现：
> - **APD 是主要加速来源**（从 139.8 → 424.3 TPS）
> - Dream 模型上启用 confidence gating 会损害 accuracy，故推荐关闭（model-adaptive policy）

#### Denoising Steps Reduction（Table 4）
| 任务 | 模型 | 固定步数 | CAI-DLLM 实际平均步数 | 减少比例 |
|------|------|--------|------------------|----------|
| GSM8K | LLaDA | 256 | 114 | **−55.5%** |
| GSM8K | Dream | 256 | 141 | −44.9% |
| HumanEval | LLaDA | 512 | 315 | −38.5% |
| HumanEval | Dream | 512 | 271 | −47.1% |

> 表明 CAI-DLLM 显著减少了实际执行的去噪步数。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **First-step confidence 是一个强大且免费的 token 难度指示器**  
   - 不需要额外计算即可用于指导整个 decoding 过程。
   - 可靠区分 easy/hard tokens，支撑 adaptive 决策。

2. **adaptive per-token 去噪预算显著优于统一策略**  
   - 允许简单 token 早提交，复杂 token 多迭代，最大化资源利用率。

3. **block-level 上下文越多，收敛越快**  
   - 支持设计递减的 commit 阈值，进一步提升后期 block 效率。

4. **grinding phase 存在且可检测**  
   - 后期大量空转迭代可通过低 yield 检测机制有效终止。

5. **性能增益随序列长度增加而放大**  
   - 在 L=512 时，Dream 模型达到 **15.29× speedup**，远超短序列收益。

### ⚠️ 局限性
1. **accuracy-speed trade-off 存在**
   - 在复杂推理任务（如 BBH）上，accuracy 下降达 4.4 个百分点（LLaDA）。
   - 更适合对延迟敏感而非质量极致敏感的应用场景。

2. **confidence gating 策略依赖模型选择**
   - 当前通过全测试集调参决定是否开启（LLaDA 开，Dream 关），存在过拟合风险。
   - 应在未来使用独立验证集自动决策。

3. **部分任务样本量受限**
   - MathQA 和 LongBench 使用子集（500 / 100 per task）以控制成本，可能影响统计置信度。

4. **硬件平台单一**
   - 所有实验基于 H200 GPU，结果在其他设备上可能不同。

5. **内存开销略有上升**
   - CAI-DLLM 需存储 per-token confidence、budgets 和 lookup tables，带来约 1.5% 内存增长（LLaDA），Dream 因并行更强缓冲区更大，增幅更明显（但仍在可控范围）。

### 🔮 未来工作方向
- 自动化 confidence gating 开关决策（基于小规模验证集）
- 将 CAI-DLLM 与其他 speculative decoding 方法（如 Spiffy）结合
- 扩展至更多类型的 diffusion 架构和多模态生成任务
- 探索更精细的 layer-wise 或 head-wise adaptive 机制

---

> ✅ **总体评价**：  
> CAI-DLLM 是一种简洁、高效、真正 plug-and-play 的推理优化框架，它揭示了 **first-step confidence 的巨大潜力**，为 diffusion LLM 的高效部署提供了新范式，在保持模型不变的前提下实现了数量级的速度提升与能耗下降。

</details>

---

### 12. [GSAR: Goal-State-Anchor Rewards for Mobile GUI Agents with Self-Evolving Data Synthesis](https://arxiv.org/abs/2608.22847)

**Authors**: Long Zhang, Yuhan Chen, Chaoran Zhang, Wanxia Cao, Kun Huang, Pengzhi Gao, Wei Liu, Jian Luan, Chenliang Li, Lixin Zou  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.22847v1  

#### Abstract
Vision-Language Models (VLMs) based GUI agents stand to benefit significantly from online reinforcement learning (RL). However, their training is bottlenecked by two fundamental issues: current data synthesis methods for GUI Agents rely on specific environments and struggle to generate diverse data,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GSAR: Goal-State-Anchor Rewards for Mobile GUI Agents with Self-Evolving Data Synthesis

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Vision-Language Models (VLMs)** 的 GUI Agent 在使用 **Reinforcement Learning (RL)** 进行训练时面临两大瓶颈：
1. **数据合成不可扩展**：现有方法依赖人工配置环境，难以自动生成多样化、可复现的初始环境与任务对。
2. **奖励信号不准确且不可扩展**：
   - **Rule-based evaluators** 虽然准确，但需手动编写验证逻辑，无法随自动任务生成而扩展；
   - **Model-based evaluators** 可扩展性强，但易受上下文限制、模型幻觉影响，导致奖励噪声大，损害策略收敛。

### 🚀 提出的新方法：GSAR（Goal-State-Anchor Reward）
提出一种全新的 RL 奖励框架 **GSAR**，结合两个核心技术模块：

#### （1）**Self-Evolving Data Synthesis（自演化数据合成）**
- 通过与移动环境交互，**动态演化应用状态**，逐步构建更复杂、多样化的界面。
- 自动执行任务并收集轨迹，形成 `(task, initial environment, trajectory)` 三元组。
- 引入 **task complexification** 机制（继承、合并、重写），从简单任务演进到多步骤、参数化复杂任务。

> ✅ 优势：无需人工干预即可生成大规模、多样化、可复现的训练数据。

#### （2）**State-Anchor Mechanism（状态锚定机制）**
- 利用成功轨迹的最终状态（goal state），自动标注与任务目标相关的 **UI 元素**（如按钮、文本框等）作为“参考锚点”。
- 将这些锚点与截图结合，构成 **goal-state-anchor reference**，用于后续 RL 中的奖励判断。

> ✅ 优势：为 model-based judge 提供“真值答案”，显著提升奖励准确性，同时保持可扩展性。

### 🔍 相比现有方法的优势
| 维度 | Rule-based 方法 | Model-based 方法 | GSAR |
|------|------------------|-------------------|-------|
| 准确性 | 高（接近 100%） | 低（易误判） | **高（>90%）** |
| 可扩展性 | 极差（需手写脚本） | 高 | **高（全自动）** |
| 复现性 | 依赖固定环境 | 不稳定 | **强（保存初始快照）** |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
1. **AndroidControl** (Li et al., 2024)
   - 包含 15,283 条跨 833 个 App 的任务演示。
   - 分为 Low / High 复杂度子集。
2. **GUI-Odyssey** (Lu et al., 2024)
   - 跨 App 导航数据集，共 8,334 个 episode，平均 15.3 步。
3. **AndroidWorld** (Rawles et al., 2024)
   - 动态基准测试环境，用于离线轨迹评估。
4. **自建 Benchmark**
   - 包含 86 个查询及其对应的初始环境快照和标注好的 goal-state-anchor reference。
   - 半数用于训练，半数用于测试。

> ⚠️ 所有评估均排除 `open_app` 类型动作以聚焦核心操作。

### 🧪 实验设置与评估指标

| 类型 | 设置 |
|------|------|
| **SFT 模型** | Qwen2.5-VL-7B + LoRA 微调 |
| **RL 主干模型** | UI-TARS-7B-DPO, GUI-Owl-7B |
| **优化算法** | GRPO |
| **Judge 模型** | Qwen3-VL-8B/32B, GPT-4o, Gemini-2.5-Pro |
| **硬件** | 2 节点 × 8 × H100 GPU；128 并行 Android 模拟器 |

#### 评估指标
| 场景 | 指标 |
|------|------|
| 数据质量（SFT） | Type Match (TM), Exact Match (EM) |
| 离线奖励评估 | Accuracy, F1 Score |
| 在线 RL 性能 | Success Rate (SR)，人工 + GSAR 双重验证 |
| 消融实验 | 移除组件后的 Accuracy/F1 下降情况 |

### 🆚 基线方法对比
#### 数据质量对比
- OS-Genesis-7B
- OS-Atlas-7B
- Aguvis-7B
- Qwen2.5-VL-7B（零样本）

#### 奖励机制对比
- **DigiRL**, **DistRL**, **StepCritic**（代表性的 model-based judge）
- **GS-Only**（仅输入 goal state 截图）
- **GSA-Only**（仅输入锚定截图）
- **Rule-based**（理想上限）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）数据质量评估（Table 1）
在 **AndroidControl-Low** 上：
- **EM 达 90.2%**（相比 Qwen2.5-VL-7B 提升 +5.2%）
- **TM 达 95.4%**

在 **GUI-Odyssey** 上：
- **TM 提升 6.3%**，显示更强泛化能力

> 表明合成数据质量优于现有开源模型。

#### （2）离线奖励评估（Table 2）
| 方法 | Avg Accuracy | Avg F1 |
|------|---------------|--------|
| DigiRL | 68.3% | 62.8% |
| DistRL | 75.0% | 74.5% |
| StepCritic | 77.6% | 81.8% |
| GS-Only | 83.0% | 83.2% |
| GSA-Only | 86.3% | 86.5% |
| **GSAR (Ours)** | **91.5%** | **91.8%** |

> ✅ GSAR 是唯一突破 **90% 准确率** 的 model-based 方法，逼近 rule-based 理论上限。

#### （3）在线 RL 性能（Figure 3）
在自建 benchmark 上：
- **UI-TARS-7B-DPO + GSAR**：
  - 训练集 SR 提升 **23.2%**
  - 全量集 SR 提升 **8.1%**
- **GUI-Owl-7B + GSAR**：
  - 分别提升 **18.6%** 和 **8.2%**

> 显示 GSAR 能有效指导策略学习，带来显著性能增益。

#### （4）消融实验（Table 3）
移除不同组件后在 Qwen3-VL-32B 上的表现：
| 方法 | Acc | F1 |
|------|-----|----|
| w/o reference（无 goal state） | 76.8% | 81.0% |
| w/o anchor（无锚点） | 90.5% | 90.7% |
| w/o history（无动作历史） | 86.4% | 86.9% |
| **GSAR (完整版)** | **92.1%** | **92.4%** |

> 结论：**三者缺一不可**，组合使用效果最佳。

#### （5）其他关键结果
- **自动标注准确率**：整体达 **91.5%**（Table 4），其中 Delete 类任务达 100%
- **推理延迟更低**：GSAR 平均 rollout 时间 **2342.02s** vs Rule-based **2754.92s**（Table 5）
- **奖励曲线稳定性**：GSAR 的 reward 曲线与 rule-based 高度一致（Figure 6b），说明其提供稳定优化信号。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Goal-State Anchoring 显著提升 reward 准确性**  
   提供视觉“真值锚点”使 model-based judge 更可靠，解决了传统方法中 FP/FN 失衡问题（Figure 5）。

2. **Self-Evolving Synthesis 支持可持续数据增长**  
   通过迭代执行与任务复杂化，系统可自主生成越来越复杂的任务与环境，模拟真实用户行为路径。

3. **高质量 reward 直接决定 RL 效果**  
   StepCritic 因高 false positive 导致 reward 过于乐观（Figure 4），策略不稳定；而 GSAR 提供更真实的反馈，促进稳健学习。

4. **GSAR 实现 accuracy 与 scalability 的平衡**  
   在不牺牲可扩展性的前提下，将 model-based judge 的 accuracy 推至接近 rule-based 水平。

### ⚠️ 局限性
1. **单目标状态假设**：每个任务只定义一个 goal state，忽略多解路径（multi-path solutions）。
2. **非视觉完成任务难处理**：如删除操作、后台服务触发等，无法完全通过截图表达。
3. **QA 类任务可能提前奖励**：模型可能在未输出正确答案前就被判定完成。
4. **依赖强 VLM 能力**：轨迹生成与过滤依赖 GPT-4o，弱模型可能导致数据质量下降。

### 🔮 未来工作方向
1. 支持 **multiple valid goal states** 的 reward 设计。
2. 引入 **多模态表示**（如自然语言描述 + 视觉锚点）来刻画非视觉任务完成状态。
3. 对任务类型进行细粒度分类，并设计差异化 reward 策略。
4. 探索更强大的 GUI-capable VLMs，进一步减少人工介入需求。

---

> 💬 **总结一句话**：  
> **GSAR 通过“自演化数据合成 + 目标状态锚定”，首次实现了高精度、可扩展、端到端自动化的 GUI Agent RL 训练闭环，在 accuracy 与 scalability 之间取得了突破性平衡。**

</details>

---

### 13. [Beyond Factual Knowledge: Benchmarking and Learning Step-Level Procedural Rule Reasoning in Large Language Models](https://arxiv.org/abs/2608.22753)

**Authors**: Bohan Yu, Pengfei Cao, Chen Han, Chenxi Zhou, Zhiheng Zhang, Zhiyang Xie, Wenhao Teng, Xiangwen Liao, Jun Zhao, Kang Liu  
**Category**: cs.CL  
**Published**: 2026-08-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.22753v1  

#### Abstract
Large language models (LLMs) excel at text understanding and generation, yet still struggle to reliably understand and apply externally provided procedural rules at scale. To evaluate this capability, we introduce RuleWorld, a large-scale benchmark that reformulates rules as globally reusable abstra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Factual Knowledge: Benchmarking and Learning Step-Level Procedural Rule Reasoning in Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前大型语言模型（LLMs）在文本理解与生成方面表现出色，但在处理外部提供的**程序性规则（procedural rules）**时仍面临挑战。现有研究多关注模型对**事实性知识（factual knowledge）**的记忆与应用，而忽略了对**可复用、抽象化规则**的动态定位与分步推理能力。

具体而言，现有基准（benchmarks）通常将规则作为特定问题的前提直接提供，这掩盖了两个关键能力：
1. **从大规模共享规则库中检索相关规则的能力**；
2. **在多步推理过程中稳定地应用并更新规则的能力**。

因此，该论文旨在解决：如何系统性地评估和提升 LLMs 在大规模注入规则池下的**规则定位（localization）**与**分步应用（step-level application）**能力。

---

### 提出了什么新方法或新思路

论文提出了两大核心贡献：

#### （1）RuleWorld：首个面向大规模程序性规则推理的大规模基准

- **特点**：
  - 包含 **494万条**抽象、非常识、实体无关的程序性规则（以 FOL 和自然语言 NL 形式呈现）。
  - 规则具有**全局一致性**和**可复用性**，支持跨实例共享。
  - 支持三种推理任务类型：
    - **Single-Rule QA**：单规则应用
    - **Parallel Multi-Rule QA**：并行多子问题，独立应用多个规则
    - **Multi-Hop Rule QA**：多跳推理，需状态追踪与规则链式组合
  - 共构建 **337万 QA 实例**，涵盖 11 个子任务，难度随规则数和推理深度递增。

- **创新性**：
  - 首次将规则视为**全局可重用的知识单元**，而非每题专属提示。
  - 引入**组合性设计**：规则结论可成为其他规则前提，形成真实交互。
  - 明确区分不同维度的规则应用能力，实现细粒度评估。

#### （2）DynaRule：端到端的动态规则集成框架

- **核心思想**：
  - 将外部规则编码后注入 LLM 的 **KV Cache** 中，实现内部化的知识存储。
  - 引入特殊 `<search>` token，在推理过程中触发**动态规则重注意（re-attention）与更新**。
  - 通过 **Stacked Step-Level Attention Training** 学习每一步应关注哪些规则。

- **关键技术**：
  - **Confidence Layer 识别**：选择注意力熵最低的一层作为规则决策层。
  - **Step-Level Attention Loss**：监督每一步的规则注意力分布，使其聚焦于当前所需规则。
  - **<search>-driven 更新机制**：在生成 `<search>` 后，模型重新计算注意力，替换过时规则，保持推理稳定性。

---

### 相比现有方法的优势

| 方法 | 缺陷 | DynaRule 的优势 |
|------|------|------------------|
| **Prompting** | 受限于上下文长度；规则噪声大；无法动态更新 | 内部 KV 注入，不占上下文；支持大规模规则 |
| **RAG** | 依赖外部检索器；易受语义错配影响；仅一次检索 | 检索过程可学习、端到端优化；支持多步重检索 |
| **KBLaM / SR-KI** | 虽然也使用 KV 注入，但缺乏对多步推理的支持 | 显式建模**步骤级规则切换**，支持动态更新 |

> ✅ **DynaRule 实现了“检索即推理”（retrieval-as-reasoning）的统一范式**。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

- **RuleWorld** 是本文提出的新基准，其核心特性如下：
  - **规则来源**：基于预定义词汇（实体、属性、动作等）与逻辑模板自动生成。
  - **规则类型**：四大类（Attribute, Action, Environment, State），七种子类。
  - **问答构造方式**：
    - 单规则：直接应用一条规则。
    - 并行多规则：合并多个独立子问题。
    - 多跳规则：按因果路径逐步推导。
  - **训练/测试分离**：使用 100,235 条规则构建训练集（111,200 QA），其余用于测试。

---

### 实验设置和评估指标

#### 模型与编码器
- **主干模型**：Qwen2.5-7B-Instruct
- **文本编码器**：Qwen3-Embedding-8B
- **适配器**：单层线性投影（adapter），用于将规则嵌入映射至模型空间

#### 评估指标
- **QA 性能**：Exact Match Accuracy（精确匹配）
- **规则检索性能**：
  - Recall@1, Recall@10, Recall@100（K 设为黄金规则数量）
- **评估方式**：
  - 所有实验使用 **5 个随机种子**，报告平均值与置信区间。
  - 测试集均匀采样，覆盖所有难度级别。

#### 基线方法对比

| 类别 | 方法 | 描述 |
|------|------|------|
| **Full Context Prompting** | Prompting | 将所有规则拼接进 prompt |
| **Retrieval-Augmented Generation** | RAGdense, RAGbm25, RAGhybrid | 使用 Dense/BM25/Hybrid 检索 top-K 规则 |
| **End-to-End KV Injection** | KBLaM | 将规则注入 KV Cache，无显式训练 |
| | SR-KI | 使用注意力损失监督检索，但未考虑多步推理 |
| | **DynaRule (Ours)** | 本文方法，支持 step-level 动态更新 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）QA 准确率（Exact Match Accuracy）

| 方法 | Avg. Accuracy (FOL/NL) @100 rules | @1000 rules | @10,000 rules |
|------|-------------------------------|------------|--------------|
| Prompting | ~0.45 | ~0.30 | 不可行（内存限制） |
| RAGhybrid | ~0.29 | ~0.29 | ~0.22 |
| KBLaM | ~0.91 | ~0.69 | 0.00 |
| SR-KI | ~0.90 | ~0.63 | ~0.37 |
| **DynaRule (Ours)** | **0.948** | **0.848** | **0.570** |

> 🔥 在 10K 规则下，DynaRule 比最强基线（SR-KI）高出 **19.88 pts**。

#### （2）规则检索性能（Recall@1）

| 方法 | FOL @10K rules | NL @10K rules |
|------|----------------|---------------|
| RAGdense | 0.2227 | 0.2241 |
| SR-KI | 0.2415 | 0.2898 |
| **DynaRule (Ours)** | **0.8613** | **0.9045** |

> ✅ DynaRule 在 10K 规则下达到 **>85% Recall@1**，远超基线（最高仅 ~29%）。

#### （3）消融实验与分析

- **<search> token 必不可少**：移除后性能显著下降，验证其在触发重检索中的作用。
- **Confidence Layer 有效性**：在不同模型（Qwen, Llama）上均能稳定识别出同一层，且该层保留正确规则时准确率峰值最高。
- **Step-Wise Attention Pattern**：可视化显示，模型在每一步确实将注意力集中在当前所需的规则上，形成“阶梯状”注意力图谱。
- **泛化能力**：在未见规则构成的测试集上，DynaRule 仍取得 50.20% 准确率（vs 原始集 57.00%），表明其学到的是**对齐策略**而非记忆规则。

---

## 4. 关键结论和发现

### 主要发现

1. **现有 LLMs 在大规模规则池下表现脆弱**：
   - Prompting 和 RAG 因语义错配或噪声干扰而失效。
   - 即使是先进的 KV 注入方法（如 KBLaM、SR-KI）也无法应对多步推理中的规则切换需求。

2. **DynaRule 实现了稳定可靠的分步规则推理**：
   - 通过 `<search>` token 实现**内部可学习的检索机制**。
   - 在 10K 规则下仍保持 >57% QA 准确率和 >85% Recall@1，大幅领先基线。

3. **规则应用瓶颈在于定位而非执行**：
   - 当只提供黄金规则时，模型准确率可达 74–89%，说明一旦规则被正确定位，应用本身相对容易。
   - 这凸显了**高效、精准的规则检索**是程序性推理的关键。

4. **多跳推理更敏感于中间错误传播**：
   - 多跳任务的召回率随步数增加而下降，早期错误会污染后续推理。

---

### 方法的局限性

1. **规则表示信息丢失**：
   - 当前使用纯 embedding 表示规则，可能丢失复杂结构或语义细节。

2. **扩展性受限于 KV Cache 容量**：
   - 虽优于全量 prompt 注入，但 KV Cache 仍存在长度限制。
   - 当前最大测试为 10K 规则，而完整规则集达百万级。

3. **硬编码 `<search>` 机制**：
   - 何时插入 `<search>` 由训练决定，缺乏完全自主判断能力。

---

### 未来工作方向

1. **改进规则表示形式**：
   - 探索可执行代码、结构化逻辑表达等更丰富的规则编码方式。

2. **提升可扩展性**：
   - 引入分层索引、聚类检索、稀疏激活等技术以支持更大规则库。

3. **联合训练规则编码器**：
   - 当前使用固定编码器，未来可尝试与 LLM 联合优化，增强语义对齐。

4. **探索更复杂的规则触发模式**：
   - 如条件触发、事件驱动、循环结构等高级控制流。

5. **应用于真实场景**：
   - 如政策解读、医疗指南遵循、法律条款推理等需要严格规则执行的领域。

---

> 📌 **总结一句话**：  
> 本论文提出了 **RuleWorld** 和 **DynaRule**，首次实现了对 LLMs 在大规模、可复用程序性规则下的**分步动态推理能力**的系统性评估与建模，推动了从“记忆事实”向“遵循规则”的认知跃迁。

</details>

---

### 14. [FormuEvo: LLM-Guided Evolution for Discovering Solver-Efficient Mixed-Integer Programming Formulations](https://arxiv.org/abs/2608.23353)

**Authors**: Haofeng Yuan, Jianing Peng, Jieyi Bi, Ni Zhang, Shiji Song, Zhiguang Cao  
**Category**: cs.CL  
**Published**: 2026-08-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.23353v1  

#### Abstract
Mixed-integer programming (MIP) lies at the core of operations research and industrial optimization. While large language models (LLMs) have recently shown promise in automated MIP modeling from natural language, they prioritize semantic correctness but overlook formulation strength, severely bottle...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FormuEvo: LLM-Guided Evolution for Discovering Solver-Efficient Mixed-Integer Programming Formulations

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Mixed-Integer Programming (MIP)** 建模依赖专家经验，设计出的公式虽然数学上正确，但可能因结构不佳导致求解效率低下。尽管 **Large Language Models (LLMs)** 已被用于自动化 MIP 建模，但它们通常只关注语义正确性和可执行性，而忽视了**公式强度（formulation strength）**，生成的模型在实际求解时往往效率极低。

此外，现代 MIP 求解器（如 Gurobi）内部机制复杂，传统“最佳实践”有时反而会干扰其预处理、剪枝等过程，造成性能下降。因此，亟需一种能自动发现**对求解器高效（solver-efficient）** 的 MIP 公式的框架。

### 提出了什么新方法或新思路
本文提出 **FormuEvo** —— 一个基于 LLM 引导的进化框架，用于自动化发现高效的 MIP 公式。其核心思想是将 MIP 公式设计视为一个在符号空间中的**进化优化问题**，通过迭代生成、评估和选择更优候选公式来逐步演化。

#### 主要创新点包括：
- **LLM-Guided Evolutionary Search**  
  将 MIP 公式表示为可执行的建模程序（如 Python + `gurobipy`），利用多个专用 LLM 模块作为进化算子：
  - **Generator LLM**：生成新公式
  - **Diagnostic LLM**：分析求解器反馈并提供改进方向
  - **Repair LLM**：修复语法或逻辑错误
  - **Reflector LLM**：从历史经验中抽象策略
  - **Distiller LLM**：提炼通用知识以支持迁移

- **Solver-Informed Diagnosis**  
  利用求解器运行时的细粒度统计信息（如 `root gap`, `node count`, `presolve info`）作为“可解释梯度”，指导 LLM 进行有针对性的修改，避免盲目探索。

- **Structured Memory**  
  构建结构化记忆库，存储 `(Condition → Strategy → Effect)` 三元组，实现经验复用，避免重复试错，并支持零样本迁移到新问题。

### 相比现有方法的优势
| 方法类型 | 局限性 | FormuEvo 的优势 |
|--------|------|----------------|
| **Human Experts** | 耗时、依赖经验、难以适应新型求解器 | 自动化、数据驱动、持续进化 |
| **Fine-tuned LLMs (e.g., ORLM, StepORLM)** | 仅保证正确性，输出多为教科书式弱公式 | 显著提升求解效率，超越专家设计 |
| **局部增强方法 (e.g., EvoCut)** | 只能在固定模板基础上加 cutting planes | 可进行全局结构重构（如变量重定义、线性化方式变更） |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖了经典的 **MILP** 和 **MINLP** 问题，以及两个少有人工建模先验的新挑战任务：

| 问题类别 | 具体问题 |
|---------|--------|
| 经典 MILP/MINLP |  
- **TSP**: Traveling Salesman Problem  
- **JSSP**: Job-Shop Scheduling Problem  
- **BPP**: Bin Packing Problem  
- **CFLP**: Capacitated Facility Location Problem  
- **QAP**: Quadratic Assignment Problem |
| 新型挑战问题 |  
- **NNV**: Neural Network Verification  
- **IMO**: IMO 2025 Problem 6（网格覆盖问题） |

每个问题按规模分为 Easy、Medium、Hard 三个难度等级，其中 Easy 用于训练/验证，Medium/Hard 用于测试最终性能。

### 实验设置和评估指标
- **下游求解器**：Gurobi 10.0（单线程，默认参数）
- **硬件平台**：AMD EPYC 9654 服务器
- **种群大小**：N = 8
- **进化代数**：T = 5
- **每代生成后代数**：8（交叉 + 突变）
- **评估指标**：
  - **Time**：在 100 个测试实例上的 **shifted geometric mean (SGM)** 运行时间（秒）
  - **Wins**：在多少实例上取得最快求解时间
  - **Solved**：在 600 秒时限内成功求解的实例数量

### 基线方法对比
| 类别 | 基线方法 |
|-----|--------|
| **专家设计公式** |  
- TSP: MTZ, SCF, MCF-RLT  
- JSSP: Disj., Enh. Disj.  
- BPP: Kant., AF, VPSolver  
- QAP: KB Quad., McC. Lin., IPQAPR 等 |
| **LLM 自动生成方法** |  
- ORLM (2025)  
- StepORLM (2026) |
| **其他 LLM 进化方法** |  
- EvoCut (2025)：基于 LLM 的 cutting plane 生成 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### 在经典问题上的表现（Hard 实例）

| 问题 | 最佳基线 Time | FormuEvo Time | 加速比 | Solved |
|------|-------------|--------------|--------|--------|
| **TSP** | 8.3315 s | **3.9469 s** | **↑ 52.6%** | 100/100 |
| **JSSP** | 17.6653 s | **15.4237 s** | ↑ 12.7% | 98/100 |
| **BPP** | 29.0496 s | **0.8384 s** | **↑ 41.9%** | 100/100 |
| **CFLP** | 59.4309 s | **44.3447 s** | ↑ 25.4% | 100/100 |
| **QAP** | 538.5478 s | **34.5034 s** | ↑ 8.8% | 100/100 |

> ⚠️ 注：MCF-RLT（理论最强松弛）在 TSP Hard 上完全失败（0/100 solved）

#### 在新型问题上的表现

| 问题 | 方法 | Time | Wins | Solved |
|------|------|------|------|--------|
| **NNV** | ORLM/StepORLM | 失败 | – | – |
| | EvoCut | 67.61 s | 13/100 | 73/100 |
| | **FormuEvo** | **21.41 s** | **78/100** | **86/100** |
| **IMO** | ORLM/StepORLM | 失败 | – | – |
| | EvoCut | 99.44 s | 0/4 | 3/4 |
| | **FormuEvo** | **17.93 s** | **4/4** | **4/4** |

✅ **最大加速达 5.5×**

### 与基线方法的对比结果
- **全面超越专家设计公式**：即使是最先进的 MCF-RLT 或 VPSolver，在大规模实例上也被 FormuEvo 超越。
- **显著优于 LLM 生成方法**：ORLM 和 StepORLM 输出多为标准教科书公式，无法应对复杂问题（如 NNV、IMO 完全失败）。
- **优于局部增强方法 EvoCut**：EvoCut 仅能加强已有公式，无法改变结构；而 FormuEvo 可进行根本性重构（如 BPP 中采用压缩流网络）。

### 消融实验结果（Table 3）

| 方法 | TSP (Hard) Time | JSSP (Hard) Time |
|------|------------------|------------------|
| Best Baseline | 8.3315 s | 17.6653 s |
| **FormuEvo (完整)** | **3.9469 s** | **15.4237 s** |
| w/o Memory | 4.6649 s | 16.3223 s |
| w/o Diagnosis | 6.5207 s | 18.0159 s |

📌 **结论**：
- **Solver-Informed Diagnosis** 是最关键组件，移除后性能大幅下降（↓20–30%）
- **Structured Memory** 有效提升搜索效率，尤其在长期演化中作用明显

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **理论最优 ≠ 实际高效**  
   如 MCF-RLT 在 TSP 上具有最紧松弛界，但由于变量过多，导致 presolve 和 branching 开销过大，反而不如紧凑且经过针对性强化的公式。

2. ✅ **FormuEvo 能发现非显而易见的强公式**  
   例如在 BPP 中，FormuEvo 发现了一种结合 GCD 缩放、状态压缩和损失弧的新型流网络结构，远超传统 Kantorovich 或 AF 公式。

3. ✅ **知识可迁移性强**  
   通过 **Distiller LLM** 提炼的通用建模策略，可用于引导小型 LLM（如 GPT-5.4-nano）快速收敛到高性能公式，缩小与大模型差距（见 Figure 4）。

4. ✅ **方法鲁棒且通用**  
   在不同求解器（Gurobi, COPT, SCIP）上均表现出色，说明其优化目标与具体求解器耦合紧密，具备自适应能力。

### 方法的局限性
- **局限于静态 MIP 公式**  
  当前框架仅适用于一次性构建即可直接求解的 MIP 模型，不支持动态算法（如 column generation、Benders decomposition），这些方法需要在求解过程中动态添加变量或约束。

- **依赖高质量 LLM 推理能力**  
  尽管框架本身对 backbone LLM 不敏感（见 Table 4），但在极端复杂问题上仍受限于 LLM 的逻辑推理深度。

- **计算成本较高**  
  需要多次调用求解器进行评估，整个进化流程耗时数小时，不适合实时场景。

### 未来工作方向
- 扩展至 **动态优化算法联合设计**：同时演化公式与分解策略（如 Benders master-subproblem 结构）。
- 探索 **zero-shot 跨域迁移**：将在一类问题上学到的知识直接应用于完全不同领域的问题。
- 构建 **开放 MIP 公式进化平台**：允许社区共享 `memory library` 和 `distilled knowledge`，形成协作式自动化建模生态。

--- 

> 🔗 **代码开源地址**：[https://github.com/Xyz-yuanhf/formuevo](https://github.com/Xyz-yuanhf/formuevo)

</details>

---

### 15. [TracingFlow: A Simulation-Free Trajectory Inference Framework Based on Second-Order Dynamics](https://arxiv.org/abs/2608.21070)

**Authors**: Yuhao Sun, Zekun Wu, Zixun Huang, Peijie Zhou  
**Category**: cs.LG  
**Published**: 2026-08-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.21070v1  

#### Abstract
Inferring continuous system evolution from sparse temporal snapshots is a key challenge in generative modeling and single-cell omics. While Optimal Transport (OT) is popular, existing frameworks are largely restricted to first-order dynamics, assuming memoryless velocity fields. This limits expressi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TracingFlow: A Simulation-Free Trajectory Inference Framework Based on Second-Order Dynamics**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在生成建模和单细胞组学（single-cell omics）中，从稀疏的时间快照（temporal snapshots）推断连续系统的演化轨迹是一个核心挑战。现有的基于 **Optimal Transport (OT)** 的轨迹推断（Trajectory Inference, TI）框架大多依赖于**一阶动力学**（first-order dynamics），即假设系统演化仅由速度场 $ \dot{x} = v(x,t) $ 决定。

这种一阶模型存在以下局限性：
- **表达能力受限**：无法捕捉生物过程中固有的“调控动量”（regulatory momentum）和延迟响应（time-delayed responses），例如细胞分化过程中的非线性高曲率轨迹。
- **轨迹平滑化**：倾向于产生过度平滑的路径，难以模拟真实生物系统中的复杂动态。
- **单值速度约束**：标准 Flow Matching 要求速度场是位置 $ x $ 的单值函数，导致无法表示在空间中交叉但速度不同的轨迹。

此外，当考虑染色质可及性、蛋白质组等多模态数据时，生物动力学可能遵循更高阶的动力学规律。

### **提出的新方法与新思路**
本文提出了 **TracingFlow**，一种**无需仿真的（simulation-free）轨迹推断框架**，其核心创新在于将动力学建模从一阶推广到**二阶动力学**（second-order dynamics）：

- **建模加速度而非速度**：引入**加速度场** $ a(x,v,t) $ 来描述系统演化 $ \ddot{x} = a(x,v,t) $，从而自然地建模力和惯性效应。
- **提出 DOAT 问题**：定义了 **Dynamical Optimal Acceleration Transport (DOAT)** 问题，目标是最小化加速度成本 $ \int \|a(x,v,t)\|^2 dt $，类比于经典力学中的最小作用量原理。
- **仿真自由求解**：通过神经网络直接回归加速度场和初始速度，避免了传统 Neural ODE 所需的数值积分，显著降低计算成本。
- **处理 VM-DOAT 问题**：针对实际单细胞数据中**速度不可观测**的问题，提出 **Velocity-Missed DOAT (VM-DOAT)** 并设计迭代策略将其转化为标准 DOAT。
- **整合生物学先验**：通过在 OT 成本矩阵中加入条形码不匹配惩罚项，将谱系追踪（lineage tracing）先验无缝嵌入连续动力学学习中。

### **相比现有方法的优势**
| 特性 | TracingFlow | 传统一阶方法（如 OT-CFM） | 3MSBM |
|------|-------------|--------------------------|--------|
| 动力学阶数 | 二阶（加速度） | 一阶（速度） | 二阶 |
| 是否需要 ODE 仿真 | ❌ 否（simulation-free） | ✅ 是（或近似） | ✅ 是（迭代训练） |
| 初始速度估计 | ✅ 作为最优控制一部分 | ❌ 通常忽略或启发式设定 | ❌ 启发式方法 |
| 支持轨迹交叉 | ✅ 在位置空间中支持 | ❌ 不支持（单值速度约束） | ✅ 支持 |
| 整合生物学先验 | ✅ 显式、原则性方式 | ⚠️ 有限或离散方式 | ⚠️ 未明确支持 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 类型 | 维度 | 时间点数 | 特点 |
|-------|------|------|---------|------|
| **2D Simulation** | 合成数据 | 2D | 5 | 高曲率、非线性轨迹，用于验证基本能力 |
| **Cite 5D / Cite 100D** | 单细胞 RNA-seq (Cite-seq) | 5D / 100D | 4 | 真实转录组数据，降维后测试高维性能 |
| **EB 5D** | 胚状体发育数据 | 5D | 5 | 用于插值/外推实验（Hold-One-Out） |
| **3D Simulation Lineage** | 合成谱系数据 | 3D | 3 | 包含三种谱系条形码，用于验证先验整合 |
| **Hematopoiesis** | 血细胞分化数据 | 50D | 3 | 真实谱系追踪数据，部分细胞有条形码 |
| **Mexico Gulf** | 流体动力学模拟 | 2D | 9 | 长时间序列，测试长期动态建模能力 |

### **实验设置与评估指标**
#### **评估指标**
- **分布重建误差**：
  - **Wasserstein-1 Distance (W₁)**
  - **Wasserstein-2 Distance (W₂)**
- **谱系一致性误差**（含 lineage prior 时）：
  - **lineage-weighted W₁ 和 W₂**：对每个谱系分别计算 W 距离后加权平均，衡量是否保留谱系结构。
- **插值/外推性能**：在 Hold-One-Out 实验中评估对未见时间点的预测能力。

#### **基线方法对比**
- **OT-CFM**, **SF2M**: 一阶 Flow Matching 方法
- **3MSBM**: 二阶 Momentum Schrödinger Bridge，需迭代训练
- **MMFM**, **HRF**, **CAF**: 其他二阶或变体 Flow Matching 方法

所有方法在相同超参数下进行公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **表 1：多数据集分布重建性能（平均 W₁/W₂）**
| Method | 2D Simulation (W₁↓) | Cite 5D (W₁↓) | Cite 100D (W₁↓) |
|--------|---------------------|---------------|----------------|
| OT-CFM | 0.5270 ± 0.0612 | 0.8298 ± 0.0162 | 10.5498 ± 0.0291 |
| 3MSBM | 1.1855 ± 0.2636 | 3.6369 ± 0.3508 | 15.7708 ± 0.1473 |
| HRF | 1.5390 ± 0.1130 | 1.4016 ± 0.0437 | 10.1532 ± 0.0643 |
| **TF (Ours)** | **0.3748 ± 0.0505** | **0.5861 ± 0.0688** | **8.0700 ± 0.1537** |

✅ **结论**：TracingFlow 在所有数据集上均取得最低的 W₁/W₂ 距离，表明其分布重建精度最高。

#### **表 2：EB 5D 插值/外推性能（Hold-One-Out）**
| Method | W₁ ↓ | W₂ ↓ |
|--------|------|------|
| OT-CFM | 4.5602 ± 0.1041 | 4.9905 ± 0.1262 |
| 3MSBM | 4.4974 ± 0.0950 | 4.7859 ± 0.1857 |
| **TF (Ours)** | **4.4974 ± 0.0950** | **4.7859 ± 0.1857** |

✅ **结论**：TracingFlow 在插值任务中表现最佳，说明其能更准确捕捉非线性演化路径。

#### **表 3：谱系数据上的 lineage-weighted W₁/W₂**
| Method | 3D Sim-Lineage (W₁↓) | Hematopoiesis (W₁↓) |
|--------|----------------------|--------------------|
| OT-CFM | 1.5255 ± 0.0513 | 18.7170 ± 0.1124 |
| 3MSBM | 2.3126 ± 0.4756 | 18.3313 ± 0.9834 |
| **TF w/o Bio-Prior** | 1.5292 ± 0.1714 | 16.6166 ± 0.1229 |
| **TF (Ours)** | **0.4538 ± 0.2039** | **14.0470 ± 0.1530** |

✅ **结论**：引入生物学先验后，TracingFlow 显著优于其他方法，证明其能有效保持谱系结构。

### **可视化证据**
- **图 2**：在 2D 合成数据上，TracingFlow 学习到了**交叉轨迹**，而 OT-CFM 因单值速度约束无法实现。
- **图 3 & 图 4**：在谱系数据上，TracingFlow 推断的轨迹严格遵循条形码分支，而基线方法出现错误跨谱系迁移。

### **消融实验与敏感性分析**
- **Minibatch-OT 敏感性**（Table 13）：在 Cite-5D 上，即使使用 minibatch（~10³），TracingFlow 仍能保持高精度，证明其可扩展性。
- **训练时间**（Table 12）：TracingFlow 训练时间略低于 3MSBM，远短于传统 ODE 方法，验证其高效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **二阶动力学显著提升表达能力**：通过建模加速度，TracingFlow 能捕捉高曲率、非线性、交叉的生物轨迹，克服了一阶模型的平滑化缺陷。
2. **仿真自由框架高效且精确**：无需 ODE 积分，通过回归加速度场即可精确求解 DOAT 问题，兼具效率与准确性。
3. **速度缺失问题可通过迭代优化解决**：提出的 VM-DOAT 转换策略能有效从位置数据中恢复隐含的速度分布。
4. **生物学先验可被原则性整合**：通过修改 OT 成本矩阵，谱系信息被自然地融入动力学学习，提升了轨迹的生物学合理性。

### **方法的局限性**
1. **SOAT 预计算开销大**：尽管采用 minibatch-OT 缓解，大规模数据上的最优传输计算仍是瓶颈。
2. **速度分布被期望近似**：当前方法用确定性初始速度近似完整条件分布 $ w(v|x) $，可能丢失不确定性信息。
3. **物理意义尚不明确**：目标函数 $ \int \|a\|^2 dt $ 缺乏明确的物理对应（如经典作用量不含二阶导），虽可通过 Hamiltonian 解释，但仍属数学构造。

### **未来工作方向**
1. **扩展至更高阶动力学**：探索三阶及以上动力学以建模更复杂的调控机制。
2. **联合建模多模态数据**：将 scRNA-seq、ATAC-seq、蛋白表达等多组学数据统一纳入二阶框架。
3. **在线学习与增量更新**：开发适用于流式单细胞数据的在线版本。
4. **理论解释增强**：进一步探究 $ \int \|a\|^2 dt $ 在生物系统中的潜在物理解释。

---

> **总结**：TracingFlow 是首个真正实现**仿真自由**的**二阶 Flow Matching** 框架，通过引入加速度场和 DOAT 问题，在分布重建、轨迹保真度和生物学先验整合方面全面超越现有方法，为单细胞轨迹推断提供了新的范式。

</details>

---

### 16. [Thermo-FL: Thermal-Aware Robust Federated Fine-Tuning of Large Language Models for Edge AI](https://arxiv.org/abs/2608.21172)

**Authors**: Shiva Shrestha, Kazi Shaharair Sharif, Zongxing Xie, Jiajing Huang, Anhao Xiang, Honghui Xu  
**Category**: cs.LG  
**Published**: 2026-08-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.21172v1  

#### Abstract
Federated fine-tuning enables large language models to adapt on edge devices without centralizing private data, but practical deployments must address hardware instability and adversarial update corruption together. Thermally constrained clients may throttle, slow local training, or delay synchronou...

---

### 17. [MCP-Universe RL: A Framework for Training MCP Tool-Use Agents via Reinforcement Learning](https://arxiv.org/abs/2608.22167)

**Authors**: Ziyang Luo, Yan Yang, Xiangru Jian, Ziji Shi, Xiaoqiang Lin, Jun Hao Liew, Silvio Savarese, Junnan Li  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.22167v1  

#### Abstract
Reinforcement learning (RL) has become an effective way to improve the tool-use ability of large language models (LLMs), but most existing RL frameworks stop at the policy update. For every new domain, the user is left with two hard systems problems: standing up an isolated environment for each of h...

---

### 18. [TailSieve: Partial-Rollout-Guided Tail Routing for LLM Rollouts](https://arxiv.org/abs/2608.22788)

**Authors**: Tianqi Xu, Lu Lv, Haoyang Huang, Wenjie Huang, Zhanming Shen, Yuhao Shen, Baolin Zhang, Xinyi Hu, Shuang Ge, Jun Dai, Tianyu Liu, Suorong Yang, Zhikai Li, Ye Bai, Jun Zhang, Lei Chen, Yue Li, Mingchen Wan  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.22788v1  

#### Abstract
Large-scale rollouts have become a core component of modern LLM systems, spanning reinforcement learning (RL) post-training, on-policy distillation (OPD), and sampling-heavy evaluation pipelines. Unlike online serving, which is typically optimized for request-level latency and throughput, a small nu...

---

### 19. [Context-Aware Cluster Decoding: Semantic Anchor-Driven Coherence in dMLLMs](https://arxiv.org/abs/2608.22367)

**Authors**: Yikai Zhao, Qiyan Zhao, Jiaquan Zhang, Xiaofeng Zhang, Xiaosong Yuan, Pengzhou Cheng  
**Category**: cs.CL  
**Published**: 2026-08-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.22367v1  

#### Abstract
Diffusion multimodal large language models (dMLLMs) frequently produce long-form outputs marred by semantic drift and repetition, with quality generally degrading as output length increases. We identify two structural deficiencies in existing decoding methods as primary drivers of these failures: co...

---

### 20. [WnW: Waxing-and-Waning KV Cache for Long-Form Speech LLMs](https://arxiv.org/abs/2608.22704)

**Authors**: Yiming Yao, Chenyang Lyu, Xuanfan Ni, Longyue Wang, Weihua Luo, Yazheng Yang, Jinsong Su  
**Category**: cs.CL  
**Published**: 2026-08-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.22704v1  

#### Abstract
Long-form audio inputs make the KV cache the dominant memory cost of speech LLMs. Prefill-only KV compression methods permanently discard audio KV positions once evicted, with no pathway to recover them during decoding. We show this is fragile on long-form audio: prefill attention concentrates near ...

---

### 21. [Bern2Edge: A Neurosymbolic Compiler for Edge Deployment via Bernstein Polynomial Networks](https://arxiv.org/abs/2608.20497)

**Authors**: Malak Gamal El-Din, Yifan Zhang, Yasser Shoukry, Sitao Huang, Salma Elmalaki  
**Category**: cs.LG  
**Published**: 2026-08-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.20497v1  

#### Abstract
Deploying high-accuracy neural networks on resource-constrained edge devices remains challenging, as existing approaches treat training, compression, and hardware synthesis as separate stages, leaving a gap between software-trained models and efficient end-to-end deployment with limited support for ...

---

### 22. [Tydra: An Efficient Hybrid Model for Tabular Data](https://arxiv.org/abs/2608.21199)

**Authors**: Mieszko Komisarczyk, Saurabh Mathur, Maurice Kraus, Sriraam Natarajan, Kristian Kersting  
**Category**: cs.LG  
**Published**: 2026-08-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.21199v1  

#### Abstract
Transformer-based tabular foundation models such as TabPFN achieve strong predictive performance but incur quadratic computational cost with context length. On the other hand, subquadratic SSM-based alternatives such as Hydra trade away accuracy for efficiency. To balance both, we introduce Tydra, a...

---

### 23. [Let Credit Follow Computation: Architecture-Aware Credit Transport for Large Language Model Reinforcement Learning](https://arxiv.org/abs/2608.21501)

**Authors**: Qifan Shi, Zhaolu Kang, Chenghua Zhu  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.21501v1  

#### Abstract
Credit assignment in large-language-model reinforcement learning (LLM RL) can be separated into three objects: evidence about success, a transport operator that converts this evidence into token-level advantages, and an update geometry that turns advantages into policy changes. Recent work has great...

---

### 24. [Beyond Success and Failure: Length-Aware Contrastive Learning for GUI Agents](https://arxiv.org/abs/2608.21830)

**Authors**: Chengyang Gu, Le Zhang, Jingbo Zhou, Yize Chen, Yu Shi, Siqi Bao, Zheng-Fan Wu, Hua Wu, Hui Xiong  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.21830v1  

#### Abstract
Graphical User Interface (GUI) agents powered by Multimodal Large Language Models (MLLMs) have shown strong potential for automating tasks across diverse digital environments, where reinforcement learning (RL) has become a dominant training paradigm. However, widely used methods such as Group Relati...

---

### 25. [Training Needs Trustworthy Worlds: Verified Synthetic Web Environments for Agent Learning](https://arxiv.org/abs/2608.21898)

**Authors**: Chenghao Zhang, Canran Xiao, SaiSai Hu, Dan Roth  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.21898v1  

#### Abstract
Web agents promise to automate complex digital workflows, but their training remains limited by synthetic environments that look plausible while hiding broken links, inconsistent states, or infeasible tasks. We address the gap between scalable environment generation and trustworthy agent learning by...

---

### 26. [Think with Structured Grounding: Perceptual Reinforcement Learning for Chart and Visual-Tabular Understanding](https://arxiv.org/abs/2608.22429)

**Authors**: Changjiang Jiang, Qiannian Zhao, Lei Xin, Jinxiang Xie, Preslav Nakov, Zhuohan Xie  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.22429v1  

#### Abstract
Multimodal Large Language Models (MLLMs) capable of thinking with images often rely on external tools for fine-grained perception. However, this reliance introduces significant inference latency and fails to effectively resolve the spatial-structural gap-a fundamental challenge in text-dense and str...

---

### 27. [GTA-RAG: Graph-Trajectory-Augmented Reinforcement Learning for Multi-Turn Retrieval-Augmented Reasoning](https://arxiv.org/abs/2608.22479)

**Authors**: Jun Chen, Yongchao Liu, Pengyu Qiu, Jiajun Zheng, Juelu Zhang, Yujie Zeng, Qin Zhang, Ziyue Qiao, Xiao Luo  
**Category**: cs.CL  
**Published**: 2026-08-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.22479v1  

#### Abstract
Retrieval-augmented generation (RAG) enables LLMs to access external knowledge for answering knowledge-intensive questions. For complex multi-hop questions, multi-turn retrieval-augmented reasoning extends RAG into an iterative process that repeatedly searches for and integrates evidence across docu...

---

### 28. [Accelerating Diffusion Language Models via Structured Suffix Modeling](https://arxiv.org/abs/2608.23167)

**Authors**: Zifeng Cheng, Keda Li, Zhiwei Jiang, Cong Wang, Fei Shen, Qing Gu  
**Category**: cs.CL  
**Published**: 2026-08-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.23167v1  

#### Abstract
Diffusion Language Models (DLMs) exhibit strong parallel decoding capabilities by denoising multiple tokens in a single generation step. However, this parallelism comes with substantial computational overhead, as each step requires interactions with all suffix tokens. Existing methods typically redu...

---

### 29. [Scaling Muon for Diffusion Transformers](https://arxiv.org/abs/2608.20818)

**Authors**: Chenghao Li, Xiao Han, Xinxin Huang, Wei Liu, Boyang Li, Bing Xiao, Heran Zhang, Juanma Perez Rua, Ke Xu, Kangning Liu, Linjun Kuang, Na Li, Tan Wang, Tian Xie, Wei Peng, Yang Pei, Yifan Xu, Yuanhao Zhai, Yuwei Lin, Zhe Wang, Zihao He, Daniel Li, Junbiao Tang, Ziyang Jiang, Dake Chen  
**Category**: cs.LG  
**Published**: 2026-08-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.20818v1  

#### Abstract
The matrix-aware optimizer Muon improves large model training by balancing updates across singular directions, yet its scaling behavior and end-to-end efficiency on large Diffusion Transformers (DiTs) remain unclear. We first establish Muon's scaling behavior on DiTs from 1.3B to 15B parameters, sho...

---

### 30. [ExecRubrics: Executable Tool-Augmented Rubrics for Verifiable and Efficient Long-Form Evaluation](https://arxiv.org/abs/2608.22559)

**Authors**: Kaustubh D. Dhole, Charles L. A. Clarke, Eugene Y. Agichtein  
**Category**: cs.AI  
**Published**: 2026-08-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.22559v1  

#### Abstract
Rubrics aim to make language-model evaluation transparent by decomposing response quality into interpretable criteria. However, natural-language rubrics are often ambiguous, require black-box LLM judges, and typically assume criteria aggregate independently through linear weighted sums, limiting the...

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

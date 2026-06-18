# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-18 10:03:59 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Towards Scalable Customization and Deployment of Multi-Agent Systems for Enterprise Applications](https://arxiv.org/abs/2606.18502)

**Authors**: Paresh Dashore, Shreyas Kulkarni, Uttam Gurram, Nadia Bathaee, Kartik Balasubramaniam, Genta Indra Winata, Sambit Sahu, Shi-Xiong Zhang  
**Category**: cs.CL  
**Published**: 2026-06-18  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2606.18502v1  

#### Abstract
Large language model (LLM)-based multi-agent systems demonstrate strong performance on complex reasoning and task execution, enabling broad enterprise applications. However, production deployment remains challenging due to domain-specific customization requirements and high latency and inference cos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Towards Scalable Customization and Deployment of Multi-Agent Systems for Enterprise Applications》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Large Language Model (LLM)** 的 **multi-agent system** 在复杂任务推理与执行中表现出色，但在企业级生产部署中面临以下挑战：
- **高延迟**：多轮 LLM 调用累积导致端到端延迟过高，难以满足严格的 **SLA (Service-Level Agreement)**。
- **高推理成本**：频繁调用大模型显著增加计算资源消耗和部署成本。
- **领域定制需求**：通用模型在特定业务场景下表现不足，需进行高效、高质量的领域适配。

### 提出了什么新方法或新思路
作者提出一个**统一的框架**，涵盖两个核心阶段：

#### 阶段一：Agentic Model Customization（代理模型定制）
通过三阶段训练流程将大型教师模型（Teacher）的能力蒸馏到小型学生模型中：
1. **Context-aware Continual Pretraining (CPT)**：利用带上下文前缀的持续预训练，缓解灾难性遗忘，提升领域适应能力。
2. **Supervised Fine-Tuning (SFT)**：使用 LoRA 进行指令微调，保留零样本泛化能力。
3. **Direct Preference Optimization (DPO)**：引入偏好对齐，纠正教师模型错误，提升鲁棒性和边界情况处理能力。

#### 阶段二：Inference Optimization（推理优化）
结合两种前沿技术实现高效推理：
- **EAGLE Speculative Decoding**：使用轻量级草稿模型预测 token，目标模型并行验证，减少自回归步数。
- **FP8 Quantization (W8A8)**：采用 FP8 权重与激活量化，降低内存占用，提升吞吐量，并支持更大的 batch size 或上下文长度。

最终模型命名为 **TEAGLE+FP8**，实现了从 `T_T` → `T_16` → `TEAGLE+FP8` 的端到端优化路径。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 实现 **4.48× 端到端吞吐提升**，显著优于仅使用 SFT 或单一优化技术的方法。 |
| **质量保持** | 在大幅提升速度的同时，任务性能持平甚至超越 70B 教师模型（如 Planner 和 Understander Agent）。 |
| **可扩展性** | 支持长上下文（>1000 tokens）、高并发场景，适用于真实企业负载。 |
| **系统集成性** | 方法模块化，可叠加于现有系统之上，兼容 TensorRT-LLM 等主流推理引擎。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **合成数据（Proprietary Synthetic Data）**：
  - 来自 **Agent Simulator (U)** 模拟的 7,172 场对话，生成约 495,772 条训练轨迹。
  - 包含真实汽车零售场景中的意图理解、工具调用、安全校验等行为。
- **公开数据（Public Data）**：
  - 开源对话数据（77k samples），用于增强泛化能力。
  - Nallapati et al. (2016) 的摘要数据集（1.4k samples）用于 FP8 calibration。
- **混合 Calibration 数据集**：
  - 为 FP8 量化提供校准数据，包含 5.8k 合成 + 1.4k 公开数据，共 7.2k 样本。

### 实验设置和评估指标
- **硬件平台**：AWS EC2 P5 实例（8× NVIDIA H100 80GB GPU）。
- **推理框架**：NVIDIA TensorRT-LLM v1.9。
- **评估指标**：
  - **P90 Latency (s)**：第90百分位延迟。
  - **QPS (Queries Per Second)**：每秒查询数。
  - **MGL (Mean Generated Length)**：平均生成长度。
  - **E2E Stress Test Pass Rate**：模拟复杂业务切换下的功能通过率。
  - **Agent-level Accuracy**：各 agent 在理解、规划、执行等子任务上的准确率。

### 基线方法对比
| 配置 | 描述 |
|------|------|
| `Llama3 70B (TT)` | 原始教师模型，作为性能上限基准。 |
| `Baseline (T_16)` | 未经优化的学生模型（BF16）。 |
| `Baseline (T_8)` | 更小规模基线模型。 |
| `EAGLE-only` | 仅启用 EAGLE 推测解码。 |
| `FP8-only` | 仅启用 FP8 量化。 |
| `TEAGLE+FP8` | 完整优化方案（EAGLE + FP8）。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| Configuration | Latency (s) | QPS | Speedup |
|-------------|------------|-----|--------|
| Llama3 70B (TT) | 3.92 | 1.46 | 1.00× |
| Baseline (T_16) | 1.69 | 3.40 | 2.33× |
| TEAGLE+FP8 | **0.92** | **6.54** | **4.48×** |

> ✅ **结论**：相比原始 70B 模型，最终方案实现 **4.48× 吞吐加速**，且延迟降至 **0.92 秒（P90）**，满足 sub-second SLA。

### 与基线方法的对比结果
- `T_16` 相比 `TT` 已有 2.33× 加速，说明蒸馏本身有效。
- `TEAGLE+FP8` 相比 `T_16` 再提速 **1.92×**，证明推理优化可叠加增益。
- 所有优化后，**QPS 达到 6.54**，远超原始系统的 1.46。

### 消融实验结果

#### （1）EAGLE 不同训练数据的影响（Table 2）
| Drafter Training Data | Speedup | MGL | Latency |
|------------------------|--------|-----|--------|
| External Only (E) | 1.19× | 3.66 | 1.50s |
| Synthetic Only (S) | 1.37× | 3.98 | 1.29s |
| Combined (C) | **1.46×** | **4.29** | **1.19s** |

> 🔍 **发现**：**合成数据更有效**，因其贴近真实业务分布；**混合数据效果最佳**，兼顾泛化与领域对齐。

#### （2）解码策略影响（Greedy vs Tree）
| Strategy | Speedup (Combined) | Latency | Notes |
|---------|--------------------|--------|-------|
| Tree Decoding | 1.46× | 1.19s | 更高接受长度，但计算开销大 |
| **Greedy Decoding** | **1.78×** | **0.96s** | 更低延迟，适合高并发 |

> ⚡ **关键洞察**：尽管 Greedy 降低 MGL，但由于其更低的 drafting 成本，在高并发下反而获得更高吞吐。

#### （3）FP8 Calibration 数据选择
- **短序列（<128 tokens）**：公共数据表现更好（覆盖广）。
- **长序列（≥256 tokens）**：**混合数据剪裁率（clip rate）最低**，在 2048 tokens 时比纯公共数据低 **6.1×**。
- 生产中 prompt 常超过 1000 tokens，因此 **混合 calibration 是必须的**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **高质量合成数据是蒸馏成功的关键瓶颈**：Agent Simulator 的 fidelity 决定了学生模型上限。
2. **LoRA 比全参数微调更适合动态系统**：能保留 zero-shot prompt adaptability，避免过拟合固定 prompt 结构。
3. **DPO 对生产级可靠性至关重要**：SFT 只能教会“格式”，而 DPO 才能修复逻辑错误和边界 case。
4. **优化应分层堆叠（Layered Optimization）**：
   - Layer 0: Base BF16 model
   - Layer 1: 系统级调用减少（Conditional Agent Invocation, Prompt Cache）
   - Layer 2: EAGLE 推测解码
   - Layer 3: FP8 量化
   - Layer 4: 端到端综合收益达 4.48×
5. **Greedy speculation 在高并发下优于 Tree**：尽管接受长度短，但整体 throughput 更高。

### 方法的局限性
- **依赖高质量教师模型和 Judge 模型**：若教师存在偏见或盲区，会传递至学生模型。
- **需要大量前期计算资源**：生成数十万条合成轨迹耗资巨大。
- **EAGLE drafter 需随系统更新重训练**：prompt 或业务逻辑变更会导致分布漂移，影响 token 接受率。
- **硬件依赖性强**：FP8 需要 NVIDIA Hopper 架构（如 H100）原生支持，否则 fallback 至 FP16 会丧失优势。

### 未来工作方向
- 构建更鲁棒的自动化 Judge 模型，减少人工构造 hard-negative pairs 的依赖。
- 探索无需重训练的自适应 EAGLE drafter，应对 prompt 动态变化。
- 将该框架推广至其他垂直领域（如金融、医疗），验证其通用性。
- 研究更低精度格式（如 INT4）与 speculative decoding 的协同潜力。

---

> 📌 **一句话总结**：本文提出了一套完整的 **multi-agent system 生产部署优化方案**，通过 **CPT+SFT+DPO 蒸馏 + EAGLE+FP8 推理加速**，在真实企业场景中实现了 **4.48× 吞吐提升而不损失智能水平**，为 LLM-based agent 的规模化落地提供了重要实践范式。

</details>

---

### 2. [Pulse: Training Acceleration for Large Diffusion Models with Automatic Pipeline Parallelism](https://arxiv.org/abs/2606.19163)

**Authors**: Boran Sun, Guoyong Jiang, Lin Zhang, Chen Chen, Yuechen Tao, Zhishu Che, Jieling Yu, Shan Chang, Huaxi Gu, Fangming Liu, Bo Li  
**Category**: cs.DC  
**Published**: 2026-06-18  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2606.19163v1  

#### Abstract
Diffusion models are now a dominant approach for high-fidelity image and video generation, yet scaling their training across GPU clusters remains challenging. Unlike transformer-only architectures, diffusion backbones commonly adopt UNet-style encoder-decoder structures with heterogeneous layers and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PULSE: Training Acceleration for Large Diffusion Models with Automatic Pipeline Parallelism**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
- 当前的 **Pipeline Parallelism (PP)** 在训练具有 **UNet-style encoder-decoder 结构** 和 **long-range skip connections** 的扩散模型时面临严重通信瓶颈。
- 传统的 PP（如 1F1B）假设层间是顺序依赖的，但 skip connections 导致早期 encoder 层的激活必须跨多个设备传输到远端 decoder 层，造成大量 **peer-to-peer (P2P) 通信开销**。
- 这些 skip-induced 通信在总通信量中占比高达 85%~90%，成为训练吞吐量的主要限制因素。

### 🚀 **提出了什么新方法或新思路**
提出 **PULSE** —— 一种专为带长距离 skip connections 的扩散模型设计的自动流水线并行训练系统，其核心思想是：

> **Enforce Skip Locality（强制跳过连接局部性）**  
> 将由 skip connection 连接的 encoder 和 decoder 层 **放置在同一设备上（collocation）**，并将 skip 激活缓存在本地供反向传播使用，从而彻底消除跨设备 skip 数据传输。

围绕这一核心洞察，PULSE 包含三个关键技术组件：

1. **Skip-aware Dynamic Programming Partitioner**  
   - 支持对称编码器-解码器块的共置约束，平衡异构阶段的工作负载。
   - 使用双向动态规划算法进行细粒度操作划分，最小化最慢 stage 的执行时间。

2. **Constraint-aware Schedule Synthesizer (基于 ILP)**  
   - 构建满足 collocation、设备独占性和执行顺序等约束的调度器。
   - 自动生成高效的 **wave-like 执行模式**，减少 pipeline bubbles。

3. **Hybrid Parallelism Tuner**  
   - 联合优化 Pipeline Parallelism (PP) 和 Data Parallelism (DP) 的度数以及 microbatch size。
   - 在内存和网络带宽限制下最大化训练吞吐量。

### 🔍 **相比现有方法的优势**
| 方面 | PULSE vs. Baselines |
|------|---------------------|
| **通信效率** | 减少 skip-induced 通信达 **89%~90%**，显著降低 P2P 流量 |
| **训练吞吐量** | 最高提升 **2.3×**，尤其在通信受限硬件上优势明显 |
| **可扩展性** | 支持更大模型训练（如 7.2B 参数 SDv2），避免 OOM |
| **自动化程度** | 自动完成 partitioning、scheduling 和 hybrid 配置搜索，无需人工调优 |

---

## 2. **核心实验方法和设置**

### 🧪 **使用的模型**
在以下三种主流扩散模型架构上验证：
- **UViT**：Vision Transformer-based diffusion model
- **Stable Diffusion v2 (SDv2)**：广泛使用的文本到图像生成模型
- **Hunyuan-DiT**：腾讯提出的多分辨率中文理解扩散 Transformer

同时测试了不同参数规模的变体（从 0.5B 到 7.2B 参数），以评估可扩展性。

### 💻 **硬件平台**
- **2-node NVIDIA V100 集群**  
  - 每节点 8× V100 GPU（32GB 显存）
  - 节点内带宽：300 GB/s (NVLink)，节点间：10 GB/s (Infiniband)

- **8-node Ascend 910A 集群（64 NPUs）**  
  - 更低带宽环境（intra: 30 GB/s, inter: 19 GB/s），模拟资源受限场景

### 📊 **评估指标**
- **Training Throughput (samples/sec)**：每秒处理的样本数
- **Communication Volume (MB)**：单个 microbatch 前向过程中的总通信量
- **End-to-end Iteration Time**：完整训练迭代耗时分解（计算 vs. 通信）
- **Memory Usage**：峰值显存占用

### ⚖️ **基线方法对比**
| 方法 | 类型 | 说明 |
|------|------|------|
| **Megatron 1F1B** | Pipeline Parallelism | 经典流水线调度，按块顺序切分 |
| **Hanayo** | Wave-style PP | 支持波浪形调度，但未考虑 skip collocation |
| **DeepSpeed ZeRO-2** | Data Parallelism | 分片数据并行，高频 AllReduce 通信 |

所有方法采用相同 hybrid 设置和 microbatch size 保证公平比较。

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据**

#### ✅ 在 **2-node V100 集群** 上：
| 模型 | 方法 | 吞吐量 (samples/sec) | 通信量 (MB) |
|-------|--------|------------------|-------------|
| SDv2 (4.6B) | PULSE | **274.6** | **6.71** |
| | Megatron 1F1B | 114.4 | 61.05 |
| | ZeRO-2 | 250.0 | 276.48 |
| | → 提升：**~140%** vs 1F1B，**~10%** vs ZeRO-2 |

| UViT (2.7B) | PULSE | 141.5 | 24.19 |
| | Hanayo | 124.5 | 294.28 |
| | → 通信减少 **90%+**，吞吐提升 **13.6%** |

#### ✅ 在 **8-node Ascend 910A 集群** 上（低带宽环境）：
| 模型 | 方法 | 吞吐量提升倍数 |
|-------|--------|------------------|
| SDv2 (4.6B) | PULSE vs Megatron 1F1B | **2.3×** |
| | PULSE vs ZeRO-2 | **2.31×** |
| | PULSE vs Hanayo | **2.87×** |

> 🔹 通信体积减少 **高达 90%**，尤其在 Hunyuan-DiT 上从 4385 MB 降至 95 MB。

---

### 🔬 **消融实验结果**

#### **(1) Skip-aware 动态划分消融**
- 对比：**Dynamic Partitioning (PULSE)** vs **Block-wise 分配**
- 结果：
  - 在 **SDv2** 上（结构不对称，有下采样/上采样）：吞吐提升 **85.5%**
  - 在 **UViT/Hunyuan-DiT**（均匀 Transformer 块）：仅提升 1–2%
- ➤ 表明：**非均匀结构模型更能从 skip-aware 平衡划分中受益**

#### **(2) Hybrid Parallelism 配置影响**
- 实验设置：固定 8 GPUs，调整 $ P \in \{2,4,8\} $，相应调整 $ G $
- 发现：
  - **SDv2 (1.7B)**：$ P=4 $ 时吞吐最高（257.0 samples/sec），因允许更大的 microbatch size（32 vs 16）
  - $ P=8 $ 时性能下降，因通信开销剧增（通信量从 0.77MB → 4.84MB）
  - **UViT/Hunyuan-DiT**：随着 $ P $ 增加吞吐单调下降，说明过度切分无益
- ➤ 表明：PULSE 的 hybrid tuner 可找到最优 $ P/G/b $ 组合

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **Skip-induced communication 是扩散模型 PP 训练的主要瓶颈**，占总通信量 **>85%**。
2. **通过 skip-aware collocation 可完全消除此类通信**，带来巨大性能收益。
3. **PULSE 实现了高达 2.3× 的吞吐加速**，并在通信受限环境中表现尤为出色。
4. **支持更大规模模型训练**：成功训练 7.2B 参数的 SDv2，而 baseline 方法 OOM。
5. **wave-like schedule 是 skip collocation 下的自然高效执行模式**，可通过 ILP 离线生成模板复用。

---

### ⚠️ **方法的局限性**
1. **主要适用于对称 encoder-decoder 架构**（如 UNet/DiT），对于非对称或稀疏 skip 结构效果可能减弱。
2. **未集成 Tensor Parallelism (TP)**：虽然可组合使用，但当前框架未联合优化 TP+PP。
3. **ILP 求解器用于小规模实例生成模板**，虽离线运行不影响在线性能，但在极端配置下泛化能力需进一步验证。

---

### 🔮 **未来工作方向**
1. **扩展至部分 skip collocation 场景**：自动识别高通信代价的 skip 对并优先共置。
2. **支持更复杂的 skip 拓扑结构**（如跳跃多层、非对称连接）。
3. **联合优化 TP + PP + DP**：构建统一的 multi-dimensional 并行策略搜索空间。
4. **适配更多模态任务**：如 text-to-video、multi-modal fusion 中的 skip 结构优化。

---

> 📌 **一句话总结**：  
> **PULSE 通过“skip-aware collocation + 自动化 pipeline 优化”范式，首次将 skip locality 作为首要优化目标，显著提升了大规模扩散模型的训练效率，在真实集群上实现了最高 2.3× 的吞吐提升和近 90% 的通信削减。**

</details>

---

### 3. [EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts](https://arxiv.org/abs/2606.18967)

**Authors**: Minseo Kim, Minjae Lee, Seunghyuk Oh, Kevin Galim, Donghoon Kim, Coleman Hooper, Harman Singh, Amir Gholami, Hyung Il Koo, Wonjun Kang  
**Category**: cs.LG  
**Published**: 2026-06-18  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.18967v1  

#### Abstract
Reinforcement learning (RL) has become a representative post-training paradigm for LLMs, enabling strong reasoning and agentic capabilities. However, rollout generation remains a dominant latency bottleneck because autoregressive sampling decodes responses sequentially and a small number of long-tai...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在基于**Reinforcement Learning (RL)** 的大语言模型（LLM）后训练中，**rollout 生成**是主要的延迟瓶颈。传统的**autoregressive (AR) 解码**逐个生成 token，而长尾响应（long-tailed generations）会显著拖慢整体训练速度。

尽管 **Speculative Decoding (SD)** 被广泛用于固定模型推理以加速解码，但它在 RL rollouts 中面临两大挑战：
1. **目标策略（target policy）持续演化**，导致任何固定的 drafter 快速失效，降低 block efficiency；
2. **active batch size 随时间动态变化**：早期 batch 大、计算密集（compute-bound），后期 batch 小、内存密集（memory-bound），使得 SD 在不同阶段表现不一。

因此，直接将现有 SD 方法应用于 RL rollouts 效果不佳。

---

### **提出的新方法：EfficientRollout**
为解决上述问题，作者提出了 **EfficientRollout**，一个专为 RL rollouts 设计的**系统感知的自推测解码框架**（system-aware self-SD framework），包含三大核心组件：

#### ✅ **(1) 自诱导量化 drafter（Target-induced Quantized Drafter）**
- 从当前目标模型（target model）**实时诱导出一个权重量化（weight-quantized）的 drafter**，而非依赖外部小模型或历史轨迹。
- 使用 **4-bit RTN 量化**（Round-to-Nearest）对 FFN 和 QKVO 投影层进行压缩，大幅降低 drafter 的权重加载开销。
- **优势**：无需额外预训练或在线适应（online adaptation），始终与演化的策略保持同步，维持高 block efficiency。

#### ✅ **(2) 系统感知的 SD 切换策略（Regime-aware SD Toggle Policy）**
- 基于 **roofline 模型**判断当前是否处于适合启用 SD 的运行时状态。
- 在早期大 batch、compute-bound 阶段禁用 SD（避免因并行验证开销过大而变慢），仅在后期小 batch、memory-bound 阶段启用 SD。
- 使用校准后的 roofline 模型预测 SD 加速潜力，当预测 speedup > 1 + ε 时才开启 SD。

#### ✅ **(3) 自适应 draft 长度控制（Adaptive Draft-Length Control）**
- 动态调整每次 speculative 的 draft 长度 γ，基于观测到的 block efficiency τ。
- 若 τ 接近当前 γ 下的理论上限，则逐步增加 γ；若接受率过低，则减小 γ。
- 引入“耐心窗口”（patience P）防止因短期波动频繁切换 γ。

---

### **相比现有方法的优势**
| 方法类别 | 是否参数化 | 是否免在线适应 | 是否免 warm-up | 本工作优势 |
|--------|-----------|----------------|---------------|------------|
| **History-based** | 否 | 是 | 否 | 无需历史收集，避免冷启动 |
| **Learned Auxiliary** | 是 | 否 | 条件性 | 无需额外训练，无适配开销 |
| **Target-induced (ours)** | 是 | 是 | 是 | ✅ 全部满足 |

- **无需额外 drafter 训练**，部署简单；
- **始终与目标策略同步**，block efficiency 高且稳定；
- **系统感知调度**，避免在不利阶段启用 SD 导致反向减速；
- **端到端加速**，同时保持 rollout 分布不变（lossless）。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **SimpleRL-8k-hard**：对应 MATH 数据集中难度等级 3–5 的数学题，用于 Qwen2.5-7B 和 Qwen2.5-14B。
- **SimpleRL-8k-medium**：对应 MATH 等级 1–4，用于 Llama3.1-8B-Instruct。
- 所有任务均采用 **RLVR（Reinforcement Learning with Verifiable Rewards）** 范式，奖励来自规则检查器。

---

### **实验设置**
- **模型规模**：
  - Qwen2.5-7B / 14B
  - Llama3.1-8B-Instruct
- **训练配置**：
  - Batch size: 128
  - Rollout 温度: 1.0
  - 最大响应长度: 8,192 tokens
  - 训练步数: 100 steps
- **硬件平台**：
  - 单节点 8× A100-80GB GPUs
  - 使用 **vLLM + veRL** 作为推理后端
  - 采用 **Marlin W4A16 kernel** 实现高效 4-bit 推理

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Preparation Time** | SD 方法特有开销（如量化、历史查找、drafter 更新等） |
| **Rollout Gen. Time** | rollout 生成耗时（主瓶颈） |
| **Step Time** | 端到端训练步耗时（含 logprob、policy update 等） |
| **Block Efficiency (τ)** | 每次 speculative 平均产出的有效 token 数 |
| **Acceptance Rate (α)** | draft token 被接受的比例 |
| **Draft Length (γ)** | 每次 speculative 的 draft token 数量 |

---

### **基线方法对比**
| 基线 | 类型 | 特点 |
|------|------|------|
| **veRL (AR)** | 基准 | 标准加速 AR 解码，无 SD |
| **History-based (Spec-RL)** | 历史重用 | 使用前轮 rollout 做 prefix 匹配，lossy |
| **Learned Auxiliary** | 辅助 drafter | EAGLE3-style，需在线训练，引入额外开销 |
| **Quantized Self-SD** | 消融版本 | 固定启用量化 self-SD，无 toggle 与 adaptive γ |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 2）**

| Model | 方法 | Rollout Gen. ↓ | Step Time ↓ | Block Eff. τ | Acceptance α |
|-------|------|----------------|-------------|--------------|-------------|
| **Qwen2.5-7B** | veRL (AR) | 82.4s | 132.6s | – | – |
| | EfficientRollout | **66.3s (-19.6%)** | **115.7s (-12.7%)** | 8.6 | 98.2% |
| **Qwen2.5-14B** | veRL (AR) | 126.6s | 221.1s | – | – |
| | EfficientRollout | **105.3s (-16.8%)** | **197.2s (-10.8%)** | 7.0 | 97.6% |
| **Llama3.1-8B** | veRL (AR) | 126.4s | 186.9s | – | – |
| | EfficientRollout | **112.9s (-10.7%)** | **172.2s (-7.9%)** | 5.4 | 95.8% |

> ✅ **最高实现 19.6% 的 rollout 加速 和 12.7% 的端到端训练加速**

---

### **与基线方法对比**
- **vs History-based (Spec-RL)**：
  - 虽然准备开销低（~1.1s），但 prefix 重用率仅 4.4% ~ 50.2%，无法覆盖验证开销，**反而增加 rollout 时间**。
- **vs Learned Auxiliary**：
  - block efficiency 仅 1.2~2.4，远低于 EfficientRollout 的 5.4~8.6；
  - 在 Llama3.1-8B 上甚至导致 **+25.6% 的 step time 增加**；
  - 受限于 drafter 与 RL rollout 分布不匹配。
- **vs Quantized Self-SD（无 toggle）**：
  - 在 Qwen2.5-14B 上出现**负加速**（+6.7% rollout time），说明盲目启用 SD 有害；
  - 验证了 **regime-aware toggle 的必要性**。

---

### **消融实验结果**
#### 🔹 **Regime-aware Toggle 的影响（Figure 4）**
- 通过关闭早期 speculation（仅前 6–11% 步骤），**避免 compute-bound 阶段的验证开销**；
- 即使 block efficiency 略低，仍能获得更高实际加速比。

#### 🔹 **Adaptive γ 控制的影响（Figure 5a）**
- 固定 γ=5 或 γ=11 的 rollout 加速分别为 13.5% 和 11.8%；
- **adaptive γ 实现 19.6% 加速**，证明动态调整可更好利用训练过程中 drafter 质量提升。

#### 🔹 **Policy Sharpening 提升 drafter 对齐度（Figure 9–10）**
- 随着 RL 训练推进，target policy 分布变得更 sharp（entropy 下降）；
- 与之对应的 **quantized drafter 的 first-token acceptance 和 block efficiency 显著上升**（Pearson r ≈ -0.99）；
- 表明 **post-training 自然提升了 self-drafter 的有效性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Self-SD + 权重量化** 是适用于 RL rollouts 的理想方案：无需外部 drafter，始终与策略同步，block efficiency 高达 8.6。
2. **系统感知调度至关重要**：盲目启用 SD 在 compute-bound 阶段会导致性能下降；**roofline 模型可有效识别 SD-beneficial regime**。
3. **draft length 应随训练动态调整**：早期 γ 宜小，后期可增大，adaptive 控制优于固定 γ。
4. **现有辅助 drafter 方法难以直接迁移至 RL 场景**：public checkpoints 在 RL rollout 上 block efficiency 普遍 < 2.5，难以抵消 overhead。

---

### **方法的局限性**
- **依赖 per-step 模型量化**：虽然使用轻量 RTN，但仍引入 ~1.3–2.6s 的 per-step 开销。
- **未探索 tree verification**：目前仅使用 chain verification，tree-based 可能进一步提升效率。
- **量化方式较基础**：RTN 可能在某些模型上初期 drafter 质量较低，更高级量化（如 AWQ）可能更好但开销更高。
- **KV-cache 未量化**：在极长上下文（>64k）中，KV-cache 成为主要瓶颈，sparse attention 或 KV quantization 可补充。

---

### **未来工作方向**
1. **集成 tree-based speculative decoding**：在动态 batch 下支持树形 speculative。
2. **更高效的 per-step 量化方法**：研究低开销、高保真的 step-wise 量化策略。
3. **KV-cache aware drafting**：在长上下文场景下结合 sparse attention 或 KV quantization 进一步优化。
4. **跨设备协作优化**：探索 CPU-offload、异构执行等策略降低 per-step 量化延迟。

---

> 📌 **总结一句话**：  
> **EfficientRollout 通过“自诱导量化 drafter + 系统感知切换 + 自适应 draft 长度”三位一体设计，在无需额外训练、无需 warm-up 的前提下，实现了高达 19.6% 的 rollout 加速，是首个真正适用于 on-policy RL rollouts 的高效 speculative decoding 框架。**

</details>

---

### 4. [JetFlow: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting](https://arxiv.org/abs/2606.18394)

**Authors**: Lanxiang Hu, Zhaoxiang Feng, Yulun Wu, Haoran Yuan, Yujie Zhao, Yu-Yang Qian, Bojun Wang, Daxin Jiang, Yibo Zhu, Tajana Rosing, Hao Zhang  
**Category**: cs.CL  
**Published**: 2026-06-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.18394v1  

#### Abstract
Speculative decoding (SD) accelerates autoregressive Large Language Models (LLMs) by drafting multiple tokens and verifying them in parallel, but it faces a scaling limitation: increasing the draft budget improves speed only when acceptance remains high and drafting overhead stays low. This ceiling ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：JetFlow: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代自回归（AR）大语言模型（LLMs）在推理时面临严重的**串行解码瓶颈**，导致生成延迟高，尤其在数学、编程和长对话等需要长序列生成的任务中表现尤为明显。

**Speculative Decoding (SD)** 是一种通过“草稿-验证”机制加速推理的技术：一个轻量级的 `drafter` 模型先生成多个候选 token，再由目标模型并行验证。然而，SD 面临一个**扩展性天花板（scaling ceiling）**：
- 增加草稿预算（draft budget）仅当**接受率（acceptance rate）高**且**草稿开销（drafting cost）低**时才能提升速度。
- 现有方法难以同时优化这两者，陷入**因果性-效率困境（causality-efficiency dilemma）**：
  - **自回归草稿器（如 EAGLE）**：路径条件强，接受率高，但需逐层生成，成本随树深增长。
  - **双向块扩散草稿器（如 DFlash）**：单次前向传播生成所有位置，效率高，但各分支独立预测，导致生成的树节点可能语义不一致，降低接受率。

---

### **提出的新方法：JETFLOW**
为打破上述困境，论文提出 **JETFLOW** —— 一种结合**并行效率**与**分支因果性**的头式（head-based）Speculative Decoding 框架。

#### **核心创新点**
1. **因果并行草稿头（Causal Parallel Draft Head）**
   - 在单次前向传播中并行生成整棵候选树。
   - 引入**树因果注意力掩码（tree-causal attention mask）**，确保每个分支节点仅依赖其祖先路径，而非全局上下文。
   - 实现了类似自回归的路径条件分布，但计算代价接近并行方法。

2. **与目标模型对齐的训练方式**
   - 利用冻结的目标模型中间隐藏状态作为输入特征。
   - 使用 **forward KL 蒸馏损失**，保留目标模型对多个合理延续的软标签偏好，避免 reverse KL 的模式坍缩问题。

3. **高效的树构建与验证算法**
   - 使用累积对数概率作为分支评分函数，进行最佳优先扩展（best-first expansion）。
   - 在 vLLM 中实现定制化的 **paged tree-attention kernel**，支持高效树形验证。

---

### **相比现有方法的优势**
| 方法 | 接受率 | 草稿成本 | 扩展性 |
|------|--------|----------|--------|
| EAGLE-3 | 高 | 高（自回归） | 差（随深度增加） |
| DFlash | 低（分支无关） | 极低 | 一般 |
| **JETFLOW** | **高**（路径条件） | **极低**（单次前向） | ✅ **强（随预算扩展显著）** |

> ✅ **JETFLOW 成功打破了“高接受率”与“低成本”的权衡，实现了真正的可扩展性。**

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **数学任务**：GSM8K, MATH-500, AIME25
- **编程任务**：HumanEval, MBPP, LiveCodeBench (LCB)
- **开放对话**：MT-Bench
- **训练数据**：Nemotron Post-Training Dataset V2（含代码、数学、STEM、聊天），共约 780K 示例；部分使用目标模型重新生成的续写作为监督信号。

---

### **实验设置与评估指标**

| 设置项 | 说明 |
|-------|------|
| **模型** | Qwen3-8B（dense）、Qwen3-30B-A3B（MoE） |
| **硬件** | H100 / B200 / H200 GPUs |
| **草稿预算** | 16, 32, 64, 128, 256 tokens |
| **评估模式** | Greedy 与 Non-greedy 解码均测试 |
| **主要指标** | 
| - **Speedup**：端到端解码速度相对于标准 AR 的加速比 |
| - **Average Accepted Length (T)**：每轮 Speculative Iteration 平均接受的 token 数 |
| - **Throughput (TPS)**：在 vLLM 服务引擎下的吞吐量（tokens/sec） |

---

### **基线方法对比**
- **EAGLE-3**：多层特征融合的自回归式头式草稿器。
- **DFlash**：基于块扩散的并行草稿头，单次生成多个 token。
- **DFlash-T**：将 DFlash 的输出用于构建树结构（类似 DDTree），作为树形版本的对比基线。

> 所有方法均在同一训练数据和超参下训练，保证公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **高预算场景（Budget=256）下的最大加速比**
| 基准 | JetFlow Speedup | EAGLE-3 / DFlash-T 最佳 |
|------|------------------|--------------------------|
| **MATH-500** | **9.64×** | ~8.78× (DFlash-T) |
| **MT-Bench** | **4.58×** | ~4.26× |
| **GSM8K** | **7.82×** | ~7.04× |
| **HumanEval** | **7.12×** | ~6.73× |

> 💡 在 MATH-500 上达到 **10.76 的平均接受长度**，远超基线。

---

#### **低预算场景（Budget=16）**
| 方法 | MATH-500 Speedup | T |
|------|-------------------|----|
| EAGLE-3 | 6.12× | 7.83 |
| DFlash | 6.12× | 7.83 |
| **JetFlow** | **6.12×** | **7.83** |

> ✅ 在小预算下与最优基线持平，表明其基础性能强劲。

---

#### **随预算增加的扩展能力**
| 方法 | MATH-500 (Budget=64 → 256) |
|------|----------------------------|
| DFlash-T | 6.51× → 8.78× (+35%) |
| **JetFlow** | **6.76× → 9.64× (+42%)** |

> ✅ **JetFlow 更能有效利用更大的草稿预算转化为更高的接受长度和加速比**。

---

### **与基线方法的对比结果**
- 在所有任务上，**JetFlow 全面优于 EAGLE-3 和 DFlash/DFlash-T**。
- 尤其在**高预算、高复杂度任务（如数学推理）** 上优势显著。
- 在 **MT-Bench** 这类开放生成任务中，仍保持超过 **4.5× 加速**，证明其泛化能力强。

---

### **消融实验结果**

#### **(1) 损失函数对比（Table 4）**
| 损失类型 | MATH-500 Speedup |
|---------|------------------|
| SFT | 8.42× |
| Forward KL | **8.46×** |
| Reverse KL | 5.25× |

> ❌ Reverse KL 导致严重性能下降（↓38%），因其倾向于集中概率质量，不利于多样性树扩展。

---

#### **(2) 学习率影响（Table 3）**
- 最优学习率为 **3×10⁻⁴**，在此附近性能稳定。
- 过小（5×10⁻⁵）则欠拟合，过大无明显增益。

---

#### **(3) 训练数据来源（Table 6）**
| 数据类型 | MATH-500 (Budget=256) |
|---------|------------------------|
| 再生目标模型序列（Regenerated） | **9.64×** |
| 原始语料库（Corpus） | 3.66× |

> ✅ 使用目标模型自身生成的数据作为监督信号效果最好，强调了**分布对齐的重要性**。

---

#### **(4) 因果 vs 扩散头（Table 7）**
| 头类型 | γ=0 (Uniform Weighting) | γ=7 (Optimal for DFlash) |
|--------|--------------------------|----------------------------|
| Causal Head | **8.29×** | **8.41×** |
| Diffusion Head | 5.46× | 8.36× |

> ✅ **因果头对训练配置鲁棒性强**，而扩散头严重依赖 γ 参数调优。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **因果性是构建高质量候选树的关键**：分支必须基于其祖先路径进行条件建模，否则即使单个 token 合理，组合后也可能语义断裂。
2. ✅ **并行生成 + 分支因果性可以兼得**：通过树因果注意力掩码，JETFLOW 实现了单次前向传播中的路径条件预测。
3. ✅ **forward KL 蒸馏优于 SFT 和 reverse KL**：更适合保留目标模型的多峰分布特性，利于树扩展。
4. ✅ **更大的草稿预算确实能带来更高加速，但前提是接受率不下降** —— JETFLOW 正是解决了这一前提。

---

### **方法的局限性**
1. **依赖目标模型中间状态**：需访问冻结目标模型的隐藏层输出，可能限制在某些封闭 API 场景的应用。
2. **训练成本存在**：虽然部署简单，但仍需额外训练一个 draft head。
3. **服务负载敏感**：在大批量请求下，过大树预算可能导致内存压力上升，收益递减（见 Table 11）。

---

### **未来工作方向**
1. **动态预算调度**：根据当前 batch size 和 GPU 负载动态调整草稿树大小，最大化吞吐。
2. **跨模型迁移**：探索是否可在不同架构的目标模型间共享或微调 draft head。
3. **与检索增强结合**：将 Prompt Lookup Decoding 或 SuffixDecoding 与 JETFLOW 结合，进一步提升长文本一致性。
4. **非自回归目标模型适配**：拓展至 diffusion LLMs 等非 AR 架构。

---

> 🔚 **总结一句话**：  
> **JETFLOW 通过“因果并行草稿头”打破了 Speculative Decoding 的扩展瓶颈，在保持极低草稿成本的同时大幅提升接受率，实现了高达 9.64× 的端到端加速，是迈向高效 LLM 推理的重要一步。**

</details>

---

### 5. [ShuntServe: Cost-Efficient LLM Serving on Heterogeneous Spot GPU Clusters](https://arxiv.org/abs/2606.18600)

**Authors**: Seungwoo Jeong, Moohyun Song, Juhyun Park, Kyungyong Lee  
**Category**: cs.DC  
**Published**: 2026-06-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.18600v1  

#### Abstract
As large language model (LLM) services become widely adopted, the cost of GPU resources for serving these models in cloud environments has emerged as a critical concern. Spot instances offer up to 90% cost savings over on-demand instances, but their frequent interruptions and limited availability po...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ShuntServe: Cost-Efficient LLM Serving on Heterogeneous Spot GPU Clusters

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大型语言模型（LLM）服务在云端部署时面临高昂的 GPU 资源成本。虽然 **spot instances** 可提供高达 90% 的成本节省，但其频繁中断和有限可用性对持续推理构成挑战。尤其在 **homogeneous GPU 集群** 中，单一 GPU 类型的 spot 实例易因需求波动而同时失效（correlated failures），导致系统脆弱。

此外，现有 LLM serving 系统多为同构环境设计，在异构 GPU 上部署时会出现严重的 **负载不均衡** 和性能瓶颈，无法有效利用多样化 spot pool 的互补可用性。

---

### 提出的新方法与创新思路

ShuntServe 是首个专为 **heterogeneous spot GPU clusters** 设计的高性价比 LLM 推理系统，提出以下三大核心技术：

#### （1）基于 Roofline 模型的分析型性能估计器（Analytical Serving Performance Estimator）
- 利用 **roofline model** 对不同并行配置（PP/TP）下的推理延迟进行建模，无需全面 profiling。
- 分别估算 **prefill**（计算密集）和 **decode**（内存密集）阶段的延迟，并结合通信开销（α-β model）精确预测端到端吞吐量。
- 仅需一次轻量级硬件校准（FLOPS、Mem BW、Net BW），即可泛化至多种配置。

#### （2）基于动态规划（DP）+ Beam Search 的模型放置优化器（Model Placement Optimizer）
- 联合优化 **node configuration**、**parallelization strategy** 和 **layer assignment**。
- 在巨大的搜索空间中使用 **DP + Beam Search** 高效探索最优解，避免穷举。
- 支持非均匀层划分（uneven layer partitioning）和 per-stage TP，以平衡异构设备间的执行时间。

#### （3）面向 spot 中断的容错机制
- **Output-preserving request migration**：通过 **recomputation** 恢复被中断请求的状态，而非传输 KV cache。
- **Concurrent initialization via shared tensor store**：
  - 引入独立的 **Shared Tensor Store** 进程管理模型权重与 KV cache。
  - 新旧 inference engine 可共享 tensor 数据，实现替换节点初始化与当前服务的重叠，显著减少停机时间。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **成本效率** | 充分利用多种 spot GPU 类型，提升资源利用率，降低对 on-demand 实例依赖 |
| **吞吐量** | 通过异构感知的模型放置最大化 pipeline throughput |
| **容错能力** | 结合 recomputation 与并发初始化，在 spot 中断下保持低延迟恢复 |
| **可扩展性** | 分析型估计器避免组合爆炸式的 profiling 开销 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **模型**：
  - `Llama-3.1-70B-Instruct`（BF16）
  - `Qwen3-32B`（BF16）
- **工作负载 trace**：
  - **Azure Conversation Dataset**：真实生产场景中的 LLM 请求轨迹，包含动态到达率、输入/输出长度分布。
  - 平均输入长度：763 tokens；平均输出长度：232 tokens；平均请求速率：4.67 req/s。

---

### 实验设置
- **集群配置**（AWS）：
  - 3 × `g6.12xlarge` → 12 × **L4** (24GB)
  - 2 × `g5.12xlarge` → 8 × **A10G** (24GB)
  - 4 × `g6e.xlarge` → 4 × **L40S** (48GB)
  - 总计：24 GPUs，672 GB GPU 内存
- **评估周期**：约 1,000 GPU 小时（不含调试）

---

### 评估指标
| 工作负载类型 | 指标 |
|-------------|------|
| **Offline**（饱和负载） | Throughput (RPS) |
| **Online**（实时请求） | 
  - TTFT（Time to First Token）
  - TPOT（Time per Output Token）
  - End-to-end latency（均值 & P90）
| **成本效率** |
  - Offline：cost per throughput
  - Online：request latency × cost

---

### 基线方法对比
| Baseline | 描述 |
|--------|------|
| **vLLM** | 主流框架，采用均匀层划分（even partitioning），适用于同构集群 |
| **AlpaServe** | 同构集群优化，使用 DP 均衡 stage latency |
| **HexGen** | 异构集群优化，结合遗传算法与 DP 进行模型放置 |
| **On-demand Only** | 全部使用按需实例，无中断 |
| **No Handle** | 使用 spot 实例但无容错机制 |
| **Request Migration** | 仅支持输出保留迁移（无并发初始化） |
| **Concurrent Initialization** | 仅支持并发初始化（无请求迁移） |

---

## 3. 主要实验结果和性能指标

### 吞吐量提升（Offline）
| 模型 | ShuntServe | 最优基线 | 提升倍数 |
|------|-----------|----------|---------|
| Llama-3.1-70B | **1.53 req/s** | HexGen (1.31 req/s) | **1.42×** |
| Qwen3-32B | **4.59 req/s** | vLLM (3.39 req/s) | **1.35×** |

> ✅ ShuntServe 在两个模型上均取得最高吞吐量，显著优于 state-of-the-art。

---

### 成本效率提升
| 场景 | 成本效率改进 |
|------|------------|
| Offline Serving | **31.9%** 低于 on-demand |
| Online Serving | **31.2%** 低于 on-demand |

> 💰 即使考虑中断带来的性能损失，ShuntServe 仍能实现近三分之一的成本节约。

---

### 在线延迟表现（Online）
| 指标 | Llama-3.1-70B (ShuntServe vs vLLM) |
|------|-------------------------------|
| TTFT（中位数） | **1.37s vs 1.70s**（↓19%） |
| TPOT（中位数） | **0.82s vs 0.67s**（略高） |
| TTFT（P90） | **2.39s vs 7.70s**（↓69%） |

> ⚠️ 虽然 TPOT 略高，但 P90 TTFT 显著改善，说明尾延迟控制优秀。

---

### 容错机制消融实验（Spot Interruption 下的表现）

#### （1）吞吐量对比（Offline）
| 方法 | Llama-3.1-70B (RPS) | 提升（vs No Handle） |
|------|---------------------|--------------------|
| No Handle | 0.85 | — |
| Request Migration | 0.92 | +8% |
| Concurrent Initialization | 1.03 | +21% |
| **ShuntServe** | **1.12** | **+32%** |

> ✅ 并发初始化比单纯请求迁移更有效，二者结合效果最佳。

#### （2）端到端延迟（Online）
| 方法 | Llama-3.1-70B（均值 / P90） |
|------|-----------------------------|
| No Handle | 162.67s / 355.51s |
| Request Migration | 155.88s / 320.27s |
| Concurrent Initialization | 126.10s / 251.95s |
| **ShuntServe** | **115.08s / 211.44s** |

> 📉 ShuntServe 将平均延迟降低约 30%，P90 降低超 40%，接近 on-demand 表现。

#### （3）并发初始化耗时分析
- 替换节点准备时间：**41.5s**
- Shared Tensor Store 加载：**61.8s**
- Inference Engine 初始化：**64.5s**
- **总初始化时间**：**~111.3s**（可在 AWS 120s grace period 内完成）

> ✅ 实现近乎零停机恢复。

---

## 4. 关键结论和发现

### 主要发现
1. **异构集群是应对 spot GPU 稀缺性的有效策略**：
   - 不同 GPU 类型具有互补的 spot 可用性模式，联合使用可大幅提升资源利用率。
2. **传统均匀划分在异构环境中严重受限**：
   - 必须采用 **非均匀层划分** 和 **per-stage TP** 来消除 pipeline 瓶颈。
3. **recomputation 比 KV cache transfer 更适合 spot 环境**：
   - 尽管长上下文下 transfer 可能更快，但在有限的 grace period 内完成传输风险极高。
4. **最小化 downtime 比恢复状态更重要**：
   - **concurrent initialization** 显著优于单纯的 request migration，尤其是在大模型场景。
5. **ShuntServe 实现了成本与性能的良好平衡**：
   - 较 on-demand 实例节省 **超 30% 成本**，同时维持可接受的服务质量。

---

### 方法的局限性
1. **性能估计器在真实动态负载下精度可能下降**：
   - 当前基于静态批处理和固定序列长度建模，未完全捕捉 kernel launch overhead、SM 利用率等非线性因素。
2. **长上下文场景下 recomputation 开销较大**：
   - 注意力计算呈二次增长，极端长文本可能导致尾延迟升高。
3. **当前设计局限于单区域部署**：
   - 未充分利用跨 region spot 价格差异和冗余备份潜力。

---

### 未来工作方向
1. **混合恢复机制（Hybrid Recovery Scheme）**：
   - 动态选择使用 **recomputation** 或 **KV cache transfer**，依据剩余 grace 时间和请求上下文长度决策。
2. **适应时空变化的 spot 资源调度**：
   - 支持随时间和区域动态调整集群成员和模型布局。
3. **phase-aware model placement**：
   - 结合 prefill-decode disaggregation 技术，将计算密集的 prefill 和内存密集的 decode 分配给最适合的 GPU 类型。
4. **集成 ML-based 性能预测模型**：
   - 将分析型模型与轻量级 ML 模型结合，提升复杂负载下的预测准确性。

---

> 🔚 **总结**：  
> ShuntServe 成功解决了在 **heterogeneous spot GPU clusters** 上高效运行 LLM 的核心难题——**成本、性能与容错之间的权衡**。它通过 **分析型性能建模 + DP+Beam Search 优化 + 并发初始化容错机制**，实现了比现有方案更高的吞吐量和成本效益，为大规模 LLM 的经济化部署提供了可行路径。

</details>

---

### 6. [BLADE: Scalable Bi-level Adaptive Data Selection for LLM Training](https://arxiv.org/abs/2606.18650)

**Authors**: Jiaxing Wang, Deping Xiang, Jin Xu, Zirui Liu, Zicheng Zhang, Guoqiang Gong, Jun Fang, Chao Liu, Pengzhang Liu, Tongxuan Liu, Ke Zhang, Qixia Jiang  
**Category**: cs.LG  
**Published**: 2026-06-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.18650v1  

#### Abstract
As Large Language Model (LLM) datasets scale to trillions of tokens, data selection has emerged as a critical frontier to filter out uninformative noise and construct adaptive learning trajectories. Beyond static heuristic filtering, advanced data selection methods for LLM training largely follow tw...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BLADE: Scalable Bi-level Adaptive Data Selection for LLM Training

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

随着 Large Language Models (LLMs) 的训练数据规模达到万亿级 token，**低质量、冗余或噪声数据**会显著降低模型性能。传统的静态过滤方法无法适应动态训练过程，而现有的 model-aware 数据选择方法存在以下两大瓶颈：

- **Influence-based 方法**：理论严谨，但需要计算逆 Hessian 矩阵（inverse-Hessian），在 LLM 规模下计算不可行。
- **Excess-loss 方法**（如 RHO-1）：计算高效，但依赖一个**预先训练且固定不变的 reference model**，导致其在训练过程中逐渐“过时”，失去判别能力。

因此，如何在保持**理论严谨性**的同时实现**可扩展性和在线适应性**，是当前 LLM 数据选择的核心挑战。

---

### 提出了什么新方法或新思路

作者提出 **BLADE (Bi-Level Adaptive Data sElection)**，一种无需 Hessian 的 bi-level 数据选择框架，核心思想如下：

- 将原始的 bi-level 优化问题（外层选数据，内层训模型）通过 **Lagrange multiplier** 转化为一个**带惩罚项的单层优化问题**，从而**避免了 inverse-Hessian 的计算**。
- 新的目标函数自然导出一种 **excess-loss 形式的评分机制**：  
  $$
  \Delta_n = \mathcal{L}_n(w) - \mathcal{L}_n(u)
  $$
  即 token $n$ 在 reference model $w$ 和 proxy model $u$ 上的损失差。
- 关键创新在于：**reference model $w$ 是动态更新的**，并定期与 proxy model 同步，确保其始终反映当前训练状态，避免滞后。

---

### 相比现有方法的优势

| 维度 | Influence-based | Excess-loss (e.g., RHO-1) | **BLADE (本文)** |
|------|------------------|----------------------------|------------------|
| 理论基础 | 强（bi-level） | 弱（启发式） | **强（从 bi-level 推导）** |
| 可扩展性 | 差（需 HVP/H^-1） | 好 | **好（first-order only）** |
| 动态适应性 | 中等 | 差（static ref） | **强（dynamic ref）** |
| 实现复杂度 | 高 | 低 | **中等（双模型同步）** |

此外，BLADE 还设计了：
- **Memoryless Randomized Block-Coordinate Frank-Wolfe 算法**，支持高效的在线 token 级选择。
- **Token-level loss masking** 机制，保留序列完整性，不影响 autoregressive 训练。

---

## 2. 核心实验方法和设置

### 使用的数据集

#### Domain-Shift Pre-training（领域专业化）
- **Validation Set**: MetaMath + Mammoth（数学推理任务）
- **Training Pool**: OpenWebMath 的 5B 子集（约 14B 数学相关网页 token）
- **Base Models**: TinyLlama-1.1B, LLaMA2-7B

#### General Pre-training（通用能力增强）
- **Validation Set**: OpenHermes-2.5
- **Training Pool**: OpenWebMath + SlimPajama 的混合 5B 数据
- **Base Model**: TinyLlama-1.1B

---

### 实验设置和评估指标

- **选择粒度**：token-level（keeping ratio $\gamma = 0.6$）
- **训练步数**：5000 steps（proxy model）
- **Reference Model 更新**：每 $T=1000$ 步同步一次，每次更新 $K=300$ 步
- **评估任务**：
  - 数学推理：GSM8K, MATH, SVAMP, ASDiv, MAWPS, TabMWP, MathQA
  - 通用能力：MMLU, Hellaswag, OpenBookQA, WinoGrande, ARC, BoolQ, PIQA
- **评估方式**：few-shot Chain-of-Thought (CoT) prompting
- **硬件**：8×NVIDIA H100

---

### 基线方法对比

| 方法 | 类型 | 是否 model-aware | 说明 |
|------|------|------------------|------|
| **Base** | 无选择 | ❌ | 原始模型 |
| **Base-CT** | 全量训练 | ❌ | 在完整候选集上继续预训练 |
| **Random** | 随机采样 | ❌ | 随机选择 3B tokens |
| **MATES*** | Influence-based | ✅ | 使用 influence predictor，sample-level |
| **RHO-1** | Excess-loss | ✅ | 当前 SOTA，token-level，static reference |

> 注：`*` 表示为 sample-level 方法；RHO-1 由作者复现以保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Domain-Shift Pre-training）

#### TinyLlama-1.1B 结果（平均准确率）

| 方法 | 平均准确率 | 相对提升 |
|------|-----------|----------|
| Base | 13.4% | — |
| Base-CT | 16.7% | +3.3pp |
| Random | 14.7% | +1.3pp |
| MATES | 17.7% | +4.3pp |
| RHO-1 | 25.7% | +12.3pp |
| **BLADE** | **29.7%** | **+16.3pp** |

> BLADE 显著优于所有基线，尤其相比 RHO-1 提升 **4.0pp**。

#### LLaMA2-7B 结果（平均准确率）

| 方法 | 平均准确率 |
|------|-----------|
| Base | 31.4% |
| Base-CT | 33.8% |
| Random | 32.3% |
| MATES | 33.4% |
| RHO-1 | 41.1% |
| **BLADE** | **43.8%** |

> BLADE 再次领先，超越 RHO-1 **2.7pp**，并在 7/9 个子任务上表现最佳。

---

### 通用预训练结果（TinyLlama-1.1B）

| 方法 | 平均准确率 |
|------|-----------|
| Base | 49.52% |
| Base-CT | 49.87% |
| Random | 49.59% |
| MATES | 50.37% |
| RHO-1 | 50.58% |
| **BLADE** | **51.73%** |

> BLADE 在通用场景下仍保持领先，**超越 SOTA RHO-1 达 1.15pp**，验证其泛化能力。

---

### 消融实验结果

#### 动态 reference model 的有效性（Figure 2a & 3）

- **RHO-1**：随着训练进行，loss gap ($\Delta$) 分布坍缩至零甚至负值，表明其 selection signal 失效。
- **BLADE**：由于 reference model 定期同步，$\Delta$ 始终保持右偏分布（positive tail），持续识别高价值 token。
- **Oracle Loss**（用 70B 模型评估 selected token 质量）显示 BLADE 所选 token 质量持续下降，而 RHO-1 很快停滞。

#### Reference 更新间隔 $T$ 的影响（Figure 2b）

- 在 $T \in \{500, 1000, 1500, 2000\}$ 下性能几乎一致。
- **结论**：可安全使用较大 $T$ 减少计算开销，无需频繁更新 reference model。

#### Keeping Ratio $\gamma$ 的影响（Figure 2c）

- 最优值在 $\gamma = 0.6$，进一步增加至 0.8 反而导致性能下降。
- **结论**：并非越多越好，过度保留会引入噪声，验证了 adaptive selection 的必要性。

#### 同步机制的重要性（Figure 4）

- 若不进行 $w$ 与 $u$ 的初始化同步（$w^{(e)} = u(eT)$），两模型参数距离迅速发散。
- 同步机制有效控制 drift，使 penalty term 数值稳定，避免梯度爆炸。

---

## 4. 关键结论和发现

### 主要发现

1. **BLADE 成功统一了 influence-based 与 excess-loss 两类范式**：
   - 从 bi-level 出发，推导出 excess-loss 形式，兼具理论与效率优势。
2. **动态 reference model 是关键**：
   - 固定 reference 会快速失效，动态同步能维持 selection signal 的 discriminative power。
3. **First-order 方法足以逼近 bi-level 解**：
   - 通过 penalized formulation + Frank-Wolfe，实现无需 Hessian 的高效优化。
4. **BLADE 具有良好的可扩展性与实用性**：
   - 支持 token-level 在线选择，VRAM 开销可控（通过 offloading）。

---

### 方法的局限性

1. **尚未在百B级以上模型验证**：
   - 当前实验最大为 7B 模型，更大规模下的通信与同步成本未知。
2. **依赖高质量 validation set**：
   - 与大多数 validation-guided 方法一样，performance 受限于 validation data 的代表性。
3. **双模型架构带来额外计算负担**：
   - 尽管比 MATES 快，但仍比 vanilla training 慢约 50%，trade-off 存在。

---

### 未来工作方向

1. **自动化构建动态 validation set**：
   - 当前 validation set 是人工构造的静态集合，未来可探索自动生成或演化机制。
2. **跨尺度数据选择迁移**（见 Appendix D）：
   - 实验发现小模型选出的数据可在大模型上部分复用（BLADE-Tr），但仍有差距，值得深入研究 capacity-aware selection。
3. **扩展到其他模态或任务**：
   - 如 vision-language 或 code generation，验证 BLADE 的通用性。
4. **更轻量化的 reference model 设计**：
   - 探索 shared-weight 或 distilled reference model 以进一步降低开销。

---

> **总结**：BLADE 提供了一条**理论上严谨、实践中可行**的大规模 LLM 数据选择路径，通过将 bi-level 优化转化为 penalty-based 单层问题，并引入动态 reference model，在保持高效的同时克服了现有方法的根本缺陷，是当前 LLM 数据工程中的重要进展。

</details>

---

### 7. [ReMP: Low-Downtime Runtime Model-Parallelism Reconfiguration for LLM Serving](https://arxiv.org/abs/2606.18741)

**Authors**: Haipeng Yuan, Kaining Zheng, Yongshu Bai, Yuchen Zhang, Yunquan Zhang, Baodong Wu, Xiang Gao, Daning Cheng  
**Category**: cs.DC  
**Published**: 2026-06-18  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.18741v1  

#### Abstract
Current large language model (LLM) inference systems universally deploy ultra-large-scale models using a combination of Tensor Parallelism (TP) and Pipeline Parallelism (PP). However, existing systems treat the model parallelism topology as a static configuration that cannot be flexibly adjusted at ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ReMP: Low-Downtime Runtime Model-Parallelism Reconfiguration for LLM Serving

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前主流的 **LLM inference serving** 系统普遍采用 **Tensor Parallelism (TP)** 和 **Pipeline Parallelism (PP)** 来部署超大规模模型。然而，这些系统的并行拓扑结构在启动后是静态固定的，无法在运行时动态调整。

现实中的推理负载具有显著的动态性（如图1所示，不同时段流量差异大），而最优的 TP/PP 配置随负载变化：
- **低负载时**：应优先高 TP 度以降低单请求延迟（TTFT）；
- **高负载时**：应优先高 PP 度以提升吞吐量，避免过度 TP 带来的通信开销。

现有系统只能通过重启服务来切换 TP/PP 配置，导致：
- 数分钟的服务中断；
- KV Cache 全部丢失；
- 大量 Prefill 重计算开销。

这使得动态适应几乎不可行。

---

### 提出的新方法与创新思路
本文提出 **ReMP** —— 一个支持**低停机时间**（low-downtime）的 **runtime model-parallelism reconfiguration** 框架，实现在线无缝切换 TP/PP 拓扑。

其三大核心技术为：

#### （1）解耦模型并行拓扑与运行时状态（Decoupling Topology from Runtime State）
- 将模型权重、KV Cache、通信组、Worker 生命周期等从初始拓扑中解耦。
- 例如，将完整模型状态持久化到 **CPU shared memory** 中，避免每次切换都重新加载 checkpoint。

#### （2）二维 KV Cache 迁移机制（2D KV Cache Migration）
- KV Cache 的布局由两个正交维度决定：
  - **PP 维度**：哪一层属于哪个 pipeline stage；
  - **TP 维度**：哪些 attention heads 属于哪个 tensor-parallel rank。
- ReMP 设计了联合的 **layer-wise + head-range remapping** 迁移策略，在拓扑变更后保留可复用的 KV 状态，避免 cache 丢弃和 prefilled tokens 的重复计算。

#### （3）端到端在线重配置能力（End-to-End Online Reconfiguration）
- 深度集成至 **vLLM v1** 核心模块（executor、worker manager、KV cache manager、communication 初始化流程）。
- 支持在服务不中断的前提下完成整个拓扑切换过程。

---

### 相比现有方法的优势
| 方面 | 传统重启方式 | ReMP |
|------|---------------|-------|
| 切换延迟 | 几分钟 | **1–7 秒**（多数在 1–3 秒内） |
| KV Cache 保留 | 完全丢失 | **选择性迁移保留** |
| 吞吐/延迟表现 | 固定配置妥协 | **按需动态优化** |
| 资源利用率 | 低（需预留最大资源） | 高（弹性适配） |

> ✅ ReMP 将 model-parallel topology 从“部署参数”转变为“运行时可调资源”。

---

## 2. 核心实验方法和设置

### 使用的数据集与工作负载
- **真实流量分析**：基于两家大型服务商的实际线上日志，展示 LLM 请求存在明显的周期性波动（见 Figure 1）。
- **仿真负载**：使用 **BurstGPT-derived request traces**，模拟不同请求压力下的推理场景。
  - 控制变量：相同请求序列用于所有配置比较，确保公平性。

---

### 实验平台与硬件环境
在两个 8-GPU 平台上进行评估：
1. **H100 平台**：
   - 8× NVIDIA H100 GPUs
   - AMD EPYC 7R13 CPU，2TB 主机内存
2. **RTX 5090 平台**：
   - 8× NVIDIA RTX 5090 GPUs
   - Intel Xeon Gold 6530 CPU，960GB 主机内存

> ⚠️ Llama2-70B 因显存限制未在 RTX 5090 上测试。

---

### 模型范围
| 模型 | 类型 | 参数量 | 架构 |
|------|------|--------|------|
| Llama2-7B | Dense | 7B | 32层 |
| Qwen3-30B-A3B | MoE | 30.5B（激活3.3B） | 48层 |
| DeepSeek-R1-Distill-Qwen-32B | Dense | 32B | 64层 |
| Llama2-70B | Dense | 70B | 80层 |

覆盖了 dense 与 MoE 架构，以及从小到大的典型 LLM 规模。

---

### 评估指标

#### （1）重配置成本相关
- **Switching Time**：从旧拓扑切换到目标拓扑所需总时间
- **Speedup over Restart**：$ T_{\text{restart}} / T_{\text{ReMP}} $
- 内部耗时分解：
  - Model loading from shared memory
  - KV cache transfer time

#### （2）服务性能相关
- **TTFT**（Time to First Token）：首 token 延迟
- **TPOT**（Time Per Output Token）：每输出 token 时间
- **Output Throughput**：单位时间内生成的 token 数
- 加权综合得分（结合 throughput、TTFT、TPOT）

---

### 基线方法对比

#### （1）重配置成本对比
- **Baseline**: 传统的 restart-based switching
  - 终止当前实例 → 重新加载 checkpoint → 重建全部 runtime state → 丢弃 KV Cache

#### （2）服务性能对比
- **Fixed Configuration Baselines**:
  - `TP1PP8`：侧重 pipeline parallelism，适合高吞吐但可能增加延迟
  - `TP2PP4`：较平衡的配置
- **ReMP**：利用快速切换能力探测多个候选配置，选择当前负载下性能最佳者作为最终配置

---

## 3. 主要实验结果和性能指标

### （1）重配置效率（Figure 5）

| 模型 | 平均切换时间（H100） | 最快 | 最慢 |
|------|------------------------|------|------|
| Llama2-7B | ~1–2 秒 | — | <2.5s |
| Qwen3-30B-A3B | ~2–3 秒 | — | <3.5s |
| DeepSeek-32B | ~2–3 秒 | — | <3.5s |
| Llama2-70B | ~6–7 秒 | — | <8s |

> 即使对于 70B 模型，也仅需数秒即可完成拓扑切换，远优于重启所需的“分钟级”。

#### 速度提升（Speedup）
- 在 H100 上：**数十倍加速**，部分转换超过 **100×**
- 在 RTX 5090 上：同样达到 **数十至上百倍**

> 💡 说明重启的主要开销来自非必要的操作（如进程重建、CUDA 初始化、NCCL group 创建等），而非参数迁移本身。

---

### （2）内部优化效果（Figure 6）—— 重叠执行的有效性

ReMP 通过 **overlap model-shard reloading with KV cache migration** 显著缩短关键路径：

- 若顺序执行：$ T = T_{\text{model}} + T_{\text{KV}} $
- ReMP 并发执行：$ T \approx \max(T_{\text{model}}, T_{\text{KV}}) $

#### 效果：
- 对小模型：KV 迁移成为瓶颈，重叠节省约 **20–30%**
- 对大模型（如 Llama2-70B）：模型加载主导，但仍能有效隐藏 KV 传输时间

> ✅ 并发设计显著提升了整体切换效率。

---

### （3）服务性能表现（Figure 7 & 8）

在动态请求压力下，ReMP 动态选择最优 TP/PP 配置，相比固定配置表现出全面优势：

| 指标 | 表现 |
|------|------|
| **TTFT** | 显著低于 TP1PP8 和 TP2PP4，尤其在高负载时 |
| **TPOT** | 更稳定，无明显上升趋势 |
| **Output Throughput** | 在中高请求率下持续领先，最高可达 **+30% 以上增益** |

#### 关键发现：
- 固定配置难以兼顾高低负载场景：
  - `TP1PP8` 在高负载下因 pipeline latency 导致 TTFT 急剧上升；
  - `TP2PP4` 虽有所改善，仍不如动态选择灵活。
- ReMP 可根据不同负载自动选择更优配置（如低负载用更高 TP，高负载切至更高 PP），实现**自适应调度**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Model-parallel topology 可以且应该被动态管理**  
   当前静态配置严重制约了 LLM serving 的灵活性与效率。

2. ✅ **低停机时间重配置是可行的**  
   ReMP 实现了 **秒级切换**（1–7s），使 runtime adaptation 成为实用技术。

3. ✅ **KV Cache 的二维迁移机制至关重要**  
   保留已有计算状态大幅减少 prefilled token 的重复计算，是性能优势的关键来源。

4. ✅ **动态选择优于任何单一固定配置**  
   不同模型、不同负载下的最优 TP/PP 不同，ReMP 能实时探索并锁定最优解。

5. ✅ **系统级解耦是实现轻量切换的基础**  
   包括共享权重存储、预建 MPU 快照、Worker 复用机制等，共同支撑了低开销切换。

---

### 方法的局限性
1. **候选拓扑集合需预先定义**
   - MPU State Space 要求提前构建各拓扑对应的通信组快照；
   - 不支持任意拓扑的即时生成（需扩展动态 group cache）。

2. **KV Cache 容量不足时仍会丢弃部分缓存**
   - 若目标拓扑提供的 cache block 更少，scheduler 会抢占部分请求并回收 block；
   - 不能保证所有请求免于重计算。

3. **依赖充足的 CPU 内存**
   - 需将完整模型状态驻留在 CPU shared memory 中；
   - 对超大规模模型（如 >100B）可能存在内存压力。

4. **目前仅集成于 vLLM**
   - 尚未验证在其他推理框架（如 TensorRT-LLM、TGI）中的通用性。

---

### 未来工作方向
1. **支持任意拓扑的动态构建**
   - 引入 on-demand communication group 构造机制，摆脱对预定义拓扑的依赖。

2. **跨节点 reconfiguration**
   - 扩展至多机多卡场景，支持 scale-in/out 类型的拓扑变更。

3. **与 Auto-Scaling 结合**
   - 联合优化实例数量、TP/PP 配置、批处理大小等多维参数。

4. **引入预测机制**
   - 基于历史流量模式预测未来负载，提前触发 reconfiguration，进一步平滑性能波动。

5. **支持更多并行范式**
   - 如 ZeRO-based DP、expert parallelism（用于 MoE 模型）等。

---

> 🔚 **总结一句话**：  
> ReMP 成功地将 LLM 推理中的 model-parallel topology 从“静态部署约束”转变为“运行时可调资源”，为构建真正**自适应、高性能、低成本**的 LLM serving 系统提供了关键技术路径。

</details>

---

### 8. [GraphPO: Graph-based Policy Optimization for Reasoning Models](https://arxiv.org/abs/2606.18954)

**Authors**: Yuliang Zhan, Xinyu Tang, Jian Li, Dandan Zheng, Weilong Chai, Jingdong Chen, Jun Zhou, Ge Wu, Wenyue Tang, Hao Sun  
**Category**: cs.CL  
**Published**: 2026-06-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.18954v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has become a standard paradigm for enhancing the capability of large reasoning models. RLVR typically samples responses independently and optimizes the policy using from final answers. This paradigm has two limitations. First, independently respo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GraphPO: Graph-based Policy Optimization for Reasoning Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于**Reinforcement Learning with Verifiable Rewards (RLVR)** 的大推理模型（Large Reasoning Models, LRMs）存在两大挑战：
- **奖励稀疏性**：仅依赖最终答案的二元奖励（correct/incorrect），难以对中间推理步骤进行有效的信用分配（credit assignment）。
- **探索冗余性**：传统的链式（chain-based）或树式（tree-based）rollout 方法在采样时独立展开路径，导致大量语义相似的中间状态被重复探索，造成计算浪费。

尽管树形方法（如 TreeRL、PROS）通过共享前缀缓解了部分冗余，但仍无法合并语义等价但路径不同的中间状态，且分支独立扩展，限制了探索效率和优势估计的稳定性。

---

### **提出的新方法：GraphPO**
作者提出了 **GraphPO (Graph-based Policy Optimization)**，一种全新的基于图结构的强化学习框架，其核心思想是将推理过程建模为一个**有向无环图（Directed Acyclic Graph, DAG）**，其中：
- **节点（Nodes）**：表示从初始提示到当前步的**语义状态摘要**（semantic state summary）。
- **边（Edges）**：表示生成的单个推理步骤。
- **等价类（Equivalence Classes）**：通过嵌入相似度检测，将语义相近的状态“虚拟合并”为同一等价类。

---

### **相比现有方法的优势**
1. **减少冗余探索**  
   当不同路径到达语义相似的状态时，GraphPO 将其合并，避免重复扩展，节省计算资源，并将预算重新分配给新颖的前沿状态。

2. **提升信用分配质量（降低方差）**  
   - **共享后缀信号**：等价类内的路径可以共享后续的正确性样本，从而将稀疏的结果奖励转化为更密集的步骤级监督信号。
   - **双组图优势估计（Dual-Group Graph Advantage）**：
     - **Correctness Group**：比较从同一状态出发的不同推理步骤，提供细粒度的步骤优劣判断。
     - **Efficiency Group**：比较到达同一语义状态的不同路径长度，鼓励策略选择更短的有效路径，提升推理效率。

3. **理论保障**
   - 理论分析表明 GraphPO 能显著**降低优势估计的方差**。
   - 可以逼近理想化 PRM（Process Reward Model）的梯度方向，实现**无需标注的自涌现过程监督**（self-emergent process supervision）。
   - 鼓励策略收敛到**最短的正确推理路径**，提高推理效率。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **数学推理任务**：
  - `AIME24`, `AIME25`：高中数学竞赛题。
  - `MATH500`：标准数学推理基准。
- **综合推理任务**：
  - `GPQA`：研究生级别、抗谷歌搜索的问答挑战。
  - `LiveCodeBench`：代码生成与执行评测。
- **智能体搜索任务（Agentic Search）**：
  - `General AI Assistant`, `WebWalkerQA`, `BrowseComp`, `XBench`：涉及网页浏览、工具调用等复杂交互任务。

---

### **实验设置与评估指标**
- **模型基础**：在多个主流 LLM 上测试，包括：
  - `Qwen2.5-7B-Math`
  - `Qwen3-8B-Base`
  - `Deepseek-R1-Distill-Qwen-7B`
- **训练框架**：基于 Verl 和 vLLM 实现高效 rollout 与推理。
- **评估指标**：
  - **准确率（Accuracy %）**：主要性能指标。
  - **响应长度（Response Length）**：衡量推理效率。
  - **熵值（Entropy）**：反映策略多样性。
  - **探索效率（Exploration Efficiency）**：单位 token 成本下获得的正确推理进展。

---

### **基线方法对比**
| 方法 | 类型 | 简要说明 |
|------|------|----------|
| **GRPO**, **DAPO** | Chain-based | 基于独立轨迹的 RLVR 方法 |
| **TreeRL**, **SPO**, **TreePO** | Tree-based | 利用树结构共享前缀，改进信用分配 |
| **TREE-GRPO**, **PROS** | Tree-based | 强调前缀复用与组相对优势估计 |
| **GraphPO** | **Graph-based (本文方法)** | 首次引入图结构，合并语义等价状态 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**
在相同 token 预算下，GraphPO 在所有模型和任务上均**显著优于所有基线方法**：

#### **在 `Qwen3-8B-Base` 上的平均准确率对比**：
| 方法 | 平均准确率 |
|------|------------|
| Base | 38.4% |
| DAPO | 42.1% |
| PROS (best tree) | 45.5% |
| **GraphPO** | **48.9%** ✅ |

#### **在 `Deepseek-R1-Distill-Qwen-7B` 上的表现**：
| 方法 | 平均准确率 |
|------|------------|
| PROS | 60.8% |
| **GraphPO** | **64.0%** ✅ （+3.2% 绝对提升）

> 在 `MATH500` 上达到 **96.3%** 准确率，接近人类水平。

---

### **与基线方法的对比结果**
- **超越树形方法**：GraphPO 在所有任务上一致优于最优的 tree-based 方法（如 PROS、TREE-GRPO），证明图结构比树结构更能有效利用过程信号。
- **优于 PRM 方法**：即使不使用任何人工标注的步骤级标签，GraphPO 性能仍超过需要昂贵标注的 **ReasonFlux-PRM**（见 Figure 5a）。
- **更高的探索效率**：在相同 token 消耗下，GraphPO 发现更多独特语义状态，探索效率远高于 chain 和 tree 方法（见 Figure 2c）。

---

### **消融实验结果**
#### **(1) 合并阈值 $ \kappa $ 的影响（Figure 6）**
- $ \kappa = 0.92 $ 时性能最佳。
- $ \kappa \to 1.0 $（无合并）时退化为 tree 结构，性能下降。
- $ \kappa $ 过低会导致过度合并（false positives），损害性能。

#### **(2) 效率优势的作用（Figure S.1）**
- 移除效率优势后，响应长度先降后升，出现冗余推理。
- 加入效率优势可稳定压缩路径长度，提升推理效率。

#### **(3) 池化系数 $ w $ 的影响（Figure S.2）**
- $ w = 0.7 $ 时性能最优。
- $ w = 0 $ 时禁用后缀共享，退化为 tree 方法。
- 表明适度池化可在**方差降低**与**语义合并偏差**之间取得平衡。

---

## **4. 关键结论和发现**

### **主要发现**
1. **语义冗余普遍存在**：链式和树式 rollout 中大量中间状态高度相似，造成严重计算浪费。
2. **图结构能有效聚合等价状态**：通过合并语义相似节点，GraphPO 显著提升了 rollout 利用率和探索效率。
3. **自涌现过程监督可行**：无需额外标注，仅从结果奖励即可导出高质量的过程监督信号。
4. **双组优势机制有效**：
   - Correctness Group 提高信用分配精度。
   - Efficiency Group 显著缩短推理路径，提升效率。
5. **训练更稳定、更快收敛**：GraphPO 保持更高策略熵，避免过早收敛到局部模式。

---

### **方法的局限性**
- **依赖语义摘要质量**：节点嵌入依赖冻结的摘要模型（如 Qwen2.5-7B-Instruct）生成的 structured summary，若摘要不准会影响合并效果。
- **未验证多模态场景**：目前实验集中在文本推理与代理任务，尚未拓展至图像、音频等多模态推理。
- **实时开销略高**：需在线计算嵌入与相似度匹配，虽效率可控，但在极端低延迟场景可能受限。

---

### **未来工作方向**
- 扩展至 **multimodal reasoning**、**code generation with execution feedback** 等更复杂任务。
- 探索动态调整合并阈值 $ \kappa $ 与池化系数 $ w $ 的自适应机制。
- 结合 lookahead 或 MCTS 进一步优化图构建策略。
- 开源代码以促进社区复现与改进。

---

> ✅ **总结一句话**：  
> **GraphPO 通过将推理轨迹组织成语义图，首次实现了在无标注情况下从结果奖励中自动提取高效、低方差的过程监督，显著提升了大模型的推理能力与效率，在多个基准上刷新 SOTA。**

</details>

---

### 9. [HI-HCQC: A Tightly-Coupled Hardware Interface with High-Efficiency Communication for Hybrid Classical-Quantum Computing](https://arxiv.org/abs/2606.18642)

**Authors**: Shibo Liang, Junchao Wang, Zeyuan Wang, Feng Wang, Xiaoyu Li, Lei Li, FuDong Liu, Zheng Shan  
**Category**: cs.DC  
**Published**: 2026-06-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.18642v1  

#### Abstract
Hybrid classical-quantum computing requires frequent data exchange between classical processors and quantum control hardware. However, existing superconducting quantum control systems are commonly connected through loosely coupled interfaces such as Ethernet, resulting in high communication latency ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《HI-HCQC: A Tightly-Coupled Hardware Interface with High-Efficiency Communication for Hybrid Classical-Quantum Computing》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题  
当前超导量子计算系统中，**classical processors** 与 **quantum control hardware** 之间的通信主要依赖于松耦合接口（如 Ethernet），导致以下问题：
- **高通信延迟**（communication latency）
- **任务吞吐量受限**（limited task throughput）
- 难以满足 NISQ（Noisy Intermediate-Scale Quantum）时代对快速反馈、高频次校准和变分算法（如 VQE、QAOA）的需求。

此外，传统系统通常由多个商用设备拼接而成，存在成本高、体积大、系统复杂度高等问题。

---

### 提出了什么新方法或新思路  
本文提出 **HI-HCQC** —— 一种基于 **RFSoC**（Radio-Frequency System-on-Chip）的紧耦合硬件接口，用于实现高效的混合经典-量子计算（hybrid classical-quantum computing）。

#### 核心架构设计：
- 采用 **Xilinx Zynq Ultrascale+ RFSoC** 芯片，集成：
  - 高速 **RF-DACs**（8 GSPS）和 **RF-ADCs**（4 GSPS）
  - 可编程逻辑（PL）与嵌入式处理器（PS）
  - **PCIe Gen3 x8 接口**（带宽高达 8 GB/s）
  - 内部时钟同步电路（MTS 支持）
- 实现 **直接微波脉冲合成**（direct microwave pulse synthesis）与 **qubit readout**
- 支持 **DMA**（Direct Memory Access）机制，减少 CPU 干预

#### 创新点：
1. **紧耦合通信架构**：首次将 **PCIe 接口** 引入量子控制硬件，替代传统的 Ethernet 连接，显著降低通信延迟。
2. **高集成度与可扩展性**：单板支持 6 路 XY 控制通道 + 1 路复用读出通道，模块化设计便于扩展。
3. **软硬协同优化**：结合 PYNQ 框架，提供 Jupyter 接口，支持高级数据分析（如量子态重构、噪声分析等）。

---

### 相比现有方法的优势  
| 维度 | 传统系统（Ethernet-based） | HI-HCQC |
|------|--------------------------|--------|
| 通信接口 | Ethernet（~1 Gbps） | **PCIe Gen3 x8**（~8 GB/s） |
| 数据传输延迟 | ~2.2 秒 | **~13 ms**（提速 **169×**） |
| 任务吞吐量 | 低（0.4 gates/s） | **128.3 gates/s**（提升 **320×**） |
| 架构耦合程度 | 松耦合 | **紧耦合**（host-server 与 control-unit 物理集成） |
| 是否支持实时反馈 | 有限 | 支持快速反馈，适用于 QEC 和动态电路 |

> ✅ 所有对比系统（QICK、ICARUS-Q、Presto 等）均使用 Ethernet 接口，而 HI-HCQC 是唯一使用 PCIe 的方案（见 Table 1）。

---

## 2. 核心实验方法和设置

### 使用的硬件平台与环境
- **HI-HCQC 板卡**：基于 CC305 平台，搭载 XCZU47DR RFSoC 芯片
- **连接方式**：通过 PCIe 插入 classical server 主机
- **低温环境**：qubit 安置于稀释制冷机中（dilution refrigerator）
- **测试对象**：固定频率 transmon qubit（Q0903 和 Q0904）

---

### 实验设置与评估指标

#### 实验类型：
1. **单比特校准实验**：
   - Qubit spectroscopy（测共振频率）
   - Rabi oscillation（标定脉冲幅度）
   - T1 measurement（能量弛豫时间）
2. **读出性能测试**：
   - Single-shot readout（单次测量保真度）
3. **门保真度测试**：
   - Randomized Benchmarking（RB）→ 单比特门保真度
   - CZ-gate characterization → 两比特门保真度
4. **系统级性能测试**：
   - End-to-end latency 测试（H gate, CZ gate, GHZ circuit）
   - Task throughput 测试（在 500k shots 下执行效率）

#### 评估指标：
- **Latency**：从指令下发到结果返回的端到端延迟
- **Throughput**：单位时间内可执行的量子门数量
- **Fidelity**：single-qubit gate fidelity, CZ gate fidelity
- **Readout fidelity**：单次测量正确率
- **SFDR / Phase Noise**：信号质量指标

---

### 基线方法对比
- 对比对象为“传统量子控制系统”（未指明具体型号，但代表当前主流基于 Ethernet 的商用系统）
- 包括 BBN、Keysight、Zurich Instruments、Quantum Machines 等厂商产品（参见 Figure 3 和 Table 1）
- 所有基线系统均使用 Ethernet 接口，无 PCIe 加速支持

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3 和正文）

| 指标 | 传统系统 | HI-HCQC | 提速倍数 |
|------|--------|--------|---------|
| **H gate latency** | 3.9309 s | **0.0166 s** | **236.8×** |
| **CZ gate latency** | 4.3122 s | **0.0167 s** | **258.2×** |
| **GHZ circuit latency** | 4.2362 s | **0.0944 s** | **44.875×** |
| **Quantum gate throughput** | 0.4 gates/s | **128.3 gates/s** | **320.75×** |
| **Data transfer latency** | ~2.2 s | **~13 ms** | **169×** |

> ⚠️ 注意：这些延迟不包括物理门作用时间，而是涵盖 waveform generation、data transfer、playback、demodulation、upload 等全流程。

---

### 其他关键实验结果

#### 🔹 单比特性能
- **Single-shot readout fidelity**：
  - Q0903: **93.45%**
  - Q0904: **93.17%**
- **Randomized Benchmarking fidelity**：
  - Q0903: **99.99(1)%**
  - Q0904: **99.98(1)%**

#### 🔹 两比特门性能
- **CZ gate fidelity**：达到 **99%**（图 11）
- 成功完成 idle point、timing、distortion、amp/phi 参数联合标定流程

#### 🔹 信道性能（Table 2）
- **DAC 输出特性**：
  - 频段：4.2–5.5 GHz
  - 最小脉宽：30 μs
  - SFDR > -45 dBc
  - 相位噪声：< -97 dBc/Hz @ 1kHz, 4.8 GHz
- **ADC 输入特性**：
  - 频段：6.0–7.5 GHz
  - 最小分析脉宽：10 μs
  - 通道隔离度：> 80.2 dBc

#### 🔹 大规模 shot 测试
- 在 500,000 shots 下：
  - HI-HCQC 执行时间：**51 秒**
  - 传统仪器：**252 秒**
  - 效率提升 **4.94×**

---

### 消融实验结果（隐含分析）
虽然文中未明确进行“消融实验”，但通过以下对比揭示了各组件的重要性：
- **PCIe vs Ethernet**：数据传输延迟从 2.2 s 降至 13 ms → 表明 **高速接口是性能瓶颈突破的关键**
- **DMA 机制**：尽管理论传输时间仅需 305 μs，实际为 13 ms → 显示 **软件栈、驱动、内存拷贝仍是主要开销**
- **shot 数量影响**：
  - 当 shot 较少时，通信延迟主导总耗时
  - 当 shot 很多时（如 10k+），reset time（~400 μs per shot）成为瓶颈 → 说明 **通信优化收益递减**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **PCIe 接口可极大降低 hybrid system 中的通信延迟**，使 classical-quantum loop 更紧凑，适合需要快速反馈的应用（如 QEC、VQE）。
2. ✅ **RFSoC + PCIe 架构能显著提升任务吞吐量**（>320×），加速大规模芯片校准和算法迭代。
3. ✅ HI-HCQC 支持完整的 qubit 控制与读出功能，并已验证多种标准实验（Rabi、T1、RB、CZ 等），具备实用价值。
4. ✅ 系统支持分布式扩展，可用于百比特以上规模的量子处理器控制。

---

### 方法的局限性
1. ❗ **当前版本仅支持 6 路控制 + 1 路读出**，对于更大规模芯片仍需多板协同。
2. ❗ **PCIe 插槽数量限制了并行服务器扩展能力**，可能成为未来瓶颈。
3. ❗ 实际数据传输仍受 **driver、OS scheduling、memory copy** 影响，未能逼近理论极限（305 μs vs 13 ms）。
4. ❗ 尚未在真实量子纠错（QEC）循环中验证闭环控制能力。

---

### 未来工作方向
1. 🔄 **增强可扩展性**：支持更多 qubit 的控制与读出，发展多板同步技术。
2. 🧠 **引入更复杂的控制算法**：如 optimal control、machine learning-based calibration。
3. 🖥️ **提升易用性**：简化量子编程流程，进一步集成 Qiskit 等框架。
4. 🔬 **探索低温控制集成**：向 cryo-CMOS 或片上控制方向演进（类似 Gooseberry 架构）。
5. 📈 **优化软件栈**：减少 DMA 开销，提升数据通路效率，逼近 PCIe 理论带宽。

---

## 总结

> **HI-HCQC 是首个将 PCIe 高速接口深度整合至 RFSoC 量子控制硬件的工作，实现了 classical-quantum computing 的“紧耦合”范式转变。其实验结果证明，在 latency 和 throughput 上相较传统 Ethernet 系统取得数量级提升，为构建高效、可扩展的 hybrid quantum-classical computing infrastructure 提供了切实可行的技术路径。**

</details>

---

### 10. [Compressed-Resident Genomics: Full-Pipeline Device-Resident GPU LZ77 Decode with Position-Invariant Random Access](https://arxiv.org/abs/2606.18900)

**Authors**: Yakiv Shavidze  
**Category**: cs.DC  
**Published**: 2026-06-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.18900v1  

#### Abstract
Genomic archives grow faster than decompression keeps up: the European Nucleotide Archive holds tens of petabytes of fastq.gz, and gzip is fundamentally sequential. GPU decompressors (nvCOMP DEFLATE at ~50GB/s on A100) decode whole files with no random access; CPU genomic tools (CRAM, samtools) supp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Compressed-Resident Genomics: Full-Pipeline Device-Resident GPU LZ77 Decode with Position-Invariant Random Access

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基因组学数据（如 FASTQ）以 `gzip` 压缩格式（如 `.fastq.gz`）大规模存储于欧洲核苷酸数据库等平台，总量已达数十 PB。然而，**gzip 的 DEFLATE 解码本质上是串行的**，无法支持高效随机访问；同时，现有 GPU 加速解压工具（如 nvCOMP）虽吞吐高（~50 GB/s），但**不支持区域寻址（random access）**，而 CPU 工具（如 samtools faidx）虽支持区域查询，却受限于主机内存带宽，难以满足 GPU 分析流水线对低延迟、高吞吐的需求。

这一矛盾导致“压缩即瓶颈”——海量数据无法在不解压全文件的前提下被快速访问，严重制约了 GPU-resident 分析流程的发展。

### 提出的新方法与创新思路
本文基于作者先前提出的 **ACEAPEX** 编解码器（一种并行 LZ77 格式），实现了三项关键扩展：

#### ✅ 贡献一：全设备驻留 GPU 解码流水线（Full Device-Resident Pipeline）
- 将熵解码（entropy decode）和匹配解析（match resolution）**全部迁移至 GPU 上执行**，实现端到端的 device-resident 解码。
- 避免 PCIe 数据往返（host round-trip），显著提升整体效率。

#### ✅ 贡献二：位置无关的随机访问能力（Position-Invariant Random Access）
- 利用 ACEAPEX 在编码时将所有 LZ77 回指偏移量转换为**绝对输出位置（absolute offset）** 的特性，使每个固定大小 block 可独立解码。
- 构建轻量级 **read-to-block index**（仅 8 字节/读段），实现 read-level 的快速定位与解码。
- 支持任意 block 区域的直接解码（range decode），无需加载整个文件。

#### ✅ 贡献三：解耦输出大小与显存限制的范围解码策略（Range-Decoding Strategy）
- 设计 **v7-RA range decoder**，按需分块处理压缩流，避免将整个解压后数据载入 VRAM。
- 即使面对 50 GB 的基因组数据，也能在有限 VRAM 下维持高性能解码。

### 相比现有方法的优势
| 方法 | 是否 GPU 加速 | 是否支持随机访问 | 是否 device-resident | 典型吞吐 |
|------|----------------|--------------------|------------------------|----------|
| gzip + samtools faidx | ❌（CPU） | ✅（chr:pos） | ❌ | ~2.3 ms/read |
| nvCOMP DEFLATE | ✅（~50 GB/s） | ❌ | ⚠️（需 D2H） | ~50 GB/s |
| CRAM/BAM indexing | ✅（CPU） | ✅ | ❌ | 主机带宽限制 |
| **ACEAPEX（本工作）** | ✅ | ✅（read-level） | ✅ | **最高 260 GB/s，单 read 解码 0.362 ms** |

> ✅ **首次实现“压缩即服务”范式**：整个基因组可长期以压缩态驻留 VRAM，按需秒级解码任意区域，突破传统 D2H 带宽天花板（~39 GB/s）。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **FASTQ NA12878**：1 GB，Illumina Platinum 数据，PCR-free，高质量。
- **FASTQ ERR194147**：5 GB 和 50 GB，含噪声质量字符串，更具现实代表性。
- **enwik9**：1 GB 文本数据，用于通用压缩性能对比。
- **silesia corpus**：标准压缩测试集。

### 实验设置
- **硬件平台**：
  - GPU：NVIDIA H100 80 GB HBM3（SXM 接口，理论带宽 ~3.35 TB/s）
  - CPU：Intel Xeon Platinum 8468（160 线程）
  - 内存：1.5 TiB RAM
  - 存储：60 GB 临时磁盘 + 网络存储卷
- **软件环境**：
  - CUDA 12.8，驱动 570.195.03
  - nvcomp 5.2.0.13（用于加速熵解码）
  - DietGPU（Meta 开源 ANS 实现）
  - 正确性验证：bit-perfect（XXH3-64/FNV 校验）

### 评估指标
- **解码吞吐量（GB/s）**
- **随机访问延迟（ms per read）**
- **索引体积（Index size）**
- **VRAM 占用与可扩展性**
- **压缩比（Compression Ratio）**
- **端到端延迟（含 PCIe 传输）**

### 基线方法对比
- **samtools faidx**：主流 CPU 级随机访问工具，cold/warm 性能分别测量。
- **nvCOMP DEFLATE**：代表当前最快 GPU 解压方案（无随机访问）。
- **zstd, xz, Gompresso, Rapidgzip, Recoil** 等作为文献对照。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔹 全设备驻留解码吞吐（Mode 2: nvcomp-accelerated）
| 数据集 | 大小 | 吞吐（GB/s） | 压缩比 |
|--------|------|-------------|--------|
| FASTQ NA12878 | 1 GB | **260.0** | 11.19 |
| FASTQ ERR194147 | 5 GB | 168.9 | 3.31 |
| FASTQ ERR194147 | 50 GB | **165.7\*** | 3.99 |

> \* 使用 range-decode 策略，避免 OOM。

- 熵解码阶段达 **~480 GB/s**，匹配解析阶段 **~203 GB/s**。
- 整体吞吐远超 nvCOMP DEFLATE（~50 GB/s on A100）。

#### 🔹 随机访问性能（Table 3）
| 操作 | 时间 | 说明 |
|------|------|------|
| 完整解码（5 GB） | 29.71 ms | 168 GB/s 基线 |
| 单 block（16 KB）seek | **0.365 ms** | 比完整解码快 **81×** |
| 100 block（1.6 MB）seek | 0.394 ms | 几乎无额外开销 |

> ✅ **延迟主要由 kernel launch overhead 主导（~270 μs）**，因此在一定范围内与请求大小无关。

#### 🔹 Read-Level Index 对比
| 指标 | ACEAPEX | samtools (.fai) |
|------|---------|------------------|
| 索引大小（1.34 GB FASTQ） | **40 MB** | 250 MB |
| 相对大小 | **6.3× 更小** | — |
| Warm 查找延迟 | ~0.3 μs | ~2.3 ms |
| End-to-end（查找+解码） | **0.362 ms** | ~2.3 ms |
| 速度优势 | **~6× 快于 warm samtools** | — |

#### 🔹 端到端 PCIe 性能（诚实评估）
- **Device-resident decode**: 8.03 ms (**166.9 GB/s**)
- **H2D staging**: 9.81 ms (39.5 GB/s)
- **D2H transfer**: 33.38 ms (39.2 GB/s)
- **Total end-to-end**: ~51 ms → **平均 ~26 GB/s**

> ⚠️ D2H 成为最大瓶颈，凸显 **compressed-resident computing 的必要性**。

#### 🔹 DietGPU 替代熵解码器潜力
- DietGPU ANS on H100:
  - **Decode: 592.5 GB/s**
  - Encode: 364.9 GB/s
- 显著高于当前使用的 nvcomp-ANS（480 GB/s）
- 表明：**完全开源的高性能解码栈可行**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **ACEAPEX 可实现高达 260 GB/s 的全 GPU 解码吞吐**，首次完成从熵解码到匹配解析的全流程 device-resident 实现。
2. ✅ **位置无关的 block 结构支持极低延迟随机访问**（<0.4 ms/read），且索引体积仅为 .fai 的 1/6.3。
3. ✅ **range-decode 架构成功解耦解压输出大小与 VRAM 容量**，可在 80 GB VRAM 中处理 50 GB 基因组数据，维持 165.7 GB/s 吞吐。
4. ✅ **压缩态驻留 VRAM 是突破 PCIe 瓶颈的关键路径**：任何返回主机的结果都受限于 ~39 GB/s 的 D2H 带宽。
5. ✅ **Meta 的 DietGPU 展示了替代闭源 nvcomp 的可能性**，为构建全开源高性能解压栈铺平道路。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **Seek 粒度为 read-level，非 chr:pos** | 当前适用于 raw FASTQ，BAM/SAM 的染色体坐标查询需后续开发 |
| **依赖 nvcomp 的高性能模式为闭源** | 高吞吐路径（Mode 2）目前依赖 proprietary nvcomp 库 |
| **编码速度较慢** | 50 GB 数据编码耗时约 147 秒（~340 MB/s），适合“一次编码，多次解码”场景 |
| **压缩比非最优** | zstd-19 比 ACEAPEX 密集 1.2–1.55×，但牺牲了解码速度与随机访问能力 |
| **Kernel Launch Overhead 显著** | 小规模访问延迟受启动开销主导，难以进一步降低至亚微秒级 |

### 未来工作方向
1. **集成 DietGPU 替代 nvcomp**，打造完全开源、高性能的 device-resident 解码栈。
2. 扩展至 **BAM/CRAM 级别的 chr:pos 随机访问支持**，对接主流基因组分析工具链。
3. 探索更细粒度 block（<16 KB）的优化空间，或引入 hierarchical indexing 提升定位精度。
4. 支持更多 transform（如 quality string delta）而不破坏 LZ77 匹配效率。
5. 推动 ACEAPEX 成为标准基因组压缩格式之一，纳入主流工具生态。

---

> 📌 **最终愿景**：  
> **Compressed-Resident Genomics** —— 整个基因组以压缩形式常驻 GPU 显存，分析任务按需即时解码局部区域，彻底摆脱 I/O 与传输瓶颈，开启下一代超高通量生物信息计算时代。

> 🔗 代码已开源（MIT License），DOI: [10.5281/zenodo.20729380](https://doi.org/10.5281/zenodo.20729380)

</details>

---

### 11. [Be Your Own Teacher: Steering Protein Language Models via Unsupervised Reward Optimization](https://arxiv.org/abs/2606.18961)

**Authors**: Lanqing Li, Shentong Mo, Yang Yu, Pheng-Ann Heng  
**Category**: cs.LG  
**Published**: 2026-06-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.18961v1  

#### Abstract
Protein language models (PLMs) have emerged as powerful tools for controllable biomolecular design, yet their post-training adaptation typically relies on costly wet-lab validation or curated preference datasets. To overcome this supervision bottleneck, we introduce unsupervised reward optimization ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Be Your Own Teacher: Steering Protein Language Models via Unsupervised Reward Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
蛋白质语言模型（**PLMs**）在可控生物分子设计中展现出巨大潜力，但其后训练（post-training）通常依赖于昂贵的湿实验验证（wet-lab validation）或人工标注的偏好数据集（preference datasets），这构成了严重的**监督瓶颈**（supervision bottleneck）。尤其在缺乏真实标签（ground-truth labels）或实验反馈的场景下，如何有效提升PLMs的可控生成能力成为一个关键挑战。

### **提出的新方法与新思路**
本文提出了**无监督奖励优化**（**Unsupervised Reward Optimization, URO**）框架，首次实现无需真实标签即可对PLMs进行可学习的后训练，使其具备自我改进能力。核心思想是：
- 设计**任务无关的代理奖励函数**（task-agnostic proxy rewards），结合**内在不确定性**（intrinsic uncertainty）与**外在语义一致性**（extrinsic semantic consistency）来衡量生成序列与提示（prompt）的对齐程度。
- 提出两种离线优化算法：
  - **Soft Reward Optimization (SRO)**：基于连续奖励信号最大化经典RLHF目标。
  - **Binarized Reward Optimization (BRO)**：基于二值化奖励信号，将正负样本分类建模为零和博弈。

### **相比现有方法的优势**
- **无需真实标签**：摆脱对人类标注或实验反馈的依赖，适用于标签稀缺或高风险领域（如生物医药）。
- **通用性强**：奖励函数不依赖具体任务，可在多种蛋白家族和生成任务上迁移。
- **性能接近Oracle**：在多个采样温度、模型规模和蛋白家族上，SRO/BRO显著优于主流基线（如DPO、KTO），甚至在某些条件下逼近使用真实标签训练的Oracle模型。
- **可扩展性好**：支持大规模离线训练，为构建**通用生物智能**（Generalist Biological Artificial Intelligence, GBAI）提供可行路径。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ProGen2-small-mix7 和 ProGen2-medium-mix7**：基于7个Pfam蛋白家族（如GPCRs、Globins等）微调的自回归PLMs（151M 和 764M 参数）。
- **ESM3**：多模态PLM（1.4B参数），支持序列、结构、功能联合推理。
- **Pfam700**：构造的**组合型OOD提示集**（compositional out-of-distribution prompts），通过拼接一个家族的功能标签与另一家族的前10个残基，测试模型的泛化能力。
- **DRAME-Func2Seq / Struct2Seq**：从AlphaFoldDB中提取的 *Dracunculus medinensis* 蛋白子集，用于功能到序列（Func2Seq）和结构到序列（Struct2Seq）任务。

### **实验设置与评估指标**
- **采样策略**：在5个温度（T ∈ {0.3, 0.5, 0.7, 1.0, 1.5}）下，每个prompt生成64条序列，构建离线训练集（共约448k样本）。
- **奖励函数设计**：
  - 内在奖励：基于`Tref`的**预测熵**（predictive entropy）和**归一化熵**（normalized entropy）。
  - 外在奖励：使用**ESMC-600M**模型计算生成序列间的**语义距离**（如L1-mean、PCA1-bos）。
  - 最终采用分段奖励函数 $ r_T $，根据温度选择最优指标。
- **评估任务**：
  - **Keyword Recovery Rate**（pass@1）：使用InterProScan检测生成序列是否恢复提示中的功能关键词。
  - **Pass@k 分析**：评估top-k解码下的成功概率。
- **训练细节**：
  - 所有方法训练1个epoch，AdamW优化器，学习率5e-7。
  - 仅保留每prompt的top-4和bottom-4样本以降低噪声。

### **基线方法对比**
- **DPO**（Direct Preference Optimization）
- **KTO**（Kahneman-Tversky Optimization）
- **Oracle**：使用真实标签（binary label）训练的BRO模型（作为性能上限）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**
在 **Pfam700 Func2Seq** 任务上的平均keyword recovery rate（pass@1, k=15）：

| 方法 | 151M 模型 | 764M 模型 |
|------|-----------|-----------|
| Base (ProGen2-small/medium) | 0.281 | 0.385 |
| DPO | 0.350 | 0.480 |
| KTO | 0.337 | 0.505 |
| **BRO (Ours)** | **0.433** | **0.613** |
| **SRO (Ours)** | **0.427** | **0.602** |
| Oracle (有监督) | 0.488 | 0.603 |

> ✅ **BRO 在764M模型上达到61.3%，超越Oracle（60.3%）！**

### **与基线方法的对比结果**
- **SRO/BRO 显著优于 DPO/KTO**：
  - 在所有温度和模型尺度下，SRO/BRO平均提升超过 **8–10个百分点**。
  - 尤其在中低温（T ≤ 0.7）时优势更明显。
- **接近甚至超越Oracle**：
  - 在764M模型上，BRO性能略高于Oracle，表明无监督奖励已足够有效。
- **跨家族泛化能力强**（见Table 10）：
  - 在全部7个Pfam家族中，BRO/SRO均取得最佳或次佳表现。

### **消融实验结果**
#### （1）不同奖励函数的影响（Table 2）
| 方法 | 平均性能 |
|------|---------|
| BRO + L1-mean | **0.433** |
| BRO + $ r_T $（分段奖励） | 0.386 |
| BRO + Entropy | 0.341 |
| SRO + $ r_T $ | **0.427** |
| SRO + Entropy | 0.364 |

> 🔍 **L1-mean 是最鲁棒的外在奖励指标**，即使单独使用也优于其他组合。

#### （2）多温度 vs 单温度采样（Table 3）
| 方法 | 平均性能 |
|------|---------|
| BRO (multi-T, L1-mean) | **0.433** |
| BRO (single-T=0.7, L1-mean) | 0.366 |
| BRO (single-T=1.0, entropy) | 0.311 |

> 📈 多温度采样显著提升性能，验证了多样性训练数据的重要性。

#### （3）超参数敏感性（Table 11）
SRO在不同 $ \beta $ 值下表现稳定，且始终优于加权DPO（wDPO），说明其对batch效应不敏感。

---

## **4. 关键结论和发现**

### **主要发现**
1. **任务无关奖励具有强相关性**：
   - 结合内在不确定性（熵）与外在语义一致性（ESMC嵌入距离）的代理奖励，能与PLM的可控性（controllability）高度相关，无需任何真实标签。
2. **存在“临界温度”现象**：
   - 模型在某一温度（如T≈0.7 for ProGen2, T≈1.0 for ESM3）从“过度自信”转向“信心不足”，此时为性能峰值，但内在奖励接近随机，需依赖外在奖励。
3. **SRO/BRO 实现高效自我改进**：
   - 仅通过自身生成经验即可显著提升指令跟随能力，在OOD任务上表现优异。
4. **L1-mean 是最优外在奖励指标**：
   - 在多个设置下表现最稳定，且计算效率高（远快于pLDDT/pTM结构预测）。

### **方法的局限性**
- **Struct2Seq任务效果有限**：由于部分结构天然难以折叠，导致正样本稀疏，影响自我改进。
- **仍低于结构感知指标**：尽管高效，L1-mean等语义距离在Struct2Seq任务上不如pLDDT/pTM准确。
- **尚未进行湿实验验证**：目前仅为**in silico**评估，实际生物学功能有待实验确认。
- **潜在双用途风险**：增强PLM可控性可能被滥用用于设计有害蛋白（如毒素、免疫逃逸蛋白）。

### **未来工作方向**
- 探索更复杂的奖励设计（如不确定性感知优化）。
- 引入少量人类反馈的混合范式（hybrid supervised-unsupervised）。
- 开发面向特定应用的安全机制（如biosecurity screening、access control）。
- 将框架推广至RNA、抗体、代谢通路等其他生物分子设计任务。

---

> 💡 **一句话总结**：  
> 本论文开创性地实现了**无监督条件下的PLM自我教学**，通过设计任务无关的混合奖励函数与SRO/BRO算法，使模型能在没有真实标签的情况下持续提升可控生成能力，为低成本、高安全性的下一代生物智能设计提供了坚实基础。

</details>

---

### 12. [MCompassRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieval](https://arxiv.org/abs/2606.18508)

**Authors**: Amirhossein Abaskohi, Raymond Li, Gaetano Cimino, Peter West, Giuseppe Carenini, Issam H. Laradji  
**Category**: cs.CL  
**Published**: 2026-06-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.18508v1  

#### Abstract
Retrieval-augmented generation (RAG) systems depend critically on how documents are chunked and searched. Fine-grained chunks can improve retrieval precision but expand the search space, increasing latency and cost; larger chunks reduce the number of candidates but make dense similarity less reliabl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MCOMPASSRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieval

## 1. 论文的主要贡献和创新点

### 解决的问题
**Retrieval-Augmented Generation (RAG)** 系统在文档分块（chunking）策略上面临一个根本性的权衡（granularity trade-off）：
- **细粒度分块**（如句子级）能提供精确证据，但显著扩大检索空间，增加延迟和成本。
- **粗粒度分块**（如段落级）可减少候选数量、提高效率，但其嵌入表示会混合多个主题，导致语义噪声，使基于余弦相似度的密集检索变得不可靠。

这一问题在需要跨大规模异构语料库进行快速且精准检索的**深度研究任务**（deep research tasks）中尤为突出。

### 提出的新方法与创新
为解决上述问题，本文提出了 **MCoMPASSRAG**，一种元数据引导的检索框架，其核心创新在于：

- **使用主题元数据作为“语义罗盘”**：通过将**主题级信号**（topic-level signals）注入到粗粒度分块的表示中，使其在保持高效的同时更易于被精确检索。
- **两阶段元数据利用机制**：
  1. **元数据选择（Metadata Selection）**：从一个离线构建的**语料库级元数据银行**（metadata bank）中，根据查询选择最相关的主题分布。
  2. **元数据抽象（Metadata Abstraction）**：将选中的多个主题分布进行聚合和去噪，生成一个紧凑的、主题感知的**查询侧主题向量**（query-side topic vector）。
- **轻量级学生检索器训练**：采用**LLM教师蒸馏**（LLM-teacher distillation）技术，训练一个轻量的MLP分类器作为学生模型。该学生模型学习如何利用元数据增强的表示来识别相关分块，而**推理时无需调用LLM**，保证了高效率。

### 相比现有方法的优势
- **效率与质量兼得**：在不增加检索搜索空间的前提下，提升了粗粒度分块的检索精度，打破了传统效率-质量的权衡。
- **无推理时LLM开销**：与依赖LLM进行重排序（reranking）、查询扩展或迭代检索的方法不同，MCoMPASSRAG的推理过程完全由轻量级模型完成，延迟极低。
- **正交于其他技术**：该框架可以与查询扩展、迭代检索、上下文压缩等其他RAG优化技术结合使用。

## 2. 核心实验方法和设置

### 使用的数据集
在**六个复杂的检索基准**（benchmarks）上进行了评估：
- **SCI-DOCS**: 科学文献摘要，多主题。
- **LegalBench-RAG**: 法律领域，要求精确的片段检索。
- **Dragonball**: 多语言（中英）、多领域（金融、法律、医疗），设计用于复杂场景。
- **HotpotQA**: 开放域多跳问答，需要跨文档推理。
- **SQuAD**: 开放域单跳问答。
- **DRBench**: 企业级深度研究任务，结合公有网络和私有知识库。

此外，还使用 **LongBenchV2** 进行下游生成性能评估。

### 实验设置和评估指标
- **检索质量指标**：
  - **Information Efficiency (IE)**: `IE@k = Precision@k × Recall@k`，在 `k ∈ {1, 3, 5}` 上取平均值。这是主要的综合性能指标。
  - Precision 和 Recall。
- **下游生成性能指标**：
  - Accuracy, F1, ROUGE-L, METEOR, BERTScore。
- **效率指标**：
  - **端到端延迟**（Latency in ms）
  - **每查询检索的Token数**（Tok/Q）
- **训练**：使用合成数据（Synthetic training data）通过LLM教师（GPT-4o）进行监督，对学生模型进行蒸馏训练。

### 基线方法对比
与多种类型的RAG基线进行了比较：
- **密集检索**：`DenseXRetrieval`
- **结构化/层次化检索**：`RAPTOR`, `Meta-Chunking` (PPL/MSP)
- **LLM-based RAG**：`SAKI-RAG`, `ReflectiveRAG`, `DF-RAG`, `REFRAG`, `PageIndex`, `A-RAG`, `Chroma Context-1`
- **长上下文基线**：直接使用长上下文LLM（`QWEN3-32B`）

## 3. 主要实验结果和性能指标

### 关键性能数据
- **平均信息效率提升**：MCoMPASSRAG 在六个检索基准上，相比最强的非LLM基线，**平均IE提升了8.24%**。
- **显著降低延迟**：相比强大的LLM-based RAG基线，其运行速度**超过5倍以上**。
- **接近LLM上限**：其性能非常接近需要在检索时调用LLM的“oracle”上限（`LLM+10 Topics`），例如在SCI-DOCS上的IE差距不足1个百分点。

### 与基线方法的对比结果
- **全面超越**：在所有六个基准的所有指标上，MCoMPASSRAG均优于所有非LLM基线。
- **在困难任务上优势明显**：在多跳、复杂的基准（如DRBench, LegalBench-RAG）上，其优势最为显著。
- **高效的下游性能**：在下游生成任务中，MCoMPASSRAG以远低于其他高效RAG方法（如SAKI-RAG, REFRAG）的成本（4,126 Tok/Q vs. >5,500 Tok/Q）和极低的延迟（174ms vs. >700ms），实现了极具竞争力的生成质量。
- **成本效益高**：相比长上下文方法（如PageIndex），其使用的Token数超过10倍，但生成性能仍在合理范围内，证明了其高性价比。

### 消融实验结果
- **元数据选择与抽象模块至关重要**：
  - 移除任一模块都会导致IE下降。
  - 同时移除两者会导致性能大幅退化，证明了这两个组件的互补作用。
- **训练数据泛化能力强**：
  - 即使在目标领域没有标注数据的情况下，使用通用数据集（如MSMarco）训练的MCoMPASSRAG仍能显著超越非LLM基线，表明其学习到了可迁移的检索行为。
- **主题数量存在最优范围**：
  - 性能随传递给模型的主题数量先升后降，在约12-15个主题时达到峰值。过多的主题会引入噪声。
- **对嵌入模型和主题模型具有鲁棒性**：
  - 在不同的嵌入模型（如QWEN3-EMBEDDING系列, BGE-M3）和主题模型（如ETM, CWTM, CEMTM）上，MCoMPASSRAG均能取得良好效果，其中`CEMTM`表现最佳。

## 4. 关键结论和发现

### 主要发现
1. **主题元数据是有效的“语义罗盘”**：将主题级信号作为元数据，可以有效地引导检索系统关注查询相关的语义方向，从而在粗粒度分块上实现精细的证据定位。
2. **轻量级蒸馏可行**：通过LLM教师蒸馏，可以成功地将复杂的检索决策能力迁移到一个无需推理时LLM调用的轻量级学生模型上。
3. **效率-质量权衡被打破**：MCoMPASSRAG成功地在保持甚至超越现有高效RAG方法效率的同时，达到了接近LLM-based方法的检索质量。

### 方法的局限性
1. **依赖主题模型质量**：检索性能直接受底层主题模型质量的影响。在低资源或专业领域，若主题模型训练不佳，则元数据信号会失效。
2. **超参数敏感**：性能对主题数量（K）、选择的元数据条目数（L）、用于检索的主题数（M）等超参数较为敏感，需要仔细调整。
3. **信息聚合有损**：当前方法将多个主题向量加权求和成一个单一的聚合向量，这种操作会丢失每个主题信号的独立结构，是一种有损压缩。

### 未来工作方向
1. **端到端联合优化**：探索主题模型和检索器的联合训练，以更好地对齐主题表示。
2. **可扩展的选择策略**：开发近似选择算法，以应对超大语料库的可扩展性挑战。
3. **集成到迭代代理**：将MCoMPASSRAG集成到迭代式的深度研究代理（deep research agents）中，其效率优势将在多次检索循环中累积放大。
4. **改进的信息整合方式**：探索稀疏整合或交叉注意力（cross-attention）等机制，以更有效地保留每个主题信号的结构。

</details>

---

### 13. [Splaxel: Efficient Distributed Training of 3D Gaussian Splatting for Large-scale Scene Reconstruction via Pixel-level Communication](https://arxiv.org/abs/2606.18588)

**Authors**: Wenqi Jia, Zhewen Hu, Ying Huang, Yu Gong, Stavros Kalafatis, Yuke Wang, Wei Niu, Chengming Zhang, Ang Li, Sheng Di, Yuede Ji, Bo Fang, Miao Yin  
**Category**: cs.DC  
**Published**: 2026-06-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.18588v1  

#### Abstract
3D Gaussian Splatting (3DGS) enables high-fidelity and real-time 3D scene reconstruction, but scaling training to large-scale scenes requires optimizing hundreds of millions of Gaussians across multiple GPUs. Existing distributed approaches either partition scenes into isolated regions, causing glob...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Splaxel: Efficient Distributed Training of 3D Gaussian Splatting for Large-scale Scene Reconstruction via Pixel-level Communication*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **3D Gaussian Splatting (3DGS)** 分布式训练框架面临严重的**通信瓶颈**，尤其是在大规模场景中：
- **Gaussian-level communication**（如 Grendel）需要在 GPU 之间频繁交换数百万甚至上亿个 Gaussians，导致通信开销随场景规模和 GPU 数量急剧增长，最终主导训练迭代时间。
- **分区隔离方法**（如 DOGS）虽然避免了通信，但牺牲了全局一致性，导致重建结果出现明显的视觉不连续。

### 🚀 提出的新方法与核心思想
论文提出了 **Splaxel**，一种基于 **pixel-level communication** 的高效分布式 3DGS 训练框架，其核心创新包括：

#### （1）**像素级通信范式（Pixel-level Communication）**
- 每个 GPU 在本地渲染其高斯子集，仅交换**部分像素值**（颜色、透射率、深度），而非原始 Gaussians。
- 通信量仅取决于图像分辨率，与高斯数量无关，实现了**恒定通信成本**，从根本上缓解了通信瓶颈。

#### （2）**凸划分保证全局排序一致性（Convex Partitioning）**
- 采用 **axis-aligned bounding box (AABB)** 进行空间划分，确保每个相机射线最多与每个分区相交一次。
- 从而保证局部渲染后的像素在全局合成时仍保持正确的深度顺序，避免因非凸划分导致的混合顺序错误。

#### （3）**像素级冗余消除技术**
- **几何不可见性预测**：通过投影视锥体与分区的交集顶点，提前确定可见像素区域，避免传输无效像素。
- **透射饱和检测**：运行时检测像素是否已被前方高斯完全遮挡（透射率接近零），动态跳过后续 GPU 的计算与通信。

#### （4）**无冲突视角合并调度（Conflict-free View Consolidation）**
- 利用不同视角访问的 GPU 集合互不重叠的特点，将多个无依赖的视角合并到同一训练周期中并行处理。
- 显著提升 GPU 利用率，减少空闲等待时间。

---

### 🔍 相比现有方法的优势
| 维度 | Grendel (Baseline) | Splaxel (本文) |
|------|---------------------|---------------|
| **通信模式** | Gaussian-level 交换 | Pixel-level 交换 |
| **通信复杂度** | 随高斯数和 GPU 数线性/超线性增长 | 固定（仅与分辨率相关） |
| **全局一致性** | ✅ 保持 | ✅ 保持 |
| **通信占比** | >70%（大场景下） | ~10–20% |
| **GPU 利用率** | 低（存在大量空闲） | 高（通过视角合并优化） |

---

## 2. 核心实验方法和设置

### 📚 数据集
使用 **MatrixCity** 基准中的四个大规模城市级数据集：
| 数据集 | 高斯数量 | 场景类型 | 图像数量 | 区域面积 |
|--------|----------|----------|-----------|------------|
| Small City Street | 55M | 街景 | 27.5K 训练图 | 27 km² |
| Small City Aerial | 37M | 航拍 | 5.6K 训练图 | 2.7 km² |
| Big City Street | 120M | 街景 | 206K 训练图 | 25.3 km² |
| Big City Aerial | 120M | 航拍 | 51.6K 训练图 | 25.3 km² |

> 所有实验均在 **1080p 分辨率** 下进行。

### ⚙️ 实验平台
- **Platform 1**: 8 × NVIDIA RTX 6000 Ada (48GB)
- **Platform 2**: 8 × NVIDIA RTX PRO 6000 Blackwell (96GB)

### 📊 评估指标
- **训练效率**：每迭代耗时（ms）、吞吐量（views/sec）
- **重建质量**：PSNR（Peak Signal-to-Noise Ratio）
- **加速比**：相对于基线的速度提升倍数
- **消融分析**：各组件对性能的影响

### 🆚 基线方法
- **Grendel [75]**：当前最先进的分布式 3DGS 框架，采用 Gaussian-level all-to-all 通信。
- 其他方法（如 DOGS）因破坏全局一致性未作为主比较对象。

---

## 3. 主要实验结果和性能指标

### 📈 总体性能对比（Table 1 & Figure 17）

| 方法 | 数据集 | #Gaussians | PSNR ↑ | 训练时间 ↓ | 加速比 |
|------|--------|-------------|--------|------------|--------|
| Grendel | Big City Street | 120M | 23.2 | 48.8h | — |
| **Splaxel** | Big City Street | 120M | **23.2** | **6.4h** | **7.6×** |
| Grendel | Big City Aerial | 120M | 30.4 | 3.6h | — |
| **Splaxel** | Big City Aerial | 120M | **31.2** | **1.2h** | **3.0×** |

> ✅ 在保持相同甚至更高 PSNR 的前提下，Splaxel 实现了 **最高达 7.6× 的端到端训练加速**。

### 📉 每迭代时间分解（Figure 20）
- 在 Grendel 中，**通信占迭代时间的 73–80%**。
- 在 Splaxel 中，**通信降至约 10–20%**，成为次要开销。
- **反向传播（backpropagation）** 成为主要计算瓶颈，表明系统已回归“计算受限”状态，说明通信瓶颈被成功打破。

### 📊 可扩展性测试（Figure 19）
- 随着 GPU 数量从 1 增加到 8：
  - Grendel 几乎无法扩展（吞吐量增长缓慢）。
  - Splaxel 吞吐量显著提升，在 Big City Aerial 上达到 **142 images/s**（vs. Grendel 的 52 images/s）。

### 🔬 消融实验（Figure 22）
在 Small City Street 上逐步添加组件的效果：
| 配置 | 每迭代时间 | 相对于纯通信方案加速 |
|------|------------|------------------|
| Pixel-level Communication (C) | ~100ms | 1.0× |
| + Redundancy Reduction (C+R) | ~45ms | 2.2× |
| + View Consolidation (C+R+S) | ~30ms | **3.3×** |

> 每个模块都带来显著性能增益，其中**视角合并**进一步提升了并行利用率。

### 🧹 冗余消除效果（Figure 21）
- **空间冗余消除后**：零值像素比例从平均 60% 降至 <10%。
- **饱和冗余消除后**：饱和像素比例下降 34–40%，有效减少了不必要的通信与计算。

### 💡 GPU 利用率提升（Figure 23）
- 原始策略下 GPU 利用率不足 50%。
- Splaxel 调度器将利用率提升 **20–35%**，尤其在多 GPU 设置下优势更明显。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Pixel-level communication 是解决大规模 3DGS 通信瓶颈的有效路径**，能实现通信成本与场景规模解耦。
2. **Convex partitioning 是维持全局渲染一致性的几何基础**，确保局部渲染可安全聚合。
3. **动态冗余检测（几何 + 透射）可大幅削减无效通信与计算**，提升整体效率。
4. **利用视角间无冲突特性进行调度优化，是提升 GPU 利用率的关键手段**。

### ⚠️ 方法的局限性
- **跨边界高斯处理**：当高斯尺度扩大跨越分区时，需特殊处理以避免混合顺序错误（文中采用 per-ray filtering 解决，略有额外开销）。
- **初始划分依赖 KD-tree 平衡性**：严重不平衡时性能会下降（但实验证明在 ≤20% 不平衡下仍稳健）。
- **暂未支持动态高斯增删（densification/pruning）期间的负载再均衡**，虽有轻量级重划分机制，但在极端情况下可能影响效率。

### 🔮 未来工作方向
- 将 **pixel-level communication 范式扩展至其他神经渲染模型**（如 NeRF）。
- 探索 **异构设备（CPU+GPU）协同训练** 架构下的 Splaxel 变体。
- 结合 **压缩编码技术** 对传输像素进一步压缩，降低带宽需求。
- 支持 **流式训练**，用于无限尺度场景的增量构建。

---

## 总结
> **Splaxel 通过引入 pixel-level communication 范式，彻底重构了分布式 3DGS 的训练流程，在保持高质量重建的同时，实现了高达 7.6× 的训练加速。该方法不仅解决了现有框架的通信瓶颈，还为未来大规模神经渲染系统的高效训练提供了新的设计范式。**

</details>

---

### 14. [Complementary Attention Head Pruning for Efficient Transformers](https://arxiv.org/abs/2606.19150)

**Authors**: Yaniv Livertovsky, Shahar Somin, Gonen Singer  
**Category**: cs.LG  
**Published**: 2026-06-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.19150v1  

#### Abstract
The remarkable success of Transformer-based models in natural language processing stems from architectural scaling, which leads to a large number of parameters and hinders deployment in resource-constrained environments. While structured pruning offers a pathway to compression, existing state-of-the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Complementary Attention Head Pruning for Efficient Transformers**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Transformer 模型在自然语言处理中表现出色，但由于其庞大的参数量（尤其是 Multi-Head Attention 中的 attention heads），导致部署在资源受限设备上困难。尽管已有大量关于 **structured pruning** 的研究，但现有方法存在以下问题：
- 需要手动设定 **pruning ratio** 或稀疏度目标；
- 依赖梯度重要性排序（gradient-based importance ranking）或随机门控机制（stochastic gating），容易引发训练不稳定、结构退化；
- 多数方法孤立地评估 attention head，忽略功能冗余与互补性。

### 🆕 提出的新方法：**CAHP (Complementary Attention Head Pruning)**

CAHP 是一种**自动化的后处理（post-hoc）结构化剪枝框架**，将 attention head 的选择重新定义为一个**全局图论问题**，核心思想包括：

1. **基于图的互补性选择（Complementary-based Selection）**  
   将所有 attention heads 视为统一的图空间节点，利用 **information-theoretic distance measures**（如 Jeffries-Matusita 距离）衡量行为差异，并通过聚类识别功能互补而非冗余的 heads。

2. **自动化确定剪枝数量（No Predefined Sparsity Level）**  
   不需要预设保留多少 heads。通过分析 **Mean Simplified Silhouette (MSS)** 随聚类数 $k$ 变化的曲线，使用 **Kneedle 算法** 自动检测“拐点”（knee point），即边际收益显著下降的位置，从而决定最优 head 数量 $k^*$。

3. **动态敏感性代理（Gradient-based Salience Proxy）**  
   在每个聚类中选择最具代表性的 head 时，采用基于梯度的 masking 方法计算 $w_{t,h}$ 作为重要性评分，替代静态权重。

4. **跨层全局优化（Global Cross-Layer Pruning）**  
   打破传统 layer-wise 剪枝限制，实现跨层 redistribution，更灵活地保留模型的功能核心。

### 🔍 相比现有方法的优势

| 优势维度 | CAHP 表现 |
|--------|---------|
| **无需调参** | 自动确定剪枝比例，避免试错式超参数搜索 |
| **稳定性高** | 图聚类 + 固定 fine-tuning 流程，多轮运行结果一致性强 |
| **避免 proximity bias** | 不像梯度法偏向保留靠近输出层的 heads，而是优先保留中间层关键 heads |
| **端到端流程简单** | 后处理 + 单次轻量 fine-tuning，适合实际部署 |
| **性能更强** | 特别是在高压缩率下（<20% heads 保留）仍保持接近 baseline 的准确率 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 任务类型 | 类别数 | 评估集 |
|------|--------|-------|-------|
| **SST-5** | 情感分类（单句） | 5-class | Test Set |
| **MNLI** | 自然语言推理（句子对） | 3-class (entail/neutral/contradict) | Dev-mismatched |

> 注：两个任务分别测试单句与多句建模能力，覆盖典型 NLP 场景。

### ⚙️ 实验设置

#### 模型架构
- **BERT-base**: $L=12$, $H=12$, 总共 144 个 attention heads
- **BERT-large**: $L=24$, $H=16$, 总共 384 个 attention heads

#### 关键工程细节
- **Signature Extraction**:
  - 移除 padding 后双线性插值归一化 attention map 到固定分辨率（SST-5: 32×32；MNLI: 48×48）
  - 使用 Welford’s algorithm 在线统计每类下的均值与方差
- **Graph Space 构建**:
  - 使用 t-SNE 将 high-dimensional JM distance matrix 投影至低维流形
- **Automated Selection**:
  - 对 $k \in [2, N]$ 进行 k-medoids 聚类（FasterPAM）
  - 计算 MSS 曲线并用 Kneedle 检测 knee point
  - 多项式拟合 degree $d \in \{2,\dots,6\}$ 控制平滑程度
- **Fine-tuning**:
  - 剪枝后进行 3 轮 fine-tuning，学习率 $2\times10^{-5}$，linear decay with 10% warmup
  - Batch size = 32，weight decay = 0.01，max grad norm = 1.0

#### 评估指标
- 主要指标：**Accuracy (%)**
- 辅助分析指标：
  - **Standard deviation across seeds**（稳定性）
  - **Jaccard similarity**（结构一致性）
  - **Per-layer pruning distribution**（剪枝模式）

### 🆚 基线方法对比

| 方法 | 类型 | 是否需训练 | 是否支持自动化 |
|-----|------|-----------|-------------|
| **CAHP (Ours)** | Post-hoc + Graph-based | 否（仅最后 fine-tune） | ✅ 自动确定 $k^*$ |
| **AttAttr [12]** | Attribution-based | 否 | ❌ 需手动阈值 |
| **Pipelined DSP [8]** | Differentiable pruning | 是（gate-only） | ❌ 需指定 sparsity |
| **Joint DSP [8]** | End-to-end trainable | 是（joint opt.） | ❌ 需指定 sparsity |
| **PASS [9]** | Hard concrete sparsity | 是（end-to-end） | ❌ 需指定目标 |

> 所有 baseline 被强制保留与 CAHP 相同数量的 heads，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Tables I 和 II）

#### ✅ SST-5 结果（平均 accuracy over 10 seeds）

| 方法 | BERT-large (Poly 2, ~47%) | BERT-large (Poly 6, ~15.3%) | BERT-base (Poly 6, ~18.1%) |
|------|--------------------------|----------------------------|----------------------------|
| Unpruned Baseline | 56.8% | — | 53.6% |
| **CAHP** | **55.5%** (-1.3%) | **53.3%** (-3.5%) | **51.0%** (-2.6%) |
| Joint DSP | 54.6% | 52.2% | 48.4% |
| PASS | 54.6% | 52.2% | 38.2% |
| Pipelined DSP | 52.1% | 31.1% | 38.2% |
| AttAttr | 51.1% | 31.0% | 38.1% |

> 💡 在极端压缩下（Poly 6），CAHP 显著优于其他方法，尤其在 large 模型上领先超过 **20 个百分点**。

#### ✅ MNLI 结果（平均 accuracy over 3 seeds）

| 方法 | BERT-large (Poly 2, ~45.7%) | BERT-large (Poly 6, ~14.8%) | BERT-base (Poly 6, ~19.4%) |
|------|----------------------------|----------------------------|----------------------------|
| Unpruned Baseline | 86.5% | — | 84.2% |
| **CAHP** | **85.9%** (-0.6%) | **84.0%** (-2.5%) | **81.8%** (-2.4%) |
| Joint DSP | 85.9% | 84.5% | 81.2% |
| PASS | 85.4% | 83.3% | 77.2% |
| Pipelined DSP | 83.9% | 57.8% | 72.5% |

> 💡 在 MNLI 上，CAHP 与最强的 Joint DSP 并驾齐驱，在 Base 模型上甚至反超。

### 🔬 结构性发现与对比分析

- **避免 Proximity Bias**：
  - Joint DSP 等梯度法倾向于保留最后几层（L10–L12）的 heads（受 loss proximity 影响）；
  - CAHP 更多地保留 **中间层（L7–L9）** 的 heads，这些层被证明对语义转换至关重要。

- **更高的结构一致性（Structural Consistency）**：
  - 如 Fig. 4 所示，CAHP 在不同 seed 下的 Jaccard similarity 更高，说明选出的子网络结构更稳定；
  - PASS 等方法波动大，部分 seed 出现崩溃式性能下降。

- **计算效率可接受**：
  - 尽管引入图构建阶段，CAHP 总体运行时间与 Joint DSP / PASS 处于同一量级（见 Table III）；
  - 例如在 BERT-large + MNLI 上耗时约 **5h40m**，远低于反复调参的成本。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **大多数 attention heads 是冗余的**，只需保留约 **15–20%** 的 complementary heads 即可维持接近原始模型的性能。
2. **功能多样性比单一重要性更重要**：基于互补性的选择策略能更好地保留模型的表达能力。
3. **中间层是功能核心所在**：相比末端层，中间层的 attention heads 更具功能性，应优先保留。
4. **自动化剪枝可行且高效**：通过 MSS + Kneedle 可自动确定最佳剪枝规模，无需人工干预。
5. **Post-hoc 方法也能媲美联合训练方法**：CAHP 作为纯后处理方法，在多数情况下达到甚至超越 Joint DSP/PASS 的性能。

### ⚠️ 局限性

- 当前仅适用于 **encoder-only** Transformer（如 BERT），尚未扩展至 decoder 或生成式模型（如 T5、LLaMA）。
- 图空间构建和 k-medoids 聚类在非常大的 head 数量下可能带来额外开销（虽可控）。
- 依赖 fine-tuning 来恢复性能，若下游任务数据极少，效果可能受限。

### 🔮 未来工作方向

1. **拓展至 generative、multilingual 和 unsupervised 设置**；
2. **探索更高效的 distance metric 与 clustering algorithm** 以提升速度；
3. **将 complementary selection 整合进训练过程**（而非仅 post-hoc），进一步增强稳定性；
4. **结合 layer pruning 与 head pruning**，实现更细粒度的结构压缩。

---

> ✅ **代码已开源**：https://github.com/yanivlivert/cahp  
> 📄 **原文链接**：https://arxiv.org/abs/2606.19150

--- 

📌 **一句话总结**：  
CAHP 提出了一种**无需手动设定稀疏度、基于图论与互补性原则的自动化 attention head 剪枝方法**，在多种模型与任务上实现了高压缩比下的高性能与高稳定性，推动了 Transformer 模型向实用化部署迈进。

</details>

---

### 15. [R2D-RL: A RoboCup 2D Soccer Environment for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2606.18786)

**Authors**: Haobin Qin, Baofeng Zhang, Hidehisa Akiyama, Keisuke Fujii  
**Category**: cs.AI  
**Published**: 2026-06-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.18786v1  

#### Abstract
Robot soccer is a challenging testbed for multi-agent reinforcement learning because it combines partial observability, cooperative and adversarial interaction, sparse rewards, and long-horizon tactical behavior. RoboCup 2D Soccer Simulation (RCSS2D) provides a mature robot-soccer platform, but its ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：R2D-RL: A RoboCup 2D Soccer Environment for Multi-Agent Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
RoboCup 2D Soccer Simulation (RCSS2D) 是一个成熟的机器人足球模拟平台，广泛用于多智能体系统研究。然而，其**以比赛为导向的客户端-服务器架构**（client-server）难以直接集成到现代基于 Python 的 **MARL（Multi-Agent Reinforcement Learning）训练流程**中，存在以下障碍：
- 缺乏与学习算法同步的环境接口；
- 难以实现集中式训练、去中心化执行（CTDE）所需的全局状态、奖励信号和动作掩码（action masks）；
- 不支持并行采样和可复现的任务控制。

因此，尽管 RCSS2D 具备完整的比赛机制和丰富的战术生态（如 HELIOS 行为模块），却难以被用于端到端的强化学习研究。

### 提出了什么新方法或新思路
本文提出了 **R2D-RL** —— 一个连接原始 RCSS2D 模拟器与 Python MARL 生态系统的**强化学习环境**。其核心设计包括：

- **共享内存通信 + 周期级同步协议**：通过 `shared-memory` 和 `sequence-counter` 协议，在不修改原生 RCSS2D 服务器逻辑的前提下，将实时运行的比赛流程转换为步进同步（step-synchronous）的 RL 环境交互。
- **双层策略动作空间设计**：
  - **Base Discrete Action Space**：高层语义动作（如“射门”、“直塞传球”、“拦截”等），由 HELIOS 行为模块自动解析为底层命令；
  - **Hybrid Parameterized Action Space**：混合离散-连续动作（如 `kick(power, angle)`），允许更细粒度控制。
- **完整的学习支持功能**：
  - 动作掩码（Action Masks）：基于 HELIOS 可执行性检查动态生成合法动作集合；
  - EPV-based Reward Shaping：引入 Expected Possession Value 进行稀疏奖励塑形；
  - 支持场景初始化、并行执行、重置控制等。

### 相比现有方法的优势
| 特性 | R2D-RL | GRF / HFO / Keepaway |
|------|--------|-----------------------|
| 是否基于真实 RoboCup 2D？ | ✅ 完整保留 RCSS2D 规则与物理 | ❌ 多为简化或自定义模拟器 |
| 是否支持全场比赛（11-vs-11）？ | ✅ | ⚠️ GRF 支持，HFO/Keepaway 否 |
| 是否提供高层行为抽象？ | ✅（Base + Hybrid） | ⚠️ GRF 提供高阶动作，但非基于 HELIOS |
| 是否有动作掩码？ | ✅ | ❌ 绝大多数无 |
| 是否兼容原有 RoboCup 社区生态？ | ✅ 可复用 HELIOS/BASE 团队 | ❌ 独立系统 |

> ✅ **R2D-RL 的最大优势在于：在保持 RCSS2D 完整性和社区生态的同时，将其转化为适合现代 MARL 研究的标准环境。**

---

## 2. 核心实验方法和设置

### 使用的数据集 / 环境配置
R2D-RL 并非使用外部数据集，而是构建了一个**可编程的仿真环境**，实验基于以下设定：

- **基础模拟器**：RoboCup 2D Soccer Server (RCSS2D)
- **AI 对手**：使用 **HELIOS Fallback** 策略作为固定对手（built-in AI），即所有非受控球员执行预设规则行为。
- **可控代理数量**：
  - 前场小规模任务：1~4 名攻击方球员；
  - 全场任务：11-vs-11，左队全部由学习策略控制。

### 实验设置和评估指标

#### 场景设计
- **Front-Goal Scenarios（渐进难度前场进攻任务）**：
  1. **Empty Goal**（1-vs-0）：空门射门
  2. **Blocked Shot**（1-vs-1）：面对防守射门
  3. **Support Option**（2-vs-1）：队友支援配合
  4. **Passing Lane**（3-vs-2）：选择传球路线
  5. **Compact Defense**（4-vs-3）：对抗密集防守
- **Full-Field Benchmark**：标准 11-vs-11 比赛，最大 3000 environment steps。

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Goal Rate** | 在限定步数内得分的比例（用于 front-goal 场景） |
| **Goal Difference** | 控制队进球数减去对手进球数（用于全场） |
| **max_epv_improvement** | 最大达到的 EPV 值减去初始 EPV 值，衡量进攻推进质量 |
| **Throughput** | 单日环境步数（steps/day），衡量并行采样效率 |

#### 基线方法对比
- **Base Action Space**：
  - **MAPPO**（Multi-Agent PPO）
  - **QMIX**（值分解方法）
- **Hybrid Action Space**：
  - **ParaDQN**（参数化 DQN，独立学习器，参数共享）

> 所有实验均使用三种随机种子（0,1,2），报告均值±标准差。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 并行采样吞吐量（Parallel Sampling Throughput）
- 在 32 个逻辑 CPU 上测试：
  - **Base Action Space**：最高达 **34.85 million steps/day**（14 个并行环境时）
  - **Hybrid Action Space**：最高达 **41.65 million steps/day**
- 表明 R2D-RL 支持高效的大规模并行训练。

#### Front-Goal 场景结果（Goal Rate）

| 场景 | 最佳模型 | 成绩（mean ± std） |
|------|---------|--------------------|
| Empty Goal | 所有模型 | ≈1.00 |
| Blocked Shot | MAPPO | 0.89 ± 0.02 |
| Support Option | MAPPO | 0.83 ± 0.11 |
| Passing Lane | MAPPO | 0.60 ± 0.03 |
| Compact Defense | MAPPO | 0.32 ± 0.09 |

> ParaDQN 在多智能体场景下表现极差（接近 0），说明 Hybrid 动作空间在复杂协作任务中训练困难。

#### 消融实验结果（Ablation Study）
在不同组件组合下的表现差异显著：

- **Action Masks 至关重要**：
  - QMIX 在 Blocked Shot 中，**无 mask 时 goal rate 为 0**；启用 mask 后提升至 ~0.88。
  - 因为 mask 强制在可射门时优先选择“射门”，避免因探索不足错过机会。
- **MaxEPV Reward Shaping 有效**：
  - 在 Support Option 和 Passing Lane 中，仅使用 MaxEPV 能部分提升性能；
  - 与 mask 结合效果最佳。
- **No EPV/mask 设置几乎无法得分**：尤其在多智能体场景中，goal rate 接近零。

#### Full-Field 11-vs-11 基准结果（@30M steps）

| Algorithm | Goal Diff. | max_epv_improvement |
|----------|------------|---------------------|
| **MAPPO** | **-10.61 ± 1.64** | 0.0317 ± 0.0235 |
| QMIX     | -19.00 ± 8.60 | 0.0323 ± 0.0355 |
| ParaDQN  | -25.90 ± 2.69 | 0.0014 ± 0.0002 |

> - MAPPO 表现最稳定，goal difference 明显优于其他；
> - QMIX 种子间波动大（-10 到 -30），表明对初始化敏感；
> - ParaDQN 几乎未学会推进球权，max_epv_improvement 极低。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **R2D-RL 成功实现了 RCSS2D 与现代 MARL 流程的桥接**，支持同步采样、集中状态、动作掩码、奖励塑形等功能。
2. ✅ **MAPPO 在多种任务中表现最优且最稳健**，尤其在 full-field 设置下能产生初步的多智能体协调行为（见附录可视化）。
3. ✅ **Action Masks 和 MaxEPV Reward Shaping 是提升训练效率的关键组件**，尤其是在复杂、稀疏奖励的多智能体任务中。
4. 🔁 **Hybrid Action Space 虽然理论上更具表达力，但在当前设置下训练难度极高**，尤其在需要协作的任务中表现不佳。
5. 📈 **R2D-RL 支持高效的并行采样**，具备用于大规模训练的潜力。

### 方法的局限性
1. **对手多样性不足**：目前仅使用固定的 HELIOS Fallback 对手，缺乏 self-play 或多样化 opponent pool，限制了策略泛化能力。
2. **训练预算有限**：30M 步对于 full 11-vs-11 仍属初级阶段，尚未体现最终性能上限。
3. **动作空间依赖手工设计**：Base 动作和 mask 规则均由 HELIOS 规则编码，引入了强归纳偏置（inductive bias）。
4. **仅开放 play_on 模式决策**：未涵盖任意球、角球等 set-piece 情况，限制了完整战术学习。

### 未来工作方向
1. ✅ 引入 **self-play** 和 **learned opponent pools**，增强策略鲁棒性与泛化能力；
2. ✅ 探索 **curriculum learning**、**demonstration pretraining** 等方法以加速训练；
3. ✅ 设计 **更少手工干预的动作抽象** 和 **自适应动作掩码机制**；
4. ✅ 扩展至 **non-play_on 模式**（如 free kick, corner kick），实现全面的战术学习；
5. ✅ 开发 **轻量化版本或蒸馏模型**，降低计算门槛。

---

> 💡 **总结**：  
> R2D-RL 是首个将 **完整 RoboCup 2D 生态系统** 无缝接入现代 MARL 训练框架的工作。它不仅提供了高保真、可扩展的足球 MARL 环境，还通过 **Base/Hybrid 动作空间、动作掩码、EPV 奖励塑形** 等设计推动了复杂多智能体任务的学习效率。虽然当前性能仍有提升空间，但它为未来研究提供了一个**可复现、标准化、富有挑战性的基准平台**。

</details>

---

### 16. [ThinkDeception: A Progressive Reinforcement Learning Framework for Interpretable Multimodal Deception Detection](https://arxiv.org/abs/2606.18988)

**Authors**: Jinhao Song, Shan Liang, Yiqun Yue, Zhuhuayang Zhang, Tianqi Gao  
**Category**: cs.AI  
**Published**: 2026-06-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.18988v1  

#### Abstract
Multimodal deception detection is critical for identifying fraudulent intentions, yet existing approaches predominantly rely on end to end black--box paradigms. These methods suffer from a severe lack of interpretability failing to provide transparent reasoning trajectories and struggling to explici...

---

### 17. [Stealthy World Model Manipulation via Data Poisoning](https://arxiv.org/abs/2606.18697)

**Authors**: Yibin Hu, Xiaolin Sun, Zizhan Zheng  
**Category**: cs.LG  
**Published**: 2026-06-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.18697v1  

#### Abstract
Model-based learning agents use learned world models to predict future states, plan actions, and adapt to new environments. However, the process of updating world models from collected experience creates a training-time attack surface: adversarially poisoned fine-tuning trajectories can manipulate t...

---

### 18. [AGDN: Learning to Solve Traveling Salesman Problem with Anisotropic Graph Diffusion Network](https://arxiv.org/abs/2606.19185)

**Authors**: Bolin Shen, Ziwei Huang, Zhiguang Cao, Yushun Dong  
**Category**: cs.LG  
**Published**: 2026-06-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.19185v1  

#### Abstract
The Traveling Salesman Problem (TSP) is a cornerstone of combinatorial optimization and arises in many practical scenarios. Although graph-based learning approaches have been explored for TSP, the question of how to exploit graph structure more effectively remains open. We present the Anisotropic Gr...

---

### 19. [P-K-GCN: Physics-augmented Koopman-enhanced Graph Convolutional Network for Deep Spatiotemporal Super-resolution](https://arxiv.org/abs/2606.19303)

**Authors**: Xizhuo (Cici),  Zhang, Zekai Wang, Fei Liu, Bing Yao  
**Category**: cs.LG  
**Published**: 2026-06-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.19303v1  

#### Abstract
High-fidelity simulation of spatiotemporal dynamics is computationally prohibitive, necessitating efficient super-resolution techniques to reconstruct high-resolution data from coarse-grained inputs. Traditional data-driven methods often lack physical constraints, and simple physics-informed learnin...

---

### 20. [Generative-Model Predictive Planning for Navigation in Partially Observable Environments](https://arxiv.org/abs/2606.18888)

**Authors**: Thomas Quilter, Yifan Zhu, Guorui Quan, Mingfei Sun, Samuel Kaski  
**Category**: cs.AI  
**Published**: 2026-06-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.18888v1  

#### Abstract
Navigation in partially observable environments presents a significant challenge for autonomous agents, requiring effective decision-making with limited sensory information in unknown environments. Belief-based methods, particularly those using neural networks to approximate the belief space, often ...

---

### 21. [X+Slides: Benchmarking Audience-Conditioned Slide Generation](https://arxiv.org/abs/2606.19256)

**Authors**: Haodong Chen, Xuanhe Zhou, Wei Zhou, Xinyue Shao, Yanbing Zhu, Bo Wang, Jiawei Hong, Anya Jia, Fan Wu  
**Category**: cs.AI  
**Published**: 2026-06-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.19256v1  

#### Abstract
Automatically generating slide decks from source documents is an important application of large language models (LLMs). Existing benchmarks primarily assess slide completeness and technical depth, while overlooking the target audience as a critical real-world factor. For instance, specialists demand...

---

### 22. [PragReST: Self-Reinforcing Counterfactual Reasoning for Pragmatic Language Understanding](https://arxiv.org/abs/2606.18624)

**Authors**: Jihyung Park, Minchao Huang, Leqi Liu, Elias Stengel-Eskin  
**Category**: cs.CL  
**Published**: 2026-06-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.18624v1  

#### Abstract
Natural language understanding often depends on meanings that are implied rather than explicitly stated, requiring pragmatic reasoning. Despite strong performance on math and logical reasoning, large language models (LLMs) still struggle with making pragmatic inferences, often choosing literal inter...

---

### 23. [Learning Robust Pair Confidence for Multimodal Emotion-Cause Pair Extraction](https://arxiv.org/abs/2606.18893)

**Authors**: Zhuangzhuang Pan, Ning Dong, Yingna Su, Yan Xia  
**Category**: cs.CL  
**Published**: 2026-06-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.18893v1  

#### Abstract
Multimodal emotion-cause pair extraction (MECPE) requires reliable pair confidence over candidate pairs. Existing pair scorers commonly use pair-level cross entropy over valid candidates, which treats links mostly independently. This leaves the relative confidence geometry among competing causes und...

---

### 24. [DreamReasoner-8B: Block-Size Curriculum Learning for Diffusion Reasoning Models](https://arxiv.org/abs/2606.19257)

**Authors**: Zirui Wu, Lin Zheng, Jiacheng Ye, Shansan Gong, Xueliang Zhao, Yansong Feng, Wei Bi, Lingpeng Kong  
**Category**: cs.CL  
**Published**: 2026-06-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.19257v1  

#### Abstract
Block diffusion language models accelerate decoding through parallel block-wise denoising, yet whether they can be reliably scaled for long chain-of-thought (CoT) reasoning remains unresolved. To this end, we develop DreamReasoner-8B, an open-source block diffusion reasoning model, and conduct a sys...

---

### 25. [INDEQS: Informed Neural controlled Differential EQuationS](https://arxiv.org/abs/2606.19138)

**Authors**: Michael Detzel, Gabriel Nobis, Kristiyan Blagov, Juri Schubert, Jackie Ma, Wojciech Samek  
**Category**: cs.LG  
**Published**: 2026-06-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.19138v1  

#### Abstract
Neural Controlled Differential Equations (NCDE) provide a powerful continuous-time framework for forecasting time series, but standard graph-based extensions typically learn spatial structure purely from data, even in settings where a directed graph structure is known a priori. We introduce Informed...

---

### 26. [ARIADNE: Agnostic Routing for Inference-time Adapter DyNamic sElection](https://arxiv.org/abs/2606.19079)

**Authors**: Enrico Cassano, Micha{\l} Brzozowski, Zuzanna Dubanowska, Paolo Mandica, Neo Christopher Chung  
**Category**: cs.AI  
**Published**: 2026-06-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.19079v1  

#### Abstract
The increasing deployment of parameter-efficient fine-tuning (PEFT) has led to model ecosystems in which a single backbone is paired with many task-specialized adapters. In this setting, inference-time queries often arrive without task labels, requiring the system to automatically select the most ap...

---

### 27. [Beyond Safe Data: Pretraining-Stage Alignment with Regular Safety Reflection](https://arxiv.org/abs/2606.19168)

**Authors**: Jinhan Li, Kexian Tang, Yihan Xu, Zhuorui Ye, Kaifeng Lyu  
**Category**: cs.AI  
**Published**: 2026-06-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.19168v1  

#### Abstract
To achieve deeper safety alignment for large language models (LLMs), recent efforts have studied how to push safety interventions earlier into the pretraining stage, primarily by filtering unsafe data or rewriting it into safer forms. We argue that pretraining-stage alignment should go beyond making...

---

### 28. [SproutRAG: Attention-Guided Tree Search with Progressive Embeddings for Long-Document RAG](https://arxiv.org/abs/2606.18381)

**Authors**: Amirhossein Abaskohi, Issam H. Laradji, Peter West, Giuseppe Carenini  
**Category**: cs.CL  
**Published**: 2026-06-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.18381v1  

#### Abstract
Retrieval-augmented generation (RAG) systems must balance retrieval granularity with contextual coherence, a challenge that existing methods address through LLM-guided chunking, single-level context expansion, or hierarchical summarization. These approaches variously depend on costly LLM calls durin...

---

### 29. [Efficient Financial Language Understanding via Distillation with Synthetic Data](https://arxiv.org/abs/2606.18875)

**Authors**: Wen-Fong (Xavier),  Huang, Edwin Simpson  
**Category**: cs.CL  
**Published**: 2026-06-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.18875v1  

#### Abstract
Large instruction-following models are powerful but costly to deploy, particularly in finance, where labelled data are limited by confidentiality and expert annotation cost. We present an efficient framework for financial sentiment analysis through distillation with synthetic data, transferring know...

---

### 30. [Mixed-Precision Communication-Avoiding SGD for Generalized Linear Models on GPUs](https://arxiv.org/abs/2606.18463)

**Authors**: Aditya Devarakonda, Irene Sim\'o Mu\~noz, Giulia Guidi  
**Category**: cs.DC  
**Published**: 2026-06-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.18463v1  

#### Abstract
Distributed stochastic gradient descent (SGD) is limited by communication rather than computation, since each iteration requires an AllReduce across processes. Communication-avoiding SGD (CA-SGD) amortizes communication over $s$ iterations by replacing $s$ consecutive AllReduces with a single AllRed...

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

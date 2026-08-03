# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-03 08:55:51 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SLIM: Saturation-Aware Lightweight Performance Modeling for LLM Serving](https://arxiv.org/abs/2607.29575)

**Authors**: Pol G. Recasens, Ferran Agullo, Yue Zhu, Chen Wang, Jordi Torres, Josep Ll. Berral  
**Category**: cs.DC  
**Published**: 2026-08-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.29575v1  

#### Abstract
Large language model (LLM) serving commonly increases batch size to improve throughput, but performance eventually reaches a deployment-dependent plateau beyond which larger batches provide marginal gains while increasing latency and GPU memory consumption. Previous studies have attributed this beha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SLIM: Saturation-Aware Lightweight Performance Modeling for LLM Serving**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLM）在服务部署中通常通过增大 **batch size** 来提升吞吐量（throughput），但实际中存在“吞吐量饱和”现象（throughput plateau）——即超过某一阈值后，继续增加 batch size 对吞吐量的提升微乎其微，反而显著增加延迟（latency）和 GPU 内存消耗。

尽管已有研究将此归因于 **HBM/DRAM 带宽瓶颈**，但缺乏底层硬件层面的实证分析，且缺乏轻量级、可泛化的性能建模工具来指导系统配置优化。

本论文旨在：
- 揭示吞吐量饱和的根本原因；
- 构建一个准确、轻量、可泛化的性能预测模型；
- 提供一种高效的批处理配置建议框架。

---

### **提出了什么新方法或新思路**

#### **(1) 低层性能分析揭示根本瓶颈**
通过 **Nsight Systems/Compute** 进行 GPU 内核级剖析，首次从硬件层面证实：
> 吞吐量饱和的根本原因是 **decode 阶段 attention kernels 的 DRAM 带宽饱和**，而非整体计算瓶颈。

关键机制是：随着 **active context**（输入长度 + 输出长度 × batch size）增长，attention kernels 的 **arithmetic intensity（算力强度）几乎恒定**，导致内存流量线性增长，最终达到 DRAM 带宽上限，而计算单元（如 Tensor Cores）利用率仍远低于峰值。

#### **(2) 提出 SLIM：饱和感知轻量级性能模型**
- **SLIM (Saturation-Aware Lightweight Performance Model)** 是一个半解析式（semi-analytical）模型，结合了：
  - Transformer 计算与内存访问的解析公式；
  - 少量实测校准参数（如 compute efficiency、decode efficiency）。
- 能够预测不同模型规模、序列长度、batch size 下的 **throughput 和 E2E latency**。
- 支持对未见过的模型和上下文长度进行泛化预测。

#### **(3) 提出 BCA：批处理配置顾问**
- **Batching Configuration Advisor (BCA)** 基于 SLIM 模型运行，目标是在满足延迟 SLO（Service Level Objective）的前提下，选择最优的 batch size 上限 $ B_{cap} $。
- 引入 **最小批处理效率约束**（minimum batching efficiency），避免进入吞吐平台区（plateau region），从而节省大量 GPU 内存用于 KV Cache。

---

### **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **分析深度** | 首次提供内核级证据，明确指出 attention kernels 的内存带宽饱和是瓶颈，超越以往概念性推断。 |
| **模型设计** | SLIM 显式建模 KV-cache 内存传输成本，支持跨模型、跨长度泛化；而 LLMVisor、Imai et al. 等依赖拟合，泛化能力弱。 |
| **实用性** | BCA 可减少高达 **43.85% 的 profiling 时间**，并释放最多 **55 GB GPU 内存**，显著提升资源利用率。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与模型**
- **合成 workload**：控制变量测试 batch size、input length、output length 影响。
- **真实 workload**：使用清洗后的 **ShareGPT 数据集**，保留真实的请求分布。
- **模型集合**：
  - 分析阶段：Mistral-7B, Granite-8B
  - 主要实验：OPT 系列（125M ~ 6.7B）
  - 大模型扩展：Qwen-32B, Qwen-72B（多 GPU 场景）

### **实验设置**
- **框架**：vLLM（version 0.15.1），启用 **paged attention** 和 **continuous batching**。
- **硬件**：NVIDIA H100（64GB HBM2），4 GPUs / node，CPU 80 cores，RAM 512GB。
- **模式**：
  - **online mode**：HTTP 请求模拟真实服务；
  - **offline mode**：Python 注入请求，用于 Nsight profiling。

### **评估指标**
- **主指标**：
  - **Throughput (tokens/s)**
  - **End-to-End Latency (ms)**
  - **Mean Absolute Percentage Error (MAPE)**：衡量预测准确性。
- **辅助指标**：
  - DRAM Read Bandwidth Utilization
  - Tensor Core Activity (%)
  - SM Issue Rate
  - L1/L2 Cache Hit Rate
  - Long Scoreboard Stalls

### **基线方法对比**
| 基线方法 | 描述 |
|--------|------|
| **LLMVisor-Agg** | 改编自 LLMVisor，使用分段拟合 prefill/decode 延迟，基于 token 数特征。 |
| **Imai et al. [11] (IMAI OOD-LR)** | 使用回归模型预测延迟，输入为 batch size、seq length、model 参数等。 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 指标 | 结果 |
|------|------|
| **Throughput MAPE（平均）** | SLIM 达到 **17.6%**，比基线平均降低 **79.3%** |
| **Latency MAPE（平均）** | SLIM 达到 **13.4%** |
| **最大内存节省** | 最多 **55.42 GB**（OPT-350M on H100） |
| **profiling 时间节省** | 减少 **43.85%**（相比 exhaustive search） |
| **DRAM 利用率峰值** | 接近 **100%**（attention kernel） |
| **Tensor Core 利用率** | 平均 < **30%**，表明严重 underutilized |

---

### **与基线方法的对比结果**

| 场景 | SLIM (Throughput MAPE) | LLMVisor-Agg | IMAI OOD-LR |
|------|------------------------|---------------|-------------|
| **Same Curve (held-out batch)** | 15.1% | 55.8% | 55.3% |
| **Input Length Generalization** | 0.7% | 68.1% | 1.2% |
| **Output Length Generalization** | 5.6% | 7.7% | 38.9% |
| **Model Generalization (→ OPT-6.7B)** | 24.3% | 141.8% | 39.9% |
| **Combined (Model + Output)** | **17.6%** | 95.8% | 91.7% |

> ✅ SLIM 在所有泛化场景下均显著优于两个基线，尤其在跨模型和长输出场景优势巨大。

---

### **消融实验结果（Ablation Study）**

| 设置 | Variant | Throughput MAPE |
|------|--------|------------------|
| **Model Transfer** | Full SLIM | 24.3% |
| | w/o KV scan ($M_{dec}$) | 23.5% |
| | w/o dense decode ($T_d$) | **31.6%** ❗ |
| **Output Length Transfer** | Full SLIM | 5.6% |
| | w/o KV scan ($M_{dec}$) | **33.0%** ❗ |
| | w/o dense decode ($T_d$) | 8.1% |

> 🔍 发现：
> - **KV scan term ($M_{dec}$)** 对输出长度泛化至关重要；
> - **dense decode term ($T_d$)** 更影响模型尺度泛化；
> - 两者互补，共同支撑高精度预测。

---

## **4. 关键结论和发现**

### **主要发现**
1. **吞吐量饱和源于 attention kernels 的 DRAM 带宽饱和**：
   - 随着 active context 增大，attention 的 arithmetic intensity 几乎不变 → 内存流量线性上升 → 触达带宽极限。
   - 此时 compute 资源（Tensor Core）利用率不足 30%，造成严重资源浪费。

2. **prefill 阶段非瓶颈**：
   - prefill 是 compute-bound，可高度并行化；
   - decode 才是主导延迟和瓶颈的关键阶段。

3. **缓存局部性差**：
   - L1/TEX 缓存命中率 < 1%
   - L2 缓存命中率 < 7%
   - 表明 KV-cache 数据难以复用，加剧 DRAM 访问压力。

4. **long scoreboard stalls 高企**：
   - 超过 50% 的 warp cycles 因内存等待而停滞，进一步验证内存瓶颈。

5. **SLIM 具备强泛化能力**：
   - 即使仅在小模型上校准，也能准确预测大模型（如 Qwen-72B）趋势；
   - 支持新型 attention 架构（如 Grouped-Query Attention in Mistral-7B）。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **未显式建模 tensor parallelism** | 在多 GPU 场景下，SLIM 将 Qwen-72B 误判为更重负载，导致预测偏保守。 |
| **依赖少量 profiling 数据** | 虽然大幅减少开销，但仍需初始校准；极端优化（如 quantization, prefix caching）可能需要重新校准。 |
| **静态 workload 假设** | BCA 当前为离线配置器，假设 workload 特征稳定；动态变化需结合 runtime monitoring。 |
| **单设备校准** | 不同 GPU 架构需重新 calibrate，无法直接迁移。 |

---

### **未来工作方向**
1. **集成在线调度机制**：将 BCA 与 runtime monitor 结合，实现动态调整 batch size。
2. **支持更多优化技术**：扩展 SLIM 以兼容量化 KV-cache、speculative decoding、prefix caching 等。
3. **跨异构 GPU 自适应**：开发自动 calibration pipeline，适配不同型号加速器。
4. **多模型共置场景优化**：利用 SLIM 进行 multi-tenant 容量规划与资源隔离。
5. **探索缓解带宽瓶颈的新架构**：如更高带宽 HBM、on-chip KV 存储、稀疏 attention 等。

---

> 📌 **总结一句话**：  
> 本文通过底层硬件剖析揭示了 LLM serving 中吞吐量饱和的本质是 **decode 阶段 attention kernels 的 DRAM 带宽饱和**，并提出 **SLIM + BCA** 框架，在极低 profiling 开销下实现高性能预测与高效资源配置，为 LLM 服务系统的容量规划与性能调优提供了坚实基础。

</details>

---

### 2. [Don't Mix Rewards, Mix Policies: Policy Decomposition and Optimization for Multi-Reward RL](https://arxiv.org/abs/2607.29246)

**Authors**: Ruiming Liang, Yi Zhong, Yizhen Yuan, Yinan Zheng, Tianyi Tan, Tianyue Wang, Haiyun Guo, Jinqiao Wang, Xianyuan Zhan  
**Category**: cs.AI  
**Published**: 2026-08-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.29246v1  

#### Abstract
Modern large language models (LLMs) are expected not just to answer correctly, but to adapt their behavior to different human values and use cases. As a result, multi-reward reinforcement learning (RL) has become an increasingly important problem for LLMs, where each reward captures a different aspe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Don't Mix Rewards, Mix Policies: Policy Decomposition and Optimization for Multi-Reward RL**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大语言模型（LLMs）需要同时满足多种人类偏好（如正确性、安全性、格式规范等），这构成了**多奖励强化学习（Multi-Reward RL）**任务。然而，传统方法在奖励空间中将多个奖励信号加权合并为单一目标进行优化，存在以下问题：

- **对齐税（alignment tax）严重**：不同奖励之间可能存在冲突，导致梯度竞争，最终策略在各维度上均表现不佳。
- **权重敏感且不可靠**：奖励尺度、分布差异使得人工设定的权重难以调节，且需针对不同任务重新调参。
- **缺乏推理时可控性**：训练完成后无法灵活调整偏好权衡。

### 🚀 提出的新方法：PRISM
作者提出 **PRISM**（**Policy-space Reward Integration via Sub-policy Mixture**），其核心思想是：

> **“不要混合奖励，要混合策略”** —— 将多奖励优化从奖励空间转移到策略空间。

#### 方法核心设计：
- **每个奖励训练一个独立的正向策略（positive policy）**：专注于捕捉该奖励鼓励的行为。
- **一个全局负向策略（global negative policy）**：统一建模所有奖励的失败模式（即任何一项不达标即视为失败）。
- 所有子策略共享同一个语言模型主干，通过不同的 **prefix token** 来区分。
- 在推理阶段，通过 **logit-level 加权组合** 构成最终输出策略：
  $$
  z^* = \sum_{k=1}^N \alpha_k z_k^+ - \gamma z^-
  $$
  其中 $\alpha_k$ 和 $\gamma$ 是可手动调节的融合权重。

### 🔍 相比现有方法的优势
| 特性 | 传统方法（如 GRPO Sum/Product, GDPO） | PRISM |
|------|----------------------------------------|-------|
| 优化方式 | 奖励空间融合（reward-space composition） | 策略空间融合（policy-space composition） |
| 冲突处理 | 多目标在同一梯度步内竞争 | 各正向策略独立优化，减少干扰 |
| 推理控制 | 固定策略，无运行时调节能力 | 支持训练后动态调整偏好权重 |
| 效率 | 单一模型 | 共享主干 + prefix conditioning，高效实现多分支 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与任务
PRISM 在三个典型多奖励对齐场景下进行了验证：

| 任务 | 数据集 | 奖励类型 |
|------|--------|----------|
| **科学问答**（Scientific QA） | `SciKnowEval`（训练），`GPQA`, `ScienceQA`（测试） | 正确性（correctness）、格式合规性（format） |
| **工具调用推理**（Tool-use Reasoning） | `ToolRL`（训练），`BFCL-v3`（测试） | 正确性、格式、长度（reasoning depth） |
| **有用性-安全性对齐**（Helpfulness-Safety Alignment） | `Alpaca`（训练），`HH-RLHF`, `PKU-SafeRLHF`（测试） | 有用性（helpfulness）、无害性（harmlessness） |

### 🧪 实验设置与评估指标

#### 模型主干：
- `DeepSeek-R1-1.5B`
- `Qwen2.5-1.5B-Instruct`
- `Qwen2.5-3B-Instruct`

#### 训练配置：
- 使用 **LoRA** 进行参数高效微调。
- 子策略通过 **prefix tuning** 实现，共享主干模型。
- 采用 **group-relative advantage estimation (GRPO)** 作为基础更新机制。
- 所有子策略并行采样，提升解码效率。

#### 评估指标：
| 任务 | 主要指标 |
|------|---------|
| 科学问答 | `Fmt`（格式准确率）、`Acc`（答案正确率）、`Joint`（两者同时满足）、`Avg`（平均分） |
| 工具调用 | `Fmt`（协议符合度）、`Acc/R`（RLLA风格严格准确率）、`Acc/B`（BFCL语义级准确率） |
| 安全对齐 | `Useful`, `Harmless`, `Avg`（人工评分，越高越好） |

### ⚔️ 基线方法对比
- **GRPO Sum**：奖励加权求和
- **GRPO Product**：基于乘积聚合的策略优化
- **GDPO**：当前最先进的多奖励归一化方法，缓解尺度问题但仍属奖励空间融合

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ 科学问答（Table 1）
| 模型 | 方法 | GPQA Avg ↑ | ScienceQA Joint ↑ |
|------|------|------------|---------------------|
| DeepSeek-R1-1.5B | GDPO | 22.69 | 58.04 |
| | **PRISM (ours)** | **47.55** (+109%) | **68.51** |
| Qwen2.5-1.5B | GDPO | 42.33 | 76.26 |
| | **PRISM** | **58.45** | **76.45** |
| Qwen2.5-3B | GDPO | 47.55 | 82.80 |
| | **PRISM** | **49.55** | **84.11** |

> 💡 PRISM 在最难的 GPQA 上平均得分翻倍以上，在所有 backbone 上均取得最高 **Joint Score**，说明其能更好兼顾多个目标。

#### ✅ 工具调用（Table 2）
| 方法 | Fmt ↑ | Acc/R ↑ | Acc/B ↑ | Overall ↑ |
|------|-------|--------|--------|----------|
| GDPO | 93.81 | 36.38 | 52.13 | — |
| **PRISM (ours)** | **95.23** | 36.20 | **53.46** | **SOTA** |

> ✅ PRISM 在非实时（Non-live）、多轮（Multi-turn）等复杂场景下全面领先，尤其在 **格式合规性** 和 **综合表现** 上优势明显。

#### ✅ 有用性-安全性对齐（Table 3）
| 方法 | Alpaca Avg ↑ | HH-RLHF Avg ↑ | PKU-SafeRLHF Avg ↑ |
|------|-------------|--------------|--------------------|
| GDPO | 3.20 | 3.53 | 5.56 |
| **PRISM** | **3.41** | **3.65** | **5.61** |

> ✅ 在所有三项评测中均达到最佳，且差距显著（+0.15 ~ +0.05），表明 PRISM 能有效平衡互斥目标。

---

### 🔬 消融实验结果（Ablation Studies）

#### 表 4：策略组成消融（Policy Composition Ablation）
| 变体 | Fmt | Acc/B ↓ |
|------|-----|--------|
| 完整 PRISM | 95.23 | 53.46 |
| 共享正向策略（shared positive） | 94.03 | 52.56 |
| 移除全局负策略（no negative） | 94.86 | 52.84 |
| 分支独立 rollout | 94.84 | 52.48 |

> ❗ 结果显示：
- **专用正向策略至关重要**（损失最大）
- **全局负策略有助于抑制共性错误**
- **从混合策略 rollout 更贴近推理分布**

#### 表 5：负策略加权函数消融
比较不同方式计算负向优势权重：
| 加权方式 | Acc/B ↓ |
|--------|--------|
| 默认（soft conjunction） | 53.46 |
| Max | 52.43 |
| Mean | 53.12 |
| LogAvgExp | 52.62 |

> ✅ 当前设计最优，说明软合取（soft AND）更适合作为“任一失败即惩罚”的逻辑建模。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **多奖励对齐税源于奖励空间融合**：直接合并奖励会导致梯度干扰，降低整体性能。
2. **策略空间分解可缓解对齐税**：PRISM 通过分离优化路径，显著提升了多目标协同能力。
3. **推理时可控性强**：通过调节 $\alpha_k$ 和 $\gamma$，可在不重训的情况下灵活切换行为倾向（见 Figure 5）。
4. **样本效率更高**：PRISM 在约 3k 步内收敛，而基线需超过 6k 步（Figure 4）。
5. **随奖励数量增加更鲁棒**：当引入第三个奖励（如 length）时，PRISM 几乎不受影响，而 GDPO 和 GRPO 明显退化（Figure 3）。

### ⚠️ 方法的局限性
1. **假设限制**：
   - 每个奖励对应一个独立正向策略的有效性依赖于任务可分解性。
   - 全局负策略能否覆盖高维失败模式尚待更大规模验证。
2. **扩展成本**：
   - 虽然使用 prefix sharing 控制开销，但内存和 FLOPs 仍随奖励数线性增长（N+1 branches）。
3. **权重依赖人工设定**：
   - 当前 merge weights 需手动指定，缺乏自动适应 prompt 或 user 的机制。

### 🔮 未来工作方向
- 设计 **自适应权重机制**（adaptive merging），根据输入动态生成 $\alpha_k$。
- 探索更高效的 **mixture decoding 架构**，如稀疏激活或多专家路由。
- 扩展到 **更多奖励信号**（>3）和更复杂的交互场景（如 agent planning）。
- 引入 **可学习的负策略结构**，以更好建模跨奖励的复合失败模式。

---

## 总结

📌 **PRISM 提供了一种全新的多奖励对齐范式**：  
它跳出传统的“先融合奖励再优化策略”框架，转而采用“分别优化子策略、最后在策略空间融合”的思路，不仅提升了性能，还实现了前所未有的**训练后可控性**。

🎯 该方法在科学推理、工具使用和安全对齐三大任务上全面超越主流基线，验证了 **“Mix Policies, Not Rewards”** 的有效性与普适性，为构建真正灵活、可控、高性能的对齐语言模型提供了重要方向。

</details>

---

### 3. [Studying quantization trade-offs for efficient inference deployment in machine translation](https://arxiv.org/abs/2607.29397)

**Authors**: Jim Zhao, Sohir Maskey, Koen Oostermeijer, Douglas Orr, Teryn Jones  
**Category**: cs.CL  
**Published**: 2026-08-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.29397v1  

#### Abstract
Deploying large language models in realistic server environments poses challenges, as the system needs to provide high-quality responses with low latency. Quantization is a common approach to reduce the memory footprint and improve inference efficiency, yet its impact on latency and throughput is ra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Studying quantization trade-offs for efficient inference deployment in machine translation*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**大语言模型在机器翻译（MT）场景下的高效推理部署**问题展开研究，重点解决以下挑战：
- **量化（quantization）对推理效率的提升效果在真实服务负载下缺乏系统评估**，尤其是对端到端延迟（latency）和吞吐量（throughput）的影响。
- **传统基于句子级（segment-level）的MT评测基准（如WMT）无法反映长上下文文档翻译中的质量变化**，导致量化模型的真实表现被高估。
- **量化与文本分块策略（document chunking）之间的交互影响未被充分研究**，而该策略在实际部署中广泛用于处理长文档。

### 提出的新方法或新思路
1. **联合优化量化与文档分块策略**  
   首次在**闭合回路（closed-loop）在线推理环境**中，系统性地评估不同量化格式（W4A8, W8A8, W4A16）与文档分块长度的组合对推理效率的影响。

2. **引入文档级评估协议 WMT24++**  
   在标准 WMT24++ 数据集上构建**文档级输入**，通过控制目标提示长度（target prompt-length threshold），评估模型在长上下文下的翻译质量退化情况。

3. **揭示量化敏感性的模型差异**  
   发现不同MT模型家族对量化表现出显著不同的鲁棒性，提出应结合模型架构与训练数据来理解量化影响。

### 相比现有方法的优势
- 超越了仅关注内存或token-level指标的传统量化研究，**聚焦于真实部署场景下的端到端性能权衡**。
- 揭示了**segment-level benchmark的盲点**，强调必须在文档级别评估量化模型。
- 提供了**可操作的部署建议**：如推荐 W8A8/W4A8 + 200–400 token 分块作为帕累托最优配置。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主评估数据集**：`WMT24++`（Deutsch et al., 2025）
  - 包含平行文档，支持多语言对。
  - 选取四个翻译方向：`EN → DE`, `DE → EN`, `EN → RU`, `RU → EN`。
- **量化校准数据集**：从 `OPUS`（Tiedemann, 2016）采样1024个样本，其目标译文由 `Seed-X`（Cheng et al., 2025）生成。

### 实验设置
- **模型家族**：
  - `EuroLLM`（Martins et al., 2025）：1.7B, 9B, 22B 参数
  - `Hy-MT2`（Zheng et al., 2026）：1.8B, 7B 参数
- **硬件平台**：
  - 单张 `A100` 或 `H100` GPU
- **量化格式**：
  - `W8A8`：8-bit weights + 8-bit activations
  - `W4A8`：4-bit weights + 8-bit activations
  - `W4A16`：4-bit weights only
  - 基线：`BF16`
- **量化方法**：
  - 使用 `GPTQ` 进行后训练量化（PTQ）
  - 对 W8A8 和 W4A8 应用 `SmoothQuant` 减少激活异常值
- **推理引擎**：`vLLM`（支持 PagedAttention 和 Continuous Batching）

### 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **推理效率** | 平均输出吞吐量（tok/s）、p99 全文档延迟（latency）、吞吐量（documents/s） | 在离线和闭合回路在线基准中测量 |
| **翻译质量** | `xCOMET`（神经评估指标）、`chrF++`（统计指标） | xCOMET用于句子级，chrF++用于文档级重建后评估 |

### 基线方法对比
- 主要对比不同量化格式（W8A8, W4A8, W4A16）与 `BF16` 基线在：
  - 推理效率（throughput vs latency）
  - 文档级翻译质量（chrF++）
  - 不同 chunk size 下的表现

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 推理效率（图1, 图2）
- **中大型模型（≥9B）**：
  - `W8A8` 在 `A100` 上提供最佳帕累托前沿（Pareto frontier），尤其在高并发下。
  - `W4A8` 在 `H100` 上表现更优，得益于其更高的计算带宽比。
- **小型模型（<2B）**：
  - 量化带来边际收益甚至轻微减速（如 `EuroLLM-1.7B` on H100），因动态激活量化开销超过收益。
- **最优分块大小**：
  - **200–400 tokens** 的 chunk size 在多数部署场景下实现最佳效率-延迟权衡。
  - 低并发时偏好小 chunk（并行处理），高并发时需平衡 KV cache 管理。

#### ❌ 翻译质量（表1, 图3, 图8–10）
| 模型 | 量化格式 | chrF++ @ T=800 tok (vs BF16) | 质量下降 |
|------|----------|-------------------------------|--------|
| `Hy-MT2-7B` | W8A8 | 63.19 vs 63.18 | <0.1% |
| `EuroLLM-9B` | W8A8 | 33.70 vs 64.93 | ↓48.1% |
| `EuroLLM-9B` | W4A16 | 24.62 vs 64.93 | ↓62.1% |

- `Hy-MT2` 家族在所有量化格式下保持接近 BF16 的质量。
- `EuroLLM` 家族对量化极度敏感，尤其在长上下文（>400 tokens）下出现**质量崩溃**（quality collapse）。

#### 消融实验结果
- **分块策略影响**：
  - 较短 chunk（如 200 tokens）有助于维持翻译质量，但可能牺牲部分吞吐。
  - 最优 chunk 大小需在效率与质量间权衡。
- **量化格式对比**：
  - `W4A8` 在 H100 上优于 `W8A8`，但在 A100 上不支持。
  - `W4A16` 在高 batch size 下因计算瓶颈失去优势。

---

## 4. 关键结论和发现

### 主要发现
1. **量化并非总是有益**：
   - 小模型（<2B）可能因动态量化开销而变慢。
   - 仅当模型足够大且硬件支持低精度计算时，量化才显著提升效率。

2. **量化与模型架构强相关**：
   - `Hy-MT2` 对量化鲁棒，`EuroLLM` 极度敏感，表明**量化鲁棒性是模型特定属性**，与训练数据、上下文长度等有关。

3. **文档分块是关键调节器**：
   - 结合 `W8A8` 或 `W4A8` 与 **200–400 token 分块** 可显著改善延迟-吞吐帕累托曲线。

4. **传统评测严重误导**：
   - `xCOMET` 等句子级指标**无法预测长文档下的质量退化**。
   - 表1显示 `EuroLLM-9B` 在 WMT24++ 上仅下降 5.3% xCOMET，但在文档级 chrF++ 下下降超 48%，暴露严重盲点。

5. **失败模式分析（图11）**：
   - 量化版 `EuroLLM` 出现多种故障：
     - **拒绝响应**（Refusal）
     - **源文复制**（Source copying）
     - **错误语言摘要**（Wrong-language summarization）
     - **重复生成**（Degenerate repetition）
     - **返回助手回复而非翻译**

### 方法的局限性
- 仅使用 `GPTQ + SmoothQuant`，未探索其他 PTQ 或 `QAT` 方法是否能缓解 `EuroLLM` 的退化。
- 动态激活量化引入额外开销，静态量化可能更快但风险更高。
- 评估基于子集文档（N=11–20），需更大规模验证。
- 未考虑自适应分块策略或注意力掩码优化。

### 未来工作方向
- 开展**量化感知训练（QAT）或蒸馏**以提升 `EuroLLM` 类模型的量化鲁棒性。
- 设计**自适应分块策略**，根据负载、语言对动态调整 chunk 大小。
- 扩展至更多语言对，特别是低资源语言。
- 探索 `KV cache quantization` 以进一步降低长上下文内存压力。

---

> **一句话总结**：  
> 本文揭示了在机器翻译部署中，**量化收益高度依赖于模型家族、硬件平台与文档分块策略的协同作用**，并警告：**仅依赖句子级评测会严重低估量化对长文档翻译的质量损害**。

</details>

---

### 4. [ResKV: Reconstructing Omitted Attention Contributions for Fixed-Budget KV Cache Compression](https://arxiv.org/abs/2607.29591)

**Authors**: Yuhang Zhan, Lisi Chen, Shuo Shang  
**Category**: cs.CL  
**Published**: 2026-08-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.29591v1  

#### Abstract
KV cache compression is essential for efficient long-context inference. Existing eviction methods permanently discard unselected tokens and consequently remove their aggregate contribution to attention. Merging-based alternatives preserve more information but can perturb retained keys and values tha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ResKV: Reconstructing Omitted Attention Contributions for Fixed-Budget KV Cache Compression

## 1. 论文的主要贡献和创新点

### 解决的问题
在长上下文大语言模型（LLM）推理中，**KV Cache**（Key-Value Cache）用于避免重复计算历史 token 的 key 和 value，从而提升解码效率。然而，KV Cache 的大小随上下文长度线性增长，导致内存占用和带宽开销急剧上升。为应对这一挑战，研究者提出了多种 **KV Cache 压缩方法**，主要包括：

- **Eviction-based 方法**：通过评分机制选择保留部分 token，永久丢弃其余 token 及其对注意力的贡献。
- **Merging-based 方法**：将被丢弃 token 的信息合并到保留的 token 中。

这些方法存在显著缺陷：
- **Eviction** 完全丢失了被删 token 的注意力贡献，在注意力分布较分散时可能导致重要信息遗漏。
- **Merging** 虽保留信息，但会污染保留 token 的精确性，影响需要高精度检索的任务。

### 提出的新方法与创新思路
本文提出 **ResKV**，一种新颖的固定预算 KV Cache 压缩框架，其核心思想是：

> 将被丢弃 token 的注意力贡献视为“残差”（residual），并用一个紧凑的 **残差缓存（residual cache）** 来重建这部分信息，而非直接删除或合并。

具体创新点如下：

- **主-残差双路缓存架构（Main-Plus-Residual Layout）**：
  - 将固定的 KV Cache 预算 $b$ 分为两部分：$m$ 个槽位的 **main cache** 和 $r$ 个槽位的 **residual cache**（$b = m + r$）。
  - **Main cache** 存储高优先级 token 的原始 key 和 value，保持其作为精确记忆不变。
  - **Residual cache** 存储被丢弃 token 的聚合统计信息，以恢复其对注意力分子和分母的贡献。

- **共享 Softmax 残差解码（Shared-Softmax Residual Decode）**：
  - Main cache 和 residual cache 的条目共同参与同一个 softmax 归一化过程。
  - 这使得残差条目能够同时恢复注意力的 **numerator mass**（值贡献）和 **denominator mass**（归一化权重），而不仅仅是后处理修正。

- **自适应残差控制（Adaptive Residual Control）**：
  - **构建时验证代理（Construction-time Validation Proxy）**：在预填充（prefill）阶段，通过一个小的验证集判断是否分配残差预算，仅当能提升注意力输出重构效果时才启用。
  - **解码时动态门控（Decode-time Dynamic Gate）**：根据当前查询在 main cache 上注意力的尖锐程度（sharpness）动态调整残差项的权重。当注意力集中时抑制残差，防止干扰；当注意力分散时增强残差，恢复聚合贡献。

### 相比现有方法的优势
- **不牺牲精确性**：保留的 token 在 main cache 中保持原样，避免了 merging 方法的污染问题。
- **不完全丢失信息**：通过残差缓存重建被丢弃 token 的集体贡献，克服了 eviction 方法的信息损失。
- **高效且实用**：在相同 KV Cache 预算下运行，保持了压缩解码的实际效率，峰值内存和长上下文吞吐量稳定。

---

## 2. 核心实验方法和设置

### 数据集
在两个主流的长上下文基准上进行了全面评估：
- **LongBench**：涵盖 16 种真实世界的长上下文理解任务，包括单/多文档问答、摘要、少样本学习、合成检索和代码补全。
- **RULER**：提供受控的长上下文测试，专注于检索和聚合行为，包含 13 个任务，在 4K 和 32K 上下文长度下进行评估。

### 实验设置与评估指标
- **模型**：在两个指令微调的骨干模型上评估：`LLaMA-3.1-8B-Instruct` 和 `Qwen-2.5-7B-Instruct`。
- **压缩比例**：测试了 $p \in \{0.6, 0.7, 0.8, 0.9\}$，即保留 40%、30%、20%、10% 的 KV 槽位。
- **构造模式**：同时评估了 **query-aware**（已知后续查询）和更具现实意义的 **query-agnostic**（未知后续查询，模拟前缀缓存复用）两种设置。
- **评估指标**：报告各数据集的平均得分（Average Score），以及详细的任务级结果。

### 基线方法对比
与以下代表性压缩方法进行了比较：
- **H2O**：基于累积注意力质量保留 token。
- **SnapKV**：基于观察窗口估计 token 重要性。
- **TOVA**：基于近期注意力评分。
- **AdaKV**：跨注意力头自适应分配缓存预算。
- **CaM**：一种 merging-based 方法，将被丢弃状态折叠回保留条目。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **在 LongBench 上**：ResKV 改善了所有 **32 个**展示的配置，平均提升 **1.02 分**。
- **在 RULER 上**：ResKV 改善了 **63 个**（共 64 个）展示的配置，平均提升 **3.38 分**。
- **在紧约束预算下优势更明显**：例如在 10% 保留率下，相比基线平均提升高达 **3.47–3.66 分**。
- **在 query-agnostic 设置下提升更大**：这表明 ResKV 特别适合实际部署场景。

### 与基线方法的对比结果
- **全面超越**：在绝大多数配置下，ResKV 的平均得分均优于对应的基线方法（如 `SnapKV + ResKV` vs `SnapKV`）。
- **尤其擅长分布式证据任务**：在需要整合分散信息的任务（如代码、检索、变量追踪）上表现最佳，因为残差缓存能有效聚合这些被丢弃 token 的贡献。
- **效率保持**：如图 3 所示，ResKV 的峰值内存占用与基线（如 SnapKV）几乎重合，解码吞吐量在长上下文下也保持稳定，远优于 full cache（后者在长上下文下 OOM 或严重降速）。

### 消融实验结果
消融研究验证了 ResKV 各组件的重要性：
- **移除验证代理（w/o validation）**：在多个任务上导致 4–7 分的性能下降。
- **移除动态门控（w/o dynamic gate）**：导致 4.5–4.8 分的下降，证明了动态调整残差强度的必要性。
- **移除共享 Softmax（w/o shared softmax）**：当改为独立归一化时，性能显著下降（如 RepoBench-P 下降 3 分），凸显了联合归一化对恢复完整注意力贡献的关键作用。

---

## 4. 关键结论和发现

### 主要发现
1. **被丢弃的注意力贡献可建模为残差**：硬删除或合并都不是最优解，将其视为可重建的残差信息是一种更优范式。
2. **主-残差分离设计有效**：将精确记忆（main cache）与近似聚合（residual cache）分离，既保证了关键信息的保真度，又恢复了全局上下文。
3. **共享 Softmax 是关键**：让残差条目参与主分支的归一化，使其成为真正的注意力参与者，而非简单的后处理修正。
4. **自适应控制至关重要**：验证代理确保资源只用在刀刃上，动态门控则实现了对不同查询模式的鲁棒响应。

### 方法的局限性
- **评估范围有限**：目前仅在两个 GQA 架构的指令模型和两个基准上进行了评估。
- **静态构建**：残差缓存仅在预填充后构建一次，未在生成过程中更新。对于超长生成任务，可能错过新生成 token 的信息。
- **残差表示简单**：当前使用的是基于聚类的均值 key/value 表示，更丰富的表示方式可能进一步提升性能。

### 未来工作方向
- **扩展模型和场景**：在更大的模型、不同的架构（如 MHA）、批量推理等更广泛的场景下评估 ResKV。
- **动态残差更新**：研究在生成过程中轻量级地更新残差缓存的方法。
- **改进残差表示**：探索更强大的残差摘要技术（如学习型表示）和专用内核以降低解码开销。

</details>

---

### 5. [BLADE: Boundary-Expanded and Layer-Adaptive Dynamic Exit for Efficient LLM Reasoning](https://arxiv.org/abs/2607.28966)

**Authors**: Keshu Fu, Keqin Peng, Jun Bai, Shuhan Qin, Chen Li, Junzhu Liang, Yefei Chen, Jiaqi Li, Yuanxin Ouyang  
**Category**: cs.CL  
**Published**: 2026-08-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.28966v1  

#### Abstract
Large language models often improve task performance by generating long reasoning traces, but the resulting computation is frequently wasted on redundant verification and revision. Existing probe-based early-exit approaches mainly inspect explicit self-doubt expressions, leaving many earlier termina...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**BLADE: Boundary-Expanded and Layer-Adaptive Dynamic Exit for Efficient LLM Reasoning**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在进行复杂推理时通常生成冗长的 **Chain-of-Thought (CoT)** 推理链，以提升任务表现。然而，这种扩展推理常导致“**overthinking**”现象——即模型在已得出正确答案后仍继续验证、反思或重复计算，造成不必要的计算开销，并可能降低最终答案质量。

现有的基于 **probe-based early exit** 的方法主要依赖显式的“自我怀疑”信号（如 “Wait”, “Let me reconsider”）作为退出检查点，但这类信号稀疏且滞后，会错过许多更早的终止机会。

---

### 🚀 提出的新方法：BLADE 框架

作者提出 **BLADE**（Boundary-Expanded and Layer-Adaptive Dynamic Exit），一种轻量级、高效的动态退出框架，用于优化 LLM 推理过程。其核心思想是将早期退出建模为 **prefix-sufficiency prediction**：判断当前生成的推理前缀是否已足以支持正确的最终答案。

#### 主要创新点包括：

| 创新模块 | 内容说明 |
|--------|--------|
| **Multi-Granular Reasoning Checkpoints (MGRC)** | 扩展检查点类型，不仅使用 self-doubt 边界，还引入 **sentence boundary** 和 **paragraph boundary**，实现多粒度、更密集的检测覆盖，从而捕捉更多潜在的充分推理状态。 |
| **Adaptive Probe-Layer Selection (APLS)** | 不同推理阶段的信息分布在不同隐藏层中。BLADE 自动学习一个紧凑的 informative layer 子集，避免手动选择或昂贵的全层拼接（all-layer concatenation），显著降低参数和内存开销。 |
| **Checkpoint-Aware Stopping Policy** | 针对不同类型检查点采用差异化停止策略：<br>- **self-doubt checkpoint**：高置信度即可立即退出；<br>- **sentence checkpoint**：需连续两次确认才允许退出，防止因语义不稳定导致的 premature exit。 |

---

### 🔍 相比现有方法的优势

| 对比维度 | BLADE 优势 |
|--------|-----------|
| **覆盖率更高** | 超越仅依赖 self-doubt 的方法（如 LYNX），通过 sentence-level 检查点提前识别充分推理状态。 |
| **效率更高** | APLS 选出的 compact layer subset 显著减少 probe 参数量（↓64%）、峰值内存（↓85%）、训练时间（↓90%）。 |
| **适应性强** | 层选择自动完成，不依赖固定层设定，适用于不同 backbone 模型。 |
| **准确率保持更好** | 在大幅减少 token 输出的同时，几乎不损失 accuracy，实现更优的 **accuracy-efficiency trade-off**。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在五个数学推理 benchmark 上进行评估：
- **GSM8K-test**
- **MATH-500**
- **AMC 2023**
- **AIME 2024**
- **AIME 2025**

共包含 1,919 个问题，划分为 192 个校准样本 + 1,727 测试样本。

---

### 🧪 实验设置

| 组件 | 设置详情 |
|------|---------|
| **Backbone 模型** | Qwen3-8B 和 Qwen3-4B |
| **训练数据** | 使用 6,000 个训练样本（来自 GSM8K train、MATH train 数值子集、DeepScaleR train）训练 probe 模块 |
| **Probe 训练方式** | 采用 K16 strict-clean supervision（强制完成 16 次取一致结果作为标签），确保低噪声标签 |
| **Calibration** | 在独立校准集上估计 conformal threshold λ，控制 early-exit 的严格程度 |
| **推理流程** | 动态监控 sentence / self-doubt 检查点 → 应用 APLS 选中的 layer 进行 sufficiency 预测 → 根据 checkpoint 类型执行相应 stopping 规则 |

---

### 🎯 评估指标

| 指标 | 定义与用途 |
|-----|----------|
| **Accuracy (%)** | 最终答案正确率 |
| **#Tok. (Generated Tokens)** | 平均每条输出生成的 token 数量，衡量推理成本 |
| **Accuracy-Efficiency Score (AES)** | 综合评价指标：<br>$$
\text{AES} =
\begin{cases}
\frac{L_b - L}{L_b} + \frac{p - p_b}{p_b}, & p \geq p_b \\
\frac{L_b - L}{L_b} + \frac{p_b - p}{p_b}, & p < p_b
\end{cases}
$$<br>其中 $L$ 和 $p$ 是方法的 token 数与 accuracy，$L_b$, $p_b$ 是 Full-CoT 基线值。<br>AES 越高表示 accuracy 与 efficiency 平衡越好。 |

---

### 🆚 基线方法对比

| 基线方法 | 描述 |
|--------|------|
| **Full-CoT** | 不启用 early exit，完整生成整个 CoT |
| **LYNX-K1 / LYNX-K16** | 基于 self-doubt 检查点 + 固定 layer 集合的 early exit 方法；K16 使用多次采样获得更干净标签 |
| **Various Layer Selection Ablations** | 包括 final layer、best/worst single layer、random/fixed four-layer、adjacent-middle、evenly-spaced、all-layer 等策略，用于验证 APLS 的有效性 |

---

## 3. 主要实验结果和性能指标

### 📊 总体性能对比（Table 1）

| Method | Avg Acc (%) | Avg #Tok. | Token Reduction | AES |
|-------|-------------|-----------|------------------|-----|
| **Qwen3-8B Base (Full-CoT)** | 76.8 | 7,837 | — | 0.000 |
| **BLADE (Ours-Mixed)** | **75.2** | **5,896** | **↓24.8%** | **0.213** |
| **LYNX-K16** | 76.2 | 6,520 | ↓16.8% | 0.188 |
| **LYNX-K1** | 76.2 | 6,650 | ↓15.2% | 0.163 |

> ✅ BLADE 在 Qwen3-8B 上实现了 **最高 AES (0.213)**，token 减少近四分之一，同时 accuracy 下降极小（仅 1.6%）。

| Method | Avg Acc (%) | Avg #Tok. | Token Reduction | AES |
|-------|-------------|-----------|------------------|-----|
| **Qwen3-4B Base (Full-CoT)** | 75.8 | 7,618 | — | 0.000 |
| **BLADE (Ours-Mixed)** | **75.6** | **6,414** | **↓15.8%** | **0.175** |
| **LYNX-K16** | 75.6 | 6,843 | ↓10.1% | 0.109 |

> ✅ 在较小模型 Qwen3-4B 上同样有效，token 减少 15.8%，AES 显著优于所有 baseline。

---

### 🔬 消融实验结果（Table 2 & Figure 4）

#### （1）Layer Selection Ablation 结果

| 方法 | Qwen3-8B AES | Qwen3-4B AES |
|------|--------------|--------------|
| **BLADE (APLS-selected K=4)** | **0.213** | **0.175** |
| Best single layer | 0.100 | 0.075 |
| Worst single layer | -0.162 | 0.024 |
| All layers | 0.103 | 0.147 |
| Random K=4 | 0.166 | 0.151 |
| LYNX-K4 (fixed) | 0.199 | 0.169 |

> 💡 发现：
> - 单一层表现差且敏感，最佳与最差层差距巨大；
> - 全层拼接反而不如 compact subset，说明存在冗余特征；
> - APLS 自动选择的 layer subset 表现最优，无需人工调参。

#### （2）Runtime Policy Ablation（Figure 4）

- **Mixed vs Doubt-only 检查点流**：
  - 加入 sentence boundary 后，在相同 accuracy 下可节省更多 tokens。
- **Asymmetric stopping (self-doubt: immediate, sentence: consecutive-2)**：
  - 显著优于统一立即退出策略，有效抑制 premature exit。

> ✅ 结论：**混合检查点 + 差异化停止策略** 是实现高效安全退出的关键组合。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Self-doubt-only 检查点覆盖不足**  
   许多充分推理状态出现在普通句子边界处，早于任何“自我怀疑”表达出现，因此必须扩展检查点粒度。

2. **不同推理状态的信息分布深度不同**  
   有些 sufficiency 信号集中在浅层，有些在深层，固定 layer 或单一 layer 难以泛化。APLS 可自适应地提取最具判别力的 layer 组合。

3. **紧凑 layer subset 可媲美甚至超越全层模型**  
   尽管 APLS 仅使用 4 层，但其性能优于 all-layer 模型，证明了信息冗余的存在以及压缩的有效性。

4. **BLADE 实现显著推理加速而不牺牲准确性**  
   在两个 Qwen3 模型上分别减少 **24.8%** 和 **15.8%** 的生成 token，同时 accuracy 基本持平，展现出卓越的 efficiency-accuracy trade-off。

---

### ⚠️ 方法局限性

1. **依赖高质量 sufficiency 标签构建**  
   需要多次 forced completion 来获取一致标签，增加了预处理成本。

2. **APLS 层选择具有随机性**  
   多次运行间 selected layer 集合重叠度低（Jaccard ~0.12），表明解空间非唯一，虽性能稳定但缺乏可解释性。

3. **目前仅应用于数学推理任务**  
   是否能推广到其他领域（如常识推理、代码生成）尚待验证。

---

### 🔮 未来工作方向

1. **探索更通用的 sufficiency labeling 策略**  
   减少对多次采样的依赖，例如结合 reward modeling 或 consistency scoring。

2. **将 APLS 扩展至多任务或多领域场景**  
   设计跨任务共享的 layer selection 机制。

3. **集成到训练过程中实现端到端优化**  
   当前 probe 是后训练附加模块，未来可尝试 joint learning 方式进一步提升性能。

4. **研究 BLADE 与其他推理优化技术的协同效应**  
   如与 pruning、distillation、speculative decoding 结合，打造更高效的推理系统。

---

## ✅ 总结

**BLADE** 是一项针对 LLM 推理冗余问题的重要进展。它通过 **扩展检查点粒度**、**自适应选择 probe layer** 和 **设计 checkpoint-aware 停止策略**，实现了在几乎不损失 accuracy 的前提下，显著减少推理 token 消耗。其实验充分、设计精巧，在多个 benchmark 和 backbone 上均表现出色，为高效 LLM 推理提供了新的范式。

</details>

---

### 6. [Curriculum Matters: Data-Efficient Relational PFN Pretraining with Synthetic Data](https://arxiv.org/abs/2607.29120)

**Authors**: Mohammad Sadeq Abolhasani, Viswanath Ganapathy  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.29120v1  

#### Abstract
Relational Prior-Data Fitted Networks (PFNs) such as RDB-PFN approximate Bayesian inference over multi-table relational databases by pretraining on millions of synthetic tasks. We investigate three intertwined questions about this paradigm. First, can a structurally different synthetic generator Plu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Curriculum Matters: Data-Efficient Relational PFN Pretraining with Synthetic Data*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前**Relational Foundation Models**（如 RDB-PFN）依赖于大规模、高质量的合成多表关系数据库进行预训练，但这类数据生成成本高、计算资源消耗大，且现有方法未系统研究**训练策略**（如 curriculum 设计）对最终性能的影响。本文聚焦以下三个核心问题：

- **Q1**: 是否必须使用 RDB-PFN 自研的复杂神经生成器？结构不同的合成数据生成器是否也能达到相近效果？
- **Q2**: 合成数据的**呈现顺序**（即 curriculum）是否显著影响下游任务表现？
- **Q3**: 在仅用单表数据预训练的情况下，模型能否获得一定的**关系推理能力**？

### 提出的新方法与思路
提出并验证了一种以 **curriculum learning** 为核心的高效预训练范式，其核心思想是：

- 使用结构上完全不同的轻量级合成数据生成器 **PLuREL** 替代 RDB-PFN 原生的复杂生成器；
- 构建**渐进式宽度 curriculum**（progressive width curriculum），从简单 schema（列数少）逐步过渡到复杂 schema；
- 探索**单表预训练向关系任务迁移的能力**，挑战“必须进行专门的关系数据微调”的假设。

### 相比现有方法的优势
- **数据效率极高**：在单表阶段使用约 **45× 更少的数据量**，在关系阶段使用 **220× 更少的合成数据库**，即可恢复 RDB-PFN 88% 的性能；
- **不依赖专用生成器**：证明了 PLuREL 这类基于随机图和块模型的非学习型生成器足以支撑高性能 PFN 预训练；
- **揭示训练轨迹的重要性**：表明 curriculum ordering 是比生成器设计或数据总量更关键的因素；
- **简化流程潜力大**：单表 curriculum 模型可直接迁移到关系任务，几乎无需额外训练。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **合成数据生成器**：全程使用 **PLuREL** 作为唯一的数据源，生成单表和多表关系数据库。
- **真实世界基准测试集**：
  - **单表任务**：来自 Grinsztajn et al. [1] 的 **23 个分类任务**（如 `higgs`, `bank-marketing`, `covertype` 等）。
  - **关系任务**：结合 **RelBench** [10] 和 **4DBInfer** [11] 的 **19 个关系预测任务**（如 `rel-f1/driver-top3`, `stackexchange/upvote` 等）。

### 实验设置与评估指标
#### 数据处理
- 所有关系数据库通过 **Deep Feature Synthesis (DFS)** 线性化为单表特征矩阵；
- 固定输出特征维度为 30 列（通过随机采样）；
- 目标变量按中位数二值化（数值）或 one-vs-rest（类别）生成。

#### 模型架构
- 使用与 RDB-PFN 相同的 **PFN 架构**：一个小型双向 Transformer（~0.7M 参数，6 层，d_model=128）；
- 分类头用于二分类任务；
- 上下文长度（context size）设为 **64 和 1024**。

#### 评估指标
- 主要指标：**ROC-AUC**（平均 across 所有任务）
- 报告 context size 为 64 和 1024 下的表现

### 基线方法对比
- **RDB-PFN (paper)**：原始论文报告的结果，使用 ~600K 单表 + ~1.2M 多表合成任务；
- **All-at-once training**：相同数据但无 curriculum 顺序；
- **Random initialization vs. warm-up**：检验单表预热的重要性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 实验配置 | 数据规模 | Avg. ROC-AUC (ctx 1024) | 相对于 RDB-PFN 的恢复率 |
|--------|---------|--------------------------|----------------------------|
| **Family A**: 单表渐进 curriculum (TF07–TF17) | ~13,300 tables | **0.703** | 88% of RDB-PFN (0.800) |
| **Family B**: 单表 all-at-once | ~13,300 tables | 0.541 | ——（下降 16.2 pts） |
| **Family C**: 关系 curriculum（从零开始） | ~5,500 DBs | **0.638** | 88% of RDB-PFN (0.725) |
| **Family D**: 单表模型直接评估于关系任务 | 0 relational DBs | **0.631** | 87% of RDB-PFN |
| **RDB-PFN (reported)** | ~1.8M synthetic tasks | 0.725–0.800 | 100% |

> 注：Family A 使用的数据量仅为原 RDB-PFN 单表 warm-up 的 **~2.2%**（约 270× 更少总行数）

### 与基线方法的对比结果
- **仅用 45× 更少的单表数据**，通过 curriculum 学习即可达到 RDB-PFN 88% 的性能；
- **仅用 220× 更少的关系合成数据**，从零开始训练仍能恢复 88% 性能；
- **单表模型未经任何关系训练**，在关系任务上表现接近专精 pipeline（0.631 vs. 0.638）；
- 使用 **PLuREL** 替代 RDB-PFN 原生生成器，性能损失极小（<12%），说明生成机制本身不是瓶颈。

### 消融实验结果

| 消融配置 | Avg. ROC-AUC (ctx 1024) | 发现 |
|--------|--------------------------|------|
| **Family B (all-at-once)** | 0.541 | 相同数据下，无 curriculum 导致性能崩溃（↓16.2 pts） |
| **Family E (warm-up + relational all-at-once)** | 0.620 | 缺乏关系 curriculum 仍优于无 warm-up，但低于 structured curriculum |
| **Family F (no warm-up, all-at-once)** | 0.596 | 无单表预热造成最大性能损失（↓4.2 pts vs. Family C） |
| **Family G (only relational, no single-table)** | 0.624 | 即使没有单表数据，也能取得不错性能，支持独立学习路径存在 |

> ✅ **关键洞见**：  
> - **Curriculum ordering > 数据总量 > 生成器设计**
> - 单表 warm-up 比关系 curriculum 更重要
> - 不良的关系训练阶段甚至会损害模型表现（Family E < Family D）

---

## 4. 关键结论和发现

### 主要发现
1. 🔑 **Curriculum 是主导因素**：  
   数据呈现的**顺序**（由简到繁的 progressive curriculum）对性能影响巨大。相同的 ~13,300 张表，**有 curriculum 达 0.703，无 curriculum 仅 0.541**，差距达 **16.2 个百分点**。

2. 🔄 **Prior Generator 可替代性强**：  
   尽管 **PLuREL** 与 RDB-PFN 使用完全不同的生成机制（随机图 vs 学习型 DAG；局部 SCM vs 全局 GNN），但其生成的数据仍能支撑高性能训练，说明真正重要的是合成数据的**分布特性**（如 DFS 特征的相关结构），而非具体生成方式。

3. 🧠 **单表预训练蕴含关系推理能力**：  
   经过良好设计的单表 curriculum 训练后，模型已学会解释 DFS 线性化后的结构化特征模式，因此可在**从未见过多表数据库的前提下**，直接在关系任务上取得近似专精模型的效果（0.631 vs 0.638）。

4. ⚖️ **两阶段流程中，单表 warm-up 更关键**：  
   移除 warm-up（Family F）造成的性能下降（↓4.2 pts）远大于移除关系 curriculum（↓1.8 pts），说明基础表征能力的建立比后续关系细化更重要。

5. 💡 **DFS 线性化大幅降低了关系学习难度**：  
   DFS 将复杂的多表结构压缩为具有特定统计模式的宽表格，使得许多关系任务本质上退化为“带结构先验的 tabular learning”，从而让强大的 PFN 能够泛化。

### 方法的局限性
- 当前实验集中在 **binary classification** 任务，未覆盖 regression 或 multi-class 场景；
- 所有任务均经 DFS 处理，无法反映端到端处理原始 schema 的能力；
- PLuREL 虽轻量，但在模拟某些真实业务逻辑（如强时间依赖、复杂约束）方面仍有不足；
- Per-task variance 较高，部分任务（如 event-frequency aggregation）表现较差，提示需要 task-aware curriculum。

### 未来工作方向
- 设计 **task-structure-aware curriculum**，根据任务类型动态调整训练顺序；
- 探索更大容量 backbone（如 TABICLv2）与 curriculum 的协同效应；
- 研究如何将单表与关系 curriculum 更有效地融合，减少冗余训练；
- 开发更贴近企业实际的语义丰富 synthetic schema families；
- 探索 zero-shot 迁移至未见 schema 结构的能力。

---

> ✅ **一句话总结**：  
> 本论文颠覆了“大规模专用合成数据 + 复杂生成器”是构建 Relational PFN 的必要条件的传统认知，证明了 **精心设计的 curriculum** 才是实现**高效、低资源关系预训练**的关键杠杆。

</details>

---

### 7. [Frugal Bayesian Optimization: Scalable Surrogates for Data- and Resource-Limited Discovery](https://arxiv.org/abs/2607.29225)

**Authors**: Panagiotis Krokidas, Christoforos Rekatsinas, Vassilis Sioros, Grigorios M. Chatziathanasiou, Efi-Maria Papia, George Giannakopoulos  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.29225v1  

#### Abstract
Bayesian Optimization (BO) is widely adopted for data-efficient optimization in scientific and engineering applications, yet its computational cost is rarely evaluated alongside optimization performance. Here we present a systematic, compute-aware study of BO that evaluates surrogate models along tw...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Frugal Bayesian Optimization: Scalable Surrogates for Data- and Resource-Limited Discovery*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统地探讨了 **Bayesian Optimization (BO)** 在实际应用中的一个被长期忽视的问题：**计算成本与资源消耗**。尽管 BO 因其高样本效率而广受赞誉，但其主流依赖的 **Gaussian Process (GP)** 由于 $O(N^3)$ 的训练时间和 $O(N^2)$ 的内存占用，在大规模或长时间运行中变得不可持续，尤其在硬件受限的研究环境中。

传统研究往往只关注“找到最优解需要多少次采样”，却忽略了“每次采样后模型更新要花多少时间与内存”。这种忽略导致在真实场景下，GP-based BO 可能因计算瓶颈而提前终止，反而不如更轻量的方法高效。

### 提出了什么新方法或新思路
作者提出了 **FruBO (Frugal Bayesian Optimization)** 框架，其核心思想是将 BO 的评估标准从单一的“样本效率”扩展为双轴评价体系：

- **优化质量 (Optimization Quality)**：如最佳目标值、Top-100 解的召回率（recall@100）
- **计算节俭性 (Computational Frugality)**：包括 wall-clock time 和 GPU memory usage 随采样次数的增长趋势

在此基础上，作者提出了一种 **基于数据特征的 surrogate model 推荐系统**：
- 利用少量易获取的数据集特征（如 dataset size、dimensionality、fractal dimension、target variance）来预测四种 surrogate 模型（GP, RF, NGBoost, BASS）的表现排名。
- 构建两个多输出分类器，分别推荐在 **compute-limited** 和 **sample-limited** 场景下的最优 surrogate。

### 相比现有方法的优势
- **首次系统量化了 BO 中的计算开销**，揭示了 GP 的高成本与其优化表现不成正比。
- **证明了非 GP surrogate 在多数情况下优于或等同于 GP**，同时具有显著更低的时间和内存开销。
- **提供了可操作的工具（推荐系统）**，帮助研究人员根据自身预算选择最合适的 surrogate，避免盲目使用 GP。
- 所有代码开源，推动科学机器学习领域建立 **compute-aware benchmarking** 的新标准。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验涵盖两类共 **17 个数据集**：

#### （1）基准函数（Benchmark Functions, 8个）
- Rastrigin, Ackley, Schwefel, Michalewicz, Schaffer7, Styblinski-Tang, Weierstrass, Expanded Schaffer F6  
- 特点：解析形式已知、全局最优明确、控制变量强，用于验证基础性能。

#### （2）真实世界问题（Real-Case Datasets, 9个）
覆盖多个科学工程领域：
- **材料科学**：COFs（甲烷存储）、Moire 多层材料（孔径均匀性）
- **力学与结构设计**：Bouligand 压力容器、C. elegans 蠕虫压痕力学
- **机器人控制**：LunarLander 连续动作序列优化
- **药物发现**：Ro4 分子对接（binding energy 最小化）
- **量子化学**：QM9 数据集（最大化 HOMO-LUMO gap）
- **机器学习超参调优**：MLP 超参数网格搜索
- **机电系统**：Multi-PZT 半主动减振器（multi-modal vibration 抑制）

这些数据集具有高维、非光滑、噪声大、规模大的特点，贴近现实科研挑战。

---

### 实验设置和评估指标

#### 实验设置
- 总采样预算统一设为 **N = 1000 次 acquisition**。
- 所有 surrogate 模型均集成于 **BoTorch** 框架下进行公平比较。
- 每个任务重复运行 20 次以统计方差。
- 硬件平台：Intel i9-10900K + 64GB RAM + RTX 3070 Ti (8GB VRAM)

#### 评估指标（双轴并重）
| 维度 | 指标 |
|------|------|
| **优化性能** | - Best-found objective value vs. #samples<br>- Recall@100 (识别出前100个最优解的数量) vs. #samples |
| **计算效率** | - Training time (wall-clock) vs. #acquisitions<br>- GPU memory usage vs. #acquisitions |
| **综合评分** | 使用 **Area Under Curve (AUC)** 对上述四条曲线（best-so-far vs samples/time, top-100 vs samples/time）进行量化，并通过乘积形成复合指标：<br>$$ \text{Performance Score} = \text{AUC}_{\text{best-vs-time}} \times \text{AUC}_{\text{top100-vs-time}} $$ |

---

### 基线方法对比
比较了四种 surrogate models：
| Model | 类型 | 主要优势 | 主要劣势 |
|-------|------|--------|--------|
| **Gaussian Process (GP)** | 贝叶斯非参数模型 | 理论上提供校准良好的不确定性估计 | $O(N^3)$ 时间复杂度，$O(N^2)$ 内存增长，难以扩展 |
| **Random Forest (RF)** | 集成树模型 | 训练快、内存稳定、对异常值鲁棒 | 不确定性估计较粗糙 |
| **NGBoost (NGB)** | 概率梯度提升框架 | 支持完整概率预测，样本效率高 | 默认配置较重，需调整以适应 BO 循环 |
| **BASS** | 基于自适应样条的贝叶斯回归 | 支持非平稳性、可解释性强、支持敏感性分析 | 实现相对小众，计算开销高于 RF/NGB |

> 注：所有模型均采用轻量化配置以适配迭代 BO 流程（如减少 MCMC 步数、降低 estimator 数量），强调“实用性”而非“理论极限”。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）计算效率方面（Time & Memory）
- **GP 明显劣于其他三种 surrogate**：
  - 训练时间随 $N$ 呈 **超线性增长**（接近 $O(N^{2.5})$~$O(N^3)$）
  - GPU memory 随 $N$ **线性累积**，在 N≈800–1000 时逼近 8GB 上限，导致崩溃风险
- **RF、NGB、BASS 表现出近似线性时间增长和恒定内存占用**：
  - 内存几乎不随 $N$ 增加，始终保持在几百 MB 级别
  - 单次训练时间远低于 GP（通常快 10–100 倍）

> 图 1 显示，在所有 benchmark 函数上，GP 的 wall-clock time 和 GPU memory 增长斜率显著更高。

#### （2）优化性能方面（Best-found & Top-100 Recall）

##### 在 **benchmark functions** 上的结果（见 Table 1）：
| 维度 | 最佳表现者 |
|------|----------|
| **Compute-time performance** | **RF**（8项中赢5项） > NGB > BASS > GP |
| **Sample-acquisition performance** | **NGB**（6项中赢5项） > RF > BASS > GP |

> GP 在样本效率上普遍垫底，仅在极少数函数（如 Schaffer7）中略有优势。

##### 在 **real-case datasets** 上的结果（见 Table 1）：
| 维度 | 最佳表现者 |
|------|----------|
| **Compute-time performance** | **RF**（5/9） > BASS ≈ NGB > GP |
| **Sample-acquisition performance** | **NGB**（5/9） > RF > BASS > GP |

> GP 仅在 Moiré 和 worm 数据集中因问题结构匹配其平滑假设而表现尚可，其余场景全面落后。

#### （3）消融实验与推荐系统效果
- 构建了一个基于 **dataset characteristics** 的 surrogate ranking classifier。
- 输入特征：dataset size, dimensionality, fractal dimension, target variance（仅需前 200 次采样即可估算）
- 输出：四个 surrogate 在 compute-time 或 sample-efficiency 上的预期排名
- 评估指标：**Normalized Discounted Cumulative Gain (nDCG@p)**

##### 结果：
- **平均 nDCG > 0.85**，表明推荐系统能高度准确地预测 surrogate 排名
- 在 **compute-efficiency ranking** 上表现更强（nDCG 更集中于 1.0），说明计算行为更具规律性和可预测性
- 成功将经验观察转化为自动化决策工具，极大降低试错成本

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **GP-based BO 并非默认最优选择**：它在绝大多数任务中既慢又耗内存，且并未带来相应的优化性能提升。
2. ✅ **非 GP surrogate 完全可以替代 GP**：特别是 **Random Forest** 和 **NGBoost**，在保持甚至超越优化能力的同时，实现了数量级的计算加速和内存节省。
3. ✅ **没有万能的 surrogate**：不同 surrogate 各有擅长场景：
   - **RF**：适合 compute-limited 场景（快速迭代）
   - **NGB**：适合 sample-limited 场景（昂贵实验/模拟）
   - **BASS**：适合需要解释性和敏感性分析的任务
4. ✅ **surrogate selection 应成为 BO 工作流的一部分**：应根据具体任务的数据特征和资源约束动态选择，而非固定使用 GP。

### 方法的局限性
- 当前仅评估了四种 surrogate，未包含其他新兴方法（如 sparse GP、deep kernel learning、neural processes）。
- 推荐系统的输入特征较为简单，未考虑目标函数的局部结构、噪声模式等高级特征。
- 所有实验基于静态 acquisition function（如 EI），未探索 surrogate 与 acquisition 的联合优化。
- 推荐模型是在本研究的数据集上训练的，泛化到全新领域仍需验证。

### 未来工作方向
- 将 FruBO 框架扩展至更多 surrogate 模型（如 DNGO、Deep Ensembles、Neural Processes）。
- 引入更丰富的 landscape descriptors（如 smoothness, multimodality, noise level）以增强推荐精度。
- 开发动态 surrogate 切换机制：在 BO 过程中根据性能反馈自动更换模型。
- 探索 **acquisition-aware surrogate design**：即设计专门服务于 acquisition function 更新的轻量代理模型。
- 推动社区采纳 **compute-aware evaluation protocol**，将 time/memory 纳入标准 benchmarking 流程。

---

> 📌 **一句话总结**：  
> 本文颠覆了“BO 必须用 GP”的固有认知，通过大规模实证证明：**更轻量的 surrogate（如 RF 和 NGBoost）不仅能跑得更快、吃得更少，还能找到更好或同样好的解**。FruBO 不仅是一个新方法，更是一种倡导 **resource-aware, sustainable AI for science** 的新范式。

</details>

---

### 8. [OnlineCache: Learning Dynamic Caching Policies with Error Correction for Efficient Diffusion Inference](https://arxiv.org/abs/2607.29398)

**Authors**: Zhikang Xie, Xichen Ye, Yifan Wu, Haoshen Yu, Li chenan, Peizhu Gong, Weizhong Zhang, Cheng Jin  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.29398v1  

#### Abstract
Diffusion models have revolutionized generative tasks but incur high latency due to iterative denoising. While cache-based strategies accelerate inference by reusing intermediate features, they largely rely on static, sample-agnostic schedules. We argue that this rigidity overlooks two facts empiric...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：OnlineCache: Learning Dynamic Caching Policies with Error Correction for Efficient Diffusion Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **cache-based acceleration** 方法在加速扩散模型（Diffusion Models）推理时存在两大局限：
- **Sample-level heterogeneity**：不同输入提示（prompt）的生成难度差异大，复杂输入需要更多计算，简单输入则可节省资源，但静态缓存策略（如固定步长缓存）无法自适应分配算力。
- **Timestep-level heterogeneity**：去噪过程中的误差敏感度随时间步动态变化，静态策略可能在关键步骤缓存导致质量下降，或在冗余步骤浪费计算。

这些静态策略忽视了生成过程的动态特性，导致效率与质量难以兼顾。

### **提出的新方法**
作者提出了 **OnlineCache**，一种动态、实例感知（instance-aware）的缓存框架，其核心创新包括：

#### **(1) 动态缓存决策（Dynamic Caching Policy）**
- 将缓存决策建模为一个 **sequential decision-making** 问题，使用 **Policy Gradient** 训练一个轻量级 **MLP policy network**，根据当前隐状态决定是否复用缓存特征。
- 状态设计包含：
  - 隐变量 $x_t$ 和缓存残差 $r_{\text{cached}}$ 的通道均值 $\mu(\cdot)$ 和标准差 $\sigma(\cdot)$
  - 最大绝对值 $\max(|r_{\text{cached}}|)$
  - 时间嵌入 $emb_{\text{time}}(t)$
- 政策网络输出为伯努利分布的概率，决定是否跳过当前 Transformer 计算。

#### **(2) 双层优化框架（Bilevel Optimization, BLO）**
- 引入一个可学习的 **error corrector**（轻量 MLP），用于修正因缓存引入的近似误差。
- 构建双层优化结构：
  - **外层目标（Outer-loop）**：优化 policy，最大化全局生成质量与速度的权衡。
  - **内层目标（Inner-loop）**：训练 corrector，最小化缓存步的局部误差。
- 通过交替更新实现联合优化，确保 policy 在可纠正范围内进行缓存。

### **相比现有方法的优势**
| 特性 | ERTACache / TeaCache | OnlineCache |
|------|------------------------|------------|
| 缓存策略 | 静态或启发式规则 | 学习型动态策略 |
| 错误处理 | 无或事后校正 | 联合训练的可学习 corrector |
| 决策粒度 | 统一策略 | 实例感知（sample & timestep 自适应） |
| 性能 | 加速有限或质量损失明显 | 更高加速比 + 更优质量 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Image Generation**:
  - `MSCOCO`（512×512, 800 samples）
  - `Parti-Prompts`（512×512, 1632 samples）
  - `ImageNet-256`（用于 DiT-XL/2）
- **Video Generation**:
  - `Parti-Prompts`（12 frames, 480P, 用于 CogVideoX-2B）

### **实验设置**
- **主干模型**：
  - **FLUX.1-dev**（30 steps, 512×512）
  - **DiT-XL/2**（基于 Peebles & Xie, 2023）
  - **CogVideoX-2B**（Yang et al., 2025）
- **缓存比例控制**：通过调整 policy 输出 logits 实现灵活加速比。
- **训练硬件**：单张 A100-80GB GPU。

### **评估指标**
| 类别 | 指标 |
|------|------|
| **效率** | Speedup×, Latency (Lat), Cache Ratio (CR) |
| **视觉质量** | LPIPS↓, SSIM↑, PSNR↑, FID↓, IS↑, Precision/Recall |
| **视频质量** | VBench↑ |

### **基线方法对比**
- **静态/启发式缓存**：
  - `ERTACache`, `TeaCache`, `TaylorSeer`, `DeepCache`
- **学习型缓存**：
  - `FastCache`, `L2C`, `AdaCache`
- **全步长推理**：
  - `FLUX (30 steps)`, `DiT (30 steps)` 等作为 GT 基准。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) FLUX.1-dev 上的结果（Table 1）**
| 方法 | Speedup× | LPIPS↓ | SSIM↑ | PSNR↑ | CR |
|------|----------|--------|-------|-------|-----|
| FLUX (30 steps) | 1.00× | – | – | – | 0.00% |
| ERTACache | 2.68× | 0.252 | 0.739 | 19.852 | 66.67% |
| **OC-BLO (Ours)** | **2.96×** | **0.245** | **0.739** | **21.442** | **72.22%** |

- **加速近 3×**，同时 **LPIPS 更低、PSNR 更高**，显著优于 ERTACache。
- 在 1024×1024 分辨率上仍保持 **3.25× 加速**，验证跨分辨率泛化能力。

#### **(2) DiT-XL/2 上的结果（Table 2）**
| 方法 | FID↓ | Speedup× |
|------|------|---------|
| Baseline | 4.45 | 1.00× |
| FastCache | 4.46 | 1.74× |
| **OC-BLO (Ours)** | **3.28** | **1.88×** |

- **FID 从 4.45 降至 3.28**，同时实现 **1.88× 加速**，甚至优于全步长基线。
- 作者认为这是因 OnlineCache 抑制了采样噪声，提升了鲁棒性。

#### **(3) CogVideoX-2B 上的结果（Table 3）**
| 方法 | Speedup× | LPIPS↓ | SSIM↑ | PSNR↑ |
|------|----------|--------|-------|-------|
| CogVideoX (12 steps) | 1.00× | – | – | – |
| TeaCache (λ=0.2) | 1.64× | 0.159 | 0.854 | 24.076 |
| **OC-BLO (Ours)** | **1.79×** | **0.106** | **0.921** | **29.983** |

- 在视频生成任务中同样取得更优加速与质量平衡。

### **消融实验结果**

#### **(1) Policy 输入组件消融（Figure 4）**
移除任一组件均导致性能下降：
- **No Residual**：性能最差 → 表明缓存历史信息至关重要。
- **No Time Embed / No Hidden State**：性能下降 → 时间感知与当前状态不可或缺。

#### **(2) Corrector 设计消融（Figure 6 & Table 5）**
| 设计选择 | 结果 |
|--------|------|
| **对齐目标（Alignment Target）** | 对齐低维 velocity 比高维 hidden_states 更高效且效果相当 |
| **修正范围（Scope）** | 仅在缓存步应用 corrector 效果更好（避免干扰早期关键步骤） |
| **优化策略** | BLO（28 outer epochs）比 Sequential Training（40+10 epochs）收敛更快、性能更优 |

#### **(3) 缓存动态分析（Figure 8）**
- **Timestep-level**：早期保守缓存（保护结构），后期积极缓存（细调阶段误差容忍度高）。
- **Sample-level**：提示词越长（越复杂），缓存率越低（r = -0.1994），体现动态资源分配。

---

## **4. 关键结论和发现**

### **主要发现**
1. **动态缓存优于静态策略**：OnlineCache 能根据样本难度和时间步敏感度自适应分配计算资源，实现更优的 speed-quality trade-off。
2. **corrector 显著抑制误差累积**：消融实验证明，corrector 可将累计 L1 误差降低 **37.48%**，防止漂移。
3. **BLO 框架有效且高效**：尽管使用一阶近似，但交替训练策略已足够实现稳定收敛与性能提升。
4. **强泛化能力**：
   - 跨分辨率（512→1024）
   - 跨数据集（MSCOCO → Parti-Prompts）
   - 跨模型（FLUX.1-dev → FLUX.1-schnell）
   - 跨模态（图像 → 视频）

### **局限性**
1. **训练开销**：需额外训练 policy 和 corrector（约 26 小时 for FLUX.1-dev），非完全 training-free。
2. **BLO 近似误差**：未展开完整反向传播，理论可能收敛至次优解。
3. **架构依赖性**：policy 依赖于特定 latent space 的统计特征，难以直接迁移到 U-Net 等不同架构。

### **未来工作方向**
- 探索 **architecture-agnostic** 的 policy 设计，提升跨模型迁移能力。
- 研究 **online adaptation**，使 policy 能在推理时持续微调以适应新数据分布。
- 扩展至更多模态（如 3D、音频）和更复杂任务（如编辑、inpainting）。

---

> ✅ **代码与模型**：作者承诺开源代码及多个主流 Diffusion Model 的训练好 policy 网络。

</details>

---

### 9. [OpenClaw and Ollama in Agentic AI: Toward Fully Autonomous and Scalable AI Agent Systems](https://arxiv.org/abs/2607.28629)

**Authors**: Konstantinos I. Roumeliotis, Ranjan Sapkota  
**Category**: cs.AI  
**Published**: 2026-08-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.28629v1  

#### Abstract
The rapid transition from reactive large language models (LLMs) to persistent, action-capable systems has exposed critical gaps in the architectural understanding of Agentic AI, particularly in separating inference, orchestration, and execution layers for autonomous AI agents. Despite recent advance...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前 Agentic AI 领域存在严重的**架构理解碎片化**问题。尽管大型语言模型（LLMs）能力强大，但多数研究仍聚焦于单一组件（如推理、记忆或工具调用），缺乏对完整自主系统从**推理层到执行层**的全栈（full-stack）系统性分析。这导致以下问题：
- 无法区分“模型能力”与“系统智能”的来源；
- 缺乏统一框架来设计、部署和评估完整的 AI Agent 系统；
- 安全、隐私、可扩展性和可信度等挑战难以在孤立模块中解决。

### 提出的新方法与新思路
本文提出了一种**分层的 Agentic AI 架构范式**，并以 **Ollama + OpenClaw** 作为典型案例进行实证验证，其核心创新如下：

- **提出了五层分层架构模型**：
  - **Inference Layer**（Ollama）：负责本地化的 LLM 推理；
  - **Runtime Layer**（OpenClaw）：实现持续运行的 Agent 行为循环（observe-think-act）；
  - **Memory Layer**：支持短期与长期记忆存储与检索；
  - **Execution Layer**：连接外部工具（API、文件系统等）；
  - **Governance Layer**：嵌入安全控制、审计日志与权限管理。

- **首次将 Ollama 和 OpenClaw 联合建模为一个完整的 Agentic AI 全栈系统**，明确划分了两者的角色：
  - Ollama 是“大脑”——提供认知能力；
  - OpenClaw 是“身体”——组织行为、维持状态、执行动作。

- **强调“系统集成”是产生自主性的关键**：真正的 Agent 能力（如工具使用、持久记忆、多步规划）并非来自更强的 LLM，而是源于各层之间的协同整合。

### 相比现有方法的优势
| 维度 | 传统方法 | 本工作 |
|------|--------|-------|
| 架构视角 | 单一模型或功能模块为中心 | 分层解耦，关注系统级集成 |
| 自主性来源 | 归因于 LLM 本身 | 来自 Runtime 对 Inference 的编排 |
| 部署灵活性 | 多依赖云服务 | 支持本地化、边缘部署（via Ollama） |
| 可解释性与可控性 | 黑箱输出 | 提供可观测的日志、追踪与控制机制 |
| 安全治理 | 后置添加 | 内生于架构设计 |

---

## 2. 核心实验方法和设置

### 数据集与任务设计
实验未使用公开基准数据集，而是构建了一个**定制化的原型任务套件（benchmark task suite）**，共包含 **15 个任务**，分为三类：

| 类别 | 任务数量 | 示例任务描述 |
|------|---------|-------------|
| **Reactive Tasks** | 3 | 从本地文本文件 `reactive_notes.md` 中提取项目名称、截止日期、负责人 |
| **Tool-use Tasks** | 6 | 执行 CSV 计算、写入文件、调用本地 API (`/api/tags`)、数学表达式求值、生成 cron 表达式 |
| **Persistent Tasks** | 6 | 跨任务的记忆存储与召回（如记住格式偏好、别名映射，并在后续任务中正确调用）|

> ✅ 所有任务定义、评分规则和支持文件均开源发布。

### 实验设置
- **硬件环境**：单节点服务器，Ubuntu 22.04 + NVIDIA H100 GPU
- **模型选择**：
  - `Qwen3.5:4b`（40亿参数）
  - `Gemma4:e4b`（40亿参数）
- **推理引擎**：Ollama（本地部署，4-bit量化）
- **Agent 运行时**：OpenClaw v2026.4.5
- **通信方式**：Python 脚本通过 REST API 与 Ollama 交互，CLI 与 OpenClaw 交互

#### 三种对比配置（消融实验设计）
| 配置 | 描述 | 目标 |
|------|------|------|
| **C1: Ollama-only baseline** | 直接调用 Ollama，无工具访问、无持久内存、无会话连续性 | 测试纯 LLM 推理能力 |
| **C2: OpenClaw stateless** | 使用 OpenClaw 运行时，启用全部工具集，但在每个任务间重置状态（清空内存） | 测试工具调用能力（无记忆） |
| **C3: OpenClaw persistent (full stack)** | 完整启用 OpenClaw + Ollama，允许跨任务持久化内存（SQLite + 文件日志） | 测试完整 Agentic 系统能力 |

每种配置重复运行 3 次，报告均值 ± 标准差。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Task Success Rate** | 成功完成的任务比例（综合多个判定标准） |
| **Tool-call Accuracy** | 在 6 个工具使用任务上的成功率 |
| **Memory Recall Accuracy** | 在 6 个持久记忆任务上的召回准确率 |
| **Average Latency (ms)** | 端到端响应时间（从输入到最终输出） |
| **Reasoning Steps** | 平均推理迭代次数（反映决策深度） |
| **Scoring Criteria** | 包括 `expected_exact`, `expected_contains`, `expected_regex`, `verify_file_exists`, `expected_not_contains` 等多种判断逻辑，减少误判 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 4 & 5）

#### 总体任务成功率（Success Rate）
| Model | C1 (Ollama only) | C2 (OpenClaw, stateless) | C3 (OpenClaw, persistent) |
|-------|------------------|--------------------------|----------------------------|
| Qwen3.5:4b | 0.467 ± 0.000 | 0.911 ± 0.038 | **0.933 ± 0.067** |
| Gemma4:e4b | 0.467 ± 0.000 | 0.955 ± 0.039 | **0.978 ± 0.039** |

✅ **所有模型均呈现严格单调增长趋势：C1 < C2 < C3**

#### 分类任务成功率（Per-category）
| Model | Config | Reactive | Tool-use | Persistent |
|-------|--------|----------|----------|------------|
| Qwen3.5:4b | C1 | 0.000 | 0.667 | 0.500 |
|               | C2 | 1.000 | 0.944 | 0.833 |
|               | C3 | 1.000 | 0.833 | **1.000** |
| Gemma4:e4b | C1 | 0.000 | 0.667 | 0.500 |
|               | C2 | 1.000 | **1.000** | 0.889 |
|               | C3 | 1.000 | 0.944 | **1.000** |

> 🔍 发现：
> - C1 在 Reactive 任务上得分为 0 —— 因无法读取文件内容；
> - C2 已能完美处理 Reactive 和 Tool-use 任务；
> - **只有 C3 能在 Persistent 任务上达到 1.000 准确率**。

#### 其他指标
| 指标 | C1 | C2 | C3 |
|------|----|----|----|
| **平均延迟 (Latency)** | ~1s | ~12s | ~11.5s |
| **Reasoning Steps** | 1（单轮） | >1（多轮） | >1（多轮） |

> ⚠️ 尽管引入了约 11–13 倍的延迟开销，但这是实现复杂行为所必需的成本。

### 与基线方法的对比结果
- **C1 vs C2/C3**：证明仅靠 LLM 推理不足以完成现实世界任务，必须依赖运行时编排；
- **C2 vs C3**：显示**持久记忆**是实现跨任务一致性行为的关键；
- **不同模型在 C3 下表现趋同**（0.933 vs 0.978），说明当系统架构完善后，底层 LLM 差异的影响被显著削弱。

### 消融实验结果
- **移除工具调用（C1）** → Tool-use 任务失败率高；
- **移除持久内存（C2）** → Persistent 任务无法跨任务回忆；
- **保留完整架构（C3）** → 实现接近完美的任务成功率与确定性记忆召回。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **自主性来源于系统集成而非单一模型**  
   Agent 的高级能力（工具使用、长期记忆、多步规划）是由 **Runtime 层对 Inference 层的有效编排**产生的，而非 LLM 自身具备。

2. ✅ **分层架构具有独立且累积的增益效应**  
   实验验证了 **C1 < C2 < C3** 的严格递进关系，表明每一层（工具接入、持久记忆）都带来独立且可测量的能力提升。

3. ✅ **持久记忆可实现零方差、确定性召回**  
   在 C3 配置下，Memory Recall Accuracy 达到 **1.000 ± 0.000**，说明只要信息被正确写入持久化存储，就能被可靠地读取。

4. ✅ **系统行为趋于架构驱动而非模型驱动**  
   不同 LLM 在完整架构下性能趋近，说明良好的 Agent Runtime 可“标准化”模型差异，使系统行为更稳定、可预测。

5. ⚠️ **存在可识别的失败模式**  
   - 工具选择歧义（如应调用本地 API 却误走搜索引擎）；
   - 数值计算中的浮点误差积累；
   - 这些属于工程优化范畴，非根本性缺陷。

### 方法的局限性
- **实验规模较小**：仅基于单机、两个模型、15个任务；
- **未涉及多 Agent 协作**：尚未测试分布式或多角色协同场景；
- **延迟较高**：约 12 秒的响应时间限制了实时交互应用；
- **安全性依赖人工配置**：当前实验假设工具接口是可信的，未深入测试对抗性攻击下的鲁棒性。

### 未来工作方向
1. **Scalable Multi-Agent Systems**  
   扩展至多 OpenClaw Agent 共享 Ollama 集群的分布式架构，研究任务分解、通信协议与冲突解决机制。

2. **Distributed & Edge Architectures**  
   探索边缘设备上的轻量级 Agent 部署方案，结合联邦学习与资源调度策略。

3. **Human-Centered & Responsible AI**  
   强化 Human-in-the-loop 设计，支持用户干预、目标修正与行为审计；发展可解释性机制以增强信任。

4. **Advanced Governance & Safety Mechanisms**  
   构建动态沙箱、细粒度权限控制、自动风险检测与熔断机制（kill-switch）。

5. **Long-Horizon Autonomy Benchmarking**  
   开发新的评估框架，衡量 Agent 在数天甚至数周尺度上的适应性、稳定性与演化能力。

---

> 📦 **代码与数据开放声明**：  
> 本文所有实验代码、任务定义、配置文件、原始日志均已开源发布于 GitHub：  
> [https://github.com/Applied-AI-Research-Lab/OpenClaw-and-Ollama-in-Agentic-AI](https://github.com/Applied-AI-Research-Lab/OpenClaw-and-Ollama-in-Agentic-AI)  
> 支持完全复现与社区共建。

</details>

---

### 10. [NeSyFS: A Neuro-symbolic Fast-Slow Thinking Framework for LLM Agent under Partial Observability](https://arxiv.org/abs/2607.28942)

**Authors**: Duo Xu, Faramarz Fekri  
**Category**: cs.AI  
**Published**: 2026-08-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.28942v1  

#### Abstract
Recently Large Language Models (LLMs) have been increasingly deployed as autonomous agents in applications such as self-reflection, retrieval-augmented generation, and scientific discovery. In these settings, agents must act based on limited observations rather than full environmental states, leadin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NeSyFS: A Neuro-symbolic Fast-Slow Thinking Framework for LLM Agent under Partial Observability

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对 **Large Language Model (LLM) Agent 在部分可观测环境（Partial Observability）** 下面临的三大核心挑战：
- **Belief State Inference**：由于只能获取局部观察，难以准确推断环境的真实状态。
- **Task Objective Misalignment**：在信息不完整的情况下，LLM 容易生成偏离任务目标的动作。
- **Planning under Uncertainty**：缺乏对环境全貌的认知，导致规划过程充满不确定性。

传统方法依赖于将完整的交互历史或其摘要作为上下文输入，容易引入冗余和噪声信息，影响决策质量。

---

### 🚀 提出的新方法与创新思路
作者提出了一种全新的 **Neuro-Symbolic Fast-Slow Thinking Framework (NeSyFS)**，融合神经网络（LLM）与符号系统（Knowledge Graph），实现更鲁棒的智能体决策。

#### 主要模块设计：
| 模块 | 功能 |
|------|------|
| **Knowledge Graph (KG)** | 作为结构化记忆，动态维护环境的信念状态（belief state）。每条观察被转化为三元组 `(entity, relation, entity)` 或 `(entity, attribute, value)` 存储于图中，并支持检索与更新。 |
| **Fast-Thinking Module** | 基于 KG 中检索到的相关三元组，快速生成反应式动作（reactive action），类似人类的“直觉思维”。使用 Chain-of-Thought (CoT) 推理机制。 |
| **Reflection Module** | 对 fast-thinking 产生的动作进行反思，判断是否符合任务目标。若连续失败 $K$ 次，则触发 slow-thinking。采用第三视角验证而非自我解释，提升可靠性。 |
| **Slow-Thinking Module** | 执行基于 **Twisted Sequential Monte Carlo (TSMC)** 的不确定性感知规划算法。通过粒子传播、权重更新与重采样机制，在高不确定环境下搜索最优路径。 |

> 🔍 **首次将 TSMC 引入 LLM Agent 规划框架**，并结合 KG 提供的状态表示，显著增强应对部分可观测性的能力。

---

### ⭐ 相比现有方法的优势
| 方面 | NeSyFS 的优势 |
|------|----------------|
| **状态表示** | 使用 KG 替代原始文本历史或 LLM 自动生成的摘要，避免信息丢失和噪声干扰，提供结构化、可查询的状态近似。 |
| **推理效率** | Fast-thinking 快速响应；仅当检测到动作不可靠时才启动计算成本更高的 slow-thinking，平衡效率与准确性。 |
| **任务一致性保障** | Reflection 模块以 stepwise granularity 进行动作级检查，比传统的 trajectory-level 反思更高效且及时。 |
| **规划鲁棒性** | TSMC 风格的粒子滤波机制能容忍 LLM 在预测中的错误（如误判任务进展），并通过 resampling 保留潜在成功路径。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在三个具有代表性的文本型交互环境中进行评估：
| 数据集 | 描述 |
|--------|------|
| **ALFWorld** | 模拟家庭环境中的具身任务（如“把苹果放进微波炉”），需多步操作与物体定位。评估指标为 **Success Rate (SR)**。 |
| **WebShop** | 模拟电商网站购物流程（搜索商品 → 添加购物车 → 支付），测试复杂决策链能力。评估指标为 **SR 和 Average Reward (AR)**。 |
| **ScienceWorld** | 科学实验类任务（如“测量水的沸点”），强调程序性与因果推理。同样使用 **SR 和 AR**。 |

---

### ⚙️ 实验设置
- **模型后端**：使用 GPT-5、GPT-5-mini 和 Llama-3.3-70B-Instruct 作为底层 LLM。
- **评估方式**：每个配置运行三次随机种子取平均值。
- **上下文长度限制**：所有方法均受限于相同上下文窗口大小，确保公平比较。

---

### 🆚 基线方法对比
选取以下代表性 LLM Agent 方法作为 baseline：
| 方法 | 简介 |
|------|------|
| **ReAct** | 结合推理与行动的经典框架，基于 CoT 决策。 |
| **Reflexion** | 失败后进行轨迹级反思，改进后续行为。 |
| **ABBEL** | 使用 LLM 总结历史构建 belief state。 |
| **RAFA** | 每步执行短视域树搜索规划。 |
| **SwiftSage** | 典型的 fast-slow 架构，Swift 为快思考，Sage 为慢规划。 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）
| Model | Method | ALFWorld (SR) | WebShop (AR/SR) | ScienceWorld (AR/SR) | **Average** |
|-------|--------|---------------|------------------|------------------------|------------|
| GPT-5-mini | ReAct | 71.2 | 42.5 / 32.7 | 65.1 / 28.3 | 44.1 |
| GPT-5-mini | SwiftSage | 76.7 | 52.3 / 38.2 | 72.1 / 35.7 | 51.0 |
| GPT-5-mini | **NeSyFS** | **91.1** | **61.2 / 51.3** | **82.2 / 61.2** | **63.5** |
| GPT-5 | ReAct | 78.2 | 49.6 / 38.7 | 72.2 / 35.5 | 50.8 |
| GPT-5 | SwiftSage | 81.9 | 56.3 / 41.2 | 75.3 / 40.2 | 54.4 |
| GPT-5 | **NeSyFS** | **93.6** | **69.4 / 63.3** | **86.1 / 69.4** | **75.3** |
| Llama-3.3 | ReAct | 79.6 | 45.6 / 37.1 | 69.2 / 33.1 | 49.9 |
| Llama-3.3 | SwiftSage | 80.6 | 51.2 / 39.9 | 76.1 / 41.3 | 53.9 |
| Llama-3.3 | **NeSyFS** | **89.6** | **68.1 / 62.5** | **85.2 / 67.1** | **73.0** |

> ✅ **NeSyFS 在所有模型和任务上均显著优于所有 baseline**，平均性能提升约 **20–30个百分点**。

---

### 🔍 消融实验与分析（见 Table 2 与 Figure 5）

#### （1）Reflection 模块的有效性（Table 2）
评估不同 context 表示下的反思准确性：
| Context Type | TDE (越小越好) | ER (越高越好) |
|--------------|----------------|----------------|
| History (完整历史) | 57–135 | 0.42–0.55 |
| Belief (LLM 摘要) | 52–125 | 0.45–0.59 |
| **KG (三元组)** | **42–91** | **0.65–0.71** |

> ✅ **KG 提供的 context 显著降低检测误差，提高批准动作的可靠性（ER）**，说明结构化信息更利于精准反思。

#### （2）Context 形式的影响（Figure 5）
比较不同 context 输入对最终成功率的影响：
- “KG Fast” > “Belief Fast” > “History Fast”
- 加入 reflection 后，“KG Ref” 提升最大，表明 **KG + Reflection 协同效应最强**

> 💡 发现：**KG 不仅本身是更好的状态表示，还能增强 reflection 和 planning 模块的效果**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KG 是比历史或摘要更优的信念状态表示形式**：结构化三元组有效压缩信息、减少噪声，提升 LLM 决策、反思与预测的准确性。
2. **Fast-Slow 架构 + 动态切换机制是高效的权衡策略**：日常任务由 fast-thinking 快速处理，困难情况自动转入 slow-thinking，兼顾效率与性能。
3. **Stepwise Reflection 比 Trajectory-Level 更实用**：逐动作检查可在早期纠正偏差，防止错误累积。
4. **TSMC-style Planning 能有效应对不确定性**：粒子机制允许探索多个可能路径，resampling 抵抗 LLM 判断失误，适合部分可观测场景。

---

### ⚠️ 方法的局限性
- **KG 构建依赖 LLM 解析能力**：从自然语言观察中提取三元组的质量直接影响整个系统的性能。
- **Slow-Thinking 计算开销较高**：尽管只在必要时启用，但在大规模环境中仍可能成为瓶颈。
- **当前实验限于文本环境**：未扩展至视觉或多模态部分可观测任务。

---

### 🔮 未来工作方向
- 将 NeSyFS 扩展到 **vision-language agents**，结合图像输入构建跨模态 KG。
- 探索 **automated KG schema induction**，减少人工定义关系的成本。
- 引入 **learned retrieval policies**，动态选择最相关的子图用于推理。
- 应用于真实世界应用，如 **robotic planning** 或 **autonomous driving simulation**。

---

## ✅ 总结
NeSyFS 是首个将 **Neuro-Symbolic 架构** 与 **Fast-Slow Cognition**、**KG Memory** 和 **TSMC-style Planning** 统一整合的 LLM Agent 框架。它在多个标准 benchmark 上实现了 SOTA 性能，证明了 **结构化知识表示 + 分层推理机制** 在解决部分可观测性问题上的巨大潜力。该工作为下一代可靠、可解释、具备长期规划能力的 LLM Agent 提供了重要范式。

</details>

---

### 11. [The Parts Are Greater Than the Sum: Automated Task Sequencing for Efficient Training of Multi-Policy LLMs](https://arxiv.org/abs/2607.29601)

**Authors**: Jiajia Tang, Sizhe Yuen, Francisco Gomez Medina, Yali Du, Adam Sobey  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.29601v1  

#### Abstract
Parameter-Efficient Fine-Tuning (PEFT) commonly adapts large language models using a single shared Low-Rank Adapter (LoRA). This shared optimization space often suffers from interference when adapting heterogeneous task sequences, leading to poor transfer and catastrophic forgetting. Existing approa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*The Parts Are Greater Than the Sum: Automated Task Sequencing for Efficient Training of Multi-Policy LLMs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Parameter-Efficient Fine-Tuning (PEFT)** 方法（如 LoRA 和 QLoRA）通常采用**单一共享的低秩适配器**（single shared adapter）来适应多个下游任务。然而，在面对**异构任务序列**（heterogeneous task sequences）时，这种共享优化路径会导致严重的**优化干扰**（optimization interference），表现为负迁移（negative transfer）和灾难性遗忘（catastrophic forgetting）。

尽管已有研究通过增加适配器容量或多模块组合提升表达能力，但大多仍依赖于**共享的优化机制**，未能从根本上组织不同任务间的优化路径。

---

### 🚀 提出的新方法与创新思路
本文提出一种全新的 **优化路径组织框架**（optimization-path organization framework），实现为一个**自动化的多策略 PEFT 架构**（automatic multi-policy PEFT architecture）。其核心思想是：

> **与其扩大共享空间，不如合理组织异构任务的学习路径。**

该框架包含两个关键阶段：

#### （1）Stage 1: 自动任务分组（Automatic Task Grouping）
- 基于任务之间的**梯度兼容性**（gradient compatibility）和**行为特征相似性**（behavioral characteristics）构建统一的任务距离矩阵。
- 使用**平衡聚类**（balanced clustering）将任务划分为若干组，每组分配一个独立的 QLoRA 适配器（即一个“policy”），从而在解耦的适应空间中进行训练，减少冲突。

#### （2）Stage 2: 自动任务排序（Automatic Task Sequencing）
- 在每个任务组内，自动生成最优的任务学习顺序。
- 考虑多种因素设计复合目标函数：
  - 相邻任务过渡成本（adjacent transition cost）
  - 首任务奖励（head reward）——鼓励先学复杂任务
  - 尾任务惩罚（tail cost）——避免最终任务造成严重干扰
  - 方向性转移效应（directional transfer）
  - 全局能力演进平滑性（capability progression）

最终形成一条**局部兼容且全局连贯**的优化轨迹。

---

### 🔍 相比现有方法的优势
| 对比维度 | 传统方法（如 LoRA, AdapterFusion） | 本文方法 |
|--------|-------------------------------|---------|
| 优化路径 | 单一共享路径，易受干扰 | 多条独立、组织良好的路径 |
| 参数利用 | 强调参数量或架构改进 | 固定参数预算下优化组织方式 |
| 可扩展性 | 多任务需更多适配器 → 效率下降 | 分组共享 + 序列优化 → 更高效 |
| 是否需要人工干预 | 手动设计任务顺序/分组 | 完全自动，无需先验知识 |

> ✅ **核心洞见**：对于异构任务的 PEFT，**如何组织任务**（grouping & sequencing）比**简单增加参数容量**更有效。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 主要基准：**TRACE**（Wang et al., 2023b）
  - 包含 8 个高度异构的语言任务：
    - C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten
  - 涵盖领域、推理模式、输出格式、监督信号等方面的显著差异
  - 是典型的**连续学习**（continual learning）场景

---

### ⚙️ 实验设置
- **基础模型**：
  - `LLaMA-2-7B-Chat`
  - `Vicuna-7B-V1.5`（基于 LLaMA-2 微调，用于验证鲁棒性）
- **PEFT 方法**：全部使用 **QLoRA**（4-bit 量化 + Low-Rank Adaptation），保证内存效率
- **总可训练秩（trainable rank）固定为 128**，确保公平比较：
  - 单策略基线：1 个 rank-128 的共享 QLoRA
  - 多策略方法：2 个 rank-64 的独立 QLoRA（multi-policy）
- **训练细节**：
  - 每任务 5,000 训练样本
  - Batch size = 2，Learning rate = 1e-5
  - 多数任务训练 3 轮，C-STANCE 和 NumGLUE-ds 训练 5 轮

---

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **Overall Performance (OP)** | 所有任务微调完成后的平均性能 |
| **Backward Transfer (BWT)** | 后续任务对先前任务性能的影响，衡量灾难性遗忘程度<br>BWT > 0 表示正向迁移，BWT < 0 表示遗忘 |

---

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Single-policy QLoRA** | 共享适配器 | 使用单个 rank-128 QLoRA |
| **O-LoRA** | 参数隔离基线 | 每个任务独立 rank-16 LoRA，总计 rank-128 |
| **Random Grouping + Random Sequencing** | 多策略控制组 | 无智能组织的多策略基线 |
| **Manual Grouping + Manual Sequence** | 专家设计 | 人工分析后设定分组与顺序 |
| **Auto Group + Auto Sequence** | 本文完整方法 | 全自动组织优化路径 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（以 LLaMA-2-7B-Chat 为例）

| 方法 | OP ↑ | BWT ↑ |
|------|-------|--------|
| O-LoRA* | 30.76 | -0.023 |
| Single-policy (rank-128) | 42.12 | -0.041 |
| Multi-policy (Random/Rand) | 36.94 | -0.092 |
| Multi-policy (Auto/Rand) | 40.75 | -0.027 |
| Multi-policy (Manual/Manual) | 44.53 | 0.012 |
| **Multi-policy (Auto/Auto)** | **44.78** | **0.013** |

> ✅ **结论**：  
> - 即使与最强的手动设计相比，本文提出的**全自动方法也实现了略优的结果**（44.78 vs 44.53）
> - 显著优于单策略共享适配器（+2.66 OP）
> - BWT 首次达到正值（0.013），表明实现了**净正向迁移**

在 Vicuna 上趋势一致，Auto/Auto 达到 **41.14 OP**，优于手动设计（39.45）。

---

### 🔬 消融实验结果

#### （1）任务分组消融（Task Grouping Ablation）
| 分组策略 | OP | BWT |
|--------|-----|------|
| Alternative Grouping 1 | 41.29 | -0.017 |
| Alternative Grouping 2 | 41.96 | -0.003 |
| **Auto Group (proposed)** | **44.78** | **0.013** |

> ✅ 自动分组显著优于其他任意划分方式，说明**基于梯度+行为融合的距离度量有效捕捉了优化兼容性**。

#### （2）秩分配消融（Rank Allocation Ablation）
| 秩分配（Group1 + Group2） | OP | BWT |
|--------------------------|-----|------|
| 32 + 96 | 41.61 | -0.018 |
| 96 + 32 | 42.58 | -0.011 |
| **64 + 64（均衡）** | **44.78** | **0.013** |

> ✅ **均衡分配**效果最好，说明性能提升不来自某一个 policy 占用更多资源，而是来自整体组织的有效性。

#### （3）任务排序消融（Task Sequencing Ablation）
| 序列策略 | Group1 OP/BWT | Group2 OP/BWT |
|--------|---------------|---------------|
| Random Seq 1 | 54.86 / 0.054 | 25.89 / -0.119 |
| Random Seq 2 | 54.02 / 0.049 | 27.75 / -0.094 |
| **Auto Seq** | **55.81 / 0.084** | **33.90 / -0.062** |

> ✅ 自动序列不仅提高 OP，还明显改善 BWT，尤其在第二组中减少了遗忘。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **优化路径的组织比单纯增加参数更重要**：
   - 在相同 trainable capacity 下，合理的任务分组与排序带来的性能增益远超扩大共享适配器规模。
   
2. **多策略 PEFT 优于单策略与完全隔离策略**：
   - O-LoRA（每任务独立小适配器）表现最差（OP=30.76），说明缺乏迁移导致效率低下。
   - 本文的“分组共享”策略在**降低干扰的同时保留有益迁移**，达到最佳平衡。

3. **自动化优于人工设计**：
   - 无需领域专家参与，仅凭梯度统计与行为特征即可生成媲美甚至超越人工设计的任务组织方案。

4. **任务顺序对连续学习至关重要**：
   - 不同顺序可能导致高达 ~10 点的性能差距，自动序列能有效缓解灾难性遗忘。

---

### ⚠️ 局限性
1. **依赖预知所有任务集合**：
   - 当前方法适用于**预部署场景**（pre-deployment setting），即所有目标任务已知；难以直接应用于开放世界的在线持续学习。
   
2. **计算临时梯度开销**：
   - 需要在初始模型上收集各任务的 mini-batch 梯度以构建表示，带来额外计算成本（虽为一次性）。

3. **聚类数量 $K$ 需预先设定**：
   - 当前实验设 $K=2$，未探讨如何自动确定最优策略数。

4. **行为特征工程依赖经验**：
   - 如 prompt length、answer type tendency 等特征需人工定义，未来可探索端到端学习任务表示。

---

### 🔮 未来工作方向
1. **动态策略扩展机制**：
   - 支持新任务到来时动态创建新 policy 或合并至现有 group。
   
2. **结合在线持续学习框架**：
   - 将本方法嵌入到 rehearsal-free continual learning pipeline 中。

3. **探索更丰富的任务表示学习**：
   - 利用 self-supervised 方法自动提取任务 embedding，替代手工特征。

4. **跨模态扩展**：
   - 将优化路径组织思想推广至多模态大模型（如 MLLMs）的参数高效微调。

---

## ✅ 总结一句话
> **“The parts are greater than the sum” —— 通过对异构任务进行自动分组与排序，构建多条解耦而兼容的优化路径，本文证明了在固定参数预算下，组织方式的优化远胜于盲目扩参，为 PEFT 提供了一个全新而高效的设计范式。**

</details>

---

### 12. [Empowering Cross-Domain Sequential Recommendation with Hybrid Tokenization and Serial-Parallel Decoding](https://arxiv.org/abs/2607.28659)

**Authors**: Yuxuan Hu, Yuhao Wang, Tianbo Huang, Chao Zhang, Ziwei Liu, Lihua Zhang, Xiangyu Zhao  
**Category**: cs.AI  
**Published**: 2026-08-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.28659v1  

#### Abstract
Cross-domain sequential recommendation (CDSR) aims to model users' dynamic interest transitions and sequential patterns across multiple domains. Recently, generative recommendation (GR) has emerged. It first learns semantic identifiers (SIDs) from item semantics and formulates recommendation as auto...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Empowering Cross-Domain Sequential Recommendation with Hybrid Tokenization and Serial-Parallel Decoding*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对**跨域序列推荐（Cross-domain Sequential Recommendation, CDSR）**中的两个关键挑战提出解决方案：

1. **Tokenization 阶段忽略跨域协同相关性**：  
   现有生成式推荐（Generative Recommendation, GR）方法在将物品映射为语义标识符（Semantic Identifiers, SIDs）时，通常采用统一共享或完全独立的量化模型，难以同时建模**跨域共性**与**领域特异性**，导致离散表示的信息损失。

2. **Generation 阶段解码效率低下**：  
   序列生成依赖逐token的自回归解码（如 Beam Search），推理延迟高；而并行解码（如 MTP）虽快但牺牲准确性，存在**准确率-效率权衡困境**。

---

### 🚀 提出的新方法与创新思路

作者提出了 **GenCDSR** —— 一个高效且有效的生成式跨域序列推荐框架，其核心创新包括：

#### （1）**Cross-Domain Hybrid Tokenization（跨域混合分词机制）**
- 采用**多塔架构（multi-tower architecture）**，结合**共享-特定双分支编码器**与**两级残差量化（RQ-VAE）**。
- 分两阶段构建层次化 SID：
  - **Stage 1: Shared-Specific Tokenization (SST)**  
    同时提取跨域共享语义和领域特异性特征，通过 Gumbel-Softmax 路由融合输出。
  - **Stage 2: Fine-Grained Specific Tokenization (FGST)**  
    在第一阶段残差基础上进行细粒度领域专属量化，增强局部表达能力。
- 实现了对**跨域共性**与**领域差异**的联合建模。

#### （2）**Cross-Domain Serial-Parallel Decoding（串并行解码策略）**
- 利用 SID 的层级结构设计两阶段解码：
  - **Step 1**: 串行预测前 $L_1$ 个共享/粗粒度 token（保留上下文依赖）
  - **Step 2**: 并行预测后 $L_2$ 个领域专属 token，基于 Step 1 携带的状态进行条件生成
- 实现部分并行化，在保证生成一致性的前提下显著降低延迟。

#### （3）训练流程优化
- **Unified Recommender Training**：在合并的跨域行为流上预训练，捕捉可迁移模式
- **Domain-Specific Fine-tuning**：冻结主干网络，使用 LoRA 微调各领域适配器，避免知识覆盖

---

### 🔍 相比现有方法的优势

| 维度 | GenCDSR | 传统方法（如 TIGER、GenCDR） |
|------|--------|-----------------------------|
| **Tokenization** | 显式建模共享+特异结构 | 单一共享或独立分词，无法兼顾 |
| **Decoding 效率** | 推理延迟 ↓85.1% | Beam Search 延迟极高 |
| **Accuracy** | 平均提升 1.5% | 受限于信息压缩与解码误差 |
| **实用性** | 支持实时服务部署 | 难以满足线上低延迟要求 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在三个公开的跨域数据集上验证效果：

| 数据集组合 | 来源 | 特点 |
|----------|------|------|
| **Clothing-Sports** | Amazon | 用户重叠率较高，中等稀疏性 |
| **Electronics-Phone** | Amazon | 异构性强，交互稀疏 |
| **Book-Movie** | Douban | 长序列、高稀疏性 |

> 所有数据集经过清洗：过滤交互少于5次的用户和少于3次的物品，并按时间顺序合并跨域行为序列。

---

### ⚙️ 实验设置

- **Backbone 模型**：T5 和 Qwen3-0.6B
- **Tokenization 参数**：
  - RQ-VAE 两阶段结构：$L_1 = 2$, $L_2 = 2$
  - 每层码本大小：256，维度：128
  - 使用 LLaMA-7B 和 Qwen2.5-7B 提取物品文本嵌入
- **训练细节**：
  - 优化器：AdamW
  - Batch Size：256 (T5), 128 (Qwen)
  - LoRA Rank: 8, α: 32 / 16
- **推理设置**：Beam Search (beam size=20)

---

### 📊 评估指标

| 类型 | 指标 | 描述 |
|------|------|------|
| **准确性** | H@K (Hit Ratio), N@K (NDCG) | K=5,10，衡量推荐排序质量 |
| **效率** | LT (Latency) | 平均生成延迟（毫秒） |
| **保真度** | Feature Fidelity | 衡量重建嵌入与原始嵌入的一致性 |

---

### 🆚 基线方法对比

分为两类：

#### （1）单域序列推荐（SDSR）
- GRU4Rec, BERT4Rec, SASRec, TIGER

#### （2）跨域序列推荐（CDSR）
- C2DSR, TriCDR, LLM4CDSR, GenCDR

> 特别对比了解码策略：**Beam Search**, **MTP**, **NEZHA**

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（RQ1）

在 **Table 2** 中显示，GenCDSR 在绝大多数指标上达到 SOTA：

- **平均准确率提升约 1.5%**（相对于最强基线 GenCDR）
- 在 **Movie** 域 H@10 达到 **0.2671**，显著优于 GenCDR 的 0.2622
- 在稀疏场景（如 Electronics）也表现稳健，说明模型具备良好泛化能力

> ✅ 观察结论：
> - CDSR 方法普遍优于 SDSR，证明跨域信号有效缓解数据稀疏
> - 生成式方法（尤其是 GenCDSR）更擅长建模复杂语义转移

---

### ⏱️ 效率分析（RQ2）

| 方法 | 相对延迟（vs Beam Search） | 准确率损失 |
|------|----------------------------|-----------|
| Beam Search | 100%（基准） | 无 |
| MTP | ~10% | 明显下降（↓~20%） |
| NEZHA | ~10% | 下降明显 |
| **GenCDSR（Ours）** | **14.9%**（↓85.1%） | **基本持平甚至反超** |

- 在 T5 上平均延迟从 4.6ms → 0.68ms
- 在 Qwen3 上从 ~40ms → ~7ms
- **实现“零牺牲加速”**：大幅提速的同时保持甚至提升精度

> 💡 原因：串并行解码利用了 SID 层级结构，既保留早期 token 的序列依赖，又允许后期并行预测。

---

### 🔬 消融实验（RQ3）

消融变体：

- **w/o SST**：移除 Stage 1 共享-特异分词
- **w/o FGST**：移除 Stage 2 细粒度领域专属分词

结果表明：

- 移除 **SST** 导致性能最大下降 → 说明**跨域共性建模至关重要**
- 移除 **FGST** 影响领域级精度 → 说明**细粒度领域刻画不可忽视**
- 二者互补，联合设计带来鲁棒增益

---

### 🧩 特征保真度分析（RQ4）

使用 **Feature Fidelity** 指标评估重建质量：

| 方法 | Clothing-Sports | Book-Movie | Electronics-Phone |
|------|------------------|------------|--------------------|
| Sh-RQ-VAE（全共享） | 低 | 中 | 最低（异构强） |
| Sp-RQ-VAE（全独立） | 中 | 中 | 中 |
| **GenCDSR（Ours）** | **最高** | **最高** | **最高** |

> 结论：混合分词机制能更完整地保留原始语义信息，减少量化过程中的信息损失。

---

### 🔢 超参数分析（RQ5）

研究不同 $L_1:L_2$ 分配比例的影响（总长度=4）：

| 比例 | 性能趋势 |
|------|--------|
| 0:4 或 4:0 | 表现最差 |
| 1:3 或 3:1 | 中等 |
| **2:2** | **最优** |

> 发现：平衡的共享与特异容量分配最为关键，过多或过少共享都会损害性能。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **混合分词机制显著提升 SID 表达能力**  
   通过共享-特异双分支结构，实现了对跨域共性与领域差异的精细解耦与联合建模。

2. **串并行解码打破效率瓶颈**  
   利用 SID 的层级结构，可在不牺牲准确性的前提下实现近 85% 的延迟压缩，适用于实时推荐系统。

3. **生成式框架更适合 CDSR 场景**  
   相比判别式模型，生成式方法更能建模复杂的兴趣转移路径和语义演化规律。

4. **信息保真是提升性能的关键因素**  
   更高的 Feature Fidelity 对应更强的下游推荐能力，验证了高质量 tokenization 的重要性。

---

### ⚠️ 方法的局限性

1. **依赖预训练语言模型提取 item embedding**  
   对文本描述质量敏感，冷启动物品可能受影响。

2. **两阶段结构引入额外超参（如 $L_1:L_2$）**  
   需要调参确定最佳配置，自动化程度有待提高。

3. **目前仅验证双域场景**  
   多于两个领域的扩展尚未充分探索。

4. **硬件依赖较强**  
   尽管推理快，但训练仍需多 GPU 支持。

---

### 🔮 未来工作方向

1. **自动学习共享/特异结构比例**  
   引入动态路由或可学习门控机制，替代固定层级划分。

2. **扩展至多目标、多任务推荐**  
   结合点击、转化、停留时长等多种反馈信号。

3. **轻量化部署方案**  
   探索蒸馏或量化技术，进一步压缩模型规模。

4. **支持增量学习与在线更新**  
   适应用户行为快速变化的实际业务需求。

5. **探索非文本模态输入**（如图像、音频）  
   构建真正的多模态 GenCDSR 框架。

---

> 🔗 **代码与数据开源地址**：[https://github.com/AppliedMachine-Learning-Lab/RecSys2026_GenCDSR](https://github.com/AppliedMachine-Learning-Lab/RecSys2026_GenCDSR)

</details>

---

### 13. [Learning Latent Reasoning Traces for Scalar Reward Models End-to-End](https://arxiv.org/abs/2607.29185)

**Authors**: Sanwoo Lee, Clive Bai, Hsiu-Yuan Huang, Kun Liang, Weijie Liu, Yunfang Wu  
**Category**: cs.CL  
**Published**: 2026-08-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.29185v1  

#### Abstract
Reward models (RMs) are central to aligning large language models with human preferences via reinforcement learning. Although traditional scalar RMs enable efficient and probabilistic reward modeling, they rely on superficial cues that fail to generalize to complex or out-of-distribution (OOD) tasks...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning Latent Reasoning Traces for Scalar Reward Models End-to-End*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **scalar reward model (RM)** 虽然在 **Reinforcement Learning from Human Feedback (RLHF)** 中被广泛使用，因其输出数值化奖励、便于优化，但存在以下缺陷：
- 容易过拟合训练数据中的**表面模式**（superficial cues）；
- 在面对复杂任务或 **out-of-distribution (OOD)** 数据时泛化能力差。

而 **generative RM**（如 LLM-as-Judge）通过生成推理链（reasoning traces）提升鲁棒性，但其自然语言形式的评分缺乏标量模型的**数值灵活性和概率可解释性**。

此外，已有混合方法（如 multi-task learning）虽然同时训练生成器和判别器，但由于目标分离，无法保证生成的推理真正服务于下游标量打分任务。

> 🔍 **核心问题**：如何让生成的推理过程（reasoning trace）**直接、有效地辅助标量奖励建模**？

---

### 🚀 提出的新方法：LatentRM

作者提出 **LatentRM** ——一种端到端（end-to-end）的标量奖励建模框架，其核心思想是：

> 将 **chain-of-thought 推理视为离散潜在变量（discrete latent variable）** $ z $，连接输入 $ x $ 和偏好标签 $ y $，并以最大化观测偏好的对数似然 $ \log p(y|x) $ 为目标进行联合优化。

#### 创新机制：
- **统一目标函数**：将整个架构形式化为一个条件生成模型（conditional generative model），其中：
  - 生成器 $ p_\theta(z|x) $ 采样推理链；
  - 标量 RM $ p_\phi(y|x,z) $ 预测偏好排名。
- **变分下界优化（ELBO）**：通过简化推断网络 $ q(z|x,y) = p_\theta(z|x) $，得到可高效优化的目标：
  $$
  \mathcal{L}_{\text{ELBO}} = \mathbb{E}_{z \sim p_\theta(z|x)}[\log p_\phi(y|x,z)]
  $$
  即：**最大化在当前推理分布下的偏好预测似然**。

- **端到端联合训练**：
  - 使用 **REINFORCE** 更新生成器，奖励信号为标量 RM 的 log-likelihood；
  - 使用监督学习更新标量 RM；
  - 所有更新基于**同策（on-policy）**生成的推理，避免训练-推理不一致。

---

### ✅ 相比现有方法的优势

| 方法 | 缺陷 | LatentRM 如何改进 |
|------|------|------------------|
| **Scalar RM** | 易过拟合，依赖表面特征 | 引入深层推理作为中间证据，增强语义理解 |
| **Generative RM** | 输出非数值，难集成进 RL 流程 | 保留标量输出，兼具可微性和解释性 |
| **Hybrid/Multi-task RM** | 生成器与标量 RM 目标脱节 | 统一目标驱动两者协同进化 |

> 💡 **关键优势**：推理不再是“旁观评论”，而是**为标量打分服务的潜变量**，实现“推理即支持”的紧耦合设计。

---

## 2. 核心实验方法和设置

### 📚 数据集

#### 训练数据池（共 80K 样本）
从多个高质量偏好数据集中采样，并经过过滤处理：

| 数据集 | 领域 | 样本数 |
|--------|------|-------|
| **UltraFeedback (UF)** | 通用对话 | 28K |
| **OpenMathReasoning (OMR)** | 数学推理 | 28K |
| **Helpsteer3 (HS3)** | STEM & 编程 | 8K |
| **WildGuard (WG)** | 安全性 | 8K |
| **OffsetBias (OB)** | 对抗性偏见 | 8K |

> ⚙️ **预处理**：采用 split-and-filter 协议剔除低损失样本（可能含虚假相关性），最终保留约 43K 训练样本。

---

### 🧪 评估基准（OOD 泛化测试）

| 基准 | 描述 |
|------|------|
| **RM-Bench** | 包含 4K 偏好对，测试对语义细微差异的敏感性和风格偏差鲁棒性 |
| **PPE Correctness** | 2.5K 复杂推理提示，每提示 5 个候选响应，需深度推理判断正确性 |

> ✅ 两个基准均模拟真实 RLHF 设置（同一 generator 生成 pair），且与下游 RLHF 表现高度相关。

---

### 📊 评估指标

| 指标 | 应用场景 | 说明 |
|------|----------|------|
| **Log-likelihood** $ \log p(y|x,z) $ | ID 测试集 | 衡量标量 RM 对真实偏好的拟合程度（越高越好） |
| **Kendall’s Tau** | ID/OOD 测试集 | 排序一致性度量（越接近 1 越好） |
| **Accuracy** | RM-Bench / PPE | 将 listwise 排名转为 pairwise 决策后计算准确率 |
| **Length-Controlled Win Rate (LC Winrate)** | RLHF 实验 | 控制长度偏差后的策略胜率，衡量在线对齐效果 |

---

### 🆚 基线方法对比

| 方法 | 类型 | 特点 |
|------|------|------|
| **ScalarRM** | 标准标量 RM | 使用 Plackett-Luce 损失训练 |
| **GenerativeRM** | 生成式 RM | 用 RL 优化 Kendall’s Tau |
| **MultitaskRM** | 混合多任务 RM | 生成器用 Kendall’s Tau，标量 RM 用 Plackett-Luce，平行训练 |
| **LatentRM (ours)** | 潜变量联合训练 | 同一目标下端到端联合优化 |

> 🧠 所有模型基于 **Qwen3-4B-Instruct** 初始化，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 结果 1：ID 测试集表现（Table 2）

| 方法 | Log-likelihood ↓ | Kendall’s Tau ↑ |
|------|------------------|-----------------|
| ScalarRM | -1.175 | 0.673 |
| MultitaskRM | -1.053 | 0.706 |
| **LatentRM (Ours)** | **-1.031** | **0.712** |

> ✅ LatentRM 在所有领域（尤其是数学、STEM、对抗性任务）上均取得最优 log-likelihood 和 Kendall’s Tau，验证其更强的偏好建模能力。

---

### 📉 结果 2：OOD 泛化能力（Table 3）

| 方法 | RM-Bench (↑) | PPE Correctness (↑) |
|------|--------------|---------------------|
| ScalarRM | 75.7 | 65.8 |
| GenerativeRM | 81.3 | 64.0 |
| MultitaskRM | 81.7 | 71.9 |
| **LatentRM (Ours)** | **82.8** | **72.1** |

> ✅ LatentRM 在 OOD 场景下全面领先：
- 在 **RM-Bench Hard 子集**达到 **81.3%** 准确率，显著优于其他方法 → 显示其对风格偏见（如长度、格式）具有强鲁棒性；
- 在 **PPE 的 MATH/GPQA** 等推理密集任务中表现突出（如 MATH 达 **92.7%**）→ 表明推理链有效提升了复杂任务判断能力。

---

### 🔍 结果 3：外部模型对比（Table 4）

| 方法 | 参数量 | RM-Bench | PPE |
|------|--------|---------|-----|
| SteerLM-RM (Llama-70B) | 70B | 72.2 | 63.2 |
| J1 (Llama-70B) | 70B | 82.7 | 70.2 |
| **LatentRM (Qwen-4B)** | **4B** | **82.8** | **72.1** |

> ✅ LatentRM 仅用 **4B 参数**即超越多数 **7B–70B** 的大型生成式或标量 RM，显示其极高的效率与性能平衡。

---

### 🎯 结果 4：RLHF 在线对齐实验（Figure 4）

使用 **GRPO** 进行 100 步 RLHF 微调，比较不同 RM 指导下的策略胜率（length-controlled）：

| 对手策略 | LatentRM 的 LC Winrate |
|----------|------------------------|
| Base Policy (no RL) | 56.9% |
| ScalarRM | **58.5%** |
| GenerativeRM | 51.5% |
| MultitaskRM | 52.0% |
| **LatentRM (ours)** | ✅ **>58.5%**（全面胜出） |

> ✅ LatentRM 不仅胜率最高，还生成最短回复（平均 **1,289 tokens**），表明其未陷入“奖励黑客”（reward hacking）导致的冗长输出。

---

### 🔬 分析实验：Score Gap 分布（Figure 5）

可视化生成器的自然语言评分 vs. 下游标量 RM 打分之间的差距：

- 当生成器打错分时（红色区域），**LatentRM 仍能通过推理内容恢复正确的排序决策**；
- 而 MultitaskRM 更倾向于“回声”自己的 NL 评分，恢复能力弱。

> ✅ 说明 LatentRM 并非简单复制语言评分，而是**真正利用推理内容作为支撑证据**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **端到端联合训练显著优于多任务学习**：
   - LatentRM 通过统一 likelihood 目标，使生成器专注于产出有助于标量打分的推理，而非追求表面正确的语言评分。

2. **推理作为潜变量可有效缓解过拟合与分布偏移**：
   - 在 OOD 和复杂推理任务中表现优异，证明其学到的是更本质的任务逻辑，而非表面统计规律。

3. **小模型也能超越大模型**：
   - 基于 4B 模型的 LatentRM 超越了 7B–70B 的主流 RM，凸显方法的有效性。

4. **在线 RLHF 表现稳定，无奖励过度优化现象**：
   - 输出简洁，胜率高，适合实际部署。

---

### ⚠️ 局限性

1. **在安全/对抗领域略逊于 MultitaskRM**：
   - 可能因训练数据中 UF 和 OMR 占比过高（70%），导致模型偏向这些领域；
   - 安全类任务需要更高优先级的风险识别机制。

2. **依赖高质量 prompt template**：
   - 推理格式必须严格遵循模板（如 `<rubrics>` 和 `<score_i>` 标签），否则会影响 scalar head 提取。

3. **训练成本较高**：
   - 需要在线 rollout 和 REINFORCE 更新，计算开销大于纯监督训练。

---

### 🔮 未来工作方向

1. **动态调整推理深度**：
   - 根据任务难度自适应决定是否调用深层推理，提升效率。

2. **引入可微分采样机制**：
   - 替代 REINFORCE，降低梯度方差，加速收敛。

3. **扩展至多模态 reward modeling**：
   - 将图像、代码等中间表示也建模为 latent variables。

4. **结合 inference-time scaling**：
   - 如 test-time search 或 self-consistency，在推理阶段进一步提升鲁棒性。

---

## 总结

> 🌟 **LatentRM 是首个将 reasoning trace 明确建模为 latent variable 并用于 end-to-end 优化 scalar reward model 的框架**。它解决了传统混合方法中“推理与打分脱节”的根本问题，实现了：
>
> - 更强的 ID/OOD 偏好建模能力；
> - 更优的 RLHF 对齐性能；
> - 更高效的参数利用率。
>
> 该工作为构建**可信、可解释、可扩展的 reward model** 提供了新范式。

</details>

---

### 14. [DeltaServe: Host-Agnostic Co-Serving of Inference and Fine-Tuning for LLMs](https://arxiv.org/abs/2607.28848)

**Authors**: Jiaxuan Chen, Jianshu She, Ye Yuan, Rajat Ghosh, Karan Gupta, Qirong Ho, Xue Liu, Oana Balmau  
**Category**: cs.DC  
**Published**: 2026-08-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.28848v1  

#### Abstract
LLM serving systems are provisioned for peak load to meet strict latency targets, leaving substantial GPU compute idle whenever traffic falls below peak. We present DeltaServe, a host-agnostic co-serving design that converts this idle inference capacity into LoRA fine-tuning throughput while preserv...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DeltaServe: Host-Agnostic Co-Serving of Inference and Fine-Tuning for LLMs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）的推理服务系统通常为应对**峰值负载**而配置资源，以满足严格的延迟目标（latency SLOs）。然而，在非高峰时段，大量 GPU 计算能力处于闲置状态。与此同时，许多部署需要持续进行模型微调（如 LoRA fine-tuning），但缺乏独立的 GPU 资源。

现有方法存在以下不足：
- **传统资源共享机制**（如时间切片、空间分区）粒度过粗，难以适应推理负载的亚秒级波动，容易违反延迟约束。
- **已有共服务系统**（如 LLMStation、FlexLLM）要么局限于 decode 阶段的细粒度融合，要么引入高昂的数据移动开销。

DeltaServe 提出了一种**在不牺牲推理 SLO 的前提下，利用空闲 GPU 容量执行 LoRA 微调**的新范式。

---

### 提出的新方法与创新思路

DeltaServe 是一种**主机无关的共服务设计**（host-agnostic co-serving design），其核心思想是将 LoRA 微调无缝集成到现有推理引擎中，而非构建一个全新的服务系统。

#### 主要创新点包括：

1. ✅ **SLO 感知的微调准入调度器（SLO-aware Admission Scheduler）**
   - 利用 **inference prefill** 和 **LoRA fine-tuning forward pass** 在计算结构上的相似性，将微调样本作为“伪预填充请求”插入推理批次中。
   - 基于分析型延迟模型预测每一步对 TTFT（Time-To-First-Token）和 TPOT（Time-Per-Output-Token）的影响，仅在有足够延迟余量时才允许微调任务进入。

2. ✅ **CUDA-Graph-Aware 延迟模型**
   - 构建轻量级解析模型，区分 **graph execution** 与 **eager execution** 模式下的执行时间差异。
   - 模型通过离线剖面分析初始化，并在线动态校准，提升预测精度。

3. ✅ **解耦的反向传播子进程（Decoupled Backward Executor）**
   - 反向传播运行在一个独立的 GPU 子进程中（under CUDA MPS），与主推理路径隔离。
   - 支持在每个 transformer 层边界处让步于推理任务，并在新请求到达时被抢占，确保只消耗“否则闲置”的计算资源。

4. ✅ **主机无关架构（Host-Agnostic Design）**
   - 不依赖特定推理引擎内部实现，仅需宿主支持 **multi-LoRA batching** 功能（现代系统普遍具备）。
   - 通过一组紧凑的集成钩子（integration hooks）实现扩展，可轻松适配不同 LLM serving 引擎。

---

### 相比现有方法的优势

| 特性 | DeltaServe | LLMStation | FlexLLM |
|------|----------|-----------|--------|
| 共享阶段 | Prefill & Decode | Decode-only | Token-level interleaving |
| 执行模式感知 | ✅ 支持 CUDA graph/eager 区分 | ❌ 忽略 graph/eager 差异 |
| 内存效率 | ✅ 激活缓存可控 | ⚠️ 缺乏优化 | ❌ 每 chunk 重载模型 |
| 调度粒度 | Batch-level + Layer-level preemption | Step-level | Token-chunk level |
| 对宿主侵入性 | 极低（仅需 multi-LoRA batching） | 高（基于 vLLM 改造） | 高 |
| SLO 合规率 | 100% | ~85% | 未报告 |

---

## 2. 核心实验方法和设置

### 数据集
- **推理工作负载**：
  - **Nutanix 生产轨迹**：来自 Nutanix 实际部署的 20 分钟推理请求日志，具有多尺度突发性和长尾分布。
  - **合成负载**：
    - `burst-light`：短时高峰（2秒@80 RPS）
    - `burst-dense`：长时间高峰（4秒@120 RPS）

- **微调数据集**：
  - **Alpaca instruction-tuning dataset**：广泛用于 LoRA 微调基准测试的指令-响应对语料库。

---

### 实验设置
- **硬件平台**：
  - 单卡消费级：NVIDIA RTX 5090 (32GB)
  - 多卡数据中心：4×NVIDIA A100 (40GB)

- **模型**：
  - **Llama 3-8B**：带 group-query attention 的主流开源大模型。

- **集成实现**：
  - DeltaServe-vLLM（基于 vLLM 0.21.0）
  - DeltaServe-SGLang（基于 SGLang）
  - DeltaServe-S-LoRA（基于 S-LoRA）

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **Fine-tuning throughput (tok/s)** | 每秒处理的微调 token 数量 |
| **Inference SLO compliance rate** | 请求满足 TTFT ≤ 400ms / TPOT ≤ 120ms 的比例 |
| **End-to-end latency (avg & tail)** | 推理请求端到端延迟均值及尾部延迟 |
| **Total throughput** | 推理 + 微调总吞吐量 |

---

### 基线方法对比
1. **LLMStation**：当前最先进的推理-微调共服务系统，专注于 decode 阶段融合。
2. **Split-Pool Baseline (vLLM + torchtune)**：
   - 3 张 GPU 运行 vLLM 推理
   - 1 张 GPU 专用 torchtune 执行 LoRA 微调
   - 无协调，代表典型的生产部署方式。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（基于 Nutanix 生产轨迹，4-GPU 设置）

| 系统 | Fine-tuning Throughput (tok/s) | SLO Compliance Rate | Avg. E2E Latency |
|------|-------------------------------|---------------------|------------------|
| **DeltaServe-vLLM** | **1418** | **100.0%** | 2.731s |
| LLMStation | 489 | 85.5% | 3.708s |
| vLLM + torchtune (split-pool) | 1014 | 100.0% | 1.879s |

> 💡 **结论**：DeltaServe 在保持 **100% SLO 合规率**的同时，实现了：
- **2.9× 更高的微调吞吐量** vs LLMStation
- **+39% 更高的微调吞吐量** vs 分池部署（无需额外硬件）

---

### 与其他场景对比

| 场景 | DeltaServe vs LLMStation | DeltaServe vs Split-Pool |
|------|--------------------------|---------------------------|
| **Burst-light** | 3.5× 更高 FT 吞吐 | +77% 更高 FT 吞吐 |
| **Burst-dense** | 2.6× 更高 FT 吞吐 | +21% 更高 FT 吞吐，且更低推理延迟 |

> 🔍 在高负载下，DeltaServe 能智能抑制微调任务，优先保障推理性能；而在低负载期则积极利用空闲资源，展现出优秀的动态适应能力。

---

### 消融实验结果

#### （1）Forward Batch Co-Serving 消融（禁用批内共服务）
- **DeltaServe-Temp**（仅在空闲步骤运行微调）：
  - 微调吞吐：507 tok/s
  - 启用完整批内共服务后（DeltaServe-vLLM）：
    - 微调吞吐提升至 **934 tok/s**（↑84%）
    - 平均延迟略有上升但仍满足 SLO

> ✅ 表明 **将微调融入推理批次** 是提升吞吐的关键机制。

#### （2）Fine-tuning Forward Interruption 消融
- **启用中断机制**（默认）：
  - 5% 尾部延迟仅比纯推理高 **8.1%**
- **禁用中断机制**：
  - 5% 尾部延迟增加 **27.8%**
  - 微调吞吐仅提高 2%

> ✅ 证明 **层边界中断机制** 能有效保护推理尾部延迟，代价极小。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **结构相似性可被高效利用**：LoRA 微调 forward 与 inference prefill 的执行结构高度一致，使其天然适合以“轻量级请求”形式嵌入推理流程。
2. ✅ **SLO 感知调度至关重要**：基于 CUDA-graph-aware 的延迟模型能精准预测干扰，实现安全、高效的资源复用。
3. ✅ **解耦反向传播是可行路径**：将 BP 移出关键路径并支持抢占，既能完成训练任务，又不影响推理服务质量。
4. ✅ **主机无关设计增强实用性**：仅依赖 multi-LoRA batching 的松耦合架构，使 DeltaServe 成功集成至 vLLM、SGLang 和 S-LoRA，验证了其通用性。

---

### 方法的局限性
- **依赖 multi-LoRA batching 支持**：若宿主系统不支持该功能，则无法直接应用。
- **激活内存占用**：虽可通过 offload 缓解，但在极端内存受限场景仍可能成为瓶颈。
- **目前聚焦 LoRA**：尚未扩展至其他 PEFT 方法（如 IA³、Adapter）或其他训练范式。

---

### 未来工作方向
- 🔄 支持更多类型的 **PEFT 方法共服务**
- 🧠 探索 **更精细的调度策略**，例如基于价值感知的任务排序
- ☁️ 向 **分布式多节点环境** 扩展，实现集群级资源协同
- 📊 引入 **自适应预算分配机制**，根据历史负载动态调整 SLO 余量使用策略

---

> ✅ **总体评价**：DeltaServe 提供了一个实用、高效且易于集成的框架，成功解决了 LLM 服务中“资源浪费”与“微调成本高”的矛盾，在不增加硬件投入的前提下显著提升了系统整体利用率。

</details>

---

### 15. [EarlyDx: An Admission-Anchored Benchmark for Open-Ended Generation of Evidence-Supported ED-Encounter Diagnoses](https://arxiv.org/abs/2607.28788)

**Authors**: Jiahui Li, Ruili Fang, Zishuai Liu, Yutong Guo, Nan Yang, Wenzhan Song, Jin Lu, Fei Dou  
**Category**: cs.AI  
**Published**: 2026-08-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.28788v1  

#### Abstract
Clinical diagnosis at hospital admission must be made rapidly from limited, incomplete evidence. Existing diagnosis-prediction benchmarks are poorly suited to this setting: they restrict prediction to closed code sets, exclude free-text notes, and supervise with discharge diagnoses that incorporate ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EarlyDx: An Admission-Anchored Benchmark for Open-Ended Generation of Evidence-Supported ED-Encounter Diagnoses

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的临床诊断预测基准（如 MDS-ED、MC-BEC）存在三大缺陷，使其不适用于评估大型语言模型（LLMs）在急诊科（ED）早期诊断中的能力：
1. **封闭标签集**：将自由文本诊断映射为固定的 ICD 编码，丢失大量临床细节（如骨折位置、移位程度等）。
2. **模态受限**：排除了主诉、影像报告、既往病史等丰富的自由文本信号，仅使用结构化表格特征。
3. **时间不一致**：输入是入院时的数据，但监督信号却是基于整个住院过程后确定的出院诊断，导致模型依赖于入院时尚未获得的信息。

这些限制使得现有基准无法真实反映模型在“不确定性下进行证据合成”的临床推理能力。

### 提出了什么新方法或新思路
作者提出了 **EarlyDx** —— 一个大规模、以入院时间为锚点（admission-anchored）、面向开放生成任务的早期诊断基准，其核心创新包括：

- **开放词汇诊断生成**：保留原始 MIMIC-IV 中的自由文本诊断标题，支持细粒度、临床相关的输出（如 “R hand, 2nd metacarpal base fracture” 而非笼统的 “hand fracture”）。
- **证据支持验证机制（Evidence-Grounded Verification）**：引入一个 LLM 审计员（LLM auditor），对每个参考诊断是否被入院时的证据所支持进行分类：
  - **Supported**：有直接证据（如实验室、影像、生命体征）
  - **Partially Supported**：仅有间接线索（如既往病史、家庭用药）
  - **Unsupported**：完全无证据或依赖后续信息
- **主赛道仅评估 Supported 标签**：确保模型必须基于当前可得证据进行推断，而非记忆或猜测。

### 相比现有方法的优势
| 维度 | 现有基准（如 MDS-ED） | EarlyDx |
|------|------------------------|---------|
| 标签形式 | 封闭 ICD 编码 | 开放自由文本 |
| 监督时间 | 出院诊断（retrospective） | 急诊期间诊断（contemporaneous） |
| 输入模态 | 多为结构化数据 | 包含多源文本（主诉、影像、病史等） |
| 评估逻辑 | 预测准确性 | 证据支持性（diagnostic validity） |

EarlyDx 更贴近真实临床决策场景，强调“诊断”而非“预测”。

---

## 2. 核心实验方法和设置

### 数据集
- **来源**：从 **MIMIC-IV** 数据库中提取，涵盖 ED、Hospital 和 Note 模块。
- **样本量**：共 **154,834** 次因急诊就诊而入院的患者记录。
- **时间窗口**：以 `admittime` 为时间锚点 $t_0$，只保留 $t_0 + W$ 时间前的记录。主基准设 $W=0$，即严格限制为入院时刻前的所有信息。
- **标签来源**：来自 MIMIC-IV-ED 的诊断表，保留为开放文本，并过滤掉症状码（R类）和非特异性编码（NOS/NEC）。

### 实验设置和评估指标
#### 输入构造
每条记录序列化为单一文本提示，包含以下字段：
- 人口统计与到达方式
- 主诉与分诊生命体征
- ED 连续生命体征
- 家庭用药（medrecon）
- 基线测量（OMR）
- 实验室结果
- ECG 解读
- 超声心动图
- 影像学发现
- 既往病史

#### 评估协议
由于诊断为自由文本，采用 **LLM-as-Judge** 协议进行语义匹配：
- 使用固定裁判模型 **qwen3.5-27B** 进行一对一匹配。
- 输出为匹配对数 $m$，计算 micro-averaged **Precision, Recall, F1**：
  - $P = m / |\hat{Y}|$
  - $R = m / |Y|$

**主赛道（Primary Track）**：仅评估 **Supported** 类别的诊断。

#### 基线方法对比
| 类型 | 模型 |
|------|------|
| **Zero-shot LLMs** | GPT-5.5, Claude Opus 4.8, GLM-5.2, Nemotron-550B, MedGemma-4B, OpenBioLLM-8B, HuatuoGPT-o1-8B |
| **Few-shot** | 在 GPT-5.5 和 Claude 上提供 5 个示例 |
| **Supervised Classifier** | ClinicalBERT 多标签分类器 |
| **Post-trained Model** | Qwen3.5-4B 微调模型（使用 gold-conditioned CoT 监督） |
| **Human Reference** | 临床医生独立诊断 1,000 例 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Supported Track, micro-F1）

| 模型 | Precision | Recall | F1 |
|------|----------|--------|-----|
| **Qwen3.5-4B (ours, post-trained)** | **0.48** | **0.54** | **0.51** ✅ |
| GPT-5.5 (zero-shot) | 0.27 | 0.77 | 0.40 |
| Claude Opus 4.8 | 0.17 | 0.83 | 0.28 |
| MedGemma-4B | 0.26 | 0.40 | 0.32 |
| ClinicalBERT (supervised) | 0.22 | 0.25 | ~0.23 |
| **Clinician (human)** | 0.20 | **0.68** | 0.31 |

> 注：人类医生召回率最高，但因列出更广泛的鉴别诊断而导致精度较低。

### 与基线方法的对比结果
- **微调模型显著优于零样本模型**：post-trained Qwen3.5-4B 达到 **0.51 F1**，比最强的 zero-shot 模型（GPT-5.5, 0.40）高出 **+0.11**。
- **医学专用模型未表现出优势**：MedGemma、OpenBioLLM 等医学 LLM 表现平平，说明问题瓶颈不在领域知识，而在**如何在不确定性下做出校准诊断**。
- **判别式模型表现有限**：ClinicalBERT（F1 ~0.23）虽优于检索基线，但仍远低于生成式微调模型。

### 消融实验结果

#### （1）显式 vs 隐式诊断（Extraction vs Inference）
- 仅 **43%** 的 supported 诊断可在输入中直接找到（explicit），其余 **57% 必须推断**（implicit）。
- Zero-shot 模型在 explicit 上表现尚可（37–65% recall），但在 implicit 上崩溃（仅 **3–31% recall**）。
- 微调模型将 implicit recall 提升至 **56%**，表明其具备更强的推理能力。

> 👉 结论：当前 LLMs 主要靠**提取**而非**推断**；微调能提升推理能力，但仍有差距。

#### （2）输入消融（Input Ablation）
当仅保留人口统计和主诉（demographics + chief complaint）时：
- 微调模型的 supported F1 从 **0.51 → 0.34**
- 表明模型确实在利用多模态证据，而非拟合标签先验。

#### （3）完整匹配率（Complete Match Rate）
- 微调模型在 **32–36%** 的病例中能完全匹配参考诊断集合。
- 所有 zero-shot 模型均低于 **4%**。
- 显示微调模型能更准确地捕捉整个诊断组合。

#### （4）时间敏感疾病的风险加权评估
针对六大急症（心梗、败血症、脑出血等）：
- 临床医生操作点：**78% recall @ 44% precision**
- Zero-shot 模型：高 recall（~70–85%）但低 precision（25–28%）
- 微调模型：高 precision（78%）但 recall 下降（54%）
- ❗ **没有系统能达到医生的平衡点**，凸显当前方法在关键任务上的不足。

---

## 4. 关键结论和发现

### 主要发现
1. **LLMs 当前主要依赖文本提取**：大多数 zero-shot 模型只能恢复明确提及的诊断，对需要推理的隐式诊断几乎失效。
2. **微调可提升推理能力**：任务对齐的微调能显著提高对隐式诊断的召回（从 <30% → 56%），但仍未达到临床可用水平。
3. **证据支持性是关键区分维度**：模型在 supported vs partially supported 诊断上表现差异巨大（F1: 0.51 vs 0.27），验证了该划分的有效性。
4. **无系统达到临床医生的敏感性-精确性平衡**：尤其在时间敏感病症上，所有模型都无法复现医生“宁可过报也不漏报”的合理权衡。

### 方法的局限性
- **单中心数据**：来自 MIMIC-IV，可能不具备全国代表性。
- **文本渲染信号**：影像、ECG 等以报告形式呈现，非原始波形或图像。
- **参考标准偏差**：
  - 参考诊断是用于计费的行政编码，非经专家裁定的真实诊断。
  - 排除了所有 unsupported 诊断，可能导致偏倚。
- **人类评估受限**：医生仅评估 1,000 例，且基于文本记录而非实际问诊，信息不完整。

### 未来工作方向
1. **引入疑似诊断标记机制**：允许模型输出“suspected pending confirmation”，并设计相应损失函数，使系统能主动提出待排除的危重诊断。
2. **扩展到分级预测**：不仅输出诊断，还输出置信度或紧急程度。
3. **改进理由生成质量**：当前的 gold-conditioned rationale 是事后解释（post-hoc justification），缺乏前瞻性推理（forward reasoning）和不确定性表达。
4. **跨机构泛化研究**：构建多中心版本以增强外部有效性。
5. **整合床旁观察信息**：如患者外貌、呼吸努力、查体发现等目前缺失的关键线索。

---

> 🔚 **总结一句话**：  
> **EarlyDx 揭示了当前 LLMs 在真实早期诊断任务中仍严重依赖文本提取，缺乏可靠的临床推理能力；尽管微调可部分缓解，但在关键病症上尚未达到医生的决策平衡，未来需发展支持“怀疑-确认”范式的新型评估框架。**

</details>

---

### 16. [Identifying Informative Environments for Cognition Parameter Inference via Bayesian Experimental Design](https://arxiv.org/abs/2607.28894)

**Authors**: Manisha Dubey, Rimvydas Rubavicius, N. Siddharth, Subramanian Ramamoorthy  
**Category**: cs.AI  
**Published**: 2026-08-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.28894v1  

#### Abstract
Computational cognitive modeling seeks to infer latent cognitive mechanisms underlying observed behavior. Bayesian inverse planning provides a principled framework for such inference, but its success depends critically on the experimental environment. Existing approaches typically treat environments...

---

### 17. [Faster but Different: Diagnosing and Controlling Content Drift in Accelerated Multimodal Diffusion Language Models](https://arxiv.org/abs/2607.29079)

**Authors**: Yaoxuan Dou, Yang Shu  
**Category**: cs.CL  
**Published**: 2026-08-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.29079v1  

#### Abstract
Training-free acceleration makes diffusion-based multimodal large language models (dMLLMs) more deployable, but it may silently change generated content. We study this serving-time consistency problem on 300 real images, comparing Fast-dLLM outputs with the same model's unaccelerated outputs. Across...

---

### 18. [End-to-End Fairness Optimization with Fair Decision-Focused Learning](https://arxiv.org/abs/2607.29441)

**Authors**: Yu Wang (Xinying),  Violet (Xinying),  Chen  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.29441v1  

#### Abstract
Many real-world systems rely on predictive models to inform decisions, and fairness concerns arise in both the prediction and decision stages. We introduce end-to-end fairness optimization (E2EFO) as a unifying framework that integrates fairness across the prediction-to-decision pipeline. We focus o...

---

### 19. [Mixture-of-Translators: Translating KV Caches Across Heterogeneous Large Language Models](https://arxiv.org/abs/2607.28979)

**Authors**: Jin-woo Lee, Minkyung Song, Junghyun Oh, Seunghoon Han, Soyoung Park, Gwangseon Jang, Sungsu Lim  
**Category**: cs.CL  
**Published**: 2026-08-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.28979v1  

#### Abstract
Heterogeneous Large Language Model (LLM) systems increasingly rely on shared contexts, retrieved evidence, and multi-agent dialogue histories, yet their internal key-value (KV) caches remain model-specific and cannot be reused across architectures. Consequently, each model must repeatedly prefill or...

---

### 20. [From Inline Notes to Collected Commentaries: Toward Context-Preserving Organization of Exegetical Knowledge in Classical Chinese Texts](https://arxiv.org/abs/2607.29044)

**Authors**: Ke Liang, Qi Su, Churen Huang  
**Category**: cs.CL  
**Published**: 2026-08-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.29044v1  

#### Abstract
Inline notes and collected commentaries are important forms of scholarly communication that evolved within the Confucian exegetical tradition, yet have received little computational attention. Drawing on traditional Chinese exegetics and philology, this paper formulates collected commentary compilat...

---

### 21. [Overcoming the Weakest-Link Effect in LLM-Driven Program Optimization via Heterogeneous Edit Recombination](https://arxiv.org/abs/2607.28947)

**Authors**: Jingwen Fu, Zhen Liu, Yuhan Liu, He Zhang, Nanning Zheng  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.28947v1  

#### Abstract
Large language models (LLMs) are increasingly used to solve complex problems by searching over program space, offering a general paradigm for scientific problems that can be naturally represented and solved as programs. Despite recent progress, identifying effective optimization directions for a can...

---

### 22. [Assessing the Generalization of Graph Neural Networks for Fault Location Across Increasing Distributed Energy Resource Penetration Levels](https://arxiv.org/abs/2607.29293)

**Authors**: Burak Karabulut, Olayiwola Arowolo, Carlo Manna, Chris Develder, Jochen L. Cremer  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.29293v1  

#### Abstract
Accurate fault location is critical for distribution network reliability. However, increasing distributed energy resource (DER) penetration complicates fault location due to intermittent generation and bidirectional power flows that reshape fault signatures. Spatio-Temporal Graph Neural Networks (ST...

---

### 23. [Adaptive FastOPD: Progress-Aware Rollout Horizon Expansion for Efficient On-Policy Distillation](https://arxiv.org/abs/2607.29494)

**Authors**: Qian Tan, Huaifei Liang, Xuanyu Zhu, Lei Jiang, Yuqiang Li  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.29494v1  

#### Abstract
On-policy distillation (OPD) provides dense teacher supervision along student-generated trajectories, but its online rollout process incurs substantial computational cost, particularly when a few long responses delay batch completion. Existing acceleration methods typically control rollout length us...

---

### 24. [How Hard Does It Think? Analyzing Step-Aware Reasoning Energy in LLM Chain-of-Thought Trajectories](https://arxiv.org/abs/2607.28674)

**Authors**: Hui Wei, Junda Wu, Sheldon Yu, Sizhe Zhou, Yizhu Jiao, Ming Zhong, Bowen Jin, Tong Yu, Shijia Pan, Jiawei Han, Julian McAuley  
**Category**: cs.AI  
**Published**: 2026-08-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.28674v1  

#### Abstract
Understanding how computational effort is allocated across individual chain-of-thought (CoT) reasoning steps remains an open challenge: existing interpretability methods rely on output-level signals or collapse processing depth into a single trajectory-level scalar, leaving step-wise effort opaque. ...

---

### 25. [Scaling Scientific Discovery Environments for Turn-Level Agentic RL](https://arxiv.org/abs/2607.28990)

**Authors**: Yucheng Xu, Keyi Zhang, Yuyang Yu, Min Zhang, Shiyuan Meng, Pei Chu, Zhongying Tu  
**Category**: cs.AI  
**Published**: 2026-08-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.28990v1  

#### Abstract
Large language model agents have shown promising capabilities in data-driven scientific discovery tasks, where an agent interacts with an execution environment and produces a statistical claim. Long-horizon scientific analysis remains constrained by the lack of process supervised environments over r...

---

### 26. [Token-Level Diagnosis of Sycophancy in LLMs with Attribution-Guided Steering](https://arxiv.org/abs/2607.28906)

**Authors**: Hieu Nguyen, Mahammed Kamruzzaman, Anshuman Chhabra, Gene Louis Kim  
**Category**: cs.CL  
**Published**: 2026-08-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.28906v1  

#### Abstract
Sycophancy refers to the tendency for large language models (LLMs) to match user beliefs at the cost of factual correctness, thereby undermining model reliability. Prior work on evaluating sycophancy in LLMs aims to assess whether a model's output matches an authority's claim, but cannot reveal whic...

---

### 27. [System-Wide Termination in Distributed Betweenness Centrality Computation](https://arxiv.org/abs/2607.29474)

**Authors**: Siamak Abdi, Lucia Cavallaro, Giuseppe Di Fatta  
**Category**: cs.DC  
**Published**: 2026-08-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.29474v1  

#### Abstract
Computing betweenness centrality on large networks is inherently expensive, as it requires aggregating shortest-path dependencies across all pairs of vertices and becomes increasingly difficult to scale as network size grows. Scalable distributed algorithms can facilitate such computations, particul...

---

### 28. [Learning Optimal Dynamic Matching via Graph Neural Networks](https://arxiv.org/abs/2607.28925)

**Authors**: Genta Okada, Shunya Noda, Junpei Komiyama, Akira Matsushita  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.28925v1  

#### Abstract
Dynamic matching markets require decisions about whom to match and when: matching now yields value but removes participants who may create better future opportunities. We develop a value-based reinforcement-learning framework for this problem on finite, evolving weighted graphs. We study an infinite...

---

### 29. [Sample Efficient Hierarchical Reinforcement Learning via Best Policy Identification](https://arxiv.org/abs/2607.29294)

**Authors**: Anders Jonsson, Emilie Kaufmann, Gianmarco Tedeschi, Lorenzo Steccanella  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.29294v1  

#### Abstract
We present HBPI-UCRL, a model-based algorithm for hierarchical reinforcement learning (HRL) that learns high-level and low-level policies in parallel. HBPI-UCRL exploits the fact that a high-level transition corresponds to a multi-step transition at the low level. We introduce two conditions on the ...

---

### 30. [Freeze, Then Select: Structured Field Adapters and Stability-Validated Weak Selection for PDE Discovery from Sparse Observations](https://arxiv.org/abs/2607.29665)

**Authors**: Juncheng Zhong, Chenghuang Shen, Jianfeng Liu, Zhengdong Xiao, Longjiu Luo, Qianrong Wang, Wenjun Xu, Wenlian Lu  
**Category**: cs.LG  
**Published**: 2026-08-03  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.29665v1  

#### Abstract
PDE discovery from sparse observations requires reconstructing a continuous field and selecting the correct differential terms. Our analysis of optimization paths in coupled neural PDE discovery reveals three behaviors: the exact support can persist to the end of training, appear only transiently, o...

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

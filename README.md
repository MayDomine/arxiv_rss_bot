# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-25 08:42:24 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Dustin: Draft-Augmented Sparse Verification for Efficient Long-Context Generation with Speculative Decoding](https://arxiv.org/abs/2606.24957)

**Authors**: WenHung Lee, Jian-Jia Chen, Xiaolin Lin, Pei-Shuo Wang, Chi-Chih Chang, Chun-Che Yang, Ning-Chi Huang, Grace Li Zhang, Kai-Chiang Wu  
**Category**: cs.CL  
**Published**: 2026-06-25  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2606.24957v1  

#### Abstract
While speculative decoding improves inference throughput for multi-batch long-context Large Language Models (LLMs), its efficiency is often limited by a verification bottleneck where Key-Value (KV) cache loading dominates latency. Existing compression methods fail in this regime: static eviction inc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Dustin: Draft-Augmented Sparse Verification for Efficient Long-Context Generation with Speculative Decoding》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**多批次、长上下文场景下的大语言模型（LLMs）推理**中，尽管 **Speculative Decoding (SD)** 能提升吞吐量，但其效率受限于 **Verification 阶段的 KV Cache 加载瓶颈**。随着上下文长度增长（如 32k），KV 缓存的内存带宽开销成为延迟主导因素。

现有 KV Cache 压缩方法存在以下问题：
- **静态压缩（如 SnapKV）**：因“saliency shift”现象导致精度下降（重要token集合随时间变化）。
- **动态选择（如 Quest）**：每步重新计算 token 重要性带来过高计算开销，尤其在验证路径上不可接受。

### 🚀 提出的新方法：Dustin
Dustin 是一种专为 **长上下文 + 多批次 + Speculative Decoding** 设计的 **稀疏验证框架（sparse verification framework）**，核心思想是：
1. **融合双信号进行关键 token 识别**：
   - 利用 **draft model 的 lookahead 信号**（前瞻预测）
   - 结合 **target model 的历史 attention 分布**
   → 构建更鲁棒、高保真的 token 重要性估计。
2. **引入稀疏估计机制（Sparse Estimation Scheme）**：
   - 仅对少量选定的 **Semantic Retrieval Heads (SRHs)** 进行注意力分数重计算。
   - 显著降低在线重要性评分的计算开销。

### 🔍 相比现有方法的优势
| 方面 | Dustin | 现有方法（SnapKV / Quest / MagicDec） |
|------|--------|-------------------------------|
| 准确性 | 接近无损（negligible accuracy degradation） | 静态方法易损失精度；动态方法依赖单一信号 |
| 效率 | 极大减少 self-attention 和端到端延迟 | 动态方法验证路径开销大 |
| 可扩展性 | 批次和上下文越长，优势越明显 | 性能增益随 context 增长趋于饱和 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **PG-19**：用于评估 **长上下文生成任务的吞吐量与延迟**（generation throughput and latency）。
- **LongBench**：涵盖多种任务类型，评估 **长上下文理解能力**，包括：
  - 单文档问答（Single-doc QA）
  - 多文档问答（Multi-doc QA）
  - 摘要生成（Summarization）
  - 少样本学习（Few-shot）
  - 合成任务（Synthetic）
  - 代码相关任务（Code）

### ⚙️ 实验设置
- **模型家族**：
  - 主要测试：`Qwen2.5-72B`（效率）、`Llama-3.3-70B`（准确性）
  - 配套轻量级同族 draft model（如 `Qwen2.5-0.5B`, `Llama-3.2-1B`）
- **硬件平台**：
  - NVIDIA H200 GPU，bfloat16 精度
  - 使用 Pipeline Parallelism（4×H200 用于 72B 模型）
- **输入配置**：
  - 上下文长度：8K ~ 32K tokens
  - Batch size：8 和 16
- **评估指标**：
  - **Self-attention latency**：分解各阶段耗时
  - **Decode-stage throughput (tokens/s)**：端到端解码速度
  - **Speedup**：相对于 Vanilla 或 ClassicSD 的加速比
  - **Accuracy**：LongBench 各子任务得分及平均分

### 🆚 基线方法对比
#### 准确性基线（Accuracy Baselines）：
| 方法 | 类型 | 特点 |
|------|------|------|
| Vanilla / Lossless SD | 无压缩 | 不进行 KV 压缩，作为准确率上限参考 |
| StreamingLLM | 流式保留 | 仅保留 sink tokens + 最近滑动窗口 |
| SnapKV | 无训练压缩 | 基于 prompt 末尾 attention 模式固定选择重要位置 |
| Quest | 页面级选择 | Block-level KV selection，适用于 decoder 层 |

#### 效率基线（Efficiency Baselines）：
| 方法 | 是否使用 SD | KV 策略 |
|------|-------------|---------|
| Vanilla | 否 | 自回归逐个生成 |
| ClassicSD | 是 | 全量 KV cache 验证 |
| MagicDec | 是 | Sparse-KV draft model + Full target verification |

> 所有方法均启用 FlashAttention-v2 以公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（基于 Qwen2.5-72B，batch=16，context=32k）

| 指标 | Dustin 表现 | 对比说明 |
|------|------------|----------|
| **Self-attention 加速比** | **27.85×** | 相比 full-cache speculative verification |
| **端到端解码速度提升** | **9.17×** | 相比 Vanilla baseline |
| **Criticality Estimation 开销** | < 0.8% of full hybrid comp. | 在线开销极低 |
| **SRH 数量占比** | 极小比例 heads/layers | 实现高效稀疏估计 |

> ✅ 随着 batch size 和 context length 增加，Dustin 的优势持续扩大。

#### 🔁 不同 context 下的端到端加速（Qwen2.5-72B, batch=16）
| Context Length | Dustin Speedup | SpecAttn (对比) |
|----------------|----------------|------------------|
| 8K             | 4.76×          | 3.91×            |
| 16K            | 7.06×          | 5.62×            |
| 32K            | **9.17×**      | **7.11×**        |

→ Dustin 在所有长度下均显著优于 SpecAttn。

---

### 🧪 消融实验结果（Ablation Studies）

#### （1）不同信号组合变体对比（Table 3 & Table 14）
| 方法 | 使用信号 | TriviaQA Acc. | MultiNews Acc. | 平均 Δ 准确率 |
|------|----------|---------------|----------------|----------------|
| Dustin-T | 仅 Target 历史 attention | 77.58% (-6.41%) | 23.95% (-1.12%) | -1.91% |
| Dustin-D | 仅 Draft lookahead | 80.51% (-3.48%) | 24.84% (-0.23%) | -0.71% |
| **Dustin-H (完整版)** | **Hybrid 融合信号** | **83.63% (-0.36%)** | **25.19% (+0.12%)** | **-0.59%** |

✅ 结论：**融合信号显著优于单一信号**，接近 lossless 表现。

#### （2）预算分配敏感性分析（Appendix K）
- 固定比例参数 `m`（控制 draft vs target 权重）通过离线调优获得。
- “Oracle”策略（每任务最优信号组合）显示仍有 ~0.3% 提升空间。
→ 当前静态策略已接近理想性能，但动态调整可进一步优化。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Verification 阶段是长上下文 SD 的主要瓶颈**，占解码延迟高达 **87.5%**（Fig. 1）。
2. **单独依赖历史 attention 或 draft lookahead 都不可靠**，尤其跨尺度和深度验证时。
3. **Dustin 通过 hybrid signal fusion + sparse head estimation**，实现了：
   - 极高的 self-attention 加速（27.85×）
   - 显著的端到端提速（9.17×）
   - 几乎无损的生成质量（accuracy drop < 1%）
4. **加速收益随 context 和 batch 增大而增强**，特别适合大规模部署场景。

---

### ⚠️ 方法的局限性
1. **静态预算分配**：
   - 参数 `m` 固定，无法根据输入内容动态调整。
   - 存在与“oracle”策略之间的性能差距。
2. **不减少 KV Cache 内存占用**：
   - Dustin 是 **sparse verification** 而非 **eviction**。
   - 仍需索引完整历史，**不能降低 VRAM footprint**。
   - 因此 **无法支持更长序列装入有限 GPU 显存**。

---

### 🔮 未来工作方向
1. **设计轻量级、输入感知的动态预算控制器**：
   - 根据任务类型或输入特征实时调节 `m`。
2. **结合 eviction 机制实现 memory footprint 压缩**：
   - 将 Dustin 的 selection logic 与物理删除结合，突破显存限制。
3. **扩展至 Tree-based Speculation 或其他加速范式**。
4. **探索更高效的 SRH 自动识别算法**，降低离线开销（当前约 12–35 分钟）。

---

## ✅ 总结
Dustin 提出了一种面向 **长上下文多批次推理** 的新型稀疏验证框架，通过 **融合 draft lookahead 与 target historical attention 信号**，并采用 **稀疏头估计（sparse estimation over SRHs）**，有效解决了传统 KV 压缩方法在 Speculative Decoding 中面临的 **精度-效率权衡难题**。实验表明其在保持几乎无损准确率的同时，实现了高达 **27.85× 的 self-attention 加速** 和 **9.17× 的端到端解码提速**，为高吞吐 LLM 推理提供了实用且高效的解决方案。

</details>

---

### 2. [Speculation at a Distance: Where Edge-Cloud Speculative Decoding Actually Pays Off](https://arxiv.org/abs/2606.25091)

**Authors**: Yuan Lyu, Bharath Irukulapati, Jaya Prakash Champati  
**Category**: cs.DC  
**Published**: 2026-06-25  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.25091v1  

#### Abstract
Speculative decoding (SD) accelerates LLM inference by $1.5$-$3$ times when the draft and target models are co-located. This has motivated a distributed variant (DSD) that places the draft model on an edge device while the target stays in the cloud. We show with closed-form inequalities that DSD's p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Speculation at a Distance: Where Edge-Cloud Speculative Decoding Actually Pays Off*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
本文系统性地分析了**分布式推测解码**（Distributed Speculative Decoding, DSD）在边缘-云（edge-cloud）架构下的实际收益，尤其是当边缘设备运行草稿模型（draft model）、云端运行目标模型（target model）时，在广域网（WAN）延迟下是否真的能加速推理。

当前许多研究声称 DSD 可提升 1.1–2.9× 推理速度，但这些结果往往基于不合理的假设或与错误的基线比较（如仅对比 cloud autoregressive decoding）。本文指出：**DSD 是否“划算”取决于部署场景和评价维度**。

### 🧩 提出的新方法/新思路
本文并未提出新的 DSD 系统，而是通过**闭式数学建模**（closed-form inequalities）对 DSD 进行理论分析，明确其适用边界，并重新定义其价值定位：

- 区分 **单请求延迟**（per-request latency） 和 **多租户容量**（multi-tenant capacity） 两种评价范式；
- 明确指出：**DSD 在单请求延迟上通常劣于 co-located SD**（即草稿与目标模型共置于云端）；
- 首次形式化证明：**DSD 的真正优势在于提升服务器可支持的并发客户端数量**，尤其是在跨客户端重叠（cross-client overlap）条件下；
- 提出应以 **server throughput 和 multi-tenant capacity** 而非单用户延迟作为 DSD 的核心评估指标。

### 🔍 相比现有方法的优势
| 维度 | 本文贡献 |
|------|--------|
| **理论深度** | 提供闭式表达式分析 latency、throughput、FLOPs、memory、network 开销等，超越经验性报告 |
| **基准校准** | 强调应将 **co-located SD** 而非 cloud AR 作为首要 baseline，纠正文献中的误导性比较 |
| **价值重构** | 将 DSD 定位为“容量增强技术”而非“低延迟技术”，指导后续系统设计方向 |

---

## 2. **核心实验方法和设置**

本文是**理论分析型论文**，未进行原始实验，而是基于已有系统的公开测量数据验证其模型预测。

### 📚 使用的数据来源与支持证据
- 引用并复现了多个已有工作的实测数据来支撑理论推导：
  - **[12]** 对 OPT-66B 和 LLaMA-65B 的 350+ 次 co-located SD 实验，用于验证 `teff` 模型；
  - **DSSD [4]** 的网络延迟测试结果，验证 Prop. 2 中的 RTT break-even 条件；
  - **SLED [7]** 和 **SpecEdge [10]** 的吞吐量提升数据，支持多租户容量增益；
  - **CoSine [15]** 数据中心内分离计算的经验，佐证资源卸载的有效性。

### ⚙️ 实验设置与建模假设
| 类别 | 设置说明 |
|------|---------|
| **三种配置对比** |  
| - Cloud AR | 仅云端运行目标模型，无推测机制 |
| - Co-located SD | 草稿与目标模型均位于云端，共享内存 |
| - DSD (同步/异步) | 草稿在边缘，目标在云端，需网络通信 |
| **关键参数** |  
| - `RTT` | 广域网往返延迟（典型值：4G ~60ms，跨区域 ~80ms）  
| - `ta` | 边缘每 token 草稿生成时间（例：10ms）
| - `ts`, `tar` | 验证与 autoregressive 步骤耗时（假设 `ts ≈ tar`，memory-bound）
| - `γ` | 每轮推测长度（如 γ=5）
| - `α` | 单位置接受概率（acceptance rate）

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| `Te` | 每输出 token 的期望 wall-clock 时间（越小越好） |
| `N_client` | 单服务器可维持的最大并发客户端数（衡量 capacity） |
| `Throughput` | 总体 goodput（tokens/sec），反映系统级效率 |
| `Network overhead` | 每轮传输开销（bytes, RTT 敏感） |
| `FLOPs per token` | 计算成本 |
| `Model memory` | 权重存储需求 |

### 🆚 基线方法对比
| 基线 | 说明 |
|------|------|
| **Cloud AR** | 传统自回归解码，作为最弱 baseline |
| **Co-located SD** | 当前最优本地推测方案，应作为主 baseline |
| **DSD variants** | 如 DSSD、PicoSpec、PipeSD、SLED 等已有系统 |

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据与对比结果

#### （1）单请求延迟：DSD 多数情况下不如 co-located SD
- **Proposition 1**：若服务器能容纳两个模型，则 **co-located SD 在延迟、通信、计算、内存上全面优于或等于 DSD**。
  - 原因：DSD 引入额外的 RTT 和传输开销，而 co-located SD 无网络延迟。
  - 结论：只要可行，**co-located SD 是更优选择**。

#### （2）DSD vs Cloud AR：仅在有限 RTT 窗口内有优势
- **Proposition 2** 给出 break-even 条件：
  $$
  \text{RTT} < \frac{\alpha}{1-\alpha} t_{ar} - \gamma t_a
  $$
- 表格 III 展示不同 `tar` 和 `α` 下的最大容忍 RTT（单位：ms）：

| Cloud AR `tar` | α=0.5 | α=0.7 | α=0.85 | α=0.9 |
|----------------|-------|-------|--------|-------|
| Slow (100ms)    | 47    | 144   | 265    | 319   |
| Standard (50ms) | —     | 47    | 108    | 134   |
| Fast (30ms)     | —     | 8     | 45     | 61    |
| Very fast (20ms)| —     | —     | 13     | 24    |

> 注：“—”表示即使 RTT=0，DSD 也慢于 Cloud AR。

- **现实意义**：
  - 在 4G 网络（RTT≈60ms）下，只有较慢的目标模型（tar≥50ms）且高接受率（α≥0.85）才可能受益；
  - 跨区域部署（RTT≈80ms）几乎无法满足条件。

#### （3）多租户容量：DSD 显著胜出
- **Proposition 9** 形式化容量增益：
  $$
  N_{\text{dsd}} / N_{\text{coloc}} = 1 + \frac{\gamma t_a}{t_s}
  $$
  即：DSD 可支持 `(1 + γta/ts)` 倍于 co-located SD 的并发客户端数。

- 示例：若 `γ=5`, `ta=10ms`, `ts=50ms` → 增益因子为 `1 + 5×10/50 = 2×`

- 支持证据：
  - **SLED [7]**：DSD 架构实现 **2.2× 系统吞吐提升** vs Cloud AR；
  - **SpecEdge [10]**：相比 co-located SD，吞吐提升 **2.22×**，验证了上述公式。

#### （4）异步流水线（Pipelining）不能逆转 WAN 下劣势
- **Proposition 13**：当 RTT > γta 时，即使采用异步流水线（如 PicoSpec），**DSD 仍无法超越 co-located SD**。
- 典型情况：
  - γta = 5 × 10ms = 50ms
  - 4G RTT ≈ 60ms > 50ms → 流水线无法掩盖延迟
- 只有在极低 RTT 场景（如局域 WiFi、5G mmWave、边缘近端）才可能反转。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **DSD 不是更快的服务方式，而是更便宜的规模化方式**：
   - 它的价值不在降低单用户延迟，而在提升服务器并发能力。
2. **Co-located SD 是更强的 baseline**：
   - 几乎所有维度都优于 DSD，除非服务器资源极度紧张。
3. **DSD 的延迟优势极其脆弱**：
   - 依赖于窄带宽窗口、高接受率、慢目标模型和低 RTT；
   - 在典型 WAN 条件下难以成立。
4. **真正的收益来自“跨客户端重叠”**：
   - 利用空闲期处理其他用户的验证任务，使服务器摆脱边缘计算瓶颈。
5. **闭源 API 用户无法使用 DSD**：
   - 缺乏 `verify-only` 接口，且商业 API 不暴露任意 token 的 logprob。

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖理想化假设** | 如 constant acceptance rate α、memory-bound verification、perfect cross-client overlap |
| **忽略排队与争用** | 分析基于单请求或饱和服务，未考虑动态负载波动 |
| **未覆盖隐私等非性能因素** | 如用户提示词上传至云端带来的隐私风险未量化 |
| **未来模型演进削弱优势** | 如 DeepSeek-V3、Qwen3-Next 已内置 multi-token prediction，降低 drafting 成本 |

### 🔮 未来工作方向
1. **建立标准评估框架**：
   - 报告 break-even RTT（RTT_max）、acceptance rate、γ、tar 的组合影响；
   - 必须同时报告 single-user latency 和 multi-tenant throughput。
2. **开发 verify-only API**：
   - 为闭源模型提供安全高效的验证接口，推动 DSD 落地。
3. **探索个性化 draft model 部署**：
   - 若每个用户拥有定制 draft model，集中部署成本过高，DSD 更具吸引力。
4. **结合加密与轻量协议**：
   - 设计保护 prefix 隐私的 DSD 协议（如 partial prefix upload）。
5. **适应 native multi-token models**：
   - 重新思考 DSD 在目标模型自带 proposal head 时代的角色。

---

## 📌 总结一句话
> **DSD 的真正价值不是让一个用户更快，而是让服务器服务更多用户——它是一个系统容量放大器，而不是单请求加速器。**

</details>

---

### 3. [CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for Resource-Efficient Long-Context LLM Inference](https://arxiv.org/abs/2606.24467)

**Authors**: Xiaolin Lin, Jingcun Wang, Olga Kondrateva, Yiyu Shi, Bing Li, Grace Li Zhang  
**Category**: cs.AI  
**Published**: 2026-06-25  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.24467v1  

#### Abstract
Long-context large language model (LLM) inference is increasingly constrained by the memory footprint and decoding cost of key-value (KV) caches, limiting sustainable deployment on resource-constrained hardware. Existing KV cache eviction methods typically apply heuristic token scoring over all head...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for Resource-Efficient Long-Context LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在基于 **Grouped Query Attention (GQA)** 的大语言模型（LLM）中，**KV-Cache** 随上下文长度线性增长，导致内存占用高、解码延迟大，限制了长上下文场景下的高效部署。现有的 **KV-Cache eviction** 方法通常对所有注意力头进行统一评分（如求和或平均），忽略了不同注意力头的功能异质性（head heterogeneity）。这导致“**Streaming Heads**”主导缓存淘汰决策，仅保留首尾 token，而中间重要的语义证据被错误地移除，从而损害模型性能。

### 🚀 提出的新方法：CompressKV
CompressKV 是一种面向 GQA 架构的资源高效的 KV-Cache 压缩框架，其核心创新在于：

- **Semantic Retrieval Head (SRH) 识别机制**  
  提出并定义了一类新的注意力头——**Semantic Retrieval Heads (SRH)**，它们不仅能定位答案 token（copy-and-paste 行为），还能关注整个答案 span 及其语义邻域。通过离线分析 calibration 数据集上的 attention 分布，计算每个 head 在正确生成时对答案 span 的总注意力质量（SRH Score），从而识别出 SRH。

- **SRH 驱动的 token 选择策略**  
  不再聚合所有头的注意力分数，而是仅使用每层中得分最高的若干个 SRH 来决定应保留的关键 token。这些 SRH 共享一组 token 索引，避免 Streaming Heads 主导淘汰过程。

- **误差感知的层自适应缓存分配（Error-Aware Layer-Adaptive Allocation）**  
  提出一种**离线估计**各层压缩误差的方法：通过比较全缓存与极小压缩缓存下 attention block 输出的 Frobenius 范数差异，量化每层对压缩的敏感度，并据此动态分配缓存预算（优先分配给误差大的层）。该过程无需在线计算，无推理开销。

### 🔍 相比现有方法的优势
| 方面 | 现有方法（如 SnapKV, CAKE, HeadKV） | CompressKV |
|------|-------------------------------------|-----------|
| **头级功能利用** | 忽视 head heterogeneity 或仅用于 head-level budget | 显式识别并利用 SRH 进行 token selection |
| **token 重要性判断** | 依赖 top-k 单个 token 注意力峰值 | 支持 span-level 语义覆盖，更鲁棒 |
| **层间预算分配** | 依赖 attention entropy/variance 等在线统计量 | 基于离线误差估计，稳定且无额外开销 |
| **性能-资源权衡** | 中等压缩下表现尚可，极端压缩下性能骤降 | 极端压缩下仍保持接近全缓存性能 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **LongBench**：包含 16 个长上下文子任务，涵盖单文档 QA、多文档 QA、摘要、代码补全、少样本学习等，支持中英双语，是综合性的长上下文理解基准。
- **Needle-in-a-Haystack (NIAH)**：将一个目标事实（needle）嵌入到长文本（haystack）中，测试模型能否准确检索并回答，专门衡量长上下文中的信息定位能力。

### ⚙️ 实验设置与评估指标
- **模型**：Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Qwen2.5-14B-Instruct、Qwen2.5-32B-Instruct。
- **KV 缓存预算**：平均每层保留 128 ~ 2048 个 token（即 `B_per-layer`），极端压缩下低至 **0.7%** 的原始 KV 存储。
- **评估方式**：
  - LongBench：报告平均得分（Acc. %）
  - NIAH：报告检索准确率（Accuracy %）
- **实现细节**：
  - SRH 离线识别一次，固定复用。
  - 层级预算也离线计算，推理时不需动态调整。
  - 所有方法仅在 prefill 阶段执行 eviction。

### 🆚 对比的基线方法
| 方法 | 类型 |
|------|------|
| **StreamingLLM** | 固定保留首尾 token |
| **SnapKV** | 观察窗口内聚类注意力得分 |
| **PyramidKV** | 动态金字塔式压缩 |
| **CAKE** | 基于 attention variance 的自适应淘汰 |
| **HeadKV / AdaKV** | 头级别预算分配，结合 retrieval signal |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & Figure 5）

#### ✅ LongBench 结果（平均得分 %）
| 方法 | Llama-3.1-8B @256 | Llama-3.1-8B @1024 | Mistral-7B @256 |
|------|-------------------|--------------------|----------------|
| FullKV | 49.08 | 47.82 | 47.82 |
| SnapKV | 45.21 | 47.82 | 43.76 |
| CAKE | 46.30 | 47.97 | 44.73 |
| **CompressKV** | **46.71** (+1.41 vs SnapKV) | **48.24** | **45.43** |

> 💡 在 **256-token 极限压缩**下，CompressKV 在多个模型上均取得最佳成绩，**达到 FullKV 性能的 97% 以上**。

#### ✅ Needle-in-a-Haystack 结果（准确率 %）
- 在 Llama-3.1-8B 上：
  - **仅使用 0.7% KV 存储（256 tokens）时，准确率达到 90%**
  - 使用 5% KV 存储（2048 tokens）时，几乎无损（~98%）
- 在 Mistral-7B 上：
  - 256 tokens 即可实现近似无损压缩（>97%）

> 📈 图 5 显示，在所有缓存预算下，CompressKV 均显著优于其他方法，尤其在低预算区优势明显。

### 🔬 消融实验结果（Table 3）

| 方法 | Mistral-7B @256 (Acc. %) |
|------|--------------------------|
| SnapKV | 43.76 |
| + SRH Selection | 44.96 (+1.20) |
| + SRH + Layer Alloc. | **45.43** (+1.67) |

> ✅ 两个组件互补：SRH 选择提升显著，加上 layer-adaptive allocation 后进一步增益。

#### SRH 数量影响（Top-k 分析）
- 最优数量为 **Top-4 SRH per layer**
- 少于或多于 4 效果下降，说明并非越多越好。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Streaming Heads 主导会导致关键信息丢失**，传统基于 aggregate attention 的 eviction 方法存在系统性偏差。
2. **Semantic Retrieval Heads 能有效捕获 span-level 语义线索**，即使未在单个 token 上获得最高 attention，也能反映其重要性。
3. **SRH 引导的 token selection 显著提升压缩后性能**，特别是在极限压缩场景下（如 <1% KV 存储）仍能维持高准确性。
4. **离线误差估计可用于高效层间预算分配**，无需在线 profiling，兼顾效率与效果。
5. **CompressKV 与多种优化技术正交**，可无缝集成：
   - 与 **prefilling acceleration**（如 MInference）结合，降低 prefill 成本；
   - 与 **KV quantization**（如 KIVI）结合，实现 bit-level + token-level 双重压缩；
   - 与 **head-level allocation**（如 HeadKV）结合，形成 HeadCompressKV / AdaCompressKV，进一步提点。

### ⚠️ 方法的局限性
- **依赖离线 calibration 数据集**来识别 SRH，若目标任务分布差异大，可能影响泛化性。
- 当前 SRH 定义基于答案 span 的 attention 聚合，难以覆盖非问答类任务中的复杂语义结构。
- 对 non-GQA 架构的支持未明确讨论（尽管原理可迁移）。

### 🔮 未来工作方向
- 探索 **online-adaptive SRH 识别**，增强跨任务鲁棒性。
- 将 SRH 思想扩展至 **prefill 阶段优化**或 **训练阶段引导稀疏化**。
- 结合 **function vector** 或 **causal mediation analysis** 更精细解析 head 功能。
- 推广至更多架构（如 MHA、MQA）和其他模态（如多模态 LLM）。

---

## ✅ 总结一句话
> **CompressKV 通过引入 Semantic Retrieval Heads 和误差感知的层自适应分配，在极低 KV-Cache 预算下实现了远超现有方法的长上下文 LLM 推理性能，显著提升了资源-性能权衡，为可持续 AI 部署提供了新路径。**

🔗 开源地址：[https://github.com/TUDa-HWAI/CompressKV](https://github.com/TUDa-HWAI/CompressKV)

</details>

---

### 4. [CKM-Driven Communication-Aware UAV Intelligent Trajectory Optimization for Urban Inspection](https://arxiv.org/abs/2606.24979)

**Authors**: Yang Xiaomeng, Jia Ziye, Zhu Qiuming, Wu Qihui  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.24979v1  

#### Abstract
Unmanned aerial vehicles (UAVs) are increasingly employed in urban inspection tasks, where reliable communication is critical but challenging due to the severe spatial channel heterogeneity. To address the issue, in this paper, we focus on the communication-aware path planning for multi-UAV tasks, a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CKM-Driven Communication-Aware UAV Intelligent Trajectory Optimization for Urban Inspection

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在城市环境中，UAV（Unmanned Aerial Vehicle）执行巡检任务时面临严重的空间信道异质性（spatial channel heterogeneity），如建筑物遮挡、街谷效应等，导致通信质量剧烈波动。传统的路径规划方法通常仅优化几何距离或能耗，忽略通信约束，容易使无人机穿越低质量通信区域，影响任务可靠性。

本文旨在解决**多UAV城市巡检中的通信感知轨迹规划问题**，实现飞行效率与通信可靠性的联合优化。

---

### 🚀 提出的新方法与创新思路

作者提出了一种 **CKM-driven GATSAC 框架**，其核心创新包括：

#### （1）**基于扩散模型的时序累积 Channel Knowledge Map (CKM) 构建**
- 利用城市无线环境具有准静态特性的假设，将不同时段的稀疏 RSS 测量值进行时间累积，形成稠密的 CKM。
- 引入 **diffusion model** 对稀疏观测下的 CKM 进行高保真重建，显著降低对实时感知的依赖，提升建图精度和鲁棒性。

> ✅ 优势：能够在极少量飞行采样下重建全局信道分布，减少探测开销。

#### （2）**GAT + SAC 分层决策架构（GATSAC）**
- **上层：Graph Attention Network (GAT)** 将多目标巡检建模为通信感知的 TSP 问题，融合节点的空间位置与 CKM 中的信道质量，学习最优访问顺序。
- **下层：Soft Actor-Critic (SAC)** 在连续动作空间中生成平滑轨迹，动态规避通信衰减区，并满足动力学约束。

> ✅ 优势：结合组合优化与强化学习，兼顾全局最优性和局部灵活性。

#### （3）**端到端通信感知路径规划框架**
- 首次将 **CKM 建模 → 图神经网络排序 → 强化学习轨迹控制** 融合在一个统一框架中，实现了从稀疏感知到智能决策的闭环。

> ✅ 优势：无需实时反馈即可引导 UAV 主动避开差信道区域，提升了系统的自主性与适应性。

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | 本论文改进 |
|------|--------|-----------|
| Distance-only TSP | 忽略通信质量，易进入弱信号区 | 显式引入 CKM 指导路径选择 |
| RL-only 方法（如 SAC） | 缺乏全局结构引导，收敛慢 | 使用 GAT 提供高质量初始策略 |
| 静态 SINR 约束方法 | 假设信道已知且固定 | 利用 diffusion model 实现稀疏→稠密 CKM 推断 |
| 单纯地图构建类方法 | 注重建图精度，忽视实际任务序列约束 | 联合考虑通信质量与任务完成顺序 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用 **Raymobtime urban dataset** [16]，该数据集提供基于射线追踪（ray-tracing）的真实城市信道测量数据和地理信息。
- 仿真区域划分为 **64×64 的网格**，用于构建 CKM。

---

### ⚙️ 实验设置
- 多UAV协同巡检场景，M 个 UAV 完成 N 个目标点的访问，起降于同一基站（depot）。
- 所有 UAV 固定飞行高度 $ h $，采用二阶运动学模型。
- 利用 **K-means 聚类** 将任务节点分配给各 UAV，转化为多个单机路径规划问题。

---

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Total Flight Distance** | 总飞行路径长度，衡量效率 |
| **Min RSS (dBm)** | 轨迹中最低接收信号强度，反映通信可靠性（越接近 0 越好，因负值） |
| **Communication Quality Lower Bound ($ q_m $)** | 整条轨迹上的最小 RSS 下界 |
| **Trajectory Smoothness & Feasibility** | 是否满足速度、加速度限制，是否平滑避障 |

> 注：由于 RSS 为负值，图中使用 `-MinRSS` 作为纵轴，数值越大表示通信质量越好。

---

### 🆚 基线方法对比
共比较五种基线方法：
1. **Distance-Only TSP + Controller**  
   - 仅基于欧氏距离求解 TSP，使用比例控制器生成轨迹。
2. **Distance-Only TSP + SAC**  
   - TSP 排序无信道感知，由 SAC 微调轨迹以避免差信道。
3. **TSP + A\***  
   - 使用 A* 替代 SAC 进行路径搜索，验证 RL 在连续控制中的优势。
4. **SAC without GAT-TSP Initialization (No TSP)**  
   - 不使用 GAT 初始化顺序，完全随机访问节点。
5. **Joint SAC**  
   - SAC 同时优化离散顺序与连续轨迹，验证分层设计必要性。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Fig. 6）
| 方法 | 总飞行距离 | -MinRSS（通信质量） |
|------|------------|---------------------|
| Distance-Only TSP | ~323.4 | ~80.1 dBm |
| Dist-TSP + SAC | **~211.7**（最短） | ~79.7 dBm |
| TSP + A* | ~275.8 | ~82.4 dBm |
| SAC (No TSP) | ~308.8 | ~75.8 dBm |
| Joint SAC | ~250.1 | ~83.7 dBm |
| **Ours (GAT-CKM + SAC)** | **~235.1** | **~85.2 dBm** ✅ |

> ✅ **本方法在通信质量和路径长度之间取得最佳平衡**

---

### 🔬 消融实验分析（隐含于对比中）
- **GAT-TSP 的作用**：相比“SAC only”和“Joint SAC”，GAT 提供了更优的初始序列，显著提升最终性能，说明**全局结构先验的重要性**。
- **CKM 的价值**：相比仅用距离的方法，“GAT-CKM”能有效避开低RSS区域，通信质量提升超过 5 dBm。
- **SAC 的优势**：相比 A*，“SAC-based planner”能更好处理连续状态空间，在复杂地形中实现更灵活避让。

---

### 🖼️ 可视化结果（Fig. 4 & Fig. 5）
- Fig. 4 展示了两个 UAV 的合作路径规划结果，显示任务被合理划分并生成高效轨迹。
- Fig. 5 显示本方法生成的轨迹明显绕开了低质量通信区域（深色区域），而其他方法（如 Distance-Only TSP）直接穿过。
- CKM 重建误差极小（Fig. 3c vs. 3d），证明 diffusion model 能准确恢复真实信道分布。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **CKM 是实现通信感知路径规划的关键基础设施**：通过时间累积与 diffusion model，可在稀疏观测下实现高保真信道建图。
2. **分层架构优于端到端 RL**：GAT 提供通信感知的初始路径，SAC 进行精细化调整，二者协同优于单一 Joint SAC 或纯 RL 方法。
3. **主动规避差信道是可行且有效的**：无需实时反馈，仅依靠预构建 CKM 即可引导 UAV 主动选择高质量通信路径。
4. **通信质量与飞行效率可以兼得**：本方法在保持较短路径的同时，显著提升了通信下限（Min RSS ↑）。

---

### ⚠️ 方法的局限性
1. **依赖准静态信道假设**：若城市环境变化剧烈（如大量移动车辆），CKM 可能失效。
2. **计算开销较高**：训练 diffusion model 和 GAT/SAC 需要一定算力资源，可能限制实时部署。
3. **未考虑多频段或多基站切换**：当前 CKM 基于单一频率和单个 BS 构建，扩展性有待验证。

---

### 🔮 未来工作方向
1. **动态 CKM 更新机制**：结合在线感知与增量学习，实现 CKM 的实时更新。
2. **跨域知识迁移**：将在一个城市学到的 CKM 模型迁移到相似城市，降低建图成本。
3. **多模态感知融合**：结合视觉、LiDAR 等传感器进一步增强环境理解能力。
4. **硬件原型验证**：在真实无人机平台上部署算法，测试实际通信性能增益。

---

> **总结一句话**：  
> 本文提出的 **CKM-driven GATSAC** 框架首次实现了从稀疏信道感知到智能轨迹决策的闭环优化，在保证飞行效率的同时大幅提升通信可靠性，为未来 6G 智能空联网中的 UAV 巡检提供了新的技术路径。

</details>

---

### 5. [Scalable Peptide Design via Memory-Efficient Equivariant Transformer](https://arxiv.org/abs/2606.25006)

**Authors**: Rui Jiao, Xiangzhe Kong, Yinjun Jia, Yijia Zhang, Ziyi Yang, Yang Liu, Jianzhu Ma  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.25006v1  

#### Abstract
Target-specific peptide design requires sequence and structure co-design under full atom geometric constraints. Latent generative frameworks offer an effective route for this problem by compressing fine grained atomic structures into block level latent representations and performing conditional gene...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scalable Peptide Design via Memory-Efficient Equivariant Transformer

---

## 1. 论文的主要贡献和创新点

### 解决的问题
靶向特异性肽设计需要在**全原子几何约束下进行序列与结构的联合设计**（sequence and structure co-design）。现有的生成模型虽然能通过潜在空间（latent space）压缩原子结构以提升效率，但其底层的几何骨干网络（geometric backbone）在处理大规模分子复合物时面临严重的**内存瓶颈**，尤其是当原子数量增加时，内存消耗呈平方级增长（$O(N^2)$），限制了模型的可扩展性。

### 提出的新方法：MEET
作者提出了 **MEET（Memory-Efficient Equivariant Transformer）**，一种专为全原子肽建模设计的 **E(3)-equivariant Transformer 骨干网络**，旨在实现高效且可扩展的肽生成。

#### 核心创新点：
- **耦合不变量与等变量特征流**：维护旋转不变的标量特征 $H$ 和旋转等变的向量特征 $V$，保证对三维空间变换的对称性（E(3) equivariance）。
- **基于注意力机制的内存优化设计**：
  - **距离感知查询-键增强（distance-aware query-key augmentation）**：将成对距离信息编码进注意力机制中的 Q 和 K 向量，避免显式存储 $N \times N$ 距离矩阵。
  - **全局坐标聚合初始化向量特征**：取代传统依赖局部图邻域的方法，使用全局注意力初始化 $V^{(0)}$，兼容 FlashAttention 等高效内核。
  - **稀疏键适配器（sparse bond adapter）**：仅对共价键边注入化学连接信息，复杂度为 $O(|E|)=O(N)$，显著降低额外开销。

### 相比现有方法的优势
| 方面 | MEET | 传统方法（如 EPT） |
|------|------|------------------|
| 内存复杂度 | **线性增长 $O(Nd)$** | 平方增长 $O(N^2)$ |
| 可扩展性 | 支持上千原子系统 | 受限于显存 |
| 几何表达能力 | 保持 E(3) 等变性 | 多数也支持，但效率低 |
| 集成性 | 可无缝嵌入 VAE + Latent Diffusion Pipeline | 通常难以扩展 |

> ✅ **核心优势总结**：MEET 在不牺牲几何表达能力的前提下，实现了**线性内存扩展**，使大规模全原子肽生成成为可能。

---

## 2. 核心实验方法和设置

### 数据集构建
- **来源**：从 **AFDB（AlphaFold DB）** 中的 864 万个蛋白质结构域出发。
- **流程**：
  1. 滑动窗口提取 3–13 个残基的线性肽段候选；
  2. 过滤条件包括：pLDDT > 70、GRAVY < 0.5、二级结构比例限制、末端距离合理；
  3. 接口筛选：要求埋藏表面积（BSA）> 400 Å²，相对 BSA 在 0.35–0.85；
  4. 序列去重聚类（overlap threshold=0.2）。
- **最终数据集**：
  - **100K 数据集**：用于基准测试；
  - **1.2M 数据集**：用于模型与数据协同扩展研究。

### 实验框架
采用两阶段**潜在生成框架**（inspired by UniMoMo）：
1. **VAE 编码器**：将全原子肽-口袋复合物压缩为块级（block-level）潜在变量 $Z = (Z_H, Z_X)$；
2. **Latent Diffusion Model (LDM)**：在潜在空间中基于目标口袋上下文 $C$ 生成肽潜在表示；
3. **VAE 解码器**：将生成的潜在变量解码回全原子结构和序列。

> ⚙️ **MEET 的角色**：作为整个流程中所有模块（Encoder、Sequence Decoder、Structure Decoder、LDM Denoiser）的统一几何骨干。

### 评估指标
| 指标 | 含义 | 越高越好？ |
|------|------|-----------|
| **$\Delta G$**（Binding Free Energy） | 预测结合自由能 | ❌（越低越好） |
| **PoseBuster (PB) Pass Rate** | 生成构象的物理有效性（无冲突、键角合理等） | ✅ |
| **Shape Complementarity** | 肽-靶标界面形状匹配度 | ✅ |
| **$\Delta G / \Delta SASA$** | 单位埋藏面积的能量收益（反映结合效率） | ❌ |
| **Sequence Diversity (Seq. Div.)** | 同一靶标下生成序列的多样性 | ✅ |

### 基线方法对比
- **PepGLAD** [7]
- **PepFlow** [9]
- **UniMoMo** [8]
- **DiffPepBuilder** [23]

---

## 3. 主要实验结果和性能指标

### （1）内存效率验证（Figure 2）
- 在合成 poly-alanine 链上测试不同长度下的峰值 GPU 内存占用：
  - **MEET**：内存随原子数呈**近似线性增长**；
  - **EPT（原生 equivariant transformer）**：呈现明显超线性增长（接近 $O(N^2)$）；
  - 当 $n_{aa}=1024$ 时，MEET 仅需约 **300MB**，而 EPT 超过 **1500MB**，节省 **5倍内存**。

> 🔍 模块级分析显示：FFN、Self-Attention、Bond Adapter、Initialization 四个组件均保持线性增长趋势，无隐藏二次项。

---

### （2）100K 数据集上的基准性能（Table 2）

| Method       | $\Delta G_{\text{mean}}$ ↓ | PB ↑     | Shape    | $\Delta G/\Delta SASA$ | Seq. Div. |
|--------------|----------------------------|----------|----------|-------------------------|------------|
| PepGLAD      | -18.02                     | 0.000    | 0.566    | -0.67                   | 0.939      |
| PepFlow      | -20.81                     | 0.000    | 0.566    | -2.36                   | 0.798      |
| UniMoMo      | -21.80                     | 0.561    | 0.633    | -2.01                   | 0.922      |
| DiffPepBuilder| 1.21                      | 0.001    | 0.607    | 0.27                    | 0.843      |
| **MEET-XS**  | **-25.67**                 | **0.660**| 0.635    | -2.39                   | 0.838      |
| **MEET-B**   | **-27.40**                 | **0.799**| **0.651**| **-2.53**               | 0.715      |

✅ **结论**：
- MEET 在 **binding affinity ($\Delta G$)** 和 **physical validity (PB)** 上全面超越所有基线；
- 尤其是 PB 达到 **0.799**，远高于第二名 UniMoMo 的 0.561；
- 但 **sequence diversity 下降**（MEET-B: 0.715），提示可能存在过拟合或分布集中现象。

---

### （3）1.2M 数据集上的扩展实验（Table 3 & Figure 3）

#### 模型规模扩展（DiT-style scaling）
| Model   | $\Delta G_{\text{mean}}$ ↓ | PB ↑     | Shape    | $\Delta G/\Delta SASA$ | Seq. Div. |
|---------|----------------------------|----------|----------|-------------------------|------------|
| MEET-XS | -26.26                     | 0.703    | 0.640    | -2.39                   | 0.927      |
| MEET-S  | -27.63                     | 0.727    | 0.651    | -2.52                   | 0.918      |
| MEET-B  | -27.61                     | 0.729    | 0.657    | -2.55                   | 0.915      |
| **MEET-L** | **-28.22**              | **0.732**| **0.659**| **-2.61**               | **0.899**  |

📈 **关键发现**：
- 随着模型容量增大（XS → L），各项指标持续提升；
- **训练损失单调下降**（Figure 3），表明 MEET 能有效利用更大模型容量；
- **序列多样性恢复至 0.899**，说明此前多样性下降是由于**数据量不足**而非架构缺陷；
- MEET-XS 在 1.2M 数据上已优于 100K 上的 MEET-B，证明**数据与模型协同扩展的有效性**。

---

## 4. 关键结论和发现

### 主要结论
1. **内存效率是可扩展生成的关键瓶颈**：传统的几何骨干因 $O(N^2)$ 内存消耗限制了实际应用，而 MEET 通过算法重构实现了 **$O(N)$ 峰值激活内存**，解决了这一瓶颈。
2. **MEET 显著提升生成质量**：在 binding affinity、physical validity、shape complementarity 等关键指标上全面领先现有方法。
3. **支持系统性缩放（systematic scaling）**：MEET 能够在更大数据集（1.2M）和更大模型（up to MEET-L）下持续提升性能，验证了其作为“基础骨干”的潜力。
4. **多样性问题可通过数据缓解**：早期出现的 sequence diversity 下降并非模型本质缺陷，而是受限于训练数据规模。

### 方法的局限性
- **依赖高质量结构输入**：当前方法基于 AFDB 结构，若初始结构误差大，可能影响生成效果；
- **未直接生成共价修饰或多环肽**：目前聚焦于线性肽，对复杂拓扑的支持有限；
- **推理速度仍待优化**：尽管内存更优，但多层 MEET + diffusion 步骤导致采样较慢；
- **泛化到全新 fold 或远端突变场景尚需验证**。

### 未来工作方向
- 扩展至 **protein-peptide binder design** 全流程联合优化；
- 引入 **multi-modal conditioning**（如功能标签、表达稳定性）；
- 探索 **MEET + Flow Matching** 替代 Diffusion 以加速生成；
- 构建 **larger and more diverse datasets** 包含 post-translational modifications；
- 推动 **MEET 在药物发现 pipeline 中的实际部署与湿实验验证**。

---

> 📌 **一句话总结**：  
> MEET 通过内存友好的 E(3)-equivariant Transformer 设计，在保持几何精确性的同时实现了全原子肽生成的线性可扩展性，并在大规模数据上展现出卓越的生成质量与系统扩展潜力，为下一代 AI-driven 肽药设计提供了强有力的骨干架构。

</details>

---

### 6. [EPTS: Elastic Post-Training Sparsity for Efficient Large Language Model Compression](https://arxiv.org/abs/2606.25285)

**Authors**: Ke Xu, Jiaqi Wan, Wenhao Hu, Han Pu, Xiaoyun Wang  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.25285v1  

#### Abstract
Post-Training Sparsity (PTS) has emerged as a crucial paradigm for compressing Large Language Models to facilitate efficient deployment on resource-constrained devices. However, existing PTS methodologies are typically confined to Single-Sparsity optimization, necessitating a separate, time-consumin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EPTS: Elastic Post-Training Sparsity for Efficient Large Language Model Compression

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现有的 **Post-Training Sparsity (PTS)** 方法大多采用**单稀疏度优化范式**（Single-Sparsity Optimization），即每个目标稀疏度（如 30%、50%、70%）都需要独立进行一次完整的重建过程。这种模式存在以下问题：

- **效率低下**：为多个稀疏度部署模型时需重复多次耗时的优化。
- **灵活性差**：无法快速适应不同硬件场景下的动态稀疏度需求。
- **参数竞争**：多稀疏度同时优化时，不同稀疏组之间会争夺参数重构能力。

这严重限制了在资源受限设备上的灵活部署。

---

### **提出了什么新方法或新思路**

本文提出 **Elastic Post-Training Sparsity (EPTS)**，一种统一的**多稀疏度压缩框架**，通过**一次性的 block-wise 重建过程**生成一个“弹性”模型，能够支持多种稀疏配置而无需重新训练。

#### 核心创新机制：

1. **Multi-Sparsity Hierarchy LoRA (MS-HiLoRA)**  
   - 构建层次化的 LoRA 模块，形成从低到高稀疏度的**参数继承链**。
   - 高稀疏度组可复用低稀疏度组已学习的知识，缓解参数竞争。
   - 补偿权重定义为累积形式：  
     $$
     \Phi_k = \sum_{i=0}^{k} B_i A_i
     $$

2. **Multi-Sparsity Feature Mixer (MSFM)**  
   - 在 Transformer block 间引入特征融合模块，将不同稀疏度下的输出特征加权混合。
   - 提升模型对剪枝扰动的鲁棒性，稳定输入分布。
   - 融合公式：
     $$
     X_{l+1} = \sum_{k=0}^{K-1} \lambda_k \cdot ((W + \Phi_k) \odot M_s) X_l
     $$

---

### **相比现有方法的优势**

| 维度 | EPTS | 传统 PTS 方法（如 SparseGPT, Wanda, ICP） |
|------|------|----------------------------------------|
| **优化次数** | 单次 one-shot 优化 | 每个稀疏度单独优化 |
| **部署灵活性** | 支持任意稀疏度切换 | 固定稀疏度模型 |
| **知识迁移** | 显式利用低稀疏度知识辅助高稀疏度恢复 | 各稀疏度独立处理 |
| **鲁棒性** | 特征混合增强抗扰动能力 | 无跨稀疏度协同机制 |

> ✅ **核心优势**：**一次训练，多场景部署**，显著提升实际应用中的效率与灵活性。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **校准数据（Calibration Data）**：  
  - C4 数据集，随机选取 128 个样本，每样本 2048 tokens。
  - 用于重建过程中补偿权重的学习。

- **评估数据集**：
  - **Wikitext2**：用于计算 **Perplexity (PPL)**，衡量语言建模能力。
  - **Zero-shot NLP 任务**（共7项）：
    - BoolQ, RTE, HellaSwag, WinoGrande, ARC-Challenge, ARC-Easy, OpenBookQA
    - 评估模型在下游任务中的泛化能力。

---

### **实验设置和评估指标**

| 设置项 | 描述 |
|-------|------|
| **模型家族** | LLaMA 系列（7B, 2-7B, 3-8B）、OPT 系列（125M, 350M, 1.3B） |
| **稀疏度范围** | 30%, 40%, 50%, 60%, 70% |
| **稀疏分组** | 分为 Low / Mid / High 三个 sparsity groups |
| **LoRA 配置** | 秩分配默认为 [24, 16, 16]，可调 |
| **评估指标** | 
| | - Perplexity (PPL) ↓ |
| | - Zero-shot 平均准确率 ↑ |
| | - 推理吞吐量（Prefill / Decode）↑ |
| | - 内存占用 ↓ |

---

### **基线方法对比**

| 方法 | 类型 | 是否更新权重 | 多稀疏度支持 |
|------|------|----------------|---------------|
| **SparseGPT** | Training-free | 是（二阶补偿） | ❌ |
| **Wanda** | Optimization-free | 否（仅基于激活评分剪枝） | ❌ |
| **RIA** | Optimization-free | 否 | ❌ |
| **ICP** | Block-level 补偿 | 是 | ❌ |
| **EPTS (Ours)** | Unified Multi-Sparsity | 是（LoRA 微调） | ✅ |

> 所有方法均使用相同校准数据和 batch size=1 进行公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Table 1）**

#### 在 **LLaMA-7B @ 70% sparsity** 上的表现：

| 方法 | PPL (↓) | Zero-shot Avg Acc (↑) |
|------|--------|------------------------|
| Dense | 5.68 | 64.24 |
| SparseGPT | 25.78 | 45.05 |
| Wanda | 82.19 | 39.74 |
| RIA | 91.23 | 40.00 |
| ICP | 65.20 | — |
| **EPTS (Ours)** | **16.94** | **47.36** |

✅ **结论**：EPTS 在 70% 稀疏度下 PPL 显著优于所有基线，且零样本平均准确率高出 **2.31%~7.62%**。

---

#### 在 **OPT-1.3B @ 70% sparsity** 上的表现：

| 方法 | PPL (↓) |
|------|--------|
| Dense | 14.62 |
| SparseGPT | 42.70 |
| Wanda | 331.00 |
| RIA | 353.96 |
| ICP | 99.01 |
| **EPTS (Ours)** | **30.65** |

✅ **结论**：EPTS 实现了最稳健的性能保持，远超其他方法。

---

### **与基线方法的对比结果**

- 在 **中低稀疏度（≤50%）** 下，EPTS 性能与 SparseGPT 相当。
- 在 **高稀疏度（≥60%）** 下，EPTS 明显领先，尤其在 OPT 和 LLaMA 家族上均表现出更强的鲁棒性。
- 相比不更新权重的方法（Wanda, RIA），EPTS 利用 LoRA 动态补偿，有效缓解信息丢失。

---

### **消融实验结果**

#### （1）MS-HiLoRA 消融（Table 2）

| 配置 | LLaMA-7B @70% PPL | OPT-1.3B @70% PPL |
|------|--------------------|-------------------|
| Independent LoRA | 17.60 | 82.29 |
| Shared LoRA | 22.01 | 91.22 |
| **MS-HiLoRA (Ours)** | **16.94** | **30.65** |

✅ 层次化设计显著优于独立或共享模式，验证了知识继承的有效性。

#### （2）LoRA 秩分配影响（Table 3）

| 秩配置 [low, mid, high] | OPT-1.3B @70% PPL |
|--------------------------|------------------|
| [24,16,16] | 30.65 |
| [16,16,24] | **30.27** ✅ |

✅ 可通过调整秩分配“倾斜”性能偏好，例如优先保障高稀疏度表现。

#### （3）MSFM 消融（Table 4）

| 配置 | LLaMA-7B @70% PPL | OPT-1.3B @70% PPL |
|------|--------------------|-------------------|
| Dense Pass-through | 25.48 | 89.94 |
| Stochastic Substitution | 19.21 | 42.22 |
| **MSFM (Ours)** | **16.94** | **30.65** |

✅ MSFM 提供确定性融合路径，避免随机波动，显著提升鲁棒性。

#### （4）融合权重 λ 的控制性（Table 5）

| 权重偏向 | 最优 sparsity level |
|---------|---------------------|
| λ_low=0.4 | 在 50% 达最优（PPL=17.37） |
| λ_mid=0.4 | 在 60% 达最优（PPL=20.56） |
| λ_high=0.4 | 在 70% 达最优（PPL=30.65） |

✅ MSFM 是**可控机制**，可通过调节 λ 实现性能定向优化。

---

## 4. 关键结论和发现

### **主要发现**

1. **Nested Information Loss Hypothesis 成立**：  
   高稀疏度的信息损失包含低稀疏度的部分，因此应建立**层级依赖关系**而非孤立优化。

2. **MS-HiLoRA 有效缓解参数竞争**：  
   通过参数继承链，使基础 LoRA 捕获通用特征，高层 LoRA 专注极端剪枝修复。

3. **MSFM 提升跨稀疏度鲁棒性**：  
   特征混合机制增强了模型对剪枝扰动的容忍度，减少输入分布偏移。

4. **EPTS 实现高效多场景部署**：  
   一次训练即可支持多个稀疏度，极大降低部署成本。

---

### **方法的局限性**

- 在**极高稀疏度（≥80%）** 下性能仍显著下降，尚未完全解决。
- 当前实验集中在 ≤8B 规模模型，更大模型（如 70B）的扩展性有待验证。
- 虽然总时间较长，但其“一次训练多用”的特性使其在总体 TCO（Total Cost of Ownership）上更具优势。

---

### **未来工作方向**

1. **结合半结构化/结构化剪枝**：构建更全面的多粒度统一剪枝框架。
2. **探索自动化 sparsity allocation**：基于硬件延迟约束自动搜索最优层间稀疏分布。
3. **扩展至 MoE 架构**：应用于 Mixtral、GLM 等稀疏专家模型。
4. **进一步优化训练效率**：减少 EPTS 自身的优化开销，提升实用性。

---

> 🔗 **开源信息**：代码已公开于 GitHub → [https://github.com/xuke225/EPTS](https://github.com/xuke225/EPTS)  
> 📦 Zenodo DOI: [10.5281/zenodo.20371910](https://doi.org/10.5281/zenodo.20371910)

</details>

---

### 7. [Leaking Circuit Secrets: Gradient Leakage Attacks on Graph Neural Networks](https://arxiv.org/abs/2606.25589)

**Authors**: Rupesh Raj Karn, Johann Knechtel, Ozgur Sinanoglu  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.25589v1  

#### Abstract
As graph neural networks (GNNs) become standard tools for critical tasks in circuit design and analysis, their security and privacy risks require careful attention. Here, we present the first comprehensive evaluation of gradient leakage attacks (GLAs) on GNNs in circuit-design and hardware-security ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Leaking Circuit Secrets: Gradient Leakage Attacks on Graph Neural Networks*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
本文首次系统性地研究了**图神经网络**（GNN）在**电路设计与硬件安全任务**中的**梯度泄露攻击**（Gradient Leakage Attacks, GLAs）。尽管GNN在电路分析、逻辑锁定和硬件木马**（Hardware Trojan, HT）**检测等关键任务中广泛应用，但其训练过程中暴露的梯度可能被攻击者利用来重构敏感电路信息（如门类型、拓扑结构、Trojan特征），从而威胁硬件安全。

该问题此前在电路领域被严重忽视，而本文填补了这一空白。

---

### ✅ 提出了什么新方法或新思路  

- **首次将GLA应用于电路级GNN模型**：构建端到端攻击框架，从GNN训练梯度中重构原始节点特征（如fan-in、centrality等），揭示敏感电路属性。
- **提出针对电路场景的威胁模型**：涵盖白盒/半诚实服务器、联邦学习、侧信道等多种现实攻击路径。
- **全面评估主流防御机制**：对 differential privacy、gradient clipping、secure aggregation、model compression/quantization 和 adversarial training 在电路GNN上的有效性进行实证分析。
- **开源完整方法论与代码**：发布可复现的实验流程与数据处理工具（GitHub: [https://github.com/rkarn/GradientAttackGNNs](https://github.com/rkarn/GradientAttackGNNs)）。

---

### ✅ 相比现有方法的优势  

| 方面 | 优势说明 |
|------|----------|
| **应用场景新颖性** | 现有GLA研究集中于图像/CV/NLP领域（如MNIST/CIFAR），本文是首个聚焦**电路网表**（netlist）和硬件安全任务的研究。 |
| **任务相关性强** | 攻击目标直接关联硬件安全实践，例如：<br>• 推断门类型 → 助力逻辑锁定破解<br>• 识别Trojan结构 → 规避检测机制 |
| **架构对比深入** | 不仅验证攻击可行性，还系统比较了四种主流GNN架构（GCN, GraphSAGE, GIN, GAT）的抗泄露能力差异。 |
| **防御实用性分析** | 揭示常见防御手段在电路任务中效果有限甚至适得其反，强调“不能照搬CV领域的防御策略”。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集  

- **ISCAS’85**：经典组合与时序电路基准（如 c432, c880）
- **EPFL**：现代综合后电路基准，更具实际代表性
- **TrustHub**：提供标准硬件木马模板（Countermux, FSMor, Andxor），用于注入HT并训练检测模型

所有电路均转换为图结构：
- 节点 = 门电路（gate）
- 边 = 连线（wire）
- 特征 = 结构属性（fan_in, fan_out, centrality, clustering coefficient 等共13维）

---

### ⚙️ 实验设置  

#### ✅ GNN模型  
评估四类SOTA GNN：
- **GCN**（Graph Convolutional Network）
- **GraphSAGE**
- **GIN**（Graph Isomorphism Network）
- **GAT**（Graph Attention Network）

#### ✅ 敏感任务  
两个典型硬件安全任务作为代理任务：
1. **Gate Classification**：多分类任务（8类门），模拟逻辑锁定分析
2. **Hardware Trojan Detection**：二分类任务（良性 vs 恶意门），模拟HT检测

#### ✅ 攻击流程（GLA）  
1. 攻击者获取某节点的输出标签及对应梯度 $ \nabla_W \mathcal{L} $
2. 构造一个随机初始化的“dummy”特征向量 $ \tilde{x}_v $
3. 通过优化以下目标函数重构原始输入特征：
   $$
   \min_{\tilde{x}_v} \sum_{l=0}^{L-1} \| \nabla_{W^{(l)}} \mathcal{L}(\tilde{x}_v, y) - G^{(l)} \|^2
   $$

#### ✅ 防御机制测试  
五种主流防御技术：
- Differential Privacy （DP）
- Gradient Clipping + Perturbation
- Secure Aggregation
- Model Compression & Quantization
- Adversarial Training

---

### 📊 评估指标  

使用三个互补指标衡量重构质量（越低表示泄露越少）：

| 指标 | 公式 | 含义 |
|------|------|------|
| **abs_l2** | $\|x - \tilde{x}\|_2$ | 绝对欧氏距离，反映数值偏差 |
| **rel_l2** | $\frac{\|x - \tilde{x}\|_2}{\|x\|_2 + \epsilon}$ | 归一化误差，考虑特征尺度 |
| **cos_sim** | $\frac{x \cdot \tilde{x}}{\|x\|\|\tilde{x}\|}$ | 余弦相似度，越高表示方向一致（理想值=1） |

> 成功攻击 ≈ **abs_l2 / rel_l2 小，cos_sim 接近1**

---

### 🔁 基线方法对比  

| 类别 | 引用文献 | 主要内容 |
|------|---------|--------|
| 图像域GLA | [9] Zhu et al. (NeurIPS 2019) | 在CNN上实现图像像素级重构 |
| 图结构GLA | [25] Sinha et al. (arXiv 2024) | 对Cora/PubMed等通用图数据发起GLA |
| 本文贡献 | —— | **首次在电路级GNN上开展系统性GLA研究，结合硬件安全语义分析泄露后果** |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（无防御情况下）

#### 表：平均GLA重构效果（No Defense）

| GNN | 任务 | cos_sim ↑ | rel_l2 ↓ |
|-----|------|-----------|---------|
| **GAT** | Gate Classification | **0.715** | 0.795 |
| **GIN** | Gate Classification | **0.193** | 1.405 |
| **GraphSAGE** | HT Detection | **0.695** | 0.670 |
| **GIN** | HT Detection | **0.050** | 1.375 |

> ✅ **GIN 最具鲁棒性**；❌ **GAT 和 GraphSAGE 泄露最严重**

---

### 🔍 与基线方法的对比结果  

| 发现 | 说明 |
|------|------|
| ✅ **电路GNN存在显著梯度泄露风险** | 所有模型均可被成功重构部分特征，尤其在门类型和Trojan标记上泄露明显 |
| ✅ **不同GNN架构泄露程度差异大** | GAT因attention机制产生更可区分的梯度，导致更高泄露；GIN因injective aggregation 更抗泄露 |
| ✅ **特定门/Trojan更容易被重构** |  
  - **OR/NAND/XOR门**：cos_sim > 0.6，易被识别  
  - **NOT门**：最难重构（cos_sim ≈ 0.2–0.3），因其结构简单且高频出现  
  - **Trojan类节点**：比clean节点泄露更多（cos_sim: 0.338 vs 0.143） |

---

### 🔧 消融实验结果（防御机制效果）

#### 表：防御机制对GLA的影响（以cos_sim下降为优）

| 防御机制 | 对哪个模型最有效？ | 效果 | 副作用 |
|--------|------------------|------|-------|
| **Secure Aggregation** | GraphSAGE (HT Detection) | cos_sim ↓74.1% | 导致GIN/GraphSAGE准确率暴跌至~25% |
| **Adversarial Training** | GCN (HT Detection) | cos_sim ↓83.3% | 几乎不影响精度（仍达99.95%） |
| **Gradient Clipping** | GIN (Gate Classification) | cos_sim ↓16.2% | 在其他模型上反而加剧泄露（如GIN-HT: cos_sim↑140%！） |
| **Differential Privacy** | —— | 效果一般，有时恶化 | 显著降低所有模型性能（如GraphSAGE↓30%） |
| **Quantization** | —— | 轻微提升鲁棒性 | 性能损失可控，但防护有限 |

> ❗ **重要发现**：**没有一种防御在所有任务和模型上都有效**，某些防御甚至会**增加泄露风险**！

---

## 4. 关键结论和发现

### ✅ 主要发现  

1. **GNN在电路任务中普遍存在梯度泄露风险**  
   - 攻击者可通过梯度重构出敏感电路特征，辅助下游攻击（如绕过逻辑锁定、规避Trojan检测）。

2. **GNN架构直接影响泄露程度**  
   - **GAT**：注意力机制增强梯度可区分性 → **高泄露**
   - **GIN**：injective aggregation + 非线性变换 → **强鲁棒性**
   - **GraphSAGE**：局部聚合保留结构信息 → 中等泄露，但在HT检测中表现最差

3. **防御机制效果高度依赖模型与任务**  
   - 没有“万能药”，某些防御（如gradient clipping）可能**适得其反**
   - **GCN + Adversarial Training** 是少数兼顾高性能与高隐私的组合

4. **电路特征本身具有一定天然抗性**  
   - 相较于图像像素，电路特征（如fan-in）具有离散性和结构性，小误差即可造成功能错乱 → 一定程度抑制精确重构
   - 但这不意味着安全，方向性信息（cos_sim）仍可被有效提取

---

### ⚠️ 方法的局限性  

| 局限 | 说明 |
|------|------|
| **静态网表假设** | 实验基于静态gate-level netlist，未涉及时序行为或物理层信息 |
| **单节点攻击为主** | 当前攻击聚焦于单个节点重构，尚未扩展到全图逆向工程 |
| **理想化威胁模型** | 假设攻击者拥有完整模型参数和梯度，在实际部署中获取难度较高（但仍可通过侧信道等方式逼近） |
| **防御调参空间未充分探索** | 所有防御采用默认超参，未系统搜索最优配置（作者指出这是未来方向） |

---

### 🔮 未来工作方向  

1. **端到端攻击链构建**  
   - 将GLA与具体硬件攻击（如SAT攻击破解逻辑锁定）结合，验证实际危害。

2. **开发面向电路的隐私保护GNN架构**  
   - 设计内生具备抗梯度泄露能力的新GNN结构，而非依赖外部防御。

3. **跨模态防御优化**  
   - 系统性探索防御机制的超参数空间，寻找精度-隐私的最佳平衡点。

4. **动态/时序电路中的GLA研究**  
   - 扩展至RTL或时序电路场景，评估更复杂模型下的泄露风险。

5. **联邦学习环境下的电路GNN安全分析**  
   - 多方协作设计场景下，研究secure aggregation的实际安全性（已有研究表明其可能泄露标签 [19]）。

---

> 💡 **总结一句话**：  
> 本文敲响警钟——**即使最先进的GNN用于硬件安全任务，也可能成为新的信息泄露通道**。必须在模型设计阶段就考虑隐私保护，不能仅靠事后加装“补丁式”防御。

</details>

---

### 8. [SARA: Unlocking Multilingual Knowledge in Mixture-of-Experts via Semantically Anchored Routing Alignment](https://arxiv.org/abs/2606.25821)

**Authors**: Tianyu Dong, Yangyang Liu, Jiang Zhou, Xinwei Wu, Xiaohu Zhao, Hao Wang, Heng Liu, Linlong Xu, Longyue Wang, Weihua Luo, Shaolin Zhu, Deyi Xiong  
**Category**: cs.CL  
**Published**: 2026-06-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.25821v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) architectures have emerged as an increasingly influential paradigm as they offer a strategic balance between parameter scalability and computational efficiency. However, low-resource languages, which suffer from a scarcity of high-quality training data, often have the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SARA: Unlocking Multilingual Knowledge in Mixture-of-Experts via Semantically Anchored Routing Alignment

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在 **Sparse Mixture-of-Experts (MoE)** 架构中，虽然模型具备强大的参数扩展能力，但在**低资源语言（low-resource languages）** 上表现不佳。其根本原因在于 **cross-lingual routing divergence**：即语义相同的输入在不同语言中被路由到不同的专家（experts），导致高资源语言（如英语）中训练出的专业知识无法有效迁移到低资源语言。

这种路由不一致使得即使模型“知道”答案，也无法激活正确的专家进行推理，从而限制了多语言能力。

---

### **提出了什么新方法或新思路**

作者提出 **SARA (Semantically Anchored Routing Alignment)** 框架，通过**语义锚定的路由对齐**来解决上述问题。其核心思想是：

> 将高资源语言（如英语或中文）的 **routing distribution** 视为“语义锚点（semantic anchors）”，并强制低资源语言的路由分布向这些锚点对齐。

#### 创新点包括：

- **内部路由对齐（Internal Routing Alignment）**  
  不同于传统的输出层蒸馏（output-level distillation），SARA 直接在 MoE 层的**中间层路由概率空间**进行对齐，确保跨语言的专家选择机制具有一致性。

- **基于 JS Divergence 的对称约束**  
  使用 **symmetric Jensen-Shannon (JS) divergence** 来最小化多语言输入与高资源锚点之间的路由分布差异，相比 KL 散度更稳定且对称，适合双向语义对齐任务。

- **三阶段训练流程**：
  1. **语义对齐数据构建**：利用 GPT-5 mini 翻译高资源语言中的正确推理样本，生成严格语义对齐的平行指令数据。
  2. **提取目标路由先验**：在高资源语言上进行前向传播，提取其密集的 routing distribution 作为监督信号。
  3. **序列级路由对齐微调**：在多语言数据上微调模型，同时优化任务损失（L_CE）、负载均衡损失（L_LB）和路由对齐损失（L_JS）。

---

### **相比现有方法的优势**

| 方法 | 缺陷 | SARA 的优势 |
|------|------|-------------|
| Standard Instruction Tuning (FFT) | 仅依赖任务监督，无法纠正内部路由逻辑 | 显式对齐路由机制，提升跨语言泛化 |
| AES / ShifCon | 聚焦表示层对齐或专家正则化，非直接路由控制 | 直接干预 routing decision，机制更精准 |
| 外部教师蒸馏（如 GPT-5） | 缺乏与当前模型内部 expert pathway 的一致性 | 使用**自锚定（self-anchoring）**，保持机制兼容 |

> ✅ SARA 是首个将 **routing distribution** 本身作为可学习目标用于多语言对齐的工作。

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### **训练数据构造方式**：
- 基于 **MMLU-ProX** 和 **GSM8K** 的英文/中文部分构建高质量指令微调数据。
- 经过以下处理：
  - **Correctness-based Filtering**：仅保留模型能正确回答的样本（通过 `\boxed{}` 提取答案验证）。
  - **Parallel Corpus Synthesis**：使用 GPT-5 mini 将完整交互（prompt + reasoning + answer）翻译成 5 种低资源语言。

#### **目标低资源语言（low-resource languages）**：
- `hi` (Hindi), `ne` (Nepali), `bn` (Bengali), `te` (Telugu), `sw` (Swahili)

#### **评估基准（测试集）**：
| Benchmark | 描述 |
|---------|------|
| **Global-MMLU** | 多语言版 MMLU，涵盖多个学科的多项选择题 |
| **BELEBELE** | 多语言阅读理解数据集，含 122 种语言变体 |
| **MGSM** | 多语言版 GSM8K，数学推理任务 |

> 所有训练样本均从测试集中去重，保证零样本（zero-shot）评估有效性。

---

### **实验设置和评估指标**

- **模型架构**：
  - 主要实验：**Qwen3-30B-A3B**（MoE 版本）
  - 验证泛化性：**Phi-3.5-MoE-instruct**

- **训练配置**：
  - 框架：PyTorch
  - Batch Size：256（16×H100 GPU）
  - 学习率：2e-5，cosine decay
  - 训练轮数：2 epochs
  - 对齐层数：Intermediate layers（如 Qwen3 中的 layer 7–34）

- **损失函数组合**：
  $$
  \mathcal{L}_{\text{total}} = \mathcal{L}_{CE} + \lambda_{LB}\mathcal{L}_{LB} + \lambda_{JS}\mathcal{L}_{JS}
  $$
  - $\lambda_{JS} = 1.5$（经敏感性分析确定最优值）

- **评估指标**：
  - 准确率（Accuracy）
  - 平均得分（Across languages）
  - 统计显著性检验：one-tailed paired t-test

---

### **基线方法对比**

| Baseline | 简介 |
|--------|------|
| **Vanilla LM** | 原始 MoE 模型，无额外对齐 |
| **FFT** | Full Fine-Tuning，标准指令微调（强基线） |
| **AES** | 引入正交性和方差损失以增强专家区分度 |
| **ShifCon** | 对齐隐藏状态子空间，提升非主导语言表征 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据（Qwen3-30B-A3B）**

| Benchmark | 方法 | 平均准确率 | 英文锚 vs 中文锚 |
|----------|------|------------|----------------|
| **Global-MMLU** | FFT | 70.73% | — |
| | **SARA (Ours)** | **71.53%** (+0.8pp) | 英文锚优于中文锚 |
| **BELEBELE** | FFT | 82.95% | — |
| | **SARA (Ours)** | **83.09%** (+0.14pp) | — |
| **MGSM** | FFT | 87.20% | — |
| | **SARA (Ours)** | **87.40%** (+0.2pp) | — |

> 💡 在 **Global-MMLU** 上，SARA 相比 FFT 提升 **+0.8%**；在 **Phi-3.5-MoE-instruct** 上甚至达到 **+1.2%** 提升，表明其在较弱多语言模型上效果更显著。

---

### **与基线方法的对比结果**

| 方法 | Global-MMLU (Avg) | BELEBELE (Avg) | MGSM (Avg) |
|------|-------------------|----------------|------------|
| FFT | 70.73 | 82.95 | 87.20 |
| AES | 68.63 | 80.31 | 83.50 |
| ShifCon | 71.12 | 82.86 | 86.90 |
| **SARA (Ours)** | **71.53** | **83.09** | **87.40** |

✅ SARA 在所有基准上均取得最佳平均性能，尤其在低资源语言（如 `te`, `sw`）上提升明显。

---

### **消融实验结果（Ablation Study）**

| 变体 | Global-MMLU (Avg) | 结论 |
|------|--------------------|------|
| `-g-en-s`（外部教师生成路由） | 71.29 | 差于自锚定，说明外部文本缺乏机制一致性 |
| `-q-en-a`（全层对齐） | 67.40 | 性能下降，浅层/深层不适合强制对齐 |
| `-q-en-r`（随机层对齐） | 67.30 | 同样劣于中间层聚焦策略 |
| `-q-sw-s`（Swahili 作锚） | 67.91 | 锚语言质量至关重要，低资源语言不适合作锚 |
| **-q-en-s (Ours)** | **71.53** | 最优配置：自锚定 + 英文锚 + 中间层对齐 |

> 🔍 发现：**中间层（intermediate layers）** 是语言无关的语义推理核心区域（U-shaped divergence pattern），最适合进行路由对齐。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Cross-lingual routing divergence 是制约 MoE 多语言能力的关键瓶颈**。
2. ✅ **显式对齐 routing distribution 比传统方法更能有效迁移高资源语言的能力**。
3. ✅ **使用模型自身的推理轨迹作为锚点（self-anchoring）优于外部教师模型**。
4. ✅ **中间层是理想的对齐目标区域**，因其具有较低的语言特异性，承担通用语义推理功能。
5. ✅ **SARA 在多种 MoE 架构（Qwen3、Phi-3.5）上均有效，具备良好泛化性**。

---

### **方法的局限性**

1. **依赖翻译质量**：合成平行数据的质量受限于翻译模型（如 GPT-5 mini）。若翻译引入语义偏差，会影响路由监督信号。
2. **可能抑制文化特异性表达**：过度对齐可能导致模型忽略语言特有的表达习惯或文化背景，在需要本地化理解的任务中表现受限。
3. **锚语言性能影响上限**：若锚语言（如中文）本身在该模型中表现较差，则难以提供可靠先验。
4. **层选择策略需适配不同架构**：目前的中间层选择基于观察到的 U-shaped pattern，未必适用于所有 MoE 设计。

---

### **未来工作方向**

- 探索 **动态路由对齐机制**，根据不同任务自动调整对齐强度。
- 引入 **多锚点融合策略**，结合多种高资源语言的优势。
- 开发 **抗噪声的路由蒸馏方法**，缓解翻译 artifacts 的负面影响。
- 将 SARA 应用于 **语音、视觉等多模态 MoE 模型**，推动跨模态知识迁移。

---

> 📦 **代码已开源**：[https://github.com/iMoriton/sara](https://github.com/iMoriton/sara)

</details>

---

### 9. [Weave of Formal Thought](https://arxiv.org/abs/2606.25987)

**Authors**: Alexandre Bouayad  
**Category**: cs.CL  
**Published**: 2026-06-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.25987v1  

#### Abstract
Large language models (LLMs) attain remarkable surface fluency on code, yet they neither formally guarantee the syntactic validity of their output nor leverage the hierarchical structure defining the target language. While existing constrained-decoding frameworks address the former, they operate und...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Weave of Formal Thought 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大型语言模型（LLMs）在代码生成任务上表现出高度的**表面流畅性**（surface fluency），但存在两个根本性缺陷：
- **缺乏形式正确性**：输出代码常无法通过语法解析、类型检查或编译，需依赖后处理修复。
- **忽略层次结构**：标准自回归训练将程序视为扁平文本序列，未显式建模编程语言固有的**抽象语法树**（AST）结构，导致模型难以进行结构性规划。

此外，现有**constrained decoding**框架虽能提升语法正确性，但在处理上下文敏感词法分析（如 Python 缩进）、最大匹配分词等方面能力有限，且通常仅近似屏蔽无效子词，牺牲了完整性。

### 提出的新方法：Weave of Formal Thought (WoFT)
WoFT 是一个统一范式，结合了**严格的语法验证**与**可学习的结构化表示**，由两大组件构成：

#### （1）形式推理引擎（Formal Engine / “Loom”）
- 基于 **Tree-sitter** 框架，扩展 **Generalized LR (GLR)** 解析器。
- 引入**推测性词法分析**（speculative-lexing）机制：维护多个并行的词法状态假设，并与 GLR 的图结构栈同步。
- 支持完整的 Tree-sitter 特性，包括：
  - 上下文敏感词法分析（context-sensitive lexing）
  - 最大匹配分词（maximal-munch tokenization）
  - 外部扫描器（external scanners，如 Python 缩进）
- **理论保证**：对任意子词序列，该解码器是**sound and complete**的——仅当子词能扩展为合法程序前缀时才被接受。

#### （2）潜在变量微调方法（Latent-variable Fine-tuning / “Weaver”）
- 将非终结符语法符号（如 `<statement>`）作为**离散潜在变量**（discrete latent variables）插入生成过程。
- 使用 **Reweighted Wake-Sleep (RWS)** 算法优化 **Importance-Weighted Evidence Lower Bound (IW-ELBO)**。
- 模型学会**选择性保留**语法符号：仅当这些符号有助于压缩后续文本时才保留，否则丢弃，形成一种**自适应的结构化草稿板**（adaptive structural scratchpad）。

### 相比现有方法的优势
| 方面 | 现有方法 | WoFT |
|------|--------|------|
| **语法正确性** | Constrained decoding（如 XGrammar, Guidance）近似处理，不完整 | Sound & complete，支持完整 Tree-sitter 规范 |
| **结构感知** | 隐式学习或固定语法注入 | 显式建模离散潜在语法，动态决定是否使用 |
| **训练效率** | 传统结构化模型（如 RNNG）复杂度 O(N³) | 共享主干网络，线性复杂度，适配现代 Transformer |
| **内部推理** | Free-form CoT 或连续隐向量（如 Coconut） | 结构化、可验证、符号化的“正式思维”链 |

---

## 2. 核心实验方法和设置

### 数据集
- **基础模型**：`StarCoder2-3B`
- **训练数据**：从 **The Stack v2** 中采样的 15,000 个 Python 文件（大小 1–100,000 字节）
- **预处理**：
  - 使用 StarCoder2-3B 的 tokenizer 进行子词切分。
  - 使用 Tree-sitter 解析源码生成 AST。
  - 深度优先遍历 AST，将非终结符以**前缀形式**（prefix placement）插入文本序列（如 `<function_definition> def foo(): ...`）。

### 实验设置
- **模型架构**：
  - 在 StarCoder2-3B 基础上扩展词汇表，加入 formal grammar tokens。
  - 使用 **LoRA**（Low-Rank Adaptation, r=32）进行参数高效微调。
  - LoRA 适配器应用于所有主要模块（注意力、FFN、嵌入层、lm_head）。
- **优化配置**：
  - 优化器：AdamW
  - 学习率：3×10⁻⁴
  - 权重衰减：0.01
  - 梯度裁剪：1.0
  - 精度：bfloat16
  - 硬件：NVIDIA A40 GPU
- **RWS 设置**：
  - 每个序列使用 K=4 个粒子（particles）进行重要性采样。

### 评估指标
- 主要指标：**每文本 token 的交叉熵损失**（per-token cross-entropy on surface tokens）
- 报告方式：在线训练损失的滑动平均（over 250 步）

### 基线方法对比
1. **Text SFT**：标准监督微调，仅在原始文本上训练。
2. **Text+Formal SFT**：在文本与 formal tokens 拼接后的序列上进行标准 SFT（即强制教师强制学习所有语法标签）。
3. **WoFT (RWS)**：本文提出的方法，使用 RWS 算法优化 IW-ELBO。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | 表面 token 交叉熵（收敛值） | 相对改进 |
|------|--------------------------|---------|
| Text SFT（基线） | ~0.77 | — |
| Text+Formal SFT | ~0.82 | **变差**（+6.5%） |
| **WoFT (RWS)** | **~0.66** | **↓14.3%** |

> ✅ **核心结果**：WoFT 微调使表面 token 的交叉熵相对降低 **14.3%**，显著优于两种基线。

### 与基线方法的对比结果
- **Text+Formal SFT 表现更差**：表明简单地将语法标签拼接到输入中并强制模型预测，反而**损害了表面文本建模能力**。原因可能是：
  - 序列长度增加，注意力稀释。
  - 模型被迫分配容量去预测“琐碎”的语法结构，而非高熵的代码内容。
- **WoFT 成功的关键**：通过将语法符号作为**潜在变量**，模型可以**自主决定何时使用结构信息**，实现了“按需调用”，避免了冗余开销。

### 消融实验（文中提及，尚未完全展示）
- 文中指出未来将进行系统消融，验证以下机制：
  - **确定性语法注入 vs. 自适应路由**：确认增益来自学习性而非单纯数据增强。
  - **随机掩码 vs. 学习性掩码**：验证 RWS 是否真正学到了有用的结构策略。
  - **不同 formal token 插入策略**（前缀/后缀/字段等）的影响。

---

## 4. 关键结论和发现

### 主要发现
1. **显式语法建模有助于表面生成**：即使目标是生成流畅代码，引入形式语法结构也能显著提升模型对表面 token 的建模能力。
2. **“如何”引入结构至关重要**：不能简单强制模型输出完整语法轨迹（teacher forcing），而应将其作为**可选的潜在变量**，让模型**自主决策**是否使用。
3. **Discretionary Latent Syntax 是有效的**：模型学会了将 formal non-terminals 用作“结构化草稿板”，仅在复杂依赖场景下保留它们，提升了长程一致性。
4. **Sound & Complete 解码是可行的**：基于 GLR + speculative-lexing 的引擎能够实现对 Tree-sitter 完整规范的支持，为高质量代码生成提供可靠基础。

### 方法的局限性
- **当前仅用于训练增强**：形式引擎目前主要用于训练时构建数据和验证，推理时仍依赖 constrained decoding。
- **RWS 训练开销**：尽管使用单粒子回传，K=4 的多粒子采样仍带来一定内存和计算负担。
- **泛化到其他语言**：虽设计为语言无关，但实证集中在 Python，需验证其在其他语言（如 C++、JS）上的效果。
- **下游任务未评估**：目前仅报告了**语言建模损失**，尚未在 HumanEval、MBPP 等**功能正确性基准**上测试最终生成质量。

### 未来工作方向
1. **集成到推理阶段**：将形式引擎用于**实时 constrained decoding**，确保输出始终语法合法。
2. **探索 Sleep Phase**：在 fine-tuning 背景下重新引入 wake-sleep 的 sleep phase 作为 posterior regularizer。
3. **优化解码效率**：
   - 引入字节级词法分析、trie-based 掩码、高效拒绝采样。
   - 融合 **Earley parser** 或 **derivative parsing** 加速。
4. **扩展外部扫描器 API**：支持在 token 流上直接处理任意状态追踪（如缩进、raw string）。
5. **评估下游任务**：在 HumanEval、MBPP 等 benchmark 上测试代码功能正确性和执行通过率。
6. **探索连续潜在空间**：尝试将 formal tokens 映射到连续空间，实现端到端可微的结构化推理。

---

> 🧠 **最终洞见**：  
> WoFT 表明，**最好的“思维链”不是自然语言，而是形式语法本身**。通过将编程语言的形式结构内化为可学习的潜在变量，模型不仅能生成更流畅的代码，还能建立起与人类程序员类似的**层次化、先验性的语法假设机制**，从而弥合“表面流畅”与“结构合理”之间的神经符号鸿沟。

</details>

---

### 10. [Latency-Aware Service Placement using Neural Combinatorial Optimisers for Edge--Cloud Systems](https://arxiv.org/abs/2606.25553)

**Authors**: Kimia Abedpour, Mohammadsadeq Garshasbi Herabad, Zheng Li, Javid Taheri  
**Category**: cs.DC  
**Published**: 2026-06-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.25553v1  

#### Abstract
The growth of Internet of Things (IoT) applications and latency-sensitive services has increased the demand for efficient service placement across compute continuum platforms, such as edge--cloud systems. Modern applications are decomposed into interdependent microservices deployed over heterogeneou...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Latency-Aware Service Placement using Neural Combinatorial Optimisers for Edge-Cloud Systems

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Edge-Cloud 系统中的服务放置问题**（Service Placement），这是一个典型的 **NP-hard 组合优化问题**。随着 IoT 应用和延迟敏感型微服务的普及，如何在异构的计算连续体（compute continuum）中高效部署服务组件，同时满足资源约束、网络延迟和通信开销等要求，成为关键挑战。

传统方法如 **heuristics** 和 **metaheuristics** 面临以下瓶颈：
- **Heuristics**：决策速度快但解质量低，缺乏全局优化能力。
- **Metaheuristics**（如 GA、PSO）：解质量较高但推理时间长，难以用于在线动态场景。
- **RL-based 方法**：虽能自适应环境变化，但在大规模系统中面临状态-动作空间爆炸、训练效率低等问题。

---

### 提出的新方法：EP-NCO
作者提出了一种基于 **Neural Combinatorial Optimisation (NCO)** 的学习框架 —— **EP-NCO**（Edge Placement Neural Combinatorial Optimiser），其核心创新如下：

#### ✅ 双图建模（Dual-Graph Representation）
- 同时建模 **基础设施图**（Network Graph）和 **服务依赖图**（Service DAG）：
  - **NodeGNN** 编码节点资源（CPU、内存）、链路属性（带宽、延迟）；
  - **ServiceGNN** 编码服务组件的资源需求与依赖关系。
- 利用 **Graph Neural Networks (GNNs)** 学习结构化嵌入表示，捕捉跨层依赖。

#### ✅ 自回归强化学习策略（Autoregressive RL Policy）
- 将服务放置建模为一个 **顺序决策过程**，每次选择一个 service-component 放置到合适的 computing node 上。
- 使用 **Reinforcement Learning** 来训练策略网络，目标是最大化负的总响应时间奖励（即最小化延迟）。
- 引入 **hard-decoder** 和 **soft-decoder** 两种解码策略处理约束：
  - **Hard-decoder**：通过可行性掩码（feasibility masking）在推理时排除违反资源/连接约束的动作，确保输出合法。
  - **Soft-decoder**：允许探索非法动作，但通过惩罚项引导模型避免违规。

#### ✅ 快速在线推理能力
- 所有计算复杂度集中在离线训练阶段，训练完成后可在 **亚秒级时间内完成推理**（~0.9–1.0 秒），适用于大规模动态系统。

---

### 相比现有方法的优势
| 方法类别 | 优势 | 劣势 |
|--------|------|-------|
| Heuristics | 推理快、无训练成本 | 解质量差、无法泛化 |
| Metaheuristics (GA/PSO) | 解质量尚可 | 推理慢、不可扩展 |
| RL-based | 可学习复杂模式 | 架构简单、缺乏结构归纳偏置 |
| **EP-NCO (本文)** | ✔️ 高质量解<br>✔️ 超快推理<br>✔️ 强泛化能力<br>✔️ 显式建模结构依赖 | 需要较长训练时间 |

> 🔑 **核心优势总结**：EP-NCO 在保持高质量解的同时实现了 **近实时推理**，解决了“高精度”与“低延迟”之间的根本权衡。

---

## 2. 核心实验方法和设置

### 数据集与问题实例生成
- 所有问题实例由作者开发的 **edge-to-cloud simulation framework** 自动生成。
- 实例覆盖四种规模（S/M/L/XL），具体配置见下表：

| 规模 | 计算节点数 | 边缘设备数 | 云连接设备数 | 服务数 | 每个服务组件数 |
|------|------------|-------------|----------------|--------|----------------|
| S (Small) | 30 | 15 | 15 | 15 | 8 |
| M (Medium) | 60 | 30 | 30 | 30 | 8 |
| L (Large) | 100 | 50 | 50 | 50 | 8 |
| XL (XLarge) | 145 | 75 | 75 | 75 | 8 |

- 所有参数（如 CPU 容量、带宽、延迟、数据大小等）从真实工业范围采样，保证模拟真实性。

---

### 实验设置与评估指标

#### ✅ 主要评估指标
| 指标 | 描述 |
|------|------|
| `Total Response Time` | 总响应时间 = 执行时间 + 传输延迟（含带宽共享影响） |
| `Inference Time` | 单次推理耗时（秒） |
| `Friedman Mean Rank` | 多组实验下的平均排名，用于统计显著性分析 |
| `Break-even Point (N_exec)` | 学习方法总执行时间追上非学习方法所需的实例数量 |

#### ✅ 基线方法对比
分为三类进行比较：

##### （1）Metaheuristics
- **GA**：Genetic Algorithm
- **PSO**：Particle Swarm Optimization

##### （2）Rule-based Heuristics
- **TCA**：Task Continuation Affinity（优先同节点部署依赖组件）
- **MP**：Most Powerful（选最强节点）
- **LP**：Least-Powerful（选刚好够用的弱节点）
- **MDS**：Most Data Size（大数据组件优先放好网络节点）

##### （3）RL-based Ablation Models
- **RL_SH / RL_MH / RL_LH**：硬解码器 RL 模型
- **RL_SS / RL_MS / RL_LS**：软解码器 RL 模型  
（均使用 MLP 编码器而非 GNN）

> 注：所有 EP-NCO 模型也按训练规模（S/M/L）和解码器类型（H/S）命名，如 `EP-NCO_SH` 表示 Small-scale + Hard-decoder。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📈 总体响应时间对比（Figure 4 & Table 6）
| 方法 | 平均响应时间（XL规模） | 相对提升（vs GA） | 相对提升（vs RL） |
|------|--------------------------|--------------------|---------------------|
| **EP-NCO_LH** | **0.3198** | ↓ **46.8%** | ↓ **33.4%** |
| EP-NCO_MH | 0.3031 | ↓ 49.9% | ↓ 37.2% |
| GA | 0.6054 | — | — |
| PSO | 0.6598 | — | — |
| RL_LH | 0.3960 | ↓ 35.6% | — |
| MDS | 1.4661 | ↑ 142% | ↑ 270% |

> ✅ **结论**：EP-NCO 在所有规模下均取得最优表现，尤其在 XL 规模下优势更明显。

---

#### ⏱️ 推理时间与计算成本（Table 4）
| 方法 | 训练时间（小时） | 推理时间（秒） | Break-even 实例数（vs GA） |
|------|------------------|----------------|-------------------------------|
| **EP-NCO_SH** | 2.52h | **0.92s** | ~95 |
| RL_SH | 3.07h | 0.80s | ~115 |
| GA | 0 | 96.57s | — |
| PSO | 0 | 85.22s | — |
| TCA | 0 | 0.016s | — |

> 💡 **关键洞察**：
- 虽然 EP-NCO 有训练开销，但在解决约 **100 个问题实例后即可回本**（end-to-end 更快）。
- 对于频繁部署的大规模系统（如智慧城市、边缘AI平台），长期运行效率远超 metaheuristics。

---

#### 🧪 消融实验结果（Ablation Study）

##### （1）GNN vs MLP 编码器（EP-NCO vs RL）
- 所有 RL 模型（MLP 编码）的表现均显著低于对应 EP-NCO 模型。
- 说明 **GNN 对结构依赖的有效建模至关重要**。

##### （2）Hard-decoder vs Soft-decoder
- 在相同训练条件下，**hard-decoder 版本收敛更快、方差更小、最终性能更好**。
- 例如 `EP-NCO_SH` > `EP-NCO_SS`，且前者稳定性更高（Figure 9）。

##### （3）跨尺度泛化能力（Table 7）
- 模型在未见过的更大规模实例上仍保持良好性能，表明具有较强的 **zero-shot generalisation** 能力。
- 尤其 `EP-NCO_LH` 在 Small/Medium/Large 测试中均排名第一。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **EP-NCO 显著优于现有方法**：
   - 相比 metaheuristics（GA/PSO）降低响应时间 **46%–50%**；
   - 相比 controlled RL 基线降低 **25%–35%**；
   - 实现 **~0.9 秒内完成大型系统（145节点+75服务）的服务放置**。

2. ✅ **GNN + RL 架构有效捕获结构性依赖**：
   - 双图编码机制使模型能够理解服务拓扑与基础设施间的耦合关系。

3. ✅ **hard-decoder 提升稳定性和性能**：
   - 显式约束处理机制（feasibility masking）有助于生成合法且高效的部署方案。

4. ✅ **学习成本可被快速摊销**：
   - 在重复部署场景中，仅需约 **100 次推理即可抵消训练开销**，适合长期运行系统。

---

### 方法的局限性
1. ❗ **假设静态拓扑结构**：未考虑动态加入/退出的节点或链路波动。
2. ❗ **直接链路通信假设**：未支持多跳路由，限制了在复杂网络中的应用。
3. ❗ **依赖合成 workload**：尚未在真实生产环境中验证迁移性能。
4. ❗ **单目标优化**：当前仅优化延迟，未考虑能耗、可靠性、安全性等多目标权衡。

---

### 未来工作方向
1. ✅ **支持动态 workload 和拓扑演化**；
2. ✅ **扩展至 multi-objective optimisation**（如安全、可靠、节能）；
3. ✅ **联邦学习架构**：实现去中心化的分布式训练，适应隐私敏感的边缘环境；
4. ✅ **引入因果推理机制**，增强对突发流量或故障的鲁棒性。

---

## 总结
> **EP-NCO 是首个将 NCO 成功应用于 Edge-Cloud 服务放置的工作，结合 GNN 与 RL 实现了“高质量 + 快速推理”的统一，在大规模延迟敏感系统中展现出巨大潜力。**

它不仅推动了 **Neural Combinatorial Optimisation** 在实际系统问题中的落地，也为未来智能调度系统的设计提供了新范式。

</details>

---

### 11. [Brevity is the Soul of Inference Efficiency: Inducing Concision in VLMs via Data Curation](https://arxiv.org/abs/2606.25432)

**Authors**: DatologyAI,  :, Matthew L. Leavitt, Siddharth Joshi, Haoli Yin, Rishabh Adiga, Haakon Mongstad, Alvin Deng, David Schwab, Bogdan Gaza, Ari Morcos  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.25432v1  

#### Abstract
Inference efficiency is typically pursued by shrinking the model: distillation, pruning, quantization, and sparse routing each lower per-token cost while treating token count as fixed. But output length has been inflating, and it is precisely the component the standard toolkit leaves untouched. Here...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Brevity is the Soul of Inference Efficiency: Inducing Concision in VLMs via Data Curation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **inference efficiency** 优化主要集中在模型层面（如蒸馏、剪枝、量化），通过降低 **FLOPs per token** 来节省成本。然而，这些方法将 **output length** 视为固定不变的因素，而现实中模型输出长度持续增长（尤其是 reasoning models），导致总推理开销不降反升。

本文指出，**output length 是被忽视的关键杠杆**，并提出：  
> **Brevity（简洁性）是推理效率的核心，而数据预处理（data curation）是实现简洁性的有效途径。**

### 提出的新方法与新思路
- **核心思想**：在预训练阶段通过 **data curation** 构建一个“简洁正确”的数据集，使模型学会用更少的 token 给出正确答案。
- **具体方法**：对 **MAmmoTH-VL** 单图子集进行数据清洗，去除冗余、啰嗦、链式思维（chain-of-thought）等长输出模式，保留直接、准确的回答。
- **理论依据**：该过程是一种 **amortized inference** —— 将本应在推理时完成的复杂推导“摊销”到训练过程中，使前向推理成为“缓存好的快速响应”。

### 相比现有方法的优势
| 方法类别 | 作用对象 | 局限性 | 本文优势 |
|--------|--------|-------|--------|
| Distillation, Quantization, MoE | 模型参数（FLOPs/token） | 忽视 output length | 同时优化 **tokens per correct answer** |
| Decoding 控制（length penalty） | 推理时干预 | 外部约束，模型未内化简洁性 | **内化简洁性**，无需外部控制 |
| RLHF with length reward | 训练目标 | 易被长度主导，质量提升有限 | 从源头构建高质量简洁数据 |

> ✅ **优势总结**：通过一次性的数据干预，实现 **Pareto 改进** —— 更短输出 + 更高准确率 + 更低成本。

---

## 2. 核心实验方法和设置

### 数据集
- **主训练数据**：
  - **MAmmoTH-VL** 单图像子集（baseline）
  - **Curated MAmmoTH-VL**（本文方法，去除冗余、强制简洁）
- **评估数据集**（20个 VLM benchmarks，组成 **20-eval suite**）：
  - 能力覆盖：**Referring & Grounding, General VQA, OCR & Document, Captioning, Spatial & 3D, Counting, Chart & Diagram, Math**
  - 具体包括：MMBench, RealWorldQA, DocVQA, OCRBench, TextVQA, CAPability, DetailCaps, AI2D, ChartQA, MathVista, 3DSRBench 等

### 实验设置
- **模型规模**：1B–4B 参数的 dense VLMs
- **对比模型**：
  - **内部对照组**：`DatologyAI`（curated） vs `Mammoth`（uncurated），同 backbone、同训练流程
  - **外部前沿模型**：`Qwen3.5`, `InternVL3/3.5`, `Perceptron-Isaac` 等 open-weight VLMs
- **训练一致性**：除数据外，其他所有设置（backbone, context length, training recipe）保持一致

### 评估指标
| 指标 | 定义 | 说明 |
|------|------|------|
| **Cost-of-Pass** | `FLOPs per correct answer` = $ \frac{2N \cdot n_{\text{out}}}{\text{accuracy}} $ | 核心指标，衡量“每得到一个正确答案所花费的计算量” |
| **Tokens per correct answer** | $ \frac{n_{\text{out}}}{\text{accuracy}} $ | 去除硬件影响，纯文本效率度量（OckScore） |
| **Length-controlled Accuracy Gap (AME)** | 在控制输出长度后，curated 模型相比 baseline 的准确率提升 | 使用 GLM 回归分离长度与质量的影响 |
| **Verbosity Ratio** | 平均输出 token 数之比 | 衡量简洁性程度 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **Cost-of-Pass 优势显著**
| 模型 | 参数 | 输出长度 | 准确率 | FLOPs/correct (TFLOPs) |
|------|------|----------|--------|-------------------------|
| **Datology 4B** | 4.4B | **32.5** | 0.691 | **0.41** |
| Qwen3.5-4B | 4.0B | 1,284 | 0.704 | 14.58 |

> 💡 **Cost-of-Pass 降低 35×**，且准确率几乎持平（仅低 1.3 pp）

#### ✅ **简洁性带来巨大 token 效率提升**
- **Tokens per correct answer**：
  - Datology 4B: **47**
  - Qwen3.5-4B: **1,823**
  > 📉 **相差约 39×**，验证简洁性是主要驱动因素

#### ✅ **在相同输出长度下，准确率更高**
- 在控制输出长度的 GLM 回归中：
  - **平均 AME = +17.55 pp**（95% CI: +13.33 ~ +21.77）
  - 且效果随模型规模增大而增强：
    - 1B: +16.7 pp
    - 2B: +15.6 pp
    - **4B: +21.2 pp**

> 🔍 说明：**简洁性 ≠ 牺牲质量**，反而提升了单位长度的信息密度

#### ✅ **综合成本优势惊人**
- **Datology 1B vs Qwen3.5-4B**：
  - 成本低 **143×**
  - 仅牺牲 5–7 pp 准确率
- **百万次正确回答成本**（H100 自托管）：
  - Datology 4B: **\$1.34**
  - Qwen3.5-4B: **\$47.24**
  > 💵 差价相当于“一瓶汽水 vs 一箱汽油”

---

### 与基线方法的对比结果

| 对比维度 | 结果 |
|--------|------|
| **vs 未裁剪数据（Mammoth）** | 在所有尺度上，**同时更短、更准、更便宜**（Pareto 改进） |
| **vs Qwen3.5 系列** | 花费 2.4–3.6× 更少的成本，换取略低 4–7 pp 的准确率（性价比极高） |
| **vs InternVL3.5 系列** | **全面胜出**：更准确 + 更便宜（2.7–3.9×） |
| **vs Qwen3.5-4B** | **无法进行长度匹配比较**：两者输出长度分布重叠仅 2.2%，Qwen3.5-4B 无“简洁操作点” |

---

### 消融实验与深入分析

#### 🔍 **Generic verbosity 不带来收益**
- `Mammoth`（冗长但非推理型） vs `Datology`（简洁）：
  - 在所有 16 个 (scale × capability) 组合中，**M-D ≤ 0**
  - 最差时准确率低 **54.9 pp**（Referring & Grounding）
  > ❌ **单纯变长 ≠ 变强**

#### 🔍 **Reasoning verbosity 的优势随规模缩小**
- 在 2B 模型中，`Qwen3.5`（推理型冗长）在 **4/8** 能力组优于简洁模型
- 在 4B 模型中，仅在 **1/8**（OCR & Document）仍占优
- 在 **Referring & Grounding** 上，简洁模型反超最多达 **9.6 pp**

> 📉 **随着模型能力增强，chain-of-thought 的边际效益递减**

#### 🔍 **简洁模型与推理模型具有互补错误集**
- 在单个样本级别：
  - `Qwen3.5` 能“救回” `Datology` 错误的 **25–78%** 的样本
  - `Datology` 也能避免 `Qwen3.5` 的“自我误导”（misled rate 高达 73.4% in PixMo Points）
- 说明：**简洁性 ≠ 压缩推理**，而是不同的推理路径

#### 🔍 **简洁性优势来自响应本身，而非拒绝回答**
- 三因子分解（refusal-rate, refusal-length, response-length）显示：
  - **98–100% 的长度差异来自“响应时更简洁”**
  - 拒绝率差异可忽略（甚至 curated 模型更少拒绝）
  > ✅ **简洁性是真正的推理效率提升，不是行为偏移**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Brevity is the soul of inference efficiency**  
   输出长度是比模型压缩更重要的效率杠杆。
2. ✅ **Data curation is a powerful lever for brevity**  
   通过训练数据引导，可让模型内化简洁性，无需推理时干预。
3. ✅ **Concision does not sacrifice quality — it enhances it**  
   在相同长度下，简洁模型更准确，且优势随模型规模扩大。
4. ✅ **Generic verbosity buys nothing; reasoning verbosity’s edge shrinks with scale**  
   冗长但无结构的输出无益；有结构的推理输出优势也在大模型上减弱。
5. ✅ **Reasoning and brevity are orthogonal capabilities**  
   简洁模型能答对推理模型错的题，反之亦然，二者可互补。

### 方法的局限性
- **依赖高质量人工标注或规则进行数据清洗**，扩展到更大数据集可能成本高。
- 当前方法主要针对 **single-image VLMs**，多图或多模态序列场景尚未验证。
- **未解决“何时需要长输出”** 的动态决策问题（如是否启用 chain-of-thought）。

### 未来工作方向
1. **自动化数据简洁性标注**：利用模型自身或小模型自动识别并压缩冗余输出。
2. **联合优化 reasoning 与 brevity**：构建既能深度推理又能简洁表达的模型。
3. **动态输出长度控制**：基于任务难度自适应决定是否展开推理。
4. **将 amortized inference 扩展到部署反馈闭环**：将线上高成本模式反馈至训练数据，形成“数据飞轮”。

---

> **结语**：  
> 如 Pascal 所言：“我写这封信太长，是因为我没有时间把它写短。”  
> 本文证明：**通过数据预处理“一次性支付”简洁性的代价，可在每一次推理中“摊销”巨大的效率红利**。  
> **Brevity is not free — but curation makes it amortizable.**

</details>

---

### 12. [Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization](https://arxiv.org/abs/2606.26002)

**Authors**: Kamar Hibatallah Baghdadi, Kawther Guoual Belhamidi, Sara Belhadj, Aissa Boulmerka, Nadir Farhi  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.26002v1  

#### Abstract
We present HiReLC, a hierarchical ensemble-reinforcement learning framework for automated joint quantization and structured pruning of deep neural networks. The framework decomposes the compression search across two levels of abstraction: low-level agents (LLAs) operate independently per block, sele...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代深度神经网络（如 Vision Transformers 和 CNNs）在视觉任务上表现优异，但其高参数量、大内存占用和推理延迟限制了在资源受限设备上的部署。现有的模型压缩方法（如静态剪枝、单精度量化或单一目标优化）存在以下不足：
- 忽视跨层依赖关系；
- 缺乏对微调过程的自适应调整；
- 在大规模动作空间中强化学习（RL）策略方差高、搜索效率低；
- 缺少基于敏感度的指导机制。

本文旨在解决**联合结构化剪枝（structured pruning）与混合精度量化（mixed-precision quantization）的自动化压缩框架设计问题**，提升压缩效率与精度保持能力。

---

### 提出了什么新方法或新思路
作者提出 **HiReLC**（Hierarchical Reinforcement Learning for Neural Network Compression），一个**分层集成式强化学习框架**，用于自动联合优化神经网络的剪枝与量化。

#### 主要创新点包括：

- **分层代理架构（Hierarchical Agent Architecture）**
  - **High-Level Agent (HLA)**：负责全局预算分配，为每个模块（block）设定压缩目标（如保留比例、位宽范围等），通过 Fisher Information 敏感度估计协调不同模块的重要性。
  - **Low-Level Agent (LLA)**：每个模块独立运行，从多离散动作空间中选择每核（per-kernel）配置（bitwidth, keep-ratio, quantization type, granularity）。
  - 分解搜索空间，降低复杂性，实现更精细控制。

- **敏感度感知机制（Sensitivity-Aware Guidance）**
  - 引入 **Fisher Information Score** 作为模块重要性的先验，在奖励函数中加入敏感度惩罚项 $ P_{\text{sens}} $，防止对关键模块过度压缩。
  - HLA 利用敏感度进行预算修正（如高敏感模块减小压缩强度），提升精度稳定性。

- **代理增强的主动学习循环（Surrogate-Augmented Active Learning Loop）**
  - 使用轻量级 MLP 代理模型预测压缩后的准确率，减少昂贵的真实微调次数。
  - 冷启动阶段使用 **logit-MSE proxy**（输出 logits 差异）提供低成本质量信号。
  - 代理仅用于奖励塑形（reward shaping），最终性能仍以真实微调为准，避免误差累积。

- **架构无关性与模块化设计**
  - 控制器与底层网络拓扑解耦，支持多种架构（ViT、ResNet、MobileNetV2 等）。
  - 支持灵活的量化粒度（uniform, log, per-channel, learned）和数值类型（INT/FLOAT）。

---

### 相比现有方法的优势
| 对比维度 | 现有方法（如 AMC, HAQ, HAWQ-V2） | HiReLC |
|--------|-------------------------------|------|
| 压缩方式 | 单一轴（仅剪枝或仅量化） | 联合剪枝 + 混合精度量化 |
| 控制结构 | 单层控制器 | 双层分层控制（HLA + LLA） |
| 敏感度利用 | 无或后处理 | Fisher 指导贯穿 HLA 与 LLA |
| 搜索效率 | 高计算成本（如 Hessian） | 代理模型 + 主动学习显著降低评估开销 |
| 架构通用性 | 特定于 CNN 或 ViT | 架构无关，统一框架 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CIFAR-10 / CIFAR-100**：用于评估 Vision Transformer 模型（DeiT-Small, CLIP ViT-B/32）。
- **Tiny ImageNet**（200类，64×64）：用于评估 CNN 模型（ResNet18, MobileNetV2）。
- 所有实验均基于预训练模型进行后训练压缩（post-training compression）。

---

### 实验设置和评估指标

#### 模型与压缩配置
| 模型 | 参数特点 | 压缩动作空间 |
|-----|---------|-------------|
| DeiT-Small / Base | Vision Transformer，含 QKV、Attn Proj、MLP 层 | per-kernel bitwidth (4–8), keep-ratio, quantization type (INT/FLOAT), granularity |
| CLIP ViT-B/32 | 多模态预训练模型 | 同上 |
| ResNet18 / MobileNetV2 | CNN 架构 | 同上 |

#### 评估指标
- **CR**（Compression Ratio）：有效参数存储压缩比，定义为 $ \text{CR} = 1/\bar{v} $，其中 $ \bar{v} $ 是加权平均的参数保留比例。
- **MSR**（Model-Size Reduction）：模型大小缩减率，$ \text{MSR} = 1 - 1/\text{CR} $。
- **Accuracy Drop**：压缩后模型 top-1 准确率相对于原始模型的变化（正值表示下降，负值表示上升）。
- **Surrogate MAE**：代理模型预测准确率的平均绝对误差。

#### 基线方法对比
未直接复现所有基线，而是进行**上下文比较**（contextual comparison），引用原论文报告的结果（多数基于 ImageNet-1K），强调方法定位差异：
- **AMC**（He et al., 2018）：基于 RL 的剪枝。
- **HAQ**（Wang et al., 2019）：硬件感知量化。
- **HAWQ-V2**（Dong et al., 2020）：基于 Hessian 的量化。
- **I-ViT**（Li and Gu, 2023）：ViT 整数量化。
- **DeepCompress-ViT**（Ahmed et al., 2025）：面向存储的 ViT 压缩。

> 注：HiReLC 在非 ImageNet 数据集上测试，因此不构成严格公平对比，而是展示其在统一框架下的有效性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 实验编号 | 模型 | 数据集 | 压缩比 CR | MSR | 准确率变化 |
|--------|------|-------|----------|-----|------------|
| Exp 1 | DeiT-Small | CIFAR-100 | **6.23×** | 83.9% | +1.72% |
| Exp 2 | CLIP ViT-B/32 | CIFAR-100 | **6.63×** | 84.9% | +0.55% |
| Exp 3 | CLIP ViT-B/32 | CIFAR-10 | **6.64×** | 84.9% | **-3.83%**（精度提升） |
| Exp 4 | ResNet18 | Tiny ImageNet | **5.99×** | 83.3% | +4.22% |
| Exp 5 | MobileNetV2 | Tiny ImageNet | **6.59×** | 84.8% | +5.62% |
| Exp 6 | DeiT-Small | CIFAR-100 (mixed INT/FLOAT + log) | **6.27×** | 84.0% | +1.17% |
| Exp 7 | DeiT-Base | CIFAR-100 (mixed + log) | **6.72×** | 85.1% | +2.11% |

> ✅ **最高压缩比达 6.72×，最大精度增益达 3.83%（Exp 3）**

---

### 与基线方法的对比结果（Table 6 上下文比较）
- HiReLC 在多个架构上实现了 **~6× 的参数存储压缩比**，优于大多数单一压缩方法（如 AMC 的 2× FLOPs 剪枝、HAQ 的 ~1.95× 推理加速）。
- 尽管 HAWQ-V2 报告高达 13× 存储压缩，但其基于 Hessian 迹计算，计算代价极高；而 HiReLC 通过代理模型大幅降低搜索成本。
- I-ViT 实现 4× INT8 量化，但未结合剪枝；HiReLC 实现更高压缩且支持联合优化。

> 💡 结论：HiReLC 并非追求极致压缩比，而是在**统一框架下实现高效、稳定、可迁移的联合压缩**。

---

### 消融实验结果（Table 5 & Figure 6）

#### （1）移除 HLA 的影响（Uniform Budget）
| 配置 | 准确率下降 | 压缩比 |
|------|-----------|--------|
| Full HiReLC (HLA + LLA) | **1.72%** | 6.23× |
| LLA only (uniform budget) | 6.80% | **8.87×** |

> ❌ 移除 HLA 导致严重过压缩，精度大幅下降，说明 HLA 对预算调控至关重要。

#### （2）是否使用敏感度指导（Figure 6）
- 使用 Fisher 敏感度时，**平均准确率下降降低 1.30 pp**（从 4.40% → 3.10%），且标准差更小。
- 表明敏感度引导能显著提升精度保持能力和训练稳定性。

#### （3）LLA 代理模型效果（Table 4）
| 模型 | Surrogate MAE |
|------|----------------|
| ResNet18 | 15.20% |
| MobileNetV2 | 10.37% |
| DeiT-Small | 3.57% |

> 📌 ViT 上代理更可靠（MAE 更低），CNN 上噪声较大，但仍可用于粗略排序。

---

## 4. 关键结论和发现

### 主要发现
1. **分层策略分解是有效的**：HLA 负责宏观调控，LLA 实现微观优化，二者协同可在大动作空间中高效探索。
2. **敏感度先验显著提升性能**：Fisher Information 可有效识别关键模块，防止其被过度压缩，带来约 1.3% 的精度增益。
3. **联合剪枝与量化优于单一压缩**：HiReLC 自动平衡两种技术，在保持精度的同时实现更高压缩比。
4. **压缩可作为正则化手段**：在 CLIP 模型上（Exp 3），压缩+微调反而使准确率**提升 3.83%**，表明压缩有助于缓解过拟合。
5. **代理模型显著降低搜索成本**：通过 surrogate + logit-MSE proxy，减少了数百次完整微调，使框架更具实用性。

---

### 方法的局限性
1. **未优化实际延迟或能耗**：当前指标聚焦“参数存储压缩比”，未考虑稀疏索引开销或硬件执行时间，压缩不一定转化为线性速度提升。
2. **代理模型在紧凑 CNN 上噪声大**：如 ResNet18 上 MAE 达 15.2%，需更大 warm-up 池或更强模型改进。
3. **预算合规性不足**：LLA 倾向于早期过压缩（budget compliance ~60%），未来需加强约束机制。
4. **对比非严格匹配**：部分基线结果来自 ImageNet-1K，无法完全横向比较。

---

### 未来工作方向
1. **引入硬件反馈闭环**：将真实设备上的 latency 或 energy 测量纳入奖励函数，构建“hardware-in-the-loop”压缩系统。
2. **扩展至更大模型家族**：应用于 Large Language Models（LLMs）或其他多模态架构。
3. **改进约束满足机制**：添加确定性后处理步骤，确保最终配置满足精度与压缩目标。
4. **增强代理建模能力**：采用图神经网络或 Transformer 建模层间依赖，提升预测准确性。
5. **开展严格基准重现实验**：在同一数据集与协议下全面对比主流压缩方法。

---

> 🔚 **总结**：HiReLC 提出了一种新颖、实用、架构无关的分层强化学习框架，成功实现了神经网络的自动化联合剪枝与量化。其实验验证充分，设计理念具有启发性，为后续研究提供了坚实基础。

</details>

---

### 13. [Accelerating Disaggregated RL for Visual Generative LLMs with Diffusion-Based Parallelism and Trainer-Assisted Generation](https://arxiv.org/abs/2606.24369)

**Authors**: Sijie Wang, Zhengyu Qing, Zhiqiang Tan, Yiming Yin, Yeqing Zhang, Yaoyuan Wang, Qiang Wang, Xiaowen Chu, Shaohuai Shi  
**Category**: cs.AI  
**Published**: 2026-06-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.24369v2  

#### Abstract
Reinforcement learning (RL) has become a dominant post-training paradigm, driving the emergence of high-performance RL systems such as veRL for autoregressive large language models (LLMs). In parallel, diffusion-oriented RL algorithms, e.g., DanceGRPO and FlowGRPO, have rapidly expanded the scope of...

---

### 14. [BluTrain: A C++/CUDA Framework for AI Systems](https://arxiv.org/abs/2606.24780)

**Authors**: Adhitya Charan, Adwaid Suresh, Anuj Kumar, Aparna A, Dhanakumar K, Dharun M S, Dinesh G, Goutham Kumar Reddy K, Harshini V M, Jenifa D, Jona Delcy C A, Kathirvel S, Killi Uma Maheswara Rao, Kiruthik Kanna M, Kurra Vishnu Sai, Madhumithaa G K, Navin Kumar V, Ram Charan Golla, Revathi T, Rishikkanth R, Sanjay Krishna M V, Surendra Vendra  
**Category**: cs.AI  
**Published**: 2026-06-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.24780v1  

#### Abstract
Progress in deep learning is, at scale, more a matter of systems engineering than of modelling: the behaviour of a model in training (its throughput, its memory footprint, and the numerical fidelity of the result) is determined less by the architecture itself than by how that architecture is express...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*BluTrain: A C++/CUDA Framework for AI Systems*

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流深度学习框架（如 PyTorch）在大规模训练中存在系统级瓶颈，包括：
- **Python 运行时开销**：Global Interpreter Lock (GIL) 和动态调度导致 GPU 利用率不足。
- **内存管理低效**：通用缓存分配器（如 PyTorch 的 `caching_allocator`）易产生碎片，限制了可训练模型规模。
- **硬件表达不精确**：高层抽象与底层硬件之间脱节，难以实现最优的数值精度和计算效率。

### 提出的新方法与架构
BluTrain 是一个从第一性原理构建的、轻量且通用的分布式训练框架，其核心创新在于**全栈原生控制**（end-to-end native control），具体体现在以下方面：

#### 架构设计原则
- **绝对控制所有软件层**：完全基于标准 C++ 和核心 CUDA 编程模型，无外部依赖。
- **静态特化（Compile-Time Specialization）**：通过模板元编程将数据类型、内存布局、算法分块等参数在编译期确定，消除运行时分支和调度开销。
- **严格保持数值保真度（Numerical Fidelity）**：全程使用数学上稳定的算法，避免近似计算，确保收敛轨迹一致。
- **抽象复杂性以简化建模**：提供简洁接口，同时暴露底层调优能力。

#### 关键组件创新
| 组件 | 创新点 |
|------|--------|
| **Tensor & Ops 模块** | 自研反向模式自动微分引擎（reverse-mode autograd），动态构建拓扑图；集成自定义 GEMM 库 BluBLAS，针对 Ampere 和 Ada 架构手调优化。 |
| **DTMS (Distributed Training Management System)** | 统一管理 DDP、Tensor Parallelism (TP)、Context Parallelism (CP) 和 Pipeline Parallelism，支持通信-计算重叠。 |
| **MLIR-based Deep-Learning Compiler** | 自研 JIT 编译器，将 autograd 图转化为 MLIR 中间表示，进行全局代数优化，并生成高度优化的 NVPTX 内核。 |
| **Deterministic Caching Allocator** | 动态分析前向-后向周期中的内存使用模式，自适应调整对齐策略，减少内部碎片；在第0步后执行定向缓存清理，永久释放初始化阶段的临时内存。 |

### 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **性能** | 更高的吞吐量（throughput）和更低的每步耗时（step time）。 |
| **内存效率** | 显著降低 VRAM 占用，提升单卡可训练模型上限。 |
| **稳定性与可靠性** | 支持预测性故障检测与自动恢复，适合长时间多卡训练任务。 |
| **可扩展性** | 原生支持多种并行范式，且通信与计算解耦，利于扩展至多节点。 |

---

## 2. 核心实验方法和设置

### 数据集
- 主要使用合成数据或公开文本语料：
  - **FineWeb-Edu**：用于 124M 参数 GPT-2 模型训练，总数据量约 10B tokens。
  - 长上下文实验未指定具体数据集，仅强调序列长度变化。

### 实验设置
| 项目 | 设置详情 |
|------|----------|
| **硬件平台** | 单卡或 8× RTX 6000 Ada（48 GiB VRAM），部分实验使用 RTX 5070 对比。 |
| **模型配置** | - **基准模型**：124M 参数 GPT-2（decoder-only），context length = 1024<br>- **长上下文模型**：context length 扩展至 16,384<br>- **大模型测试**：2.42B 参数 GPT-2，验证最大可训练规模 |
| **训练配置** | - 精度：FP32（为公平比较）<br>- Global Batch Size：524,288<br>- Optimizer：AdamW<br>- 并行策略：DDP（8 GPUs）为主，辅以 TP 和 CP 测试 |

### 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput (tok/s)** | 每秒处理的 token 数量，衡量训练速度。 |
| **Memory Footprint (VRAM)** | GPU 显存占用峰值及稳态值，决定模型可扩展性。 |
| **Numerical Fidelity** | 最终验证损失（validation loss）是否与基线一致，以及训练曲线是否重合。 |
| **Checkpoint Latency** | 模型保存/加载延迟，影响容错效率。 |

### 基线方法对比
- **PyTorch (eager mode)**：默认动态图执行。
- **PyTorch (compile mode)**：启用 `torch.compile` 进行图优化。
- **Megatron-LM**：作为 Tensor Parallelism 的参考实现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 吞吐量（Throughput）
在 **124M GPT-2, 8×RTX 6000 Ada, FP32** 上：

| 框架 | 平均吞吐量 (tok/s) | 平均步时 (ms) |
|-------|---------------------|---------------|
| PyTorch (eager) | 394,663 | 1359.14 |
| PyTorch (compile) | 402,453 | 1326.73 |
| **BluTrain (eager)** | **406,595** | **1313.43** |

👉 **结论**：BluTrain 较 PyTorch eager 快 **~3%**，较 PyTorch compile 仍快 **~1%**。

#### ✅ 内存效率（Memory Efficiency）
相同配置下稳态显存占用：

| 框架 | 显存占用 (GiB/GPU) | 相对节省 |
|-------|--------------------|-----------|
| PyTorch (default) | 27.54 | — |
| PyTorch (pow-2) | 26.35 | — |
| PyTorch (compile) | 24.38 | — |
| **BluTrain** | **21.48** | **↓22%** |

> 在长上下文（16,384）场景下，BluTrain 峰值显存为 **40.4 GiB**，而 PyTorch 达到 **47.8 GiB**（仅剩 0.2 GiB headroom），BluTrain 节省 **15%** 并保留 **7.6 GiB** 安全余量。

#### ✅ 数值保真度（Numerical Fidelity）
- 两者的训练/验证损失曲线几乎完全重合。
- 最终验证损失：
  - PyTorch: **3.0695**
  - BluTrain: **3.0675**（略优）
- 最小训练损失：
  - PyTorch: **2.8793**
  - BluTrain: **2.8771**

👉 表明 BluTrain 不仅没有因优化引入数值误差，反而实现了更优收敛。

#### ✅ 可训练模型上限突破
| 模型 | 参数量 | 硬件 | 是否成功训练 |
|------|--------|--------|----------------|
| GPT-2 | 2.42B | 单张 RTX 6000 Ada (48 GiB) | ❌ PyTorch OOM<br>✅ **BluTrain 成功**（峰值 46.9 GiB） |

👉 BluTrain 在固定硬件上显著提升了“最大可训练参数”天花板。

#### ✅ 分布式并行性能
- **Tensor Parallelism**：相比 Megatron-LM，在 TP=2 下吞吐提升 **19.7%**。
- **Context Parallelism**：在双卡 RTX 6000 Ada 上，相比 PyTorch 实现 **+30.1%** 吞吐增益，且最终 loss 更低。
- **Checkpointing**：异步检查点机制将 GPU stall 时间从数百毫秒降至 **20–57 ms**（其余写盘操作后台完成）。

### 消融实验结果
论文**未提供系统性的组件消融实验**（ablation study），即未量化每个子模块（如自定义 allocator、compiler、kernel 优化）对整体性能的具体贡献比例。

> 作者承认这是当前工作的局限之一：“Lack of systematic component ablations... remains pending.”

---

## 4. 关键结论和发现

### 主要发现
1. **系统工程决定训练效能上限**：模型行为更多由系统实现而非架构本身决定。
2. **全栈原生控制可带来稳定收益**：即使在 FP32 这种“无加速 trick”的条件下，通过精细控制每一层，仍能实现 **~3% 的端到端加速** 和 **高达 22% 的显存节省**。
3. **内存管理是关键瓶颈**：BluTrain 的 deterministic allocator 是其显存优势的核心来源，尤其在长序列和大模型场景下表现突出。
4. **无需牺牲数值精度换取性能**：严格的数值保真控制使得训练曲线精确复现甚至超越基线。

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **缺乏组件级归因分析** | 无法明确指出各模块（如 allocator、compiler）对性能提升的具体贡献。 |
| **验证范围有限** | 当前实证集中在 GPT-2 架构，尚未在 CNN、ViT、MoE 等其他模型上全面验证。 |
| **分布式扩展性待验证** | 实验集中于单节点（8 GPU），跨节点多机扩展下的网络拥塞、同步开销等问题尚未充分测试。 |
| **故障恢复机制未经实战检验** | 故障分类和恢复逻辑虽设计完整，但尚未在真实数据中心的大规模异构环境中验证其有效性。 |
| **硬件绑定性强** | 当前版本专为 NVIDIA GPU（特别是 Ada 架构）优化，缺乏对 AMD ROCm 或其他加速器的支持。 |

### 未来工作方向
1. **硬件无关化**：将 MLIR 编译器后端扩展至支持多种硬件（如 AMD, Apple Silicon, ASICs），实现真正的跨平台高性能。
2. **建立数学保真模型**：系统研究算术格式、融合顺序、累加策略对 end-to-end 收敛的影响，构建可预测的数值行为理论。
3. **扩大模型验证范围**：在 NLP、CV、Speech、Recommender Systems 等多个领域部署大规模模型训练实验。
4. **增强分布式鲁棒性**：完善多节点拓扑感知调度、抗网络降级能力、跨集群容错机制。
5. **开放生态建设**：目前为闭源框架，未来可能考虑开源部分模块以促进社区协作。

---

> **总结一句话**：  
> BluTrain 证明了通过**从第一性原理出发、全栈原生控制**的方式，可以在不依赖高级语言灵活性的前提下，构建出比主流框架更快、更省内存、数值更精确的训练系统，为下一代 AI 系统基础设施提供了新的设计范式。

</details>

---

### 15. [Supervised Reinforcement Learning for the Coordination of Distributed Energy Resources](https://arxiv.org/abs/2606.24947)

**Authors**: Haoyuan Deng, Yihong Zhou, Thomas Morstyn, Yi Wang  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.24947v1  

#### Abstract
The increasing integration of distributed energy resources (DERs) is crucial for power system decarbonization, yet unlocking DERs' flexibility is challenged by their inherent uncertainties and modelling complexity. As traditional optimization methods struggle with such uncertainty and complexity of ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Supervised Reinforcement Learning for the Coordination of Distributed Energy Resources**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
分布式能源资源（**DERs**）在电力系统脱碳中扮演关键角色，但其内在不确定性与建模复杂性（如非线性、非凸模型）使得传统优化方法难以有效释放其灵活性。标准的 **Reinforcement Learning (RL)** 方法虽然能适应复杂环境，但在从零开始训练时面临**样本效率低**和**策略次优**的问题。

### **提出的新方法与新思路**
本文提出了一种名为 **Supervised Reinforcement Learning (SRL)** 的两阶段框架，用于学习 DER 协调控制策略，其核心思想借鉴自大语言模型（**LLMs**）中的“监督预训练 + RL 微调”范式：

- **第一阶段：监督预训练（Supervised Pre-training）**  
  利用历史操作数据（demonstration data）以监督学习方式对策略网络进行初始化，使其具备基本的操作能力，避免从随机探索开始。

- **第二阶段：两步微调（Two-step Fine-tuning）**  
  - **离线微调（Offline Fine-tuning）**：在高保真模拟环境中通过 **PPO** 等 RL 算法进一步优化策略，提升性能。
  - **在线微调（Online Fine-tuning）**：将策略部署到真实环境中，适应实际动态，弥合“sim-to-real”差距。

### **相比现有方法的优势**
| 对比维度 | 现有方法局限 | SRL 框架优势 |
|--------|-------------|------------|
| **训练起点** | 从随机策略开始，探索效率低 | 通过监督预训练获得“warm start”，显著提升样本效率 |
| **数据利用** | 多数方法仅将历史数据存入 replay buffer 用于训练 | 将历史数据用于**策略初始化**，实现更高效的知识迁移 |
| **微调机制** | 多为单阶段微调或仅离线 RL | 引入**两步微调**，先在模拟中优化，再在现实中适应，降低风险并提升最终性能 |
| **鲁棒性** | 对演示数据质量敏感 | 在低质量演示下仍能超越标准 RL 方法，表现出强鲁棒性 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **电力需求与可再生能源出力数据**：来自英国国家电网（UK National Grid）。
- **环境温度数据**：来自 MERRA-2 气象数据库。
- **电价方案**：采用分时电价（**TOU**），分为峰、平、谷三个时段（见 Table I）。
- **演示数据生成**：基于模拟模型重新求解历史状态下的最优动作，但引入人为约束（目标函数值放宽至最优的 1.1~1.3 倍），以模拟不同质量的历史数据：
  - **Near-Optimal (NO)**：无约束，接近全局最优
  - **High-Quality (HQ)**：目标放宽 10%
  - **Medium-to-High-Quality (MtHQ)**：放宽 20%，储能未被使用
  - **Low-Quality (LQ)**：放宽 30%，存在反直觉行为

### **实验设置**
- **研究场景**：一个含 PV、WT、DG、ES 和 TCL 的微网能量管理问题。
- **时间划分**：
  - 第1月：预训练集
  - 第2月：离线微调集
  - 第3–4月：在线微调集
  - 第5月：测试集（30天）
- **硬件平台**：NVIDIA RTX 3080 Ti GPU，Intel Xeon CPU，188GB RAM
- **实现工具**：Python + PyTorch，优化器使用 Gurobi 10.0.3

### **评估指标**
- **主指标**：30天测试期内的**累计运行成本**（越低越好）
- **辅助指标**：
  - 收敛速度（episode reward 曲线）
  - 训练时间开销
  - 不同规模 DER 下的成本削减效果

### **基线方法对比**
共比较六种方法：
1. **PP (Pre-trained Policy)**：仅监督预训练，不进行 RL 微调
2. **Vanilla PPO**：标准 PPO，从零开始训练
3. **sMPC (Stochastic Model Predictive Control)**：基于场景的随机 MPC，预测窗口 1–3 小时
4. **Perfect MILP**：理想情况下的确定性混合整数线性规划（性能上限）
5. **SRL-1**：SRL 框架，仅含在线微调（无离线微调）
6. **SRL-2**：完整 SRL 框架，含**两步微调**

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 方法 | 累计成本（£） | 相对于 PPO 成本降幅 | 与完美上限差距 |
|------|--------------|---------------------|----------------|
| **PPO** | ~13,000 | — | — |
| **sMPC-3** | ~12,000 | ~7.7% | — |
| **SRL-1 (HQ)** | < sMPC-3 | >7.7% | — |
| **SRL-2 (MtHQ)** | < sMPC-3 | 显著优于 | — |
| **SRL-2 (NO)** | **~11,500** | **~11.5%** | **仅高出 £723.88（+6.57%）** |

> ✅ **SRL-2 在 NO 数据下已非常接近理论最优（Perfect MILP）**

### **与基线方法的对比结果**
- **SRL-1 vs PPO & PP**：
  - 在所有演示质量下均**显著优于 PPO 和 PP**
  - 即使使用 **LQ 演示数据**，SRL-1 仍能收敛到优于从零训练的 PPO
  - 表明 SRL 能有效提炼并改进**次优甚至劣质专家经验**

- **SRL-2 vs SRL-1**：
  - 在**初始性能**和**最终收敛奖励**上全面领先
  - 例如，在 HQ 数据下，SRL-2 在在线微调初期即达到更高 reward，且最终 reward 更高
  - 验证了**离线微调**作为“桥梁”的有效性，避免陷入局部最优

- **SRL-2 vs sMPC**：
  - **SRL-2 (MtHQ)** 已优于 **sMPC-3**
  - **SRL-2 (NO)** 接近完美 MILP，远超所有 sMPC 变体
  - 特别值得注意的是，**SRL 仅依赖单步预测信息**，而 sMPC 依赖多步预测，说明 SRL 在信息劣势下仍表现更优

### **消融实验结果**
- **是否进行离线微调（SRL-1 vs SRL-2）**：
  - 加入离线微调后，**在线微调起始性能提升明显**，收敛更快更稳
  - 证明离线微调有效提升了策略对控制目标的对齐度

- **演示数据质量影响**：
  - 随着数据质量下降（NO → LQ），所有方法性能下降，但 **SRL 下降幅度最小**
  - 即使在 LQ 数据下，SRL 仍能通过 RL 微调“纠正”错误行为，体现出强大泛化能力

- **可扩展性分析（Scalability）**：
  - 当 DER 数量从 5 增加到 120：
    - **训练时间**：SRL 比 PPO 多约 14%（小规模）→ 5%（大规模），相对开销随规模增大而减小
    - **成本削减**：在 30 DER 时达峰值（>10% 成本降低），之后因“维数灾难”略有下降
    - 建议未来采用 **multi-agent RL** 分解联合状态-动作空间以缓解此问题

---

## **4. 关键结论和发现**

### **主要发现**
1. **SRL 框架显著优于现有方法**：在样本效率、最终性能和鲁棒性方面全面领先。
2. **监督预训练是高效“warm start”**：避免了 RL 从零探索的低效与不稳定。
3. **两步微调至关重要**：离线微调提升策略质量，在线微调实现现实适配，二者协同作用。
4. **对低质量数据具有强鲁棒性**：即使演示数据严重次优，SRL 仍能通过 RL 微调实现高性能。
5. **经济调度逻辑合理**：可视化显示，SRL 学会了基于电价的充放电策略、TCL 预冷等节能行为。

### **方法的局限性**
- **依赖一定量的历史数据**：若完全无历史操作记录，则无法进行监督预训练。
- **模拟环境需有一定保真度**：离线微调依赖模拟器，若模拟偏差过大，可能影响迁移效果。
- **单智能体架构限制可扩展性**：随着 DER 数量增加，状态-动作空间膨胀导致探索效率下降。
- **未显式处理极端事件**：在线微调虽可适应，但可能产生额外成本；建议在训练中引入极端场景。

### **未来工作方向**
1. **引入 Multi-Agent RL**：将联合决策分解为多个智能体，缓解“维数灾难”。
2. **集成通信机制**：支持智能体间协调，提升大规模系统下的控制效率。
3. **增强对极端事件的鲁棒性**：在演示数据和模拟环境中加入罕见场景，提升抗风险能力。
4. **探索更高效的离线到在线迁移机制**：如 domain randomization 或 sim-to-real adaptation techniques。
5. **应用于真实物理系统验证**：当前为仿真验证，下一步应部署于真实微网进行实测。

---

> **总结**：本文提出的 **SRL 框架**成功将 LLM 中“监督预训练 + RL 微调”的成功范式迁移到 **DER 控制领域**，不仅大幅提升 RL 的训练效率与最终性能，还展现出对现实世界中常见“次优历史数据”的强大容忍能力，为智能电网的自主优化控制提供了极具前景的新路径。

</details>

---

### 16. [A Hybrid CNN-LSTM Intrusion Detection Framework for Cybersecurity in Smart Renewable Energy Grids](https://arxiv.org/abs/2606.25200)

**Authors**: Sajib Debnath, Remon Das  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.25200v1  

#### Abstract
The accelerated digitalization of renewable energy smart grids through IoT sensors, AMI, and SCADA systems has significantly expanded the attack surface for sophisticated cyberattacks, FDI attacks that stealthily distort state estimation and DoS/DDoS attacks that flood communication channels. Curren...

---

### 17. [Distill on a Diet: Efficient Knowledge Distillation via Learnable Data Pruning](https://arxiv.org/abs/2606.25488)

**Authors**: Yifan Wu, Yiqi Wang, Xichen Ye, Wenjing Yan, Xiaoqiang Li, Cheng Jin, Xiangyu Yue, Weizhong Zhang  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.25488v1  

#### Abstract
Knowledge Distillation (KD) is widely used to obtain compact models for efficient inference in resource-constrained environments. Yet the computational overhead of the distillation process itself is often overlooked, raising the question of whether a better student model can be obtained with less da...

---

### 18. [Decentralised AI Training and Inference with BlockTrain](https://arxiv.org/abs/2606.24722)

**Authors**: Peter Toth  
**Category**: cs.AI  
**Published**: 2026-06-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.24722v1  

#### Abstract
Frontier AI training is increasingly shaped by access to dense, centrally controlled accelerator clusters. This creates a structural advantage for hyperscalers and large centralized laboratories, and makes open or independent AI efforts depend on scarce capital, privileged infrastructure, and data-c...

---

### 19. [PolicyAlign: Direct Policy-Based Safety Alignment for Large Language Models](https://arxiv.org/abs/2606.25442)

**Authors**: Chang Wu, Junfeng Fang, Houcheng Jiang, Kai Tang, Pengyu Cheng, Xiaoxi Jiang, Guanjun Jiang, Xiang Wang  
**Category**: cs.CL  
**Published**: 2026-06-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.25442v1  

#### Abstract
Safety alignment of large language models (LLMs) typically depends on high-quality supervision data, such as safe demonstrations or preference pairs. However, in real-world deployment, emerging safety requirements are often specified as natural-language policies, while corresponding supervision data...

---

### 20. [SFL-MTSC: Leveraging Semantic Frame-Level Multi-Task Self-Consistency for Robust Multi-Intent Spoken Language Understanding](https://arxiv.org/abs/2606.25552)

**Authors**: Po-Yen Chen, Berlin Chen  
**Category**: cs.CL  
**Published**: 2026-06-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.25552v1  

#### Abstract
Prompt-based spoken language understanding (SLU) with large language models (LLMs) often suffers from inconsistent intent--slot structures due to decoding stochasticity, particularly in multi-intent scenarios. In view of this, we propose Semantic Frame-Level Multi-Task Self-Consistency (SFL-MTSC), a...

---

### 21. [BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents](https://arxiv.org/abs/2606.25556)

**Authors**: Hanyang Wang, Weijieying Ren, Yuxiang Zhang, Ding Cao, Zhizhao Zeng, Ke Zeng, Tianxiang Zhao  
**Category**: cs.CL  
**Published**: 2026-06-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.25556v1  

#### Abstract
Stepwise group-based RL is an attractive way to train long-horizon LLM agents without a learned critic: it reuses multiple sampled rollouts to estimate local advantages. Its weakness is less visible but more fundamental: every group-relative estimator assumes that the steps it compares are equivalen...

---

### 22. [BitNet Text Embeddings](https://arxiv.org/abs/2606.25674)

**Authors**: Zhen Li, Xin Huang, Liang Wang, Nan Yang, Ting Song, Yan Xia, Xun Wu, Shaohan Huang, Huishuai Zhang, Furu Wei, Dongyan Zhao  
**Category**: cs.CL  
**Published**: 2026-06-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.25674v1  

#### Abstract
LLM-based text embedders have substantially improved retrieval and semantic representation quality, but their deployment remains costly: large backbone models slow down embedding inference, while high-dimensional full-precision embeddings impose substantial storage and bandwidth overhead on large-sc...

---

### 23. [Programmable Probabilistic Computer with 1,000,000 p-bits](https://arxiv.org/abs/2606.25313)

**Authors**: Navid Anjum Aadit, Xiuqi Zhang, Shuvro Chowdhury, Kevin Callahan-Coray, Kyle Lee, Saleh Bunaiyan, Sanjay Seshan, Clayton Thomas, Jason Twigg, Andrew Seawright, Forrest Brewer, Tathagata Srimani, Kerem Y. Camsari  
**Category**: cs.DC  
**Published**: 2026-06-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.25313v1  

#### Abstract
Probabilistic computers built from p-bits have been proposed as hardware accelerators for sampling and optimizing Ising models, but existing systems have been confined to a single chip, capped by its capacity and memory bandwidth. Here we break this limit by networking FPGAs into a single Ising mach...

---

### 24. [Interference-Aware Cross-Application Placement: A Multi-Objective Optimization Approach for Microservice Clusters](https://arxiv.org/abs/2606.25922)

**Authors**: Iqra Zafar, Christian Medeiros Adriano, Holger Giese  
**Category**: cs.DC  
**Published**: 2026-06-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.25922v1  

#### Abstract
In modern cloud architectures, multiple applications often run within the same clustered environment, sharing underlying resources. This resource sharing can cause interference among applications, leading to degraded latency and reduced system stability. As containerized microservices become increas...

---

### 25. [Don't Go Breaking My LLM: The Impact of Pruning Attention Layers on Explanation Faithfulness and Confidence Calibration](https://arxiv.org/abs/2606.24970)

**Authors**: Pietro Tropeano, Maria Maistro, Tuukka Ruotsalo, Christina Lioma  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.24970v1  

#### Abstract
Pruning Large Language Models (LLMs) reduces memory and inference costs by removing parts of the network, producing smaller models that retain most of their accuracy. As attention layers are the most resource-intensive parts of LLMs, pruning them is a promising compression strategy. Prior work shows...

---

### 26. [TRACER: Training-Free Closed-Loop Structured Inference for Traffic Accident Reconstruction](https://arxiv.org/abs/2606.25002)

**Authors**: Yanchen Guan, Chengyue Wang, Bin Rao, Haicheng Liao, Jiaxun Zhang, Shang Gao, Chengzhong Xu, Zhenning Li  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.25002v1  

#### Abstract
Traffic accident reconstruction is a forensic inverse problem that requires recovering physically consistent motion from sparse and heterogeneous evidence. Existing learning-based approaches predominantly optimize for semantic plausibility or visual realism, rather than quantitative agreement with m...

---

### 27. [Speculative Decoding at Temperature Zero: A Scoped Safety-Invariance Screen with a 48,072-Sample Expansion](https://arxiv.org/abs/2606.25097)

**Authors**: Sahil Kadadekar  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.25097v1  

#### Abstract
Speculative decoding accelerates inference by letting a draft model propose tokens for a target model to verify, raising a concrete safety question: at temperature zero, can draft-side behavior leak into safety-scored outputs? We answer with Typical-Acceptance Invariance Screen (TAIS), a behavioral-...

---

### 28. [Constraint Tax in Open-Weight LLMs: An Empirical Study of Tool Calling Suppression Under Structured Output Constraints](https://arxiv.org/abs/2606.25605)

**Authors**: Fangzheng Li, Aimin Zhang, Chen Lv  
**Category**: cs.CL  
**Published**: 2026-06-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.25605v1  

#### Abstract
Tool Calling and Structured Output are two core capabilities of modern Agent systems, yet their interaction under joint deployment conditions remains insufficiently understood. This paper reports a reproducible phenomenon observed in a production Agent system: when Tool Calling and JSON Schema const...

---

### 29. [Endeavor: Efficient PairHMM for Detection of DNA Variants in Genome-Scale Datasets](https://arxiv.org/abs/2606.25738)

**Authors**: Miguel Gra\c{c}a, Aleksandar Ilic  
**Category**: cs.DC  
**Published**: 2026-06-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.25738v1  

#### Abstract
DNA variant calling represents a key operation in bioinformatics pipelines that aims at identifying genetic variants. Given an evidenced explosion in genomic data availability, there is an urgent need for a high-performant, portable and efficient solution for variant calling, which can further impro...

---

### 30. [Towards Continuous Power Forecasting: Practical Continual Learning for Real-World Energy Systems in Nonstationary Time Series](https://arxiv.org/abs/2606.24955)

**Authors**: Yujiang He, Frederic Uhrweiller, Bernhard Sick  
**Category**: cs.LG  
**Published**: 2026-06-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.24955v1  

#### Abstract
Power forecasting models deployed in real-world energy markets must operate under nonstationary conditions, where data distributions continually evolve due to weather variability, infrastructure upgrades, and changing consumption behaviors. In practice, these models face strict operational constrain...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-21 11:23:22 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SpecQuant: Speculative Decoding with Multi-Parent Quantization for Adaptive LLM Inference](https://arxiv.org/abs/2609.21704)

**Authors**: Harish KB, Jagadeeswaran M, Pradheep P, Yuvanesh S, Sivakumar T  
**Category**: cs.LG  
**Published**: 2026-09-21  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2609.21704v1  

#### Abstract
Running large language models (LLMs) locally continues to be limited by restrictions of compute and memory on consumer hardware. The popular acceleration technologies, such as quantization, speculative decoding, and adaptive inferencing, offer substantial speed boosts but usually necessitate retrain...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SpecQuant: Speculative Decoding with Multi-Parent Quantization for Adaptive LLM Inference**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题
当前在消费级硬件上本地部署 **Large Language Models (LLMs)** 面临两大挑战：
- **计算与内存资源受限**：主流 LLMs 参数量大（如 7B 模型需 >14GB 内存），难以在普通设备运行。
- 现有加速技术存在缺陷：
  - **Speculative Decoding**（如 EAGLE、Medusa）依赖独立的 draft model，带来兼容性差、额外存储开销。
  - **Quantization** 虽然降低内存占用，但多为静态应用，缺乏动态适应能力。
  - **Adaptive Inference**（如 early exit）常需模型重训练或架构修改，不适用于即插即用场景。

### 🚀 提出了什么新方法或新思路
提出 **SpecQuant** —— 一种无需训练、支持自适应推理的高效 LLM 推理框架，其核心创新包括：

1. **Multi-Parent Quantization**  
   基于同一个 FP16 全精度模型（Q16），通过 post-training quantization 构建两个轻量化变体：
   - **INT4（Q4）**：极低内存、高速度，适合简单任务。
   - **FP8（Q8）**：中等精度与速度，平衡复杂度。
   - 所有版本共享原始权重，确保 token 分布高度一致，提升 speculative decoding 中的 token acceptance rate。

2. **Complexity-Aware Routing Mechanism**  
   设计一个无需训练的路由机制，根据输入 prompt 的复杂度自动选择合适的 parent model：
   - **Low Complexity → Q4**
   - **Medium Complexity → Q8**
   - **High Complexity → Q16（跳过 speculative decoding）**

   复杂度由三个因素决定：
   - Prompt Length（token 数量）
   - Syntactic Complexity（句法结构深度）
   - Named Entity Density（命名实体密度）

3. **Hardware-Aware Deployment Strategy**  
   支持灵活部署于不同硬件环境：
   - GPU 可用时：优先将 parent 放入 GPU，draft 可放 CPU（hybrid placement）
   - 无 GPU 时：完全 CPU 运行（fallback mode），保证通用性。

4. **Training-Free & Plug-and-Play Design**  
   整个框架基于预训练模型构建，无需微调或重新训练，便于快速部署。

### 🔍 相比现有方法的优势
| 维度 | 现有方法（如 EAGLE、FrugalGPT） | SpecQuant |
|------|-------------------------------|----------|
| 是否需要训练 | 是（draft model 或路由策略） | 否 |
| Draft/Parent 兼容性 | 差（不同架构导致 acceptance 低） | 高（共享权重） |
| 自适应能力 | 弱或仅云环境适用 | 强，本地设备友好 |
| 内存效率 | 较低（需维护多个独立模型） | 高（共享 base weights） |
| 硬件适配性 | 通常依赖 GPU | 支持 CPU-only 和混合部署 |

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集
在以下三个代表性 benchmark 上进行评估，覆盖多种推理类型：
- **MMLU**：考察 factual reasoning 能力（常识、学科知识）
- **Alpaca Eval Subset**：测试 instruction-following 表现
- **GSM8K**：评估 mathematical reasoning（数学推理解题）

### ⚙️ 实验设置
- **Base Model**: Qwen2.5-7B-Instruct（FP16）
- **Quantized Variants**:
  - Q4: INT4 量化（via GPTQ）
  - Q8: FP8 量化
- **Speculative Decoding Setup**:
  - Draft model: 来自同一 base model 的轻量 head（类似 Medusa 结构）
  - Parent models: Q4 / Q8 / Q16（共享 backbone 权重）
- **Routing Logic**:
  - 根据 prompt 的 length、syntax、NER density 判断复杂度等级
- **Hardware Simulation**:
  - 测试 GPU-accelerated 与 CPU-only 模式下的性能表现

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Average Inference Time (s)** | 单条样本生成耗时 |
| **Speedup (%)** | 相对于标准 autoregressive decoding 的加速比 |
| **Accuracy (%)** | 在各 benchmark 上的任务准确率 |
| **Token Acceptance Rate (%)** | speculative tokens 被 parent 接受的比例 |
| **Memory Footprint** | 模型加载所需显存/内存 |

### 🔁 基线方法对比
- **Baseline**: 标准自回归解码（no speculation, no quantization）
- 对比方案隐含包括：
  - 单一量化模型推理（如仅用 INT4）
  - 传统 speculative decoding（如 EAGLE 使用独立 draft）
  - 静态模型切换（无动态路由）

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（来自 Table I & Fig. 1）

| Benchmark | Baseline (s) | SpecQuant (s) | Speedup (%) |
|---------|--------------|----------------|-------------|
| MMLU     | 4.07         | 3.01           | **35.2%**   |
| Alpaca Eval | 6.42       | 4.50           | **42.6%**   |
| GSM8K    | 8.83         | 6.43           | **37.3%**   |
| **Avg**  | —            | —              | **~38.4%**  |

> ✅ **平均实现 38.4% 的推理延迟下降**

### 🎯 准确率与 Token Acceptance（Table II）

| Benchmark | Normal Acc.(%) | SpecQuant Acc.(%) | Δ Accuracy | Acceptance Rate (%) |
|----------|------------------|--------------------|------------|-----------------------|
| MMLU      | 72.5             | 72.3               | -0.2       | **66.1%**             |
| Alpaca Eval | 75.0           | 74.9               | -0.1       | **60.9%**             |
| GSM8K     | 70.0             | 69.9               | -0.1       | **57.9%**             |

> ✅ **精度损失 < 0.2%，可忽略不计**  
> ✅ **Token Acceptance Rate 达到 57.9% ~ 66.1%**，显著高于跨模型 speculative decoding（通常 <50%）

### 🔍 路由有效性分析
- **Low Complexity Prompts (~28%)** → 使用 Q4，获得最高加速（35.2%）
- **Medium Complexity (~52%)** → 使用 Q8，兼顾精度与速度（42.6% 加速）
- **High Complexity (~20%)** → 直接使用 Q16，保障输出质量
- 总体约 **80% 的请求可通过 speculative decoding 加速**

### ❌ 消融实验（文中未明确列出表格，但有定性分析）
- **共享权重 vs 独立 draft model**：
  - 共享权重设计使 token 分布更接近，acceptance rate 显著提高。
- **路由机制有效性**：
  - 若全部使用 Q4，accuracy 下降明显；若全部走 speculative，高复杂任务失败率上升。
  - 动态路由是实现“高效+高质”平衡的关键。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **Quantization 不仅压缩模型，还能增强 speculative decoding 兼容性**  
   当 draft model 是 parent 的量化版本时，二者 token 分布高度对齐，大幅提升 acceptance rate。

2. **Shared-weight + Multi-Precision Parents 是高效的 speculative 架构范式**  
   避免了传统 speculative decoding 中 draft/parent 不匹配的问题，同时节省内存。

3. **无需学习的 complexity-based routing 即可有效分配 workload**  
   仅基于 prompt length、syntax、NER density 就能实现合理分流，无需额外训练。

4. **SpecQuant 实现了真正的 on-device LLM inference 可行性**  
   在无专用 GPU 的设备上也能运行，且性能提升显著，适合边缘计算（Edge Computing）场景。

### ⚠️ 方法的局限性
- **依赖高质量 post-training quantization**：若量化引入过大偏差（如 outlier weights 未处理），会影响 acceptance rate。
- **未支持更细粒度的动态切换**：例如 layer-wise 或 token-level 自适应，仍为 prompt-level routing。
- **目前仅验证于 7B 规模模型**：在更大模型（如 70B）或多模态场景中的扩展性有待验证。
- **routing heuristic 为手工设计**：虽无需训练，但在极端 prompt 类型下可能误判复杂度。

### 🔮 未来工作方向
1. **结合 Learned Router**：引入小型轻量网络自动预测复杂度，进一步优化路由决策。
2. **Support for Larger Models & Modalities**：拓展至 MoE 架构或多模态 LLMs。
3. **Integration with Early Exit / Layer Skipping**：与 AdaInfer 等方法融合，实现多层次自适应推理。
4. **Real-world Edge Device Deployment**：在手机、树莓派等真实终端设备上实测功耗与响应延迟。

---

## ✅ 总结一句话
> **SpecQuant 提出了一种无需训练、基于共享权重的 multi-parent quantization 框架，结合 complexity-aware routing，在保持几乎无损 accuracy 的前提下，实现了平均 38.4% 的推理加速，为消费级硬件上的高效 LLM 部署提供了实用解决方案。**

🔗 开源地址：[https://github.com/HyperKuvid-Labs/SpecQuant](https://github.com/HyperKuvid-Labs/SpecQuant)

</details>

---

### 2. [Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap](https://arxiv.org/abs/2609.21483)

**Authors**: Ziyu Huang, Yangjie Zhou, Chenhao Zhu, Zihan Liu, Jinyu Liu, Shulai Zhang, Xingxun Tang, Hongzhe Yan, Xinhao Luo, Minyi Guo, Xiu Lin, Yinghao Yu, Guodong Yang, Liping Zhang, Shixuan Sun, Jingwen Leng  
**Category**: cs.DC  
**Published**: 2026-09-21  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.21483v1  

#### Abstract
Mixture-of-Experts (MoE) inference under expert parallelism (EP) turns each MoE layer into a distributed computation with costly dispatch and combine communication. State-of-the-art systems reduce this cost through communication-computation overlap, splitting the GPU's SMs for communication and comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Mixture-of-Experts (MoE)** 模型的 **Expert Parallelism (EP)** 推理中，由于 token 需跨 GPU 路由到不同专家进行计算，导致频繁的 **dispatch** 和 **combine** 通信操作。这些通信开销可占 MoE 层延迟的高达 40%，成为性能瓶颈。

现有系统通过 **compute-communication overlap** 来隐藏通信延迟，但在 **SM（Streaming Multiprocessor）调度** 上存在两大资源浪费问题：

- **空间维度浪费（Spatial Waste）**：固定或粗粒度的 SM 分配策略无法适应每层、每 GPU 上因路由结果动态变化而产生的通信/计算负载不均衡。
- **时间维度浪费（Temporal Waste）**：复杂的依赖关系（如 GEMM 必须等 dispatch 完成）导致流水线中出现“气泡”（bubbles），使部分 SM 空闲。

### 🚀 提出的新方法：Weave
Weave 是首个实现 **细粒度动态 SM 调度** 的 MoE 重叠系统，其核心思想是：  
> **在运行时、按层、按 GPU，基于实际路由结果动态决定 SM 分配策略**。

#### 创新设计包括：
- **Persistent Megakernel 架构**：将 dispatch、GEMM0、activation、GEMM1、combine 五阶段融合为单个持久化 kernel，消除 kernel launch 开销并提供统一调度域。
- **轻量级成本模型（Lightweight Cost Model）**：在 megakernel 启动阶段运行，结合当前层的路由信息与硬件吞吐曲线，快速决策最优调度方案。
- **双维度联合调度器**：
  - **Spatial Scheduler（空间调度器）**：决定多少 SM 分配给通信任务（dispatch/combine），以平衡通信带宽与计算吞吐。
  - **Temporal Scheduler（时间调度器）**：通过 **chunk pipelining** 和 **bubble stealing** 减少 SM 空闲时间，提升利用率。

### 🔍 相比现有方法的优势
| 方法 | 缺陷 | Weave 的改进 |
|------|------|-------------|
| **Triton-Distributed (TD)** | 无显式 SM 分区，串行执行，重叠极少 | 显式分区 + 动态协调，显著提升重叠率 |
| **DeepEP / ParallelKittens (PK)** | 静态 SM 分区（编译时固定） | 运行时动态调整，适配每层负载 |
| **Comet** | 每迭代一次调整 SM 数量（粗粒度） | 每层、每 GPU 细粒度调整，更精准 |

> ✅ Weave 实现了真正的 **routing-aware**、**per-layer & per-GPU** 的 SM 资源调度，最大化 GPU 利用率。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **ShareGPT**：用于生成推理输入 prompt。
- **CodeSearchNet**：用于分析路由分布对性能的影响（如图5）。

### ⚙️ 实验设置
- **硬件平台**：4×NVIDIA H100 80GB SXM5 GPU，全连接 NVLink（450 GB/s 单向带宽），每 GPU 132 SMs。
- **软件环境**：
  - CUDA 12.9.86, NCCL 2.21.5, PyTorch 2.6.0
  - 实现基于 **ParallelKittens (PK)** 修改构建
- **测试模型**：6 个主流 BF16 MoE 模型（见下表）
- **序列长度**：2048, 4096, 8192（prefill 阶段，batch size=1）
- **并行方式**：Expert Parallelism (EP=4)

#### 表：评估的 MoE 模型配置
| Model | E（专家数） | top-k | H（隐维） | I（中间维） |
|-------|-------------|--------|----------|------------|
| DSv3 | 256 | 8 | 7168 | 2048 |
| Phi-3.5-MoE | 16 | 2 | 4096 | 6400 |
| Qwen3-30B | 128 | 8 | 2048 | 768 |
| Qwen3.5-35B | 256 | 8 | 2048 | 512 |
| DSv2-Lite | 64 | 6 | 2048 | 1408 |
| DSv2 | 160 | 6 | 5120 | 1536 |

### 🎯 评估指标
- **Per-layer MoE latency**：单个 MoE 层平均延迟
- **End-to-End (E2E) latency**：完整推理延迟（含 attention + MoE）
- **SM Active Rate**：SM 活跃比例
- **NVLink Utilization / Tensor Core HMMA**：通信与计算资源利用率
- **Comm-Comp Overlap Ratio**：通信与计算同时活跃的时间占比
- **Speedup**：相对于基线的加速比（几何平均）

### 🆚 基线方法对比
共比较 **5 个 state-of-the-art 基线**：
1. **SGLang**：生产级服务系统，使用 NCCL All-to-All + cuBLAS grouped GEMM，无 kernel 内重叠。
2. **Triton-Distributed (TD)**：融合 dispatch+GEMM0 和 GEMM1+combine，依赖全局抢占实现有限重叠。
3. **DeepEP**：专家并行通信库，固定分配 20 SMs 给通信任务。
4. **Comet**：支持 tile-level 重叠，可根据序列长度每轮调整 SM 数量。
5. **ParallelKittens (PK)**：compile-time 固定 SM 分区，融合 dispatch 与 up-projection。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据
| 指标 | Weave 结果 | 提升幅度 |
|------|-----------|---------|
| **MoE-layer Speedup (几何平均)** | **2.89×** | vs 所有基线 |
| **End-to-End Speedup (几何平均)** | **1.33×** | vs 所有基线 |
| 最高单模型加速比 | **4.76×** | vs PK（DSv2-Lite） |
| 最低 E2E 加速比 | **1.12×** | vs DeepEP |

#### 详细对比（图10）：
- 在所有 **18 种配置**（6模型 × 3序列长）中，Weave 均取得最低延迟。
- 对比各基线的 MoE 层加速比：
  - vs DeepEP: **1.95×**
  - vs Comet: **2.01×**
  - vs TD: **2.95×**
  - vs SGLang: **3.63×**
  - vs PK: **4.76×**

### 🔬 消融实验结果（Ablation Study）
在 Qwen3-30B 和 Qwen3.5-MoE 上进行消融实验，验证两个调度器的作用：

| 配置 | 描述 | 相对性能 |
|------|------|----------|
| **Weave w/o S+T** | 无任何调度，串行执行 | 基准 |
| **Weave w/o T** | 仅启用空间调度（S） | 性能显著提升 |
| **Weave (full)** | 完整双调度器 | 进一步提升 |

#### 典型结果（Qwen3-30B, seq=4096）：
- 启用 **Spatial Scheduler (S)**：降低延迟 **14.4%**
- 再启用 **Temporal Scheduler (T)**：额外再降 **15.7%**
- 总计减少延迟 **27.8%**

> ✅ 两个调度器效果叠加，且都至关重要。

### 📈 硬件利用率分析（Nsight Profiling）
在 DSv2-Lite 上的 profiling 显示 Weave 显著优于其他系统：

| 指标 | Weave | Comet | DeepEP | PK | TD |
|------|-------|--------|--------|-----|-----|
| **SM Active (%)** | **91%** | 47.4% | ~25% | ~23% | ~23% |
| **Comm-Comp Overlap (%)** | **47.1%** | 14.3% | 9.2% | 5.8% | 5.1% |
| **NVLink Utilization** | 高且持续 | 短暂脉冲 | 不足 | 极低 | 极低 |

> 💡 Weave 成功实现了高 SM 利用率与高重叠率的双重目标，而其他系统只能做到其一。

### ✅ 成本模型准确性与开销
- **选择准确率**：成本模型选出的 `(c*, K*)` 配置，实测延迟与穷举最优仅差 **8.2%**。
- **在线开销极低**：在 DSv3 上仅为 **0.54 μs**，不足 MoE 层总时间的 **0.021%**，几乎可忽略。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MoE 推理中的 SM 分配必须是动态且细粒度的**：静态或粗粒度策略无法匹配每层、每 GPU 的真实负载，造成严重资源浪费。
2. **空间与时间调度需联合优化**：仅优化 SM 分配（Spatial）不足以消除“中段气泡”，还需通过 chunk pipelining 和 bubble stealing 实现时间维度上的高效协作。
3. **Persistent Megakernel + 轻量成本模型 是可行路径**：可在微秒级完成运行时调度决策，带来巨大收益而几乎无额外开销。
4. **Weave 实现了前所未有的硬件利用率**：在多个维度上全面领先，尤其在 SM Active 和 Comm-Comp Overlap 上达到新高度。

### ⚠️ 局限性
- 当前仅在 **单节点 4×H100 NVLink 架构** 上验证。
- 支持的 EP 规模较小（EP=4），尚未扩展至更大规模或多节点场景。
- 成本模型依赖预校准的硬件吞吐曲线，在异构环境中可能需要重新调优。

### 🔮 未来工作方向
- 扩展至 **多节点、大规模 EP** 场景（如 EP > 8）。
- 支持 **异构 GPU 集群** 下的自适应调度。
- 将 Weave 思路推广至 **MoE training** 场景。
- 探索更智能的成本建模方法（如引入 ML 预测）。

---

## 总结
Weave 提出了一种全新的 **fine-grained dynamic SM scheduling** 范式，首次实现了 **per-layer、per-GPU、routing-aware** 的 MoE 通信-计算重叠优化。通过融合 spatial 与 temporal 双调度器，并嵌入轻量成本模型于 persistent megakernel 中，Weave 在保持极低运行时开销的同时，取得了高达 **2.89× 的 MoE 层加速** 和 **1.33× 的端到端加速**，显著超越现有 SOTA 系统，为高效 MoE 推理提供了新的设计范式。

</details>

---

### 3. [Elastic Threshold Attention: Learned Contextual Sparsity for Long-Context Decoding](https://arxiv.org/abs/2609.20888)

**Authors**: Themistoklis Haris, Henry Li, Maryam Karimzadehgan  
**Category**: cs.LG  
**Published**: 2026-09-21  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.20888v1  

#### Abstract
Massive KV caches can cause severe memory-bandwidth bottlenecks during long-context decoding. Sparse attention methods mitigate this via selective loading, but that comes at a cost: rigid heuristics drop necessary context, leading to quality degradation. We introduce \textbf{Elastic Threshold Attent...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Elastic Threshold Attention: Learned Contextual Sparsity for Long-Context Decoding

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在长上下文解码中，大规模的 **KV Cache** 会引发严重的内存带宽瓶颈，导致推理延迟高、吞吐量低。现有的稀疏注意力方法（如 H2O、StreamingLLM）通过启发式策略删除历史 token，虽然减少了内存访问，但往往因**刚性预算**（rigid token budget）而丢失关键上下文，造成模型质量显著下降。

此外，许多可训练的稀疏架构（如 NSA、SeerAttention）依赖复杂的多分支设计或训练期间的硬剪枝（hard top-k），难以部署且对硬件不友好。

### 提出的新方法：Elastic Threshold Attention (ETA)
ETA 是一种端到端可训练的稀疏注意力机制，其核心思想是：

- **动态阈值预测**：每个查询 `q` 通过一个轻量级线性投影（每层仅增加 <0.1% 参数）生成**动态、上下文感知的标量阈值** `t`，用于决定哪些 key 应被保留。
- **乘法门控抑制（Multiplicative Suppression）**：不同于传统方法将未选中的 token 的 logit 设为 `-∞`（即硬删除），ETA 使用 sigmoid 门控将低于阈值的得分**乘性收缩至 0**，形成一个平滑的均匀注意力底座（uniform attention floor）。
- **硬件对齐的推理内核**：设计了一个基于 Triton 的 fused decode kernel，在 O(1) 时间内利用缓存的块统计信息（质心、方差、最大范数）筛选非显著 KV 块，避免加载无关数据。

### 相比现有方法的优势
| 维度 | ETA | 传统稀疏方法（如 H2O、BigBird） | 可训练稀疏方法（如 NSA） |
|------|-----|-------------------------------|------------------------|
| **质量保持** | ✅ 接近 dense model 质量 | ❌ 因刚性预算导致推理退化 | ⚠️ 复杂架构可能不稳定 |
| **训练稳定性** | ✅ 乘法抑制防止表示坍缩 | — | ❌ 硬删除导致分布脆弱 |
| **硬件效率** | ✅ O(1) 块筛选，支持 block-sparse | ✅ 启发式高效 | ⚠️ 多分支带来额外开销 |
| **灵活性** | ✅ 支持动态阈值 + 静态校准 | ❌ 固定规则 | ❌ 固定模式 |

---

## 2. 核心实验方法和设置

### 数据集
- **预训练数据**：FineWeb（约 42B tokens）
- **评估数据集**：
  - **语言建模**：WikiText-2、FineWeb、C4
  - **常识推理**：ARC-Easy、HellaSwag
  - **长上下文检索**：Needle-in-a-Haystack（上下文长度达 8192）

### 实验设置
- **模型规模**：1.45B 参数（22 层 LLaMA 架构），以及 126M 小模型用于消融分析
- **上下文长度**：训练时 L=2048，测试时扩展至 512K
- **硬件平台**：NVIDIA H100/A100 GPU，使用 float16 精度
- **评估指标**：
  - **Perplexity (PPL)**：语言建模能力
  - **Accuracy (%)**：零样本推理任务表现
  - **Decode Latency / Speedup**：单步解码延迟，相对于 FlashAttention-2 的加速比
  - **Active Density**：实际参与 attention 的 token 比例（衡量稀疏性）

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Dense SDPA / FlashAttention-2** | 密集注意力 | 性能上限基准 |
| **H2O** | Heuristic Eviction | 动态追踪重要 token，保留 sink |
| **StreamingLLM** | Heuristic Eviction | 显式保留前几个 sink tokens |
| **BigBird / SWA** | Fixed Window | 固定局部窗口注意力 |
| **NSA (Native Sparse Attention)** | 可训练稀疏 | 多分支结构（sliding + coarse + fine） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）语言建模与推理性能（1.45B 模型）
| 方法 | FineWeb PPL | C4 PPL | ARC-Easy Acc | HellaSwag Acc | 平均密度 |
|------|-------------|--------|--------------|---------------|----------|
| Dense SDPA | 14.70 | 18.96 | 44.0% | 36.0% | 100% |
| **ETA (ours)** | **14.83** | **18.99** | **44.5%** | **40.0%** | **~38%** |

> ✅ ETA 在仅使用 ~38% 的 token 密度下，实现了与 dense model 相当的语言建模性能，并在 HellaSwag 上表现出更强的“注意力去噪”效果。

#### （2）长上下文检索（Needle-in-a-Haystack）
| 方法 | L=1024 Acc | L=2048 Acc | L=4096 Acc | L=8192 Acc |
|------|-----------|-----------|-----------|-----------|
| Dense SDPA | 100.0% | 88.9% | 66.7% | **0.0%** |
| H2O / SWA | 22.2% | 11.1% | 0.0% | 0.0% |
| **ETA (ours)** | **100.0%** | **88.9%** | **66.7%** | **22.2%** |

> ✅ ETA 在超长序列（8192）仍保持 22.2% 的检索准确率，远优于所有基线。其动态扩展机制可在需要时自动提升活跃密度（从 30.1% → 69.3%）以保留关键信息。

#### （3）解码速度提升（vs. FlashAttention-2）
在 **512K 序列长度、batch=64** 下：
- 在 **~38% 活跃密度**下，达到 **1.12×** 速度提升；
- 在更稀疏配置下（10% head density），最高实现 **2.5× wall-clock speedup**。

> 💡 速度优势随 batch size 和序列长度增长而增强，表明其有效缓解了 memory-bound 问题。

#### （4）与 NSA 的对比（126M 模型）
| 指标 | NSA | ETA | 差距 |
|------|-----|-----|------|
| 预训练损失 | 4.508 | **4.370** | ↓ 0.138 |
| FineWeb PPL | 103.55 | **99.80** | ↓ 3.75 |
| 解码延迟 @ L=8K, B=32 | 69.07 ms | **25.35 ms** | ↓ 2.72× |
| 吞吐量 | 463 tok/s | **1,262 tok/s** | ↑ 2.72× |

> ✅ ETA 不仅训练更快、收敛更好，且因单通路 fused kernel 实现显著更低的推理延迟。

---

### 消融实验结果

#### （1）乘法抑制 vs 加法掩码（Multiplicative vs Additive）
| 方法 | FineWeb PPL | Compute Density | GQA Union Sparsity |
|------|-------------|----------------|--------------------|
| Multiplicative (ETA) | **99.80** | **34.09%** | **49.23%** |
| Additive (Hard Mask) | 116.38 (+16.6%) | 62.74% | 23.59% |

> 🔍 加法掩码导致分布脆弱，即使密度更高也无法恢复质量，验证了“均匀底座”的必要性。

#### （2）动态阈值 vs 静态阈值
- 移除 query conditioning 后，模型无法区分简单与复杂 token，最终活跃密度高出 **2.13×**。
- 回归分析显示，仅 **32.7%** 的阈值变化可用表面统计量解释，说明 ETA 学到了真正的语义上下文。

#### （3）静态阈值离线校准（Offline Calibration）
- 在目标领域进行校准后，静态阈值可匹配动态模型的困惑度（91.60 vs 91.96），同时减少 **27% 的 attention compute**。
- 在外推长度（L=2048）下，静态阈值还能抑制因 RoPE 旋转引起的密度漂移。

---

## 4. 关键结论和发现

### 主要发现
1. **乘法抑制优于硬删除**：将 sub-threshold logits 收缩至 0（而非 -∞）能建立一个**均匀注意力底座**，既消除 attention sinks，又使模型对推理时的块级剪枝和过包含具有鲁棒性。
2. **上下文感知稀疏是可行的**：通过 query-conditioned 动态阈值，模型可自主调节注意力预算，在简单步骤压缩上下文，在复杂推理时扩展视野。
3. **算法稀疏 ≠ 实际加速**：必须结合硬件对齐的 block-sparse kernel 才能将理论稀疏转化为 wall-clock speedup。ETA 的 dual screening index 实现了 O(1) 块过滤。
4. **无需显式 sink pinning**：由于均匀底座天然承担“无操作”概率分配，ETA 自动消除了对初始 sink tokens 的依赖，简化了缓存管理。

### 方法的局限性
- 当前实现依赖 Triton 编程，对开发者有一定门槛。
- 在极短序列或高密度场景下，加速收益有限。
- 离线校准仅适用于分布稳定的特定领域，通用性受限。

### 未来工作方向
1. **Scaling to Frontier Models**：将 ETA 扩展到 7B–70B+ 规模的模型进行端到端预训练。
2. **跨模态与代码生成应用**：探索其在 multimodal sequences 和 repository-scale code generation 中的表现。
3. **下一代 kernel co-design**：
   - 集成 Hopper TMA 异步流水线
   - 支持 FP8/INT4 KV quantization
   - 开发 block-sparse prefill 加速
   - 利用 warp specialization 进一步优化 SM 占用率

--- 

> 📌 **总结一句话**：  
> **Elastic Threshold Attention (ETA)** 通过“乘法抑制 + 动态阈值 + 硬件对齐内核”的三位一体设计，在几乎不牺牲 dense model 质量的前提下，实现了高达 **2.5× 的解码加速**，为长上下文大模型的高效部署提供了新范式。

</details>

---

### 4. [RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding](https://arxiv.org/abs/2609.21827)

**Authors**: Qiao Hu, Yepeng Weng, Bo Zhang, Takehisa Yairi  
**Category**: cs.CL  
**Published**: 2026-09-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.21827v1  

#### Abstract
Speculative decoding accelerates LLM inference by drafting multiple tokens in parallel, with tree-based methods further improving efficiency through hierarchical structures. Dynamic-tree methods such as EAGLE-3 perform well under greedy decoding via deterministic top-K expansion and global pruning. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Dynamic-Tree Speculative Decoding**（如 EAGLE-2/3）中，当前主流方法依赖 **deterministic top-K 扩展** 和全局剪枝来构建 draft tree，这在 **greedy decoding (T=0)** 下表现优异。然而，在 **stochastic decoding (T>0)** 场景下，该机制会将 draft 分布退化为 one-hot 概率，导致：
- **严重降低 acceptance rate**
- **无法探索长尾分布**
- **牺牲了随机采样的多样性**

这形成了一个“两难困境”：
- **Dynamic-tree 方法**：保持上下文感知的拓扑结构，但牺牲随机性；
- **Static-tree 方法**：保留随机采样，但结构是上下文无关的。

根本原因在于：**同一个概率分布 $q(x)$ 被同时用于树构造和 token 验证**，二者角色耦合，难以直接引入随机性。

---

### 提出了什么新方法或新思路
本文提出 **RheoSampling**（Rheostat Sampling），通过 **dual-identity decoupling** 解决上述困境：

#### 核心思想：解耦两个角色
- 给一个 **stochastically sampled token** 分配两个独立的概率身份：
  - **Proxy probability**：用于树的扩展、重排序和剪枝（construction）
  - **True sampling probability**：用于最终验证（verification）

#### 具体实现机制
1. **混合采样机制（Hybrid Sampling）**：
   - 在每个扩展步骤中，候选池包含：
     - **Top-m deterministic tokens**（最高概率的 m 个 token）
     - **One stochastically sampled token $x_s$** 来自残差分布 $q$
     - **K−m−1 fill tokens**（剩余最高排名的 token）

2. **对采样 token 的双重处理**：
   - **Construction 阶段**：赋予其 proxy probability  
     $$
     q_{\text{proxy}}(x_s) = \min\{q(x_m), z\}, \quad z = 1 - \sum_{i=1}^m q(x_i)
     $$
     确保它在候选池中的排名固定（通常高于 fill tokens），从而使其生存不依赖于自身身份。
   - **Verification 阶段**：使用真实的 $q(x_s)$ 进行验证，保证 losslessness。

3. **Rheostat 参数 $m$**：
   - 控制 deterministic 与 stochastic 成分之间的权衡：
     - 小 $m$ → 更高 proxy 概率 → 更可能存活，利于探索
     - 大 $m$ → 更强的确定性主干，但采样 token 易被剪枝

4. **等价类分析（Equivalence-Class Analysis）**：
   - 首次为 **stochastic dynamic tree** 提供严格的 **losslessness 证明**
   - 将复杂的随机过程压缩为可管理的等价类空间

5. **高效算法设计**：
   - **Optimal Transport (OT)-based verification**（RheoVerification）：提升 acceptance rate
   - **Sparse draft distribution**：仅保留 top-128 logits，减少计算开销而不影响性能

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **理论保障** | 首个兼具 context-aware topology 与 stochastic sampling 的动态树方法，并严格证明 losslessness |
| **灵活性** | 支持标准随机解码（T > 0），适用于多样化的生成任务 |
| **效率** | OT-based verification 提升 acceptance rate；sparse draft 减少延迟 |
| **通用性** | 可作为分析其他复杂随机树结构的模板 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共六个多样化基准，覆盖多种任务类型：
- **Alpaca**：指令跟随
- **GSM8K**：数学推理
- **HumanEval**：代码生成
- **MT-bench**：多轮对话质量评估
- **Natural Questions**：问答
- **CNN/DailyMail**：摘要生成

每数据集包含 80 个问题，确保全面评估。

---

### 实验设置和评估指标

#### 目标模型（Target Models）
- `Llama-3.1-8B-Instruct` (L31-8B)
- `Vicuna-13B-v1.3` (V-13B)
- `DeepSeek-R1-Distill-Llama-8B` (DSL-8B)

#### Draft 模型
- 使用官方发布的 **EAGLE-3 checkpoints**，未进一步微调

#### 关键参数
- 解码树大小：60
- Draft depth：8
- 温度：默认 $T=1.0$
- 硬件：单张 NVIDIA A6000 GPU
- 多次运行取均值（3 次不同 seed）

---

### 评估指标
1. **平均接受长度（Average Acceptance Length, $\bar{T}$）**  
   每轮 drafting-verification 周期中平均接受的 token 数量 → 衡量 speculative 效率的核心指标。

2. **端到端加速比（End-to-End Speedup）**  
   相对于 autoregressive decoding 的 wall-clock 时间减少比例。

---

### 基线方法对比
- **Top-K Baseline**：原始 EAGLE-3 的纯 deterministic top-K 扩展
- **RheoSampling (ours)**：提出的混合采样 + dual-identity 方法
- 消融变体：
  - 不同 $m$ 值（$m=0,1,2,\dots$）
  - 不同 verification 策略（RRSw vs. RheoVerification）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| Model | 方法 | 平均 $\bar{T}$ | 加速比 |
|-------|------|----------------|--------|
| V-13B | Top-K | 5.75 ± 0.03 | 3.37× |
| V-13B | **Rheo** | **5.89 ± 0.03** (+0.14) | **3.43×** |
| L31-8B | Top-K | 5.04 ± 0.02 | 2.84× |
| L31-8B | **Rheo** | **5.25 ± 0.03** (+0.21) | **2.93×** |
| DSL-8B | Top-K | 4.99 ± 0.01 | 2.89× |
| DSL-8B | **Rheo** | **5.21 ± 0.02** (+0.22) | **2.99×** |

✅ **所有配置下均显著优于 Top-K 基线**

---

### 与基线方法的对比结果
- **Acceptance Length 提升**：+0.14 ~ +0.22，提升幅度达 **4.2%**（L31-8B）
- **Speedup 提升**：尽管有额外采样开销，仍带来 **净正增益**（+3.2% ~ +3.5%）
- **跨任务鲁棒性强**：在数学、代码、对话等任务上均有稳定收益

---

### 消融实验结果

#### （1）Rheostat 参数 $m$ 影响（Figure 3）
- **最佳性能出现在 $m=1$**
- $m=0$ 和 $m=2$ 仍优于 Top-K，但稍弱
- $m \geq 3$ 时 proxy 概率过低，采样 token 易被剪枝，增益消失

> ✅ **$m=1$ 是 sweet spot**：平衡了结构质量和随机探索能力

#### （2）Verification 策略比较（Table 3 & 4）

| 方法 | Acceptance Rate（MT-bench, controlled） |
|------|----------------------------------------|
| Top-K + Vanilla | 80.0% |
| Rheo + RRSw | 83.5% |
| **Rheo + RheoVerification (OT)** | **85.4%** |

- **+3.5% 来自 hybrid sampling**
- **+1.9% 额外来自 OT-based verification**

> ✅ **RheoVerification 显著优于 RRSw**，验证了 OT 分配的有效性

#### （3）Sparse Draft Ablation（Table 5）
- 使用 top-128 截断后：
  - **Draft coverage > 91%**
  - **Acceptance length 几乎无损**
  - **Drafting latency 仅增加 ~0.2ms**
- 相比 full vocabulary 实现接近最优性能，且更高效

> ✅ **Sparse draft 是实用且高效的工程优化**

---

## 4. 关键结论和发现

### 主要发现
1. **Dual-identity decoupling 成功解决了 one-hot dilemma**：
   - 首次实现了 **context-aware dynamic tree** 与 **stochastic sampling** 的兼容
   - 通过 proxy probability 解耦 construction 与 verification 角色

2. **Losslessness 得到严格证明**：
   - 利用 **equivalence-class analysis** 压缩复杂随机空间
   - 为后续研究提供理论框架

3. **OT-based verification 显著提升 acceptance rate**：
   - 充分利用了随机 token 带来的分布灵活性
   - 实现 per-layer acceptance rate 接近理论上限

4. **RheoSampling 在真实场景中有效提速**：
   - 即使考虑额外开销，也能带来 **一致的 wall-clock speedup**
   - 对温度变化具有鲁棒性（见 Figure 4）

---

### 方法的局限性
1. **仅插入一个随机 token**：
   - 当前设计限制了对长尾区域的探索广度
   - 多随机 token 扩展尚未探索

2. **Proxy probability 设计敏感**：
   - 必须满足 $q_{\text{proxy}} \geq q(x_{m+1})$ 才能保证 losslessness
   - 不当设计会导致条件分布偏移

3. **依赖高质量 draft model**：
   - 性能增益依赖于 draft model 与 target model 的 alignment

4. **实现复杂度较高**：
   - 相比简单 top-K，需维护 dual probability 和 OT verification 逻辑

---

### 未来工作方向
1. **扩展至 multiple stochastic probes**
2. **自适应选择 $m$ 或 proxy probability**
3. **结合硬件优化（如 Sequoia）进行系统级部署**
4. **应用于非语言模态的 speculative decoding**
5. **探索更多基于 OT 的 verification 策略**

---

> 🔚 **总结**：  
> **RheoSampling 是首个成功融合 dynamic tree 结构优势与 stochastic sampling 灵活性的方法**，不仅解决了长期存在的 one-hot dilemma，还提供了坚实的理论基础和实际性能提升，有望成为下一代 speculative decoding 的标准范式之一。

</details>

---

### 5. [RBS-Attention: Radius-Bounded Sparse Prefill for Long-Context Large Language Models](https://arxiv.org/abs/2609.20971)

**Authors**: Chuxu Song, Jiuqi Wei, Zhencan Peng  
**Category**: cs.AI  
**Published**: 2026-09-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.20971v1  

#### Abstract
Long-context large language model inference is increasingly limited by prefill, where dense self-attention processes the entire prompt before generation begins. Sparse block selection can reduce this cost, but a block centroid may hide a highly relevant token among many irrelevant ones. We call this...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RBS-Attention: Radius-Bounded Sparse Prefill for Long-Context Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在长上下文大语言模型（LLM）推理中，**prefill 阶段的计算成本已成为主要瓶颈**。传统 dense self-attention 在处理数十万 token 上下文时面临 $O(N^2)$ 的计算复杂度，导致延迟极高。

现有稀疏化方法（如基于 block centroid 的选择）存在一个关键缺陷：当一个关键 token 被大量无关 token 包围时，其所在 key block 的平均向量（centroid）可能无法准确反映该 token 的高相关性，从而被错误地剪枝。作者将此现象称为 **mean dilution（均值稀释）**。

### 提出了什么新方法或新思路
提出 **RBS-Attention**，一种无需训练的稀疏 prefill 方法，通过双分支选择机制缓解 mean dilution 问题：

- **Base Branch（基础分支）**：沿用传统的 centroid 得分（$q^\top c_b$），捕捉块级平均相关性。
- **Rescue Branch（救援分支）**：引入 **最大 key-block 半径（maximum key-block radius）** 及其分布自适应的缩放系数 $\beta_b$，构建风险信号：
  $$
  l_{\text{rescue}}(q, b) = q^\top c_b + \|q\|_2 \cdot r_b \cdot \beta_b
  $$
  其中 $\beta_b$ 基于当前 prompt、layer 和 head 的半径分布动态调整（使用 median 和 90th percentile 归一化），使高度离散的 block 更容易被“救援”。

两个分支独立进行相对阈值筛选（relative thresholding），最终通过 **mask union** 合并选中的 key blocks，并保留 sink、local window 和 recent blocks。

### 相比现有方法的优势
- **更鲁棒的选择机制**：避免因 block 内部异质性导致的关键信息丢失。
- **无需训练**：完全基于运行时统计信息，部署简单。
- **兼容性强**：可无缝集成到 block-sparse FlashAttention 中，保持 GPU 执行效率。
- **内容自适应密度**：实际保留的 block 密度随输入内容变化，而非固定比例。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **RULER**：评估长上下文检索与推理能力，涵盖多种长度（4K–128K）任务。
- **LongBench-v2**：多任务基准，包含短、中、长三种长度的任务，共 503 个样本。
- **InfiniteBench**：扩展至超长文本（>100K tokens）的评测集，选取代表性任务如 Ret.KV、En.Dia 等。
- **Video-MME**：多模态视频理解基准，测试 1,000 个视频，每视频 32 帧。

### 实验设置和评估指标
- **硬件平台**：
  - 主要性能测试：**H100 GPU**
  - 补充分析：A100 GPU（TP=2 或 TP=4）
- **模型**：
  - Qwen3-30B-A3B-Instruct-2507-FP8（MoE 架构）
  - Qwen3-32B（dense 架构）
  - Qwen3-VL-30B-A3B-Thinking-FP8（多模态）
- **评估指标**：
  - **系统性能**：
    - Standalone prefill-attention speedup
    - vLLM prefill-attention speedup
    - End-to-end Time-To-First-Token (TTFT) speedup
  - **质量指标**：
    - RULER 准确率（%）
    - LongBench-v2 总体得分（unit interval）
    - InfiniteBench 宏平均得分
    - Video-MME 总体得分（%）
  - **稀疏度测量**：
    - Actual density：实际保留的因果 block 对比例

### 基线方法对比
- **Sparse Prefill Baselines**：
  - FlashPrefill：基于 centroid 和相对阈值的 block 选择
  - FlexPrefill：上下文感知稀疏注意力
  - MInference：动态稀疏注意力
  - XAttn：反向对角线评分的 block 稀疏注意力
- **控制变量比较**（消融实验）：
  - Centroid-only：仅使用 centroid 得分
  - Full-L2：使用完整 L2 上界作为得分
  - Quest-style：基于坐标最小/最大值的查询感知页检索风格选择

---

## 3. 主要实验结果和性能指标

### 关键性能数据（H100，128K context）
| 指标 | RBS-Attention | 最佳基线 |
|------|----------------|----------|
| Standalone prefill-attention speedup | **20.65×** | 11.98× (FlashPrefill) |
| vLLM prefill-attention speedup | **11.92×** | — |
| End-to-end TTFT speedup | **5.97×** | — |

> 在 256K 上下文下，TTFT 加速比进一步提升至约 7×。

### 与基线方法的质量对比
#### 在 Qwen3-32B 上的 RULER 整体准确率：
| 方法 | 准确率 (%) |
|------|-----------|
| Dense | 89.52 |
| MInference | 88.63 |
| **RBS-Attention** | **88.65** |

> RBS 在仅保留 **6.028%** 的 block 对的情况下，在 128K 长度上达到 80.57% 准确率（Dense 为 80.70%），几乎无损。

#### 在 LongBench-v2 上的表现：
| 模型 | 方法 | 总体得分 | 估计密度 (%) |
|------|------|--------|------------|
| Qwen3-32B | Dense | 0.394 | 100 |
| | RBS-Attention | **0.376** | **9.719** |
| | XAttn | 0.376 | 29.197 |
| Qwen3-30B-A3B | Dense | 0.388 | 100 |
| | RBS-Attention | **0.370** | **7.912** |

> RBS 以更低的密度实现了与更高密度方法相当甚至更优的性能。

#### InfiniteBench 宏平均得分（Qwen3-32B）：
| 方法 | Macro Average |
|------|---------------|
| Dense | 0.372 |
| FlashPrefill | 0.338 |
| XAttn | 0.352 |
| **RBS-Attention** | **0.362** |

#### Video-MME 多模态表现（Qwen3-VL-30B-A3B）：
| 方法 | 总体得分 (%) |
|------|-------------|
| Dense | 66.16 |
| FlashPrefill | 65.26 |
| XAttn | 65.76 |
| **RBS-Attention** | **65.98** |

> RBS 是所有稀疏方法中得分最高的，尤其在长视频（Long）任务上表现突出（62.1 vs XAttn 的 61.2）。

### 消融实验结果
#### 控制实际密度为 ~5.34% 时的 RULER-128K 准确率（130 示例）：
| Selector | Accuracy (%) |
|---------|--------------|
| Centroid-only | 73.63 |
| Full-L2 bound | 72.41 |
| Quest-style | 73.71 |
| **RBS adaptive union** | **76.67** |

> 尽管未达统计显著性，但趋势表明 **adaptive rescue + independent thresholding + union** 的设计优于单一几何边界或静态策略。

#### 不同 block size 的敏感性分析（固定 ~5.34% 密度）：
| Block Size | RBS Accuracy (%) |
|-----------|------------------|
| 64 | 71.86 |
| **128** | **80.28** |
| 256 | 70.73 |

> 表明 **B=128** 是当前设置下的最优粒度。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Mean dilution 是长上下文稀疏 prefill 的关键失效模式**：高分散 block 中的关键 token 易被 centroid 低估而丢弃。
2. **Block radius 是有效的风险信号**：高 radius block 更可能包含被低估的重要 token；前 5% 的 attention block 中有 **39.5% 属于最高 radius quintile**。
3. **Radius-adaptive rescue 分支显著提升选择质量**：结合 base 和 rescue 分支可在极低密度下保留更多关键信息。
4. **双分支独立阈值 + union 设计至关重要**：防止 radius uplift 干扰原有高相关 block 的保留。
5. **RBS-Attention 实现了卓越的加速比与质量平衡**：在 128K 上实现近 **6× 端到端 TTFT 加速**，同时保持接近 dense attention 的准确率。

### 方法的局限性
- **主要适用于长上下文 prefill**：在短上下文下 selector 开销可能超过收益。
- **thresholds 需要校准**：虽然运行时自适应，但全局阈值仍需离线调优。
- **不改变 worst-case 复杂度**：最坏情况下仍为 $O(N^2)$。
- **decode 阶段未优化**：当前工作聚焦于 prefill，decode 仍依赖标准 KV-cache。
- **内存占用分析显示临时 workspace 较大**：尽管持久化元数据小（仅 +3%），但中间计算空间可达 1GB 以上。

### 未来工作方向
- 探索 radius 信号在 **decode-time KV-cache retrieval** 中的应用。
- 设计 **自动 threshold tuning** 机制，减少人工配置。
- 结合 **pattern-based 方法**（如 local/window）进一步提升选择效率。
- 研究更高效的 **block statistic 存储与计算方案**，降低临时内存开销。
- 将 RBS 思路推广至其他注意力变体或多模态融合架构中。

--- 

> ✅ **总结一句话**：  
> RBS-Attention 通过引入 **radius-adaptive dual-branch selection**，有效缓解了 mean dilution 问题，在几乎无损质量的前提下实现了高达 **20× 的 prefill attention 加速** 和 **6× 的端到端 TTFT 提升**，是当前最先进的训练免费长上下文稀疏 prefill 方案之一。

</details>

---

### 6. [Distributed Balanced Butterfly Counting in Signed Bipartite Graphs](https://arxiv.org/abs/2609.21848)

**Authors**: Kiran Mekala, Apurba Das, Suman Banerjee  
**Category**: cs.DC  
**Published**: 2026-09-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.21848v1  

#### Abstract
The balanced butterfly is a fundamental primitive for analyzing signed bipartite graphs and provides a basis for studying higher-order structural properties, such as clustering coefficients and community structure. Despite its importance, existing approaches primarily rely on serial algorithms for b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Distributed Balanced Butterfly Counting in Signed Bipartite Graphs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**大规模有符号二分图（signed bipartite graphs）中的平衡蝴蝶（balanced butterfly）计数问题**提出了一种高效的分布式解决方案。  
- 平衡蝴蝶是基于社会平衡理论（social balance theory）定义的基本motif，用于分析用户-产品、立法者-法案等网络中的正负关系模式。
- 现有方法多为串行或共享内存并行算法，在处理大规模图时受限于单机计算和内存资源。
- 已有的分布式butterfly计数算法（如Monarch）仅适用于无符号图，无法直接扩展到有符号场景，因为需要额外验证边符号是否满足“平衡”条件。

### 提出了什么新方法或新思路
作者提出了 **D-BBC**（Distributed Balanced Butterfly Counting），一个基于 **混合 MPI+TBB 框架**的分布式算法，其核心创新包括：

1. **S-Monarch 基线构建**  
   将原用于无符号图的分布式算法 Monarch 扩展至有符号场景，得到 S-Monarch，作为首个可比较的分布式baseline。

2. **M-BBC 多核并行算法设计**  
   在原有串行算法 BB2K 上进行顶点级并行化，利用 TBB 实现共享内存加速，避免枚举非平衡子结构。

3. **D-BBC 分布式框架设计**  
   - 采用 **hybrid MPI+TBB** 架构：MPI 负责跨节点通信，TBB 实现节点内多线程并行。
   - 设计五阶段流水线：
     - Phase 1: 并行图加载（Parallel Graph Loading）
     - Phase 2: 全局度数计算（Global Degree Computation）
     - Phase 3: 工作负载感知的枢纽分配（Workload-aware Pivot Assignment）
     - Phase 4: 分布式局部子图构造（Distributed Local Subgraph Construction）
     - Phase 5: 并行本地平衡蝴蝶计数（Parallel Local Counting）

4. **关键技术优化**
   - **负载均衡策略**：基于 `W(u) = Σ d(v)` 的工作量估计 + 贪心bin-packing调度，显著减少straggler现象。
   - **通信优化**：通过两次 `MPI_Alltoallv` 完成邻域交换，最小化跨进程通信开销。
   - **去重机制**：使用顶点优先级规则（vertex priority rule）确保每个平衡蝴蝶只被统计一次。
   - **对称/非对称楔形分离**：将wedge分为 symmetric (`++/--`) 和 asymmetric (`+-/-+`) 两类，利用引理（Lemma 1）快速判断能否形成平衡蝴蝶。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可扩展性** | 支持跨多节点分布式执行，突破单机内存限制 |
| **效率** | 利用负载均衡与通信优化，大幅降低执行时间 |
| **准确性** | 保证精确计数，无重复或遗漏 |
| **实用性** | 首个支持大规模 signed bipartite graph 上 balanced butterfly counting 的分布式系统 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
在 **15个真实世界 bipartite 数据集**上进行了评估，涵盖多种领域和规模：

| 类型 | 示例数据集 |
|------|-----------|
| 自然有符号图 | `Senate`, `House`, `BookCrossing`, `Last.fm`, `Epinions` |
| 可转化为有符号图 | `Movielens`, `Jester`（评分转±） |
| 人工生成有符号图 | `DBLP`, `NAP`, `NIPS`, `KDD`, `AOL`, `DG`, `Yahoo`, `Netflix`（按70%+/30%-随机赋号） |

> 注：部分图存在重复边冲突，保留最新交互。

### 实验设置和评估指标

- **硬件平台**：HPC集群，每节点配备双AMD EPYC 9655（共192核）、385GB RAM、200Gbps InfiniBand
- **实现语言**：C++
- **并行模型**：MPI（跨节点） + Intel TBB（节点内）
- **评估指标**：
  - 总执行时间（end-to-end time）
  - 各阶段耗时分解（IO, Index, Exchange, Count, Reduce）
  - 加速比（speedup）
  - 通信量（bytes sent）
  - 内存峰值使用（peak RSS）
  - 负载不平衡度（straggler ratio = max_count_time / min_count_time）

### 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **BB2K** | Serial | 作者先前提出的串行算法，单核运行 |
| **M-BBC** | Shared-memory | 多核并行版本，单节点192线程 |
| **S-Monarch** | Distributed | 由Monarch扩展而来，支持有符号图但未做符号剪枝优化 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 单节点性能对比（vs. BB2K 和 M-BBC）
- **相比串行 BB2K**：
  - 平均加速比达 **540.9×**
  - 最高提速 **3723.9×**（在某些中等规模图上）
- **相比多核 M-BBC**：
  - 平均加速比 **11.9×**
  - 最高提速 **49.4×**（DG数据集）
- 对最大图 `NX` 和 `YH`：
  - BB2K 超过5小时未完成
  - D-BBC 成功完成，且比 M-BBC 快 **9.6×**（NX）和 **4.9×**（YH）

#### vs. 分布式基线 S-Monarch
- 在所有完成的数据集上，D-BBC 平均提速 **21.07×**
- 最大提速达 **96.00×**（JE数据集）
- S-Monarch 在 `NX` 和 `YH` 上因内存不足失败，而 D-BBC 成功运行

> **摘要原文强调**：D-BBC 在单节点下相较 BB2K 和 M-BBC 分别达到 **1321×** 和 **16.2×** 的平均加速，并在端到端时间上相较 S-Monarch 最高提速 **23.58×**

### 消融实验结果

#### A. Hybrid vs Pure MPI
- 对比配置：
  - Hybrid: 24 MPI ranks × 8 TBB threads
  - Pure MPI: 192 ranks × 1 thread
- 结果：
  - 平均提速 **20.2×**
  - 小图上最高达 **97×**（DBLP）
  - 表明 **TBB共享内存并行有效减少了MPI通信开销**

#### B. 负载感知分区 vs 朴素顶点分配
- 使用 straggler ratio 衡量负载均衡性
- 结果：
  - 在 AOL 上从 5.67 降至 2.89（改善48.97%）
  - 在 KDD、EP、DG 上均有显著改善
  - 对最大图 YH：朴素方法因严重负载倾斜导致OOM崩溃，而 D-BBC 成功完成（straggler ratio=1.43）

#### C. 符号分布鲁棒性测试
- 改变正负边比例（10%~90%）
- 发现：**运行时间几乎不变**，说明算法复杂度不依赖于符号分布
- 但平衡蝴蝶数量随符号分布剧烈变化，符合预期

#### D. 对称/非对称楔形分离有效性
- 若不分桶，则需显式枚举候选4环并逐一验证是否平衡
- 实验表明该优化带来巨大收益：

| Dataset | Without Separation (s) | With Separation (s) | Speedup |
|--------|-------------------------|---------------------|---------|
| SE     | 3.69                   | 0.0316              | **123×** |
| HO     | 157.00                 | 0.935               | **128×** |

---

## 4. 关键结论和发现

### 主要发现
1. **D-BBC 是首个可用于大规模 signed bipartite graphs 的分布式 balanced butterfly counting 算法**。
2. **混合 MPI+TBB 架构能高效利用现代多核分布式系统资源**，显著优于纯MPI方案。
3. **负载感知的任务划分（workload-aware pivot assignment）对性能至关重要**，尤其在大图上防止straggler。
4. **通信虽随MPI进程增加而上升，但在单节点内仍可通过共享内存高效处理**。
5. **算法运行时间对边符号分布具有鲁棒性**，适合实际应用中多样化的符号模式。

### 方法的局限性
1. **仍依赖静态图假设**，未支持动态或流式图更新。
2. **边界wedge通信仍有优化空间**，特别是跨物理节点时带宽可能成为瓶颈。
3. **贪心调度为集中式（Rank0完成）**，可能在超大规模下成为瓶颈。
4. **当前仅支持 balanced butterfly，尚未推广至更高阶motif（如 balanced biclique）**。

### 未来工作方向
1. **优化边界通信**：引入 locality-aware 或 communication-avoiding 图划分策略。
2. **扩展至动态图**：支持 temporal 和 streaming 场景下的增量 counting。
3. **支持更高阶 signed motifs**：如 balanced (k,l)-bicliques 或 signed bitrusses。
4. **Web-scale scaling**：探索在数千核以上集群上的可扩展性极限。

--- 

> ✅ **总结一句话**：  
> D-BBC 通过创新的 hybrid MPI+TBB 架构与负载感知调度，在真实大规模 signed bipartite graphs 上实现了高达 **1321×** 的加速，首次使分布式平衡蝴蝶计数成为现实，为高阶 signed network analysis 提供了强有力的基础设施支持。

</details>

---

### 7. [TierKV: Long-Context On-Device LLMs via Predictive Multi-Tier KV Caching](https://arxiv.org/abs/2609.21172)

**Authors**: Zhihao Shu, Md Musfiqur Rahman Sanim, Jie Hu, Kun Yuan, Minghai Qin, Gagan Agrawal, Wei Niu  
**Category**: cs.LG  
**Published**: 2026-09-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.21172v1  

#### Abstract
Large language models (LLMs) are moving onto mobile devices for increasingly diverse workloads over text, images, video, and audio. These applications often require long contexts, making the Key-Value (KV) cache a dominant memory bottleneck because it grows linearly with sequence length and is acces...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TierKV: Long-Context On-Device LLMs via Predictive Multi-Tier KV Caching —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在移动设备上部署 **Large Language Models (LLMs)** 面临严重内存瓶颈，尤其是 **Key-Value (KV) Cache** 在长上下文场景下呈线性增长，导致内存耗尽、应用崩溃（如触发 Android 的 Low Memory Killer）。现有方法存在以下缺陷：
- **低秩压缩 (Low-rank compression)**：引入重建开销（reconstruction overhead），影响推理延迟。
- **Token 蒸发 (Eviction)**：永久丢失信息，损害模型准确性。
- **闪存卸载 (Flash offloading)**：受限于带宽，造成 I/O 停顿。

### 🚀 提出的新方法：TierKV 与 PMCO
提出 **TierKV**，一个面向移动端的 LLM 推理框架，其核心是 **Predictive Multi-Tier Cache Optimization (PMCO)**，将 KV 缓存管理建模为**预测性资源分配问题**，而非被动压缩决策。

#### 创新点：
1. **联合预测多级缓存优化 (Joint Predictive Multi-Tier Optimization)**
   - 在解码前基于预填充阶段的隐藏状态预测总序列长度 $L$。
   - 联合决定三个缓存层级的边界和每层的 SVD 秩：
     - **Tier-0 (Exact)**：高敏感位置保留原始精度。
     - **Tier-1 (Compressed)**：中间部分采用 SVD 压缩。
     - **Tier-2 (Offloaded)**：尾部卸载至闪存，控制 RAM 占用。

2. **无需训练的运行时配置 (Training-Free Runtime Configuration)**
   - 利用预填充阶段的隐藏状态和输出 logits 的熵加权生成 `prompt fingerprint`。
   - 通过本地历史数据库进行 k-NN 匹配，预测输出长度，无需额外模型或人工标注。

3. **硬件感知异构注意力机制 (Hardware-Aware Heterogeneous Attention)**
   - 设计 **Split-Path Fused Kernel**，避免在关键路径上重建完整 KV。
   - 支持片上重建（on-chip reconstruction）和**潜空间值累积 (Latent-space value accumulation)**，显著降低 FLOPs。

4. **端到端移动推理框架**
   - 实现完整的跨平台支持（Adreno/Mali GPU），兼容文本、视觉、音频等多模态模型。

### 🔍 相比现有方法的优势
| 维度 | TierKV | 现有方法（如 SVD-only / Eviction / Offloading） |
|------|--------|---------------------------------------------|
| **内存效率** | ✅ 显著减少 RAM 占用（12.5–34%） | ❌ 固定策略，无法动态适配 |
| **准确性** | ✅ 全上下文保留，仅压缩非关键部分 | ❌ 蒸发导致不可逆信息丢失 |
| **延迟** | ✅ 重叠 I/O 与计算，隐藏重建成本 | ❌ 重建或 I/O 成为瓶颈 |
| **灵活性** | ✅ 每请求动态配置缓存布局 | ❌ 静态或反应式调整 |

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **模型集合**：共 8 个，涵盖多种模态与架构：
  - **文本**：Llama-3.2-1B/3B, Qwen2.5-3B, TinyLlama-1.1B, Gemma4 E2B
  - **视觉语言**：Qwen2-VLM-2B, SmolVLM2-1.7B
  - **语音语言**：Ultravox-1B
- **评估任务与基准**：
  - 文本理解：MMLU, ARC-Challenge, GSM8K
  - 多模态理解：MMMU-val
  - 语音识别：LibriSpeech-clean (WER)

### ⚙️ 实验设置
- **测试平台**：
  - OnePlus 12（Snapdragon 8 Gen 3, Adreno 750, 12GB RAM）
  - OnePlus 11（Adreno 740）
  - Google Pixel 8（Mali-G715, 8GB RAM）
- **内存预算**：稳定可用约 6.8GB（OnePlus 12）
- **精度设置**：FP16（权重与 KV Cache）
- **批大小**：1（模拟交互式推理）

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Prefill/Decoding Throughput** | 预填充与解码阶段吞吐量（tokens/s） |
| **KV Cache Memory Footprint** | RAM 中 KV 缓存占用（GB） |
| **Max Context Length ($L_{\text{max}}$)** | 在不 OOM 下支持的最大上下文长度 |
| **Accuracy** | 各基准任务得分变化（Δpp） |
| **End-to-End Latency** | 完整请求处理时间 |
| **Prefetch Hit Rate** | 闪存预取命中率 |

### 🆚 基线方法
- **llama.cpp**：主流开源移动端推理引擎
- **MNN-LLM**：阿里轻量化推理框架
- **MLC-LLM**：支持自动调度的移动端 LLM 框架

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（OnePlus 12 上平均表现）

| 指标 | TierKV 表现 | 对比提升 |
|------|-----------|---------|
| **Prefill Throughput** | 平均 **1.34×** llama.cpp，最高达 **17.6×** MNN-LLM | 显著加速 |
| **Decoding Throughput** | 平均下降 28%（因 SVD 重建） | 但被预填充增益抵消 |
| **KV Cache 内存节省** | **12.5% – 34%**，平均 **25%** | 显著释放内存 |
| **最大上下文长度** | 最高延长 **2.6×**（如 Llama-3.2-1B 达 72k） | 突破 DRAM 限制 |
| **端到端速度** | 平均 **1.10×** llama.cpp | 整体更快 |
| **I/O 预取命中率** | **94.4%** | 高效隐藏 I/O 延迟 |

### 🔁 与基线方法对比（Table 7 总结）
| 模型 | Prefill Speedup (vs llama.cpp) | Memory Saving | Max Context Increase |
|------|-------------------------------|----------------|------------------------|
| Llama-3.2-1B | **1.5×** | 29% | 32k → **72k** |
| Llama-3.2-3B | 1.2× | 26% | 7k → **14k** |
| Qwen2.5-3B | 1.5× | 20% | 25k → **32k** |
| Ultravox-1B | 1.2× | 34% | 50k → **55k** |

> 💡 **MLC-LLM** 仅支持 Llama-3.2-1B（短上下文），TierKV 在该模型上实现 **183× 更高的预填充吞吐量**。

### 🔍 消融实验结果（Ablation Studies）

#### （1）PMCO 联合优化 vs 启发式策略（Table 10）
| 策略 | KV 内存节省 | 延迟（ms） |
|------|------------|----------|
| Fixed Boundaries | 12.8% | 166.7 |
| Sequential Optimization | 14.9% | 184.9 |
| **PMCO (Ours)** | **26.0%** | **162.5** |

✅ 结论：**联合优化显著优于分步或固定策略**，在满足准确性的前提下找到更优内存-延迟平衡点。

#### （2）各组件对性能的贡献（Table 12）
| 优化模块 | 内存节省 | 预填充加速 | 端到端加速 |
|--------|--------|----------|----------|
| PMCO | 13.8% | 1.00× | 1.00× |
| + Hierarchical Cache | 23.4% | 1.10× | 0.63× |
| + Latent-V Accumulation | 23.6% | 1.17× | 0.85× |
| + Fused Attention | 24.6% | 1.28× | 0.99× |
| + I/O Overlap | **25.0%** | **1.34×** | **1.10×** |

✅ 结论：**I/O 与计算重叠是最终实现正向收益的关键**。

#### （3）长度预测鲁棒性（Table 11）
| 方法 | 内存开销（相对） | 延迟开销（ms/请求） |
|------|------------------|--------------------|
| Static Mean Length | 1.00× | 22.8 |
| **TierKV** | **0.33×** | **7.9** |

✅ 结论：**基于熵的预测器显著降低资源浪费**，即使预测不准也不会导致 OOM。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **预测性缓存规划优于反应式蒸发**  
   TierKV 通过 **prefill 阶段的隐藏状态预测** 实现**先发制人**的缓存布局，避免了传统方法“边用边删”的不确定性。

2. **三层次缓存可有效解耦内存与上下文长度**  
   通过 Tier-2 闪存卸载，**将内存瓶颈转移为 I/O 带宽瓶颈**，从而支持远超物理 RAM 容量的上下文。

3. **SVD 重建开销可通过系统级优化隐藏**  
   - **Latent-V 累积** 将 $O(L \cdot z \cdot d)$ 降为 $O(L \cdot z + z \cdot d)$
   - **Split-Path Fused Kernel** 消除中间张量
   - **I/O 与重建重叠** 将 I/O 空泡转化为计算窗口

4. **动态 per-request 配置优于静态策略**  
   不同任务（如 GSM8K vs MMLU）对压缩敏感度不同，**统一压缩率会牺牲质量或浪费内存**，而 PMCO 可自适应选择最优配置。

### ⚠️ 局限性
1. **依赖 SVD 离线校准**  
   虽然只需一次，但仍需为每个模型执行谱分析以确定每层秩配置 $\{R_l\}$。

2. **移动端 GPU 缺乏并发 DMA**  
   当前仍需等待 H2D 传输完成，若硬件支持并发传输，性能可进一步提升。

3. **未支持量化 KV Cache**  
   当前仅处理 FP16 KV，未来可结合 INT8/INT4 KV 量化进一步压缩。

4. **多轮对话中的缓存复用挑战**  
   虽然优于蒸发型方法，但如何高效复用压缩后的缓存仍需研究。

### 🔮 未来工作方向
1. **支持 Hybrid SSM-Attention 模型**  
   如 Jamba、Zamba 等混合架构，其 SSM 层状态固定，可与 TierKV 的 KV 管理互补。

2. **集成 Prefix Caching**  
   支持提示词前缀复用（如 RadixAttention），并将其纳入 PMCO 的内存预算中。

3. **CPU + NPU 协同优化**  
   利用 ARM big.LITTLE 架构，在 CPU 小核上执行 SVD 重建，进一步释放 GPU 资源。

4. **支持动态 rank adaptation during decoding**  
   当前 rank 在 prefill 后即固定，未来可探索在解码过程中动态调整。

---

> **总结一句话**：  
> **TierKV 通过“预测 + 分层 + 融合”三位一体设计，在移动端实现了长上下文 LLM 推理的内存、速度与准确性的帕累托前沿突破。**

</details>

---

### 8. [Accelerating Dense LLMs via L0-regularized Mixture-of-Experts](https://arxiv.org/abs/2609.21672)

**Authors**: Zhenyu Zhang, Jiudong Yang, Zhaowen Tao, Meng Chen  
**Category**: cs.AI  
**Published**: 2026-09-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.21672v1  

#### Abstract
Large language models (LLMs) achieve strong performance but suffer from slow and costly inference. Existing acceleration methods often lead to noticeable performance degradation, while Mixture-of-Experts (MoE) models require extensive computational resources. In this paper, we propose L0-MoE, a ligh...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerating Dense LLMs via L0-regularized Mixture-of-Experts

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）虽然在多项任务上表现出色，但其**推理速度慢、计算成本高**，限制了实际部署。现有的加速方法如量化（Quantization）、剪枝（Pruning）和知识蒸馏（Knowledge Distillation）通常会导致明显的**性能下降**。而 Mixture-of-Experts（MoE）模型虽能提升效率，但训练需要海量数据和算力（如万亿级 token 预训练），难以低成本复现。

本文旨在解决：  
> 如何以**极小规模训练语料**（仅 30B tokens）构建高效的 MoE 模型，在几乎不损失性能的前提下显著加速密集 LLM 的推理？

---

### 🚀 提出的新方法：L0-MoE
作者提出 **L0-MoE** —— 一种轻量化的 MoE 架构，通过以下三个核心技术实现高效推理加速：

1. **基于 L0-regularization 的专家构建**  
   - 利用 L0 正则化从预训练的 dense LLM 中选择关键隐藏维度，形成 specialized experts。
   - 不是从头训练，而是对 FFN 层进行稀疏化选择，保留最重要的参数路径。
   - 实现方式可微分，支持端到端优化。

2. **Cluster Confusion Matrix (CCM) 引导的数据集采样**
   - 使用 BGE-M3 编码器提取语义向量，结合 K-means 聚类划分不同语义域。
   - 设计 CCM 来衡量跨迭代聚类的一致性，优先选择语义区分度高的子数据集用于训练。
   - 实现“领域感知”的数据构造，提升专家专业化程度。

3. **动态批处理策略（Dynamic Batching）**
   - 两阶段调度：
     - 初期使用语义相近样本，帮助 router 快速学习专家分配；
     - 后期引入多样化语义样本，增强 token-level 的路由能力。
   - 提升 MoE 训练稳定性和泛化性。

---

### 🔍 相比现有方法的优势

| 方法 | 数据需求 | 性能保持 | 推理加速 | 是否依赖大规模训练 |
|------|----------|-----------|------------|------------------------|
| GPTQ / AWQ（量化） | 少 | 明显下降 | ~1.8x | ❌ |
| LLM-Shearing（剪枝） | 少 | 下降明显 | ~2.6x | ❌ |
| RKD+CoT（蒸馏） | 中等 | 显著下降 | ~5.1x | ❌ |
| DeepSeek-MoE / Mixtral（传统 MoE） | 极大（T+ tokens） | 高 | 高 | ✅ |
| **L0-MoE（本文）** | **极小（30B tokens）** | **几乎无损** | **2.0–2.5x** | **❌** |

✅ **核心优势总结**：
- **低资源友好**：仅需 30B token 微调即可完成 MoE 构建。
- **高性能保持**：平均性能与原始 dense LLM 几乎持平，部分甚至略有提升。
- **高推理效率**：达到 2.5× 推理加速，优于多数非 MoE 加速方案。
- **框架无关**：可在 FSDP、SGlang 等主流框架中部署。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **训练数据**：RedPajama（LLaMA 复刻预训练语料）
  - 总采样量：30B tokens
  - 通过 CCM 迭代筛选出高质量、多样化的子集
- **评估基准**（四大公开测试集）：
  - **MMLU**：多任务理解（涵盖 STEM、人文、社科等 57 个任务）
  - **GSM8K**：小学数学应用题（考察推理能力）
  - **HumanEval**：代码生成（函数补全准确性）
  - **BigBench Hard (BBH)**：极具挑战性的复杂任务集合

---

### ⚙️ 实验设置与评估指标

| 项目 | 设置说明 |
|------|----------|
| **基础模型** | Llama-3-8B、Mistral-7B、Qwen2-7B |
| **MoE 配置** | K=64 个专家，top-2 路由机制 |
| **训练策略** | FSDP + Zero-3 参数分片，无 CPU 卸载 |
| **推理框架** | SGlang（统一用于所有模型，确保公平比较） |
| **评估指标** | 准确率（Accuracy） + 推理速度（Speedup） |
| **超参设置** | 学习率 1e-4，序列长度 4096，batch tokens=512K |

---

### 🆚 基线方法对比

- **原始 dense LLMs**：作为性能上限参考
- **GPTQ**：典型量化方法（4-bit）
- **LLM-Shearing**：结构化剪枝方法
- **RKD + CoT Distillation**：反向 KL 散度 + 思维链蒸馏
- **Random MoE / Magnitude / OBS / SVD**：用于消融研究的替代专家构建方式

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

| 模型 | MMLU | GSM8K | HumanEval | BBH | 平均得分 | Speedup |
|------|-------|--------|-------------|-------|------------|---------|
| Llama-3-8B | 66.6 | 56.0 | 33.5 | 57.7 | 53.5 | 1.0x |
| Llama-3-8B w/ L0-MoE | 66.3 | 55.9 | 33.7 | 57.2 | 53.3 | **2.0x** |
| Mistral-7B | 64.1 | 52.2 | 29.3 | 56.1 | 50.4 | 1.0x |
| Mistral-7B w/ L0-MoE | 64.8 | 53.6 | 31.1 | 55.9 | **51.4** | **2.1x** |
| Qwen2-7B | 70.3 | 79.9 | 51.2 | 62.6 | 66.0 | 1.0x |
| Qwen2-7B w/ L0-MoE | 70.4 | 80.5 | 52.0 | 61.5 | **66.1** | **2.5x** |

✅ **结论**：
- 所有 L0-MoE 变体均实现 **2.0–2.5x 推理加速**
- 性能不仅未下降，**Mistral 和 Qwen2 版本平均得分反而略有上升**

---

### 🔁 与其他加速方法对比（Table 2）

| 方法 | MMLU | GSM8K | Speedup |
|------|-------|--------|---------|
| Qwen2-7B（原版） | 70.3 | 79.9 | 1.0x |
| **L0-MoE** | **70.4** | **80.5** | **2.5x** |
| GPTQ | 67.8 | 73.8 | 1.8x |
| LLM Shearing | 68.2 | 75.5 | 2.6x |
| RKD+CoT | 61.2 | 60.2 | 5.1x |

📌 **观察**：
- 尽管 RKD+CoT 达到最高 **5.1x 加速**，但性能严重退化（↓近 10 分）
- L0-MoE 在**加速与性能之间取得最佳平衡**

---

### 🔍 消融实验结果（Table 3）

| 消融设置 | MMLU | GSM8K |
|----------|-------|--------|
| Full L0-MoE | 70.4 | 80.5 |
| CCM w/o K-means | 68.2 | 78.1 |
| w/ random order batching | 68.2 | 75.5 |
| w/ random batch batching | 66.6 | 77.1 |
| Random MoE | 48.1 | 69.6 |
| Magnitude | 52.6 | 69.1 |
| OBS | 68.4 | 74.1 |
| SVD | 55.2 | 73.8 |

📌 **关键发现**：
- 移除 **K-means 聚类**导致性能显著下降 → 表明**语义划分的有效性至关重要**
- 替换为随机批处理会削弱训练效果 → **动态批处理设计有效**
- 替换 L0-regularization 为其他方法（尤其是 Random MoE）性能暴跌 → **L0 正则化是专家构建的关键**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **L0-regularization 是构建高效 MoE 的可行路径**  
   - 可在不重新预训练的情况下，从 dense LLM 中“提炼”出 specialized experts。
   - 比随机或基于幅值的选择更有效。

2. **小规模语料也能训练高性能 MoE**  
   - 仅用 30B tokens 即可完成 MoE 微调，远低于传统 MoE 动辄万亿 token 的需求。

3. **CCM + 动态批处理显著提升训练质量**  
   - 数据采样策略直接影响专家专业化水平和路由稳定性。

4. **推理加速与性能保持可以兼得**  
   - L0-MoE 实现了接近无损压缩下的 **2.5x 推理提速**，超越多数现有加速技术。

---

### ⚠️ 方法的局限性

1. **存在暴露偏差（Exposure Bias）**  
   - 当前专家训练基于 sequence-level 聚类，但推理时路由发生在 token-level，可能导致不匹配。

2. **缺乏显式的专家差异性约束**  
   - 不同专家可能学到相似功能，造成参数冗余，影响加速上限。

3. **未与大规模 MoE 直接对比**  
   - 未与 DeepSeek-MoE、Mixtral 等 full-scale MoE 对比，因后者训练成本过高。

4. **扩展性尚未验证于更大模型**  
   - 当前实验集中在 7B–8B 模型，对 70B+ 模型的效果仍待探索。

---

### 🔮 未来工作方向

1. **探索 token-level 数据划分机制**  
   - 缓解 sequence-level 训练与 token-level 推理之间的 mismatch。

2. **设计专家差异化正则项**  
   - 引入 contrastive learning 或 diversity loss，减少专家冗余。

3. **扩展至更大规模 LLM（如 70B+）**  
   - 验证是否能在更大模型上获得更高加速比。

4. **结合量化/剪枝进一步压缩 MoE**  
   - 探索 L0-MoE + Quantization 的联合优化空间。

5. **开放数据与代码促进复现**  
   - 作者承诺将发布 curated dataset 和 code，推动社区发展。

---

> 💡 **一句话总结**：  
> L0-MoE 提供了一种**低成本、高性能、易部署**的 LLM 推理加速新范式——用 L0 正则化“挖出”dense 模型中的潜在专家，配合智能数据采样与训练调度，在仅 30B token 上实现了接近无损的 2.5x 加速，为工业级高效 LLM 部署提供了实用解决方案。

</details>

---

### 9. [ExpBoN: Exponential-Noise Best-of-$n$ for Efficient Test-Time LLM Alignment](https://arxiv.org/abs/2609.21899)

**Authors**: Yanxiao Liu, Sicheng Wan, Deniz G\"und\"uz  
**Category**: cs.LG  
**Published**: 2026-09-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.21899v1  

#### Abstract
Best-of-$n$ (BoN) sampling is a simple yet effective inference-time alignment method, but hard maximization provides only coarse control over the trade-off between reward and distribution shift. Soft Best-of-$n$ (Verdun et al. 2025) provides smoother control and converges to the optimal distribution...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ExpBoN: Exponential-Noise Best-of-n for Efficient Test-Time LLM Alignment**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **Inference-time alignment** 中的 **Best-of-n (BoN)** 方法虽然简单有效，但存在两个主要缺陷：
  1. **控制粒度粗**：候选数量 $ n $ 是唯一调节参数，同时影响奖励优化和分布偏移，难以精细权衡。
  2. **Reward hacking 风险**：硬最大化（hard maximization）容易过拟合代理奖励模型（proxy reward model），导致选择高代理奖励但低真实奖励的样本。

### **提出的新方法**
- 提出 **ExpBoN**（Exponential-Noise Best-of-n），一种基于 **exponential-noise report-noisy-max** 机制的软 BoN 方法。
- ExpBoN 在有限 $ n $ 下具有 **精确的指数倾斜分解（exact finite-n decomposition）**，其输出分布可表示为：
  $$
  P_{n,\lambda} = (1 - p_n) P + p_n P^*
  $$
  其中 $ P^* $ 是目标倾斜分布，$ p_n $ 是“命中”概率，随 $ n $ 增加而指数增长。

### **相比现有方法的优势**
- **收敛速度更快**：
  - 相比 Soft BoN (SBoN)，ExpBoN 在 **Total Variation (TV)**、**KL 散度** 和 **期望奖励** 上均实现 **指数级收敛**，而非多项式收敛（如 $ O(1/n) $）。
- **理论保证更强**：
  - 提供了 **双向 KL 散度** 的上下界，而 SBoN 仅提供单向。
  - 在 **regret 分析** 中，ExpBoN 的有限样本项为 $ O(p^n) $，远优于 SBoN 的 $ O(1/\sqrt{n}) $。
- **可集成到 GSI 框架**：
  - 提出 **ExpGSI**，将 ExpBoN 替换 GSI 中的 SBoN 组件。
  - 利用 **截断拒绝采样（truncated rejection sampling）** 实现 **early-exit**，在命中时提前返回，显著降低计算开销。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MATH500**：400 道数学竞赛题，用于评估推理能力。
- **MMLU-STEM**：STEM 领域的多选题，测试专业知识。
- **Minerva Math**：大学级别数学题，更具挑战性。

### **实验设置**
- **模型配置**：
  - **Qwen2.5-Math**：1.5B draft / 7B target / 7B PRM。
  - **Qwen3**：1.7B draft / 14B target / 7B PRM（禁用 thinking mode）。
- **候选数 $ n \in \{2,4,8,16\} $**（Qwen3 使用 $ \{4,16\} $）。
- **温度参数 $ \beta = 20 $**，接受阈值 $ u = 0.5 $。
- **剪裁水平 $ C = 0.45 $**（通过校准集 95% 分位数确定）。

### **评估指标**
- **Accuracy**：最终答案正确率（macro average）。
- **Time per step (s)**：每步推理耗时。
- **Acceptance (%)**：使用 draft 模型直接接受的比例。
- **Estimated TFLOPs/prob**：按 RSD 方案估算的每题浮点运算量。

### **基线方法对比**
- **S-BoN(ns)**：仅使用 draft 模型生成并评分。
- **S-BoN(TB)**：使用 target 模型生成所有候选（质量上限）。
- **RSD**：Reward-guided Speculative Decoding，基于奖励门控的加速方法。
- **GSI**：Guided Speculative Inference，原版基于 SBoN 的对齐框架。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | $ n $ | 方法 | Acc. (%) | Est. TFLOPs/prob | Compute Reduction vs GSI |
|------|-------|--------|----------|-------------------|--------------------------|
| Qwen2.5-Math | 16 | GSI | 59.7±0.8 | 1627 | — |
| | | **ExpGSI (ours)** | **59.7±0.8** | **999** | **39%** |
| Qwen3 | 16 | GSI | 61.0±0.1 | 5769 | — |
| | | **ExpGSI (ours)** | **61.8±0.6** | **3169** | **45%** |

### **与基线方法的对比结果**
- **ExpGSI 在所有 $ n $ 下均显著降低计算成本**：
  - Qwen2.5-Math：节省 **14%–39%** 的 TFLOPs。
  - Qwen3：最高节省 **45%**（$ n=16 $）。
- **精度保持不变**：
  - 与 GSI 相比，准确率差异在 **±0.8%** 内，无统计显著下降。
  - 甚至在 Qwen3 $ n=16 $ 时略优（61.8 vs 61.0）。
- **优于 RSD**：
  - ExpGSI 比 RSD 高 **1.2–4.9 个百分点** 准确率，同时计算量更低。

### **消融实验结果**
#### **(1) 奖励模型鲁棒性测试**
- 将 7B 数学专用 PRM 替换为 1.5B 通用 PRM（SKYWORK-O1-OPEN-PRM-QWEN-2.5-1.5B）。
- 结果：ExpGSI 仍保持与 GSI 相当的准确率，并继续降低计算成本。

#### **(2) 剪裁水平 $ C $ 的敏感性分析**
- 测试 $ C/C_0 \in \{0.5, 0.75, 1.0, 1.5, 2.0, \infty\} $。
- 发现：
  - $ C \in [0.5, 1.5] \times C_0 $ 范围内，准确率稳定。
  - $ C $ 越大，平均评分候选数越多，计算成本上升。
  - $ C=\infty $（无剪裁）时，early-exit 失效，计算成本恢复至全扫描水平。
- 表明 **剪裁是高效 early-exit 的关键**。

#### **(3) ExpBoN vs SBoN 在真实候选池上的收敛行为**
- 在真实 LLM 生成的候选池上比较两者收敛速度。
- 结果：
  - ExpBoN 在 **KL 散度** 和 **相对奖励差距** 上均呈 **几何收敛**。
  - SBoN 呈 **多项式衰减**，验证了理论优势。

---

## **4. 关键结论和发现**

### **主要发现**
1. **ExpBoN 是更高效的软 BoN 机制**：
   - 基于 exponential noise 的 report-noisy-max 机制，实现了 **指数级收敛**。
   - 理论上优于 Gumbel noise 的 SBoN（多项式收敛）。
2. **ExpGSI 显著提升 GSI 效率**：
   - 通过 **clipped ExpBoN + early-exit**，在不牺牲准确率的前提下，**减少 39–45% 的计算量**。
   - 特别适用于大 target 模型场景（如 Qwen3），因避免的 target 推理占比更高。
3. **方法具有强泛化性和鲁棒性**：
   - 在不同模型族（Qwen2.5-Math vs Qwen3）、不同奖励模型下表现一致。
   - 对剪裁水平 $ C $ 不敏感，在合理范围内均可获得收益。

### **方法的局限性**
- **依赖高质量 draft 模型**：若 draft 模型生成质量差，则 early-exit 的命中率低，无法有效节省计算。
- **剪裁引入偏差**：虽然可通过定理 4.1 界定，但仍改变了原始 GSI 的目标分布。
- **理论假设较强**：如 reward 有界、finite alphabet 等，在极端情况下可能不成立。

### **未来工作方向**
1. **联合 alignment 与 watermarking**：
   - ExpBoN 与 **Permute-and-Flip** 机制等价，后者已被证明在 watermarking 中具有 Pareto 最优性，可探索联合优化。
2. **结合 reward hacking 缓解技术**：
   - 如 Khalaf et al. (2025) 提出的防御方法，防止过快收敛加剧 overoptimization。
3. **扩展到 diffusion language models**：
   - SBoN 已被用于 diffusion LM，可研究 ExpBoN 是否能带来类似收益。
4. **动态 $ C $ 或 adaptive clipping**：
   - 当前 $ C $ 固定，未来可设计自适应剪裁策略以进一步优化 trade-off。

---

> **总结**：  
> ExpBoN 通过引入 **exponential noise** 和 **exact finite-n decomposition**，为 test-time LLM alignment 提供了一个 **理论上更优、实践中更高效** 的解决方案。其与 GSI 的结合（ExpGSI）实现了 **“零精度损失”的显著加速**，为大规模部署对齐 LLM 提供了实用路径。

</details>

---

### 10. [TokaGLINT: A Scalable GPU-Tailored Implicit Solver for Full 3D Tokamak Electromagnetic Simulations](https://arxiv.org/abs/2609.21366)

**Authors**: Zifan Yang, Haoyuan Zhang, Jialin Li, Wu Yuan, Xiazhen Liu, Jian Zhang, Jianyuan Xiao, Shan Liang  
**Category**: cs.DC  
**Published**: 2026-09-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.21366v1  

#### Abstract
We introduce TokaGLINT, a GPU-accelerated implicit solver for electromagnetic field computations in full 3D tokamak simulations, aimed at efficient large-scale parallel GPU computing. Its central innovation lies in the co-design of hierarchical domain decomposition and a fast exact local solver, whe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TokaGLINT: A Scalable GPU-Tailored Implicit Solver for Full 3D Tokamak Electromagnetic Simulations**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在托卡马克（tokamak）等磁约束聚变装置的全三维电磁场模拟中，传统显式 Particle-in-Cell (PIC) 方法受限于 **Courant-Friedrichs-Lewy (CFL)** 条件，导致时间步长极小，难以进行长时间尺度的高效模拟。尤其是在低密度等离子体或采用减小质量比的情况下，**EM wave CFL** 成为主要瓶颈。

此外，隐式求解器（如 CN-FDTD）虽可突破 CFL 限制，但其核心是大规模稀疏线性系统的迭代求解，面临 **并行扩展性差** 和 **通信开销高** 的挑战。

### **提出的新方法与创新思路**
作者提出了 **TokaGLINT** —— 一种专为 GPU 架构优化的、可扩展的隐式线性求解器，用于加速全 3D 托卡马克电磁场模拟。其核心创新包括：

#### **(1) 层次化加性施瓦茨方法（Hierarchical Additive Schwarz Method, HASM）**
- **两级域分解**：
  - **L1（跨 GPU）**：传统的重叠域分解，每个 GPU 负责一个大子域。
  - **L2（片内多子域）**：在单个 GPU 内部进一步划分多个重叠子域，充分利用 GPU 的高带宽内存（HBM）处理子域间通信。
- 支持 **批量融合求解（batched and fused subdomain solves）**，显著提升硬件利用率。

#### **(2) 面向曲坐标系的快速精确局部求解器（Fast Exact Local Solver）**
- 针对柱坐标系下的 **symplectic CN-FDTD 离散化 Maxwell 方程** 设计。
- 利用 **离散变换（discrete transforms）** 和 **张量结构运算（tensor-structured operations）** 对未知量进行部分解耦。
- 将原系统转化为两个子系统：
  - 一个 **对角系统**（直接求解）
  - 一个 **沿 x 方向耦合的二维块对角系统**（预计算逆矩阵，实现高效求解）

#### **(3) 算法-硬件协同设计（Co-design）**
- **操作融合（Operator Fusion）**：将多个子域的正/反变换、置换、块对角求解等操作融合为批量 GEMM 运算。
- **几何驱动分组（Geometry-driven Binning）**：具有相同度量张量的子域被归入同一“bin”，共享变换矩阵，减少冗余计算。
- 数据布局定制化，支持高效的批处理和流水线执行。

### **相比现有方法的优势**
| 特性 | TokaGLINT | 传统方法（如 HYPRE 中的 AMG、ILU） |
|------|-----------|-------------------------------|
| 并行扩展性 | ✅ 超过 10,000 GPU 下仍保持良好效率 | ❌ 多数预条件子在大规模下收敛恶化 |
| 收敛速度 | ✅ 迭代次数极少（仅需 ~8 步） | ❌ 迭代次数随规模增长而上升 |
| 硬件适配性 | ✅ 充分利用 GPU 张量核心与 HBM | ❌ 多为 CPU 友好设计，GPU 利用率低 |
| 数值保真性 | ✅ 保留辛结构（symplectic structure），能量长期守恒 | ⚠️ 显式方法易累积误差；ADI 等破坏辛结构 |

---

## **2. 核心实验方法和设置**

### **实验平台**
- **超级计算机环境**：中国新一代异构超算系统
- **节点配置**：
  - 每节点 8 个 GPU
  - 每 GPU：32.7 TFLOPS FP64 性能，64GB HBM，理论带宽 1.8 TB/s
  - 节点间通过 4×400 Gbps InfiniBand RDMA 互联
- **软件栈**：HIP 兼容内核 + GPU-aware MPI

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Weak Scaling Efficiency** | 固定每 GPU 负载，衡量总性能随 GPU 数增加的增长比例 |
| **Strong Scaling Efficiency** | 固定问题总大小，衡量加速比 |
| **Iteration Count** | BiCGStab 收敛所需迭代次数 |
| **Time-to-Solution** | 单次线性系统求解平均耗时 |
| **Communication Overhead** | 各阶段通信占比分析 |

### **基线方法对比**
- **Baseline**: 无预条件的 BiCGStab（来自 HIP-enabled HYPRE）
- **对比项**：HYPRE 中的标准预条件子
  - Jacobi
  - SOR
  - ILUT
  - AMG
  - ISAI
- 所有对比均使用默认参数并通过 PETSc/HYPRE 接口调用

### **测试场景**
- **物理模型**：基于 SymPIC 框架的 EAST 托卡马克电磁场模拟
- **典型网格尺寸**：从 $64^3$ 到 $128^3$ 每 GPU
- **时间步长 $\Delta t$**：覆盖 $1.0 \sim 16.0$（相对单位），验证大步长稳定性

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 单节点性能提升**
| 方法 | np=8（8 GPU）求解时间 (ms) | 迭代次数 | 加速比 |
|------|--------------------------|--------|-------|
| HYPRE (unprecond.) | 212.38 | 125 | 1.0× |
| **TokaGLINT** | **79.67** | **8** | **2.67×** |

> ✅ 在单节点实现 **2.67 倍端到端加速**

#### **(2) 弱扩展性（Weak Scaling）**
- 测试范围：从 16 到 **10,000 GPUs**
- 结果：
  - BiCGStab 迭代数稳定在 **~7–8 步**
  - 并行效率达 **90.1%**
- 图表显示通信开销被有效隐藏，SpMV 与预条件应用与全局规约重叠

#### **(3) 强扩展性（Strong Scaling）**
- 固定全局网格大小：$2048×1536×1792$
- GPU 数从 672 扩展至 **10,752**
- 并行效率维持在 **53.9%**

#### **(4) 收敛性优势（Table I）**
在不同时间步长下，TokaGLINT 的迭代次数远低于所有通用预条件子：

| $\Delta t$ | AMG (HYPRE) | TokaGLINT (l=4) |
|------------|-------------|------------------|
| 1.0        | 5           | **2**            |
| 8.0        | 37          | **7**            |
| 16.0       | 63          | **10**           |

> ✅ 即使在最大步长下也只需 10 次以内收敛，而 AMG 需 63 次

#### **(5) 消融实验（Ablation Study）**
##### **L2 域分解有效性（Table III）**
| L2 子域大小 | 求解时间 (ms) @ tol=$10^{-12}$ | 迭代次数 |
|------------|-------------------------------|--------|
| $64^3$ (单一大域) | 302.7 | 9 |
| $32^3$ | 92.1 | 9 |
| $16^3$ (**推荐配置**) | **74.6** | 9 |

> ✅ 尽管迭代次数相同，但 **L2 分解带来显著性能提升**（>4× 加速），得益于更好的批处理效率和内存访问模式

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **TokaGLINT 实现了前所未有的可扩展性**：在超过 **10,000 GPU** 上实现了 **90.1% 弱扩展效率** 和 **53.9% 强扩展效率**，是目前报道中最可扩展的隐式 EM 求解器之一。
2. ✅ **HASM + 快速局部求解器显著降低迭代次数**：相比 AMG 等主流方法，迭代次数减少 **6–8 倍**，且不随问题规模恶化。
3. ✅ **算法-硬件协同设计释放 GPU 潜力**：通过 operator fusion、batching 和 custom data layout，将原本不规则的稀疏求解转化为规则的批处理 GEMM，极大提升了 GPU 利用率。
4. ✅ **成功集成至 SymPIC 并验证物理正确性**：
   - 波传播测试表明数值解符合物理预期；
   - 长时间模拟（$10^6$ 步）中总能量波动控制在 **±1‰** 以内，验证了辛结构保真性。

### **方法的局限性**
1. 🛑 **依赖特定离散格式**：当前求解器针对 **symplectic CN-FDTD on cylindrical mesh** 定制，推广至其他坐标系（如球坐标）或离散方案需重新推导。
2. 🛑 **内存占用较高**：由于需要存储预计算的 $(2n_x)×(2n_x)$ 块逆矩阵，内存复杂度为 $O(N^{4/3})$，对极端细粒度划分不利。
3. 🛑 **开发复杂度高**：高度定制化的数据布局和融合策略增加了代码维护难度。

### **未来工作方向**
1. 🔮 **拓展至 fully implicit PIC 框架**：将粒子运动也纳入隐式求解，构建完全一致的隐式框架。
2. 🔮 **支持更多物理场景**：如包含材料边界、非理想效应（resistivity, Hall term）等。
3. 🔮 **自动化参数调优**：开发自适应选择 overlap width $l$ 和 L2 划分策略的机制。
4. 🔮 **跨架构移植**：探索在 NVIDIA CUDA、Intel Xe 和国产加速器上的高效实现。

---

> **总结一句话**：  
> TokaGLINT 通过 **层次化域分解 + 曲坐标快速求解器 + GPU 批处理融合** 的协同设计，在万卡级 GPU 集群上实现了高效、稳定、可扩展的全 3D 托卡马克电磁场隐式求解，为聚变等离子体的大规模长期模拟提供了关键技术支撑。

</details>

---

### 11. [Learning to Move Cities: Deep Meta-Models and Reinforcement Policies for Calibration and Control in Urban Networks](https://arxiv.org/abs/2609.21945)

**Authors**: Adewumi Augustine Adepitan, Christopher J. Haruna, Oluwasegun Adegoke, Ayooluwatomiwa Ajiboye, Oluwatobi Oluwasakin  
**Category**: cs.LG  
**Published**: 2026-09-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.21945v1  

#### Abstract
Urban transportation networks present complex optimization challenges spanning calibration of high-fidelity simulators and real-time operational control. This paper presents a shared latent-space framework that connects simulator calibration and reinforcement learning control through a common learne...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning to Move Cities: Deep Meta-Models and Reinforcement Policies for Calibration and Control in Urban Networks*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦城市交通网络中的两大核心挑战：
- **高保真仿真模型的校准**（Calibration）：传统方法在面对大规模、高维参数空间时效率低下。
- **实时动态控制策略优化**（Control）：强化学习（RL）在高维状态/动作空间中训练不稳定且样本效率低。

这两个任务通常被独立处理，导致信息割裂、系统集成困难。

### 🚀 提出的新方法与创新思路
提出了一种**共享潜在空间框架**（shared latent-space framework），将 **simulator calibration** 与 **reinforcement learning 控制** 统一在一个共同的学习表示下。

#### 核心架构包括：
- **Combinatorial MLP-Autoencoder**：
  - 联合建模仿真器输入（如 OD demand、网络参数）与输出（如 travel times、congestion patterns）。
  - 学习一个低维的 **latent manifold** 来捕捉交通动力学的本质特征。
- **Bayesian Optimization in Latent Space**：
  - 在压缩后的 latent space 中进行贝叶斯优化，显著提升校准的样本效率。
- **Deep Q-Network (DQN) with Latent State Representation**：
  - 将校准阶段学到的 latent representation 注入 RL 的 state 输入中，使控制器基于“已校准”的交通动态做出决策。

> 🔗 创新在于：**同一个 latent representation 同时服务于 calibration 和 control**，实现从规划到运营的端到端连接。

### ⚖️ 相比现有方法的优势
| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| **Calibration维度** | 高维原始参数空间搜索，计算昂贵 | 在低维 latent space 搜索，样本效率更高 |
| **Dimension Reduction** | 使用线性方法（如 Active Subspaces），难以捕获非线性关系 | 使用深度 autoencoder 学习非线性流形 |
| **RL State Design** | 基于 raw traffic states（易受噪声影响） | 引入 simulator-informed latent features，更稳定、更具语义意义 |
| **系统整合性** | Calibration 与 Control 分离 | 共享 latent 表示，形成闭环反馈路径 |

---

## 2. 核心实验方法和设置

### 📊 数据集与仿真平台
- **Calibration 实验**：
  - 使用 **POLARIS agent-based simulator** 构建中等规模城市网络。
  - 网络配置：50个区域（zones），500条路段（links）。
  - 观测数据由真实参数运行模拟并添加 Gaussian noise 生成。
- **Control 实验**：
  - 使用基于 **Frank-Wolfe算法** 的动态交通分配（DTA）模拟器。
  - 简化概念验证网络，用于测试 latent-space control 机制。

> 所有实验均在配备 NVIDIA V100 GPU 的集群上完成，使用 TensorFlow 实现。

### 🧪 实验设置与评估指标

#### Calibration 阶段
| 指标 | 定义 |
|------|------|
| **NRMSE**（Normalized Root Mean Square Error） | 衡量模拟输出与观测值之间的误差 |
| **Reconstruction Accuracy** | 解码器还原输入参数的能力 |
| **Goodness-of-fit** | 模拟输出对实际条件的拟合程度 |
| **Time (h)** | 总耗时（小时） |

#### Control 阶段
| 指标 | 定义 |
|------|------|
| **System-wide Travel Time Reduction** | 全网总出行时间下降百分比 |
| **Average Network Delay** | 平均延误时间 |
| **Queue Accumulation** | 排队长度峰值 |
| **Congested-link Ratio** | 拥堵路段占比 |

#### 基线方法对比
- **Calibration Baselines**：
  - Standard Bayesian Optimization（标准 BO）
  - Active Subspaces + BO（降维后 BO）
- **Control Baselines**：
  - No-control baseline（无干预路由）
  - Raw state DQN（不使用 latent 表示）

#### 消融实验（Ablation Study）
分析以下组件的影响：
- 是否启用 latent compression
- 是否在 RL 中共享 latent state
- 是否使用 Prioritized Replay

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### Table I: Calibration 方法比较（Benchmark Network）

| Method | Dim | Samples | Error (NRMSE) | Time (h) |
|-------|-----|--------|----------------|-----------|
| Bayesian Opt. | 50 | 500 | 0.152 | 48.2 |
| Active Subspaces + BO | 8 | 300 | 0.098 | 28.7 |
| **MLP-AE (Ours)** | **6** | **200** | **0.064** | **18.3** |

> ✅ **优势明显**：仅用 200 次模拟（< 50% 样本量），达到最低误差（↓58% vs BO，↓35% vs AS+BO），节省 62% 时间。

#### Table II: 消融实验结果

| Configuration | NRMSE | Travel Time Reduction |
|---------------|--------|--------------------------|
| **Full Framework** | **0.064** | **51%** |
| Without Latent Compression | 0.089 | 38% |
| Without Shared Latent RL State | 0.081 | 41% |
| Without Prioritized Replay | 0.074 | 46% |

> 🔍 发现：
> - 移除 latent compression 导致误差上升 39%，说明降维有效性；
> - 不共享 latent state 使控制效果下降 10 个百分点，证明跨模块信息复用的重要性；
> - Prioritized Replay 提升训练稳定性，贡献约 5% 性能增益。

#### 控制性能表现
- **系统级旅行时间减少达 51%**（见 Fig. 2 学习曲线）；
- 平均网络延迟 ↓34%，最大排队积累 ↓27%，拥堵链路比例 ↓31%；
- 控制器展现出智能行为：
  - **预测性 rerouting**：提前重定向以避免即将发生的拥堵；
  - **自适应调度**：调整出发时间平滑需求高峰。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **城市交通系统存在低维内在结构**：
   - 尽管输入输出空间高维，但可通过 deep autoencoder 学习有效的 latent manifold。
2. **共享 latent space 可桥接 calibration 与 control**：
   - 校准中学到的知识可直接赋能控制策略，提升其泛化性和鲁棒性。
3. **深度元模型显著提升优化效率**：
   - 在有限计算预算下，本文方法在精度和速度上全面超越传统方法。
4. **latent-aware RL 更稳定高效**：
   - 相比 raw state 输入，基于 simulator-calibrated latent features 的策略学习更快、性能更高。

### ⚠️ 局限性
- 当前实验基于 **benchmark 网络和理想观测假设**，尚未在真实大城市网络中验证；
- 假设 **完全可观测性**（full observability），现实中传感器稀疏可能限制应用；
- latent space 的 **可解释性不足**，不利于政策制定者理解和信任；
- 对 **非平稳环境**（non-stationary demand, infrastructure changes）适应能力有待加强。

### 🔮 未来工作方向
- 在 **大规模都市区网络** 上验证框架的 scalability；
- 探索 **transfer learning** 能力，实现跨城市的模型迁移；
- 引入 **partially observable MDP (POMDP)** 和 RNN 结构应对感知缺失；
- 开发 **online adaptation 机制** 应对动态变化；
- 提升 **latent representation 和 control policy 的可解释性**，支持实际部署；
- 扩展至多模式交通（multimodal mobility）、CAV 协调等场景。

---

## ✅ 总结一句话
> 本文通过构建一个**共享 latent-space 框架**，首次实现了从高维仿真器校准到深度强化学习控制的统一路径，在样本效率、计算成本和控制性能上均取得显著突破，为智能交通系统的“规控一体”提供了全新范式。

</details>

---

### 12. [One Prompt Does Not Fit All: Self-Meta-Evolve for Personalized Information Extraction](https://arxiv.org/abs/2609.21626)

**Authors**: Hongliang Li, Lu Wang, Yong Xu, Hanyang Chen, Zhitao Hou, Xiaoting Qin, Song Ge, Qingwei Lin, Dongmei Zhang  
**Category**: cs.AI  
**Published**: 2026-09-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.21626v1  

#### Abstract
Large language models (LLMs) are increasingly deployed for enterprise information extraction (IE), where the same document must be reorganized differently for each user. Existing prompt optimization methods, however, rely on a single prompt optimized against a global objective, which is misaligned w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题**: *One Prompt Does Not Fit All: Self-Meta-Evolve for Personalized Information Extraction*

---

## 1. 主要贡献和创新点

### ✅ 解决的问题
传统的大语言模型（LLM）信息抽取（Information Extraction, IE）系统依赖于一个**全局共享的静态 prompt**，无法满足企业环境中不同角色用户的个性化需求。例如：
- 项目经理关注任务截止日期和负责人；
- 财务分析师更关心预算项和风险指标。

这种“千人一面”的设计导致输出与用户实际偏好严重不匹配，需要大量人工后处理。本文指出：**“正确”的抽取结果是用户相关的（user-relative correctness）**，而非存在统一的“ground truth”。

### 🚀 提出的新方法：Self-Meta-Evolve
提出一种**分层式的个性化 prompt 自适应框架** —— **Self-Meta-Evolve**，其核心思想是：
> “每个用户拥有专属 prompt，并通过交互反馈持续优化。”

该框架包含两个循环：
- **内环（Inner Loop）**：基于 AI User 的 persona-conditioned 反馈，对当前用户的 structured prompt 进行编辑。
- **外环（Outer Loop）**：从多个用户的成功优化轨迹中提炼模式，进化 **meta-prompt**（即“如何修改 prompt 的策略”），实现跨用户知识迁移。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | Self-Meta-Evolve |
|------|--------|----------------|
| **优化目标** | 单一全局目标（global objective） | 每用户个性化目标（per-user adaptation） |
| **反馈来源** | Ground-truth 标签或固定历史 | 动态生成的 persona-conditioned 批评 |
| **可扩展性** | 需为每个用户重新训练/搜索 | 外环积累经验提升后续适应效率 |
| **结构化程度** | 黑盒文本 prompt | 结构化 JSON prompt（支持细粒度编辑） |

---

## 2. 核心实验方法和设置

### 📚 数据集：Persona-Driven Enterprise Benchmark
作者构建了一个全新的、可复现的企业级个性化 IE 基准测试集，包含：
- **292 个模拟企业用户 persona**
- 每个 persona 包含：
  - 角色描述（如 Senior PM, Patent Counsel）
  - 文档分布（email/chat/report 等类型比例）
  - 信息偏好（3–6 条声明式规则，如“只提取有明确负责人和截止日期的任务”）

#### 构建流程（四阶段）：
1. **种子角色**：来自公开职业数据库 [O*NET](https://www.onetcenter.org/)（过滤出 412 个企业相关职业）
2. **LLM 扩展**：用 GPT-5.1 将简略职业描述扩展为结构化 persona JSON
3. **质量控制**：去重、一致性评分（≥4/5）、偏好具体性检查
4. **文档合成**：为每个 persona 合成 8–12 篇企业文档（邮件、聊天记录等），**不泄露偏好信号**

#### 数据划分：
- 主基准：`train/dev/eval = 60%/20%/20%`（按 ID 固定划分）
- 专用测试集：
  - STEM-Personas (`N=50`)
  - Humanities-Personas (`N=50`) —— 用于检验跨领域泛化能力

> ⚠️ 所有数据均为 LLM 合成，无真实企业数据，保障隐私。

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Success Rate (SR)** | 达标用户占比：部署损失 ≤ 阈值 `T`（设为最强基线 ProTeGi 在 dev 上的中位数损失，归一化为 0.75） |
| **Mean Loss (ML)** | 平均部署损失 |
| **SR@t** | 前 `t` 次迭代内的累计成功率（衡量收敛速度） |
| **API 成本** | 每轮 token 开销、总成本、每百分点增益的成本效率 |

### 🔁 基线方法对比
共比较 7 种主流 prompt 优化方法：
| 方法 | 类型 |
|------|------|
| **ProTeGi** | Gradient-inspired（伪梯度法） |
| **OPRO**, **APE** | Search-based（元 prompt 搜索） |
| **EvoPrompt** | Evolutionary（遗传算法风格） |
| **Bandit-UCB** | Multi-armed bandit 探索机制 |
| **MetaSPO**, **Pareto Prompt** | Meta-learning / Pareto 优化变体 |

---

## 3. 主要实验结果和性能指标

### 📈 主要性能表现（在 hold-out eval set, `N=59`）

| 方法 | **SR (%)** | **ML ↓** | **SR@2** | **SR@4** | **SR@8** |
|------|------------|----------|----------|----------|----------|
| ProTeGi (best baseline) | 61.02 | 0.754 | 0.390 | 0.475 | 0.542 |
| OPRO | 45.8 | 1.124 | 0.271 | 0.339 | 0.424 |
| APE | 33.9 | 1.423 | 0.203 | 0.271 | 0.305 |
| **Self-Meta-Evolve (ours)** | **74.58** | **0.512** | **0.525** | **0.610** | **0.678** |

> ✅ **绝对提升 13.56 pp**（vs. ProTeGi），相对提升约 **22.2%**
>
> ✅ **仅 2 步即达 52.5% SR**，已超过多数基线最终性能
>
> ✅ 平均损失下降 **32%**

#### 统计显著性：
- vs. ProTeGi: `p = 0.0038`
- vs. Self-Frozen（禁用外环）: `p = 0.0009`

### 🔄 跨 backbone 鲁棒性验证（Table 2）
在 5 个不同 backbone（GPT/Claude 系列）上验证，**Self-Meta-Evolve 始终领先 11.87–15.25 pp**，说明效果非特定模型耦合所致。

### 🧪 消融实验（Ablation Study）

| 变体 | SR | ML | 说明 |
|------|-----|-----|------|
| Full Model | 74.6 | 0.512 | 完整框架 |
| - Outer Loop | 57.6 | 0.865 | ❌ 最大降幅（-17.0 pp）→ 外环至关重要 |
| - Success Buffer | 66.1 | 0.685 | 缓冲区帮助经验回放 |
| - Edit Plan | 69.5 | 0.621 | 结构化编辑计划提升样本效率 |

> 💡 结论：**meta-evolution 是后期性能跃升的关键驱动力**

### 📐 效率与成本分析
尽管每次迭代 token 消耗更高（12.4K vs. ProTeGi 的 6.5K），但 **每百分点 SR 增益的成本最低（4.98K tok/%SR）**，优于其他非平凡优化器。

### 🌍 跨领域泛化能力（Table 4）
| 方法 | STEM-SR | Humanities-SR | 下降幅度 △ |
|------|---------|---------------|-----------|
| ProTeGi | 62.0 | 54.0 | 8.0 |
| OPRO | 48.0 | 34.0 | 14.0 |
| **Self-Meta-Evolve** | **76.0** | **72.0** | **4.0** |

> ✅ 外环提炼的编辑策略具有更强的**跨域迁移能力**，尤其在人文类偏好（更语境化、模糊）场景下优势明显。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **“One Prompt Fits All” 范式失效**：企业级 IE 必须考虑用户异质性（user heterogeneity）。
2. **Self-Meta-Evolve 显著优于现有 prompt 优化方法**：
   - SR 提升 **13.56 pp**
   - 收敛更快（SR@2 = 52.5%）
   - 泛化更强（跨领域退化最小）
3. **双盲人类评估验证有效性**：
   - 在 20 名真实专业人士参与的配对比较中，**自适应 prompt 获胜率达 71%**
   - 合成 persona 与真实职业资料无显著差异（Cohen’s Kappa = 0.71）
4. **meta-evolution 是关键**：外环通过提炼成功编辑路径，实现了跨用户的知识共享与策略升级。

### ⚠️ 局限性（Limitations）
1. **依赖 AI User 反馈**：训练时使用 LLM 模拟用户反馈，可能存在“LLM judge bias”，未必完全反映真实人类偏好。
2. **persona 覆盖范围有限**：基于美国 O*NET 数据库，主要面向英语企业环境；文化、语言、行业特异性未充分覆盖。
3. **仅靠 prompt 无法弥补知识缺口**：部分失败源于 LLM 缺乏领域知识（如专利法律术语），需结合 retrieval 或 tool-augmentation。

### 🔮 未来工作方向
- 引入 **real-user feedback loop** 实现持续校准
- 构建多语言、本地化的 persona 生成管道
- 探索 **Retrieval-Augmented Prompt Adaptation** 以解决 domain knowledge gaps
- 扩展至多模态企业文档（PDF、会议视频等）

---

## 总结（TL;DR）
> Self-Meta-Evolve 提出了一种**面向企业个性化信息抽取的双循环 prompt 自进化框架**，首次将 per-user adaptation 与 meta-strategy evolution 结合，在新发布的 292-persona benchmark 上取得 **74.58% Success Rate**，大幅超越现有方法。实验证明其不仅性能优越，且具备良好收敛性、鲁棒性和跨域泛化能力，为下一代企业智能助手提供了重要范式参考。

</details>

---

### 13. [Understanding LLM Quantization through Activation-Guided Compensation and Orthogonal Residuals](https://arxiv.org/abs/2609.21450)

**Authors**: Yamato Narita, Issei Sato  
**Category**: cs.LG  
**Published**: 2026-09-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.21450v1  

#### Abstract
Post-training weight-activation quantization reduces the memory and inference costs of large language models, but aggressive W4A4 quantization remains difficult because activation outliers degrade effective quantization resolution. Although weight optimization, channel-wise scaling, and orthogonal r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Understanding LLM Quantization through Activation-Guided Compensation and Orthogonal Residuals

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于**低比特权重量化（W4A4）**在大型语言模型（LLM）中的挑战，尤其是**激活值中的异常通道（persistent channel-wise outliers）**导致量化分辨率下降的问题。这些异常通道会反复主导每token的量化尺度，降低对普通激活值的有效表示能力。

尽管已有多种方法（如权重优化、通道缩放、正交旋转）被提出缓解该问题，但它们各自解决的误差成分及其相互关系尚不明确。

### 提出了什么新方法或新思路
论文提出了一个**局部权重量化误差的精确分解框架**，将量化误差分解为两个正交部分：

- **Activation-Guided Weight Compensation (AGWC)**：可通过权重优化（如GPTQ、GPTAQ）补偿的部分。
- **Orthogonal Residual**：无法通过权重优化改变、必须依赖变换设计（如旋转、缩放）来抑制的部分。

在此基础上，进一步分析了正交残差项，并推导出其上界由两类激活量决定：
- **Persistent CO Quantity**（$J_{\text{co}}$）：来自持续存在的异常通道。
- **Regular Quantity**（$J_{\text{reg}}$）：来自其余常规激活变化。

这一分解为理解不同技术的作用提供了统一视角：
- 权重优化仅能减少 AGWC 项；
- 旋转与缩放则需针对正交残差进行设计。

基于此理论，作者提出了一套无需反向传播的配置方案——**L2-SmoothRot**，包含三个核心组件：
1. **Signed Online Rotation (SOR)**：在线Hadamard旋转前加入随机符号矩阵，以破坏异常通道间的建设性干扰。
2. **Sign Sampling (SS)**：从多个随机符号模式中选择最优者，提升鲁棒性。
3. **L2 Channel Scaling (L2S)**：基于第二矩平衡推导出的缩放规则，用于均衡激活与权重侧的能量分布。

此外，论文还解释了为何 **L2 scaling** 和 **SmoothQuant-style $L_\infty$ scaling** 是同一目标的不同松弛形式，前者保留了通道间统计信息，后者进一步简化为最大值近似。

### 相比现有方法的优势
- **理论清晰性**：首次明确区分了“可由权重补偿”与“需靠变换设计”的误差成分，为组合多种量化技术提供原则性指导。
- **无需训练/反向传播**：所提方法完全基于校准数据选择变换参数，避免耗时的梯度调优过程。
- **性能竞争力强**：在多个主流LLM上达到与需要梯度训练的 **SpinQuant** 相当甚至更优的表现。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准与验证数据**：
  - **WikiText-2 (WT2)**：用于校准激活统计、旋转选择及超参调优。
  - **C4 validation subset**：用于评估泛化性能。
- **零样本任务测试集**（共6项）：
  - PIQA
  - ARC-Easy (ARC-E)
  - ARC-Challenge (ARC-C)
  - HellaSwag (HS)
  - WinoGrande (WG)
  - LAMBADA (LB)

### 实验设置和评估指标
- **量化配置**：W4A4 + KV4（即权重、激活、Key/Value均4-bit）
  - 权重：按输出通道对称量化（per-output-channel symmetric quantization）
  - 激活：动态每token对称量化（dynamic per-token symmetric quantization）
  - K/V：每token每头非对称量化
  - 激活与K/V剪裁比例分别为 0.9 和 0.95
- **权重量化器**：统一采用 **GPTAQ**（无微调、高效）
- **校准细节**：
  - GPTAQ 使用 128 个 WT2 训练样本
  - 缩放统计使用 512 个未变换模型的 WT2 样本
  - 旋转训练参考 SpinQuant 协议：冻结16-bit权重 + 4-bit激活/KV，训练100步后用GPTAQ量化

### 基线方法对比
- **QuaRot**：固定Hadamard旋转，无符号扰动
- **SmoothRot**：结合Hadamard旋转与SmoothQuant风格的 $L_\infty$ 缩放
- **SpinQuant**：通过梯度训练学习最优旋转（当前SOTA之一）

所有方法均使用相同的 GPTAQ 权重量化流程，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（WikiText-2 PPL / 平均零样本准确率）

| Model             | FP Ref (PPL/Avg%) | QuaRot     | SmoothRot  | SpinQuant  | **L2-SmoothRot** |
|-------------------|-------------------|------------|------------|------------|------------------|
| Llama-7B          | 5.67 / 69.39      | 6.12 / 66.17 | 6.13 / 66.16 | 6.19 / 66.18 | **6.09 / 66.67** |
| Llama-13B         | 5.05 / 71.78      | 5.42 / 69.49 | 5.42 / 69.09 | 5.44 / 69.31 | **5.37 / 69.91** |
| Llama2-7B         | 5.47 / 69.82      | 5.96 / 66.32 | 5.98 / 66.45 | 6.06 / 65.98 | **5.94 / 66.59** |
| Llama2-13B        | 4.86 / 72.57      | 5.24 / 69.87 | 5.26 / 70.15 | 5.26 / 70.23 | **5.23 / 70.50** |
| Llama3-8B         | 5.94 / 73.30      | 7.45 / 65.71 | 7.57 / 65.85 | 7.33 / 68.36 | **7.20 / 68.10** |
| Llama3.2-1B       | 9.41 / 59.80      | 13.81 / 49.58| 14.13 / 48.63| 13.33 / 50.37| **12.92 / 50.71** |
| Llama3.2-3B       | 7.55 / 68.20      | 9.27 / 61.12 | 9.35 / 59.91 | 9.19 / 62.03 | **9.08 / 60.94** |
| Mistral-7B-v0.3    | 5.35 / 73.73      | 5.73 / 70.55 | 5.77 / 69.80 | 5.74 / 71.17 | **5.71 / 71.00** |

> ✅ **L2-SmoothRot 在全部8个模型上实现了最低的 WikiText-2 PPL，在5个模型上取得最高平均准确率**

### 与基线方法的对比结果
- **相比 QuaRot**：显著优于原始旋转方法，尤其在 Llama3 系列上表现突出（PPL ↓1+），说明 SOR + SS + L2S 组合有效。
- **相比 SmoothRot**：全面超越基于 $L_\infty$ 缩放的方法，验证了 L2 缩放更具优势。
- **相比 SpinQuant**：
  - 在 **Llama3-8B** 上 PPL 更低（7.20 vs 7.33），准确率接近（68.10 vs 68.36）
  - 在 **Mistral-7B** 上 PPL 更低（5.71 vs 5.74），准确率略低（71.00 vs 71.17）
  - 总体性能**具有竞争力且无需任何梯度训练**

### 消融实验结果（Llama3-8B, W4A4）

| Method Configuration           | PPL ↓ |
|-------------------------------|--------|
| QuaRot baseline               | 7.45   |
| + SOR (signed online rot)     | 7.30   |
| + SS (sign sampling)          | 7.40   |
| + L2S (L2 scaling)            | 7.32   |
| **Full L2-SmoothRot (all)**   | **7.20** |

- **SOR 贡献最大**：单独添加即可带来明显增益，说明随机符号能有效抑制异常通道的构造性干扰。
- **三者互补**：联合使用效果最佳，表明 $J_{\text{co}}$ 与 $J_{\text{reg}}$ 需分别处理。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **误差可解耦**：本地权重量化误差可分为 **AGWC** 与 **Orthogonal Residual**，前者可通过权重优化消除，后者必须通过变换设计缓解。
2. **旋转的本质是控制干扰**：固定Hadamard旋转可能导致异常通道在某些坐标上叠加放大；引入**随机符号**可打破这种一致性，实现期望意义上的能量去相关。
3. **符号采样提升稳定性**：通过对多个随机符号模式进行筛选（基于校准集PPL），可以进一步提高性能鲁棒性。
4. **L2 scaling 比 $L_\infty$ scaling 更合理**：它是对正则项 $J_{\text{reg}}$ 第二矩界的直接优化结果；而 $L_\infty$ scaling 是对其的双重松弛（谱范数 → Frobenius → 最大值），虽简单但损失更多信息。
5. **无需梯度也能达到SOTA水平**：提出的 L2-SmoothRot 完全基于解析推导与校准选择，在多个模型上媲美甚至超越需梯度训练的 SpinQuant。

### 方法的局限性
- 当前分析基于**单层线性变换**，未显式建模深层传播误差累积。
- 对“异常通道”的定义依赖经验观察与稀疏假设，尚未完全自动化检测机制。
- 所有变换仍限于正交类（Hadamard + Diagonal Sign），未探索更广义的等价变换空间。

### 未来工作方向
- 将误差分解框架扩展至整个Transformer块或端到端路径。
- 探索自适应识别与处理异常通道的机制。
- 结合本文理论与OmniQuant等可学习变换，发展兼具效率与表达力的新一代PTQ方法。
- 推广至其他模态（如视觉、多模态）的大模型量化场景。

--- 

> 📌 **总结一句话**：  
> 本论文通过提出 **AGWC-Residual 分解框架**，为LLM量化中的权重优化、旋转、缩放等操作提供了统一的理论基础，并据此设计出无需反向传播的高性能配置 **L2-SmoothRot**，在实践中达到了与梯度训练方法相媲美的效果，推动了PTQ方法从“启发式拼凑”走向“原理驱动设计”。

</details>

---

### 14. [Multi-Domain Clustering via Measure Quantization](https://arxiv.org/abs/2609.21664)

**Authors**: Rafael Pereira Eufrazio, Eduardo Fernandes Montesuma, Charles Casimiro Cavalcante  
**Category**: cs.LG  
**Published**: 2026-09-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.21664v1  

#### Abstract
Clustering is a fundamental task in data analysis, typically addressed through centroid-based methods such as K-means. In this work, we present a general framework for multi-domain clustering via measure quantization: given samples from multiple domains, we learn a shared set of cluster prototypes b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Multi-Domain Clustering via Measure Quantization》总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**多域聚类（multi-domain clustering）**中的分布偏移（distribution shift）问题。在现实应用中，来自不同域的数据（如不同风格、光照、设备采集的图像或音频）具有不同的统计特性，传统聚类方法（如 K-Means）直接对混合数据进行聚类会因忽略域间差异而导致性能下降。

### 提出的新方法与新思路
提出了一种基于**测度量化（measure quantization）**的通用多域聚类框架，其核心思想是：
- 将每个域的数据视为一个经验概率测度 $ \mu_k $
- 学习一组共享的聚类原型（centroids），构成另一个经验测度 $ \nu $
- 通过最小化所有域与其之间某种**概率度量（probability metric）**的距离来优化原型：
  $$
  \nu^* = \arg\min_{\nu \in \text{Emp}_C(\mathbb{R}^d)} \sum_{k=1}^K \mathcal{D}(\mu_k, \nu)
  $$
  其中 $\mathcal{D}$ 可以是 **Sinkhorn divergence** 或 **Maximum Mean Discrepancy (MMD)**。

#### 创新点包括：
- **统一的概率视角建模多域聚类**：将多域聚类形式化为多个输入测度向低支撑集测度的联合逼近问题。
- **支持多种概率度量**：不仅限于 Wasserstein 距离，还探索了 MMD 等核方法，增强了灵活性。
- **可扩展的 mini-batch 优化算法**：采用梯度下降更新原型，并利用 mini-batch 近似梯度，显著降低内存和计算开销，适用于大规模数据。
- **两种分配策略**：
  - **Greedy Assignment**：最近邻分配（nearest centroid）
  - **OT Assignment**：基于最优传输（optimal transport）计划的协同分配，考虑样本间的全局耦合关系。

### 相比现有方法的优势
- **优于经典聚类方法**：相比忽略域差异的 pooled K-Means、Spectral Clustering 等，在多个跨域任务上表现更优。
- **优于其他多域方法**：提出的 Sinkhorn-based 方法在多数数据集上优于现有的多级 Wasserstein 聚类方法（如 MWMS）。
- **良好的可扩展性**：在超大规模数据集 DomainNet（近60万样本）上仍保持高效且性能领先，而传统方法难以处理。
- **理论基础强**：建立在 Optimal Transport 和测度理论之上，提供了更丰富的几何结构用于建模数据分布。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖图像、音频和传感器三大模态共 **6 个基准数据集**，具体如下：

| 数据集 | 类型 | # 样本 | # 类别 | # 域 |
|--------|------|--------|--------|-------|
| **Office 31** | 图像 | 4,110 | 31 | 3 |
| **Caltech-Office 10 (C10)** | 图像 | 2,533 | 10 | 4 |
| **Office-Home (OH)** | 图像 | 15,500 | 65 | 4 |
| **DomainNet** | 图像 | 586,575 | 345 | 6 |
| **TAU Urban Scenes** | 音频 | 20,800 | 10 | 10 |
| **Tennessee Eastman Process (TEP)** | 传感器数据 | 17,289 | 29 | 6 |

特征提取方式：
- 图像：ResNet-50 / ResNet-101 / DeCaf 提取特征（ImageNet预训练，无微调）
- 音频：PANN backbone
- 传感器：时间序列的一阶与二阶统计量

### 实验设置与评估指标
- **硬件环境**：单机（AMD EPYC CPU, NVIDIA L4 GPU, 47GB RAM）
- **实现工具**：PyTorch + PythonOT
- **评估指标**（按域平均后计算）：
  - **Hungarian Accuracy (Hung. Acc.)**：通过最优传输匹配预测簇与真实标签后的准确率
  - **Adjusted Rand Index (ARI)**：衡量两个聚类结果的一致性
  - **Normalized Mutual Information (NMI)**：衡量预测与真实聚类的信息共享程度
  - **Geometric Mean (GM)**：上述三项指标的几何平均，作为综合性能指标

### 基线方法对比
分为两类：

#### Pooled Methods（忽略域差异）：
- K-Means
- Spectral Clustering
- Ward
- BIRCH

#### Multi-domain Methods：
- **MWMS** [20]：基于 Wasserstein 的多级聚类方法
- **MMD (ours)**：本文提出的基于 MMD 的变体
- **Sinkhorn (ours)**：本文主推的基于 Sinkhorn divergence 的方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）
在五个主要数据集上的综合性能（GM）排名如下（越低越好）：

| 方法 | 平均排名（GM） |
|------|----------------|
| **Sinkhorn (ours)** | **1.4** ✅ |
| MWMS | 3.4 |
| Ward | 4.2 |
| K-Means / BIRCH | 4.6 |
| Spectral | 5.4 |
| MMD (ours) | 4.4 |

> ✅ **Sinkhorn 方法在所有数据集上取得最佳或接近最佳的 GM 表现**

#### 典型性能示例（GM值）：
- **Caltech-Office 10**：Sinkhorn 达到 **0.79**，远超 K-Means (0.57) 和 MWMS (0.64)
- **Office-Home**：Sinkhorn 为 **0.62**，优于 MWMS (0.57)
- **TEP**：Sinkhorn 为 **0.64**，显著高于 pooled 方法 (~0.18)

### 与基线方法的对比结果
- 所有**多域方法**均优于 **pooled 方法**，说明建模域结构至关重要。
- 在多域方法中，**基于 Wasserstein 几何的方法（Sinkhorn 和 MWMS）整体优于 MMD**。
- **Sinkhorn 方法全面超越 MWMS**，验证了所提框架的有效性。

### 消融实验结果

#### （1）超参数分析（Figure 3）
- **Sinkhorn 参数 $ \epsilon $** 影响显著：$ \epsilon = 10^{-3} $ 时性能最好（GM≈0.66），过大（如 $ 10^{-1} $）导致过度平滑，性能下降至 0.49。
- **距离函数 $ p $ 和度量 $ m $**（Euclidean vs Cosine）影响较小（<0.01 差异）。
- **MMD 核选择影响大**：从 RBF 切换到 Riesz 核可使 GM 提升 0.19（0.31→0.50），表明核设计是关键杠杆。

#### （2）分配机制比较（Figure 4）
- 在大多数数据集（OH, O31, C10, TAU）上，**OT assignment 与 nearest centroid 效果相近**（差距 <0.01 GM）。
- 但在 **TEP 数据集**（类别最多、不平衡最严重）上，**OT assignment 明显优于 nearest centroid（0.65 vs 0.54）**，提升达 +0.11 GM。
  > 表明在复杂、难分的场景下，基于最优传输的协同分配能更好地区分模糊簇。

#### （3）可扩展性测试（Figure 5，DomainNet）
- 在 **DomainNet（586k 样本）** 上：
  - **Sinkhorn (ours)** 完胜 mini-batch K-Means：
    - Acc: **0.27 vs 0.23**
    - NMI: **0.45 vs 0.40**
    - ARI: **0.16 vs 0.10**
  - MMD 版本在此任务上表现较差（Acc=0.14, NMI=0.32, ARI=0.03）

---

## 4. 关键结论和发现

### 主要发现
1. **Sinkhorn-based 多域聚类框架显著优于传统及现有多域方法**，尤其在分布差异明显的真实世界数据上。
2. **Wasserstein 几何（特别是 Sinkhorn divergence）比 MMD 更适合建模多域数据分布**，提供更丰富的结构信息。
3. **mini-batch 优化策略有效实现了大规模可扩展性**，可在数十万样本级别保持高性能。
4. **OT-based 协同分配在高维、多类、不平衡数据中更具优势**，揭示了全局样本耦合的重要性。
5. 超参数中，**Sinkhorn 正则化系数 $ \epsilon $ 和 MMD 核函数选择对性能影响显著**，需仔细调参。

### 方法的局限性
- 当前假设所有域共享相同的特征空间；若域间特征不可比（如图像 vs 文本），无法直接应用。
- 对 $ \epsilon $ 和核函数敏感，可能增加实际部署中的调参成本。
- 理论收敛性尚未严格证明，依赖经验验证。

### 未来工作方向
- **理论分析**：研究算法的收敛性质与泛化能力。
- **推广到非对齐空间**：引入 **Gromov-Wasserstein distance** 来处理不同度量空间下的多域聚类问题。
- 探索更多类型的概率度量和正则化策略。
- 应用于更多实际场景，如联邦学习、异常检测等。

</details>

---

### 15. [Beyond Kinematics: Benchmarking Simulation Fidelity for Muscle-Driven Imitation Learning](https://arxiv.org/abs/2609.21909)

**Authors**: Ayah G. Ahmad, Claire E. Borden, Maegan Tucker  
**Category**: cs.LG  
**Published**: 2026-09-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.21909v1  

#### Abstract
In this work, we conduct a systematic comparison of two state-of-the-art motion-imitation reinforcement learning (MIRL) pipelines, one built on SCONE/HyFyDy and one built on MuJoCo/MyoSim. HyFyDy emphasizes physiological realism through detailed musculotendon modeling, while MuJoCo prioritizes compu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Beyond Kinematics: Benchmarking Simulation Fidelity for Muscle-Driven Imitation Learning**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
当前用于 **motion-imitation reinforcement learning (MIRL)** 的 musculoskeletal simulation pipelines 多数仅以 **kinematics（运动学）** 为优化目标，即关注关节角度和身体姿态是否匹配真实人类动作。然而，在 **robotic assistive device（如外骨骼、假肢）设计** 中，更关键的是预测 **muscle activation patterns（肌肉激活模式）** 和 **metabolic cost（代谢消耗）** 等生理层面的结果。

本文指出：尽管现有模拟器（如 MuJoCo/MyoSim 和 SCONE/HyFyDy）能高保真地复现人体运动轨迹，但其在 **neuromuscular behavior（神经肌肉行为）建模上的准确性尚未系统验证**。因此，缺乏对肌肉激活等生理输出的可信度评估，限制了其在辅助设备优化中的应用。

---

### ✅ **提出了什么新方法或新思路**
本研究首次提出并实施了一个 **跨平台、基于实测 EMG 数据的 benchmarking framework**，用于评估不同 MIRL 框架在 **肌肉激活预测能力** 上的表现。

具体创新包括：
- 构建统一的比较基准，使用 **同步的 motion-capture 与 surface EMG 数据** 作为 ground truth。
- 首次直接对比两个最先进的 MIRL 框架：
  - **HyFyDy-IL**（基于 SCONE + HyFyDy，强调生理真实性）
  - **MuscleMimic**（基于 MuJoCo + MyoSim，强调计算效率与可扩展性）
- 在多个维度进行系统性消融分析：
  - 模型维度（2D vs 3D）
  - 是否个性化建模（generic vs subject-specific）
  - 不同 reward 结构的影响

---

### ✅ **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **评估标准** | 超越 kinematics，引入 **EMG 对齐度** 作为核心评价指标，更具临床意义 |
| **公平性** | 统一训练数据源（Scherpereel et al. dataset）、预处理流程和评估协议 |
| **可复现性** | 公开提供 **OpenSim-to-MuscleMimic 的转换工具** 和一个训练好的 **subject-specific HyFyDy 模型**，促进后续研究 |
| **系统性** | 多变量控制实验设计，揭示 simulator fidelity、personalization、dimensionality 的独立影响 |

---

## 2. **核心实验方法和设置**

### 📁 **使用的数据集**
- 主要数据来源：**Scherpereel et al. 开源生物力学数据集 [30]**  
  包含：
  - 运动捕捉（motion-capture）数据（140秒步行片段，来自受试者 AB08）
  - 同步采集的 **surface EMG 信号**
  - 多种步行速度（0.6–2.2 m/s），支持速度分箱分析

---

### ⚙️ **实验设置**
#### **对比框架**
| 框架 | 物理引擎 | 控制算法 | 特点 |
|------|--------|---------|------|
| **HyFyDy-IL** | SCONE / HyFyDy (CPU) | SAC (off-policy) | 高生理保真度，Millard 肌腱模型，弹性肌腱、变羽角 |
| **MuscleMimic** | MuJoCo / MyoSim (GPU) | PPO (on-policy, JAX) | 高效并行训练，简化 Hill-type 模型，刚性肌腱 |

#### **模型对齐策略**
为实现公平比较，作者对两类模型进行了结构对齐：
- **MM240-2D / MM240-3D**：将 MyoFullBody 改造为与 H1090/H2190 可比的结构（移除手臂、融合躯干、肌肉群映射）
- 最终肌肉数量统一至 **240 肌肉（vs HyFyDy 的 90）**，保留主要功能组

#### **五种训练模型**
1. `H1090`：HyFyDy, 2D, generic model  
2. `H-AB08`：HyFyDy, 2D, subject-specific（缩放参数匹配 AB08）  
3. `H2190`：HyFyDy, 3D, default model  
4. `MM240-2D`：MuscleMimic, 2D  
5. `MM240-3D`：MuscleMimic, 3D  

---

### 📊 **评估指标**
对每个 gait cycle 内的速度 bin 进行 pooling 分析，计算以下指标：

| 指标 | 定义 | 意义 |
|------|------|------|
| **RMSE** | Root Mean Squared Error between predicted and measured muscle activation | 数值越低越好，反映幅值偏差大小 |
| **Pearson r** | 相关系数，衡量波形形状一致性 | 越接近 1 表示时间动态越相似 |

同时报告 **joint kinematics** 和 **muscle activation** 的 pooled performance。

---

### 🔁 **基线方法对比**
- **HyFyDy-IL vs MuscleMimic**（主对比）
- **Generic vs Personalized Model**（H1090 vs H-AB08）
- **2D vs 3D Models**（H1090 vs H2190；MM240-2D vs MM240-3D）
- **Reward Modification Study**：尝试调整 MuscleMimic 的 reward 权重以匹配 HyFyDy-IL 的 effort penalties

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据（Table II & Fig. 5）**

#### ✅ **总体肌肉激活预测性能（pooled across 8 muscles）**

| 模型 | RMSE ↓ | Pearson r ↑ |
|------|-------|------------|
| **H1090 (HyFyDy-IL, 2D)** | **0.164** | **0.40** |
| **MM240-2D (MuscleMimic, 2D)** | 0.344 | 0.11 |
| **H-AB08 (scaled HyFyDy)** | 0.192 | 0.20 |
| **H2190 (3D HyFyDy)** | 0.240 | 0.26 |

> 💡 **结论**：HyFyDy-IL 显著优于 MuscleMimic，尤其在相关性（r）方面。

#### ✅ **典型肌肉表现举例（H1090 vs MM240-2D）**
| 肌肉 | H1090 (RMSE/r) | MM240-2D (RMSE/r) |
|------|----------------|--------------------|
| Medial Gastrocnemius | 0.131 / **0.81** | 0.232 / 0.44 |
| Tibialis Anterior | 0.224 / 0.38 | 0.284 / 0.36 |
| Rectus Femoris | 0.128 / 0.23 | 0.630 / 0.15 |
| Gluteus Maximus | 0.052 / 0.44 | 0.201 / 0.35 |

> 🔍 可见 HyFyDy 更好捕捉 **medial gastrocnemius** 的双峰激活特征。

---

### 🔍 **与基线方法的对比结果**

| 维度 | 发现 |
|------|------|
| **Kinematics 跟踪能力** | 两者均表现良好（pooled r > 0.76），说明都能完成基本动作模仿 |
| **Muscle Activation 预测** | **HyFyDy-IL 明显更贴近 EMG**，尤其在动态趋势上（r 更高） |
| **Personalization 影响** | 缩放后的 H-AB08 性能略差于 H1090，表明简单几何缩放不足以提升预测精度 |
| **Dimensionality 影响** | 3D 模型（H2190, MM240-3D）训练难度显著增加，**MM240-3D 无法稳定行走全程**，H2190 肌肉预测也退化 |
| **Reward 修改尝试** | 尝试将 MuscleMimic 加入类似 effort penalty，导致 policy 完全失败（无法迈步），说明 reward 设计敏感且需精细调参 |

---

### 🔻 **消融实验结果**
- **模型复杂度增加（→3D） → 性能下降**：更高的自由度带来不稳定性，需要更强的正则化或更长训练时间
- **个性化建模未带来增益**：可能因肌肉力-长度特性未随个体调整所致
- **reward 权重迁移不可行**：不同框架间 reward scale 差异大，直接移植会导致训练崩溃

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **HyFyDy-IL 在肌肉激活预测上显著优于 MuscleMimic**  
   得益于其 **high-fidelity musculotendon modeling**（弹性肌腱、变羽角、误差控制积分器），更能反映真实 neuromuscular dynamics。

2. **MuJoCo-based MuscleMimic 虽然训练快、可扩展性强，但在生理真实性上仍有差距**  
   其简化的 Hill-type 模型（刚性肌腱）可能导致肌肉激活动力学失真。

3. **增加模型维度（→3D）反而降低预测准确性**  
   训练难度剧增，policy 难以收敛到稳定步态，提示当前 MIRL 方法在复杂模型上仍面临挑战。

4. **简单的几何缩放不能有效提升个性化模型性能**  
   真正的 subject-specific modeling 应包含肌肉生理参数（如 PCSA、optimal fiber length）的适配。

5. **reward 函数高度敏感**，跨框架迁移需谨慎，目前尚无通用 reward design 原则。

---

### ⚠️ **方法的局限性**
- 实验仅基于一名健康受试者（AB08），泛化性有待验证
- 所有模型均为 healthy gait，未涉及病理步态（如中风、帕金森）
- EMG 数据未完全校准为绝对激活水平，依赖相对趋势比较
- HyFyDy 当前不支持 GPU 并行，训练速度远慢于 MuscleMimic
- MuscleMimic 缺乏成熟的 subject-scaling 工具链（如 MyoConverter 不兼容）

---

### 🔮 **未来工作方向**
1. **开发兼具生理真实性和 GPU 可扩展性的新型模拟器**  
   → 推动 “physiologically plausible + scalable” 的下一代 MIRL 平台

2. **构建标准化 benchmark suite**  
   包括多种人群（年龄、疾病状态）、任务（上下楼梯、跑步）、多模态数据（kinematics + EMG + GRF + metabolic）

3. **探索更鲁棒的 reward design 和 curriculum learning 策略**  
   以应对高维、不稳定模型的训练挑战

4. **集成 assistive devices into simulation**  
   如在模型中加入 exoskeleton 或 prosthesis，实现闭环人机交互仿真，加速控制器开发

5. **推动 open-source model conversion tools**  
   实现 OpenSim ↔ MuJoCo / HyFyDy 的高质量互转，打破生态壁垒

---

> 🏁 **最终结论**：  
> **“Beyond Kinematics” 是迈向真正可用于辅助设备设计的 musculoskeletal simulation 的必经之路。**  
> 当前，**HyFyDy 因其生理保真度更适合科学研究和临床假设验证**，而 **MuscleMimic 更适合大规模策略探索和快速原型开发**。  
> 未来理想系统应融合二者优势——**既 fast 又 faithful**。

</details>

---

### 16. [Information-Gain Rewards over Diversity-Pruned Tests: GT-Anchored Verifier Co-Training for Reliable Code Generation](https://arxiv.org/abs/2609.21208)

**Authors**: Ana Nunez, Peyman Najafirad  
**Category**: cs.AI  
**Published**: 2026-09-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.21208v1  

#### Abstract
Self-play methods that co-train a single language model as both coder and test author promise to move code-generation RL beyond fixed test suites, but they suffer from two coupled pathologies: permissiveness collapse, where pass-rate rewards are maximised by trivial, non-discriminative tests, and co...

---

### 17. [COAL-SQL: Coverage-Guided Augmentation and Failure-Driven Learning for Text-to-SQL Post-Training](https://arxiv.org/abs/2609.20842)

**Authors**: Qifeng Cai, Xuanguang Pan, Hao Liang, Chang Xu, Wentao Zhang  
**Category**: cs.CL  
**Published**: 2026-09-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.20842v1  

#### Abstract
Text-to-SQL translates natural-language questions into executable SQL queries, but open-source large language models still require task-specific post-training for complex, real-world SQL generation. Effective post-training requires both training data that cover the capabilities demanded by the targe...

---

### 18. [PoVD: Efficient Consensus Protocol based on Verifiable Delay Function](https://arxiv.org/abs/2609.21627)

**Authors**: Rui Jiang, Xintong Ling, Bin Cao, Jiaheng Wang, Xiqi Gao, Zhi Ding  
**Category**: cs.DC  
**Published**: 2026-09-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.21627v1  

#### Abstract
Consensus protocols ensure the robustness and scalability of blockchains and decentralized applications built on them. However, existing consensus mechanisms often impose high computational cost or require heavy communication overhead. To address these challenges, we propose proof of verifiable dela...

---

### 19. [Particle Competition and Cooperation for Robust Graph Convolutional Network Learning Under Label Noise](https://arxiv.org/abs/2609.22053)

**Authors**: Fabricio Breve  
**Category**: cs.LG  
**Published**: 2026-09-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.22053v1  

#### Abstract
Graph Convolutional Networks (GCNs) are highly sensitive to label noise, since corrupted supervision can propagate through the graph and degrade learned node representations. This work proposes PCC+GCN, a hybrid framework that uses Particle Competition and Cooperation (PCC) as a graph-based label-re...

---

### 20. [AutoViewMem: Self-Configuring Orthogonal Views for Conversational Long-Term Memory](https://arxiv.org/abs/2609.21940)

**Authors**: Zijie Cao, Xijun Qu, Zhicheng Gu, Xiaoshu Chen, Duanyang Yuan, Yanning Hou, Sihang Zhou, Jianxing Gong, Jian Huang, Yang Mei  
**Category**: cs.AI  
**Published**: 2026-09-21  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.21940v1  

#### Abstract
Long-term memory is essential for large language model (LLM) agents to maintain consistency and personalization over extended interactions. Existing memory systems typically rely on fixed granularities or static schemas, but these designs struggle when heterogeneous information, such as preferences,...

---

### 21. [Multi-Subject Pretraining Enables Short-Calibration Personalization for Closed-Corpus Surface EMG Speech Decoding](https://arxiv.org/abs/2609.21288)

**Authors**: Chenqian Le, Beatrice Fumagalli, Yasamin Esmaeili, Xupeng Chen, Tianyu He, Nikasadat Emami, Adeen Flinker, Yao Wang  
**Category**: cs.LG  
**Published**: 2026-09-21  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.21288v1  

#### Abstract
Surface electromyography (sEMG)-based silent speech interfaces are limited by cross-user variability and calibration burden. We study a limited-data setting in which each of 27 speech-typical participants contributed less than 0.5 h of data (21.3 min on average) across Aloud and Mimed speech. Within...

---

### 22. [TinyCeNN-LM: Quality-Gated Conversion of Pretrained Attention with CeNN-Inspired Cellular-Recurrent Layers](https://arxiv.org/abs/2609.21139)

**Authors**: Kabeh Mohsenzadegan, Vahid Tavakkoli, Kyandoghere Kyamakya  
**Category**: cs.AI  
**Published**: 2026-09-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.21139v1  

#### Abstract
Replacing attention in a pretrained language model is a compatibility problem: a plausible substitute may alter representations expected by later layers. TinyCeNN-LM introduces a \emph{quality-gated post-training conversion} framework using CeNN-inspired cellular-recurrent layers with bounded local ...

---

### 23. [GVPO++: Group Variance Policy Optimization for LLM Post-Training and On-Policy Distillation](https://arxiv.org/abs/2609.21432)

**Authors**: Kaichen Zhang, Yuzhong Hong, Junwei Bao, Hongfei Jiang, Yang Song, Dingqian Hong, Hui Xiong  
**Category**: cs.AI  
**Published**: 2026-09-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.21432v1  

#### Abstract
Post-training plays a pivotal role in enhancing the reasoning capabilities and task-specific expertise of large language models (LLMs). Despite recent advances in post-training methods, such as Group Relative Policy Optimization (GRPO), their practical deployment remains impeded by training instabil...

---

### 24. [Learning-to-Optimize as the Missing Architectural Layer of AI-Native Networks](https://arxiv.org/abs/2609.21519)

**Authors**: Giambattista Amati, Federica Mangiatordi, Pierpaolo Salvo, Emiliano Pallotti, Simone Angelini  
**Category**: cs.AI  
**Published**: 2026-09-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.21519v1  

#### Abstract
Artificial Intelligence (AI) is becoming a fundamental design principle of future AI-native communication networks, enabling autonomous resource management, adaptive control, and zero-touch network operation. While current AI-native architectures increasingly embed intelligence across network functi...

---

### 25. [CodeMidas: Scaling Agentic Coding RL Environments from Code Itself](https://arxiv.org/abs/2609.22068)

**Authors**: Bowen Ye, Lei Li, Shicheng Li, Zihao Yue, Linghao Zhang, Hanglong Lv, Yuanxin Liu, Wenhan Ma, Hao Tian, Rang Li, Jinhao Dong, Yikai Zhao, Xiangwei Deng, Hailin Zhang, Liang Zhao, Qi Liu, Lingpeng Kong, Tong Yang, Fuli Luo  
**Category**: cs.AI  
**Published**: 2026-09-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.22068v1  

#### Abstract
Training capable coding agents via reinforcement learning (RL) requires diverse tasks with reliable verifiers. Open-source codebases offer a rich source of such tasks, while existing methods typically rely on development artifacts such as issues and commits, limiting the range of tasks that can be e...

---

### 26. [Beyond Atomic Tokens: Factorizing Syllables for Language Model Pretraining](https://arxiv.org/abs/2609.21362)

**Authors**: Nghia Hieu Nguyen, Thai Bao Huynh, Binh-An Dinh-Le, Phu Gia Hoang, Dat Tien Nguyen, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen  
**Category**: cs.CL  
**Published**: 2026-09-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.21362v1  

#### Abstract
Conventional tokenizers represent text as characters or statistically derived subwords, overlooking the internal phonological structure of syllables and often requiring large vocabularies. We introduce \textbf{Phonemic Tokenizer}, a linguistically motivated tokenizer for Vietnamese and Chinese that ...

---

### 27. [ArenaFlow: From Trajectory Ranking to Hierarchical Credit Propagation for Open-Ended Agent RL](https://arxiv.org/abs/2609.21378)

**Authors**: Qiang Zhang, Ruixue Ding, Fanrui Zhang, Xi Chen, Boli Chen, Shihang Wang, Yinfeng Huang, Yi Zheng, Pengjun Xie, Kaipeng Zhang, Jiawei Liu, Zheng-Jun Zha  
**Category**: cs.CL  
**Published**: 2026-09-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.21378v1  

#### Abstract
Reinforcement learning has substantially improved large language model (LLM) agents in verifiable domains, but remains difficult to apply to open-ended agent tasks, where solutions are diverse and reliable scalar rewards are hard to obtain. Recent pairwise evaluation methods alleviate reward discrim...

---

### 28. [PRISM-BN: A Controlled Corpus and Benchmark for Text-to-Parameterized Bayesian Network Extraction](https://arxiv.org/abs/2609.21673)

**Authors**: Amartya Bhattacharya, Nikhil Singh, Neeti Pokhriyal, Soroush Vosoughi  
**Category**: cs.CL  
**Published**: 2026-09-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.21673v1  

#### Abstract
Probabilistic Graphical Models (PGMs), especially Bayesian Networks (BNs), expose directed structure and probabilistic parameters, making them natural symbolic targets for neurosymbolic AI. Yet training text-to-parameterized-BN systems requires paired text-to-BN resources unavailable at scale. We in...

---

### 29. [An Interpretable Memory Decision Controller for LLM Agents Based on Three-Signal Complementarity: Decoupling Confidence and Consistency](https://arxiv.org/abs/2609.22043)

**Authors**: Yiming Zhang, Jinghong Zhang, Haoran Zhao, Yiren Ma, Chunlei Zhao  
**Category**: cs.CL  
**Published**: 2026-09-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.22043v1  

#### Abstract
Memory systems for large language models have focused predominantly on efficient retrieval, whereas the decision of whether retrieved memories should be trusted has received comparatively little attention. When the memory store contains conflicting positions, standard retrieval-augmented generation ...

---

### 30. [Cloud-Side Transactional Orchestration Framework for Resource-Constrained Embedded Systems](https://arxiv.org/abs/2609.21143)

**Authors**: Pravin Nagare, Aditya Sabbineni, Preetam Dedu, Willison Lopes  
**Category**: cs.DC  
**Published**: 2026-09-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.21143v1  

#### Abstract
As digital commerce ecosystems expand into low-end consumer electronics (CE), hardware constraints-specifically limited CPU duty cycles and volatile heap fragmentation-become significant bottlenecks for complex transactional flows. Traditional on-device middleware requires high "network chattiness" ...

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

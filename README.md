# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-15 10:20:17 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Echo: Error-Compensated 4-Bit Tensor-Parallel Communication for LLM Training and Inference](https://arxiv.org/abs/2610.14277)

**Authors**: Xingyu Xie, Xiaoyong Xie, Zhouchen Lin, Kim-Chuan Toh  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2610.14277v1  

#### Abstract
Tensor parallelism ends every layer with an all-reduce that sits between two dependent matrix multiplications, so its cost cannot be hidden and grows as compute outpaces interconnect bandwidth. Sending 4-bit instead of 16-bit values would cut this cost by nearly 4x, but the all-reduce must add the v...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《ECHO: Error-Compensated 4-Bit Tensor-Parallel Communication for LLM Training and Inference》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模语言模型（LLM）训练和推理中，**tensor parallelism** 是一种常用的并行策略，其核心是每层末尾的 **all-reduce** 操作。该操作位于两个矩阵乘法之间，无法被计算完全掩盖，且通信成本随模型规模增长而加剧。传统上，all-reduce 传输的是 16-bit 数据，通信带宽成为瓶颈。

虽然低比特量化（如 4-bit）可减少 75% 的通信量，但直接对 all-reduce 结果进行低比特量化会引入**累积舍入误差**，导致模型精度显著下降甚至训练不稳定。现有方法仅优化单次调用的量化误差，**缺乏对跨层、跨步长误差累积的控制机制**。

### 提出的新方法：EcHO
本文提出 **EcHO**（Error-Compensated 4-bit all-reduce），一种用于 LLM 训练和推理的 4-bit tensor-parallel 通信框架，核心创新如下：

- **高精度求和 + 低比特传输架构**：
  - 所有数据以 4-bit 整数格式在 GPU 间传输。
  - 求和操作在拥有者 GPU 上以高精度（如 FP32）完成，避免低比特累加溢出。
  - 每个元素在整个 all-reduce 调用中**最多被量化两次**（发送一次，返回一次），独立于 GPU 数量 $P$。

- **双时间尺度误差反馈补偿器（Dual Error Feedback Compensator）**：
  - 利用 all-reduce 操作后产生的残差（residual）精确恢复本次注入的舍入误差 $\delta_k$。
  - 设计一个双移动平均（two moving averages）的补偿器，将历史误差反馈到后续 all-reduce 的输入中。
  - 补偿项 $C_k$ 使得净误差 $p_k = \delta_k + C_k$ 在残差流（residual stream）中相互抵消，从而**保证累积误差有界**，不随网络深度增长。

- **理论保障**：
  - 证明了在任意误差序列下，**累积误差的偏移和保持有界**，且该界限独立于网络深度 $N$。
  - 输出偏差的一阶项被控制在单次舍入误差的水平。

### 相比现有方法的优势
| 方面 | 现有方法（如 FlashComm, TACO） | EcHO |
| :--- | :--- | :--- |
| **误差处理** | 单次调用内优化量化（如分组、保留离群值） | **跨调用补偿累积误差**，主动抵消 |
| **理论保证** | 无深度无关的误差累积控制 | **严格证明累积误差有界**，独立于深度 |
| **适用范围** | 多为推理，训练应用少且不稳定 | **同时支持训练和推理**，训练稳定 |
| **硬件依赖** | 部分依赖 FP8/FP4 硬件 | **无需 FP8/FP4 硬件支持**，纯软件实现 |

---

## 2. 核心实验方法和设置

### 数据集
- **预训练数据**：`Dolma C4`（10B tokens），用于 Qwen3-0.6B 的从头预训练。
- **评估数据集**：
  - **语言建模**：`WikiText`, `C4`
  - **下游任务**：`PIQA`, `ARC-easy`, `ARC-challenge`, `HellaSwag`, `WinoGrande`, `MMLU`, `MMLU-Pro`
  - **数学推理**：`GPQA`, `GSM8K`

### 实验设置和评估指标
- **硬件平台**：单台服务器，8 块 A100 GPU，通过 NVLink 连接。
- **模型**：
  - 推理：`Llama-3-8B`, `Qwen3-8B`
  - 预训练：`Qwen3-0.6B`
- **并行度**：$P=4$ 或 $P=8$
- **量化配置**：4-bit 或 8-bit，group size $g=128$，采用 zero-aware asymmetric quantization。
- **评估指标**：
  - **速度**：all-reduce 延迟、端到端 `time to first token`（TTFT）、训练步长时间。
  - **精度**：语言建模 `perplexity`、下游任务 `accuracy`、数学推理 `accuracy`。
  - **通信效率**：与 16-bit NCCL all-reduce 的加速比。

### 基线方法对比
- **通信基线**：
  - **NCCL 16-bit**：标准的 16-bit all-reduce，作为主要性能基线。
  - **Two-collective 4/8-bit**：基于 DeepSpeed 量化原语构建的两阶段量化 all-reduce。
- **精度基线**：
  - **16-bit inference/training**：全精度基准。
  - **4-bit no compensation**：相同量化方案但无误差补偿。
  - **FlashComm**：近期提出的低比特通信方法，用于对比精度。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **通信延迟**：
  - 对于 1GB 消息，**EcHO 4-bit all-reduce 比 16-bit NCCL 快 2.65×**。
  - 经过融合优化后，**补偿版 4-bit all-reduce 比 NCCL 快 2.09×**。

- **端到端加速**（P=8, 无背景流量）：
  - **推理**（TTFT）：最高提升 **1.16×**（Llama-3-8B）。
  - **训练**（step time）：最高提升 **1.09×**（Llama-3-8B）。
  - **背景流量增加时加速更明显**：在 100% 背景流量下，推理和训练分别提速 **1.44×** 和 **1.38×**。

- **模型精度**：
  - **推理**：在 7 个基准任务上，4-bit EcHO 的平均准确率与 16-bit 相比，**差距小于 0.4 点（Qwen3-8B）和 1.7 点（Llama-3-8B）**。
  - **训练**：在 10B tokens 预训练后，4-bit EcHO 的最终损失比 16-bit 基线**低不到 0.9%**。

### 与基线方法的对比结果
| 指标 | EcHO vs. 基线 |
| :--- | :--- |
| **All-reduce 延迟** | 比 two-collective 4-bit 快 1.69× |
| **TTFT 加速** | 比 16-bit NCCL 最高快 1.44×（高负载下） |
| **推理精度** | 比无补偿 4-bit 高 3.46 点（Llama-3-8B 平均） |
| **训练稳定性** | 损失曲线平滑，无发散，接近 16-bit 性能 |

### 消融实验结果
- **零点感知量化（Zero-aware quantization）**：
  - 相比 naive asymmetric 量化，perplexity 显著降低（如 WikiText 上从 7.296 降至 6.362）。
- **误差补偿器（Compensator）**：
  - 移除补偿器后，perplexity 从 6.362 升至 6.730，验证了其对抑制误差累积的关键作用。
  - 使用单个移动平均（single average）替代双时间尺度设计，性能略有下降，表明双尺度能更好适应误差变化。
- **超参数敏感性**：
  - 方法对补偿增益 $n$、混合权重 $a$ 和衰减率 $(\beta_1, \beta_2)$ 不敏感，在合理范围内性能稳定。

---

## 4. 关键结论和发现

### 主要发现
1. **误差累积是低比特通信的核心障碍**：单纯优化单次量化质量不足以维持模型精度，必须控制跨层的误差传播。
2. **EcHO 实现了误差的主动抵消**：通过利用 all-reduce 的内在结构（残差可得）和残差连接（补偿可生效），实现了类似“回声”（echo）的误差反馈机制。
3. **4-bit tensor-parallel 通信是可行且高效的**：EcHO 首次证明了 4-bit 整数通信可以同时用于 LLM 的**训练和推理**，且精度损失极小。
4. **性能收益随网络拥塞而增大**：在高背景流量下，EcHO 的优势更加明显，凸显了其在生产环境中的实用价值。

### 局限性
- **额外状态开销**：补偿器需要存储两个与通信张量同形的状态变量（moving averages），增加了显存占用。
- **主要针对 tensor parallelism**：当前设计聚焦于 tensor-parallel 的 all-reduce，未直接解决 pipeline 或 data parallelism 中的其他通信模式。
- **假设条件**：理论分析依赖于对网络 Jacobian 的平滑性假设（Assumption 1），尽管实验证明其在实际模型中成立。

### 未来工作方向
- 将补偿机制扩展到 **sequence parallelism** 的 reduce-scatter 和 **expert parallelism** 的 combine 操作。
- 探索更轻量化的补偿器设计，减少状态存储开销。
- 研究在动态稀疏激活或 MoE 架构下的自适应补偿策略。
- 将 EcHO 集成到主流训练框架（如 Megatron-LM, DeepSpeed）中，推动其工业级应用。

</details>

---

### 2. [Scaling LLM Inference via Parallelism: Throughput-Latency Trade-off](https://arxiv.org/abs/2610.19405)

**Authors**: Javad Mirzaei, Jeebak Mitra  
**Category**: cs.DC  
**Published**: 2026-10-15  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2610.19405v1  

#### Abstract
Large Language Model (LLM) serving systems must balance latency and throughput while efficiently utilizing distributed GPU resources. This challenge arises because inference consists of two fundamentally different phases: prefill, which determines time-to-first-token (TTFT), and decode, which determ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scaling LLM Inference via Parallelism: Throughput-Latency Trade-off

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 Large Language Model (LLM) 推理系统面临 **latency**（延迟）与 **throughput**（吞吐量）之间的根本性权衡，尤其是在分布式 GPU 资源下进行高效服务时。传统评估方式仅依赖单一指标（如总延迟或吞吐量），无法准确反映真实场景下的性能表现。

该论文聚焦于以下挑战：
- LLM 推理分为两个阶段：**prefill**（决定 Time-to-First-Token, TTFT）和 **decode**（决定 Inter-Token Latency, ITL），二者具有不同的计算、通信和内存特性。
- 不同的并行策略（Tensor Parallelism, TP；Pipeline Parallelism, PP）在不同 workload 下表现差异显著。
- 当前缺乏一个系统性的框架来指导如何根据 **workload 特征、SLO（Service-Level Objectives）要求、context length** 动态选择最优的 parallelism 配置。

### 提出了什么新方法或新思路
论文提出了一套 **系统性框架** 来理解与优化 LLM 推理中的并行配置，从三个互补视角切入：

1. **Parallelism Operating Region（并行操作区域）**
   - 构建 **TTFT-ITL 可行集（feasible set）** 和 **Pareto 前沿（Pareto frontier）**，揭示不同 workload 组成下各并行配置的性能边界。
   - 定义“操作区域”概念：随着 workload 从 decode-heavy 向 prefill-heavy 转变，最优配置从 TP-dominant 过渡到 PP-dominant。

2. **SLO-aware Parallelism Selection（SLO 意识的并行选择）**
   - 引入 **goodput** 作为核心评估指标：即满足指定延迟要求的有效请求完成率（$ \text{Goodput} = \text{Throughput} \times \text{SLO Compliance Rate} $）。
   - 将并行配置选择形式化为一个优化问题，在给定 TTFT SLO 下最大化 goodput。

3. **Long-context Scalability Analysis（长上下文可扩展性分析）**
   - 分析 context length 增加对不同并行策略的影响，发现 PP-dominant 配置在长上下文中展现出更优的 SLO 保持能力和可扩展性。

### 相比现有方法的优势
| 方面 | 本文优势 |
|------|--------|
| **评估维度** | 超越单一指标，联合考虑 TTFT 与 ITL，支持多维权衡分析 |
| **决策依据** | 明确将 workload composition、SLO 要求、context length 纳入配置选择逻辑 |
| **实用性** | 提供可视化工具（如 Pareto frontier、CDF 曲线）辅助部署决策 |
| **动态适应性** | 支持未来构建自适应系统，实现运行时 parallelism switching |

---

## 2. 核心实验方法和设置

### 使用的模型与硬件平台
- **模型**：Qwen2.5-14B-Instruct
- **精度格式**：BF16
- **GPU**：8 × NVIDIA H200（每卡 141GB）
- **推理引擎**：vLLM v0.15.0

### 实验设置
#### Workload 定义
使用三元组表示 workload：
$$
w = (L_{in}, L_{out}, C)
$$
其中：
- $ L_{in} $：Input Sequence Length (ISL)
- $ L_{out} $：Output Sequence Length (OSL)
- $ C $：Request Concurrency（并发请求数）

固定总序列长度 $ L_{seq} = L_{in} + L_{out} $ 以隔离 workload composition 的影响。

#### 并行配置命名（基于 8 GPUs）
| 符号 | 配置说明 |
|------|---------|
| TP8 | (8,1): Tensor Parallelism Degree=8, Pipeline=1 |
| TP4PP2 | (4,2) |
| TP2PP4 | (2,4) |
| PP8 | (1,8): Pipeline Parallelism Dominant |

### 评估指标
| 指标 | 定义 |
|------|-----|
| **TTFTavg(p, w)** | 所有请求的平均 Time-to-First-Token |
| **ITLavg(p, w)** | 自回归生成过程中的平均 Inter-Token Latency |
| **Throughput ($\lambda_p(w)$)** | 单位时间内处理的请求数 |
| **SLO Compliance ($F_p(T_s \| w)$)** | 满足 TTFT ≤ $T_s$ 的请求比例 |
| **Goodput ($G_p(w, T_s)$)** | $ \lambda_p(w) \cdot F_p(T_s \| w) $，为核心优化目标 |

### 基线方法对比
本文并非提出新的并行算法，而是对主流并行策略进行横向比较：
- **TP-dominant**: TP8
- **PP-dominant**: PP8
- **Hybrid**: TP4PP2, TP2PP4

通过构建 **Pareto frontier** 和 **goodput 曲线** 对比其在不同 workload 和 SLO 下的表现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）TTFT-ITL Pareto Frontier（图1）
- **TP8**：最低 ITL（约 2–4 ms），但最高 TTFT（>50 ms）→ 适合 decode-heavy 场景
- **PP8**：最低 TTFT（<10 ms），但最高 ITL（~70 ms）→ 适合 prefill-heavy 场景
- **Hybrid 配置**：位于中间地带，提供平滑过渡
- **趋势**：随着 workload 从 decode-heavy → balanced → prefill-heavy，Pareto 最优点从 TP8 → Hybrid → PP8 系统性迁移

#### （2）OSL 对 ITL 的影响（图2）
- 增大 OSL 可更好摊销 decode 开销，降低实测 ITL
- **TP-dominant 配置受益最大**：TP8 在 OSL=800 时 ITL 显著下降
- **PP8 改善有限**：受限于 pipeline bubble 和 stage propagation 开销

#### （3）SLO-aware Goodput 分析（图3–5）
- **严格 SLO（如 Ts=5s）**：
  - PP8 虽然 throughput 较低，但 SLO compliance 更高（CDF 上升更快）
  - **PP8 实现最高 goodput**
- **宽松 SLO（如 Ts=25s）**：
  - 所有配置 compliance 接近 1
  - TP8 凭借更高 throughput 成为最优
- **结论**：随着 SLO 放松，最优配置从 **PP8 → Hybrid → TP8**

#### （4）Long-context Scaling（图6–7）
- 固定 $ L_{out}=1000 $，逐步增加 $ L_{in} $（即 $ L_{seq} $ 从 ~1K 到 30K）
- **结果**：
  - 在短 context 下，TP8 因高 throughput 占优
  - 随着 $ L_{seq} $ 增加，PP8 的 goodput 下降更缓慢
  - 在 $ L_{seq} > 15K $ 且 SLO 严格（如 Ts=2s）时，**PP8 显著超越 TP8**
- **原因**（图7）：
  - TP8 的 SLO compliance 随 context 增长急剧下降（fine-grained communication 成为瓶颈）
  - PP8 更能维持较高的 SLO attainment，表现出更强的 **scalability**

### 消融实验结果
虽然未明确标注“ablation”，但实验设计本质上是消融式的：
- 控制变量法分析 **$ L_{in} $、$ L_{out} $、$ T_s $、$ L_{seq} $** 对性能的影响
- 分离 prefill 与 decode 阶段的行为差异
- 验证了 **parallelism operating region** 的存在性和迁移规律

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **不存在全局最优的并行配置**  
   TP、PP 及其混合配置在 TTFT-ITL 设计空间中占据不同位置，最优选择高度依赖 workload composition。

2. ✅ **工作负载组成驱动 Pareto 前沿迁移**  
   - Decode-heavy → TP-dominant 更优  
   - Prefill-heavy → PP-dominant 更优  
   - 存在一个系统性的“操作区域”转移路径。

3. ✅ **SLO 要求决定 goodput 最优配置**  
   - 严格 TTFT SLO → PP-dominant 更好（牺牲 throughput 换取低 TTFT）
   - 宽松 SLO → TP-dominant 更好（利用高 throughput 提升 goodput）

4. ✅ **长上下文强化 PP 的优势**  
   随着 context length 增加，PP-dominant 配置在 **SLO attainment 的衰减速率** 上明显优于 TP，展现出更好的 long-context scalability。

5. ✅ **goodput 是比 raw throughput 更合理的评估指标**  
   特别是在生产环境中，只有满足 SLO 的请求才构成有效服务。

### 方法的局限性
- ❗ 实验集中在 **8 GPU 规模**，更大规模集群下的通信开销可能改变结论。
- ❗ 未考虑 **dynamic batch size** 或 **request arrival pattern** 的随机性。
- ❗ 分析主要针对 dense attention 模型，sparse attention 或 MoE 架构需进一步研究。
- ❗ 当前框架为静态分析，尚未集成到实时调度器中实现动态切换。

### 未来工作方向
- 🔮 构建 **adaptive parallelism switching system**：根据实时 workload 和 SLO 动态调整 TP/PP 配置。
- 🔮 结合 **context parallelism (CP)** 进一步提升长上下文性能。
- 🔮 将本框架扩展至 **multi-SLO、multi-tenant、异构资源环境**。
- 🔮 探索 **hardware-aware 编译优化** 与并行策略协同设计。

---

> **总结一句话**：  
> 本文建立了首个系统性框架，揭示了 LLM 推理中 **parallelism 配置的选择本质是一个由 workload、SLO 和 context length 共同决定的多维权衡问题**，并倡导以 **goodput** 和 **Pareto frontier** 为核心工具指导实际部署决策。

</details>

---

### 3. [Joint Subspace Sparse Attention for Efficient Long-Context Decoding](https://arxiv.org/abs/2610.19026)

**Authors**: Jiwon Song, Yoongon Kim, Jae-Joon Kim  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2610.19026v1  

#### Abstract
Sparse attention reduces the KV-cache traffic that dominates long-context LLM decoding, but practical methods largely operate at coarse granularity. Token-level selection offers finer control, yet presents two challenges: low-dimensional scoring may lose query--key structure needed for accurate rank...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Joint Subspace Sparse Attention for Efficient Long-Context Decoding》论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在长上下文自回归解码中，**KV-cache 的访问是主要瓶颈**。随着序列长度增长，全注意力机制需要对整个上下文进行查询-键评分并读取所有值，导致内存流量呈线性增长。虽然稀疏注意力（sparse attention）可以减少 KV 访问量，但现有方法存在以下两个核心挑战：

1. **低维打分质量差**：token-level 的选择通常依赖于低维投影来降低计算成本，但如果投影目标不当（如仅重建 key），会丢失 query-key 的双线性交互结构，影响 token 排序准确性。
2. **执行效率低下**：token-level 选择会产生不规则索引集合，传统实现需先“gather”选中的 KV 条目到连续缓冲区再进行注意力计算，这一过程带来显著延迟开销。

### 提出的新方法：JSSA（Joint Subspace Sparse Attention）
JSSA 是一种无需训练的 token-level 稀疏注意力方法，从**选择质量和执行效率**两方面系统性地解决了上述问题。

#### 创新点一：联合查询-键子空间（Joint Query-Key Subspace）
- **目标函数设计**：提出一个以最小化**双线性得分误差**为目标的选择准则，直接保留决定 token 排序的关键 query-key 交互。
- **可解闭式解法**：将非凸优化问题转化为一个可处理的边际上界目标（marginal upper bound），其具有**闭式谱解**（closed-form spectral solution），即通过 `(GQ + GK)` 的前 `r` 个特征向量得到最优投影矩阵。
- **有效性验证**：实验表明，直接优化原始目标无法带来可测量的性能提升，因此使用该闭式解即可达到最优效果。

#### 创新点二：无 gather 的 indexed attention 内核（Gather-Free Indexed Attention Kernel）
- 针对 GQA 架构，设计了一个新的 Triton 内核，支持每个 KV head 拥有独立的 token 选择集合，并能**原位访问原始 KV 缓存中的分散条目**，无需物理聚集（no physical gathering）。
- 这是首个在标准 GQA KV-cache 布局下支持 per-KV-head 精细选择且无需 gather 的内核。

### 相比现有方法的优势
| 维度 | JSSA | 其他方法（如 Loki, FASA, QUEST） |
|------|------|-------------------------------|
| **选择精度** | 更好保留双线性得分结构 | 仅重建 key 或固定通道选择，忽略 query-key 交互 |
| **执行效率** | 无 gather 开销，直接 indexed 访问 | gather 步骤耗时高，尤其在大 batch 下 |
| **灵活性** | 支持 per-KV-head 不同选择集 | 多数要求共享选择集或块级选择 |
| **适用性** | 跨 GQA、混合滑窗/全局、MLA 架构通用 | 往往针对特定架构设计 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LongBench**：多任务长文本理解基准，用于评估模型在多种任务下的长上下文理解能力。
- **RULER**：专门测试真实上下文长度能力的基准，包含“needle-in-a-haystack”等检索与聚合任务。
- **PG-19**：长文本语言建模数据集，用于评估困惑度（perplexity）。
- **推理任务**：
  - **AIME 2025**：数学竞赛题，测试复杂推理能力。
  - **MATH-500**：数学问题集。
  - **GPQA-Diamond**：研究生级别问答挑战。

### 实验设置和评估指标
- **模型覆盖广泛**：
  - GQA：LLaMA-3.1-8B-Instruct, Qwen3-8B, Qwen3-14B, Ministral-3-8B
  - Hybrid Local/Global GQA：Gemma4-26B-A4B-It
  - MLA：GLM-4.7-Flash, GLM-5.2（~700B）
- **上下文长度**：最高达 **128K tokens**
- **硬件平台**：NVIDIA B200 GPU
- **评估指标**：
  - 准确率（Accuracy）：LongBench, RULER, AIME, MATH, GPQA
  - 困惑度（Perplexity）：PG-19
  - 吞吐量（Throughput）：tokens/sec
  - 生成长度相对比（Generation Length Ratio）：衡量推理稳定性
  - Oracle Recall@B：衡量低维打分是否准确预测全维 top-B token

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Full-KV (FKV)** | 全注意力 | 性能上限 |
| **Loki** | token-level | 在 key 重建子空间中打分 |
| **FASA** | token-level | 选择原始 RoPE 通道进行打分 |
| **QUEST** | page-level | 查询感知的页级稀疏注意力 |
| **SnapKV**, **SqueezedAttn**, **TriAttention**, **SeerAttention-R** | 其他稀疏方法 | 包括压缩、滑窗、粗粒度选择等 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 长上下文理解（LongBench）
| Model | FKV | JSSA | 最佳基线 |
|-------|-----|------|---------|
| LLaMA-3.1-8B | 48.35 | **49.16** | 47.95 (SqueezedAttn) |
| Qwen3-14B | 50.49 | **49.82** | 49.54 (Loki) |
| Ministral-3-8B | 47.90 | 47.79 | **48.07** (Loki) |

> JSSA 在多数模型上接近甚至超过 full-KV，且表现最稳定。

#### ✅ 长上下文推理（RULER @ 128K）
- JSSA 在所有模型上均保持最接近 full-KV 表现：
  - LLaMA-3.1-8B：75.38 vs FKV 86.35
  - Qwen3-14B：82.21 vs FKV 88.86
  - Ministral-3-8B：85.97 vs FKV 90.72
- 显著优于 Loki 和 FASA，后者在长上下文中严重退化。

#### ✅ 语言建模（PG-19 @ 128K）
| Model | FKV | JSSA | Loki | FASA |
|-------|-----|------|------|------|
| LLaMA-3.1-8B | 10.38 | **11.49** | 56.29 | 56.59 |
| Qwen3-14B | 10.41 | **10.53** | 14.61 | 15.05 |
| Ministral-3-8B | 8.37 | **9.05** | 9.46 | 10.88 |

> JSSA 困惑度几乎与 full-KV 相当，而其他方法在某些模型上出现灾难性上升。

#### ✅ 推理性能（Pass@1 Accuracy @ 1K tokens）
| Model | Method | AIME 2025 | MATH-500 | GPQA-Diamond | Gen Length |
|-------|--------|-----------|----------|---------------|------------|
| Qwen3-8B | FKV | 69.58 | 94.53 | 57.95 | 1.00× |
| Qwen3-8B | JSSA | **68.33** | **94.23** | **57.39** | **1.07×** |
| Qwen3-8B | FASA | 60.42 | 93.58 | 57.45 | 1.30× |
| Gemma4 | JSSA | **86.67** | **97.08** | **81.94** | **1.17×** |
| GLM-4.7-Flash | JSSA | **91.67** | **96.88** | **77.53** | **1.17×** |

> JSSA 在保持高准确率的同时，生成长度膨胀极小（仅 7–17%），远优于 TriAttention（可达 4×）。

#### ✅ 解码吞吐量（Decoding Throughput @ 128K）
在单张 NVIDIA B200 GPU 上：
- **Qwen3-8B**：**1.51×** FKV
- **Gemma4-26B**：**1.88×** FKV
- **GLM-4.7-Flash**：高达 **6.43×** FKV

> 在 MLA 架构上优势尤为明显，得益于 FlashMLA 的高效稀疏 kernel 支持。

#### ✅ 大规模模型验证（~700B 参数）
在 DeepSeek-V3 和 GLM-5.2 上对比 **post-hoc JSSA** 与 **model-native DSA**：
- **推理准确率**：JSSA 与 DSA **基本持平**（差异 <1%）
- **解码吞吐量**：JSSA 达到 DSA 的 **96.1–99.2%**
- **无需重新训练**：JSSA 可直接应用于 full-attention 模型，无需任何 fine-tuning

> 结论：JSSA 可作为 model-native sparse attention 的有效替代方案。

---

### 消融实验结果

#### 投影维度（Projection Rank r）
- 在低秩（如 r=4）下，JSSA 明显优于 FASA 和 Loki（+5.4 ~ +24.48 LongBench 分）
- 表明其联合子空间更有效地利用有限维度。

#### Calibration 鲁棒性
- 对校准语料敏感性低（LongAlign vs Qasper 差异 <0.1 分）
- 但对校准长度敏感：**必须用接近部署长度（如 128K）的数据进行 calibration**

#### 无 gather 执行收益
- 在 LLaMA-3.1-8B 上，**gather-free indexed attention 将 attention 阶段加速从 25.35× 提升至 65.26×**（batch=16）
- 在 Gemma4 上，端到端吞吐量提升 **1.07–2.22×**

> 证明 gather 开销是 token-level 稀疏化的关键瓶颈。

#### per-KV-head 选择 vs 层级共享选择
- 在 128K 上，per-KV-head 选择比 head-shared soft voting 平均高出 **2–5 分**
- 说明保留每个 KV head 的独立偏好至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **联合 query-key 子空间显著优于单边重建**：保留双线性得分结构是提高 token 排序准确性的关键。
2. **闭式谱解足够有效**：无需迭代优化即可获得高质量投影空间。
3. **gather 是 token-level 稀疏化的性能杀手**：即使选择正确，gather 开销也会抵消大部分收益。
4. **JSSA 实现了精度与效率的双重突破**：
   - 几乎无损于 full-KV accuracy
   - 吞吐量提升高达 **6.43×**
   - 可扩展至 **700B 级别模型**
5. **无需模型训练即可媲美 model-native DSA**：为已有大模型提供高效的后装稀疏化方案。

### 方法的局限性
- **依赖 calibration 数据**：需要代表性长序列数据来构建投影基，若部署场景与 calibrate 数据差异过大可能影响效果。
- **仅适用于 inference**：不改变训练过程，无法像 DSA 那样在训练中联合优化稀疏模式。
- **对 MLA 架构依赖 FlashMLA**：在 MLA 上依赖外部 kernel 支持。

### 未来工作方向
- 探索更轻量化的 calibration 方案（如合成数据、迁移 calibration）
- 将 JSSA 思路扩展至 prefill 阶段
- 设计可微分版本以支持训练时联合优化
- 探索动态调整 `r` 或 `B` 的策略以适应不同输入

---

> 🔗 **代码开源地址**：[https://github.com/jiwonsong-dev/JSSA](https://github.com/jiwonsong-dev/JSSA)

</details>

---

### 4. [Stable Token-level Scheduling for Serverless LLM Inference](https://arxiv.org/abs/2610.17689)

**Authors**: Hardi Ashishbhai Desai, Balaji N K, Sourav Kanti Addya  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2610.17689v1  

#### Abstract
Serverless inference is emerging as an efficient deployment model for Large Language Models (LLMs) due to its ability to elastically scale resources, improve GPU utilization through resource sharing, and reduce infrastructure costs. However, existing approaches primarily optimize time to first token...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Stable Token-level Scheduling for Serverless LLM Inference 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Serverless LLM Inference** 调度系统主要优化 **Time-to-First-Token (TTFT)**，而忽略了 **Inter-Token Latency (ITL)** ——即用户在流式生成过程中感知到的逐个 token 之间的延迟。这种忽略导致在请求突发（bursty workloads）时出现严重的 **latency spikes** 和 **ITL 不稳定**。

根本原因在于：
- LLM 推理分为 **prefill 阶段**（计算密集型）和 **decode 阶段**（带宽受限、同步执行）。
- 新请求的 admission 会引入 prefill 操作，干扰正在进行的 decode 步骤，造成 **noisy-neighbour penalty**，提升所有活跃用户的 ITL。
- 现有调度器多为“开环”设计，在请求到达时立即决定是否接纳，缺乏对 decode 引擎实时负载状态的反馈。

### 提出了什么新方法或新思路
提出 **Token Window Pacing (TWP)** ——一种基于运行时反馈的 **closed-loop, token-level admission control** 框架。

#### 核心思想：
- 将调度决策从 **request-level / batch-level** 下沉到 **decode-step level**，实现细粒度控制。
- 利用每个 decode step 结束后的 **ITL、queue depth、KV-cache occupancy** 等信号动态调节一个自适应的 **token window**（软上限），用于限制可接纳的总剩余输出 token 数量（token load）。
- 引入 **chunked prefill**，将长 prompt 分块处理，分散 prefill 开销，降低单步内存压力。

#### 关键组件：
1. **双门控准入机制（Two-guard Admission Gate）**
   - Token Budget Check：确保新增请求后总 token load ≤ 当前 token window $ H_t $
   - KV-cache Headroom Check：确保有足够的显存页可用
2. **复合排序策略（Composite Ranking Policy）**
   - 综合考虑：等待时间（防饥饿）、任务长度匹配度（提高批效率）、公平性（防短任务饿死长任务）
3. **反馈控制器**
   - 动态调整 $ H_t $：当 ITL 上升或队列积压时收缩窗口；反之扩张以吸收负载

### 相比现有方法的优势
| 方面 | TWP 优势 |
|------|--------|
| **稳定性** | 显著减少 ITL 波动和 latency spikes，提供更平滑的流式体验 |
| **吞吐量** | 最高提升 **58%** 吞吐（vs 传统调度器），**45%** vs ServerlessLLM |
| **尾部延迟** | P99 TTFT 降低 **15–20%**，优于所有基线 |
| **部署兼容性** | 无需修改模型架构、kernel 或 GPU 执行逻辑，可集成至任意支持 step-level 接口的推理框架 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GSM8K benchmark**：数学推理任务，用于生成真实感强的请求输入。
- Prompt 长度分布：
  - 60% 短（30–80 tokens）
  - 30% 中（120–250 tokens）
  - 10% 长（350–512 tokens）
- 输出长度异构，模拟实际交互场景。

### 实验设置
- **硬件环境**：NVIDIA Tesla V100 GPU (32GB)，CUDA 11.8
- **模型**：Meta-Llama-3.2-3B（FP16）
- **请求模式**：Poisson 过程，λ = 18 req/s，持续 60 秒（最多 ~250 请求）
- **饱和批大小（Bo）**：通过离线 profiling 测得为 8
- **TWP 参数配置**：
  - 初始 token window $ H_0 = B_o \times \bar{L} = 400 $ tokens
  - 控制增益 $ \gamma = 0.15 $
  - Prefill chunk size $ C = 64 $ tokens

### 评估指标
| 指标 | 描述 |
|------|------|
| **TTFT** | Time-to-First-Token，含冷启动 + prefill + 第一次 decode |
| **End-to-end Latency (P50/P95/P99)** | 完整响应时间 |
| **Throughput (tokens/s)** | 单位时间内生成的 token 总数 |
| **Inter-Token Latency (ITL)** | 相邻两个 decode step 的 wall-clock 时间差 |
| **ITL Coefficient of Variation (CV)** | 衡量 ITL 稳定性的统计量（越低越稳定） |

### 基线方法对比
| Baseline | 特点 |
|---------|------|
| **FCFS** | 先来先服务，无并发控制 |
| **Round-Robin** | 轮询分配容器 |
| **Priority (Shortest-Prompt-First)** | 优先处理短 prompt 请求 |
| **DynamicBatch** | 设定静态 ITL 阈值进行批大小控制 |
| **ServerlessLLM [2]** | 快速 checkpoint 加载 + 冷启动优化，但在高并发下易导致 batch saturation |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Fig. 6, 7, 8 及 Table III）

| 指标 | TWP 表现 | 对比优势 |
|------|--------|--------|
| **Throughput** | **0.197 req/s** | ↑ **58%** vs 传统调度器（~0.124 req/s）<br>↑ **45%** vs ServerlessLLM (0.136 req/s) |
| **P99 TTFT** | **66 ms** | ↓ 比 ServerlessLLM (78 ms) 低 **15%+**<br>↓ 比 DynamicBatch (68 ms) 更优 |
| **Median TTFT (P50)** | ~50 ms | 与其他方法相当，说明未牺牲平均性能 |
| **ITL 平均值** | ~43.5–43.8 ms | 显著低于基线峰值（可达 60+ ms） |
| **ITL CV** | **0.0144** | 低于 Round-Robin (0.0203)、FCFS (0.0141)，为最低波动 |

### 与基线方法的对比结果（综合分析）
- **TTFT 稳定性**（Fig. 4 & 5）：
  - TWP 在整个 workload 中保持 TTFT 接近 50ms，几乎无 spike；
  - ServerlessLLM 和 Round-Robin 出现高达 85ms 的延迟尖峰。
- **吞吐量优势**（Fig. 6）：
  - TWP 实现最高请求吞吐，因其有效防止了因过载导致的 decode pipeline 效率下降。
- **ITL 分布集中**（Fig. 7）：
  - 基线方法存在大量 high-latency outliers（>60ms），而 TWP 分布紧致集中在 43.5ms 左右。
- **不同负载下的鲁棒性**（Table III）：
  - 在 `low`, `medium`, `high`, `burst` 四种负载下，TWP 始终保持最优或接近最优的 **throughput、P99 latency、ITL CV**。
  - 尤其在 **burst load** 下，传统方法性能急剧恶化，而 TWP 仍能维持稳定表现。

### 消融实验（隐含于设计分析中）
虽然文中未明确列出消融表，但从机制设计可推断以下关键因素的作用：
- **ITL 反馈信号**：相比仅使用 queue depth 或 KV-cache occupancy，ITL 是最直接反映 decode 拥塞的指标。
- **chunked prefill**：显著缓解了 large prompt 导致的瞬时内存争用，避免单次 prefill 抬高全局 ITL。
- **token-level 控制 vs batch-level**：token load ($ K_t $) 比 batch size ($ B_t $) 更准确地刻画未来负载压力，尤其适用于输出长度差异大的场景。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **ITL 是理想的拥塞信号**  
   ITL 直接反映了 decode 阶段的实际资源竞争情况，且无需额外测量成本，是构建闭环调度系统的理想反馈变量。

2. **token-level admission control 显著优于 request/batch-level 控制**  
   通过在每个 decode step 动态调节 token window，TWP 能主动预防过载，而非被动应对，从而实现 **高吞吐 + 低延迟 + 高稳定性** 的统一。

3. **现有 Serverless LLM 系统存在“反馈缺失”问题**  
   即使如 ServerlessLLM 这样优化了冷启动的先进系统，也因缺乏对 decode 状态的感知而导致 batch saturation 和 ITL 恶化。

4. **TWP 可广泛部署**  
   不依赖特定硬件或模型结构，仅需接入 step-level 接口即可运行，具备良好的通用性和可集成性。

### 方法的局限性
- **依赖准确的 Bo 和 l_min 标定**：需要一次性的 offline profiling 来确定系统拐点参数。
- **chunked prefill 带来轻微 TTFT 增加**：虽然整体 TTFT 更稳定，但分块处理会使首个 token 稍微延后（trade-off 设计）。
- **未考虑跨节点调度**：当前实验基于单 GPU 设置，扩展到分布式或多租户集群需进一步研究。

### 未来工作方向
- 支持 **multi-GPU / distributed setting** 下的协同 token window 控制。
- 引入 **predictive control**（如结合 workload forecasting）提前调整窗口，增强抗突发能力。
- 探索 **fine-grained per-request SLA 保障**，例如为不同优先级用户提供差异化 token window 配置。
- 结合 **LoRA / adapter-based serving** 场景，实现轻量化模型切换下的动态调度。

--- 

> ✅ **总结一句话**：  
> **TWP 通过引入基于 ITL 反馈的 token-level 动态准入控制，在不修改底层模型的前提下，实现了 Serverless LLM Inference 的高吞吐、低延迟与高稳定性三重目标，填补了现有调度系统中“decode 状态感知”的空白。**

</details>

---

### 5. [FleetPilot: Fine-Grained Steering Empowers Real-World LLM Personalization](https://arxiv.org/abs/2610.14970)

**Authors**: Kounianhua Du, Jianxing Liu, Kangning Zhang, Wenxiang Jiao, Yuan Lu, Jiarui Jin, Jianghao Lin, Weiwen Liu, Yong Yu, Weinan Zhang  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.14970v1  

#### Abstract
With the rapid evolution of large language models (LLMs), there is an increasing demand for effective personalization techniques that can adapt model behavior to individual user preferences. Personalized LLMs must infer a user's query-dependent preferences from limited, heterogeneous, and evolving i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FleetPilot: Fine-Grained Steering Empowers Real-World LLM Personalization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前大语言模型（LLM）个性化面临三大现实挑战：
- **数据稀疏性**（Data Scarcity）：新用户或低频用户的交互历史极少（常少于10次），导致冷启动问题。
- **偏好漂移**（Temporal Adaptability）：用户兴趣随时间快速变化，传统依赖周期性训练的方法难以实时适应。
- **上下文异质性**（Instance Specificity）：同一用户在不同查询中表现出不同偏好，静态的用户级表示无法捕捉细粒度、查询相关的个性化需求。

现有方法如 **prompt-based personalization**、**parametric adaptation** 和 **personalized reward modeling** 在这些场景下存在明显瓶颈：
- **Prompt-based** 方法受限于上下文窗口，易引入噪声或丢失关键信息；
- **Parametric adaptation** 需要为每个用户微调参数，带来高昂的存储、训练和部署成本；
- **Reward modeling** 依赖多候选生成与打分，推理开销大，且受限于候选覆盖范围。

---

### **提出了什么新方法或新思路**
作者提出 **FleetPilot**，一种**细粒度、实例定制化（instance-tailored）的推理时激活引导框架**，其核心思想是：
> 不通过修改模型参数或扩展输入上下文来实现个性化，而是动态地从用户历史中提取**样本级干扰向量**（steering vectors），并在前向传播过程中注入到模型中间层，从而实现轻量、灵活、实时的个性化适配。

#### **三大创新设计**：
1. **Matched Contrast（匹配对比）**  
   固定历史查询 $X_i$ 和响应 $Y_i$，仅改变上下文（用户相关 vs. 用户无关），通过对比模型激活差异，提取与**个性化条件作用**相关的信号，避免响应内容差异带来的干扰。

2. **Fine-Grained Hooking（细粒度钩取）**  
   不再将对比信号压缩为单一的后块残差方向，而是分别从 **Attention 残差输出**（post-attention residual）和 **MLP 输出**（before residual addition）两个接口提取引导方向，保留更精细的行为调控路径。

3. **Query-Conditioned Aggregation（查询条件聚合）**  
   在推理时，基于当前查询语义，从用户的“引导向量字典”中检索最相关的若干历史条目，分别聚合 Attention 和 MLP 侧的方向，形成**查询特定的双路引导向量**，并通过 **Sequential Dual-Injection** 注入模型。

---

### **相比现有方法的优势**
| 维度 | FleetPilot | 传统方法 |
|------|----------|---------|
| **训练成本** | 无训练，无需梯度更新 | 需微调或优化 |
| **存储开销** | 每用户 <1MB（仅存向量） | 每用户 ~18MB（LoRA权重） |
| **推理延迟** | 轻量增加（主要来自检索） | 高（多候选生成/打分） |
| **更新速度** | 即时更新（新增交互即可构建新向量） | 滞后（需重新训练） |
| **数据效率** | 极高，在极低数据下仍有效 | 在稀疏数据下易过拟合 |
| **灵活性** | 支持查询级个性化 | 多为用户级或固定行为 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在三个多样化任务上进行评估，涵盖短文本、长文本和结构化函数调用：

| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **Headline Generation** | 短文本生成 | 给定新闻文章，生成符合用户风格的标题 |
| **Abstract Writing** | 长文本生成 | 给定论文标题与关键词，生成符合用户写作风格的摘要 |
| **PersonalWAB** | 个性化Web功能调用 | 推断用户意图并调用相应Web API（如搜索、预订等） |

所有数据集均按时间划分：早期交互用于构建引导向量，后期用于测试。

---

### **实验设置和评估指标**

#### **评估指标**
- **内容生成任务**（Headline & Abstract）：
  - `ROUGE-1`, `ROUGE-L`, `METEOR`：衡量生成文本与参考文本的重叠质量。
- **功能调用任务**（PersonalWAB）：
  - `Accuracy (ACC)`：判断是否正确调用函数及参数。

#### **基线方法对比**
| 类别 | 方法 |
|------|------|
| **Prompt-based** | In-Context Learning (ICL) with k=1,3,5,10 |
| **Parametric Adaptation** | ReLLa (shared adapter), OPPU, PER-PCS (user-specific PEFT) |
| **Reward Modeling** | SynthesizeMe (personalized reward model) |
| **Steering-based** | REPE, PersonaVector, StyleVector 等激活引导方法 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 2）**

| 方法 | Headline (R-1) | Abstract (R-1) | PersonalWAB (ACC) |
|------|----------------|----------------|--------------------|
| ZeroShot | 0.1432 | 0.3503 | 0.6017 |
| ICL (k=10) | 0.1630 | 0.3768 | 0.7966 |
| OPPU (best parametric) | 0.1779 | 0.3848 | 0.8394 |
| SynthesizeMe (reward) | 0.1672 | 0.3906 | 0.8275 |
| **FleetPilot** | **0.1816*** | **0.3990*** | **0.8566*** |

> ✅ 所有指标均显著优于所有基线（p ≤ 0.001）

---

### **与基线方法的对比结果**
- **全面超越**：FleetPilot 在所有任务和指标上均取得最优表现。
- **尤其在复杂任务中优势明显**：在 Abstract Writing 中，FleetPilot 显著优于 OPPU（+1.42 R-1）和 SynthesizeMe（+0.84 R-1）。
- **在极端稀疏场景下鲁棒性强**：即使只有 5% 的训练数据，FleetPilot 仍保持领先（见 Table 5）。

---

### **消融实验结果**

#### **(1) Hooking 策略对比（Table 3）**
验证细粒度钩取的有效性：

| Hooking 方式 | Headline (R-1) | Abstract (R-1) |
|-------------|----------------|----------------|
| Post-Attention | 0.1752 | 0.3938 |
| Post-MLP | 0.1704 | 0.3826 |
| Post-Block | 0.1737 | 0.3946 |
| **Hybrid Strategy (FleetPilot)** | **0.1816*** | **0.3990*** |

> ✅ 双路顺序注入（Sequential Dual-Injection）带来显著增益（+0.79~1.64%），证明细粒度控制的重要性。

#### **(2) 数据效率分析（Table 5 & Fig. 4）**
- 在仅使用 **5% 数据**（平均约6条记录）时，FleetPilot 仍显著优于其他方法。
- 参数化方法在低数据下性能下降明显，而 FleetPilot 下降平缓，体现其高数据效率。

#### **(3) 分布漂移鲁棒性（Table 4）**
在跨时间段测试中，FleetPilot 在所有时间区间均表现最佳，尤其在中间时段（分布变化最大）优势最明显，说明其能快速适应用户偏好漂移。

#### **(4) 开销分析（Table 6）**
- **推理延迟**：相比直接推理，FleetPilot 增加约 10–15% 延迟（主要来自检索）。
- **存储开销**：每用户 <1MB，远低于 LoRA 方法的 ~18MB。
- **无需训练**：更新用户档案只需构建新向量，无需反向传播。

---

## **4. 关键结论和发现**

### **主要发现**
1. **推理时激活引导是高效个性化的可行范式**：FleetPilot 证明了无需微调即可实现高质量个性化，打破了“必须训练”的思维定式。
2. **细粒度控制优于粗粒度**：分别从 Attention 和 MLP 接口提取信号，并顺序注入，能更精准地调控生成行为。
3. **查询条件聚合提升相关性**：动态检索与当前查询最相关的引导向量，避免无关历史干扰，支持实例级适配。
4. **在稀疏与动态环境中极具优势**：FleetPilot 特别适合真实世界中数据少、变化快的场景，性能稳定且可即时更新。

---

### **方法的局限性**
- **依赖高质量的历史响应**：需要用户提供明确的“正例”响应用于 teacher-forcing 模式下的激活提取。
- **检索质量影响性能**：若检索不到相关历史，个性化效果会下降。
- **目前为单层引导**：虽可扩展至多层，但当前实现聚焦于单一层级的干预。
- **未处理负反馈**：仅利用正面交互，尚未整合显式负面偏好信号。

---

### **未来工作方向**
- 扩展至 **多层联合引导**，探索层级间协同效应。
- 引入 **负样本对比**，建模用户不喜欢的内容模式。
- 结合 **隐式反馈**（如停留时间、点击行为）自动构建引导信号。
- 探索 **跨用户迁移**，在冷启动用户上利用相似用户的引导向量进行初始化。

---

> 🔚 **总结**：FleetPilot 是一种**无需训练、轻量高效、灵活实时**的 LLM 个性化新范式，特别适用于**数据稀疏、偏好快速变化的真实应用场景**，为大规模部署个性化 LLM 提供了一条切实可行的技术路径。

</details>

---

### 6. [Chain Dimension Scaling: Communication-Efficient Loop-Style Scaling for Industrial Short-Video Ranking](https://arxiv.org/abs/2610.16415)

**Authors**: Zhimeng Jiang, Yuening Li, Ruixiao Sun, Yuanzhen Lin, Kun Jin, Chuan He, Liang Liu  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.16415v1  

#### Abstract
Industrial recommendation rankers keep scaling in width and depth to billions of parameters, but this increasingly incurs costs in parameter growth, distributed communication overhead, and training inefficiency. We introduce \underline{C}hain \underline{D}imension \underline{S}caling (CDS), a scalin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Chain Dimension Scaling: Communication-Efficient Loop-Style Scaling for Industrial Short-Video Ranking

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
工业级推荐系统中的排序模型（ranker）持续通过增加 **width** 和 **depth** 来提升建模能力，导致参数量达到数十亿级别。这种扩展带来了显著的训练成本问题，尤其是在分布式训练中：
- 参数数量增长导致 **optimizer 开销** 上升；
- 多层独立参数化模块引发频繁的跨设备通信（如 all-gather、reduce-scatter），造成严重的 **communication overhead**；
- 训练效率下降，成为大规模推荐系统的瓶颈。

传统 scaling 方法（宽度、深度、数据量）虽然有效，但会同步增加参数和通信开销，难以兼顾性能与效率。

---

### 提出了什么新方法或新思路
本文提出 **Chain Dimension Scaling (CDS)**，一种正交于 width、depth 和 data scaling 的新型扩展范式。

#### 核心思想：
- 在前向传播过程中，**重复应用同一个 feature-interaction block**，共享其参数；
- 引入一个新的可调维度——**chain length $C$**，即迭代次数，作为新的 scaling 轴；
- 参数总数仅由独立 block 数 $L'$ 决定，而非总计算层数（effective depth = $L' \times C$）；
- 配合轻量级的 **intra-iteration residual connection** 以稳定优化过程。

> ✅ **关键洞察**：在现代分片式数据并行训练（sharded data-parallel training, 如 ZeRO-3 / FSDP）下，通信开销取决于“独立参数化层”的数量，而不是实际计算深度。因此，可以通过参数共享来解耦“参数规模”与“计算深度”。

---

### 相比现有方法的优势

| 维度 | CDS | 传统 Depth Scaling |
|------|-----|------------------|
| 参数增长 | ❌ 不随 chain length 增加 | ✅ 随层数线性增长 |
| 分布式通信开销 | ❌ 仅依赖独立 block 数 $L'$ | ✅ 每层都引入 all-gather / reduce-scatter |
| 可调节性 | ✅ 显式控制 chain length 实现灵活 trade-off | ⚠️ 固定堆叠结构 |
| 扩展方向 | ✅ 支持两个操作模式：<br>• 压缩模式（compression）<br>• 质量扩展模式（quality-scaling） | ⚠️ 主要用于质量提升 |

此外，CDS 是 **parameter-free scaling axis**，可在固定参数预算下提升模型容量，也可在保持性能前提下压缩模型。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 实验基于 Google 内部的大规模 **short-form video (SFV) 推荐平台**；
- 数据来源于真实用户行为日志（interaction logs）；
- 模型服务于每日数十亿次 ranking 请求；
- 多任务目标包括 engagement（点击、观看时长等）和 satisfaction（点赞、负反馈抑制等）；
- 尾部任务标签稀疏（positive rate ~ $10^{-4}$ 到 $10^{-5}$），采用标准 multi-task loss weighting 处理。

> 📌 注：未使用公开 benchmark，而是部署在生产环境进行离线 + 在线 A/B 测试。

---

### 实验设置和评估指标

#### 模型架构背景：
- 输入 token 序列经 embedding 后进入 feature-interaction stack；
- 原始 baseline 使用 $L$ 个独立参数化的 interaction blocks；
- CDS 替换为 $L'$ 个 shared blocks，每个 block 迭代执行 $C$ 次（effective depth = $L' \cdot C$）；
- 输出接入 shared projection 和多个 task-specific towers。

#### 两种运行模式：
| 模式 | 设置方式 | 目标 |
|------|--------|------|
| **Compression Regime** | $L' < L$, $C > 1$, s.t. $L'\cdot C = L$ | 减少参数，提高训练吞吐，维持质量 |
| **Quality-Scaling Regime** | 固定 $L'$，逐步增大 $C$ | 固定参数预算下探索质量上限 |

#### 评估指标：
| 类型 | 指标 |
|------|------|
| **Offline Evaluation** | AUC（分类任务），相对 baseline 的增益（单位：bps, 1 bps = 0.01%） |
| **Online Evaluation** | A/B 测试中的参与度指标（engagement delta），置信区间报告 |
| **效率指标** | 训练 throughput 提升比例、每步 wall-clock 时间分解（forward/backward/optimizer/communication） |

#### 基线方法对比：
- **Production Baseline**: 原始 $L$ 层独立 block 架构（$C=1$）
- **Naive Compression**: 直接减少 block 数量（$L' < L, C=1$），无残差连接
- **Standard Depth Scaling**: $L \in \{1,3,6\}, C=1$
- **CDS Variants**: 不同 residual 设计（no residual / intra-iteration / inter-iteration / combined）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Compression Regime）

| 指标 | 结果 |
|------|------|
| **Trainable Parameters Reduction** | ↓ **26.6%** |
| **Training Throughput Improvement** | ↑ **6.4%** |
| **Offline AUC Degradation** | ≤ **0.03%**（within 1 bps） |
| **Online Engagement Impact** | **+0.06%**（95% CI: [-0.03%, +0.15%]）→ neutral-to-positive |

> 🔹 更激进配置（$L=L/6, C=6$）：参数↓44.4%，吞吐↑12.3%，AUC↓9bps，线上表现轻微负向但不显著。

---

### 与基线方法的对比结果

#### vs. Naive Compression（图3）
- 在相同参数预算下，**CDS + intra-residual** 显著优于直接删层：
  - AUC 高出约 **13 bps**；
  - 线上 engagement 差距达 **+0.15pp**；
- 表明：简单减少层数不可取，而 CDS 能更高效地利用参数。

#### vs. Standard Depth Scaling（图4）
| 模型 | 参数数 | 训练速度（relative） | Offline AUC Gain | Online Engagement |
|------|-------|--------------------|------------------|-------------------|
| Base ($L=3$) | 417M | 1.00× | +9.2 bps | +0.04% |
| Base ($L=6$) | 568M | 0.77× | +11.0 bps | +0.05% |
| **CDS ($L'=3, C=2$)** | **417M** | **0.90×** | **+10.0 bps** | **+0.11%** |

> ✅ CDS 在仅多花 10% 时间、零新增参数的前提下，接近 $L=6$ 模型的质量，并在线上表现反超！

---

### 消融实验结果（Ablation Study）

#### RQ3: Residual Design 对比（图5）
| 设计 | AUC Task1 | AUC Task2 | Online Engagement |
|------|----------|----------|-------------------|
| No Residual (Naive CDS) | ↓13 bps | ↓11 bps | -0.09% |
| **Intra-iteration Residual** | ±1 bp | ↑1 bp | **+0.06%** |
| Intra + Inter Residual | ≈ same | ≈ same | ≈ same |

> ✅ **intra-iteration residual 是关键设计**，极大改善梯度传播和训练稳定性；
> ❌ 添加 inter-iteration residual 并无额外收益，反而略增开销。

#### RQ4: 效率来源分析（图6）
| 成本项 | 改进来源 | 节省时间（ms/step） |
|--------|---------|------------------|
| **Cross-device Communication** | 减少 all-gather / reduce-scatter 次数 | ↓2.41 ms<br>（其中 all-gather ↓1.33ms, reduce-scatter ↓1.08ms） |
| **Optimizer Step** | 参数减少 → 更新量减少 | ↓0.56 ms |
| **Forward Pass** | 总体节省主导项 | ↓2.62 ms |
| **FLOPs / Compute** | ✘ 基本不变甚至微增（conv fusion ↑2.92ms） | N/A |

> 🔍 **核心发现**：CDS 的加速并非来自计算量降低（FLOPs 几乎不变），而是源于：
> - **Reduced communication per layer**
> - **Smaller optimizer state updates**

这颠覆了以往对“循环结构提速”的理解，强调其本质是 **communication-efficient scaling**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Chain length 是一个有效的第四 scaling axis**：
   - 正交于 width、depth、data；
   - 可独立调节，在固定参数下提升 effective depth；
   - 支持 compression 与 quality-scaling 双重用途。

2. ✅ **CDS 是通信高效的 scaling 策略**：
   - 加速主因不是减少 FLOPs，而是降低分布式训练中的 **cross-device synchronization overhead** 和 **optimizer 开销**；
   - 特别适用于现代 sharded data-parallel 框架（ZeRO-3/FSDP）。

3. ✅ **intra-iteration residual 至关重要**：
   - 显著提升长链下的优化稳定性；
   - 是实现高质量压缩的关键组件。

4. ✅ **CDS 在工业场景中具备实用价值**：
   - 生产环境中验证：参数↓26.6%，训练吞吐↑6.4%，质量几乎无损；
   - 在线 engagement 中性偏正，适合大规模部署。

---

### 方法的局限性
- 当前设计依赖固定的 chain length $C$，缺乏动态适应能力（如 ACT 自适应计算时间）；
- 仅应用于 feature-interaction 模块，尚未拓展至其他子网络（如 tower 或 user model）；
- 所有 block 共享参数可能限制表达多样性，极端情况下存在表征坍塌风险；
- 实验集中在 Google 内部系统，通用性需更多外部验证。

---

### 未来工作方向
- 探索 **adaptive chain length**：根据输入复杂度动态调整迭代次数；
- 结合 CDS 与其他 scaling axes（如 width scaling）构建复合 scaling law；
- 将 CDS 思想推广到 MoE、sequence modeling 等其他推荐组件；
- 研究更复杂的循环结构（如 feedback loops、latent reasoning paths）；
- 探索 CDS 在推理阶段的潜力（test-time compute scaling）。

---

## 总结一句话
> **Chain Dimension Scaling (CDS)** 通过参数共享的迭代计算，在不增加参数的情况下引入 chain length 作为新的 scaling 维度，实现了 **通信高效的模型扩展**，在工业短视视频推荐系统中成功达成 **参数压缩、训练加速、质量持平甚至反超** 的多重目标，揭示了 **parameter-shared iterative computation** 作为下一代推荐系统 scaling axis 的巨大潜力。

</details>

---

### 7. [TreeSpark: Learning to Draft the Trees You Verify for Speculative Decoding](https://arxiv.org/abs/2610.20175)

**Authors**: Jihao Zhao, Qianyu Jiang, Shuaishuai Zu, Pengfei Li, Guanghuan Fang, Ben Feng, Biao Qin, Zhibin Wang, Baochuan Yang  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.20175v1  

#### Abstract
Parallel drafters such as DFlash and DSpark reduce proposal latency, but early mismatches limit how much of each block can be accepted. Retaining alternatives introduces parent states and candidate choices that reference-conditioned training does not directly supervise. We introduce TreeSpark, a uni...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TreeSpark: Learning to Draft the Trees You Verify for Speculative Decoding**

---

## **1. 主要贡献和创新点**

### **解决的问题**
现有的 **speculative decoding** 方法（如 DFlash 和 DSpark）虽然通过并行生成草案（drafting）降低了推理延迟，但由于早期匹配失败会导致整个预测块被丢弃，限制了可接受长度（accepted length）。此外，这些方法在训练时仅基于参考序列进行监督，无法有效指导模型学习如何保留多个候选路径及其父子关系，导致在部署时生成的候选树中，许多潜在有用的分支未被充分训练。

### **提出的新方法**
本文提出了 **TreeSpark**，一个统一的训练与推理框架，扩展了半自回归（semi-autoregressive）架构 DSpark，其核心思想是：
- 在一次 backbone 传递中构建以父节点为条件的候选树（parent-conditioned candidate trees）；
- 在训练阶段“回放”（replay）部署时的候选策略（包括扩展、剪枝和打包），使模型在实际使用的状态上进行学习；
- 引入两个新的训练目标：
  - **Survival NLL**：根据参考路径质量对保留边的概率进行加权，提升高价值路径的学习；
  - **Capped Coverage Hinge**：鼓励模型在局部选择边界附近保留目标模型偏好的替代候选，即使它们未被选中。

这两个目标均复用参考模型的目标分布，无需额外的 target forward passes。

### **相比现有方法的优势**
- **更高的接受长度**：相比 DSpark，平均接受长度提升 **23.4%~31.6%**；
- **更高的吞吐量**：单请求吞吐量提升 **15.1%~25.0%**，批处理吞吐量提升 **12.6%~17.2%**；
- **训练与推理对齐**：首次将候选树的选择过程纳入训练监督，实现端到端的策略一致性；
- **无额外计算开销**：不增加 target model 的前向传播次数，也不引入额外评分模块。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：混合了 50,000 条来自数学（GSM8K, MATH）、代码（MBPP, Magicoder）、对话（OASST1, Dolly-15K）和写作（Writer-R1）领域的提示，最大序列长度为 4,096。
- **评估基准**：
  - **通用任务套件（General Benchmarks）**：9个任务，涵盖数学（GSM8K, MATH-500, AIME）、编程（HumanEval, MBPP, LiveCodeBench）和对话（MT-Bench, Alpaca, ArenaHard-v2）；
  - **写作领域（WritingBench）**：6个领域，包括学术工程、广告营销、教育、金融商业、文学艺术、政治法律。

### **实验设置**
- **目标模型（Target Model）**：Qwen3-4B 和 Qwen3-8B，采用 greedy decoding；
- **草案模型配置**：
  - 草案深度 $ B = 7 $
  - 局部分支因子 $ k = 4 $
  - 非根节点预算 $ N = 32 $
  - 总共最多支持 33 个验证位置（含锚点）
- **硬件平台**：Ascend 910B 加速器，测试不同并发度（$ C = 1, 6, 12, 36 $）下的系统表现。

### **评估指标**
- **Acceptance Length**：每轮验证中成功提交的新 token 数量（含 bonus token）；
- **Single-request Throughput**：单个请求的输出 token/s；
- **System Throughput**：系统总输出 token/s，考虑并发和 KV-cache 利用率；
- **Per-request Throughput**：排除首 token 延迟后，每个请求的平均解码速率。

### **基线方法对比**
| 方法 | 类型 | 特点 |
|------|------|------|
| **Vanilla AR** | 自回归 | 基准线 |
| **EAGLE-3** | 自回归草案 | 单路径草案，轻量级特征预测 |
| **DFlash** | 并行草案 | 单一连续路径，block diffusion 结构 |
| **DSpark** | 半自回归草案 | 支持相邻 token 修正，仍为链式结构 |
| **TreeSpark (Ours)** | 树形草案 + 对齐训练 | 多路径保留 + 回放策略训练 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **接受长度（Acceptance Length）**
| 模型 | Qwen3-4B (Mean) | 提升 vs DSpark | Qwen3-8B (Mean) | 提升 vs DSpark |
|------|------------------|----------------|------------------|----------------|
| DSpark | 4.23 | — | 4.20 | — |
| **TreeSpark** | **5.25** | **+24.1%** | **5.18** | **+23.4%** |
| 写作领域（WritingBench） | 2.68 → **3.52** (**+31.4%**) | | 2.55 → **3.35** (**+31.6%**) | |

> 所有 9 个通用任务和 6 个写作领域均取得最优结果。

#### ✅ **单请求吞吐量（Single-request Throughput）**
| 场景 | Qwen3-4B 提升 | Qwen3-8B 提升 |
|------|---------------|---------------|
| 通用任务（General Suite） | **+15.1%** | **+17.9%** |
| 写作任务（Writing） | **+21.9%** | **+25.0%** |

#### ✅ **系统吞吐量（System Throughput）**
| 并发度 $ C $ | Qwen3-4B 提升（vs DSpark） |
|---------------|----------------------------|
| $ C=6 $ | **+16.3%** |
| $ C=12 $ | **+17.2%** |
| $ C=36 $ | **+12.6%** |

> 表明 TreeSpark 在真实并发服务场景下依然显著优于基线。

---

### **消融实验结果（Ablation Studies）**

#### 🔍 **分离训练对齐的影响（DSpark + Tree vs TreeSpark）**
使用相同容量（head dim=192）和推理策略，仅关闭 TreeSpark 的两个损失函数：
| 变体 | 接受长度提升 | 系统吞吐提升（$ C=12 $） |
|------|--------------|--------------------------|
| DSpark + Tree（仅扩宽表示） | — | +4.2% |
| **TreeSpark（完整训练目标）** | **+6.4%** | **+5.0%** |

> 说明 **训练对齐本身带来约 4.2%~6.4% 的增益**，独立于模型容量。

#### 🔍 **各组件贡献分析**
| 消融项 | 系统吞吐提升（vs DSpark） | 影响分析 |
|--------|--------------------------|---------|
| 完整 TreeSpark | **+17.2%** | — |
| w/o $ \mathcal{L}_{\text{surv}} $ | +11.9% | 生存损失影响最大 |
| w/o $ \mathcal{L}_{\text{cov}} $ | +16.0% | 覆盖损失补充边界信号 |
| Head dim 128（原配置） | +14.6% | 宽头提升 +2.0% |

> 结论：**生存损失主导效果，覆盖损失提供互补增益；扩大 attention head 维度也有助于多路径建模。**

---

## **4. 关键结论和发现**

### **主要发现**
1. **训练与推理策略对齐至关重要**：在部署时实际生成的候选树结构上进行监督，能显著提升草案质量；
2. **多路径保留 + 选择感知训练** 是提升 speculative decoding 效率的关键；
3. **Survival + Coverage 双目标协同作用**：前者强化已保留路径的质量，后者防止遗漏潜在优质候选；
4. **无需额外 target 计算即可实现高效训练**：复用 reference-chain 分布，保持训练效率。

### **方法的局限性**
- 当前方法依赖 **greedy verification** 和固定 budget，尚未探索随机采样或动态预算调整；
- 所有实验基于 **Qwen3 系列模型**，泛化性需在更多架构上验证；
- 回放机制增加了训练复杂性和内存占用（尽管可控）；
- 未处理 **end-of-sequence** 或长尾分布下的鲁棒性问题。

### **未来工作方向**
- 将 TreeSpark 框架扩展至 **stochastic speculative decoding**；
- 探索 **adaptive tree budgeting** 和 **cost-aware packing**；
- 结合 **online distillation** 实现持续优化；
- 应用于 **多模态生成** 或 **code generation with complex control flow**；
- 进一步压缩训练开销，例如通过 **gradient checkpointing** 或 **low-rank approximation**。

---

> 📌 **一句话总结**：  
> **TreeSpark 通过“回放部署策略 + 双重监督目标”，首次实现了草案树结构的端到端对齐训练，在不增加 target 计算的前提下，显著提升了 speculative decoding 的接受长度与系统吞吐。**

GitHub 开源地址：[https://github.com/Robot2050/TreeSpark](https://github.com/Robot2050/TreeSpark)

</details>

---

### 8. [Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules](https://arxiv.org/abs/2610.19210)

**Authors**: Sarah Pan, Ali Behrouz  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.19210v1  

#### Abstract
Transformers are efficient to pre-train at scale because attention permits parallel computation across sequence positions, but long-context inference requires increasingly large KV caches. Efficient recurrent alternatives reduce the asymptotic complexity of inference, but their sequential dependence...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **KV Cache 膨胀问题**：传统 Transformer 在长上下文推理时需要维护不断增长的 Key-Value (KV) Cache，导致内存占用高、解码延迟大。
- **训练与推理效率的矛盾**：虽然线性注意力（Linear Attention）和状态空间模型（SSMs）在推理上更高效，但它们通常需要从头训练，无法复用已有的大规模预训练 LLM。
- **信息保留 vs 推理效率的权衡**：现有方法在压缩历史上下文时往往牺牲了对远距离依赖的精确建模能力。

### **提出了什么新方法或新思路**
- **Titan-Llama**：一种**后训练（post-training）方法**，用于为基于 attention 的 LLM 添加外部神经记忆模块（Neural Memory Module, NMM），无需重新训练整个模型。
- **分层记忆架构**：
  - **局部窗口使用 Full Attention**：保留最近 token 的完整注意力计算，确保短期依赖的高保真度。
  - **历史上下文压缩进 NMM**：将早期 token 的 Key-Value 对通过一个固定大小的 RNN 结构（NMM）进行压缩和存储。
- **Test-Time Training (TTT)**：在推理过程中，利用自监督的 Key-Value 重建损失（Key-Value Reconstruction Loss）动态更新 NMM 参数，实现“边推理边学习”。
- **Lambda-Gated 残差融合**：将 NMM 输出作为残差项，与局部 attention 输出加权融合，公式为：
  $$
  O = \lambda \cdot O_{\text{hist}} + (1 - \lambda) \cdot O_{\text{local}}
  $$

### **相比现有方法的优势**
| 方面 | Titan-Llama | 传统方法（如 Mamba、LaCT） |
|------|-------------|---------------------------|
| **是否需重训模型** | ❌ 不需要，仅添加外部模块 | ✅ 通常需要从头训练或全量微调 |
| **参数开销** | ⬇️ 仅增加约 1% 的可训练参数 | ⬆️ 高 |
| **推理效率** | ⬆️ 更低 decode 内存、更快 prefill 和 decode | 中等 |
| **信息保留能力** | ⬆️ 在多种长程任务中接近甚至超越 Full Attention | 通常有明显性能下降 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练阶段**：
  - **FineWeb**：10 亿 token 的网页文本，最大序列长度 16K。
  - **LongAlign**：用于长上下文对齐的监督微调。
  - **LoongRL**：强化学习优化长程推理。
  - **QuALITY 训练集**：用于多选问答任务适配。
- **评估阶段**：
  - **QuALITY**：长输入文本的多选问答任务。
  - **LongHealth**：基于长临床文档的医学问答。
  - **RULER**：系统性评测不同类型的长程回忆能力，包括：
    - S-NIAH（单针检索）
    - Multi-key / Multi-value 检索
    - FWE/CWE（词频统计）
    - UUID / Pass-key 检索等

### **实验设置和评估指标**
- **模型基础**：Llama-3.2-1B-Instruct
- **改造方式**：
  - 将前 4 层的 attention 替换为 blockwise attention + NMM。
  - 每个 KV Head 配备一个 3 层 MLP 构成的 NMM（hidden dim=64）。
- **chunk size**：测试了 512、1024、2048 三种分块大小。
- **评估指标**：
  - **准确率（Accuracy）**：在 QuALITY、LongHealth、RULER 上的多项任务。
  - **推理延迟**：Prefill 和 Decode 阶段的 latency（A100 GPU）。
  - **内存占用**：Decode 阶段的 steady-state 内存使用。
  - **相对 decode 时间**：相对于 Full Attention 的加速比。

### **基线方法对比**
- **Full Attention**：标准的全局 attention 实现。
- **Segmented Attention**：无记忆机制的分段 attention。
- **Mamba-2**：基于 SSM 的高效推理模型。
- **Gated DeltaNet**：结合 delta rule 的 Mamba 变体。
- **LaCT**：基于 TTT 的长上下文方法。
- **E2E-TTT**：端到端 test-time training 方法。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **下游任务准确率提升**
| 模型 | QuALITY | LongHealth |
|------|--------|----------|
| Segmented Attention | 23.87% | 14.50% |
| **Titan-Llama** | **40.51%** (+16.64%) | **40.75%** (+26.25%) |

> 显著恢复了因分段 attention 丢失的语义理解能力。

#### **RULER 基准表现**
- **Pass-key 检索**：在 8K–32K 上达到 **100% 准确率**，与 Full Attention 持平。
- **UUID 检索**：在 8K 和 16K 上 **超过 Full Attention**，分别为 92.0 vs 83.0 和 86.0 vs 64.0。
- **Multi-key Retrieval**：在 4K–16K 上 **显著优于 Full Attention**。
- **Multi-value Retrieval**：仍低于 Full Attention，表明同时提取多个值仍有挑战。

#### **推理效率**
| 指标 | 提升幅度 |
|------|---------|
| **Prefill 速度** | >20% 更快（中长上下文） |
| **Decode 速度** | 最高 **37% 更快** |
| **Steady-State Decode 内存** | **降低 24%** |
| **相对 decode 时间（平均）** | **0.689×**（即比 Full Attention 快约 31%） |

> 在 Figure 2 中显示，Titan-Llama 在 16K–128K 上的语言建模 loss 低于 Full Attention，说明其不仅更快，而且更准。

### **消融实验结果**
- **NMM 的作用验证**：
  - 移除 NMM 后，Segmented Attention 在 RULER 上几乎完全失效（如 Single Needle 从 97.3% 降至 10.7%）。
  - 加入 NMM 后，准确率大幅回升，证明其有效补偿了信息损失。
- **chunk size 影响（Figure 3）**：
  - 更大的 chunk（如 2048）有助于 QuALITY 和 LongHealth 性能。
  - 但在某些 RULER 子任务（如 needle retrieval）上反而下降，说明存在**任务依赖的最优 chunk 大小**。
- **后期层 visibility 实验（Table 3 & 4）**：
  - 若后续 attention 层也限制为局部窗口，则性能显著下降。
  - 表明当前 NMM 尚不能完全替代深层的全局 attention。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **NMM 能有效补偿分段 attention 的信息损失**，在多种长程任务上显著提升性能。
2. ✅ **Test-Time Training + 自监督 KV 重建** 是一种高效的在线学习机制，无需标签即可持续更新记忆。
3. ✅ **Titan-Llama 实现了推理效率与性能的双赢**：既降低了内存和延迟，又提升了准确率。
4. ✅ **方法具有强兼容性**：可在不改动预训练 backbone 的前提下，仅通过添加少量外部参数完成 retrofitting。
5. 🔁 **chunk size 是关键超参**：需根据任务类型权衡局部 attention 范围与记忆更新频率。

### **方法的局限性**
- **Multi-value retrieval 能力有限**：难以同时召回多个独立值。
- **长度外推能力受限**：在超过训练长度（16K）的 32K 上性能有所下降。
- **峰值内存较高**：在 prefill 阶段由于需维护多个 NMM 状态，peak memory 高于 Full Attention。
- **NMM 依赖后续 full-attention 层**：若所有层都改为局部窗口，性能大幅下降（见 Table 4）。

### **未来工作方向**
- 🔄 **探索更大的 NMM 容量或更复杂的架构**（如 Transformer-based NMM）。
- 📈 **研究更优的 inner-loop objective**，例如直接预测 attention output 或 logits。
- 🧠 **深入分析 NMM 的记忆机制**：如何编码、存储和检索语义信息？
- 🌐 **扩展至其他模态**：如视觉-语言模型中的跨模态记忆。
- ⚙️ **优化实现**：开发 fused kernel 以进一步减少中间内存访问和 launch overhead。

---

> **总结一句话**：  
> **Titan-Llama 提出了一种高效、轻量、无需重训的 retrofitting 方法，通过引入 test-time trained Neural Memory Module，在保持预训练 LLM 强大表示能力的同时，实现了长上下文推理的速度与精度双重提升。**

</details>

---

### 9. [Towards Training and Inference of Graph Neural Networks in Graph Databases](https://arxiv.org/abs/2610.19142)

**Authors**: Victor Pekkari, Harry Wik, Alfred Clemedtson, Borun Shi, Xuan-Son Vu  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.19142v1  

#### Abstract
Graph Neural Networks on large graphs commonly rely on in-memory graph storage with ad hoc data processing pipelines for both training and inference. Graph databases provide scalable storage and efficient traversal for graphs that can evolve over time, but their applicability in GNN training and inf...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Towards Training and Inference of Graph Neural Networks in Graph Databases》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前主流的 **Graph Neural Networks (GNNs)** 训练与推理严重依赖于将整个图加载到内存中（in-memory），并使用定制化的数据处理流水线。这带来了以下挑战：
- **高内存开销**：大图无法完全放入RAM。
- **系统复杂性**：需要构建专用的存储和预处理流程。
- **动态图支持差**：图结构变化时需重建索引（如CSR/CSC）。
- **缺乏与图数据库集成**：尽管许多工业级图已存于 **graph DBs** 中，但GNN训练/推理极少直接利用这些系统。

该论文旨在解决 **如何在图数据库（graph DB）中原生支持 GNN 的训练与推理**，减少对内存密集型系统的依赖，并提升端到端效率。

---

### 提出了什么新方法或新思路
作者提出了一种全新的视角：**将 GNN 的前向计算视为一种数据库查询负载（database workload）**，并通过以下方式实现紧耦合：

#### （1）**基于图数据库的 GNN 训练流水线**
- 图数据保留在 **Neo4j** 等图数据库中。
- 每个 mini-batch 的子图采样（sampling）和特征检索（feature fetching）由图数据库完成。
- 只有采样子图及其特征被传送到外部训练进程进行 forward/backward 计算。

#### （2）三种优化策略
| 方法 | 描述 |
|------|------|
| **Cypher Baseline** | 使用声明式 Cypher 查询执行采样与特征获取 |
| **UDP (User-Defined Procedure)** | 用 Java 编写的存储过程，在 DB 内核层控制遍历逻辑，避免中间结果膨胀 |
| **Pre-aggregation** | 在 DB 内部执行第一层 GNN 的邻居聚合（neighborhood aggregation），仅返回聚合后的向量，减少传输数据量 |

#### （3）全内嵌式（Fully In-Database）GNN 推理
- 首次提出并实现了 **完全在图数据库内部执行 GNN 推理** 的方案。
- 支持两种实现方式：
  - **Cypher Query**：通过复杂的 `reduce()` 或原生向量操作表达矩阵乘法。
  - **UDP Implementation**：将训练好的模型权重部署为 Java UDP，直接在 DB 中完成完整 forward pass。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **训练灵活性** | 无需将整图载入内存，适合大规模、动态更新的图 |
| **系统简化** | 复用现有的图数据库基础设施，无需额外构建 ETL 流水线 |
| **推理效率** | 全内嵌推理可 >10x 快于冷启动加载图至 RAM，且无额外内存占用 |
| **通用性** | 支持任意训练框架（PyTorch, TensorFlow等），模型可导出后在 DB 中运行 |
| **实时性支持** | 特别适用于图持续演化的场景，无需重新导出或缓存 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖多个标准 benchmark 数据集，从小规模到超大规模：

| 数据集 | #Nodes | #Edges | 平均度数 | 特征维度 |
|--------|--------|--------|----------|-----------|
| **Cora** | 2.7K | 10.6K | 3.9 | 1,433 |
| **CoAuthor Physics** | 34.5K | 495.9K | 14.4 | 8,415 |
| **arXiv** | 169.3K | 1.17M | 6.9 | 128 |
| **ogbn-products** | 2.4M | 61.9M | 25.3 | 100 |
| **ogbn-papers100M** | 111M | 1.62B | 14.6 | 128 |

> 注：`papers100M` 是十亿边级别的真实学术引用网络。

---

### 实验设置和评估指标

#### 硬件环境
- 单机：32 vCPUs, 256 GB RAM
- 存储：4块 NVMe 组成 RAID 0，用于 Neo4j 存储
- 软件栈：Python 3.12, PyTorch 2.7.1, PyG 2.7, Neo4j (Java UDP)

#### 模型配置
- GNN 架构：GraphSAGE 和 GraphSAINT（带采样）
- 层数：2-layer
- 采样策略：Fanout `[10, 5]`

#### 评估指标
| 类型 | 指标 |
|------|------|
| **正确性验证** | KL 散度（节点/边分布）、验证准确率曲线对比 |
| **训练性能** | 各阶段耗时分解（采样、fetching、compute）、端到端每 batch 时间 |
| **推理性能** | 不同 seed node 数量下的延迟（latency）、内存使用情况 |
| **通信开销** | 从 DB 传出的数据字节数 |

---

### 基线方法对比

#### 训练部分
| 方法 | 描述 |
|------|------|
| **In-memory (PyG)** | 图全加载进 RAM，使用 PyG 自带 sampler（参考基准） |
| **Neo4j + Cypher** | 用 Cypher 查询执行采样与特征提取 |
| **Neo4j + UDP** | 用 Java UDP 实现高效采样 |
| **Neo4j + Pre-aggregation** | DB 内完成首层 aggregation |

#### 推理部分
| 方法 | 描述 |
|------|------|
| **Cold Load (Full RAM)** | 每次推理前从磁盘加载图到 RAM |
| **Hot Load (Full RAM)** | 图已在 RAM 中（理想状态） |
| **Neighbor Sampling (External)** | 图存在 DB，外部进程采样 + 推理 |
| **In-DB Inference (UDP)** | 完整 forward pass 在 DB 内执行（Java UDP） |
| **In-DB Inference (Cypher)** | 用单条 Cypher 查询实现推理（含优化版 `in_db_cypher_opt`） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 正确性验证
- 所有基于 Neo4j 的实现与 PyG 原生实现的 **KL divergence < 0.01**
- 验证准确率曲线高度一致（见 Figure 2），证明功能等价

#### ⏱️ 训练速度提升
| 方法 | 性能表现 |
|------|---------|
| **UDP vs Cypher (Sampling)** | - ogbn-papers100M 上快 **3.8x**<br>- ogbn-products（更密）上快 **5.2x** |
| **UDP vs Cypher (Feature Fetching)** | 显著降低网络传输时间（序列化为 byte array） |
| **Overall Speedup (vs Cypher baseline)** | **最高达 2.5x 加速**（得益于 UDP + Pre-aggregation） |

#### 💾 数据传输减少（Pre-aggregation）
- 对于高维特征数据（如 CoAuthor Physics, dim=8415）：
  - 数据传输量减少 **~60%**
  - Feature fetching 延迟 **降低超过一半**
- 对低维特征（如 papers100M, dim=128）：
  - 数据量仍减少 40–60%，但由于 DB 内计算开销，未显著提速

> 见 Figure 6 & 7：Pre-aggregation 在高维特征下收益明显。

#### 🚀 推理性能突破
| 方法 | 表现 |
|------|------|
| **In-DB UDP Inference** | - 对 2048 个 seed nodes，<1s 完成推理（32GB RAM）<br>- 比“冷加载”图到 RAM **快 >10x**<br>- 内存占用极低（只需 materialize 当前 batch 节点） |
| **In-DB Cypher Inference** | 较慢，因查询规划器非专为 GNN 设计 |
| **Memory Efficiency** | 全图无需驻留内存，避免 OOM（PyG 加载 papers100M 直接崩溃） |

> 见 Figure 8：随着 batch size 增加，in-DB UDP 推理始终保持高效。

---

### 消融实验结果
| 实验 | 发现 |
|------|------|
| **Cypher vs UDP** | UDP 减少不必要的中间 tuple materialization，尤其在高平均度图中优势显著 |
| **Projection & OrderedAggregation 开销** | 在 Cypher 中占主导，说明声明式模型有固有开销 |
| **Expand(All) 扩展过度** | 图中实际采样约 7808 节点，但 Expand 操作产生 >10,000 行（冗余） |
| **Embedding 存储格式优化** | 使用 `byte[]` 替代 `f64[]` 可使驱动解码时间减半（见 Figure 9） |

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **图数据库可以作为 GNN 训练的有效支撑平台**  
   - 支持 on-the-fly sampling 与 feature retrieval，无需整图入内存。
   
2. ✅ **UDP 实现显著优于 Cypher 查询**  
   - 更细粒度控制遍历过程，避免中间结果膨胀，带来 **2.5x 训练加速**。

3. ✅ **Pre-aggregation 在高维特征下极具价值**  
   - 将参数无关的聚合移至 DB 内，大幅减少传输数据量和内存压力。

4. ✅ **首次实现全内嵌式（fully in-DB）GNN 推理**  
   - 可独立于训练方式运行，特别适合生产环境中不断演化的图数据。

5. ✅ **In-DB 推理比冷加载快 >10x，且内存友好**  
   - 为资源受限或图频繁变更的应用提供了新范式。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **仅测试 Neo4j** | 结论是否适用于其他图数据库（如 Kuzu、JanusGraph）尚待验证 |
| **仅支持 GraphSAGE / GraphSAINT** | 未涵盖 Layer-wise Sampling 或其他高级采样技术 |
| **未结合分布式训练/GPU** | 当前为单机设置，扩展性有待研究 |
| **UDP 开发门槛较高** | 需要 Java 编程能力，不如 Cypher 易用 |

---

### 未来工作方向
1. **支持更多 GNN 架构与采样策略**  
   如 GCN、GAT、ClusterGCN 等。
   
2. **集成 GPU 加速与分布式训练**  
   将 in-DB 采样与 GPU 训练结合，进一步提升吞吐。

3. **自动查询优化器生成 GNN 查询计划**  
   类似传统 DB 中的 query optimizer，自动生成高效的 GNN 执行路径。

4. **探索增量更新机制（Incremental Inference）**  
   图发生变化时只更新受影响节点的 embedding（类似 Ripple [16]）。

5. **跨图数据库移植性研究**  
   验证本方法在不同 backend 上的通用性和性能差异。

---

> 🔗 **开源代码**：https://github.com/harrywik/neo4j-mpgnn-thesis  
> 包含完整的 Cypher 查询、Java UDP 实现、复现实验脚本。

</details>

---

### 10. [Hardware-Friendly Token-Group Activation Quantization for Low-Bit Mamba Super-Resolution](https://arxiv.org/abs/2610.16261)

**Authors**: Jinwoo Chung, Jangho Kim  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.16261v1  

#### Abstract
Mamba-based super-resolution models are efficient alternatives to Transformer-based restoration backbones, but their low-bit post-training quantization remains challenging because selective scan, Mamba's input-dependent state-space operator, induces different activation ranges across spatial tokens....

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Hardware-Friendly Token-Group Activation Quantization for Low-Bit Mamba Super-Resolution**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **低比特量化在 Mamba-based 超分辨率模型中的挑战**：  
  Mamba 模型通过 **selective scan** 操作捕获长程依赖，但该操作导致不同空间 token 的激活范围差异显著（token-wise activation heterogeneity），使得传统的层级统一剪裁范围（layer-wise clipping range）无法有效适配所有 token。
- **硬件部署限制**：  
  动态分组（如基于后扫描激活进行分组）虽然理论上更优，但在静态低比特推理中不可行，因为它需要运行时统计和动态更新，破坏了 ahead-of-time 编译。

### **提出的新方法**
- **Token-Group Quantization (TGQ)**：一种硬件友好的静态激活 PTQ 框架，核心思想是：
  - **离线构建“下游 Oracle”**：利用后扫描激活的描述符（descriptor）进行 K-means 聚类，生成理想的 token 分组标签（pseudo-labels）。
  - **训练轻量级预测器**：从**前扫描特征**（pre-scan features）中训练一个共享的两层 FFN 预测器，以预测 token 所属的组别。
  - **静态查找表机制**：推理时，预测器输出组索引，用于从离线校准的**组级剪裁边界表**（group-wise clipping-bound tables）中选择对应的量化参数，无需运行时统计或动态更新。

- **Frequency-Preserving Refinement (FPR)**：  
  为缓解低比特量化对高频纹理和边缘的退化，提出了一种教师监督的剪裁边界优化目标，通过**对齐全精度教师模型和量化学生模型输出的傅里叶幅度和相位**来保留高频结构。

### **相比现有方法的优势**
- **有效性**：显著优于现有的 Mamba 和 SR PTQ 方法，在多种 backbone、任务、bit-width 下均取得最优性能。
- **硬件友好性**：避免了运行时分组和动态剪裁更新，支持静态编译和 CUDA graph，实现真正的低比特加速。
- **通用性**：可迁移至 Transformer SR（如 SwinIR）和 Vision Mamba 分类任务，并能与通道轴变换方法（如 SpinQuant）组合使用，进一步提升性能。
- **高效性**：在 Jetson Orin Nano 上实现 **2.23× 的速度提升**（相对于 FP16）。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **校准集 (Calibration)**：DF2K 数据集的 96 张低分辨率补丁（无高分辨率标签）。
- **测试集 (Evaluation)**：
  - **超分辨率**：Set5, Set14, BSD100, Urban100, Manga109
  - **去噪**：CBSD68, Kodak24, McMaster
  - **JPEG 伪影去除**：Classic5, LIVE1
  - **分类**：ImageNet

### **实验设置和评估指标**
- **主干网络 (Backbone)**：
  - 主要：MambaIRv2-Light/Base
  - 对比：MambaIRv1, SwinIR-light, Vision Mamba
- **量化设置**：
  - **Weight-freeze PTQ**：仅校准量化参数，不更新权重。
  - **Weight-training PTQ with LoRA**：结合 LoRA 适配器微调。
  - **量化位宽**：W4A4, W2A2
- **评估指标**：
  - **超分辨率**：PSNR, SSIM（Y 通道）
  - **分类**：Top-1 Accuracy
- **硬件平台**：NVIDIA RTX A6000, Jetson Orin Nano

### **基线方法对比**
- **Mamba-specific PTQ**：PTQ4VM, Quamba, MambaQuant
- **SR-oriented PTQ**：OAQ-SR, 2DQuant
- **Transformer PTQ**：FIMA-Q, APHQ-ViT, SPR2Q
- **通道轴变换方法**：SpinQuant, AffineQuant, Hadamard (QuaRot)

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **MambaIRv2-Light (×4, W4A4)**
| Method       | Set5 PSNR | Set14 PSNR | Urban100 PSNR |
|--------------|----------|------------|---------------|
| Full Precision | 32.51    | 28.84      | 26.82         |
| PTQ4VM        | 30.82    | 27.69      | 24.76         |
| Quamba       | 31.01    | 27.77      | 25.01         |
| OAQ-SR       | 30.40    | 27.41      | 24.31         |
| **Ours**     | **31.46**| **28.14**  | **25.39**     |

#### **MambaIRv2-Light (×2, W4A4)**
| Method       | Set5 PSNR | Set14 PSNR | Urban100 PSNR |
|--------------|----------|------------|---------------|
| Full Precision | 38.26    | 34.09      | 33.26         |
| PTQ4VM        | 37.17    | 32.86      | 30.47         |
| Quamba       | 37.07    | 32.77      | 30.54         |
| OAQ-SR       | 37.22    | 32.60      | 30.49         |
| **Ours**     | **37.56**| **33.24**  | **31.34**     |

> ✅ 在多个数据集上，Ours 均显著领先，接近全精度性能。

### **与基线方法的对比结果**
- **在 W4A4 下**：Ours 平均比最强基线（OAQ-SR）高出约 **0.3–0.8 dB**。
- **在 W2A2 下**：优势更加明显，例如在 ×4 超分任务中，Ours 比 OAQ-SR 高出 **0.81 dB**（Urban100）。
- **在 LoRA 协议下**：Ours+LoRA 几乎完全恢复全精度性能，例如在 ×2 超分中，BSD100 上仅下降 **0.12 dB**。

### **消融实验结果**
#### **(1) 分组数量 $K$ 的影响 (Table 6)**
| $K$ | Set14 PSNR | Urban100 PSNR |
|-----|-----------|--------------|
| 1   | 32.62     | 30.59        |
| 4   | 33.07     | 31.03        |
| 8   | **33.24** | **31.34**    |
| 16  | 33.37     | 31.71 (on Base) |

> ✅ 性能随 $K$ 增加而提升，$K=8$ 已接近饱和。

#### **(2) 组件分析 (Table 7)**
| 方法                         | Set14 PSNR | Urban100 PSNR |
|------------------------------|------------|--------------|
| Layer-wise PTQ               | 32.45      | 30.34        |
| + Token-Group Quantization   | 33.06      | 30.99        |
| + FPR (Ours)                 | **33.24**  | **31.34**    |

> ✅ Token-Group 贡献了约 77% 的增益，FPR 进一步带来 **0.35 dB** 提升。

#### **(3) 分组策略对比 (Table 8)**
| 分组方式             | Set14 PSNR | Urban100 PSNR |
|----------------------|------------|--------------|
| Layer-wise           | 32.45      | 30.34        |
| Random groups        | 31.76      | 29.91        |
| Pre-scan K-means     | 32.84      | 30.77        |
| Post-scan Oracle (upper bound) | 33.41      | 31.57        |
| **Predicted (Ours)** | **33.24**  | **31.34**    |

> ✅ Ours 恢复了 Oracle 约 **82%** 的增益，且无需运行时统计。

#### **(4) 推理速度 (Table 5)**
| 平台              | 方法       | 速度提升 |
|-------------------|------------|---------|
| NVIDIA RTX A6000  | Ours (W4A4)| 1.70×   |
| Jetson Orin Nano  | Ours (W4A4)| **2.23×** |

> ✅ 实现了显著的实际加速，远超动态 Oracle 方法（后者甚至慢于 FP16）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **selective scan 是低比特 Mamba SR 退化的根本原因**：它显著放大了 token 间的激活范围差异（平均 1.80×），使得统一剪裁范围失效。
2. **Token-Group Quantization 是有效的静态解决方案**：通过前扫描预测 + 离线索引表，实现了接近动态 Oracle 的性能，同时保持硬件友好性。
3. **FPR 显著提升高频保真度**：通过频域对齐，有效缓解了量化对纹理和边缘的模糊。
4. **方法具有强泛化能力**：成功迁移到 SwinIR、Vision Mamba 分类，并能与 SpinQuant 等通道轴方法组合，实现协同增益。

### **方法的局限性**
- **引入少量推理开销**：预扫描预测器和查表操作带来了轻微的延迟增加（约 2–3%），尽管远小于动态方法。
- **分组数 $K$ 需要设定**：虽然 $K=8$ 在多数情况下表现良好，但最优值可能随模型规模变化。
- **依赖预训练模型质量**：作为 PTQ 方法，其性能上限受限于原始全精度模型。

### **未来工作方向**
- **探索在其他边缘加速器上的部署**：如 NPU、DSP、移动 GPU 或厂商专用低比特运行时。
- **自适应分组机制研究**：探索在不牺牲硬件效率的前提下，如何实现更灵活的分组策略。
- **扩展到更多视觉任务**：如检测、分割等密集预测任务，验证其在更广泛场景下的有效性。

--- 

> **总结**：本文提出的 **Token-Group Quantization** 框架有效解决了 Mamba-based 超分辨率模型在低比特量化中的核心挑战，通过**静态分组预测**和**频域保真优化**，在保持硬件友好性的同时，实现了显著的性能提升和实际加速，为高效边缘部署提供了可靠方案。

</details>

---

### 11. [The Cost of Context: Profiling the Energy Footprint of Input Tokens in Large Language Models](https://arxiv.org/abs/2610.19819)

**Authors**: Boris Ruf, Marcin Detyniecki  
**Category**: cs.DC  
**Published**: 2026-10-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.19819v1  

#### Abstract
The rapid adoption of Retrieval-Augmented Generation (RAG) and long-context LLMs has shifted the computational burden toward the prefill phase---the initial stage where the model processes all input tokens in parallel before generating a response---yet its energy impact remains poorly characterised....

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*The Cost of Context: Profiling the Energy Footprint of Input Tokens in Large Language Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文聚焦于当前大语言模型（LLM）推理中一个看似矛盾的现象——**“Prefill Paradox”**（预填充悖论）：
- 硬件层面观测到 `prefill` 阶段出现高达 3× 的 GPU 功率峰值；
- 但系统级研究却报告输入 token 的边际能耗呈**亚线性增长**，似乎影响微弱。

这一矛盾掩盖了长上下文场景下（如 RAG、长文档摘要）真实能效瓶颈的本质。本文旨在揭示：  
> **输入 token 到底如何影响 LLM 推理的能源消耗？其成本随上下文长度如何变化？**

### 🚀 提出的新方法与新思路
- **提出“context-scaling laws”框架**：首次系统性地将 LLM 推理的能耗分解为 `prefill` 和 `decode` 两个阶段，并量化它们在不同上下文长度下的能量强度（Energy Intensity, mJ/token）。
- **识别“效率阈值”（Efficiency Threshold）**：定义了一个关键转折点——当上下文长度超过 GPU 并行处理能力时，O(n²) 的 self-attention 复杂度开始显现，导致能耗急剧上升。
- **区分引擎优化与模型固有约束**：明确指出当前观察到的高能耗更多源于内存调度策略（如 KV-cache 管理），而非不可改变的模型计算复杂度。

### 🔍 相比现有方法的优势
| 维度 | 传统方法局限 | 本文改进 |
|------|--------------|----------|
| 能耗估算方式 | 多基于输出 token 数量，忽略输入代价 | 显式建模输入 token 的能耗，尤其适用于 RAG 等长输入场景 |
| 分析粒度 | 系统级粗略估计 | 结合硬件级 telemetry（NVML, 1ms 采样）与 phase-level 控制 |
| 对比维度 | 忽视 inference engine 差异 | 对比 vLLM（PagedAttention）与 HuggingFace（SDPA）的工程优化差异 |

---

## 2. 核心实验方法和设置

### 📊 数据集与输入构造
- **不依赖真实任务数据集**，而是采用**合成 prompt** 来精确控制变量。
- 使用高熵基础文本重复生成长度为 $ L \in [2^8, 2^{17}] $（即 256 ~ 131,072 tokens）的输入序列。
- 使用模型原生 tokenizer（Llama-3.1 128k, Qwen-2 1M）确保 token 数准确。
- 所有输入语义无关，仅用于测试**序列长度对能耗的影响**。

### ⚙️ 实验设置
- **硬件平台**：NVIDIA A100 (80GB)，bfloat16 精度。
- **监控工具**：
  - 使用 **NVML API** 以 **1ms 分辨率**采集 GPU 功耗。
  - 时间戳对齐 `prefill_start` 与 `prefill_end` 事件标记。
- **模型**：
  - Llama-3.1-8B
  - Qwen-2-7B
- **推理引擎对比组**：
  1. **HF Baseline**：HuggingFace Transformers 默认实现，使用 SDPA + 连续 KV-cache 分配。
  2. **vLLM (Optimized)**：启用 PagedAttention 内存管理机制。

### 🎯 评估指标
| 指标 | 定义 | 单位 |
|------|------|------|
| `E_prefill` | Prefill 阶段总能耗 | Joules |
| `Energy Intensity (Ein)` | 输入 token 能耗密度 | mJ/token |
| `Eout` | 输出 token 平均能耗 | mJ/token |
| `Eout / Ein` | 输入 vs 输出能耗比 | × |
| Latency per token | 每 token 延迟 | ms/token |

### 🔁 实验设计流程
1. **Phase 1: Prefill Isolation**
   - 固定生成长度为 1 token，测量纯 `prefill` 能耗。
2. **Phase 2: Comparative Intensity Protocol**
   - 固定输出长度 G=128，减去已知 `E_prefill` 得到 `E_decode`，进而计算 `Eout`。
3. **Warm-up & Determinism**
   - 每个配置前进行预热运行，稳定 GPU 状态。
   - 所有实验为确定性执行（greedy decoding, no batching），单次运行即可代表结果。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 在 ~112,000 tokens 上的能效对比（表1）
| Model / Engine | Latency (s) | Total Energy (J) | Energy per Token (J/Tok) |
|----------------|-------------|------------------|----------------------------|
| Llama-3.1-8B (HF) | 25.54 | 7,600 | 0.0678 |
| Llama-3.1-8B (vLLM) | 3.54 | 1,005 | 0.0090 |
| Qwen-2-7B (HF) | 23.04 | 6,832 | 0.0610 |
| Qwen-2-7B (vLLM) | 2.71 | 769.6 | 0.0069 |

👉 **vLLM 实现最高达 8.9× 的能效提升**（Qwen-2 场景）。

---

#### ✅ 输入 vs 输出 token 能耗比（Figure 2）
| 设置 | 小上下文 (1k) | 大上下文 (112k) | 变化趋势 |
|------|---------------|------------------|---------|
| vLLM (优化引擎) | ~850× | ~750× | 缓慢缩小 |
| HF Baseline (未优化) | ~155–220× | ~160–220× | 几乎恒定 |

> 💡 解读：vLLM 的 decode 极高效，因此即使 prefill 成本随 n² 上升，仍保持较大差距；而 HF 自身 prefill 就很贵，拉近了与 output 的距离。

---

#### ✅ 效率阈值位置差异
- **HF Baseline**：在约 **16,384 tokens** 后进入 compute saturation cliff，延迟开始线性上升。
- **vLLM**：得益于 PagedAttention 和 kernel fusion（FlashAttention），延迟平坦直到 **>32,000 tokens**。

👉 表明优化引擎可将效率阈值推迟一个数量级。

---

#### ✅ 功耗与延迟解耦现象（Figure 3）
- 当 context > 16k，GPU 功耗已达上限（~310W），不再增加。
- 但 HF 的 latency 持续上升 → 能耗增长来自**时间延长**，而非功率提高。
- 支持假设：后期能耗由 memory bandwidth wall 主导（大量 VRAM 数据搬运）。

---

## 4. 关键结论和发现

### 🔑 主要发现
1. **“Prefill Paradox”并非矛盾，而是并行效率的体现**：
   - 高瞬时功耗是 GPU 充分利用并行性的表现，反而有助于摊薄单位 token 能耗。
   - “Race to Finish”原则成立：快速完成 prefill 更节能。

2. **真正的瓶颈在于 memory orchestration，而非算术本身**：
   - O(n²) 注意力复杂度是否显现，取决于是否突破 GPU 并行容量。
   - **PagedAttention 等技术可通过减少内存碎片和数据移动，显著延缓效率拐点的到来**。

3. **输入 token 的能耗不可忽视，但在优化系统中仍远低于输出 token**：
   - 在 vLLM 中，output token 仍比 input 耗能约 **750–850×**。
   - 但在 HF 中该比例仅为 ~150–230×，反映的是 prefill 本身的低效。

4. **现代 LLM 能效优化重心应从模型转向基础设施层**：
   - 引擎级别的优化（如 vLLM）带来的收益远超单纯扩大 batch 或调整 prompt。

---

### ⚠️ 局限性
- 实验仅限于 **text-only** 模型，未考虑多模态输入（image/video 的 prefill 成本可达文本的 15×）。
- 测试最大上下文为 ~112k，尚未覆盖真正百万 token 级应用（如 Llama-3 1M）。
- 未涉及 MoE 架构或更大参数模型（70B+），其激活模式可能改变 scaling 规律。

---

### 🔮 未来工作方向
1. **高吞吐摊销研究**：
   - 探索 request batching 和 continuous batching 如何进一步压缩 decode 阶段能耗。
2. **精度-能效权衡分析**：
   - 评估 FP8 / INT4 量化是否真正在内存流量上带来净收益，还是被专用单元功耗抵消。
3. **扩展至 MoE 与超大规模模型**：
   - 研究 Mixture-of-Experts 架构下，prefill 与 decode 的能耗分布是否会因稀疏激活而改变。
4. **碳足迹映射增强**：
   - 结合 LLMCO2 等工具，将实测能耗转化为动态碳排放指标，支持绿色 AI 决策。

---

> 📌 **一句话总结**：  
> 本文通过细粒度硬件 telemetry 揭示，LLM 输入 token 的能耗并非固定低廉，而是受 inference engine 内存调度策略深刻影响；**PagedAttention 等优化可将能效拐点推迟三个数量级，在 112k 上实现近 9 倍节能优势**，从而重新定义可持续 LLM 部署的设计空间。

</details>

---

### 12. [Queue-Aware Token Communications: Bridging AI Representation and Wireless Service](https://arxiv.org/abs/2610.16730)

**Authors**: Yuquan Xiao, Qinghe Du, Chaoyang Zhang, Wenchi Cheng, Jingqing Wang, Tao Chen, Jiangzhou Wang  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.16730v1  

#### Abstract
The rapid advancement of foundation models and edge intelligence is transforming wireless networks from data-oriented communication systems into artificial intelligence (AI)-native service platforms. Token communications (TokenCom) has emerged as a promising paradigm that directly transmits task-ori...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Queue-Aware Token Communications: Bridging AI Representation and Wireless Service

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统通信系统以传输原始数据（bit-level）为目标，而现代AI服务更关注**任务相关的语义信息传递**。尽管已有研究提出 **Token Communications (TokenCom)** 范式，直接传输AI模型中的token表示来提升效率，但这些工作普遍假设生成的token可以立即被传输。

然而，在实际场景中，由于无线信道动态变化（如衰落、干扰）、资源受限以及持续感知任务带来的连续token生成，**token workload的到达过程与无线服务能力之间存在时间错配**，导致队列积压和额外延迟。现有方法忽略了这一**queueing dynamics**，无法准确建模真实AI服务的端到端性能。

### 提出了什么新方法或新思路
本文提出 **Queue-Aware TokenCom**，首次将**token队列机制显式引入AI通信管道**，构建了一个联合优化AI表征生成、token压缩、队列管理与无线传输的端到端架构。

核心创新包括：
- **引入Token Queue Model**：用离散时间队列模型 $ Q(t+1) = [Q(t) - S(t)]^+ + A(t) $ 描述token workload的动态演化。
- **定义Token Budget作为关键接口**：token budget（可表现为数量、维度、精度等）成为连接AI性能与通信效率的关键控制变量。
- **提出双向优化框架**：
  - **Budget-aware Token Representation Optimization**：在给定预算下最大化AI任务效用。
  - **Queue-aware Token Service Optimization**：基于队列状态和信道状态动态调整服务策略。

### 相比现有方法的优势
| 维度 | 现有TokenCom方法 | Queue-Aware TokenCom |
|------|------------------|------------------------|
| 队列建模 | 忽略或隐含假设即时传输 | 显式建模token queue dynamics |
| 控制机制 | 单向：仅优化表征 | 双向耦合：表征 ↔ 队列 ↔ 无线服务 |
| 动态适应性 | 固定压缩策略 | 支持自适应token budget控制 |
| 性能权衡分析 | 仅关注accuracy vs. bandwidth | 引入accuracy-delay tradeoff分析 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Imagenette**：ImageNet的一个简化子集，用于图像分类任务验证。
- 模型采用 **DeiT-Tiny/16** vision transformer 架构。

### 实验设置
- 输入图像划分为 $14 \times 14 = 196$ 个patches，加上一个CLS token，共197个初始tokens。
- **Transmitter-side submodel**：前3层Transformer处理tokens，并通过**adaptive Top-K token selection module**选择K个最具信息量的tokens。
- **Receiver-side submodel**：剩余9层Transformer + classification head完成推理。
- 所选tokens进入**token queue**，经由时变无线信道传输。
- 无线信道建模为**一阶高斯马尔可夫过程**的瑞利衰落信道：
  $$
  h(t) = \rho h(t-1) + \sqrt{1 - \rho^2} w(t),\quad \rho = 0.99
  $$
- 任务到达间隔为10个时隙（slot），每个slot为10ms。

### 评估指标
| 类别 | 指标 |
|------|------|
| **AI Performance Metrics** | Classification Accuracy (%) |
| **Communication Performance Metrics** | Average task-level communication delay (slots), Queue length, Deadline violation probability |
| **Joint Tradeoff** | Accuracy-Delay Tradeoff Curve |

### 基线方法对比
- **Fixed Token Budget Strategy**：每项任务使用固定数量的tokens（如K=5, 10, 25, 50等），作为静态基准。
- **Proposed Adaptive Token Budget Strategy**：根据当前队列长度 $Q(t)$ 和预测的服务能力 $C(t)$ 动态决定token budget：
  $$
  K_{\text{raw}}(t) = \max\{1, \alpha (C(t) - Q(t))\}
  $$
  其中$\alpha$控制激进程度，最终投影到预设集合 $\mathcal{K}_{\text{set}} = \{1,5,10,25,50,100\}$。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 固定预算策略下的Accuracy-Delay Tradeoff（图4）
| Token Budget $K$ | Accuracy (%) | Avg. Delay (slots) |
|--------------------|-------------|---------------------|
| 25                 | 94          | 14.94               |
| 50                 | 96          | 86.06               |
| 100                | 97.24       | >100                |

> ⚠️ 观察：从K=25到K=50，accuracy仅提升2%，但delay增加约5.76倍，表明**边际收益递减且队列代价高昂**。

#### 自适应预算策略性能（图4）
| 目标Accuracy (~%) | 固定预算方案（Delay） | 自适应策略（Delay） | 延迟降低幅度 |
|-------------------|------------------------|-----------------------|--------------|
| 89                | K=5 → 2.94 slots       | $\alpha=0.1$ → 1.49 slots | ↓ ~50%       |
| 92                | K=10 → 5.17 slots      | $\alpha=0.4$ → 3.57 slots | ↓ ~31%       |

✅ 结果显示：**自适应策略在相同或更高accuracy下显著降低通信延迟**，尤其在低至中等延迟区域表现优越。

### 与基线方法的对比结果
- 在相似accuracy水平下，**自适应策略比最优固定预算策略平均减少30%-50%的任务级通信延迟**。
- 自适应策略能有效避免拥塞期的过度负载注入，平滑队列波动。

### 消融实验结果（图5）
- **Token Budget Distribution vs. Training Frequency**：
  - 当某个budget值（如K=50）在训练中采样频率更高时，其对应的per-budget accuracy也更高。
  - 例如：当$\alpha$增大时，大budget（K=100）出现概率上升，其accuracy从~97.6%升至~98.0%。
- **发现**：共享参数的AI模型倾向于“记忆”高频出现的budget配置，对稀有budget的泛化能力下降。
- **启示**：**训练阶段的budget分布必须与部署环境匹配**，否则会导致性能偏差。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **Token Budget是AI性能与通信效率之间的根本权衡变量**，需作为联合设计的核心参数。
2. ✅ **显式建模token queue dynamics至关重要**，忽略队列效应会严重低估实际延迟并误导系统设计。
3. ✅ **自适应token budget控制优于固定策略**，能在动态无线环境中实现更好的accuracy-delay tradeoff。
4. ✅ **AI模型训练必须考虑budget分布的一致性**，否则会出现性能偏移（distribution mismatch problem）。
5. ✅ **TokenCom是一个双向闭环系统**：队列状态影响budget决策，而budget又反过来影响后续队列行为和AI性能。

### 方法的局限性
- 当前研究聚焦于**单用户、单一任务类型**（图像分类），尚未扩展到多任务、多智能体场景。
- 自适应控制依赖信道相关性（$\rho=0.99$），在高速移动等低相关性场景下预测能力下降。
- 所提Top-K selection模块虽轻量，但仍引入额外计算开销，未量化边缘设备上的能耗影响。
- 缺乏真实硬件平台验证，仍为仿真层面的结果。

### 未来工作方向
1. **Closed-Loop TokenCom Optimization**  
   构建反馈机制，使AI模型、token生成策略与队列控制器能够实时交互与协同优化。

2. **Task-aware and Risk-Sensitive TokenCom**  
   针对不同任务需求（如自动驾驶要求低延迟+高可靠性），引入deadline violation probability、task failure rate等风险敏感指标进行优化。

3. **Scalable Multi-Agent TokenCom Networks**  
   扩展至多机器人、车联网等场景，研究语义感知的token调度、协作式队列管理与分布式AI推理机制。

4. **Semantic Relationship Exploitation**  
   利用多个agent间的语义关联性，支持跨节点token融合与协同压缩，进一步提升整体系统效率。

---

> 🔚 **总体展望**：Queue-Aware TokenCom为AI-native网络提供了新的系统视角，推动通信范式从“传比特”向“传智能”的演进。该框架有望在未来6G、具身智能、边缘AI等场景中发挥重要作用。

</details>

---

### 13. [An FPGA-Based Spiking Neural Network for Spatiotemporal Pattern Learning and Classification](https://arxiv.org/abs/2610.17463)

**Authors**: Jingang Jin, Yue Ma, Zhenhang Zhang, Qinru Qiu  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.17463v1  

#### Abstract
Spiking Neural Networks (SNNs) are promising for energy-efficient spatiotemporal pattern recognition at the edge, but efficient deployment and on-device adaptation require specialized hardware. In this work, we present an FPGA implementation of multilayer SNNs for spatiotemporal pattern classificati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An FPGA-Based Spiking Neural Network for Spatiotemporal Pattern Learning and Classification

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前边缘设备上的神经网络面临两大挑战：
- **时序模式识别需求**：传感器数据具有时空结构（spatiotemporal），传统静态处理方式效率低、延迟高。
- **资源受限下的在线学习困难**：主流训练算法如 BPTT（Backpropagation Through Time）需要展开时间步，导致内存消耗随序列长度线性增长，难以在边缘部署。

此外，现有 FPGA 上的 SNN 实现大多仅支持推理（inference），缺乏高效的**片上训练能力**（on-chip learning），限制了模型对动态环境的适应性。

### 🔧 提出的新方法与思路
本文提出了一种**全片上运行的多层 Spiking Neural Network (SNN)** 架构，基于 FPGA 实现，支持**同时进行推理与训练**。其核心创新包括：

- **采用 Temporal Forward-Only Training 算法（SOLSA）**  
  避免使用 BPTT，无需展开计算图，梯度在前向传播中累积，显著降低内存开销且允许逐样本更新，适合实时流式输入。

- **混合精度方案（Mixed-Precision Scheme）**  
  - 权重使用 FP16 存储以节省 BRAM 资源；
  - 膜电位（V[t]）、PSP、误差项等关键状态变量使用 FP32 保证训练稳定性。
  平衡了能效与收敛性。

- **硬件级优化策略**
  - **Resource-Aware Loop Unrolling**：通过建模预测不同展开因子下的资源与延迟关系，采用贪心策略分配最优 unroll factor。
  - **Coarse-Grained Pipelining**：将权重更新模块与下一层的反向传播并行执行，提升吞吐。
  - **BRAM 分区与交错访问（Partition & Interleave）**：解决正向（row-major）与反向（column-major）访问权重矩阵引发的端口冲突。
  - **Buffer Insertion 缓解 RAW Hazard**：为浮点累加器插入缓冲区，避免因流水线延迟导致的 Read-After-Write 冲突。

### 🆚 相比现有方法的优势
| 维度 | 本工作 | 现有典型方法 |
|------|--------|-------------|
| **训练位置** | 完全 on-chip，无需外部主机 | 多数需离线训练后部署（如 HaSiST） |
| **学习机制** | 支持多层监督学习（gradient-based） | 多为单层 STDP 或无监督规则 |
| **内存效率** | 不依赖历史缓存，常量内存占用 | BPTT 类需存储所有时间步激活值 |
| **灵活性 vs 效率** | FPGA 可重构性强于 ASIC，优于 CPU/GPU 能效 | Neuromorphic ASIC（如 Loihi）功能固定；CPU/GPU 功耗过高 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
共评估六类任务，涵盖静态图像与时序信号：

| 数据集 | 类型 | 输入维度 | 序列长度 | 类别数 | 场景 |
|-------|------|----------|---------|--------|------|
| MNIST | 静态图像 | 784 | 1 | 10 | 手写数字识别基准 |
| BasicMotions | 时间序列 | 6 | 100 | 4 | 动作传感 |
| Epilepsy | 时间序列 | 3 | 207 | 4 | 癫痫脑电检测 |
| FingerMovements | 时间序列 | 28 | 50 | 2 | 脑机接口 |
| Racket Sports | 时间序列 | 6 | 30 | 4 | 运动识别 |
| Japanese Vowels | 时间序列 | 12 | 29 | 9 | 语音识别 |

> 所有时间序列来自 UEA archive，代表典型的边缘应用场景。

### ⚙️ 实验设置
- **硬件平台**：AMD ZCU102 Evaluation Board（含 ARM Cortex-A53 + FPGA PL）
- **开发工具**：Vivado HLS + Vitis（版本 2024.2）
- **网络结构**：三层全连接 SNN（Dim-100-100-Dout），Leaky-Integrate-and-Fire (LIF) 神经元 + PSP 滤波器
- **训练参数**：
  - PSP 滤波系数 α=β=0.9，reset coefficient λ=0.5
  - 学习率：0.0001
  - 使用 SOLSA 算法进行在线梯度更新

### 📈 评估指标
- **准确率（Accuracy）**：与基线模型比较分类性能
- **吞吐量（Throughput）**：samples/sec（训练全过程：前向 + 反向 + 更新）
- **能量效率**：mJ/sample
- **资源利用率**：LUT、FF、DSP、BRAM 占用情况
- **时钟频率**：MHz

### 🔁 基线方法对比
| 基线 | 描述 | 是否 on-chip learning |
|------|------|---------------------|
| **HaSiST [27]** | 在 FPGA 上实现硬 sigmoid 替代函数的 BP，但训练在主机完成，仅转换后部署 | ❌（伪 on-chip） |
| **TripleBrain [23]** | 基于 STDP 的单层事件驱动架构，轻量高效 | ✅（但非监督/强化学习） |
| **[7]** | BPTT 离线训练 SNN，FPGA 推理加速 | ❌ |
| **[16]** | SOLSA 在 CPU/GPU 上训练与评估 | ✅（软件实现） |
| **LSTM** | 传统 RNN 基线，用于时间序列性能参考 | — |

---

## 3. 主要实验结果和性能指标

### 📈 准确率表现

#### ✅ MNIST 图像分类
| 方法 | 准确率 | 是否 on-chip training |
|------|--------|----------------------|
| TripleBrain | 95.10% | ✅ |
| HaSiST | 97.5% | ❌（训练在主机） |
| **Ours** | **95.92%** | ✅ |

> 本工作在完全片上训练前提下超越 TripleBrain，并接近 HaSiST（后者依赖外部训练）。

#### ✅ 多变量时间序列分类（平均准确率）
| 方法 | Racket Sports | Japanese Vowels | BasicMotions | Epilepsy | FingerMovements |
|------|---------------|------------------|--------------|----------|------------------|
| [7] (BPTT SNN) | 0.869 | 0.969 | 1.000 | 0.961 | 0.620 |
| [16] (SOLSA CPU/GPU) | 0.864 | 0.973 | 1.000 | 0.965 | 0.612 |
| LSTM | 0.877 | 0.935 | 0.925 | 0.839 | 0.625 |
| **Ours (FPGA)** | **0.842** | **0.959** | **1.000** | **0.956** | **0.610** |

> 尽管略低于软件训练的 SNN，但整体表现与 BPTT/SOLSA 基准相当，甚至优于 LSTM，在多个任务上达到或接近最优水平。

### ⚡ 效率与性能对比（时间序列任务）

| 平台 | 吞吐量 (samples/sec) | 时钟频率 (MHz) | 能量/样本 (mJ) |
|------|------------------------|----------------|----------------|
| FPGA (ZCU102) | ~50,000 | 125 | **0.5** |
| GPU (RTX 2080 Ti) | ~20,000 | 1,350 | 5 |
| CPU (Core i7-8700) | ~8,000 | 3,600 | 8.125 |

> - 吞吐量是 GPU 的 **2.5×**，CPU 的 **6×**
> - 能耗仅为 GPU 的 **1/10**，CPU 的 **1/16**

✅ **实时性保障**：最高采样率不超过 1k Hz 的传感器场景中，系统提供 **5× 至 50× 的处理余量**，足以支撑实时在线学习。

### 🧪 消融分析与资源利用
- **Loop Unrolling + Pipeline 优化** 显著提升并行度，资源消耗与吞吐量呈近似线性关系（见 Fig. 5），表明设计可扩展性强。
- BRAM 利用率先上升后趋于稳定，说明低吞吐配置下可迁移到更小容量 FPGA。
- Buffer Insertion 成功维持流水线 II=1，避免因 FP 加法延迟导致的性能退化。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **FPGA 可有效支持多层 SNN 的 on-chip learning**：结合 SOLSA 等非 BPTT 算法，可在有限资源下实现高性能时空模式学习。
2. **混合精度 + 硬件协同设计是关键**：FP32/FP16 混合方案在保持训练稳定的同时大幅减少存储压力。
3. **内存访问冲突是主要瓶颈**：正向与反向传播对权重的不同访问模式（row vs col major）必须通过 BRAM 分区解决。
4. **能效优势显著**：相比通用处理器，FPGA 实现获得数量级的能量节约和更高的吞吐密度。

### ⚠️ 局限性
- 当前使用固定 PSP 核（α, β 不训练），可能影响表达能力。
- 混合精度仍保守（FP32 关键变量），进一步量化可能导致训练不稳定。
- 当前架构尚未探索更大规模网络或更深层数。

### 🔮 未来工作方向
- 引入可学习的 PSP 参数并探索更低比特量化（如 INT8/FP8）。
- 扩展至卷积型 SNN 和事件相机（event camera）应用。
- 结合稀疏编码与压缩感知技术进一步降低带宽需求。
- 探索异构调度策略，动态调整 unroll factor 以适应负载变化。

--- 

> **总结一句话**：该论文展示了在 FPGA 上实现高效、灵活、真正意义上的“边端自适应”SNN 学习系统的可行性，在准确率不牺牲的前提下，实现了比 CPU/GPU 高 2.5–6 倍吞吐、低一个数量级能耗的卓越性能，为下一代边缘智能提供了重要范例。

</details>

---

### 14. [The Physics Is the Network: Fast, Training-Free Neural Emulation](https://arxiv.org/abs/2610.20525)

**Authors**: Luka Todorovic, Kaushal Gianchandani, Sai Ravela  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.20525v1  

#### Abstract
Neural emulation need not require learning a substitute for known physics. We investigate PolyNets that couple a continuous-time neural representation of polynomial dynamics with a simple recurrent network for discrete-time integration. When the governing coefficients are known, the resulting tensor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**The Physics Is the Network: Fast, Training-Free Neural Emulation**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统神经网络在物理系统建模中通常通过学习来近似动力学演化，这带来了以下挑战：
- 需要大量训练数据；
- 存在泛化能力差的问题（尤其在训练分布外）；
- 引入额外的学习误差，可能偏离已知的物理规律。

本文提出：**当系统的控制方程（governing equations）已知且具有多项式结构时，无需“学习”其动态，而可以直接将其编码为神经网络架构本身。**

### 🚀 提出的新方法与新思路
提出了 **PolyNet** —— 一种基于物理方程直接构建的、无需训练的神经网络框架，其核心思想是：
- 将多项式形式的微分方程（如 KdV 方程）中的非线性项（如 $ u \cdot u_x $）用 **乘法节点（product nodes）** 显式表示；
- 空间导数使用固定的有限差分算子（$ D_1, D_3 $）实现；
- 时间积分采用显式的 **Runge-Kutta (RK4)** 或其他数值格式，以递归神经电路方式嵌入；
- 整个计算流程被编译为一个静态的 **tensor graph**，可在 GPU/TPU 上高效执行。

> 🔍 关键洞见：**“The Physics Is the Network”**  
> 物理定律本身就是网络结构——不是用神经网络去拟合物理，而是让神经网络成为物理的计算实现。

### ⚖️ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **准确性** | 与经典数值求解器（CPU/GPU-RK4）精度相当，保留了原始数值方案的离散误差特性，避免学习引入的偏差。 |
| **无需训练** | 完全免训练（training-free），节省训练时间和资源；适用于系数已知的系统。 |
| **泛化性强** | 在初始条件（如波高 $ h $）超出训练范围时仍保持稳定性能，而训练型模型显著退化。 |
| **硬件效率高** | 利用 JAX 编译优化，在 GPU 上实现接近恒定的吞吐量（throughput），对分辨率变化不敏感。 |
| **并行扩展性好** | 对大规模 ensemble 模拟具有良好可扩展性，适合气候、天气等多轨迹预测场景。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集 / 测试系统
- **1D Korteweg-de Vries (KdV) 方程**：
  $$
  u_t + 6uu_x + u_{xxx} = 0
  $$
- 具有解析孤子（soliton）解：
  $$
  u(x,t) = h \cdot \text{sech}^2\left(\sqrt{h}/2 \cdot (x - 2ht - x_0)\right)
  $$
- 初始波高 $ h \in \{2, 2.5, 3, 3.5, 4\} $，其中 $ h=3 $ 用于训练所有学习型模型。

> 所有实验均在周期边界条件下进行，空间域为 $[-20, 20)$，长度 $ L=40 $。

### ⚙️ 实验设置
| 参数 | 设置 |
|------|------|
| 空间分辨率 $ N $ | $ 100 \times 2^n $，$ n=0,\dots,8 $，即从 100 到 25600 网格点 |
| 时间步长 $ \Delta t $ | 按照色散关系缩放：$ \Delta t \sim (\Delta x)^3 $，具体为 $ \Delta t = 0.5 \times \Delta t_{\text{base}} \propto (\Delta x)^3 / 6 $ |
| 积分步数 | 固定 $ 10^6 $ 步（按分辨率调整物理时间跨度） |
| 并行模拟数量（ensemble size） | 最高达 $ 10^4 $ 条独立轨迹 |
| 编译平台 | JAX（用于 PolyNet 和 GPU-RK4），PyTorch（训练阶段），最终统一在 JAX 中评估 |

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Throughput (steps/ms)** | 单位时间内完成的时间步数，衡量计算速度 |
| **Setup/Training Time** | 构建或训练所需时间 |
| **Max Amplitude Error** | 数值解与解析解峰值高度的最大绝对误差 |
| **Max Phase Error** | 波峰位置的最大偏移（考虑周期性边界） |
| **Generalization Performance** | 在未见过的 $ h \neq 3 $ 条件下的误差表现 |
| **Ensemble Scaling** | 随着轨迹数量增加，批量处理的速度变化趋势 |

### 🔁 基线方法对比
共比较六种方法：

| 方法 | 类型 | 是否训练 | 时间积分 | 备注 |
|------|------|--------|----------|------|
| **CPU_RK4_FD** | 经典数值方法 | 否 | RK4 + 有限差分 | CPU 实现，作为基准参考 |
| **GPU_RK4_FD** | 经典数值方法 | 否 | RK4 + 有限差分 | GPU 加速版本 |
| **PolyNet** | Equation-derived NN | 否 | RK4 内嵌 | 本文提出的方法 |
| **Emulator** | Learned state-map | 是 | 直接预测下一状态 | CNN 结构，监督学习 |
| **PINN** | Physics-informed NN | 是 | 直接预测下一状态 | 包含物理约束损失 |
| **Neural PDE (NPDE)** | Learned tendency + integrator | 是 | RK4 积分学习到的趋势 | 类似 Neural ODE 架构 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### （1）**速度与吞吐量（Speed & Throughput）**
- **图 2（训练/构建时间）**：
  - PolyNet、CPU/GPU-RK4_FD 几乎无训练开销，仅需编译；
  - PINN 和 NPDE 训练成本极高，尤其在高分辨率下；
  - Emulator 训练较快，但仍远高于免训练方法。

- **图 3（稳态吞吐量 vs 网格点数）**：
  - **PolyNet 与 GPU-RK4_FD 吞吐量几乎持平且随分辨率变化平缓**；
  - Emulator 在低分辨率下最快，但在高分辨率下性能下降明显；
  - PINN 和 NPDE 吞吐量最低。

> ✅ **结论：PolyNet 实现了与专用 GPU 数值代码相当的运行效率。**

#### （2）**精度表现（Accuracy）**
- **图 4（最大相位与振幅误差 vs 分辨率）**：
  - **PolyNet 与 CPU/GPU-RK4_FD 的误差曲线高度一致**，说明其忠实复现了数值算法行为；
  - 学习型模型（Emulator/PINN/NPDE）误差更大且波动剧烈，尤其在粗网格上；
  - 振幅误差排序不一致，表明学习过程不稳定。

> ✅ **结论：PolyNet 在精度上等效于传统 RK4 求解器。**

#### （3）**并行扩展性（Ensemble Scaling）**
- **图 5（N=400，ensemble size 从 1 到 $10^4$）**：
  - **PolyNet 与 GPU-RK4_FD 在大批量下领先**，吞吐量下降缓慢；
  - 其他学习模型扩展性较差，尤其 PINN 和 NPDE；
  - 表明结构化 tensor graph 更利于 GPU 并行优化。

> ✅ **结论：PolyNet 支持高效的 ensemble 模拟，适用于概率预报任务。**

#### （4）**泛化能力测试（Generalization）**
- **图 6（不同初始波高 $ h $ 下的表现）**：
  - **PolyNet 与传统求解器随 $ h $ 变化的误差趋势一致**，符合理论预期 $ O(h^2 \Delta x) $；
  - 所有训练型模型在 $ h=3 $ 附近尚可，但一旦偏离即出现严重相位漂移；
  - 尤其 PINN 和 Emulator 在 $ h=4 $ 时相位误差已达饱和上限（20单位）；

> ✅ **结论：PolyNet 具备天然泛化能力，不受训练数据限制。**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **神经网络可以作为已知物理方程的精确计算载体，而非近似工具**：
   - 当动力学为多项式形式时，可通过 **multiplicative node + fixed weights** 精确表达；
   - 不需要训练即可实现高保真模拟。

2. **PolyNet 实现了“零学习成本 + 高精度 + 高效率”的统一**：
   - 无需训练，部署快；
   - 精度与经典 RK4 方法相当；
   - GPU 吞吐量媲美手工优化的 GPU 数值代码；
   - ensemble 扩展性优异。

3. **训练型模型存在严重的泛化瓶颈**：
   - 即使在同一解析解族内改变参数（如波高），也会导致显著误差积累；
   - 尤其是相位误差，会随时间线性增长，破坏长期预测可靠性。

4. **误差来源分析清晰可控**：
   - PolyNet 的误差完全来自数值离散（finite-difference truncation + floating-point），而非学习偏差；
   - 可通过标准数值分析（如 Appendix B）解释误差随 $ h $ 和 $ \Delta x $ 的变化趋势。

---

### ⚠️ 局限性
| 局限 | 说明 |
|------|------|
| **仅适用于多项式系统** | 当 governing equations 包含非多项式项（如指数、对数、分数阶导数）时，难以直接构造 PolyNet |
| **当前验证局限于 1D PDE** | 尚未在更高维或多场耦合系统（如 Navier-Stokes、shallow water equations）中全面测试 |
| **依赖精确的离散算子** | 若空间导数无法准确构造（如不规则网格），则优势减弱 |
| **不能替代数据驱动建模** | 对未知或部分未知的动力学（如 subgrid closure），仍需依赖 PINN 或 Neural Operator |

---

### 🔮 未来工作方向
1. **拓展至更复杂的 PDE 系统**：
   - 如 2D turbulence、shallow water equations、Navier-Stokes；
   - 探索如何将非多项式项近似为可微分模块并集成进 PolyNet。

2. **结合学习机制处理不确定性**：
   - 对未知参数或闭包项（closure terms）使用小规模 trainable layer；
   - 构建 hybrid 架构：主干为物理编码，局部可调参。

3. **探索专用硬件加速潜力**：
   - 在 TPU 或神经形态芯片上运行 tensor graph，进一步提升能效比。

4. **应用于地球系统建模**：
   - 如气象 ensemble forecasting、海洋环流模拟等大规模确定性+随机性混合场景。

---

> 💬 **一句话总结**：  
> **PolyNet 展示了一条新路径——将物理定律直接“编译”为神经网络，实现免训练、高保真、高性能的科学计算，标志着“神经仿真”从“学习物理”走向“执行物理”的范式转变。**

</details>

---

### 15. [SpecFit: Not All Draft Tokens Cost the Same in Offloaded MoE Inference](https://arxiv.org/abs/2610.15417)

**Authors**: Hongliang Cao, Xiuyu Li, Siqi Wang, Ziming Wang, Jingmin Zhu, Hailong Yang  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.15417v1  

#### Abstract
Mixture-of-experts (MoE) models activate only a few experts per token, but when their expert weights are offloaded to host memory, host-to-device (H2D) expert transfers enter the critical path of decoding. Speculative decoding verifies several draft tokens in one forward pass, yet in an MoE model ea...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SpecFit: Not All Draft Tokens Cost the Same in Offloaded MoE Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在 **Mixture-of-Experts (MoE)** 模型中，尽管每个 token 只激活少数专家（sparse activation），但在推理时若专家权重被卸载到主机内存（host memory），则每次验证 draft token 都需要通过 **Host-to-Device (H2D)** 传输加载对应的专家。  
传统的 speculative decoding 方法（如 EAGLE-3）倾向于选择能接受最长路径的 draft 子树，**忽略了不同 draft token 激活的专家集合不同，导致 H2D 成本差异巨大**。这使得即使接受了更多 token，也可能因传输开销过大而变慢。

> **核心冲突**：更大的 draft tree 虽然可能提交更多 token，但也激活了更多的专家，增加了 H2D 开销，反而降低吞吐量。

---

### **提出的新方法：SpecFit**
SpecFit 提出了一种新的 **子树选择策略**，其核心思想是：

> **不以“预期接受长度”为目标，而是最大化“预测吞吐量”——即“每周期提交 token 数 / 预测周期时间”的比率。**

#### **关键创新点**：
- **Reward 设计**：使用 draft path probability 加权求和，估计可提交的 token 数 $ R(S) = 1 + \sum_{v \in S} p_v $。
- **Cost 建模**：周期时间 $ T(S) $ 不仅考虑节点数量，还建模为：
  $$
  T(S) = b|S| + B \cdot F(S)
  $$
  其中 $ F(S) $ 是该子树所有节点在各层激活的 **专家并集大小（expert union）**，$ B $ 是拟合的单位专家传输成本。
- **子模性利用**：专家并集是一个 **单调子模函数（monotone submodular coverage function）**，意味着新增节点只为其引入的新专家付费。
- **精确搜索**：在合法的祖先闭合子树（ancestor-closed subtrees）空间中进行深度优先枚举，找到使 $ R(S)/T(S) $ 最大的子树。
- **自适应 drafting 深度**：SpecFit 在每一层 draft 后都运行一次选择；如果最新一层没有被选中的节点，则停止 drafting。

#### **无需修改模型组件**：
- 不改变 drafter（仍用 EAGLE-3）
- 不改变 target 的路由机制
- 不跳过任何应执行的专家
- 仅改变“验证哪个子树”和“draft 多深”

---

### **相比现有方法的优势**
| 方面 | 传统方法（如 EAGLE-3） | SpecFit |
|------|------------------------|--------|
| 选择标准 | 最大化预期接受长度（acceptance length） | 最大化吞吐率 $ R/T $ |
| 成本模型 | 忽略专家差异，视为固定成本 | 显式建模 H2D 传输成本 |
| 决策依据 | 只看 draft 概率 | 综合概率 + 专家重叠情况 |
| 实现复杂度 | 简单排序 | 枚举搜索，但规模小可接受 |

> ✅ **优势本质**：SpecFit 主动剪枝那些“高成本低收益”的 draft 分支，从而缩短验证周期时间，提升整体吞吐。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **CNN/DailyMail**：40 篇新闻摘要任务
- **GSM8K**：80 道数学应用题子集（来自 Spec-Bench）

### **模型配置**
- **Target 模型**：`Qwen3-30B-A3B-Instruct-2507`（48 层 MoE，每层 128 专家，top-8 路由）
- **Drafter 模型**：RedHatAI 提供的 EAGLE-3 speculator
- **硬件平台**：
  - 主要实验：单张 A100 PCIe 40GB GPU
  - 对比实验：RTX 4090 24GB（更小显存、更慢 PCIe）

### **评估指标**
| 指标 | 含义 |
|------|------|
| **Decode TPS** | 解码阶段每秒生成 token 数（不含 prefill） |
| **Speedup** | 相对于 AR（autoregressive）无缓存 baseline 的加速比 |
| **Draft nodes / cycle** | 每轮验证的 draft 节点数 |
| **Committed tokens / cycle** | 每轮实际提交的 token 数 |
| **ms/cycle** | 平均每轮耗时 |
| **H2D/token** | 每输出一个 token 所需的专家传输次数 |

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **AR** | 自回归解码，无 drafter |
| **EAGLE-3** | 固定选择 top-7 最大概率路径节点进行验证 |
| **EAGLE-3 + Expert Cache** | 使用 LRU 缓存（每层 48 个 slot） |
| **SpecFit (ours)** | 提出的方法，带 adaptive depth 和 expert-cost-aware selection |

> 所有方法均基于相同的 offload 推理后端，专家不在 GPU 上常驻。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### **在 A100 上的结果**

| 方法 | 数据集 | Decode TPS | Speedup vs AR (no cache) | 相对 EAGLE-3 (+cache) 提升 |
|------|--------|------------|----------------------------|------------------------------|
| AR (no cache) | CNN/DM | 4.21 | 1.00× | — |
| EAGLE-3 (no cache) | CNN/DM | 6.54 | 1.55× | — |
| **SpecFit (no cache)** | CNN/DM | **8.60** | **2.04×** | **+31.6%** |
| EAGLE-3 (+cache) | CNN/DM | 6.80 | 1.61× | — |
| **SpecFit (+cache)** | CNN/DM | **8.27** | **1.97×** | **+21.6%** |

| 方法 | 数据集 | Decode TPS | Speedup vs AR (no cache) | 相对 EAGLE-3 (+cache) 提升 |
|------|--------|------------|----------------------------|------------------------------|
| AR (no cache) | GSM8K | 4.67 | 1.00× | — |
| EAGLE-3 (no cache) | GSM8K | 6.26 | 1.34× | — |
| **SpecFit (no cache)** | GSM8K | **7.08** | **1.52×** | **+13.1%** |
| EAGLE-3 (+cache) | GSM8K | 6.64 | 1.42× | — |
| **SpecFit (+cache)** | GSM8K | **7.08** | **1.52×** | **+7.0%** |

> 🔥 **最高达 1.97× AR 加速，相对 EAGLE-3 提升 21.6%（summarization）、7.0%（GSM8K）**

---

### **详细分析与消融**

#### **(1) 性能提升来源分析（CNN/DM）**
| 指标 | EAGLE-3 (+cache) | SpecFit (+cache) | 变化 |
|------|------------------|------------------|------|
| Draft nodes / cycle | 7.00 | 2.88 | ↓ 58.9% |
| Committed tokens / cycle | 2.54 | 2.21 | ↓ 13.1% |
| ms / cycle | 371.7 | 265.8 | ↓ 28.5% |
| H2D/token | 73.4 | 57.2 | ↓ 22.1% |

> 尽管提交 token 减少，但由于 **cycle 时间下降更快**，最终吞吐更高。

#### **(2) GSM8K 上的表现**
虽然 EAGLE-3 在 GSM8K 上已能接受较长路径（4.35 tokens/cycle），但 SpecFit 依然通过减少无效传输获得增益：

| 指标 | EAGLE-3 (+cache) | SpecFit (+cache) |
|------|------------------|------------------|
| Draft nodes / cycle | 7.00 | 5.09 |
| Committed / cycle | 4.35 | 4.01 (-7.8%) |
| ms / cycle | 429.8 | 370.4 (-13.8%) |
| H2D/token | 92.7 | 82.0 (-11.6%) |

> 表明：**即使 draft 路径长，仍有优化空间**。

#### **(3) 缓存容量敏感性实验**
当每层专家 slot 从 48 → 24 → 12 时，SpecFit 的优势持续扩大：

| Slot per layer | 48 | 24 | 12 |
|----------------|----|----|----|
| SpecFit 相对 EAGLE-3 提升 | +21.6% | +24.3% | **+29.2%** |

> 原因：缓存越小，H2D 成本占比越高，SpecFit 的 cost-aware selection 越有价值。

#### **(4) 不同 GPU 上的表现（RTX 4090）**
| 方法 | Decode TPS (+cache) | Speedup vs AR |
|------|---------------------|---------------|
| EAGLE-3 | 6.05 | 2.27× |
| **SpecFit** | **6.70** | **2.51×** |
| ➜ 提升 | **+10.8%** | — |

> 即使在更低带宽设备上，SpecFit 仍有效，说明其泛化性强。

#### **(5) 选择开销极低**
- SpecFit 自身选择耗时：
  - GSM8K：中位数 **2.82 ms / cycle**
  - Summarization：**1.43 ms / cycle**
- 远小于节省的时间（如 GSM8K 上每 cycle 节省 ~59 ms）

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **并非所有 draft token 成本相同**：在 offloaded MoE 中，每个 token 激活的专家不同，H2D 成本差异显著。
2. ✅ **传统 speculative decoding 忽视了这一点**，盲目追求长接受路径可能导致吞吐下降。
3. ✅ **SpecFit 通过建模“专家并集”成本，并优化 $ R/T $ 比率，实现了更高吞吐**。
4. ✅ **性能提升来源于更短的验证周期，而非更长的接受路径**。
5. ✅ **在多种 workload、硬件、缓存配置下均稳定有效**，尤其在资源受限场景下优势更大。

---

### **方法的局限性**
- **依赖 route prediction**：当前使用 offline atlas（预计算路由表），部署时需轻量级 route predictor。
  - 实验显示即使预测准确率仅 60%，仍能保持大部分性能。
- **搜索空间虽小但仍需枚举**：目前依赖 $ K \leq 7 $ 的限制实现高效 exact search。
- **未改变 drafter 结构**：仍有进一步联合优化空间（如训练更“易验证”的 draft tree）。

---

### **未来工作方向**
1. **轻量化在线 route prediction**：设计低开销、高精度的 expert routing 预测头。
2. **扩展到更大的 draft tree**：探索近似算法（如 beam search 或 MCTS）处理更大候选池。
3. **动态调整 cost coefficient $ B $**：根据实时系统负载自适应调整传输成本权重。
4. **结合 prefetching**：在 SpecFit 选定子树后，提前加载其专家以进一步隐藏延迟。
5. **应用于其他 sparse 架构**：如 block-sparse、conditional computation 等。

---

> 📌 **一句话总结**：  
> **SpecFit 揭示了 offloaded MoE 中 speculative decoding 的隐性成本，并提出一种简单却高效的 cost-aware 子树选择机制，在不改动模型的前提下显著提升了推理吞吐。**

</details>

---

### 16. [FOLD: Position-Free Far Memory with Local Order for Long-Context Speculative Decoding](https://arxiv.org/abs/2610.20392)

**Authors**: Shuxuan Li, Jialiang Kang, Yanghao Zhou, Ziye Zhu, Xiao Yang  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.20392v1  

#### Abstract
Speculative decoding accelerates LLM generation through lightweight drafting and parallel verification. It is particularly promising for long inputs, where KV-cache access costs can be amortized over multiple accepted tokens. Yet most drafters are trained on short sequences and degrade on long input...

---

### 17. [Adaptive Computation with Conformal Certification](https://arxiv.org/abs/2610.19189)

**Authors**: Shehtab Zaman, Aneesh Rangnekar, Nancy Lee, Harini Veeraraghavan  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.19189v1  

#### Abstract
Scientific imaging data involving medical (e.g., 3D radiographic, microscopic images) and astronomical sensors pose different challenges from natural images. One key challenge is that these datasets are often orders of magnitude larger than natural images. Efficient training and inference on such da...

---

### 18. [SchemaWeaver: Accelerating On-Device Mobile Agentic AI via Schema-Guided Parallel LLM Decoding](https://arxiv.org/abs/2610.17436)

**Authors**: Boyuan Yang, Wei Gao  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.17436v1  

#### Abstract
Large language models (LLMs) on mobile devices can act as agents that call external software tools to complete complex tasks, but the major challenge of supporting on-device mobile agentic AI with LLMs is the long latency of LLM tool calling on mobile devices. In particular, the decode phase is the ...

---

### 19. [Towards Efficient Simultaneous Inverse Text Normalization with Pretrained Text-to-Text Language Model and Read-Tag-Write Policy](https://arxiv.org/abs/2610.19492)

**Authors**: Kiet Anh Hoang, Khanh Le, Bao Nguyen, Linh Pham, Dung Vo, Thai Tran, Mai Nguyen, Tri Nguyen, Vu Le  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.19492v1  

#### Abstract
Streaming Inverse Text Normalization (ITN) is vital for converting spoken-form outputs from streaming Automatic Speech Recognition into formatted written text. Existing streaming ITN methods rely on hybrid systems combining neural tagging with expert-crafted finite-state transducer rules, limiting s...

---

### 20. [Qwen-MT-Lumen at WMT2026: Instruction-Specialized Translation via Quality-Aware Training and On-Policy Distillation](https://arxiv.org/abs/2610.19758)

**Authors**: Hao Wang, Yangyang Liu, Xiaohu Zhao, Heng Liu, Zifu Shang, Tianhao Li, Ruize Gao, Jialong Tang, Liangying Shao, Haoran Wei, Baosong Yang, Linlong Xu, Longyue Wang, Weihua Luo  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.19758v1  

#### Abstract
We present Lumen, The open-source model is available at, our submission to the WMT 2026 General Machine Translation Shared Task. We focus on instruction-specialized translation, where systems must produce accurate translations while satisfying fine-grained requirements on style, terminology, and str...

---

### 21. [Measuring the Inference Energy Efficiency of Open-Weight LLMs across Benchmarks, Reasoning Modes, and Hardware](https://arxiv.org/abs/2610.14389)

**Authors**: Zeyu Yang, Wes Armour  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.14389v1  

#### Abstract
Benchmark scores tell us what an LLM can do, but not how much energy is required to deliver that capability. We directly measure this trade-off for 86 open-weight model checkpoints, evaluated on seven benchmarks and six hardware platforms ranging from a consumer GPU to an eight-GPU server. We find l...

---

### 22. [Bridging Risk Approximation Gaps in Model Predictive Task Sampling via In-Context Modeling](https://arxiv.org/abs/2610.14913)

**Authors**: Jiarong Wen, Qi Tao, Kaiyu Zhang, Yun Qu, Lizhou Cai, Heming Zou, Yuhang Jiang, Qi Wang  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.14913v1  

#### Abstract
Reinforcement learning (RL) is essential for adaptive decision-making, e.g., robust Meta-RL and domain randomization (DR), and efficient post-training of large language models (LLMs), where tasks are commonly structured as Markov decision processes or prompts. A central bottleneck in all these setti...

---

### 23. [MemDuet: Learning Dual-Process Memory for Long-Context Reasoning](https://arxiv.org/abs/2610.19928)

**Authors**: Jiangcheng Song, Haodong Jing, Yulin Mu, Lesong Tao, Jixin Wang, Jin Cui, Tianjing Ren, Runxuan Xie, Yongqiang Ma, Nanning Zheng  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.19928v1  

#### Abstract
Reasoning over infinitely long documents is becoming a reality with recurrent memory agents. However, increasing context length exacerbates the trade-off between reasoning performance and inference efficiency. Inspired by Kahneman's account of fast and slow thinking, we propose MemDuet, a dual-proce...

---

### 24. [ReaDy: Recurrent States as Dynamic Context for Speculative Drafting in Hybrid Linear Attention Large Language Models](https://arxiv.org/abs/2610.20503)

**Authors**: Bohan Sun, Zunhai Su, Xialie Zhuang, Jianjin Li, Ngai Wong, Chuan-Wei Kuo  
**Category**: cs.CL  
**Published**: 2026-10-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.20503v1  

#### Abstract
Speculative decoding accelerates autoregressive generation by drafting multiple tokens per step, yet feature-based drafters rely on target hidden features whose contextual emphasis may not align with the needs of subsequent draft positions. Hybrid linear attention (HLA) models maintain compact recur...

---

### 25. [Inverse Reasoning Models for Efficient Post-Training and Verification from Answer-Only Data](https://arxiv.org/abs/2610.18878)

**Authors**: Eric O. Korman  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.18878v1  

#### Abstract
Post-training reasoning LLMs often requires making a tradeoff between costly supervision and compute inefficiency: supervised finetuning typically requires reasoning traces, which is data inefficient since in many applications one only has prompt-answer tuples and lacks the groundtruth chain-of-thou...

---

### 26. [Memory-Efficient Subspace Optimization with Residual Correction: From Momentum SGD to Muon](https://arxiv.org/abs/2610.19844)

**Authors**: Sifan Yang, Hongyi Tao, Dingzhi Yu, Haomin Bai, Wei Jiang, Silin Gao, Pi Bu, Lijun Zhang  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.19844v1  

#### Abstract
Memory-efficient subspace optimization has emerged as a practical approach to reducing the memory cost of LLM training while preserving optimization performance. Previous algorithms typically rely on the leading singular subspace of stochastic gradients, yet the gradient noise may dominate the subsp...

---

### 27. [Optimizing Sparse Boolean Rule Networks using Mixed Integer Linear Programming](https://arxiv.org/abs/2610.20170)

**Authors**: Thomas Pammer, Johannes F\"urnkranz, Markus Sinnl  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.20170v1  

#### Abstract
We study the problem of learning accurate and cognitively simple binary classifiers that are directly interpretable as Boolean rules. We propose mixed integer linear programming (MILP) formulations for training sparse disjunctive normal form (DNF) networks, viewed as small networks with a conjunctiv...

---

### 28. [SpectrumQuant: Spectrum Alignment for LLM Quantization](https://arxiv.org/abs/2610.20303)

**Authors**: Dwaipayan Haldar, Indranil Patra, Azhar Yousuf, Chandra Sekhar Seelamantula  
**Category**: cs.LG  
**Published**: 2026-10-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.20303v1  

#### Abstract
Orthogonal transforms have emerged as an effective tool for low-bit quantization of large language models (LLMs), as they reshape activation distributions without changing the underlying model function. Existing approaches typically rely either on fixed data-agnostic transforms, such as Hadamard rot...

---

### 29. [EvoQuantLab: Training-Free Dual-Loop Evolution for Auditable Quantitative Research](https://arxiv.org/abs/2610.17130)

**Authors**: Haolin Yang, Zhiqiang Qian, Chutian Chen, Jipeng Zhang, Zhenghao Zhu, Tangjia Deng, Jian Xie, Shuaishuai Gong, Yike Guo, Sirui Han  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.17130v2  

#### Abstract
Automated alpha factor discovery is critical for modern quantitative research. While large language models (LLMs) have increasingly been integrated into quantitative workflows, existing agent-based approaches remain highly susceptible to severe backtest overfitting. To address this challenge, we pro...

---

### 30. [Unlocking Token Rewards via Training-Free Reward Attribution](https://arxiv.org/abs/2610.18704)

**Authors**: Sitong Wu, Haoru Tan, Bin Xia, Xichen Zhang, Jingyao Li, Shaofeng Zhang, Xiaojuan Qi, Bei Yu, Jiaya Jia  
**Category**: cs.AI  
**Published**: 2026-10-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.18704v1  

#### Abstract
In this paper, we propose an extremely efficient, training-free method to extract token-level reward signals directly from an existing deep reward model. Our core idea is to attribute the overall process reward to individual tokens by estimating each token's influence. This influence is defined as t...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-28 11:40:22 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [PipeLink: Contract-Guided Linking of GPU Kernel Components](https://arxiv.org/abs/2610.37200)

**Authors**: Jize Guo, Xupeng Miao  
**Category**: cs.DC  
**Published**: 2026-10-28  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2610.37200v1  

#### Abstract
The fastest machine-learning kernels are compositions of components, such as loaders, matrix-multiply mainloops, epilogues and communication stages, that run concurrently on one GPU streaming multiprocessor and share its on-chip memory, barriers, warps and registers. Because these components hard-co...

---

### 2. [INTRA: Interleaved Non-contiguous Token spaRse Attention](https://arxiv.org/abs/2610.34056)

**Authors**: Yuxuan Xia, Wenbo Lu, Shaoyi Zheng, Shengjie Wang  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.34056v2  

#### Abstract
Transformers achieve strong performance across modalities but are bottlenecked by the quadratic cost of attention. Sparse attention can reduce this cost, but effective sparsity must preserve useful long-range communication and map well to GPU hardware. We introduce INTRA--Interleaved Non-contiguous ...

---

### 3. [End-to-End MXFP8 Reinforcement Learning on Blackwell: What Matching Buys, When It Pays Off, and How It Fails](https://arxiv.org/abs/2610.34915)

**Authors**: Wengee Zhang  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.34915v1  

#### Abstract
Low-precision arithmetic can make reinforcement learning (RL) for large language models cheaper, but RL requires the sampler and the learner to agree. Several frameworks now run FP8 on both sides. We implement end-to-end MXFP8 RL in verl (Megatron/Transformer Engine training; vLLM and SGLang rollout...

---

### 4. [INC-MoE: Overlapping In-Network Collectives and Expert Computation for Expert-Parallel Mixture-of-Experts Inference](https://arxiv.org/abs/2610.36339)

**Authors**: John Stephan, Razvan Puscasu, Xu Yong, Ding Jiutao, Guan Zixuan  
**Category**: cs.DC  
**Published**: 2026-10-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.36339v1  

#### Abstract
In expert-parallel mixture-of-experts (MoE) serving, dispatch moves tokens to the accelerators hosting their selected experts, and combine returns expert outputs to each token's owner. Both operations lie on the critical path and can dominate MoE layer latency in multi-node deployments. We present I...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# INC-MoE 论文核心总结

## 1. 主要贡献和创新点

### 解决的问题
在 **Expert-Parallel Mixture-of-Experts (MoE)** 推理中，每个 MoE 层的关键路径上存在两个通信密集型阶段：
- **Dispatch**：将 token 分发到其选中的专家所在的加速器；
- **Combine**：将各专家的输出聚合回 token 的拥有者。

这两个阶段通常通过 **AllGather (AG)** 和 **ReduceScatter (RS)** 集合操作实现，在多节点部署中，由于跨节点带宽远低于节点内带宽，这些通信开销会成为主要瓶颈，导致层延迟受通信限制（communication-bound）。

### 提出的新方法：INC-MoE
论文提出 **INC-MoE**，一种用于 scale-out 场景的专家并行 MoE 层，其核心创新在于：
- **In-Network Collectives**：将 AG 和 RS 操作卸载到支持 **In-Network Computing (INC)** 的网络交换机上执行。
  - **Dispatch → In-Network AllGather**：由交换机侧的 multicast 实现。
  - **Combine → In-Network ReduceScatter**：由交换机侧的 reduction 实现。
- **Overlap Fusion**：在一个持久化内核（persistent kernel）中，将 **AG、专家计算（FFN）、RS** 三个阶段融合，并以 **band 为粒度进行流水线处理**，实现三者的重叠（overlap）。
  - 同一时刻，不同线程块分别处理不同 band 的 AG、FFN 和 RS，从而实现通信-通信（AG/RS）和通信-计算（AG/FFN, RS/FFN）的双重重叠。

### 相比现有方法的优势
- **打破通信瓶颈**：传统方法中，AG 和 RS 序列化执行，且环形（ring）算法会同时占用链路的收发双向，无法通过简单重叠缩短总时间。而 INC-MoE 利用 in-network AG 是接收密集型、in-network RS 是发送密集型的特点，两者并发时可以**同时利用链路的两个方向**，理论上可将通信时间减少 $\frac{N-2}{2(N-1)}$（N=4 时为 33.3%，N=8 时为 42.9%）。
- **端到端性能提升**：通过设备端发起的通信（device-initiated communication），实现了细粒度的、无需主机干预的同步，使得 AG、FFN、RS 的重叠更加高效。
- **即插即用**：作为 vLLM 中 MoE 层的直接替代品，无需修改模型定义。

## 2. 核心实验方法和设置

### 模型和工作负载
- **模型**：OLMoE-1B-7B，包含 16 个 MoE 层，E=64 个专家，top-k=8 路由，隐藏层大小 H=2048。
- **并行策略**：数据并行（DP=4）用于注意力层，专家并行（EP=4）用于 FFN 层，无张量并行。
- **任务**：评估 **prefill** 阶段的性能。

### 实验设置
- **测试平台**：模拟一个四节点（four-node）scale-out 部署。
  - 使用 **5 个高性能加速器**：其中 4 个模拟 EP 节点，第 5 个专用加速器模拟支持 INC 的交换机。
  - 所有通信通过 RoCE 网络进行，每个节点与模拟交换机的有效带宽为 100 Gbps。
- **输入规模**：变化每个加速器上的提示词（prompt tokens）数量，范围从 24K 到 126K。
- **评估指标**：
  - **Time to First Token (TTFT)**：首 token 时间。
  - **Prefill Throughput**：预填充吞吐量（tokens/s）。
  - **等效带宽（Equivalent Bandwidth）**：用于独立评估集合操作性能。

### 基线方法对比
- **Baseline**：vLLM 默认的专家并行路径，使用基于平台默认集合库（如 NCCL）的 **ring AllGather** 和 **ring ReduceScatter**，中间夹着 MoE 计算内核。该基线的集合操作在端点（end hosts）执行，而非交换机。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比
在 OLMoE prefill 任务中，INC-MoE 相比 vLLM 默认路径取得了显著提升：

| 指标 | 提升范围 | 最大提升 |
| :--- | :--- | :--- |
| **TTFT (Time to First Token)** | 减少 20.2% ~ 33.1% | **33.1%** (在 126K tokens/accelerator 时) |
| **Prefill Throughput** | 提高 25.3% ~ 49.5% | **49.5%** (在 126K tokens/accelerator 时) |

- **趋势分析**：随着每个加速器上的 token 数量增加，INC-MoE 的优势愈发明显。这是因为更大的 batch size 使得流水线的填充（fill）和排空（drain）阶段占比减小，重叠效率更高。

### 独立集合操作性能
- 在并发执行 AG 和 RS 的独立基准测试中，INC-MoE 的等效带宽比 ring collectives 高出 **25.6% ~ 31.8%**，验证了其理论优势。

### 消融实验与关键发现
- **输出一致性**：INC-MoE 与基线方法的输出在所有 16 个 MoE 层、所有 4 个节点上均保持完全一致（per-token cosine similarity = 1.0）。
- **尾部辅助（Tail Assistance）**：在流水线末尾，让空闲的 gather 线程块协助 reduce 线程块完成最后几个 band 的本地规约，此优化在大 token 数量（≥64K）时效果显著，有助于进一步缩短尾部通信时间。

## 4. 关键结论和发现

### 主要发现
1. **互补性是关键**：单纯地将集合操作卸载到交换机（offload）或单纯地在端点重叠通信与计算，都无法最大化性能收益。**INC-MoE 的核心价值在于将两者结合**：交换机卸载使得 AG 和 RS 可以高效并发，而内核级的融合流水线则充分利用了这一特性，实现了端到端的性能飞跃。
2. **网络成为机遇而非瓶颈**：在专家并行架构中，通信本是瓶颈，但通过 INC-MoE 的设计，网络被转化为一个可以主动参与计算和优化的资源，从而将瓶颈转变为性能提升的机会。

### 方法的局限性
- **依赖特定硬件**：该方法依赖于支持 **In-Network Computing** 的交换机硬件。虽然论文使用专用加速器进行了模拟，但其最终性能（尤其是接近理论带宽上限）需要真正的硬件支持才能完全发挥。
- **模拟开销**：当前原型中的模拟器引入了额外的固定开销（如网格同步），导致小消息下的效率未达最优，但这被认为是模拟器的局限，而非方法本身的缺陷。

### 未来工作方向
- **硬件协同设计**：推动支持原生 AG 和 RS 操作的商用 INC 交换机的发展。
- **扩展到其他场景**：探索将类似的设计思想应用于其他通信密集型的分布式训练或推理任务，例如 tensor parallelism 或 all-reduce 操作。
- **动态路由优化**：结合更复杂的路由模式，进一步优化 in-network collectives 的效率。

</details>

---

### 5. [TAO: Tool-Aware Joint Scheduling for LLM-Based Agentic Serving](https://arxiv.org/abs/2610.37230)

**Authors**: Shuhuai Li, Shengdong Ye, Dongdong Ge, Yinyu Ye, Jianghao Lin  
**Category**: cs.DC  
**Published**: 2026-10-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.37230v1  

#### Abstract
Agentic services alternate between model inference on clustered GPUs and external tools on distributed CPUs. In this scenario, existing systems (1) overlook tool-return readiness in service evaluation, wasting limited resources, and (2) use heuristic strategies to decide admission and placement on G...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TAO: Tool-Aware Joint Scheduling for LLM-Based Agentic Serving 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 LLM 推理服务系统（如 vLLM、Dynamo、ThunderAgent）在处理基于 LLM 的智能体（agentic）任务时存在两大缺陷：

1. **忽略外部工具调用的返回准备状态（tool-return readiness）**：  
   在 **Clustered-GPUs with Distributed-CPUs (CGDC)** 架构中，agent 会交替执行 LLM 推理（在 GPU 上）和外部工具调用（如编译器、检索器，在 CPU 上）。现有系统在调度时只关注 KV cache 复用、token 数量等静态特征，而忽略了“某个 session 是否即将从工具调用中返回”这一关键动态信号，导致高价值的 ready session 被延迟调度。

2. **采用启发式、逐个决策的调度策略（heuristic, myopic scheduling）**：  
   现有系统通常对每个 session 单独打分并分配资源，缺乏全局视角。这种“evaluate-then-decide”模式无法建模多个 session 之间的资源竞争关系，容易造成次优甚至冲突的调度决策。

---

### 提出的新方法：TAO 框架

TAO 是一个 **工具感知的联合调度框架（Tool-Aware Joint Scheduling Framework）**，其核心创新在于两个层面的改进：

#### （1）**工具感知的会话评估（Tool-Aware Session Evaluation）**
引入 **三元组价值接口（value triplet）** 来综合评估每个 session 在特定节点上的调度优先级：
$$
\pmb{v}_{i,b} = (\underbrace{k_{i,b}}_{\text{Decayed Cache Credit}}, \underbrace{r_i}_{\text{Binary Readiness}}, \underbrace{t_i}_{\text{Token Footprint}})
$$
- **Decayed Cache Credit ($k_{i,b}$)**：KV 缓存信用随中断时间指数衰减（half-life = 10s），鼓励近期活跃的缓存重用。
- **Binary Readiness ($r_i$)**：基于历史工具执行时间、暴露的截止时间或进度事件，预测 session 是否将在短期内（默认 1s 内）返回推理请求。这是 TAO 的核心新增信号。
- **Token Footprint ($t_i$)**：作为后备排序依据，偏好上下文更长的任务。

#### （2）**联合调度决策（Joint Scheduling with Dynamic Programming）**
不再逐个决策，而是将所有待调度 session 视为一个集合，通过 **动态规划（Dynamic Programming）** 求解满足容量约束下的最优组合分配问题：
$$
x^{\star} \in \arg\operatorname{lexmax}_{x \in \mathcal{X}} \sum_{i \in P} \sum_{b \in B} x_{i,b} \pmb{v}_{i,b}
$$
该方法实现了真正的 **多 knapsack 与广义指派问题（multiple-knapsack and generalized-assignment）** 的联合优化，最大化整体价值。

---

### 相比现有方法的优势
| 方面 | 现有方法 | TAO |
|------|--------|-----|
| **评估信号** | 忽略 tool-return readiness，依赖固定规则 | 显式建模 readiness，结合缓存衰减与 token 成本 |
| **决策方式** | 启发式、逐个分配（myopic） | 联合优化、全局最优（joint optimization） |
| **调度目标** | 局部最优或负载均衡 | 最大化加权价值（cache + readiness + footprint） |
| **扩展性** | 固定评分逻辑难扩展 | 解耦评估与决策，支持灵活替换价值函数 |

---

## 2. 核心实验方法和设置

### 使用的数据集与工作负载
- **SWE-Lite**：基于 GitHub issue 修复任务的 agentic 工作负载，使用 MiniSWE 测试框架。
- **Terminal-Bench 2.0**：命令行界面下的复杂自动化任务，更具挑战性。
- **SWE-verified**：更高并发下的 SWE 任务变体，用于压测小模型副本能力。
- **GLM-4.7 on SWE-Lite/SWE-verified**：用于单节点高并发扩展性测试。
- **InferenceX Trace Replay**：固定轨迹回放，用于控制变量分析。

---

### 实验设置
- **模型**：
  - DeepSeek-V4 Flash（2 nodes × 4×H100）
  - Qwen3-32B（4 nodes × 2×H100）
  - GLM-4.7（1 node × 8×H100）
- **部署架构**：CGDC 架构，GPU 集群用于推理，CPU 分布式执行外部工具。
- **并发等级**：从 16 到 160 workers 不等，覆盖低、中、高并发场景。

---

### 评估指标
| 指标 | 含义 |
|------|------|
| **Latency ↓** | 平均端到端任务延迟（E2E latency） |
| **TTFT ↓** | 平均 time to first token（首 token 时间） |
| **Decode Throughput ↑** | 解码吞吐量（tokens per second, TPS） |
| **Cache Hit Rate ↑** | KV 缓存命中率 |
| **Pause Wait ↓** | session 中断期间等待工具返回的时间 |

---

### 基线方法对比
| 基线方法 | 特点 |
|--------|------|
| **Dynamo** | KV-aware 调度器，支持负载感知 |
| **vLLM Scheduler w/ RR** | 轮询策略，负载均衡 |
| **vLLM Scheduler w/ CA** | Cache-Aware 策略，优先重用缓存 |
| **ThunderAgent** | 引入生命周期管理与 token 需求衰减，SOTA 基线 |

---

## 3. 主要实验结果和性能指标

### 关键性能提升（来自 Table 1 & 2）
在 **12 个不同的 model–workload–concurrency 设置下**，TAO 相对于各指标的最佳基线实现了显著提升：

| 指标 | 最大提升幅度 | 典型场景 |
|------|------------|---------|
| **Latency ↓** | **26.8%** | DeepSeek-V4 Flash @ 64 workers |
| **TTFT ↓** | **36.3%** | DeepSeek-V4 Flash @ 64 workers |
| **Decode Throughput ↑** | **29.5%** | DeepSeek-V4 Flash @ 64 workers |

> ✅ TAO 是唯一在所有三项指标上都优于任何单一基线的方法。

---

### 消融实验结果（Table 4）
消融实验验证了 TAO 各组件的有效性（以 DeepSeek-V4 Flash @ 32 workers 为例）：

| 配置 | Decode (tok/s) | Cache Hit (%) | TTFT (mean) | Pause Wait (mean) |
|------|----------------|----------------|-------------|--------------------|
| **TAO (Ours)** | **1417.97** | **96.31** | **0.154** | **0.972** |
| w/o Cache Value | 1398.35 | 96.24 | 0.157 | 1.028 |
| w/o Readiness | 1363.06 | 96.16 | 0.161 | 1.067 |
| w/o Cache & Readiness | 1373.51 | 96.24 | 0.182 | 1.025 |
| w/o Session Evaluation | 1355.20 | 96.20 | 0.162 | 1.025 |
| w/o Decision Making (BFD) | 1233.72 | 96.05 | 0.195 | 1.081 |

> 🔍 发现：
> - 移除 **Readiness** 导致 decode throughput 下降 **4.03%**，pause wait 增加 9.8%。
> - 移除 **Joint DP**（改用 Best-Fit Decreasing）导致 throughput 暴跌 **13.0%**，TTFT 上升 26.6%。
> - 两者共同贡献了绝大部分性能增益。

---

### 其他关键实验结果
- **Figure 4 & 7–8**：TAO 完成任务所需时间远少于基线（如 GLM-4.7 @ 160 workers：TAO 60.0 min vs. Dynamo 189.4 min）。
- **Figure 6**：启用 readiness 后，**82.9% 的工具预测误差 < 0.5s**，且 **tool-end-to-first-token 延迟降低 10.88%**。
- **Table 8**：即使在工具延迟延长至 20s 的极端情况下，TAO 仍能保持 **19.28% 的 makespan 优势**。

---

## 4. 关键结论和发现

### 主要发现
1. **Tool-return readiness 是一个强信号**：  
   外部工具的返回时机是决定何时重新调度 session 的关键因素，不应被忽略。

2. **联合优化显著优于启发式调度**：  
   在资源受限下，必须考虑 session 间的竞争关系，DP 联合求解能有效避免“局部最优陷阱”。

3. **缓存价值具有时效性**：  
   KV 缓存的重用价值随中断时间快速衰减（10s 半衰期），因此调度应优先考虑近期活跃的 session。

4. **TAO 具备良好的通用性和兼容性**：
   - 支持混合 workload（Table 3：Terminal-Bench + SWE-Lite）
   - 可与 TokenCake 等 cache 管理引擎集成（+12.9% throughput）
   - 在 trace replay 和 live run 中均表现稳定

---

### 方法的局限性
1. **预测误差影响 readiness 准确性**：  
   当工具执行时间高度不确定时，readiness 判断可能出现误判。
2. **DP 求解成本随规模增长**：  
   虽然实测调度开销 < 0.12%（Table 9），但在超大规模集群中可能成为瓶颈。
3. **假设工具执行不可抢占**：  
   当前模型假设工具一旦启动就必须完成，未考虑可中断或部分执行机制。

---

### 未来工作方向
1. **引入更精细的 readiness 预测模型**：  
   如使用 ML 模型预测工具剩余时间，替代当前的历史均值 + margin 方法。
2. **支持 speculative execution 与 partial tool execution**：  
   与 Conveyor、PASTE 等系统结合，进一步压缩工具延迟。
3. **扩展至异构硬件与多租户场景**：  
   支持 GPU/CPU/NPU 协同调度，并保障 SLO。
4. **探索强化学习或在线优化算法**：  
   在动态变化环境中实现自适应调度策略。

--- 

> ✅ **总结**：TAO 通过 **显式建模 tool-return readiness** 并结合 **联合动态规划调度**，在 LLM-based agentic serving 场景中实现了显著的性能突破，为下一代智能体服务系统提供了新的设计范式。

</details>

---

### 6. [AutoPRL: Automated Path Representation Learning via a Learnable View Generator](https://arxiv.org/abs/2610.35032)

**Authors**: Sean Bin Yang, Ying Sun, Yunyao CHeng, Hao Miao, Jilin Hu, Bin Yang, Torben Bach Pedersen, Christian S. Jensen  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.35032v1  

#### Abstract
Self-supervised learning has emerged as a powerful paradigm for Path Representation Learning (PRL) in road networks. PRL aims to learn informative representations of paths traversed by moving objects, such as vehicles, where each path consists of a sequence of connected road segments. These represen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AutoPRL 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Path Representation Learning (PRL)** 方法通常依赖于启发式的数据增强（如随机节点丢弃、掩码等），这类方法容易导致严重的**语义信息丢失**，并且生成的视图缺乏多样性。同时，传统的 **contrastive learning** 框架在训练过程中需要大量的负样本，导致计算开销大、训练效率低，难以扩展到大规模场景。

### 提出的新方法与核心思路
本文提出了 **AutoPRL**，一种高效且有效的自监督 PRL 框架，其核心在于引入了一个**可学习的视图生成器**（learnable view generator），并通过三大模块实现自动化、高质量的路径表示学习：

1. **AutoDA (Automatic Data Augmentation)**  
   - 使用 **Variational Node Autoencoder (VNAE)** 建模每个拓扑感知的节点嵌入（topology-aware node embedding）为高斯潜在分布。
   - 通过从该分布中采样，生成多样且语义一致的**特征级路径视图**（feature-level path views），而非直接修改原始路径结构。
   - 优点：避免了离散扰动带来的信息丢失，增强了表示质量。

2. **CVDC (Cross-View Decoupled Contrastive Learning)**  
   - 引入了解耦的对比学习机制，将正负样本对的优化分离。
   - 分别定义了 **Cross-Path DCL**（路径级）和 **Cross-Node DCL**（节点级）损失，使模型能同时捕捉全局语义和局部结构一致性。
   - 优点：提升了训练稳定性与效率，减少了正负样本间的竞争干扰。

3. **ACT (Adaptive Co-Training)**  
   - 动态调整多个目标函数的损失权重，基于不确定性（uncertainty-based）进行自适应加权。
   - 无需手动调参即可平衡 AutoDA 与 CVDC 模块的学习目标。
   - 优点：提高训练稳定性，增强模型泛化能力。

### 相比现有方法的优势
| 维度 | 传统方法 | AutoPRL |
|------|--------|-------|
| 视图生成 | 启发式、静态（如 Random Drop） | 可学习、动态、语义保持 |
| 增强粒度 | 结构级（discrete node sequence） | 特征级（continuous embedding sequence） |
| 对比学习 | SimCLR-style，依赖大批量负样本 | Decoupled Contrastive，更高效稳定 |
| 多目标优化 | 固定权重（fixed λ） | 自适应权重（uncertainty-aware） |

---

## 2. 核心实验方法和设置

### 使用的数据集
在三个真实世界轨迹数据集上进行了广泛实验：
- **Aalborg**（丹麦）：10,017 节点，11,597 边，33,264 条路径
- **Chengdu**（中国成都）：6,632 节点，17,038 边，54,491 条路径
- **Harbin**（中国哈尔滨）：8,497 节点，14,497 边，259,615 条路径  
数据来源于 OpenStreetMap 并经过 map-matching 处理。

### 实验设置
- **Embedding 维度**：统一设为 128
- **Batch Size**：64
- **Optimizer**：AdamW + Cosine Decay 学习率调度
- **训练方式**：先无监督预训练路径编码器，再用于下游任务微调

### 评估指标（按任务分类）
| 下游任务 | 评估指标 |
|---------|--------|
| Travel Time Estimation | MAE ↓, MAPE ↓, RMSE ↓ |
| Path Ranking | MAE ↓, TAU ↑, RHO ↑ |
| Most Similar Path Search | Mean Rank ↓ |
| Path Recommendation | Accuracy ↑, Precision ↑, F1 ↑ |
| Path Imputation | Recall@3 ↑, Recall@5 ↑, MAP ↑ |

### 基线方法对比
共对比了 **13 种基线方法**，分为两类：

#### 无监督方法（Unsupervised）
- Node2vec, VGAE-AMI, DVGA
- Toast, PIM, START, LightPath, TrajRL, RED

#### 有监督方法（Supervised）
- PathRank, HierETA, CompactETA, HMTRL

> 所有无监督方法均在大规模无标签数据上训练，而有监督方法仅使用少量标注数据（如 12K），体现其在标签稀缺场景下的优势。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（摘自 Tables I-V）

#### ✅ Travel Time Estimation（MAE）
| Method | Aalborg | Chengdu | Harbin |
|--------|--------|--------|--------|
| RED (best baseline) | 2.457 | 4.845 | 4.097 |
| **AutoPRL (Ours)** | **1.971** | **4.672** | **3.883** |
| **Improvement (%)** | **19.8%** | **3.6%** | **5.2%** |

#### ✅ Path Ranking（TAU）
| Method | Aalborg | Chengdu | Harbin |
|--------|--------|--------|--------|
| TrajRL (best baseline) | 0.896 | 0.839 | 0.929 |
| **AutoPRL (Ours)** | **0.939** | **0.845** | **0.958** |
| **Improvement (%)** | **4.8%** | **0.7%** | **3.1%** |

#### ✅ Most Similar Path Search（Mean Rank）
| Dataset | 10K DB | 30K DB | 50K DB | 100K DB |
|--------|--------|--------|--------|--------|
| RED | 1.453 | 1.578 | 1.690 | 3.457 |
| TrajRL | 1.326 | 1.467 | 1.582 | 3.757 |
| **AutoPRL** | **1.217** | **1.307** | **1.389** | **3.212** |

> AutoPRL 在所有数据库规模下均取得最低 Mean Rank，表明其路径表示具有更强的语义判别力。

#### ✅ Path Recommendation & Imputation
- **Recommendation**：AutoPRL 在 Acc./Pre./F1 上全面领先（如 Chengdu 上 F1 达 0.881）
- **Imputation**：在 Recall@5 和 MAP 上显著优于基线（如 Harbin 上 Recall@5 = 0.924）

---

### 消融实验结果（Ablation Study）

#### (1) 是否使用 AutoDA？
- 移除 AutoDA（改用 START 的随机丢弃）后：
  - Aalborg 上 MAE 从 1.971 升至 2.865（↑45%）
  - 证明 AutoDA 对表示质量至关重要。

#### (2) 是否使用 Progress-Based Prior？
- 使用固定先验（w/o PPD）后：
  - Chengdu 上 TAU 从 0.836 降至 0.798
  - 说明动态调整先验有助于匹配训练进程，提升泛化性。

#### (3) 是否使用 CVDC？
- 仅用 CPDCL（w/o CNDCL）：Harbin 上 MAE ↑ 至 4.326
- 仅用 CNDCL（w/o CPDCL）：MAE ↑ 至 4.460
- 二者结合效果最佳，说明**全局+局部对比信号互补**。

#### (4) 是否使用 ACT？
- 使用固定损失权重（w/o ACT）：
  - 收敛速度慢，最终损失更高（见 Fig. 11）
  - 证明 ACT 显著提升训练效率与稳定性。

---

## 4. 关键结论和发现

### 主要发现
1. **可学习的视图生成优于启发式增强**  
   AutoDA 通过 VNAE 在潜空间生成语义一致的特征级视图，有效缓解了传统结构扰动的信息丢失问题。

2. **解耦对比学习提升训练效率**  
   CVDC 机制避免了正负样本之间的梯度竞争，使得模型即使在小批量下也能稳定训练。

3. **自适应多目标优化是关键**  
   ACT 模块通过不确定性建模自动调节损失权重，消除了繁琐的手动调参过程，提升了训练鲁棒性。

4. **AutoPRL 具备强可扩展性与迁移能力**  
   - 在不同城市、不同任务上均表现最优；
   - 可作为预训练模型显著提升有监督方法性能（如 PathRank）；
   - 随着训练数据量增加，性能持续提升（Fig. 10）。

### 局限性
- **依赖高质量拓扑嵌入**：AutoDA 的性能受限于初始 topology-aware node embeddings（如 Node2vec）的质量。
- **计算开销略高**：相比 LightPath 等轻量方法，AutoPRL 因包含 VNAE 和 Transformer，训练时间稍长（见 Fig. 11a）。
- **未考虑时空动态性**：当前框架聚焦空间路径结构，未显式建模时间维度（如拥堵模式）。

### 未来工作方向
- 将 AutoPRL 扩展至 **spatio-temporal trajectory representation learning**，融合时间上下文。
- 探索更高效的生成器架构（如 Diffusion-based），替代 VAE。
- 将 AutoPRL 集成到端到端导航系统中，支持实时路径推荐与重规划。
- 研究跨城市/跨路网的迁移学习能力，构建通用路径基础模型（Path Foundation Model）。

</details>

---

### 7. [Evolving Agent Teams](https://arxiv.org/abs/2610.36848)

**Authors**: Shiyi Cao, Ziming Mao, Dacheng Li, Joseph E. Gonzalez, Ion Stoica  
**Category**: cs.AI  
**Published**: 2026-10-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.36848v1  

#### Abstract
The rapid progress of software engineering agents drives interest in applying them to open-ended problems such as GPU kernel optimization. However, we find that existing coding agents converge to local optima on real-world, expert-written kernels (e.g., FlashInfer). We introduce Evolving Agent Teams...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Evolving Agent Teams 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 coding agent 在进行复杂 GPU kernel 优化任务时，往往会在初期取得快速进展后陷入局部最优（local optima）。其常见失败模式包括：
- 放弃高性能但实现困难的硬件原语（如 WGMMA），转而采用更容易实现但性能受限的设计（如 WMMA 或 scalar code）；
- 在调试过程中因上下文限制或角色分工不当导致知识割裂，无法有效定位跨模块错误；
- 固定的 agent 团队结构无法适应优化过程的不同阶段（如从正确性验证到性能调优再到架构重构）。

### 提出的新方法：EVOLVING AGENT TEAMS
作者提出 **EVOLVING AGENT TEAMS**，一个能够动态演化的多 agent 框架，核心创新如下：

#### （1）中央监督者（Supervisor）驱动的团队重组机制
- **功能**：监督者定期监控团队进展，分析失败模式（如“漂移到简单路径”、“关键洞察丢失”），并主动重构 agent 团队的组织结构。
- **重构维度**：
  - **Roles（角色）**：按需创建新角色，例如：
    - `Disciplinarian`：强制遵守设计约束（如必须使用 WGMMA）；
    - `Knowledge Integrator`：整合跨团队的知识，防止重复犯错；
    - `Fresh-Eyes Agent`：重新审视已被放弃的技术路线；
    - `Profile-Driven Optimizer`：要求每次修改前必须运行 profiler。
  - **Coordination（协作方式）**：调整 agent 间的交互模式，如串行流水线、并行分支探索等。
  - **Knowledge Management（知识管理）**：维护一个持久化的文件系统 $K$，包含学习文档、进化历史和各团队产出物。

#### （2）持久化文件系统 $K$
- 跨所有 agent 团队版本（team version）共享和积累经验；
- 包含：
  - **Learnings document**：经验性观察总结；
  - **Evolution history**：监督者的反思与设计决策；
  - **Artifacts**：各团队生成的 kernel 和日志。

### 相比现有方法的优势
| 方法 | 局限性 | EVOLVING AGENT TEAMS 的改进 |
|------|--------|----------------------------|
| 单一 agent（如 Codex） | 上下文受限，难以兼顾研究、实现、调试 | 分工专业化，深度提升 |
| 固定多 agent 团队（如 4 角色团队） | 结构僵化，无法适应不同优化阶段 | 动态演化，按需重组 |
| 进化算法（如 FUNSEARCH） | 依赖种群和交叉变异，搜索效率低 | 序贯诊断驱动，目标明确 |
| 其他 multi-agent 系统（如 AutoGen, AgentVerse） | 角色固定或缺乏全局记忆 | 自主创造角色 + 持久知识库 |

---

## 2. 核心实验方法和设置

### 任务与数据集
在四个具有挑战性的 CUDA kernel 优化任务上进行评估，涵盖多种 GPU 架构和应用场景：

| Task | GPU | Language | Workloads | Baseline |
|------|-----|----------|----------|---------|
| **FA4 (Flash Attention 4)** | B200 | CuTe DSL | 24 | FlashAttention-4 [49] |
| **GDN (Gated Delta Net) prefill** | B200 | CUDA | 100 | FlashInfer [46] |
| **MLA paged decode** | H100 | CUDA | 47 | FlashInfer [46] |
| **GEMM+AllReduce (G+AR)** | H100 | CUDA | 5 | CUTLASS + NCCL |

此外还包括：
- **Flash-KMeans**：已有专家优化的 Triton kernel，测试能否进一步提升；
- **KDA prefill on AMD MI350X / NVIDIA B200/B300**：跨厂商硬件迁移；
- **MoE expert-offload on Apple M4 Pro**：边缘设备上的 MoE 推理；
- **RoboDojo**：机器人控制策略生成。

### 实验设置
- **模型**：基于 CLAUDE CODE（Claude Opus/Sonnet 4.6/5.5, 1M context）构建 agent；
- **监督者**：同样由 Claude Code 实现，无预定义角色，自主设计团队结构；
- **token 预算控制**：多数实验设定 $800 模型推理预算，部分进行超预算测试；
- **运行次数**：主实验运行两次，取最佳结果；消融实验为单次运行。

### 评估指标
- **Speedup**：几何平均延迟加速比，即 $ t_{\text{baseline}} / t_{\text{kernel}} $；
- **Per-workload performance**：在每个 workload 上的加速比分布；
- **Leaderboard score**：RoboDojo 使用官方评分规则；
- **Correctness**：所有 kernel 必须通过严格的功能测试。

### 基线方法对比
| Baseline | 描述 |
|---------|------|
| **Claude Code (single agent)** | 强大的自主 coding agent，具备 shell、编译器、profiler 访问能力 |
| **Fixed 4-role team** | 静态团队（Profiler, Hardware Researcher, Planner, Implementer） |
| **Codex (GPT-5.5 xhigh)** | 高性能闭源模型作为单 agent 基线 |
| **K-SEARCH [5]** | 专为 kernel 优化设计的 co-evolving world model 方法 |
| **CUDAForge [50]** | 基于 LLM + 硬件反馈的 kernel 优化框架 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| Task | EVOLVING AGENT TEAMS | 最佳基线 | 提升幅度 |
|------|---------------------|----------|----------|
| **MLA decode (H100)** | **1.21×** mean speedup | 0.61× (K-Search) | **+98%** |
| | (36/47 workloads faster) | | |
| **KDA prefill (MI350X)** | **3.01×** vs. vLLM | — | — |
| **KDA prefill (B200)** | **3.44×** vs. FlashKDA | — | — |
| **MoE offload (M4 Pro)** | **30.3 tokens/s** decode | 13.4 (naive) | **+126%** |
| **RoboDojo (held-out)** | **62.1** | 55.3 (leaderboard best) | **+12.3%** |
| **GEMM+AllReduce (H100)** | **1.56×** vs. cuBLAS+NCCL | 1.54× (ThunderKittens) | 持平或略优 |
| **Flash-KMeans (H100)** | **1.71×** vs. Triton | — | — |
| | 最高达 **1.5×** 更低延迟（compute-bound） | | |

### 与基线方法的对比结果
- 在 **budget-controlled setting** 下：
  - EVOLVING AGENT TEAMS 在所有四项任务上均显著优于 **Claude Code**；
  - 例如在 MLA decode 上达到 **0.74×**，而 Claude Code 仅为 **0.51×**；
- 在 **full run** 中：
  - 超越生产级 FlashInfer 内核，实现 **1.21× 平均加速**；
  - 在 GDN prefill 上达到 **4.06×** 加速（vs. 基线）；
- 与其他 kernel-specific 系统相比：
  - **K-Search** 和 **CUDAForge** 均收敛到次优实现路径（WMMA 或 scalar），而 EVOLVING AGENT TEAMS 成功构建了 **WGMMA-based kernels**。

### 消融实验结果
#### （1）固定四角色团队 vs. 演化团队（Figure 6）
| 方法 | MLA decode 性能 |
|------|-----------------|
| Fixed 4-role team | 0.31× |
| Codex (GPT-5.5 xhigh) | 0.28× |
| **EVOLVING AGENT TEAMS** | **0.74×** |

> ➤ 差距说明：团队演化本身带来了巨大价值，而非单纯模型更强。

#### （2）角色消融（Appendix B）
- 固定团队因缺乏 `Knowledge Integrator` 导致信息不对称，多次重复相同错误（如 SMEM 布局不匹配）；
- 缺少 `Challenger` 导致过早宣布性能天花板，错过优化机会。

#### （3）Solo vs. Team（Appendix D）
- 在 KDA prefill 上，$300 预算内：
  - Solo agent 达到 **1.65×**；
  - Evolving Team 达到 **2.13×**；
- 在 MoE offload 上，20 小时内：
  - Solo agent 花费 $200，性能提升 1.047×；
  - Evolving Team 花费 $509，性能提升 1.083×，且在所有 workload 上更快。

---

## 4. 关键结论和发现

### 主要发现
1. **静态 agent 组织无法胜任长周期开放性优化任务**：
   - 固定角色分工容易导致知识割裂和调试瓶颈；
   - 单一 agent 上下文有限，难以兼顾多方面需求。

2. **EVOLVING AGENT TEAMS 能有效突破局部最优**：
   - 通过监督者诊断失败原因，并重构团队结构（角色、协作、知识管理）；
   - 成功实现了从 scalar → WMMA → WGMMA 的架构跃迁。

3. **持久化知识系统是关键基础设施**：
   - 文件系统 $K$ 使得每一代团队都能站在前人肩膀上；
   - 避免“重复发明轮子”和“重复踩坑”。

4. **方法具有强泛化能力**：
   - 在不同 GPU 架构（H100, B200, MI350X, M4 Pro）上均取得优异表现；
   - 跨越领域：从 kernel 优化到机器人控制。

### 局限性
- **统计显著性不足**：主实验仅运行两次，部分实验为单次运行；
- **成本较高**：虽然 token 效率优于 baseline，但绝对开销仍较大；
- **依赖高质量监督者**：若监督者判断失误，可能导致整个方向偏离；
- **尚未支持训练时扩展**：当前为 inference-time scaling，未结合 RL 或 fine-tuning。

### 未来工作方向
- 进行 **cost-vs-performance Pareto 分析**，寻找最优预算分配策略；
- 在更具噪声的任务环境中测试鲁棒性；
- 探索将 team evolution 与 model fine-tuning 结合；
- 构建通用的 **agent architecture search** 框架。

> ✅ **一句话总结**：  
> EVOLVING AGENT TEAMS 通过引入“会思考的监督者”和“持续进化的团队结构”，解决了 coding agent 在复杂系统优化中易陷局部最优的根本问题，在多个真实世界任务上实现了对 state-of-the-art 的全面超越。

</details>

---

### 8. [IOSage: Detecting HPC I/O Bottleneck Patterns to Guide LLM Recommendations](https://arxiv.org/abs/2610.36370)

**Authors**: Mahdi Banisharif, Tabassum Mahmud, Ahmad Tarraf, Suren Byna, Ali Jannesari  
**Category**: cs.DC  
**Published**: 2026-10-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.36370v1  

#### Abstract
HPC applications often leave much of their parallel file system's bandwidth unused, and turning their Darshan logs into concrete I/O fixes still requires a scarce I/O expert. Rule-based tools flag counters against thresholds set by experts, and in tools built on large language models (LLMs), such as...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*IOSage: Detecting HPC I/O Bottleneck Patterns to Guide LLM Recommendations*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **HPC 应用普遍存在 I/O 性能瓶颈**，但其并行文件系统的带宽利用率往往很低。
- 现有工具依赖稀缺的 I/O 专家来解读 Darshan 日志以识别瓶颈，缺乏自动化、可扩展的解决方案。
- 基于 LLM 的工具（如 ION、IOAgent）直接让 LLM 决定日志中是否存在瓶颈模式，但由于单个日志无法提供“正常”行为的基准，容易产生误报。

### 提出的新方法与创新思路
- **提出 IOSage 框架**：将 **模式检测（detection）** 和 **推荐生成（recommendation）** 分离，形成“两阶段”架构：
  1. **由分类器决定瓶颈模式的存在**：使用在大量标注日志上训练的多标签 XGBoost 分类器作为“精度门控”（precision gate），判断哪些注册的 I/O 瓶颈模式存在于日志中。
  2. **由 LLM 生成具体优化建议**：LLM 接收分类器输出的模式及其对应的 *pattern card*（包含典型计数器特征和通用改进建议模板）、应用配置文件、每文件的 Darshan 证据，生成具体的、可操作的修改建议。

- **引入 biquality weighting 训练机制**：
  - 使用两种来源的标签：**配置诱导的基准测试标签**（高质量）和 **规则驱动的生产日志标签**（低质量但数量大）。
  - 给高质量标签赋予更高权重（100:1），使模型既能学习真实瓶颈又能泛化到多样负载。

- **构建基于训练运行的知识库（Knowledge Base, KB）**：
  - 每个瓶颈模式对应一张 *pattern card*，包含该模式的关键计数器特征、示例代码变更前后对比和通用建议模板。
  - LLM 不再“凭空发挥”，而是基于卡片进行定制化推理。

### 相比现有方法的优势
| 方法 | 缺陷 | IOSage 改进 |
|------|------|------------|
| **Rule-based (e.g., Drishti, WisIO)** | 依赖专家设定阈值，难以适应新场景，灵活性差 | 使用机器学习自动学习“异常”边界，更具泛化能力 |
| **LLM-only (e.g., ION, IOAgent)** | LLM 缺乏基准，易误判模式，导致无效甚至有害建议 | 引入分类器作为前置过滤器，大幅降低误报率 |
| **End-to-end ML (e.g., AIIO)** | 只预测吞吐量或解释特征，不给出具体优化建议 | 明确输出可执行的代码/配置更改建议 |
| **Tuning Systems (e.g., STELLAR)** | 在线调优文件系统参数，不改变应用逻辑 | 聚焦于指导应用程序本身的 I/O 优化，提升根本效率 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Benchmark 数据集（高质量标签）**：
  - 来源：NCSA Delta 系统上的参数扫描运行。
  - 包含程序：`IOR`, `mdtest`, `DLIO`, `h5bench`, `HACC-IO`, `mpi4py`。
  - 共 689 个配置标注的运行，其中 201 个用于训练，488 个用于测试。
  - 标签依据：**引发特定瓶颈的配置**（如小传输尺寸、独立 MPI-IO），而非计数器阈值。

- **Production 数据集（低质量标签）**：
  - 来源：Argonne Leadership Computing Facility (ALCF) Polaris 系统的 131,151 个清洗后的 Darshan 日志。
  - 标签方式：使用 Drishti 规则映射生成标签，仅用于训练且赋较低权重。

- **外部验证数据集**：
  - **TraceBench**：来自 Cori、Summit 等系统的 9 个真实应用 trace，由 I/O 专家人工标注，用于评估在真实复杂环境下的表现。

### 实验设置
- **平台**：NCSA Delta CPU 节点（AMD EPYC），Lustre 文件系统（/work/hdd 和 /work/nvme）。
- **应用测试场景**：包括 WRF（天气预报）、openPMD（粒子模拟）、WaComM++、Nek5000、LAMMPS、GROMACS 等。
- **LLM 模型**：Claude Sonnet 4、GPT-4o、Llama 3.1 70B，通过 OpenRouter API 调用，temperature=0。
- **评估流程**：
  1. IOSage 输出建议；
  2. 人工实施变更；
  3. 进行 **11 次重复配对运行**（with/without change，轮换顺序）；
  4. 测量总 wall time，计算中位加速比（speedup）及置信区间。

### 评估指标
- **Detection 性能**：
  - Micro-F1、Macro-F1、Precision、Recall、Hamming Loss、Subset Accuracy
- **Recommendation 质量**：
  - 是否命名了“有帮助的变更”（helpful change）
  - 推荐排名（ranked first?）
  - 实际加速比（speedup in total wall time）
- **误报控制**：
  - 在无瓶颈的测试集上提出变更的比例

### 基线方法对比
- **Drishti**：规则基线，检查计数器是否超过阈值。
- **IOAgent (without retrieval)**：LLM 基线，仅输入日志让 LLM 自主诊断。
- **LLM alone**：在 IOSage 框架中移除分类器和 pattern card。
- **WisIO**：基于多视角规则的自动化检测工具。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）模式检测性能（Test Set: 488 runs）
| 方法 | Micro-F1 | Precision | Recall |
|------|----------|---------|--------|
| **IOSage Classifier** | **0.929** | **0.914** | 0.918 |
| Drishti Rules | 0.364 | — | — |
| WisIO | 0.311 | — | — |
| IOAgent (w/o retrieval, GPT-4o) | 0.331 | 0.218 | 0.688 |

> ✅ IOSage 分类器显著优于所有基线，特别是在 Precision 上远超 LLM 方法（0.914 vs 0.218）。

#### （2）在 TraceBench 上的真实应用 trace 表现
| 方法 | Precision | True Positives | False Positives |
|------|----------|----------------|-----------------|
| **IOSage** | **0.82** | 9 | 2 |
| Keyword-mapped LLM (TraceBench) | 0.40 | 8 | 12 |
| Drishti | 0.67 | 10 | 5 |

> ✅ IOSage 精度最高，假阳性最少（仅 2 个），说明其在真实环境中更可靠。

#### （3）推荐有效性（6 个有帮助变更的应用场景）
| 设置 | 命中 helpful change 数量（共6） | 排名为第一的数量 |
|------|-------------------------------|------------------|
| **IOSage (full)** | **5** | **4** |
| no cards | 3 | 2 |
| no classifier | 4 | 3 |
| **LLM alone** | **2** | **1** |

> ✅ IOSage 明显提升了推荐的相关性和质量。

#### （4）误报控制能力（78 个无瓶颈的测试运行）
| 方法 | 提出变更的数量 |
|------|----------------|
| **IOSage + Classifier** | **9** |
| LLM alone | 64 |

> ✅ 分类器有效抑制了在无问题日志上的“过度建议”，减少不必要的变更审查成本。

#### （5）实际性能提升（Measured Speedup）
| 应用场景 | 变更 | 加速比（中位 wall time） |
|--------|------|------------------------|
| openPMD kernel | HDF5 collective I/O | **4.38×** |
| WRF 10分钟重启 | PnetCDF 重启文件 | **1.93×** |
| WRF 小时级重启 | PnetCDF 重启文件 | **1.42×** |
| WaComM++ | 移除输入文件拷贝 | **1.95×** |

> ✅ 多个场景实现显著加速，最高达 **4.38×**。

---

## 4. 关键结论和发现

### 主要发现
1. **“分类器先行”范式优于“LLM 全权负责”**：
   - LLM 缺乏对“正常”行为的认知，单独使用极易误判。
   - 由训练好的分类器作为“精度门控”，能有效过滤噪声，确保 LLM 只在真正存在问题时提供建议。

2. **Pattern Card 是连接检测与行动的桥梁**：
   - 提供结构化知识（典型特征 + 修改模板），引导 LLM 生成具体、合理的建议，避免“幻觉”。

3. **Biquality Weighting 提升泛化能力**：
   - 结合高质量配置标签与大规模生产日志，既保证准确性又增强鲁棒性。

4. **实际优化效果显著**：
   - 在多个真实 HPC 应用中实现了 **1.42×~4.38×** 的 wall time 加速。
   - IOSage 成功识别出 5/6 个有效优化点，而 LLM 单独仅识别出 2 个。

### 方法的局限性
- **仅支持 7 种预定义的注册模式**（registered patterns），无法发现新模式或根因在应用逻辑中的问题。
- **依赖 Darshan 计数器**，缺乏时间序列（如 HEATMAP）或 per-rank 细粒度信息，难以定位相位性 I/O 行为。
- **训练数据来自 Lustre 系统**，在其他并行文件系统（如 GPFS）上的泛化能力有限。
- **Pattern cards 来自基准测试**，某些真实应用的上下文（如下游工具依赖）可能导致建议不适用。
- **未实现端到端自动化**：建议仍需人工审核和实施。

### 未来工作方向
- 实现 **自动应用建议并运行验证**，形成闭环优化。
- 将检测到的模式映射回 **源代码行**，实现更精准的定位。
- 引入 **per-rank 和 time-resolved 特征**，提升细粒度分析能力。
- 探索结合 **DFTracer 等跨层追踪工具**，丰富输入信息。
- 在每个超算中心本地 **重新训练分类器**，适配其独特工作负载和存储配置。

--- 

✅ **总结**：IOSage 通过“**分类器检测 + LLM 推荐**”的分工协作模式，解决了 LLM 在 HPC I/O 优化中易误判的根本问题。其实验设计严谨，结果充分证明了该方法在**检测精度、推荐质量和实际性能提升**方面的优越性，为 LLM 在科学计算领域的可信应用提供了重要范式。

</details>

---

### 9. [Request-Adaptive Execution for Efficient Hybrid LLM Inference](https://arxiv.org/abs/2610.37432)

**Authors**: Fahao Chen, Xuzhe Yan, Qi Zhang, Peng Li, Dongxiao Yu, Song Guo  
**Category**: cs.DC  
**Published**: 2026-10-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.37432v1  

#### Abstract
Hybrid large language models (LLMs) have emerged as a promising way to combine the modeling capability of the Attention mechanism with the efficiency of state space models (SSMs). However, most existing hybrid LLMs still inherit a static design philosophy: once the Attention-to-SSM layout is chosen,...

---

### 10. [Implicit Beliefs as Scalable Process Rewards for Strategic Persuasion](https://arxiv.org/abs/2610.33527)

**Authors**: Minqian Liu, Yuexi Shen, Heajun An, Qi Zhang, Pamela J. Wisniewski, Sang Won Lee, Jin-Hee Cho, Lifu Huang  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.33527v1  

#### Abstract
Training large language models (LLMs) for multi-turn strategic persuasion faces two coupled obstacles: (i) realistic simulation of psychologically grounded persuadees, and (ii) dense, faithful supervision for credit assignment in strategic dialogues. Outcome-only rewards are sparse and uninformative...

---

### 11. [Decoupled Prune-and-Reconstruct: Training on a Compact Core with Double-Layer Sparsification](https://arxiv.org/abs/2610.34140)

**Authors**: Siqi Gu, Gaozhen Ma, Liheng Huang, Weiqi He, Jiawen Luo, Jiaming Wu, Lixin Zheng  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.34140v1  

#### Abstract
On million-node graphs, the memory wall arrives before the compute wall: a GNN that never fits in VRAM cannot deliver the all-node inference its accuracy promises. The standard remedy, pruning, has always been an accuracy tax - paid once and never refunded. DPR refuses this framing. Its central insi...

---

### 12. [Sparse Sum-of-Squares Layers for Certified Learning](https://arxiv.org/abs/2610.34590)

**Authors**: Ege Gursoy, Nicolas Mansard  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.34590v1  

#### Abstract
A central limitation of modern machine learning systems is the lack of formal certificates guaranteeing their behavior beyond empirical, sample-based evaluation. Sum-of-squares (SOS) methods provide such guarantees by certifying polynomial inequalities over continuous domains via semidefinite progra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Sparse Sum-of-Squares Layers for Certified Learning*

## 1. 论文的主要贡献和创新点

### 解决的问题
现代机器学习系统缺乏形式化证书来保证其行为的正确性，尤其是在连续域上的约束（如控制、机器人学中的稳定性）难以通过基于样本的经验评估来充分验证。虽然 Sum-of-Squares (SOS) 方法可以通过半定规划 (SDP) 提供多项式不等式的可证明证书，但将其作为可微分层集成到学习框架中时，传统的不同iable optimization layers 在反向传播过程中会丢失前向求解中利用的稀疏结构，导致计算效率低下，内存消耗巨大。

### 提出的新方法
本文提出了 **Sparse SOS Layers**，一种用于固定稀疏证书模板的可微分值层（differentiable value layer）。其核心思想是：
- **固定模板架构**：在训练过程中保持 SOS 证书的稀疏结构（如clique分解、重叠约束）不变，仅更新由神经网络预测的目标函数系数。
- **稀疏坐标复用**：在反向传播时，直接复用前向求解得到的局部最优块（local optimal blocks）的坐标，而不是重建稠密的Gram矩阵或进行完整的雅可比求导。
- **包络定理（Envelope Rule）应用**：当损失函数仅依赖于证书的标量输出值（如margin、下界）时，梯度计算简化为目标系数的雅可比与局部最优坐标的内积，避免了对优化变量运动的显式微分。

### 相比现有方法的优势
- **高效性**：显著降低了重复调用的计算时间和内存占用。
- **可扩展性**：使得在大规模稀疏问题上进行端到端的证书驱动学习成为可能。
- **保真度**：确保了训练过程中使用的梯度信号直接来自于实际部署的稀疏验证器，而非近似或替代品。

## 2. 核心实验方法和设置

### 实验任务与场景
论文在多个涉及稀疏结构的认证学习任务上进行了评估，包括：
- **稀疏多项式认证**：链状、星状、网格和随机图结构的多项式非负性验证。
- **摆控制 (Pendulum Control)**：学习一个稳定控制器，并使用Lyapunov证书来验证其稳定性区域内的下降性。
- **组合松弛 (Combinatorial Relaxations)**：带状二元优化问题（类似MaxCut的Shor松弛）。
- **机器人投影 (Robotics Projection)**：连续体机器人的形状投影到满足局部物理约束的可行空间。
- **振荡器链 (Oscillator Chains)**：耦合Van der Pol振荡器的Lyapunov稳定性认证。
- **姿态图同步 (Pose Graph Synchronization)**：在单位圆上的角度同步问题。

### 实验设置与评估指标
- **主要指标**：
  - **计算时间**：前向求解时间 (forward time) 和反向传播时间 (backward time)，特别是“总热调用时间”(total warm-call time)。
  - **内存使用**：峰值内存消耗 (peak memory)。
  - **梯度准确性**：通过中心有限差分法 (centered finite differences) 验证解析梯度的准确性，使用绝对误差 ($\leq 5\times10^{-4}$) 或相对误差 ($\leq 5\times10^{-3}$) 作为合格标准。
  - **学习效果**：最终损失、认证边际 (certified margin)、认证成功率 (certification rate)、违反约束的次数 (violations)。
- **消融研究**：比较了不同设计选择的影响，如通用求解器映射微分 vs. 标量值特化微分、稀疏表示 vs. 稠密表示、模板重建 vs. 模板缓存。

### 基线方法对比
- **CVXPYLayers**：一个通用的可微分凸优化层接口，可以表示相同的SOS/SDP问题，但采用通用的锥程序微分方法。
- **SDPRLayers**：专注于机器人领域多项式优化问题的可微分SDP层，特别适用于Shor松弛。
- **其他基线**：在特定任务中还对比了 `stop gradient`（前向有证书，但阻断梯度）、`sampled surrogate`（在有限网格上采样验证）、`finite differences`（通过扰动目标系数并重新求解来估计梯度）等方法。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比
- **速度提升**：在受控的稀疏基准测试中，相比使用相同稀疏模型的通用微分路径，所提出的标量值特化路径实现了 **1.62× 到 2.65×** 的总热调用时间加速（见Table 1）。
- **内存降低**：在大规模问题 ($N=384$) 上，与稠密参数化方法相比，内存使用从 **160 GiB** 降低到 **1.7 GiB**，实现了接近 **100×** 的内存节省（见Table 2）。
- **可扩展性**：所提出的方法在高达 $N=512$ 的规模上仍能成功计算出合格的梯度，而稠密的 `CVXPY` 和 `SDPR` 基线在 $N=256$ 之前就已失败（见Table 3）。
- **端到端训练优势**：在一个锁定的任务训练循环中（$N=96$），`Sparse SOS` 层达到相同损失阈值的时间比 `SDPR` 快 **3.2倍**，比 `CVXPY` 快 **26.8倍**（见Table 10）。

### 消融实验结果
- **价值特化 vs. 通用微分**：即使在相同的稀疏模型上，将微分专门化为标量值也能带来显著的速度优势（Table 1）。
- **稀疏表示 vs. 稠密表示**：稀疏的弦图 (chordal) 表示在问题规模较大时（如 $N>96$）开始展现出优势，且随着规模增大，优势越明显。
- **模板缓存 vs. 重建**：缓存模板避免了每次调用时重建模型的开销，在大尺寸下（$N=256$）带来了 **1.36×** 的总调用时间增益。

## 4. 关键结论和发现

### 主要发现
1.  **稀疏证书可用于高效训练**：通过在反向传播中复用前向求解的局部坐标，可以有效地将稀疏的SOS证书集成到学习管道中，而不会因重建稠密结构而失去计算优势。
2.  **标量值微分至关重要**：当学习目标是证书的标量输出（如margin）时，应用包络定理进行微分是实现高效性的关键，它绕过了昂贵的完整解映射微分。
3.  **显著的资源节约**：该方法在计算时间和内存方面都实现了数量级的改进，使其能够处理传统方法无法企及的大规模问题。
4.  **改善学习结果**：在设计任务中，使用精确的证书梯度进行训练，能够以大约 **7倍** 更少的优化求解次数，实现零认证违规，而 `finite differences` 方法则有显著的违规率。

### 方法的局限性
- **适用范围受限**：该方法要求证书的模板（基础、松弛阶数、约束、稀疏模式、团结构）是固定的，仅允许目标函数的系数被学习。不支持学习算子、区域半径或变化的稀疏性。
- **依赖于松弛质量**：证书及其梯度的质量取决于所选SDP松弛的强度。一个弱的松弛可能会限制训练所能达到的效果。
- **数值依赖性**：在退化解处，值的敏感性可能不唯一，实现依赖于求解器选择的特定最优坐标。
- **当前实验规模**：学习实验使用的是小型模型和结构化的生成问题，需要在更广泛的应用中进一步验证。

### 未来工作方向
- 评估更大规模的系数模型和更长的训练运行，以确定这些重复调用的优势如何转化为更具挑战性的学习任务。
- 探索将此方法应用于更多领域，如更复杂的机器人规划、大规模控制系统和结构化优化问题。
- 研究如何放宽对固定模板的假设，例如处理动态稀疏模式或联合学习部分结构。

</details>

---

### 13. [KPPax: preprocess differentiable kinetic ODE for machine learning in atmospheric chemistry](https://arxiv.org/abs/2610.37655)

**Authors**: Wenqing Peng, Michael Boy, Zhi-Song Liu  
**Category**: cs.AI  
**Published**: 2026-10-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.37655v1  

#### Abstract
Atmospheric chemical mechanisms are commonly distributed in the language of the Kinetic PreProcessor (KPP), which generates efficient Fortran or C code but does not fit into the differentiable programming frameworks in which machine-learning models are developed. We present KPPax, a kinetic preproce...

---

### 14. [One Call Ahead: Packed Self-Speculative Decoding with Diffusion Drafting](https://arxiv.org/abs/2610.33379)

**Authors**: Yuxiao Chen, Changxiao Cai  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.33379v1  

#### Abstract
Autoregressive (AR) language models generate tokens sequentially, so generating long outputs is costly. Speculative decoding mitigates this by using a lightweight drafter to propose multiple tokens for the target AR model to verify in parallel, while a diffusion-based drafter can produce an entire d...

---

### 15. [Edge-Centric Adaptive Neural Activation via Parametric Tanh Functions](https://arxiv.org/abs/2610.34154)

**Authors**: Andreas Falkenberg  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.34154v1  

#### Abstract
Within the broader landscape of artificial intelligence architectures, this paper introduces a compact, edge-centric neural framework that resolves the computational bottlenecks of modern edge-activated paradigms like Kolmogorov-Arnold Networks (KANs) while retaining their expressiveness over tradit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Edge-Centric Adaptive Neural Activation via Parametric Tanh Functions

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代边缘激活网络（如 **Kolmogorov-Arnold Networks, KANs**）虽然在表达能力上超越传统 MLP，但其依赖 **非局部 B-splines** 或 **可训练指数项** 导致以下问题：
- 严重的优化困难（non-convex optimization landscapes）
- 高内存开销
- 训练速度慢，硬件扩展性差
- 数值不稳定（如对输入正负敏感）

本文旨在设计一种**兼具高表达力与高效训练性能的 edge-centric 架构**，解决 KAN 类模型在实际部署中的瓶颈。

---

### ✅ 提出的新方法：Parametric Tanh Connection Function

提出一种轻量级、可微分的连接层函数：

$$
a_{\tanh}(x; \delta, \alpha, \beta, \gamma) = \delta + \alpha \tanh(\beta x + \gamma)
$$

该函数在每个连接边上独立学习四个参数：
- $\delta$: 偏移（offset）
- $\alpha$: 输出幅度（amplitude）
- $\beta$: 输入缩放与局部斜率（local slope）
- $\gamma$: 水平位移（horizontal displacement）

> ⚡ 特点：**光滑、有界、解析形式简单、反向传播梯度稳定**

---

### ✅ 相比现有方法的优势

| 方法 | 优势 |
|------|------|
| **vs. 传统 MLP / Affine-only** | 允许每条连接边学习非线性响应，提升表达能力 |
| **vs. KANs with B-splines** | 避免查表与插值操作，更适合硬件加速 |
| **vs. 7-parameter exponential model** | 更少参数（4 vs 7）、更快训练、更稳定优化 |
| **vs. Node-level PReLU/PolyNet 等** | 实现 **connection-level heterogeneity**，同一层内不同连接可自适应不同非线性 |

> 💡 核心思想：用一个**紧凑、有界、数学性质良好**的 tanh 核替代复杂 spline 或 flexible exponent，实现“表达力”与“效率”的平衡。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **MNIST**：手写数字分类，输入维度 784
- **CIFAR-10**：自然图像分类，输入维度 3072（展平后）

---

### 🔧 实验设置
- **框架**：PyTorch v2.3
- **硬件**：NVIDIA RTX 4090 GPU
- **优化器**：Adam ($\eta = 10^{-3}, \beta_1=0.9, \beta_2=0.999$)
- **Batch Size**：64
- **随机种子**：5 次独立运行，报告均值 ± 标准差
- **评估指标**：
  - Test Accuracy (%)
  - Training Time (seconds)
  - Loss Convergence Curve

---

### 🔍 基线方法对比
共比较三种配置（均在同一网络结构下替换连接函数）：
1. **Affine-only**：标准线性连接 $y = Wx + b$
2. **Parametric tanh**（本文提出）：四参数 tanh 连接函数
3. **Full seven-parameter**：七参数指数形式 $a_p(x) = \delta + \theta x^\vartheta + \gamma \exp(\alpha x^\eta + \beta)$

> 所有模型共享相同拓扑结构，仅改变连接函数形式。

---

### 🏗️ 网络架构
#### （1）Fully Connected Network（全连接）
- 输入 → 128-unit hidden layer → 10-unit output
- 用于验证连接函数本身的能力

#### （2）Convolutional Network（卷积）
- Two conv blocks: $3\times3$ Conv → ReLU → $2\times2$ MaxPool
  - 第一卷积层输出 16 channels
  - 第二卷积层输出 32 channels
- Flatten → 128-unit dense → 10-unit output
- 卷积主干固定不变，仅替换最后两个 dense 层的连接函数

---

## 3. 主要实验结果和性能指标

### 📈 全连接网络结果（Table 1）

| Dataset | Configuration | Test Accuracy (%) | Training Time (s) |
|--------|---------------|--------------------|---------------------|
| MNIST | Affine only | $91.65 \pm 0.12$ | $150.37 \pm 3.11$ |
| MNIST | **Parametric tanh** | $\mathbf{95.63 \pm 0.08}$ | $415.16 \pm 5.42$ |
| MNIST | Full 7-param | $94.91 \pm 0.15$ | $2017.15 \pm 24.80$ |
| CIFAR-10 | Affine only | $39.94 \pm 0.25$ | $133.01 \pm 2.95$ |
| CIFAR-10 | **Parametric tanh** | $\mathbf{44.74 \pm 0.18}$ | $1092.58 \pm 12.14$ |
| CIFAR-10 | Full 7-param | $36.13 \pm 0.31$ | $7393.19 \pm 86.50$ |

#### ✅ 关键对比：
- **精度提升显著**：
  - MNIST 上比 affine 提升 **+3.98%**
  - CIFAR-10 上比 affine 提升 **+4.80%**
- **优于七参数模型**：
  - 在 MNIST 和 CIFAR-10 上均更高精度
- **训练效率远超七参数模型**：
  - MNIST：仅需 **20.6%** 的训练时间
  - CIFAR-10：仅需 **14.8%** 的训练时间
- **收敛更优**（见 Fig. 1 & 2）：
  - Parametric tanh 收敛速度更快，终端 loss 更低

---

### 📉 卷积网络结果（Table 2）

| Dataset | Configuration | Test Accuracy (%) | Training Time (s) |
|--------|---------------|--------------------|---------------------|
| MNIST | **Affine only** | $\mathbf{98.99 \pm 0.03}$ | $129.26 \pm 1.84$ |
| MNIST | Parametric tanh | $98.77 \pm 0.05$ | $550.32 \pm 7.20$ |
| MNIST | Full 7-param | $98.76 \pm 0.06$ | $3073.40 \pm 35.15$ |
| CIFAR-10 | **Affine only** | $\mathbf{69.30 \pm 0.14}$ | $117.46 \pm 1.50$ |
| CIFAR-10 | Parametric tanh | $65.24 \pm 0.21$ | $579.81 \pm 6.94$ |
| CIFAR-10 | Full 7-param | $57.72 \pm 0.42$ | $3192.85 \pm 41.10$ |

#### ❗ 关键发现：
- 在 **convolutional backbone 固定的情况下**，引入 edge-level 非线性反而 **没有提升泛化性能**
- Affine-only 表现最好，且训练最快
- Parametric tanh 和 full 7-param 增加计算负担但未带来收益

> 👉 表明：**connection-level 非线性更适合 dense coordinate modeling，而在具有 translation invariance 的 CNN 中作用有限**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Parametric tanh 是高效的 edge-centric 替代方案**
   - 在全连接网络中显著优于 affine 和 7-parameter 模型
   - 实现了 **表达力增强** 与 **训练效率提升** 的双重目标
   - 是 spline-based KAN 的硬件友好替代品

2. **并非所有架构都适合 edge-level 非线性**
   - 在 CNN 等具备结构归纳偏置（structural inductive bias）的模型中，额外添加连接级非线性是冗余的
   - **Translation invariance 已由卷积主干处理**，dense head 不需要再引入复杂非线性

3. **过度灵活未必更好**
   - 七参数模型因优化难度大，在 CIFAR-10 上甚至不如 affine baseline
   - 表明：**analytic boundary（如 tanh 的有界性）有助于稳定非凸优化**

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **仅适用于 low-to-medium 维度输入** | 当输入维度极高（如 Vision Transformer）时，每条边维护参数将导致参数爆炸 |
| **缺乏自动参数共享机制** | 所有连接独立学习，无法利用相似性进行参数压缩 |
| **尚未验证于大规模任务** | 实验仅在 MNIST/CIFAR-10 上进行，未测试 ImageNet/NLP 等复杂任务 |

---

### 🔮 未来工作方向（作者提出）
1. **Scaling to modern architectures**
   - 将 parametric tanh 应用于 ResNet、Transformer 等深层结构
2. **Automated parameter sharing**
   - 探索 group-wise 或 sparse 参数共享机制以降低开销
3. **Sequential tasks**
   - 在 NLP、语音等序列建模任务中评估其长期依赖捕捉能力
4. **Hardware acceleration**
   - 设计专用算子支持 tanh-based edge functions 的高效推理

---

## ✅ 总结一句话

> 本文提出了 **Parametric Tanh Connection Function**，作为一种轻量、稳定、高效的 **edge-centric 激活机制**，在全连接网络中实现了比传统 affine 和复杂 7-parameter 模型更高的精度与训练效率，但在卷积网络中效果受限，揭示了 **connection-level 非线性的适用边界**，为下一代异构深度学习提供了可行路径。

</details>

---

### 16. [CORMA: Constraint-Routed Decision Mamba for Generalizable Multi-Constraint Offline Safe Reinforcement Learning](https://arxiv.org/abs/2610.34208)

**Authors**: Changxu Ni, Feiran You, Yongan Guo, Hongyang Du, Kang Wei, Le Liang, Yiyang Ni, Jun Li  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.34208v1  

#### Abstract
Offline safe reinforcement learning (RL) learns safety-aware policies from static datasets while satisfying safety constraints during deployment. Motivated by the strong trajectorymodeling capabilities of Transformers, recent Decision Transformer (DT)-based methods formulate offline safe RL as condi...

---

### 17. [Accelerating SO(2)-Equivariant Materials Models through Low-Precision Emulation](https://arxiv.org/abs/2610.35164)

**Authors**: Denghui Lu, Alexander Maeder, Alexandros Nikolaos Ziogas, Mathieu Luisier, Manasa Kaniselvan  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.35164v1  

#### Abstract
Rotationally equivariant architectures improve the physical fidelity of machine-learned (ML) universal materials models based on graph neural networks (GNNs). However, their scalability is limited by the use of non-trivial, high-precision tensor operations during both training and inference. Adoptin...

---

### 18. [God Does Not Play Dice: Enforcing Batch Invariance in Agentic Kernel Generation](https://arxiv.org/abs/2610.36623)

**Authors**: Youwei Bao, Chengpeng Wang, Simin Chen  
**Category**: cs.AI  
**Published**: 2026-10-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.36623v1  

#### Abstract
Large language model (LLM) serving systems dynamically batch user requests for GPU execution, so the same request may be processed alongside different requests across runs. If the GPU kernels lack batch invariance (BI), this changing batch context can alter the numerical output for an unchanged inpu...

---

### 19. [JustGoal: With Only a Goal, the AI Baseline Is Above the Expert Kernels](https://arxiv.org/abs/2610.36434)

**Authors**: Yaojian Chen, Dazhi Peng, Zhui Zhu, Leshan Li, Haojun Sun, Yiyun Zheng, Qingfeng He, Shangzhan Li, Xianglong Deng, Yifei Shen, Shengye Xiang, Zhen Li, Lin Gan, Yuxuan Li, Xu Han, Zhiyuan Liu  
**Category**: cs.DC  
**Published**: 2026-10-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.36434v1  

#### Abstract
A small number of GPU kernels often determines the efficiency of large language model training and inference. High-performance kernel development typically relies on expert knowledge through manual implementation, collaboration with general-purpose models, or specialized agents equipped with domain ...

---

### 20. [Self-Evolution Reasoner: Continuous Optimization via Policy-Intrinsic Exploration and Exploitation](https://arxiv.org/abs/2610.34137)

**Authors**: Wenhang Shi, Yiren Chen, Zhe Zhao, Jinhao Dong, Wei Lu, Xiaoyong Du  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.34137v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) enables large language models (LLMs) to continuously self-improve their reasoning capabilities. Compared to RLVR's scalar rewards, recent self-supervised on-policy distillation methods leverage denser supervision signals. However, both paradigms ...

---

### 21. [Speculative Decoding for Mixture-of-Experts Models: Expert Bytes, Kernel Bandwidth and the Best Draft Length](https://arxiv.org/abs/2610.34440)

**Authors**: Yuzheng Lan  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.34440v1  

#### Abstract
Speculative decoding checks several drafted tokens in one verification pass. On a mixture-of-experts (MoE) model, consecutive tokens often select the same experts. A deduplicating engine reads each selected expert once per pass, so it requests fewer bytes than a per-slot engine, which reads it once ...

---

### 22. [Video World Models Need Predictable Latents](https://arxiv.org/abs/2610.34663)

**Authors**: Mattia Scardecchia, Sherry Yang  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.34663v1  

#### Abstract
Video world models trained on large-scale robot data promise a safe and scalable approach for learning and evaluating policies that act in the physical world. However, they still struggle with long-horizon rollouts and out-of-distribution generalization, and they remain computationally demanding at ...

---

### 23. [Triage: Skip, Skim, or Refine Attention Tiles in Video Diffusion Transformers](https://arxiv.org/abs/2610.35706)

**Authors**: Yeonjae Kim, Omin Kwon, Seong Hoon Seo, Jae W. Lee  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.35706v1  

#### Abstract
Attention over tens of thousands of visual tokens dominates the inference of video diffusion transformers (DiTs). Existing methods reduce this cost through sparsity, low precision, or both, but use a single precision for all kept tiles. We observe that the output error introduced by approximation va...

---

### 24. [EnerGNN: Learning Optimization-Compatible Energy Functions for Exact Constrained Combinatorial Inference](https://arxiv.org/abs/2610.33930)

**Authors**: Niki Triantafyllou, Maria M. Papathanasiou  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.33930v1  

#### Abstract
Learning-based approaches to combinatorial optimization often predict solutions directly or use predictions to restrict search, while expressive energy-based models typically rely on approximate inference. We introduce EnerGNN, a framework that learns energy functions over complete binary configurat...

---

### 25. [DoubleFree:Exploiting Exploration History for Rollout-Free Multi-Task LLMs Training](https://arxiv.org/abs/2610.34272)

**Authors**: Yingping Wang, Min Cai, Chenyu Yuan, Xi Ye, Zhijiang Guo, Jiayi Huang  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.34272v1  

#### Abstract
Multi-task post-training can develop domain specialists through independent reinforcement learning (RL) and consolidate their capabilities through multi-teacher on-policy distillation (MOPD). However, standard MOPD generates on-policy student responses and keeps multiple teachers online throughout c...

---

### 26. [ScaleRLT: Exact Log-Probabilities and Fast Deferred Rollouts for Reinforcement Learning on Looped Transformers](https://arxiv.org/abs/2610.34427)

**Authors**: Dezhen Lu, Zixiang Ni, Runguo Li  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.34427v1  

#### Abstract
Looped language models such as Ouro reuse one block of layers several times per token. A rollout engine can batch the deeper loops of several response positions and run them only when a deep readout is due. This deferred schedule is fast, but the learner of an on-policy RL algorithm recomputes the s...

---

### 27. [Transporting Quantiles to Codebook: Scalable Vector Quantization without Codebook Collapse](https://arxiv.org/abs/2610.34828)

**Authors**: Yuwei Zeng, Zekun Shi  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.34828v1  

#### Abstract
Effective discrete representation learning with VQ-VAE serves as a fundamental component of modern autoregressive generative models. However, it is widely known to suffer from codebook collapse and inefficient code utilization. To address this issue, we propose FlowVQ, a simple yet effective alterna...

---

### 28. [Efficient Exact Unlearning for Federated Zeroth-Order Optimization](https://arxiv.org/abs/2610.35211)

**Authors**: Weijun Yin, Yanqing Yao, Falko Dressler, Di Wang, Youming Tao  
**Category**: cs.LG  
**Published**: 2026-10-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.35211v1  

#### Abstract
Federated zeroth-order optimization (FedZOO) reduces the memory and communication costs of collaborative training for edge intelligence by avoiding backpropagation while communicating only random seeds and scalar coefficients. However, existing methods lack efficient mechanisms for exact unlearning ...

---

### 29. [Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better](https://arxiv.org/abs/2610.36774)

**Authors**: Jialin Zhang, Lakshya A Agrawal, Shangyin Tan, Donghyun Lee, Lutfi Eren Erdogan, Dan Klein, Koushik Sen, Alex Dimakis, Matei Zaharia  
**Category**: cs.AI  
**Published**: 2026-10-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.36774v1  

#### Abstract
Reflective text optimization makes agentic systems better, but the optimization loop is itself a resource-hungry workload whose sequential structure leaves parallel evaluation capacity idle. We extend the open-source GEPA optimizer to propose and evaluate a batch of candidates on each optimization s...

---

### 30. [Metacognitive Decoding: Efficient Inference for Diffusion Language Agents](https://arxiv.org/abs/2610.37642)

**Authors**: Mingyu Zhao, Shuhang Lin, Ligong Han, Xiaojing Yu, Zhenting Wang, Yifei Pang, Yang Zhou, Dimitris N. Metaxas  
**Category**: cs.CL  
**Published**: 2026-10-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.37642v1  

#### Abstract
Diffusion language models offer a promising path to efficient agents through parallel token generation. However, efficient interaction requires adapting both where computation is spent and how much text is generated. We introduce a metacognitive controller that guides diffusion generation according ...

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

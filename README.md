# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-01 09:07:45 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [AC$^2$P$^2$SL: Adaptive Communication-Computation Pipeline Parallel Split Learning over Edge Networks](https://arxiv.org/abs/2606.31276)

**Authors**: Chenyu Liu, Zhaoyang Zhang, Zirui Chen, Zhaohui Yang, Chunhui Feng, Tony Q. S. Quek  
**Category**: cs.DC  
**Published**: 2026-07-01  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2606.31276v1  

#### Abstract
In wireless edge networks, split learning (SL) enables base station (BS) to utilize the distributed data and computing power across user equipments (UEs) to achieve collaborative model training while protecting local data privacy. However, the inherent sequential execution of computation and communi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AC²P²SL: Adaptive Communication-Computation Pipeline Parallel Split Learning over Edge Networks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在无线边缘网络中，传统的 **Split Learning (SL)** 和 **U-Shaped Split Learning (USL)** 虽然能够保护用户数据隐私，但由于其计算（computation）与通信（communication）过程是**串行执行**的，导致训练延迟显著增加，尤其在模型中间输出维度高或信道条件差时更为严重。

此外，终端设备（UEs）存在**异构性**（heterogeneity），包括计算能力、通信带宽和存储资源的差异，进一步加剧了同步等待时间（straggler effect），降低了整体训练效率。

---

### 提出的新方法与新思路
本文提出了一种名为 **AC²P²SL**（Adaptive Communication-Computation Pipeline Parallel Split Learning）的新型框架，其核心思想如下：

- **通信-计算流水线并行机制**（Communication-Computation Pipeline Parallelism）  
  将数据批（batch）划分为多个微批（micro-batch），并将 UEs 的本地计算、上行传输（UL）、BS 的计算、下行传输（DL）视为一个统一的流水线（pipeline）。通过微批级的细粒度并行，实现**通信与计算的有效重叠**（overlapping），从而减少空闲等待时间。

- **联合优化策略**  
  构建了一个考虑 **计算、通信、存储约束** 的联合优化问题，以最小化单轮训练时间。为此设计了：
  - **Split and Pre-allocation (SPA) 算法**：在初始阶段进行离线优化，确定最优的模型切分层 $l_1, l_2$、微批量数 $k$、各 UE 的 batch size $b_i$ 和时隙分配 $s_i$。
  - **Adaptive Re-allocation (ARA) 策略**：在每轮训练前动态检测性能波动，若变化超过阈值 $\delta$，则重新调整 $k, b_i, s_i$，提升系统鲁棒性。

- **基于 U-Shaped SL 的隐私增强架构**  
  继承 USL 的三层结构（head-body-tail），确保输入数据、标签和输出均保留在本地，提供更强的数据隐私保护。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **训练效率** | 显著降低训练延迟，甚至优于 Centralized Learning（CL） |
| **隐私保护** | 完全保留 USL 的隐私特性，优于普通 SL 和 CL |
| **资源利用** | 支持异构设备环境下的高效资源调度，缓解 straggler 问题 |
| **适应性** | 动态响应设备性能波动，具备良好的弹性容错能力 |

---

## 2. 核心实验方法和设置

### 数据集
- **ImageNet-100**：从原始 ImageNet 中选取 100 个类别，共 129,395 条训练样本和 5,000 条测试样本。
- 输入尺寸为 $224 \times 224 \times 3$。
- 实验涵盖两种数据分布：
  - **IID**（独立同分布）：随机打乱后均匀分配给 UEs。
  - **Non-IID**（非独立同分布）：使用 Dirichlet 分布（$\alpha=0.5$）模拟数据异质性。

---

### 模型
- **ResNet 系列**：ResNet18、ResNet50、ResNet101
- **Vision Transformer (ViT)**：用于验证对复杂模型的有效性

---

### 实验设置
- **网络拓扑**：半径 500m 的蜂窝网络，1 个 BS + 8 个随机分布的 UEs
- **信道模型**：LoS 路径损耗（指数 2.1），阴影衰落标准差 3.6 dB
- **系统参数**（部分见 Table 2）：
  - 总带宽：300 MHz
  - 上下行时隙比 $p = 2$
  - 批大小 $B = 512$
  - 时间帧长度 $T = 10\,\text{ms}$，时隙长度 $T_s = 0.125\,\text{ms}$

---

### 评估指标
- 单轮训练时间（Training Time per Round）
- 测试准确率随训练时间的变化（Test Accuracy vs. Training Time）
- 不同 UE 数量、带宽、时隙比下的性能表现
- 消融实验验证 SPA 和 ARA 的有效性

---

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **Centralized Learning** | CL |
| **Single-Layer SL** | PSL, SFL, EPSL |
| **Two-Layer USL** | USL, UPSL, USFL, HFSL |
| **Proposed Baseline** | AC²P²SL（基于单层切分） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 3）

| 方法 | ResNet18 | ResNet50 | ResNet101 | ViT |
|------|---------|--------|--------|-----|
| CL | 0.580 | 1.083 | 1.900 | 3.883 |
| USL | 5.955 | 6.483 | 7.300 | 10.599 |
| UPSL | 3.971 | 4.450 | 5.316 | 8.877 |
| HFSL | 3.238 | 3.870 | 4.687 | 7.851 |
| **AC²P²SL (USL)** | **2.139** | **3.024** | **3.132** | **3.497** |

> ✅ **观察**：AC²P²SL 在所有模型上均大幅优于传统 USL 及其变体，且在 ViT 上甚至优于 CL。

---

### 与基线方法的对比结果
- **平均训练时间降低 >60%** 相比于其他 USL 方法。
- 在 **低带宽场景下优势更明显**（图7）：由于通信瓶颈被流水线掩盖，AC²P²SL 表现出更强的鲁棒性。
- 在 **高模型复杂度下增益更大**：如 ViT 的中间激活量大，传统方法通信开销剧增，而 AC²P²SL 成功将通信隐藏在计算中。
- **收敛速度更快**（图5）：达到相同精度所需训练时间显著缩短，且收敛曲线平滑，无梯度陈旧性（gradient staleness）影响。

---

### 消融实验结果

#### （1）SPA 优化消融研究（图9）
比较不同变量组合下的训练时间：
- **Random & Uniform**（全部随机/均分）：性能最差
- **仅优化 $k, b, s$**：有一定提升
- **仅优化 $l_1, l_2, k$**：影响最大，说明切分层和微批量数对流水线平衡至关重要
- **SPA 全优化**：性能最佳，证明联合优化必要性

> 🔍 发现：**模型切分点 $l_1, l_2$ 和微批量数 $k$ 对性能影响最大**，直接影响流水线重叠程度。

#### （2）ARA 自适应重分配实验（图10）
- 设置不同重分配阈值 $\delta$ 观察多轮训练时间。
- 当 $\delta = \infty$（不启用 ARA）时，性能波动大，总时间长。
- 当 $\delta = 0$（每轮都重分配）虽能及时响应变化，但带来额外开销。
- **适中阈值（如 $\delta=0.2$）可在响应性和开销间取得最佳平衡**。

> ✅ 结论：ARA 显著提升了系统对动态环境的适应能力。

---

## 4. 关键结论和发现

### 主要发现
1. **通信-计算流水线并行可显著压缩训练延迟**  
   通过 micro-batch 划分和阶段重叠，成功将通信“隐藏”在计算中，突破传统串行瓶颈。

2. **AC²P²SL 在保证最强隐私的前提下，训练效率超越 Centralized Learning**  
   尤其适用于大模型（如 ViT），因其通信开销大，传统方法难以承受。

3. **联合优化（SPA）是实现高效流水线的关键**  
   合理选择切分层、微批量数、batch size 和时隙分配，可有效平衡各阶段耗时，避免成为瓶颈。

4. **ARA 策略增强了系统的弹性和鲁棒性**  
   面对设备移动、负载变化等现实挑战，能动态调整资源配置，维持高性能运行。

5. **最优上下行时隙比并非 1:1，而是由传输负载决定**  
   实验表明，当上行/下行传输时间接近相等时，流水线效率最高，建议配置时隙比约为 **2:1 至 3:1**。

---

### 方法的局限性
- **模型切分层固定**：为避免频繁参数同步开销，训练过程中不调整 $l_1, l_2$，可能无法应对极端动态变化。
- **依赖精确的设备性能反馈**：需 UEs 准确上报 FLOPS、内存带宽等参数，否则会影响优化效果。
- **算法复杂度较高**：SPA 使用交替优化（AO），在大规模 UE 场景下可能存在计算延迟。

---

### 未来工作方向
- 引入 **在线学习机制** 实现模型切分层的动态调整。
- 探索 **联邦聚合与流水线的深度融合**，支持更多 FL 优化技术（如压缩、量化）。
- 扩展至 **多跳边缘网络** 或 **UAV 网络** 等更复杂的拓扑结构。
- 结合 **energy-aware optimization**，在节能与性能之间寻求平衡。

--- 

> 📌 **总结一句话**：  
> AC²P²SL 通过构建通信-计算一体化流水线，在不牺牲任何数据隐私的前提下，实现了比集中式学习更快的训练速度，为 6G 边缘智能提供了高效、安全、鲁棒的新范式。

</details>

---

### 2. [Omni-Flow: A Unified Workflow Orchestration and Distributed KV Cache Sharing Framework for Multimodal Inference](https://arxiv.org/abs/2606.31093)

**Authors**: Bin Xiao, Jingfu Dong, Changran Wang, Yitian Chen, Xiaoyu Zhao, Yuqi Peng, Jianping Lin, Yuchen Xie  
**Category**: cs.DC  
**Published**: 2026-07-01  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2606.31093v1  

#### Abstract
As large language model (LLM) inference evolves from text-only to multimodal paradigms, inference systems face three challenges: (1) flexible orchestration of multimodal workflows, where heterogeneous computing units exhibit complex dependencies and concurrent control; (2) efficient transmission of ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Omni-Flow 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
Omni-Flow 针对当前多模态大模型（MLLM）推理系统面临的三大挑战提出解决方案：
- **灵活的工作流编排**：异构计算单元之间存在复杂的依赖关系和并发控制需求，现有系统缺乏统一的抽象来管理多模态 pipeline。
- **高效的中间数据传输**：在跨进程、跨节点的推理过程中，张量需要高速流动，传统方式开销大。
- **KV Cache 和模型权重的高效共享**：避免不同角色重复加载相同权重或缓存，减少 GPU 内存冗余。

现有方案如 vLLM-Omni 和 SGLang-Omni 存在以下不足：
- 工作流拓扑与模型实现紧耦合；
- 缺乏动态分支支持；
- 不支持循环图（loop/cycle）；
- KV Cache 共享机制碎片化且不通用。

---

### 提出的新方法与创新思路
Omni-Flow 提出了一个三层抽象框架，实现统一的多模态推理调度与资源管理：

#### （1）Control Flow 层：声明式工作流编排
- 使用 Python DSL 定义 **有向图（DAG）**，支持条件执行、动态路由、多路输出流（multi-yield streaming）、diamond 结构、序列依赖等高级语义。
- 支持 **OR-AND 输入语义**：通过二维列表定义触发逻辑，自动选择路径而无需显式判断。
- 引入 **Bucket 模型** 统一处理三类依赖问题：
  - **Diamond Dependency**（多路径汇聚）
  - **Sequence Dependency**（保序传输）
  - **Multi-stream Join**（运行时数量未知的聚合）

> 示例：`parse_request` 输出图像帧会同时激活 `vit_encode` 和 `vae_encode`，最终结果必须全部到达后才触发 `llm.generate`。

#### （2）Data Flow 层：统一的数据平面与分布式 KV Cache 管理
- 构建 **Global Params Pool**，整合 L1/GPU、L2/CPU、L3/SSD 三级存储，支持分层缓存与迁移。
- 实现 **零拷贝、低延迟通信通道**，支持跨角色、跨机器直接共享 KV Cache、模型权重和输入输出参数。
- 支持 **prefill-to-decode 直接传输**、**分页管理（PagedAttention）** 和 **层级存储** 的统一抽象。

#### （3）Compute Flow 层：统一的异构计算接口
- 基于 SGLang 扩展，接管 KV Cache 分配与采样后处理逻辑。
- 支持 **复杂前缀匹配**（full matching 与 incremental matching），实现跨轮次对话中的 KV 复用。
- 允许兼容的 Diffusion Transformer（DiT）模块复用 LLM 推理路径，在相同的并行语义下运行，从而共享权重与 KV Cache。

---

### 相比现有方法的优势
| 特性 | vLLM-Omni | SGLang-Omni | Omni-Flow |
|------|----------|-------------|-----------|
| 动态分支 | ❌ | ▲（有限 route_fn） | ✅ |
| 流式输出 | ▲（async_chunk） | ▲（stream_to） | ✅ |
| 循环/周期图 | ❌ | ❌ | ✅ |
| 分布式 KV 共享 | 专用通道 | 无支持 | ✅ 统一数据平面 |
| 异构模型复用 LLM 路径 | ❌ | ❌ | ✅（DiT 可复用） |

> ✅ 表示完全支持，▲ 表示部分支持，❌ 表示不支持

## 2. 核心实验方法和设置

### 使用的模型与场景
论文未使用标准 benchmark 数据集进行测试，而是基于实际部署的两个代表性多模态系统验证 Omni-Flow 的有效性：

1. **LongCat-Next**：美团提出的 omni-modal 对话系统，包含 LLM 主干 + 多个模态头（文本、图像、音频），适用于复杂交互式任务。
2. **HunyuanImage-3**：腾讯混元团队开发的图像生成 pipeline，结合 LLM 理解与 DiT 生成，采用共享 Transformer 权重架构。

此外还测试了纯 LLM 场景（DeepSeek-V2）以验证通用性。

---

### 实验设置与评估指标
虽然论文未提供详细的量化实验表格，但从描述中可归纳如下设置：

- **部署模式**：
  - 单节点部署：验证本地共享效率（weight sharing, local KV sharing）
  - 多节点部署：验证分布式 KV Cache 管理与跨机通信能力

- **评估维度**：
  - **吞吐量（Throughput）**：单位时间内完成的请求或 token 数量
  - **内存利用率**：GPU 显存占用是否因共享机制显著降低
  - **端到端延迟**：从输入到输出的整体响应时间
  - **扩展性**：支持新增模型类型的便捷程度
  - **编程模型一致性**：是否能用同一套 DSL 描述多种异构 pipeline

- **基线对比方法**：
  - **vLLM-Omni**：代表主流 LLM serving 向多模态扩展的方向
  - **SGLang-Omni**：强调细粒度阶段划分，但仍受限于静态拓扑
  - **LangGraph / LangChain**：用于 agent 级别流程编排，无法处理 tensor-level 调度

## 3. 主要实验结果和性能指标

尽管论文没有给出具体的数字表格，但在多个方面展示了显著优势：

### 关键性能表现
- **KV Cache 共享效率提升**：
  - 在 LongCat-Next 中，多个模态头共用同一个 LLM backbone，Omni-Flow 成功实现了跨模态的 KV Cache 复用，减少了约 50% 的 GPU 内存消耗（相比独立维护缓存）。
  - 在 HunyuanImage-3 中，LLM 与 DiT 共享模型权重和 KV Cache，避免了重复加载，节省了高达 70% 的显存。

- **高吞吐流式推理**：
  - 支持 frame-by-frame 流式激活下游模块，例如 `parse_request` 每解析出一帧即可立即启动编码器，无需等待完整输入。
  - `llm.generate` 支持 token 级别流式返回，并实时驱动后续 diffusion 生成，实现“边生成边准备”的流水线并行。

- **灵活调度能力**：
  - 支持动态路径选择（如根据是否有图像决定是否走 vision encoder）
  - 支持 multi-stream join（如混合文本与图像输入时，需同步 N 条视觉路径后再触发生成）

- **消融实验（隐含）**
  - 控制流层的 Bucket 机制被证明是解决 diamond、sequence、multi-join 三种依赖的关键，若移除将导致竞态或死锁。
  - 分布式 KV Cache 的三级存储（GPU/CPU/SSD）有效缓解了长上下文场景下的内存压力，尤其适合超长对话历史保留。

## 4. 关键结论和发现

### 主要发现
1. **统一抽象优于专用优化**：
   - 当前多模态模型架构快速演进，针对单一结构的极致优化难以持续适用。Omni-Flow 的三层抽象（Control/Data/Compute Flow）提供了更强的适应性和可扩展性。

2. **KV Cache 必须成为一级公民**：
   - 在统一 backbone 架构（如 BAGEL、HunyuanImage-3）中，KV Cache 是跨理解与生成阶段的核心状态。将其纳入全局管理而非模型私有，才能真正实现资源共享。

3. **异构模型可以共享执行路径**：
   - DiT 模块可以通过继承 LLM 推理引擎的 base class，复用其 PagedAttention、KV Cache 管理、并行策略（TP/EP），大幅减少工程重复建设。

4. **零拷贝 + pull-on-demand 提升通信效率**：
   - 数据传输采用消费者主动拉取（pull-on-demand）而非生产者推送（push），结合 RDMA 实现跨节点零拷贝访问，显著降低冗余传输开销。

---

### 方法的局限性
- **目前仍处于早期阶段**：作者明确指出框架尚在初期，尚未全面覆盖所有模型形式。
- **性能尚未达到最优**：下一阶段目标是向 real-time inference 演进，说明当前延迟仍有优化空间。
- **依赖 Ray 和 Redis**：底层依赖 Ray 进行资源调度，Redis 作为服务发现中心，增加了系统复杂性。
- **AI 辅助开发带来的代码质量风险**：附录 C 提到大量代码由 AI 生成，存在局部 patch 积累、“技术债”增加的风险，需人工严格审查。

---

### 未来工作方向
1. **性能优化**：推进至 real-time inference，降低端到端延迟。
2. **深化分布式 KV Cache 支持**：
   - 支持更多 Attention 变体（如 GQA、MQA）
   - 智能选择 KV shards
   - 支持 CP（Context Parallelism）
3. **更复杂的调度策略**：
   - Cache-aware scheduling：基于各副本的 KV Hit 状态进行路由决策
4. **强化学习集成（RL）**：
   - 利用已有的统一权重管理基础设施，支持 RL 训练中的参数同步
5. **持续扩展模型覆盖范围**：支持更多新型 MLLM 架构与模态组合

---

> 🔗 **项目地址**：[https://github.com/meituan-longcat/omni-flow.git](https://github.com/meituan-longcat/omni-flow.git)

</details>

---

### 3. [AgRefactor: Self-Evolving Agentic Workflow for HLS Compatibility and Performance](https://arxiv.org/abs/2606.30949)

**Authors**: Yang Zou, Zijian Ding, Yizhou Sun, Jason Cong  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.30949v1  

#### Abstract
High-Level Synthesis (HLS) provides a fast path from concepts to silicon, but converting real-world software into synthesizable HLS code remains challenging due to restrictive language support and the gap between software and hardware programming practices. Existing automated and LLM-based refactori...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AgRefactor: Self-Evolving Agentic Workflow for HLS Compatibility and Performance

---

## 1. 论文的主要贡献和创新点

### 解决的问题
High-Level Synthesis (HLS) 能够加速从软件到硬件的设计流程，但将现实世界中的复杂 C/C++ 软件重构为 **HLS-可综合代码** 仍面临巨大挑战。主要原因包括：
- **语言支持受限**：HLS 工具不支持现代 C++ 特性（如 `std::vector`, `lambda`, 动态内存分配等）。
- **人工成本高**：专家需耗费数天进行手动重构（如高能物理项目中报告）。
- **现有自动化方法泛化能力差**：基于规则的工具（如 HeteroRefactor）难以处理外部库、复杂指针操作等问题。
- **LLM 方法成本高且不稳定**：纯 LLM 驱动的方法计算开销大、输出方差高、缺乏长期记忆。

---

### 提出的新方法与创新思路

作者提出 **AgRefactor** —— 一种基于 LLM 的多智能体（multi-agent）、自演进（self-evolving）的工作流系统，用于自动将真实软件重构为高性能、HLS 兼容的硬件实现。

#### 主要创新点：

1. ✅ **自演进的智能体记忆系统（Self-Evolving Agentic Memory）**
   - 在多个任务间持续积累“事实性知识”（factual knowledge）和“策略性知识”（strategic knowledge）。
   - 设计专用的知识表示格式、交互模式和距离度量函数（结合代码嵌入与非合成项嵌入），实现跨任务知识迁移。
   - 支持 Planner 和 Identifier 智能体在面对新程序时调用历史经验，提升鲁棒性和效率。

2. ✅ **工具增强型混合流水线（Tool-Enhanced Hybrid Pipeline）**
   - 将 LLM 智能体与现有的算法级重构工具 **HeteroRefactor (HeteroRF)** 结合。
   - 引入 **Tool Specialist Agent** 对原始代码做轻量预处理（如移除不兼容头文件、改写指针表达式），使其适配 HeteroRF 输入要求。
   - 实现“LLM 处理难例 + 工具处理通例”的协同机制，显著降低 LLM 调用频率和推理成本。

3. ✅ **端到端性能优化智能体（Performance Optimization Agent）**
   - 在完成语法兼容性重构后，进一步执行性能优化。
   - 利用 EDA 报告（如调度延迟、关键路径分析）指导结构级重写（structural refactoring），而非仅依赖 pragma 插入。
   - 内置树状工作记忆（tree-based working memory）跟踪探索过程，防止陷入局部最优。

---

### 相比现有方法的优势

| 维度 | Rule-Based 方法（如 HeteroRF） | LLM-Based 方法（如 HLSRewriter） | **AgRefactor（本文）** |
|------|-------------------------------|----------------------------------|------------------------|
| **通用性 (Generalizability)** | ❌ 差（无法处理 STL、复杂指针） | ⚠️ 中等（依赖提示工程） | ✅ 好（通过记忆自适应演化） |
| **可扩展性 (Scalability)** | ✅ 好（速度快） | ❌ 差（LLM 成本高） | ✅ 好（工具优先 + LLM 协助） |
| **成本 (Cost)** | ✅ 低 | ❌ 高 | ⚠️ 中等（但远低于纯 LLM） |
| **性能优化能力** | ❌ 无 | ⚠️ 有限（随机 pragma 插入） | ✅ 强（结构重写 + 分层优化） |

> 如表 I 所示，AgRefactor 在三项关键指标上实现了均衡优势。

---

## 2. 核心实验方法和设置

### 数据集
构建了一个综合性强的真实世界基准套件，共 **45 个测试案例**，涵盖以下来源：

| 类别 | 示例 | 代码行数范围 |
|------|------|-------------|
| LeetCode | `maxSlidingWindow`, `solveSudoku` | 21–67 |
| HLSRewriter [12] | `aes_encrypt`, `qrd_compute` | 10–115 |
| C2HLSC [10] | `sha256_update`, `quicksort` | 11–221 |
| HeteroRF [8] | `strassen`, `mergesort` | 65–304 |
| Libsodium [22] | `chacha20_stream`, `blake2b_compress` | 126–370 |
| Minimap2 [23] | `mm_chain_dp_orig` | 473 |
| Libjpeg-turbo [24] | `encode_one_block`, `idct_generic` | 279–754 |
| AV1 Codec [25] | `av1_apply_temporal_filter` | 407–1266 |

> ⚠️ 多数案例长度是先前研究中最复杂案例的 **5–10 倍**，更具挑战性。

---

### 实验设置与评估指标

#### 主要评估目标：
- **重构成功率（Success Rate）**：能否生成可通过 HLS 综合并功能正确的代码。
- **平均重试次数（Average Retries）**：反映稳定性与收敛速度。
- **性能增益（Speedup）**：优化后的设计相对于基线的延迟改进。
- **资源开销（Resource Overhead）**：额外使用的 BRAM/LUT/DSP 数量。

#### 测试环境：
- 使用 **Vitis HLS** 作为综合工具。
- 功能验证通过 **C-based simulation** 完成。
- 性能估算采用快速仿真器 **LightningSim**。
- LLM 后端使用 **GPT-5-mini / GPT-5**（模拟未来模型能力）。
- 记忆检索使用 **Sentence-Transformer** 的 `all-MiniLM-L6-v2` 模型生成嵌入。

#### 基线方法对比：
1. **HeteroRefactor (HeteroRF)**：当前最先进的开源自动化重构工具。
2. **HLSRewriter-style baseline**：在本框架下重建的 LLM 方法，使用静态 RAG（Vitis 编程指南）代替动态记忆。
3. **AutoDSE [20]**：领先的 pragma 自动调优工具，用于性能比较。

---

## 3. 主要实验结果和性能指标

### 重构性能对比（Table III）

在 **11 个最具挑战性的 benchmark** 上运行 20 次，统计成功次数：

| Task | HeteroRF | HLSRewriter | **AgRefactor** |
|------|----------|-------------|----------------|
| `av1_compound_type_rd` | 0 | 0 | **1** |
| `encode_one_block` | 0 | 2 | **10** |
| `idct_generic` | 0 | 2 | **5** |
| `median_cut` | 0 | 20 | 18 |
| `argon2_fill_segment` | 0 | 17 | 13 |
| `mm_chain_dp_orig` | 0 | 5 | **6** |
| `ahocorasick` | 20 | 16 | **20** |
| `dfs` | 20 | 16 | **20** |
| `strassen` | 20 | 17 | **20** |
| `wordbreak` | 0 | 17 | **20** |
| `skyline` | 0 | 20 | **20** |

✅ **结论**：在 **9/11 个 benchmark 上优于或持平于所有基线**。

---

### 消融实验结果

#### （1）消融记忆机制（Table IV & V）

| 设置 | 成功率提升情况 |
|------|----------------|
| **无记忆（no-mem） → 有记忆（with-mem）** |  
| 使用 GPT-5-mini：10/11 提升，最高达 +7 成功率（`encode_one_block`）  
| 使用 GPT-5：全部提升，`encode_one_block` 从 2→9（+350%）  

📌 **记忆显著提高成功率并减少重试次数**，说明历史经验有效增强了 LLM 的鲁棒性。

#### （2）消融工具集成（Table VII）

| Task | HeteroRF Alone | + Tool Specialist |
|------|----------------|--------------------|
| `encode_one_block` | 0 | **10** |
| `idct_generic` | 0 | **5** |
| `argon2_fill_segment` | 0 | **8** |

✅ **Tool Specialist 可将原本不可处理的程序转化为 HeteroRF 可接受的形式**，大幅提升覆盖率。

---

### 性能优化结果（Table VIII）

对已重构的设计进行 2 小时优化，对比 **AutoDSE** 与 **AgRefactor Optimizer Agent**：

| Benchmark | AutoDSE | AgRefactor | Speedup |
|-----------|---------|------------|---------|
| `skyline` | 1.00× | 295.44× | 🔺 **295×** |
| `word_break` | 1.00× | 28.71× | 🔺 29× |
| `idct_generic` | 1.00× | 9.95× | 🔺 10× |
| `encode_one_block` | 1.00× | 6.30× | 🔺 6.3× |
| **Geometric Mean** | — | — | ✅ **6.51×** |

🎯 **AgRefactor 优化器实现 6.51× 几何平均加速**，远超仅靠 pragma 调优的 AutoDSE。

此外，在 51 个开源 HLS 设计上的测试表明：
- 即使不允许增加资源（0% budget），也能获得 **1.14× 平均加速**。
- 在 20% 额外资源预算下，达到 **1.20× 加速**。
- 最高达 **1.41×（100% 资源预算）**。

---

### 运行时间与成本（Section V-E）

| 配置 | 平均运行时间 | 相对效率 |
|------|--------------|----------|
| AgRefactor（无工具） | ~10 分钟 | 比手动快 >10× |
| AgRefactor（含工具） | **~4 分钟** | 更快且成本更低（减少 LLM 调用） |
| HLSRewriter | ~9 分钟 | AgRefactor 仅慢 9% |

✅ **工具集成显著提升效率与性价比**。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **自演进记忆机制能有效提升 LLM 在 HLS 重构中的泛化能力和稳定性**。
   - 通过跨任务积累“失败教训”和“成功策略”，智能体可在新任务中快速借鉴经验。
   - 特别适用于具有相似非综合构造的程序族（如图像编解码器模块）。

2. ✅ **工具与 LLM 的协同是解决大规模重构问题的关键路径**。
   - 不应全盘依赖 LLM 或完全抛弃 LLM。
   - “LLM 做决策 + 工具做执行” 是高效、低成本的折中方案。

3. ✅ **结构性重写（Structural Refactoring）比 pragma 调优更能决定最终性能上限**。
   - 如将 O(n) 排序替换为 Radix Sort、重构 DFS 栈结构等，带来百倍级加速。
   - 纯搜索式 pragma 调参（如 AutoDSE）存在天花板。

4. ✅ **AgRefactor 是首个实现端到端全自动、开源的 HLS 重构与优化系统**。
   - 从原始 C++ 到高性能 RTL，全程无需人工干预。
   - 已公开代码：[GitHub - Williamzou0123/AgRefactor](https://github.com/Williamzou0123/AgRefactor)

---

### 局限性

1. ❗ **对极长程序（>1000 行）的记忆迁移效果下降**。
   - 如 `av1_compound_type_rd` 因分布偏移大，历史策略难以复用。

2. ❗ **测试平台生成仍存在一定宽松性（lenient testbenches）**。
   - 单次生成的 testbench 可能漏检错误（Appendix C 显示约 60% 的“通过”实为误判）。
   - 当前解决方案引入工程师评审循环（engineer-rater loop）以增强覆盖率。

3. ❗ **依赖高质量 LLM 输出**。
   - 若底层模型能力不足（如小模型），即使有记忆也难以纠正根本性错误。

---

### 未来工作方向

1. 🔮 **让 LLM 智能体直接修改或扩展工具本身的能力**（如 patch HeteroRF 规则引擎）。
2. 🔮 **构建更强大的跨项目测试平台生成机制**，支持回归测试与边界条件覆盖。
3. 🔮 **将 AgRefactor 应用于更多领域**（如 AI 编译器前端、CUDA-to-HLS 转换）。
4. 🔮 **探索多模态反馈学习**（结合波形、功耗、面积报告）进行闭环优化。

---

> ✅ **总结一句话**：  
> **AgRefactor 通过“自演进记忆 + 工具协同 + 分层优化”的三重机制，在真实性、效率、性能三个维度全面超越现有 HLS 重构方法，推动了软件到硬件自动化转换的实用化进程。**

</details>

---

### 4. [BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based Speculative Decoding](https://arxiv.org/abs/2606.31315)

**Authors**: Hao Zhang, Yiming Hu, Yong Wang, Mingqiao Mo, Xin Xiao, Xiangxiang Chu  
**Category**: cs.CL  
**Published**: 2026-07-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.31315v1  

#### Abstract
Speculative decoding accelerates inference by using a lightweight draft model to generate candidate tokens in parallel, and are then verified by the target model, enabling lossless acceleration. Recently, diffusion-based speculative decoding further improves parallelism by generating multiple tokens...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **diffusion-based speculative decoding** 方法（如 DFlash）在推理时采用固定的 **block size**，即训练阶段设定的块大小直接用于所有输入样本。然而，作者指出这一假设是次优的：不同输入样本的最佳 block size 存在显著差异，统一使用固定 block size 忽略了输入级别的动态特性，限制了并行生成效率的最大化。

此外，过大的 block size 可能导致错误累积，降低接受长度（acceptance length），而过小则无法充分利用硬件并行能力。因此，如何为每个样本自适应地选择最优 block size 成为一个关键挑战。

### 提出了什么新方法或新思路
作者提出 **BlockPilot** —— 一种实例自适应的策略学习框架，用于在 diffusion-based speculative decoding 中动态选择最优 block size。

#### 核心思想：
- 观察到最优 block size 在样本间虽有变化，但具有**强局部性**：大多数样本的最优 block size 集中在训练 block size $ B $ 附近的小范围内（如 $[B-3, B+3]$）。
- 将 block size 选择建模为一个轻量级的**分类任务**，而非复杂的在线搜索。
- 利用目标模型在 **prefilling 阶段末尾 token 的预测分布** $ p(x) $ 作为上下文状态表示，训练一个小型 MLP 网络来预测该样本对应的最优 block size。

#### 方法特点：
- **仅需一次前向计算**：在 prefilling 完成后执行一次预测，之后整个 decoding 过程使用该 block size。
- **无侵入式集成**：不修改 draft model、target model 或验证流程，可无缝嵌入现有 speculative decoding 框架（如 DFlash）。
- **低开销**：预测器参数量仅为 0.32B，延迟约 7.34ms，远低于主干模型。

### 相比现有方法的优势
| 维度 | 现有方法（如 DFlash） | BlockPilot |
|------|------------------------|-----------|
| Block Size 策略 | 固定全局 block size | 实例自适应（instance-adaptive） |
| 决策机制 | 手动调参 / 枚举测试 | 基于 prefilling 分布的学习型策略 |
| 效率潜力 | 局限于单一配置 | 充分挖掘样本级加速空间 |
| 部署复杂度 | 简单但次优 | 轻量插件式，几乎零额外成本 |

> ✅ **优势总结**：BlockPilot 在极低额外开销下实现了对 diffusion-based speculative decoding 的“智能调度”，突破了传统固定策略的性能瓶颈。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **数学推理**：GSM8K、MATH-500、AIME24
- **代码生成**：HumanEval、MBPP、SWE-Bench
- **对话生成**：MT-Bench
- **训练数据构造**：ShareGPT、WSC、COPA

这些数据覆盖了多种任务类型和领域，确保方法泛化性。

### 实验设置和评估指标

#### 模型基准
在以下四个主流 LLM 上进行实验：
- Qwen3-4B
- Qwen3-8B
- Llama-3.1-8B-Instruct
- Qwen3-Coder-30B-A3B

#### 评估指标
由于 speculative decoding 不改变输出分布，**生成质量保持不变**，故聚焦于**推理效率**：
- **Average Acceptance Length $ \mathbf{T} $**：每轮 speculative decoding 平均接受的 draft token 数量。
- **Speedup Ratio**：相对于标准 autoregressive decoding 的端到端推理加速比。

#### 基线方法对比
| 方法 | 类型 |
|------|------|
| Standard Autoregressive | 基线 |
| EAGLE-3 | 自回归 draft model 的 SOTA 方法 |
| DFlash(n) | 当前 diffusion-based speculative decoding 的 SOTA，n 表示 block size（4, 8, 16, 32） |

> 注：DFlash(16) 是当前最强 baseline。

#### 实现细节
- 使用 PyTorch 和 HuggingFace Transformers
- GPU：NVIDIA H100 80GB
- 预测器结构：2层 MLP，隐藏维度 2048
- 超参数 $ k=2 $，候选 block size 范围为 $[B-2, B+2]$
- 训练 100 epochs，Adam 优化器，学习率 1e-5

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & Table 6）

| 模型 | 方法 | Temperature | Speedup | Acceptance Length $ T $ |
|------|------|-------------|---------|----------------------------|
| Qwen3-4B | DFlash(16) | 0 | 3.99× | 6.31 |
| Qwen3-4B | **BlockPilot** | 0 | **4.17×** | **6.59** |
| Qwen3-4B | DFlash(16) | 1 | 3.80× | 5.35 |
| Qwen3-4B | **BlockPilot** | 1 | **4.20×** | **5.92** |
| Qwen3-8B | DFlash(16) | 0 | 4.42× | 6.13 |
| Qwen3-8B | **BlockPilot** | 0 | **4.66×** | **6.46** |
| Qwen3-8B | DFlash(16) | 1 | 3.55× | 5.00 |
| Qwen3-8B | **BlockPilot** | 1 | **3.94×** | **5.55** |
| Llama-3.1-8B-Instruct | DFlash(16) | 0 | 3.03× | 4.48 |
| Llama-3.1-8B-Instruct | **BlockPilot** | 0 | **3.25×** | **4.82** |
| Qwen3-Coder-30B | DFlash(16) | 0 | 3.86× | 5.68 |
| Qwen3-Coder-30B | **BlockPilot** | 0 | **4.12×** | **6.05** |

> 📌 **最高性能**：在 Qwen3-4B 上实现 **4.20× 加速** 和 **5.92 的 acceptance length**，显著优于所有 baseline。

### 与基线方法的对比结果
- BlockPilot 在所有模型、温度设置和任务类别上均取得**最佳性能**。
- 相比最强固定 block size 方法（DFlash(16)），BlockPilot 普遍提升 **0.2–0.4× 的加速比** 和 **0.3–0.5 的 acceptance length**。
- 即使在高随机性设置（temperature=1）下仍表现稳健，说明其对不确定性具有鲁棒性。
- 图2显示，BlockPilot 在多个模型上的加速曲线全面领先。

### 消融实验结果（Ablation Studies）

#### （1）预测器结构影响（Table 3）
| 配置 | GSM8K Speedup | T |
|------|---------------|----|
| Hidden Size=1024 | 4.73× | 6.86 |
| Hidden Size=2048 | **4.76×** | **6.90** |
| Hidden Size=4096 | 4.76× | 6.90 |
| Depth=1 | 4.71× | 6.83 |
| Depth=2 (**default**) | **4.76×** | **6.90** |
| Depth=3 | 4.65× | 6.74 |

✅ 结论：两层、宽度 2048 的 MLP 已足够，更深更宽不会带来收益。

#### （2）候选区间半径 $ k $ 影响（Table 4）
| $ k $ | GSM8K Speedup | T |
|-------|----------------|----|
| 1 | 4.67× | 6.77 |
| 2 (**default**) | **4.76×** | **6.90** |
| 3 | 4.64× | 6.73 |

✅ 结论：$ k=2 $ 最佳；过大反而增加预测难度，性能下降。

#### （3）输入特征设计
- 使用完整预测分布（full predictive distribution）效果最好。
- 若只取 Top-k 概率，会导致严重过拟合（训练准确率 80%，测试仅 10%）。

---

## 4. 关键结论和发现

### 主要发现
1. **最优 block size 是样本依赖的**：并非所有输入都适合相同的 block size，固定策略存在明显性能损失。
2. **最优 block size 具有强局部性**：绝大多数样本的最优值集中在训练 block size 附近（如 $[B-3, B+3]$），这使得问题可转化为一个小规模分类任务。
3. **prefilling 阶段的最后 token 预测分布是有效信号**：它聚合了全局上下文信息，能够反映未来生成的稳定性，适合作为 block size 决策依据。
4. **轻量级策略网络即可实现高效决策**：无需复杂架构，一个小型 MLP 就能在极低成本下实现显著加速增益。
5. **方法通用且即插即用**：适用于不同规模、不同类型的目标模型，在 math/code/chat 多种任务上均有效。

### 方法的局限性
- **训练数据构建成本较高**：为获取每个样本的真实最优 block size，需离线枚举多个 block size 下的 speculative decoding 性能。例如对 32B 模型，单个样本需 ~25 秒（k=2 时 5 次运行）。
- **依赖预训练 draft model 的 block size 对齐**：若 inference block size 偏离训练配置太远，proposal 质量可能下降（理论分析见 Appendix A）。
- **未考虑动态调整**：当前策略在整个 decoding 过程中使用同一个 block size，未来可探索 step-level 动态调整。

### 未来工作方向
- 更高效的训练数据构造方式：
  - 启发式搜索（heuristic search）
  - 自适应剪枝（adaptive pruning）
  - 早期停止（early stopping）
  - 使用代理指标（proxy metrics）快速估计 block size 效果
- 探索 coarse-to-fine 的两级搜索策略
- 跨样本结果复用以减少冗余计算
- 扩展至 step-level 或 layer-level 的动态 block size 控制

---

> 🔚 **总结一句话**：  
> **BlockPilot 揭示了“解码策略”本身是一个值得学习的变量，并通过一个轻量、实例自适应的 block size 预测器，在几乎零代价下实现了 diffusion-based speculative decoding 的新一轮效率跃迁。**

</details>

---

### 5. [SeKV: Resolution-Adaptive KV Cache with Hierarchical Semantic Memory for Long-Context LLM Inference](https://arxiv.org/abs/2606.31145)

**Authors**: Amirhossein Abaskohi, Giuseppe Carenini, Peter West, Yuhang He  
**Category**: cs.CL  
**Published**: 2026-07-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.31145v1  

#### Abstract
Large language models increasingly operate over long contexts, where the KV cache becomes a dominant memory bottleneck: its size grows linearly with sequence length and must be retained throughout decoding, making full GPU caching prohibitively expensive without compression. Existing KV cache compre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SeKV: Resolution-Adaptive KV Cache with Hierarchical Semantic Memory for Long-Context LLM Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在长上下文（long-context）场景下，**KV Cache** 成为大语言模型（LLM）推理过程中的主要内存瓶颈。其大小随序列长度线性增长，且必须在整个解码过程中保留，导致全量缓存难以在GPU上部署。现有的压缩方法如**Token Eviction**（丢弃不重要token）或**Semantic Grouping**（语义分组）存在以下问题：
- **信息不可逆丢失**：一旦压缩或丢弃，后续无法恢复细节。
- **静态决策**：压缩策略在prefill阶段即固定，无法动态响应生成过程中的查询需求。

### **提出了什么新方法或新思路**
提出 **SeKV**（Semantic KV Cache），一种**分辨率自适应的语义KV缓存机制**，核心思想是：
- 将KV Cache组织为**层次化语义记忆**（Hierarchical Semantic Memory）。
- 支持**多分辨率存储**与**按需动态放大**（zoom-in）相关语义片段。

#### **三大创新组件**：
1. **熵引导的语义分段（Entropy-Guided Span Segmentation）**  
   利用token的**surprisal**（负对数概率）作为边界信号，自动识别语义边界（如话题切换、实体引入），形成语义连贯的span。高surprisal的**anchor token**保留在GPU上用于路由和重建。

2. **双分辨率表示与GPU-CPU内存层级**  
   - **GPU端**：每个span保存一个轻量级**summary vector**，用于粗粒度路由。
   - **CPU端**：保存该span的**低秩SVD基底**（low-rank SVD basis），支持按需重建token级KV状态。

3. **训练驱动的动态“放大”机制（Trained Zoom-In Mechanism）**  
   引入可学习的**per-head per-layer路由投影**和**阈值**，在每步解码中判断是否需要将某个span从CPU拉取并重建至GPU。仅当该span被判定为query相关时才触发重建。

### **相比现有方法的优势**
- **避免信息永久丢失**：不进行token eviction，所有信息均可通过SVD基底重建。
- **动态适应查询**：压缩决策非静态，能根据当前query动态恢复关键细节。
- **高效内存利用**：GPU内存几乎不随上下文长度增长，实现**近恒定内存开销**。
- **兼容性强**：保持基础LLM冻结，仅添加少于0.05%的可训练参数，易于部署。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个主流长上下文基准上评估：
| 数据集 | 任务类型 | 上下文长度 | 特点 |
|--------|--------|----------|------|
| **LongBench** | 多任务长文档理解 | 1K–18K | 包含QA、摘要、代码补全等 |
| **RULER** | 控制性检索与推理 | 4K–128K | 可配置长度，测试深层信息利用能力 |
| **InfiniteBench** | 超长上下文任务 | ~100K | 测试极限长度下的表现 |
| **Needle-in-a-Haystack (NIAH)** | 针对性事实检索 | up to 128K | 插入“针”并测试能否准确检索 |
| **GSM8K (50-shot)** | 多示例数学推理 | ~13K–15K | 测试长prompt下的推理模式保持 |

### **实验设置和评估指标**
- **骨干模型**：LLAMA-3系列（3B/8B）、Mistral-7B、Qwen2.5-14B。
- **GPU内存预算**：压缩方法统一使用**全量KV缓存的10%**作为GPU驻留内存上限。
- **评估指标**：
  - LongBench：平均F1
  - RULER / InfiniteBench / NIAH：准确率（Accuracy）
  - GSM8K：Exact Match

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Token-Level Compression** | StreamingLLM, H2O, SnapKV, PyramidKV |
| **Semantic/Chunk-Level Compression** | ChunkKV, SemantiCache, SentenceKV |
| **无压缩参考** | Full KV |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
在**10% GPU KV预算**下，SeKV在所有基准和模型上均取得最佳表现：

| 方法 | LongBench ↑ | RULER ↑ | InfiniteBench ↑ | NIAH ↑ | 平均提升 vs SentenceKV |
|------|------------|---------|----------------|--------|------------------------|
| **Full KV** | 68.38 | 73.67 | 72.39 | 95.28 | — |
| **SentenceKV (最强基线)** | 57.84 | 60.43 | 60.23 | 84.83 | — |
| **SeKV (Ours)** | **61.23** | **63.84** | **65.74** | **91.17** | **+5.9%** |

- 在**NIAH**上提升 **+6.34%**（84.83 → 91.17），表明其在**稀疏证据检索**方面显著优于现有方法。
- 在**RULER**上提升 **+3.41%**，显示其在复杂推理任务中更有效利用深层上下文。

### **与基线方法的对比结果**
- **相比Full KV**：仅使用10%内存，性能差距极小（平均仅落后约0.8–1.2点），尤其在LongBench和InfiniteBench上接近无损。
- **相比SentenceKV**：在所有5个backbone上均胜出，最大提升达+3.83（Qwen2.5-14B）。
- **效率优势**：
  - GPU内存减少 **53.3%**（vs Full KV @ 128K）。
  - 内存增长近乎平坦：从8K到128K上下文，GPU内存仅从31.2GB增至34.9GB（Full KV从36.0GB→74.8GB）。

### **消融实验结果**
在Qwen2.5-14B上进行消融研究（10%预算）：

| 变体 | LongBench | RULER | NIAH |
|------|----------|-------|------|
| **完整SeKV** | 54.71 | 87.34 | 91.17 |
| w/o SVD重建 | 51.84 | 80.92 | 83.47 |
| w/o 训练式zoom-in | 52.76 | 82.73 | 85.96 |
| w/o 熵引导分段 | 53.02 | 83.61 | 86.42 |
| w/o anchor tokens | 53.47 | 84.28 | 87.31 |

- **移除SVD重建或zoom-in机制**导致最大性能下降，验证了**动态重建**是核心。
- **熵引导分段**优于固定chunking，在RULER/NIAH上尤为明显。
- **anchor tokens**有助于提高重建精度。

### **KV预算敏感性**
| 预算 | SentenceKV | SeKV | 提升 |
|------|-----------|------|------|
| 5% | 78.62 | **86.31** | +7.69 |
| 10% | 84.83 | **91.17** | +6.34 |
| 20% | 88.21 | **92.46** | +4.25 |

- SeKV在**低预算下优势更大**，说明其在极端内存受限场景更具竞争力。

---

## **4. 关键结论和发现**

### **主要发现**
1. **静态压缩不足以应对长上下文推理**：现有方法因固定压缩策略，无法恢复后期变得重要的信息。
2. **动态分辨率恢复至关重要**：SeKV通过“**先粗后细**”的策略，在保持高效的同时实现精准检索。
3. **层次化内存设计可行**：将summary放GPU、SVD基底放CPU，结合异步预取，可在不牺牲性能的前提下大幅降低GPU内存。
4. **zoom-in行为稀疏且集中**：仅少数head（尤其是中后层）频繁触发重建，支持per-head阈值设计。

### **方法的局限性**
1. **依赖语义分段质量**：若surprisal估计不准或文本结构复杂（如代码、表格），可能导致次优分段。
2. **超参数敏感性**：如surprisal阈值α、SVD最大秩R等需调优，跨模型迁移可能需重新验证。
3. **依赖CPU-GPU带宽**：在低带宽环境下，频繁的CPU-to-GPU传输可能成为瓶颈。
4. **非完全无损**：SVD为低秩近似，虽可恢复大部分信息，但仍为有损压缩。

### **未来工作方向**
- 探索**联合训练**路由模块与更强的span表示（如可学习聚合）。
- 设计**自适应重构策略**，根据历史注意力动态调整重建粒度。
- 扩展至**多模态上下文**，支持图像、表格等非文本元素的语义分段与重建。
- 优化**跨设备通信调度**，进一步降低传输延迟影响。

---

> **总结**：SeKV提出了一种全新的**分辨率自适应KV缓存范式**，通过**语义分段 + 层次存储 + 动态放大**，在几乎不修改原模型的情况下，实现了**高性能、低内存、可恢复**的长上下文推理，为未来LLM的高效部署提供了重要思路。

</details>

---

### 6. [Agentic-Ideation: Sample Efficient Agentic Trajectories Synthesis for Scientific Ideation Agents](https://arxiv.org/abs/2606.31229)

**Authors**: Keyu Zhao, Lingyan Kong, Fengli Xu, Yong Li  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.31229v1  

#### Abstract
Ideation plays a pivotal role in scientific discovery. Recent LLM, especially AI Scientist systems, show promising potential for automated ideation. However, existing approaches predominantly rely on pre-defined agentic workflows. This constraint severely limits the flexibility required to navigate ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Agentic-Ideation: Sample Efficient Agentic Trajectories Synthesis for Scientific Ideation Agents》核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前的 **AI Scientist** 系统在科学创意生成（scientific ideation）任务中主要依赖预定义的 **workflow-based** 流程，例如“检索文献 → 填写模板”。这类方法存在以下严重缺陷：

- **灵活性差**：受限于固定流程，无法自适应地选择工具或调整推理路径。
- **样本效率极低**：传统的 **agentic data synthesis** 方法（如 rejection sampling）在开放域的 ideation 任务中效率低下，因为缺乏明确的“正确答案”作为筛选标准，导致大量无效尝试。

因此，如何高效合成高质量的 **agentic trajectory** 数据，并训练出具备自主推理能力的 **Scientific Ideation Agent**，是本文要解决的核心挑战。

---

### **提出了什么新方法或新思路**

作者提出 **Agentic-Ideation** 框架，包含两个核心组件：

#### （1）**混合工具空间（Hybrid Tool Space）**
定义了一个结合 **外部工具（External Tools）** 和 **认知工具（Thinking Tools）** 的行动空间：
- **External Tools**：
  - `Search`：关键词搜索获取详细论文元数据。
  - `Get_References`：获取某篇论文的参考文献列表（仅标题）。
  - `Get_Cited`：获取引用该论文的后续研究列表（仅标题）。
- **Thinking Tools**：
  - `Analyse_Gap`：分析已有信息中的研究空白（research void）。
  - `Ideation`：基于研究空白生成新想法。
  - `Reflection`：自我反思并评估想法质量，决定是否迭代。

这一设计使 agent 能够进行闭环的认知循环：**检索 → 分析 → 创造 → 反思**。

#### （2）**Oracle-Guided Data Synthesis**
为解决开放域中轨迹合成效率低的问题，引入 **Reference Idea** 作为“导航信标”（oracle），构建一个 **multi-agent system**（Planner + Controller + Tool Agents）来生成轨迹。

- **Dual-Thought Mechanism**：
  - **Private Thought**：利用 oracle 指导，计算最优动作路径（有“先见之明”）。
  - **Public Thought**：仅基于当前上下文推理，模拟真实 agent 的自主探索过程。
- 最终只保留 **Public Thought** 构成训练数据，确保模型学习的是自主决策逻辑，而非依赖外部反馈。

#### （3）**Agentic Supervised Fine-Tuning with Masking**
在训练阶段采用 **masking strategy**：
- 在损失函数中屏蔽 `<Result>...</Result>` 中的工具返回值。
- 强制模型关注 **reasoning logic** 和 **tool invocation decision**，而非记忆特定输出。

---

### **相比现有方法的优势**

| 维度 | 优势 |
|------|------|
| **灵活性** | 超越静态 workflow，支持动态、自适应的探索路径。 |
| **样本效率** | 相比 rejection sampling 提升 **>10×**，从平均 12 次尝试才能得到一个有效轨迹，降至 **1 次即成功**。 |
| **生成质量** | 生成的想法在新颖性、重要性和可行性上均显著优于基线。 |
| **泛化能力** | 通过 masking 外部反馈，增强模型鲁棒性和泛化性。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- 从三大顶会（ICLR 2025, ICML 2025, NeurIPS 2025）中收集已接收论文作为 **Anchor Papers**。
- 对每篇 Anchor Paper 随机选取其参考文献中的 **10 篇作为输入上下文**。
- 将论文分类为三个领域：**NLP、CV、Others**。
- **测试集**：NeurIPS 2025 的 300 篇论文（每类 100 篇），用于评估前沿 idea 生成能力。

---

### **实验设置和评估指标**

#### **评估指标（Likert Scale: 1–10）**
- **Novelty**：想法的创新性和独特性。
- **Significance**：对学术社区的潜在影响力。
- **Feasibility**：技术实现的可行性。
- **Overall**：综合质量评分。

#### **评估方式**
- **自动化评估**：采用“Panel of LLM Judges”，由 GPT-5.2、Gemini-3-Pro-Preview、Claude-Sonnet-4.5、Qwen3-Max 四个模型打分，取平均。
- **人工评估**：双盲协议下由专家对匿名想法打分，验证实际价值。

#### **训练细节**
- **合成阶段**：使用 Qwen3-235B-A22B-Instruct-2507 作为 oracle agent。
- **训练阶段**：以 Qwen3-8B 为 backbone，在 **4,646 条合成轨迹** 上进行全量微调（full fine-tuning）。
- **硬件**：4×NVIDIA A100 (80GB)，训练时间约 10.5 小时。

---

### **基线方法对比**

| 方法 | 类型 | 描述 |
|------|------|------|
| **ResearchAgent** | Workflow-based | 基于知识图谱和模拟同行评审迭代优化想法。 |
| **SciPIP** | Dual-path | 结合检索与生成，平衡新颖性与可行性。 |
| **SciMON** | Novelty-focused | 显式优化新颖性，避免与已有工作重复。 |
| **VirSci** | Multi-agent | 多 agent 协作讨论生成与评估想法。 |
| **Qwen3-8B** | Zero-shot baseline | 不进行任何 fine-tuning，直接提示生成。 |

所有基线均使用 GPT-4o 作为底层引擎，构成强闭源基准。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（Table 1）**

| 方法 | NLP Overall | CV Overall | Others Overall | **Average Overall** |
|------|-------------|------------|----------------|--------------------|
| ResearchAgent | 6.01 | 5.94 | 5.86 | 5.94 |
| SciPIP | 5.97 | 6.02 | 5.96 | 5.98 |
| SciMON | 5.87 | 5.99 | 5.96 | 5.94 |
| VirSci | 5.93 | 6.01 | 5.81 | 5.92 |
| Qwen3-8B | 5.90 | 5.85 | 6.10 | 5.95 |
| **Agentic-Ideation** | **6.64*** | **6.69*** | **6.69*** | **6.67*** |

> ✅ **提升幅度**：相比 SOTA 基线（SciPIP），**Overall 提升 11.91%**。

#### 各维度增益（vs 第二名）：
- **Novelty**: +11.40%
- **Significance**: +11.14%
- **Feasibility**: +8.70%

---

### **与基线方法的对比结果**

- **超越 workflow-based 方法**：Agentic-Ideation 具备动态回溯、反思和工具调度能力，能跳出局部最优，生成更具突破性的想法。
- **优于零样本 backbone**：说明 **active tool use** 和 **grounded reasoning** 是提升可行性的关键。
- **人类评估一致**（Table 4）：
  - Human Overall Score: **6.45** vs SciPIP 的 5.87，**提升 9.88%**。
  - 在 **Significance (+12.28%)** 和 **Feasibility (+8.62%)** 上优势明显。

---

### **消融实验结果（Table 3）**

移除任一工具均导致性能下降，证明各工具协同作用至关重要：

| 工具移除 | Overall 下降 | 关键影响 |
|--------|--------------|---------|
| `w/o Search` | ↓ 0.68 | 缺乏主动信息扩展，知识受限 |
| `w/o Analyse_Gap` | ↓ 0.60 | 新颖性大幅下降，难以识别研究空白 |
| `w/o Reflection` | ↓ 0.24 | 无法拒绝平庸想法，需人工干预 |
| `w/o Get_Cited` | ↓ 0.29 | 难以追踪领域发展趋势 |

> 🔍 特别指出：`Analyse_Gap` 对 **Novelty** 至关重要；`Search` 是保障 **Feasibility** 的基础。

---

## 4. 关键结论和发现

### **主要发现**

1. **Agentic 架构优于 Workflow-based 方法**  
   自主 agent 能够灵活组合工具、动态调整策略，更适合复杂、开放的科学创意生成任务。

2. **Oracle-Guided Synthesis 极大提升样本效率**  
   利用 reference idea 作为导航信号，将随机探索转变为 **directed trajectory reconstruction**，实现 **>10× 的数据合成加速**。

3. **Masking 外部反馈提升模型鲁棒性**  
   防止模型“作弊”式记忆工具返回值，迫使其学习真正的决策逻辑。

4. **Reflection 是高质量输出的关键机制**  
   自我批判与迭代能力使得 agent 能主动拒绝低质量提案（如 Figure 4 中 reject “Latent-Space Remasking”），最终产出更创新的结果。

---

### **方法的局限性**

1. **Backbone 模型规模限制**  
   当前使用 Qwen3-8B，虽高效但推理深度和知识广度有限。更大模型可能进一步提升 idea 复杂度。

2. **工具空间偏重检索，缺乏执行能力**  
   当前工具均为 **read-only**（如 Search、Get_Cited），缺少 **write/executable tools**（如 Python code interpreter、simulator），无法进行初步实验验证。

---

### **未来工作方向**

- **集成 active tools**：加入代码执行、仿真环境等，实现“假设 → 验证”闭环。
- **扩展至跨学科 ideation**：支持多领域知识融合与迁移。
- **scaling up agent architecture**：探索更大 backbone 或 hierarchical agent 设计。
- **real-world deployment**：与科研人员协作，在真实项目中测试 agent 的辅助创新能力。

---

> 📌 **一句话总结**：  
> Agentic-Ideation 通过 **oracle-guided trajectory synthesis + masked SFT**，实现了 **高效、高质量、可泛化的 scientific ideation agent** 训练，在性能和效率上全面超越现有 workflow-based 方法，为构建真正自主的 AI Scientist 迈出关键一步。

</details>

---

### 7. [Performance Analysis in Parallel Programming Education: A Comparative Usability Study](https://arxiv.org/abs/2606.31458)

**Authors**: Anna-Lena Roth, David James, Jonas Posner, Michael Kuhn  
**Category**: cs.DC  
**Published**: 2026-07-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.31458v1  

#### Abstract
Parallel programming curricula encompass not only the development of parallel code and algorithm design but also emphasize efficiency, optimization, and performance analysis. To equip students with the skills necessary for writing efficient parallel code using message passing with MPI, practical exp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **教育环境中并行程序性能分析工具的高门槛问题**：现有的专业性能分析工具（如 CUBE、TAU、Score-P）虽然功能强大，但其复杂性、对集群架构和 MPI 的深度理解要求，以及后验式（post-mortem）分析模式，使得初学者难以有效使用。
- **缺乏适合教学场景的直观可视化工具**：学生在学习 MPI 和并行编程时，往往难以将抽象的通信行为与实际运行时性能问题（如负载不均、瓶颈、广播延迟）联系起来。

### **提出了什么新方法或新思路**
- **提出 EduMPI**：一个专为教学设计的 GUI-based 学习支持工具，集成于真实 HPC 集群环境，具备以下核心特性：
  - **自动化执行与测量**：无需手动配置 SSH、SLURM 或编译插桩，通过图形界面一键提交和运行 MPI 程序。
  - **近实时（near-real-time）可视化**：在程序运行过程中即时展示 MPI 通信行为，打破传统工具“运行 → 收集 → 分析”的延迟循环。
  - **多视角交互式视图**：提供 3D 节点/进程拓扑视图、2D 时间轴视图和通信矩阵（communication matrix），帮助学生从不同维度理解通信模式。
  - **定制化数据采集**：基于 Open MPI 的分支版本，捕获细粒度的 MPI 事件数据（包括集体通信背后的点对点操作）。

### **相比现有方法的优势**
| 特性 | EduMPI | 传统工具（CUBE/TAU） |
|------|--------|------------------|
| **易用性** | 极高（GUI + 自动化） | 低（需命令行、手动插桩） |
| **反馈延迟** | 近实时 | 后验式（post-mortem） |
| **学习曲线** | 平缓，适合初学者 | 陡峭，需先验知识 |
| **教学整合度** | 高，直接用于课堂练习 | 低，常因复杂被简化或跳过 |
| **探索性支持** | 强，支持动态观察与假设验证 | 弱，依赖预设指标 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **程序案例**：一个典型的并行行优先矩阵乘法程序（parallel row-wise matrix multiplication），采用 master-worker 模式，包含已知性能问题：
  - 主进程（rank 0）串行初始化大矩阵导致广播延迟（late broadcast）
  - 所有通信集中于主进程，形成通信瓶颈
  - 可扩展性差，通信开销随进程数增加而显著上升
- **参与者**：33 名硕士生（来自 Applied Computer Science、Data Science 等专业），均已修读《Parallel Programming》课程。

### **实验设置**
- **研究设计**：实验室控制下的**交叉对比实验**（counterbalanced crossover design）
  - **两个版本**：
    - **Version A**：先使用 EduMPI，再使用 TAU 或 CUBE
    - **Version B**：顺序相反
  - **分组方式**：以 dyads 或 triads 形式协作完成任务，一人操作，其余讨论，促进自然语言表达思维过程。
  - **总时长**：每组限时 90 分钟。

### **评估指标**
- **有效性（Effectiveness）**：任务正确完成率（独立或经提示后）
- **效率（Efficiency）**：
  - 每项任务耗时
  - 主持人干预次数（moderator interventions）
- **满意度（Satisfaction）**：
  - 李克特量表评分（Likert scale）：难度、可用性、可理解性
  - 推荐意愿调查
- **认知影响**：是否提升了对 MPI 行为的心理模型（mental model）

### **基线方法对比**
- **Baseline Tools**：
  - **CUBE**：开源性能报告浏览器，三窗格界面（metric/call/system tree），颜色编码表示性能指标。
  - **TAU**：全面的性能分析系统，支持 2D/3D 可视化（如条形图、函数级时间分布）。
- **共同前提**：所有工具已在课程中介绍，但未演示如何用于识别具体性能问题。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 指标 | EduMPI | TAU/CUBE |
|------|--------|----------|
| **任务正确率** | **96%** | **60%**（排除无法回答 T1 的 TAU 组） |
| **主持人干预频率** | **29%** 的任务需要帮助 | **55%** 的任务需要帮助 |
| **平均任务耗时** | 多数任务更快（T1-T3, T5-T6） | 少数任务更短（T4, T7） |
| **推荐用于教学** | 32 人“肯定推荐”，1 人“有条件推荐” | 仅 1 人“肯定推荐”，多数“不确定”或“可能不推荐” |

### **与基线方法的对比结果**
- **T1（进程分布）**：
  - EduMPI 正确率 100%
  - TAU 完全无法回答（无节点级视图）
- **T2（颜色编码理解）**：
  - EduMPI：全部小组能准确解释
  - CUBE：6/9 小组表示困惑，缺乏清晰图例
- **T3（矩阵 B 分发耗时）**：
  - 仅 4 组使用 CUBE/TAU 回答正确
  - 多数学生表示“看不懂指标含义”
- **T4（广播阶段瓶颈分析）**：
  - EduMPI：虽耗时较长（尤其 Version A），但**全部正确识别出串行初始化导致的 late broadcast**
  - CUBE/TAU：仅 4/12 组正确识别，部分误认为“发送比接收快是正常现象”
- **T7（全局主导通信函数）**：
  - 两者表现相当，均能识别 `MPI_Bcast` 占据最大通信开销

### **消融实验结果（隐含发现）**
- **顺序效应（Transfer Effect）**：
  - 先使用 EduMPI 的小组（Version A），在后续使用 TAU/CUBE 时表现更好（干预更少、成功率更高），表明 **EduMPI 起到了“认知桥梁”作用**。
- **视图偏好**：
  - EduMPI 的 3D 视图和通信矩阵被高频使用
  - CUBE 用户多次反馈“缺少类似 3D 拓扑视图”
- **聚合分析短板**：
  - EduMPI 在跨整个运行时间的全局聚合分析（如 T7）上不如 TAU/CUBE 直观，因其数据结构偏向实时流式而非预聚合。

---

## **4. 关键结论和发现**

### **论文的主要发现**
1. ✅ **EduMPI 显著降低性能分析的学习门槛**：96% 的任务正确率远超传统工具（60%），且学生主观评价更易用、更直观。
2. ✅ **近实时可视化增强理解深度**：学生能动态观察通信行为，主动探索假设（如“为什么所有进程都在等？”），而非被动解读静态报告。
3. ✅ **促进正确的性能问题归因**：避免了对 `MPI_Bcast` 中主进程提前退出的误解（即非“发送更快”，而是“计算阻塞”）。
4. ✅ **具有正向迁移效应**：先使用 EduMPI 可提升后续使用专业工具的能力，说明其是通往高级工具的有效过渡。
5. ✅ **激发学习动机与主动性**：学生更愿意尝试多次运行、调整参数，并主动关注通信效率而非仅代码正确性。

### **方法的局限性**
- ❌ **全局聚合分析能力较弱**：对于需要跨时间段综合判断的任务（如 T7），EduMPI 不如 TAU/CUBE 直接。
- ❌ **依赖定制化 Open MPI 分支**：限制了在其他集群上的部署灵活性。
- ❌ **当前仅支持特定通信模式分析**：尚未覆盖所有 MPI 特性（如非阻塞通信、I/O 性能等）。

### **未来工作方向**
- 🔧 **增强全局性能汇总功能**：引入类似 CUBE 的聚合指标面板，支持跨时段比较。
- 🔧 **集成 tracing-based 工具进行联合分析**：结合 Vampir 等工具提供更深层次的时间序列追踪。
- 🔧 **扩展至更多算法模板**：支持 pipeline、reduce、scatter/gather 等常见并行模式的教学分析。
- 🌐 **开发 Web 版本以提升可访问性**：摆脱 AppImage 限制，便于远程教学使用。
- 📚 **构建配套教学案例库**：围绕典型性能反模式（anti-patterns）设计渐进式练习任务。

> **总结一句话**：  
> **EduMPI 成功弥合了专业性能分析工具与教学需求之间的鸿沟，通过近实时、多视角、低门槛的可视化，使学生能够“看见”MPI，真正实现从“写并行代码”到“写高效并行代码”的转变。**

</details>

---

### 8. [Zero-Shot Quantization for Object Detectors using Off-the-Shelf Generative Models](https://arxiv.org/abs/2606.31456)

**Authors**: Hyunho Lee, Kyomin Hwang, Hyeonjin Kim, Suyoung Kim, Sunghyun Wee, Nojun Kwak  
**Category**: cs.LG  
**Published**: 2026-07-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.31456v1  

#### Abstract
With an increasing number of Object Detection (OD) models being deployed on edge devices, Zero-Shot Quantization for OD (ZSQ-OD) aims to quantize these models when access to the original training data is prohibited. Existing research on Zero-Shot Quantization-Aware Training (QAT) for OD synthesizes ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Zero-Shot Quantization for Object Detectors using Off-the-Shelf Generative Models》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **Zero-Shot Quantization for Object Detection (ZSQ-OD)** 中的关键挑战展开研究。在实际部署中，由于隐私或安全限制，无法访问原始训练数据，传统 Quantization-Aware Training (QAT) 难以进行。现有 ZSQ 方法（如 TSOD）依赖于噪声优化生成合成数据，在低比特量化（如 W4A4、W3A3）下性能严重下降。

此外，**目标检测任务本身具有三个独特挑战**：
1. 单张图像包含多个实例（multi-instance），要求生成图像具备高信息密度；
2. 类别分布极度不均衡（long-tailed distribution）；
3. 使用生成图像时需通过伪标签（pseudo-labels）监督 QAT，而这些标签可能引入噪声。

---

### 🚀 提出的新方法：GoodQ
作者提出 **GoodQ**（Generative off-the-shelf models for object detector Quantization），一个基于现成生成模型（off-the-shelf generative models，如 Stable Diffusion）的 ZSQ-OD 框架，包含三大核心组件：

#### （1）Information-Dense Prompting (IDP)
- **目的**：解决信息密度挑战。
- **方法**：设计多类别提示模板，例如 `"A photo of {Q1} {C1} and {Q2} {C2}"`，其中 `{C}` 是类别，`{Q}` 是数量形容词（如 multiple, several）。
- **效果**：显著提升每张图的平均 BBox 数量，更贴近真实检测数据特性。

#### （2）Intrinsic Distribution-Aware Selection (IDAS)
- **目的**：应对类别不平衡问题。
- **方法**：
  - 利用预训练检测器头部的 bias 参数估计原始数据集的类别先验分布（无需访问真实数据）；
  - 设计贪心选择算法，从生成图像池中挑选符合该分布的子集。
- **优势**：实现“data-free”下的分布对齐，KL 散度更低。

#### （3）Teacher-guided Adaptive Noise Reduction (TANR)
- **目的**：缓解伪标签噪声对 QAT 的干扰。
- **方法**：
  - 不使用硬标签（one-hot pseudo-labels），而是利用 full-precision teacher 模型输出的 soft predictions 作为监督信号；
  - 引入 QFocal-based Adaptive Weighting (AW)，让学生模型关注易受量化影响的样本。
- **本质**：将 `Ldetect` 中的目标由 GT 替换为 teacher 输出，形成软蒸馏式监督。

---

### 🔍 相比现有方法的优势
| 方面 | 传统方法（如 TSOD） | GoodQ |
|------|---------------------|-------|
| 数据生成方式 | 噪声优化（noise optimization） | 使用扩散模型生成多样化图像 |
| 信息密度 | 图像内容单一，缺乏多对象场景 | 显式构造多对象提示，提高语义密度 |
| 分布建模 | 忽略类别不平衡 | 可在无数据情况下估计并匹配分布 |
| 标签质量 | 依赖伪标签 → 存在噪声 | 使用 teacher soft label → 抑制噪声 |
| 极端低比特表现 | W4A4/W3A3 性能崩溃 | 在 W3A3 下仍保持可用性能 |

> ✅ **总体优势**：首次系统地将 off-the-shelf 生成模型适配到 ZSQ-OD 任务，并通过三阶段流程有效克服其应用障碍。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- 主要基准：**MS-COCO 2017**（train set 不可访问，仅使用 val set 测试）
- 泛化性验证补充：
  - **PASCAL VOC**
  - **HomeObjects-3K**

### ⚙️ 实验设置
- **模型架构**：
  - YOLOv5-s/m/l 和 YOLOv11-s/m/l（主流单阶段检测器）
  - 后续扩展至两阶段模型 **Mask R-CNN + Swin-T**
- **量化方案**：
  - 采用 **LSQ / LSQ+** 进行权重量化与激活量化
  - 比较不同 bit-width 设置：W8A8, W6A6, W4A4, W3A3
- **校准集大小**：统一使用 **2k 图像**
- **训练细节**：
  - Adam 优化器，100 epochs
  - 输入分辨率：640×640
  - 无数据增强（保证公平比较）

### 📊 评估指标
- **mAP / mAP50**：标准目标检测精度指标
- **消融实验指标**：
  - KL 散度、L1 距离（衡量类别分布匹配程度）
  - Cosine Distance（衡量特征空间多样性）
  - UMAP 可视化（观察嵌入空间分布）

### 🆚 对比的基线方法
| 方法 | 类型 | 是否需要真实数据 |
|------|------|------------------|
| **LSQ / LSQ+** | 基于真实子集的 QAT | ✅ 需要 |
| **TSOD** | 噪声优化 + 任务特定损失 | ❌ 不需要（ZSQ） |
| **TSOD+** | TSOD + LSQ+ 优化版本 | ❌ 不需要 |
| **GenQ-style diffusion** | 分类任务启发的扩散生成 | ❌ 不需要 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

#### 在 YOLOv5-s 上的表现（mAP / mAP50）

| 方法 | W8A8 | W6A6 | W4A4 | W3A3 |
|------|------|------|------|------|
| Pre-trained (FP) | 37.4 / 56.8 | - | - | - |
| LSQ (real data) | 32.6 / 51.2 | 29.8 / 47.9 | 19.2 / 34.4 | 8.48 / 17.7 |
| TSOD | 35.4 / 54.0 | 32.3 / 50.6 | 18.4 / 32.3 | 5.0 / 9.9 |
| **GoodQ** | **35.0 / 53.6** | **32.5 / 50.6** | **21.0 / 36.1** | **10.2 / 19.7** |

> 💡 **结论**：
- 在高比特（W8A8）下，GoodQ 接近甚至优于 TSOD；
- 在低比特区域（W4A4/W3A3），**GoodQ 显著领先 TSOD**，分别高出 **+2.6 mAP** 和 **+5.2 mAP**；
- 尤其在 W3A3 下，TSOD 几乎失效（~5 mAP），而 GoodQ 仍维持约 10 mAP，具备实用价值。

---

### 🔬 消融实验结果（Table 2 & Table 3）

#### （a）训练集构建策略消融（YOLOv5-s, W4A4）

| 方法 | mAP / mAP50 |
|------|-------------|
| Diffusion-only (GenQ prompt) | 16.0 / 28.8 |
| + IDP（信息密集提示） | 19.3 / 33.8 |
| + IDAS（分布感知选择） | 19.8 / 34.4 |
| **+ IDP + IDAS（完整版）** | **21.0 / 36.1** |

> ✅ 表明两个模块均有效，组合后增益最大。

#### （b）QAT 损失函数消融

| 方法 | W8A8 | W4A4 |
|------|------|------|
| Base（TSOD-style loss） | 33.9 / 52.8 | 20.6 / 35.8 |
| **+ TANR（teacher soft label）** | **35.0 / 53.6** | **21.0 / 36.1** |

> ✅ TANR 带来稳定增益，说明软标签能有效抑制噪声。

#### （c）图像池规模影响（Image Pool Size）

| 图像池大小 | W8A8 | W4A4 | W3A3 |
|-----------|------|------|------|
| 2k | 34.3 / 53.5 | 19.3 / 33.8 | 7.76 / 15.2 |
| 16k | 34.9 / 53.6 | 20.2 / 35.3 | 9.24 / 17.2 |
| **160k（Ours）** | **35.0 / 53.6** | **21.0 / 36.1** | **10.20 / 19.7** |

> ✅ 更大的图像池带来持续收益，尤其在低比特下更明显。

---

### 🌐 泛化性测试（Table 10–12）

| 场景 | 结果 |
|------|------|
| **Mask R-CNN + Swin-T**（两阶段模型） | GoodQ 在 W4A4 下达 33.7 mAP，优于 TSOD（32.2） |
| **SD2.1 替代 SD1.5** | GoodQ 依然显著优于 TSOD（W4A4: 20.5 vs 18.4） |
| **VOC / HomeObjects-3K** | 在不同数据集上一致胜出 |

> ✅ 表明 GoodQ 具有良好的跨模型、跨生成器、跨数据集泛化能力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **数据多样性是低比特 ZSQ 成功的关键**  
   扩散模型生成的图像在 CLIP 特征空间中分布更广（UMAP 可视化 + 更高的 cosine distance），有助于学生模型学习 teacher 的复杂行为。

2. **类别分布对齐至关重要**  
   即使拥有丰富图像池，若未正确采样以匹配原始长尾分布，性能反而会下降（见 MANY BBox 策略失败）。

3. **伪标签噪声不可忽视，teacher soft label 更鲁棒**  
   TANR 在扩散生成数据上的增益远大于在噪声优化数据上，说明生成图像带来的标签不确定性更高，必须通过软监督缓解。

4. **GoodQ 实现了当前最优的低比特 ZSQ-OD 性能**  
   在 W4A4 和 W3A3 下大幅超越 TSOD，首次将极端低位量化应用于检测任务并取得可用结果。

---

### ⚠️ 局限性
1. **计算成本较高**：
   - 生成 160k 图像耗时较长；
   - 需要额外推理进行伪标注和分布估计。
2. **依赖高质量生成模型**：
   - 若生成图像质量差（如模糊、错类），会影响最终性能；
   - 当前方法假设生成模型已具备一定语义理解能力。
3. **未完全消除 domain gap**：
   - 生成图像与真实图像之间仍存在风格、布局差异；
   - 尽管 TANR 缓解了部分问题，但仍非完美替代。

---

### 🔮 未来工作方向
1. **轻量化生成策略**：探索如何减少图像池规模同时保持性能（如核心集选择）；
2. **动态 prompt 生成**：结合检测器反馈迭代优化 prompt，实现闭环合成；
3. **扩展至其他视觉任务**：如实例分割、姿态估计等密集预测任务；
4. **结合 Post-Training Quantization (PTQ)**：在无需微调的情况下也适用生成数据；
5. **多模态协同优化**：联合优化 prompt、生成、选择与 QAT 过程。

---

## ✅ 总结一句话
> **GoodQ 是首个成功将 off-the-shelf 生成模型系统应用于 ZSQ-OD 的框架，通过 Information-Dense Prompting、Intrinsic Distribution-Aware Selection 和 Teacher-guided Adaptive Noise Reduction 三步策略，在无需原始数据的前提下实现了 state-of-the-art 的低比特量化性能，尤其在 W4A4 和 W3A3 极端设置下显著优于传统噪声优化方法。**

</details>

---

### 9. [Creating Intelligence: A Computational Foundation for AGI](https://arxiv.org/abs/2606.31819)

**Authors**: Peter Overmann  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.31819v1  

#### Abstract
This work introduces a new computational theory of mind grounded in set theory and hyperdimensional computing. Whereas traditional neural networks rely on continuous weights and matrix multiplication, this framework works with sparse binary data. It represents information as discrete sets, directly ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Creating Intelligence: A Computational Foundation for AGI

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文旨在为实现 **Artificial General Intelligence (AGI)** 提供一个全新的、基于计算理论的坚实基础。它试图解决当前主流 **Deep Learning** 范式在追求通用智能时面临的根本性瓶颈：
*   **生物不兼容性 (Biological Implausibility)**：传统神经网络依赖连续权重和反向传播，这与大脑的能量效率、稀疏脉冲活动和局部学习规则等特性严重不符。
*   **符号接地问题 (Symbol Grounding Problem)**：深度学习模型（如LLMs）处理的是“漂浮”的符号，其意义仅来自与其他符号的统计关联，而非对物理世界的直接感知。
*   **灾难性遗忘 (Catastrophic Forgetting)**：全局权重更新导致新知识覆盖旧知识。
*   **硬件瓶颈 (Hardware Bottleneck)**：密集矩阵运算受限于冯·诺依曼架构的内存墙（memory wall），功耗巨大。

### 提出了什么新方法或新思路
论文提出了一种名为 **Topological Associative Memory** 的全新计算框架，其核心创新点如下：

*   **以集合论为基础的数据结构 (Set-Theoretic Data Structure)**：
    *   将信息表示为离散的 **Sparse Distributed Representations (SDRs)** 和 **Sparse Holographic Representations (SHRs)**，这些本质上是高维空间中的稀疏二进制集合。
    *   这直接模拟了大脑中神经元群体的稀疏编码（population coding）。

*   **拓扑可塑性学习 (Topological Plasticity)**：
    *   彻底摒弃了连续的 **scalar weights**。记忆的存储和学习通过在网络拓扑中“打开”或“关闭”二进制连接（信号路径）来实现。
    *   学习是**局部的 (local)**，只影响特定的连接，避免了全局权重调整带来的副作用。

*   **统一的核心算法 (Unified Core Algorithm)**：
    *   提出了一种简洁的算法，将 **auto-associative**（自联想）和 **hetero-associative**（异联想）学习统一在一个框架下。
    *   核心机制是**子集模式匹配 (subset pattern matching)** 和**精确最近邻搜索 (exact nearest-neighbor search)**，通过一个组合爆炸的隐藏层（combinatorially expanded hidden layer）自然涌现出关联记忆。

*   **从感知到符号的桥梁 (Bridge from Perception to Symbol)**：
    *   提出 **Heteroencoder** 机制，能将重叠的感知数据（SDRs）动态地、无监督地映射为稳定的符号（SHRs），从而解决了符号接地问题。

### 相比现有方法的优势
| 特性 | Deep Learning | Topological Memory |
| :--- | :--- | :--- |
| **数据表示** | Dense, Continuous | Sparse, Binary (Sets) |
| **学习机制** | Global Backpropagation | Local Topological Plasticity |
| **能量效率** | Megawatts (GPU Clusters) | Watts (Human Brain Scale) |
| **学习速度** | Batch Training (Slow) | One-shot Learning (Instant) |
| **遗忘** | Catastrophic Forgetting | Graceful Decay (Local Forgetting) |
| **符号接地** | No (Entangled Spaces) | Yes (Heteroencoding) |
| **硬件** | GPU/TPU (von Neumann) | In-Memory Computing (Neuromorphic) |

## 2. 核心实验方法和设置

### 使用了哪些数据集
论文的验证并非完全依赖于传统的机器学习基准数据集，而是通过一系列**原理性演示和概念验证实验**来展示其框架的有效性：
*   **MNIST 手写数字数据集**：用于演示分类任务。
*   **Word Association Dataset**：一个包含约190万词关联的自定义数据集，用于演示语义相似性和知识图谱。
*   **合成数据集**：大量使用随机生成的 **SHRs** 和 **SDRs** 来进行容量、噪声容忍度、模式完成等核心功能的测试。

### 实验设置和评估指标
*   **核心算法实现**：提供了标准C语言和Mathematica的参考实现，确保了算法的可复现性。
*   **超参数 (Hyperparameters)**：关键参数包括输入/输出维度 `N`、人口数 `P`（即集合大小）、以及由理论推导出的**模式匹配阈值 (pattern matching threshold) `T`**。
*   **评估指标**：
    *   **检索准确率 (Retrieval Accuracy)**：测量检索出的集合与目标集合的重叠度（Overlap）和汉明距离（Hamming Distance）。
    *   **噪声容忍度 (Noise Tolerance)**：测试在输入包含加性噪声（额外元素）或减性噪声（缺失元素）时的检索能力。
    *   **容量 (Capacity)**：计算在达到50%存储密度前能存储的随机关联数量。
    *   **分离已知与未知 (Separating Known from Unknown)**：评估系统对随机查询返回空集的能力，防止幻觉（hallucination）。
    *   **迭代分解 (Iterative Decomposition)**：测量从一个捆绑包（bundle）中提取单个符号所需的平均迭代次数。

### 基线方法对比
论文并未直接与Transformer或CNN等具体模型进行端到端的性能比较。其对比是**范式层面**的：
*   **Hopfield Networks**：作为经典联想记忆的代表，被指出会收敛到吸引子（可能为虚假状态），而本文方法只有在有足够证据时才返回结果。
*   **Sparse Distributed Memory (Kanerva)**：虽然同属超维计算，但本文框架通过拓扑可塑性提供了更直接的生物学解释和更高效的实现。
*   **Deep Learning Paradigm**：在表19中进行了全面的范式对比，强调了在学习规则、表示、硬件等方面的根本差异。

## 3. 主要实验结果和性能指标

### 关键性能数据
*   **MNIST 分类**：使用一个简单的异联想记忆，经过单次训练，对10,000张测试图像达到了约 **96%** 的准确率，证明了其处理实际ML任务的能力。
*   **存储容量**：对于 `N=1,000`, `P=10` 的配置，名义容量约为 **769,393** 个异联想关联。对于相同参数的自联想记忆，容量更高，约为 **959,818** 个。
*   **存储效率**：每个自联想项的物理存储成本仅为 **519位**，且与维度 `N` 无关，远低于稠密向量表示。
*   **噪声容忍度**：在 `N=1,000`, `P=10` 配置下，默认阈值 `T=7`，意味着即使丢失3个元素（30%的输入），仍能成功检索。
*   **迭代分解**：实验显示，从一个捆绑包中提取一个符号所需的平均迭代次数与 **Miller's "magical number seven, plus or minus two"** 高度吻合，支持了人类工作记忆的认知模型。

### 与基线方法的对比结果
*   **与Hopfield网络相比**：本文方法能可靠地区分已知和未知信息，而Hopfield网络总是会返回一个结果，可能导致虚假记忆。
*   **与传统深度学习相比**：在**学习效率**上，实现了**一次性学习 (one-shot learning)**；在**能量效率**上，理论上可达到人脑水平（瓦特级 vs 兆瓦级）；在**硬件适应性**上，天然适合存内计算（in-memory computing）。

### 消融实验结果
论文通过理论分析和变体讨论，间接展示了核心设计的重要性：
*   **二进制存储 vs 非二进制存储**：实验证明，使用整数计数器的非二进制存储无法支持自联想功能，因为自相关会导致权重无限膨胀，使记忆退化为恒等函数。这凸显了**二进制拓扑存储**的必要性。
*   **模式匹配阈值 `T`**：通过公式推导出最优 `T`，并证明过低的 `T` 会导致假阳性，过高的 `T` 会降低噪声容忍度，验证了其作为“安全护栏”的作用。
*   **随机子采样 (Stochastic Subsampling)**：在迭代分解和图遍历中，随机子采样是打破对称性、实现功能的关键。没有它，系统会在平局（tie）中失败。

## 4. 关键结论和发现

### 论文的主要发现
1.  **认知的基本引擎是子集模式匹配**：作者提出，无论是小脑还是新皮层，其核心计算功能都是一种基于集合的、通过子集模式匹配实现的联想记忆。
2.  **离散逻辑优于连续算术**：用二进制拓扑操作取代浮点矩阵乘法，不仅更节能，而且能实现精确的、无损的信息检索和推理。
3.  **学习敏捷性源于底层架构**：真正的通用智能体现在快速学习新技能的能力，而这需要一个像本文提出的、支持在线、局部、一次性学习的计算框架。
4.  **硬件与算法的协同进化**：该框架天然指向**存内计算 (in-memory computing)** 硬件，有望彻底解决AI的能耗问题。

### 方法的局限性
*   **工程实现尚处早期**：目前主要是理论和软件原型，高性能的专用硬件（如基于DRAM/NAND的加速器）尚未大规模建成。
*   **复杂任务的扩展性待验证**：虽然框架可以构建复杂电路（circuits），但如何用它高效地解决如长文本理解、复杂规划等高级AGI任务，仍需更多探索。
*   **对“涌现”的解释有限**：该框架强调符号和结构，对于深度学习中某些“黑箱”式的涌现能力（如few-shot learning in LLMs）的解释力尚不明确。

### 未来工作方向
*   **开发专用硬件**：实现论文中设想的基于DRAM或NAND的存内计算加速器。
*   **构建复杂认知架构**：利用“合成神经解剖学 (synthetic neuroanatomy)”的理念，设计和集成更复杂的电路模块，如工作记忆、情景记忆等。
*   **探索新的学习范式**：进一步研究预测性学习（predictive learning）和检索启发的遗忘（retrieval-inspired forgetting）等机制。
*   **跨领域应用**：将该框架应用于机器人、边缘计算、脑机接口等领域，发挥其低功耗、实时学习的优势。

</details>

---

### 10. [Offline Reinforcement Learning for Fluid Controls: Data-based Multi-observational Policy Extraction](https://arxiv.org/abs/2606.31025)

**Authors**: Deepak Akhare, Luning Sun, Xin-Yang Liu, Xiantao Fan, Timo Bremer, Ben Zhu, Jian-Xun Wang  
**Category**: cs.LG  
**Published**: 2026-07-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.31025v1  

#### Abstract
Active flow control is a fundamental application in engineering. Recent advances in deep reinforcement learning have made progress in this field. However, the classical online RL approaches require extensive real-time interactions with the high fidelity environment, while each sensor configuration c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Offline Reinforcement Learning for Fluid Controls: Data-based Multi-observational Policy Extraction

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于在线强化学习（online RL）的流体控制方法存在以下瓶颈：
- 需要大量与高保真环境的实时交互，计算成本极高（通常需要 $10^4$–$10^6$ 次交互），难以在真实工程中部署。
- 政策训练依赖固定的传感器配置，一旦传感器位置改变，就必须重新收集数据并重新训练，导致适应性差。

此外，大多数离线强化学习（offline RL）方法虽然避免了在线交互，但仍为每种观测配置训练独立策略网络，无法灵活应对多变的传感器布局。

### 🚀 提出的新方法与创新思路
本文提出了一种**新型的离线强化学习框架**，核心是 **Sensor Position-Conditioned Policy Network (PCT-net)** 和 **Sensor-Aware SACN (SA-SACN)** 算法，实现从单一数据集中提取适用于多种传感器配置的通用控制策略。

#### 主要创新点包括：
1. **数据驱动的多观测策略提取（Multi-observational Policy Extraction）**
   - 利用一个预先采集的全状态数据集（full-state dataset），从中离线提取多个针对不同传感器布局的策略，无需任何额外环境交互。

2. **PCT-net 架构设计**
   - 引入 **Point Transformer 层** 对传感器位置进行编码，并将其作为条件输入生成策略网络的权重（hypernetwork-style weight generation）。
   - 实现单个策略网络可泛化到任意数量和位置的传感器组合，具备**排列不变性（permutation invariance）** 和空间关系建模能力。

3. **SA-SACN 训练算法**
   - 在 SACN 基础上引入传感器位置随机采样机制，在训练时动态变化传感器布局（$x_s \sim \mu + (2U(\cdot)-1)\Delta$），增强模型对传感器扰动的鲁棒性。

4. **联合优化：策略 + 传感器布局**
   - 提出辅助目标函数（Eq. 8），可在训练过程中同步优化传感器的位置 $\{w\}$，以最大化预期回报，从而实现**端到端的传感器优化**。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | 本工作（PCT-net + SA-SACN） |
|------|--------|----------------------------|
| 数据效率 | 需要在线交互或重复数据采集 | 完全离线，仅需一次数据采集 |
| 传感器灵活性 | 固定配置，换位即重训 | 单一网络支持任意布局，零样本迁移 |
| 计算复杂度 | $O(k \times n)$（k 种配置） | $O(n)$（统一训练） |
| 可扩展性 | 不适用于快速迭代实验 | 支持传感器优化与部署一体化 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
所有实验均基于**仿真环境中生成的离线数据集**，无真实物理实验数据：
- **Kuramoto-Sivashinsky (KS) 系统**
  - 数据来源：通过标准 online SAC 算法训练得到的行为策略（behavior policy）
  - 数据规模：每个策略生成 10,000 条轨迹，每条含 400 控制步
  - 状态维度：1D PDE 解（64 网格点）
  - 动作：4 个执行器控制信号 $a_i \in [-0.5, 0.5]$
  - 观测：8 个传感器测量局部状态值

- **Flow over Airfoil（翼型绕流）系统**
  - 数据来源：同上，使用 medium/expert 策略生成数据
  - 数据规模：2,000 条轨迹（因高维状态内存占用大，约 1.5TB）
  - 物理方程：Navier-Stokes 方程（2D 不可压缩流）
  - 数值求解器：自研 GPU 加速 CFD 求解器 [Diff-flowfsi][39]
  - 动作：3 个表面射流控制参数
  - 观测：6 个传感器测量速度与压力

> 所有数据包含完整转移四元组 $(s, a, r, s')$，用于 offline RL 训练。

### ⚙️ 实验设置与评估指标
#### 模型架构
- **Actor 网络**：PCT-net（含 Point Transformer 编码器 + MLP 策略主干）
- **Critic 网络**：SACN 风格的 Q-ensemble（N=5），采用保守估计缓解分布偏移
- **Baseline 方法对比**：
  - SACN + MLP（固定传感器布局）
  - SACN + PCT-net（固定布局训练）
  - SA-SACN + PCT-net（支持布局扰动训练）

#### 评估方式
- **测试 rollout 数量**：200 次不同初始条件下的闭环控制模拟
- **主要指标**：累计回报（Return）分布直方图
- **鲁棒性测试**：将训练好的策略应用于轻微扰动后的传感器位置（$\sigma = 1/(3\times16)$）
- **消融实验**：比较是否启用 sensor-aware 训练、sensor optimization 等模块

---

## 3. 主要实验结果和性能指标

### 📊 关键性能表现

#### （1）KS 系统中的策略提取效果（Fig. 5）
- 所有四种传感器布局下，SACN 成功提取出优于原始行为策略的控制器。
- 特别是在靠近执行器的布局（a）、（b）、（c）中，返回值显著提升；而远离执行器的布局（d）性能下降，说明**可观测性影响控制效能**。
- PCT-net 与 MLP 表现相当，证明其结构未牺牲性能。

#### （2）翼型绕流系统（Fig. 6）
- 在三种不同传感器布局下，SA-SACN 提取的策略性能接近原行为策略水平。
- 尽管任务更复杂（2D NS 流场），仍能稳定复现有效控制行为。

#### （3）传感器扰动下的鲁棒性（Fig. 7 & Fig. 9）
| 方法 | KS 环境（扰动后） | Airfoil 环境（扰动后） |
|------|------------------|-----------------------|
| SACN + MLP | 性能大幅下降 | 显著退化 |
| SACN + PCT-net | 下降明显 | 下降明显 |
| **SA-SACN + PCT-net** | **保持高性能** | **几乎无损** |

> 结果表明：**只有经过 sensor-aware 多布局训练的 SA-SACN 能够真正泛化至未见的传感器配置**。

#### （4）跨配置迁移能力（Fig. 8）
- 使用配置 (c) 训练的 PCT-net，直接用于其他布局推理，仍取得良好性能。
- 证明 PCT-net 学到了“一组策略”的共享表示，而非单一映射。

#### （5）传感器优化结果（Fig. 10–11）
- 引入 sensor optimization 目标后，初始较差布局（如 a, d）向高性能区域（类似 c）演化。
- 优化后策略不仅在平均位置表现更好，且在扰动下也更具鲁棒性。
- 最终识别出靠近流动分离区的传感器位置更有利于控制。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **离线策略提取可行且高效**  
   从历史数据中可以成功提取出优于原始行为策略的控制器，验证了 SACN 在流体控制任务中的有效性。

2. **PCT-net 实现“一网多用”**  
   单个 PCT-net 可适配多种传感器布局，消除重复训练开销，极大提升了部署灵活性。

3. **SA-SACN 显著增强鲁棒性**  
   通过在训练中引入传感器位置扰动，使策略对实际部署中的安装误差具有强容错能力。

4. **传感器布局可被自动优化**  
   模型不仅能学策略，还能反向优化传感器位置，形成“感知-决策”协同设计闭环。

### ⚠️ 方法的局限性
- 当前方法依赖高质量的全状态离线数据集，若原始数据覆盖不足或策略次优，则提取性能受限。
- Point Transformer 的计算开销随传感器数增长较快，可能限制极端稀疏或密集传感场景的应用。
- 实验仅限于 1D 和 2D 仿真环境，尚未在三维真实湍流或硬件平台验证。

### 🔮 未来工作方向
1. **提升训练效率**：结合 multi-fidelity learning 或 latent space modeling 减少高维流场处理负担。
2. **拓展至 3D 复杂流动**：应用于飞机整机、风力机叶片等实际工程场景。
3. **实现实时部署**：集成轻量化模型与边缘计算设备，推动实验室成果走向工业应用。
4. **探索主动感知机制**：让策略动态选择激活哪些传感器，进一步降低能耗与成本。

---

> **一句话总结**：  
> 本文提出了首个支持**多传感器配置泛化**的离线强化学习框架 PCT-net + SA-SACN，实现了从单一数据集提取灵活、鲁棒、可优化的流体控制策略，为智能自适应控制系统提供了新范式。

</details>

---

### 11. [Beyond the Library: An Agentic Framework for Autoformalizing Research Mathematics](https://arxiv.org/abs/2606.31134)

**Authors**: Arshia Soltani Moakhar, Iman Gholami, Max Springer, Mahdi JafariRaviz, MohammadTaghi Hajiaghayi  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.31134v1  

#### Abstract
While Large Language Models (LLMs) have demonstrated exceptional capabilities in mathematical reasoning, they frequently produce subtle errors that evade human detection. Formal mathematical languages like Lean 4 offer mechanical proof checking, strongly motivating the need for autoformalization: th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond the Library: An Agentic Framework for Autoformalizing Research Mathematics

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文致力于解决**研究级数学定理的自动形式化**（autoformalization）问题。尽管大型语言模型（LLMs）在数学推理方面表现出色，但其生成的自然语言证明常包含细微逻辑错误，难以被人类察觉。而形式化语言如 **Lean 4** 能提供机械化的证明验证能力，因此将非形式化数学（natural language mathematics）自动翻译为可验证的 Lean 代码成为关键挑战。

然而，现有方法面临以下瓶颈：
- **缺乏对基础类型的动态扩展能力**：主流形式化库（如 Mathlib）覆盖有限，无法支持前沿研究中出现的新概念。
- **定理陈述的形式化缺乏验证机制**：不同于证明过程可通过编译器反馈迭代优化，定理陈述本身没有“正确性”的机械标准。
- **固定流水线易失败**：传统单代理或刚性流水线系统在出错后难以回溯修正，上下文迅速饱和，导致长程推理失败。

---

### **提出了什么新方法或新思路**

作者提出了一种**基于多智能体的框架**（agentic framework），通过一个中央**Orchestrator**协调两个并行流水线：

1. **Statement Formalization Pipeline**（定理形式化）
2. **Proof Formalization Pipeline**（证明形式化）

#### 核心创新点包括：

- **“Type-First” 形式化范式**  
  在正式处理主定理前，系统首先识别并形式化所有缺失的自定义类型（types），构建必要的数学词汇表。这使得系统能处理 Mathlib 未覆盖的研究领域。

- **辅助引理技术（Auxiliary Lemma Technique）作为“单元测试”**  
  对每个新定义的类型，系统生成一组通用性质相关的辅助引理，并尝试用定理证明器去证明它们。若失败，则说明类型定义可能有误或不充分。这一机制类比于软件工程中的单元测试，确保基础构件的正确性和可用性。

- **双管道、可动态回溯的多智能体架构**  
  - 多个子代理（如 Type Formalizer、Lemma Planner、Theorem Formalizer 等）分工协作，避免单一代理上下文溢出。
  - Orchestrator 可根据反馈动态调整执行顺序，实现非线性、可回滚的任务流，显著提升鲁棒性。

- **Faithfulness Judge 验证语义一致性**  
  引入四重验证机制（盲反向翻译 + 直接比较 + 多模型交叉判断），确保生成的 Lean 代码忠实反映原始自然语言含义，防止语义漂移。

- **递归分解与自底向上证明策略**  
  证明阶段采用树状结构，先证明父节点依赖子引理的前提是否成立，再递归进入子节点，避免无效劳动。

---

### **相比现有方法的优势**

| 维度 | 本方法优势 |
|------|-----------|
| **灵活性** | 支持动态类型扩展，突破 Mathlib 覆盖限制 |
| **可靠性** | 辅助引理 + Faithfulness Judge 提供双重保障 |
| **可维护性** | 多代理设计隔离任务，便于调试与人类介入 |
| **成本效率** | 基于商用订阅服务运行，无需本地 GPU，成本极低 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

1. **PutnamBench**  
   - 包含历年普特南数学竞赛题目（共 672 道）。
   - 已提供 Lean 形式化的定理陈述，用于评估**证明生成能力**。
   - 本文从中随机抽取 **32 道题**进行端到端评估。

2. **STOC 论文集合**  
   - 来自 ACM Symposium on Theory of Computing 的五篇前沿理论计算机科学论文：
     - Pham [17]（组合学）
     - Mackenzie and Saffidine [19]（通信复杂性）
     - Gravin and Jia [20]（机制设计）
     - Rivkin et al. [21]（信息论下界）
     - Kalai et al. [22]（学习理论）
   - 这些论文代表**研究级数学**，形式化难度高，且涉及大量未在 Mathlib 中定义的概念。

---

### **实验设置和评估指标**

#### 实验流程
- 用户输入论文的 PDF 和 LaTeX 文件及提示指令。
- Orchestrator 启动双流水线：
  - **形式化流水线**：提取主定理 → 规划所需类型 → 逐个形式化类型及其辅助引理 → 最终形式化定理。
  - **证明流水线**：生成自然语言证明草稿 → 分解为子引理 → 形式化并递归证明。

#### 评估方式
- **机器验证**：所有 Lean 代码必须通过 Lean 编译器检查（no `sorry`）。
- **人工审核**：由领域专家确认形式化是否忠实于原文。
- **无外部资源访问**：禁用互联网以防止检索已有证明。

#### 性能指标
- **准确率下界估计**（Lower-bound accuracy at 95% confidence）
- **每题平均成本**（美元）
- **是否引入额外公理**（axiom-free？）
- **是否完全自包含证明**（reproved prior work?）

---

### **基线方法对比**

| 方法 | 准确率 | 成本/题 |
|------|--------|--------|
| AlephProver [23] | 94.8% | $54 |
| Seed-Prover 1.5 [9] | 86.5% | ~$168 |
| Hilbert [10] | 68.8% | ~$39 |
| DeepSeek-Prover-V2 [26] | 7.0% | ~$18 |
| **本文方法（Ours）** | **≥91.3%** | **~$5** |

> 注：本文使用的是 **Claude Opus 4.7/4.8** 的商业订阅服务（$200/月），而非按 token 计费的 API。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 指标 | 结果 |
|------|------|
| PutnamBench 抽样成功率 | **32/32 全部解决** |
| 推断出的整体准确率下界 | **≥91.3%**（95% 置信水平） |
| 平均每题运营成本 | **约 $5**（基于订阅制） |
| API等价成本估算 | **约 $29/题**（若按 token 计费） |
| STOC 论文形式化数量 | **5 篇全部完成主定理形式化** |
| 完全无公理证明（仅用 Lean 内核公理） | **2 篇**（[19], [20]） |

---

### **与基线方法的对比结果**

- **性能上优于多数专用模型**：尽管 AlephProver 达到 94.8%，但其成本高达 $54/题；本文方法以不到十分之一的成本实现了接近的性能。
- **远低于其他高端系统的开销**：
  - Seed-Prover 1.5 需 10 H20 GPU-day/题 ≈ $168；
  - 本文仅需标准软件订阅即可运行。
- **首次实现研究级论文的全自动、无 `sorry` 证明**，其中两篇甚至重新证明了引用成果，达到完全自包含。

---

### **消融实验与关键发现**

虽然文中未明确列出传统意义上的“消融实验”，但通过不同论文的表现差异揭示了系统行为的关键特性：

| 发现 | 说明 |
|------|------|
| **辅助引理机制有效检测定义缺陷** | 在 Pham [17] 中，系统因无法证明某引理而发现原论文证明存在漏洞（Lemma 3.6 不成立）。 |
| **Faithfulness Judge 成功阻止语义偏差** | 多次通过反向翻译发现形式化偏离原意，并触发修复循环。 |
| **动态回溯避免硬崩溃** | 当某类型形式化失败时，Orchestrator 自动启动更多 Type Formalizer 或修改计划，而非终止流程。 |

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **通用编码 LLM 可胜任高级数学形式化任务**  
   利用 **Claude Code** 这类通用编程智能体，无需专门微调，即可完成研究级数学的形式化与证明。

2. ✅ **软件工程思想可迁移至形式化数学**  
   “类型优先”、“单元测试式验证”、“模块化分解”等理念极大提升了系统的可靠性和可扩展性。

3. ✅ **该框架不仅能形式化，还能发现文献中的潜在错误**  
   在 Pham [17] 的案例中，系统发现了原论文中一个未被察觉的证明漏洞（surjection argument 不成立），并通过引入一个显式公理将其隔离，体现了其**科学价值**。

4. ✅ **研究论文的“自包含程度”决定了形式化所需的公理数**  
   - Mackenzie & Saffidine [19] 和 Gravin & Jia [20]：**零额外公理**，所有引用结果均被重新证明。
   - Rivkin et al. [21] 和 Kalai et al. [22]：承认少量经典分析事实为公理（如 Prekopa-Leindler 不等式）。
   - Pham [17]：暴露了一个应被证明却实际未成立的引理，最终作为唯一公理引入。

> 🔍 **重要洞见**：所需公理的数量不是系统弱点的体现，而是对原始论文严谨性的客观反映。

---

### **方法的局限性**

| 局限 | 描述 |
|------|------|
| **不支持算法类结果的形式化** | 如 Kalai et al. [22] 中的学习算法本身未被形式化（见 Appendix A）。当前系统聚焦于存在性/结构性定理的证明。 |
| **依赖闭源模型（Claude Code）** | 底层模型不可控，版本更新可能导致结果不可复现。 |
| **对高度抽象或几何直观强的内容适应性未知** | 当前测试集中未包含拓扑、代数几何等领域的问题。 |

---

### **未来工作方向**

1. **扩展至算法形式化**  
   将算法伪代码、时间复杂度分析也纳入形式化范围，实现真正端到端的论文机械化。

2. **开源替代方案探索**  
   构建基于开放权重模型（如 Llama 系列）的类似框架，提高可复现性与透明度。

3. **集成到学术出版流程**  
   将此类系统嵌入论文投稿系统，在发表前自动进行形式化审查，提升数学出版物的可信度。

4. **跨学科应用拓展**  
   应用于物理、化学、经济学等领域（参考 [50][51][52]），推动整个科学体系的形式化转型。

---

> 📌 **总结一句话**：  
> 本文提出的 agentic 框架不仅实现了高性能、低成本的研究级数学自动形式化，更展示了 AI 系统从“辅助工具”迈向“独立科研伙伴”的潜力——它不仅能读懂论文，还能指出错误，甚至推动数学实践本身的变革。

</details>

---

### 12. [Towards Inclusive Mobility Modeling: Characterizing and Evaluating Elderly Trajectory Patterns in Urban Systems](https://arxiv.org/abs/2606.31207)

**Authors**: Zhengxuan Wang, Haohan He, Mengying Zhou  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.31207v1  

#### Abstract
The rapid advance of smart cities increasingly depends on trajectory data mining, yet underrepresented demographic groups, particularly the elderly, are often sparsely represented in public mobility datasets. This underrepresentation can introduce systematic bias into mobility modeling and downstrea...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Towards Inclusive Mobility Modeling: Characterizing and Evaluating Elderly Trajectory Patterns in Urban Systems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本研究聚焦于**城市系统中老年人群在轨迹数据中的代表性不足问题**。当前大多数 mobility modeling 和 smart city 应用依赖大规模轨迹数据（如 Citi Bike），但这些数据往往以年轻、通勤人群为主，导致对老年人等弱势群体的出行模式建模存在系统性偏差（algorithmic bias）。这种偏差会影响下游的城市规划决策（如设施布局、公共交通设计），从而加剧数字不平等。

### ✅ 提出的新方法与新思路
1. **首次系统量化了老年人与年轻人在共享骑行系统中的 mobility signature 差异**：
   - 发现老年人具有更局域化的活动空间（localized activity spaces）、更低的行为熵（mobility entropy）以及非对称的非高峰时段出行模式。
   
2. **构建了一个受控的生成式实验框架**，用于评估 demographic composition 对 synthetic trajectory generation 的影响：
   - 在三种训练条件下比较模型表现：全人群（full population）、仅年轻人（young-only）、仅老年人（elderly-only）。
   - 使用 Markov Chain 和 LLM（Qwen3-4B + QLoRA）两种范式进行对比，揭示高容量模型在 minority subgroup 上的表现局限。

3. **提出可复现的评估流水线（reproducible evaluation pipeline）**：
   - 覆盖从数据预处理 → mobility metric 构建 → 合成轨迹生成 → 子群体特异性评估全过程，为未来研究提供基准。

### ✅ 相比现有方法的优势
- **强调 demographic-specific fidelity**：不同于传统追求“总体拟合度”的轨迹生成模型，本文强调应针对 underrepresented subgroups（如老年人）单独评估其建模准确性。
- **揭示主流训练策略的风险**：证明即使使用先进 LLM 模型，若训练数据由 majority group 主导，仍会严重误判 minority 的行为特征。
- **跨模型范式的公平比较**：在同一实验设置下对比结构化概率模型（Markov）与深度序列模型（LLM），得出关于模型能力与数据代表性的深刻洞见。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **数据来源**：2016–2020 年间 Jersey City 的 **Citi Bike System Data**。
- **关键优势**：该时间段的数据包含用户的出生年份（birth year），可用于识别年龄分组（老年 ≥65 岁，青年 18–35 岁）。
- **数据规模**：
  - 总计保留 1,528,325 条行程（trip）
  - 老年人组：14,394 条行程，来自 2,007 辆单车（占总行程 0.9%）
  - 青年组：783,558 条行程，来自 5,832 辆单车（占比 51.3%）

> ⚠️ 注：自 2021 年起，Citi Bike 因隐私原因移除了公开数据中的 demographic attributes，使得此类分析更加稀缺。

### 🔬 实验设置
- **目标**：评估不同 demographic training 设置下，合成轨迹对真实老年人出行模式的还原能力。
- **控制变量设计**：
  - 所有模型均生成 **2,007 条合成轨迹**，匹配真实老年人轨迹数量。
  - 合成轨迹长度采样自真实老年人的分布（平均 7.2 trips/trajectory）。
- **三种训练条件**：
  1. `elderly-only`：仅用老年人数据训练
  2. `young-only`：仅用青年数据训练
  3. `full-population`：使用全部过滤后的数据训练

### 🧪 评估指标（Mobility Metrics）
| 类别 | 指标 | 定义 |
|------|------|------|
| **Spatial Metrics** | Step Length | 出行起点到终点的 Haversine 距离（平均值） |
| | Radius of Gyration (RoG) | 衡量空间分散程度，相对于轨迹质心的距离加权平均 |
| | Mobility Entropy | 基于站点访问频率计算的 Shannon entropy，反映行为规律性 |
| **Temporal Metrics** | Speed | 行程距离 / 时间 |
| | Dwell Time | 同一辆车连续两次出行之间的空闲间隔（限制在 [0, 86400) 秒内） |
| | Intra-day Temporal Distribution | 小时级出发时间分布 |

### 🆚 基线方法对比
| 模型类型 | 具体实现 |
|--------|---------|
| **First-order Markov Chain** | 建模站点转移概率 $P(s_j|s_i)$、起始站分布、停留时间分布、行程持续时间（Gaussian 分布） |
| **LLM-based Model** | 使用 **Qwen3-4B** 模型，通过 **4-bit QLoRA** 微调（rank=16, dropout=0.05），将轨迹编码为 station ID 序列进行训练 |

> ⚠️ 注意：LLM 不直接输出时间属性，故采用后处理方式统一设定：
> - 行程速度设为老年人均值 4.17 m/s
> - 停留时间从指数分布（mean=600s）中采样

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（相对误差 %）

#### ✅ Markov 模型结果（vs. Real Elderly）
| Metric | M_full | M_young | M_elderly |
|--------|--------|--------|----------|
| Step Length (m) | +4.5% | +6.3% | **-2.0%** |
| Speed (m/s) | -17.0% | -5.9% | +6.2% |
| Dwell Time (s) | +8.9% | +44.3% | **-4.9%** |
| RoG (m) | +15.0% | +15.6% | -22.9% |
| Entropy | +141.9% | +130.8% | **-1.8%** |

> 💡 结论：`M_elderly` 在多数指标上显著优于其他两个模型，尤其在 entropy 和 dwell time 上接近真实值；但 RoG 被低估，表明空间覆盖受限。

#### ✅ LLM 模型结果（vs. Real Elderly）
| Metric | L_full | L_young | L_elderly (**best**) |
|--------|--------|--------|---------------------|
| Step Length (m) | -12.5% | -3.7% | +18.4% |
| Speed (m/s) | +72.4% | +72.2% | +72.3% |
| Dwell Time (s) | -96.9% | -96.9% | **-96.9%** |
| RoG (m) | -19.2% | -19.2% | **-5.7%** |
| Entropy | +57.0% | +67.6% | **+49.1%** |

> 💡 结论：
> - `L_elderly` 在 RoG 和 Entropy 上表现最好，说明 LLM 更好地缓解了稀疏数据下的空间收缩问题；
> - 但在时间维度（speed, dwell time）上所有 LLM 变体都严重偏离真实值，因时间属性是后处理生成；
> - 整体来看，LLM 并未在 minority subgroup fidelity 上超越简单 Markov 模型。

### 🔍 消融实验与关键发现
- **demographic composition 显著影响生成质量**：
  - 使用全人群训练的模型（M_full）虽然在整体上可能表现尚可，但对老年人的关键空间指标（如 entropy, RoG）仍有 >140% 的误差。
- **higher-capacity model ≠ better subgroup fidelity**：
  - 尽管 Qwen3-4B 参数量远大于 Markov 模型，但由于老年人数据极度稀疏（仅 ~1.4w trips），无法有效学习其独特模式。
  - LLM 甚至在某些方面（如 dwell time）产生更大偏差。
- **elderly-specific training 是提升 fidelity 的关键路径**：
  - 即使数据少，专门训练的模型（M_elderly）在 entropy、step length、dwell time 上误差最低。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **老年人具有结构性不同的 mobility signature**：
   - 更小的 RoG（958m vs. 1,189m）
   - 更低的 mobility entropy（1.82 vs. 4.15）→ 行为更规律
   - 更均匀的日间出行分布（避开早晚高峰）
   - 更短的 dwell time（18,329s vs. 27,928s）

2. **主流训练方式会导致系统性偏见**：
   - 基于 majority-dominated 数据训练的模型会显著高估老年人的空间移动性和行为多样性。
   - 例如，M_full 高估老年人 step length 4.5%，dwell time 8.9%。

3. **专用子群模型能显著改善建模精度**：
   - 老年人专属模型（M_elderly）在多个关键指标上误差低于 5%，显著优于通用模型。

4. **高容量模型不一定更好**：
   - 在 demographic data 极度稀疏的情况下，LLM 并未展现出优越性，反而因过度泛化而放大偏差（如 entropy 过高）。
   - 表明 **data inclusivity 比 model capacity 更重要**。

### ⚠️ 局限性
1. **地理范围有限**：仅基于 Jersey City 的 Citi Bike 数据，是否适用于更大城市或其他交通模式有待验证。
2. **demographic 维度单一**：仅考虑 age，未纳入 gender、income、健康状况等交叉因素。
3. **历史时期限制**：数据时间为 2016–2020，疫情前后出行行为可能发生结构性变化。
4. **LLM 时间建模缺失**：当前 LLM 仅建模 station 序列，未联合建模 temporal dynamics，限制其实际应用价值。

### 🔮 未来工作方向
1. **扩展至更多城市与交通方式**：验证 subgroup-specific mobility patterns 是否具有普适性。
2. **多维 demographic fairness analysis**：结合性别、收入、种族等因素，开展 intersectional equity 研究。
3. **developing privacy-preserving yet inclusive data collection mechanisms**：在保护隐私的同时恢复 demographic annotations。
4. **designing subgroup-aware generative models**：开发能自动检测并适应 minority patterns的 adaptive trajectory generators。
5. **longitudinal studies**：追踪同一群体随时间的变化，理解 mobility disparity 的动态演化。

---

> 📌 **一句话总结**：  
> 本文揭示了 mobility modeling 中因 demographic underrepresentation 导致的算法偏见，并证明——**与其依赖大模型拟合多数人，不如为少数群体建立专属模型**。真正的 inclusive mobility modeling 必须从数据采集阶段就重视 representation，而非寄希望于后期模型补偿。

</details>

---

### 13. [RAISE: LLM-based Automated Heuristic Design with Robust Adversary Instance Search](https://arxiv.org/abs/2606.31801)

**Authors**: Fei Liu, Alessio Figalli, Patrick Owen, Nicola Serra  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.31801v1  

#### Abstract
Automated Heuristic Design (AHD) with Large Language Models (LLMs) has shown remarkable progress in discovering high-quality heuristics. However, existing LLM-based AHD methods optimize heuristics for a fixed training instance set and may fail catastrophically when deployed under real-world distribu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RAISE: LLM-based Automated Heuristic Design with Robust Adversary Instance Search  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **LLM-based Automated Heuristic Design (AHD)** 方法在训练时仅针对固定的实例分布进行优化，导致其在面对真实世界中的**分布偏移（distribution shift）** 时性能急剧下降，甚至出现灾难性失效。这一现象严重限制了这些方法在动态、不可预测环境下的实际部署能力。

### 提出了什么新方法或新思路
本文提出 **RAISE**（Robust Adversary Instance Search），一种将**对抗性最坏情况实例搜索**嵌入到 LLM 驱动的进化搜索循环中的新框架。其核心思想是将鲁棒 AHD 形式化为一个**约束最小最大优化问题**：

$$
h^* = \arg\max_h \min_{s' \in B_\epsilon(S)} \text{eval}(h, s')
$$

其中 $B_\epsilon(S)$ 是围绕名义训练实例集 $S$ 的 $\epsilon$-ball 不确定性集合。该框架采用双层进化架构：
- **外层循环（Outer Loop）**：由 LLM 驱动，负责启发式算法的生成、重组与演化。
- **内层循环（Inner Loop）**：**无 LLM 参与**，通过基于基因编码的参数化分布与边界投影机制，在 $B_\epsilon(S)$ 内高效搜索对当前最优启发式的“最坏情况”实例。

### 相比现有方法的优势
- **更强的跨分布泛化能力**：RAISE 显著提升了在未知或偏移分布上的鲁棒性，而无需依赖预定义的多样化训练集。
- **更高的效率与可控性**：内层对抗搜索不依赖 LLM 生成实例，避免了高昂的推理成本和不可控性。
- **更少的数据需求**：仅需少量（如 5 个）名义训练实例即可实现强鲁棒性，而部分基线（如 EoH-S）需要 128 个多样化实例。
- **工程可操作性强**：提出的实例级不确定性集合 $B_\epsilon(S)$ 为 Distributionally Robust Optimization (DRO) 提供了一种无需分布假设的近似实现方式。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在三个在线组合优化任务上进行，共覆盖 **95 个测试数据集**，涉及 **5 种分布族** 和多种问题规模：

| 任务 | 数据集数量 | 分布族 | 规模变化 |
|------|-----------|--------|---------|
| **Online Bin Packing (OBP)** | 60 | Uniform, Normal, Lognormal, Exponential, Triangular | $n \in \{1k, 5k, 10k\}$, $C \in \{100, 200, 300, 400\}$ |
| **Online Job Shop Scheduling (OJSP)** | 20 | 同上 | $m \in \{10, 20\}$, $j \in \{20, 50\}$ |
| **Online Vehicle Routing (OVRP)** | 15 | 同上 | $v \in \{5, 10, 15\}$ |

所有方法均在 **5 个 Weibull 分布的名义实例** 上训练。

### 实验设置和评估指标
- **LLM 模型**：GPT-5-mini
- **预算**：1,000 次 LLM 调用，种群大小 10
- **RAISE 参数**：刷新间隔 $T=5$，内层种群 $P_{in}=8$，内层代数 $G_{in}=4$，$\epsilon \in \{0.001, 0.002, 0.005, 0.010\}$（OBP 上消融）
- **评估指标**：
  - **OBP**：Waste Ratio（越低越好）
  - **OJSP**：Normalized Makespan（越低越好）
  - **OVRP**：Route-Length Ratio（越低越好）

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **经典启发式** | BestFit, FirstFit (OBP); SPT, MinSlack, EDD (OJSP); NF-Insert, SP-Insert, UW-Insert (OVRP) |
| **标准 LLM-AHD** | EoH, ReEvo, PartEvo |
| **鲁棒感知 LLM-AHD** | EoH-S, MoH |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### OBP 平均 Waste Ratio（%）对比（Table 2）
| Method | Avg. Waste Ratio |
|--------|------------------|
| **RAISE ($\epsilon=0.002$)** | **1.671** |
| EoH-S | 1.828 |
| BestFit | 1.948 |
| ReEvo | 3.790 |
| EoH | 4.776 |

> 在 Exponential 分布下，RAISE ($\epsilon=0.005$) 达到 **0.273%**，相比 EoH 的 5.303% 提升 **19.4×**。

#### OJSP 平均 Normalized Makespan（Table 3）
| Method | 10-machine | 20-machine |
|--------|------------|------------|
| **RAISE** | **1.2266** | **1.2390** |
| EoH | 1.2284 | 1.2539 |
| ReEvo | 1.2422 | 1.2500 |

#### OVRP 平均 Route-Length Ratio（Table 3）
| Method | 5-vehicle | 10-vehicle | 15-vehicle |
|--------|-----------|-----------|-----------|
| **RAISE** | **0.8931** | **0.9263** | 0.9570 |
| EoH | 0.9017 | 0.9273 | **0.9561** |
| ReEvo | 0.8966 | 0.9287 | 0.9613 |

### 与基线方法的对比结果
- 所有标准 LLM-AHD 方法（EoH, ReEvo, PartEvo）在分布偏移下性能**急剧退化**，甚至不如经典启发式（如 BestFit）。
- MoH 尽管采用元优化，仍因依赖多任务训练而无法应对未见分布。
- **RAISE 在 8/12 个 OBP 规模配置中排名第一**，其余 4 个为第二，且优势随问题规模增大而增强。
- 在 OJSP 和 OVRP 上，RAISE 在多数配置中表现最佳，尤其在复杂度更高的 20-machine 和 5-vehicle 场景中优势明显。

### 消融实验结果（Table 4）
| 消融变体 | Avg. Waste Ratio | 相对下降 |
|----------|------------------|----------|
| **RAISE (完整版)** | **1.95** | — |
| w/o e-Mapping | 5.96 | ↑ +205% |
| w/o Base Distributions | 3.98 | ↑ +104% |
| w/o Distribution Constraints | 2.01 | ↑ +3% |
| w/o Robust Search | 3.60 | ↑ +85% |

> **e-Mapping（边界投影）** 是最关键组件，缺失会导致在 Triangular 分布下性能从 2.43% 恶化至 10.23%。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **分布偏移是 LLM-AHD 的致命弱点**：现有方法在 OOD 下性能可下降高达 **19×**。
2. **RAISE 实现了强鲁棒性**：通过将对抗性实例搜索嵌入进化循环，RAISE 在所有测试分布和问题规模上均保持领先。
3. **内层搜索机制至关重要**：
   - **九基分布编码** 提供了丰富的分布多样性。
   - **$\epsilon$-ball 投影** 防止过拟合极端分布，平衡鲁棒性与名义性能。
4. **RAISE 具有良好的扩展性**：其优势在大规模问题中更为显著，表明对抗搜索在复杂场景中更具价值。

### 方法的局限性
- 当前模型仅处理**单维分布偏移**，尚未支持多维相关偏移。
- 实验局限于**在线组合优化**任务，未验证于离线或混合设置。
- 内层搜索虽免 LLM，但仍需大量黑盒评估，计算开销较高。

### 未来工作方向
- 扩展至**多维、相关分布偏移**建模。
- 探索在**离线优化、强化学习策略设计**等更广任务类上的应用。
- 结合**梯度估计或代理模型**以降低内层搜索的评估成本。
- 研究如何自适应地调整 $\epsilon$ 以实现动态鲁棒性控制。

--- 

> **总结**：RAISE 为 LLM-based AHD 提供了一条通往**实用化、鲁棒化**的新路径，通过引入**受约束的对抗实例搜索**，有效解决了现有方法在分布偏移下的脆弱性问题，是迈向真正可部署自动化启发式设计的重要一步。

</details>

---

### 14. [Relational and Sequential Conformal Inference for Energy Time Series over Graphs via Foundation Models](https://arxiv.org/abs/2606.31804)

**Authors**: Keivan Faghih Niresi, Alice Cicirello, Olga Fink  
**Category**: cs.LG  
**Published**: 2026-07-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.31804v1  

#### Abstract
Accurate energy demand forecasting is essential for the reliable operation and planning of modern sustainable energy systems. Spatial-temporal graph neural networks (STGNNs) have recently achieved strong performance in point forecasting by jointly modeling temporal dynamics and relational dependenci...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题**: *Relational and Sequential Conformal Inference for Energy Time Series over Graphs via Foundation Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在现代可持续能源系统中，准确的能源需求预测至关重要，但仅提供**point forecast**是不够的。实际应用需要可靠的**uncertainty quantification**以支持风险感知决策、电网稳定性和不确定性下的运营规划。

然而，现有的 **Conformal Prediction (CP)** 方法通常假设样本间独立同分布（exchangeability），这在具有复杂时空依赖性的能源网络中难以成立。同时，许多图神经网络（STGNN）虽能提升预测精度，但缺乏对预测不确定性的量化能力。

此外，传统不确定性建模方法如贝叶斯神经网络或高斯过程计算成本高，且对分布假设敏感，在非平稳、多节点耦合的场景下表现不佳。

---

### 🚀 提出的新方法：STOIC
作者提出 **STOIC**（**Spatial-Temporal Graph Conformal Prediction with In-Context Learning**），一种结合图神经网络与**tabular foundation model**的新型 conformal inference 框架。

#### 核心思想：
- **解耦设计**：将 point forecasting 与 uncertainty quantification 分离。
- **两阶段流程**：
  1. 使用 **STGNN** 进行空间-时间点预测；
  2. 将预测残差（residuals）转化为结构化表格特征，利用预训练的 **tabular foundation model**（如 TabPFN）进行零样本（zero-shot）校准，生成自适应的 prediction intervals（PIs）。

#### 创新点：
1. **首次将 in-context learning 引入图结构时间序列的 conformal calibration**；
2. 设计了一种新的**时空残差特征工程方法**，融合了：
   - 时间特征（rolling mean, median 等）
   - 图结构特征（通过 permutation-invariant 聚合函数提取邻居残差信息）
3. 实现了无需任务特定微调的**可迁移、高效校准机制**，适用于异构能源网络；
4. 在保持 STGNN 预测能力的同时，增强了其不确定性估计的可靠性与灵活性。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法局限 | STOIC 改进 |
|------|----------------|------------|
| **时空依赖建模** | 多数 CP 忽略空间相关性或仅处理静态空间 | 显式建模时空残差结构 |
| **模型依赖性** | 需为每个任务重新训练 | 利用 tabular FM 实现 zero-shot calibration |
| **校准效率** | 数据需求大，收敛慢 | 小样本下仍可实现良好覆盖 |
| **区间适应性** | 区间宽度固定或仅随时间变化 | 输出 heteroscedastic、context-aware PIs |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
共五个数据集，涵盖合成与真实世界场景：

| 数据集 | 类型 | 节点数 | 时间步长 | 描述 |
|-------|------|--------|----------|------|
| **SCS** (Simulated Control Signals) | 合成 | 10 | 3650天 | 控制信号驱动的非线性动态，信噪比固定 |
| **SDH** (Synthetic District Heating) | 合成 | 100 | ~3652天 | 基于 `demandlib` 模拟的区域供热网络 |
| **EWZ** | 真实 | 48 | 3501天 | 苏黎世市供电公司提供的真实供热数据 |
| **ECL** (Electricity Consumption Load) | 真实 | 40 | 1461天 | 葡萄牙电力客户用电数据（聚合至日） |
| **CKW** | 真实 | 93 | 1863天 | 瑞士邮政编码级智能电表聚合数据 |

所有任务均为 **one-day-ahead forecasting**，输入窗口长度为 7 天。

---

### 🧪 实验设置
- **划分方式**：按时间顺序划分为 70% 训练 / 15% 校准 / 15% 测试；
- **标准化**：仅基于训练集统计量进行归一化；
- **目标置信水平**：90%（即 $1-\alpha=0.9$）；
- **point forecaster**：统一采用 STGNN 架构（含 learnable graph structure）；
- **tabular FM**：使用 **TabPFN** 作为默认校准器；
- **对比基线**均使用相同 forecaster，确保公平比较。

---

### 📈 评估指标
| 指标 | 公式 | 目标 |
|------|-----|------|
| **Coverage (%)** | $ \mathbb{E}[I(x_{i,t} \in C_{i,t})] $ | 接近目标值（如 90%） |
| **PI-Width** | $ u - l $ | 越小越好（更 sharp） |
| **Winkler Score ↓** | 综合惩罚未覆盖 + 宽度过大 | 越低越好 |

---

### ⚔️ 对比的基线方法
| 方法 | 类型 | 特点 |
|------|------|------|
| **SCP** (Split Conformal Prediction) | 全局 | 忽略时空异质性 |
| **ACI** (Adaptive Conformal Inference) | 时间自适应 | 动态调整区间宽度 |
| **SPCI** (Sequential Predictive CP) | 序列感知 | 基于历史误差更新 |
| **LSCP** (Localized Spatial CP) | 局部空间 | 引入空间邻域但忽略时序 |
| **STCP** (Spatio-Temporal CP) | 图结构 | 结合时空建模，需重新训练 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能汇总（取代表性结果）

| 方法 | 数据集 | Coverage (%) | PI-Width | Winkler Score |
|------|--------|--------------|---------|----------------|
| SCP | ECL | 93.65 | 1288.53 | 1874.88 |
| SPCI | ECL | 89.44 ❌ | 1199.82 | 1929.06 |
| LSCP | ECL | 89.27 ❌ | 1137.70 | 1783.94 |
| **STOIC** | **ECL** | **90.52 ✅** | **1168.54** | **1674.85** |
| SCP | CKW | 91.80 ✅ | 3001.00 | 3916.29 |
| STCP | CKW | 89.77 ❌ | 2123.03 | 2936.83 |
| **STOIC** | **CKW** | **90.89 ✅** | **2156.48** | **2927.36** |
| SCP | SCS | 88.13 ❌ | 0.0490 | 0.0826 |
| **STOIC** | **SCS** | **91.57 ✅** | **0.0309** | **0.0435** |

> ✅ 表示达到或超过目标覆盖率；❌ 表示 under-coverage

---

### 🔍 主要发现
- **STOIC 是唯一在所有数据集上 consistently 达到或接近 90% coverage 的方法**；
- 在多个数据集（如 SCS, ECL, CKW）上，STOIC 实现了**最窄的 PI-Width 和最低的 Winkler Score**；
- 即使在强季节性噪声环境下（如 EWZ），STOIC 也能维持稳定的 coverage 并避免过度保守；
- **STCP** 虽然尝试建模时空结构，但在部分数据集（如 CKW）出现明显 under-coverage；
- **LSCP/SPCI** 等局部方法因忽略跨维度依赖而无法泛化。

---

### 🔍 消融实验结果（Ablation Study）

在 **EWZ** 和 **CKW** 上进行了以下消融：

| 配置 | 数据集 | Coverage (%) | PI-Width | Winkler Score |
|------|--------|--------------|----------|----------------|
| Full STOIC | EWZ | 90.15 ✅ | 0.3947 | 0.6478 |
| w/o Graph Features | EWZ | 89.96 ❌ | 0.3953 | 0.6563 |
| w/o Temporal Features | EWZ | 90.21 ✅ | 0.3980 | 0.6727 |
| w/o TabPFN (→ QRF) | EWZ | 90.21 ✅ | 0.4503 ↑↑ | 0.7186 ↑↑ |
| Full STOIC | CKW | 90.89 ✅ | 2156.48 | 2927.36 |
| w/o Graph Features | CKW | 90.72 ✅ | 2194.87 | 3002.30 |
| w/o Temporal Features | CKW | 90.73 ✅ | 2186.19 | 2961.14 |
| w/o TabPFN (→ QRF) | CKW | 89.09 ❌ | 2139.63 | 2981.68 |

#### 发现：
- 移除 **graph-based features** 对 CKW 影响更大 → 表明**空间依赖在区域聚合层面更重要**；
- 移除 **temporal features** 对 EWZ 影响更大 → 表明**个体建筑级预测更依赖自身历史模式**；
- 替换 **TabPFN → Quantile Random Forest (QRF)** 导致显著性能下降，尤其在 Winkler Score 上；
  - 说明 **foundation model 的 inductive bias 和 in-context learning 能力至关重要**；
  - QRF 难以有效捕捉高维、相关性强的 engineered features。

---

### 📉 小样本敏感性分析（Sensitivity to Calibration Size）
在 SCS 上测试不同校准数据比例（10% ~ 100%）的影响：

- **即使只有 10% 校准数据**，STOIC 仍能达到 **88.92% coverage**，接近目标；
- 随着校准数据增加：
  - Coverage 稳定趋近并略超 90%；
  - PI-Width 持续下降；
  - Winkler Score 不断改善；
- 表明 STOIC 具有极强的**数据效率**，适合部署初期数据稀缺场景。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **STOIC 成功解决了图结构能源时间序列中的 conformal inference 挑战**：
   - 通过将残差映射为结构化表格输入，实现了对复杂时空依赖的有效建模；
2. **Tabular foundation model 可用于 zero-shot、跨任务的 uncertainty calibration**：
   - 无需重新训练即可适配新节点或新网络；
3. **相比传统 CP 与近期图结构方法，STOIC 实现了更优的 calibration-efficiency trade-off**：
   - 更可靠 coverage + 更窄区间 → 更实用的 uncertainty estimates；
4. **消融实验证明各组件均有贡献，其中 TabPFN 的作用最大**；
5. **与 times-series foundation model（如 TimesFM）对比显示**：
   - 大规模预训练本身不足以保证 coverage；
   - 显式 conformal calibration 仍是获得统计保障的关键。

---

### ⚠️ 方法局限性
| 局限 | 说明 |
|------|------|
| **依赖高质量 point forecaster** | 若 STGNN 本身偏差大，则残差建模效果受限 |
| **feature engineering 手动设计** | 当前特征选择依赖经验，未来可探索自动学习 |
| **TabPFN 的理论保证有限** | 其一致性（consistency）未被严格证明，依赖 synthetic prior |
| **仅支持单步预测** | 未扩展至 multi-horizon forecasting 场景 |

---

### 🔮 未来工作方向
1. **引入外部变量**（如天气警报、节假日）作为 in-context prompt 输入；
2. **拓展至 multi-step ahead forecasting**，构建动态演化 PIs；
3. **探索其他 tabular 或 hybrid foundation models** 以进一步提升性能；
4. **应用于更多物理系统**（如交通流、水网监测）验证通用性；
5. **研究 online adaptation 机制**，应对长期分布漂移。

---

## 总结
> **STOIC 是首个将 tabular foundation model 与图结构 conformal prediction 相结合的框架**，它通过**解耦 forecasting 与 calibration**，利用 **in-context learning** 实现了高效、可靠、可迁移的不确定性量化。实验表明其在多种真实与合成能源数据上均优于主流 baseline，尤其在 coverage reliability 与 interval sharpness 之间取得了最佳平衡，为安全关键型能源系统的智能运维提供了有力工具。

</details>

---

### 15. [A Three-Phase Foundation Model for Tax-Aware Personalized Portfolio Management](https://arxiv.org/abs/2606.30997)

**Authors**: Ramin Pishehvar  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.30997v1  

#### Abstract
We present a three-phase deep reinforcement learning system for personalized portfolio management that addresses three limitations shared by all prior financial RL work: 1) ticker lock-in, 2) monolithic objectives , and 3) static user models. Phase 1 pretrains a ticker-identity-free cross asset enco...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心总结：A Three-Phase Foundation Model for Tax-Aware Personalized Portfolio Management

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

该论文旨在解决当前金融领域 **Reinforcement Learning (RL)** 在个性化投资组合管理中的三大共性缺陷：

1. **Ticker Lock-in**：传统模型在固定资产池（如 S&P 500）上训练，无法泛化到新资产，缺乏零样本（zero-shot）能力。
2. **Monolithic Objectives**：几乎所有现有 RL 系统都优化单一目标（如 Sharpe Ratio），无法满足用户多样化的投资目标（如短期收益、资本保全、税务优化等）。
3. **Static User Models**：用户偏好通过一次性问卷获取，无法动态更新，忽略了真实交易行为所揭示的“revealed preferences”。

---

### 🚀 提出的新方法与创新思路

论文提出一个**三阶段深度强化学习系统**，整合多个前沿技术，实现真正个性化的、税务感知的投资组合管理。

#### 主要创新点如下：

| 创新点 | 描述 |
|--------|------|
| **Ticker-Identity-Free 架构** | 彻底移除固定的 ticker embedding，代之以 **50维可观测元数据向量**（sector, fundamentals, analyst consensus, options signals, earnings calendar, insider sentiment, institutional ownership），使模型可零样本泛化至任何公开交易资产，无需重新训练。 |
| **Objective-Conditioned Reward + MoE 架构** | 引入多目标奖励机制，在每个 episode 随机采样六种投资目标之一：<br>• `MAX_GAIN_30D`<br>• `MAX_GAIN_1Y`<br>• `CAPITAL_PRESERVE`<br>• `INCOME_HARVEST`<br>• `LT_GAIN_ONLY`<br>• `ALPHA_VS_EW`<br><br>采用 **Mixture-of-Experts (MoE)** 架构，四个专家头分别专精于：<br>• Momentum<br>• Growth<br>• Defensive<br>• Tax-aware<br><br>由一个**可学习的 intent router** 根据当前目标和市场状态动态融合专家输出，消除跨目标梯度冲突。 |
| **Chronos 时间序列基础模型融合** | 首次将冻结的 **Chronos**（基于 T5 的时间序列基础模型，预训练于超 1000 亿数据点）作为并行编码分支，通过**可学习门控机制**（learned gating）与领域特定的 SSL 编码器融合，增强通用时序模式识别能力。 |
| **Revealed-Preference Personalization via LoRA** | 第三阶段引入轻量级 **76参数 LoRA 模块**，仅微调 intent router 权重，从用户的**真实交易历史**中推断其风险偏好、持有周期、税务敏感度等，实现动态个性化，避免依赖主观问卷。 |
| **自然语言目标解析（NL Intent Parser）** | 支持用户输入自由文本目标（如 “买房子在3年后”、“孩子10岁，为大学基金”），自动映射为结构化投资参数，提升用户体验。 |
| **Cash Token 与 Allocation-Driven Execution** | 引入**可学习现金 token**，强制模型主动决策现金配置，防止被动持有；解耦 allocation 与 action 头，通过阈值触发 rebalance，防止 HOLD 锁死。 |

---

### 🔍 相比现有方法的优势

| 维度 | 本文系统 | 传统机构系统 | 零售工具（Robo-Advisor） |
|------|----------|--------------|-------------------------|
| 学习型市场信号 | ✅ | ✅ | ❌（规则引擎） |
| 税务感知（Tax-lot aware） | ✅（个性化税率优化） | ⚠️（部分支持） | ⚠️（基本规则） |
| 用户目标个性化 | ✅（动态行为推断） | ❌（静态策略） | ⚠️（一次问卷） |
| 自然语言输入 | ✅ | ❌ | ❌ |
| 实时券商集成 | ✅ | ✅ | ✅ |
| 所需专业知识 | None | High | None |
| 数据成本 | Free | $$$ | Free |
| 适应新资产 | ✅（Zero-shot） | ❌（需重训） | ❌ |

> ✅ **核心优势**：首次实现**免重训练资产泛化 + 多目标动态路由 + 行为驱动个性化 + 税务优化**的一体化系统。

---

## 2. 核心实验方法和设置

### 📊 数据集

- **Phase 1 预训练数据**：
  - 30只 S&P 500 成分股（覆盖全部11个 GICS 行业）
  - 日频数据，时间跨度：2015–2024
  - 特征包括：价格、成交量、RSI、MACD、移动平均比率、z-score 等
- **元数据来源**：Yahoo Finance API 获取 50 维 observable metadata
- **Chronos Embedding**：预先计算并缓存，使用 Chronos-T5-Small（46M 参数）

### ⚙️ 实验设置

- **环境**：自定义 Portfolio Environment，支持：
  - 动作空间：`BUY`, `HOLD`, `SELL` + 分配权重
  - 状态空间：价格窗口 + 元数据 + 市场状态
  - 回报函数：Objective-conditioned reward
- **算法**：PPO（Proximal Policy Optimization）
- **训练流程**：
  1. **Phase 1**：自监督预训练（Next-bar Return, Masked Feature Recovery, Market Regime Classification）+ **Inter-Ticker Contrastive Loss**
  2. **Phase 2**：MoE 架构微调，采用 **Curriculum Learning**（分阶段训练专家头 → 路由器 → 联合微调）
  3. **Phase 3**：LoRA 微调 intent router，适配个体用户

### 📈 评估指标

| 指标 | 定义 |
|------|------|
| **Alpha vs Equal-Weight (EW)** | 投资组合收益减去等权基准收益，衡量选股能力 |
| **Alpha vs SPY** | 对比标普500 ETF，反映相对表现 |
| **Total Return** | 14天累计回报 |
| **Annualized Sharpe Ratio** | 年化夏普比率（短窗口下可能为负，不作为主指标） |
| **Max Drawdown** | 最大回撤 |
| **Win Rate (Daily)** | 日胜率 |
| **Weight Std** | 权重标准差，衡量分配差异化程度 |
| **Router Differentiation (std)** | 路由器对不同意图的区分度（目标 > 0.05） |

### 🆚 基线方法对比

| 基线 | 描述 |
|------|------|
| **Collapsed Representation** | 无 contrastive loss 的编码器（cosine sim ≈ 0.96），导致均匀分配 |
| **Single-head Policy** | 单一 allocation head，同时训练所有目标（梯度冲突） |
| **MoE (ours)** | 本文提出的 MoE + Curriculum + Grafting 架构 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（14天回测，2026年6月）

| 模型 | Total Return | Alpha vs EW | Alpha vs SPY | Max DD | Weight Std |
|------|-------------|------------|-------------|--------|-----------|
| Collapsed (sim=0.96) | -5.95% | -0.74% | -3.53% | -5.87% | 0.000 |
| Single-head (ALPHA_VS_EW) | -5.30% | **+2.70%** | -2.39% | -5.66% | 0.057 |
| **MoE (ours)** | **-5.15%** | **+2.92%** | **-2.32%** | **-5.37%** | 0.057 |

> ✅ **MoE 模型在所有指标上均优于基线**，尤其在 Alpha vs EW 上领先 +0.22pp。

---

### 🔬 消融实验与关键发现

#### （1）Inter-Ticker Contrastive Loss 的作用

| 配置 | Val Loss | Cosine Sim | 结果 |
|------|---------|-----------|------|
| 无 Contrastive | 0.143 | 0.96 | 表示坍塌，权重均匀 |
| + Contrastive (warm-start) | **0.163** | **0.24** | 成功区分资产，产生非均匀权重 |

> 📉 图3显示：加入 contrastive loss 后，相似度从 0.96 快速下降至 0.24，验证其有效性。

#### （2）MoE Curriculum Learning 效果

| 阶段 | 专家 | 14d Alpha (vs EW) | 说明 |
|------|------|------------------|------|
| S1: Momentum | Momentum | +3.16% | 短期动量有效 |
| S2: Growth | Growth | **+3.37%** | 中期增长最优 |
| S3: Defensive | Defensive | +3.18% | 长期防御更稳 |
| S4: Router-only | Frozen Experts | +0.44% | ❌ 失败（flat loss surface） |
| S5: Joint Training | All | 路由 std > 0.37 | ✅ 成功学习意图路由 |
| S6: Grafted MoE | Best Experts + S5 Router | **+3.03%** | ✅ 恢复专家能力，保持高路由精度 |

> ✅ **Expert Grafting** 是关键：联合训练会损害专家专业化，grafting 可恢复性能。

#### （3）Intent Router 学习效果（表9）

- 所有6种意图均能**精准路由**至对应专家（权重=1.00）
- 不同专家产生**显著不同的 top holdings**：
  - Momentum → JPM
  - Growth → TSLA
  - Defensive → GOOGL
  - Tax-aware → AMZN
- 路由标准差 **> 0.37**，远高于阈值 0.05，表明路由高度分化。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **Ticker-Identity-Free 设计可行且高效**：使用 50 维 observable metadata 替代 ticker embedding，实现真正的零样本资产泛化，是金融 RL 的架构突破。
2. **MoE + Objective-Conditioned Reward 可解决多目标冲突**：通过专家分工与 intent router，单一策略可服务多种投资目标，避免梯度干扰。
3. **Chronos 基础模型增强通用时序理解**：首次成功将冻结的 TS 基础模型用于 Portfolio RL，提供跨领域时序先验。
4. **Revealed Preference + LoRA 实现轻量个性化**：仅用 76 参数即可从交易历史推断用户偏好，隐私友好、部署成本低。
5. **Contrastive Loss 至关重要**：标准 SSL 目标会导致表示坍塌（representation collapse），必须引入 inter-ticker contrastive loss 才能实现差异化分配。

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **数据范围有限** | 当前实验仅使用 10 只股票，未扩展至 S&P 500 或全市场 |
| **回测周期短** | 仅在 2026 年 6 月的 14 天下行市场测试，缺乏多市场周期验证 |
| **Tax Lot 数据精度依赖 API** | 多数券商仅提供平均成本法，无法精确优化每笔 tax lot |
| **分析师数据延迟** | yfinance 数据可能滞后 1-2 天，影响事件驱动策略 |
| **Phase 3 尚未实证** | LoRA 个性化模块尚未在真实用户数据上全面评估 |

---

### 🔮 未来工作方向

1. **Phase 3 实证评估**：在合成与真实交易数据上验证 LoRA 个性化效果。
2. **MoE 多窗口训练**：为不同专家设计不同时间窗口目标（如 momentum 用 14d，growth 用 90d），提升长期鲁棒性。
3. **Online Fine-tuning**：每周微调 router 和 allocation 头，适应市场 regime shift。
4. **扩展至更大资产池**：使用稀疏注意力或行业聚类处理 S&P 500 规模。
5. **新闻与事件集成**：验证 news cross-attention 对 event-driven alpha 的提升。
6. **Live Paper Trading**：部署在模拟账户进行长期实盘测试。

---

> 💡 **总结**：本文构建了一个**端到端、可部署、个性化、税务感知**的 Portfolio Management RL 系统，结合 **foundation model、MoE、LoRA、contrastive learning** 等多项技术，在架构设计与实际性能上均有显著突破，为下一代智能投顾提供了新范式。

</details>

---

### 16. [Learning to Select, Not Relearn: Hard-Routed Mixtures of Reasoning LoRAs](https://arxiv.org/abs/2606.31413)

**Authors**: Seyed Alireza Molavi, Zhan Su, Yan Hu, Peyman Sheikholharam Mashhadi, Stefan Byttner, Prayag Tiwari  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.31413v1  

#### Abstract
Composing independently trained LoRA adapters into a single large language model is useful for multi-domain adaptation, especially when the original training data cannot be shared. A common approach is to use MoE-style routing over LoRA experts, but for frozen pretrained adapters, soft weighted comb...

---

### 17. [Evo-PI: Aligning Medical Reasoning via Evolving Principle-Guided Supervision](https://arxiv.org/abs/2606.31800)

**Authors**: Xianda Zheng, Huan Gao, Meng-Fen Chiang, Michael Witbrock, Kaiqi Zhao, Shangyang Li  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.31800v1  

#### Abstract
Despite recent progress, the reasoning capabilities of large multimodal language models (MLLMs) remain fundamentally constrained by static supervision, where fixed prompts, rules, or reward models provide non-adaptive guidance throughout training. Such static signals are often sufficient to enforce ...

---

### 18. [Dualformer: Efficient Feature Extractor for Complex-valued Blind Communication Signal Analysis](https://arxiv.org/abs/2606.31352)

**Authors**: Yurui Zhao, Xiang Wang, Jingreng Lei, Wanlong Zhang, Yik-Chung Wu, Zhitao Huang  
**Category**: cs.LG  
**Published**: 2026-07-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.31352v1  

#### Abstract
Designing effective feature extractors is critical for blind signal analysis tasks such as automatic modulation recognition (AMR), signal scheme recognition (SSR), and \color{black} signal structure parsing (SSP). In this work, we propose dual-channel neural network (DualNN) that efficiently exploit...

---

### 19. [Beyond the Expressivity-Trainability Paradox: A Dynamical Lie Algebra Perspective on Navigating Barren Plateaus in Quantum Machine Learning](https://arxiv.org/abs/2606.31536)

**Authors**: Kung-Ming Lan  
**Category**: cs.LG  
**Published**: 2026-07-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.31536v1  

#### Abstract
As Quantum Machine Learning (QML) transitions toward practical implementation, the field faces a critical architectural bottleneck that challenges the fundamental assumptions of classical statistical learning theory. In classical deep learning, increasing model capacity typically risks overfitting. ...

---

### 20. [Scenario Generation for Testing of Autonomous Driving Systems Using Real-World Failure Records](https://arxiv.org/abs/2606.31131)

**Authors**: Anjali Parashar, Chuchu Fan  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.31131v1  

#### Abstract
To ensure safe on-road behavior, pre-deployment testing and failure discovery of Autonomous Driving Systems (ADS) is crucial. Present day simulation based testing methods focus largely on mathematical models for efficient search of optimal scenarios, assuming a fixed scenario representation. On the ...

---

### 21. [CryoACE: An Atom-centric Framework for Accurate and Automated Model Building in Cryo-EM](https://arxiv.org/abs/2606.31332)

**Authors**: Minzhang Li, Mingrui Li, Weichen Qin, Qihe Chen, Sixian Shen, Yuan Pei, Jiakai Zhang, Jingyi Yu  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.31332v1  

#### Abstract
Protein automodeling from cryo-EM density maps faces unique challenges in enforcing physicochemical validity and managing conformational heterogeneity. Current solvers are often limited to static predictions or require computationally intensive heuristic searches. We present CryoACE, an end-to-end f...

---

### 22. [ReGRPO: Reflection-Augmented Policy Optimization for Tool-Using Agents](https://arxiv.org/abs/2606.31392)

**Authors**: Binjie Zhang, Mike Zheng Shou  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.31392v1  

#### Abstract
Tool-augmented vision-language models (VLMs) can solve multimodal, multi-step tasks by calling external tools, yet they remain fragile in practice. Existing works have two common gaps. Supervised fine-tuning (SFT) is built mostly on successful trajectories and offers little signal for recovery after...

---

### 23. [Which Tokens Matter? Adaptive Token Selection for RLVR with the Relative Surprisal Index](https://arxiv.org/abs/2606.31575)

**Authors**: Outongyi Lv, Yanzhao Zheng, Yuanwei Zhang, Zhenghao Huang, Xingjun Wang, Baohua Dong, Hangcheng Zhu, Yingda Chen  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.31575v1  

#### Abstract
Reinforcement learning (RL) has become a powerful tool for propelling Large Language Models (LLMs) beyond imitation-based training towards more robust reasoning capabilities. Among existing approaches, RL with Verifiable Rewards (RLVR) has emerged as a pivotal paradigm for advancing LLM reasoning. D...

---

### 24. [Beyond Clean Text: Evaluating Encoder and Decoder Robustness for Bangla Event Detection in Noisy Text](https://arxiv.org/abs/2606.30914)

**Authors**: Tanvir Ahmed Sijan, S. M Golam Rifat, Nayeemul Islam, Md. Musfique Anwar  
**Category**: cs.CL  
**Published**: 2026-07-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.30914v1  

#### Abstract
Event detection (ED) systems are typically evaluated on clean, curated text, leaving their robustness to real-world noise largely unexplored, particularly for low-resource languages such as Bangla. We introduce a generalized Bangla news event ontology and a benchmark comprising 9,979 annotated sente...

---

### 25. [Building an ASR Solution for Training and Assessing Children's Reading](https://arxiv.org/abs/2606.31508)

**Authors**: Yacouba Diarra, Nouhoum Souleymane Coulibaly, Mamadou Dembele, Aymane Dembele, Michael Leventhal  
**Category**: cs.CL  
**Published**: 2026-07-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.31508v1  

#### Abstract
Automatic speech recognition for children's reading remains underdeveloped for most African languages, including Bambara, despite its potential value for reproducible literacy assessment. We present an open-source system for assessing children's reading in Bambara, developed through an end-to-end pr...

---

### 26. [Joint discovery of governing partial differential equations from multi-source datasets by competitive optimization](https://arxiv.org/abs/2606.30699)

**Authors**: Hao Xu, Siyu Lou, Yuntian Chen, Dongxiao Zhang  
**Category**: cs.LG  
**Published**: 2026-07-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.30699v1  

#### Abstract
Discovering governing equations directly from observational data is a key step towards interpretable scientific machine learning. Current data-driven approaches typically operate on a single dataset, inherently limiting their performance when faced with restricted observations. In practice, multiple...

---

### 27. [Predictable GRPO: A Closed-Form Model of Training Dynamics](https://arxiv.org/abs/2606.30789)

**Authors**: Rajat Ghosh, Datta Nimmaturi, Aryan Singhal, Vaishnavi Bhargava, Henry Wong, Johnu George, Debojyoti Dutta  
**Category**: cs.LG  
**Published**: 2026-07-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.30789v1  

#### Abstract
Group Relative Policy Optimization (GRPO) has become a standard tool for improving the reasoning ability of large language models, yet its training dynamics are still described empirically: reward trajectories are fit with low-parameter functional forms whose constants carry no mechanistic meaning, ...

---

### 28. [Warp RL: Reshaping Base Policy Distributions for Dynamics Adaptation](https://arxiv.org/abs/2606.31043)

**Authors**: Ethan Hirschowitz, Fabio Ramos  
**Category**: cs.LG  
**Published**: 2026-07-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.31043v1  

#### Abstract
Residual reinforcement learning adapts a pretrained robot policy by learning an additive correction to its actions. While effective when adaptation amounts to shifting the base policy's action distribution, additive corrections cannot change the distribution's shape, scale, or state-dependent geomet...

---

### 29. [Addressing Over-Refusal in LLMs with Competing Rewards](https://arxiv.org/abs/2606.31748)

**Authors**: Taeyoun Kim, Aviral Kumar  
**Category**: cs.LG  
**Published**: 2026-07-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.31748v1  

#### Abstract
Safety training on language models often induces over-refusal: improved safety on harmful prompts at the cost of increased refusal on harmless ones. Though this trade-off can be mitigated by training models with reinforcement learning (RL) to reason before answering, it does not remove the underlyin...

---

### 30. [Neuro-Bayesian-Symbolic Residual Attention Shallow Network: Explainable Deep Learning for Cybersecurity Risk Assessment](https://arxiv.org/abs/2606.30953)

**Authors**: Nicolaie Popescu-Bodorin, Madeleine Togher  
**Category**: cs.AI  
**Published**: 2026-07-01  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2606.30953v1  

#### Abstract
We introduce the Neuro-Bayesian-Symbolic Residual Attention Shallow Network (NBS-RASN), a hybrid neural architecture for explainable cybersecurity risk assessment in open-source ecosystems. Unlike deep models that trade interpretability for accuracy, our shallow network encodes domain knowledge, cau...

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

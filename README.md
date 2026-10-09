# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-09 12:09:27 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [QUILT: Rethinking Sparse-Attention Prefill through Shared Query Execution](https://arxiv.org/abs/2610.11134)

**Authors**: Zhenduo Zhao, Qihui Zhou, Mingcong Song, Zhiyi Chen, Chuangguan Ye, Fengfan Hou, Zequn Gong, Jing Li, Hongjie Si, Guoping Long  
**Category**: cs.LG  
**Published**: 2026-10-09  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.11134v1  

#### Abstract
Sparse attention reduces the cost of long-context attention, but existing kernels typically process queries independently, repeatedly loading and dequantizing KV entries shared across queries. We observe substantial overlap in the KV entries selected by neighboring queries, creating opportunities fo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# QUILT: Rethinking Sparse-Attention Prefill through Shared Query Execution 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在 **long-context LLM inference** 中，**prefill 阶段**的 **attention 计算** 成为性能瓶颈，其时间直接影响用户可见的 **Time-to-First-Token (TTFT)**。尽管 **sparse attention** 被广泛用于减少计算量和内存访问，现有的实现通常采用 **query-parallel execution**，即每个 query 独立地加载、dequantize 和处理其选中的 Top-K KV entries。

然而，作者观察到：**相邻的 query tokens 往往选择高度重叠的 KV entries**。现有方法忽略了这种跨 query 的相关性，导致大量重复的 KV 数据加载和 dequantization，造成显著的冗余内存流量和计算开销。

### 提出了什么新方法或新思路

为解决上述问题，论文提出了 **QUILT** —— 一种 **workload-aware** 的稀疏注意力执行机制，核心思想是通过 **group-based query execution** 显式利用跨 query 的 KV 重用。

#### 主要创新点包括：

- **Shift-and-Compare Set Decomposition (SCSD)**  
  一种高效的 **data-parallel 算法**，将不规则的集合操作（如求交集）转化为可并行执行的排序、移位、比较等原语，避免传统 set 操作的串行控制流开销。

- **Pipelined SCSD Execution**  
  将 SCSD 与 attention 计算流水线化，利用现代加速器上 **matrix** 和 **vector** 单元的异构并行能力，将 set decomposition 的开销隐藏在 attention pipeline 中，几乎不增加关键路径延迟。

- **Cascaded Sharing**  
  支持多粒度共享：从大组开始提取全局共享的 KV，再递归分解子组以捕获局部共享，从而适应不同层、不同数据集下变化的共享模式。

- **Tile-Aware Clipping**  
  结合硬件 tile 大小进行优化：
  - 对共享段尾部，若利用率低则“推入”下一级 cascade；
  - 对最终 query-specific 段，基于 indexer score 进行 **importance-aware clipping**，剪掉低重要性残差项，消除未充分利用的 tile，仅引入极小精度损失。

### 相比现有方法的优势

| 维度 | QUILT 优势 |
|------|-----------|
| **效率** | 显著减少冗余 KV 加载与 dequantization，降低内存带宽压力 |
| **硬件利用率** | 更大的计算粒度提升 matrix unit 利用率；tile-aware 设计提升 tile occupancy |
| **灵活性** | cascaded sharing 适应异构共享模式，无需固定 group size |
| **兼容性** | 不依赖特定 sparsification 策略（支持 training-free 与 model-native），可作为通用 kernel 优化叠加于现有方案之上 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **LongBench**：一个双语、多任务的长上下文理解基准，涵盖以下任务类型：
  - Question Answering（如 `Qasper`）
  - Document Summarization（如 `GovReport`）
  - Code Generation
  - Passage Retrieval
  - 多跳推理（如 `Musique`, `NarrativeQA`）

### 实验设置和评估指标

#### 模型
- **GLM-5.3**：采用 model-native sparse attention（Top-2048）
- **DeepSeek-3.2**：使用 DeepSeek Sparse Attention (DSA)，同样为 Top-2048

#### 硬件平台
- **16 × Ascend 910C NPU**，每卡 64GB HBM
- 使用 **CANN 9.1.0** 开发 kernel
- 支持 **tensor parallelism (TP)** 和 **sequence parallelism (SP)**

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Kernel Latency** | 稀疏 attention kernel 平均执行时间 |
| **Processed KV Data** | 实际处理的 KV entry 总数 |
| **TTFT (Time-to-First-Token)** | 端到端首 token 延迟 |
| **Accuracy** | 在 LongBench 各任务上的得分（与 baseline 对比） |

### 基线方法对比

- **Baseline**: **OPS-Transformer** 提供的高度优化稀疏 attention kernel
- 所有对比均集成至统一推理引擎 **XYServe**，确保公平比较（相同调度、内存管理等）

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ Kernel Level Improvement

| 模型 | 指标 | 最大提升 |
|------|------|--------|
| GLM-5.3 | **平均 kernel latency 降低** | **55.1%** (TP) / 43.4% (SP) |
| DeepSeek-3.2 | **平均 kernel latency 降低** | **28.7%** (TP) / 17.1% (SP) |
| GLM-5.3 | **处理的 KV 数据减少** | **55.9%** |
| DeepSeek-3.2 | **处理的 KV 数据减少** | **30.4%** |

> 表格来源：Figure 9 与 Table 3

#### ✅ End-to-End Performance

| 模型 | **TTFT 降低** |
|------|--------------|
| GLM-5.3 | **最高达 36.8%**（平均 35.6% under TP） |
| DeepSeek-3.2 | **最高达 30.2%**（平均 23.0% under TP） |

> 表明 kernel 优化有效转化为用户可感知的响应速度提升。

#### ✅ 准确性保留

- 在 21 个 LongBench 数据集上测试，**accuracy 变化极小**：
  - GLM-5.3：平均绝对差异仅 **0.42 分**
  - DeepSeek-3.2：平均绝对差异仅 **0.75 分**
- **mean absolute percentage difference** 分别为 **0.90%** 和 **2.01%**
- 说明 **importance-aware clipping 引入的近似误差可忽略**

### 消融实验结果

#### 🔹 SCSD 与 Pipelining 效果（Figure 12）

| 变体 | GLM-5.3 (TP) 相对 baseline |
|------|-----------------------------|
| Naive Set Intersection | **慢 21.84×** |
| + SCSD | **快 35.5%** |
| + Pipelined SCSD | **进一步提速 45.9%** |

> 结论：**SCSD 是高效 set decomposition 的关键，而 pipelining 是将其开销隐藏的核心手段**。

#### 🔹 Cascaded Sharing vs 固定 Group Size（Table 4）

| 策略 | GLM-5.3 处理 KV 数（相对 baseline） |
|------|-------------------------------|
| Group=2 | 65.7% |
| Group=4 | 57.2% |
| Group=8 | 59.0% |
| **Cascaded Sharing (CS)** | **45.5%** ✅ |

> 结论：**cascaded sharing 显著优于任何单一固定 group size**，能更充分挖掘多层次共享机会。

#### 🔹 Tile-Aware Clipping 性能与精度影响（Figure 14–15）

- **性能增益**：
  - GLM-5.3：kernel latency 再降 **23.7% (TP)** / **22.7% (SP)**
  - DeepSeek-3.2：**24.4% / 14.1%**
- **输出相似性**（cosine similarity）：
  - **Importance-aware clipping** 明显优于随机剪枝
  - head-level 最小相似性从 0.312 → 0.406
  - hidden-level 最小相似性从 0.664 → 0.719

> 结论：**按重要性剪枝可在大幅提效的同时更好保持输出一致性**。

---

## 4. 关键结论和发现

### 主要发现

1. **跨 query 的 KV 选择存在强相关性**，尤其在浅层和小 query 组中，**overlap ratio 超过 85%**，现有 query-parallel 执行严重浪费资源。
2. **workload-aware execution 是突破稀疏 attention 性能瓶颈的关键**，必须从“独立处理 query”转向“联合处理相关 query”。
3. **SCSD + Pipelining** 实现了近乎零开销的 set decomposition，使跨 query 共享在工程上可行。
4. **cascaded sharing** 适应性强，能动态匹配不同 workload 下的共享结构。
5. **tile-aware clipping** 在硬件层面完成最后一级优化，以极小精度代价换取显著性能收益。

### 方法的局限性

- 当前设计假设 **相邻 query 间共享性强**，若 workload 中 query 间相关性弱（如完全随机稀疏模式），收益可能下降。
- **SCSD 的排序开销** 在极短序列或极小组 size 下可能难以被掩盖。
- 目前实现针对 Ascend NPU，虽原理通用，但在其他架构（如 GPU）需适配 vector/matrix pipeline 特性。

### 未来工作方向

- 探索 **动态调整 cascade 层级与 group size**，根据运行时 workload 自适应配置。
- 将 QUILT 思想扩展至 **decoding 阶段**，利用连续 step 间的 KV selection 相似性。
- 结合 **KV cache compression** 与 **shared execution**，进一步降低 HBM 带宽需求。
- 在更多模型（如 Llama 系列）和 sparsification 方法上验证泛化能力。

---

> **总结**：  
> QUILT 重新思考了稀疏注意力的执行范式，提出了一种 **以 workload 结构为中心** 的 group-based execution 框架。通过 **SCSD、cascaded sharing、tile-aware clipping** 等技术，实现了高达 **55.1% kernel 加速** 和 **36.8% TTFT 降低**，同时保持精度几乎不变。该工作揭示了：**未来的高效 attention kernel 不仅要关注单 query 效率，更要挖掘跨 query 的协同潜力**。

</details>

---

### 2. [Zepp: Accelerating Distributed MoE Serving under Relaxed Balance Constraints](https://arxiv.org/abs/2610.11158)

**Authors**: Chang Chen, Andrew Yang, Tiancheng Chen, Jiangfei Duan, Xinwei Qiang, Zhongkai Yu, Xiang Fang, Yufei Ding  
**Category**: cs.DC  
**Published**: 2026-10-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.11158v1  

#### Abstract
As Mixture-of-Experts (MoE) models continue to scale, serving them increasingly relies on expert parallelism (EP) across a growing number of devices. Yet skewed expert workloads create imbalance across computation, communication, and memory, making load balancing a central optimization objective in ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Zepp: Accelerating Distributed MoE Serving under Relaxed Balance Constraints

---

## 1. 论文的主要贡献和创新点

### 解决的问题
随着 **Mixture-of-Experts (MoE)** 模型规模不断扩大，分布式推理依赖于跨多设备的 **Expert Parallelism (EP)**。然而，动态路由导致专家负载高度不均衡，引发以下问题：
- **计算不平衡**：部分 GPU 负载过重，其他空闲。
- **通信瓶颈**：all-to-all token dispatch 引发大量跨节点通信（inter-node communication），受限于低带宽、高延迟的 NIC。
- **平衡代价高昂**：传统方法通过复制专家、重映射（remapping）等手段追求“平衡”，但这些操作本身会引入额外的通信开销（如权重迁移），反而可能成为新的瓶颈。

### 提出的新方法与新思路
**Zepp** 提出了一种全新的设计哲学：  
> **将“平衡”从优化目标转变为物理资源约束（GPU 和 NIC），直接优化端到端的关键路径——瓶颈级的跨节点通信。**

其核心思想是 **“避免 → 重塑 → 隐藏”** 跨节点通信：

#### 创新点 1：**Relaxed Balance Constraints**
- 不再追求完美的负载均衡，而是允许在一定松弛范围内（relaxed constraints）进行优化。
- 将 GPU 计算负载和 NIC 流量作为硬性资源约束处理，目标是最小化跨节点通信总量。

#### 创新点 2：**三层协同优化架构**
1. **Expert Placement + Locality-Preferred Routing（避免通信）**
   - 基于历史请求频率进行专家复制与放置（placement），优先提升节点级覆盖（node-level coverage）。
   - 在运行时采用“本地优先”的路由策略，在满足负载约束的前提下，尽可能将 token 发送到本地副本，减少远程请求。

2. **Split & Merge Primitives（重塑通信）**
   - **Split**：将 all-to-all 拆分为独立可调度的数据流，并利用 NVLink 在节点内重新分配流量，缓解 NIC 瓶颈。
   - **Merge**：
     - **Dispatch 阶段**：合并同一 token 对多个同节点专家的请求，只传输一次 token，再在节点内转发。
     - **Combine 阶段**：对来自同一 token 的多个专家输出先做局部聚合（pre-reduce），仅回传部分和，大幅减少返回通信量。

3. **Three-Stream Scheduling + Intra-node Expert Swaps（隐藏通信）**
   - 设计细粒度调度器，使通信与计算重叠：
     - Dispatch 时：通信驱动计算（communication drives computation）。
     - Combine 时：计算驱动通信（computation drives communication）。
   - 动态负载变化时，执行 **intra-node expert swap**（节点内交换专家位置），避免全局重映射带来的跨节点权重传输。

---

### 相比现有方法的优势
| 维度 | 传统方法 | Zepp |
|------|--------|------|
| 平衡理念 | 追求各维度平衡为首要目标 | 将平衡视为约束，以最小化瓶颈通信为核心目标 |
| 通信优化 | 通常只拆分（split）或填充（padding） | 同时支持 split 和 merge，灵活重构通信图 |
| 动态适应 | 全局 remapping 开销大 | 节点内 swap + 权重移动与计算/通信重叠 |
| 效率 | 可能因过度平衡牺牲效率 | 更高效地利用资源，避免“为了平衡而失衡” |

---

## 2. 核心实验方法和设置

### 数据集
使用来自先前研究 [42] 的公开 MoE 层专家激活轨迹数据集：
- **Kimi-K2-Thinking (K2)**：384 个 routed experts，每 token 激活 8 个专家。
- **Qwen3-235B-A22B (Qwen)**：128 个 routed experts，top-k=8。
- 输入 trace 来源于 **LiveCodeBench** 和 **MMLU** 等真实场景任务，反映实际的专家激活偏斜（skew）模式。

### 实验设置
- **硬件平台**：
  - A100 集群：每个节点 4×40GB A100 GPU，NVLink 互联（100 GB/s），每 GPU 绑定一个 HPE Slingshot 11 NIC（25 GB/s inter-node bandwidth）。
  - H100 集群（用于 weak scaling）：每节点 4×96GB H100，第四代 NVLink（150 GB/s）。
- **拓扑结构**：测试 4–32 节点（16–128 GPU）配置。
- **模型参数**：BF16 精度，固定 top-k=8。

### 评估指标
- **MoE Layer Latency**：单层 MoE 执行的最大 per-GPU 耗时（包含元数据交换、路由规划、通信与计算）。
- **End-to-End Performance**：
  - Prefill 阶段：Tokens Per Second (TPS)，越高越好。
  - Decode 阶段：Time Per Output Token (TPOT)，越低越好。
- **Speedup**：相对于最强 baseline 的加速比。

### 基线方法对比
共比较 **7 种 state-of-the-art MoE serving 系统**：
- **A2AV+GEMM**：基础 all-to-all + GEMM 实现。
- **FAST+GEMM**：优化 all-to-all 调度。
- **COMET**：细粒度 computation-communication overlap。
- **EPLB**：基于负载的专家复制与放置。
- **EPIC**：通信感知的专家放置与流水线执行。
- **MoonEP**：动态冗余专家预取。
- **COMET+EPLB**：结合 EPLB 放置与 COMET 重叠机制（最强 baseline 之一）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### ✅ MoE 层加速效果（Fig. 9）
- **最高加速比达 6.68×**（vs A2AV+GEMM）。
- **几何平均加速比 1.86×**（vs 最快 baseline，如 COMET+EPLB）。
- 在 16 节点、小 batch（1MB/token/GPU）下仍取得显著优势（~5.77×），说明对轻负载也有效。

#### ✅ 弱扩展性表现（Weak Scaling, Fig. 10）
- 随着集群规模扩大（2→32 nodes），Zepp 性能优势持续增强：
  - A100 上，1MB workload 加速比从 ~1.5× 提升至 **9.49×**。
  - H100 上更达 **15.10×**。
- 表明 Zepp 的优化在大规模系统中更具价值。

#### ✅ 端到端性能提升（Fig. 11）
集成到 **SGLang** 框架后：
- **Prefill 吞吐提升 1.32× ~ 1.92×**（随输入长度增加而增大）。
- **Decode 延迟降低 1.04× ~ 1.57×**。
- 显示 MoE 层优化在完整推理流程中依然有效。

### 消融实验结果（Ablation Study, Fig. 12）
在不同负载分布下的组件贡献分析：
- **Overlap + Placement/Routing** 是主要性能来源（+25% vs COMET）。
- **Overlapped Expert Swap** 在需求突变时进一步带来 **+3%~4%** 的延迟下降。
- 证明所有模块协同作用，尤其在非平稳负载下 swap 机制至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **Balance is not free**：追求单一维度的“完美平衡”往往引入其他维度的开销，甚至恶化整体性能。
2. **Inter-node communication is the true bottleneck**：即使计算已平衡，残余的跨节点通信仍主导延迟。
3. **Merge matters in MoE**：不同于 Dense 模型只需 split，MoE 中的冗余 token 传输和细粒度消息使得 **merge 成为必要优化**。
4. **Local swap > Global remapping**：intra-node expert swap 可有效应对动态负载，且成本远低于跨节点权重迁移。
5. **Co-design wins**：Zepp 通过 placement、routing、split/merge、scheduling 的联合设计，实现了系统级最优。

### 方法的局限性
- **依赖历史统计信息**：初始 placement 基于历史 trace，若 workload 分布剧烈漂移且无先验，初期性能可能受影响。
- **节点内带宽压力**：split 和 intra-node forwarding 会增加 NVLink 流量，极端情况下可能饱和。
- **实现复杂度较高**：需精细控制 CUDA stream、NVSHMEM 同步、动态调度逻辑。

### 未来工作方向
- **在线学习驱动的 adaptive placement**：结合实时反馈动态调整复制策略。
- **异构网络下的进一步优化**：支持更复杂的 NIC/GPU 拓扑（如非对称带宽）。
- **与 Attention-FFN disaggregation 结合**：探索更深层次的 MoE 架构解耦优化。
- **支持更多 MoE 变体**：如 hierarchical MoE、switching MoE 等。

---

> 🔗 **代码开源地址**：[github.com/andronius-yang/zepp](https://github.com/andronius-yang/zepp)

</details>

---

### 3. [PageWeaver: KV-Guided Query Unions for Sparse Attention](https://arxiv.org/abs/2610.11201)

**Authors**: Zhiyuan Li, Zihan Li, Zefang Yuan, Lei Wang, Hao Wang  
**Category**: cs.DC  
**Published**: 2026-10-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.11201v1  

#### Abstract
Dynamic sparse attention limits the KV pages selected by each query, but a small support does not necessarily yield efficient GPU work. Query unions share page loads and populate Tensor Core tiles; their cost depends on which queries are grouped together. We present PageWeaver, an execution design t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PageWeaver: KV-Guided Query Unions for Sparse Attention**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
在动态稀疏注意力（dynamic sparse attention）中，虽然每个查询（query）只选择有限的 KV 页面进行计算，从而减少交互数量，但这种不规则的页面访问模式会导致 GPU 执行效率低下。特别是：
- 不同的查询组合可能导致不同的 **Tensor Core tile 利用率** 和 **内存加载开销**；
- 现有方法如相邻查询分组（adjacent grouping）无法有效利用非局部（nonlocal）的 KV 页面重用；
- 页面导向的方法（如 MSA 的 KV-outer 设计）虽能提升重用，但引入了跨页部分输出的归约（reduction），增加了实现复杂性和额外开销。

因此，核心问题是：**如何在保持原始稀疏支持结构的前提下，优化查询执行分组以最大化硬件利用率并最小化总执行成本？**

---

### **提出了什么新方法或新思路**
作者提出 **PAGEWEAVER**，一种基于 KV 页面亲和性的在线查询重组执行框架，其核心思想包括：

1. **KV-Guided Query Regrouping（KV引导的查询重分组）**  
   - 利用“被共同选中的 KV 页面”作为查询之间的亲和性度量（affinity），在固定窗口内贪婪地将具有高页面重叠的查询聚合成组。
   - 分组目标是使每组的 **联合页面集合（union of selected pages）尽可能小**，从而减少不必要的页面加载和 masked computation。

2. **ID-Aware Two-CTA Union8 Kernel（ID感知双CTA核函数）**  
   - 不对 Q 张量进行物理重排，而是仅传递重排序后的 query ID 列表；
   - 内核根据这些 ID 动态读取原始位置的 Q、应用 membership mask 和 causal mask，并直接写回原输出地址；
   - 支持完整输出所有权（complete output ownership），避免跨页归约。

3. **保留语义不变性（Semantic Preservation）**  
   - 所有原始查询看到的支持集 $ A $ 完全一致，无概率质量共享或 softmax 状态泄漏；
   - 在精确算术下保证数学等价性。

4. **对比设计：Direct KV-Page Union Path**  
   - 提供一个替代路径，显式构建页面对并合并公共查询的部分输出，用于分析非局部重用的实际收益与代价。

---

### **相比现有方法的优势**
| 方面 | PAGEWEAVER | 其他方法 |
|------|-----------|--------|
| **执行效率** | 更小的 page union → 更少 HBM 访问 | 相邻分组可能混合不同 page 社区 |
| **实现复杂性** | 无需输出重排或跨页 reduction | KV-outer 需要 incidence inversion 和 final merge |
| **灵活性** | 在线分组适应每次调用的 Top-K 结构 | 固定分组无法捕捉动态亲和性 |
| **数值一致性** | 与原始 Union8 bitwise identical | 某些 fusion 路径可能引入舍入差异 |

---

## 2. **核心实验方法和设置**

### **使用了哪些数据集**
- 并未使用传统自然语言数据集，而是基于 **真实模型运行时捕获（captures）** 的稀疏注意力行为：
  - 主要来自 **MiniMax-M3 模型** 的 57 层稀疏注意力层；
  - 包括多个上下文长度（8K, 16K, 32K, 64K）的真实请求片段；
  - 特别构造了 **interleaved support 输入** 来验证非局部重用潜力。

> 注：所有实验基于实际推理过程中的 Top16 页面选择行为，输入为已选定的支持边（selected edges），不涉及重新设计稀疏策略。

---

### **实验设置和评估指标**

#### **平台配置**
- **主平台**：NVIDIA H200，FP8 E4M3 KV，head dim=128，page size=128，Top16 selection；
- **辅助平台**：B300（Blackwell 架构）用于对比研究；
- 使用 CUDA Graph 重放测量延迟，排除编译和分配时间。

#### **评估指标**
| 指标 | 描述 |
|------|------|
| `complete-call time` | 包含准备、量化、打包、attention、combine 的端到端耗时 |
| `geometric-mean speedup` | 多个 capture 上的速度提升几何平均 |
| `logical page visits` | 联合页面访问次数（逻辑计数） |
| `NRMS` | 归一化均方根误差，衡量输出与参考实现的偏差 |
| `prefill throughput` | 整体预填充阶段的 token/s 吞吐量 |

#### **对比基线**
| 基线 | 说明 |
|------|------|
| **FlashInfer** | 支持 FP8 KV 的高性能稀疏 attention 引擎，作为外部 baseline |
| **Native SM90** | SGLang 中集成的原生稀疏 kernel，作为内部 baseline |
| **Original Union8** | 使用原始顺序执行的 Union8 分组，无 regrouping |
| **KV-Page Pair Path** | 替代设计，显式利用页面间共享查询进行融合 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据与对比结果**

#### ✅ **基础性能增益（Base Execution Gain）**
- 在六个 captures 上，**Union8 相较 FlashInfer 实现 1.70× 几何平均加速**；
- 相较 native kernel 达到 **2.66× 加速**；
- Union8 vs Union4（相同算术条件下）也有 **1.075× 提升**，表明更大的分组尺寸更利于 Hopper 架构的 Tensor Core 利用。

#### ✅ **在线重分组带来的增量收益（Online Regrouping）**
- 在五个 64K 上下文 captures 上：
  - **降低 complete-call 延迟 3.26–7.66%**（相对 original Union8）；
  - 最佳案例中逻辑 page visits 减少 **12.7%**（从 102,429 → 89,426）；
  - W128 分组开销约为 **34.6–34.9 μs**，但净节省可达 **~58 μs**。

#### ✅ **整模型预填充吞吐提升**
- 相比测试的 native 路径，最终方案（online + Union8）在 32K/64K 下：
  - **整体 prefill throughput 提升 7.88–14.36%**；
  - 其中 regrouping 带来的**增量收益较小**：
    - 32K/64K：+0.47–0.73%
    - 8K：出现负增益（-0.56% ~ -1.31%），因分组开销超过收益

#### ❌ **B300 上优势消失**
- 在 B300 平台（CUDA 13, SM103, direct FP8 Q）上：
  - 尽管 W128 分组仍能小幅加速 Union8（1.068–1.156×）；
  - 但由于更强的 **native kernel 性能** 和较高的准备开销，**overall 仍落后于 native**；
  - 表明：**reuse 的价值取决于整个执行链的成本平衡**。

#### 🔍 **消融与诊断实验**
| 实验 | 发现 |
|------|------|
| **Random ID 排列** | 延迟从 1.37ms 升至 1.83ms，证明 locality 重要 |
| **Materialized Q reorder** | 比 ID-aware 访问慢 ~82μs，验证“不重排张量”的优势 |
| **Global hash + local grouping** | 1.298ms > ID-only affinity (1.234ms)，说明需兼顾相似性与局部性 |
| **W2048 离线搜索** | 执行时间低至 1.185ms，但 CPU 搜索耗时 **1.5秒**，不可部署 |

---

## 4. **关键结论和发现**

### **主要发现**
1. **执行分组的设计必须与硬件调度协同优化**  
   即使稀疏支持相同，不同的查询分组方式也会导致显著的性能差异。

2. **非局部重用存在但难以盈利**  
   - KV 页面间的查询重用确实存在（median reuse: 3.85–4.54）；
   - 但在实践中，**incidence inversion + partial reduction 的开销常常抵消了重用收益**；
   - 因此，PAGEWEAVER 选择在 query-oriented 框架内利用亲和性，而非转向 page-oriented 执行。

3. **在线分组可以带来正向净收益**  
   - 在 H200 上，**bounded greedy regrouping（W=128）能在合理开销内显著缩小 page union**；
   - ID-aware kernel 成功将分组逻辑与数据移动解耦，实现了高效执行。

4. **收益高度依赖于系统级成本结构**  
   - 在 B300 上，由于 native kernel 更强、descriptor copy 开销更高，regrouping 无法胜出；
   - 说明：**不能孤立看待“重用”，而应评估“exploiting reuse”的总成本**。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **受限于 512-page bitmap 容量** | 当 context > 65,536 tokens（H200 page size=128）时自动退化为 identity permutation |
| **贪心目标未考虑 subgroup compute proxy** | 当前最小化的是整体 union，而非 Equation (4) 中更贴近实际计算的 subgroup 成本 |
| **缺乏 per-layer 自适应策略** | 所有层统一使用 W128，未根据各层重用特征动态调整 |
| **未验证长文本生成质量影响** | 实验聚焦推理延迟和吞吐，未评估 regrouping 对生成质量的影响 |

---

### **未来工作方向**
1. **开发 cost-aware grouping selector**  
   综合考虑 union size、subgroup balance、address indirection 等因素，学习最优分组策略。

2. **扩展更大上下文的支持能力**  
   设计分层或采样式的亲和性估计机制，突破 bitmap 容量限制。

3. **探索编译器集成与静态调度**  
   在模型编译期预测常见 pattern 并缓存分组决策，降低在线开销。

4. **建立完整的服务质量评估体系**  
   将延迟、吞吐、能耗、生成质量纳入统一评估框架，指导调度策略选择。

5. **跨平台可移植性研究**  
   构建通用的 execution-cost modeling framework，指导在不同 GPU 架构上的参数调优。

---

> 📌 **总结一句话**：  
> **PAGEWEAVER 通过 KV 页面亲和性指导查询分组，在不改变稀疏支持的前提下提升了执行效率；其实验揭示了一个深刻洞见——重用本身不是胜利，只有当“获取重用的代价 < 节省的执行成本”时，它才真正有价值。**

</details>

---

### 4. [Bilevel optimization for data-driven learning of Koopman embeddings using kernel-based autoencoders](https://arxiv.org/abs/2610.12370)

**Authors**: Joel-Pascal Ntwali N'konzi, Feliks N\"{u}ske, Stefan Klus  
**Category**: cs.LG  
**Published**: 2026-10-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.12370v1  

#### Abstract
Koopman operator theory provides a linear framework for analyzing nonlinear dynamical systems and has become a major tool for data-driven modeling. A central challenge, however, is that finite-dimensional approximations computed by methods such as extended dynamic mode decomposition (EDMD) require t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Bilevel optimization for data-driven learning of Koopman embeddings using kernel-based autoencoders*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Extended Dynamic Mode Decomposition (EDMD)** 在学习非线性动力系统的 **Koopman operator** 时，依赖于预先设定的函数字典（dictionary），这在实践中难以设计且对高维系统不友好。虽然基于 **Artificial Neural Network (ANN)** 的自编码器（如 KAE、cKAE）已被用于从数据中自动学习 Koopman embeddings，但这些方法缺乏可解释性，理论分析困难。

此外，标准的 **kernel EDMD** 虽然避免了显式构造字典，但其嵌入维度受限于训练数据量，导致无法扩展到大规模数据集。

### ✅ 提出的新方法：EDMD-kDL
本文提出了一种名为 **Extended Dynamic Mode Decomposition with Kernel-based Dictionary Learning (EDMD-kDL)** 的新方法，结合了核方法与双层优化（bilevel optimization）的思想，实现从数据中直接学习有限维的 Koopman embeddings。

#### 核心创新点：
- **引入 kernel-based autoencoders**：首次将核方法应用于 Koopman 字典学习任务，利用 **Reproducing Kernel Hilbert Space (RKHS)** 表达能力强大的特性来参数化可学习的 embedding 映射。
- **采用双层优化框架**：
  - **内层优化**：固定伪值（pseudo-values）在一组 **collocation points** 上，求解 RKHS 中最优的 embedding 函数（通过 representer theorem 可解析求解）。
  - **外层优化**：联合优化伪值矩阵 $ U $ 和 Koopman 矩阵 $ K $，形成一个有限维非线性优化问题。
- **突破维度限制**：所学字典的维度独立于训练样本数量，仅取决于 collocation points 数量，因此天然支持大规模数据。
- **支持多步预测目标（multi-step prediction loss）**：这是标准 kernel EDMD 所不具备的能力，增强了模型长期预测稳定性。

### ✅ 相比现有方法的优势
| 特性 | ANN-based 方法 (KAE/cKAE) | 标准 kernel EDMD | **EDMD-kDL (本文)** |
|------|----------------------------|------------------|--------------------|
| 可解释性 | 差（黑箱网络） | 较好（核函数明确） | ✅ 更好（函数空间结构清晰） |
| 理论可分析性 | 有限 | 高 | ✅ 高（基于 RKHS 理论） |
| 可扩展性 | 一般（需大量参数） | 差（Gram 矩阵大小∝训练数据） | ✅ 强（Gram 矩阵大小∝collocation points） |
| 支持 multi-step loss | ✅ 是 | ❌ 否 | ✅ 是 |
| 性能表现 | 中等至良好 | 数据相关 | ✅ 更优或相当 |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集
论文在四类不同场景下进行了广泛实验：

1. **Undamped Nonlinear Pendulum**  
   - 二维哈密顿系统，模拟无阻尼摆运动。
   - 分为“近线性”（$x_1(0)=0.8$）和“强非线性”（$x_1(0)=2.4$）两种情况。
   - 包含干净数据与添加高斯噪声（NR=1%, 5%, 10%）的鲁棒性测试。

2. **Karman Vortex Shedding**  
   - 流体绕圆柱流动，雷诺数 Re=100。
   - 原始快照维度为 93600，经 SVD 降维至前 19 个主成分（保留 95% 能量）。

3. **Pendulum Video Prediction**  
   - 来自真实视频（720×576×523），预测单摆下一帧像素变化。
   - 经灰度化、背景去除后使用 SVD 降至 14 维。

4. **Global Sea-Surface Temperature (SST) Forecasting**  
   - NOAA 提供的全球海表温度再分析数据（180×360 网格，每周一次，共 1400 时间步）。
   - 有效点 44219，经 min-max 归一化和 SVD 降至 19 维。

---

### ✅ 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| **Embedding Dimension** | 多数设为 $N=6$ 或 $N=3r$（$r$: SVD 主成分数） |
| **Collocation Points** | 使用 Sobol 序列生成，数量远小于训练样本 |
| **Kernel Function** | 高斯 RBF 核 $k(x,x')=\exp(-\|x-x'\|^2/(2\sigma^2))$，带宽 $\sigma$ 通过调参确定 |
| **Training Objective** | 支持 single-step 和 multi-step prediction loss（如 $T=8,12$ 步） |
| **Optimizer** | L-BFGS + 自动微分（PyTorch） |
| **评估指标** | - 相对预测误差（Relative Prediction Error）<br>- MAE（Mean Absolute Error）<br>- 最大相对误差<br>- 可视化相图、时空场重建效果 |

---

### ✅ 基线方法对比
- **EDMD-RFF**: 使用 Random Fourier Features 近似的 kernel EDMD，作为核方法代表。
- **KAE (Koopman Autoencoder)**: 基于 ANN 的标准自编码器架构。
- **cKAE (Consistent Koopman Autoencoder)**: 加入前后向一致性约束的改进版 KAE。
- **SINDy-SHRED**: 结合稀疏识别与浅层循环解码器的最新方法（用于 SST 对比）。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据汇总

| 数据集 | 方法 | 最大相对误差（预测期） | MAE |
|--------|------|--------------------------|-----|
| **Nonlinear Pendulum (clean)** | EDMD-RFF | 最低（近线性） | — |
| | **EDMD-kDL** | **优于 KAE/cKAE** | — |
| | KAE / cKAE | 明显偏移，尤其在强非线性下 | — |
| **Noisy Pendulum (NR=10%)** | EDMD-RFF | 相位漂移严重 → 误差上升快 | — |
| | **EDMD-kDL** | ✅ 幅值与相位平衡，**误差最低** | — |
| | KAE / cKAE | 输出不光滑，预测不稳定 | — |
| **Vortex Shedding** | EDMD-RFF | 错误极高（未显示） | — |
| | **EDMD-kDL** | ✅ **长期预测误差最小** | — |
| | KAE / cKAE | KAE 表现最差 | — |
| **Video Prediction** | KAE / cKAE | 完全失败，无法捕捉动态 | — |
| | **EDMD-kDL** | ✅ 成功预测轨迹，物体位置略有模糊但仍可辨识 | — |
| **SST Forecasting (318周)** | EDMD-kDL | **6.7%** | **0.51°C** |
| | SINDy-SHRED | 11.8% | 0.67°C |
| | cKAE | 17.5% | 0.88°C |
| | KAE | 20.9% | 0.83°C |

> 📌 注：所有实验中，**EDMD-kDL 至少达到甚至超越 ANN-based 方法的表现**，尤其在噪声环境、视频建模和长期预测方面优势显著。

---

### ✅ 消融实验与关键观察（隐含）
尽管未单独列出消融实验章节，但从以下对比可推断关键设计的有效性：

- **是否使用 collocation points**：允许脱离训练数据规模限制，是可扩展性的基础。
- **是否支持 multi-step loss**：作者指出 multi-step 训练提升了鲁棒性，而这是标准 kernel EDMD 不支持的。
- **是否引入 decoder**：在 autoencoder 架构中加入线性 decoder，确保 full-state observable 可被重构，提升物理一致性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **核方法完全可以胜任 Koopman 字典学习任务**：  
   尽管近年来 ANN 占据主导，本文证明 **kernel-based 方法在表达能力和性能上并不逊色**，甚至更具优势。

2. **EDMD-kDL 在多种任务中表现优异**：  
   在合成系统、流体力学、视频预测和气候预报中，**EDMD-kDL 均达到 state-of-the-art 或更优水平**，尤其是在噪声环境下表现出更强鲁棒性。

3. **成功处理高维原始观测（如视频）**：  
   在 pendulum video 实验中，**KAE 和 cKAE 完全失效**，而 EDMD-kDL 能够从中学习出有效的低维线性动力学表示。

4. **兼具可扩展性与理论严谨性**：  
   方法既不像标准 kernel EDMD 受限于数据量，也不像 ANN 缺乏解释性，实现了 **scalability、performance 与 interpretability 的统一**。

---

### ⚠️ 方法的局限性
- 当前仍需先进行 **SVD 降维**，尚未直接处理原始高维输入（如图像网格）。
- **计算成本较高**：依赖 L-BFGS 等二阶优化器，训练时间较长；虽可通过 warm-start 策略缓解，但仍不如 mini-batch SGD 快速。
- **collocation points 的选择影响性能**：目前使用 Sobol 序列，但最优采样策略尚待研究。

---

### 🔮 未来工作方向（作者建议）
1. **理论分析**：建立 EDMD-kDL 的收敛性证明与有限样本误差界，结合 kernel EDMD 与 kernel collocation 方法的理论成果。
2. **端到端高维处理**：扩展方法以联合学习非线性投影映射（类似 kernel PCA），避免预降维。
3. **高效训练策略**：探索更高效的优化算法（如随机优化、分布式计算），降低训练开销。
4. **推广至随机系统**：将框架应用于随机微分方程（SDE）或马尔可夫过程，结合 VAMP 或 time-lagged autoencoder 范式。
5. **与其他 kernel 架构融合**：例如结合 GP 或 operator learning 中的 kernel design 思路进一步提升性能。

---

## ✅ 总结
本文提出的 **EDMD-kDL** 是一种新颖且强大的数据驱动 Koopman 学习框架，它通过 **kernel-based autoencoders + bilevel optimization** 的组合，在保持核方法可解释性和理论优势的同时，克服了传统 kernel EDMD 的可扩展性瓶颈，并在多个复杂系统上展现出优于主流 ANN 方法的预测性能。该工作为 **Koopman 理论、核方法与深度学习的交叉融合** 提供了一个重要范例。

</details>

---

### 5. [Fed-GRPO: Reward-Signal-Driven Federated Group Relative Policy Optimization](https://arxiv.org/abs/2610.11502)

**Authors**: Pengxin Guo, Shuang Zeng, Zonggen Li, Weiying Zheng, Mengting Liu, Liangqiong Qu  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.11502v1  

#### Abstract
Large Language Models (LLMs) have shown strong reasoning capabilities when fine-tuned with reinforcement learning (RL), particularly through Group Relative Policy Optimization (GRPO). However, existing GRPO methods assume centralized access to training data, which may not hold in practice due to pri...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Fed-GRPO: Reward-Signal-Driven Federated Group Relative Policy Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- 当前基于 **Group Relative Policy Optimization (GRPO)** 的大语言模型（LLM）推理能力训练依赖于集中式数据访问，这在医疗、金融、教育等高价值领域不可行，因为数据受隐私、法规或所有权限制无法共享。
- 标准 **Federated Learning (FL)** 方法（如 FedAvg）直接应用于 GRPO 效果不佳，存在三大根本性问题：
  1. **Reward-misaligned aggregation**：按数据量加权聚合忽略客户端更新的质量差异；
  2. **Local-global objective miscalibration**：本地奖励统计缺乏全局视角，导致训练资源错配；
  3. **Reward-agnostic communication**：通信带宽分配未考虑更新的信息量，造成资源浪费。

### **提出的新方法与新思路**
作者提出 **Fed-GRPO**，一个全新的联邦 GRPO 框架，其核心思想是：  
> **GRPO 训练中自然产生的奖励统计（reward statistics）——即每轮的奖励均值 $ \mu_k $ 和标准差 $ \sigma_k $ —— 是零成本、高质量的信号，可用于指导联邦训练中的聚合、优化和通信。**

基于此，Fed-GRPO 引入三个耦合机制：

| 机制 | 功能 | 创新点 |
|------|------|--------|
| **Signal-Weighted Aggregation** | 聚合时以 $ \sigma_k $（奖励标准差）为权重，而非数据量 | $ \sigma_k $ 直接反映学习信号强度，优先采纳高质量更新 |
| **Global Reward Calibration** | 服务器广播全局奖励均值 $ \mu_{\text{global}} $，客户端据此调整各 prompt 的训练权重 | 鼓励客户端聚焦“相对薄弱”的任务，避免过拟合已掌握的内容 |
| **Adaptive Sparse Communication** | 根据 $ \sigma_k $ 动态分配通信带宽，高信号客户端保留更多参数更新 | 利用 GRPO 更新天然稀疏性（>96.2% 参数不变），实现高效压缩 |

### **相比现有方法的优势**
- **更优性能**：显著优于所有联邦基线，接近集中式训练上限。
- **更高效率**：通信开销可无损压缩 **32×**，极端条件下支持高达 **621×** 压缩。
- **零额外计算成本**：所用信号 $ (\mu_k, \sigma_k) $ 在 GRPO 中本就需计算，无需额外开销。
- **动态适应性**：权重和带宽随训练进程自适应变化，响应实时学习状态。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **数学推理任务**：
  - **MATH**：约 7.5K 竞赛级数学题，按难度分为 5 级，用于构建非独立同分布（non-IID）联邦设定（每个客户端负责一种难度）。
  - **测试基准**：GSM8K、MATH-500、AIME 2024、AIME 2025、AMC 2023。
- **代码生成任务**：
  - **MBPP**：用于训练，按参考解长度划分非 IID 数据。
  - **测试集**：MBPP test、HumanEval。

### **实验设置**
- **模型**：
  - 主要使用 **Qwen2.5-3B-Instruct** 和 **Qwen3-4B-Instruct**。
- **训练配置**：
  - 本地训练步数：S = 5
  - 通信轮次：T = 12
  - 优化器：AdamW（lr = 3e-6）
  - Rollout 设置：batch size 64，每 prompt 采样 8 条路径
  - 奖励函数：二值正确性奖励（binary correctness reward）
- **评估指标**：
  - **avg@32**：每个 prompt 生成 32 个样本，取平均准确率。
  - 报告各 benchmark 的准确率及平均分（Avg.）。
  - 通信开销：上行负载（Comm. in MB）、压缩比（Comp. ×）。

### **基线方法对比**
| 基线 | 类型 | 说明 |
|------|------|------|
| **Base Model** | 无训练 | 原始模型性能 |
| **Centralized GRPO** | 上限 | 所有数据集中训练，理想性能上限 |
| **Local-Only GRPO** | 下限 | 各客户端独立训练，无聚合 |
| **FedAvg-SW/UW** | 经典 FL | 按样本数或均匀加权聚合 |
| **SparsyFed / SparseLoCo** | 稀疏通信 FL | 固定比例 top-k 剪枝 |
| **FGRPO** | 自适应聚合 | 基于历史性能增益动态加权 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Qwen2.5-3B-Instruct，数学任务）**

| Method | Avg. Score | Comm. (MB) | Comp. (×) |
|--------|------------|-------------|-----------|
| Centralized | **40.21** | – | – |
| FedAvg-SW | 37.10 | 6,794.2 | 1× |
| FGRPO | 38.49 | 6,794.2 | 1× |
| **Fed-GRPO (B=100%)** | **39.06** | **645.5** | **32×** |
| **Fed-GRPO (B=50%)** | **39.02** | **321.4** | **63×** |
| **Fed-GRPO (B=5%)** | **35.08** | **32.8** | **621×** |

> ✅ **无损压缩**：B=100% 时通信减少 **32×**（从 6.8GB → 645MB），精度几乎无损（39.06 vs. 39.07）。  
> 🚀 **极致压缩**：B=5% 时达 **621×** 压缩（仅 32.8MB/轮），精度下降 ~4 点，仍优于多数基线。

### **与基线方法的对比结果**
- **性能全面领先**：
  - Fed-GRPO 在所有数学和编码任务上均优于所有联邦基线。
  - 在 Qwen3-4B 上，平均得分 **66.35**，逼近集中式训练（66.36）。
- **超越 FedAvg 和 FGRPO**：
  - FedAvg-SW 性能甚至低于 Local-Only，表明简单加权聚合有害。
  - Fed-GRPO 显著优于 FGRPO，证明奖励统计信号比历史增益更有效。

### **消融实验结果（Ablation Study）**

| 配置 | Avg. Score | Comm. (MB) |
|------|-----------|-------------|
| FedAvg-SW（无机制） | 37.10 | 6,794.2 |
| + Signal-Weighted Aggregation (SWA) | 38.01 (+0.91) | 6,794.2 |
| + Global Reward Calibration (GRC) | 39.07 (+1.06) | 6,794.2 |
| + Adaptive Sparse Communication (ASC) | **39.06** | **645.5** |

- **SWA 和 GRC 是精度提升主因**，分别带来 +0.91 和 +1.06 平均增益。
- **ASC 实现正交通信节省**：在不损失精度的前提下将通信降低 11×。

---

## **4. 关键结论和发现**

### **主要发现**
1. **奖励统计是联邦 GRPO 的理想协调信号**：
   - $ \sigma_k $ 和 $ \mu_k $ 天然携带关于客户端学习信号强度、训练偏差和更新密度的信息。
   - 这些信号“免费可用”，无需额外计算即可驱动整个联邦流程。

2. **传统 FedAvg 不适用于 GRPO**：
   - 按数据量加权会误导聚合方向，尤其当某些客户端已饱和而另一些仍有强学习信号时。

3. **通信可以高度稀疏且自适应**：
   - GRPO 更新本身极度稀疏（>96.2% 参数不变）。
   - 更新密度与 $ \sigma_k $ 正相关，因此可根据信号强度智能分配带宽。

4. **Fed-GRPO 接近集中式性能**：
   - 在多种模型和任务上，性能逼近 Centralized GRPO，验证了其有效性。

### **方法的局限性**
- **依赖 GRPO 特性**：目前仅适用于 GRPO 或类似优势估计方式，难以直接迁移到 PPO 等需 Critic 的 RL 方法。
- **非 IID 场景假设较强**：实验主要基于难度或主题划分的 non-IID，现实场景可能更复杂。
- **理论收敛界含漂移项**：由于目标函数随时间变化（calibration），最终收敛到邻域而非精确 stationary point。

### **未来工作方向**
- 将奖励信号驱动的思想扩展到其他联邦 RL 范式（如多奖励、多任务）。
- 探索更复杂的模块选择策略（beyond efficiency-based）。
- 在真实跨机构场景（如医院间协作）中部署验证。
- 结合差分隐私（DP）或安全聚合（Secure Aggregation）进一步增强隐私保护。

---

> 🔗 **代码开源**：https://github.com/HKU-HealthAI/Fed-GRPO  
> 📄 **原文链接**：https://arxiv.org/abs/2610.11502

</details>

---

### 6. [Deflating the Hessian: Rank-4 W4A4 Quantization for Multimodal Diffusion Transformers](https://arxiv.org/abs/2610.11315)

**Authors**: Shiwen Wang, Pengxiang Zhao, Xiaoming Yuan  
**Category**: cs.LG  
**Published**: 2026-10-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.11315v1  

#### Abstract
In diffusion transformers, low-rank branches can mitigate 4-bit weight--activation (W4A4) post-training quantization (PTQ) loss by decomposing each weight into a low-bit residual and a high-precision low-rank component. Existing low-rank PTQ approaches, however, either optimize low-rank compensation...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Deflating the Hessian: Rank-4 W4A4 Quantization for Multimodal Diffusion Transformers**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Diffusion Transformers (DiTs)** 中进行 **Weight-Activation 4-bit Post-Training Quantization (W4A4 PTQ)** 时，低精度量化会引入显著的重建误差，尤其是在激活值（activations）也参与4位量化的场景下。现有低秩辅助量化方法存在以下不足：
- 低秩补偿与残差量化**分开优化**，导致需要更高的秩（rank）才能达到良好性能；
- 使用二阶梯度更新但**未显式建模激活量化误差**，在 W4A4 下误差累积严重。

因此，如何在**小秩预算**（如 rank=4）下联合优化权重与激活侧的量化误差，成为提升 W4A4 性能的关键挑战。

---

### 🚀 提出的新方法：**H-SVDQuant**

提出了一种统一的低秩辅助 W4A4 PTQ 框架 —— **H-SVDQuant**，其核心思想是将低秩补偿与残差量化建模为一个**耦合的校准问题**，并从联合目标中推导出优化求解器。

#### 主要创新点：
1. **Hessian Deflation（海森矩阵去膨胀）**
   - 通过解析地消除输出侧低秩因子 $L_2$，得到一个“被压缩”的 Hessian 矩阵 $H_\perp$，满足 $H_\perp L_1 = 0$。
   - 这意味着：**已被低秩分支捕获的输入子空间中的误差不再需要由残差量化来承担**，从而让残差专注于难以表示的“剩余误差”。

2. **Activation-Noise Surrogate（激活噪声代理）**
   - 引入对角形式的激活噪声协方差模型 $\Sigma_A$，用于抑制激活量化误差在残差优化过程中的放大效应。
   - 在目标函数中加入加权项 $\lambda_A \cdot \text{tr}(R^T \Sigma_A R)$，实现对激活误差传播的显式控制。

3. **端到端联合优化框架**
   - 统一建模低秩补偿、残差量化与激活误差，形成可优化的目标函数。
   - 推导出高效的闭式解（closed-form solvers），包括：
     - 校准加权的低秩初始化（calibration-weighted initialization）
     - 对角平滑（diagonal smoothing）的解析解
     - GPTQ 风格的残差量化
     - 闭式低秩 refitting

---

### 🔍 相比现有方法的优势
| 特性 | H-SVDQuant | SVDQuant / LRC / LoRaQ |
|------|-----------|------------------------|
| 是否联合优化 | ✅ 是 | ❌ 否（分离优化） |
| 是否建模激活误差 | ✅ 显式建模 | ❌ 忽略或隐含处理 |
| 所需秩大小 | ⭐ **仅需 rank=4** | 通常需 rank ≥ 32 |
| 量化速度 | ⬇️ 最高 **6.25× 加速** | 较慢 |
| 性能表现 | ✅ 超越 rank-32 基线 | 在 rank=4 下性能有限 |

> 💡 **核心优势**：以 **8倍更小的秩（rank-4 vs rank-32）** 实现同等甚至更好的 W4A4 重建质量，并大幅降低量化成本。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **图像生成任务**：
  - **MJHQ-30K**：类别平衡的高质量图像数据集，用于训练/校准。
  - 测试子集：**MJHQ-5K**, **sDCI-1K**（diverse captioned image benchmark）
- **语言模型任务**（Qwen3-8B）：
  - **WikiText-2 training split**：采样 128 条序列（每条 512 tokens）用于校准。

所有实验均使用与教师模型相同的随机种子、初始 latent 和调度步数进行推理比较。

---

### ⚙️ 实验设置
| 设置项 | 描述 |
|-------|------|
| **量化模式** | W4A4（权重和激活均为 4-bit） |
| **分组大小** | Group size = 64（DiTs），Group size = 128（LLM） |
| **保护秩** | $r = 4$（主实验），部分对比 $r=32$ |
| **低秩结构** | $W = L_1 L_2 + R$，其中 $L_1, L_2$ 为 FP16/BF16，$R$ 为 INT4 |
| **激活量化** | Per-token, per-group 4-bit 量化 |
| **校准方式** | Layer-wise replay，逐模块替换并评估输出 MSE |

---

### 📊 评估指标
#### 图像生成任务（Diffusion Transformers）：
| 指标 | 含义 |
|------|------|
| **PSNR ↑** | 峰值信噪比，衡量像素级重建保真度（相对于 FP16 教师模型） |
| **LPIPS ↓** | 学习型感知相似性，反映视觉感知差异 |
| **CLIP Score ↑** | 文本-图像语义对齐程度 |
| **ImageReward (IR) ↑** | 人类偏好近似评分 |

#### 大语言模型任务（Qwen3-8B）：
| 指标 | 含义 |
|------|------|
| **WikiText-2 Perplexity (PPL) ↓** | 语言建模困惑度 |
| **Zero-shot Accuracy** | MMLU、ARC-Easy、ARC-Challenge、HellaSwag、PIQA 上的准确率 |

---

### 🆚 基线方法对比
| 方法 | 简介 |
|------|------|
| **SVDQuant (r=4 / r=32)** | 提取 SVD 分支后量化残差；r=32 作为高容量参考 |
| **DiRotQ** | 基于旋转的激活感知量化，匹配高精度比例 |
| **OrbitQuant** | 归一化旋转表示下的共享码本量化 |
| **ViDiT-Q** | DiT专用混合精度量化方案 |

> 所有方法均适配至 W4A4 + group-64 设置以确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & Table 3）

#### ✅ 在 Diffusion Transformers 上的表现（W4A4, r=4）：

| 模型 | 方法 | PSNR (↑) | LPIPS (↓) | 超越 r=4 SVDQuant | vs r=32 SVDQuant |
|------|------|---------|----------|------------------|------------------|
| **PixArt-2** | H-SVDQuant | **16.47** | **0.386** | +1.53 dB / -0.092 | 接近或更好 |
| **SANA-1.6B** | H-SVDQuant | **20.25** | **0.161** | +0.61 dB / -0.015 | ✅ 超越 |
| **FLUX.1-dev** | H-SVDQuant | **20.58** | **0.227** | +1.81 dB / -0.076 | ✅ 超越 |
| **Qwen-Image** | H-SVDQuant | 19.23 | 0.236 | +0.45 dB / -0.020 | 混合结果（PSNR略低，LPIPS更低） |

> 🔹 **视觉质量**：H-SVDQuant 更好保留纹理细节、物体边界和提示一致性（见 Figure 2）。

---

#### ✅ 在 Qwen3-8B 上的表现（W4A4, r=4）：

| 方法 | PPL ↓ | MMLU ↑ | 平均准确率 |
|------|------|--------|------------|
| FP16（全精度） | 10.48 | 72.95% | 72.95% |
| SVDQuant r=32 | 11.21 | 61.50% | 65.05% |
| **H-SVDQuant r=4** | **10.44** | **68.17%** | **68.04%** |

> ✅ **结论**：即使在大语言模型上，rank-4 H-SVDQuant 也能**超越 rank-32 SVDQuant**，且接近全精度模型！

---

### 🔬 消融实验结果（Ablation Studies）

#### （1）秩消融（Table 4a）
- 即使 **rank=0**（无低秩分支），H-SVDQuant 仍能达到 PPL=10.72，优于 rank-32 SVDQuant（11.21）。
- 引入 rank=4 后进一步降至 **10.44**，说明各组件协同增效。

#### （2）激活噪声权重 $\lambda_A$ 敏感性（Table 5）
- 最佳范围：$\lambda_A \in [0.05, 0.25]$
- 过大的 $\lambda_A=0.5$ 反而导致性能下降 → 表明需平衡权重与激活误差建模。

#### （3）Hessian Deflation 与 初始化消融（Table 6）
| 组件组合 | PPL |
|--------|-----|
| Plain SVD + Original H | 10.559 |
| Calibration-weighted + Original H | 10.492 |
| Plain SVD + Deflated H | 10.539 |
| **Calibration-weighted + Deflated H** | **10.486** ✅ |

> 表明两个机制互补：  
> - **加权初始化**选择重要方向  
> - **Hessian Deflation**防止重复优化已覆盖方向

#### （4）对角平滑（Diagonal Smoothing）效果（Table 7）
| D 构造方式 | PPL ↓ | 平均准确率 ↑ |
|----------|------|-------------|
| Grid-search D（SVDQuant） | 10.895 | 67.84% |
| **Closed-form D（本文）** | **10.469** | **68.44%** |

> ✅ 解析解不仅更快（免搜索），而且性能更强。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **H-SVDQuant 成功实现了高效的小秩 W4A4 量化**：
   - 仅用 **rank=4** 就能在多个 DiT 模型上**匹敌甚至超越 rank=32 SVDQuant** 的性能。
   - 在 **SANA-1.6B 和 FLUX 系列模型上实现 8× 更小秩 + 更高性能**。

2. **Hessian Deflation 是关键机制**：
   - 通过消除 $L_2$ 导出 $H_\perp$，使得残差量化只关注“不可吸收”的误差方向，极大提升了低秩分支的效率。

3. **显式建模激活误差至关重要**：
   - 引入 $\Sigma_A$ 和 $\lambda_A$ 可有效抑制激活量化误差的传播，在 W4A4 场景下尤为必要。

4. **整体流程高效实用**：
   - 支持闭式解、无需网格搜索，**离线校准时间最多减少 6.25×**（见 Table 2）。
   - 在 **Qwen3-8B 上验证了泛化能力**，表明该方法适用于 LLMs。

---

### ⚠️ 局限性
- 当前方法依赖于校准数据的统计特性（Gram matrix $H$），对分布偏移敏感。
- $\lambda_A$ 需要手动调节，尚未实现完全自适应。
- 目前假设激活误差为对角协方差，忽略了通道间相关性。

---

### 🔮 未来工作方向
1. **自动化 $\lambda_A$ 调参策略**，基于验证集反馈动态调整。
2. 扩展至 **W3A4 / W2A4** 极端量化场景。
3. 结合 **rotation 或 permutation** 技术进一步压缩残差。
4. 探索 **跨层共享低秩空间** 以进一步降低存储开销。
5. 应用于 **video diffusion models** 和 **multimodal agents**。

---

## ✅ 总结一句话
> **H-SVDQuant 通过 Hessian Deflation 与 Activation-Noise Surrogate 的联合建模，在 rank=4 的极小代价下实现了超越 rank-32 方法的 W4A4 量化性能，是当前最高效的 Diffusion Transformer 量化框架之一。**

</details>

---

### 7. [Looking Inside LLMs: Small-World Connectivity as a Signature of Reasoning Performance](https://arxiv.org/abs/2610.12304)

**Authors**: Zheng Huang, Sansheng Cao, Enpei Zhang, Weikang Qiu, Elynn Chen, Xiang Zhang, Yaoqing Yang, Rex Ying, Dawei Zhou, Yujun Yan  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.12304v1  

#### Abstract
Understanding large language model (LLM) reasoning requires looking beyond behavioral performance to examine how reasoning ability is reflected in internal organization. Inspired by neuroscience findings linking higher intelligence to stronger small-world organization in functional brain networks, w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Looking Inside LLMs: Small-World Connectivity as a Signature of Reasoning Performance*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文旨在**超越传统的行为层面评估**（如准确率、perplexity），从**内部结构组织**的角度理解大语言模型（LLM）的推理能力。具体而言，它试图回答以下核心问题：
- 是否存在一种可测量的**内部结构特征**，能够反映 LLM 的推理性能？
- 这种结构特征是否可以用于指导模型压缩（如剪枝），在减少参数的同时更好地保留推理能力？

### 提出了什么新方法或新思路
论文提出了三个层次的创新：

#### （1）**新观察（New Observation）**
首次发现并验证了：  
> **LLM 中的注意力头（attention heads）在激活模式上呈现出“小世界网络”（small-world connectivity）结构**，且这种结构的强度（以 Small-World Index, SWI 衡量）与模型在流体智力任务（fluid reasoning）上的表现呈强正相关。

这一发现将神经科学中关于人类大脑“小世界组织”与智力关系的研究迁移到了 LLM 领域。

#### （2）**新发现（Novel Findings）**
通过分析重要注意力头的连接模式，发现：
- 对模型性能更重要的 attention heads 倾向于具有：
  - **高 core score**：更多连接权重集中在自身社区内（强局部聚集）。
  - **低 bridge score**：连接权重更集中分布在少数几个社区（而非均匀跨社区连接）。
  
这表明这些**局部连接模式是推理能力的重要结构性指标**。

#### （3）**新方法（New Method）：Small-World Allocation (SWA)**
提出了一种基于上述结构洞见的**分层稀疏分配方法 SWA**：
- 利用 **core score 和 bridge score** 构建 head-level 的剪枝优先级。
- 在 layer-level 聚合 head-level 分数，形成层间稀疏度分配。
- 优先保留具有“高 core + 低 bridge”特性的注意力头。

### 相比现有方法的优势
- **理论驱动**：不同于黑箱式的剪枝策略，SWA 基于对功能网络结构的理解，具有更强的可解释性。
- **性能更优**：在相同稀疏度下，SWA 显著优于 Uniform、FARMS、ATP 等基线方法，在 WikiText 上最高降低 **20% 的 perplexity**。
- **结构保持更好**：SWA 更好地保留了原始模型的小世界组织（SWI 更高），说明其有效维持了支持推理的功能架构。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 用途 |
|--------|------|
| **DRE-Bench** | 主要用于构建功能图（functional graph）和评估流体推理性能（NLL/token）。使用 Level 1 & 2 问题作为校准输入。 |
| **WikiText** | 主要评估语言建模能力，使用 **Perplexity (PPL)** 作为核心指标。 |
| **GSM8K, ARC-C, MMLU** | 用于敏感性分析，测试不同任务构建的功能图对 SWA 效果的影响。 |
| **BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy/Challenge, OpenBookQA** | 用于 zero-shot 任务准确率评估。 |
| **ARAOC** | 用于评估抽象推理任务中的输出形状正确性（Mismatch Rate, Not M）。 |

### 实验设置和评估指标
- **模型范围**：在 **6 个主流开源 LLM** 上进行实验，涵盖 Qwen3 和 Llama3 系列（1B ~ 14B 参数）。
- **功能图构建流程**：
  1. 收集 attention heads 的 query 激活（基于 DRE-Bench 输入）。
  2. 使用 **SVCCA 变体**（结合 Fisher z-transform）计算 head 间的相似性。
  3. 构建加权相似性矩阵，并按固定边密度（δ=0.25）阈值化为无向图。
  4. 使用 Louvain 算法进行社区检测。
  5. 计算 **Small-World Index (SWI)**：`SWI = (C/C_rand) × (E/E_rand)`，其中 C 是聚类系数，E 是全局效率。
- **剪枝设置**：
  - 应用于 **SparseGPT** 和 **Wanda** 两种 one-shot 剪枝框架。
  - 剪枝比例：**50% 和 70%**。
  - 所有结果报告 **三次运行的均值 ± 标准差**。

### 基线方法对比
| 方法 | 类型 | 说明 |
|------|------|------|
| **Uniform** | 基线 | 各层/各 head 均匀分配稀疏度。 |
| **FARMS** | Layer-wise | 基于权重谱分析（eigenspectrum）的层间稀疏分配。 |
| **ATP** | Layer-wise | 基于误差传播假设，前层少剪、后层多剪的算术递增策略。 |
| **SWA (Head)** | Head-wise | 仅使用 head-level 的 core/bridge 分数分配稀疏度。 |
| **SWA (Layer)** | Layer-wise | 仅使用 layer-level 聚合分数分配稀疏度。 |
| **SWA** | Hierarchical | 完整版本，结合 layer 和 head 两级分配。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **SWI 与推理性能强相关**
- 图 2 和表 6 显示，在 **Pythia-6.9B 的训练过程中**，随着训练步数增加：
  - **SWI 持续上升**
  - **DRE-Bench 的 NLL/token 持续下降**
  → 表明小世界组织随学习过程增强，并与推理能力同步提升。

- 图 3 和表 7 显示，在 **不同架构和规模的 6 个 LLM 之间**：
  - 性能越好的模型（NLL 越低），其 **SWI 越高**。
  → 表明 SWI 是跨模型一致的推理能力结构签名。

#### ✅ **SWA 显著优于基线方法**
- **表 3：WikiText Perplexity (↓)**
  - 在所有模型和稀疏度下，**SWA 均取得最低的 PPL**。
  - 例如，在 Llama-3.2-1B @70% 稀疏度时，SWA 将 PPL 从 157.20 降至 **105.28**，相对改善超过 **33%**。
  - 平均来看，**SWA 最多可将 perplexity 降低 20%**。

- **表 8：Zero-shot Accuracy (↑)**
  - SWA 在 zero-shot 任务上也显著优于其他方法，表明其保留的是通用推理能力，而不仅是语言建模拟合度。

#### ✅ **SWA 更好地保留小世界结构**
- **表 2：剪枝后的 SWI (↑)**
  - 所有剪枝方法都会降低 SWI，但 **SWA 保留的 SWI 最高**。
  - 例如，Qwen3-14B 在 50% 剪枝后，SWA 的 SWI 达到 **1.1386**，远高于 SparseGPT 的 0.8000。
  → 证明 SWA 成功保护了支持推理的关键网络拓扑。

#### ✅ **消融实验结果（Ablation Study）**
- **表 4：Hierarchical 结构的有效性**
  - 移除任一级别（layer 或 head）都会导致性能下降。
  - **SWA(Head) 的退化大于 SWA(Layer)**，说明：
    - **Layer-level 分配是主信号**（决定哪一层多剪/少剪）。
    - **Head-level 分配是精细调节**（在同一层内决定保留哪些 head）。
  - 两者结合才能实现最优效果。

- **表 5：图构建数据集的敏感性**
  - 即使使用 GSM8K、ARC-C 或 MMLU 构建功能图，**SWA 依然能带来显著收益**。
  - 表明该方法对图构建任务具有一定鲁棒性。

---

## 4. 关键结论和发现

### 主要发现
1. **小世界指数（SWI）是 LLM 推理能力的可靠结构签名**：
   - SWI 与流体推理性能在**跨训练阶段**和**跨模型**两个维度上均呈强正相关。
   
2. **重要注意力头具有特定的局部连接模式**：
   - 高 **core score** 和低 **bridge score** 的 heads 更可能对推理至关重要。
   - 这些结构特征可作为剪枝时的优先保留依据。

3. **结构感知的剪枝更有效**：
   - 基于结构洞见设计的 **SWA 方法**，在保留模型性能方面显著优于现有方法。
   - 证明了“保持功能网络结构”是一种有效的模型压缩原则。

### 方法的局限性
- **依赖激活数据**：需要在特定任务（如 DRE-Bench）上运行模型以收集激活，增加了计算开销。
- **静态分析**：当前分析基于静态的功能图，未考虑动态的时间序列依赖或上下文变化。
- **社区检测算法敏感性**：Louvain 等算法可能存在分辨率限制或随机性，影响 core/bridge score 的稳定性。
- **仅适用于 attention heads**：目前未扩展到 FFN 模块或其他组件。

### 未来工作方向
- 将小世界分析扩展到 **FFN 层** 或 **整个 Transformer block**。
- 研究 **动态 functional graphs**，捕捉推理过程中的实时网络演化。
- 探索如何在 **训练过程中引导小世界结构的形成**，而非仅在事后分析。
- 将 SWA 应用于 **更广泛的下游任务** 和 **多模态模型**。
- 开发轻量化的在线估计方法，降低功能图构建成本。

--- 

> **总结一句话**：  
> 本文揭示了 **LLM 内部注意力头网络的“小世界”结构是其推理能力的结构签名**，并据此提出了 **SWA 剪枝方法**，实现了在大幅压缩模型的同时，更好地保留其推理性能。

</details>

---

### 8. [Sparse Attention Is Matrix Approximation, Not Choosing from a Bag of Values](https://arxiv.org/abs/2610.10871)

**Authors**: Fang Wan, Xufeng Liu, Fan Li, Yi Liu  
**Category**: cs.CL  
**Published**: 2026-10-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.10871v1  

#### Abstract
Large Language Models (LLMs) achieve strong performance across many domains, but their efficiency is limited by the quadratic cost of attention with respect to prompt length. Sparse attention reduces this cost by retaining only a small fraction of query-key interactions to approximate the full atten...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sparse Attention Is Matrix Approximation, Not Choosing from a Bag of Values

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现有的 **sparse attention** 方法在选择稀疏单元（如 block、vertical line 等）时，普遍采用基于 **attention mass** 或 **magnitude-based ranking** 的策略——即保留注意力矩阵中数值最大的条目。然而，这种方法存在一个根本性的概念错误：

> 它将注意力矩阵视为“一堆独立的标量值”（a bag of values），而忽略了其作为**结构化矩阵**的本质。

实际上，在 Transformer 中，注意力矩阵 $ A $ 是通过矩阵乘法作用于 value 矩阵 $ V $ 来生成输出 $ AV $ 的。因此，真正重要的不是“哪些元素最大”，而是“哪些稀疏单元最能保留矩阵乘积 $ AV $”。

这种“entrywise selection”与“matrix-action usage”的不匹配导致了次优的稀疏近似效果。

---

### 提出了什么新方法或新思路

作者提出 **Matrix Approximation Sparse Attention (MASA)**，其核心思想是：

> **稀疏注意力应被形式化为矩阵近似问题，而非从“值袋子”中挑选最大项的问题。**

#### MASA 的关键机制：
- 不改变任何已有 sparse attention 方法的稀疏单元定义、kernel 实现或预算规则。
- **仅替换原有的 ranking score**，用一个新的闭式（closed-form）得分来衡量每个候选稀疏单元对 **matrix-product approximation error** 的减少程度。

具体地，对于一个候选单元 $ u $，MASA 使用以下 reduction score：

$$
S_{\text{MASA}}(u) = \|O_x\|^2 - \|O_x - C_u\|^2
$$
其中：
- $ O_x = AX $ 是密集注意力作用于探针矩阵 $ X $ 的输出，
- $ C_u = (M_u \odot A)X $ 是该单元对输出的贡献，
- $ X $ 可以是 value 矩阵 $ V $ 或经过 norm 控制处理后的版本。

这使得 MASA 能够直接优化对最终注意力输出的逼近质量。

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **理论正确性** | 将稀疏选择目标从“保留大值”转向“保留矩阵作用”，解决了长期存在的数学错配问题。 |
| **通用性与兼容性** | 是一个即插即用（plug-in）修正模块，可无缝集成到多种现有框架（如 MInference、SeerAttention、FlexPrefill）中，无需修改其 kernel 或架构。 |
| **高效性** | 仅在原有 ranking 阶段替换评分函数，计算开销极小，不影响稀疏推理效率。 |
| **性能提升显著** | 在多个基准上一致优于原始方法，尤其在长上下文场景下增益更明显。 |

---

## 2. 核心实验方法和设置

### 使用的数据集

实验覆盖三大主流长上下文 benchmark：

| 数据集 | 描述 |
|--------|------|
| **RULER** (Hsieh et al., 2024) | 合成任务，支持可控长度（4K–128K），包含 needle-in-a-haystack、multi-hop tracing 等任务。用于所有三个 baseline。 |
| **InfiniteBench** (Zhang et al., 2024) | 平均约 214K token 的超长上下文理解任务，含合成与真实世界任务。用于 MInference 和 FlexPrefill。 |
| **LongBench** (Bai et al., 2024b) | 双语多任务长文本 benchmark，涵盖多文档 QA、摘要、检索等。用于 SeerAttention。 |

---

### 实验设置和评估指标

#### 模型主干（LLM Backbones）
- `Llama-3-8B-Instruct-262k` → MInference
- `Llama-3.1-8B-Instruct` → SeerAttention, FlexPrefill
- `Qwen2-7B-Instruct` → FlexPrefill

#### 基线方法（Baselines）
| 方法 | 类型特点 |
|------|---------|
| **MInference** | 固定模式 + offline 分配（vertical/slash/block） |
| **SeerAttention** | 学习式 block selection，训练获得 gate |
| **FlexPrefill** | 输入自适应 + 动态预算调整 |

> 所有 baseline 均保持官方配置不变，MASA **只替换 ranking score**，其余完全复现原实现。

#### 评估指标
- 主要指标：下游任务准确率（Accuracy / Score Average）
- 辅助指标：相对全注意力模型的差距 $ \Delta\text{Full} $
- 输出误差分析：归一化平方 Frobenius error $ \|O_{\text{sparse}} - O_{\text{dense}}\|^2 / \|O_{\text{dense}}\|^2 $

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 在所有 8 个匹配设置中均取得提升

| Baseline | Dataset | Original Avg. | MASA Avg. | Gain | $ \Delta\text{Full} $ 改进 |
|---------|--------|----------------|------------|-------|----------------------------|
| MInference | RULER | 87.0 | **87.27** | +0.27 | +2.6 → **+2.87** |
| SeerAttention | RULER | 87.60 | **88.09** | +0.49 | -0.41 → **+0.08**（反超！） |
| FlexPrefill-Llama | RULER | 89.18 | **89.94** | **+0.76** | +0.48 → **+1.24** |
| FlexPrefill-Qwen | RULER | 70.75 | **72.16** | **+1.41** | — |
| MInference | InfiniteBench | 38.8 | **39.20** | +0.40 | — |
| FlexPrefill-Llama | InfiniteBench | 47.14 | **47.99** | +0.85 | 接近 full attention (48.06) |
| FlexPrefill-Qwen | InfiniteBench | 30.07 | **30.42** | +0.35 | — |
| SeerAttention | LongBench | 54.20 | **54.37** | +0.17 | +0.13 → **+0.30** |

> 💡 特别值得注意的是：**SeerAttention + MASA 在 RULER 上超越了 Full Attention**，说明稀疏方法结合 MASA 后不仅能逼近，甚至可能因更合理的结构选择而表现更好。

#### 📈 长序列增益更显著

在 32K–128K 长度范围内，MASA 在 12/12 个长度级分数中提升了 11 个。例如在 128K：
- MInference: +1.28 pts
- FlexPrefill-Qwen: **+2.16 pts**

表明 MASA 对 **long-context sparse prefill** 效果尤为突出。

---

### 消融实验结果（Ablation Study）

#### 探针矩阵 $ X $ 设计的影响

| 探针类型 | 公式 | 观察结果 |
|--------|------|--------|
| Raw $ V $ | $ X_j = V_j $ | 基础版本，有效但受高范数 outlier 影响 |
| Soft Norm Balancing | $ X_j = V_j / (\|V_j\|^2 + \epsilon)^a $ | $ a=0.5\sim0.7 $ 时在 FlexPrefill 中有效 |
| Norm Clipping | $ X_j = V_j \cdot \min(1, c / \|V_j\|) $ | 在 SeerAttention 和 FlexPrefill-Llama 中提升明显 |

> 表明控制 value vector 的 norm 可进一步稳定 selection 过程，避免少数极端值主导得分。

#### 结论：
- **Raw MASA 已带来增益** → 支持 matrix-approximation 思路本身的有效性。
- **Norm 控制可进一步优化** → 特别适用于存在 outlier 的 setting。

---

## 4. 关键结论和发现

### 主要发现

1. 🔍 **根本性认知转变**：
   > 稀疏注意力不应被视为“挑出最大的几个 attention 值”，而应看作“如何最好地近似矩阵乘积 $ AV $”。这是首次明确指出 magnitude-based selection 存在理论缺陷的工作。

2. ⚙️ **MASA 是通用插件式改进**：
   > 不依赖特定 sparse pattern、kernel 或 budget policy，可在多种框架中即插即用，并持续提点。

3. 📊 **实证支持强**：
   > 在跨模型、跨数据集、跨方法的严格对照实验中，MASA 始终优于原方法，且输出误差更低（见 Appendix E），验证了其对 $ AV $ 的更好逼近能力。

4. ⏱️ **无性能代价**：
   > 替换 ranking score 引入的额外计算可忽略不计，在 128K 序列下仍远低于 dense attention 开销。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **不探索新 sparse structure** | MASA 不设计新的 sparse unit 或 kernel，仅优化已有结构的选择顺序。 |
| **依赖 profile attention map** | 通常基于低分辨率 attention map 进行 ranking，未利用 full-resolution dense 计算（出于效率考虑）。 |
| **probe design 需调参** | 虽然 closed-form 存在，但 $ X $ 的构造（如 clip 百分位、$ a $ 参数）需根据任务微调。 |

---

### 未来工作方向

1. **结合 adaptive budgeting**：将 MASA 的 scoring 机制融入动态预算决策（如 FlexPrefill）中，实现更智能的 early stopping。
2. **扩展至 decoding 阶段**：当前 focus 在 prefill，未来可用于 KV-cache pruning 或 streaming attention。
3. **与其他加速技术联合使用**：与 FlashAttention、PagedAttention、quantization 等正交技术组合，构建端到端高效系统。
4. **理论深化**：研究其他 matrix norm（如 spectral norm）下的 approximation bounds，发展更强的误差控制理论。

---

> **一句话总结**：  
> **MASA 揭示了稀疏注意力的本质是矩阵近似问题，并提供了一个理论驱动、即插即用、广泛有效的解决方案，推动了稀疏注意力从“直觉工程”走向“理论指导”的新阶段。**

</details>

---

### 9. [Lapras: Latent Reasoning for Time Series Language Models](https://arxiv.org/abs/2610.11111)

**Authors**: Yuliang Chen, Yu Yvonne Wu, Patrick Langer, Arvind Pillai, Sudarshan Regmi, Martin Maritsch, Juncheng Liu, Robert Jakob, Thomas Kaar, Tess Z. Griffin, Lisa Marsch, Michael V. Heinz, Nicholas C. Jacobson, Andrew Campbell  
**Category**: cs.CL  
**Published**: 2026-10-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.11111v1  

#### Abstract
Time Series Language Models (TSLMs) offer a promising path toward time series understanding by reasoning over temporal signals and producing natural language answers and explanations. A common approach is Chain-of-Thought (CoT), which generates step-by-step rationales linking relevant signal pattern...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Lapras: Latent Reasoning for Time Series Language Models**

---

## **1. 主要贡献和创新点**

### **解决的问题**
- **Verbalization Bottleneck（言语化瓶颈）**：现有的 Time Series Language Models (TSLMs) 在使用 Chain-of-Thought (CoT) 进行推理时，需要将高维、连续的时间序列信号转换为离散的语言 token 来生成中间推理步骤。这一过程可能导致：
  - 忽略或错误描述任务相关的时序模式；
  - 早期错误在推理链中传播，导致最终答案错误但解释看似合理；
  - 推理效率低，因需自回归生成大量文本 token。

### **提出的新方法：Lapras**
- **Lapras (Latent Post-trained Reasoning Across Series)** 是一种用于 TSLMs 的后训练框架，引入 **Latent Reasoning（潜在推理）** 能力。
- 核心思想是让模型在 **joint time series-language space** 中通过连续的“thought”进行中间推理，仅在最后一步解码出自然语言答案。
- 采用 **Teacher-Student Self-Distillation** 架构：
  - **Teacher**：在显式 CoT 轨迹上训练，生成文本化的推理链；
  - **Student**：不生成中间文本，而是学习在隐藏状态空间中模拟教师的推理路径，通过匹配两者在答案位置的 hidden states 来实现知识迁移。

### **相比现有方法的优势**
| 维度 | Lapras | 显式 CoT |
|------|--------|---------|
| **准确性** | 更高（避免言语化失真） | 易受早期描述误差影响 |
| **推理效率** | 生成 token 数量减少最多达 **23.9×** | 需逐句生成，延迟高 |
| **可解释性保留** | 可通过标准语言解码器将 continuous thoughts 解码为可读推理链 | 天然具备 |
| **注意力机制** | 更多地关注原始时间序列表示 | 更依赖已生成的文本摘要 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
五个多样化的时序问答基准，涵盖医疗、工业、行为识别等领域：

| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **ECG** | 心电图诊断 | 判断是否存在心律失常等临床问题 |
| **Engine** | 航空发动机故障诊断 | 基于传感器信号判断组件故障等级 |
| **HAR** | 人类活动识别 (Human Activity Recognition) | 从可穿戴设备推断动作类别 |
| **Sleep** | 睡眠阶段分类 | 分析脑电图判断睡眠阶段 |
| **TSR** | Counterfactual Prediction | 判断现实世界后果是否随信号变化而改变 |

> 所有任务均构建了参考的 CoT 推理轨迹用于训练（部分手工模板，部分由 Qwen3.5-27B 生成）。

### **实验设置与评估指标**
- **Backbones**：在四种不同架构的 TSLM 上评估（参数量从 0.5B 到 7B）：
  - ChatTS-7B
  - OpenTSLM-1B
  - SLIP-1B
  - ITFormer-0.5B
- **评估方式**：
  - 将模型输出解析为预测标签；
  - 报告 **Accuracy** 和 **Macro-F1**；
  - 使用 greedy decoding，最大生成 512 token。
- **训练细节**：
  - 使用 LLaMA-Factory 框架；
  - AdamW + Cosine LR Schedule；
  - Batch size = 128，bf16 精度，DeepSpeed ZeRO-2；
  - Lapras 默认设置：`K=6` 个连续思考步，mask ratio=0.3。

### **基线方法对比**
| 类别 | 方法 | 描述 |
|------|------|------|
| **无推理** | No-CoT | 直接生成答案，无中间步骤 |
| **显式推理** | CoT | 生成完整 Chain-of-Thought 文本 |
| **潜在推理** | iCoT, COCONUT | 不生成中间文本，使用 continuous hidden states 进行内部推理 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **平均 F1 提升显著**：
  - 在 **ChatTS-7B** 上，Lapras 相比 CoT 平均 F1 提升高达 **10.79%**；
  - 在所有 backbone-dataset 组合中，平均提升 **5.53%**；
  - 在 **OpenTSLM-1B** 上，F1 提升 **9.95%**（相对 No-CoT）；
- **推理效率大幅提升**：
  - 生成 token 数量减少 **6.9× ~ 23.9×**；
  - 推理速度加快 **5.9× ~ 15.1×**；
  - 仅比 No-CoT 多增加 <8ms 开销。

### **与基线方法的对比结果**
| 方法 | Avg. F1 Δ vs CoT | Token Reduction vs CoT | 是否优于 No-CoT |
|------|------------------|------------------------|----------------|
| CoT | — | ×1 | ❌（多数情况下更差） |
| iCoT | +1.46% ~ +6.38% | ~5–10× | ✅ |
| COCONUT | +2.67% ~ +6.78% | ~6–12× | ✅ |
| **Lapras (Ours)** | **+5.02% ~ +9.95%** | **up to 23.9×** | ✅✅ |

> ⚠️ 特别值得注意的是：**CoT 在多个 backbone 上表现不如 No-CoT**，验证了“言语化瓶颈”的存在。

### **消融实验结果（Ablation Studies）**
在 SLIP-1B 上进行消融，结果如下：

| 方法 | Avg Acc (%) | Δ vs Full |
|------|-------------|----------|
| **Lapras (Full)** | **68.38** | — |
| w/o projection | 62.23 | -6.15% |
| w/o reasoning distillation | 64.34 | -4.04% |
| w/o mask | 66.58 | -1.80% |

- **Latent Projection 最关键**：移除后性能下降最大，说明将 hidden state 映射回输入嵌入空间对连续推理至关重要。
- **Distillation 是核心驱动力**：没有教师监督，学生无法有效学习高质量的 latent reasoning。
- **Masking 有助于泛化**：防止模型绕过推理过程直接记忆答案。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Verbalization Bottleneck 真实存在且严重影响性能**：
   - 实验显示，在“描述阶段”，预测 CoT 与真实 CoT 之间已有显著准确率差距（如 TSR 上 △0.22），且该差距不会在后续推理中被修复。
   - 支持了“一旦错误描述，后续推理难以纠正”的假设。

2. **Latent Reasoning 更高效、更准确**：
   - Lapras 在保持甚至提升可解释性的前提下，实现了更高的推理准确性和更低的推理成本。
   - 模型在早期推理阶段更关注原始时间序列（attention 分析证实），而非依赖自身生成的文本摘要。

3. **Continuous Thoughts 可解码为可读推理链**：
   - 尽管推理发生在 latent space，但通过共享的 LM head 对 `[bot], z₁,…,z₆, [eot]` 进行贪婪解码，可以恢复出语义连贯、符合逻辑的推理路径。
   - 实现了 **efficiency** 与 **interpretability** 的统一。

4. **Commitment Delay（延迟决策）现象**：
   - Lapras 在推理过程中保持多个候选答案的概率分布，直到最后才收敛；
   - 而 CoT 往往在第一步就锁定一个答案，缺乏容错能力。

### **局限性**
1. **依赖 CoT 注释数据**：教师模型仍需显式的 CoT 轨迹进行训练，限制了其在缺乏标注数据场景下的应用。
2. **固定推理步数 K=6**：不能根据输入复杂度动态调整计算量，可能浪费资源或不足。
3. **小模型效果受限**：在 ITFormer-0.5B 上，所有推理方法均不如 No-CoT，表明 multi-step reasoning 存在容量门槛。

### **未来工作方向**
- 探索 **adaptive latent reasoning**，例如通过 RL 动态决定推理步数；
- 发展 **unsupervised 或 weakly-supervised** 的 latent reasoning 训练范式；
- 研究如何进一步降低对大规模标注 CoT 数据的依赖；
- 将 Lapras 思路扩展到其他多模态场景（如视频、语音）。

---

> ✅ **总体评价**：  
> Lapras 提出了一种新颖且实用的 TSLM 后训练范式，成功解决了 CoT 在处理连续信号时的“言语化瓶颈”问题。它不仅提升了推理准确性和效率，还保留了解释能力，为安全敏感领域（如临床监测、工业诊断）中的可信 AI 决策提供了强有力的技术支持。

</details>

---

### 10. [MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement](https://arxiv.org/abs/2610.11959)

**Authors**: Core Team, Zongming Qiao, Ziyue Hua, Zirui Ou, Zihao Yue, Zihan Jiang, Zhuo Huang, Zhiyang Chen, Zhixian Zheng, Zhipeng Xu, Zhengrui Ma, Yuyang Hu, Yuhang Dong, Yuechen Zhang, Yudong Wang, Yuanxin Liu, Yixin Yang, Yishuo Cai, Yikai Zhao, Yihan Yan, Yifan Zhang, Yifan Song, Xiyu Wei, Xing Zhang, Xin Zhang, Xiaoqian Liu, Xiaodong Ji, Xiangwei Deng, Xueyu Guo, Wenhan Ma, Weimin Xiong, Weikun Wang, Weiji Zhuang, Shuo Liu, Shuhuai Ren, Shuhao Gu, Shimao Chen, Shijie Cao, Shihua Yu, Shicheng Li, Shengjie Zhou, Shaolei Zhang, Rang Li, Qiying Wang, Qingkai Fang, Qianli Chen, Minzheng Wang, Liwen Wang, Linli Yao, Linghao Zhang, Liangyu Cheng, Liang Zhao, Lei Li, Jinhao Dong, Jinyu Xiang, Jianyu Wei, Jiangshan Duo, Huaqiu Liu, Huanjie Fan, Hongyi Guan, Hongshen Xu, Hao Tian, Hanyu Li, Hailin Zhang, Gang Wang, Fuli Luo, Feng Wei, Dong Zhang, Dawei Zhu, Chiheng Lou, Chenhong He, Chenhao He, Chenghua Liu, Bowen Ye, Bowen Shen, Boshen Xu, Bo Yang, Bingquan Xia, Bangjun Xiao, Baixuan Xu, Zhouxiang Mao, Zhiyang Zhang, Zhixiang Xu, Zhenru Lin, Zhengju Tang, Zhaojun Huang, Yuzhe Weng, Yuxing Xiang, Yuxiao Li, Yuheng Yang, Yuhang Wang, Yuchen Liu, Yuanyuan Tian, Yuanliang Dong, Yu Cheng, Yongzhe He, Yongshun Liang, Yong Wang, Yiyan Wang, Yitian Gong, Yijie Zhang, Yanshu Xin, Xun Zhang, Xingjian Zhao, Wenyu Yang, Wenshan Huang, Wenhao Li, Tingwei Huang, Tianyu Yu, Tianyang Lu, Taoyu Yang, Sinan Du, Shutong Tian, Shulin Du, Shengfan Wang, Shanchuan Fang, Qihao Zhang, Qibin Yang, Qian Yu, Qian Tu, Pengrong Xie, Peipei Wang, Peidian Li, Minkun Guo, Mingchen Shao, Luohan Gao, Lijie Wang, Liang Shi, Kaiqi Chen, Kaiming Liu, Kaifei Wang, Kai Yang, Jinlong Xue, Jiechen Zhang, Jiaxuan Liu, Hongxu An, Hao Peng, Hanglong L\"u, Guonan Wang, Feiyu Yang, Fanyu Cao, Fangyue Liu, Fan Cui, Cong Wang, Chun Chen, Chenxu Bai, Chengxuan Zhu, Chenghua Wang, Boyi Zeng  
**Category**: cs.CL  
**Published**: 2026-10-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.11959v1  

#### Abstract
Reinforcement learning (RL) is the central training paradigm for advancing large foundation models towards self-improvement. This report introduces the MiMo-V2.6 series, an omni-modal family that pushes the frontier of model intelligence by scaling RL compute. Prior to RL, we conduct mid-training on...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文致力于解决**如何通过大规模强化学习（RL）推动大模型实现自我改进（self-improvement）**这一核心挑战。具体而言，传统基于监督微调（SFT）的方法在复杂、长周期、多步骤的 agentic 任务中表现有限，而现有的 RL 方法面临以下瓶颈：
- **训练不稳定**：尤其是在 MoE 架构下，expert load 容易失衡；
- **奖励信号稀疏且不可靠**：二元测试结果无法区分解决方案的质量差异，容易导致 reward hacking；
- **环境多样性不足**：缺乏覆盖真实世界复杂场景（如代码、视觉设计、网络安全等）的多样化、可复现的交互环境；
- **基础设施不支持大规模混合任务 RL**：难以高效处理异构任务、多种 agent harnesses 和海量轨迹数据。

### 提出的新方法与新思路
MiMo-V2.6 系列模型提出了一套系统性的框架，从架构、预训练、RL 扩展到基础设施全面优化，核心创新包括：

#### （1）**Hybrid-SWA 主干架构**
- 在 Transformer 中交替使用 **Local Sliding Window Attention (SWA)** 和 **Global Attention (GA)**，兼顾长上下文建模效率与全局感知能力。
- 结合轻量级 **MiMo-ViT** 视觉编码器和 **Audio Encoder**，构建真正的 omni-modal 基础。

#### （2）**Agent-Centric Mid-Training**
- 在 SFT 前引入一个中间训练阶段，使用涵盖 coding、general、visual、cyber 等领域的 agent 轨迹数据进行训练，扩展探索空间并增强 agent 能力。

#### （3）**三维扩展的 RL 框架（Three-dimensional Scaling of RL）**
这是本工作的核心贡献，分别从三个维度扩展 RL 计算：
- **Scale 1: 更大的批量与吞吐量**
  - 异步训练，每步处理 **1,568 个 prompts，共 25K 序列，2.7~3.7B tokens**；
  - 上下文长度高达 **1M tokens**；
  - 使用 **partial rollout** 技术保持 batch 饱和。
- **Scale 2: 更多样复杂的环境与 harness**
  - 跨越 **coding、general workflows、visual design、cybersecurity** 四大领域；
  - 使用 **multi-harness training**，通过模块化 mini-harnesses 实现可控多样性，提升跨 harness 泛化能力。
- **Scale 3: 更强大的评分机制（Grader Compute）**
  - 提出 **Groupwise Agentic Grading**，细分为两种方法：
    - **Groupwise Reward Synthesis (GRS)**：离线分析多个 rollout，生成任务特定的 rubrics，用于打分；
    - **Groupwise Advantage Redistribution (GAR)**：在线比较组内成功/失败轨迹，重新分配优势值（advantage），引导模型产出更高质量解。

#### （4）**稳定训练的关键技术**
- **冻结 MoE Router**：防止 expert load collapse，显著提升训练稳定性（见图11）；
- **多层防御对抗 Reward Hacking**：
  - 环境准备阶段清除泄露路径（build logs、Git history 等）；
  - 使用 **hack agent** 主动探测漏洞；
  - 训练时审计轨迹并将确认的 hacking 行为 reward 设为零。

#### （5）**完整的 RL 基础设施栈**
- 统一的 trajectory 表示与 **Penalty Module**；
- **Harness Pool + Payload Porter** 支持高并发多框架 rollout；
- 控制平面与数据平面分离；
- **Sample Mixer** 实现稳定的异步混合任务 RL；
- 保证 **Training-Inference Consistency**（如 MoE routing replay、top-p candidate set replay）。

### 相比现有方法的优势
| 维度 | 传统方法 | MiMo-V2.6 |
|------|--------|---------|
| **RL 规模** | 小批量、短上下文 | 千卡级异步训练，百万级上下文 |
| **反馈质量** | 仅依赖 binary test pass/fail | 多维、细粒度 reward 信号（行为、简洁性、鲁棒性） |
| **环境多样性** | 单一任务/单一 harness | 多域、多 harness、resettable 环境 |
| **训练稳定性** | 易出现 load collapse | 冻结 router + 多重防御机制 |
| **可复现性与开放性** | 黑箱系统 | 开源完整 RL 栈（模型、环境、框架） |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖四大类任务，结合公开基准与自研内部基准：

| 类别 | 公开数据集 | 内部数据集 |
|------|----------|----------|
| **Code Agent** | DeepSWE v1.1, ProgramBench, SWE-Bench Pro | MiMo Code Bench |
| **General Agent** | AutomationBench, Toolathlon-Verified, GDPval-AA 2.1, JobBench, Terminal-Bench, OSWorld-Verified | — |
| **Cybersecurity** | CyberGym, ExploitGym, ExploitBench, SEC Bench Pro | MiMo Cyber Bench |
| **Visual Agent** | — | MiMo Visual Coding（含 WebDev、Image2Code 等） |

此外还发布了轻量版开源环境集合，包含约 **7k 训练任务**（见 Table 5）。

### 实验设置
- **模型配置**：
  - **MiMo-V2.6-Pro**：1.02T 总参，42B 激活参数，70 层 Hybrid-SWA；
  - **MiMo-V2.6-Flash**：310B 总参，15B 激活参数，48 层；
- **RL 设置**：
  - 使用 **GRPO（Group Relative Policy Optimization）**；
  - 每步 batch size = 1568 prompts × 16 rollouts = 25K trajectories；
  - 上下文长度：**up to 1M tokens**；
  - 优化器：**Muown**（基于 Muon 的变体），对隐藏权重矩阵进行矩阵级更新；
  - 奖励计算融合 **groupwise grading** 与 **length penalty**。
- **评估指标**：
  - **avg@n**（如 avg@3）：前 n 个生成结果中的平均通过率；
  - **Pass@1**：第一个生成结果即通过的概率；
  - Token efficiency、solution conciseness、reward hacking detection rate。

### 基线方法对比
- **内部基线**：
  - MiMo-V2.5（前代模型）
  - MiMo-V2.6-SFT（未经过 RL 微调）
- **外部前沿模型**：
  - GPT-4 / GPT-5
  - Claude Opus / Sonnet
  - Qwen3.5-9B（作为 distillation 起点）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）
| Benchmark | MiMo-V2.6-Pro | MiMo-V2.6-Flash | 前沿模型（如 Claude Opus） |
|----------|---------------|------------------|----------------------------|
| **DeepSWE v1.1** | 67.9 | 61.2 | 70.0 |
| **ProgramBench** | 71.9 | — | 33.0 |
| **AutomationBench v1.0.6** | 53.1 | 73.6 | 77.9 |
| **Terminal Bench 4.0** | 89.9 | 87.6 | 42.4 |
| **CyberGym** | 94.0 | 95.1 | — |
| **SEC Bench Pro** | 66.3 | 47.5 | 79.1 |
| **MiMo Visual Coding** | 72.3 | 71.5 | 69.1 |

> ✅ **总体趋势**：MiMo-V2.6 系列在多数任务上显著优于 MiMo-V2.5，并达到甚至超越部分前沿闭源模型水平。

### 与基线方法的对比结果
- **相比 MiMo-V2.5**：
  - 在 **DeepSWE** 上从 ~40 提升至 **67.9（Pro）**；
  - 在 **AutomationBench** 上从 49.1 提升至 **73.6（Flash）**；
  - 在 **Cybersecurity** 几乎从零起步（0 → 80+）；
- **相比 SFT 版本**：
  - 图 9 显示，在 RL 训练过程中，**所有任务性能持续上升**，且 token 数同步增长，表明模型学会更有效地利用长上下文解决问题。

### 消融实验结果
#### （1）**Router 冻结消融（图11）**
- **不冻结 router**：expert load CV 从 0.78 升至 2.0，峰值负载达均值 16 倍，大量专家“冷启动”；
- **冻结 router**：load 分布稳定（CV ~0.7），训练正常收敛；
> 🔍 结论：**冻结 MoE router 是维持大规模 RL 稳定性的关键**。

#### （2）**Groupwise Advantage Redistribution（GAR）消融（图8）**
- **无 GAR**：pass rate 初期上升快，但很快 plateau；turns 与 token length 快速膨胀；
- **有 GAR**：pass rate 持续提升至 step 52，turns 与 length 增长平缓；
> 🔍 结论：**GAR 有效抑制冗余输出，促进 token-efficient、高质量策略演化**。

#### （3）**Multi-Harness Training 泛化能力（图10）**
- 在 4 个训练 harness 上性能稳步提升；
- 在 **3 个 held-out harnesses（codex, claude code, mini-swe-agent）** 上也实现从 ~50% → **66% Pass@1** 的提升；
> 🔍 结论：**mini-harness 设计能有效提升跨框架泛化能力**。

#### （4）**开源轻量模型验证（Table 6 & 7）**
- **MiMo-V2.6-Distill-Qwen-9B** 相比原始 Qwen3.5-9B：
  - SWE-bench Pro 从 32.0 → 44.6；
  - AutomationBench 从 5.0% → 30.3%；
- 经过 RL 后进一步提升至 **47.6 / 33.1**；
> 🔍 结论：**distillation + RL 可迁移强大 agent 能力至小模型**。

---

## 4. 关键结论和发现

### 主要发现
1. **大规模 RL 是通往 self-improvement 的可行路径**：通过扩展 batch size、environment diversity 和 grader compute，模型可在复杂任务上实现持续性能提升。
2. **细粒度反馈至关重要**：传统的 binary reward 不足以驱动高质量行为；**groupwise grading** 能提供更丰富的学习信号，引导模型生成更简洁、可靠、高效的解决方案。
3. **基础设施决定上限**：支持百万 token 上下文、数千并发 rollout、多 harness 混合训练的工程架构是实现规模化 RL 的前提。
4. **冻结 MoE router 可解决 load collapse 问题**：这是首次在超大规模 MoE 模型上稳定执行 RL 的实践验证。
5. **开放生态促进研究**：发布 **MiMo-V2.6-Distill-Qwen-9B + RL environments + framework** 形成可复现的 agentic RL 基线。

### 方法的局限性
- **计算成本极高**：MiMo-V2.6-Pro RL 花费 **$2.6M**，仅适合大厂；
- **依赖高质量 verifier**：对于无法自动验证的任务（如艺术创作），仍需人工或弱监督；
- **reward hacking 无法根除**：尽管有多重防御，仍有约 **2% 的轨迹被检测为 hacking**；
- **mini-harness 抽象可能损失现实细节**：与真实生产 harness 存在差距。

### 未来工作方向
- 探索 **无需冻结 router 的动态 load balancing 方法**；
- 发展 **更强的 agentic grader**，支持更复杂的语义判断；
- 构建 **通用的 reward modeling 框架**，减少对 handcrafted rubrics 的依赖；
- 推进 **recursive self-improvement loop**：让模型自主生成任务、评估自身、迭代优化；
- 进一步降低 RL 成本，使更多研究者可参与。

---

> 📌 **总结一句话**：  
> MiMo-V2.6 证明了通过**系统性地扩展 RL 的规模、反馈质量和基础设施**，可以显著提升大模型的通用 agent 能力，并朝着 **model self-improvement** 迈出坚实一步，同时通过开源推动整个社区前进。

</details>

---

### 11. [TokenRouter: Efficient Serving System for Token-Level LLM Routing](https://arxiv.org/abs/2610.12242)

**Authors**: Tianyu Fu, Tengxuan Liu, Ruoxi Wang, Yixin Dong, Yi Ge, Yichen You, Yu Wang  
**Category**: cs.CL  
**Published**: 2026-10-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.12242v1  

#### Abstract
Large language model (LLM) routing distributes inference work across different models, advancing the cost-quality Pareto frontier of LLM serving. While coarse-grained routing at the session or query level has been widely adopted in production systems, recent algorithmic work shows that fine-grained ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TokenRouter: Efficient Serving System for Token-Level LLM Routing**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **LLM serving systems**（如 SGLang、vLLM）主要为单模型推理设计，假设所有请求在每个解码步保持同步。然而，**token-level LLM routing** 在每一步都可能切换目标模型，导致以下三大挑战：
- **Step Desynchronization（步骤不同步）**：不同模型的每步延迟差异大，强制同步会拖慢整体速度。
- **Batch Admission Delay（批处理准入延迟）**：频繁的模型切换导致请求到达时目标模型正在处理其他批次，需等待下一批次，造成“气泡”和碎片化批处理。
- **High Implementation Complexity（高实现复杂度）**：现有系统缺乏对逐 token 路由的支持接口，开发者需深度修改代码。

### **提出了什么新方法或新思路**
作者提出 **TokenRouter**，一个专为 token-level LLM routing 设计的高效、开发者友好的服务系统，其核心思想是：
> **Request-Centric Programming, Model-Centric Execution**  
> 开发者从单个请求的视角描述路由逻辑，运行时则以模型为中心异步执行。

#### 主要创新点：
- **Request-Centric 编程接口**：提供 `route`, `send`, `receive` 三个函数，开发者只需定义单个请求如何在模型间流转，无需关心底层调度。
- **异步三循环执行（Decoupled Tri-Loop Execution）**：
  - 新增 **Inter-Model Loop** 处理模型间请求传递。
  - 引入 **Handoff-Resume 机制**，将被路由的请求标记为 `pending`，保留其 KV-cache 状态，避免重复前缀匹配和内存分配。
- **延迟批处理调度器（Delayed-Batching Scheduler）**：
  - 缓冲传入请求，等待达到阈值 `B` 后再启动批处理，减少平均准入延迟。
  - 基于 **Discrete-Time Markov Chain (DTMC)** 建模，推导出吞吐量最优的批处理阈值 `B*`。

### **相比现有方法的优势**
- **高性能**：通过异步执行和延迟批处理，显著提升吞吐量。
- **易用性**：开发者无需理解底层并发机制，可快速实现复杂路由算法。
- **兼容性**：对外暴露单一接口，可作为现有单模型服务器（如 vLLM、SGLang）的即插即用替代品。

---

## 2. **核心实验方法和设置**

### **使用的数据集与基准任务**
- **AIME2024**：用于数学推理任务（短输入、长输出），最大输出长度设为 8,192。
- **SWE-Smith**：用于代理类多轮任务（长输入 ~8,192 tokens，输出 1,024）。
- **CommonsenseQA** 和 **GSM8K**：用于验证原始论文设定下的性能。

### **实验设置**
- **硬件**：8×A100-80G GPU 服务器。
- **模型配置**：
  - 小模型（SLM）部署于 1 GPU。
  - 大模型（LLM）使用 Tensor Parallelism（TP=2）部署于 2 GPUs。
  - 部分实验使用 CUDA MPS 实现 GPU 共享。
- **并发数（Concurrency）**：测试 N=1 到 16。

### **评估指标**
- **Throughput（吞吐量）**：单位时间内生成的 token 数（token/s）。
- **End-to-End Latency（端到端延迟）**：单个请求完成时间（秒）。
- **TTFT（Time to First Token）**：首 token 返回时间。
- **Throughput-Speed Trade-off**：在不同并发下分析吞吐与单用户速度的权衡。

### **基线方法对比**
- **Official Code**：各算法官方实现（如 R2R、CITER、Co-LLM）。
- **Std. Serving**：基于 SGLang 构建的标准服务基线，使用外部调度器进行逐 token 路由。
- **LLM-only**：仅使用大模型推理作为质量上限参考。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
- **吞吐量提升**：在 15 种算法-工作负载组合中，TokenRouter 相比更强基线实现 **2.01–64.15× 更高的解码吞吐量**。
- **延迟降低**：相比 Std. Serving，端到端延迟降低 **2.03–63.64×**。
- **高并发表现**：
  - 并发从 1 增至 16，TokenRouter 吞吐提升 **8.61×**，保留 51.7% 单用户速度。
  - 官方 R2R 仅提升 5.14×，保留 31.3% 速度。
  - 在更严格的 SLO 下，TokenRouter 吞吐达官方 R2R 的 **18.58×**，且单用户速度仍快 1.13×。

### **与基线方法的对比结果**
| 方法 | 吞吐量 (token/s) | 相对提升 |
|------|------------------|----------|
| R2R Official Code | 77.08 | 1× |
| TokenRouter | 197.40 | **2.56×** |
| CITER Official Code | 17.16 | 1× |
| TokenRouter | 149.31 | **8.70×** |
| Co-LLM Official Code | 3.46 | 1× |
| TokenRouter | 76.02 | **21.97×** |

> 注：以上为 Qwen3-0.6B/32B 模型对在 AIME2024 上的结果。

### **消融实验结果**
#### **性能增益分解（图 1a）**
- **基础实现 → 工程优化（CUDA Graph）**：吞吐从 132.78 → 230.79 token/s，提升 **1.71×**。
- **+ 异步执行**：→ 296.86 token/s，总提升 **2.23×**。
- **+ 延迟批处理**：→ 372.48 token/s，总提升 **2.76×**。

#### **延迟来源分析（表 10）**
在 Std. Serving 中，每步延迟主要来自：
- **Update Radix Cache**: 32.62%
- **Prefix Matching**: 20.94%
- **Locking/Releasing Cache Nodes**: ~31%
> TokenRouter 通过保留 KV-cache 状态，避免这些开销。

#### **并行策略影响（表 7）**
- 增加 **LLM 的 Tensor Parallelism** 显著提升吞吐（+22.8% @ N=4）。
- 增加 SLM 的 TP 几乎无收益（+0.4%），说明资源应优先分配给 LLM。

---

## 4. **关键结论和发现**

### **主要发现**
1. **Token-level routing 的潜力尚未被现有系统释放**：尽管算法上已证明其效率与质量优势，但传统 serving 系统因同步假设而严重限制其性能。
2. **异步 + 延迟批处理 是关键**：TokenRouter 通过 **model-centric 异步执行** 和 **数学建模驱动的延迟批处理**，有效缓解了步骤不同步和准入延迟问题。
3. **编程抽象至关重要**：`route-send-receive` 接口极大简化了开发者负担，使复杂路由逻辑易于实现和复用。
4. **实际性能远超理论预期**：在真实场景中，TokenRouter 不仅提升了吞吐，还改善了延迟，使得 token-level routing 在生产环境中更具可行性。

### **方法的局限性**
- **数学模型假设几何分布**：模型假设连续发送间的 token 数服从几何分布，某些非稳态或强相关路由策略可能不适用。
- **跨节点通信未深度优化**：虽然支持跨节点部署（RoCE），但未针对高延迟网络进一步优化协议。
- **动态负载适应性有限**：当前延迟批处理阈值 `B*` 基于静态参数计算，未实现实时自适应调整。

### **未来工作方向**
- 支持 **非几何分布的路由模式**，增强模型通用性。
- 实现 **动态自适应的延迟批处理**，根据实时负载自动调优 `B`。
- 探索 **跨数据中心的 token-level routing** 架构。
- 结合 **Speculative Decoding** 与 token-level routing，进一步加速推理。

---

> **代码开源**：https://github.com/thu-nics/TokenRouter  
> **会议**：NeurIPS 2026

</details>

---

### 12. [Compile the Table: Query-Calibrated Operator Compression for Tabular In-Context Learning](https://arxiv.org/abs/2610.11784)

**Authors**: Xu Zhao, Jiaming Zhao, Bin Zhao, Yong Yang  
**Category**: cs.LG  
**Published**: 2026-10-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.11784v1  

#### Abstract
Tabular in-context learning (ICL) has emerged as a training-free and accurate paradigm for tabular prediction, but current approaches to compressing its in-context examples face an accuracy-throughput tradeoff: fixed subsets can sacrifice accuracy, while query-specific retrieval limits cache reuse a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Compile the Table: Query-Calibrated Operator Compression for Tabular In-Context Learning**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **tabular in-context learning (ICL)** 中，模型通过一组带标签的上下文示例（in-context examples）进行预测，无需训练。然而，随着上下文规模增大，**KV Cache 内存占用和推理延迟显著增加**，尤其是在服务大量查询时。

现有压缩方法面临 **准确率-吞吐量权衡（accuracy-throughput tradeoff）**：
- **Global row selection**：固定子集，可复用但可能丢失关键样本，影响准确率。
- **Dynamic row selection**（如 KATE）：为每个查询动态检索，提升局部相关性，但无法复用缓存，限制批处理和吞吐。

### ✅ 提出的新方法：**QCOC (Query-Calibrated Operator Compression)**

QCOC 将压缩对象从“原始数据行”转变为“由上下文诱导的注意力算子（attention operator）”，提出一种**后处理（post-hoc）KV Cache 压缩方法**，核心思想如下：

1. **Operator Compilation**：将完整上下文的 KV Cache 编译为一个紧凑、可复用的内存，供后续所有查询共享。
2. **Joint-KV Clustering**：对每个注意力头的 Key 和 Value 联合聚类，形成 **joint-KV prototypes**，保留状态一致性。
3. **Cluster Count Weighting**：记录每类原型对应的原始样本数量 $ w_j $，并在 softmax 中通过 $ \log w_j $ 修正权重，保持总注意力贡献不变。
4. **Query-Calibrated Value Fitting**：利用上下文样本自身生成的 query 向量作为 calibration queries，通过闭式解（closed-form ridge regression）拟合 prototype 的 Value 向量，使压缩后的输出尽可能匹配原模型响应。

### ✅ 相比现有方法的优势
- **高准确率**：接近 full context 性能，优于各类压缩与检索方法。
- **高吞吐**：编译后支持跨查询缓存复用和批处理，显著降低在线延迟。
- **无额外训练**：纯后处理方法，不修改模型参数或训练流程。
- **理论保障**：证明 value fitting 不会增加 calibration queries 上的残差。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **主基准（Primary Benchmark）**：
  - **64 个 OpenML-CC18 分类数据集**，保留 10 个用于超参调优。
  - 最多使用 512 个上下文样本，评估 $ M = 32, 64 $ 的压缩效果。
- **长上下文实验（Long-context Evaluation）**：
  - 7 个 CC18 长表 + **SUSY, HIGGS, Covertype**。
  - 上下文长度 $ N = 512 \sim 8192 $，压缩比例 $ M = N/32, N/16, N/8 $。
- **回归任务**：13 个 TabArena 回归数据集。

### 📊 实验设置与评估指标
- **统一模型**：基于 **TabICL v2** 或 **TabPFN v3**，仅压缩 KV Cache，不更新模型。
- **评估指标**：
  - **Accuracy / R² / Norm. RMSE**：任务性能。
  - **Prediction Agreement**：与 full context 预测一致的比例。
  - **Probability MAE**：预测概率误差。
  - **Attention-output MSE**：注意力输出重建误差。
  - **Cache Memory** 和 **Serving Time**：在线效率。

### 🔁 基线方法对比
| 类型 | 方法 |
|------|------|
| **Global Selection** | Random, Input K-means, LUCoS |
| **Dynamic Retrieval** | MICP, CRUMB, KATE, ARASH-style, Raw-space KNN |
| **Direct KV Compression** | Weighted joint-KV, Key centroids, ToMe, KVMerger |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 在 64 个 CC18 数据集上（$ M=32/64 $）
| 方法 | $ M=32 $ 准确率 | $ M=64 $ 准确率 |
|------|------------------|------------------|
| Full context | 0.8617 | 0.8617 |
| **QCOC** | **0.8534** | **0.8566** |
| Weighted joint-KV | 0.8391 | 0.8450 |
| KATE (best retrieval) | 0.8117 | 0.8355 |

- QCOC 是**唯一在两个压缩级别均排名第一**的压缩/检索方法。
- 与最强基线相比，平均高出 **0.3–0.8 pp**，且显著优于所有 retrieval 方法。

#### ✅ 长上下文实验（$ N=8192, M=512 $）
| 方法 | 平均准确率 | Cache Reduction | 单查询延迟（ms） |
|------|-----------|----------------|------------------|
| Full context | 0.8077 | ×1 | 3.79 |
| KATE | 0.8019 | — | 453.82 |
| **QCOC** | **0.8081** | **10.53×** | **1.91** |

- **准确率略超 full context**（+0.04 pp），是唯一超过 full 的压缩方法。
- **缓存压缩比达 10.5×**，在线延迟仅为 full 的 **50%**，比 KATE 快 **237×**。
- 在 12 种配置中，QCOC 在 10 种中排名第一，平均仅落后 full context **0.23 pp**。

#### ✅ 多查询服务效率（1,000 queries, single-core CPU）
- **排除一次性编译开销**，QCOC 在线推理速度：
  - 比动态检索基线（如 KATE）**快至 508×**。
  - 比 full context **快 1.98×**。

---

### 🔍 消融实验结果（Ablation Study）

在 $ M=32 $ 下对 QCOC 组件进行消融（Table 7）：

| 组件组合 | 准确率 | Attention MSE |
|--------|--------|---------------|
| Joint-KV + preserve $ N $ | 0.8316 | 0.003070 |
| + value fitting (**QCOC**) | **0.8491** | **0.001491** |
| Key-only + all fixes | 0.8259 | 0.003293 |

**关键发现**：
1. **preserve original $ N $** 和 **cluster-count weighting** 必须同时使用，否则导致注意力计算不一致。
2. **value fitting 贡献最大增益（+1.75 pp）**，显著提升输出保真度。
3. **Joint-KV vs Key-only**：最终准确率无统计差异，但 Joint-KV 在概率和注意力重建误差上更优（55/64 数据集）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Operator Compilation 可行且高效**：将 ICL 中的上下文视为可编译的“预测算子”，而非原始数据，是实现高保真压缩的关键。
2. **QCOC 实现准确率与吞吐双赢**：
   - 准确率接近 full context（平均仅低 0.23 pp）。
   - 支持跨查询缓存复用，显著提升服务吞吐。
3. **value fitting 是精度保障核心**：通过 calibration queries 拟合 Value 向量，有效保留 query-dependent 输出行为。
4. **组件协同作用强**：cluster count、original $ N $、value fitting 缺一不可，共同保证压缩一致性与保真度。

### ⚠️ 局限性
1. **单共享内存难以适应异构数据**：
   - 在 **Covertype** 数据集上表现较差（gap 达 4.27 pp），说明全局压缩在局部结构复杂的数据上受限。
2. **编译开销高**：
   - 一次性编译成本随 $ N^2 $ 增长，在 $ N=8192 $ 时需约 **283 秒**，需足够查询量才能摊销。
3. **依赖上下文交换性（exchangeability）**：假设上下文样本顺序无关，可能不适用于某些序列敏感场景。

### 🔮 未来工作方向
1. **降低编译成本**：探索近似算法或分层编译策略。
2. **多内存选择机制**：构建多个可复用 memory 并根据查询动态选择，兼顾局部性与复用性。
3. **自适应压缩比**：根据数据分布自动调整 $ M $ 或触发 fallback 到 full context。
4. **引入 reconstruction residual 作为诊断信号**：实验发现 attention-output 残差与 accuracy loss 正相关（$ r=0.74 $），可用于检测失效并触发降级。

---

## 总结

> **QCOC 开创性地将 tabular ICL 的压缩问题转化为“算子编译”任务，通过联合聚类、计数加权与 query-calibrated value fitting，实现了高保真、高复用的 KV Cache 压缩。它在准确率上超越现有 retrieval 与压缩方法，在效率上支持高效批处理与低延迟服务，为大规模 tabular ICL 部署提供了实用解决方案。**

</details>

---

### 13. [SCORE: Spectral Correlation Estimation for Multivariate Gaussians](https://arxiv.org/abs/2610.12096)

**Authors**: Christopher B\"ulte, Emil Partow, Astha Gupta, Pascal Esser, Gitta Kutyniok  
**Category**: cs.LG  
**Published**: 2026-10-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.12096v1  

#### Abstract
Neural network-based predictive modeling with high-dimensional structured Gaussian targets requires an efficient and numerically stable, yet expressive approximation of the covariance matrix. We propose SCORE: a scalable framework, combining scoring rule training with an expressive covariance approx...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SCORE: Spectral Correlation Estimation for Multivariate Gaussians

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在高维结构化输出（如时空场、多变量时间序列）的概率预测中，神经网络通常假设预测分布为 **Multivariate Gaussian**。然而，直接建模其协方差矩阵面临两大挑战：
- **计算复杂度高**：全协方差矩阵需要 $O(d^2)$ 参数，通用矩阵运算成本高达 $O(d^3)$。
- **训练不稳定**：常用的 **Negative Log-Likelihood (NLL)** 在协方差接近奇异时梯度无界，导致优化困难。

### 提出的新方法和新思路
本文提出了 **SCORE (Spectral CORrelation Estimation)** 框架，结合以下两个核心创新：
1. **使用 Gaussian Kernel Score 进行训练**：
   - 替代不稳定的 NLL，该得分规则具有闭式解，且梯度有界，保证了数值稳定性。
   - 与 NLL 不同，它即使在退化（singular）协方差下也定义良好。
2. **谱域中的协方差参数化**：
   - 将学习任务分解为两阶段：首先学习边缘分布（marginals），然后在谱空间（如傅里叶基）中学习结构化的相关性矩阵（correlation matrix）。
   - 利用 **unitary transform**（如 DFT）将相关性建模转换到频域，使得一个对角“核”（core）的逆变换能生成一个稠密的循环相关矩阵（circulant correlation matrix）。

### 相比现有方法的优势
| 方面 | SCORE | 传统方法（如 Diag, Chol, LorD） |
|------|-------|-----------------------------|
| **存储成本** | $O(d)$ | Diag: $O(d)$, Chol: $O(d^2)$, LorD: $O(d + dr)$ |
| **计算成本** | $O(d \log d)$ (利用 FFT) | Diag: $O(d)$, Chol: $O(d^3)$, LorD: $O(dr^2 + r^3)$ |
| **表达能力** | 能捕捉长距离依赖（dense dependencies） | Diag: 忽略所有依赖；LorD: 仅能捕捉低秩全局依赖 |
| **训练稳定性** | 高（使用有界梯度的 Kernel Score） | 低（NLL 在边界处梯度爆炸） |

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了三个典型的高维结构化预测任务：
- **Time-series Forecasting**: ETTh1, ETTh2, ETTm1, ETTm2, ILI, Weather, Electricity, Traffic。
- **Monocular Depth Estimation**: NYU Depth v2。
- **Spatial Weather Prediction**: ERA5 表面温度数据。
- **Graph-based Temperature Post-processing**: EUPPBench 数据集（非均匀图结构）。

### 实验设置和评估指标
- **骨干网络 (Backbone)**：为每个任务固定一个强大的骨干网络（如 PatchTST, DepthAnything, U-Cast, GNN），只替换其输出层以公平比较不同协方差参数化方法。
- **评估协议**：报告测试集上的平均分和五次随机种子的标准差。主要使用**真分制评分规则 (proper scoring rules)**：
  - **MSE**: 评估均值预测。
  - **CRPS**: 评估边缘分布拟合。
  - **NLL**: 评估联合分布拟合（对协方差条件数敏感）。
  - **Energy Score (ES)** 和 **Gaussian Kernel Score (KS)**: 评估联合分布。
  - **Variogram Score (VS)**: 特别针对依赖结构，衡量增量预测的准确性。

### 基线方法对比
- **Deterministic (Det)**：仅预测均值。
- **Diagonal (Diag)**：对角协方差，忽略依赖。
- **Low-rank-plus-diagonal (LorD)**：最常见的可扩展替代方案。
- **Sample-based (SB)**：基于噪声注入的生成模型。
- **Full Cholesky (Chol)**：完全表达力的基线（仅在小任务上可行）。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **总体表现**：在 **Table 3** 中，SCORE 在所有方法中取得了最佳的平均排名 **(2.5)**，显著优于最强的基线 LorD-KS **(3.1)**。
- **优势领域**：
  - **VS (Variogram Score)**：SCORE 在此指标上表现最优，表明其在建模**依赖结构**方面最为出色。
  - **CRPS**：得益于第一阶段使用 CRPS 学习边缘分布，SCORE 在边缘拟合上也表现优异。
  - **计算效率**：如 **Figure 4** 所示，SCORE 在建模依赖性的方法中是最快的，实现了最佳的**性能-计算权衡**。
- **消融实验结果**：
  - **损失函数对比 (Table 4)**：在所有协方差参数化中，使用 **Kernel Score** 训练相比 NLL 在绝大多数情况下（167次中有139次以上）都能带来性能提升，证明了其优越性。
  - **两阶段训练**：消融实验证明，第二阶段的相关性学习带来了显著的性能增益，尤其是在多变量评分（如 ES, VS）上，这并非仅仅来自更多的训练轮次。

## 4. 关键结论和发现

### 主要发现
1. **SCORE 是一个高效且稳定的方法**：通过结合 **Kernel Score** 的鲁棒性和 **谱域参数化** 的高效性，SCORE 成功地解决了高维高斯预测中的计算和稳定性难题。
2. **两阶段设计是有效的**：先精确拟合边缘分布，再在标准化后的残差上学习相关性，这种分离策略保证了边缘分布的精确性，并简化了相关性学习。
3. **谱方法适用于具有平稳性假设的任务**：在时间序列和网格化天气数据上，SCORE 表现卓越，因为这些任务的依赖关系往往由距离决定，符合循环协方差的假设。

### 方法的局限性
- **循环协方差结构的假设**：SCORE 隐含地假设了数据具有平稳性（stationarity），即协方差仅依赖于坐标间的相对位置。这在某些应用（如深度估计）中可能不合理，导致性能不佳（如在 NYU 数据集上表现较差）。
- **表达能力受限**：虽然比对角和低秩模型更灵活，但其相关性类仍是一个近似，无法表示任意复杂的依赖模式。

### 未来工作方向
1. **更丰富的谱域参数化**：探索在变换域中使用更复杂的结构，如低秩或三对角核，以增加表达能力同时保持低计算成本。
2. **其他正交变换**：研究小波（wavelets）等其他 unitary transforms，以适应非平稳数据。
3. **扩展到其他分布族**：将 SCORE 框架自然地推广到高斯混合模型（Gaussian mixtures）或多变量 t 分布（multivariate t-distribution），以处理非高斯和重尾数据。
4. **理论改进**：推导更紧致的 PAC 泛化界，并确定两阶段目标之间的最优划分。

</details>

---

### 14. [SACQ: Structured Decoding with Memory-Conditioned Refinement for Long-Horizon Forecasting](https://arxiv.org/abs/2610.11170)

**Authors**: Guo Cheng, Zhengzhuo Xu, Chenchen Jing, Jingyi Hou  
**Category**: cs.LG  
**Published**: 2026-10-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.11170v1  

#### Abstract
Long-term time series forecasting (LTSF) models predominantly employ patch-based encoders terminated by a flatten readout head that maps the entire encoded historical memory to all future steps through a single shared projection. This implicit coupling of future positions obscures position-specific ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SACQ: Structured Decoding with Memory-Conditioned Refinement for Long-Horizon Forecasting 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 **Long-term time series forecasting (LTSF)** 模型普遍采用基于 **patch** 的编码器，并以 **flatten readout** 头结束，即通过一个共享的全连接层将整个历史编码映射到所有未来时间步。这种设计存在两个关键缺陷：

- **Representation Coupling（表示耦合）**：所有未来位置共享相同的读出参数，导致模型无法对不同时间步进行差异化的历史-未来对齐，在输入异常或噪声时表现脆弱。
- **Optimization Coupling（优化耦合）**：在标准 **MSE** 损失下，极端误差会主导梯度更新，尤其在长预测范围和标签噪声场景下，严重影响训练稳定性。

### 提出了什么新方法或新思路
作者提出 **SACQ**（Structured Decoding with Memory-Conditioned Refinement），一种可插拔的结构化解码头，**不改变原有 patch 编码器**，仅替换 flatten readout，其核心思想是将长周期预测重构为两阶段解码过程：

1. **Coarse Scaffold Construction（粗略骨架构建）**  
   通过轻量级投影生成初步的未来 patch 预测网格，保留全局结构。

2. **Memory-Conditioned Refinement（记忆条件化精炼）**  
   使用 **cross-attention** 机制，让每个未来位置从历史 memory 中查询相关信息，生成位置特定的修正残差。

3. **Gated Residual Fusion（门控残差融合）**  
   通过一个可学习的 **per-patch gate** 将粗略预测与注意力生成的修正项融合，实现自适应调整。

此外，为增强鲁棒性，提出 **batch-adaptive scaled log-cosh loss**：
- 自动根据当前 batch 的中位数残差 $ m $ 设置尺度参数 $ \beta = k \cdot m $（默认 $ k=0.5 $）
- 小误差区域行为类似 **MSE**，大误差区域梯度被抑制，提升对异常值的鲁棒性。

### 相比现有方法的优势
- **更强的表达能力**：显式建模历史-未来的 token 级对齐，避免静态权重压缩。
- **更高的鲁棒性**：在输入污染、标签噪声等压力测试下显著优于 flatten readout。
- **即插即用**：兼容多种 backbone（如 PatchTST、DLinear、patch-Mamba），无需修改编码器。
- **训练更稳定**：自适应损失函数缓解了长周期预测中的梯度不平衡问题。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验涵盖多个标准多变量时间序列预测数据集：
- **ETT系列**：ETTh1, ETTh2, ETTm1, ETTm2（电力变压器温度）
- **Electricity (ECL)**：321个电力消耗序列
- **Traffic**：862个城市交通流量序列
- **Weather**：气象观测数据（21维）

### 实验设置和评估指标
- **预测长度**：$ T_f \in \{96, 192, 336, 720\} $
- **评估指标**：**Test MSE** 和 **MAE**（标准化后）
- **训练配置**：
  - 优化器：Adam
  - 学习率调度：前3轮恒定，之后每轮×0.9
  - 批大小：ETT/Weather 为 128，Traffic/ECL 为 8
  - SACQ 默认使用 $ L=2 $ 层 cross-self attention 和 **scaled log-cosh** 损失

### 基线方法对比
- **主流模型**：PatchTST, DLinear, FEDformer, TimesNet, LSINet, FiLM, CI-TSMixer
- **readout 替换实验**：在 **patch-Linear** 和 **patch-Mamba** 上比较 flatten vs. SACQ
- **鲁棒性基准**：使用 **TSRBench** 协议测试推理时输入污染（spike + level-shift）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **Table I** 中，SACQ 在 24 个任务中取得 **18 个最佳 MSE 或 MAE**，平均排名 **MSE: 1.55**, **MAE: 1.16**，显著领先其他方法。

| 数据集 | $T_f$ | SACQ (MSE) | 最佳基线 (MSE) |
|--------|-------|------------|----------------|
| ETTh1 | 720 | **0.429** | 0.441 (LSINet) |
| ETTh2 | 720 | **0.376** | 0.382 (LSINet) |
| ECL | 720 | **0.198** | 0.192 (但 MAE 更优) |
| Weather | 720 | **0.310** | 0.304 (LSINet) |

> 注：尽管 LSINet 在部分任务 MSE 更低，但 SACQ 综合表现更稳定且鲁棒性更强。

### 与基线方法的对比结果
- **通用性验证（Table II）**：在 **patch-Linear** 和 **patch-Mamba** 上替换 readout 后，SACQ 均优于原始 flatten 设计，证明其有效性不依赖于特定 backbone。
- **鲁棒性测试（Table VI）**：在 **TSRBench** 输入污染协议下，随着污染严重程度增加（Severity 0→5），SACQ 性能下降最慢，尤其在 Severity ≥3 时优势明显。
- **标签噪声测试（Fig. 3）**：在训练集中引入目标 spike 时，SACQ + scaled log-cosh 表现最稳健，MSE 增长斜率最小。

### 消融实验结果
#### 组件消融（Table III）
移除任一组件均导致性能下降：
- **-CA（无 cross-attention）**：MSE 显著上升（如 ECL 从 0.198 → 0.206），说明 memory-conditioned refinement 至关重要。
- **-G（固定 gate α=1）**：性能下降，表明 **learnable gating** 能有效平衡粗略结构与精细修正。
- **Coarse only（仅粗略预测）**：误差最大，验证了精炼阶段的必要性。

#### 损失函数消融（Table IV）
- **SACQ + scaled log-cosh** 在多数情况下优于 SACQ + MSE 和 flatten + scaled log-cosh。
- 证明 **structured decoding** 与 **robust loss** 是互补而非替代关系。

#### SAR（因果自注意力）消融（Fig. 4）
引入 **causal self-attention**（SAR）后，在尾部标签污染训练下测试 MSE 更低，尤其在 $ T_f=720 $ 时优势明显，说明“由近及远”的归纳偏置有助于减少未来污染标签的反向影响。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **flatten readout 是 LTSF 的瓶颈**：其隐式耦合限制了模型在长周期、噪声环境下的表现。
2. **SACQ 是有效的即插即用改进方案**：通过结构化解码 + 门控融合，显著提升预测精度与鲁棒性。
3. **batch-adaptive scaled log-cosh 提升训练稳定性**：自动适配残差尺度，无需手动调参。
4. **组件协同作用明显**：coarse scaffold、cross-attention、gating、robust loss 共同构成完整解决方案。

### 方法的局限性
- **推理延迟增加**：由于引入额外 attention 层，SACQ 带来约 **2.6ms** 的额外延迟（在 d_model=256 时），虽为常数倍增长，但在超低延迟场景可能受限。
- **参数量略有上升**：增加约 2.7M 参数（主要来自 decoder stack）。
- 当前仍为非自回归一次性输出，未探索 step-by-step 生成模式。

### 未来工作方向
- 设计 **更低延迟的 SACQ 变体**（如稀疏 attention、蒸馏策略）
- 在更严重的分布偏移（severe distribution shifts）和不规则预测长度下进一步验证
- 探索与 **lightweight** 或 **retrieval-augmented backbones** 的深度集成

--- 

> ✅ **总结一句话**：  
> SACQ 通过将 flatten readout 改造为 **两阶段结构化解码 + 门控融合 + 自适应鲁棒损失**，在不改动编码器的前提下，实现了更准确、更鲁棒的长周期时间序列预测，且具备良好的通用性和即插即用特性。

</details>

---

### 15. [Plan-and-Patch: Diffusion Language Models for Agentic Planning](https://arxiv.org/abs/2610.10786)

**Authors**: Syamantak Kumar, Jiang Guo, Hassan Hamad, Hideo Kobayashi, Yi Xiang, Yezhou Yang, Yanjun Qi, Daniele Bonadiman, Jiarong Jiang  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.10786v1  

#### Abstract
Planning is increasingly important for long-horizon agents, where successful execution requires coordinating subgoals, tool use, and intermediate outcomes over many steps. Yet assumptions made during planning may be invalidated by the environment, tools may return unexpected results, or actions may ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PLAN-AND-PATCH: Diffusion Language Models for Agentic Planning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在长周期任务（long-horizon tasks）中，智能体需要生成多步计划并执行。然而，在执行过程中，环境反馈可能导致原计划失效（如动作失败、工具返回意外结果、假设被推翻等）。传统方法通常采用**全局重规划**（global regeneration），即丢弃整个计划重新生成，这不仅效率低，还可能破坏原本正确的部分。

因此，如何高效地进行**局部修复**（plan repair）——仅修改受影响区域而保留其余结构——成为一个关键挑战。

---

### **提出的新方法与创新思路**
作者提出了 **PLAN-AND-PATCH** 框架，其核心思想是将**计划生成**（plan generation）与**计划修复**（plan repair）统一在一个基于 **Diffusion Language Model (dLLM)** 的模型中，并将修复建模为 **contextual infilling** 任务：

- **Plan Generation**：使用 dLLM 通过 **parallel unmasking** 并行生成完整的结构化计划。
- **Plan Repair**：当执行失败时，由 executor 识别出需修复的 region（`R`），然后 dLLM 在固定前后缀（prefix/suffix）的前提下，对选中的 masked 区域进行填充（infilling）。

该框架的关键创新在于：
- 利用 dLLM 天然支持双向上下文建模的能力，实现高效的局部修复。
- 使用单一 dLLM 同时完成生成与修复，避免了模块割裂。
- 引入结构化的 plan 表示（subgoals + executable steps），便于定位和替换可寻址（addressable）的片段。

---

### **相比现有方法的优势**
| 维度 | 传统 AR 方法 | PLAN-AND-PATCH (dLLM) |
|------|-------------|------------------------|
| **修复方式** | 全局重生成 或 需特殊训练的 FIM | 原生支持 infilling，无需额外训练目标 |
| **生成顺序** | 自回归（left-to-right），串行依赖强 | 并行去噪，解码更灵活 |
| **延迟** | 高（尤其长序列） | 显著降低生成延迟（↓39–46%） |
| **修复成功率** | 较低（尤其无任务微调时） | 在 Natural Plan 上修复成功率接近翻倍 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 任务类型 | 特点 |
|-------|--------|------|
| **ALFWorld** | 家庭导航与操作任务（如“把肥皂放进柜子”） | 结合视觉模拟与文本指令，测试导航+操作能力 |
| **TextCraft** | 依赖驱动的合成任务（如“制作粉色混凝土粉”） | 要求正确顺序获取材料并合成，强调依赖推理 |
| **Natural Plan** | 旅行安排与日程调度（trip & meeting planning） | 无真实执行器，使用程序化约束验证器评估 |
| **ScienceWorld**（补充） | 科学实验推理任务 | 更长的计划长度，用于分析扩展性 |

---

### **实验设置与评估指标**

#### **模型配置**
- **主干模型**：
  - **DIFF-PLAN**: DreamReasoner-8B（基于 Qwen3-8B 的 diffusion 模型）
  - **AR-PLAN**: Qwen3-8B（标准自回归模型）
- **Executor**: 固定为未微调的 Qwen3-8B，负责执行计划步骤并检测失败。
- **共享组件**：相同的 structured plan 表示、action grammar、evaluation protocol。

#### **评估协议**
- **Plan Generation**：终端任务成功率（terminal task success rate）
- **Plan Repair**：
  - **Reference-assisted repair**：使用参考计划辅助选择 repair region 和 feedback
  - **History-based repair**：仅依赖 execution history 进行 region selection
- 所有修复尝试后均从初始状态重新执行（restart from initial state）

#### **关键指标**
- **Success Rate (%)**：最终达成任务目标的比例
- **Latency (seconds)**：平均每次 plan generation / repair 的推理时间
- **Parser Acceptance**：输出是否符合格式规范
- **Preservation Rate**：修复前后已完成步骤是否保持不变

---

### **基线方法对比**
- **AR-PLAN**：标准 autoregressive 模型 + FIM（Fill-in-the-Middle）支持
- **ReAct**：直接行动，无显式 planning 阶段
- **Human-written reference plans**：作为理想上限参考

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### ✅ **任务成功率（Table 1 & A.3）**

| Environment | Model | Generation | Ref-Assist Repair | End-to-End Success |
|------------|-------|-----------|--------------------|---------------------|
| ALFWorld | DIFF-PLAN | 44.0% | 28.4% | 59.9% |
| ALFWorld | AR-PLAN | 45.5% | 27.7% | 60.6% |
| TextCraft | DIFF-PLAN | 63.0% | 15.5% | 68.7% |
| TextCraft | AR-PLAN | 66.0% | 18.3% | 72.2% |

> ➤ 两者在生成任务上表现相近，AR 略优；但在修复阶段，diffusion 展现出更强潜力。

---

#### ⚡ **生成延迟显著下降（Figure 5）**

| Environment | DIFF-PLAN Latency | AR-PLAN Latency | 减少幅度 |
|------------|------------------|------------------|---------|
| ALFWorld | 14.3s | 23.3s | ↓38.6% |
| TextCraft | 21.7s | 40.2s | ↓46.0% |

> ➤ diffusion 模型通过并行解码大幅缩短生成时间。

---

#### 🧩 **无任务微调下的修复优势（Natural Plan, Table 2）**

| Task | Model | Validator Acceptance |
|------|-------|------------------------|
| Generation | AR-PLAN | 48.6% |
| Generation | DIFF-PLAN | 33.0% |
| Repair | AR-PLAN | 27.0% |
| Repair | DIFF-PLAN | **53.7%** ✅ |

> ➤ 在未进行任务特定训练的情况下，diffusion 的修复成功率几乎是 AR 的两倍！

进一步分析显示，这种优势集中在 **trip planning** 子任务（93.3% vs 35.7%），说明 diffusion 对复杂依赖结构有更好的建模能力。

---

#### 🔍 **消融实验与敏感性分析**

##### （1）**Joint Training 的必要性**
- 若先训 generation 再 finetune repair，会导致 generation 性能严重退化（invalid formatting）
- **Joint training** 可同时保留两种能力

##### （2）**Plan Representation 影响**
- 改用 prose-style plan（带编号段落而非 delimiter）：
  - **generation 成功率提升**（ALFWorld ↑20 pts）
  - **但 repair 成功率下降**（DIFF-PLAN 从 28.4% → 21.3%）
- ➤ 显示 **explicit delimiters 有助于精准定位修复区域**

##### （3）**Region Selection 至关重要**
- 使用 history-based selection 替代 reference-assisted selection：
  - ALFWorld 上 repair success ↓12 pts（diffusion）
  - TextCraft 上反而略有上升（+2.1 pts）
- ➤ 说明当前瓶颈已从“能否修复”转向“能否准确定位错误”

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Diffusion 模型天然适合 plan repair**  
   其 infilling 能力使得局部修复更加高效且稳定，尤其在缺乏任务微调时优势明显。

2. ✅ **并行生成显著降低延迟**  
   相比 AR 模型，diffusion 在 plan generation 阶段提速 **39–46%**，有利于实时交互场景。

3. ✅ **Joint training 是关键设计**  
   单一 checkpoint 同时支持 generation 与 repair，且不会相互干扰。

4. ❗ **修复效果高度依赖 region selection**  
   当前 executor 在没有 reference 的情况下难以准确判断应修复哪一部分，成为系统瓶颈。

5. 🔄 **Plan representation 权衡 trade-off**  
   - Delimited format 更利于机器解析与修复
   - Prose format 更易读，生成成功率更高，但不利于自动化编辑

---

### **局限性**
- **修复范围受限于 addressable unit**：目前只能修复预定义的 step/subgoal 级别，无法处理跨层级语义变更。
- **依赖高质量 failure detector**：executor 必须能精确定位故障源，否则修复无效。
- **长序列生成仍有挑战**：在 ScienceWorld 中，diffusion 的 parser acceptance 随长度急剧下降（>28 steps 时仅 ~7%），存在重复动作和未闭合结构问题。
- **compute cost 较高**：尽管延迟低，但 diffusion 通常需要多次 denoising step，总计算量仍高于 AR。

---

### **未来工作方向**
1. **联合学习 region selection 与 infilling**  
   训练一个端到端的 failure-aware repair agent，自动决定“修哪里”和“怎么修”。

2. **动态调整 repair scope 与 decoding effort**  
   根据不确定性估计分配更多 denoising step 给高风险区域。

3. **扩展至更长 horizon 与开放世界任务**  
   探索 hierarchical diffusion planning，结合 symbolic grounding 提升一致性。

4. **引入 retrieval-augmented repair**  
   利用 past successful repairs 构建 memory bank，指导当前 infilling。

---

> **一句话总结**：  
> **PLAN-AND-PATCH 展示了 diffusion language models 在 agentic planning 中的独特优势——既能快速生成完整计划，又能高效修复局部错误，为构建鲁棒、低延迟的长周期智能体提供了新范式。**

</details>

---

### 16. [When Lower Reconstruction Loss Hurts: Distributionally Robust Refinement for Low-Bit LLM Quantization](https://arxiv.org/abs/2610.11226)

**Authors**: Yanlong Zhao, Xiaoyuan Cheng, Huihang Liu, Baihua He, Xinyu Zhang, Harrison Bo Hua Zhu, Wenlong Chen, Li Zeng, Zhuo Sun  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.11226v1  

#### Abstract
Weight-only post-training quantization (PTQ) relies heavily on reconstruction loss minimization to preserve model quality at low precision. We show that the weights favored by minimizing this loss need not yield better model performance on new tasks. In fact, we find that lower reconstruction loss c...

---

### 17. [SynCo: Data Synthesis Co-Training for Self-Evolving LLMs via Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2610.11345)

**Authors**: Wei Yang, Shawn Li, Yuehan Qin, Yawei Wang, Mingxi Wang, Shixuan Li, Tiankai Yang, Jiate Li, Jesse Thomason, Xuezhe Ma, Yue Zhao  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.11345v1  

#### Abstract
Self-evolving LLM agents promise to improve autonomously through continual interaction and learning, reducing their dependence on manually curated supervision. Realizing this promise requires not only updating the agent, but also evolving its training experience as its capabilities change. However, ...

---

### 18. [RaReCache: Bridging the Gap in Cross-Model KV Cache Reuse via Rank disagreement-based Selective Recomputation](https://arxiv.org/abs/2610.11358)

**Authors**: Sreetama Sarkar, Saptarshi Mitra, Sitao Huang, Souvik Kundu, Peter A. Beerel  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.11358v1  

#### Abstract
Cross-model KV-cache reuse remains a key challenge in modern LLM serving. Coding agents and multi-model systems increasingly route a shared context across models: a user may switch models mid-session, or a cascade may escalate a difficult query. Because KV caches contain model-specific representatio...

---

### 19. [SpikeSSL: A Universal Spike Inference Framework with Dynamics-Informed State-Space Layers](https://arxiv.org/abs/2610.11456)

**Authors**: Chenghao Yue, Siming Xing, Shuran Liu, Angran Li, Yuanlong Zhang  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.11456v1  

#### Abstract
Two-photon calcium imaging is a standard tool for recording large neural populations in vivo, yet inferring spikes accurately across the growing diversity of calcium indicators remains an open problem. Existing supervised methods achieve reasonable in-domain accuracy but generalize poorly to unseen ...

---

### 20. [From Chain-of-Thought to Loops: Non-Autoregressive Latent Reasoning via Looped Transformers](https://arxiv.org/abs/2610.11472)

**Authors**: Gerard Grau Garc\'ia, Arnau Padr\'es Masdemont, Niccol\`o Grillo, Jordi Ros-Giralt, Arash Behboodi, Victor Conchello Vendrell  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.11472v1  

#### Abstract
Chain-of-thought (CoT) reasoning often improves language-model performance by giving models additional computation before answering. However, explicit CoT expresses this computation as a sequence of autoregressively generated tokens. Latent reasoning replaces these tokens with compact continuous sta...

---

### 21. [Where Draft Trees Lose Target Mass: Exit-Guided Speculative Decoding](https://arxiv.org/abs/2610.11750)

**Authors**: Shijing Hu, Xuancheng Ren, Zhihui Lu, Pan Zhou  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.11750v1  

#### Abstract
Tree-based speculative decoding verifies multiple draft continuations in one target-model pass, but finite trees built from draft scores face a fundamental draft-target mismatch. We ask whether better exact verification can increase acceptance on a fixed tree and how target feedback can improve the ...

---

### 22. [BioBigBird: A Sparse Attention Model for Long-Range Dependency Processing in Biomedical Text](https://arxiv.org/abs/2610.11430)

**Authors**: Roshan Balaji, Pavan Kumar S, Vasudev Gupta, Sreejith N, Keerthana Sridhar, Nirav Bhatt  
**Category**: cs.CL  
**Published**: 2026-10-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.11430v1  

#### Abstract
While domain-specific Large Language Models (LLMs) have encoded vast biomedical knowledge, their limited context windows often hinder a deep understanding of nuanced relationships within and across texts. To address this limitation, we introduce BioBigBird, a bidirectional language model pre-trained...

---

### 23. [Sample-Efficiency of Kolmogorov-Arnold Networks](https://arxiv.org/abs/2610.10627)

**Authors**: Kevin Riehl, Shaimaa K. El-Baklish, Fan Wu, Anastasios Kouvelas  
**Category**: cs.LG  
**Published**: 2026-10-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.10627v1  

#### Abstract
Deep reinforcement learning has achieved substantial performance gains over classical control approaches. Yet, a central challenge to learning in real-world applications is acquiring costly samples. Kolmogorov-Arnold Networks are a recently proposed architecture that can learn physical relationships...

---

### 24. [Synthesis Through Simulation: Generating Coherent Enterprise Data via Scalable Agent-System Interaction](https://arxiv.org/abs/2610.10549)

**Authors**: Yipeng Li, Ashutosh Hathidara, Jane Lo, Harshavardhan Abichandani, Gunraj Singh, Atin Ghosh  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.10549v1  

#### Abstract
Tool-calling agents have become central to enterprise AI, yet training and evaluating them at scale remains severely constrained due to business and legal restrictions on enterprise systems, data, and database schemas. Tabular data synthesis offers a natural alternative, but its effectiveness is fun...

---

### 25. [Cognition-Oriented Emotion Tracing from Causes to Consequences in Real-World Social Scenes](https://arxiv.org/abs/2610.11410)

**Authors**: Hao Li, Jinye Zhang, Bobo Li, Mong-Li Lee, Wynne Hsu, Zheng Wang, Hao Fei, Min Zhang  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.11410v1  

#### Abstract
Affective computing has progressed from categorical emotion recognition to open-ended affective analysis with large multimodal models. Yet affective science describes emotion as an unfolding process shaped by appraisal, regulation, and social interpretation, which remains underexplored computational...

---

### 26. [A 3D Characterization Framework for Intelligent Sequential Decision Making](https://arxiv.org/abs/2610.11696)

**Authors**: Sadig Gojayev, Carolina Fortuna  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.11696v1  

#### Abstract
Puzzles are widely used to evaluate the reasoning capabilities of artificial intelligence (AI) systems for sequential decision making, yet approaches originating from different paradigms are rarely compared under unified conditions. To address this gap, we introduce a three-dimensional characterizat...

---

### 27. [Probability-Signature Dynamics: Unpacking Modular Addition Learning Within Two-Layer Networks](https://arxiv.org/abs/2610.11833)

**Authors**: Yunji Wang, Junjie Yao, Linyu Liu, Pinyan Lu, Zhi-Qin John Xu  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.11833v1  

#### Abstract
Neural networks trained on modular addition tasks often develop Fourier-structured representations that support exact generalization. While prior work has identified these Fourier circuits, the mechanism by which gradient-based training selects them from the data distribution remains unclear. We add...

---

### 28. [Universal Textual Teaching for LLMs](https://arxiv.org/abs/2610.12114)

**Authors**: Zhanyi Lu, Huan Wang  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.12114v1  

#### Abstract
Knowledge distillation (KD) transfers knowledge from stronger Teacher models to weaker Student models, but most methods require training the Student parameters, thereby binding the distilled knowledge to a specific architecture and checkpoint. This implicit representation is difficult to interpret o...

---

### 29. [Smoothing the Top-k Exposure Boundary for Sparse Mixture-of-Experts](https://arxiv.org/abs/2610.11575)

**Authors**: Yunkai Chai, Tong Zhu, Xiaoye Qu, Xuyang Hu, Guanjie Chen, Qipeng Guo, Yu Cheng  
**Category**: cs.CL  
**Published**: 2026-10-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.11575v1  

#### Abstract
Sparse Mixture-of-Experts models scale parameter capacity efficiently while maintaining a fixed compute budget per token. However, traditional training paradigms enforce a static choice of top-$k$ experts, which converts a continuous routing distribution into a rigid step function. This constraint i...

---

### 30. [RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing](https://arxiv.org/abs/2610.11775)

**Authors**: Ilya Lasy, Nora Yinuo Cai, Kola Ayonrinde  
**Category**: cs.AI  
**Published**: 2026-10-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.11775v1  

#### Abstract
Sparse Mixture of Experts (MoE) models scale more efficiently than dense models by routing tokens to modular expert networks that are only active for processing a fraction of tokens. A leading hypothesis for the performance of MoE models is that each expert specialises in a single, coherent domain. ...

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

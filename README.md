# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-12 09:53:04 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)

**Authors**: Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Pengyu Zhao  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.13392v1  

#### Abstract
Ultra-long-context capability is becoming indispensable for frontier LLMs: agentic workflows, repository-scale code reasoning, and persistent memory all require the model to jointly attend over hundreds of thousands to millions of tokens, yet the quadratic cost of softmax attention makes this untena...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MiniMax Sparse Attention 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前前沿大语言模型（LLMs）对**超长上下文能力**的需求日益增长，例如智能体工作流（agentic workflows）、代码库级推理和持久记忆等任务需要模型同时关注数十万到数百万个token。然而，标准的Softmax Attention具有**二次计算复杂度**（$O(N^2)$），在实际部署中难以扩展。

为解决这一瓶颈，本文提出了一种高效稀疏注意力机制，在保持模型性能的同时显著降低计算开销。

---

### 提出的新方法：MiniMax Sparse Attention (MSA)

MSA 是一种基于 **Grouped Query Attention (GQA)** 的块级稀疏注意力机制，其核心设计遵循“奥卡姆剃刀”原则——仅保留最必要的组件以实现简洁性和可扩展性。

#### 主要创新点：
- **双分支架构（Two-Branch Architecture）**
  - **Index Branch（索引分支）**：轻量级模块，对每个 GQA 组独立评分并选择 Top-k 的 KV 块。
    - 仅增加两个投影矩阵（$W_{idx}, W_{kdx}$）。
    - 使用 `max-pooling` 对块内得分聚合，避免逐token计算。
    - 引入 **KL Alignment Loss** 进行训练，使索引分布与主干注意力模式对齐。
  - **Main Branch（主分支）**：在选定的块上执行精确的 block-sparse attention。

- **Group-Specific Block Selection**
  - 每个 GQA 组独立选择自己的 Top-k KV 块，而非全局共享。
  - 支持更细粒度的内容感知检索，提升语义相关性。

- **强制本地块保留（Local Block Inclusion）**
  - 当前查询所在的“本地块”始终被选中，确保局部上下文不丢失，提高训练稳定性。

- **端到端系统协同设计**
  - 算法与 GPU 内核协同优化，将理论稀疏性转化为实际速度提升。
  - 包括：
    - **Exp-free Top-k Kernel**：跳过 softmax 中的 exp/max/sum 步骤，直接比较原始分数。
    - **KV-outer Sparse Attention**：按 KV 块外循环组织计算，提升 Tensor Core 利用率。
    - **预调度分块 + 两阶段归约（Pre-scheduled Chunking & Two-phase Combine）**：处理热门 KV 块导致的负载不均衡问题。

---

### 相比现有方法的优势

| 特性 | MSA | 其他稀疏方法（如 NSA, MoBA, DSA） |
|------|-----|-------------------------------|
| 架构兼容性 | 基于 GQA，易于集成主流模型 | 多基于 MQA/MHA 或需特殊结构 |
| 选择粒度 | 每 GQA 组独立选择 Top-k 块 | 多为全局共享或固定模式 |
| 训练方式 | 原生训练 + 可从 GQA Checkpoint 转换 | 部分依赖预训练后剪枝 |
| 实际加速 | 显著 wall-clock 加速（见下文） | 往往理论 FLOPs 下降但实际加速有限 |
| 多模态支持 | 已验证支持图文视频统一建模 | 多集中在文本场景 |

> ✅ **优势总结**：MSA 在保持高性能的前提下，实现了极简设计、高硬件效率和良好的迁移能力。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **预训练语料**：混合文本与图像/视频数据，总预算 **3T tokens**。
- **评估基准**（覆盖多个维度）：

| 类别 | 基准名称 |
|------|--------|
| 通用问答 | MMLU, MMLU-Pro, BBH, GPQA Hard, ARC Challenge, TriviaQA, WinoGrande |
| 数学 | GSM8K, MGSM, MathVista, OlymMATH |
| 编程 | HumanEval, EvalPlus, BigCodeBench, MultiPL-E MBPP |
| 图像理解 | AI2D, ChartQA, MMMU, OCRBench v2, CharXiv, VisualWebBench, CVBench |
| 视频理解 | EgoSchema, LongVideoBench, MLVU, MMVU, VideoMME, TemporalBench |
| 长上下文 | RULER, HELMET |
| Agent任务困惑度 | x2-bench, TheAgentCompany, Humanity's Last Exam (HLE), SWE-bench |

---

### 实验设置

- **模型结构**：
  - MoE 架构，共 41 层，总计 ~109B 参数，激活参数 ~6B/token。
  - 使用 GQA：64 query heads, 4 KV heads, head dim = 128。
  - 每层 MoE 含 128 个路由专家 + 1 共享专家，top-4 路由。
  - 词表大小：200K，hidden size $d_{model}=3072$。

- **稀疏配置**：
  - Block size $B_k = 128$
  - 每 query 每 GQA 组保留 $k = 16$ 个 KV 块 → 总计最多 $16 \times 128 = 2048$ 个 token 被关注

- **训练策略**：
  - **MSA-PT**：从零开始训练，先进行 40B token 的 indexer warmup（全注意力），再切换至稀疏训练。
  - **MSA-CPT**：从已有的 Full-Attention GQA Checkpoint（训练 2.6T token）出发，替换为 MSA 后继续训练 400B token（含 40B warmup）。

- **评估指标**：
  - 准确率（Accuracy）
  - 困惑度（PPL ↓）
  - 推理速度（Prefill & Decode Wall-clock Time）
  - Attention FLOPs 消耗
  - Block Recall / Score Recall（衡量索引质量）

- **基线对比方法**：
  - **Full-Attention GQA**：完整注意力基线
  - （隐含对比）其他稀疏注意力方法（如 NSA, MoBA, DSA 等）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 结果 |
|------|------|
| **Attention FLOPs Reduction @ 1M context** | **28.4×** 低于 GQA |
| **Prefill Speedup @ 1M context (H800)** | **14.2×** |
| **Decode Speedup @ 1M context (H800)** | **7.6×** |
| **Top-k Selection Latency (N=512K, B=4096, k=16)** | **7.88ms**（比 PyTorch 快 4.3×） |

> 🔥 **说明**：这些是**端到端真实运行时加速**，而不仅仅是理论 FLOPs 下降。

---

### 与基线方法的对比结果（Table 2 & 3）

#### 整体性能几乎持平（Table 2）
- MSA-PT 和 MSA-CPT 在绝大多数任务上与 Full-Attention GQA 基线表现相当。
- **MSA-PT**（从头训练）在数学、图像、视频和长上下文检索任务上略优。
- **MSA-CPT**（微调转换）在文本、编程和 PPL 上更稳定，适合已有模型升级。

| 类别 | 示例任务 | Full | MSA-PT | MSA-CPT |
|------|---------|------|--------|---------|
| 数学 | GSM8K | 76.2 | **77.7** | 73.7 |
| 图像 | ChartQA | 75.0 | **75.4** | 71.4 |
| 视频 | EgoSchema | 29.6 | **37.6** | 25.8 |
| 长上下文 | RULER-32K | 75.0 | **77.5** | 75.7 |
| PPL | TAU2 | 1.155 | **1.148** | 1.150 |

> ✅ 表明 MSA **未造成明显性能损失**，甚至某些方面更强。

#### 长上下文扩展能力（Table 3）
- 在 MSA-CPT 上追加 ~140B tokens 的长上下文训练后，在 HELMET 和 RULER 上仍接近 Full-Attention 基线。
- 尤其在 **RULER MK/MQ/MV 子任务**上反超 +2.24 分。
- 表明即使只关注 2048 个 token，也能有效捕捉长距离依赖。

---

### 消融实验结果（Appendix）

#### B.2 Gradient Detach 的重要性
- 若不使用 `stop_gradient`，KL loss 的梯度会回传至主干网络，导致：
  - 梯度爆炸（gradient norm spike）
  - LM loss 发散
  - 下游任务性能退化
- 使用 detach 后训练完全稳定。

#### B.4 Indexer Warmup 的作用
- 不使用 warmup 会导致早期注意力分布剧烈变化，索引器无法学习。
- 使用 warmup（先全注意力训练索引器）后：
  - 短上下文任务 +2~3 pts
  - 长上下文检索显著改善

#### C.1 Block Size 影响
- 测试不同 block size（32, 64, 128）但保持总 token 数相同：
  - PPL 几乎无差异
  - RULER 得分略有波动但无系统下降
- 结论：**可用更大 block size 提升 kernel 效率而不损性能**

#### C.2 Forced Sink & Local Window
- 移除硬编码的“首块强制保留”和“局部窗口”后：
  - 模型仍能自然学到 attention sink 和局部偏好
  - 性能无显著差异
- 最终设计仅强制保留“自环块”（即当前 token 所在块）

#### C.3 Index Branch Value Head
- 移除 index branch 的 value head 后：
  - 性能基本一致，部分任务互有胜负
- 结论：该 head 并非必需，最终移除以简化结构

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **稀疏注意力可以做到“近零损失”且高效**
   - MSA 在 109B MoE 模型上实现了与 Full GQA 相当的整体性能，但在 1M 上下文长度下带来 **28.4× FLOPs 下降** 和 **14.2× 预填充加速**。

2. ✅ **算法-硬件协同设计至关重要**
   - 单纯稀疏化不足以获得实际加速；必须配合定制化 kernel（如 exp-free Top-k、KV-outer MMA 填充）才能释放潜力。

3. ✅ **每组独立选择（Per-GQA-Group Selection）优于全局共享**
   - 不同 GQA 组倾向于关注不同的远程模式（stripes），表明 group-specific selection 更具表达力。

4. ✅ **Attention Sink 是自然涌现现象**
   - 即便未显式强制，模型也会自发地将大量注意力分配给序列起始 token，说明 sink 是一种鲁棒的学习行为。

5. ✅ **可平滑迁移已有模型**
   - MSA-CPT 方案证明可以从成熟的 Full-Attention Checkpoint 成功转换，适用于生产环境升级。

---

### 方法的局限性

1. ⚠️ **极端长上下文下的检索能力仍有差距**
   - 如 Table 3 所示，在 RULER Rerank/RAG 子任务上有 -2.1 分差距。
   - 可能因固定 budget（仅 2048 tokens）限制了全局信息获取。

2. ⚠️ **依赖高质量的 warmup 初始化**
   - 若 warmup 不充分，indexer 可能陷入次优状态，影响后续训练。

3. ⚠️ **目前仅验证于 MoE 架构**
   - 是否同样适用于 Dense-only 模型尚待验证。

---

### 未来工作方向

1. 🔄 **缩小长上下文检索差距**
   - 探索动态调整 selection budget
   - 引入更丰富的 indexer scoring function（如 hierarchical 或 cross-layer coordination）

2. 🤖 **扩展至 RL 后训练与 Agent 部署**
   - 在强化学习和真实 agent 场景中进一步验证 MSA 的有效性。

3. 🔁 **探索 dense-to-sparse 的无缝切换机制**
   - 实现训练时密集、推理时稀疏的灵活适配（类似 InfLLM-V2 的目标）。

4. 💾 **进一步优化内存访问与缓存策略**
   - 特别是在 KV Cache 管理方面结合 H2O、SnapKV 等思想。

---

> 🔗 **开源信息**：
> - 推理内核已开源：[https://github.com/MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA)
> - 支持 MSA 的多模态模型已发布：[https://huggingface.co/MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)

</details>

---

### 2. [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/abs/2606.12556)

**Authors**: Hakbeom Jang, Younghoon Min, Sunwoong Kim, Taeyoung Ahn, Hanyee Kim, Youngpyo Joo, Hoshik Kim, Jongryool Kim  
**Category**: cs.DC  
**Published**: 2026-06-12  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.12556v1  

#### Abstract
The rapid shift toward agentic and long-context workloads in Large Language Models (LLMs) is pushing the industry beyond the capacity of individual servers toward disaggregated shared storage to handle TB-scale context states. This movement has led to the emergence of specialized shared context laye...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories

---

## 1. 论文的主要贡献和创新点

### 解决的问题
随着大模型（LLMs）在**agentic AI** 和 **long-context 推理任务** 中的应用日益广泛，推理系统面临两大瓶颈：
- **内存容量不足**：模型权重（如 OPT-175B 需要 325GB FP16）和 KV Cache 的累积状态远超单台服务器的 GPU/HBM 容量。
- **共享上下文管理低效**：传统基于本地 SSD 或 DPU-JBOF 的 disaggregated 存储架构依赖复杂的软件栈，存在高延迟、资源碎片化、成本高等问题。

现有方案（如 DPU-based JBOF + NVMe-oF）虽能提供大容量存储，但其访问路径长、协议开销大，难以满足实时推理对低延迟和高带宽的需求。

---

### 提出的新方法与思路
作者提出 **ITME (Inference Tiered Memory Expansion)**，一种基于 **CXL-Hybrid Memory** 的远程内存扩展架构，核心思想是：
> 将以 SSD 为后端的大容量存储“虚拟化”为一个可字节寻址（byte-addressable）、高性能的远程内存池，并通过硬件级预取机制隐藏访问延迟。

#### 核心创新点：
1. **CXL-Hybrid Memory 架构设计**
   - 结合 **NAND Flash**（低成本大容量）与 **DRAM Cache**（高速缓存），通过 CXL 协议暴露为统一的 NUMA 节点。
   - 支持 **cache-coherent** 和直接 load/store 访问，简化软件栈。
   - 内部集成 **硬件级预取器（HW Prefetcher）**，自动将 SSD 数据预加载到 DRAM 缓存中。

2. **多级 DMA 预取流水线（Pipelined Multi-tier DMA-based Prefetching）**
   - 利用 LLM 推理中 **模型权重** 和 **prefix KV Cache** 的强可预测性（layer-wise、append-write），实现跨层级（remote → host → GPU）的数据预取。
   - 使用 **RDMA + DMA** 技术绕过 CPU，实现零拷贝、高吞吐的数据迁移。
   - 在 GPU 执行当前层时，提前从 CXL-Hybrid Memory 流水线式地拉取下一层权重和 KV 块。

3. **用户态显式预取接口（User-Directed Prefetching API）**
   - 提供 `chm_prefetch()` 等轻量级 API，允许推理框架（如 vLLM）主动触发预取。
   - 实现应用感知调度（application-aware scheduling），最大化 PCIe 带宽利用率。

4. **读优先 I/O 调度策略（Read-Priority I/O Scheduling）**
   - 解决 SSD 后台写入（flush chunks）阻塞前台读请求的问题。
   - 在 I/O 争用时，优先保障读操作，后台写入被推迟至 decode 阶段空闲窗口执行。

---

### 相比现有方法的优势
| 维度 | 传统 DPU-JBOF + NVMe-oF | ITME |
|------|------------------------|------|
| **访问粒度** | 块设备（block-level） | 字节可寻址（byte-addressable） |
| **协议开销** | 高（需处理 NVMe 协议栈） | 低（标准 RDMA + load/store） |
| **预取能力** | 软件控制，复杂 | 硬件支持，高效流水线 |
| **成本效率** | 高（需要昂贵 DPU） | 低（仅需 RNIC + CXL 设备） |
| **延迟掩盖** | 弱 | 强（多级重叠流水线） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ShareGPT Dataset**：用于构建多轮对话场景，评估不同 turn 下 KV Cache 增长的影响。
- **Mooncake Dataset**：专门用于分析 KV Cache 复用行为和预取效果。

### 实验设置
- **测试平台**：
  - **GPU Server**：Dell R770，双路 Xeon 6730，256GB DDR5，NVIDIA A100 (80GB HBM)，双 ConnectX-6 100Gbps NIC。
  - **Remote Server**：同构配置，部署 CXL-Hybrid Memory。
- **网络互联**：双 100Gbps RDMA（RoCEv2）连接。
- **CXL-Hybrid Memory 配置**：
  - **DRAM Cache**：32GB（SK hynix CMM）
  - **Backend Storage**：2× KIOXIA PCIe Gen5 NVMe SSDs（总容量 2TB）
  - **接口**：PCIe Gen5 x8，理论带宽 ~22 GB/s

### 评估指标
- **Throughput**：tokens/sec
- **Speedup**：相对于 baseline 的加速比
- **Hit Rate**：KV Cache 命中率
- **TTFT (Time to First Token)**：首 token 延迟
- **Bandwidth Utilization**：实际达到的 RDMA/CXL 传输带宽

### 基线方法对比
1. **Recomputation Only**：不缓存 KV，每次重新计算。
2. **CPU Offloading (with NVMe-oF)**：将 KV Cache 卸载到本地 NVMe-oF 存储，无预取优化。
3. **Ideal GPU Memory**：假设所有 KV Cache 可放入 GPU 显存（理想上限）。
4. **Local NVMe-oF Baseline**：作为性能上界参考（消除网络延迟）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 场景 | ITME 性能提升 |
|------|------|---------------|
| Llama-3.1 8B | 多轮推理（Turn 5） | 较 Recomputation 提升 **1.81×** |
| Llama-3.1 70B | 多轮推理（Turn 5） | 较 Recomputation 提升 **1.80×** |
| Llama-3.1 70B | 扩展至 35 turns | 较 CPU-offload 提升 **35.7% throughput** |

> 注：当 host memory 耗尽（>21 turns）后，CPU-offload 性能崩溃至 recompute 水平，而 ITME 因有 T3.5 层仍保持稳定服务。

---

### 与基线方法的对比结果
- **vs. CPU Offloading**：
  - 初始阶段性能接近（因仍在 host memory 中命中）。
  - 当 host memory 满后，CPU-offload 性能骤降；ITME 凭借远程 CXL-Hybrid Memory 继续提供有效缓存。
  - 最终 **ITME 实现最高达 35.7% 的吞吐提升**。

- **vs. Local NVMe-oF**：
  - 初始性能一致（buffer hit）。
  - 一旦进入远程访问阶段，Local NVMe-oF 因缺乏读优先调度和预取机制，出现严重 I/O 阻塞，性能迅速下降至 recompute 水平。
  - ITME 通过 **read-priority scheduling** 和 **multi-tier prefetching** 成功避免该问题。

- **vs. Ideal GPU Memory**：
  - ITME 性能约为理想上限的 **60%~70%**，但成本显著更低，且具备可扩展性。

---

### 消融实验结果
#### （1）Weight Prefetching 分析（图 12a）
- 即使模型权重超出 DRAM Cache（如 70B 模型 105GB > 32GB），ITME 仍能通过预取维持接近 host memory baseline 的性能（仅差 1–5%）。
- 表明 **pipelined prefetching 能有效掩盖存储与网络延迟**。

#### （2）DRAM Cache 容量敏感性分析（图 12b）
- 使用 4GB DRAM 缓冲即可实现 **92% 的基准性能**。
- 16GB 以上收益趋于饱和。
- 说明 **小容量 DRAM + 高效预取 > 大容量静态缓存**。

#### （3）FPGA 原型验证（图 14）
- FPGA 实现平均读带宽 **18 GB/s**，写带宽 **12 GB/s**。
- 较 CMM 平台低 20–25%，主要受限于逻辑延迟和元数据更新开销。
- 验证了硬件可行性，仍有优化空间。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **CXL-Hybrid Memory 是实现大规模推理内存扩展的有效路径**：
   - 将 SSD 容量转化为 byte-addressable 远程内存，打破物理内存墙。
2. ✅ **确定性访问模式是高效预取的前提**：
   - 权重的 layer-wise 访问和 prefix KV 的 append-write 特性，使得精准预取成为可能。
3. ✅ **多级 DMA 流水线可有效掩盖延迟**：
   - 通过 overlapping computation with data movement，实现了接近本地内存的性能表现。
4. ✅ **读优先调度至关重要**：
   - 在混合读写负载下，必须优先保障关键路径上的读请求，否则会导致性能雪崩。

---

### 方法的局限性
1. **依赖特定硬件生态**：
   - 需要支持 CXL.mem 的设备（如 SK hynix CMM）和高性能 RDMA 网络。
2. **对非规则访问模式支持有限**：
   - 工作中的 working KV cache 若随机性强，则难以有效预取。
3. **FPGA 原型性能尚未达理论峰值**：
   - 元数据锁竞争和通道不平衡限制了最大带宽发挥。
4. **冷启动问题未深入讨论**：
   - 初始加载大量权重或 prefix KV 时仍存在显著延迟。

---

### 未来工作方向
1. **支持动态模型切换与共享缓存**：
   - 实现跨会话、跨用户的 KV Cache 共享与快速恢复。
2. **结合压缩与编码技术进一步降低成本**：
   - 在 CXL-Hybrid Memory 中引入 KV Cache 压缩。
3. **探索更智能的预取调度器**：
   - 基于 workload prediction 动态调整预取深度与粒度。
4. **向 CXL 3.0 及更高版本演进**：
   - 利用 CXL.cache 和 CXL.mem 更高级特性实现更细粒度一致性管理。

--- 

> **总结一句话**：  
> ITME 通过 **CXL-Hybrid Memory + 多级 DMA 预取 + 读优先调度**，实现了面向 LLM 推理的高效、低成本、可扩展的远程内存扩展方案，在真实 workload 下相较传统方法获得高达 **35.7% 的吞吐提升**，为下一代 disaggregated inference infrastructure 提供了可行架构范式。

</details>

---

### 3. [Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training](https://arxiv.org/abs/2606.12240)

**Authors**: Shilong Zong, Almuatazbellah Boker, Hoda Eldardiry  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.12240v1  

#### Abstract
Multivariate time-series data often exhibit complex temporal dependencies, irregular sampling, and heterogeneous dynamics across multiple time scales, making accurate sequence modeling particularly challenging. Traditional recurrent neural networks (RNNs), such as Long Short-Term Memory (LSTM) netwo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 RNN（如 LSTM）在建模**多变量时间序列**时面临以下挑战：
- 在离散时间步上操作，难以有效捕捉**连续且不规则的时间动态**；
- 单一模型结构无法很好地建模**异质性、多尺度的时间依赖关系**（例如快速变化的生理信号 vs 缓慢演化的临床趋势）；
- Liquid Neural Networks (LNNs) 虽然通过连续时间动力学缓解了部分问题，但标准 LNN 仍基于单一动态系统，表达能力受限。

此外，LNN 因需数值积分微分方程，计算开销较高，训练效率低。

---

### 🚀 提出的新方法与创新思路
作者提出了一种新型架构：**Multi-Rate Mixture-of-Experts (MR-MoE)**，并进一步扩展为 **MR-MoE-Attention**，其核心创新包括：

#### （1）基于 LNN 的 Mixture-of-Experts 框架
- 将多个 LNN 作为“专家”集成在一个 MoE 结构中；
- 每个专家可学习不同的时间模式，提升模型表达能力和鲁棒性；
- 引入门控网络（gating network）实现输入自适应的专家选择。

#### （2）Multi-Rate 设计：显式建模多时间尺度动态
- 不同专家被分配到不同时间常数（time constants），分别处理快、中、慢速演化的过程；
- 利用奇异摄动理论（singular perturbation theory）对快速专家进行准稳态近似（quasi-steady-state approximation），降低计算负担；
- 显式分离多尺度动态，减少不同过程间的干扰。

#### （3）双注意力机制增强
- **Feature-level Attention**：识别重要输入特征，抑制噪声或无关变量；
- **Temporal Attention**：聚焦于关键历史状态，改善长距离依赖建模；
- 首次将 MoE + 多速率 + 双注意力 + 连续时间动态整合于统一框架。

---

### ⚖️ 相比现有方法的优势
| 方法 | 局限性 | MR-MoE 的优势 |
|------|--------|----------------|
| LSTM | 离散时间建模，难处理不规则采样；易遗忘长期依赖 | 支持连续时间建模，更自然地处理不规则序列 |
| Monolithic LNN | 单一动态系统，难以捕获异质多尺度行为 | 多专家分工协作，显式解耦快慢动态 |
| 标准 MoE (with LNN) | 未考虑时间尺度差异，专家间可能重叠或冲突 | 引入 multi-rate 结构，提升专家专业化程度 |
| 单独使用注意力机制 | 忽视动态系统的结构性分解 | 结合 MoE 与注意力，兼顾结构多样性与选择性关注 |

> ✅ **MR-MoE 是首个同时结合连续时间动态、专家分解、多时间尺度建模与双注意力机制的统一框架**。

---

## 2. 核心实验方法和设置

### 📊 数据集
- **任务**：脓毒症预测（sepsis prediction）
- **数据来源**：ICU 患者的多变量临床时间序列数据（来自 Moor et al. [2023]）
- **特征维度**：`d` 维生理指标（如心率、血压、血氧等）和实验室检测值
- **预处理**：
  - 归一化（normalization）
  - 前向填充缺失值（forward filling）
- **划分方式**：训练集 / 验证集 / 测试集（标准划分）

---

### 🔬 实验设置
- **实现平台**：PyTorch
- **优化器**：Adam
- **学习率**：1e-3
- **批量大小**：固定（across all models）
- **专家数量 K**：3（对应 fast, intermediate, slow 时间尺度）
- **每专家隐藏神经元数**：1500
- **门控网络**：小型 MLP
- **注意力模块**：
  - Feature-level Attention：两层 MLP
  - Temporal Attention：标准点积注意力（dot-product attention）

---

### 🎯 评估指标
针对类别不平衡的医学预测任务，采用以下指标：
- **AUROC**（Area Under the ROC Curve）
- **AUPRC**（Area Under the Precision-Recall Curve）

> 注：AUPRC 对正负样本不平衡更敏感，在脓毒症这类稀有事件预测中尤为重要。

---

### 🆚 基线方法对比
| 模型 | 描述 |
|------|------|
| **LSTM** | 单层 LSTM，隐藏维度 1500 |
| **Monolithic LNN** | 单一 LNN，无专家分解 |
| **MoE (LNN experts)** | 多专家 MoE，每个专家为 LNN，但无 multi-rate 设计 |
| **MR-MoE** | 提出的多速率 MoE，不含注意力 |
| **MR-MoE-Attention** | 完整模型，含 feature-level 和 temporal attention |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（测试集）

| 模型 | AUROC | AUPRC |
|------|-------|--------|
| LSTM | ~0.53 | ~0.22 |
| Monolithic LNN | ~0.55 | ~0.32 |
| MoE (LNN experts) | ~0.58 | ~0.36 |
| MR-MoE | ~0.61 | ~0.42 |
| **MR-MoE-Attention** | **~0.65 (~0.68)** | **~0.45** |

> ✅ **MR-MoE-Attention 达到最优性能，显著优于所有基线**

---

### 🔍 与基线方法的对比结果
- 相比 LSTM：
  - AUROC ↑ 12%（0.53 → 0.65）
  - AUPRC ↑ 105%（0.22 → 0.45）
- 相比 Monolithic LNN：
  - AUROC ↑ 10%（0.55 → 0.65）
  - AUPRC ↑ 40.6%
- 相比标准 MoE：
  - AUROC ↑ 7%，AUPRC ↑ 25%
- 表明：**multi-rate 结构 + 注意力机制带来持续增益**

---

### 🔁 消融实验分析（Ablation Study）

#### （1）逐步添加组件的效果（见 Figure 11 & 12）
- 从 LNN → MoE：引入专家分工，AUROC 提升约 3 pts
- 从 MoE → MR-MoE：加入 multi-rate 分解，再提升 3 pts
- 从 MR-MoE → MR-MoE-Attention：加入双注意力，再提升 4 pts
> ➤ 各组件贡献可叠加，验证设计有效性

#### （2）注意力机制的作用
- **Feature-level Attention**：提升信噪比，尤其在高噪声环境下表现稳定
- **Temporal Attention**：增强长程依赖建模，避免信息瓶颈

#### （3）multi-rate 结构的有效性
- 快速专家捕捉瞬时波动（如突发性心率升高）
- 慢速专家跟踪长期趋势（如乳酸水平缓慢上升）
- 减少动态过程之间的相互干扰

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **连续时间建模优于离散时间建模**  
   LNN 整体优于 LSTM，说明连续动态更适合不规则时间序列。

2. **专家分解显著提升表达能力**  
   MoE 结构通过条件计算（conditional computation）提高模型多样性，降低偏差与方差。

3. **显式 multi-rate 建模是关键突破**  
   不同时间尺度的动态应被显式分离，而非由单一系统强行拟合。

4. **注意力机制进一步提升鲁棒性与解释性**  
   - Feature-level attention 可用于特征重要性分析；
   - Temporal attention 可可视化关键历史时刻，辅助医生决策。

5. **MR-MoE 在性能与效率之间取得良好平衡**
   - 内存消耗低于 LSTM（见 Figure 13）
   - 快速专家采用准稳态近似，降低积分成本
   - 注意力带来适度内存开销，但收益更高

6. **更强的抗噪能力**
   - 在噪声增加场景下（Figure 14），MR-MoE-Attention 性能下降最缓慢；
   - 多专家结构 + 注意力共同作用，有效隔离噪声传播路径。

---

### ⚠️ 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **计算复杂度仍高于简单模型** | 尽管有优化，但 LNN 的 ODE 积分和 attention 仍带来一定开销 |
| **时间常数需手动设定** | 当前 Tk 固定且人为指定，未能自动学习最优时间尺度 |
| **缺乏统计显著性检验** | 实验未报告误差棒或 p-value，受算力限制 |
| **尚未开源代码与数据** | 可复现性依赖作者后续发布 |

---

### 🔮 未来工作方向
1. **Decoupled Multi-Time-Scale Training**
   - 探索分层或交替训练策略，解耦快慢专家的优化过程，避免梯度干扰。

2. **Learnable Time Constants**
   - 将时间常数 $ \tau_k $ 设为可学习参数，让模型自动发现最佳时间尺度。

3. **轻量化设计以部署至边缘设备**
   - 结合稀疏激活、知识蒸馏等技术，推动在实时医疗监测中的应用。

4. **拓展至其他领域**
   - 如金融时序预测、工业传感器数据分析等具有多尺度动态的场景。

---

## ✅ 总结
该论文提出了一个结构化、可解释、高性能的时间序列建模框架 **MR-MoE(-Attention)**，成功融合了：
- **Continuous-time dynamics**（via LNN）
- **Structured diversity**（via MoE）
- **Multi-scale decomposition**（via multi-rate design）
- **Adaptive selection**（via dual attention）

在脓毒症预测任务上取得了 SOTA 性能，并展现出良好的鲁棒性与计算效率，为复杂临床时间序列建模提供了新范式。

</details>

---

### 4. [Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling](https://arxiv.org/abs/2606.12370)

**Authors**: Yucheng Li, Huiqiang Jiang, Yang Xu, Jianxin Yang, Yi Zhang, Yizhong Cao, Yuhao Shen, Fan Zhou, Rui Men, Jianwei Zhang, An Yang, Bowen Yu, Bo Zheng, Fei Huang, Junyang Lin, Dayiheng Liu, Jingren Zhou  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.12370v1  

#### Abstract
Reinforcement learning (RL) has become a key component in modern large language models, yet the rollout stage remains the key bottleneck in RL training pipelines. Although Multi-Token Prediction (MTP) offers a natural solution to accelerate rollouts through speculative decoding, many studies have ob...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在大语言模型（LLM）的强化学习（RL）训练中，**rollout 阶段是主要计算瓶颈**。虽然 Multi-Token Prediction（MTP）通过 speculative decoding 能加速推理，但在 RL 训练过程中，MTP 的 **acceptance rate 显著下降**，导致实际加速效果有限。

已有研究将此归因于策略模型更新带来的分布不匹配（distribution mismatch），并提出在线更新 MTP 模块来缓解，但这引入了额外的内存和延迟开销。

本文指出，**真正主导 acceptance rate 下降的是策略模型熵（entropy）的波动**，而非分布不匹配。

---

### **提出的新方法与新思路**

作者提出了 **Bebop**，一个系统性的 MTP 在 LLM 后训练中的优化框架，其核心创新包括：

#### **(1) 揭示熵对 MTP 接受率的根本约束**
- 发现 MTP 的 acceptance rate 与目标模型的 entropy 存在**显著的负线性关系**。
- 该关系在多种任务和模型规模下均成立，表明 entropy 是 MTP 性能退化的主要驱动因素。

#### **(2) 提出基于 Rejection Sampling 的概率性采样机制**
- 相比传统的 **Target-Only Sampling**（仅用 argmax 和目标概率接受），**Rejection Sampling** 利用完整的分布重叠（distributional overlap）进行接受判断。
- 其接受率由 Total Variation (TV) 距离决定：  
  $$
  \alpha_{\text{RS}} = 1 - d_{\text{TV}}(p, q)
  $$
- 由于 TV 距离对 entropy 变化更鲁棒，Rejection Sampling 显著缓解了 entropy 波动带来的影响。

#### **(3) 提出端到端的 TV Loss（e2e TV Loss）**
- 传统 MTP 训练使用 CE 或 KL 损失，这些目标间接优化 TV 距离，效率低下。
- 本文提出直接最小化 TV 距离作为训练目标，并进一步设计了 **end-to-end 多步 TV Loss**，以优化整个 MTP 链条的期望接受长度：
  $$
  \mathcal{L}_{\text{e2e}} = 1 - \prod_{i=1}^{\gamma} (1 - d_{\text{TV}}(p_i, q_i))
  $$
- 该损失具有**有界梯度**，训练稳定，且能实现“概率比例型”误差（probability-proportional mismatch），使 draft 分布更贴近目标。

#### **(4) 提出 Pre-RL MTP 适应策略**
- 实验证明，在 RL 开始前使用 e2e TV Loss 进行一次轻量级 MTP 微调（pre-RL adaptation），即可在整个 RL 训练中保持高且稳定的 acceptance rate。
- **无需在 RL 中在线更新 MTP 模块**，避免了额外开销。

---

### **相比现有方法的优势**
| 维度 | 现有方法 | Bebop |
|------|--------|-------|
| **采样方式** | Target-Only Sampling | Rejection Sampling（更鲁棒） |
| **训练目标** | CE / KL Loss | e2e TV Loss（直接优化接受率） |
| **MTP 更新策略** | 在线 co-training（高开销） | Pre-RL 一次性训练（低开销） |
| **对 entropy 敏感性** | 高 | 极低（近乎不变） |
| **实际加速比** | 有限（~1.2–1.4×） | 高达 **1.8×** |

---

## **2. 核心实验方法和设置**

### **使用的数据集与任务**
- **数学推理**：HMMT25, AIME25, LiveCodeBench
- **代码生成**：SWE-Bench（多轮代码编辑）
- **代理任务（Agentic Tasks）**：Hybrid, Long-Horizon, Tool Use
- **通用对话**：MT-Bench（含 OOD 评估）

### **模型**
- 主要使用 **Qwen3.5、Qwen3.6、Qwen3.7** 系列模型（从 3.5B 到 37B 参数）
- MTP 设置为 γ=3（预测 3 个 token）

### **实验设置**
- **SFT 阶段**：使用混合 RFT 数据训练 MTP 模块，比较不同损失函数的效果。
- **RL 阶段**：采用异步 RL 框架（async RL），rollout 引擎为 SGLang，策略优化算法为 GRPO。
- **MTP 部署**：在 rollout 阶段启用 MTP 加速。

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Acceptance Rate / Accept Length** | 每次验证步骤平均接受的 token 数 |
| **Throughput Gain** | 推理吞吐提升百分比 |
| **End-to-End Speedup** | RL 训练总时间加速比 |
| **Latency Reduction** | 单步 rollout 延迟降低 |
| **TV Distance** | draft 与 target 分布之间的 Total Variation 距离 |

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **CE Loss + Target-Only** | 传统 MTP 训练方式 |
| **KL Loss + Target-Only** | 使用 KL 散度训练 |
| **CE Loss + Rejection Sampling** | 改进采样方式 |
| **Online MTP Update (CE/TV)** | 在 RL 中持续更新 MTP 权重 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | 结果 |
|------|------|
| **最高 Acceptance Rate** | 达到 **95%**（如 Qwen3.7-Plus 在 Agent 任务上） |
| **Acceptance Rate 提升** | 相比 CE 基线，**提升 ~10%**（e.g., 从 75% → 85%） |
| **推理吞吐增益** | 最高 **+25%** 吞吐提升 |
| **端到端加速比** | **高达 1.8×** 异步 RL 训练加速 |
| **Agent 任务加速** | rollout 阶段最高 **2.4×** 速度提升 |

---

### **与基线方法的对比结果**
#### **Table 2 & Table 3：不同任务下的 Acceptance Rate 对比（γ=3）**
| MTP Loss | Math | Code | SWE | Agent | MT-Bench (OOD) |
|---------|------|------|-----|-------|----------------|
| CE Loss | 75.0 | 71.3 | 75.1 | 90.3 | 65.3 |
| KL Loss | +0.0 | +0.0 | +0.2 | +0.2 | +0.0 |
| Reverse KL | +1.3 | +1.0 | -0.2 | +1.0 | +0.5 |
| TV Loss | +2.4 | +2.5 | +3.3 | +5.2 | +1.4 |
| **e2e TV Loss (Ours)** | **+3.0** | **+3.3** | **+8.0** | **+6.7** | **+2.3** |

> ✅ 在所有任务上均优于 CE/KL，尤其在 SWE 和 Agent 任务上提升最大。

---

### **消融实验结果**
#### **(1) 不同采样方式的影响**
- **Target-Only Sampling**：acceptance rate 严重依赖 entropy，随 entropy 上升快速下降。
- **Rejection Sampling + CE Loss**：虽有所改善，但仍受 entropy 影响。
- **Rejection Sampling + TV Loss**：acceptance rate 几乎不受 entropy 变化影响（slope 从 -1.68 降至 -0.06）。

#### **(2) 是否需要在线更新 MTP？**
- 实验表明：**在 RL 中继续更新 MTP 权重无明显收益**。
- 若使用 CE Loss 更新，反而会使 TV 训练获得的优势退化。
- **Pre-RL 一次性训练 + Rejection Sampling 已足够**。

#### **(3) 模型大小的影响**
- MTP acceptance rate 随模型增大而提高：
  - Qwen3.6-35A3B: ~78%
  - Qwen3.7-Plus: ~87%
  - Qwen3.7-Max: ~95%
- 表明更大模型的 draft head 更容易拟合目标分布。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Entropy 是 MTP 接受率下降的主因**：  
   在 RL 训练中，策略模型 entropy 的上升直接限制了 MTP 的接受能力，其影响远大于分布不匹配。

2. **Rejection Sampling 显著缓解熵敏感性**：  
   相比 Target-Only Sampling，其接受率基于完整分布重叠，对 entropy 变化更鲁棒。

3. **TV Loss 直接优化接受率，优于 CE/KL**：  
   - CE/KL 优化的是 KL 散度，与 TV 距离仅有松散上界关系。
   - TV Loss 实现“概率比例误差”，使 draft 分布更贴近目标，且训练更稳定。

4. **Pre-RL MTP 训练已足够**：  
   只需在 SFT 阶段用 e2e TV Loss 训练一次 MTP 模块，即可在整个 RL 中保持高性能，**无需在线更新**。

5. **实际加速效果显著**：  
   在 Qwen3.5–3.7 系列模型上，实现 **高达 1.8× 的端到端 RL 加速**，尤其在长上下文、多轮交互任务中优势明显。

---

### **局限性**
1. **理论假设依赖近似**：  
   如“概率比例误差”是基于梯度行为的经验建模，尚未严格证明。

2. **熵不变性有条件限制**：  
   TV 训练的 entropy 鲁棒性依赖于 SFT 数据覆盖的 entropy 范围。若 RL 探索导致 entropy 超出训练范围，则仍可能出现退化。

3. **高内存开销**：  
   全词表 TV Loss 计算峰值内存较高，虽通过 fused kernel 优化，但仍可能限制部署。

---

### **未来工作方向**
- **动态调整 MTP 步数 γ**：根据局部 entropy 自适应选择预测长度。
- **Top-K TV Loss 优化**：探索更高效的 TV 近似方法以降低内存。
- **扩展至其他 speculative decoding 架构**：如早期退出（early-exit）、小模型 draft。
- **结合 RL 探索策略设计 entropy-aware MTP**：主动控制 entropy 以维持高接受率。

---

> **总结**：Bebop 通过揭示 entropy 对 MTP 的根本约束，提出 **Rejection Sampling + e2e TV Loss + Pre-RL Training** 的组合方案，实现了高效、稳定、低成本的 RL rollout 加速，为大规模 LLM 强化学习提供了实用且可扩展的技术路径。

</details>

---

### 5. [Re-evaluating Confidence Remasking in Masked Diffusion Language Models](https://arxiv.org/abs/2606.12232)

**Authors**: Stipe Frkovic, Metod Jazbec, Dan Zhang, Christian A. Naesseth, Ilija Bogunovic, Eric Nalisnick  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.12232v1  

#### Abstract
Masked diffusion language models (dLLMs) have recently emerged as a competitive alternative to autoregressive language models, with the promise of faster inference via parallel token generation. A notable limitation of the masked formulation, however, is that once a token has been unmasked it can no...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Re-evaluating Confidence Remasking in Masked Diffusion Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文聚焦于**post-hoc confidence-based remasking**在 **masked diffusion language models (dLLMs)** 中的实际有效性问题。尽管近期如 WINO [Hong et al., 2026] 等方法声称通过训练后、无需微调的方式引入 remasking 能显著提升生成质量，但其评估存在以下不足：
- 与较弱的 baseline（如随机或高置信度 unmasking）比较，而非当前最优的 **Fast-dLLM**；
- 主要评估在非标准的大 block length（如 BL=128）下进行，而实际中更常用的是小 block（如 BL=32）；
- 忽略了非贪婪解码（non-greedy decoding）等更具挑战性的生成场景。

因此，本文旨在**重新评估 remasking 的真实增益**，揭示其效果是否被高估，并提出更全面的评估框架。

### 提出的新思路
本文**并未提出新的 remasking 方法**，而是对现有方法（以 WINO 为代表）进行了系统性再评估，提出了以下关键分析视角：
- 将 remasking 的收益与当前最强的 confidence-based unmasking 方法（Fast-dLLM）进行公平对比；
- 在不同 decoding 设置（block length、sampling temperature、unmasking 策略）下测试 remasking 的鲁棒性；
- 引入 **flip-flop frequency** 和 **shadow-token approximation quality** 等诊断工具，深入分析 remasking 失效的根本原因。

### 相比现有方法的优势
- **批判性视角**：指出当前 remasking 研究可能存在“虚假增益”，即其优势源于补偿了次优 baseline 的缺陷，而非真正提升了模型能力上限。
- **系统性评估框架**：提出应综合考虑 block size、unmasking 策略、sampling temperature 等因素，建立更全面的评估标准。
- **深入归因分析**：通过 flip-flop 分析揭示了 remasking 失效的本质是 dLLM 自身无法在相同上下文中生成更好的替代 token。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GSM8k**：数学应用题数据集
- **MATH-500**：高等数学问题数据集
- **HumanEval**：代码生成任务
- **MBPP**：面向编程的文本到代码任务

### 实验设置和评估指标
- **模型**：
  - LLaDA-8B-Instruct [Nie et al., 2025]
  - Dream-v0-Instruct-7B [Ye et al., 2025]
- **Block Length (BL)**：32（标准设置） vs 128（WINO 原始设置）
- **Decoding 温度**：T = 0（greedy），T = 0.8 / 1.5（non-greedy）
- **评估指标**：
  - **Accuracy / Pass@k**：衡量任务完成率
  - **NFEs (Network Function Evaluations)**：衡量计算效率
  - **Throughput (tokens/sec)**：衡量实际推理速度
  - **Flip-flop frequency**：衡量 remask 后 token 是否恢复原值的比例

### 基线方法对比
- **Fast-dLLM** [Wu et al., 2025]：基于置信度阈值的 adaptive unmasking，作为主要强 baseline
- **WINO** [Hong et al., 2026]：在 Fast-dLLM 基础上增加基于 shadow token 的 confidence-based remasking
- **Saber** [Dong et al., 2025]：另一种 post-hoc remasking 方法，用于补充验证
- **dUltra** [Chen et al., 2025]：基于 RL 学习的 Bernoulli unmasking 策略，用于测试 remasking 与随机 unmasking 的交互

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 在标准设置下（BL=32, T=0）：
- **WINO 相比 Fast-dLLM 几乎无增益**：
  - 平均 accuracy 提升仅 **~0.4–0.5%**
  - 在 **throughput** 指标上，WINO 因 shadow block 开销更大，**性能反而更低**
  - 图 7 显示，在 wall-clock time 上 WINO 的优势进一步缩小甚至消失
- **消融实验（图 10）**：
  - 改变 remasking 判据（如 consistency-based）、loop-guard、attention mask 结构等，均未带来显著提升
  - 表明 WINO 的设计选择对其在 BL=32 下的无效性影响不大

#### ⚠️ 在大 block 设置下（BL=128）：
- WINO 相比 Fast-dLLM 有较大提升（平均 **+1.3–1.5%**）
- 但 **Fast-dLLM @ BL=32 仍优于 WINO @ BL=128**
- 说明 remasking 主要是**修复了 Fast-dLLM 在大 block 下的退化问题**，而非提升模型本质能力

#### 🔄 Flip-flop 分析（图 3）：
- **极高 flip-flop 率**：
  - LLaDA：~75–90%
  - Dream：~85–95%
- 表明即使位置被 remask，模型仍倾向于预测回原 token，**remasking 未能引导出更好结果**

#### 🔥 在非贪婪解码下（T > 0）：
- **Pass@1 提升明显**（T=0.8 时平均 +2.6%），说明 remasking 可修正部分随机错误
- 但 **Pass@k 增益随 k 增大迅速衰减**（Pass@64 仅 +0.7%）
- 表明 remasking **加剧了 confidence-based unmasking 已有的 diversity collapse 问题**

#### 🎲 在随机 unmasking（dUltra）下：
- **WINO 带来显著提升**（平均 +3.2% accuracy）
- 且仅增加约 2 NFEs
- 说明 remasking 在**引入更多随机性**的 unmasking 策略下更有价值

---

## 4. 关键结论和发现

### 主要发现
1. **Post-hoc remasking 的收益高度依赖于解码设置**：
   - 在标准设置（BL=32, greedy）下，**WINO 几乎不优于 Fast-dLLM**，且因额外开销导致实际效率更低。
   - 其在大 block 下的增益主要是**补偿 Fast-dLLM 的缺陷**，而非提升模型天花板。
2. **Shadow token 是高质量近似**：
   - 与 oracle leave-one-out 相比，shadow token 在性能和效率上达到良好平衡（图 2）。
3. **Remasking 失效的根本原因是模型自身限制**：
   - 高 flip-flop 率表明，**dLLM 无法在相同上下文中生成更优 token**，即使位置被重新 masked。
   - 扩展 remasking 范围（邻居或时间步）也无法缓解此问题（Appendix B.2）。
4. **Remasking 在随机生成中更有潜力**：
   - 当与 **stochastic unmasking 策略（如 dUltra）** 结合时，remasking 能有效纠正随机错误，带来显著提升。
5. **Remasking 可能损害生成多样性**：
   - 在 non-greedy decoding 中，remasking 虽提升 Pass@1，但抑制了多解探索，**加剧 diversity collapse**。

### 方法的局限性
- **Post-hoc remasking 无法突破 dLLM 自身建模能力的瓶颈**：若模型无法在给定上下文中生成更好 token，则 remasking 无效。
- **引入额外计算开销**：shadow block 增加 FLOPs 和延迟，可能抵消其微弱准确性增益。
- **可能抑制生成多样性**：过度依赖置信度进行 remasking 会限制探索空间。

### 未来工作方向
- **开发更有效的 remasking 机制**：需结合模型重训练（fine-tuning / RL）以增强模型自我纠错能力。
- **转向 uniform diffusion**：放弃 pure absorbing 过程，采用支持自然 remasking 的 uniform transition（如 Wang et al., 2026）。
- **建立标准化评估框架**：未来研究应统一在多种设置（block size、temperature、unmasking 策略）下评估 remasking 方法。
- **探索 remasking 对多样性的影响**：无论是 post-hoc 还是训练式 remasking，都需系统研究其对生成多样性的权衡。

--- 

> **总结一句话**：  
> 当前的 post-hoc confidence-based remasking（如 WINO）在标准设置下对 masked dLLMs 的提升**非常有限**，其效果高度依赖于解码策略，且受限于模型自身的生成能力；未来应转向更根本的建模改进或建立更全面的评估体系。

</details>

---

### 6. [Eidola: Modeling Multi-GPU Network Communication Traffic in Distributed AI Workloads](https://arxiv.org/abs/2606.12638)

**Authors**: Ranganath R. Selagamsetty, Matthew Poremba, Bradford M. Beckmann, Joshua San Miguel, Mikko H. Lipasti  
**Category**: cs.DC  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.12638v1  

#### Abstract
As distributed AI workloads grow in scale, multi-GPU systems have become essential for training large models. Although techniques like kernel fusion and overlapping communication with computation help reduce delays, they also introduce irregular and transient traffic patterns that are difficult to m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Eidola: Modeling Multi-GPU Network Communication Traffic in Distributed AI Workloads**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代分布式AI训练工作负载（如Transformer、LLMs）广泛采用 **kernel fusion** 和 **通信-计算重叠** 技术以提升效率。然而，这些技术引入了复杂、瞬态且不规则的 **multi-GPU 通信流量模式**，尤其是依赖细粒度同步（fine-grained synchronization）和 **peer-to-peer 写操作**（如通过 xGMI）。现有的仿真工具（如 gem5、gem5-gpu、GPGPU-Sim）缺乏对这类通信行为的建模能力，限制了架构研究。

### ✅ 提出的新方法与创新
作者提出了 **Eidola** —— 一个可扩展的 **gem5 扩展框架**，用于在大规模 multi-GPU 系统中精确建模网络通信流量。

#### 主要创新点：
- **轻量级 GPU 抽象模型（eidolon）**  
  非目标 GPU 被抽象为“**eidolon**”（幻影），仅保留必要的通信行为特征，显著降低模拟开销。
  
- **基于时间标注的通信回放机制**  
  利用真实应用采集的 **timing profiles**（含 peer-to-peer 写操作的时间戳），通过新的 GPU 伪操作 `register_write` 在模拟前注册写事件，实现 **cycle-level 精度** 的通信建模。

- **Write Tracking Table (WTT)**  
  引入优先队列结构 WTT 来管理待触发的跨 GPU 写操作，支持任意顺序注册、按时间排序执行，确保时序准确性。

- **支持灵活的通信场景分析**  
  支持配置 per-GPU 流量模式，允许隔离分析不同通信延迟、同步机制的影响，适用于 **architectural exploration**。

### ✅ 相比现有方法的优势
| 工具 | 局限性 | Eidola 的优势 |
|------|--------|----------------|
| **gem5 / gem5-gpu** | 缺乏原生 multi-GPU 通信建模 | 支持 xGMI 级别的 peer-to-peer 写建模 |
| **GPGPU-Sim** | 单 GPU 微架构仿真为主 | 支持多 GPU 间通信与同步行为 |
| **MGPUSim** | 全细节模拟所有 GPU，扩展性差 | 仅详细模拟目标 GPU，其余抽象为 eidolon，**可扩展至数百 GPU** |

> ✅ **核心优势：高保真 + 可扩展 + 易于集成真实 workload 行为**

---

## 2. 核心实验方法和设置

### 🧪 实验目标
验证 Eidola 是否能准确建模 multi-GPU 通信行为，并支持对新型同步机制（如 SyncMon）的研究。

### 🔧 实验设置
- **目标 kernel**：**fused GEMV+AllReduce**（来自 [30]），典型用于 Transformer 中的注意力层。
- **硬件配置（模拟）**：
  - 模拟 GPU 数量：从 3 到 255 个 eGPUs（emulated GPUs）
  - 每个 GPU 包含 43 个 Compute Units (CUs)
  - 使用 AMD CDNA 架构模型（MI100/MI200/MI300 系列）
- **通信接口**：建模 **xGMI**（external Global Memory Interconnect）进行 peer-to-peer 写操作。
- **同步机制**：基于非缓存内存区域的 flag polling，使用 rocSHMEM 接口。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Memory Read Requests** | 分为 flag-related（轮询标志位）和 non-flag（正常计算访存） |
| **Simulation Wall-clock Time** | 评估模拟效率与可扩展性 |
| **Scaling Behavior** | 输入规模（M 维度）和 GPU 数量变化下的模拟时间增长趋势 |

### ⚖️ 基线方法对比
- **Baseline**：标准 spin-wait 同步（持续轮询 flag）
- **Enhanced**：使用 `monitor()` / `mwait()` 实现的 **spin-yield** 同步（受 SyncMon[16] 启发）
- **对比维度**：
  - 内存读请求数量随等待时间的变化
  - 模拟时间随输入大小和 GPU 数量的增长情况

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）Spin-wait 流量随延迟线性增长（图6）
- 当 `wakeupTime` 从 0μs 增加到 40μs：
  - **flag-related read requests** 从 ~134 增加到 ~660
  - 呈现出明显的 **线性关系**
- ➤ 验证了 Eidola 能精确复现 **fine-grained synchronization behavior**

#### （2）启用 SyncMon-style spin-yield 后，内存流量大幅下降（图9）
- 在相同等待时间内：
  - **flag reads 保持恒定在 728–788 次之间**
  - 相比 baseline 下降超过 **90%**
- non-flag reads 维持约 66K，说明计算行为未受影响
- ➤ 成功复现了 SyncMon 的核心收益：**减少无谓的 polling memory traffic**

#### （3）模拟时间随输入规模线性增长（图10）
- 输入矩阵维度 M 从 1024 到 24576
- 模拟时间呈近似线性增长（r² = 0.76~0.98）
- 启用 `mwait` 后趋势一致，表明新增机制 **不影响整体缩放特性**

#### （4）模拟时间随 eGPU 数量亚线性增长（图11）
- 最大配置：255 eGPUs
- 归一化模拟时间仅为单 GPU 的 **7.3× 至 35.9×**
- 远低于理想全模拟的 256× 开销
- ➤ 证明 Eidola 具备 **良好的可扩展性**

#### （5）每 eGPU 开销极低
- 通过线性回归估计：
  - 单 GPU 模拟成本：`t₁GPU`
  - 每增加一个 eGPU 的额外开销：`t_eGPU` 极小
- 表明 WTT 和通信回放机制 **引入的 overhead 很小**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Eidola 成功填补了 multi-GPU 通信建模的空白**  
   提供了一个 **cycle-accurate、configurable、scalable** 的仿真平台，可用于研究 fused kernels 中的通信与同步行为。

2. **真实 timing profile 驱动的通信回放是有效的**  
   使用轻量级 profiling 数据即可重建复杂的跨 GPU 交互行为，无需完整模拟所有设备。

3. **支持快速原型化新型同步机制**  
   成功实现了 SyncMon 的关键功能，验证其在减少 memory traffic 上的有效性，展示了 Eidola 对 **microarchitectural innovation** 的支撑能力。

4. **具备优异的可扩展性**  
   模拟数百 GPU 的系统仍保持较低 overhead，适合研究大规模分布式训练中的性能瓶颈。

### ⚠️ 方法的局限性
- **依赖外部 profiling 数据**：需要先运行真实程序获取 timing traces，不能完全脱离硬件。
- **当前基于 polling**：虽然支持 mwait，但底层仍是周期性检查 WTT，未来可改用 gem5 原生 event queue 提升效率。
- **侧重通信建模，非计算细节**：非目标 GPU 的计算行为被忽略，仅关注其通信输出。

### 🔮 未来工作方向
1. **扩展到更大规模系统**：在跨节点、复杂拓扑（如 2D torus）上验证通信模式。
2. **支持更多 fused kernels**：如 embedding pooling + All-to-All、GEMM+All-to-All 等。
3. **集成 event-driven backend**：将 WTT 替换为 gem5 原生 event queue，进一步降低模拟开销。
4. **支持异构与非对称 workload**：利用 user-defined profiling 支持 producer-consumer 类型任务。

---

## 总结

> **Eidola 是首个将 cycle-level 多 GPU 通信建模与高度可扩展性结合的 gem5 扩展框架**。它通过“**目标 GPU 详尽模拟 + 其余 GPU 抽象为 eidolon**”的设计，在保证精度的同时实现了对上百 GPU 系统的高效仿真。该工作为未来 GPU 架构设计、互连优化和同步机制探索提供了强有力的工具支持。

</details>

---

### 7. [CRUMB: Efficient Prior Fitted Network Inference via Distributionally Matched Context Batching](https://arxiv.org/abs/2606.11473)

**Authors**: Jamie Heredge, Mattia J. Villani, Pranav Deshpande, Akshay Seshadri, Niraj Kumar  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11473v1  

#### Abstract
Prior-fitted networks (PFNs) are a promising class of tabular foundation models that perform in-context learning, whereby the entire labelled training set is supplied as context, and predictions for test queries are produced in a single forward pass. However, the quadratically scaling self-attention...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CRUMB: Efficient Prior Fitted Network Inference via Distributionally Matched Context Batching

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Prior-fitted networks (PFNs) 是一种强大的 tabular foundation model，能够通过 in-context learning 在单次前向传播中完成预测。然而，其基于 self-attention 的架构导致推理成本随训练数据量 $N$ 呈 **二次方增长**（$O(N^2)$），在大规模数据集上变得不可行。

此外，现有的上下文选择方法如 **kNN** 虽能提升局部相关性，但每个测试点可能拥有不同的上下文，无法进行批量处理（batching），导致需要 $T$ 次独立的前向传播，效率低下。

### 提出了什么新方法或新思路
本文提出 **CRUMB**（Clustered Retrieval Using Minimised-MMD Batching），一个无需重新训练的 inference wrapper，包含三个阶段：
1. **Clustering Test Queries**：使用 k-means 将测试样本划分为 $K$ 个簇。
2. **Distributionally Matched Context Selection**：对每个测试簇 $C_k$，通过 **greedy MMD minimisation (kernel herding)** 从训练集中选出一个分布匹配的小子集 $S_k$。
3. **Batched PFN Inference**：为每一对 $(S_k, C_k)$ 执行一次 PFN 推理，实现高效批处理。

### 相比现有方法的优势
- **兼顾质量与效率**：相比 uniform subsampling 更精准；相比 per-query kNN 可批量处理，显著减少前向传播次数（从 $T$ 到 $K \ll T$）。
- **架构无关、无需微调**：作为 inference-time wrapper，适用于任何 PFN 架构（如 TabPFNv2, TabICLv1/v2）。
- **对协变量漂移（covariate drift）鲁棒**：MMD 对齐机制使训练上下文动态适应当前测试分布，优于固定训练侧聚类的方法（如 MICP）。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要基准：**TabArena**，包含 **51 个多样化的表格分类与回归数据集**。
- 大规模补充实验：引入超大数据集，如 **Higgs**（粒子物理）、**Diabetes130US**（57k）、**APSFailure**（61k）、**SDSS17**（62k）、**customer_satisfaction_in_airline**（104k）、**GiveMeSomeCredit**（120k），部分截断至 100k 样本。

### 实验设置和评估指标
- **模型**：在三种 PFN 架构上测试 —— **TabPFNv2**, **TabICLv1**, **TabICLv2**。
- **上下文预算**：固定为 $n = 0.1N$（即仅使用 10% 的训练样本作为上下文）。
- **评估指标**：
  - 分类任务：**Accuracy**
  - 回归任务：**RMSE**
  - 综合排名：使用 **平均排名（Average Rank）** 和 **Wilcoxon signed-rank test with Bonferroni correction** 进行统计显著性分析。
  - 大数据实验额外报告 **wall-clock inference time**（单 A100 GPU）。

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Full Context** | 使用全部训练数据，作为性能上限（但计算昂贵） |
| **Uniform Subsampling** | 随机均匀采样 $n$ 个训练点作为共享上下文 |
| **per-query kNN** | 每个测试点取其最近邻 $n$ 个训练点作为上下文，无法批量 |
| **MICP**（Mixture of In-Context Prompters）| 在训练数据上聚类，构建固定支持集，测试点路由到最近训练簇 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 在 TabArena 上的整体表现（Table 1）
| Method | Avg. Rank | Significant vs CRUMB? | # PFN Passes |
|--------|-----------|------------------------|--------------|
| Full Context | 2.20 | Yes | 1 |
| per-query k-NN | 2.78 | No ($p_{adj}=0.106$) | T |
| **CRUMB (ours)** | **2.98** | — | **K=20** |
| MICP | 3.30 | Yes | K* |
| Uniform | 3.74 | Yes | 1 |

> **结论**：CRUMB 性能接近 per-query kNN（无显著差异），远优于 MICP 和 Uniform，且仅需 **20 次前向传播**（vs kNN 的数百次），实测速度提升约 **10×**（35s vs 361s）。

#### ✅ 多模型 & 多采样比例下的稳定性（Table 2）
在 38 个分类数据集、3 种 PFN 模型、5 种采样比例下验证：

| Method | Avg. Rank | Significant vs CRUMB? |
|--------|-----------|------------------------|
| **CRUMB** | **1.809** | — |
| MICP | 2.022 | Yes ($p_{adj} \sim 10^{-8}$) |
| Uniform | 2.169 | Yes ($p_{adj} \sim 10^{-9}$) |

> **结论**：CRUMB 的优势具有高度一致性，不受模型或上下文大小影响。

#### ✅ 大数据场景下结合 Early Stopping 的效果（Figure 4）
- 在 $N > 50k$ 数据集上，CRUMB 启用 early stopping 后：
  - 平均上下文大小 **小于预算 $n$**，但仍保持高性能。
  - 在多个数据集上 **胜率高于 MICP**，尤其在 TabPFN-v2 和 TabICL-v1 上差异显著（$p < 0.01$）。

#### ✅ 协变量漂移（Covariate Drift）下的鲁棒性（Table 3, Figure 5）
模拟可控的协变量漂移（通过 PCA 第一分量滑动窗口控制漂移强度 $T$）：

| $T$（漂移强度） | CRUMB 准确率 | MICP 准确率 | $\Delta$ Accuracy |
|------------------|---------------|-------------|--------------------|
| 0（无漂移）      | 0.794         | 0.746       | **+4.9%**          |
| 0.5              | 0.788         | 0.663       | **+12.5%**         |
| 1.0（完全OOD）   | 0.699         | 0.528       | **+17.1%**         |

> **结论**：随着漂移加剧，CRUMB 相对于 MICP 的优势持续扩大，证明其天然具备抗分布偏移能力。

### 消融实验结果（Ablation Studies）

#### 🔹 不同检索策略对比（Figure 7）
在同一测试聚类框架下比较三种上下文选择方式：
- **MMD Herding**（CRUMB 默认）
- **Centroid-NN**（选离测试簇中心最近的训练点）
- **Voronoi-Uniform**（按 Voronoi 区域均匀采样）

> **结果**：MMD Herding 在 accuracy 和 R² 上均显著优于其他两种，说明 **显式的分布对齐目标至关重要**。

#### 🔹 聚类与检索的交互作用（Figure 9）
设计四组对照实验：
- Case A: No clustering + Uniform → Baseline
- Case B: Clustering + Uniform → 效果未改善
- Case C: No clustering + MMD → 改善有限
- **Case D: Clustering + MMD → 最优**

> **结论**：性能提升来源于 **聚类与 MMD 检索的协同效应** —— 聚类创造局部化目标分布，MMD 才有可优化的空间。

#### 🔹 加速技术有效性（Appendix D）
- 使用 **Random Fourier Features (RFF)** 和 **Batched Greedy Selection**（$B=50$）后：
  - 推理时间降低近 **10×**
  - 性能损失极小（< 0.2 pp）
- **Early stopping** 可自适应决定上下文大小，进一步节省资源。

---

## 4. 关键结论和发现

### 主要发现
1. **CRUMB 成功解决了 PFN 推理中的“质量-效率”权衡难题**：
   - 通过 **test-side clustering + MMD-based context selection**，实现了高质量上下文选择与高效批处理的统一。
2. **分布对齐（distributional matching）优于几何邻近（geometric proximity）**：
   - MMD 最小化比单纯找最近邻更能代表测试分布，尤其在稀疏或非均匀区域。
3. **动态适应测试分布带来天然抗漂移能力**：
   - 与 MICP 等静态训练侧聚类不同，CRUMB 的上下文选择是 **测试驱动** 的，因此对 covariate drift 更鲁棒。

### 方法的局限性
- **需要完整测试集才能聚类**：不直接支持在线/流式场景（online setting）。
- **依赖 Euclidean k-means**：假设特征空间各向同性，在高维或混合类型数据上可能失效。
- **MMD 使用 isotropic RBF kernel**：未针对特定数据结构调整带宽或核函数。
- **预处理开销仍可观**：虽然为线性复杂度 $O(Nd(T+Kn))$，但在百万级数据上常数项较大。
- **受限于底层 PFN 能力**：若 pretraining prior 与目标任务差距过大，再好的上下文也难以补偿。

### 未来工作方向
- 设计 **streaming variant of CRUMB**，例如定期刷新缓存的聚类中心与上下文（见 Appendix C）。
- 引入 **adaptive kernels** 或 **learnable bandwidths** 以更好捕捉数据结构。
- 结合 **fine-tuning** 方法（如 LoCalPFN, CAPFN），探索 “context selection + lightweight adaptation” 的联合优化路径。
- 扩展至 **time-series 或 structured data** 场景，研究时序对齐下的 MMD 应用。

--- 

> 💡 **一句话总结**：  
> **CRUMB 通过 test-side clustering 与 MMD-driven context selection，在几乎不牺牲性能的前提下，将 PFN 的推理扩展到了数十万规模的数据集，并展现出卓越的分布外泛化能力。**

</details>

---

### 8. [Accurate and Resource-Efficient Federated Continual Learning](https://arxiv.org/abs/2606.11480)

**Authors**: Jebacyril Arockiaraj, Dhruv Parikh, Jayashree Adivarahan, Rajgopal Kannan, Viktor Prasanna  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11480v1  

#### Abstract
Federated continual learning (FCL) must learn from distributed task streams under limited resources, such as communication, computation, memory, and label availability. Existing FCL methods often rely on repeated local optimization, replay, and full supervision. Analytic alternatives avoid iterative...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Accurate and Resource-Efficient Federated Continual Learning**

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**联邦持续学习**（Federated Continual Learning, FCL）中的多重资源约束问题展开研究。现实场景下的FCL面临以下挑战：
- **通信开销大**：传统基于梯度的方法需要多轮模型更新，上传大量参数。
- **计算成本高**：客户端需进行多次反向传播和本地训练。
- **标签稀缺**：标注数据昂贵且稀疏，限制了监督信号的有效利用。
- **灾难性遗忘与特征漂移**：在非独立同分布（non-IID）客户端上迭代更新共享特征提取器会导致表示空间不稳定。

现有方法通常依赖于重复优化、回放机制或全监督设定，在资源受限环境下性能显著下降。

---

### 提出的新方法：FedRAN
作者提出 **FedRAN**（**Fed**erated **RAN**dom-feature Analytic Framework），一种**资源感知的解析式联邦持续学习框架**，其核心思想是：
- **用紧凑的随机特征统计量替代梯度更新**，避免迭代训练。
- 在服务器端通过**两层OR-SVD子空间合并**实现跨客户端（空间）和跨任务（时间）的知识聚合。
- 使用**闭式解**（closed-form solution）求解岭分类器（ridge classifier），无需反向传播。
- 引入**原型伪标签机制**（prototype-based pseudo-labeling）以应对标签稀缺问题。

---

### 相比现有方法的优势
| 维度 | FedRAN优势 |
|------|------------|
| **通信效率** | 客户端仅上传低秩SVD摘要 `(V, σ)` 和标签-特征统计 `B`，通信复杂度从 $O(M^2)$ 降至 $O(Mr)$，其中 $M$ 是随机特征维度，$r \ll M$ 是保留秩。 |
| **计算效率** | 客户端只需前向推理和一次SVD；服务器执行轻量级QR-SVD合并与对角矩阵求逆，平均比梯度法快 **190.3×**。 |
| **稳定性** | 冻结骨干网络（backbone），避免因客户端异构更新导致的**特征漂移**（feature drift）。 |
| **准确性** | 利用高维随机特征提升可分性，并保留主导Gram方向，精度优于主流基线最多达 **+4.8个百分点**。 |
| **标签效率** | 支持半监督变体 **FedRAN-SSL**，利用未标记样本生成伪标签，即使只有 **20%标签**也能显著提效（最高+6.61点）。 |

---

## 2. 核心实验方法和设置

### 数据集
在三个广泛使用的视觉基准上进行评估：

| 数据集 | 类型 | 总类数 | 任务数 | 特点 |
|-------|------|--------|--------|------|
| **CIFAR-100** | 图像分类 | 100 | 5 / 10 | 标准增量学习基准 |
| **ImageNet-R** | OOD泛化 | 200 | 10 | 包含艺术、雕塑等重分布图像，测试鲁棒性 |
| **VTAB** | 多样视觉域 | 50 | 5 | 跨自然、医学、遥感等领域，评估迁移能力 |

---

### 实验设置
- **客户端数量**：$K=5$
- **非IID划分**：使用 **Dirichlet分布**（浓度参数 $\beta \in \{0.1, 0.5, 1\}$）控制类别偏斜程度，$\beta$ 越小越不均衡。
- **骨干网络**：两种预训练模型
  - **ResNet-18**
  - **ViT-B/16**
- **随机投影维度**：
  - ResNet 设置：$M=8192$, 秩 $r=2048$（CIFAR/ImageNet-R）、$r=512$（VTAB）
  - ViT 设置：$M=2048$, $r=512$

---

### 评估指标
#### 准确性
- **最终准确率** $A_T$：所有任务完成后在全部已见类上的准确率。
- **平均准确率** $A_{avg}$：各任务结束时准确率的均值，衡量持续学习稳定性。

#### 资源效率
- **通信成本**：每客户端每任务最大上传字节数（MB）。
- **运行时间**：墙钟时间（秒），包含客户端与服务器总耗时。
- **特征漂移**（Feature Drift）：衡量固定参考集表示变化的程度。

---

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **优化型FCL** | Finetune, FedLwF, FedEWC, FediCaRL, TARGET |
| **提示/适配器型FCL**（基于预训练模型） | DualPrompt, CodaPrompt, Fed-CPrompt, PiLoRA |
| **分析型FCL** | STSA（Spatial-Temporal Statistics Aggregation） |

> FedRAN 主要与最强基线 **STSA** 对比，因其同属“无训练”分析范式。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3 & 5）

| 指标 | FedRAN vs 最强基线（STSA） |
|------|----------------------------|
| **平均准确率提升** | 最高达 **+4.8 pp**（CIFAR-100）<br>+4.24 pp（ImageNet-R）<br>+2.68 pp（VTAB） |
| **通信开销降低** | **30.6–121.8× 更少** 每客户端通信量<br>例如：VTAB 上仅需 **35.13 MB** vs TARGET 的 **4276.96 MB** |
| **训练速度提升** | 平均 **190.3× 更快**<br>CIFAR-100: **96.9×**, ImageNet-R: **246.9×**, VTAB: **227.2×** |
| **伪标签增益**（20%标签） | 最高提升 **+6.61 pp**（ImageNet-R）<br>CIFAR-100: +3.26 pp, VTAB: +5.76 pp |

> ✅ FedRAN 在所有数据集和 $\beta$ 设置下均取得最佳 $A_{avg}$ 和 $A_T$。

---

### 与基线方法的详细对比

#### 通信效率（Table 4）
| 方法 | CIFAR-100 (MB) | ImageNet-R (MB) | VTAB (MB) |
|------|----------------|------------------|-----------|
| TARGET（代表优化型） | 4283.82 (**31.9×**) | 4306.32 (**30.65×**) | 4276.96 (**121.75×**) |
| STSA | 194.59 (1.45×) | 389.18 (2.77×) | 97.29 (2.77×) |
| **FedRAN** | **134.27** | **140.52** | **35.13** |

> FedRAN 不仅远超优化型方法，也优于分析型基线 STSA。

#### 运行时间（Table 5）
| 方法 | CIFAR-100 (s) | ImageNet-R (s) | VTAB (s) |
|------|---------------|----------------|----------|
| Finetune | ~320 | ~580 | ~230 |
| STSA | 1.75 | 2.13 | 1.01 |
| **FedRAN** | **3.54** | **2.48** | **0.98** |

> 尽管略慢于 STSA（因额外SVD操作），但仍为**秒级响应**，适合边缘部署。

---

### 消融实验结果

#### （1）组件消融（Figure 7）
在 CIFAR-100 + ViT 上逐步添加组件：
- 原始 ViT 特征：87.71%
- + 随机投影（Random Projection）：→ 90.21%
- + ReLU 非线性：→ 93.95%
- + 低秩 SVD 摘要：→ **93.96%**（无精度损失）

> 表明 **低秩压缩几乎无损**，成功将通信从 $O(M^2)$ 降为 $O(Mr)$。

#### （2）投影维度 $M$ 与秩 $r$ 影响（Figure 6）
- 提升 $r$ 或 $M$ 可提高准确率，但存在饱和效应。
- 当 $M=8192$, $r=1024$ 时已达较优平衡，继续增加收益有限而通信翻倍。

> 支持“适度秩即可捕获主要信息”的设计哲学。

#### （3）不同骨干模型表现（Table 6）
在 ViT 设置下：
- FedRAN 在 CIFAR-100 上比 STSA 高 **+1.03 pp**
- 在 ImageNet-R 上高 **+2.40 pp**
- 同时通信更低（如 ImageNet-R: 11.13 MB vs 21.00 MB）

> 显示 FedRAN 在现代架构上更具优势。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **解析式方法可以同时实现高性能与高效率**：FedRAN 证明了无需反复训练也能达到甚至超越优化型FCL的精度。
2. ✅ **低秩Gram摘要优于一阶估计**：相比仅用类均值估算Gram矩阵的方法（如STSA-E），直接传输SVD摘要能更精确保留主导特征方向。
3. ✅ **冻结骨干+解析分类器是稳定高效的组合**：有效缓解了non-IID带来的特征漂移问题。
4. ✅ **伪标签可在低标签率下大幅提升性能**：FedRAN-SSL 在仅20%标签时仍能接近全监督性能。
5. ✅ **两层OR-SVD合并机制支持时空联合建模**：实现了真正的联邦+持续双重聚合。

---

### 方法的局限性
- **依赖预训练骨干质量**：性能高度依赖冻结的 `f(·)` 的表达能力。
- **随机投影引入额外超参**：$M$ 和 $r$ 需调优，影响通信-精度权衡。
- **理论边界为确定性而非概率性**：当前误差界基于最坏情况分析，实际中可能过于保守。
- **尚未集成差分隐私或安全聚合**：虽天然适合加密（因仅传统计量），但文中未实现。

---

### 未来工作方向
1. **隐私增强版本**：结合 Secure Aggregation 或同态加密保护上传的统计量。
2. **扩展至其他任务类型**：如检测、分割、语言任务等非分类场景。
3. **动态秩调整机制**：根据任务难度自适应选择 $r$，进一步优化资源利用率。
4. **纳入伪标签噪声建模**：理论分析伪标签错误如何影响最终性能。
5. **探索更高效子空间合并算法**：如分布式SVD、草图法（sketching）等。

---

> 🔚 **总结一句话**：  
> **FedRAN 通过“冻结骨干 + 随机特征 + 低秩Gram摘要 + 闭式分类器”四步走策略，在保证高精度的同时实现了极致的通信与计算效率，为资源受限场景下的联邦持续学习提供了全新范式。**

</details>

---

### 9. [Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching](https://arxiv.org/abs/2606.11583)

**Authors**: Zhuoyi Peng, Hanlin Gu, Lixin Fan, Yi Yang  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11583v1  

#### Abstract
Text-attributed graphs (TAGs) underlie real-world applications such as citation networks, social media, and e-commerce. Few-shot graph learning on TAGs is hard: with only a handful of labels per class and the rest of the graph unannotated, neither GNNs nor LLMs can learn well on their own. GNNs read...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**文本属性图**（Text-attributed Graphs, TAGs）上的**少样本图学习**（few-shot graph learning）问题。在仅有极少量标签（如每类3个）的情况下，传统方法面临两大挑战：
- **GNNs** 依赖图拓扑，但在低度节点（cold nodes）上表现差，因邻居信息不足。
- **LLMs** 依赖文本语义，但在文本模糊或简短时难以准确分类。

现有方法普遍采用“**黄金教师范式**”（golden-teacher assumption），即固定一个模型（如GNN或LLM）作为权威教师，另一个作为学生进行单向监督。然而，在稀疏标注下，任一模型都不可靠，这种单向监督会将教师的盲区直接传递给学生，限制性能上限。

### 提出了什么新方法或新思路
作者提出 **LLM-GNN Co-Teaching**，一种**双向协同教学框架**，其核心思想是：
- **摒弃黄金教师假设**：不指定任何一方为固定教师，GNN 和 LLM 在多轮迭代中**互相教学、共同进化**。
- 引入 **Round-based Pseudo-Label Preference Optimization (RPL-PO)**：从训练轨迹中自监督地挖掘偏好信号。当一个节点在第 $t$ 轮被两模型预测矛盾，而在第 $t+1$ 轮达成一致时，LLM 在该节点上的两个预测构成一个偏好对（旧错 vs 新对），用于 DPO 训练。

### 相比现有方法的优势
- **无监督信号来源更丰富**：利用模型间动态一致性变化生成监督信号，无需人工标注、奖励模型或外部评判器。
- **错误互补性强**：GNN 和 LLM 具有本质不同的归纳偏置（结构 vs 语义），能有效纠正对方的弱点。
- **适用于极端少样本场景**：在 3-shot 甚至零样本跨域迁移中均表现出色。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在六个标准 TAG 数据集上进行，涵盖不同领域和规模：
- **学术引用网络**：`Cora`, `Citeseer`, `PubMed`, `ogbn-arxiv`
- **维基百科链接图**：`WikiCS`
- **电商共购图**：`ogbn-products`

### 实验设置和评估指标
- **任务**：少样本半监督节点分类（few-shot semi-supervised node classification）
- **设置**：每类随机选取 3/5/10 个标签作为训练集，其余未标注节点用于伪标签生成。
- **评估指标**：测试集上的分类准确率（Accuracy %），报告多次运行的均值与标准差。

### 基线方法对比
对比了三类主流方法：
1. **经典 GNN 模型**：`GCN`, `GAT`, `GraphSAGE`
2. **LLM-as-Predictor 方法**：`Zero-shot`, `Graph-CoT`, `Neighbor-Augmented Prompting`
3. **LLM-GNN 融合方法**：
   - LLM-as-Enhancer: `TAPE`, `GLEM`, `LLM-GNN`
   - LLM-as-Predictor: `LLaGA`, `GraphGPT`
   - GNN-as-Judge: `GNN-as-Judge`（最接近的竞争者）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **3-shot 设置下**，LLM-GNN Co-Teaching 在所有六大数据集上均达到 SOTA 性能：

| 数据集 | 准确率 (%) | 相对 GNN-as-Judge 提升 |
|--------|------------|------------------------|
| `Cora` | **85.75 ± 0.88** | **+7.86%** |
| `Citeseer` | **77.12 ± 0.97** | +3.53% |
| `PubMed` | **91.32 ± 1.56** | +4.20% |
| `WikiCS` | **74.80 ± 0.95** | +2.58% |
| `ogbn-arxiv` | **69.94 ± 0.84** | **+7.73%** |
| `ogbn-products` | **82.82 ± 1.02** | +1.80% |

> 平均绝对增益达 **+5.40%**，在最难的 `ogbn-arxiv` 上提升近 8%。

### 与基线方法的对比结果
- 显著优于所有单向监督方法（如 GNN-as-Judge、TAPE、GraphGPT 等）。
- 即使在 5-shot 和 10-shot 下仍保持领先。
- 在**零样本跨域迁移**（zero-shot cross-dataset transfer）任务中（如在 `ogbn-arxiv` 上训练，在 `Cora` 上测试），也显著优于现有方法，表明其提升了 LLM 的通用图推理能力。

### 消融实验结果
通过消融实验证明各组件的有效性（以 3-shot 为例）：

| 变体 | Cora (%) | 说明 |
|------|----------|------|
| 完整模型 | **85.75** | 基准 |
| 移除双向教学（单轮） | 78.66 | 性能大幅下降，证明多轮互教必要 |
| 移除 RPL-PO | 83.03 | 验证轨迹偏好优化带来额外增益 |
| 固定选择比例 R=0.5 | 83.20 | 动态退火策略更优 |
| 改用同意过滤（agreement selection） | 82.52 | 小损失准则比简单一致性更有效 |
| 移除邻居信息 | 85.08 | 结构上下文对 LLM 至关重要 |

---

## 4. 关键结论和发现

### 论文的主要发现
- ✅ **双向协同优于单向监督**：放弃“黄金教师”假设，让 GNN 和 LLM 互为师生，能更有效地融合结构与语义信息。
- ✅ **训练轨迹蕴含监督信号**：RPL-PO 成功从模型演化过程中提取出高质量的偏好对，实现完全自监督的优化。
- ✅ **互补性是成功关键**：GNN 和 LLM 的失败模式高度正交，使得它们能在各自强项上为对方提供可靠指导。
- ✅ **特别适合困难任务**：在类别多、图结构复杂的数据集（如 `ogbn-arxiv`）上提升最大，说明该方法能有效缓解小样本下的过拟合与歧义问题。

### 方法的局限性
- **时间开销较大**：由于多轮迭代和 LLM 微调，总训练时间高于单次方法（约 2.6× GNN-as-Judge），尽管可通过早停控制。
- **依赖强基础 LLM**：若使用较弱 LLM（如 Vicuna-7B 替代 Llama-3-8B），性能显著下降，说明 LLM 是性能瓶颈之一。
- **适用范围有限**：目前仅验证于文本描述性强的图数据（如论文、商品），在文本缺失或噪声大的图（如分子图、金融交易图）上效果未知。

### 未来工作方向
- 探索更高效的协同机制以降低计算成本。
- 将该框架扩展至其他模态（如图像属性图）或多任务场景。
- 研究如何在文本质量差或无文本的图上构建有效的协同信号。
- 分析并缓解 LLM 伪标签可能带来的偏见放大问题。

> **代码地址**：https://github.com/llmgnncoteaching/LLM-GNN-Coteaching

</details>

---

### 10. [Physics-Distilled Neural Network enabled by Large Language Models for Manufacturing Process-Property Predictive Modeling](https://arxiv.org/abs/2606.11605)

**Authors**: Ge Song, Kiarash Naghavi Khanghah, Anandkumar Patel, Rajiv Malhotra, Hongyi Xu  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11605v1  

#### Abstract
Predicting process-property relationships in manufacturing is often challenged by high experimental costs and the limited interpretability of complex 'black-box' models. This paper proposes a novel knowledge distillation framework designed to achieve high-accuracy predictions in data-scarce scenario...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Physics-Distilled Neural Network enabled by Large Language Models for Manufacturing Process-Property Predictive Modeling

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题  
现代先进制造中，**process-property 预测建模**面临两大挑战：
- **数据稀缺性**：高成本、耗时的实验导致训练数据极少；
- **模型可解释性差**：传统“黑箱”机器学习模型缺乏物理一致性，难以泛化到未见工况。

此外，现有 Physics-Informed Neural Networks（PINNs）依赖人工设计物理损失函数，限制了其在复杂、新兴制造工艺中的扩展性。

---

### 提出了什么新方法或新思路  
本文提出了一种名为 **Physics-Distilled Neural Network** 的新型教师-学生框架，结合 **Large Language Models (LLMs)** 与 **Knowledge Distillation**，实现高效、轻量且鲁棒的预测建模。其核心思想是：
- 利用 LLM 自动从科学文献中提取并迭代优化 **analytical physics priors**（物理先验方程）；
- 将这些先验嵌入一个具有特权输入（privileged information）的 **Privileged Teacher 模型**；
- 教师模型通过 **Graph-Masked Attention (GMA) 层** 结构性地编码物理依赖关系；
- 最终将教师学到的物理一致的潜在表示（latent representation）蒸馏至一个仅依赖标准工艺参数的轻量级 **Student Predictor**。

---

### 相比现有方法的优势  
| 维度 | 优势 |
|------|------|
| **自动化物理建模** | 无需专家手动推导物理方程，LLM-RAG 自动完成知识抽取与精炼 |
| **结构化物理约束** | GMA 层通过 adjacency matrix 强制注意力机制遵循物理变量间的耦合关系，优于仅在 loss 中加入物理项的传统 PINN |
| **小样本下高性能** | 在仅有 ~30 个训练样本的情况下仍保持高精度与强泛化能力 |
| **故障容忍性强** | 即使 LLM 提取的物理先验不完整或次优，模型仍能维持稳健性能 |
| **边缘部署友好** | 学生模型推理频率 >6000 Hz，适用于实时工业监控 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集  
实验覆盖 **五个差异显著的制造过程**，分为两类：

#### （1）静态参数基准（Benchmarks 1–3）
| Benchmark | Process | 输入参数 | 输出属性 |
|---------|--------|----------|-----------|
| 1 | FLIPMM | Pulse energy, Frequency, Scanning speed, Water speed | Heat-Affected Zone (HAZ) |
| 2 | MSLA | Layer thickness, Exposure time, Build orientation | Ultimate Tensile Strength (UTS) |
| 3 | TADCR | Rolling force, Ball diameter, Initial roughness, Passes | Hardness (HV) |

> 数据来源：基于已有 Response Surface Methodology (RSM) 方程生成的合成因子网格（factorial grid），模拟真实实验数据。

#### （2）多模态动态数据基准（Benchmarks 4–5）
| Benchmark | Process | 静态参数 + 时间序列信号（privileged during training） | 输出 |
|---------|--------|---------------------------------------------|-------|
| 4 | Injection Molding | Mold temp, Speed, Switchpoint + Viscometer pressure (bar) | Melt cushion volume (cm³) |
| 5 | MaRoReS Machining | Feed rate, Speed, Depth, Radius + Force, Acceleration, Sound, Current | Residual Stress (MPa) |

> 数据来源：真实实验数据集（各含 68 个实例），时间序列作为训练阶段可用的“特权信息”。

---

### 实验设置和评估指标  

#### 数据划分策略
- **Benchmarks 1–3**：采用 **分层外推划分（Hierarchical extrapolation split）**
  - 训练集：取每个输入维度中间 75% 区间内的组合（共 30 个随机采样）
  - 测试集：所有至少在一个维度上超出 75% 范围的点 → 严格测试外推能力
- **Benchmarks 4–5**：采用 **5-fold Cross Validation**，因原始数据量极小（n=68）

#### 重复验证增强统计可靠性
- 对每个基准进行多次独立运行（不同 seed），报告均值与分布，提升结果可信度。

#### 评估指标
| 指标 | 定义 | 含义 |
|-----|------|-------|
| $ R^2 $ | $ 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} $ | 解释方差比例，越高越好 |
| RMSE | $ \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2} $ | 平均预测误差，越低越好 |
| Inference Speed (Hz) | $ v = 1 / \overline{t_{\text{inference}}} $ | 推理频率，衡量边缘部署可行性 |

---

### 基线方法对比  
共比较三类主流方法：

| 类别 | 模型 | 描述 |
|------|------|------|
| **Physics-only** | Physics (finit / frefine) | LLM 提取的初始/精炼物理方程直接用于预测 |
| **Traditional Regression** | RF, XGBoost | 经典树模型，代表非神经网络方法 |
| **Traditional NN** | MLP, KD | 多层感知机；传统知识蒸馏（无物理约束） |
| **Physics-Incorporated** | PG-MLP | 加入物理 loss 的 MLP，类似 PINN 思路 |
| **Proposed** | **Physics-Distilled NN** | 本文提出的完整框架 |

> 所有模型均在相同稀疏训练集上训练，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 3 & Tables 4–8）

| Benchmark | Model | $ R^2 $ (finit) | $ R^2 $ (frefine) | RMSE (finit) | RMSE (frefine) | Speed (Hz) |
|----------|--------|------------------|--------------------|---------------|----------------|------------|
| 1 (FLIPMM) | Proposed | **0.949** | **0.957** | **2.410** | **2.204** | 6537 |
| 2 (MSLA) | Proposed | **0.824** | **0.826** | **2.516** | **2.505** | – |
| 3 (TADCR) | Proposed | **0.957** | **0.964** | **3.240** | **3.109** | – |
| 4 (IM) | Proposed | **0.963** | **0.989** | **0.169** | **0.099** | 6198 |
| 5 (MaRoReS) | Proposed | **0.612** | **0.725** | **0.498** | **0.347** | – |

> 注：加粗为各 benchmark 下最优表现。

---

### 与基线方法的对比结果

#### ✅ 在所有基准任务中全面领先
- 在 **Benchmark 1–3** 上，即使使用弱先验（finit），所提方法也显著超越纯物理模型和所有数据驱动模型（如 MLP、XGBoost）；
- 在 **Benchmark 4**（注塑成型）中，尽管传统 MLP 和 KD 表现尚可，但本文方法在 **稳定性（tight IQR）和泛化性**方面更优；
- 在 **Benchmark 5**（MaRoReS）这一高度非线性、小样本场景中，传统 MLP 几乎失效（$ R^2 \approx 0.425 $），而本文方法达到 **0.725**，远超其他模型。

#### 🔍 特别观察：当物理先验很差时仍有效
- 如在 MSLA（Benchmark 2）中，初始物理方程 $ R^2 = 0.622 $，非常弱；
- 但所提框架仍能将其“提炼”为 $ R^2 = 0.824 $ 的学生模型，说明具备**从劣质先验中学习有用结构的能力**。

---

### 消融实验分析（隐含于对比中）

| 比较对象 | 差异点 | 发现 |
|--------|--------|------|
| vs KD (Knowledge Distillation only) | 是否包含 GMA 层与物理 loss | 本文方法性能更高 → **GMA 提供更强归纳偏置** |
| vs PG-MLP | 是否使用特权信息与蒸馏机制 | 本文方法更稳定 → **蒸馏保留了动态特征中的深层物理逻辑** |
| vs Physics-only | 是否融合数据驱动学习 | 本文方法弥补了方程残差误差 → **不是替代而是增强物理模型** |

> 图表显示，本文方法在所有交叉验证 fold 中 **interquartile range 更窄**，表明模型更稳定、抗过拟合能力强。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 可作为自动物理建模引擎**：通过 RAG 从文献中提取并迭代优化 analytical priors 是可行且有效的，大幅降低对领域专家的依赖。
2. ✅ **GMA 层提供结构性物理约束**：相比仅在 loss 中添加物理项，将物理导数信息编码进 attention mask 能更有效地防止模型学习非物理解。
3. ✅ **知识蒸馏成功传递“物理思维”**：学生模型虽只看静态参数，却能模仿教师对物理机制的理解，实现高质量外推预测。
4. ✅ **框架具有强 fault tolerance**：即使 LLM 抽取出的物理方程质量较差，整体系统仍能保持较高预测性能。
5. ✅ **支持高速边缘部署**：学生模型推理速度 >6000 Hz，满足实时工业控制需求。

---

### 方法的局限性
1. 🛑 当前输出为 **scalar quantity**（如硬度、应力），尚未扩展至空间场输出（如全场残余应力分布图）；
2. 🛑 模型输出为 **point prediction**，缺乏不确定性量化（uncertainty quantification），不利于风险敏感的应用（如闭环控制）；
3. 🛑 LLM 提取的方程受限于文献质量和覆盖范围，若某工艺研究稀少，则可能无法获得有效先验。

---

### 未来工作方向
1. **扩展至场输出建模**：将框架应用于预测完整的 residual stress field 或 distortion map，服务于 design-for-manufacturing；
2. **引入不确定性估计**：集成贝叶斯神经网络或 conformal prediction，提升决策安全性；
3. **跨工艺迁移学习**：探索如何将在某一工艺中学到的物理结构迁移到相似但数据更少的新工艺中；
4. **闭环控制系统集成**：将该轻量高准模型嵌入 CNC 控制器或 PLC，实现实时自适应工艺调整。

--- 

> **总结一句话**：  
> 本工作构建了一个 **“理论物理 ↔ 工业实践”之间的桥梁**——利用 LLM 自动读取文献、教师模型吸收物理规律与动态数据、学生模型实现高速精准推理，为数据稀缺下的智能制造提供了可扩展、可部署、可信赖的建模范式。

</details>

---

### 11. [ICA Lens: Interpreting Language Models Without Training Another Dictionary](https://arxiv.org/abs/2606.11722)

**Authors**: Sida Liu, Feijiang Han  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11722v1  

#### Abstract
Finding interpretable directions in language-model representations is critical for understanding and controlling model behavior. Sparse autoencoders (SAEs) have become the standard tool for this purpose, but using them as the default first lens often requires training, storing, and evaluating large ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ICA Lens: Interpreting Language Models Without Training Another Dictionary

## 1. 论文的主要贡献和创新点

### 解决的问题
当前对大型语言模型（LLM）进行 **Mechanistic Interpretability**（机制可解释性）分析时，主流方法是使用 **Sparse Autoencoders (SAEs)** 来学习一个过完备字典（overcomplete dictionary），从而将模型激活分解为稀疏的、人类可解释的特征。然而，这种方法存在显著的实践瓶颈：
- 需要大量的计算资源、存储空间和训练时间来为每个模型、每一层、每种激活位置和稀疏度设置训练 SAE。
- 这使得快速探索和迭代变得困难，限制了其作为“第一视角”（first lens）的实用性。

### 提出的新方法与新思路
论文提出 **ICA Lens (ICALens)**，一种无需训练另一个神经网络字典即可解释 LLM 表征的新范式。其核心思想是：
- **重新审视 ICA**：利用经典的 **Independent Component Analysis (ICA)** 作为寻找非高斯（non-Gaussian）方向的工具，因为这些方向在统计上是“异常”的，更可能对应于有意义的语义或句法模式。
- **直觉基础**：许多可解释的方向具有选择性（selective），即它们只在特定上下文中被激活，这会导致其激活分布呈现重尾（heavy-tailed）或高尖峰（high-kurtosis）的非高斯特性。
- **核心假设**：通过直接搜索非高斯方向，可以在不经过昂贵的 SAE 字典训练的情况下，揭示出大量已存在于激活几何中的可解释结构。

### 相比现有方法的优势
- **高效且低成本**：ICALens 不需要梯度训练，避免了 SAE 的巨大计算开销，可以快速部署到新模型或新层上。
- **稳定且实用**：论文提出了针对 LLM 激活特性的稳定性配方（recipes），解决了标准 ICA 在 LLM 上不稳定的问题。
- **互补而非替代**：ICALens 被定位为 SAE 的**轻量级、互补的第一视角**，用于快速识别有潜力的层和概念，指导后续是否值得投入成本进行 SAE 分析。

---

## 2. 核心实验方法和设置

### 数据集
- **激活语料库 (Activation Corpus)**：从以下三个模型中收集残差流（residual-stream）激活：
  - **GPT-2 Small**
  - **Gemma 2 2B**
  - **Qwen 3.5 2B Base**
- 每个模型使用 **Pile-10k** 训练集的随机样本，截断至 1024 上下文长度，并均匀采样 100 万个 token 位置的激活向量用于 ICA 训练。

### 实验设置
- **方法实现**：基于 PyTorch 实现 **GPU 并行的 FastICA** 算法，以提高效率。
- **稳定性配方 (Stability Recipes)**：
  1. **Row-Normalization**：在白化前对每个激活向量进行 L2 归一化，减少大范数 token 的影响。
  2. **Robust Convergence Acceptance (p95-LIM)**：采用更鲁棒的收敛判断标准，当 95% 的组件收敛时即接受该层，而非要求所有组件都收敛。
  3. **Adaptive Refit**：对于难以收敛的层，自适应地降低目标组件数量，确保能获得最高分辨率的有效结果。
- **评估对象**：在每个模型的嵌入层和所有残差流层独立拟合 ICA。

### 评估指标
- **非高斯性 (Non-Gaussianity)**：使用**超额峰度 (excess kurtosis)** 量化投影分布的非高斯程度。
- **有效感受野 (Effective Receptive Field, ERF)**：衡量一个组件的激活需要多长的左上下文才能被恢复，用以分析组件的上下文依赖性。
- **人类可解释性 (Human Interpretability)**：
  - 人工标注协议（annotation protocol）。
  - 随机组件审计（random component audit）。
  - 二级专家对比验证（secondary expert audit）。
- **下游任务性能**：
  - **Sparse Probing**：测试 ICA 组件是否浓缩了概念相关信息。
  - **Targeted Probe Perturbation (TPP)**：测试 ICA 组件是否支持选择性干预。

### 基线方法对比
- **Sparse Autoencoders (SAEs)**：使用公开的 SAE 检查点（如 Gemma Scope, Qwen Scope）作为主要高性能基线。
- **Matryoshka SAEs**：使用较小字典尺寸的 SAE 变体，用于在紧凑预算下进行公平比较。
- **ITDA (Inference-Time Decomposition of Activations)**：另一种无需训练的轻量级替代方案。
- **PCA (Principal Component Analysis)**：作为经典线性降维方法的基线，用于对比“非高斯性” vs “方差”哪个是更好的可解释性信号。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1. **非高斯性显著更高**：
   - 如图3所示，**ICA 组件的超额峰度远高于随机方向和公开 SAE 的解码器方向**，验证了 ICA 成功找到了统计上最异常的方向。

2. **人类可解释性高**：
   - **随机审计**：在 150 个随机抽样的 ICA 组件中，**127 个获得了高置信度标签**，表明大多数组件是可解释的。
   - **二级审计**：127 个高置信度标签中有 **121 个得到完全支持，112 个得分 ≥ 8/10**，证明了标注的可靠性。

3. **下游任务性能优异**：
   - **Sparse Probing**（图10, 11）：
     - ICA 在所有三个模型上均**与公开 SAE 性能相当**。
     - 显著优于 **ITDA** 和 **PCA**。
     - 即使是小型 **Matryoshka SAE** 也未能超越 ICA。
   - **Targeted Probe Perturbation (TPP)**（图12）：
     - 在**小到中等规模的干预预算 (small-to-medium intervention budgets)** 下，**ICA 的表现优于公开 SAE**。
     - 这表明少量 ICA 组件就能有效实现选择性干预，使其成为轻量级编辑的理想起点。

### 消融实验结果
- **稳定性配方的效果**（表1）：
  - 仅通过引入 **Row-Normalization** 和 **p95-LIM**，在 GPT-2 Small 上，成功收敛的层数从 2 层提升到了 10 层，总迭代次数减少了 21.5%。
  - 这证明了所提出的稳定性配方对提升 ICA 在 LLM 上的实用性至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **ICA 被严重低估**：由于早期实现不稳定和缺乏系统性评估，ICA 在 LLM 可解释性领域未受到足够重视。本研究证明，经过适当优化后，ICA 是一个强大且实用的工具。
2. **非高斯性是强大的可解释性信号**：直接最大化非高斯性（而非稀疏重建）是一种高效的方法，能够揭示出大量可解释的结构。
3. **ICA 与 SAE 互补**：
   - **方向重叠**：许多 ICA 组件与 SAE 特征有较高的余弦相似性，说明两者发现了相关结构。
   - **独特价值**：ICA 也能发现一些 SAE 字典中没有明确捕捉到的组件。
   - **激活模式差异**：如图15所示，**SAE 特征倾向于局部峰值**，而 **ICA 组件往往在相关 token 序列上平滑变化**，更像是一种“上下文状态”。
4. **ICA 是高效的“第一视角”**：它提供了一种低成本、快速的方法来探索 LLM 内部表示，帮助研究人员决定在何处投入资源进行更精细的 SAE 分析。

### 方法的局限性
- **紧凑性 (Compactness)**：标准 FastICA 最多只能返回 `d` 个组件（`d` 为隐藏维度），无法像 SAE 那样生成过完备的大型字典。
- **算法边界**：目前的方法是紧凑的，无法提供 SAE 所具有的高分辨率特征发现能力。

### 未来工作方向
- 探索更高容量的 ICA 变体，如 **Overcomplete ICA** 或 **Deflationary ICA**，以突破组件数量的限制。
- 将 ICA 应用于分析模型的**变换过程**（如 MLP 输出、注意力输出、残差更新），而不仅仅是状态。
- 开发基于 ICA 的自动标注系统和可控的 steering 应用。

</details>

---

### 12. [Efficient Time Series Clustering from Multiscale Reservoir Dynamics with Granular-Ball Anchoring Graph Optimization](https://arxiv.org/abs/2606.12077)

**Authors**: Yifan Wang, Lifeng Shen, Shuyin Xia, Yi Wang  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.12077v1  

#### Abstract
Time-series clustering remains challenging due to the inherent trade-off between clustering effectiveness and computational efficiency. Similarity-based methods often suffer from quadratic complexity caused by pairwise distance computations, while deep learning-based approaches typically rely on cos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient Time Series Clustering from Multiscale Reservoir Dynamics with Granular-Ball Anchoring Graph Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
时间序列聚类面临两大核心挑战：
- **计算效率低**：基于相似性的方法（如 DTW）因成对距离计算导致 $O(N^2)$ 复杂度，难以扩展到大规模数据。
- **表达能力与训练成本的权衡**：深度学习方法虽能学习复杂动态特征，但依赖迭代训练、大量参数和超参调优，计算开销大。

如何在保证高聚类性能的同时实现高效、可扩展的时间序列聚类，是当前研究的关键瓶颈。

### 提出的新方法：MSRGC-Net
本文提出 **MSRGC-Net**（Multi-Scale Reservoir Granular-ball Consensus Network），一个无需训练、高效的端到端时间序列聚类框架，融合三大核心技术：

1. **Multi-Scale Reservoir Encoding（多尺度储层编码）**
   - 利用多个具有不同谱半径（spectral radius）的固定 ESN（Echo State Network）构建多尺度 reservoir views。
   - 不同谱半径捕获互补的时间动态：小值偏好短期变化，大值接近“混沌边缘”，保留长期依赖。
   - 完全避免反向传播和参数更新，实现 **training-free** 表示学习。

2. **Granular-Ball Anchoring Graph Construction（粒球锚定图构建）**
   - 引入 Granular-Ball Computing（GBC）进行区域级抽象，将样本映射为密度一致的局部区域（granular balls）。
   - 选取高质量的 granular-ball 中心作为 **anchors**，构建 sample-to-anchor 的锚定图（anchoring graph），显著降低内存和计算复杂度。

3. **Consensus-Based Anchoring Graph Optimization（基于共识的锚定图优化）**
   - 设计统一的目标函数，通过加权聚合多视图锚定图，学习一个全局共识图（consensus graph）。
   - 融合多尺度表示中的互补信息，并通过自适应权重机制强调更一致的视图。

### 相比现有方法的优势
| 维度 | MSRGC-Net | 传统方法 |
|------|-----------|--------|
| **训练方式** | Training-free（无反向传播） | 需要迭代训练（如 AE、DEC） |
| **复杂度** | 近线性 $O(N)$ | 二次型 $O(N^2)$（如 DTW）或高参数量模型 |
| **可扩展性** | 支持百万级样本 | 多数方法在大数据上不可行 |
| **鲁棒性** | 区域级建模抑制噪声影响 | 点对点建模易受噪声干扰 |

---

## 2. 核心实验方法和设置

### 数据集
在 **UCR 和 UEA 归档** 的 10 个基准数据集上进行评估，涵盖多种应用场景：
- **多变量数据集（Multivariate）**：CharacterTrajectories (CT), JapaneseVowels (JV), BasicMotions (BM), Cricket (Cric), SelfRegulationSCP1 (SCP1)
- **单变量数据集（Univariate）**：BCCrop, EPG-R, EPG-S, Wafer 等
- 数据长度从几十到上千不等，类别数从 2 到 20 不等。

### 实验设置
- **Reservoir 参数**：固定大小 $R=400$，连接率 $\beta=0.25$，输入缩放 $w=0.15$，噪声水平 $8=0.001$
- **视图数量**：$V=3$ 个不同谱半径的 reservoir
- **锚点数量**：$m \ll N$，控制粒度与效率平衡
- **优化参数**：$\gamma = 10^3$, $\lambda = 10^{-1}$，通过网格搜索微调
- **硬件平台**：Intel i7-12700 CPU + 64GB RAM
- **重复次数**：所有实验重复 10 次取平均

### 评估指标
采用三个标准聚类评价指标：
- **NMI**（Normalized Mutual Information）
- **ARI**（Adjusted Rand Index）
- **RI**（Rand Index）

### 基线方法对比
分为三类共 11 种代表性方法：
1. **原始数据方法**：k-Shape, Fuzzy-kShape, TCK
2. **表示学习方法**：Modular-RC, GRAIL, Time2Feat, DEC, TimeSURL, TFMCC
3. **多视图聚类方法**：GB-SMKKM, MV-CAGAF

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以多变量数据集为例，见 Table 1）

| Dataset | MSRGC-Net (NMI / ARI / RI) | 最佳基线 (NMI / ARI / RI) | 提升幅度（ARI） |
|--------|----------------------------|--------------------------|----------------|
| CT     | **0.789 / 0.645 / 0.964**   | GRAIL: 0.742 / 0.608 / 0.961 | +6.1%         |
| JV     | **0.781 / 0.750 / 0.948**   | GRAIL: 0.703 / 0.581 / 0.917 | **+16.9%**    |
| BM     | **0.678 / 0.615 / 0.857**   | Time2Feat: 0.654 / 0.600 / 0.873 | +2.5%         |
| Cric   | **0.929 / 0.884 / 0.983**   | Time2Feat: 0.921 / 0.822 / 0.966 | +7.6%         |
| SCP1   | **0.225 / 0.270 / 0.635**   | GB-SMKKM: 0.207 / 0.263 / 0.631 | +2.7%         |

- 在 **15 项评估中，MSRGC-Net 获得 12 项第一，2 项第二**，表现稳定且全面领先。
- 尤其在高维、长序列任务（如 JV, Cric）上优势明显，说明其对复杂动态建模能力强。

### 与基线方法的对比结果
- **相比 raw-data 方法（如 k-Shape）**：大幅超越，尤其在 ARI 和 NMI 上提升显著（例如 JV 上 ARI 从 ~0.1 提升至 0.75）。
- **相比 deep learning 方法（如 DEC, TimeSURL）**：在无需训练的前提下达到甚至超过其性能。
- **相比 multi-view 方法（如 GB-SMKKM）**：性能更优且运行速度快一个数量级。

### 消融实验结果（Ablation Study，见 Table 2）
使用 RI 指标验证各组件贡献：

| 变体 | 是否含 Multi-Scale | 是否含 Granular-Ball | 是否含 Consensus Opt. | 平均 RI（多变量） |
|------|--------------------|-----------------------|------------------------|------------------|
| w/o Multi-scale | ✗ | ✓ | ✓ | 0.844 |
| w/o Granular-ball | ✓ | ✗ | ✓ | 0.832 |
| w/o Optimization | ✓ | ✓ | ✗ | 0.811 |
| **MSRGC-Net（完整）** | ✓ | ✓ | ✓ | **0.883** |

- 所有模块均有正向贡献，其中 **multi-scale reservoir 编码贡献最大**。
- Granular-ball 锚定有效提升结构感知能力。
- Consensus optimization 实现多视图信息的有效融合。

---

## 4. 关键结论和发现

### 主要发现
1. **Training-free 多尺度表示可行且高效**  
   利用 reservoir computing 的内在动力学特性，无需训练即可提取多样化的时间表征，打破了“高性能必须高成本”的固有认知。

2. **Region-level 抽象优于 Point-wise 建模**  
   Granular-ball 通过密度一致区域建模，提升了对噪声的鲁棒性和结构捕捉能力，特别适合 reservoir 输出的空间分布特性。

3. **Consensus graph 优化实现轻量融合**  
   通过自适应加权整合多尺度视图，在保持低复杂度的同时增强了聚类一致性。

4. **效率与效果的帕累托前沿突破**  
   如 Figure 4 所示，MSRGC-Net 位于 **Pareto frontier 上方左侧**，兼具高准确率与低运行时间（数十秒内完成）。

5. **卓越的可扩展性**  
   在包含 **180万样本** 的 Pedestrian 数据集上测试（Figure 5），runtime 呈近线性增长，RI 达到 0.947，远超 k-Shape（0.882），验证了其工业级部署潜力。

### 方法的局限性
- **Reservoir 设计依赖先验配置**：虽然无需训练，但 reservoir 大小、谱半径策略仍需经验设定。
- **对极短序列可能欠拟合**：reservoir dynamics 在非常短的时间序列上可能无法充分展开。
- **目前仅适用于等长序列**：未直接处理变长输入（尽管可通过 padding 或 pooling 缓解）。

### 未来工作方向
- 探索 **adaptive reservoir 构造机制**，根据数据自动调整拓扑结构。
- 扩展至 **streaming time series clustering** 场景，支持在线更新。
- 结合 **causal discovery** 或 **interpretability tools**，增强聚类结果的可解释性。
- 应用于更多领域，如 **financial time series segmentation**, **IoT sensor fault detection** 等。

--- 

> ✅ 总结一句话：  
> **MSRGC-Net 成功实现了“高效”与“有效”的统一——它以 training-free 方式生成多尺度动态表示，借助 granular-ball 锚定图与 consensus 优化，在多个 benchmark 上全面超越 SOTA 方法，同时具备百万级可扩展能力，为大规模时间序列聚类提供了新的范式。**

</details>

---

### 13. [From Verdict to Process: Agentic Reinforcement Learning for Multi-Stage Fact Verification](https://arxiv.org/abs/2606.13262)

**Authors**: Rongxin Yang, Shenghong He, Siyuan Zhu, Chao Yu  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.13262v1  

#### Abstract
Recent approaches combining Large Language Models (LLMs) with retrieval-augmented reasoning have shown promise for automated fact verification. To process complex claims, these verification pipelines typically execute multi-stage workflows that coordinate tightly coupled modules, including claim dec...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Verdict to Process: Agentic Reinforcement Learning for Multi-Stage Fact Verification

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的多阶段事实验证系统（如 InFact、HerO）通常将**claim decomposition**、**evidence retrieval**、**answer generation** 和 **verdict prediction** 分为独立模块，并分别优化或依赖固定启发式规则进行协调。这种分离导致各阶段决策之间缺乏适应性协调，中间行为（如问题生成）可能无法有效支持最终的判决预测，从而影响整体验证效果。

此外，传统方法仅依赖最终的真值标签（final veracity label）作为监督信号，该信号稀疏且延迟，难以准确归因于各个阶段的贡献（credit assignment problem），限制了端到端优化能力。

### 提出的新方法：ProFact
作者提出 **ProFact** —— 一种基于 **Agentic Reinforcement Learning (RL)** 的框架，用于对多阶段事实验证轨迹进行**端到端优化**。

#### 核心创新点：
- **统一策略建模（Unified Policy Modeling）**  
  将整个验证流程建模为一个长周期的 **Markov Decision Process (MDP)**，由单一策略网络（policy）控制所有阶段的行为：生成验证问题 → 调用检索工具获取证据 → 生成基于证据的答案 → 预测最终 verdict。
  
- **过程感知奖励机制（Process-Aware Reward）**  
  引入分阶段的密集奖励函数，在每个阶段提供学习信号：
  - **QUESTION 阶段**：使用 METEOR 指标衡量生成的问题与黄金问题集的匹配度。
  - **SEARCH 阶段**：评估生成的 question-answer 对与标注过程的一致性。
  - **VERDICT 阶段**：判断最终预测标签是否正确。
  > 这种设计将稀疏的结果监督转化为**阶段级的密集反馈**，显著改善信用分配。

- **轨迹级优化（Trajectory-Level Optimization）**  
  使用 **Group-Relative Policy Optimization (GRPO)** 对完整验证路径进行优化，鼓励模型探索更有效的分解与检索策略，实现跨阶段协同。

### 相比现有方法的优势
| 方面 | 传统方法（如 InFact, HerO） | ProFact |
|------|----------------------------|--------|
| 架构 | 多个独立模块拼接 | 单一统一策略 |
| 优化方式 | 各阶段孤立训练或提示工程 | 端到端 RL 联合优化 |
| 监督信号 | 仅最终 verdict 标签（稀疏） | 分阶段 process-aware 奖励（密集） |
| 协调机制 | 固定流程或启发式 | 自适应学习最优路径 |
| 推理效率 | 步骤冗余较多 | 更少 token 消耗与更快推理 |

---

## 2. 核心实验方法和设置

### 数据集
- **AVeriTeC [13]**：面向真实世界声明的事实验证基准数据集。
  - 包含自然语言声明、真值标签（`SUPPORTED`, `REFUTED`, `NOT ENOUGH EVIDENCE`, `CONFLICTING EVIDENCE`）
  - 提供人工标注的验证问题-答案对（gold QA pairs）
  - 内置静态知识库（pre-collected web documents）作为检索源

### 实验设置
- **Backbone 模型**：在四个开源 LLM 上进行后训练（post-training）：
  - Qwen2.5-3B-Instruct
  - Qwen2.5-7B-Instruct
  - Qwen3-4B-Instruct-2507
  - Qwen3-8B-Instruct
- **训练细节**：
  - 使用 GRPO 算法进行强化学习训练
  - 每个 claim 采样 8 条轨迹构成 group
  - Mini-batch size: 32, Micro-batch size: 4
  - KL 正则系数：0.001
  - 最大交互步数：12 步/episode
- **推理设置**：
  - 温度设为 0（deterministic decoding）
  - 最多生成 5 个验证问题
  - 每次检索返回 top-3 证据项

### 评估指标
| 指标 | 描述 |
|------|------|
| **Q-only METEOR** | 衡量生成的验证问题与黄金问题之间的文本相似性，反映 claim decomposition 能力 |
| **Q&A METEOR** | 衡量生成的 question-answer 对与标注过程的整体一致性 |
| **Accuracy** | 最终 verdict 分类准确率 |
| **AVeriTeC Score** | 官方综合指标：要求证据质量得分 > 0.25 **且** verdict 正确，联合评估证据充分性与决策准确性 |

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Consistency** | Prompting-based | 使用固定黄金 QA 作为输入，通过一致性聚合预测 verdict |
| **InFact [12]** | Workflow-based | 六阶段流水线系统，模块化设计，表现强但非端到端优化 |
| **HerO [21]** | Strong baseline | 利用假设性文档增强检索，fine-tuned LLM 做判决，性能领先 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | Backbone | Q-only METEOR | Q&A METEOR | Accuracy | AVeriTeC Score |
|------|----------|----------------|-------------|-----------|----------------|
| **ProFact (ours)** | Qwen2.5-3B | **46.01** | **31.14** | **68.80** | **47.80** |
| **ProFact (ours)** | Qwen2.5-7B | 45.26 | 30.36 | **70.20** | **48.00** |
| **ProFact (ours)** | Qwen3-4B | **46.08** | 30.11 | 69.60 | 46.20 |
| **ProFact (ours)** | Qwen3-8B | 46.05 | 30.02 | 70.28 | 46.40 |

> ✅ 在所有 backbone 上，ProFact 在 **Accuracy** 和 **AVeriTeC Score** 上均优于最强基线（HerO）

### 与基线方法对比结果
- **相比 HerO**：
  - 平均提升约 **+1.4~5.4 pts** 在 AVeriTeC Score 上
  - 显著更高的 **Q-only METEOR**，说明其 claim decomposition 更优
- **相比 InFact**：
  - 在较小模型上优势更大（如 Qwen2.5-3B 上 AVeriTeC Score 提升超 39 pts）
  - 即使在大模型上也保持稳定领先

### 消融实验（Ablation Study: w/o PR）
移除中间阶段的过程奖励（process rewards），仅保留最终 verdict 奖励：

| 方法 | Backbone | Q-only METEOR | Q&A METEOR | Accuracy | AVeriTeC Score |
|------|----------|----------------|-------------|-----------|----------------|
| **w/o PR** | Qwen2.5-3B | 38.98 ↓ | 27.20 ↓ | 64.60 ↓ | 34.40 ↓ |
| **w/o PR** | Qwen2.5-7B | 33.79 ↓ | 25.67 ↓ | 62.80 ↓ | 31.00 ↓ |

> ❗ 移除 process-aware reward 导致全面性能下降，尤其在中间质量和综合评分上，证明其对 credit assignment 至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **端到端轨迹优化优于模块化设计**  
   ProFact 通过统一策略联合优化多阶段行为，实现了更好的跨阶段协调，避免了“局部最优但全局次优”的问题。

2. **过程感知奖励至关重要**  
   仅靠最终 verdict 标签不足以指导复杂决策链的学习；引入阶段级的密集奖励能显著提升中间推理质量和最终性能。

3. **更大的模型不一定更好**  
   实验显示某些 larger backbones（如 Qwen3-8B）反而表现不如中等规模模型，印证了 **inverse scaling** 现象：大模型更易依赖参数记忆和先验知识，忽视外部证据。

4. **ProFact 更高效**  
   如 Table 2 所示，ProFact 在推理时间与 token 消耗上远低于 InFact：
   - 平均每 claim 时间减少 **60–90%**
   - 输入输出 token 数量降低 **数倍**
   > 归因于流程简化（去重写、合并步骤）、上下文隔离机制及训练中学到的精简策略。

5. **GRPO 是更适合的 RL 算法**  
   比较 PPO、DAPO、GiGPO 后发现，**GRPO** 表现最佳，因其无需 critic network，直接基于组内相对优势计算梯度，更适合异构阶段与延迟奖励场景。

### 方法的局限性
- **依赖高质量标注过程数据**：需要黄金 QA 对来构建 process-aware reward，限制了在低资源领域的应用。
- **搜索动作空间有限**：当前仅支持简单 tool call，未建模复杂的查询重构或多跳检索策略。
- **固定检索源**：实验中使用预建索引的知识库，未考虑动态网页抓取或 API 调用延迟。

### 未来工作方向
- 扩展至多跳、递归式检索任务（multi-hop, recursive verification）
- 结合主动学习（active learning）减少对黄金过程标注的依赖
- 探索更细粒度的内部状态建模（如 anchor states）以进一步改进 credit assignment
- 应用于其他多阶段 agentic 任务（如法律分析、医疗诊断）

---

> 📌 **总结一句话**：  
> ProFact 成功将事实验证从“结果导向”转向“过程优化”，通过 **agentic RL + process-aware reward** 实现了更可靠、更高效的多阶段验证，为构建可解释、可优化的 LLM agent 提供了新范式。

</details>

---

### 14. [Using Explainability as a Training-Time Reliability Signal for Efficient ECG Classification](https://arxiv.org/abs/2606.12252)

**Authors**: Veerendhra Kumar Dangeti, Xiao Gu, Ying Weng, Shreyank N Gowda  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.12252v1  

#### Abstract
Training deep neural networks for clinical time-series analysis is computationally demanding, yet many healthcare settings lack the resources required for repeated model development and deployment. This challenge is particularly evident in electrocardiogram classification, where large datasets and l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Using Explainability as a Training-Time Reliability Signal for Efficient ECG Classification

---

## 1. 论文的主要贡献和创新点

### 解决的问题
深度神经网络在临床时间序列分析（如 ECG 分类）中的训练成本高昂，尤其在医疗资源有限的场景下难以反复进行模型开发与部署。现有的 **Progressive Data Dropout (PDD)** 虽然通过排除“已学会”的样本减少计算量，但其依赖 **model confidence** 作为筛选标准，容易保留因噪声或标签模糊导致困难的不可靠样本，而非真正具有学习价值的“硬样本”。

### 提出的新方法：ERTS
本文提出 **ERTS (Explainability-based Reliability Training Signal)**，一种基于可解释性的训练时可靠性信号机制，用于高效 ECG 分类。

- **核心思想**：利用 **Grad-CAM** 生成的注意力图（attention map）来衡量模型预测是否基于**连贯且局部化的生理形态特征**（即有意义的不确定性），从而区分“信息性不确定”与“不可靠不确定”。
- **实现方式**：
  - 在 PDD 的基础上增加第二阶段过滤；
  - 对候选样本计算 Grad-CAM 图并导出 **focus score**（聚焦得分），量化注意力集中程度；
  - 只有同时满足“低置信度”（uncertain）和“高 focus score”（meaningful attention）的样本才参与梯度更新。

### 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **更智能的数据选择** | 不再仅依据 confidence 判断样本重要性，而是结合解释质量，避免将噪声样本误判为有价值样本 |
| **提升效率与性能的双重增益** | 在降低训练成本的同时，反而提升了 macro-F1 性能，打破了传统“效率 vs. 性能”权衡 |
| **通用性强** | 可集成到多种 PDD 变体（DBPD, SMRD, SRD）中，并在不同 backbone 和数据集上稳定有效 |
| **无需架构修改** | 仅需在训练流程中插入 Grad-CAM 计算与过滤逻辑，不改变模型结构 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共三个公开 ECG 数据集，覆盖多样化的标签分布与采集条件：
- **PTB-XL**：大规模多标签 ECG 数据集，常用于 benchmark。
- **CPSC 2018**：中国心律失常挑战赛数据，侧重 arrhythmia 检测。
- **Georgia 2020**：来自美国乔治亚州医院的真实世界 ECG 记录，更具临床多样性。

### 实验设置
- **Backbone 架构**：三种不同容量的 CNN 模型
  - EfficientNetV2-S（高性能）
  - ResNet-18（标准）
  - MobileNetV2（轻量级）
- **训练策略**：
  - 动态样本选择：每轮 epoch 先用 PDD 筛选 uncertain 样本，再用 ERTS 过滤低 focus 样本；
  - 最终一轮（revision stage）恢复全数据集以确保全局覆盖。
- **Grad-CAM focus score 定义**：
  $$
  f(x_i) = \sum_{w \in \Omega} A(x_i)[w],\quad \text{其中 } \Omega = \{w \mid A(x_i)[w] \geq Q_{90}(A(x_i))\}
  $$
  即取 top 10% 最显著区域的激活值总和，反映注意力集中性。

### 评估指标
| 指标 | 说明 |
|------|------|
| **macro-F1** | 主要性能指标，对类别不平衡敏感，适合 ECG 多类分类任务 |
| **Effective Epochs (EE)** | 衡量训练效率：`total backprop samples / dataset size`，数值越小表示优化成本越低 |
| **Sample Count** | 实际参与反向传播的样本总数，反映真实计算开销 |

### 基线方法对比
- **Full-data baseline**：所有样本每轮都参与训练
- **Standard PDD variants**：
  - **DBPD**（Difficulty-Based Progressive Dropout）：基于 confidence 的难样本保留
  - **SMRD**（Schedule-Matched Random Dropout）：按相同 schedule 随机丢弃
  - **SRD**（Scalar Random Dropout）：随机比例衰减
- **ERTS-enhanced variants**：上述各方法 + Grad-CAM focus filtering

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| Dataset | Backbone | Best PDD (macro-F1 / EE) | Best ERTS (macro-F1 / EE) | ΔF1 | EE Saved |
|--------|----------|---------------------------|----------------------------|-----|----------|
| PTB-XL | EfficientNetV2-S | 0.6488 / 21.8 | **0.6586 / 18.9** | +0.0098 | 13.3% |
| PTB-XL | ResNet-18 | 0.6539 / 22.3 | **0.6624 / 19.3** | +0.0085 | 13.5% |
| PTB-XL | MobileNetV2 | 0.6336 / 21.6 | **0.6420 / 18.7** | +0.0084 | 13.4% |
| CPSC 2018 | EfficientNetV2-S | 0.7166 / 23.8 | **0.7188 / 19.4** | +0.0022 | 18.3% |
| Georgia 2020 | ResNet-18 | 0.6521 / 40.3 | **0.6569 / 34.0** | +0.0048 | 15.7% |

> ✅ 所有 9 种组合中，**ERTS 均优于对应最强 PDD 基线**，且在提升 macro-F1 的同时显著减少 EE。

### 与基线方法的对比结果
- **一致性优势**：无论 backbone 类型（大/中/小）、数据集特性（噪声水平、类别分布）、还是 PDD 子策略（DBPD/SMRD/SRD），ERTS 均带来正向增益。
- **效率提升显著**：
  - 平均节省 **5–18% 的 Effective Epochs**；
  - 图 4 显示实际参与 backprop 的样本数明显下降，意味着更低能耗与更快 retraining。
- **性能提升来源分析**（Fig. 6 & 7）：
  - ERTS 更倾向于移除 **NORM 类别中注意力分散的正常记录**；
  - 同时保留 **MI、STTC、CD 等诊断相关 hard cases**，即使它们 confidence 较低；
  - 相比之下，confidence-only 方法会过早丢弃这些早期被学会但临床上重要的类别。

### 消融实验结果
#### （1）Focus Score 阈值的影响（φ）
- φ = 0.5（激进过滤）：EE 下降但 macro-F1 显著下降 → **过度剪枝损害学习**
- φ = 0.7 或 0.9（适度过滤）：最佳平衡点，macro-F1 提升且 EE 减少
- 结论：**ERTS 应作为“可靠性过滤器”，而非“极端剪枝工具”**

#### （2）跨 PDD 策略有效性验证
- 在非 confidence-driven 的 **SMRD 和 SRD** 上也观察到性能提升
- 说明 Grad-CAM focus score 提供的是**独立于 confidence 的额外可靠性信号**

#### （3）Qualitative 示例（Fig. 8）
- **Retained samples**：Grad-CAM 注意力集中在 QRS 波群、J-point 等临床相关区域，形态合理
- **Filtered samples**：注意力弥散、振荡、无明确峰值，提示模型未建立可靠决策依据

---

## 4. 关键结论和发现

### 主要发现
1. **解释质量可作为有效的训练时信号**  
   xAI 方法（如 Grad-CAM）不仅能用于 post-hoc 解释，还可嵌入训练流程，指导数据选择，提升学习效率与可靠性。

2. **不确定性 ≠ 信息性**  
   低 confidence 的样本可能是由于噪声、伪影或弱标签造成，不应一视同仁地继续训练；而 ERTS 成功识别出那些“虽难但解释清晰”的样本予以保留。

3. **效率与性能可以兼得**  
   ERTS 实现了 **更高 macro-F1 + 更少 EE** 的双赢局面，表明去除冗余训练样本有助于模型更专注于高质量学习信号。

4. **方法具有广泛适用性**  
   在多个数据集、多种 backbone 和不同 PDD 策略下均表现稳健，证明其泛化能力。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖 Grad-CAM** | 当前仅使用 Grad-CAM，其空间平均可能忽略细粒度模式；其他方法（如 Integrated Gradients, LIME）因计算代价过高难以用于训练时 |
| **focus score 简单化** | 当前 focus score 基于阈值统计，缺乏对复杂病理模式（如分布式异常）的建模能力 |
| **未引入专家标注验证** | 缺乏医生对 filtered samples 是否确实为 artefact/noisy 的人工评估 |
| **超参数敏感性** | φ 阈值需根据 dataset 和 model capacity 调整，尚无自适应机制 |

### 未来工作方向
1. **探索更先进的 xAI 方法**：尝试 Grad-CAM++、Integrated Gradients 或 attention regularization 技术。
2. **设计自适应 thresholding 策略**：根据训练动态自动调整 φ。
3. **扩展至其他模态**：应用于 EEG、PPG、多模态临床数据等 time-series learning 场景。
4. **结合主动学习与 label cleaning**：利用解释信号识别潜在 mislabeled 样本，辅助数据清洗。
5. **绿色 AI 实践**：量化 ERTS 对碳足迹与能源消耗的实际减排效果。

---

> 📌 **一句话总结**：  
> 本论文开创性地将 **explainability 从 post-hoc 工具转变为 training-time reliability signal**，提出 **ERTS** 方法，在不影响甚至提升 ECG 分类性能的前提下，显著降低训练成本，为构建高效、可信的临床机器学习系统提供了新范式。

</details>

---

### 15. [Otters++: A Time-to-first-spike Based Energy Efficient Optical Spiking Transformer](https://arxiv.org/abs/2606.13016)

**Authors**: Zhanglu Yan, Jiayi Mao, Kaiwen Tang, Fanfan Li, Gang Pan, Tao Luo, Bowen Zhu, Qianhui Liu, Weng-Fai Wong  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.13016v1  

#### Abstract
Spiking neural networks (SNNs) are promising for energy-efficient inference, and time-to-first-spike (TTFS) coding is especially attractive because each neuron fires at most once. In practice, however, this benefit is often reduced by the cost of computing a temporal decay term and multiplying it by...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Otters++: A Time-to-first-spike Based Energy Efficient Optical Spiking Transformer

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于 **Time-to-first-spike (TTFS)** 编码的 Spiking Neural Networks (SNNs) 虽然理论上具有高时间稀疏性和低能耗潜力，但在实际实现中仍需进行显式的数字时序衰减函数计算（如指数或线性衰减），并将其与突触权重相乘。这一过程引入了额外的 **算术开销、内存访问和数据移动**，削弱了其能效优势。

此外，直接训练 TTFS-SNN 存在梯度传播困难、过稀疏化（over-sparsity）等问题，尤其在复杂模型（如 Transformer）中更为严重。

---

### 🔧 提出的新方法与核心创新

Otters++ 提出了一种“**硬件缺陷即功能**”的设计理念，将物理器件中的非理想特性转化为计算原语，主要贡献如下：

#### **(1) 物理驱动的 TTFS 时间项实现（Physically Grounded TTFS Computation）**
- 利用自研的 **In₂O₃ optoelectronic synapse** 的自然光信号衰减响应，直接实现 TTFS 中的时间调制项。
- 将原本需要软件/数字电路计算的 `decay(t) × weight` 过程，替换为通过模拟器件物理响应采样完成，**完全避免了显式的数字衰减计算**。
- 实现了 **时间调制与突触计算的物理融合**，显著降低算术成本。

#### **(2) 混合正向/反向传播训练框架（Hybrid SNN-forward / QNN-backward Training）**
- 构建了 Otters++ 层与 **unsigned Quantized Neural Network (QNN)** 的逐层功能等价关系。
- 正向传播保留真实的、设备保真的 SNN 计算流程（SNN-forward）；
- 反向传播则通过等效 QNN 路径使用 **Straight-Through Estimator (STE)** 计算梯度，规避对离散 first-spike 事件求导的难题。
- 结合 **knowledge distillation** 和 **noise-aware forward sampling**，提升鲁棒性。

#### **(3) 更真实的系统级能效评估模型**
- 细粒度建模包括：
  - Compute（MAC/ACC）
  - Memory access（weight/KV read/write）
  - Data movement（sparse spike transfer）
  - Analog read cost
  - Device sharing 效应
  - Multi-hop communication 开销
- 首次在 SNN 推理评估中综合考虑 **模拟读取代价** 与 **通信距离影响**，提供更贴近现实部署的能量估算。

---

### 📈 相比现有方法的优势
| 方面 | Otters++ 优势 |
|------|---------------|
| **能效** | 移除数字衰减计算 + 二值KV注意力 + 低精度通信 → 显著降低能量消耗 |
| **可训练性** | 混合训练策略解决 TTFS-SNN 梯度不可微问题，避免过稀疏化 |
| **硬件兼容性** | 充分利用器件物理特性而非试图补偿非理想行为，实现真正的硬件-算法协同设计（hardware-software co-design） |
| **鲁棒性** | 在前向传播中注入实测器件变异噪声，增强模型对制造偏差的容忍能力 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **GLUE benchmark**：包含 7 个自然语言理解任务
  - QQP, MNLI-m, SST-2, QNLI, RTE, MRPC, STS-B
- 所有任务均用于评估语言理解能力与泛化性能。

---

### ⚙️ 实验设置
- **模型架构**：基于 BERT-base 的 Spiking Transformer 架构
- **教师模型**：BERT-base（418M 参数）
- **学生模型大小**：13.4M（与其他轻量级 SNN 模型对齐）
- **知识蒸馏**：采用输出 logits 和隐藏状态双层级监督
- **量化配置**：
  - Key (K) 和 Value (V) 投影量化至 **1-bit**（{+1, -1}）
  - 使用 **4-bit 模拟窗口**（对应 T=15 时间步）
- **训练策略**：
  - 混合 SNN-forward / QNN-backward
  - Adam 优化器（BertAdam）
  - 注入实测器件 run-to-run variation 噪声（来自 In₂O₃ TFT 测量包络）

---

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **性能** | 各任务准确率（Accuracy）、STS-B 使用 Pearson 相关系数、GLUE 平均得分 |
| **能效** | 每层推理能耗（mJ/layer），细分为：<br>• MAC/ACC<br>• Weight/KV Read<br>• Data/Spike Movement<br>• Analog Read<br>• Leakage 等 |
| **鲁棒性** | 在 “Measured Variation” 设置下的性能标准差（std） |

---

### 🔁 基线方法对比
| 基线模型 | 类型 |
|--------|------|
| BERTbase | Full-precision Transformer |
| DistilBERT, TinyBERT6 | 压缩 ANN 模型 |
| Q2BERT, BiT | 低比特量化模型 |
| SpikingFormer, SpikingBERT, SpikeLM, 1-bit SpikeLM, 1-bit Sorbet | SNN-based Transformer |
| Otters (原始版本) | 前作，作为直接比较对象 |

---

## 3. 主要实验结果和性能指标

### 📈 性能表现（GLUE 平均得分）

| 模型 | GLUE Average Score |
|------|--------------------|
| BERTbase | 87.31 |
| Otters (原版) | 83.22 |
| **Otters++ (Nominal)** | **84.17** |
| **Otters++ (Variation)** | **84.17 ± 0.28** |

> ✅ **Otters++ 相比 Otters 提升 +0.95 分，相比其他 SNN 基线高出 3.34–4.37 分**

- 在 QNLI (+1.46) 和 MRPC (+2.44) 上增益最大，表明混合训练有效缓解了稀疏激活导致的信息丢失。
- 即使在 **实测器件变异下运行 20 次**，平均性能未下降，说明训练具备良好鲁棒性。

---

### 💡 能效分析（每层能耗）

| 模型 | 每层能耗 (mJ) | 能耗缩减倍数（vs Otters++） |
|------|----------------|----------------------------|
| Quantized BERT | 40.8 | 2.87× |
| Sorbet | 42.9 | 3.02× |
| SpikingBERT | 80.6 | 5.68× |
| SpikingLM | 26.1 | 1.84× |
| **Otters++ (Ours)** | **14.2** | — |

> ✅ **Otters++ 是所有方法中能耗最低的，最高节能达 5.68×**

#### 能耗构成分析（Figure 7）
- **传统 SNNs**：数据移动占 ~54%，权重读取占 ~31%
- **Otters++**：
  - 数据移动仍为主（58%），但 **高精度 weight/KV read 被大幅削减**
  - 引入 **analog read** 成本仅占个位数百分比（~12%）
  - MAC/ACC 和 leakage 占比合理

#### 多跳通信敏感性分析（Figure 8）
- 随着通信跳数增加（0 → 10 hops），所有模型能耗上升
- **Otters++ 始终保持最低能耗**
- 即使在低跳数区域（local compute 场景），优势依然明显
- 不同 analog read 成本假设下的能效带宽较窄，说明结论稳健

---

### 🔍 消融实验结果

#### (1) **训练策略消融（Table II）**
| 方法 | GLUE Avg |
|------|----------|
| QNN（纯量化网络） | 83.81 |
| Otters（QNN → SNN 后转换） | 83.22 |
| **Otters++（集成 TTFS 训练）** | **84.17** |

> ❗ 直接后转换导致性能下降 0.59 分，验证了 **训练-部署域不匹配问题**
>
> ✅ Otters++ 不仅弥补差距，还进一步提升性能，证明混合训练的有效性

#### (2) **与传统 TTFS 方法对比**
- 在相同量化时间域（T=15）、相同架构下重估传统 TTFS：
  - 传统方法需额外 MAC 操作计算 `T−t` 时间编码
  - 引入更多 weight access 和运算
- 结果：传统 TTFS 注意力模块能耗为 **19.09 mJ**，比 Otters++ 高 **34.3%**
- 表明 **利用器件衰减替代数字解码确实带来实质性节能**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **物理信号衰减可以被重构为核心计算资源**，而非需要校正的硬件“缺陷”。
2. **SNN-forward / QNN-backward 混合训练框架** 成功解决了 TTFS-SNN 可微性与稀疏性难题，实现了高精度训练。
3. **知识蒸馏 + 器件噪声注入** 显著提升了模型在真实硬件扰动下的鲁棒性。
4. **系统级能效建模必须包含 analog read、device sharing 和 multi-hop communication**，否则会高估收益。
5. Otters++ 在 **保持最高能效的同时，达到当前 SNN-based Transformer 的最优性能**。

---

### ⚠️ 方法的局限性
1. **依赖特定器件响应曲线**：当前设计基于 In₂O₃ TFT 的测量衰减函数，推广到其他材料需重新校准映射。
2. **光学读出速度受限**：目前使用 SMU 测量，最小积分时间约数十 μs，限制了推理延迟（当前 ~0.8ms/block）。
3. **尚未实现端到端光电集成芯片**：原型仍为分离式测量系统，未完成片上光-电协同处理闭环。
4. **KV 二值化可能限制表达能力**：虽然节省了注意力计算，但在某些复杂任务上可能存在容量瓶颈。

---

### 🔮 未来工作方向
1. **开发高速光电读出接口**：压缩光学衰减窗口至 ns 级，提升推理吞吐量。
2. **构建 3D 集成架构**：结合 ALD 工艺实现 In₂O₃ TFT 与 CMOS 的单片三维堆叠，缩小器件面积（有望缩小近三个数量级）。
3. **探索线性复杂度 Spiking Attention**：结合 QKFormer、SSSA 等线性注意力机制，进一步降低 O(N²) 开销。
4. **扩展至多模态任务**：应用于 audio-visual speech recognition 或 event-based vision 系统。
5. **构建开放工具链**：支持从 PyTorch 模型自动编译至 Otters++ 硬件执行流。

---

## 总结

Otters++ 是一次成功的 **hardware-software co-design** 实践，它不仅提出了一种新颖的物理计算范式，还将理论上的能效潜力转化为实际可观测的性能与能耗双重优势。该工作为下一代低功耗 AI 推理系统提供了重要的技术路径参考：**不是一味追求更低比特或更高稀疏，而是让算法去拥抱并利用硬件的本质特性**。

</details>

---

### 16. [Mental-R1: Aligning LLM Reasoning for Mental Health Assessment](https://arxiv.org/abs/2606.13176)

**Authors**: Xin Wang, Boyan Gao, Yibo Yang, David A. Clifton  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.13176v1  

#### Abstract
Mental health problems such as anxiety, depression, and suicide remain urgent global challenges, where timely and accurate assessment is critical for effective intervention. Recently, large language models have been explored for mental health assessment. However, existing general-purpose post-traini...

---

### 17. [APEX: A Network-Native Time-Series Foundation Model for Forecasting and Anomaly Detection for Wireless Edge Operations](https://arxiv.org/abs/2606.11553)

**Authors**: Swadhin Pradhan, Niloo Bahadori, Peiman Amini  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11553v1  

#### Abstract
Generic time-series foundation models transfer poorly to wireless network telemetry whose signals are bursty, zero-inflated, and coupled across protocol layers. We present APEX, a network-native, decoder-only transformer for forecasting enterprise AP telemetry, and evaluate it on DHCP degradation as...

---

### 18. [Reinforcement Learning Disrupts Gradient-Based Adversarial Optimization](https://arxiv.org/abs/2606.12251)

**Authors**: Xinhai Zou, Chang Zhao, Alireza Aghabagherloo, Dave Singel\'ee, Robin Degraeve, Bart Preneel  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.12251v1  

#### Abstract
Gradient-based adversarial attacks remain a dominant threat to deep neural networks (DNNs), as they exploit gradient information to efficiently optimize adversarial perturbations. To address this, we investigate whether reinforcement learning (RL) training can disrupt the gradient structure used by ...

---

### 19. [Holding the FP8 Quality Ceiling at 8-Bit Weights and Activations: INT8 and GGUF Post-Training Quantization of Ideogram 4.0 for Consumer GPUs](https://arxiv.org/abs/2606.12280)

**Authors**: Deep Gandhi, Ali Asaria, Tony Salomone  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.12280v1  

#### Abstract
Post-training quantization lets large text-to-image diffusion transformers run on consumer GPUs, yet the hardware-specific trade-offs are seldom measured directly. We quantize Ideogram 4.0 - a 9.3B flow-matching diffusion transformer (DiT), shipped as two separate-weight copies of a single-stream 34...

---

### 20. [Reward Modeling for Multi-Agent Orchestration](https://arxiv.org/abs/2606.13598)

**Authors**: King Yeung Tsang, Zihao Zhao, Vishal Venkataramani, Haizhou Shi, Zixuan Ke, Semih Yavuz, Shafiq Joty, Hao Wang  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.13598v1  

#### Abstract
Multi-Agent Systems (MAS) built on Large Language Models (LLMs) require effective orchestration to coordinate specialized agents, yet training such orchestrators is hindered by limited supervision and high computational cost. We propose Orchestration Reward Modeling (OrchRM), a self-supervised frame...

---

### 21. [Multi-Turn Reasoning When Context Arrives in Pieces: Scalable Sharding and Memory-Augmented RL](https://arxiv.org/abs/2606.12941)

**Authors**: Shu Tong Luo, Wenqin Liu, Rui Liu, Mingming Gong, Jiaxian Guo  
**Category**: cs.CL  
**Published**: 2026-06-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.12941v1  

#### Abstract
When a user reveals task-critical information across several conversation turns, LLM accuracy drops by up to 65% despite full context availability. We show that this Lost in Conversation degradation can be substantially mitigated by training models to maintain a compact rolling memory instead of att...

---

### 22. [Beyond Uniform Tokens: Adaptive Compression for Time Series Language Models](https://arxiv.org/abs/2606.13624)

**Authors**: Jialin Gan, Xin Qiu, Guangzhe Chen, Xue Wang  
**Category**: cs.CL  
**Published**: 2026-06-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.13624v1  

#### Abstract
Large language models (LLMs) have enabled time series (TS) analysis by jointly modeling numerical observations and textual context through a shared token interface. However, TS tokens and prompt tokens exhibit fundamentally different information structures, making uniform token processing inefficien...

---

### 23. [Maestro: Workload-Aware Cross-Cluster Scheduling for LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2606.12950)

**Authors**: Jinghao Wang, Xiao Zhou, Xiaoyang Sun, Yihui Zhang, Yilong Li, Tianyu Wo, Xu Wang, Chunming Hu, Renyu Yang  
**Category**: cs.DC  
**Published**: 2026-06-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.12950v1  

#### Abstract
Large Language Model based Multi-Agent Systems (LLM-MAS) have emerged as a powerful paradigm for tackling complex tasks by breaking them into collaborative workflows of specialized LLM-powered agents. However, deploying such multi-agent workloads at scale poses significant system challenges. Each us...

---

### 24. [SwiftCTS: Fast Cross-Design Prediction and Pareto Optimization of Clock Tree Metrics via Few-Shot Calibration](https://arxiv.org/abs/2606.11348)

**Authors**: Barsat Khadka, Kawsher Roxy, Md Rubel Ahmed  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.11348v1  

#### Abstract
Clock Tree Synthesis (CTS) is a computationally expensive stage in the physical design flow, requiring iterative EDA tool invocations to navigate a vast configuration space for optimal power, wirelength, and timing skew. Existing machine learning approaches require computationally expensive retraini...

---

### 25. [Harness In-Context Operator Learning with Chain of Operators](https://arxiv.org/abs/2606.12318)

**Authors**: Minghui Yang, Ling Guo, Liu Yang  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.12318v1  

#### Abstract
Neural operators approximate mappings between function spaces, but often generalize poorly to other operators and usually require fine-tuning or retraining. In-Context Operator Networks (ICON) addresses this issue by prompting the model with numerical context so that the model learns specific operat...

---

### 26. [SkillCAT: Contrastive Assessment and Topology-Aware Skill Self-Evolution for LLM Agents](https://arxiv.org/abs/2606.13317)

**Authors**: Kunfeng Chen, Qihuang Zhong, Juhua Liu, Bo Du  
**Category**: cs.CL  
**Published**: 2026-06-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.13317v1  

#### Abstract
Skill self-evolution methods for LLM agents aim to turn execution trajectories into reusable skill documents, but current pipelines typically learn from one trajectory per task, merge candidate skill patches before checking them, and load the full skill corpus before inference. We propose SkillCAT, ...

---

### 27. [To Intervene or Not: Guiding Inference-time Alignment with Probabilistic Model Blending](https://arxiv.org/abs/2606.11201)

**Authors**: Jin Gan, Xin Li, Jun Luo  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.11201v1  

#### Abstract
The wide deployment of LLMs has made model alignment necessary to make newly trained models safely and effectively respond to user instructions. Among different methods, inference-time alignment is often cheaper as it intervenes (i.e., offers guidances) only during output generation. Existing propos...

---

### 28. [SirenFNO: Efficient and Full Frequency Learning of Fourier Neural Operators](https://arxiv.org/abs/2606.11518)

**Authors**: Pengqing Shi, Jie Yin, Stephen Tierney, Junbin Gao  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.11518v1  

#### Abstract
Fourier neural operators (FNOs) are effective and efficient surrogates for approximating solutions of PDEs and generalize across discretizations. However, owing to the reliance on frequency truncation to maintain learning efficiency of FNOs, empirical studies suggest that FNOs exhibit spectral bias ...

---

### 29. [Range-Aware Bayesian Optimization for Discovering Diverse Designs within Target Property Windows](https://arxiv.org/abs/2606.11574)

**Authors**: Shengli Jiang, Jason Wu, Charles M. Schroeder, Michael A. Webb  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.11574v1  

#### Abstract
In many materials and product design problems, desirable candidates exhibit properties that fall within an acceptable range rather than achieve a single optimum. Recovering multiple, distinct solutions that satisfy such specifications is also practically valuable, as some candidates may be preferred...

---

### 30. [DeMix: Debugging Training Data with Mixed Data Error Types by Investigating Influence Vectors](https://arxiv.org/abs/2606.11616)

**Authors**: Jiale Deng, Yanyan Shen, Xiaogang Shi, Chai Junjun  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.11616v1  

#### Abstract
High-quality training data is essential for the success of machine learning models. However, real-world datasets often contain mixed types of errors arising from systematic flaws in data preparation pipelines, including label errors, feature errors, and spurious correlations. Effective debugging of ...

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

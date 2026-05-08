# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-08 07:14:57 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Requests of a Feather Must Flock Together: Batch Size vs. Prefix Homogeneity in LLM Inference](https://arxiv.org/abs/2605.06046)

**Authors**: Saksham Rathi,  Preeti, Mythili Vutukuru  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.06046v1  

#### Abstract
Auto-regressive token generation in large language models is memory-bound because it requires "attending to" key and value tensors (KV cache) of all previous tokens. Prior work aims to improve the efficiency of this decode process by batching multiple requests together, and maximizing batch size sub...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Requests of a Feather Must Flock Together: Batch Size vs. Prefix Homogeneity in LLM Inference》论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**大语言模型（LLM）推理中的 decode 阶段效率瓶颈**提出新的优化视角。传统方法普遍认为应最大化 batch size 以提升 GPU 利用率，但作者指出这一策略忽略了 **prefix homogeneity（前缀同质性）** 对性能的关键影响。

具体而言，现有系统存在两大问题：
- **调度策略缺陷**：即使请求间存在共享前缀（如 few-shot prompting、system prompts），主流调度器仍倾向于构建大的异构 batch，牺牲了内存访问局部性带来的性能增益。
- **CPU 开销过高**：基于 radix tree 的 prefix detection 在高并发长序列场景下引入显著 CPU 开销，有时甚至超过 GPU 执行时间。

### 提出的新方法与思路
论文提出了 **FEATHER** —— 一种新型的 prefix-aware 调度框架，其核心思想是**在 batch size 和 prefix homogeneity 之间进行智能权衡**，而非一味追求最大 batch。

#### 主要创新组件：

| 组件 | 描述 |
|------|------|
| **Chunked Hash Tree (CHT)** | 一种轻量级数据结构，用于高效检测请求间的前缀共享关系。相比 radix tree，它避免了昂贵的 token-level traversal，通过分块哈希向量实现 O(log W) 的候选选择复杂度。 |
| **Reinforcement Learning (RL) 调度策略** | 将“是否继续添加请求到当前 batch”建模为一个强化学习决策问题。RL 策略根据当前 batch 的大小、共享前缀长度损失（Δ）、以及潜在可加入的同前缀请求数量（w）动态决定 `ADD` 或 `STOP`。 |

### 相比现有方法的优势
| 方面 | FEATHER 优势 |
|------|-------------|
| **性能** | 在具有 prefix sharing 的负载下，端到端吞吐量（throughput）达到 vLLM FCFS 和其他基线的 **2–10×**。 |
| **硬件无关性** | 不依赖特定 GPU 架构或定制 kernel，可无缝集成至 vLLM 和 SGLang 等主流引擎。 |
| **低开销** | CHT 将 prefix detection 的 CPU 开销降低 **高达 1000×**，远低于 DFS-based 方法。 |
| **自适应性强** | RL 策略能自动适应不同 workload 特征，在无 prefix sharing 场景下表现不劣于 FCFS，确保“零退化”。 |

---

## 2. 核心实验方法和设置

### 数据集
- **L-Eval**：包含摘要、问答等任务的人工标注 query-response 对，序列长度从 2.7K 到 210.5K tokens。
- **LongBench**：涵盖多文档 QA、代码补全等六类长上下文样本，context length 在 4K–10K tokens 之间。

### 实验设置
- **模型**：Llama 3 8B、Qwen 0.5B/1.5B/8B、LongChat 13B。
- **硬件**：NVIDIA RTX 6000 Ada GPU（48GB GDDR6，96MB L2 cache）；部分对比在 A100-80GB 上运行。
- **请求模式**：使用泊松过程模拟请求到达，控制输入速率（如 100 req/s）、decode token 数（1–200）、共享前缀长度（1K–10K tokens）等变量。
- **token budget**：默认 32K tokens。

### 评估指标
| 指标 | 含义 |
|------|------|
| **Throughput (toks/s)** | 每秒生成的输出 token 数，衡量系统整体服务能力。 |
| **Time Between Tokens (TBT)** | 平均每两个解码 token 之间的间隔，反映响应延迟。 |
| **Average Batch Size** | 每次 forward pass 的平均 batch 大小，关联调度行为与性能。 |
| **DRAM Bandwidth Utilization** | GPU 显存带宽利用率，反映内存访问效率。 |

### 基线方法对比
| 基线 | 说明 |
|------|------|
| **vLLM (FCFS)** | 默认先来先服务策略，作为基础 baseline。 |
| **SGLang (FCFS/LPM/DFS-W)** | 支持 prefix-aware 调度，其中 LPM（最长前缀匹配）和 DFS-W（加权深度优先）旨在提高 prefix reuse。 |
| **Dynamic Batching [28]** | 动态调整 batch size 以满足内存和延迟约束。 |
| **PAT [41]** | 一种 prefix-aware attention kernel，通过查询打包减少冗余 KV cache 访问。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 场景 | FEATHER 性能提升 |
|------|------------------|
| 共享前缀长度 5K–10K tokens，request rate=100 req/s | 达到 vLLM FCFS 的 **~4× 吞吐量** |
| 100 个 prefix groups，LongChat 13B 模型 | 达到 vLLM FCFS 的 **22× 吞吐量** |
| 高 decode length（200 tokens） | 吞吐量达 **~3800 toks/s**，超 SGLang 政策 3× 以上 |
| CPU 调度开销（20K tokens 序列） | 比 SGLang LPM/DFS-W 低 **1000×**，始终低于 GPU 时间的 1% |

### 与基线方法的对比结果
- **vs. vLLM FCFS**：
  - FEATHER 构建更小但更 homogenous 的 batches，却实现了更高 throughput。
  - 当 prefix groups 从 1 增加到 2 时，FCFS 吞吐急剧下降，而 FEATHER 几乎不受影响。
- **vs. SGLang LPM/DFS-W**：
  - 尽管也利用 prefix sharing，但由于 radix tree traversal 开销巨大，实际 TBT 更高。
  - FEATHER 在相同 prefix-aware 能力下，CPU 开销极低，GPU 利用率更高。
- **vs. PAT (kernel-level 优化)**：
  - FEATHER 在 A100 上的表现**持平甚至超越**专门为该硬件优化的 PAT kernel。
  - 表明仅通过调度层优化即可达到 kernel-level 优化的效果，且更具通用性。

### 消融实验结果
| 实验 | 发现 |
|------|------|
| **替换 CHT 为 Radix Tree** | 即使使用相同的 RL 策略，吞吐量大幅下降，TBT 显著升高 → 证明 **CHT 对降低 CPU 开销至关重要**。 |
| **移除 RL 策略（always maximize batch）** | 在低负载下性能接近 FCFS；但在中高负载下无法形成 homogenous batch → 吞吐明显低于完整 FEATHER → 证明 **RL 决策对性能增益起关键作用**。 |
| **不同 chunk size 敏感性测试** | CHT 对 chunk size 不敏感，即使 chunk=1（等价于 radix tree 粒度），仍比真实 radix tree 高效得多 → 体现 **设计鲁棒性**。 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Prefix Homogeneity 是比 Batch Size 更重要的性能决定因素**  
   所有请求共享一个共同前缀时，KV cache 的空间和时间局部性极大增强，带来更高的 DRAM 带宽利用率和更低的数据搬运量。

2. ✅ **Moderately Small Homogeneous Batches 可胜过 Large Heterogeneous Ones**  
   实验表明，8×100 的 homogenous batch 比 1×800 的 heterogeneous batch 吞吐高出近 2×，尤其在 decode length 较长时优势更明显。

3. ✅ **现有 prefix-aware 调度器存在严重 CPU Overhead**  
   SGLang 的 LPM 和 DFS-W 在长序列下的调度开销可达 GPU 时间的 50–90%，成为新的瓶颈。

4. ✅ **FEATHER 实现了调度层面的最优权衡**  
   通过 CHT + RL 的组合，实现了低开销、高性能、强自适应性的统一。

### 方法的局限性
- **依赖 workload 中存在一定程度的 prefix sharing**：若所有请求完全独立（no prefix overlap），则无法获得额外收益（尽管也不会变差）。
- **目前仅支持单 GPU 场景**：未考虑分布式或多 GPU setting 下的扩展性。
- **RL 策略需要一定训练/探索期**：虽然 bandit 收敛快（约 100 秒），但在极端动态变化的负载下可能需进一步优化。

### 未来工作方向
- 扩展至 **distributed and multi-GPU settings**。
- 探索更轻量、更自适应的 learning policy（如 online adaptation）。
- 与 prefix-aware attention kernels（如 PAT）进行**深度协同设计**，实现调度层与 kernel 层的联合优化。

---

> 🪶 **命名由来（Bonus）**  
> “FEATHER” 名称源于其 batch 结构的视觉隐喻：共享前缀如同羽毛的中央羽轴（shaft），各个请求在其后分叉延伸，形似羽枝（barbs）。这既形象地表达了“同源分支”的结构特征，也寓意着将相似请求聚合在一起所带来的轻盈高效（lightweight & efficient）之感。

</details>

---

### 2. [CCL-Bench 1.0: A Trace-Based Benchmark for LLM Infrastructure](https://arxiv.org/abs/2605.06544)

**Authors**: Eric Ding, Byungsoo Oh, Bhaskar Kataria, Kaiwen Guo, Jelena Gvero, Abhishek Vijaya Kumar, Arjun Devraj, Lindsey Bowen, Atharv Sonwane, Emaad Manzoor, Rachee Singh  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.06544v1  

#### Abstract
Evaluative claims about LLM infrastructure -- ``workload X is fastest on hardware Y with software Z'' -- depend on a complex configuration space spanning hardware accelerators, interconnect bandwidth, software frameworks, parallelism plans, and communication libraries. Current infrastructure evaluat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CCL-Bench 1.0: A Trace-Based Benchmark for LLM Infrastructure

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前的 **LLM 基础设施基准测试**（如 MLPerf、LLM-Perf）存在三大根本性局限：

- **结果导向而非解释导向**：仅报告端到端性能指标（如 step time、MFU），无法解释“为何”某一配置表现更好。
- **度量冻结在实验时**：若需分析新的维度（如通信开销、计算利用率），必须重新运行实验，成本高昂。
- **调优过程不可见**：公开榜单上的“胜出”往往依赖于高度调优的引擎对抗默认配置，但调优路径不透明，难以复现。

这些问题导致基准测试沦为“记分牌”，缺乏诊断价值，阻碍了硬件设计、软件优化和生产部署的深入决策。

---

### 提出了什么新方法或新思路

论文提出了 **CCL-Bench 1.0**，一个基于执行轨迹（execution trace）的新型 LLM 基础设施基准框架，其核心是 **证据驱动的评估范式**。

#### 创新架构：三层体系

1. **证据层（Evidence Layer）**
   - 每次实验提交三个标准化组件：
     - **Execution Trace**：细粒度执行日志（Kineto 或 XLA Profiler 格式），记录算子、内核、通信事件、时间戳等。
     - **Workload Card (YAML)**：描述模型、批大小、序列长度、并行策略、硬件平台、框架版本等完整上下文。
     - **Run Scripts**：可复现的启动脚本。

2. **分析层（Analysis Layer）**
   - 提供一个**社区可扩展的度量工具包**（community-extensible metric toolkit）。
   - 工具为函数 `f: (WorkloadCard, Trace) → ScalarMetric`，支持后验（post-hoc）计算新指标，无需重跑实验。

3. **应用层（Application Layer）**
   - 支持下游分析插件，如：
     - **What-if 分析**：将 trace 转换为 Chakra 图，输入 Astra-Sim 等模拟器，预测不同网络带宽下的性能。
     - **自动化配置优化**：集成 CCL-Search，一个基于 LLM Agent 的自动调优系统。

---

### 相比现有方法的优势

| 维度 | 传统基准（MLPerf 等） | CCL-Bench |
|------|------------------------|-----------|
| **可解释性** | ❌ 仅有结果 | ✅ 可归因瓶颈（计算/内存/通信） |
| **灵活性** | ❌ 度量固定 | ✅ 支持后验添加新度量 |
| **可复现性** | ❌ 配置常不完整 | ✅ 完整 trace + 卡片 + 脚本 |
| **调优可见性** | ❌ 黑箱 | ✅ 全过程记录为 benchmark 条目 |
| **跨平台比较** | ❌ 难以对齐 | ✅ 统一 trace 格式支持公平对比 |

---

## 2. 核心实验方法和设置

### 使用的模型与工作负载（Workloads）

CCL-Bench 1.0 定义了一组标准工作负载（见 Table 4），覆盖训练与推理、密集与 MoE 模型：

| ID  | Model               | Phase      | Batch Size | Sequence/Input Length |
|-----|---------------------|------------|------------|------------------------|
| WL1 | Qwen3-4B            | Inference  | 128        | 1024 input             |
| WL2 | Llama-3.1-8B        | Inference  | 128        | 1024 input             |
| WL3 | DeepSeek-MoE-16B    | Inference  | 128        | 1024 input             |
| WL4 | Llama-3.1-8B        | Training   | 4          | 512 sequence           |
| WL5 | DeepSeek-V3-16B     | Training   | 8          | 1024 sequence          |
| WL6 | DeepSeek-V3-16B     | Training   | 64         | 2048 sequence          |
| WL7 | DeepSeek-V3-236B    | Training   | 64         | 1024 sequence          |

---

### 实验设置

- **硬件平台**：
  - **GPU**：NERSC Perlmutter 超算，A100 GPU，NVLink 3.0 (300 GB/s)，Slingshot-11 (200 Gbps)。
  - **TPU**：Google TPU v6e，2D Torus 拓扑，ICI 带宽 100 GB/s。
- **软件栈**：
  - 框架：TorchTitan、Megatron-LM、MaxText、vLLM、SGLang。
  - 通信库：NCCL、MSCCL++、XLA Collectives。
  - 并行策略：TP、DP、PP、EP、微批次大小、激活检查点等。

---

### 评估指标（Metrics）

CCL-Bench 工具包提供多维度量，超越单一标量：

| 类别 | 指标 | 说明 |
|------|------|------|
| **模型执行** | `avg_step_time`, `MFU`, `TTFT`, `TPOT` | 端到端延迟与计算效率 |
| **计算** | `SM_Coverage`, `Dominant_Kernel_Concentration` | GPU/TPU 利用率与瓶颈内核 |
| **内存** | `Avg_Mem_BW`, `Memory_Transfer_Overhead` | 内存拷贝效率与暴露开销 |
| **通信** | `Comm_Fraction`, `Compute_Comm_Overlap`, `AllReduce_BW` | 通信占比、重叠程度、集体通信带宽 |
| **效用** | `Scale-Up_BW_Utility` | 带宽翻倍带来的步时间改进百分比 |

---

### 基线方法对比

CCL-Bench 支持两类比较：

1. **软件基础设施比较**  
   固定硬件与工作负载，比较不同软件组件（如 NCCL vs. MSCCL++）。

2. **硬件基础设施比较**  
   固定工作负载与 XPU 预算，允许各平台使用原生部署（如 PyTorch+NCCL on GPU vs. MaxText+XLA on TPU）。

---

## 3. 主要实验结果和性能指标

### 关键发现一：更高的计算-通信重叠不一定更快

- 在 DeepSeek-V3-16B MoE 训练中（WL5），**更高的 `Compute-Comm Overlap` 反而对应更长的 `step time`**。
- 原因：较小的 Expert Parallelism (EP=4) 导致专家在数据并行域复制，引发大量 `ReduceScatter` 和 `AllGather` 通信，尽管部分可重叠，但总通信量增加导致整体变慢。
- **结论**：盲目追求高重叠可能掩盖低效的并行策略。

> 🔹 **Claim 1**: Higher compute-communication overlap can coincide with longer training step time and reveal inefficient parallelization choices.

---

### 关键发现二：TPU 与 GPU 对带宽升级的响应差异巨大

- **Doubling TPU ICI bandwidth** 在小中型任务上带来高达 **100×** 的端到端收益提升，远超 GPU。
- 原因：TPU 基线带宽更低（100 GB/s vs. 300 GB/s），且拓扑连接性差（Torus vs. Fully-connected），因此带宽是更紧的瓶颈。
- 对于大型 GPU 训练任务，扩大 scale-up 域（如从单节点到多节点 NVLink）比单纯提升带宽更有效。

> 🔹 **Claim 2**: Doubling TPU interconnect bandwidth yields a much higher end-to-end improvement in step time than doubling GPU interconnect bandwidth on small and medium workloads.

---

### 关键发现三：框架间最优配置不可迁移

- 在相同硬件（16× A100）和工作负载（Llama-3.1-8B, WL4）下：
  - **TorchTitan 最优配置**：TP=1, DP=4, PP=4 → step time = **1.50s**
  - **Megatron-LM 最优配置**：TP=4, DP=1, PP=4 → step time = **0.44s**
  - 性能差距达 **3.4×**
- 若将 TorchTitan 的最优配置用于 Megatron-LM，性能仅为 1.3s，仍比其自身最优慢 **3×**。

> 🔹 **Claim 3**: The best-tuned configuration on one training framework can run up to 3× slower than the best-tuned configuration on a peer framework on identical hardware.

---

### CCL-Search 自动调优效果

- CCL-Search 在 15 轮迭代内：
  - 在 TorchTitan 上将 step time 降低 **8×**
  - 在 Megatron-LM 上降低 **19×**
- 支持复合目标（如平衡 step time 与 GPU 数量），找到帕累托前沿配置。

---

## 4. 关键结论和发现

### 主要发现总结

1. **性能不能仅看 summary statistics**：必须结合 trace 才能解释“为什么”。
2. **通信效率比绝对带宽更重要**：TPU 架构对带宽更敏感，优化空间更大。
3. **并行策略需协同优化**：如 MoE 中 EP 与 DP 的权衡。
4. **框架特定调优至关重要**：不存在通用最优配置，跨框架迁移会严重损失性能。

---

### 方法的局限性

- **覆盖率有限**：目前仅支持中小规模开源模型，缺少在线服务、量化、稀疏性等场景。
- **存储开销大**：单次多卡 trace 可达数十 GB，长期存储与共享挑战大。
- **隐私与合规风险**：trace 可能泄露集群拓扑或内部配置，部分企业不愿公开原始 trace。
- **Trace 格式异构**：GPU (Kineto) 与 TPU (XLA) trace 在时间单位、FLOPs 报告等方面存在差异，需额外处理。

---

### 未来工作方向

- **扩展工作负载范围**：加入更大模型、更多加速器（如 Trainium）、在线推理、准确率影响优化（如量化）。
- **开发 trace 压缩与采样格式**：降低存储与传输成本。
- **构建 trace 模拟器插件生态**：支持更多模拟器（如 SimAI、Vidur）。
- **推动 trace 匿名化工具**：实现隐私保护下的共享。
- **建立主题驱动的提交机制**：如“通信后端专项”、“MoE 优化挑战赛”。

---

### 开源与社区建设

- **代码与工具包**：https://github.com/cornell-sysphotonics/ccl-bench
- **基准平台**：https://cclbench.ai/
- 鼓励社区贡献 trace、开发新 metric 工具，共同构建可解释、可复现、可持续演进的 LLM 基础设施评估生态。

> ✅ **CCL-Bench 的愿景**：从“谁最快”的记分牌，走向“为何如此”、“如何改进”的科学基准。

</details>

---

### 3. [Relay Buffer Independent Communication over Pooled HBM for Efficient MoE Inference on Ascend](https://arxiv.org/abs/2605.06055)

**Authors**: Tianlun Hu, Tiancheng Hu, Shengsheng Litang, Sheng Wang, Xiaoming Bao, Yuxing Li, Wei Wang, Zhongzhe Hu, Lijun Li, Hongwei Sun, Jingbin Zhou\\  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.06055v1  

#### Abstract
Mixture-of-Experts (MoE) inference requires large-scale token exchange across devices, making dispatch and combine major bottlenecks in both prefill and decode. Beyond network transfer, routing-driven layout transformation, temporary relay, and output restoration can add substantial overhead. Existi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Relay Buffer Independent Communication over Pooled HBM for Efficient MoE Inference on Ascend

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 **Mixture-of-Experts (MoE)** 模型在推理过程中依赖大规模跨设备的 token 交换，导致 **dispatch** 和 **combine** 成为吞吐量导向（prefill）和延迟敏感（decode）场景下的主要瓶颈。传统实现中，通信开销不仅来自网络传输，还包括由路由驱动的布局转换、中间 relay 缓冲区和输出恢复等操作，这些都会显著增加端到端延迟。

许多现有 MoE 通信路径仍采用“缓冲区中心”（buffer-centric）设计，依赖显式的进程间通信（IPC）relay 和中间重排序缓冲区，造成额外内存占用和延迟。

### 提出了什么新方法或新思路
本文提出了一种 **relay-buffer-free（继电器缓冲区无关）的通信设计**，用于在华为 Ascend 平台上加速 MoE 推理。其核心思想是：

- 利用 Ascend 的 **全局池化高带宽内存（globally pooled HBM）** 和对称内存分配能力；
- 将 dispatch 和 combine 重构为：
  - **Direct Placement**：将 token 直接写入目标专家窗口（destination expert window），跳过中间 relay 缓冲区；
  - **Direct Reading**：从远程专家窗口直接读取输出，避免恢复阶段；
- 仅保留轻量级控制状态（如计数、偏移量、同步元数据），大幅减少中间数据拷贝和临时内存使用。

该设计针对 MoE 推理的两个主要阶段分别优化：
- **Prefill Schedule**：保留更丰富的规划状态以支持高吞吐执行；
- **Decode Schedule**：采用更紧凑的流程以降低延迟敏感路径的开销。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **通信效率** | 减少冗余的数据复制和布局变换，缩短关键路径 |
| **内存占用** | 显著降低瞬态 HBM 占用，有利于更大 batch 或更长上下文处理 |
| **硬件适配性** | 遵循 Ascend “读优先”（read-favored）执行特性，提升实际性能 |
| **灵活性** | 支持不同调度策略，在 prefill 和 decode 场景下均有效 |

相比通用 collectives（如 All-to-All）或 DeepEP 中的传统路径，本方法实现了更低的 dispatch/combine 延迟，并提升了端到端服务性能。

---

## 2. 核心实验方法和设置

### 数据集与模型
未使用公开标准数据集进行训练，而是基于以下代表性 MoE 模型和服务场景构建测试负载：
- **DeepSeek-V3** 系列（如 DeepSeek 3.1, 3.2）
- **Qwen2.5-1M**, **Qwen-235B**

这些模型具有典型的 MoE 结构，支持 long-context（最长达百万 token）和多模态扩展。

### 实验设置
- **平台**：Ascend AI 处理器集群，利用 `AclShmem` / `Memfabric` 提供的全局池化 HBM 和共享内存访问机制。
- **对比基线**：
  - **HCCL-enabled DeepEP baseline**：基于 Huawei Collective Communication Library 的传统 MoE 通信实现。
- **评估阶段**：
  - **Prefill**：大批次输入 token 的处理，关注吞吐；
  - **Decode**：自回归生成阶段，关注单步延迟。
- **量化配置**：包含非量化（non-quant）与量化（quant）两种模式，验证鲁棒性。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Kernel-level Latency** | dispatch 和 combine 内核的执行时间（μs） |
| **TTFT (Time to First Token)** | 从请求开始到首个输出 token 的延迟（ms） |
| **TPOT (Time Per Output Token)** | 每个输出 token 的平均生成时间（ms） |
| **Feasible Scheduling Space** | 在给定 TTFT 和 TPOT 约束下可满足的服务配置数量 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Kernel-Level 性能提升（Decode 阶段）
| Operator | Hidden Size | Quant Mode | Baseline (μs) | Proposed (μs) | Avg. Speedup |
|---------|-------------|------------|----------------|----------------|---------------|
| Dispatch | 4096        | non-quant  | 94.95          | 73.81          | **20.18%**     |
| Dispatch | 4096        | quant      | 88.62          | 59.46          | **31.02%**     |
| Dispatch | 7168        | non-quant  | 124.73         | 104.48         | **14.83%**     |
| Dispatch | 7168        | quant      | 101.06         | 71.48          | **27.72%**     |
| Combine  | 4096        | non-quant  | 59.88          | 46.03          | **22.75%**     |
| Combine  | 4096        | quant      | 61.88          | 46.02          | **24.45%**     |
| Combine  | 7168        | non-quant  | 89.76          | 70.23          | **22.43%**     |
| Combine  | 7168        | quant      | 91.41          | 69.80          | **24.34%**     |

> 📌 **Dispatch 加速尤为明显**，最高达 **52.8%**（Qwen-235B 场景）；Combine 提升稳定在 ~22–24%。

#### ✅ Prefill 阶段性能（随 token 数增长）
- 在 16384 tokens 输入时：
  - Dispatch 延迟从 ~9.4ms（baseline）降至 ~6.8ms（proposed），相对降低约 **27.7%**；
  - Combine 同样呈现差距扩大趋势，表明 buffer-centric 开销在大数据量下累积更严重。

#### ✅ 端到端服务性能（DeepSeek 3.1 场景）
| 指标 | HCCL-enabled DeepEP | ZeroBufferEP (Proposed) | 变化 |
|------|------------------------|----------------------------|-------|
| **TTFT** | 11197 ms               | **6793 ms**                | ↓ **39.3%** |
| **TPOT** | 30.10 ms               | 31.31 ms                   | ≈ 持平 |

> ⚠️ TPOT 轻微上升但仍低于目标阈值（60ms），说明稳态生成速度未受损。

#### ✅ 调度空间扩展（DeepSeek 3.2 场景）
- 在 TTFT < 5000ms 且 TPOT < 60ms 的约束下：
  - 提出的方法使更多配置进入可行区域；
  - 最佳 QPS 点更接近边界但仍满足要求；
  - 表明系统获得了更大的调度自由度。

---

## 4. 关键结论和发现

### 主要发现
1. **MoE 通信瓶颈不仅是带宽问题，更是“缓冲区架构”问题**  
   中间 relay 和重排序缓冲区带来的内存流量和同步开销不可忽视，尤其在 long-context 和低延迟场景下。

2. **基于全局池化 HBM 的 direct placement & direct reading 是有效的优化方向**  
   利用 Ascend 的硬件特性（symmetric memory + shmem-style remote access），可以绕过传统 sender-pack / receiver-restore 流程，实现更高效的 MoE 通信。

3. **同一通信模型可统一应用于 prefill 和 decode**  
   尽管两阶段需求不同（吞吐 vs 延迟），但均可通过调整控制状态复杂度来适配，形成统一的 relay-buffer-free 架构。

4. **性能增益在多种模型和配置下具有一致性**  
   在 DeepSeek、Qwen 等多个 MoE 模型上均观察到显著延迟下降，且在量化/非量化设置下保持稳健。

### 方法的局限性
- 当前实现依赖 Ascend 特有的 **全局地址空间** 和 `AclShmem/Memfabric` 支持，难以直接迁移到其他缺乏对称内存访问能力的平台（如部分 GPU 集群）；
- 对于极小 batch 或短序列，优化收益可能被控制逻辑开销抵消；
- 缺乏详细的内存 footprint 测量数据（文中提及但未展示具体数值）；
- 实验覆盖的模型种类有限，尚未涵盖所有 MoE 变体（如 hierarchical MoE）。

### 未来工作方向
- 扩展至更多模型架构和应用场景（如 multimodal MoE）；
- 进行更系统的 serving benchmark 和 memory profiling；
- 探索在其他支持 one-sided communication 的平台（如 NVLink + TensorRT-LLM）上的移植可能性；
- 结合动态路由预测进一步优化偏移计算和地址缓存机制；
- 引入编译器支持，自动融合 dispatch/combine 与 FFN 计算。

---

> 🔚 **总结一句话**：  
> 本文通过消除 MoE 推理中 dispatch/combine 的中间 relay 缓冲区，提出一种基于全局池化 HBM 的 relay-buffer-free 通信范式，在 Ascend 上实现了显著的 kernel-level 和 end-to-end 性能提升，为高效稀疏模型推理提供了新的系统设计思路。

</details>

---

### 4. [ResiHP: Taming LLM Training Failures with Dynamic Hybrid](https://arxiv.org/abs/2605.06374)

**Authors**: Tenghui Ma, Jihu Guo, Wei Gao, Sitian Lu, Zhisheng Ye, Hanjing Wang, Dahua Lin  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.06374v1  

#### Abstract
Hybrid parallelism underpins large-scale LLM training across tens of thousands of GPUs. At such scale, hardware failures on individual devices lead to performance skew across devices, diminishing overall training efficiency. Existing resilient systems overlook sequence length variability in datasets...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ResiHP: Taming LLM Training Failures with Dynamic Hybrid Parallelism —— 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在大规模 LLM 训练中，**硬件故障**（如 fail-stop 和 fail-slow）在数千甚至上万 GPU 集群中变得不可避免，导致严重的 **device performance skew**，显著降低训练效率。现有系统存在以下两大缺陷：

- **误报频发**：由于序列长度可变（sequence length variability），迭代时间（iteration time）自然波动，现有 fail-slow 检测机制易将正常波动误判为故障，引发不必要的验证开销。
- **适应策略低效**：现有系统通常只在单一维度（如 PP 或 DP）进行调整，无法协同优化 **hybrid parallelism** 的多个维度（TP/PP/DP），导致资源浪费、负载不均。

### 提出了什么新方法或新思路
本文提出 **ResiHP**，一个支持动态混合并行的容错训练系统，其核心创新在于：

- **轻量级、高精度的故障检测器（Detector）**：
  - 采用 **workload-aware execution time predictor**，通过建模 micro-batch 的计算成本（分离线性与二次项），准确预测“健康”迭代时间，从而过滤由序列长度变化引起的时序波动。
  - 结合在线时间序列分析，仅在实际执行时间显著偏离预测值时才触发验证，大幅减少误报。

- **渐进式、细粒度的调度器（Scheduler）**：
  - 在 **TP、PP、DP 三个维度** 上进行 **progressive adaptation**：
    - **TP 维度**：选择性排除故障设备，重构 TP group 大小，保留可用算力，避免整组丢弃。
    - **PP 维度**：动态重划分模型层（layer repartition），缓解因 TP 不均衡导致的流水线气泡。
    - **DP 维度**：基于各阶段进度（progress-aware）迁移微批次任务，平衡全局同步时间。

### 相比现有方法的优势
- **更高的检测准确性**：避免 workload-induced 误报，检测开销极低。
- **更强的资源利用率**：通过细粒度重构，最大化利用残存硬件资源。
- **更优的训练吞吐**：多维协同适应有效抑制故障传播，维持高训练效率。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 实验未直接使用传统 NLP 数据集，而是基于真实 LLM 训练场景，模拟具有 **多样化序列长度** 的输入分布（如开源 GitHub 数据集风格）。
- 采用 **sequence packing** 技术打包微批次，但仍保留注意力计算的二次复杂度差异。

### 实验设置和评估指标

#### 测试平台
- **集群配置**：32 节点 × 8 NVIDIA A100 GPU，节点间通过 200Gbps HDR InfiniBand 连接，节点内通过 NVSwitch 互联。
- **模型规模**：使用 **LLaMA2** 和 **Qwen 2.5** 系列模型，覆盖 7B 到 70B 参数规模。
- **并行策略**：采用多种 (TP, DP, PP) 配置，最大扩展至 **256 GPU**（TP=4, DP=4, PP=16）。

#### 故障注入方式
- **fail-stop**：手动终止 worker 进程。
- **fail-slow**：
  - 计算层面：使用 `nvidia-smi` 锁定 GPU SM 频率。
  - 通信层面：引入侧信道通信任务制造网络带宽竞争。

#### 评估指标
- **Detector 准确性**：
  - MAPE（Mean Absolute Percentage Error）用于评估 micro-batch 和 iteration time 预测精度。
  - 检测准确率、误报数（false alarms）、单次误报开销。
- **Scheduler 有效性**：
  - **端到端吞吐量**（throughput, samples/s）。
  - 与基线相比的加速比（speedup）。
- **系统开销**：检测、调度、重构等阶段的时间开销。

### 基线方法对比
| 基线方法 | 类型 | 主要功能 |
|--------|------|---------|
| **Greyhound [48]** | fail-slow 检测与缓解 | 基于迭代时间异常检测，DP 层面重分配微批次 |
| **Adaptra [47]** | fail-slow 缓解 | 优化 PP 层级调度 |
| **ReCycle [10]** | fail-stop 容忍 | PP 层面迁移失败任务 |
| **Oobleck [21]** | fail-stop 恢复 | 切换预定义 pipeline 模板 |
| **Strengthened ReCycle/Oobleck** | 混合故障处理 | 集成 Greyhound 的 fail-slow 检测能力 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 检测器性能（Detector）
- **micro-batch 时间预测 MAPE**：1.19% ~ 1.58%
- **iteration 时间预测 MAPE**：2.81% ~ 5.06%（见 Table 4）
- **fail-slow 检测准确率**：**>99.0%**
- **fail-stop 检测准确率**：**>99.6%**
- **误报数量显著降低**：相比 Greyhound，误报数从平均 3.7~8.7 降至 0~0.3（见 Table 5）
- **误报开销**：仅 **34–49ms**，远低于 Greyhound 的 **2.24–3.72s**

#### 调度器性能（Scheduler）
- 在 256-GPU 集群上，ResiHP 相比基线实现 **1.04–4.39× 吞吐提升**（见 Figure 10, 14）
- 在 fail-stop 场景下（每 30 分钟一次），ReCycle 和 Oobleck 因资源浪费而训练中断，**ResiHP 仍可持续运行**（见 Table 6）

### 与基线方法的对比结果
| 场景 | ResiHP vs. 基线 | 加速比范围 |
|------|----------------|-----------|
| **fail-stop**（频繁发生） | vs. ReCycle / Oobleck | **1.22–1.82× / 1.07–1.51×** |
| **fail-slow**（弱/中/强） | vs. Adaptra / Greyhound | **1.18–2.30× / 1.22–1.46×** |
| **混合故障**（交替发生） | vs. Strengthened ReCycle / Oobleck | **1.48–4.39× / 1.04–3.57×** |

> 注：在混合故障场景中，Strengthened ReCycle 改进有限，因其可能将任务迁移到已降速的设备上，加剧瓶颈。

### 消融实验结果（Ablation Study）
- **组件贡献排序**（以吞吐提升衡量）：
  1. **Selective Device Exclusion (TP)**：贡献最大，直接挽救 TP 资源，减少浪费。
  2. **Workload Migration (DP)**：细粒度平衡 DP 副本进度，灵活应对残余 skew。
  3. **Layer Repartition (PP)**：虽有效，但受限于跨 DP 副本的统一应用，灵活性较低。
- **故障传播抑制效果**：
  - ResiHP 将故障影响延迟从原始的 25.43×（DP 层）压缩至 **11.14×**，显著抑制了故障放大效应。

---

## 4. 关键结论和发现

### 主要发现
- **序列长度变化是 fail-slow 误检的主因**：现有检测机制缺乏 workload-aware 建模，导致高误报率。
- **单一维度适应不足以应对混合并行中的故障传播**：必须在 TP → PP → DP 路径上进行 **渐进式、协同式适应** 才能有效抑制性能 skew。
- **细粒度资源回收至关重要**：即使单个设备故障，也应尽可能保留其余设备参与计算，避免整组丢弃。
- **ResiHP 可保持训练收敛性**：在注入故障后，损失曲线与无故障训练高度一致，说明其 **不改变训练语义**，保证最终模型质量。

### 方法的局限性
- **依赖显式的性能信号**：目前主要针对有性能下降或心跳丢失的故障，对 **silent data corruption (SDC)** 等无显式信号的故障尚未覆盖。
- **P2P 通信优化依赖拓扑假设**：scatter/gather 优化要求 sender/receiver 具备兼容的 TP degree，异构程度过高时需额外协调。
- **重构过程仍有短暂停顿**：尽管开销低，但通信组重建和状态迁移仍需短暂暂停训练。

### 未来工作方向
- **集成 SDC 检测模块**：结合 loss spike、参数漂移等信号，构建统一的多类故障检测框架。
- **支持更复杂的异构并行策略**：如 Ulysses、Ring Attention 等新型 TP 模式。
- **自动化并行策略搜索**：结合 Alpa、TensorOpt 等 auto-parallelism 工具，动态生成最优并行计划。
- **扩展至弹性训练场景**：支持节点动态加入/退出，进一步提升资源利用率。

---

> ✅ **总结一句话**：  
> ResiHP 通过 **workload-aware 故障检测 + 多维动态 hybrid parallelism 适应**，实现了高精度、低开销、高吞吐的大规模 LLM 容错训练，在多种故障场景下相较现有系统提升 **1.04–4.39× 吞吐**，是迈向稳定、高效千亿级模型训练的重要一步。

</details>

---

### 5. [MDN: Parallelizing Stepwise Momentum for Delta Linear Attention](https://arxiv.org/abs/2605.05838)

**Authors**: Yulong Huang, Xiang Liu, Hongxiang Huang, Xiaopeng Lin, Zunchang Liu, Xiaowen Chu, Zeke Xie, Bojun Cheng  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.05838v1  

#### Abstract
Linear Attention (LA) offers a promising paradigm for scaling large language models (LLMs) to long sequences by avoiding the quadratic complexity of self-attention. Recent LA models such as Mamba2 and GDN interpret linear recurrences as closed-form online stochastic gradient descent (SGD), but naive...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MDN: Parallelizing Stepwise Momentum for Delta Linear Attention

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现有的 **Linear Attention (LA)** 模型虽然避免了传统 **Self-Attention** 的 $O(L^2)$ 复杂度，实现了 $O(L)$ 的线性计算效率，但其更新机制通常基于 **naive SGD**，导致以下问题：

- **信息衰减过快**：历史信息在长序列中迅速丢失。
- **优化收敛不佳**：缺乏对梯度噪声的鲁棒性，影响模型表达能力和上下文检索能力。
- **训练-推理不一致**：现有并行化方案（如 blockwise momentum）破坏了严格的时间因果性，导致训练和推理阶段状态不一致。

### **提出了什么新方法或新思路**

本文提出 **Momentum DeltaNet (MDN)**，其核心创新在于：

1. **Stepwise Momentum Rule**  
   将 **momentum-based optimizer** 引入 Linear Attention 的递归更新中，通过累积历史梯度来增强信息保留和优化稳定性。

2. **Chunkwise Parallel Algorithm**  
   设计了一种**几何重排序（geometric reordering）** 的并行算法，将 stepwise momentum 的递归形式转化为可在 chunk 内高效并行计算的形式，同时保持严格的因果性。

3. **Dynamical Systems 视角分析**  
   将 momentum 更新视为一个**二阶动力系统（second-order dynamical system）**，揭示其引入了复共轭特征值（complex conjugate eigenvalues），从而支持**阻尼振荡行为（damped oscillations）**，增强了模型的表达能力。

4. **稳定门控约束（Stable Gating Constraints）**  
   为防止数值不稳定（如 NaN），提出约束条件 $\beta \leq 1-\alpha$ 和 $\mu \in (e^{-1}, 1)$，确保特征值位于右半平面，维持系统稳定。

### **相比现有方法的优势**

| 方面 | MDN 的优势 |
|------|-----------|
| **表达能力** | 二阶系统支持振荡动态，优于传统一阶系统的纯衰减动态。 |
| **训练效率** | chunkwise 并行实现，训练吞吐量与 Mamba2、KDA 相当。 |
| **推理一致性** | 保持 stepwise 因果性，无训练-推理不匹配问题。 |
| **检索能力** | 显著提升 in-context retrieval 性能，尤其在长上下文场景。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **合成任务**：
  - **MQAR (Multi-Query Associative Recall)**：评估模型的上下文记忆与检索能力。
- **语言建模与下游任务**：
  - **WikiText-4K**, **LAMBADA**：评估语言建模能力（PPL）。
  - **Commonsense Reasoning**：HellaSwag, ARC-easy/challenge, PIQA, WinoGrande, BoolQ, SciQ。
  - **In-context Retrieval**：SWDE, SQuAD, FDA, NQ, TQA, DROP。
  - **长上下文建模**：
    - **LongBench**：评估代码生成、摘要、问答等任务在 16K 长度下的表现。
    - **Needle-In-A-Haystack (NIAH)**：从 RULER 基准测试中评估长程信息检索能力。

### **实验设置**

- **模型规模**：400M 和 1.3B 参数。
- **训练配置**：
  - 序列长度：4K。
  - 优化器：AdamW。
  - 学习率：余弦退火，峰值 $3\times10^{-4}$。
  - 训练步数：400M 模型训练 15B tokens，1.3B 模型训练 100B tokens。
- **硬件**：单张 H100 GPU，使用 Triton 实现高效 kernel。

### **基线方法对比**

| 基线模型 | 类型 |
|---------|------|
| **Transformer** | 自回归注意力模型（LLaMA 架构） |
| **Mamba2** | 基于 decay rule 的 SSM 模型 |
| **GDN (Gated DeltaNet)** | 基于 delta rule 的 LA 模型 |
| **Comba** | GDN 的改进版，引入闭环校正 |
| **KDA (Kimi Delta Attention)** | 向量化门控的 delta rule 模型 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **语言建模（400M / 1.3B 模型）**

| 模型 | Lamb. PPL ↓ | Wiki. PPL ↓ | Avg. Reasoning ↑ | In-context Retrieval Avg. ↑ |
|------|-------------|-------------|------------------|----------------------------|
| **MDN (Ours)** | **41.62** | **31.51** | **49.42** | **26.76** |
| KDA | 43.44 | 31.96 | 49.06 | 24.47 |
| Comba | 46.19 | 31.73 | 48.91 | 22.93 |
| GDN | 45.63 | 32.10 | 47.85 | 21.12 |
| Mamba2 | 60.42 | 33.45 | 47.85 | 30.42 |

> ✅ MDN 在 **所有指标上均优于或持平最强基线**，尤其在推理和检索任务上领先明显。

#### **长上下文建模（LongBench, 16K）**

| 模型 | Code | Summarization | Single QA | Multi QA | **Avg.** |
|------|------|---------------|-----------|----------|--------|
| **MDN (Ours)** | **50.50** | **18.85** | **6.87** | **15.65** | **20.18** |
| KDA | 37.99 | 16.95 | 6.58 | 25.30 | 18.62 |
| Comba | 44.74 | 16.74 | 6.87 | 5.79 | 18.11 |
| GDN | 42.94 | 16.82 | 5.89 | 26.27 | 19.28 |
| Mamba2 | 38.51 | 14.73 | 5.68 | 4.46 | 15.00 |

> ✅ MDN 在 **代码生成和摘要任务上显著领先**，平均得分最高。

#### **Needle-In-A-Haystack (NIAH) 检索任务**

在 8K 上下文长度下，MDN 在多针检索任务中大幅超越基线：

- **MK-NIAH**: **38.60** vs 最强基线 25.20 (+13.4)
- **MQ-NIAH**: **35.15** vs 最强基线 23.70 (+11.45)
- **MV-NIAH**: **27.60** vs 最强基线 18.65 (+8.95)

> ✅ 表明 MDN 能更有效地保留和检索长距离依赖信息。

### **消融实验结果**

| 变体 | Lamb. PPL | Wiki. PPL | Reasoning | Retrieval |
|------|----------|----------|----------|----------|
| **MDN (完整)** | 41.62 | 31.51 | 49.42 | **26.76** |
| w/o Output Corr. | 42.31 | 31.72 | 49.19 | 25.52 |
| w/o Momentum | 47.01 | 32.11 | 49.26 | 20.12 |
| w/o Clamp (log μ min) | NaN (发散) | NaN | NaN | — |
| w/o $\alpha_{\text{max}}$ | NaN (发散) | NaN | NaN | — |

> 🔍 关键发现：
> - **Momentum 是性能提升的关键**，移除后检索能力下降超 6 点。
> - **稳定约束至关重要**，缺少对 $\mu$ 或 $\alpha$ 的约束会导致训练崩溃。
> - 即使移除输出校正，MDN 仍优于 GDN 和 Comba，说明 momentum 本身具有强大增益。

---

## 4. 关键结论和发现

### **主要发现**

1. **Stepwise Momentum 显著提升 LA 模型性能**  
   momentum 机制有效缓解了信息衰减和梯度噪声问题，提升了模型的记忆力和推理能力。

2. **Chunkwise 并行是可行且高效的**  
   通过几何系数解耦，实现了 stepwise momentum 的高效并行训练，兼顾了**表达能力**与**训练效率**。

3. **二阶动力系统视角指导稳定设计**  
   特征值分析揭示了 momentum 引入的振荡动态，并指导了门控参数的稳定约束设计。

4. **MDN 在长上下文任务中表现卓越**  
   在 MQAR、NIAH、LongBench 等任务上全面领先，验证了其在长程依赖建模上的优势。

### **方法的局限性**

1. **训练吞吐量略低于 GDN/Comba**  
   由于需维护 momentum 状态，当前实现的训练 throughput 仍低于最先进的 first-order LA 模型。

2. **尚未在更大规模（如 7B+）上验证**  
   当前实验限于 400M 和 1.3B 模型，更大规模的扩展性有待验证。

3. **内存开销增加**  
   需存储额外的 momentum 状态，在大规模分布式训练中可能带来挑战。

### **未来工作方向**

1. **进一步优化 kernel 实现**  
   开发更高效的 Triton kernel，减少 memory footprint，提升训练吞吐。

2. **探索更复杂的优化器**  
   将 Nesterov Momentum、Adam 等更高级优化策略融入 LA 框架。

3. **扩展到多模态与时间序列**  
   将 MDN 应用于语音、视频、基因组学等需要长序列建模的领域。

4. **系统级优化**  
   探索与 tensor parallelism 更好的兼容性，支持超大规模训练。

---

> **GitHub 代码**：[github.com/HuuYuLong/MomentumDeltaNet](https://github.com/HuuYuLong/MomentumDeltaNet)

</details>

---

### 6. [UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification](https://arxiv.org/abs/2605.06221)

**Authors**: Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, Ran He  
**Category**: cs.CL  
**Published**: 2026-05-08  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.06221v1  

#### Abstract
As large language models (LLMs) continue to advance rapidly, they are becoming increasingly capable while simultaneously demanding ever-longer context lengths. To improve the inference efficiency of long-context processing, several novel low-complexity hybrid architectures have recently been propose...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **prefill 加速方法**（如基于稀疏注意力的方法）存在两大局限性：
1. **架构依赖性强**：仅在全注意力（full-attention-only）模型上有效，在新兴的混合架构（hybrid architectures）如 linear/full 或 sliding window/full attention 混合模型中加速效果显著下降。
2. **不兼容连续批处理（continuous batching）**：难以集成到现代推理引擎（如 vLLM），限制了其在生产环境中的实际部署。

### 🚀 提出的新方法：UniPrefill
提出一种**通用、架构无关**的 prefill 加速框架 **UniPrefill**，核心思想是：
- 在每个 **full attention 层**进行 **token importance 估计**，通过 **block-wise top-p 动态剪枝**识别并丢弃冗余 token。
- 将这种 **sparsity（稀疏性）传播至后续所有子层**（包括 linear attention、FFN、sliding window 等），实现跨层计算量压缩。

### 🔍 创新点与优势
| 特性 | 说明 |
|------|------|
| **Architecture-Agnostic** | 适用于任意模型架构（全注意力、linear/full 混合、sliding window/full 混合等） |
| **Token-Level Sparsification** | 在 token 粒度上操作，而非仅 attention matrix 稀疏化 |
| **Sparsity Propagation** | 单次 drop 决策影响后续所有层，同时减少 attention 和 GEMM FLOPs |
| **Continuous Batching 兼容** | 实现为 vLLM 中的 fused kernel operator，支持 prefill-decode co-processing 和 tensor parallelism |
| **无缝集成** | 无需修改模型权重或服务基础设施，可直接嵌入 vLLM |

相比现有方法（如 MInference、FlexPrefill、SnapKV），UniPrefill 不仅保持高精度，还能在混合架构上实现更优加速。

---

## 2. 核心实验方法和设置

### 📚 数据集
使用 **RULER** [11] 长上下文基准测试集：
- 覆盖 retrieval、multi-hop tracing、aggregation、QA 等任务
- 支持从 4K 到 128K 的可配置上下文长度
- 更全面地评估真实长上下文理解能力

### ⚙️ 实验设置
| 项目 | 设置 |
|------|------|
| **模型架构** | <ul><li>LLaMA-3.1-8B-Instruct（全注意力）</li><li>Qwen3-Next-80B-A3B（linear/full attention 混合，比例 3:1）</li><li>Gemma-3-12B（sliding window/full attention 混合，比例 5:1）</li></ul> |
| **上下文长度** | 4K, 8K, 16K, 32K, 64K, 128K |
| **Batch Size (BSZ)** | 1, 4, 16, 64 |
| **Tensor Parallelism (TP)** | 8 |
| **实现平台** | 基于 vLLM v0.16.0，使用 Triton fused kernels 实现 |
| **关键参数** | <ul><li>top-p 阈值：0.99（LLaMA）、0.99（Qwen）、0.98（Gemma）</li><li>重要性估计窗口 `n` = 128</li><li>block size `G` = 64</li><li>保留前 128 个 token（attention sinks）</li></ul> |

### 📊 评估指标
| 指标 | 说明 |
|------|------|
| **RULER Score** | 准确率，衡量任务性能保留情况 |
| **Time-To-First-Token (TTFT) Speedup** | 首 token 时间加速比 |
| **Prefill Throughput (tokens/s)** | 每秒处理的预填充 token 数量 |
| **Ablation Studies** | 分析 block size (`G`) 和 last `n` 查询数的影响 |

### 🆚 对比的基线方法
| 方法 | 类型 | 是否兼容 CB | 在混合架构上有效？ |
|------|------|-------------|------------------|
| **Baseline** | 标准 full attention | 是 | 是 |
| **LazyLLM [9]** | 动态 token 剪枝 | 否 | 有限 |
| **SlimInfer [20]** | 动态 token 剪枝 | 否 | 有限 |
| **MInference [13]** | Sparse Attention | 否 | ❌（仅对 full attention 有效） |
| **FlexPrefill [16]** | Context-aware Sparse Attention | 否 | ❌ |
| **ProxyAttn [27]** | Representative Head 引导稀疏 | 否 | ❌ |
| **SnapKV [19]** | KV Cache 压缩（decode 阶段） | 是 | ❌（不影响 prefill FLOPs） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & Figure 1）

#### ✅ RULER 准确率表现
- UniPrefill 在所有模型和上下文长度下均**几乎无精度损失**：
  - LLaMA-3.1-8B @128K: **79.87 vs Baseline 76.89**
  - Qwen3-Next-80B @128K: **91.41 vs 92.09**
  - Gemma-3-12B @128K: **58.38 vs 61.22**

> ➤ 表明 UniPrefill 能有效保留关键语义信息。

#### ⚡ TTFT 加速比（vs Baseline）
| 模型 | 8K | 16K | 32K | 64K | 128K | 最大加速 |
|------|----|-----|-----|-----|-------|--------|
| **LLaMA-3.1-8B** | 1.21× | 1.34× | 1.37× | 1.62× | **2.26×** | **2.26×** |
| **Qwen3-Next-80B** | 1.08× | 1.21× | 1.24× | 1.39× | **1.68×** | **1.68×** |
| **Gemma-3-12B** | 1.15× | 1.21× | 1.22× | 1.26× | **1.49×** | **1.49×** |

> ➤ 加速随 context length 增加而增强，尤其在 **128K 达到最高 2.26×**。

#### 🚀 Prefill Throughput 提升（Table 2）
在 vLLM 中实测吞吐量提升（TP=8）：
- **LLaMA-3.1-8B @128K, BS=64**: **+109%** 吞吐提升
- **Qwen3-Next-80B @128K, BS=64**: **+68%**
- **Gemma-3-12B @128K, BS=64**: **+42%**

> ➤ 吞吐增益随 batch size 和 context length 上升而扩大，适合高并发场景。

### 🔬 消融实验结果（Ablation Study）

#### ▶️ Block Size $ G $
| $ G $ | 优势 | 缺陷 |
|-------|------|------|
| **32** | 细粒度，drop 更多 token → 更高吞吐（@128K +121%） | 开销略高 |
| **64** | ✅ 默认选择，平衡开销与收益 |
| **128** | 低决策频率，短序列更快 | 长序列 drop 不够精细 |

> ➤ 推荐 $ G=64 $ 作为通用配置。

#### ▶️ Last $ n $ 查询数量
| $ n $ | RULER Score (@128K) | 说明 |
|-------|------------------------|------|
| 32 | 75.13 | 明显下降（方差大） |
| 128 | **79.87** | ✅ 最佳平衡点 |
| 512 | 79.38 | 精度恢复但计算开销上升 |

> ➤ $ n=128 $ 是最优选择。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **UniPrefill 实现了真正的架构通用 prefill 加速**：
   - 在全注意力和各类混合架构上均取得显著加速（最高 **2.26× TTFT**）。
2. **sparsity propagation 是关键机制**：
   - 单次 token drop 可级联减少后续所有层的计算量，远超仅优化 attention 子层的传统方法。
3. **系统集成至关重要**：
   - 成功将算法实现为 vLLM 的 continuous batching operator，支持 tensor parallelism 和 prefill-decode co-processing，具备**生产可用性**。
4. **加速效果随负载增长而放大**：
   - 在长 context、大批量、高并发场景下优势最明显，契合现实生产需求。

### ⚠️ 局限性
1. 当前聚焦于 **prefill 阶段加速**，未涉及 decoding 优化。
2. 虽然精度损失极小，但在极端复杂任务中仍可能存在边缘 case 影响。
3. 依赖 full attention 层的存在来触发 importance 估计，纯 linear 或 state space 模型无法直接受益。

### 🔮 未来工作方向
1. 扩展至 **decoding 阶段的动态 token 管理**。
2. 探索训练时引入 sparsity learning，进一步提升压缩率。
3. 应用于 **多模态长序列建模**（如视频、音频）。
4. 结合 speculative decoding 等技术实现端到端推理加速。

---

> 💡 **总结一句话**：  
> **UniPrefill 是首个真正实现“架构无关 + 生产就绪”的 prefill 加速方案，通过 block-wise dynamic sparsification 与 sparsity propagation，在几乎不失效的前提下，为长上下文 LLM 推理带来高达 2.26× 的 TTFT 加速，极具实用价值。**

</details>

---

### 7. [Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key](https://arxiv.org/abs/2605.06638)

**Authors**: Tianle Wang, Zhaoyang Wang, Guangchen Lan, Xinpeng Wei, Sipeng Zhang, Guanwen Qiu, Abulhair Saparov  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.06638v1  

#### Abstract
Reinforcement learning (RL) has been applied to improve large language model (LLM) reasoning, yet the systematic study of how training scales with task difficulty has been hampered by the lack of controlled, scalable environments. We introduce ScaleLogic, a synthetic logical reasoning framework that...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在使用 **Reinforcement Learning (RL)** 对 **Large Language Models (LLMs)** 进行推理能力后训练时，缺乏一个**可控且可扩展的环境**来系统研究任务难度（尤其是长推理链）如何影响训练效率和下游泛化能力。现有的推理任务（如数学、编程）虽然具有可验证性，但在**推理深度（horizon）和逻辑表达力（expressiveness）上难以独立控制**，导致无法精确分析 RL 的缩放规律。

### 提出了什么新方法或新思路
作者提出了 **SCALELOGIC** —— 一种**合成的、可控的逻辑推理框架**，具备以下核心特性：
- **独立控制两个难度维度**：
  - **推理深度（Reasoning Depth, D）**：所需证明步骤的数量。
  - **逻辑表达力（Logical Expressiveness）**：从仅含“如果-那么”（implication-only）到包含合取（conjunction）、析取（disjunction）、否定（negation）和全称量词（universal quantification）的一阶逻辑。
- **完全可验证性**：每个问题都有唯一正确答案，可通过形式化验证器（Z3）自动验证。
- **低成本自动生成**：支持无限规模的数据生成，适用于大规模 RL 训练。

### 相比现有方法的优势
| 特性 | SCALELOGIC | 现有方法（如 Math/Code, SAT, Knights and Knaves） |
|------|------------|---------------------------------------------|
| 可验证性 | ✅ 显式可验证 | ✅ 部分可验证 |
| 可扩展性 | ✅ 自动生成，无限数据 | ⚠️ 依赖人工标注或有限模板 |
| 推理深度控制 | ✅ 精确控制 | ❌ 或间接控制 |
| 逻辑表达力控制 | ✅ 独立调节 | ❌ 固定或不可控 |

SCALELOGIC 是首个同时满足 **Verifiable、Scalable、Controllable Horizon、Controllable Expressiveness** 的 RL 推理训练框架（见 Table 1）。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **SCALELOGIC**：本文提出的新合成数据集，用于 RL 后训练。
  - 包含 5 种逻辑表达力层级：
    1. `→` Implication-only
    2. `→∧` + Conjunction
    3. `→∧∨¬` + Negation
    4. `→∧∨¬∨` + Disjunction
    5. `→∧∨¬∨∀` + Quantification（最复杂）
  - 每个实例为多选题：给定一组公理，判断哪个候选结论是**唯一可推导出的**。
- **下游评测基准（共 8 个）**：
  - 数学类：AIME 2024/2025, AMC2023, MATH-500, Minerva
  - 综合推理类：OlympiadBench (text-only), GPQA-Diamond, MMLU-Pro (STEM subset)

### 实验设置和评估指标

#### 主要模型
- **主模型**：Qwen3-4B（non-thinking 版本）
- **跨尺度验证**：部分实验复现于 Qwen3-8B

#### RL 训练设置
- **算法**：DAPO（基于 GRPO 的改进版）
- **奖励设计**：二值奖励（R=1 若答案格式正确且匹配；否则 R=0）
- **训练目标**：达到 **90% 验证准确率** 所需的 **RL 训练步数（T）**

#### 评估指标
- **主指标**：
  - **T ∝ D^γ**：训练步数随推理深度 D 的幂律关系，拟合指数 γ。
  - **下游平均准确率**：在 8 个基准上的 Avg@8 平均值。
- **其他指标**：
  - OOD 泛化：测试未见过的更深推理任务。
  - 控制变量：固定训练步数或固定推理深度，比较不同表达力的影响。

#### 基线方法对比
- **无 RL 训练的 Base Model**
- 不同逻辑表达力设置下的训练效果对比
- 不同训练策略：Uniform Sampling vs. Curriculum vs. Difficult-only
- 不同 RL 算法：DAPO vs. GRPO vs. GSPO

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### (1) 训练成本遵循幂律缩放
- 在所有 5 种逻辑表达力下，训练步数 $ T $ 与推理深度 $ D $ 满足幂律关系：  
  $$
  T \propto D^\gamma,\quad R^2 > 0.99
  $$
- **幂律指数 γ 随逻辑表达力单调递增**：
  | 表达力 | γ 值 |
  |--------|------|
  | Implication-only | 1.04 |
  | +Conjunction | 1.72 |
  | +Negation | 1.81 |
  | +Disjunction | 2.11 |
  | +Quantification | **2.60** |

> 结论：**更复杂的逻辑需要不成比例地更多训练计算资源**。例如，+Quantification 下深度翻倍，训练成本增加约 6 倍（$2^{2.6}$），而简单逻辑仅增加 2 倍。

#### (2) 下游推理性能显著提升
- 在 8 个下游基准上的平均准确率从 Base Model 的 **49.39%** 提升至最高 **60.05%**，**绝对增益达 +10.66 个百分点**。
- 更高表达力训练带来更大且更持续的收益：
  - 简单逻辑（如 implication-only）很快饱和（~52%）
  - 最复杂逻辑（+Quantification）在整个训练过程中持续提升

#### (3) 控制实验验证表达力的关键作用
- **固定推理深度（D=12）**：
  - 下游增益从 +0.49（implication-only）上升到 **+8.10**（+Quantification）
- **固定训练步数（~100 步）**：
  - 增益从 +2.32 上升到 **+6.33**
> 表明：**训练数据的“质量”（表达力）比“数量”更重要**。

#### (4) 消融实验结果

##### (a) 训练策略影响显著
- **Curriculum Learning** 显著改善缩放效率：
  - +Conjunction 设置下，γ 从 1.70（uniform）降至 **1.33**
  - +Quantification 下，γ 从 2.60 降至 **2.30**
- **Difficult-only** 训练最难，γ 高达 2.36，方差大，不稳定。

##### (b) 多种 RL 算法均呈现相同幂律行为
- 在 +Conjunction 设置下测试三种算法：
  - DAPO: γ = 1.70
  - GSPO: γ = 1.65
  - GRPO: γ = 2.05（样本效率更低）
> 表明幂律关系是**普遍现象**，不依赖特定优化器。

##### (c) OOD 泛化有限
- 即使训练到深度 D=14，模型在测试深度超过 **3×D_train** 时性能退化至随机水平。
> 表明：**训练能扩展有效推理范围，但不能消除“horizon limit”**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **RL 训练成本与推理深度呈幂律关系**：$ T \propto D^\gamma $
2. ✅ **幂律指数 γ 随逻辑表达力单调递增**：表达力越强，训练越难缩放。
3. ✅ **训练数据的表达力决定下游迁移效果**：更丰富的逻辑结构带来更大、更高效的性能增益。
4. ✅ **Curriculum Learning 显著提升训练效率**：通过渐进式难度引入，降低缩放指数。
5. ✅ **该现象在多种 RL 算法上具有一致性**：非特定算法产物。
6. ✅ **OOD 泛化存在硬边界**：最大可解深度约为训练深度的 3 倍。

> 📌 **核心洞见**：**What a model is trained on, not just how much it is trained, shapes downstream transfer.**  
> 数据的**结构性丰富度**（expressiveness）是决定 RL 推理训练成败的关键因素。

### 方法的局限性
- **模型规模限制**：主要实验在 Qwen3-4B 上进行，虽在 8B 上复现趋势，但仍需验证是否适用于更大模型。
- **表达力覆盖有限**：未包含等式（equality）、高阶逻辑、非单调推理等更复杂结构。
- **理论解释缺失**：观察到幂律和 γ 增加现象，但缺乏形式化理论解释其成因。
- **单一任务形式**：所有任务均为多选题，可能限制对自由生成式推理的建模。

### 未来工作方向
1. **探索更大模型和更广训练范式下的缩放规律**。
2. **扩展表达力层级**：加入 equality、higher-order reasoning、multi-agent relational structures。
3. **建立形式化理论**：解释为何不同逻辑操作符会改变缩放指数 γ。
4. **设计更高效的 curriculum 和 sampling 策略**，进一步降低训练成本。
5. **将 SCALELOGIC 思路迁移到真实世界任务构造中**，实现可控的真实推理数据生成。

--- 

> 🔚 **总结**：SCALELOGIC 提供了一个前所未有的**受控实验室**，揭示了 RL 推理训练中“表达力”的核心地位。它不仅是一个工具，更是一种**科学方法论**——让我们能够像物理学家研究材料性质一样，系统地探究 LLM 推理能力的本质边界与增长规律。

</details>

---

### 8. [Long Context Pre-Training with Lighthouse Attention](https://arxiv.org/abs/2605.06554)

**Authors**: Bowen Peng, Subho Ghosh, Jeffrey Quesnelle  
**Category**: cs.CL  
**Published**: 2026-05-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.06554v1  

#### Abstract
Training causal transformers at extreme sequence lengths is bottlenecked by the quadratic time and memory of scaled dot-product attention (SDPA). In this work, we propose Lighthouse Attention, a training-only symmetrical selection-based hierarchical attention algorithm that wraps around ordinary SDP...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Long Context Pre-Training with Lighthouse Attention》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在长上下文（如 128K、1M 及以上）的 LLM 预训练中，**scaled dot-product attention (SDPA)** 存在 $O(N^2)$ 的计算和内存开销，成为训练瓶颈。尽管 FlashAttention 等优化技术缓解了常数项开销，但无法改变其二次复杂度。

现有稀疏注意力方法（如 MoBA、DSA、HISA）虽然提升了推理效率，但在**训练场景下存在以下缺陷**：
- **不对称压缩**：仅压缩 Key 和 Value，Query 保持全分辨率，导致表示不一致。
- **架构耦合**：选择逻辑嵌入到自定义稀疏 attention kernel 中，难以复用高度优化的 dense kernel（如 FlashAttention）。
- **训练正确性存疑**：训练时使用稀疏注意力，最终模型是否仍能胜任 full attention 推理？缺乏验证机制。

### 提出了什么新方法或新思路
提出 **Lighthouse Attention**，一种专为长上下文预训练设计的**对称选择式分层注意力机制**，具有以下三大创新：

#### （i）子二次的分层预处理与后处理
通过构建多级金字塔结构，对 Query、Key、Value 进行**对称平均池化**（symmetric average-pooling），实现序列的自适应压缩与解压。该过程是梯度无关的，避免复杂的反向传播核。

#### （ii）对称压缩策略
在每一层级上同时池化 Q、K、V，形成 $(Q^{(l)}, K^{(l)}, V^{(l)})$ 三元组，保证左至右因果性的同时极大提升并行性。这使得压缩不仅是“可寻址的记忆”，而是真正的**多尺度表示**。

#### （iii）两阶段训练范式
- **第一阶段**：使用 Lighthouse Attention 进行长上下文预训练，显著加速。
- **第二阶段**：移除 Lighthouse 模块，以标准 full SDPA 继续微调少量步数，恢复完整注意力能力。

此设计允许在训练中享受稀疏带来的速度优势，而在推理前无缝切换回 dense 模型。

### 相比现有方法的优势
| 特性 | Lighthouse | 其他稀疏方法（如 NSA, DSA, HISA） |
|------|------------|-------------------------------|
| 是否对称压缩 Q/K/V | ✅ 是 | ❌ 否（通常只压缩 KV） |
| 是否复用 stock FlashAttention | ✅ 是（selection 在 kernel 外） | ❌ 否（需定制稀疏 kernel） |
| 是否无额外参数/损失 | ✅ 是（parameter-free scorer） | ❌ 否（常需 learnable indexer） |
| 是否支持 end-to-end 恢复 full attention | ✅ 是（通过 SDPA resume） | ❌ 否（模型被锁定为稀疏结构） |
| 训练正确性验证机制 | ✅ 明确测试恢复后的性能 | ⚠️ 缺乏系统验证 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **C4 dataset**：用于大规模语言建模预训练。
- **合成任务**：**Needle-in-a-Haystack (NIAH)**，用于评估长距离信息检索能力。

### 实验设置
- **模型架构**：530M 参数的 Llama-3-style decoder（30 层，head dim=128，FFN=1536）。
- **序列长度**：主实验为 98,304；扩展至 256K 和 1M 使用 Context Parallelism (CP)。
- **硬件平台**：NVIDIA B200 GPU（单节点 8×B200，多节点最多 32×Blackwell GPU）。
- **训练总预算**：固定为 16,000 步（约 50.3B tokens），比较不同配置下的 end-to-end 时间与最终性能。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Training Loss** | 主要优化目标，衡量模型拟合能力 |
| **Validation Loss** | 泛化性能 |
| **Tokens/s per GPU** | 吞吐量，反映训练效率 |
| **B200-Hours** | 总耗时 × GPU 数量，衡量端到端成本 |
| **NIAH Retrieval Rate** | 在指定位置准确提取“针”的概率，评估长程依赖捕捉能力 |

### 基线方法对比
- **Dense SDPA Baseline**：全程使用 full attention（cuDNN 实现），作为性能上限。
- **Lighthouse → SDPA Resume**：前段用 Lighthouse，后段切换为 dense SDPA，进行公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 配置 | 最终 Loss | Tokens/s (k) | B200-Hours | Speedup vs Baseline |
|------|-----------|----------------|-------------|---------------------|
| **Dense SDPA Baseline** | 0.7237 | 45.6 | 303.2 | 1.0× |
| **LH→SDPA (10k+6k)** | **0.6980** | 75.0 | 228.0 | **1.33×** |
| **Best Ablation (k=1536, p=2)** | **0.6825** | 93.9 | 203.9 | **1.49×** |
| **Projection-Norm Scorer** | ~0.69 | up to **126.0** | down to **179.6** | **1.69×** |

> 注：所有 Lighthouse 配置均在相同 token 预算下达到 **更低 loss + 更快训练速度**

### 与基线方法的对比结果
- 所有 Lighthouse 配置在完成 **dense SDPA resume** 后，**训练 loss 均优于或匹配从头训练的 dense baseline**。
- 在 N=512K 时，Lighthouse 的 forward 比 dense SDPA 快 **21×**，forward+backward 快 **17.3×**。
- 等效地，Lighthouse 在 512K 上的运行时间 ≈ dense SDPA 在 ~113K 上的时间。

### 消融实验结果（Ablation Studies）
#### （1）Scorer 类型
| Scorer | Loss (k=1536) | B200-Hours | Tok/s |
|--------|---------------|------------|-------|
| Dilated Softmax | 0.6881 | 197.2 | 99.5k |
| **Projection-Norm (l2)** | **0.6946** | **179.6** | **126.0k** |

✅ 结论：**projection-norm 虽略逊于 dilated 在 loss 上，但速度快 ~9%，且无需额外参数，性价比更高。**

#### （2）Pooling Factor $p$
- $p=2$: 0.6825
- $p=4$: 0.6881
- $p=8$: 0.6828  
➡️ 小 $p$ 更优，$p=2$ 为默认。

#### （3）Level 数 $L$
- $L=3$: 0.6825
- $L=4$: 0.6978
- $L=5$: 0.6991  
➡️ **更深的金字塔反而更差**，因选择预算分散至粗粒度层，影响细粒度敏感性。**L=3 最佳**。

#### （4）Top-K 预算 $k$
- $k=1536$: 0.6825
- $k=2048$: 0.6880
- $k=4096$: 0.6951
- $k=6144$: 0.6831  
➡️ 并非越大越好！**较小 $k$ 表现更好**，可能因起到了正则化作用。**k=1536 为帕累托最优**。

#### （5）Needle-in-a-Haystack 检索任务
| 配置 | 平均 Retrieval Rate |
|------|--------------------|
| Dense SDPA Baseline | 0.72 |
| Lighthouse (k=2048, dilated) | **0.76** ✅ |
| Lighthouse (k=1536, norm) | 0.65 ❌ |

⚠️ 发现：**loss 低 ≠ retrieval 强**。norm scorer 对 retrieval 损伤更大，说明 scorer 设计需根据下游任务权衡。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Lighthouse 可安全用于长上下文预训练**：经过短暂 dense resume 后，模型不仅能恢复 full attention 能力，还能**超越从头训练的 dense baseline**。
2. ✅ **训练效率大幅提升**：相比 dense SDPA，在 ≥100K 上下文下实现 **1.4–1.7× 的端到端加速**。
3. ✅ **模块化设计优越**：selection 与 attention 解耦，可直接复用 FlashAttention，兼容性强。
4. ✅ **对称压缩有效**：Q/K/V 同步池化构建真正多尺度表示，优于传统 KV-only 压缩。
5. ⚠️ **并非越“稀疏”越差**：小 $k$ 配置表现更优，暗示**分层选择本身具有正则化效应**。

### 方法的局限性
- ❗ **依赖 dense resume 才能获得推理可用模型**：当前版本不能直接用于 autoregressive decoding，因为对称压缩破坏了逐 token 生成假设。
- ❗ **inner attention 仍是 $O(S^2d)$**：虽为 sub-quadratic，但非严格线性；若 $k$ 需随 $N$ 增长，则优势减弱。
- ❗ **目前仅适用于预训练**：未集成 KV-cache 管理、连续批处理等 serving 机制。

### 未来工作方向
- 🔮 **替换 dense resume 为目标稀疏结构**：例如将最终模型转为 DSA 或 MoBA，实现原生可服务 checkpoint。
- 🔮 **动态自适应 $k$**：按层或按 head 动态调整 selection budget。
- 🔮 **跨模态扩展**：将多尺度金字塔应用于 vision、audio、video 等序列。
- 🔮 **部署集成**：结合 speculative decoding、KV-cache pruning 等技术，将训练加速转化为实际推理收益。

---

> 📌 **一句话总结**：  
> **Lighthouse Attention 通过“对称分层选择 + 解耦 attention kernel + 两阶段训练”，实现了长上下文预训练的速度突破，并首次系统验证了“稀疏训练 → 密集恢复”的可行性，在保持甚至提升性能的前提下达成 1.7× 加速。**

</details>

---

### 9. [ADELIA: Automatic Differentiation for Efficient Laplace Inference Approximations](https://arxiv.org/abs/2605.06392)

**Authors**: Afif Boudaoud, Lisa Gaedke-Merzh\"auser, Alexandros Nikolaos Ziogas, Vincent Maillou, Alexandru Calotoiu, Marcin Copik, H{\aa}vard Rue, Mathieu Luisier, Torsten Hoefler  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.06392v1  

#### Abstract
Spatio-temporal Bayesian inference drives environmental and health sciences using latent Gaussian models. Integrated Nested Laplace Approximations (INLA) enable inference for these models at HPC scale but rely on derivative-based optimization over $d$ hyperparameters. State-of-the-art INLA implement...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ADELIA: Automatic Differentiation for Efficient Laplace Inference Approximations —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

- **背景**：在大规模时空贝叶斯建模中，Integrated Nested Laplace Approximations (**INLA**) 是一种高效的推理框架，广泛应用于环境科学、健康监测等领域。
- **挑战**：
  - INLA 的优化阶段依赖对 `d` 个超参数（hyperparameters）的梯度计算。
  - 当前最先进的实现（如 R-INLA 和 DALIA）使用 **Central Finite Differences (FD)** 近似梯度，需要 `2d+1` 次函数评估，导致计算开销随 `d` 线性增长。
  - 对于多变量模型（multivariate models），`d` 可达 15 或更高，使得 FD 在收敛性和效率上严重受限。
  - 此外，标准 **Reverse-mode Automatic Differentiation (AD)** 虽然理论上可独立于 `d` 计算精确梯度，但在具有 Block-Tridiagonal Arrowhead (**BTA**) 结构的大规模稀疏矩阵上难以高效应用。

> **核心问题**：如何在保持 BTA 结构稀疏性的前提下，实现高效、可扩展、精确的 AD 来替代 FD？

---

### 🚀 **提出了什么新方法或新思路**

作者提出 **ADELIA** —— 首个支持 **结构感知反向传播（structure-exploiting reverse-mode AD）** 的 INLA 实现，具备以下四大创新点：

#### 1) **结构化稀疏性感知的梯度计算（Structure-exploiting gradient computation）**
- 设计了针对 BTA 精度矩阵（precision matrix）的自定义微分规则。
- 将 Cholesky 分解、三角求解和 Selected Inversion（SI）融合为一个前向过程，仅保留紧凑的中间状态（Schur complement carries 和 solve vectors），减少内存占用约 **2×**。
- 在反向传播中按需重建 Cholesky 因子（reconstruction-on-the-fly），避免存储密集的中间变量。

#### 2) **支持多变量联合建模（Extension to multivariate models）**
- 扩展至建模 `k` 个相关响应变量（如多种污染物浓度）。
- 利用协区域化权重矩阵（coregionalization weight matrix）分离每变量与跨变量超参数的影响，降低每步计算复杂度。
- 支持最多 `k=3`, `d=15` 的生产级多变量模型。

#### 3) **多 GPU 分布式 AD（Multi-GPU distributed AD）**
- 采用 Serinv 的两阶段域分解策略（two-phase domain decomposition）：
  - **Phase 1**：各 GPU 并行处理内部时间块链；
  - **Phase 2**：通过 MPI allgather 同步边界块，本地冗余求解缩减系统。
- 引入 **CPU-staged Schur carries**，将中间状态暂存到主机内存，突破单卡显存限制。
- 成功运行高达 **1.9M latent variables** 的模型，远超单 GPU 容量。

#### 4) **完整的端到端评估分析框架**
- 不仅报告性能加速，还提供：
  - 框架效应分解（framework vs algorithmic gains）
  - 时间/能量效率对比
  - 内存瓶颈定位
  - 收敛质量比较

---

### 🔍 **相比现有方法的优势**

| 维度 | FD (DALIA/R-INLA) | ADELIA |
|------|-------------------|--------|
| **梯度精度** | 近似（O(h) 截断误差） | **精确梯度**（无截断误差） |
| **每梯度评估次数** | `2d+1` 次 | **1 次前向 + 1 次反向** |
| **可扩展性** | 并行但总能耗高 | 单次调用即可完成，天然节能 |
| **大模型支持** | 易 OOM，收敛困难 | 支持 **1.9M latent vars**, 多 GPU 扩展 |
| **收敛质量** | 多变量模型常停滞 | 实现低梯度范数、良好条件 Hessian |

> ✅ **结论**：ADELIA 不仅是“更快”，更是“唯一能可靠收敛”的选择。

---

## 2. 核心实验方法和设置

### 📊 **使用的数据集与模型**

共测试 **10 个基准模型**，涵盖合成与真实世界场景：

| 类型 | 模型 | 应用 | 特征 |
|------|------|------|------|
| 合成小规模 | GST-S | 单变量 | `n=5`, `b=92`, `d=4` |
| 中等规模 | GST-M, GST-L | 单变量 | 最大 `1.0M` latent vars |
| 温度预测 | GST-T | 日气温监测 | `365天×2865站点` |
| 多变量合成 | GST-C2 (`k=2`), GST-C3 (`k=3`) | 协同建模 | `d=9/15` |
| 真实空气污染 | AP1 | 北意大利 PM2.5/O₃ | `48天×12,630空间节点` |
| 生产级基准 | SA1, WA1, WA2 | 大规模时空建模 | `d=15`, latent vars 达 **1.9M** |

---

### ⚙️ **实验设置**

- **硬件平台**：CSCS Alps 超算，基于 **NVIDIA GH200 Grace Hopper 节点**
  - 每节点：4× Hopper GPU (96 GiB HBM3)，72核 ARM CPU (480 GiB LPDDR5X)
  - 多节点间通过 HPE Slingshot-11 互联
- **软件栈**：
  - ADELIA：基于 **JAX** + `jax.custom_vjp` + `mpi4jax` 实现 AD
  - Baseline：**DALIA (CuPy + Serinv GPU solver)** 使用 FD
- **对比方式**：
  - 相同模型配置、相同初始值、固定迭代次数（以排除收敛路径差异影响）

---

### 🎯 **评估指标**

| 指标类别 | 具体指标 |
|---------|----------|
| **性能** | 每梯度耗时（per-gradient time）、端到端 wall-clock time |
| **资源效率** | 峰值 GPU 内存、跨 GPU 扩展能力、所需最小 GPU 数（Pmin） |
| **能源效率** | 总能耗（node-level power meter @10Hz） |
| **准确性** | 梯度相对误差（vs FD / AD-Loop）、收敛轨迹（objective & ∥∇f∥） |
| **算法优势分解** | 成本比 `c_AD = T_AD / t_eval`，理论加速上限 `(2d+1)/c_AD` |

---

### 🆚 **基线方法对比**

| 方法 | 描述 |
|------|------|
| **FD** | DALIA 中央有限差分（2d+1 evaluations） |
| **AD-Dense** | 密集 Cholesky + 标准 AD（OOM） |
| **AD-Loop** | BTA loop + 存储所有中间状态（OOM） |
| **AD-Ckpt** | Checkpointing 技术缓解内存压力（仍 OOM） |
| **ADELIA (Ours)** | 自定义 backward + carry reconstruction + CPU staging |

---

## 3. 主要实验结果和性能指标

### 📈 **关键性能数据**

#### ✅ **每梯度速度提升（Per-gradient speedup）**

| 模型 | ADELIA vs FD 加速比 | 备注 |
|------|---------------------|------|
| GST-S | **2.4×** | 小模型，XLA kernel fusion 发挥优势 |
| GST-T | **2.7×** | 实际温度模型 |
| SA1 | **7.9×** | 生产级，单 GPU |
| AP1 / WA1 / WA2 | **4.2–6.4×** | 分布式模型（4 GPUs） |
| **总体范围** | **2.4–50.9×** | 取决于 `d` 和框架效率 |

> 💡 注：理论最大加速为 `2d+1`；实际受 `c_AD` 影响。

#### ✅ **端到端运行时间缩短**

| 模型 | 端到端加速比 | 实际节省时间 |
|------|-------------|------------|
| SA1 | **7.3×** | 从 **14.6h → 2.0h** |
| WA2 | **5.0×** | 从 **9.6h → 2.7h** |
| 总计 | — | **累计节省 26 小时** |

#### ✅ **Hessian 计算加速**

- 使用 AD gradients 构造 Hessian（via finite diff of grads）：
  - 替代传统 `2d²+1` 次目标函数调用
  - 仅需 `2d+1` 次 AD gradient 调用
  - 实测 Hessian 阶段提速 **1.1–23.3×**

#### ✅ **内存节省**

| 模型 | ADELIA 峰值 GPU 内存 | FD 或其他 AD 方法 |
|------|------------------------|------------------|
| GST-L | **63.3 GiB** | AD-Loop/Dense 均 OOM |
| GST-T | **46.5 GiB** | 同样 OOM |
| WA1/AP1 | **< O(b) per GPU** | 得益于 CPU staging |

> ✅ ADELIA 是**唯一能在单卡或分布式下运行百万变量模型的方法**

#### ✅ **能源效率碾压**

即使将 FD 扩展到 **32–128 GPUs** 来匹配 ADELIA 的 wall-clock 时间：

| 场景 | 能源消耗对比 |
|------|--------------|
| SA1 (8–16 GPUs) | FD 消耗 **5–8× 更多能量** |
| WA1 (128 GPUs) | FD 快 4.2×，但仍耗能 **7.3× 更多** |

> ⚠️ 并行无法解决能耗问题，且不能改善梯度质量！

---

### 🔬 **消融实验与框架效应分解**

#### 🔄 **Framework Effect Decomposition（图6）**

引入比率 `r = t_Serinv / t_JAX` 衡量底层框架效率：

| 模型类型 | `r` 值 | 说明 |
|--------|-------|------|
| 小模型（GST-S） | `r > 1` | JAX/XLA 编译更优，放大算法增益 |
| 中等计算密集型 | `r ≈ 0.9–1.0` | 两者接近持平 |
| 分布式模型 | `r = 0.1–0.25` | Serinv 更好重叠 CPU-GPU 传输，JAX 同步调度略慢 |

> ❗ 当前 ADELIA 的部分性能损失源于 **工程层面限制**，非 AD 本质缺陷，未来可通过优化调度进一步提升。

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **Exact gradients are essential for convergence**
   - 在大型多变量模型（如 AP1, SA1）上，FD 因累积误差导致梯度范数停滞在 `10³–10⁴` 级别，而 ADELIA 可降至 `~1.5`。
   - FD 得到的 Hessian 常非正定，无法用于不确定性量化。

2. **Structure-exploiting AD is necessary, not optional**
   - 所有通用 AD 方法（AD-Dense, AD-Loop, AD-Ckpt）在百万变量级别全部 OOM。
   - ADELIA 是目前**唯一可行路径**。

3. **Algorithmic gain dominates performance win**
   - 减少 `2d+1 → 1` 次评估是根本原因。
   - 即使在框架劣势下（如分布式），仍取得显著加速。

4. **Energy efficiency favors AD overwhelmingly**
   - FD 通过增加 GPU 数可缩短时间，但总能耗不降反升。
   - 科学计算应追求“绿色推理”。

5. **Scalability proven up to 1.9M latent variables**
   - 成功部署于 4-GPU 配置，验证了多 GPU AD 的可行性。

---

### ⚠️ **局限性**

1. **当前实现依赖手动推导 backward rule**
   - 不适用于任意结构，需针对特定矩阵模式设计。
2. **JAX 分布式调度尚未完全优化**
   - 相比 Serinv 的 CUDA stream 控制，存在同步开销。
3. **仅支持 BTA 类结构**
   - 对更复杂的稀疏模式（如非规则网格）需重新开发。

---

### 🔮 **未来工作方向**

1. **自动化结构感知 AD 编译器**
   - 开发 DSL 或 IR，自动识别并生成 carry-based backward rules。
2. **集成到 Probabilistic Programming Languages**
   - 如与 NumPyro/TensorFlow Probability 耦合，构建支持 INLA 的 PPL。
3. **扩展至非高斯似然与动态模型**
   - 探索在非线性观测下的 Laplace-adjoint 框架。
4. **异构加速（CPU+GPU+NPU）协同调度**
   - 优化 CPU staging 与通信隐藏。

---

## ✅ 总结一句话

> **ADELIA 通过结构感知的 reverse-mode AD，在百万变量级 INLA 推理中实现了 4.2–7.9× 的每梯度加速、5–8× 的节能，并首次确保了多变量模型的可靠收敛，标志着 HPC-scale 贝叶斯推理进入“精确梯度时代”**。

</details>

---

### 10. [LLMSpace: Carbon Footprint Modeling for Large Language Model Inference on LEO Satellites](https://arxiv.org/abs/2605.05615)

**Authors**: Lei Jiang, Adrian Ildefonso, Daniel Loveless, Fan Chen  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.05615v1  

#### Abstract
Large language models (LLMs) impose rapidly growing energy demands, creating an emerging energy and carbon crisis driven by large-scale inference. Solar-powered, AI-enabled low Earth orbit (LEO) satellites have been proposed to mitigate terrestrial electricity consumption, but their lifecycle carbon...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*LLMSpace: Carbon Footprint Modeling for Large Language Model Inference on LEO Satellites*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大规模部署的 **Large Language Models (LLMs)** 推理任务消耗巨大能源，预计到2030年全球数据中心电力需求将翻倍以上，带来显著的碳排放压力。为缓解地面电网负担，业界提出利用**低地球轨道（LEO）卫星**搭载AI硬件进行空间推理，依赖太阳能实现“绿色计算”。然而，现有研究在评估此类系统的**全生命周期碳足迹**时存在严重不足：

- 忽略发射、制造和辐射加固硬件带来的高**embodied carbon**（隐含碳）
- 缺乏对LLM推理特有行为（如prefill-decode阶段差异）的建模
- 对外围子系统（如太阳能板、电池、散热面板）建模粗糙或缺失

这导致无法准确判断“太空AI”是否真正环保。

---

### 🚀 提出的新方法：LLMSpace
本文提出了 **LLMSpace** ——首个专门面向**LEO卫星上LLM推理任务**的端到端碳足迹建模框架。

#### 主要创新点：
| 创新维度 | 具体内容 |
|--------|--------|
| **Comprehensive Peripheral Modeling** | 显式建模关键外围组件：<br>• 太阳能阵列（Si/GaAs/多结）<br>• 电池（LFP/NMC/辐射加固型）<br>• 辐射冷却面板（被动式/蜂窝/热管）<br>支持不同卫星类别（CubeSat → Starlink级）配置 |
| **Radiation-Hardened Hardware Support** | 首次纳入空间专用硬件的碳成本：<br>• 采用FD-SOI等工艺的rad-hard逻辑芯片<br>• MRAM替代DRAM用于KV缓存<br>• 架构级容错机制（ECC, TMR）带来的面积与碳开销 |
| **LLM Inference-Aware Analysis** | 区分prefill与decode阶段的能耗与延迟<br>支持多样化LLM workload分析（对话、代码生成、摘要等）<br>结合prompt长度、token生成数等参数动态估算碳消耗 |

---

### 🔍 相比现有方法的优势

| 方法 | 是否支持空间部署 | 外围建模 | 辐射加固硬件 | LLM workload特性 |
|------|------------------|----------|---------------|--------------------|
| [14,15,1] (Terrestrial LLM) | ❌ | ❌ | ❌ | ✅ |
| EIR [7] | ✅ | ❌（缺冷却板） | ❌ | ❌ |
| NE [9] | ✅ | ❌（聚合估计） | ❌ | ❌ |
| **LLMSpace (Ours)** | ✅ | ✅（细粒度分解） | ✅ | ✅ |

> ✅ LLMSpace 在**embodied carbon估算精度上比先前工作提升最多达27.8%**

---

## 2. 核心实验方法和设置

### 📊 数据集与基准任务
使用 **HELM (Holistic Evaluation of Language Models)** 中的11个代表性任务进行 workload 分析：
- `banking77`, `bigcodebench`, `cleva:coreference_resolution`
- `cleva:mathematical_calculation`, `cleva:mathematical_reasoning`
- `financebench`, `ifeval`, `infinite_bench_en_sum` (ensum)
- `mmlu_pro`, `omni_math`, `paraphrase_generation`

模型：`codellama/CodeLlama-34b-Instruct-hf`（bfloat16精度）  
硬件平台：NVIDIA H100 GPU，batch size = 1

---

### ⚙️ 实验设置

#### 卫星配置对比：
| 配置 | 类型 | 运行寿命 | GPU平台 | 是否辐射加固 |
|-----|------|---------|--------|-------------|
| COTS | 商业现货 | ~2年 | DGX H100 / Jetson Nano | 否 |
| rad-hard | 辐射加固 | ~10年 | 28nm FD-SOI H100 + MRAM | 是 |
| rad-opt | 优化版rad-hard | 10年 | 同上 + 多结太阳能 + 蜂窝冷却 |
| A100 | 替代方案 | 10年 | rad-hard DGX-A100（功耗更低） |

#### 发射系统
- 使用 **Falcon-9** 火箭
- 发射碳强度：14.5 kgCO₂e/kg payload

#### 功耗假设
- rad-hard GPU 操作功耗为 COTS 的 **1.2×**
- 采用 **Vidur-Energy** 模型预测LLM推理能耗
- 地面通信能耗：~0.5 pJ/bit

---

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Total Embodied Carbon (tCO₂e)** | 制造 + 发射全过程隐含碳排放总量 |
| **Annualized Carbon Emissions** | 总碳排放 / 使用年限，反映年均影响 |
| **Inference Latency** | Time to First Token (TTFT), Time Between Tokens (TBT), End-to-End (E2E) 延迟 |
| **Operational Energy Consumption** | 单次推理耗电量（Wh） |
| **Carbon Efficiency** | 单位碳排放所能完成的有效推理量 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）embodied carbon 估算准确性验证（Table 6）
以 Starlink-V1 + DGX H100 为例，真实值为：
- COTS 配置：18.3 tCO₂e
- rad-hard 配置：22.5 tCO₂e

| 方法 | COTS 误差 | rad-hard 误差 |
|------|-----------|--------------|
| EIR [7] | -33% | — |
| NE [9] | -30.7% | — |
| **LLMSpace** | **-11.4%** | **-13.4%** |

> ✅ LLMSpace 将误差降低约 **20个百分点以上**，显著优于已有工具。

---

#### （2）轨道 vs 地面数据中心碳足迹比较（Figure 2）

| 场景 | 年均碳排放趋势 | 结论 |
|------|----------------|------|
| **Clean Grid Terrestrial (20 gCO₂e/kWh)** | 随时间线性下降（仅运行碳） | 最清洁 |
| **Dirty Grid Terrestrial (380 gCO₂e/kWh)** | 初始高，缓慢下降 | 污染严重 |
| **Orbital COTS (2年)** | 初期极高，2年后失效 | 不可持续 |
| **Orbital rad-hard (10年)** | 初始更高，但随年限摊薄后介于 clean 与 dirty 之间 | 可部分减压地面碳排 |

> ✅ 对于**长期运行的大规模推理负载**，rad-hard LEO系统可成为一种**折中的低碳选择**。

---

#### （3）小规模GPU（Jetson Nano）不适用于轨道部署（Figure 2c）
- Jetson Nano 的碳足迹主要由**制造embodied碳主导**（占80–90%）
- 上天后虽无运行碳，但发射+辐射加固使其总碳远超地面clean场景
> ❌ **小型移动GPU不适合部署于LEO卫星**

---

#### （4）LLM workload 特性分析（Figure 3）
- 推理能耗与 **generated token count** 强相关（decode阶段主导）
- prompt length 影响较小
- 通信能耗（~μJ级）相比推理能耗（~mJ级）可忽略（相差10⁴–10¹⁰倍）

> ✅ **适合在轨执行长文本生成类任务**（如摘要、代码补全），传输代价低而本地卸载收益大。

---

#### （5）延迟-碳权衡实验（Figure 4）
| 配置 | 相比 H100 的变化 | 结果 |
|------|------------------|------|
| **A100 (rad-hard)** | 更低功耗（6.8kW vs 12kW） | • 总embodied碳↓30%<br>• TTFT↑85%，TBT↑43%，E2E↑47%<br>• 运行能耗↓10% |

> ✅ **可通过牺牲一定延迟换取显著碳减排**，适用于非实时应用场景。

---

#### （6）外围优化效果（rad-opt）
通过升级为：
- 多结太阳能阵列
- 辐射加固电池
- 铝蜂窝冷却面板

> 可进一步减少 **8% 总embodied碳**，尽管单位制造碳更高，但质量更轻 → 发射碳更低。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Embodied Carbon Dominates in Space**
   - LEO卫星上的AI系统，其碳足迹主要来自**发射、制造和外围设备**，而非运行能耗。
   - 即使使用零碳太阳能，**high upfront carbon cost** 是主要瓶颈。

2. **Radiation Hardening 是必须且昂贵的**
   - COTS硬件在轨寿命短（<2年），难以支撑持续服务
   - rad-hard设计虽增加初期碳成本，但延长寿命可有效摊薄年均碳排放

3. **Peripherals Are Critical**
   - 太阳能板、电池、冷却面板的质量与效率直接影响发射碳和系统规模
   - 改进外围技术（如多结太阳能、热管冷却）是降低总体碳的关键路径

4. **Not All Workloads Benefit Equally**
   - **长输出任务**（如摘要、代码生成）更适合在轨执行
   - **短交互任务** 或 **小模型推理** 应保留在地面

5. **Small GPUs Are Inefficient in Orbit**
   - 如 Jetson Nano 等边缘设备，其制造碳占比过高，上天得不偿失
   - 规模效应只在大型GPU系统中显现

6. **Trade-off Between Latency and Sustainability**
   - 使用低功耗但低性能硬件（如A100）可大幅降低碳足迹，适合容忍延迟的应用

---

### ⚠️ 方法的局限性

1. **依赖公开参数估算**
   - 某些rad-hard芯片（如28nm FD-SOI H100）尚无实际产品，参数基于保守推断
   - MRAM、rad-hard SSD 的碳强度来自学术文献外推

2. **未考虑再入销毁或回收**
   - 当前模型未包含卫星退役后的处理碳成本

3. **静态 workload 假设**
   - 未模拟动态调度或多任务并发下的资源竞争与能效波动

4. **未涵盖训练任务**
   - 仅聚焦inference，onboard training仍被认为不可行

---

### 🔮 未来工作方向

1. **扩展至其他轨道层级**
   - 建模 GEO/MEO 卫星的碳足迹特性

2. **引入可重构/模块化设计**
   - 支持在轨升级硬件，延长整体系统寿命

3. **整合绿色发射技术**
   - 考虑可重复使用火箭、清洁能源燃料对发射碳的影响

4. **构建联合优化引擎**
   - 自动搜索最优硬件配置、外围选型与任务分配策略，实现最小碳路径

5. **探索混合架构（Hybrid Orbital-Terrestrial）**
   - 动态决定任务应在地面还是空间执行，基于实时碳价与QoS需求

---

> 🌍 **最终结论**：  
> **LLMSpace 表明，“太空AI”并非天然绿色。只有通过精细化建模、合理设计和任务匹配，才能让LEO卫星真正成为可持续AI基础设施的一部分。**

</details>

---

### 11. [Scene-Adaptive Continual Learning for CSI-based Human Activity Recognition with Mixture of Experts](https://arxiv.org/abs/2605.06447)

**Authors**: Wenhan Zheng, Yuyi Mao, Ivan Wang-Hei Ho  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.06447v1  

#### Abstract
Channel state information (CSI)-based human activity recognition (HAR) is vulnerable to performance degradation under domain shifts across varying physical environments. Continual learning (CL) offers a principled way to learn new domains sequentially while preserving past knowledge, but existing CL...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **CSI-based Human Activity Recognition (HAR)** 中的**跨域性能退化**问题，即当模型部署到新的物理环境（如不同房间布局、家具摆放）时，由于 **domain shift** 导致 CSI 信号分布发生显著变化，从而造成识别准确率大幅下降。传统方法面临以下挑战：
- **灾难性遗忘（Catastrophic Forgetting）**：在学习新场景时丢失旧场景的知识；
- **推理开销线性增长**：如 MEMO 等架构扩展方法需激活所有专家模块，导致计算成本随领域数量增加而上升；
- **依赖大规模回放缓冲区（Replay Buffer）**：受限于边缘设备存储资源。

### 提出的新方法与思路
作者提出 **Scene-Adaptive Mixture of Experts with Clustered Specialists (SAMoE-C)** 框架，其核心思想是将跨域 HAR 建模为一个基于 **Mixture of Experts (MoE)** 的持续学习系统，具备以下创新设计：

1. **模块化网络架构**：
   - **Shared Backbone**：共享的浅层特征提取主干，用于提取 CSI 张量中的通用时空特征。
   - **Domain-Specific Specialists**：每个领域拥有独立的深层分类专家网络，负责捕捉特定环境下的细粒度活动模式。
   - **Semantic Router（语义路由器）**：基于注意力机制的门控网络（ResGateNet），根据输入 CSI 的上下文向量选择最合适的专家进行推理。

2. **稀疏激活机制**：
   - 路由器通过 `argmax` 仅激活一个专家，实现**恒定推理成本**（constant inference cost），不随领域数增长。

3. **轻量级训练协议**：
   - **分阶段训练（Decoupled Training Protocol）** 包括三个阶段：
     1. **Initial Domain Training**：在首个领域上联合训练 backbone 和第一个专家，之后冻结 backbone。
     2. **Incremental Learning**：新增领域时只训练对应的新专家。
     3. **Router Update**：利用极小比例的历史数据（tiny replay buffer）微调路由器，防止对已有领域的遗忘。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可扩展性** | 支持无限增量添加新领域，无需重新训练整个模型 |
| **推理效率** | 推理复杂度恒定 ~199.1 MFLOPS/sample，远低于 MEMO 的 797.8 MFLOPS |
| **抗遗忘能力** | 有效缓解灾难性遗忘，保留历史领域性能 |
| **资源友好性** | 仅需极小回放缓冲区（p=0.05）即可稳定路由性能 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **MM-Fi dataset** [13]，包含来自 **4个不同物理环境** 的 CSI 数据：
  - 两个 Living Rooms (D1, D2)
  - 两个 Meeting Rooms (D3, D4)
- 涵盖 **K=27 种人类活动**（如走路、挥手、跌倒等）
- CSI 数据由 TP-Link N750 路由器采集，使用 Atheros CSI Tool 提取
- 输入张量形状：`(3, 10, 114)`，表示天线对 × 时间步 × 子载波

### 实验设置
- **任务设定**：**Domain-Incremental Learning**，按顺序依次引入 D1 → D2 → D3 → D4
- **评估方式**：
  - 在每个新领域训练后，在所有已见领域的测试集上评估平均 HAR 准确率
  - 报告 **Final Average HAR Accuracy (%)**
- **超参数配置**：
  - Optimizer: AdamW
  - Batch Size: 64
  - Learning Rate: 3×10⁻⁴
  - Replay Buffer Ratio: p = 0.05（即每领域保留 5% 样本用于路由器更新）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Basic CL** | 单一模型在所有领域上顺序训练，无专家分离，易出现灾难性遗忘 |
| **MEMO** [8] | 多专家框架，但无路由器；推理时聚合所有专家输出，导致线性增长的计算开销 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table III）

| Method | Final Avg. HAR Acc. (%) | Inference Cost (MFLOPS/sample) |
|--------|--------------------------|-------------------------------|
| Basic CL | 29.56 | 198.4 |
| MEMO | 87.69 | 797.8 |
| **SAMoE-C** | **81.66** | **199.1** |
| SAMoE-C (Alt. Order) | 81.24 | 199.1 |

> 注：测量基于平衡测试集，样本形状 `(3, 114, 10)`

### 与基线方法对比结果
- **相比 Basic CL**：
  - 准确率提升 **+52.10% 绝对增益**（29.56% → 81.66%）
  - 推理成本几乎相同（~199 MFLOPS），说明额外路由开销极低（仅约 0.7 MFLOPS）
- **相比 MEMO**：
  - 虽然准确率略低（87.69% vs 81.66%），但**推理成本降低 75%以上**（797.8 → 199.1 MFLOPS）
  - 实现了更优的 **accuracy-efficiency trade-off**，更适合边缘部署

### 消融实验结果（Ablation Study on Replay Buffer Size）
- **无回放缓冲（p=0）**：
  - 路由器验证准确率仅为 **25.00%**，严重偏向最新领域（D4），表明存在严重灾难性遗忘
- **引入最小缓冲（p=0.01）**：
  - 路由准确率跃升至 **91.86%**
- **p=0.05 时**：
  - 路由准确率达到 **98.05%**，接近饱和
- **进一步增大缓冲（p=0.25）**：
  - 性能提升有限（99.32%），边际效益递减

👉 结论：**极小的 replay buffer（5%）即可实现高效且稳定的路由器训练**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **SAMoE-C 成功实现了高效、可扩展的跨域 CSI-HAR**：
   - 通过 MoE 架构与 selective activation，解决了灾难性遗忘与推理开销增长的双重难题。
2. ✅ **语义路由器具有高判别力**：
   - 利用注意力生成的 context vector 可有效区分不同场景，配合轻量 replay buffer 实现 >98% 的路由准确率。
3. ✅ **分阶段训练协议高效实用**：
   - 冻结 backbone + 独立训练专家 + 定期更新路由器，适合流式数据到达的真实场景。
4. ✅ **对初始领域顺序鲁棒性强**：
   - 更换领域顺序（D3→D1→D2→D4）下仍保持 81.24% 平均准确率，显示良好的泛化性。

### 方法的局限性
- ❌ **硬路由（Hard Routing）限制表达能力**：
  - 当前采用 `argmax` 仅激活单一专家，可能忽略多领域共现特征或模糊边界情况。
- ❌ **专家数量等于领域数**：
  - 若领域过多可能导致内存占用上升（尽管推理成本不变）。
- ❌ **尚未探索预训练策略**：
  - backbone 从零开始训练，若能在大规模 CSI 数据上预训练，有望进一步提升迁移性能。

### 未来工作方向
1. 探索 **soft routing** 机制，允许多专家加权融合，增强模型表达能力；
2. 引入 **backbone 预训练**，在更大规模多场景 CSI 数据集上进行初始化；
3. 研究 **专家聚类与复用机制**，避免为高度相似场景创建冗余专家；
4. 将 SAMoE-C 扩展至 **无线感知中的其他任务**，如呼吸监测、人数估计等。

---

> 📌 **总结一句话**：  
> SAMoE-C 通过 **Mixture of Experts + Semantic Router + Lightweight Continual Training**，在保持恒定推理成本的前提下，实现了高性能、可扩展、资源友好的跨域 CSI-based HAR，为实际边缘部署提供了可行方案。

</details>

---

### 12. [LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG](https://arxiv.org/abs/2605.06285)

**Authors**: Yijia Zheng, Marcel Worring  
**Category**: cs.CL  
**Published**: 2026-05-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.06285v1  

#### Abstract
Single-step retrieval-augmented generation (RAG) provides an efficient way to incorporate external information for simple question answering tasks but struggles with complex questions. Agentic RAG extends this paradigm by replacing single-step retrieval with a multi-step process, in which the large ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Agentic RAG** 方法通过多步推理（如 Chain-of-Thought）生成自然语言形式的中间思考（thoughts）和子查询（subqueries），以迭代检索外部知识。虽然这类方法在复杂问答任务上表现优异，但其**推理过程严重依赖自回归（autoregressive）生成**，导致显著的延迟（latency），尤其是在生成长文本时。

具体瓶颈在于：
- **Thought Generation** 和 **Subquery Generation** 阶段占总延迟约 **90%**。
- 每一步都需要逐 token 生成，无法并行化，限制了效率。

### 提出了什么新方法或新思路
本文提出 **LatentRAG**，一种全新的高效 Agentic RAG 框架，其核心思想是将推理与检索从离散的语言空间转移到连续的**隐空间（latent space）**中进行。

#### 核心创新点：
1. **Latent Reasoning（隐式推理）**  
   不再显式生成自然语言的 thought 和 subquery，而是向 LLM 输入特殊标记（如 `<think>`、`<query>`），直接利用其最后一层隐藏状态作为 **latent thought tokens** 和 **latent subquery tokens**。这一过程仅需一次前向传播（single forward pass），避免了自回归解码。

2. **Latent Retrieval（隐式检索）**  
   将生成的 latent subquery tokens 投影到检索模型的输入空间，并用于稠密检索（dense retrieval）。这使得 subquery 可以不经过语言解码即可触发检索。

3. **端到端联合优化（End-to-End Joint Optimization）**  
   引入基于 **KL 散度** 的对齐目标函数，使 latent subquery embeddings 逼近由原始自然语言 subquery 产生的 reference embeddings，从而实现 LLM 与检索模型的可微分联合训练。

4. **可选的 Latent Decoding（隐式解码）机制**  
   为提升透明性，引入一个轻量级解码器，可在推理阶段将 latent tokens 解码回自然语言，支持事后解释（post-hoc explanation）。该过程可跨步骤并行执行，进一步提高效率。

### 相比现有方法的优势
| 维度 | LatentRAG | 传统 Agentic RAG |
|------|---------|----------------|
| 推理方式 | 在 latent space 中并行计算 | 自回归生成自然语言 |
| 检索触发 | 使用 latent tokens 直接检索 | 必须先解码出 subquery 文本 |
| 训练方式 | 支持端到端联合优化 | 多为两阶段或非可微 |
| 效率 | 极大降低延迟（↓~90%） | 高延迟（↑15× Naive RAG） |
| 透明性 | 可选解码，兼顾效率与可解释性 | 完全透明但冗长 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
共使用 **7 个标准 QA 数据集**，分为两类：

- **通用问答（General QA）**：
  - NQ (Natural Questions)
  - TriviaQA
  - PopQA

- **多跳问答（Multi-hop QA）**：
  - HotpotQA
  - 2wiki
  - Musique
  - Bamboogle

所有方法均在 **2018 Wikipedia dump** 上进行检索，共约 2100 万文档块，构成大规模真实检索场景。

### 实验设置和评估指标

#### 主要评估指标：
- **Exact Match (EM)**：预测答案是否完全匹配真实答案。
- **Average Latency (ms)**：每道题从输入到输出的端到端响应时间，细分为 prefill、thought gen、subquery gen、retrieval、answer gen 等阶段。

#### 模型配置：
- **LLM**：默认使用 `Qwen2.5-7B`，也测试了 3B 和 14B 版本。
- **Retriever**：采用多个轻量级 dense retriever，包括：
  - Qwen3-Embedding-0.6B
  - e5-base-v2
  - jina-embeddings-v5-text-nano
  - harrier-oss-v1-270m
  - F2LLM-v2-330M

#### 训练策略：
- 采用 **Supervised Fine-Tuning (SFT)**，轨迹来自 Search-R1 和 AutoRefine 两种强基线。
- 使用 LoRA 进行参数高效微调（rank=16）。
- 总体损失函数为：
  $$
  \mathcal{L} = \mathcal{L}_{\text{gen}} + \lambda_{\text{ret}} \mathcal{L}_{\text{ret}} + \mathcal{L}_{\text{dec}}
  $$

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 方法 | 平均 EM (%) | 平均延迟 (ms) | 延迟下降 vs. Agentic Baseline |
|------|-------------|---------------|-------------------------------|
| Naive RAG | ~25–35 | ~300–400 | — |
| Search-R1 / AutoRefine | ~42–48 | ~4800–5400 | — |
| **LatentRAG** | **≈42–49**（±5%） | **~500–600** | **↓ ~90%** |

> ✅ LatentRAG 在保持与 Search-R1 / AutoRefine **相当性能**的同时，**平均减少约 90% 的推理延迟**，接近单步 RAG 的效率水平。

### 与基线方法的对比结果

- 在 Table 1 中显示：
  - LatentRAG 在多数数据集上的 EM 分数略高于或持平于 Search-R1 和 AutoRefine。
  - 延迟方面，LatentRAG 仅为 Search-R1 的 **~10–14%**（即提速 7–10 倍）。
  - 例如，在 HotpotQA 上，Search-R1 耗时 6.9s，而 LatentRAG 仅需 **704ms**。

- 图 1 显示：
  - Search-R1 的延迟主要集中在 **Thought Gen** 和 **Subquery Gen** 阶段。
  - LatentRAG 几乎消除了这两个阶段的时间开销。

### 消融实验结果（Ablation Studies）

| 变体 | EM (%) | Success Rate | Overlap |
|------|--------|--------------|---------|
| LatentRAG (Full) | **43.46** | **61.27%** | 59.41% |
| w/ Cosine Loss | 42.55 | 60.76% | **68.31%** |
| w/ InfoNCE Loss | 41.86 | 58.60% | 47.08% |
| w/o Retriever | 41.85 | 59.07% | 50.92% |
| w/o Decoding Loss | 40.61 | 60.64% | 57.38% |

#### 结论：
- **KL 散度目标最优**：相比 Cosine 或 InfoNCE，KL 更适合处理伪相关文档中的噪声。
- **预训练检索器至关重要**：移除后性能明显下降，说明其提供了重要归纳偏置。
- **Latent Decoding 提升表示学习**：即使只在训练中使用，也能显著提升最终性能。

---

## 4. 关键结论和发现

### 主要发现
1. **Latent Space 可有效承载 Agentic 推理语义**  
   实验证明，LLM 的隐藏状态足以编码有意义的 intermediate thoughts 和 subqueries，无需显式语言生成。

2. **LatentRAG 实现了性能与效率的平衡**  
   在几乎不牺牲准确率的前提下（差异 <5%），将 Agentic RAG 的延迟压缩至原来的 **10% 左右**，极大提升了实用性。

3. **并行解码机制显著提升透明性效率**  
   即便启用 latent decoding 输出自然语言，其延迟仍比传统方法低 **47–63%**，且支持跨步并行。

4. **模型缩放具有良好一致性**  
   在不同规模的 LLM 和 retriever 上，LatentRAG 均能稳定提升效率，尤其在小模型上优势更明显。

### 方法的局限性
- **依赖高质量训练轨迹**：当前采用 SFT，性能受限于 teacher model（如 Search-R1）的质量。
- **对检索器几何敏感**：如 e5-base-v2 因嵌入空间高度各向异性（anisotropic），导致适配困难，性能下降较大。
- **精确词元生成能力较弱**：案例分析表明，latent 表示擅长抽象推理，但在需要精确拼写输出的任务中可能出现错误（如 “Montmoreiras”）。

### 未来工作方向
- **结合强化学习（RL）**：摆脱对 teacher 轨迹的依赖，直接通过环境反馈学习最优检索策略。
- **改进 latent-to-language 解码器**：增强对专有名词、数字等细节的重建能力。
- **构建面向 Agent 的 Embedding-based Search Engine**：推动搜索引擎从“人类友好”向“Agent 友好”演进，原生支持 latent query 输入。

---

> **一句话总结**：  
> **LatentRAG 成功地将 Agentic RAG 的推理与检索迁移到隐空间，实现了接近单步 RAG 的效率，同时保留了多步推理的强大性能，为高效率、可扩展的智能代理系统提供了新范式。**

</details>

---

### 13. [LLM-Enhanced Deep Reinforcement Learning for Task Offloading in Collaborative Edge Computing](https://arxiv.org/abs/2605.05727)

**Authors**: Hao Guo, Kaixiang Xv, Ziwu Ge, Lei Yang  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.05727v1  

#### Abstract
Collaborative edge computing uses edge nodes in different locations to execute tasks, necessitating dynamic task offloading decisions to maintain low latency and high reliability, especially under unpredictable node failures. Although deep reinforcement learning (DRL) and large language models (LLMs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM-Enhanced Deep Reinforcement Learning for Task Offloading in Collaborative Edge Computing

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**Collaborative Edge Computing (CEC)** 场景中，任务卸载面临以下挑战：
- **动态环境不确定性**：节点故障、链路中断、拓扑变化频繁发生，导致传统静态策略失效。
- **DRL 的局限性**：Deep Reinforcement Learning（DRL）虽然具备自适应能力，但存在**样本效率低、收敛慢、易陷入局部最优**等问题，尤其在大规模网络中表现不佳。
- **LLM 的实时性瓶颈**：Large Language Models（LLMs）虽具有强大的语义推理能力，但其**高延迟和非确定性输出**使其难以直接用于实时决策。

### 🚀 提出的新方法：LeDRL
作者提出 **LeDRL** —— 一种融合轻量级 LLM 与 DRL 的混合决策框架，实现**语言引导的实时任务卸载**。其核心创新包括：

#### （1）**Dec-POMDP 建模 + MINLP 优化问题形式化**
- 将任务卸载建模为 **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**，更贴合分布式边缘节点仅能获取局部观测的现实。
- 形式化为 **Mixed Integer Nonlinear Programming (MINLP)** 问题，并证明其 **NP-hard**，为后续算法设计提供理论依据。

#### （2）**LLM-Guided Semantic Abstraction with Reflective Evaluator**
- 构造**结构化上下文感知 prompt**，整合节点状态、任务特征、链路动态等信息，供 LLM 进行高层策略推理。
- 设计 **Reflective Evaluator** 模块，将历史轨迹中的失败经验提炼为**语义反馈**，持续优化后续 prompt 内容，形成“试错-反思”闭环。

#### （3）**Self-Attention-Based Policy Alignment Module**
- 引入基于 **self-attention** 的融合机制，将 LLM 输出的“意图”与 DRL 的本地观测进行对齐。
- 实现**选择性吸收语义先验知识**，避免盲目依赖 LLM 输出，提升策略鲁棒性和泛化能力。

#### （4）**轻量化 LLM 部署支持边缘推理**
- 集成 **lightweight LLM（如 Qwen3-4B）** 到在线决策环路，在保证推理速度的前提下引入语义指导。
- 在 Jetson 边缘设备上部署原型系统 **CoEdgeSys**，验证了实际可行性。

### 🔍 相比现有方法的优势
| 维度 | 传统 DRL 方法 | 纯 LLM 方法 | LeDRL |
|------|----------------|--------------|--------|
| 样本效率 | 低 | 不适用 | ✅ 显著提升（借助 LLM 先验） |
| 收敛速度 | 慢 | 快但不稳定 | ✅ 更快且稳定 |
| 实时性 | 高 | ❌ 差（延迟大） | ✅ 平衡良好 |
| 动态适应性 | 有限 | 波动大 | ✅ 强（结合记忆与注意力） |
| 可部署性 | ✅ 良好 | ❌ 困难 | ✅ 支持边缘部署 |

---

## 2. 核心实验方法和设置

### 📊 数据集与仿真环境
- **无真实公开数据集**，采用**自定义模拟器**构建动态边缘网络。
- 时间划分为 100 个 slot，每个 slot 执行一次决策。
- 网络规模：10 ~ 20 个异构节点（Jetson 类设备模拟），随机生成稀疏连通图。
- 任务到达服从 **Bernoulli 分布**，输入大小 [2000, 4000] KB，计算强度 [800, 2400] cycles/bit，截止时间固定为 **4 秒**。
- 节点 CPU 主频 3GHz，链路速率 [10, 40] MB/s。
- 故障模型：节点硬件/软件故障率 0.01，链路故障率可调；节点加入概率 0.1。

### 🎯 评估指标
| 指标 | 定义 |
|------|------|
| **Task Success Rate (%)** | 满足延迟 ≤ 截止时间和可靠性 ≥ 阈值的任务占比（主指标） |
| **Convergence Speed** | 达到稳定高成功率所需的训练 episode 数 |
| **Inference Latency (s)** | 单次决策耗时，衡量实时性 |
| **Robustness** | 在不同扰动（任务大小、复杂度、故障率）下的性能稳定性 |

### ⚔️ 基线方法对比
共比较六种方法：
1. **DRL 方法**：
   - `VDN-TO`：基于价值分解的多智能体 RL
   - `MAPPO-TO`：基于 MAPPO 的策略梯度方法
   - `MASAC-TO`：适配离散动作空间的 SAC 变体
2. **启发式方法**：
   - `RATC`：基于 deadline 和 reliability 的轻量规则
   - `AGSP`：基于遗传-模拟退火的混合优化引擎
3. **LLM 方法**：
   - `Reflexion`：使用迭代自我反馈优化 LLM 决策的框架

所有方法均在相同拓扑下运行 10 次取平均值，确保公平性。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table I 和 Fig. 4–8）

| 方法 | 10节点 成功率↑ | 推理延迟↓ | 20节点 成功率↑ | 推理延迟↓ |
|------|------------------|------------|------------------|------------|
| RATC | 45.48 ± 1.24% | 0.0004s | 59.98 ± 1.45% | 0.0008s |
| AGSP | 50.01 ± 1.63% | 0.0005s | 39.48 ± 1.72% | 0.0018s |
| VDN-TO | 44.47 ± 1.92% | 0.0028s | 51.27 ± 1.82% | 0.0026s |
| MASAC-TO | 51.81 ± 2.88% | 0.0025s | 51.80 ± 1.77% | 0.0024s |
| MAPPO-TO | 52.68 ± 3.40% | 0.0035s | 60.68 ± 2.24% | 0.0043s |
| Reflexion | 48.34 ± 1.97% | 1.5786s | 60.86 ± 1.39% | 3.0128s |
| **LeDRL (Ours)** | **59.46 ± 1.42%** | **0.7046s** | **63.78 ± 1.68%** | **0.7379s** |

> ✅ **LeDRL 在成功率上全面领先**，相比最强 DRL 基线 `MAPPO-TO` 提升约 **12.87%（10节点）** 和 **5.1%（20节点）**  
> ✅ 相比纯 LLM 方法 `Reflexion`，**延迟降低超过 70%**，同时保持更高成功率

### 🔁 与基线方法的对比结果
- **训练效率**：LeDRL 收敛更快，在早期阶段即表现出更高的成功率和更低方差（Fig. 4）。
- **鲁棒性更强**：在任务规模增大、计算复杂度上升、故障率提高等扰动下，LeDRL 性能下降最缓（Fig. 5）。
  - 当执行失败率为 0.25 时，LeDRL 比 MAPPO-TO 提升 **12%**；
  - 当传输失败率为 0.25 时，提升达 **17%**。
- **拓扑适应性强**：在 ring topology（环形结构）这种路由灵活性差的场景中，其他方法性能显著下降，而 LeDRL 仍保持稳定优势（Fig. 6a）。

### 🔍 消融实验结果（Ablation Study）
作者对比了多个变体以验证各模块有效性（Fig. 6b）：
| 变体 | 成功率（约） | 分析 |
|------|---------------|------|
| `Plain MAPPO` | ~50% | 仅靠试错学习，探索效率低 |
| `LLM + MLP + MAPPO` | ~54% | 加入 LLM 提示有帮助，但未对齐导致噪声 |
| `RATC + SA + MAPPO` / `AGSP + SA + MAPPO` | ~56% | 注意力机制改善融合效果 |
| **LeDRL（完整版）** | **~64%** | ✅ 同时具备**结构化提示 + 自注意力融合 + 反思机制**，实现最佳协同 |

> 结论：**Reflective Evaluator 和 self-attention fusion 是关键增益来源**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 可作为高质量“认知先验”注入 DRL 学习过程**，显著提升样本效率和初始探索质量。
2. **轻量级 LLM 完全可以在边缘端参与实时决策**，只要通过合理架构设计控制延迟。
3. **self-attention 是连接符号级语义与数值型策略的有效桥梁**，实现上下文感知的知识融合。
4. **反思机制（reflective feedback）能有效沉淀经验**，使系统具备“从失败中学习”的类人能力。
5. **LeDRL 在仿真与真实测试床（CoEdgeSys）中均优于主流基线**，最高提升 **17% 成功率**，且具备良好的可扩展性和鲁棒性。

### ⚠️ 方法的局限性
- **LLM 仍需外部服务器部署**：当前 Qwen3-4B 部署于云端 GPU 服务器，尚未完全嵌入边缘设备，存在通信开销。
- **Prompt engineering 依赖人工设计**：prompt 结构和 memory retrieval 规则需要领域知识支持，自动化程度有待提升。
- **泛化能力受限于训练分布**：若出现极端拓扑或未知任务模式，可能无法有效应对。
- **多跳路径规划能力未充分验证**：实验主要关注单跳或短路径卸载，长程依赖处理尚待深入研究。

### 🔮 未来工作方向
1. **解耦 LLM 与在线推理**：探索在训练阶段利用 LLM 提供监督信号，部署时移除 LLM，保留其“内化知识”。
2. **进一步压缩 LLM 模型**：尝试 TinyLLM 或蒸馏技术，实现端侧完整闭环。
3. **扩展至更大规模网络**：验证在百节点以上场景中的可扩展性。
4. **支持复杂 DAG workflow 卸载**：突破当前原子任务假设，处理有向无环图任务流。
5. **增强安全性与隐私保护机制**：防止 prompt 泄露敏感信息，适用于工业场景。

---

> 🔗 **代码开源地址**：[https://github.com/GalleyG5/LeDRL.git](https://github.com/GalleyG5/LeDRL.git)  
> 💡 **一句话总结**：LeDRL 成功将 LLM 的“大脑”与 DRL 的“小脑”结合，实现了**既聪明又敏捷**的边缘任务卸载决策系统。

</details>

---

### 14. [A Privacy-Preserving Machine Learning Framework for Edge Intelligence: An Empirical Analysis](https://arxiv.org/abs/2605.05751)

**Authors**: Quoc Lap Trieu, Bahman Javadi, Jim Basilakis  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.05751v1  

#### Abstract
As Edge Intelligence (EI) becomes increasingly prevalent in domains such as smart healthcare, manufacturing, and critical infrastructure, ensuring data privacy while maintaining system efficiency is a growing challenge. This paper presents a new privacy-preserving machine learning (PPML) framework t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《A Privacy-Preserving Machine Learning Framework for Edge Intelligence: An Empirical Analysis》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题  
随着 **Edge Intelligence (EI)** 在智能医疗、智能制造和关键基础设施等领域的广泛应用，如何在资源受限的边缘设备上实现**隐私保护**与**系统效率**之间的平衡成为一大挑战。现有研究多集中于云环境下的隐私保护，缺乏对边缘计算场景下不同 **Privacy-Preserving Machine Learning (PPML)** 技术的**系统性实证评估**。

本文旨在填补这一空白，全面评估三种主流 PPML 技术——**Differential Privacy (DP)**、**Secure Multi-party Computation (SMC)** 和 **Fully Homomorphic Encryption (FHE)**——在真实边缘环境中的性能表现，并揭示其在准确性、延迟和能耗等方面的权衡关系。

---

### 提出了什么新方法或新思路  

- **提出了一套面向 EI 应用的四层 PPML 框架架构**：  
  架构分为 **Cloud、Edge Server、Edge Device、Sensor** 四个层级，整合了 **Data、Model、System、Application** 四大核心组件，支持端到端的隐私保护推理任务部署。

- **设计并实现了统一的训练与推理算法流程（Algorithm 1 & 2）**：  
  明确了在不同隐私机制（DP/FHE/SMC）下，从云端模型训练、加密编译、密钥分发，到边缘设备上的预处理、加密/共享、安全推理及结果解密的完整流程。

- **结合真实硬件实验与 trace-based simulation 进行综合评估**：  
  不仅在真实异构边缘设备上进行测试以收集执行时间 trace，还基于这些 trace 在 **EdgeSimPy** 模拟器中构建大规模仿真环境，实现更贴近现实的性能分析。

---

### 相比现有方法的优势  

- **首次对 EI 场景下的三大 PPML 技术进行了横向、多维度的实证比较**，涵盖准确率、响应时间、能量消耗和安全性。
- 考虑了实际部署因素如网络带宽、参与方数量、量化位宽等，提供了可指导工程实践的参数调优建议。
- 引入了对 **black-box model stealing attack** 的防御能力分析，拓展了传统“隐私-效用”权衡至“隐私-效用-可提取性”三维前沿。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集  
实验采用来自 **UEA & UCR Time Series Classification Archive** 的三个真实世界时序数据集：

| 数据集 | 领域 | 类别数 | 输入长度 |
|-------|------|--------|----------|
| **FordA** | 制造业（故障检测） | 2 | 500 |
| **ElectricDevices** | 能源管理 | 7 | 96 |
| **ECG5000** | 医疗健康（心电监测） | 5 | 140 |

---

### 实验设置和评估指标  

#### 实验平台配置  
- **云端训练**：AWS `r6i.4xlarge` EC2 实例（16核，128GB RAM）
- **边缘服务器**：桌面级主机（16核，32GB RAM）
- **边缘设备**：Intel NUC、Jetson AGX、Raspberry Pi 5
- **模拟工具**：**EdgeSimPy**，模拟 4 台边缘服务器 + 12 台边缘设备

#### 评估指标  
- **Accuracy**（推理准确率）
- **Response Time / Inference Latency**（响应时间）
- **Energy Consumption**（每轮推理能耗，单位：Wh）
- **Communication Overhead**（通信开销，字节数与往返次数）
- **Privacy Budget (ε)** 与 **Model Extraction Resistance**

#### 隐私技术实现工具  
| 技术 | 工具 | 关键参数 |
|------|------|---------|
| **DP** | TensorFlow Privacy (DP-SGD) | Noise multiplier σ ∈ [0.1, 0.7] |
| **SMC** | CrypTen | 2-party / 3-party, TLS 加密通信 |
| **FHE** | Concrete-ML (TFHE-based) | Quantization bit-width q ∈ {4,5,6}, Precision p=5 |

---

### 基线方法对比  
- **Baseline**：无任何隐私保护的原始模型推理（Plaintext Inference）
- 对比对象为上述三种 PPML 方法在相同模型（LeNet-5, SqueezeNet, AlexNet）和数据集上的性能差异。

---

## 3. 主要实验结果和性能指标

### 关键性能数据  

#### 准确率影响（Accuracy Drop）  
| 方法 | 模型 | 数据集 | 准确率下降幅度 |
|------|------|--------|----------------|
| **DP** | AlexNet | FordA | ↓35% |
| **DP** | LeNet-5 | FordA | ↓13–18% |
| **FHE** | AlexNet | ECG5000 | 接近基线（<2% loss @ 6-bit） |
| **SMC** | 所有模型 | 所有数据集 | ≈ 基线（无损失） |

> ✅ **DP 对复杂模型敏感；FHE 在适当量化后可保持高精度；SMC 几乎不影响准确率**

---

#### 响应时间对比（vs. Plaintext）  
| 方法 | 相对延迟增长 | 典型值（AlexNet on ECG5000, 70并发用户） |
|------|--------------|----------------------------------------|
| **DP** | ≈ 基线 | 0.185 s |
| **SMC (2-party, 500Mbps)** | ×415 | 76.775 s |
| **FHE** | ×1000+ | 3102.03 s |

> ⚠️ **FHE 引入巨大计算开销；SMC 受限于通信延迟；DP 最轻量**

---

#### 能耗对比（Per Inference, ECG5000）  
| 方法 | LeNet-5 | AlexNet |
|------|--------|--------|
| **DP** | 0.0009 Wh | 0.0011 Wh |
| **SMC (2-party, 500Mbps)** | 0.0088 Wh | 0.3093 Wh |
| **FHE** | 1.1168 Wh | **20.0167 Wh** |

> 🔋 **FHE 是最耗能的技术，尤其对复杂模型；DP 几乎无额外能耗**

---

#### 消融实验结果  

##### （1）FHE 参数影响  
- **量化位宽（q）提升 → 准确率上升，延迟超线性增长**  
  - AlexNet 从 4-bit 到 6-bit：延迟 ↑7–9倍
- **精度位宽（p）增加 → 小幅提升延迟，显著影响 AlexNet 准确率**
- **LeNet-5 在 4–5 bit 即接近饱和，适合低功耗 FHE 部署**

##### （2）SMC 通信影响  
- 带宽从 250 Mbps → 500 Mbps：**AlexNet on FordA 延迟降低约 30%**
- 从 2-party → 3-party：延迟普遍增加 50%~100%，通信轮次增多

##### （3）DP 噪声影响  
- 噪声 multiplier σ↑ → ε↓（隐私更强），但准确率单调下降
- 在 σ≈0.5 附近出现“最佳操作点”：ε < 10，且 △Acc（提取差距）最大

---

## 4. 关键结论和发现

### 主要发现  

1. **DP 是延迟最低、能耗最小的选择**，适用于对实时性要求高的场景，但需谨慎调节噪声水平以避免严重精度损失，尤其在复杂模型上。
2. **SMC 性能由通信主导**，适合局域网内高带宽、低延迟连接的协作推理，可通过减少参与方和提高带宽优化性能。
3. **FHE 计算开销极高（约 ×1000 延迟）**，虽提供最强输入保密性，但必须通过模型简化、低位宽量化和高性能边缘服务器支撑才能实用化。
4. **模型复杂度是决定性因素**：  
   - AlexNet 在所有 PPML 下均表现最差（延迟高、能耗大、精度损严重）  
   - LeNet-5 和 SqueezeNet 更适合作为边缘隐私推理的基础模型
5. **DP 可有效增强抗模型窃取能力**：  
   - 随着噪声增加，攻击者训练出的替代模型（surrogate model）与目标模型差距（△Acc）先增大后减小，在中等噪声下达到最优防御效果
   - SMC 和 FHE 虽保护中间状态，但若输出未加扰动或访问控制，仍易被提取

---

### 方法的局限性  

- **未考虑联邦学习（FL）训练过程**：本研究聚焦于推理阶段，而许多 EI 应用涉及分布式训练。
- **FHE 当前版本限制**：Concrete-ML 仅支持最多 8-bit 整数量化，限制了模型表达能力。
- **假设半诚实敌手模型**：SMC 安全性基于 semi-honest assumption，未防御恶意篡改行为。
- **模拟中调度策略固定**：使用简单启发式调度，未探索动态负载均衡的影响。

---

### 未来工作方向  

- 优化 PPML 技术组合（如 DP + FHE 或 SMC + DP）以实现更好的隐私-效率折衷。
- 开发针对边缘场景的 **adaptive quantization** 和 **model partitioning** 策略。
- 设计新型任务调度算法，考虑隐私机制带来的计算与通信特征变化。
- 探索 **malicious-secure SMC** 与 **lightweight FHE** 方案在资源受限设备上的可行性。
- 将框架扩展至视频、语音等多模态边缘 AI 场景。

--- 

> 💡 **总结一句话**：  
> 在边缘智能中部署 PPML 必须根据应用场景精细选择技术路径——**追求极致低延迟选 DP，强调多方协作保密选 SMC，要求端到端加密计算则接受 FHE 的高昂代价**，同时优先选用轻量级模型以缓解性能瓶颈。

</details>

---

### 15. [FalconGEMM: Surpassing Hardware Peaks with Lower-Complexity Matrix Multiplication](https://arxiv.org/abs/2605.06057)

**Authors**: Honglin Zhu, Jiaping Cao, Jiang Shao, Siyuan Feng, Qian Qiu, Peng Chen, Xu Zhang, Yixian Zhou, Man Lung Yiu, Guang Ji, Minwen Deng, Wenxi Zhu, Jintao Meng  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.06057v1  

#### Abstract
Peak breaking Matrix Multiplication is a promising technique to improve the performance of DL, especially in LLM training and inference. We present FalconGEMM, a cross-platform framework that automates the deployment, optimization, and selection of Lower-Complexity Matrix Multiplication Algorithms (...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FalconGEMM: Surpassing Hardware Peaks with Lower-Complexity Matrix Multiplication

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **GEMM**（General Matrix Multiplication）在现代深度学习（DL）尤其是大语言模型（LLM）训练与推理中占据主导地位，其 $O(N^3)$ 的计算复杂度已成为性能瓶颈。尽管厂商优化的库（如 cuBLAS、Intel MKL）已将标准 GEMM 性能推向硬件峰值，进一步提升空间有限。

**Lower-Complexity Matrix Multiplication Algorithms**（LCMAs），如 Strassen 算法、AlphaTensor 发现的算法等，理论上可通过减少乘法次数降低计算复杂度（如降至 $O(N^{\log_2 7})$）。然而，这些算法在实际部署中面临三大挑战：
1. **跨平台可移植性差**：LCMAs 引入复杂的线性组合与数据依赖，难以高效适配不同硬件（GPU/CPU、不同架构）。
2. **内存开销高**：中间张量需写入片外内存，带来巨大带宽压力，抵消了算力节省。
3. **决策困难**：LCMAs 并非总是更快，其优势取决于矩阵形状、硬件带宽与算力比，缺乏轻量级决策机制。

### 提出的新方法与框架
作者提出 **FalconGEMM**，一个跨平台自动化框架，系统性解决上述问题，实现“突破硬件峰值”的 GEMM 加速。

#### 三大核心模块与创新点：

| 模块 | 创新点 |
|------|--------|
| **Deployment Module**（部署模块） | 基于 **代码生成**（Code Generation）技术，利用 TVM、Triton 等编译器，将 LCMA 的数学定义（系数张量 $U,V,W$）自动编译为针对特定硬件、数据类型和输入尺寸的高性能内核，实现“一次编写，处处运行”。 |
| **Execution Module**（执行模块） | 提出 **Group-Parallel Optimization**：<br>• 将计算按“组”（Group）组织，最大化片上数据复用。<br>• **融合 GEMM 与 Combine H 阶段**，避免中间结果 $H_r$ 写回片外内存。<br>• 进一步引入 **Split-Group Parallelism** 解决负载不均，**Cache-Aware Scheduling** 缓解缓存冲突。 |
| **Decision Module**（决策模块） | 构建轻量级 **分析性性能模型**，基于理论算术强度（Arithmetic Intensity）分析，预测 LCMA 是否能超越标准 GEMM，并选择最优算法。该模型考虑了内存与计算瓶颈的权衡。 |

### 相比现有方法的优势
- **全面性**：首个同时解决可移植性、内存开销、算法选择三大挑战的统一框架。
- **高效性**：通过融合执行与智能调度，显著降低内存流量，释放 LCMA 的理论潜力。
- **实用性**：决策模块确保在各种场景下自动选择最优策略，无需人工调优。
- **跨平台**：支持 GPU（NVIDIA H20/A100）、CPU（x86/ARM）及多种数据类型（FP32/BF16/FP16/FP8）。

---

## 2. 核心实验方法和设置

### 数据集与工作负载
- 从三个开源 LLM 中提取线性层的矩阵形状 $(N, K)$：
  - **DeepSeek-R1**
  - **Qwen3.5-397B**
  - **HunyuanVideo**
- 生成 960 组测试形状 $(M, N, K)$，其中 $M$ 从 512 到 20480，步长 512。

### 实验设置
- **硬件平台**：
  - **GPU**: NVIDIA H20 (Hopper), A100 (Ampere)
  - **CPU**: Intel Xeon Platinum 8255C (x86), AMD EPYC 9K84 (x86), AWS EC2 M7g (ARM Neoverse-V1)
- **数据类型**：FP32, BF16, FP16, FP8（Hopper 上使用 FP8E4M3 块量化）
- **软件栈**：TVM, TileLang, Triton 用于代码生成。

### 评估指标
- **有效 TFLOPS**（Effective TFLOPS）：
  $$
  \text{Performance} = \frac{2MNK}{\text{time\_seconds}} \times 10^{-12}
  $$
  使用标准 GEMM 的 FLOPs 数（$2MNK$）作为基准，即使实际执行的是 LCMA。这使得性能可以“超过硬件峰值 TFLOPS”，体现算法优势。

### 基线方法对比
- **标准 GEMM 库**：
  - GPU: **cuBLAS**, **CUTLASS**
  - x86 CPU: **Intel MKL**, **OpenBLAS**
  - ARM CPU: **ACL** (Arm Compute Library)
- **LCMA 竞争对手**：
  - **AlphaTensor**（基于 JAX 实现）
  - （注：无其他支持 FP8 的 LCMA 实现）

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **相比标准 GEMM 库**：
  - 在各类硬件和数据类型上，FalconGEMM **平均提升 7.59% ~ 17.85%**。
  - 具体示例：
    - H20 + FP8: **+7.59%**
    - H20 + BF16: **+7.50%**
    - A100 + FP32: **+11.82%**
    - Intel CPU + FP32: **+12.94%**
  - 当仅统计 FalconGEMM 选择 LCMA 的情况时，加速比更高（如 Intel CPU 上达 **+14.33%**）。

- **相比 LCMA 竞争对手 AlphaTensor**：
  - **全面大幅领先**，性能提升 **12.41% ~ 55.61%**。
  - 在 H20 (BF16) 上达到 **+55.61%** 的惊人优势。
  - AlphaTensor 在小矩阵上常劣于标准 GEMM，而 FalconGEMM 始终稳定优于或等于最佳基线。

- **端到端 LLM 推理加速**（替换 PyTorch 后端）：
  - 在 **prefill 阶段**，平均性能增益达 **11.46% ~ 18.12%**。
  - 例如，在 A100 + FP32 上对 Qwen3.5 实现 **18.12%** 的加速。

### 消融实验结果
- **Step-wise Evaluation**（图7）验证了 Execution Module 各优化的累积效果：
  1. **Algorithm 1**（基础 LCMA）：相对 cuBLAS 平均 **+5.32%**。
  2. **Group-Parallel**：因粗粒度并行导致尾部效应，性能不稳定。
  3. **Split-Group Parallelism**：解决负载不均，小矩阵性能提升。
  4. **Cache-Aware Scheduling**：缓解缓存冲突，最终实现 **+7.83%** 的稳定加速，验证了各优化的有效性。

- **Roofline 分析**（图8）：
  - 验证了 Decision Module 的有效性：在高算术强度区域选择高 R 的 LCMA，在低算术强度区域回退到标准 GEMM。
  - 实现了 **19.31%** 和 **11.37%** 的性能增益，分别相对于标准 GEMM 和 Strassen 算法。

---

## 4. 关键结论和发现

### 主要发现
1. **LCMAs 可以实用化**：通过 FalconGEMM 的系统性优化，LCMAs 的理论算力优势可以在真实硬件上稳定转化为“突破硬件峰值”的实际性能提升。
2. **融合执行至关重要**：单纯的 LCMA 实现因内存开销无法胜出，必须通过 **Group-Parallel** 等融合技术消除中间结果的片外访问。
3. **智能决策不可或缺**：没有万能的最快算法，轻量级分析模型是实现全自动、自适应优化的关键。
4. **跨平台自动化可行**：基于现代深度学习编译器的代码生成，能够有效应对百万级实现变体的工程复杂性。

### 方法的局限性
- **数值精度**：LCMAs 固有地比标准 GEMM 数值稳定性差，尽管 FalconGEMM 通过融合减少了精度损失，但仍是一个潜在风险，尤其在对精度敏感的任务中。
- **适用范围**：主要针对稠密 GEMM，对稀疏矩阵的支持未深入探讨。
- **实现复杂度**：虽然框架自动化，但底层仍高度复杂，依赖于先进的编译器技术。

### 未来工作方向
- 扩展至更多类型的 **structured matrices** 或 **sparse-dense 混合计算**。
- 探索更鲁棒的 **numerical stabilization** 技术。
- 将框架集成到主流 DL 框架（如 PyTorch, TensorFlow）作为默认后端。
- 支持更多新兴硬件（如 TPUs, 国产 AI 芯片）和数据格式。

</details>

---

### 16. [Expert Routing for Communication-Efficient MoE via Finite Expert Banks](https://arxiv.org/abs/2605.05278)

**Authors**: Mohammad Reza Deylam Salehi, Ali Khalesi  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.05278v1  

#### Abstract
Resource-efficient machine learning increasingly uses sparse Mixture-of-Experts (MoE) architectures, where the gate acts as both a learning component and a routing interface controlling computation, communication, and accuracy. Motivated by finite-rate interpretations of MoE gating, we treat the gat...

---

### 17. [FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking](https://arxiv.org/abs/2605.05482)

**Authors**: Denys Katerenchuk, Pablo Duboue, Keelan Evanini, David Gondek, Nithin Govindugari, Olivier Allauzen, Joshua Baptiste, David J More, Joshua Schechter  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.05482v1  

#### Abstract
Large language models (LLMs) are rapidly being adopted across various domains. However, their adoption in banking industry faces resistance due to demands for high accuracy, regulatory compliance, and the need for verifiable and grounded responses. We present a unified, data-efficient framework for ...

---

### 18. [Shallow Prefill, Deep Decoding: Efficient Long-Context Inference via Layer-Asymmetric KV Visibility](https://arxiv.org/abs/2605.06105)

**Authors**: Jungsuk Oh, Hyeseo Jeon, Hyunjune Ji, Kyongmin Kong, Jay-Yoon Lee  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.06105v1  

#### Abstract
Long-context inference in decoder-only language models is costly because long prompts are processed during Prefill, cached at every layer, and repeatedly attended to during autoregressive Decode. We introduce \emph{Shallow Prefill, dEEp Decode} (SPEED), a phase-asymmetric KV-visibility policy that m...

---

### 19. [Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving: Practical Online Routing at Scale](https://arxiv.org/abs/2605.06113)

**Authors**: Tianci Bu, Yuan Lyu, Zixi Chen, Chendong Song, Hong Liang, Tsepten Gurung, Yuwei Fan, Yinyu Ye, Zijie Zhou  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.06113v1  

#### Abstract
Data-parallel (DP) load balancing has emerged as a first-order bottleneck in large-scale LLM serving. When a model is sharded across devices via tensor parallelism (TP) or expert parallelism (EP) and replicated across many DP workers, every decode step ends in a synchronization barrier whose latency...

---

### 20. [HCInfer: An Efficient Inference System via Error Compensation for Resource-Constrained Devices](https://arxiv.org/abs/2605.05819)

**Authors**: Shen Xu, Xiangwen Zhuge, Zhe Xu, Yingkun Hu, Zheng Yang, Yunhao Liu  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.05819v1  

#### Abstract
LLMs often struggle with memory-constrained deployment on consumer-grade hardware due to their massive parameter sizes. While existing solutions such as model compression and offloading improve deployment feasibility, they often suffer from substantial accuracy degradation or severe throughput bottl...

---

### 21. [Federation of Experts: Communication Efficient Distributed Inference for Large Language Models](https://arxiv.org/abs/2605.06206)

**Authors**: Muhammad Shahir Abdurrahman, Chun Deng, Azalia Mirhoseini, Philip Levis  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.06206v1  

#### Abstract
Mixture of experts has emerged as the primary mechanism for making Large Language Models (LLMs) computationally efficient. However, in distributed settings, communicating token embeddings between experts is a significant bottleneck.

---

### 22. [Large Vision-Language Models Get Lost in Attention](https://arxiv.org/abs/2605.05668)

**Authors**: Gongli Xi, Ye Tian, Mengyu Yang, Huahui Yi, Liang Lin, Xiaoshuai Hao, Kun Wang, Wendong Wang  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.05668v1  

#### Abstract
Despite the rapid evolution of training paradigms, the decoder backbone of large vision--language models (LVLMs) remains fundamentally rooted in the residual-connection Transformer architecture. Therefore, deciphering the distinct roles of internal modules is critical for understanding model mechani...

---

### 23. [Saliency-Aware Regularized Quantization Calibration for Large Language Models](https://arxiv.org/abs/2605.05693)

**Authors**: Yanlong Zhao, Xiaoyuan Cheng, Huihang Liu, Baihua He, Xinyu Zhang, Harrison Bo Hua Zhu, Wenlong Chen, Li Zeng, Zhuo Sun  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.05693v1  

#### Abstract
Post-training quantization (PTQ) is an effective approach for deploying large language models (LLMs) under memory and latency constraints. Most existing PTQ methods determine quantization parameters by minimizing a layer-wise reconstruction error on a predetermined calibration dataset, usually optim...

---

### 24. [Event-Causal RAG: A Retrieval-Augmented Generation Framework for Long Video Reasoning in Complex Scenarios](https://arxiv.org/abs/2605.06185)

**Authors**: Peizheng Yan, Yu Zhao, Liang Xie, Juntong Qi, Mingming Wang, Erwei Yin  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.06185v1  

#### Abstract
Recent large vision-language models have achieved strong performance on short- and medium-length video understanding, yet they remain inadequate for ultra-long or even infinite video reasoning, where models must preserve coherent memory over extended durations and infer causal dependencies across te...

---

### 25. [Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning](https://arxiv.org/abs/2605.06241)

**Authors**: \"Omer Faruk Akg\"ul, Rajgopal Kannan, Willie Neiswanger, Viktor Prasanna  
**Category**: cs.CL  
**Published**: 2026-05-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.06241v1  

#### Abstract
Reinforcement learning has become the standard for improving reasoning in large language models, yet evidence increasingly suggests that RL does not teach new strategies; it redistributes probability mass over solutions the base model already contains. In this work, we ask: if RL merely steers the m...

---

### 26. [Beyond Negative Rollouts: Positive-Only Policy Optimization with Implicit Negative Gradients](https://arxiv.org/abs/2605.06650)

**Authors**: Mingwei Xu, Hao Fang  
**Category**: cs.CL  
**Published**: 2026-05-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.06650v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR), due to the deterministic verification, becomes a dominant paradigm for enhancing the reasoning ability of large language models (LLMs). The community witnesses the rapid change from the Proximal Policy Optimization (PPO) to Group Relative Policy...

---

### 27. [Towards Scalable One-Step Generative Modeling for Autoregressive Dynamical System Forecasting](https://arxiv.org/abs/2605.05540)

**Authors**: Tianyue Yang, Xiao Xue  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.05540v1  

#### Abstract
Fast surrogate modeling for high-dimensional physical dynamics requires more than low short-term error: useful models must roll out efficiently while preserving the statistical structure of long trajectories. Neural operators provide inexpensive autoregressive forecasts but can drift in turbulent re...

---

### 28. [QuadraSHAP: Stable and Scalable Shapley Values for Product Games via Gauss-Legendre Quadrature](https://arxiv.org/abs/2605.05870)

**Authors**: Majid Mohammadi, Grigory Reznikov, Pavel Sinitcyn, Krikamol Muandet, Siu Lun Chau  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.05870v1  

#### Abstract
We study the efficient computation of Shapley values for \emph{product games} -- cooperative games in which the coalition value factorizes as a product of per-player terms. Such games arise in machine learning explainability whenever the value function inherits a multiplicative structure from the un...

---

### 29. [Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions](https://arxiv.org/abs/2605.05983)

**Authors**: Yuntai Bao, Qinfeng Li, Xinyan Yu, Xuhong Zhang, Ge Su, Wenqi Zhang, Liu Yan, Haiqin Weng, Jianwei Yin  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.05983v1  

#### Abstract
Recently, steering vectors (SVs) have emerged as an effective and lightweight approach to steer behaviors of large language models (LLMs), among which fine-tuned SVs are more effective than optimization-free ones. However, current approaches to fine-tuned SVs suffer from two limitations. First, they...

---

### 30. [BoostLLM: Boosting-inspired LLM Fine-tuning for Few-shot Tabular Classification](https://arxiv.org/abs/2605.06117)

**Authors**: Yi-Siang Wang, Kuan-Yu Chen, Yu-Chen Den, Darby Tien-Hao Chang  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.06117v1  

#### Abstract
Large language models (LLMs) have recently been adapted to tabular prediction by serializing structured features into natural language, but their performance in low-data regimes remains limited compared to gradient-boosted decision trees (GBDTs). In this work, we revisit the boosting paradigm, tradi...

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

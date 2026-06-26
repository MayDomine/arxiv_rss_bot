# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-26 08:46:03 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM](https://arxiv.org/abs/2606.26120)

**Authors**: Tianyi Wu, Xiaoxi Sun, Yanhua Jiao, Yulin Li, Yixin Chen, YunHao Cao, YiQi Hu, Zhuotao Tian  
**Category**: cs.CL  
**Published**: 2026-06-26  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.26120v1  

#### Abstract
Diffusion Large Language Models (dLLMs) offer a promising alternative to autoregressive models, excelling in text generation tasks due to their bidirectional attention mechanisms. However, their computational complexity scales on the order of L cubed with the sequence length L. This poses significan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
Diffusion Large Language Models (dLLMs) 虽然在文本生成任务中表现出色，但由于其非自回归（non-autoregressive）特性，推理过程需要在每一步对整个序列进行并行去噪，导致计算复杂度高达 $O(L^3)$，远高于自回归模型（AR LLMs）的 $O(L^2)$。此外，dLLMs 无法利用传统的 **KV-Cache** 机制，且现有加速方法多采用静态策略（如固定缓存比例或固定解码阈值），忽略了 token 在不同层和不同步骤中的动态行为，从而限制了效率提升。

### **提出了什么新方法或新思路**
本文提出 **Dynamic-dLLM**，一个无需训练的推理加速框架，包含两个核心组件：

- **Dynamic Cache Updating (DCU)**  
  动态分配各层的缓存更新预算（cache-update budget），基于 token 表示在相邻步骤间的输入变化（通过 cosine distance 估计）来判断是否需要更新缓存。同时引入 **Mandatory Update Window** 防止关键 token “陷入泥潭”（token stuck in the mud）。

- **Adaptive Parallel Decoding (APD)**  
  引入动态解码阈值机制，根据每个 token 的预测分布集中度（如第二高概率）和历史置信度波动（cosine distance of confidence distributions）来自适应调整解码阈值，实现更早、更安全的 token 解锁。

### **相比现有方法的优势**
- **完全无需训练**：即插即用，适用于任何 dLLM。
- **动态适应性**：突破了现有方法（如 dLLM-Cache, Fast-dLLM）中静态缓存/解码策略的局限，能根据层间和步间 token 动态特性灵活调整资源分配。
- **高效且保质**：在显著提升吞吐量的同时，几乎不损失模型准确率。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖多个主流基准，涵盖通用知识、数学、科学、编程等任务：
- **MMLU** (5-shot)：大规模多任务语言理解
- **ARC-C** (0-shot)：AI2 Reasoning Challenge
- **GSM8K** (4-shot)：小学数学应用题
- **GPQA** (5-shot)：研究生级别问答
- **HumanEval** (0-shot)：代码生成能力

### **实验设置和评估指标**
- **模型**：LLaDA-8B-Instruct、LLaDA-1.5、Dream-v0-7B-Instruct
- **硬件**：NVIDIA Pro6000 GPU（主实验），RTX 4090（低资源验证）
- **评估指标**：
  - **准确率（Accuracy）**：各项任务的得分
  - **吞吐量（Throughput）**：Tokens Per Second (TPS)，衡量推理速度
  - **相对加速比**：以 baseline 为 1.0×，计算 TPS 提升倍数

### **基线方法对比**
| 方法 | 类型 | 特点 |
|------|------|------|
| **dLLM-Cache** | Feature Caching | 固定比例缓存更新，跨层一致 |
| **dKV-Cache** | KV Caching | 缓存已解码 token，不再更新 |
| **Fast-dLLM** | Parallel Decoding + Cache | 固定阈值并行解码，块内更新 |
| **Dynamic-dLLM (Ours)** | 动态缓存 + 自适应解码 | 本文方法，双模块协同 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
在 **LLaDA-8B-Instruct** 上的平均加速效果：
- **仅启用 DCU**：平均 **2.63×** 加速
- **结合 APD（并行解码）**：平均 **3.21×** 加速，最高达 **4.48×**（GSM8K 任务）

| 模型 | 任务 | Dynamic-dLLM TPS | Baseline TPS | 加速比 |
|------|------|------------------|---------------|--------|
| LLaDA-8B | GSM8K | 37.29 | 8.32 | **4.48×** |
| LLaDA-1.5 | GSM8K | 37.02 | 8.30 | **4.46×** |
| Dream-7B | GSM8K | 31.48 | 8.05 | **3.91×** |

> 所有方法均保持与 baseline 几乎相同的准确率（误差 < 0.5%）。

### **与基线方法的对比结果**
- **在所有任务上，Dynamic-dLLM 均取得最高的 TPS 和加速比**。
- 相比 Fast-dLLM，在 GSM8K 上提速 **~20–30%**，且避免了因固定阈值导致的过早锁定错误 token。
- 在 HumanEval 上，Dynamic-dLLM 达到 **2.37×** 加速（仅 APD）和 **3.21×**（完整版），显著优于其他方法。

### **消融实验结果**
#### **(1) Blayer 与 Bwindow 的影响（图6a-b）**
- 当 `Blayer` 和 `Bwindow` 增大时，准确率上升但吞吐下降。
- 经权衡，选择 **32** 作为默认值，在性能与效率间达到最佳平衡。
- 实验表明，强制更新窗口（Bwindow）对防止 token 卡死至关重要。

#### **(2) 动态 vs. 固定阈值（图6c）**
- 在相同初始阈值下，**动态阈值可减少约 30% 的推理步数**。
- 尤其在高初始阈值（如 0.9）时优势明显，避免了因保守阈值导致的冗余迭代。

#### **(3) 超参数稳定性分析（表7）**
- 超参数 $\alpha=0.001$, $\beta=0.0008$ 在不同生成长度下表现稳定。
- 过大的 $\alpha/\beta$ 会导致质量严重下降（>10% 准确率损失），而微调则影响较小。

---

## **4. 关键结论和发现**

### **主要发现**
1. **dLLM 中 token 的动态行为具有显著的层间和步间差异**，静态策略无法充分利用这些特性。
2. **输入层的变化可有效代理中间特征的更新需求**，无需重复计算 Key/Value 向量即可指导缓存更新。
3. **动态调整缓存预算 + 自适应解码阈值** 可协同优化效率与质量，实现 **>3× 平均加速** 而无性能损失。
4. **Mandatory Update Window** 有效缓解“token stuck in the mud”问题，保障局部上下文响应能力。

### **方法的局限性**
- 当前设计主要针对 **单模态文本输入**，未考虑多模态场景下的跨模态对齐与表示融合。
- 对于极长序列（>1k），需适当调大 `Blayer` 和 `Bwindow` 以维持精度，可能略微增加开销。
- 依赖 token 输入相似性作为代理信号，极端情况下可能误判（如语义不变但表示漂移）。

### **未来工作方向**
- 探索 **多模态 dLLM** 中的动态缓存与解码机制。
- 设计 **自动超参数调节策略**（如根据任务难度或输入长度自适应调整 $\alpha, \beta$）。
- 将 Dynamic-dLLM 思路扩展至 **video diffusion models** 或 **structured generation** 任务。
- 研究如何将该框架应用于 **边缘设备部署**，进一步降低延迟与内存占用。

---

> ✅ **代码开源地址**：[https://github.com/TianyiWu233/DYNAMIC-DLLM](https://github.com/TianyiWu233/DYNAMIC-DLLM)  
> 📌 **一句话总结**：Dynamic-dLLM 通过**动态感知 token 层级与时间维度的行为变化**，实现了无需训练、即插即用的高效 dLLM 推理加速方案，在保持性能的同时达成 **平均 3× 以上加速**，是当前最先进的 training-free dLLM 加速框架。

</details>

---

### 2. [Simulating Unified Tensor Resharding in heterogeneous AI systems](https://arxiv.org/abs/2606.26633)

**Authors**: Sumit Kumar, Sayantan Dasgupta, Kushal Mitra, Meet Dadhania, Rohan Sudhir Basugade, Praveen Tammana, Satananda Burla, Abed Mohammad Kamaluddin, Rinku Shah  
**Category**: cs.DC  
**Published**: 2026-06-26  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.26633v1  

#### Abstract
State-of-the-art AI training simulators assume homogeneous compute and network infrastructure. However, real-world training infrastructure is becoming increasingly heterogeneous since: (a) Model architectures such as multimodal and MoE exploit heterogeneity to improve device utilization, (b) Public ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Simulating Unified Tensor Resharding in heterogeneous AI systems*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现代大规模 AI 训练系统日益**异构化**（heterogeneous），体现在：
- **硬件异构**：不同代际的 GPU（如 A100、H100、H200）混合部署；
- **网络异构**：NVLink、PCIe、Ethernet、InfiniBand 等多种互联技术并存；
- **部署异构**：跨数据中心、云上资源动态分配导致非均匀设备配置。

然而，现有的主流训练模拟器（如 ASTRA-sim、SimAI、Multiverse 等）均假设**同构基础设施**，无法准确建模异构环境下的负载均衡、通信同步与性能瓶颈，导致对真实训练时间预测偏差大，误导系统设计。

本文提出 **Xsim**，是首个支持**异构 AI 训练全栈模拟**的系统级仿真器，填补了该领域的空白。

---

### **提出了什么新方法或新思路**

Xsim 的核心创新在于引入了一套**统一且可扩展的抽象机制**，以支持在任意异构配置下进行高效的 tensor resharding 和 collective communication 模拟。

#### 主要技术贡献：

1. **非均匀工作负载划分（Non-uniform Workload Partitioning）**
   - 支持按设备算力（TFLOPS）、内存容量等属性，将模型层、数据批次、张量切片**非均匀地**分配到不同 Device Group（DG）中，提升资源利用率。

2. **基于扫描线算法的动态 DP 分组（Sweep-line-based DP Grouping）**
   - 针对 pipeline-parallel 中各 stage 处理不同层数的问题，提出扫描线算法，自动识别重叠层段，并为每个层段构建独立的 DP synchronization group，避免无效同步。

3. **基于 LCM 的多环集体通信与张量分块（LCM-based Multi-Ring Collectives & Chunking）**
   - 当不同 DG 具有不同 TP degree（如 TP=3 vs TP=2）时，通过计算其最小公倍数（LCM）来定义统一的细粒度 chunk 大小；
   - 构建多个独立的通信环（multi-ring），每个 ring 负责一个 chunk 的 AllReduce，实现跨异构 TP 配置的高效同步；
   - 所有参与 rank 在同一 ring 内处理相同大小的数据块，保证正确性和公平性。

4. **灵活的输入抽象与插件式网络后端**
   - 支持自定义 Device Group、并行策略映射、拓扑描述（scale-up/scale-out）；
   - 可插拔集成 **NS-3**（高保真包级模拟）和 **htsim**（高速流级模拟），允许用户在精度与速度之间权衡。

5. **暴露可操作的性能指标**
   - 提供 `pipeline bubble time`、`straggler waiting time`、`TCO`（Total Cost of Ownership）等指标，辅助容量规划与优化决策。

---

### **相比现有方法的优势**

| 维度 | 现有模拟器（如 SimAI） | Xsim |
|------|------------------------|------|
| 异构支持 | ❌ 不支持 | ✅ 完整支持 |
| 非均匀分区 | ❌ 假设均匀 | ✅ 支持 |
| 张量重分片（resharding） | ❌ 忽略或简化 | ✅ 统一建模（LCM-based） |
| 通信拓扑灵活性 | ⚠️ 固定结构 | ✅ 自适应 multi-ring |
| 网络模拟可扩展性 | ⚠️ 包级模拟慢 | ✅ 支持 flow-level（htsim）加速达 47× |
| 性能预测准确性 | ❌ 在异构场景误差高达 80% | ✅ 平均误差 <5% |

---

## 2. 核心实验方法和设置

### **使用的模型与硬件配置**

- **模型**：
  - Llama-2 7B / 13B
  - GPT-175B
  - Llama-450B（用于扩展性测试）

- **硬件平台**：
  - **真实集群**：包含 A100（40GB）、H100（80GB）、H200（141GB）、B200（192GB）等多种 GPU；
  - **互联方式**：
    - Scale-up：NVLink（Gen3/Gen4）、PCIe（Gen4/Gen5）
    - Scale-out：Ethernet（10G/400G）、InfiniBand、RoCEv2

- **异构配置示例**（见 Table 4）：
  - C9: 1×A100 + 1×H100 （纯 DP）
  - C10–C16: 混合 TP/DP/PP 配置，跨异构节点

---

### **实验设置与评估指标**

#### **评估问题（Evaluation Questions）**

共设计 10 个核心问题，涵盖模拟精度、扩展性、通信建模、成本分析等方面。

#### **主要评估指标**

| 指标 | 描述 |
|------|------|
| **Prediction Accuracy** | 模拟训练时间 vs 真实执行时间的相对误差（%） |
| **Simulation Runtime** | 单次迭代模拟所需墙钟时间（wall-clock time） |
| **Straggler Waiting Time** | GPU 因同步等待而空闲的时间 |
| **Pipeline Bubble Time** | pipeline 中因通信/resharding 导致的空泡时间 |
| **TCO ($/GPU-hour)** | 单位训练时间的总拥有成本，反映性价比 |
| **Collective Communication Time** | AllReduce、Grad Gather 等操作耗时 |

#### **基线方法对比**

| 基线 | 类型 | 说明 |
|------|------|------|
| **SimAI [60]** | 同构模拟器代表 | 仅支持均匀配置，作为 baseline 对比异构建模能力 |
| **HexiScale [65]** | 异构训练框架 | 提供部署计划，其分析模型用于对比 |
| **HetAuto [44]** | 异构 resharding 方案 | GCD-based 三阶段（Gather→P2P→Scatter） |
| **AlpaComm [75]** | 异构 resharding 方案 | Cutpoint-union，生成不规则 chunk |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **训练时间预测精度高**

- 在多数异构 DP/TP 配置下，Xsim 的训练时间预测误差 **< 5%**；
- 在 pipeline-parallel 场景下，误差进一步降至 **~2%**；
- 相比之下，SimAI 在异构场景下误差高达 **80%**（图6）；
- HexiScale 的分析模型低估训练时间 **90–99%**，因其忽略通信开销。

> 图7 显示 Xsim 与真实硬件高度一致，而 SimAI 和 HexiScale 偏差显著。

#### ✅ **通信建模精准**

- **Scale-up（NVLink）通信**：Xsim（NS-3/htsim）对 TP collective 时间预测平均误差 **< 5.5%**（图9）；
- **Scale-out（DP multi-ring）通信**：对梯度同步时间预测随模型增大误差减小，最大误差约 **7%**（图10）；
- 层级粒度验证显示，compute 和 communication 操作误差均控制在 **1–4%** 内（图11）。

#### ✅ **模拟效率高，可扩展性强**

- 使用 **htsim** 作为后端时，相比 NS-3 加速 **16×–47×**（图8）；
- 在 1024 节点规模下仍保持高效运行；
- 随着集群规模增加，Xsim 实现最高 **6.25× 的模拟加速**（图17）；

#### ✅ **优于现有 resharding 方法**

| 方法 | 总训练时间 | Pipeline Bubble Time |
|------|------------|---------------------|
| **HetAuto** | 较高（尤其不对称配置） | 低（chunk 均匀） |
| **AlpaComm** | 较低 | 较高（chunk 不均导致负载倾斜） |
| **Xsim (LCM-based)** | **最低** | **最低** |

> Xsim 在对称和非对称配置中均表现最优，兼顾通信效率与负载均衡（图12）。

#### ✅ **消融实验结果（Ablation Study）**

- **LCM-based chunking 减少事件数量**：相比 SimAI 的 mockNCCL 实现，Xsim 减少了通信事件碎片化，降低调度开销；
- **htsim 加速有效**：在不需要协议细节时，flow-level 模拟能大幅提速；
- **非均匀分区显著减少 straggler**：从 DP 配置的 4.42 秒 idle time 下降到 TP+PP 的 0.07 秒（图18）；
- **异构部署可降低成本**：混合 H100+A100 集群比纯 H100 节省 **~30% TCO**，同时性能相当（图19）。

---

## 4. 关键结论和发现

### **主要发现**

1. **异构建模至关重要**：忽略硬件/网络异构性会导致训练时间预测严重失真（误差 >80%），影响系统设计决策。
2. **统一的 resharding 抽象可行且高效**：Xsim 提出的 LCM-based multi-ring 机制能统一处理任意 TP/DP/PP 组合下的 tensor layout mismatch，无需依赖特定 resharding 算法。
3. **非均匀分区 + 自适应通信 = 高效利用异构资源**：通过匹配设备能力分配 workload，并配合定制化通信拓扑，可显著减少 straggler 和 pipeline bubbles。
4. **异构集群可以更便宜且高效**：合理组合新旧 GPU 可实现接近纯高端集群的性能，但 TCO 显著更低，具备商业价值。
5. **模拟器需支持精度/速度权衡**：htsim 提供了实用的 flow-level 替代方案，在大规模探索中极具优势。

---

### **方法的局限性**

1. **LCM 最大值受限于实际 TP degree**：虽然理论上 LCM 可能增长，但实践中 TP ≤ 8，因此 LCM 上限为 840，仍可控（附录 E）；
2. **未模拟 NCCL 内部资源竞争**：真实环境中 multi-ring 可能因 link allocation 序列化而导致轻微延迟，当前模型假设完全并行；
3. **尚未支持所有 PP 调度策略**：目前主要实现 GPipe，对 1F1B、Interleaved 等仍在扩展中；
4. **依赖高质量 trace 生成**：性能预测精度依赖于 SUTRAAwG 生成的工作负载 trace 是否准确反映真实行为。

---

### **未来工作方向**

1. **支持更多 pipeline-parallel 调度算法**：如 1F1B、Bubble-Free Pipelining 等；
2. **集成学习-based planner**：结合 RL 或搜索算法自动优化异构部署策略；
3. **支持 disaggregated architecture**：如 CXL、GPU pooling 等新型解耦架构；
4. **开放仿真平台生态**：推动 Xsim 成为异构 AI 训练的标准评估基准工具链；
5. **与生产系统联动**：实现“仿真-部署-反馈”闭环，持续优化真实集群性能。

--- 

> **总结一句话**：  
> **Xsim 是首个真正面向现实世界的异构 AI 训练模拟器，它通过 LCM-based 统一张量重分片机制，实现了高精度、高可扩展性的全栈性能建模，为下一代异构训练系统的协同设计提供了可靠基础。**

</details>

---

### 3. [DMuon: Efficient Distributed Muon Training with Near-Adam Overhead](https://arxiv.org/abs/2606.27153)

**Authors**: Vincent Chen, Starrick Liu, Regis Cheng, Dance Yang, Shalfun Li, Ryan Yu, Lucy Liang, Hang Su, Roy Gan, Hao Wang, Qian Wang  
**Category**: cs.DC  
**Published**: 2026-06-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.27153v1  

#### Abstract
Matrix-orthogonalization-based optimizers, exemplified by Muon, have demonstrated strong convergence behavior across a wide range of modern deep learning workloads. The matrix-aware updates offer a compelling alternative to conventional element-wise optimization, particularly as model architectures ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**DMuon: Efficient Distributed Muon Training with Near-Adam Overhead**

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代大规模深度学习训练广泛采用 **element-wise optimizers**（如 AdamW），而新兴的 **matrix-aware optimizers**（如 Muon）因其更优的收敛特性受到关注。然而，Muon 这类优化器在分布式训练中面临严重系统开销问题：

- **粒度不匹配（Granularity Mismatch）**：分布式训练框架（如 FSDP、ZeRO）通常以参数分片（shard）为单位执行优化器步骤，而 Muon 需要对整个权重矩阵进行 **Newton-Schulz 迭代**，必须先将所有分片聚合才能计算更新。
- **通信与计算冗余**：传统实现方式（gather-then-compute）导致每个 rank 都需全量收集梯度并重复执行相同的正交化计算，造成显著的通信和计算开销，使得每步耗时远超 AdamW。

这使得尽管 Muon 在算法层面具有优势，但在实际训练中难以部署。

---

### 提出了什么新方法或新思路
作者提出 **DMuon** —— 一个高效、可扩展的分布式 Muon 实现，作为即插即用模块集成到现有训练流程中，无需修改框架底层代码。

#### 核心创新点包括：
1. **Owner-Centric 执行策略**
   - 每个 weight matrix 被分配给一个唯一的 **owner rank**。
   - 只有 owner 执行完整的 Muon 更新（Newton-Schulz iteration），避免跨 rank 冗余计算。
   - 非 owner ranks 仅在前向/反向传播时临时 materialize 参数。

2. **细粒度通信优化（Fine-grained Communication Optimization）**
   - 将参数发布（broadcast）和梯度归约（reduce）分解为 **intra-node 和 inter-node 两级通信**。
   - 利用 XOR 规则设计均衡的 owner 分布，使并发通信分散在不同通信组，减少拥塞。
   - 通过 **pipeline 设计** 实现通信与计算重叠（overlap），隐藏通信延迟。

3. **形状自适应执行引擎（Shape-adaptive Execution Stack）**
   - 采用 **Gram Newton-Schulz 公式**，将迭代从 $O(m^2n)$ 降为 $O(m^3)$（当 $m < n$ 时更高效）。
   - 引入 **对称感知内核（symmetry-aware kernels）**，利用 Gram 矩阵对称性，只计算下三角部分，节省近一半算力。
   - 支持 **批处理小矩阵（batching small matrices）**，提升 GPU 利用率，摊薄 kernel 启动开销。

4. **基于实测的负载均衡（Computation-aware Load Balancing）**
   - 不依赖理论模型，而是通过运行时 profiling 获取每种矩阵形状的实际执行时间。
   - 将 owner 分配建模为 **混合整数线性规划（MILP）问题**，最小化最慢 rank 的完成时间（makespan）。
   - 大规模场景下自动 fallback 到贪心策略，保证初始化开销可控。

5. **即插即用模块化设计（Drop-in Module）**
   - 完全兼容 PyTorch FSDP/HSDP，支持 Tensor Parallelism。
   - 用户只需三行代码即可替换 AdamW，保留原有训练逻辑不变。

---

### 相比现有方法的优势
| 维度 | 传统方法（Vanilla Muon / Muon-AG） | DMuon |
|------|-------------------------------|--------|
| 通信模式 | All-gather every step | Owner-centric hierarchical broadcast/reduce |
| 计算模式 | 每个 rank 独立执行相同 NS iteration | 仅 owner 执行一次，消除冗余 |
| 性能 | optimizer step 时间 > forward + backward | 接近 AdamW 水平 |
| 易用性 | 需深度定制训练栈 | 三行 API 替换，零侵入 |

---

## 2. 核心实验方法和设置

### 使用的模型与任务
实验覆盖多种前沿大模型训练场景，包括：
- **机器人基础模型（Embodied Foundation Models）**：
  - `WALL-OSS-0.5`（Vision-Language-Action, VLA）
  - `Pi0`
- **大型语言模型（LLM）**：
  - `WALL-WM`
  - `Qwen2.5-7B`

这些模型代表了当前高异构性、多模态、长序列控制等复杂训练负载。

---

### 实验设置
- **硬件平台**：A800-SXM4-80GB GPU 集群（8卡/节点，NVLink + 200Gb/s InfiniBand）
- **精度格式**：bf16（训练主精度），fp16（Newton-Schulz iteration 中间计算）
- **并行策略**：FSDP（Fully Sharded Data Parallelism）、HSDP、Tensor Parallelism 支持
- **GPU 数量范围**：8 ~ 256 GPUs，验证可扩展性

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **End-to-end step time** | 单个训练 step 的总 wall-clock 时间（ms） |
| **Optimizer step time** | 优化器更新阶段耗时（不含 forward/backward） |
| **Speedup** | 相对于 vanilla gather-then-compute Muon 的加速比 |
| **△A (%)** | DMuon 相对于 AdamW 的额外开销百分比 |
| **Throughput (tokens/sec)** | 模型吞吐能力 |

---

### 基线方法对比
1. **AdamW**：主流 element-wise optimizer，作为性能天花板基准。
2. **Vanilla Muon (Muon-AG)**：gather-then-compute 实现，每个 rank 全局 all-gather 梯度后独立运行 NS iteration。
3. **DMuon**：本文提出的方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 模型 | GPU 数 | DMuon Step Time (ms) | AdamW Step Time (ms) | △A (%) | Optimizer Speedup vs Vanilla |
|------|-------|------------------------|------------------------|--------|------------------------------|
| WALL-OSS | 256 | 1519 | 1496 | +1.5% | 109.44× |
| PiO | 256 | 1648 | 1637 | +1.2% | 93.43× |
| WALL-WM | 256 | 3011 | 2915 | +3.3% | 96.89× |
| Qwen2.5-7B | 256 | 2850 | 2844 | +0.2% | 163.82× |

> ✅ **平均来看，DMuon 的端到端 step time 仅比 AdamW 慢约 2%**，实现了“near-Adam overhead”。

---

### 与基线方法的对比结果
- **端到端 step time 加速比**：相比 vanilla Muon，DMuon 实现 **1.48× ~ 3.01×** 的整体加速。
- **optimizer step time 加速比**：达到惊人的 **6.85× ~ 163.00×**，说明其成功消除了主要瓶颈。
- **可扩展性表现优异**：随着 GPU 数增加，DMuon 吞吐持续逼近 AdamW，且在低卡数下甚至因更大 batch size 获得更高吞吐（见 Figure 8）。

---

### 消融实验结果（Table 2）
分析各组件对 optimizer step speedup 的贡献比例：

| 组件 | 对加速的贡献占比 |
|------|------------------|
| **Symmetric Gram Kernel** | 48% |
| **Owner Scheduling & Load Balancing** | 32% |
| **Auto-tuning & NS Batching** | 16% |

> 🔍 结论：**对称性优化是最大收益来源**，其次是负载均衡；三者协同作用，共同将 Muon 开销压缩至接近 AdamW。

---

## 4. 关键结论和发现

### 主要发现
1. **Muon 的收敛优势可以转化为实际训练速度优势**：过去 Muon 因高昂系统开销被限制在小规模实验，DMuon 成功将其带入生产级训练。
2. **分布式矩阵优化器的关键在于打破“element-wise”假设下的执行范式**：通过 owner-centric + pipelined communication + shape-aware kernel，可有效解决粒度不匹配问题。
3. **通信与计算的 overlap 是降低感知延迟的核心手段**：即使无法完全消除通信，只要能与计算并行，就能极大缓解影响。
4. **负载均衡必须基于实测而非理论估算**：由于 batching、autotuning 等动态因素，静态成本模型失效，需 runtime profiling + MILP 才能达到最优。

---

### 方法的局限性
- **不改变 Muon 算法本身**：DMuon 是数学等价的系统优化，因此继承 Muon 的全部特性（包括潜在数值稳定性问题）。
- **单卡训练收益有限**：虽然仍可提速 ~2×，但无法发挥分布式通信优化的优势。
- **对极小矩阵优化空间较小**：若模型中大量参数为 scalar 或 vector 形式，则 Muon 本身的适用性受限。
- **初始化阶段有一定开销**：profiling + MILP 求解需要时间，虽可接受但仍非零成本。

---

### 未来工作方向
1. **扩展至其他 matrix-aware optimizers**：如 Shampoo、SOAP、Dion 等，构建通用分布式矩阵优化器运行时。
2. **支持动态形状模型（如 MoE、稀疏激活）**：当前假设参数形状固定，未来可探索动态负载再平衡机制。
3. **结合低秩近似进一步降低计算成本**：例如在 NS iteration 中引入低秩分解，在保持效果的同时减少 $O(m^3)$ 开销。
4. **硬件协同设计**：针对 NS iteration 特性设计专用 kernel 或 FPGA/ASIC 加速方案。

---

## 总结
✅ **DMuon 成功解决了 Muon 在分布式训练中的高开销难题，使其成为真正实用的“drop-in replacement for AdamW”**。

它不仅提升了训练效率，更重要的是推动了 **matrix-aware optimization 范式向主流生产环境落地**，为下一代高性能大模型训练提供了新的基础设施选择。

🔗 **开源地址**：[https://github.com/X-Square-Robot/dmuon](https://github.com/X-Square-Robot/dmuon)

</details>

---

### 4. [Mapping Political-Elite Networks in Europe with a Multilingual Joint Entity-Relation Extraction Pipeline](https://arxiv.org/abs/2606.27347)

**Authors**: Kirill Solovev, Jana Lasser  
**Category**: cs.CL  
**Published**: 2026-06-26  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.27347v1  

#### Abstract
Whether political elites organise into rent-seeking coalitions that capture public resources or civic networks that sustain governance is a central question in comparative politics. Yet observing these complex, informal, and adversarial ties at scale has historically required intensive manual coding...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Mapping Political-Elite Networks in Europe with a Multilingual Joint Entity-Relation Extraction Pipeline*

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**大规模、跨语言政治精英网络研究中的数据瓶颈**。传统上，构建此类网络依赖于人工编码，成本高昂且难以扩展；而现有的自动化文本分析方法（如共现分析）过于简单，无法捕捉复杂、非正式且带有对抗性的关系。此外，基于大模型（LLM）的方法常受限于专有API、缺乏跨语言能力以及实体消解（entity resolution）的可扩展性。

### 提出的新方法与创新
作者提出了一种**模块化、完全开源权重（open-weight）的多语言联合实体-关系抽取管道（pipeline）**，其核心创新如下：

- **模块化与开放性**：整个流程由独立的、可替换的模块构成（NER、实体链接、关系抽取），全部基于开源模型，避免了对专有API的依赖，确保了研究的**可复现性（reproducibility）**。
- **多语言实体链接级联（Multilingual Linking Cascade）**：设计了一个三阶段实体链接系统，将文本中提到的实体映射到语言无关的 **Wikidata QID** 上，解决了跨语言、形态变化（如波兰语的格变化）和缩写带来的歧义问题。
- **本体约束的关系抽取（Ontology-Constrained Relation Extraction）**：利用一个包含109种实体类型和99种关系类型的 **SKOS本体（ontology）**，通过**引导解码（guided decoding）** 强制模型输出符合预定义类型、符号（正/负/中性）和时间范围（事件/状态/属性）的关系，从而生成结构化的、带符号的时序知识图谱（signed, temporal knowledge graphs）。
- **可定制的研究框架**：本体与管道代码分离，允许其他研究者根据自身需求替换不同的分类体系。

### 相比现有方法的优势
根据文中 **Table 1** 的对比，该方法在多个维度上超越了现有代表性系统：

| 特性 | 本文方法 | 其他典型方法（如 Bro, 2025） |
| :--- | :--- | :--- |
| **Open weights** | ✅ 是 | ❌ 否（依赖GPT-4等API） |
| **Multilingual** | ✅ 是 | ❌ 否（通常为单语言） |
| **KB-linked** | ✅ 是（链接至Wikidata） | ❌ 否 |
| **Fixed ontology** | ✅ 是（强制执行） | ❌ 否 |
| **Signed/temporal** | ✅ 是 | ⚠️ 部分或否 |

## 2. 核心实验方法和设置

### 数据集
- **主语料库**：
  - **奥地利语料**：来自Factiva的499,851篇奥地利新闻文章（2005-2017年）。
  - **波兰语料**：约50万篇波兰语新闻文章（1997-2025年）。
- **评估用黄金标准（Gold Standard）**：
  - 在502篇波兰文章上，使用三个前沿的、独立的LLM家族（GPT-5.4, Gemini 3.1 Pro, Mistral Large 3）进行双轮标注（先实体后关系），并通过一个推理模型（Claude Opus 4.8）进行仲裁，最终形成包含3491个关系的黄金标准。
- **人类验证数据**：一个包含100篇文章的人工标注德语实体黄金标准，用于独立评估NER性能。

### 实验设置与评估指标
- **评估策略**：采用“全覆盖抽查”（full-coverage spot-check）作为主要评估方式，以克服黄金标准不完整的问题。
- **评估指标**：
  - **NER性能**：使用精确率（Precision）、召回率（Recall）和F1分数，在人类标注的德语数据上评估。
  - **关系抽取质量**：报告一个**严格-宽松正确率区间（strict-lenient correctness band）**：
    - **严格正确率（Strict）**：要求关系与黄金标准完全匹配，或虽被黄金标准遗漏但确实有效。
    - **宽松正确率（Lenient）**：只要在合理解读下能捕捉到真实联系即算正确，仅虚构关系为错误。
  - **外部验证**：通过两个独立的案例研究，将抽取结果与公开记录（如政党生命周期、法院判决、政府人事变动）进行对比，验证其真实性。

### 基线方法对比
论文未直接与单一基线模型进行端到端的F1分数对比，而是通过 **Table 1** 对现有LLM-based政治信息抽取系统的**系统性缺陷**进行了全面对比，凸显了本方法在开放性、多语言、知识库链接、固定本体和符号化关系方面的综合优势。

## 3. 主要实验结果和性能指标

### 关键性能数据
- **NER性能**：在人类标注的德语数据上，检测F1达到 **83.8%** （精确率85.5%，召回率82.3%），表明其具有良好的平衡性和高精度。
- **关系抽取正确率**：在100篇测试文章的全覆盖抽查中，取得了以下成绩：
  - **严格正确率**：**68.2%**
  - **宽松正确率**：**93.7%**
  - 这表明绝大多数抽取结果是文本上有据可依的。
- **幻觉率（Hallucination Floor）**：不可减少的幻觉率为 **6.3%**。
- **黄金标准不完整性**：在严格评估下，**27.9%** 的抽取关系是有效的，但被黄金标准遗漏，这说明传统的三元组匹配会严重低估实际性能。

### 与基线方法的对比结果
虽然没有直接的数字对比，但论文明确指出，其设置（零样本、跨语言、99类开放本体）比标准基准（如TACRED）更难。即便如此，其宽松正确率远超TACRED上零样本LLM抽取的约31% micro-F1，证明了其方法的有效性。

### 消融实验结果
论文通过错误分析（Error Taxonomy, Table 4）揭示了性能瓶颈：
- **假阳性（False Positive）**：20.1%为幻觉，8.7%为关系类型错误，3.0%为方向错误。
- **假阴性（False Negative）**：真正漏检的原因中，**57.5%** 归因于LLM未能从已检测到的实体对中提取关系，**39.2%** 归因于NER漏检实体。
- **关键发现**：最大的改进空间在于提升LLM在已有实体对上的召回率，而非单纯改进NER。

## 4. 关键结论和发现

### 主要发现
1. **方法有效性得到双重验证**：
   - **内部评估**：关系抽取的文本正确率高达68.2%-93.7%。
   - **外部验证**：两个案例研究成功重建了独立可验证的历史事实。
     - **奥地利案例**：准确重建了“奥地利未来联盟”（BZO）政党的完整生命周期，包括分裂时间点、人员去向及与腐败案相关的司法判决。
     - **波兰案例**：成功识别出围绕国有企业（SOE）的经济与治理网络重叠，并绘制了高度极化的公民纲领党（PO）与法律与公正党（PiS）之间的**符号化冲突网络（signed conflict network）**，证实了“寻租集团”（O-group）的存在。
2. **符号化网络至关重要**：标准的无符号社区检测无法区分对立阵营，只有引入关系符号（支持/批评）才能揭示真实的两极分化结构。
3. **基础设施优于提示工程**：实验发现，调整采样参数和修复逻辑漏洞比反复优化提示词（prompt engineering）更能显著提升性能。

### 方法的局限性
- **实体链接覆盖率低**：节点级别的QID填充率仅为18.4%-21.9%，尽管高中心性实体的链接率较高，但中层精英的覆盖率仍需提高。
- **黄金标准依赖LLM**：关系评估的黄金标准和抽查均由LLM完成，缺乏最终的人类ground truth，可能存在共享偏见。
- **时间信息有限**：仅有58.4%的关系有显式的开始日期，34.4%有结束日期，其余关系的时间锚点仅为文章发表日期。
- **潜在的误链接**：存在同形异义词（homonym）误链接的风险（如“PO”被链接为钋元素），需要手动排除。

### 未来工作方向
- **提升实体链接覆盖率**：计划整合企业注册数据库（如Orbis）作为链接源，以解决商业实体链接不足的问题。
- **推动跨国家比较**：当前框架已扩展至更多欧洲国家，下一步是计算跨国网络指标（如中心性、密度、网络开放度），以实现真正的比较政治学研究。
- **建立人类标注基准**：未来目标是创建一个人工标注的子集，以更可靠地基准化黄金标准和抽查过程。

</details>

---

### 5. [CHAMB-GA: A Containerized HPC Scalable Microservice-Based Framework for Genetic Algorithms](https://arxiv.org/abs/2606.27217)

**Authors**: Felix Bonhoff, Thiemo Pesch, Andrea Benigni, Alexander Mitsos, Manuel Dahmen  
**Category**: cs.DC  
**Published**: 2026-06-26  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.27217v1  

#### Abstract
Metaheuristic-based global optimization with embedded, long-running simulations is a computationally expensive process. To support various stages of development and execution, a seamless transition from personal computers to distributed clusters is desired, enabling execution across all computationa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CHAMB-GA: A Containerized HPC Scalable Microservice-Based Framework for Genetic Algorithms

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于遗传算法（Genetic Algorithm, GA）的全局优化任务常依赖于计算密集型的嵌入式仿真（如电力系统中的AC潮流计算），其执行过程面临以下挑战：
- **可扩展性差**：传统框架难以在个人电脑、云环境和HPC集群之间无缝迁移。
- **硬件绑定**：多数工具链与特定硬件或调度器（如MPI、multiprocessing）紧密耦合，限制了跨平台部署能力。
- **集成复杂度高**：缺乏对异构软件栈、多阶段工作流（multi-stage workflows）以及外部工具（如不同仿真器）的灵活支持。

### 提出的新方法或新思路
本文提出 **CHAMB-GA** —— 一种**容器化、HPC可扩展的微服务架构框架**，用于支持带有嵌入式仿真的遗传算法。其核心设计思想包括：
- **微服务解耦**：将进化操作（propagation）与适应度评估（fitness evaluation）分离为独立运行的容器组件。
- **消息代理协调**：采用 **RabbitMQ** 作为中央消息队列，实现异步 manager-worker 通信，支持动态负载均衡和弹性伸缩。
- **统一调度接口**：兼容 **Kubernetes**（云原生编排）和 **SLURM**（HPC批处理调度），实现跨环境的一致性部署。
- **双轴扩展机制**：同时支持水平扩展（增加worker数量）和垂直扩展（提升单个worker资源），以适配不同规模的优化任务。

### 相比现有方法的优势
| 类别 | 典型代表 | 局限性 | CHAMB-GA 的优势 |
|------|--------|-------|----------------|
| **Monolithic** | DEAP, Pagmo | 紧耦合，难扩展至分布式环境，不支持异构硬件 | 松耦合，模块化，支持异构基础设施 |
| **Functional/Data-flow** | Spark-GA, FlexGP | 固定数据流模式，难以处理动态演化逻辑 | 支持任意控制流与复杂状态管理 |
| **Microservice-based** | AMQPGA, KafkEO | 不兼容 SLURM，仅适用于云环境 | 同时支持 Kubernetes 和 SLURM，打通科研与生产环境 |

> ✅ **核心创新**：首次实现了**从本地开发到超算中心的端到端无缝迁移**，并原生支持多级嵌套优化（如超参数调优）、岛模型（island model）及多oracle分析。

---

## 2. 核心实验方法和设置

### 使用的数据集与问题场景
- **基准测试**：无实际计算负载，使用 `sleep()` 函数模拟不同长度的适应度评估时间（0.1~10秒）。
- **真实应用案例**：德国高压输电网络（2715个节点，871台发电机，5351条线路），引入18条计划中的 **HVDC**（High Voltage Direct Current）线路进行调度优化。
- **目标函数**：最小化全网线路总功率传输费用：
  $$
  \min F(x) = \sum_{i \in C} P_i(x)
  $$
  其中 $ x $ 是18维HVDC功率设定向量。

### 实验设置
#### 硬件配置（三类层级）
| Tier | #Nodes | Cores per Node | 调度环境 |
|------|--------|----------------|----------|
| 单节点 Kubernetes | 1 | 18 | Kubernetes |
| 多节点 Kubernetes | 3 | 128 | Kubernetes |
| JURECA-DC 超算 | 最多25 | 128 | SLURM |

> 总计最大扩展至 **3200 CPU cores**，远超同类研究（如Khalloof 2023年仅用4节点128核）。

#### 评估指标
- **并行效率（Parallel Efficiency, $ p $）**：
  $$
  p = \frac{S \cdot P \cdot M \cdot N_E}{T \cdot N_w}
  $$
  其中 $ S $ 为个体数，$ P $ 为种群大小，$ M $ 为世代数，$ N_E $ 为纪元数，$ T $ 为实测墙钟时间，$ N_w $ 为并行worker数。
- **收敛速度与最优解质量**：记录每代最佳适应度变化趋势。
- **可移植性验证**：在 Kubernetes 与 SLURM 间迁移任务，观察性能一致性。

#### 基线方法对比
- **无直接替代品**：因CHAMB-GA是首个同时支持Kubernetes和SLURM的开源微服务GA框架。
- **间接比较对象**：
  - **Pagmo**（HPC导向但非微服务）
  - **AMQPGA**（微服务但仅支持云）
  - **Spark-GA**（大数据范式，固定流程）
- 通过引用文献中的扩展极限（如Sherry et al. 2012使用400核）进行横向对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）基准测试：并行效率接近理想线性扩展
- 在 **JURECA-DC** 上使用高达 **3500个worker**（3200核）时：
  - 当适应度评估时间 ≥ 1秒，**并行效率 > 0.95**
  - 即使在最短0.1秒任务下，最低效率仍达 **0.8**
- 扩展曲线显示近乎线性增长，未出现吞吐饱和现象。

> 🔹 图4表明：无论Kubernetes还是SLURM环境，CHAMB-GA均能高效维持大规模并行。

#### （2）HVDC调度优化：应对安全约束的大规模仿真
- 引入 **N−1安全准则**，每个适应度评估需执行 **2004个事故场景** 的AC潮流计算，计算量提升三个数量级。
- 使用两种资源配置策略对比：
  | 配置 | Worker 数量 | 每Worker核心数 | 总核心数 |
  |------|-------------|----------------|---------|
  | (a) 水平优先 | 384 | 8 | 3072 |
  | (b) 垂直优先 | 24 | 128 | 3072 |

- 结果：
  - (a) 水平扩展完成约 **6000万次** AC潮流计算
  - (b) 垂直扩展完成约 **3600万次**
  - 两者均在6小时内完成，证明双轴扩展的有效性。

> 🔹 图5显示：两种策略均可持续改进解的质量，且无明显停滞。

#### （3）分层优化：超参数自动调优（Hyperparameter Tuning）
- 构建两层GA结构：
  - **外层 Meta-GA**：搜索NSGA-II超参数（种群大小、交叉/变异概率等）
  - **内层 Worker-GA**：执行HVDC调度任务，返回适应度
- 使用共享评估池（shared AC powerflow evaluators），实现跨worker GA的负载均衡。
- 运行12小时后，Meta-GA成功收敛部分参数（如交叉概率趋于上限），而种群大小保持较低值，说明**垂直扩展更有效**。

> 🔹 图6揭示：无需手动干预即可发现适合当前硬件与问题结构的最佳配置组合。

---

## 4. 关键结论和发现

### 主要发现
1. **CHAMB-GA 实现了前所未有的跨平台可移植性**：
   - 可在笔记本、私有云、公有云、HPC集群上一键部署，无需修改代码。
2. **微服务+消息队列架构显著降低通信开销**：
   - RabbitMQ 的异步机制避免了同步阻塞，尤其在异构评估耗时场景下大幅提升资源利用率。
3. **双轴扩展策略提升了硬件利用灵活性**：
   - 对小种群高计算成本问题，应优先垂直扩展；对大种群低延迟问题，则宜水平扩展。
4. **支持复杂多级工作流**：
   - 成功实现“GA优化GA”的分层结构，展示了框架在自动化算法设计（AutoML-like）方面的潜力。

### 方法的局限性
- **尚未支持GPU加速**：目前仅针对CPU密集型仿真优化，未整合CUDA或HIP等异构计算能力。
- **语言依赖性**：虽可通过容器封装多语言程序，但主控脚本基于Python，可能影响纯C++/Fortran用户的体验。
- **冷启动开销**：容器构建与部署有一定初始化延迟，在极短任务中占比不可忽略（但在长仿真任务中可忽略）。

### 未来工作方向
1. **集成GPU支持**：将适应度评估容器扩展至支持CUDA/OpenCL，进一步提升计算密度。
2. **异构编程语言互操作性增强**：探索gRPC或REST API方式连接不同语言的服务。
3. **智能自适应扩展策略**：基于实时监控自动调整水平/垂直扩展比例。
4. **与Workflow引擎集成**：对接Nextflow、Snakemake等科学工作流系统，提升易用性。

---

> 📌 **总结一句话**：  
> **CHAMB-GA 是一个轻量级、开放源码、高度可扩展的微服务框架，它打破了传统GA工具在硬件、调度器和架构上的壁垒，为大规模科学优化提供了从实验室到超算中心的“一次编写，处处运行”解决方案。**

</details>

---

### 6. [Hot AI in Cold Space: Thermal-Crosstalk-Aware Scheduling for Sustainable Orbital AI Clusters](https://arxiv.org/abs/2606.26150)

**Authors**: Shuyi Chen, Zhengchang Hua, Nikos Tziritas, Georgios Theodoropoulos  
**Category**: cs.DC  
**Published**: 2026-06-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.26150v1  

#### Abstract
Terrestrial AI training faces an unsustainable energy and water crisis, positioning Orbital Data Centers (ODCs) as a "zero operational carbon" alternative. However, the sub-$10\mu\text{s}$ communication latency required for distributed Large Language Model (LLM) training forces ODCs into extreme phy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Hot AI in Cold Space: Thermal-Crosstalk-Aware Scheduling for Sustainable Orbital AI Clusters*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**轨道数据中心**（Orbital Data Centers, ODCs）在支持分布式大语言模型**（Large Language Model, LLM）训练时面临的“近距-热悖论”**（Proximity-Thermal Paradox）。  
为实现亚10微秒通信延迟，ODC必须采用极高物理密度架构（如单体结构或邻近卫星群），但这导致严重的**热串扰**（Thermal Crosstalk）：
- **Thermal-Fluid Crosstalk**：在集中液冷系统中，下游节点接收被上游预热的冷却液，形成“热陷阱”；
- **Thermal-Radiative Crosstalk**：在自由飞行的卫星群中，核心节点因几何遮挡和相互加热而难以通过深空辐射散热。

这种热堆积会引发：
- **硬件热节流**（Thermal Throttling），降低训练吞吐；
- **热疲劳**（Thermal Fatigue），显著缩短硬件寿命，产生“空间电子垃圾”（space e-waste）；
- 最终导致**Model Flops Utilization**（MFU）下降，无法摊销火箭发射带来的巨大**内含碳**（embodied carbon）。

> ❗传统均匀负载均衡策略加剧了这一问题——它无视空间冷却能力差异，强制所有节点承担相同计算负载，导致冷却条件差的节点成为“拖后腿”的**straggler**。

---

### 🚀 提出的新方法与新思路

#### （1）**Thermal-Aware Heterogeneity Thesis**（热感知异构性假说）
提出一个根本性范式转变：**不应将空间冷却不均视为缺陷，而应将其作为可调度的一等资源**。  
即，在ODC中，**冷却能力的空间异质性**（spatial cooling variance）是决定任务分配的关键维度。

#### （2）**Thermal-Load Balancing**（TLB）框架
一种闭环、软件定义的调度框架，动态迁移LLM工作负载以规避热瓶颈。其核心机制包括：
- **三阶段控制环**：
  1. **Thermal Telemetry**：实时采集硅片温度、冷却液状态、有效视场因子（Effective View Factor）等；
  2. **Capability Profiling**：将热数据转化为每个节点的**热能力评分**（capability score $ w_i $）；
  3. **Asymmetric Workload Slicing**：按评分非均匀分配micro-batch大小。

- 支持两种架构的启发式策略：
  - **Monolithic 结构**：优先上游节点（冷却液最冷），评分公式：  
    $$
    w_i = 1.0 + \alpha \cdot (L_{\text{pipe}} - \text{idx}_i)
    $$
  - **Proximity Swarm**：优先边缘节点（辐射视野最佳），评分公式：  
    $$
    w_i = 1.0 + \alpha \cdot p_i \quad (\text{其中 } p_i \text{ 为有效视场因子})
    $$

- 动态采样器集成于PyTorch DDP或Megatron-LM等AI框架，替代静态DataLoader。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | TLB |
|------|--------|-----|
| 调度目标 | 最大化算力利用率 / 最小化能耗（operational carbon） | 最大化硬件寿命，摊销**embodied carbon** |
| 对待热分布 | 忽略或被动缓解 | 主动利用作调度资源 |
| 架构适应性 | 假设独立散热 | 显式建模multi-scale thermal crosstalk |
| 可持续性视角 | 地面适用（有对流冷却） | 面向太空真空环境设计 |
| 实现开销 | 低（静态分配） | 中等（需动态编译支持，如`torch.compile`） |

> ✅ TLB首次将**热可持续性**（thermal sustainability）与**AI调度**深度融合，提出“用软件延长硬件寿命”的新路径。

---

## 2. 核心实验方法和设置

### 🧪 实验平台：时间步进式热-计算协同仿真器
开发了一个模拟器，联合建模热力学演化与LLM训练过程。

### 📐 架构设置（两类ODC范式）

| 参数 | Monolithic Structure | Proximity Swarm |
|------|------------------------|------------------|
| 节点数 | 64（8条冷却管 × 8节点） | 36（6×6平面网格） |
| 冷却方式 | 集中式液冷（串联流动） | 分布式被动辐射 |
| 热干扰机制 | 上游加热下游冷却液 | 几何遮挡 + 相互辐射 |
| 视场因子模型 | —— | 动态计算 $ p_i \in (0,1] $，边缘高，中心低 |

### 📊 工作负载模型
- 使用**Data Parallelism**进行LLM训练；
- 全局batch被划分为micro-batch；
- 每个节点处理 $ b_i $ tokens；
- 步骤延迟由最慢节点决定：  
  $$
  t_{\text{step}} = \max_i(t_{\text{comp}}(i) + t_{\text{comm}})
  $$

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Node Temperature Distribution** | 各节点芯片温度空间分布 |
| **MFU**（Model Flops Utilization） | 实际达到的FLOPs占峰值比例，反映训练效率 |
| **Tail Latency Spread** | 节点间计算时间差异，衡量straggler影响 |
| **MTTF**（Mean Time To Failure） | 基于Arrhenius方程估算硬件寿命 |
| **Lifespan Extension** | TLB相比Baseline提升的MTTF百分比 |

### ⚖️ 基线方法对比
| 方法 | 描述 |
|------|------|
| **Baseline (Uniform)** | 均匀分配micro-batch，模仿标准PyTorch DDP行为 |
| **TLB (Greedy Heuristic)** | 按热能力评分成比例分配负载（本文提出） |

> 注：未使用复杂优化器（如DRL），仅验证**基本可行性**，故结果为下界估计。

---

## 3. 主要实验结果和性能指标

### 📈 温度分布改善（图2）

#### Monolithic 架构
- **Baseline**：下游节点（Row 7）温度高达 **354.4 K**（81.3°C），接近节流阈值；
- **TLB**：最大温度降至 **353.3 K**（80.2°C），消除明显热梯度。

#### Proximity Swarm
- **Baseline**：核心节点达 **357.2 K**（84.1°C），边缘仅 **344.5 K**（71.4°C）；
- **TLB**：核心降温至 **353.9 K**（80.8°C），边缘升温至 **351.9 K**（78.8°C），**热梯度大幅平坦化**。

✅ 表明TLB成功将计算负载从“热陷阱”转移至“冷区”。

---

### ⏱️ MFU 与尾部延迟（图3）

| 架构 | 方法 | MFU | 尾部延迟压缩效果 |
|------|------|-----|----------------|
| Monolithic | Baseline | 75.1% | 存在明显阶梯状延迟 |
| Monolithic | TLB | **82.7%** (+7.6%) | 尾部显著压缩 |
| Swarm | Baseline | 90.0% | 边缘节点快，核心慢 |
| Swarm | TLB | **90.2%** | **完全扁平化延迟分布**，无同步等待 |

> 💡 即使绝对增益不大，但在GW级集群中，**消除straggler意味着避免全局停滞**，具有重大工程意义。

---

### 🔋 硬件寿命延长（基于Arrhenius模型）

| 节点类型 | 寿命提升（vs. Baseline） |
|----------|-------------------------|
| Swarm 核心节点 | **+6.15%** MTTF |
| Monolithic 下游出口节点 | **+1.71%** MTTF |

📌 关键洞察：**小幅降低峰值温度即可带来指数级寿命增长**（因MTTF ∝ exp(-1/T)）。

---

### 🔍 消融分析（隐含于讨论中）
虽然没有显式消融实验，但文中指出：
- 当前启发式仅为**greedy approximation**，远非最优；
- 忽略了多跳光路由延迟、瞬态流体滞后等次要瓶颈；
- 因此当前性能增益是**保守估计**，未来仍有巨大优化空间。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Proximity-Thermal Paradox真实存在且严重**：为满足LLM训练的极低延迟需求，ODC的高密度必然引发热串扰，造成热瓶颈与硬件早衰。
2. **传统均匀负载均衡在ODC中不可持续**：它放大了冷却不均的影响，制造straggler并加速热疲劳。
3. **Thermal-Aware Heterogeneity 是突破口**：将冷却能力的空间差异作为调度资源，可主动规避热陷阱。
4. **TLB能同时提升MFU与硬件寿命**：
   - 提升Monolithic MFU达 **82.7%**（+7.6%）；
   - 在Swarm中实现**完美同步**（flat latency）；
   - 延长最热节点寿命最高达 **6.15%**。
5. **延长硬件寿命 = 摊销embodied carbon的关键**：在零运行碳排的太空环境中，**可持续性的核心是硬件耐久性**。

---

### ⚠️ 局限性
| 局限 | 说明 |
|------|------|
| 启发式策略较简单 | 使用比例分配而非全局优化，性能非最优 |
| 忽略网络动态影响 | 未联合优化FSO链路重配置带来的热变化 |
| 假设理想传感与响应 | 实际中BMC采样频率、控制延迟可能影响效果 |
| 缺乏真实部署验证 | 当前为仿真研究，尚未在轨测试 |
| 内存预分配挑战 | 动态batch size需依赖新兴编译技术（如`torch.compile`） |

---

### 🔮 未来工作方向（作者明确提出）
1. **Dynamic Network-Thermal Co-design**  
   联合优化自由空间光通信**（FSO）拓扑重构**与热管理——机械转向改变朝向，进而影响辐射散热，需协同调度。

2. **E-Waste vs. Carbon Trade-off Analysis**  
   开展全生命周期评估**（Life Cycle Assessment, LCA）**，量化“一次性高端AI加速器”是否真能抵消发射碳成本？是否需发展**在轨回收**技术？

3. **Federated Thermal Telemetry Standards**  
   建立跨厂商的“热余量”广播协议，在保护商业机密前提下实现多租户ODC的协作热调度。

---

## 总结

> 🔭 本论文提出了一个面向未来的洞见：**轨道AI的可持续性不在于节能，而在于延寿**。  
>  
> 通过引入 **Thermal-Aware Heterogeneity Thesis** 和 **TLB框架**，首次将热力学特性作为空间计算资源进行软件调度，不仅恢复了MFU，更从根本上延长了硬件生存周期，为摊销火箭发射的巨大**embodied carbon**提供了可行路径。  
>
> 这是一次从“地面思维”到“太空原生设计”的范式跃迁，标志着可持续AI进入**多物理场协同优化**的新时代。

</details>

---

### 7. [Optimizing CUDA like a Human: Micro-Profiling Tools as Expert Surrogates for LLM-Based GPU Kernel Optimization](https://arxiv.org/abs/2606.26453)

**Authors**: Jiading Gai, Shuai Zhang, Kaj Bostrom, Jin Huang, Vihang Patil, Haoyang Fang, Bernie Wang, Huzefa Rangwala, George Karypis  
**Category**: cs.LG  
**Published**: 2026-06-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.26453v1  

#### Abstract
We present KernelPro, a closed-loop multi-agent system that automatically generates, profiles, and iteratively optimizes GPU kernel code by integrating large language model (LLM) code generation with hardware profiler feedback and pluggable bottleneck detection tools. KernelPro introduces four contr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
该论文旨在解决当前基于 **Large Language Model (LLM)** 的 GPU 内核优化系统中存在的一个根本性缺陷：这些系统通常将原始硬件性能分析数据（如 `ncu` 指标）直接输入给 LLM，并期望模型能够隐式地模拟专家工程师的推理过程。这混淆了两个不同的任务：
- **解释硬件遥测数据**（结构化、规则驱动）
- **生成优化代码**（创造性、上下文依赖）

这种做法导致优化质量不稳定、诊断不一致，并且难以融入新的分析模式。

### 提出的新方法与思路
作者提出了 **KERNELPRO**，一个闭环多智能体系统，其核心创新在于将专家经验显式编码为可插拔的 **micro-profiling tools**，作为 LLM 的“专家代理”（expert surrogates）。其主要贡献如下：

1. **语义反馈算子 (Semantic Feedback Operator)**  
   将专家启发式规则形式化为可执行的微分析工具。这些工具接收原始硬件指标，通过触发-分析-推荐（trigger-analyze-recommend）流程，将其转化为**可操作的自然语言指导**，再提供给 LLM，而非原始数字。

2. **两阶段工具调用架构 (Two-Stage Tool Invocation Architecture)**  
   - **第一阶段**：通过 **roofline 分析**对内核瓶颈类型进行分类（计算密集型、内存密集型等）。
   - **第二阶段**：仅激活与该瓶颈类型相关的专用分析工具，结合 `ncu`（内核级）、`SASS`（指令级）和 `nsys`（系统级）的分析，减少噪声并聚焦于相关优化。

3. **领域自适应的蒙特卡洛树搜索 (Domain-Adapted MCTS)**  
   引入一种改进的 MCTS 策略用于代码搜索，包含：
   - **渐进扩展 (Progressive Widening)**：控制节点扩展速率，避免昂贵的编译/分析开销。
   - **非对称分支因子 (Asymmetric Branching)**：成功节点获得更多子节点，反映优化空间的不对称性。
   - **对数奖励校准 (Log-Reward Calibration)**：防止高加速比异常值主导搜索。
   - **死胡同剪枝 (Dead-End Pruning)** 和 **搜索记忆 (Search Memory)**：提升搜索效率和跨迭代学习能力。

4. **直接 CuTe 源码生成 (Direct CuTe Source-Level Code Generation)**  
   通过自主在 **CUTLASS/CuTe** 代码库中进行 LLM 驱动的代码搜索（如 `search_cutlass`, `read_cutlass_file`），模仿专家工程师从零开始编写高性能 CUDA+CuTe 内核的方式，实现真正的源码级优化，而非简单的模板实例化。

5. **首个能效感知的 CUDA 内核编码代理 (Energy-Aware CUDA Kernel Coding Agent)**  
   在速度优先的前提下，引入**字典序能效奖励**，首次实现了 LLM 驱动的能效优化，而不仅仅是速度优化。

### 相比现有方法的优势
- **更可靠**：通过显式的工具链保证了全面、一致的瓶颈分析。
- **更高效**：MCTS 避免了贪心搜索的局部最优陷阱。
- **更深入**：能生成底层 CuTe 源码，触及更高性能上限。
- **更全面**：同时优化速度和能效。

---

## 2. 核心实验方法和设置

### 数据集
- **KernelBench (Ouyang et al., 2025)**：包含 250 个 GPU 内核任务，分为三个难度级别：
  - **Level 1**: 100 个单算子任务
  - **Level 2**: 100 个融合/组合算子任务
  - **Level 3**: 50 个完整模型架构
- **VeOmni MoE 生产内核**：来自生产环境的专家手工优化的 MoE 训练内核（`dW1 weight-gradient kernel`），用于验证在真实场景下的有效性。

### 实验设置和评估指标
- **模型**：Claude Sonnet 4.6
- **硬件**：NVIDIA A100 (AWS p4 instances)
- **配置**：15 个种子（5 个温度 × 3 轮），30 次迭代，每次迭代 2 个候选解，使用 MCTS 搜索。
- **主要指标**：
  - **几何平均加速比 (Geometric Mean Speedup)**：相对于 PyTorch Eager 基线的运行时间加速比。
  - **正确性**：输出满足 `torch.allclose(out, ref, atol=1e-2, rtol=1e-3)`。
- **消融实验**：在 KernelBench 的 42 个代表性任务子集上进行，以控制变量评估各组件贡献。

### 基线方法对比
- **KernelBlaster (Dong et al., 2026)**：当时最先进的方法，使用带持久知识库的上下文强化学习。
- **其他对比系统**：CudaForge, StitchCUDA, KernelFoundry, AVO 等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **KernelBench** 上，KERNELPRO 取得了当前最佳（SOTA）性能：

| System       | Level 1 | Level 2 | Level 3 | Solved        |
|--------------|---------|---------|---------|---------------|
| **KERNELPRO** | **2.42×** | **4.69×** | **5.30×** | 100/100 / 50 |
| KernelBlaster | 1.43×   | 2.50×   | 1.50×   | -             |

- **提升幅度**：分别提升了 **+69%**, **+88%**, **+253%**。
- **任务覆盖率**：在所有级别均达到 100% 成功求解。

### 与基线方法的对比结果
- **在 VeOmni MoE 内核上**：KERNELPRO 生成了一个从头开始的 raw-CUDA+CuTe Hopper WGMMA 内核，相比专家手工优化的 Triton 基线，实现了 **1.23×** 的加速。
- **鲁棒性验证**：即使对加速比进行上限截断（capping），KERNELPRO 的优势依然显著，证明其性能提升并非由少数极端案例驱动。

### 消融实验结果
消融研究证实了每个设计组件都独立且显著地提升了优化质量：

| 组件                     | 处理组 vs 对照组           | 几何平均加速比 | 提升幅度 | p-value     |
|--------------------------|----------------------------|----------------|----------|-------------|
| **Micro-profiling tools** | 完整流程 vs 原始 ncu 指标 | 4.00× vs 1.77× | +125%    | **<0.0001** |
| **MCTS 搜索**            | MCTS vs 贪心搜索           | 4.60× vs 3.65× | +26%     | **0.004**   |
| **主动工具编排**          | 确定性 vs 反应式调用       | 1.37× vs 1.12× | +23%     | **0.035**   |
| **搜索记忆**              | 开启 vs 关闭               | 1.82× vs 1.72× | +6%      | 0.181 (n.s.)|

**关键发现**：
- **原始指标有害**：提供未经解释的原始 `ncu` 指标 (`Raw ncu Metrics`) 的效果甚至**显著差于不提供任何反馈** (`No Feedback`)，证明了显式解释的必要性。
- **MCTS 至关重要**：在需要多步优化链的复杂任务上，MCTS 能够逃离贪心搜索的局部最优。
- **主动编排更优**：确定性地执行所有相关工具，确保了全面覆盖，优于 LLM 随机选择工具的反应式模式。

---

## 4. 关键结论和发现

### 主要发现
1. **瓶颈在于反馈，而非生成**：LLM 本身具备强大的代码生成能力，限制性能的关键是**如何将硬件分析数据呈现给 LLM**。将原始指标转换为**语义化的、可操作的指导**是提升优化质量的核心。
2. **显式优于隐式**：与其让 LLM 隐式学习专家推理，不如将专家经验**显式编码为可执行的工具**，这种方法更可靠、可解释且易于扩展。
3. **系统性探索至关重要**：MCTS 等结构化搜索策略对于解决复杂的多步优化问题是必要的，它能有效探索广阔的优化空间。
4. **能效可以被优化**：首次证明了 LLM 驱动的内核优化可以超越单纯的速度目标，在保持相同速度的情况下，通过选择更低能耗的指令序列（如使用 `__fdividef` 而非 IEEE 除法），实现了 **11.6%** 的实测能耗降低。

### 方法的局限性
- **依赖高质量工具**：系统的性能高度依赖于所实现的 micro-profiling tools 的质量和完备性。
- **计算成本高**：MCTS 和多次编译/分析循环带来了较高的计算开销。
- **领域特定**：虽然框架通用，但具体的工具和提示工程需要针对 CUDA/GPU 架构进行深度定制。

### 未来工作方向
- **系统化表征固定速度下的能效优化**：对能耗优化进行更全面的研究。
- **扩展到更多 GPU 架构和编程模型**：如 AMD ROCm 或 Intel oneAPI。
- **构建开放的工具生态系统**：允许社区贡献和共享新的 micro-profiling tools。
- **探索更高效的搜索策略**：在保证性能的同时降低计算成本。

</details>

---

### 8. [EGG: An Expert-Guided Agent Framework for Kernel Generation](https://arxiv.org/abs/2606.26758)

**Authors**: Yaochen Han, Ke Fan, Hongxu Jiang, Wanqi Xu, Weiyu Xie, Runhua Zhang, Chenhui Zhu, Yixiang Zhang  
**Category**: cs.AI  
**Published**: 2026-06-26  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.26758v1  

#### Abstract
High-performance GPU kernels are critical for reducing the exponentially growing computational costs of large language models (LLMs), but their development heavily relies on manual tuning by domain experts. While recent advances in LLM-based approaches show promise for automating kernel generation, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# EGG: An Expert-Guided Agent Framework for Kernel Generation —— 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **高性能 GPU Kernel 开发依赖专家手动调优**：传统上，为深度学习模型（尤其是 LLM）生成高性能 GPU kernels 高度依赖领域专家的经验，开发成本高、周期长。
- **现有 LLM 自动化方法效果有限**：尽管 LLM 在通用代码生成方面表现出色，但在 GPU kernel 生成任务中，由于设计空间巨大且硬件约束严格，直接应用 LLM 往往导致生成的 kernel 编译失败或性能低下。
- **缺乏有效的优化引导机制**：当前基于 Agent 或 Reinforcement Learning 的方法多采用试错式探索（trial-and-error），缺乏对优化路径的结构性指导，难以稳定地达到高性能。

### 提出的新方法与思路
论文提出 **EGG**（**Expert-Guided Agent Framework for Kernel Generation**），一种融合专家优化经验的多智能体框架，用于自动化生成高性能 Triton kernels。其核心创新在于：

#### ✅ **专家引导的分阶段优化策略（Expert-Guided Staged Optimization）**
将 kernel 生成分解为两个层次分明的阶段：
1. **Algorithmic Structure Design**  
   - 多种子搜索（multi-seed search）探索不同算法范式（如 im2col vs. direct conv）。
   - 算法精炼（algorithmic refinement）进行语义保持的结构优化（如 operator fusion、online softmax）。
   - 目标是建立高质量的计算结构基础，决定性能上限。

2. **Hardware-Specific Tuning**  
   在固定算法结构的基础上，依次执行三个子阶段：
   - **Parallel Mapping**：确定如何将计算映射到 GPU grid。
   - **Tensor Tiling**：选择合适的 BLOCK_M/N/K 尺寸以最大化片上数据复用。
   - **Memory Optimization**：优化内存访问模式和流水线（如 num_warps, num_stages）。

> 这种分层分解显式定义了各阶段的优化目标，有效缩小搜索空间，并引导 LLM 做出更合理的决策。

#### ✅ **阶段感知的多智能体协作机制（Stage-Aware Multi-Agent Collaboration）**
引入三个专业化 Agent 构成闭环系统：
- **Code Agent**：负责生成或修改 kernel 代码。
- **Profile Agent**：分析 Nsight Compute (NCU) 性能指标，识别瓶颈并提出改进建议。
- **Debug Agent**：诊断编译错误、运行时异常或数值不一致问题。

通过 **结构化上下文管理** 实现稳定优化轨迹：
- **跨阶段传播（Inter-stage）**：仅保留最终决策，丢弃中间产物，避免历史干扰。
- **阶段内交互（Intra-stage）**：使用 JSON 格式的结构化反馈（如 `bottleneck`, `modification_plan`），确保信息高效传递。

---

### 相比现有方法的优势
| 维度 | 现有方法（如 CudaForge, AutoTriton） | EGG |
|------|-------------------------------|-----|
| **优化引导** | 缺乏专家先验，依赖粗粒度反馈（如运行时间） | 显式建模专家工作流，提供阶段性优化目标 |
| **探索效率** | 试错式搜索，收敛慢，易陷入局部最优 | 分阶段约束搜索空间，提升搜索效率 |
| **稳定性** | 上下文累积导致目标漂移（objective drift） | 结构化上下文管理，保障优化轨迹稳定 |
| **性能上限** | 受限于初始实现质量 | 多种子 + 算法精炼打开更高天花板 |

---

## 2. 核心实验方法和设置

### 数据集
- **KernelBench** (Ouyang et al., 2025)：主要评测基准，包含 250 个 kernel 任务，分为三级难度：
  - **Level 1 (Basic)**：基础算子（如 matmul, ReLU）
  - **Level 2 (Medium)**：融合算子（如 Conv+BN+ReLU）
  - **Level 3 (Hard)**：完整网络结构（如 ResNet, ViT）
- **TritonBench**：来自 GitHub 的真实生产级 Triton kernels（如 FlashAttention, RoPE），验证实际部署价值。

### 实验设置
- **硬件平台**：NVIDIA RTX 4090（主平台），并在 RTX 5090、H20、RTX PRO 6000 上验证泛化性。
- **软件环境**：
  - CUDA 13.0, PyTorch 2.9.1, Triton 3.5.1
  - LLM 后端：GPT-5.1（主）、Claude Opus 4.5（消融验证）
- **交互预算**：每个任务约 50k output tokens（显著低于 CudaForge 的 110k）

### 评估指标
遵循 KernelBench 协议，采用三项标准指标：
1. **Success Rate (%)**：成功编译并通过正确性验证的任务比例。
2. **Fast1 Rate (%)**：在正确 kernel 中，性能优于 PyTorch Eager 的比例。
3. **Speedup (×)**：相对于 PyTorch Eager 的平均加速比（仅在正确 kernel 上计算）。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **默认执行** | PyTorch Eager |
| **编译器优化** | Torch Compile（default & max-autotune） |
| **通用 LLM** | ChatGPT-5.1, DeepSeek-V3.2 |
| **RL-based** | AutoTriton（强化学习微调） |
| **Agent-based** | CudaForge（多轮反馈迭代） |
| **其他编译器** | TVM Relax（补充对比） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（KernelBench）

| Method | Level 1 Speedup | Level 2 Speedup | Level 3 Speedup | Avg Speedup | Success Rate |
|--------|------------------|------------------|------------------|-------------|---------------|
| PyTorch Eager | 1.00× | 1.00× | 1.00× | 1.00× | 100% |
| Torch Compile | 1.09× | 1.38× | 1.36× | 1.28× | 100% |
| AutoTriton | 1.20× | 0.96× | 0.83× | ~1.00× | 36–56% |
| CudaForge | 1.43× | 2.00× | 1.30× | 1.58× | 100% |
| **Ours (EGG)** | **1.83×** | **2.73×** | **1.52×** | **2.13×** | **100%** |

> ✅ **EGG 平均提速 2.13× 超越 PyTorch，且在所有任务上均成功生成正确 kernel。**

#### 亮点表现：
- 在 **Level 2 融合算子** 上，EGG 达到 **100% Fast1 Rate**，即每一个生成的 kernel 都快于 PyTorch。
- 成本更低：相比 CudaForge，EGG 仅消耗 **~50k tokens/kernell**（vs. 110k），效率提升超 2 倍。

---

### 消融实验结果（Ablation Study）

| 配置 | Fast1 Rate | Speedup |
|------|-----------|---------|
| 完整 EGG | 87.6% | 2.13× |
| 移除 Multi-Seed Search | 78.4% | 1.84× |
| 移除 Algorithmic Refinement | 72.0% | 1.46× |
| 移除 Hardware-Specific Tuning | 60.8% | 1.52× |
| 单一 Code Agent（无协作） | ~50% | ~1.3× |

> 🔍 发现：
- **Multi-Seed Search** 对复杂任务（Level 2/3）贡献最大，带来额外 **1.7–2.0×** 加速。
- **Algorithmic Refinement** 是性能上限的关键，尤其在融合算子中作用显著（如 FlashAttention 式融合）。
- **Multi-Agent 协作** 显著提高成功率与稳定性，避免退化。

---

### 实际应用场景验证（TritonBench）

| Task | Hand-written Triton | EGG Generated | Speedup |
|------|----------------------|----------------|---------|
| Flash Attention | 0.046ms | 0.037ms | **1.24×** |
| RoPE Embedding | 0.072ms | 0.044ms | **1.63×** |
| INT8 Dequant MatMul | 0.067ms | 0.062ms | **1.08×** |

> 💡 表明：即使专家手工调优的生产 kernel，EGG 仍能进一步优化，证明其**实用部署潜力**。

---

## 4. 关键结论和发现

### 主要发现
1. **专家知识可被形式化并用于引导 LLM**：通过将专家优化流程拆解为阶段性目标，可以显著提升 LLM 在复杂硬件编程任务中的有效性。
2. **分阶段优化优于端到端搜索**：将庞大的优化空间分解为多个受限子问题，不仅提高了搜索效率，也增强了优化过程的可控性和稳定性。
3. **多智能体协作 + 结构化上下文 = 稳定优化轨迹**：避免了传统单 Agent 的“目标漂移”问题，实现了跨阶段的累积改进。
4. **EGG 可超越人类专家水平**：在多个真实 Triton kernels 上实现了比手写版本更高的性能，说明 LLM + 专家引导的组合具备“超专家”潜力。

---

### 局限性（Limitations）
1. **依赖预设的专家先验（Expert Priors）**  
   - 当前框架依赖人工设计的优化规则（如 tiling 规则、fusion 模式），可能限制非常规创新方案的发现。
2. **缺乏跨阶段联合优化能力**  
   - 各阶段顺序执行，局部最优未必全局最优（例如某 tiling 下的最佳 parallel mapping 不一定适用于另一 tiling）。
3. **Token 消耗仍较高**  
   - 尽管优于 CudaForge，但每 kernel 50k tokens 的开销仍限制大规模部署。

---

### 未来工作方向
- **自适应学习专家先验**：让框架从历史优化轨迹中自动归纳新的优化模式。
- **Top-k 候选传播机制**：在 token 预算允许时保留多个候选 kernel，支持有限的跨阶段联合探索。
- **轻量化代理模型替代 LLM**：训练小型 policy model 替代大模型进行高频调优，降低推理成本。
- **扩展至其他 DSL 和硬件架构**：如应用于 CUDA、Metal 或 NPU 平台。

---

> 📌 **总结一句话**：  
> **EGG 通过“专家引导 + 分阶段优化 + 多智能体协作”的设计，在无需额外训练的前提下，实现了比现有 RL 和 Agent 方法更高效、更稳定、更高性能的 GPU kernel 自动生成，平均提速 2.13× 于 PyTorch，且达到 100% 正确率。**

</details>

---

### 9. [RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning](https://arxiv.org/abs/2606.26997)

**Authors**: Rongjian Chen, Jianmin Hu, Kejiang Ye, Minxian Xu  
**Category**: cs.DC  
**Published**: 2026-06-26  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.26997v1  

#### Abstract
Large language model (LLM) post-training for reasoning increasingly relies on reinforcement learning with verifiable rewards (RLVR), where models learn from ground-truth feedback on mathematical, logical, and scientific tasks. To enable flexible resource allocation and support heterogeneous training...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在**disaggregated 架构下的 on-policy LLM 强化学习**（如 GRPO）中，现有的同步训练系统（如 Slime）采用“先完成全部 rollout 再开始训练”的串行模式，导致 **trainer GPU 在 rollout 阶段长时间空闲**。尽管部分 prompt 组已可训练，但由于必须等待整个 rollout 完成，造成严重的资源浪费。

此外，异步流水线虽能重叠阶段，但会引入 **stale data**，破坏 on-policy 的正确性。

### 提出了什么新方法或新思路
本文提出 **RolloutPipe**，一个用于 disaggregated RLVR（Reinforcement Learning with Verifiable Rewards）系统的 post-training 框架，通过以下两个核心技术实现训练与 rollout 的重叠，同时保持 on-policy 正确性：

- **Complete-Group Pipelining (CGP)**  
  将每个可训练的完整组（complete group）在 materialization 完成后立即送入 trainer 的 FIFO 队列，无需等待整个 rollout 结束，从而提前启动训练。

- **Frontier-Group Dispatch (FGD)**  
  在 Rollout 节点设计一种 admission 控制策略，优先调度即将构成下一个训练 batch 的“前沿组”（frontier groups），使其更早、更稳定地到达 trainer，减少 trainer 等待下一批数据的时间。

### 相比现有方法的优势
- ✅ **保持 on-policy 正确性**：所有组共享相同的 rollout 权重，不使用过时权重。
- ✅ **显著缩短端到端时间**：相比 Slime 缩短 30.7%–42.3%。
- ✅ **大幅降低 trainer 等待比例**：从 47%–52% 下降至 14%–33%。
- ✅ **兼容现有系统架构**：基于 Slime、Megatron-LM、SGLang 和 Ray 实现，易于集成。

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个推理与科学类 benchmark 上进行评估：
- **LSAT-AR**：来自 AGIEval 和 AR-LSAT 的逻辑推理题
- **Sci-XW**：SciBench 中的大学级别数学、化学、物理题
- **Sci-JL**：SciBench 的另一个子集，问题分布不同
- **OlyPhys**：国际及中国奥赛级别的数理难题

### 实验设置
- **模型**：Qwen3-1.7B（具备基础推理能力的最小 Qwen3 模型）
- **训练节点**：8×RTX 4090（TP=4, DP=2）
- **Rollout 节点**：2×A100 40GB（TP=2）
- **GRPO 参数**：
  - $ K = 8 $（每 prompt 采样 8 个响应）
  - $ B = 16 $（全局 batch size）
  - $ U = B/K = 2 $（每次逻辑更新消耗 2 个完整组）
- **Rollout 组数 $ R $**：测试 32、64、96 三种配置
- **每轮重复 4 次取均值**

### 评估指标
- **rollout-to-train-end time**：从 rollout 开始到训练结束的总耗时（主指标）
- **dispatch timing**：首个 U-group 批次被派发至 trainer 的时间
- **trainer waiting ratio**：$ \text{wait} / (\text{wait} + \text{compute}) $
- **response length & trainer compute time**：验证是否因减少计算而提速

### 基线方法对比
- **Slime**：当前最先进的 rollout-training 系统，采用串行执行
- **CGP**：仅启用 complete-group 流水线，保留默认 FIFO admission
- **CGP+FGD**：完整 RolloutPipe 方法，结合 CGP 与 FGD

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **rollout-to-train-end 时间缩短** | **30.7% – 42.3%**（vs. Slime） |
| **trainer waiting ratio 降低** | **37% – 76%**（从 47%–52% → 14%–33%） |
| **首个 dispatch 时间提前** | Slime 平均 >500s → CGP+FGD 最快 **52–61s** |

### 与基线方法的对比结果
- 在所有 12 种设置（4 个 workload × 3 个 R）下，**CGP+FGD 均显著优于 Slime**
- 随着 $ R $ 增大（rollout 规模扩大），加速效果更明显：
  - $ R=32 $：平均提速 ~32%
  - $ R=96 $：平均提速 ~41%

### 消融实验结果
- **CGP 是主要贡献者**：占总加速效果的 **71%–96%**
  - 因为它打破了 rollout 完成才能训练的壁垒
- **FGD 进一步优化调度稳定性**：在 CGP 基础上再提速 **2.5%–11.4%**
  - 使 group 到达更集中，减少 mid-round 等待
- **CGP+FGD 的 dispatch 时间方差更小**，说明组到达更平稳

> 示例：在 Sci-XW, $ R=96 $ 下，CGP 总耗时 659s，CGP+FGD 降至 642s，差异来自更稳定的 group 供给。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **串行 rollout-then-train 模式存在巨大空转开销**：早期完成的可训练组被阻塞，trainer 空等近半时间。
2. **以 complete-group 为粒度进行流水线是可行且高效的**：只要保证 rollout 权重固定，即可安全提前训练。
3. **调度策略对 pipeline 效率至关重要**：FGD 通过优先服务“前沿组”，提升了 trainer 输入流的稳定性。
4. **性能提升来源于时间重叠而非计算削减**：
   - response length 和 trainer compute time 几乎不变
   - 加速完全来自将原本 idle 的时间转化为有效训练时间

### 方法的局限性
- 当前设计依赖于 **group materialization 完成后才可训练**，无法进一步细粒度到单个样本。
- FGD 的性能增益受限于 **GPU 内存和并发容量**（由 $ F_w $ 控制），过大可能导致内存溢出。
- 目前仅验证于 1.7B 规模模型，超大规模模型（如 70B+）的通信与调度开销需进一步研究。

### 未来工作方向
- 将 RolloutPipe 的思想推广至其他 **cognitive computing 系统**，如多模态 RAG、agent 系统中的感知-推理-更新循环。
- 探索动态调整 $ F_w $ 和 $ U $ 的自适应机制，以应对负载波动。
- 结合异构硬件（如 CPU-offloading）进一步优化 disaggregated 架构下的资源利用率。

--- 

> 💡 **一句话总结**：RolloutPipe 通过 **CGP + FGD** 实现了 **on-policy 正确性下的 rollout 与 training 流水线化**，在不牺牲训练质量的前提下，将端到端训练时间缩短超过 30%，为高效 LLM post-training 提供了新的系统范式。

</details>

---

### 10. [TOPS: First-Principles Visual Token Pruning via Constructing Token Optimal Preservation Sets for Efficient MLLM Inference](https://arxiv.org/abs/2606.27161)

**Authors**: Tinghao Wang, Yichen Guo, Rui Huang, Zheng Lu, Qizhe Zhang, Chenxi Li, Yuan Zhang, Jiajun Cao, Zhirong Shen, Yaosong Du, Guangyan Gan, Wenya Wang, Lin William Cong, Shanghang Zhang  
**Category**: cs.AI  
**Published**: 2026-06-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.27161v1  

#### Abstract
Multimodal large language models (MLLMs) have achieved strong multimodal reasoning capabilities, but their efficiency is limited by the large number of visual tokens, which introduces substantial computational overhead. Visual token pruning offers a natural solution, yet existing methods are imperfe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《TOPS: First-Principles Visual Token Pruning via Constructing Token Optimal Preservation Sets for Efficient MLLM Inference》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
多模态大语言模型（**MLLMs**）在处理高分辨率图像或视频时，会产生大量的 **visual tokens**，导致计算开销呈平方级增长（由于 self-attention 机制），严重影响推理效率。现有的视觉 token 剪枝方法大多基于启发式评分（如注意力权重、多样性等），缺乏对“最优保留子集”本质目标的理论建模，容易引入冗余或丢失关键信息。

### 提出的新方法与新思路
论文提出 **TOPS**（Token Optimal Preservation Sets），从第一性原理出发，将视觉 token 剪枝形式化为一个信息论驱动的最优子集选择问题。其核心思想是：**保留的 token 子集 $S$ 应最大化与完整视觉输入 $V$ 和文本查询 $Q$ 的互信息 $I(S; V, Q)$**。

通过链式法则分解该目标，作者推导出三个根本原则：
- **Task Relevance**（任务相关性）：$I(S; Q)$，确保 token 与用户提问相关。
- **Information Coverage**（信息覆盖度）：$I(S; V|Q)$，确保保留足够的原始视觉信息。
- **Semantic Diversity**（语义多样性）：由条件熵约束自然导出，避免保留高度相似的冗余 token。

基于此，TOPS 设计了一个训练免费（training-free）、模型无关（model-agnostic）的剪枝模块，在每个剪枝点贪婪地选择能联合优化上述三项得分的 token。

### 相比现有方法的优势
- **理论基础坚实**：首次从信息论角度系统推导出 token 剪枝的根本原则，而非依赖经验性启发。
- **综合性能优越**：融合三大原则，克服了单一标准（如仅注意力或仅多样性）的局限性。
- **即插即用**：无需微调，可无缝集成到多种 MLLM 架构中。
- **支持多阶段剪枝**：实现粗粒度到细粒度的渐进式压缩，提升鲁棒性。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了广泛的多模态基准，分为以下几类：

#### 图像理解
- **通用 VQA**：GQA, ScienceQA-IMG, TextVQA
- **幻觉检测**：POPE（Polling-based Object Probing Evaluation）
- **综合能力评测**：MME, MMBench-EN/CN, MM-Vet, MMStar, AI2D, HallusionBench

#### 视频理解
- **长视频理解**：MLVU, LongVideoBench
- **视频多模态评测**：Video-MME

### 实验设置和评估指标
- **模型架构**：在 7 种主流 MLLM 上验证，包括：
  - LLaVA-1.5 / LLaVA-NeXT（不同尺寸）
  - LLaVA-Video（视频专用）
  - Qwen2.5-VL-7B-Instruct
  - InternVL3-8B
- **压缩率**：测试多个 token 预算（如保留 77.8%、88.9%、94.4% 的 token 被移除）。
- **评估指标**：
  - **相对准确率（Rel. %）**：剪枝后模型准确率相对于未剪枝基线的百分比。
  - **绝对准确率（Acc.）**
  - **计算效率**：FLOPs、CUDA 延迟（Latency）、GPU 内存占用（Memory）
  - **Logit Fidelity**：衡量剪枝前后模型输出 logits 的差异。

### 基线方法对比
与 10 种前沿剪枝方法进行比较：
- **Attention-based**：FastV, PyramidDrop, SparseVLM
- **Diversity-based**：DivPrune, DART
- **Coverage-based**：SCOPE, VisionZip
- **混合策略**：CDPruner, PruMerge+, TRIM

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **LLaVA-NeXT-7B** 上，移除 **77.8%** 的 visual tokens 后，TOPS 实现了 **100.0%** 的原始性能保持率（甚至略有提升）；在 **LLaVA-NeXT-13B** 上达到 **100.6%**。
- 在 **LLaVA-1.5-7B** 上，移除近 90% tokens 后仍保持 **97.1%** 性能。
- 在极端压缩下（如每帧仅保留 16 tokens 的视频任务），TOPS 仍能维持 **92.3%** 的性能，显著优于 FastV（78.6%）。

### 与基线方法的对比结果
- **全面超越所有基线**：在所有测试模型和压缩比例下，TOPS 均取得 SOTA 表现。
- **以 LLaVA-NeXT-7B 为例（77.8% 压缩）**：
  - TOPS: **100.0%**
  - SCOPE（次优）: 99.8%
  - VisionZip: 99.4%
  - FastV: 95.0%
- **在 Qwen2.5-VL 和 InternVL3 上同样领先**，尤其在 ~90% 压缩率下优势更明显（分别高出 3.3% 和 5.7%）。

### 消融实验结果
- **三原则缺一不可**（Table 5）：
  - 仅用 Relevance：91.1%
  - 仅用 Diversity：92.4%
  - 仅用 Coverage：91.9%
  - **TOPS（三者结合）**：**94.6%**
- **多阶段设计有效**（Table 6）：
  - 仅 Stage I 或 Stage II：~93.5%
  - 完整两阶段 TOPS：**94.6%**
- **超参数鲁棒性强**（Figure 4 & 6）：
  - 平衡系数 $\alpha$（多样性）和 $\lambda$（覆盖度）在 [0.5, 1] 区间内性能稳定，波动小于 1–2%。

---

## 4. 关键结论和发现

### 主要发现
1. **有效的 token 剪枝必须同时满足任务相关性、信息覆盖度和语义多样性**，单一标准无法构建最优保留集。
2. **TOPS 不仅高效，有时还能轻微提升性能**，表明合理剪枝可去除噪声和冗余，缓解幻觉（hallucination）。
3. **多阶段渐进式剪枝优于单层剪枝**，因 token 重要性随网络深度动态变化（见附录 F.2 的低跨层 Jaccard 相似性）。
4. **TOPS 是通用且可扩展的框架**，在图像、视频、OCR 密集等多种任务上均表现优异。

### 方法的局限性
1. **贪心构造存在额外开销**：虽然增量维护降低了复杂度，但在极高压缩比或多剪枝层场景下，$O(KN)$ 的贪心循环仍有一定延迟。
2. **依赖注意力信号作为相关性代理**：对于不使用标准 cross-modal attention 的模型（如线性注意力），效果可能受限。
3. **全局固定超参数**：$\alpha$ 和 $\lambda$ 未针对不同任务或层级自适应调整，可能非最优。
4. **评估范围有限**：未在医疗影像、遥感等高度专业化领域验证泛化性。

### 未来工作方向
- 探索 **并行化或近似算法** 以加速贪心搜索。
- 设计 **自适应权重机制**，根据任务类型或剪枝深度动态调整 $\alpha$ 和 $\lambda$。
- 将三原则框架拓展至 **其他模态压缩任务**（如音频 token 剪枝）。
- 结合 **test-time adaptation** 进一步提升在边缘设备上的部署效率。

> **总结**：TOPS 为视觉 token 剪枝提供了坚实的理论基础和高效的实践方案，证明了“少即是多”的潜力，为未来轻量化 MLLM 设计指明了新方向。

</details>

---

### 11. [Target-Aware Bandit Allocation for Scalable Surrogate Optimization in Chemical Space](https://arxiv.org/abs/2606.26657)

**Authors**: Mohammad Haddadnia, Yuvan Chali, Abhilash Jayaraj, Constance Kraay, Joana Reis, Felix Strieth-Kalthoff, Haribabu Arthanari  
**Category**: cs.LG  
**Published**: 2026-06-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.26657v1  

#### Abstract
Identifying high-utility candidates from massive discrete spaces under expensive evaluations is a recurring challenge across the sciences, with structure-based drug discovery as a prominent example. While surrogate-based optimization can increase sample efficiency by reducing the number of expensive...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Target-Aware Bandit Allocation for Scalable Surrogate Optimization in Chemical Space

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在基于结构的药物发现等科学领域，从超大规模离散空间（如数十亿至数万亿分子）中识别高价值候选物是一个核心挑战。传统 **Surrogate-based Optimization** 虽能提升样本效率，但在现代“按需合成”化学库（make-on-demand libraries）背景下，**全库范围的 Surrogate 推理（inference）本身已成为计算瓶颈**。

标准 Active Learning 或 Bayesian Optimization（BO）假设推理成本远低于真实评估（如分子对接），但这一假设在超大库下不再成立。因此，如何在**同时受限于昂贵的真实评估（evaluation）和高昂的模型推理（inference）** 的双重约束下进行高效优化，是本文要解决的关键问题。

### 提出了什么新方法或新思路
作者提出了 **BoBA**（**Bayesian Optimization with BAndits**），一种目标感知的、可扩展的代理优化框架，其核心思想是：

- 将庞大的化学空间 $ \mathcal{X} $ 划分为 $ K $ 个子空间（$ \{X_1, ..., X_K\} $），每个子空间视为一个 **multi-armed bandit** 的“臂”（arm）。
- 在每轮迭代中，**bandit 算法选择最有希望的子空间**，然后仅在该子空间内执行 Surrogate 模型推理和候选选择。
- 通过这种方式，**避免了对整个库进行全量推理**，实现了全局探索（global exploration）与局部开发（local exploitation）的解耦。

### 相比现有方法的优势
- **显著降低推理成本**：将推理复杂度从 $ \mathcal{O}(NT) $ 降至约 $ \mathcal{O}(NT/K) $，其中 $ N $ 是库大小，$ T $ 是评估轮次。
- **保持高优化性能**：通过合理的 bandit 策略和空间划分，可在大幅减少推理的情况下，接近甚至媲美全库推理 BO 的性能。
- **可调权衡机制**：用户可通过调整分区数量 $ K $，灵活控制“优化性能”与“推理成本”之间的权衡。
- **适用于超大规模库**：为未来处理千亿至万亿级虚拟筛选提供了可行路径。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在多个真实世界药物发现数据集上进行，均来自 **Enamine** 的超大化合物库：
- **ENAMINE-5M**：从约 690 亿分子库中随机抽取的 500 万化合物，对接靶点为 **CKB** 和 **NEDD4**。
- **ENAMINE-S-3.9M**：从 S-class 数据库中抽取的 390 万化合物，对接靶点为 **CKB** 和 **NEDD4**。
- **ENAMINE-HTS**：含 200 万分子的 HTS 库，对接靶点为 **Thymidylate Kinase (TMK)**。
- **ZINC-AmpC**：用于可扩展性测试，库大小从 $ 10^5 $ 到 $ 10^8 $ 分子不等。

### 实验设置和评估指标
- **总评估预算**：$ T = 20 $ 轮，每轮批量 $ B = 5,000 $，共 $ 100,000 $ 次真实评估。
- **评估指标**：
  - 主要指标：**检索到的 Top-100 / Top-1000 / Top-10000 高分分子数量**。
  - 成本指标：**总 Surrogate 推理次数**（反映计算开销）。
- **Surrogate 模型**：两层前馈神经网络（512→128），使用 **Linearized Laplace Approximation** 进行不确定性估计。
- **Acquisition Function**：**Upper Confidence Bound (UCB)**。
- **Partitioning 方法**：使用 **k-Means Clustering** 在不同特征空间（如 T5Chem embeddings、physicochemical descriptors）中进行划分。

### 基线方法对比
- **Full-library BO**：标准贝叶斯优化，每轮对全库进行推理（作为性能上限）。
- **Random Subsampling (SS-BO)**：每轮随机选取一个与 BoBA 子集同规模的子库进行推理。
- **Random Partitioning**：将库随机划分为 $ K $ 个静态子集，用相同 bandit 策略选择。
- **不同 Bandit 策略**：比较 **UCB1**、**e-greedy**、**Softmax** 在不同探索参数下的表现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **ENAMINE-5M (CKB)** 上，BoBA (UCB1, K=100) 可以在仅进行约 **1% 的推理量**（相比全库 BO）的情况下，**恢复超过 90% 的 Top-100 分子**。
- 在 **ZINC-AmpC** 上，随着库大小从 $ 10^5 $ 增加到 $ 10^8 $，BoBA 的相对性能（相对于全库 BO）**并未下降**，反而在某些 $ K $ 下趋近于全库 BO，验证了其可扩展性。

### 与基线方法的对比结果
| 方法 | 性能（Top-100 检索率） | 推理成本 |
|------|------------------------|----------|
| Full-library BO | 100% (基准) | 100% |
| BoBA (UCB1, K=100) | ~90–95% | ~1–2% |
| BoBA (e-greedy, K=100) | <50% | ~1–2% |
| Random Subsampling | ~60–70% | ~1–2% |
| Random Partitioning | ~40–60% | ~1–2% |

> ✅ **BoBA (UCB1)** 显著优于其他低推理成本方法。

### 消融实验结果
#### （1）Bandit 策略的影响
- **UCB1** 表现最佳且最稳健，尤其在 $ K $ 较大时仍能维持高性能。
- **e-greedy** 和 **Softmax** 在 $ K $ 增大时性能急剧下降，因其缺乏基于不确定性的探索机制，易陷入局部最优。

#### （2）Partitioning 方式的必要性
- **Structured Clustering (k-Means)** 明显优于 **Random Partitioning**，表明**化学空间的内在结构对有效分配至关重要**。
- **Persistent partitions + bandit feedback** 优于每轮随机采样子库（SS-BO），说明**持续的空间记忆和反馈驱动是关键**。

#### （3）特征空间的影响
- 使用 **T5Chem embeddings** 构建的分区显著优于基于传统 **Physicochemical Descriptors** 的分区。
- 表明**预训练 Foundation Model 提供的表示更能捕捉与优化目标相关的化学多样性**。

#### （4）分区数量 $ K $ 的影响
- 存在一个**可调的性能-成本权衡**：$ K $ 越大，推理成本越低，但 bandit 分配难度增加。
- 理论分析给出启发式最优尺度：$ K^* \propto N^{2/3}T^{1/3} $，即**库越大，支持更细粒度的划分**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **推理成本已成为超大库优化的新瓶颈**，必须被显式建模。
2. **BoBA 通过 bandit 引导的分区推理，成功解耦了全局搜索与局部优化**，实现了高效的资源分配。
3. **Optimism-under-uncertainty 的 bandit 策略（如 UCB1）对维持高性能至关重要**，尤其是在多臂场景下。
4. **有意义的空间划分（如基于 T5Chem 的聚类）是 BoBA 成功的前提**，随机划分无法获得收益。
5. **存在一个可调的性能-推理成本权衡**，用户可根据实际需求选择 $ K $。
6. **该权衡随库规模增大而更有利**，表明 BoBA 特别适合未来超大规模虚拟筛选。

### 方法的局限性
- 当前实现依赖于**库的显式枚举和全局特征化/聚类**，这在真正无限库（如 synthon-level）中不可行。
- 聚类是一次性预处理，未考虑任务特定目标的动态适应。
- 所有实验基于已知对接分数，真实湿实验可能存在噪声和非平稳性。

### 未来工作方向
- 将 BoBA 扩展到 **synthon-level 搜索**，直接在构建块空间进行 bandit 分配，避免全库枚举。
- 探索更先进的 bandit 算法，如 **rotting bandits**（考虑高分分子耗尽效应）或 **contextual bandits**。
- 结合领域专家知识设计**任务定制的特征空间**，进一步提升分区质量。
- 研究更鲁棒的不确定性估计方法，并探索其在主动学习中的作用。
- 将框架推广至其他大规模离散优化场景，如材料设计、催化剂发现等。

> 💡 **总体而言，BoBA 为在“评估贵 + 推理也贵”的新时代下进行高效科学发现提供了一个原则性强、可扩展且实用的新范式**。

</details>

---

### 12. [EvoOptiGraph: Weakness-Driven Coevolution via Graph-Based Structural Generation for Optimization Modeling](https://arxiv.org/abs/2606.26578)

**Authors**: Qingcan Kang, Mingyang Liu, Xiaojin Fu, Shixiong Kai, Tao Zhong, Mingxuan Yuan  
**Category**: cs.AI  
**Published**: 2026-06-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.26578v1  

#### Abstract
Automating optimization modeling from natural language with large language models (LLMs) faces two key challenges. First, training corpora lack structural diversity. Second, data generation pipelines remain static and decoupled from model learning. To address these challenges, we propose EvoOptiGrap...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EvoOptiGraph: Weakness-Driven Coevolution via Graph-Based Structural Generation for Optimization Modeling**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于大语言模型（LLMs）的优化建模（optimization modeling）面临两大挑战：
1. **训练数据缺乏结构性多样性**：现有合成方法多在固定模板内调整参数或文本表达，难以生成结构新颖的混合整数线性规划（MILP）实例。
2. **数据生成与模型学习脱节**：传统流程中数据集一旦构建即固定，无法动态响应模型在训练过程中的失败模式。

### **提出的新方法与思路**
作者提出 **EvoOptiGraph**，一种**基于图的弱点击中协同进化框架**，实现数据与模型的闭环共演化：
- **图表示**：将每个 MILP 表示为一个带属性的二分图（attributed bipartite graph），节点为变量与约束，边为非零系数。
- **图演化生成**：在图空间上应用遗传操作（crossover、mutation），直接对结构进行变异，超越参数扰动。
- **弱点击中机制**：通过验证集上的图特征分析（如变量比例、稀疏性等），利用 SHAP 值提取模型失败的“弱点向量”（weakness vector），指导后续数据生成偏向这些结构。
- **两阶段训练**：
  - **监督微调（SFT）**：在初始验证数据集上预热模型。
  - **强化学习（RLVR）**：使用可验证奖励（verifiable rewards）进行策略更新，奖励基于代码可执行性、解可行性、目标值接近度和正确性。

### **相比现有方法的优势**
- **结构可控性更强**：图表示支持细粒度结构操作，避免文本生成的不可控性。
- **动态适应性**：数据分布随模型能力变化而演进，持续暴露盲点。
- **高保真验证**：所有生成图均经确定性编译为 Gurobi 代码并执行验证，确保语义一致性。
- **高效反馈闭环**：从模型失败 → 结构诊断 → 弱点引导生成 → 再训练，形成自适应循环。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在六个公开的 OR 基准上进行评估：
| 数据集 | 描述 |
|--------|------|
| **NL4Opt** | 自然语言到 LP/MILP 的映射任务，含约 1,100 个问题，本文使用其清洗后版本（214 实例）。 |
| **MAMO** | 分为 EasyLP 和 ComplexLP 子集，强调建模而非求解，使用清洗后版本（545 + 111 实例）。 |
| **NLP4LP** | 人工标注的 LP/MILP 问题，覆盖调度、背包等经典场景（178 实例）。 |
| **ComplexOR** | 复杂 OR 问题，如批量排程、供应链设计，需多步推理（18 实例）。 |
| **IndustryOR** | 来自制造、物流、金融等真实工业场景的优化问题（42 实例）。 |

### **实验设置与评估指标**
- **主干模型**：基于 **Qwen3-8B** 进行微调。
- **训练流程**：
  - 初始 SFT 使用 5,000 个由种子生成器产生的实例。
  - 每轮 RL 使用 1,000 个新生成的弱点对齐实例。
- **评估指标**：
  - **Pass@1 准确率**：生成的代码运行后得到的目标值与基准最优值一致的比例。
  - **可执行性、可行性、接近度、正确性** 四阶段奖励用于 RL。
- **消融研究**：比较 SFT-only 与 RL-only 变体。

### **基线方法对比**
涵盖三类主流方法：
| 类别 | 基线模型 |
|------|---------|
| **零样本通用 LLM** | GPT-4o, DeepSeek-V3, Qwen3-32B, Qwen2.5-72B-Inst |
| **代理方法（Agentic）** | OptiMUS, Chain-of-Thought (CoT), Chain-of-Experts (CoE) |
| **专用微调模型** | ORLM (8B), LLMOPT (14B), OptMATH (32B) |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
在六大数据集上的 **Pass@1 平均准确率**如下：

| 模型 | NL4Opt | MAMO-Easy | MAMO-Complex | NLP4LP | ComplexOR | IndustryOR | **Avg.** |
|------|--------|-----------|--------------|--------|-----------|------------|----------|
| GPT-4o | 73.4 | 95.2 | 46.0 | 87.6 | 38.9 | 66.7 | 68.0 |
| DeepSeek-V3 | 79.9 | 96.7 | 55.0 | 94.9 | 44.4 | 69.1 | 73.3 |
| ORLM (8B) | 73.8 | 90.4 | 59.5 | 76.4 | 50.0 | 42.9 | 65.5 |
| **EvoOptiGraph (8B)** | **96.7** | **98.0** | **69.4** | **97.2** | **50.0** | **57.1** | **78.1** |

> ✅ **EvoOptiGraph 在平均准确率上显著领先（78.1%），且在多数单项上达到最佳或次佳**。

### **与基线方法的对比结果**
- 超越了更大规模的通用模型（如 671B 的 DeepSeek-V3）。
- 显著优于专用微调模型（如 ORLM、LLMOPT）和代理方法（如 OptiMUS）。
- 尤其在 **MAMO-Complex** 和 **NL4Opt** 上提升显著，表明对复杂结构建模能力强。

### **消融实验结果**
| 方法 | 平均 Pass@1 | 相对 EvoOptiGraph 差距 |
|------|------------|------------------------|
| **SFT-only** | 68.5 | ↓9.6% |
| **RL-only** | 63.4 | ↓14.7% |
| **EvoOptiGraph (完整)** | **78.1** | — |

> 🔍 发现：
> - **SFT-only** 初期有效但很快饱和甚至退化，说明静态数据易导致过拟合。
> - **RL-only** 缺乏良好初始化，训练不稳定。
> - **两者结合 + 弱点击中生成** 是关键。

此外，**数据多样性对比**显示：
- EvoOptiGraph 生成的数据与原始 53 类问题的 **Structural Similarity 仅为 66.93%**。
- 对比方法 OptMATH 达到 93.04%，说明其数据更接近模板。
> ➡️ EvoOptiGraph 成功生成了更多样化的结构。

---

## **4. 关键结论和发现**

### **主要发现**
1. **针对性数据-模型协同进化是有效的**：通过图表示连接数据生成与模型诊断，能持续暴露并修复模型弱点。
2. **结构级生成优于参数级扰动**：图上的 crossover 和 mutation 可产生真正新颖的混合问题（如生产调度 + 背包）。
3. **弱点击中机制提升了训练效率**：生成的数据更具信息量，避免重复学习已掌握的模式。
4. **小模型也能超越大模型**：8B 的 EvoOptiGraph 超越了数十倍参数的通用 LLM，说明**高质量、适配的数据比单纯扩大模型更有效**。

### **方法的局限性**
- 当前仅支持 **MILP**，不适用于非线性或动态规划。
- 种子生成器覆盖有限，影响初始种群多样性。
- 验证成本高，尤其对大规模实例，求解耗时成为瓶颈。
- 弱点向量是经验信号，非因果解释，可能引入偏差。

### **未来工作方向**
- 扩展至更广的优化类别（如 MINLP、Stochastic Programming）。
- 提升生成与验证效率，例如使用代理模型预测可行性。
- 研究弱点信号在不同问题族间的迁移能力。
- 探索与其他推理增强技术（如 ToT、MCTS）的结合。

---

> 📌 **总结**：  
> **EvoOptiGraph 展示了一条通往高效、自适应 LLM 训练的新路径——不是被动接受数据，而是主动塑造数据以驱动模型进化。其核心思想“以模型弱点为导向的结构化数据生成”具有广泛启发意义，不仅限于优化建模，也可推广至其他形式化推理任务。**

</details>

---

### 13. [At the Edge of Understanding: Sparse Autoencoders Trace The Limits of Transformer Generalization](https://arxiv.org/abs/2606.26396)

**Authors**: Praneet Suresh, Jack Stanley, Sonia Joseph, Luca Scimeca, Danilo Bzdok  
**Category**: cs.LG  
**Published**: 2026-06-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.26396v1  

#### Abstract
Pre-trained transformers have demonstrated remarkable generalization abilities, at times extending beyond the scope of their training data. Yet, real-world deployments often face unexpected or adversarial data that diverges from training data distributions. Without explicit mechanisms for handling s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*At the Edge of Understanding: Sparse Autoencoders Trace The Limits of Transformer Generalization*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于 **Transformer 模型在面对分布外（Out-of-Distribution, OOD）输入时的脆弱性问题**。尽管大型语言模型（LLMs）具备强大的泛化能力，但在实际部署中，面对细微扰动（如拼写错误）、对抗性提示（jailbreak prompts）或风格偏移等 OOD 输入时，其表现可能急剧下降，甚至产生有害输出。传统 OOD 检测方法多基于输入统计特征（如 softmax 概率、熵），缺乏对模型内部机制的理解，难以精准定位和修复。

### 提出了什么新方法或新思路
论文提出了一种**基于 Sparse Autoencoders（SAEs）的机制性框架**，用于探测和增强 LLMs 在 OOD 场景下的鲁棒性。核心思想是将 SAE 视为一个“显微镜”，观察 LLM 内部表示空间中的异常激活模式。

- **SAE 能量分数（Energy Score）**：提出一个复合指标，结合 SAE 的两个信号：
  - **L₀（非零激活概念数）**：OOD 输入会激活更多稀疏且通常不活跃的语义概念。
  - **重建误差（Reconstruction Error）**：OOD 输入难以被 SAE 准确重建。
  该能量分数量化了输入样本偏离训练数据流形的程度。

- **机制性诊断与干预**：
  - 利用 SAE 追踪 OOD 输入如何导致模型内部出现“虚假概念”（spurious concepts）。
  - 基于能量分数进行**流形感知微调（manifold-informed fine-tuning）**，优先选择高能量样本以提升鲁棒性。
  - 提出 **SAE-informed LoRA** 方法，通过调整注意力投影矩阵，将成功 jailbreak 提示的特征激活向未成功提示的中心对齐，从而抑制攻击。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **视角** | 数据驱动（data-driven），关注输入本身 | 模型内在（model-inherent），关注模型内部计算过程 |
| **可解释性** | 黑盒，仅提供是否 OOD 的判断 | 白盒，揭示具体哪些概念被异常激活 |
| **干预能力** | 多为输入过滤或拒绝 | 可进行精细化的内部参数调整（如 LoRA） |
| **有效性** | 对细微扰动敏感度低 | 能检测到如单字符 typo 等微小变化 |
| **通用性** | 通常针对特定 OOD 类型设计 | 统一框架适用于 typo、jailbreak、ASR 噪声、写作风格等多种 OOD |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **TinyStories**：用于预训练小型 GPT-2 模型及训练 SAE，确保其训练数据纯净无 typo。
- **MMLU / MMLU-CF**：标准多任务理解基准，用于评估 OOD 输入（注入 typo）对模型性能的影响。
- **WildJailbreak**：包含 9,938 条对抗性 jailbreak 提示，用于测试模型安全性和防御效果。
- **Spoken-SQuAD**：ASR 转录的 SQuAD 数据集，用于模拟真实语音场景中的噪声输入。
- **Shakespeare / Modern Poetry**：用于测试写作风格作为 OOD 的情形。

### 实验设置和评估指标
- **模型**：
  - **GPT-2 (25M)**：玩具模型，用于控制实验。
  - **Llama 3.1 8B**：主流开源大模型，用于真实世界验证。
  - **GPT-4o mini / GPT-5-thinking-nano**：闭源前沿模型，通过 API 测试其 OOD 鲁棒性。
- **SAE 设置**：
  - 训练于残差流（residual stream）激活。
  - 使用 L1 正则化，不采用 top-k。
  - GPT-2：每层 d_SAE = 4096；Llama 3.1 8B：使用 Goodfire 提供的 d_SAE = 65536 的 SAE。
- **评估指标**：
  - **MMLU 准确率**：衡量任务性能下降。
  - **SAE 能量分数**：核心 OOD 度量。
  - **LoRA 微调后的 jailbreak 成功率**：衡量防御效果。
  - **Exact Match (EM)**：用于 Spoken-SQuAD 的问答任务。
  - **Pearson 相关系数 / ROC-AUC**：比较不同 OOD 指标的有效性。

### 基线方法对比
- **Entropy Score**：基于模型输出概率的不确定性。
- **Mahalanobis Distance**：基于类条件高斯假设的距离度量。
- **Llama Prompt Guard 2**：Meta 提供的输入侧分类器，用于 jailbreak 检测。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **Typos 对性能的影响**：
  - Llama 3.1 8B 在 MMLU 上，当 5% 的词含 typo 时，准确率从 **66.7% → 51.01%**。
  - GPT-4o mini 从 **82.10% → 74.85%**。
  - GPT-5-thinking-nano 从 **91.26% → 86.45%**。
- **SAE 能量分数有效性**：
  - 在 20% typo 下，能量分数与 entropy/Mahalanobis 的 Pearson 相关系数均 < 0.09，表明其提供了独特信号。
  - 在区分 ID/OOD（20% typo）任务上，能量分数的 ROC-AUC 达 **0.8150**，显著优于 entropy (0.5732) 和 Mahalanobis (0.7617)。
- **Jailbreak 防御效果**：
  - 基础 Llama 3.1 8B 的 jailbreak 成功率为 **46%**。
  - 经 SAE-informed LoRA 微调后，成功率降至 **7%**。
  - 对比 Llama Prompt Guard 2，仅能标记 **48.3%** 的成功 jailbreak 提示。
- **ASR 噪声鲁棒性提升**：
  - SAE-aligned Llama 3.1 8B 在 Spoken-SQuAD 上的 EM 从 **49.45% → 58.33%**。
  - 有趣的是，在原始 SQuAD（ID）上的 EM 也从 **59.97% → 67.88%**，显示了正向迁移。

### 与基线方法的对比结果
- **微调效率**：在 GPT-2 上，使用高能量分数样本微调，达到与低能量样本相当的验证损失，所需训练步数仅为后者的 **三分之二**。
- **样本选择质量**：使用能量分数选择样本进行微调，最终验证损失比使用 entropy 或 Mahalanobis 选择的样本更低（见 Table 11）。
- **OOD 检测**：One-Class SVM 基于 SAE 特征激活数检测 jailbreak，总体准确率为 **67.8%**，虽不高但证明了可行性。

### 消融实验结果
- **SAE 初始化稳定性**：使用三个不同随机初始化的 SAE 进行实验，能量分数分布和微调趋势高度一致，说明结果依赖于 SAE 子空间而非具体特征。
- **层间一致性**：不同 transformer 层的能量分数对 ID/OOD 的分类具有高度一致性（Cohen's Kappa 最高达 0.984），支持了方法的稳健性。
- **LoRA 影响范围**：微调后，MMLU 整体准确率仅下降 **0.09%**，且在 OR-BENCH 上的过度拒绝率从 3.5% 降至 3.0%，表明干预是精确且非破坏性的。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **OOD 输入导致内部表示退化**：即使是轻微的分布偏移（如 typo），也会使 LLM 在内部激活大量**虚假且分散的概念**，这些概念在正常输入中极少出现。
2. **SAE 是有效的 OOD 探针**：SAE 能够以机制性的方式捕捉这种内部退化，其能量分数是一个强大且独特的 OOD 指标。
3. **OOD 检测与鲁棒性提升可统一**：基于 SAE 的分析不仅能诊断问题，还能直接指导模型加固。无论是通过**样本选择微调**还是**特征对齐微调**，都能显著提升模型对 OOD 的鲁棒性。
4. **Jailbreak 是一种 OOD 攻击**：成功的 jailbreak 提示之所以有效，是因为它们在表示空间中触发了 OOD 激活模式，从而绕过了对齐机制。这为理解 LLM 安全漏洞提供了新的机制性视角。

### 方法的局限性
- **依赖高质量 SAE**：方法效果取决于 SAE 的训练质量（高解释方差、适当稀疏性）。当前 SAE 训练仍具挑战性。
- **计算开销**：需要额外训练或获取 SAE，并在推理时计算激活，增加了部署成本。
- **OOD 定义边界模糊**：并非所有 OOD 激活都是有害的，如何区分“良性创新”与“有害漂移”仍是开放问题。
- **未探索所有 OOD 类型**：实验集中在 typo、jailbreak、ASR 噪声和风格迁移，其他复杂 OOD 场景有待验证。

### 未来工作方向
- **扩展至更多模型和模态**：将框架应用于更大规模模型（如 Llama 3 70B）和多模态模型（如 vision transformers）。
- **实时动态防御**：开发基于 SAE 能量分数的实时监控系统，在检测到高风险 OOD 输入时动态调整模型行为。
- **自动化特征编辑**：探索无需人工干预的自动化方法，识别并“切除”或“重定向”与有害行为相关的 spurious concepts。
- **理论深化**：进一步研究 SAE 子空间与模型泛化能力之间的理论联系，建立更坚实的数学基础。

> **总结**：本文开创性地将 SAE 从解释工具转变为**主动的 OOD 诊断与防御设备**，实现了从“数据分布”到“模型内部计算过程”的范式转变，为构建更安全、可靠、可解释的 LLM 部署提供了重要路径。

</details>

---

### 14. [Sample-efficient Transfer Reinforcement Learning via Adaptive Reward Shaping and Policy-Ratio Reweighting Strategy](https://arxiv.org/abs/2606.26527)

**Authors**: Wenjie Huang, Yang Li, Jingjia Teng, Mingwei Jin, Kai Song, Yougang Bian, Yongfu Li, Qisong Yang, Helai Huang  
**Category**: cs.LG  
**Published**: 2026-06-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.26527v1  

#### Abstract
Transfer learning improves policy learning efficiency by reusing knowledge from source tasks, providing a feasible paradigm for safe and efficient autonomous highway lane changing decision-making. Existing methods frequently encounter transfer mismatch induced by distribution shifts between source a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sample-efficient Transfer Reinforcement Learning via Adaptive Reward Shaping and Policy-Ratio Reweighting Strategy

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对**自动驾驶高速公路变道决策**中的两个核心挑战：

- **训练安全性不足**：传统强化学习（RL）在目标域中需要大量探索，容易产生危险行为（如碰撞），难以满足安全关键场景的需求。
- **迁移学习中的分布偏移与负迁移**：源任务与目标任务之间存在环境差异（distribution shift），导致知识迁移效率低下，甚至出现性能下降（negative transfer）。

现有方法通常仅在动作层面提供教师指导，缺乏对价值函数学习的安全引导，且干预机制固定，无法自适应调整。

---

### **提出了什么新方法或新思路**

作者提出了一种**基于教师引导的安全迁移强化学习框架**（Teacher-Guided Safe Transfer RL），整合了三个核心机制：

#### （1）**自适应教师干预机制（Adaptive Teacher Intervention）**
- 基于瞬时安全成本（instantaneous safety cost）触发教师干预。
- 设计动态衰减的干预阈值 $ T $，随着学生策略安全性的提升逐步减少干预频率，实现从“依赖教师”到“自主决策”的平滑过渡。
- 干预过程自然生成两类样本（student-generated 和 teacher-intervened），用于后续双源数据联合训练。

#### （2）**教师引导的安全迁移学习机制（Teacher-guided Safe Transfer Learning）**
- 将教师策略的动作评估信息（action-evaluation）作为**奖励塑形项**（adaptive reward shaping）引入学生策略的价值目标更新中：
  $$
  \text{bonus} = \log \pi_t(a|s),\quad Q_{\text{target}} = r + \beta \cdot \text{bonus} + \gamma V_{\text{next}}
  $$
- 引入可学习的安全权重 $ \beta $（类似Lagrange multiplier），根据当前策略的安全状态动态调节教师引导强度：越不安全则引导越强。

#### （3）**教师引导的优化加权机制（Teacher-guided Optimization Reweighting）**
- 构造基于似然比的加权因子：
  $$
  \rho = \exp(\log \pi_s(a|s) - \log \pi_t(a|s))
  $$
- 对教师干预样本进行重加权，增强与当前学生策略一致的样本影响力，抑制分布不一致区域带来的不稳定更新，提高迁移稳定性。

此外，论文还对混合行为策略（mixed-behavior policy）下的回报上下界进行了理论分析，证明了该机制既能提升性能上界，又能保证性能下界。

---

### **相比现有方法的优势**

| 维度 | 优势 |
|------|------|
| **安全性** | 显著降低训练阶段的碰撞风险和安全成本，尤其适用于安全关键任务。 |
| **效率** | 利用教师知识加速收敛，减少环境交互次数，提升sample efficiency。 |
| **稳定性** | 通过自适应干预衰减和样本重加权，缓解负迁移和训练波动。 |
| **知识利用深度** | 不仅传递动作建议，还将教师的“价值判断”融入学生的学习过程，实现更深层次的知识迁移。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **仿真环境**：基于 `gym-highway-env` 修改构建的高速公路变道模拟器。
- **真实交通数据验证**：采用 **NGSIM US-101 数据集** 进行现实场景下的泛化能力测试。

---

### **实验设置**

- **任务设定**：智能体需完成向最右侧车道的安全高效变道，同时避免碰撞、保持合理车速。
- **状态空间**：包含自车及周围车辆的位置、速度等相对信息。
- **动作空间**：离散高阶指令（左转、右转、加速、减速、巡航）。
- **奖励设计**：综合安全（collision penalty）、效率（speed reward）、目标达成（goal reward）三项加权。
- **成本函数**：基于时间车头距（Time Headway, THW）定义安全成本，用于约束安全边界。

---

### **评估指标**

| 指标 | 含义 | 方向 |
|------|------|-------|
| **Average Reward** | 每回合平均累积奖励 | ↑ |
| **Average Cost** | 每回合平均安全成本 | ↓ |
| **Crash Ratio** | 发生碰撞的比例 | ↓ |
| **Average Velocity** | 平均行驶速度 | ↑ |

---

### **基线方法对比**

- **SAC** [44]：标准的Soft Actor-Critic算法，无安全约束。
- **PPO-Lag** [45]：带Lagrangian约束的PPO，用于安全RL。
- **TS2C** [36]：基于在线演示的安全策略优化方法。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（中等交通密度下）**

| 方法 | Avg Reward | Avg Cost | Crash Ratio | Avg Velocity (m/s) |
|------|------------|----------|-------------|---------------------|
| **Ours** | **47.34** | **0.11** | **0.00** | **21.80** |
| SAC | 43.57 | 1.69 | 0.08 | 20.76 |
| PPO-Lag | 42.54 | 0.68 | 0.05 | 20.31 |
| TS2C | 27.00 | 0.23 | 0.00 | 6.65 |

> ✅ **改进幅度**：
> - 安全性提升超过 **52.2%**（以 crash ratio 和 cost 衡量）
> - 效率提升约 **5.0%**（以平均速度衡量）

---

### **与基线方法的对比结果**

- **vs SAC**：
  - 成本降低 **93.5%**（0.11 vs 1.69）
  - 碰撞率降低至 **0**
  - 收敛更快、更稳定，早期无剧烈波动
- **vs PPO-Lag**：
  - 成本降低 **83.8%**
  - 碰撞率更低（0 vs 0.05）
  - 速度更高（+7.3%）
- **vs TS2C**：
  - 虽然都实现了低 crash ratio，但 TS2C 过于保守，平均速度仅为 6.65 m/s，严重牺牲效率；
  - 本文方法在保障安全的同时维持高速运行，真正实现**安全与效率的平衡**。

---

### **消融实验结果（Ablation Study）**

| 配置 | Avg Reward (Med) | Avg Cost (Med) | Crash Ratio (Med) | 结论 |
|------|------------------|----------------|--------------------|------|
| 完整模型（SG+ID+DS） | 47.34 | 0.11 | 0.00 | 最优性能 |
| w/o SG（无安全引导） | 46.99 | 0.28 | 0.01 | 安全性显著下降 |
| w/o SG & ID（无引导+无衰减） | 41.76 | 0.44 | 0.21 | 性能全面恶化 |
| w/o 所有模块 | 37.30 | 0.75 | 0.39 | 接近崩溃水平 |

> 🔍 **发现**：
> - **SG（安全引导）** 是提升安全性和效率的关键。
> - **ID（干预衰减）** 对防止过度依赖教师至关重要。
> - **DS（双源数据训练）** 即使单独使用也能有效降低碰撞率，说明异构数据融合本身具有价值。

---

## 4. 关键结论和发现

### **主要发现**

1. **教师干预不仅能保安全，还能提性能上限**  
   理论分析表明，混合行为策略可在保证性能下界的同时提升上界，支持“安全即促进学习”的理念。

2. **将教师的“价值判断”而非仅“动作建议”迁移到学生端，是高效迁移的关键**  
   通过 reward shaping 和 policy-ratio reweighting，实现了深层知识迁移。

3. **自适应机制优于固定规则**  
   动态调整干预强度和引导权重，使得系统能随学习进程自动演化，兼顾初期安全与后期自主性。

4. **双源数据联合训练增强了样本多样性与鲁棒性**  
   学生自主探索样本 + 教师纠正样本共同构成更丰富的经验池。

5. **在真实交通流（NGSIM）中仍表现优异**  
   在 US-101 数据集中：
   - 平均速度达 **29.9 m/s**（≈107.6 km/h），远高于 PPO-Lag 的 10.55 m/s
   - 碰撞率为 **0**，而 PPO-Lag 为 0.35
   - 验证了方法在真实复杂交通中的泛化能力。

---

### **方法的局限性**

- 当前教师策略是在简化环境中预训练获得，其最优性未被保证。
- 教师与学生的动作空间必须一致，限制了跨模态迁移的应用。
- 干预机制为“硬切换”，即一旦触发就全程接管，未来可探索软融合方式。

---

### **未来工作方向**

1. **更灵活的教师-学生动作融合机制**：例如加权平均或门控机制，替代当前的硬切换。
2. **Meta-transfer RL 范式探索**：实现少样本（few-shot）安全迁移学习。
3. **多教师集成与不确定性建模**：结合多个专家策略并估计其可靠性。
4. **扩展至城市道路、交叉口等更复杂场景**。

---

> 📌 **总体评价**：  
> 本文提出的 **Teacher-Guided Safe Transfer RL 框架** 在**安全性、效率、稳定性**三方面取得了显著突破，为面向安全关键型应用的迁移强化学习提供了系统性解决方案，具备较强的实用前景和理论深度。

</details>

---

### 15. [fTNN: a tensor neural network for fractional PDEs](https://arxiv.org/abs/2606.27140)

**Authors**: Qingkui Ma, Hehu Xie, Xiaobo Yin  
**Category**: cs.LG  
**Published**: 2026-06-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.27140v1  

#### Abstract
We develop the fTNN, a deterministic tensor neural network subspace method for problems involving the fractional Laplacian on bounded domains, taking the fractional Poisson equation and time-dependent fractional advection-diffusion equation as typical representatives. The work employs a geometry-ada...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：fTNN: a tensor neural network for fractional PDEs

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**bounded domains**上涉及**fractional Laplacian**的**fractional PDEs**（如fractional Poisson方程和time-dependent fractional advection-diffusion方程）的数值求解难题。这类问题存在三大挑战：
- **hypersingular nonlocal kernels**：导致积分核在近场区域高度奇异。
- **exterior Dirichlet constraints**：需要处理外部区域的边界条件。
- **reduced boundary regularity**：解在边界附近通常具有**boundary singularity**（如 $ \text{dist}(x,\partial\Omega)^{\alpha/2} $），导致传统方法精度下降。

现有基于**Monte Carlo**的fPINN方法（如MC-fPINN, QE-MC-fPINN）虽然能处理高维问题，但在低至中等维度（d=1,2,3）下，其**stochastic noise**（尤其是angular Monte Carlo sampling引入的噪声）成为精度提升的主要瓶颈。

### 提出的新方法和新思路
作者提出了一种名为 **fTNN (fractional Tensor Neural Network)** 的确定性张量神经网络子空间方法，其核心创新点如下：

1. **Fully Deterministic Integration Framework**：
   - 采用**geometry-adapted integration split**，将fractional Laplacian分解为三个部分：**singular near field**, **regular interior far field**, 和 **analytical exterior far field**。
   - 对所有部分进行完全确定性的数值积分：
     - **Radial integrals**: 近场奇异积分用**Gauss-Jacobi quadrature**，远场正则积分用**Gauss quadrature**。
     - **Angular integrals**: 所有方向角积分均用**deterministic angular quadrature**（如Gauss-Legendre规则），彻底消除了Monte Carlo采样带来的随机噪声。

2. **Boundary-Singularity-Aware Trial Functions**：
   - 构造了显式包含边界特征的试函数：$ \psi(x;c,\theta) = \sum_j c_j b(x)^{\mu_j} \phi_j(x;\theta) $，其中 $ b(x) $ 是一个在边界上为零的**boundary feature function**。
   - 提出两种自适应策略来选择主导指数 $ \mu_1 $：
     - **BFE (Boundary Feature Enhanced)**: 当源项 $ f(x) $ 不引入额外奇异性时，设 $ \mu_1 = \alpha/2 $，匹配算子诱导的奇异性。
     - **BRFE (Boundary and Right-hand-side Feature Enhanced)**: 当源项 $ f(x) $ 本身也具有奇异性（即 $ f_{\text{reg}}(x) \sim b(x)^s, s<0 $）时，设 $ \mu_1 = \alpha + s $，以捕捉算子和源项共同作用下的复合奇异性，并使用**weighted Gauss-Jacobi quadrature**计算损失函数。

3. **Spatiotemporally Separable Neural Network (STSNN) & Alternating Optimization**：
   - 针对time-dependent fPDEs，设计了**STSNN**架构，将时空残差分解为时空低维积分的乘积和。
   - 结合**alternating neural network subspace optimization**策略：交替优化线性系数 $ c $（通过最小二乘法求解线性系统）和神经网络参数 $ \theta $（通过梯度下降）。这改善了优化条件，加速了收敛。

### 相比现有方法的优势
- **更高的精度**：在d=1,2,3维度上，相对 $ L^2 $ 误差比fPINN、MC-fPINN等基线方法低**1到3个数量级**。
- **更好的稳定性**：完全确定性的框架避免了Monte Carlo方法固有的方差和不稳定性，训练过程更平滑。
- **高效的长时模拟**：STSNN的分离结构允许在时间维度上使用密集的Gauss quadrature，而不会因内存爆炸而失效，特别适合处理Caputo导数的非局部记忆效应。
- **自适应奇异性处理**：BFE/BRFE策略能有效应对不同来源的边界奇异性，提升了方法的鲁棒性。

---

## 2. 核心实验方法和设置

### 数据集与测试问题
实验涵盖了多种维度和类型的fPDEs，主要包括：
- **Fractional Poisson Equation (fPE)**:
  - 1D: 区间 $ (0,1) $ 上的多个构造解问题，包括光滑解 $ u=x(1-x)^k $ 和具有不同指数的紧支撑函数 $ u=(1-x^2)^{\beta_1}_+ + (1-x^2)^{\beta_2}_+ $。
  - 2D/3D: 单位正方形/立方体 $ (0,1)^d $ 上的构造解问题 $ u=\prod_k b(x)^{\beta_k} $。
  - 2D/3D: 单位球 $ \{ \|x\|<1 \} $ 上的解析解问题，如 $ u=(1-\|x\|^2)^{1+\alpha/2} $ 和 $ u=(1-\|x\|^2)^{\alpha/2} $。
- **Time-dependent Fractional Advection-Diffusion Equation (ADE)**:
  - 在单位球上的时空分数阶对流扩散方程，用于短时和长时模拟测试。

### 实验设置和评估指标
- **硬件**：在NVIDIA GeForce RTX 4090 D GPU上进行。
- **神经网络架构**：每个子网络（subnetwork）为2层隐藏层，激活函数为Tanh。
- **评估指标**：
  - **Relative $ L^2 $ error ($ e_{L^2} $)**: $ \frac{\|\hat{u}-u\|_{L^2(\Omega\times(0,T])}}{\|u\|_{L^2(\Omega\times(0,T])}} $，通过高精度数值积分计算。
  - **Relative $ L^2 $ test error ($ e_{\text{test}} $)**: $ \sqrt{\frac{\sum_k (\hat{u}(x_k,t_k)-u(x_k,t_k))^2}{\sum_k u(x_k,t_k)^2}} $，在均匀网格的测试点上计算。
- **训练细节**：先用Adam（学习率0.003）训练1000个epoch，再用L-BFGS微调。

### 基线方法对比
- **fPINN** [25]: 经典的物理信息神经网络。
- **MC-fPINN** [11]: 使用Monte Carlo采样的fPINN。
- **Improved MC-fPINN** [13]: 改进的Monte Carlo fPINN。
- **QE-MC-fPINN** [22]: 作者团队之前的工作，结合了Gauss-Jacobi径向积分和Monte Carlo角向采样。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1. **1D fPE (光滑解)**:
   - 表2显示，在 $ u=x(1-x)^3, \alpha=1.9 $ 时，fTNN的 $ e_{\text{test}} $ 达到 $ 1.38 \times 10^{-6} $，而fPINN仅为 $ 2.83 \times 10^{-4} $，精度高出两个数量级以上。

2. **1D fPE (奇异解)**:
   - 表4展示了在不同参数组合下的结果。当 $ s<0 $ 时，BRFE策略能保持误差在 $ 10^{-5} $ 量级；当 $ s\geq0 $ 时，BFE策略同样能达到 $ 10^{-7} $ 的高精度。

3. **2D/3D fPE (单位球)**:
   - 表9是核心对比结果。对于强奇异性解 $ u=(1-\|x\|^2)^{\alpha/2} $，在 $ d=2, \alpha=1.9 $ 时：
     - MC-fPINN: $ e_{\text{test}} = 7.20 \times 10^{-1} $
     - fPINN: $ e_{\text{test}} = 1.34 \times 10^{-2} $
     - QE-MC-fPINN: $ e_{\text{test}} = 6.05 \times 10^{-3} $
     - **fTNN**: $ e_{\text{test}} = \mathbf{1.72 \times 10^{-5}} $
   - fTNN的误差比次优的QE-MC-fPINN降低了**两个多数量级**。

4. **Time-dependent ADE (短时模拟)**:
   - 表10显示，在2D $ \gamma=0.5, \alpha=1.5 $ 时，fTNN ($ n_1=16 $) 的 $ e_{\text{test}} $ 为 $ 7.408 \times 10^{-5} $，而fPINN为 $ 1.241 \times 10^{-3} $，精度提高了一个数量级。

5. **Time-dependent ADE (长时模拟, T=100)**:
   - 表12显示，在3D长时模拟中，fTNN的 $ e_{\text{test}} $ 为 $ 9.486 \times 10^{-5} $，而QE-MC-fPINN为 $ 1.222 \times 10^{-3} $，fTNN依然保持显著优势。

### 消融实验结果
- **角分辨率 $ n_1 $ 的影响**（表7, 8, 图5）:
  - 实验系统地研究了角向离散化精度的影响。结果显示，随着 $ n_1 $ 增加，误差单调下降，且衰减速率符合球面积分理论预测的 $ O(n_1^{-1}) $。这证实了在径向奇异性被精确处理后，**角向离散化误差**是剩余误差的主要来源，凸显了使用确定性角向积分的有效性。
- **方法演进对比**:
  - 从MC-fPINN -> Improved MC-fPINN -> QE-MC-fPINN -> fTNN的演进过程，本质上是一个逐步消除误差源的过程：先是改进径向积分，最后彻底消除角向的Monte Carlo噪声。fTNN作为该系列工作的顶峰，实现了最高的确定性精度。

---

## 4. 关键结论和发现

### 主要发现
1. **确定性优于随机性**：在低至中等维度（d ≤ 3）的structured domains上，**fully deterministic integration framework** 能够提供远超Monte Carlo方法的精度和稳定性。
2. **奇异性需显式处理**：显式地在试函数中嵌入**boundary feature**并自适应选择指数（BFE/BRFE）是准确捕捉解的低正则性行为的关键。
3. **分离结构赋能长时模拟**：**STSNN**架构通过将高维时空积分解耦，使得在时间维度上使用密集的确定性积分成为可能，从而高效且准确地解决了长时非局部动力学问题。
4. **子空间优化更高效**：交替优化线性系数和网络参数的**alternating subspace optimization**策略，相比联合优化，能获得更好的收敛性和更快的训练速度。

### 方法的局限性
- **维度限制**：该方法依赖于确定性的角向积分，其计算成本随维度增加而急剧上升（curse of dimensionality）。因此，它主要适用于**low-to moderate-dimensional**问题（d=1,2,3），而不像Monte Carlo方法那样天然适合高维（d≥4）。
- **几何限制**：方法依赖于**structured geometries**（如超矩形、单位球）来进行有效的积分分割和坐标变换。对于非常复杂的不规则域，应用起来会更具挑战性。

### 未来工作方向
1. 将该方法扩展到**more general geometries**。
2. 探索**nonlinear fractional models**。
3. 开发**hybrid deterministic-stochastic strategies**，在保持当前方法高精度的同时，通过引入随机性来提高其在更高维度上的可扩展性（scalability）。

</details>

---

### 16. [NebulaExp-8B: An Empirical Post-Training Pipeline via Full-Scale Ablation Research](https://arxiv.org/abs/2606.26671)

**Authors**: Qiaobo Hao, Yangqian Wu, Shunyi Wang, Zhongjian Zhang, Ziqun Li, Yayin He, Muqing Li, Chen Zhong  
**Category**: cs.AI  
**Published**: 2026-06-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.26671v1  

#### Abstract
Post-training alignment determines the reasoning and human preference following capabilities of large language models, yet most existing works withhold detailed data construction, filtering rules and training recipes, which hinders community reproducibility and lightweight model optimization. This w...

---

### 17. [OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning](https://arxiv.org/abs/2606.26790)

**Authors**: Shuo Yang, Jinyang Wu, Zhengxi Lu, Yuhao Shen, Fan Zhang, Lang Feng, Shuai Zhang, Haoran Luo, Zheng Lian, Zhengqi Wen, Jianhua Tao  
**Category**: cs.CL  
**Published**: 2026-06-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.26790v1  

#### Abstract
Outcome-based reinforcement learning provides a stable optimization backbone for language agents, but its sparse trajectory-level rewards provide little guidance on which intermediate decisions should be reinforced or suppressed. On-policy self-distillation offers dense token-level supervision, yet ...

---

### 18. [PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on Commodity GPUs](https://arxiv.org/abs/2606.26666)

**Authors**: Muhammad Ahmed  
**Category**: cs.LG  
**Published**: 2026-06-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.26666v1  

#### Abstract
Autoregressive large language model (LLM) serving is increasingly limited by key-value (KV) cache movement rather than dense matrix multiplication. Modern paged-attention systems reduce KV-cache fragmentation and mature kernels such as FlashInfer provide highly optimized native-paged decode attentio...

---

### 19. [Kalman Prototypical Networks for Few-shot Fault Detection in Combined Cycle Gas Turbines](https://arxiv.org/abs/2606.26710)

**Authors**: Mohammed Ayalew Belay, Lucas Ferreira Bernardino, Adil Rasheed, Rub\'en M. Monta\~n\'es, Pierluigi Salvo Rossi  
**Category**: cs.AI  
**Published**: 2026-06-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.26710v1  

#### Abstract
Combined-cycle gas turbines (CCGTs) play a key role in modern power generation, offering both high efficiency and reduced environmental impact. However, their complex thermo-fluid and mechanical interactions complicate fault detection, particularly when labeled fault data are scarce. In this paper, ...

---

### 20. [Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement](https://arxiv.org/abs/2606.27226)

**Authors**: Sangwoo Cho, Kushal Chawla, Pengshan Cai, Zefang Liu, Chenyang Zhu, Shi-Xiong Zhang, Sambit Sahu  
**Category**: cs.AI  
**Published**: 2026-06-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.27226v1  

#### Abstract
Evaluating LLM outputs remains a major bottleneck in NLP: human evaluation is expensive and slow, lexical metrics correlate poorly with human judgments on open-ended generation, and holistic LLM judges often produce opaque scores that are hard to debug. We propose BINEVAL, a framework that decompose...

---

### 21. [Simulation-based inference for rapid Bayesian parameter estimation in epidemiological models: a comparison with MCMC](https://arxiv.org/abs/2606.27286)

**Authors**: Alina Bazarova, Johann Fredrik Jadebeck, Henrik Zunker, Carolina J. Klett-Tammen, Torben Heinsohn, Wolfgang Wiechert, Katharina Noeh, Stefan Kesselheim  
**Category**: cs.AI  
**Published**: 2026-06-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.27286v1  

#### Abstract
Mechanistic epidemiological models are widely used to support infectious disease forecasting and public-health decision making. Bayesian calibration of such models is commonly performed using Markov chain Monte Carlo (MCMC), which can become computationally expensive for high-dimensional nonlinear s...

---

### 22. [Mesh-RL: Coupled subgrid reinforcement learning](https://arxiv.org/abs/2606.26333)

**Authors**: Behnam Gheshlaghi, Bahador Rashidi, Shahin Atakishiyev  
**Category**: cs.LG  
**Published**: 2026-06-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.26333v1  

#### Abstract
Reinforcement learning in large or sparse-reward environments suffers from slow temporal-difference reward propagation, as value information spreads only locally across the state space. We propose Mesh-RL, a spatial domain-decomposition framework inspired by the finite element method and domain deco...

---

### 23. [A Multi-Fidelity Convolutional Autoencoder-Transfer Learning Framework for Guided-Wave-Based Damage Diagnosis Using Large Simulated and Limited Experimental Datasets](https://arxiv.org/abs/2606.27304)

**Authors**: Santosh Kapuria,  Abhishek  
**Category**: cs.LG  
**Published**: 2026-06-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.27304v1  

#### Abstract
Guided wave-based structural health monitoring (GWSHM) with onboard transducers offers significant potential for the early diagnosis of damage in engineering structures. However, the practical deployment of deep learning models is often hindered by the limited availability of labelled experimental d...

---

### 24. [Reinforcement Learning without Ground-Truth Solutions can Improve LLMs](https://arxiv.org/abs/2606.27369)

**Authors**: Yingyu Lin, Qiyue Gao, Nikki Lijing Kuang, Xunpeng Huang, Kun Zhou, Tongtong Liang, Zhewei Yao, Yi-An Ma, Yuxiong He  
**Category**: cs.LG  
**Published**: 2026-06-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.27369v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) for training LLMs typically rely on ground-truth answers to assign rewards, limiting their applicability to tasks where the ground-truth solution is unknown. We introduce a \textbf{R}anking-\textbf{i}nduced \textbf{VER}ifiable framework (RiVER) t...

---

### 25. [Generative Retrieval via Diffusion Transformer with Metric-Ordered Sequence Training and Hybrid-Policy Preference Optimization](https://arxiv.org/abs/2606.26899)

**Authors**: Chenghao Liu, Yu Zhang, Zhongtao Jiang, Kun Xu, Zhenwei An, Renzhi Wang, Zhao Wang, Jiachen Zhang, Yuxiao Zhang, Kun Xu, Songfang Huang  
**Category**: cs.AI  
**Published**: 2026-06-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.26899v1  

#### Abstract
Embedding-based retrieval ranks items by their similarity to a query in a shared vector space and usually aims to return the highest-scoring items. In many production settings this is not what is wanted: given a seed set that expresses a fine-grained pattern, one needs more items that both satisfy a...

---

### 26. [Semantic Early-Stopping for Iterative LLM Agent Loops](https://arxiv.org/abs/2606.27009)

**Authors**: Sahil Shrivastava  
**Category**: cs.AI  
**Published**: 2026-06-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.27009v1  

#### Abstract
Multi-agent large language model (LLM) loops, for example a Writer that drafts and a Critic that revises, are almost always terminated by a fixed iteration cap (max_iterations). This is a syntactic kill-switch: it is blind to whether the answer is still improving, so it over-spends tokens on easy in...

---

### 27. [Nemotron-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context](https://arxiv.org/abs/2606.26493)

**Authors**: Fitsum Reda, John Kamalu, Roger Waleffe, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro  
**Category**: cs.CL  
**Published**: 2026-06-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.26493v1  

#### Abstract
Diffusion language models offer a promising alternative to autoregressive models due to their potential for parallel and iterative generation. However, existing approaches use a single network for both context representation and iterative denoising, forcing one model to serve both roles and limiting...

---

### 28. [Cascaded Multi-Granularity Pruning for On-Device LLM Inference in Industrial IoT](https://arxiv.org/abs/2606.26861)

**Authors**: Jinghan Wang, Yanjun Chen, Wei Zhang, Xiaotong Huang, Tianchen Liu, Gaoliang Peng  
**Category**: cs.CL  
**Published**: 2026-06-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.26861v1  

#### Abstract
Deploying large language models (LLMs) on Industrial Internet of Things (IIoT) edge devices demands extreme compression, yet existing structured pruning methods collapse at high compression ratios due to one-shot importance estimation, and their cross-architecture behavior remains unpredictable. Thi...

---

### 29. [Moebius: Serving Mixture-of-Expert Models with Seamless Runtime Parallelism Switch](https://arxiv.org/abs/2606.26607)

**Authors**: Shaoyu Wang, Yizhuo Liang, Jaeyong Song, Chong Li, Seo Jin Park  
**Category**: cs.DC  
**Published**: 2026-06-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.26607v1  

#### Abstract
Mixture-of-Experts (MoE) architectures scale large language models (LLMs) to hundreds of billions of parameters. Serving a single MoE model requires multiple GPUs operating in parallel, typically through tensor parallelism (TP) or expert parallelism (EP). The optimal choice depends on the number of ...

---

### 30. [A General Framework for Learning Algebraic Properties from Cayley Graphs using Graph Neural Networks](https://arxiv.org/abs/2606.26212)

**Authors**: Tal Weissblat  
**Category**: cs.LG  
**Published**: 2026-06-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.26212v1  

#### Abstract
A Graph Neural Network (GNN) framework for predicting the solvability of finite groups from their Cayley graph representations was introduced in [1]. In the present work, we generalize this approach and develop a property-independent framework for learning algebraic properties of finite groups direc...

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

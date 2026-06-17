# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-17 10:23:12 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [AoiZora: Topology-Aware Auto-Parallel Optimization for Inference of Diffusion Transformers](https://arxiv.org/abs/2606.17566)

**Authors**: Kaijian Wang, Yuanyuan Xu, Fanjiang Ye, Ye Cao, Jingwei Zuo, T. S. Eugene Ng, Yarong Mu, Yuke Wang  
**Category**: cs.DC  
**Published**: 2026-06-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.17566v1  

#### Abstract
Video diffusion has quickly grown into a key generative serving workload, yet producing each clip demands many denoising iterations over large spatio-temporal latents, which puts low-latency inference out of reach on a single device. A denoising step is therefore typically distributed across multipl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AoiZora: Topology-Aware Auto-Parallel Optimization for Inference of Diffusion Transformers

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代视频生成模型（如基于 Diffusion Transformer 的 Wan、CogVideoX 等）在推理时面临巨大的计算和通信开销，尤其是在处理长时空序列（spatio-temporal tokens）时。尽管已有自动并行系统（auto-parallel systems）可将计算分布到多个加速器上，但它们通常只在**逻辑设备网格**（logical device mesh）层面进行优化，而忽略了物理 TPU 芯片间的互连拓扑结构（interconnect topology）。这导致即使相同的逻辑分片策略，在不同物理布局下也可能因链路争用（shared-link contention）产生高达 16.6% 的延迟差异。

因此，现有方法未能充分利用 TPU 子切片（sub-slice）的物理拓扑特性，造成性能次优。

### 提出了什么新方法或新思路
本文提出 **AoiZora**，一个面向 TPU 子切片的编译器中介型拓扑感知自动并行规划器（compiler-mediated topology planner），其核心思想是：

- 将**逻辑分片**（logical sharding）与**物理放置**（physical placement）解耦，并在编译流程的不同阶段分别优化。
- 利用“**编译阶段成本/保真度不对称性**”（cost/fidelity asymmetry）：
  - 在预编译阶段（pre-compilation），使用轻量级 IR（如 StableHLO 和 Shardy）快速筛选出潜在有效的逻辑分片方案；
  - 在后编译阶段（post-compilation），利用完整的 HLO 图获取真实的 collective 操作、payload 大小和 replica groups，结合物理拓扑模型对候选方案进行精确评分和排序。

AoiZora 不修改模型代码、不改动 XLA 编译流程、不替换 collective 内核或路由策略，而是通过标准 JAX/XLA 执行路径注入最优的 `sharding` 和 `placement` 配置。

### 相比现有方法的优势
| 维度 | 现有方法（如 Alpa-style） | AoiZora |
|------|--------------------------|--------|
| 分片粒度 | 仅逻辑分片 | 联合优化逻辑分片 + 物理放置 |
| 拓扑感知 | 否（假设均匀通信代价） | 是（建模真实 ICI 链路带宽与争用） |
| 成本模型保真度 | 抽象通信模型（abstract collective cost） | 基于编译后 HLO 的真实 collective 几何结构 |
| 实现侵入性 | 可能需自定义调度或内核 | 完全兼容现有栈（zero-change to runtime） |
| 性能提升 | 有限（忽略物理布局影响） | 显著降低延迟（最高达 1.42× 加速） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- 并非传统意义上的“数据集”，而是以 **Wan 2.1** 视频扩散模型作为代表性 workload。
- 支持多种分辨率与时长组合：**480p / 720p**，帧数为 **21 / 41 / 81 frames**。
- 模型结构为典型的 DiT（Diffusion Transformer），具有静态计算图和张量形状，适合离线规划。

### 实验设置和评估指标
#### 硬件平台
- Google TPU v5e 子切片：包括 **v5e-4**（4 chips）、**v5e-8**（8 chips）、**v5e-16**（16 chips）三种配置。
- 拓扑为二维 mesh 结构（无环形连接），链路带宽受物理距离和共享程度影响。

#### 评估指标
- **One-step denoising latency**：单步去噪延迟，为主要性能指标（占端到端延迟的 97.7%）。
- **Speedup**：相对于 baseline 的加速比。
- **Rank correlation (Spearman’s ρ)**：衡量预测得分与实测延迟的相关性。
- **Top-1 recall**：是否保留了实际最优的分片-放置组合。
- **Offline planning time**：规划耗时（不计入在线推理时间）。

#### 消融设计
- 是否启用 Post-Compilation Stage 的拓扑感知排名。
- 使用单 block vs. 多 block proxy 进行编译的影响。

### 基线方法对比
| 基线名称 | 描述 |
|--------|------|
| **Existing Deployment** | 基于开源实践（如 Wan/xDiT/SGLang）的手动调优方案，采用 FSDP、Sequence Parallelism、Tensor Parallelism 等常见策略，但物理布局固定（默认顺序） |
| **Logical-search baseline (Alpa-style)** | 受 Alpa 启发的自动化搜索方法，在内存约束下通过 ILP 优化抽象通信代价，但仍停留在逻辑分片层，未考虑物理拓扑 |

两者均未进行物理 placement 优化，构成 AoiZora 的前后参照。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| Workload | v5e-4 | v5e-8 | v5e-16 |
|---------|-------|-------|--------|
| **Geometric Mean Speedup vs. Existing** | 1.09× | 1.12× | **1.42×** |
| **Geometric Mean Speedup vs. Alpa-style** | 1.00×~1.07× | 1.06×~1.18× | **1.37×** |

> ✅ 最高实现 **1.42× 的单步去噪延迟降低**（在 v5e-16 上运行 480p21f 任务）

### 与基线方法的对比结果
- 在所有可行配置中，AoiZora **从不低于任何 baseline**（within measurement precision）。
- 在较小规模（v5e-4）上改进有限，因为合法分片空间小且拓扑自由度低。
- 在更大规模（v5e-8 和 v5e-16）上优势明显，说明拓扑感知在复杂拓扑中有更大优化空间。
- Alpa-style 方法在某些情况下失败（如 v5e-4 上无法满足 HBM 内存限制），而 AoiZora 能有效规避此类不可行方案。

### 消融实验结果
#### （1）禁用 Post-Compilation Stage（仅依赖 Pre-Compile 排名）
| Workload | Gap vs Full Pipeline |
|--------|---------------------|
| 480p21f | 45.2–81.9% 更高延迟 |
| 720p81f | 2.7–10.8% 更高延迟 |

👉 表明：**仅靠预编译信号不足以选出最终最优方案**，必须结合编译后的 collective 信息。

#### （2）使用单 block proxy 替代双 block proxy
- 单 block 导致跨块通信和流水线重叠信息丢失。
- Placement ranking 的 Spearman’s ρ 最高仅为 **0.371**，多数接近 0。
- 而双 block proxy 达到 **0.817–0.985**，显著更准确。

👉 表明：**需要足够长的计算图片段来暴露真实的通信模式和依赖关系**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **物理放置显著影响性能**：相同逻辑分片策略下，不同物理布局可导致 **最高 16.6% 的延迟差异**，主因是 shared-link contention 而非 hop count。
2. **编译流程存在天然的成本-保真度权衡**：
   - Pre-compilation IR（StableHLO/Shardy）可用于高效 pruning；
   - Compiled HLO 才能提供用于拓扑感知 ranking 的真实 collective 图。
3. **两阶段分离设计是实用且高效的**：
   - 先广度搜索 + 快速剪枝；
   - 再深度评估 + 精确打分；
   - 总体规划时间可控（v5e-16 上约 6 分钟），远低于穷举搜索。
4. **AoiZora 可无缝集成现有系统**：无需修改模型、编译器或运行时，只需装饰函数即可生效。

### 方法的局限性
- 当前适用于 **静态图、固定形状请求** 的场景（如视频生成），对动态 shape 或 MoE 架构支持不足。
- 依赖于 TPU 子切片的规则矩形拓扑；对于非连续或 irregular 分配的支持尚未验证。
- 规划过程仍有一定开销（数百秒），不适合频繁变更的部署环境（但可通过缓存缓解）。
- 当前仅针对 DiT 类模型验证，虽具代表性，但未覆盖所有视频生成架构变种。

### 未来工作方向
- 扩展至 **MoE-style DiT** 或其他动态负载模型。
- 支持 **异构设备集群** 或 **多租户并发流量** 下的联合拓扑优化。
- 探索 **在线自适应 placement**，应对运行时变化（如链路降级）。
- 将框架推广至 **TPU v6e（2D）和 v7x（3D torus）** 等新型拓扑。
- 与 job scheduler 联合优化，实现从作业分配到 placement 的端到端协同设计。

---

> 📌 **一句话总结**：  
> AoiZora 通过在编译流程的关键节点引入拓扑感知，实现了逻辑分片与物理放置的联合优化，在不改变现有 ML 栈的前提下，为 Diffusion Transformer 的低延迟推理提供了高达 **1.42× 的加速**，揭示了“placement is part of the parallel execution plan”的重要理念。

</details>

---

### 2. [SpecGen: Accelerating Agentic Kernel Optimization with Speculative Generation](https://arxiv.org/abs/2606.17518)

**Authors**: Jihu Guo, Sitian Lu, Tenghui Ma, Wei Gao, Zhisheng Ye, Xingcheng Zhang, Dahua Lin  
**Category**: cs.DC  
**Published**: 2026-06-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.17518v1  

#### Abstract
Agentic kernel optimization automates manual GPU kernel tuning via iterative generation, validation, and profiling with reasoning LLMs, casting the optimization task as feedback-guided search. However, our workload characterization reveals three system-level inefficiencies that limit search efficien...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SpecGen: Accelerating Agentic Kernel Optimization with Speculative Generation**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前基于 **agentic kernel optimization** 的自动化 GPU 内核优化系统存在三大系统级效率瓶颈：
1. **长生成延迟（Long generation latency）**：LLM 的推理生成阶段占迭代时间的 70–99%，严重限制了在固定时间预算下的搜索次数。
2. **不足的性能反馈（Insufficient profiling feedback）**：约 60% 的迭代因验证失败而无法进入性能分析阶段，导致缺乏有效的优化指导。
3. **验证/分析资源利用率低（Underutilized validation/profiling resources）**：由于候选内核仅在生成完成后才出现，验证和分析 GPU 大部分时间处于空闲状态，平均利用率仅为 4.2–17.6%。

### **提出的新方法：SpecGen**
为解决上述问题，论文提出了 **SpecGen**，一种结合 **speculative generation（推测式生成）** 的高效内核优化框架，其核心思想是：
> 在主推理生成（reasoning generation）进行过程中，利用其输出前缀（reasoning prefix）并行地启动多个非推理的 **speculative generation**，提前生成候选内核。

#### **关键技术组件**
- **SpecController**：
  - 监控主生成流，识别触发信号（如代码块、设计决策等），从当前前缀派生出 speculative prompt。
  - 动态分叉 K 个非推理生成任务，增加候选数量。
  - 当某个推测内核满足终止条件（如速度超过历史平均值）时，**提前终止主生成**，显著降低延迟。
  
- **ElasticScheduler**：
  - 将验证和分析 GPU 组成一个弹性池，动态分配资源以应对突发负载。
  - 验证请求采用 **LAF（Last-Arrival-First）** 调度，优先处理更新的、更可能成功的候选。
  - 分析请求采用 **FIFO** 调度，确保最快返回反馈。
  - 利用验证/分析 GPU 的闲置内存作为远程 **KV cache** 存储，避免重复计算前缀。

### **相比现有方法的优势**
- **不依赖模型微调**：完全在运行时系统层面优化，无需修改 LLM 或训练过程。
- **正交于已有算法**：可与 multi-agent、evolutionary search 等搜索策略结合。
- **实现端到端加速**：通过早期终止和并行生成，直接缩短最长路径（critical path）。
- **提升反馈密度和资源利用率**：显著增加成功进入 profiling 的内核数量，并保持 GPU 高负载。

---

## **2. 核心实验方法和设置**

### **数据集**
- **KernelBench** 基准测试中的 20 个任务：
  - **Level 1/2/3 共 20 个任务**（T1–T20）
  - 包括多种矩阵乘法（Matmul）、卷积（Conv2d）、激活函数组合（GELU, ReLU）等典型 LLM 算子。
  - 示例：`T2: 3D tensor Matmul`, `T19: MLP (Level 3)`, `T20: ReLU Self-Attn (Level 3)`。

### **实验设置**
- **硬件平台**：H200 GPU 集群，最多使用 18 张 H200。
- **LLM 模型**：
  - **GLM-5.1**（本地部署 via vLLM）
  - **DeepSeek-V4-Pro**（通过官方 API 调用）
- **每任务预算**：100 次迭代。
- **评估方式**：每个内核性能取 40 次运行的平均值（10 次预热）。

### **评估指标**
| 指标 | 描述 |
|------|------|
| **E2E 执行时间** | 完成 100 次迭代的总耗时 |
| **Profiling Feedback 数量** | 成功完成性能分析的有效内核数 |
| **Resource Utilization** | 验证/分析 GPU 的平均利用率 |
| **Final Kernel Speedup** | 最终找到的最佳内核相对于参考实现的速度提升倍数 |
| **Token Consumption** | 总生成 token 数量 |

### **基线方法对比**
1. **CudaForge**：基于多智能体的 Coder-Judge 框架，依赖 NCU 反馈。
2. **AlphaEvolve**：Google DeepMind 提出的进化算法驱动的编码代理。
3. **KernelAgent**：Meta 提出的自主内核生成系统，结合并行生成与硬件引导优化。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**
| 指标 | 提升幅度 |
|------|----------|
| **E2E 时间减少** | **1.68–1.82×** 快于基线 |
| **Profiling Feedback 增加** | **1.58–1.98×** 更多有效反馈 |
| **资源利用率提升** | 从 **4.2–17.6% → 88.2–96.1%** |
| **最终内核加速比提升** | **1.24–1.91×** 更高性能内核 |
| **Token 开销** | 仅增加 **~30%**，部分任务甚至更低（因早停节省） |

### **与基线方法的详细对比**
- **E2E 时间**（图 10）：
  - 在 GLM-5.1 上平均快 **1.68×**（vs CudaForge）至 **1.81×**（vs AlphaEvolve）。
  - 在 DeepSeek-V4-Pro 上最高达 **1.96×** 加速。
- **Profiling Feedback**（图 11）：
  - SpecGen 平均产生 **66–70 个** 成功分析的内核，远超基线的 **36–49 个**。
  - 几何平均提升 **1.58–1.98×**。
- **资源利用率**（表 4）：
  - 基线系统利用率极低（4.2–17.6%），SpecGen 达到 **88.2–96.1%**，接近饱和。
- **最终内核性能**（表 6）：
  - 在所有 10 个任务上均找到更快内核。
  - 几何平均速度提升达 **5.78×（GLM）** 和 **3.49×（DeepSeek）**，优于所有基线。

### **消融实验结果**（表 5）
| 配置 | E2E 时间加速比（vs CudaForge） | 贡献增量 |
|------|-------------------------------|---------|
| Baseline | 1.00× | — |
| + Speculative Generation | 1.46× | +0.46 |
| + Resource Reallocation | 1.61× | +0.15 |
| + Priority Queue | 1.69× | +0.08 |
| + Remote Prefix Cache | 1.77× | +0.09 |
- **Speculative Generation 是最大贡献者**，其余组件协同优化调度与缓存效率。

---

## **4. 关键结论和发现**

### **主要发现**
1. **推理过程存在“窗口期”可用于并行生成**：
   - 即使在主推理未完成时，其前缀已包含足够信息用于生成有效内核。
   - 前缀条件化（prefix conditioning）显著提升非推理生成的质量。

2. **早期终止可行且高效**：
   - 实验表明，在推理完成前 **14–70%** 的位置即可生成优于历史平均的内核。
   - 使用 **mean(H)** 作为阈值可在性能与延迟间取得良好平衡。

3. **bursty 负载需弹性调度**：
   - 静态 “one GPU per kernel” 架构无法应对 speculative generation 的突发请求。
   - ElasticScheduler 通过动态资源再分配和优先级队列，有效吸收负载波动。

4. **系统优化可独立于算法改进**：
   - SpecGen 不改变搜索逻辑，而是优化执行流程，因此可与任何 agent 算法叠加增益。

### **局限性**
- **依赖高质量前缀解析**：若 LLM 输出格式混乱或无明确触发信号，SpecController 可能错过最佳分叉时机。
- **对非常短的生成任务收益有限**：若原生成时间很短，则 speculative generation 的边际效益下降。
- **远程 KV cache 依赖高速互联**：RDMA 和 NVLink 是实现高效迁移的前提，在普通网络下可能成为瓶颈。

### **未来工作方向**
- **自适应分叉数量控制**：根据实时队列压力和成功率动态调整 K。
- **跨任务前缀迁移学习**：将一个任务中学到的有效前缀模式迁移到相似任务中。
- **集成 token-level speculative decoding**：与 Medusa 等技术结合，进一步加速单次生成。
- **扩展至其他编译优化场景**：如自动向量化、调度空间探索等。

---

> ✅ **总结一句话**：  
> **SpecGen 通过在推理过程中“投机式”地并行生成候选内核，并配合弹性调度与早期终止机制，在不牺牲最终性能的前提下，实现了高达 1.8× 的端到端加速和近 10 倍的资源利用率提升，为 agentic kernel optimization 提供了一种高效的系统级解决方案。**

</details>

---

### 3. [Amortized Probabilistic Retrieval of Atmospheric CO2 from OCO-2 Spectra Using Deep Learning with Laplace Approximations and Normalizing Flows](https://arxiv.org/abs/2606.17413)

**Authors**: Alejandro Calle-Saldarriaga, Felix Jimenez, Jack Grosskreuz, Jiazheng Wang, Jonathan Hobbs, Matthias Katzfuss  
**Category**: cs.LG  
**Published**: 2026-06-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.17413v1  

#### Abstract
Space-based monitoring of atmospheric carbon dioxide (CO2) is essential for constraining the global carbon budget. NASA's Orbiting Carbon Observatory-2 (OCO-2) estimates column-averaged dry-air mole fractions of CO2 (XCO2) using high-resolution spectra. However, current operational retrieval algorit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Amortized Probabilistic Retrieval of Atmospheric CO₂ from OCO-2 Spectra Using Deep Learning with Laplace Approximations and Normalizing Flows

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前用于从 NASA 的 OCO-2 卫星光谱中反演大气 CO₂ 浓度（XCO₂）的“全物理”（full-physics）反演算法存在三大瓶颈：
- **计算成本高**：基于迭代优化的 Optimal Estimation（OE）方法每条 sounding 需要数秒至数十秒；
- **不确定性量化（UQ）不足**：假设后验分布为高斯分布，忽略了非线性效应导致的非对称、多峰等复杂结构；
- **忽略前向模型误差（model discrepancy）**：标准反演假设前向模型 $ F(x) $ 完美，而实际中存在系统性偏差（如吸收系数不准确、气溶胶散射近似误差等），导致估计偏差。

### 提出的新方法与思路
本文提出一种**基于深度学习的摊销概率反演框架**（amortized probabilistic retrieval），结合两种先进的不确定性建模技术：
- **Laplace Approximation**：在训练好的神经网络最后一层上构建局部高斯后验，捕捉参数不确定性（epistemic uncertainty）；
- **Normalizing Flows**：通过可逆变换建模复杂的、潜在非高斯的条件后验分布 $ p(x|y) $，突破传统高斯假设限制。

该框架利用**高保真模拟数据集**进行训练，其中显式包含了经过校准的前向模型误差 $ \delta $，使模型能隐式学习并纠正这些系统性偏差。

### 相比现有方法的优势
| 优势维度 | 具体表现 |
|--------|--------|
| **计算效率** | 推理速度提升数个数量级（毫秒级 vs 数百秒级），支持实时处理大规模数据流（amortization） |
| **鲁棒性** | 在训练中引入 model discrepancy，提升了对系统性物理误差的鲁棒性，优于忽略 $ \delta $ 的传统方法 |
| **点估计精度** | RMSE 显著低于 NASA ACOS L2 全物理反演，在 XCO₂ 和完整剖面预测中均取得更优结果 |
| **不确定性量化（UQ）质量** | 产生的预测区间具有更好的**校准性**（calibration），经验覆盖率接近名义水平（如 95% 区间覆盖约 95% 真值） |
| **后验表达能力** | 利用 Normalizing Flows 成功捕捉到**非高斯、不对称的后验结构**，揭示了传统方法无法表示的物理模糊性 |

---

## 2. 核心实验方法和设置

### 数据集
- **数据来源**：JPL 开发的 OCO-2 不确定性量化（UQ）模拟数据集（Braverman et al., 2021; Hobbs et al., 2020）
- **数据特点**：
  - 包含 47,900 组合成光谱-状态对 $(y, x)$
  - 显式建模了前向模型误差 $ \delta \neq 0 $，反映真实世界中的系统偏差
  - 覆盖 9 个 geophysical clusters（不同地表反照率、气溶胶类型、纬度等）
  - 每个参考 sounding 生成 100 次随机扰动样本，增强泛化性
- **输入特征**：
  - 三个 OCO-2 光谱波段：O₂ A-band (0.76 μm), Weak CO₂ band (1.61 μm), Strong CO₂ band (2.06 μm)
  - 辅助变量：Solar Zenith Angle (SZA)、卫星视角、先验表面压力、五类气溶胶类型（DU, SO, SS, OC, BC）

### 实验设置
- **训练/验证/测试划分**：80%/10%/10%，按 cluster 分层抽样，确保地理代表性
- **预处理**：光谱强度标准化；目标变量（如 XCO₂）标准化；分类变量整数编码
- **模型架构**：
  - 多分支结构：三个独立的 1D CNN 编码器分别处理各光谱带
  - 辅助变量单独编码后融合
  - 使用 Transformer 注意力模块实现跨模态融合
  - 最终接任务特定的 MLP 或 Normalizing Flow 头部

### 评估指标
| 指标 | 描述 |
|------|------|
| **RMSE** | 衡量点估计准确性，单位 ppm |
| **Negative Log-Likelihood (NLL)** | 综合评价点估计与不确定性质量，越低越好 |
| **Empirical Coverage** | 名义置信水平下真实值落入预测区间的比例（如 95% 区间应覆盖 ~95% 样本） |
| **PIT Histograms** | 概率积分变换直方图，均匀分布表示良好校准 |
| **Coverage Curves** | 不同名义置信水平下的实证覆盖率曲线，理想为对角线 |

### 基线方法对比
- **NASA ACOS L2 Full Physics Retrieval**：当前业务化反演算法，基于 Optimal Estimation，迭代求解，输出高斯后验
- **Deterministic Neural Network**：仅提供点估计的基准
- **Laplace Approximation on NN**：提供高斯后验的概率扩展
- **Conditional Normalizing Flow (NF)**：提供灵活、可能非高斯的后验建模

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### ✅ XCO₂ 标量预测（Table 1）
| 方法 | RMSE (ppm) | NLL | Emp. Cov. (95%/68%) |
|------|------------|-----|---------------------|
| NASA Retrieval | 2.85 | 19.59 | 21.9% / 11.2% |
| Laplace | **1.53** | -1.12 | **98.1% / 82.2%** |
| Normalizing Flow | 1.55 | **-1.25** | 95.6% / 69.5% |

> 📌 **解读**：  
> - 所有 DL 方法显著优于 NASA 方法（RMSE ↓50%, NLL ↓90%+）
> - Laplace 在 RMSE 上最优，NF 在 NLL 上最优，表明其概率建模更优
> - NASA 方法严重**欠覆盖**（undercoverage），说明其不确定性被严重低估

#### ✅ 完整 20 层 CO₂ 剖面预测（Table 2）
| 方法 | Avg. RMSE (ppm) | Joint NLL | Emp. Cov. (95%/68%) |
|------|------------------|-----------|-----------------------|
| NASA Retrieval | 5.23 | 45.20 | 89.6% / 64.2% |
| Laplace | **4.71** | 22.90 | 94.7% / 73.2% |
| Normalizing Flow | 4.82 | **-40.63** | 90.6% / 63.1% |

> 📌 **解读**：
> - NF 在联合 NLL 上大幅领先，说明其能更好建模剖面间的依赖结构
> - Laplace 略有过覆盖（over-coverage），NF 更接近理想校准

#### ✅ 近地面四层预测（Table 3）
| 方法 | RMSE (ppm) | Joint NLL | Cov. (95%/68%) |
|------|------------|-----------|-----------------|
| NF (Native 4D) | **8.52** | **-7.28** | 96.2% / 70.5% |
| NF (Marginal from 20D) | 10.32 | -0.52 | 91.4% / 64.4% |

> 📌 **关键发现**：
> - **直接训练于目标子空间（如 near-surface）的 NF 性能远优于从完整剖面边际化而来的方法**
> - 强调“**任务定制化建模**”的重要性：若关注特定层次，应直接训练对应输出维度

### 消融实验与分析
- **是否使用 Normalizing Flow 影响显著**：
  - NF 能捕捉到明显的非椭圆、弯曲甚至多模态的双变量后验结构（见 Figure 7），而 Laplace 和 NASA 输出始终为椭圆形高斯置信域
  - 尽管某些情况下 Gaussian 方法也能覆盖真值，但其形状和尺度常不合理（如过度扩散或收缩）
- **cluster-dependent 后验几何变化**：
  - 尽管未显式输入 cluster 标签，NF 学会根据不同 geophysical regime 输出不同的后验形态，说明其学到的是可观测光谱特征驱动的真实物理模糊性

---

## 4. 关键结论和发现

### 主要发现
1. **摊销推理是可行且高效的路径**：通过离线训练换取在线毫秒级推理，解决了传统反演的计算瓶颈。
2. **模拟数据可用于训练高质量反演模型**：只要模拟足够忠实于现实（尤其是包含 model discrepancy），即可训练出超越传统物理方法的模型。
3. **Normalizing Flows 可有效建模非高斯后验**：首次在 OCO-2 反演中成功展示复杂后验结构的存在，并证明其科学意义——它反映了真实的物理歧义（如气溶胶-路径长混淆）。
4. **不确定性校准至关重要**：NASA 方法虽有一定精度，但其置信区间严重失准；而本文方法（尤其 NF）实现了接近理想的 coverage，增强了下游应用（如碳通量反演）的可靠性。
5. **任务导向设计优于通用建模**：针对特定科学目标（如近地表浓度）设计专用模型，比从全局剖面提取边际更优。

### 方法的局限性
- **合成差距（Synthetic Gap）**：目前完全依赖模拟数据训练，尚未在真实 TCCON 观测上验证泛化能力。
- **领域适应需求**：真实仪器噪声、季节性变化可能导致分布偏移，需进一步 domain adaptation。
- **一次性训练成本高**：尽管推理快，但 Normalizing Flow 的训练耗时较长（GPU 数天），需要大量资源投入。

### 未来工作方向
1. **迁移学习 + TCCON 微调**：将在模拟数据上预训练的模型迁移到真实共址观测（coincident TCCON-OCO-2）上进行 fine-tuning。
2. **持续验证与动态更新**：建立长期监控机制，确保 learned bias correction 在时空变化下仍有效。
3. **扩展至其他气体与传感器**：将本框架推广至 CH₄、CO 等痕量气体，以及 OCO-3、GOSAT、Sentinel-5P 等平台。
4. **集成至 operational pipeline**：推动该方法成为下一代业务化处理系统的核心组件，实现“fast, accurate, and honest” retrieval。

---

> 🔚 **总结一句话**：  
> 本文展示了**基于模拟数据的深度学习 + 概率建模**（Laplace + Normalizing Flows）是一条通往**快速、准确、可信**的大气温室气体遥感反演的新范式，有望重塑未来卫星数据处理体系。

</details>

---

### 4. [Expanding SPHERE-JEPA: A Family of Statistical Regularizers for the Hypersphere](https://arxiv.org/abs/2606.17603)

**Authors**: L\'eo Nicollier (CB, ATT), Enric Meinhardt-Llopis (CB), Max Dunitz (ATT), Marc Pic (ATT), Pablo Mus\'e (CB, IFUMI), Gabriele Facciolo (CB)  
**Category**: cs.LG  
**Published**: 2026-06-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.17603v1  

#### Abstract
In Self-Supervised Learning (SSL), preventing representation collapse by explicitly enforcing a uniform distribution on the unit hypersphere has proven to be effective. However, current frameworks typically rely on sliced statistical regularizers such as SIGReg (used in LeJEPA) and SUSReg (used in S...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Expanding SPHERE-JEPA: A Family of Statistical Regularizers for the Hypersphere*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Self-Supervised Learning (SSL)** 中，为防止表示坍缩（representation collapse），通常通过正则化手段强制特征分布在单位超球面 $ \mathcal{S}^{d-1} $ 上趋于均匀分布。然而，当前主流方法（如 SUSReg）依赖于**切片统计正则化**（sliced statistical regularizers），即沿随机一维方向进行投影并计算统计量（如 Epps-Pulley 测试）。这种基于 **Monte Carlo 采样**的方法引入了“**投影方差**”（projection variance），导致训练梯度不稳定、收敛缓慢。

该论文旨在解决这一由随机投影带来的优化噪声问题，并探索更优的高维统计测试方法来直接在流形上实现均匀性约束。

---

### 🚀 提出的新方法与新思路

1. **确定性 MMD 正则化（Deterministic MMD）**  
   首次证明：对 SUSReg 所用的随机 1D 投影进行**解析积分**后，其期望等价于一个闭式（closed-form）的 **Maximum Mean Discrepancy (MMD)**，且该 MMD 使用的是一个由投影诱导的旋转不变核（rotationally invariant kernel）。这使得可以完全移除 Monte Carlo 投影，转而使用确定性的全维度目标函数。

2. **构建完整的超球面统计正则化家族**  
   在此基础上，提出一组新的、直接定义在超球面流形上的精确多变量统计测试作为正则项：
   - **MMD**（Maximum Mean Discrepancy）
   - **KSD**（Kernel Stein Discrepancy）
   - **KL Divergence**（基于 Kernel Density Estimation）

3. **谱域核设计（Spectral Kernel Design）**  
   为避免空间偏差，所有核均设计为**旋转不变**（zonal），并通过**谱理论**系统构造两类典型滤波器：
   - **Heat Kernel**：指数衰减频率权重，平滑多尺度相似性
   - **Bandlimited Kernel**：硬性低通滤波，仅保留前 $ L $ 个本征模态

4. **几何感知的正则化选择机制**  
   揭示不同统计测试会塑造不同的潜在空间几何结构：
   - MMD 和 KSD 容忍局部聚类 → 适合对象中心任务（object-centric）
   - KL 强制细粒度实例分离 → 适合无簇连续纹理检索

---

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **优化稳定性** | 消除投影方差 → 更稳定的梯度，更快收敛（见 Figure 2） |
| **计算效率** | 当 batch size $ B < |\mathcal{A}| $（通常 1024）时，$ O(B^2) $ 的 exact test 比 $ O(B|\mathcal{A}|) $ 的 sliced 方法更快 |
| **表达能力** | 支持多种统计测试，可根据下游任务拓扑灵活选择 |
| **理论严谨性** | 统一框架下推导出闭式目标，具备明确的几何解释 |

---

## 2. 核心实验方法和设置

### 📚 数据集

| 数据集 | 类型 | 特点 |
|-------|------|------|
| **ImageNet-100** | 图像分类 | 包含 100 个语义类别，适合评估类别级语义分组能力 |
| **Galaxy10** | 天文图像分类 | 星系形态分类，具有较强视觉多样性 |
| **Procedural Textures**（Disk, Cloud, Flake, Wood） | 连续纹理检索 | 无离散类别，强调实例级区分能力，用于非参数检索任务 |

---

### ⚙️ 实验设置

- **模型架构**：
  - ImageNet-100：ResNet-18
  - Galaxy10：ResNet-50
- **输出维度**：投影头输出至 $ \mathcal{S}^{255} $
- **Batch Size**：全局 batch size 固定为 256（确保统计一致性）
- **正则化系数 $ \lambda $**：
  - MMD/KSD: $ \lambda = 0.05 $
  - KL: $ \lambda = 0.5 $（类比 InfoNCE 结构）
- **温度参数 $ t $**：
  - MMD/KSD 使用 Heat Kernel: $ t = 5/d $
  - KL 使用 KDE: $ t = 2/d $

---

### 📊 评估指标

| 任务 | 指标 |
|-----|------|
| 分类任务（ImageNet/Galaxy10） | Linear probing accuracy, k-NN accuracy |
| 纹理检索任务 | Recall@1, Recall@3, Recall@5, mAP, mAP (emb)（在骨干网络嵌入上评估） |

---

### 🆚 基线方法对比

| 方法 | 类型 | 说明 |
|------|------|------|
| **SUSReg (Stochastic Baseline)** | 切片法基准 | SPHERE-JEPA 中使用的随机投影 + EP 测试 |
| **MMD (Induced SUSReg k)** | 确定性 MMD | 解析积分得到的等效 MMD，验证去噪效果 |
| **MMD/KSD with Heat/Bandlimited Kernels** | 新提出方法 | 不同谱核下的确定性正则化器 |
| **KL (Heat)** | 新提出方法 | 基于 KDE 的连续 KL 正则化 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & Table 2）

#### ✅ 分类任务表现（ImageNet-100 / Galaxy10）

| 方法 | ImageNet-100 (Linear) | ImageNet-100 (k-NN) | Galaxy10 (Linear) | Galaxy10 (k-NN) |
|------|------------------------|----------------------|--------------------|------------------|
| SUSReg (Baseline) | 71.22 ± 0.58 | 65.67 ± 0.56 | 72.13 ± 1.39 | 68.67 ± 1.04 |
| MMD (Bandlimited, L=2) | **72.26 ± 0.44** | **67.25 ± 0.43** | 76.21 ± 0.49 | 72.73 ± 1.00 |
| KSD (Bandlimited, L=2) | 72.19 ± 0.38 | 66.91 ± 0.14 | **77.21 ± 0.55** | **72.70 ± 0.77** |
| KL (Heat) | 67.72 ± 0.21 | 62.09 ± 0.42 | 75.76 ± 0.99 | 70.29 ± 0.85 |

> 💡 **结论**：MMD 和 KSD 变体在分类任务上一致优于 SUSReg，最高提升达 **+1.3% (Linear)** 和 **+1.6% (k-NN)**；KL 表现较差，因其抑制了必要的局部聚类。

---

#### ✅ 纹理检索任务表现（平均 across 4 procedural datasets）

| 方法 | Recall@1 | Recall@3 | Recall@5 | mAP | mAP (emb) |
|------|----------|----------|----------|-----|-----------|
| SUSReg | 88.7 | 96.1 | 97.7 | 92.6 | 91.4 |
| MMD (Heat) | 91.4 | 97.0 | 98.3 | 94.4 | 93.2 |
| KSD (Heat) | 92.0 | 97.1 | 98.2 | 94.8 | 93.5 |
| **KL (Heat)** | **95.3** | **97.6** | **98.3** | **96.7** | **96.3** |

> 💡 **结论**：在无簇连续纹理任务中，**KL (Heat)** 实现压倒性优势，Recall@1 达 **95.3%**，相比基线提升 **+6.6 pts**，显著优于其他方法。

---

### 🔬 消融实验结果

#### （1）投影噪声的影响（Figure 1 & 2）
- **方差分析**：SUSReg 存在明显投影方差，随 batch size 增大下降缓慢；而 MMD (induced) 完全消除该噪声。
- **训练动态**：MMD 收敛速度显著快于 SUSReg，在 ImageNet-100 上提前约 50 个 epoch 达到稳定性能。

#### （2）温度敏感性分析（Appendix E, Table 3）
- 对于 MMD：$ t = 5/d $ 是最优操作点
- 对于 KSD：$ t = 4/d $ 导致严重不稳定性（标准差高达 ±35%），推荐 $ t = 5/d $

#### （3）核类型比较
- 在分类任务中，**Bandlimited (L=2)** 略优于 Heat Kernel 和 Induced Kernel
- 在纹理检索中，Heat Kernel 更适合 KL 的密度估计需求

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **切片方法是次优的**  
   随机投影引入不必要的方差，可通过解析积分转化为确定性 MMD，从而获得更稳定高效的训练。

2. **精确的全维度统计测试更优**  
   MMD 和 KSD 在标准分类任务上全面超越 SUSReg，表明去除投影噪声可带来一致收益。

3. **统计测试的选择决定潜在空间几何**  
   - **MMD / KSD**：允许局部聚集 → 有利于类别划分
   - **KL (via KDE)**：强制全局均匀排斥 → 有利于连续实例区分

4. **没有“万能”的正则化器**  
   应根据任务拓扑选择合适的统计测试：
   - 对象中心任务 → 推荐 MMD 或 KSD
   - 连续纹理/实例检索 → 推荐 KL-based 正则化

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **计算复杂度为 $ O(B^2) $** | 虽然在中小 batch 下更快，但在大规模分布式训练（如 $ B > 4096 $）时，二次复杂度成为瓶颈 |
| **KL 依赖 KDE 近似** | 尽管比最近邻熵估计更平滑，但仍是对连续密度的近似，存在偏差风险 |
| **核设计仍需人工调参** | 如 Bandlimited 的截断阶数 $ L $、Heat Kernel 的温度 $ t $，尚无自适应选择机制 |

---

### 🔮 未来工作方向

1. **开发线性复杂度的近似算法**  
   设计 Nyström-type 或随机傅里叶特征近似，使 exact tests 可扩展至超大 batch 场景。

2. **自动化核选择机制**  
   引入可学习的谱权重 $ w(\lambda_l) $，让模型自动适应数据拓扑。

3. **推广至其他流形**  
   将该框架拓展至 Stiefel 流形、Grassmann 流形等更复杂的结构化表示空间。

4. **结合生成建模**  
   利用这些正则化器指导扩散模型或 VAE 在球面上生成均匀分布的隐变量。

---

> **总结一句话**：本文提出了一个统一、确定性、几何感知的超球面正则化家族，揭示了统计测试不仅是工具，更是塑造表示几何的“雕刻刀”，为 SSL 中的表示学习提供了新的理论视角与实践路径。

</details>

---

### 5. [From Brewing to Resolution: Tracing the Internal Lifecycle of Code Reasoning in LLMs](https://arxiv.org/abs/2606.17648)

**Authors**: Siyue Chen, Yifu Guo, Yuquan Lu, Zishan Xu, Jiaye Lin, Jianbo Lin, Siyu Zhang, Cheng Yang, Junxin Li, Yujia Li, Yu Huo, Ruixuan Wang  
**Category**: cs.AI  
**Published**: 2026-06-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.17648v1  

#### Abstract
Standard accuracy metrics cannot explain why LLMs handle variable tracking but fail on semantically equivalent loops. We study an internal lifecycle of code reasoning in which models first brew the answer, making it linearly recoverable many layers before it becomes self-decodable, and then diverge ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Brewing to Resolution: Tracing the Internal Lifecycle of Code Reasoning in LLMs

## 1. 论文的主要贡献和创新点

### 解决了什么问题
标准准确率（accuracy）指标无法解释大语言模型（LLMs）在代码推理任务中的内部失败机制。例如，一个模型可能轻松追踪变量赋值，但在处理语义上等价的循环展开时却表现不佳。这种差异在传统评估中被掩盖，导致无法理解模型为何失败。

本文旨在揭示LLMs进行代码推理的**内部生命周期**，区分“答案是否被编码”与“模型是否能利用该信息”的不同阶段，从而诊断出表面准确率无法捕捉的根本性失败模式。

### 提出了什么新方法或新思路
论文提出了一个名为 **“从酝酿到解决”（Brewing-to-Resolution）** 的内部生命周期框架，并引入了一套**双重诊断框架**来追踪这一过程：

1.  **Layer-wise Linear Probing (逐层线性探针)**：用于检测答案信息何时在隐藏状态中变得**线性可恢复**（information availability），即外部观察者能否读取答案。
2.  **Context-Stripped Decoding (CSD, 上下文剥离解码)**：通过将源提示的隐藏状态注入到仅包含问题后缀的目标提示中，并减去语言先验，来检测模型自身何时能**自解码**（self-decodable）该信息（information readiness）。

通过比较这两个信号，作者定义了四个**分辨率结果**（resolution outcomes）：
- **Resolved (已解决)**：模型正确计算并输出了答案。
- **Overprocessed (过度处理)**：模型曾正确计算出答案，但后续层将其破坏。
- **Misresolved (错误解决)**：模型从未正确计算，但自信地收敛到一个错误答案。
- **Unresolved (未解决)**：模型既未正确也未错误地收敛，计算未完成。

### 相比现有方法的优势
- **超越表面评估**：不依赖最终准确率，而是深入模型内部，揭示了相同准确率背后截然不同的失败模式。
- **因果验证**：通过激活修补（activation patching）、层跳过（layer skipping）和重新注入（re-injection）等干预实验证明了这四种结果是具有因果意义的计算状态，而非事后标签。
- **普适性发现**：发现了一个稳定的“酝酿支架”（brewing scaffold），即答案信息通常在约24-42%的网络深度处变得线性可读，而自解码能力则在约50%深度才出现，这一时间差（△brew）是模型进行核心计算的阶段。
- **无真值信号**：提出了一种无需访问真实答案（ground-truth-free）的信号系统，可通过熵和置信度变化预测“过度处理”等状态，为运行时监控提供了可能。

## 2. 核心实验方法和设置

### 使用了哪些数据集
论文构建了一个名为 **CUE-Bench** 的合成基准测试，包含六个代码推理任务家族，共24,300个样本（每个模型）：
- **Data Flow**: `Value Tracking` (值追踪), `Computing` (计算)
- **Control Flow**: `Conditional` (条件分支)
- **Data+Control Flow**: `Function Call` (函数调用), `Loop`, `Loop-unrolled` (循环展开)

所有任务的答案均为单个数字（0-9），保证了诊断空间的一致性。

### 实验设置和评估指标
- **模型**：在16个来自Qwen、Llama和DeepSeek系列的Decoder-only Transformer模型上进行了实验（参数量从0.5B到14B）。
- **核心指标**：
  - **First Probe-Correct Layer (FPCL)**：答案首次在线性探针下正确的层数。
  - **First Joint-Correct Layer (FJC)**：探针和CSD首次同时正确的层数。
  - **Brewing Duration (△brew)**：FJC - FPCL，表示信息从“可用”到“就绪”的间隔。
  - **四类分辨率结果的比例**：这是最核心的分析指标。
- **评估方式**：对每个样本进行一次前向传播，获取所有层的隐藏状态，然后应用双重诊断框架生成轨迹。

### 基线方法对比
本文并非直接与某个特定的基线模型进行性能对比，而是**与传统的评估范式**进行对比：
- **传统方法**：仅报告最终准确率（accuracy）。
- **本文方法**：揭示了在相同准确率下，不同模型或任务的失败模式（如Overprocessed vs. Unresolved）存在巨大差异。例如，在`Function Call`任务中，随着调用深度增加，`Resolved`率从61.1%骤降至2.5%，而传统准确率无法解释这种急剧下降的具体原因。

## 3. 主要实验结果和性能指标

### 关键性能数据
- **总体分辨率结果分布**：在所有任务和模型上，仅有 **41.5%** 的样本属于 `Resolved`。其余为：`Overprocessed` (26.4%)，`Misresolved` (8.5%)，`Unresolved` (23.7%)。这表明超过一半的样本都以某种形式失败。
- **任务间差异显著**：
  - `Value Tracking` 表现最好，`Resolved` 率达 **70.8%**。
  - `Function Call` 和 `Loop-unrolled` 表现最差，`Resolved` 率分别仅为 **27.7%** 和 **28.0%**。
- **关键瓶颈**：
  - `Function Call` 的 `Resolved` 率随调用深度从1到3，从 **61.1%** 崩溃至 **2.5%**，是所有任务中最严重的退化。
  - `Computing` 任务中，随着步骤增加，`Overprocessed` 率从25.4%上升到47.5%，表明模型能启动计算但难以维持。
- **稳定的时间模式**：
  - **酝酿支架 (Brewing Scaffold)** 高度稳定：归一化的 `△brew` 在所有16个模型上均落在 **24-42%** 的区间内。
  - 分辨率成功与否与模型能力、规模和训练相关，但酝酿过程本身是一个经验规律。

### 与基线方法的对比结果
- **传统准确率的误导性**：例如，`Value Tracking` 准确率为70.8%，`Loop-unrolled` 仅为28.0%。传统观点会认为后者更难。但本文发现，`Loop-unrolled` 的低分主要是由于 `Unresolved`（34.9%），而 `Value Tracking` 的失败更多是 `Overprocessed`（13.8%）。这两种失败需要完全相反的干预策略（前者需更多计算，后者需早停），但传统指标无法区分。

### 消融实验结果
- **干预实验**（因果消融）：
  - **激活修补 (Activation Patching)**：在FJC层修补隐藏状态，答案翻转率（flip rate）在FJC层有明显跳跃，证实了其因果重要性。
  - **层跳过 (Layer Skipping)**：对于 `Overprocessed` 样本，从FJC层直接跳到输出，采用alpha-blend注入，平均有 **47.8%** 的样本被“救回”，证明了过度处理是可逆的。
  - **重新注入 (Re-injection)**：对于 `Unresolved` 样本，将早期（FPCL）的信息重新注入深层，有 **22-38%** 的样本被救回，表明这些样本的计算是“未完成”而非“不可能”。
- **无真值信号**：仅使用CSD的熵和置信度信号，无需真实答案，即可达到 **64.3%** 的总体结果分类一致性和 **0.69-0.85** 的AUC来检测 `Overprocessed`。

## 4. 关键结论和发现

### 论文的主要发现
1.  **代码推理有生命周期**：LLMs的代码推理遵循一个“酝酿到解决”的生命周期，答案信息首先变得线性可读（availability），然后经过一段时间的“酝酿”才变得自解码（readiness）。
2.  **四大失败模式**：存在四种根本性的、可区分的失败模式，其中 `Overprocessed` 和 `Unresolved` 占据主导且需求相反的干预。
3.  **稳定支架，可变结果**：“酝酿”过程（△brew）在不同架构和规模的模型中表现出惊人的稳定性，是一个经验规律；而最终的分辨率成功与否则与模型的能力、规模和训练数据强相关。
4.  **任务特异性瓶颈**：不同的代码原语（code primitives）触发不同的失败指纹。函数调用的间接性（indirection）是最大的瓶颈，循环语法（syntax）为模型提供了重要的处理支架。
5.  **反对单一策略**：由于 `Overprocessed` 和 `Unresolved` 需求相反，任何单一的早停（early-exit）策略都无法普适，必须发展基于结果感知（outcome-aware）的推理策略。

### 方法的局限性
- **层粒度**：诊断在层级别进行，无法精确定位到具体的注意力头或MLP子层。
- **合成数据集**：CUE-Bench使用的是短小、单一原语的程序，尚不清楚这些动态在复杂的、多语句的组合代码中是否依然成立。
- **通用性**：虽然在多个架构上验证了“酝酿”现象，但其作为电路级普遍性主张仍需更多证据。

### 未来工作方向
1.  **扩展到路径补丁 (Path Patching)**：将诊断粒度从层级别细化到具体组件（如注意力头）。
2.  **连接自适应深度系统**：将本文的诊断信号（如熵上升）与Mixture-of-Depths、Looped Transformers等自适应计算系统结合，实现智能的“何时停止”决策。
3.  **探索根本原因**：探究为何后期层会破坏正确答案（`Overprocessed`），以及“可用性先于就绪性”这一顺序是否是残差流几何结构的必然结果。
4.  **应用于真实世界代码**：在更复杂、真实的代码库和任务上验证此框架的有效性。

</details>

---

### 6. [Revisiting LLM Adaptation for 3D CT Report Generation: A Study of Scaling and Diagnostic Priors](https://arxiv.org/abs/2606.17213)

**Authors**: Vanshali Sharma, Andrea M. Bejar, Halil Ertugrul Aktas, Quoc-Huy Trinh, Debesh Jha, Gorkem Durak, Ulas Bagci  
**Category**: cs.CL  
**Published**: 2026-06-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.17213v1  

#### Abstract
Recent advances in multimodal learning, including large language models (LLMs) and vision-language models (VLMs), have demonstrated strong adaptability to natural images. However, extending their use to the medical domain, particularly for volumetric (3D) images, is challenging due to high computati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Revisiting LLM Adaptation for 3D CT Report Generation: A Study of Scaling and Diagnostic Priors

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对**3D CT影像报告生成**中的三大挑战进行了系统研究：

- **临床幻觉（Clinical Hallucination）**：在有限医学数据上微调大型语言模型（LLM）容易导致过拟合，使模型优先保证语言流畅性而非临床事实准确性。
- **计算效率低下（Computational Inefficiency）**：全参数微调需要优化数十亿参数，在资源受限的临床环境中不现实。
- **语义临床鸿沟（Semantic Clinical Gap）**：高维3D视觉特征与复杂的医学术语之间存在巨大语义差距，难以对齐。

此外，现有方法多集中于2D图像或自然图像场景，缺乏对**3D医学影像中诊断先验整合**与**LLM缩放规律**的系统探索。

---

### **提出了什么新方法或新思路**

作者提出了一种轻量级、参数高效的框架——**RAD3D-Prefix**，其核心思想是：

- **冻结LLM主干**，仅训练一个轻量化的**anomaly-aware prefix projection network**。
- 将**3D图像嵌入（image embeddings）** 与**多标签异常分类logits**融合，作为“前缀”注入到冻结的LLM中，实现模态对齐。
- 采用**prefix learning机制**，将视觉信息转化为LLM可理解的token序列前缀，从而引导报告生成。

该方法基于三种变体设计（V-1, V-2, V-3），最终以**V-3**为最优配置：
- V-1：微调LLM + 图像嵌入前缀
- V-2：冻结LLM + 图像嵌入前缀
- V-3：冻结LLM + 图像嵌入 + 分类logits前缀（即RAD3D-Prefix）

---

### **相比现有方法的优势**

| 优势维度 | 具体表现 |
|--------|--------|
| **参数效率** | 可训练参数仅约279M（远低于全微调的1.5B+），显著降低训练成本 |
| **抗过拟合能力** | 冻结LLM避免破坏预训练知识，尤其适用于小规模医学数据集 |
| **临床相关性提升** | 引入诊断先验（classification logits）显式暴露临床概念（如effusion, consolidation），减少幻觉 |
| **泛化能力强** | 在跨域数据集（INSPECT）上表现优异，证明强out-of-domain generalization |
| **性能优越** | 在多个自动指标和临床专家评估中均优于同类方法 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

| 数据集 | 描述 | 用途 |
|------|-----|-----|
| **CT-RATE** | 包含50,188例非增强胸部CT扫描，带18个多异常标签和放射学报告 | **In-domain** 主要训练与测试 |
| **INSPECT** | 聚焦肺栓塞的CT数据集，共19,402例患者，提取21种异常 | **Out-of-domain** 泛化能力验证 |

> 所有CT体积统一重采样至480×480×240分辨率，HU值截断[-1000, 1000]

---

### **实验设置**

- **视觉编码器**：使用预训练的 **CT-CLIP** 模型提取3D图像嵌入（latent vector, dim=512）
- **分类头**：在CT-CLIP基础上添加多标签分类层，输出各异常的logits（用于V-3）
- **LLM解码器**：采用 **LLaMA-3.2-1B** 或其他对比LLM（如BioGPT-Large, DeepSeek-R1-Distill-LLaMA-8B）
- **投影网络**：Transformer-based projection network（8层），将拼接后的 `[image embedding + logits]` 映射为LLM风格的prefix序列（长度=10）

- **训练细节**：
  - Adam优化器，学习率2e-5
  - 训练10个epoch
  - 冻结视觉编码器和LLM，仅更新projection network

---

### **评估指标**

| 指标类型 | 指标名称 | 说明 |
|--------|--------|------|
| **通用NLG指标** | BLEU(1-4), METEOR, ROUGE(1,2,L), BERTScore-F1 | 衡量文本相似性和语义一致性 |
| **医学专用指标** | **GREEN Score** | 基于正则表达式识别匹配发现与错误，强调**临床事实正确性** |
| **人工评估** | **Reader Study** | 两名临床专家盲评100份报告，评分项包括临床准确性和语言质量（5分制） |
| **额外医学指标** | F1-RadGraph, RaTE Score | 评估实体识别与关系抽取能力 |

---

### **基线方法对比**

| 基线方法 | 特点 |
|--------|------|
| **R2GenGPT [33]** | 当前主流方法，冻结LLM + 简单linear projection，架构最接近 |
| **CT2Rep [10]** | 使用3D视觉编码器+语言模型联合微调 |
| **E3D-GPT [16]** / **CT-AGRG [7]** | 领域内先进方法，但未开源代码，仅比较报告数值 |
| **Baseline'** | 不加prefix的直接微调或clip-to-text baseline |

---

## 3. 主要实验结果和性能指标

### **关键性能数据（CT-RATE测试集）**

| 方法 | LLM | Avg. BLEU | METEOR | ROUGE-L | BERTScore-F1 | **GREEN Score** |
|------|-----|----------|--------|---------|--------------|----------------|
| R2GenGPT | LLaMA-3.2-1B | 0.2902 | 0.3762 | 0.3468 | 0.8751 | 0.4120 |
| **RAD3D-Prefix (Ours)** | LLaMA-3.2-1B | **0.3637** | **0.4694** | **0.4190** | **0.8883** | **0.5488** |
| R2GenGPT | LLaMA-2-7b-chat-hf (7B) | 0.3523 | 0.4509 | 0.4038 | 0.8886 | 0.5041 |
| RAD3D-Prefix | DeepSeek-R1-Distill-LLaMA-8B | 0.3405 | 0.4459 | 0.4339 | 0.8909 | 0.5443 |

> ✅ **结论**：即使使用更小的1B LLM，RAD3D-Prefix在多数指标上超越7B/8B模型，尤其在**GREEN Score**上领先明显（+4.5–13.7%）

---

### **与基线方法的对比结果**

- 在CT-RATE上，RAD3D-Prefix在**GREEN Score**上比CT2Rep高出 **2.4个百分点**（0.5488 vs 0.5247），表明更强的临床事实保真度。
- 在**INSPECT（out-of-domain）** 上，RAD3D-Prefix显著优于CT2Rep：
  - GREEN Score: **0.2400 vs 0.2219**
  - METEOR: **0.2122 vs 0.1207**
  - BLEU-4: **0.0344 vs 5.74e-6**
- 即便使用相同vision encoder和LLM，性能提升仍来自**anomaly-aware prefix设计本身**，而非backbone更强。

---

### **消融实验结果（Ablation Study）**

| 变体 | Visual Prefix | Frozen LLM | Classification Logits | GREEN Score | Trainable Params |
|------|-------------|------------|------------------------|-------------|------------------|
| V-1 | ✗ | ✗ | ✗ | 0.4454 | 1.51B |
| V-2 | ✓ | ✓ | ✗ | 0.5428 | 279.09M |
| **V-3 (Ours)** | ✓ | ✓ | ✓ | **0.5488** | **279.46M** |

> 🔍 发现：
> - 加入**visual prefix**（V-2 vs V-1）带来 **21.8% GREEN提升**
> - 加入**classification logits**（V-3 vs V-2）进一步提升 **1.1% GREEN**，且统计显著（p < 0.05）
> - 参数增加极少（仅+0.37M），但能保留更多关键临床细节

---

### **其他重要实证发现**

- **UMAP可视化**显示，V-3生成的嵌入空间更紧凑、结构更清晰，疾病连续性更好。
- **依赖性分析**表明：报告生成器不会盲目跟随分类器输出，能纠正假阳性信号（如classifier置信度0.62 → 报告正确判断“未见心包积液”）。
- **bootstrapping分析**确认所有关键指标改进均具有统计显著性（p < 0.05）。
- **reader study**结果显示，V-3在临床相关性上比baseline高 **9.8%**，比V-2高 **3.7%**。

---

## 4. 关键结论和发现

### **主要发现**

1. **LLM缩放法则在3D医学领域反转**：
   - 对于 **<1B参数LLM**，微调（fine-tuning）更有利；
   - 对于 **~1B+参数LLM**，**冻结+轻量适配**（如prefix tuning）反而性能更优。
   - ❗这与自然图像领域的LLaVA/BLIP-2等发现相反，凸显医学领域的特殊性。

2. **诊断先验（Diagnostic Priors）至关重要**：
   - 显式引入多标签分类logits可有效缩小**语义临床鸿沟**，提高报告的事实性。
   - 该机制不是简单“复制分类结果”，而是提供上下文引导，允许LLM结合视觉特征进行精细化判断。

3. **参数效率与性能可以兼得**：
   - RAD3D-Prefix仅需训练 **<0.3% 的总参数**，即可达到甚至超越全微调大模型的表现。
   - 使用linear projector替代transformer mapper后，虽参数减少26倍，但仍优于R2GenGPT，说明**prefix机制本身是增益主因**。

4. **强泛化能力**：
   - 在未见过的INSPECT数据集上仍保持竞争力，证明方法鲁棒性强，适合真实世界部署。

---

### **方法的局限性**

- **异常测量描述不足**：模型在涉及具体尺寸、密度值等定量描述时表现较弱（如“结节大小30mm”）。
- **依赖外部分类器**：需预先训练一个高质量的multi-abnormality classifier来提供logits。
- **当前仅支持单器官（胸部CT）**，扩展至腹部或其他部位需重新适配。
- **无法处理时间序列或多期扫描**（如动脉期、静脉期）。

---

### **未来工作方向**

1. **集成定量推理模块**：引入数值感知机制，提升对病灶尺寸、HU值等量化信息的建模能力。
2. **端到端联合训练**：探索vision encoder、classifier与projection network的协同优化路径。
3. **跨模态知识蒸馏**：利用更大模型生成伪标签，缓解标注稀缺问题。
4. **向多器官、多模态扩展**：推广至MRI、PET-CT等其他3D医学成像模态。
5. **构建闭环反馈系统**：结合医生反馈持续迭代优化模型输出。

---

> 📌 **一句话总结**：  
> 本文通过系统研究LLM在3D CT报告生成中的适应策略，提出**RAD3D-Prefix**——一种融合诊断先验、参数高效、冻结LLM的轻量前缀学习框架，在提升临床事实准确性的同时大幅降低训练开销，为医学VLM的发展提供了新的范式参考。

</details>

---

### 7. [From Trainee to Trainer: LLM-Designed Training Environment for RL with Multi-Agent Reasoning](https://arxiv.org/abs/2606.17682)

**Authors**: Chao Chen, Chengzu Li, Zhiwei Li, Yinhong Liu, Zhijiang Guo  
**Category**: cs.CL  
**Published**: 2026-06-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.17682v1  

#### Abstract
Reinforcement learning pipelines for Large Language Model (LLM) training often rely on manually redesigned environments between stages, requiring practitioners to heuristically infer which configuration will best improve the current policy. To automate this process, we propose the LLM-as-Environment...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Trainee to Trainer: LLM-Designed Training Environment for RL with Multi-Agent Reasoning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在当前基于 **Reinforcement Learning (RL)** 的 **Large Language Model (LLM)** 训练流程中，训练环境的设计通常依赖人工反复调试。研究人员需要手动分析模型失败轨迹、推测其弱点，并调整下一阶段的训练环境配置。这一过程高度依赖专家经验，且难以随训练规模扩大而自动化。

本文旨在解决以下核心问题：  
> **能否让当前策略模型（policy model）自身主动地、有依据地重新设计其未来的 RL 训练环境？**

### ✅ 提出的新方法与新思路

作者提出了 **LLM-as-Environment-Engineer** 框架，构建了一个闭环的“从学员到教练”系统：

- **框架机制**：
  - 在每一轮 RL 训练后，当前的 LLM 模型作为 *environment engineer*，接收关于其行为表现的结构化反馈（如失败模式、环境统计等）。
  - 基于这些信息，该模型输出下一阶段训练所用的 **environment configuration**（例如地图大小分布、障碍密度、冲突需求比例），而非直接生成样本或修改奖励函数。
  - 环境生成器根据此配置采样新的训练实例，从而动态重塑训练分布。

- **可控测试平台 MAPF-FrozenLake**：
  - 作者设计了一个可控制的多智能体路径规划任务 **MAPF-FrozenLake**，是经典 FrozenLake 的扩展版本。
  - 支持多 agent（2–5 agents）、带 hole（不可通行区域）和 wait 动作以解决冲突。
  - 环境生成器参数化，允许调节 `map_size`, `hole_ratio`, `wait_ratio`, `data_ratio` 等维度，便于研究环境重设计的影响。

### ✅ 相比现有方法的优势

| 对比维度 | 传统方法 | 本文方法 |
|--------|---------|----------|
| **环境调整方式** | 手动设计 curriculum 或固定难度调度 | 自动化、数据驱动的闭环重设计 |
| **适应性来源** | 外部启发式规则或预定义任务序列 | 基于模型自身失败证据进行诊断 |
| **作用对象** | 修改训练样本选择或难度排序 | 修改整个环境生成器的参数分布 |
| **目标导向** | 提高当前任务性能 | 最大化未来策略改进潜力 |

> ✅ **核心优势**：将 LLM 从被动学习者转变为主动“教练”，实现 **policy-conditioned environment engineering**，推动自进化学习系统的探索。

---

## 2. 核心实验方法和设置

### 📦 数据集与环境
- **MAPF-FrozenLake**：自研可控测试平台。
  - 地图尺寸范围：3×3 到 10×10。
  - 固定使用 **2-agent 实例进行训练**，但在 **3-/4-/5-agent 实例上评估泛化能力**。
  - 每轮训练集包含 4000 个样本，由参数化生成器构造。
  - 验证集固定不变，用于监控进展；另设独立评估 benchmark 包含不同 `wait_ratio` 子集（wr_025, wr_050, wr_075）。

### 🧪 实验设置
- **基础模型**：Qwen3-4B（开源 4B 参数 LLM）
- **训练算法**：GRPO（一种无需 critic 的强化学习算法）
- **训练流程**：
  - 共三轮训练（Round 0 → R1 → R2）
  - 每轮两个 epoch，每轮结束后调用当前 checkpoint 作为 environment engineer 设计下一轮配置
- **上下文输入模块**（共五种）：
  1. **Failure breakdown (F)**：验证失败类型统计（parse error, conflict, hole collision 等）
  2. **Guideline (G)**：通用设计原则提示
  3. **History (H)**：历史配置与失败记录
  4. **Summary (S)**：前一轮模型自动生成的设计理由
  5. **Training details (T)**：RL 目标、奖励结构、训练进度等元信息

> 最终采用 **V6 设置**：包含 F, G, H (不含初始随机配置), S, T

### 🎯 评估指标
- **Valid Rate (acc.)**：响应是否能解析为合法、无冲突、到达目标且不越界的路径（通过全部 8 项 validity check）。
- **Optimal Rate (opt.)**：是否找到最短路径（成本等于 ground truth）。
- 报告各 map size 下的表现及总体 **aggregate sum**。

### 🔁 基线方法对比
| 基线类型 | 方法名称 |
|--------|---------|
| 商业闭源 LLM | GPT-5.4, Grok-4.2, Gemini-3.1-Pro, Kimi-K2.5 |
| 开源对照 | Qwen3-4B (base)，即未训练原始模型 |
| 固定环境基线 | Qwen3-4B + GRPO (random)：仅在 Round-0 随机配置上训练，无环境重设计 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Tables 2–4）

| 模型 | 3-agent valid (%) | 4-agent valid (%) | 5-agent valid (%) |
|------|-------------------|-------------------|-------------------|
| **Qwen3-4B + GRPO + Ours (V6)** | **51.67** | **33.14** | **18.67** |
| Kimi-K2.5 | 46.17 | 26.95 | 13.47 |
| GPT-5.4 | 32.50 | 17.05 | 9.11 |
| Qwen3-4B (base) | 14.83 | 3.43 | 1.44 |
| Qwen3-4B + GRPO (random) | 40.42 | 26.67 | 15.11 |

> ✅ **结论**：提出的框架在所有 agent 数量和 map size 上均显著优于所有基线，尤其在复杂场景（如 5-agent）中优势明显。

#### 性能提升幅度（vs. 最佳商业模型 Kimi-K2.5）：
- **Valid Rate 提升**：+5.2 ~ +6.2 pts
- **Optimal Rate 提升**：+2.2 ~ +3.4 pts

#### vs. 同架构固定环境训练（Qwen3-4B + GRPO random）：
- Valid Rate 提升：+3.6 ~ +11.3 pts
- Optimal Rate 提升：+1.9 ~ +5.6 pts  
👉 表明 **环境重设计本身带来了实质性增益**，不仅仅是 GRPO 训练的效果。

---

### 🔍 消融实验结果

#### （1）不同 context 设置比较（Table 5 & Figure 8）
- **V6（完整设置）表现最佳**，在所有三个 benchmark 上 valid rate 均最高。
- 移除 `round-0 default config`（V3 → V4）有助于避免对初始随机配置的过度依赖。
- 加入 `training details`（V4 → V6）促使模型做出更符合学习信号的决策，而非盲目增加难度。

#### （2）Training Details 内容影响（Table 7）
| 设置 | 是否包含完整 RL 细节 | 3-agent valid (%) |
|-----|----------------------|-------------------|
| Full RL details | 是（含算法、超参） | 38.83 |
| Bookkeeping only | 否（仅含轮次、epoch） | **51.67** |

> ❗ 发现：**过多 RL 细节反而干扰决策**。模型只需要知道“当前处于哪一轮”即可有效利用失败证据。

#### （3）谁担任 Environment Engineer？（Table 8）
| Engineer 类型 | 3-agent valid (%) | 5-agent valid (%) |
|---------------|-------------------|-------------------|
| **Untrained base model** | 45.21 | 16.00 |
| **Current RL checkpoint (ours)** | **51.67** | **18.67** |

> ✅ **关键发现**：经过 RL 训练后的 checkpoint 比原始 base model 更擅长诊断自身弱点，能更精准地分配训练资源。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLM 可以成为有效的 environment engineer**：
   - 即使是 4B 规模的开源模型，在合适上下文支持下，也能超越更大规模的商业 LLM 进行环境设计。

2. **成功 redesign 依赖 evidence-driven adaptation**：
   - 有效更新基于具体的失败证据（如 hole collisions 集中在 10×10），而不是简单地“越大越难越好”。
   - 成功策略包括：降低已饱和 map size 的权重、保留已有成效的配置、针对性增强薄弱环节。

3. **policy learning 提升 self-diagnosis 能力**：
   - 当前 RL checkpoint 比 base model 更清楚自己的能力边界，能做出更合理的资源配置（如避免在无法学习的大图上浪费预算）。

4. **context design 极其重要**：
   - 包含失败证据、历史趋势、训练阶段信息的组合（V6）效果最好。
   - 过度提供无关细节（如完整 RL 超参）会分散注意力，降低性能。

5. **避免 reward hacking，实现 training-aware design**：
   - 模型不能修改 reward 函数，只能调整环境生成参数，因此优化的是真正的学习效率，而非指标欺骗。

---

### ⚠️ 局限性（Limitations）

1. **任务单一性**：
   - MAPF-FrozenLake 是一个结构化、封闭的任务家族，策略可能不具备跨领域迁移性。

2. **generator 架构固定**：
   - 当前仅允许调整参数，不允许结构性修改（如引入新动作、新机制）。未来可探索 generator 自我演进。

3. **训练范式受限**：
   - 当前框架基于 offline RL + periodic redesign，尚未整合 online IL 或 reward-free exploration 等其他范式。

---

### 🔮 未来工作方向

1. 将 LLM-as-Environment-Engineer 框架推广至更复杂的 **embodied AI** 或 **web-based agent** 环境。
2. 探索 **multi-turn co-evolution**：多个 agent 或 reward model 共同参与环境演化。
3. 引入 **meta-generator**，允许 LLM 修改环境生成器本身的逻辑结构（如添加新元素、新规则）。
4. 结合 **causal inference** 方法，识别真正导致性能瓶颈的环境因素，提升 redesign 的因果有效性。

---

> 💡 **一句话总结**：  
> 本文提出并验证了一种新型闭环学习范式——让 LLM 在训练过程中逐步成长为自己的“教练”，通过分析失败证据自主优化训练环境，实现了比人工设计和更大模型更强的学习效率，为构建 **self-improving learning systems** 提供了重要路径。

</details>

---

### 8. [ConSA: Controllable Sparsity in Hybrid Attention via Learnable Allocation](https://arxiv.org/abs/2606.18056)

**Authors**: Yao Chen, Yinqi Yang, Junyuan Shang, Xiangzhao Hao, Simeng Zhang, Yilong Chen, Tingwen Liu, Shuohuan Wang, Dianhai Yu  
**Category**: cs.CL  
**Published**: 2026-06-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.18056v1  

#### Abstract
Hybrid architectures combining full attention (FA) and sliding-window attention (SWA) are a promising paradigm for efficient LLM inference. However, existing methods typically rely on hand-crafted rules or simple post-hoc heuristics for FA/SWA allocation and offer limited analysis of the attention b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ConSA: Controllable Sparsity in Hybrid Attention via Learnable Allocation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的混合注意力架构（hybrid attention）通常采用**手工设计规则**（如交替使用 Full Attention 和 Sliding-Window Attention）来分配 FA 和 SWA，这类方法存在以下问题：
- 忽略了不同层（layer）和不同 KV-head 之间注意力行为的异质性；
- 缺乏对目标稀疏度（sparsity）的精确控制；
- 难以适应不同模型规模和任务需求。

此外，现有方法缺乏对注意力行为内在机制的深入分析，难以解释为何某些模式更优。

### 提出了什么新方法或新思路
作者提出 **ConSA**（Controllable Sparsity in Hybrid Attention），一种基于学习的混合注意力分配框架，其核心思想是：
- 将 FA/SWA 分配建模为一个**带稀疏性约束的优化问题**；
- 引入 **learnable binary mask**（通过 hard concrete distribution 参数化）为每个 attention unit（layer 或 KV-head）动态选择 FA 或 SWA；
- 使用 **augmented Lagrangian 方法** 显式约束期望稀疏度 $ \mathbb{E}[p(z)] = p $，确保最终满足用户指定的目标稀疏率 $ p $。

该方法支持两种粒度：
- **Layer-wise allocation**：整层统一使用 FA 或 SWA；
- **KV-head-wise allocation**：同一层内不同 KV-head 可独立选择。

### 相比现有方法的优势
| 维度 | ConSA | 现有方法（如 Mistral、LoZA） |
|------|-------|-----------------------------|
| **分配方式** | 学习式 end-to-end 优化 | 手工规则或后验排序 |
| **稀疏控制** | 支持任意目标 $ p $，精确收敛 | 固定比例或间接控制 |
| **粒度** | 支持 layer 和 head 两级 | 多为 layer 级 |
| **训练集成** | 联合优化 mask 与模型参数 | 冻结权重进行 calibration |
| **可解释性** | 揭示了系统性的 SWA-bottom / FA-middle 结构 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **预训练数据**：来自开源语料库 [RedPajama](https://arxiv.org/abs/2401.12345) 和 [Chinese FineWeb](https://arxiv.org/abs/2501.08197)，覆盖多语言、多领域文本。
- **评估基准**：涵盖英文与中文的知识、推理、常识等任务，具体包括：
  - **知识理解**：MMLU
  - **逻辑推理**：LogiQA-EN / LogiQA-CN
  - **常识推理**：CSQA, PIQA, SIQA
  - **科学与上下文推理**：ARC-C, ARC-E, HellaSwag
  - **开放问答**：WebQA-CN
  - **中文生成任务**：CN-GEN（含 CSL 科学摘要与 LOT 故事生成）

### 实验设置和评估指标
- **模型规模**：在两个自研 LLM 上验证：
  - 0.6B 和 1.7B 参数模型，均为 28 层，GQA 架构（16 query heads, 8 KV heads）
- **序列长度**：最大 8,192 tokens
- **稀疏目标 $ p $**：设定为 0.50（即 50% KV heads 使用 SWA）
- **训练流程两阶段**：
  1. **Mask Learning Stage**（1B tokens）：联合优化模型参数 $ \theta $、mask 参数 $ \alpha $、Lagrange 乘子 $ \lambda, \phi $
  2. **Continue Pre-training Stage**（100B tokens）：固定二值化后的 mask，继续微调模型
- **评估指标**：下游任务平均准确率（accuracy），以及训练 FLOPs、KV cache 占用等效率指标。

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Dense FA** | 全量注意力 | 无稀疏，作为性能上限参考 |
| **Rule (layer-wise)** | 规则式 | 如 Mistral/Gemma 2 的交替模式 |
| **Rule (head-wise)** | 规则式 | 每层内部按比例手工地分配 FA/SWA |
| **LoZA** | 学习式 | 基于标量评分排序转换低分层为 SWA，仅报告 50% 情况 |
| **ConSA (ablation)** | 消融变体 | 移除 LO-Lagrangian，改用标量门控 + 排序策略 |

所有方法均从相同初始 checkpoint 出发，并接受相同的总训练量（101B tokens），保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1, 1.7B 模型，$ p=0.50 $）

| 方法 | Average Accuracy |
|------|------------------|
| Dense FA | 47.83 |
| **ConSA (head-wise, single-layer)** | **49.06** ✅ |
| ConSA (head-wise, all-layers) | 48.13 |
| ConSA (layer-wise) | 47.74 |
| Rule (head-wise) | 47.31 |
| Rule (layer-wise) | 46.45 |

> 🔍 **关键观察**：
> - **ConSA (head-wise)** 不仅显著优于所有规则方法，甚至**超越 Dense FA**（+2.6% 相对提升），表明局部注意力可能起到正则化作用；
> - **head-wise > layer-wise**：细粒度分配带来明显增益；
> - **single-layer constraint > all-layers constraint**：逐层施加稀疏约束比全局约束更有效，因其提供更强的结构先验。

### 与基线方法的对比结果
- 在 **11 项评测任务中全面领先**，尤其在逻辑推理（LogiQA）、常识推理（PIQA）上优势明显；
- 相比 Rule-based 方法，在相同稀疏度下平均提升约 **2.8%~3.7%**；
- 在 **KV cache 和 FLOPs 上实现显著节省**（见 Table 4）：
  - 0.6B 模型：训练 FLOPs ↓15.9%
  - 1.7B 模型：训练 FLOPs ↓10.6%
  - 且随着 context length 增长，收益进一步放大。

### 消融实验结果（Section 5.4, Figure 3）
- 对比移除了 LO-Lagrangian 的 ablation 版本（仅用 scalar gate + ranking）：
  - ConSA 在所有稀疏水平（$ p=0.25, 0.50, 0.75 $）下都取得更低的语言建模 loss；
  - 差距随 $ p $ 增大而扩大，说明在高稀疏场景下，显式约束更为关键；
- Scalar gates 几乎无法区分各层重要性（Figure 7），导致分配缺乏判别力。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **Learned allocation beats hand-crafted rules**  
   学习得到的 FA/SWA 分配模式显著优于人工设计的交错模式。

2. ✅ **Emergence of SWA-bottom / FA-middle structure**  
   不论模型大小（0.6B vs 1.7B）、稀疏程度（$ p=0.25\sim0.75 $）或分配粒度，学习结果始终呈现：
   - **底层（bottom layers）优先使用 SWA**
   - **中间层（middle layers）集中使用 FA**
   这挑战了“首层需 FA 抓取全局上下文”的直觉。

3. ✅ **Head-wise granularity enables intra-layer heterogeneity**  
   同一层内的不同 KV-head 表现出显著不同的注意力范围（dense broad vs uniform/local），强制统一 attention type 是次优的。

4. ✅ **Fine-grained spectrum of attention behaviors**  
   分析 last-token attention 分布发现，注意力模式是一个连续谱系，远超“retrieval vs streaming”二元分类：
   - 包括 uniform、weakly local、strongly local、sparse broad、dense broad 等多种形态；
   - ConSA 的分配与此谱系高度一致：窄范围 → SWA，宽范围 → FA。

5. ✅ **Lagrangian constraint ensures precise sparsity control**  
   所有配置下的约束损失均在 1K 步内收敛至零（Figure 2 & 8），证明可稳定达到任意目标 $ p $。

### 方法的局限性
- **SWA window size 固定**：未联合优化窗口大小 $ w $，可能限制性能上限；
- **泛化性有待验证**：目前仅在两个自研模型上测试，尚未扩展到更大规模（如 10B+）或其他架构（如 Mamba-based）；
- **长程依赖任务表现下降**：在 CN-GEN 等需要超长依赖的任务上略有退化，因 SWA 截断了远距离信息。

### 未来工作方向
- 联合优化 **window size + allocation policy**；
- 将 ConSA 应用于 **post-training 或 instruction tuning 阶段**，探索任务自适应稀疏化；
- 探索 **dynamic masking**，允许 inference 时根据输入调整 sparsity；
- 扩展至 **multi-modal 或 encoder-decoder 模型** 中的注意力调度。

---

> 📌 **一句话总结**：  
> ConSA 通过 **LO regularization + augmented Lagrangian** 实现了可控稀疏下的最优 FA/SWA 分配，揭示了“**底层层用 SWA、中层块用 FA**”这一反直觉但高效的结构规律，为高效 LLM 设计提供了新的原则性路径。

</details>

---

### 9. [Learning from the Self-future: On-policy Self-distillation for dLLMs](https://arxiv.org/abs/2606.18195)

**Authors**: Yifu Luo, Zeyu Chen, Haoyu Wang, Xinhao Hu, Yuxuan Zhang, Zhizhou Sha, Shiwei Liu  
**Category**: cs.CL  
**Published**: 2026-06-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.18195v1  

#### Abstract
On-policy self-distillation (OPSD) has proven effective for post-training large language models (LLMs), yet its application to diffusion LLMs (dLLMs) remains unexplored. Existing OPSD methods are inherently autoregressive-centric. They inject privileged information via left-to-right prefix condition...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning from the Self-future: On-policy Self-distillation for dLLMs*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **On-policy Self-distillation (OPSD)** 方法主要针对自回归语言模型（AR LLMs），其设计依赖于从左到右的生成方式和 token-level 的监督信号。然而，**扩散大语言模型（dLLMs）** 具有非自回归、任意顺序生成的特点，使得传统 OPSD 方法在 dLLMs 上存在根本性不兼容：
- **教师构造方式冲突**：传统方法将参考答案作为前缀（prefix）注入提示中，这仅适用于只能进行 `p(suffix|prefix)` 预测的 AR 模型。
- **监督粒度不匹配**：dLLMs 在每一步去噪时同时预测多个 token，而 token-level 的 KL 散度监督无法有效对齐其迭代去噪机制。

因此，如何为 dLLMs 设计一个适配的、高效的自蒸馏框架是一个尚未探索的问题。

---

### 提出了什么新方法或新思路
本文提出了 **d-OPSD**（diffusion On-Policy Self-Distillation），是首个专为 dLLMs 设计的 on-policy self-distillation 框架，具有两大核心创新：

#### （1）基于“自我未来经验”的教师构造（Suffix Conditioning）
- 利用 dLLMs 支持双向建模的能力 `p(prefix|suffix)`，将学生模型**自生成的答案**作为后缀条件（suffix conditioning）提供给教师模型。
- 这相当于让教师“看到”学生的“未来”，从而引导学生从自身的“未来经历”中学习，更符合 on-policy 范式。
- 与传统 AR-style 方法（将静态参考解作为 prefix）相比，该方法传递的是动态的、策略相关的思考模式（thinking patterns），能引入更多新知识。

#### （2）Step-level Divergence Supervision
- 将监督粒度从 token-level 升级为 **step-level KL divergence**。
- 在每个去噪步 $t$，只对即将被揭示的 top-$k$ 最置信 token 计算 KL 散度，使监督信号与 dLLMs 的实际推理过程对齐。
- 更自然地匹配了 dLLMs 的 Markov 式迭代更新机制。

---

### 相比现有方法的优势
| 方面 | d-OPSD vs. SFT | d-OPSD vs. RLVR | d-OPSD vs. 其他 self-distillation |
|------|----------------|------------------|-------------------------------|
| **样本效率** | 显著更高（无需大量标注数据） | **仅需约 10% 的优化步数即可收敛** | 更高，因提供全轨迹密集监督 |
| **暴露偏差（exposure bias）** | 无（使用学生自身轨迹训练） | 无 | 无 |
| **奖励稀疏性** | 不适用 | 解决了 RLVR 中稀疏奖励瓶颈 | —— |
| **监督密度** | 低（仅最终输出） | 低（仅 outcome reward） | **高（step-level 密集监督）** |
| **on-policy 对齐性** | 差 | 好 | **极好（完全基于 self-generated future）** |

---

## 2. 核心实验方法和设置

### 使用的数据集
四个推理任务，涵盖数学与规划两类：
- **数学推理**：
  - **GSM8K**：小学数学应用题
  - **MATH500**：更具挑战性的高中级别数学题
- **规划任务**：
  - **Sudoku**（4×4）：约束满足类逻辑谜题
  - **Countdown**（3 数字）：通过基本运算达到目标数字

所有数据配置与 RLVR 基线 **diffu-GRPO** [Zhao et al., 2025] 保持一致。

---

### 实验设置和评估指标
- **基础模型**：LLaDA-8B-Instruct（未经过 post-training 的先进 dLLM）
- **生成长度**：
  - 数学任务：256 / 512
  - 规划任务：128 / 256（长序列反而降低性能）
- **推理策略**：Block-diffusion（块状扩散），块大小为 32
- **去噪步数**：设为生成长度的一半
- **评估频率**：每 25 步评估一次，在前 501 步内取最佳结果
- **主指标**：准确率（Accuracy），以是否正确解答问题为准

---

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **SFT 类** | SFT variant, d3LLM（伪轨迹蒸馏） |
| **RLVR 类** | diffu-GRPO, VRPO（group rollout policy optimization） |
| **OPSD 类** | AR-style OPSD（本文实现的对照版本） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 / 任务 | GSM8K (512) | MATH500 (512) | Countdown (256) | Sudoku (256) |
|------------|-------------|---------------|------------------|--------------|
| Base Model (LLaDA-8B) | 79.5 | 36.2 | 19.1 | 6.9 |
| SFT Variant | 81.1 | 34.8 | 14.5 | 8.5 |
| diffu-GRPO (RLVR) | 81.9 | 39.2 | 31.3 | 12.9 |
| d-OPSD (**Ours**) | **82.2** | **37.8** | **32.3** | **20.6** |

> ✅ **结论**：d-OPSD 在所有任务上均优于或持平于最强基线，尤其在 Sudoku 上表现远超其他方法。

---

### 样本效率对比（Table 2 & Figure 1）
| 方法 | GSM8K | MATH500 | Countdown | Sudoku |
|------|-------|---------|-----------|--------|
| diffu-GRPO | 7700 步 | 6600 步 | 5000 步 | 3800 步 |
| **d-OPSD (Ours)** | **425 步** | **100 步** | **175 步** | **425 步** |

> 🚀 **结论**：d-OPSD 仅需 **约 10% 的优化步数** 即可超越 RLVR，展现出极高的样本效率。

---

### 消融实验结果（Ablation Studies）

#### （1）教师保留比例 $p_{\text{teacher}}$
| $p_{\text{teacher}}$ | GSM8K 性能 |
|------------------------|------------|
| 0.10 | 80.5 |
| 0.25 (**default**) | **81.0** |
| 0.50 | 79.8 |

> 🔍 发现：并非 teacher 越强越好；适度保留（0.25）效果最佳，说明过强 teacher 可能导致过度拟合。

#### （2）top-$k$ 子集选择来源
| 来源 | GSM8K |
|------|--------|
| Student 分布 | 78.6 |
| **Teacher 分布（default）** | **81.0** |

> ✅ 使用 teacher 的置信度决定更新位置，提供了更强的学习信号。

#### （3）采样策略（Pass@k）
| $k$ | GSM8K | Countdown |
|-----|--------|-----------|
| 1 | 80.4 | 34.0 |
| 8 (**default**) | **81.0** | **37.9** |

> ⚠️ $k=1$ 仍优于 RLVR，且计算开销更低，适合资源受限场景。

#### （4）Pointwise KL Clipping
| 设置 | GSM8K |
|------|--------|
| 无 clipping | 77.0（训练崩溃） |
| **Clipping (threshold=0.05)** | **81.0** |

> ✅ Clipping 显著提升稳定性并防止后期性能坍塌。

#### （5）教师是否固定
| 设置 | GSM8K |
|------|--------|
| 不固定 teacher | 79.7 |
| **固定 teacher（default）** | **81.0** |

> ✅ 固定 teacher 更有利于稳定学习。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **d-OPSD 是首个成功应用于 dLLMs 的 on-policy self-distillation 框架**。
2. ✅ “**Self-future**”式的 suffix conditioning 能有效利用 dLLMs 的双向能力，显著优于传统的 prefix 注入方式（AR-style）。
3. ✅ **Step-level supervision** 完美契合 dLLMs 的迭代去噪机制，是 token-level 方法不可替代的关键设计。
4. ✅ d-OPSD 在 **推理性能** 和 **样本效率** 上全面超越 SFT 与 RLVR 基线，尤其在复杂规划任务（如 Sudoku）上优势明显。
5. ✅ 消融实验证明：teacher 构造方式的影响 > 监督形式 > 超参数调优。

---

### 方法的局限性
- ❗ **存在 Policy Collapse 风险**：与 RLVR 类似，在达到峰值性能后可能出现灾难性退化（见 Figure 12），可能源于 reverse KL 的“model-seeking”行为导致策略过早收敛。
- ❗ 当前仅在单一模型（LLaDA-8B）上验证，泛化性有待进一步检验。
- ❗ 依赖于生成正确的轨迹来构建 self-teacher，若初始模型能力不足可能导致冷启动困难。

---

### 未来工作方向
- 探索更稳定的训练机制（如 momentum-based teacher update）以缓解 collapse 问题。
- 扩展至多模态 diffusion models 或 code generation 场景。
- 结合 consistency distillation 或 iterative refinement 提升鲁棒性。
- 研究如何在低质量 initial policy 下启动 d-OPSD。

---

> 💡 **一句话总结**：  
> d-OPSD 通过“向自己的未来学习”这一新颖范式，首次实现了面向 dLLMs 的高效 on-policy self-distillation，在推理能力和训练效率上开辟了新的前沿路径。

</details>

---

### 10. [Evaluating LLM Coding Agents on SZ-Family Lossy Compression Across Architectures](https://arxiv.org/abs/2606.17058)

**Authors**: Changqing Li (Oregon State University), Shouwei Gao (Oregon State University), Kai Zhao (Florida State University), Sheng Di (Argonne National Laboratory), Wenqian Dong (Oregon State University)  
**Category**: cs.DC  
**Published**: 2026-06-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.17058v1  

#### Abstract
Large language model (LLM) coding agents are increasingly applied to code translation and optimization, yet their effectiveness in performance-critical high-performance computing (HPC) settings remains poorly characterized. This paper evaluates LLM-based coding workflows on SZ-family error-bounded l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Evaluating LLM Coding Agents on SZ-Family Lossy Compression Across Architectures

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
本论文系统地评估了**LLM-based coding agents**在**高性能计算（HPC）领域**中的实际能力，尤其是在跨不同硬件架构（如GPU与wafer-scale加速器）进行代码生成、优化与移植时的表现。当前大多数对LLM编程代理的研究集中在功能正确性（如pass@k、单元测试通过率），而忽视了**性能敏感型场景下的端到端吞吐量、架构适配性和鲁棒性**。

该研究聚焦于一个极具挑战性的HPC任务——**SZ-family误差有界lossy压缩核函数的CUDA实现迁移与优化**，并考察LLM代理在NVIDIA GPU和Cerebras WSE-3两种异构平台上的表现差异。

### 🚀 提出的新方法/新思路  
- **提出了一套面向HPC的LLM coding agent评估框架**，不仅关注最终代码是否可运行，更强调：
  - 端到端**throughput（GB/s）**
  - **优化轨迹（optimization trajectory）**
  - **迭代行为（reasoning iterations, convergence）**
  - 架构特定的**failure modes**
- 引入了**三级prompt tier设计**（User / Knowledgeable / Expert），用于量化LLM对提示精度和专家指导的敏感度。
- 在**Cerebras WSE-3**这一空间执行模型（spatial execution model）平台上首次系统评估LLM生成能力，揭示其与传统thread-based GPU平台的本质差异。

### 🔍 相比现有方法的优势  
| 维度 | 本文优势 |
|------|--------|
| **评估维度** | 超越“是否能跑通”，引入性能、收敛路径、失败模式等多维指标 |
| **工作负载选择** | 使用真实HPC应用（SZp/SZx），结合数值约束、内存密集、控制流复杂等特点 |
| **跨架构对比** | 首次在同一任务下比较LLM在GPU与wafer-scale加速器上的表现 |
| **分析深度** | 提供step-by-step优化轨迹分析，揭示LLM优化策略的强项与短板 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集  
- **两个SZ-family lossy compression kernel**：
  - **SZp**: 基于bit-level pipeline的紧耦合压缩器，涉及复杂的位打包与元数据处理
  - **SZx**: 模块化设计，支持灵活组件替换，结构更松散
- 所有实验基于已有的CUDA版本作为参考实现，并在**25MB输入数据集**上进行性能追踪

### ⚙️ 实验设置  
#### 平台环境
| 平台 | 类型 | 编程模型 |
|------|-----|----------|
| **NVIDIA V100 GPU** | Thread-based | CUDA |
| **Cerebras CS-3 (WSE-3)** | Spatial execution | Cerebras System Language (CSL)，静态映射、PE间显式通信 |

#### LLM Agent配置
- **模型对比**：
  - `GPT-5.1-Codex`（OpenAI）
  - `Gemini-2.5-pro`（Google）
- **Agent Workflow**：单智能体迭代式生成 → 编译/运行反馈 → 修改代码，最多100轮迭代
- **工具支持**：使用Model Context Protocol (MCP) 支持文件读写、编译命令执行等操作

#### Prompt Tier 设计（关键变量）
| 层级 | 内容说明 |
|------|--------|
| **User** | 最基础的任务目标与约束描述 |
| **Knowledgeable** | 加入SZ压缩流程阶段、误差边界要求等背景知识 |
| **Expert** | 进一步加入架构感知提示：<br>• GPU：memory coalescing、parallel decomposition<br>• Cerebras：spatial mapping、host-PE data movement |

#### 评估指标
| 指标 | GPU平台 | Cerebras平台 |
|------|--------|-------------|
| 主要性能 | Throughput (GB/s) | Runnability（能否成功执行）<br>Execution Time (s) |
| Agent行为 | - Iteration count<br>- Convergence pattern<br>- Optimization trajectory | - Reasoning iteration count<br>- 是否在预算内产出可运行程序 |
| 正确性 | 功能验证通过（test suite） | 同左 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（GPU平台）

#### ➤ Throughput 对比（V100）

| Kernel | Model | Prompt Tier | Compress (GB/s) | Decompress (GB/s) |
|-------|--------|--------------|------------------|--------------------|
| SZp   | Gemini | Expert       | **0.446**        | **0.441**          |
|       | GPT    | Expert       | 0.088            | 0.069              |
| SZx   | Gemini | Expert       | **0.474**        | —                  |
|       | GPT    | Expert       | 0.126            | —                  |

> 💡 **Gemini显著优于GPT**，最高达5倍以上吞吐提升。

#### ➤ Prompt Sensitivity 分析
- **Gemini高度依赖prompt质量**：从User → Expert，性能提升可达5×
- **GPT相对稳定**：不同prompt tier下性能波动小于5%，表现出更强的“prompt robustness”

#### ➤ Reasoning Iterations vs. Performance
- GPT平均使用更多迭代（20–41次），但性能仍远低于Gemini
- Gemini可在较少迭代（15–25次）中达到高吞吐
- **结论：更多推理步数 ≠ 更高性能优化**，多数额外迭代用于调试而非带宽关键优化

---

### ❌ Cerebras WSE-3 结果（Runnability为主）

| Model | Prompt Tier | Task | Executable? | Execution Time (s) | Reasoning Iterations |
|--------|------------|------|-------------|----------------------|------------------------|
| GPT-5.1-Codex | All tiers | SZp/SZx | ✅ Yes | **134 s** | 74–100+ |
| Gemini-2.5-pro | All tiers | SZp/SZx | ❌ No | **0+ s**（失败） | 9–100 |

> ⚠️ **Gemini在Cerebras上完全无法生成可运行程序**

#### 失败原因分析：
- Host-to-PE数据传输错误
- 违反静态空间映射规则
- 缺乏显式邻居通信逻辑
- Kernel stall 或零执行时间报告

> 尽管Gemini在GPU上表现优异，但在Cerebras这种需要**显式布局与通信推理**的架构上彻底失效。

---

### 🔬 消融实验：优化轨迹分析（Step-by-Step）

#### ➤ Cu-SZx 优化路径（Table II）
- 早期步骤（Step 2–5）带来最大收益：
  - GPU并行化
  - 移除Host-GPU往返
  - Kernel fusion
- 后期微调（Step 7–12）出现非单调性：
  - 如Step 6因移除XOR前缀导致压缩比骤降
  - Step 8~9引入chunk pipeline反而降低性能

#### ➤ Cu-SZp 优化路径（Table III）
- 初期改进有效（GPU bit pack/unpack, kernel fusion）
- 后期尝试warp-level ballot、vectorized write等战术优化时频繁回归：
  - Step 7: 引入不必要的同步开销
  - Step 10: 存储未对齐导致store efficiency下降

> 🧠 **LLM擅长战略级优化（high-impact bottlenecks removal）**  
> 🛠️ **但在战术级CUDA craft（bit-packing细节）上易出错**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **存在显著的跨架构性能鸿沟（Cross-Architecture Divergence）**
   - 在GPU上表现优秀的LLM（如Gemini）在Cerebras上可能完全失败
   - 成功不能跨平台迁移：**GPU优化经验不适用于PE-centric spatial架构**

2. **LLM模型之间存在根本性差异**
   - **Gemini**：高潜力但**高度依赖prompt质量**，适合专家引导下的高性能优化
   - **GPT**：性能较低但**prompt鲁棒性强**，且在Cerebras上能稳定生成可运行代码

3. **kernel模块化程度影响优化成功率**
   - **SZx（模块化）** > **SZp（bit-level紧耦合）**
   - 结构依赖性强的pipeline更难被LLM逐步优化

4. **优化过程是非单调且脆弱的**
   - 微小改动可能导致性能大幅回退
   - 后期bit-packing优化极易破坏memory coalescing或增加sync开销

5. **迭代次数 ≠ 优化质量**
   - 更多推理步主要用于修复编译/运行错误，而非发现高性能优化方案

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| 单agent设计 | 未探索multi-agent协作（如translator + optimizer + validator） |
| 固定prompt tier | 未动态调整提示策略或引入外部检索增强 |
| 仅两个kernel | SZp/SZx虽具代表性，但仍属特定类别 |
| 无profiling-guided feedback | 当前反馈仅来自运行结果，未接入Nsight等性能剖析工具 |

---

### 🔮 未来工作方向

1. **扩展至更多HPC kernels与硬件平台**
   - 如AMD GPU、Intel Ponte Vecchio、Tenstorrent等
   - 探索LLM在不同ISA和内存层次下的泛化能力

2. **构建hybrid agent-analyzer workflow**
   - 将LLM生成与性能剖析工具（如Nsight, Roofline）结合
   - 实现“生成 → 测评 → 反馈瓶颈 → 再优化”的闭环

3. **开发面向spatial architecture的专用prompt engineering策略**
   - 显式引导LLM考虑layout、communication、static scheduling

4. **引入multi-agent协同机制**
   - 分离职责：一个负责功能正确性，另一个专注性能优化

5. **建立HPC-oriented LLM evaluation benchmark**
   - 包含throughput、energy efficiency、portability等多个维度
   - 推动社区标准化评估标准

---

> 📌 **一句话总结**：  
> 当前LLM coding agents在HPC场景下面临“性能与鲁棒性不可兼得”、“跨架构迁移困难”的核心挑战；其优化能力受模型选择、prompt设计、kernel结构和目标架构的共同制约，需从单一代码生成转向**性能感知、架构适配、反馈驱动的智能协同开发范式**。

</details>

---

### 11. [Memory-Efficient Meta-Reinforcement Learning for Adaptive Safety-Critical Control in Adversarial Spacecraft Proximity Operations](https://arxiv.org/abs/2606.17414)

**Authors**: Alejandro Posadas-Nava, Richard Linares, Minduli Wijayatunga  
**Category**: cs.LG  
**Published**: 2026-06-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.17414v1  

#### Abstract
Autonomous spacecraft rendezvous and proximity operations (RPO) require controllers that guarantee safety under thrust constraints while minimizing fuel expenditure. Input-constrained control barrier functions (ICCBFs) provide a control method for nonlinear systems with actuation constraints that co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Memory-Efficient Meta-Reinforcement Learning for Adaptive Safety-Critical Control in Adversarial Spacecraft Proximity Operations

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文聚焦于**航天器近距离操作（Spacecraft Proximity Operations, RPO）中的安全关键控制问题**，特别是在存在以下挑战的情况下：
- **推力约束（Input constraints）**
- **模型不确定性（Model uncertainty）**
- **对抗性目标行为（Adversarial target behavior）**
- **有限的星载计算资源**

传统方法如基于最优控制或MPC的方法在面对未建模动态和非合作目标时缺乏鲁棒性，而标准的强化学习（RL）又无法提供形式化的安全性保证。

### 提出的新方法或新思路
本文提出并系统评估了一种**基于Meta-Reinforcement Learning（Meta-RL）的自适应输入约束控制屏障函数（ICCBF）框架**，其核心创新在于：
- 将ICCBF中的`class-KC`函数层次结构参数化，并通过Meta-RL进行端到端学习。
- 引入**记忆机制（recurrent policy）**，使控制器能够从观测历史中隐式地识别隐藏物理参数（如质量、推力上限等），实现在线自适应。
- 首次将**Selective State Space Model（Mamba）** 引入航天器安全控制领域，作为LSTM和GRU的替代架构，以提升序列建模效率与性能。
- 构建了**对抗性测试场景**（adversarial docking & inspection），验证控制器在主动规避或干扰下的鲁棒性。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **安全性** | 继承ICCBF的形式化安全证书，确保前向不变性（forward invariance）。 |
| **适应性** | Meta-RL策略能在线适应未知/变化的动力学参数和环境扰动。 |
| **燃料效率** | 学习得到的非贪婪`class-KC`函数减少了保守性，显著降低Δv消耗。 |
| **实时性** | 使用轻量级QP求解器 + 高效序列模型（如Mamba），适合星载部署。 |

---

## 2. 核心实验方法和设置

### 使用的测试案例（Test Cases）
论文采用三个典型RPO任务进行验证，均考虑状态、推力和参数不确定性：

1. **Cruise Control（一维巡航控制）**  
   - 非RPO教学示例，用于初步验证。
2. **Cooperative Docking（二维协同对接）**  
   - 追踪航天器需进入旋转目标的视线锥内完成对接。
3. **Cooperative Inspection（三维协同巡检）**  
   - 围绕球形目标飞行，检查表面点，受限于keep-in/out zones和太阳遮挡。

此外还设计了两个**对抗性变体**：
- **Adversarial Docking**：目标主动调整角速度以最小化安全裕度。
- **Adversarial Inspection**：目标施加径向脉冲推开追踪器。

### 实验设置
- **Meta-RL训练配置**：
  - Agent输出ICCBF的`class-KC`参数 $\alpha_k$ 和CLF增益 $c_{v,k}$。
  - 所有策略结合一个**凸二次规划（QP）** 在运行时生成最终控制指令。
- **网络架构对比**：
  - LSTM（长短期记忆）
  - GRU（门控循环单元）
  - Mamba2（选择性状态空间模型）
- **训练算法对比**：
  - PPO（On-policy, Proximal Policy Optimization）
  - SAC（Off-policy, Soft Actor-Critic）
- **总组合数**：3种架构 × 2种算法 = **6种配置**

### 评估指标
| 指标 | 定义 |
|------|------|
| **Safety (%)** | 整个任务周期内始终满足所有安全约束的比例。 |
| **Fuel / Control Effort** | 总Δv或$\int \|u(t)\| dt$，衡量燃料消耗。 |
| **Task Completion (%)** | 对接成功率 或 最终覆盖率（inspection）。 |
| **Time of Flight (TOF)** | 到达目标所需时间。 |

### 基线方法对比
本文不直接比较外部基线，而是通过**内部消融实验**进行横向对比：
- 不同recurrent架构之间的性能差异（LSTM vs GRU vs Mamba）
- 不同训练算法的影响（PPO vs SAC）
- 在合作与对抗场景下的泛化能力

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自Table 3）

#### ✅ Cooperative Scenarios

| 方法 | Docking 成功率 | 安全率 | Δv (m/s) |
|------|----------------|--------|----------|
| LSTM+PPO | 0.0% | 98.3% | 7.64 ± 2.97 |
| GRU+PPO | 0.0% | 98.3% | 6.90 ± 2.47 |
| **Mamba2+PPO** | **95.0%** | **97.9%** | **6.26 ± 1.70** |
| LSTM+SAC | 0.0% | 98.2% | 5.23 ± 1.05 |
| Mamba2+SAC | 48.6% | 98.3% | 6.06 ± 1.88 |

> 📌 **Mamba2+PPO是唯一成功学会完成对接任务且保持高安全性的组合。**

| 方法 | Inspection 覆盖率 | 安全率 | Δv (m/s) |
|------|--------------------|--------|----------|
| LSTM+PPO | 97.7% | 94.4% | 8.31 ± 4.30 |
| GRU+PPO | 98.1% | 67.8% | 9.69 ± 4.24 |
| **Mamba2+PPO** | **99.6%** | **99.4%** | **6.59 ± 1.38** |
| LSTM+SAC | 76.5% | 6.4% | 20.1 ± 11.0 |
| GRU+SAC | 37.4% | 0.0% | 20.7 ± 21.9 |

> 📌 **SAC全面崩溃；Mamba2+PPO在覆盖、安全、节能三方面均最优。**

#### ✅ Adversarial Scenarios（仅PPO参与）

| 方法 | Adversarial Docking 成功率 | 安全率 | Δv (m/s) |
|------|----------------------------|--------|----------|
| LSTM+PPO | 95.1% | 95.6% | 6.77 ± 2.92 |
| GRU+PPO | 74.3% | 88.3% | 7.24 ± 2.68 |
| **Mamba2+PPO** | **95.2%** | **95.4%** | **5.18 ± 1.87** |

| 方法 | Adversarial Inspection 覆盖率 | 安全率 | Δv (m/s) |
|------|-------------------------------|--------|----------|
| LSTM+PPO | 98.4% | 95.6% | 7.79 ± 2.80 |
| GRU+PPO | 98.1% | 67.8% | 9.69 ± 4.24 |
| **Mamba2+PPO** | **99.4%** | **99.0%** | **5.29 ± 1.37** |

> 📌 **Mamba2+PPO在对抗环境下仍保持最高成功率、最高安全性和最低燃料消耗（比LSTM节省约30–45% Δv）。**

---

## 4. 关键结论和发现

### 主要发现
1. **架构选择至关重要**：
   - **Mamba2 + PPO 是最佳组合**，在所有任务中表现最均衡且领先。
   - Mamba的线性递归结构缓解了梯度消失问题，使其能更有效地利用完整观测历史来推断隐藏参数和预测对手行为。

2. **训练算法影响巨大**：
   - **PPO 明显优于 SAC**，尤其是在复杂任务中。
   - SAC因off-policy学习导致value估计过时（stale），加上最大熵目标引入过多探索噪声，在紧致的安全边界附近频繁违规，造成“安全崩溃”。

3. **任务难度放大性能差距**：
   - 在简单任务（如Cruise Control）中各模型表现接近；
   - 但在复杂任务（如Inspection）和对抗场景下，Mamba2+PPO的优势显著扩大。

4. **对抗鲁棒性验证成功**：
   - 即使目标主动规避或施加扰动，Mamba2+PPO仍能维持高任务完成率与安全性，证明其具备强适应能力。

### 方法的局限性
- **依赖仿真环境的真实性**：尽管进行了domain randomization，但真实轨道动力学、传感器延迟等因素尚未完全建模。
- **SAC未能适配安全关键任务**：off-policy方法在此类非平稳、稀疏奖励环境中表现不佳，需专门改进。
- **Mamba的可解释性较低**：相比传统控制器，神经网络参数缺乏明确物理意义，可能影响工程可信度。

### 未来工作方向
1. 改进off-policy算法（如SAC）以适应由learned safety filter引起的非平稳环境。
2. 将该框架部署至**硬件在环（HIL）测试平台**，验证其实时性和可靠性。
3. 探索**multi-agent MARL**框架应对多目标对抗场景。
4. 结合**symbolic regression**尝试从learned $\alpha_k$ 中提取解析表达式，增强可解释性。

--- 

> 🔚 **总结一句话**：  
> 本研究表明，**Mamba2 + PPO** 是当前实现**高效、安全、自适应**的航天器近距离操作控制的最佳Meta-RL方案，在合作与对抗场景下均展现出卓越的性能与鲁棒性。

</details>

---

### 12. [Toward Controllable Catalyst Inverse Design via Large-Scale Autoregressive Pretraining](https://arxiv.org/abs/2606.17445)

**Authors**: Dong Hyeon Mok, Jonggeol Na, Seoin Back  
**Category**: cs.LG  
**Published**: 2026-06-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.17445v1  

#### Abstract
Inverse design of heterogeneous catalysts remains challenging because catalyst surfaces exhibit substantial structural complexity with coupled surface-adsorbate interactions across a vast chemical space that is difficult to explore efficiently through conventional screening alone. Although machine l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Toward Controllable Catalyst Inverse Design via Large-Scale Autoregressive Pretraining*

## 1. 论文的主要贡献和创新点

### 解决的问题
异质催化剂的**逆向设计**（inverse design）长期面临挑战，主要原因在于：
- 催化剂表面具有高度复杂的**结构多样性**；
- 表面-吸附物（surface-adsorbate）相互作用耦合性强；
- 化学空间巨大，传统高通量筛选（high-throughput screening）效率随搜索空间扩大而急剧下降。

尽管已有基于机器学习的高通量筛选方法加速了催化剂发现，但其本质仍是“先生成后筛选”，在大规模探索中效率受限。此外，现有的生成模型（如CatGPT）多为**无条件生成模型**，无法直接根据目标性质（如吸附能、组分等）生成催化剂，需额外任务特定微调，实用性受限。

### 提出的新方法与创新
本文提出了一种**条件式催化剂生成模型**，基于 **Generative Pretrained Transformer (GPT)** 架构，并引入以下关键创新：

1. **数值嵌入层（numerical embedding layer）**  
   首次将连续数值属性（如binding energy）通过可学习的线性投影嵌入到GPT的self-attention机制中，实现对**连续属性的显式条件控制**：
   $$
   q_i = z_i W_Q + z_c W_Q^c,\quad k_i = z_i W_K + z_c W_K^c,\quad v_i = z_i W_V + z_c W_V^c
   $$
   其中 $z_c$ 是由binding energy映射得到的condition embedding。

2. **统一的自回归框架（autoregressive framework）**  
   支持同时对**类别型**（categorical，如adsorbate type、composition）和**连续型**（continuous，如binding energy）属性进行联合条件生成，无需任务特定微调。

3. **大规模预训练 + 微调策略**  
   - 在 **1.33亿** 条催化剂结构（OC20-S2EF）上进行大规模预训练，建立通用化学理解；
   - 再在约 **46万优化结构**（OC20-IS2RE）上微调，引导生成更接近基态的稳定结构。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可控性** | 支持直接条件生成，无需为每个目标重新微调，真正实现“按需设计” |
| **效率** | 显著提升目标催化剂的筛选效率（1.5–4倍），减少无效计算 |
| **泛化能力** | 预训练模型可作为**foundation model**，在少量数据下快速适配新领域（如氧化物、单原子催化剂） |
| **架构兼容性** | 在标准GPT架构基础上扩展，易于集成与部署 |

---

## 2. 核心实验方法和设置

### 数据集
| 数据集 | 描述 | 规模 | 用途 |
|-------|------|------|------|
| **OC20-S2EF** | 包含DFT松弛轨迹中的非优化结构及其binding energy | 1.33亿条 | 大规模预训练 |
| **OC20-IS2RE** | 已完全优化的催化剂结构 | ~46万条 | 微调阶段，提升结构合理性 |
| **OOD测试集** | 用于评估跨域泛化能力：<br>• **OOD-Cat**: 新合金表面<br>• **OOD-Oxide**: 氧化物表面（来自OC22）<br>• **OOD-SAC**: 单原子催化剂（自建） | 数千至数万 | 评估模型作为foundation model的能力 |

### 实验设置
- **模型架构**：基于GPT-2，12层，12头注意力，隐藏维度768（133M-FT版本）
- **训练流程**：
  1. **预训练**：在133M结构上训练600k步（8×H200 GPU）
  2. **微调**：在460k优化结构上微调10轮
- **输入表示**：将催化剂结构编码为token序列，格式如下：
  ```
  [A][M1][v1]...[G][H][l1][l2][l3][θ1][θ2][θ3][e1][x1][y1][z1]...
  ```
  其中A为adsorbate，M为composition，G为空间群，H为Miller指数，坐标采用固定精度字符串离散化。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Structural Validity (S. Val)** | 最小原子间距 ≥ 0.5 Å 且晶胞体积 > 1.0 Å³ |
| **Optimization Validity (O. Val)** | 使用MLFF（UMA）优化后，最大原子力 < 0.05 eV/Å（200步内收敛） |
| **Match Rate (分类)** | 生成结构的adsorbate type / composition 与条件token一致的比例 |
| **Joint Match Rate** | 同时匹配adsorbate和composition的比例 |
| **Binding Energy Match Rate** | 生成结构的binding energy落在目标值±0.2 eV范围内的比例 |
| **Uniqueness & Novelty** | 基于EquiformerV2提取的latent space中最近邻距离判断（避免真空层等伪差异干扰） |

### 基线方法对比
- **CatGPT-2M / CatGPT-2M-FT**：仅在200万数据上训练/微调的同类模型
- **CatFlow**：基于flow-matching的生成模型
- **从零训练模型（Scratch）**：不进行预训练，直接在OOD数据上训练

---

## 3. 主要实验结果和性能指标

### 关键性能数据（CatGPT-133M-FT）
| 指标 | 性能 |
|------|------|
| **Structural Validity** | **98.05%** |
| **Optimization Validity** | **95.39%** |
| **Adsorbate Type Match Rate** | **98.33%** |
| **Composition Match Rate** | **92.93%** |
| **Joint Match Rate (adsorbate + composition)** | **92.93%** |
| **Binding Energy Match Rate** | **~20%**（相对baseline提升4倍） |

> 注：baseline指原始OC20训练分布下的随机生成match率约为5%

### 与基线方法对比
#### （1）生成质量对比（Table 1）
| Model | S. Val | O. Val |
|-------|--------|--------|
| CatGPT-2M | 0.9123 | 0.5691 |
| CatGPT-2M-FT | 0.9147 | 0.8313 |
| **CatGPT-133M-FT** | **0.9805** | **0.9539** |
| CatFlow | 0.9733 | 0.6970 |

→ 大规模预训练显著提升结构合理性和可优化性。

#### （2）条件生成能力对比（Table 2）
| Match Rate | 2M-FT | **133M-FT** |
|------------|--------|-------------|
| Adsorbate type | 0.9811 | 0.9833 |
| Composition | 0.2194 | **0.9293** |
| **Both** | 0.2153 | **0.9293** |

→ **133M-FT的composition匹配率是2M-FT的4倍以上**，表明大规模预训练极大增强了对组成-结构关系的理解。

#### （3）跨域适应能力（OOD微调，Table 2）
| Base Model | Dataset | S. Val | O. Val | Both Match |
|-----------|---------|--------|--------|------------|
| 133M-FT | Cat | 0.8915 | 0.8368 | **0.7378** |
| 2M-FT | Cat | 0.9362 | 0.8995 | 0.2163 |
| Scratch | Cat | 0.0105 | 0.6000 | 0.2105 |
| 133M-FT | Oxide | 0.6686 | 0.7226 | **0.2051** |
| 2M-FT | Oxide | 0.7717 | 0.7452 | 0.0433 |
| Scratch | Oxide | 0.0095 | 0.5429 | 0.0288 |
| 133M-FT | SAC | 0.8840 | 0.9471 | **0.0615** |
| 2M-FT | SAC | 0.9961 | 0.9697 | 0.0000 |
| Scratch | SAC | 0.0000 | - | - |

→ 尽管2M-FT在部分OOD任务上structural validity更高，但**133M-FT在条件生成准确率上全面碾压**，尤其在SAC任务上唯一能有效生成符合条件的结构。

### 消融实验结果
- **预训练规模影响**：133M预训练相比2M显著提升composition匹配率（>4倍），验证了**大规模数据对捕捉复杂化学规律的重要性**。
- **微调必要性**：未微调模型虽能生成语法正确的结构，但优化有效性低（O. Val仅~57% vs 微调后>95%），说明微调对物理合理性至关重要。
- **数值条件有效性**：binding energy分布随目标值系统偏移（见Figure 2b），证明数值嵌入机制有效引导生成方向。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **大规模自回归预训练可建立有效的催化剂“化学语言”模型**，能够学习复杂的结构-性质关系。
2. ✅ 引入**数值嵌入层**使GPT架构首次支持对连续属性（如binding energy）的条件生成，突破原有离散token限制。
3. ✅ 所提模型可在无需额外微调的情况下，**直接生成满足目标adsorbate、composition和binding energy的催化剂**，实现高效逆向设计。
4. ✅ 模型具备强**跨域迁移能力**，可作为**catalyst foundation model**，在极少量数据下快速适配新催化体系（如氧化物、SAC）。
5. ✅ 应用于HER和ORR反应时，筛选效率提升**1.5–4倍**，验证了其在实际反应导向催化剂发现中的潜力。

### 方法的局限性
1. ❌ **词汇表限制**：模型无法生成预训练中未见过的元素、adsorbate或空间群（受限于token vocabulary）。
2. ❌ **稳定性评估困难**：当前缺乏可靠方法从slab结构反推体相（bulk reconstruction），难以准确评估热力学稳定性。
3. ❌ **评价指标模糊**：基于latent space的距离度量（如uniqueness/novelty）仍缺乏明确阈值，存在主观性。
4. ❌ **极端OOD场景适应有限**：当目标域与训练分布差距过大（如SAC），条件生成准确率仍较低（~6%）。

### 未来工作方向
1. **扩展token表示能力**：结合元素特征向量（如atomic number, electronegativity）增强对新元素的泛化能力。
2. **发展更鲁棒的稳定性评估方法**：利用生成的bulk composition和Miller index辅助体相重构与能量比较。
3. **构建真正的多目标生成框架**：当前一次只能设定一个adsorbate和一个binding energy，未来需支持多个中间体联合条件生成。
4. **推动实验验证闭环**：将生成-预测-合成-测试形成自动化pipeline，加速真实新材料发现。

---

> **总结**：该工作通过**大规模预训练 + 数值条件嵌入**，实现了对异质催化剂的**可控逆向生成**，不仅在性能上大幅超越现有方法，更提出了“**催化剂基础模型**（catalyst foundation model）”的新范式，为AI驱动的材料逆向设计提供了可扩展、可迁移的实用路径。

</details>

---

### 13. [Learning to Refine Hidden States for Reliable LLM Reasoning](https://arxiv.org/abs/2606.17524)

**Authors**: Chia-Hsuan Hsu, Jui-Ming Yao  
**Category**: cs.LG  
**Published**: 2026-06-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.17524v1  

#### Abstract
Large language models show strong reasoning ability, but their internal reasoning process can remain unstable in complex multi-step settings, where early hidden-state errors may propagate to incorrect predictions. We propose ReLAR, a reinforcement-guided latent refinement framework that iteratively ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning to Refine Hidden States for Reliable LLM Reasoning*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）虽然具备强大的推理能力，但在复杂的多步推理任务中，其内部推理过程可能不稳定。早期隐藏状态（hidden state）中的微小错误会逐步传播，最终导致错误预测。传统方法如 **Chain-of-Thought (CoT)** 虽能提升可解释性，但仅在输出文本层面操作，无法直接调控模型的**内部推理动态**，且生成长推理链带来显著的推理延迟。

此外，现有隐空间（latent space）干预方法多为静态或单步操作，缺乏对推理路径的**迭代优化**和**自适应控制**，难以应对高风险场景（如医疗诊断）中对稳定性和可靠性的要求。

### 提出了什么新方法或新思路
本文提出 **ReLAR**（Reinforcement-Guided Latent Refinement），一种基于强化学习引导的**隐状态迭代精炼框架**，在解码前对模型的隐藏表示进行可控的多步优化。

核心思想是将推理视为一个**潜层过程**，通过以下机制实现：
- **隐状态精炼**：在不生成任何中间 token 的情况下，迭代更新隐藏状态 $ h_t $ 和一个紧凑的**推理状态** $ s_t $。
- **双控制器架构**：
  - **深度控制器 (depth controller)**：从初始推理状态 $ s_0 $ 预测应执行的精炼步数 $ T $，实现输入相关的计算分配。
  - **动作控制器 (action controller)**：在每一步预测精炼方向 $ v_t $ 和更新幅度参数 $ \gamma_t, \beta_t $，决定如何修改隐藏状态。
- **强化学习训练**：控制器通过**策略梯度**（policy gradient）目标进行训练，奖励信号来自每步精炼后对目标输出的**似然增益**（likelihood gain），减去计算成本。

### 相比现有方法的优势
- **更可靠的推理**：直接在隐空间修正中间表示，防止早期错误传播，提升推理稳定性。
- **更高的效率**：无需生成显式的 CoT 文本，推理开销远低于 CoT 和 Self-Consistency (SC-CoT) 等方法（见 Table 4）。
- **自适应计算**：根据输入难度动态调整精炼步数，避免对简单样本过度计算。
- **无需暴露中间步骤**：保持“隐式推理”优势，同时提供比标准 SFT 更强的控制能力。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖多个领域和任务类型：
- **医疗推理**：`PubMedQA`（生物医学问答，三分类）
- **数学推理**：`GSM8K`（小学数学题）、`GSM-Hard`（更具挑战性的变体）
- **多跳问答**：`HotpotQA`（需跨文档推理）
- **开放生成**：`CommonGen`（概念到句子生成）、`WritingPrompts`（故事生成）

### 实验设置和评估指标
- **骨干模型**：LLaMA-1.1B, Gemma-2B, Qwen-3B（消融实验统一用 Gemma-2B）
- **评估模式**：0-shot 和 5-shot 设置
- **评估指标**：
  - 分类任务：Accuracy, Macro-F1
  - 数学任务：Accuracy, pass@5
  - 开放生成：BERTScore, ROUGE-L

### 基线方法对比
- **通用 LLM 基线**：LLaMA-2-7B, Mistral-7B, Gemma-7B, Llama-3-8B-Instruct, Qwen2.5-7B 等
- **医疗专用 LLM 基线**：Med42-Mistral, Med42-Llama3, MedGemma-4B, Qwen2.5-Med-7B
- **推理策略对比**：SFT-only, ICL, CoT, SC-CoT

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Table 2）

| 数据集 | 方法 | 准确率 (0-shot) | F1 / 其他 |
|--------|------|----------------|----------|
| **PubMedQA** | **ReLAR (ours)** | **77.67%** | **72.54** |
| | Llama-3-8B-Instruct | 68.47% | 62.83 |
| | MedGemma-4B | 72.45% | 68.52 |
| **GSM-Hard** | **ReLAR (ours)** | **45.02%** | **47.58 (pass@5)** |
| | Qwen2.5-7B | 39.83% | 47.64 |
| **HotpotQA** | **ReLAR (ours)** | **57.50%** | **75.23** |
| | Llama-3-8B-Instruct | 48.83% | 63.47 |

> ✅ 在 PubMedQA 上超越所有 7B 级模型；在 GSM-Hard 和 HotpotQA 上达到 SOTA。

#### 开放生成性能（Table 2）
| 任务 | 指标 | ReLAR (ours) | 最佳基线 |
|------|------|-------------|---------|
| CommonGen | BERTScore | **0.934** | 0.921 (Mistral-7B-Instruct) |
| | ROUGE-L | **38.92** | 35.67 |
| WritingPrompts | BERTScore | **0.878** | 0.871 |
| | ROUGE-L | **11.47** | 9.28 |

> ✅ 显著提升生成质量，表明隐状态精炼不仅利于精确推理，也改善自由生成的语义一致性。

### 与基线方法的对比结果
- **相比 CoT 类方法**：ReLAR 在 PubMedQA 上准确率高于 CoT（64.68%）和 SC-CoT（72.83%），且**推理时间仅为 CoT 的 1/65，SC-CoT 的 1/117**（见 Figure 5 和 Table 4）。
- **相比大模型基线**：尽管骨干模型较小（2-3B vs 7-8B），ReLAR 在多个任务上仍超越更大模型，说明**方法本身带来的增益显著**。

### 消融实验结果（Table 3）
| 消融设置 | PubMedQA Acc. | GSM8K Acc. | GSM-Hard Acc. |
|----------|---------------|------------|---------------|
| No Refinement (SFT only) | 55.02 | 48.52 | 29.14 |
| Static Refinement (固定步数) | 73.01 | 63.84 | 36.52 |
| + Adaptive Depth (无方向) | 76.72 | 66.72 | 42.28 |
| + Adaptive Direction (无深度) | 68.90 | 60.21 | 33.85 |
| **Ours (Adaptive Depth + Direction)** | **77.67** | **68.45** | **45.02** |

> 🔍 结论：
> - 隐状态精炼本身带来巨大提升（+22+ pts）
> - **自适应深度**比自适应方向更重要
> - 两者结合效果最佳，证明双控制器设计的有效性

---

## 4. 关键结论和发现

### 主要发现
1. **隐状态精炼能显著提升 LLM 推理可靠性**：通过在解码前迭代优化隐藏表示，有效抑制错误传播，提升多步推理的稳定性。
2. **强化学习可有效指导隐空间推理**：基于似然增益的奖励信号能成功训练控制器，实现对精炼方向和深度的自适应控制。
3. **高效优于显式推理**：ReLAR 在性能上优于或媲美 CoT/SC-CoT，但推理开销接近标准 SFT，为高时效场景（如临床决策）提供了实用方案。
4. **泛化能力强**：在医疗、数学、多跳问答和开放生成任务上均取得一致提升，表明方法具有广泛适用性。

### 方法的局限性
1. **依赖监督信号**：RL 训练需要每步的 ground-truth likelihood，限制其在无监督或弱监督场景的应用。
2. **模型规模较小**：实验使用的骨干模型（1.1B–3B）小于主流 7B+ 模型，虽性能领先，但直接比较需谨慎。
3. **不可解释性**：精炼过程完全在隐空间进行，缺乏像 CoT 那样的可读推理轨迹，可能影响在高风险领域（如医疗）的信任建立。
4. **额外训练复杂度**：引入控制器和 RL 训练增加了实现和调优难度。

### 未来工作方向
- 探索**无监督或弱监督**的奖励机制，减少对标注数据的依赖。
- 将 ReLAR 扩展到**视觉-语言模型**或多模态推理任务。
- 设计**可解释的隐状态监控工具**，增强对精炼过程的理解和信任。
- 研究更高效的控制器架构，降低训练和部署成本。

---

> **总结**：ReLAR 提出了一种新颖的“**隐式但可控**”的推理范式，通过强化学习引导的隐状态迭代精炼，在不牺牲效率的前提下显著提升了 LLM 的推理可靠性，为构建安全、高效的 AI 决策系统提供了重要思路。

</details>

---

### 14. [Distributed General-Purpose Agent Networks: Architecture, Key Mechanisms, and Prototypes](https://arxiv.org/abs/2606.17368)

**Authors**: Shengli Zhang, Deen Ma, Zibin Lin, Taotao Wang  
**Category**: cs.AI  
**Published**: 2026-06-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.17368v1  

#### Abstract
Large language models have accelerated the transition from passive conversational assistants to autonomous agents that can understand goals, plan actions, invoke tools, and execute multi-step tasks. Yet the capability of a single agent remains constrained by its local data, tool permissions, runtime...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Distributed General-Purpose Agent Networks: Architecture, Key Mechanisms, and Prototypes**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本文针对当前 **LLM agents** 在执行复杂任务时面临的**能力边界限制**问题，提出了一种新的系统架构——**分布式通用 Agent 网络（Distributed General-Purpose Agent Networks）**。单个 agent 受限于本地数据、工具权限、运行环境和治理边界，难以独立完成开放、动态、多步骤的任务协作。

传统方法如集中式多智能体平台或简单的 P2P 网络无法有效支持语义驱动的协作，因为它们缺乏对以下三方面的系统级整合：
- **语义发现（Semantic Discovery）**：如何在大规模网络中高效传播任务意图并找到合适的协作者；
- **可信治理（Trustworthy Governance）**：如何防止身份伪造、跨域作恶和责任逃避；
- **自动化机制设计（Automated Mechanism Design）**：如何为开放任务自动生成激励相容的合作规则。

---

### **提出了什么新方法或新思路**

本文提出一个以 **Protocol Adaptation Layer（协议适配层）** 为核心的三层架构，并围绕该层构建三个关键技术路线：

#### **1. 分层架构设计**
- **General-Purpose Agent Layer**：负责自然语言理解、任务规划、工具调用等高层语义处理。
- **Protocol Adaptation Layer（核心创新）**：将上层语义转化为底层网络操作（广播、连接、验证、协商、执行），是实现“语义到行为”转换的关键控制平面。
- **P2P Network Stack**：基于 LibP2P 构建去中心化通信基础设施。

> ✅ 创新点：首次将 **LLM-driven semantic reasoning** 与 **P2P network control** 耦合在一个反馈闭环中，形成可迭代的协作系统。

#### **2. 三大核心技术机制**

| 模块 | 技术方案 | 创新点 |
|------|---------|--------|
| **Collaborator Discovery** | Two-stage Bodyless Gossip + Sequential Logs | 提出轻量级摘要传播 + 按需负载拉取机制，降低冗余流量；通过主题域内的因果一致性日志维护事件顺序语义 |
| **Cooperation Governance** | BAID Identity + MG-EigenTrust Reputation | BAID 实现用户-代码-链上身份的可验证绑定；MG-EigenTrust 支持跨主题声誉传播与惩罚机制，防“洗白攻击” |
| **Task Execution** | Semantic-Gradient Stackelberg Loop | 基于 LLM 的语义归因反馈生成合作机制，结合博弈论实现激励兼容（IC）、个体理性（IR）和鲁棒性 |

---

### **相比现有方法的优势**

| 维度 | 现有方法局限 | 本方案优势 |
|------|-------------|-----------|
| **发现机制** | 静态索引（DHT/Registry）易过期；全网广播冗余高 | 动态语义过滤 + 主题窄播，在保持开放性的同时减少通信开销 |
| **信任机制** | 单一全局声誉易被滥用；静态身份无法追踪演化 | 多层耦合声誉模型 + 版本化身份链，支持跨域风险识别与经济惩罚 |
| **机制设计** | 手工规则难适应新攻击；数值优化不适用于文本规则 | 利用 LLM 进行语义归因，自动修复漏洞，支持非结构化规则空间搜索 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 无真实外部数据集，采用**模拟器生成合成数据流**。
- 模拟场景包括：`code_generation`, `code_review_security`, `devops_tool_execution`, `data_analysis`, `creative_writing` 五个 topic。
- 请求模式：每个 epoch 每 topic 发起 50 次查询，每次从活跃节点中采样 8 个候选提供者。

### **实验设置**
- **节点规模**：100~200 个 agent 节点。
- **随机种子**：30 次独立运行取均值。
- **生命周期**：共 50 个 reputation 更新 epoch。
- **攻击注入**：恶意节点先在 `code_generation` 积累声誉，再迁移到 `code_review_security` 执行低质量服务（disguise-collusion attack）。
- **评估周期**：分为 warm-up 和 attack phase。

### **评估指标**
| 类别 | 指标 |
|------|------|
| **安全性** | Attack Success Rate（攻击成功率）、Attacker ROI（燃烧后收益） |
| **有效性** | Reputation Convergence、Cold-start Initial Reputation |
| **效率** | Control Entries per Epoch（控制面消息量）、Latency/P95 |
| **鲁棒性** | False Positive/Negative Detection Rate、Regression Pass Rate |
| **机制质量** | IC Violation、IR Violation、System Loss |

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| `Random/no trust` | 不使用声誉，随机选择协作者 |
| `Public EigenTrust` | 全局单一声誉层，含固定预信节点 |
| `Independent Topic EigenTrust` | 各 topic 独立维护声誉，无跨域反馈 |
| `Centralized Registry / Kademlia DHT` | 中心化目录 / 分布式哈希表用于发现 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **A. 协作者发现模块（Discovery）**
| 方法 | 成功率（200-node, 20% churn） | 每请求字节数（MB） | P95 延迟（ms） |
|------|-------------------------------|---------------------|----------------|
| Topic/OpenAgent | 1.000 | ~0.33 | < 3000 |
| Public Broadcast | 1.000 | ~1.00 | > 5000 |
| Centralized Registry | 0.875 | - | ~4500 |
| Kademlia DHT | 0.890 | - | ~5000 |

> 🔍 发现：**Topic/OpenAgent 比公共广播节省约 2/3 流量**，且在节点频繁上下线时仍能维持高成功率。

#### **B. 声誉治理模块（MG-EigenTrust）**
| 方法 | Attack Success | Burn-only ROI | 控制面条目/epoch |
|------|----------------|---------------|------------------|
| Random/no trust | 0.2176 | 3.3523 | 0 |
| Public EigenTrust | 0.3759 | 6.5174 | 285 |
| Independent Topic | 0.2160 | 3.3209 | 285 |
| **MG-EigenTrust** | **0.0291** | **-0.8022** | **45** |

> ✅ 结果表明：
- 公共声誉机制反而加剧了跨域声誉污染（attack success 更高）；
- MG-EigenTrust 将攻击成功率降低至 **2.91%**，并使攻击者净收益为负（ROI = -0.8）；
- 控制面通信量下降 **84.21%**（285 → 45 entries/epoch）。

#### **C. 冷启动表现**
| 方法 | 新诚实节点初始声誉 |
|------|--------------------|
| Independent Topic EigenTrust | 0.0019 |
| **MG-EigenTrust** | **0.0146** |

> 📌 利用跨主题可验证证据作为贝叶斯先验，显著提升新节点的信任起点。

#### **D. 消融实验结果（Ablation Study）**
| 变体 | Attack Success | Burn-only ROI | Detection Rate |
|------|----------------|---------------|----------------|
| Full MG-EigenTrust | 0.0291 | -0.8022 | 0.9509 |
| Without slashing | 0.3332 | 5.6637 | 0.0916 |
| Without consistency gate | 0.0290 | -0.7931 | - |
| Without semantic weight | 0.0285 | -0.7948 | - |

> 🔬 关键发现：
- **Slashing 是抑制攻击的核心机制**，移除后攻击成功率飙升；
- Consistency gate 和 semantic weighting 对当前攻击路径影响小，但在低相关迁移场景下更具价值。

#### **E. BAID 身份验证开销测试**
| 参数变化 | Proof Generation Time | Verification Time |
|--------|------------------------|------------------|
| Recursion Depth: 1→32 | ~39–48s（稳定） | ~67–75ms（稳定） |
| Payload Size: 1KB→16KB | ~45s → ~185s（显著上升） | ~67–69ms（几乎不变） |

> 💡 设计启示：**证明生成可在离线/批处理中进行，而第三方验证始终低延迟**，适合争议触发式审计。

---

## **4. 关键结论和发现**

### **主要发现**
1. **必须引入 Protocol Adaptation Layer** 来桥接语义层与网络层，否则无法实现真正的开放式协作。
2. **语义感知的两阶段传播机制（bodyless gossip）** 显著优于传统广播或静态索引，在动态环境中兼具效率与可靠性。
3. **单一全局声誉机制存在严重安全隐患**，容易导致跨域声誉套利；**MG-EigenTrust 通过动态预信+ slashing 实现负向激励**，有效遏制伪装协作攻击。
4. **BAID 身份绑定机制实现了用户-代码-责任的可验证锚定**，支持版本演化与分层验证，为问责提供技术基础。
5. **Semantic-gradient 机制设计范式可行**：利用 LLM 对攻击轨迹进行归因分析，指导规则修复，形成闭环进化。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **仿真级别证据为主** | 当前仅为 mechanism-level simulation 或 prototype overhead test，尚未部署端到端系统 |
| **依赖高质量 LLM 归因能力** | Semantic attribution 的效果受限于 LLM 推理与解释能力，可能存在误判 |
| **经济参数需人工校准** | 如 slashing ratio、punishment threshold 等需结合具体应用场景调整 |
| **高负载下的 zkVM 性能未知** | 完整零知识证明在大规模并发下的可行性有待验证 |

---

### **未来工作方向**
1. **扩展实验规模**：在更大规模网络中验证发现、声誉收敛与机制演化的稳定性。
2. **实现完整证据链**：构建从行为日志 → 审计证据 → 链上惩罚的全流程 pipeline。
3. **优化语义归因质量**：引入人类反馈（RLHF）或形式化验证增强 attribution 准确性。
4. **探索 Hierarchical Topic Structure**：将细粒度 topic 聚合成 hierarchy，进一步提升声誉管理效率。
5. **集成真实应用负载**：在 crowdsourcing、multi-hop QA、decentralized marketplace 等场景中实测性能。

---

> ✅ **总体评价**：  
> 本文提出了首个面向 **open-ended semantic collaboration** 的分布式 agent 网络系统框架，不仅具有理论深度，也提供了可落地的技术路径。其核心思想——**将语义、信任与机制纳入统一的反馈控制系统**——有望成为下一代去中心化 AI 生态的基础设施蓝图。

</details>

---

### 15. [DeepInsight: A Unified Evaluation Infrastructure Across the Physical AI Stack](https://arxiv.org/abs/2606.17574)

**Authors**: Siyi Li, Chunyu Sun, Jiahao Zhang, Yuchen Kang, Wuliang Wang, Yu Qiu, Rui Jiang, Haitao Cui, Jie Chen  
**Category**: cs.AI  
**Published**: 2026-06-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.17574v1  

#### Abstract
Evaluating a Physical AI stack spans operators that differ by more than three orders of magnitude -- from a single foundation-model decoding step to thousands of physics ticks of whole-body control -- varying orthogonally in modality, reward semantics, and resource profile. No existing framework spa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# DeepInsight: A Unified Evaluation Infrastructure Across the Physical AI Stack 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在 **Physical AI**（物理人工智能）系统中，从高层语义推理（System 2）、到感知运动策略执行（System 1），再到全身控制与稳定（System 0），整个堆栈（stack）涉及多种异构任务，其评估需求差异巨大：

- **Episode 长度** 跨越三个数量级：从单步 LLM 解码到数千次物理仿真步。
- **模态多样性**：文本、图像、音频、连续物理状态。
- **奖励语义不同**：精确匹配、模型判断、轨迹分析。
- **资源消耗模式各异**：GPU 推理、I/O 沙箱、并行物理仿真。

目前缺乏一个统一的评估框架来覆盖这一完整谱系。现有方案通常是多个独立的评估工具（harnesses）拼接而成，导致：

- 各层之间 **缺乏共享运行时（runtime）和评分机制**；
- 无法追踪跨层回归（cross-layer regression）；
- 新任务接入成本高，需大量代码重写。

---

### 提出的新方法与创新思路

作者提出 **DeepInsight**，一个面向全栈 Physical AI 的统一评估基础设施，其核心是通过 **三大抽象** 在异构性之上建立统一性：

#### ✅ 三大核心抽象（Three Key Abstractions）

| 抽象 | 功能 | 实现方式 |
|------|------|--------|
| **Task Abstraction** | 统一任务接口 | 所有任务实现 `reset(handle)` 和 `step(handle, action)` 接口，所有临时状态存储于 `per-episode handle` |
| **Resource Abstraction** | 解耦昂贵后端资源 | 将 LLM 推理和沙箱运行时封装为独立的 **Control Planes**，通过统一的 `acquire/release` 协议调用 |
| **Result Abstraction** | 统一日志与追踪 | 所有事件写入统一格式的 `TraceRecord`，具备全局唯一身份标识和因果链路 |

这三大抽象共同构成一个 **单一运行时（single runtime）**，支持从 LLM 解码到全身控制的端到端评估。

---

### 相比现有方法的优势

| 优势维度 | DeepInsight | 现有框架（如 lm-eval, Inspect AI, Isaac Lab） |
|---------|------------|------------------------------------------|
| **覆盖范围** | ✅ 全栈覆盖（System 2 → 0） | ❌ 仅覆盖单一层次（如仅 LLM 或仅仿真） |
| **跨层诊断能力** | ✅ 支持基于 trace 的跨层故障定位 | ❌ 各层日志孤立，无法追溯根源 |
| **扩展性** | ✅ 新任务主要通过配置接入 | ❌ 需要修改代码或重构 pipeline |
| **吞吐效率** | ✅ 异步流水线 + 阶段解耦 → 更高并发 | ❌ 多数为同步批处理，易受长尾任务阻塞 |
| **多节点扩展** | ✅ 近似线性横向扩展 | ❌ 多数不支持分布式执行 |

---

## 2. 核心实验方法和设置

### 使用的数据集

实验覆盖多个层级的代表性基准：

#### 🔹 System 2（语言/多模态模型）
- **Knowledge & Reasoning**: MMLU-Pro, MMLU-Redux, GPQA-Diamond, C-Eval, SuperGPQA
- **Math**: MATH-500, AIME-2024/2025, HMMT Feb 2025
- **Code**: LiveCodeBench v6, BFCL-v3
- **Multimodal VQA**: MMMU, MathVista, OCRBench, Video-MME 等共 15 项
- **Omni-modal**: LibriSpeech (ASR), WeNetSpeech (中文语音), WorldSense (音视频理解)

#### 🔹 System 1（子系统规划）
- 导航：VLN-CE, NaVILA
- 操作：LIBERO
- 动作生成：自定义音频驱动动作生成任务（含主观评测）

#### 🔹 System 0（全身控制）
- 自研 WBC（Whole-Body Controller）策略集，用于机器人行走、姿态保持等任务
- 评估指标：Success Rate (SR), MPJPE（平均关节位置误差）

#### 🔹 Full-System Composition
- **Vehicle-Guide Task**：集成 System 2（对话理解）、System 1（导航）、System 0（行走控制）的复合任务，在模拟展厅环境中完成车辆介绍。

---

### 实验设置与评估指标

| 层级 | 设置 | 评估指标 |
|------|------|---------|
| **System 2** | 使用 Qwen3.6-27B、Qwen3-32B、Qwen3-Omni-30B 模型，vLLM 推理后端，8×A100 节点 | 准确率（Acc）、Pass@k、WER（词错误率）、CER（字符错误率） |
| **System 1** | 闭环仿真环境，支持主观偏好打分（10 名盲评员） | 客观成功率、人类偏好比例 |
| **System 0** | 物理仿真环境（Isaac Gym 类），批量 rollout | SR, MPJPE, 行为级诊断指标（如步态对称性、足部离地高度） |
| **Full-System** | 复合任务执行，记录完整 trace | 端到端成功率、子目标达成率、失败归因分布 |

---

### 基线方法对比

| 基线框架 | 覆盖范围 | 是否可比 |
|---------|--------|--------|
| **lm-evaluation-harness (lm-eval)** | 仅短文本 QA | ✅ 对比 |
| **Inspect AI** | LLM Agent（多轮工具调用） | ✅ 对比 |
| **VLMEvalKit / lmms-eval** | 多模态视觉问答 | ✅ 对比 |
| **Isaac Lab / RoboHive** | 全身控制仿真 | ❌ 不可直接对比（非 Orchestrator） |

> 注：DeepInsight 是唯一能横跨所有层级的 orchestrator。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 准确率对标实验（Alignment）

DeepInsight 在多个基准上 **复现了参考结果**，且多数情况下最接近官方报告值：

| 框架 | 最接近参考值的测试项占比 |
|------|---------------------|
| **DeepInsight** | **5/8 (Qwen3.6)**, **4/9 (Qwen3.32B)**, **8.0/15 (VQA)**, **4/5 (Omni-modal)** |
| Inspect AI | 3/8 | 2/9 | 2.0/15 | — |
| lm-eval | 0/8 | 3/9 | — | — |

> 表明 DeepInsight **未牺牲准确性换取速度**。

---

#### ⚡ 端到端吞吐提升（Throughput）

在相同硬件（单节点 8×A100）下，DeepInsight 显著快于各基线：

| 基线 | 加速比（Speedup） |
|------|------------------|
| **lm-eval** | **1.29×** |
| **Inspect AI** | **1.13×** |
| **VLMEvalKit** | **1.03×** |
| **lmms-eval** | **1.04×** |
| **lmms-eval (Omni)** | **1.20×** |

> 图 5 显示，对于长尾输出任务（如数学推理），异步流水线优势明显。

---

#### 🔍 消融实验（Ablation Studies）

##### （1）异步流水线 vs 同步批处理（Async vs Sync）
- 测试任务：AIME-2024（数学竞赛题）
- 结果：
  - Async：108 分钟，LLM 吞吐 **2,178 tokens/s**
  - Sync：152 分钟，LLM 吞吐 **1,547 tokens/s**
  - ➜ **提速 1.41×，GPU 利用率提升 41%**

> 原因：同步批处理受最长任务拖累，异步可立即释放资源。

##### （2）阶段解耦 vs 共享并发池（Decoupled vs Coupled）
- 测试任务：LiveCodeBench v6（代码生成 + 沙箱执行）
- 设置：
  - Decoupled：生成（128 并发）+ 沙箱（14 并发）
  - Coupled：共享池限制为 14（瓶颈在沙箱）
- 结果：
  - Decoupled：**1h48m**
  - Coupled：**5h57m**
  - ➜ **提速 3.31×**

> 原因：耦合模式下 GPU 被严重限流，利用率极低。

##### （3）水平扩展能力（Horizontal Scaling）
- 任务：27 个 System 2 套件
- 结果：
  - 1 节点：80h55m
  - 2 节点：39h53m（**2.03×**）
  - 4 节点：20h14m（**4.00×**）
- ➜ **近似线性扩展，无显著通信开销**

---

## 4. 关键结论和发现

### 主要发现

1. **统一运行时可行且必要**  
   通过三大抽象（Task/Resource/Result），可在异构任务上构建统一评估基础设施，无需为每层定制 harness。

2. **跨层诊断是核心价值**  
   所有层共享 trace identity，使得“高层决策错误导致底层失控”类问题可被精准定位，这是分立框架无法做到的。

3. **架构设计直接影响吞吐**  
   异步流水线 + 阶段解耦 + 控制平面分离，显著提升资源利用率，尤其在长尾任务中优势巨大。

4. **组合任务暴露模块间鸿沟**  
   在 Vehicle-Guide 任务中：
   - 子目标局部成功率普遍 >80%
   - 但端到端成功率仅 **60.4%**
   - 最大失败来源是 **System 1 内部执行（42.1%）** 和 **系统边界交接失败（28.9%）**

> 表明：**模块独立表现好 ≠ 系统整体可用**，必须进行全栈集成验证。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **当前仍以仿真为主** | 所有 rollout 均在模拟器中运行，尚未接入真实机器人 |
| **System 1/0 任务覆盖面有限** | 当前案例为示范性质，尚未形成广泛 benchmark 支持 |
| **依赖内部资源调度体系** | Control Plane 的 autoscaling 依赖公司级 infra，开源复现难度较高 |

---

### 未来工作方向

1. **拓展 System 1 和 System 0 的任务家族**  
   支持更多导航、操作、安全、全身控制任务，通过统一接口接入。

2. **打通 Sim-to-Real Gap**  
   将真实机器人接入相同的 `resource-handle` 协议，使仿真与实机 rollout 共享 trace schema，实现 gap 可测量、可诊断。

3. **开放框架与生态建设**  
   推动 DeepInsight 成为行业标准评估平台，支持第三方 benchmark 快速集成。

---

> **总结一句话**：  
> **DeepInsight 不只是一个更快的 evaluator，而是一个让 Physical AI 系统“可观测、可诊断、可持续迭代”的工程基础设施。**

</details>

---

### 16. [Bridging Functional Correctness and Runtime Efficiency Gaps in LLM-Based Code Translation](https://arxiv.org/abs/2606.17683)

**Authors**: Longhui Zhang, Jiahao Wang, Chenhao Hu, Bingyu Liang, Jing Li, Min Zhang  
**Category**: cs.CL  
**Published**: 2026-06-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.17683v1  

#### Abstract
While large language models (LLMs) have greatly advanced the functional correctness of automated code translation systems, the runtime efficiency of translated programs has received comparatively little attention. With the waning of Moore's law, runtime efficiency has become increasingly important f...

---

### 17. [Environment-Grounded Automated Prompt Optimization for LLM Game Agents](https://arxiv.org/abs/2606.17838)

**Authors**: Rean Clive Fernandes, Lukas Fehring, Theresa Eimer, Marius Lindauer, Matthias Feurer  
**Category**: cs.CL  
**Published**: 2026-06-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.17838v1  

#### Abstract
LLM agents in interactive environments are highly sensitive to their prompts, yet prompt engineering remains a manual, task-specific process. We introduce an automated prompt optimization framework for LLM agents that decomposes the observation-to-action pipeline into a goal-conditioned descriptor a...

---

### 18. [RISE: Relay Inference and Online Scheduling for Efficient Edge-Device Collaborative Diffusion Model Services](https://arxiv.org/abs/2606.17378)

**Authors**: Zilan Huang, Zhiqing Tang, Hanshuai Cui, Tian Wang, Yuan Wu, Weijia Jia, Wei Zhao  
**Category**: cs.DC  
**Published**: 2026-06-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.17378v1  

#### Abstract
Text-to-image diffusion models are increasingly deployed at the network edge to serve heterogeneous workloads with diverse quality and latency requirements. However, existing deployment strategies choose either large edge-side models with high fidelity but high latency or lightweight device-side mod...

---

### 19. [From GPU to Microcontroller: Online Ridge Regression for Edge-Deployable Traffic Prediction](https://arxiv.org/abs/2606.17613)

**Authors**: Suresh Purini, Archit Narwadkar, Deepak Gangadharan  
**Category**: cs.DC  
**Published**: 2026-06-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.17613v1  

#### Abstract
State-of-the-art traffic flow forecasting models, including Graph Convolutional Networks and graph-less MLPs, require centralized GPU training across all sensors, making them impractical for resource-constrained intelligent transportation deployments. We show that much of this complexity is unnecess...

---

### 20. [SpatioTemporal Causal Network Diagnostics for Geographic Tipping Point Early Warning](https://arxiv.org/abs/2606.17553)

**Authors**: Zhaoyuan Yu, Zhangyong Liang  
**Category**: cs.LG  
**Published**: 2026-06-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.17553v1  

#### Abstract
Geographic tipping points in ecosystems, climate subsystems, or ice sheets pose severe challenges for localized early warning. Classical spatial indicators such as Moran's I summarize global spatial structure, but they struggle with three issues: spatial dilution, Euclidean assumptions, and correlat...

---

### 21. [A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation](https://arxiv.org/abs/2606.18075)

**Authors**: Haoyang Zhong, Yifei Sun, Antong Zhang, Chunping Wang, Lei Chen, Yang Yang  
**Category**: cs.AI  
**Published**: 2026-06-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.18075v1  

#### Abstract
Retrieval-Augmented Generation (RAG) has emerged as a paradigm for enhancing large language models (LLMs) with external knowledge, yet existing graph-based methods face a fundamental limitation: entity-centric and chunk-centric approaches operate on representations anchored to original text without ...

---

### 22. [CheckMIABench: Firm Foundations For Membership Inference Attacks on Language Models](https://arxiv.org/abs/2606.17464)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel  
**Category**: cs.LG  
**Published**: 2026-06-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.17464v1  

#### Abstract
Membership inference attacks (MIAs) are a canonical way to assess a machine learning model's privacy properties. Although several attempts have been made to evaluate MIAs on language models, the extant literature has suffered numerous difficulties in constructing clean evaluations to test new techni...

---

### 23. [MGUP: A Momentum-Gradient Alignment Update Policy for Stochastic Optimization](https://arxiv.org/abs/2606.17526)

**Authors**: Da Chang, Ganzhao Yuan  
**Category**: cs.LG  
**Published**: 2026-06-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.17526v1  

#### Abstract
Efficient optimization is essential for training large language models. Although intra-layer selective updates have been explored, a general mechanism that enables fine-grained control while ensuring convergence guarantees is still lacking. To bridge this gap, we propose \textbf{MGUP}, a novel mecha...

---

### 24. [How Inference Compute Shapes Frontier LLM Evaluation](https://arxiv.org/abs/2606.17930)

**Authors**: Jessica McFadyen, Ole Jorgensen, Harry Coppock, Kevin Wei, Cozmin Ududec  
**Category**: cs.AI  
**Published**: 2026-06-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.17930v1  

#### Abstract
AI evaluations are shifting toward harder tasks that benefit from longer trajectories involving tool use and iterative problem solving. As a result, performance is increasingly sensitive to the amount and allocation of compute available at test time ("inference compute"). Yet many evaluations still ...

---

### 25. [STAR: SpatioTemporal Adaptive Reward Allocation for Text-to-Image RL Post-Training](https://arxiv.org/abs/2606.17979)

**Authors**: Jinjie Shen, Wei Deng, Xian Hu, Daiguo Zhou, Jian Luan  
**Category**: cs.AI  
**Published**: 2026-06-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.17979v1  

#### Abstract
Existing RL post-training methods for text-to-image generation usually convert the final-image reward into a single scalar advantage and apply it with the same strength to the entire generative trajectory. However, text-to-image generation naturally has temporal and spatial structure: different deno...

---

### 26. [The Stanford EDGAR Filings Dataset: Reconstructing U.S. Corporate and Financial Disclosures into Layout-Faithful and Token-Efficient Pretraining Data](https://arxiv.org/abs/2606.18192)

**Authors**: Nick Bettencourt, Xiaowei Ding, Kay Giesecke  
**Category**: cs.AI  
**Published**: 2026-06-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.18192v1  

#### Abstract
As high-quality public web corpora become increasingly exhausted, clean long-context documents have become a scarce and expensive source of training data for large language models (LLMs). Existing long-context corpora are often proprietary and costly to acquire, synthetically generated, or concentra...

---

### 27. [LUMEN: Coordinated Failure Recovery for Distributed LLM Serving](https://arxiv.org/abs/2606.17787)

**Authors**: Zhang Cao, Shujie Han, Juncheng Zhang, Yuanming Ren, Yongkun Li, Patrick P. C. Lee  
**Category**: cs.DC  
**Published**: 2026-06-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.17787v1  

#### Abstract
Modern large language model (LLM) serving clusters distribute inference requests across multiple worker processes on different GPUs, but failures are prevalent at scale. When a worker fails, the cluster simultaneously loses the failed worker's GPU-resident key-value (KV) caches and serving capacity,...

---

### 28. [Latency Prediction for LLM Inference on NPU Systems](https://arxiv.org/abs/2606.18042)

**Authors**: Juhyun Park, Seungwoo Jeong, Jingyu Lee, Kyungyong Lee  
**Category**: cs.DC  
**Published**: 2026-06-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.18042v1  

#### Abstract
Deploying Large Language Models (LLMs) requires exploring a large configuration space spanning parallelization strategies, batching techniques, and scheduling policies. Exhaustive measurement across this space is impractical, making latency prediction essential for system optimization. While NPUs ha...

---

### 29. [ResAware: Cross-Environment Website Fingerprinting via Resource-Privileged Distillation](https://arxiv.org/abs/2606.17462)

**Authors**: Chongru Fan, Wei Wang, Wentao Huang, Zhenquan Ding, Jinqiao Shi, Lei Cui, Zhiyu Hao, Xiaochun Yun  
**Category**: cs.LG  
**Published**: 2026-06-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.17462v1  

#### Abstract
While Website Fingerprinting (WF) attacks achieve high accuracy in controlled laboratory settings, they often degrade substantially in real-world environments due to spatio-temporal drift, browser heterogeneity, proxy obfuscation and etc. This limitation stems from their sole reliance on low-level t...

---

### 30. [Multi-Adapter PPO: A Cross-Attention Enhanced Wavelength Selection Framework for LIBS Quantitative Analysis](https://arxiv.org/abs/2606.17476)

**Authors**: Hao Li, Man Fung Zhuo  
**Category**: cs.LG  
**Published**: 2026-06-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.17476v1  

#### Abstract
Laser-induced breakdown spectroscopy (LIBS) quantitative analysis faces critical challenges in wavelength selection due to high-dimensional spectral data and the fundamental trade-off between prediction accuracy and feature efficiency. This paper presents a novel Multi-Adapter PPO framework that tra...

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

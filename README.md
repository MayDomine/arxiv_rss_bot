# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-25 06:47:09 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training](https://arxiv.org/abs/2602.20656)

**Authors**: Guanbin Xu, ZhenGuo Xu, Yuzhe Li, Youhui Bai, Ping Gong, Chaoyi Ruan, Cheng Li  
**Category**: cs.DC  
**Published**: 2026-02-25  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2602.20656v1  

#### Abstract
Overlapping communication with computation is crucial for distributed large-model training, yet optimizing it - especially when computation becomes the bottleneck-remains challenging. We present Lagom, a system that co-tunes communication parameters to balance resource usage between computation and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
在分布式大模型训练中，**通信与计算重叠（Communication-Computation Overlapping）** 是提升训练效率的关键手段。然而，当计算成为瓶颈时（computation-bottlenecked），现有的优化方法往往失效。这是因为：
- 通信操作会与计算竞争 GPU 资源（如 SMs、全局内存带宽、L2 缓存），导致即使通信时间缩短，也会加剧对计算的干扰，反而降低整体性能。
- 多个通信操作之间存在时序依赖和资源级联效应，联合调优参数空间呈指数增长（超过 $10^6$ 种组合），难以高效搜索最优配置。

现有方法如 **AutoCCL** 主要针对通信瓶颈场景设计，在计算受限场景下表现不佳；而静态资源配置（如 Liger）缺乏适应性。

---

### 🚀 **提出了什么新方法或新思路**
本文提出 **Lagom**，一个用于自动调优集体通信参数的系统，核心创新如下：

#### （1）**统一的重叠性能建模（Unified Overlap Performance Model）**
- 构建了一个适用于 **计算瓶颈** 和 **通信瓶颈** 场景的统一代价模型。
- 明确量化了通信参数（如 `Number of Channels` (NC)、`Chunk Size` (C)）如何通过两种机制影响性能：
  - **SM 竞争（SM Competition）**：通信占用 SMs，减少可用于计算的资源，增加计算波次（waves）。
  - **全局资源竞争（Global Resource Contention）**：大 NC 或 C 导致更高的内存带宽争用，延长每波执行时间。

#### （2）**基于优先级的搜索算法（Priority-Based Search Algorithm）**
- 引入 **优先级度量 H_j** 来衡量每个通信操作的调优性价比：
  $$
  H_j = \frac{Y' - Y}{x - x'}
  $$
  其中 $Y$ 是调优前后的总计算时间，$x$ 是通信时间。**H 越小，表示单位通信加速带来的计算开销越低，优先级越高。**
- 采用贪心策略，每次选择 H 最小的通信进行迭代调优，避免穷举所有组合。

#### （3）**线性复杂度优化**
- 将原本指数级的联合搜索空间（$O(r^N)$）压缩为近似线性复杂度，显著提升调优效率。

---

### 🔍 **相比现有方法的优势**
| 方法 | 局限性 | Lagom 的改进 |
|------|--------|-------------|
| **NCCL** | 默认配置激进（如高 NC），在 NVLink 上加剧资源争用 | 动态降配，减轻对计算干扰 |
| **AutoCCL** | 仅优化通信性能，忽略其对计算的影响 | 综合考虑通信增益与计算损失，实现协同优化 |
| **Liger / Libra** | 静态分配或仅关注部分参数 | 动态、自适应地联合调优多参数 |

> ✅ **核心思想转变**：从“尽可能快完成通信” → “以最小代价平衡通信与计算”。

---

## 2. **核心实验方法和设置**

### 📚 **使用的模型与 workload**
在多种大规模 DNN 模型上验证，涵盖不同并行范式：

| 模型 | 类型 | 并行方式 |
|------|------|----------|
| **Phi-2-2B**, **Llama-3-8B**, **MPT-7B** | Dense Models | FSDP, TP |
| **DeepSeek-MoE-16B**, **OLMoE-1B-7B** | MoE Models | EP |

支持的并行模式包括：
- Fully Sharded Data Parallelism (**FSDP**)
- Tensor Parallelism (**TP**)
- Expert Parallelism (**EP**)

微批次大小设为 GPU 内存允许的最大值，全局批次遵循工业标准。

---

### 💻 **硬件基础设施**
使用两个异构集群测试泛化能力：

| 集群 | GPU 数量 | 内部连接 | 节点间连接 |
|------|---------|-----------|------------|
| **Cluster A** | 2×8 A40 | NVLink (400Gbps) | InfiniBand (2×400Gbps) |
| **Cluster B** | 2×8 A40 | PCIe 4.0 | InfiniBand (100Gbps) |

软件栈：CUDA 12.8 + PyTorch 2.3.0 + Megatron-LM

---

### 📊 **评估指标**
- **主指标**：单次训练迭代时间（Training Iteration Time）
- **辅助分析**：
  - 通信/计算耗时分解
  - 收敛曲线（Convergence Curve）
  - 参数搜索收敛速度（Tuning Efficiency）

---

### ⚖️ **基线方法对比**
- **NCCL v2.18.3-1**：默认通信库，广泛使用
- **AutoCCL**：当前最先进的自动通信调优系统
- **Lagom**：本文方法，构建于 AutoCCL 框架之上，增强其对计算瓶颈的支持

---

## 3. **主要实验结果和性能指标**

### 📈 **端到端性能提升**

#### （1）**FSDP 下的结果（图7a）**
| 场景 | vs NCCL | vs AutoCCL |
|------|--------|------------|
| 所有模型平均 | **1.10–1.33× speedup** | **最高达 1.35× 更优** |
| 在 NVLink 高带宽环境下 | 提升更明显（因 NCCL 默认高 NC 更易引发 contention） |

> ❗ AutoCCL 在某些情况下甚至劣于 NCCL（如 Phi-2 FSDP 下仅为 0.87×），因其过度优化通信导致计算严重受阻。

#### （2）**TP / EP 下的结果（图7b）**
| 并行类型 | vs NCCL | vs AutoCCL |
|--------|--------|------------|
| Tensor Parallelism (TP) | **1.08–1.16×** | 略胜 AutoCCL |
| Expert Parallelism (EP) | **1.07–1.08×** | 同样优于 AutoCCL |

表明 Lagom 具备良好的跨并行范式通用性。

---

### 🔍 **细粒度分析（Breakdown）**

#### ▶️ **Pattern 1（计算瓶颈）**
- **场景**：FSDP 中 AllGather 操作与 FFN 计算重叠
- **NCCL 配置**：NC=8, C=2MB → 强通信但高 contention
- **AutoCCL**：进一步提升 NC=61 → 通信略快，但计算变慢 → 整体退化至 **0.87×**
- **Lagom**：主动降低 NC=2, C=684KB → 通信稍慢，但计算提速显著 → 实现 **1.35× end-to-end speedup**

#### ▶️ **Pattern 2（多通信耦合）**
- 包含多个通信（如 ReduceScatter）
- Lagom 根据 H 指标优先调优关键通信
- 将 ReduceScatter 从 (NC=8, C=2MB) 调整为 (NC=4, C=1366KB)
- 成功实现 **1.43× speedup**

---

### ⏱️ **调优效率（图8c）**
- 在双通信任务中：
  - **AutoCCL**：约 16 次迭代收敛
  - **Lagom**：约 33 次迭代收敛
- 迭代比约为 **1:2**，符合预期的线性增长趋势（vs 指数）
- 相对于百万级训练迭代而言，调优开销可忽略不计

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **通信并非越快越好**：在计算瓶颈场景下，过度优化通信会导致严重的资源争用，反而拖慢整体训练。
2. **参数选择存在权衡（trade-off）**：增大 `NC` 或 `C` 可能加快通信，但也显著延长计算时间（实测差异可达 **30% 以上**）。
3. **H 指标有效指导调优顺序**：通过量化“每单位通信收益所付出的计算代价”，能准确识别应优先优化的通信操作。
4. **Lagom 实现了跨架构、跨并行范式的稳定加速**：在高带宽（NVLink）和低带宽（PCIe）环境中均表现优异。

---

### ⚠️ **方法的局限性**
- 当前模型仍依赖于 **AutoCCL 的 divide-and-conquer 框架**，未完全覆盖所有通信算法子空间。
- 对 **非稳态工作负载**（如动态序列长度、稀疏激活）的适应性有待验证。
- 当前实现聚焦于 **collective communication**（如 AllReduce, AllGather），尚未扩展至点对点通信或流水线调度。

---

### 🔮 **未来工作方向**
- 将 Lagom 的思想扩展至 **全流程通信-计算调度器**，结合 pipeline scheduling 与 memory management。
- 探索 **在线自适应调优机制**，应对训练过程中动态变化的工作负载。
- 集成至主流训练框架（如 DeepSpeed、PyTorch FSDP）作为默认通信优化模块。
- 开源代码即将发布，推动社区共建。

---

## ✅ 总结一句话
> **Lagom 通过建立通信与计算之间的资源竞争模型，并引入基于成本效益的优先级搜索机制，首次实现了在计算瓶颈场景下的高效通信参数协同优化，在多种 LLM 训练场景中取得最高达 1.43× 的端到端加速，超越 NCCL 与 AutoCCL。**

</details>

---

### 2. [ReviveMoE: Fast Recovery for Hardware Failures in Large-Scale MoE LLM Inference Deployments](https://arxiv.org/abs/2602.21140)

**Authors**: Haley Li, Xinglu Wang, Cong Feng, Chunxu Zuo, Yanan Wang, Hei Lo, Yufei Cui, Bingji Wang, Duo Cui, Shuming Jing, Yizhou Shan, Ying Xiong, Jiannan Wang, Yong Zhang, Zhenan Fan  
**Category**: cs.DC  
**Published**: 2026-02-25  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.21140v1  

#### Abstract
As LLM deployments scale over more hardware, the probability of a single failure in a system increases significantly, and cloud operators must consider robust countermeasures to handle these inevitable failures. A common recovery approach is to simply restart the LLM serving instance; however, this ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ReviveMoE: Fast Recovery for Hardware Failures in Large-Scale MoE LLM Inference Deployments

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
随着大规模 **LLM** 部署规模扩大，硬件故障概率显著上升。在 **Mixture-of-Experts (MoE)** 架构中，单个硬件（如 NPU）失效可能导致整个推理服务中断。传统恢复方式是重启实例，但该过程涉及重新加载模型权重、重建通信域和图编译，耗时长达数分钟，严重影响 **SLO (Service Level Objectives)**。

本文聚焦于解决 **MoE LLM 推理部署中的快速容错恢复问题**，目标是在不重启实例的前提下实现秒级恢复。

---

### 提出了什么新方法或新思路
作者提出 **ReviveMoE**，一种专为 xDeepServe 平台设计的快速故障恢复系统，支持两种部署架构：
- **MA-collocated**：Attention 和 MoE 共置于同一设备
- **MA-disaggregated**：Attention 与 MoE 分离部署

ReviveMoE 的核心思想是 **避免全量重启**，通过以下机制实现快速恢复：
- **故障隔离与状态迁移**：仅终止失败进程，将活跃请求迁移到健康节点
- **日志回滚恢复 Block Table**：基于数据库式的日志机制还原 KV cache 管理状态
- **权重完整性保障**：
  - 利用 **Data Parallelism (DP)** 和 **redundant experts** 避免权重重载
  - 支持 **role switching**：将冗余的 Attention 节点转为 MoE 节点以替代丢失专家
  - 容忍少量专家丢失（missing experts），通过 gating 函数屏蔽失效专家
- **通信域重建与图缓存编译**：
  - 动态重构 XCCL 通信域（如 A2E/E2A）
  - 使用 **cached compilation** 技术将图编译时间从 12.9 分钟缩短至 <10 秒

---

### 相比现有方法的优势
| 维度 | 传统方法（重启） | ReviveMoE |
|------|------------------|---------|
| 恢复时间 | >80 秒（含权重加载、图编译） | 最快 **10.2 秒**（减少 87.8%） |
| 是否需要权重重载 | 是 | 多数情况否（利用冗余） |
| 图编译开销 | 从头编译（~15 min） | 使用预编译缓存（~6–10 s） |
| 支持部署模式 | 有限 | 同时支持 **collocated 与 disaggregated** |
| 用户影响 | 请求中断严重 | 请求可迁移，服务连续性高 |

> ✅ **优势总结**：ReviveMoE 在保证模型可用性和准确性的前提下，极大降低了恢复延迟，更适合面向客户的 **MaaS (Model-as-a-Service)** 场景。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要使用 **DeepSeek-V3** 模型（671B 参数，EP=32）
- 评估任务来自 **Language Model Evaluation Harness**，包括：
  - ARC Challenge / Easy
  - WinoGrande
  - HellaSwag
  - PIQA
  - RACE
  - TruthfulQA MC1/MC2
  - GSM8k
  - MMLU

---

### 实验设置和评估指标

#### 硬件平台
- **CloudMatrix384** 架构
- 使用 **80 张 Huawei Ascend 64GB NPU**
- 软件栈：CANN 8.2.1, Torch NPU 2.1.0, XCCL 0.23.5

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Recovery Time** | 故障发生到恢复推理的时间（代表服务中断时长） |
| **Model Accuracy** | 使用多个 NLP 任务的平均得分衡量精度损失 |
| **Compilation Overhead** | 图编译耗时对比（完整 vs 缓存） |

#### 基线方法对比
- **Baseline**：使用缓存的完整实例重启（cached reinitialization）
- **ReviveMoE**：本文提出的多种恢复路径：
  - 注意力失败（Attention failure）
  - MoE 失败 + 冗余专家可用
  - MoE 失败 + 允许专家丢失
  - MoE 失败 + 需 role switching

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 恢复时间对比（Figure 5）
| 场景 | 恢复时间 | 相比基线提升 |
|------|--------|-------------|
| Baseline（重启） | 83.1 秒 | — |
| ReviveMoE（Attention failure） | **10.2 秒** | ↓ 87.8% |
| ReviveMoE（MoE failure, redundant） | 10.7 秒 | ↓ 87.2% |
| ReviveMoE（MoE failure, missing experts） | 10.8 秒 | ↓ 87.1% |
| ReviveMoE（MoE failure, role switch） | 12.8 秒 | ↓ 84.7% |

> ⚡ 即使最坏情况（需 role switching + 权重加载），仍比基线快 **36.6%**

#### 图编译优化效果
- 完整图编译时间：**12.9 分钟**
- 使用 **cached compilation** 后：仅需 **6–8 秒**
- 节省超过 **95% 的编译时间**

---

### 与基线方法的对比结果
- 所有 ReviveMoE 场景均显著优于重启方案
- **无需重启引擎和执行器进程** 是提速主因（节省 ~50 秒）
- **避免权重重载** 进一步压缩延迟
- **日志回滚机制** 确保 block table 一致性，避免 KV cache 错乱

---

### 消融实验结果（S4.2 & S4.3）

#### 模型精度受专家丢失的影响（Table 2 & Figure 6）
- 当 **≤1/32 的专家丢失**（即 EP ≥ 32 时单 NPU 故障），平均准确率下降 <0.05
- 在 **ARC Challenge、HellaSwag、MMLU** 等任务上，精度几乎无损
- 即使丢失 1/4 专家，多数任务仍保持 >90% 原始性能
- 结论：**在大 EP 部署中容忍少量专家丢失是可行且高效的选择**

#### Role Switching 的必要性分析（S4.3）
- 尽管容忍丢失有效，但在以下场景必须启用 role switching：
  1. **EP < 32** → 专家丢失比例过高，精度不可接受
  2. **冗余专家的最后一份副本丢失** → 必须恢复完整权重集
- 可结合策略：先以“缺失专家”模式快速恢复服务，后台异步执行 role switching 以最终恢复完整性

---

## 4. 关键结论和发现

### 论文的主要发现
1. **重启不是最优解**：在 MoE LLM 推理中，重启实例带来巨大延迟，远超实际所需。
2. **状态可局部恢复**：通过心跳检测、日志回滚、通信域重建等技术，可在不重启的情况下完成故障隔离与恢复。
3. **权重冗余 + 角色切换 = 高可用基石**：
   - 利用 DP 和 redundant experts 可避免大多数权重重载
   - role switching 提供兜底机制，增强系统鲁棒性
4. **图编译瓶颈可通过缓存突破**：**cached compilation** 是实现秒级恢复的关键，否则图编译将成为最大延迟来源。
5. **大 EP 部署天然具备容错能力**：当专家数量足够多时，丢失少量专家对整体性能影响极小。

---

### 方法的局限性
1. **仅处理单点故障**：不支持网络分区或多节点同时失效的大规模故障。
2. **依赖特定硬件生态**：深度集成华为自研的 **XCCL、xDeepServe、Ascend NPU**，通用性受限。
3. **未处理慢速设备（straggler）问题**：仅关注完全失效，未应对性能退化类软故障。
4. **角色切换成本较高**：需从磁盘加载 MoE 权重，耗时约 40 秒，在低冗余场景仍是瓶颈。

---

### 未来工作方向
1. **支持 straggler 检测与恢复**：识别并隔离性能下降设备，防止拖慢整体推理。
2. **扩展至多节点故障恢复**：引入更复杂的冗余策略（如跨机架冗余专家放置）。
3. **自动化恢复策略选择**：根据 EP 规模、冗余配置、SLO 要求动态决策是否启用 role switching 或容忍丢失。
4. **进一步优化 role switching 流程**：探索权重预加载、增量同步等机制降低切换开销。
5. **开放 XCCL 与 xDeepServe 接口**：提升方法在其他平台上的可移植性。

---

> ✅ **总体评价**：  
> ReviveMoE 是首个系统性解决 **MoE LLM 推理容错恢复** 的工业级方案，在真实生产环境中验证了其有效性。它不仅大幅缩短恢复时间，还提出了“容忍轻微精度损失换取极致可用性”的新范式，对构建高可靠 MaaS 平台具有重要指导意义。

</details>

---

### 3. [Deep unfolding of MCMC kernels: scalable, modular & explainable GANs for high-dimensional posterior sampling](https://arxiv.org/abs/2602.20758)

**Authors**: Jonathan Spence, Tob\'ias I. Liaudat, Konstantinos Zygalakis, Marcelo Pereyra  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.20758v1  

#### Abstract
Markov chain Monte Carlo (MCMC) methods are fundamental to Bayesian computation, but can be computationally intensive, especially in high-dimensional settings. Push-forward generative models, such as generative adversarial networks (GANs), variational auto-encoders and normalising flows offer a comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Deep unfolding of MCMC kernels: scalable, modular & explainable GANs for high-dimensional posterior sampling*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **Markov Chain Monte Carlo (MCMC)** 方法在贝叶斯反问题中能提供可靠的后验采样和不确定性估计，但在高维场景下计算成本高昂。而基于深度学习的 **push-forward generative models**（如 **GANs**, **VAEs**, **normalising flows**）虽然推理速度快，但缺乏可解释性，且对似然函数（likelihood）的变化不鲁棒，难以适应新的观测模型。

本文旨在弥合这两类方法之间的鸿沟：**如何构建一个既高效又可解释，并能灵活适应不同 likelihood 参数的生成模型？**

### 提出的新方法与新思路
作者提出了一种名为 **deep unfolding of MCMC kernels** 的新框架，将 MCMC 算法的迭代过程展开为一个深度神经网络架构。具体来说：

- 将 **Langevin MCMC** 或 **split-Gibbs sampler** 等固定步数的迭代算法映射为一个 $L$ 层的递归神经网络。
- 每一层对应一次 MCMC 转移核（transition kernel），其参数（如步长 $\gamma_l$、噪声水平 $t_l$、先验参数 $\theta$）作为可训练权重进行端到端优化。
- 使用 **regularized Wasserstein GAN** 框架进行训练，结合对抗损失、数据一致性（L1）和样本多样性（SD）正则项。

该方法被称为 **unfolded MCMC**，例如：
- 在 MNIST 上使用的 **U-SGS (Unfolded Split-Gibbs Sampler)**
- 在射电干涉成像中使用的 **U-LATINO (Unfolded LATINO)**

### 相比现有方法的优势
| 特性 | 传统 MCMC | 黑盒 GAN (如 RCGAN) | 本文方法 (Unfolded MCMC) |
|------|-----------|------------------------|----------------------------|
| **可解释性** | 高（物理建模清晰） | 低（黑箱） | ✅ 高（保留 MCMC 结构） |
| **效率** | 低（需数千次迭代） | ✅ 高（单次前向传播） | ✅ 高（仅需 $L=8\sim64$ 步） |
| **灵活性/适应性** | ✅ 高（可调 likelihood） | 低（依赖训练分布） | ✅ 高（支持 inference-time 参数输入） |
| **采样质量** | 准确但慢 | 快但可能模式坍塌 | ✅ 高保真 + 多样性 |
| **模块化** | ✅ 强（分离 prior 和 likelihood） | 弱 | ✅ 强（天然嵌入 likelihood） |

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **MNIST**  
   - 用于图像去模糊任务（image deblurring）
   - 输入维度小，便于消融研究和精确评估。

2. **PROBES dataset**  
   - 包含超过 2000 张晚型星系图像，用于 **射电干涉成像 (radio interferometry, RI)** 任务。
   - 更具现实意义，挑战性强，涉及稀疏傅里叶采样和复杂天体结构。

### 实验设置
- **任务形式**：解决线性逆问题 $ y = Ax^* + e $，其中 $A$ 是退化算子（卷积核或掩码傅里叶变换），$e$ 是噪声。
- **训练方式**：
  - 使用监督式训练，基于成对数据 $(x^{(i)}, y^{(i)})$
  - 采用 **end-to-end** 的 **Wasserstein GAN + 正则化** 损失函数：
    $$
    \mathcal{L}(\Theta) = \underbrace{\mathcal{L}_{\text{adv}}}_{\text{对抗损失}} + w_1\underbrace{\mathcal{L}_1}_{\text{数据一致性}} - w_{\text{SD}}\underbrace{\mathcal{L}_{\text{SD}}}_{\text{样本多样性}}
    $$
  - 动态调整 $w_{\text{SD}}$ 权重以匹配真实后验的标准差（Robbins-Monro 更新）。

### 评估指标
| 指标 | 含义 |
|------|------|
| **PSNR** | 峰值信噪比，衡量像素级重建精度 |
| **SSIM** | 结构相似性，感知质量 |
| **LPIPS** | 学习型感知图像块相似度，更符合人类视觉 |
| **SW (Sliced Wasserstein)** | 衡量后验分布逼近程度 |
| **CFID / W-latent** | 嵌入空间下的 Fréchet 距离，评估分布相似性 |
| **CMMD** | 最大均值差异，基于 CLIP 嵌入 |
| **NFE (Neural Function Evaluation)** | 单个样本所需的网络前向次数，反映效率 |

### 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **VAE-SGS** | Zero-shot MCMC | 使用预训练 VAE prior 的 split-Gibbs sampler，运行上万次迭代 |
| **RCGAN** | End-to-end GAN | 基于 [4] 的正则化条件 GAN，U-Net 架构 |
| **LATINO** | Zero-shot SBM | 不展开的 score-based MCMC 方法，使用预训练 denoiser |
| **IRIS** | Zero-shot SBM | 当前最先进的射电成像 SBM 方法，需 4000+ E-M 步骤 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1 & 2）

#### ✅ MNIST 去模糊任务（Table 1）
| Method (NFEs) | PSNR (mean) | LPIPS | SW ($\times10^{-5}$) | W-latent ($\times10^{-2}$) | Time (ms) |
|---------------|-------------|-------|------------------------|----------------------------|-----------|
| **U-SGS (64)** | **28.46** | **0.003** | **6.95** | **0.35** | 1.89 |
| RCGAN (1) | 26.69 | 0.009 | 9.21 | 9.08 | **0.10** |
| VAE-SGS (10⁴) | 23.49 | 0.015 | 11.66 | 0.92 | 120 |

> **结论**：随着 $L$ 增加，U-SGS 性能持续提升，在 $L=64$ 时全面超越 RCGAN，尤其在分布保真度（SW, W-latent）上优势显著。

#### ✅ 射电干涉成像任务（Table 2）
| Model (NFEs) | PSNR (mean) | LPIPS | CFID | Time (s) |
|--------------|-------------|-------|------|----------|
| **U-LATINO (8)** | **45.61** | **0.01** | **0.06** | **0.08** |
| RIGAN (1) | 43.77 | 0.07 | 0.52 | 0.04 |
| IRIS (64) | 46.09 | 0.07 | 4.06 | 3.91 |
| IRIS (1000) | 48.99 | 0.07 | 2.24 | 55.93 |

> **结论**：
> - U-LATINO 仅用 **8 NFEs** 就达到接近 IRIS (64 steps) 的 PSNR，但速度**快 50 倍以上**。
> - IRIS (1000) 虽 PSNR 更高，但耗时极长，且 CFID 差，说明样本多样性不足（过平滑）。
> - U-LATINO 在 **LPIPS 和 CFID 上大幅领先**，表明其生成样本更具细节和真实性。

### 消融实验与关键发现
- **迭代次数 $L$ 的影响**（Fig 2, Table 1）：
  - $L=8$ 时已接近 RCGAN 表现；
  - $L=64$ 时性能饱和，验证了“越深越好”的趋势。
- **out-of-distribution 泛化能力**（Appendix B.1 & B.3）：
  - 当测试时使用未见过的模糊核或更稀疏的射电掩码（K2h vs K4h），**U-SGS 和 U-LATINO 表现出更强鲁棒性**，而 RCGAN 性能下降明显。
  - 证明了 unfolded 架构对物理模型的显式编码使其更能适应新任务。
- **样本多样性分析**（Fig 8, B.2.2）：
  - Zero-shot LATINO 样本高度集中，缺乏多样性；
  - **U-LATINO 展现出丰富的后验采样路径**，标准差图与残差高度相关，说明其能准确量化不确定性。

---

## 4. 关键结论和发现

### 主要发现
1. **Deep unfolding 是连接 MCMC 与 GAN 的有效桥梁**：
   - 成功将 MCMC 的**可解释性、模块化、物理一致性**与 GAN 的**高效推理能力**结合起来。
2. **所提方法兼具高性能与高效率**：
   - 在多个任务上实现了优于或媲美 SOTA 方法的采样质量，同时将采样步骤压缩至 $L=8\sim64$。
3. **具备出色的泛化能力和鲁棒性**：
   - 对似然参数变化（如不同模糊核、掩码）具有天然适应性，适合实际应用场景。
4. **能有效捕捉后验不确定性**：
   - 通过多步采样获得相关但多样化的样本，可用于可靠的风险评估和决策支持。

### 方法的局限性
- **需要成对训练数据**：依赖 $(x, y)$ 数据集进行监督训练，限制了在无真值场景的应用。
- **仍需离线训练阶段**：不像 pure zero-shot MCMC 可直接部署，必须针对特定任务类别进行训练。
- **内存开销略增**：相比原始 MCMC，需存储额外的可学习参数（如 LoRA weights），但远小于完整 GAN。

### 未来工作方向
1. **自监督/无监督训练**：探索无需 ground truth 的训练范式（如 self-supervised learning），扩展至更多实际场景。
2. **与其他生成模型结合**：将该框架应用于 **consistency models** 或 **flow matching** 等新兴生成技术。
3. **模型压缩与蒸馏**：进一步减少 NFE 数量，实现“few-step”甚至“one-step”高质量采样。
4. **理论分析**：建立 unfolded MCMC 收敛性的数学理论基础，指导网络设计与训练策略。

--- 

> **总结一句话**：  
> 本文提出的 **unfolded MCMC** 框架成功打造了一个**可解释、高效、灵活且分布保真的生成采样器**，为高维贝叶斯反问题提供了一条兼具科学严谨性与工程实用性的新路径。

</details>

---

### 4. [Scaling Vision Transformers: Evaluating DeepSpeed for Image-Centric Workloads](https://arxiv.org/abs/2602.21081)

**Authors**: Huy Trinh, Rebecca Ma, Zeqi Yu, Tahsin Reza  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.21081v1  

#### Abstract
Vision Transformers (ViTs) have demonstrated remarkable potential in image processing tasks by utilizing self-attention mechanisms to capture global relationships within data. However, their scalability is hindered by significant computational and memory demands, especially for large-scale models wi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Scaling Vision Transformers: Evaluating DeepSpeed for Image-Centric Workloads*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Vision Transformers (ViTs) 在图像分类等任务中展现出强大潜力，但由于其自注意力机制带来的高计算复杂度和内存消耗，**在大规模训练时面临严重的可扩展性瓶颈**。尽管 DeepSpeed 已被广泛用于优化大语言模型（LLMs）的分布式训练，但在 **image-centric workloads（如 ViTs）上的应用尚未系统研究**。

本文旨在填补这一空白，探索如何利用 DeepSpeed 提升 ViTs 的训练效率与可扩展性。

---

### 🚀 提出的新方法与思路
- **首次将 DeepSpeed 框架系统应用于 Vision Transformers 的分布式训练场景**，涵盖 intra-node 和 inter-node 设置。
- 构建了一个基于 DeepSpeed + NCCL + MPI 的分布式训练流水线，支持多种 GPU 集群配置。
- 系统性地评估了 **data parallelism** 在不同硬件环境下的表现，并分析软件参数（如 `batch size`、`gradient accumulation`）对训练效率的影响。
- 探索了 **strong scaling** 与 **weak scaling** 行为，揭示通信开销与计算负载之间的权衡关系。

---

### ⚖️ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **框架适配性** | 将原本面向 NLP 的 DeepSpeed 成功迁移至 CV 领域，验证其通用性 |
| **训练效率** | 通过合理设置 batch size 可显著降低 synchronization 开销，提升 speedup |
| **可复现性与开源** | 完整代码公开于 GitHub ([trinhgiahuy/Scalable_ViT_DT](https://github.com/trinhgiahuy/Scalable_ViT_DT))，便于后续研究 |

> ❗ 注：未使用 ZeRO 优化（受限于资源），但指出了其在未来的重要性。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 类别数 | 图像数量 | 分辨率 | 备注 |
|-------|--------|----------|--------|------|
| **CIFAR-10** | 10 | 60,000 | 32×32 | 主要用于初步验证 |
| **CIFAR-100** | 100 | 60,000 | 32×32 | 更具挑战性的分类任务 |
| **ImageNet-100\*** | 100 | ~100,000 | 224×224 | *因时间限制未能完成训练* |

---

### ⚙️ 实验设置
- **模型架构**：采用标准 **ViT_B_16** 架构
- **训练模式**：
  - **Intra-node**：单节点多 GPU（共享内存）
  - **Inter-node**：跨节点多 GPU（需网络通信）
- **并行策略**：仅使用 **Data Parallelism (DP)**，未启用 ZeRO 或其他高级内存优化
- **通信后端**：
  - **NCCL**（PyTorch 分布式后端）
  - **MPI (OpenMPI)** 用于进程启动与管理
- **集群平台**：
  - **Nebula**：ECE Linux 集群，配备 NVIDIA T4 GPU（intra-node）
  - **Tesla Cluster**：异构 GPU（RTX 3070 / GTX 1070 / Tesla P4），用于 inter-node 测试
  - **Vector Cluster**：54 节点，每节点 8×Tesla T4 (16GB)，高性能 inter/intra-node 实验平台

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Training Time per Epoch** | 平均每个 epoch 所需时间（秒） |
| **Speedup** | 相对于单 GPU 的加速比（理想为线性） |
| **Communication Overhead** | 同步梯度（AllReduce）所占时间比例 |
| **Accuracy** | Top-1 分类准确率 |
| **Strong Scaling** | 固定总数据量，增加 GPU 数量 → 观察训练时间下降趋势 |
| **Weak Scaling** | 每个 GPU 处理相同数据量，总数据随 GPU 增加 → 观察时间是否恒定 |

---

### 🔁 基线方法对比
- **Baseline**：单 GPU 训练作为基准
- **对比维度**：
  - 不同 GPU 数量下的训练时间与 speedup
  - 不同 batch size 下的同步开销与精度变化
  - intra-node vs inter-node 性能差异

> ❗ 未直接与其他分布式框架（如 Megatron-LM、HuggingFace Accelerate）进行横向 benchmark（列为未来工作）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ Vector 集群（T4V2 GPUs）——CIFAR-10 强缩放（Strong Scaling）
| GPU 数量 | 平均训练时间（秒） | Speedup |
|---------|------------------|--------|
| 1       | 1817.65          | 1.00× |
| 2       | 941.75           | ~1.93× |
| 4       | 621.72           | ~2.92× |
| 8       | 489.64           | ~3.71× |

> ✔️ 显示良好的强缩放特性，尤其从 1→2 GPU 接近线性加速

#### ✅ Vector 集群——CIFAR-10 弱缩放（Weak Scaling）
- 训练时间基本保持稳定（约 170–200 秒），符合预期
- 表明系统能有效处理随规模增长的工作负载

#### ✅ Batch Size 对性能影响（Nebula 集群）
| Batch Size | 2-GPU 同步开销占比 | 总训练时间（秒） |
|-----------|--------------------|------------------|
| 16        | 极高               | 276.31           |
| 64        | 中等               | 232.61           |
| 128       | 较低               | 212.21           |
| 256       | 几乎饱和           | 无明显改善       |

> 🔍 发现：**batch size = 64 或 128 是最优折衷点**，兼顾同步成本与内存利用率

#### ✅ Gradient Accumulation 的潜在价值
- 在小 batch size 场景下可通过 gradient accumulation 模拟更大 batch，减少同步频率
- 特别适用于内存受限的 GPU 设备

#### ✅ Inter-node vs Intra-node 对比（Vector 集群）
- 图 15 显示：**multi-node single-GPU 与 single-node multi-GPU 的训练时间几乎一致**
- 表明在当前设置下，**inter-node 通信开销可控**，DeepSpeed + NCCL 能高效支持跨节点训练

#### ⚠️ Tesla 集群失败案例
- 使用非均匀 GPU（RTX 3070 / GTX 1070 / Tesla P4）
- 结果偏离理想 scaling 曲线，甚至出现 **训练时间随 GPU 增加而上升**
- 原因：弱 GPU 成为同步瓶颈，拖慢整体进度

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DeepSpeed 可有效支持 ViTs 的分布式训练**，无论 intra-node 还是 inter-node 设置。
2. **batch size 是影响 scalability 的关键因素**：
   - 过小 → 高频同步 → 高通信开销
   - 过大 → 内存压力 & CPU-GPU 数据传输瓶颈
   - **推荐值：64–128**（视 GPU 显存而定）
3. **GPU 硬件同质性至关重要**：
   - 异构设备会导致严重性能退化（如 Tesla 集群实验所示）
4. **communication overhead 可控但不可忽视**：
   - 在合理配置下，inter-node 与 intra-node 性能接近
5. **accuracy 不一定随 GPU 数量增加而提高**：
   - 多 GPU 可加快训练，但不保证更高精度（见图 10）

---

### ⚠️ 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **未使用 ZeRO 优化** | 受限于资源与时间，未启用 ZeRO-1/2/3，无法评估其对超大 ViT 的影响 |
| **未测试高分辨率图像** | ImageNet-100 未成功训练，缺乏对真实大规模数据的支持验证 |
| **仅使用 data parallelism** | 未结合 model/pipeline parallelism，难以扩展到百亿参数以上模型 |
| **缺乏与其他框架对比** | 缺少与 Megatron-LM、HuggingFace Accelerate 的性能 benchmark |

---

### 🔮 未来工作方向
1. **深入研究 ZeRO 各阶段（Stage 1–3）对 ViT 训练的内存节省与开销影响**
2. **引入 sequence parallelism**（借鉴 DeepSpeed-Ulysses）：
   - 将图像 patch 序列沿 sequence dimension 切分
   - 支持长序列图像处理（如医学影像、遥感图）
3. **集成稀疏注意力机制**：
   - 如 SparseViT、Long-Sequence-Segmentation 等，降低计算复杂度
4. **跨领域应用拓展**：
   - Vision-Language Models（VLMs）
   - 科学成像任务（如 fastMRI、CoSTAR、GTDB）
5. **benchmark 其他分布式训练框架**：
   - 对比 Megatron-LM、HuggingFace Accelerate 在 ViT 上的表现

---

## 总结
本论文是 **将 DeepSpeed 应用于 Vision Transformers 的早期探索性研究**，提供了宝贵的实践经验与系统级洞察。虽然受限于资源未能实现极致扩展，但它为后续构建高效、可扩展的视觉模型训练 pipeline 奠定了坚实基础。

> 🧩 **一句话总结**：  
> **DeepSpeed can scale ViTs effectively — if you have homogeneous hardware and tune your batch size wisely.**

</details>

---

### 5. [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2602.20309)

**Authors**: Jingxuan Zhang, Yunta Hsieh, Zhongwei Wang, Haokun Lin, Xin Wang, Ziqi Wang, Yingtie Lei, Mi Zhang  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.20309v1  

#### Abstract
Vision-language-action (VLA) models unify perception, language, and control for embodied agents but face significant challenges in practical deployment due to rapidly increasing compute and memory demands, especially as models scale to longer horizons and larger backbones. To address these bottlenec...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Vision-Language-Action (VLA)** 模型在机器人控制中表现出色，但由于其庞大的模型规模和复杂的跨模态依赖关系，面临严重的计算与内存开销问题。尤其是基于 **Diffusion Transformer (DiT)** 的动作头（action head）对量化极为敏感，导致现有 **Post-Training Quantization (PTQ)** 方法难以直接应用。

已有方法如 TinyVLA、EfficientVLA 等主要聚焦于架构设计或推理流程优化（如剪枝、缓存），**并未有效处理数值精度层面的低比特部署挑战**，特别是 DiT 动作头的量化稳定性问题。

### 提出了什么新方法或新思路
本文提出 **QuantVLA** —— 首个专为 VLA 模型设计的训练无关（training-free）、尺度校准型 PTQ 框架，成功实现了对语言主干和 DiT 动作头的同时低比特量化。

其核心创新在于三个“尺度校准”组件：

1. **Selective Quantization Layout（选择性量化布局）**  
   - 对所有线性层进行整数量化（integerize），但保留注意力投影 $Q, K, V, O$ 在浮点表示。
   - 目的是避免因输入分布漂移放大 attention logits 温度变化和残差流能量失衡。

2. **Attention Temperature Matching (ATM)**  
   - 引入轻量级每头标量 $\alpha$，用于匹配教师模型与量化学生模型之间的 logits 标准差。
   - 该标量被折叠进反量化尺度（dequantization scales），不引入额外算子。

3. **Output Head Balancing (OHB)**  
   - 引入每层标量 $\beta$，用于对齐输出头在残差接口处的能量（RMS），恢复 layer norm 的操作点。
   - 同样通过 scale folding 实现，无额外计算开销。

整个框架无需微调、仅需小批量无标签校准数据，支持低比特权重与激活（如 W4A8），且保持原始架构不变。

### 相比现有方法的优势
| 维度 | QuantVLA | 现有方法（如 DuQuant, SmoothQuant） |
|------|----------|-------------------------------|
| 是否支持 DiT 动作头量化 | ✅ 是（首次实现） | ❌ 否或严重性能下降 |
| 是否训练自由 | ✅ 是 | ✅ 多数是 |
| 是否改变执行顺序/结构 | ❌ 否 | ❌ 否（但效果差） |
| 性能表现 | ⬆️ 超越 FP16 基线 | ⬇️ 显著下降 |
| 内存节省 | ~70% | ~60–70%，但以性能为代价 |

> ✅ **关键优势**：**首次实现对高度耦合 VLA 模型中 DiT 动作头的安全 PTQ，同时提升任务成功率并大幅降低内存占用。**

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 主要基准：**LIBERO**（一个面向终身机器人学习的知识迁移模拟环境）
  - 包含四个任务套件：
    - **Spatial**：空间关系推理与精确定位
    - **Object**：物体为中心的抓取与操作
    - **Goal**：指令到目标对齐
    - **Long**：长视野任务与误差累积控制
- 补充验证：**Simpler** 和 **Pick-and-Can** 操控基准（见附录）

### 实验设置和评估指标
- **模型**：
  - OpenPI 0.5（高效型 VLA）
  - GR00T N1.5（大容量、双系统设计的人形机器人策略模型）
- **量化配置**：
  - 主要采用 **W4A8**（4-bit 权重，8-bit 激活）
  - 部分测试 W4A4 设置
- **校准数据**：
  - 小规模无标签缓冲区（unlabeled calibration buffer）
  - 使用 32 batch 进行 scale 估计
- **评估指标**：
  - **Success Rate (%)**：标准 LIBERO 协议下的任务完成率
  - **Memory Usage (GB)**：量化模块的内存占用
  - **Relative Memory Savings (%)**：相比 FP16 基线的相对节省

### 基线方法对比
| 方法 | 类型 | 是否适用 DiT |
|------|------|-------------|
| **FP16 Baseline** | 全精度参考 | ✅ |
| **DuQuant [18]** | 旋转+平滑 PTQ | ❌（应用于 DiT 导致严重性能下降） |
| **SmoothQuant [40]** | 通道平滑 PTQ | ❌（扩展至 DiT MLP 后仍不稳定） |
| **EfficientVLA [45]** | 层剪枝/特征复用 | ✅（但非量化路径） |
| **VLA-Cache [41]** | KV 缓存机制 | ✅（互补而非替代） |

> QuantVLA 与上述方法正交，可组合使用。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

#### OpenPI 0.5 @ LIBERO
| 方法 | Avg. Succ. Rate | Memory (LLM+DiT) | Relative Saving |
|------|------------------|-------------------|------------------|
| FP16 Baseline | 97.1% | 4.27 GB | 0.0% |
| +DuQuant (LLM+DiT) | 76.3% | 1.17 GB | 72.6% |
| **+QuantVLA** | **97.6%** | **1.28 GB** | **70.0%** |

> 🔥 **QuantVLA 不仅节省约 70% 内存，还略微超越全精度基线的成功率！**

#### GR00T N1.5 @ LIBERO
| 方法 | Avg. Succ. Rate | Memory (LLM+DiT) | Relative Saving |
|------|------------------|-------------------|------------------|
| FP16 Baseline | 86.5% | 2.02 GB | 0.0% |
| +DuQuant (LLM+DiT) | 70.0% | 0.74 GB | 63.4% |
| **+QuantVLA** | **88.0%** | **0.91 GB** | **55.0%** |

> 💡 在更大更复杂的模型上依然稳定增益。

### 与基线方法的对比结果
- **vs DuQuant**：在两个模型上均显著优于 DuQuant（+21.3% 和 +18.0% 平均成功率），说明通用 PTQ 方法无法适应 VLA 中紧密耦合的 DiT 结构。
- **vs SmoothQuant**（见 Table 5）：
  - 在 W8A8 下表现接近；
  - 但在更激进的 **W4A8** 下，QuantVLA 明显胜出，尤其在 **Long** 任务中（+3.5–4.0%）。
- **Pick-and-Can 基准**（Table 6）：
  - SmoothQuant 成功率从 31→16/50；
  - QuantVLA 保持 27/50，显示更强鲁棒性。

### 消融实验结果（Table 1 & Figure 3）

#### 不同量化粒度的影响（Table 1）
| Layer Selection | OpenPI 0.5 Avg SR |
|------------------|--------------------|
| No Quant | 97.1% |
| LLM Only | 96.5% |
| DiT Only | 71.6% |
| Full LLM+DiT | 76.3% |
| **LLM + DiT (MLP only)** | **95.4%** ✅ |

> ✅ 证明：**只量化 DiT 的 MLP 层（保留 QKV/O 浮点）是最优选择**。

#### ATM 与 OHB 效果可视化（Figure 3）
- **ATM**：显著缩小 logits 标准差与教师模型的差距，特别是在深层 block；
- **OHB**：有效对齐 attention 输出后的 RMS 值，缓解残差能量漂移；
- 两者共同作用下，各层统计量几乎完全贴合教师模型。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **DiT 动作头对量化极其敏感的根本原因** 是：
   - 量化引起的 scale drift 改变了 attention logits 的有效温度（effective temperature）；
   - 扰动了残差流中的能量注入强度，破坏 layer norm 的动态平衡。
2. **选择性量化 + 轻量校准机制** 可以在不修改架构的前提下，安全地将 VLA 模型压缩至 W4A8 甚至 W4A4。
3. **QuantVLA 是首个成功的 VLA PTQ 框架**，不仅没有性能损失，反而在多个任务上**超过 FP16 基线**，实现“超分辨率”般的量化增强。
4. 该方法具有良好的泛化能力，在不同模型规模（OpenPI vs GR00T）、不同去噪步数（8 vs 16 steps）下均保持稳健。

### 方法的局限性
- 当前 ATM 和 OHB 设计针对 **DiT-based VLA** 定制，对于非 DiT 架构（如 OpenVLA 使用 autoregressive token 输出）虽可用（见 Table 7），但未发挥全部潜力。
- 当前仅支持统一 bit-width（如 W4A8），尚未探索混合精度（mixed-precision）分配。
- 校准过程依赖少量真实交互轨迹，若环境变化剧烈可能需要重新校准。

### 未来工作方向
- 将 ATM/OHB 推广至其他生成式多模态系统（如视觉-语言规划、具身世界模型）；
- 探索与 EfficientVLA、VLA-Cache 等效率框架的联合部署方案；
- 发展自适应在线校准机制，应对动态环境下的分布偏移；
- 扩展至更低比特（如 W3A4/W2A8）并研究硬件协同优化。

---

> 📌 **总结一句话**：  
> **QuantVLA 是首个实现对 VLA 模型中 DiT 动作头安全、高效、训练无关低比特量化的框架，在显著降低 ~70% 内存的同时，反而提升了任务成功率，为具身智能的边缘部署提供了实用路径。**

</details>

---

### 6. [WeirNet: A Large-Scale 3D CFD Benchmark for Geometric Surrogate Modeling of Piano Key Weirs](https://arxiv.org/abs/2602.20714)

**Authors**: Lisa L\"uddecke, Michael Hohmann, Sebastian Eilermann, Jan Tillmann-Mumm, Pezhman Pourabdollah, Mario Oertel, Oliver Niggemann  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.20714v1  

#### Abstract
Reliable prediction of hydraulic performance is challenging for Piano Key Weir (PKW) design because discharge capacity depends on three-dimensional geometry and operating conditions. Surrogate models can accelerate hydraulic-structure design, but progress is limited by scarce large, well-documented ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：WeirNet: A Large-Scale 3D CFD Benchmark for Geometric Surrogate Modeling of Piano Key Weirs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **数据稀缺性**：在水工结构设计中，尤其是Piano Key Weirs（PKWs）的几何代理建模领域，缺乏大规模、公开且高质量的3D CFD基准数据集。现有的研究多依赖小规模实验数据或封闭的CFD模拟，限制了机器学习模型的训练与公平比较。
- **几何敏感性挑战**：PKW的水力性能对三维几何形状高度敏感，传统经验公式难以泛化，而高保真CFD模拟计算成本高昂，无法支持快速设计迭代。

### 提出的新方法与创新
- **WeirNet 数据集发布**：
  - 包含 **3,794 种参数化生成的矩形与梯形 PKW 几何体**，每种几何体在 **19 个不同流量条件**下进行CFD仿真，共完成 **71,387 次成功模拟**。
  - 提供多种模态表示：紧凑的参数描述符（parametric descriptors）、无渗漏表面网格（watertight surface meshes）、高分辨率点云（high-resolution point clouds）。
  - 所有样本均提供完整的 **discharge coefficient $ c_p $** 标签，适用于正向预测与曲线重建任务。

- **扩展的PKW参数化体系**：
  - 在 Pralong et al. [56] 的基础上，引入了新的几何参数（如 $ R_{B.i}, R_{B.o} $ 表示悬挑长度比），并明确定义了梯形PKW中的侧壁倾斜角 $ \alpha $ 和投影厚度 $ T_{s.2}, T_{s.3} $，提升了参数表达能力与自动化建模可靠性。

- **可复现的全流程框架**：
  - 公开发布CAD生成脚本、CFD配置文件、后处理流程及评估代码，确保从几何生成到模型训练的全链路可复现。
  - 遵循工程数据集最佳实践（如 datasheets for datasets [22]），提升透明度与社区可用性。

### 相比现有方法的优势
| 维度 | WeirNet | 现有工作（如 [14][41][63]） |
|------|--------|-----------------------------|
| 数据规模 | 71,387 simulations, 3,794 geometries | 数百至数千次模拟，通常 <50 几何体 |
| 几何多样性 | 多参数联合变化（key widths, overhangs, wall inclinations） | 单一参数扫描（如仅 key-width ratio） |
| 输出模态 | Parametric + Mesh + Point Cloud + $ c_p $ 曲线 | 通常仅有参数或实验标量 |
| 开放性 | 完整公开（CC BY-NC 4.0） | 多数未公开或部分开放 |
| 基准任务 | 明确划分 ID/OOD splits，支持泛化评估 | 缺乏标准化测试协议 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **WeirNet** 是唯一使用的主数据集，其构建过程如下：
  - **几何生成**：基于 Rhinoceros 3D + Grasshopper 实现参数化建模，采用拉丁超立方采样（LHS）在物理可行范围内生成候选设计，并通过 SolidUnion 自动检测自交失败样本。
  - **CFD仿真**：使用 OpenFOAM@ v2212 的 `interFoAM` 求解器，结合 VOF 方法捕捉自由液面，采用 RANS + $ k\text{-}\omega $ 湍流模型。
    - 流域尺寸：上游 30×P，下游 15×P（P=0.33m）
    - 网格量级：约 45万–80万个单元
    - 模拟时长：50秒以达到准稳态
    - 流量范围：50–250 L/s，共19个离散值（见 Table 6）

### 实验设置与评估指标
#### 任务定义
- **Task 1（ID任务）**：给定几何表示与流量 $ Q $，预测 discharge coefficient $ c_p $
- **OOD任务**：
  - **OOD-Geom**：按侧壁倾角 $ \alpha $ 划分训练/测试集，评估几何外推能力
  - **OOD-Head**：按流量 $ Q $ 范围划分，评估操作条件外推能力

#### 评估指标
- **MAE**（Mean Absolute Error）：反映典型误差幅度
- **MSE**（Mean Squared Error）：强调大偏差惩罚
- **Max AE**（Maximum Absolute Error）：捕获最坏情况
- **$ R^2 $**（Coefficient of Determination）：解释方差比例，越高越好

所有结果在 **80%/10%/10%** 的 geometry-discharge pair 层面划分训练/验证/测试集。

### 基线方法对比
| 类型 | 模型 | 描述 |
|------|------|------|
| **Parametric Models** | RandomForest, XGBoost, LightGBM, GradientBoosting | 基于 scikit-learn 的树集成模型，输入为8维参数 + $ Q $ |
| **Geometric Deep Learning** | PointNet, RegDGCNN, Mesh-GCN | 分别作用于点云（5k点）、动态图、三角网格 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（In-Distribution 结果）

#### 参数化模型表现（Table 8）
| Model | MSE ↓ | $ R^2 $ ↑ | MAE ↓ | Max AE ↓ |
|-------|--------|------------|--------|-----------|
| **RandomForest** | **5.20×10⁻⁵** | **99.67%** | **3.94×10⁻³** | 2.90×10⁻¹ |
| XGBoost | 6.80×10⁻⁵ | 99.58% | 5.53×10⁻³ | **1.44×10⁻¹** |
| LightGBM | 7.50×10⁻⁵ | 99.54% | 6.00×10⁻³ | **1.05×10⁻¹** |
| GradientBoosting | 119.50×10⁻⁵ | 92.63% | 19.61×10⁻³ | 3.54×10⁻¹ |

> ✅ **RandomForest 表现最优**，尤其在 $ R^2 $ 和 MAE 上领先。

#### 几何深度学习模型表现（Table 7）
| Model | MSE ↓ | $ R^2 $ ↑ | MAE ↓ | Inference Time ↓ |
|-------|--------|------------|--------|------------------|
| **PointNet** | **15.30×10⁻⁵** | **98.96%** | **9.06×10⁻³** | **0.49ms** |
| RegDGCNN | 97.40×10⁻⁵ | 83.11% | 22.01×10⁻³ | 53.15ms |
| Mesh-GCN | 241.60×10⁻⁵ | 64.33% | 36.88×10⁻³ | 0.90ms |

> ✅ **PointNet 在几何模型中全面领先**，兼具精度与速度优势。

### 与基线方法对比结论
- **参数模型显著优于几何模型**：RandomForest 的 MSE 比 PointNet 低近 **3倍**，$ R^2 $ 更高。
- **推理效率极高**：所有代理模型单样本推理时间在 **毫秒级**，相比 CFD（小时级）实现 **数个数量级加速**。
- **几何模型仍具价值**：尽管精度略逊，但 Mesh/GCN/PointNet 不依赖特定参数化方案，适合非标准或仅有网格数据的应用场景。

### 消融实验结果

#### （1）Out-of-Distribution 泛化能力（Table 9）
| Split | Best Model ($ R^2 $) | Worst Model ($ R^2 $) | 发现 |
|-------|------------------------|--------------------------|------|
| **OOD-Head**（新 $ Q $） | LightGBM: ~96.6% | GradientBoosting: ~83% | 外推未知流量相对容易，性能下降有限 |
| **OOD-Geom**（新 $ \alpha $） | XGBoost: ~94.7% ($ \alpha \in [3^\circ,5^\circ] $) | RandomForest: -19.43% ($ \alpha < 2^\circ $) | **几何迁移是主要失败模式**，特别是从梯形推向矩形（$ \alpha \to 0 $）时崩溃 |

> 🔍 **关键发现**：几何分布偏移（geometry shift）远比操作条件偏移更难应对。

#### （2）数据效率分析（Figure 14）
- 所有模型在训练数据达到 **60%** 后趋于饱和，继续增加数据带来的增益递减。
- RandomForest 在少量数据下即表现出色，适合低资源场景。
- GradientBoosting 出现“过拟合”现象——更多数据反而导致性能下降，表明其容量不足。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **参数化代理模型在ID任务上表现最佳**：精心设计的参数特征（如 $ \alpha, W_{o.a}, T_{s.3} $）能有效压缩几何信息，Tree-based models 可达接近完美的 $ R^2 > 99.5\% $。
2. ✅ **几何深度学习模型具备竞争力且更具通用性**：PointNet 在点云上表现良好，虽稍逊于参数模型，但在无需人工参数化的场景中具有重要应用潜力。
3. ⚠️ **OOD泛化瓶颈在于几何分布偏移**：模型可以较好地外推到未见过的流量 $ Q $，但一旦遇到新类型的几何结构（如矩形 vs 梯形），性能急剧下降。
4. ⏱️ **代理模型实现极快推理**：**<1ms 推理延迟** 支持交互式设计探索，极大提升设计效率。
5. 📉 **数据收益存在边际效应**：超过约 **60% 训练数据** 后性能提升趋缓，建议优先保证几何多样性而非单纯扩大样本量。

### 方法的局限性
- **设计空间受限**：当前仅涵盖 Type-A PKW，固定总宽与高度，排除了 parapet walls、noses 等实际结构特征。
- **单一CFD设置依赖**：所有标签来自同一 RANS $ k\text{-}\omega $ 模型，可能存在数值偏差，影响向实验或其他CFD求解器的迁移。
- **目标简化**：目前仅提供 scalar $ c_p $ 输出，未包含完整 flow fields、uncertainty estimates 或 multi-objective criteria。
- **尺度效应**：所有模拟基于实验室尺度（1m clear width），需谨慎外推至工程原型。

### 未来工作方向
1. **扩展设计空间**：纳入其他 PKW 类型（B/C/D）、变全局尺寸、带挡墙结构等，逼近真实工程项目。
2. **多保真度融合**：结合低成本2D模拟、中保真3D-CFD与高保真实验数据，发展 multi-fidelity learning 框架。
3. **不确定性量化**：开发 UQ-aware surrogate models，在OOD区域给出置信区间，辅助安全决策。
4. **逆向设计与生成模型**：
   - 利用 Conditional VAE / Diffusion Models 生成满足性能要求的新PKW几何。
   - 探索基于 PointFlow [75]、Latent Point Diffusion [71] 的生成架构。
5. **跨任务迁移**：利用原始70TB CFD数据重启更高分辨率模拟，用于研究空化、振动、泥沙冲刷等复杂问题。
6. **语言模型辅助设计**：探索 LLMs 在 hydraulic engineering 中作为设计助手的能力（参考 [55][26]）。

---

> 💡 **总体评价**：  
> WeirNet 不仅是一个高质量数据集，更是一个推动 **AI for Civil Engineering** 发展的基础设施。它填补了通用流体力学基准（如 FlowBench）与小型封闭实验之间的鸿沟，为水工结构的智能化设计提供了坚实基础。其发布的全流程工具链也为后续研究树立了可复现性标杆。

</details>

---

### 7. [GauS: Differentiable Scheduling Optimization via Gaussian Reparameterization](https://arxiv.org/abs/2602.20427)

**Authors**: Yaohui Cai, Vesal Bakhtazad, Cunxi Yu, Zhiru Zhang  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.20427v1  

#### Abstract
Efficient operator scheduling is a fundamental challenge in software compilation and hardware synthesis. While recent differentiable approaches have sought to replace traditional ones like exact solvers or heuristics with gradient-based search, they typically rely on categorical distributions that f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GauS: Differentiable Scheduling Optimization via Gaussian Reparameterization**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
- **Operator Scheduling** 是软件编译与硬件综合中的核心难题，属于 **NP-hard** 问题，目标是在满足依赖、延迟等约束的前提下，优化资源使用（如 LRes）、内存占用（LMem）或通信开销（Lcom）。
- 传统方法存在明显瓶颈：
  - **Exact solvers**（如 ILP/SMT）虽能求得最优解，但计算复杂度高，难以扩展到大规模图。
  - **Heuristics**（如 List Scheduling, FDS）速度快但常陷入局部最优。
  - **现有 Differentiable 方法**（如 GS-Schedule）采用 **categorical distribution + Gumbel-Softmax**，参数量随图深度 $D$ 和节点数 $|V|$ 线性增长（即 $O(D \cdot |V|)$），导致内存爆炸且无法有效利用 GPU 并行性。

### 🚀 **提出了什么新方法或新思路**
提出 **GauS** —— 一种基于 **Gaussian Reparameterization** 的可微调度框架：

- 将每个 operator $v_i$ 的执行时间建模为一个连续随机变量 $X_i \sim \mathcal{N}(\mu_i, \sigma_i)$，其中：
  - $\mu_i$: 预期调度步长（mean）
  - $\sigma_i$: 不确定性（standard deviation），控制松弛程度
- 利用 **Gaussian PDF/CDF** 构造平滑、可微的目标函数与约束违反项期望（如 $E[\text{Dep violation}]$, $E[L_{\text{Mem}}]$）。
- 使用 **Augmented Lagrangian Method (ALM)** 动态调整约束权重，避免手动调参。
- 最终通过 **round($\mu$)** 得到离散调度方案，并辅以轻量级 **legalization heuristic** 修复非法调度。

### ⭐ **相比现有方法的优势**
| 维度 | GauS | GS-Schedule（前作） |
|------|------|---------------------|
| 参数规模 | $O(|V|)$，仅需学习 $\mu, \sigma$ 向量 | $O(D \cdot |V|)$，需学习完整概率分布 |
| 时间序贯建模能力 | ✔️ 天然支持时间邻近性（temporal proximity） | ❌ 分类分布忽略时间顺序 |
| GPU 可扩展性 | ✔️ 支持大规模图（>5万节点） | ❌ 在大图上频繁出现 OOM |
| 模型灵活性 | ✔️ 支持 **modulo scheduling**（带循环依赖的流水线调度） | ❌ 仅支持常规调度 |
| 优化效率 | ✔️ 收敛速度提升 **1–2个数量级** | 较慢 |

---

## 2. **核心实验方法和设置**

### 📊 **使用的数据集**
- **Realistic Benchmarks**:  
  - **EPFL Benchmark Suite**（来自 Amarú et al., 2015），包含真实硬件设计 DAGs，节点数从 182 到 57,375 不等。
- **Synthetic Benchmarks**:  
  - **Random Workloads (RW)**，共12组，节点数从 ~900 到 ~9,400。
- **Pipelined Extension**:  
  - 对上述两套数据集人工添加 **recurrence constraints**（回边），用于测试 **modulo scheduling** 场景。

> 数据统计见 Table 1，最大图达 **57,375 节点**，远超以往可微调度工作的规模。

### 🧪 **实验设置和评估指标**
- **时间限制**：所有方法统一设定 **15分钟超时**。
- **硬件平台**：NVIDIA A100 GPU（80GB）+ AMD EPYC CPU。
- **实现框架**：PyTorch + CUDA。
- **评估指标**：
  - **Solution Quality**：目标函数值（如 $L_{\text{Res}}, L_{\text{Mem}}, L_{\text{MMem}}$）相对于 GauS 的比率（越低越好）。
  - **Runtime**：达到某一质量水平所需时间。
  - **Feasibility**：是否能在时限内找到合法调度。
  - **GPU Utilization / Memory Usage**：衡量系统效率。

### 🔁 **三种问题 Formulation**
| Formulation | 目标 | 约束 | 应用场景 |
|------------|------|-------|---------|
| **A** | min $L_{\text{Res}} + \alpha L_{\text{com}}$ | Dep, Lat | 通信瓶颈应用 |
| **B** | min $L_{\text{Mem}}$ | Dep, Lat | 存储敏感硬件 |
| **C** | min $L_{\text{MMem}}$ | MRes ≤ Cres, Dep, Lat, Rec | 流水线调度（modulo scheduling） |

### 🆚 **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Exact Solvers** | CPLEX, Gurobi（商用 ILP/SMT 求解器） |
| **Heuristics** | List Scheduling, Force-Directed Scheduling (FDS) |
| **Differentiable Baseline** | GS-Schedule（唯一可比的可微调度方法） |

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据与对比结果**

#### ✅ **Formulation A（资源+通信优化）**
- 图 **Figure 3** 显示：
  - GauS 在大多数实例上达到 **Pareto-optimal**。
  - 相比 GS-Schedule，**几何平均提升 71.8%**，且在大型图上完全避免 OOM。
  - CPLEX/Gurobi 在小图表现尚可，但在中大型图上要么超时，要么解质量差于 GauS 4 倍以上。

#### ✅ **Formulation B（内存足迹最小化）**
- 图 **Figure 5** 显示：
  - GauS 显著优于 List/FDS（降低 20%-60% 内存占用）。
  - CPLEX/Gurobi 在部分小图略优，但多数情况下无法在 15 分钟内完成。
  - GauS 是唯一能在所有图上稳定输出高质量解的方法。

#### ✅ **Formulation C（流水线调度）**
- 图 **Figure 6** 显示：
  - GauS 成功处理带 recurrence 约束的 modulo scheduling，而 GS-Schedule 无法支持此类问题。
  - 在 EPFL 大图上（如 `div`），exact solvers 完全失效，heuristics 表现平庸，GauS 保持领先。
  - 注意：`dec` 实例无可行解，被排除。

#### ⏱️ **质量-速度权衡分析（Any-Time Performance）**
- 图 **Figure 7–14** 展示：
  - GauS **收敛极快**，通常在几秒内就逼近最优解。
  - 相比之下，CPLEX/Gurobi 需要数十甚至数百秒才能取得类似质量。
  - 即使最终解相近，GauS 实现同等质量的速度快 **1–2个数量级**。

#### 💾 **GPU 效率分析（Appendix D）**
- **内存使用**：GauS 内存增长接近线性；GS-Schedule 在 `square`, `multiplier`, `div` 上直接 OOM。
- **GPU 利用率**：
  - GauS 平均利用率接近 **100%**。
  - GS-Schedule 下降至 **<40%**，表明其并行性差。

> 结论：**GauS 实现了“高质量 + 高速度 + 高可扩展性”的统一。**

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **Gaussian Reparameterization 是高效可微调度的关键创新**：
   - 参数量从 $O(D|V|)$ 降至 $O(|V|)$，极大提升可扩展性。
   - 自然编码时间邻近性，提供更清晰的梯度信号。
2. **首次将可微优化应用于 modulo scheduling**：
   - 支持复杂流水线场景下的 recurrence constraint 和周期性资源建模。
3. **GauS 实现 Pareto-optimal 性能**：
   - 在质量、速度、可行性之间取得最佳平衡。
   - 特别适合现代大规模、异构计算系统的调度需求（如 TSP, PIM 架构）。

### ⚠️ **方法的局限性**
- **独立 Gaussian 假设忽略了节点间相关性**：
  - 当多个 operator 共享同一 functional unit 或存在强耦合依赖时，当前模型未显式建模协方差。
  - 可能导致次优解或收敛不稳定。
- **偶尔收敛到不可行解**：
  - 尽管使用 ALM 和 legalization，仍可能出现需后处理修复的情况。
- **初始化敏感性**：
  - 虽然支持 informed initialization（如 ASAP/ALAP 中点），但对极端差初值鲁棒性有待验证。

### 🔮 **未来工作方向**
1. 引入 **Gaussian Process** 或低秩协方差矩阵建模节点间依赖关系。
2. 探索更先进的 **optimization strategies**（如二阶优化、自适应学习率）以增强稳定性。
3. 扩展至 **multi-objective scheduling**，结合 RL 或 preference learning 进行动态权衡。
4. 集成进实际 HLS 工具链（如 Vivado HLS, Catapult），进行端到端部署验证。

---

## ✅ **总结一句话**
> **GauS 通过 Gaussian Reparameterization 实现了可微调度的革命性突破，在参数效率、GPU 利用率、问题表达力和求解速度上全面超越已有方法，是首个支持大规模流水线调度的可微框架，为下一代智能编译器提供了强有力的基础工具。**

</details>

---

### 8. [Exploring the Impact of Parameter Update Magnitude on Forgetting and Generalization of Continual Learning](https://arxiv.org/abs/2602.20796)

**Authors**: JinLi He, Liang Bai, Xian Yang  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.20796v1  

#### Abstract
The magnitude of parameter updates are considered a key factor in continual learning. However, most existing studies focus on designing diverse update strategies, while a theoretical understanding of the underlying mechanisms remains limited. Therefore, we characterize model's forgetting from the pe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Exploring the Impact of Parameter Update Magnitude on Forgetting and Generalization of Continual Learning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于 **Continual Learning** 中一个核心挑战：**catastrophic forgetting**（灾难性遗忘）。尽管已有大量方法尝试缓解此问题，但大多数研究集中在设计复杂的更新策略（如正则化、回放、模块扩展），而对参数更新机制本身的理论理解仍不足。

本文从 **parameter update magnitude**（参数更新幅度）的角度出发，系统地分析其对 **forgetting** 和 **generalization** 的影响，试图回答以下关键问题：
- 遗忘是否仅由部分参数变化引起？
- 在何种条件下，冻结大部分参数（frozen training）优于全量微调（initialized training）？
- 多少参数更新是“足够”的？

### 🚀 提出的新方法与新思路
1. **理论建模遗忘为参数空间中的任务特定漂移（task-specific drift）**  
   与以往假设统一参数空间不同，本文提出将模型遗忘形式化为由于任务间参数漂移导致的 **knowledge degradation**，从而更准确刻画连续学习中的动态过程。

2. **推导最小化遗忘的最优参数更新幅度**  
   在固定模型容量下，通过优化框架推导出能够最小化遗忘的 **optimal parameter update magnitude**，并据此统一了两种典型训练范式：
   - **Frozen Training**：冻结共享参数，仅更新少量任务特定参数
   - **Initialized Training**：以旧模型为初始化，重新优化所有参数

3. **提出自适应混合更新框架（Adaptive Hybrid Update Framework）**  
   受理论启发，设计了一种新型训练策略：根据当前任务与前一任务之间的 **gradient direction**（梯度方向一致性，用 cosine similarity 衡量）动态调整参数更新范围：
   - 若梯度方向相似（高一致性），采用更保守的 frozen training
   - 若差异大，则允许更大范围的参数更新（接近 initialized training）

### 🔍 相比现有方法的优势
- **理论驱动设计**：不同于经验性方法，本工作提供了一个基于优化理论的解释框架，揭示了参数更新幅度与遗忘之间的内在关系。
- **高效且可扩展**：无需额外存储数据（非 replay-based）、不增加网络结构（非 architecture-based），在固定模型容量下实现高性能。
- **灵活性强**：提出的 hybrid 方法能自适应任务相似性，平衡 **stability**（稳定性）与 **plasticity**（可塑性）。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验在四个标准的 Continual Learning 基准上进行：
- **Split CIFAR-10**
- **Split CIFAR-100**
- **Split CUB-200**（细粒度鸟类分类）
- **Split Permuted MNIST**

此外，还构建了两个变体用于验证理论预测：
- **Correlated Split CIFAR-100**：增强任务间的相关性（共享类别）
- **Corrupted Split CUB-200**：引入图像噪声以降低任务相似性

### ⚙️ 实验设置与评估指标
- **模型架构**：基于 **ResNet**，包含9个卷积块（第一个单层，其余带残差连接），最后接平均池化和全连接层。
- **任务划分**：每个数据集被划分为多个 sequential tasks（例如，CIFAR-100 分为10或20个任务）。
- **训练方式**：
  - 所有方法均无数据回放（no replay）
  - 不保存历史数据或模型副本
- **评估指标**：
  - **Avg.Acc**（Average Accuracy）：最终所有任务上的平均测试准确率
  - **Task.Acc**（Current Task Accuracy）：各任务训练结束时在其上的准确率
  - **Forgetting Rate**：衡量先前任务性能下降程度，定义为 $\frac{1}{M-1} \sum_{k=1}^{M-1} (\alpha_{k,k} - \alpha_{M,k})$

### 🆚 对比的基线方法
- **Initialized Training**：标准 fine-tuning，每次从之前模型初始化并更新全部参数
- **Frozen Training**：冻结主干网络，仅训练一小部分 task-specific 参数
- **Adaptive Training**（本文提出）：根据梯度方向动态切换更新策略

所有结果均为三次随机种子运行的均值 ± 标准差。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table I）

| 方法 | Split CIFAR-100 (Avg.Acc / Forgetting) | Split CUB-200 | Split CIFAR-10 |
|------|-----------------------------|--------------|---------------|
| Initialized Training | 42.27±0.09 / 24.84±0.31 | 37.98±0.14 / 21.28±0.50 | 75.74±0.32 / 23.30±0.20 |
| Frozen Training | 43.38±0.11 / 23.96±0.38 | 37.15±0.12 / 21.18±0.20 | 75.98±0.06 / 21.10±0.72 |
| **Adaptive Training** | **47.19±0.79 / 24.15±0.27** | **39.01±0.42 / 20.76±0.13** | **80.55±0.58 / 19.51±0.39** |
| **Improvement** | **+4.92 / -0.69** | **+1.03 / -0.52** | **+4.81 / -3.79** |

> 注：Improvement 是相对于 Initialized Training 的增益。

#### 在相关性增强/退化场景下的表现（Table I 第二部分）：
| 方法 | Correlated CIFAR-100 | Corrupted CUB-200 |
|------|------------------------|--------------------|
| Adaptive Training | **50.21±0.17 / 20.03±0.12** (+6.43 Acc, -2.42 Forget) | **41.26±0.29 / 22.76±0.12** (+4.09 Acc, -0.45 Forget) |

表明 Adaptive 方法在任务相关性强时优势更明显。

### 🔬 消融与深入分析（Table II & III）

#### 当前任务准确率提升显著（Table II）：
| Dataset | Init. Train. | Ada. Train. | Δ |
|--------|------------|------------|----|
| Split CIFAR-100 | 60.72±0.05 | **67.30±0.74** | **+6.58** |
| Split CUB-200 | 57.32±0.36 | **60.69±0.46** | **+3.37** |

说明该方法不仅能减少遗忘，还能更好掌握新任务。

#### 长序列任务表现（Table III，20 tasks）：
| 方法 | Task.Acc | Avg.Acc | Forgetting |
|------|---------|--------|-----------|
| Initialized Training | 79.36 | 37.79 | 46.68 |
| Frozen Training | 78.28 | 37.59 | 45.88 |
| **Adaptive Training** | **80.81** | **39.09** | **45.08** |
| **Improvement** | **+1.45** | **+1.30** | **-1.60** |

在长序列下依然保持领先，兼具良好的新任务学习能力和抗遗忘能力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **参数更新幅度直接影响遗忘与泛化性能**  
   过大的更新会破坏已学知识，过小则无法有效学习新任务；存在一个理论最优的更新幅度。

2. **当任务在参数空间中距离较近时，frozen training 显著优于 initialized training**  
   支持了“共享知识主导”的假设，在任务相似性强的情境下，保留通用特征更有利。

3. **gradient direction 可作为任务相似性的有效代理信号**  
   利用前后任务梯度的 cosine similarity 动态调节更新强度，是一种简单而有效的自适应机制。

4. **提出的 hybrid update framework 综合了两种范式的优点**  
   在 stability 与 plasticity 之间实现了更优权衡，尤其适用于任务序列具有异质相似性的现实场景。

### ⚠️ 局限性
- 当前理论分析依赖于简化假设（如 Gaussian features），可能难以完全推广到复杂非凸情形。
- 梯度方向仅反映局部信息，对于高度非线性或模式突变的任务，其判别能力有限。
- 实验未涵盖 **class-incremental** 或 **task-free** 设置，在无明确任务边界时需进一步适配。

### 🔮 未来工作方向
- 将理论扩展至更广泛的模型族（如 Transformers）和损失函数
- 探索更高阶的梯度信息（如 Hessian）来指导参数更新
- 结合 replay 或 parameter isolation 方法，形成复合型高效 CL 框架
- 在真实流式数据（online streaming）中验证方法鲁棒性

---

## 总结

> 本文通过理论驱动的方式，揭示了 **parameter update magnitude** 在 Continual Learning 中的关键作用，提出了统一 frozen 与 initialized training 的优化视角，并据此设计出一种 **gradient-aware adaptive hybrid update strategy**。实验证明该方法在多个基准上 consistently 超越标准训练策略，不仅降低了遗忘，也提升了新任务的学习效率，为构建高效、可扩展的持续学习系统提供了新的理论基础与实践路径。

</details>

---

### 9. [CHESS: Context-aware Hierarchical Efficient Semantic Selection for Long-Context LLM Inference](https://arxiv.org/abs/2602.20732)

**Authors**: Chao Fei, Guozhong Li, Chenxi Liu, Panos Kalnis  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.20732v1  

#### Abstract
Long-context LLMs demand accurate inference at low latency, yet decoding becomes primarily constrained by KV cache as context grows. Prior pruning methods are largely context-agnostic: their token selection ignores step-wise relevance and local semantics, which undermines quality. Moreover, their ir...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CHESS: Context-aware Hierarchical Efficient Semantic Selection for Long-Context LLM Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Long-Context LLM Inference** 中，随着上下文长度增长，KV Cache 的内存访问成为推理延迟的主要瓶颈（memory bandwidth-bound）。传统方法通过 **KV Cache Pruning** 减少缓存大小以提升效率，但存在以下问题：
- **Context-agnostic selection**：基于全局重要性（如 attention score）选择 token，忽略当前生成步骤的语义相关性。
- **系统开销高**：token-level 的选择导致不规则内存访问、频繁数据移动，破坏 kernel fusion 和 batching 效率。
- **质量下降明显**：静态或粗粒度的选择容易丢失关键局部上下文，影响生成质量。

### 提出了什么新方法或新思路
论文提出 **CHESS**（Context-aware Hierarchical Efficient Semantic Selection），一种 **algorithm-system co-design** 的 KV Cache 管理系统，核心创新如下：

#### ✅ 算法层面
- **Context-aware 动态选择机制**  
  每个 decoding step 动态构建与当前 query 语义相关的上下文子集，而非依赖全局固定的“重要 token”。
- **Hierarchical Semantic Selection（分层语义选择）**  
  构建 **Page → Chunk → Grid** 三级逻辑结构，自顶向下进行粗到细的语义匹配：
  - 使用 Key-Key Semantic Affinity（$S(u) = \mathbf{v}_{\text{anchor}} \cdot \mathbf{v}_u$）作为相似性度量。
  - 利用 mean-pooling + flatten 构建各层级的语义向量 $\mathbf{v}_p, \mathbf{v}_c, \mathbf{v}_g$。
- **Uncertainty-aware Backtracking（质量感知回溯）**  
  监控生成过程中的 **Entropy** 和 **Varentropy**，仅当不确定性过高时才触发 full context 重建，保证鲁棒性。

#### ✅ 系统层面
- **Page-aligned & Zero-copy Execution**  
  基于 **PagedAttention**，操作单位为 page（默认 32 tokens），通过操纵逻辑页索引实现“零拷贝”选择，避免物理数据移动。
- **Batched GEMM-based Selection**  
  将所有层级的语义向量拼接成一个大张量，用单次 GEMM 完成全部相似度计算，最大化 GPU 利用率。
- **CUDA Graph 封装全流程**  
  将 selection + attention 整合进 CUDA Graph，消除 kernel launch 开销，支持高效 batching。

### 相比现有方法的优势
| 维度 | CHESS | 其他方法（如 H2O, SnapKV, KeyDiff） |
|------|-------|-------------------------------|
| **Selection Granularity** | Block/Page-level | Mostly token-level |
| **Context Awareness** | ✅ 动态适配当前 query | ❌ 全局静态或 attention-based |
| **System Efficiency** | ✅ Zero-copy + GEMM + CUDA Graph | ❌ 频繁 copy + 控制流分支 |
| **Quality Preservation** | ✅ 层级保留语义连续块 + 回溯机制 | ❌ 孤立 token 易造成 context dilution |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **LongBenchV2**（Bai et al., 2025）：用于评估 **generation quality**，涵盖多种长文本任务（QA、ICL、代码、对话等），按难度和长度分组。
- **Synthetic Workloads**：用于评估 **system efficiency**（吞吐量、延迟），输入长度从 4k 到 32k 不等，batch size 覆盖 1–192。

### 实验设置和评估指标

| 类别 | 指标 |
|------|------|
| **Quality Metrics** | Accuracy / Score on LongBenchV2（Overall, Easy/Hard, Short/Med/Long） |
| **Efficiency Metrics** | <br>- **Throughput**（tokens/sec）<br>- **Speedup**（vs. Full-KV）<br>- **Time Per Output Token (TPOT)**<br>- **Latency Breakdown**（selection overhead） |
| **Hardware** | 4× H20 GPUs（主实验），补充测试在 A800 上 |
| **Software** | PyTorch 2.5.1, CUDA 12.4, 基于 nanoVLLM 实现 |

### 基线方法对比
- **Full-KV**：保留完整 KV Cache，作为性能上限基准。
- **H2O**（Zhang et al., 2023）：基于累计 attention score 的 “Heavy Hitter” 机制。
- **KeyDiff**（Park et al., 2025）：基于 key matrix 差异性的 token 选择。
- **SnapKV**（Li et al., 2024）：基于 attention cluster 的上下文压缩。
- **Quest**（Tang et al., 2024）：query-aware 的 block-level pruning。

> 所有 baseline 均集成至统一评估框架，Quest 因仅支持 batch=1 单独运行。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📊 质量表现（LongBenchV2）
| 方法 | KV Cache Budget | Overall Score |
|------|------------------|---------------|
| Full-KV | 100% | 30.2 |
| H2O (best) | 20% | 34.0 |
| Quest (best) | 2048 tokens (~1%) | 32.0 |
| **CHESS (Aggressive)** | **1%** | **33.2** ✅ |

> 🔥 **关键发现**：CHESS 在仅使用 **1% KV Cache** 的情况下，**超越 Full-KV 基线**，接近最强 baseline H2O（后者使用 20× 更多缓存）。

#### 🚀 吞吐量提升（End-to-End Throughput Speedup）
- 最高达到 **4.56×** 吞吐加速（vs. Full-KV），在 32k context + 大 batch 场景下。
- 在 A800 上进一步提升至 **5.72×**。
- 随着 batch size 增大，CHESS 性能优势持续扩大，而其他方法出现 diminishing returns 或 OOM。

#### ⏱️ 延迟稳定性
- **TPOT（Time Per Output Token）几乎恒定**，不受序列增长影响。
- Full-KV 和 SnapKV 表现出明显的线性延迟上升。
- CHESS 的 selection overhead 极低：
  - 在 32k context 下仅占总延迟 **1.49%**。

#### 🔍 消融实验与分析（见 Appendix）
- **动态回溯优于周期重建**：即使 baseline 更频繁地重建 context，CHESS 的动态策略仍取得更高性能（Fig. 9）。
- **非单调质量曲线揭示三个阶段**：
  1. **Distraction Phase**：中等 budget 引入噪声 token 导致性能下降。
  2. **Contextual Recovery**：足够容量恢复语境后性能回升。
  3. **Attention Dilution**：过大缓存稀释注意力，趋近 Full-KV。
- **CHESS 在极端稀疏（1%）下仍保持最优**，说明其能有效过滤干扰信息。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Token 重要性是动态且上下文相关的**，静态或全局选择策略无法适应生成过程中语义焦点的变化。
2. **Context-aware selection + page-level pruning** 可同时实现高质量与高效率：
   - 语义连贯的 context block 比孤立 token 更有利于 long-context reasoning。
   - page-aligned 设计天然契合现代 inference engine（如 PagedAttention），实现 zero-copy 加速。
3. **理论稀疏性必须转化为实际 wall-clock speedup**，这需要算法与系统的深度协同设计（co-design）。
4. **极低 KV budget（1%）下仍可超越 Full-KV**，表明原始 full context 包含大量冗余甚至有害信息（distraction），合理剪枝反而提升推理能力。

### 方法的局限性
- 当前 selection metric 依赖 Key-Key Semantic Affinity，在某些高度离散的任务（如 code）上略逊于 H2O。
- 回溯机制依赖 offline-calibrated 阈值（99% percentile of entropy/varentropy），可能需针对不同模型或任务微调。
- 目前未整合 speculative decoding 或 RAG pipeline，未来可扩展应用场景。

### 未来工作方向
- 结合 **speculative decoding** 进一步加速推理。
- 集成至 **RAG-augmented pipelines**，优化外部知识检索与内部 context selection 的协同。
- 探索更丰富的 **uncertainty signals**（如 gradient norm, activation variance）用于 adaptive reconstruction。
- 支持 **multi-modal context**（text + image + audio）下的 hierarchical selection。

---

> 💡 **一句话总结**：  
> CHESS 通过 **context-aware 分层语义选择 + page-aligned zero-copy 系统实现**，在仅用 **1% KV Cache** 的条件下实现了 **超越 Full-KV 的生成质量** 和高达 **4.56× 的吞吐提升**，为 long-context LLM inference 提供了一种高效、稳定、可扩展的新范式。

</details>

---

### 10. [PyVision-RL: Forging Open Agentic Vision Models via RL](https://arxiv.org/abs/2602.20739)

**Authors**: Shitian Zhao, Shaoheng Lin, Ming Li, Haoquan Zhang, Wenshuo Peng, Kaipeng Zhang, Chen Wei  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.20739v1  

#### Abstract
Reinforcement learning for agentic multimodal models often suffers from interaction collapse, where models learn to reduce tool usage and multi-turn reasoning, limiting the benefits of agentic behavior. We introduce PyVision-RL, a reinforcement learning framework for open-weight multimodal models th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PyVision-RL: Forging Open Agentic Vision Models via RL**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 **Reinforcement Learning (RL)** 的 **agentic multimodal models** 在训练过程中普遍存在 **interaction collapse** 问题，即模型倾向于减少工具调用（tool usage）和多轮推理（multi-turn reasoning），导致无法充分发挥“智能体”行为的潜力。这一现象在视频理解等长序列任务中尤为严重，限制了模型的推理深度和表现。

此外，现有方法大多依赖 **静态工具集（static toolsets）**，缺乏灵活性；而动态工具调用虽具表达力，但在 **开放权重（open-weight）模型** 上的 RL 训练仍不成熟，尤其对视频任务支持不足。

---

### **提出的新方法与新思路**
本文提出了 **PyVision-RL** ——一个面向开放权重多模态大模型的统一 **agentic RL 框架**，其核心创新包括：

#### ✅ **(1) 动态工具化（Dynamic Tooling）以 Python 为原语**
- 将 **Python** 视为基本工具，允许模型在运行时动态生成代码来处理图像或视频。
- 支持任意组合操作（如裁剪、绘图、统计分析、帧采样等），极大提升灵活性与泛化能力。
- 统一支持 **PyVision-Image**（图像理解） 和 **PyVision-Video**（视频理解）两个子系统。

#### ✅ **(2) 防止交互崩溃的 RL 机制设计**
- **累积工具奖励（Accumulative Tool Reward）**：
  - 在最终准确奖励 $ R_{acc} \in \{0,1\} $ 的基础上，增加一项与工具调用次数成正比的奖励项：  
    $$
    R = R_{acc} + 0.1 \cdot n_{tc} \cdot \mathbb{1}\{R_{acc}=1\}
    $$
  - 只有当答案正确时才鼓励更多工具使用，避免无效行为被强化。
- 此机制显著提升了多轮交互的持续性和稳定性。

#### ✅ **(3) 过采样-过滤-排序 Rollout 策略（Oversampling-Filtering-Ranking）**
- 在每轮 RL 中先过采样多个 rollout；
- 过滤掉执行失败或全组无差异的样本；
- 按组内奖励标准差排序，优先选择“难度适中”的样本进行训练（类似课程学习）；
- 引入 **Standard Deviation Sorting** 提高训练信号质量，缓解 group-level normalization 导致的负优势问题。

#### ✅ **(4) 按需上下文构建（On-Demand Context Construction）——专用于视频**
- 不将视频帧直接注入 MLLM 上下文，而是仅加载到 **Python runtime**；
- 模型通过 Python 代码按需提取并绘制相关帧（`fetch_frames_and_plot`）；
- 极大降低视觉 token 消耗，实现高效推理。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | PyVision-RL |
|------|--------|-----------|
| 工具范式 | 静态工具集（如 crop/zoom） | 动态 Python 工具调用，灵活可组合 |
| 训练稳定性 | 易出现 interaction collapse | 通过 accumulative tool reward 抑制崩溃 |
| 视频处理效率 | 均匀抽帧 → 高 token 开销 | 按需抽取关键帧 → 节省 90%+ token |
| 开放性 | 多依赖闭源 API | 全流程开源，支持 open-weight 模型 |
| 统一性 | 图像/视频方案割裂 | 统一框架支持两类任务 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

| 模型 | SFT 数据 | RL 数据 |
|------|---------|--------|
| **PyVision-Image** | - MMK12（多模态推理）<br>- GMAI-Reasoning（医疗）<br>- ChartQA / InfoVQA（图表）<br>- MMPR（通用 VQA） | - DeepEyes, Mini-o3（视觉搜索）<br>- V-Thinker, WeMath（多模态数学） |
| **PyVision-Video** | - SpaceR（空间推理）<br>- LongVILA（长视频理解） | - SpaceR（15K 样本） |

> 所有数据均基于合成方式由 GPT-4.1 生成，并经过严格过滤确保高质量。

---

### **实验设置与评估指标**

#### **基础架构**
- 主干模型：**Qwen2.5-VL-7B**
- RL 算法：改进版 **GRPO**（移除 advantage 中的标准差归一化）
- 训练步数：700 步
- 硬件：8×H100 GPU
- 超参数：group size=8, batch size=16, lr=1e-6

#### **评估协议**
- 最大回合数（max turn budget）：30
- 上下文长度上限：32K tokens
- 温度设置：
  - PyVision-Image：V* 使用 `temp=0.01`，其余 `temp=0.5, top-k=20`
  - PyVision-Video：统一 `temp=0.01`

#### **评估基准**
| 类别 | 基准名称 | 描述 |
|------|--------|------|
| **视觉搜索** | V*, HRBench-4K/8K | 测试细粒度定位与主动感知能力 |
| **多模态推理** | MathVerse, MathVision, WeMath, DynaMath | 数学图文联合推理 |
| **代理式推理** | TIR-Bench | 多轮工具调用任务 |
| **空间推理（视频）** | VSI-Bench | 视频中的物体计数、距离估计、路线规划等 |

---

### **基线方法对比**
| 类型 | 方法 | 特点 |
|------|------|------|
| **静态工具集** | Pixel-Reasoner, Mini-o3, DeepEyes(-v2), VITAL | 固定工具（如 crop/zoom/video clip） |
| **动态工具调用** | Thyme, CodeV, CodeDance, CodeVision, DeepEyes-v2 | 使用 Python 作为工具 |
| **纯文本推理** | Video-R1 | 无工具调用，仅靠内部表示推理 |
| **强 baseline** | Qwen2.5-VL-7B | 原始模型，未进行 RL 微调 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 📊 **PyVision-Image 性能汇总（Table 1）**

| Benchmark | Qwen2.5-VL-7B | PyVision-Image | 提升幅度 |
|----------|----------------|----------------|---------|
| **V*** (avg@32) | 78.5 | **88.7** | **+10.2%** |
| HRBench-4K | 71.6 | **78.1** | +6.5% |
| HRBench-8K | 67.9 | **74.3** | +6.4% |
| **DynaMath** | 53.3 | **61.6** | +8.3% |
| **MathVerse** | 45.6 | **55.8** | +10.2% |
| **WeMath** | 34.6 | **47.7** | **+13.1%** |
| **TIR-Bench** | 16.0 | **19.8** | +3.8% |

> ✅ 在所有类别上达到 **state-of-the-art** 表现。

---

#### 📊 **PyVision-Video 在 VSI-Bench 上的表现（Table 2）**

| 方法 | Avg. Score | Obj. Count | Rel. Dir. | Route Plan | Appr. Order |
|------|------------|-------------|-----------|------------|--------------|
| Qwen2.5-VL-7B | 36.7 | 41.9 | 38.5 | 29.9 | 34.1 |
| VITAL (预定义剪辑工具) | 41.8 | – | – | – | – |
| **PyVision-Video** | **44.0** | **53.8** | **46.3** | **26.3** | **58.6** |

> ✅ 超越最强 agent 基线 **VITAL (+2.2%)**，并在多个子任务大幅领先。

---

### **效率对比（Token Usage）**

| 方法 | 平均视觉 token / 样本 | 准确率（VSI-Bench） |
|------|------------------------|--------------------|
| Qwen2.5-VL-7B (@1FPS) | ~45K | 38.0% |
| Video-R1 | ~25K | ~37% |
| SpaceR | ~25K | 45.6% |
| **PyVision-Video** | **~5K** | **44.0%** |

> ⚡️ **PyVision-Video 仅用约 1/9 的视觉 token 即达到更高准确率**，实现最优 **accuracy-efficiency trade-off**。

---

### **消融实验结果（Ablation Study）**

#### 🔍 **组件消融（Fig. 5 & Table 3）**

| 设置 | 相较于完整模型下降幅度（~600步） |
|------|-------------------------------|
| 移除 **accumulative tool reward** | ↓ ~1.93% |
| 最大回合数从 4 降到 2 | ↓ ~1.93% |
| 移除 **standard deviation sorting** | ↓ ~2–3%（早期影响大） |
| 保留 **std normalization** | 导致训练波动，收敛困难 |

> ✅ **Accumulative Tool Reward** 是维持长期工具使用的最关键因素；
> ✅ 更大的 **turn budget** 决定了性能上限；
> ✅ **Standard Deviation Sorting** 显著减少“正确但被惩罚”的样本比例（见 Fig. 6）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Interaction collapse 可被有效抑制**：
   - 通过 **accumulative tool reward** 和合理的 rollout 选择策略，可以稳定地训练出具有多轮工具调用能力的 agent。
2. **动态工具调用优于静态工具集**：
   - Python-based dynamic tooling 在灵活性、泛化性和性能上全面超越 hand-crafted 工具。
3. **On-demand context construction 极大提升效率**：
   - 对视频任务而言，“按需取帧”比均匀抽帧更符合认知逻辑，节省大量 token。
4. **RL 训练动态稳定且持续进步**：
   - 如 Fig. 3 所示，entropy loss、gradient norm 平稳下降，tool call 数量、response length 和 accuracy 持续上升，表明训练有效。

---

### **方法的局限性**
- **安全性风险**：由于使用 Python 执行环境，存在潜在的 **host system access 风险**（如文件读写、命令执行），部署需严格沙箱隔离。
- **依赖高质量 SFT 初始化**：需要大量合成 SFT 数据来冷启动 multi-turn tool use 能力。
- **计算成本较高**：RL 训练涉及大量代码执行与环境交互，rollout 成本远高于 SFT。

---

### **未来工作方向**
- 探索更安全的受限执行环境（如 WASM-based sandbox）；
- 将框架扩展至其他模态（音频、3D 场景）；
- 结合 test-time scaling（如 self-reflection, search）进一步释放 agent 潜力；
- 构建更大规模的 agentic multimodal benchmark（如 TIR-Bench 的扩展）。

---

## **项目资源**
- **代码 & 模型开源地址**：[https://github.com/agents-x-project/PyVision-RL](https://github.com/agents-x-project/PyVision-RL)
- 包含完整的训练 pipeline、prompt 设计、evaluation script 和 checkpoint。

--- 

> 💡 **一句话总结**：  
> **PyVision-RL 证明了，在合适的激励机制和训练策略下，开放权重多模态模型可以通过 RL 学会稳定、高效的多轮工具调用行为，成为真正意义上的“视觉智能体”，并在性能与效率之间取得突破性平衡。**

</details>

---

### 11. [Nonparametric Teaching of Attention Learners](https://arxiv.org/abs/2602.20461)

**Authors**: Chen Zhang, Jianghui Wang, Bingyang Cheng, Zhongtao Chen, Wendong XU, Cong Wang, Marco Canini, Francesco Orabona, Yik Chung WU, Ngai Wong  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.20461v1  

#### Abstract
Attention learners, neural networks built on the attention mechanism, e.g., transformers, excel at learning the implicit relationships that relate sequences to their corresponding properties, e.g., mapping a given sequence of tokens to the probability of the next token. However, the learning process...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Nonparametric Teaching of Attention Learners》总结

## 1. 论文的主要贡献和创新点

### 解决的问题
- **Attention Learners**（如Transformer、LLMs、ViTs）在训练过程中通常需要大量计算资源和时间，尤其是在大规模任务上进行预训练或微调时，训练成本高昂。
- 现有的**机器教学**（Machine Teaching）和**非参数化教学**（Nonparametric Teaching）方法主要针对多层感知机（MLP）等简单模型，无法直接应用于具有注意力机制的复杂网络。

### 提出的新方法和新思路
- 提出了 **Attention Neural Teaching (AtteNT)**，一种基于**非参数化教学理论**的新型教学范式，用于提升Attention Learners的学习效率。
- 首次从理论上证明了：
  - 在参数空间中，注意力机制通过自适应地为输入序列中的不同元素分配重要性，驱动了参数梯度更新。
  - 这种参数空间的演化过程与函数空间中的**功能梯度下降**（functional gradient descent）是一致的。
  - 动态的**Attention Neural Tangent Kernel (ANTK)** 收敛于非参数化教学中使用的**重要性自适应规范核**（importance-adaptive canonical kernel）。
- 因此，可以将Attention Learner的训练过程解释为一个“老师”选择最具信息量的样本（即预测误差最大的样本）来加速学习的过程。

### 相比现有方法的优势
- **理论统一性**：首次建立了非参数化教学理论与Attention Learner训练之间的桥梁，为理解注意力模型的优化提供了新的理论视角。
- **高效性**：通过智能的数据选择策略，显著减少了训练时间，而无需修改模型架构或引入额外的复杂性。
- **通用性**：该框架不依赖于特定的注意力变体（如multi-head），可广泛适用于各种基于注意力的模型。
- **性能保持甚至提升**：不仅没有牺牲精度，反而在多个下游任务上观察到性能的稳定保持或提升。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **自然语言处理 (NLP) 任务**：
  - **数学推理**：在 `MetaMathQA` 上微调，在 `GSM8K` 和 `MATH` 上评估。
  - **代码生成**：在 `CodeFeedback` 上微调，在 `HumanEval` 和 `MBPP` 上评估。
  - **多轮对话**：在 `WizardLM-Evol-Instruct` 上微调，在 `MT-Bench` 上评估。
- **计算机视觉 (CV) 任务**：
  - **图像分类**：在 `ImageNetS50` 上评估。
  - **语义分割**：在 `NYUv2(S)` 上评估，指标为 mIoU。
  - **深度估计**：在 `NYUv2(D)` 上评估，指标为 δ1（误差低于1.25的比例）。

### 实验设置和评估指标
- **模型**：
  - **LLM场景**：对 `LLaMA 2-7B`, `Mistral-7B`, `Gemma-7B` 进行微调。
  - **CV场景**：使用 `Multi-Modal MAE`（基于ViT-B）进行从零开始的预训练。
- **训练设置**：
  - LLM微调：5个epoch，使用LoRA。
  - ViT预训练：800个epoch，batch size为2048。
- **AtteNT策略**：
  - **LLM**：第一轮在全数据集上训练，后续轮次根据每条样本的损失分数选择最困难的70%样本。
  - **ViT**：采用动态选择策略，初始选择率较低，并随训练进程逐步增加，同时定期重新计算损失以更新选择子集。
- **评估指标**：
  - **准确性**：GSM8K, MATH, HumanEval, MBPP, MT-Bench, ImageNetS50。
  - **mIoU**：NYUv2(S)。
  - **δ1**：NYUv2(D)。
  - **训练时间**：平均微调时间（LLM）、预训练总耗时（ViT）。

### 基线方法对比
- **标准训练**：在整个数据集上进行常规训练。
- **其他采样方法**（在消融实验中对比）：
  - **Class Weight Sampling**：按类别频率的倒数加权采样。
  - **Fixed Weight Sampling**：为不同模态（RGB, SemSeg, Depth）设定固定的采样权重。
  - **GradNorm Sampling**：根据不同任务组的梯度范数动态调整采样权重。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **LLM 微调**：
  - **训练时间减少**：平均减少 **13.01%**（原文Table 1中计算得出，例如LLaMA从246分钟降至213分钟）。
  - **性能提升**：在所有任务和模型上均实现增益。
    - **GSM8K**：最高提升2.14（Mistral-7B）。
    - **MATH**：最高提升2.89（Mistral-7B）。
    - **HumanEval**：最高提升3.66%（LLaMA 2-7B）。
    - **MBPP**：最高提升3.31%（Gemma-7B）。
- **ViT 从零训练**：
  - **训练时间减少**：节省 **20.58%**（从1234分钟降至980分钟）。
  - **性能提升**：
    - **ImageNetS50**：92.2 → 92.3 (+0.1)。
    - **NYUv2(S)**：51.9 → 52.6 (+0.7)。
    - **NYUv2(D)**：52.1 → 57.2 (**+5.1**)，提升最为显著。

### 与基线方法的对比结果
- 如 **Table 8** 所示，AtteNT在所有对比方法中表现最佳：
  - **训练时间更短**：AtteNT (980m) < GradNorm (1112m) < Class Weight (1108m) < Fixed Weight (1065m)。
  - **性能更高**：AtteNT在ImageNetS50、NYUv2(S)和NYUv2(D)上的得分均高于所有基线。
- 结论：AtteNT的优势并非来自简单的贪心采样启发式，而是源于其**原则性的非参数化教学机制**，能更有效地适应模型不确定性和任务交互。

### 消融实验结果
- **采样策略**：
  - **Soft (Gumbel-Top-k)** 策略优于确定性的 **Hard** 策略和随机的 **Random** 策略，因为它引入了概率性，增强了训练的鲁棒性。
- **选择间隔**：
  - **Incremental**（增量更新）策略优于 **Fixed**（固定更新）策略，表明定期重新评估样本难度是必要的。
- **最终配置**：
  - 最优配置为 **(Incremental Ratio, Incremental Interval, Soft Selection)**，该配置在减少训练时间和提升下游任务性能之间取得了最佳平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **理论一致性**：Attention Learner的参数梯度更新过程在函数空间中与非参数化教学的功能梯度下降是**一致的**。这为将教学理论应用于复杂神经网络提供了坚实的理论基础。
2. **教学有效性**：AtteNT通过选择“最难”的样本（即当前模型预测误差最大的样本），构建了一个有效的**课程学习**（curriculum learning）效应，避免了对已掌握样本的梯度稀释。
3. **高效且无损**：该方法在**不损害甚至提升模型性能的前提下**，实现了显著的训练加速（LLM微调-13.01%，ViT训练-20.58%）。
4. **普适性强**：该方法在NLP和CV两大领域的多种模型和任务上都验证了其有效性。

### 方法的局限性
- **依赖于损失计算**：AtteNT需要在每个选择周期计算所有候选样本的损失，这会带来一定的额外计算开销，尽管远小于完整训练的成本。
- **超参数敏感**：选择比率（ratio）和更新间隔（interval）等超参数的选择会影响最终效果，需要根据具体任务进行调整。
- **理论假设**：部分理论推导基于凸损失函数等理想化假设，实际应用中可能存在偏差。

### 未来工作方向
- **扩展到更多模型**：探索AtteNT在图注意力网络（Graph Attention Networks）等其他注意力变体上的应用。
- **噪声标签鲁棒性**：研究在真实世界存在标签噪声的情况下，AtteNT的鲁棒性，并结合最新的抗噪学习技术进行改进。
- **实际应用**：探索AtteNT在提高数据驱动方法（如world models）效率方面的潜力。
- **更高效的实现**：设计近似算法以降低选择高损失样本的计算成本。

</details>

---

### 12. [Physics-based phenomenological characterization of cross-modal bias in multimodal models](https://arxiv.org/abs/2602.20624)

**Authors**: Hyeongmo Kim, Sohyun Kang, Yerin Choi, Seungyeon Ji, Junhyuk Woo, Hyunsuk Chung, Soyeon Caren Han, Kyungreem Han  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.20624v1  

#### Abstract
The term 'algorithmic fairness' is used to evaluate whether AI models operate fairly in both comparative (where fairness is understood as formal equality, such as "treat like cases as like") and non-comparative (where unfairness arises from the model's inaccuracy, arbitrariness, or inscrutability) c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Physics-based phenomenological characterization of cross-modal bias in multimodal models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**多模态大语言模型（MLLMs）中的跨模态偏差（cross-modal bias）问题**，即在融合多种输入模态（如文本、图像、音频）时，模型往往过度依赖某一主导模态（如文本），而忽略其他模态的信息，导致决策不公、鲁棒性差。这种偏差在传统基于准确率的评估中难以被发现。

具体而言，论文指出：
- 多模态输入并未真正实现“互补”，反而可能**强化单一模态的主导地位**；
- 当前主流的嵌入层或表示层面分析（embedding-level analysis）无法充分揭示此类系统性偏差的动态机制。

### 🆕 提出的新方法与新思路
作者提出了一种**基于物理现象学（physics-based phenomenological）的解释框架**，其核心创新包括：

1. **引入“物理代理模型”（physical surrogate model）**  
   构建一个**multi-oscillator 动力学系统**来模拟 Transformer 中的 self-attention 和 cross-attention 机制，将语义表征演化过程类比为耦合振荡器的动力学行为。该模型能捕捉到注意力机制中非线性的交互动态。

2. **采用现象学视角进行可解释性分析**  
   不同于传统的认知主义符号解释（cognitivist symbolic account），本文强调从机器在训练/推理过程中实际经历的“内部物理实体”出发，关注其动力学轨迹而非外部世界的符号映射，从而更真实地刻画模型内部偏差的生成机制。

3. **设计标签扰动策略（label perturbation）以暴露隐含偏好层级**  
   通过系统性地禁止模型选择某些情绪类别，观察其错误预测是否集中于特定“备选”类别，进而识别出**错误吸引子结构（error-attractor patterns）**，揭示模型内在的偏见层次。

### 🔍 相比现有方法的优势
| 方面 | 传统方法局限 | 本论文优势 |
|------|--------------|-----------|
| 分析粒度 | 聚焦整体准确率或静态表示空间 | 揭示失败情形下的**动态偏差结构** |
| 可解释性 | 黑箱式归因（如梯度、SHAP） | 基于物理系统的**机制级建模与可视化** |
| 偏差检测能力 | 难以发现无显式群体划分的非比较型不公平（non-comparative unfairness） | 能识别由任意性、不可解释性引发的深层不公平 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **CREMA-D**（Crowdsourced Emotional Multimodal Actors Dataset）  
  包含 7,442 个视频样本，来自 91 名演员，表达六种基本情绪：`happy`, `neutral`, `sad`, `angry`, `disgust`, `fear`。每个样本配有同步的面部视频和语音音频，并有多个标注者提供感知情绪标签（multimodal-perceived, visual-perceived, audio-perceived）。

### ⚙️ 实验设置
#### （1）零样本情绪分类任务（Zero-shot Emotion Classification）
- **模型**：Qwen2.5-Omni 和 Gemma 3n（两种架构不同的 MLLMs）
- **输入条件对比**：
  - 视频 + 音频（Face + Voice）
  - 仅视频（Face-only，音频替换为静音）
  - 仅音频（Voice-only，视频帧替换为空白占位符）
- **提示模板统一**（见 Table 1），要求输出 JSON 格式的情绪判断，禁止额外说明。

#### （2）标签扰动分析（Label Perturbation Analysis）
- 系统性移除部分情绪选项（每次排除 1–4 个类别），迫使模型在受限输出空间中做决策。
- 分析错误预测是否呈现结构性回退（如总是 fallback 到 `neutral` 或 `happy`），构建**错误吸引子图**（error-attractor graphs）和 Sankey 图。

#### （3）物理代理模型实验：Lorenz 混沌时间序列预测
- 构建 multi-oscillator 动力学系统模拟 Transformer 注意力机制：
  - 子群 X 受 Lorenz 系统 $ x(t) $ 驱动
  - 子群 Y 受 $ y(t) $ 驱动
  - 目标是预测 $ z(t) $
- 引入 **dynamical SHAP** 来量化各模态对预测的贡献度。
- 控制变量：self-attention 强度 $ B_{\text{self}} $ 和 cross-attention 强度 $ B_{\text{cross}} $

### 📊 评估指标
| 实验类型 | 主要指标 |
|--------|---------|
| 情绪分类 | 错误转移频率、Sankey 流宽度、错误吸引子结构 |
| 动力学模拟 | Normalized Mean Squared Error (NMSE)、dynamical SHAP 差异 $ \phi(Y) - \phi(X) $ |

### ❌ 基线方法对比
本文未直接对比传统 ML 基线（如 SVM、CNN-LSTM），而是：
- 对比不同输入模态组合下的同一模型表现（Video+Audio vs. Video-only vs. Audio-only）
- 在物理代理模型中对比不同注意力强度配置下的模态偏好变化
- 本质上是在**挑战“多模态一定优于单模态”的默认假设**

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据与发现

#### （1）情绪分类实验结果（Qwen2.5-Omni & Gemma 3n）
| 发现 | 具体表现 |
|-----|--------|
| **显著模态偏差存在** | 两模型均表现出强烈偏向 `neutral` 类别的倾向，尤其在 Face-only 和 Face+Voice 条件下 |
| **错误具有结构化而非随机性** | 即使禁止选择首选类别，模型也不会均匀分布错误，而是集中 fallback 到次优类别（如 `happy` → `neutral`） |
| **Video+Audio 输入更接近 Video-only 而非 Audio-only** | 表明视觉模态占据主导地位，添加音频并未纠正偏差，反而被抑制 |
| **Gemma 3n 在 Voice-only 下极端偏向 `neutral`** | 但在 Face+Voice 中此偏差消失，说明视觉信息压制了听觉偏差 |

> ➤ 图 3 显示：随着被禁用标签增多，错误持续向 `neutral` 和 `happy` 收敛，形成稳定偏好层级。

#### （2）物理代理模型实验结果（Lorenz 预测）
| 设置 | NMSE | 模态偏好（SHAP 差异） | 结论 |
|------|------|------------------------|-------|
| 低 $ B_{\text{self}}, B_{\text{cross}} $ (1e-4) | 高 (~0.12) | X 明显主导（左向箭头） | 注意力弱时，系统无法有效整合 Y 模态 |
| 高 $ B_{\text{self}}, B_{\text{cross}} $ (100,100) | 最低 (~0.03) | X 与 Y 贡献均衡（箭头居中） | 强注意力促进平衡融合，重建混沌吸引子结构良好 |

> ➤ 图 5(c) 显示高注意力条件下预测几乎完美复现目标轨迹。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **多模态输入不一定缓解模态偏差，反而可能固化主导模态**  
   添加次要模态不会自动带来“纠错”效果，反而可能因交叉注意力失衡而导致主模态进一步垄断决策路径。

2. **Transformer 内部存在结构性错误吸引子**  
   模型在失败时并非随机出错，而是遵循固定的偏好层级（如 neutral > happy > sad），这反映了其语义网络中的不对称连接结构。

3. **adequate self- and cross-attention 是防止偏差的关键**  
   物理代理实验证明：只有当 self-attention 和 cross-attention 达到足够强度时，才能实现真正的跨模态平衡融合；否则系统会退化为单模态主导模式。

4. **传统评估指标无法捕获非比较型不公平（non-comparative unfairness）**  
   尽管整体准确率尚可，但模型在不确定性下的任意性（arbitrariness）和不可解释性（inscrutability）构成了深层次的算法不公平。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **代理模型简化了真实 Transformer 结构** | 忽略了 FFN 层细节、LayerNorm 的非线性影响等，仅为理想化近似 |
| **仅验证于特定任务（情绪识别、时间序列预测）** | 是否普适于其他多模态任务（如 VQA、医疗诊断）需进一步验证 |
| **缺乏干预后的去偏实验** | 提出的是诊断工具，尚未展示如何利用该框架主动修正偏差 |
| **现象学立场较哲学化** | “机器体验”概念仍具争议，工程社区接受度有待检验 |

---

### 🔮 未来工作方向
1. **发展基于物理代理的偏差校正机制**  
   利用 multi-oscillator 模型指导注意力权重初始化或正则化，实现动态平衡调节。

2. **扩展至更多模态与任务场景**  
   如图文问答（VQA）、医学影像+报告诊断等，验证跨领域适用性。

3. **结合因果推断与动力学建模**  
   探索 cross-attention 中的因果流向，识别偏差传播路径。

4. **构建标准化的“偏差动力学测试平台”**  
   提供开源工具包用于检测新 MLLM 的 error-attractor 结构和模态偏好稳定性。

---

> 💬 **一句话总结**：  
> 本文突破传统表示分析范式，提出一种基于物理现象学的动态建模范式，揭示 MLLMs 中跨模态偏差的本质是**注意力动力学失衡所致的结构性错误吸引子**，并证明多模态融合未必公平——有时只是让强者更强。

</details>

---

### 13. [On Electric Vehicle Energy Demand Forecasting and the Effect of Federated Learning](https://arxiv.org/abs/2602.20782)

**Authors**: Andreas Tritsarolis, Gil Sampaio, Nikos Pelekis, Yannis Theodoridis  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.20782v1  

#### Abstract
The wide spread of new energy resources, smart devices, and demand side management strategies has motivated several analytics operations, from infrastructure load modeling to user behavior profiling. Energy Demand Forecasting (EDF) of Electric Vehicle Supply Equipments (EVSEs) is one of the most cri...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《On Electric Vehicle Energy Demand Forecasting and the Effect of Federated Learning》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本论文聚焦于**Electric Vehicle Supply Equipment (EVSE)** 的 **Energy Demand Forecasting (EDF)** 问题。该问题是智能电网管理中的关键挑战，原因在于：
- EV 充电行为具有高度**间歇性 (intermittent)** 和 **不规则性 (lumpy)**；
- 用户行为、天气、充电习惯等外部因素导致负荷模式复杂；
- 数据因隐私法规（如 GDPR）而分散在不同设备或站点（data silos），难以集中训练。

传统集中式机器学习面临**隐私泄露风险**和**通信开销大**的问题，因此需要一种兼顾**预测精度、隐私保护和能源效率**的解决方案。

### 提出了什么新方法或新思路
论文系统性地比较了多种时间序列预测模型在 **Centralized Learning** 和 **Federated Learning (FL)** 范式下的表现，并提出了一个统一的实验框架 **[Fed]EDF**，其核心创新包括：

- **首次联合分析 EDF 中的“准确性 vs. 能源消耗”权衡**：不仅评估预测误差，还量化了训练过程中的 **energy consumption** 和 **CO₂-equivalent emissions**。
- **引入 Federated Learning 架构用于 EVSE 预测**：将 EVSE 按地理位置聚类为多个 **EVSE Hubs**，作为 FL 中的客户端，实现去中心化建模。
- **全面覆盖模型谱系**：从统计模型（ARIMA）、传统 ML（XGBoost）到深度神经网络（LSTM, GRU, BiLSTM, BiGRU），并在 FL 下实现可比实验。

### 相比现有方法的优势
- **更贴近现实部署场景**：考虑了数据碎片化、隐私约束和边缘计算资源限制。
- **多维度评估体系**：超越单一准确率指标，综合考量 **accuracy, privacy, energy efficiency, carbon footprint**。
- **开源复现性强**：作者公开了全部代码（GitHub），提升了研究透明度与可重复性。

---

## 2. 核心实验方法和设置

### 使用的数据集
共使用四个真实世界 EVSE 数据集，涵盖不同城市与时间段：

| 数据集 | 时间跨度 | #EVSEs | 地理位置 | 类型 |
|-------|--------|--------|--------|------|
| **Dundee** | 696 天 | 67 | 苏格兰 | 开放数据 |
| **FEUP** | 586 天 | 12 | 葡萄牙波尔图 | 私有数据 |
| **Boulder** | 1185 天 | 27 | 美国科罗拉多州 | 开放数据 |
| **Palo Alto** | 3444 天 | 47 | 美国加州 | 开放数据 |

所有数据均经过清洗、聚合为每 12 小时的时间序列，并提取特征（如小时、星期、downtime、session count 等）。

### 实验设置和评估指标

#### 模型类别
- **Statistical Models**: ARIMA, SARIMA, SARIMAX
- **Machine Learning**: XGBoost
- **Neural Networks**: LSTM, GRU, BiLSTM, BiGRU（含 Embedding 层处理位置/型号）
- **Federated Variants**: FedAvgXGB, FedProxXGB, FedLSTM, FedGRU 等

#### FL 设置
- 客户端数 $N$：8（Dundee/Boulder/Palo Alto），4（FEUP）
- 每轮本地 epoch 数：5（“heavy”）或 1（“light”）
- 总通信轮次：50
- 聚合算法：FedProx（用于 NN），FedXGBllr（用于 XGBoost）

#### 评估指标（预测未来 12 小时需求）
| 指标 | 描述 |
|------|------|
| **MASE** | Mean Absolute Scaled Error（越低越好） |
| **SMAPE** | Symmetric MAPE（对称百分比误差） |
| **MAAPE** | Mean Arctangent Absolute Percentage Error |
| **WAPE** | Weighted Absolute Percentage Error |
| **RMSE / MAE** | 绝对误差（单位：kW） |
| **R²** | 决定系数（越高越好） |
| **Energy Consumption & CO₂e** | 训练能耗与碳排放（新增维度） |

#### 基线方法对比
- **Centralized Baseline**: 单一全局模型训练（XGBoost/LSTM 等）
- **Federated Baseline**: 各客户端独立训练 + 联邦聚合
- **Statistical Baseline**: ARIMA/SARIMAX 每站单独训练（天然去中心化）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 MASE 和 MAE 为主）

#### ✅ 中心化学习结果（Centralized Learning）
| 模型 | 平均 MASE ↓ | 平均 MAE (kW) ↓ | R² ↑ |
|------|-------------|------------------|-------|
| **XGBoost** | **0.17–0.69** | **0.31–7.49** | **0.08–0.29** |
| GRU | 0.19–0.85 | 0.30–7.58 | 0.02–0.43 |
| BiGRU | 0.21–0.86 | 0.37–7.37 | 0.01–0.45 |
| LSTM | 0.21–1.03 | 0.38–7.35 | 0.06–0.41 |
| SARIMAX | 0.42–0.92 | 0.76–11.23 | 0.03–0.36 |
| ARIMA | 0.68–1.14 | 1.18–16.83 | -0.25–(-0.11) |

> 📌 **结论**：**XGBoost 在所有数据集上均取得最优预测性能**，显著优于统计模型和 RNN 变体。

#### ✅ 联邦学习结果（Federated Learning）
| 模型 | 最佳 MASE ↓ | 特点 |
|------|------------|------|
| **FedBiGRU**（Dundee/Boulder） | 0.17; 0.58 | 在异构数据中表现最佳 |
| **FedBiLSTM**（Palo Alto） | 0.49 | 捕捉长期依赖有效 |
| **FedProxXGB**（FEUP） | 0.73 | 接近中心化 XGBoost |
| **FedLSTM**（Palo Alto） | 0.54 | 表现稳定 |

> 📌 **关键发现**：
> - 在联邦范式下，**最佳模型不再是 XGBoost，而是双向 RNN（BiGRU/BiLSTM）**，尤其在数据异构性强的场景（如 Palo Alto）。
> - 中心化 XGBoost 仍是最强整体模型，但在某些数据集（如 FEUP）中，联邦模型可接近其性能。

#### ⚖️ 准确性 vs. 隐私 vs. 能耗 权衡（Table V）

| 模型 | Centralized Energy (kJ/epoch) | Federated ("heavy") (kJ/client/round) | log₁₀(E_fed/E_cent) ↑ | CO₂e (g) "heavy" |
|------|-------------------------------|-------------------------------------|------------------------|------------------|
| **XGBoost** | **0.01** | 0.55 | **2.65** | 0.35 |
| LSTM | 0.48 | 1.94 | 1.51 | 1.21 |
| GRU | 0.49 | 1.29 | 1.32 | 0.79 |
| BiLSTM | 0.72 | 2.87 | 1.51 | 1.79 |
| BiGRU | 0.65 | 3.21 | 1.60 | 2.01 |

> 📌 **能耗结论**：
> - **中心化 XGBoost 能耗最低**（仅 0.01 kJ/epoch），比 RNNs 低约 98%。
> - **联邦版本普遍能耗更高**，尤其是 XGBoost 因需额外训练 1D-CNN 学习率模块，能效最差。
> - 采用 “**light**” 配置（1 个本地 epoch）后，能耗下降高达 **68.21%**，且 CO₂e 排放减少 **80.6%**。

#### 🔍 “Light” vs “Heavy” 配置对比（Dundee 数据集）
| 指标 | Heavy | Light | 变化趋势 |
|------|-------|-------|---------|
| FedBiGRU R² (median) | 0.14 | 0.14 | 基本持平 |
| MAE | 0.39–1.70 | 0.46–2.67 | 上升最多 57% |
| RMSE | 1.06 | 1.36 | +28% |
| 能耗 | 高 | 极低 | ↓68% |

> 📌 **Trade-off 明确**：降低本地 epoch 数可大幅节能减碳，但会牺牲一定预测精度。

---

## 4. 关键结论和发现

### 主要发现
1. **XGBoost 是当前 EDF 任务中最优模型**  
   在中心化设定下，**XGBoost 在预测精度（MASE、MAE、R²）和能源效率方面全面领先**，优于 ARIMA 和所有 RNN 架构。

2. **Federated Learning 改变了最优模型选择**  
   在联邦学习范式中，由于数据分布差异（non-IID），**BiGRU 和 BiLSTM 等双向 RNN 模型能够更好地捕捉局部序列动态，在 Dundee、Boulder 和 Palo Alto 上超越联邦 XGBoost**。

3. **隐私与效率之间存在明确权衡**  
   - 若无隐私顾虑，应优先使用 **centralized XGBoost**；
   - 若强调隐私保护，则应根据数据特性选择：
     - 数据相似 → 使用 **FedXGBoost**
     - 数据异构 → 使用 **FedBiGRU/FedBiLSTM**

4. **Federated Learning 不一定“绿色”**  
   - 当前 FL 实现（特别是 FedXGBllr）可能比集中式训练产生更多碳排放（最高达 2 个数量级）；
   - 通过优化配置（如减少本地 epoch 数）可显著降低能耗和 CO₂e，使 FL 成为可持续选择。

5. **ARIMA 类模型不适合现代 EDF 任务**  
   所有 ARIMA 变体（包括 SARIMAX）表现最差，说明简单线性假设无法捕捉 EV 充电的非线性和突发性。

---

### 方法的局限性
- **未测量通信能耗**：实验未计入模型更新传输带来的网络能耗，实际部署中可能不可忽略。
- **静态拓扑假设**：EVSE Hubs 使用 k-Means 固定划分，未考虑动态负载迁移或拓扑演化。
- **缺乏实时推理延迟测试**：仅关注训练阶段能耗，未评估在线服务时延。
- **数据集规模有限**：尽管多样，但仍不足以代表全球所有城市 EV 使用模式。

---

### 未来工作方向
1. **探索更高效的 FL 聚合策略**  
   如结合 **FedNova** 或 **SCAFFOLD** 以缓解客户端漂移，提升收敛速度与泛化能力。

2. **引入时空图结构建模**  
   利用 EVSE 的地理邻近性和交通流关系构建 **Graph Neural Networks (GNNs)**，增强联邦学习中的空间感知能力。

3. **动态自适应 FL 配置**  
   设计可根据数据质量、客户端活跃度自动调整本地 epoch 数和通信频率的机制，实现精度与能耗的动态平衡。

4. **扩展至 V2G 和多目标优化**  
   将预测模型集成进 **Vehicle-to-Grid (V2G)** 场景，支持双向能量调度与电网辅助服务。

5. **跨城市迁移学习与领域自适应**  
   研究如何将在一个城市训练的 FedEDF 模型迁移到新城市，加速冷启动过程。

--- 

> ✅ **一句话总结**：  
> 本文揭示了在 EVSE 能源需求预测中，**XGBoost 是精度与效率的最佳平衡点**，而 **Federated Learning 提供了一条兼顾隐私与合理性能的路径**，但必须谨慎设计以避免高昂的能源代价。

</details>

---

### 14. [PromptCD: Test-Time Behavior Enhancement via Polarity-Prompt Contrastive Decoding](https://arxiv.org/abs/2602.20696)

**Authors**: Baolong Bi, Yuyao Ge, Shenghua Liu, Yuchen He, Siqian Tong, Lizhe Chen, Lingrui Mei, Zehao Li, Yiwei Wang, Yujun Cai, Ming-Hsuan Yang, Xueqi Cheng  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.20696v1  

#### Abstract
Reliable AI systems require large language models (LLMs) to exhibit behaviors aligned with human preferences and values. However, most existing alignment approaches operate at training time and rely on additional high-quality data, incurring significant computational and annotation costs. While rece...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PromptCD: Test-Time Behavior Enhancement via Polarity-Prompt Contrastive Decoding**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前主流的 LLM 对齐方法（如 RLHF、DPO）依赖于昂贵的训练过程和高质量标注数据，在**测试时缺乏灵活性**。这些静态策略难以动态适应多样化的用户意图和上下文需求，且在面对参数化知识（parametric knowledge）与上下文知识（contextual knowledge）冲突时，模型容易表现出“顽固知识”（stubborn knowledge）——即无法采纳新的上下文信息。

此外，现有 test-time steering 方法存在以下局限：
- **Prompting**：效果不稳定，依赖语言表达；
- **Representation editing**：需要行为特定的调优；
- **Decoding-level control**：通常局限于固定目标（如事实性或安全性）。

### **提出了什么新方法或新思路**
本文提出 **Polarity-Prompt Contrastive Decoding (PromptCD)**，一种通用的、无需训练的 test-time 行为增强框架，其核心思想是：

- 构造一对**极性提示**（polarity prompts）：
  - **Positive Prompt**：鼓励期望行为（如忠实于上下文）；
  - **Negative Prompt**：抑制该行为或激发竞争倾向（如依赖内部先验）。
- 在解码过程中，通过**对比两个提示下模型输出的概率分布**（LLM 中为 token-level logits，VLM 中为视觉注意力图），放大目标行为信号，抑制通用先验。

该方法实现了从“指令级对比”（instruction-level contrast）的角度进行生成控制，统一了文本与多模态场景下的行为引导机制。

### **相比现有方法的优势**
| 维度 | PromptCD 的优势 |
|------|------------------|
| **无需训练** | 完全在推理阶段操作，不修改模型参数，成本低 |
| **灵活可切换** | 只需更换 polarity prompts 即可适配不同对齐目标（helpfulness, honesty, harmlessness 等） |
| **跨模态适用** | 同时支持 LLM 和 VLM（Vision-Language Models） |
| **通用性强** | 不依赖辅助模型或特定层选择启发式规则（unlike DoLa） |
| **可控性强** | 能有效缓解“顽固知识”，提升 context-faithfulness |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

#### **LLM 实验（三大“3H”维度）**
| 对齐维度 | 数据集 | 任务描述 |
|---------|--------|----------|
| **Helpfulness** | NQ, ConFiQA, CoConflictQA | 测试模型是否能根据反事实上下文生成答案（对抗参数记忆） |
| **Honesty** | TruthfulQA, FActScore | 评估事实准确性（多项选择 + 开放生成） |
| **Harmlessness** | SafeEdit | 测试对有害输入的防御能力（9类 unsafe 内容） |

#### **VLM 实验（视觉问答）**
| 数据集 | 特点 |
|-------|------|
| A-OKVQA | 需要外部世界知识的 VQA |
| POPE | 评估对象幻觉（object hallucination） |
| V* | 强调视觉搜索与定位能力 |
| TextVQA | 要求识别图像中的文字 |

### **实验设置和评估指标**

#### **LLM 评估指标**
- **Helpfulness**：
  - `ConR`（Context Recall）↑：响应符合上下文的比例
  - `ParR`（Parameter Recall）↓：响应依赖内部知识的比例
  - `MR = ParR / (ParR + ConR)` ↓：记忆化比率
- **Honesty**：
  - `TruthfulQA-MC1/MC2/MC3` ↑：多项选择准确率
  - `FActScore` ↑：基于 GPT-4.1-nano 打分的事实原子性评分
- **Harmlessness**：
  - `Defense Success (DS)` ↑：对抗样本防御成功率
  - `Defense Generalization (DG)` ↑：泛化到其他攻击形式的能力
  - `Fluency`：n-gram 流畅度评分

#### **VLM 评估指标**
- 所有任务均报告 **Accuracy (%)**
- 注意力可视化分析：对比正负提示下的 cross-modal attention maps

### **基线方法对比**
| 类型 | 基线方法 |
|------|----------|
| **Prompting** | Attr (Attributed), Opin (Opinion-based) |
| **Layer Contrast** | DoLa, SLED |
| **Knowledge Editing** | IKE, MeLLo |
| **Safety Editing** | FT-L, DINM |
| **外部工具对比（VLM）** | SAM, YOLO, CLIP, ViCrop |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据与对比结果**

#### ✅ **LLM 结果汇总（Tables 1–3）**

| 模型 | 方法 | ConR ↑ | ParR ↓ | MR ↓ |
|------|------|--------|--------|------|
| LLAMA2-7B-CHAT | Vanilla | 43.35 | 43.75 | 50.24 |
| | PromptCD (**ours**) | **75.99** | **6.39** | **7.76** |

> ➤ **平均 ConR 提升达 62.64%**，显著优于所有 prompting 基线。

| 模型 | 方法 | TruthfulQA-MC1 ↑ | FActScore ↑ |
|------|------|------------------|-------------|
| LLAMA3-8B-INSTRUCT | Vanilla | 31.95 | 34.20 |
| | PromptCD (**ours**) | **49.21** | **44.50** |

> ➤ 在 honesty 上全面领先，尤其在开放生成中超越 DoLa 和 RECITE。

| 模型 | 方法 | DS ↑ | DG-AVG ↑ | Fluency |
|------|------|------|-----------|---------|
| LLAMA2-7B-CHAT | DINM | 90.00 | 57.63 | 6.61 |
| | PromptCD (**ours**) | **96.67** | **82.37** | **7.29** |

> ➤ 在保持高流畅度的同时实现最强安全防御，**平衡性远超 DINM**。

#### ✅ **VLM 结果（Table 4）**
| 模型 | 方法 | A-OKVQA | V* | TextVQA |
|------|------|--------|-----|--------|
| LLAVA-1.5-13B | Vanilla | 75.7 | 42.4 | 57.1 |
| | PromptCD (**ours**) | **76.9** | **70.0** | **61.2** |

> ➤ 在 V* 上取得 **65.09% 的相对提升**，说明 PromptCD 显著增强了视觉定位能力。

#### ✅ **与其他工具对比（Table 6）**
| 方法 | Accuracy | GPU Time (s/sample) |
|------|----------|--------------------|
| ViCrop (grad-att) | 56.06 | 0.89 |
| **PromptCD (ours)** | **58.20** | **1.34** |

> ➤ **精度最高且推理时间合理**，优于依赖 SAM/YOLO 等外部检测器的方法。

### **消融实验结果**

#### 🔍 **调整系数 γ 的影响（Table 5）**
| γ | Helpfulness | Honesty | Harmlessness |
|----|------------|---------|--------------|
| 0.2 | 77.32 | 39.59 | 83.29 |
| **0.5** | **78.89** | **46.84** | **89.98** |
| 0.8 | 70.53 | 44.87 | 85.34 |

> ➤ 性能呈倒 U 形，**γ=0.5 最优**：过小则对比不足，过大破坏连贯性。

#### 🔍 **APC 模块有效性（Table 8）**
| 模型 | w/o APC | w/ APC | Vanilla |
|------|--------|--------|--------|
| LLAMA3-8B | 53.8 | **85.5** | 83.7 |

> ➤ 移除 APC 导致 hit rate 断崖式下降 → **APC 是防止生成乱码的关键设计**。

#### 🔍 **注意力融合层数选择**
- 最佳层区间为 `[20,25]`（深层）
- 深层注意力更聚焦语义相关区域，适合 contrastive refinement

---

## **4. 关键结论和发现**

### **主要发现**
1. **Latent Alignment ≠ Behavioral Emergence**  
   正向提示虽能提升上下文知识的 logits，但常不足以使其成为 top-1 输出 —— 存在“决策边界跨越失败”。

2. **PromptCD 成功将潜在偏好转化为显式行为**  
   通过 contrastive decoding 动态放大差异，使原本“接近但未胜出”的正确 token 成为主导输出。

3. **适用于多种对齐目标与模态**  
   在 LLM 的“3H”维度和 VLM 的视觉接地任务上均取得一致提升，验证了方法的**通用性与可扩展性**。

4. **有效缓解“顽固知识”问题**  
   在知识编辑（IKE/MeLLo）任务中，PromptCD 显著提升了大模型对新信息的接纳能力（Figure 8）。

5. **计算开销可控**  
   推理延迟增加约 1.6–1.8×（Table 7），但在高风险领域（医疗、金融等）是值得接受的代价。

### **方法的局限性**
- **双路径同步机制带来额外计算负担**：每个 token 需两次前向传播。
- **依赖人工设计 polarity prompts**：虽然灵活，但仍需领域知识来构造有效的正负指令。
- **极端情况下可能过度抑制合理输出**：若 negative prompt 过强，可能导致语义失真（需 APC 缓解）。
- 当前未探索多轮对话中的长期一致性控制。

### **未来工作方向**
- 自动化 polarity prompt 的生成与优化（e.g., prompt tuning at test time）
- 将 PromptCD 与 RAG 系统深度集成，用于动态知识更新
- 扩展至多轮对话、Agent 规划等复杂场景的行为调控
- 探索更高效的 contrastive 实现方式（如缓存共享前缀）

---

> 💡 **一句话总结**：  
> **PromptCD 提供了一种简单、通用、无需训练的 test-time 对齐机制，通过极性提示对比解码，在 LLM 和 VLM 上实现了对 helpfulness、honesty 和 harmlessness 的一致增强，为构建可靠、可控的 AI 系统提供了新范式。**

</details>

---

### 15. [ICON: Indirect Prompt Injection Defense for Agents based on Inference-Time Correction](https://arxiv.org/abs/2602.20708)

**Authors**: Che Wang, Fuyao Zhang, Jiaming Zhang, Ziqi Zhang, Yinghui Wang, Longtao Huang, Jianbo Gao, Zhong Chen, Wei Yang Bryan Lim  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.20708v1  

#### Abstract
Large Language Model (LLM) agents are susceptible to Indirect Prompt Injection (IPI) attacks, where malicious instructions in retrieved content hijack the agent's execution. Existing defenses typically rely on strict filtering or refusal mechanisms, which suffer from a critical limitation: over-refu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ICON: Indirect Prompt Injection Defense for Agents based on Inference-Time Correction

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **Indirect Prompt Injection (IPI)** 攻击对大型语言模型（LLM）代理的安全威胁。IPI攻击通过在检索内容（如网页、邮件）中嵌入恶意指令，诱导代理执行未经授权的操作（如调用工具），而传统防御机制往往因**过度拒绝（over-refusal）** 而中断合法任务流程。

现有方法（如输入模板化、后处理过滤、安全微调）存在以下缺陷：
- **启发式方法** 易被自适应攻击绕过；
- **安全微调** 导致模型过于敏感，损害任务连续性；
- 多数方法牺牲 **task utility** 来换取安全性。

### 提出了什么新方法或新思路
提出 **ICON**（Inference-Time Correction），一种基于推理时修正的两阶段防御框架：

1. **Latent Space Trace Prober (LSTP)**  
   利用 **Focus Intensity Score (FIS)** 检测潜在空间中的注意力异常模式。研究发现，成功的IPI攻击会导致模型在少数注入token上出现“注意力坍缩”（attention collapse），即异常集中的注意力分布。

2. **Mitigating Rectifier (MR)**  
   在检测到攻击后，不终止任务，而是进行**手术式注意力引导**（surgical attention steering）：
   - 抑制与恶意查询相关的 query-key 依赖；
   - 通过 Contrastive Steering 操作重新分配注意力至任务相关上下文；
   - 恢复代理原本的功能轨迹。

### 相比现有方法的优势
- ✅ **保持任务连续性**：避免了“全有或全无”的拒绝策略，显著提升 utility；
- ✅ **高效轻量**：训练仅需 <2分钟 和 ~255样本，参数量小（约31k）；
- ✅ **强泛化能力**：在未见分布（OOD）和多模态场景下表现优异；
- ✅ **无需重训练**：为冻结LLM提供即插即用（plug-and-play）防御。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
| 数据集 | 类型 | 描述 |
|-------|------|------|
| **TrojanTools** (Anonymous, 2025) | 训练集 | 包含多样化恶意工具调用的自适应IPI攻击样本，用于离线生成训练数据 |
| **InjectAgent** (Zhan et al., 2024) | 测试集（OOD） | 广泛使用的IPI基准，评估跨分布泛化能力 |
| **AgentDojo** (Debenedetti et al., 2024) | 测试集（OOD） | 动态环境下的IPI攻防评测平台 |
| **Visual Prompt Injection Benchmarks** (Wan et al., 2024) | 多模态测试集 | 用于评估视觉-语言模型中的IPI防御效果 |

### 实验设置和评估指标
- **基础模型**：
  - 文本模型：`QWEN-3-8B`, `LLAMA-3.1-8B`, `MISTRAL-8B`
  - 多模态模型（MLLM）：`QWEN-VL`, `INTERNVL`, `MINICPM`

- **评估指标**：
  - **ASR (Attack Success Rate)** ↓：攻击成功率，越低越好
  - **UA (Utility under Attack)** ↑：攻击下的任务完成度，越高越好
  - **ADR (Attack Detection Rate)** ↑：攻击检出率
  - **URR (Utility Recovery Rate)** ↑：效用恢复率

- **集成框架**：ReAct (Yao et al., 2022)

### 基线方法对比
| 方法类别 | 具体方法 | 特点 |
|--------|---------|------|
| **Template-based** | Repeat Prompt, Delimiting | 重复系统提示或使用分隔符隔离不可信内容 |
| **Filter-based** | MELON (Zhu et al., 2025) | 工具调用后过滤机制 |
| **Fine-tuning-based** | Qwen3Guard, Gemini (LLM-Detector) | 商业级安全微调模型，作为上限基线 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 5）

#### 文本模型平均表现（Table 2）
| 方法 | ASR ↓ | UA ↑ |
|------|--------|-------|
| No Defense | 32.7% | 43.1% |
| Repeat Prompt | 16.8% | 51.0% |
| MELON | 21.8% | 48.9% |
| LLM-Detector (Qwen3Guard) | **0.2%** | 43.2% |
| **ICON (Ours)** | **0.4%** | **53.8%** |

> 🔍 **结论**：ICON在几乎同等安全水平下（ASR ≈ 0.4%），实现了超过 **50% 的效用增益**（相比LLM-Detector）。

#### 多模态模型表现（Table 5）
| 方法 | ASR ↓ | UA ↑ |
|------|--------|-------|
| No Defense | 59.0% | 28.1% |
| Repeat Prompt | 49.7% | 31.7% |
| LLM-Detector (Gemini) | **0.0%** | 28.8% |
| **ICON (Ours)** | 2.9% | **47.2%** |

> 🎯 ICON在多模态场景下仍实现极低ASR，并将UA提升近 **20个百分点**，远超Gemini。

#### OOD 泛化能力（Table 3）
在 **TrojanTools 上训练，在 AgentDojo 上测试**

| 模型 | ADR ↑ | URR ↑ |
|------|--------|--------|
| QWEN-3-8B | 98.0% ±1.2 | 69.6% ±2.7 |
| LLAMA-3.1-8B | 97.3% ±1.8 | 55.9% ±3.3 |

> ✅ 表明 ICON 学习的是 IPI 的**本质潜层特征**，而非表面语义模式，具备强大零样本迁移能力。

### 与基线方法的对比结果
- **vs 安全微调模型（Qwen3Guard/Gemini）**：
  - 安全性相当（ASR ~0.4% vs 0.0%），但 **UA 高出 >10%**
  - 微调模型倾向于“拒绝一切”，破坏任务流；ICON则修复并继续执行
- **vs MELON**：
  - ASR更低（3.2% → 0.4%），UA更高（+10%）
  - MELON是事后过滤，无法回溯推理过程
- **vs 模板方法**：
  - 对抗自适应攻击（如TrojanTools）无效，而ICON依然有效

### 消融实验结果（文中隐含分析）
虽然未设独立消融表，但从设计逻辑可推断：
- 若仅使用 LSTP 不启用 MR，则会退化为拒绝机制，导致 utility 下降；
- 若仅依赖FIS而不进行 attention steering，则无法恢复功能路径；
- 双因子干预机制（scope T + intensity γ）确保精准控制干预强度与范围。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **IPI攻击会在潜层留下可识别的“注意力坍缩”信号**，可通过FIS量化检测；
2. ✅ **无需拒绝即可防御攻击**：通过 inference-time 的 latent space manipulation，可在不中断任务的前提下纠正被劫持的行为；
3. ✅ **轻量训练即可获得强泛化能力**：仅需少量合成攻击样本训练探测器，即可迁移到不同模型和任务；
4. ✅ **适用于多模态代理**：在视觉-语言模型中同样有效，验证了架构无关性（architecture-agnostic）。

### 方法的局限性
- ❗ 依赖白盒访问权限：需要读取中间层 attention weights，限制其在黑盒API场景的应用；
- ❗ 对极端隐蔽攻击可能失效：若攻击完全模拟正常注意力分布，则FIS可能无法识别；
- ❗ 当前主要针对 tool-calling 场景，是否适用于其他代理行为（如记忆篡改）尚待验证。

### 未来工作方向
- 探索 **black-box setting 下的替代探针机制**（如基于输出扰动推断潜层状态）；
- 将 ICON 扩展至 **memory-augmented agents** 和 **multi-agent systems**；
- 结合 **causal tracing** 技术进一步定位关键干预头，减少计算开销；
- 构建 **动态自适应探针**，在线更新以应对新型攻击变种。

---

> 💡 **总体评价**：ICON 提供了一种全新的“检测+修复”范式，打破了“安全 vs 效用”的零和博弈，为构建**既安全又可靠**的自主代理系统提供了实用且高效的解决方案。

</details>

---

### 16. [HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG](https://arxiv.org/abs/2602.20926)

**Authors**: Yuqi Huang, Ning Liao, Kai Yang, Anning Hu, Shengchao Hu, Xiaoxing Wang, Junchi Yan  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.20926v1  

#### Abstract
Large Language Models (LLMs) often struggle with inherent knowledge boundaries and hallucinations, limiting their reliability in knowledge-intensive tasks. While Retrieval-Augmented Generation (RAG) mitigates these issues, it frequently overlooks structural interdependencies essential for multi-hop ...

---

### 17. [NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning](https://arxiv.org/abs/2602.21172)

**Authors**: Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.21172v1  

#### Abstract
Vision-Language-Action (VLA) models are advancing autonomous driving by replacing modular pipelines with unified end-to-end architectures. However, current VLAs face two expensive requirements: (1) massive dataset collection, and (2) dense reasoning annotations. In this work, we address both challen...

---

### 18. [Stability and Generalization of Push-Sum Based Decentralized Optimization over Directed Graphs](https://arxiv.org/abs/2602.20567)

**Authors**: Yifei Liang, Yan Sun, Xiaochun Cao, Li Shen  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.20567v1  

#### Abstract
Push-Sum-based decentralized learning enables optimization over directed communication networks, where information exchange may be asymmetric. While convergence properties of such methods are well understood, their finite-iteration stability and generalization behavior remain unclear due to structur...

---

### 19. [Rethink Efficiency Side of Neural Combinatorial Solver: An Offline and Self-Play Paradigm](https://arxiv.org/abs/2602.20730)

**Authors**: Zhenxing Xu, Zeyuan Ma, Weidong Bao, Hui Yan, Yan Zheng, Ji Wang  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.20730v1  

#### Abstract
We propose ECO, a versatile learning paradigm that enables efficient offline self-play for Neural Combinatorial Optimization (NCO). ECO addresses key limitations in the field through: 1) Paradigm Shift: Moving beyond inefficient online paradigms, we introduce a two-phase offline paradigm consisting ...

---

### 20. [PIME: Prototype-based Interpretable MCTS-Enhanced Brain Network Analysis for Disorder Diagnosis](https://arxiv.org/abs/2602.21046)

**Authors**: Kunyu Zhang, Yanwu Yang, Jing Zhang, Xiangjie Shi, Shujian Yu  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.21046v1  

#### Abstract
Recent deep learning methods for fMRI-based diagnosis have achieved promising accuracy by modeling functional connectivity networks. However, standard approaches often struggle with noisy interactions, and conventional post-hoc attribution methods may lack reliability, potentially highlighting datas...

---

### 21. [Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning](https://arxiv.org/abs/2602.20722)

**Authors**: Xu Wan, Yansheng Wang, Wenqi Huang, Mingyang Sun  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.20722v1  

#### Abstract
Traditional on-policy Reinforcement Learning with Verifiable Rewards (RLVR) frameworks suffer from experience waste and reward homogeneity, which directly hinders learning efficiency on difficult samples during large language models post-training. In this paper, we introduce Batch Adaptation Policy ...

---

### 22. [Motivation is Something You Need](https://arxiv.org/abs/2602.21064)

**Authors**: Mehdi Acheli, Walid Gaaloul  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.21064v1  

#### Abstract
This work introduces a novel training paradigm that draws from affective neuroscience. Inspired by the interplay of emotions and cognition in the human brain and more specifically the SEEKING motivational state, we design a dual-model framework where a smaller base model is trained continuously, whi...

---

### 23. [Tensor Network Generator-Enhanced Optimization for Traveling Salesman Problem](https://arxiv.org/abs/2602.20175)

**Authors**: Ryo Sakai, Chen-Yu Liu  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.20175v1  

#### Abstract
We present an application of the tensor network generator-enhanced optimization (TN-GEO) framework to address the traveling salesman problem (TSP), a fundamental combinatorial optimization challenge. Our approach employs a tensor network Born machine based on automatically differentiable matrix prod...

---

### 24. [Fuz-RL: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty](https://arxiv.org/abs/2602.20729)

**Authors**: Xu Wan, Chao Yang, Cheng Yang, Jie Song, Mingyang Sun  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.20729v1  

#### Abstract
Safe Reinforcement Learning (RL) is crucial for achieving high performance while ensuring safety in real-world applications. However, the complex interplay of multiple uncertainty sources in real environments poses significant challenges for interpretable risk assessment and robust decision-making. ...

---

### 25. [Counterfactual Simulation Training for Chain-of-Thought Faithfulness](https://arxiv.org/abs/2602.20710)

**Authors**: Peter Hase, Christopher Potts  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.20710v1  

#### Abstract
Inspecting Chain-of-Thought reasoning is among the most common means of understanding why an LLM produced its output. But well-known problems with CoT faithfulness severely limit what insights can be gained from this practice. In this paper, we introduce a training method called Counterfactual Simul...

---

### 26. [ID-LoRA: Efficient Low-Rank Adaptation Inspired by Matrix Interpolative Decomposition](https://arxiv.org/abs/2602.20727)

**Authors**: Xindian Ma, Rundong Kong, Peng Zhang, Ruoxiang Huang, Yongyu Jiang  
**Category**: cs.CL  
**Published**: 2026-02-25  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.20727v1  

#### Abstract
LoRA has become a universal Parameter-Efficient Fine-Tuning (PEFT) technique that equips Large Language Models (LLMs) to adapt quickly to new tasks. However, when these models are scaled up, even the latest LoRA variants still introduce considerable overhead in trainable parameters. Conversely, aggr...

---

### 27. [Heterogeneity-Aware Client Selection Methodology For Efficient Federated Learning](https://arxiv.org/abs/2602.20450)

**Authors**: Nihal Balivada, Shrey Gupta, Shashank Shreedhar Bhatt, Suyash Gupta  
**Category**: cs.DC  
**Published**: 2026-02-25  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.20450v1  

#### Abstract
Federated Learning (FL) enables a distributed client-server architecture where multiple clients collaboratively train a global Machine Learning (ML) model without sharing sensitive local data. However, FL often results in lower accuracy than traditional ML algorithms due to statistical heterogeneity...

---

### 28. [Is the Trigger Essential? A Feature-Based Triggerless Backdoor Attack in Vertical Federated Learning](https://arxiv.org/abs/2602.20593)

**Authors**: Yige Liu, Yiwei Lou, Che Wang, Yongzhi Cao, Hanpin Wang  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.20593v1  

#### Abstract
As a distributed collaborative machine learning paradigm, vertical federated learning (VFL) allows multiple passive parties with distinct features and one active party with labels to collaboratively train a model. Although it is known for the privacy-preserving capabilities, VFL still faces signific...

---

### 29. [Bikelution: Federated Gradient-Boosting for Scalable Shared Micro-Mobility Demand Forecasting](https://arxiv.org/abs/2602.20671)

**Authors**: Antonios Tziorvas, Andreas Tritsarolis, Yannis Theodoridis  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.20671v1  

#### Abstract
The rapid growth of dockless bike-sharing systems has generated massive spatio-temporal datasets useful for fleet allocation, congestion reduction, and sustainable mobility. Bike demand, however, depends on several external factors, making traditional time-series models insufficient. Centralized Mac...

---

### 30. [Modality-Guided Mixture of Graph Experts with Entropy-Triggered Routing for Multimodal Recommendation](https://arxiv.org/abs/2602.20723)

**Authors**: Ji Dai, Quan Fang, Dengsheng Cai  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.20723v1  

#### Abstract
Multimodal recommendation enhances ranking by integrating user-item interactions with item content, which is particularly effective under sparse feedback and long-tail distributions. However, multimodal signals are inherently heterogeneous and can conflict in specific contexts, making effective fusi...

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

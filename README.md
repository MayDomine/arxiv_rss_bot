# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-16 10:09:29 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Efficient KV Cache Migration for Geo-Distributed LLM Inference in Collaborative Edge Computing](https://arxiv.org/abs/2610.22172)

**Authors**: Mingjin Zhang, Jiannong Cao, Tao Wu, Xiangchun Chen, Ne Wang  
**Category**: cs.DC  
**Published**: 2026-10-16  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2610.22172v1  

#### Abstract
Large Language Models (LLMs) have shown exceptional capabilities but rely heavily on centralized cloud GPU clusters, leading to high bandwidth costs, latency, and privacy issues. Edge computing offers a solution by deploying LLMs closer to users on edge servers, yet single edge servers often cannot ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient KV Cache Migration for Geo-Distributed LLM Inference in Collaborative Edge Computing

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**协同边缘计算**（Collaborative Edge Computing, CEC）环境下，大型语言模型（LLM）推理面临以下挑战：
- 单个边缘服务器资源有限，无法独立承载完整的 LLM 推理任务；
- 因此采用分布式方式将模型分片（shard）部署于多个地理分布的异构边缘节点上；
- 在此架构中，每个节点维护其对应部分的 **KV Cache**（Key-Value Cache），用于加速自回归生成过程；
- 当某些边缘节点过载或计划退出网络时，必须将其状态（尤其是 KV Cache）迁移到其他节点以保证服务连续性。

然而，传统云环境中的 KV Cache 迁移方法（如直接传输或基于 token 的重计算）在边缘场景下存在严重缺陷：
- **Direct KV Cache Transmission**：通信开销巨大，在带宽受限的边缘网络中效率低下；
- **Token-based Recomputation**：需要重新执行 Prefill 阶段，计算成本高，对消费级 GPU 不友好。

因此，本文提出一个全新的问题：如何在**资源受限、带宽受限、设备异构**的 CEC 环境中高效迁移 KV Cache？

---

### 🚀 提出的新方法与创新思路

作者提出了名为 **EdgeFlow** 的混合型 KV Cache 迁移框架，其核心思想是：

#### （1）引入 **Activation-based KV Cache Recomputation**
- 发现 KV Cache 可通过中间层的 **activations** 和已知的投影权重 $W_k$, $W_v$ 快速重建；
- 由于 **activation 数据量仅为 KV Cache 的 50%**，显著降低传输体积；
- 缺点是引入额外的计算负载。

#### （2）设计 **Hybrid Migration Strategy**
- 结合两种策略优势：一部分层使用 activation 传输 + 本地重算，另一部分层直接传输 KV Cache；
- 自适应决定每层采用哪种方式，实现通信与计算之间的最优权衡。

#### （3）构建 **多节点联合优化问题**（Multi-Node Hybrid KV Cache Migration, MN-KVM）
- 考虑多个源节点同时迁移的情况；
- 联合优化目标为最小化最慢节点的迁移时间（makespan）；
- 决策变量包括：
  - 每个源节点的目标迁移节点（Target Node Selection）；
  - 每个模型分片的“分割点” $p$ —— 前 $p$ 层用 activation 重算，后 $L-p$ 层直接传输。

#### （4）理论算法保障
- 将原问题建模为 NP-Hard 的 min-max 优化问题；
- 通过 **改进的线性规划松弛**（Improved LP Relaxation）和 **二分图匹配技术**（Bipartite Graph Matching）设计了一个具有 **2-近似比** 的多项式时间算法；
- 时间复杂度为 $O(\log K + \log T_{\text{max}})$，适用于大规模系统调度。

---

### 🔍 相比现有方法的优势

| 方法 | 优势 |
|------|------|
| **相比 Direct KV Transmission** | 显著减少通信量（尤其在低带宽场景），节省高达 45% 的迁移时间 |
| **相比 Pure Activation Recomputation** | 利用并行流水线隐藏计算延迟，避免弱节点成为瓶颈 |
| **相比云中心方案（如 Llumnix, vLLM）** | 专为边缘异构性和低带宽设计，不依赖 NVLink/InfiniBand 高速互联 |
| **相比 Greedy 或 Brute-force** | 在可接受时间内接近最优解（优于 Greedy，远快于 Brute-force） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与模型
- **数据集**：WikiText-2（来自 HuggingFace），用于生成输入 token 序列；
- **测试模型**：
  - **Qwen2.5-32B**：64 层 Transformer，激活维度 5120；
  - **LLaMA3.1-70B**：80 层 Transformer，激活维度 8192；
- 所有实验使用 **FP16 精度**进行推理。

### ⚙️ 实验平台与基础设施
- 构建真实世界分布式平台 **PolyLink**，连接来自多所高校的 **100+ 异构消费级 GPU**；
- 包含多种设备类型（见下表）：

| 设备型号 | 显存容量 | AI 性能（TFLOPS） |
|--------|---------|------------------|
| RTX2080Ti | 11GB | 26.9 |
| RTX3060 | 12GB | 26.0 |
| RTX3090 | 24GB | 35.58 |
| RTX4090 | 24GB | 82.58 |
| A6000 Ada | 48GB | 91.06 |

- 使用 **NetBird** 实现跨公网安全通信；
- 推理后端基于 **llama-box**（源自 llama.cpp）；
- 系统持续监控各节点的 GPU 类型、利用率、可用内存等信息。

### 📊 评估指标
- 主要指标：**KV Cache Migration Time**（迁移耗时）
- 辅助分析指标：
  - 吞吐提升
  - 资源利用率
  - 服务恢复速度

### 🆚 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **DKV**（Direct KV Transmission） | 全部 KV Cache 直接传输，目标节点由算法2选择 |
| **ACT**（Activation-based Recomputation） | 所有层均传输 activation 并在目标端重算 KV Cache |
| **EdgeFlow-G**（Greedy Version） | 使用相同 hybrid 策略，但目标节点采用贪心选择 |
| **BF**（Brute Force） | 遍历所有可能组合，获得理论上界（Upper Bound） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 场景 | EdgeFlow 相比基线的性能提升 |
|------|----------------------------|
| 平均带宽 900Mbps，Qwen2.5-32B | 比 DKV ↓39%，比 ACT ↓20%，比 EdgeFlow-G ↓25.1% |
| 平均带宽 900Mbps，LLaMA3.1-70B | 比 DKV ↓35%，比 ACT ↓29.2%，比 EdgeFlow-G ↓26% |
| Token 长度 512，LLaMA3.1-70B | 比 DKV ↓40%，比 ACT ↓18%，比 EdgeFlow-G ↓26.4% |
| 多种异构配置平均表现 | **最高减少 45% 迁移时间** |

> ✅ **EdgeFlow 在绝大多数情况下性能接近 Brute Force 最优解**，说明调度算法有效性极高。

---

### 🔬 消融实验与关键发现

#### （1）带宽影响（Fig. 6）
- 随着带宽增加，所有方法迁移时间下降；
- 在低带宽（100 Mbps）时，EdgeFlow 表现接近 ACT —— 因此时系统倾向于更多使用 activation 传输；
- 在高带宽下，EdgeFlow 更多地利用直接传输 + 并行流水线，进一步压缩总时间。

#### （2）Token 长度影响（Fig. 7）
- 所有方法迁移时间随 token length 上升而上升（因 KV Cache 规模增大）；
- EdgeFlow 增幅最小，体现出更强的可扩展性；
- 对比 DKV，EdgeFlow 减少了大量冗余数据传输。

#### （3）源节点异构性（Fig. 8）
- 设置不同层数分配模式（A: 5-10-15；B: 10-10-10；C: 5-20-5）；
- EdgeFlow 在方差更大的配置（A/C）中优势更明显（最多 ↓48.2% vs DKV）；
- 原因：层数差异越大，优化空间越多，hybrid 策略越能发挥调节作用。

#### （4）目标节点异构性（Fig. 9）
- 在混合设备环境（Env. A: 多种 GPU）中，EdgeFlow 能智能地将重计算任务分配给高性能节点（如 A6000 Ada）；
- 在纯高端设备环境（Env. B/C）中仍保持领先；
- 特别是在 Env. C 中，EdgeFlow 比 DKV 快 **45.8%**。

#### （5）大规模仿真（Fig. 10 & Table IV）
| 方法 | 目标节点数=50 时运行时间 | 是否可行 |
|-----|--------------------------|---------|
| **Brute Force** | 325 分钟（约5.4小时） | ❌ 实际不可行 |
| **EdgeFlow-G** | 0.022 秒 | ✅ 快但次优 |
| **EdgeFlow** | 3.22 秒 | ✅ 接近最优且高效 |

- 虽然 BF 理论上更优（提升 ~15%），但其搜索空间爆炸，无法用于实际系统；
- EdgeFlow 在 **3秒内完成决策**，实现了性能与效率的最佳平衡。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **Hybrid KV Cache Migration 是边缘环境下更优的选择**：
   - 在通信与计算之间取得良好折衷；
   - 可充分利用异构资源特性；
   - 支持并行流水线，有效隐藏延迟。

2. **EdgeFlow 显著优于现有云中心化方案**：
   - 在典型边缘网络条件下，**迁移时间最多减少 45%**；
   - 适用于动态、不稳定、资源受限的真实边缘环境。

3. **提出的 2-近似算法具备强实用性**：
   - 在合理时间内逼近最优解；
   - 支持多节点并发迁移场景下的全局协调。

4. **系统已在真实平台上验证可行性**：
   - PolyLink 已接入百台以上异构 GPU；
   - 支持开放共享与激励机制，具备落地潜力。

---

### ⚠️ 方法的局限性
1. **依赖模型结构先验知识**：
   - 需要知道每一层的 activation size 和参数矩阵 $W_k/W_v$；
   - 对黑盒模型或加密模型支持较弱。

2. **未考虑动态带宽波动**：
   - 实验假设带宽相对稳定；
   - 实际公网中链路质量可能剧烈变化，需加入预测模块。

3. **仅关注 KV Cache 迁移，未涉及模型权重迁移**：
   - 若整个 shard 需要迁移，还需结合模型切分与加载机制。

4. **当前实现基于 llama.cpp 生态**：
   - 对 PyTorch/TensorRT-LLM 等主流框架集成尚待完善。

---

### 🔮 未来工作方向
1. **支持动态自适应调整策略**：
   - 在迁移过程中根据实时反馈动态切换 transmission/recompute 比例。

2. **扩展至多模态模型迁移**：
   - 如视觉-语言模型中的 KV Cache 与 feature map 联合迁移。

3. **结合联邦学习与隐私保护机制**：
   - 在迁移过程中引入加密传输或差分隐私。

4. **构建自动化边缘 LLM Orchestration 平台**：
   - 整合模型切分、负载均衡、故障恢复、KV Cache 管理于一体。

5. **探索更高效的压缩与编码机制**：
   - 对 activation 或 KV Cache 本身进行量化/稀疏化处理，进一步减小传输量。

---

> 💡 **一句话总结**：  
> **EdgeFlow 是首个面向协同边缘计算环境的高效混合 KV Cache 迁移系统，通过融合传输与重算，并辅以理论保障的调度算法，在真实异构平台上实现了高达 45% 的迁移加速，显著提升了分布式 LLM 推理的鲁棒性与效率。**

</details>

---

### 2. [On-Policy Residual Learning for Diffusion Language Models](https://arxiv.org/abs/2610.20687)

**Authors**: Yifan Yu, Yuqing Jian, Jisen Li, Yuezhou Hu, Tianyi Zhang, Xiaoxia Wu, Rui-Jie Zhu, Junxiong Wang, Zhongzhu Zhou, Fan Lai, James Zou, Chenfeng Xu  
**Category**: cs.LG  
**Published**: 2026-10-16  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2610.20687v1  

#### Abstract
Residual context is key to unlocking the speed-quality Pareto frontier of diffusion language models (DLMs) by carrying forward predictive distributions across denoising steps. Yet turning this seemingly simple mechanism into an effective learning paradigm is far from simple: there is no clear recipe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：On-Policy Residual Learning for Diffusion Language Models

## 1. 主要贡献和创新点

### 解决的问题
本文针对**扩散语言模型**（DLMs）在并行解码过程中存在的“速度-质量权衡”（speed-quality Pareto frontier）难以优化的问题。尽管残差上下文（residual context）已被证明能显著提升DLM的效率，但如何有效且高效地学习这些残差路径仍缺乏明确方案。现有方法存在两大挑战：
- **训练与推理状态不匹配**：离线生成的残差轨迹无法反映当前模型的真实行为，导致训练不稳定。
- **缺乏对逐步精炼过程的显式优化目标**：仅监督最终输出会忽略每一步应减少不确定性的需求。

### 提出的新方法：ORL（On-Policy Residual Learning）
作者提出了一种名为 **ORL** 的端到端框架，用于在DLM中进行**在线策略残差学习**（on-policy residual learning），其核心思想包括：

- **On-Policy 学习机制**：ORL在每个优化步骤中都使用当前模型实时生成的、带有精确残差输入的轨迹进行训练，确保训练与推理分布一致。
- **相邻熵收缩目标**（Adjacent Entropy Contraction, AEC）：引入一个新目标函数，要求每一步的未解决位置的平均归一化熵必须比前一步下降一定比例（由超参数 $ \rho $ 控制）。该损失采用铰链形式，只在熵未充分收缩时施加惩罚，避免过度锐化已有高置信度预测。
- **系统级协同设计**（Co-design）：为支持大规模精确的on-policy训练，ORL设计了以下关键技术：
  - **异步精确状态捕获**（Asynchronous exact-state capture）：在不影响解码关键路径的前提下，快照中间状态。
  - **打包计算平衡训练**（Packed compute-balanced training）：将来自不同响应的状态打包成密集批次，并按计算成本负载均衡。
  - **阶梯注意力核**（Staircase attention kernel）：特化的Triton内核，隐式处理block-diffusion可见性规则，无需存储稠密掩码。

### 相比现有方法的优势
- **更优的速度-质量权衡**：相比基于固定自回归主干的推测解码（speculative decoding），ORL直接优化DLM自身的并行能力，在相同或更高准确率下实现更快的吞吐量。
- **更高的数据效率**：仅需数百万自生成token即可完成后训练，远低于先前方法（如d3LLM-LLaDA需约40M tokens）。
- **通用性强**：可直接应用于多种现有DLM（如LLaDA-2.1-mini, DiffusionGemma），无需特定训练流水线。

---

## 2. 核心实验方法和设置

### 数据集
实验覆盖多个主流评测基准，分为两类：
- **数学推理**：GSM8K（1319题）、MATH-500（500题）、AIME-24
- **代码生成与理解**：HumanEval（164题）、MBPP（427题）、LiveCodeBench-v6
- **综合知识与指令遵循**：MMLU-Pro、GPQA-Diamond、IFEval

### 实验设置
- **基础模型**：
  - **QwenDLM**：从Qwen3-4B转换而来，作为AR转DLM的起点。
  - **LLaDA-2.1-mini** 和 **DiffusionGemma**：作为现成DLM的微调起点。
- **训练方式**：
  - ORL采用on-policy训练，每轮采样8次rollout，仅保留成功验证的轨迹进行更新。
  - 使用AdamW优化器，结合Cross-Entropy（CE）和AEC目标。
- **推理引擎**：
  - 自研单请求引擎 **DLMRT**，通过CUDA图、commit folding等技术降低延迟。
  - 支持 **NVFP4量化** 以进一步加速推理。

### 评估指标
- **Tokens Per Forward**（TPF）：每次去噪前向传播所提交的有效token数，衡量并行效率。
- **Throughput**（TPS）：每秒生成的token数量，反映实际吞吐性能。
- **Accuracy/Pass@1**：任务正确率。
- **Clean-request TPS**：排除长度截断失败后的有效请求吞吐量。

### 基线方法对比
- **DLM基线**：d3LLM-Dream、Fast-dLLM、RCD-SDAR、TraDo-4B、dParallel-LLaDA、DMax-Math 等。
- **推测解码基线**：EAGLE3、DFlash（搭配Qwen3-4B和Gemma-4-26B）。
- **自回归基线**：标准AR解码。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 方法 | GSM8K Acc. | TPF | HumanEval Pass@1 | TPF |
|------|------|-------------|-----|------------------|-----|
| QwenDLM | Base | 90.4% | 2.4 | 82.9% | 1.8 |
| QwenDLM | +ORL | **92.4%** | **5.9** | **89.6%** | **11.1** |
| LLaDA-2.1-mini | Base | 91.9% | 5.6 | 86.6% | 11.3 |
| LLaDA-2.1-mini | +ORL | **92.0%** | **8.5** | **89.0%** | **13.1** |
| DiffusionGemma | Base | 95.8% | 26.0 | 90.9% | 24.5 |
| DiffusionGemma | +ORL | **96.0%** | **30.0** | **91.9%** | **27.9** |

> ✅ **TPF提升显著**：QwenDLM提升 **2.5–6.2×**，LLaDA提升 **1.5×**，DiffusionGemma提升 **1.1–1.2×**。

### 与基线方法的对比结果
- 在单请求场景下（one H100），QwenDLM+ORL达到 **1,298 tokens/s**，比其原始版本快 **6.9×**（含DLMRT引擎）。
- 相比推测解码：
  - 比 **DFlash** 快 **1.74×**（宽松验证，同精度）
  - 比 **DFlash** 快 **2.26×**（严格验证，精度略低但仍在合理范围）
- 在消费级显卡上表现优异：
  - **RTX 5090** 上，LLaDA-2.1-mini+ORL 达到 **2,788 tokens/s**（HumanEval），媲美H100上的BF16性能。

### 消融实验结果
#### （1）AEC目标的作用
| 配置 | GSM8K Acc. | TPF |
|------|------------|-----|
| Full ORL | 94.5% | 6.25 |
| -AEC | 92.6% | 4.48 |
> ❌ 移除AEC导致TPF下降 **28%**，说明AEC是提升并行效率的关键。

#### （2）残差路径缩放（Residual Scaling）
- 不进行缩放时，残差路径更新速度过快，导致训练崩溃（verifier reward从0.677降至0.225）。
- 缩放后保持稳定长期优化，验证了控制残差路径更新节奏的重要性。

#### （3）数据效率
- LLaDA-2.1-mini仅用 **4.2M自生成token** 和 **36 H100小时** 完成训练，约为d3LLM-LLaDA的 **1/10数据量**，却取得更高TPF和相近精度。

#### （4）系统优化累积效果
| 优化项 | 训练步时间加速 | Token吞吐提升 |
|--------|----------------|--------------|
| 阶梯注意力 + 打包 | 1.35–1.69× | 1.43–2.17× |
| 负载均衡 | +0.25–0.35× | +0.51–1.35× |
| 异步GPU驻留传输 | +0.62–0.67× | +0.51–0.70× |
> 🔧 总计最高达 **5.2×训练加速**，使大规模on-policy训练变得可行。

---

## 4. 关键结论和发现

### 主要发现
1. **残差学习本质上是on-policy问题**：必须使用当前策略生成的轨迹进行训练，否则会导致状态分布偏移和性能退化。
2. **仅监督准确性不足以加速精炼过程**：需要像AEC这样的显式目标来推动每一步减少不确定性。
3. **系统协同设计至关重要**：高效的on-policy训练不仅依赖算法，还需底层系统的支持（如异步捕获、打包调度、专用注意力）。
4. **ORL具有高度泛化性**：可在不同架构的DLM上快速迁移并取得一致收益，且所需数据极少。

### 方法的局限性
- **训练开销仍然较高**：虽然系统优化提升了效率，但on-policy训练本身需要大量实时rollout，资源消耗大于离线训练。
- **依赖高质量验证器**：性能受限于验证逻辑的准确性；若验证器误判，可能阻碍有效梯度传播。
- **当前主要面向单请求场景**：虽有潜力扩展至批量服务，但DLMRT目前专注于低延迟单请求优化。

### 未来工作方向
- 探索更轻量的on-policy机制，例如部分重放或重要性采样，以降低训练成本。
- 将ORL思想推广至其他序列生成任务（如语音合成、图像生成）中的扩散模型。
- 结合强化学习进一步优化解码策略，超越纯监督学习的边界。
- 开发支持动态批处理的DLMRT变体，适应多样化的部署场景。

> 📣 **开源声明**：作者已公开代码、模型、训练数据、DLMRT引擎及项目页面，便于复现与后续研究。

</details>

---

### 3. [Exact Makespan Bounds and the GPipe Homogeneity Bias for Heterogeneous LLM Inference Pipelines](https://arxiv.org/abs/2610.21815)

**Authors**: Zhu Wenbo  
**Category**: cs.DC  
**Published**: 2026-10-16  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2610.21815v1  

#### Abstract
We give an exact critical-path characterization of the makespan of micro-batch pipelines serving LLMs on heterogeneous GPU clusters, quantifying how the classic homogeneous GPipe formula errs. In the pure-forward, communication-free, infinite-buffer setting, T(M) = (M-1)tau_max + Sum_j tau_j, where ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Exact Makespan Bounds and the GPipe Homogeneity Bias for Heterogeneous LLM Inference Pipelines*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

该论文针对 **Heterogeneous LLM Inference Pipelines**（异构大语言模型推理流水线）中的 **makespan 预测不准确** 问题展开研究。传统基于 **GPipe** 的分析假设所有 pipeline stage 处理时间相同（homogeneous），但在真实部署中，GPU 设备往往跨代混合（如 RTX 5090、3090、1070），导致各 stage 的处理时间差异巨大（可达 10–1000×）。这种异构性使得经典公式严重高估或低估实际延迟。

此外，通信机制（如 NCCL 异步 vs 同步传输）对性能的影响也缺乏精确建模。

### ✅ 提出了什么新方法或新思路

作者提出了 **三个精确的闭式表达式（closed-form formulas）**，用于计算在不同通信模式下的 **makespan**（完成最后一个 micro-batch 在最后一级 stage 的时间）：

1. **Theorem 1（Communication-free）**  
   $$
   T(M) = (M-1)T_{\text{max}} + \sum_{j=0}^{S-1} T_j
   $$

2. **Theorem 2（Asynchronous communication）**  
   $$
   T(M) = 2\sum T_j + 2\sum d_j + (M-1)T_{\text{max}}
   $$

3. **Theorem 3（Blocking communication）**  
   定义有效服务时间 $\theta_s = T_s + d_s$，则  
   $$
   T(M) = (M-1)\theta_{\text{max}} + \sum \theta_s
   $$

这些公式基于 **critical-path 分析**，并利用“夹逼”（sandwich argument）证明其上下界一致，从而得到精确解。

### ✅ 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **精度** | 公式为**精确解**，非近似或启发式；验证误差 < 1e-9 |
| **可解释性** | 明确分解出 GPipe 公式的偏差来源：<br>• `homogeneity-overestimate`: $2(T_{\text{max}} - \bar{T})$<br>• `communication-underestimate`: $2\sum d_j$ |
| **实用性** | 给出优化优先级建议：“先平衡 compute，再优化通信” |
| **泛化能力** | 覆盖三种通信模式，并提出 real system 的 **async-blocking sandwich bound**：<br>$T_{\text{async}} \leq T_{\text{real}} \leq T_{\text{block}}$ |

---

## 2. 核心实验方法和设置

### ✅ 使用了哪些数据集

- **合成配置集（Synthetic Configurations）**：
  - 手工设计 11 种典型场景（见 Appendix B 表格），覆盖：
    - 不同瓶颈位置（head/middle/tail）
    - 均匀/非均匀通信延迟
    - 是否触发 phase transition（即 $d_j > T_{\text{max}} - T_j$）
  - 随机模糊测试（fuzz testing）生成 20,000 随机配置，共 1.6×10⁵ 测试点

- **真实生产日志（Production Logs）**：
  - 来自一个运行 legalone 系列模型（1.7B/4B/8B）的异构推理集群
  - 包含 **8,571 条已完成请求记录**
  - 涉及设备：RTX 5090 Laptop、RTX 3090、RTX 3080、Tesla P40、云节点等

### ✅ 实验设置和评估指标

| 设置项 | 内容 |
|-------|------|
| **系统模型** | S-stage pipeline，M 个 micro-batches，infinite buffer，in-order fill-drain |
| **通信模式** | 分别测试 communication-free、asynchronous、blocking 三种情况 |
| **评估方式** | 将闭式公式结果与动态规划递推公式（recurrence）逐点对比 |
| **评估指标** | 最大相对误差（max relative error）、是否出现反例（counterexample） |
| **验证层次** | • 数值验证（vs DP recurrence）<br>• 生产调度器行为验证（logs consistency） |

### ✅ 基线方法对比

- **Classic GPipe Formula（基线）**：
  $$
  T_{\text{GPipe}}(M) = (S + M - 1)T_{\text{max}}
  $$
  假设所有 stage 时间均为 $T_{\text{max}}$，忽略异构性和通信细节。

- 对比维度：
  - Makespan 预测准确性
  - 过度估计程度（overestimation ratio）
  - 优化指导意义（如是否引导错误资源分配）

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

| 指标 | 结果 |
|------|------|
| **最大相对误差** | < 1e⁻⁹（在 1.6×10⁵ 配置上无任何反例） |
| **浮点精度偏差** | 实际观测 < 1e⁻¹⁴，远低于理论舍入误差 |
| **经典公式过估幅度** | 在 M=1 时高达 **166.7%**（T=[0.5,1,1,5] 示例）<br>在 M=8 仍达 **29.4%**，且绝对误差恒定为 $2(T_{\text{max}} - T)$ |
| **真实系统验证** | 8,571 条请求中，end-to-end time ≥ compute time（支持 $T_{\text{real}} \geq T_{\text{async}}$） |

### ✅ 与基线方法的对比结果

| 场景 | Classic GPipe 表现 | 本文方法表现 |
|------|------------------|-------------|
| 异构性强（如 10× 差异） | 严重高估 makespan（+166.7% @ M=1） | 精确匹配 DP 递推结果 |
| 异步通信存在 | 忽略通信成本，可能低估 | 正确建模为加性常数 |
| 阻塞通信且 $d_j > T_{\text{max}} - T_j$ | 完全无法捕捉吞吐下降 | 准确预测 slope 从 $T_{\text{max}}$ 跃升至 $\theta_{\text{max}}$ |

> 图表支持：
> - Figure 1：异构 vs GPipe makespan 曲线对比
> - Figure 2：异步通信下平行线结构（slope = $T_{\text{max}}$）
> - Figure 3：阻塞通信下的 phase transition
> - Figure 4：async-blocking sandwich bound

### ✅ 消融实验结果

通过 Table 1 展示不同异构程度下的偏差增长趋势：

| 异构比（max/min T） | M=1 时过估比例 | M=8 时过估比例 |
|---------------------|---------------|---------------|
| 1.0×（均衡）       | 0.0%          | 0.0%          |
| 1.5×               | 20.0%         | 6.5%          |
| 4.0×               | 77.8%         | 18.9%         |
| 10.0×              | 166.7%        | 29.4%         |
| 50.0×              | 257.1%        | 35.5%         |

结论：
- 过估量随异构性单调增加
- 绝对误差为常数 $2(T_{\text{max}} - T)$，因此相对误差随 M 增大而衰减

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **GPipe 公式偏差本质是 fill-time artifact**  
   其高估部分 $2(T_{\text{max}} - T)$ 是一次性填充开销，不影响稳态吞吐，但会导致低 batch 场景下过度资源配置。

2. **通信影响取决于是否可重叠**  
   - 异步通信：$\sum d_j$ 仅为加性常数，不影响 steady-state throughput
   - 阻塞通信：当 $d_j > T_{\text{max}} - T_j$ 时，通信成为 throughput bottleneck（slope 变为 $\theta_{\text{max}}$）

3. **真实系统可被夹在两个极端之间**  
   $$
   T_{\text{async}} \leq T_{\text{real}} \leq T_{\text{block}}
   $$
   由此可定义 **overlap ratio** $p$ 并量化当前 runtime 接近理想程度。

4. **优化优先级明确**：
   - 第一优先级：**balance compute**（降低 $T_{\text{max}}$）
   - 第二优先级：仅当 compute slack 小于 link latency 时才优化通信或启用 overlap

5. **bottleneck 位置不影响总 makespan**（在 async/infinite-buffer 下）  
   只需关注 $T_{\text{max}}$ 本身，而非其所在 stage。

### ✅ 方法的局限性（Limitations）

论文在 §10 中坦诚指出以下未解决挑战：

| 限制 | 描述 |
|------|------|
| **(i) Bandwidth-limited communication** | 多 micro-batch 共享链路时，通信时间随并发上升，超出 constant-$d_j$ 模型 |
| **(ii) Bounded buffers** | 有限缓冲区引入 backpressure，可能导致上游阻塞，破坏 ASAP 调度最优性 |
| **(iii) Prefill-decode interleaving** | 实际 serving 中 prefill 与 decode 切换，KV cache 更新带来依赖复杂性 |
| **(iv) Backward dependencies** | 不适用于 training（有反向传播） |
| **(v) Time-varying service times** | 忽略 thermal throttling、cache hit、负载波动等动态因素 |

### ✅ 未来工作方向

- 构建 **stochastic extension** 模型以处理动态 service time 和 queueing 效应
- 发展 **bandwidth-constrained scheduling theory**，替代 constant-$d_j$ 假设
- 将本框架集成到自动并行系统（如 Alpa、FlexFlow）作为 partitioner objective
- 开发在线监控工具，实时估算 $p$（overlap ratio）以指导运维决策

---

> 🔗 **代码与数据开放**：  
> 所有实现、验证脚本、生产数据集已开源：https://github.com/ytidb6688/heteropipe  
> 支持 pip 安装 `heteropipe` 库直接调用公式进行 makespan 计算与 bias 分析。

</details>

---

### 4. [DropKV: Decoupling Residual-Output Perturbation for Near-Optimal KV-Cache Eviction](https://arxiv.org/abs/2610.22130)

**Authors**: Aozhong Zhang, Selcuk Gurses, Yanxia Deng, Naigang Wang, Chi-Chun Liu, Davis Wertheimer, Derrick Liu, Xin Li, Zi Yang, Felix X. -F. Ye, Penghang Yin  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.22130v1  

#### Abstract
Key-value (KV) cache management has become a critical bottleneck for scaling Large Language Models (LLMs) to long-context scenarios due to linear memory growth. While recent eviction methods have moved beyond simple attention-score heuristics to incorporate value representations, they typically face...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **DropKV: Decoupling Residual-Output Perturbation for Near-Optimal KV-Cache Eviction**  
**核心结论与实验结果总结**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
在基于 Transformer 的大语言模型（LLM）中，自回归推理依赖于 **Key-Value (KV) Cache** 来缓存历史注意力状态，以避免重复计算。然而，KV Cache 的内存占用随序列长度线性增长，在长上下文场景（如处理 128K 或百万级 token）下会迅速耗尽 GPU 内存，成为可扩展推理的关键瓶颈。

现有的 **KV Cache 压缩方法** 主要分为三类：
- **量化（Quantization）**：降低数值精度。
- **低秩压缩（Low-rank Compression）**：减少表示维度。
- **缓存驱逐（Eviction）**：选择性丢弃部分 token。

其中，**驱逐方法** 因其简单高效而受到关注，但面临两大挑战：
1. **优化目标复杂**：最优驱逐策略需最小化对注意力输出的扰动，这是一个 **NP-hard 组合优化问题**。
2. **实现效率低**：许多先进方法因设计复杂，难以高效 kernel 化，影响实际部署。

### **提出了什么新方法或新思路**
本文提出 **DropKV**，一种基于 **Decoupled Residual-Output Perturbation（解耦残差-输出扰动）** 的 KV Cache 驱逐框架，核心思想如下：

#### ✅ **创新点 1：解耦联合驱逐决策**
传统方法试图联合优化多个 token 的驱逐组合，导致计算不可行。DropKV 提出将组合优化问题 **解耦为独立的 per-token 扰动最小化**，通过以下方式近似最优解：
- 定义每个 token $j$ 被驱逐时对注意力输出的扰动为：
  $$
  \Delta(\{j\}) = \frac{p_j}{1 - p_j}(a - v_j)
  $$
  其中 $p_j$ 是注意力权重，$a$ 是原始输出，$v_j$ 是对应值向量。
- 采用 **top-k 最小扰动** 策略选择驱逐 token，而非联合搜索。

#### ✅ **创新点 2：理论近似保证**
在 **cone condition（锥条件）** 下，DropKV 的解耦策略能提供 **常数因子近似保证**，即其性能与 NP-hard 问题的最优解相差一个常数倍。该条件在实践中被验证广泛成立。

#### ✅ **创新点 3：高性能 Triton 实现**
为避免显式构建完整的注意力矩阵（$O(n^2)$ 内存），作者开发了一套 **融合 Triton 内核（fused Triton kernels）**，实现了：
- 在线计算注意力权重和扰动分数。
- 显著减少内存占用（从 ~9GB 降至 ~17MB）。
- 加速评分内核达 **19.8×**，端到端预填充速度提升最高 **9.9%**。

### **相比现有方法的优势**
| 方面 | DropKV | 现有方法（如 SnapKV, AnDPro） |
|------|--------|-------------------------------|
| **理论基础** | 有近似最优性证明 | 多为启发式或次优松弛 |
| **价值感知** | 显式建模 value 向量扰动 | 多仅依赖 attention score |
| **实现效率** | 支持 fused kernel，内存友好 | 易受 materialization 瓶颈限制 |
| **性能表现** | 准确率更高，延迟更低 | 在高压缩比下性能下降明显 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **RULER**：合成基准，支持可配置的长序列任务，用于评估模型在超长上下文下的能力。
- **LongBench**：多任务、双语基准，涵盖单文档问答、多文档问答、摘要、少样本学习、代码补全等，平均长度约 6.7K。

### **实验设置**
- **模型**：
  - `Llama-3.1-8B-Instruct`
  - `Mistral-7B-512K`（支持 512K 上下文）
  - `Qwen2.5-7B-1M`（支持 1M 上下文）
- **缓存预算**：5%、10%、30%
- **查询窗口大小（w）**：8
- **池化核大小（K）**：11
- **硬件平台**：NVIDIA A100-80GB GPU
- **评估工具**：统一使用 `lm-evaluation-harness`

### **评估指标**
- **准确率（Accuracy）**：在 RULER 和 LongBench 上的平均得分。
- **延迟（Latency）**：端到端生成时间、首次响应时间（TTFT）、预填充速度。
- **吞吐量（Throughput）**：每秒生成 token 数（tok/s）。
- **内存占用**：峰值内存使用量。

### **基线方法对比**
| 方法 | 类型 | 是否价值感知 |
|------|------|--------------|
| `SnapKV` | Query-aware, attention-only | ❌ |
| `PyramidKV` | Adaptive budget | ❌ |
| `StreamingLLM` | Sink + recent window | ❌ |
| `KeyDiff` | Key similarity | ❌ |
| `CriticalKV` | Output-aware, upper bound | ✅ |
| `AnDPro` | Output-aware, convex relaxation | ✅ |
| `DropKV` (**ours**) | Output-aware, exact decoupled | ✅ |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **RULER 基准（128K 序列，5% 缓存预算）**
| 方法 | Llama-3.1-8B | Mistral-7B-512K | Qwen2.5-7B-1M | 平均 |
|------|---------------|------------------|----------------|------|
| Full KV | 84.84 | 89.02 | 87.01 | 86.96 |
| SnapKV | 79.57 | 77.72 | 76.68 | 77.99 |
| CriticalKV | 78.72 | 78.06 | 77.43 | 78.07 |
| AnDPro | 79.65 | 81.37 | 79.90 | 80.31 |
| **DropKV** | **82.00** | **85.40** | **83.77** | **83.72** |

> ✅ **DropKV 在所有模型上均取得最高准确率**，相比最强基线分别提升 **+2.35**, **+4.03**, **+3.87** 分。

#### 🔹 **LongBench 基准（5% 缓存预算）**
| 方法 | Llama-3.1-8B | Mistral-7B-512K | Qwen2.5-7B-1M | 平均 |
|------|---------------|------------------|----------------|------|
| Full KV | 49.91 | 40.15 | 46.47 | 45.51 |
| AnDPro | 44.53 | 35.04 | 41.58 | 40.38 |
| **DropKV** | **45.51** | **36.22** | **42.66** | **41.46** |

> ✅ DropKV 在所有模型上均优于其他驱逐方法，且显著优于 AnDPro。

### **与基线方法的对比结果**
- **准确率优势**：
  - 在 RULER 上，DropKV 比 AnDPro 平均高 **+3.41** 分。
  - 在 LongBench 上，DropKV 比 AnDPro 平均高 **+1.08** 分。
- **延迟优势**：
  - **端到端延迟最低**：在 `Mistral-7B-512K` 上，`B=12` 时 DropKV 比 SnapKV 快 **~1.2×**。
  - **首次响应时间（TTFT）接近动态缓存**：仅高出 **0.6%**，说明预填充开销极小。
- **吞吐量优势**：
  - 在 5% 预算下，DropKV 的最大批处理规模是动态缓存的 **1.75–8.22×**。
  - 解码吞吐量提升最高达 **8.22×**（Mistral, n=128K）。

### **消融实验结果**
- **Triton 实现加速效果**：
  - 在 `n=128K` 时，评分内核速度提升 **19.8×**。
  - 峰值内存从 **~9GB** 降至 **~17MB**（减少 **540×**）。
  - 端到端预填充速度提升 **+9.9%**（匹配批次下）。
- **理论假设验证**：
  - **Cone Condition** 在 Llama、Mistral、Qwen2.5 上分别有 **91.2%、91.1%、76.5%** 的 (layer, head, query) 三元组满足。
  - 实际近似比（approximation ratio）中位数仅为 **1.16**，远优于理论最坏情况（~10–12），表明算法在实践中非常接近最优。

---

## 4. **关键结论和发现**

### **主要发现**
1. **解耦策略有效且近似最优**：尽管将组合问题解耦为独立 token 评分看似粗略，但在 **cone condition** 成立时，DropKV 能提供 **常数因子近似保证**，且实证显示其实际性能远优于理论界。
2. **value 向量至关重要**：仅依赖 attention score 的方法（如 SnapKV）会忽略 value 向量的方向和大小，导致误删关键 token。DropKV 显式建模 $\|a - v_j\|$，显著提升准确性。
3. **实现效率决定实用性**：即使算法理论上更优，若无法高效 kernel 化（如需 materialize attention matrix），仍难部署。DropKV 的 **fused Triton kernel** 成功规避此瓶颈，兼具高精度与高性能。
4. **DropKV 是当前最优驱逐方案**：在多个长上下文模型和基准上，DropKV 在 **准确率、延迟、吞吐量** 三项指标上全面领先，是目前最高效的 KV Cache 驱逐方法。

### **方法的局限性**
- **依赖 cone condition**：虽然该条件在大多数情况下成立，但在早期层或特定头中可能不满足，此时性能可能略有下降。
- **未覆盖 decode-time 优化**：当前 Triton 实现仅加速 **prefill 阶段** 的评分，decode 阶段仍使用常规实现。
- **静态预算分配**：默认版本未结合 adaptive budget allocation（如 AdaKV），但可通过 AdaDropKV 扩展支持。

### **未来工作方向**
- 将 DropKV 扩展至 **decode-time Triton kernel**，实现全流程加速。
- 探索更鲁棒的 cone condition 替代形式，提升在边缘情况下的稳定性。
- 结合 **adaptive head budget** 与 DropKV 评分机制，进一步优化资源分配。
- 探索 DropKV 在 **MoE 架构** 或 **多模态模型** 中的应用潜力。

---

> **总结**：DropKV 通过 **解耦残差-输出扰动**，在理论近似性和工程实现效率之间取得了卓越平衡。它不仅提供了坚实的数学基础，还通过高性能 Triton 实现解决了实际部署中的内存瓶颈，是当前长上下文 LLM 推理中最先进的 KV Cache 驱逐方案。

</details>

---

### 5. [Functional Subnetwork Localization for Efficient Continual Learning in Large Language Models](https://arxiv.org/abs/2610.21537)

**Authors**: Shuoran Jiang, Wentao Zhu, Yang Xiang, Dongfeng Li, Yunan Zhang, Mengchen Zhao, Yuhui Guo, Kai Chen, Youcheng Pan, Xiangping Wu, Qingcai Chen  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.21537v1  

#### Abstract
Biological intelligence continually adapts to new information while retaining prior skills through localized plasticity. In contrast, continual learning in large language models (LLMs) remains vulnerable to catastrophic forgetting. A promising direction is to identify functionally specialized subnet...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Functional Subnetwork Localization for Efficient Continual Learning in Large Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在持续学习（Continual Learning）过程中面临**灾难性遗忘**（Catastrophic Forgetting）的挑战。传统方法通常对全局参数进行更新，导致新知识覆盖旧知识，破坏已习得的表示，并且标准优化器（如AdamW）维护密集的动量状态，带来高昂的内存开销。

### 🚀 提出的新方法：CiMEO
作者提出了一种名为 **Cognition-isolated Memory-Efficient Optimization (CiMEO)** 的新框架，其核心思想是：
- 利用**梯度归因分析**（Gradient Attribution）识别与不同认知功能相关的稀疏子网络；
- 将每个训练样本路由到对应的功能子网络中进行局部更新；
- 仅在这些选定区域更新参数并维护优化器状态，实现**功能隔离**与**内存高效**。

### 🔍 创新点
1. **功能子网络定位**（Functional Subnetwork Localization）  
   首次系统地通过梯度归因揭示预训练LLM中存在与 **Bloom's Taxonomy** 六大认知类别（Memory, Understand, Apply, Analyze, Evaluate, Create）高度特异化的子网络，且它们之间的重叠极低（平均 < 0.002%），表明LLMs内部具有“内生性”的功能模块化结构。

2. **认知引导的持续学习范式**  
   引入一个冻结的认知分类器（Cognition Classifier），为每条指令分配认知标签，并据此将优化过程路由至对应的子网络，避免跨任务干扰。

3. **状态序列化优化器 S2-AdamW**  
   设计了 **State-Serialized AdamW (S2-AdamW)**，只存储被选中参数的优化器状态（first- 和 second-moment），显著降低内存占用，尤其适用于长期部署场景。

### 💡 相比现有方法的优势
| 维度 | 传统方法 | CiMEO |
|------|--------|-------|
| 更新方式 | 全局参数更新（Global Updates） | 局部子网络更新（Localized Updates） |
| 干扰控制 | 依赖正则化/回放缓冲区 | 参数隔离 + 功能解耦 |
| 内存效率 | AdamW需维护全部动量状态 | 仅保留活跃子网的状态 |
| 可扩展性 | 随任务增长成本剧增 | 固定稀疏比例，可扩展性强 |
| 是否需要回放 | 多数需要Rehearsal Buffer | 完全无需回放机制 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 主要数据集：**AM-DeepSeek-R1-Distilled-1.4M**  
  包含140万条指令数据，经过标注映射到 Bloom’s Taxonomy 的六个认知层级。
- 用于评估的五个基准测试集：
  1. **HumanEval-CN**：中文代码生成能力（Pass@1）
  2. **GaokaoBench_2010**：高考多学科推理题
  3. **CMMLU**：涵盖人文、社科、STEM等67个领域的多任务理解
  4. **CEVAL**：公务员、法律、医学等专业考试问答
  5. **GSM8K**：小学数学应用题（算术与逻辑推理）

### ⚙️ 实验设置
- 模型基础：**Qwen3-1.5B-Base**
- 批大小（Batch Size）：64
- 学习率：1e-5（恒定）
- 子网络选择比例：top p% 参数（基于梯度归因得分排序）
- 持续学习顺序：按1.4M指令流顺序训练，模拟真实增量场景

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| `Average Performance` | 在五个benchmark上的平均准确率 |
| `Forgetting Rate (Vf)` | 新模型忘记原始模型正确预测的比例 |
| `Incremental Rate (Vi)` | 成功从旧模型错误中纠正的比例 |
| `Subspace Overlap` | 权重空间主子空间相似度，衡量稳定性 |
| `Principal Angles` | 衡量微调前后主导变换方向的变化程度 |

### 🆚 基线方法对比
包括以下几类主流方法：
- **Full-Parameter Fine-tuning**：AdamW
- **PEFT 方法**：LoRA, MiSS, Bone, SHiRA, DoRA, LoRA-FA, OLoRA, PiSSA, ROAD
- **MEFT 方法**（Memory-Efficient FT）：AdaLOMO, GaLore, AdaRankGrad, ZO-AdaMU

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table I）

| 方法 | HumanEval-CN | Gaokao | CMMLU | CEVAL | GSM8K | **Avg (%)** |
|------|--------------|--------|--------|--------|--------|------------|
| Qwen3-1.5B-Base | 4.27 | 5.14 | 0.02 | 58.01 | 66.87 | 26.86 |
| Full-Parameter (AdamW) | 3.05 | 11.42 | 0.00 | 67.40 | 39.27 | 24.23 |
| LoRA | 3.66 | 6.99 | 0.02 | 60.44 | 68.08 | 27.84 |
| SHiRA | 1.22 | 9.26 | 29.61 | 63.13 | 68.01 | **34.25** |
| LoRA-FA | 6.10 | 9.62 | 48.40 | 60.88 | 66.03 | 38.20 |
| **CiMEO (Ours)** | **25.00** | 6.98 | **28.85** | **66.22** | **67.55** | **38.92** |

> ✅ **CiMEO取得最高平均性能（38.92%）**，远超大多数基线，在HumanEval-CN上仅次于OLoRA（29.27%），但在其他任务更均衡。

### 🔁 抗遗忘与增量学习表现（图4）
- **平均遗忘率最低之一**，接近最优水平；
- **增量学习能力强劲**，优于AdaLOMO、GaLore等强调稳定性的方法；
- 实现了良好的 **Stability-Plasticity Trade-off**：既不过度遗忘，又能有效吸收新知识。

### 🔍 消融实验（Ablation Study）
| 变体 | Avg Performance | 说明 |
|------|------------------|------|
| CiMEO (default) | **38.92** | 使用最大梯度归因选择子网络 |
| CiMEO w/ min_grad | 27.36 | 使用最小梯度参数 → 性能大幅下降 |
| CiMEO w/ random_grad | 25.95 | 随机选择参数 → 几乎无提升 |

> ❗ 结果验证了**梯度幅度反映神经元对特定认知刺激的敏感性**，是有效子网络选择的关键信号。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLMs存在内生性功能模块化结构**  
   通过梯度归因分析发现，不同认知任务激活的参数子集几乎不重叠（<0.002%），支持“认知隔离”假设。

2. **局部更新有助于保持预训练表示稳定性**  
   子空间重叠分析和主角分析显示，CiMEO引起的权重变化较小，主导特征空间基本保持不变，解释了其抗遗忘机制。

3. **功能导向的参数更新优于任务/数据驱动划分**  
   相较于按任务ID或数据集切分参数的方法，基于认知语义的路由更具泛化性和可解释性。

4. **内存与性能可以兼得**  
   CiMEO在减少优化器状态的同时，反而提升了整体性能，打破了“节省资源必牺牲效果”的固有印象。

### ⚠️ 局限性
- 当前方法依赖于高质量的认知标签构建训练语料（需人工或强模型标注）；
- 子网络选择比例（p%）为超参，缺乏自适应调节机制；
- 路由策略基于静态分类器，未考虑动态上下文或混合认知需求；
- 尚未在更大规模模型（如70B以上）验证可扩展性。

### 🔮 未来工作方向
- 构建自动化的认知标签生成 pipeline；
- 探索动态稀疏更新与自适应子网生长机制；
- 将该范式推广至多模态持续学习场景；
- 结合神经科学启发机制（如记忆巩固、突触固化）进一步增强稳定性。

---

## 总结
> **CiMEO 是一种受生物学习启发的新型持续学习框架，它通过识别并利用LLM中的功能子网络，实现了高效、稳定、低干扰的知识积累。其实验证明：在大型语言模型中，“在哪里更新”可能比“如何更新”更重要。**

该研究为解决LLM长期部署中的灾难性遗忘问题提供了新视角——**不是对抗遗忘，而是设计结构使其自然免疫**。

</details>

---

### 6. [LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models](https://arxiv.org/abs/2610.21594)

**Authors**: Zixuan Liu, Nan Xue, Rui Bao, Yuhang Xu, Xiaofeng Cao, Jiangchao Yao, Zhiyong Chen  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.21594v1  

#### Abstract
Looped Transformers improve parameter efficiency by repeatedly applying the same Transformer layers. However, generating each token still involves sequential computation and repeated access to model weights, leading to high decoding latency. Recent pipelined self-speculative decoding methods reduce ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Looped Transformers 通过重复应用相同的 Transformer 层来提升参数效率，但在全深度自回归解码（autoregressive decoding）下，每个 token 的生成仍需完成所有循环层计算，并反复访问模型权重，导致**高解码延迟**和内存带宽压力。

现有的流水线自推测解码方法（如 LoopSpec）虽然能并行处理不同循环深度的 token，但仍受限于：
- 无法充分利用早期循环层的状态进行更长的前瞻预测（lookahead）；
- 错误的推测分支会持续消耗计算资源，直到最终验证。

### **提出的新方法**
本文提出了 **LoopAhead**，一种面向 Looped Transformers 的**无损推测解码框架**（lossless speculative decoding），其核心思想是：
- 利用**早期循环层状态**（early-loop states）提前预测多个未来 token；
- 引入**中间层反馈机制**（intermediate-loop feedback）动态控制推测分支的执行；
- 最终输出仍由全深度模型决定，保证输出分布不变。

### **相比现有方法的优势**
- **更早的 token 预测**：通过轻量级的 **Lookahead Drafter (LAD)** 在早期循环中预测多个未来 token，突破了 LoopSpec 中“必须等待前一个 token 达到指定深度”的限制。
- **动态调度减少浪费**：利用中间层反馈判断推测分支是否冲突，对不 promising 的分支进行暂停（suspend），避免无效计算。
- **保持无损性**：所有最终 token 决策仍由全深度模型验证，确保输出与原始模型一致。
- **更高的吞吐量增益**：在多种任务上实现高达 **7.82× 的解码加速**，显著优于 LoopSpec 和 DFlash。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖多个推理与代码生成基准：
- **数学推理**：GSM8K（1,319题）、MATH500（500题）
- **代码生成**：HumanEval（164题）、MBPP（500题）
- **通用推理**：BBH（540题）
- 补充任务：CNN/DailyMail（摘要）、AlpacaEval、IFEval、MT-Bench

### **实验设置**
- **模型**：在两个 Looped LM 家族上测试：
  - **Ouro**：Ouro-1.4B、Ouro-2.6B、Ouro-1.4B-Thinking（4次循环）
  - **Raven**：Raven-Llama-3.2、Raven-OLMo-2、Raven-TinyLlama（32次循环）
- **硬件**：NVIDIA A40 GPU（48GB），每 GPU 单请求
- **精度**：BF16，本地 PyTorch 后端
- **解码模式**：greedy decoding 与 sampling（T=1, top-p=0.9）

### **评估指标**
- **解码吞吐速度提升**（speedup over AR）：以 tokens/sec 为单位，相对于标准自回归解码的加速比。
- **平均循环次数 / 输出 token**（loops per token）
- **候选推进次数 / 输出 token**（candidate advances per token）
- **每循环推进候选数**（candidates per loop）

### **基线方法对比**
| 方法 | 类型 | 是否需要训练 |
|------|------|-------------|
| **AR** | 自回归解码 | 否 |
| **LoopSpec** | 流水线自推测解码 | 否（无需额外训练） |
| **DFlash** | 扩散模型驱动的块推测 | 是 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **Greedy Decoding 结果（Table 1）**
- **最高加速比**：**7.82×**（Raven-TinyLlama / HumanEval）
- **普遍加速范围**：**2.82× ~ 7.82×** 超过 AR
- **相对 LoopSpec 的增益**：
  - 最高提升 **96.2%**（Raven-TinyLlama / MATH500）
  - Ouro-2.6B / BBH 提升 **79.2%**

#### **Sampling Decoding 结果（Table 2）**
- 平均加速比：**3.00× ~ 4.32×** 超过 AR
- 相对于 LoopSpec 的吞吐提升：**16.2% ~ 36.8%**
- 在 MATH500 和 BBH 上增益稳定超过 28%

#### **补充任务表现（Table 8）**
在摘要与指令遵循任务上同样有效：
- Raven-TinyLlama 在 IFEval 上达 **6.37×** 加速
- 所有 16 个 model-benchmark 对均优于 LoopSpec

### **消融实验结果（Table 3, Figure 5）**
| 组件 | 影响 |
|------|------|
| **仅 LAD（无调度）** | 可能减慢解码（因无效计算） |
| **启用反馈调度** | 显著提升速度，减少废弃节点的计算量 |
| **紧凑 draft vocabulary** | 降低投影开销，提升效率 |
| **冲突暂停机制** | 有效限制错误路径的深层执行 |

> 示例：在 Ouro-1.4B 上，启用调度后，废弃候选的平均推进次数从 **2.19** 降至 **0.93**，提速 **48.4%**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **早期循环层具有强预测能力**：尽管单独使用早期循环无法达到最终准确率，但其 token 预测与全深度输出高度一致（Figure 1），支持用于前瞻推测。
2. ✅ **多 token 前瞻 + 动态调度 = 更高效推测**：LAD 提供早期候选，反馈机制动态剪枝，二者协同作用，显著减少无效计算。
3. ✅ **LoopAhead 是无损且通用的**：在 greedy 与 sampling 下均保持输出一致性，适用于多种 Looped LM 架构。
4. ✅ **两种加速路径并存**：
   - Ouro 类模型：通过减少循环次数加速
   - Raven 类模型：通过减少每循环候选数加速（更高效的执行控制）

### **方法的局限性**
1. **需要模型特定适配**：
   - 不同循环次数的模型需调整 LAD 输入深度和反馈深度
   - 需基于开发集选择超参（如 draft length、feedback depth）
2. **收益受任务影响**：
   - 在某些任务上，LAD 预测准确性不足时，额外推测可能带来净成本
   - 收益并非在所有任务上都巨大（部分仅提升 2–5%）
3. **静态调度策略**：
   - 当前调度预算和规则固定，未适应动态负载或请求长度变化
   - 多请求并发场景下的表现未知

### **未来工作方向**
- **自适应调度机制**：根据候选质量动态调整执行预算
- **跨架构泛化**：设计统一的 LAD 接口，减少对特定模型的依赖
- **在线学习与更新**：让 LAD 在部署中持续优化
- **多用户并发优化**：研究在高并发、变长请求下的资源分配策略

---

> **总结一句话**：  
> **LoopAhead 通过“早期预测 + 中间反馈调度”机制，在不牺牲输出质量的前提下，实现了 Looped Language Models 的高效推测解码，最高达 7.82× 解码加速，为参数高效大模型的实际部署提供了有力支持。**

</details>

---

### 7. [When the KV Cache Is Slower Than the GPU: Scaling Remote KV Tiers for Context-Parallel Prefill](https://arxiv.org/abs/2610.21500)

**Authors**: Muhammad Ibrahim  
**Category**: cs.DC  
**Published**: 2026-10-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.21500v1  

#### Abstract
Reusing the attention KV cache across requests is a standard way to cut prefill cost in long-context LLM serving, and production systems increasingly back the per-engine cache with a remote tier shared by every engine. We study such a tier in a disaggregated deployment of GLM-5.3, an FP8 mixture-of-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When the KV Cache Is Slower Than the GPU: Scaling Remote KV Tiers for Context-Parallel Prefill*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

本文揭示并解决了在**长上下文 LLM 推理服务中，远程 KV 缓存（remote KV tier）反而比 GPU 重新计算更慢**的问题。在基于 `context-parallel prefill`（PCP）架构的服务部署中，启用远程缓存（如 Valkey）后，尽管命中率高，但尾延迟（p99 TTFT）显著恶化，甚至导致系统不稳定。

根本原因在于：
- 远程加载带宽不足，无法匹配现代 GPU 的 prefill 吞吐；
- 同步加载阻塞所有 PCP ranks，形成不稳定的队列；
- 客户端存在导致无限挂起的 bug。

### 提出了什么新方法或新思路

作者提出了一套完整的优化方案，核心创新如下：

#### ✅ **(1) 跨 PCP ranks 的 MLA KV 存储分片（Sharded MLA KV Store）**
- 将 KV chunk 按索引模 `W`（rank 数）分配给不同 rank 拥有（ownership），每个 rank 只存储、查找和加载自己负责的 chunk。
- 利用 vLLM 调度器已有的“取各 rank 最小缺失 chunk”机制，证明该策略能正确返回最长可复用前缀，无需新增通信协议。
- 所有 rank 执行相同的 collective 序列，避免因异步操作导致的死锁。

#### ✅ **(2) 按 rank 分区的远程缓存层（Rank-Partitioned Remote Tier）**
- 每个 PCP rank 对应一个独立的 Valkey 实例，无须集群协调或一致性哈希。
- 因为分片规则是静态确定的（chunk index → rank），天然支持跨引擎复用（cross-engine reuse）。
- 显著提升 aggregate GET 带宽（达 6.3×）。

#### ✅ **(3) 异步加载支持（Asynchronous Loading for Sharded Store）**
- 扩展 LMCache 的异步路径以支持分片模式。
- 各 rank 独立预取自有 chunk，并通过自定义累计长度向量实现与同步路径一致的命中判断逻辑。
- 避免单个慢请求阻塞整个引擎。

#### ✅ **(4) 修复三个关键客户端 Bug**
- **丢失 completion race**：提交请求与注册 future 不原子，导致 completion 被丢弃 → 加锁保证原子性。
- **GET 缺失 key 不返回**：RESP 协议解析错误，读取固定头长度导致连接卡死 → 改为按 CRLF 解析，正确处理 `$-1\r\n`。
- **异步 GET 使用错误 key 名**：未重写为 worker-0 的统一命名 → 修正 key 映射。

### 相比现有方法的优势

| 维度 | 传统方法（如原版 LMCache） | 本文方法 |
|------|--------------------------|--------|
| 缓存 I/O 负载 | 集中于 rank 0，成为瓶颈 | 均匀分布到所有 ranks |
| 远程 tier 扩展性 | 单实例受限于单线程吞吐 | 多实例无协调扩展，线性提升带宽 |
| 加载方式 | 同步阻塞，易引发排队 | 支持异步，隔离慢请求影响 |
| 系统稳定性 | 易因 bug 或负载激增而挂起 | 更健壮，具备故障降级能力 |
| 性能收益 | 在高命中流下反而劣化 | 在 98.5% L2 命中率下仍大幅优化尾延迟 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **真实生产请求日志回放（trace replay）**
  - 来源于企业级 SLA 工作负载，包含 683 个 9k–200k token 的长提示请求。
  - 时间戳保留，用于模拟真实到达模式。
- **重点分析子集**：选取前 450 个请求构成“高命中流”（hit-heavy stream），总大小约 7.35M tokens，在私有 Valkey 中可完全预热，L2 命中率达 **98.5%**。

### 实验设置

- **模型**：GLM-5.3（FP8 混合专家 MoE 模型，采用 DeepSeek 风格的 **Multi-Head Latent Attention, MLA**）
- **硬件**：8×NVIDIA B300 GPU 节点，运行 **PCP8**（Prefill Context Parallelism，序列分块跨 8 个 ranks）
- **缓存系统**：LMCache + Valkey（Redis 兼容 KV 存储）
- **部署架构**：Prefill/Decode disaggregation，prefill 与 decode 分离节点执行
- **对比配置**：
  - Baseline：仅本地 L1 缓存（host memory）
  - Original：启用单实例远程 L2 缓存（同步加载）
  - Proposed：分片 + 多实例 + 异步 + 修复 bug

### 评估指标

| 指标 | 描述 |
|------|------|
| **TTFT（Time-To-First-Token）** | 主要延迟指标，特别是 p50、p90、p99 |
| **Prefill Latency** | 仅 prefill 阶段完成时间（不含 decode） |
| **GPU Utilization** | GPU 实际计算占用率 |
| **Throughput (k tokens/s)** | 每秒处理的输入 token 数 |
| **Cache Hit Rate / Source** | L1 vs L2 命中比例 |
| **Queue Length & Stability** | 引擎内并发请求数变化趋势 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔹 高命中流量下的尾延迟改善（Table 2, Fig. 4）

| 配置 | p50 (s) | p99 (s) | max (s) | GPU 利用率 |
|------|--------|--------|--------|-----------|
| H1b（1实例，同步） | 9.70 | **83.32** | 86.72 | 7% |
| H5b（8实例，异步+全修复） | 0.40 | **0.67** | 0.86 | 22% |

➡️ **p99 下降 124×**（83.32s → 0.67s），且延迟曲线平坦，不再随时间增长。

#### 🔹 远程 tier 带宽提升（Fig. 3）

| 客户端数 | 实例数 | Aggregate GET 带宽 |
|---------|-------|--------------------|
| 1 client | 1 instance | 1.51 GB/s |
| 8 clients | 1 instance | 1.71 GB/s（仅 +13%） |
| 8 clients | 8 instances（每 rank 一个） | **10.76 GB/s**（↑6.3×） |

➡️ 单实例受主线程限制，多实例实现近线性扩展。

#### 🔹 Prefill 吞吐提升（Fig. 7）

在 8 个 ~200k-token 并发请求下：

| 配置 | Throughput (k tokens/s) | 提升幅度 |
|------|------------------------|----------|
| 原始（rank-0 存储） | 43.2k | — |
| 分片存储 | 49.4k | **+14%** |
| 分片 + 上游 kernel fix（FlashInfer PR #3969） | 57.3–58.0k | **+34%** |
| 关闭 LMCache | ~60.6k | 接近上限 |

➡️ 分片本身带来 **14% 吞吐提升**，结合 kernel 优化可达 **34%**，接近无缓存开销水平。

#### 🔹 与重新计算的比较

- 16k-token 请求：
  - 重新计算 median: **0.31s**
  - 最优远程加载（H5b）median: **0.40s**
  - ➡️ 当前实现仍有 **~0.09s 固定开销（To）**
- 但远程命中的优势在于：
  - **不消耗 GPU compute**，适合饱和场景；
  - **尾延迟优于 recompute**（p99 0.67s vs 1.28s）

---

## 4. 关键结论和发现

### 主要发现

1. 🚨 **“远程缓存更快”并非默认成立**  
   在高性能 GPU（如 B300）上，只有当远程 tier 提供的带宽超过 **break-even bandwidth $ B^* = sR $** 时，加载才优于重新计算。  
   - 对本模型：$ s \approx 48.4\,\text{KB/token}, R \approx 52.2k\,\text{tokens/s} \Rightarrow B^* \approx \mathbf{2.53\,GB/s} $
   - 单 Valkey 实例仅提供 ~1.71 GB/s < B*，因此“命中比未命中还慢”。

2. ⚠️ **同步加载在高命中流下会导致系统 instability**  
   - 若 $ \rho = \lambda \cdot (ns/B) > 1 $，队列将无限增长。
   - 实测中单实例下 $ \rho \sim 1.13 > 1 $，导致 median 延迟从 3.5s 涨至 52s。

3. 🔍 **固定成本 $ T_0 \approx 0.33\,\text{s} $ 决定了短提示无法受益于远程缓存**  
   - 只有当 $ n > R \cdot T_0 \approx 17k $ tokens 时，远程加载才可能胜出。
   - 这意味着对于 ≤16k 的提示，**recompute 是最优选择**。

4. 💡 **Context Parallelism + Static Sharding 可实现免协调扩展**  
   - 利用 PCP 的静态 chunk 分布特性，自然实现 rank-partitioned remote tier，无需额外路由逻辑。

5. 🐞 **客户端 bug 对系统稳定性影响巨大**  
   - 一个 completion race 或 missing-key hang 可导致整个引擎永久阻塞。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **单次运行验证** | 所有实验仅运行一次，小幅度差异（如 <10%）缺乏统计显著性支撑 |
| **单一引擎测试** | 回放环境只有一个客户端访问 Valkey，未测量多引擎竞争下的表现 |
| **特定长度范围** | 结果集中在 14.7k–17.9k token 区间，未覆盖全谱系 |
| **To 成本尚未消除** | 当前 $ T_0 \approx 0.33s $ 仍是瓶颈，需进一步优化 collective 开销 |
| **依赖 MLA 和 PCP 架构** | 方法对其他 attention 结构（如 dense attention）需调整 |

---

### 未来工作方向

1. **降低固定成本 $ T_0 $**
   - 合并多个 chunk 的 broadcast 操作（batched collectives）
   - 重叠 metadata exchange 与数据 fetch
   - 引入 load-vs-recompute 动态决策策略

2. **扩展到更大规模部署**
   - 支持超过 rank 数量的 Valkey 实例（二级哈希分区）
   - 探索 RDMA 或 GPUDirect Storage 加速远程传输

3. **动态负载感知调度**
   - 根据当前网络状态和 prompt 长度决定是否尝试加载

4. **通用化设计**
   - 将分片机制推广至非 MLA 模型（如标准 multi-head attention）

---

> ✅ **最终成果**：所有改进均已提交至上游 **LMCache** 项目（PR #5545, #5547–#5553），具备实际工程落地价值。

</details>

---

### 8. [Long-Context Drafting Does Not Require Long-Context Drafters](https://arxiv.org/abs/2610.20865)

**Authors**: Aref Jafari, Vansh Bhatia, Matt Qin, Subhajit Dutta Chowdhury, Ben Lee, Vikram Appia, Ali Ghodsi, Mehdi Rezagholizadeh  
**Category**: cs.LG  
**Published**: 2026-10-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.20865v1  

#### Abstract
Speculative decoding pairs a cheap drafter with parallel verification against the target. At ultra-long context it stops paying, because a drafter that attends over the full prefix inherits the target's context-dependent cost, so the speedup it was meant to provide first erodes and then reverses. Th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Long-Context Drafting Does Not Require Long-Context Drafters》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在 **speculative decoding** 中，一个轻量级的 **drafter** 模型生成多个候选 token，由目标模型（target）并行验证，从而加速推理。然而，在超长上下文（ultra-long context）场景下，传统方法面临严重瓶颈：

- 如果 drafter 模型也需 attend 整个长上下文，则其计算成本随上下文长度 $N$ 增长，失去“廉价”的优势。
- 现有解决方案通常需要 **训练或微调** 专门的长上下文 drafter，或依赖目标模型内置的多 token 预测头（如 EAGLE、MTP），这限制了通用性和灵活性。

本文指出：**构建长上下文 drafter 是不必要的**，因为现有的 target-conditioned drafters 已通过目标模型提供的 hidden states 获取远距离上下文信息，无需自身 attention 覆盖全部历史。

---

### 提出了什么新方法或新思路

作者提出一种 **无需训练、冻结权重** 的方法，使短上下文 drafter 可直接用于百万 token 级别的长上下文场景。该方法包含两个关键机制：

#### （1）**Windowed Attention（窗口化注意力）**

- 限制 drafter 仅能访问最近 $W$ 个 token 的 KV cache，而非整个上下文。
- 这使得 drafter 的每步计算成本 **恒定于 $N$**，解决了成本随长度增长的问题。

#### （2）**Rotary Index Re-basing（旋转位置索引重映射）**

- 现有 drafter 模型（如基于 RoPE 的实现）对绝对位置索引敏感，当上下文超过其训练时的最大位置 $P_{\text{max}}$（如 1M）时，RoPE 查表会越界或精度下降。
- 作者提出将实际位置 $m$ 映射为 $m \mod P_{\text{max}}$，即在每个大小为 $P_{\text{max}}$ 的“索引带”（index band）内进行循环重用。
- **理论保证**：只要一个预测的所有依赖项落在同一个索引带内，则 re-basing 不改变其计算结果（Proposition 1 & Corollary 1）。

> ✅ **核心思想**：利用 target-conditioned 结构传递全局信息，用窗口化控制计算开销，用索引重映射解决 RoPE 外推问题。

---

### 相比现有方法的优势

| 方面 | 本文方法 | 现有方法 |
|------|---------|--------|
| **是否需要训练** | ❌ 否，冻结发布模型即可 | ✅ 通常需要训练/微调长上下文 drafter |
| **是否依赖特定架构** | ❌ 否，适用于外部 drafter | ✅ 是，如 MTP 需目标模型自带 multi-token head |
| **验证准确性** | ✅ 完全保留 full-context target 输出分布 | ⚠️ 如 SpecPV 使用稀疏 KV 缓存，牺牲准确性 |
| **部署复杂度** | ✅ 极简，仅需配置 vLLM 补丁 | ⚠️ 或高，涉及模型修改、训练流程 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **PG19**：书籍文本延续任务，主实验基准。
- **LongBench v1 & v2**：多语言、多任务长上下文理解基准。
- **BEAM / BEAM v2**：长对话场景，测试真实聊天模板下的表现。
- **SPEED-Bench**：短上下文吞吐量测试集，用于分析窗口代价。
- **RULER**：针在 haystack 中检索任务，测试长程记忆能力。

---

### 实验设置和评估指标

#### 模型配置

- **Target 模型**：
  - Kimi-K3（2.8T 参数）
  - Qwen2.5-7B-Instruct-1M
  - MiniMax-M3（block-sparse attention）
- **Drafter 模型**（均冻结、未微调）：
  - DSpark variants（Inferact 和 RadixArk，分别基于 MLA/GQA）
  - EAGLE-3 / EAGLE-3.1 heads

#### 推理设置

- **Speculative 参数**：$K=3$，即每次 draft 3 个 token。
- **窗口大小**：$W=4096$（除非消融实验中变化）。
- **批大小**：默认 batch=1，部分实验测试 batched serving。
- **平台**：基于 **vLLM** 实现，所有机制以补丁形式集成。

#### 评估指标

| 指标 | 含义 |
|------|------|
| **Speedup (S)** | 相比纯 target-only 推理的加速比 |
| **MAL (Mean Accepted Length)** | 每轮平均接受的 draft token 数（含 bonus token），理想值为 4 |
| **Retention** | 在 2M 上下文下的 speedup 占 8k 下 speedup 的比例，衡量“平坦性” |
| **Truncation Gap ($A$)** | full-context 与仅最后 $W$ token 的 target 分布差异（TV 距离） |
| **Fidelity Gap ($\delta$)** | 实际 drafter 输出与截断版 target 的输出差距 |

---

### 基线方法对比

论文采用 **内部对照设计**，同一模型栈下比较三种变体：

| 方法 | 是否窗口化 | 是否 re-base 索引 |
|------|------------|----------------|
| **Unwindowed** | ❌ 否 | ❌ 否 |
| **Windowed only** | ✅ 是 | ❌ 否 |
| **Ours (windowed + re-based)** | ✅ 是 | ✅ 是 |

> 📌 注意：不与其他外部系统横向对比，因多数方法在“冻结短上下文 drafter + 百万 token 上下文”这一设定下不可用。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1 & Figure 1）

在 **Kimi-K3 + PG19** 设置下，上下文从 8k 扩展到 **2 million tokens**：

| Drafter | 方法 | Speedup @8k | Speedup @2M | Retention (%) |
|--------|------|-------------|------------|--------------|
| **Inferact** | Unwindowed | 2.51× | 0.56× | 22% |
| | Windowed only | 2.41× | 0.88× | 36% |
| | **Ours** | **2.51×** | **2.25×** | **90%** |
| **RadixArk** | Unwindowed | 1.75× | 0.14× | 8% |
| | Windowed only | 1.91× | 0.87× | 45% |
| | **Ours** | **1.75×** | **2.08×** | **119%** |

> 🔍 **关键观察**：
> - 传统方法在 2M 上完全失效（speedup << 1）。
> - 仅窗口化可缓解成本问题，但在超过 $P_{\text{max}}=1M$ 后仍崩溃。
> - **完整方法（窗口 + 重映射）实现了近乎平坦的性能保持**。

---

### 与其他模型/数据集的泛化性验证

#### （1）Qwen2.5-7B-Instruct + EAGLE-3（Table 2）

- 上下文：2k → 512k
- 结果：windowed 方法在 512k 保留 **122%** 的 2k 加速比，而 unwindowed 仅保留 91%。

#### （2）MiniMax-M3 + EAGLE-3.1（Table 5）

- 特点：target 自身为 block-sparse，不会随上下文显著变慢。
- 结果：windowed 方法在 1M 上保留 **111%** 的 4k 加速比，证明优势来自 drafting 效率本身，而非 baseline 崩溃。

#### （3）跨数据集泛化（Table 7）

在 **LongBench、BEAM、SPEED-Bench** 上重复实验：

- 所有达到 1M 上下文的任务中，**speedup retention 均在 0.93–1.30 倍之间**。
- 表明“平坦性”是稳健现象，不依赖特定文本类型。

---

### 消融实验结果

#### （1）窗口大小 $W$ 的影响（Table 3 & E.7）

- 在 Qwen 对应设置下，$W$ 从 1024 到 4096，MAL 几乎不变（~1.67）。
- 在 Kimi-K3 @1M 下，$W=1024, 2048, 4096$ 的 MAL 分别为 2.97, 2.94, 3.03。
- **结论**：一旦 $W$ 足够大（如 ≥1k），acceptance 达到饱和，无需随 $N$ 增加。

#### （2）sink tokens 的影响（Table 10）

- 添加 sink tokens（如前几个固定 token）反而 **显著降低 performance**（MAL 从 3.26 降至 ~2.1）。
- 原因推测：长上下文中，sink 位于绝对位置 0，与当前百万级位置形成巨大 offset，超出训练分布。

#### （3）gap 测量（Table 12）

- **Truncation Gap $A$**：随 $W$ 增大而减小（从 0.22 到 0.04 @W=16384）。
- **Fidelity Gap $\delta$**：稳定在 ~0.68，说明 drafter 本身误差较大，且不随 $W$ 改善。
- **结论**：acceptance 主要受限于 drafter 自身质量（fidelity-limited），而非窗口大小（window-limited）。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **长上下文 drafting 不需要长上下文 drafter**  
   冻结的短上下文 drafter 可通过 **windowing + index re-basing** 有效扩展至 **2 million tokens**。

2. ✅ **性能保持“平坦”**  
   无论 drafter 本身质量如何，其在短上下文下的 speedup 增益可在极长上下文中 **几乎完全保留**（retention ≥90%）。

3. ✅ **两种机制缺一不可**  
   - 仅 windowing：解决成本问题，但无法处理 RoPE 越界。
   - 仅 re-basing：无法控制 drafter 的 attention 成本。
   - 必须两者结合才能实现稳定高性能。

4. ✅ **target-conditioned 结构是关键**  
   drafter 通过接收 target 的 hidden states 获取远距离信息，因此局部 attention 窗口已足够。

5. ✅ **方法具有强泛化性**  
   在不同 target-drafter 组合、多种数据集上均验证有效。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **依赖 target-conditioned drafter** | 不适用于 standalone drafter（如 N-gram、retrieval-based） |
| **RoPE 假设** | 理论分析基于 fixed-frequency RoPE 和 translation-invariant mask，对其他 PE（如 ALiBi、learned pos emb）未覆盖 |
| **数值不确定性** | 实际运行中存在 fp16/bf16 数值误差，可能导致输出轻微偏差（非功能错误） |
| **batched serving 性能增益减弱** | 在 batch 较大时，内存带宽红利已被 baseline 利用，speculative 增益相对缩小 |

---

### 未来工作方向

1. **扩展至其他位置编码机制**  
   将 index re-basing 思想推广到 ALiBi、YaRN、NTK-by-parts 等动态外推方法。

2. **更高效的边界处理**  
   当前 re-basing 在 band boundary 处可能引入扰动，可探索 smooth wrapping 或 interpolation 策略。

3. **与 verification-side 方法结合**  
   本文保持 verification 完整，未来可与 SpecPV、TriForce 等稀疏 verification 方法组合，进一步提升效率。

4. **动态窗口调度**  
   当前 $W$ 固定，未来可根据文本局部性自动调整窗口大小。

5. **端到端 lossless decoding 验证**  
   当前虽无输出差异报告，但仍需更严格验证在大规模、多样化输入下的输出一致性。

---

> 💡 **一句话总结**：  
> 本文颠覆了“长上下文需长 drafter”的直觉，证明只需两个简单 inference-time 机制（**windowing + index re-basing**），即可让冻结的短上下文 drafter 在百万 token 场景下依然高效工作，为低成本、高兼容性的 speculative decoding 提供了一条全新路径。

</details>

---

### 9. [Sparse Federated Echo State Networks for Time-Series Classification via Smooth-L1 Regularization](https://arxiv.org/abs/2610.21742)

**Authors**: Gengcheng Lyu, Manuel R\"oder, Frank-Michael Schleif  
**Category**: cs.LG  
**Published**: 2026-10-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.21742v1  

#### Abstract
Echo State Networks (ESNs), as lightweight implementations of reservoir computing, provide efficient solutions for temporal data processing in resource-constrained environments. While standard ESNs offer computational efficiency, their dense readout layers inflate the stored model and the per-sample...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sparse Federated Echo State Networks for Time-Series Classification via Smooth-L1 Regularization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
标准的 **Echo State Network (ESN)** 虽然训练高效，但其 **dense readout layer** 导致模型存储开销大、推理成本高，尤其在资源受限的边缘设备（如 IoT）和联邦学习（Federated Learning, FL）场景中成为瓶颈。此外，传统的 **L1 正则化**虽然能促进稀疏性，但由于其在零点不可导，难以与基于梯度的优化（尤其是二阶优化）兼容，限制了其在分布式框架中的应用。

### 提出的新方法
本文提出了一种新的 ESN 架构：**Smooth-L1 Regularized ESN (SL1ESN)**，并扩展至联邦学习场景下的 **FedSL1ESN**。核心创新如下：

- **Smooth-L1 正则化**：采用一个**二次可微**的 Smooth-L1 函数近似传统 L1 范数，保留了稀疏诱导能力的同时，使目标函数保持光滑，从而支持高效的**二阶优化（如牛顿法）**。
- **中央化与联邦学习统一框架**：该方法既适用于集中式训练，也自然适配于联邦学习，提出了基于**局部梯度与 Gauss-Newton Hessian 聚合**的二阶联邦更新机制（FedSL1ESN），服务器执行带阻尼的牛顿步长。
- **固定阈值全局适用**：所有实验均使用**单一固定的软阈值（T = 0.1）**进行权重剪枝，无需针对每个数据集调参，增强了方法的实用性和鲁棒性。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **稀疏性与效率** | 实现 75–95% 的 readout 权重稀疏率，显著压缩模型大小、降低推理计算量和服务器广播负载。 |
| **优化兼容性** | Smooth-L1 支持二阶优化，收敛速度快，在联邦设置下仅需少数几轮通信即可达到接近最优的准确率。 |
| **联邦集成性** | 与 FedAvg 等一阶方法不同，本方法允许客户端上传 Hessian 信息，实现更精准的全局更新，且聚合过程天然支持安全聚合（secure aggregation）。 |
| **泛化性** | 在多个时间序列分类任务上表现稳定，稀疏读出层对多数任务几乎无精度损失。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在 **7 个公开的时间序列分类基准**上进行，涵盖活动识别、手写轨迹、生理信号等场景：

| 数据集 | 类别数 | 序列长度 | 类型 |
|--------|--------|----------|------|
| UCI HAR | 6 | 128 | 多变量 |
| Character Trajectories | 20 | 205 | 多变量 |
| JapaneseVowels | 9 | 29 | 多变量 |
| ECG5000 | 5 | 140 | 单变量 |
| DistalPhalanxOutlineCorrect (Distal.) | 2 | 80 | 单变量 |
| Yoga | 2 | 426 | 单变量 |
| Strawberry | 2 | 235 | 单变量 |

数据来自 **UCR 和 UCI 时序档案**，划分遵循原始协议。

### 实验设置与评估指标
- **模型结构**：固定 reservoir size $ N_R = 500 $，连接密度 $ d = 0.1 $，输入缩放、谱半径、泄漏率等通过随机搜索调优。
- **训练方式**：
  - **集中式**：使用 dampened Newton 法优化 Smooth-L1 损失，配合 continuation schedule（平滑参数 $\alpha$ 逐步增大）。
  - **联邦式**：5 个客户端，IID 分割，每轮客户端计算本地梯度 $ g_k $ 和 Gauss-Newton Hessian $ H_k $，服务器聚合后执行牛顿步长并广播稀疏读出。
- **评估指标**：
  - **测试准确率（Accuracy %）**
  - **读出层稀疏率（Sparsity %）**
  - **通信轮次 vs 准确率/稀疏率曲线**

### 基线方法对比
- **none**：无正则化的交叉熵损失（即标准 ESN）
- **L2-ESN**：岭回归形式的 L2 正则化
- **FedSL1ESN**：本文提出的联邦稀疏 ESN，与上述两种 baseline 在相同 Newton 框架下比较，确保公平。

> 注：未包含 FedAvg 作为独立列，因其为一阶方法，无法生成 Hessian，且作者指出其在凸问题下只会更慢地逼近同一解。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（集中式）

| Dataset | Best Acc (%) | SL1ESN Acc (%) | SL1ESN Sparsity (%) |
|--------|---------------|----------------|---------------------|
| UCI HAR | 88.01 (L2) | **88.05** | 77.6 |
| Character | 99.58 (none) | 98.97 | 92.7 |
| JapaneseVowels | 98.24 (none) | 97.68 | 94.6 |
| ECG5000 | 92.37 (L2) | 91.91 | 93.3 |
| Distal. | 73.55 (none) | 73.51 | 89.6 |
| Yoga | 79.53 (none) | 73.70 | 77.4 |
| Strawberry | 96.11 (none) | 90.38 | 88.6 |

- **五项任务**中，SL1ESN 的准确率与最强 dense baseline 差距 **≤ 0.7%**，且稀疏率达 **77–95%**。
- Yoga 和 Strawberry 上有约 5–6% 的差距，但消融实验证明这主要源于 Smooth-L1 拟合能力略弱，而非剪枝所致。

### 联邦学习结果（FedSL1ESN）

| Dataset | Best Acc (%) | FedSL1ESN Acc (%) | Sparsity (%) |
|--------|---------------|--------------------|--------------|
| UCI HAR | 89.41 (none) | 87.86 | 74.9 |
| Character | 98.74 (L2) | 98.20 | 84.8 |
| JapaneseVowels | 96.19 (none) | 94.84 | 95.2 |
| ECG5000 | 91.38 (L2) | 91.30 | 91.6 |
| Distal. | 73.77 (ours) | **73.77** | 86.7 |
| Yoga | 73.45 (L2) | 69.00 | 80.8 |
| Strawberry | 91.32 (L2) | 87.76 | 89.5 |

- 在 **五个任务**中，FedSL1ESN 与最佳 baseline 差距 **≤ 1.6%**，稀疏率仍达 **75–95%**。
- **Distal.** 上甚至成为最准确模型。
- **收敛极快**：多数任务在 **1–2 轮通信内**达到接近峰值准确率（见 Figure 3a）。

### 消融实验结果
- **阈值 $ T $ 扫描**（Figure 2 & 4）：在大多数数据集上，准确率在稀疏率达到 80–95% 前基本不变，之后才急剧下降，说明 readout 存在大量冗余。
- **正则化强度 $ \lambda $ vs 阈值 $ T $**（Figure 5）：
  - $ T $ 控制稀疏性更“干净”，几乎不牺牲准确率；
  - $ \lambda $ 过大会导致准确率明显下降，稀疏性饱和。
- **连续性调度因子 $ K $**：对准确率影响小，无需调参。
- **客户端数量敏感性**（Figure 8）：从 $ K=2 $ 到 $ K=50 $，准确率和稀疏率均保持稳定，表明方法对客户端数量变化鲁棒。

---

## 4. 关键结论和发现

### 主要发现
1. **ESN 读出层高度冗余**：在时间序列分类任务中，绝大多数 reservoir 到输出的连接是冗余的，可通过稀疏化大幅压缩而不损精度。
2. **Smooth-L1 是有效的稀疏化路径**：通过光滑近似实现端到端可导优化，结合软阈值剪枝，可在保持高精度的同时获得高稀疏性。
3. **联邦二阶优化可行且高效**：FedSL1ESN 成功实现了基于 Hessian 聚合的牛顿式联邦更新，**收敛速度远超一阶方法**，仅需少数几轮即可完成训练。
4. **稀疏性带来多重收益**：
   - 模型存储减少 4–20 倍（取决于稀疏率）
   - 推理计算量（sparse mat-vec）成比例下降
   - 服务器广播负载显著降低（CSR 格式传输）

### 局限性
1. **非 IID 数据未验证**：所有联邦实验基于 IID 数据划分，未考虑现实中的统计异质性（client drift），可能导致 Hessian 聚合偏差。
2. **支持集不可解释**：由于 reservoir 特征强相关，稀疏支持集不稳定，本文强调**模型压缩**而非特征选择。
3. **上传开销较大**：客户端上传的是 Hessian 而非模型本身，通信量远大于 FedAvg（约 250 倍），虽轮次少但单轮代价高。
4. **隐私暴露风险更高**：Hessian 包含更多关于本地数据分布的信息，相比梯度更易泄露隐私，需依赖 secure aggregation 或 DP 保护。

### 未来工作方向
- 将方法扩展至 **non-IID 场景**，结合已有的 Newton-type FL 方法（如 FedNL、FedNew）处理异构数据。
- 探索 **低秩或 Sketching 技术**压缩 Hessian 上传，提升通信效率。
- 研究 **自适应正则化策略**，动态调整 $ \lambda $ 或 $ T $ 以进一步优化稀疏-精度权衡。
- 将框架推广至其他 **Reservoir Computing 架构**（如 Liquid State Machines、Deep ESNs）。
- 加强隐私保护机制，集成 **差分隐私（DP）或同态加密**。

---

> ✅ **总结一句话**：  
> 本文提出的 **SL1ESN / FedSL1ESN** 通过引入 **Smooth-L1 正则化**，实现了在保持高精度的同时，将 ESN 读出层压缩至 **75–95% 稀疏率**，并在联邦学习中利用**二阶优化实现快速收敛**，为资源受限环境下的高效时序建模提供了实用解决方案。

</details>

---

### 10. [ABNAT: Attention-Based Noise-Aware Training for Robust Transformers on Analog Compute-in-Memory Systems](https://arxiv.org/abs/2610.22065)

**Authors**: Wenshuai Yao, Wenyong Zhou, Hanyong Shao, Zhiyuan Ning, Yizhe Chen, Yuannuo Feng, Kechao Tang, Ru Huang  
**Category**: cs.LG  
**Published**: 2026-10-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.22065v1  

#### Abstract
Transformer-based models have become the backbone of a wide range of tasks, but their intensive computation and data movement incur substantial energy overhead. Analog compute-in-memory (CIM) offers an energy-efficient alternative, yet its inherent hardware non-idealities perturb stored weights and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ABNAT: Attention-Based Noise-Aware Training for Robust Transformers on Analog Compute-in-Memory Systems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **Transformer 在 Analog Compute-in-Memory (CIM) 系统上的鲁棒性问题**  
  Analog CIM 虽然能高效执行矩阵运算（MVM），但其硬件非理想性（如设备变异、写噪声、读噪声等）会扰动存储的权重，导致模型推理性能严重下降。
- **现有 Noise-Aware Training (NAT) 方法的局限性**  
  当前的 NAT 方法仅关注输出层的预测准确性（output-centric），忽略了 Transformer 中 **attention 机制** 的内部语义路径稳定性。这可能导致模型在噪声下通过错误的 attention 路由得出“正确”答案，缺乏真正的鲁棒性。

### 🚀 提出的新方法：ABNAT
提出 **ABNAT (Attention-Based Noise-Aware Training)**，一种基于 **attention distillation** 的噪声感知训练框架：
- **核心思想**：不仅要求 noisy student 模型输出正确的预测，更要求其 **保留 clean teacher 模型的 headwise attention 分布**。
- **无需额外教师模型**：teacher 是同一预训练 checkpoint 的 clean 副本，student 是注入 CIM 噪声的版本，两者共享结构。
- **多层级监督目标**：
  - `L_task`：标准交叉熵损失（任务级）
  - `L_logits`：logit 层蒸馏（输出分布对齐）
  - `L_attn`：attention map 对齐（内部路由保护）

### 🔍 相比现有方法的优势
| 方法 | 是否关注 attention 路由 | 是否提升鲁棒性 | 推理开销 |
|------|------------------------|----------------|----------|
| NAT [19] | ❌ | 有限 | 无 |
| NAT+KD [5] | ❌（仅输出蒸馏） | 中等 | 无 |
| FeatureAlign [27] | ❌（中间特征对齐） | 弱 | 高 |
| **ABNAT (本文)** | ✅（headwise attention 对齐） | **显著提升** | **无** |

- **优势总结**：
  - 显著降低 attention drift（最多减少 85.77%）
  - 提升噪声下的准确率（最高 +11.28%）
  - 无推理时开销（training-only 技术）
  - 支持 selective alignment 变体以降低训练成本

---

## 2. 核心实验方法和设置

### 📊 使用的数据集与模型
| 类别 | 模型 | 数据集 |
|------|------|--------|
| **Vision** | ViT-Tiny/16, ViT-Small, Swin-Tiny | CIFAR-100, ImageNet-100 |
| **Language** | Pythia-160M, Pythia-2.8B | WikiText-103 |
| **Multimodal** | CLIP ViT-B/32 | Flickr30K |

### ⚙️ 实验设置
- **噪声建模**：
  - 使用 bit-level hardware variation 模拟 CIM 权重扰动
  - 噪声强度 $\sigma \in \{0.05, 0.10, 0.15, 0.20\}$
  - 匹配训练与部署噪声（matched train-deploy noise）
- **量化配置**：W8A8
- **训练参数**：
  - Optimizer: AdamW
  - Batch Size: 1024
  - Epochs: 100（除非特定任务）
  - 学习率：$1\times10^{-3}$，cosine decay，5 epoch warmup
- **评估方式**：
  - 报告 30 次 inference seed 下的平均值 ± 标准差
  - 主要指标：accuracy / perplexity / Mean Recall

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **Original + Noise** | 未经 NAT 的原始模型加噪声 |
| **NAT [19]** | 标准噪声感知训练 |
| **NAT+KD [5]** | 输出层知识蒸馏增强的 NAT |
| **FeatureAlign [27]** | 中间特征图对齐 |
| **ABNAT (Ours)** | Headwise attention map 对齐 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（ViT-Tiny/CIFAR-100）

| 方法 | $\sigma=0.05$ | $\sigma=0.10$ | $\sigma=0.15$ | $\sigma=0.20$ | **Avg** |
|------|---------------|---------------|---------------|---------------|--------|
| NAT [19] | 84.98 | 79.23 | 71.52 | 58.86 | 73.65 |
| NAT+KD [5] | 87.20 | 82.69 | 74.99 | 62.45 | 76.83 |
| **ABNAT (full)** | **87.62** | **83.95** | **78.66** | **70.14** | **80.09** |

- **最大增益**：在 $\sigma=0.20$ 下，相比 NAT 提升 **11.28% 绝对准确率**
- **attention drift 减少**：相比 NAT 减少 **85.77%**（从 1.272 → 0.181 KL 散度）

### 🔁 泛化能力验证
#### 跨模型与跨任务表现一致提升：
| 场景 | $\sigma=0.20$ 下 ABNAT 提升 |
|------|----------------------------|
| ViT-Small @ CIFAR-100 | +6.34 pp |
| ViT-Tiny @ ImageNet-100 | +8.15 pp |
| Swin-Tiny @ CIFAR-100 | +4.15 pp |
| Pythia-160M (perplexity ↓) | 7682 → 1813 (**↓76.4%**) |
| Pythia-2.8B (perplexity ↓) | 2330 → 941 (**↓59.6%**) |
| CLIP @ Flickr30K (Mean Recall ↑) | 71.70 → 72.27 |

#### 训练-部署噪声不匹配场景（robust generalization）
- 固定 $ \sigma_{\text{train}} = 0.15 $ 或 $ 0.20 $，测试不同 $ \sigma_{\text{infer}} $
- ABNAT 在所有噪声水平下均优于 NAT 和 NAT+KD
- 即使在 clean inference 下，ABNAT 仍保持更高准确率（79.29% vs. 69.59% for NAT）

### 🔍 消融实验（Ablation Study）

#### 表 1：组件消融（ViT-Tiny/CIFAR-100）
| 变体 | $\sigma=0.20$ Accuracy | Avg |
|------|------------------------|-----|
| NAT | 58.86 | 73.65 |
| NAT+KD | 62.45 | 76.83 |
| NAT+FeatureAlign | 58.35 | 74.60 |
| **NAT+AttnAlign (ABNAT)** | **70.14** | **78.39** |

- **FeatureAlign** 对高噪声无效，甚至有害
- **Attention Alignment** 贡献最大，尤其在高噪声下

#### 选择性对齐变体（Selective Alignment）
| 对齐策略 | 平均准确率 | 训练时间开销 vs NAT | 峰值内存开销 |
|---------|------------|--------------------|--------------|
| Full (all blocks) | 80.09% | 82.9% | 54.4% |
| **Last4 blocks** | **79.68%** | **40.8%** | **23.3%** |
| First4 blocks | 77.01% | — | — |

- **Late-block alignment 更有效**：因 attention drift 随深度累积
- **Last4** 达到接近全对齐性能，但训练开销减半

#### 超参数敏感性分析（Table 2）
- 在合理范围内（$\lambda_{\text{logits}} \in [3,12], \lambda_{\text{attn}} \in [25,100]$），性能稳定（79.7–80.2%）
- 不需要精细调参

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Attention drift 是 CIM 噪声下性能退化的关键瓶颈**  
   - 注意力漂移随网络深度加剧，集中在 late blocks
   - attention drift 与 accuracy degradation 高度相关（Pearson r=0.9633）

2. **传统 NAT 存在“自信-路由不一致”问题**  
   - NAT 模型可能高度自信地做出正确预测，但其 attention 路由已严重偏离 clean 模型
   - ABNAT 通过 attention distillation 显著改善这一问题

3. **ABNAT 实现了更忠实的内容理解保留**  
   - 在相同输入上，ABNAT 的 attention map 更接近 clean teacher
   - 即使在噪声不匹配或 clean 推理下也表现出更强泛化能力

4. **Selective alignment 可大幅降低训练成本**  
   - 仅对 late blocks 或 high-sensitivity heads 进行 alignment，即可保留 >99% 的鲁棒性增益

### ⚠️ 方法的局限性
- **依赖 clean teacher 拷贝**：需维护一个 clean 模型副本用于监督，增加训练内存占用
- **alignment 温度敏感**：过低的 $T_{\text{attn}}$ 导致训练不稳定（如 $T=0.125$ 时性能骤降）
- 目前未探索动态 alignment 策略（如每轮更新 sensitive heads）

### 🔮 未来工作方向
- 将 ABNAT 扩展至更多架构（如 Mamba、RetNet）
- 探索 **online sensitivity estimation** 以实现自适应 alignment
- 结合硬件反馈进行 co-design，进一步优化 noise resilience
- 应用于更大规模的 multimodal 和 generative models

---

> **一句话总结**：  
> ABNAT 通过引入 **attention-level 知识蒸馏**，首次系统性解决了 Analog CIM 上 Transformer 的 **attention drift** 问题，在几乎不增加推理开销的前提下，实现了高达 **11.28% 的准确率提升** 和 **85.77% 的 attention 偏差降低**，为鲁棒 AI 部署提供了新范式。

</details>

---

### 11. [Translation Circuit Discovery and Targeted Optimization for Low-Resource Multilingual Translation](https://arxiv.org/abs/2610.21456)

**Authors**: Atai Waris Khan, Abdullah Mazhar, Raghu Kumar, Md Shad Akhtar  
**Category**: cs.CL  
**Published**: 2026-10-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.21456v1  

#### Abstract
Recent multilingual large language models achieve strong translation performance, yet the internal mechanisms responsible for low-resource multilingual translation remain insufficiently understood. In this paper, we study translation circuits in multilingual decoder-only LLMs for Hindi-to-Urdu and H...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Translation Circuit Discovery and Targeted Optimization for Low-Resource Multilingual Translation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前多语言大模型（Multilingual LLMs）在低资源语言对上的翻译机制仍不明确，尤其是其内部负责翻译行为的**可解释性机制**（mechanistic interpretability）缺乏系统研究。传统全参数微调（full-parameter fine-tuning）效率低下且无法识别真正影响翻译的关键组件。

本文聚焦于两个低资源南亚语言对：
- **Hindi→Urdu**（强语法与词汇重叠，但脚本不同）
- **Hindi→Sindhi (Devanagari)**（共享脚本，但在音系和词汇上差异显著）

旨在揭示这些语言对中翻译行为背后的**内在电路机制**（translation circuits），并提出一种高效优化策略。

---

### 🚀 提出的新方法：TITAN 框架

作者提出 **TITAN**（Translation-aware Interpretability through Targeted Activation Networks），一个结合**机械可解释性**与**目标优化**的框架，其核心流程如下：

1. **CIRCE**（Causal Intervention for Representation Circuit Engineering）  
   - 使用因果干预（causal intervention）识别对翻译有因果影响的稀疏组件（如 attention heads 和 MLP blocks）。
   - 构造对比样本（contrastive prompts）分离翻译相关激活与通用语言建模行为。

2. **Sparse Feature Decomposition**  
   - 在识别出的关键层上训练 **Sparse Autoencoders (SAEs)**，解耦特征表示，提取稀疏、可解释的翻译特征。

3. **Targeted Preference Optimization**  
   - 仅对识别出的“翻译关键电路”进行 **Supervised Fine-Tuning (SFT)** 和 **Direct Preference Optimization (DPO)**，冻结其余参数。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | TITAN |
|------|--------|-------|
| 参数效率 | 需要更新全部参数（>100%） | 更新 **<5%** 的参数 |
| 可解释性 | 黑箱微调，难以定位关键组件 | 明确识别因果性翻译电路 |
| 性能提升 | 有限，易过拟合 | 显著 BLEU 提升（+33.75% ~ +48.98%） |
| 泛化能力 | 多依赖英语中心化（English-centric）机制 | 发现基于**共享语义子空间**的非英语中心机制 |

> ✅ **核心优势**：TITAN 实现了**参数高效 + 可解释 + 高性能**的统一，突破了传统微调的“全量更新”范式。

---

## 2. 核心实验方法和设置

### 📚 数据集

使用 **Bharat Parallel Corpus Collection (BPCC)** 中的两个子集：

| 语言对 | 训练句数 | 特点 |
|--------|---------|------|
| **Hindi→Urdu** | 60K 平行句对 | 脚本不同（Devanagari ↔ Nastaliq），语法高度相似 |
| **Hindi→Sindhi (Devanagari)** | 35K 平行句对 | 共享脚本，但含 **implosive consonants** 和波斯-阿拉伯借词 |

数据划分为：
- **SFT Train**: 用于监督微调
- **DPO Train**: 用于偏好优化
- **Contrastive Subset (10K)**: 用于因果分析

---

### ⚙️ 实验设置

- **基础模型**：`Gemma-2 9B`
- **微调方式**：
  - 所有 baseline 使用 LoRA（rank=16）
  - TITAN/CIRCE 仅更新识别出的稀疏电路
- **训练配置**：
  - SFT: 3 轮，batch size=8
  - DPO: 2 轮
  - SAE latent dim: 14336
- **硬件**：NVIDIA A100 40GB GPU

---

### 📊 评估指标

| 指标 | 含义 |
|------|------|
| **BLEU** | 标准机器翻译评分 |
| **chrF2 / chrF++** | 字符级 F-score，更关注形态准确性 |
| **TER** | 编辑距离，越低越好 |
| **COMET** | 基于预训练模型的语义相似度评分 |
| **Human Evaluation** | 由母语专家评估 **adequacy**（充分性）和 **fluency**（流畅性），评分 ≥4 视为高质量 |

---

### 🆚 基线方法对比

| 类型 | 模型 |
|------|------|
| 多语言 NMT 模型 | `NLLB-200`, `IndicTrans2` |
| 解码器-only LLMs | `Gemma-2`, `LLaMA-3.1`, `Qwen-3.5`, `Mistral-3` |
| 微调策略 | Zero-shot, SFT, DPO, LoRA-SFT+DPO |

> TITAN 与所有上述模型在相同设置下比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2）

#### Hindi→Urdu
| 方法 | BLEU (H→U) | Δ vs Baseline |
|------|------------|----------------|
| Gemma-2 (DPO) | 25.78 | — |
| **TITAN** | **31.82** | **+23.4%** (+6.04) |

> 注：摘要中称 +22.67%，正文中为 +33.75%（相对提升），此处以绝对差值为准。

#### Hindi→Sindhi
| 方法 | BLEU (H→S) | Δ vs Baseline |
|------|------------|----------------|
| Gemma-2 (DPO) | 23.25 | — |
| **TITAN** | **33.44** | **+43.8%** (+10.19) |

> 摘要中报告 **+45.86%** 相对提升，是目前最高的低资源翻译增益之一。

---

### 🆚 与其他方法对比

- TITAN 在 **所有指标**（BLEU, chrF2, COMET）上均显著优于所有 baseline。
- 即使与参数量相同的 **LoRA-SFT+DPO**（3.14% 可训练参数）相比：
  - LoRA-SFT+DPO: **26.37 BLEU**
  - TITAN: **31.82 BLEU**（+5.45 BLEU）
  > 表明性能增益来自**电路选择机制**，而非参数数量。

---

### 🔬 消融实验（Ablation Study）

#### ✅ CIRCE 的有效性验证（Progressive Knockout）
- 逐步移除 CIRCE 识别出的关键组件（attention heads / MLPs）：
  - **CIRCE 组件移除** → BLEU 从 25.53 降至 16.54（↓35.2%）
  - **随机组件移除** → BLEU 几乎不变（↓0.45%）
- 表明识别出的电路具有**强因果必要性**。

#### ✅ DPO 的增量贡献
- CIRCE（仅 SFT）已优于多数 baseline。
- 加入 DPO 后进一步提升，说明**分阶段优化有效**。

---

## 4. 关键结论和发现

### 🎯 主要发现

1. ✅ **翻译电路存在于稀疏组件中**  
   - 翻译行为由少量 attention heads 和 MLP blocks 因果驱动，分布于中高层网络。

2. ✅ **非英语中心化机制（Non-English-Centric Pivot）**  
   - 分析显示模型内部对 **English 的表征对齐始终为负**。
   - 翻译通过 **共享的印度语族语义子空间**（shared Indic semantic subspace）进行，而非经由英语中转。

3. ✅ **标签感知路由机制（Label-Aware Routing）**  
   - 关键 attention heads 更多地关注 `[Hindi]`, `[Urdu]` 等语言标识符，而非源文本内容。
   - 这是因为 Indic 语言间词汇重叠高，语言标签提供了更强的区分信号。

4. ✅ **MLP 层实现语义转换，注意力层实现路由控制**  
   - 注意力头负责：结构锚定、语言识别、方向路由。
   - MLP 层负责：语义编码与目标语言生成，尤其在后期才出现目标语言对齐。

5. ✅ **稀疏特征 ≠ 高因果影响力**  
   - 高 selectivity 的 SAE 特征未必有高 causal lift。
   - 真正驱动翻译的是少数 **high-lift features**，分布在源/目标语言特征之间。

---

### ⚠️ 方法的局限性

1. **计算成本高**：因果干预与 SAE 训练在电路发现阶段开销大，不适合快速部署。
2. **Unicode 一致性问题**：Sindhi 输出偶现未正确渲染的 **implosive phoneme** 字符。
3. **长句处理弱**：复杂嵌套句子流畅性下降，尽管充分性保持较好。
4. **贷款词保留**：部分技术术语中的 Hindi loanwords 未被翻译。
5. **泛化边界待验证**：虽在 English→French 和 English→Irish 上表现良好，但仍需更多语言家族测试。

---

### 🔮 未来工作方向

1. 将 TITAN 框架扩展至 **更多低资源语言对**，特别是跨语系组合。
2. 探索 **自动化电路发现 pipeline**，降低人工设计对比提示的成本。
3. 结合 **factuality** 和 **cultural appropriateness** 的评估维度，提升翻译质量全面性。
4. 研究如何将发现的电路 **迁移至其他模型架构**（如 encoder-decoder）。
5. 引入 **human-in-the-loop** 机制，动态修正错误电路连接。

---

## ✅ 总结

TITAN 是首个将 **mechanistic interpretability** 与 **parameter-efficient fine-tuning** 深度融合的多语言翻译框架。它不仅实现了在更新 **<5% 参数**的情况下大幅提升翻译质量（**+33.75% ~ +48.98% BLEU**），更重要的是揭示了低资源 Indic 语言翻译的内在机制——**基于共享语义子空间的稀疏路由电路**，挑战了传统的“英语中转”假设。该工作为构建**可解释、高效、高性能**的多语言系统提供了新范式。

</details>

---

### 12. [MAI-Gym: An Open All-in-One Platform for GUI Agent Evaluation and RL Training across Mobile and Computer Environments](https://arxiv.org/abs/2610.20939)

**Authors**: Panrong Tong, Quyu Kong, Pengxiang Li, Tianyu Xia, Hanzhang Zhou, Lei Wang, Chenglin Cai, Long Li, Long Chen, Jianan Zhang, Zhengwen Zeng, Xu Zhang, Gongjie Zhang, Yue Wang, Steven Hoi  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.20939v1  

#### Abstract
Online reinforcement learning (RL) with verifiable rewards has driven rapid progress in mathematics, coding, and tool use, but scaling this success to GUI agents remains challenging due to the high cost of interactive environments and the scarcity of realistic training tasks. We introduce MAI-Gym, a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MAI-Gym 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 GUI Agent 研究面临两大挑战：
- **环境成本高、难以规模化**：GUI 环境（如手机、电脑）无法像代码沙箱一样共享实例，每个任务需要独立的设备状态，导致训练和评估成本高昂。
- **缺乏多样化且可重置的真实任务**：真实世界中的 GUI 工作流通常涉及多个应用之间的信息流转（如邮件+附件+日历），而现有基准多为孤立任务，缺乏跨应用连贯性。

此外，不同平台（Mobile/Web/Desktop）的评估与训练系统割裂，容易因 `agent harness` 不一致导致性能不可复现。

### 提出的新方法与思路
本文提出 **MAI-Gym** —— 一个开源的一体化平台，统一支持跨移动、计算机和网页环境的 GUI Agent 评估、数据收集与在线 RL 训练。

#### 核心创新点包括：

| 创新维度 | 具体实现 |
|--------|---------|
| **统一架构设计** | 所有任务共享一条 `rollout path` 和相同的 `agent harness`，确保训练、评估、诊断行为完全一致，避免 harness drift 问题。 |
| **可扩展基础设施** | 支持高达 **20,000 并行环境实例**，通过异步启动、warm pool、对象存储传输截图等机制实现高效部署与恢复。 |
| **自动化环境与任务合成** | 引入“**mocked apps**”概念：<br>• **Real-world clones**：对银行、打车、购票等真实交易类 App 构建可重置副本，用于安全训练；<br>• **Novel apps**：由 coding agent 自动生成全新界面的应用，测试模型适应陌生 UI 的能力。<br>同时构建“**cross-app world**”，在多个 App 中生成共享人物、事件、文件的连贯初始状态，使任务需跨应用检索整合信息。 |
| **执行验证的任务生成流程** | 采用“先定义需求再生成验证器”的范式：<br>1. 明确任务指令、资产引用、成功标准；<br>2. 独立生成 golden code（直接设置完成态）和 verifier code（检查最终状态）；<br>3. 在真实环境中运行 `initial → golden → reset` 流程，验证 verifier 是否正确返回 0/1/0 分数。 |

### 相比现有方法的优势

| 对比维度 | MAI-Gym 优势 |
|--------|-------------|
| **覆盖范围** | 支持 **10 个基准**（3 移动 + 4 计算机 + 3 Web），远超同类系统（多数仅限单一平台）。 |
| **规模能力** | 最大支持 **20,000 并行环境**，显著高于 MobileGym (256)、SCALECUA (600) 等。 |
| **任务真实性** | 支持跨应用连贯世界（cross-app world），而非孤立输入文件或样本数据。 |
| **奖励可靠性** | 验证基于可执行代码（code-based verifier），而非易被欺骗的 VLM judge。 |
| **开放程度** | 完整开源框架、训练代码、49 个 mocked apps、2,000 个经执行验证的任务及模型 checkpoint。 |

---

## 2. 核心实验方法和设置

### 使用的数据集与基准
MAI-Gym 支持以下 **10 个公开基准**，涵盖三大领域：

| 领域 | 基准名称 | 任务数 | 步骤上限 | 主要指标 |
|------|--------|-------|--------|--------|
| **Mobile** | MobileWorld | 117 | 100 | Success Rate |
| | AndroidWorld | 116 | 动态 | Success Rate |
| | MobileGym | 256 | 100 | Success Rate / Partial Score |
| **Computer** | OSWorld-Verified | 361 | 100 | Success Rate |
| | OSWorld 2.0 | 108 | 500 | Binary / Partial Score |
| | MacAgentBench | 676 | 50 | Success Rate |
| | MyPCBench | 184 | 100 | Perfect-task Rate |
| **Web** | WebVoyager | 643 | 15 | Success Rate |
| | OnlineMind2Web | 123 | 50 | Success Rate |
| | WebArena | 812 | 30 | Success Rate |

> 注：其中 MobileWorld 和 OSWorld-Verified 被用作 **Task OOD**（训练未见任务）测试集；其余五个为 **Env OOD**（训练未见环境）测试集。

### 实验设置与评估指标
- **模型基础**：基于 `Qwen3.5-9B` 进行 Supervised Fine-Tuning (SFT) 后进行在线 RL 训练。
- **训练方式**：使用 **GRPO**（Group Relative Policy Optimization）算法，在 MobileWorld 和 OSWorld 合成任务上进行 **联合训练（joint RL）**。
- **训练资源**：使用 512 GPU，峰值并发 20,000 环境，训练约 96 小时。
- **评估方式**：
  - 使用 `mai-gym eval` 命令一键评估所有基准；
  - 每个任务运行 3 次（trials），报告 `success rate`；
  - 区分 Task-OOD 和 Env-OOD 性能，并计算平均得分（Cross Avg）。

### 基线方法对比
对比三类模型：
1. **前沿闭源模型**：Claude Opus 4.8、GPT-5.5、Qwen3.8-Max（API 接入）
2. **开源 GUI Agent 模型**：
   - UI-Venus-2-9B（同参数量）
   - SCALECUA-Qwen3.5-9B（同参数量）
   - GUI-Owl-1.5-8B / 32B
   - 更大规模模型如 Qwen-CUA (397B-A17B)
3. **消融基线**：Qwen3.5-9B（base）、SFT warm-up checkpoint

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 4）

| 模型 | MobileWorld (Task OOD) | OSWorld-Verified (Task OOD) | Env-OOD Avg |
|------|------------------------|------------------------------|------------|
| **MAI-Gym-9B (Ours)** | **78.4%** | **76.2%** | **34.9%** |
| Qwen3.5-9B (Base) | 18.0% | 41.8% | 18.8% |
| SFT Warm-up | ~70% | ~66% | ~32% |
| UI-Venus-2-9B | 75.2% | 70.8% | 37.3% |
| SCALECUA-Qwen3.5-9B | 12.8% | 68.7% | 16.0% |
| GUI-Owl-1.5-8B | 38.2% | 52.9% | 17.7% |
| GPT-5.5 | 62.4% | 78.7% | 45.5% |
| Qwen3.8-Max | 77.8% | 86.1% | 54.9% |

> ✅ MAI-Gym-9B 是当前 **<10B 参数中最强的开源 GUI Agent**

### 与基线方法的对比结果
- 在 **MobileWorld** 上超越所有同规模开源模型（+3.2% vs UI-Venus, +40.2% vs GUI-Owl-8B），甚至优于部分更大模型。
- 在 **OSWorld-Verified** 上达到 76.2%，仅次于 Qwen-CUA（397B-A17B）和 GPT-5.5。
- 在 **Env OOD** 场景下仍保持竞争力，尤其在 MacAgentBench 和 MyPCBench 上领先同类开源模型。

### 消融实验结果

#### （1）Joint RL vs. Expert Merging（表 7）
| 设置 | MobileWorld | OSWorld-Verified |
|------|-----------|----------------|
| Mobile Expert Only | 76.4% | 71.4% |
| Computer Expert Only | 73.5% | 73.8% |
| Merged (50/50) | 76.4% | 73.7% |
| **Joint RL (Ours)** | **78.4%** | **76.2%** |

👉 **结论**：联合训练优于分别训练后合并专家模型，说明跨域共享策略更有效。

#### （2）Scaling Synthesized Tasks（图 8）
随着合成训练任务数量从 0.5k 增加到 5k：
- MobileWorld 成功率从 69.2% → **78.4%**
- OSWorld-Verified 从 69.7% → **76.2%**

👉 **结论**：更多样化的合成任务显著提升泛化能力。

#### （3）Clone-to-Real Transfer（图 9 & 11）
在 **12306 购票 App 克隆体** 上进行 RL 训练后：
- 在克隆环境中成功率从 36.5% → **73.1%**
- 在 **真实物理手机上的真实 App** 中，成功率从 28.8% → **48.1%**（↑19.3pp）

👉 **结论**：在安全模拟环境中训练的能力可以有效迁移到真实世界应用。

#### （4）Learning Unfamiliar Interfaces（图 12）
在 **novel apps**（从未见过的全新界面）上：
- RL 前成功率：25.8%
- RL 后成功率：**45.2%**
- 平均步数从 33.3 ↓ 至 16.4，失败尝试减少（27 → 5）

👉 **结论**：即使面对全新 UI，仅靠 reward 反馈也能学会高效操作。

#### （5）Efficiency of Window-wise History（表 6）
相比 step-wise packing：
- Prefix-cache hit ↑ 从 42.5% → **85.9%**
- 训练 token 数 ↓ 84.7%
- 单次更新时间 ↓ 62.6%

👉 **结论**：窗口式历史打包大幅提升训练效率。

---

## 4. 关键结论和发现

### 主要发现
1. **一体化平台至关重要**：统一的 `rollout path` 和 `harness` 能消除训练与评估间的偏差，提高研究可复现性。
2. **合成环境是规模化训练的关键**：通过构建 **mocked apps** 和 **cross-app worlds**，可在低成本环境下生成大量多样化、可验证的真实任务。
3. **联合训练优于专家融合**：在一个 policy 中联合学习 mobile 和 computer 任务，效果优于分别训练后合并。
4. **sim-to-real 迁移可行**：在交易类 App 的克隆体中训练，能显著提升在真实 App 上的表现（28.8% → 48.1%）。
5. **模型能适应全新界面**：仅通过 reward 学习，即可掌握 novel apps 的操作逻辑，表明其具备较强泛化能力。

### 方法的局限性
- 当前 mocked apps 主要依赖 coding agent 手工编写，尚未完全自动化。
- 跨平台一致性虽强，但移动端目前仅支持 GUI actions，缺少 native CLI 支持。
- 真实 App 的 clone 仍可能存在细微差异，影响迁移效果（如图 11 中 clone 表现优于 real app）。
- 训练依赖大量计算资源（512 GPU + 20K 并发），中小团队难以复现。

### 未来工作方向
- 进一步自动化 mocked app 和 task 的生成流程，形成闭环的 **agent-driven environment evolution**。
- 扩展至更多平台（如 iOS、特定操作系统发行版）。
- 探索更高效的 RL 算法（如 SAO）与分布式训练策略，降低资源门槛。
- 开放更多真实场景的 clone 应用（如政务、医疗等），推动实际落地。
- 引入 human feedback 或 preference modeling，优化 reward 设计。

> 🔗 **项目地址**：[https://github.com/Tongyi-MAI/MAI-Gym](https://github.com/Tongyi-MAI/MAI-Gym)  
> 📦 **已开源内容**：框架代码、49 个 mocked apps、2,000 个验证任务、MAI-Gym-9B 模型 checkpoint

</details>

---

### 13. [X-RIC: An Agentic RAN-Intelligent Controller Benchmark for Enhancing Reinforcement Learning](https://arxiv.org/abs/2610.22269)

**Authors**: Adhwaa Al-Chaab, Ayman Younis, Dario Pompili  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.22269v1  

#### Abstract
The O-RAN ALLIANCE introduces architectural flexibility through disaggregation and AI-driven control. While the RAN Intelligent Controller (RIC) enables centralized optimization, current AI approaches lack reasoning capabilities and explainability. We propose eXplainable RIC (X-RIC), a novel framewo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：X-RIC: An Agentic RAN-Intelligent Controller Benchmark for Enhancing Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前在 O-RAN 架构中，虽然 **Reinforcement Learning (RL)** 被广泛用于 RAN Intelligent Controller (RIC) 的资源管理与优化，但仍面临以下三大挑战：
- **样本效率低**：RL 需要大量交互训练，在真实网络中不可行；
- **泛化能力差**：面对未知流量模式时表现不稳定；
- **缺乏可解释性**：AI 决策过程“黑箱”，难以满足运营和监管要求。

此外，现有的 LLM 或 **Large Language Model** 方法多作为静态推理引擎使用，无法实现闭环自适应控制，且存在延迟、非确定性和格式错误等部署风险。

---

### 🚀 提出的新方法与创新思路
本文提出 **X-RIC**（eXplainable RIC），一种融合 **Agentic AI** 与 RL 的新型 RAN 智能控制器框架，其核心创新包括：

#### （1）**混合控制架构：反应式 RL + 战略性 Agentic 推理**
- 引入 **Telecom Language Model (TLM)** ——一个基于 Transformer 的领域专用语言模型，具备自然语言推理与解释能力。
- TLM 提供高层战略指导，并生成带解释的控制建议，增强决策透明度。

#### （2）**自动化行为克隆（Behavioral Cloning, BC）机制**
- 利用 TLM 生成高质量专家轨迹（demonstrations），无需人工标注；
- 设计 **双过滤机制（Dual-Filtering）**：
  - **Episode-level 过滤**：仅保留对性能有正向影响的完整 episode；
  - **Action-level 过滤（Q-filter）**：通过 critic 网络筛选出优于当前策略的动作，确保示范质量。
- 显著提升 RL 的样本效率，同时避免次优动作污染训练数据。

#### （3）**闭环协同进化机制**
- 在 Near-RT RIC 控制环路中实现 TLM 与 RL Agent 的动态协作：
  - TLM 定期干预并提供修正建议；
  - RL Agent 执行快速实时决策；
  - 高质量交互被自动采集为新示范，持续优化策略。

#### （4）提出 **BARX**：首个面向 Agentic RAN 控制的评估基准
- 专为评估 TLM 类模型在电信控制场景下的适用性设计；
- 四维评估指标：
  - **Latency**（延迟）
  - **Compliance**（JSON/YAML 合规性）
  - **Determinism**（输出一致性）
  - **Policy Impact**（对控制性能的实际增益）

---

### 🔍 相比现有方法的优势
| 维度 | 传统 RL | 现有 LLM 方法 | X-RIC |
|------|--------|---------------|-------|
| 样本效率 | 低（需百万级交互） | 不适用 | 高（利用 TLM 自动生成示范） |
| 可解释性 | 差 | 中（输出文本但无上下文理解） | 强（结合意图、状态与行动进行自然语言解释） |
| 实时性 | 高（<10ms） | 通常超时（>600ms） | 平衡良好（TLM ~465ms，RL <10ms） |
| 自适应能力 | 弱（依赖探索） | 弱（静态推理） | 强（闭环共适应） |
| 部署可靠性 | 高 | 低（常出现语法错误） | 高（BARX 筛选 + 双重过滤） |

---

## 2. 核心实验方法和设置

### 📊 数据集与环境
- **仿真环境**：基于 O-RAN 架构构建的 SLA-aware 网络切片系统；
- **网络配置**：
  - 单小区 gNB 支持三个 Slice：**eMBB**, **URLLC**, **mMTC**；
  - 用户数：Ne=10, Nu=10, Nm=6；
  - 资源单位：vRB（虚拟资源块）、vCW（虚拟计算负载）；
- **流量模型**：动态到达率与服务需求，模拟突发拥塞场景（如 URLLC 紧急业务爆发）；
- **SLA 要求**：
  - eMBB/mMTC 最小速率：12 Mbps；
  - URLLC 最大延迟：10 ms；
  - 违约惩罚项纳入奖励函数。

### ⚙️ 实验设置
- **MDP 建模**：
  - State：历史 KPI、到达率、服务需求特征；
  - Action：连续分配 (bi,k, Ci,k)，经舍入得离散 vRB 分配；
  - Reward：负成本 + SLA 违约惩罚；
- **算法基础**：采用 **DDPG**（Deep Deterministic Policy Gradient）作为 RL 主干；
- **TLM 模型**：基于 **LLaMA3-8B** 微调，集成 LangChain 工具链；
- **训练方式**：混合回放缓冲区（Mixed Replay Buffer）——同时采样探索数据与 TLM 生成的示范数据；
- **控制周期**：子秒级（sub-second），符合 Near-RT RIC 要求（10ms–1s）；

### 📈 评估指标
| 类别 | 指标 |
|------|------|
| **学习效率** | 收敛速度、训练步数、方差 |
| **运行性能** | 总运营成本、SLA 满足率、资源利用率 |
| **稳定性** | 奖励波动（coefficient of variation） |
| **部署可行性** | 延迟分布、合规性、确定性（BARX 框架） |

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **RL-only** | 标准 DDPG，无任何先验知识引导 |
| **TLM-RL** | TLM 直接参与控制，但无 BC 或示范学习机制 |
| **Equal Allocation** | 静态平均分配资源，代表传统配置策略 |
| **Non-fine-tuned LLM** | 未经领域微调的语言模型，用于验证微调必要性 |
| **X-RIC-n** | 使用 n 条示范数据训练的 X-RIC 版本（n=100/300/500） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自仿真与实测）

| 指标 | X-RIC (500 demos) | RL-only | 提升幅度 |
|------|------------------|---------|----------|
| **运营成本降低** | **89.0%** | 0%（基准） | ⬆️ +89% |
| **收敛速度** | **快 75%** | 基准 | ⬆️ 显著加速 |
| **训练步数减少** | **50% 更少** | 基准 | ⬇️ 大幅节省 |
| **奖励方差下降** | **46% 降低** | 基准 | ⬇️ 更稳定 |
| **解释方差（Explained Variance）** | **0.90** | ~0.87 | ⬆️ 更高拟合度 |
| **部署成本（相对）** | **~6× 低于基线** | 基准 | ⬇️ 极大节约 |

> 注：X-RIC300 成本降低 85.2%，X-RIC100 为 65.8%，呈明显正相关趋势。

---

### 🔬 消融实验结果

#### （1）不同干预策略对比（Ablation on Intervention Schedule）
测试六种 TLM 干预频率策略：
- Constant, Step-based, Linear Decay, Exponential Decay, Factor-based, **Periodic**

| 策略 | 性能增益/干预 | 波动（CV） | 推理延迟（中位数） |
|------|----------------|------------|--------------------|
| **Periodic**（每10步一次） | **11.86**（最高） | **15.2%**（最低） | **465.6 ms** |
| Constant | 0.92 | 36.9% | 465.6 ms |
| Factor-based | 1.86 | 26.8% | — |

✅ 结论：**Periodic Correction** 是最优选择 —— 平衡了性能增益与系统开销。

#### （2）示范数量的影响（Demonstration Size Study）
- 随着示范数量增加（100→500），策略性能稳步上升；
- X-RIC500 比 X-RIC100 奖励高出 **81.8% vs 80.2%**，且更稳定（CV 更低）；
- 表明 **示范质量和数量共同决定最终性能**。

#### （3）双过滤机制有效性
- 若不加过滤，直接克隆所有 TLM 输出会导致性能下降（因包含次优动作）；
- 加入 Q-filter 后，仅保留 critic 认可的动作，显著提升策略鲁棒性。

---

### 🧪 实地测试（Over-the-Air Testbed）

#### 测试平台组成（见 Fig. 9）：
- **RAN 层**：OpenAirInterface gNB + USRP B210 SDR；
- **RIC 层**：FlexRIC（Near-RT RIC）通过 E2 接口连接；
- **核心网**：OAI 5G Core（AMF/SMF/UPF），支持切片；
- **终端**：Sierra Wireless 5G UE + USRP B210 模拟器；
- **数据库**：Kinetica 实时时间序列数据库接收 KPI 流。

#### 实测结果（Fig. 10）：
| 指标 | X-RIC500 表现 |
|------|--------------|
| **eMBB 吞吐量 CDF** | 95% 时间 >17 Mbps（SLA=12 Mbps），余量充足 |
| **URLLC 延迟 CDF** | 中位数 ≈5 ms（SLA=10 ms），安全性高 |
| **vCW 利用率** | 下降至 **0.65**（原接近 1.0），节省 35% 计算资源 |
| **vRB 利用率** | 降至 **8.5–9.5 blocks**（总量 12），节省约 25% 频谱资源 |

✅ 对比 Equal Allocation 和 TLM-RL，X-RIC 在 SLA 满足与资源效率上均取得压倒性优势。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Agentic AI 可有效增强 O-RAN 控制智能化水平**：
   - 将 TLM 从“静态问答器”转变为“战略协作者”，实现可解释、可追溯的智能决策。
   
2. **自动化 Behavioral Cloning 显著提升 RL 效率与稳定性**：
   - 无需人工标注即可获得高质量示范；
   - 双过滤机制保障示范可靠性，防止“垃圾进垃圾出”。

3. **闭环共适应机制是实现持续优化的关键**：
   - TLM 与 RL 相互促进：TLM 提供初始知识，RL 快速执行并反馈经验，反哺 TLM 指导质量。

4. **BARX 是评估 Agentic 控制器的重要工具**：
   - 揭示多数 LLM 存在合规性或延迟问题（如 TL 模型虽快但 100% 报错）；
   - LLaMA3-8B 凭借综合得分成为最佳候选。

5. **X-RIC 实现了部署级性能**：
   - 在仿真与 OTA 测试中均表现出色；
   - 成本降低近 90%，满足 Near-RT RIC 实时性约束。

---

### ⚠️ 方法局限性
1. **TLM 推理延迟仍较高**（~450ms）：
   - 虽然未突破 600ms 上限，但在极端时延敏感场景仍有压力；
   - 未来需轻量化 TLM 或边缘部署优化。

2. **依赖高质量 prompt engineering 和 schema 设计**：
   - 当前系统对提示词敏感，需精细调优才能保证输出结构正确。

3. **扩展性待验证**：
   - 当前实验限于单小区三切片；
   - 多小区、大规模用户场景下的可扩展性尚未充分测试。

4. **长期稳定性未知**：
   - 缺乏长时间运行的日志分析，闭环系统的漂移风险需进一步研究。

---

### 🔮 未来工作方向
1. **开发更高效的 TLM 蒸馏版本**（如 Tiny-TLM）以降低延迟；
2. **引入多智能体架构**，支持跨小区协调与联邦式 X-RIC；
3. **结合 Offline RL**，利用历史运维日志进行预训练；
4. **增强安全性机制**，防御对抗性提示攻击或误导性指令注入；
5. **推动 BARX 成为行业标准**，建立 Agentic RAN 控制的统一评测体系。

---

## ✅ 总结
X-RIC 是首个将 **Agentic AI** 深度融入 O-RAN RIC 架构的框架，通过 **TLM + RL + BC + 闭环共适应** 的设计，成功解决了传统 RL 在样本效率、可解释性和部署可靠性方面的瓶颈。实验表明，X-RIC 在 SLA 满足、资源效率和学习速度方面全面超越现有方法，为下一代 AI-Native RAN 控制提供了可行路径。

</details>

---

### 14. [Hybrid Generative-Tagging Decoding for Efficient Product Attribute-Value Extraction](https://arxiv.org/abs/2610.21329)

**Authors**: Keshav Kolluru, Manas Gupta, Maroof Ahmad, Suchitra Sathyanarayana  
**Category**: cs.CL  
**Published**: 2026-10-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.21329v1  

#### Abstract
Attribute-value extraction is important for building high-quality product catalogs in e-commerce. Existing systems either rely on generative or tagging strategies for extracting the attribute names and their corresponding values. Tagging models directly label spans in the source, making them fast an...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Hybrid Generative-Tagging Decoding for Efficient Product Attribute-Value Extraction**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在电商场景中，**Attribute-Value Extraction (AVE)** 是构建高质量商品目录的关键任务。现有方法主要分为两类：
- **Tagging 模型**：通过标注文本中的连续片段（spans）提取属性值，速度快、结果可溯源，但无法处理需要格式化或隐含推理的值，且受限于固定的标签空间。
- **Generative 模型**：使用 LLM 自回归生成属性名和值，支持开放标签空间、归一化和隐含值提取，但推理速度慢，且受训练数据不完整影响，常遗漏稀有属性。

因此，单一范式难以兼顾**效率、覆盖率和准确性**。

### **提出的新方法**
本文提出 **Hybrid Generative-Tagging Decoding**，一种结合生成与标注的混合解码架构：
- **共享一个 LLM 的 prefill 阶段**，对输入商品文本进行一次编码得到隐藏状态 $ H $。
- **双通道并行解码**：
  - **Tagging Channel**：使用轻量级 **Span Head** 并行检测前 K=500 个高频属性是否存在，并直接从源文本复制连续 span 作为值（无需自回归解码）。
  - **Generative Channel**：自回归生成剩余部分——包括非连续、需格式化、隐含或不在标签空间内的属性。
- 最终输出合并两个通道的结果，冲突时优先保留生成值。

### **相比现有方法的优势**
- **更高覆盖率**：解决了生成模型因自回归长度限制而忽略稀有属性的问题。
- **更低延迟**：将大量可复制属性的提取变为非自回归操作，显著减少生成 token 数。
- **保持高精度**：Tagging 保证可复制值的准确性和可溯源性，生成通道专注复杂推理。
- **正交优化**：该方法与 vLLM 等高效推理系统兼容，加速效果可叠加。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **主数据集**：来自 Amazon US 商品目录的 111K 条商品记录（训练集 100,195，验证集 11,132），共包含 2.62M 属性-值对。
  - 其中 **81.1% 为 taggable（可复制）**，18.9% 为 generative（需生成）。
- **公开数据集**：
  - **AE-110K**（Xu et al., 2019）
  - **OA-Mine**（Zhang et al., 2022）
  - 所有数据均标注为“可复制”或“需生成”。

### **实验设置**
- **Base Model**：Qwen3-0.6B（主实验），并在 Qwen3-1.7B、Qwen3-4B（LoRA）、Gemma-3-1B 上做消融。
- **Span Head 设置**：
  - K = 500（覆盖语料中 87.8% 的属性实例）
  - 使用 cross-attention + 双线性指针（start/end）定位 span
  - 引入 **learned NULL slot** 判断属性是否存在
- **训练方式**：
  - 混合损失：$ \mathcal{L} = \mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{tag}} $
  - Route supervision：若值可在源中找到连续匹配，则路由至 tagging；否则保留在 generative 中
- **推理平台**：vLLM，单张 A100 GPU

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Precision** | LLM judge（GPT-5.6 Luna）判断生成对是否正确（允许合理归一化） |
| **Recall** | 覆盖的黄金属性比例（exact match） |
| **Num. Extractions** | 每条商品平均提取的属性数 |
| **Quality** | 人工盲评打分（1–5 分），评估正确性、完整性、可溯源性 |
| **Speed** | 单条商品平均响应时间（ms） |

### **基线方法对比**
- **Generative**：纯自回归生成所有属性
- **Tagging**：仅使用 span head，无生成能力
- **GenToC**（Subhalingam et al., 2025）：先生成候选属性，再用分类器填充值
- **MVP-RAG**（Zou et al., 2025）：检索增强生成
- **HPD**（Glavas et al., 2026）：超并行解码策略

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Amazon 数据集）**

| Model | Precision | Recall | Num Extr. | Speed (ms) |
|-------|----------|--------|-----------|------------|
| Generative | 97.5% | 69.8% | 21.3 | 56.6 (1.00×) |
| Tagging | 97.4% | 65.3% | 29.3 | 9.9 (5.69×) |
| **Hybrid** | **97.1%** | **80.2%** | **34.8** | **31.5 (1.80×)** |

#### **核心结论**
- **召回率提升 +10.4 pts**（69.8% → 80.2%），多提取 **63% 正确属性对**（33.8 vs 20.7 对/商品）
- **精度持平**（97.1% vs 97.5%），TOST 等价性检验确认无统计差异（±1 pt 内）
- **延迟降低 44%**（56.6ms → 31.5ms），速度提升 **1.80×**
- **生成 token 数下降 69%**（257.8 → 80.0）

### **与其他系统的对比**
| System | F1 (AE-110K) | Speed-up (vs Gen) |
|--------|---------------|-------------------|
| Hybrid | **0.841** | 2.1× |
| HPD | 0.839 | 2.5× (但不兼容 vLLM) |
| MVP-RAG | 0.836 | 0.9× |
| GenToC | 0.705 | 1.7× |

- 在公开数据集上，Hybrid 达到最佳 F1，且在 vLLM 下仍保持显著加速优势。
- HPD 虽报告更高加速比，但其定制解码逻辑无法运行于标准推理框架（如 vLLM），实际优势被削弱。

### **消融实验结果**

#### **(1) Span Head 设计对比（Table 6）**
| Delimitation | Precision | Recall |
|-------------|----------|--------|
| Per-token tagging | 87.7% | 74.1% |
| Anchored pointer | 93.1% | 71.5% |
| **Independent pointer** | **97.1%** | **80.2%** |

- **独立指针（independent pointer）表现最优**：start 和 end 分别评分，避免起始点误差传播。

#### **(2) 标签空间大小 K 的影响（Table 9）**
| K | Vocab Cov. | Recall | Speed |
|----|------------|--------|--------|
| 100 | 70.0% | 69.2% | 37.5ms |
| 250 | 82.0% | 76.2% | 33.3ms |
| **500** | **87.8%** | **80.2%** | **31.5ms** |
| 1000 | 92.0% | 80.9% | 29.0ms |

- K=500 后 recall 增益趋于饱和（+0.7 pts），选择 **K=500 为最优权衡点**。

#### **(3) 不同 backbone 的泛化性（Table 5）**
- 在 Qwen3 和 Gemma 系列模型上均一致取得：
  - **+9.4 ~ +13.0 pts Recall 提升**
  - **1.30–1.80× 速度提升**
  - 精度维持在 ±1 pt 内
- 表明方法具有良好的模型无关性。

#### **(4) 错误分析与可溯源性审计（Appendix B）**
- **90.8% 的输出值在原文中可找到 verbatim 匹配**
- 所有真实幻觉（fabrication）均来自 **Generative Channel**
- Tagging Channel 输出 **100% 可溯源**
- Hybrid 的幻觉率（~0.04%）低于 Generative（~0.12%），优势来自结构设计而非生成更可靠

---

## **4. 关键结论和发现**

### **主要发现**
1. **混合解码是高效 AVE 的有效范式**：
   - 将“复制型”属性交给非自回归 tagging，释放生成通道压力。
2. **召回率瓶颈源于自回归机制**：
   - 即使更强的 generative model 也无法弥补长序列中稀有属性被跳过的现象。
3. **Tagging Channel 显著提升 extractive 子集表现**：
   - 对 extractive 属性 recall 从 70.2% 提升至 82.5%。
4. **加速效果可与 vLLM 等系统叠加**：
   - 本方法减少的是“需生成 token 数”，与内存管理、KV cache 优化正交。

### **局限性**
- **依赖 LLM 注释数据**：训练与评估均基于 LLM 自动生成标签，存在噪声风险。
- **评估依赖 LLM Judge**：精度与质量指标可能继承 judge 模型的偏见。
- **当前仅限文本来源属性**：未考虑图像、DOM 结构等多模态信号。
- **语言与领域局限**：实验集中于英文电商数据，跨语言、跨领域迁移能力未知。
- **标签空间固定**：虽然覆盖 87.8%，但仍无法动态扩展至全新属性类别。

### **未来工作方向**
- 扩展至 **multimodal 输入**（图像、布局、HTML 结构）
- 探索 **dynamic vocabulary expansion** 以支持完全开放世界 AVE
- 研究 **cross-domain transfer** 能力
- 将 hybrid decoding 思想推广至其他信息抽取任务（如事件抽取、关系抽取）
- 进一步优化 Span Head 架构，提升边界识别准确率（尤其对复合值）

---

> ✅ **一句话总结**：  
> 本文提出的 **Hybrid Generative-Tagging Decoding** 通过分离“复制”与“生成”路径，在共享 LLM prefill 的基础上实现高效、高召回、高精度的商品属性提取，**在保持 97%+ 精度的同时将召回率提升 10.4 pts，延迟降低 44%**，为大规模 catalog 构建提供了实用且可扩展的解决方案。

</details>

---

### 15. [ShiftQAT: Quantization-Aware Training for the Numerical Contract of a Fixed-Function Edge CNN Accelerator](https://arxiv.org/abs/2610.21695)

**Authors**: Oguzhan Buyuksolak, Osman Erman Okman, Mehmet Gorkem Ulkar  
**Category**: cs.LG  
**Published**: 2026-10-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.21695v1  

#### Abstract
Fixed-function CNN accelerators make edge inference fast and energy-efficient, but they execute only networks that obey a precise numerical contract. On the MAX7800x family, this contract requires power-of-two, per-tensor, symmetric scales, 8-bit activations and biases, integer-only arithmetic, and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ShiftQAT: Quantization-Aware Training for the Numerical Contract of a Fixed-Function Edge CNN Accelerator**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
固定功能（fixed-function）边缘CNN加速器（如 Analog Devices 的 MAX7800x 系列）要求神经网络必须满足严格的**数值合约（numerical contract）**才能部署。该合约包括：
- **Power-of-two、per-tensor、对称（symmetric）量化尺度**
- **8-bit 激活和偏置存储**
- **整数运算（integer-only arithmetic）**
- **每层仅允许一个静态配置的位移（shift）**
- **无动态重缩放（no dynamic rescaling）**

传统 Quantization-Aware Training（QAT）方法通常无法同时满足这些硬性约束，导致训练出的模型无法在硬件上编译运行。

---

### **提出了什么新方法或新思路**
本文提出 **ShiftQAT**，一种专为 MAX7800x 加速器设计的 QAT 流程，其核心机制包括：

#### ✅ **静态尺度传播（Static Scale Propagation）**
- 利用激活函数的**正齐次性（positively homogeneous）**（如 ReLU），将每一层的激活尺度向前折叠到下一层的权重和偏置中。
- 所有重缩放操作在训练前离线完成，中间层不恢复实值尺度，最终只在网络输出端进行一次全局恢复。
- 避免了运行时动态 rescaling，符合硬件“每层仅一个 shift”的限制。

#### ✅ **联合 power-of-two 权重/偏置尺度（Joint Power-of-Two Weight/Bias Scaling）**
- 权重和偏置共享同一个 power-of-two 尺度 $ S_w = 2^{s_w} $，确保二者在同一量化网格上。
- 在每个训练迭代中重新估计该尺度，使网络能适应部署时的量化约束。
- 解决了偏置吸收上游激活尺度后仍需压缩到 8-bit 的挑战。

#### ✅ **拓扑依赖解析（Topology Dependency Resolution）**
- 显式处理残差连接（residual）、拼接（concatenation）和权重共享（weight sharing）等结构中的尺度冲突。
- 统一输入分支的尺度，避免运行时对齐。

#### ✅ **PoT Scaler 即硬件 shift 寄存器值**
- 每层的 `PoT Scaler` 直接计算硬件所需的 shift 值 $ k = \text{clip}(s_w - s_x'; -15, 15) $。
- 训练过程直接优化硬件可执行的配置，无需后校准（post-calibration）。

---

### **相比现有方法的优势**
| 特性 | ShiftQAT | 其他 QAT/PTQ 方法 |
|------|----------|------------------|
| Power-of-two scales | ✅ | ❌（如 LSQ 使用非 2 的幂） |
| Per-tensor scales | ✅ | ❌（如 HPTQ 使用 per-channel） |
| 8-bit biases | ✅ | ❌（如 TQT 使用 16-bit） |
| Signed 8-bit activations | ✅ | ❌（如 TQT 使用 unsigned） |
| 静态 shift per layer | ✅ | ❌（需要动态 rescaling） |
| 无需后校准 | ✅ | ❌（多数需要） |

> **ShiftQAT 是唯一满足 MAX7800x 完整数值合约的方法**，且精度损失极小。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ImageNet-1K (ILSVRC-2012)**：标准图像分类任务，输入分辨率 224×224（训练）和 128×128（部署）。

---

### **实验设置和评估指标**

#### **模型架构**
- **MobileNetV2**
- **ResNet-18**

#### **量化配置**
- **W8A8**：8-bit weights, 8-bit activations
- **W4A8**：4-bit weights, 8-bit activations
- 偏置默认使用 32-bit 进行公平比较（表 2、3），实际部署使用 8-bit（表 4）

#### **训练细节**
- 从官方 PyTorch FP32 checkpoint 开始微调。
- QAT 微调最后 50（MobileNetV2）或 30（ResNet-18）个 epoch。
- 使用 SGD + Cosine/MultiStep LR 调度。
- 所有结果来自单次运行。

#### **评估指标**
- **Top-1 Accuracy (%)**
- **Accuracy Drop (Δ, percentage points)**
- **Latency (ms)** 和 **Energy (mJ)**：在 MAX78002 硅片上实测
- **Memory Usage**：权重和偏置大小

---

### **基线方法对比**
| 方法 | 类型 | 是否满足合约 |
|------|------|-------------|
| **TQT** (Jain et al., 2020) | QAT | ❌（16-bit bias, unsigned activation） |
| **HPTQ** (Habi et al., 2021) | PTQ | ❌（per-channel scales） |
| **LSQ** (Esser et al., 2020) | QAT | ❌（非 2 的幂 scales） |
| **AdaRound**, **BRECQ**, **EPTQ** | PTQ | ❌（均违反 C2/C3/C4） |
| **Jacob et al. (2018)** | QAT | ❌（非对称、非 2 的幂） |

> 所有基线都无法完整满足 MAX7800x 合约，因此只能进行**精度层面的间接比较**。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **W8A8 精度表现（Table 2）**
| Model | Method | FP32 (%) | Quantized (%) | Δ (pp) |
|-------|--------|---------|--------------|--------|
| MobileNetV2 | **ShiftQAT** | 71.878 | **70.852** | **1.026** |
| ResNet-18 | **ShiftQAT** | 69.758 | **69.200** | **0.558** |

- 与更宽松的 QAT 方法（如 Jacob et al. 的 1.0 pp drop）**精度损失相当**。
- 唯一满足所有合约约束的方法。

#### **W4A8 精度表现（Table 3）**
| Model | Method | FP32 (%) | Quantized (%) | Δ (pp) |
|-------|--------|---------|--------------|--------|
| MobileNetV2 | **ShiftQAT** | 71.878 | **69.014** | **2.864** |
| ResNet-18 | **ShiftQAT** | 69.758 | **67.854** | **1.904** |

- 优于 AdaRound（3.37 / 2.46 pp drop），接近 per-tensor QAT（1.55 / -0.08 pp）。
- **在严格约束下，仅落后 FP32 不到 2.9 pp**。

---

### **硅片部署结果（Table 4）**
> 实际部署模型使用 **mixed 4/8-bit weights + 8-bit biases**，并缩小模型以适配 on-chip memory。

| Model | Input | Acc. (%) | Weight Size | Biases (kB) | Latency (ms) | Energy (mJ) |
|-------|-------|----------|------------|-------------|--------------|-------------|
| MobileNetV2 (m=0.75) | 128×128 | 62.812 | 2.03 MB | 6.6 | **14.6** | **5.1** |
| ResNet-18 (m=0.5) | 128×128 | 53.230 | 2.05 MB | 3.4 | **8.4** | **2.8** |

- **毫秒级延迟，毫焦耳级能耗**，适合电池供电边缘设备。
- 所有层在 CNN 加速器上以整数运算执行，仅最终输出由 Arm Core 处理。

---

### **消融实验与分析**
虽然未提供显式的消融表，但文中通过以下方式验证机制有效性：
- **静态传播 vs 动态 rescaling**：若不提前折叠尺度，中间层会因 8-bit 存储而严重失真。
- **联合尺度 vs 独立量化**：若权重和偏置独立缩放，会导致偏置溢出或精度丢失。
- **拓扑解析必要性**：残差/拼接结构若不统一尺度，硬件无法合并输入。

---

## **4. 关键结论和发现**

### **主要发现**
1. **数值合约是部署的关键瓶颈**：“W4A8” 并不足以定义可部署模型，必须明确对称性、尺度粒度、bias 宽度等细节。
2. **ShiftQAT 可在严格约束下保持高精度**：
   - W8A8 下精度损失与宽松 QAT 相当；
   - W4A8 下仍保持在 FP32 的 2.9 pp 以内。
3. **训练应直接输出硬件配置**：
   - PoT Scaler 直接对应硬件 shift 寄存器值；
   - 无需后校准，路径可预测。
4. **内存限制驱动模型设计**：
   - 权重内存决定了宽度、分辨率、混合精度策略；
   - ShiftQAT 的 W4A8 准确性使得使用更多 4-bit 层成为可能。

---

### **方法的局限性**
1. **依赖正齐次激活函数**：
   - 仅支持 ReLU、Leaky ReLU、Modulus；
   - 不支持 Sigmoid、Tanh、GELU、SiLU 等非齐次函数。
2. **无法利用 per-channel 量化增益**：
   - per-channel scales 被硬件排除，牺牲了一定精度潜力。
3. **ReLU 输出仅使用 7/8 bits**：
   - 硬件强制将 ReLU 输出 clip 到 [0,127]，浪费一半编码空间。

---

### **未来工作方向**
1. **引入可学习的 power-of-two 阈值**（如 TQT）：
   - 可进一步缩小 W4A8 下与放松约束方法的差距。
2. **扩展至其他 shift-based 固定功能加速器**：
   - ShiftQAT 的机制具有通用性，适用于类似数值合约的硬件。
3. **探索非均匀量化兼容方案**：
   - 在保持整数运算的前提下，研究 lookup table 或编码映射是否可行。
4. **自动化混合精度分配**：
   - 基于敏感度分析自动决定哪些层可用 4-bit，提升能效比。

---

> 🔗 **代码开源地址**：[https://github.com/analogdevicesinc/ai8x-training](https://github.com/analogdevicesinc/ai8x-training)  
> 📌 **ShiftQAT 已作为 ai8x-training 工具链的默认训练路径**，支持 MAX7800x 全系列芯片。

</details>

---

### 16. [BitSparse: Compact Activation Storage for ReLU Feed-Forward Networks](https://arxiv.org/abs/2610.22256)

**Authors**: Max Zhu, Bohang Sun, Pietro Lio, Jialin Yu  
**Category**: cs.LG  
**Published**: 2026-10-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.22256v1  

#### Abstract
Reducing memory use in large language model (LLM) training is an ongoing challenge. Activation checkpointing saves memory by recomputation, while lossy compression changes the stored activations. The reemergence of sparse $\relu$ and $\relu^2$ activations creates a third opportunity: losslessly comp...

---

### 17. [ARGUS: An Agenda-Driven Framework for Reliable User Simulation](https://arxiv.org/abs/2610.21346)

**Authors**: Hang Zheng, Nan Jiang, Zichen Zhu, Zixu Yang, Zhiyang Tang, Shuai Fan, Lu Chen, Xiaobao Wu, Kai Yu  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.21346v1  

#### Abstract
High-fidelity user simulators are essential for evaluating and training agents in interactive environments. However, current LLM-based simulators struggle with reliability issues, such as goal omission and policy non-compliance. We propose ARGUS (Agenda-dRiven Goal-oriented User Simulator), a neuro-...

---

### 18. [DPA-MoE: A Runtime Precision Control Framework for MoE Inference](https://arxiv.org/abs/2610.21378)

**Authors**: Jaeyun Kim, Hoonki Lee, Susie Shin, Hyunmin Kim, Dahyun Lee, Sangyeob Kim  
**Category**: cs.CL  
**Published**: 2026-10-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.21378v1  

#### Abstract
Mixture-of-Experts (MoE) models reduce computation by routing each token to a small set of experts, but each selected expert is still used at a fixed precision. We present DPA-MoE, a runtime precision control framework that makes routed experts any-precision. DPA-MoE encodes expert feed-forward weig...

---

### 19. [Resource-aware Sub-network Allocation for Heterogeneous Decentralized Learning](https://arxiv.org/abs/2610.22015)

**Authors**: Zhuojun Tian, Ming Xiao, Mikael Skoglund  
**Category**: cs.DC  
**Published**: 2026-10-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.22015v1  

#### Abstract
System heterogeneity in computation, communication, and memory can substantially limit synchronous decentralized learning, where each training round is constrained by resource-limited nodes. Existing approaches provide limited support for jointly determining node-specific model capacities from deplo...

---

### 20. [Feature Elicitation for SAE Steering of Deceptive Behaviour in Small LLMs](https://arxiv.org/abs/2610.20638)

**Authors**: J Koorndijk  
**Category**: cs.LG  
**Published**: 2026-10-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.20638v1  

#### Abstract
Research on deceptive alignment is becoming increasingly urgent as frontier-model capabilities advance and their risk of causing catastrophic harm increases. Alignment faking is a form of deceptive alignment, with the compliance gap serving as a key behavioural indicator. This behaviour remains unde...

---

### 21. [GRADS: Graph-Regulated Agentic Data Synthesis with Controllable Difficulty](https://arxiv.org/abs/2610.21167)

**Authors**: Gaocheng Yang, Ruijie Tao, Kai Ye, Senhua Zhang, Zhengqin Zhu, Ziyang Li  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.21167v1  

#### Abstract
Deep research agents are increasingly deployed to retrieve precise information from large corpora, yet multi-hop, long-horizon tasks remain challenging. Synthetic data generation has emerged as a scalable solution, with existing pipelines typically expanding seed tasks to increase their apparent dep...

---

### 22. [In-Context Preference-Conditioned Pareto Set Learning from Limited Offline Data](https://arxiv.org/abs/2610.21736)

**Authors**: Yiming Yao, Liang Zhao, Qingfu Zhang  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.21736v1  

#### Abstract
Multi-objective optimization involves diverse preferences over conflicting objectives. Rather than returning a finite set of trade-off solutions, preference-conditioned Pareto set learning seeks to map a specified preference directly to a corresponding Pareto optimal solution. Yet in many real-world...

---

### 23. [Probabilistic Optical-Layer Surrogate for Risk-Aware Control-Plane Decision Making](https://arxiv.org/abs/2610.21069)

**Authors**: Venkat Sai Suman Lamba Karanam  
**Category**: cs.DC  
**Published**: 2026-10-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.21069v1  

#### Abstract
In this paper, we propose a generative surrogate for the optical layer. Current control-plane decisions in optical transport networks are made using analytical impairment models, deterministic learned predictors, or quantile-regression-based uncertainty estimates. These approaches expose only a limi...

---

### 24. [Coordination-Free Workload Regularization for Variable-Cost Distributed Training](https://arxiv.org/abs/2610.21162)

**Authors**: Tao Huang, Chen Zhu, Antonio Alonso, Daming Li, Adarsh Goura, Jaideep Ray, Yitong Zhou  
**Category**: cs.DC  
**Published**: 2026-10-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.21162v1  

#### Abstract
Synchronous data-parallel training usually assigns every rank the same number of records, even when record costs vary by orders of magnitude. For variable-length models, this converts input variation into idle time because each step waits for its most expensive rank-local batch. We present Batch Cos...

---

### 25. [Architecting for Hardware Diversity: Meta's End-to-End Co-Design for High-Performance AI Training on AMD GPUs](https://arxiv.org/abs/2610.22230)

**Authors**: Haoyu Zhang, Arshad Rahman, Adnan Aziz, Ajith Prabhudev, Amin Farmahini, Ben Rhoads, Benson Ma, Brian Shi, Cen Zhao, CK Luk, Dingming Wu, Emad El-Haraty, Eric Chen, Frank Li, Gary Huang, Gefei Zuo, Haoran Zhang, Hiwot Kassa, Hongyi Zeng, Jack Taylor, Jeremy Hadidjojo, John Donati, Jonathan Sheffield, Julio Machado Silva, Karthik Ganesan, Kent Russell, KR Kishore, Lei Tian, Liu Ke, Muhammad Osama, Peng Chen, Rakesh Hemnani, Ryan Swann, Saleel Kudchadker, Shankar Yanamandram, Shobhit Kanaujia, Shucai Xiao, Srinath Bayareddy, Srinivas Vaidyanathan, Sudharssun Subramanian, Supadchaya Puangpontip, Taylor Robie, Tianhao Huang, Tyler Hart, Viswesh Sankaran, Weijun Jiang, Wenkai Du, Xiaodong Wang, Xiaohu Guo, Yao Rong, Yinbin Ma, Yue Dong, Yuzhen Huang, Yvonne Liu, Zhan Shu, Dan Rabinovitsj, Ajit Mathews, Chunqiang Tang  
**Category**: cs.DC  
**Published**: 2026-10-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.22230v1  

#### Abstract
Meta deploys a heterogeneous fleet comprising NVIDIA GPUs, AMD GPUs, and proprietary MTIA accelerators to reduce costs and mitigate vendor lock-in. However, the extensive history of NVIDIA-specific optimizations hinders performance portability to other hardware. This paper presents our methods for o...

---

### 26. [FlashAttention with Statistics: Low-Overhead Attention Features for Compact Hallucination Detectors](https://arxiv.org/abs/2610.21483)

**Authors**: Andrei Volodichev, Maksim Savichev, Nikita Sviridov, Artem Yuliy, Dmitry Simakov, Alexey Zaytsev  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.21483v1  

#### Abstract
Large language models can produce fluent answers that the supplied context does not support. A compact frozen proxy model can detect such answers by reading the context and another model's response, even from a proprietary generator. Attention-based detectors use summary statistics of the proxy's at...

---

### 27. [AgentPerturb: Evaluating LLM Agent Robustness under Controlled Interaction Perturbations](https://arxiv.org/abs/2610.21624)

**Authors**: Yuanjian Xu, Jinyi Han, Jianing Hao, Yuxuan Sun, Fanyang Lu, Richeng Xuan, Guang Zhang, Zhichao Hu  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.21624v1  

#### Abstract
Real-world tasks are often complex and noisy, posing substantial challenges for LLM agents. Yet existing evaluations lack a systematic framework for jointly characterizing potential perturbation targets and types. We use this two-dimensional view to conduct a unified study that quantifies the robust...

---

### 28. [PROBE: Learning to Audit Policy Compliance in Tool-Using LLM Agents](https://arxiv.org/abs/2610.22124)

**Authors**: Kshitij Mishra, Abhijith Sharma, Nils Lukas, Salem Lahlou  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.22124v1  

#### Abstract
Modern agentic systems extend language models with external tools, enabling them to complete complex tasks through multi-turn interactions. As these agents are deployed in consequential settings, they are often constrained by policies governing pricing, privacy, confirmation, safety, and other requi...

---

### 29. [GNN-Guided Local Refinement for Efficient Layout Decomposition with DSA-MP](https://arxiv.org/abs/2610.22214)

**Authors**: Yu Feng, Liyi Zhou, Haoyu Wang, Yiming Xu, Hao Geng, Shaozhi Deng, Ningsheng Xu, Shisheng Xiong  
**Category**: cs.AI  
**Published**: 2026-10-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.22214v1  

#### Abstract
Multiple patterning lithography (MPL) has been widely used to improve lithography resolution in advanced nodes, but it faces cost limits as feature sizes shrink. Directed self-assembly (DSA) offers a cost-effective complement, extending the resolution of optical lithography to sub-10 nm and beyond. ...

---

### 30. [Automatic Sentence Completion Using Bengali Audio Speech with a Transformer-Based Early Speech Recognition System](https://arxiv.org/abs/2610.21164)

**Authors**: Md Asif Akram, Arif Faisal, Sakir Hossain Faruque, Sharun Akter Khushbu  
**Category**: cs.CL  
**Published**: 2026-10-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.21164v1  

#### Abstract
Dementia severely impairs communication abilities and turns everyday conversations into challenging experiences. Individuals affected by dementia frequently struggle to express their thoughts or complete sentences, making it difficult for family members and caregivers to understand their needs. This...

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

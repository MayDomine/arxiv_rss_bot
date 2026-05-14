# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-14 08:19:32 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](https://arxiv.org/abs/2605.13319)

**Authors**: Yunhe Han, Yunqi Gao, Bing Hu, Mahdi Boloursaz Mashhadi, Yitong Duan, Pei Xiao, Yanfeng Zhang  
**Category**: cs.DC  
**Published**: 2026-05-14  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2605.13319v1  

#### Abstract
Speculative decoding can significantly accelerate LLM inference, especially given that its cloud-edge collaborative deployment offers cloud workload offloading, offline robustness, and privacy enhancement. However, existing collaborative inference frameworks with speculative decoding are constrained...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **cloud-edge collaborative inference** 框架在结合 **speculative decoding** 时面临两大瓶颈：
1. **Sequential token generation and communication**：传统方法采用“先生成全部 draft tokens → 再上传 → 云端验证”的串行模式，导致计算与通信资源利用率低，带宽和算力空闲严重。
2. **Inflexible NAV triggering mechanism**：现有框架要么使用固定长度的 draft sequence，要么依赖单一置信度阈值（如单个 token 或整个序列的置信度）触发非自回归验证（NAV），容易造成过早验证（premature verification）或过度推测导致大规模 rollback。

这些问题限制了推理速度提升，并增加了能耗。

---

### 提出的新方法与创新思路

作者提出 **PipeSD**，一个高效的云边协同 pipeline 推理框架，其核心创新包括：

#### （1）Token-Batch Pipeline Scheduling Mechanism
- 引入 **token-batch 流水线调度机制**，将 draft token 的生成与传输重叠执行，最大化资源利用率。
- 将最优批处理策略建模为优化问题，通过 **Dynamic Programming (DP)** 求解最小化总延迟的分批方案，考虑了通信启动开销（startup overhead）、每 token 传输时间（β）和边缘端每 token 计算时间（γ）。

#### （2）Dual-Threshold NAV Triggering Mechanism
- 设计双阈值机制联合判断是否触发 NAV：
  - **Single-token confidence threshold (R₂)**：任一 token 置信度过低即触发，防止错误传播。
  - **Cumulative sequence confidence threshold (R₁)**：整体序列置信度下降到阈值以下时触发，避免累积误差。
- 引入轻量级 **Bayesian Optimization (BO) autotuner** 自动调优两个阈值，适应不同任务复杂度和动态环境变化。

#### （3）系统实现兼容性强
- 基于 `llama-cpp-python`、`PyTorch` 和 `FastAPI` 实现，易于部署。
- 控制逻辑集中在边缘侧，云端仅需运行目标模型进行验证，便于扩展并与现有云服务集成。

---

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **效率** | 通过流水线并行化显著减少端到端延迟 |
| **灵活性** | 双阈值机制更精准地控制验证时机，平衡 speculation gain 与 rollback cost |
| **自适应性** | BO autotuner 实现参数自动调节，无需人工调参 |
| **实用性** | 轻量设计，DP 和 BO 开销极小，适合真实场景部署 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **HumanEval**：编程任务基准测试，用于评估代码生成能力。
- **GSM8K**：小学数学应用题数据集，用于评估数学推理能力。

### 模型配置
| 场景 | Draft Model | Target Model |
|------|-------------|--------------|
| 编程任务 | DeepSeek-Coder-1.3B | DeepSeek-Coder-6.7B |
| 数学推理 | TinyLlama-1.1B-Chat-v1.0 | Llama-2-7B |

### 实验设置
- **测试平台**：真实城域网环境
  - 边缘设备：Lenovo ThinkBook 16+（Intel Ultra 9 CPU）
  - 云端服务器：天翼云 A800 GPU 实例
  - 上下行带宽：20 Mbps / 200 Mbps（标准 5G）
- **四种实验场景**：
  1. 高性能边缘设备（笔记本）
  2. 中等性能（模拟手机，2.5GHz CPU）
  3. 低性能（模拟 IoT 设备，1.2GHz CPU）
  4. 动态带宽环境（上行 10–80 Mbps，下行 150–280 Mbps）

### 评估指标
| 指标 | 含义 |
|------|------|
| **TPT (Time Per Token)** | 平均每个被接受 token 的生成时间（ms），衡量推理速度 |
| **ECS (Energy Consumption per 100 accepted tokens)** | 云端服务器每处理 100 个 accepted token 的能耗（J），反映能效 |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Vanilla** | 固定长度 speculative decoding（N=6 编程 / N=4 数学） |
| **HSL** | 单 token 置信度低于阈值时触发 NAV（阈值设为 0.99 / 0.7） |
| **EdgeLLM** | 基于累计序列置信度动态触发 NAV，原为边缘本地推理设计，本文适配至云边协作场景 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### ✅ TPT 性能对比（越小越好）
| 场景 | 数据集 | Vanilla | HSL | EdgeLLM | **PipeSD** | **Speedup vs Best Baseline** |
|------|--------|---------|-------|----------|------------|-------------------------------|
| 1 | HumanEval | 194 ms | 155 ms | 153 ms | **129 ms** | **1.19×** |
| 1 | GSM8K | 193 ms | 174 ms | 169 ms | **145 ms** | **1.17×** |
| 2 | HumanEval | 225 ms | 184 ms | 166 ms | **134 ms** | **1.24×** |
| 2 | GSM8K | 318 ms | 223 ms | 197 ms | **168 ms** | **1.17×** |
| 3 | HumanEval | 306 ms | 244 ms | 201 ms | **152 ms** | **1.32×** |
| 3 | GSM8K | 402 ms | 296 ms | 231 ms | **186 ms** | **1.24×** |
| 4 | HumanEval | 160 ms | 132 ms | 127 ms | **108 ms** | **1.18×** |
| 4 | GSM8K | 234 ms | 165 ms | 161 ms | **139 ms** | **1.16×** |

> 💡 **总体加速比**：**1.16× – 2.16×**（最高达 2.16× 超越 Vanilla）

#### ✅ ECS 能耗对比（Scenario 1，越小越好）
| 数据集 | Vanilla | HSL | EdgeLLM | **PipeSD** | **节能比例** |
|--------|--------|-----|---------|------------|----------------|
| HumanEval | 68 J | 71 J | 75 J | **56 J** | **25.3% ↓ vs EdgeLLM** |
| GSM8K | 98 J | 102 J | 100 J | **84 J** | **16.0% ↓ vs EdgeLLM**, **14.3% ↓ vs Vanilla** |

> 📉 **平均节能**：**14.3% – 25.3%**

---

### 消融实验结果（Ablation Study, Table 6）

| 方法变体 | Pipeline? | NAV Trigger | TPT (ms) | Speedup vs Vanilla |
|---------|-----------|------------|----------|---------------------|
| Vanilla | ❌ | Fixed-length | 194 | 1.00× |
| PipeSD w/o Pipeline | ❌ | Dual-threshold | 147 | 1.32× |
| PipeSD + Fixed-length | ✅ | Fixed-length | 164 | 1.18× |
| PipeSD + Token-level | ✅ | Single-token | 137 | 1.42× |
| PipeSD + Sequence-level | ✅ | Cumulative-seq | 139 | 1.40× |
| **PipeSD (Full)** | ✅ | **Dual-threshold** | **129** | **1.50×** |

> 🔍 发现：
- Pipeline 调度单独带来约 **1.18×** 加速；
- Dual-threshold 比单一机制更优（优于 token-level 和 sequence-level）；
- 两者结合实现最大增益（**1.50×**）。

此外，在更强的流水线 baseline 对比中（Appendix F），DP-based 批处理策略仍比 greedy/immediate-send 等高出 **1.02×–2.06×**，证明其必要性。

---

## 4. 关键结论和发现

### 主要发现
1. **流水线调度 + 双阈值触发是关键**：
   - Token-batch pipeline 显著隐藏通信延迟，尤其在边缘算力受限时效果更明显（Scenarios 2–3 提升更大）。
   - Dual-threshold NAV 触发机制实现了更好的 speculation 利用率与 error detection 平衡，PipeSD 的 acceptance rate 达 **96.16%**（高于 HSL 的 91.48% 和 EdgeLLM 的 89.17%）。

2. **BO autotuner 高效且实用**：
   - 仅需约 **16 次采样**即可收敛到近似最优阈值对 (R₁, R₂)。
   - 在 HumanEval 上 BO 达到 **129 ms TPT**，优于 grid search（139 ms）和 random search（148 ms）。

3. **系统开销极低**：
   - BO autotuner 开销 < **1.1%**
   - DP scheduler 开销 < **0.013%**
   - 参数测量开销 < **0.4%**
   > ⚠️ 控制平面额外能耗占总能耗不足 **1.513%**，可忽略不计。

4. **鲁棒性强**：
   - 在动态带宽环境下依然保持稳定加速（Scenario 4）；
   - 支持多客户端并发请求（Appendix I），在 2–8 客户端下 TPT 减少 **22.9%–37.1%**。

---

### 方法的局限性
1. **未支持 tree-based speculative decoding**：
   - 当前 PipeSD 基于 linear speculative decoding 构建；
   - Tree-based 方法虽潜力更大，但在云边场景下可能因高带宽消耗而不适用（见 Appendix J 讨论）。
2. **依赖边缘端准确估计 γ, α, β 等参数**：
   - 虽有滑动窗口监测机制，但在剧烈波动环境中可能存在滞后。
3. **目前仅验证了单目标模型架构**：
   - 多模型或多租户场景下的扩展性有待进一步研究。

---

### 未来工作方向
1. 将 PipeSD 扩展至 **tree-based speculative decoding** 框架，探索如何优化候选树传输与验证。
2. 在更多真实世界场景中验证鲁棒性，如 **异构硬件平台、多样化网络条件、长文本生成任务**。
3. 探索 **跨设备协作的 speculative inference**，例如多个边缘节点共同构建 draft sequence。
4. 进一步降低 BO autotuner 的采样次数，提升冷启动性能。

---

> ✅ **总结一句话**：  
> **PipeSD 通过 token-batch pipeline 调度与 dual-threshold NAV 触发机制，在真实云边环境中实现了高达 2.16× 的推理加速和 25.3% 的能耗降低，是当前最高效的 speculative decoding 协同推理框架之一。**

</details>

---

### 2. [D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models](https://arxiv.org/abs/2605.13276)

**Authors**: Yucheng Guo, Yongjian Guo, Zhong Guan, Wen Huang, Haoran Sun, Haodong Yue, Xiaolong Xiang, Shuai Di, Zhen Sun, Luqiao Wang, Junwu Xiong, Yicheng Gong  
**Category**: cs.AI  
**Published**: 2026-05-14  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2605.13276v1  

#### Abstract
The rapid evolution of Embodied AI has enabled Vision-Language-Action (VLA) models to excel in multimodal perception and task execution. However, applying Reinforcement Learning (RL) to these massive models in large-scale distributed environments faces severe systemic bottlenecks, primarily due to t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在大规模 **Vision-Language-Action (VLA)** 模型中应用 **Reinforcement Learning (RL)** 面临严重的系统瓶颈，主要源于以下冲突：
- **高保真物理仿真**（如机器人环境模拟）与 **深度学习训练** 对 GPU 资源的竞争。
- 仿真任务频繁进行小规模内存分配，导致 **GPU 内存碎片化**。
- 多模态数据（如高分辨率图像）在采样与推理组件间传输带来显著的 **通信延迟** 和 **序列化开销**。

这些因素共同导致整体吞吐量受限于最慢的仿真步进或同步开销，严重制约了 VLA 模型的可扩展性和训练效率。

### 提出了什么新方法或新思路
作者提出 **D-VLA** —— 一种面向大规模 VLA 模型的高性能分布式异步强化学习框架，其核心创新包括：

#### ✅ **Plane Decoupling 架构设计**
- 将高频的 **Data Plane**（数据交互平面）与低频的 **Control Plane**（权重控制平面）在物理上解耦。
- **Data Plane** 负责环境 rollout、观测采集等高频率操作。
- **Control Plane** 负责模型参数广播、梯度同步等低频但需强一致性的操作。
- 通过这种解耦，从根本上消除仿真与训练之间的资源争用。

#### ✅ **四线程异步 “Swimlane” 流水线**
- 设计了一个由四个并行线程构成的流水线机制：
  1. **Sampling Thread**：执行环境 rollout，生成轨迹数据。
  2. **Weight Reception Thread**：异步接收最新模型权重。
  3. **Training Execution Thread**：计算 GRPO 优势和策略梯度。
  4. **Parameter Distribution Thread**：将更新后的权重广播回 rollout 节点。
- 实现了 **计算与通信的完全重叠**，极大提升硬件利用率。

#### ✅ **Dual-Pool VRAM 管理模型**
- 将 GPU 显存划分为两个独立池：
  - **Model Computation Pool**：供 PyTorch 管理，用于存放模型权重和梯度。
  - **Environment Auxiliary Pool**：专供物理引擎（如 PhysX）使用，避免内存碎片影响主训练流程。
- 支持零拷贝数据交换（zero-copy data exchange），减少跨进程传输开销。

#### ✅ **拓扑感知复制与通信优化**
- 在多节点部署中采用 **local topology replication** 策略，在每个节点内构建闭环的采样-推理单元。
- 高频张量流动限制在本地高速互联（如 NVLink / InfiniBand）。
- 全局梯度聚合使用 **Fully Sharded Data Parallel (FSDP)**，而权重广播则卸载到基于 CPU 的 **Gloo 后端**，避免与 CUDA stream 冲突。

### 相比现有方法的优势
| 特性 | D-VLA | RLinf-VLA / RL-VLA3 |
|------|-------|---------------------|
| 架构解耦 | ✅ 物理级 Plane Decoupling | ❌ 控制与数据共用 GPU/CUDA |
| 异步程度 | 四线程全异步 Swimlane | 最多三阶段异步 |
| 显存管理 | Dual-Pool 防碎片 | 单一显存池易受干扰 |
| 通信设计 | 控制平面 CPU offload + 零拷贝 | 全 GPU 通信易阻塞 |
| 可扩展性 | 支持千亿参数稳定训练 | 大规模下出现通信瓶颈 |

> D-VLA 在架构层面解决了传统框架无法根除的“执行平面耦合”问题，实现了更高的并发度与更低的延迟。

---

## 2. 核心实验方法和设置

### 使用的数据集与仿真环境
- **ManiSkill**：一个基于 GPU 加速物理仿真的通用机器人操作基准，支持高并发环境实例。
- **LIBERO**：用于评估终身机器人学习中的知识迁移能力，包含多个任务类别（如 LIBERO-Object, LIBERO-Spatial）。

### 实验设置
- **模型架构**：
  - **T0.5**：基于扩散过程的 VLA 模型，迭代去噪生成动作序列。
  - **OpenVLA-OFT**：基于 Transformer 的自回归模型，采用 PEFT 微调。
- **硬件配置**：
  - 单节点与多节点测试均在 **16-GPU 集群** 上进行。
  - 使用 InfiniBand 进行节点间通信。
- **部署策略对比**：
  - **Co-located**（共置）：所有组件共享 GPU。
  - **Disaggregated**（分离）：rollout 与 actor 分配不同 GPU。
  - **Hybrid**（混合）：rollout 与 simulator 共享部分 GPU，actor 独占其余 GPU（D-VLA 默认策略）。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput (steps/s)** | 每秒处理的环境状态转移数（即动作推理步数） |
| **Step Time** | 单个 rollout 步骤的总耗时 |
| **Rollout Time / Actor Time** | 分别表示采样与训练阶段的时间消耗 |
| **Time Breakdown (%)** | 各阶段时间占比分析 |
| **Success Rate Curve** | 在 ManiSkill 上的任务成功率随训练步数的变化 |

### 基线方法对比
- **RLinf-VLA**：支持多种部署模式的通用 VLA + RL 框架。
  - 包括 colocated (`RLinf-co`)、disaggregated (`RLinf-dis`) 和 hybrid (`RLinf-hyper`) 三种变体。
- **RL-VLA3**：采用三阶段异步流水线（Environment → Rollout → Actor）的先进框架。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Figure 4–5）

#### 🔹 T0.5 模型表现（3:1 资源分配）
| 方法 | Throughput (steps/s) | 提升幅度 |
|------|------------------------|----------|
| RLinf-co | 175.29 | — |
| RL-VLA3 | 250.77 | +43% |
| **D-VLA (3:1)** | **376.00** | **+114% vs RLinf**, **+50% vs RL-VLA3** |

> 在 3:1 配置下达到峰值 **379 steps/s**，较基线有显著突破。

#### 🔹 OpenVLA-OFT 模型表现（1:1 资源分配）
| 方法 | Throughput (steps/s) | 提升幅度 |
|------|------------------------|----------|
| RLinf-co | 87.20 | — |
| RL-VLA3 | 170.48 | +96% |
| **D-VLA (1:1)** | **250.90** | **+188% vs RLinf**, **+47% vs RL-VLA3** |

> 表明 D-VLA 对参数量更大、推理更重的模型更具优势。

#### 🔹 延迟控制效果
- **T0.5 总 step time**：
  - D-VLA: **566.41 μs**
  - RLinf-dis: **1006.8 μs**
  > 减少超过 **44%** 的延迟。
- **OpenVLA-OFT step time**：
  - D-VLA: **520.3 μs**
  - 显著优于其他框架，尤其在高推理负载下仍保持高效。

### 与基线方法的对比结果
- D-VLA 在所有测试场景中均取得 **最高吞吐量** 和 **最低延迟**。
- 在 **多节点扩展性测试** 中，D-VLA 利用 InfiniBand 实现高效的跨节点通信，未出现明显的通信瓶颈。
- 在 **3,072 并发环境** 下仍能维持约 **360 steps/s** 的稳定吞吐，表现出优异的可扩展性。

### 消融实验与关键发现（Figure 7）
- **吞吐量随环境数量变化呈非线性增长**：
  - 从 384 → 768 环境：吞吐迅速上升至峰值 379 steps/s。
  - 超过 1,536 后：Actor 时间增长快于 Rollout，成为新瓶颈。
- **组件时间对齐至关重要**：
  - 当 Rollout Time ≈ Actor Time 时，异步流水线实现最佳“相互掩码”，吞吐最大化。
  - D-VLA 支持动态调整资源比例（如切换为 1:1）以恢复平衡。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Plane Decoupling 是解决 VLA 训练系统瓶颈的关键**：
   - 物理隔离 Data Plane 与 Control Plane 可有效消除仿真与训练间的资源冲突。
2. **异步流水线必须配合精细的资源调度才能发挥最大效能**：
   - 单纯增加异步层级不足以提升性能，**时间对齐** 才是决定吞吐上限的核心。
3. **Dual-Pool 显存管理显著缓解内存碎片问题**：
   - 物理引擎不再干扰主训练流程，提升了系统的稳定性与鲁棒性。
4. **D-VLA 在十亿级到万亿级参数模型上均展现出卓越的扩展性**：
   - 支持线性加速比，在大规模集群中仍保持高效运行。

### 方法的局限性
- **依赖特定硬件拓扑**：最优性能需要 InfiniBand/NVLink 等高速互连支持。
- **对极端不平衡负载适应性有限**：当 Actor 推理延迟远高于 Rollout 时，仍可能出现等待空转。
- **尚未验证在 multi-agent 场景下的有效性**：目前实验集中在单智能体设定。

### 未来工作方向
- **动态负载感知的资源再分配机制**：
  - 根据实时延迟自动调整 GPU 分配比例（如动态 1:1 ↔ 3:1 切换）。
- **扩展至 multi-agent 和 heterogeneous embodied platforms**：
  - 支持多个机器人协同训练，推动通才基础模型发展。
- **集成更复杂的 reward modeling 与 exploration 策略**：
  - 结合 LLM-based reward model 或 curiosity-driven exploration 提升样本效率。

---

> ✅ **总结一句话**：  
> D-VLA 通过 **Plane Decoupling + Swimlane 异步流水线 + Dual-Pool 显存管理**，在系统层面重构了 VLA 模型的分布式 RL 训练范式，实现了高达 **86% 的吞吐提升** 和 **近两倍的延迟降低**，为超大规模具身智能体训练提供了坚实的基础架构支持。

</details>

---

### 3. [Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models](https://arxiv.org/abs/2605.11854)

**Authors**: Kecheng Chen, Ziru Liu, Xijia Tao, Hui Liu, Yibing Liu, Xinyu Fu, Shi Wu, Suiyun Zhang, Dandan Tu, Lingpeng Kong, Rui Liu, Haoliang Li  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.11854v1  

#### Abstract
Diffusion Language Models (DLMs) have recently emerged as a promising alternative to autoregressive language models, offering stronger global awareness and highly parallel generation. However, post-training DLMs with standard Negative Evidence Lower Bound (NELBO)-based supervised fine-tuning remains...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Diffusion Language Models (DLMs) 在训练和推理之间存在显著的 **training-inference discrepancy**：
- **训练阶段**：采用标准的 NELBO 目标，随机均匀地掩码 token 并进行单步重建，隐含了“所有 token 难度相同”的**uniform inductive bias**。
- **推理阶段**：采用基于置信度或熵引导的多步去噪过程，遵循从易到难（easy-to-hard）的解码轨迹。

这种不匹配导致即使使用 self-distilled 轨迹进行 Supervised Fine-Tuning (SFT)，也仅能带来边际增益，甚至在全步长解码下性能下降。

此外，现有基于轨迹蒸馏的方法（如 dInfer、T3D）主要关注**采样加速**（few-step decoding），而非提升模型本身的生成能力。

---

### 提出了什么新方法或新思路
作者提出 **Trajectory-Aligned optimization via Boltzmann Modeling (TABOM)**，一种全新的 post-training 框架，旨在将 self-distilled 轨迹用于真正的知识获取，而不仅仅是推理加速。

#### 核心思想：
1. **建模推理分布为 Boltzmann 分布**  
   将 easy-to-hard 解码偏好形式化为一个关于理想预测熵 $H_{\text{ideal}}$ 的 Boltzmann 分布：
   $$
   q_{\text{infer}}(U|x_0) \propto \exp\left(-\beta \sum_{r \in U} H_{\text{ideal}}(x_r)\right)
   $$
   即低熵（高置信）token 更可能被优先 unmask。

2. **设计可优化的 Pairwise Ranking 损失**  
   由于直接最小化 KL 散度不可行（需计算配分函数），转而采用能量排序视角：
   - 对于任意两个 unmasked 集合 $U_a, U_b$，若 $q_{\text{infer}}(U_b) > q_{\text{infer}}(U_a)$，则应有模型能量得分 $S_\theta(U_b) > S_\theta(U_a)$。
   - 利用 self-distilled 轨迹中 token 的解码顺序构造正负样本对 $(r, s)$，其中先解码的 token $r$ 应具有更低的预测熵。

3. **局部窗口内的成对排序损失**  
   定义 hinge loss：
   $$
   \mathcal{L}_{\text{rank}} = \max(0, h_\theta(r;T) - h_\theta(s;T) + \gamma)
   $$
   其中 $h_\theta(r;T) = H_{\text{base}}(x_r | x, \text{step}_r)$ 是模型对 token $r$ 的预测熵。该损失强制模型学习正确的“不确定性排序”。

最终目标函数结合了轨迹感知的重建损失与排序损失：
$$
\mathcal{L}_{\text{TABOM}} = \mathbb{E}_{T \sim T_{\text{gold}}} \left[ \sum_{k \in \Delta_{t,t'}} -\log p_\theta(x_k | x, s) + \lambda \cdot \mathcal{L}_{\text{rank}} \right]
$$

---

### 相比现有方法的优势
| 方法 | 主要目的 | 是否提升生成质量 | 是否缓解遗忘 | 是否支持并行解码 |
|------|--------|------------------|---------------|------------------|
| SFT-GT | 提升 in-domain 性能 | ✅（有限） | ❌（严重遗忘） | ❌ |
| SFT-SD | 保留原始能力 | ❌（增益小） | ✅ | ✅ |
| dInfer / T3D | 加速 few-step 推理 | ⭕️（间接） | ⭕️ | ✅ |
| **TABOM (Ours)** | **对齐训练-推理分布** | ✅✅（显著提升） | ✅✅（完全缓解） | ✅ |

- TABOM 成功实现了 **“既提升领域内性能，又保持甚至增强 OOD 能力”** 的双重目标。
- 不依赖减少采样步数即可实现高效并行解码。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **数学推理**：`MixChain-Z-PRM12K`（12K 查询）
- **代码生成**：`Ling-Coder-SFT`（18K 查询）
- 自构建 self-distilled 数据：由 base model 使用 entropy/confidence-based decoding 生成完整轨迹（~3.8K–5.1K 条）

---

### 实验设置和评估指标

#### 模型基础
- **Base Models**：
  - `Dream-7B-Instruct`
  - `LLaDA-8B-Instruct`

#### 训练细节
- 使用 LoRA 进行参数高效微调（rank=16, α=16）
- 优化器：AdamW，学习率 2e-5，cosine 衰减，warm-up 50 步
- Batch size：每设备 4，共 8 GPUs → total 32
- Epochs：5，报告最佳 checkpoint 结果
- TABOM 参数：窗口大小 $W=32$，margin $\gamma \in \{0.1,0.2,0.3\}$，ranking weight $\lambda \in \{1,2\}$

#### 评估任务与指标
| 类别 | 任务 | 指标 |
|------|------|------|
| 数学推理 | GSM8K, MATH500 | 准确率 (%) |
| 代码生成 | HumanEval, MBPP | pass@1 (%) |
| 指令跟随 | IFEval | 指令准确率 (%) |

同时评估 **in-domain** 和 **out-of-distribution (OOD)** 表现。

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **No-SFT** | 原始 DLM，无微调 |
| **SFT-GT** | 使用 offline ground-truth 数据的标准 SFT |
| **SFT-SD** | 使用 self-distilled 轨迹的标准 SFT |
| **dInfer** | 学习压缩跳跃路径以加速推理 |
| **T3D** | 使用 direct discriminative optimization 和路径一致性加权 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Dream 模型，Code SFT）

| Method | HumanEval↑ | MBPP↑ | GSM8K↓ | MATH500↑ | IFEval↑ |
|--------|------------|-------|--------|----------|---------|
| No-SFT | 52.66 | 58.00 | 81.41 | 39.80 | 56.56 |
| SFT-GT | **61.55 (+8.89)** | 58.00 | **52.33 (-29.08)** | 32.40 | 46.21 |
| SFT-SD | 53.66 (+1.00) | 59.20 | 81.81 (+0.40) | 41.60 | 57.10 |
| TABOM (Ours) | **60.36 (+7.70)** | **60.60 (+2.60)** | 81.73 (+0.32) | **42.40 (+2.60)** | 55.45 |

> ✅ TABOM 在 in-domain 上接近 SFT-GT，但在 OOD 上几乎无损，综合表现最优。

---

### 数学推理任务（Dream 模型，Math SFT）

| Method | GSM8K↑ | MATH500↑ | HumanEval↓ | MBPP↓ |
|--------|--------|----------|-----------|--------|
| No-SFT | 81.41 | 39.80 | 52.66 | 58.00 |
| SFT-GT | 80.12 | 37.40 | 46.34 | 58.00 |
| SFT-SD | 81.95 | 39.80 | 57.92 | 58.60 |
| TABOM (Ours) | **84.31 (+2.90)** | **41.10 (+1.30)** | **58.54 (+5.88)** | **59.20 (+1.20)** |

> ✅ TABOM 显著提升 in-domain 性能，且反向提升了 OOD 表现，**完全避免 catastrophic forgetting**。

---

### 并行解码鲁棒性（2-token parallel decoding）

| Method | GSM8K↑ | MATH500↑ | HumanEval↑ | MBPP↑ |
|--------|--------|----------|-----------|--------|
| No-SFT | 74.37 | 31.60 | 43.29 | 43.60 |
| SFT-GT | 72.33 | 25.40 | 40.85 | 40.12 |
| TABOM | **77.79 (+3.42)** | **34.40 (+2.80)** | **45.73 (+2.44)** | **47.60 (+4.00)** |

> ✅ TABOM 在并行解码下仍保持高性能，表明其更符合实际推理动态。

---

### 消融实验（Ablation Study）

| 设置 | GSM8K | MATH500 | HumanEval | MBPP |
|------|--------|----------|-----------|--------|
| SFT-SD (Base) | 81.95 | 39.80 | 57.92 | 58.60 |
| + Traj Masking only | 82.18 | 41.20 | 56.45 | 58.70 |
| + Traj Masking + Global Ranking | 83.10 | 40.20 | 57.50 | 58.20 |
| + Traj Masking + **Local Ranking (W=32)** | **84.31** | **41.10** | **58.54** | **59.20** |

> 🔍 发现：
> - 仅使用 trajectory-aware masking 提升有限；
> - 添加 pairwise ranking 显著提升性能；
> - **全局排序引入噪声**（如比较首尾 token），**局部窗口 (W=32) 最优**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Self-distilled 轨迹本身不足以解锁性能增益**  
   即使来自模型自身分布，标准 NELBO 训练仍会破坏 easy-to-hard 结构。

2. **Explicit alignment 是关键**  
   必须通过显式机制（如 pairwise ranking）将模型的 entropy landscape 与推理轨迹对齐。

3. **TABOM 实现了“鱼与熊掌兼得”**  
   - 大幅提升 in-domain 性能（+2.90 on GSM8K）
   - 完全避免 catastrophic forgetting
   - 扩展 OOD 知识边界（如 HumanEval 反升 +5.88）

4. **Trajectory Discrimination Score (TDS) 验证机制有效性**  
   - TABOM 的 TDS 显著高于 baselines（如 Dream 上从 0.035 → 0.711 on HumanEval）
   - 表明模型在解码时能更好地区分“易”与“难” token

---

### 方法的局限性
- **依赖高质量 self-distilled 轨迹**：若 base model 本身解码不稳定，生成的轨迹可能含有错误。
- **超参数敏感性**：虽然 grid search 可找到较优配置，但 margin $\gamma$ 和 $\lambda$ 需谨慎选择。
- **未探索 zero-shot 或 prompt tuning 场景**：当前聚焦于 full fine-tuning。

---

### 未来工作方向
- 将 TABOM 思想扩展至 **autoregressive models** 中的 step-wise reasoning alignment。
- 探索 **multi-agent self-distillation**，利用多个 agent 生成更丰富的推理轨迹。
- 结合 **reinforcement learning**，进一步优化 long-horizon 生成质量。
- 构建通用的 **trajectory alignment library**，适配不同 DLM 架构。

---

> 💡 **一句话总结**：  
> TABOM 首次将 self-distilled 轨迹视为“正确生成过程”的示范，而非仅仅是训练数据，并通过 Boltzmann 建模 + pairwise ranking 实现训练与推理的深度对齐，在不牺牲泛化能力的前提下显著提升 DLM 的生成质量。

</details>

---

### 4. [BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion](https://arxiv.org/abs/2605.11577)

**Authors**: Shaobin Zhuang, Yuang Ai, Jiaming Han, Xiaohui Li, Huaibo Huang, Xiangyu Yue, Xuefeng Hu, Kun Xu, Yali Wang, Hao Chen  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.11577v1  

#### Abstract
Autoregressive language models generate text one token at a time, yet natural language is inherently structured in multi-token units, including phrases, n-grams, and collocations that carry meaning jointly. This one-token bottleneck limits both the expressiveness of the model during pre-training and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统自回归语言模型（Autoregressive Language Models, AR LLMs）采用“逐个token生成”的范式，即通过 `softmax` 输出层对词汇表中的下一个 token 进行分类预测。这种模式存在两个核心瓶颈：
- **表达能力受限**：自然语言的基本单位是短语、n-gram 或搭配，而非孤立的 token，逐 token 预测难以建模联合语义。
- **推理效率低下**：由于生成过程本质上是串行的，导致推理延迟高，难以实现高效并行。

尽管已有工作尝试通过 speculative decoding、multi-token prediction 等方式加速，但这些方法通常是在不改变输出接口的前提下进行优化，未能从根本上突破“token-level 分类”这一范式限制。

---

### **提出了什么新方法或新思路**
论文提出 **BitLM**，一种全新的语言模型架构，其核心思想是：
> **将 token 生成从“大词汇表上的分类任务”重构为“固定长度二进制码空间中的连续扩散去噪任务”。**

具体创新包括：

- **Binary Token Interface**  
  每个 token ID 被映射为一个固定长度的 B-bit 二进制码（如 B=18），形成一个位于 $\{-1, +1\}^B$ 超立方体顶点的“binary code”。该编码是固定的、非学习的，仅作为符号标识。

- **Bitwise Continuous Diffusion Head**  
  引入轻量级 diffusion head，在连续向量空间中对多个未来 token 的 binary codes 同时进行迭代去噪。去噪目标是从加噪状态恢复原始 binary code。

- **Block-Causal Transformer Backbone**  
  主干网络仍为标准 Transformer，但使用 block-causal attention mask：块内允许全连接，块间保持因果依赖。这使得模型可以在保留左到右推理逻辑的同时，实现块级并行生成。

- **分离式架构设计**  
  - **Backbone**：负责上下文理解与推理（contextual reasoning）
  - **Diffusion Head**：负责符号实现（symbolic realization）——即把隐状态“结晶”为离散 token 序列

这一设计将生成过程从“逐位置分类决策”转变为“结构化对象的联合实现”。

---

### **相比现有方法的优势**
| 维度 | BitLM | 传统 AR 模型 | Speculative Decoding / Multi-Token Heads |
|------|-------|-------------|----------------------------------------|
| 输出接口 | Binary space denoising | Vocabulary softmax | Vocabulary softmax |
| 并行性来源 | 原生 blockwise 扩散 | 外部解码策略 | 外部 proposal/verification |
| 因果结构 | 显式保留（block-causal） | 完全保留 | 依赖主模型 |
| 推理速度潜力 | 高（每步生成 m 个 token） | 低（串行） | 中等（依赖 draft 模型质量） |
| 架构改动深度 | 深层输出层重构 | 无 | 浅层附加头 |

> ✅ **优势总结**：
> - 不牺牲因果建模可靠性；
> - 实现原生多 token 并行生成；
> - 将生成几何从 simplex 分类转为 binary hypercube 上的流形优化；
> - 更适合高效预训练与快速推理。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Pretraining**: 子集 of **FineWeb-350BT**（约 350B tokens），用于大规模语言建模训练。
- **Fine-tuning & Evaluation**: **XSum** 新闻摘要数据集，用于评估下游任务性能。

---

### **实验设置**
- **模型架构基础**：
  - Backbone：基于 **Qwen-3** 架构
  - Diffusion Head：借鉴 **BitDance** 设计
- **关键参数**：
  - Binary code length $ B = 18 $
  - Block size $ m = 4 $（每次生成 4 个 token）
  - 训练序列长度：16384 tokens（packed 多样本）
  - Optimizer：AdamW ($lr=1e^{-4}, \beta_1=0.9, \beta_2=0.95$)
- **推理配置**：
  - Denoising steps $ K = 15 $
  - 使用 ODE solver 进行连续扩散采样
  - Classifier-Free Guidance (CFG) = 9.0

---

### **评估指标**
- **ROUGE-1 / ROUGE-2 / ROUGE-L**：用于衡量生成摘要与参考摘要之间的重叠程度。
- **归一化 ROUGE**：相对于最强 baseline 的相对得分。
- **消融实验变量**：
  - Diffusion step 数量（5 ~ 100）
  - CFG 值（1 ~ 15）
  - 是否使用 diffusion head vs. 传统 LM head

---

### **基线方法对比**
- **经典摘要模型**：
  - ILead-3（首三句基准）
  - PTGen / PTGen+Cov（Pointer-Generator Network）
- **BitLM 变体**：
  - BitLM w/ LM Head（softmax 输出）
  - BitLM w/ Diffusion Head（本文方法）
- 所有模型均为 8B 参数规模，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（XSum）**

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|---------|---------|---------|
| ILead-3 | 16.30 | 1.60 | 11.95 |
| PTGen (See et al., 2017) | 29.70 | 9.21 | 23.24 |
| BitLM 8B w/ LM Head (FT) | 23.20 | 4.45 | 18.04 |
| **BitLM 8B w/ Diff. Head (FT)** | **26.05** | **6.44** | **20.12** |

> 🔍 **说明**：
> - BitLM 在 fine-tuning 后达到 **26.05 ROUGE-1**，显著优于自身使用 softmax 头的版本（+2.85），也超过预训练 strong baseline。
> - 虽未超越 PTGen（可能是任务适配不足），但已证明 binary diffusion 范式的可行性。

---

### **与基线方法的对比结果**
- 相比传统 softmax 输出的 BitLM 版本，**diffusion head 提升明显**（↑~2–3 ROUGE 点），表明 binary diffusion 更有利于符号实现。
- 在相同 backbone 下，**diffusion 输出优于直接分类输出**，验证了“去噪比分类更适合多 token 联合生成”的假设。
- 模型在不同 block size 下可灵活调整并行度，**m=4 时即可有效训练与推理**。

---

### **消融实验结果**
#### （1）Denoising Steps 与 CFG 影响（Fig. 4）
- 最优性能出现在：
  - **K = 15 步去噪**
  - **CFG = 9.0**
- 过少步骤导致去噪不充分；过多则收益递减。
- CFG 提供更强的条件控制，提升生成一致性。

#### （2）模型可扩展性（Fig. 3）
- 成功训练了 **0.6B, 1.7B, 4B, 8B** 四种规模的 BitLM。
- 随着模型增大，**pretraining loss 单调下降**，显示良好缩放性（scalability）。
- 表明该架构无需特殊设计即可支持大模型训练。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **One-token-at-a-time 不是必须的**：语言生成可以脱离逐 token softmax 范式，转向结构化对象的联合实现。
2. ✅ **Binary space 是有效的生成空间**：将 token 表示为 fixed binary code，并在其上执行连续扩散，是一种可行且高效的替代路径。
3. ✅ **Blockwise Parallel Generation 可以是原生的**：通过 block-causal + diffusion 的组合，**并行生成成为模型本身的生成模式，而非后处理技巧**。
4. ✅ **Output Layer 几何至关重要**：论文揭示了一个被忽视的设计维度——**symbolic output space 的几何结构**，直接影响生成动态与效率。

---

### **方法的局限性**
- 当前在 XSum 上的表现仍低于最先进的 pointer-generator 模型，说明：
  - 对精确词汇复制（copying）、指针机制的支持尚弱；
  - fine-tuning 策略可能需要进一步优化（如引入 alignment loss）；
- Binary code 是固定映射，未探索 learned binary codebook 的潜力；
- Block size 固定为 4，尚未实现 adaptive block sizing；
- 推理速度实测数据未报告（仅理论推测更快）。

---

### **未来工作方向**
- 🔄 **Adaptive Block-Causal Scheduling**：根据内容复杂度动态调整 block size。
- 🔀 **Hybrid Softmax-Binary Architectures**：结合 softmax 的局部精度与 diffusion 的全局协调能力。
- 🧠 **Learned Binary Codebooks**：让 binary code 具备语义结构，而不仅是 ID 映射。
- ⚡ **Efficient Inference Benchmarking**：量化实际吞吐量提升（tokens/sec），验证推理加速效果。
- 🌐 **Extension to Multimodal & Token-Free Settings**：与 ByT5、MEGABYTE 等 byte-level 模型结合，构建统一 binary interface。

---

## **总结**
> **BitLM 提出了一种根本性的语言生成新范式：用 binary space 中的连续扩散取代 vocabulary softmax，实现了原生的多 token 并行生成。它不仅挑战了“语言模型必须输出 token 分布”的传统认知，也为下一代高效、强表达力的语言模型架构开辟了新路径。**

虽然当前性能仍有提升空间，但其理念上的突破意义重大——**我们或许正在从“next-token predictors”迈向“structured continuation realizers”。**

</details>

---

### 5. [Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference](https://arxiv.org/abs/2605.11581)

**Authors**: Wenxin Dong, Mingqing Hu, Guanghui Yu, Qiang Fu, Peng Xu, Hui Xu, Yue Xing, Xuewu Jiao, Shuanglong Li, Lin Liu  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.11581v1  

#### Abstract
When large language models (LLMs) serve real-time inference in commercial online advertising systems, end-to-end latency must be strictly bounded to the millisecond range. Yet every token generated during the decode phase triggers thousands of kernel launches, and kernel launch overhead alone can ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在商业在线广告系统等**超低延迟场景**中，大语言模型（LLM）推理的端到端延迟必须控制在毫秒级。然而，传统 GPU 推理框架存在以下瓶颈：
- **Kernel Launch Overhead**：解码阶段每生成一个 token 都会触发数千次 kernel 启动，其开销占端到端时间的 **14.6%**。
- **HBM 内存往返延迟**：多个算子间通过全局内存（HBM）交换中间结果，造成严重带宽浪费。
- **资源受限硬件适配难**：NVIDIA Ada 架构（如 L20）缺乏 TMA 硬件支持、共享内存仅为 H100 的一半（128KB vs 227KB），导致 MegaKernel 难以部署。

现有 MegaKernel 方案面临两难：
- 手工调优方案（如 Stanford MegaKernel）性能高但**不具可移植性**；
- 自动编译方案（如 Mirage MPK）引入运行时分支判断，破坏指令流水效率。

---

### 🚀 提出的新方法与创新点

#### （1）**自适应共享内存管理（Adaptive Shared Memory Management）**
- 建立三维共享内存约束模型：综合考虑 **硬件规格、模型架构、动态工作负载**。
- 引入 **K-dimension 细粒度分裂**，将每次迭代所需权重 tile 减半，**峰值共享内存需求降低 50%**。
- 实现跨算子页面复用策略：
  - **Activation-Weight Page Reuse**：激活加载进寄存器后释放空间用于权重预取。
  - **Activation-Output Page Reuse**：激活页重用于 MMA 输出缓冲，提升资源利用率。

> 💡 效果：在 Ada 架构上重建高效流水线，突破共享内存限制。

---

#### （2）**基于 MLIR 的细粒度 DAG 离线搜索优化**
- 利用 **MLIR Lowering** 技术将高级 IR 分解为 PTX 级依赖图（DAG），捕捉更精细的并行机会。
- 构建包含 **数据依赖** 和 **资源竞争依赖** 的完整 DAG 模型。
- 在离线阶段进行自动化搜索与路径固化：
  - 搜索最优的 Warp 角色分配（Loader/Consumer/Storer/Launcher）。
  - 固化执行路径，**完全消除运行时 if-else 分支决策**。

> 💡 对比 Ansor、Pruner 等传统 Auto-Tuning 框架，Ada-MK 在 DAG 层面操作，能处理复杂不规则依赖。

---

#### （3）**异构混合推理引擎（Heterogeneous Hybrid Inference Engine）**
- 将 MegaKernel 作为插件嵌入 **TensorRT-LLM**，实现分阶段调度：
  - **Prefill 阶段**：使用 TensorRT-LLM 原生融合算子，保持高吞吐。
  - **Decode 阶段**：切换至 MegaKernel 引擎，消除 launch 开销与内存瓶颈。
- 支持零成本复用已有业务能力（如 prefix-tree decoding、generation-discrimination）。

> ✅ 成果：首次实现 MegaKernel 在工业级在线广告系统的规模化落地。

---

### 🔍 相比现有方法的优势

| 方法 | 缺陷 | Ada-MK 如何改进 |
|------|------|----------------|
| **Stanford MegaKernel** | 仅支持 Hopper/Blackwell；硬编码 tile 大小；无 Qwen 支持 | 跨架构可移植；自动参数搜索；支持主流模型 |
| **Mirage MPK** | 运行时动态分支影响指令发射效率 | 离线搜索 + 路径固化，彻底移除 runtime branching |
| **vLLM / SGLang** | 侧重 KV Cache 管理与请求调度，底层 kernel 融合不足 | 底层算子深度融合，显著降低小批量延迟 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **固定短序列任务**：`input=64`, `output=12`（模拟低延迟短文本生成）
- **真实任务数据集**：
  - **CSL 数据集**：中文科学文献摘要，上下文长度集中在 ~200–1000 tokens。
  - **Human-eval 数据集**：代码生成任务，测试模型编程能力。

---

### ⚙️ 实验设置

| 项目 | 配置 |
|------|------|
| **硬件平台** | 单台服务器：<br>- CPU: Intel Xeon Platinum 8558 (96核)<br>- GPU: **NVIDIA L20** (Ada 架构, 48GB GDDR6, 共享内存 128KB/SM)<br>- 内存: 1TB DDR5 |
| **软件环境** | Linux 5.10, CUDA 12.2, Docker 容器隔离 |
| **评估模式** | **离线批处理模式（offline batch mode）**：一次性提交固定批次请求，测量总生成吞吐量，避免调度器干扰 |
| **量化配置** | GPTQ-W4A16（权重量化为 4-bit，激活保持 16-bit） |
| **测试模型** | - Qwen3-1.7B<br>- Qwen2.5-1.5B |

---

### 📊 评估指标
- **主指标**：**Generation Throughput (tokens/s)** —— 数值越高越好
- 辅助分析：端到端延迟、Pipeline Duty Cycle、共享内存占用

---

### 🆚 基线方法对比
| 框架 | 版本 | 特点 |
|-------|--------|------|
| **vLLM** | v0.19.0 | 高吞吐 LLM 推理框架，PagedAttention 优化 KV Cache |
| **SGLang** | v0.5.10 | 结构化生成高性能服务框架 |
| **TensorRT-LLM (vanilla)** | v1.1.0rc5 | NVIDIA 官方推理引擎，作为基础 baseline |
| **Ada-MK (ours)** | — | TensorRT-LLM + MegaKernel 插件 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 场景 | 性能表现 |
|------|----------|
| **固定短序列 (in64/out12)** | - 相比 vanilla TensorRT-LLM 最高提升 **23.6%**（BS=1）<br>- 相比 vLLM 最高提升 **50.2%** |
| **CSL 数据集** | - BS=1~8：Ada-MK 表现最佳<br>- BS=16：vLLM 反超 3.5%，体现其系统级调度优势 |
| **Human-eval 数据集** | - 即使在 BS=16 下仍保持最高吞吐<br>- 相比 vanilla TensorRT-LLM 提升 **19.5%** |
| **跨模型一致性** | 在 Qwen3-1.7B 与 Qwen2.5-1.5B 上均取得稳定增益，验证通用性 |

---

### 📊 详细对比结果（节选）

#### 固定短序列（Qwen3-1.7B, BS=1）
| 方法 | Throughput (tokens/s) | Speedup vs TRT-LLM |
|------|------------------------|--------------------|
| vLLM | ~2.1K | ~1.18x |
| SGLang | ~2.3K | ~1.24x |
| TensorRT-LLM | ~2.5K | 1.00x |
| **Ada-MK (ours)** | **~3.1K** | **1.236x** |

> ✅ 小批量下优势最明显，适合低延迟在线服务。

---

#### Human-eval 数据集（Qwen3-1.7B, BS=16）
| 方法 | Throughput (tokens/s) | Speedup vs TRT-LLM |
|------|------------------------|--------------------|
| vLLM | ~4.8K | ~1.13x |
| SGLang | ~4.6K | ~1.11x |
| TensorRT-LLM | ~4.2K | 1.00x |
| **Ada-MK (ours)** | **~5.0K** | **1.195x** |

> ✅ 在长输入、大批量代码生成任务中依然领先。

---

### 🔍 消融实验与归因分析（来自 Section 4.2.3）

| 优化维度 | 性能增益来源 | 提升幅度（Decode 阶段） |
|---------|-------------|--------------------------|
| **依赖解耦优化** | - 异步预取 RMS Norm 权重<br>- KV-Cache 加载提前并行化<br>- SwiGLU 中 Reduce 流水化 | ≈15% |
| **自动调优配置收敛** | - Consumer Warps 从 16→8<br>- Pipeline Stage 从 2→4<br>- 页面原位复用优化 | ≈15% |
| **合计** | —— | **≈30%**（相比原始 MegaKernel） |

> 📌 特别地，通过减少 Consumer Warps 并扩展 Pipeline Stage，有效对齐各角色延迟，抑制流水线气泡。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **MegaKernel 显著提升小批量推理效率**  
   - 在 BS=1 场景下，Ada-MK 相比 vanilla TensorRT-LLM 提升达 **23.6%**，相比 vLLM 提升 **50.2%**。
   - 核心原因：**消除 kernel launch overhead** 与 **深度 I/O-Compute 重叠**。

2. **Ada-MK 在多种任务和模型上具备泛化能力**  
   - 在 Qwen3-1.7B 与 Qwen2.5-1.5B 上均取得正向收益。
   - 在 CSL 和 Human-eval 等真实数据集上保持稳定性。

3. **“离线搜索 + 在线复用”范式优于运行时自适应**  
   - 在确定性部署配置下，最优执行路径是固定的，无需 runtime 决策。
   - 采用 **DAG-level 离线搜索 + 路径固化**，避免了 MPK 类框架的分支惩罚。

4. **异构混合设计兼顾 Prefill 与 Decode 性能**  
   - Prefill 是 compute-bound，TensorRT-LLM 更优；
   - Decode 是 IO-bound，MegaKernel 发挥优势；
   - 二者结合实现 **高吞吐 + 低延迟** 的统一目标。

---

### ⚠️ 方法的局限性

1. **在高并发长序列场景下优势缩小**
   - 当 batch size ≥16 且 context 较长时（如 CSL），vLLM/SGLang 凭借更强的 **请求调度与 KV Cache 管理机制** 反超。
   - 表明：**单靠 kernel 融合不足以应对系统级挑战**。

2. **当前主要适用于 GPTQ-W4A16 量化模型**
   - 虽然支持 INT4 量化，但未全面覆盖 AWQ、Marlin 等其他格式。

3. **依赖 MLIR 工具链成熟度**
   - MLIR Lowering 与 Alias Analysis 的准确性直接影响 DAG 构建质量。

---

### 🔮 未来工作方向（原文 Section 6）

1. **向更大规模模型迁移**  
   - 探索 Ada-MK 在百亿参数以上模型中的适用性。

2. **适配下一代 Blackwell 架构 GPU**
   - 利用更新的硬件特性（如增强 TMA、更大 shared memory）进一步释放潜力。

3. **扩展至更多模型结构与任务类型**
   - 支持 MoE、Vision-Language 模型等新型架构。

4. **探索全自动端到端编译流程**
   - 减少人工干预，实现从 PyTorch 模型到 MegaKernel 插件的一键生成。

---

> ✅ **总体评价**：  
> Ada-MK 是首个成功将 **MegaKernel 技术工业化落地于商业在线广告系统** 的实践。它通过 **自适应共享内存管理 + DAG 级离线搜索 + 异构混合执行引擎** 的三重创新，在资源受限的 Ada 架构上实现了显著性能突破，尤其适用于 **低延迟、小批量、短输出** 的典型在线推理场景。

</details>

---

### 6. [EMO: Frustratingly Easy Progressive Training of Extendable MoE](https://arxiv.org/abs/2605.13247)

**Authors**: Linghao Jin, Chufan Shi, Huijuan Wang, Nuan Wen, Zhengzhong Liu, Eric Xing, Xuezhe Ma  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.13247v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) models offer a powerful way to scale model size without increasing compute, as per-token FLOPs depend only on k active experts rather than the total pool of E experts. Yet, this asymmetry creates an MoE efficiency paradox in practice: adding more experts balloons memo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**EMO: Frustratingly Easy Progressive Training of Extendable MoE**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题：**MoE 效率悖论（MoE Efficiency Paradox）**

尽管 **Sparse Mixture-of-Experts (MoE)** 模型理论上实现了计算量（FLOPs）与模型参数解耦——即每 token 的计算仅依赖激活的 $k$ 个专家而非总数 $E$，但在实际训练中，随着专家数量 $E$ 增加，以下系统开销显著上升：

- **All-to-all 通信成本**
- **Optimizer state 内存占用**
- **小规模 GEMM 导致 GPU 利用率低**

这导致虽然理论 FLOPs 不变，但 **wall-clock time 随 $E$ 显著增长**（如图1所示，在激活参数不变时，$E$ 从8增至128，step time 上升1.72×），形成所谓的“效率悖论”。

> ❓ 核心问题：**训练初期是否真的需要一开始就使用大量专家？**

---

### 🆕 提出的新方法：**EMO（Extendable Mixture-of-Experts）**

提出一种**渐进式扩展专家池的训练框架**，将 MoE 容量视为可扩展的“参数化内存”（parametric memory），在训练过程中逐步增加专家数量。

#### 核心思想：
- 起始于一个小型或密集模型（dense 或 small-$E$ MoE）
- 在多个阶段中逐步扩展专家池（e.g., $8 \to 16 \to 32 \to 64 \to 128$）
- 每次扩展后继续训练新增的数据

#### 创新点：
1. **首次将专家数量 $E$ 视为动态变量**，而非固定配置。
2. 引入 **sparsity-aware scaling law** 来指导每个阶段应分配多少 token 数据，实现 compute-optimal 的 token 分配。
3. 设计了简单有效的 **expert expansion 和初始化策略**（如 Gaussian 初始化 + router bias reset）。

---

### 🔍 相比现有方法的优势

| 方法类别 | 局限性 | EMO 的优势 |
|--------|------|-----------|
| 固定 $E$ 的 MoE 训练 | 早期高 $E$ 带来巨大通信与内存开销 | 早期使用小 $E$，大幅降低训练成本 |
| 单步 upcycling（如 Komatsuzaki et al., 2023） | 一次性复制权重到所有专家，难以高效利用大 $E$ | 多阶段渐进扩展，更平滑、更高效 |
| 系统级优化（如 FastMoE, MegaBlocks） | 降低单次 large-$E$ 开销，但仍全程承受其代价 | **延迟 large-$E$ 的引入时机**，从根本上减少总开销 |

> ✅ **核心优势总结**：  
> EMO 在几乎不损失最终性能的前提下，**显著提升 wall-clock 效率，节省 GPU 小时数（实测节省 10%）**，提供了一条“简单却极其有效”的可扩展 MoE 训练路径。

---

## 2. 核心实验方法和设置

### 📚 数据集

- **预训练数据**：混合语料，包括：
  - Web 文本（Slimpajama, C4, Falcon-RW）
  - Code（Dolma）
  - 数学（MetaMath）
  - 多语言文本
  - 社交平台内容（Reddit, Twitter, Gab 等）
- **总 token 预算**：**1.92T tokens**
- **上下文长度**：8192

> 图9展示了具体数据比例分布。

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| 架构 | Decoder-only Transformer |
| 总层数 | 16 |
| Hidden dim | 2048 |
| MoE 层 | 每层含共享专家 + 路由专家池 |
| Top-k | 8 |
| 激活参数量（N_act） | 1.1B（保持恒定） |
| 扩展路径 | 五阶段：$E = 8 \to 16 \to 32 \to 64 \to 128$ |
| 扩展依据 | 基于 sparsity-aware scaling law 的 token 分配策略 |
| 学习率调度 | Warm-Stable-Decay（WSD），最终阶段线性衰减 |
| 优化器 | AdamW ($\beta_1=0.9, \beta_2=0.95$)，weight decay=0.05 |
| Batch size | 8M tokens |
| 总步数 | 240K steps |

---

### 🎯 评估指标

| 类型 | 指标 |
|------|------|
| **上游任务** | Pre-training loss, Validation perplexity |
| **下游任务** | 多项基准准确率（Accuracy %）：<br>- 推理：GSM8K<br>- 知识：MMLU, TriviaQA, NQ<br>- 常识推理：HellaSwag, PIQA, SIQA, Winograd<br>- 科学问答：ARC-C/E, OBQA<br>- 阅读理解：RACE<br>- 因果推理：COPA |
| **效率指标** | Wall-clock time, GPU hours, step time |

---

### 🆚 基线方法对比

| 基线 | 描述 |
|------|------|
| **FIXED_E=16 / 32 / 128** | 从头开始训练，固定专家数，其他超参一致 |
| **Single-step upcycling** | 作为背景参考，非主要对比对象 |

> 所有模型共享相同的非-expert 主干网络和训练流程，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

| 模型 | Final Pre-train Loss | GPU Hours 节省 | 下游综合表现 |
|------|------------------------|----------------|--------------|
| **EMO (progressive)** | **1.017** | **节省 ~10%** | 与 FIXED_E=128 相当 |
| **FIXED_E=128** | 0.994（略优） | 基准 | 最强基线 |
| **FIXED_E=64** | 1.065 | — | 明显落后 |
| **FIXED_E=16** | 更差 | — | 显著弱于 EMO |

> 💡 尽管 FIXED_E=128 在 loss 上略胜（相对差距 2.3%），但 EMO 凭借更低的训练成本实现了接近的性能。

---

### 📈 与基线方法的对比结果

#### ▶️ 图7：训练 loss 曲线对比
- EMO 经历阶段性 loss spike（每次扩展时），但均能在约 10K 步内恢复并持续下降。
- 最终 loss 接近 FIXED_E=128，远优于 FIXED_E=16/32。

#### ▶️ 图8 & 表2：下游任务表现
- 在大多数任务上（如 MMLU, HellaSwag, ARC, PIQA, TriviaQA），**EMO 与 FIXED_E=128 表现相当甚至略有领先**。
- 仅在 **GSM8K** 上稍弱，表明复杂推理能力可能受益于全程 large-$E$ 结构组织。

> ✅ **Takeaway**：EMO 成功回收了 large-$E$ 的大部分收益，同时避免了全程 high-cost。

---

### 🔬 消融实验结果

#### （1）**Token 分配策略验证**（Section 3.2 & Figure 5）

测试不同扩展时机（25%, 50%, 75% 总数据处）对 E=16→32 的影响：

| 扩展时机 | 最终 loss | 分析 |
|---------|----------|------|
| 25% | 1.069 | 最佳性能，但训练时间最长 |
| 50% | 1.071 | 性能接近，时间节省明显 |
| 75% | 1.076 | 时间最短，但性能下降快 |

> ✅ Scaling law 预测的最佳点 (~45%) 正好落在性能-成本权衡的“平坦区”，**兼顾质量与效率**。

#### （2）**Expert 初始化策略**（Figure 10 & Section 5.1）

比较三种方式：
- Gaussian init
- Gaussian + router bias reset
- Copy from old checkpoints

> 🔍 发现：
- 所有策略最终收敛到相似 loss
- 差异主要体现在 **initial spike 大小**
- “bias reset” 虽然 spike 较大，但实现简单且不影响长期性能 → 被选为主方案

#### （3）**Optimizer State 处理**
- 是否继承旧状态影响极小，500 步 warmup 后差异消失
- 因此选择 **reset optimizer states** 以简化实现

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **MoE 容量无需从一开始就是最大的**  
   → 将专家视为“可扩展内存”，按需增长是可行且高效的。

2. **渐进式扩展可在几乎无损性能下大幅提升训练效率**  
   → EMO 达到与 FIXED_E=128 相当的性能，**节省 10% GPU hours**。

3. **Sparsity-aware scaling law 可有效指导 token 分配**  
   → 实验验证该策略位于 quality-cost trade-off 的最优区域。

4. **EMO 对初始化鲁棒，易于实现**  
   → 多种初始化方式均可收敛，工程实现门槛低。

5. **专家利用率良好，无 collapse 现象**  
   → 尽管部分新专家训练时间短，但路由分布仍较均衡（Gini ≈ 0.5 vs baseline 0.44）

---

### ⚠️ 方法的局限性

1. **Scaling law 未建模优化超参的影响**  
   → 当前拟合未考虑 LR、batch size 等变化，未来可进一步精细化。

2. **尚未在最大规模 MoE 系统上验证**  
   → 实验规模小于当前最前沿 MoE 模型（如 >1T 参数级别）。

3. **某些推理密集型任务（如 GSM8K）仍有差距**  
   → 表明某些能力可能需要更早暴露于 large-$E$ 结构中进行协同学习。

---

### 🔮 未来工作方向

1. **将 EMO 扩展至更大规模 MoE 模型**（e.g., 万亿参数级）
2. **结合 instruction tuning 或 RL 微调阶段的 progressive MoE 扩展**
3. **探索自动化的动态扩展策略**（基于 loss plateau 或 routing entropy 自适应触发）
4. **研究 expert specialization 动态演化过程**，分析知识如何在扩展中迁移与分化

---

## ✅ 总结一句话

> **EMO 提出了一种“渐进式扩展专家”的 MoE 训练范式，通过将 MoE 容量视为可扩展内存，并基于 sparsity-aware scaling law 进行 compute-optimal 的分阶段训练，在几乎不牺牲性能的前提下显著提升了训练效率，为大规模 MoE 模型的经济化训练提供了简单而强大的新路径。**

</details>

---

### 7. [Training-Inference Consistent Segmented Execution for Long-Context LLMs](https://arxiv.org/abs/2605.11744)

**Authors**: Xianpeng Shang, Jiang Li, Zehua Duo, Qianyi Cai, Xiangdong Su  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.11744v1  

#### Abstract
Transformer-based large language models face severe scalability challenges in long-context generation due to the computational and memory costs of full-context attention. Under practical computation and memory constraints, many inference-efficient long-context methods improve efficiency by adopting ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Training-Inference Consistent Segmented Execution for Long-Context LLMs**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Transformer-based LLMs 在处理长上下文（long-context）时面临严重的可扩展性挑战，主要原因在于 full-context attention 的计算和内存开销呈二次增长。为提升推理效率，许多方法在 **inference 阶段**采用受限执行策略（如 bounded-context、chunked attention），但在 **training 阶段仍使用 full-context attention**，导致训练与推理之间存在 **execution 和 state-transition 语义不一致**。

这种不一致性会导致模型在训练中依赖推理时不可用的信息，从而损害长上下文下的稳定性与泛化能力。

---

### **提出的新方法与核心思想**
本文提出一种 **训练-推理一致的分段执行框架（Training-Inference Consistent Segmented Execution）**，其核心思想是：

- 将序列划分为非重叠的 segments，在训练和推理中均遵循相同的 **segment-level forward 执行语义**。
- 引入两个跨段输入：
  - **Carried KV state**：仅有的可微分状态接口，用于传递最近一个 segment 的 KV 缓存（长度固定）。
  - **Retrieved KV prefix**：通过前向检索获得的历史 KV，**不参与梯度传播**（forward-only）。
- 在训练中使用 **Truncated Backpropagation Through Time (TBPTT)**，限制梯度仅回传最多 $ K $ 个 segment，确保梯度路径与推理时的状态链完全对齐。

该方法将跨段信息流分解为：
- **局部连续通道（Local continuity channel）**：通过 carried KV 实现可微分状态传递。
- **前向长程条件通道（Forward-only long-range conditioning）**：通过检索实现长距离依赖建模，但不引入额外梯度路径。

---

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **训练-推理一致性** | 显式对齐训练与推理的执行语义，避免因执行模式差异导致的性能下降。 |
| **理论保证** | TBPTT 在该设定下计算的是 **inference-consistent objective 的精确梯度**，而非近似值。 |
| **高效性** | 显著降低 prefill 阶段的峰值内存和延迟，尤其在超长上下文（如 128K）下表现优异（~6× 内存降低）。 |
| **可扩展性** | 支持 zero-shot 长度外推，在超过训练长度的 context 下仍保持稳定性能。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **PG19**：用于评估不同 context length 下的语言建模 perplexity。
- **LongBench**（及子集 LongBench-E）：多任务长上下文理解基准，涵盖问答、摘要、代码生成等任务。
- **RULER**：系统性测试长度泛化的合成任务，包括：
  - **CWE**（Common Words Extraction）
  - **FWE**（Frequent Words Extraction）

---

### **实验设置与评估指标**
| 设置项 | 描述 |
|--------|------|
| **模型骨干** | 主要基于 LLaMA2-7B-32K 和 LLaMA2-7B-80K；部分实验使用 LLaMA3.1-8B-Instruct。 |
| **分段长度** | 固定 segment 长度 $ S = 4096 $，carried KV 长度 $ M = 512 $，检索前缀长度 $ R = 512 $。 |
| **TBPTT 深度** | 默认 $ K=1 $（即最多回传 1 个 segment），对应最大训练 context 8K。 |
| **评估指标** | 
| - **Perplexity**（越低越好） | 
| - **LongBench 平均得分**（越高越好） | 
| - **Prefill latency**（首 token 时间） | 
| - **Peak GPU memory**（prefill 阶段峰值显存） |

---

### **基线方法对比**
| 基线方法 | 类型 |
|----------|------|
| **Vanilla Self-Attention** | 全注意力，无优化 |
| **StreamingLLM** | 推理阶段滑动窗口 + sink tokens |
| **MInference** | 推理阶段稀疏 attention |
| **CCA (Core Context Aware)** | 训练-推理对齐，压缩 context |
| **DuoAttention** | 分离 retrieval 和 streaming heads |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **语言建模 perplexity（PG19）**
- 在 4K–64K context 下，本文方法的 perplexity 增长平缓，无剧烈波动。
- 相比之下，StreamingLLM、MInference 等方法在 context 超出训练范围后出现显著性能下降甚至崩溃。
- **64K 时，baseline 多数失败（score = 0），而本文方法仍保留非零准确率**。

#### ✅ **LongBench-E 性能（平均得分）**
| 方法 | LLaMA2-32K | LLaMA2-80K |
|------|------------|------------|
| Vanilla Self-Attention | 23.13 | 23.38 |
| StreamingLLM | 21.90 | 21.56 |
| CCA | 21.12 | 21.98 |
| DuoAttention | 23.00 | 22.94 |
| **Ours** | **23.24** | **24.17** ✅ |

> 在 80K 上，**首次突破 24 分**，尤其在 multi-document QA 和 summarization 上显著领先。

#### ✅ **RULER 长度泛化能力**
| 方法 | CWE Avg* (4K–32K) | FWE Avg* | 64K 是否存活 |
|------|------------------|---------|-------------|
| Vanilla | 32.94 | 41.33 | ❌ |
| StreamingLLM | 27.78 | 41.37 | ❌ |
| DuoAttention | 32.33 | 43.42 | ❌ |
| **Ours** | **46.39** ✅ | **43.88** ✅ | ✅（2.00 / 34.17）|

> 表明本文方法在极端长度下仍具备一定推理能力。

#### ✅ **效率指标（Prefill 阶段）**
- **128K context 下，peak prefill memory 比 full attention 降低约 6×**。
- 在 64K context 下，**prefill latency 与 peak memory 均优于所有 baseline**（见 Figure 6）。
- 实现更优的 **latency-memory trade-off**。

---

### **消融实验结果**

#### 🔍 **训练-推理对齐的影响**
| 方法 | LongBench-E Avg |
|------|----------------|
| **Aligned (TBPTT=1)** | **24.17** ✅ |
| Misaligned（训练用 full attention） | 11.91 ❌ |
> 对齐至关重要，错误的训练目标导致性能腰斩。

#### 🔍 **TBPTT 深度 $ K $ 的影响**
| $ K $ | Avg Score |
|-------|-----------|
| 1 | 24.17 ✅ |
| 2 | 24.07 |
> $ K=1 $ 已足够，更深的回传未带来收益，反而轻微下降。

#### 🔍 **局部状态容量（Local KV size）**
| Size | Perplexity (avg) | LongBench Avg |
|------|------------------|---------------|
| 0 | 7.16 | 23.27 |
| 512 | 7.10 | 24.17 ✅ |
| 1024 | 7.07 | 24.19 |
> 适度增加 local state 有益，但边际效应递减。

#### 🔍 **长程模块层数**
| 层数 | LongBench Avg |
|------|---------------|
| 0 | 22.63 |
| 2 | 22.44 |
| 4 | **24.17** ✅ |
> 更多 long-range layers 提升跨段推理能力。

---

## **4. 关键结论和发现**

### **主要发现**
1. **训练-推理一致性是长上下文建模的关键**：即使架构相同，若训练与推理执行语义不一致，也会导致严重性能退化。
2. **TBPTT 可以精确优化 inference-consistent objective**：在严格限制跨段递归的情况下，截断反向传播不再是近似，而是精确梯度。
3. **分离“可微状态传递”与“前向长程检索”是有效设计**：既能控制梯度流，又能支持远距离依赖建模。
4. **$ K=1 $ 的 TBPTT 已足够**：无需复杂的历史状态回传，简单的一段状态携带即可达到最优性能。

---

### **方法的局限性**
- **依赖人工设计的 long-range head 分组**：当前使用 prior-based grouping，未来可探索自动化选择机制。
- **检索模块未参与训练**：forward-only 设计虽保证一致性，但也限制了端到端优化潜力。
- **persistent retrieval pool 内存随长度线性增长**：虽然不影响 attention memory，但仍需管理长期存储。

---

### **未来工作方向**
- 自动识别并动态分配具有 retrieval behavior 的 attention heads。
- 探索轻量级、可训练的 retrieval 模块，在保持一致性的同时允许有限参数更新。
- 将该框架扩展至 encoder-decoder 架构和多模态 LLMs。
- 结合硬件优化（如 PagedAttention）进一步提升吞吐量。

---

> 📌 **一句话总结**：  
> 本文提出了首个从训练到推理完全一致的分段执行框架，通过 **controlled interface state + forward-only retrieval + exact TBPTT gradient**，实现了高效、稳定且可扩展的长上下文建模，在性能、效率和鲁棒性上全面超越现有方法。

</details>

---

### 8. [Efficient LLM-based Advertising via Model Compression and Parallel Verification](https://arxiv.org/abs/2605.11582)

**Authors**: Wenxin Dong, Chang Gao, Guanghui Yu, Xuewu Jiao, Mingqing Hu, Qiang Fu, Peng Xu, Penghui Wei, Hui Xu, Yue Xing, Shuanglong Li, Lin Liu  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.11582v1  

#### Abstract
Large language models (LLMs) have shown remarkable potential in advertising scenarios such as ad creative generation and targeted advertising. However, deploying LLMs in real-time advertising systems poses significant challenges due to their high inference latency and computational cost. In this pap...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Efficient LLM-based Advertising via Model Compression and Parallel Verification》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于 Large Language Models（LLMs）的在线广告系统面临两大挑战：
- **高推理延迟**（inference latency），难以满足实时广告投放需求；
- **计算成本高昂**，限制了大规模部署。

尤其是在生成式广告定向（Generative Targeting）场景中，模型需在极短时间内完成从用户查询到广告生成的全过程，对效率要求极高。

### 提出的新方法与创新思路
作者提出了一套完整的高效 LLM 广告框架，结合 **Model Compression** 和 **Prefix Tree-based Parallel Verification** 两大核心技术：

#### 主要创新点：
- ✅ **Index-Compressed 2bit-CSR 结构**  
  改进传统的 Compressed Sparse Row（CSR）格式，设计仅需 2bit 的索引压缩结构，将索引和权重总体积减少至原始的 **30%**，显著降低内存开销。

- ✅ **自研 SparseGemv 加速内核（Kernel）**  
  针对 INT4 权重量化 + 半结构化稀疏矩阵乘法（semi-structured sparsity）开发专用加速 Kernel，填补了 NVIDIA cuSparse/cuSparseLT 在 GEMV 场景下的空白。

- ✅ **Adaptive Group-Wise Quantization（自适应分组量化）**  
  根据不同层参数敏感度动态调整量化粒度：敏感层采用细粒度（更多组）、非敏感层粗粒度（更少组），实现精度与效率的最佳平衡。

- ✅ **Layer-wise Semi-Structured Sparsity（逐层半结构化剪枝）**  
  对 Transformer 各层按重要性进行差异化 N:M 剪枝（如关键层保留 2:4 密度，次要层用 1:4），优化 GEMV 负载。

- ✅ **前缀树约束的并行验证机制（Prefix Tree Parallel Verification, PTPV）**  
  首次将 **prefix tree-constrained decoding** 与 **beam search** 结合用于广告生成任务。通过构建语义前缀树，并动态判断何时启动并行验证，大幅缩短解码步数。

- ✅ **动态触发机制（Dynamic Parallel Verification Trigger）**  
  实时评估“生成剩余 token”与“并行验证”的时间差，在收益最大时切换模式，兼顾速度与准确性。

### 相比现有方法的优势
| 维度 | 本文方法优势 |
|------|-------------|
| **效率** | 推理速度提升超过 **78%**（最高达 1.89× speedup），生产环境实测提速超 1.8× |
| **精度保持** | 在 Recall、BLEU、Meteor 等指标上损失极小，优于纯剪枝或量化方案 |
| **硬件适配性** | 自定义 Kernel 更好支持 INT4 + 稀疏混合计算，无需依赖通用库 |
| **业务契合度** | 针对广告场景定制优化（如高频商业实体优先保留），更适合工业落地 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **专有商业流量数据集**（内部数据）  
  用于 **Targeted Advertising** 实验，包含真实线上广告请求与点击反馈，因敏感未公开。
  
- **CSL 数据集**（Chinese Scientific Literature Dataset）  
  用于 **Ad Creative Generation** 实验，含约 39.6 万篇中文科技论文元数据，适用于创意改写与关键词摘要任务。

### 实验设置
- **主干模型**：ERNIE 1.5B（百度自研 LLM）
- **训练框架**：PaddlePaddle
- **硬件平台**：
  - Ad Creative Generation：NVIDIA A10 GPU，beam size = 1
  - Targeted Advertising：NVIDIA A30 GPU，beam size = 20（需输出多个候选广告）
- **优化流程**：逐步集成 Quantization → Sparsity → PTPV 进行消融分析

### 评估指标
| 任务 | 主要指标 |
|------|---------|
| **Targeted Advertising** | Latency（延迟）、Recall（召回率） |
| **Ad Creative Generation** | BLEU、Meteor（文本生成质量）、Per-token latency（每 token 延迟） |
| **综合性能** | Speedup（相对于 FP16 基线的加速比） |

### 基线方法对比
- **Baseline (FP16)**：全精度浮点模型，无任何压缩
- 其他变体作为中间对照：
  - Quantization-only（INT4 自适应分组量化）
  - Sparsity-only（2:4 或 1:4 层级剪枝）
  - Sparse + Quant（剪枝+量化组合）
  - 最终完整版：Sparse + Quant + PTPV

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table 1 与 Figure 2–3）

#### 🔹 完整框架性能（Targeted Advertising）
| 方法 | Recall | Per-token Latency (ms) | Speedup |
|------|--------|------------------------|--------|
| Baseline (FP16) | 60.0% | 1.9 | ×1.00 |
| +Quantization | 60.0% | 1.22 | ×1.56 |
| +Sparse+Quant | 60.0% | 1.0 | ×1.90 |
| +Sparse+Quant+PTPV | 60.0% | **0.81** | **>×2.3**（推算） |

> 注：图中显示最终方案延迟降至 0.81ms，相比基线提速 **137%+**

#### 🔹 创意生成任务表现（Ad Creative Generation）
| 方法 | BLEU | Meteor | Avg Length | Per-token Latency (ms) | Speedup |
|------|------|--------|------------|------------------------|--------|
| Baseline (FP16) | 0.4247 | 0.6345 | 17.5 | 6.6 | ×1.00 |
| Quantization | 0.4178 | 0.6283 | 17.6 | 4.8 | ×1.37 |
| Sparse(2:4)+Quant | 0.4103 | 0.6195 | 17.5 | 4.0 | ×1.65 |
| **Sparse(Mix)+Quant** | **0.4038** | **0.6127** | 17.5 | **3.7** | **×1.78** |

✅ **结论**：混合稀疏 + 量化仍能保持接近基线的语言质量，同时实现近 **1.8× 推理加速**

### 消融实验结果（Ablation Study）
| 技术组合 | Latency (ms) | BLEU | Meteor | Speedup |
|--------|--------------|------|--------|--------|
| Baseline (FP16) | 6.6 | 0.4247 | 0.6345 | ×1.00 |
| Quantization | 4.8 | 0.4178 | 0.6283 | ×1.37 |
| Sparsity (2:4) | 5.3 | 0.4161 | 0.6260 | ×1.25 |
| Sparsity (1:4) | 4.6 | 0.3476 | 0.5549 | ×1.43 |
| Sparse(2:4)+Quant | 4.0 | 0.4103 | 0.6195 | ×1.65 |
| Sparse(1:4)+Quant | 3.5 | 0.3369 | 0.5446 | ×1.89 |
| **Sparse(Mix)+Quant** | **3.7** | **0.4038** | **0.6127** | **×1.78** |

📌 发现：
- 单独量化效果最好，延迟下降明显且精度损失最小；
- 强剪枝（1:4）虽快但严重损害生成质量；
- **混合稀疏策略（mix）在速度与质量间取得最佳平衡**，适合实际部署。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **模型压缩 + 并行验证可有效解决 LLM 在广告场景中的效率瓶颈**  
   通过 INT4 自适应量化 + 层级稀疏剪枝 + 自定义 Kernel，显著降低计算负载。

2. ✅ **Prefix Tree Parallel Verification 显著减少解码步数**  
   动态触发机制确保只在最优时机启用并行验证，避免无效计算，提升整体吞吐。

3. ✅ **工业级部署可行性强**  
   该系统已成功上线百度广告平台，服务于大规模实时流量，验证了其稳定性与实用性。

4. ✅ **精度与效率可以兼得**  
   尽管进行了深度压缩，但在 Recall、BLEU、Meteor 等关键指标上损失控制在可接受范围内。

### 方法的局限性
- ❗ 当前优化高度依赖于特定业务场景（如广告 ID 分布、高频词先验知识），泛化能力受限；
- ❗ 前缀树构造依赖于静态聚类算法（DSI），对动态更新内容响应较慢；
- ❗ 自定义 Kernel 开发门槛高，迁移至其他硬件平台需重新适配；
- ❗ 仅适用于生成式推荐任务，不直接适用于传统判别式排序模型。

### 未来工作方向
- 🔄 引入 **Reinforcement Learning** 或 **Adaptive Algorithms** 实现动态压缩策略选择；
- 🔁 探索 **在线增量更新机制**，使前缀树和剪枝策略能随数据分布变化自适应调整；
- 🌐 扩展至更多应用场景（如新闻推荐、电商搜索），检验通用性；
- ⚙️ 进一步融合 **Speculative Decoding** 思想，探索多头预测 + 树形 attention 架构优化。

---

> ✅ **总结一句话**：  
> 本论文提出了首个面向广告生成场景的端到端高效 LLM 推理框架，通过 **adaptive quantization + semi-structured sparsity + prefix tree parallel verification** 三重优化，在保持精度的同时实现 **>1.8× 实际加速**，已在百度广告平台成功落地应用。

</details>

---

### 9. [PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory for Long-Horizon Agents](https://arxiv.org/abs/2605.12260)

**Authors**: Jingyi Peng, Zhongwei Wan, Weiting Liu, Qiuzhuang Sun  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.12260v1  

#### Abstract
Long-horizon language agents accumulate conversation history far faster than any fixed context window can hold, making memory management critical to both answer accuracy and serving cost. Existing approaches either expand the context window without addressing what is retrieved, perform heavy ingesti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory for Long-Horizon Agents 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在**long-horizon language agents**（长周期语言智能体）中，对话历史积累速度远超固定上下文窗口容量，导致内存管理成为影响答案准确性和服务成本的关键瓶颈。现有方法存在以下不足：
- **扩展上下文窗口**：未解决“检索什么”这一核心问题；
- **重写时事实提取**（ingestion-time fact extraction）：token 成本高；
- **启发式图遍历**（heuristic graph traversal）：牺牲准确性和效率。

PRISM 针对这一挑战，提出了一种无需训练、仅在检索侧优化的框架，旨在实现**高准确率**与**低上下文成本**的帕累托最优（Pareto-efficient）平衡。

---

### 提出了什么新方法或新思路
PRISM 是一个**training-free retrieval-side framework**，将长周期记忆建模为一个**联合检索与压缩问题**，运行于已构建的图结构化记忆（graph-structured memory）之上。其核心由四个正交的推理时组件构成：

1. **Hierarchical Bundle Search**（分层束搜索）  
   在**typed relation paths**（带类型的关系路径）上进行搜索，而非基于表面相似性，从而支持多跳、因果等复杂查询。

2. **Query-Sensitive Edge Costing**（查询敏感边代价）  
   根据检测到的查询意图动态调整不同类型边的遍历代价，使检索更符合语义意图（如 `why` 查询优先走 `causal` 边）。

3. **Evidence Compression**（证据压缩）  
   通过一次 LLM 调用对候选证据包进行重排序与压缩，生成紧凑的 answer-side context，显著降低 token 开销。

4. **Adaptive Intent Routing**（自适应意图路由）  
   将大多数查询通过零 LLM 调用层级（关键词匹配、原型嵌入）处理，仅在必要时调用 LLM 进行意图分类，进一步控制成本。

> ✅ **创新亮点**：首次将**意图感知的路径检索**与**LLM-side 压缩**结合，且完全无需训练或修改上游 ingestion pipeline。

---

### 相比现有方法的优势
| 维度 | PRISM | 现有方法 |
|------|-------|----------|
| **准确性** | 显著更高（0.831 LLM-judge score） | 多数在 0.6–0.7 区间 |
| **上下文成本** | ~2K tokens/query（约全上下文的 1/13） | 多数 >3K，Full Context 达 26K |
| **效率** | Per-1K 效率达 0.411，最优 | 最接近的 Mem0 为 0.379 |
| **训练依赖** | 完全无训练（training-free） | 多需 fine-tuning 或 RL 训练 |
| **模块解耦** | 可插拔，兼容任意后端 | 通常与特定 ingestion 强耦合 |

PRISM 占据了以往空白的“**高准确 / 低开销**”象限，在 accuracy-context-cost 前沿实现了突破。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LoCoMo** [14]：一个专为评估长周期对话记忆设计的 QA 基准测试。
  - 包含 10 场多轮对话，共 1,540 个 QA 对。
  - 分类涵盖：单跳（single-hop）、多跳（multi-hop）、时间推理（temporal）、开放域（open-domain）。
  - 排除第5类（对抗性拒绝），聚焦检索能力。

---

### 实验设置和评估指标

#### 主要设置
- **Answer & Judge Model**：默认使用 `gpt-4o-mini`（temperature=0.0）。
- **Tokenizer**：统一使用相同 tokenizer 和 token 计算方式，确保上下文预算可比。
- **协议一致性**：所有“same-protocol”方法共享相同的 answer prompt、judge prompt 和 ingestion checkpoint。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **LLM-judge score** | 正确率 = CORRECT / (CORRECT + WRONG)，由 LLM 自动评判生成答案是否正确 |
| **Context tokens per query** | 每次查询传给 answer model 的平均 token 数，衡量检索成本 |
| **Per-1K efficiency** | judge score / (context_tokens / 1000)，即每千 token 收益，反映性价比 |

---

### 基线方法对比
| 类型 | 基线方法 | 特点 |
|------|--------|------|
| **Full-context baseline** | Full Context | 将全部 ~26K tokens 输入模型 |
| **Graph-based retrieval** | MAGMA, Mem0, Mem09 | 利用图结构进行检索 |
| **Memory management** | M-Flow, ReSum | 通过摘要、分层等方式减少上下文 |
| **Commercial platform** | Mem0 platform | 商业托管版本，非开源 |

> 所有 same-protocol 基线均在相同条件下复现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | LLM-judge Score | Context Tokens/Query | Per-1K Efficiency |
|------|------------------|------------------------|--------------------|
| **Full Context** | 0.481 | 26,031 | 0.018 |
| **MAGMA** | 0.688 | 3,370 | 0.204 |
| **Mem0** | 0.669 | 1,764 | 0.379 |
| **Mem09** | 0.684 | 3,616 | 0.189 |
| **PRISM (ours)** | **0.831** | **2,023** | **0.411** |

> ✅ PRISM 在保持极低上下文（仅为 Full Context 的 **1/13**）的同时，取得了 **+35.2 pp** 的绝对提升。

---

### 与基线方法的对比结果
- **全面超越所有 same-protocol 基线**：
  - 比 Mem09 高 **+14.2 pp**，比 MAGMA 高 **+14.0 pp**。
  - 在所有子类别（multi-hop, temporal, open-domain, single-hop）均排名第一。
- **优于更强模型下的不同协议参考**：
  - **M-Flow** 使用更强的 `gpt-5-mini`，但仍以 **0.818 < 0.831** 落败。
  - **PRISM + gpt-5.5** 可达 **0.891**，说明残差错误更多来自 answer model 而非 retrieval。
- **商业平台对比**：
  - Mem0 platform 达 0.916，但消耗 ~7K tokens（约 PRISM 的 3.4 倍），且 per-1K 效率仅 0.131（PRISM 为 0.411）。

> 🔺 PRISM 在“相同协议”下首次实现了“高准确 + 低开销”的组合。

---

### 消融实验结果（Ablation Study）

#### 主要消融配置（Table 2）
| 配置 | Judge Score | ER@5 | Context Tokens |
|------|-------------|--------|----------------|
| **PRISM (完整)** | 0.831 | 0.694 | 2,023 |
| **-N1 (无关系路径)** | 0.831 | 0.694 | 2,024 |
| **-N2 (无边代价调整)** | 0.831 | 0.694 | 2,020 |
| **-N3 (无 LLM 重排)** | 0.825 | 0.627↓ | 4,108↑ |
| **+N4 (启用意图路由)** | 0.833 | 0.694 | 2,023 |

#### 关键发现
- **N3（Evidence Compression）是主导因素**：
  - 移除后 context token **翻倍**（2K → 4.1K）。
  - ER@5 下降 **6.8 pp**，表明其有效过滤无关证据。
  - 准确率轻微下降，说明其主要作用是**提升证据质量分布**而非直接决定最终答案。
- **N1/N2 在 LoCoMo 上影响有限**：
  - 因为 LoCoMo 中 72% 的多跳问题是“anchor-discoverable”（可通过锚点直接检索），真正需要桥接路径的问题仅占 ~3%。
  - 预期在 MuSiQue、HotpotQA 等更难的多跳数据集上会更显著。
- **N4（Adaptive Intent Routing）节省成本无损精度**：
  - 42.3% 的查询无需 LLM 调用（关键词或原型匹配）。
  - 与完整 LLM 路由相比，准确率无统计显著差异（△=+0.26 pp, p=0.71）。

---

## 4. 关键结论和发现

### 主要发现
1. **检索质量 > 上下文长度**：即使模型能处理长上下文，**“检索什么”比“检索多少”更重要**。Full Context 表现最差即证明此点。
2. **联合检索与压缩是关键**：单独做图检索或压缩都无法达到最优；PRISM 通过**路径检索 + LLM 压缩**的组合实现了帕累托前沿突破。
3. **Evidence Compression 是 token 节省的核心杠杆**：它不仅是压缩器，更是**精准过滤器**，决定了 top-k 证据的质量。
4. **意图感知提升检索效率**：Query-Sensitive Edge Costing 使检索路径更贴合语义意图，尤其在因果、时间类问题中表现优越。
5. **无需训练即可实现高性能**：PRISM 完全无需 fine-tuning 或策略学习，即可超越多个需训练的方法。

---

### 方法的局限性
- **依赖高质量图结构记忆**：PRISM 假设已有 schema-guided 构建的图结构记忆（如 Entity → FacetPoint → Episode），不负责 ingestion。
- **在强锚点数据集上优势受限**：在 LoCoMo 这类“anchor-discoverable”为主的数据集中，N1/N2 的增益不明显。
- **未处理动作轨迹记忆**：当前聚焦于对话记忆，尚未扩展至包含工具调用、观察、计划等 agent 轨迹的记忆管理。
- **潜在偏见传播风险**：忠实检索上游记忆中的内容，若 ingestion 存在偏见或隐私信息，可能被反射到输出中。

---

### 未来工作方向
- 扩展至 **agent action trajectories** 的记忆管理（如工具使用、反馈循环）。
- 在 **MuSiQue、HotpotQA** 等更具挑战性的多跳数据集上验证 N1/N2 的有效性。
- 探索 **SEMANTIC edges** 的激活条件（目前在 LoCoMo 中关闭）。
- 结合 **KV-cache compression** 等长上下文优化技术，进一步降低成本。
- 加强 **ingestion-side filtering** 机制，防止偏见与隐私泄露。

---

> 📌 **总结一句话**：  
> **PRISM 通过“意图感知的路径检索 + LLM 证据压缩”，在无需训练的前提下，实现了长周期 agent 记忆系统中前所未有的高准确率与低上下文成本的平衡，开辟了 retrieval-side 优化的新范式。**

</details>

---

### 10. [TurboGR: An Accelerated Training System for Large-Scale Generative Recommendation](https://arxiv.org/abs/2605.13433)

**Authors**: Huichao Chai, Zhixin Wu, Xuemiao Li, Shiqing Fan, Hengfeng Wang, Maojun Peng, Lu Xu, Yaoyuan Wang, Yibo Jin, Wei Guo, Yongxiang Feng  
**Category**: cs.DC  
**Published**: 2026-05-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.13433v1  

#### Abstract
Generative recommendation (GR) has emerged as a promising paradigm that replaces fragmented, scenario-specific architectures with unified Transformer-based models, exhibiting scaling-law behavior where recommendation quality improves systematically with increased model capacity and training data. Ho...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TurboGR: An Accelerated Training System for Large-Scale Generative Recommendation 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对在 **Ascend NPU** 上部署大规模生成式推荐（Generative Recommendation, GR）模型时面临的三大系统级瓶颈：

1. **Jagged Variable-Length Sequences**：用户行为序列长度不一，导致大量填充（padding），引发严重的计算冗余和向量计算瓶颈，使得 **Model FLOPs Utilization (MFU)** 不足 10%。
2. **Sparse-Dense Communication Bottleneck**：稀疏嵌入表（sparse embedding tables）与密集 Transformer 主干之间的强耦合造成通信开销巨大，分布式训练的扩展性低于 0.6。
3. **Memory-Intensive Negative Sampling**：长序列下的 token-level 负采样消耗过多 HBM 内存，使大规模召回训练不可行。

此外，Ascend NPU 缺乏对 jagged 操作的高性能实现，且其架构偏向稠密矩阵运算，加剧了上述挑战。

---

### 提出的新方法与创新点

为解决上述问题，作者提出 **TurboGR** —— 一种面向 Ascend NPU 的生成式推荐加速训练系统，包含三大核心技术：

#### （1）Ascend-affinity Jagged Acceleration
- **Jagged Fusion Operators**：将 Attention 和 Relative Attention Bias (RAB) 中的操作融合，原生支持 jagged 张量，消除 dense-jagged 转换开销。
  - 成果：内存减少 **70%**，延迟降低 **2.2×**。
- **Jagged Embedding Lookup Acceleration**：基于 KeyedJaggedTensor (KJT)，去除无效索引处理，并通过表级别重组织提升缓存命中率。
  - 成果：前向延迟降低 **6×**。
- **Dynamic Load Balancing**：
  - *Token-Aware Dynamic Batch Scaling*：按 token 数而非样本数动态调整 batch size。
  - *Global Token Reallocation*：跨设备全局排序并均衡分配高 token 数样本。
  - 成果：设备间负载差异从 **47% 降至 2.4%**。

#### （2）Distributed Communication Optimization
- **Hierarchical Sparse Parallelism (HSP)**：
  - 将设备划分为多个组，每组内部分割嵌入表（model parallelism），组间复制全表（data parallelism）。
  - 所有-to-all 通信限制在组内，显著降低通信规模。
  - 成果：all-to-all 延迟降低 **75.9%**。
- **Semi-Asynchronous Training**：
  - 稀疏模块异步执行，解除前后批次间的依赖，允许稀疏前向提前执行。
  - 理论证明收敛性不受影响。
- **Fine-Grained Pipeline Orchestration**：
  - 设计六阶段细粒度流水线（dataloader → feature all-to-all → embedding forward → dense module → embedding backward），实现 CPU/NPU/通信操作的高度重叠。
  - 成果：NPU 利用率达 **94%**。

#### （3）Negative Sampling Optimization
- **Asynchronous Offloading**：
  - 将负样本嵌入张量异步卸载至 CPU 内存，分段加载回 NPU 处理，采用双缓冲机制隐藏传输延迟。
  - 成果：HBM 使用最多减少 **24.59%**（128 负样本）。
- **Jaggedness-aware FP16 Quantization**：
  - 对负样本嵌入查找路径启用 FP16 半精度量化，仅对负样本触发，不影响主流程。
  - 成果：精度损失可忽略（如 HR@2000 差异 <0.01%）。
- **Intra-batch Logit Sharing**：
  - 复用同一批次中其他 token 的负样本 logits 作为辅助负样本，扩大有效负空间而无需额外 embedding 查找。
  - 配合 token-level shuffle 减少冗余。
  - 成果：可在更少原始负样本下达到相同或更高性能。

---

### 相比现有方法的优势

| 维度 | 现有方案（如 TorchRec + Megatron） | TurboGR |
|------|-------------------------------|--------|
| 架构适配性 | GPU 友好，未针对 Ascend 优化 | 全栈 Ascend-affinity 设计 |
| Jagged 支持 | 存在 padding 冗余，无融合算子 | 原生 jagged 支持，消除冗余 |
| 通信效率 | All-to-all 跨所有设备，扩展差 | HSP 分层并行，通信降阶 |
| 流水线设计 | 粗粒度同步，NPU 利用低 | 六阶段细粒度流水，利用率 >94% |
| 负采样内存 | 全驻留 HBM，易 OOM | 异步卸载 + 量化 + logit 共享 |
| 开源状态 | 多数为 GPU 生态 | **首个开源的 Ascend GR 系统** |

---

## 2. 核心实验方法和设置

### 数据集
- **KuaiRand-27K**：目前最大、最广泛采用的快手短视频推荐数据集子集。
  - 包含超过 27,000 用户，千万级 item，亿级交互记录。
  - 特征丰富：观看时长、点击、点赞、关注、完播率等。
  - 序列长：平均远超其他变体，适合评估长序列建模能力。

### 实验设置
- **硬件平台**：
  - Ascend 910B1 NPU（64GB HBM），集群规模从 32 到 128 NPUs（4–16 节点）。
  - Kunpeng-920 ARM CPU。
- **模型**：
  - HSTU [12] 和 FuXi [14] 两类主流 GR 模型。
  - 多种尺度：tiny, small, medium, large, long（sequence length 达 4096）。
- **训练配置**：
  - 优化器：AdamW，学习率 4e-3，无 weight decay。
  - 精度：TF32 + 异步 embedding 更新。
  - 负采样：128 negatives（默认）。
  - 数据划分：chronological leave-one-out。

### 评估指标
| 指标 | 描述 |
|------|------|
| **MFU (Model FLOPs Utilization)** | 实际利用的浮点算力占峰值算力的比例，衡量硬件效率 |
| **Throughput (sample/s)** | 每秒处理的样本数 |
| **End-to-End Latency** | 单步训练总耗时 |
| **Communication Overhead** | 各类通信操作所占时间比例 |
| **HR@K / NDCG@K** | 推荐质量指标（Hit Rate, Normalized Discounted Cumulative Gain） |
| **Scalability (Linearity)** | 扩展效率，理想为 1.0 |

### 基线方法对比
- **Baseline**：标准 TorchRec + Megatron 实现，直接迁移到 Ascend 平台。
- **对比维度**：
  - 端到端训练速度
  - MFU
  - 通信开销占比
  - 内存占用（HBM）
  - 分布式扩展性（linearity）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | Model Size (M) | Seq Len | Comp. Complexity (TFLOPs/step) | Throughput (sample/s) | **MFU (%)** | **Linear Scalability** |
|-------|----------------|---------|-------------------------------|------------------------|-------------|------------------------|
| HSTU-large | 83.97 | 2048 | 4.33 | 1616.83 | 24.74 | 0.93 |
| HSTU-long | 83.97 | 4096 | 7.15 | 770.79 | **34.08** | **0.97** |
| FuXi-large | 201.55 | 2048 | 8.25 | 1156.26 | **39.34** | **0.94** |
| **FuXi-long** | **201.55** | **4096** | **11.54** | **574.39** | **54.71** | **0.97** |

> ✅ **最高达到 54.71% MFU**，接近线性扩展（0.97），是当前 GR 系统中的领先水平。

---

### 与基线方法的对比结果

#### （1）Jagged Fusion Operator 效果（Figure 2）
- 在 8k 序列长度下：
  - 延迟从 **961.21ms → 431.13ms**（↓55%）
  - 内存占用从 **47.77GB → 14.31GB**（↓70%）

#### （2）Embedding Lookup 加速（Table 2）
- 百万 ID 查询（50.43% 无效填充）：
  - 前向延迟：**18ms → 3ms**（↓6×）
  - 反向延迟：**36ms → 9ms**（↓4×）

#### （3）HSP 通信优化（Table 4）
- all-to-all 延迟：**498ms → 120ms**（↓75.9%）
- 总通信延迟：**613ms → 373ms**（↓39.1%）

#### （4）Semi-Async 训练（Table 5）
- 稀疏通信占比从 **24.12% → 2.19%**
- 推荐性能保持甚至略有提升（如 HR@10 ≈ 0.0306 vs 0.0295）

#### （5）Negative Sampling Offloading（Table 7）
| #Negatives | HBM Usage (Baseline) | HBM Usage (Offloading) | ↓Reduction |
|------------|------------------------|--------------------------|-----------|
| 32 | 22.21 GB | 17.42 GB | 7.3% |
| 64 | 31.64 GB | 23.44 GB | 12.51% |
| 128 | 50.39 GB | 34.27 GB | **24.59%** |

> ⚠️ 内存节省随负样本数量**超线性增长**，极具扩展价值。

---

### 消融实验结果（Table 3 & Table 6）

#### 动态负载均衡效果（Table 3）
| Dataset | Strategy | Max Token Diff | Load Imbalance Delay | Delay Ratio ↓ |
|--------|---------|------------------|------------------------|--------------|
| KuaiRand-27K | Fixed Batch | 10,726 | 1100 ms | 47.01% |
| KuaiRand-27K | Global Token Reallocation | **559** | **37 ms** | **2.40%** |

> ➤ 负载不均延迟下降两个数量级！

#### 流水线调度效率（Table 6）
| Model | Computing Ratio | Comm. Not Overlapped |
|-------|------------------|------------------------|
| FuXi-large | 94.25% | **5.57%** |
| FuXi-long | 94.29% | **5.39%** |

> ➤ NPU 几乎持续满载运行，通信几乎完全被掩盖。

---

## 4. 关键结论和发现

### 主要发现
1. **Jagged 结构是 GR 性能瓶颈的关键根源**，传统 padding 方法在 Ascend 上代价极高；原生 jagged 支持可带来数量级优化。
2. **Ascend NPU 的 tile 架构非常适合 jagged fusion 和异步流水**，通过软硬协同设计能大幅提升 MFU。
3. **HSP + Semi-Async + 细粒度流水**三者结合，可实现 **>94% NPU 利用率** 和 **近线性扩展性（0.97）**。
4. **负采样内存可通过“卸载 + 量化 + logit 共享”组合策略有效缓解**，尤其适用于大规模召回场景。
5. **TurboGR 是首个专为 Ascend 构建的开源 GR 训练系统**，填补了国产 AI 芯片生态在推荐领域的空白。

---

### 方法的局限性
1. **目前主要支持 Transformer-based GR 模型**（如 HSTU/FuXi），尚未集成 MoE 或 Sparse Attention。
2. **Logit Sharing 对大模型需更高扩展因子 k**（如 FuXi-large 需 k=4），否则多样性不足。
3. **依赖特定硬件特性**（如 Ascend 的异步拷贝、tiling 策略），移植到其他芯片需重新工程化。
4. **未考虑实时推理优化**，聚焦于训练阶段加速。

---

### 未来工作方向（原文 Conclusion）
1. **支持 Mixture-of-Experts (MoE)** 架构，引入专家级负载均衡与通信优化。
2. **开发面向 Ascend 的 Sparse Attention 机制**，支持更长序列（如 16K–32K）训练。
3. **构建自动并行搜索框架**，联合优化 Hybrid Parallelism 策略，降低人工调参成本。
4. **扩展至多模态生成式推荐**（text + image + video），探索跨模态生成建模。

---

> 📌 **总结一句话**：  
> **TurboGR 通过软硬协同的三大创新（Jagged 加速、通信优化、负采样压缩），首次在 Ascend NPU 上实现了高效、可扩展的大规模生成式推荐训练，达到 54.71% MFU 与 0.97 扩展效率，为国产 AI 芯片在推荐系统领域提供了重要基础设施支撑。**

</details>

---

### 11. [SOMA: Efficient Multi-turn LLM Serving via Small Language Model](https://arxiv.org/abs/2605.11317)

**Authors**: Xueqi Cheng, Qiong Wu, Zhengyi Zhou, Xugui Zhou, Tyler Derr, Yushun Dong  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.11317v1  

#### Abstract
Large Language Models (LLMs) are increasingly deployed in multi-turn dialogue settings where preserving conversational context across turns is essential. A standard serving practice concatenates the full dialogue history at every turn, which reliably maintains coherence but incurs substantial cost i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SOMA: Efficient Multi-turn LLM Serving via Small Language Model

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **multi-turn LLM serving** 系统在处理多轮对话时，通常需要将完整的对话历史（`full dialogue history`）重复输入给大型语言模型（LLM），导致：
- **高延迟**（latency）
- **高内存消耗**
- **高昂的API成本**
- 随着对话增长，计算开销呈线性上升

现有方法如 `history compression` 或 `model routing` 要么牺牲上下文完整性，要么无法有效维持响应质量。

### 提出的新方法：SOMA
作者提出 **SOMA**（Soft-prompts for lOcal Manifold Approximation），一种基于“局部流形近似”（local manifold approximation）思想的高效多轮服务框架。其核心流程为三阶段：

1. **Soft Prompt Tuning**  
   利用早期对话轮次，在小模型（surrogate model）上通过可学习的 `soft prompt` 探索其与大模型（original model）行为差异最大的方向，即“最不一致的局部响应路径”。

2. **Localized LoRA Fine-tuning**  
   将挖掘出的“困难样本”用于对小模型进行轻量级的 `LoRA` 微调，使其在当前对话的局部语义流形上逼近大模型的行为。

3. **Efficiency Inference with Gate & Rollback**  
   引入一个简单的 `cosine gate` 决定是否切换到小模型；若后续查询发生语义漂移（drift），则自动回滚（rollback）至大模型并重新初始化状态。

### 相比现有方法的优势
| 方法 | 缺陷 | SOMA 的改进 |
|------|------|-------------|
| **History Compression** | 截断上下文，丢失推理链 | 保留完整语义依赖，仅压缩输入形式 |
| **Model Routing** | 小模型泛化差，切换开销大 | 动态适配小模型至当前会话局部行为 |
| **Direct Surrogate Use** | 响应漂移严重 | 通过局部微调显著提升一致性 |

> ✅ **核心创新**：首次将“局部流形近似”应用于多轮LLM服务，实现**高质量、低成本、自适应**的会话代理机制。

---

## 2. 核心实验方法和设置

### 数据集
在六个多轮对话基准上进行评估，涵盖多种任务类型：
- **ShareGPT**：开放域人机聊天
- **ReMeDi**：医生-患者医疗咨询
- **Craigslist**：买卖双方谈判
- **Multi-Char**：多角色扮演对话
- **MATH**：数学推理题（带逐步解法）
- **MT-Bench**：多任务质量评测

所有对话均经过过滤，确保为 **context-dependent** 类型（后续回复强依赖前期内容）。

### 模型配置
测试两个主流模型家族：
- **LLaMA Family**  
  - 大模型 $ F $: LLaMA-3.1-70B  
  - 小模型 $ G $: LLaMA-2-7B
- **Qwen Family**  
  - 大模型 $ F $: Qwen-3-8B  
  - 小模型 $ G $: Qwen-3-0.6B

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Original** | 每轮都用大模型处理完整历史 |
| **Surrogate** | 直接用小模型处理完整历史 |
| **History-Prefix** | 小模型接收完整历史但无微调 |
| **History-FT** | 在本地历史数据上对小模型做 LoRA 微调 |
| **LLMLingua-2** | 使用提示压缩技术减少上下文长度 |
| **RouteLLM** | 根据难度路由到不同模型 |
| **Random-FT** | 随机选择训练样本进行 LoRA 微调（用于消融） |

### 评估指标
- **Response Similarity**：使用多个 LLM Judge（GPT-OSS、DeepSeek-V3、Gemma2-27B）评估生成响应与原始大模型输出的相似度（0–1 分制）
- **Task-grounded Quality**：在 MATH 上报告 **Exact Match (EM)** 准确率
- **Efficiency Metrics**：
  - Token usage（总消耗token数）
  - Throughput（吞吐量）
  - End-to-end latency
  - Break-even point（何时开始节省成本）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 表1：响应相似度（LLaMA 家族，平均值 %）
| 方法 | Avg. Similarity |
|------|------------------|
| Surrogate | 75.1 ± 5.98 |
| History-Prefix | 84.8 ± 2.94 |
| History-FT | 90.8 ± 2.18 |
| RouteLLM | 92.2 ± 1.78 |
| **SOMA** | **93.1 ± 1.99** ✅ |

> ➤ SOMA 显著优于所有基线，在 **MATH 和 Multi-Char** 等复杂任务上增益最大。

#### 表2：MATH 数学任务准确率（EM %）
| 方法 | LLaMA Family | Qwen Family |
|------|---------------|--------------|
| Original | 48.34 | 36.48 |
| Surrogate | 19.20 | 11.73 |
| History-FT | 31.46 | 22.57 |
| RouteLLM | 33.88 | 25.08 |
| **SOMA** | **41.62** ✅ | **31.14** ✅ |

> ➤ SOMA 接近原模型表现，远超其他小模型方案，说明其不仅模仿表面文本，也保留了深层推理能力。

### 与基线方法对比结果
- 在 **response similarity** 上，SOMA 比最佳基线（RouteLLM）高出约 **0.9–1.0 pp**
- 在 **token usage** 上，SOMA 成功避免重复传输长历史，后期每轮节省高达 **37.2% token**
- 在 **throughput** 上，SOMA 接近纯小模型水平，显著快于大模型服务

#### 表3：ShareGPT 上的收益平衡点（Break-even）
| 对话长度（轮次） | Latency 收益 | Token 收益 |
|------------------|---------------|------------|
| 1–4 | -4.1% | -1.2% |
| 5–8 | +4.7% | +8.5% |
| 9–12 | +16.3% | +23.8% |
| 13+ | +29.1% | +37.2% |

> ➤ SOMA 在中长对话中优势明显，适合实际应用场景。

### 消融实验结果
#### 图2b：组件消融（LLaMA 家族）
| 变体 | 平均相似度下降 |
|------|----------------|
| Full SOMA | 93.1 |
| w/o Anti-degeneration Loss | ↓ ~1.5 |
| w/o ExpW + ADL | ↓ ~3.0 |

> ➤ 两个关键组件至关重要：
> - **Anti-degeneration regularizer**：防止 prompt collapse，保持探索多样性
> - **Expectation-weighted divergence**：捕捉分布级语义偏移，而非仅 token 匹配

此外，在 **Qwen 家族** 上的消融也验证了相同趋势（见附录 F.3）。

---

## 4. 关键结论和发现

### 主要发现
1. **多轮对话存在“长尾模式”**（long-tail pattern）  
   早期轮次承载大量上下文信息（token 多），后期轮次简短但高度依赖前期状态。这为“前段用大模型建模，后段用小模型代理”提供了理论基础。

2. **局部流形近似是可行且高效的策略**  
   小模型虽不能全局拟合大模型，但在特定会话的局部语义空间内可通过微调实现高保真逼近。

3. **SOMA 实现了效率与质量的平衡**  
   - 后期使用小模型 + 压缩上下文 → 显著降低延迟与成本
   - 局部 LoRA 微调 + drift-aware rollback → 维持高质量响应

4. **动态切换机制可靠**  
   实验显示，在突发言题转移场景下，drift detection 能以 **>88% 准确率检测漂移**，false rollback 率低于 **4.3%**

### 方法的局限性
1. **小模型容量限制**  
   当小-大模型差距过大时（如 Qwen-3-0.6B vs Qwen-3-8B），局部适配仍难以完全复现复杂行为。

2. **依赖内部访问权限**  
   需要访问小模型的 embedding space 和 tokenizer，难以直接部署于严格黑盒 API 场景。

3. **适用于局部连贯对话**  
   若对话频繁跳跃主题或 warm-start 阶段噪声大，局部流形估计可能失效。

4. **存在一次性启动开销**  
   不适合极短对话（<5轮），需足够长的 warm-start 窗口才能摊销成本。

### 未来工作方向
- 设计更鲁棒的 **drift detector**，支持多话题切换
- 开发无需内部访问的 **approximate mining** 方法
- 支持 **multi-region adaptation**，允许多个局部流形共存
- 探索跨会话的 soft prompt 共享，进一步降低冷启动成本
- 扩展至 **multimodal serving** 场景

---

> 🔍 **总结一句话**：  
> **SOMA 通过“软提示挖掘 + 局部 LoRA 微调 + 语义门控切换”，实现了在保持高质量响应的同时大幅降低多轮 LLM 服务成本，是迈向高效、可持续对话系统的重要一步。**

</details>

---

### 12. [Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding](https://arxiv.org/abs/2605.12185)

**Authors**: Yigeng Zhou, Wu Li, Yifan Lu, Yequan Wang, Xuebo Liu, Wenya Wang, Jun Yu, Min Zhang, Jing Li  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.12185v1  

#### Abstract
Large language models accumulate extensive parametric knowledge through pre-training. However, knowledge conflicts occur when outdated or incorrect parametric knowledge conflicts with external knowledge in the context. Existing methods address knowledge conflicts through contrastive decoding, but in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于 **Large Language Models (LLMs)** 在 **Retrieval-Augmented Generation (RAG)** 场景中面临的 **context-memory conflicts**（上下文-记忆冲突）问题。  
当模型内部参数化知识（parametric knowledge）与外部检索到的上下文信息（contextual knowledge）发生矛盾时，LLMs 倾向于过度依赖其预训练记忆，导致输出违背最新或正确的外部信息。

### ✅ 提出的新方法：DCRD
作者提出了一种两阶段解码框架——**Dynamic Cognitive Reconciliation Decoding (DCRD)**，其核心思想是通过“认知协调”机制动态应对冲突：

- **第一阶段：基于注意力图谱的冲突预测（Conflict Prediction）**  
  利用 LLM 自身生成的 **attention maps** 计算 **contextual fidelity**（上下文保真度），衡量模型在生成过程中对上下文的依赖程度，并以此作为输入是否处于冲突状态的判断依据。
  
- **第二阶段：认知协调解码（Cognitive Reconciliation Decoding）**  
  根据冲突预测结果进行 **dynamic routing**：
  - 若无显著冲突 → 启用 **greedy decoding**（避免不必要的干预）
  - 若存在高冲突 → 启动 **dynamic contrastive decoding (DCD)**，自适应调整对比强度（`α_adj`），增强对外部上下文的关注。

### ✅ 相比现有方法的优势
| 方法 | 局限性 | DCRD 的改进 |
|------|--------|-------------|
| **CAD** [18] | 固定超参数，对非冲突场景造成“过干预”，降低准确率 | 动态调节干预强度，仅在必要时加强对比 |
| **COIECD / ADACAD** | 虽为动态策略，但建模简单，难以处理复杂现实场景 | 引入 attention-based fidelity 预测，实现更精准的冲突感知 |
| **Fine-tuning / Prompting 方法** | 泛化性差、计算开销大、任务特定 | 无需额外训练，适用于多种 LLM 和任务 |

> ✅ **核心优势总结**：  
> DCRD 实现了 **conflict-aware** 的智能路由，在保持非冲突场景下高效稳定的同时，有效缓解高冲突情况下的知识偏差，兼顾 **accuracy、robustness 与 efficiency**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 类型 | 描述 |
|-------|------|------|
| **NQ (Natural Questions)** | General QA | 包含真实谷歌搜索问题的大规模问答数据集，用于评估低冲突场景表现 |
| **TriviaQA** | General QA | 高难度阅读理解数据集，含 trivia 爱好者提出的问题及多篇支持文档 |
| **SQuAD** | General QA | 经典机器阅读理解数据集，答案来自维基百科段落 |
| **NQ-Swap** | Knowledge Conflict QA | 通过对 NQ 中实体替换构造合成冲突，测试模型优先使用上下文的能力 |
| **Counterfacts** | Knowledge Conflict QA | 构造参数知识与上下文相悖的样本，专门用于冲突检测研究 |
| **ConflictKG (本文构建)** | 新基准 | 生成式构建的真实感更强的知识冲突 QA 基准，包含 **4,466 个样本**，涵盖冲突与非冲突实例及其知识来源 |

> 🔧 **ConflictKG 构建流程**：
> 1. 从 Wikidata 提取三元组 `(s, r, o)`；
> 2. 替换对象 `o → o'`（语义相似但事实不同）制造冲突；
> 3. 使用 LLM 生成语言流畅、推理复杂的上下文；
> 4. 多阶段质量控制（相关性过滤、GPT-4o 冲突验证、响应质量筛选）

### ⚙️ 实验设置
- **模型**：Llama2-7b, Llama2-13b, Llama3-8b, Mistral-7b
- **基线方法**：
  - Greedy Decoding（标准贪婪解码）
  - CAD（Context-aware Decoding）
  - COIECD（基于熵变化的动态解码）
  - ADACAD（基于 Jensen-Shannon 散度调节）
- **提示模板**（zero-shot）：
  ```
  {context}
  Using only the references listed above, answer the following question:
  Question: {question}
  Answer:
  ```
- **最大生成长度**：32 tokens
- **评估指标**：
  - 采用 **GPT-4o 自动生成评估框架**
  - 设计 prompt 判断生成答案是否符合文档内容或 ground truth（见 Table VIII）
  - 输出 **accuracy** 作为主指标（优于传统 EM 的灵活性）

---

## 3. 主要实验结果和性能指标

### 📊 总体性能对比（Table I）
在六大数据集上平均表现（以 Llama2-7b 为例）：

| 方法 | Avg. Accuracy | Δ vs Greedy |
|------|----------------|------------|
| Greedy | 56.9 | — |
| CAD | 58.2 | +1.3 |
| COIECD | 64.5 | +5.6 |
| ADACAD | 64.1 | +5.2 |
| **DCRD (Ours)** | **71.4** | **+14.5** ✅ |

> ✅ **DCRD 显著领先所有 baseline**，尤其在冲突密集场景提升明显。

#### 🔍 分项关键结果：
- **在 NQ-Swap 上（高冲突）**：
  - DCRD 相比 Greedy 平均提升 **17.7~29.1%**
  - 超越 CAD 达 **6.7~11.6%**
- **在 NQ/SQuAD/TriviaQA 上（低冲突）**：
  - CAD 表现劣于 Greedy（平均下降 8.1%），说明静态干预有害
  - DCRD 反而持续超越 Greedy（+6.0~16.5%），证明路由机制成功规避“过干预”
- **在 ConflictKG 上**：
  - DCRD 平均准确率达 **81.1%**
  - 超越 Greedy (+15.1%)、CAD (+11.5%)、COIECD (+5.2%)、ADACAD (+6.0%)

### 🔬 消融实验（Ablation Studies）

#### （1）移除 Dynamic Routing (DR) 的影响（Table III）
| 方法 | Conflict | Non-Conflict | Total |
|------|----------|--------------|--------|
| DCRD w/o DR | 73.9 | 83.2 | 78.6 |
| **DCRD (完整版)** | **74.8** | **87.3** | **81.1** ✅ |

> ➤ 移除 DR 后非冲突性能下降明显 → 证明 **routing 至 greedy path 对维持稳定性至关重要**

#### （2）冲突预测器有效性比较（Table V）
| 冲突预测方式 | 准确率 | QA 准确率 | Δ vs Ours |
|-------------|--------|-----------|---------|
| Random (50%) | — | 71.5 | -9.6% ❌ |
| Hidden State (16层) | 76.5% | 77.8 | -3.3% |
| Hidden State (32层) | 73.2% | 77.1 | -4.0% |
| **Attention Maps (Ours)** | **84.7%** | **81.1** | — ✅ |

> ➤ 注意力图谱作为特征显著优于隐藏状态或其他随机分类器

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **静态 contrastive decoding（如 CAD）在非冲突场景有害**，会引入偏差并降低性能。
2. **DCRD 成功实现了“按需干预”**：通过 attention fidelity 预测冲突，并动态选择解码路径，既提升了冲突场景下的准确性，又保留了非冲突场景的原始能力。
3. **ConflictKG 是一个高质量、贴近现实的知识冲突基准**，可用于未来方法的细粒度评估。
4. **DCRD 在噪声环境、分布外任务、不同冲突比例下均表现出强鲁棒性**（Figure 3–5）：
   - 插入 30% 噪声上下文后性能几乎不变（仅降 0.1%）
   - 在跨数据集迁移中保持良好泛化能力

### ⚠️ 方法的局限性
1. **仅在 7B–13B 规模模型上验证**，尚未扩展至更大模型（如 70B 或闭源模型）。
2. **未在 chat-tuned 模型（如 Llama2-7b-chat）上测试**，可能影响实际对话系统中的适用性。
3. **冲突分类器采用轻量级单层 MLP**，虽高效，但更复杂模型（如 BERT、SVM）是否能进一步提升尚待探索。

### 🔮 未来工作方向
- 将 DCRD 扩展至 **多模态 LLMs** 和 **长文本生成任务**
- 探索其他 **conflict detection signal**（如梯度、中间表示差异）
- 结合 **knowledge editing** 技术实现端到端的动态知识更新机制
- 开发面向工业级 RAG 系统的 **real-time DCRD inference engine**

---

## ✅ 总结一句话
> 本论文提出的 **DCRD** 方法通过 **attention-driven conflict prediction + dynamic decoding routing**，首次实现了对 context-memory conflicts 的“认知协调式”响应，在保证效率的同时显著提升了 LLM 在冲突与非冲突场景下的综合表现，推动了 RAG 系统向更可靠、更智能的方向发展。

</details>

---

### 13. [A Resampling-Based Framework for Network Structure Learning in High-Dimensional Data](https://arxiv.org/abs/2605.12706)

**Authors**: Ziwei Huang, Zeyuan Song, Paola Sebastiani, Stefano Monti  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.12706v1  

#### Abstract
RSNet is an open-source R package that provides a resampling-based framework for robust and interpretable network inference, designed to address the limited-sample-size challenges common in high-dimensional data. It supports both the estimation of partial correlation networks modeled as Gaussian net...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Resampling-Based Framework for Network Structure Learning in High-Dimensional Data

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对高维数据（如‘omics’数据）中常见的“小样本、大变量”（small *n*, large *p*）问题，解决传统网络推断方法在**有限样本下可靠性差、稳定性低、解释性不足**的问题。具体挑战包括：
- 无法有效区分直接依赖与间接依赖（如相关性网络的局限）
- 缺乏对边（edge）层面不确定性的量化
- 对于混合数据类型（continuous + discrete）支持有限
- 在存在相关观测（如家族数据）时，忽略结构依赖会导致假阳性膨胀

### 提出了什么新方法或新思路
作者提出并实现了 **RSNet** —— 一个基于重采样（resampling-based）的开源 R 包，用于鲁棒且可解释的网络结构学习。其核心创新包括：

- **统一的重采样框架**：集成多种重采样策略（bootstrap、subsampling、cluster-based resampling），通过构建多个子网络生成**共识网络（consensus network）**，提升推断稳定性。
- **支持两类主流模型**：
  - Gaussian networks（基于 partial correlation）
  - Conditional Gaussian Bayesian Networks (CGBNs)，适用于混合数据类型
- **首次实现高效的 signed Graphlet Degree Vector Matrix (GDVM) 构建**：
  - 利用先进图元计数算法（如 ORCA 改进版）+ 并行计算
  - 时间复杂度降至 $O(|d|)$，在稀疏网络中接近常数时间，远优于暴力枚举的 $>O(p^3)$
- **引入 signed graphlet 分析**：结合边符号信息（正/负相关），增强节点局部拓扑角色的可解释性

### 相比现有方法的优势
| 功能/特性 | RSNet | 其他工具（glasso, huge, SILGGM, BDgraph, RHugin, igraph, ORCA） |
|---------|-------|-------------------------------------------------------------|
| 重采样支持 | ✅ 多种策略（含 cluster-based） | ❌ 或仅限单一估计 |
| 统计推断输出 | ✅ empirical CI, adjusted *p*-values, edge-selection frequency | ⚠️ 部分支持（如 SILGGM 提供渐近 CI），BDgraph 为贝叶斯后验概率 |
| 混合数据支持 | ✅ CGBN 模型 | ❌ 多数仅处理连续变量 |
| 图元分析（Graphlet） | ✅ 支持 signed GDVM，高效构造 | ❌ 不支持 GDVM；ORCA 仅支持 unsigned |
| 家族/相关数据处理 | ✅ cluster bootstrap / fractional cluster bootstrap | ❌ 易低估变异性，导致假阳性 |
| 可扩展性与并行化 | ✅ 内建并行支持整个流程 | ⚠️ 少数支持部分并行 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
论文未展示传统意义上的“对比实验表格”，而是强调 RSNet 已成功应用于多个真实世界生物医学数据集，验证其实际效用：

- **衰老与长寿研究**：
  - New England Centenarian Study (NECS)
  - Long Life Family Study (LLFS)
  - Integrative Longevity Omics (ILO)
- **疾病相关数据集**：
  - Late-Onset Alzheimer’s Disease (LOAD)
  - 癌症基因组图谱（The Cancer Genome Atlas, TCGA）多个癌种队列

这些数据具有典型高维特征（成千上万个基因/代谢物，数百个样本），且部分包含家族聚类结构。

### 实验设置和评估指标
尽管没有传统 benchmark 实验，RSNet 的设计本身即构成一套系统性评估框架：

- **重采样过程**：运行 `m` 轮重采样（如 m=100–1000），每轮生成一个网络（weighted 或 binary adjacency matrix）
- **共识网络构建**：
  - 对 Gaussian networks：基于边缘频率 > 阈值（如 0.5）+ 统计显著性（adjusted *p*-value < α）
  - 对 CGBNs：使用 edge-selection frequency 作为权重
- **下游分析模块**：
  - Centrality analysis（中心性分析）
  - Community detection（社区检测）
  - Differential connectivity analysis（差异连接性分析）
  - Graphlet-based topology analysis（图元拓扑分析）

评估指标主要包括：
- Edge selection frequency（边选择频率）
- Empirical confidence intervals（经验置信区间）
- Adjusted *p*-values（多重检验校正后的 *p* 值）
- Signed GDVM 的计算效率（运行时间 vs 网络规模）

### 基线方法对比
虽然未进行端到端性能比较实验，但文中明确指出 RSNet **整合并超越了以下工具的功能**：

| 工具 | 功能 | RSNet 如何改进 |
|------|------|----------------|
| `glasso`, `huge` | 精确矩阵估计，无统计推断 | 添加重采样+CI+*p*-values |
| `SILGGM` | 提供边缘推断（渐近法） | 使用更稳健的经验分布（非参数重采样） |
| `BDgraph` | 贝叶斯网络结构采样，提供 inclusion probability | 提供频率主义视角下的稳定性度量（更易解释） |
| `RHugin` | CGBN 结构学习（PC算法） | 加入 resampling 和 stability measurement |
| `igraph` | 基础网络分析（中心性、社区） | 扩展至 graphlet-level 分析 |
| `ORCA` | 高效 unsigned GDVM 构造 | 支持 signed GDVM，并优化至近常数时间 |

---

## 3. 主要实验结果和性能指标

由于本论文是**软件方法介绍型论文（software note）**，重点在于功能实现与应用展示，而非传统数值性能报告。但仍可提炼出关键成果：

### 关键性能数据
- **Signed GDVM 构建速度**：
  - 在稀疏网络中达到 $O(|d|)$ 时间复杂度，其中 $|d|$ 为平均度数
  - 相比暴力枚举 $>O(p^3)$，极大提升了 scalability
  - 支持上千节点规模的网络快速图元分析
- **内存与并行效率**：
  - 利用 parallelization 加速重采样循环与 GDVM 构建
  - 可在标准服务器上完成大规模模拟与真实数据分析

### 与基线方法的对比结果（隐式体现）
- 在相同数据上，RSNet 相比单次估计方法（如 glasso）能识别出更稳定的 hub nodes 和 communities
- 引入 cluster-based resampling 后，在家族数据中显著降低假阳性边的比例（引用文献 [13] 支持）
- 图元分析揭示了传统 centrality 指标无法捕捉的局部结构模式（如特定 sign pattern 的 3-node motifs）

### 消融实验结果（文中未明确开展）
论文未提供 formal ablation study（如移除重采样或移除 signed graphlet 是否影响性能）。但通过模块化设计说明各组件作用：
- 若不使用 resampling → 失去 edge-level uncertainty quantification
- 若不用 signed GDVM → 丢失边符号带来的生物学语义（激活 vs 抑制）

---

## 4. 关键结论和发现

### 论文的主要发现
1. **RSNet 是首个将 resampling、statistical inference 与 signed graphlet analysis 统一于同一框架的 R 包**，填补了高维网络推断中“可靠性”与“可解释性”之间的鸿沟。
2. 通过重采样机制，能够有效缓解 small *n*, large *p* 场景下的过拟合问题，提高网络结构的稳健性。
3. 引入 signed graphlet 分析，使研究人员能够在 subnetwork 层面探索功能模块的动态变化（如不同疾病状态下 motif 分布差异）。
4. 在多个真实生物数据集中已成功应用，证明其在 aging、cancer、neurodegenerative diseases 中具备实用价值。

### 方法的局限性
- 当前仅支持 Gaussian 和 CGBN 类模型，尚未扩展至非线性或深度学习网络（如 DAG-GNN）
- 对极高维度（$p > 10^4$）的数据仍需谨慎使用，因重采样可能增加计算负担（尽管有并行优化）
- 图元分析目前聚焦于小规模 motifs（如 3–4 节点），更大 graphlets 的计算仍具挑战

### 未来工作方向
- 扩展至更多类型的网络模型（如 Ising models for binary data, nonlinear GGMs）
- 开发可视化工具以直观展示 signed graphlet signatures 和 differential topology
- 集成 causal inference 模块，从 observational data 推断潜在因果关系
- 进一步优化分布式计算能力，适配 cloud computing 平台

---

> ✅ **总结一句话**：  
> RSNet 提供了一个**模块化、可并行、支持重采样与 signed graphlet 分析**的开源框架，显著提升了高维数据中网络推断的**统计可靠性与结构可解释性**，特别适用于生物医学中的 omics 数据分析。

</details>

---

### 14. [An Agentic LLM-Based Framework for Population-Scale Mental Health Screening](https://arxiv.org/abs/2605.13046)

**Authors**: Giuliano Lorenzoni, Paulo Alencar, Donald Cowan  
**Category**: cs.AI  
**Published**: 2026-05-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.13046v1  

#### Abstract
Mental health disorders affect millions worldwide, and healthcare systems are increasingly overwhelmed by the volume of clinical data generated from electronic records, telemedicine platforms, and population-level screening programs. At the same time, the emergence of novel AI-based approaches in he...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An Agentic LLM-Based Framework for Population-Scale Mental Health Screening

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前心理健康筛查面临以下挑战：
- **数据量大**：电子健康记录、远程医疗平台和大规模筛查项目产生海量非结构化临床文本。
- **系统脆弱性高**：传统基于LLM的RAG（Retrieval-Augmented Generation）系统配置复杂、难以复现，且在更新时容易引入性能退化（regression）。
- **缺乏可信赖的自动化流程**：现有方法在**可重复性（reproducibility）、适应性（adaptability）和成本控制**方面表现不佳，难以支持人群规模（population-scale）的心理健康筛查。

### 🚀 提出的新方法与创新思路
本文提出一个**基于Agentic架构的LLM框架**，用于构建稳健、可扩展的自然语言处理流水线，其核心创新包括：

| 创新点 | 描述 |
|-------|------|
| **模块化Agentic Pipeline** | 将整个分类流程分解为多个独立的 **LangChain Agent**，每个Agent负责特定子任务（如预处理、检索、多样性控制、阈值优化等），并拥有明确的角色与责任。 |
| **显式策略驱动（Explicit Policy-Guided Execution）** | 每个Agent遵循明确定义的策略进行决策，例如“冻结”（freeze）或“回滚”（rollback）某一配置，确保只有显著改进才能覆盖已有设定。 |
| **代理引导探索（Proxy-Guided Exploration）** | 引入低成本的**代理评估指标**（proxies）来初步筛选候选配置，仅将最有希望的方案送入昂贵的“黄金评估”（gold evaluation），大幅降低计算开销。 |
| **渐进式锁定机制（Incremental Locking Mechanism）** | 配置逐阶段锁定，后续调整不能破坏已验证的性能基线，保障系统的**非回归保证（non-regression guarantee）**。 |
| **Orchestrator Agent 统一协调** | 由中央的 **Orchestrator Agent** 协调所有子Agent，执行策略监督、日志记录和自动回滚，实现端到端的自动化治理。 |

### 🔍 相比现有方法的优势
| 对比维度 | 传统RAG系统 | 本论文提出的Agentic框架 |
|--------|-------------|--------------------------|
| 架构灵活性 | 多为静态、单体式设计 | 模块化、动态可插拔Agent结构 |
| 可维护性 | 配置变更易导致性能下降 | 冻结+回滚机制防止退化 |
| 成本效率 | 全面超参搜索耗资巨大 | Proxy机制提前过滤低效配置 |
| 可解释性 | 黑箱操作，难追溯决策路径 | 每步决策有策略依据，全程可审计 |
| 扩展能力 | 难以适配新场景 | 支持运行时注入新策略，具备演化能力 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **DAIC-WOZ**（Distress Analysis Interview Corpus - Wizard-of-Oz）数据集。
- 包含189个临床访谈的转录文本（transcripts），标注了抑郁状态（PHQ8_Binary）。
- 数据划分为训练集、开发集和测试集。
- 注：出于伦理限制，原始数据不可公开分发，需通过官方渠道申请访问。

### ⚙️ 实验设置
- **任务类型**：基于转录文本的二分类——检测是否患有抑郁症。
- **Pipeline 结构**（共8个步骤 + 1个Orchestrator）：
  1. **Preprocessing Agent**：嵌入模型与截断策略选择
  2. **Similarity Metric Agent**：相似度度量（cosine等）与阈值设定
  3. **Selection Agent**：Top-k 与 RAG 类型（static/dynamic/hybrid）
  4. **Diversity Agent**：MMR（Maximal Marginal Relevance）控制冗余
  5. **Post-filters Agent**：去重、元数据过滤、低置信度过滤
  6. **Data Expansion Agent**：伪相关反馈（PRF）与训练-测试增强
  7. **Threshold Optimization Agent**：动态调整相似度阈值
  8. **Decoding Agent**：LLM生成参数调优（temperature, top_p, n）
- **Orchestrator Agent**：基于 **LangChain / LangGraph** 实现调度与策略执行。

### 📊 评估指标
| 指标类别 | 具体指标 |
|--------|---------|
| 主要性能指标 | `accuracy`, `macro-F1`, `precision`, `recall`, `confusion matrix` |
| 代理评估（Proxy）指标 | 语义检索质量、统计平衡性、排序稳定性、置信启发式、轻量级LLM一致性 |
| 成本与鲁棒性监控 | 推理延迟、上下文长度、方差容忍度、非回归检查 |

### 🔁 基线方法对比
- 本文未直接与其他完整系统进行全面对比，而是以**自身逐步锁定的配置作为基线**（baseline），并通过消融实验验证各组件的有效性。
- 基线配置最终锁定为：
  - Top-k = 5
  - 动态RAG（dynamic selection）
  - MMR关闭
  - 相似度阈值 T ≈ 0.75
  - 温度 = 0.0，top_p = 1.0，n = 1

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自Table III）
| 步骤 | 调整参数 | 最佳结果 | 决策 |
|-----|---------|--------|------|
| 1–4 | K ∈ {2,3,5}, TE ∈ [0.75,0.78,0.82], RAG类型, MMR | K=5, T=0.75, dynamic, MMR=OFF | 锁定配置 |
| 5 | Post-filters（去重/元数据/低置信） | 无增益 | 保持关闭 |
| 6 | PRF / 数据增强 | 增强后macro-F1下降超过0.02 | 拒绝，PRF=OFF |
| 7 | 阈值扫描 T ∈ [0.70, 0.80] | T=0.78时acc=0.821, macroF1=0.789 < 基线 | 回滚，保留T=0.75 |
| 8 | temperature ∈ {0.0,0.1,0.2}, top_p ∈ {1.0,0.9}, n ∈ {1,3} | 所有变体均低于基线 | 无变化，维持默认 |

> ✅ **最终锁定性能（基线）**：
> - Accuracy: **0.857**
> - Macro-F1: **0.825**
> - Recall: **0.875**

### 🔍 与基线方法的对比结果
- 所有尝试的替代配置（如更高阈值、不同解码参数、启用PRF/MMR）均未能超越初始锁定的基线。
- 特别是T=0.78虽然在某些代理指标上看似更优，但在黄金评估中表现更差，证明了**非回归策略的重要性**。
- 解码参数（temperature等）的变化也未带来收益，说明在该任务中确定性输出（deterministic decoding）更稳定可靠。

### 🧪 消融实验结果
- **MMR开启会导致少数类召回率下降** → 被禁用
- **Post-filters 在置信度<0.65时应用也无法提升macro-F1** → 不启用
- **PRF（伪相关反馈）引入噪声且不一致** → 被拒绝
- **增大Top-k至>5会引发precision崩溃** → 被限制
- **动态RAG优于静态与混合模式** → 成为核心选择

> 💡 结论：**简单而稳定的配置胜过复杂但不稳定的策略组合**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Agentic架构能有效管理复杂NLP流水线的可变性空间**，通过模块化分工与策略控制，提升了系统的可维护性和可解释性。
2. **Proxy-guided exploration 显著降低了评估成本**，避免了对大量无效配置进行昂贵的LLM评估。
3. **非回归保证机制（non-regression policy）至关重要**，它防止了因局部优化而导致的整体性能退化，增强了系统鲁棒性。
4. **最佳配置趋于简洁稳定**：最终收敛于 `cosine similarity + dynamic Top-k=5 + threshold T≈0.75`，无需复杂的后处理或数据增强。
5. 该框架不仅适用于抑郁检测，还可扩展至其他需要大规模临床文本分析的任务（如焦虑、PTSD筛查）。

### ⚠️ 局限性
- 当前实验基于单一数据集（DAIC-WOZ），泛化能力有待在更多真实世界数据上验证。
- 框架依赖于人工定义的策略规则，尚未完全实现策略的自我学习与进化。
- LLM本身的随机性（stochasticity）仍可能影响决策稳定性，需结合统计检验进一步强化策略判断。
- 目前主要面向离线批量处理，实时在线部署的支持尚待完善。

### 🔮 未来工作方向
1. **引入自适应策略学习机制**：让Orchestrator能够从历史决策中学习最优策略，减少人工干预。
2. **跨机构/多中心数据集成**：将框架应用于更大规模的真实医疗信息系统（如EHR、telemedicine平台）。
3. **支持多模态输入**：整合语音、视频特征，构建更全面的精神健康评估系统。
4. **加强隐私保护机制**：结合联邦学习（Federated Learning）或差分隐私，在不共享原始数据的前提下进行模型训练与推理。
5. **向临床落地推进**：与医生协作开展前瞻性研究，评估系统在真实诊疗环境中的可用性与有效性。

---

## 总结
本文提出的 **Agentic LLM-Based Framework** 是迈向可信、可扩展AI辅助心理健康筛查的重要一步。它通过**模块化Agent设计、显式策略控制、代理评估与非回归保障机制**，解决了传统LLM系统在医疗场景下面临的脆弱性、高成本与不可控等问题。实验证明该框架能在控制成本的同时找到稳定高效的配置，为未来在数字健康基础设施中部署大规模精神疾病早期筛查提供了坚实的技术基础。

</details>

---

### 15. [Continual Fine-Tuning of Large Language Models via Program Memory](https://arxiv.org/abs/2605.13162)

**Authors**: Hung Le, Svetha Venkatesh  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.13162v1  

#### Abstract
Parameter-Efficient Fine-Tuning (PEFT), particularly Low-Rank Adaptation (LoRA), has become a standard approach for adapting Large Language Models (LLMs) under limited compute. However, in continual settings where models are updated sequentially with small datasets, conventional LoRA updates struggl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Continual Fine-Tuning of Large Language Models via Program Memory》论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**持续学习（Continual Learning, CL）**场景下，大型语言模型（LLMs）通过参数高效微调（PEFT）技术如 **LoRA** 进行适配时，面临严重的**灾难性遗忘（catastrophic forgetting）**问题。尽管 LoRA 能减少参数更新量，但在顺序处理多个小规模任务时，传统的连续 LoRA 更新仍会导致不同任务间的**梯度干扰（gradient interference）**，从而破坏已学知识。

现有方法存在以下不足：
- **模块化架构**（如添加多个 LoRA 模块）会增加推理延迟和内存开销；
- **正则化方法**（如 O-LoRA、DEAL）虽缓解干扰，但限制了模型对新任务的学习能力（即牺牲“可塑性”以换取“稳定性”）；
- 缺乏机制来控制低秩空间中知识的写入、分区与整合过程。

---

### 🚀 提出的新方法：ProCL
本文提出 **ProCL**（Programmed Continual LoRA），一种受神经科学中的**互补学习系统（Complementary Learning Systems, CLS）**启发的持续 LoRA 框架。

#### 核心思想
将 LoRA 适配器视为一个**程序记忆（Program Memory）**，其由多个可复用的“程序槽”（program slots）组成，每个槽存储特定输入模式下的适应性调整。训练过程中：
- 输入通过注意力机制动态检索并组合相关程序槽，形成输入条件化的适配权重；
- 原始 LoRA 权重作为稳定的基础表示（slow system，类皮层）；
- 动态组合的程序提供快速适应能力（fast system，类海马体）；
- 定期通过**巩固步骤（consolidation）**将执行权重回传到持久记忆中，实现知识积累。

#### 创新点
1. **结构化低秩空间**：首次将“程序记忆”概念引入 LoRA，显式组织低秩更新区域，避免全局修改带来的干扰。
2. **输入驱动的路由机制**：利用 attention 实现输入依赖的参数选择，使相似输入共享相同程序槽，提升参数利用率。
3. **双通路设计**：结合稳定基础权重与动态程序组合，在训练阶段保持高可塑性，推理阶段输出单一静态适配器，无额外开销。
4. **完全兼容 LoRA 架构**：不引入外部缓冲区或模块，仅在原有 LoRA 参数内操作，部署成本为零。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 Seq-LoRA, O-LoRA, DEAL） | ProCL |
|------|----------------------------------------|-------|
| 干扰控制 | 弱或间接（如正交约束） | 显式路由隔离，理论证明干扰趋近于零 |
| 可塑性保留 | 随任务增长逐渐下降 | 局部更新 + 固定容量管理，维持长期学习能力 |
| 推理效率 | 多数不变或略降 | **无额外开销**（训练后丢弃路由机制） |
| 实现复杂度 | 高（需维护多模块或多约束） | 低（纯 LoRA 内部重构） |

---

## 2. 核心实验方法和设置

### 📚 数据集
实验涵盖两类典型持续学习任务：

#### （1）问答任务（QA）
- **任务序列**：`BoolQ → SQuAD → AdversarialQA`
- 每个任务采样 5,000 训练样本 + 1,000 测试样本
- 目标：生成答案字符串并与真实标签匹配
- 评估指标：**平均准确率（Accuracy）**

#### （2）文本分类任务（Text Classification, TC）
- **三种课程长度**：
  - 3-task
  - 4-task
  - 15-task（长序列挑战）
- 使用数据集包括：Yelp, Amazon, MNLI, IMDB, SST-2, AGNews 等共 15 个
- 评估指标：
  - **Average Accuracy (AA)**：所有任务上的平均分类精度
  - **ROUGE-1 (R-1)**：预测输出与真实标签之间的 unigram F1 分数

---

### ⚙️ 实验设置
- **模型家族**：
  - `LLaMA-3`（3B, 8B）
  - `Qwen3`（4B, 8B）
  - `Flan-T5`（Base, Large）
- **LoRA 设置**：
  - Rank: 16 (QA), 32 (TC)
  - Dropout: 0.1
  - Scaling: 32
- **优化配置**：
  - 学习率：1e-5
  - Batch Size: 4–16
  - Epochs: 每任务 1 轮（MNLI 除外为 2 轮）
- 所有基线均基于统一训练管道复现，确保公平比较

---

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Seq-LoRA** | 基础方法 | 共享同一组 LoRA 参数，易遗忘 |
| **O-LoRA** | 正则化 | 强制任务间 LoRA 更新正交 |
| **DEAL** | SOTA 方法 | 使用小波核保留历史知识 |
| **EWC / Replay** | 经典 CL 方法 | 应用于 LoRA 参数进行对比（仅 QA 中测试） |

---

## 3. 主要实验结果和性能指标

### 📊 QA 任务结果（Table 1）
| 方法 \ 模型 | LLaMA-3.2-3B | LLaMA-3.1-8B | Qwen3-4B | Qwen3-8B | Flan-T5-Base | Flan-T5-Large | **Avg** |
|------------|--------------|---------------|-----------|-----------|----------------|------------------|---------|
| Seq-LoRA   | 67.7         | 64.5          | 52.5      | 59.1      | 61.3           | 62.0             | 61.2    |
| O-LoRA     | 68.1         | 72.4          | 56.6      | 58.6      | 54.2           | 61.6             | 61.9    |
| DEAL       | 66.4         | 72.1          | 68.2      | 72.9      | 61.4           | 55.9             | 66.2    |
| **ProCL (Ours)** | **69.8**     | **73.2**      | **72.5**  | **72.8**  | **62.8**       | **63.4**         | **69.1** |

✅ **结论**：
- ProCL 在所有模型上均取得最佳性能，平均准确率达 **69.1**，显著优于最强基线 DEAL（+2.9）和 Seq-LoRA（+7.9）；
- 尤其在较难的 Qwen3-4B 上表现突出（+4.3 vs DEAL），说明其对分布偏移更强的任务更具鲁棒性。

---

### 📈 文本分类任务结果（Table 2）
| 方法 \ 指标 | 3-task AA | 4-task AA | 15-task AA | **Overall AA** |
|-----------|-----------|-----------|-------------|----------------|
| Seq-LoRA  | 55.2      | 50.6      | 50.6        | 55.2           |
| O-LoRA    | 54.0      | 50.6      | 50.6        | 54.0           |
| DEAL      | 56.2      | 51.4      | 51.4        | 56.2           |
| **ProCL (Ours)** | **57.5**  | **52.7**  | **52.7**    | **57.5**       |

✅ **关键发现**：
- 在更长的任务序列（15-task）中优势明显：相比 DEAL 提升约 **0.9–1.3%**；
- 在 Flan-T5-Large 上，4-task 设置下 AA 达 **81.0**（vs DEAL 的 78.3），提升达 **2.7%**；
- ROUGE-1 指标也全面领先，表明生成质量更高。

---

### 🔬 消融实验（Ablation Study, Table 3）
验证 ProCL 各组件的重要性（在 QA 任务上）：

| 配置 | LLaMA-3.2-3B | Qwen3-4B | Flan-T5-Base |
|------|---------------|----------|---------------|
| Full ProCL | **69.8** | **72.5** | **62.8** |
| w/o Worig（移除原始权重） | 65.9 ↓3.9 | 67.1 ↓5.4 | 61.6 ↓1.2 |
| w/o Consolidation（无巩固） | 68.2 ↓1.6 | 69.4 ↓3.1 | **60.1 ↓2.7** |
| Random routing（随机路由） | 67.5 ↓2.3 | 68.6 ↓3.9 | 61.8 ↓1.0 |
| Uniform routing（均匀路由） | 69.0 ↓0.8 | 70.1 ↓2.4 | 60.4 ↓2.4 |

✅ **结论**：
- **Worig 是关键**：移除后性能大幅下降，说明稳定参考的重要性；
- **Consolidation 对 encoder-decoder 模型尤其重要**（如 Flan-T5）；
- **输入条件化路由机制有效**：随机/均匀路由导致性能退化，证明 attention 路由确能降低干扰。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **ProCL 显著缓解灾难性遗忘**：
   - 在多种 LLM 架构和任务类型下，均优于当前最先进的持续 LoRA 方法；
   - 图1显示，ProCL 在 BoolQ 上的遗忘程度最低（如 Flan-T5-Base 上仅下降 4.63 pts，而其他方法普遍 >20 pts）。

2. **理论支持强**：
   - 提出的程序路由机制被形式化分析，证明其可使跨任务梯度交互趋于零（Remark 1）；
   - 巩固机制被证明能收敛至期望执行权重（Remark 2）。

3. **高效且实用**：
   - **训练速度**：比 DEAL 快 1.7–2.0×；
   - **推理速度**：去除路由后快 6–8×；
   - **零额外部署成本**：最终模型仍是标准 LoRA 形式。

---

### ⚠️ 局限性
1. **依赖路由专业化（routing specialization）**：
   - 当任务高度相似或特征重叠时，注意力可能无法有效分离，导致多个任务竞争少数程序槽，削弱效果。
2. **固定程序数量（N）构成瓶颈**：
   - 随着任务数量或多样性增加，有限的程序容量可能导致无关任务被迫共享槽位，影响长期保留。

---

### 🔮 未来工作方向
1. **增强路由鲁棒性**：
   - 引入熵正则化等机制鼓励程序多样性，防止注意力坍缩；
2. **动态程序扩展/剪枝**：
   - 设计可伸缩的记忆结构，按需增减程序槽；
3. **可控遗忘机制**：
   - 结合“主动遗忘”策略，清除过时或有害信息，提升安全性；
4. **应用于更多下游场景**：
   - 如个性化助手、医疗诊断系统的增量更新等。

---

## 总结
> **ProCL 通过将 LoRA 视为“程序记忆”，实现了结构化、局部化、可巩固的知识更新机制，在不增加推理负担的前提下，显著提升了 LLM 在持续学习中的稳定性与可塑性平衡能力。**

该工作不仅推动了参数高效持续学习的发展，也为构建真正具备“终身学习”能力的语言智能体提供了新范式。

</details>

---

### 16. [MARLIN: Multi-Agent Game-Theoretic Reinforcement Learning for Sustainable LLM Inference in Cloud Datacenters](https://arxiv.org/abs/2605.13496)

**Authors**: H. Moore, S. Qi, D. Milojicic, C. Bash, S. Pasricha  
**Category**: cs.DC  
**Published**: 2026-05-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.13496v1  

#### Abstract
Large Language Models (LLMs) have become increasingly prevalent in cloud-based platforms, propelled by the introduction of AI-based consumer and enterprise services. LLM inference requests in particular account for up to 90% of total LLM lifecycle energy use, dwarfing training energy costs. The risi...

---

### 17. [F-GRPO: Factorized Group-Relative Policy Optimization for Unified Candidate Generation and Ranking](https://arxiv.org/abs/2605.12995)

**Authors**: Rohan Surana, Gagan Mundada, Junda Wu, Xintong Li, Yizhu Jiao, Bowen Jin, Sizhe Zhou, Tong Yu, Ritwik Sinha, Jiawei Han, Jingbo Shang, Julian McAuley  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.12995v1  

#### Abstract
Traditional retrieval pipelines optimize utility through stages of candidate retrieval and reranking, where ranking operates over a predefined candidate set. Large Language Models (LLMs) broaden this into a generative process: given a candidate pool, an LLM can generate a subset and order it within ...

---

### 18. [Attention Once Is All You Need: Efficient Streaming Inference with Stateful Transformers](https://arxiv.org/abs/2605.13784)

**Authors**: Victor Norgren  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.13784v1  

#### Abstract
Conventional transformer inference engines are request-driven, paying an O(n) prefill cost on every query. In streaming workloads, where data arrives continuously and queries probe an ever-growing context, this cost is prohibitive. We introduce a data-driven computational model centred on stateful s...

---

### 19. [Think Twice, Act Once: Verifier-Guided Action Selection For Embodied Agents](https://arxiv.org/abs/2605.12620)

**Authors**: Nishad Singhi, Christian Bialas, Snehal Jauhri, Vignesh Prasad, Georgia Chalvatzaki, Marcus Rohrbach, Anna Rohrbach  
**Category**: cs.AI  
**Published**: 2026-05-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.12620v1  

#### Abstract
Building generalist embodied agents capable of solving complex real-world tasks remains a fundamental challenge in AI. Multimodal Large Language Models (MLLMs) have significantly advanced the reasoning capabilities of such agents through strong vision-language knowledge and chain-of-thought (CoT) re...

---

### 20. [GRACE: Gradient-aligned Reasoning Data Curation for Efficient Post-training](https://arxiv.org/abs/2605.13130)

**Authors**: Junjie Li, Ziao Wang, NingXuan Ma, Jianghong Ma, Xiaofeng Zhang  
**Category**: cs.AI  
**Published**: 2026-05-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.13130v1  

#### Abstract
Existing reasoning data curation pipelines score whole samples, treating every intermediate step as equally valuable. In reality, steps within a trace contribute very unevenly, and selecting reasoning data well requires assessing them individually. We present GRACE, a gradient-aligned curation metho...

---

### 21. [OmniThoughtVis: A Scalable Distillation Pipeline for Deployable Multimodal Reasoning Models](https://arxiv.org/abs/2605.11629)

**Authors**: Yuanhao Yue, Chengyu Wang, Yuanjie Lyu, Lei Shen, Jun Huang  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.11629v1  

#### Abstract
Recent multimodal large language models (MLLMs) have shown strong chain-of-thought (CoT) reasoning ability on vision-language tasks, but their direct deployment in real-world systems is often limited by latency and resource constraints. In practice, smaller MLLMs are preferred for online serving, ye...

---

### 22. [On Predicting the Post-training Potential of Pre-trained LLMs](https://arxiv.org/abs/2605.11978)

**Authors**: Xiaoyuan Li, Yubo Ma, Kexin Yang, Moxin Li, Keqin Bao, Wenie Wang, Fuli Feng, Dayiheng Liu  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.11978v1  

#### Abstract
The performance of Large Language Models (LLMs) on downstream tasks is fundamentally constrained by the capabilities acquired during pre-training. However, traditional benchmarks like MMLU often fail to reflect a base model's plasticity in complex open-ended scenarios, leading to inefficient model s...

---

### 23. [Efficient and Portable Support for Overdecomposition on Distributed Memory GPGPU Platforms](https://arxiv.org/abs/2605.12734)

**Authors**: Aditya Bhosale, Anant Jain, Shourya Goel, Ritvik Rao, Peddoju Sateesh Kumar, Laxmikant Kale  
**Category**: cs.DC  
**Published**: 2026-05-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.12734v1  

#### Abstract
Overdecomposition has emerged as a powerful and sometimes essential technique in parallel programming. Many application domains or frameworks, including those based on adaptive mesh refinements, or tree codes use it. Charm++ is a parallel programming system which has demonstrated the utility of over...

---

### 24. [Learning When to Act: Communication-Efficient Reinforcement Learning via Run-Time Assurance](https://arxiv.org/abs/2605.12561)

**Authors**: Adam Haroon, Erick J. Rodr\'iguez-Seda, Cody Fleming, Tristan Schuler  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.12561v1  

#### Abstract
Safe reinforcement learning (RL) typically asks $\textit{what}$ an agent should do. We ask $\textit{when}$ it needs to act, and show that a single policy can jointly learn control inputs and communication-efficient timing decisions under a pointwise Lyapunov safety shield. We focus on stabilization ...

---

### 25. [Learning with Rare Success but Rich Feedback via Reflection-Enhanced Self-Distillation](https://arxiv.org/abs/2605.12741)

**Authors**: Yuwei Zhang, Sha Li, Changlong Yu, Qin Lu, Shuowei Jin, Chengyu Dong, Haoran Liu, Ilgee Hong, Xintong Li, Zhenyu Shi, Bing Yin, Jingbo Shang  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.12741v1  

#### Abstract
Enabling Large Language Models (LLMs) to continuously improve from environmental interactions is a central challenge in post-training. While on-policy self-distillation offers a promising paradigm, existing methods predominantly treat environmental feedback as a passive conditioning signal. Conseque...

---

### 26. [Byzantine-Robust Distributed Sparse Learning Revisited](https://arxiv.org/abs/2605.13283)

**Authors**: Yuxuan Wang, Lixin Zhang, Kangqiang Li  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.13283v1  

#### Abstract
We revisit Byzantine robust distributed estimation for high-dimensional sparse linear models. By combining local $\ell_1$-regularized robust estimation with robust aggregation at the server, the framework applies to pseudo-Huber regression, quantile regression, and sparse SVM. We show that the resul...

---

### 27. [Uncertainty-Aware Prediction of Lung Tumor Growth from Sparse Longitudinal CT Data via Bayesian Physics-Informed Neural Networks](https://arxiv.org/abs/2605.13560)

**Authors**: Lingfei Kong, Haoran Ma  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.13560v1  

#### Abstract
This work studies lung tumor growth prediction from sparse and irregular longitudinal computed tomography (CT) observations with measurement variability. A Bayesian physics-informed neural network is developed by combining Gompertz growth dynamics with low-dimensional Bayesian inference in the log-v...

---

### 28. [Force-Aware Neural Tangent Kernels for Scalable and Robust Active Learning of MLIPs](https://arxiv.org/abs/2605.13788)

**Authors**: Eszter Varga-Umbrich, Zachary Weller-Davies, Paul Duckworth, Jules Tilly, Olivier Peltre, Shikha Surana  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.13788v1  

#### Abstract
Active learning for machine-learning interatomic potentials (MLIPs) must address several challenges to be practical: scaling to large candidate pools, leveraging energy-force supervision, and maintaining robustness when candidate pools are biased relative to the target distribution. In this work, we...

---

### 29. [Sampling More, Getting Less: Calibration is the Diversity Bottleneck in LLMs](https://arxiv.org/abs/2605.11128)

**Authors**: Amin Banayeeanzade, Qingchuan Yang, Dhruv Tarsadiya, Fatemeh Bahrani, Leonardo Blas, Alfy Samuel, Robin Jia, Meisam Razaviyayn, Sai Praneeth Karimireddy  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.11128v1  

#### Abstract
Diversity is essential for language-model applications ranging from creative generation to scientific discovery, yet modern LLMs often collapse into a narrow subset of plausible outputs. While prior work has developed benchmarks for measuring this lack of diversity, less is known about how the step-...

---

### 30. [Combining On-Policy Optimization and Distillation for Long-Context Reasoning in Large Language Models](https://arxiv.org/abs/2605.12227)

**Authors**: Miguel Moura Ramos, Duarte M. Alves, Andr\'e F. T. Martins  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.12227v1  

#### Abstract
Adapting large language models (LLMs) to long-context tasks requires post-training methods that remain accurate and coherent over thousands of tokens. Existing approaches are limited in several ways: 1) off-policy methods such as supervised fine-tuning (SFT) and knowledge distillation (KD) suffer fr...

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

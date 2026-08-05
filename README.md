# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-05 08:12:29 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [When Does Disaggregation Pay? Simulating Prefill--Decode--Attention--FFN Specialization for Agentic LLM Inference](https://arxiv.org/abs/2608.03741)

**Authors**: Przemyslaw Forys, Haoran Wu, Can Xiao, Jiayi Nie, Tony Liu, Rika Antonova, Timothy Jones, Robert Mullins, Wayne Luk, Aaron Zhao, George A. Constantinides  
**Category**: cs.DC  
**Published**: 2026-08-05  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2608.03741v1  

#### Abstract
Agentic inference now dominates the LLM inference landscape, requiring LLMs to actively engage in multi-turn interactions with tool-calling capabilities. This introduces a more complex workload for the underlying inference system: serving stages such as prefill and decode exhibit substantially diffe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：When Does Disaggregation Pay? Simulating Prefill--Decode--Attention--FFN Specialization for Agentic LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Agentic LLM Inference**（如工具调用、多轮交互代理）导致上下文长度急剧增长（可达 100K tokens），使得传统的单体 GPU 架构难以高效处理。不同推理阶段（Prefill 和 Decode）以及子模块（Attention 与 FFN）在计算、内存带宽需求上差异显著，单一硬件配置无法最优支持所有阶段。

现有系统虽已开始采用 **Prefill-Decode (PD) Disaggregation**，但对更细粒度的异构硬件适配（如 Attention vs. FFN）缺乏系统性探索。

### ✅ 提出的新方法：HeteroPanacea
作者提出一个名为 **HeteroPanacea** 的仿真框架，用于模拟和优化 **disaggregated serving** 架构下的 LLM 推理性能。

#### 创新维度：
- **Disaggregated Quantization**：允许每个阶段独立设置量化精度（如 Prefill-Attention 用 FP16，Decode-FFN 用 INT4）。
- **Automated Parallelization Scheduling**：自动搜索最优的 Tensor Parallelism (TP)、Pipeline Parallelism (PP)、Data Parallelism (DP) 和 Expert Parallelism (EP) 配置。
- **PDAF NPU Architectural Heterogeneity**：将推理划分为四个独立阶段：
  - Prefill-Attention (PA)
  - Prefill-FFN (PF)
  - Decode-Attention (DA)
  - Decode-FFN (DF)
  每个阶段可分配不同的 NPU 架构（compute/memory/bandwidth）。

### ✅ 相比现有方法的优势
| 特性 | HeteroPanacea | 其他工具（如 LLMCompass, MemExplorer） |
|------|----------------|-------------------------------|
| 支持 PDAF 四阶段拆分 | ✅ | ❌ |
| 支持 per-stage 量化 | ✅ | ❌ 或有限 |
| 支持 custom NPU 设计空间搜索 | ✅ | ❌（固定硬件） |
| 联合优化 hardware + parallelism + quantization | ✅ | ❌ |

> 📌 **优势总结**：HeteroPanacea 是首个支持跨栈（cross-stack）、多维联合设计空间探索的仿真器，能准确建模 disaggregation 在真实 workload 下的收益边界。

---

## 2. 核心实验方法和设置

### ✅ 使用的模型与数据集
- **模型集合**（共8个）：
  - Dense: Llama-3.1-405B, Qwen3-235B-A22B, GLM-4.6
  - MoE: DeepSeek-V4-Pro/Flash, Llama4-Scout/Maverick, GPT-OSS
- **Workload 参数化**：
  - 输入/输出比例（I/O ratio）从 `0.01` 到 `1000` 扫描
  - 输出长度固定为 1000 tokens
  - 请求服从正态分布，模拟 500 个请求 @ 125 req/sec
  - 代表任务：BFCL（tool calling）、GSM8K（reasoning）
- **Context Length**：OSWorld 上平均达 38K tokens，最高达 100K

### ✅ 实验设置
- **两种硬件环境对比**：
  1. **Custom NPU Design Space**：基于 PLENA 架构，自由组合 compute (25–20,000 TFLOPS) 和 memory（SRAM/HBM/GDDR等）
  2. **Commercial GPUs**：AWS EC2 实例（H100/A100/L40S/T4/V100/M60），按成本预算分配
- **Interconnect Modeling**：
  - Intra-node: NVSwitch (3600 GB/s)
  - Inter-node: InfiniBand (50 GB/s)

### ✅ 评估指标
- **Throughput (tokens/sec)**：主要性能指标
- **Latency (TTFT, TPOT)**
- **Power/Cost Efficiency**
- **Accuracy Impact of Quantization**

### ✅ 基线方法对比
| 方法 | 描述 |
|------|------|
| **ND (No Disaggregation)** | 所有阶段运行在同一设备上（传统方式） |
| **PD Disaggregation** | 分离 Prefill 与 Decode 阶段 |
| **AF Disaggregation** | 分离 Attention 与 FFN 模块（仅 Decode 中） |
| **PDAF Disaggregation** | 四阶段完全分离（本文提出） |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### 🔹 在 Custom NPU 上的结果（图5 & 图6）
- 当 I/O = 100（即输入 100K tokens）时：
  - **PDAF 达到最高吞吐增益**：相比 ND 提升 **1.05–1.92×**
  - 最佳案例：Llama-4 Maverick 达到 **1.81×**，Scout 达到 **1.77×**
  - **PD 也优于 ND**（7/8 模型胜出），但弱于 PDAF
  - **AF 表现最差**：始终低于 ND（0.20–0.65×）

- **转折点（Crossover Point）**：
  - PDAF 只有在 **I/O ≥ 10** 时才超过 ND
  - 在 I/O = 1 时仍仅为 ND 的 0.48×
  - 在 I/O = 1000 时回落至 1.27×（因 Decode 不再是瓶颈）

> 💡 **结论**：**Prefill-heavy workloads 是 disaggregation 发挥作用的前提**

#### 🔹 在 Commercial GPUs 上的结果（图7）
- 即使在轻量级 I/O=0.01 时，**PD 已优于 ND**（1.18–2.50×）
- 但在高 I/O 场景下表现不稳定：
  - PD 在 8/8 模型中优于 ND @ I/O=0.01
  - 但在 I/O=1 时仅 4/8 模型优于 ND
  - **PDAF 并未稳定优于 PD**，甚至多数情况下被反超

> ⚠️ **原因分析（见 Table V）**：
- GPU 的 compute 与 memory bandwidth 强耦合，无法像 NPU 那样独立调节
- 几乎所有阶段都使用 H100（相同 compute-to-bandwidth ratio），导致无法真正实现“stage-specialized”硬件

### ✅ 消融实验结果（Ablation Study, Table VI）

| 因子 | 发现 |
|------|------|
| **KV Traffic (DA_AI)** | ↑ kv_lora_rank → ↑ PDAF 增益（+67%~100%）<br>说明：KV 流量越大，DA 与 DF 差异越明显，PDAF 越有利 |
| **Sparsity (num_active_experts)** | ↑ 活跃专家数 → ↓ PDAF 增益（最多下降 73%）<br>原因：FFN 计算压力上升，削弱其与 Attention 的差异 |
| **Capacity (total experts)** | 扩大总专家数量对 PDAF 影响极小（<0.5%）<br>→ 内存容量不是瓶颈 |
| **FFN Size (ffn_expansion)** | 扩大 FFN 尺寸会轻微降低 PDAF 增益（~30%） |
| **Precision (dtype_bytes)** | 效果非单调：<br>- 对 GPT-OSS：FP4 提升 57%<br>- 对 GLM-4.6：FP4 导致下降 41%<br>→ 精度优化需 workload-aware |

### ✅ 量化敏感性测试（Quantization Sensitivity, Table VII）
- 在 Qwen3.5-32B 上进行：
  - 统一 4-bit → Accuracy 崩溃（GSM8K: 75% → 11%，BFCL: 21% → 6%）
  - 单阶段降为 4-bit：
    - **降低 FFN 精度** → 严重损害 GSM8K 性能（↓ 至 15–47%），但对 BFCL 影响小
    - **降低 Attention 精度** → 显著影响 BFCL（↓ 至 11–12%），但对 GSM8K 几乎无损

> 🎯 **关键洞察**：**哪个阶段可以安全低精度化，取决于 workload 类型，而非模型本身**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Disaggregation 是否“Pay Off”高度依赖 workload 特征**：
   - 只有当 **Prefill 占主导（I/O > 10）** 时，PD/PDAF 才能带来显著收益
   - 在 decode-heavy 场景中，disaggregation 开销大于收益

2. **四阶段 PDAF Disaggregation 最具潜力**：
   - 在 custom NPU 上，PDAF 是最一致提升吞吐的方法（up to 1.92×）
   - 成功的关键在于 **Attention 与 FFN 的硬件需求本质不同**，必须通过异构架构才能释放红利

3. **Hardware Flexibility 决定 Disaggregation 收益上限**：
   - 在 GPU 上由于 compute 与 memory 耦合，PDAF 无法发挥优势
   - **只有当硬件允许 stage-specialized design 时，细粒度 disaggregation 才值得部署**

4. **量化策略必须 workload-aware**：
   - 不能全局统一量化
   - FFN 更适合在长 Prefill 场景中保持高精度，Attention 则相反

### ✅ 局限性
- **仿真器未端到端验证**：组件模型经过校准，但完整调度行为尚未在真实 disaggregated 系统上验证
- **量化研究范围有限**：仅在一个模型（Qwen3.5-32B）上做了敏感性分析，缺少大规模泛化
- **GPU 实验受限于定价模型**：结果可能随云服务商价格变化而改变

### ✅ 未来工作方向
1. **构建真实 disaggregated serving stack 进行验证**
2. **开发轻量级 accuracy proxy**，加速 per-stage quantization 搜索
3. **建立预测模型**：根据 model config（如 kv_lora_rank, sparsity）直接判断是否应启用 PDAF
4. **扩展至更多 disaggregation 维度**：如 MoE expert routing disaggregation

---

> 🏁 **最终结论一句话总结**：  
> **Disaggregation only pays when the workload is prefill-heavy and the hardware allows genuine architectural specialization — especially between Attention and FFN. PDAF is the most promising path forward for next-gen agentic LLM serving, but only if we move beyond commodity GPUs toward truly heterogeneous NPUs.**

</details>

---

### 2. [FedCritic-MIMO: Communication-Efficient Serverless Federated Critic Learning for Massive-MIMO Resource Control in Open and Disaggregated 6G RANs](https://arxiv.org/abs/2608.03852)

**Authors**: Amin Farajzadeh, Melike Erol-Kantarci  
**Category**: cs.LG  
**Published**: 2026-08-05  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2608.03852v1  

#### Abstract
This paper proposes FedCritic-MIMO, a communication-efficient serverless federated multi-agent reinforcement learning framework for AI-native resource control across independently deployable cell-level controllers in open and disaggregated 6G RANs. Controllers share no trainer, retain local actors a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FedCritic-MIMO

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**开放化、解耦化（open and disaggregated）的6G RAN架构**中，多个独立部署的小区级控制器（cell-level controllers）在缺乏中央训练器（common trainer）的情况下，如何实现高效协同资源控制这一挑战。具体而言，在**reuse-1 多小区 massive-MIMO OFDMA网络**中，由于强跨小区干扰（inter-cell interference）和小区内SDMA干扰的存在，传统的集中式或联邦学习方法难以适用，因为它们通常依赖于：
- 集中的轨迹收集（centralized trajectory collection）
- 参数服务器聚合（parameter-server aggregation）
- 或者同质化的策略模型（actor homogenization）

这些要求与开放RAN中各控制器独立运营、保留本地模型隐私的设计理念相冲突。

---

### 提出的新方法与核心思路
作者提出了 **FedCritic-MIMO** —— 一种**通信高效的无服务器联邦批评家学习框架（communication-efficient serverless federated critic learning）**，用于多智能体强化学习（MARL）下的联合调度、功率分配与波束成形控制。

#### 核心创新点包括：

1. **Serverless Critic Federation 架构**
   - 每个BS保留其**本地actor**、个性化critic头（personalized critic head）以及本地经验。
   - 只有预定义的**共享critic子网络（shared critic subnetwork）参数**通过**对等（peer-to-peer）方式**在邻居间交换。
   - 完全去除了中心化参数服务器和全局聚合机制，支持真正的“serverless”协作。

2. **无线感知的事件触发通信机制（Utility-Aware Event Triggering）**
   - Critic更新仅在满足一定“重要性”时才触发通信。
   - 触发条件综合考虑三个因素：
     - **Critic创新度**（参数变化）
     - **队列紧迫性**（QoS压力）
     - **干扰强度**（物理层耦合程度）
   - 实现按需通信，避免冗余传输。

3. **自适应分层Top-k稀疏压缩 + Error Feedback**
   - 对共享critic增量进行**逐层top-k稀疏化**，显著减少通信开销。
   - 引入**error feedback机制**补偿因压缩带来的偏差，保证收敛性。

4. **基于干扰图的平衡融合机制（Balanced Interference-Aware Fusion）**
   - 融合权重由**双向干扰相关性**决定，优先采纳来自强干扰邻居的信息。
   - 使用对称边权重确保融合矩阵为双随机（doubly stochastic），保障平均一致性。

5. **理论保证**
   - 在固定策略、冻结目标的critic回归模型下，建立了**有限时间内的平稳性（stationarity）与共识（consensus）边界**。
   - 收敛速率可达 $ O(T^{-1/2}) + O(\log T / T) $。

---

### 相比现有方法的优势
| 维度 | FedCritic-MIMO | 传统方法（如CTDE/FedAvg） |
|------|----------------|----------------------------|
| 架构兼容性 | ✅ 支持独立部署、异构控制器 | ❌ 依赖中心训练器或同质模型 |
| 通信效率 | ✅ 降低约76% critic通信量 | ❌ 周期性全模型同步开销大 |
| 协同效果 | ✅ 干扰感知融合提升协调质量 | ❌ 无差别平均稀释局部价值信息 |
| 隐私保护 | ✅ 不共享actor与原始数据 | ❌ 可能暴露完整模型 |

---

## 2. 核心实验方法和设置

### 实验环境设置
- **网络拓扑**：7个BS，每个服务8个UE，共56个UE
- **频谱配置**：
  - 子载波数 $ K = 16 $
  - 频率复用模式：Reuse-1
  - 总带宽 $ B $，子载波间隔 $ \Delta f = B/K $
- **天线配置**：
  - 每BS配备 $ L_n = 32 $ 根发射天线
  - 最大空间流数 $ S_n = 3 $ per subcarrier
- **信道模型**：
  - 小尺度衰落：时空相关的Rayleigh衰落（Gauss-Markov, $ \rho = 0.55 $）
  - 大尺度衰落：路径损耗 + 对数正态阴影（log-normal shadowing）
- **QoS约束**：最小速率目标 $ R_{\text{min}} = 1.9 $（归一化单位）

### 控制任务
每时隙决策变量包括：
- 用户调度 $ x_{n,k,m}(t) \in \{0,1\} $
- 每流功率 $ p_{n,k,m}(t) \geq 0 $
- 波束成形向量 $ \mathbf{v}_{n,k,m}(t) \in \mathbb{C}^{L_n \times 1} $

优化目标为最大化长期加权吞吐量并满足QoS需求，建模为一个**Dec-POMDP**问题。

---

### 评估指标
| 指标类别 | 具体指标 |
|--------|---------|
| **性能指标** | - 持久化episode奖励（held-out reward）<br>- 网络总吞吐量<br>- 用户平均速率分布（CDF）<br>- 平均SINR |
| **服务质量** | - QoS满足率（long-term QoS satisfaction ratio） |
| **干扰管理** | - 单位吞吐量的干扰成本（interference cost per bit） |
| **通信效率** | - 训练阶段critic参数通信总量（Gbits） |

---

### 基线方法对比
| 类型 | 方法名称 | 描述 |
|-----|----------|------|
| **启发式方法** | Random, Greedy-MaxGain, Greedy-Queue, Greedy-IA-Queue | 非学习型调度策略 |
| **独立学习** | Strict-Independent-PPO | 无任何critic共享 |
| **无联邦但相同观测** | No-Federation-IA-PPO | 含干扰感知特征，但不通信 |
| **集中训练+分散执行** | CTDE-MAPPO | 中央critic训练，本地执行 |
| **通信消融变体** | Periodic-Full, Event-Uncompressed | 分别测试周期通信 vs. 事件触发；是否压缩 |
| **提出方法** | **Proposed (FedCritic-MIMO)** | 完整框架：事件触发 + 分层压缩 + 干扰感知融合 |

所有方法使用相同的actor结构、PPO超参、warm-start初始化，差异仅在于信息交互范围与critic协调机制。

---

## 3. 主要实验结果和性能指标

### ✅ 性能全面领先

#### （1）持久化奖励（Held-out Episodic Reward）
- **FedCritic-MIMO**: ~57.0
- CTDE-MAPPO: ~53.1 (+7.2% 提升)
- Event-Uncompressed: ~53.3 (+6.8%)
- No-Federation-IA-PPO: ~52.2 (+9.1%)

> 表明所提方法泛化能力最强，在未见信道条件下仍保持最优表现。

#### （2）QoS满足率（Long-term QoS Satisfaction）
- **FedCritic-MIMO**: ~0.78
- CTDE-MAPPO: ~0.77
- No-Federation-IA-PPO: ~0.74
- Strict-Independent-PPO: ~0.69

> 接近最佳集中训练方法，显著优于非协作方法。

#### （3）平均SINR
- **FedCritic-MIMO**: **-2.3 dB**
- Event-Uncompressed: -4.0 dB (**↑1.7 dB**)
- CTDE-MAPPO: -4.8 dB (**↑2.5 dB**)
- Periodic-Full: -5.6 dB

> 显著改善信号质量，说明协同调度有效抑制了干扰。

#### （4）用户速率分布（Per-UE Rate CDF）
- 曲线整体右移，尤其在低分位段（如10%-50%）增益明显。
- 更多边缘用户获得更高服务速率，体现公平性提升。

#### （5）干扰效率（Interference Cost per Unit Sum Rate）
| 方法 | 干扰成本 ($ \times 10^{-4} $) | 相对降低 |
|------|-------------------------------|----------|
| **FedCritic-MIMO** | **1.78** | — |
| CTDE-MAPPO | 1.93 | ↓8% |
| No-Federation-IA-PPO | 1.98 | ↓10% |
| Event-Uncompressed | 2.03 | ↓12% |
| Periodic-Full | 2.40 | ↓26% |

> 所提方法以最低干扰代价换取最高吞吐量，实现更高效的资源利用。

---

### 📉 通信开销大幅下降

#### 累计训练侧通信量（Cumulative Communication Overhead）
- **FedCritic-MIMO**: **2.65 Gbits**
- CTDE-MAPPO: 3.65 Gbits
- Periodic-Full / Event-Uncompressed: ~11.0–11.2 Gbits

> **相比未压缩分布式critic交换，通信量减少约76%！**

> 相比CTDE也降低了约27%，体现了其在性能与通信之间的优越权衡。

> 注：该通信指训练期间critic模型参数交换，不含用户面流量或前传链路。

---

### 🔍 消融实验分析（Ablation Study）

| 方法变体 | 通信量 | 奖励 | 说明 |
|--------|-------|------|------|
| Periodic-Full | 11.2 Gbits | ~50.3 | 周期性全参数同步效率低下 |
| Event-Uncompressed | 11.0 Gbits | ~53.3 | 事件触发本身收益有限 |
| **Proposed** | **2.65 Gbits** | **~57.0** | **压缩是降通信主因，融合机制提性能** |

> 结论：**分层top-k压缩 + error feedback 是实现高通信效率的关键**；而干扰感知融合进一步提升了协同性能。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Serverless Critic Sharing 是可行且高效的**  
   在无需中心节点的前提下，仅交换兼容的共享critic参数即可实现高性能协同控制。

2. **通信应由“无线语义”驱动**  
   单纯减少通信次数（event triggering）不足以带来显著增益；必须结合**压缩技术**与**语义感知的融合机制**才能同时优化性能与效率。

3. **干扰图可作为自然协作拓扑**  
   利用物理层干扰关系构建协作图（coordination graph），使信息交换更具针对性。

4. **性能-通信权衡达到最优**  
   FedCritic-MIMO在所有协调学习方法中实现了最佳的**performance-communication operating point**。

---

### ⚠️ 局限性
1. **假设理想P2P通信**  
   当前仿真假设可靠的对等通信，未考虑实际网络中的延迟、丢包等问题。

2. **静态网络拓扑**  
   实验基于固定BS位置与连接图，未考虑移动性或动态拓扑变化。

3. **CSI完整性假设**  
   假设本地可获取直接链路CSI及部分交叉链路估计，未处理不完美或缺失CSI场景。

4. **同步训练假设**  
   当前算法采用同步训练轮次，未来需扩展至异步协调。

---

### 🔮 未来工作方向
- 扩展至**异步peer coordination**
- 引入**mobility-aware interference graphs**
- 研究**不完美CSI下的鲁棒性**
- 探索更大规模、异构部署场景
- 结合**RIS、AI-Radio Sensing**等新兴6G技术

---

## 总结

📌 **FedCritic-MIMO 成功解决了开放化6G RAN中独立控制器间的协同难题**，提出了一种**去中心化、高效率、强鲁棒性的联邦critic学习范式**。它不仅在**吞吐量、QoS、SINR、用户公平性等方面超越现有方法**，还通过**事件触发 + 分层压缩 + 干扰感知融合**将critic通信开销**降低76%**，为未来6G智能无线资源管理提供了极具前景的技术路径。

</details>

---

### 3. [Pruning-Aware Multi-Cluster Co-Inference for Large AI Models in AI-RANs](https://arxiv.org/abs/2608.03026)

**Authors**: Xiaowen Cao, Zhonghao Lyu, Shicheng Chu, Zezhong Zhang, Dingzhu Wen, Guangxu Zhu, Kaibin Huang, Shuguang Cui, Jie Xu  
**Category**: cs.DC  
**Published**: 2026-08-05  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.03026v1  

#### Abstract
The increasing scale and computational demands of large artificial intelligence models (LAIMs) present significant challenges for efficient inference in resource-constrained distributed environments. In this paper, we propose a multi-cluster LAIM co-inference framework, where an edge server equipped...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Pruning-Aware Multi-Cluster Co-Inference for Large AI Models in AI-RANs**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对在资源受限的边缘网络（AI-RANs）中部署大规模人工智能模型（LAIMs）所面临的挑战，提出了一种**多集群协同推理框架**。主要解决以下三个核心问题：
- **资源竞争**：多个用户集群共享有限的无线带宽和边缘GPU资源，导致任务调度与资源分配复杂。
- **模型压缩与性能权衡**：模型剪枝（pruning）虽能降低计算与通信开销，但会损害推理精度，需系统性建模其影响。
- **设备异构性**：不同设备因视角、传感器质量差异，对最终推理结果的贡献不均，传统均匀处理策略次优。

### **提出的新方法与新思路**
论文提出了一个**剪枝感知的多集群协同推理框架（Pruning-Aware Multi-Cluster Co-Inference Framework）**，其核心创新包括：

- **多集群协同架构设计**  
  构建了一个由多GPU边缘服务器协调多个用户集群的系统，每个集群内设备从多视角采集数据并运行轻量级本地LAIM提取特征，服务器聚合后执行下游推理。

- **基于率失真理论与部分信息分解（PID）的分析框架**  
  首次将**rate-distortion theory**用于建模剪枝比例与推理失真的关系，并引入**Partial Information Decomposition (PID)** 和 **Shapley Value** 来量化不同设备的独特（unique）、冗余（redundant）和协同（synergistic）信息贡献，定义设备重要性系数 $ \alpha_{m,k} $。

- **联合优化问题建模与高效求解算法**  
  将模型剪枝比、任务调度、带宽分配、传输功率等变量统一建模为一个**非凸混合整数非线性规划（MINLP）问题**，目标是最小化推理失真，满足延迟与能耗约束。采用**交替优化 + SCA（Successive Convex Approximation）** 策略高效求解。

### **相比现有方法的优势**
| 维度 | 现有方法局限 | 本文优势 |
|------|----------------|---------|
| **场景覆盖** | 多为单用户或单集群 | 支持**多集群并发请求**，考虑资源竞争 |
| **模型压缩建模** | 忽视剪枝对融合推理的影响 | 引入**率失真分析 + 设备贡献加权**，更精准刻画失真 |
| **设备异构性** | 假设设备同质或平均处理 | 利用**PID + Shapley Value**实现**重要性感知剪枝与资源分配** |
| **优化粒度** | 分离优化通信或计算 | 实现**剪枝、调度、通信、功率联合优化** |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **图像分类任务**：`CIFAR-10`，每张图像划分为三个局部视图（cropping ratios: 0.2, 0.4, 0.6），模拟多设备协作感知。
- **视频到文本描述任务**：`MSR-VTT`，采样8帧（224×224），使用 `TinyViT` 编码器 + `Qwen3-4B-Base` 下游生成器。

### **实验设置**
- **系统参数**（见Table I）：
  - 集群数 $ M = 8 $，每集群设备数 $ K_m = 3 $
  - GPU数 $ G = 2 $，队列长度 $ Q_g = 8 $
  - 总带宽 $ B = 5 $ MHz，噪声谱密度 $ N_0 = 2 \times 10^{-19} $ W/Hz
  - 信道模型：i.i.d. Rayleigh衰落，路径损耗 $ 10^{-5} $

- **模型配置**：
  - 图像任务：ViT-Tiny，分割点在第3个Transformer层
  - 视频任务：TinyViT + LoRA适配器

### **评估指标**
| 任务 | 主要指标 | 辅助指标 |
|------|----------|-----------|
| 图像分类 | **Classification Accuracy** | 推理延迟、能耗 |
| 视频描述 | **BLEU-4**, **CIDEr** | 推理延迟、能耗 |

此外，通过**压力测试**（提升带宽至120/160 MHz）可视化任务级延迟分解（arrival, waiting, server computing）。

### **基线方法对比**
1. **Fixed power**：所有设备以最大功率传输
2. **Fixed bandwidth**：总带宽均分给所有设备
3. **Round-robin scheduling**：循环分配GPU队列，忽略负载状态
4. **Random scheduling**：随机分配任务
5. **Equal importance**：所有设备重要性系数设为 $ \alpha_{m,k} = 1/K_m $

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **图像分类任务（CIFAR-10）**
- 在 $ E_0 = 1.35 $ J, $ T_0 = 1.40 $ s 下：
  - **本文方法准确率 ≈ 0.80**
  - 固定功率/带宽 ≈ 0.70–0.75
  - 轮询/随机调度 ≈ 0.65–0.70
  - **相对提升约 5–15%**

#### **视频到文本任务（MSR-VTT）**
- 在 $ E_0 = 3.92 $ J, $ T_0 = 11.8 $ s 下：
  - **BLEU-4 ≈ 0.30**, **CIDEr ≈ 0.40**
  - 基线方法 BLEU-4 ≈ 0.20–0.25, CIDEr ≈ 0.30–0.35
  - **相对提升约 20–50%**

### **与基线方法的对比结果**
- **在所有能量与延迟阈值下，本文方法均显著优于所有基线**，尤其在资源紧张时优势更明显。
- **固定功率/带宽**：缺乏灵活性，无法适应信道变化与任务需求。
- **轮询/随机调度**：导致GPU负载不均衡，高负载任务排队时间长（见图6、7）。
- **Equal importance**：忽视设备贡献差异，在关键视图上过度剪枝，影响融合质量。

### **消融实验与关键观察（来自图8）**
- **设备重要性与资源分配强相关**：
  - 在剪枝程度较深的集群（如 m=4, m=8）中，高重要性设备（Device 2, 3）获得更高保留率 $ p_{m,k} $ 和更大带宽分配。
  - 在接近无剪枝的集群（m=3, m=5），带宽分配趋于均匀，说明“剪枝点”是调节重要性影响的关键开关。
- **剪枝与通信耦合机制被有效建模**：重要性不仅影响剪枝，还通过上传延迟间接影响带宽分配决策。

---

## **4. 关键结论和发现**

### **主要发现**
1. **多集群资源竞争必须统一建模**：通信、计算、调度高度耦合，分离优化无法达到全局最优。
2. **设备贡献异质性不可忽略**：利用 **PID + Shapley Value** 可有效识别关键设备，指导差异化剪枝与资源倾斜。
3. **剪枝不仅是压缩手段，更是控制变量**：剪枝比直接影响通信负载与计算延迟，应作为联合优化的一阶变量。
4. **重要性感知资源分配在中度剪枝区最有效**：当模型未完全保留时，重要性才能有效驱动带宽与保留率分配。

### **方法的局限性**
- **静态假设**：当前框架假设设备集群、任务请求、信道条件静态，未考虑移动性与动态环境。
- **离线重要性估计**：Shapley Value 需基于历史数据离线计算，难以实时响应快速变化的感知场景。
- **仅考虑剪枝**：未集成量化（quantization）、知识蒸馏等其他压缩技术。
- **内存瓶颈未建模**：实际LAIM部署中KV-cache、显存容量也是关键限制，本文未显式建模。

### **未来工作方向**
1. **扩展至动态环境**：支持用户移动、任务到达动态性、时变信道与负载。
2. **融合多种压缩技术**：构建**pruning + quantization + distillation** 的联合压缩-协同推理框架。
3. **自适应多智能体协作**：支持异构基础模型间的协作，联合优化模型选择、特征融合与资源分配。
4. **在线重要性更新机制**：设计轻量级在线算法动态调整 $ \alpha_{m,k} $，适应场景变化。
5. **显式建模内存与访问开销**：将KV-cache管理、数据加载延迟纳入优化框架。

--- 

> ✅ **总结一句话**：本文首次将**设备贡献异质性**与**模型剪枝**纳入多集群协同推理的联合优化框架，通过**理论建模 + 联合优化算法**，实现了在严格资源约束下的高性能边缘AI推理，为LAIM在AI-RAN中的高效部署提供了新范式。

</details>

---

### 4. [Separating Intelligence from Inference: A Standard for Edge-Native AI Computing](https://arxiv.org/abs/2608.02608)

**Authors**: Venkat Vinjam, Krishnaiah Narukulla  
**Category**: cs.DC  
**Published**: 2026-08-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.02608v1  

#### Abstract
The artificial intelligence industry has constructed a USD 300 billion centralized data center infrastructure to serve a workload, large language model inference, that does not architecturally require centralization. This paper articulates the central architectural inefficiency of contemporary AI in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Separating Intelligence from Inference: A Standard for Edge-Native AI Computing

## 1. 论文的主要贡献和创新点

### 解决的问题
当前 AI 基础设施存在一个根本性的**架构效率问题**：将 **training（训练）** 和 **inference（推理）** 这两种性质截然不同的计算任务运行在相同的硬件上。尽管训练是集中化、资本密集型的一次性任务，而推理是可并行、低延迟、高频次的任务，行业却普遍使用如 H100/MI300X 等高端加速器来处理推理，导致巨大的能源浪费、高成本和数据主权风险。

### 提出的新方法与新思路
论文提出 **“分离原则”（Separation Principle）**，即：
> **Intelligence is trained centrally and shipped as software; inference executes on hardware near the data source — at the edge.**  
> （智能在中心训练并以软件形式分发；推理在靠近数据源的边缘设备上执行。）

基于此，论文系统性地提出了一个完整的 **edge-native AI computing 架构标准**，其核心创新点包括：

- **架构理念创新**：明确提出训练与推理应解耦，推理应下沉至边缘（个人设备与企业本地服务器），从根本上重构 AI 计算范式。
- **设备类规范（Device Class Specifications）**：
  - 提出 **Personal AI Computer (PAC)**：面向个人用户的 AI 设备标准，分为 T1（入门级）和 T2（专业级）。
  - 提出 **Corporate AI Workstation (CAW)**：面向企业的多用户共享 AI 工作站标准，支持部门级部署。
- **八组件参考架构（Reference Architectural Stack）**：为实现边缘原生 AI 提供完整技术栈，包括：
  1. **Sovereignty-Aware Routing**：基于数据敏感度的路由决策，确保合规。
  2. **Local-Network Federation Protocol**：组织内 CAW 节点间的联邦协作。
  3. **Multi-Tenant KV Cache Partitioning**：多租户环境下 KV 缓存的资源隔离与配额管理。
  4. **Distributed KV Cache**：跨节点扩展 context window，支持长上下文推理。
  5. **Thermal-Adaptive Quantization**：根据设备温度动态调整量化位宽，平衡功耗与性能。
  6. **Weight Distribution Network with TEE-Bound Licensing**：通过稀疏增量更新 + TEE 绑定授权安全分发模型权重。
  7. **Cryptographic Inference Provenance Certificates**：基于 TEE 的推理溯源证书，满足审计与合规要求。
  8. **Privacy-Preserving Telemetry**：采用差分隐私（LDP）收集操作数据，保护用户隐私。

### 相比现有方法的优势
| 维度 | 当前集中式架构 | 本文提出的边缘原生架构 |
|------|----------------|------------------------|
| **能效** | 高能耗，数据中心持续高负载 | 边缘设备按需唤醒，空闲时功耗极低 |
| **延迟** | 100–800 ms（网络往返主导） | 20–100 ms（本地计算） |
| **数据主权** | 查询数据需传至云端 | 数据保留在本地设备或企业 LAN 内 |
| **经济模型** | 按 token 收费，持续支出 | 一次性资本投入 + 可摊销电费 |
| **可靠性** | 云服务中断影响所有用户 | 本地推理不受云服务影响 |
| **更新控制** | 由厂商隐式控制 | 用户/组织显式控制，支持计划更新 |

---

## 2. 核心实验方法和设置

本论文并非传统意义上的实验性研究，而是基于**第一性原理分析**和**大规模建模推演**，结合已有硬件性能数据进行量化论证。

### 实验设置
- **目标规模**：预测到 2030 年全球有 **10 亿日活 AI 用户**，每人每天平均发起 **30 次查询**，总计约 $1.1 \times 10^{13}$ 次/年。
- **基准查询定义**：
  - 输入：1,000 个 prompt tokens
  - 输出：500 个生成 tokens
  - 模型：70B 参数模型（如 Llama-3.1-70B）
  - 推理模式：自回归生成（decode phase）

### 评估指标
- **每查询能耗（Energy per query）**：单位为 mWh，综合考虑硬件 TDP、利用率和推理时间。
- **年总能耗（Annual energy consumption）**：单位为 TWh。
- **碳排放当量（CO₂ equivalent）**：基于美国电网平均排放因子（0.385 kg CO₂/kWh）换算。
- **等效核反应堆数量**：用于直观展示节能潜力。
- **吞吐量（Tokens per second, tok/s）**
- **首次响应延迟（Time-to-First-Token, TTFT）**

### 对比的基线方法
- **Cloud Inference (Current)**：使用 8× H100 SXM5 的 DGX 节点，代表当前主流云服务架构。
- **Cloud Inference (Efficient)**：单个 H100 分摊给 32 个并发用户，代表优化后的云推理。
- **Edge Architectures**：
  - **PAC T1/T2**：基于 Apple M4 Pro / M4 Max 的个人设备。
  - **CAW T1/T2**：基于 RTX 4090 / H100/MI300X 的企业工作站。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（每查询能耗）
| 架构 | 硬件 | 每查询能耗 |
|------|------|-----------|
| 云推理（当前） | 8× H100 | ≈ 3.42 mWh |
| 云推理（高效） | 1× H100（32用户） | ≈ 3.3 mWh |
| CAW T2 | 1× MI300X（10用户） | ≈ 0.078 mWh |
| PAC T2 | Apple M4 Max | ≈ 0.52 mWh |
| PAC T1 | Apple M4 Pro | ≈ 0.51 mWh |

> ✅ **边缘架构每查询能耗仅为集中式架构的 5–10%**。

### 大规模能耗与碳排结果（10亿用户/年）
| 场景 | 年能耗 (TWh) | 碳排放 (Mt CO₂) | 等效核反应堆 |
|------|---------------|------------------|--------------|
| **Status Quo**（100% 云） | 21.9 | 8.4 | 2.3 |
| **Hybrid**（50% 云） | 12.4 | 4.8 | 1.3 |
| **Edge-Dominant**（80% 边缘） | 5.8 | 2.2 | 0.6 |
| **Edge-Only**（100% 边缘） | 2.9 | 1.1 | 0.3 |

> ✅ **从现状转向全边缘推理，每年可节省约 19 TWh 能源和 7.3 百万吨 CO₂**，相当于避免建设两座大型核电站。

### 其他关键结果
- **经济效益**：
  - 电力成本节约：约 **16 亿美元/年**（按 $0.085/kWh 计算）。
  - 资本支出节约：避免采购约 **3.5 万台 H100 GPU**，节省约 **13 亿美元**。
- **设备能力验证**：
  - Apple M4 Max 在 4-bit 量化下可达到 **10–11 tokens/s**，足以支撑高质量交互体验。
  - PAC T2 可本地运行 70B 模型，CAW T2 可支持数百用户并发。

> ❗ 论文中未包含传统意义上的消融实验（ablation study），因其重点在于架构设计与宏观分析，而非单一算法模块的验证。

---

## 4. 关键结论和发现

### 主要发现
1. **训练与推理的混淆是当前 AI 基础设施危机的根本原因**，该问题可通过架构解耦解决。
2. **现代消费级硬件已具备运行主流 LLM 推理的能力**，限制因素不再是硬件，而是软件、分发机制与组织架构。
3. **边缘原生 AI 架构在能效、延迟、数据主权和成本方面全面优于集中式架构**。
4. **19 TWh/年的节能潜力具有文明尺度意义**，对实现可持续 AI 发展至关重要。
5. **可信执行环境（TEE）、差分隐私、加密溯源等技术是构建可信边缘 AI 生态的关键支柱**。

### 方法的局限性
- **前沿模型仍依赖云端**：超大规模模型（如 >405B 参数）或需要实时互联网检索的任务仍需调用 Cloud APIs。
- **初始模型分发带宽压力**：首次下载百亿参数模型对终端网络仍有挑战，依赖高效的 CDN 与增量更新。
- **企业运维复杂性增加**：本地部署要求组织具备一定的 AI 基础设施管理能力。
- **专利与标准化进程不确定**：部分组件涉及待批专利，实际落地依赖于 IEEE/ISO 等标准组织的采纳进度。

### 未来工作方向（Open Research Problems）
1. **无云比较的模型质量估计**：如何在不发送查询到云端的情况下准确评估本地与云端模型的能力差距？
2. **边界内的联邦持续学习**：如何在不泄露专有数据的前提下，在 CAW 上实现模型的持续适应？
3. **统一的能效评测标准**：建立类似 MLPerf 的标准化 AI 能耗评测基准。
4. **跨模型家族的迁移优化**：探索不同模型架构间的权重复用机制，降低迁移成本。
5. **量化-质量前沿探索**：研究低于 2-bit 量化的实用极限及对热自适应的影响。
6. **面向推理的硬件协同设计**：设计专为边缘推理优化的 SoC，强调高内存带宽、低待机功耗与 TEE 深度集成。

---

> 📌 **结语**：  
> “Train once. Distribute as software. Infer everywhere — forever.”  
> 本文不仅是一篇学术论文，更是一份面向硬件厂商、AI 实验室、企业与政策制定者的**设计蓝图**，呼吁构建一个更高效、更私密、更可持续的下一代 AI 计算基础设施。

</details>

---

### 5. [Getting the Parameters Right: A Difficulty-Graded Benchmark and Probe-Guided Training for LLM Tool Calls](https://arxiv.org/abs/2608.03071)

**Authors**: Guoyao Yu, Xiaoqing Sun, Ziqi Huang, Shaojing Fan, Zhongyi Zhang, Xiaomeng Hu, Xiaobo Xue, Yangyang Shi, Xiong Xiao, Yang Song, Biao Lyu, Rong Wen, Xing Li, Qinming He, Shunming Zhu, Zhenguang Liu  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.03071v1  

#### Abstract
Large language model agents derive much of their capability from tool use. Existing research on tool use has largely focused on selecting the right tool and orchestrating the order of calls. However, correctly filling the parameters of a tool call is equally critical for successful execution and has...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Getting the Parameters Right: A Difficulty-Graded Benchmark and Probe-Guided Training for LLM Tool Calls*

---

## 1. 主要贡献和创新点

### 解决的问题
当前大语言模型（LLM）在工具调用（tool call）任务中的研究主要集中在**选择正确的工具**和**调用顺序的编排**上，而对**如何正确填充工具调用参数**（parameter generation）这一关键环节关注不足。然而，在真实场景（如云网络管理）中，即使工具选对，参数错误仍会导致调用失败。本文指出，参数生成面临三大挑战：
- **CH-1: Deep Nesting**（深度嵌套）：参数结构多层嵌套，需精确放置值。
- **CH-2: Inter-Field Conditional Dependencies**（字段间条件依赖）：某些字段是否必需取决于其他字段的值。
- **CH-3: Cross-Call Value Derivation**（跨调用值推导）：部分参数值需从先前调用的输出中提取。

### 提出的新方法与思路
本文提出了一种基于**隐藏状态探针信号**（probe signal）的统一框架，包含两个互补组件：

#### ✅ **Probe-Filtered Bootstrapped Training (PBT)**
- 利用一个在模型隐藏状态上训练的**线性探针**（linear probe），预测每个参数值是否正确。
- 在自训练（self-training）过程中，仅保留探针评分高的伪标签样本，用于后续微调，从而提升训练数据质量。

#### ✅ **Probe-Guided Reranking (PGR)**
- 在推理阶段，对多个采样得到的候选调用进行重排序，选择探针评分最高的作为最终输出。
- 支持多种策略：**candidate-level**, **field-level**, 或 **field-set** 策略。

此外，作者还提出了 **PARAMBENCH** —— 一个专为评估参数生成能力设计的细粒度基准。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **监督方式** | 不依赖人工标注，利用模型内部信号提供弱监督，实现高质量伪标签筛选。 |
| **通用性** | PBT 和 PGR 可独立或联合使用，适用于不同资源场景（训练时增强 vs 推理时优化）。 |
| **有效性** | 显著优于传统自训练方法（如基于 log-probability 或一致性投票），尤其在复杂参数结构上表现突出。 |

---

## 2. 核心实验方法和设置

### 数据集
- **PARAMBENCH**（本文提出）
  - 来源于真实云网络系统的 agent 执行轨迹和 81 个 API schema。
  - 包含 **1,022 个实例**，按难度分为 **L1–L5 五个等级**，依据：
    - `nesting_depth`
    - `num_conditional_dependencies`
    - `num_upstream_transfers`
    - `upstream_extract_depth`
  - 提供训练/测试划分（按 API 划分，保证泛化性）。
- **外部基准**（共6个）：
  - NESTFUL, Seal-Tools, xLAM, BFCL, API-Bank, ComplexFuncBench
  - 经过“per-call”转换，聚焦于单次调用的参数生成任务。

### 实验设置
- **模型**：
  - 开源模型：Qwen3-8B/14B, Gemma-4-12B, Ministral-3-8B, Llama-3.1-8B
  - 前沿闭源模型对比：Claude Opus 4.7, GPT-5.4, DeepSeek-V4-Pro, Qwen-3.6-Plus
- **微调方法**：
  - 使用 LoRA 进行参数高效微调。
  - PBT 中探针阈值 $ T \in \{0.8, 0.9, 0.95\} $ 通过验证集选择。
- **推理设置**：
  - PGR 使用 1 次贪心解码 + 多次采样构建候选池。
  - 探针策略通过 5 折交叉验证在训练集上选择最优。

### 评估指标
- **Exact Match (EM)**：整个参数对象完全匹配才计为正确。
- **Field-level F1**：将参数展平为 `<path, value>` 对，计算字段级别的 F1 分数，给予部分匹配奖励。

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **0-shot** | 无微调，直接提示 |
| **SeedSFT** | 在小规模黄金数据上进行监督微调 |
| **SelfTrain** | 自训练，不加过滤地使用所有自生成样本 |
| **LogprobTrain** | 基于 token log-probability 选择高置信度样本 |
| **ConsistTrain** | 基于多数投票（consensus）选择一致样本 |
| **PBT**（本文）| 基于探针信号过滤样本 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **平均 Exact Match 提升显著**：
  - 基础 SFT 将 EM 从 **19.7% → 51.6%**
  - **PBT 进一步提升至 59.6%**（+8.0 EM）
- 在最复杂的两个数据集（NESTFUL, PARAMBENCH）上，**PGR 再带来 +4.6 EM 提升**。
- **Qwen3-8B + PBT + PGR** 在 7 个数据集上的平均 EM 达到 **62.5%**，超越所有对比的开源和闭源模型。

### 与基线方法对比
| 方法 | 平均 EM (%) | 相比 SeedSFT |
|------|------------|-------------|
| 0-shot | 19.7 | - |
| SeedSFT | 51.6 | 基线 |
| SelfTrain | 53.9 | +2.3 |
| LogprobTrain | 44.2 | -7.4 ❌ |
| ConsistTrain | 48.9 | -2.7 ❌ |
| **PBT** | **59.6** | **+8.0 ✅** |

> ✅ PBT 是唯一在所有 35 个模型-数据组合中均优于 SeedSFT 的方法（McNemar 检验 $ p < 10^{-10} $）。

### 与前沿模型对比
- **Qwen3-8B + PBT + PGR** 在全部 7 个数据集上达到或超过 **Claude Opus 4.7 和 GPT-5.4** 的水平。
- 在 **3 个数据集上表现最佳**，例如在 BFCL 上以 **73.6 EM** 超越最强闭源模型（63.8 EM）。

### 消融实验结果
#### 🔹 RQ3: PGR 策略选择
- **基础模型**：field-level 策略效果最好（因多数候选都不完整）。
- **PBT 微调后模型**：保守策略（如 field-set）更优，因贪心输出已较可靠。
- PGR 即使应用于未微调模型，也能平均提升 **+4.2 EM**；结合 PBT 后再提升 **+4.6 EM**。

#### 🔹 RQ4: 按难度级别分析（Figure 5）
| 难度 | SeedSFT EM | PBT+PGR EM | 提升 |
|------|-----------|-----------|------|
| L1 | ~85% | ~85% | ≈0 |
| L2 | ~70% | ~73% | +3 |
| L3 | ~55% | ~66% | **+11** |
| L4 | ~50% | ~55% | **+5** |
| L5 | ~0% | ~12% | **+12** |

> 📌 提升主要集中在 **L3–L5** 高难度级别，说明 PBT/PGR 真正解决了深层嵌套、条件依赖和跨调用推导等结构性难题。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 隐藏状态蕴含强参数正确性信号**：
   - 在写入参数前一刻的隐藏状态，可通过简单线性探针预测其正确性，**AUC 高达 0.986**，远超 token log-probability（0.914）。
2. **探针信号可用于有效训练与推理优化**：
   - PBT 显著提升微调数据质量，避免噪声累积。
   - PGR 在推理时低成本地选出更优候选。
3. **PARAMBENCH 揭示了现有模型的真实短板**：
   - 即使是前沿模型，在深度嵌套和跨调用场景下参数生成成功率不足一半。
4. **方法可迁移且高效**：
   - 一个轻量级探针即可驱动整个流程，无需修改主模型架构。

### 局限性
1. **领域依赖性强**：
   - 探针需针对特定 dataset、model、temperature 单独训练，**不可跨域迁移**。
2. **依赖初始标注数据**：
   - PBT 需要一个小的黄金标注集来训练初始探针和 seed model。
3. **应用场景受限于云网络**：
   - PARAMBENCH 当前仅覆盖云网络 API，尽管难度分级规则具有一般性。

### 未来工作方向
- 探索更通用的探针设计，减少对特定设置的依赖。
- 将探针机制扩展到其他结构化生成任务（如 SQL 生成、代码补全）。
- 构建更多领域的难度分级 benchmark，推动参数生成能力的系统评估。
- 结合 reasoning 与 probing，实现动态纠错与自我修正。

---

> 💡 **一句话总结**：  
> 本文首次将 **LLM 工具调用中的参数生成** 作为核心问题研究，提出利用 **隐藏状态探针信号** 指导训练（PBT）与推理（PGR），并在新发布的细粒度基准 **PARAMBENCH** 上验证了其显著优越性，为提升 LLM agent 的实际执行可靠性提供了新范式。

</details>

---

### 6. [OPTD: On-Policy Transition Distillation with Consistency-Guided Adaptive Compression for Few-Step Diffusion Language Models](https://arxiv.org/abs/2608.02942)

**Authors**: Xiaocheng Lu, Hualei Zhang, Shuhan Guo, Jie Zhang, Xiaoyi Pang, Jian Liu, Haoxi Li, Bohai Gu, Haoxuan Che, Jingcai Guo, Song Guo  
**Category**: cs.CL  
**Published**: 2026-08-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.02942v1  

#### Abstract
Diffusion language models (dLLMs) can predict many tokens in parallel, but accurate generation still requires many iterative denoising steps. Few-step distillation accelerates decoding by compressing multiple teacher steps into a single student transition. However, existing methods construct supervi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# OPTD: On-Policy Transition Distillation with Consistency-Guided Adaptive Compression for Few-Step Diffusion Language Models  
**——核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **few-step dLLM distillation** 方法大多采用 **off-policy** 蒸馏策略，即在固定的教师轨迹上构建训练目标。然而，在推理时，学生模型（student）会并行释放多个 token，导致其访问的状态（partial states）与教师轨迹严重偏离。这种 **state-action mismatch** 导致监督信号失效，尤其是在压缩率较高时。

此外，简单地合并多个未来动作（future actions）可能破坏中间依赖关系，而仅匹配教师的下一步动作又无法实现有效压缩。

### 🚀 提出的新方法：OPTD
作者提出 **OPTD**（On-Policy Transition Distillation），一种基于**一致性引导的自适应压缩**（consistency-guided adaptive compression）的 on-policy 蒸馏框架，核心思想如下：

- **On-Policy State Sampling**：从学生模型自身推理路径中采样 partial states，确保训练状态分布与推理一致。
- **Frozen Question-Only Teacher**：使用一个冻结的、仅看到问题和当前 partial state 的教师模型生成保守续写（rollout），用于构造目标。
- **Consistency-Guided Adaptive Compression**：
  - 识别出与教师最终输出对齐的候选 future tokens；
  - 按当前置信度排序；
  - 选择**最长的前缀集合**，使得这些 token **联合提交后仍能保持教师 rollout 的最终结果不变**（outcome-preserving）。
- **Verified-Set Certainty Forcing**：通过 set-bottleneck 损失强制所有验证通过的 future tokens 推进到解码器的释放阈值。
- **KL Anchor Regularization**：对其他未释放位置使用冻结教师的分布进行正则化。

### 🔍 相比现有方法的优势
| 维度 | OPTD | 传统 off-policy 方法 |
|------|------|------------------|
| **状态对齐** | ✅ 使用学生实际访问的状态 | ❌ 基于固定教师轨迹 |
| **压缩灵活性** | ✅ 自适应决定每步压缩长度 | ❌ 固定压缩步数或预设调度 |
| **动作有效性** | ✅ 验证联合提交的一致性 | ❌ 仅验证单个 token 或忽略一致性 |
| **无需黄金标签** | ✅ 完全无监督目标构造 | ✅ 多数也无需，但目标质量差 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在四个具有挑战性的推理与代码生成任务上进行评估：
- **GSM8K**（5-shot）：数学应用题
- **MATH-500**（4-shot）：高等数学问题
- **MBPP**（3-shot）：Python 编程任务
- **HumanEval**（0-shot）：函数级代码生成

### ⚙️ 实验设置
- **基础模型**：基于 `LLaDA-8B-Instruct`，初始化自 `TAD-S` checkpoint。
- **训练配置**：
  - 使用 LoRA（rank=128, α=128）微调所有线性层；
  - 学习率：1e-6；
  - 响应长度：256 tokens；
  - Block size：32，最多 K=3 active blocks；
  - Confidence threshold T = 0.8。
- **推理配置**：
  - Greedy decoding，batch size=1；
  - 使用 Multi-Block 解码策略；
  - 报告指标：`Acc.`（准确率）、`TPF`（Tokens Per Forward）、`NFE`（Neural Function Evaluations）、`AUP`（Area Under the quality-constrained Performance curve）。

### 📊 评估指标
- **TPF**：衡量并行效率，越高越好；
- **Accuracy / Pass@1**：衡量生成质量；
- **AUP**：综合考虑质量和效率的面积指标，锚定 TPF=1 的保守解码为基准，积分范围限制在“不超过基准精度5个百分点”的高效点；
- 所有方法在相同 prompt、scorer 和 budget 下比较，保证公平性。

### 🆚 基线方法
- **LLaDA (vanilla)**：原始扩散语言模型，TPF≈1
- **Fast-dLLM**, **D2F**, **dParallel**：系统级优化（KV cache、并行机制）
- **d3LLM**, **TAD-Q**, **TAD-S**：主流 few-step 蒸馏方法（off-policy）
- 所有 baseline 均使用其原生 inference stack 进行完整堆栈对比。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1 & Figure 1）

| Method | Average AUP | Max TPF | Accuracy (%) |
|--------|-------------|---------|--------------|
| LLaDA (vanilla) | 46.82 | 1.00× | 46.82 |
| TAD-Q | 218.07 | 5.13× | ~47.88 |
| d3LLM | 230.50 | 6.94× | ~45.38 |
| TAD-S (**best prior**) | **245.59** | 5.70× | ~50.62 |
| **OPTD (Ours)** | **313.18** | **7.87×** | **47.21** |

> ✅ **OPTD 达到了最高的平均 AUP（313.18），比最强基线 TAD-S 提升 27.5%**  
> ✅ **TPF 达到 7.87×，是 vanilla LLaDA 的 7.87 倍，优于所有 baseline**

#### 各项任务表现亮点：
- 在 **GSM8K** 和 **HumanEval** 上 AUP 表现最佳；
- 在 **MATH-500** 上略低于 TAD-S，但仍具竞争力；
- 在 **MBPP** 上略逊于 TAD-Q，但整体平衡更优。

---

### 🔬 消融实验结果（Ablation Studies）

#### （1）**State Source 对比（On-policy vs Off-policy）**
- 图3显示：随着训练步数增加，on-policy 准确率持续上升，最终超过 off-policy +1.88pp；
- TPF 几乎持平 → 改进来自更好的状态对齐而非更多并行。

#### （2）**Loss 设计对比（Table 2）**
| Loss Type | Avg Acc | TPF |
|----------|--------|-----|
| Mean Hinge | 46.04 | 7.44 |
| Projected KL | 46.20 | 7.19 |
| **Set Bottleneck (OPTD)** | **47.62** | **7.20** |

→ **Set-bottleneck 更关注最难释放的 token，提升整体准确性**

#### （3）**Verification Unit 分析**
- 独立验证（independent）允许更多 token 被选中（2.875 vs 1.327），但 **10 out of 416 联合提交失败**；
- 联合验证（joint）虽牺牲少量速度，但显著提升可靠性与准确率（+0.67~0.89pp）；
→ **必须以“联合一致性”作为验证单位**

#### （4）**Teacher 更新方式**
- 若动态更新教师（self-teaching），会导致 confidence drift（KL ↓ 至 4.66e-9）；
- 冻结教师维持 KL ≈ 0.00697，防止过度自信漂移；
→ **冻结教师 + KL anchor 是稳定训练的关键**

#### （5）**Decoder-Matched 对比（Table 3）**
使用相同的 Multi-Block decoder 后：
- OPTD 相比 TAD-S：
  - **Macro Acc ↑ +0.67 pts (47.30 → 47.97)**
  - **Avg TPF ↑ +4.4% (6.86 → 7.16)**
  - 所有任务 TPF 均提升（3.0–6.7%）
→ 表明性能增益来自**学习到的 transition policy 本身**，而非 decoder 差异。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Off-policy 蒸馏存在严重状态偏移**：
   - 学生访问的状态中仅 **33.7%** 出现在教师轨迹上；
   - 超过第40步后覆盖率降至 **6.5%**；
   - 即使进度匹配的动作，也只有 **50.3%** 在学生状态下仍有效。

2. **On-policy + 自适应压缩至关重要**：
   - 固定压缩步长无法充分利用不同状态下的并行潜力；
   - 自适应压缩可在不损失太多精度的前提下大幅提升 TPF。

3. **Joint Outcome Consistency 是可靠压缩的前提**：
   - 单个 token 匹配 ≠ 联合提交有效；
   - 必须验证整个 action block 是否保持教师输出一致性。

4. **OPTD 实现最优权衡**：
   - 在四个 benchmark 上取得最高 AUP（313.18）；
   - 显著优于所有 few-step baseline，尤其在高并行区域。

---

### ⚠️ 局限性
1. **依赖高质量初始化**：
   - 从 base LLaDA 直接训练的版本 AUP 仅为 138.77（远低于 313.18）；
   - 表明 TAD-S 初始化提供了大量跨 block 并行结构。

2. **未提供推理时一致性保障**：
   - Lemma 1 仅保证训练目标的一致性；
   - 推理时学生行为不受控，依赖 confidence calibration。

3. **计算开销较高**：
   - 构造目标需多次反事实 rollout（counterfactual outcome check）；
   - 不适用于极低延迟场景。

---

### 🔮 未来工作方向
1. **改进初始化策略**，降低对强 few-step checkpoint 的依赖；
2. **增强 confidence calibration**，使推理行为更接近训练分布；
3. **扩展至更大响应长度、多样化 domain 和 decoder policy**；
4. **结合系统优化**（如 KV cache、sharding）进行端到端部署评测；
5. **探索跨 backbone transferability**，推动 OPTD 成为通用 few-step 加速范式。

---

## 总结一句话
> **OPTD 通过 on-policy 状态采样 + 一致性引导的自适应压缩，在无需黄金标签的情况下实现了当前最优的 few-step dLLM 质量-效率权衡，显著提升了 AUP 与 TPF，为 diffusion language models 的高效推理提供了新范式。**

</details>

---

### 7. [SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning for LLMs](https://arxiv.org/abs/2608.03573)

**Authors**: Kejian Zhu, Zhuoran Jin, Shangqing Tu, Hongbang Yuan, Yushi Bai, Kang Liu, Juanzi Li, Jun Zhao  
**Category**: cs.CL  
**Published**: 2026-08-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.03573v1  

#### Abstract
Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) exhibit fundamentally different behaviors in enhancing multi-task reasoning for large language models (LLMs). Our preliminary experiments revealed a phenomenon: SFT suffers from severe task conflicts under multi-stage training, whereas RL ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning for LLMs 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文系统地研究了在多任务学习（multi-task learning）场景下，**Supervised Fine-Tuning (SFT)** 和 **Reinforcement Learning (RL)** 在提升大语言模型（LLMs）推理能力时的根本性差异。作者发现，尽管两者都是主流的后训练范式，但在多阶段（multi-stage）训练中表现出截然不同的行为：
- **SFT** 会引发严重的**任务冲突（task conflicts）**，导致模型在后续任务上出现灾难性遗忘和性能崩溃。
- **RL** 则能实现不同任务间的**共存（coexists）**，支持稳定且累积性的性能增长。

这一现象解释了为何现有工作普遍采用混合数据（mixed-data）进行 SFT，而 RL 却常采用多阶段训练策略。

### 提出的新方法与新思路
基于上述发现，作者提出了一个名为 **Parallel-RL** 的新型多任务学习范式，其核心思想是：
- **解耦多任务训练**：将不同任务的 RL 训练过程完全独立并行化。
- **参数合并**：在各自独立训练完成后，通过特定的合并函数（如平均、SVD 或 TIES）将各任务的参数更新（△W）融合到最终模型中。

该方法的关键洞察是：**RL 在不同任务上的优化方向近似正交（approximately orthogonal）**，因此可以安全地并行训练和合并。

### 相比现有方法的优势
- **更高的效率**：Parallel-RL 允许并行训练，显著缩短了总训练时间。
- **更强的灵活性**：支持模块化的模型能力组合，便于按需添加或移除特定任务能力。
- **更优的性能**：在保持单任务性能的同时，实现了接近甚至超越传统多阶段训练的多任务综合表现。
- **理论支撑强**：首次从梯度干扰（gradient interference）的角度，为 RL 的“共存”特性提供了理论解释。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖四个典型的推理领域，分别用于训练和评估：
- **Math**: 使用 `OpenR1-Math-220k` 子集进行 SFT，`DeepScaleR-Preview-Dataset` 子集进行 RL。
- **Science**: 使用 `AM-Thinking-v1-Distilled` 子集。
- **Code**: 使用 `AM-DeepSeek-Distilled-40M` 子集（SFT），`DeepCoder-Preview-Dataset` 子集（RL）。
- **Logic**: 使用 `knights-and-knaves` 数据集，并利用 DeepSeek-R1 API 蒸馏出 Long-CoT 数据。

**评估基准（Benchmarks）**：
- **Math**: `MATH500`, `AIME2025`
- **Science**: `MMLU`（科学相关子集）, `GPQA-Diamond`
- **Logic**: `Knights & Knaves`
- **Code**: `LiveCodeBench`

### 实验设置和评估指标
- **基础模型**：`DeepSeek-R1-Distill-Qwen-1.5B` 和 `7B`。
- **高效微调**：部分实验使用 **LoRA**（rank=64, α=32）以方便分析。
- **RL 算法**：主要采用 **GRPO**（Group Relative Policy Optimization），采样温度为 0.6，top-p 为 0.95，每提示生成 16 个 rollouts。
- **评估工具**：使用 `lighteval` 工具包，在 A100 GPU 上进行一致性评估。
- **主要指标**：**准确率（Accuracy %）**，并报告相对于基础模型和单任务模型的变化。

### 基线方法对比
论文对比了多种多任务训练范式：
- **Single-Task SFT/RL**：单一任务训练，作为性能上限参考。
- **Mixed Data SFT/RL**：将所有任务数据混合后联合训练。
- **Multi-Stage SFT/RL**：按顺序分阶段训练每个任务。
- **Naive Parallel-SFT/RL**：并行训练后简单平均或求和参数。
- **Sparse Parallel-RL**：使用 **TIES** 或 **SVD** 进行稀疏化合并。
- **Adapted Parallel-RL**：在合并后使用少量样本（5%）进行快速适应。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
根据 **Table 4** 的核心实验结果：

| 方法 | 平均性能提升 (vs Base) | 性能保留率 (vs Single-Task) |
| :--- | :--- | :--- |
| **Multi-Stage SFT** | -8.3% | — |
| **Multi-Stage RL** | +10.2% | — |
| **Naive Parallel-RL (sum)** | +6.6% | 94.2% |
| **TIES Parallel-RL** | +8.0% | 97.4% |
| **Adapted Parallel-RL** | **+10.7%** | **103.2%** |

- **SFT 冲突严重**：多阶段 SFT 导致平均性能下降 8.3%，远低于基础模型。
- **RL 支持多阶段**：多阶段 RL 实现了 10.2% 的平均增益，验证了“共存”特性。
- **Parallel-RL 高效有效**：即使是最简单的 `Naive Parallel-RL (sum)` 也能保留 94.2% 的单任务性能。
- **Adapted Parallel-RL 表现最佳**：不仅性能保留率达 103.2%（甚至超过单任务模型），平均提升也达到 10.7%，优于多阶段 RL，同时训练效率更高。

### 消融实验结果
- **任务排除实验（Ablation Study, Table 5）**：
  - 移除某个任务的更新（△W）会导致该任务性能大幅下降（平均 -7.1%）。
  - 对其他任务的影响极小，甚至有轻微提升（平均 +0.6%）。
  - **结论**：Parallel-RL 成功实现了任务能力的**解耦（decoupling）**，证明了其模块化特性。

- **探索-干扰权衡（Exploration-Interference Trade-off, Figure 5）**：
  - 增加采样温度（T）可提升单任务探索能力，但也会增大梯度方差（V），从而增加任务间干扰。
  - Parallel-RL 性能在 T≈0.65 处达到峰值，揭示了**单任务性能**与**多任务兼容性**之间的权衡。

---

## 4. 关键结论和发现

### 主要发现
1. **SFT Conflicts, RL Coexists**：这是论文最核心的发现。SFT 在多阶段训练中因密集且重叠的参数更新而产生冲突；而 RL 由于其 on-policy 特性和优势函数（advantage function）的归一化，诱导出稀疏且近似正交的更新，从而实现任务共存。
2. **梯度干扰机制不同**：
   - **SFT** 的干扰是 **norm-limited**，取决于梯度绝对大小。
   - **RL** 的干扰是 **variance-limited**，被 rollouts 内部的方差所限制。
3. **Parallel-RL 的有效性**：得益于 RL 更新的正交性，独立并行训练后再合并是可行的，且通过稀疏化和轻量适应可进一步提升性能。

### 方法的局限性
- **任务选择敏感**：并非所有任务对都适合并行训练。如果两个任务的高奖励路径激活相似的神经回路（如文中提到的 "Game" 任务与 Math/Code 有重叠），仍会产生干扰。
- **依赖 RL 的特性**：该方法的有效性建立在 RL 诱导正交更新的基础上，不直接适用于 SFT。
- **理论假设**：理论推导中的正交性是“近似”的，实际效果受模型规模、任务复杂度等因素影响。

### 未来工作方向
- **自动化任务兼容性判断**：开发更鲁棒的方法来预测哪些任务可以安全地并行训练。
- **优化探索策略**：设计新的 RL 算法或训练策略，在保证单任务性能的同时最小化梯度方差，以更好地服务于 Parallel-RL。
- **扩展至更多任务和模态**：将 Parallel-RL 应用于更大规模的任务集合和多模态场景。
- **探索 SFT 的正交化**：研究是否可以通过修改 SFT 的目标或损失函数，使其也具备类似 RL 的低干扰特性。

</details>

---

### 8. [FedRings: A Scalable and Topology-Aware Federated Learning Framework for LEO Satellite Constellations](https://arxiv.org/abs/2608.03436)

**Authors**: Ziwu Liu, In\^es Pinto Gouveia, Rehana Yasmin, Paulo Esteves-Verissimo, Ali Shoker  
**Category**: cs.DC  
**Published**: 2026-08-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.03436v1  

#### Abstract
Federated learning over low Earth orbit (LEO) satellite networks is limited by frequent link changes, short contact times, and a highly dynamic topology, making centralized or synchronized training inefficient and hard to scale. To address this, we propose FedRings, a decentralized framework that or...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FedRings: A Scalable and Topology-Aware Federated Learning Framework for LEO Satellite Constellations

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
低地球轨道（LEO）卫星星座中的 **Federated Learning (FL)** 面临以下挑战：
- **高度动态的网络拓扑**：由于卫星高速运动，通信链路频繁建立和中断。
- **短接触时间与间歇连接**：Inter-satellite links (ISLs) 存在时间窗口短暂，难以完成同步训练。
- **中心化架构不可行**：依赖地面站或参数服务器的方法存在高延迟、可扩展性差等问题。
- **通信开销大**：传统 FL 全量传输模型更新，在带宽受限环境下效率低下。

这些问题导致现有 FL 方法在 LEO 环境中收敛慢、通信成本高、鲁棒性差。

---

### 🚀 提出的新方法与创新思路

FedRings 是一种 **去中心化、拓扑感知的 FL 框架**，专为 LEO 卫星星座设计，其核心创新包括：

#### （1）**Ring-Based Topology 组织通信结构**
- 利用 LEO 星座天然的“轨道平面”结构，将每个轨道上的卫星组织成逻辑环（intra-orbit ring）。
- 支持跨轨道（inter-orbit）通信窗口进行环间交互，形成动态多环拓扑。
- 实现结构化、有序的模型协调，避免盲目广播。

#### （2）**Spatio-Temporal Routing（时空路由策略）**
- 结合预模拟的 **TimeTable**（基于 STK 仿真生成的可见性窗口）、实时链路质量（Link Metrics）和历史记录（Historical Records），构建 **Communication Opportunity Matrix (COM)**。
- 使用 **Yen’s Algorithm** 进行多路径选择，支持快速故障切换。
- 动态适应链路变化，提升通信可靠性与资源利用率。

#### （3）**Adaptive Sparse Incremental Aggregation (ASIA)**
- 在环内逐步聚合模型更新：每个卫星接收前驱聚合结果，融合本地更新后转发。
- 引入 **Top-Q sparsification** 和 **time-correlated sparsification**（全局掩码 + 局部自适应），仅传输重要梯度分量。
- 固定消息大小，显著降低通信负载。

#### （4）**Historical Compensation Mechanism（历史补偿机制）**
- 当邻居更新丢失时，利用存储的历史参数进行补偿：
  - 对高质量邻居使用其最近有效更新；
  - 对低质量邻居则回退到当前聚合状态。
- 维持训练连续性，缓解因链路中断导致的性能下降。

---

### 🔍 相比现有方法的优势

| 特性 | FedRings | 其他主流方法（如 DSFL, DFedSat, FedSN 等） |
|------|----------|---------------------------------------------|
| 架构 | 完全去中心化 | 多数仍假设理想连接或依赖地面/HAP |
| 拓扑利用 | 显式建模轨道环结构 | 忽视物理拓扑，采用泛化图模型 |
| 路由机制 | 时空感知动态路由（COM + Yen） | 静态/随机通信模式 |
| 通信效率 | ASIA 减少冗余传输 | 全量更新或简单压缩 |
| 中断处理 | 历史补偿维持聚合连续性 | 重传或丢弃 |
| 可扩展性 | 环状结构天然支持大规模部署 | 随节点增加通信负担剧增 |

> ✅ 总结优势：**更高效、更稳定、更低通信开销、更强可扩展性**

---

## 2. 核心实验方法和设置

### 📚 数据集
使用三个真实遥感图像数据集，均来自卫星观测任务：
- **EuroSAT**：Sentinel-2 卫星影像，10类地物分类，输入尺寸 64×64。
- **So2Sat LCZ42**：多光谱遥感数据，42类城市气候区分类，原始 32×32 → 插值至 224×224。
- **DeepGlobe Land Cover**：土地覆盖分类，6类，原图 512×512 → 下采样至 224×224。

所有数据集按 80%:10%:10% 划分训练/验证/测试集，并假设 **IID 分布**（符合均匀轨道布局假设）。

---

### ⚙️ 实验设置

#### 模拟环境
- 使用 **Systems Tool Kit (STK)** 构建 **Walker Star (6/90/1)** 星座模型：
  - 6个轨道面，每轨15颗卫星（共90颗）。
  - 轨道高度 HLEO = 550 km，倾角 65°。
- 生成 **TimeTable**：预测所有可能的 ISL 通信窗口。
- 模拟信号中断、包丢失等现实干扰条件。

#### 模型配置
- **模型**：DenseNet-121（约 8MB 参数量，适合边缘设备）
- **优化器**：SGD，学习率 0.1，weight decay = 0.001
- **训练轮次**：300–350 rounds

#### 评估指标
| 指标 | 说明 |
|------|------|
| **Test Accuracy** | 最终模型精度，衡量学习效果 |
| **Convergence Speed** | 达到目标精度所需训练轮数 |
| **Communication Overhead** | 总传输字节数（MB/Gb） |
| **Scalability** | 不同规模下性能稳定性（改变轨道数 t 和每轨卫星数 n） |
| **Ablation Study** | 移除 COM 或 ASIA 后的影响 |

---

### 🆚 基线方法对比
由于缺乏公开基准，作者实现并比较了两种基础去中心化 FL 方法：
- **FedAvg-D**：去中心化版本的 FedAvg，基于 gossip 协议交换模型。
- **DSGD**：Decentralized SGD，直接与邻居同步梯度。

两者均在相同网络动态条件下运行，以确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**收敛性能（Figures 6–8）**
| 方法 | EuroSAT (acc) | So2Sat (acc) | DeepGlobe (acc) |
|------|---------------|--------------|------------------|
| **FedRings** | >85% @ 50轮 | ~68% | ~80% |
| FedAvg-D | <70% @ 50轮 | ~62% | ~74% |
| DSGD | ~65% @ 50轮 | ~60% | ~72% |

> ✅ FedRings 在所有数据集上 **收敛更快、最终精度更高**，尤其在复杂非 IID 场景（So2Sat）中优势明显。

---

#### （2）**通信效率（Figure 9）**
- 在三类数据集中，FedRings 的总通信开销显著低于基线：
  - 相比 FedAvg-D 和 DSGD，**减少约 40%-60% 的通信量**。
- 尤其在 DeepGlobe 和 So2Sat（高分辨率/多通道）中节省更为突出。
- 原因：ASIA 的稀疏增量聚合大幅削减传输内容。

---

#### （3）**消融实验（Ablation Study）——移除 COM（Figure 10）**
- 移除 Communication Opportunity Matrix（COM）后：
  - 所有数据集上收敛速度变慢，初期波动加剧。
  - 最终准确率下降 3–7%，尤其在 So2Sat 和 DeepGlobe 上更严重。
- 表明：**COM 对路径规划、延迟控制和稳定性至关重要**。

> 示例：在 DeepGlobe 上，无 COM 时需额外 80+ 轮才能达到相同精度。

---

#### （4）**可扩展性测试（Figure 11）**
- 测试不同规模组合（t=6/n=15 → t=8/n=18）下的表现：
  - 随着卫星数量增加，FedRings 仍能保持稳定收敛和较高精度。
  - 虽然收敛略有放缓（因环更长），但远优于基线方法的崩溃趋势。
- 表明：**ASIA 与环状拓扑设计天然支持大规模扩展**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LEO 星座的轨道环结构可以被显式用于 FL 设计**，而非视为一般图网络。
2. **时空感知路由（spatio-temporal routing）结合 COM 可显著提升通信效率与鲁棒性**。
3. **ASIA 实现了高效的 in-network aggregation**，在不牺牲精度的前提下极大降低通信开销。
4. **历史补偿机制有效应对链路中断**，保障训练连续性和模型一致性。
5. FedRings 在真实遥感任务中 **全面超越去中心化基线方法**，是首个真正面向 LEO 拓扑特性的 FL 框架。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **安全性未考虑** | 当前框架未集成加密、防篡改或拜占庭容错机制（如区块链、QKD）。 |
| **数据异构性假设较弱** | 实验基于 IID 设置，实际中卫星观测区域差异可能导致 Non-IID 问题加剧。 |
| **全局掩码同步开销** | 虽然周期更新，但在超大规模网络中仍可能存在同步延迟。 |
| **未适配 MEO/GEO** | 目前仅针对 LEO 设计，对中高轨卫星适用性待验证。 |

---

### 🔮 未来工作方向（原文提出）
1. **拓展至 MEO/GEO 星座**：研究不同轨道层级下的联邦学习协同机制。
2. **引入 ML-based 轨道预测模型**：进一步优化 COM 的准确性与时效性。
3. **构建层次化聚合结构（Hierarchical Aggregation）**：适用于 mega-constellation 规模。
4. **增强安全机制**：
   - 集成 **Blockchain-based FL** 提供信任保障；
   - 探索 **sat-QFL** 类量子安全通信方案。
5. **支持个性化 FL（Personalized FL）**：应对 Non-IID 数据分布，例如结合 ALANINE 思路。

---

## 总结

> **FedRings 是首个将 LEO 卫星星座的物理轨道结构（multi-ring topology）深度融入 FL 框架的设计**。它通过 **Spatio-Temporal Routing + ASIA + Historical Compensation** 三位一体机制，在去中心化前提下实现了：
>
> - 更快收敛 ✅  
> - 更低通信开销 ✅  
> - 更强鲁棒性 ✅  
> - 更好可扩展性 ✅  
>
> 实验充分证明其在真实遥感场景下的优越性，为未来太空智能计算提供了坚实基础。下一步应关注 **安全增强** 与 **跨轨道协同学习**，推动 FL 在空间系统中的广泛应用。

</details>

---

### 9. [GLOBE: Trajectory-Aligned Gradient Matching with Structured SparseOptimization for Coreset Selection](https://arxiv.org/abs/2608.02690)

**Authors**: Hetian Liu, Jin Cui, Mengcheng Shi, Yanbin Hu, Xinyue Long, Boran Zhao, Pengju Pen  
**Category**: cs.LG  
**Published**: 2026-08-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.02690v1  

#### Abstract
On-device training of deep neural networks is fundamentally constrained by the computational and memory costs of large-scale datasets. Coreset selection offers a practical solution by retaining only a compact subset of real training samples. However, existing gradient-based methods commonly rely on ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GLOBE: Trajectory-Aligned Gradient Matching with Structured Sparse Optimization for Coreset Selection

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在边缘设备上进行深度神经网络（DNN）训练面临**计算资源、内存和能耗限制**，难以直接处理大规模数据集。因此，如何从原始数据中选择一个紧凑且具有代表性的子集（即 **coreset**），以保留完整数据的训练效果，成为关键挑战。

现有的基于梯度匹配的 coreset 选择方法存在以下缺陷：
- 仅依赖单个模型快照（single model snapshot）的梯度，无法捕捉训练过程中的动态演化；
- 多采用贪心策略（如 facility location 或 OMP），对强相关样本敏感，易产生局部次优解；
- 忽略样本梯度之间的高阶统计关系（如协方差结构）；
- 使用轻量级 proxy 模型时可能引入梯度偏差。

---

### 🚀 提出的新方法：GLOBE（Gradient Local-Balanced Extraction）

GLOBE 是一种**轨迹对齐的梯度匹配框架**，将 coreset 选择建模为一个**全局优化的稀疏加权问题**，其核心创新包括：

#### （1）多检查点梯度轨迹（Multi-checkpoint Gradient Trajectories）
- 不再使用单一时刻的梯度，而是收集多个训练阶段的 checkpoint，构建每个样本的 **gradient trajectory**。
- 能够同时捕获早期粗粒度决策边界和后期精细分类信号，更全面反映样本在整个训练过程中的影响。

#### （2）多阶分布匹配目标（Multi-order Distribution Matching）
- 引入两层匹配目标：
  - **一阶匹配**：对齐梯度轨迹的均值（mean），确保总体更新方向一致；
  - **二阶匹配**：对齐投影后的非中心化二阶矩（projected uncentered second-order moments），保留梯度间的相关性和各向异性结构。
- 更好地模拟完整数据集的优化动力学。

#### （3）结构化稀疏优化（Structured Sparse Optimization）
结合三种正则化机制实现稳定而紧凑的选择：
- **Group LASSO**：在语义相似的样本组（通过聚类获得）层面施加稀疏性，避免冗余选择；
- **Elastic Net**（ℓ₁ + ℓ₂ 正则）：提升对强相关样本的权重稳定性，防止 ℓ₁ 导致的任意选择；
- **非负预算约束 + 类平衡 Top-K 选择**：保证各类别覆盖，适用于低采样率场景。

#### （4）教师-代理模型对齐机制（Teacher-Proxy Alignment）
- 利用知识蒸馏初始化轻量级 proxy 模型，使其输出分布接近更强的 teacher 模型，减少梯度偏差，提高 trajectory 构建的可靠性。

---

### 🔍 相比现有方法的优势
| 方面 | GLOBE | 传统方法（如 CRAIG、GradMatch） |
|------|-------|-------------------------------|
| 梯度表示 | 动态轨迹（multi-stage） | 静态单步梯度（single snapshot） |
| 匹配目标 | 一阶 + 二阶统计量联合对齐 | 通常只对齐一阶均值 |
| 优化方式 | 全局稀疏权重优化（non-greedy） | 贪心或追踪算法（greedy/OMP） |
| 样本相关性处理 | Elastic Net 提升稳定性 | 易受相关样本干扰，选择不稳定 |
| 结构先验 | Group-level sparsity 控制冗余 | 缺乏显式结构建模 |

> ✅ 总体优势：GLOBE 实现了更准确的训练动力学逼近、更高的测试精度，尤其在**低保留率（如 10%）下表现显著优于现有方法**。

---

## 2. 核心实验方法和设置

### 📚 数据集
在六个图像分类 benchmark 上进行评估，涵盖不同规模与复杂度：

| 数据集 | 类数 | 图像尺寸 | 特点 |
|--------|------|----------|------|
| CIFAR-10 | 10 | 32×32 | 自然物体识别 |
| CINIC-10 | 10 | 32×32 | 分布偏移增强版 CIFAR |
| SVHN | 10 | 32×32 | 街道数字识别 |
| CIFAR-100 | 100 | 32×32 | 细粒度分类 |
| ImageNet-100 | 100 | 224×224 | 中等规模真实世界数据 |
| ImageNet-1K | 1,000 | 224×224 | 大规模复杂数据集 |

分为两组验证泛化能力：小尺度（CIFAR/SVHN/CINIC）与大尺度（ImageNet 系列）。

---

### ⚙️ 实验设置
- **保留比例（Retention Ratio）**：10%、20%、30%
- **评估架构（Evaluation Architectures）**：共五种，验证跨模型泛化性：
  - ResNet18、ResNet50（标准 CNN）
  - ShuffleNetV2、MobileNetV2（轻量级部署模型）
  - ViT（Vision Transformer）
- **评估指标**：下游任务的 **test accuracy (%)**
- **硬件配置**：AMD EPYC 7H12 CPU + 四块 NVIDIA RTX 6000 Ada GPU + 512GB RAM

---

### 🔁 基线方法对比（共14种）
按类别划分如下：

| 类别 | 方法 |
|------|------|
| 几何覆盖 | Herding, k-Center |
| 不确定性 | Entropy, Forgetting |
| 决策边界 | DeepFool (DF) |
| 梯度匹配 | CRAIG, GradMatch (GM), CAL, GraNd |
| 双层优化 | GLISTER, DQ |
| 近期先进方法 | NMS (Near Memory Sampling), GC |

> 所有方法均在同一条件下复现或引用原论文结果。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1）

| 数据集 | Ratio | 最佳基线 | GLOBE | 提升幅度 |
|--------|-------|---------|--------|-----------|
| CIFAR-10 | 10% | 85.33% (NMS) | **86.56%** | +1.23 pp |
| CINIC-10 | 10% | 62.88% (NMS) | **64.75%** | +1.87 pp |
| SVHN | 10% | 89.91% (DQ) | **91.40%** | +1.49 pp |
| ImageNet-100 | 10% | 58.61% (GC) | **60.69%** | +2.08 pp |
| ImageNet-100 | 20% | 65.04% (GC) | **68.97%** | **+3.93 pp** ✅最大增益 |
| ImageNet-1K | 10% | 39.57% (GC) | **50.32%** | +10.75 pp（相对提升巨大）|

> 💡 在所有 **18 个实验设置**（6 数据集 × 3 比例）中，GLOBE 均取得**最高 test accuracy**，且优势随压缩程度增加而扩大。

---

### 🔍 消融实验结果（Ablation Study，Table 2）

在 CIFAR-10（10% 保留率，ResNet-18）上的消融分析：

| 配置 | Test Accuracy (%) |
|------|-------------------|
| Full GLOBE（完整模型） | **90.21** |
| Final Checkpoint Only（仅最后梯度） | 88.53（↓1.68） |
| w/o Group LASSO | 85.32（↓4.89） |
| w/o Elastic Net | 81.21（↓9.00） |
| Random Sampling | 74.31 |

#### 发现：
- **多检查点轨迹至关重要**：仅用最终梯度导致明显性能下降；
- **Group LASSO 提升结构多样性**：去除后冗余样本增多，泛化变差；
- **Elastic Net 对稳定性极为关键**：无 ℓ₂ 正则时权重集中，难以转化为高质量离散 coreset；
- 各模块协同作用，缺一不可。

---

### 📈 超参数敏感性分析（Figure 3）
- **λ₁（ℓ₁ 正则强度）**：控制样本级稀疏性，增大 → 更多样本被剪枝；
- **λ₂（ℓ₂ 正则强度）**：平滑权重分配，过大 → 权重动态范围压缩，区分度降低；
- **λ_g（Group LASSO 强度）**：驱动整组剔除，实现语义层级稀疏；
- 可视化显示权重分布合理，未出现极端集中或孤立峰值。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **动态梯度轨迹比静态梯度更具代表性**：聚合多阶段梯度能更好刻画样本长期贡献，缓解瞬时噪声干扰。
2. **联合一阶与二阶匹配显著提升分布保真度**：不仅对齐平均梯度方向，还保留梯度间相关结构，使 coreset 更贴近原始数据的动力学行为。
3. **结构化稀疏优化带来稳定且紧凑的选择**：
   - Group LASSO 抑制语义重复；
   - Elastic Net 缓解强相关样本下的不稳定性；
   - 整体形成“先组后样”的分层选择机制。
4. **GLOBE 在极低保留率下仍保持高性能**：尤其适合边缘设备等资源受限场景。

---

### ⚠️ 局限性
1. **计算开销较高**：需多次前向/反向传播收集梯度轨迹，proxy 训练时间较长；
2. **依赖 proxy 模型质量**：尽管有 teacher alignment，proxy 与 target 架构差异过大时仍可能存在梯度失配；
3. **超参数调优需求**：λ₁, λ₂, λ_g 等需根据数据集调整，自动化程度有待提升；
4. **当前主要用于图像分类**：在 NLP、图学习等领域的扩展尚未验证。

---

### 🔮 未来工作方向
1. **加速 trajectory 收集**：设计更高效的 checkpoint 采样策略或近似估计方法；
2. **端到端可微分 coreset 学习**：探索 soft selection 与训练联合优化；
3. **跨任务与跨模态迁移**：验证 GLOBE 在检测、分割、语音等任务中的有效性；
4. **理论分析**：建立梯度轨迹匹配与泛化误差之间的理论联系；
5. **硬件协同设计**：结合边缘芯片特性（如存内计算）进一步优化部署效率。

---

## ✅ 总结
GLOBE 是一种全新的 **trajectory-aware、distribution-preserving、structured-sparse** 的 coreset 选择框架，在多个基准和架构上实现了**state-of-the-art 的性能**，特别是在**低数据保留率下优势显著**。它通过引入**多阶段梯度轨迹建模**、**多阶统计匹配目标**以及**分层稀疏正则化机制**，有效解决了传统方法在动态性、结构性和稳定性方面的不足，为高效 on-device training 提供了强有力的技术支持。

</details>

---

### 10. [PhyAI: Real-Time Physical AI at the Edge, Scalable Rollouts in the Cloud](https://arxiv.org/abs/2608.03682)

**Authors**: Chenghua Wang, Daliang Xu, Dongqi Cai, Duojin Sun, Hao Zhang, Haoze Qian, Huaiyuan Zhang, Jinshuo Cui, Kezhao Zhao, Longxi Gao, Mengwei Xu, Rongjie Yi, Tianyue Zhang, Weikai Xie, Xiyuan Tan, Xuanzhe Liu, Yingying Qin, Yiwen Lu, Yuan Yao, Yuezhi Zu, Yunhan Guo, Ziqi Guo  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.03682v1  

#### Abstract
Physical AI policies require inference throughout their lifecycle, including model evaluation, cloud reinforcement learning rollout, edge GPU serving, and onboard deployment. Although these settings share the same checkpoint and action semantics, they often rely on separate inference programs. To un...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PhyAI: Real-Time Physical AI at the Edge, Scalable Rollouts in the Cloud**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
物理AI（Physical AI）策略在生命周期中需要在多种场景下进行推理，包括模型评估、云端强化学习（RL）rollout、边缘GPU服务以及机载部署。尽管这些场景使用相同的checkpoint和动作语义，但通常依赖**独立的推理程序**，导致开发、优化和验证成本高昂，且行为一致性难以保证。

此外，不同模型架构（如VLA与WAM）和部署需求（低延迟 vs 高吞吐）对执行策略提出了多样化要求，现有系统无法提供统一高效的推理路径。

---

### **提出的新方法或新思路**

作者构建了 **PhyAI** —— 一个面向物理AI的**统一推理引擎（unified inference engine）**，其核心设计思想是：

- **分离模型逻辑与执行服务**：
  - **Model Adapters** 负责保留架构特定的条件处理、求解器状态、缓存有效性、调度和动作转换等语义。
  - **Runtime Core** 提供图重放（graph replay）、优化内核（kernels）、内存管理、量化路径、通信和并行执行等共享服务。

这种设计实现了“一次实现，多端运行”（onboard, edge, cloud），支持从Jetson Thor到H100/A100服务器的跨设备部署。

- **引入 Control-Time Roofline 分析模型**
  - 区分控制循环中的 **inference-bound** 和 **environment-bound** 状态。
  - 定义理想重叠下的控制周期为 `L_overlap = max(L_inference, L_env)`。
  - 当推理快于环境执行时，进一步加速不会提升控制频率，而是产生时间裕量（timing margin），可用于容错或多任务复用。

---

### **相比现有方法的优势**

| 方面 | PhyAI优势 |
|------|----------|
| **统一性** | 支持 VLA 和 WAM 多种模型家族，在同一代码库下运行，避免重复开发。 |
| **效率** | 相比官方实现平均提速 **1.40× ~ 4.65×**，尤其在大batch和分布式场景表现优异。 |
| **灵活性** | 支持 DP、TP、CFG 并行策略组合，适配不同模型结构（如Cosmos3使用TP+CFG）。 |
| **可扩展性** | Adapter接口允许快速集成新模型（如MiniCPM-Robot发布当天即接入）。 |
| **分析能力** | 提出 Control-Time Roofline 工具，指导何时应优化推理 vs 接受其为非瓶颈。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与模型**
- **任务套件**：LIBERO系列（libero_spatial, libero_object, libero_goal, libero_10）
- **评估模型**：
  - Vision-Language-Action (VLA): `o`, `0.5`, `GR00T N1.7`, `MiniCPM-Robot`
  - World-Action Model (WAM): `Cosmos3-Nano-Policy-DROID`, `MiniCPM-Track`

### **实验设置**
- **硬件平台**：
  - 边缘设备：Jetson Thor, Orin NX
  - GPU服务器：RTX 5090, A40, H100, H20×8
- **精度配置**：主要使用 BF16；部分测试含 FP8 对比
- **部署模式**：
  - 单请求延迟（single-request latency）
  - 静态批处理（static batching, batch size up to 32）
  - 分布式 rollout 模拟（8×A100, batch=40）

### **评估指标**
| 指标 | 描述 |
|------|------|
| `Latency` | 单个action chunk生成时间（不包含传输、排队等） |
| `Throughput (Q(B))` | 批处理吞吐（samples/sec） |
| `Amortized Latency` | 批处理总耗时 / batch size |
| `Speedup` | 相对于官方实现的加速比 |
| `MFU (Model FLOPs Utilization)` | 实际算力利用率 |
| `Success Rate` | 在LIBERO任务上的任务完成率 |

### **基线方法对比**
- **官方实现**：各模型原生PyTorch路径（如OpenBMB/MiniCPM-Robot）
- **专用推理框架**：
  - `FlashRT`
  - `vla.cpp`
  - `realtime-vla`
  - `LeRobot`（异步栈，非kernel级统一）

> 注意：PhyAI以**官方路径为基准**进行比较，而非宣称在所有配置下都超越最优化的专用runtime。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **单请求延迟显著降低**
| 模型 | 设备 | 官方延迟 | PhyAI延迟 | 加速比 |
|-------|--------|-----------|------------|---------|
| `o` | RTX 5090 | 127.48 ms | **31.475 ms** | **4.05×** |
| `0.5` | RTX 5090 | 52.076 ms | **28.613 ms** | **1.82×** |
| `GR00T N1.7` | A40 | 199.703 ms | **46.348 ms** | **4.31×** |
| `MiniCPM-Robot` | H100 | 105.380 ms | **22.640 ms** | **4.65×** |
| `Cosmos3-Nano` | H20×8 (CFG=2, TP=4) | 2.46 s | **1.18 s** | **2.08×** |

> 所有11组对比均优于官方路径，**最高达4.65×加速**。

---

#### ✅ **批处理吞吐分析揭示不同模型瓶颈**
| 模型 | Batch Size | 吞吐变化趋势 | 关键发现 |
|------|------------|----------------|-----------|
| **PI0.5** | 1 → 32 | 44.26 → 100.02 samples/s | 小batch时action expert占主导（57.2%延迟），随batch增大转向vision-language主干 |
| **GR00T** | 1 → 32 | 48.8 → 138.8 samples/s | batch=8已达92.2%峰值吞吐，更大batch仅增加等待时间 |
| **Cosmos3** | 1 → 16 | 0.883 → 1.010 chunks/s (+14.3%) | generation阶段始终compute-bound，batch收益极小 |

> 表明：**不能用单一策略优化所有模型**，需结合phase profiling定制方案。

---

#### ✅ **消融实验与优化技术贡献**
- **Kernel Fusion**：
  - 自定义 AdaRMSNorm、GeGLU/SwiGLU融合激活函数。
  - 在H20上使MiniCPM-Robot吞吐从33.28 Hz提升至36.77 Hz（+10.5%）。
- **Graph Replay**：
  - 利用CUDA Graph捕获稳定控制流，减少kernel launch开销。
- **Operator Selection**：
  - 动态选择最优Attention后端（如FlashInfer FA2 > FA3）。
- **Quantization**：
  - 支持 W4A16/W8A8 等量化格式，但本报告结果基于BF16，未计入量化增益。

---

#### ✅ **模拟云RL Rollout 性能提升**
- 设置：8×A100, batch=40, 41次inference调用/step
- 结果：
  - 推理时间占比从 **53.1% ↓ 36.2%**
  - rollout step延迟预计下降 **26.5%**
  - 训练吞吐提升约 **1.36×**

> 显示PhyAI在大规模RL训练中有显著端到端收益潜力。

---

#### ✅ **任务成功率保持一致甚至略优**
| 模型 | 基线成功率 | PhyAI成功率 |
|------|-------------|--------------|
| GR00T-N1.7 (LIBERO-10) | 91.2% | **91.8%** |
| o.5 (LIBERO四套件) | - | **97.45% aggregate** |
| o (LIBERO两套件) | - | **71.5%** |

> 表明加速未牺牲行为一致性，反而可能因更稳定执行而略有提升。

---

## **4. 关键结论和发现**

### **主要发现**

1. 🔍 **统一推理路径可行且高效**  
   PhyAI通过adapter隔离语义、runtime共享基础设施，成功支持VLA/WAM多类模型，实现跨边云一致部署。

2. ⚖️ **Control-Time Roofline 是实用分析工具**  
   - `o.5` 在LIBERO中已处于 **environment-bound**，说明继续优化推理只能换取裕量；
   - `Cosmos3` 仍为 **inference-bound**，仍有加速空间。

3. 📊 **不同模型具有截然不同的性能瓶颈**  
   - `o.5`：小batch下action expert kernel launch overhead严重；
   - `GR00T`：backbone随batch上升成为瓶颈；
   - `Cosmos3`：generation阶段高度compute-bound，batch scaling无效。

4. 💡 **加速≠必须最快，关键是“够用+灵活”**  
   PhyAI目标不是在每个配置下做到绝对最快，而是提供**接近最优但通用性强**的解决方案。

---

### **方法的局限性**

| 局限 | 说明 |
|------|------|
| ❌ 未完全覆盖尾延迟（tail latency） | 实验聚焦平均延迟，未深入p99分析 |
| ❌ 不支持动态批处理（continuous batching） | 当前为静态批处理，限制高并发场景效率 |
| ❌ 缺少对ARM平台深度优化内核 | 如Thor上多数算子仍回退至PyTorch默认实现 |
| ❌ 未集成完整闭环机器人安全验证流程 | 成功率测试依赖仿真，真实世界泛化待验证 |

---

### **未来工作方向**

1. **IMega Kernels**  
   将反复执行的小算子编译为持久化mega-kernel，利用MPK或Event Tensor降低launch开销。

2. **Kernel Agents for Thor Optimization**  
   使用AutoKernel风格代理自动识别热点算子，并生成高性能Triton/CUDA内核。

3. **PhyAI作为RLinf Rollout Backend**  
   正式集成进RLinf框架，实测端到端训练加速效果。

4. **扩展模型与硬件支持**  
   - 新增支持：Xiaomi-Robotics-1, LingBot-VA 2.0, HY-Embodied-0.5等
   - 新增硬件：Ascend 910/950, Kunlunxin P800, AMD Instinct MI355X

5. **构建生产级MaaS协议**  
   设计支持delta更新、key帧、deadline协商的状态化流式推理协议，适用于工厂级Robot MaaS部署。

---

> 🌐 **最终愿景**：  
> PhyAI致力于成为物理AI时代的**基础设施层**，让算法研究者无需再为部署碎片化所困，真正实现“训练即服务，推理无处不在”。  
> 项目开源地址：[github.com/mingti-org/phyai](https://github.com/mingti-org/phyai)

</details>

---

### 11. [CastFSR: A Fast--Slow--Reflect Agentic Reasoning Framework for Context-Aware Time Series Forecasting](https://arxiv.org/abs/2608.03031)

**Authors**: Xiaoyu Tao, Mingyue Cheng, Bokai Pan, Chuang Jiang, Huanjian Zhang, Tian Gao, Yaguo Liu, Qi Liu, Enhong Chen  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.03031v1  

#### Abstract
Time series forecasting is fundamental to decision-making in complex systems, where future dynamics are influenced not only by historical observations but also by evolving contextual features. Recent advances in large language models (LLMs) have extended forecasting beyond numerical extrapolation to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# CastFSR: A Fast-Slow-Reflect Agentic Reasoning Framework for Context-Aware Time Series Forecasting —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统时间序列预测（TSF）方法主要依赖于**历史观测值的数值模式外推**，难以有效融合复杂的上下文特征（如天气、节假日、政策事件等）。尽管大型语言模型（LLMs）被引入用于语义推理，但现有方法仍存在以下问题：
- 缺乏机制来**识别哪些上下文真正影响未来动态**；
- 无法系统地**推理上下文如何改变趋势**；
- 预测结果可能违反**时间一致性、领域约束或物理合理性**。

### 🚀 提出的新方法：CastFSR
作者提出 **CastFSR**，一个基于“**Fast-Slow-Reflect**”范式的**智能体式推理框架**，将上下文感知的时间序列预测建模为一个**顺序决策过程**，包含三个阶段：

| 阶段 | 功能 |
|------|------|
| **Fast-thinking Forecasting** | 快速构建基于历史模式的数据驱动预测先验（forecast prior），通过自动选择轻量级预测器（lightweight forecasters）实现。 |
| **Slow Deliberative Reasoning** | 检索长程上下文历史中的相关证据，自适应识别特定上下文的回看窗口（look-back window），并推理其对未来动态的影响以修正先验。 |
| **Reflective Evaluation** | 迭代验证候选预测是否满足时间规律性、上下文支持性和领域约束，并进行局部修正而非全局重生成。 |

该框架强调：
- **LLM 不直接生成数值**，而是作为协调者（coordinator）调用工具、整合异构信息、引导推理流程；
- 支持**无需训练的部署**（off-the-shelf LLMs）和**紧凑模型蒸馏**（via SFT + RL）两种模式。

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **架构设计** | 明确分离“快速预测”、“慢速推理”、“反思验证”，避免盲目依赖 LLM 数值生成。 |
| **上下文利用** | 主动检索与任务相关的上下文证据，而非被动接收预定义输入。 |
| **可解释性** | 输出不仅包含预测值，还包括使用的证据、修订理由和一致性检查报告。 |
| **灵活性与通用性** | 可适配多种 LLM 协调器（如 GPT-5.6, DeepSeek-V4），也可通过两阶段训练内化到小型模型中。 |
| **鲁棒性增强** | 通过反射机制确保预测符合非负性、容量限制、时间连续性等现实约束。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
实验覆盖多个真实世界时间序列基准，分为两类：

#### **长期预测（Long-term Forecasting）**
| 数据集 | 域 | 长度 | 变量数 | 频率 | 描述 |
|-------|-----|------|--------|--------|------|
| ETTh1 / ETTh2 | 电力 | 17,420 | 7 | 1小时 | 变压器油温（OT）及6个负载特征 |
| ETTm1 / ETTm2 | 电力 | 69,680 | 7 | 15分钟 | 更高分辨率的变压器温度数据 |
| Wind | 能源 | 48,673 | 7 | 15分钟 | 风电场功率 + 气象预报（风速、温度等） |

#### **短期电价预测（Short-term EPF）**
| 数据集 | 区域 | 频率 | 特征 | 任务 |
|-------|------|--------|--------|--------|
| BE, DE, FR, NP, PJM | 欧洲/北美市场 | 1小时 | 电价 + 负载/发电预测 | 预测未来24小时价格 |

> 所有任务采用统一设定：长时任务使用 `look-back=96`, `horizon=96`；短时任务使用 `168/24`。

---

### 📈 评估指标
- **MSE**（均方误差）
- **MAE**（平均绝对误差）

越低越好。

---

### ⚔️ 对比的基线方法
| 类别 | 方法列表 |
|------|---------|
| **统计模型** | ARIMA, Prophet |
| **深度学习模型** | DLinear, ConvTimeNet, PatchTST, iTransformer, TimeXer |
| **基础模型（Foundation Models）** | TimesFM, Sundial |
| **LLM-based 方法** | OFA, Time-LLM, TokenCast, S²IP-LLM, PromptCast, TimeReasoner |
| **智能体系统** | TimeSeriesScientist, AlphaCast |

---

### 💡 实现细节
- **CastFSR-Zero**：使用 **DeepSeek V4 Flash** 作为 LLM 协调器，无需微调，直接执行三阶段流程。
- **CastFSR-R1**：在 **Qwen3-4B** 上进行两阶段训练：
  1. **Supervised Fine-Tuning (SFT)**：从 CastFSR-Zero 蒸馏高质量轨迹，学习基本流程；
  2. **Multi-turn Reinforcement Learning (RL)**：使用 GRPO 算法优化端到端预测质量，奖励包括准确性、趋势一致性、变化点对齐等。
- 工具池包含：ARIMA, DLinear, PatchTST, iTransformer, Chronos-2 等预测器。

---

## 3. 主要实验结果和性能指标

### 📊 总体性能对比（Table 1）
CastFSR 在绝大多数数据集上达到**最优或次优性能**，显著优于各类基线：

| 方法 | ETTh1 (MSE↓) | ETTm1 (MSE↓) | Wind (MSE↓) | DE (MAE↓) | NP (MAE↓) |
|------|--------------|-------------|------------|-----------|-----------|
| ARIMA | 0.110 | 0.055 | 1.361 | 0.429 | 0.373 |
| PatchTST | 0.102 | 0.058 | 1.478 | 0.393 | 0.389 |
| Sundial | 0.100 | 0.058 | 1.315 | 0.389 | 0.389 |
| TimeSeriesScientist | 0.123 | 0.085 | 1.250 | 0.422 | 0.606 |
| **CastFSR-Zero** | **0.081** | **0.055** | **1.596** | **0.393** | **0.221** |
| **CastFSR-R1** | **0.077** | **0.055** | **1.757** | **0.386** | **0.172** |

> ✅ **CastFSR-R1 全面领先**，尤其在复杂场景（如 NP）表现突出。

---

### 🔪 消融实验（Ablation Study, Table 3）
移除任一模块均导致性能下降，验证各阶段互补性：

| 变体 | ETTh1 (MSE) | ETTm1 (MSE) | DE (MAE) | NP (MAE) |
|------|------------|------------|----------|----------|
| w/o Fast-thinking | 0.089 | 0.061 | 0.481 | 0.334 |
| w/o Slow Reasoning | 0.086 | 0.055 | 0.421 | 0.284 |
| w/o Reflective Eval | 0.086 | 0.055 | 0.435 | 0.237 |
| **CastFSR-Zero** | **0.081** | **0.055** | **0.393** | **0.221** |

- 移除 **Fast-thinking** 影响最大 → 表明**数据驱动先验是推理的基础**；
- 移除 **Slow Reasoning** 导致上下文利用不足；
- 移除 **Reflective** 引发非法输出（如负风电功率）。

---

### 🔄 训练策略有效性（Table 6 & 9）
| 变体 | ETTh1 (MSE) | DE (MAE) | NP (MAE) |
|------|------------|----------|----------|
| w/o SFT | 0.081 | 0.435 | 0.237 |
| w/o RL | 0.082 | 0.392 | 0.217 |
| **CastFSR-R1** | **0.077** | **0.386** | **0.172** |

- **SFT 是必要初始化**，教会模型遵循三阶段逻辑；
- **RL 进一步提升决策能力**，特别是在上下文选择与调整幅度控制方面。

---

### 🌐 不同 LLM 协调器的表现（Table 5 & 8）
| LLM Coordinator | ETTh1 (MSE) | ETTm1 (MSE) | DE (MAE) | NP (MAE) |
|------------------|-------------|-------------|----------|----------|
| GPT-5.6-sol | 0.081 | 0.055 | 0.400 | 0.218 |
| DeepSeek V4 Flash | 0.081 | 0.055 | 0.393 | 0.221 |
| GLM-5.2 | **0.080** | **0.054** | 0.405 | **0.207** |

- 性能差异较小，表明 **CastFSR 的有效性不依赖单一 LLM**；
- 不同 LLM 各有优势，体现框架的**模型无关性（model-agnostic）**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Fast-Slow-Reflect 架构有效提升了上下文感知预测的准确性和可靠性**：
   - 分阶段处理使系统既能快速响应稳定模式，又能深入分析突变情境。
2. **LLM 应作为“推理控制器”而非“数值生成器”**：
   - 利用其调度能力整合专家模型、上下文检索与知识验证，比直接生成更可靠。
3. **上下文不是越多越好，关键是“适配性检索”与“影响推理”**：
   - 自适应选择 look-back window 和判断影响方向至关重要。
4. **反射机制显著提高预测的物理合理性和稳定性**：
   - 尤其在防止非法值（如负功率）方面作用明显。
5. **两阶段训练（SFT + RL）可成功将复杂推理流程内化至紧凑模型**：
   - 实现高效部署的同时保持高性能。

---

### ⚠️ 局限性
1. **依赖高质量上下文信号**：
   - 若外部变量不可靠或缺失，Slow Reasoning 效果受限。
2. **推理延迟较高（尤其 CastFSR-Zero）**：
   - 多轮交互与工具调用带来额外开销，不适合超实时场景。
3. **Look-back Window 选择仍有改进空间**：
   - 当前策略在部分案例中选择了误导性历史片段（见 Figure 7 下方示例）。
4. **对 LLM 工具理解能力敏感**：
   - 若 LLM 误解诊断工具输出或错误调用预测器，可能导致连锁错误。

---

### 🔮 未来工作方向
1. **动态扩展工具集**：
   - 引入更多领域专用模型（如气象模拟器、经济模型）增强推理能力。
2. **多模态上下文融合**：
   - 接入文本新闻、图像卫星云图等非结构化信息。
3. **在线学习与持续适应**：
   - 在部署过程中不断更新策略以应对概念漂移。
4. **降低推理成本**：
   - 设计更高效的提示模板或轻量化代理架构。
5. **开放平台建设**：
   - 构建标准化的 **Agentic TSF Benchmark**，推动社区发展。

---

## 总结一句话
> **CastFSR 成功将 LLM 的推理能力与经典预测模型的优势结合，提出了一个可解释、可验证、可部署的上下文感知时间序列预测新范式——Fast-Slow-Reflect 智能体框架，在多个真实场景中实现了 SOTA 性能。**

</details>

---

### 12. [DocTrace: Towards Traceable Long Document VQA via Hierarchical Evidence Graph Reasoning](https://arxiv.org/abs/2608.03292)

**Authors**: Le Xiang, Zhicheng Guan, Hong Chen, Xiaocong Lin, Zhenghua Lei, Teng Hu, Bolei He, Long Zeng  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.03292v1  

#### Abstract
Long Document Visual Question Answering (LongDocVQA) requires Multimodal Large Language Models (MLLMs) to locate, integrate, and reason over heterogeneous document elements distributed across multiple pages. Existing approaches, including end-to-end MLLMs, retrieval-augmented generation (RAG) pipeli...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DocTrace: Towards Traceable Long Document VQA via Hierarchical Evidence Graph Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本文针对 **Long Document Visual Question Answering (LongDocVQA)** 中的关键挑战——**多页、异构文档中的多跳推理（multi-hop reasoning）** 和 **答案可追溯性缺失**的问题。

现有方法（如端到端 MLLMs、RAG、Agent 框架）存在以下不足：
- **证据组合过程隐式化**：无法明确展示模型如何从分散在多个页面的文本、表格、图表等元素中逐步推导出答案。
- **缺乏透明性和可验证性**：预测被视为“黑箱”，难以审计或信任，尤其在金融、医疗等高风险场景下不可接受。

---

### ✅ 提出了什么新方法或新思路

作者提出 **DocTrace**，一种**分层显式证据图推理框架**，将 LongDocVQA 视为一个**显式的证据图构建与推理任务**，而非直接的答案生成。

#### 核心思想：Hierarchical Coarse-to-Fine 推理流程
1. **Stage 1: Evidence Localization（证据定位）**
   - 在低分辨率文档图像上进行粗粒度检索，识别与问题相关的候选页（`Pevid`），缩小推理空间。
2. **Stage 2: Structured Document Parsing（结构化解析）**
   - 对选中的页面使用 PaddleOCR-VL-1.5 进行高分辨率解析，提取带有语义类型（text/table/figure）、边界框和内容的布局块（layout elements），形成结构化的证据池 `B`。
3. **Stage 3: Evidence Graph Reasoning（证据图推理）**
   - 构建显式的有向无环图（DAG）`G = (V, E)`：
     - 节点 `V`：原始证据块 或 中间推理结果（derived node）
     - 边 `E`：表示推理依赖关系
   - 最终答案由该图生成，并附带完整的节点级溯源路径。

> 🌟 创新点：首次将 **证据图（evidence graph）** 作为显式推理对象，实现**可追溯、可验证的多跳推理**。

---

### ✅ 相比现有方法的优势

| 维度 | DocTrace | 现有方法（E2E/ RAG / Agent） |
|------|---------|-----------------------------|
| **可解释性** | ✅ 显式输出证据链，支持逐节点溯源 | ❌ 隐式推理，轨迹不透明 |
| **准确性** | ✅ 多跳推理更精准，减少幻觉 | ⚠️ 容易遗漏跨页依赖或产生错误连接 |
| **效率** | ✅ 分阶段处理，仅对关键页进行高分辨率推理 | ❌ E2E 模型需处理全部页面，计算开销大 |
| **训练监督** | ✅ 引入结构化监督信号（证据页 + 图） | ❌ 通常只有最终答案标签 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 特点 |
|-------|------|
| **MMLongBench-Doc** | 超长文档基准（平均 47.5 页），强调全文档理解，33.7% 为跨页问题，20.9% 不可回答 |
| **LongDocURL** | 最长可达 150 页，52.9% 多页问答，固定 30 页窗口输入 |
| **SlideVQA** | 幻灯片文档，49.3% 多跳/数值类问题，非线性布局为主 |

> 所有数据均来自公开 LongDocVQA 基准 + 自建 arXiv 技术报告（最长达 120 页）

---

### 🔬 实验设置与评估指标

| 设置项 | 描述 |
|--------|------|
| **Backbone** | Qwen3-VL-8B-Instruct |
| **训练方式** | 两阶段训练：<br>1. **联合 Supervised Fine-Tuning (SFT)**<br>2. **任务特定 Group Relative Policy Optimization (GRPO)** |
| **推理机制** | 分三阶段执行：<br>- Stage 1: 512px 低分辨率扫描<br>- Stage 2: OCR 解析选定页<br>- Stage 3: 1568px 高分辨率图推理 |
| **上下文长度** | 训练 32K，推理 128K |
| **硬件** | 16 × NVIDIA A800 GPU |

#### 评估指标
| 数据集 | 主要指标 |
|--------|----------|
| MMLongBench-Doc, LongDocURL | **Accuracy** |
| SlideVQA | **F1 Score** |
| 细粒度分析 | Page F1, GT Page Coverage, 单页/多页/不可答分类准确率 |

---

### 🆚 基线方法对比

涵盖三大类主流范式：

| 类别 | 代表方法 |
|------|----------|
| **End-to-End (E2E)** | mPLUG-DocOwl2, InternVL3, DocSeeker |
| **Retrieval-Augmented (RAG)** | VisRAG, SV-RAG, VDocRAG, MoLoRAG, URaG |
| **Agent-based** | VRAG-RL, Doc-V*, MM-Doc-R1 |
| **闭源模型** | GPT-4o, GPT-4.1, Claude-3.7-Sonnet, Gemini-1.5-Pro |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）

| 方法 | MMLongBench-Doc (Acc) | LongDocURL (Acc) | SlideVQA (F1) |
|------|------------------------|------------------|---------------|
| Qwen3-VL-8B-Instruct (Baseline) | 38.5 | 45.1 | 73.4 |
| **DocTrace (SFT)** | **50.3** (+11.8) | **53.2** (+8.1) | **83.8** (+10.4) |
| **DocTrace (GRPO)** | **52.9** (+14.4) | **56.4** (+11.3) | **85.1** (+11.7) |

> ✅ **全面超越所有开源和闭源模型**：
- 超越最强开源模型 MM-Doc-R1（49.7 → 52.9）
- 超越 GPT-4.1（45.6）和 Claude-3.7-Sonnet（76.3 → 85.1）

---

### 🔍 消融实验结果（Ablation Study）

#### （1）训练策略有效性（Table 1 & Table 2）

| 阶段 | 效果 |
|------|------|
| **Joint SFT** | 显著提升基础性能（+8~12 pts） |
| **GRPO (Stage 1)** | 改进证据定位（Page F1 70.8 → 71.3） |
| **GRPO (Stage 3)** | 提升复杂推理能力，尤其改善不可答问题判断（66.0 → 70.5） |

> 💡 结论：Stage 1 强化**证据获取**，Stage 3 优化**证据利用与可靠性**

---

#### （2）组件消融（Table 4）

| 设置 | Acc | Multi-page Acc ↓ |
|------|-----|------------------|
| DocTrace (SFT) | 50.3 | 37.7 |
| w/o Structural Parsing | 46.6 | 30.6 (-7.1) |
| w/o Graph Reasoning | 47.2 | 30.2 (-7.5) |
| w/ Vanilla CoT | 47.8 | 34.8 (-2.9) |

> ❗ 发现：
- 移除 **Graph Reasoning** 导致多页性能大幅下降，说明其对跨页整合至关重要。
- 使用普通 Chain-of-Thought 替代图推理效果差，表明**线性推理不足以建模复杂依赖**。

---

#### （3）Set-of-Marks (SoM) 消融（Table 9）

| 设置 | Acc | Multi-page Acc ↓ |
|------|-----|------------------|
| With SoM | 50.3 | 37.7 |
| Without SoM | 48.1 | 32.5 (-5.2) |

> ✅ SoM 提供视觉锚点（block_id + bbox 渲染），显著增强细粒度定位能力，尤其利于跨页推理。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **显式证据图推理能有效提升 LongDocVQA 性能与可信度**
   - 将推理过程显式建模为 DAG，使每一步都可追溯至文档源块。
   - 实现了 **node-level evidence provenance**，满足高风险应用需求。

2. **分层 coarse-to-fine 设计兼顾效率与精度**
   - Stage 1 快速筛选相关页，避免全文档高分辨率处理。
   - Stage 3 只对关键页进行精细推理，节省 token 开销（prefill 减少 2.3×）。

3. **两阶段训练范式（SFT + GRPO）有效**
   - SFT 初始化结构化能力；
   - GRPO 通过定制奖励函数进一步优化定位与图忠实度。

4. **当前瓶颈在于 Stage 1 的召回能力**
   - 多页问题性能受限主因是**未能完整检索所有证据页**（见 Table 12）。
   - 若能获取全部黄金证据页，Stage 3 推理成功率高达 ~60%。

---

### ⚠️ 局限性

1. **依赖高质量 OCR 输出**
   - 当前 Stage 2 使用外部 OCR 工具（PaddleOCR-VL），若 OCR 错误会传导至后续推理。

2. **Stage 1 的误召导致幻觉**
   - 若错误地将不可答问题判定为可答，并返回无关页，则 Stage 3 更可能生成幻觉答案（见 Table 13）。

3. **人工验证成本高**
   - 训练数据依赖 Gemini 3.1 Pro + GPT-5.5 自动生成并验证，虽保证质量但扩展性受限。

---

### 🔮 未来工作方向

1. **端到端联合优化 OCR 与推理模块**
   - 减少对外部工具依赖，提升鲁棒性。

2. **改进 Stage 1 的召回率与拒答能力**
   - 引入更强的预训练检索器或迭代召回机制。

3. **动态图构建机制**
   - 当前图为静态生成，未来可探索基于反馈的动态修正。

4. **推广至其他多模态推理任务**
   - 如法律文书审查、科研论文问答、财报分析等需要强可追溯性的领域。

---

> 🏁 **总结一句话**：  
> **DocTrace 通过“分层定位 + 结构化解析 + 显式证据图推理”的设计，在大幅提升 LongDocVQA 准确性的同时，实现了前所未有的推理透明性与可验证性，为可信多模态推理提供了新范式。**

</details>

---

### 13. [LatentGuard: Efficient and Inspectable Latent Reasoning for LLM Safeguards](https://arxiv.org/abs/2608.03838)

**Authors**: Zhinan Liu, Jie Li, Mingyu Kang, Jiayi Ji  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.03838v1  

#### Abstract
Reasoning-based guard models improve LLM safeguards, but decoding explicit rationales for every interaction makes them costly to deploy. Although latent-reasoning methods reduce token generation by moving reasoning into continuous states, they remain underexplored for safety moderation and lack an i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LatentGuard: Efficient and Inspectable Latent Reasoning for LLM Safeguards**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **LLM 安全守卫模型**（LLM safeguards）面临效率、准确性和可检查性之间的权衡：
- **分类式守卫模型**（classification-based guards）高效但缺乏决策透明度，难以审查具体判断依据。
- **基于推理的守卫模型**（reasoning-based guards），如 GuardReasoner，通过生成自然语言的 **chain-of-thought（CoT）** 推理过程提升决策质量和可解释性，但每次交互都需要生成数百个 token 的推理文本，带来显著的 **延迟和计算开销**，不适合高吞吐、低延迟的部署场景。

此外，现有 **latent reasoning** 方法虽能将推理压缩到连续隐状态中以减少 token 生成，但主要用于数学/逻辑任务，且缺乏用于部署时审查的接口。

### **提出了什么新方法或新思路**
本文提出 **LatentGuard**，一种高效且可检查的 LLM 守卫框架，其核心思想是：  
**解耦常规安全预测与理由生成**（decouple safety prediction from rationale generation）。

具体创新包括：
1. **任务对齐的潜在推理压缩**（Task-aligned latent rationale compression）：
   - 采用 **分阶段课程学习**（staged curriculum learning），逐步将显式的文本推理（textual rationales）压缩为连续的 **潜在状态**（latent states）。
   - 在标准推理路径上，模型直接从潜在状态预测安全标签，无需生成完整 CoT。

2. **独立的按需审计解码器**（Isolated on-demand audit decoder）：
   - 引入一个与主守卫模型分离的辅助解码器（auxiliary decoder）。
   - 仅在需要人工审查时（audit mode），该解码器才被调用，将潜在状态转换为紧凑的 **审计证据**（audit artifacts）。
   - 该解码器不参与主推理路径，不影响效率。

### **相比现有方法的优势**
- **高效性**：标准推理路径几乎不产生推理 token，大幅降低延迟。
- **可检查性**：保留了按需生成人类可读审计证据的能力。
- **高性能**：在保持甚至提升安全判断准确率的同时，实现了极高的推理效率。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：`GuardReasonerTrain`，由多个公开安全数据集合成，包括：
  - `WildGuardTrain`
  - `AegisTrain`
  - `BeaverTailsTrain`
  - `ToxicChatTrain`
- 包含 127,544 个样本，每个样本包含用户请求、助手响应及对应的链式思维推理轨迹和三个安全标签。

### **实验设置和评估指标**

#### **任务定义**
模型需预测三个耦合的安全标签：
1. **Request harmfulness**（请求有害性）
2. **Refusal detection**（拒绝检测）
3. **Response harmfulness**（响应有害性）

#### **评估维度**
1. **Moderation Effectiveness**（安全性判断效果）：
   - 指标：**加权 F1 分数**（weighted F1），报告各任务及三任务平均值。
2. **Critical-path Efficiency**（关键路径效率）：
   - 指标：
     - **推理 token 数量**（reasoning tokens）
     - **单样本延迟**（latency in seconds）
3. **Audit Utility**（审计效用）：
   - 指标：
     - **Verify F1**：使用 `Qwen3.5-27B` 作为 judge，判断审计证据是否足以恢复正确标签。
     - **Accept. Rate**：judge 判断审计证据是否“可用”于审查。
     - **AUS**（Audit Utility Score）：Verify F1 和 Accept. Rate 的调和平均。

#### **基线方法对比**
- **Reasoning-based**：
  - `GuardReasoner-1B/3B/8B`
- **Classification-based**：
  - `LlamaGuard3-8B`, `LlamaGuard4-12B`
  - `RobloxGuard-1.0`, `WildGuard`

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **安全性判断效果（Table 1）**
| Model | Request F1 | Refusal F1 | Response F1 | **Mean F1** |
|-------|------------|------------|-------------|-------------|
| GuardReasoner-8B | 80.98 | 89.90 | 80.98 | **83.95** |
| **LatentGuard-8B** | **82.82** | **90.30** | **81.60** | **84.91** |

- **LatentGuard-8B 将平均加权 F1 从 83.95 提升至 84.91**，尤其在请求有害性检测上提升显著（+1.84）。

#### **效率与延迟（Table 2）**
| Model | Latency (s) | Reasoning Tokens |
|-------|-------------|------------------|
| GuardReasoner-8B | 0.792 | 268.56 |
| **LatentGuard-8B** | **0.089** | **1.60** |

- **延迟降低 8.9 倍**（0.792s → 0.089s）
- **推理 token 减少 168 倍**（268.56 → 1.60）
- 实现了接近分类模型的效率，同时保持推理模型的准确性。

#### **审计效用（Table 3）**
| Model | Verify F1 | Accept. Rate | **AUS** |
|-------|-----------|-------------|--------|
| LatentGuard-8B (full) | 83.86 | 87.73 | **85.75** |

- 审计解码器生成的证据具有较高的 **实用性和可接受性**，支持事后审查。

### **与基线方法的对比结果**
- **相比 GuardReasoner**：在所有模型规模下均取得更高的 F1 和显著更低的延迟。
- **相比 LlamaGuard 等分类模型**：在拒绝检测等任务上表现更优，且提供可审查的审计路径。
- **效率-性能权衡图**（Figure 3）显示 LatentGuard 显著优于其他方法，位于“高 F1、低推理成本”的理想区域。

### **消融实验结果（Table 5）**
| Variant | Mean F1 Δ (8B) |
|--------|----------------|
| Full model | — |
| w/o curriculum | -1.05 |
| w/o latent | -1.41 |

- **移除分阶段课程学习**（w/o curriculum）导致性能下降，说明渐进式压缩训练更稳定。
- **完全不使用潜在推理**（w/o latent）性能下降更大，证明潜在推理本身是性能提升的关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. **潜在推理可用于安全守卫**：首次成功将 latent reasoning 应用于 LLM 安全 moderation 任务，在保持甚至提升性能的同时极大提升效率。
2. **解耦设计有效**：将常规预测与理由生成解耦，既能保证线上效率，又能支持按需审计。
3. **分阶段课程学习有益**：从显式推理逐步过渡到潜在推理，提供了更稳定的训练路径。
4. **审计证据实用性强**：生成的 compact audit artifacts 能有效支持人工审查，AUS 达 85.75。

### **局限性**
1. **审计模式仍有额外开销**：虽然不在关键路径，但开启 audit mode 仍需额外计算，适合抽样审查而非每条记录都审计。
2. **审计证据非完全忠实**：生成的审计 artifact 是证据摘要，而非模型内部计算的精确重建，应结合原始输入和标签一起解读。
3. **当前为纯文本场景**：未扩展到多模态输入、长对话或多轮策略演进等复杂场景。

### **未来工作方向**
- 优化系统级设计，支持高频审计需求。
- 扩展至 **multimodal inputs**（如图像、视频）、**longer conversations** 和动态安全策略。
- 探索更高效的潜在表示和投影机制。
- 结合人类反馈进一步优化审计 artifact 的质量和忠实度。

---

> **总结**：LatentGuard 提出了一种 **高效、可检查、高性能** 的 LLM 安全守卫新范式，通过 **潜在推理压缩** 和 **按需审计解码**，成功平衡了效率、准确性和透明性，为可部署的 LLM 安全系统提供了实用解决方案。

</details>

---

### 14. [TAOT: Topology-Aware Optimal Transport for Dynamic Expert Replica Placement in MoE Training](https://arxiv.org/abs/2608.03676)

**Authors**: Lingyun Zhang, Henghua Zhang, Shilei Gu, Kai Mo, Shuai Han, Shiyong Li, Yanpeng Wang, Dou Shen  
**Category**: cs.DC  
**Published**: 2026-08-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.03676v1  

#### Abstract
Mixture-of-Experts (MoE) has become a key architecture for scaling large language models (LLMs), yet its dynamic routing causes severe load imbalance in expert-parallel training. Existing dynamic-replica methods copy hot experts onto idle ranks to share computation, but they optimize load balance al...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TAOT: Topology-Aware Optimal Transport for Dynamic Expert Replica Placement in MoE Training

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Mixture-of-Experts (MoE)** 模型的分布式训练中，由于动态路由（dynamic routing）导致专家负载严重不均衡（load imbalance），某些 GPU 成为“straggler”拖慢整体训练速度。现有系统级负载均衡方法（如 hot-expert replication）通过复制热点专家到空闲 GPU 上来缓解该问题，但**忽略了跨节点通信成本**。

这些方法仅优化负载平衡，却可能因在多节点集群中频繁进行跨节点（inter-node）的专家权重传输而引入高昂的通信开销（如 InfiniBand vs. NVLink），最终反而增加端到端训练时间。

---

### 提出了什么新方法或新思路
本文提出 **TAOT (Topology-Aware Optimal Transport)**，一种面向 MoE 动态专家副本放置的拓扑感知最优传输方法，其核心思想是：

> **在副本放置决策中同时考虑“峰值削减收益”和“专家移动的通信代价”，实现两者的最优权衡。**

具体创新包括：

#### ✅ 创新点一：拓扑感知的动态副本放置建模
首次将 **每 GPU 的峰值削减增益** 和 **基于网络拓扑的专家权重移动成本** 统一建模为一个带熵正则化的 **Optimal Transport (OT)** 问题：
- 以过载（excess）作为“供应”
- 以空闲容量（spare capacity）作为“需求”
- 以通信成本矩阵 $ W_{r,e} $（intra-node=1, inter-node=λ）作为运输代价

这使得副本放置不仅追求负载均衡，还优先选择低通信成本路径。

#### ✅ 创新点二：GPU 友好、低开销的三阶段规划算法
设计了一个高效的三阶段调度流程：
1. **Phase 1: Sinkhorn-Knopp 流规划**  
   求解熵正则化 OT 得到软流提示（soft flow hint），体现全局拓扑偏好。
2. **Phase 2: 列优先整数匹配（Column-first Matching）**  
   结合溢出量、空闲容量与 OT 提示，生成整数级副本分配计划。
3. **Phase 3: 拉格朗日拍卖机制（Lagrangian Auction）**  
   实现 token 级精确调度，在满足约束的同时嵌入拓扑偏好。

整个过程可高效运行于 GPU，且支持 CUDA Graph。

#### ✅ 创新点三：通信-计算重叠执行设计
在系统层面，**重叠 guest-expert 权重传输与 home-expert 的前向计算**，有效隐藏通信延迟，进一步提升吞吐。

---

### 相比现有方法的优势
| 方面 | TAOT | 现有方法（如 Echo, LPLB, LLEP） |
|------|------|-------------------------------|
| **目标函数** | 同时优化负载平衡 + 通信成本 | 仅优化负载平衡 |
| **拓扑感知** | 显式建模 intra/inter-node 成本差异 | 忽略或硬编码图结构（如 Cube/Torus） |
| **通信效率** | 主动减少跨节点副本迁移 | 可能引发大量高成本跨节点通信 |
| **扩展性** | 随 EP 规模增大性能增益更显著 | 固定图结构限制候选空间，易出现局部资源耗尽 |
| **系统集成** | 支持 computation-communication overlap | 多数未考虑通信隐藏 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **The Pile-test dataset**：用于提供真实、多样化的预训练流量。
- 注意：作者强调 TAOT 是系统级调度器，其行为与具体语料无关，因此使用 Pile-test 仅为模拟典型 MoE 路由分布。

---

### 实验设置
- **硬件平台**：4 节点 × 8 × NVIDIA A800 GPU（共 32 卡）
  - 节点内：NVLink 连接
  - 节点间：InfiniBand
  - 设定通信成本比：intra : inter = 1 : 3（即 λ = 3）
- **模型**：Qwen3-30B-A3B MoE 模型
- **并行配置**：TP=4, PP=2, EP=16 或 EP=32
- **框架基础**：基于 Megatron-Core 自研训练框架

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **End-to-end Speedup** | 单步 Forward+Backward 时间加速比 |
| **Balance Quality** | 最终负载不平衡度（最大偏离均值的比例）及改善幅度（improvement in pp） |
| **Weighted Expert-Communication Cost** | 加权通信成本 = intra-node transfers × 1 + inter-node × 3 |
| **Intra/Inter-node Transfer Count** | 分别统计两类副本迁移次数 |
| **Online Planning Overhead** | 调度算法耗时占前向传播比例 |

---

### 基线方法对比
| 方法 | 简介 |
|------|------|
| **Megatron-LM (ECHO)** | 基于溢出量快速匹配副本，无拓扑感知 |
| **LLEP (ICML 2026)** | 将超载 token 和专家迁移到最轻载 rank，近似在线 spill scheduling |
| **LPLB (DeepSeek, 2025)** | 使用线性规划 + 预定义图结构（Cube/Torus）限制副本路径 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **端到端加速比**：相比 Megatron-LM 提升 **42.82%**（从 155.4ms → 108.8ms per step）
- **等效加速比**：论文摘要称达 **1.43× end-to-end speedup**
- **通信成本降低**：最高降低 **74%** 的加权专家通信成本
- **负载不平衡改善**：初始不平衡达 70% 时仍能有效削峰

---

### 与基线方法的对比结果（见 Table 1 & Figure 3）

#### 在 **EP=16** 下：
- TAOT 与 LPLB 平衡质量接近（略差 ~1pp），但通信成本大幅下降：
  - TAOT: **17–27** vs. LPLB 固定 **32**
  - 最多降低 **53%**

#### 在 **EP=32** 下（更具挑战性）：
- **平衡质量最优或并列最佳**
- **通信成本最低**：**33–44**，比 ECHO 低最多 55%，比 LPLB 低最多 **74%**
- LPLB 性能退化明显：受限于 Torus 图结构，远端空闲 rank 不可达，导致“局部资源枯竭”

> 📈 结论：**随着 EP 规模扩大，TAOT 的优势更加突出。**

---

### 消融实验结果（Ablation Study, Table 2）
在 EP=32、初始不平衡 70% 场景下验证两个核心组件作用：

| 配置 | Final Imbalance | Inter-node Transfers | Weighted Cost |
|------|------------------|------------------------|---------------|
| 无 Phase 1 & 2 | 1.95% | 18.33 | 59.67 |
| 仅有 Phase 2（通信成本建模） | 2.00% | 10.00 | 44.67 |
| 完整 TAOT（+ Phase 1 流提示） | **1.48%** | **9.67** | **44.00** |

✅ 发现：
- Phase 2 显著减少跨节点迁移（↓45%），控制通信成本；
- Phase 1 的全局流提示进一步提升负载均衡能力（↓0.52pp），避免局部贪婪导致不合理路径；
- 二者协同达到最佳 balance-communication trade-off。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **仅优化负载平衡不足以提升端到端性能**：忽视通信拓扑可能导致“越优化越慢”。
2. 🧭 **拓扑感知至关重要**：应将 intra/inter-node 带宽差异显式建模进调度决策。
3. ⚖️ **TAOT 实现了“近最优平衡 + 最低通信成本”的双重优势**：
   - 在小规模（EP=16）下以微小平衡损失换取巨大通信节省；
   - 在大规模（EP=32）下实现全面领先。
4. 📈 **可扩展性强**：随着 EP 规模和初始不平衡加剧，TAOT 增益持续上升（最高达 1.79×）。
5. ⏱️ **调度开销极低**：<1% 前向时间，适合微批次粒度在线调度。

---

### 方法的局限性
- 当前依赖预设 spare slot 数量（`--moe-num-echo-experts`），未自动学习最优冗余度；
- 假设通信成本静态（λ=3），未考虑运行时链路波动；
- 主要针对 EP 并行，与其他并行策略（如 TP/PP）耦合较深，通用性有待验证；
- 实验集中在 A800 集群，对其他硬件架构（如 H100 + NVSwitch）适应性需进一步测试。

---

### 未来工作方向
- 结合 **动态预测机制**（如 ARIMA、RL）提前识别热点专家，实现 proactive replica placement；
- 探索 **自适应 spare slot 分配**，根据实时负载弹性调整副本数量；
- 扩展至 **异构集群环境**，支持不同带宽层级的多级拓扑建模；
- 与 **routing algorithm co-design** 联合优化，实现算法-系统协同调优；
- 开源 TAOT 实现（GitHub: [https://github.com/baidu-baige/LoongForge](https://github.com/baidu-baige/LoongForge)），推动社区共建。

--- 

> ✅ **一句话总结**：  
> TAOT 首次将 **Optimal Transport** 引入 MoE 动态副本调度，通过 **拓扑感知的软流引导 + 整数匹配 + 通信隐藏**，实现了 **负载均衡与通信效率的双重突破**，为大规模 MoE 训练提供了高效、可扩展的新范式。

</details>

---

### 15. [Maglev: Sliding Recurrent Memory](https://arxiv.org/abs/2608.02870)

**Authors**: Bo Liu, Qiang Liu  
**Category**: cs.LG  
**Published**: 2026-08-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.02870v1  

#### Abstract
We introduce \ours{}, a recurrent Transformer architecture with fixed-size memory that generalizes sliding-window attention while remaining parallelizable during training. \ours{} consists of two coupled models: a prefiller $Q$, which leverages full attention\footnote{In practice, we use interleaved...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Maglev: Sliding Recurrent Memory》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 **Transformer** 模型依赖 **full attention**，其计算和内存开销随上下文长度增长而增加，难以处理长序列。  
**Sliding-window attention** 虽然限制了计算成本，但会丢弃窗口外的历史信息，缺乏持久的长期记忆能力。  
同时，现有的 **recurrent memory** 方法（如 LRT、RetNet）通常在块级或段级更新记忆，引入了人为的粒度边界，且非线性记忆更新难以并行训练。

Maglev 旨在解决以下矛盾：
- 如何实现 **token-level 的非线性记忆更新**
- 同时保持 **bounded inference memory**
- 并支持 **parallel pretraining**

---

### 提出了什么新方法或新思路
Maglev 提出了一种 **双阶段训练 + 单阶段推理** 的架构，核心是 **lifted parallel training** 和 **memory consistency loss**：

- **Prefiller Q**：一个更强的因果模型（使用 full + sliding attention），在训练时并行地为整个序列生成“理想”的记忆目标 $ m' $
- **Decoder P**：一个仅使用 sliding-window attention 的模型，在训练时接收前一时刻的 $ m'_{t-1} $，预测下一 token 并生成自己的记忆 $ m_t $
- **Consistency Loss**：通过 $ \|m_t - m'_t\|^2 $ 对齐 $ m_t $ 和 $ m'_t $，使 P 学会在没有 Q 的情况下自洽地生成有效记忆
- **Inference 时**：丢弃 Q，P 使用自身生成的记忆 $ m_{t-1} $ 进行递归推理，形成真正的 token-wise recurrence

此外，Maglev 将记忆注入到 **KV-cache** 中（via recurrent K/V injection），无需额外 memory tokens 或特殊结构。

---

### 相比现有方法的优势
| 特性 | Maglev | Sliding Window | LRT | Linear RNNs (e.g., Mamba) |
|------|--------|----------------|-----|----------------------------|
| 非线性记忆更新 | ✅ 全深度 Transformer 更新 | ❌ | ✅ | ❌（多为线性/仿射） |
| 固定推理内存 | ✅ | ✅ | ✅ | ✅ |
| 可并行预训练 | ✅（via lifted training） | ✅ | ✅（类似机制） | ✅ |
| 无块级边界 | ✅ | ✅ | ✅ | ✅ |
| 参数效率 | ✅（P/Q 可共享参数） | - | ⚠️（需额外路径） | - |

> ✅ **关键优势**：Maglev 在保持 sliding-window 推理效率的同时，实现了更强大的非线性记忆建模，并通过一致性学习避免了训练时的序列展开。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **预训练数据**：FineWeb-Edu（43.52B tokens）
- **验证集**：
  - FineWeb-Edu validation（用于评估 BPB）
  - LAMBADA（语言建模难度）
- **下游任务基准**（zero-shot evaluation）：
  - PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, SocialIQA, BoolQ

---

### 实验设置
- **模型规模**：nanochat d20 架构
  - 20 layers, hidden dim 1280, 10 heads, head dim 128
  - 总参数量约 **435M**（不含 embedding）
- **序列长度**：2048
- **滑动窗口大小**：W = 512
- **Optimizer**：MuonAdamW，每步 524,288 tokens
- **训练总 token 数**：43.52B（相当于 Chinchilla 的 100× token budget）

---

### 基线方法对比
| 模型 | 注意力模式 | 是否有记忆机制 |
|------|-----------|----------------|
| Transformer (SLSL) | Sliding + Full (interleaved) | ❌ |
| SWA (SSSS) | Pure sliding-window | ❌ |
| LRT (SLSL / SSSS) | 含 recurrent memory injection | ✅ |
| Maglev (shared/separate) | P: SSSS, Q: SLSL | ✅（with consistency loss） |

> 所有模型共享相同的 tokenizer、data pipeline 和 evaluation script。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | FW BPB ↓ | PPL ↓ | Avg Acc ↑ |
|-------|----------|--------|------------|
| SWA (SSSS) | 0.7413 | 8.54 | 54.1 |
| LRT (SSSS) | 0.7331 | 7.92 | 55.0 |
| **Maglev (shared, λ=0.1)** | **0.7295** | 8.06 | **56.2** |
| **Maglev (sep., λ=1)** | **0.7251** | 8.06 | **56.4** |

> ✅ **最佳 Maglev 模型将 FW BPB 从 0.7413 提升至 0.7251**，平均下游准确率从 54.1 提升至 **56.4**

---

### 与基线方法的对比结果
- 相比纯滑窗模型（SSSS）：
  - BPB ↓ 0.0162（相对提升 ~2.2%）
  - 下游平均准确率 ↑ 2.3 pts
- 相比同结构 LRT 模型（SSSS）：
  - 进一步提升 BPB 和多数任务表现（如 BoolQ 从 56.2 → 64.0）
- 表明 **prefiller 提供的更强监督信号优于 LRT 的自回归 refine 机制**

---

### 消融实验结果
#### 参数共享 vs 分离
| 设置 | FW BPB | Avg Acc |
|------|--------|---------|
| Shared P/Q (λ=0.1) | 0.7295 | 56.2 |
| Separate P/Q (λ=1) | **0.7251** | **56.4** |

- 分离参数性能更好，说明更强的 prefiller 能提供更有价值的记忆目标
- 但共享参数仍保留大部分增益，显著节省参数内存

#### 一致性损失权重 λ 影响
| λ | Shared Model 性能 | Separate Model 性能 |
|----|--------------------|----------------------|
| 0.1 | 较好（平衡预测与一致性） | 次优 |
| 1.0 | 下游部分任务下降（如 BoolQ 51.1） | 最佳整体表现 |

> 发现：当 P 和 Q 共享参数时，过强的 consistency loss 会约束 decoder 表示空间；分离时则可承受更高 λ

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Maglev 成功实现了 token-level 非线性记忆更新**，且推理时保持 bounded memory
2. ✅ **通过 prefiller 提供并行化的 memory 目标**，解决了非线性 recurrence 难以并行训练的问题
3. ✅ **一致性损失能有效引导 decoder 学习自洽的记忆传播行为**
4. ✅ **P 和 Q 可高度共享参数而不显著损失性能**，提升了参数效率
5. ✅ 在相同部署成本下，Maglev 显著优于 sliding-window 和 LRT 基线

---

### 方法的局限性
- **训练需要两倍计算量**：因需执行 prefiller 和 decoder 两次前向传播
- **对 prefiller 容量敏感**：若 Q 不够强，则 memory 目标质量受限
- **尚未在更大规模上验证**：当前实验限于 435M 模型，scaling behavior 待进一步研究
- **recurrent K/V injection 的稳定性** 可能在极长序列中面临挑战

---

### 未来工作方向
1. **Scaling Maglev**：研究 prefiller 强度与 decoder 容量之间的 trade-off
2. **高效 inference kernel**：优化 recurrent K/V 注入的实际部署效率
3. **Prefiller 知识蒸馏**：使用预训练好的大模型作为固定 prefiller，只训练 compact decoder
4. **探索不同的 memory sharing 模式**：
   - 共享 embedding / MLP / attention projection
   - 分层共享策略
5. **替代 memory 注入方式比较**：
   - Residual-stream injection
   - Cross-attention based memory read/write
   - Layer-specific memory projection

> 🚀 **最终愿景**：构建一种既能享受 Transformer 表达力，又具备 RNN 式持久记忆和高效推理的语言模型架构 —— Maglev 是朝这一方向迈出的重要一步。

</details>

---

### 16. [Schedule-Informed Temporal Fusion Forecasting of Hourly Airport Security-Checkpoint Throughput](https://arxiv.org/abs/2608.02950)

**Authors**: Yinxiao Zhang, Sen Wang, Yi Gao  
**Category**: cs.LG  
**Published**: 2026-08-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.02950v1  

#### Abstract
Checkpoint staffing requires accurate forecasts of when screening demand will occur, yet flight schedules record departure times rather than passenger arrival times at security checkpoints. This study develops a framework that converts known flight schedules into temporally aligned signals for forec...

---

### 17. [LLaDA MoE v2: Scaling Mixture-of-Experts Diffusion Language Models](https://arxiv.org/abs/2608.03457)

**Authors**: Fengqi Zhu, Shaoxuan Xu, Jingyang Ou, Zebin You, Yipeng Xing, Huabin Liu, Xiaolu Zhang, Jun Zhou, Zhenzhong Lan, Yankai Lin, Wayne Xin Zhao, Jianguo Li, Chongxuan Li, Ji-Rong Wen  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.03457v1  

#### Abstract
Diffusion language models (dLLMs) offer an alternative to autoregressive (AR) language modeling, yet the scaling behavior of Mixture-of-Experts (MoE) dLLMs remains poorly understood. We systematically characterize how optimization hyperparameters, compute allocation, and architecture scale for MoE d...

---

### 18. [When Efficiency Becomes Fragility: Exploiting Dynamic Routing Vulnerabilities in Adaptive UAV Tracking](https://arxiv.org/abs/2608.03902)

**Authors**: Shaofeng Liang, Runwei Guan, Wenshuo Chen, Jiemin Wu, Bowen Tian, Haozhe Jia, Kaishen Yuan, Songning Lai, Daizong Liu, Yutao Yue  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.03902v1  

#### Abstract
Resource constraints on UAV platforms have driven a paradigm shift in aerial tracking, from pursuing performance toward balancing accuracy with efficiency. Adaptive Transformer Trackers, which leverage an input-dependent dynamic routing architecture, have emerged as a representative solution to this...

---

### 19. [Efficient Multilingual Neural Machine Translation via Corpus-Driven Vocabulary Pruning: An English-Arabic Case Study](https://arxiv.org/abs/2608.03480)

**Authors**: Ahmed Amine Aliane, Nasredine Semmar, Hassina Aliane  
**Category**: cs.CL  
**Published**: 2026-08-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.03480v1  

#### Abstract
The adoption of large pre-trained multilingual models for neural machine translation (MNMT) faces a major challenge: excessive memory and computational consumption due to overly large vocabularies and embedding layers. Although existing compression methods like pruning, quantization and knowledge di...

---

### 20. [Beyond Initialization Loss: A Systematic Study of Token Embedding Initialization Strategies for LLM Vocabulary Extension](https://arxiv.org/abs/2608.03494)

**Authors**: Raviraj Joshi, Utkarsh Vaidya, Sanjay Singh Chauhan, Niranjan Wartikar  
**Category**: cs.CL  
**Published**: 2026-08-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.03494v1  

#### Abstract
Vocabulary extension is an efficient way to adapt pretrained large language models (LLMs) to new languages, but the initialization of newly added token embeddings can strongly affect continued pre-training (CPT) efficiency. We present a systematic study of more than 20 initialization strategies for ...

---

### 21. [Accelerating Dynamic Graph Clustering on GPU Architectures with cuGraph](https://arxiv.org/abs/2608.03695)

**Authors**: Nelson Aloysio Reis de Almeida Passos, Emanuele Carlini, Salvatore Trani  
**Category**: cs.DC  
**Published**: 2026-08-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.03695v1  

#### Abstract
This work addresses community detection in temporal networks through GPU-accelerated extensions of spectral clustering and modularity-based algorithms originally designed for static graphs. Built on the NVIDIA RAPIDS ecosystem, the framework enables the characterization and tracking of communities i...

---

### 22. [A Physics-Flavored Transformer Network for Parametrizing Contraction Dynamics of Engineered Skeletal Muscle Tissues](https://arxiv.org/abs/2608.03927)

**Authors**: Mattias Luber, Timo Betz  
**Category**: cs.LG  
**Published**: 2026-08-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.03927v1  

#### Abstract
Engineered Skeletal Muscle Tissues (ESMs) have become a key structure for biomedical disease modeling and pharmacological screening, yet their functional characterization often relies on simplistic metrics like peak force, discarding critical kinetic information. This is partially due to the high le...

---

### 23. [Don't Peek at the Answer: Outcome-Masked Group Relative Policy Optimization for Label-Free RLVR](https://arxiv.org/abs/2608.03119)

**Authors**: Yongshi Ye, Liang Zhang, Yidong Chen, Xiaodong Shi, Biao Fu  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.03119v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) improves LLM reasoning but typically relies on ground-truth (GT) answers, limiting scalability. Voting-based label-free RLVR replace gold supervision with answer-level consensus from model samples. However, collapse arises when the same answer-le...

---

### 24. [LeanMem: Simple and Efficient Long-Term Memory for LLM Agents](https://arxiv.org/abs/2608.03463)

**Authors**: Yuxin Liao, Le Wu, Min Hou, Hao Liu, Han Wu, Zishu Wang  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.03463v1  

#### Abstract
Long-term memory is essential for LLM-based agents to sustain interactions and reliably leverage distant history. However, existing memory systems typically process heterogeneous dialogue content through a uniform summarization and retrieval pipeline, leading to either excessive token consumption or...

---

### 25. [Noise-Aware Shrinkage for Differentially Private Zeroth-Order Fine-Tuning of Large Language Models](https://arxiv.org/abs/2608.03277)

**Authors**: Lele Zheng, Weifeng Kong, Xinyi Zhang, Ke Cheng, Tao Zhang, Yulong Shen  
**Category**: cs.LG  
**Published**: 2026-08-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.03277v1  

#### Abstract
Differentially private zeroth-order optimization (DP-ZO) enables memory-efficient private fine-tuning of large language models using only forward evaluations. Existing aggregation-based DP-ZO methods reconstruct model updates at a fixed scale, ignoring that the strength of useful signals varies thro...

---

### 26. [AS-FedBridge: Pseudo-Spike Bridge Distillation for Heterogeneous ANN-SNN Federated Learning](https://arxiv.org/abs/2608.03324)

**Authors**: Shengyang Li, Yiting Dong, Liuyang Song, Ximing Wang, Luyuan Xie, Cong Li, Qingni Shen, Zhaofei Yu  
**Category**: cs.LG  
**Published**: 2026-08-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.03324v1  

#### Abstract
Federated learning enables collaborative model training across distributed edge devices while strictly preserving data privacy. To facilitate practical deployment on resource-constrained edge devices, Spiking Neural Networks (SNNs) have emerged as a promising alternative to traditional Artificial Ne...

---

### 27. [Design-Time Optimization of Deep Neural Networks for Intermittent Learning on Microcontrollers](https://arxiv.org/abs/2608.03589)

**Authors**: Jakob Schubert, Maximilian Kasper, Maximilian Linke, Benedict Herzog, Mark Deutel, Axel Plinge, Dominik Seuss, Christopher Mutschler  
**Category**: cs.LG  
**Published**: 2026-08-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.03589v1  

#### Abstract
We present a method for designing deep neural networks (DNNs) for intermittent, energy-autonomous, on-device learning on microcontroller units (MCUs). In mobile applications where the energy can run out, e.g., when solar-powered, executing artificial intelligence (AI) faces a technical issue as lear...

---

### 28. [UniNav: A Unified World-Action Diffusion Model for Visual Navigation](https://arxiv.org/abs/2608.03244)

**Authors**: Changqing Zhou, Yueru Luo, Zeyu Jiang, Changhao Chen  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.03244v1  

#### Abstract
Image-goal visual navigation is a fundamental capability for embodied agents. Existing navigation policies efficiently predict waypoint trajectories but lack visual foresight, while navigation world models can anticipate future observations but often require costly planning rollouts. We present UniN...

---

### 29. [Taming the Implicit: Dual-Channel Risk-Aware Reinforcement Fine-Tuning for Continual Multimodal Post-Training](https://arxiv.org/abs/2608.03660)

**Authors**: Yibei Liu, Jiajun Chen, Qianle Zhang, Tangyue Jin, Mengying Zhu, Meng Xi, Yangyang Wu  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.03660v1  

#### Abstract
Reinforcement fine-tuning (RFT) is widely believed to inherently resist catastrophic forgetting in continual post-training of multimodal large language models. Under pronounced task distributional shifts, however, forgetting across representative RFT algorithms escalates sharply. This stems from the...

---

### 30. [Oilbird: Training-Free Speculative Decoding with Keys the Verifier Already Computes](https://arxiv.org/abs/2608.03839)

**Authors**: Tao Jin, Phuong Minh Nguyen, Zhenzhu Yan, Teeradaj Racharak, Naoya Inoue  
**Category**: cs.AI  
**Published**: 2026-08-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.03839v1  

#### Abstract
Training-free speculative decoding drafts by matching an exact suffix of the context against a pool of earlier context. That lookup misses correct drafts already in the pool, most visibly on tool-calling traffic, where a request repeats almost everything but the few values minted for it, and where o...

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

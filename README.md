# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-07 08:57:31 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [BrownoutMoE: Structure-Aware Expert Grouping for Efficient and Accurate LLM Web-based Services](https://arxiv.org/abs/2607.04164)

**Authors**: Yi Ding, Minxian Xu, Zhengxin Fang, Kejiang Ye, Chengzhong Xu  
**Category**: cs.DC  
**Published**: 2026-07-07  
**Score**: 15.5  
**Type**: new  
**ArXiv ID**: 2607.04164v1  

#### Abstract
Mixture-of-Experts (MoE) large language models (LLMs) are increasingly deployed in Web-facing services, where inference must be both accurate and responsive under bursty demand. Although MoE models improve parameter efficiency through sparse expert activation, efficient MoE inference remains challen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BrownoutMoE: Structure-Aware Expert Grouping for Efficient and Accurate LLM Web-based Services

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在基于 **Mixture-of-Experts (MoE)** 架构的大型语言模型（LLM）中，推理服务面临两个核心挑战：
- **专家负载不均衡**（Expert Load Imbalance）：少数“热”专家处理大部分请求，而多数“冷”专家利用率极低，导致 GPU 并行效率低下。
- **次优专家分组带来的精度下降**：现有系统（如 BrownoutServe）采用简单的索引顺序分组（sequential grouping），忽略了专家之间的行为相似性，导致知识蒸馏时误差增大，显著影响下游任务准确率。

这些问题限制了 MoE 模型在 Web 服务场景下的高效部署，尤其是在突发流量下难以兼顾 **throughput**、**tail latency** 和 **accuracy**。

---

### 🚀 提出的新方法与创新思路

论文提出 **BrownoutMoE** —— 一种**结构感知的专家分组框架**，其核心思想是将专家按**行为相似性**进行智能分组，而非简单地按索引排列。

#### 主要创新点包括：

1. **将专家分组建模为离散策略优化问题**
   - 首次将 `expert grouping` 视为一个可学习的决策过程，而非固定启发式规则。
   - 使用 **Group Relative Policy Optimization (GRPO)** 进行强化学习搜索最优分组方案。

2. **以真实蒸馏误差作为奖励信号**
   - 奖励函数定义为负的 post-distillation MSE（均方误差），直接反映模型压缩后的行为保真度。
   - 通过短周期蒸馏（short-horizon distillation）快速评估候选分组质量，提升搜索效率。

3. **两阶段训练流程**
   - **第一阶段（Offline）**：使用 GRPO 搜索最优分组映射 `g*`；
   - **第二阶段（Offline）**：基于该分组执行完整的 **grouping-consistent united-expert distillation**，生成可用于标准推理流水线的轻量化模型。

4. **warm-start 初始化 + early stopping 加速收敛**
   - 利用层次聚类（hierarchical clustering）对专家输出相似性进行预分析，提供高质量初始分组，加快 RL 收敛。
   - 引入 early stopping 机制避免无效迭代。

5. **SLO-aware 动态控制机制集成**
   - 在线服务时结合 SLO 分析器动态调整 brownout 阈值，在延迟与精度之间实现自适应权衡。

---

### 🔍 相比现有方法的优势

| 方面 | BrownoutServe（Baseline） | BrownoutMoE（本文） |
|------|----------------------------|---------------------|
| 分组策略 | 固定索引顺序分组（non-structure-aware） | 学习式行为相似分组（structure-aware） |
| 分组依据 | 无语义依据，可能合并差异大的专家 | 基于输出行为相似性，最小化蒸馏误差 |
| 优化目标 | 执行效率优化（调度/通信重叠等） | 结构层面优化，解决根本性碎片化问题 |
| 准确性保留 | 显著退化，尤其在高 k 值时 | 大幅减少精度损失 |
| 推理开销 | 无额外运行时计算 | GRPO 仅用于离线校准，不影响线上延迟 |

> ✅ 核心优势：**在不增加在线计算成本的前提下，显著提升 accuracy-throughput 权衡表现**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### 精度评估基准（Accuracy）：
- **PIQA**：常识推理任务
- **COPA**：因果推理任务
- **C-Eval（validation set, few-shot）**：中文综合知识评测
- **OpenBookQA（OBQA, 5-shot）**：科学知识问答

#### 吞吐量与系统性能测试：
- **ShareGPT** 和 **Alpaca**：实际对话轨迹数据集，用于模拟真实请求分布

#### 校准数据（Calibration Data）：
- 从上述数据集中采样用于收集专家激活频率和隐藏状态输出，供 GRPO 搜索与蒸馏使用

---

### ⚙️ 实验设置

- **模型**：Qwen1.5-MoE-A2.7B-Chat（总参数 14.3B，每层 60 个 routed experts，top-4 路由）
- **硬件平台**：8 × NVIDIA A100-80GB GPUs，启用 tensor parallelism
- **MoE 层数**：24 层 FFN 替换为 MoE
- **分组配置**：2-way、4-way、8-way grouping（即每组分别包含 2、4、8 个原始专家）

---

### 🎯 评估指标

| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 下游任务正确率，衡量模型语义能力保持程度 |
| **Throughput (tokens/s)** | 系统每秒处理的 token 数量，反映吞吐性能 |
| **Latency (TTFT / TPOT)** | Time-To-First-Token 和 Time-Per-Output-Token，关注响应延迟 |
| **MSE (Mean Squared Error)** | 蒸馏前后专家输出的均方误差，用于衡量分组合理性 |
| **Speedup** | 相对于 baseline 的加速比 |

---

### 🆚 基线方法对比

| 方法 | 描述 |
|------|------|
| **Zero Brownout** | 不启用 united expert，所有专家独立运行 → 最高 accuracy 上限 |
| **Full Brownout** | 所有冷专家请求都路由到 united expert → 最高速度，最低 accuracy 下限 |
| **BrownoutServe (Sequential Grouping)** | 原始索引顺序分组，作为主要 baseline |
| **BrownoutMoE (Ours)** | 本文提出的 GRPO + 行为感知分组方法 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **最大 accuracy 提升** | 最多减少 **71.4% 的 accuracy degradation**（PIQA, 8-way, threshold=0.4） |
| **平均 accuracy 改善** | 在 C-Eval 上减少 46.1%~45.3% 退化；OBQA 绝对 accuracy 提升达 **+7.2%** |
| **最大吞吐提升** | 达到 **2.24× 更高 throughput**（Alpaca 数据集，unfused MoE） |
| **峰值吞吐量** | 融合内核（fused MoE）下可达 ~1800 tokens/s |
| **扩展性** | 支持从 2-GPU 到 8-GPU 的良好横向扩展，8-GPU 比 2-GPU 提升约 **3.2× 吞吐**

---

### 🔁 与基线方法的对比结果（代表性案例）

#### 在 8-way grouping、threshold=0.4 时：
| Benchmark | BrownoutServe (Acc) | BrownoutMoE (Acc) | Improvement |
|----------|------------------------|--------------------|-------------|
| **PIQA** | 显著下降 | 仅轻微下降 | ↓ **71.4% degradation** |
| **C-Eval** | 下降 20.4%（vs zero-brownout） | 下降仅 11.1% | ↓ **45.3% degradation** |
| **COPA** | 下降严重 | 控制良好 | ↓ **50.0% degradation** |
| **OBQA** | 性能较差 | 提升明显 | ↑ **绝对 +7.2% accuracy** |

> 💡 发现：随着 grouping factor `k` 增大（如 8-way），传统 sequential grouping 的劣势愈发明显，而 BrownoutMoE 的优势也更加突出。

---

### 🔍 消融实验与关键验证

1. **分组质量 vs 蒸馏误差**
   - 对比 hierarchical clustering 初始化 vs 随机初始化：前者使初始 MSE 降低 **17.9%–48.8%**
   - 证明：**行为相似性先验显著提升分组质量**

2. **GRPO 搜索的有效性**
   - GRPO 找到的分组相比 sequential grouping，在相同蒸馏条件下 MSE 平均降低 >40%
   - 且搜索过程稳定，early stopping 可有效终止冗余迭代

3. **不同 grouping factor 影响**
   - 2-way grouping 差距较小（因合并专家少）
   - 但 **4-way 和 8-way grouping 中，BrownoutMoE 明显更接近 zero-brownout 上限**

4. **fused vs unfused MoE 对 throughput 的影响**
   - 在 unfused 情况下，brownout 效果显著（up to 2.24×）
   - 在 fused MoE 下，kernel fusion 成为主要瓶颈，不同 brownout 配置趋于收敛

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **专家分组策略直接影响蒸馏质量和最终 accuracy**
   - 单纯减少专家数量不足以保证性能，**如何组合专家才是关键**。
   - 行为相似的专家更适合被合并，否则会引入不可逆的信息损失。

2. **结构感知分组优于固定启发式规则**
   - sequential grouping 忽略了专家功能多样性，造成“强弱混合”，加剧误差传播。
   - BrownoutMoE 通过 GRPO 学习到更合理的专家聚类结构。

3. **离线优化 + 在线轻量部署是可行路径**
   - GRPO 搜索虽耗时，但完全可在离线阶段完成，不影响线上服务延迟。
   - 最终只需加载 grouping map 和 united expert weights，兼容现有推理引擎（如 vLLM）。

4. **accuracy 与 efficiency 的 trade-off 得到显著改善**
   - 在相同 brownout threshold 下，BrownoutMoE 能提供更高 accuracy；
   - 或在相同 accuracy 要求下，允许更激进的 brownout 设置以换取更高 throughput。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖校准数据** | 需要一定量的真实输入来采集 routing statistics 和 expert outputs，若 workload shift 可能需重新校准 |
| **离线搜索时间较长** | GRPO + 多轮 short-horizon distillation 带来较高离线成本（主要受 `T`, `N`, `B` 影响） |
| **未支持动态结构调整** | grouping map 是静态的，无法在运行时根据负载变化实时调整结构 |
| **适用于 top-k routing MoE** | 当前设计假设明确的 expert selection pattern，对 soft-routing 或 dense MoE 不适用 |

---

### 🔮 未来工作方向

1. **自动化 workload 自适应机制**
   - 当检测到输入分布漂移时，自动触发重新校准流程，更新 grouping map。

2. **跨层联合分组优化**
   - 当前为 per-layer 独立优化，未来可探索 layer-wise joint grouping 以捕捉更深的结构相关性。

3. **轻量化 GRPO 替代方案**
   - 探索基于 proxy metric（如 cosine similarity + clustering）的快速近似方法，替代部分 RL 搜索。

4. **与 routing algorithm 联合设计**
   - 将 grouping-aware routing 机制纳入训练阶段，形成端到端优化闭环。

5. **扩展至其他稀疏架构**
   - 如 DeepSeek-V3 的 shared+routed MoE、Mixtral 等模型，验证通用性。

---

> ✅ **总体评价**：  
> BrownoutMoE 提出了一种新颖且实用的视角——**将 MoE 推理效率问题从“执行层”上升到“结构层”**。它不仅提升了当前 brownout 类系统的性能上限，也为未来的 MoE 模型压缩与服务优化提供了新的研究范式。

</details>

---

### 2. [Communication-Aware Placement and Pruning for Efficient Mixture-of-Experts Inference](https://arxiv.org/abs/2607.05116)

**Authors**: Xiao Shi, Yingying Sun, Jiangsu Du, Zhiguang Chen, Yutong Lu  
**Category**: cs.DC  
**Published**: 2026-07-07  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2607.05116v1  

#### Abstract
As MoE models scale to hundreds of experts, placement and pruning decisions increasingly dictate communication volume, affecting the performance of distributed inference across GPUs and nodes. We propose CAP (Communication-Aware Assignment and Pruning), a framework that considers computation, commun...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Communication-Aware Placement and Pruning for Efficient Mixture-of-Experts Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着 MoE（Mixture-of-Experts）模型规模扩大至数百个专家，**分布式推理中的通信开销**成为性能瓶颈。现有方法在优化 **expert placement** 和 **expert pruning** 时，通常只关注计算负载均衡或精度损失，而忽略了其对 **通信体积（communication volume）** 的显著影响，导致次优性能。

特别是现代大型 MoE 模型（如 Qwen3-30B-A3B 含 128 个专家）中，每个设备需托管多个专家，placement 决策直接影响 token 需访问的设备数量，从而决定 all-to-all 通信成本。

---

### 🚀 提出的新方法：CAP 框架
本文提出 **CAP (Communication-Aware Assignment and Pruning)**，一个综合考虑 **computation、communication 和 accuracy** 的高效 MoE 推理框架，包含三个核心组件：

#### （1）Co-activation Driven Expert Placement  
- **动机**：分析推理过程中的路由轨迹，发现某些专家存在强共激活（co-activation）模式。
- **方法**：构建专家间的 co-activation 图，将频繁共同激活的专家尽可能放置在同一 device 或 node 上，减少跨设备通信。
- **实现**：采用两阶段贪心算法进行图划分（GPU级 → Node级），降低通信目标函数 $ C(\pi) $。

#### （2）Comm.-Comp. Trade-off Adjustment  
- **动机**：最小化通信可能导致热专家集中，引发 load imbalance；反之亦然。二者存在权衡。
- **方法**：从通信最优的初始 placement 出发，通过交换专家逐步生成一系列具有不同 **communication-load 平衡** 的候选 placement。
- **优势**：形成一条 Pareto 前沿，可根据硬件特性选择最优策略。

#### （3）Communication-Aware Expert Pruning  
- **动机**：传统 pruning 只基于 routing score 忽略通信代价。
- **方法**：以 **device 为单位进行剪枝决策**，引入通信成本系数 $ c $（intra-node vs inter-node），建模为：
  $$
  \min \sum_d c_d x_d \quad \text{s.t.} \quad \sum_d a_d x_d \geq 1 - p
  $$
  其中 $ a_d $ 是设备精度贡献，$ c_d $ 是通信成本，$ p $ 是允许的精度损失阈值。
- **效果**：优先移除高通信成本且低贡献的 routing destination，更有效地降低通信。

---

### 🔍 相比现有方法的优势
| 方面 | 现有方法 | CAP |
|------|--------|-----|
| **Placement** | 仅优化 load balance（如 DeepSeek EPLB）或默认顺序分配 | 联合优化 communication + load balance，支持硬件自适应 |
| **Pruning** | 仅按 routing score 动态剪枝，忽略通信影响 | 显式建模通信代价，device-level 剪枝提升通信效率 |
| **整体设计** | 单一目标优化 | 多目标协同优化 pipeline，支持 lossless 与 lossy 加速 |

> ✅ **核心创新**：首次系统性地将 **communication volume** 引入 expert placement 与 pruning 的联合优化中，并提出可部署的端到端框架。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **推理流量数据集**：
  - `LMSYS-Chat`：真实对话请求，用于 end-to-end 性能测试。
  - `Arxiv Abstracts`：学术文本摘要，多样性较高。
- **精度评估数据集**：
  - `HumanEval`：代码生成能力
  - `MMLU`：多任务语言理解
  - `GSM8K`：数学推理

---

### ⚙️ 实验设置

#### 硬件平台（单节点 & 多节点）
| 节点 | GPU | 互联技术 | 类型 |
|------|-----|----------|------|
| Node A | 8×RTX 3090 | PCIe（无 NVLink/P2P） | **Comm.-constrained** |
| Node B | 8×A100 | PCIe + GPU-Direct P2P | **Balanced** |
| Node C | 8×H100 | NVLink | **Comm.-rich** |
| Cluster A/B | 2×8 H100，NVLink + InfiniBand（400Gbps / 200Gbps） | 多节点通信分析 |

#### 模型
- `Qwen3-30B-A3B`：每层 128 个专家
- `DeepSeek-V2-Lite`：每层 64 个专家

#### 基线方法（Baselines）
1. **Default**：按 expert ID 顺序依次分配（vLLM/SGLang 默认方式）
2. **DeepSeek EPLB**：当前最先进的 load-balance-oriented placement 方法

#### 评估指标
- **吞吐量（Throughput, RPS）**：满足 SLO 下的最大请求处理速率
- **延迟（Latency）**：TTFT（Time to First Token）、TBT（Time Between Tokens）
- **服务等级目标（SLO）**：
  - TTFT ≤ 6s
  - TBT ≤ 500ms
  - ≥90% 请求达标视为合规
- **精度指标**：
  - Perplexity（PPL）
  - HumanEval / MMLU / GSM8K 准确率
- **通信开销**：平均每个 token 访问的 device 数（dpt）
- **负载不均衡度**：$ B(\pi) = \max_g L_g / \frac{1}{k}\sum_g L_g $

---

## 3. 主要实验结果和性能指标

### 📈 End-to-End 吞吐量提升（Figure 11）
在多种硬件配置下，CAP 均显著优于基线：

| 设置 | CAP 提升倍数 |
|------|--------------|
| Node A（comm-constrained） | **1.86×** vs Default / EPLB |
| Node B（balanced） | **1.4–1.6×** |
| Node C（comm-rich） | **1.23–1.35×** |
| Cluster A（multi-node） | **~1.5×** |

> 💡 在通信受限环境下增益最大，说明 CAP 对通信敏感场景特别有效。

---

### 🔍 与基线对比分析
- **EPLB vs Default**：
  - 在 comm-rich 或 balanced 节点上表现良好（因 load balance 更好）
  - 但在 comm-constrained 节点（Node A）上甚至不如 Default —— 因其追求负载均衡导致通信剧增。
- **CAP vs 所有基线**：
  - 始终处于 Pareto 最优前沿（见 Figure 13）
  - 在相同速度目标下保持更高 accuracy（Figure 14）

---

### 🧪 消融实验与关键发现

#### （1）Placement Spectrum 分析（Figure 12）
- 不同 placement index 对应不同的 comm-load 权衡：
  - Index 0：通信最少，但负载最不均
  - Index 高：负载更均衡，通信更多
- **最佳选择依赖硬件**：
  - Comm-constrained（Node A）：选低 index（通信最小）
  - Balanced（Node B）：选中间 index（如 Placement 3）
  - Comm-rich（Node C）：选高 index（负载最均衡）

> ✅ 验证了“没有统一最优 placement”，必须硬件感知。

#### （2）Pruning 方法对比（Figure 14）
- 在相同 pruning threshold $ p=0.2 $ 下：
  - CAP 比 naive pruning 实现 **更高的延迟降低（speedup）**
  - CAP 在达到相同 speedup 时（如 33% latency reduction），可用更小的 $ p $（即保留更多专家），从而 **保留更高 accuracy**

> 示例：CAP ($p=0.2$) ≈ Naive Pruning ($p=0.3$)，但前者 accuracy 更高。

#### （3）Accuracy 表现（Table I）
| 方法 | $p=0.2$ | $p=0.3$ |
|------|---------|---------|
| PPL ↑ | +0.81 | +1.76 |
| HumanEval ↓ | -0.61pp | -3.04pp |
| MMLU ↓ | +0.07pp | -2.80pp |
| GSM8K ↓ | -2.73pp | -3.49pp |

> 结论：增大 $ p $ 导致一致性的 accuracy 下降，尤其在复杂任务（code/math）上更明显。

#### （4）Multi-node Pruning 中通信成本的影响（Table II）
| Cluster | Config | Latency (ms) | Node per Token |
|--------|--------|---------------|----------------|
| A (high BW) | c=1 | 1434.68 | 1.8875 |
|           | c=5 | 1447.89 (+0.92%) | 1.8337 (-2.85%) |
| B (low BW) | c=1 | 2285.37 | 1.8875 |
|           | c=5 | **1980.26 (-13.35%)** | 1.8337 |

> ✅ 当 inter-node 带宽较低时，设置更高的 $ c $（强调跨节点通信惩罚）可显著降低延迟。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Communication 是 MoE 推理的关键瓶颈**，尤其在专家数量多、设备间带宽有限时。
2. **Expert placement 不仅影响 load balance，也深刻影响 communication volume**，不能仅以负载均衡为目标。
3. **Co-activation patterns 存在且稳定**，可用于指导 placement 设计。
4. **Pruning 应该是 communication-aware 的**，device-level 剪枝比 expert-level 更高效。
5. **最优策略是硬件相关的**：comm-constrained → 重通信；comm-rich → 重负载均衡。
6. **CAP 形成的 placement spectrum 支持自动适配**，结合轻量 profiling 即可选出最优配置。

---

### ⚠️ 局限性
1. **依赖离线路由 trace 收集**：需要先运行一定量请求获取 co-activation 统计，可能增加部署前开销。
2. **假设 workload 稳定性**：虽然验证了 in-workload 和 cross-workload 的 grouping transferability 较好，但在极端分布漂移下可能失效。
3. **未覆盖所有并行策略组合**：主要基于 Expert Parallelism + Data Parallelism，未深入研究 Tensor Parallelism 耦合情况。

---

### 🔮 未来工作方向
1. **在线自适应调整**：动态监测 workload 变化，实时更新 expert grouping。
2. **与 TP/Tensor Parallelism 深度集成**：探索更复杂的 hybrid parallelism 下的联合优化。
3. **扩展至训练场景**：将 communication-aware 思路应用于 MoE 训练阶段的 load balancing。
4. **自动化 $ c $ 参数调优**：根据 runtime profiling 自动估计 inter/intra-node 通信比值。

---

## ✅ 总结
**CAP 是首个将 communication volume 显式纳入 MoE 推理优化框架的工作**，通过 co-activation-driven placement、comm-comp trade-off adjustment 和 communication-aware pruning 三步 pipeline，在多种硬件平台上实现了 **1.23×～1.86× 的吞吐提升**，并在相同加速目标下 **保留更好的模型 accuracy**。该工作揭示了未来 MoE 系统优化必须从“纯计算视角”转向“通信-计算-精度”三位一体的设计范式。

</details>

---

### 3. [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)

**Authors**: Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, Yi Qian, Jiaqi Zhu, Shirong Ma, Xiaokang Zhang, Jiasheng Ye, Qinyu Chen, Chengqi Deng, Jiping Yu, Damai Dai, Zhengyan Zhang, Yixuan Wei, Yixuan Tan, Wenkai Yang, Runxin Xu, Yu Wu, Zhean Xu, Xuanyu Wang, Muyang Chen, Rui Tian, Xiao Bi, Zhewen Hao, Shaoyuan Chen, Huanqi Cao, Wentao Zhang, Anyi Xu, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang  
**Category**: cs.AI  
**Published**: 2026-07-07  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2607.05147v1  

#### Abstract
Speculative decoding accelerates Large Language Model (LLM) inference by decoupling draft generation from target verification. While recent parallel drafters efficiently propose long token sequences in a single forward pass, they suffer from rapid acceptance decay due to a lack of inter-token depend...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前 **Speculative Decoding** 在加速大语言模型（LLM）推理时面临两大瓶颈：

1. **生成质量瓶颈**：并行 drafter（如 DFlash）虽然能快速生成长 token 序列，但由于缺乏 **token 间依赖建模**，导致后缀位置接受率迅速衰减（suffix decay），影响整体效率。
2. **系统效率瓶颈**：盲目验证长 draft block 会浪费目标模型的计算资源，尤其在高并发场景下，低接受率的后缀 token 占用宝贵的 batch 容量，严重降低吞吐量。

---

### 提出的新方法与新思路

DSpark 提出了一种统一 **高吞吐并行生成** 与 **自适应、负载感知验证** 的框架，包含两个核心机制：

#### ✅ **Semi-Autoregressive Generation（半自回归生成）**

- **架构设计**：采用“并行主干 + 轻量级串行头”的混合结构。
  - **并行主干**（如 DFlash）负责高效生成整个 block 的初始 logits。
  - **轻量级串行模块**（Markov Head 或 RNN Head）引入局部 token 依赖，提升后缀一致性。
- **优势**：保留了并行生成的低延迟特性，同时缓解了独立预测带来的多模态冲突（multi-modal collision）和后缀衰减。

#### ✅ **Confidence-Scheduled Verification（置信度调度验证）**

- **置信度头（Confidence Head）**：为每个 draft token 输出一个标量 $ c_k \in (0,1) $，表示该 token 在前缀被接受的前提下存活的概率。
- **硬件感知前缀调度器（Hardware-Aware Prefix Scheduler）**：
  - 动态决定每个请求的验证长度。
  - 综合考虑 **token 的生存概率** 和 **当前系统负载下的实际吞吐曲线 SPS(B)**，最大化全局 token 吞吐量 $ \Phi = t^* \cdot \text{SPS}(B) $。
- **优势**：避免在高负载时验证低置信度 token，防止资源浪费；在低负载时充分利用空闲算力。

---

### 相比现有方法的优势

| 对比维度 | 传统 Autoregressive Drafter（如 Eagle3） | 并行 Drafter（如 DFlash） | **DSpark（本文）** |
|----------|----------------------------------------|--------------------------|--------------------|
| 生成延迟 | 高（$ T_{\text{draft}} \propto y $） | 极低（单次前向传播） | 极低（仅增加轻微串行开销） |
| 接受率稳定性 | 高（有依赖建模） | 初始高，后缀衰减快 | 初始高，后缀稳定 |
| 验证策略 | 固定长度或静态阈值 | 固定长度 | **动态、负载感知调度** |
| 系统效率 | 受限于短 block | 易因盲目验证而浪费资源 | **高并发下仍保持高吞吐** |

---

## 2. 核心实验方法和设置

### 使用的数据集

训练与评估基于 **Open-PerfectBlend** 数据集（开源版 PerfectBlend），包含：
- **Chat**（17.6%）
- **Math**（39.4%）：GSM8K, MATH500, AIME25
- **Code**（38.9%）：MBPP, HumanEval, Live-CodeBench
- **Instruction Following**（4.1%）

---

### 实验设置与评估指标

#### ✅ 目标模型（Target Models）
- Qwen3 系列：4B, 8B, 14B
- Gemma4-12B

#### ✅ Draft 模型对比
- **Eagle3**：自回归 drafter（基于 TTT）
- **DFlash**：并行 drafter（state-of-the-art）
- **DSpark**：本文提出（Markov Head 默认，RNN Head 可选）

#### ✅ 评估协议
- 使用标准 Speculative Decoding 流程（Chen et al., 2023）
- 温度采样设为 1.0
- 主要指标：**每轮解码的平均接受长度（accepted length $ t $）**
  > 注：包含最终由 target model 生成的 bonus token

#### ✅ 训练细节
- 所有 draft model 共享 target model 的 embedding 层和 LM head（冻结）
- 块大小（block size）固定为 7
- 训练目标包含三项加权损失：
  - $ L_{\text{ce}} $：交叉熵损失
  - $ L_{\text{tv}} $：总变差距离（Total Variation Distance），直接优化接受率
  - $ L_{\text{conf}} $：置信度二元交叉熵损失

---

## 3. 主要实验结果和性能指标

### 关键性能数据（离线基准测试）

在多个目标模型上，DSpark 显著优于基线：

| Target Model | vs. Eagle3（提升） | vs. DFlash（提升） |
|--------------|---------------------|----------------------|
| Qwen3-4B     | **+30.9%**          | **+16.3%**           |
| Qwen3-8B     | **+26.7%**          | **+18.4%**           |
| Qwen3-14B    | **+30.0%**          | **+18.3%**           |
| Gemma4-12B   | **+25.8%**（平均）  | **+16.1%**（平均）   |

> 示例：在 Qwen3-4B 上，数学任务平均接受长度从 DFlash 的 ~5.0 提升至 **6.1**。

---

### 与基线方法的对比结果

#### ✅ 位置级接受率分析（Figure 2）
- **DFlash**：首位置接受率高（得益于更深网络），但后续迅速衰减 → “前高后低”
- **Eagle3**：起始较低（浅层限制），但后期稳定或上升
- **DSpark**：兼具两者优点 —— **首位置高接受率 + 后续稳定不衰减**

#### ✅ 不同 proposal length 下的表现（Figure 4）
- 随着 block size 增大（4→16），DSpark 相对 DFlash 的优势持续扩大。
- 在 $ y=15 $ 时，提升可达：
  - 数学：**+30%**
  - 代码：**+26%**
  - 聊天：**+22%**

#### ✅ 延迟开销极小（Figure 4 右图）
- 引入串行 head 后，每轮延迟仅增加 **0.2% ~ 1.3%**（batch=128）
- 实现了“几乎零代价”的依赖建模增强

---

### 消融实验结果

#### ✅ 不同 Sequential Head 的比较
- **Markov Head** 已足够有效，在大多数场景下接近 RNN Head 性能
- RNN Head 仅在更长序列上有微弱优势（+1~2%）
- 最终选择 **Markov Head** 作为默认配置（部署友好）

#### ✅ 置信度头的有效性（Figure 5）
- 通过设定置信度阈值可显著提升整体接受率：
  - **聊天任务**：从 45.7% → **95.7%**
  - **代码任务**：从 67.6% → **92.0%**
- 表明置信度头能有效识别易被拒绝的低价值 token

#### ✅ STS 校准的重要性（Figure 6）
- 原始置信度估计存在过自信问题（ECE: 3–8%）
- 使用 **Sequential Temperature Scaling (STS)** 后，ECE 降至 ~1%，实现可靠校准
- 对动态调度至关重要（需准确估计累积生存概率）

---

## 4. 关键结论和发现

### 主要发现

1. **并行生成可以超越自回归**：
   - 尽管违反直觉，但并行 drafter 凭借更强的架构容量（更深网络）在首 token 上取得压倒性优势，足以弥补后缀衰减。
   - DSpark 进一步结合了两者的优点。

2. **“少量自回归”即可带来巨大收益**：
   - 仅添加一个轻量级 Markov Head，就能显著缓解后缀衰减，且延迟代价极小。
   - 参数效率远高于堆叠更多并行层。

3. **动态验证调度是系统级优化的关键**：
   - 固定长度验证在不同领域表现差异大（结构化任务 vs 开放对话）
   - 硬件感知调度器可根据实时负载自动调整验证预算，实现“轻载多验、重载精验”。

4. **真实部署中性能跃迁明显**：
   - 在 DeepSeek-V4 生产系统中，相比原 MTP-1 基线：
     - **V4-Flash**：用户生成速度提升 **60%–85%**
     - **V4-Pro**：提升 **57%–78%**
   - 更重要的是，在严格 SLA（如 120 TPS）下，DSpark 能维持可用吞吐，而基线已崩溃。

---

### 方法的局限性

- **固定 draft 成本**：无论接受率高低，都需要完整执行一次并行 backbone 前向传播，对于天生难预测的 query 存在不可回收的成本。
- **未支持动态 early exiting**：未来可在 draft model 内部引入难度感知机制，在确认低接受潜力时提前终止生成。
- **调度异步近似**：为兼容 ZOS 和 CUDA graph，使用了两步前的历史预测来决定截断长度，存在轻微时间偏移。

---

### 未来工作方向

1. **Difficulty-Aware Drafting**：让 draft model 自主判断是否值得生成完整 block。
2. **更细粒度的调度控制**：探索 per-layer 或 per-head 的动态激活机制。
3. **扩展到多模态 LLM**：将 semi-autoregressive 与 confidence scheduling 应用于图像、音频生成。
4. **开放生态建设**：作者已开源 DSpark checkpoints 与 **DeepSpec** 训练库，鼓励社区共建高效 LLM 推理基础设施。

---

> 🔥 **一句话总结**：  
> DSpark 通过 **semi-autoregressive generation** 提升 draft 质量，再通过 **confidence-scheduled verification** 实现智能验证，在算法层面和系统层面双重突破，成功将 LLM 推理的 **吞吐-交互性帕累托前沿向外推移**，实现了生产环境中前所未有的性能平衡。

</details>

---

### 4. [AdaptiveSD A Stability-Aware, Runtime-Adaptive Speculative Decoding Framework with Multi-Policy Orchestration for CPU-Constrained LLM Inference](https://arxiv.org/abs/2607.03876)

**Authors**: Sadra Saremi  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.03876v1  

#### Abstract
With the rise of small quantized GGUF-based language models and their increasing use for on-device inference tasks, we have seen the growing need for an approach capable of reliably delivering these models at scale even under severe memory bandwidth constraints such as those imposed by pure CPU impl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AdaptiveSD: A Stability-Aware, Runtime-Adaptive Speculative Decoding Framework with Multi-Policy Orchestration for CPU-Constrained LLM Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **Speculative Decoding** 技术通常采用固定的 draft depth（如固定为2），这在资源受限、动态变化的 **CPU-constrained** 推理环境中存在严重缺陷：
- 固定深度无法适应不同 workload（如代码生成、推理、聊天）带来的 acceptance rate 波动；
- 在内存带宽受限的 CPU 上，过深的 draft 会导致 **bandwidth saturation**、**CPU 利用率过高** 或 **tail latency 激增**，甚至引发系统崩溃；
- 现有自适应方法大多只关注吞吐量优化，忽视了 **稳定性** 和 **资源安全边界**。

### 提出的新方法
作者提出 **AdaptiveSD**，一个完全运行时自适应的 speculative decoding 框架，专为 **CPU 受限环境下的量化 GGUF 模型** 设计。其核心是一个闭环控制架构，包含四个紧密耦合的组件：

| 组件 | 功能 |
|------|------|
| **Runtime Monitoring Engine** | 实时监控 TPS、inter-token latency（P95/P99）、CPU 利用率、memory bandwidth、token entropy、context length 等多维信号 |
| **Adaptive Draft Controller** | 基于 **11 条规则的优先级策略层级**，动态调整 draft depth（1–4），优先保障系统稳定而非最大吞吐 |
| **Dynamic Policy Engine** | 融合三种子策略：<br>- **基于 workload 的启发式规则**<br>- **UCB 多臂老虎机（Multi-Armed Bandit）**<br>- **EMA 奖励追踪器**<br>通过置信度加权集成进行决策 |
| **KV Cache Coordination Layer** | 使用 **INT8 量化 shadow buffer** 和 **位置感知驱逐机制**，减少 KV cache 内存压力 |

### 相比现有方法的优势
| 方面 | AdaptiveSD | 现有方法（如 SpecDec++, Parallel SD） |
|------|-----------|-----------------------------|
| **适应维度** | 同时响应 CPU、memory bandwidth、latency、entropy、context length 等 | 通常仅基于 acceptance rate 或 entropy |
| **目标导向** | 强调 **稳定性、资源效率、避免崩溃** | 主要追求 **吞吐量最大化** |
| **部署适用性** | 无需离线调优，适用于边缘设备 | 需要 GPU 或离线 profiling |
| **安全性机制** | 显式设计了 CPU 饱和切断、acceptance 崩溃保护、振荡抑制、latency 方差 spike 规则等 | 缺乏系统级安全保障 |

---

## 2. 核心实验方法和设置

### 数据集与 Workload
未使用传统 NLP 数据集，而是构建了四类典型 prompt 类型用于测试：
- **Coding**（代码生成）
- **Reasoning**（数学/逻辑推理）
- **Chat**（对话）
- **Creative**（创意写作）

每类 prompt 构成 benchmark，生成固定 token 数量（如 128 tokens × 3 prompts）。

### 实验设置
- **硬件平台**：
  - 主要测试平台：`HP EliteBook 850 G4`，搭载 `Intel Core i5-7300U @ 2.60GHz`（双核四线程），24GB DDR4 内存
  - 该平台代表典型的 **低端边缘 CPU 设备**
- **模型配置**：
  - **Target Model**: Qwen3.5-2B（UD-Q4_K_XL 量化）
  - **Draft Model**: Qwen3.5-0.8B（Q4_0 量化）
  - 上下文长度：`n_ctx=4096`
- **后端实现**：
  - 使用 `llama-cpp-python` 的 GGUF backend
  - 同时开发了一个高保真 **statistical simulation backend** 用于快速迭代

### 评估指标
作者提出两个新的核心评估指标，替代传统的 throughput speedup：

| 指标 | 定义 | 意义 |
|------|------|------|
| **Wasted Compute Fraction (WCF)** | $ \text{WCF} = \frac{N_{\text{wasted}}}{N_{\text{committed}} + N_{\text{wasted}}} $ | 衡量被拒绝的 draft token 占比，反映内存带宽浪费程度 |
| **Latency Coefficient of Variation (CV_ITL)** | $ \text{CV\_ITL} = \frac{\sigma_{\text{ITL}}}{\mu_{\text{ITL}}} $ | 衡量 inter-token latency 的波动性，反映系统稳定性 |
| **Speculative Efficiency (η)** | $ \eta = \frac{N_{\text{committed}}}{N_{\text{committed}} + N_{\text{wasted}}} $ | 成功提交 token 的比例，理想值接近 1 |

此外还报告：
- TPS（tokens per second）
- Speedup ratio（vs. non-speculative baseline）
- P95/P99 ITL
- TTFT（Time-to-First-Token）
- KV cache hit rate 与压缩比

### 基线方法对比
- **Fixed-depth-2**：非自适应 baseline
- **Acceptance-only heuristic**：仅基于 acceptance rate 调整 depth
- **Parallel Speculative Decoding with Adaptive Draft Length [13]**
- **SpecDec++ [12]**（近似实现）
- **Dynamic Speculation Lookahead [11]**（近似实现）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（GGUF Backend）
| 指标 | 结果范围 |
|------|--------|
| **Speculative Efficiency (η)** | **78.2% – 81.6%** |
| **Wasted Compute Fraction (WCF)** | **18.4% – 21.8%** |
| **CV_ITL** | **0.84 – 0.97**（远低于 1.5 安全阈值） |
| **Speedup Ratio** | **0.375x – 0.431x**（低于1，因 CPU 带宽瓶颈） |
| **P95/Mean Tail-Latency Ratio** | **2.57x – 3.09x** |
| **INT8 KV Cache 压缩比** | **~3.7x** |
| **Shadow Buffer Hit Rate** | **100%**

> ✅ 尽管绝对速度提升为负，但 **资源利用效率极高且系统极其稳定**

### 与基线方法对比（Table 4, n=15 runs）
| 方法 | Speedup | WCF | CV_ITL | AR-Fallback |
|------|--------|-----|-------|------------|
| **specdec_plus_approx** | **0.771x**（最高） | **15.3%**（最差） | 0.758 | **13.7%**（最低） |
| **fixed_depth=2** | 0.664x | 10.8% | 0.747 | 46.3% |
| **AdaptiveSD (ensemble)** | 0.653x | **10.3%** | **0.642**（最优） | 37.3% |
| **AdaptiveSD (bandit)** | 0.642x | **9.5%** | 0.696 | 43.8% |
| **AdaptiveSD (heuristic)** | 0.613x | **8.8%** | 0.819 | 53.3% |
| **AdaptiveSD (ema)** | 0.535x | **8.0%** | **0.933**（最差） | **72.3%**（最高） |

#### 关键观察：
- **specdec_plus_approx** 虽然速度最快，但 **WCF 最高（15.3%）**，说明其通过 **过度 drafting** 换取速度，资源浪费严重；
- **AdaptiveSD ensemble** 在速度上略低于 fixed-depth，但在 **WCF 和 CV_ITL 上表现更优**，体现了其 **资源效率优先** 的设计理念；
- **EMA 子策略表现最差**，因其收敛慢，在短会话中付出高昂探索代价。

### 消融实验结果（Simulation Backend, Table 3）
在短会话（96 tokens）下比较三种控制器：
| 控制器 | Speedup | WCF | CV_ITL | Avg Depth |
|--------|--------|-----|--------|----------|
| **static_depth2** | 1.279x | 0.333 | 0.147 | 2.00 |
| **acceptance_only** | 1.279x | 0.333 | 0.148 | 2.00 |
| **AdaptiveSD_full** | **1.209x** | **0.398** | **0.205** | **2.42** |

> ❗ AdaptiveSD 在短会话中表现更差，原因是 **探索成本未摊销**。但在长会话或真实 GGUF 环境中，其优势显现。

---

## 4. 关键结论和发现

### 主要发现
1. 🔹 **AdaptiveSD 的核心价值不是提速，而是稳定**  
   在 CPU 受限环境下，**避免系统崩溃、控制尾延迟、最小化资源浪费** 比单纯提高吞吐更重要。AdaptiveSD 成功将 WCF 控制在 ~20%，CV_ITL < 1.0，实现了高稳定性运行。

2. 🔹 **固定深度策略在异构边缘设备上风险极高**  
   一个在某设备上调优的 fixed-depth=2 策略，在另一台设备上可能导致 CPU 饱和或内存溢出。AdaptiveSD 的 **运行时反馈机制** 是应对硬件多样性的必要设计。

3. 🔹 **多策略融合优于单一策略**  
   单独使用 bandit 或 EMA 收敛慢、不稳定；而 **heuristic + bandit + EMA 的置信加权集成** 能兼顾响应速度与长期优化。

4. 🔹 **INT8 KV Cache 量化有效降低内存压力**  
   实现 ~3.7x 压缩比，shadow buffer hit rate 达 100%，验证了轻量级缓存协调机制的有效性。

### 局限性
| 限制 | 描述 |
|------|------|
| **硬件依赖性强** | 所有实验均在老旧双核 CPU 上完成，结果可能不具普适性 |
| **缺乏正式敏感性分析** | 11 条规则的阈值（如 `cpu_critical=0.95`, `cv_itl_threshold=1.5`）是经验设定，未进行网格搜索 |
| **未支持并发请求** | 当前为单会话设计，未考虑多租户竞争场景 |
| **深度上限为4** | 未对 depth > 4 进行正式消融实验，仅凭经验排除 |
| **未复现完整 SOTA 方法** | 对比的是 rule-based approximation，非原始 [11][12][13] 的完整实现 |

### 未来工作方向
1. **在现代硬件上验证性能**：在多核、DDR5 内存的桌面/笔记本 CPU 上重复实验，预期 speedup 可达或超过 1.0x；
2. **形式化阈值敏感性分析**：系统性地扫描各规则阈值的影响；
3. **支持并发与多租户**：扩展 Runtime Monitor 至 per-request 粒度；
4. **深度范围消融实验**：正式评估 depth=5~8 的收益与成本；
5. **集成到主流框架**：作为插件集成至 `llama.cpp` 或 HuggingFace Transformers；
6. **根因分析尾延迟**：通过 CPU 绑核 + perf tracing 区分 bandwidth contention 与 OS scheduling jitter。

---

> 📌 **总结一句话**：  
> **AdaptiveSD 不是最快的 speculative decoder，但它是最稳的——它用一点速度换取了在任何边缘设备上都不会崩溃的可靠性，这才是 CPU 推理真正需要的。**

</details>

---

### 5. [Nemotron-Labs-3-Puzzle-75B-A9B: Compressing Hybrid MoE LLMs](https://arxiv.org/abs/2607.04371)

**Authors**: Akhiad Bercovich, Talor Abramovich, Daniel Afrimi, Shay Aharon, Nir Ailon, Vladimir Anisimov, Omer Ullman Argov, Maor Ashkenazi, Tomer Asida, Nave Assaf, Tomer Bar Natan, Alexander Bukharin, Grzegorz Chlebus, Marcin Chochowski, Eric Chung, Mohammad Dabbah, Carlo del Mundo, Ewa Dobrowolska, Ido Galil, Yaniv Galron, Amnon Geifman, Yonatan Geifman, Izik Golan, Alex Gronskiy, Tomasz Grzegorzek, Netanel Haber, Lior Kadoch, Grzegorz Karch, Tomer Keren, Abhinav Khattar, Amir Klein, Tugrul Konuk, Roi Koren, Daniel Korzekwa, Shaun Kotek, Konstantinos Krommydas, Itay Levy, Ofri Masad, Yoav Miron, Pavlo Molchanov, Shahar Mor, Zach Moshe, Saurav Muralidharan, Najeeb Nabwani, Besmira Nushi, Mostofa Patwary, Omri Puny, Johannes Rausch, Tomer Ronen, Sepehr Sameni, Itamar Schen, Elad Segal, Daniel Serebrenik, Ido Shahaf, Soumye Singhal, Daniil Sorokin, Sharath Turuvekere Sreenivas, Marta Stepniewska-Dziubinska, Ali Taghibakhshi, Nima Tajbakhsh, Oren Tropp, Dor Tzur, Anna Warno, Yi-Fu Wu, Michal Zawalski, Jiaqi Zeng, Yian Zhang, Ran Zilberstein, Amit Zuker, Ran El-Yaniv  
**Category**: cs.AI  
**Published**: 2026-07-07  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.04371v1  

#### Abstract
We present Nemotron-Labs-3-Puzzle-75B-A9B, a compressed variant of Nemotron-3-Super optimized for interactive deployment. We designed the model to maximize server throughput under high user throughput constraints. In interactive serving workloads on a single 8xB200 node, Puzzle-75B-A9B achieves appr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Nemotron-Labs-3-Puzzle-75B-A9B: Compressing Hybrid MoE LLMs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大规模语言模型（LLMs），尤其是混合架构的 **Hybrid MoE** 模型（如结合 Mamba、Attention 和 MoE 层），在推理时面临高计算成本、内存占用大、延迟高以及部署灵活性差等问题。这些限制在生产环境中尤为突出，特别是在需要满足严格吞吐量（throughput）、低延迟和超长上下文（long-context）的场景下。

本文旨在解决如何在**不显著牺牲模型能力的前提下，大幅压缩 Hybrid MoE 架构的大模型以提升推理效率**，实现高效部署。

---

### 提出的新方法与创新思路

作者提出了 **Nemotron-Labs-3-Puzzle-75B-A9B** —— 一个基于 **Nemotron-3-Super** 的压缩优化版本，并引入了一套多阶段联合优化框架：

#### （1）Iterative Puzzle：迭代式硬件感知架构搜索
- 是对原始 **Puzzle** 框架的扩展，采用**分阶段压缩 + 短期恢复训练**的方式。
- 不同于一次性完成压缩（single-shot），而是通过多个轮次交替进行：
  - **结构化压缩**（structural pruning）
  - **知识蒸馏**（Knowledge Distillation, KD）恢复
- 在每一轮中重新评估模块质量分数，使压缩过程能适应中间表示的变化，缓解层间依赖带来的误差累积。

#### （2）多维度异构剪枝（Heterogeneous Pruning）
联合优化以下三个关键维度：
- **MoE 中间通道剪枝**（Intermediate Channel Pruning）：减少每个专家内部的激活参数。
- **Top-k 减少**：降低每个 token 路由到的专家数量（从 22 → 平均 11）。
- **Mamba SSM State 剪枝**：将 SSM state size 从 128 降至 96，显著降低 decode 阶段的 cache IO 开销。

> ✅ 所有剪枝策略均为**非均匀分布**（heterogeneous），即不同网络层分配不同容量，依据其敏感度动态调整。

#### （3）端到端恢复流程
- **大规模 KD**：先用 32K 上下文蒸馏，再逐步扩展至 512K，恢复长上下文能力。
- **强化学习微调**（RL Fine-tuning）：聚焦软件工程任务（SWE-RL），恢复对压缩敏感的能力。
- **Checkpoint Averaging**：多学习率运行后加权平均，提高稳定性。

#### （4）部署级优化集成
- **Quantization**：支持 FP8 和 NVFP4 量化，适配 Hopper / Blackwell GPU。
- **Multi-Token Prediction (MTP)**：继承并改进原生 MTP 头部，增强 speculative decoding 效果。
- **Prefill-Decoding Disaggregation**：探索前缀处理与生成解耦的轻量 prefill 模型设计。

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **压缩方式** | 迭代式压缩优于单步压缩，在相同压缩率下保持更高准确率（+0.57 pts） |
| **架构灵活性** | 支持异构剪枝，保留关键层容量，避免“一刀切”损失 |
| **效率增益来源多样** | 结合剪枝、KD、RL、量化、MTP 等多种技术，实现系统级优化 |
| **实际部署表现** | 显著提升服务器吞吐（server throughput），支持更高并发请求 |

---

## 2. 核心实验方法和设置

### 使用的数据集与评估基准

模型在多个领域进行全面评估，涵盖：

| 类别 | 数据集 |
|------|--------|
| **通用知识** | MMLU-Pro, MMLU-ProX |
| **推理能力** | AIME25, HMMT, GPQA, HLE |
| **编程能力** | LiveCodeBench, SciCode |
| **长上下文理解** | RULER @ 256K / 512K / 1M tokens |
| **代理行为（Agentic）** | Terminal Bench, SWE-Bench, TauBench V2 |
| **指令跟随** | IFBench, Arena-Hard-V2 |
| **多语言** | WMT24++ |
| **MTP 质量测试** | SPEED-Bench（用于测量 Acceptance Length） |

---

### 实验设置与评估指标

#### 主要部署场景设定：
1. **交互式服务**（Interactive Serving）
   - 硬件：单节点 8×B200 GPU
   - 场景：`50K input / 2K output`（prefill-heavy）、`8K / 64K`（decode-heavy）
   - 约束条件：用户吞吐 ≥100 TPS（tokens per second per user）

2. **超长上下文部署**
   - 硬件：单张 H100 GPU
   - 上下文长度：1M tokens
   - 目标：提升最大可持续并发请求数

#### 评估指标：
- **Server Throughput (TPS)**：总输出 token 数/秒
- **User Throughput (UT)**：每位用户的生成速度（目标 ≥100 TPS）
- **Relative Request Completion Rate**：考虑生成长度（verbosity）后的有效请求完成率
- **Benchmark Accuracy (%)**：各任务上的零样本或少样本准确率
- **Acceptance Length (AL)**：MTP 的平均每步接受 token 数

#### 基线对比模型：
- **Nemotron-3-Super**（父模型，120.7B total / 12.8B active params）
- **Nemotron-3-Nano**（小型基线，30B total / 3.5B active params）
- **Single-step Puzzle**（非迭代压缩基线）

所有模型均在相同量化级别（NVFP4 或 FP8）下比较，确保公平性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 数值 |
|------|------|
| 总参数压缩比 | 120.7B → **75.3B**（**62.4%**） |
| 激活参数压缩比 | 12.8B → **9.3B**（**73.1%**） |
| Mamba SSM State | 128 → **96**（75%） |
| Top-k（平均） | 22 → **11**（50%） |
| MTP Acceptance Length（平均） | 3.45 → **4.34**（↑25–30%） |

---

### 与基线方法的对比结果

#### （1）推理吞吐提升（8×B200 节点）

| 场景 | 用户吞吐要求 | Super TPS | Puzzle-75B-A9B TPS | 提升倍数 |
|------|--------------|-----------|---------------------|----------|
| `8K/64K` | ≥100 TPS | 20,939 | **42,601** | **2.03×** |
| `8K/64K` | ≥150 TPS | 8,522 | **18,047** | **2.12×** |
| `50K/2K` | ≥100 TPS | 5,128 | **8,210** | **1.60×** |

> ✅ 达成设计目标：在匹配用户体验条件下，实现约 **2× server throughput 提升**

#### （2）MTP 加速效果

启用 MTP 后进一步放大优势：
- Super + MTP：吞吐提升至 **3.04×**
- Puzzle-75B-A9B + MTP：吞吐达 **4.63×**（相对 Super）

> ⚠️ 注意：即使 Nano 拥有更高的 raw throughput（5.84×），但由于其生成更 verbose（verbosity=1.6），实际请求完成率仅为 **3.87×**，低于 Puzzle-75B-A9B+MTP 的 **4.91×**

#### （3）超长上下文并发能力（1M tokens on H100）

| 模型 | 最大并发请求数 | KV Cache 容量利用 |
|------|----------------|--------------------|
| Nemotron-3-Super | **1 请求** | 接近饱和（~74GB） |
| Puzzle-75B-A9B | **8 请求** | 利用剩余 HBM 空间容纳额外 7 个请求 |

> 💡 原因：模型权重从 ~70GB（Super）压缩至 ~44.5GB（Puzzle），释放足够显存支持更多 KV cache

---

### 消融实验结果

#### （1）Iterative Puzzle vs Single-step Puzzle

| 方法 | 平均准确率 | 相对提升 |
|------|------------|----------|
| Single-step Puzzle | 68.48 | — |
| 3-step Iterative Puzzle | **69.05** | **+0.57 pts** |

> ✅ 表明迭代式压缩能更好建模层间依赖，获得更优最终架构

#### （2）训练阶段影响分析（Training Progression）

- **短上下文 KD**：快速恢复基础准确率至 Super 的 97%+
- **长上下文 KD**：显著提升 RULER 等任务表现
- **RL 微调**：对 SWE 类任务有帮助，但整体影响较小（未来可加强）

#### （3）Prefill-Decoding Disaggregation 消融

构建两个轻量 prefill 模型（FastPuzzleV1/V2）：
- 仅用于 prefill，decode 仍用完整 Puzzle-75B-A9B
- 结果：prefill 吞吐提升 **5–7%**，且精度几乎无损
- 若同时用于 decode，则精度大幅下降（↓5–7 pts）

> ✅ 支持“prefill 可压缩，decode 应保真”的分离式部署理念

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **大型 Hybrid MoE 模型可以被深度压缩而不严重损失性能**
   - 尽管总参数减少 38%，激活参数减少 27%，但在多数任务上仍保持接近 Super 的准确率（如 MMLU-Pro: 82.4 vs 83.8；RULER-1M: 92.2 vs 93.9）

2. ✅ **Iterative Puzzle 显著优于单步压缩**
   - 通过多次 score recomputation 和中间恢复，实现了更稳定的压缩路径

3. ✅ **异构剪枝优于均匀缩放**
   - 图 2 显示，Puzzle 自动选择在中后层保留更多 MoE 容量，说明层重要性存在差异

4. ✅ **部署效率需综合考量 token throughput 与 request-level performance**
   - 单纯看 TPS 不够，必须结合 **generation verbosity** 分析真实请求完成率

5. ✅ **MTP 与量化兼容良好，且可通过继续训练提升鲁棒性**
   - Puzzle-75B-A9B 的 MTP 在 NVFP4 下性能衰减远小于 Super

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **Mamba 层剪枝受限** | 当前推理框架不支持 per-layer SSM state size 配置，只能全局统一剪枝 |
| **Latent Dimension Pruning 未采用** | 因为当前 NVFP4 kernel 要求 latent dim 为 512 的倍数，唯一可行跳变（1024→512）过于激进 |
| **RL 恢复效果有限** | 当前 RL 阶段未能完全弥补压缩带来的 agentic 能力下降（如 Arena-Hard-V2 差距较大） |
| **Prefill-Decoding 兼容性挑战** | 需要 joint training 来保证状态一致性，增加训练复杂度 |

---

### 未来工作方向

1. **更激进的 Iterative Puzzle 压缩**
   - 探索更高压缩率下的极限性能边界
2. **Per-layer Mamba SSM 支持**
   - 推动底层框架升级，实现细粒度控制
3. **引入更多 RL 环境**
   - 特别是针对工具使用、复杂规划等 agentic 场景
4. **探索弹性架构（Elastic Architecture）**
   - 如 Star Elastic 训练，支持运行时动态资源调配
5. **极致压缩路线图**
   - 目标是实现“Extreme LLM Compression”，在强约束下维持可用能力

---

> 🔚 **总结一句话**：  
> 本文展示了通过 **Iterative Puzzle + 异构剪枝 + 多阶段恢复 + 部署优化** 的组合拳，可在基本不损失能力的前提下，将 Hybrid MoE 大模型压缩至原规模的 60% 以下，并实现 **2× 以上吞吐提升** 和 **8倍超长上下文并发能力**，为下一代 LLM 的高效部署提供了实用路径。

</details>

---

### 6. [Heaviside Continuity of Rolling Coefficients for Eliminating Epistemic Entropy in Large Language Models](https://arxiv.org/abs/2607.04562)

**Authors**: MY Pitsane, Hope Mogale  
**Category**: cs.AI  
**Published**: 2026-07-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.04562v1  

#### Abstract
Large language models (LLMs) generate fluent outputs that can be wrong. Unlike humans, who often exhibit cues when providing false information, LLMs produce errors that are difficult to detect because autoregressive decoding provides no mechanism for verifying intermediate reasoning before state pro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Heaviside Continuity of Rolling Coefficients for Eliminating Epistemic Entropy in Large Language Models*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在生成流畅文本的同时，常出现“**epistemic errors**”（认知错误），即输出看似合理但与外部世界状态不符的内容，这种现象通常被称为“**hallucination**”。传统方法难以在推理过程中实时拦截这些错误，因为自回归解码缺乏对中间推理步骤的验证机制。

本文关注的是**LLM驱动外部系统**（如代码仓库、数据库）时的控制问题，其中“幻觉”表现为模型声称的状态（claim state）与真实系统状态（reference state）之间的偏差，作者将此偏差定义为“**epistemic entropy**”。

### 提出了什么新方法或新思路
提出了一种名为 **Heaviside Continuity of Rolling Coefficients (HCRC)** 的执行框架，其核心思想是将推理过程重构为一系列由谓词（predicate）控制的状态转移，并通过一个**Heaviside Gate**（阶跃门）来决定是否推进到下一步。

- **HCRC 框架**：引入了一个执行层的控制机制，只有当预定义的正确性谓词被满足时，才允许状态推进。
- **ARIAL Worker Pool**：并行运行多个非LLM检查器（non-LLM checkers），从不同证据通道（如文件系统、语法解析、测试运行）提取独立的验证信号，并聚合为单一的验证分数 $ V $。
- **Heaviside Gate**：决策基于 $ H(C \cdot V - T) $，其中 $ C $ 是模型置信度，$ V $ 是验证分数，$ T $ 是阈值。仅当 $ C \cdot V \geq T $ 时才允许推进（ADVANCE），否则暂停（HALT）并反馈失败谓词。

### 相比现有方法的优势
| 对比维度 | 现有方法 | HCRC |
|--------|--------|------|
| **验证时机** | 事后验证（post-hoc）或训练时监督 | **强制性、每步验证**，与执行强绑定 |
| **验证主体** | 使用另一个LLM作为judge（易偏倚） | **非LLM检查器**（filesystem, AST, tests），更可靠 |
| **控制机制** | 软性打分、重排序 | **硬性执行屏障**（hard execution barrier），无部分提交 |
| **错误处理** | 错误可能传播至下游 | 将残余错误转化为**诚实暂停**（honest halts），暴露给操作员 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **合成任务集**：由 `experiments/tasks.py` 生成的50个软件工程任务，涵盖五类：
  - FLASK scaffold
  - FASTAPI scaffold
  - Python argparse CLI
  - data-transform script
  - sqlite3 migration
- 每个任务包含 4–8 个可见谓词（如 `file_exists`, `parses`, `file_contains`）和 **1个隐藏谓词**（`command_succeeds`，例如导入模块并断言对象存在）。
- **SWE-bench Lite**：用于初步验证，加载前10个实例，仅使用 `file_exists` 和 `parses` 作为可见谓词（未克隆完整仓库，故禁用 `test_passes`）。

### 实验设置和评估指标
#### 模型与提供方
共测试 **13个 proposer 模型**，来自四个提供方：
- **Groq**: Llama-3.3-70B, Llama-3.1-8B
- **OpenAI**: GPT-3.5-turbo, GPT-4系列, GPT-5.4/5.5
- **Anthropic**: Claude Haiku, Sonnet, Opus
- **OpenRouter**: GPT-OSS-120B（开源权重）

#### 实验条件对比
- **Unwrapped LLM**：原始模型，依赖自我报告的成功。
- **HCRC + ARIAL**：启用HCRC框架，阈值 $ T=1 $，最大重试次数 $ T_{\text{max}}=3 $。

#### 评估指标
| 指标 | 定义 |
|-----|------|
| **FCR (False-Completion Rate)** | 运行以“成功”结束但至少一个谓词失败的比例（核心指标） |
| **V+ (Final Verification Score)** | 终止时平均验证得分（越高越好） |
| **r (Mean Retries per Committed Step)** | 每次提交的平均重试次数 |
| **Wall-clock Time (s)** | 平均每任务耗时（秒） |
| **Honest Halt** | 耗尽重试预算且 $ V < 1 $，明确上报失败谓词 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 在强模型上的表现（Groq & OpenRouter）
| Model | FCR (Unwrapped) | FCR (HCRC) | V+ | r | Wall (s) |
|-------|------------------|------------|-----|----|----------|
| `groq:llama-3.3-70b` | 7% | **0%** | 1.00 | 0.8 | 5.3 |
| `openrouter:gpt-oss-120b` | 4% | **0%** | 1.00 | 0.0 | 23.5 |
| `groq:llama-3.1-8b` | 7% | 3% | 1.00 | 1.2 | 10.0 |

> ✅ **结论**：在能力强的模型上，**HCRC将FCR降至0%**，且重试开销极小。

#### 在中等模型上的表现（OpenAI & Claude）
| Model | FCR (Unwrapped) | FCR (HCRC) | Change |
|-------|------------------|------------|--------|
| `gpt-4o` | 40% | 20% | ↓50% |
| `gpt-4o-mini` | 40% | 20% | ↓50% |
| `claude-haiku` | 20% | 20% | — |
| `claude-sonnet` | 20% | 20% | — |

> ⚠️ **说明**：残余FCR主要源于**重试预算耗尽**，而非静默错误传播；HCRC将其转化为“诚实暂停”，提升了可观测性。

### 与基线方法的对比结果
- **False Completion vs Honest Halt**：
  - 未包装模型只能以“虚假完成”失败（silent corruption）。
  - HCRC 将残余失败转化为 **honest halts**，明确暴露失败谓词，避免下游污染。
- **能力归一化效应**：
  - 不同能力模型在启用HCRC后，**残余FCR趋于一致的地板水平（0–3%）**。
  - 即使是较小的8B模型，在HCRC下也能达到接近120B模型的可靠性。

### 消融实验结果
#### （1）阈值 $ T $ 扫描（Table 10）
| $ T $ | FCR |
|-------|-----|
| 1.0   | 0% |
| 0.95  | 0% |
| 0.85  | **5%**（泄漏） |
| 0.7   | 0% |
| 0.5   | 0% |

> 🔍 **发现**：**$ T=1 $ 是唯一安全设置**。任何 $ T<1 $ 都可能导致 $ C \cdot V $ 偶然超过阈值而放行错误状态，验证了理论预测。

#### （2）Worker Pool 组成分析
在 `llama-3.3-70b` 上测试不同worker组合：
| Pool Composition | FCR | r |
|------------------|-----|----|
| Validator only | 0% | 0.03 |
| Validator + Tests | 0% | 0.3 |
| Full Pool | 0% | 0.63 |

> 🔧 **发现**：**Validator-Tests 组合承担了主要验证负载**。Syntax Guard 和 Citation Guard 在强模型上几乎不触发halt，但在弱模型或复杂任务中价值显著。

#### （3）延迟表现（Surprising Result）
| Model | Wall (Unwrapped) | Wall (HCRC) | Δ |
|-------|------------------|-------------|----|
| `llama-3.3-70b` | 11.9s | **5.3s** | ↓56% |
| `llama-3.1-8b` | 11.5s | 10.0s | ↓13% |

> ⏱️ **机制**：当首次尝试即通过验证（$ V=1 $）时，HCRC立即提交并终止，**跳过模型的冗长结尾**（epilogue）。而未包装模型必须完成整个输出流程。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Heaviside Barrier 成立**：在 $ T=1 $ 下，**所有通过门控的步骤都保证 $ V=1 $**，即所有谓词均满足（Theorem 6.3）。
2. **能力归一化**：HCRC 极大缩小了不同规模模型间的可靠性差距，**可靠性更多取决于门控而非模型本身**。
3. **延迟可降低**：在高成功率任务上，HCRC 可能比原始模型**更快**，因其支持早期终止。
4. **熵单调递减**：在HCRC控制下，**epistemic entropy $ S_e $ 随提交步骤单调下降**，最终趋近于0。
5. **强模型不易被游戏化**：即使暴露谓词列表，强模型（如Llama-3.3-70B）仍倾向于生成真正工作的实现，而非最小化通过检查的stub。
6. **生产环境可行性**：该框架已在 **Chalk**（代理编码环境）中作为生产控制平面运行数月，支撑文件变更授权、进度报告和内存压缩。

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **Predicate Completeness** | 仅能防止已定义谓词的违反；语义正确性、安全性、性能等问题需额外谓词。 |
| **Decidability Cost** | 昂贵谓词（如完整测试套件）会增加延迟；需权衡精度与吞吐量。 |
| **Predicate Gaming** | 若LLM看到谓词，可能写出“刚好通过”的代码；需依赖**隐藏测试**（hidden tests）防御。 |
| **Progress without Commit** | 任务可能永远无法通过门控；当前策略是耗尽重试后上报给人类操作员。 |

### 未来工作方向
1. **更深的谓词层级**（Deeper predicate ladders）：集成类型检查、长期运行进程的内存泄漏检测、打包验证等。
2. **基于连击的预算策略**（Streak-conditioned budgets）：根据连续成功提交的“连击”动态调整重试预算。
3. **谓词发现**（Predicate discovery）：在SWE-bench等基准上运行HCRC，其中谓词集合需从问题中推断，而非预先给出。
4. **绝对真理路径**（Absolute truth path）：建立不可被游戏化的验证路径，进一步增强安全性。

---

> 💡 **总结**：  
> HCRC 提供了一种**无需修改模型本身**即可大幅提升LLM推理可靠性的通用框架。它通过**执行层的硬性验证门控**，将“可信推理”从模型规模的竞赛转变为**可控执行的设计问题**。实验表明，该方法不仅能将虚假完成率降至0%，甚至在某些场景下还能**提升效率**，具有重要的理论与实践价值。

</details>

---

### 7. [Multi-Turn Distributed Inference with Mixture of Experts for 6G Edge--Cloud Networks](https://arxiv.org/abs/2607.02522)

**Authors**: Bo Liu, Haiyuan Li, Yuelin Liu, Yulei Wu, Rasheed Hussain, Shadi Moazzeni, Dimitra Simeonidou  
**Category**: cs.DC  
**Published**: 2026-07-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.02522v1  

#### Abstract
Mixture-of-Experts (MoE) architectures are increasingly deployed across 6G edge--cloud networks, where sparse activation reduces the computational footprint of each inference to only a fraction of the full expert set. However, MoE inference in edge-cloud networks creates a tension between KV state l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Multi-Turn Distributed Inference with Mixture of Experts for 6G Edge-Cloud Networks》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **6G edge-cloud networks** 中部署基于 **Mixture-of-Experts (MoE)** 架构的大语言模型（LLM）推理时，面临以下核心矛盾：
- **KV Cache 的持久性需求**：多轮对话（multi-turn inference）中，Key-Value (KV) 缓存随每一轮增长，需跨轮次复用以避免重复计算和传输开销。
- **专家计算的弹性调度需求**：MoE 模型稀疏激活特性允许将不同 expert 分布在网络各处，通过弹性调度提升资源利用率。

现有方法通常将状态与计算耦合处理，导致：
- KV state 频繁迁移带来高昂的跨站点传输开销；
- 或者无法有效利用边缘网络中的分布式算力。

### 提出的新方法：StateFlow
作者提出 **StateFlow** ——一种面向多轮 MoE 推理的分布式执行策略，其核心思想是：

> **Decouple persistent KV state from transient sparse computation**  
> 将持久化的 KV 状态与瞬态的稀疏专家计算解耦。

具体实现机制包括：
- ✅ **Sticky Owner Selection（粘性所有者选择）**：为每个对话选定一个稳定的“拥有站点”（owner site），长期锚定该对话的 KV state，确保跨轮次高效复用。
- ✅ **Latency-Aware Sparse Expert Routing（延迟感知稀疏专家路由）**：在每一层动态选择最优的远程 expert 执行位置，综合考虑网络延迟、链路拥塞和计算负载。
- ✅ **Path-Aware Aggregation（路径感知聚合）**：智能决定 expert 输出的聚合地点（可在 owner 或某个 expert site），最小化整体通信代价。
- ✅ **Online Congestion Adaptation（在线拥塞适应）**：引入 congestion price 机制动态调整调度决策，防止热点节点过载。

### 相比现有方法的优势
| 维度 | StateFlow | 现有方法（如 CommRoute, Llumnix, PRe-QuaL） |
|------|---------|----------------------------------------|
| **状态管理** | 显式分离 KV state 与计算，支持跨轮次连续复用 | 多数假设集中式执行或忽略跨轮状态一致性 |
| **调度灵活性** | 动态优化 expert 路由与聚合点 | 固定调度模式或仅基于单请求优化 |
| **系统效率** | 更高并发能力 + 更低尾部延迟 | 在高并发下性能急剧下降 |

---

## 2. 核心实验方法和设置

### 实验平台
- **真实测试床（real-world testbed）** 搭建于一台配备 **8块 NVIDIA A100 GPU** 的服务器上。
- 使用 **Linux network namespace + veth pair + tc-netem** 实现内核级网络仿真，精确控制带宽、延迟与队列行为。
- 模拟三层架构：
  - **Access-Edge (V₁)**：4 个站点（各 1 shard）
  - **Edge-Cloud (V₂)**：1 个站点（2 shards）
  - **Cloud-Core (V₃)**：1 个站点（2 shards）

### 模型与工作负载
- **模型**：`Mixtral-8x7B-Instruct`（32 层 MoE，每层 8 个 experts，top-2 激活）
- **最大上下文长度**：1536 tokens
- **对话结构**：多轮交互式任务（conversational AI 场景）
- **输入流量模式**：
  - 并发对话数从 4 到 40 扫描（capacity sweep）
  - 请求速率：1.5 req/s，最多 8 个 in-flight 请求

### 评估指标
| 指标 | 定义 |
|------|------|
| **Goodput** | 成功完成全部轮次且满足延迟预算的对话数量 |
| **p95 Turn-Level Completion Latency** | 单轮推理完成时间的第 95 百分位 |
| **Remote Dispatch Latency** | 跨站点发送激活值到专家并返回输出的时间 |
| **KV Hit Rate** | 注意力查询命中本地缓存的比例 |
| **Throughput (rps)** | 每秒成功响应的请求数量 |

### 基线方法对比
1. **CommRoute [12]**：通信感知 MoE 路由，但无 KV 连续性保护
2. **Llumnix [9]**：支持运行时迁移的 LLM 服务系统，反应式负载均衡
3. **PRe-QuaL [18]**：基于延迟与队列的负载均衡方案，不维护持久状态

> 所有方法使用相同模型、拓扑、硬件分配和网络配置，保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）稳定对话并发能力（Stable Dialogue Concurrency）
- **StateFlow 可维持稳定服务质量至 28 个并发对话**
- 其他基线在超过 **13 个并发**后即出现 p95 延迟陡增和吞吐崩溃
- ➤ **StateFlow 支持 >2× 更高的稳定并发量**

#### （2）p95 推理延迟降低
- 在 W=28 的负载下：
  - **StateFlow**: p95 = **281s**
  - **CommRoute**: p95 = **787s**（高出 180%）
  - **Llumnix**: p95 = **902s**
  - **PRe-QuaL**: p95 = **767s**
- ➤ **StateFlow 减少 turn-level p95 延迟达 53.0%**

#### （3）吞吐与延迟分布
| 方法 | Mean (s) | P95 (s) | Std. Dev. (s) | Throughput (rps) |
|------|----------|---------|----------------|------------------|
| **StateFlow** | 244 | 281 | **22** | **0.107** |
| CommRoute | 714 | 787 | 55 | 0.036 |
| Llumnix | 803 | 902 | 72 | 0.033 |
| PRe-QuaL | 720 | 767 | 41 | 0.036 |

- StateFlow 不仅平均延迟更低，且 **标准差最小（22s）**，表明性能更稳定可预测。

#### （4）远程调度开销显著降低
| 方法 | Remote Dispatch Latency |
|------|--------------------------|
| **StateFlow** | **468.5 ms** |
| CommRoute | 1408.7 ms |
| Llumnix | 1835.4 ms |
| PRe-QuaL | 1466.6 ms |

- ➤ **减少远程调度延迟约 66.7%~75%**

#### （5）KV 缓存命中率
所有方法均保持较高 KV hit rate（>94%），说明 StateFlow 的优势并非来自更好的缓存命中，而是源于更优的 **通信与调度设计**。

---

### 消融实验结果（Ablation Study）

对 StateFlow 各组件进行关闭测试（over 50-turn 对话）：

| 配置 | 相对基准延迟增加 | 尾部延迟上升 |
|------|------------------|-------------|
| **Full StateFlow** | ref | ref |
| **Sticky-Off**（禁用粘性 owner） | **+26.4%** | 显著上升 |
| **Routing-Off** | +3.7% | 轻微上升 |
| **PathAgg-Off** | +5.1% | 轻微上升 |
| **All-Off** | **+37.9%** | **+64.6%** |

> 🔍 发现：
- **Sticky owner 是最关键组件**，尤其对长对话影响巨大（KV 迁移成本随轮次累积）；
- Routing 与 Aggregation 提供稳定 per-turn 优化；
- 三者具有**协同效应**，共同作用下性能增益最大。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **KV state 与 sparse computation 必须解耦** 才能在 edge-cloud 环境中高效支持多轮 MoE 推理。
2. ✅ **Sticky ownership 是实现跨轮次连续性的关键**，能大幅减少不必要的跨站点状态迁移。
3. ✅ **动态、细粒度的 expert routing 与 aggregation 决策** 可显著降低通信开销，尤其在高并发场景下。
4. ✅ 实验验证：**StateFlow 实现 >2× 更高并发 + 53.0% p95 延迟下降 + 66.7% 远程调度开销削减**。

### 方法的局限性
- 当前依赖预定义的 **expert placement map**，未联合优化部署阶段；
- 假设 expert shards 已静态分布，未涉及 shard 动态迁移或预加载；
- 测试环境为单机多 namespace 模拟，尚未在广域物理边缘节点上验证。

### 未来工作方向
- 结合 **proactive expert prefetching** 与 **dynamic sharding**，进一步提升调度灵活性；
- 扩展至 **multi-user multi-tenant 场景** 下的资源隔离与 SLA 保障；
- 探索 **learning-based 控制器** 替代启发式策略，实现更优长期调度；
- 支持 **异构加速器混合部署**（如 GPU/NPU/FPGA）下的 MoE 分布式执行。

--- 

> 📌 总结一句话：  
> **StateFlow 通过“锚定状态、放飞计算”的设计理念，在 6G edge-cloud 网络中实现了高性能、高并发、低延迟的多轮 MoE 推理服务，为下一代分布式 LLM 系统提供了重要范式。**

</details>

---

### 8. [FAST: A Holistic Framework for Optimizing Memory-I/O, Computation, and Sampling in Temporal GNN Training](https://arxiv.org/abs/2607.05095)

**Authors**: Yushu Cai, Qingrui Zhu, Lei Liu, Kai Sheng, Hao Chen, Xin He  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.05095v1  

#### Abstract
Temporal Graph Neural Networks (TGNNs) are widely used for learning from dynamic graphs in applications such as recommendation, social network analysis, and traffic forecasting. However, scaling TGNN training to large dynamic graphs remains challenging due to three intertwined bottlenecks: memory I/...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FAST: A Holistic Framework for Optimizing Memory-I/O, Computation, and Sampling in Temporal GNN Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Temporal Graph Neural Networks (TGNNs) 在处理大规模动态图时面临三大瓶颈：
- **Memory I/O Bottleneck**：频繁的 host-device 数据传输导致 PCIe 带宽饱和，GPU 利用率低。
- **Computation Bottleneck**：稀疏且不规则的子图结构导致负载不均衡、缓存利用率低，影响 AGG 和 edgeSoftmax 等核心算子效率。
- **Sampling Bottleneck**：CPU 采样器未能有效利用 CPU 缓存层次结构，而 GPU 采样器虽高效但缺乏通用性和可复用性。

现有系统通常孤立优化这三个阶段，忽略了它们之间的协同效应，留下大量性能提升空间。

---

### 提出的新方法与创新思路
论文提出 **FAST** —— 一个端到端联合优化的框架，通过以下三个核心技术解决上述瓶颈：

#### ✅ **SlimCache**（内存 I/O 优化）
- **结合压缩与缓存**：利用 batch 内部的重复性（within-batch repetition）进行 ID 压缩，减少传输量；同时利用跨 batch 的重叠性（cross-batch overlap）在 GPU 上缓存“热”节点/边特征。
- **差异化处理节点与边**：基于预采样统计访问频率，采用贪心策略动态分配缓存资源，优先保留高复用度的数据。
- 支持在有限 GPU memory 下最大化缓存命中率。

#### ✅ **Thread-Efficient Graph Operators**（计算优化）
- **Balanced Aggregate (AGG)**：采用 **edge-centric COO 模式** 替代传统的 node-parallel 实现，消除因度分布偏斜造成的负载不均。使用 `atomicAdd` 合并结果，在小度图中开销可控。
- **Fast Edge-Softmax (ESM)**：设计基于 **thread-loop 的 CSR 风格 reduction**，避免 warp shuffle 中无效线程浪费，显著提高线程利用率，并扩大每个 block 处理的节点数（从 64 → 1024），增强缓存局部性。

#### ✅ **Topology-Aware Sampling**（采样优化）
- 分析不同 root node 子图间的拓扑相似性，构建 **thread affinity matrix**。
- 使用 **Blossom 算法** 进行最大权重匹配，将高相似性的采样任务绑定到共享 L2/L3 cache 的物理 CPU core 上，提升 CPU cache locality。
- 预先生成绑定配置，避免运行时开销。

---

### 相比现有方法的优势
| 特性 | FAST | 其他方法（如 TGL, ETC, SIMPLE, TASER） |
|------|------|----------------------------------------|
| **协同优化** | ✔️ 联合优化 I/O、计算、采样 | ❌ 各自为战，未考虑交互影响 |
| **缓存 + 压缩** | ✔️ 统一设计，区分节点/边模式 | ❌ 单独使用，或固定比例分配 |
| **算子定制化** | ✔️ 针对 TGNN 小度稀疏特性优化 AGG & ESM | ❌ 使用通用 gSpMM/gSDDMM 抽象 |
| **采样通用性** | ✔️ CPU-based，通用性强，无需专用硬件 | ❌ GPU 采样器高度定制，难迁移 |
| **实用性** | ✔️ 单机单卡即可部署，无额外硬件依赖 | ❌ 如 SWIFT 依赖大容量磁盘 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个真实世界的大规模动态图上进行评估：

| Dataset | 节点数 | 边数 | 描述 |
|--------|-------|-----|------|
| **LastFM** | 2K | 1.3M | 用户-音乐交互记录 |
| **Wiki-Talk** | 1.1M | 7.8M | Wikipedia 用户讨论页互动 |
| **Bitcoin** | 24.5M | 122.9M | 比特币交易网络 |
| **GDELT** | 17K | 191.3M | 全球事件知识图谱（近十亿级时间戳事件） |

> 特征维度从 128 到 413 不等，总特征大小最高达 **130GB**

---

### 实验设置与评估指标

#### 硬件环境
- CPU: 双路 Intel Xeon Gold 6133 (共 80 cores)
- RAM: 512 GB
- GPU: NVIDIA A100 (40 GB VRAM)
- 软件栈: PyTorch 2.1.2, DGL 0.9.1, CUDA 11.8

#### 模型选择
测试三种代表性 TGNN 模型：
- **TGN**：基于 memory vector 的时序建模
- **TGAT**：基于 attention 的聚合机制
- **DySAT**：结合结构与时间维度的 self-attention

#### 训练参数
- Batch Size: 2000
- Sampling: top-k recent neighbor (k=10)
- Epochs: 10
- Threads: 8（默认）

#### 评估指标
- **训练效率**：每 epoch 执行时间（秒）
- **模型精度**：Link Prediction 任务下的 Test-set Average Precision (AP)
- **加速比**：相对于 baseline 的 speedup
- **消融分析**：各组件贡献分解

---

### 基线方法对比
与以下 state-of-the-art 框架比较：
- **TGL**：基础框架，支持大规模 TGNN 训练
- **ETC**：基于特征压缩 + 计算/压缩流水线
- **SIMPLE**：基于动态缓存放置策略
- **SWIFT**（仅用于 overhead 对比）：基于磁盘 I/O 的异步训练框架

---

## 3. 主要实验结果和性能指标

### 端到端性能表现（Table 4）

| 方法 | 平均加速比 | 最高加速比 | AP 准确率 |
|------|-----------|----------|---------|
| vs. TGL | **2.6×** | **4.7×** | ✅ 无损失 |
| vs. ETC | **1.3×** | **1.4×** | ✅ 无损失 |
| vs. SIMPLE | **1.6×** | **2.6×** | ✅ 无损失 |

> **平均整体加速达 2.1×，最高达 4.7×**

#### 典型案例：
- 在 **GDELT + TGN** 上，FAST 达到 **4.2× speedup**
- 在 **BITCOIN + TGN** 上，达到 **2.5× speedup**
- **SIMPLE 和 ETC 在 GDELT 上出现 OOM**，无法完成训练

---

### 消融实验结果（Ablation Study）

#### ✅ SlimCache 贡献
- 在 WIKITALK 上，仅压缩即可提速 **3.0×**
- 加入贪心缓存后，I/O 流量从 49GB → 13GB，实现 **5.3× I/O 加速**
- 在 BITCOIN 上，流量从 612GB → 114GB，**3.1× I/O speedup**

> 表明 **压缩 + 缓存联合设计** 显著优于单一手段

#### ✅ Thread-Efficient Operators 贡献
| 算子 | 最高加速比 | 数据集 |
|------|----------|--------|
| AGG | **2.3×** | BITCOIN |
| ESM | **4.2×** | LASTFM/BITCOIN |

> 表 5 显示：
- ESM 的 L1 cache hit rate 从 32.46% → **90.26%**
- 每 SM active warps 数从 12.95 → **29.58**
- block 内处理节点数从 32 → **512**，极大提升空间局部性

#### ✅ Topology-Aware Sampling 贡献
- 采样阶段加速 **1.14× ~ 1.46×**
- 在 WIKITALK 上，L2 hit rate 从 41% → **50–52%**，L3 hit rate 从 80% → **98%**
- 预处理开销仅占端到端训练时间的 **平均 2.7%**（最高 5.4%）

---

### 端到端加速分解（Figure 11(c)）
以 TGN 为例，总加速来源如下：
1. **SlimCache (SC)**：最大贡献者，主要降低 I/O 时间
2. **Thread-Efficient (TE)**：带来额外 **1.21×** 加速，缓解 AGG 负载不均与 ESM reduction 开销
3. **Topology-Aware (TA)**：再增 **1.11×**，改善采样缓存命中

> 三者协同作用，解释了全部加速效果

---

## 4. 关键结论和发现

### 主要发现
1. **三大瓶颈紧密耦合**：单独优化任一阶段难以突破性能天花板，必须进行 **cross-stage co-design**。
2. **节点与边具有不同的冗余模式**：节点重复率高适合压缩，边跨 batch 重叠多适合缓存 → 应区别对待。
3. **小度稀疏图不适合传统 warp shuffle reduction**：应改用 thread-loop 设计提升线程利用率。
4. **CPU 缓存可被显式利用**：通过 topology-aware thread binding 可显著提升采样效率。
5. **预处理开销可控**：轻量级 pre-sampling 代价远小于收益，适合实际部署。

---

### 方法的局限性
1. **当前聚焦于单机单卡场景**：尚未扩展至分布式或多 GPU 架构。
2. **主要针对 CTDG（Continuous-Time Dynamic Graphs）**：对 DTDG 的优化仍非最优。
3. **Greedy Cache Selection 是启发式的**：理论上非全局最优，但实践中已足够有效。
4. **依赖 OpenMP 4.0 的 thread binding 能力**：在某些平台可能受限。

---

### 未来工作方向
1. 扩展至 **multi-GPU / distributed setting**，研究跨设备缓存一致性与通信优化。
2. 针对 **DTDG-style training** 设计专用优化策略。
3. 探索更智能的 **dynamic cache replacement policy**，适应训练过程中热点变化。
4. 将 FAST 的思想应用于其他图学习任务，如 Temporal Link Prediction、Event Forecasting 等。

---

> 🔗 **代码开源地址**：[https://github.com/NoneBone/FAST](https://github.com/NoneBone/FAST)  
> 📄 **会议录用信息**：Accepted to ICPP 2026

</details>

---

### 9. [Edge-Deployable LLM Fine-Tuning on a Single GPU for Telecom Network Troubleshooting](https://arxiv.org/abs/2607.02523)

**Authors**: Chenhua Shi, Bhavika Jalli, John Zou, Gregor Macdonald, Wanlu Lei, Mridul Jain, Joji Philip  
**Category**: cs.DC  
**Published**: 2026-07-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.02523v1  

#### Abstract
Telecom troubleshooting at edge sites requires low-latency model responses and localized model adaptation to satisfy operational and data sovereignty requirements. However, deploying large language models (LLMs) at telecom edge sites is constrained by limited power, cooling, space, and weight budget...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Edge-Deployable LLM Fine-Tuning on a Single GPU for Telecom Network Troubleshooting*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本论文针对**电信网络边缘站点（如 O-RAN near-RT RIC、MEC 平台）中部署大语言模型（LLM）进行故障排查时面临的实际挑战**，解决了以下关键问题：

- **低延迟需求**：云上训练/推理存在往返延迟，无法满足实时网络故障诊断要求。
- **数据主权限制**：运营商敏感数据（告警日志、性能计数器等）受 GDPR 等法规约束，不能上传至云端。
- **硬件资源受限**：边缘站点 GPU 资源有限（典型为单块 RTX A6000，48GB VRAM），且受限于功耗、散热、空间。
- **模型适配效率低**：现有研究多聚焦于边缘**推理**而非**本地微调**，缺乏对单 GPU 微调资源边界的系统性刻画。

因此，如何在**单块边缘级 GPU 上实现稳定、高效、可自动化的 LLM 微调**成为亟需解决的问题。

---

### 提出了什么新方法或新思路

1. **提出并验证了“单 GPU 边缘可部署微调”（edge-deployable fine-tuning）的可行性**  
   首次系统性地展示了在 **NVIDIA RTX A6000（48GB VRAM）** 上完成从 SFT 到 RFT 全流程微调的实践路径，证明无需依赖云基础设施即可生成高质量领域模型。

2. **构建了基于 Unsloth 框架的 GPU 资源画像体系（GPU Profiling Study）**  
   系统分析了多个关键参数对内存和稳定性的影响：
   - 最大序列长度（max sequence length）
   - GPU 内存利用率（GPU utilization）
   - LoRA rank
   - 生成数量（number of generations）
   - KV Cache 占用与 activation memory 开销

   提出“安全操作边界”（safe operating envelope），为自动化边缘训练管道提供配置指南。

3. **提出混合 Tokenizer 策略以支持推理感知训练（hybrid tokenizer strategy）**  
   发现不同模型架构（reasoning vs. non-reasoning）在 `<think>` 标签处理上的不一致性：
   - `Unsloth` 版本的 DeepSeek-R1 在 SFT 中会自动剥离 `<think>` 内容；
   - `AutoTokenizer` 可保留标签用于训练，但推理时不激活。

   因此提出：
   - **SFT 阶段使用 AutoTokenizer（保留 reasoning trace）**
   - **RFT 和推理阶段使用 Unsloth 实现（启用内置 reasoning visibility）**

   这种组合策略实现了训练期间的知识注入与推理期间的显式推理输出之间的协同。

---

### 相比现有方法的优势

| 维度 | 现有工作局限 | 本文优势 |
|------|---------------|---------|
| **部署层级** | 多集中于边缘推理或联邦学习 | 支持**本地完整微调流程**，真正实现闭环适应 |
| **资源配置指导** | 缺乏具体 VRAM 使用建模 | 提供量化内存分配公式与安全阈值建议 |
| **模型行为理解** | 忽视 chat template 对 reasoning 的影响 | 揭示并解决 reasoning 模型在 SFT/RFT 中的行为不一致问题 |
| **实用性** | 多假设理想环境 | 基于真实电信数据集，在贴近生产的边缘硬件上验证 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **自研电信故障排查数据集** [25]，包含：
  - **50 条专家标注种子样本（SME-validated seed pairs）**：用于 SFT。
  - **500 条合成生成样本（synthetically generated RFT pairs）**：用于 RFT，缓解数据稀缺问题。
- 每个样本均结合 **top-3 检索到的上下文文档块**（来自知识图谱），平均总 token 数约为 **16,126**。
- 示例任务：根据高温告警，生成分步排障方案（含检查风扇、气流、重启指令等）。

---

### 实验设置

#### 硬件平台
- **NVIDIA RTX A6000（48GB VRAM）**，代表典型的区域 NOC 和边缘聚合节点设备。

#### 模型选择
- **非推理型模型**：`Qwen2.5-7B`（7.61B 参数，4-bit quantized）
- **推理增强型模型**：`DeepSeek-R1-0528-Qwen3-8B`（8.2B 参数，通过 CoT 蒸馏获得）

#### 微调流程
采用两阶段策略：
1. **Supervised Fine-Tuning (SFT)**：学习标准输出格式与步骤逻辑。
2. **Reinforcement Fine-Tuning (RFT)**：使用 **Group Relative Preference Optimization (GRPO)**，基于奖励函数优化响应质量，适用于小样本场景。

#### 工具链
- **Unsloth 框架**：集成 LoRA、FlashAttention-2、XFormers、KV Cache 加速与 `torch.compile`（inductor）。
- **4-bit NF4 量化**：降低模型权重占用（Qwen2.5-7B 仅占 3.8GB）。
- **LoRA 配置**：仅更新 `q_proj`, `k_proj`, `v_proj` 模块，rank ∈ {8, ...}。

---

### 评估指标

使用 **RAGAS 自动化评估框架** 在同一边缘 GPU 上运行本地评估，指标包括：

| 指标 | 含义 |
|------|------|
| **Question Specificity (Q.Spec)** | 问题是否明确、可回答 |
| **Answer Relevance (Ans.Rel)** | 回答是否相关且完整 |
| **Groundedness (Ground)** | 回答是否基于检索上下文，减少幻觉 |
| **AspectCritic (Answerable)** | 是否判断问题本身是否合理 |

此外还监控：
- **OOM / CAIM 错误发生情况**
- **训练耗时（duration）**
- **最大 batch size 承载能力**
- **KV Cache 利用率与编译开销**

---

### 基线方法对比

- **Zero-shot 推理 vs. Fine-tuned 模型**：衡量微调带来的增益。
- **Qwen2.5-7B vs. DeepSeek-R1-0528-Qwen3-8B**：比较 reasoning 架构优势。
- **跨架构验证**：在 `Llama-3.1-8B-Instruct` 上复现实验，检验结论泛化性。
- **多 GPU 设置（4×RTX A6000 + DDP）**：验证单 GPU 结论对分布式系统的预测价值。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table V）

| Model | Q.Spec | Ans.Rel | Ground | Answerable |
|-------|--------|---------|--------|------------|
| **DeepSeek (Zero-Shot)** | 1.00 | 0.77 | 0.67 | 0.96 |
| **DeepSeek (Fine-Tuned)** | 1.00 | **0.96** | **0.78** | **1.00** |
| **Qwen2.5 (Zero-Shot)** | 0.88 | 0.72 | 0.15 | 0.84 |
| **Qwen2.5 (Fine-Tuned)** | 0.96 | 0.85 | 0.22 | 0.92 |

> ✅ **Fine-tuning 显著提升所有指标，尤其在 groundedness 上效果显著（+0.63）**

> ✅ **DeepSeek-R1 在 fine-tuned 后全面优于 Qwen2.5，尤其在 answer relevancy (+0.11) 和 groundedness (+0.56) 上表现突出**

---

### 与基线方法的对比结果

| 对比项 | 结果 |
|--------|------|
| **vs. Cloud-based Training** | 完全避免数据外传，延迟归零；训练可在 3–4 小时内完成，适合夜间更新 |
| **vs. Zero-shot LLMs** | 微调后模型在 relevance 和 grounding 上大幅提升，尤其在复杂排障任务中更可靠 |
| **DeepSeek-R1 vs. Qwen2.5** | 前者在 reasoning 表达、事实依据方面明显更强，得益于其 CoT 蒸馏设计与 hybrid tokenizer 策略的有效利用 |

---

### 消融实验与关键发现（Ablation Insights）

#### （1）序列长度与 GPU 利用率的权衡
- 当 max sequence length > 60k 时，Qwen 出现 **CannotAccessIllegalMemory (CAIM)** 错误（编译期崩溃）。
- 在 50k 序列下，GPU utilization < 0.5 会导致 **Cache Block Error**（KV Cache 不足）。
- 推荐设置：**max_seq_len = 数据集第 75 百分位长度（推荐 50k）**，**utilization ≈ 0.7**

#### （2）输入 token 数对 OOM 的影响
- 输入 tokens > 10k 且 batch size ≥ 2 → 几乎必然 OOM。
- 输入 tokens = 5k → 训练可在 **~173 分钟（约 3 小时）内完成**，支持每日增量更新。

#### （3）generation 数量的影响
- 增加 generations 至 4 或 6 对内存无显著增加，但延长训练时间（+40% ~ +67%）。
- 表明 **memory 主要由 activation weights 和 KV Cache 决定，而非 generation 数量本身**。

#### （4）跨架构差异显著（Llama vs. Qwen）
| 指标 | Qwen2.5-7B | Llama-3.1-8B |
|------|-----------|-------------|
| 支持最长序列 | ≤50k（60k CAIM） | 成功训练至 60k |
| 250 步训练时间（20k input） | ~229 min | **~29 min（快 7×）** |
| Batch Size limit (>20k seq) | ≥4 → OOM | ≥4 → OOM（共通限制） |

> 🔍 **发现**：batch size 限制具有跨架构通用性（源于 activation memory scaling），但编译行为和训练速度高度依赖模型结构（如 Q/KV head 数量）。

---

## 4. 关键结论和发现

### 主要发现

1. **单 GPU 边缘微调是可行且必要的**  
   在 RTX A6000 上可成功完成 SFT+RFT 流程，满足电信边缘场景下的低延迟与数据主权需求。

2. **必须进行 per-model profiling**  
   不同模型家族（Qwen vs. Llama）在最大序列长度容忍度、编译稳定性等方面存在显著差异，“一刀切”的配置不可靠。

3. **reasoning 模型需特殊处理 chat template**  
   忽视 tokenizer 对 `<think>` 标签的处理方式将导致训练与推理脱节。提出的 **hybrid tokenizer strategy** 是实现推理连贯性的关键。

4. **activation memory 是 OOM 主因**  
   长输入序列下，activation weights 快速增长，远超 LoRA adapter 开销，是训练失败的根本原因。

5. **Unsloth + GRPO + RAGAS 形成完整边缘闭环**  
   从高效训练、强化学习优化到本地评估，整个 pipeline 可在单卡上独立运行，无需外部服务。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **模型规模上限** | 当前仅支持 7–8B 参数级别，更大模型（如 70B）仍难以在单卡部署 |
| **依赖特定框架** | 高度依赖 Unsloth 和 inductor 编译优化，其他工具链可能不具备相同性能 |
| **数据生成依赖人工种子** | 虽然使用合成数据扩展，但仍需少量高质量 SME 种子启动流程 |
| **未测试更多边缘芯片** | 目前仅验证 NVIDIA GPU，尚未覆盖 Jetson、L4、ASIC 等新型边缘加速器 |

---

### 未来工作方向

1. **Federated Fine-Tuning Across Edge Sites**  
   在多个边缘节点间协作微调，共享知识而不集中原始数据。

2. **Event-Triggered Auto-Update Pipelines**  
   当检测到新告警类型、拓扑变更或性能漂移时，自动触发本地模型再训练。

3. **扩展至新兴边缘硬件**  
   将 profiling 方法推广至 **NVIDIA L4、Jetson AGX Orin、FP8 量化架构**，探索更低功耗部署路径。

4. **动态 context compression 技术**  
   开发更智能的长文本压缩机制（如 context-weighted pruning），突破 10k+ tokens 的 VRAM 瓶颈。

5. **多 Agent System (MAS) 集成**  
   结合 fine-tuned SLMs 构建自动化排障 agent 团队，实现端到端故障闭环处理 [27]。

--- 

> 📌 **总结一句话**：  
> 本文首次系统论证了在 **单块边缘 GPU 上完成高质量 LLM 微调的可行性与最佳实践路径**，不仅提供了实用的部署 recipe，更重要的是揭示了 **模型架构、tokenizer 行为与硬件资源之间复杂的交互关系**，为未来边缘 AI 的落地奠定了坚实基础。

</details>

---

### 10. [HyperParallel-Mpipe: A Composable Algebra System for Optimizing MLLM Training over Supernode Clusters](https://arxiv.org/abs/2607.03229)

**Authors**: Chong Li, Zhengdao Yu, Nelson Lossing, Thibaut Tachon, Pierre Leca, Etienne Filhol, Yujie Yuan, Chong Bao, Teng Su  
**Category**: cs.DC  
**Published**: 2026-07-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.03229v1  

#### Abstract
Modern AI applications have expanded beyond text-only interaction into a wide range of multimodal scenarios, making multimodal large language models (MLLMs) crucial for both research and industry. However, compared with traditional decoder-only LLM training, large-scale MLLM training often shows muc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*HyperParallel-Mpipe: A Composable Algebra System for Optimizing MLLM Training over Supernode Clusters*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代多模态大语言模型（**MLLM**）训练面临两个核心挑战：
- **组件异构性（Component Heterogeneity）**：MLLM 包含功能不同的模块（如 ViT 编码器、LLM backbone），其计算、内存和通信特征差异显著。
- **动态编码器负载（Dynamic Encoder Workloads）**：图像/视频等模态输入的分辨率、长度变化导致每 microbatch 的 encoder 计算量波动剧烈，引发 pipeline 中的“气泡”（bubbles），降低 **MFU（Model FLOPs Utilization）**。

传统并行策略（如 Pipeline Parallelism, Data Parallelism）难以有效处理这种异构性和动态性，导致训练效率远低于纯文本 LLM。

---

### 🚀 提出的新方法与思路
论文提出 **Mpipe**，一个基于**调度代数（schedule algebra）** 的可组合系统，用于优化 MLLM 在超节点集群上的训练。

#### 核心创新点：
1. **调度代数（Schedule Algebra）**
   - 将并行训练调度形式化为一种**小型可组合代数系统**。
   - 调度由 `cut`（模型分段）和 `schedule`（每个段的执行骨架）定义。
   - 支持从抽象调度推导出具体的运行时行为（device placement, collective communication, 执行顺序），并预测 step makespan。

2. **Transpose 调度机制**
   - 利用代数推导出名为 **transpose** 的新型静态调度策略：
     - 将 **modality encoder 复制到所有 pipeline ranks** 上。
     - 在 pipeline 的 **warmup bubbles**（预热阶段空闲时间）中执行 encoder forward。
     - 通过 `Gather` 操作将输出聚合至第一个 LLM stage。
   - 实现 encoder 计算与 pipeline 填充阶段重叠，避免其成为关键路径瓶颈。

3. **静态不变性与零开销**
   - Transpose 是**静态调度**，不依赖于每 batch 的模态混合或 encoder 负载。
   - 无需 runtime profiling、load balancing 或 per-iteration schedule search。
   - 支持 encoder 冻结或微调（trainable），调度不变（得益于 *Schedule-Invariance Corollary*）。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | Mpipe 的优势 |
|------|--------|--------------|
| **DistTrain** | 需要 cluster-level 资源分区，对动态负载适应差 | 不需资源重划分，静态调度应对动态输入 |
| **Optimus** | 依赖 encoder workload profiling，动态变化下失效 | 无需 profiling，利用 warmup bubble 吸收波动 |
| **DIP** | 运行时负载均衡、元数据预取带来额外开销 | 完全静态，无 runtime overhead |
| **通用 PP/GPipe** | encoder 作为普通 stage 引发严重 bubble | 将 encoder 移出关键路径，隐藏在其 warmup 中 |

> ✅ **核心优势总结**：Mpipe 通过 **静态重映射（static remapping）** 将 encoder 计算“转置”进 pipeline 固有空闲区域，实现 **zero scheduling overhead + high robustness to dynamic workloads**。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Experiment A（大规模生产级）**：内部多模态数据集（未公开细节）
- **Experiment B（小规模验证）**：**CapsFusion** 图像-文本数据集（Yu et al., 2024）

---

### ⚙️ 实验设置
| 项目 | 设置详情 |
|------|----------|
| **硬件平台** | Ascend 910C NPU 集群 |
| **环境** | 512-device CloudMatrix384（Experiment A）；8-device（Experiment B） |
| **框架实现** | 基于 **Hyper-Parallel** 框架实现 Mpipe |
| **评估指标** | - 平均端到端 step time<br>- Speedup（加速比）<br>- MFU（隐含在 speedup 中） |
| **对比基线** | - Experiment A：类 DistTrain 架构（encoder 单独 stage + 定制并行策略 + LLM 使用 5D parallelism）<br>- Experiment B：Megatron-LM 风格 baseline |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Tables 3 & 4）

| 实验 | 模型 | Baseline Step Time | Mpipe Step Time | Speedup |
|------|------|---------------------|------------------|---------|
| **A（512卡）** | ViT + DeepSeek-V3 | 16.26s | 13.42s | **1.21×** |
| **B（8卡）** | Qwen3.5 MLLM | 11.01s | 4.07s | **2.70×** |

---

### 🔍 结果分析
- **加速来源**：成功将 encoder 计算隐藏在 pipeline warmup bubbles 中，减少了暴露的 bubble 时间。
- **不同规模下表现差异原因**：
  - **Experiment A（1.21×）**：LLM backbone 规模大，主导 step time，encoder 占比较小 → 改善空间有限。
  - **Experiment B（2.70×）**：encoder 占比更高，且 warmup slack 更易被充分利用 → 加速更显著。
- **损失一致性**：Mpipe **不改变训练 loss 曲线**，仅重排计算顺序，保证训练正确性。

> ❗ 注：文中未报告消融实验（ablation study），但通过理论建模（cost model）解释了性能增益的边界条件（如 encoder work > warmup slack 时会溢出）。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **MLLM 训练低效主因是 encoder 的源侧方差（source-side variance）暴露在关键路径上**。
2. **Transpose 调度能有效将 encoder 计算转移到 pipeline warmup bubbles 中**，实现计算重叠，提升 MFU。
3. **基于代数的调度系统（Mpipe）可统一表达多种 pipeline 调度（如 1F1B, GPipe, VPP）**，并支持新调度的推导与组合。
4. **静态调度优于动态调度**：在 encoder-bound 场景下，Mpipe 以近零开销获得显著加速，且运行可复现。

---

### ⚠️ 方法的局限性
1. **当 encoder work 超过 warmup slack 时，部分计算仍会溢出到关键路径**，无法完全消除影响。
2. **不处理 LLM 序列长度变化带来的内部 pipeline 动态性**（需依赖 sequence bucketing/packing）。
3. **不适用于 sink-side 组件（如生成器）**：这些模块依赖 backbone 输出，无法前置到 warmup 阶段。
4. 当前实现仅支持两段式切割（encoder / LLM），尚未扩展至更复杂的多段异构结构。

---

### 🔮 未来工作方向
1. **支持更丰富的调度结构**：
   - 引入 `fold`（对应 Hanayo 的 wave-like 并行）
   - 支持 `event split`（对应 DualPipeV）
   - 探索双向馈送（bidirectional feed，如 DualPipe）
2. **结合轻量级 runtime 元数据引导重排序**：在保持 loss-preserving 前提下，进一步减少 spill。
3. **扩展至 encoder-generator 全流程调度优化**。
4. **进行系统化的 scale sweep 实验**，量化 encoder-to-bubble ratio 对性能的影响。

---

## 总结
> **Mpipe 通过“调度代数 + transpose 重映射”，将 MLLM 中最不稳定的 encoder 计算“转置”进 pipeline 的天然空闲窗口，实现了静态、高效、无开销的训练加速。它不是简单添加一个新调度，而是构建了一个可组合、可推理、可预测的并行调度设计语言，为未来复杂异构模型的训练系统提供了新的范式。**

</details>

---

### 11. [Sangam: Efficiently Serving Diffusion LLMs with the AR Stack](https://arxiv.org/abs/2607.04206)

**Authors**: Nitin Kedia, Saurabh Agarwal, Myungjin Lee, Aditya Akella  
**Category**: cs.DC  
**Published**: 2026-07-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.04206v1  

#### Abstract
Diffusion language models (dLLMs) generate text by iteratively denoising a masked response and can commit multiple output positions per model invocation. Their bidirectional attention prevents exact autoregressive-style KV caching, since committing one position shifts the KV activations of all other...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Sangam: Efficiently Serving Diffusion LLMs with the AR Stack**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
扩散语言模型（**dLLMs**）因其并行去噪机制，在单次推理中可生成多个输出位置，显著提升了生成速度。然而，其**双向注意力机制**导致传统的 **KV Cache** 无法像自回归（**AR**）模型那样稳定复用，从而破坏了现有 LLM 推理系统中的关键假设，如：
- **KV 缓存复用**（exact KV caching 不成立）
- **Prefill-Decode 干扰**（prefill-decode interference）
- **静态资源划分**（static prefill/decode partitioning）

这使得现有的 AR 推理优化技术（如连续批处理、分块 Prefill、解耦执行）难以直接应用于 dLLMs。

### **提出了什么新方法或新思路**
论文提出 **Sangam**，一个专为缓存型 dLLMs 设计的高效推理服务系统，核心创新如下：

#### ✅ **1. Deficit Token-Budget Scheduler（赤字令牌预算调度器）**
- **目标**：在**不支持 Prefill 分块**（chunked prefill）的前提下，实现近似无阻塞的共置调度（amortized stall-free colocated serving）。
- **机制**：
  - 每个迭代周期有一个固定令牌预算 `t`。
  - 优先调度所有进行中的 Decode 请求。
  - 剩余预算 + 上一轮未使用的“赤字”用于调度等待的 Prefill。
  - 大型 Prefill 可通过累积赤字被接纳，避免饥饿。
- **优势**：用“时间延迟”替代“空间分块”，在无法 chunk 的情况下实现干扰控制。

#### ✅ **2. Hybrid Serving 架构**
- **结合共置（colocated）与解耦（disaggregated）的优点**：
  - 保留专用 Prefill 工人池。
  - 将 Decode 工人替换为使用**紧预算的 deficit 调度器**的共置工人。
- **溢出机制**：当 Prefill 池饱和时，请求溢出到 Decode 工人上执行本地 Prefill + Decode，无需跨节点 KV 传输。
- **保护机制**：通过 deficit 预算限制溢出 Prefill 对 Decode 的干扰。

#### ✅ **3. 统一架构支持多种部署模式**
Sangam 支持三种部署模式：
- **Colocated**：所有工人同时处理 Prefill 和 Decode。
- **Disaggregated**：Prefill 与 Decode 工人完全分离。
- **Hybrid**：Prefill 专用 + Decode 侧使用 deficit 调度的共置工人。

---

### **相比现有方法的优势**
| 方面 | 现有方法（如 Fast-dLLM, dLLM-Serve） | Sangam |
|------|----------------------------------------|--------|
| **批处理能力** | 通常无在线批处理，batch=1 | 支持动态连续批处理 |
| **Prefill 干扰控制** | 无有效机制，常采用 Prefill 优先策略 | 通过 deficit 预算实现干扰控制 |
| **资源利用率** | 静态划分易导致资源碎片化 | Hybrid 模式动态利用空闲 Decode 容量 |
| **灵活性** | 固定调度策略 | 支持多种部署模式，适应不同负载 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ShareGPT Chat**：约 53k 条英文对话，代表**Decode-heavy**场景。
- **arXiv Summarization**：科研论文摘要任务，提示词更长，代表**Prefill-heavy**场景。

| 数据集 | Prompt 中位数 | Prompt P99 | Output 中位数 | Output P99 |
|--------|----------------|-------------|----------------|-------------|
| ShareGPT | 795 | 2007 | 596 | 2668 |
| arXiv Summ. | 2827 | 3923 | 890 | 1623 |

### **实验设置**
- **硬件**：AWS p5.48xlarge 节点，8× NVIDIA H100 GPU（80GB HBM）。
- **模型**：
  - **LLaDA-8B-Instruct**（基于 LLaMA-3-8B）
  - **Dream-7B-Instruct**（基于 Qwen2.5-7B）
- **缓存策略**：采用 **Fast-dLLM** 的块级 KV 缓存（block size = 32）。
- **采样策略**：置信度阈值采样（confidence-threshold sampling, threshold=0.9）。

### **评估指标**
- **End-to-End Latency**（均值 & P99）
- **Prefill Queueing Delay**
- **Decode Time**
- **Throughput**（以 QPS 衡量）

### **基线方法对比**
1. **Fast-dLLM**（in-system 版本）：作为批处理前的基线。
2. **Colocated**：使用 deficit 调度器，预算 `t ∈ {512, 1024, 2048, 16384}`。
3. **Disaggregated**：固定 5P3D（5 Prefill + 3 Decode 工人）。
4. **Hybrid**：5P3C（5 Prefill + 3 deficit 调度的共置工人）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **Q1: Sangam vs. Fast-dLLM**
- 在 **LLaDA-8B + ShareGPT** 上：
  - Fast-dLLM 最高支持 **QPS 0.5**，平均延迟 ~20.7s。
  - Sangam（Colocated, t=1024）支持 **QPS 1.0**，延迟相当 → **吞吐提升 3×**。
- 在 **arXiv Summ.** 上也有 **2.5–3× 吞吐提升**。

> 📌 **说明**：批处理对 dLLM 至关重要，Sangam 显著优于无批处理方案。

#### ✅ **Q2: Colocated vs. Disaggregated vs. Hybrid**
| 场景 | 最优配置 | 性能优势 |
|------|----------|----------|
| **Decode-heavy**（LLaDA-8B + ShareGPT） | **Colocated** | 比 Hybrid 降低 **9–20%** 平均延迟 |
| **Prefill-heavy**（Dream-7B + arXiv） | **Hybrid / Disaggregated** | 比 Colocated 降低 **8–20%** 平均延迟 |

- **Disaggregated** 在负载上升时迅速饱和（如 QPS=7.5 时延迟飙升）。
- **Hybrid** 在所有场景下表现稳健，接近最优。

#### ✅ **Q3: Deficit 调度器有效性（消融实验）**
- **预算 `t` 的影响**：
  - **小预算（t=512）**：Decode 干扰少，但 Prefill 排队严重 → **尾部延迟高**。
  - **大预算（t=16384）**：Prefill 优先，Decode 被频繁中断 → **尾部 Decode 时间长**。
  - **中等预算（t=1024）**：在 ShareGPT 上取得最佳平衡，**P99 延迟最低**。

> 📌 **结论**：存在一个最优预算值，在干扰与排队之间权衡。

#### ✅ **Q4: Hybrid 调度中预算的影响**
- 在 **5P3C** 配置下：
  - **低预算（t=1024）**：P99 延迟比高预算（t=4096）低 **8–14%**（QPS=7.5）。
  - **原因**：更严格地限制溢出 Prefill 对 Decode 的干扰。
- 在 **3P5C**（Prefill 池较小）：
  - 低预算导致 Prefill 排队增加，**平均延迟上升 9%**。
  - 说明需根据 Prefill 池大小调整预算。

---

## **4. 关键结论和发现**

### **主要发现**
1. **dLLM 推理的核心挑战仍是 Prefill/Decode 干扰与资源划分问题**，与 AR 模型类似，但因以下特性加剧：
   - **Block-sized Decode**：Decode 成本高，批量增益有限。
   - **Recurring Prefill**：每个块边界都可能触发 Prefill，干扰更频繁。
   - **Bidirectional Attention**：禁止 Prefill 分块，传统干扰缓解机制失效。

2. **Colocated 与 Hybrid 是更优选择**：
   - **Decode-heavy 负载**：**Colocated + deficit 调度** 最优，减少干扰且资源利用率高。
   - **Prefill-heavy 负载**：**Hybrid** 最优，通过溢出机制缓解 Prefill 池瓶颈。

3. **Deficit Token-Budget 是关键机制**：
   - 实现了在**无法分块 Prefill** 的情况下，对干扰的精细控制。
   - 为 Hybrid 架构提供了**保护 Decode 的手段**。

4. **Hybrid 是对静态解耦部署的简单而有效的增强**：
   - 仅需将 Decode 工人替换为 deficit 调度的共置工人。
   - 即可显著提升在 Decode-heavy 场景下的性能，同时保持 Prefill-heavy 场景的优势。

---

### **方法的局限性**
- **依赖近似 KV 缓存**：如 Fast-dLLM 或 dKV-Cache，不适用于精确 KV 复用不可行的 dLLM 变体。
- **赤字调度引入延迟不确定性**：大型 Prefill 可能被延迟多轮，影响公平性。
- **未考虑跨节点部署**：实验基于单节点，KV 传输开销可能在多节点场景中放大。

---

### **未来工作方向**
- **动态调整 deficit 预算**：根据实时负载自动调节 `t`，而非手动设定。
- **支持更多 dLLM 架构**：如 Block-Causal dLLMs（如 SDAR）、Tri-Mode 模型。
- **跨节点 Hybrid 调度**：研究如何在分布式环境下高效实现 KV 共享与溢出。
- **与推测解码（Speculative Decoding）结合**：探索 dLLM 与 SpecDec 的协同优化。

---

> 🔗 **代码开源**：Sangam 已在 GitHub 开源：[https://github.com/UT-InfraAI/sangam](https://github.com/UT-InfraAI/sangam)

</details>

---

### 12. [Quantize the Target, Quantize the Drafter: Efficient Inference with Qwen3.5-4B](https://arxiv.org/abs/2607.04244)

**Authors**: Jaeyeon Kim, Jewon Lee, Bo-Kyeong Kim  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.04244v1  

#### Abstract
This report describes our approach to the Efficient Qwen Competition, where the goal is to enable low-latency serving of Qwen3.5-4B on a resource-constrained NVIDIA A10G GPU. Our system combines a quantized target model with speculative decoding. To recover accuracy, we apply quantization-aware dist...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Quantize the Target, Quantize the Drafter: Efficient Inference with Qwen3.5-4B*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文针对 **在资源受限的硬件（NVIDIA A10G GPU）上实现 Qwen3.5-4B 大语言模型的低延迟推理** 这一挑战，旨在显著降低生成延迟，同时满足严格的质量阈值要求。

### 🚀 提出的新方法与创新思路
作者提出了一套结合 **量化目标模型（Quantized Target）与轻量级扩散草稿模型（Diffusion Drafter）** 的高效推理系统，其核心创新包括：

- **Quantization-Aware Distillation (QAD) for Target Model**  
  在已有 AWQ INT4 量化模型基础上，采用 QAD 技术进行知识蒸馏，恢复因量化导致的精度损失，且**保留原始的量化网格（quantization grid）**，确保部署兼容性。

- **Two-Stage Training for Block-Diffusion Drafter**  
  针对量化后的目标模型训练专用的 DFlash 草稿模型：
  1. 第一阶段：用高精度（BF16）目标模型预训练草稿模型；
  2. 第二阶段：用 QAD 微调后的 INT4 目标模型进一步微调草稿模型。  
  此设计使草稿模型能先学习通用模式，再适配量化分布，提升 draft token 接受率。

- **Drafter-Level Optimization via PTQ + Sliding-Window Attention (SWA)**  
  对草稿模型进一步应用 GPTQ 量化，并引入 SWA 减少长上下文中的注意力计算开销，降低每步 speculative decoding 的成本。

### 🔍 相比现有方法的优势
| 方法 | 优势 |
|------|------|
| 单纯 PTQ | 速度快但精度下降明显（如 IFEval 不达标） |
| 传统 Speculative Decoding | 草稿模型未针对量化目标优化，acceptance length 较低 |
| 本文方法 | **兼顾速度、精度与部署可行性**：<br>• 6.978× 平均加速<br>• 所有质量指标均满足竞赛门槛<br>• 支持 W4A16 格式部署 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **训练数据来源**：
  - `Nemotron-Post-Training-Dataset-v2`（共 220K 和 400K 样本用于不同阶段）
  - 教师模型（Qwen3.5-4B）重新生成对话数据用于 QAD 与 drafter 训练
- **验证/测试基准**：
  - **MMLU-Pro**（≥0.621）、**IFEval**（≥0.814）、**GPQA-Diamond**（≥0.630）——用于评估模型质量
  - **GSM8K**, **HumanEval**, **LongBench v2**——用于测量平均接受长度（mean acceptance length）

### ⚙️ 实验设置
- **硬件平台**：
  - 官方评测环境：AWS g5.xlarge（NVIDIA A10G, 24GB VRAM）
  - 实际测试平台：NVIDIA RTX 5000 Ada Generation（用于消融实验）
- **输入长度配置**：
  - Short: 64/128（input/output tokens）
  - Medium: 2048/256
  - Long: 8192/256
- **评估方式**：
  - **Latency**：5 次预热 + 50 次实测取平均
  - **Acceptance Length**：64 prompts per benchmark，最大生成 256 tokens

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **BF16 Baseline** | 原始未优化的 Qwen3.5-4B 模型（FP16/BF16） |
| **INT4 PTQ (AWQ)** | 直接后训练量化至 INT4，无微调 |
| **Public DFlash Checkpoint** | 公开的 BF16 drafter + BF16 target 组合 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 指标 | Baseline | Ours |
|------|----------|------|
| **Avg. Speedup** | 1.000x | **6.978x** |
| Short (64/128) 延迟 | 2582ms | **232ms** |
| Medium (2048/256) 延迟 | 5441ms | **782ms** |
| Long (8192/256) 延迟 | 6576ms | **2313ms** |
| MMLU-Pro | 0.690 | 0.659 (**≥0.621**) |
| IFEval | 0.857 | 0.845 (**≥0.814**) |
| GPQA-Diamond | 0.700 | 0.667 (**≥0.630**) |

✅ 所有质量指标均高于最低阈值，排名第 3（共 40+ 团队）

---

### 🔬 消融实验结果（Table 2 & Table 4–6）

#### ✅ 各组件对加速的累积贡献（Table 2）
| 方法 | 平均加速比 | Long 输入延迟 |
|------|------------|----------------|
| BF16 Baseline | 1.00x | 5692ms |
| + INT4 Target (AWQ → QAD) | 2.16x | 2876ms |
| + BF16 Drafter (Two-stage) | 3.32x | 2522ms |
| + INT4 Drafter (GPTQ) | 3.55x | 2290ms |
| + Drafter with SWA | **3.57x** | **2027ms** |

> 注：最终提交版本通过更优工程实现达到 **6.978x** 加速，远超模块化累加效果，说明系统级协同优化重要。

#### ✅ Drafter 训练策略比较（Table 4）
| 训练方式 | 平均接受长度 |
|---------|---------------|
| Direct on INT4 target | 4.97 |
| Two-stage (BF16 → INT4) | **5.03** ✅ |

👉 两阶段训练略优于直接训练，表明从高精度模型迁移有助于稳定 draft 分布。

#### ✅ Drafter 量化影响（Table 5）
| 量化方法 | 平均接受长度 |
|--------|----------------|
| BF16 Drafter | 5.03 |
| INT4 RTN | 4.91 |
| INT4 AWQ | 4.92 |
| INT4 GPTQ | **4.98** ✅ |

👉 GPTQ 在保持最小性能损失的同时实现高效压缩。

#### ✅ SWA 对延迟的影响（Table 6）
| SWA 窗口大小 | Long 输入延迟 (W4) |
|-------------|--------------------|
| Full Attention | 2290.0ms |
| SWA 2048 | 1966.0ms |
| **SWA 1024** | **2026.5ms** ✅（最终选择） |
| SWA 512 | 2136.3ms |

👉 中等窗口（1024–2048）最优；过小窗口损害 context coverage 导致收益下降。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **QAD 可有效恢复量化模型精度**  
   在不改变原始 AWQ 量化网格的前提下，QAD 成功将 IFEval 从 0.8046 提升至 0.8285，满足质量门限。

2. **两阶段 drafter 训练优于直接训练**  
   利用 BF16 模型作为“教师”初始化，再迁移到 INT4 目标模型，可提升 draft token 的 alignment 与 acceptance length。

3. **Drafter 的轻量化至关重要**  
   由于 speculative decoding 每步都调用 drafter，其延迟直接影响整体性能。通过 **PTQ + SWA** 显著降低 drafter 开销，尤其在 long-context 场景下效果显著。

4. **系统级协同优化带来巨大加速增益**  
   尽管各模块累加仅约 3.57x 加速，但实际端到端优化实现了 **6.978x** 加速，体现工程实现与算法设计的深度耦合价值。

---

### ⚠️ 局限性
- **依赖高质量教师模型再生数据**：QAD 和 drafter 训练均需大量由教师模型生成的数据，增加训练成本。
- **drafter 需定制化训练**：无法即插即用现有草稿模型，必须针对特定目标模型（尤其是量化版本）专门训练。
- **硬件差异影响可复现性**：实验基于 RTX 5000 测评，而官方平台为 A10G，可能存在性能偏差。

---

### 🔮 未来工作方向
- 探索 **无需再训练的 zero-shot drafter 适配方法**，降低部署门槛。
- 研究 **动态滑动窗口机制**，根据输入复杂度自适应调整 SWA 窗口大小。
- 将该框架扩展至更多模型架构（如 Mixtral、Phi 系列）和任务场景（如检索增强生成 RAG）。
- 结合 **KV Cache 压缩** 与 **early exiting** 等技术进一步优化内存与计算效率。

---

> 🔗 **代码与资源公开**：https://github.com/nota-github/adaptfm-quant-dflash  
> 本文为高效 LLM 推理提供了实用且可复现的技术路径，具有较强的工业落地潜力。

</details>

---

### 13. [Online Linear Programming for Multi-Objective Routing in LLM Serving](https://arxiv.org/abs/2607.03948)

**Authors**: Zixi Chen, Yinyu Ye, Zijie Zhou  
**Category**: cs.AI  
**Published**: 2026-07-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.03948v1  

#### Abstract
We study the online routing problem in large language model serving, where requests arrive sequentially and must be dispatched to parallel decode workers under tight batch-size and KV-cache constraints. Unlike widely used routing heuristics that are not tied to explicit service-level objectives (SLO...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Online Linear Programming for Multi-Objective Routing in LLM Serving**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在大语言模型（LLM）服务中，请求需要被动态地路由到多个并行的解码设备（decode workers），而这些设备受到**批处理大小（batch-size）** 和 **KV-cache 内存容量** 的严格限制。传统的路由策略（如轮询、最短队列优先等）通常是基于启发式的（heuristic-based），缺乏对明确的服务级别目标（SLOs）的支持，难以灵活控制延迟（latency）、吞吐量（throughput）、首token时间（TTFT）和尾部性能之间的权衡。

本文旨在解决以下核心挑战：
- 如何在**在线、实时**场景下做出高效的路由决策；
- 如何统一优化多个相互冲突的 SLO 指标；
- 如何建模具有**时间耦合约束**（time-coupled constraints）的资源消耗（例如一个请求占用多步解码时间）。

---

### **提出了什么新方法或新思路**
作者提出了一种基于 **Online Linear Programming (OLP)** 的多目标路由框架，其核心思想是将路由问题建模为一个带有时序资源约束的在线线性规划问题，并通过**对偶理论**（duality）导出高效的“出价-价格”控制策略（bid-price control）。

#### **主要创新点包括：**

1. ✅ **多目标优化框架（Multi-objective Optimization Framework）**
   - 将多个 SLO（如平均/尾部 end-to-end latency、TTFT、throughput、QPS）统一建模为可解释的目标函数。
   - 引入**可调节权重参数**（如 α, β, γ, σ₁, σ₂），使系统操作员可以根据业务需求灵活调整不同指标间的权衡。

2. ✅ **基于影子价格的 Bid-Price 控制算法**
   - 利用 LP 对偶变量作为资源的**影子价格**（shadow prices），量化每个设备在未来时间段内 batch slot 和 KV-cache 的稀缺程度。
   - 路由决策规则：仅当请求的 SLO 加权收益超过其资源机会成本（即 `reward > shadow_cost`）时才接受该请求。

3. ✅ **高效在线更新机制：Warm-Started Projected First-Order Updates**
   - 避免每一步都求解完整的 LP，而是采用基于历史样本的 **Sample-Average Approximation (SAA)** 来估计对偶变量。
   - 使用 warm-started 投影梯度下降（projected subgradient descent）在线追踪影子价格变化，确保单次决策耗时在 **1–2ms** 内，满足 LLM 推理系统的毫秒级响应要求。

4. ✅ **尾部延迟的即时奖励分解**
   - 提出一种新颖的方法，将统计性质的尾部延迟（如 P99 TTFT）转化为 per-request 的指示函数奖励（indicator reward），从而可以直接嵌入在线优化目标中。

---

### **相比现有方法的优势**
| 维度 | 传统启发式方法（如 Round Robin, LOR, Power-of-2-Choices） | 本文方法 |
|------|--------------------------------------------------------|---------|
| **目标导向性** | 无显式优化目标，行为不可控 | 明确支持多目标优化，权重可调 |
| **SLO 控制能力** | 无法精细控制尾部延迟或特定指标 | 可通过权重系统性调节性能前沿（Pareto frontier） |
| **资源感知** | 忽略 KV-cache 或时间耦合特性 | 显式建模 batch 和 memory 的时序占用 |
| **计算效率** | 极快但“盲目” | 毫秒级运行，适合生产部署 |
| **鲁棒性** | 对负载突变敏感 | 支持滑动窗口重估，适应非平稳流量 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **真实数据集（Real Data）**
   - 使用来自 [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) 的对话请求数据。
   - 请求到达过程建模为泊松过程（Poisson process），到达率 λ ∈ {0.4, 0.5}（单位：requests/ms）。
   - 包含真实的 prompt 长度和 decode 长度分布。

2. **合成数据集（Synthetic Data）**
   - 控制 prefill-to-decode (P/D) 比例：
     - P/D = 1:4（decode-heavy）
     - P/D = 4:1（prefill-heavy / decode-light）
   - 用于测试不同 workload 特征下的算法表现。

---

### **实验设置**
- **硬件模拟环境**：4 × NVIDIA A100 GPUs
- **仿真平台**：集成至 **Vidur simulator**（一个高保真的 LLM 推理模拟器）
- **调度模式**：Prefill/Decode disaggregation + continuous batching
- **预测误差模拟**：
  - 完美预测：$\hat{o}_j = o_j$
  - 不完美预测：$\hat{o}_j \sim \text{Unif}(0.8o_j, 1.2o_j)$（±20% 误差）

---

### **评估指标**
| 类别 | 指标 |
|------|------|
| **服务级指标（SLO）** | - 平均 / 尾部 End-to-End Latency（EEL）<br>- 平均 / 尾部 Time-to-First-Token（TTFT） |
| **系统级指标** | - Throughput（总生成 token 数）<br>- Queries Per Second（QPS） |
| **综合性能** | - SLO Violation Rate（SLO 违规比例）<br>- 相对提升百分比（vs. baseline） |

---

### **基线方法对比**
所有基线均为 Vidur 默认实现的路由策略：
1. **Round Robin**：循环分配
2. **Least Outstanding Request (LOR)**：选择队列最短的设备
3. **Random**：随机分配
4. **Power-of-2-Choices**：随机选两个设备，挑负载更轻的一个

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table I & II）**

#### **表 I：稳定负载下（λ=0.4），噪声预测场景（$\hat{o}_j$ 有 ±20% 误差）**

| Method | Avg EEL↓ | P95 EEL↓ | P99 EEL↓ | Throughput↑ | SLO Viol.↓ |
|--------|----------|----------|----------|-------------|------------|
| LOR    | +0.67%   | +4.01%   | +6.19%   | +0.53%      | +0.98%     |
| Power-of-2 | +1.30% | +3.84%   | +6.06%   | +0.73%      | +1.84%     |
| **Ours** | **+45.75%** | **+42.49%** | **+25.85%** | **+0.90%** | **+11.10%** |

> 注：数值表示相对于 Round Robin 的相对改进百分比（负值为降低，正值为提升）。我们的方法在几乎所有指标上显著优于基线。

#### **表 II：非平稳负载（rate shift: λ 从 0.4 → 0.6）下的表现**

| Method | Avg EEL↓ | P95 EEL↓ | P99 EEL↓ | Throughput↑ | SLO Viol.↓ |
|--------|----------|----------|----------|-------------|------------|
| LOR    | +1.47%   | +6.72%   | +9.67%   | +1.48%      | +3.92%     |
| Power-of-2 | +1.80% | +6.35%   | +8.28%   | +1.39%      | +3.28%     |
| **Ours** | **+44.81%** | **+45.69%** | **+31.15%** | **+1.93%** | **+14.57%** |

> 在突发流量下，本文方法仍保持强大优势，尤其在尾部延迟控制方面远超启发式方法。

---

### **消融实验与关键观察**

1. **目标权重调节影响明显**
   - 当设置 `(α, β, γ, σ₁, σ₂) = (0,1,1,0,0)` 时，侧重平均延迟，EEL 和 TTFT 显著改善。
   - 当设置 `(0,0,0,0,1,1)` 时，侧重尾部延迟，P99 EEL 和 P99 TTFT 大幅下降。
   - 表明：**权重调整能系统性地引导性能沿 Pareto 前沿移动**。

2. **尾部权重直接影响达标率**
   - 图 13 显示：随着尾部权重 $σ_2$ 增加，TTFT ≤ 49 的请求占比单调上升。
   - 说明：**尾部权重不仅缩放目标，还能直接控制达成目标的请求比例（即有效设定目标分位数）**。

3. **对 decode-length 预测误差鲁棒**
   - 即使预测存在 ±20% 误差，性能增益依然显著。
   - 影子价格机制能够在线自适应调整，缓解预测偏差的影响。

4. **在 decode-heavy 场景下优势更大**
   - 在 P/D=1:4（长解码）场景中，性能提升显著；
   - 在 P/D=4:1（短解码）场景中，提升较小，因为资源约束不紧张。
   - 结论：**本方法最适合于解码密集型、资源受限、高并发的 LLM 服务场景**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **科学驱动的优化方法优于启发式策略**
   - “A science-based approach outperforms others based on heuristics.”
   - 通过 OLP + bid-price 控制，实现了对 SLO 的透明、可控、系统化管理。

2. ✅ **影子价格揭示瓶颈资源**
   - 批处理 slot 和 KV-cache 的影子价格可作为实时拥塞信号：
     - 高 arrival rate → batch slot 成为瓶颈（batch price 上升）
     - 长 decode length → memory 成为瓶颈（memory price 上升）
   - 为系统监控和容量规划提供诊断工具。

3. ✅ **多目标权衡变得可解释、可配置**
   - 操作员无需修改路由逻辑，只需调整目标权重即可改变行为偏好。
   - 支持 SLO-aware admission control：低价值请求可在过载时被拒绝。

4. ✅ **方法具备良好的扩展性和工程可行性**
   - 时间复杂度为 $O(KGH)$，线性于集群规模；
   - 实现延迟约 **1–2ms/step**，远低于典型 decode iteration（10–30ms on A100），不会成为瓶颈。

---

### **局限性**
1. ❗ **依赖仿真验证**
   - 当前实验基于 **Vidur simulator**，尚未在真实生产系统（如 vLLM）中部署。
   - 实际中可能面临 preemption、KV-cache swapping、分布式协调开销等问题。

2. ❗ **未考虑请求优先级与公平性**
   - 假设所有请求同等重要，未建模用户优先级、租户隔离或多类请求公平调度。

3. ❗ **预测误差虽鲁棒但仍有限**
   - 虽然对 ±20% 误差表现良好，但在极端误判下（如低估长序列）可能导致资源争用加剧。

---

### **未来工作方向**
1. 🔧 **集成至实际 Serving Engine**
   - 将算法集成进 vLLM、TensorRT-LLM 等主流推理引擎，进行端到端性能验证。

2. 🎯 **引入优先级与公平性约束**
   - 扩展目标函数以支持多优先级队列、SLA 分层保障、公平性正则项。

3. 🔄 **支持异构 GPU 集群**
   - 自然扩展至 heterogeneous fleet：为不同设备设置不同的 $(B_g, M_g)$ 容量参数。

4. 📈 **动态权重自动调优**
   - 开发反馈控制器，根据当前 SLO 达成情况自动调整目标权重，实现闭环 SLO 管理。

5. ⚙️ **探索更细粒度的 Prefill/Decode Mixing 模型**
   - 当前主要针对 PD disaggregation，未来可拓展至混合执行模式下的联合调度。

---

> **Impact Statement 总结**：  
> 本文首次将 **operations research 中的 online LP 与 bid-price control** 系统性应用于 LLM 路由问题，建立了首个支持多目标、可解释、资源感知的科学路由框架。它不仅提升了性能，更重要的是提供了**结构性洞察**（structural insights），有望推动 LLM serving 从“经验调参”迈向“原理驱动”的新时代。

</details>

---

### 14. [STAPO: Selective Trajectory-Aware Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2607.04963)

**Authors**: Qiuyi Qi, Tian Liang, Mutian Bao, Jinjian Zhang, Dongnan Liu, Wei Zhou, Linjian Mo, Ming Kong, Jie Liu, Feng Zhang, Qiang Zhu  
**Category**: cs.AI  
**Published**: 2026-07-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.04963v1  

#### Abstract
Reinforcement Learning (RL) is the dominant paradigm for training Large Language Model (LLM) agents on long-horizon tasks. However, sparse and delayed rewards often lead to trajectory neglect, in which agents lose focus on the task goal and interaction history at intermediate steps. Prior work has e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# STAPO: Selective Trajectory-Aware Policy Optimization for LLM Agent Training — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **轨迹忽视（Trajectory Neglect）**：在长视野任务中，由于奖励稀疏且延迟，LLM Agent 在中间步骤容易失去对任务目标和交互历史的关注，导致生成低质量动作。
- **现有不确定性度量不可靠**：传统基于 **Shannon Entropy** 的不确定性信号混淆了“状态本身的复杂性”与“模型自身的置信度”，无法准确识别因轨迹忽视导致的异常步骤。

### 🚀 提出的新方法与创新
1. **Normalized Entropy**  
   - 创新地提出一种新的不确定性度量方式：**Normalized Entropy**。
   - 它通过在同一锚定状态（anchor state）下聚合多个采样轨迹的动作分布，计算每个步骤的熵相对于该状态平均值的**标准化偏差**。
   - 有效解耦了“环境状态固有复杂性”与“模型信心不足”，从而更精准定位由轨迹忽视引起的异常步骤。

2. **Selective Trajectory-Aware Policy Optimization (STAPO)**  
   - 构建了一个分层的 group-based RL 框架，包含两个阶段：
     - **Outlier Localization**：利用 normalized entropy 动态识别异常步骤。
     - **Selective Optimization**：仅针对这些异常步骤进行优化，采用联合机制：
       - **Trajectory-Aware Reward (R_TA)**：鼓励模型依赖完整的历史上下文做出决策。
       - **Trajectory-Independent Penalty (P_TI)**：防止模型通过模式坍缩等方式“作弊”提升 R_TA。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | STAPO 改进 |
|------|--------|-----------|
| **PPO / RLOO / GRPO** | 仅依赖最终结果奖励，缺乏过程监督 | 引入细粒度过程优化机制 |
| **GiGPO / EMPG** | 对所有步骤无差别优化，未聚焦关键错误 | **选择性优化**，只处理轨迹忽视相关的 outlier 步骤 |
| **基于 Shannon Entropy 的方法** | 易将高复杂状态误判为低置信 | 使用 **normalized entropy** 提升检测准确性 |

> ✅ **核心优势总结**：STAPO 实现了“哪里出错，就优化哪里”的精细化训练策略，在不破坏整体稳定性的前提下显著缓解轨迹忽视问题。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 类型 | 任务描述 |
|-------|------|----------|
| **ALFWorld** | Embodied Control | 文本化模拟环境中完成多步操作（如“把冷却的碗放进柜子”），涵盖 Pick, Look, Clean, Heat, Cool, Pick2 六类任务 |
| **WebShop** | Web Navigation | 模拟电商网站购物，需搜索、浏览、筛选并购买指定商品，含 >1.1M 商品和 12k 用户指令 |
| **Search-Augmented QA** | 多跳问答 | 包括单跳（NQ, TriviaQA, PopQA）和多跳（HotpotQA, 2Wiki, MuSiQue, Bamboogle）任务，测试工具调用能力 |

### ⚙️ 实验设置与评估指标
- **Base Models**：Qwen2.5 系列（1.5B–32B）、Llama3.1-8B-Instruct
- **Group-based Sampling**：每轮 rollout 使用 N=5~8 条轨迹组成 group 进行优势估计
- **Anchor State Grouping**：
  - 结构化观测：精确匹配
  - 自由文本：最长子序列相似度 ≥0.9 聚类
- **评估指标**：
  - ALFWorld/WebShop：Success Rate (%) 和 Average Score
  - QA 任务：Exact Match (EM) 或 Accuracy
- **实现细节**：
  - 使用 IQR 法动态判定 outlier（系数 λ=1.5）
  - 两层优势设计：episode-level + step-level
  - 所有实验运行 3 个随机种子取均值

### 🆚 基线方法对比
| 类别 | 方法 |
|------|------|
| **Closed-Source** | GPT-4o, Gemini-2.5-Pro |
| **Prompting** | ReAct, Reflexion |
| **Outcome RL** | PPO, RLOO, GRPO |
| **Process RL** | EMPG, GiGPO |
| **Search-Specific** | Search-R1, ZeroSearch, StepSearch |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

#### ✅ ALFWorld（Qwen2.5-7B-Instruct）
| 方法 | Overall Success Rate |
|------|---------------------|
| GiGPO | 90.8 ± 1.3 |
| **STAPO** | **96.9 ± 1.6** ✅ |

> → 提升近 **6.1%**，接近完美表现（Pick2 达 100%）

#### ✅ WebShop（Qwen2.5-1.5B-Instruct）
| 方法 | Score | Succ. Rate |
|------|-------|------------|
| GiGPO | 83.1 | 65.0% |
| **STAPO** | **85.9** | **69.0%** ✅ |

> → 成功率提升 **4.0%**，绝对领先

#### ✅ Search-Augmented QA（Qwen2.5-7B-Instruct）
| 方法 | Avg. Performance |
|------|------------------|
| GiGPO | 47.2 |
| **STAPO** | **48.4** ✅ |

> → 平均提升 **1.2%**，尤其在多跳任务上稳健增益

> 💡 注：QA 任务提升较小，作者解释为“交互回合短，天然缓解轨迹忽视”。

---

### 🔬 消融实验结果（Table 3 & Figure 4–7）

| 变体 | ALFWorld | WebShop | 分析 |
|------|----------|---------|------|
| w/ Shannon Entropy | 89.6 | 67.2 | 性能下降 → 验证 normalized entropy 更优 |
| w/ None (no masking) | 86.7 | 65.0 | 移除 masking 损害效果 |
| w/ Goal Only | 89.0 | 66.6 | 历史信息更重要 |
| w/ History Only | 90.9 | 68.2 | 结合两者最佳 |
| w/ Single Penalty | 88.3 | 65.9 | 全局惩罚有害 |
| **STAPO (Full)** | **92.2** | **69.0** | ✅ 最佳组合 |

#### 其他关键发现：
- **IQR 系数敏感性分析（Figure 4）**：随着 outlier 判定越严格（λ↑），性能持续上升，说明“精准打击”优于全量优化。
- **权重系数分析（Figure 7）**：当 α=γ=0.005 时达到最优；过大则导致过拟合辅助目标。
- **归因分析（Figure 5）**：STAPO 显著降低 outlier 步骤比例，且成功率与 outlier 数量呈负相关。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **轨迹忽视是长视野任务失败的关键原因**，表现为偏离目标或重复动作（见 Appendix A 案例）。
2. **Normalized Entropy 是一个可靠的 outlier 检测器**，能有效区分“真不确定”与“假复杂”。
3. **选择性优化优于全局优化**：只对异常步骤施加 trajectory-aware 监督，既增强关注力又保持训练稳定性。
4. **STAPO 具备良好泛化性**：
   - 可集成到 DAPO 等先进框架中（Table 4）
   - 在不同架构（Llama3.1, Qwen2.5）和规模（1.5B–14B）上均有效（Table 5）
   - 扩展至 Vision-Language Agent 也取得 SOTA（Table 11）

### ⚠️ 方法的局限性
1. **计算开销略增**：需额外前向传播计算 masked prompt 输出，带来约 **5.7% 时间开销**（Table 8），但远低于扩大 group size 的成本。
2. **Batch 内状态覆盖要求**：若某 anchor state 在 batch 中仅出现一次，则无法进行组内归一化，退化为原始 Shannon Entropy。
3. **超参敏感性存在边界风险**：过高的 α/γ 权重可能导致主任务被压制（Figure 7）。
4. **当前验证集中于文本 Agent**：虽初步尝试 VLM（Table 11），但在复杂视觉界面下的行为仍需深入研究。

### 🔮 未来工作方向
- 探索更高效的 outlier 检测机制（如在线估计状态统计量）
- 将 STAPO 应用于真实世界 GUI 控制、机器人控制等多模态场景
- 结合 memory 或 reflection 机制进一步强化长期记忆
- 研究如何在低资源或小批量设置下维持 normalized entropy 的有效性

---

## ✅ 总结一句话
> **STAPO 通过提出 normalized entropy 和构建 selective 优化机制，首次实现了对“轨迹忽视”的精准识别与定向修复，在 ALFWorld、WebShop 和 Search-Augmented QA 上全面超越 SOTA，为长视野 LLM Agent 训练提供了高效且鲁棒的新范式。**

</details>

---

### 15. [WPG-MoE: Weak-Prior-Guided Dense Mixture-of-Experts for User-Level Social Media Depression Detection](https://arxiv.org/abs/2607.04350)

**Authors**: Xian Li, Yuanhe Tian, Yang Yang, Guoqing Wang, Yan Song  
**Category**: cs.CL  
**Published**: 2026-07-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.04350v1  

#### Abstract
Online social media posts provide scalable signals for early depression screening, and recent studies mainly improve pre-classification evidence through risk-post selection, symptom grounding, and clinically informed feature construction. However, these screening-stage designs often leave final deci...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# WPG-MoE: Weak-Prior-Guided Dense Mixture-of-Experts for User-Level Social Media Depression Detection 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于社交媒体的**user-level depression detection**方法虽然在证据提取阶段（如风险帖筛选、症状建模）取得了进展，但最终决策仍依赖于单一分类器（single detector）。这种“单体模型”（monolithic classifier）对异质性用户表达的抑郁风险信号进行平均化处理，导致以下问题：
- **稀疏证据被稀释**：非自我披露（non-self-disclosing）用户的微弱、零散但高强度的风险信号容易被忽略。
- **模式混淆**：不同类型的证据布局（如持续性症状、孤立危机帖、混合表达）被压缩到同一表示空间，造成边界模糊。

### 提出的新方法与创新
作者提出 **WPG-MoE**（Weak-Prior-Guided Dense Mixture-of-Experts），其核心创新如下：

#### （1）引入 **Dense Mixture-of-Experts (MoE)** 架构
- 在共享的 **LLM backbone** 上构建多个专家（expert），每个专家专注于处理特定类型的证据布局（如 self-disclosure, episode-supported, sparse-evidence）。
- 采用**软路由**（soft routing）机制，允许用户被多个专家共同处理，实现细粒度的条件专业化（conditional specialization）。

#### （2）使用 **Weak Priors** 进行路由引导
- 不直接使用硬标签（如临床亚型）进行监督，而是从外部 LLM 提取的丰富结构化证据中导出**弱语义先验**（weak semantic priors），用于指导训练时的路由学习。
- 这些先验包括：自我披露倾向、持续性症状支持、稀疏高危证据强度等。

#### （3）采用 **Learning Using Privileged Information (LUPI)** 范式
- **训练时**：利用 LLM 提取的结构化信息（Path A）作为“特权信息”（privileged information）来构建 weak priors 和辅助监督。
- **推理时**：仅保留轻量级的 **PHQ-9 模板匹配**（Path B）和共享 backbone，确保部署高效且无需实时调用外部 LLM。
- 实现了**训练增强、推理简洁**的理想平衡。

### 相比现有方法的优势
- **更鲁棒地捕捉异质性表达**：通过多专家分工，避免单一模型对多样化证据的平均化损失。
- **可解释性强**：路由权重（gate weights）反映了模型对不同类型证据的依赖程度。
- **部署友好**：推理不依赖昂贵的 LLM 调用，仅需模板匹配和共享 backbone。
- **性能优越**：在多个数据集上显著超越强基线。

---

## 2. 核心实验方法和设置

### 数据集
实验在三个文本为主的用户级抑郁症检测数据集上进行：
| 数据集 | 语言 | 抑郁用户数 | 控制组用户数 | 特点 |
|--------|------|------------|--------------|------|
| **SWDD** | 中文 | 3,711 | 19,526 | 来自中国社交媒体，经人工修正标签噪声 |
| **Twitter** | 英文 | 1,218 | 1,273 | 经典英文数据集，历史较长 |
| **eRisk25** | 英文 | 102 | 807 | 风险早期检测任务，帖子密集 |

所有实验采用统一的 **80/10/10 分层划分**（stratified holdout），支持跨数据集迁移比较。

### 评估指标
- **Recall**（召回率）
- **F1**（F1分数）
- **AUROC**（ROC曲线下面积）
- **AUPRC**（PR曲线下面积，尤其适用于类别不平衡场景）

报告五次运行的均值。

### 基线方法（共7个）
分为四类：
1. **PHQ-9 模式匹配**：`Pattern (threshold)`、`Pattern (CNN)`
2. **精神科量表引导筛选**：`HAN-BERT(Psych)`、`Bert(Clus+Abs)`、`E2-LPS`
3. **症状结构化表示学习**：`DeCapsNet`
4. **LLM 辅助临床检测**：`DORIS`

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 4）
在**同域测试**（in-domain）下，WPG-MoE 取得全面领先：

| 模型 | SWDD → SWDD (F1/AUPRC) | Twitter → Twitter (F1/AUPRC) | eRisk25 → eRisk25 (F1/AUPRC) |
|------|-------------------------|-------------------------------|------------------------------|
| **WPG-MoE (Ours)** | **0.7490 / 0.8300** | **0.7630 / 0.8520** | **0.7440 / 0.7350** |
| 最佳基线 | 0.7436 / 0.8092 (`Bert(Clus+Abs)`) | 0.7630 / 0.8520 (`DeCapsNet`) | 0.7440 / 0.7350 (`DORIS`) |

> 注：在 `Twitter→Twitter` 和 `eRisk25→eRisk25` 上，WPG-MoE 与最佳基线持平或略优；但在 `SWDD→SWDD` 上显著领先。

### 跨数据集迁移表现
- 当以 **SWDD** 为源训练时，WPG-MoE 在迁移到 Twitter 和 eRisk25 时表现出更强的泛化能力。
- 表明大规模、多样化的中文数据结合 WPG-MoE 架构能有效提升跨域鲁棒性。

### 消融实验（Ablation Study, Table 6）
在 SWDD 上的关键组件消融结果（F1 下降）：

| 移除组件 | SWDD (F1↓) | Twitter (F1↓) | eRisk25 (F1↓) |
|----------|-----------|-------------|--------------|
| **w/o MoE**（移除 MoE） | -0.1472 | -0.1733 | -0.1738 |
| **w/o Path A**（无特权信息） | -0.0767 | -0.0697 | -0.0746 |
| **w/o Weak Priors**（无弱先验） | -0.0203 | -0.0291 | -0.0312 |
| **w/o Route Loss**（无路由损失） | -0.0078 | -0.0144 | -0.0165 |
| **w/o DP Dropout**（无双重路径Dropout） | -0.1096 | -0.1155 | -0.1187 |

**结论**：
- **Dense MoE** 是最关键的组件，移除后性能下降最大。
- **Path A** 和 **DP Dropout** 对性能也至关重要，验证了 LUPI 设计的有效性。
- 弱先验和路由损失虽影响较小，但稳定提升了路由质量。

---

## 4. 关键结论和发现

### 主要发现
1. **用户级抑郁检测存在显著的证据异质性**：不同用户通过自我披露、持续症状、稀疏危机帖等方式表达风险，单一模型难以兼顾。
2. **WPG-MoE 能有效保留并利用这种异质性**：通过弱先验引导的软路由，模型将用户分配给最匹配的专家，避免证据稀释。
3. **LUPI 范式成功桥接训练与部署鸿沟**：训练时利用 LLM 提供的丰富结构信息，推理时仅依赖轻量级 PHQ-9 模板，实现了高性能与高效率的统一。
4. **案例分析证实优势**：对于稀疏证据用户，传统模型可能漏检，而 WPG-MoE 通过激活 **Sparse-evidence Expert** 成功识别；对于持续症状用户，**Episode-supported Expert** 起主导作用。

### 方法的局限性
- **依赖高质量的弱先验构建**：若 LLM 提取的结构化信息有偏或错误，可能误导路由学习。
- **专家数量固定**：当前设计为 5 个专家（含全局专家），未探索动态扩展。
- **中文数据依赖较强**：在 SWDD 上优势明显，但在其他语言上的普适性有待进一步验证。

### 未来工作方向
- 探索更灵活的专家生成机制（如动态 MoE）。
- 将该框架应用于其他心理健康问题（如焦虑、自杀倾向）的检测。
- 研究如何减少对 LLM 提取先验的依赖，例如通过自监督方式学习证据布局。
- 在真实临床场景中进行前瞻性验证与部署测试。

---

> **总结**：WPG-MoE 通过 **Weak-Prior-Guided Dense MoE + LUPI** 的创新架构，有效解决了用户级抑郁检测中的**证据异质性**难题，在保持部署效率的同时实现了性能与可解释性的双重提升，为心理健康 AI 检测提供了新的范式。

</details>

---

### 16. [Heterogeneous Graph Condensation via Role-Aware Clustering](https://arxiv.org/abs/2607.03097)

**Authors**: Fuyan Ou, Yulin Hu, Ye Yuan  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.03097v1  

#### Abstract
Heterogeneous Graph Neural Networks (HGNNs) have exhibited remarkable efficacy in modeling complex systems with multiple types of nodes and relations, yet their training on large-scale heterogeneous graphs remains computationally prohibitive. Although graph condensation methods can effectively impro...

---

### 17. [Transformers with Physics-Informed Encodings and Simulation-Based Inference for Robust Detection of Eccentric Binary Black Holes in Pulsar Timing Array Data](https://arxiv.org/abs/2607.03904)

**Authors**: Subhajit Dandapat, Alvin J. K. Chua  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.03904v1  

#### Abstract
Pulsar timing arrays (PTAs) provide a unique window into nanohertz gravitational waves (GWs), but extracting astrophysical parameters from noisy, long-baseline timing residuals remains computationally challenging with traditional Bayesian techniques due to the high dimensionality of the parameter sp...

---

### 18. [NKI-Agent: Domain-Specific Fine-Tuning and Agentic Tool Use for Neuron Kernel Generation](https://arxiv.org/abs/2607.04395)

**Authors**: Junjie Tang, Jun Huan, Hao Zhou, Yuhao Zhang, Lin Wang  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.04395v1  

#### Abstract
Recent agentic approaches to LLM-based kernel generation have achieved impressive results on CUDA. For emerging AI accelerators such as AWS Trainium and Inferentia, automated kernel generation and optimization remain largely unaddressed. Writing kernels for these chips via the Neuron Kernel Interfac...

---

### 19. [F-ACVAE: A Federated Adaptive Conditional Variational Auto-Encoder for Privacy-Preserving Intrusion Detection in IoT Networks](https://arxiv.org/abs/2607.04698)

**Authors**: Mohammad Ansarimehr, Somayeh Changiz, Ehsan Baghishani, Ali Mousavi  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.04698v1  

#### Abstract
The rapid proliferation of Internet of things (IoT) devices has significantly expanded the cyber-attack surface, necessitating robust and privacy-preserving intrusion detection systems (IDS). However, centralized learning approaches often suffer from severe performance degradation due to high-dimens...

---

### 20. [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391)

**Authors**: Jacky Kwok, Shulu Li, Pranav Atreya, Yuejiang Liu, Yixing Jiang, Chelsea Finn, Marco Pavone, Ion Stoica, Azalia Mirhoseini  
**Category**: cs.AI  
**Published**: 2026-07-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.05391v1  

#### Abstract
Scaling pre-training, post-training, and test-time compute have become the central paradigms for improving the capabilities of LLMs. In this work, we identify verification, the ability to determine the correctness of a solution, as a new scaling axis. To unlock this and demonstrate its effectiveness...

---

### 21. [Tile-Level Activation Overlap for Efficient LLM Inference](https://arxiv.org/abs/2607.02521)

**Authors**: Abhinav Jangda, Tyler Sorensen, Sebastian Burckhardt, Jianlan YE, Chaoyin Li, Atul Gupta  
**Category**: cs.DC  
**Published**: 2026-07-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.02521v1  

#### Abstract
SwiGLU is the dominant MLP activation in modern large language models, yet its intermediate tensor materialization costs 9-37% of MLP execution time. We present two complementary CUTLASS-based SM90 kernels that fuse SwiGLU into GeMM at the tile level. Kernel-1 overlaps Swish computation on the Gate ...

---

### 22. [Tensor-Train Joint Modeling for Few-Step Discrete Diffusion](https://arxiv.org/abs/2607.03788)

**Authors**: Byoungkwon Kim, Minhyuk Sung  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.03788v1  

#### Abstract
Discrete diffusion promises orders-of-magnitude faster generation than autoregressive (AR) models for sequential discrete data, yet its full potential of few-step generation has remained out of reach due to a fundamental structural limitation. The conditional-independence assumption underlying curre...

---

### 23. [SMART: A Machine Learning and Monte Carlo Framework for Rapid Analysis of Stochastic Transistor Aging and Process Variation in Digital Circuits](https://arxiv.org/abs/2607.05187)

**Authors**: Arash Esshaghi, Siavash Es'haghi, Gholamreza Shahabadi, Alireza Moradi  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.05187v1  

#### Abstract
As CMOS technology scales into the deep nanometer regime, digital circuit reliability is increasingly threatened by the combined stochastic effects of Bias Temperature Instability (BTI) and Process Variation (PV). Traditional reliability analysis methods, which rely on computationally intensive simu...

---

### 24. [MentalThink: Shaping Thoughts in Mental SVG World](https://arxiv.org/abs/2607.03530)

**Authors**: Kangheng Lin, Jisheng Yin, Dingming Li, En Yu, Yana Wei, Han Zhou, Liang Zhao, Hongyu Zhou, Hongbo Peng, Jianjian Sun, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Jingyu Wang  
**Category**: cs.AI  
**Published**: 2026-07-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.03530v1  

#### Abstract
We introduce MentalThink, a visual-symbolic reasoning paradigm that equips Multimodal LLMs (MLLMs) with an executable mechanism for "mental" visualization. The core of MentalThink is a think-with-SVG pipeline, where the model learns to generate, render, and interpret scalable vector graphics (SVG) c...

---

### 25. [Can Dialects Be Steered Like Languages? Sparse Neurons and Distributed Directions in Arabic LLMs](https://arxiv.org/abs/2607.03936)

**Authors**: Kareem Elozeiri, Mervat Abassy, Omar Kallas, Fahim Dalvi, Preslav Nakov, Kentaro Inui, Nadir Durrani  
**Category**: cs.CL  
**Published**: 2026-07-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.03936v1  

#### Abstract
A key challenge in Arabic NLP is the scarcity of dialectal data relative to Modern Standard Arabic (MSA), causing LLMs to overproduce MSA and struggle with dialectally accurate generation. From an interpretability perspective, this raises a fundamental question: where and how are dialectal features ...

---

### 26. [Reduced-Order Models: The Mother of World Models](https://arxiv.org/abs/2607.03198)

**Authors**: Rajat Ghosh  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.03198v1  

#### Abstract
World models -- compressed latent representations of an environment that support action-conditioned prediction and planning -- are typically presented as a product of modern self-supervised learning. This paper argues that the functional anatomy of a world model was independently developed, deployed...

---

### 27. [Explainable Reinforcement Learning for Adaptive Traffic Signal Control](https://arxiv.org/abs/2607.03703)

**Authors**: Dickens Kwesiga, Nishu Choudhary, Angshuman Guin, Michael Hunter  
**Category**: cs.AI  
**Published**: 2026-07-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.03703v1  

#### Abstract
Reinforcement Learning (RL) has emerged as a powerful paradigm for adaptive traffic signal control. However, in safety-critical infrastructure like traffic control, the opaque, black-box nature of deep RL models poses challenges for transportation agency acceptance, regulatory compliance, operationa...

---

### 28. [Folding, Reasoning, and Scaling with Open-source Drug Discovery Engine](https://arxiv.org/abs/2607.03787)

**Authors**: Aureka AI OpenDDE project  
**Category**: cs.AI  
**Published**: 2026-07-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.03787v1  

#### Abstract
Accurately modeling biomolecular interactions is a central bottleneck in biology and therapeutic discovery. Here, we introduce Open Drug Discovery Engine (OpenDDE), an open-source, all-atom biomolecular foundation model that uses co-folding as the entry point to a scalable AI-driven drug discovery e...

---

### 29. [TacReasoner: A Dynamic Tactile-Language Framework for Interactive Reasoning in Real-World Scenarios](https://arxiv.org/abs/2607.05131)

**Authors**: Kailin Lyu, Di Wu, Long Xiao, Jianning Zeng, Jianwei He, Chang Lin, Lianyu Hu, Lin Shu, Jie Hao, Ce Hao  
**Category**: cs.AI  
**Published**: 2026-07-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.05131v1  

#### Abstract
Among the five primary human senses, tactile is arguably the most fundamental to survival, as it enables the perception of physical contact and interaction in real-world environments. In this paper, we explore two key challenges of integrating tactile sensing into intelligent systems for multimodal ...

---

### 30. [Less Tokens, Better Forecasts: Sparse Residual Routing for Efficient Weather Prediction](https://arxiv.org/abs/2607.02829)

**Authors**: Janet Wang, Yunbei Zhang, Lin Zhao, Xi Xiao, Jihun Hamm, Xiao Wang  
**Category**: cs.LG  
**Published**: 2026-07-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.02829v1  

#### Abstract
Existing ViT-based weather forecasting models apply uniform computation across all spatial tokens, even though nearby atmospheric grid points often contain similar values and large regions evolve smoothly over time. This makes much of the intermediate per-token computation redundant. Standard token-...

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

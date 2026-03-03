# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-03 06:34:48 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verification](https://arxiv.org/abs/2603.01399)

**Authors**: Guang Huang, Zeyi Wen  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2603.01399v1  

#### Abstract
Speculative Decoding (SD) has emerged as a premier technique for accelerating Large Language Model (LLM) inference by decoupling token generation into rapid drafting and parallel verification. While recent advancements in self-speculation and lookahead decoding have successfully minimized drafting o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verification》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Speculative Decoding (SD)** 和 **Self-Speculative Decoding** 中，尽管 drafting 阶段已被有效加速（如通过跳层、n-gram 匹配等），但 **verification 阶段** 成为了新的性能瓶颈。该阶段需要对候选 token 序列执行完整的 **full-precision forward pass**，受限于 **memory bandwidth**，导致整体加速效果受限。

### 提出了什么新方法或新思路
提出 **Quasar** —— 一种无需训练的 **Quantized Verification** 框架，其核心思想是：
- 在 verification 阶段使用 **W8A8 量化版本的目标模型** 替代 full-precision 模型进行验证。
- 利用 **SmoothQuant 的增强变体** 处理激活中的 outlier，实现鲁棒的低比特推理。
- 保持 drafting 阶段不变（如使用 Ngram 或其他 self-speculative 方法），仅加速 verification。

### 相比现有方法的优势
| 维度 | Quasar | 传统方法 |
|------|--------|----------|
| **目标环节** | 加速 verification | 主要优化 drafting |
| **精度假设** | 允许 verifier 为 W8A8 | 要求 verifier 必须为 BF16/FP16 |
| **正交性** | 与 drafting 策略正交，可插拔集成 | 往往依赖特定 drafter 结构 |
| **内存效率** | 减少 50% 权重加载量，显著降低 memory traffic | full-precision 加载，带宽压力大 |

> ✅ **核心洞见**：现代 post-training quantization 技术已足够成熟，**W8A8 模型足以作为高保真 verifier**，从而打破“verification 是 memory wall”的限制。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
在五个典型任务上进行评估，覆盖多类下游场景：
- **MT-bench**：多轮对话能力
- **HumanEval**：代码生成
- **GSM8K**：数学推理
- **Alpaca**：指令遵循
- **CNN/Daily Mail**：文本摘要

### 实验设置和评估指标

#### 模型
- **Qwen3-8B**
- **OpenPangu-7B**

#### 硬件平台
- 单卡 **Ascend 910B2 (64GB) NPU**
- 使用 **vLLM-Ascend** 推理引擎 + **NCLL** 实现 INT8 计算

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Speedup** | 端到端吞吐提升倍数（relative to vanilla autoregressive） |
| **Mean Acceptance Length (L)** | 平均连续接受的 draft token 数量，反映 generation quality |
| **End-to-end Throughput** | 总体推理速度 |
| **Accuracy (Acc)** | 在 MMLU、CEval 等基准上的任务表现，验证无损性 |

#### 温度设置
- **T=0**：greedy decoding
- **T=1**：stochastic sampling，测试鲁棒性

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Figure 2）

| 模型 | 方法 | 平均 Speedup | 最高 Speedup（GSM8K） | 平均 L |
|------|------|--------------|------------------------|--------|
| Qwen3 | Ngram (BF16) | 1.18× | 1.43× | 1.33 |
| Qwen3 | **Quasar (W8A8)** | **1.28×** | **1.64×** | **1.40** |
| OpenPangu | Ngram (BF16) | 1.08× | 1.23× | 1.27 |
| OpenPangu | **Quasar (W8A8)** | **1.13×** | **1.26×** | **1.29** |

> 🔥 在 **GSM8K（强推理任务）** 上达到 **最高 1.64× 的端到端加速**，说明 memory-bound 场景下收益最大。

### 与基线方法的对比结果

#### 基线方法
- **Vanilla (Auto-regressive)**：标准逐 token 生成
- **Ngram (Self-Speculative)**：基于 prompt lookup 的 drafting + full-precision verification

#### 对比结论
- Quasar 在所有任务上均优于 Ngram 基线，且：
  - **Speedup 更高**：平均高出约 0.1–0.2×
  - **Acceptance Length 更长或持平**：表明未因量化损失质量
  - **尤其在 memory-intensive 任务中优势明显**（如 GSM8K）

#### 不同温度下的鲁棒性（Table 2）
| 温度 T | Ngram Speedup | Quasar Speedup |
|-------|---------------|----------------|
| 0.0   | 1.18×         | 1.28×          |
| 0.6   | 1.15×         | 1.25×          |
| 1.0   | 1.15×         | 1.23×          |

→ 表明 Quasar 在 **高熵采样场景下依然稳定有效**。

---

### 消融实验结果

#### (1) Structural Pruning vs. Quantization（Table 5）
尝试用 **layer pruning（保留 90%/75%/50% 层）** 构建轻量 verifier，结果失败：
| 方法 | L | Speedup |
|------|----|---------|
| Pruned-90% | 1.62 | 0.80× |
| Pruned-75% | 1.27 | 0.68× |
| Pruned-50% | 1.03 | 0.62× |
| **Quasar (W8A8)** | **1.40** | **1.28×** |

> ❌ **Pruning 失败原因**：
> - 浅层模型分布偏移严重 → acceptance rate 下降
> - 深度减少破坏 residual stream → feature gap 扩大

> ✅ **Quantization 成功原因**：
> - 保持完整网络拓扑结构
> - 均匀噪声不影响 top-1 预测稳定性

#### (2) Draft Length 敏感性分析（Table 3）
- 最优 speculative token 数 `y` 存在峰值（如 y=5 时达 1.47×）
- 过长的 draft 导致 verification 开销超过增益 → speedup 下降
- 提示：需根据模型能力和硬件平衡选择 `y`

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Verification 是当前 speculative decoding 的主要瓶颈**，特别是在 memory-bound 设置下。
2. ✅ **W8A8 量化 verifier 可以高保真地替代 full-precision verifier**，logit 分布对齐良好。
3. ✅ **Quantized Verification 显著降低 memory traffic（约 50%）**，直接转化为 **1.28× 端到端吞吐提升**。
4. ✅ **相比 structural pruning，quantization 更适合 verifier 加速**——结构完整性比数值精度更重要。
5. ✅ **方法正交于 drafting 策略**，可无缝集成进各类 self-speculative 框架（如 Ngram, Medusa, EAGLE）。

### 方法的局限性
1. **依赖 INT8 硬件支持**：当前实现基于 Ascend 的 INT8 tensor core，在缺乏专用硬件的设备上可能难以部署。
2. **极端低比特尚未验证**：目前仅验证 W8A8，更低比特（如 W4A4）可能导致 logit 分布失真。
3. **复杂推理任务中 acceptance rate 可能轻微下降**：虽然总体影响小，但在 chain-of-thought 类任务中需进一步观察。

### 未来工作方向
1. **Ultra-low Bit Verification**：探索 W4A4 或混合精度方案，进一步压缩 bandwidth。
2. **Dynamic Precision Scaling**：根据 draft confidence 动态切换 verification 精度（high-precision for uncertain tokens）。
3. **Hardware-Aware Optimization**：针对不同 NPU/GPU 架构定制 kernel，最大化 INT8 GEMM 效率。
4. **Integration with Tree-based Speculation**：将 Quasar 与 Medusa/EAGLE 等树形 speculative 方法结合，拓展至非线性验证场景。

---

> 📌 **一句话总结**：  
> **Quasar 通过引入 quantized verification 打破了 speculative decoding 中的 memory wall，实现了高达 1.28× 的端到端加速，同时保持 generation quality，为高效 LLM inference 提供了一条通用、实用的新路径。**

</details>

---

### 2. [Trident: Adaptive Scheduling for Heterogeneous Multimodal Data Pipelines](https://arxiv.org/abs/2603.02075)

**Authors**: Ding Pan, Zhuangzhuang Zhou, Long Qian, Binhang Yuan  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.02075v1  

#### Abstract
The rapid adoption of large language models and multimodal foundation models has made multimodal data preparation pipelines critical AI infrastructure. These pipelines interleave CPU-heavy preprocessing with accelerator-backed (GPU/NPU/TPU) inference and produce massive intermediate artifacts. Achie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Trident: Adaptive Scheduling for Heterogeneous Multimodal Data Pipelines**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代多模态数据处理流水线（如用于训练大语言模型 LLM 和多模态基础模型 MFMs 的文档、图像、视频处理）面临以下挑战：
- **高度非平稳的工作负载**：输入数据特征（如文档长度、视频分辨率）导致执行时间和资源消耗剧烈波动。
- **异构算力调度复杂**：流水线混合了 CPU 密集型操作（如解析、过滤）和 AI 加速器（GPU/NPU/TPU）上的推理任务，且中间数据量巨大。
- **动态批处理与异步执行**：LLM 推理通常采用连续批处理（continuous batching），使得传统基于“有效时间”（useful time）的吞吐估计失效。
- **内存瓶颈与 OOM 风险**：配置优化（如增大 batch size）可能提升吞吐但引发 Out-of-Memory（OOM）崩溃。
- **现有调度器局限**：多数系统依赖阈值触发的自动扩缩容（autoscaling），假设同步执行、忽略通信开销，无法应对上述复杂性。

### 🚀 提出的新方法：TRIDENT
TRIDENT 是一个面向固定资源集群的自适应调度框架，集成三个紧密耦合的闭环控制层：

#### （1）**Observation Layer（观测层）**
- 使用 **Gaussian Process (GP) 回归** 建模算子吞吐与工作负载特征的关系。
- 引入 **两阶段异常过滤机制**：
  - 第一阶段：基于利用率（GPU/CPU）、队列长度等运行时信号剔除非稳态样本（如上游饥饿、下游积压）。
  - 第二阶段：基于 GP 模型残差进行模型级异常检测，排除瞬时干扰。
- 输出：对每个算子的**可持续吞吐能力**（sustainable throughput）的鲁棒估计。

#### （2）**Adaptation Layer（适配层）**
- 通过 **在线聚类（incremental clustering）** 检测工作负载模式漂移（regime shift）。
- 使用 **Memory-Constrained Bayesian Optimization（MCBO）** 在安全前提下搜索最优配置（如 batch size, seq length）。
  - 将峰值设备内存建模为黑盒约束，避免 OOM。
  - 以“可行性概率”（Probability of Feasibility, PoF）引导探索，平衡性能与安全性。

#### （3）**Scheduling Layer（调度层）**
- 构造 **Mixed-Integer Linear Program (MILP)** 联合优化：
  - 算子并行度（parallelism）
  - 实例部署位置（placement）
  - 配置迁移策略（rolling updates）
- 显式建模异构资源（CPU/GPU/内存）、网络带宽限制，并最小化跨节点传输。
- 决策时权衡预期吞吐增益与冷启动开销（cold-start overhead），仅在收益大于代价时执行滚动更新。

#### 🔁 闭环反馈机制
三层形成**闭环控制系统**：
- 观测层输出作为调度层 MILP 和适配层 BO 的输入。
- 当调度层决定切换配置后，会通知观测层**清空旧配置的历史样本**，触发模型重新初始化，确保状态一致性。

### ⭐ 相比现有方法的优势
| 维度 | 传统方法缺陷 | TRIDENT 改进 |
|------|---------------|----------------|
| 吞吐建模 | 依赖同步假设，useful-time 不适用于异步批处理 | 使用 GP + 双重滤波，准确估计可持续吞吐 |
| 配置调优 | 静态配置或离线调优，不适应运行时变化 | 在线聚类 + 内存约束 BO，动态适应负载漂移 |
| 资源调度 | 忽略放置影响，独立扩缩各算子 | 联合优化并行度、放置、迁移，考虑全局约束 |
| 安全性 | 探索易触发 OOM，造成服务中断 | MCBO 显式规避高风险配置，保障稳定性 |

---

## 2. 核心实验方法和设置

### 📚 数据集
使用两个生产级代表性多模态流水线：

| 流水线 | 类型 | 数据规模 | 特征 |
|--------|------|----------|------|
| **PDF Pipeline** | 文档处理 | ~200k 文档 | 学术论文、年报、财报三类，依次处理，构成 workload shift |
| **Video Pipeline** | 视频清洗 | ~410k 视频片段 | 短视频（10–30s, ≤720p）和长视频（5–10min, 1080p–4K）交替处理 |

### 💻 实验环境
- **集群配置**：8 台服务器，每台含 8 × Ascend 910B NPU、256 CPU 核心、1TB 内存，100Gbps 网络互联。
- **实现平台**：基于 **Ray Data v2.46.0** 扩展实现 TRIDENT。

### 📊 评估指标
- **端到端吞吐（End-to-end throughput）**：原始输入记录/秒（normalized to Static baseline）
- **MAPE（Mean Absolute Percentage Error）**：观测层吞吐估计准确性
- **OOM 事件数 & 累计宕机时间**：验证内存安全性
- **消融实验**：分析各组件贡献
- **MILP 求解耗时**：评估系统开销

### 🆚 基线方法对比
| 方法 | 描述 | 局限性 |
|------|------|--------|
| **Static** | 手动调优静态资源配置 | 无任何运行时适应能力 |
| **Ray Data** | 默认阈值驱动 autoscaler | 按算子独立扩缩，无视全局资源竞争 |
| **DS2** | 基于 useful-time 的速率估计 + 并行度推导 | 假设同步执行，对异步算子建模错误 |
| **ContTune** | 在 DS2 基础上引入保守贝叶斯优化 | 仍为 per-operator 优化，缺乏全局视图 |
| **SCOOT** | 对 LLM 推理引擎参数进行离线 BO 调优 | 固定配置，无法响应运行时变化，无跨算子调度 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（RQ1）
在两种流水线上，TRIDENT 相比 Static 基线显著提升吞吐：

| 方法 | PDF Pipeline | Video Pipeline |
|------|-------------|---------------|
| **TRIDENT** | **2.01×** | **1.88×** |
| SCOOT | 1.21× | 1.17× |
| ContTune | 1.04× | 0.96× |
| Ray Data | 1.12× | 1.18× |
| DS2 | 0.87× | 0.79× |

> ✅ **TRIDENT 分别达到 2.01× 和 1.88× 的端到端吞吐加速**，远超所有基线。

### 🔍 控制变量比较（RQ2）
当所有方法共享 TRIDENT 的观测层和适配层输出时，仅比较调度层效果：

| 方法 | PDF Pipeline | Video Pipeline |
|------|-------------|---------------|
| **TRIDENT** | **2.01×** | **1.88×** |
| TRIDENT (all-at-once) | 1.92× | 1.79× |
| ContTune | 1.42× | 1.36× |

> ✅ 即使输入相同，TRIDENT 调度层仍大幅领先，说明其 **MILP 联合优化机制是主要优势来源**。
> ✅ 滚动更新带来约 **5% 额外增益**，减少冷启动冲击。

### 🎯 观测层精度（RQ3）
使用 MAPE 衡量吞吐估计误差（越低越好）：

| 方法 | PDF Pipeline | Video Pipeline |
|------|-------------|---------------|
| True Processing Rate | 62.7% | 54.3% |
| EMA | 28.3% | 25.7% |
| GP w/o filtering | 24.3% | 21.8% |
| GP + signal filtering | 8.4% | 7.1% |
| **TRIDENT (two-stage)** | **5.6%** | **4.8%** |

> ✅ TRIDENT 的双阶段滤波将误差降低至 **5.6% 和 4.8%**，显著优于其他方法。

### 🔬 适配层有效性（RQ4）
#### 配置优化效果（Normalized to default config）：
| 方法 | TextOCR (PDF) | Captioning (Video) |
|------|----------------|--------------------|
| Random Search | 1.18× | 1.14× |
| Grid Search | 1.22× | 1.19× |
| Unconstrained BO | 1.38× | 1.35× (OOM) |
| **Constrained BO (TRIDENT)** | **1.36×** | **1.33×** |

> ✅ 约束 BO 达到接近无约束 BO 的性能，但**完全避免 OOM**。

#### OOM 保护效果：
| 指标 | PDF Pipeline (Uncon.) | PDF Pipeline (Constr.) | Video Pipeline (Uncon.) | Video Pipeline (Constr.) |
|------|------------------------|-------------------------|--------------------------|----------------------------|
| OOM 事件数 | 14 | **3** | 11 | **2** |
| 累计宕机时间 | 462s | **102s** | 352s | **68s** |
| 有效吞吐损失 | 8.7% | **3.2%** | 7.2% | **2.7%** |

> ✅ 内存约束 BO 减少 **79–82% 的 OOM 事件**，大幅提升系统可用性。

### 🔪 消融实验（RQ5）
关闭某一模块后的相对性能（以完整 TRIDENT = 100%）：

| 变体 | PDF Pipeline | Video Pipeline |
|------|-------------|---------------|
| **TRIDENT (Full)** | 100.0% | 100.0% |
| w/o Observation Layer | 66.5% | 60.9% |
| w/o Adaptation Layer | 79.6% | 78.1% |
| w/o Placement-Aware Scheduling | 90.5% | 84.0% |
| w/o Rolling Update | 95.5% | 95.2% |

> ✅ **观测层最重要**，缺失导致性能下降近 40%，凸显准确吞吐建模的关键作用。

### ⏱️ 系统开销（RQ6）
- **观测层 & 适配层**：每次调用仅增加 2ms / 4ms，远小于 Ray 调度器本身的 ~400ms 开销。
- **MILP 求解时间**：
  - 8 节点：PDF 流水线 206ms，视频流水线 62ms
  - 16 节点：最长 1.5s，仍在分钟级重调度周期内，**不影响在线性能**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **异步、动态批处理下的吞吐建模必须结合 workload 特征与异常过滤**，否则会导致严重误判。
2. **配置调优需显式建模内存约束**，否则高性能配置往往不可靠，实际吞吐反而更低。
3. **全局联合优化（parallelism + placement + transition）比局部决策更高效**，尤其在网络受限场景下。
4. **滚动更新能有效缓解冷启动冲击**，提升配置切换的实用性。
5. **闭环反馈机制至关重要**：配置变更后必须同步更新容量模型，否则将破坏系统一致性。

### ⚠️ 方法局限性
- **MILP 可扩展性有限**：虽然当前规模可行，但在超大规模 DAG 或数千节点集群中求解时间可能成为瓶颈。
- **聚类数量上限**：在线聚类限制最大簇数 $L_{\text{max}}$，极端复杂的 workload 分布可能被合并。
- **冷启动阶段依赖 EMA**：初期样本不足时估计不够精确，需一定 warm-up 时间。
- **未支持动态拓扑变更**：目前假设流水线结构固定，不支持运行时 DAG 修改。

### 🔮 未来工作方向
- 探索 **轻量化替代 MILP** 的调度算法（如 RL 或启发式规则），提升可扩展性。
- 引入 **元学习或迁移学习** 加速新 workload 模式的冷启动过程。
- 支持 **多目标优化**（如成本、延迟、能耗）而不仅是吞吐。
- 扩展至 **在线推理 Serving 场景**，实现训练与推理统一调度架构。

---

> **总结**：TRIDENT 通过构建“观测-适配-调度”三位一体的闭环系统，在真实多模态数据流水线上实现了高达 **2.01× 的端到端吞吐提升**，同时保障了系统的稳定性与安全性，为下一代 AI 数据基础设施提供了重要的调度范式。

</details>

---

### 3. [A Cascaded Graph Neural Network for Joint Root Cause Localization and Analysis in Edge Computing Environments](https://arxiv.org/abs/2603.01447)

**Authors**: Duneesha Fernando, Maria A. Rodriguez, Rajkumar Buyya  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.01447v1  

#### Abstract
Edge computing environments host increasingly complex microservice-based IoT applications that are prone to performance anomalies propagating across dependent services. Identifying the faulty component (root cause localization) and the underlying fault type (root cause analysis) is essential for tim...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Cascaded Graph Neural Network for Joint Root Cause Localization and Analysis in Edge Computing Environments

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
在边缘计算环境中，基于微服务的IoT应用日益复杂，性能异常会通过服务间的依赖关系传播，导致多个服务表现出异常行为。传统的 **Root Cause Localization (RCL)** 和 **Root Cause Analysis (RCA)** 方法通常采用集中式处理全系统图的方式，存在以下问题：
- **高推理延迟**：随着图规模增大，GNN的消息传递复杂度急剧上升（$O(N^2)$），难以满足实时诊断需求。
- **可扩展性差**：在大规模分布式边缘环境中，集中式模型无法高效运行。

此外，大多数现有方法仅关注诊断准确性，忽视了效率与延迟之间的平衡。

---

### **提出了什么新方法或新思路**
本文提出了一种 **级联式图神经网络框架（Cascaded GNN）**，用于联合执行 **RCL** 与 **故障类型识别（fault type identification）**，其核心思想是：
- **通信驱动的聚类（Communication-driven clustering）**：利用 Louvain 社区检测算法，将微服务按通信强度划分为高度交互的社区（clusters），每个 cluster 表示一组频繁通信的服务。
- **两级级联架构**：
  - **Proposal Network (P-Net)**：在每个 cluster 内部运行，进行局部 RCL 并生成 cluster-level embedding。
  - **Output Network (O-Net)**：将每个 cluster 视为一个节点，构建 cluster-level 图，进行全局 RCA 判断。
- **分层推理机制**：通过限制消息传递范围至子图（subgraphs），显著降低计算复杂度。

该方法实现了 **RCL 与 RCA 的联合建模**，同时兼顾了精度与效率。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **可扩展性** | 推理时间随图规模增长几乎保持恒定，而传统集中式 GNN 延迟呈指数增长。 |
| **效率** | 通过聚类压缩搜索空间，减少 GCN 层的输入规模，显著降低计算开销。 |
| **准确性** | 在中等规模数据集上达到与集中式 GNN 相当的诊断准确率。 |
| **实用性** | 更适合部署于资源受限、对延迟敏感的大规模边缘环境。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**
1. **MicroCERCL**  
   - 公开可用的云边协同微服务基准数据集。
   - 包含 81 个微服务，来自四个应用（SockShop、Hipster、Bookinfo、AI-Edge）。
   - 部署在 4 台云服务器 + 4 台边缘服务器上。
   - 注入多种真实异常：CPU 耗尽、内存泄漏、网络延迟、丢包等。
   - 共 682 种故障场景，训练/验证/测试集划分比例为 60:20:20。

2. **iAnomaly 模拟框架生成的数据集**  
   - 支持生成从 50 到 10,000 节点不等的服务依赖图。
   - 基于真实执行轨迹模拟异常，保留了真实的时序与因果特性。
   - 用于评估模型在不同图规模下的 **可扩展性**。

---

### **实验设置和评估指标**

#### **评估任务**
- **RCL（Root Cause Localization）**
  - 指标：Top-K 准确率（Acc@1, Acc@3, Acc@5）、MAR（Mean Average Rank）、MRR（Mean Reciprocal Rank）
- **RCA（Root Cause Analysis）**
  - 指标：Accuracy、Precision、Recall、F1-score（多分类任务）

#### **基线方法对比**
- **Centralized Joint GNN**：本文实现的集中式 GNN 架构作为主要 baseline。
- 对比其他相关工作如 MicroCERCL、DejaVu、DiagFusion 等的报告结果。

#### **实现细节**
- 使用 PyTorch Geometric 实现 GNN。
- 时间特征提取使用 1D-CNN。
- 超参数优化采用 TPE 贝叶斯优化。
- 所有实验在 Spartan HPC 集群上完成。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **在 MicroCERCL 数据集上的表现**

| 模型 | Acc@1 (RCL) | Acc@5 (RCL) | F1-score (RCA) |
|------|-------------|-------------|----------------|
| Centralized GNN (本文实现) | **0.9275** | **1.0000** | **0.8715** |
| Cascaded GNN (本文) | **0.9130** | 0.9493 | **0.8640** |

> ✅ 结论：级联模型在 RCL 和 RCA 上均取得了与集中式模型相当的精度，差距极小。

#### **推理延迟对比（MicroCERCL ~50 nodes）**
- Centralized GNN：平均 17.78 ms
- Cascaded GNN：平均 20.24 ms  
> ⚠️ 在中小规模图上，级联模型因引入聚类和两阶段推理略慢，但差异不大。

---

### **在 iAnomaly 大规模数据集上的可扩展性分析**

#### **推理时间随图规模变化趋势（图节点数：50 → 10,000）**

| 模型配置 | 推理时间趋势 |
|--------|--------------|
| Centralized GNN | 明显上升，呈现非线性增长 |
| Cascaded GNN (固定 10 clusters) | 增长缓慢 |
| **Cascaded GNN (自适应聚类)** | **近乎恒定（~0.03s）** |

> 📈 图表见原文 Fig. 3：随着节点数增加，集中式模型延迟迅速攀升，而自适应聚类的级联模型维持稳定推理时间。

---

### **消融实验结果（Ablation Study）**

| 变体 | Acc@1 (RCL) ↓ | F1-score (RCA) ↓ | 分析 |
|------|---------------|------------------|------|
| 完整模型（Joint + Comm. Clustering + GCN） | **0.9130** | **0.8640** | 基准 |
| 移除联合学习（Separate Models） | 0.8696 | 0.6692 | 性能明显下降，说明共享表示有效 |
| 替换为随机聚类（Random Clustering） | 0.7971 | 0.8512 | RCL 显著退化，证明通信驱动聚类必要 |
| 使用 GAT 替代 GCN | 0.8478 | 0.8346 | 性能下降且计算更重，GCN 更优 |

> 🔍 发现：**联合学习** 和 **通信驱动聚类** 是性能保障的关键；**GCN** 在效率与效果间取得更好平衡。

---

## 4. 关键结论和发现

### **主要发现**
1. **级联架构可在不牺牲准确性的前提下大幅提升可扩展性**：
   - 在中等规模图上，诊断精度与集中式 GNN 相当。
   - 在大规模图上，推理时间接近常数，远优于集中式模型。

2. **通信驱动聚类能有效保留异常传播路径中的关键依赖信息**：
   - 异常通常沿调用链传播，因此基于通信的聚类天然契合故障定位逻辑。

3. **联合学习有助于提升 RCL 与 RCA 的协同能力**：
   - 共享底层特征提取器使两个任务互相促进，尤其提升了 RCA 的 F1-score。

4. **传统集中式 GNN 不适用于超大规模边缘系统**：
   - $O(N^2)$ 的计算复杂度使其难以应对数千节点级别的微服务拓扑。

---

### **方法的局限性**
- **依赖高质量的聚类结果**：若通信模式稀疏或噪声大，聚类可能失效。
- **当前为集中推理设计**：虽然模型轻量化，但仍需中心节点聚合 cluster 输出。
- **未考虑动态拓扑变化**：假设服务依赖图静态，实际中可能存在弹性扩缩容。
- **标签依赖性强**：属于监督学习方法，需要大量标注的故障样本进行训练。

---

### **未来工作方向**
1. **分布式/联邦部署**：
   - 将 P-Net 部署到边缘设备本地，实现去中心化的初步诊断，进一步降低通信开销。

2. **融合共置依赖（Colocation Dependencies）**：
   - 当前聚类仅基于通信，未来计划结合部署在同一主机上的服务间资源竞争关系。

3. **支持在线增量学习**：
   - 应对动态变化的服务拓扑和新型故障模式。

4. **探索无监督/弱监督方案**：
   - 减少对大规模标注数据的依赖，增强泛化能力。

---

> ✅ **总体评价**：本文提出的 **Cascaded GNN** 是首个明确面向边缘计算环境的高效、可扩展的联合 RCL/RCA 框架，在保持高精度的同时解决了传统 GNN 的可扩展性瓶颈，为 AIOps 在大规模边缘系统的落地提供了可行路径。

</details>

---

### 4. [Energy-Efficient Information Representation in MNIST Classification Using Biologically Inspired Learning](https://arxiv.org/abs/2603.00588)

**Authors**: Patrick Stricker, Florian R\"ohrbein, Andreas Knoblauch  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.00588v1  

#### Abstract
Efficient representation learning is essential for optimal information storage and classification. However, it is frequently overlooked in artificial neural networks (ANNs). This neglect results in networks that can become overparameterized by factors of up to 13, increasing redundancy and energy co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Energy-Efficient Information Representation in MNIST Classification Using Biologically Inspired Learning

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对当前 **Deep Neural Networks (DNNs)** 中普遍存在的**过参数化**（overparameterization）问题展开研究。传统基于 **Backpropagation (BP)** 的训练方法往往导致网络存储大量冗余信息和噪声，造成模型体积膨胀、计算资源消耗高以及碳排放增加，尤其在大规模语言模型（LLMs）快速发展的背景下，这一问题引发了严重的环境与伦理关切。

此外，尽管人工神经网络受大脑启发，但现有模型大多仅关注**突触权重可塑性**（synaptic weight plasticity），而忽略了生物神经系统中更为高效的**结构可塑性**（structural plasticity）机制，从而难以实现真正高效的信息表示与学习。

---

### 提出的新方法与新思路
作者提出了一种**受生物学启发的学习框架**（biologically inspired learning framework），其核心是结合以下机制：
- **竞争性兴奋性Hebbian可塑性**（competitive excitatory Hebbian plasticity）
- **非负性约束**（nonnegativity constraints）
- **权重扰动**（weight perturbation, WP）
- **偏置神经元上的稳态可塑性**（homeostatic plasticity）

该方法通过模拟大脑中的**结构可塑性**机制，在训练过程中动态地保留对分类任务至关重要的突触连接，自动剪枝不必要连接，从而自然防止过参数化。

特别地，作者将 MNIST 分类任务重新建模为一个**异联想记忆**（heteroassociative memory）任务，并采用两层网络架构进行建模，其中输入图像作为“地址模式”（address pattern），输出标签作为“内容模式”（content pattern）。

---

### 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **效率与稀疏性** | 自动优化突触使用，显著减少非静默突触数量（nonsilent synapses），提升存储效率 |
| **无需预设架构** | 不需要手动调整网络大小或进行后处理剪枝，具备自适应能力 |
| **能量与可持续性** | 大幅降低能耗，符合绿色AI发展方向 |
| **信息压缩能力** | 在更低的互信息 $I(X;Z)$ 下实现良好性能，表明更优的信息瓶颈特性 |
| **生物合理性** | 更贴近真实大脑的学习机制，如突触生成/删除、空间预留等 |

相比标准 BP 和 Chorowski 等约束型 BP 方法，本方法在**突触容量**（synaptic capacity, $C_S$）上表现最优。

---

## 2. 核心实验方法和设置

### 数据集
- 主要使用 **MNIST 手写数字数据集**
  - 包含 70,000 张 28×28 灰度图像（60,000 训练 + 10,000 测试）
  - 本文聚焦于识别数字 **1、2 和 6**

### 实验设置
- 模型结构：单隐藏层前馈网络（one-hidden-layer feedforward network）
- 隐藏层激活函数：修改后的 Sigmoid 函数，输出范围从 [0.5, 1.0] 映射到 [0.0, 1.0]，以匹配非负性要求
- 权重初始化：均匀分布 U(0.01, 0.1)
- 损失函数：交叉熵损失（cross-entropy loss）
- 批量训练：mini-batch processing，时间序列扩展用于模拟连续学习过程
- 平台：Keras + TensorFlow；小规模本地原型开发（RTX 4080），最终实验部署于 bwUniCluster 2.0 上的 A100/H100/P100 GPU

---

### 评估指标
1. **测试准确率**（Test Accuracy）
2. **互信息 $I(X;Z)$**：衡量输入 $X$ 与潜在变量 $Z$ 之间的信息量，反映信息压缩程度
3. **突触容量 $C_S$**（Synaptic Capacity）：
   $$
   C_S = \frac{I(X;Z)}{\text{Number of nonsilent synapses}} \quad [\text{bits/synapse}]
   $$
   衡量单位非静默突触所能承载的信息量，越高表示信息利用越高效

4. **非静默突触数**（Number of nonsilent synapses）：体现模型稀疏性和资源占用情况

---

### 基线方法对比
- **标准 Backpropagation (BP)**
- **Chorowski et al. [17]**：施加非负性和稀疏性约束的 BP 方法
- 本文提出的**生物启发式学习规则**（Authors）

所有模型均提取确定性隐藏表示 $H$ 后接入一个变分编码器（Variational Encoder）来估计 $I(X;Z)$，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）

| 隐藏神经元数 | 方法 | Test Accuracy | $I(X;Z)$ (bits) | $C_S$ (bits/synapse) |
|-------------|-------|----------------|------------------|------------------------|
| 10          | BP               | 99.01%         | 22.50            | 2.87×10⁻³              |
|             | Chorowski        | 98.34%         | 16.10            | 1.08×10⁻²              |
|             | **Authors**      | **64.29%**     | **12.80**        | **1.63×10⁻²**          |
| 30          | BP               | 99.17%         | 127.62           | 5.43×10⁻³              |
|             | Chorowski        | 99.01%         | 72.99            | 2.22×10⁻²              |
|             | **Authors**      | **86.21%**     | **52.18**        | **6.66×10⁻²**          |
| 100         | BP               | 99.23%         | 435.93           | 5.56×10⁻³              |
|             | Chorowski        | 99.10%         | 217.43           | 5.93×10⁻²              |
|             | **Authors**      | **89.79%**     | **198.33**       | **2.53×10⁻¹**          |
| 200         | BP               | 99.17%         | 518.79           | 3.31×10⁻³              |
|             | Chorowski        | 99.10%         | 402.91           | 1.31×10⁻¹              |
|             | **Authors**      | **95.55%**     | **372.66**       | **4.75×10⁻¹**          |

> 注：表中最佳值已加粗。虽然本文方法在准确率上略低于 BP，但在 $C_S$ 上全面领先。

---

### 与基线方法的对比结果
- **突触容量 $C_S$**：
  - 在所有配置下，本文方法的 $C_S$ 均显著高于 BP 和 Chorowski 方法。
  - 例如在 200 隐藏神经元时，$C_S = 0.475$，远超 BP 的 0.0033 和 Chorowski 的 0.131。
- **信息压缩效率**：
  - 尽管 $I(X;Z)$ 较低，但分类性能仍保持较高水平，说明信息表示更加紧凑有效。
- **稀疏性**：
  - 由于非负性和局部竞争机制，网络自动形成稀疏连接，减少了冗余参数。

---

### 消融实验与分析（隐含在文中讨论）
虽然未明确列出消融实验表格，但文中指出以下关键因素的作用：
- **修改后的 Sigmoid 激活函数**：解决了早期工作中 baseline 模型无法训练的问题，显著提升了性能。
- **权重扰动（WP）机制**：使权重更新更具探索性，避免陷入局部最优。
- **非负性与竞争机制**：驱动稀疏性，促进结构可塑性，提高泛化能力。

---

## 4. 关键结论和发现

### 主要发现
1. **高效信息表示可通过模拟结构可塑性实现**：
   - 生物启发式学习规则能自然抑制过参数化，仅保留必要的突触连接。
2. **更高的突触容量意味着更优的资源利用率**：
   - 本文方法在单位突触上传递更多信息，优于传统 BP 及约束型 BP。
3. **信息压缩与高性能可以兼得**：
   - 虽然 $I(X;Z)$ 更低，但分类准确率接近甚至超过某些设置下的基线，支持“压缩有助于泛化”的假设（Alemi et al.）。
4. **无需人工设计稀疏策略**：
   - 稀疏性由学习动力学自发产生，无需额外正则化或剪枝步骤。

---

### 方法的局限性
- **分类准确率略低于 BP**：
  - 特别是在较小网络中差距较明显（如 10 neurons 时仅为 64.29% vs BP 的 99.01%），可能因缺乏精确梯度传播所致。
- **尚未验证于深层网络或多任务场景**：
  - 当前实验限于浅层单任务（MNIST 子集），扩展性有待验证。
- **依赖特定激活函数调整**：
  - 修改后的 Sigmoid 是性能提升的关键之一，通用性需进一步检验。

---

### 未来工作方向
1. **缩小与 BP 的性能差距**：
   - 探索融合局部扰动与近似梯度的方法，提升收敛速度与精度。
2. **扩展至更深网络与复杂任务**：
   - 应用于 CNN、Transformer 架构及更复杂的视觉或语言任务。
3. **多任务持续学习能力研究**：
   - 利用“预留空间”特性，探索模型如何在不遗忘旧知识的前提下学习新任务。
4. **硬件友好性与能效实测**：
   - 结合神经形态芯片（neuromorphic hardware）评估实际功耗表现。
5. **理论建模深化**：
   - 进一步建立与 Information Bottleneck、Markov Chain 表示之间的形式化联系。

---

> ✅ 总结一句话：  
> 本文提出一种受脑科学启发的稀疏学习框架，通过模拟**结构可塑性**实现了**高突触容量、低冗余、自适应剪枝**的高效信息表示，在 MNIST 上验证了其卓越的存储效率与节能潜力，为构建可持续、可扩展的绿色 AI 提供了新范式。

</details>

---

### 5. [Accelerating PDE Surrogates via RL-Guided Mesh Optimization](https://arxiv.org/abs/2603.02066)

**Authors**: Yang Meng, Ruoxi Jiang, Zhuokai Zhao, Chong Liu, Rebecca Willett, Yuxin Chen  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.02066v1  

#### Abstract
Deep surrogate models for parametric partial differential equations (PDEs) can deliver high-fidelity approximations but remain prohibitively data-hungry: training often requires thousands of fine-grid simulations, each incurring substantial computational cost. To address this challenge, we introduce...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Accelerating PDE Surrogates via RL-Guided Mesh Optimization*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

深度学习驱动的 **PDE surrogate models**（如 FNO、DeepONet）在科学计算中展现出强大潜力，但其训练极度依赖大量高保真数值模拟（fine-grid simulations），而每次模拟都计算成本高昂。这导致训练过程 **data-hungry 且 computationally prohibitive**。

传统方法通常在固定、均匀的网格上进行求解，即使解具有强非均匀性（如激波、边界层），也浪费了大量计算资源于平滑区域。

### **提出了什么新方法或新思路**

本文提出 **RLMESH** —— 一个端到端的、基于强化学习（Reinforcement Learning, RL）的自适应网格优化框架，用于高效训练 PDE surrogate。

#### 核心思想：
- 将每个 PDE 实例的 **mesh grid point 选择** 视为一个 **sequential decision process**。
- 使用 **RL policy** 在每一步动态决定在哪个空间位置采样，以最大化信息增益。
- 引入一个轻量级的 **proxy model**（如 kernel ridge regression）来快速估计新采集数据对主 surrogate 模型（如 FNO）的改进程度，作为 RL 的奖励信号，避免频繁重训练主模型。

#### 创新点：
1. **首次将 per-instance spatial adaptivity 与 active learning 结合**：不同于以往仅选择“哪个实例”或“全局分辨率”，RLMESH 在单个实例内部实现细粒度的空间自适应采样。
2. **引入 proxy model for reward estimation**：解决了 RL 中稀疏反馈和昂贵奖励计算的问题，使在线策略优化成为可能。
3. **端到端可训练框架**：实现了 RL policy、proxy model 和 PDE surrogate 的协同演化。

### **相比现有方法的优势**

| 对比维度 | 现有方法（如 MRA-FNO, AL4PDE） | RLMESH |
|--------|-------------------------------|--------|
| 采样粒度 | 实例级（instance-level）或全局分辨率 | 实例内空间点级（per-instance spatial point） |
| 自适应性 | 静态或粗粒度 | 动态、细粒度、输入依赖 |
| 反馈机制 | 依赖完整解或不确定性估计 | 使用 proxy model 快速估计下游 surrogate 改进 |
| 效率 | 节省实例数量 | 显著减少每个实例的求解点数，大幅降低总计算成本 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

在三个经典 PDE 基准上进行验证：
- **1D Burgers' Equation**：终端时间预测，含激波。
- **2D Darcy Flow**：稳态映射，含高导通道。
- **Lorenz-96 System**：混沌格点系统，无连续空间几何。

所有数据来自 **PDEBench** 数据集。

### **实验设置和评估指标**

#### 设置：
- 总训练实例：1,000；测试实例：200。
- 前 100 个实例用于 surrogate 预训练。
- 剩余 900 个实例用于主动采样（active acquisition），分 18 轮，每轮 50 个实例。
- 每个实例有 **per-instance budget B**（如 60 个网格点），RL policy 需在此预算内选择最优采样点。
- 主 surrogate 模型：**Fourier Neural Operator (FNO)**。
- Proxy model：**kernel ridge regression (RBF kernel)**。
- RL 算法：**Deep Q-Network (DQN)**。

#### 评估指标：
- **RMSE**（均方根误差）在密集评估网格上的表现。
- **时间-误差权衡**（time-error tradeoff）：考虑实际求解时间的累积误差下降速度。
- 所有结果取 **5 次独立运行的平均值 ± 标准差**。

### **基线方法对比**

- **Uniform**：在规则网格上均匀采样。
- **Random**：随机选择网格点。
- **Gradient**：优先选择梯度大的区域。
- **Variance**：基于模型不确定性采样。
- **Intensity**：基于输入场强度采样。
- **Self-MI / LCMD**：适配自 MRA-FNO 和 AL4PDE 的实例级主动学习方法。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 在 **Burgers 方程** 上：
- RLMESH 在约 **第 6 轮迭代** 达到 RMSE ≈ 0.02。
- 其他启发式方法需 **9–12 轮** 才能达到相同精度。
- **节省 33% ~ 50% 的标注成本**。

#### 时间-误差分析（Burgers）：
- 达到 RMSE = 0.02：
  - RLMESH：约 **40 秒** 累积求解时间。
  - Variance/Intensity：约 **80–120 秒**。
  - Uniform/Random：超过 **150 秒**。
- 表明 RLMESH **单位时间内的精度提升效率显著更高**。

#### 最优预算分析：
- 当 **B = 60** 时，性能接近饱和，继续增加至 80 或 100 点带来的边际收益极小。
- 因此 **B = 60 是性价比最高的设置**。

### **与基线方法的对比结果**

- 在所有三个任务（Burgers, Darcy, Lorenz-96）上，RLMESH **始终优于所有基线方法**。
- 学习曲线更陡峭，早期阶段提升最快。
- 更接近 **full-information oracle**（即在完整网格上训练的理想情况）的性能下限。
- 特别是在 **Burgers 的激波区域** 和 **Darcy 的高对比通道**，RLMESH 显式地将采样点集中在这些关键区域（见可视化图 6）。

### **消融实验结果**

#### Proxy Model 对齐性（Ablation）
- **kernel ridge regression** 与 FNO surrogate 的误差变化具有极强的 **Spearman 相关性（ρ = 0.9908）**。
- 表明 proxy 能准确反映主模型的真实泛化改进趋势，是稳定 RL 训练的关键。

#### Per-instance Adaptivity 的必要性
- 与仅做实例级选择的方法（如 Self-MI, LCMD）对比（Table 1）：
  - 即使总查询预算相同，RLMESH 仍取得更低 RMSE。
  - 证明 **细粒度空间自适应** 比单纯选择“更有价值的实例”更能提升效率。

#### 不同预算下的性能（Fig. 9）
- 当 B = 20 时，所有方法性能下降明显，说明 **FNO 容量受限于过少的观测点**，而非策略失败。
- 进一步支持 B = 60 为合理选择。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Solver-level spatial adaptivity 可极大提升 surrogate 训练效率**：通过在关键区域集中计算资源，可在显著减少求解点数的情况下达到与全网格训练相当的精度。
2. ✅ **RL + proxy model 是实现高效自适应采样的可行路径**：proxy model 提供了低成本、高对齐性的奖励信号，使 RL policy 能有效学习复杂的空间采样策略。
3. ✅ **Per-instance adaptivity 优于 instance-level adaptivity**：在单个实例内部进行细粒度决策，能捕捉到启发式方法无法识别的局部结构。
4. ✅ **RLMESH 具有普适性**：不仅适用于传统 PDE（Burgers, Darcy），也能在混沌动力系统（Lorenz-96）上取得优势。

### **方法的局限性**

1. **依赖高质量的非均匀网格求解器**：当前实验依赖定制的 finite-volume solver 来处理稀疏、不规则网格。若求解器不稳定，会影响采样质量。
2. **2D/3D 扩展挑战大**：文中未在 2D Darcy 上进行时间-误差分析，因构建鲁棒的非均匀 2D 求解器复杂度高。
3. **FNO 在极稀疏采样下容量受限**：当每个实例采样点过少（如 B=20），即使最优策略也无法克服模型表达能力瓶颈。
4. **Proxy model 的泛化性**：虽然在实验中表现良好，但其对不同 PDE 或更复杂几何的迁移能力有待验证。

### **未来工作方向**

1. **扩展至时空联合感知**（space-time sensing）：同时优化时间和空间上的采样点。
2. **多保真度建模**（multi-fidelity costs）：结合粗/细网格求解，进一步降低成本。
3. **高维与不规则几何扩展**：结合 geometry-aware operators 处理复杂域。
4. **理论分析**：提供关于样本效率的理论保证。
5. **联合优化**：探索 instance selection 与 grid-point selection 的联合策略。

---

> **总结**：RLMESH 通过 **RL-guided per-instance mesh optimization**，成功将 PDE surrogate 训练从“暴力求解”转向“智能采样”，在多个基准上实现了 **更少查询、更快收敛、更高效率** 的突破，为大规模科学机器学习的实用化提供了新范式。

</details>

---

### 6. [HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents](https://arxiv.org/abs/2603.00977)

**Authors**: Hongbo Jin, Rongpeng Zhu, Jiayu Ding, Wenhao Zhang, Ge Li  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00977v1  

#### Abstract
Large language model (LLM) agents have recently demonstrated strong capabilities in interactive decision-making, yet they remain fundamentally limited in long-horizon tasks that require structured planning and reliable execution. Existing approaches predominantly rely on flat autoregressive policies...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Large Language Model (LLM)** 的智能体在处理**长视野任务**（long-horizon tasks）时面临三大挑战：
- **指数级探索复杂度**（exponential exploration complexity）
- **延迟奖励分配**（delayed credit assignment）
- **语义漂移**（semantic drift），即推理过程中逐渐偏离原始目标

现有方法普遍采用“扁平化”（flat）策略架构，将高层推理与底层动作生成混合在一个自回归序列中，导致错误传播严重、探索效率低下。

---

### 提出了什么新方法或新思路
本文提出 **HiMAC**（Hierarchical Macro-Micro Agentic Control），一种**分层强化学习框架**，通过显式解耦决策过程为两个层级：

- **Macro-Policy（宏观策略）**：作为**规划器**（Planner），负责生成一个结构化的自然语言子目标蓝图（structured blueprint），将长期任务分解为可管理的里程碑。
- **Micro-Policy（微观策略）**：作为**执行器**（Executor），在给定蓝图条件下逐个完成子任务，并通过 `<sub_done>` 标记自主触发阶段切换。

该框架共享同一个 LLM 参数，但通过不同训练阶段分别优化蓝图生成和动作执行。

---

### 相比现有方法的优势
- **降低探索维度**：将联合搜索空间拆分为两层独立优化，显著减少每步的搜索复杂度。
- **抑制错误传播**：执行层的错误被限制在单个子目标内，不会影响全局计划。
- **无需 Critic 网络**：引入 **Critic-Free Hierarchical Policy Optimization**，利用层级化的 **Group Relative Advantage Estimation** 实现稳定信用分配。
- **解决非平稳性问题**：提出 **Iterative Co-Evolution Training**，交替进行 Planner 探索与 Executor 适应，避免双层策略同时更新带来的训练不稳定性。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个具有代表性的长视野基准上进行，涵盖文本与视觉模态：

| 数据集 | 任务描述 | 特点 |
|-------|--------|------|
| **ALFWorld** | 在模拟家庭环境中执行多步操作（如“把蜡烛放进马桶”） | 多模态感知 + 复杂状态转移 |
| **WebShop** | 在噪声电商网站中导航并购买符合属性的商品 | 高维观察 + 噪声信号 + 长交互链 |
| **Sokoban** | 推箱子游戏，需精确顺序推理 | 视觉接地 + 空间逻辑规划 |

---

### 实验设置和评估指标

#### 模型架构
- 文本任务（ALFWorld、WebShop）：使用 **Qwen2.5-Instruct**（1.5B 和 7B）
- 视觉任务（Sokoban）：使用 **Vision-Language Models (VLMs)** 如 Qwen2.5-VL 和 Qwen3-VL

#### 训练细节
- 最大提示长度：1024–2048
- Rollout 组大小：N = 8
- KL 正则系数 β = 0.01
- 使用统一 LLM 参数，仅对不同 token 类型（蓝图 vs 动作）施加梯度屏蔽

#### 评估指标
- **Success Rate（成功率）**
- **Score（综合得分）**
- **Sample Efficiency（样本效率）**：达到特定阈值所需的训练迭代次数

---

### 基线方法对比
| 类型 | 方法 |
|-----|------|
| Prompting 方法 | ReAct, Reflexion, Qwen2.5-PE |
| 强化学习方法 | PPO, RLOO, GRPO, GiGPO |
| 闭源模型 | GPT-4o, Gemini-2.5-Pro, Claude Sonnet 4.5 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ALFWorld（7B 模型）
| 方法 | 成功率（All） |
|------|-------------|
| HiMAC (Ours) | **92.1%** |
| GiGPO（最强 RL 基线） | 90.8% |
| GRPO | 81.3% |
| ReAct | 31.2% |

> ✅ 在复杂任务如 Pick2 和 Clean 上提升尤为明显，表明其擅长结构化规划。

#### WebShop（7B 模型）
| 方法 | Success Rate | Score |
|------|--------------|-------|
| HiMAC | **84.1%** | **93.8** |
| GiGPO | 75.2% | 86.2 |
| GRPO | 65.7% | 80.3 |
| ReAct | 19.5% | 46.2 |

> ✅ **相对最强 RL 基线提升达 16.0% 成功率**，是目前 WebShop 上表现最佳的方法。

#### Sokoban（Qwen2.5-VL-7B）
| 方法 | Success Rate | Score |
|------|--------------|-------|
| HiMAC | **87.5%** | **6.70** |
| GiGPO | 82.8% | 5.27 |
| GRPO | 83.9% | 5.39 |

> ✅ 显示 HiMAC 可泛化至视觉接地环境，且优于所有基线。

---

### 消融实验结果（Ablation Study）

| 变体 | ALFWorld (%) | WebShop Score | WebShop Succ. (%) |
|------|---------------|----------------|--------------------|
| **HiMAC (Full)** | 92.1 | 93.8 | 84.1 |
| w/o Hierarchy（Flat GRPO） | 77.6 (-14.5) | 79.3 (-14.5) | 66.1 (-18.0) |
| w/o Iterative Co-Evolution | 85.3 (-6.8) | 86.7 (-7.1) | 74.8 (-9.3) |
| w/o `<sub_done>` | 88.2 (-3.9) | 90.1 (-3.7) | 79.8 (-4.3) |
| Random Blueprint | 89.7 (-2.4) | 91.6 (-2.2) | 81.5 (-2.6) |

> 🔍 结论：
- 层级结构本身带来最大收益（-14.5% drop）
- 迭代共进化训练对稳定性至关重要，尤其在 WebShop 中影响更大
- `<sub_done>` 机制支持动态节奏控制，优于固定步数预算
- 高置信蓝图选择有效过滤低质量计划，提高训练效率

---

## 4. 关键结论和发现

### 主要发现
1. **结构化层级优于单纯扩大模型规模**  
   HiMAC 在仅 1.5B 参数下就超越闭源 Gemini-2.5-Pro（60.3% → 89.9%），说明**结构归纳偏置**（structural inductive bias）比参数量更重要。

2. **分层设计显著提升样本效率**  
   如 Table 4 所示，HiMAC 在 ALFWorld、WebShop 和 Sokoban 上均以更少训练迭代达到目标性能，收敛速度更快。

3. **自发涌现高级行为**  
   宏观策略在后期训练中发展出**自我验证机制**（self-verification），例如添加 “Inventory or look to confirm” 步骤来确认任务完成，这是扁平策略从未出现的行为。

4. **共进化形成隐式课程学习**  
   随着 Executor 能力增强，Planner 自动探索更复杂的蓝图，形成**渐进式难度上升的内在课程**（emergent curriculum），无需人工设计。

---

### 方法的局限性
- 当前蓝图仍由同一 LLM 生成，未完全实现模块化；未来可探索异构 Planner-Executor 架构。
- 子目标数量固定或受限于上下文窗口，难以处理极长任务。
- 对 `<sub_done>` 的依赖要求环境提供明确反馈信号，在部分开放域场景可能不可靠。

---

### 未来工作方向
- 将 HiMAC 应用于更开放的现实世界环境（如安卓设备控制、真实网页操作）
- 探索跨任务、跨领域的**蓝图迁移能力**
- 引入记忆机制支持长期状态追踪
- 扩展到多智能体协作场景中的分层协调

---

> 💡 **最终结论**：  
> **Structured hierarchy, rather than increased model scale alone, is the decisive factor for robust long-horizon agentic intelligence.**  
> （相比单纯扩大模型规模，结构化层级才是实现稳健长视野智能体的关键。）

</details>

---

### 7. [CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging](https://arxiv.org/abs/2603.00573)

**Authors**: Jie Cao, Zhenxuan Fan, Zhuonan Wang, Tianwei Lin, Ziyuan Zhao, Rolan Yan, Wenqiao Zhang, Feifei Shao, Hongwei Wang, Jun Xiao, Siliang Tang  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00573v1  

#### Abstract
Large language models (LLMs) achieve remarkable performance on diverse downstream and domain-specific tasks via parameter-efficient fine-tuning (PEFT). However, existing PEFT methods, particularly MoE-LoRA architectures, suffer from limited parameter efficiency and coarse-grained adaptation due to t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 MoE-LoRA 架构在参数高效微调（PEFT）中面临两大挑战：
- **参数效率低下**：引入多个 LoRA 专家和路由网络显著增加了可训练参数量，违背了 PEFT 的初衷。
- **粗粒度适应能力不足**：如 SMEAR 等方法采用实例级（instance-level）路由，无法实现对每个 token 的细粒度动态适配。

### **提出了什么新方法或新思路**
本文提出 **Core Space Mixture of LoRA (CoMoL)**，一种新型 MoE-LoRA 框架，其核心思想是将专家参数压缩并融合于一个共享的低秩“核心空间”中。主要包括两个关键组件：

- **Core Space Experts（核心空间专家）**  
  将每个 LoRA 专家的参数 $ \Delta W = B_i A_i $ 通过 SVD 分解为 $ U_B M_i V_A^\top $，其中 $ M_i \in \mathbb{R}^{r \times r} $ 是维度仅为 $ r \times r $ 的**核心矩阵**（core matrix），用于存储专家特异性知识。所有专家共享 $ U_B $ 和 $ V_A $ 子空间，从而极大减少参数增长。

- **Core Space Routing（核心空间路由）**  
  路由器直接作用于输入在低秩空间中的投影 $ v_c = V_A^\top x $，输出 token-level 的专家权重，并在核心空间内进行软合并（soft-merging），最终生成单一专用 LoRA 模块。

### **相比现有方法的优势**
| 维度 | CoMoL | 传统 MoE-LoRA |
|------|-------|----------------|
| 参数数量 | 接近标准 LoRA（~1.0×） | 多个专家导致参数膨胀（N×） |
| 计算开销（FLOPs） | ~1.0× LoRA | Soft-weighted: N×, Sparse: 高延迟 |
| 路由粒度 | **Token-level**（细粒度） | Instance-level 或 Token-level 输出加权 |
| 路由延迟 | 低 | Sparse MoE 路由机制复杂，延迟高 |
| 表达能力 | 保留多专家多样性 | 存在冗余与过拟合风险 |

> ✅ **核心优势**：在保持接近标准 LoRA 的参数量和计算成本的同时，实现了 token-level 的细粒度、输入自适应的专家选择与融合。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
#### 数学推理任务（Mathematical Reasoning）
- **微调数据集**：Math14k（GSM8K + AQuA 的 CoT 增强版）
- **测试数据集**：GSM8K, SVAMP, MultiArith, AddSub, AQuA, SingleEq

#### 代码生成任务（Code Generation）
- **微调数据集**：CodeAlpaca-20k
- **测试数据集**：HumanEval
- **评估指标**：pass@1, pass@5, pass@10

### **实验设置**
- **主干模型**：
  - Qwen3-8B 和 Qwen3-14B（数学推理）
  - Qwen3-8B 和 Llama3.1-8B（代码生成）
- **LoRA 设置**：
  - LoRA rank = 8（除特定 baseline 外）
  - 应用于 Q, K, V, O, Down-projection 层
- **专家数量**：
  - Qwen3-8B: 8 个专家
  - Qwen3-14B: 4 个专家（避免 OOM）
- **训练配置**：
  - 数学任务：1 epoch
  - 代码任务：Qwen3-8B 训练 5 epochs，Llama3.1-8B 训练 1 epoch，基于验证集选最优 checkpoint

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| 标准 LoRA | LoRA |
| Soft-weighted MoE-LoRA | MoLoRA, HydraLoRA |
| Sparse MoE-LoRA | MoLA, AdaMoLE, SparseMoA |
| Soft-merging 方法 | SMEAR（实例级）、FlyLoRA（稀疏学习） |
| 其他先进方法 | DenseLoRA |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 📊 数学推理任务（Qwen3-8B 上平均准确率）
| 方法 | 平均 Accuracy (%) | 可训练参数量 |
|------|------------------|-------------|
| LoRA | 82.78 | 24.77M |
| MoLoRA | 83.85 | 107.35M |
| SparseMoA | 83.69 | 24.49M |
| **CoMoL** | **84.48** | **25.16M** |

> 🔺 CoMoL 在仅增加约 0.4M 参数的情况下，超越 LoRA **1.7个百分点**，且优于参数量超 4 倍的 MoLoRA。

#### 📊 代码生成任务（HumanEval 上 pass@1）
| 方法 | Llama3.1-8B (pass@1) | Qwen3-8B (pass@1) | 参数量 |
|------|--------------------|------------------|--------|
| LoRA | 26.22 | 39.69 | ~24M |
| MoLoRA | 34.15 | 43.78 | >100M |
| FlyLoRA | 32.32 | 20.18 | ~23M |
| **CoMoL** | **35.00** | **48.11** | **23.24M / 24.97M** |

> ✅ CoMoL 在两个模型上均取得最佳性能，尤其在 Qwen3-8B 上远超其他方法（+8.4% vs LoRA）。

### **与基线方法的对比结果**
- CoMoL 在数学和代码任务上**全面优于所有 MoE-LoRA 变体**，包括参数更多的 MoLoRA、MoLA 和 AdaMoLE。
- 性能甚至超过部分参数量数倍的方法，表明传统 MoE-LoRA 中存在严重**参数冗余**。
- 在相同参数预算下，CoMoL 显著优于标准 LoRA，说明其有效提升了表达能力。

### **消融实验结果**
| 方法 | Qwen3-8B Math Avg | 参数量 |
|------|------------------|--------|
| CoMoL w/o CR（无 Core Space Routing） | 84.47 | 33.39M |
| **CoMoL（完整）** | **84.48** | **25.16M** |

> 🔍 结果显示：
> - 移除 Core Space Routing 后性能几乎不变，但参数量明显上升 → **Core Space Routing 有效降低路由开销而不影响性能**。
> - 即使不使用该模块，CoMoL 仍具竞争力，证明核心设计本身已足够强大。

### **扩展性实验**
- **随 rank 扩展**：CoMoL 在不同 rank 下始终优于 LoRA，最佳表现出现在 rank=16。
- **随专家数扩展**：
  - CoMoL 可稳定扩展至 **64 个专家**而无 OOM。
  - 相比之下，HydraLoRA 在 16 专家时即出现内存溢出。

---

## **4. 关键结论和发现**

### **主要发现**
1. **专家冗余普遍存在**：多数 MoE-LoRA 方法因复制完整 $ B_i, A_i $ 对而导致参数浪费，而 CoMoL 证明只需在核心空间维护 $ M_i $ 即可捕获多样化表征。
2. **token-level 动态路由可行且高效**：通过在核心空间完成 soft-merging，CoMoL 实现了真正的 token-level 自适应，同时将 FLOPs 控制在单个 LoRA 水平。
3. **低秩结构可用于路由优化**：将路由器也映射到 LoRA 的低秩空间（$ V_A^\top x $），大幅削减其参数量（从 $ O(Nn) \to O(Nr) $），是提升整体效率的关键。
4. **卓越的可扩展性与鲁棒性**：CoMoL 在不同模型规模（8B/14B）、架构（Qwen/Llama）和任务类型下均表现稳定领先。

### **方法的局限性**
- 当前研究聚焦于**参数效率**，尚未系统探讨不同 PEFT 方法的**学习容量边界**。
- 如 FlyLoRA 在 Llama 和 Qwen 上表现差异大，提示当前缺乏统一框架来评估 PEFT 方法在不同任务分布下的泛化能力。
- CoMoL 的有效性依赖于“共享子空间假设”，若专家间子空间差异过大，可能限制表达力（但实验证明此情况较少发生）。

### **未来工作方向**
- 构建系统的 **PEFT 容量评测基准**，量化不同方法的学习能力与过拟合倾向。
- 探索更灵活的子空间共享机制，例如分组共享或层级共享。
- 将 CoMoL 思想推广至其他 PEFT 方法（如 Adapter, IA³）或其他模态（视觉、多模态）。

---

> 💡 **总结一句话**：  
> **CoMoL 通过“核心空间专家 + 核心空间路由”的双创新，在几乎不增加参数的前提下，实现了兼具高表达力与细粒度适应性的 MoE-LoRA 新范式，是通往真正高效、智能微调的重要一步。**

</details>

---

### 8. [Scalable Gaussian process modeling of parametrized spatio-temporal fields](https://arxiv.org/abs/2603.00290)

**Authors**: Srinath Dama, Prasanth B. Nair  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00290v1  

#### Abstract
We introduce a scalable Gaussian process (GP) framework with deep product kernels for data-driven learning of parametrized spatio-temporal fields over fixed or parameter-dependent domains. The proposed framework learns a continuous representation, enabling predictions at arbitrary spatio-temporal co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scalable Gaussian Process Modeling of Parametrized Spatio-Temporal Fields

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**高维参数化时空场（parametrized spatio-temporal fields）的可扩展高斯过程（Gaussian Process, GP）建模难题**。传统GP在处理大规模科学计算模拟产生的高分辨率时空数据时面临严重瓶颈，其训练复杂度为 $O(n^3)$，难以应对百万级甚至更高维度的数据点。

此外，现有方法如**算子学习（operator learning）模型**（如FNO、DeepONet）虽然具有良好的预测精度，但通常作为确定性预测器，缺乏可靠的不确定性量化能力。而许多下游任务（如贝叶斯反演、优化设计）对不确定性估计有强烈需求。

### 提出的新方法与思路
作者提出了一种**基于深度乘积核（deep product kernel, DPK）的可扩展GP框架**，核心思想如下：

- **深度乘积核（Deep Product Kernel, DPK）**：将输入分解为参数 $\mu$、空间坐标 $x$ 和时间 $t$，并分别通过神经网络 $G_\mu, G_x, G_t$ 映射到潜在空间，再在每个子空间上定义独立的协方差函数（如Matérn核），最终形成乘积形式的核函数：
  $$
  k_{\text{DPK}}(z, z') = k_\mu(G_\mu(\mu), G_\mu(\mu')) \times k_x(G_x(x), G_x(x')) \times k_t(G_t(t), G_t(t'))
  $$
  这种结构既保留了非平稳性和复杂相关性建模能力，又引入了**Kronecker结构先验**以实现高效计算。

- **Kronecker矩阵代数加速**：当训练数据位于笛卡尔网格（Cartesian grid）上时，协方差矩阵自然具备Kronecker乘积结构 $K_z = K_\mu \otimes K_x \otimes K_t$，从而使得矩阵求逆、行列式计算等操作可通过特征分解高效完成，将计算复杂度从 $O((NMN_t)^3)$ 降低至近似线性于空间点数 $M$。

- **“Gappy-Grid” 扩展用于非结构化网格**：针对实际应用中常见的非结构化空间网格，提出一种“空缺网格”嵌入策略——将原始不规则网格映射到一个更大的矩形背景网格，并将无效区域视为空缺（gaps）。通过构造伪观测值（pseudovalues）使整个系统保持Kronecker结构，从而仍能利用快速线性代数运算。

- **高效的后验方差估计**：
  - 对于笛卡尔网格：可精确且高效地计算后验方差。
  - 对于非结构化网格：提出了**严格理论上下界**来逼近真实后验方差，其计算成本与后验均值相当。

### 相比现有方法的优势
| 特性 | 本文方法（DPK-GP） | FNO / DeepONet | 传统GP |
|------|---------------------|----------------|--------|
| 预测精度 | ✅ 高，媲美甚至超越算子学习方法 | ✅ 高 | ✅ 高 |
| 不确定性量化 | ✅ 内生、校准良好 | ❌ 通常无或需额外机制 | ✅ 准确但昂贵 |
| 可扩展性 | ✅ 近线性于空间点数 | ✅ 良好 | ❌ 极差 ($O(n^3)$) |
| 物理一致性 | ⭕ 数据驱动为主 | ⭕ 可结合物理约束 | ⭕ 可建模先验 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖多个流体力学与固体力学基准问题，包括：

1. **1D Unsteady Burgers' Equation**  
   - 参数化的一维无粘Burgers方程，用于与投影型降阶模型（ROM）比较。
   - 输入参数：$(\mu_1, \mu_2) \in [4.25, 5.50] \times [0.015, 0.03]$，空间点 $M=256$，时间步 $N_t=500$，训练样本 $N=80$。

2. **Hyper-Elastic Problem**  
   - 单位域内含任意形状孔洞的超弹性材料应力场预测。
   - 数据来自有限元求解器，提供在O型网格上的快照（65×41）。
   - 参数通过PCA压缩至前20个主成分。

3. **2D Transonic Flow Around Airfoil**  
   - 跨音速绕翼型流动的速度幅值场预测。
   - 翼型几何由8个控制节点垂直偏移参数化。
   - 数据在C型网格（200×50）上生成，参数经PCA降至6维。

4. **Navier-Stokes Equations (Pipe Flow)**  
   - 管道内二维不可压流动的水平速度场预测。
   - 管道中心线由分段三次多项式描述，共9个参数。
   - 数据在129×129网格上生成。

### 实验设置与评估指标
- **评估指标**：采用 **相对测试误差（relative $l^2$ test error）**：
  $$
  \text{Relative Error} = \frac{\|u_h(\mu^{(i)}) - \hat{u}(\mu^{(i)})\|_2}{\|u_h(\mu^{(i)})\|_2}
  $$
  其中 $u_h$ 为高保真解，$\hat{u}$ 为预测结果。

- **训练细节**：
  - 使用Adam优化器进行超参数学习。
  - 深度核中神经网络架构：全连接三层（1000→500→50），ReLU激活。
  - 初始噪声方差设为 $5\times10^{-3}$，权重衰减 $2.5\times10^{-5}$。

### 基线方法对比
- **Operator Learning 方法**：
  - **FNO (Fourier Neural Operator)**
  - **DeepONet**
  - **Geo-FNO**（专为一般几何设计的FNO变体）
- **传统降阶模型（Physics-based ROMs）**：
  - **POD-Galerkin**
  - **POD-LSPG**
  - **Deep-LSPG**
- **其他机器学习方法**：
  - **UNet**
  - **POD-GPR**（基于主成分分析的高斯过程回归）

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）1D Burgers 方程（vs. 物理ROM）
| 方法 | Test Parameter 1 $(4.3, 0.021)$ | Test Parameter 2 $(5.15, 0.0285)$ |
|------|-------------------------------|----------------------------------|
| **Proposed GP (DPK-Matern-5/2)** | **0.0033** | **0.0029** |
| Deep-LSPG ($p=20$) | 0.0009 | 0.0010 |
| Deep-Galerkin ($p=20$) | 0.015 | 0.013 |
| POD-Galerkin ($p=20$) | 0.032 | 0.030 |

> ✅ **结论**：所提方法显著优于传统POD类方法，接近最先进的Deep-LSPG，且为纯数据驱动方式。

#### （2）超弹性问题（vs. 算子学习）
| 方法 | Training Error | **Testing Error** |
|------|---------------|------------------|
| **Proposed GP (DPK)** | 0.0181 | **0.0326** |
| Geo-FNO (O-mesh) | 0.0344 | 0.0363 |
| FNO interpolation | 0.0314 | 0.0508 |
| DeepONet | 0.0528 | 0.0965 |

> ✅ **结论**：在测试误差上优于FNO和DeepONet，略优于Geo-FNO，显示强大泛化能力。

#### （3）跨音速绕流与管道流（综合对比）
| Method | Airfoil (Test) | Pipe (Test) |
|--------|----------------|-------------|
| **Proposed GP (DPK)** | **0.0142** | 0.0131 |
| Geo-FNO | 0.0138 | **0.0067** |
| FNO interpolation | 0.0421 | 0.0151 |
| UNet interpolation | 0.0519 | 0.0182 |

> ✅ **结论**：在Airfoil任务上表现优异，在Pipe任务上稍逊于Geo-FNO，但仍优于标准FNO和UNet。

#### （4）消融实验（Elastic Block Problem）
- 随着训练快照数量增加（从50到5000），**DPK-GP的测试误差持续下降**。
- 当 $N > 2000$ 时，**DPK-GP超越了物理驱动的POD-Galerkin方法**。
- 在所有数据量下，**DPK-GP均显著优于POD-GPR**。

> 🔍 **发现**：深度核结构赋予模型更强的学习能力和表达力，尤其在大数据场景下优势明显。

---

## 4. 关键结论和发现

### 主要发现
1. **可扩展性突破**：通过**深度乘积核 + Kronecker代数**，实现了对百万级时空网格点的**精确GP推理**，复杂度近乎线性于空间点数。
2. **精度竞争力强**：在多个基准问题上，预测精度**媲美甚至超过FNO、DeepONet等主流算子学习方法**。
3. **不确定性天然支持**：相比大多数算子学习模型，本方法**原生提供高质量的不确定性估计**（见图3、4中的置信区间），适用于风险敏感任务。
4. **统一框架适用广**：支持固定域与参数化域（parametrized domains）、结构化与非结构化网格，具有广泛适用性。
5. **数据效率高**：在足够数据下，纯数据驱动的DPK-GP可超越依赖物理方程投影的传统ROM方法。

### 方法的局限性
1. **分离性假设限制表达力**：尽管使用深度核缓解了问题，但乘积核本质上仍是**协方差可分离的**，可能无法捕捉强烈的跨维度交互效应（如参数-空间强耦合）。
2. **嵌入误差**：将非结构化网格映射到矩形背景网格会引入插值误差，影响最终预测质量。
3. **迭代求解开销**：对于“gappy-grid”情况，伪值求解依赖共轭梯度法（CG），收敛速度受条件数影响，可能需要预处理改进。
4. **内存占用仍较高**：虽比传统GP大幅降低，但在极端大规模问题中仍需进一步稀疏化或分布式实现。

### 未来工作方向
1. **部分非可分核设计**：开发允许特定维度间耦合的部分非可分核（partially non-separable kernels），例如低秩交叉因子修正项。
2. **减少嵌入误差**：研究更优的空间映射与插值策略，或直接在原始网格上构建近似Kronecker结构。
3. **提升gappy-grid求解效率**：探索有效的预处理器以加速CG收敛。
4. **融合物理信息**：引入PDE残差正则项，发展**Physics-Informed DPK-GP**，进一步提升小数据下的泛化能力。
5. **多输出联合建模**：扩展至向量场（如速度场、位移场）的联合建模，利用多输出GP结构。

---

> 📌 **总结**：本文提出的**DPK-GP框架是数据驱动PDE代理建模领域的一项重要进展**，它成功平衡了**预测精度、不确定性量化与计算可扩展性**三大关键需求，为科学计算中的高频查询任务（如UQ、优化、反演）提供了强有力的工具。

</details>

---

### 9. [3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs](https://arxiv.org/abs/2603.01376)

**Authors**: Mehdi Makni, Xiang Meng, Rahul Mazumder  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.01376v1  

#### Abstract
Sparse plus Low-Rank $(\mathbf{S} + \mathbf{LR})$ decomposition of Large Language Models (LLMs) has emerged as a promising direction in model compression, aiming to decompose pre-trained model weights into a sum of sparse and low-rank matrices $(\mathbf{W} \approx \mathbf{S} + \mathbf{LR})$. Despite...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）虽然在多项任务上表现出色，但其庞大的参数量导致部署时面临巨大的计算和内存开销，限制了在资源受限设备上的应用。**Sparse plus Low-Rank (S + LR)** 分解是一种有前景的模型压缩技术，旨在将预训练权重矩阵 $W$ 近似为稀疏矩阵 $S$ 和低秩矩阵 $L$ 的和（即 $W \approx S + LR$）。然而，现有的 (S + LR) 方法通常采用**交替最小化（alternating minimization）**策略，这种方法优化过程复杂，缺乏收敛保证，并且难以有效联合优化稀疏和低秩分量。

### 提出的新方法和新思路
本文提出了 **3BASiL-TM**，一个高效的、一次性的（one-shot）、后训练（post-training）的 (S + LR) 分解框架，以解决上述问题。

该框架由两个核心部分组成：

1.  **3BASiL (3-Block ADMM for Sparsity and Low-Rank Constraints)**：
    *   **方法**：提出了一种新颖的**三块交替方向乘子法（3-Block ADMM）**算法，用于逐层分解。该方法将稀疏分量 $S$、低秩分量 $L$ 和原始权重 $W$ 显式地建模为三个变量块，在一个统一的目标函数下进行联合优化。
    *   **创新**：与传统的交替最小化不同，3BASiL通过ADMM框架显式地捕捉了 $S$ 和 $L$ 之间的交互作用，提供了**可证明的收敛保证**，并能更有效地最小化重建误差。

2.  **Transformer-Matching (TM) Refinement**：
    *   **方法**：设计了一个高效的**Transformer级匹配（TM）**微调步骤。该步骤利用一个小的校准数据集，对整个Transformer块内的所有稀疏和低秩分量进行联合梯度优化，目标是最小化压缩模型与原始密集模型在Transformer输出层面的差异。
    *   **创新**：TM提供了一个比逐层重建更接近真实端到端损失的中间代理损失函数。它不仅能优化低秩分量，还能显著提升稀疏分量的质量。**最关键的是，TM具有通用性（universal）**，可以作为任何现有 (S + LR) 或纯稀疏压缩方法的增强模块。

### 相比现有方法的优势
*   **更高的精度**：在相同的压缩配置下，3BASiL-TM显著降低了困惑度（Perplexity），缩小了与原始密集模型的性能差距。
*   **更快的速度**：3BASiL算法本身比现有的SOTA (S + LR) 方法快数倍（例如，比HASSLE-free-ALPS快7倍以上）。
*   **更强的理论基础**：提供了ADMM算法的收敛性证明。
*   **通用的增强工具**：TM步骤可以独立应用于其他方法，普遍提升其性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
*   **校准数据集（Calibration Set）**：用于执行一次性压缩。从C4数据集的第一个分片中随机抽取128个文本段落，每个段落包含2048个token。
*   **评估数据集**：
    *   **困惑度（Perplexity）**：WikiText2 (WT2), Penn Treebank (PTB), C4 validation。
    *   **零样本任务（Zero-Shot Tasks）**：使用LM Harness框架评估8个任务，包括PIQA, ARC-Easy/Challenge, HellaSwag, Winogrande, RTE, OpenbookQA, BoolQ。

### 实验设置和评估指标
*   **模型**：主要在 **Llama-3** 和 **Llama-3.2** 系列模型（1B, 3B, 8B）上进行实验，并扩展到 **OPT-30B** 模型。
*   **压缩配置**：主要研究 **(N:M Sparse + Rank-r LR)** 配置，例如 (2:4 + 64LR)，这允许利用GPU的专用CUDA内核实现加速。
*   **评估指标**：
    *   **困惑度 (Perplexity)**：越低越好，衡量语言建模质量。
    *   **零样本准确率 (Zero-Shot Accuracy)**：越高越好，衡量下游任务能力。
    *   **压缩运行时间 (Compression Runtime)**：越短越好，衡量效率。
*   **LoRA微调**：在压缩后的模型上进行有限的LoRA微调，以验证压缩效果的可恢复性。

### 基线方法对比
*   **OATS**：首个为LLMs设计的 (S + LR) 压缩方法。
*   **HASSLE-free-SparseGPT (Hf-SparseGPT)**：结合了SparseGPT剪枝和交替最小化的 (S + LR) 框架。
*   **HASSLE-free-ALPS (Hf-ALPS)**：结合了ALPS剪枝和交替最小化的 (S + LR) 框架。
*   **EoRA**：一种快速的 (S + LR) 方法，仅执行一次低秩拟合。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
*   在 **Llama-8B** 模型上，采用 **(2:4 Sparse + 64 LR)** 配置时，3BASiL-TM 将 **WikiText2 困惑度与密集模型的差距减少了超过30%**。
*   在 **Llama-8B** 模型上，3BASiL-TM 的压缩速度比当前最先进的 (S + LR) 方法（HASSLE-free-ALPS）**快2.5倍以上**（在A100 GPU上）。

### 与基线方法的对比结果
*   **困惑度全面领先**：如表1和表2所示，在所有测试的 (N:M + LR) 配置下，`3BASiL-TM` 的困惑度均显著低于所有基线方法（OATS, Hf-SparseGPT, Hf-ALPS）。例如，在Llama-8B的(2:4+64LR)配置下，3BASiL-TM的C4困惑度为14.34，而最好的基线Hf-ALPS为16.15。
*   **零样本任务表现优异**：在零样本任务上，3BASiL-TM同样取得了最高的平均准确率（见表2 Avg列）。
*   **速度优势巨大**：图2显示，3BASiL的压缩运行时间远低于Hf-ALPS等基线，实现了数量级的加速。

### 消融实验结果
*   **3BASiL vs. 交替最小化**：图5展示了在第一层Transformer块中，3BASiL在优化目标函数（重建误差）方面明显优于交替最小化方法，证明了其联合优化的有效性。
*   **TM步骤的增益**：所有表格中，`-TM` 版本（如 `3BASiL-TM`）的性能都远超其无TM版本（如 `3BASiL`），表明TM步骤带来了巨大的性能提升。
*   **TM的通用性**：表3明确展示了TM步骤可以显著提升纯稀疏压缩方法（如Wanda, SparseGPT, ALPS）的性能，证明了其作为通用增强工具的价值。

---

## 4. 关键结论和发现

### 主要发现
1.  **联合优化至关重要**：通过3-Block ADMM进行联合优化，比传统的交替最小化方法能更有效地求解 (S + LR) 分解问题，从而获得更高精度的压缩模型。
2.  **Transformer级匹配是有效的**：在Transformer级别而非单层级别进行微调，能更好地保留模型的整体行为，是连接局部重建与全局性能的关键桥梁。
3.  **TM是一个强大的通用工具**：TM步骤不仅提升了3BASiL自身，也能普遍增强其他 (S + LR) 和纯稀疏方法，为模型压缩领域提供了一个即插即用的性能提升方案。
4.  **高效且高性能**：3BASiL-TM在保持甚至超越SOTA压缩精度的同时，实现了极高的压缩速度，使其成为一个实用且强大的工具。

### 方法的局限性
*   **计算资源依赖**：尽管3BASiL很快，但完整的3BASiL-TM流程仍需要一定的GPU资源来完成TM步骤。
*   **理论假设**：ADMM的收敛性证明依赖于惩罚参数 $p_t$ 的特定增长条件。
*   **未探索的优化空间**：论文提到，目前的方法为各层分配固定的稀疏/低秩配置，未来可以探索算法化地为不同层分配最优配置。

### 未来工作方向
*   探索将3BASiL-TM框架推广到**量化（Quantization）**或**量化-稀疏（Quantized-Sparse）**约束下的模型压缩。
*   设计专门的算法，为不同的网络层动态分配最优的稀疏率和低秩维度，以进一步优化效率-性能权衡。
*   研究如何将TM的思想集成到模型的训练过程中。

</details>

---

### 10. [Multi-Head Low-Rank Attention](https://arxiv.org/abs/2603.02188)

**Authors**: Songtao Liu, Hongwu Peng, Zhiwei Zhang, Zhengyu Chen, Yue Guo  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.02188v1  

#### Abstract
Long-context inference in large language models is bottlenecked by Key--Value (KV) cache loading during the decoding stage, where the sequential nature of generation requires repeatedly transferring the KV cache from off-chip High-Bandwidth Memory (HBM) to on-chip Static Random-Access Memory (SRAM) ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Multi-Head Low-Rank Attention》核心结论与实验结果总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在大语言模型（LLM）的长上下文推理过程中，解码阶段的 **Key-Value (KV) cache** 加载成为性能瓶颈。传统的 **Multi-Head Attention (MHA)** 和其变体如 **Multi-Head Latent Attention (MLA)** 虽然通过压缩 KV cache 减少了存储开销，但在采用 **Tensor Parallelism (TP)** 进行分布式解码时存在严重限制。

具体而言，**MLA** 将所有注意力头的信息压缩到一个单一的“潜在头”（latent head），导致该结构无法被分片（sharded）。因此，在 TP 场景下，每个设备都必须重复加载完整的 KV cache，造成内存带宽浪费，削弱了并行计算的优势。

---

### **提出了什么新方法或新思路**
本文提出了一种新的注意力机制：**Multi-Head Low-Rank Attention (MLRA)**，旨在解决 MLA 在 TP 下的分片瓶颈问题。

#### **核心思想：**
- 将 MLA 中不可分割的单个 latent head **显式分解为多个独立的 latent heads**（如 MLRA-2 使用 2 个，MLRA-4 使用 4 个）。
- 每个 latent head 独立进行上投影（up-projection）生成 NoPE 键值对，并将各分支的注意力输出求和。
- 这种设计使得 latent states 可以按块（block-wise）划分，从而支持高效的 **4-way Tensor Parallelism**。

#### **关键技术洞察：**
作者发现，MLA 中的上投影矩阵可以被划分为若干行块（row blocks），而对应的 KV 投影可表示为这些块的加权和。受此启发，MLRA 将这种“先分块再求和”的操作从 KV 计算转移到注意力输出层面，形成多个低秩分支。

---

### **相比现有方法的优势**
| 特性 | MLA | GQA | MLRA |
|------|-----|-----|-------|
| KV cache 大小 | 小（4.5dn） | 较大（2qdn） | 小（4.5dn） |
| 是否支持 TP 分片 | ❌ 不支持（全量复制） | ✅ 支持 | ✅ 支持（4-way） |
| 每设备 KV 加载量（4 GPU） | 4.5dn（不变） | 4dn | **1.5dn** |
| 解码速度 | 快 | 中等 | **更快（2.8× MLA）** |
| 模型质量 | 高 | 中 | **更高（SOTA）** |

> ✅ **MLRA 在保持 MLA 高效 KV 压缩的同时，首次实现了对 TP 的原生支持，显著降低每设备内存流量。**

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **预训练数据**：FineWeb-Edu-100B（98.3B 训练 token，0.1B 验证）
- **评估数据集（7个）**：
  - Wikipedia
  - C4
  - The Pile
  - RefinedWeb
  - Cosmopedia
  - FineWeb
  - FineWeb-Edu

- **下游零样本推理任务（7项常识推理基准）**：
  - ARC-Easy (ARC-E)
  - ARC-Challenge (ARC-C)
  - OpenBookQA
  - BoolQ
  - HellaSwag
  - Winogrande
  - PIQA

---

### **实验设置和评估指标**

#### **模型配置**
- 基于 **Llama-3 架构**
- 总参数量统一控制在约 **2.9B**
- 上下文长度：2048
- 所有模型从头预训练 100,000 步
- 使用 AdamW 优化器，cosine 学习率衰减，峰值学习率 1.6e-4
- 硬件：8×NVIDIA H100 80GB GPU

#### **评估指标**
| 类别 | 指标 |
|------|------|
| **模型质量** | Validation Perplexity（越低越好） |
| **下游能力** | Zero-shot Accuracy（越高越好） |
| **推理效率** | Decoding Latency（越低越好）、Throughput（越高越好） |

---

### **基线方法对比**
本文对比了多种主流注意力机制：
- **MHA**（标准多头注意力）
- **MQA**（Multi-Query Attention）
- **GQA**（Grouped-Query Attention）
- **MLA**（Multi-Head Latent Attention）
- **MFA**, **TPA**, **GLA-2/4**, **GTA**

其中，**MLA 是主要对比对象**，因其在 KV 压缩方面表现优异但缺乏 TP 支持。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 模型质量：验证困惑度（Perplexity）**
> **表3：平均验证 perplexity 对比（越低越好）**

| Method | Avg Perplexity |
|--------|----------------|
| MHA    | 13.860         |
| GQA    | 14.139         |
| MLA    | 13.727         |
| **MLRA-4** | **13.672** ✅ |
| MLRA-2 | 13.804         |

> 🔍 **MLRA-4 在所有模型中取得最低 perplexity，优于 MLA 和其他变体。**

#### **(2) 下游任务：零样本准确率**
> **表4：常识推理任务平均准确率（越高越好）**

| Method | Avg Accuracy (%) |
|--------|------------------|
| MHA    | 57.81            |
| GQA    | 57.89            |
| MLA    | 58.75            |
| **MLRA-4** | **58.84** ✅     |
| MLRA-2 | 58.68            |

> 🔍 **MLRA-4 同样达到最高零样本推理性能，表明其更强的语言理解能力。**

---

### **与基线方法的对比结果**

| 维度 | 结果 |
|------|------|
| **vs. MLA** | MLRA-4 perplexity ↓0.055，accuracy ↑0.09%，且支持 4-way TP |
| **vs. GQA** | perplexity ↓0.467，accuracy ↑0.95%，解码延迟更低 |
| **vs. GLA-2** | perplexity ↓0.107，accuracy ↑0.54%，TP 更高效（2.5dn → 1.5dn/device） |

---

### **消融实验结果**

#### **(1) 初始化策略（Zero vs. N(0,0.02)）**
- 使用 **zero initialization** 在所有模型上均优于随机初始化。
- 图1显示收敛更稳定，最终 loss 更低。

#### **(2) 缩放因子（Scaling）有效性**
- 应用 variance calibration（调整 query/KV latent 的方差）后，MLA、GLA-2、MLRA-2 均显著提升收敛速度和最终性能。
- 表39 显示 MLA 平均 perplexity 从 13.779 降至 13.727。

#### **(3) 注意力头数量加倍（Double Heads）**
- 将 GQA/MLA/GLA-2 的 head 数翻倍（如 24→48），固定 KV cache 大小。
- 结果：**性能下降**（表40），说明单纯增加 head 数无益，反而损害效果。

#### **(4) 引入门控机制（Gated Attention）**
- 在注意力输出前加入 sigmoid 门控（参考 Qiu et al., 2025）
- 所有模型（含 MLRA）均受益，进一步降低 perplexity。
- **MLRA-4 + gating 达到 13.621 平均 perplexity（SOTA）**

> 📊 **图4 显示引入 gating 后训练 loss 下降更快。**

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **MLRA 成功打破了 MLA 无法支持 Tensor Parallelism 的瓶颈**，通过将 latent state 分解为多个可分片的分支，实现真正的分布式高效解码。
2. ✅ **MLRA-4 在 2.9B 规模下达到了当前最优的 perplexity 和 zero-shot 推理性能**，超越 MLA、GQA 等主流方法。
3. ✅ **MLRA 实现了高达 2.8× 的解码速度提升（vs. MLA）**，并在长序列（达 2M tokens）场景下持续领先。
4. ✅ **4-way TP 下，MLRA-4 每设备 KV cache 加载仅为 1.5dn**，远低于 MLA 的 4.5dn（不随 TP 缩减）。
5. ✅ **variance scaling 和 gated attention 等技巧能进一步提升性能**，验证了架构设计的可扩展性。

---

### **方法的局限性**
1. **仅支持 4-way TP**：目前 MLRA 设计基于固定的 4 分支结构，灵活性不如 GQA 可适配任意 TP 数。
2. **未探索更大规模模型**：实验集中在 2.9B 模型，尚未验证在 10B+ 模型上的泛化性。
3. **硬件依赖性强**：高性能依赖定制 kernel（基于 FlashAttention-3 实现），通用性受限。
4. **理论分析简化**：Assumption 1（权重 i.i.d.）在训练后未必成立，实际 variance behavior 需更多研究。

---

### **未来工作方向**
1. **推广至任意 TP 度数**：设计动态可扩展的 latent head 分组机制。
2. **结合量化技术**：将 MLRA 与 KV cache 量化（如 KIVI、SnapKV）结合，进一步压缩内存占用。
3. **应用于 MoE 架构**：探索 MLRA 在 Mixture-of-Experts 中的路由与缓存协同优化。
4. **构建端到端推理系统**：集成 MLRA 到 vLLM、SGLang 等框架，实测真实服务吞吐。
5. **探索更复杂的融合方式**：替代简单的 branch-wise sum，尝试 weighted 或 learned fusion。

---

> 💡 **总结一句话**：  
> **MLRA 是首个兼具高模型质量与强分布式解码效率的注意力机制，在保持 MLA 高效 KV 压缩的同时，通过多分支低秩设计原生支持 Tensor Parallelism，实现了推理性能与质量的双重突破。**

</details>

---

### 11. [Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning](https://arxiv.org/abs/2603.01639)

**Authors**: Jiebin Zhang, Zhenghan Yu, Liang Wang, Nan Yang, Eugene J. Yu, Zheng Li, Yifan Song, Dawei Zhu, Xingxing Zhang, Furu Wei, Sujian Li  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.01639v1  

#### Abstract
Speculative decoding accelerates large language model (LLM) inference by using a small draft model to generate candidate tokens for a larger target model to verify. The efficacy of this technique hinges on the trade-off between the time spent on drafting candidates and verifying them. However, curre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning

## 1. 主要贡献和创新点

### 解决的问题
当前主流的 **speculative decoding** 方法（如 Eagle3）在加速大语言模型（LLM）推理时，通常采用静态策略来决定 draft tree 的深度（draft depth）和验证 token 数量（verification size）。这些方法存在以下问题：

- **忽略真实时间成本**：多数动态方法优化的是代理指标（proxy metrics），例如 *acceptance length*（每轮接受的 token 数），而忽略了 drafting 和 verification 阶段的实际 wall-clock 时间开销。
- **阶段孤立优化**：drafting 和 verification 被视为独立过程，缺乏协同机制，导致次优决策。
- **缺乏自适应性**：固定参数无法应对不同输入上下文的复杂度变化。

### 提出的新方法：LTD (Learning to Draft)
本文提出 **LTD**，一种基于 **Reinforcement Learning (RL)** 的新型 speculative decoding 框架，其核心思想是：

- 将 draft-and-verify 循环建模为一个 RL 环境，直接以 **throughput**（单位时间内生成的有效 token 数）作为奖励信号进行优化。
- 设计两个 **co-adaptive policies**：
  - **Depth Policy (Tp)**：动态控制 draft tree 的深度，决定何时停止生成候选 token。
  - **Size Policy (Ty)**：动态选择提交给 target model 验证的候选 token 数量（verification size）。
- 通过迭代训练使两个策略相互适应，共同学习如何权衡 **acceptance length** 与 **time cost**，从而最大化整体吞吐量。

### 相比现有方法的优势
- ✅ **直接优化目标更合理**：以 throughput 为 reward，而非间接指标（如 acceptance length），确保加速效果真实有效。
- ✅ **联合优化 drafting 与 verification**：两个策略协同工作，避免“draft 很强但 verify 不跟上”或“verify 能力强但 draft 太保守”的失配问题。
- ✅ **高度自适应**：能根据不同上下文动态调整策略，提升鲁棒性和泛化能力。
- ✅ **轻量级设计**：两个 policy 均为小型 MLP，引入的计算开销极低（<1.5%）。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练集**：`HumanEval`（用于 RL policy 的训练）
- **测试集 / 评估任务**（共四个基准任务）：
  - `MT-bench`：多轮对话
  - `GSM8K`：数学推理
  - `Alpaca`：指令遵循
  - `Natural Questions (NQ)`：问答任务
- **泛化性验证**：`MMLU`（57个子任务，涵盖数学、历史、法律等多领域）

### 实验设置
- **基础框架**：构建于当前 SOTA 方法 **Eagle3** 之上，替换其静态策略。
- **Policy 架构**：
  - Depth Policy：单层 FFN（1024 units），低延迟设计。
  - Size Policy：两层 FFN（[1024, 256]），平衡性能与效率。
- **RL 算法**：使用 **PPO** 进行训练，reward 定义为：
  $$
  R = \frac{L_A}{T_{\text{draft}} + T_{\text{verify}}}
  $$
  其中 $L_A$ 是 acceptance length，$T$ 是实际耗时。
- **训练流程**：
  1. **初始独立训练**：分别训练 depth 和 size policy。
  2. **迭代协同优化**：交替冻结一方，优化另一方，实现 co-adaptation。

### 评估指标
- **Speedup**：相对于 vanilla auto-regressive decoding 的加速比。
- **Average Acceptance Length ($\bar{L}_A$)**：每轮 draft-and-verify 接受的平均 token 数。
- **Throughput**：每秒生成的有效 token 数（隐含在 speedup 中）。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| 默认基线 | Eagle3 (static) |
| 动态 depth 方法 | DDD, SVIP, Gammatune, Disco, SpecDec++ |
| 动态 size 方法 | C2T |
| 强基线 | Grid Search (GS) on Eagle3 |
| 消融变体 | -Size, -Depth, -Iterative Training |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Greedy Decoding）
在五种主流 LLM 上均取得显著提升：

| 模型 | LTD vs Eagle3 Speedup Gain | 最高加速比 (LTD) |
|------|----------------------------|------------------|
| Qwen3-32B | **+36.4%** | 2.47× |
| Deepseek-Distilled-Llama-8B | +9.5% | 4.16× |
| Llama-3-8B | +6.5% | 3.92× |
| Vicuna-13B | +5.0% | 4.32× |
| Qwen3-14B | +4.0% | 2.37× |

> 📌 **注**：对于 Qwen 系列，因无默认配置，对比对象为 Grid Search 结果，LTD 仍大幅领先。

### 与基线方法的对比结果
- 在所有模型和任务上，**LTD 均优于所有动态方法**（DDD、SVIP、C2T 等）及 Grid Search。
- 即便某些方法获得更高的 acceptance length，其 speedup 仍低于 LTD，说明其 time cost 更高。
- **高温度采样 (Temperature=1.0) 下表现尤为突出**：
  - 多数动态方法在此场景下失效，性能下降明显。
  - LTD 保持稳定增益（约 +5%），展现出更强的鲁棒性。

### 消融实验结果
#### （1）组件贡献分析（Vicuna-13B）
| 变体 | 平均 Speedup | 平均 $\bar{L}_A$ |
|------|--------------|------------------|
| Eagle3 | 4.11× | 6.35 |
| -Size | 4.29× | 6.57 |
| -Depth | 4.13× | 6.84 |
| -Iterative Training | 4.20× | 7.06 |
| **LTD (完整)** | **4.32×** | **7.00** |

✅ **结论**：
- Depth Policy 贡献最大，能有效减少无效 drafting。
- Iterative training 显著提升协同效果，证明 co-adaptation 必要性。

#### （2）Reward 函数对比（Size Policy）
| Reward 类型 | 平均 Speedup | 平均 $\bar{L}_A$ |
|------------|--------------|------------------|
| Acceptance Length | 3.59× | **6.89** |
| Time Cost | 3.45× | 5.66 |
| **Throughput (LTD)** | **4.13×** | 6.84 |

✅ **结论**：仅优化 acceptance length 会导致过度验证，增加 time cost；而 throughput 作为 reward 能自然平衡二者。

#### （3）观察空间消融（Observation Space）
| 输入特征 | 平均 Speedup |
|---------|--------------|
| P+D+L (LTD) | **4.13×** |
| P+D | 4.07× |
| P+L | 4.05× |
| P+D+L+H (加 hidden states) | 4.10× |
| P+D+L+E (加 entropy) | 4.11× |

✅ **结论**：加入计算昂贵的 hidden states 或 entropy 并未带来收益，反而可能因 overhead 降低性能。LTD 的轻量状态设计更高效。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **Throughput 是更优的优化目标**：直接优化 throughput 比优化 acceptance length 更能反映真实加速效果。
2. 🤝 **Co-adaptation 至关重要**：drafting 与 verification 应联合优化，单一模块改进难以发挥全部潜力。
3. ⚖️ **动态权衡优于固定策略**：LTD 学会根据上下文难度自动切换策略：
   - 简单文本 → 深而窄（deep & narrow）
   - 困难文本 → 浅而宽（shallow & wide），提高容错率。
4. 🧠 **轻量 RL 策略可行且高效**：两个 MLP policy 的额外开销 <1.5%，却带来高达 36.4% 的加速增益，性价比极高。
5. 🌐 **强泛化能力**：在 MMLU 的 57 个子任务中，LTD 在 54 项上超越 Eagle3，尤其在数学与逻辑类任务中优势显著。

### 方法的局限性
- 依赖于高质量的 draft model（如 Eagle3），若 draft model 性能差，LTD 提升有限。
- 当前 policy 训练需额外 RL 开销（约 30 GPU-hours），虽可摊销但仍有一定门槛。
- 目前仅支持 tree-based speculative decoding，是否适用于 chain-based 场景有待验证。

### 未来工作方向
- 将 LTD 思想扩展到 **self-speculative decoding** 或 **multi-agent drafting** 场景。
- 探索 **online adaptation**，让 policy 在部署过程中持续学习。
- 结合 **hardware-aware modeling**，进一步优化 GPU 利用率与内存访问模式。
- 研究 **zero-shot transfer**，使在一个模型上学到的 policy 能迁移到其他架构的 LLM。

---

> ✅ **总结一句话**：  
> **LTD 通过强化学习实现了 drafting 与 verification 的协同自适应优化，以 throughput 为直接目标，在多种 LLM 和任务上实现了 SOTA 加速效果，并具备良好的鲁棒性与泛化能力。**

</details>

---

### 12. [Optimizing In-Context Demonstrations for LLM-based Automated Grading](https://arxiv.org/abs/2603.00465)

**Authors**: Yucheng Chu, Hang Li, Kaiqi Yang, Yasemin Copur-Gencturk, Kevin Haudek, Joseph Krajcik, Jiliang Tang  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.00465v1  

#### Abstract
Automated assessment of open-ended student responses is a critical capability for scaling personalized feedback in education. While large language models (LLMs) have shown promise in grading tasks via in-context learning (ICL), their reliability is heavily dependent on the selection of few-shot exem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimizing In-Context Demonstrations for LLM-based Automated Grading

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Models (LLMs)** 的自动评分系统依赖于 **In-Context Learning (ICL)**，其性能高度敏感于所选的 few-shot 示例（exemplars）和 rationale 质量。传统方法通常基于语义相似性检索示例，但这类方法难以捕捉评分标准（rubric）中的**细微决策边界**，尤其在学生回答处于“边缘情况”时容易出错。此外，高质量专家撰写的 **rationale** 构建成本高昂，限制了系统的可扩展性。

### 🚀 提出的新方法：GUIDE
作者提出 **GUIDE (Grading Using Iteratively Designed Exemplars)**，一个将 exemplar 选择与优化重构为**边界感知的迭代优化框架**，核心思想是：
- 将 exemplar 选择视为一个**边界定义问题**，而非简单的语义匹配。
- 引入**对比操作符（contrastive operators）** 主动识别“边界对”（boundary pairs）——即语义相近但得分不同的样本。
- 在生成阶段，通过 **discriminative rationale generation** 自动生成能明确解释“为何得此分而非相邻等级”的高区分度推理链。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | GUIDE |
|------|--------|-------|
| **Exemplar Selection** | 基于语义相似性（如 KNN-SBERT），易忽略关键差异 | 主动寻找 boundary pairs，聚焦决策边缘 |
| **Rationale 生成** | 手工编写或通用 CoT，缺乏针对性 | 对比填充（contrastive infill），强调排除相邻分数的理由 |
| **优化目标** | 最大化整体准确率（如 BRIDGE） | 多目标优化：准确性 + 稀疏性 + **contrastive density** |
| **可扩展性** | 依赖大量人工标注 | 可从少量种子开始，自动生成高质量 exemplars |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验涵盖三个教育领域的真实数据集，覆盖不同评分场景：

| 数据集 | 领域 | 标签类型 | 样本数 | 特点 |
|-------|------|---------|--------|------|
| **Dr** | 物理科学（电相互作用） | 二分类 {0,1} ×11 rubrics | 314 | 无专家 rationale，真实高中生作答 |
| **Dc** | 化学教育（3DLP 框架） | 三分类 {0,1,2} ×2 任务 | ~163–184 | 分 DCI 和 SEP 两个维度评分 |
| **DT** | 教师教育（数学教学知识） | 三分类 {0,1,2} ×4 任务 | ~229–236 | 含少量专家 rationale（每类3–5个） |

划分比例均为 **train:valid:test = 3:1:1**

### ⚙️ 实验设置
- **模型基础**：使用 `GPT-4o-mini` 进行所有生成、评分与 embedding。
- **Embedding 模型**：`text-embedding-3-small` 用于计算语义相似度。
- **优化轮次**：$ T = 5 $ 轮迭代。
- **候选池大小**：最大 $ N_{\text{max}} = 512 $
- **演示集大小约束**：$ |E| \in [4, 16] $
- **边界对阈值**：cosine similarity ≥ 0.7 且标签差为1

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 完全匹配正确率 |
| **Quadratic Weighted Kappa (QWK)** | 序数评分常用指标，惩罚偏离程度（如 0→2 比 0→1 更严重） |
| **Adjacent Error Rate (AdjErr)** | 预测评分与真实评分相差 ±1 的比例（反映边界模糊） |
| **Non-Adjacent Error Rate (NonAdjErr)** | 跨级错误（如 0→2），代表严重逻辑失败 |

### 🆚 基线方法对比
| 方法 | 类型 | 简介 |
|------|------|------|
| **NAIVE** | 固定集 | 不进行优化，直接使用初始 exemplars |
| **RANDOM** | 随机采样 | 随机选取 k 个 exemplars |
| **KNN-SBERT** | 动态检索 | 按语义相似性为每个查询检索最近邻 |
| **Vote-K** | 多样性选择 | 选择语义分散的 exemplars 以覆盖更广空间 |
| **BRIDGE** | 优化生成 | 先前最优的 optimize-generate 方法，最大化验证集准确率，但不关注边界 |

---

## 3. 主要实验结果和性能指标

### 📈 总体性能表现（见 Table 2）
| 方法 | Dr (Acc/QWK) | Dc (Acc/QWK) | DT (Acc/QWK) |
|------|---------------|--------------|--------------|
| NAIVE | 0.74 / 0.42 | 0.69 / 0.39 | 0.59 / 0.54 |
| KNN-SBERT | 0.78 / 0.44 | 0.58 / 0.26 | 0.52 / 0.52 |
| BRIDGE | 0.90 / 0.57 | 0.76 / 0.53 | 0.66 / 0.65 |
| **GUIDE (Ours)** | **0.92 / 0.62** | **0.80 / 0.59** | **0.71 / 0.67** |

> ✅ GUIDE 在所有数据集上均取得**最佳 Accuracy 和 QWK**，显著优于基线。

### 🔁 边界错误分析（AdjErr）
| 方法 | Dr (AdjErr) | Dc (AdjErr) | DT (AdjErr) |
|------|------------|------------|------------|
| NAIVE | 0.26 | 0.31 | 0.37 |
| BRIDGE | 0.19 | 0.24 | 0.32 |
| **GUIDE** | **0.08** | **0.20** | **0.28** |

> ✅ GUIDE 将相邻错误率（AdjErr）大幅降低，尤其是在 **Dr 上减少超过 60%**，说明其有效解决了“边缘案例”判分难题。

### ❌ 严重错误控制（NonAdjErr）
| 方法 | Dr | Dc | DT |
|------|----|----|----|
| GUIDE | 0.00 | 0.00 | 0.02 |

> ✅ NonAdjErr 几乎为零，表明 GUIDE 不会因追求边界精度而引入更严重的跨级误判。

### 💡 消融实验发现（文中未列详细表格，但讨论明确指出）：
- **去除 contrastive operators** → AdjErr 显著上升，说明边界对的选择至关重要。
- **使用普通 CoT 替代 discriminative rationale** → 性能下降，验证了“解释为何不是其他分数”的必要性。
- **固定 exemplar 集合不更新 rationale** → 后续轮次提升停滞，证明 rationale 反馈循环的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **边界案例是自动化评分的核心挑战**  
   学生回答常位于两个评分等级之间，仅靠语义相似性无法区分，必须显式建模决策边界。

2. **contrastive density 是比 accuracy 更优的优化信号**  
   单纯追求全局准确率（如 BRIDGE）不足以教会模型理解 rubric 的精细结构；引入“边界密度”作为正则项可显著提升鲁棒性。

3. **discriminative rationale 提供更强的学习信号**  
   明确说明“为什么不是 0 或 2”比单纯说“这是 1”更能帮助 LLM 掌握评分逻辑。

4. **GUIDE 实现高效冷启动（cold-start）**  
   即使只有极少数专家 rationale，也能通过迭代合成高质量 exemplars，适用于新课程或动态 curriculum。

### ⚠️ 局限性
- 当前方法依赖 LLM 多次调用，在训练阶段有一定计算开销（尽管单次约 $5–8，属可接受范围）。
- boundary pair 的定义基于文本 embedding，可能无法很好处理**多模态输入**（如图表、公式推导）。
- 假设评分标准是明确且有序的（ordinal），对开放性评分任务适用性待验证。

### 🔮 未来工作方向
- 扩展到 **multimodal grading** 场景，结合图像、代码等模态定义“语义邻居”。
- 探索 **human-in-the-loop** 版本，让教师参与修正 boundary pairs 或 rationale。
- 将 GUIDE 思想应用于其他需要精细判断的任务，如写作反馈、辩论评价等。

---

## ✅ 总结
**GUIDE** 成功将 LLM-based 自动评分的关注点从“整体准确率”转向“边界精确性”，通过**迭代式的 contrastive selection + discriminative rationale generation**，构建了一个既能泛化又能深挖 rubric 细节的可靠评分系统。其实验结果充分证明了该方法在多种教育场景下的优越性和实用性，为实现**可信、可扩展、符合人类教学标准的 AI 评卷系统**铺平了道路。

</details>

---

### 13. [Agents Learn Their Runtime: Interpreter Persistence as Training-Time Semantics](https://arxiv.org/abs/2603.01209)

**Authors**: Victor May, Aaditya Salgarkar, Yishan Wang, Diganta Misra, Huu Nguyen  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.01209v1  

#### Abstract
Tool-augmented LLMs are increasingly deployed as agents that interleave natural-language reasoning with executable Python actions, as in CodeAct-style frameworks. In deployment, these agents rely on runtime state that persists across steps. By contrast, common training pipelines treat agent traces a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Agents Learn Their Runtime: Interpreter Persistence as Training-Time Semantics

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文探讨了一个在 **Tool-augmented LLM agents**（工具增强型大语言模型智能体）中被广泛忽视的问题：**interpreter persistence（解释器持久性）是否仅仅是推理时的运行时机制（runtime scaffold），还是一个在训练阶段就被学习到的语义特征（training-time semantic）？**

当前许多智能体框架（如 CodeAct、ReAct）允许模型通过执行 Python 代码与外部环境交互，并将中间状态存储在持久化解释器中。然而，在训练过程中，用于微调的数据（即 agent traces）通常来自特定运行时环境，其执行语义（如状态是否持久）往往被隐式处理。这导致一个关键风险：**如果训练时的状态是持久的，而部署时切换为无状态解释器，模型是否会失败？反之亦然？**

### 提出了什么新方法或新思路
论文提出了以下三个核心创新点：

- **Execution Semantics as a Training Variable**  
  首次明确提出 **interpreter persistence** 应被视为 agent 数据集中的“头等语义”（first-class semantic），而非隐藏的实现细节。作者设计了一个受控的 **2×2 实验**，系统性地解耦了 *训练语义*（persistent vs. stateless traces）与 *运行时语义*（persistent vs. stateless runtime），以验证其影响。

- **OPAQUE KNAPSACK：一个非坍缩（non-collapsible）基准任务**  
  设计并开源了一个新的部分可观测优化任务 **OPAQUE KNAPSACK**，其特点包括：
  - 物品属性和类别约束不可见，需通过预算化的 `inspect()` 工具获取；
  - 存在硬性的工具调用次数限制；
  - 允许的物品类别是隐藏的，需从 `take_item()` 的反馈中推断；
  这些设计强制要求多轮交互、状态维护和计划修订，防止模型通过单次脚本解决任务。

- **Paired Trace Generation Pipeline**  
  对每个任务实例生成成对的训练轨迹（paired trajectories），仅在 **persistence/reset 执行契约** 上不同，其余（任务、提示、工具接口、监督信号）完全一致，从而精确控制变量。

### 相比现有方法的优势
- **揭示了训练-推理错配的根本风险**：现有方法通常假设运行时可迁移，本文证明 **persistence 是一种必须对齐的 learnable behavioral prior（可学习的行为先验）**。
- **提供了可复现的诊断工具**：通过引入“amnesia tax”、“cascading recovery loops”等概念，量化了效率与稳定性代价。
- **推动了 agent 数据集标准化**：呼吁将 execution semantics 明确纳入 agent 数据协议（如 ADP）。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **OPAQUE KNAPSACK**：本文提出的新任务，分为两个难度级别：
  - **Easy**：用于训练和领域内评估（1,000 个训练实例，100 个测试实例）；
  - **Hard**：用于跨难度泛化评估（100 个测试实例），具有更大的问题规模和更紧的约束。

### 实验设置和评估指标
#### 实验设计（2×2 Cross-Evaluation）
| 训练语义 \ 运行时语义 | Persistent Runtime | Stateless Runtime |
|----------------------|--------------------|-------------------|
| **Persistent-Trained** | ✅ 对齐条件         | ❌ 错配条件        |
| **Stateless-Trained** | ❌ 错配条件         | ✅ 对齐条件         |

- **模型**：基于 **Qwen3-8B**，使用 **QLoRA** 微调两个独立的 LoRA 适配器。
- **训练数据**：由 **Gemini 3 Flash** 作为教师模型生成的 interleaved reasoning-action 轨迹，每种语义各 1,000 条。
- **few-shot demonstrations**：明确体现执行契约（如持久化中重用变量 vs. 无状态中重新定义）。

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Normalized Optimality (%)** | 模型获得的价值 / 最优价值，衡量求解质量 |
| **Exact Solved (#)** | 达到最优解的任务数量 |
| **Total Tokens** | 整个 episode 的总 token 数（prompt + completion），衡量推理开销 |
| **Steps** | 交互步数 |
| **Wall-clock Time (s)** | 实际运行时间 |
| **Score / 1k Tokens** | 单位推理成本下的得分，衡量效率 |
| **Unresolved Reference Errors** | 出现 `NameError` 等未定义变量错误的比例 |
| **Execution Instability** | 是否进入高错误密度的恢复循环 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Hard Split，n=100）

| Train → Runtime | Normalized Optimality (%) | Total Tokens | Score / 1k Tokens |
|------------------|----------------------------|---------------|---------------------|
| **Persistent → Persistent** | **75.4 ± 4.7** | **18,612** | **4.05** |
| **Stateless → Stateless** | 67.7 ± 6.3 | 67,898 | 1.00 |
| **Stateless → Persistent** | 72.5 ± 4.3 | 54,665 | 1.33 |
| **Persistent → Stateless** | 68.2 ± 6.9 | 67,925 | 1.00 |

> 注：所有 fine-tuned 模型显著优于 base model（~2.2%），但不同 fine-tuned 配置间的 optimality 差异 **不具统计显著性**（Wilcoxon test, p > 0.05）。

### 与基线方法的对比结果
- **效率差异巨大**：
  - 对齐的 **Persistent → Persistent** 配置比 **Stateless → Stateless** 少用 **约 3.5× 的 token**。
  - 即使 **Stateless-trained** 模型部署在 **Persistent runtime** 中，仍消耗 54,665 tokens（是前者的 2.9×），因其仍重复导入状态。
- **稳定性差异显著**：
  - **Persistent-trained → Stateless runtime** 导致 **~80% 的 episodes 出现 unresolved reference errors**，陷入 cascading recovery loops，token 预算耗尽却无进展。
  - 其他配置几乎无此类错误。

### 消融实验结果（Behavioral Metrics）

| Train → Runtime | State Utilization | Imports/Step | Context Symbol Lifespan |
|------------------|--------------------|--------------|--------------------------|
| **Persistent → Persistent** | 1.11 | 0.31 | 2.32 |
| **Stateless → Stateless** | 0.00 | 1.00 | 0.00 |
| **Persistent → Stateless** | 0.02 | 0.86 | **5.60** |
| **Stateless → Persistent** | 0.00 | **1.00** | 2.49 |

- **State Utilization > 0** 仅出现在 **Persistent-trained → Persistent runtime**，表明真正的可执行状态重用。
- **Imports/Step ≈ 1.0** 在所有 **Stateless-trained** 模型中出现，即使 runtime 支持持久化，说明这是 **learned prior**。
- **Context Symbol Lifespan** 在错配条件下飙升至 5.60，表明模型仍在文本中引用已失效的变量。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **Interpreter persistence is a learnable training-time semantic**，而非零样本能力。模型会学习依赖于训练期间暴露的执行契约。
2. ✅ **训练-推理语义错配会导致严重后果**：
   - **Persistent-trained → Stateless**：引发 **cascading recovery loops** 和 **~80% 的 missing-variable errors**，稳定性崩溃。
   - **Stateless-trained → Persistent**：支付 **“amnesia tax”**，冗余地在文本中重建状态，效率低下（+3.5× tokens）。
3. ✅ **Execution semantics primarily affect *how* agents solve tasks, not *whether***：求解质量（optimality）差异不显著，但推理成本和行为模式截然不同。
4. ✅ **对齐训练与部署语义可显著降低推理开销**：在相同运行时下，**Persistent-trained** 模型比 **Stateless-trained** 更高效。

### 方法的局限性
- **评估能力有限**：n=100 的样本量不足以检测 optimality 上的小幅差异。
- **Token 预算混淆**：Stateless 训练集天然包含更多 token，未来需进行 token-matched 消融。
- **泛化范围有限**：实验基于单一任务族（knapsack）和单一模型（Qwen3-8B）。
- **共现线索干扰**：尽管提供了 `active_globals` 元数据，但仍需进一步分离 visibility 与 persistence 的影响。

### 未来工作方向
- 验证结论在 **多样化任务**（如数学推理、软件工程）和 **不同模型尺度** 下的普适性。
- 探索 **介于完全持久与完全无状态之间的中间态运行时**（如 selective persistence）。
- 将 **execution semantics** 正式纳入 **Agent Data Protocol (ADP)** 等标准。
- 研究如何通过 **instruction tuning** 或 **RL** 让模型具备更强的运行时适应能力。

</details>

---

### 14. [SimpleTool: Parallel Decoding for Real-Time LLM Function Calling](https://arxiv.org/abs/2603.00030)

**Authors**: Xiaoxin Shi, Jiaxin Wan, Linkang Dong, Wei Jiang, Yue Liu, Zengfeng Huang  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.00030v1  

#### Abstract
LLM-based function calling enables intelligent agents to interact with external tools and environments, yet autoregressive decoding imposes a fundamental latency bottleneck that limits real-time applications such as embodied intelligence, game AI, and interactive avatars (e.g., 10 Hz control frequen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*SimpleTool: Parallel Decoding for Real-Time LLM Function Calling*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **LLM 函数调用中的延迟瓶颈**：传统基于 `autoregressive decoding` 的 LLM 在生成函数调用（如 JSON 或 API 调用）时存在显著延迟，难以满足实时应用需求（如具身智能、游戏 AI、交互式虚拟人等要求 5–30 Hz 控制频率的场景）。
- 当前加速方法（如 speculative decoding、constrained decoding）在短输出任务中收益有限，且无法突破逐 token 解码的根本限制。

### 提出了什么新方法或新思路
提出 **SimpleTool**，一种专为 LLM 函数调用设计的并行解码框架，其核心思想是：
- 利用函数调用输出的两个特性：
  1. **结构冗余性**（structural redundancy）：大量低熵 token（如 `{`, `"name":`, 分隔符等）可被压缩；
  2. **弱因果依赖性**（weak causal dependencies）：函数参数之间通常无强顺序依赖，可独立生成。
- 设计 **special tokens**（如 `<function>`, `<arg1>`...），兼具双重作用：
  - **压缩冗余 token**（实现 4–6× 输出长度压缩）；
  - **作为模式选择器**（mode selector），引导模型并行生成函数名和各参数。

架构上采用多头并行解码（parallel decoding heads），所有 head 共享输入前缀的 KV Cache，仅附加不同的 special token 来区分任务流（function 名、arg1、arg2…），从而实现真正的端到端并行。

### 相比现有方法的优势
| 方法 | 是否解决冗余 | 是否支持并行 | 是否利用 idle compute | 适用性 |
|------|---------------|----------------|--------------------------|--------|
| **Medusa / EAGLE** | 否 | 是（推测式） | 是 | 通用文本生成 |
| **SGLang / Outlines** | 部分（语法约束） | 否（仍自回归） | 否 | 结构化输出 |
| **SimpleTool (Ours)** | ✅（显式压缩） | ✅（确定性并行） | ✅（内存带宽瓶颈下近零开销） | 专用于 function calling |

- **端到端加速 3–6×，最高达 9.6×**，远超现有方法在短输出场景下的增益；
- **与量化、speculative decoding 正交兼容**，可进一步叠加优化；
- **保持甚至提升准确率**，而非以精度换速度。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
共五个 benchmark，覆盖多样化函数调用场景：
- **BFCL-v3**（Berkeley Function Calling Leaderboard）：主流工具调用评测集，包含单轮与多轮、简单与复杂调用。
- **Mobile Actions**（Google AI, 2025）：面向移动端设备控制的任务，如“打开手电筒”、“导航至最近书店”，强调边缘部署能力。
- **SealTools**：自指令学习构建的工具学习数据集。
- **OpenFunction**：真实世界 API 接口测试集。
- **ToolAlpaca**：模拟 3000 个工具使用案例的数据集。

> 所有 benchmark 中，**>6 参数的函数仅占 4.76%**，因此默认设置 6 个 argument head 已覆盖 95.2% 场景。

### 实验设置和评估指标

#### 模型系列
- 基于 **Qwen2.5-Instruct** 系列（0.5B–14B）及 **Qwen3-4B-Instruct** 进行 LoRA 微调。
- SimpleTool 模型命名为 `ST-Qwen2.5-*B`。

#### 硬件平台
- 主要推理测试在 **RTX 4090** 和 **H100** 上进行。
- 支持 AWQ 4-bit 量化部署。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Overall Accuracy** | 函数名 + 所有参数完全正确 |
| **Function Accuracy** | 仅函数名选择正确 |
| **P50 / P90 Latency** | 中位数与第 90 百分位延迟，反映典型与尾部性能 |
| **Speedup Ratio** | 相对于 baseline 的端到端延迟加速比 |

#### 基线方法对比
- **Baseline**: 原始 Qwen 模型 + JSON 格式输出；
- **vLLM + Vanilla**: 使用 vLLM 推理引擎的基础版本；
- **Transformers**: HuggingFace 原生实现；
- **FunctionGemma (270M)**：Google 发布的轻量级边缘函数调用模型，用于横向比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ⚡ 推理速度与延迟
| 模型 | Backend | Speedup | P50 Latency (ms) |
|------|---------|--------|------------------|
| Qwen3-4B (Baseline) | vLLM | 1.0× | 468.5 |
| Qwen2.5-0.5B (Baseline) | vLLM | 4.3× | 110.1 |
| **ST-Qwen-0.5B (Ours)** | vLLM | **9.2×** | **51.0** |
| **ST-Qwen-4B + AWQ** | vLLM | — | **61.2** |

- **SimpleTool 实现 3–6× 端到端加速，最大达 9.6×**（图1a）；
- 结合 AWQ 量化后，**4B 模型可达 61.2ms P50 延迟 → 支持 16 Hz 实时控制**，首次在消费级 GPU 上实现亚百毫秒函数调用；
- 并行 head 数从 2 增加到 8，延迟几乎不变，验证了 **parallelization overhead < 8.2%**，接近“免费操作”。

#### ✅ 准确率表现
| 模型 | Overall Acc (%) | Function Acc (%) |
|------|------------------|-------------------|
| Qwen2.5-0.5B (Baseline) | 70.9 | 90.2 |
| **ST-Qwen2.5-0.5B (Ours)** | **75.3** | **98.0** |
| Qwen2.5-14B (Baseline) | 82.0 | 91.7 |
| **ST-Qwen2.5-14B (Ours)** | **85.5** | **99.3** |

- 在所有模型尺度上均取得 **+0.5% ~ +3.5% 的整体准确率提升**；
- **Function Accuracy 提升尤为显著（平均 +7.1%）**，得益于 special tokens 明确引导输出结构，减少歧义；
- 在 **Mobile Actions** 上，`ST-Qwen-0.5B` 零样本准确率达 **69.3%**，优于 FunctionGemma（58.0%），微调后达 **86.2%**，超越后者（85.0%）。

#### 🔍 Token 压缩效果（Table 1）
| 模型 | 平均压缩比 (CR) | P50 压缩比 |
|------|------------------|------------|
| Qwen2.5-0.5B | 5.06× | 6.00× |
| Qwen2.5-7B | 5.35× | 5.00× |
| **平均** | **4.66×** | **5.02×** |

- 输出 token 数量减少 **4–6 倍**，直接降低 decode 阶段计算负担；
- bottleneck head（最长 head）平均仅需生成 **~8–9 个高熵 value tokens**，而 baseline 需生成 30–50 个。

### 与基线方法的对比结果
| 对比维度 | SimpleTool 优势 |
|--------|----------------|
| **vs FunctionGemma** | 更小模型（0.5B vs 270M）却更快更准；P90 尾延迟仅 74.5ms vs 139.5ms，更适合实时系统 |
| **vs Medusa/EAGLE** | 不依赖 draft model，无验证失败风险；专为结构化输出优化，压缩 + 并行双管齐下 |
| **vs SGLang/XGrammar** | 虽然也支持 constrained decoding，但仍为自回归；SimpleTool 实现真正并发生成 |

### 消融实验结果（Table 6）
| 设置 | Accuracy (%) | 观察 |
|------|--------------|------|
| LoRA Rank 64 | 85.5 | 容量不足，head 间干扰严重 |
| LoRA Rank 512 | 86.3 | 默认配置，平衡性能与成本 |
| LoRA Rank 1024 | **87.0** | 更大容量带来持续增益，说明多 head 学习需要足够参数空间 |
| 仅 xLAM 数据 | 85.3 | 公共数据即可达到良好性能 |
| + 合成数据增强 | **86.3** | 数据平衡（尤其是 arg5/arg6）显著提升泛化能力 |

---

## 4. 关键结论和发现

### 主要发现
1. **函数调用输出具有高度结构性与弱依赖性**，应专门设计加速机制，而非沿用通用文本生成策略；
2. **special tokens 可同时实现 token 压缩与模式切换**，是连接压缩与并行的关键桥梁；
3. **并行解码在内存带宽受限的现代 GPU 上近乎零开销**，通过 batch 内多 head 利用 idle compute，效率高达 93%；
4. **SimpleTool 在保持甚至提升准确率的同时，实现 3–6× 加速**，结合量化可达 **sub-100ms 延迟**，使 LLM 控制实时系统成为可能；
5. **与 speculative decoding 完全正交**，联合使用理论加速可达 **14.6×**（见 Appendix J）。

### 方法的局限性
| 局限 | 说明 | 应对策略 |
|------|------|----------|
| **参数独立性假设** | 若参数间存在强语义依赖（如 `end_line` 依赖 `file_path`），并行生成可能导致不一致 | 合并相关参数为一个 head，或通过 multi-turn refinement 修正 |
| **固定 head 数量（6）** | 超过 6 个参数的 API 需顺序处理剩余参数 | 将溢出参数拼接到最后一个 head，牺牲部分并行性保功能完整 |
| **训练开销较高** | 每个 head 视为独立序列，训练 batch 包含 8 个分支，计算成本上升 | 接受一次性训练代价，适合模型预训练阶段 |

### 未来工作方向
- **开源发布**：计划公开训练代码与模型权重（已在 GitHub、Hugging Face、ModelScope 提供）；
- **视觉语言扩展**（Vision-Language Extension）：将 SimpleTool 扩展至 VLM，构建实时 VL-Agent 系统；
- **端侧部署优化**：适配 LiteRT、Core ML、ExecuTorch 等移动端框架；
- **自适应 head 数量**（Adaptive Heads）：根据 tool schema 动态调整 active head 数，提高灵活性；
- **大模型扩展**（Scaling）：探索在 30B+ 模型上的表现，验证增益是否随规模持续。

---

> 💡 **一句话总结**：  
> *SimpleTool* 通过引入 dual-role special tokens 实现 **冗余压缩 + 并行解码** 的协同优化，在不损失精度的前提下达成 **3–6× 端到端加速**，首次让消费级 GPU 上运行的 LLM 实现 **<100ms 函数调用延迟**，为具身智能、游戏 AI 等实时系统铺平道路。

</details>

---

### 15. [GRIP: Geometric Refinement and Adaptive Information Potential for Data Efficiency](https://arxiv.org/abs/2603.00031)

**Authors**: Changhao Wang, Jiaolong Yang, Xinhao Yao, Yunfei Yu, Peng Jiao, Lu Yu, Junpeng Fang, Riccardo Cantoro, Qing Cui, Jun Zhou  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.00031v1  

#### Abstract
The performance of Large Language Models (LLMs) is increasingly governed by data efficiency rather than raw scaling volume. However, existing selection methods often decouple global distribution balancing from local instance selection, compromising the hierarchical integrity of the training set. We ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GRIP: Geometric Refinement and Adaptive Information Potential for Data Efficiency

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前 Large Language Models（LLMs）的性能瓶颈已从**单纯的数据量扩展**转向**数据效率**（data efficiency）。尽管已有大量研究关注数据选择，但大多数方法存在以下缺陷：

- **割裂全局分布与局部实例选择**：现有方法通常将语义簇间的资源分配（inter-cluster budgeting）与簇内样本筛选（intra-cluster selection）分开处理，破坏了训练数据的层次完整性。
- **忽略长序列的几何坍缩问题**：Transformer 模型中，长上下文序列的嵌入（embedding）容易发生 **Length-Induced Embedding Collapse**，导致高价值的长尾逻辑序列被误判为“冗余”而丢弃。
- **静态质量评分无法反映动态学习需求**：基于固定规则（如难度、语法正确性）的质量打分不能适应模型在训练过程中不断变化的信息获取状态。

这些问题在代码生成等对结构性依赖强的任务中尤为致命。

---

### 🚀 提出的新方法与新思路

作者提出 **GRIP**（Geometric Refinement and Adaptive Information Potential），一个统一的、基于几何空间建模的数据选择框架，其核心思想是：

> 将语料库视为一个**信息密集的几何空间**，通过层级优化实现宏观资源调配与微观样本选择的协同。

#### 主要创新点包括：

1. **统一的层级选择框架（Unified Hierarchical Framework）**
   - 同时优化跨簇（inter-cluster）资源分配与簇内（intra-cluster）样本选择，保持数据集的结构一致性。

2. **自适应信息潜力探测机制：RAP（Rapid Adaptation Probe）**
   - 引入一种轻量级探针机制，冻结底层网络、重置顶层参数，在各语义簇上进行短步训练，测量损失下降幅度（Adaptation Delta △Lk）。
   - 利用该信号识别“表示缺陷区域”（representation deficit），动态增加这些区域的采样权重（replay multiplier），从而实现**闭环反馈式资源再分配**。

3. **长度校正的几何优先采样（Length-Rectified Geometric Prior）**
   - 针对长序列嵌入坍缩问题，设计了一个带长度补偿项的采样策略：
     $$
     P_{\text{select}}(x) \propto \frac{1}{p(x)} \cdot e^{\beta \cdot l(x)}
     $$
     其中 $ p(x) $ 是局部密度估计，$ l(x) $ 是序列长度，$ \beta $ 为恢复系数。
   - 显式“重新展开”坍缩的嵌入锥体，保留稀有的长尾逻辑模式。

4. **基于训练动态的数据可学性建模**
   - 建立了**瞬时信息势能**（Instantaneous Information Potential）理论联系，证明损失下降率可作为数据 learnability 的有效代理指标。
   - 发现静态质量评分 $ Q_k $ 与动态 learnability $ \Delta L_k $ 几乎无关（Pearson ≈ -0.202），说明必须结合动态反馈才能捕捉边际信息增益。

---

### 🔍 相比现有方法的优势

| 维度 | 现有方法局限 | GRIP 改进 |
|------|---------------|----------|
| **选择粒度** | 要么只调簇比例，要么只筛个体 | 统一宏观预算 + 微观采样 |
| **反馈机制** | 多为静态启发式（如质量、多样性） | 动态闭环反馈（loss-driven） |
| **长序列处理** | 密度采样易误删长序列 | 显式长度校正防止压制 |
| **计算开销** | 高成本全模型评估 | 使用小型 proxy model 探测，<1% FLOPs 开销 |

---

## 2. 核心实验方法和设置

### 📚 数据集

- 构造了一个 **100B token 的混合候选池** `D_pool`：
  - **背景数据**（Fixed Background）：CommonCrawl（不参与筛选）
  - **前景可选数据**（Selectable Foreground）：The Stack v2（用于 GRIP 筛选）
- 所有文档使用 **Qwen3 embedding model** 映射到统一向量空间。
- 最终训练数据 = GRIP 精选子集 + 固定 CommonCrawl。

---

### ⚙️ 实验设置

#### 模型架构
- 使用 **Mixture-of-Experts (MoE)** Transformer：
  - **8B 总参 / 1.4B 激活参数**（32 experts）
  - **16B 总参 / 1.4B 激活参数**（64 experts）
- 所有模型从零开始训练，仅数据不同。

#### 语义聚类
- 使用 **spherical k-means** 对嵌入进行聚类（K 个簇）。
- 定义每个簇的 **Geometric Consistency** $ o_k $ 衡量簇内紧凑性。

#### 探针构建（Probe Set）
- 使用 **Neyman 最优分配** 构建探针集 $ P $（约 0.5% 数据）：
  $$
  n_k^{\text{probe}} \propto N_k \cdot o_k
  $$
  优先覆盖规模大且分散的簇，提升估计鲁棒性。

#### 质量评分
- 使用 **LLM-as-a-Judge** 范式，由 Qwen3-235B 对探针样本打分，综合四个维度：
  - Code Quality
  - Algorithmic Design
  - Training Suitability
  - Knowledge Density

#### 动态 Learnability 测量（RAP）
- 在 proxy model 上执行 **Partial Reset Protocol**：
  - 冻结主干，重置最后 N 层 + LM head
  - 在每个簇上训练 10 步，记录初始与收敛损失
  - 计算相对损失下降：$ \delta L_k = (L_{\text{init}} - L_{\text{final}})/L_{\text{init}} $

---

### 📊 评估指标与基准任务

| 类别 | 基准名称 | 描述 |
|------|---------|------|
| **代码生成** | HumanEval+, MBPP+ | Pass@1 指标，测试函数补全能力 |
| **推理与鲁棒性** | LiveCodeBench (LCB), CruxEval | 测试时间泛化、执行预测能力 |
| **多语言能力** | MultiPL-E | 多编程语言细粒度分析 |

---

### 🔁 基线方法对比（Ablation Path）

| 方法 | 特点 |
|------|------|
| **Random Sampling** | 均匀采样，下界 |
| **Static Quality Budgeting** | 仅用 $ Q_k $ 分配基础预算 |
| **Static + Quality-Based Replay** | 用 $ Q_k $ 控制 replay 倍数 |
| **Static + Loss-Based Replay** | 用 $ \Delta L_k $ 控制 replay，但簇内随机采样 |
| **+ Diversity (no length fix)** | 加入核密度多样性采样，但无长度校正 |
| **GRIP (Full)** | 完整框架：动态 replay + 长度校正采样 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table 1）

#### Panel A: 300B tokens 规模表现（8B & 16B MoE）

| Model | Avg. Score | HumanEval Pass@1 | MBPP Pass@1 | LiveCodeBench | MultiPL-E |
|-------|------------|-------------------|-------------|----------------|-----------|
| Random 8B | 20.2 | 38.5 | 27.4 | 1.2 | 13.5 |
| **GRIP 8B** | **24.8** (+4.6%) | **52.4** | **36.0** | **5.3** | **23.7** |
| Random 16B | 20.8 | 40.8 | 29.1 | 1.5 | 14.8 |
| **GRIP 16B** | **25.6** (+4.8%) | **55.4** | **38.5** | **6.4** | **25.5** |

> ✅ GRIP 在更大模型上收益更显著，表明其能更好匹配高容量模型的信息需求。

---

### 🔍 消融实验结果（GRIP-8B, 100B tokens）

| 方法 | Avg. Score | MultiPL-E |
|------|------------|----------|
| 1. Random | 18.7 | 11.3 |
| 2. +Static Budget | 19.4 | 13.1 |
| 3. +Static Replay | 20.2 | 14.2 |
| 4. +Loss Replay | 21.2 | 16.5 |
| 5. +Diversity (no length fix) | 21.3 | 16.0 ↓ |
| 6. **GRIP (Full)** | **22.0** | **19.2** ↑ |

#### 关键发现：

- **静态质量过滤只能带来有限提升**（+1.5%）
- **引入动态 loss 反馈带来显著跃升**（+1.0%）
- **仅加多样性反而损害性能**（MultiPL-E 从 16.5→16.0）→ 存在“**多样性陷阱**”（diversity trap）
- **加入长度校正后全面反弹**（+0.7%，MultiPL-E 回升至 19.2）→ 验证了 **Length Rectification 的必要性**

---

### 📉 性能超越更大未筛选数据集

> GRIP 训练的模型在 **100B token 精选数据**上的表现，**超过在 300B token 未筛选数据**上训练的模型。

这直接验证了论文主张：**数据效率 > 数据体积**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **静态质量 ≠ 动态可学性**
   - 高质量数据不一定难学，低质量数据也可能蕴含高 learnability。
   - 必须通过 **训练动态**（如 loss 下降速度）来识别真正的“信息富矿”。

2. **Transformer 存在严重长度诱导嵌入坍缩**
   - 长序列嵌入聚集于高密度窄锥区，造成“伪冗余”错觉。
   - 若不显式纠正，标准多样性采样会系统性抛弃长上下文代码。

3. **proxy model 可可靠指导大模型数据选择**
   - 即使使用极小模型（如 SmolLM-135M）或仅重训头部（N=0），也能获得一致的 cluster utility 排名。
   - 支持低成本部署 RAP 探针。

4. **层级联合优化至关重要**
   - 单独优化宏观或微观都无法达到最优；只有 GRIP 这种**几何一致性 + 自适应信息势能 + 长度校正**三位一体的方法才能最大化 epistemic gain。

---

### ⚠️ 方法的局限性

1. **依赖高质量嵌入空间**
   - 若初始 embedding model（如 Qwen3）本身有偏见或表达能力不足，会影响聚类与探测效果。

2. **额外计算开销虽小但仍存在**
   - RAP 探针需运行多个 mini-trainings，虽然总开销 <1% FLOPs，但在极端资源受限场景仍可能成为瓶颈。

3. **超参数敏感性**
   - 如温度 $ T $、强度 $ a $、长度惩罚 $ \beta $ 等需仔细调优，尤其在不同领域迁移时。

4. **目前验证集中在代码领域**
   - 虽然代码是最具挑战性的测试床，但文本、多模态等领域的普适性有待进一步验证。

---

### 🔮 未来工作方向

1. **扩展至多模态与跨模态数据选择**
   - 将几何空间建模推广到图文、音视频等异构数据。

2. **在线自适应数据流调度**
   - 在持续预训练中实时更新采样策略，形成真正的“活数据管道”。

3. **减少对 LLM-as-a-Judge 的依赖**
   - 探索完全无监督或弱监督方式估计静态质量 $ Q_k $。

4. **理论深化：建立更严格的 usable information bound**
   - 结合信息论与优化理论，给出 GRIP 的收敛性与近似比保证。

---

## 总结

> **GRIP 成功地将数据选择从“经验驱动”推向“几何+动态反馈驱动”的新范式**。它不仅提升了数据效率，更重要的是揭示了：  
> 
> 🔑 **真正高效的数据不是最多的，而是最契合模型当前认知缺口的那一部分。**

该工作为大规模语言模型的可持续训练提供了一条**可扩展、可解释、高性价比**的技术路径。

</details>

---

### 16. [SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs](https://arxiv.org/abs/2603.00357)

**Authors**: Jin Lee, Zhonghao Chen, Xuhang He, Robert Underwood, Bogdan Nicolae, Franck Cappello, Xiaoyi Lu, Sheng Di, Zheng Zhang  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.00357v1  

#### Abstract
In large-scale LLM pre-training systems with 100k+ GPUs, failures become the norm rather than the exception, and restart costs can dominate wall-clock training time. However, existing fault-tolerance mechanisms are largely unprepared for this restart-dominant regime. To address this challenge, we pr...

---

### 17. [Training Dynamics of Softmax Self-Attention: Fast Global Convergence via Preconditioning](https://arxiv.org/abs/2603.01514)

**Authors**: Gautam Goel, Mahdi Soltanolkotabi, Peter Bartlett  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.01514v1  

#### Abstract
We study the training dynamics of gradient descent in a softmax self-attention layer trained to perform linear regression and show that a simple first-order optimization algorithm can converge to the globally optimal self-attention parameters at a geometric rate. Our analysis proceeds in two steps. ...

---

### 18. [Expanding LLM Agent Boundaries with Strategy-Guided Exploration](https://arxiv.org/abs/2603.02045)

**Authors**: Andrew Szot, Michael Kirchhof, Omar Attia, Alexander Toshev  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.02045v1  

#### Abstract
Reinforcement learning (RL) has demonstrated notable success in post-training large language models (LLMs) as agents for tasks such as computer use, tool calling, and coding. However, exploration remains a central challenge in RL for LLM agents, especially as they operate in language-action spaces w...

---

### 19. [DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](https://arxiv.org/abs/2603.01106)

**Authors**: Haowen Gao, Zhenyu Zhang, Liang Pang, Fangda Guo, Hongjian Dou, Guannan Lv, Shaoguo Liu, Tingting Gao, Huawei Shen, Xueqi Cheng  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01106v1  

#### Abstract
Reinforcement learning (RL) with group relative policy optimization (GRPO) has become a widely adopted approach for enhancing the reasoning capabilities of multimodal large language models (MLLMs). While GRPO enables long-chain reasoning without a critic, it often suffers from sparse rewards on diff...

---

### 20. [Harmonizing Dense and Sparse Signals in Multi-turn RL: Dual-Horizon Credit Assignment for Industrial Sales Agents](https://arxiv.org/abs/2603.01481)

**Authors**: Haojin Yang, Ai Jian, Xinyue Huang, Yiwei Wang, Weipeng Zhang, Ke Zeng, Xunliang Cai, Jingqing Ruan  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01481v1  

#### Abstract
Optimizing large language models for industrial sales requires balancing long-term commercial objectives (e.g., conversion rate) with immediate linguistic constraints such as fluency and compliance. Conventional reinforcement learning often merges these heterogeneous goals into a single reward, caus...

---

### 21. [Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents](https://arxiv.org/abs/2603.01548)

**Authors**: Neeraj Bholani  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01548v1  

#### Abstract
Tool-using LLM agents face a reliability-cost tradeoff: routing every decision through the LLM improves correctness but incurs high latency and inference cost, while pre-coded workflow graphs reduce cost but become brittle under unanticipated compound tool failures. We present Self-Healing Router, a...

---

### 22. [CARD: Towards Conditional Design of Multi-agent Topological Structures](https://arxiv.org/abs/2603.01089)

**Authors**: Tongtong Wu, Yanming Li, Ziye Tang, Chen Jiang, Linhao Luo, Guilin Qi, Shirui Pan, Gholamreza Haffari  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01089v1  

#### Abstract
Large language model (LLM)-based multi-agent systems have shown strong capabilities in tasks such as code generation and collaborative reasoning. However, the effectiveness and robustness of these systems critically depend on their communication topology, which is often fixed or statically learned, ...

---

### 23. [LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval](https://arxiv.org/abs/2603.01425)

**Authors**: Jiajie Jin, Yanzhao Zhang, Mingxin Li, Dingkun Long, Pengjun Xie, Yutao Zhu, Zhicheng Dou  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01425v1  

#### Abstract
LLMs have fundamentally transformed dense retrieval, upgrading backbones from discriminative encoders to generative architectures. However, a critical disconnect remains: while LLMs possess strong reasoning capabilities, current retrievers predominantly utilize them as static encoders, leaving their...

---

### 24. [AeroDaaS: A Programmable Drones-as-a-Service Platform for Intelligent Aerial Systems](https://arxiv.org/abs/2603.00506)

**Authors**: Kautuk Astu, Suman Raj, Priyanshu Pansari, Yogesh Simmhan  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.00506v1  

#### Abstract
The increasing adoption of UAVs equipped with advanced sensors and GPU-accelerated edge computing has enabled real-time AI-driven applications in domains such as precision agriculture, wildfire monitoring, and environmental conservation. However, the integrated design and orchestration of navigation...

---

### 25. [Subliminal Signals in Preference Labels](https://arxiv.org/abs/2603.01204)

**Authors**: Isotta Magistrali, Fr\'ed\'eric Berdoz, Sam Dauncey, Roger Wattenhofer  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01204v1  

#### Abstract
As AI systems approach superhuman capabilities, scalable oversight increasingly relies on LLM-as-a-judge frameworks where models evaluate and guide each other's training. A core assumption is that binary preference labels provide only semantic supervision about response quality. We challenge this as...

---

### 26. [DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](https://arxiv.org/abs/2603.01762)

**Authors**: Yingjie Tan, Quanming Yao, Yaqing Wang  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01762v1  

#### Abstract
Spatiotemporal partial differential equations (PDEs) underpin a wide range of scientific and engineering applications. Neural PDE solvers offer a promising alternative to classical numerical methods. However, existing approaches typically require large numbers of training trajectories, while high-fi...

---

### 27. [BAED: a New Paradigm for Few-shot Graph Learning with Explanation in the Loop](https://arxiv.org/abs/2603.01941)

**Authors**: Chao Chen, Xujia Li, Dongsheng Hong, Shanshan Lin, Xiangwen Liao, Chuanyi Liu, Lei Chen  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.01941v1  

#### Abstract
The challenges of training and inference in few-shot environments persist in the area of graph representation learning. The quality and quantity of labels are often insufficient due to the extensive expert knowledge required to annotate graph data. In this context, Few-Shot Graph Learning (FSGL) app...

---

### 28. [SWE-Hub: A Unified Production System for Scalable, Executable Software Engineering Tasks](https://arxiv.org/abs/2603.00575)

**Authors**: Yucheng Zeng, Shupeng Li, Daxiang Dong, Ruijie Xu, Zimo Chen, Liwei Zheng, Yuxuan Li, Zhe Zhou, Haotian Zhao, Lun Tian, Heng Xiao, Tianshu Zhu, Longkun Hao, Jianmin Wu  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.00575v1  

#### Abstract
Progress in software-engineering agents is increasingly constrained by the scarcity of executable, scalable, and realistic data for training and evaluation. This scarcity stems from three fundamental challenges in existing pipelines: environments are brittle and difficult to reproduce across languag...

---

### 29. [S5-HES Agent: Society 5.0-driven Agentic Framework to Democratize Smart Home Environment Simulation](https://arxiv.org/abs/2603.01554)

**Authors**: Akila Siriweera, Janani Rangila, Keitaro Naruse, Incheon Paik, Isuru Jayanada  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.01554v1  

#### Abstract
The smart home is a key domain within the Society 5.0 vision for a human-centered society. Smart home technologies rapidly evolve, and research should diversify while remaining aligned with Society 5.0 objectives. Democratizing smart home research would engage a broader community of innovators beyon...

---

### 30. [ToolRLA: Fine-Grained Reward Decomposition for Tool-Integrated Reinforcement Learning Alignment in Domain-Specific Agents](https://arxiv.org/abs/2603.01620)

**Authors**: Pengbo Liu  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.01620v1  

#### Abstract
Tool-integrated reasoning agents interleaving natural language deliberation with external API calls show promise for complex multi-step tasks. However, aligning such agents for high-stakes domain-specific deployment is challenging, as existing reinforcement learning uses coarse binary rewards (succe...

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

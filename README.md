# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-03 08:45:43 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [HCMS: Head-Chunked Multi-Stream Pipeline for Communication-Computation Overlap in Long-Sequence Parallel Attention](https://arxiv.org/abs/2607.01817)

**Authors**: Chao Yuan, Pan Li, Yingnan Sun, Jing Liu  
**Category**: cs.DC  
**Published**: 2026-07-03  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2607.01817v1  

#### Abstract
All-to-all based sequence parallelism methods execute communication and computation strictly in serial when processing medium-long sequences, resulting in hardware resource underutilization. This paper proposes Head-Chunked Multi-Stream Pipeline (HCMS), which exploits the computational independence ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HCMS: Head-Chunked Multi-Stream Pipeline for Communication-Computation Overlap in Long-Sequence Parallel Attention

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在中长序列（如视频生成中的 31K–56K tokens）场景下，基于 **all-to-all** 的 **Sequence Parallelism** 方法（如 DeepSpeed Ulysses）存在严重的效率瓶颈：**通信与计算严格串行执行**，导致 GPU 等硬件资源利用率低下。当通信占比 $ p = T_{\text{comm}} / T_{\text{total}} $ 达到 15%–40% 时，这一问题尤为显著。

### 提出了什么新方法或新思路
本文提出 **Head-Chunked Multi-Stream Pipeline (HCMS)**，其核心思想是利用 **Multi-Head Attention** 中各注意力头之间**计算独立性**（Computational Independence），将注意力头划分为多个 chunk，并通过 **双 CUDA Stream 架构**实现细粒度的 **communication-computation overlap**。

- **Head Chunking**：将 H 个 attention heads 划分为 C 个 chunk，每个 chunk 可独立进行通信与计算。
- **Dual-Stream Pipeline**：
  - `Scomm` 流负责 all-to-all 通信；
  - `Scomp` 流负责 attention 计算；
  - 利用 **CUDA Event** 同步机制确保依赖正确。
- 支持 **不均匀划分**（uneven partitioning），且性能波动小于 1%。
- 完全兼容 FlashAttention、SDPA 等优化 kernel，无需修改底层实现。

### 相比现有方法的优势
| 方法 | 通信模式 | 通信轮数 | 是否支持 overlap | 兼容性 |
|------|----------|-----------|------------------|--------|
| **Ring Attention** | P2P Ring | $ P-1 $ 轮 | 块级 overlap，受限于块大小 | 需适配 causal mask |
| **Ulysses (Baseline)** | All-to-All | 2 轮 | ❌ 无 overlap（原生串行） | ✅ 支持 FlashAttention |
| **HCMS (本工作)** | All-to-All | 2 轮 | ✅ 头级别（head-level）fine-grained overlap | ✅ 原生兼容 FlashAttention/SDPA |

- **更细粒度的 overlap**：从 Ring Attention 的 block-level 提升到 head-level。
- **更低通信延迟**：相比 Ring 的 $ P-1 $ 轮通信，HCMS 仅需 2 轮 all-to-all。
- **即插即用**：无需修改 attention kernel，可直接集成进现有训练框架（如 PyTorch autograd）。

---

## 2. 核心实验方法和设置

### 使用的数据集
未使用传统 NLP/Vision 数据集，而是聚焦于 **视频生成模型中的典型序列长度**，模拟真实应用场景：
- 序列长度覆盖：**56K、131K、206K、281K tokens**
- 对应视频帧数：9/21/33/45 帧，分辨率 60×104（latent space）

### 实验设置
#### 硬件平台（跨平台验证）
| 平台 | GPU | 数量 | 互联方式 | Attention 实现 |
|------|-----|-------|------------|----------------|
| A | 4×L20 | 46GB | PCIe 4.0 | FlashAttention-2 |
| B | 4×RTX 4090 | 24GB | PCIe 4.0 | SDPA |
| C | 4×A10 | 22GB | PCIe 4.0 | SDPA |
| D | 8×RTX 5090 | 32GB | PCIe 5.0 | SDPA |

#### 模型配置（参考 Wan2.2 视频生成模型）
- Hidden dim: 5120
- Attention heads H: 40 或 24
- Head dim: 128
- Sequence length: 最高达 281K

### 评估指标
- **端到端延迟（end-to-end latency）**
- **吞吐量（throughput）**
- **加速比（speedup ratio）**
- **数值一致性**：最大误差（max diff）、平均误差（mean diff）
- **峰值显存占用**

### 基线方法对比
- **DeepSpeed Ulysses**：主流 all-to-all 序列并行方案，作为主要 baseline。
- **Ring Attention**：代表 ring-based 分布式 attention 方法。
- 所有方法均在同一软硬件环境下测试（PyTorch 2.9.1, CUDA 12.8, NCCL, BF16 精度）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在不同平台上的加速效果（典型 131K tokens, C=4）
| 平台 | Ulysses 延迟 | HCMS 延迟 | 加速比 |
|------|-------------|-----------|--------|
| L20+FA (4-GPU) | 996.2 ms | 945.3 ms | **1.054×** |
| 4090+SDPA (4-GPU) | 725.9 ms | 677.8 ms | **1.07×** |
| A10+SDPA (4-GPU) | 1643.2 ms | 1458.8 ms | **1.13×** |
| 5090+SDPA (8-GPU) | 288.3 ms | 276.1 ms | **1.044×** |

> A10 平台因通信占比更高（$ p \approx 39.6\% $），获得最高达 **13.9%** 的加速。

#### 不同序列长度下的表现（4-GPU, 56K tokens）
| 方法 | L20+FA | 4090+SDPA | 5090 (8-GPU) |
|------|--------|-----------|-------------|
| Ulysses | 234.5 ms | 182.8 ms | 73.3 ms |
| HCMS | 212.6 ms | 155.6 ms | 63.0 ms |
| **Speedup** | **1.10×** | **1.18×** | **1.16×** |

👉 结论：**序列越短，通信占比越高，HCMS 加速越明显**。

#### 三方法对比（L20+FA, 4-GPU）
| Seq Length | Ulysses | Ring Attention | HCMS | 最优者 |
|------------|--------|----------------|--------|--------|
| 31K | 95.5 ms | 95.4 ms | **83.3 ms** | **HCMS (+14.5%)** |
| 56K | 234.7 ms | 223.3 ms | **212.7 ms** | **HCMS (+5.0%)** |
| 131K | 996.9 ms | **929.4 ms** | 945.4 ms | **Ring (+1.7%)** |

📌 发现：在 **31K–56K** 典型视频生成长度上，**HCMS 显著优于 Ring Attention 和 Ulysses**；但在超长序列（>131K）时，Ring 更优。

### 消融实验结果

#### 不同 chunk 数 $ C $ 的影响（4-GPU, 131K tokens）
| Chunks $ C $ | 延迟 (ms) | Speedup | 相对最优 |
|---------------|-----------|---------|----------|
| 1 (Ulysses) | 996.2 | 1.000× | -5.2% |
| 2 | 951.5 | 1.047× | -0.7% |
| 4 | 945.3 | 1.054× | -0.1% |
| **5** | **944.5** | **1.055×** | **Best** |
| 8 | 963.8 | 1.034× | -2.0% |

✅ 最优 chunk 数 $ C^* \approx \sqrt{T_{\text{comm}} / \beta} $，过多会导致 event 同步开销上升。

#### 不均匀划分鲁棒性（H=40, C=3/4/6）
| C | 类型 | Chunk sizes | 负载不平衡度 | Speedup |
|----|------|--------------|----------------|---------|
| 3 | Uneven | [4,3,3] | 30% | 1.056× |
| 4 | Uneven | [3,3,2,2] | 40% | 1.055× |
| 6 | Uneven | [2,2,2,2,1,1] | 60% | 1.052× |

💡 即使严重不均衡（60%），性能损失也小于 0.4%，说明 **pipeline 掩盖了负载差异**。

#### 正确性与训练支持
- 输出与 baseline **完全一致**（max error = 0）
- 支持 PyTorch autograd，前向+反向联合运行加速 **3.5%**
- 峰值显存降低 **8.7%**

---

## 4. 关键结论和发现

### 主要发现
1. **通信-计算重叠空间巨大**：在中长序列（31K–56K）场景下，通信占比 $ p $ 可达 20%–40%，理论加速上限为 $ 1/(1-p) $，HCMS 成功挖掘该潜力。
2. **HCMS 显著提升性能**：
   - 相比 Ulysses：**10%–17.5%** 加速
   - 相比 Ring Attention：**5%–14.5%** 加速（在 31K–56K 区间）
   - 端到端在 Wan2.2 模型上实现 **6.8%** 整体加速
3. **适用性明确**：当 $ p > 20\% $ 时推荐使用 HCMS；若 $ p < 10\% $（如极长序列），收益有限。
4. **正交兼容性强**：可无缝集成 FlashAttention、SDPA、USP、LoongTrain 等系统。

### 方法的局限性
- 在 **compute-bound 场景**（如 281K 超长序列，$ p < 10\% $）下加速效果弱（仅约 3%）。
- 当 chunk 数 $ C $ 过大时，CUDA Event 同步开销累积，反而降低性能。
- 引入多 stream 管理复杂度，对开发者有一定工程门槛。

### 未来工作方向
- 扩展至 **16-GPU 及以上** 和 **跨节点分布式训练** 场景。
- 将 HCMS 应用于 **反向传播阶段**，实现梯度通信与计算的 overlap。
- 设计 **自适应调度器**，根据 runtime profiling 动态选择最优 chunk 数 $ C $。

--- 

> ✅ 总结一句话：  
> **HCMS 通过 head-level chunking + dual-stream pipeline，在不改变计算语义的前提下，实现了高效、兼容、可扩展的 communication-computation overlap，是中长序列分布式 attention 的实用化重要进展。**

</details>

---

### 2. [OmniPilot: An Uncertainty-Aware LLM Inference Advisor for Heterogeneous GPU Clusters](https://arxiv.org/abs/2607.01579)

**Authors**: D. Balamurugan, Thomas W. Bush  
**Category**: cs.DC  
**Published**: 2026-07-03  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.01579v1  

#### Abstract
Serving large language models (LLMs) on a shared, heterogeneous GPU cluster requires users and operators to select the GPU type, tensor-parallel degree, and precision before committing valuable node-hours. Making these choices is challenging because effective throughput, launch-success rates, and cl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# OmniPilot: An Uncertainty-Aware LLM Inference Advisor for Heterogeneous GPU Clusters  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在共享、异构的 GPU 集群上部署大语言模型（LLM）推理任务时，用户需提前选择 **GPU 类型**、**tensor-parallel (TP) degree** 和 **数值精度（precision）**。这些决策直接影响成本、吞吐量和成功率，但由于以下复杂因素而极具挑战：
- **硬件-软件交互复杂**：例如，4-bit 量化可能比 16-bit 更慢；
- **KV Cache 压力导致非单调行为**：高精度可能导致内存饱和，反而降低性能；
- **多GPU启动失败率随 TP 度上升显著增加**（如 H100×4 成功率仅 38%）；
- **集群资源受硬件故障等不可预测事件影响**。

传统调度器（如 Slurm）缺乏对工作负载历史行为的认知，无法提供预提交建议。

### 🚀 提出的新方法：OmniPilot
OmniPilot 是一个**不确定性感知的 LLM 推理配置推荐系统**，具备以下三大核心组件：

1. **Conformally Calibrated Quantile Cost Model**
   - 使用梯度提升回归树（gradient-boosted regressor）预测多个目标（吞吐量、延迟、功耗、KV 缓存占用等），输出带校准置信区间的分位数预测。
   - 在 log 空间建模以处理数量级差异大的目标。

2. **Out-of-Distribution (OOD) Abstention Layer**
   - 当请求超出训练数据覆盖范围（如新模型家族、超长上下文）时，主动“拒绝”推荐（abstain），避免高风险错误。
   - 判断依据是“支持距离”（support distance），而非仅依赖区间宽度。

3. **Decision-Coupled Utility Function**
   - 基于经济效用（economic utility）进行排序：`utility = value - cost - risk`
     - 包括节点小时成本、排队时间、预期失败重启成本、SLO 违约概率、KV 缓存饱和惩罚项。
   - 效用函数通过操作员的实际偏好（revealed preference）进行校准。

此外还引入了：
- **Regression-Gated Update Loop**：确保模型更新不会导致性能回退；
- **Feature-Before-Data Methodology**：新特征（如新量化格式）应先加入特征工程再收集数据；
- **Live Telemetry Integration**：融合 DCGM 实时指标，修正因采样频率低导致的测量偏差（measurement-stale problem）。

### 🔍 相比现有方法的优势
| 维度 | 现有方法局限 | OmniPilot 改进 |
|------|---------------|----------------|
| **决策粒度** | 固定规则或静态策略（如“一律用 H100/FP8”） | 动态、细粒度优化 GPU + TP + precision 组合 |
| **不确定性建模** | 多数系统无显式不确定性估计 | 提供 conformal prediction 区间 + 显式 abstention |
| **失败预防** | 忽视启动失败先验 | 引入 per-configuration launch-success prior（来自 50万 job 数据） |
| **KV Cache 影响** | 忽略缓存压力对 TTFT 的影响 | 显式建模并加入 utility 中的 penalty 项 |
| **可扩展性设计** | 被动学习 | 主动通过 benchmark gate 决定是否值得探测 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **Benchmark Dataset**：共 **460 条标注行**（最终扩展至 476），涵盖：
  - **6 个模型家族**：qwen2, llama3, deepseek distill, mistral, phi, gemma
  - **3 种 GPU**：A100-40GB, H100-80GB, H200-141GB
  - **4 种精度**：bf16, FP8, AWQ, GPTQ
  - **TP Degree**：1, 2, 4, 8
  - **变量参数**：context length（最高 32k）、concurrency（最高 2000）
- **Cluster-Wide Harvest**：从 Slurm 日志中提取 **500,000 个作业元数据**，用于构建 demand 分布和 launch-success prior。
- 所有记录均包含 schema version、content hash、software-stack fingerprint，保证可复现性。

### ⚙️ 实验设置
- **评估方式**：采用 **leakage-free GroupKFold**（5 折），按 workload cell（base model × context × concurrency）分组，确保测试集中的完整 workload 未出现在训练集中。
- **软件环境固定**：vLLM 0.11.2，runtime fingerprint 跟踪版本变化。
- **OОD 测试集**：包含不在训练包络内的 5 个单元（如 OLMo-13B 新家族、Qwen-7B 在 16k/32k 上下文等）。

### 📏 评估指标
| 指标 | 定义 |
|------|------|
| **MAPE** | Mean Absolute Percentage Error，衡量预测误差 |
| **R²(log)** | Log-space R²，适用于跨数量级的目标 |
| **Top-1 Accuracy** | 推荐最优配置的准确率 |
| **Utility Regret** | 推荐配置与真实最优之间的效用差距，越小越好 |
| **Coverage** | 预测区间覆盖真实值的比例（目标为 80%） |
| **Abstention Rate** | 对低置信请求的拒绝比例 |

### 🆚 基线方法对比
| Baseline | 描述 |
|--------|------|
| **Cheapest-fit** | 选择最便宜且可行的配置 |
| **Size-threshold TP rule** | 小模型单卡，大模型多卡 |
| **H100/FP8 recipe** | 默认使用最新硬件和先进精度 |
| **Throughput-max (newest GPU)** | 总是选最强 GPU |
| **Nearest-neighbor cell** | 查找最近的历史配置 |
| **OmniPilot (bf16-only)** | 消融版，不考虑其他精度 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 成本模型准确性（Table 1）
| Target | MAPE | R²(log) | 80% Coverage |
|-------|------|---------|-------------|
| **Aggregate tokens/s** | **6.2%** | **0.92** | 81% |
| Request throughput | ~7.6% | 0.93 | ~81% |
| TTFT (p50) | ~12% | 0.92 | 80% |
| KV-cache usage | 25.3% | ~0.83 | ~66% |
| Average power | ~15% | ~0.46 | — |

> ✔️ 主要决策驱动目标（吞吐、延迟）误差极低，R² 达 0.92–0.93。

#### ✅ 推荐系统性能（Table 2）
| Policy | Top-1 Accuracy | Regret |
|-------|----------------|--------|
| **OmniPilot (advisor)** | **92%** | **0.006** |
| Cheapest-fit | 91% | 0.010 |
| Size-threshold TP rule | 80% | 1.970 |
| H100/FP8 recipe | 42% | 1.651 |
| Throughput-max | 3% | 4.728 |
| Nearest-neighbor | 70% | 0.722 |

> ✔️ OmniPilot 在保持 92% 准确率的同时，**平均效用遗憾（regret）仅为 0.006**，比次优基线低一个数量级以上。

#### ✅ OOD 行为分析（Table 3）
| OOD Cell | Prediction Error | Interval Covered? | Confidence |
|--------|------------------|--------------------|------------|
| OLMo-13B (new family) | 46% | ❌ | low |
| Qwen-7B ctx 16k | 31% | ❌ | low |
| ... | 24–46% | **0/5 覆盖** | all flagged as low |

> ❗ conformal 区间在 OOD 下完全失效（覆盖率为 0），但 **abstention layer 成功标记全部 5 个 OOD 请求为 low-confidence**。

#### ✅ 消融实验（Table B2）
| 移除组件 | 影响 |
|--------|------|
| **Quantization feature** | FP8 吞吐预测误差 ↑90%（3.88% → 7.36%） |
| **KV-saturation utility term** | 导致 Qwen2.5-32B/A100 错误选择 bf16（实际应降级） |
| **Per-config success prior** | 过度推荐 multi-GPU（忽略其仅 38% 成功率） |
| **Model-family encoding** | Gemma-2-27b 吞吐预测严重退化 |
| **Promotion gate** | 可能部署更高误差模型（当新增噪声数据时） |

> ✔️ 每个模块都针对特定失败模式设计，移除后均造成显著性能下降。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **静态规则在异构集群中表现极差**：
   - “一律用 H100/FP8” 的策略仅有 **42% top-1 准确率**；
   - “最大化吞吐” 导致过度配置，**regret 高达 4.7**。

2. **量化并非总是带来收益**：
   - AWQ 在某些 kernel 路径下比 bf16 **更慢**；
   - GPTQ 在 Gemma-2-27b 上直接失败；
   - 必须结合具体硬件路径建模。

3. **KV Cache 压力引发非线性权衡**：
   - Qwen2.5-32B 在 A100 上运行 bf16 会导致 KV 占用率达 1.0，频繁 preemption，TTFT 激增至 ~16s；
   - 引入 **KV-saturation penalty** 可纠正此问题。

4. **OOD 检测至关重要**：
   - 超出支持包络的预测误差高达 **24–46%**；
   - conformal 区间失效，必须依赖 **support-distance abstention**。

5. **scheduler metadata 不足以支撑全局吞吐预测**：
   - Slurm 日志只记录“GPU 被如何使用”，**不记录运行的是哪个模型、何种精度、并发多少**；
   - 50万 jobs 中仅 903 个可识别为 vLLM 类型 → **无法仅靠基础设施元数据构建 advisor**。

### ⚠️ 局限性
- **当前仅支持 single-node inference serving**，不支持 training 或 fault recovery；
- **未建模 vLLM 内部 runtime 参数**（如 block size、scheduler policy）；
- **DCGM 60秒采样频率限制了 activity 指标的分辨率**（SM/tensor activity MAPE 高达 96–100%）；
- **abstention 机制尚未闭环集成反馈学习**（虽计划未来将 OOD 场景纳入训练集）；
- **单集群验证**，绝对阈值（如 KV knee=0.85）具有部署特异性。

### 🔮 未来工作方向
1. **扩展至更多 ML workload**：
   - 支持 training、multi-node serving、fault recovery；
   - 复用现有 telemetry substrate，新增目标与标签。

2. **改进 benchmark gate**：
   - 当前 gate 仅缩小方差，**不更新 posterior mean**；
   - 需实现基于 refitting 的 posterior 更新机制以获得真正信息价值。

3. **纵向 drift study**：
   - 多轮 promotion cycle 下观察模型稳定性与数据漂移应对能力。

4. **集成 vLLM runtime 参数建模**：
   - 将内部引擎配置作为输入变量，进一步提升预测精度。

5. **跨集群迁移验证**：
   - 在不同硬件构成与负载分布的集群中重新校准阈值与效用函数。

---

> 💬 **总结一句话**：  
> **OmniPilot 将 LLM 推理资源配置转化为一个“带刹车”的经济决策过程——不仅推荐最优配置，还能识别未知风险并主动拒绝，从而在动态复杂的共享 GPU 集群中实现高效、安全、可持续的服务部署。**

</details>

---

### 3. [Spec-AUF: Accept-Until-Fail Training under Train-Inference Misalignment for Masked Block Drafters](https://arxiv.org/abs/2607.01893)

**Authors**: Tianjian Yang, Meng Li  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.01893v1  

#### Abstract
Speculative decoding accelerates autoregressive generation by drafting a block of tokens that the target model verifies left-to-right, committing only the longest accepted prefix. Block (DLM-style) drafters predict the whole block in parallel, which is fast but trained with a full-block cross-entrop...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Spec-AUF: Accept-Until-Fail Training under Train-Inference Misalignment for Masked Block Drafters

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题  
该论文针对 **speculative decoding** 中广泛存在的 **train-inference misalignment（训练-推理不一致）** 问题。具体来说：

- 在 **block-style drafters**（如 DFlash、Domino）中，模型在训练时使用 **full-block cross-entropy loss**，即对整个预测块的所有位置进行监督。
- 但在推理阶段，**verifier** 是从左到右逐个验证 token 的，一旦某个位置失败，后续所有 token 都会被丢弃。
- 因此，训练时对“失败前缀之后的位置”进行监督是无效甚至有害的，造成 **exposure bias** 和资源浪费。

> 这种不一致性导致模型虽然在 token 准确率上可能更高，但实际加速效果（即平均接受长度 $T$）反而更差。

---

### 🚀 提出的新方法：Accept-Until-Fail (AUF)

AUF 是一种 **简单而有效的损失函数修改策略**，其核心思想是：

> **只在第一个错误预测 token 及其之前的 token 上计算 cross-entropy 损失**，即保留“被接受的前缀 + 第一个失败 token”，其余全部 mask 掉。

#### 创新点：
- **动机来源于 teacher-forcing**：类比于 autoregressive 模型在训练时接收 gold prefix 输入，loss 天然具有 prefix 条件性；而 masked block drafter 无法从输入侧获得这种条件信号，因此 AUF 尝试在 **loss 层面模拟这一机制**。
- **硬截断（hard truncation）而非软加权**：不同于 D-PACE 或 GRIFFIN 使用动态权重或 Top-K mask，AUF 使用 **基于 greedy prediction 的第一个 exact mismatch 作为边界 $j^*$**，直接将 $i > j^*$ 的位置从 loss 支持集中移除。
- **零额外开销**：
  - 不引入辅助目标（no auxiliary objective）
  - 不需要 verifier rollouts
  - 不改变推理流程或 exactness contract
  - 不增加超参数

---

### 🔍 相比现有方法的优势

| 方法 | 是否改变架构 | 是否需 rollout | 是否含超参 | 是否软加权 | AUF 更优之处 |
|------|----------------|---------------|------------|-------------|----------------|
| DFlash (decay-CE) | ❌ | ❌ | ✅ ($\gamma_d$) | ❌ | 手动先验，固定衰减 |
| GRIFFIN | ❌ | ❌ | ✅ (Top-K) | ❌ | 使用 Top-K 判断前缀是否断裂，不够精确 |
| D-PACE | ❌ | ❌ | ✅ (dynamic weights) | ✅ | 软加权仍保留全支持集 |
| SpecDiff-2 | ❌ | ✅ | ✅ | ✅ | 引入 distillation 和采样路径，复杂度高 |
| **AUF (Ours)** | ❌ | ❌ | ❌ | ❌ | **最简干预，仅改 loss 支持集** |

> ✅ AUF 是目前最轻量、最直接逼近“prefix-aware supervision”的方法。

---

## 2. 核心实验方法和设置

### 📚 数据集与训练配置

- **训练数据**：ShareGPT 对话数据（未用 target model 重新生成 response）
- **目标模型（target model）**：Qwen3-8B
- **drafter 模型**：
  - **DFlash**：5-layer block drafter，block size $B=16$
  - **Domino**：相同 backbone + GRU-based two-branch head（base + final）
- **训练设置**：
  - 共训练 6 个 epoch
  - 使用 SGLang 后端进行 speculative decoding 测试
  - 单 GPU，batch size = 1
  - draft proposal 为 greedy（top-1）

---

### 📊 评估指标

| 指标 | 定义 | 说明 |
|------|------|------|
| $\mathbf{T}$ | 平均 acceptance length = $A + 1$ | $A$ 是连续被接受的 draft token 数，+1 是 verifier 发出的纠正 token；**主评价指标** |
| Conditional acceptance rate $\alpha_k$ | $\frac{\#\{A \geq k\}}{\#\{A \geq k-1\}}$ | 衡量每个位置在前缀已成功前提下的接受概率 |
| Common-token accuracy | 在完整 block 上计算的 token 准确率 | 固定分母，用于跨方法比较 |
| Support accuracy | 在 AUF 动态选择的支持集上的准确率 | 反映当前 first-failure depth 的学习状态 |
| Active token ratio $\tau_{\text{active}} = |S| / B$ | 每步参与 loss 更新的 token 比例 | AUF 特有动态维度 |

---

### 🧪 基线方法对比

| 方法 | 描述 |
|------|------|
| **Decay-only** | DFlash 默认设置：exponential decay weight ($\gamma_d = 7$)，full-block CE |
| **AUF-only** | 仅使用 AUF 截断支持集，无 decay |
| **AUF+decay** | AUF 支持集内再应用 exponential decay |
| **GRIFFIN-style baseline** | Top-K mask（文中指出其表现不如 decay） |
| **D-PACE** | 动态加权，非截断（本文未重训，引用已有结论） |

---

## 3. 主要实验结果和性能指标

### 📈 总体性能提升（Table 2 & Table 4）

#### ✅ DFlash 结果（Greedy decoding, $T=0$）

| Method | GSM8K | MATH-500 | HumanEval | MBPP | MT-Bench | Alpaca | **Avg. $T$** |
|--------|-------|----------|-----------|------|----------|--------|--------------|
| Decay-only | 2.29 | 2.39 | 2.86 | 2.85 | 2.03 | 2.01 | **2.40** |
| AUF-only | 2.51 | 2.64 | 3.09 | 3.09 | 2.18 | 2.15 | **2.61** |
| AUF+decay | 2.50 | 2.64 | 3.09 | 3.08 | 2.19 | 2.14 | **2.61** |

> 💡 **平均 $T$ 从 2.40 → 2.61（↑8.8%），且在所有 6 个 benchmark 上均有增益**

#### ✅ Domino 结果（Greedy decoding, $T=0$）

| Method | Avg. $T$ |
|--------|---------|
| Decay-only | 2.56 |
| B-AUF | 2.66 |
| **B-AUF+D** | **2.68** |

> 💡 **最高达 2.68（↑4.7%）**，表明 AUF 在更强 baseline 上仍有提升空间

---

### 🔬 消融实验与关键发现

#### （1）AUF 是否需要 decay？→ **不需要！**

- 在 DFlash 上，**AUF-only 与 AUF+decay 完全等效**（2.61 vs 2.61）
- 表明：**一旦支持集被正确截断，position decay 成为 inert（冗余）**
- ✅ AUF 自动实现了“earlier tokens matter more”的归纳偏置，无需手动设计

#### （2）哪个分支应用 AUF 最有效？

在 Domino 中尝试多种组合：

| Variant | 描述 | Greedy $T$ |
|--------|------|-----------|
| B-AUF | AUF 仅用于 base branch | 2.66 |
| S-AUF | 两分支共享 base-auf 支持 | 2.63 |
| F-AUF | 两分支各自用自身 first-error | 2.63 |
| **B-AUF+D** | base 分支用 AUF + decay | **2.68** |

> 🔍 发现：**gain 主要来自 base branch（proposer）的训练改进**，final branch 加 AUF 无显著收益

#### （3）训练动态分析（Figure 4）

- **Common-token accuracy 反常现象**：
  - Decay-only 的 common-token accuracy **始终高于 AUF**
  - 但其 decode performance（$T$）却更低
- ✅ 解释：Decay-only 学会了“重建后缀”，但这部分在推理中从未被验证，属于 **off-deployment learning**
- AUF 则专注于提升 prefix reliability，推动 $j^*$ 向右移动，形成 **implicit easy-to-hard curriculum**

#### （4）Conditional acceptance rate 提升（Figure 5）

- AUF 显著提升了各位置的 conditional acceptance $\alpha_k$
- 曲线整体上移，尤其在早期位置（1~4）
- 表明 AUF 真正优化了 **left-to-right prefix consistency**

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **Train-inference misalignment 是真实且可修复的问题**  
   - Full-block CE 导致模型学习无关任务（suffix reconstruction）
   - AUF 通过 **hard truncation at first greedy error** 有效对齐训练与推理语义

2. **AUF 是最小干预的有效方案**  
   - 仅修改 loss 支持集，不改架构、不增 rollout、不调超参
   - 实现了与更复杂方法（如 D-PACE、SpecDiff-2）相似甚至更好的效果

3. **Position decay 在 AUF 下变得冗余**  
   - “Earlier tokens matter more” 可由模型自身 failure boundary 自动体现
   - ✅ AUF 移除了一个 hand-designed inductive bias

4. **Mask-only drafters 需要在 loss 侧实现 prefix conditioning**  
   - 与 AR 或 Domino 不同，masked drafter 无法从输入获取 gold prefix
   - AUF 提供了一种 loss-side 的替代路径

---

### ⚠️ 局限性

- 所有实验基于单一配置：**Qwen3-8B + B=16 + ShareGPT**
- 未与其他 acceptance-aware 方法（如 D-PACE）在同等算力下 head-to-head 比较
- 未探索 AUF 与 RL 类方法结合的可能性
- 当前为 SFT 阶段干预，尚未验证在 post-training 或 RL 微调中的潜力

---

### 🔮 未来工作方向

1. **扩展至不同 block size 和模型规模**，验证 AUF 是否具备通用默认性
2. **结合 soft gate（如 D-PACE）与 AUF 的 hard truncation**，探索 hybrid 设计
3. **应用于 diffusion drafter 的其他变体**（如 D2SD、JetSpec）
4. **探索 AUF 在 RL setting 中的作用**：是否可作为稳定初始化？
5. **理论分析 AUF 的收敛性质与偏差-方差 tradeoff**

---

## ✅ 总结一句话

> **AUF 通过一个极简改动——将 cross-entropy loss 截断至第一个预测错误处——解决了 masked block drafter 中长期存在的 train-inference misalignment 问题，在不增加任何开销的前提下，显著提升了 speculative decoding 的平均接受长度 $T$，并揭示了传统 position decay 的冗余性。**

</details>

---

### 4. [Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context Inference](https://arxiv.org/abs/2607.01831)

**Authors**: Wenchen Han, Gingfung Matthew Yeung, Marco Barletta, William Toner, Amory Hoste, Adam Barker  
**Category**: cs.DC  
**Published**: 2026-07-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.01831v1  

#### Abstract
Long-context inference is increasingly common in large language model (LLM) serving, driven by retrieval-augmented generation and agentic systems. In disaggregated inference, these workloads require transferring large Key-Value (KV) caches across the network, where decoding cannot begin until the tr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **disaggregated inference** 架构中，大型语言模型（LLM）的 **prefill 阶段** 和 **decode 阶段** 被部署在不同设备上，导致必须通过网络传输庞大的 **Key-Value (KV) cache**。该传输过程成为长上下文推理中的主要瓶颈，显著增加 **Time-to-First-Token (TTFT)**。

现有 KV quantization 方法虽然减少了数据量，但仍要求 **完整接收并解码 KV cache 后才能开始 decoding**，无法真正降低暴露在网络上的延迟。

---

### 提出的新方法与核心思路
Lynx 提出了一种全新的视角：**KV cache 不是必须完整接收的原子单元**，而是可以 **渐进式使用** 的资源。

其核心思想基于一个关键观察：
> KV cache 中的不同比特位对 attention 计算的贡献不均等：
> - **Most Significant Bits (MSBs)** 决定了 attention 分数的粗略结构和 token 排序；
> - **Least Significant Bits (LSBs)** 主要用于精度微调。

基于此，Lynx 引入 **Progressive Speculative Quantization**，将 KV cache 拆分为两个流进行 **split-stream 传输**：

1. **Anchor Stream**（高优先级）：包含 MSBs，用于快速启动 speculative decoding。
2. **Residual Stream**（低优先级）：包含 LSBs，在后台并发传输，用于后续验证和修正。

解码器在收到 Anchor Stream 后立即开始生成 **speculative tokens**，并在 Residual Stream 到达后执行验证，确保最终输出与全精度模型一致。

---

### 相比现有方法的优势
| 维度 | 现有方法（如 INT4/INT8/Cachegen） | Lynx |
|------|-------------------------------|------|
| **延迟** | 减少数据量，但 decoding 必须等待完整传输 | 显著降低 TTFT，实现通信与计算重叠 |
| **精度** | 低比特量化导致精度下降（尤其 INT4） | 保持 BF16/INT8 级别的高精度 |
| **机制** | 静态压缩 + 单次传输 | 动态分层量化 + 流水线 speculative 执行 |

Lynx 成功实现了 **“INT4 级别的 TTFT” + “BF16/INT8 级别的 accuracy”**，打破了传统方法中“低延迟 vs 高精度”的权衡。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 任务类型 | 上下文长度 | 评估指标 |
|--------|----------|------------|-----------|
| **MMLU-Pro** | 多选问答（Chain-of-Thought） | 16K–128K | Accuracy |
| **Needle-in-the-haystack** | 信息检索 | ~10K | Rouge-L |
| **QMSum** | 会议摘要生成 | ~10K | Rouge-L |

---

### 实验设置
- **硬件平台**：华为 Atlas A2 服务器 ×2，每台配备 8 个 Ascend 910B4 NPU（32GB HBM）
- **网络模拟**：通过限速器模拟 10–50 Gbps 带宽环境，以复现长上下文下的通信瓶颈
- **模型**：
  - LLaMA 3.1 8B Instruct
  - Qwen3 32B
  - Mistral 3 24B Instruct
- **上下文长度**：最高扩展至 128K tokens（使用 YaRN 方法）

---

### 评估指标
- **TTFT (Time-to-First-Token)** / **TTKT (Time-to-kth-Token)**：衡量响应延迟
- **End-to-End Inference Accuracy**：根据不同任务定义（Accuracy 或 Rouge-L）
- **Compression Error**：使用 **vNMSE**（normalized Mean Squared Error）衡量 KV cache 重建误差

---

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **BF16** | 无压缩，原始精度基准 |
| **INT8** | 标准 8-bit 量化 |
| **INT4** | 标准 4-bit 量化 |
| **Cachegen [33]** | 先进的 delta encoding + token-wise 压缩方案（混合 INT4/INT8） |

> 注：由于 Cachegen 未支持 Ascend NPU，作者进行了 best-effort 移植，仅报告其 accuracy 表现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Figure 9）

#### 在 MMLU-Pro + Qwen32B + 16K context 下的表现：
| 方法 | TTFT | TT32T | Accuracy |
|------|------|-------|---------|
| BF16 | 4.4s | 5.9s | 85.25% |
| INT8 | 2.3s | 3.8s | 85.06% |
| INT4 | 1.4s | 2.9s | 76.46% |
| Cachegen | N/A | N/A | 80.07% |
| **Lynx (Ours)** | **1.6s** | **3.4s** | **85.20%** |

✅ **Lynx 实现了接近 INT4 的 TTFT（仅比 INT4 慢 0.2s），同时保持了 BF16/INT8 级别的 accuracy**

---

### 与基线方法的对比结果

#### ✅ 延迟优势
- 相比 **INT8**，Lynx 平均提升 **TTFT 达 1.43×**（最高减少 30.5%）
- 相比 **INT4**，TTFT 接近，但 accuracy 提升高达 **8.7%**
- 在 128K context 下，Lynx 的 TT64T 比 INT8 快 **0.84s**

#### ✅ 精度优势
- Lynx 的 accuracy 与 BF16、INT8 无统计差异（±0.3%）
- 相比 Cachegen，accuracy 提升高达 **5.1%**
- 相比 INT4，accuracy 提升 **~8–9%**

#### ✅ 压缩误差更低
| 方法 | vNMSE (LLaMA 8B + MMLU) |
|------|--------------------------|
| INT4 | 0.53 |
| INT8 | 0.0042 |
| Cachegen | 0.015 |
| **Lynx** | **0.00017** |

➡️ Lynx 的重建误差比所有基线低 **至少一个数量级**

---

### 消融实验结果（Ablation Study）

#### Lynx-INT4 / Lynx-INT8 对比
- **Lynx-INT4**：仅用 hierarchical quantization，不分流 → accuracy 高于标准 INT4（因优化量化算法）
- **Lynx-INT8**：同理，优于标准 INT8
- **完整 Lynx**：在上述基础上加入 split-stream + speculative decoding → 显著降低 TTFT

👉 结论：**hierarchical quantization 提升精度，split-stream + speculative decoding 提升速度**

#### Acceptance Rate 分析（Figure 10）
- 在 MMLU + Qwen 工作负载下：
  - 平均生成 **21.43 个 speculative tokens**
  - 其中 **19.38 个被接受**（接受率 >90%）
  - 有 **64.8% 概率整段序列全部接受**
- 理论分析显示：70% 概率接受 ≥20 个 token

👉 高接受率说明 Anchor Stream 足够准确，能有效支撑 speculative decoding

---

## 4. 关键结论和发现

### 主要发现
1. **KV cache 可以被渐进式使用**：MSBs 足以支撑高质量的 speculative decoding。
2. **通信不再是阻塞操作**：通过 split-stream + speculative execution，将 KV transfer 从“阻塞依赖”转变为“可流水线资源”。
3. **无需牺牲精度换取速度**：Lynx 实现了 **低延迟（INT4-level）与高精度（BF16-level）的统一**。
4. **增益随上下文增长而放大**：context 越长、带宽越低，Lynx 的优势越明显。

---

### 方法的局限性
1. **硬件依赖性**：当前原型基于 Ascend NPU 实现，虽设计通用，但需适配其他平台（如 GPU）。
2. **最大 speculative token 数限制为 64**：超过后需等待 Residual 完成，可能影响极长生成任务。
3. **Anchor/Residual 比特宽度固定为 4+4**：未探索其他配置（如 5+3、6+2），存在进一步优化空间。

---

### 未来工作方向
1. **探索不同的 bit-width 配置**：研究 Anchor 与 Residual 的最优比特分配策略。
2. **跨硬件平台移植**：将 Lynx 扩展到 NVIDIA GPU、TPU 等主流加速器。
3. **结合其他压缩技术**：如与 sparsification 或 lossless encoding 融合，进一步提升效率。
4. **动态调整 speculative depth**：根据网络状况和模型行为自适应控制 speculative token 数量。

---

## 总结
Lynx 重新定义了 KV cache 的传输范式，提出 **progressive speculative quantization**，通过 **分层量化 + split-stream 传输 + speculative decoding** 的协同设计，在不牺牲精度的前提下显著降低了长上下文推理的启动延迟。其实验结果表明，Lynx 是目前唯一能同时达到 **INT4 级延迟** 和 **BF16 级精度** 的 KV transfer 方案，为下一代高效 LLM serving 系统提供了新的设计思路。

</details>

---

### 5. [Hawk: Harnessing Hardware-Aware Knowledge for High-Performance NPU Kernel Generation](https://arxiv.org/abs/2607.01590)

**Authors**: Junyi Wen, Ruiyan Zhuang, Yongjia Xu, Pengtu Li, Rui Zou, Hongyi Chen, Chingman Wan, Puxu Yang, Wuhui Chen, Yanlin Wang  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.01590v1  

#### Abstract
Developing high-performance kernels for Neural Processing Units (NPUs) is a critical industry bottleneck, requiring developers to manually navigate implicit hardware constraints and strict memory hierarchies. While large language models offer immense automation potential, they fail catastrophically ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Hawk: Harnessing Hardware-Aware Knowledge for High-Performance NPU Kernel Generation**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Models (LLMs)** 的 NPU kernel 自动生成面临严重瓶颈：
- LLMs 在 NPU 编程语言（如 Ascend C）上的 **Comp@1（编译成功率）和 Pass@1（功能正确率）极低**（仅约 13.3%），远低于在 CUDA 上的表现（可达 100%/80%）。
- 失败主因是缺乏对硬件约束的先验知识，导致出现大量 **syntax-level 错误**（如非法 API 调用、内存布局不匹配）和 **hardware-level 运行时崩溃**（如 Unified Buffer 溢出）。
- 即使代码能运行，也常因未启用架构特定优化（如 HF32 模式、多核并行）而性能低下，**执行速度比等效 GPU 内核慢近 20×**。

传统方案（fine-tuning 或 IR-based 编译器）存在可扩展性差的问题：
- Fine-tuning 需要持续更新训练数据和重新训练；
- IR/DSL 方法依赖专家手动维护语法和转换规则，难以适应快速迭代的 NPU 工具链。

---

### 提出的新方法：Hawk
Hawk 是一个 **无需训练（training-free）** 的框架，通过动态利用“硬件感知知识”来指导 LLM 生成高性能 NPU kernel。其核心思想是：**从已有正确的 NPU kernels 中提取并结构化硬件相关的经验知识，并在生成过程中按需检索与精炼这些知识**。

#### 三大核心模块（创新设计）：

| 模块 | 功能 |
|------|------|
| **Run-Time Knowledge Synthesis Module** | 引入 **Triple-Part Executable Knowledge Representation**：<br>1. **Metadata Layer**（索引触发条件，如算子类型）<br>2. **Rationale Layer**（自然语言解释硬件限制）<br>3. **Artifact Layer**（可执行代码模板）<br>支持运行时自动捕获新知识，实现知识库自进化。 |
| **Bottleneck-Aware Knowledge Retrieval Module** | 提出 **2D-Retrieval 范式**：<br>- **Syntactic Space**：确保 API 签名精确匹配（使用 BM25）<br>- **Hardware-Aligned Semantic Space**：捕捉深层硬件瓶颈（使用 dense embedding）<br>采用 **Reciprocal Rank Fusion (RRF)** 融合两个维度，避免权重调参。 |
| **Effect-Driven Knowledge Distillation Module** | 利用 LLM 驱动的语义仲裁机制，结合实际执行反馈：<br>- 自动剔除引发 crash 的错误知识<br>- 合并冗余策略<br>- 解决冲突规则（contextual merging）<br>保证知识库长期高质量、高效率。 |

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| ✅ **免训练** | 不需要 fine-tuning 或 RL，适应硬件更新只需向知识库添加条目，成本极低。 |
| ✅ **高准确性** | 显著提升生成代码的功能正确性和编译成功率。 |
| ✅ **高性能输出** | 生成的 kernel 具备架构级优化，显著优于 baseline。 |
| ✅ **强可扩展性** | 支持跨 kernel 类型迁移知识（如从 L1 到 L2），适用于复杂算子。 |
| ✅ **抗噪声能力强** | 内建知识蒸馏机制防止低质量知识污染上下文。 |

---

## 2. 核心实验方法和设置

### 数据集
- **训练/初始化知识库来源**：`ops-nn` 开源项目中的 158 个 L1 kernel（共 2,690 个文件），严格与测试集隔离。
- **测试基准**：`CANNBench`，包含 L1 和 L2 级别的 Ascend C kernels，每个 operator 提供 20 个测试用例。
- **评估算子数量**：
  - L1：8 个（如 `exp`, `gelu`, `foreach_norm`）
  - L2：16 个（如 `group_norm`, `softmax`, `rms_norm`）

---

### 实验设置
- **平台**：搭载 8 块 Ascend 910B2 NPU，运行 CANN 8.5.1。
- **LLM Backbone**：主要使用 GLM-5.1；补充实验使用 DeepSeek-V4-Flash 验证泛化能力。
- **Agent 框架基础**：基于 CANNBot 构建，Hawk 作为增强插件集成。
- **最大迭代次数**：统一设为 30 次。
- **Top-K 检索数**：K=6（经消融实验确定最优值）。

---

### 评估指标
| 指标 | 定义 |
|------|------|
| **Compilation Success (CS)** | 是否成功编译（二值） |
| **Correctness (Acc)** | 测试用例通过比例（0~1） |
| **Speedup** | $ \text{Latency}_{\text{std}} / \text{Latency}_{\text{generated}} $，越高越好 |
| **Score** | 综合评分：<br>$ \text{Score} = 20\times CS + 30\times Acc + 50\times PR $<br>其中 $ PR = \frac{L_{\text{std}} - L_{\text{hw}}}{(L_{\text{opt}} - L_{\text{hw}}) + (L_{\text{std}} - L_{\text{hw}})} $，反映接近理论下限的程度 |

---

### 基线方法对比
| Baseline | 描述 |
|---------|------|
| **Vanilla LLM** | 原始 Claude Code，无任何辅助 |
| **Few-Shot** | 提供功能相似 kernel 的代码片段作为 prompt 示例 |
| **CANNBot** | 官方 agent 工具，提供标准工具链和文档支持，但无隐式硬件知识 |

---

## 3. 主要实验结果和性能指标

### 总体性能表现（RQ1）
- **准确率提升**：平均 Acc 从 baseline 最高的 **49.4%（CANNBot）提升至 80.0%**。
- **执行加速**：平均 Speedup 达到 **2.2×**，最高达 **2.39×**。
- **综合得分领先**：在多个 L1 operator 上，Hawk 快速收敛到高分（如 `exp` 第一次尝试即得 62.82 分，最终达 83.82），而其他方法停滞不前或无法通过测试。

> 🔹 图 6 和图 7 展示了 Hawk 在 `exp`, `gelu`, `foreach_norm` 等算子上全面超越所有 baseline。

---

### 消融实验结果（RQ2）
| 变体 | Acc | Speedup | 分析 |
|------|-----|--------|------|
| **Hawk-Full** | 0.84 | 1.00× | 完整版本 |
| **w/o Structured Representation** | 0.10 | 0.09× | 去掉三段式表示后性能崩塌 → 证明结构化知识表达至关重要 |
| **w/o 2D-Retrieval** | 0.45 | 0.41× | 仅靠语义检索无法精准定位硬件瓶颈 |
| **w/o Distillation** | 0.28 | 0.25× | 加入少量未验证知识即可导致严重退化 → 蒸馏模块是防御逻辑污染的关键 |

> ⚠️ 仅增加 7 条未经验证的知识（总量 +4.4%）就使 Acc 下降超过 60%，说明 **知识纯度极其重要**。

---

### 可扩展性与泛化性（RQ3）
| 场景 | 结果 |
|------|------|
| **L1 → L2 kernel 迁移** | 在 16 个 L2 算子中，CANNBot 有 12 个 Acc 为 0%，而 Hawk 实现最高 **65% Acc** 和 **3.5× Speedup**，表明硬件知识具有跨层级可迁移性。 |
| **弱 LLM Backbone 泛化**（DeepSeek-V4-Flash） | 平均 Acc 从 44.4% → **93.1%**，Speedup 从 0.53× → **1.35×**，证明 Hawk 对模型能力要求较低，通用性强。 |

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 在 NPU kernel 生成失败的根本原因是缺乏 hardware-aware knowledge**，尤其是 API 使用规范、内存容量限制和架构优化技巧。
2. **简单的代码复用（few-shot）不足以解决问题**：即使语法正确，忽略硬件边界会导致运行时崩溃（如 UB overflow）。
3. **共享硬件模式广泛存在于不同 kernels 中**：高达 70% 的 API 和功能范式重叠，为知识迁移提供了坚实基础。
4. **显式注入硬件约束信息（如 UB=256KB）可显著恢复功能正确性**，但需系统化管理而非人工提示。
5. **性能瓶颈可通过微小优化提示解决**（如 `SetHF32(true)`），说明高层语义无法替代底层硬件认知。

---

### 方法的局限性
- **依赖初始知识沉淀**：冷启动阶段需一定量高质量 kernel 进行预提取（cold-start data pre-precipitation）。
- **检索效率受限于 Top-K 设置**：过大浪费 token，过小影响覆盖率，需权衡。
- **目前聚焦 Ascend NPU 架构**：虽基于 Da Vinci 架构具备通用潜力，但仍需验证在其他厂商 NPU（如寒武纪、昆仑芯）上的适用性。

---

### 未来工作方向
- 扩展至更复杂的 **fused operators** 和 **multi-NPU cluster** 场景。
- 探索 **自动化知识标注与版本管理机制**，应对 NPU SDK 快速演进。
- 将 Hawk 范式推广至其他领域专用加速器（如 TPU、Gaudi）。
- 结合轻量级 fine-tuning 与 knowledge harnessing，构建混合增强体系。

---

> ✅ **一句话总结**：  
> **Hawk 通过“结构化知识合成 + 瓶颈感知检索 + 效果驱动蒸馏”的闭环机制，在无需训练的前提下，将 NPU kernel 生成的准确率从 49.4% 提升至 80.0%，并实现最高 2.2× 的性能加速，解决了 LLM 在专用硬件编程中的“知其然不知其所以然”难题。**

</details>

---

### 6. [OPINE-World: Programmatic World Modeling with Ontology-error-Prioritized Interactive Exploration](https://arxiv.org/abs/2607.01531)

**Authors**: David Courtis, Wenhao Li, Scott Sanner  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.01531v1  

#### Abstract
Learning how an environment behaves from interaction is central to building agents that adapt to unfamiliar tasks. World models learned with deep networks are flexible but data-hungry and transfer poorly beyond their training distribution. Program-synthesized world models, written as source code by ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：OPINE-World: Programmatic World Modeling with Ontology-error-Prioritized Interactive Exploration

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **World Model** 学习方法存在两大瓶颈：
- **基于深度网络的方法**（如 Dreamer、MuZero）虽然通用性强，但**数据效率低**，且在训练分布之外泛化能力差。
- **基于程序合成的方法**（如 WorldCoder、PoE-World）虽数据高效、可解释性强，但通常依赖预定义的 **object vocabulary** 和状态结构，难以扩展到像素级、结构未知的环境。

OPINE-World 旨在解决这一挑战：**如何从原始像素交互中在线学习一个可复用、可验证、对象中心化的程序化世界模型（programmatic world model），而无需先验的对象结构或任务语义**。

### 提出的新方法与创新思路
OPINE-World 是一种由两个协作的 LLM Agent 构成的系统，通过“假设-测试”循环实现在线世界建模。其核心创新包括：

#### （1）双代理协作架构（Two Cooperating LLM Agents）
- **Goal-Directed Agent**：负责在环境中探索，执行动作并记录交互日志。
- **World-Model Agent**：负责从回放缓冲区（replay buffer）中合成和修复程序化世界模型。
- 两者共享一个符号化的世界模型 `M = (S, A, T, G)`，但职责分离，避免单代理的认知过载。

#### （2）对象本体发现机制（Object Ontology Discovery）
- 不依赖引擎提供的 sprite 列表或对象 ID。
- 自主从 64×64 像素帧中提取对象（via `extract_objects` 函数），进行跨时间步配对，并动态推断对象类型（type）划分。
- 支持在无结构先验的像素渲染环境中运行。

#### （3）本体误差引导探索（Ontology Error）
- 提出一种贝叶斯度量——**ontology error**，用于衡量当前对象类型划分对观测行为的解释能力。
- 结合类型不确定性（type uncertainty）和效应不确定性（effect uncertainty），以 **noisy-OR** 形式聚合为每对象的探索信号。
- 高 `n` 值区域被优先探索，驱动系统主动减少模型不确定性。

#### （4）精确重放验证 + Counterexample-Guided Synthesis
- 模型接受标准是 **exact replay**：必须完美复现所有历史状态转移（attribute-by-attribute, object-for-object）。
- 若预测失败，则生成 **counterexample** 并触发新一轮合成。
- 保证模型始终与观测一致，防止错误累积。

---

## 2. 核心实验方法和设置

### 数据集
- **ARC-AGI-3** [ARC Prize Foundation, 2026]：一个面向技能获取效率的基准测试平台。
  - 包含 25 个游戏，每个游戏有多关卡。
  - 所有语义信息（object vocabulary、goal、action semantics）均被隐藏。
  - 仅提供原始像素帧（64×64 color indices）和稀疏奖励信号（仅当通关时报告成功）。

### 实验设置
- **在线策略学习**（on-policy, no reset）：每个游戏只玩一次，不允许重采样。
- **无每游戏训练**（no per-game training）：不进行微调、提示工程或演示学习。
- 使用 **Claude Opus 4.8** 作为底层 LLM，在文件系统沙箱中运行，禁止访问真实游戏源码或网络。

### 评估指标
- **Action Efficiency Score**：基于 `(human / agent)^2` 的加权平均得分，上限为 1.15，按关卡索引加权，单游戏总分上限为 100。
- **Games Won**：通关的游戏数量（全部关卡完成）。
- **Levels Cleared**：总共通过的关卡数。
- **Action Count**：完成游戏所用的动作总数。

### 基线方法对比
| 方法 | 类型 | 是否预训练 | 主要特点 |
|------|------|-----------|---------|
| **baseline1** | 单代理编码Agent | 否（使用 GPT-5.5） | 强推理模型，自行合成并规划 |
| **Vision** | 连续学习视觉Agent | 是（离线训练于评测集） | 具备先验权重，非零样本 |
| **WorldCoder** | 程序合成系统 | 否 | CEGIS + optimism constraint |
| **Latent World Models** | Dreamer/MuZero 类 | 否 | 学习隐空间动力学 |

> 注：Vision 因在相同评测集上预训练，不参与“无训练”比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1 & Figure 5）

| System | Score (%) | Games Won (/25) | Mean Levels Cleared |
|--------|------------|------------------|-----------------------|
| **OPINE-World (ours)** | **78.4** | **20** | **0.90** |
| baseline1 | 63.8 | 14 | 0.80 |
| Vision | 63.2 | 12 | 0.73 |
| WorldCoder | 0.0 | 0 | 0.00 |
| Latent WMs | 0.0 | 0 | 0.00 |

- OPINE-World 在 **25 个游戏中解决了 20 个**，共 **183 个关卡中通过了 160 个**。
- 总体 action-efficiency 得分为 **78.4%**，显著优于最强零样本基线 baseline1（63.8%）。

### 与基线方法的对比结果
- **在困难游戏上优势明显**：baseline1 完全无法解决的 6 个游戏（如 `re86`, `tn36`, `vc33` 等），OPINE-World 全部攻克。
  - 例如，在 `m0r0` 上，OPINE-World 仅用 **259** 步通关，而 baseline1 花费 **2573** 步仍未完成。
- **行动更高效**：在 OPINE-World 解决的 20 场游戏中，**16 场所用动作少于人类基准**。
  - 平均而言，**human / OPINE-World ≈ 1.7**，即人类平均花费 1.7 倍于该系统的动作。
  - 最大提速达 **4.3×**（`m0r0`）、**3.5×**（`lp85`）、**3.0×**（`cn04`）。
- **避免无效探索**：baseline1 经常在一个难题关卡上消耗上千动作仍失败；OPINE-World 更快识别不可解状态并转向建模修复。

### 消融实验（Ablations）
- 文中未提供详细的逐组件消融研究，因各模块高度耦合，移除任一部分会导致系统崩溃。
- 作者指出未来将开展更精细的 ablation study。

---

## 4. 关键结论和发现

### 主要发现
1. **双代理协作 + 假设-测试循环** 显著提升了复杂任务下的建模鲁棒性和效率。
2. **本体误差（ontology error）是一种有效的内在探索信号**，能引导系统聚焦于尚未理解的对象与上下文。
3. **exact replay 验证机制确保了模型可靠性**，避免了基于似然或乐观主义准则带来的误判。
4. **无需先验结构也能发现有效对象本体**，使程序化世界模型适用于像素级开放环境。
5. OPINE-World 在 **零样本设定下超越强基线**，甚至在多个维度上优于经过预训练的 Vision Agent。

### 方法的局限性
1. **可观测马尔可夫假设限制**（Observable-Markov Determinism）：
   - 当存在隐藏状态时，exact replay 测试可能永远无法通过，ontology error 也无法降至零。
2. **感知模块为合成产物**：
   - `extract_objects` 函数由 LLM 生成，可能存在分割错误或遗漏对象。
3. **规划器规模有限**：
   - 使用的是有界前向搜索（bounded forward search），在高分支因子任务中容易达到搜索上限。
4. **单次运行无方差估计**：
   - 每个游戏仅运行一次，结果不具备统计显著性检验基础。

### 未来工作方向
- 将方法扩展至 **部分可观测环境**（POMDP）和 **随机动力学环境**。
- 改进感知模块，引入更鲁棒的 **raw-pixel object segmentation** 方法。
- 设计更强的合成搜索策略（如启发式引导、模块化归纳）。
- 探索多轮迭代下的稳定性与收敛性分析。
- 在真实机器人或更复杂的模拟环境中部署验证。

---

> ✅ **一句话总结**：  
> OPINE-World 通过双 LLM 代理协作、本体误差引导探索与精确重放验证，在完全未知的像素环境中实现了高效的程序化世界建模，在 ARC-AGI-3 上以零样本方式解决 20/25 游戏，行动效率全面超越现有方法。

</details>

---

### 7. [Atomic Task Graph: A Unified Framework for Agentic Planning and Execution](https://arxiv.org/abs/2607.01942)

**Authors**: Yue Zhang, Sihan Chen, Ziwen Huang, Hanyun Cui, Kangye Ji, Zhi Wang  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.01942v1  

#### Abstract
LLM-based agents have shown strong potential for solving complex multi-step tasks, yet existing performance improvements often rely on either scaling to larger backbone models or task-specific fine-tuning. The former incurs substantial computational costs, while the latter typically generalizes poor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Atomic Task Graph: A Unified Framework for Agentic Planning and Execution**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 LLM 的智能体（LLM-powered agents）在处理复杂多步任务时面临以下挑战：
- **依赖线性决策流程**：大多数 prompt-based 方法将任务求解视为线性文本轨迹（textual trajectory），导致子任务间的输入-输出依赖关系隐式存在，难以追踪和复用中间结果。
- **错误传播严重**：一旦某一步出错，错误会沿轨迹传播，修复通常需要全局回溯或重新规划（replanning），效率低下。
- **上下文膨胀引发幻觉**：随着执行过程增长，累积的文本上下文容易导致模型产生无效或幻觉动作（hallucinated actions）。
- **并行性差**：缺乏对独立分支的显式建模，无法实现并行执行。

### **提出的新方法：Atomic Task Graph (ATG)**
作者提出了 **Atomic Task Graph (ATG)** ——一个统一的、贯穿 planning 和 execution 阶段的控制框架。其核心思想是将任务求解过程建模为一个显式的有向无环图（DAG），其中每个节点是一个原子工具调用单元（atomic tool-use unit），边表示输入-输出依赖关系。

#### **三大关键技术组件**：
1. **Interface-Preserving Recursive Graph Compilation（接口保持的递归图编译）**
   - 将高层任务从粗粒度逐步分解为细粒度子图，直到所有节点均为可执行的原子操作。
   - 每次分解都保持父节点的输入-输出接口不变，确保模块化和组合性。
   - 记录整个图演化历史（refinement history），用于后续错误溯源和局部修复。

2. **Dependency-Aware Graph Execution（依赖感知的图执行）**
   - 基于图结构进行调度：只有当所有前置节点完成且输入就绪时，节点才可执行。
   - 支持**并行执行**独立分支，显著提升执行效率。
   - 引入“思维实验”（thought experiment）作为预执行验证机制，在真实环境交互前检测潜在失败（如工具误选、缺失步骤等）。

3. **Minimal Necessary Subgraph Repair（最小必要子图修复）**
   - 当运行时或预执行阶段检测到失败时，系统通过图演化历史定位最小子图范围（lowest common historical ancestor）。
   - 仅对该受影响区域进行修复，冻结已验证部分，避免不必要的全局重规划。

### **相比现有方法的优势**
| 维度 | 传统方法（如 ReAct, ToT） | ATG |
|------|--------------------------|-----|
| 结构表达能力 | 线性链或树状探索 | 显式 DAG 表达非线性依赖 |
| 错误处理方式 | 全局回溯或重试 | 局部定位 + 最小修复 |
| 执行模式 | 串行决策 | 并行执行独立分支 |
| 上下文管理 | 文本历史不断增长 | 节点上下文局部化，减少幻觉 |
| 可复用性 | 中间结果难复用 | 已验证子图可冻结保留 |

> ✅ **优势总结**：ATG 不仅提升了成功率，还增强了执行效率、鲁棒性和可靠性，尤其适用于长视野（long-horizon）、高复杂度的任务场景。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在三个具有代表性的交互式智能体基准上进行了评估：
- **ALFWorld**：基于文本的具身任务环境，模拟家庭中的多步操作（如“把苹果放进微波炉加热”）。
- **WebShop**：模拟电商网站上的商品搜索、比较与购买决策任务。
- **ScienceWorld**：面向科学推理的多步探索任务，测试智能体理解物理规律和实验设计的能力。

### **评估指标**
- **主指标**：平均任务奖励（average reward），越高越好。
  - ALFWorld 使用二值奖励（0/1）
  - WebShop 和 ScienceWorld 使用密集奖励（0~1）
- **辅助指标**：
  - 平均执行步数（Average Steps）：越低越好 → 反映执行效率
  - 幻觉动作率（Hallucinatory Action Rate）：越低越好 → 反映行为可靠性

### **实验设置**
- **骨干模型**（Backbone Models）：
  - 开源小模型：`Mistral-7B`, `Gemma-7B`, `Llama-3-8B`
  - 商业大模型对比：`GPT-3.5-Turbo`, `GPT-4`（均配合 ReAct）
- **基线方法对比**：
  - **隐式控制**：ReAct, Reflexion
  - **显式结构化控制**：Tree-of-Thoughts (ToT), Plan-over-Graph (PoG)
  - **多智能体控制**：CAMEL
- **实现细节**：
  - ATG 完全在推理时运行，无需 fine-tuning 或参数更新。
  - 所有方法共享相同的 backbone checkpoint、tokenizer、decoding config 和 action budget，保证公平比较。
  - 实验平台：6 × NVIDIA RTX 4090 GPU

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| Backbone | Method | ALFWorld | WebShop | ScienceWorld |
|---------|--------|----------|---------|-------------|
| Mistral-7B | ReAct | 6.57 | 14.63 | 19.12 |
|            | PoG (best baseline) | 23.72 | 24.18 | 28.63 |
|            | **ATG** | **55.73** | **62.75** | **49.81** |
| Gemma-7B   | ReAct | 4.38 | 5.93 | 3.54 |
|            | PoG | 32.85 | 12.56 | 13.85 |
|            | **ATG** | **58.71** | **64.93** | **52.03** |
| Llama-3-8B | ReAct | 3.29 | 19.32 | 23.67 |
|            | PoG | 21.17 | 31.66 | 35.72 |
|            | **ATG** | **63.65** | **68.36** | **56.79** |

> 🔺 **观察**：
> - ATG 在所有 backbone 和 benchmark 上均大幅领先。
> - 即使使用 7B–8B 规模的开源模型，ATG 的表现也超过了 GPT-4 + ReAct（例如在 ALFWorld 和 WebShop 上）。

### **与基线方法的对比结果**
- 相比最强 baseline（PoG），ATG 提升幅度高达：
  - **+32.01 pts on ALFWorld (Mistral)**
  - **+38.57 pts on WebShop (Mistral)**
- 相比 ReAct，提升超过 **40+ points** 多个场景。
- 表明：仅仅引入图结构（如 PoG）不足以带来质变，而 ATG 利用图作为**可执行底座**（executable substrate）进行依赖跟踪、调度和恢复，才是成功关键。

### **消融实验结果（Ablation Studies）**

#### **(1) 移除 “Thought Experiment”**
- 导致性能下降：
  - Mistral 上 ALFWorld ↓4.87 pts，ScienceWorld ↓3.99 pts
- 说明：预执行验证能有效识别潜在错误（如工具不匹配、依赖断裂），防止资源浪费。

#### **(2) 移除 “Subgraph Repair”**
- 性能下降更明显：
  - Mistral 上 ALFWorld ↓7.72 pts，ScienceWorld ↓6.48 pts
- 说明：**局部修复机制对长程任务至关重要**，避免因局部错误导致全局崩溃。

> 📊 图表支持：Figure 4 显示两个组件缺一不可，共同构成鲁棒闭环。

#### **其他关键指标对比**

| Method | Avg Steps (ALF) | Avg Steps (Web) | Hallucination Rate (ALF) |
|-------|------------------|------------------|----------------------------|
| ReAct | 31.42 | 8.76 | 42.86% |
| PoG | 24.57 | 7.12 | 28.57% |
| **ATG** | **18.36** | **5.84** | **12.14%** |

> ✅ **结论**：
> - ATG 将平均执行步数降低约 **25%-40%**，得益于并行执行。
> - 幻觉动作率下降至 **12.14%**，相对 ReAct 减少 **71.7%**，体现上下文局部化的有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **显式图结构 + 可执行语义 = 更强智能体控制**
   - 将任务建模为 DAG 并维护输入-输出依赖，使得计划更具结构性、可解释性和可复用性。
2. **递归编译 + 接口保持 → 支持模块化与演化追溯**
   - 分解过程记录图演化路径，为错误定位提供依据。
3. **依赖感知执行显著提升效率**
   - 并行调度独立分支，减少总执行深度；预执行“思维实验”提前拦截风险。
4. **最小必要修复优于全局重规划**
   - 局部修复保留已验证成果，提高容错能力和稳定性。
5. **优秀控制框架可缩小大小模型差距**
   - ATG + Llama-3-8B 超越 GPT-4 + ReAct，表明高效 control 框架是弥补 backbone 规模不足的有效途径。

### **局限性**
- 对 backbone LLM 的**任务分解能力**高度依赖，若初始分解错误，可能影响整体效果。
- 在**噪声观测或多跳长距离依赖**场景中，故障定位仍具挑战。
- 当前实验集中于**纯文本环境**，尚未扩展到多模态或真实世界机器人任务。
- 对简单任务引入额外开销（overhead），可能不如轻量级方法高效。

### **未来工作方向**
- 扩展至多模态输入输出（vision + language + action）。
- 探索动态图重构机制，适应环境变化。
- 结合 memory 机制增强长期状态追踪。
- 应用于真实世界机器人控制、自动驾驶、自动化科研等领域。

---

> 💡 **一句话总结**：  
> **ATG 提出了一种训练免费、通用性强的智能体控制范式，通过构建显式的 Atomic Task Graph，实现了从“线性试错”到“结构化执行+精准修复”的跃迁，在多个复杂基准上以小模型超越大模型基线，展示了控制机制设计的巨大潜力。**

</details>

---

### 8. [I\textsuperscript{2}RiMA: Spectral Riemannian Representation with Temporal Attention for Mental Stress Detection based on EEG Signals](https://arxiv.org/abs/2607.01279)

**Authors**: Cheng He, Kunyu Peng, Shangen Han, Jinming Ma, Jinhong Ding, Likun Xia  
**Category**: cs.LG  
**Published**: 2026-07-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.01279v1  

#### Abstract
Cross-subject EEG stress detection remains challenging because discriminative stress-related patterns are both subject-dependent and frequency-specific. Conventional Riemannian methods model spatial covariance mainly in the time domain, overlooking neural oscillations that are critical for high-leve...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**跨被试（cross-subject）EEG压力检测**中的两个关键挑战：
- **频段特异性模式丢失**：传统Riemannian方法在时域构建协方差矩阵，忽略了神经振荡的频率特性，导致高频/低频脑电节律中与压力相关的空间模式被混合或掩盖。
- **时间连贯性断裂**：标准的EEG分片处理将连续信号切分为独立窗口，破坏了相邻片段之间的时序依赖关系，而这种上下文信息对认知状态识别至关重要。

### 提出的新方法：I²RiMA
提出了一种名为 **I²RiMA（Intra-Inter Riemannian Manifold Attention Network）** 的新型网络架构，其核心创新包括：

#### （1）频谱感知的Riemannian表示（Spectral Riemannian Representation）
- 在**每个频率点独立构建SPD（Symmetric Positive Definite）协方差矩阵**，而非单一的时间域协方差。
- 使用**Log-Euclidean映射**将其投影到切线空间，保留通道间的Riemannian几何结构。
- 引入**频率聚类聚合（frequency cluster aggregation）**：通过K-Means将相关频率自动聚类为生理可解释的EEG波段（如delta, theta, beta等），实现数据驱动的特征选择与冗余消除。

#### （2）内外切片注意力融合机制（Intra-Inter Slice Attention Fusion）
- 设计了一个**Unsupervised Slice Attention Aggregation (USAA)** 模块：
  - **Intra-slice**：捕捉单一片段内的局部频谱动态；
  - **Inter-slice**：建模多个片段间的全局时间依赖；
- 利用**注意力机制自适应加权不同时间片段**，突出高判别力的时间窗口，抑制噪声。

### 相比现有方法的优势
- **几何保持性**：在整个流程中维护SPD流形结构，增强跨被试泛化能力。
- **生理合理性**：频率聚类结果与经典EEG节律高度一致，提升模型可解释性。
- **高效性**：仅需 **1.60M参数** 和 **31.95M FLOPs**，远低于主流Transformer类模型，适合可穿戴设备部署。
- **鲁棒性**：同时建模局部频谱细节与全局时间演化，显著提升复杂任务下的分类性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
在三个公开EEG数据集上进行验证：
| 数据集 | 被试数 | 通道数 | 采样率 | 类别 | 任务描述 |
|-------|--------|--------|--------|------|----------|
| **MIST Control** | 30 | 64 | 200 Hz | 4-class | 无压力条件下的心理算术任务 |
| **MIST Stress** | 30 | 64 | 200 Hz | 4-class | 施加时间压力与负面反馈的压力诱导任务 |
| **SEED** | 15 | 62 | 200 Hz | 3-class | 观看情绪视频引发的情绪分类任务 |

> 所有实验采用**被试间交叉验证（subject-level 5-fold CV）**，训练集与测试集无被试重叠，严格评估跨被试泛化能力。

### 实验设置
- **预处理**：重采样至200Hz、ICA去伪迹、0.5–50Hz带通滤波、z-score归一化、非重叠8秒分段。
- **输入形式**：原始EEG → FFT变换 → 频域幅度谱 → 各频率点构建协方差矩阵。
- **最优分段数 $m$**：通过“边际效应”（Marginal Effect）分析确定：
  - MIST Control: $m=29$
  - MIST Stress: $m=28$
  - SEED: $m=26$

### 评估指标
使用以下五项指标衡量性能（均以百分比报告）：
- **Balanced Accuracy (B.ACC)**：平衡准确率，应对类别不平衡
- **Precision (PreT)**：精确率
- **Recall (RecT)**：召回率
- **F1 Score (F1T)**：F1分数
- **AUC**：ROC曲线下面积

### 对比的基线方法
| 方法 | 类型 | 参数量（M） | 特点 |
|------|------|-------------|------|
| **EEGNet** | CNN | 0.05 | 小型卷积网络 |
| **BIOT** | Transformer | 3.20 | 生物信号Transformer，支持跨数据学习 |
| **LaBraM** | 大规模预训练模型 | 6.23 | 基于掩码建模的大脑语言模型 |
| **NeuroBOLT** | 多维表征学习 | 10.45 | 从EEG合成fMRI信号 |
| **CorrAtt** | 相关性注意力 | 0.22 | 基于相关矩阵的自注意力机制 |

所有模型均从零开始训练，使用相同数据划分与超参配置，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）
| 方法 | MIST Control (B.ACC) | MIST Stress (B.ACC) | SEED (B.ACC) |
|------|------------------------|----------------------|--------------|
| EEGNet | 53.26% | 48.82% | 58.01% |
| BIOT | 37.09% | 32.63% | 54.70% |
| LaBraM | 34.08% | 27.33% | 57.10% |
| NeuroBOLT | 36.45% | 29.56% | 50.23% |
| CorrAtt | 45.37% | 46.67% | 57.26% |
| **I²RiMA** (**Ours**) | **77.59%** | **75.88%** | **82.78%** |

> ✅ **I²RiMA在所有三个数据集上均达到SOTA水平**，相比最强基线提升达 **+24.33% ~ +29.21% B.ACC**。

#### 其他指标亮点：
- **AUC高达94.45%~95.61%**，表明模型具有极强的判别能力。
- 在最具挑战性的 **MIST Stress** 上，超越CorrAtt近 **29个百分点**，说明其在细微认知差异下仍能有效建模。

### 消融实验结果（Table 2）
| 变体 | MIST Control (B.ACC) | MIST Stress (B.ACC) | SEED (B.ACC) |
|------|------------------------|----------------------|--------------|
| Baseline | 69.36% | 65.93% | 76.93% |
| R-I²RiMA（仅Riemannian） | 77.59% | 75.88% | 82.78% |
| I-I²RiMA（仅Inter-slice Attention） | 74.11% | 42.29% | 58.84% |
| **Full I²RiMA** | **77.59%** | **75.88%** | **82.78%** |

> 🔍 发现：
- **Riemannian模块贡献显著**：平均提升约 **+8~10% B.ACC**，证明几何保持对跨被试迁移至关重要。
- **Inter-slice注意力在MIST上增益巨大**（+37.09%），说明时间上下文对压力任务尤为关键。
- **两者互补**：完整模型优于任一单独模块，体现“空间+时间”联合建模的有效性。

### 效率对比（Table 3）
| 方法 | 参数量（M）↓ | FLOPs（M）↓ |
|------|---------------|-------------|
| EEGNet | 0.05 | 268.10 |
| BIOT | 3.20 | 8140.23 |
| LaBraM | 6.23 | 11095.78 |
| NeuroBOLT | 10.45 | 10504.58 |
| CorrAtt | 0.22 | 1674.45 |
| **I²RiMA** | **1.60** | **31.95** |

> 💡 I²RiMA的FLOPs仅为：
- EEGNet的 **11.9%**
- BIOT的 **0.4%**
- LaBraM/NeuroBOLT的 **<0.3%**

> ⚖️ 在**精度-效率帕累托前沿**中占据左上角优势区域（见Figure 2），是目前最高效的高性能EEG压力检测模型。

---

## 4. 关键结论和发现

### 主要发现
1. **频谱分解+Riemannian建模显著提升跨被试泛化能力**：
   - 频率点级协方差构造保留了“空间-频率双重性”（spatial-frequency duality），使模型能够分别捕获低频全局调节与高频局部加工的变化。
   - Log-Euclidean映射提供了对电极位移、阻抗变化等个体差异的内在不变性。

2. **内外切片注意力机制恢复了因分片造成的信息损失**：
   - 自适应注意力权重揭示了某些时间段承载更强判别信息（如压力转换期）。
   - 时间聚合不仅提升了性能，还增强了模型对动态应激过程（allostatic process）的理解。

3. **数据驱动的频率聚类具备生理可解释性**：
   - 聚类结果自然对应delta (1–3Hz)、theta/alpha (4–12Hz)、beta (13–25Hz)、gamma (26–49Hz)，验证了方法的生物学合理性。

4. **通道重要性图谱揭示认知负荷与情绪处理的不同神经机制**：
   - MIST任务中前颞叶（FT11/FT12）和枕小脑区（CB1/CB2）权重最高 → 支持执行控制与视觉注意。
   - SEED任务中颞叶（T7/T8）与额极区（Fp1/Fp2）主导 → 符合情绪记忆与情感调控的经典回路。

### 局限性
1. 当前评估集中于**压力与情绪识别**，尚未验证其在其他EEG解码任务（如运动想象、癫痫检测）上的通用性。
2. 数据集规模有限（最大30被试），更大规模人群的泛化表现有待进一步验证。
3. 分析基于离线批处理，未实现实时在线推理部署。

### 未来工作方向
- 推进**在线实时部署**，集成至可穿戴BCI系统中，用于闭环压力调节。
- 探索**多模态融合**（如结合ECG、PPG等生理信号），提升检测鲁棒性。
- 开展**长期纵向研究**，监测个体压力轨迹演变。
- 加强**公平性审计**，确保模型在不同性别、年龄、文化背景群体中的公正性。

---

> 📌 **总结一句话**：  
> I²RiMA通过**频谱感知的Riemannian几何建模**与**内外切片注意力融合**，实现了高效、鲁棒且可解释的跨被试EEG压力检测，在三个基准数据集上取得SOTA性能，同时保持极低计算开销，为便携式心理健康监测提供了强有力的技术基础。

</details>

---

### 9. [An Optimisation Framework for the Well-Conditioned Training of Physics-Informed Neural Networks](https://arxiv.org/abs/2607.02194)

**Authors**: Joseph Webb, Sadok Jerad, Coralia Cartis  
**Category**: cs.LG  
**Published**: 2026-07-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.02194v1  

#### Abstract
Physics-informed neural networks (PINNs) have emerged as a promising route to solve partial differential equations, yet they have struggled to reach the precision of classical solvers. The obstacle is increasingly understood to be one of optimisation, owing to the severely ill-conditioned loss lands...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An Optimisation Framework for the Well-Conditioned Training of Physics-Informed Neural Networks

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **Physics-Informed Neural Networks (PINNs)** 在训练过程中普遍存在的**严重病态（ill-conditioning）损失景观**问题。这种病态导致传统优化器（如 Adam 或 BFGS）难以收敛到高精度解，限制了 PINNs 在科学计算领域与经典数值求解器竞争的能力。

尽管已有研究尝试通过二阶优化（如 Gauss-Newton、KFAC）缓解此问题，但仍受限于计算成本、内存消耗以及对超参数的敏感性。

### 提出了什么新方法或新思路
作者提出了一种名为 **DSGNAR (Doubly-Sketched Gauss-Newton with Adaptive Ratio)** 的新型二阶优化框架，其核心创新点如下：

- **双重压缩（Doubly-Sketched）Gauss-Newton 模型**：
  - 同时在**残差维度（行）** 和 **参数维度（列）** 上对 Jacobian 矩阵进行压缩。
  - 行压缩采用 **CountSketch**，允许对大量残差点进行密集采样而无内存压力。
  - 列压缩采用 **子采样随机余弦变换（SRCT）**，作为近等距嵌入以保持参数空间几何结构。
  - 最终得到一个小型方阵 $ \tilde{J} \in \mathbb{R}^{s \times s} $，可通过一次 SVD 高效获得候选步长。

- **基于自适应比率的目标下降控制（Adaptive Ratio Strategy）**：
  - 不直接手动调节正则化参数 $ \lambda $ 或信任域半径 $ \Delta $，而是通过目标**下降比率（ratio $ \rho $）** 来隐式控制。
  - 定义为实际目标函数减少量与模型预测减少量之比：
    $$
    \rho_k = \frac{\mathcal{L}(\theta_k) - \mathcal{L}(\theta_k + p_k)}{m_k(0) - m_k(p_k)}
    $$
  - 采用两阶段策略：
    1. **第一阶段（正则化递减）**：设定小目标比率（如 0.075），优先降低有效正则化 $ \lambda $，寻找良好条件区域。
    2. **第二阶段（目标快速下降）**：当 $ \lambda $ 达到最小值后，自动切换至大目标比率（如 0.5），追求目标函数的快速下降。

- **高效且鲁棒的实现机制**：
  - 支持**动态条件权重更新（UpdateWeights）**，平衡不同物理条件（PDE、边界、初值）的残差尺度。
  - 实现了**自动阶段切换（UpdateTargetRatio）**，基于 $ \lambda $ 的滑动窗口趋势判断是否进入第二阶段。

### 相比现有方法的优势
- **前所未有的精度**：可达到双精度浮点数极限（相对误差低至 $ 3 \times 10^{-16} $）。
- **显著加速**：相比当前最优方法，速度更快，尤其在单精度下可在数秒内完成高精度求解。
- **强健性**：对网络架构、算术精度、初始超参数不敏感。
- **消除了残差点选择难题**：得益于 CountSketch 对密集残差的天然聚合能力，无需精心设计采样策略。

---

## 2. 核心实验方法和设置

### 使用了哪些 PDE 问题（“数据集”）
论文在一系列具有挑战性的偏微分方程（PDE）上进行了验证，涵盖多种物理场景：

| PDE 问题 | 类型特点 |
|--------|--------|
| **Burgers' Equation** | 非线性，含激波层（sharp internal layer），标准基准 |
| **Kuramoto-Sivashinsky (KS) Equation** | 四阶，混沌动力学，长期演化困难 |
| **High-Dimensional Poisson (5D & 10D)** | 高维，超越传统网格法能力 |
| **Navier-Stokes (Lid-Driven Cavity)** | 耦合系统，多输出（速度、压力），约束复杂 |
| **Wave Equation** | 时间二阶，测试时间感知能力 |
| **KdV Equation** | 色散型 PDE，多时间步推进 |
| **Multi-scale Poisson** | 多尺度频率共存，需 Fourier 特征 |

### 实验设置和评估指标

#### 评估指标
- **相对 $ l_2 $ 误差（Relative $ l_2 $ Error）**：
  $$
  \epsilon_{\text{rel}} = \frac{\|u_{\text{pred}} - u_{\text{true}}\|_2}{\|u_{\text{true}}\|_2}
  $$
  在独立于训练点的均匀网格或密集样本上计算。
- **墙钟时间（Wall-clock Time, $ t_{\text{wall}} $）**：从开始到收敛的总耗时（秒）。
- **其他监控指标**：损失函数值、各条件权重变化、正则化 $ \lambda $、信任域半径 $ \Delta $、下降比率 $ \rho $。

#### 实验配置
- **硬件平台**：单块 NVIDIA H100 GPU。
- **软件框架**：JAX + XLA 编译，利用 JIT 加速。
- **精度模式**：同时报告 **single precision** 和 **double precision** 结果。
- **收敛标准**：信任域半径 $ \Delta < \Delta_{\min} $（$ 10^{-4} $ 单精，$ 10^{-8} $ 双精）。
- **Sketch Size $ s $**：通常设为参数数量 $ d_\theta $ 的 1/3 至 1/2。

### 基线方法对比
论文将 DSGNAR 与以下文献中的当前最优方法进行比较：

| 基线来源 | 方法类型 |
|---------|--------|
| `Kiy+25` | Adam + quasi-Newton 混合策略 |
| `Guz+25` | KFAC + 随机化 Gauss-Newton |
| `Chi+26` | Scale-PINN（特定优化策略） |
| `Dai+26` | TINNs（时间诱导网络） |
| `And+26` | FBPINNs（有限基 PINN） |

所有基线结果均引用原文报告的数据。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| PDE | 基线方法 ($ \epsilon_{\text{rel}} $) | **DSGNAR ($ \epsilon_{\text{rel}} $)** | 基线 ($ t_{\text{wall}} $) | **DSGNAR ($ t_{\text{wall}} $)** |
|-----|--------------------------|-------------------------------|----------------------|-----------------------------|
| **Burgers (双精)** | $ 1.62 \times 10^{-6} $ | **$ 7.97 \times 10^{-14} $** | 2,878 s | **346.1 s** |
| **Burgers (单精)** | $ 4.04 \times 10^{-5} $ | **$ 4.75 \times 10^{-7} $** | 179 s | **9.8 s** |
| **KS (双精)** | $ 6.51 \times 10^{-2} $ | **$ 5.12 \times 10^{-7} $** | 130,222 s | **5,226.6 s** |
| **10D Poisson (双精)** | $ \sim 1 \times 10^{-4} $ | **$ 5.96 \times 10^{-12} $** | ~8,000 s | **3,239.7 s** |
| **Navier-Stokes (双精)** | $ 1.43 \times 10^{-2} $ | **$ 1.13 \times 10^{-4} $** | ~90 s | **402.9 s** |
| **5D Poisson (双精)** | $ \sim 1 \times 10^{-3} $ | **$ 3.03 \times 10^{-16} $** | ~7,000 s | **406.7 s** |

> ✅ **关键突破**：在 **Burgers 方程**上，相对误差**提升五个数量级**；在 **10D Poisson** 上，提升**高达八个数量级**。

### 与基线方法的对比结果
- **精度全面碾压**：在所有测试问题上，DSGNAR 均大幅超越现有最优方法的精度，部分达到机器精度极限。
- **速度优势明显**：尽管某些任务绝对时间较长（如 KS），但考虑到其精度的巨大飞跃，效率极高。对于 Burgers 单精任务，仅用 **9.8 秒**即达到 $ 10^{-7} $ 精度。
- **单精度可用性**：证明了在许多应用中，**single precision PINNs** 已能提供足够精确的解，大幅降低计算成本。

### 消融实验结果
论文虽未单独设立“消融实验”章节，但通过多个角度验证了设计的有效性：

- **目标比率 $ \rho $ 的影响（Figure 4）**：
  - 小目标比率（如 0.075）能更彻底地降低正则化 $ \lambda $，最终获得更高精度解。
  - 两阶段策略结合了“探索”与“开发”，实现了精度与效率的最佳平衡。

- **不同架构下的表现（Solution 14）**：
  - 即使使用与主流不同的 **GaborNet** 架构，DSGNAR 依然能在 100 次迭代内将 $ \lambda $ 降至 $ 10^{-35} $，验证了其**架构无关性**。

- **Sketch Size 的权衡（Table 1, Navier-Stokes）**：
  - 使用较小的 sketch size（s=1000）时，误差为 $ 6.34 \times 10^{-4} $，耗时 80.4 秒。
  - 使用较大的 sketch size（s=4000）时，误差降至 $ 1.13 \times 10^{-4} $，耗时 402.9 秒。
  - 说明 **sketch size 是控制精度-成本权衡的关键参数**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **PINNs 的精度瓶颈在于优化过程的病态性**，而非表达能力本身。
2. **DSGNAR 通过双重压缩 Gauss-Newton 模型和自适应比率策略，有效克服了这一病态性**。
3. **该方法能够稳定地将 PINNs 的求解精度推向机器精度极限**，首次真正意义上接近甚至超越经典求解器的潜力。
4. **单精度 PINNs 在 DSGNAR 下也能快速获得高精度解**，打破了“必须使用双精度”的固有认知。
5. **方法具有极强的通用性和鲁棒性**，适用于非线性、混沌、多尺度、高维及耦合系统等多种复杂 PDE。

### 方法的局限性
- **理论保证尚待完善**：双重压缩（CountSketch + SRCT）的联合子空间嵌入性质尚未有严格的理论分析。
- **对某些耦合系统仍有挑战**：如 Navier-Stokes 问题中，由于物理约束间的内在冲突，最小正则化无法降得像其他问题那样低，表明瓶颈可能已转移到 PINN 公式的本质层面。
- **大规模 sketch 的 SVD 成本**：虽然 sketch size 可控，但当 $ s $ 很大时，SVD 仍可能成为瓶颈（尽管远优于存储完整矩阵）。

### 未来工作方向
- **理论分析**：为双重压缩 Gauss-Newton 提供严格的收敛性与误差界分析。
- **结合先进 PINN 公式**：将 DSGNAR 与 **FBPINNs**、**XPINNs** 等域分解方法结合，进一步扩展其处理更大规模、更长时间问题的能力。
- **集成到科学计算库**：将 DSGNAR 集成到如 **PETScML**、**Dedalus** 等科学机器学习库中，便于社区使用并与传统求解器直接对比。
- **探索更低精度训练**：研究在 FP16 或 BF16 下使用 DSGNAR 的可能性，进一步提升效率。

</details>

---

### 10. [SemHash-LLM: A Multi-Granularity Semantic Hashing Framework for Document Deduplication](https://arxiv.org/abs/2607.01601)

**Authors**: Xinyi Fang, Kejian Tong, Jiabei Liu, Tao Ning, Yuhang He  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.01601v1  

#### Abstract
Large scale document deduplication must preserve semantic equivalence while remaining efficient over massive corpora. We present SemHash LLM, a multi granularity framework that unifies semantic projection hashing, attention weighted MinHash, contrastive boundary learning, and selective LLM based adj...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SemHash-LLM: A Multi-Granularity Semantic Hashing Framework for Document Deduplication

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大规模文本语料库在预训练语言模型中广泛应用，但其中存在大量重复或近似重复的内容（如模板包裹、短文本扰动、内容包含关系等），严重影响模型训练效率与质量。传统的去重方法面临以下矛盾：
- **基于精确匹配的方法**（如 SimHash、MinHash）速度快，但在面对语义等价但字面不同的文本时表现脆弱（如 paraphrasing、template pollution）。
- **纯语义嵌入方法**虽然能捕捉语义相似性，但计算开销大，难以扩展到万亿级文档。

因此，本文旨在解决 **高效且鲁棒的大规模文档去重问题**，在保持高精度的同时实现可扩展性。

---

### 🚀 提出的新方法与创新思路

作者提出 **SemHash-LLM** —— 一种多粒度语义哈希框架，融合了语义学习与高效哈希技术，其核心创新包括：

#### （1）**Semantic Projection Hashing (SPH)**
- 在蒸馏后的 LLM embedding 空间中学习可训练的二值化哈希函数。
- 使用 contrastive 学习目标优化哈希位，使语义相近的文档产生相似的哈希码。
- 引入正交性（orthogonality）和比特平衡（bit balance）约束，提升编码效率。

#### （2）**Attention-Weighted MinHash (AW-MinHash)**
- 利用 LLM 中的 multi-head attention 分数作为 token 重要性信号。
- 结合 IDF 权重，构建加权 MinHash，抑制 boilerplate 内容（如导航栏、广告），增强对关键内容的敏感性。
- 自适应调整 LSH band 配置，依据 attention entropy 动态调节灵敏度。

#### （3）**Contrastive Boundary Learning + Uncertainty Estimation**
- 学习类型感知的动态决策边界（type-specific thresholds），而非固定阈值。
- 使用 Monte Carlo Dropout 进行不确定性估计，识别模糊候选对。

#### （4）**Selective LLM-as-Judge 机制**
- 对高不确定性的候选对，路由至 instruction-tuned LLM 进行精细化判断。
- 设计结构化 prompt，输出 structured judgment（duplicate/distinct/contained）并附带置信度。
- 最终通过 confidence-weighted ensemble 融合自动系统与 LLM 判断。

#### （5）**Multi-Granularity Fusion Network (MGFN)**
- 统一建模字符级（CNN）、词元级（BiLSTM）、语义级（LLM embedding）特征。
- 使用门控机制（gated fusion）自适应加权不同粒度特征，适配不同类型去重任务。

#### （6）**Cascaded Filtering Pipeline**
- 四阶段级联过滤流程显著降低需神经验证的样本量：
  1. Bloom Filter（精确去重初筛）
  2. Semantic Hash Blocking（粗粒度聚类）
  3. Attention-Weighted LSH（细粒度语义签名比对）
  4. Neural Verification（MGFN + LLM-refinement）
- 最终仅约 **0.7% 的候选对进入最终神经验证阶段**，确保整体效率。

---

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **准确性** | 显著优于传统哈希方法，在五种去重类型上均取得 SOTA 性能 |
| **效率** | 级联架构将昂贵的神经计算控制在 <1% 的候选对内，适合超大规模部署 |
| **鲁棒性** | 支持多种复杂场景：模板污染、短文本扰动、containment、viral fragments |
| **灵活性** | 可动态适应不同类型文档的最优阈值，避免“一刀切”策略 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **RedPajama** 中的 **100GB web content** 构建测试语料。
- 包含多样化的网页结构：长文档、短片段、boilerplate 密集页面、嵌套包含关系等。
- 模拟真实开放预训练语料的异构性和噪声特性。

---

### 🎯 评估指标

定义了五个去重类别及其对应指标：

| 类型 | 描述 | 指标 |
|------|------|------|
| **Type A** | Template-wrapped near-duplicates | Deletion Recall $ S_t $ |
| **Type B** | Hot-spot documents with skew | Deletion Recall |
| **Type C** | Short texts with char perturbations | Deletion Recall |
| **Type D** | Parent-child containment hierarchies | Containment Accuracy $ S_p $ |
| **Type E** | High-frequency viral fragments | Fragment Removal Rate $ S_E $ |

综合得分公式为加权平均：

$$
\text{Score}_{\text{final}} = 100 \times \sum_{t \in \{A,B,C,D,E\}} w_t \cdot S_t
$$

权重：$ w_A=0.25, w_B=0.15, w_C=0.20, w_D=0.20, w_E=0.20 $

> ⚠️ 若非重复文档误删率超过 10%，则强制 $\text{Score}_{\text{final}} = 0$（circuit breaker）

---

### 🔁 基线方法对比

选取代表性去重方法进行比较：

| 方法 | 类型 |
|------|------|
| **SimHash-64** | 基于局部敏感哈希的传统方法 |
| **MinHash-LSH** | 基于 n-gram 的集合相似性方法 |
| **NearDup-BERT** | 使用 BERT 嵌入进行近似去重 |
| **DedupLM** | 基于语言模型的去重系统 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table I）

| Method | SA | SB | SC | SD | SE | Final Score |
|--------|----|----|----|----|----|-------------|
| SimHash-64 | 0.71 | 0.65 | 0.42 | 0.31 | 0.18 | 48.20 |
| MinHash-LSH | 0.82 | 0.78 | 0.68 | 0.45 | 0.22 | 61.35 |
| NearDup-BERT | 0.85 | 0.79 | 0.74 | 0.68 | 0.55 | 73.45 |
| DedupLM | 0.88 | 0.83 | 0.81 | 0.76 | 0.71 | 81.20 |
| **SemHash-LLM** | **0.94** | **0.91** | **0.89** | **0.92** | **0.88** | **91.05** |

✅ **SemHash-LLM 在所有单项和总分上均达到 SOTA 表现**，综合得分领先 DedupLM 接近 10 个百分点。

---

### 🔍 消融实验结果（Ablation Study）

| 变体 | Final Score | 分析 |
|------|------------|------|
| w/o SPH | 79.32 | 移除语义投影哈希导致严重下降，说明其对语义保留至关重要 |
| w/o AW-MinHash | 85.18 | 缺少注意力加权后，对 boilerplate 更敏感，性能下降明显 |
| w/o MGFN | 84.56 | 多粒度融合网络提升了整体表征能力，尤其利于复杂模式识别 |

> 结果表明各模块协同作用，缺一不可。

---

### 💡 效率成果
- **<1% 的候选对需要神经验证**（实际约 0.72%）
- LLM-as-Judge 仅处理约 **3% 的候选对**，大幅节省推理成本
- 级联流水线支持 trillion-scale 文档处理

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **多粒度融合是应对多样化去重挑战的关键**  
   不同类型的重复需要不同层次的分析（字符、token、semantic），统一建模显著提升鲁棒性。

2. **LLM 的中间表示（如 attention map）可用于轻量级语义增强**  
   即使不直接调用 LLM 推理，也能从中提取 valuable signals（如重要性权重）用于传统算法改进。

3. **选择性 LLM 判决可行且高效**  
   将 LLM 作为“仲裁者”仅用于高不确定性案例，可在极低成本下获得人类级别的判断能力。

4. **级联过滤 + 不确定性路由 是大规模系统的理想范式**  
   实现了 accuracy-efficiency trade-off 的突破，兼顾精度与可扩展性。

---

### ⚠️ 方法的局限性

1. **依赖高质量的 LLM 蒸馏学生模型**  
   若 student encoder 无法有效模仿 teacher 的语义空间，会影响 SPH 效果。

2. **LLM-as-Judge 成本仍受限于 API 调用延迟与费用**  
   虽然只处理 3% 的样本，但在极端高吞吐场景下可能成为瓶颈。

3. **对非常规语言或低资源语言支持有限**  
   当前实验基于英文 web 文本，未验证跨语言泛化能力。

4. **suffix array-based fragment detection 可能耗费内存**  
   对超长公共子串检测可能带来额外存储压力。

---

### 🔮 未来工作方向

1. **探索更高效的 LLM 蒸馏与量化策略**  
   如结合 mixed-precision quantization 或 sparse activation，进一步压缩 student model。

2. **引入 multi-agent LLM 协作机制**  
   参考相关研究 [12][13]，设计 specialized agents 分别处理不同类型去重任务。

3. **动态检索与工具调用集成**  
   借鉴 DynaRAG 思路，在证据不足时主动查询外部知识库辅助判断。

4. **扩展至多模态去重场景**  
   将该框架推广至图文、音视频等 multimodal content 的冗余消除。

---

## 总结

📌 **SemHash-LLM 是一个面向大规模文档去重的先进框架**，它成功地将 LLM 的强大语义理解能力与经典高效哈希技术相结合，提出了 **“语义感知 + 级联过滤 + 不确定性驱动 LLM 精修”** 的新范式。

🎯 其在 RedPajama 上实现了 **91.05 的综合得分**，远超现有方法，并将神经验证比例控制在 **<1%**，具备出色的实用价值和系统扩展潜力。

🔧 该工作不仅推动了 document deduplication 技术的发展，也为如何在大规模系统中 **经济有效地利用 LLM** 提供了重要参考路径。

</details>

---

### 11. [Beyond Skepticism: Evaluating LLMs Pedagogical Intent Reasoning with the Adaptive Pedagogical Vigilance Framework](https://arxiv.org/abs/2607.01581)

**Authors**: Minghao Chen, Ruihan Zhou, Jiayi Tang, Zihan Xu, Bowen Huang, Yuxin Liu  
**Category**: cs.CL  
**Published**: 2026-07-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.01581v1  

#### Abstract
The capacity of Large Language Models (LLMs) to reason about pedagogical intent within instructional communication remains underexplored, particularly in educational domains such as translation pedagogy. To address this, we propose the \textbf{Adaptive Pedagogical Vigilance (APV)} framework, a novel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Skepticism: Evaluating LLMs' Pedagogical Intent Reasoning with the Adaptive Pedagogical Vigilance Framework

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Large Language Models (LLMs)** 在处理教学性沟通时，缺乏对**教学意图 (pedagogical intent)** 的推理能力。它们通常表现出两种极端行为：
- **过度信任 (over-trust)**：盲目接受所有输入为真实或有益；
- **过度怀疑 (over-skepticism)**：对所有信息都持怀疑态度，错失学习机会。

这种“非黑即白”的社会认知模式限制了 LLMs 在教育场景中的可靠性。本文旨在解决这一问题，提出一种更精细、自适应的**教学性警觉机制**。

---

### 🚀 提出的新方法：Adaptive Pedagogical Vigilance (APV) 框架

#### 核心思想
将“vigilance”从单纯的“社会怀疑”重新定义为一种**基于贝叶斯推理的自适应学习优化机制**，通过推断教师的教学意图来动态调整信念更新。

#### 创新组件
- **Pedagogical Intent Inference Engine (PIIE)**：一个形式化的贝叶斯模型，用于建模教师如何选择内容以最大化教学效用，并让学习者反向推理其潜在意图。
- **三层评估体系 (Three-tier evaluation hierarchy)**：
  1. 区分**教学类型 (instructional genre)**：刻意教学 vs. 偶然暴露
  2. 推理**结构化教学配置 (structured pedagogical setup)**：包括教师立场 (stance) 和激励 (incentives)
  3. 泛化到**真实的教学话语 (authentic educational discourse)**

#### 相比现有方法的优势
| 维度 | 传统方法 | APV 框架 |
|------|--------|---------|
| 警觉建模 | 静态怀疑或盲信 | 动态、情境依赖的信念调节 |
| 形式化程度 | 缺乏统一计算框架 | 明确的贝叶斯生成与推理模型 |
| 可解释性 | 黑箱判断 | 支持对 `genre`, `stance`, `incentives` 的显式归因 |
| 生态有效性 | 多在人工任务中测试 | 成功迁移到真实在线教学语料 |

> 🔍 **优势总结**：APV 不仅提升了 LLM 对教学动机的理解能力，还提供了一个可量化、可推广、与人类判断高度一致的评估范式。

---

## 2. 核心实验方法和设置

### 📚 数据集与实验设计

#### 实验层级与对应设置

| 层级 | 场景描述 | 数据来源 |
|------|--------|----------|
| **Level 1**: Genre Discrimination | 将经典“蓝圈/黄圈”任务重构为**翻译纠错任务**，区分教师是主动纠正错误（Deliberate Pedagogy）还是无意泄露答案（Incidental Exposure） | 合成数据集，模拟合作/竞争课堂环境 |
| **Level 2**: Structured Configuration Reasoning | 引入四位具有不同 `stance` 和 `RT` 的虚拟导师角色（如应试教练、友好伙伴），要求 LLM 推理其推荐背后的动机 | 基于 Oktar et al. [2024] 改编的多角色提示数据集 |
| **Level 3**: Generalization to Real Discourse | 构建真实世界语言教学语料库，包含在线课程片段、教师反馈视频、论坛讨论等 | 自建自然语言教学文本数据集（transcribed segments） |

---

### 📊 评估指标

| 指标 | 定义 |
|------|------|
| **Discrimination Gap** | 在 Pedagogy vs. Exposure 条件下信念更新的差异（越大越好） |
| **Pearson’s r** | LLM 输出与贝叶斯理性模型预测、人类评分之间的相关系数 |
| **△BLEU / TER Prediction Accuracy** | 对学生翻译改进程度的预测与专家标注的相关性 |
| **Robustness across prompts** | 在不同提示方式（direct/CoT）、视角（first-person/user）下的稳定性表现 |

---

### 🆚 基线方法对比

参与比较的主流 LLMs 包括：
- GPT-4o
- Claude 3.5 Sonnet
- Gemini 2.0 Flash
- Llama 3.3-70B
- DeepSeek-R1
- Gemma 等多个开源模型

所有模型均在以下条件下测试：
- Direct prompting
- Chain-of-Thought (CoT) prompting
- 第一人称 vs 用户视角
- 默认提示 vs 引导式提示（steering prompt）

> 💡 APV 方法作为增强型提示策略嵌入上述模型中进行对比。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ Level 1: Genre Discrimination
- 所有模型都能区分教学与暴露，但 **APV 表现最优**
- 在竞争环境下，APV-CoT 的 pedagogy 条件下信心变化最小（仅 0.28），表明其具备最强的情境化怀疑能力
- **APV 实现最大区分度**（见 Figure 4），尤其在 CoT 设置下显著优于基线

#### ✅ Level 2: Configuration Reasoning（核心突破）
| 模型 | LLM-Human Correlation (r) |
|------|----------------------------|
| GPT-4o | 0.943 |
| Claude 3.5 | 0.941 |
| **APV (Ours)** | **0.958** ⬆️ |

- APV 与人类判断的相关性达到 **r = 0.958**，远超其他模型
- 与贝叶斯模型预测的相关性也最高 (**r = 0.937**)，说明其内部推理更符合理性标准
- 在 `incentive` 和 `stance` 两个维度上均实现平衡且优越的表现（Table 4）

#### ✅ Level 3: Naturalistic Generalization
| 设置 | GPT-4o (Steering) | APV (Default) |
|------|------------------|---------------|
| Best Correlation | ~0.312 | **0.345** ✅ |
| △BLEU 预测准确率 (vs. human) | 0.22 | **0.41** ✅ |

- 即使不使用额外引导提示，APV 在真实教学语境中仍保持稳健性能
- 基线模型在默认提示下几乎失效（相关性接近零），需靠 steering prompt 回升
- APV 在包含明确纠正、规则讲解等内容中表现尤为出色（r 达 0.41）

---

### 🔍 消融实验结果（Ablation Study）

| 配置 | Human Correlation (r) | 性能下降幅度 |
|------|------------------------|-------------|
| Full APV | 0.958 | — |
| w/o PIIE Structure | 0.783 | ↓18.3% ❗ |
| w/o Incentives (RT) | 0.852 | ↓11.1% |
| w/o Stance (T) | 0.874 | ↓8.8% |
| w/o Genre (G) | 0.891 | ↓7.0% |

> 🔎 **发现**：
> - **PIIE 结构本身贡献最大（40.5%）**，证明贝叶斯形式化不是“prompt engineering 技巧”，而是真正的归纳偏置
> - 教师激励 (`RT`) 是最关键的内容参数
> - 教学类型 (`G`) 的影响相对较小，可能部分可通过上下文推断

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLMs 具备基本的动机敏感性**，但在复杂教学情境中容易失调。
2. **APV 框架显著提升 LLM 的 pedagogical intent reasoning 能力**，使其能够：
   - 准确区分教学与非教学信息
   - 根据教师立场和激励动态调整信任权重
   - 在真实教学语料中做出更合理的信念更新
3. **APV 与人类判断高度一致 (r = 0.958)**，验证了其心理现实性和生态有效性。
4. **贝叶斯形式化（PIIE）提供了强大而稳定的推理骨架**，优于简单提示工程。

---

### ⚠️ 局限性

1. **语言范围有限**：目前实验集中在英语教学语境，跨语言泛化尚未验证。
2. **文化偏差风险**：不同教育文化中师生互动规范不同，APV 是否普适有待检验。
3. **单向交互假设**：当前框架假设为单轮教师→学生通信，未考虑多轮对话或多主体协作学习。
4. **数据规模受限**：Level 3 的真实语料虽具生态效度，但仍属小规模样本。

---

### 🔮 未来工作方向

1. **扩展至多轮教学对话系统**，研究 vigilance 的持续建模与更新机制。
2. **集成进智能辅导系统 (ITS)**，构建能动态调整教学策略的 AI 导师。
3. **探索跨文化教学 vigilance 差异**，结合跨文化心理学设计更具包容性的框架。
4. **应用于 content moderation**，识别伪装成教学内容的操纵性或误导性信息。
5. **结合神经符号方法**，将 PIIE 与内部激活模式关联，实现可解释性增强。

---

## ✅ 总结一句话

> 本论文提出的 **Adaptive Pedagogical Vigilance (APV)** 框架首次将教学意图推理形式化为一个可计算、可评估的贝叶斯过程，显著提升了 LLMs 在教育场景中的社会认知能力，实现了与人类判断的高度对齐，在理论深度与应用潜力上均取得重要进展。

</details>

---

### 12. [PARTREP: Learning What to Repeat for Decoder-only LLMs](https://arxiv.org/abs/2607.01792)

**Authors**: Andikawati P Widjaja, Yongjun Kim, Hyounghun Kim, Jaeho Lee  
**Category**: cs.CL  
**Published**: 2026-07-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.01792v1  

#### Abstract
While decoder-only LLMs excel at a vast array of natural language tasks, it suffers from an asymmetric information flow induced by causal attention: later tokens are richer in contextual grounding than earlier ones. A simple and effective remedy is prompt repetition -- just appending a second copy o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PARTREP: Learning What to Repeat for Decoder-only LLMs**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Decoder-only LLMs 由于 **causal attention** 的机制，导致信息流不对称：序列中靠后的 token 拥有更丰富的上下文信息，而靠前的 token 容易“丢失在中间”（lost in the middle），影响模型推理能力。虽然 **prompt repetition**（将提示重复一次）能有效缓解该问题，但它会：
- 将输入长度翻倍（2L）
- KV cache 占用翻倍
- Prefill 阶段的 attention FLOPs 变为原来的四倍（O(4L²)）

这使得全量重复在长上下文场景下计算成本过高，难以实用。

---

### **提出的新方法：PARTREP**
PARTREP 是一种**选择性提示增强方法**（selective prompt augmentation），其核心思想是：
> **只重复最具有信息量的 token，而非整个 prompt。**

#### **核心创新点：**
1. **基于 token-wise NLL 的重要性评分**
   - 使用 **negative log-likelihood (NLL)** 作为 token 信息密度的代理信号。
   - 高 NLL 的 token 表示模型难以从上下文中预测，因此更可能携带独特信息，重复这些 token 能带来最大收益。

2. **轻量级门控机制（lightweight gate）实现高效选择**
   - 为了避免对每个 token 进行完整前向传播来计算 NLL，作者训练一个**轻量级 gating module**。
   - 该模块基于早期 transformer 层的 hidden states 预测哪些 token 具有高 NLL，从而实现 **early exit** 下的选择，显著降低开销。

3. **端到端推理流程优化**
   - 在 prefill 阶段中途暂停，提取 hidden states → 通过 gate 选出 top-T% 高 NLL token → 拼接至原 prompt 后 → 继续完成生成。

---

### **相比现有方法的优势**
| 方法 | KV Cache | Prefill FLOPs | 是否任务通用 | 是否需训练 |
|------|----------|---------------|----------------|------------|
| Full Repetition | 2× | 4× | ✅ | ❌ |
| LLMLingua（压缩） | ↓ | ↓ | ✅ | ✅ |
| Echo/H2O Eviction | ↓ | 4×（仍需全 prefilled） | ✅ | ✅ |
| **PARTREP (ours)** | **59.4% of full rep.** | **79.0% of full rep.** | ✅ | ✅ |

✅ **优势总结：**
- 保留了 full repetition 的大部分准确率增益；
- 显著降低 KV cache 和 prefill 计算开销；
- 支持任意开放域任务（非仅限于多选题）；
- 设计上可扩展至不同模型架构。

---

## **2. 核心实验方法和设置**

### **使用的数据集（共8个）**
| 类型 | 数据集 |
|------|--------|
| **知识检索类** | MMLU, ARC-Challenge, OpenBookQA, MedQA, SciQ |
| **复杂推理类** | GSM8K, MMLU-Pro |
| **长上下文能力测试** | RULER |

> 所有测试均使用完整测试集，RULER 使用 1300 个子样本（每任务100题）。

---

### **实验设置**
- **模型家族**（instruction-tuned）：
  - Qwen 2.5-3B
  - Llama 3.2-3B
  - Gemma 4-E4B（含滑动窗口 + 全局 attention 混合机制）
- **评估指标**：
  - **Accuracy**：任务准确率
  - **KV cache usage**：存储的 prompt token 数量（layer-wise）
  - **Prefill FLOPs**：prefill 阶段总浮点运算量（含选择开销）

- **重复比例控制**：设定阈值 $ T \in (0,1) $ 控制重复 token 比例（默认 T=0.15）

---

### **基线方法对比**
| 类别 | 方法 | 描述 |
|------|------|------|
| **无重复** | No Repetition | 原始 prompt |
| **摘要式增强** | + Naive Summary<br>+ LLMLingua | 添加模型自产摘要 / 使用 LLMLingua 压缩 |
| **全量重复** | Full Repetition | 整个 prompt 复读一遍 |
| **基于 eviction 的优化** | + Echo Eviction<br>+ H2O Eviction<br>+ LLMLingua Comp. | 全重复后删除部分 KV 或压缩输入 |
| **本文方法** | **PARTREP (ours)** | 仅重复高 NLL token 子集 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Qwen 2.5-3B 上平均表现）**

| 方法 | Avg Accuracy | KV Cache (vs Full Rep.) | Prefill FLOPs (vs Full Rep.) |
|------|--------------|--------------------------|-------------------------------|
| No Repetition | 66.9% | 235.5 | 1.549T |
| Full Repetition | **68.5%** | 471.1 (**2×**) | 3.140T (**4×**) |
| **PARTREP (T=0.15)** | **68.3%** | **280.4** (**59.4%**) | **2.481T (79.0%)** |

> ✅ **结论**：PARTREP 达到了与 full repetition 几乎相同的准确率提升（+1.4pp vs +1.6pp），但仅消耗约 **60% 的 KV cache 和 79% 的 prefill FLOPs**。

---

### **与其他方法对比（Table 1 & 2）**
- **优于所有摘要类方法**（如 LLMLingua、Naive Summary）：
  - 因为它们替换原始上下文，破坏了结构完整性。
- **优于各种 eviction 方法**：
  - 如 H2O/Echo Eviction 虽节省内存，但仍需执行全量 prefill（FLOPs 不降）。
- **在多个模型上一致有效**（Table 2）：
  - 在 **MMLU** 上，PARTREP 在 Qwen、Llama、Gemma 上分别达到 **63.2**, **54.3**, **75.8**，全面超越 full repetition。
- **在长上下文场景下表现最优**（Table 3, RULER）：
  - 在 16k context 下，PARTREP 以 **73.2%** 超过 full repetition（72.3%）和 LLMLingua（63.5%），说明其对“长文本冗余”更具鲁棒性。

---

### **消融实验结果**

#### **(1) 门控模块设计（Table 4）**
| 架构 | Accuracy | Gate FLOPs |
|------|---------|-----------|
| Linear | 80.7% | 270.6k |
| **2-layer MLP (ours)** | **81.4%** | **549.0k** |
| Transformer-based | 81.3% | 654.3k |

> ✅ 2-layer MLP 在精度与效率之间取得最佳平衡。

#### **(2) Token Windowing（Table 5）**
| 设置 | Accuracy | KV Cache |
|------|---------|---------|
| No windowing | 61.5 | 90.1 |
| +1 word | 64.6 | 94.5 |
| ±1 neighbor words | **66.0** | **110.2** |

> ✅ 局部上下文扩展显著提升准确率，尤其适用于 subword 分词系统。

#### **(3) 其他关键组件分析（Appendix B）**
- **NLL 是最优评分标准**（Fig 4）：优于 attention、TF-IDF、随机等。
- **最佳 early-exit 层为第18层**（Fig 5）：太浅语义不足，太深延迟高。
- **最优重复预算 T ≈ 15%**（Fig 6）：过多重复反而稀释关键信息。
- **连接提示词推荐使用自然语言指令**（Fig 7）：
  > `\nPay attention to these key tokens:\n` 效果最好。

---

## **4. 关键结论和发现**

### **主要发现**
1. **高 NLL token 是最值得重复的信息单元**：
   - 它们代表上下文中“最难预测”的部分，即信息密度最高。
2. **选择性重复 ≫ 全量重复 ≫ 摘要压缩**：
   - 保持原始 prompt 完整 + 增强关键信息，优于直接替换或删减。
3. **PARTREP 实现了精度与效率的帕累托前沿突破**：
   - 在仅增加少量计算的前提下，逼近 full repetition 的性能上限。

---

### **局限性**
1. **需要离线训练 gate 模块**：
   - 无法零样本应用，且不跨模型迁移（需为每个 target LLM 单独训练）。
2. **不适用于非文本模态**：
   - 如图像 token 重复拼接可能导致语义扭曲，难以推广到 VLM。
3. **依赖 subword tokenization 的局部完整性**：
   - 需借助 token windowing 缓解分词碎片化问题。

---

### **未来工作方向**
- 探索 **zero-shot 或 prompt-driven gate**，避免额外训练。
- 将 PARTREP 思想拓展至 **multi-modal setting**，设计视觉 token 重要性度量。
- 结合 **dynamic repetition budgeting**，根据 prompt 内容自适应调整 T。
- 与 **speculative decoding** 或 **caching mechanisms** 进一步协同优化推理效率。

---

> 📌 **一句话总结**：  
> **PARTREP 提出了一种高效的选择性 prompt 重复机制，通过轻量级 gate 识别并复述高 NLL token，在几乎不损失 full repetition 性能的同时，大幅降低 KV cache 与 prefill 开销，为长上下文 LLM 推理提供了实用解决方案。**

</details>

---

### 13. [Arachne: Orchestrating Cascades for Efficient Text-to-Video Model Training](https://arxiv.org/abs/2607.01701)

**Authors**: Peng Yu, Yuankai Fan, Yang Qiu, Tian Li, Bihuan Chen, Yin Chen, Qizhen Weng  
**Category**: cs.DC  
**Published**: 2026-07-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.01701v1  

#### Abstract
The rising demand for AI-generated videos is fueled by advances in large-scale Text-to-Video (T2V) models, trained on extensive datasets of video clips spanning diverse resolutions and durations. To address this data heterogeneity, current training methods often use a bucketing strategy that groups ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Arachne: Orchestrating Cascades for Efficient Text-to-Video Model Training — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大规模 **Text-to-Video (T2V)** 模型训练面临以下挑战：
- **数据异构性**：视频样本在分辨率、时长（frame 数）上差异巨大，导致批量处理困难。
- **静态并行策略的局限性**：传统采用的 **Data Parallelism (DP)** 和 **Sequence Parallelism (SP)** 是静态配置，无法适应不同长度序列带来的计算负载不均。
- **严重的资源浪费**：由于最长序列决定整体迭代时间（makespan），短序列设备大量空闲（idle time），造成 **GPU 利用率低下**。

这些问题在 T2V 中尤为突出，因为其序列远长于 LLM，且 DiT 架构具有 $O(L^2)$ 复杂度，**计算是主要瓶颈**，而非通信。

---

### 🚀 提出的新方法：Arachne
Arachne 是一种专为高效 T2V 模型训练设计的新型分布式训练框架，其核心思想是 **细粒度时空协同调度（fine-grained spatio-temporal orchestration）**。

#### 主要创新点：
1. **引入“级联”（Cascade）作为最小调度单元**
   - 将每个样本的训练过程分解为多个自包含的 **cascade**，每个 cascade 定义了输入数据、模块（如 VAE/DiT）、并行策略、GPU 分配和启动时间等。
   - 实现对计算任务的精细化控制。

2. **三阶段协同优化架构**
   - **① Cascade-level Parallelism Planner**  
     将训练迭代建模为 **DAG（有向无环图）**，通过混合整数线性规划（MILP）或遗传算法联合优化每个 cascade 的启动时间和并行度（SP degree），以最小化理论 makespan。
   - **② Topology-aware Resource Mapper**  
     考虑物理网络拓扑（如 NVLink、InfiniBand），进行两级空间映射：
     - **Intra-cascade**：优先将单个 cascade 放置在同一节点内，减少跨节点通信。
     - **Inter-cascade**：使依赖 cascade 共享 GPU，实现高效的 in-place 数据传递。
   - **③ Runtime Executor**  
     执行最终的时空计划，协调 cascade 间的数据传递，并提出 **异构梯度聚合机制（heterogeneous gradient accumulation）**，解决动态调度下梯度组合复杂的问题。

3. **打破静态时空绑定**
   - 动态分配 GPU 给不同的 cascade，避免固定分组造成的资源僵化。
   - 时间上错峰执行 cascade，填补空闲周期，提升整体利用率。

---

### 🔍 相比现有方法的优势
| 方面 | 传统方法（如 Megatron-LM, DeepSpeed） | Arachne |
|------|-------------------------------|--------|
| 并行策略 | 静态 DP + SP，全局统一配置 | 动态、细粒度、按 cascade 定制 |
| 资源利用 | 严重不平衡，长序列拖慢整体 | 显著降低 idle time，提高吞吐 |
| 通信效率 | 忽略物理拓扑 | 显式优化 placement 减少跨节点流量 |
| 可扩展性 | 随规模增大性能增益下降 | 性能优势随模型/数据/集群规模扩大而增强 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
基于 **Open-Sora** 的三阶段渐进式课程学习（curriculum learning）：
- **Stage 1**: WebVid (~7M clips, 360p)
- **Stage 2**: Koala (~20M clips, 720p)
- **Stage 3**: 内部数据集 Lynx (~10M clips, 1080p)

这些数据集体现了真实的 **高异构性**：帧数从几十到上百不等，分布呈右偏、集中、离散特点（见 Observation 1）。

---

### ⚙️ 实验设置
- **硬件平台**：64-GPU NVIDIA H100 集群（8卡/节点，NVLink + 400Gbps InfiniBand）
- **测试规模**：主结果使用 16 GPUs；可扩展性测试扩展至 32 和 64 GPUs
- **模型**：三种主流 T2V 模型
  | 模型 | 参数量 | 特点 |
  |------|-------|------|
  | Wan2.1 | 1.3B | 标准 self/cross-attention |
  | CogVideoX | 5B | 3D full attention |
  | HunyuanVideo | 13B | hybrid dual/single-stream |

- **评估指标**：
  - **平均迭代时间（Average Iteration Time）**
  - **GPU 空闲率（Idle Ratio）**
  - **吞吐量加速比（Speedup / Throughput Improvement）**
  - **消融实验中的性能退化分析**

- **控制变量**：
  - 所有系统使用相同 bucket 配置、自适应批大小、激活检查点（activation checkpointing）
  - 最大帧数限制为 129，防止 OOM

---

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **Megatron-LM** | 强静态基线：DP + ZeRO-1 + Context Parallelism (CP)，手动调优 |
| **DeepSpeed** | 强静态基线：ZeRO + Ulysses-style Sequence Parallelism (SP) |
| **FlexSP** | 动态基线：针对变长文本优化的 LLM 训练框架，每轮动态调整 SP 度 |

> Arachne 在每一轮训练中动态生成最优的 cascade 执行计划，而基线均为静态或粗粒度动态。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 HunyuanVideo-13B 为例）

#### （1）端到端迭代时间对比（Fig. 6）
| 方法 | Stage 1 (360p) | Stage 2 (720p) | Stage 3 (1080p) |
|------|----------------|----------------|------------------|
| Megatron-LM | 50.0s | 40.0s | 38.8s |
| DeepSpeed | 47.0s | 38.8s | 37.6s |
| FlexSP | 45.8s | 34.6s | 32.8s |
| **Arachne** | **22.07s** | **32.96s** | **26.80s** |

✅ **最大加速比达 65%**（vs Megatron-LM），在高分辨率阶段优势更明显。

---

#### （2）GPU 空闲率对比（Fig. 7）
| 方法 | Stage 1 → Stage 3 空闲率变化 |
|------|----------------------------|
| Megatron-LM | 12.1% → 30.1% ↑ |
| DeepSpeed | 9.0% → 26.8% ↑ |
| FlexSP | 8.1% → 33.6% ↑ |
| **Arachne** | **8.1% → 8.1%** ✅（稳定低位） |

➡️ 表明 Arachne 成功将数据异构性转化为调度机会，而非负担。

---

#### （3）可扩展性表现（Fig. 8）
- **模型尺寸扩展**（1.3B → 13B）：
  - 对 Megatron-LM 加速比从 35% 提升至 **43%**
- **工作负载异构性增加**（max frames: 53 → 105）：
  - 加速比从 21% 提升至 **35%**
- **集群规模扩展**（16 → 64 GPUs）：
  - 保持 **30%-40%** 的稳定加速优势

✅ 展现出明显的“**scaling law**”行为：**训练规模越大，Arachne 的优势越强**。

---

#### （4）消融实验（Table III）
验证各组件有效性（以 HunyuanVideo-13B Stage 3 为例）：

| 方法 | 迭代时间 | 相对退化 |
|------|---------|----------|
| Arachne（完整） | 26.80s | — |
| w/o Inter-cascade | 27.78s | +3.64% |
| w/o Intra-cascade | 30.41s | +13.67% |
| w/o Mapper（完全移除） | 31.89s | +19.26% |

📌 结论：
- **Topology-aware resource mapper 至关重要**
- **Intra-cascade placement 是主导因素**，说明内部通信优化影响最大

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **T2V 训练瓶颈本质是计算密集型而非通信密集型**，因此 LLM 领域的通信优化方法（如 FlexSP）效果有限。
2. **数据异构性不应被规避，而应被利用**：Arachne 将其转化为丰富的调度空间，通过细粒度 cascade 控制实现负载均衡。
3. **时空解耦调度显著提升资源利用率**：GPU idle time 下降超过 70%，迭代时间最多减少 **65%**。
4. **性能增益具有正向扩展性**：随着模型更大、数据更多样、集群更庞大，Arachne 的优势持续放大，符合实际生产需求。

---

### ⚠️ 方法的局限性
- **调度开销存在**：虽然 planner 使用 GA 实现高效求解，但在超大规模下仍需权衡优化精度与运行时开销。
- **依赖准确的成本模型**：Latency 和 Memory 的预测准确性直接影响调度质量，尤其对于新架构可能需要重新校准。
- **目前聚焦于 DiT 类 T2V 模型**：是否适用于其他生成范式（如 AR-based）尚待验证。

---

### 🔮 未来工作方向
- 开发更轻量化的在线调度器，支持实时反馈调节。
- 探索 cascade 粒度的自动搜索机制。
- 将 Arachne 范式推广至其他多模态生成任务（如 Text-to-3D、Video Editing）。
- 支持异构硬件（如 H100 + A100 混合集群）下的 cascade 分配。

---

## 🔚 总结
Arachne 提出了一种面向 **T2V 模型特性的全新训练范式**，通过 **cascade 分解 + 时空协同调度**，有效解决了由数据异构性和计算密集性引发的资源浪费问题。实验证明其在多种模型、数据集和集群规模下均取得显著性能提升，且具备良好的可扩展性，是迈向下一代高效生成模型训练基础设施的重要一步。

> 🔗 **开源承诺**：作者表示将开源 Arachne 框架及评测 workload，促进社区复现与进一步研究。

</details>

---

### 14. [WattGPU: Predicting Inference Power and Latency on Unseen GPUs and LLMs](https://arxiv.org/abs/2607.02391)

**Authors**: Mauricio Fadel Argerich, Jonathan F\"urst, Marta Pati\~no-Mart\'inez  
**Category**: cs.DC  
**Published**: 2026-07-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.02391v1  

#### Abstract
Large Language Model (LLM) inference workloads are a rapidly growing contributor to data center energy consumption. Optimizing these deployments requires matching specific LLMs to the most efficient GPUs, but operators currently lack the tools to do so without exhaustively profiling each combination...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**WattGPU: Predicting Inference Power and Latency on Unseen GPUs and LLMs**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前在部署 **Large Language Models (LLMs)** 时，为了实现能效最优，需要为特定模型选择最合适的 GPU。然而，这一过程通常依赖于对每个 (LLM, GPU) 组合进行实际的硬件测试（profiling），成本高昂且难以扩展，尤其对中小规模部署者不友好。

此外，现有的预测模型大多：
- 需要训练阶段的 profiling 数据；
- 无法泛化到未见过的 GPU 或 LLM；
- 仅适用于少数高端 GPU（如 H100）。

因此，缺乏一种**无需访问硬件、无需实际测试、可泛化至未见设备和模型**的推理功耗与延迟预测工具。

---

### 🚀 提出的新方法与创新
作者提出了 **WattGPU**，一个基于公开元数据的双任务预测框架，用于估计 LLM 在 GPU 上推理时的两个关键指标：

- **Mean GPU Power Draw (P)**：平均 GPU 功耗（单位：瓦特）
- **Inter-Token Latency (ITL)**：生成连续 token 之间的平均时间（单位：毫秒）

#### 创新点包括：
1. **首次在“未见 GPU”和“未见 LLM”上验证的预测模型**  
   模型训练后可直接应用于从未参与训练的 GPU 和 LLM，仅需其公开规格（如 TDP、memory bandwidth）和 LLM 架构参数（如参数量、层数等）。

2. **完全免 profiling（zero hardware access）**  
   不依赖任何运行时测量或专用基准测试，仅使用厂商公布的 GPU specs 和 Hugging Face 提供的 LLM metadata。

3. **引入三个关键衍生特征增强建模能力**：
   - **Boosting Ratio**：反映频率动态缩放对功耗的影响；
   - **Bandwidth Latency**：读取全部权重所需时间，衡量内存瓶颈；
   - **Compute Latency (Clat)**：理论最小计算延迟，基于 FLOPS 和参数量估算。

4. **开源代码与数据集**  
   所有模型、训练流程及评估脚本均已开源：[https://github.com/maufadel/wattgpu](https://github.com/maufadel/wattgpu)

---

### 🔍 相比现有方法的优势
| 方面 | 现有方法局限 | WattGPU 改进 |
|------|---------------|-------------|
| 泛化性 | 多数需已知 (GPU, LLM) 组合或特定硬件 | 可泛化至**未见 GPU / LLM** |
| 数据需求 | 依赖 profiling 或实测数据 | 仅需**公开 metadata** |
| 应用门槛 | 要求物理访问硬件 | 完全远程可用，适合云服务商/中小企业 |
| 准确性 | 分析模型（如 Roofline、TDP）误差大 | 显著降低 MdAPE，提升排名相关性 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **Watt Counts Dataset** [Argerich et al., 2026]：公开的 LLM 推理能效基准数据集。
- 包含：
  - **42 个开源 LLMs**（参数范围：0.125B ~ 27B）
  - **8 款 NVIDIA 服务器级 GPU**（涵盖 Volta 到 Hopper 架构）
- 场景覆盖：
  - **Offline**：高吞吐批量处理
  - **Server**：低负载与高负载下的在线服务（Poisson 请求流）

> ⚠️ 排除了消费级 GPU（如 RTX 4090）和 MoE 模型以聚焦主流云部署场景。

---

### 🧪 实验设置

#### 评估策略（三种递进难度）：
| 策略 | 描述 | 目标 |
|------|------|-------|
| **5-Fold CV** | 随机划分，每折中所有 GPU 和 LLM 均出现过 | 测试未见组合 `(GPU, LLM)` |
| **Leave-One-GPU-Out (LOGO)** | 每次留出一个完整 GPU 用于测试 | 测试**未见 GPU** 的泛化能力 |
| **Leave-One-LLM-Out (LOLO)** | 每次留出一个完整 LLM 用于测试 | 测试**未见 LLM** 的泛化能力 |

#### 评估指标：
- **MdAPE**（Median Absolute Percentage Error）：中位绝对百分比误差，抗异常值干扰
- **Pearson r**：预测值与真实值的线性相关系数
- **Kendall T (τ)**：
  - **GPU T**：在固定 LLM 下，模型对不同 GPU 的排序质量
  - **LLM T**：在固定 GPU 下，模型对不同 LLM 的排序质量  
  （高 τ 表示可用于高效选型决策）

#### 模型架构：
- 使用 **XGBoost** 回归器（优于线性模型、MLP、Elastic Net）
- 输入特征全部来自公开数据，无 runtime profiling
- 输出标准化：功率模型输出为 `P / TDP`，提高跨 GPU 可迁移性

---

### 🆚 基线方法对比

#### 功耗模型 Baselines：
1. **Plain TDP**：假设平均功耗等于 TDP
2. **Load-Scaled TDP**：$ P = \rho \times \text{TDP} $，其中 $\rho$ 是利用率因子（offline=1.0, server_low=0.2, server_high=0.6），通过调优使 MdAPE 最小

#### 延迟模型 Baseline：
- **Roofline Model**：将 ITL 近似为从内存加载一次 FP16 权重的时间  
  $$
  T_{\text{roof}} = \frac{2 \times P_{\text{model}} \times 10^9}{\text{Memory Bandwidth}}
  $$
  其中 batch size 设为 B=1（server）、B=256（offline）

> 所有 baseline 均为分析性公式，不依赖训练，故其表现不受 split 影响。

---

## 3. 主要实验结果和性能指标

### ✅ 功耗预测结果（Mean Power Draw）

| 场景 | 策略 | MdAPE | Pearson r | GPU T |
|------|------|--------|-----------|--------|
| Offline | LOGO | **3.4%** | 0.988 | 0.95 |
| Server | LOGO | **13.5%** | 0.965 | **0.76** |

> 💡 在 server 场景下，Load-Scaled TDP 的 MdAPE 为 26.0%，而 WattGPU 仅为 13.5%，**误差减少约 2×**

#### 关键发现：
- TDP 类方法在 offline 场景尚可（MdAPE≈4.4%），但在 server 场景严重高估功耗（MdAPE 达 190% for plain TDP）
- WattGPU 显著改进了 GPU 排序能力（GPU T ≥ 0.76），支持有效节能选型

---

### ⏱️ 延迟预测结果（Inter-Token Latency, ITL）

| 场景 | 策略 | MdAPE | Pearson r | GPU T |
|------|------|--------|-----------|--------|
| Server | LOGO | **8.5%** | 0.972 | 0.78 |
| Offline | LOGO | 24.9% | 0.727 | 0.72 |

> 💡 Roofline baseline 在 server 场景 MdAPE=29.6%，WattGPU 将其降至 **8.5%**，**误差减少约 3.5×**

#### 关键发现：
- ITL 主要是 memory-bound，bandwidth latency 是最重要特征（见图3）
- 尽管 roofline 有一定排序能力（GPU T=0.88），但绝对误差大；WattGPU 同时优化了精度与排序质量
- 在 offline 场景挑战更大（MdAPE=24.9%），可能因 vLLM 的 continuous batching 行为未被静态特征捕捉

---

### 🔍 特征重要性分析（Feature Importance）
来自 XGBoost 模型的特征重要性显示：

#### 功耗模型：
1. **Scenario**（场景）  
2. **GPU Memory Bandwidth**  
3. **Bandwidth Latency**（衍生特征）  
→ 表明内存带宽是决定功耗的关键因素

#### 延迟模型：
1. **Bandwidth Latency**（最关键）  
2. **Scenario**  
3. **Compute Latency**, **FP16 TFLOPS**  
→ 再次验证 LLM 推理是 memory-bound

> ⚠️ GPU 特征普遍比 LLM 特征更重要 → GPU 差异主导性能变化

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **WattGPU 是首个可在“未见 GPU”上准确预测 LLM 推理功耗与延迟的方法**，MdAPE ≤ 13.5%（server 功耗）和 ≤ 8.5%（server ITL）。
2. 模型仅依赖公开数据，无需任何 profiling，极大降低了部署前能效评估门槛。
3. 相比传统启发式方法（TDP、Roofline），WattGPU 在绝对误差和 GPU 排序质量上均有显著提升（**误差减少约 2–4×**）。
4. 成功揭示了一些反直觉现象：例如 H100 并非总是最优选择 —— 对 Llama 3.1 8B，在满足 SLA 前提下，A30 比 H100 节能 **43%**。

---

### ⚠️ 局限性
1. **Offline 场景预测较弱**（ITL MdAPE 高达 24.9%）  
   → 可能由于 vLLM 的 dynamic batching 和调度开销未被建模
2. **未支持量化模型、MoE 架构或多 GPU 配置**  
   → 这些具有不同的性能模式，需单独建模
3. **依赖静态特征**，未能利用 prompt length、KV-cache usage 等 workload-aware 信息

---

### 🔮 未来工作方向
1. 引入更丰富的 workload 特征（如平均 prompt 长度、生成长度），但仍保持免 profiling 特性
2. 扩展至 **quantized models** 和 **Mixture-of-Experts (MoE)** 架构
3. 支持 **multi-GPU** 和分布式推理场景
4. 构建 **ensemble 模型**，结合 roofline 的强排序能力和学习模型的高精度（类似 Imai et al. [19] 思路）
5. 集成至 **energy-aware workload scheduler** 或 **request router** 中，实现动态节能调度
6. 支持 **EU AI Act** 等法规要求的碳排放报告，提供透明可持续性评估路径

---

## 总结一句话
> **WattGPU 实现了仅凭公开数据即可精准预测未见 LLM-GPU 组合的推理功耗与延迟，推动绿色 AI 部署迈向自动化、普惠化。**

</details>

---

### 15. [Revisiting Decentralized Online Convex Optimization with Compressed Communication](https://arxiv.org/abs/2607.01665)

**Authors**: Hao Zhou, Xiaoyu Wang, Chang Yao, Mingli Song, Yuanyu Wan  
**Category**: cs.LG  
**Published**: 2026-07-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.01665v1  

#### Abstract
Decentralized online convex optimization (D-OCO) is a popular framework for distributed applications with streaming data. To tackle the communication bottleneck, previous studies have investigated D-OCO with compressed communication and proposed several algorithms that are variants of online gradien...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Revisiting Decentralized Online Convex Optimization with Compressed Communication**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
本文研究的是**Decentralized Online Convex Optimization (D-OCO)** 在**压缩通信**（compressed communication）场景下的算法设计问题。在分布式在线学习中，多个节点通过网络协作优化全局目标函数，但由于通信带宽限制，直接传输高维梯度或参数成本高昂。因此，如何在保证学习性能的同时减少通信开销是一个关键挑战。

以往的研究大多基于 **Online Gradient Descent (OGD)** 类型的算法，并结合压缩通信技术（如 Choco-Gossip），但这些方法存在以下问题：
- 对图谱间隙 $ p $ 和压缩率 $ \omega $ 的依赖较强，导致 regret 上界较差；
- 算法设计复杂，引入多级阻塞更新机制和重复压缩器以控制共识误差；
- 分析繁琐，难以扩展到更复杂的设置（如 bandit setting）。

---

### **提出的新方法与新思路**

本文首次将 **Follow-the-Regularized-Leader (FTRL)** 类型算法成功应用于 D-OCO 的压缩通信场景，提出了两个新型算法：

1. **CD-FTGL**（Compressed Decentralized Follow-The-Generalized-Leader）
   - 针对 **full-information setting**，即每个节点能观测完整损失函数梯度。
   - 利用 FTRL 的**对偶变量更新机制**，使得共识误差可以在不经过投影操作的情况下被有效控制。
   - 结合 **Choco-Gossip** 实现压缩通信下的平均共识。

2. **CD-FTBL**（Compressed Decentralized Follow-The-Bandit-Leader）
   - 扩展至 **bandit setting**，即只能获得损失值而非梯度。
   - 使用 one-point gradient estimator 构造梯度近似。
   - 引入更大的 block size 来降低通信频率。

---

### **相比现有方法的优势**

| 方面 | 优势说明 |
|------|----------|
| **算法简洁性** | 相比 Yang et al. [2026] 的 OGD 型算法，CD-FTGL/CD-FTBL 设计更优雅，无需复杂的双层阻塞更新或重复压缩器。 |
| **理论性能提升** | 尤其在 bandit setting 下：<br>- Regret bounds 显著优于已有结果：<br>  - 凸函数：从 $ O(\omega^{-1/4}p^{-1/2}nT^{3/4}) $ 改进为 $ O(nT^{3/4}) $<br>  - 强凸函数：从 $ O(\omega^{-1/3}p^{-2/3}nT^{2/3}(\log T)^{1/3}) $ 改进为 $ O(nT^{2/3}(\log T)^{1/3}) $<br>- **通信轮次显著减少**：<br>  - 仅需 $ O(\omega^{-1}p^{-2}\sqrt{T}) $ 或 $ O(\omega^{-1}p^{-2}T^{1/3}(\log T)^{2/3}) $ 轮通信，远低于基线的 $ O(T) $。 |
| **去耦效应** | regret 中关于 $ \omega $ 和 $ p $ 的依赖仅出现在低阶项，表明 bandit 反馈与压缩通信的影响被有效解耦。 |

> ✅ **关键洞察**：FTRL 的对偶变量天然适合压缩通信，因为其共识误差不受投影操作干扰，可直接通过 Choco-Gossip 控制。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ijcnn1** 和 **a9a**：来自 LIBSVM 数据仓库的标准分类数据集。
- 每个数据集包含约 3–5 万样本，维度 $ d \approx 20 $。
- 设置总时间步 $ T = 10,000 $，每轮一个样本流式输入。

### **实验设置**
- **网络拓扑**：
  - 小规模图：$ n=9 $ 个节点，随机生成 $ G(9,18) $ 图；
  - 大规模图：$ n=50 $ 个节点，$ G(50,100) $ 图。
- **通信模型**：
  - 使用满足 Assumption 3.1 的 gossip matrix（基于 local-degree weights 构造）。
  - 压缩方式：采用 **Top-k** 压缩器，压缩比 $ \omega = k/d $，分别测试 $ \omega = 0.5, 0.1, 0.05 $。
- **任务形式**：
  - **Full-information**：使用 logistic regression，反馈为梯度；
  - **Bandit**：仅反馈损失值，使用 one-point estimator 近似梯度。

### **评估指标**
- **Average Cumulative Loss (AL)**：
  $$
  AL(t,i) = \frac{1}{t} \sum_{\tau=1}^t f_{\tau,i}(x_i(\tau))
  $$
- **通信效率对比**：
  - 横轴为 **communication rounds** 和 **total transmitted bits (log scale)**，衡量收敛速度与通信成本。

### **基线方法**
| 场景 | 基线算法 | 来源 |
|------|---------|------|
| Full-info | Top-DOGD, DC-DOGD | Yang et al. [2026], Tu et al. [2022] |
| Bandit | Top-DOBD-1, DC-DOBD | Yang et al. [2026], Tu et al. [2022] |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Figure 1–6）**

| 指标 | CD-FTGL / CD-FTBL 表现 |
|------|------------------------|
| **收敛速度** | 在所有设置下均达到更低的平均损失，尤其在早期阶段表现更快。 |
| **通信轮次需求** | CD-FTBL 仅需约 **2,000 轮通信** 即可接近最优，而基线需满 $ T=10,000 $ 轮。 |
| **通信比特总量** | 曲线明显左移，表明在相同损失水平下所需传输比特数更少，通信效率更高。 |

#### **不同压缩比下的鲁棒性**
- 即使在极低压缩比 $ \omega = 0.05 $ 下，CD-FTGL/CD-FTBL 仍保持稳定收敛；
- 基线方法性能下降明显，验证了所提方法对压缩噪声更强的鲁棒性。

#### **网络规模扩展性（Figures 4–6）**
- 在 $ n=50 $ 的大网络上，CD-FTGL/CD-FTBL 依然优于基线；
- 表明算法具有良好的可扩展性。

---

### **消融实验：通信频率敏感性分析（Figures 7–9）**

- 固定 block size $ L=100 $，调整每块内通信轮数 $ K \in \{2, 5, 10, 20, 50, 100\} $
- **发现**：
  - 较小的 $ K $（如 $ K=2 $）虽然最终精度略低，但**初期下降极快**，且总通信量大幅减少；
  - 验证了理论中“少量通信即可维持足够共识”的结论；
  - 存在明显的 **accuracy-efficiency trade-off**，可根据应用场景灵活调节。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **FTRL 范式可以自然适配压缩通信**，且比 OGD 更优雅、更高效。
2. ✅ **CD-FTGL 匹配当前最优 regret bounds**，同时简化了算法与分析。
3. ✅ **CD-FTBL 在 bandit setting 下实现双重突破**：
   - regret bounds 去除了对 $ \omega $ 和 $ p $ 的多项式依赖；
   - 通信轮次降至 **sublinear level**（$ o(T) $），是首个同时减少通信轮次和每轮比特数的 D-OCO 算法。
4. ✅ 实验全面验证了算法在多种数据集、网络结构和压缩比下的优越性与鲁棒性。

---

### **方法的局限性**
- 当前分析依赖于 **oblivious adversary** 假设（对手事先确定所有损失函数），尚未推广到 adaptive adversary。
- 使用 block update 机制，可能牺牲一定的实时响应能力。
- 在非光滑或非强凸情形下，性能增益有待进一步探索。

---

### **未来工作方向**
1. **加速 gossip 技术的融合**：
   - 当前 CD-FTGL 使用标准 gossip，若能结合 **accelerated gossip**（如 AD-FTGL 中的技术），有望进一步改善对 $ p $ 的依赖。
2. **取消 block 更新机制**：
   - 是否可在 $ L=1 $（逐轮更新）下仍保持良好性能？这是理论上的开放问题。
3. **扩展至异构网络与动态拓扑**：
   - 当前假设静态无向图，未来可考虑有向图、时变连接等更现实场景。
4. **支持 biased & unbiased compressors 统一框架**：
   - 文中已证明可通过 scaling 兼容 unbiased compressors（Appendix I），但可进一步统一建模。

---

> 🔚 **总结一句话**：  
> 本文通过引入 FTRL 范式，重新审视了 D-OCO 与压缩通信的结合方式，不仅实现了更优的理论边界和通信效率，还带来了算法设计与分析上的根本性简化，为后续研究提供了新的范式选择。

</details>

---

### 16. [Set Diffusion: Interpolating Token Orderings Between Autoregression and Diffusion for Fast and Flexible Decoding](https://arxiv.org/abs/2607.01775)

**Authors**: Marianne Arriola, Volodymyr Kuleshov  
**Category**: cs.LG  
**Published**: 2026-07-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.01775v1  

#### Abstract
Discrete diffusion models have steadily improved in quality relative to autoregressive (AR) models. However, these models are normally constrained to fixed-length generation and do not support key-value (KV) caching. Block diffusion partially bridges diffusion and AR by generating token blocks left-...

---

### 17. [The Wiola Architecture for Efficient Small Language Models](https://arxiv.org/abs/2607.01394)

**Authors**: Aryuemaan Kumar Chowdhury, Afreen Shaik, Yaparla Bhargavi, Brahma Kumar  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.01394v1  

#### Abstract
We present Wiola, a fully original Small Language Model (SLM) architecture built from first principles, sharing no structural lineage with any existing model family including GPT, LLaMA, Mistral, or Falcon. Wiola introduces five independently novel components: (i) Spiral Rotary Positional Encoding (...

---

### 18. [Scaling with Confidence: Calibrating Confidence of LLMs for Adaptive Test Time Scaling](https://arxiv.org/abs/2607.01612)

**Authors**: Xuqing Yang, Yi Yuan, Shanzhe Lei, Xuhong Wang  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.01612v1  

#### Abstract
Training large language models (LLMs) with reinforcement learning (RL) has significantly advanced their performance on reasoning and question-answering tasks. However, prevailing RL reward designs typically prioritize response correctness, neglecting to incentivize models to express their confidence...

---

### 19. [MMIR-TCM: Memory-Integrated Multimodal Inference and Retrieval for TCM Clinical Decision Support](https://arxiv.org/abs/2607.01814)

**Authors**: Lihui Luo, Joongwon Chae, Ziyan Chen, Yang Liu, Siyi Cheng, Weihan Gao, Zelin Zeng, Xiaoming Yin, Samaneh Beheshti Kashi, Dongmei Yu, Lian Zhang, Jing Sui, Zeming Liang, Jiansong Ji, Peter E. Lobie, Peiwu Qin  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.01814v1  

#### Abstract
Traditional Chinese Medicine (TCM) diagnosis, particularly through tongue inspection, faces persistent challenges in subjectivity and reproducibility. The application of multimodal artificial intelligence to TCM clinical tasks, such as syndrome differentiation and prescription generation, is signifi...

---

### 20. [Fast Multi-dimensional Refusal Subspaces via RFM-AGOP](https://arxiv.org/abs/2607.02396)

**Authors**: Thomas Winninger  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.02396v1  

#### Abstract
Steering and monitoring activations in Large Language Models (LLMs) are increasingly used for both safety and interpretability. Early work assumed behaviours are encoded along single linear directions, but recent findings suggest complex behaviours, such as the refusal to answer harmful queries, liv...

---

### 21. [Exploiting Task-Based Parallelism for the Red-Black Gauss-Seidel Method on 2D Grids](https://arxiv.org/abs/2607.01735)

**Authors**: Shiting Long, Gustavo Ramirez-Hidalgo, Andreas Frommer, Dirk Pleiter  
**Category**: cs.DC  
**Published**: 2026-07-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.01735v1  

#### Abstract
Gauss-Seidel is a well-established iterative method for the solution of linear systems, and multicoloring has been widely used to increase parallelism in iterative solution techniques. Implementing multi-color Gauss-Seidel with conventional divide-and-conquer parallelization strategies, however, may...

---

### 22. [Message Passing Based Two-Timescale Bayesian Learning for Joint Channel and Memory Hardware Impairments Tracking](https://arxiv.org/abs/2607.01660)

**Authors**: Wei Xu, An Liu  
**Category**: cs.LG  
**Published**: 2026-07-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.01660v1  

#### Abstract
Hardware impairments in massive multiple-input multiple-output (MIMO) receivers introduce inter-symbol memory and inter-element coupling, severely degrading channel estimation. This paper employs a residual recurrent gated unit (RGRU) to model the intra-slot memory of the hardware impairments and pr...

---

### 23. [Efficient Temporal Point Processes via Monotone Alternating Splines](https://arxiv.org/abs/2607.01752)

**Authors**: Cheng Wan, Quyu Kong, Feng Zhou  
**Category**: cs.LG  
**Published**: 2026-07-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.01752v1  

#### Abstract
Temporal point processes (TPPs) have widespread applications across various domains. Compared to modeling the conditional intensity of a TPP, modeling its cumulative conditional intensity function (CCIF) improves computational efficiency and eliminates numerical approximation errors. However, curren...

---

### 24. [EO-Agents: A Three-Agent LLM Pipeline for Earth Observation Hypothesis Generation](https://arxiv.org/abs/2607.01584)

**Authors**: Mahyar Ghazanfari, Amin Tabrizian, Armin Mehrabian, Peng Wei  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.01584v1  

#### Abstract
Large language models have recently been explored for scientific hypothesis generation, but most prior work relies on unstructured literature and free-form textual claims. We present a pipeline for Earth observation that grounds hypothesis generation directly in the NASA Earth Observation Knowledge ...

---

### 25. [Safe and Adaptive Cloud Healing: Verifying LLM-Generated Recovery Plans with a Neural-Symbolic World Model](https://arxiv.org/abs/2607.01595)

**Authors**: Junyan Tan, Haoran Lin, Siyuan Guo, Yichen Fang, Xinyue Luo, Tianyu Shen, Zeyu Qiao  
**Category**: cs.AI  
**Published**: 2026-07-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.01595v1  

#### Abstract
As the scale and complexity of cloud-based AI systems continue to escalate, ensuring service reliability through rapid fault detection and adaptive recovery has become a critical challenge. While existing approaches integrate Large Language Models (LLMs) for semantic understanding and Deep Reinforce...

---

### 26. [FaithMed: Training LLMs For Faithful Evidence-Based Medical Reasoning](https://arxiv.org/abs/2607.01440)

**Authors**: Zhiyun Zhang, Liwen Sun, Xiang Qian, Chenyan Xiong  
**Category**: cs.CL  
**Published**: 2026-07-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.01440v1  

#### Abstract
Faithful reasoning is essential in medicine, where clinical decisions require transparent justification grounded in reliable evidence. Current medical LLMs either lack active access to evidence or use retrieved evidence without supervising how it should be appraised and applied during reasoning. To ...

---

### 27. [ProWAFT: A ROMA-LPD Instance for Workload-Aware and Dynamic Fault Tolerance in FPGA-Based CNN Accelerators](https://arxiv.org/abs/2607.01602)

**Authors**: Xinxin Chen, Haoran Qiao, Yiming Guo, Kecheng Luo, Siyuan Feng, Jingwen Ma  
**Category**: cs.CL  
**Published**: 2026-07-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.01602v1  

#### Abstract
SRAM-based FPGAs provide an attractive platform for energy- and latency-constrained CNN inference at the network edge, yet transient faults can lead to silent errors that compromise reliability. Always-on redundancy (e.g., full TMR) improves correctness but incurs substantial performance and energy ...

---

### 28. [HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching](https://arxiv.org/abs/2607.01299)

**Authors**: Yifei Liu, Juntong Wu, Yang Liu, Junhao Hu, Minghao Li, Xiaoxu Chen, Weihang Chen  
**Category**: cs.DC  
**Published**: 2026-07-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.01299v1  

#### Abstract
In retrieval augmented generation (RAG) and agentic LLM serving, prompts are assembled from independent segments into long contexts, making the prefill stage dominate the per-request computation cost. To this cost, two directions have emerged in parallel: position-independent caching (PIC) admits KV...

---

### 29. [Cadence: Extreme Pipelining with Multiple Concurrent Proposers](https://arxiv.org/abs/2607.02275)

**Authors**: Fatima Elsheimy, Mohammad Mussadiq Jalalzai, Tobias Klenze, Jovan Komatovic, Mike Setrin, Victor Shoup, Kushal Babel, Lioba Heimbach, Jason Milionis  
**Category**: cs.DC  
**Published**: 2026-07-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.02275v1  

#### Abstract
We present Cadence, a Byzantine fault-tolerant multi-proposer consensus protocol with arbitrarily low block intervals, optimal resilience, and optimal fast-path latency. Cadence divides time into equally spaced slots, one block per slot, each finalized in its own consensus instance. Blocks do not bu...

---

### 30. [The risk of KV cache compression](https://arxiv.org/abs/2607.01520)

**Authors**: Lukas Haverbeck, Carmen Amo Alonso, Andres Felipe Posada-Moreno, Sebastian Trimpe, Marco Pavone  
**Category**: cs.LG  
**Published**: 2026-07-03  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.01520v1  

#### Abstract
Transformer inference on long sequences is expensive because softmax attention repeatedly reads from a large KV cache. The prevalent approach to this bottleneck is KV cache compression, which replaces the full cache with a compact summary. Despite its practical importance, the design of such summari...

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

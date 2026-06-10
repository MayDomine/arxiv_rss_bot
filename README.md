# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-10 09:13:00 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [RATrain: A Resource-Aware Training Runtime for Large Language Models on Bandwidth-Constrained Heterogeneous Supercomputing Platforms](https://arxiv.org/abs/2606.10415)

**Authors**: Yao Lu, Shiqing Ma, Zhongzhi Luan, Gen Li, Jiaxing Qi, Bin Han, Hailong Yang, Depei Qian  
**Category**: cs.DC  
**Published**: 2026-06-10  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.10415v1  

#### Abstract
Production heterogeneous supercomputing platforms are increasingly used to host large language model (LLM) training workloads. However, existing GPU-oriented training runtimes typically rely on high-bandwidth device memory, fast interconnects, and mature collective communication libraries, making th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RATrain: A Resource-Aware Training Runtime for Large Language Models on Bandwidth-Constrained Heterogeneous Supercomputing Platforms

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 LLM 训练系统（如 Megatron-LM、DeepSpeed）主要为 GPU 集群设计，依赖高带宽设备内存、高速互联和成熟的 collective communication 库。然而，在以 **MT-3000** 为代表的异构超级计算平台上，存在以下瓶颈：
- 显式的内存层次结构（SM/AM/GSM/DDR）
- 每计算集群仅 20GB 可用 DDR 内存
- 跨集群通信带宽受限（约 3.7GB/s）

这些限制使得传统 GPU 式训练策略难以直接适配，导致训练效率低下。

### 🚀 提出的新方法与思路
论文提出 **RATrain** —— 一种面向带宽受限异构超算平台的资源感知型 LLM 训练运行时系统，其核心思想是将标准非交错 1F1B（One Forward One Backward）训练建模为 **training-state lifecycle scheduling** 问题。

#### 主要机制包括：
- **Layer-wise State Pipeline**  
  将梯度同步（GradSync）、参数更新（UpdateShard）、权重预取（PrefetchW）等状态任务分解到层级别，并在前向访问顺序中提前调度，避免集中在 step-end 形成“尾部延迟”。
  
- **Forward-Side Activation Recovery (FSR)**  
  在反向传播到达之前，利用 pipeline 中的空闲窗口恢复激活值，从而将部分 recomputation 开销从反向关键路径移出。

- **MT-3000-aware 执行后端**  
  针对 MT-3000 架构优化 FP16 GEMM 和 Attention Backward 实现，采用显式数据移动和内存驻留调度，减少 DDR 访问开销。

- **Resource-aware Configuration Planner**  
  结合模型结构、硬件资源和执行代价模型，自动搜索满足内存约束且预期步长时间最短的训练配置（PP/DP/ZeRO/Z/mb 等）。

### 🔍 相比现有方法的优势
| 维度 | 传统 GPU-style 方法 | RATrain |
|------|---------------------|--------|
| 并行策略 | 倾向于 TP-heavy 或 ZeRO-3 | 使用轻量级 ZeRO-2 + PP/DP，避免高频 intra-layer collectives |
| 状态处理 | 集中式 step-end 处理 | 分布式 layer-level 生命周期调度 |
| 激活管理 | Backward-time checkpointing | Forward-side recovery，隐藏恢复延迟 |
| 数据移动 | 隐式假设高带宽 | 显式建模 DDR/GSM/AM 数据流 |
| 配置选择 | 启发式调参 | 基于代价模型的自动化规划 |

> ✅ **优势总结**：RATrain 不改变训练语义，但在资源受限环境下通过细粒度调度显著降低暴露开销，提升端到端吞吐。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **训练数据流**：英文 C4 数据集（固定 token 流），用于控制变量下的性能对比。
- **验证正确性**：使用相同初始权重、tokenizer、学习率调度进行长序列训练轨迹比对。

### ⚙️ 实验设置
- **硬件平台**：真实的 MT-3000 异构超算平台
- **序列长度**：默认 2048
- **每集群可用内存上限**：20GB DDR
- **评估模型规模**：
  - LLaMA-2-7B / 13B / 70B
  - Baichuan2-13B
  - Qwen2.5-32B

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| `tokens/s` | 端到端吞吐量 |
| `step time` | 单步训练耗时 |
| `scaling efficiency` | 扩展至多集群时的加速效率 |
| `peak memory usage` | 最大内存占用 |
| `loss trajectory deviation` | 训练正确性验证（相对损失偏差） |
| `planner prediction error` | 规划器预测步长 vs 实测步长误差 |

### 🆚 基线方法对比
所有基线均基于相同的 MT-3000 后端实现（operator/backend/communication 一致），仅调度策略不同：
- **TP-heavy**：高张量并行度，引入大量 intra-layer collectives
- **ZeRO-3-heavy**：深度状态分片，增加参数视图重建开销
- **Backward Ckpt**：传统反向侧激活恢复
- **Full-save**：不启用 checkpoint，保存全部激活
- **Tuned PP/DP/ZeRO**：手动调优的传统策略，但禁用 RATrain 的三项核心机制

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 模型 | 方法 | Tokens/s | Step Time (s) | Speedup |
|------|------|----------|---------------|---------|
| LLaMA-2-13B | RATrain | **12,191.13** | 688.09 | 1.00× |
| | TP-heavy | 10,149.20 | 826.53 | 1.20× slower |
| | ZeRO-3-heavy | 11,684.48 | 717.93 | 1.04× slower |
| | Backward Ckpt | 8,952.21 | 937.04 | 1.36× slower |
| | Tuned PP/DP/ZeRO | 8,868.90 | 945.84 | 1.37× slower |

| Qwen2.5-32B | RATrain | **5,267.52** | 1,592.51 | 1.00× |
| | TP-heavy | 4,363.01 | 1,922.66 | 1.21× slower |
| | ZeRO-3-heavy | 4,663.50 | 1,798.78 | 1.13× slower |
| | Backward Ckpt | 3,879.06 | 2,162.54 | 1.36× slower |

> ✅ **最高达 1.37× 的端到端加速**

### 🔬 消融实验结果（Ablation Study on Qwen2.5-32B）

| 变体 | Step Time (norm.) | Exposed Tail (×) | 说明 |
|------|--------------------|------------------|------|
| Full RATrain | 1.00× | 1.00× | 完整版本 |
| -FSR | 1.33× | 1.01× | 移除 FSR 导致恢复延迟暴露 |
| -U-P (无 update-prefetch) | 1.24× | 2.31× | 参数准备无法重叠，引发前向阻塞 |
| -LSP (无 layer-wise pipeline) | 1.30× | 4.59× | GradSync 集中爆发形成尾部 |

> 💡 发现：性能增益来自三大机制协同作用，而非单一优化。

### 📦 资源可行性支持能力
RATrain 成功在 **20GB/cluster** 限制下支持从 7B 到 70B 的密集 LLM 训练：

| 模型 | 最小集群数 | Pipeline Degree (P) | Peak Mem (GB) |
|------|------------|---------------------|---------------|
| LLaMA-2-7B | 8 | 2 | 19.57 |
| Baichuan2-13B | 16 | 8 | 19.06 |
| Qwen2.5-32B | 64 | 16 | 18.14 |
| LLaMA-2-70B | 96 | 48 | 19.46 |

> ✅ 表明 **pipeline parallelism 是应对严格内存约束的关键手段**

### 📊 规划器准确性
| 模型 | 预测步长 (s) | 实测步长 (s) | 误差 |
|------|-------------|-------------|------|
| LLaMA-2-7B | 140.92 | 144.28 | 2.33% |
| Baichuan2-13B | 268.74 | 276.61 | 2.85% |
| Qwen2.5-32B | 441.83 | 455.21 | 2.94% |

> ✅ 平均预测误差仅 **2.67%**，表明代价模型准确可靠

### 📈 扩展性测试（LLaMA-2-7B, Scale-out）

| 集群数 | Tokens/s | Speedup | Scaling Efficiency |
|--------|----------|---------|---------------------|
| 256 | 29,069.73 | 1.00× | 100.0% |
| 512 | 57,558.07 | 1.98× | 99.0% |
| 768 | 85,465.01 | 2.94× | 98.0% |
| **1024** | **112,790.55** | **3.88×** | **97.0%** |

> ✅ 展现出优秀的弱扩展性，在千卡规模仍保持近线性加速

### ✅ 正确性验证
- 进行了长达 **1.028B tokens** 的训练运行
- 与 Baseline-1F1B 的损失轨迹几乎完全重合
- 最大相对损失偏差仅为 **0.081%**
- 最终损失绝对差为 **0.00064**

> ✅ 证明 RATrain **完全保留了原始训练语义**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 训练瓶颈已从算力转向资源耦合约束**  
   在异构超算上，关键瓶颈在于 **critical-path execution、parallel organization、training-state lifecycle 和 platform resource constraints** 的耦合效应。

2. **training-state lifecycle scheduling 是有效抽象**  
   将 GradSync、Update、Prefetch、Recovery 建模为具有明确生命周期的对象，可在 layer-level 和 stage-local 窗口内灵活调度，显著降低暴露开销。

3. **避免 intra-layer collectives 更优于进一步拆分计算**  
   在低带宽环境下，使用 PP+DP+lightweight ZeRO 比 TP-heavy 或 ZeRO-3 更高效。

4. **forward-side activation recovery 可有效隐藏 recomputation 开销**  
   利用 pipeline 中的空闲窗口提前恢复激活，能显著缓解反向路径压力。

5. **自动化 planner 可精准预测最优配置**  
   基于 profile 的代价模型可高效剪枝不可行空间，选择接近最优的训练计划。

### ⚠️ 方法的局限性
- 当前主要针对 **dense decoder-only LLMs**，未涵盖 MoE 架构。
- 对 extremely long sequence（如 >8k）的支持尚未充分验证。
- 规划器依赖离线 profiling，动态 workload 下适应性有待加强。
- 未探索更复杂的 hybrid pipeline scheduling（如 interleaved 1F1B）。

### 🔮 未来工作方向
1. 支持 **sparse models** 和 **MoE routing-aware scheduling**
2. 引入 **runtime feedback-driven adaptive planning**
3. 探索 **inter-operator rematerialization** 与 FSR 的联合优化
4. 扩展至其他类型的异构架构（如 CPU+FPGA+DSP 混合集群）
5. 结合 compiler 技术实现更高层次的自动并行生成

---

> **总结一句话**：  
RATrain 通过将 LLM 训练重构为 **training-state lifecycle scheduling** 问题，在保持训练语义不变的前提下，实现了在资源受限异构超算上的高效、可扩展、正确的大型语言模型训练，为 AI 与 HPC 融合提供了新的系统范式。

</details>

---

### 2. [PADD: Path-Aligned Decompression Distillation for Non-Router Teacher to Guide MoE Student Learning](https://arxiv.org/abs/2606.10369)

**Authors**: Xinyue Peng, Yi Qian, Jiaojiao Lin, Wenjian Shao, Yanming Liu  
**Category**: cs.CL  
**Published**: 2026-06-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.10369v1  

#### Abstract
As large language models (LLMs) continue to scale, it becomes increasingly challenging to grow model capacity under fixed computation budgets. We propose Path-Aligned Decompression Distillation (PADD), a framework for distilling knowledge from dense teachers without explicit routing into mixture-of-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PADD: Path-Aligned Decompression Distillation for Non-Router Teacher to Guide MoE Student Learning

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

PADD 针对 **dense-to-MoE 知识蒸馏**中的结构性挑战，解决以下四个核心难题：

- **Router Cold Start**：MoE 学生模型的路由器从零开始学习，无法区分语法与推理 token，导致专家功能同质化。
- **Capacity Gap**：当 MoE 学生的激活参数远小于 dense 教师时，难以吸收细粒度的 logits 输出。
- **Path Rupture**：离散的路由决策破坏了 chain-of-thought 的连续性，导致梯度不稳定。
- **Expert Homogenization**：传统负载均衡仅控制激活频率，忽略专家质量，导致专家趋向功能一致。

此外，现有方法如 MoE-to-MoE 蒸馏受限于路由策略不兼容，而 dense-to-dense 蒸馏无法指导 MoE 的路由学习。

---

### **提出了什么新方法或新思路**

论文提出 **Path-Aligned Decompression Distillation (PADD)**，一个四阶段、两相统一框架：

#### **Phase 1: 初始化阶段 (Stage I)**
- **Neuron-Cluster-Based Expert Initialization**  
  对 dense 教师 FFN 层的神经元进行 **cardinality-constrained K-Means 聚类**，将教师隐含的功能模块映射为学生专家的初始化目标。
- **Expert Warmup**  
  在固定均匀路由下对学生专家进行预热训练，使其初步形成差异化功能，避免早期路由不稳定。

#### **Phase 2: 训练阶段 (Stages II–IV)**
- **Stage II: Online Adaptive Distillation**  
  动态调整教师输出温度：当学生路径优势 $A_{i,s} > 0$ 时降低温度（增强监督信号），反之提高温度（鼓励探索），缓解 capacity gap。
- **Stage III: PR-GRPO (Path-Refined Group Relative Policy Optimization)**  
  引入 **routing shift** 作为抑制因子，在重要性比率中加入 $\exp(-\lambda \cdot \|I_{i,t,s}\|)$，抑制在路由剧烈变化且表现差的路径上的更新，稳定梯度。
- **Stage IV: Reward-Augmented Load Balancing**  
  将专家激活频率 $f_{j,s}$ 与平滑后的相对优势 $EMA(A_{j,s})$ 结合，动态更新路由偏置 $b_{j,s}$，优先激活高质量专家，防止 homogenization。

---

### **相比现有方法的优势**

| 优势维度 | PADD 的改进 |
|--------|------------|
| **结构对齐** | 通过 neuron clustering 显式传递教师的隐含模块结构，实现“路径对齐”的蒸馏 |
| **训练稳定性** | PR-GRPO 抑制路径突变，显著降低 router-shift，提升 routing consistency |
| **专家专业化** | 初始功能分化 + 奖励感知负载均衡 → 专家形成明确子领域专长（NMI 提升 2.3×） |
| **性能上限突破** | MoE 学生可在相同 inference cost 下 **超越其 dense 教师**（Qwen 家族 avg +2.5%） |
| **通用性** | 不依赖 teacher 的显式路由，适用于任意 dense 模型指导 MoE 学习 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **训练数据**：`DeepScaleR`（基于 AIME/AMC/MATH500 构建的大规模数学 RL 数据集）
- **评估数据集（数学推理）**：
  - `AIME24`, `AMC23`
  - `MATH500`, `Minerva`, `OlympiadBench`（简称 Olymp.）
- **非数学泛化测试**：
  - `MMLU-Pro`（多任务理解）
  - `LiveCodeBench v6`, `MultiPL-E`（代码生成）
  - `HumanEval`, `MBPP`（Python 函数补全）

> 所有评估均未使用训练数据，Pass@1 为主要指标。

---

### **实验设置和评估指标**

- **模型对**：
  - **Qwen Family**：Qwen2.5-Math-7B (dense teacher) → Qwen3-30B-A3B (MoE student, 3.3B active)
  - **DeepSeek Family**：DeepSeek-Math-7B → DeepSeek-V2-Lite (2.4B active)

- **训练流程划分**：
  - $D_A$: 激活统计与聚类（Stage I）
  - $D_B$: 专家预热
  - $D_C$: 主训练（Stages II–IV）
  - $D_P$: 评估

- **评估协议**：
  - 所有方法共享相同 student checkpoint、训练数据、decoding budget
  - 多随机种子平均（3 seeds），报告均值 ± 95% CI

---

### **基线方法对比**

| 基线方法 | 简要说明 |
|--------|--------|
| **Base** | 未经训练的 MoE 学生 |
| **Dense-GRPO** | 同 active 参数量的 dense 模型训练 GRPO |
| **MoE-Vanilla-GRPO** | MoE 学生仅用 GRPO，无蒸馏 |
| **RSPO** | 加入 router-shift 权重的 GRPO 变体 |
| **GSPO** | 序列级重要性比率裁剪 |
| **Online KD** | 固定温度的在线知识蒸馏 + GRPO |
| **Teacher (GRPO)** | 教师模型经 GRPO 微调后的性能（上界参考） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **表 1：数学推理主实验结果（Pass@1 %）**

| Method | Qwen Avg | DeepSeek Avg |
|--------|----------|--------------|
| Teacher (GRPO) | 77.7 | 58.1 |
| Base | 72.9 | 37.2 |
| Dense-GRPO | 53.5 | 45.6 |
| MoE-Vanilla-GRPO | 71.4 | 46.8 |
| Online KD | 73.6 | 46.7 |
| RSPO | 77.2 | 54.3 |
| **PADD (Ours)** | **80.2** | **55.2** |

> ✅ **PADD 在 Qwen 家族上以 3.3B active 参数超越 7B dense 教师（+2.5%）**  
> ✅ 在 DeepSeek 家族接近教师性能（仅差 2.9%），且在 AIME24/AMC23 上反超

---

### **与基线方法的对比结果**

- **vs. MoE-Vanilla-GRPO**：+8.8% (Qwen), +8.4% (DeepSeek) → 显示蒸馏机制的关键作用
- **vs. Online KD**：+6.6% (Qwen), +8.5% (DeepSeek) → 动态温度调节 + 路径对齐更有效
- **vs. RSPO/GSPO**：+3.0%/+3.9% → PR-GRPO 和奖励感知负载均衡带来额外增益
- **vs. Dense-GRPO**：+26.7% (Qwen), +9.6% (DeepSeek) → MoE 架构在同等 active 参数下表达能力更强

---

### **消融实验结果**

#### **图 2 & 表 8：移除各阶段的影响（Qwen 家族 avg）**

| 方法 | AIME24 | Minerva | Olymp. | Avg |
|------|-------|--------|--------|-----|
| PADD (Full) | 83.0 | 55.0 | 70.7 | **80.2** |
| w/o Stage I | 79.1 | 52.2 | 60.3 | 74.1 (-6.1) |
| w/o Stage II | 79.2 | 52.3 | 60.8 | 75.1 (-5.1) |
| w/o Stage III | 80.7 | 51.2 | 66.8 | 77.6 (-2.6) |
| w/o Stage IV | 81.3 | 52.7 | 68.2 | 78.7 (-1.5) |

> 🔹 **Stage I 最关键**：无结构初始化导致性能暴跌，验证了 neuron clustering 的必要性  
> 🔹 **Stage II 缓解 capacity gap**：固定温度无法适应学生路径质量  
> 🔹 **Stage III 稳定训练**：PR-GRPO 显著减少 path rupture  
> 🔹 **Stage IV 提升专家质量**：虽增益较小，但长期防止 homogenization

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **dense 教师可有效指导 MoE 学生学习路由策略**，即使自身无显式 router。
2. ✅ **neuron activation clustering 能揭示 dense 模型的隐含模块结构**，并成功迁移至 MoE 专家。
3. ✅ **路径对齐的蒸馏（path-aligned distillation）优于输出对齐**，能传递内部处理偏好。
4. ✅ **PR-GRPO 显著提升 routing stability**：router-shift 下降 ≥47%，expert consistency 提高。
5. ✅ **小而专业的 dense 教师（如 7B Math）可驱动大 MoE 学生超越自身性能**，实现“知识放大”。

#### **专家专业化可视化（图 3）**
- 教师 neuron clusters 展现出清晰的 subdomain specialization（如代数 vs 几何）
- 经 Stage I 后，学生专家激活模式与教师高度对齐
- 完整训练后，pattern 更加集中，表明 PADD 成功继承并优化了功能结构

#### **NMI/ESI 量化分析（表 3）**
| Method | NMI | ESI |
|--------|-----|-----|
| Vanilla-GRPO | 0.013 | 0.014 |
| Random-Cluster | 0.017 | 0.016 |
| **PADD (Stage I)** | **0.030** | **0.029** |

> PADD 的专家-子领域对应关系强度是 Vanilla 的 **2.3 倍以上**

---

### **方法的局限性**

- **依赖高质量的 dense 教师**：若教师内部结构混乱（如大模型 entangled representations），聚类效果下降。
- **Stage I 对聚类质量敏感**：需监控 silhouette、cluster variance 等指标，必要时采用 over-clustering + hierarchical merging。
- **训练开销增加**：因在线查询教师，训练时间增加约 20–32%（见 Table 6），但推理成本不变。
- **当前聚焦数学领域**：扩展到其他领域需重新设计 domain-specific 教师与聚类策略。

---

### **未来工作方向**

- 探索 **offline logit caching** 或 **progressive distillation** 以支持更大教师（如 34B/70B）。
- 将 PADD 扩展至 **vision-language** 或 **speech MoE** 模型。
- 研究 **自监督 clustering** 替代人工定义 subdomain，提升通用性。
- 结合 **sparse upcycling**，构建 “dense → MoE 初始化 → PADD 路由精炼” 的完整 pipeline。

---

> **总结**：PADD 提供了一种 **principled、efficient、effective** 的 dense-to-MoE 蒸馏范式，首次实现了在无显式路由监督下，让 MoE 学生不仅匹配、甚至超越 dense 教师的推理能力，为高效语言模型扩展提供了新路径。

</details>

---

### 3. [Sim2Schedule: A Simulator-Guided LLM Framework for Autonomous Open-Pit Mine Scheduling](https://arxiv.org/abs/2606.10286)

**Authors**: Mustavi Ibne Masum, Thiago Eustaquio Alves de Oliveira, Mahzabeen Emu  
**Category**: cs.AI  
**Published**: 2026-06-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.10286v1  

#### Abstract
Open-pit mine scheduling is a critical process for maximizing economic return under complex geotechnical and operational constraints. While Mixed-Integer Linear Programming (MILP) provides mathematically optimal baselines, its exponential computational complexity and inability to adapt in real time ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sim2Schedule: A Simulator-Guided LLM Framework for Autonomous Open-Pit Mine Scheduling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对**露天矿生产调度（Open-Pit Production Scheduling, OPPS）**这一复杂工业优化问题展开研究。该问题旨在在满足地质、工程和运营约束的前提下，最大化矿山项目的净现值（NPV）。传统方法如 **Mixed-Integer Linear Programming (MILP)** 虽能提供数学最优解，但存在以下问题：
- 计算复杂度呈指数增长，难以扩展到大规模实例；
- 忽略现实中的动态容量变化、严格的开采-加工耦合关系和三维空间前序依赖；
- 缺乏实时适应性和可解释性，无法应对现场条件变化。

此外，现有AI方法多集中于短期任务（如卡车调度），尚未有效解决长期战略级的块体开采序列规划问题。

---

### 🚀 提出的新方法与创新思路

作者提出了一种名为 **Sim2Schedule** 的新型框架，其核心是将 **Large Language Model (LLM)** 作为自主决策代理，在一个定制的**迭代式模拟器（simulator）**引导下进行调度决策。

#### 主要创新点包括：

1. **Simulator-Guided LLM 架构**
   - 将LLM置于一个闭环、零训练（zero-shot）环境中，仅通过推理参与决策。
   - 所有物理与操作约束（如geotechnical precedence、capacity limits、extraction-processing coupling）均由**模拟器在行动生成阶段隐式强制执行**，而非由LLM自行判断。
   - LLM专注于基于上下文的长期价值推理，从而实现“高阶认知”与“低层约束”的分离。

2. **完全可解释且安全的本地化部署**
   - 整个系统运行在一个封闭的数据安全环境中，无需调用云端API，保护敏感矿山数据。
   - 输出为结构化的调度日志（dispatch log），可被矿场规划人员实时监控和验证。

3. **提出更真实的MILP基准模型**
   - 针对传统MILP模型中存在的三大简化缺陷，提出了改进的MILP公式：
     - 引入**动态下限机制**（dynamic lower bound），避免在矿山末期因剩余资源不足导致不可行；
     - 显式建模**完整的3D前序块集合**（full 3D predecessor set），确保边坡稳定性；
     - 使用辅助二元变量强制**严格开采-加工耦合**（strict extraction-processing coupling），防止部分开采即加工的不合理假设。

4. **开源模拟器工具**
   - 开发并公开发布了一个交互式开放坑道调度模拟器，支持LLM集成，促进可复现研究。

---

### ⚖️ 相比现有方法的优势

| 维度 | 传统MILP | Metaheuristics / RL | Sim2Schedule (LLM + Simulator) |
|------|----------|---------------------|-------------------------------|
| **最优性** | 全局最优（理论上） | 近似解 | 接近最优（94%-99% MILP NPV） |
| **可扩展性** | 指数时间复杂度 O(2^N) | 中等 | 线性时间复杂度 O(N) |
| **实时适应性** | 无（批处理模式） | 有限 | 支持在线调整与实时反馈 |
| **可解释性** | 差（需后处理） | 差 | 强（每步输出可读指令） |
| **部署成本** | 高（依赖专家建模） | 中等 | 低（zero-shot，无需微调） |

---

## 2. 核心实验方法和设置

### 📊 数据集与仿真环境
- 使用合成的**块体模型（block model）**，包含不同规模（18–45 blocks）和时间周期（最多15期）的矿山实例。
- 每个block具有固定质量（50吨）、品位（ore grade）、开采成本和处理收益。
- 地质结构遵循标准3D网格布局，并定义了符合实际坡角要求的**precedence graph**。

### 🔧 实验设置
- **LLM模型**：
  - GPT-OSS (20B parameters)
  - DeepSeek-R1 (14B parameters)
  - 均通过 **Ollama** 框架本地部署，确保数据不出域。
- **提示工程**：
  - 设计结构化system prompt（见图4），明确LLM角色为“expert mine scheduler”。
  - 每轮输入当前mine state和可行action列表，输出JSON格式动作选择。
- **模拟器逻辑**：
  - 动态维护可行动作集（feasible action set），仅允许合法操作进入LLM视野。
  - 时间推进基于容量耗尽自动触发。

### 🎯 评估指标
| 指标 | 定义 |
|------|------|
| **NPV (Net Present Value)** | 主要目标函数，衡量经济回报 |
| **Optimality Gap (%)** | $(NPV_{MILP} - NPV_{method}) / NPV_{MILP}$，越小越好 |
| **Execution Time** | 总求解耗时（对数尺度比较） |
| **Operational Transparency** | 是否支持实时监控与人工干预 |

### 🆚 对比基线方法
1. **MILP**：本文提出的增强型MILP模型，作为理论最优基准。
2. **Greedy Heuristic**：每步选择即时NPV最高的动作。
3. **Random Policy**：从可行集中随机选动作为下界参考。
4. **No-Context LLM**：不提供实时状态更新的对照组，用于验证context重要性。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自Table 4 和 Fig. 8）

| 方法 | 实例大小 (N) | NPV ($K) | 时间 (秒) | 相对于MILP的NPV占比 |
|------|--------------|-----------|------------|------------------------|
| MILP | 18 | 24,650.41 | 3,247.72 | 100% |
| GPT-OSS (Context) | 18 | 24,083.20 | 1,527.55 | **97.7%** |
| DeepSeek (Context) | 18 | 24,083.20 | 2,827.75 | **97.7%** |
| MILP | 45 | 25,664.32 | 12,603.30 | 100% |
| GPT-OSS (Context) | 45 | 25,369.34 | 1,880.35 | **98.8%** |
| DeepSeek (Context) | 45 | 24,687.95 | 3,541.30 | **96.2%** |

> 注：MILP在N=45时接受约5–6%的gap提前终止。

---

### 🔍 与基线方法对比结果

| 方法 | NPV表现 | 可扩展性 | 决策质量 |
|------|---------|----------|-----------|
| **Random** | 最差（~13K vs MILP ~20K） | 高 | 完全无序，优先级混乱 |
| **Greedy** | 较好但饱和早（~19K） | 高 | 短视行为，错过深层高品位矿 |
| **No-Context LLM** | 与Greedy几乎一致 | 高 | 表明缺乏环境反馈时退化为贪婪策略 |
| **Context-Aware LLM** | **接近MILP（94%-99%）** | **线性扩展** | 展现出前瞻性规划能力 |

> 💡 图7显示：随着block数量增加，LLM+Simulator的optimality gap显著低于greedy/random，且优于no-context版本。

---

### 🔁 消融实验结果

#### （1）**上下文刷新率的影响（Fig. 11）**
- 刷新频率为1（仅保留最新状态）时效果最佳（NPV: $12,455.24）
- “Full history”反而导致性能下降（NPV: $11,988.92）
- **结论**：LLM更关注局部状态变化，全局历史会引入噪声；轻量级上下文管理即可维持高性能。

#### （2）**是否提供实时mine state？**
- 提供完整mine state和可行action列表的LLM明显优于无上下文版本。
- **发现**：没有外部感知的LLM本质上退化为greedy policy。

#### （3）**不同LLM架构的表现差异**
- GPT-OSS 在大尺度上表现更稳健（N=45仍达98.8%）
- DeepSeek 虽参数少但推理延迟更高，因其采用chain-of-thought生成中间推理链。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLM可在zero-shot下逼近MILP级别的调度性能**
   - 在多种实例中恢复 **94%–99% 的MILP最优NPV**，证明LLM具备强大的组合优化潜力。

2. **Simulator-in-the-loop范式有效分离“推理”与“约束”**
   - 将hard constraints交给simulator处理，使LLM专注long-term value reasoning，大幅提升可行性与效率。

3. **实时context至关重要**
   - 仅有结构化prompt不足以驱动高质量决策；持续的状态反馈是性能跃升的关键。

4. **线性可扩展性打破传统瓶颈**
   - MILP随问题规模呈指数增长，而LLM方法保持线性趋势，适用于大型矿山。

5. **高度可解释的操作输出**
   - 每一步都生成可解析的调度指令（如{"choice":1,"block_number":12,"amount":50}），便于工程师审计与干预。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖高质量模拟器设计** | 若simulator未能正确编码所有约束，则可能误导LLM做出无效决策。 |
| **LLM幻觉风险虽低但仍存在** | 尽管action space受限，极端情况下仍可能出现格式错误或越界选择（可通过parser过滤）。 |
| **未考虑不确定性** | 当前框架基于确定性模型，尚未整合grade uncertainty或设备故障等随机因素。 |
| **模型容量影响性能** | 更小或推理路径复杂的LLM（如DeepSeek）可能牺牲速度或精度。 |

---

### 🔮 未来工作方向

1. **扩展至真实世界大型block模型**
   - 应用于千万级block的真实矿山数据，测试极限可扩展性。

2. **引入不确定性建模**
   - 在simulator state中加入地质品位波动、市场价格变动、设备可用性等随机变量。

3. **多智能体协作框架**
   - 多个LLM agents分别负责开采、运输、加工等子系统，协同优化全流程。

4. **优化prompt策略以缩小残余gap**
   - 探索few-shot in-context learning、self-refinement loops等方式进一步提升决策质量。

5. **通用化至其他工业调度场景**
   - 如港口集装箱调度、智能制造排程、能源系统运维等具有强物理约束的long-horizon planning任务。

---

> 📌 **总体评价**：  
> Sim2Schedule 成功展示了 **LLM + Simulator** 范式在复杂工业系统中的巨大潜力——它不仅实现了接近数学最优的经济效益，还兼具**可解释性、安全性、实时性和线性可扩展性**，为传统优化方法难以覆盖的大规模动态调度问题提供了极具前景的新路径。

</details>

---

### 4. [Density Field State Space Models: 1-Bit Distillation, Efficient Inference, and Knowledge Organization in Mamba-2](https://arxiv.org/abs/2606.10932)

**Authors**: Chirag Shinde  
**Category**: cs.CL  
**Published**: 2026-06-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.10932v1  

#### Abstract
We present Density Field State Space Models (DF-SSM), a framework for compressing SSMs to a 1-bit scaffold with int8 low-rank correction. Applied to Mamba-2 1.3B, we achieve a 278 MB model (9.7x smaller than the 2.7 GB FP16 teacher) that runs at 21.4x faster inference on GPU (batch=1, relative to th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Density Field State Space Models: 1-Bit Distillation, Efficient Inference, and Knowledge Organization in Mamba-2*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）参数量大、内存占用高，难以部署在资源受限设备（如移动设备、边缘计算设备）上。尽管已有量化方法（如 4-bit 或更低），但极端量化（1–4 bits）通常导致显著的质量下降，且训练成本高昂。

**State Space Models**（SSMs）虽具备固定大小的隐藏状态（避免 Transformer 的 KV Cache 随序列增长），但在**权重压缩方面仍缺乏系统研究**。

### 提出的新方法与新思路
作者提出 **Density Field State Space Models**（DF-SSM），一种针对 Mamba-2 架构的高效压缩框架，包含三个阶段：

1. **Density Field Weights**（DFW）训练  
   - 将权重初始化为连续值，通过线性钳位映射到 17 个离散级别（-1 到 +1）。
   - 引入**量化退火机制**（quantization annealing）：从连续权重平滑过渡到离散二值化权重，避免“质量悬崖”（quality cliff）。

2. **Frozen Scaffold + LoRA Correction**  
   - 主干权重被冻结为 **1-bit 二值化 scaffold**（仅占原始模型 56% 存储）。
   - 使用一个小型 **int8 低秩适配器**（LoRA, rank=16）来恢复因量化损失的信息，仅增加约 12MB 参数。

3. **优化推理管线**（Optimized Inference Pipeline）  
   - **GPU**：利用 `cuBLAS INT8` tensor cores 加速 scaffold 的矩阵乘法；自定义 CUDA kernel 处理有状态操作（SSM recurrence 和卷积）；使用 CUDA graph 消除内核启动开销。
   - **CPU**：基于 AVX-512 VNNI 指令实现 bit-packed 权重展开与高效推理。

此外，论文首次对 SSM 内部的知识组织进行了系统可解释性分析。

### 相比现有方法的优势
| 维度 | DF-SSM | 现有方法（如 BitMamba-2、Bi-Mamba） |
|------|--------|-------------------------------|
| **压缩方式** | 混合精度（1-bit scaffold + int8 LoRA） | 全模型统一低比特（如 1.58-bit） |
| **训练范式** | 蒸馏自预训练 FP16 教师模型 | 从零开始训练（需百亿 token） |
| **训练效率** | 仅需 **32M token + 6小时 A100** | 需 105B–150B token，多 GPU 数天 |
| **推理速度** | GPU 上达 **21.4× 加速**（batch=1） | 无公开推理优化报告 |
| **知识保留** | 下游任务性能接近 BitMamba-2（差 2–4 pp） | 性能略优但训练成本极高 |

> ✅ **核心优势**：以极低成本完成高质量压缩，在模型大小、推理速度、任务性能之间取得优异平衡。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **蒸馏训练数据**：仅使用 **32M tokens**（来自 The Pile 数据集），用于 DFW 和 LoRA 微调。
- **下游评估任务**（共 5 项）：
  - **BoolQ**（自然语言推理）
  - **PIQA**（物理常识推理）
  - **HellaSwag**（情境常识推理）
  - **WinoGrande**（代词消解）
  - **ARC-easy**（科学问答）

> 注：未在这些任务上进行微调，直接 zero-shot 评估。

### 实验设置与评估指标
| 设置项 | 描述 |
|-------|------|
| **教师模型** | Mamba-2 1.3B（FP16，PPL=14.3） |
| **学生模型** | DF-SSM（1-bit scaffold + int8 LoRA，总大小 278MB） |
| **硬件平台** | A100-SXM4-40GB（GPU）、Intel Xeon（CPU） |
| **批处理大小** | batch=1 至 512（测试吞吐量变化） |
| **主要指标** | 
| - 模型大小（MB） | 
| - 推理吞吐量（tokens/sec） | 
| - Perplexity（PPL） | 
| - 下游任务准确率（Accuracy %） | 
| - 压缩比 & 速度提升倍数 |

### 基线方法对比
| 基线 | 类型 | 特点 |
|-----|------|------|
| **FP16 Teacher**（Mamba-2 1.3B） | 全精度参考 | PPL=14.3，大小 2.688GB |
| **mamba-ssm 库实现** | 官方优化实现 | 含定制 CUDA kernels，作为推理基准 |
| **BitMamba-2** | 1.58-bit 从头训练 | 在 150B tokens 上训练，精度更高但成本巨大 |
| **Bi-Mamba** | 1-bit 从头训练 | 针对 Mamba-1 架构，训练耗时长 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 📦 模型压缩效果
| 组件 | 精度 | 大小 | 占比 |
|------|------|------|------|
| Scaffold（投影层） | 1-bit packed | 155 MB | 56% |
| Embedding + LM Head | int8 | 103 MB | 37% |
| LoRA Correction | int8 | 12 MB | 4% |
| 其他（norms, conv, SSM params） | FP16/FP32 | 8 MB | 3% |
| **总计** | — | **278 MB** | — |
| 教师模型（FP16） | FP16 | 2,688 MB | — |
| **压缩比** | — | **9.7×** | — |

> 🔹 LoRA 和 embedding 层实现了 **lossless int8 量化**（PPL 几乎不变）。

#### ⚡ 推理性能（A100 GPU）
| Batch Size | FP16（mamba-ssm） | DF-SSM | Speedup |
|------------|--------------------|--------|---------|
| 1 | 14 tok/s | 299 tok/s | **21.4×** |
| 8 | 116 tok/s | 647 tok/s | 5.6× |
| 32 | 482 tok/s | 1,963 tok/s | 4.1× |
| 512 | — | ~5,081 tok/s | 接近 HBM 带宽极限 |

> 💡 加速来源：
> - 8× 更少的内存加载（bit-packed weights）
> - INT8 tensor core 高吞吐
> - CUDA graph 消除 launch overhead
> - 固定状态缓存避免重复计算

#### 💻 CPU 推理表现（Intel Xeon, 4线程）
| 实现 | 吞吐量（tok/s） | 内存占用 |
|------|------------------|----------|
| FP16 PyTorch | 12 | 2,688 MB |
| DF-SSM（AVX-512） | **22** | **567 MB** |

> ✅ 在 CPU 上也实现 **1.8× 加速 + 80% 内存节省**

#### 📈 下游任务准确率（Zero-shot）
| Task | Random | DF-SSM | Teacher (FP16) | BitMamba-2 | Retention* |
|------|--------|--------|---------------|-------------|-----------|
| BoolQ | 50.0% | **60.8%** | 64.2% | 62.4% | 94.7% |
| PIQA | 50.0% | **67.1%** | 73.2% | 68.8% | 91.7% |
| HellaSwag | 25.0% | **41.4%** | 59.9% | 45.6% | 69.1% |
| WinoGrande | 50.0% | **54.7%** | 60.9% | 52.8% | 89.8% |
| ARC-easy | 25.0% | **50.2%** | 64.1% | — | 78.3% |

> *Retention = (DF-SSM - Random) / (Teacher - Random)*  
> 🔹 DF-SSM 在多数任务上保留了教师模型 **78–95% 的性能**，尤其在 BoolQ 和 WinoGrande 表现优异。
> 🔹 与 **BitMamba-2** 相比，差距仅为 **2–4 个百分点**，甚至在 WinoGrande 上反超 +1.9pp。

#### 🔍 消融实验结果
| 配置 | PPL | 说明 |
|------|-----|------|
| Teacher (FP16) | 14.3 | 原始性能 |
| Scaffold only（无 LoRA） | 101.5 | 严重退化 |
| Scaffold + LoRA（FP16） | 49.2 | LoRA 显著恢复质量 |
| Scaffold + LoRA（int8） | **49.1** | 几乎无损量化 |
| Scaffold + LoRA（int4） | 52.9 | 开始出现轻微退化 |

> ✅ LoRA 是关键组件，其作用不是简单修正舍入误差（correlation ≈ 0），而是纠正输入加权输出误差。

---

## 4. 关键结论和发现

### 主要发现

#### ✅ **极端量化可通过蒸馏高效实现**
- 仅用 **32M tokens 和 6 小时单卡 A100** 即可完成高质量压缩。
- **混合精度架构**（1-bit scaffold + int8 LoRA）优于全模型均匀低比特量化。

#### ✅ **内部知识具有系统性组织结构**
通过对 DF-SSM 的可解释性分析，首次揭示 SSM 中存在 **三阶段处理流程**：

| 阶段 | 层范围 | 功能 | 证据 |
|------|--------|------|------|
| **Categorize**（分类） | L0–L3 | 识别问题模板类型 | 早层聚类明显，分类准确率达 94% |
| **Recall**（检索） | L25–L35 | 检索具体事实 | 因果干预显示知识集中在 L32–L36（5层窗口） |
| **Format**（格式化） | L36–L47 | 输出分布准备 | 所有类别向共享表示收敛 |

> 🔹 发现“**抽象意图空间**”（abstract intent space）：早期层（L0–L3）已能分类问题类型，但其表示**不与词汇表对齐**（logit lens 输出无意义）。
> 🔹 知识定位在 **10–16 维子空间**中即可区分 10 个首都城市。

#### ✅ **语法先于语义**
- 早层分类由**模板结构驱动**而非语义内容：
  - “The capital of [X] is” 类别（如 continents, currencies）迅速聚类；
  - 模板多样类别（如 physics, medical）始终无法有效聚类。
- 表明模型优先学习句法模式，再填充具体内容。

#### ✅ **结构可能先于强度**
- 尽管模型整体 PPL 较高（49.2），**知识组织结构完整**（三阶段、因果定位、维度集中等）。
- 暗示：**表征结构的发展可能独立于或早于实际事实记忆能力**。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖预训练教师模型** | 无法端到端训练，必须已有高质量 FP16 模型 |
| **PPL 仍有较大差距** | 从 14.3 → 49.2，质量下降显著，尤其影响复杂推理任务（如 HellaSwag） |
| **HellaSwag 性能偏低** | 可能因蒸馏数据不足（仅 32M tokens），非精度限制 |
| **可解释性结论泛化性未知** | 当前发现基于单一模型（Mamba-2 1.3B），是否适用于其他规模/架构尚不清楚 |
| **“1-bit”标签需谨慎理解** | 实际仅 scaffold 投影层为 1-bit，embedding 和 LoRA 仍为 int8 |

---

### 未来工作方向

1. **探索结构形成过程**  
   在从头训练的不同训练阶段追踪知识组织演化，验证“结构先于强度”的假设。

2. **扩展至其他架构**  
   将 DF-SSM 应用于扩散模型（如 SDXL → ~1GB）、FLUX 等大规模生成模型。

3. **改进 LoRA 训练策略**  
   尝试更高 rank、多阶段 distillation 或引入 sensitivity-aware 机制进一步缩小 PPL 差距。

4. **轻量化 CPU 部署优化**  
   结合 ONNX Runtime、TensorRT-LLM 等工具链，推动移动端落地。

5. **探索 syntax-first 的认知意义**  
   是否反映人类语言处理机制？能否用于构建更鲁棒的句法感知模型？

---

> 🧠 **最终结论**：  
> DF-SSM 不仅是一种高效的 SSM 压缩方案，更提供了一个观察模型内部知识组织的“显微镜”。它证明了**即使在极端量化下，模型也能保持清晰的内部结构**，这为未来轻量化、高可解释性的 LLM 设计提供了重要启示。

</details>

---

### 5. [Operator Fusion for LLM Inference on the Tensix Architecture](https://arxiv.org/abs/2606.09879)

**Authors**: Qingbo Wu, Ke Li, Wenzhu Wang, Jie Yu, Ruian Zhang, Lili Liu  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.09879v1  

#### Abstract
This study addresses on-device inference bottlenecks of Transformer models on Tenstorrent's Tensix architecture and proposes an operator fusion strategy that enhances data locality. RMSNorm is fused with matrix multiplication in self-attention and in the FFN, enabling back-to-back execution of memor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Operator Fusion for LLM Inference on the Tensix Architecture》核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对在 **Tenstorrent Tensix 架构** 上进行大语言模型（LLM）推理时存在的 **on-device 推理瓶颈**，特别是由频繁的中间结果读写导致的高内存延迟和调度开销问题。

Transformer 模型在推理过程中表现出典型的“read-compute-write”模式，其中 memory-bound 算子（如 RMSNorm、激活函数）与 compute-bound 算子（如 MatMul）交替执行，导致大量中间结果需在片外 DRAM 和片上 SRAM 之间搬运，严重制约边缘设备上的推理效率。

---

### **提出了什么新方法或新思路**

作者提出了一种 **数据流与内存层次感知的算子融合策略（operator fusion strategy）**，主要包含以下两个层面的创新：

#### ✅ 单核算子融合（Single-Core Operator Fusion）
- 将 **RMSNorm** 与后续的矩阵乘法（MatMul）进行融合：
  - 在 self-attention 中：将输入处的 RMSNorm 融合进 QKV 投影 MatMul；
  - 在 FFN 中：将 RMSNorm 融合进第一个线性层（Up/Gate 投影）的 MatMul。
- 融合后，RMSNorm 的输出直接在 **on-chip SRAM** 中被 MatMul 消费，避免写回 DRAM 再读取，形成“pass-through”执行模式。
- 实现方式基于 TT-metalium 编程模型，在 compute kernel 中集成 SFPU/FPU 执行 RMSNorm + MatMul 流水线。

#### ✅ 多核 NoC 多播加速机制（Multi-Core NoC Multicast Acceleration）
- 利用 Tensix 的 2D mesh 架构和 NoC（Network-on-Chip），设计了 **行主控（row master）与列主控（column master）多播机制**：
  - 每行最左侧 core 作为 row master，负责从 DRAM 读取输入 A 和 RMSNorm 参数 γ，并通过 NoC **水平广播** 给同行其他 core；
  - 每列最上方 core 作为 column master，负责读取权重 B 并通过 NoC **垂直广播** 给同列 core。
- 此机制显著减少多个 core 对相同数据的重复 DRAM 访问，缓解带宽争用。

---

### **相比现有方法的优势**

| 方面 | 本文方法优势 |
|------|--------------|
| **数据局部性** | 显著提升 on-chip 数据复用，减少中间结果的 off-chip 存储访问 |
| **内存效率** | 利用 NoC 多播将密集的 DRAM→core 通信转为高效的 core→core on-chip 通信 |
| **调度开销** | 消除 RMSNorm 与 MatMul 之间的任务切换与同步代价 |
| **通用性** | 不依赖特定模型结构，适用于主流 decoder-only LLM（含 RMSNorm、RoPE、SwiGLU） |
| **可扩展性** | 支持不同规模 core 数量（64/128 cores）和模型大小（0.5B~4B） |

---

## 2. 核心实验方法和设置

### **使用的模型**
- **Qwen2.5-0.5B**
- **Qwen3-0.6B**
- **Qwen3-4B**

均为基于 decoder-only Transformer 架构的开源 LLM，具备典型组件：RMSNorm、RoPE、SwiGLU 激活等，适合验证融合策略的有效性。

---

### **实验平台**
- **硬件平台**：Tenstorrent Wormhole N300 加速卡
  - 集成两颗 Tensix 芯片，共 **128 个 Tensix cores**
  - 总计 24 GB GDDR6（DRAM）、192 MB on-chip SRAM
  - 峰值性能：466 TFLOPS (FP8)，131 TFLOPS (FP16)
  - 支持单芯片模式（64 cores，模拟 N150）
- **软件栈**：TT-metalium + ttnn-visualizer
  - 使用 TT-metalium 实现底层 kernel 编程（reader/compute/writer kernels）
  - 使用 **ttnn-visualizer** 进行细粒度性能分析（操作执行时间、调度间隙）

---

### **评估指标**

| 指标 | 描述 |
|------|------|
| **Latency ↓** | 关键模块（Attention、MLP）及单个 decoder layer 的端到端延迟 |
| **Pearson Correlation Coefficient (PCC)** | 衡量融合实现与原始 baseline 输出之间的数值一致性，范围 [-1,1]，越接近 1 越一致 |
| **DRAM Read/Write Volume** | 间接评估（通过延迟改善反映） |
| **Hardware Utilization** | 通过流水线重叠程度和 Op-to-Op gap 缩小情况体现 |

---

### **基线方法对比**
- **Baseline**：标准非融合实现，即 RMSNorm 与 MatMul 分别独立执行，中间结果写回 DRAM。
- **对比项**：
  - Attention 模块延迟
  - MLP 模块延迟
  - 单 decoder layer 延迟
  - 数值精度保持能力（PCC）

> 注：未与其他 fusion 框架（如 Optimus、TileFlow）直接比较，而是聚焦于在 Tensix 架构下的定制化优化效果。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Table 1）**

| Tensix Cores | Model | Attention Latency ↓ | MLP Latency ↓ | Decoder Layer Latency ↓ | PCC |
|-------------|-------|------------------------|----------------|----------------------------|-----|
| 64 | Qwen2.5-0.5B | **37.44%** | 12.04% | **7.91%** | 99.94% |
| 64 | Qwen3-0.6B | 18.53% | 5.66% | 4.58% | 99.57% |
| 128 | Qwen3-4B | 10.63% | **15.89%** | 3.58% | 98.75% |

---

### **与基线方法的对比结果**

- **最大延迟降低**：
  - Attention 模块最高提速 **37.44%**（Qwen2.5-0.5B @64 cores）
  - MLP 模块最高提速 **15.89%**（Qwen3-4B @128 cores）
  - 单 decoder layer 最高提速 **7.91%**
- **所有配置下 PCC ≥ 98.75%**，表明融合未引入显著数值偏差，满足部署级精度要求。
- 更大的模型（如 Qwen3-4B）虽然绝对加速比略低，但仍取得可观收益，说明方法具有良好可扩展性。

---

### **消融实验结果（隐含分析）**

尽管文中未明确列出消融实验表格，但从设计逻辑中可推断出以下关键因素的作用：

| 因素 | 效果 |
|------|------|
| **RMSNorm-MatMul 融合** | 消除中间结果 DRAM 写回与再读，是延迟下降主因 |
| **NoC 多播机制** | 减少多 core 对相同输入/权重的竞争性读取，提升内存效率 |
| **Grid-aligned 分区** | 确保数据自然分布，避免跨 core 移动 |
| **Weight Reuse** | 批处理场景下进一步提高带宽利用率 |

> 实验结果显示，在不同模型规模和 core 数量下均稳定增益，验证了各组件协同有效性。

---

## 4. 关键结论和发现

### **主要发现**

1. **算子融合能有效缓解边缘 LLM 推理中的内存墙问题**：
   - 通过将 memory-bound 的 RMSNorm 与 compute-bound 的 MatMul 融合，显著减少 DRAM 访问次数。
   
2. **Tensix 架构支持高效 on-chip 数据流优化**：
   - 利用其 2D mesh + NoC + 大容量 SRAM 特性，结合 TT-metalium 底层控制，实现了高并行、低延迟的数据分发与计算流水。

3. **NoC 多播机制大幅提升多核协作效率**：
   - 行/列主控节点的广播策略有效降低了对 DRAM 带宽的压力，提升了整体吞吐。

4. **方法在保持高数值一致性的同时实现显著加速**：
   - 所有测试模型 PCC > 98.75%，证明融合过程无损，具备工程落地价值。

---

### **方法的局限性**

1. **依赖特定硬件特性**：
   - 方法高度依赖 Tensix 的 NoC 结构、SRAM 容量和 TT-metalium 编程能力，迁移到其他架构（如 GPU、NPU）需要重新适配。

2. **仅融合 RMSNorm 类轻量算子**：
   - 当前融合范围局限于 RMSNorm → MatMul，未涵盖更多 activation 或复杂 fusion pattern（如 QKV + softmax + matmul）。

3. **未考虑动态 batching 或 KV Cache 优化**：
   - 实验集中在静态推理路径，未涉及生成式任务中的动态序列长度或缓存管理优化。

---

### **未来工作方向**

1. **扩展融合范围**：
   - 探索更复杂的 fusion subgraph，例如将 RoPE、softmax 或 SwiGLU 纳入融合链路。

2. **支持动态推理与 streaming 场景**：
   - 结合 KV Cache 管理，实现持续生成过程中的算子融合与数据流优化。

3. **自动化 fusion 策略搜索**：
   - 借鉴 TileFlow、Optimus 等思想，构建面向 Tensix 的自动 fusion plan generator。

4. **跨芯片融合与分布式推理**：
   - 在多 Wormhole 设备间实现协同 fusion 与通信优化，支撑更大模型部署。

---

> ✅ **总体评价**：  
> 本工作提供了一个 **面向 Tensix 架构的高度实用化 LLM 推理优化方案**，在 operator-level 与 dataflow-level 实现联合优化，为在带宽受限的边缘环境中高效部署中大型 LLM 提供了清晰的技术路径。

</details>

---

### 6. [Alignment Defends LLMs from Property Inference Attacks](https://arxiv.org/abs/2606.10217)

**Authors**: Pengrun Huang, Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.10217v1  

#### Abstract
Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets that may contain sensitive, dataset-level properties. Recent work has shown that such dataset-level information can be effectively extracted through property inference attacks, posing a confidentiality risk. Existin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Alignment Defends LLMs from Property Inference Attacks*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

该论文针对**大语言模型（LLMs）中的属性推断攻击（Property Inference Attacks）**带来的隐私风险提出防御方案。  
这类攻击允许攻击者通过查询已部署的 LLM，从其生成输出中推断出训练数据集层面的敏感统计属性（如患者性别比例、特定疾病出现频率等），从而威胁数据保密性。

传统防御方法（如重采样、差分隐私）通常需要访问原始训练数据并重新训练模型，这在实际部署场景中往往不可行。

---

### ✅ 提出了什么新方法或新思路

作者提出了一种**基于对齐（alignment-based）的后训练防御机制**，无需修改训练数据或重新训练模型，即可有效缓解属性推断攻击。

#### 核心思想：
利用 LLM 特有的**对齐技术**（如 RLHF），在 fine-tuning 完成后，通过 post-training alignment 将模型的输出分布“重塑”为一个预设的目标属性比例 $ r_t $（例如公开先验或平衡分布），使得攻击者无法恢复真实的训练数据属性比例 $ r_{\text{true}} $。

#### 具体实现方式：
将两种主流的偏好优化框架适配为防御工具：
- **DPO (Direct Preference Optimization)**：构建偏好对（preference pairs），根据当前估计的属性比例与目标的比例差异，动态决定哪些输出应被偏好。
- **GRPO (Group Relative Policy Optimization)**：设计特定奖励函数，在每轮迭代中根据样本是否有助于向目标比例靠拢来分配奖励，并进行 on-policy 更新。

---

### ✅ 相比现有方法的优势

| 方法 | 是否需原始数据 | 是否需重训练 | 是否依赖推理时控制 | 实用性 |
|------|----------------|--------------|--------------------|--------|
| Resampling / Subsampling | 是 | 是 | 否 | 低（部署后难应用） |
| Temperature Scaling | 否 | 否 | 是（用户可绕过） | 中（影响生成质量） |
| Differential Privacy | 是 | 是 | 否 | 低（对分布级保护有限） |
| **本文方法 (DPO/GRPO)** | **否** | **否** | **否** | **高（适用于已部署模型）** |

> ✅ **优势总结**：
> - **Post-training 防御**：适用于已经部署的模型。
> - **无需访问训练数据**：仅操作模型输出行为。
> - **保持模型效用（utility）**：实验显示几乎不损害任务性能。
> - **同时抵御多种攻击**：不仅防御 generation-based 攻击，也间接削弱 shadow-model 攻击。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 描述 | 属性任务 |
|-------|------|---------|
| **ChatDoctor** | 医疗问答数据集，基于 LLaMA-1-7B 微调 | 推断训练集中女性患者的占比（ground-truth ratio ∈ {0.3, 0.5, 0.7}） |
| **MedCalc-Bench** | 新引入的医学计算基准，测试数值推理能力 | 推断“CKD-EPI 方程”在训练数据中的出现比例（ratio ∈ {0.03, 0.05, 0.07}） |

此外还设置了**多属性场景**（multi-class alignment），在 ChatDoctor 上同时控制多个诊断相关属性（如消化系统疾病、精神障碍）的比例。

---

### ⚙️ 实验设置

#### 模型
- **LLaMA-1-7B** + ChatDoctor（CC & QA 模式）
- **Qwen-2.5-7B-Instruct** + MedCalc-Bench（CC & QA 模式）

#### 微调模式
- **QA 模式**：仅预测答案部分
- **CC 模式**：完成整个对话序列（instruction + input + output）

#### 攻击类型
1. **Generation-based Attack**：直接用生成输出估计属性比例。
2. **Shadow-model-based Attack**：训练多个 shadow models，提取 word-frequency 特征，训练回归元模型预测属性比例。

#### 防御目标
- 设定目标比例 $ r_t $：
  - ChatDoctor: $ r_t = 0.5 $
  - MedCalc: $ r_t = 0.05 $
  - 多属性：$ r_t = 0.05 $

#### 基线方法对比
| 方法 | 类型 |
|-----|------|
| No Defense | 无防御 |
| Subsampling | 训练前重采样使属性比接近 $ r_t $ |
| Temperature Scaling | 调整解码温度以混淆输出分布 |
| **DPO / GRPO** | 本文提出的 alignment-based 防御 |

---

### 📊 评估指标

| 指标 | 含义 |
|------|------|
| **Attack MAE**：<br>$ |\hat{r}_\alpha(f) - r_{\text{true}}| $ | 衡量攻击准确性；越大表示防御越成功（攻击误差高） |
| **Alignment Error (MAE<sub>target</sub>)**：<br>$ |\hat{r}_{\text{generation}}(f) - r_t| $ | 衡量模型输出是否对齐到目标比例；越小越好 |
| **Utility** | 保留原始任务性能：<br>- ChatDoctor: F1 score<br>- MedCalc: Accuracy |

---

## 3. 主要实验结果和性能指标

### 🔢 关键性能数据（来自 Table 1 和 Table 2）

#### ✅ 在 **MedCalc (CC 模式)** 上的结果：

| 方法 | Generation MAE ↑ | Shadow MAE ↑ | Acc ↓ | MAE<sub>target</sub> ↓ |
|------|------------------|---------------|--------|------------------------|
| No Defense | 0.0104 | 0.0092 | 0.3741 | 0.0169 |
| DPO | **0.0155** | 0.0139 | 0.3678 | **0.0089** |
| GRPO | 0.0117 | **0.0133** | **0.3701** | **0.0066** |

> ✔️ DPO 和 GRPO 显著提升攻击误差（即降低泄露），且几乎不损失准确率。

#### ✅ 在 **ChatDoctor (CC 模式)** 上的结果：

| 方法 | Generation MAE ↑ | Shadow MAE ↑ | F1 ↓ | MAE<sub>target</sub> ↓ |
|------|------------------|---------------|-------|------------------------|
| No Defense | 0.0354 | 0.0332 | 0.8407 | 0.1429 |
| DPO | 0.1738 | **0.0692** | 0.8421 | **0.0434** |
| GRPO | **0.1357** | 0.0823 | **0.8410** | 0.0353 |

> ✔️ 两种方法均显著提高攻击 MAE（尤其是 generation-based），说明真实属性更难被恢复；
> ✔️ GRPO 对 shadow attack 抵抗更强，DPO 在 generation 攻击上表现略优；
> ✔️ 所有方法均保持甚至略微提升 utility。

#### ✅ 多属性对齐效果（Table 3）

| 方法 | Avg MAE<sub>true</sub> ↑ | Avg MAE<sub>target</sub> ↓ |
|------|--------------------------|----------------------------|
| No Defense | 0.0182 | 0.0571 |
| DPO | 0.0233 | 0.0204 |
| GRPO | **0.0308** | **0.0199** |

> ✔️ 两种方法都能有效调整多个属性的生成比例，使其趋近于统一目标值 $ r_t=0.05 $；
> ✔️ 初始偏离大的属性（如 digestive disorder $ r_{\text{true}}=0.127 $）被显著修正。

---

### 🔍 消融与扩展实验

#### （1）对抗性提示泛化（Adversarial Prompt Generalization）

使用未见形式的提示（role-play、叙述式）测试防御鲁棒性：

| 方法 | MAE<sub>true</sub> ↑ | MAE<sub>target</sub> ↓ |
|------|----------------------|------------------------|
| No Defense | 0.0422 | 0.1406 |
| DPO | **0.1172** | **0.0575** |
| GRPO | 0.0757 | 0.0862 |

> ✔️ DPO 泛化能力更强，表明其偏好构造更具鲁棒性。

#### （2）关键词-属性关联分析（Table 5）

| 关键词 | No Defense (corr.) | After DPO | After GRPO |
|--------|--------------------|-----------|------------|
| his | -0.9324 → **+0.5414** | → -0.6627 |
| he | -0.9081 → **+0.4464** | → -0.6410 |
| female | +0.7951 → **-0.4494** | → +0.1504 |

> ✔️ 多数强相关关键词（如 his, he）的属性关联显著减弱；
> ✔️ 但仍存在残留关联（如 her 仍强相关），说明未完全解耦。

#### （3）词频分布可视化（Figure 1 & 2）

> ✔️ 经 DPO/GRPO 防御后，“female”、“his” 等词的频率不再随真实属性变化而单调变化，说明其作为 proxy 的有效性下降。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Alignment 可作为有效的 post-training 防御手段**：
   - 利用 DPO 和 GRPO 成功将模型输出分布对齐至目标属性比例，有效干扰属性推断攻击。

2. **良好的 Confidentiality-Utility 权衡**：
   - 攻击 MAE 提升明显（最高达 3–5 倍），而模型 utility 几乎不变。

3. **不仅防御 generation 攻击，也能间接抵抗 shadow-model 攻击**：
   - 因为改变了生成样本的属性分布，进而扰动了 word-frequency 特征的相关性。

4. **GRPO 更精确地逼近目标比例**：
   - 得益于 on-policy 更新机制，能逐步逼近 $ r_t $，对齐误差最小。

5. **DPO 泛化能力更强**：
   - 在对抗性提示下仍能维持较高防御效果。

---

### ⚠️ 方法的局限性

1. **不能完全消除 keyword-attribute 关联**：
   - 某些词（如 her）仍保持较强相关性，可能成为新的攻击入口。

2. **依赖攻击提示的一致性**：
   - 若攻击者使用与防御训练不同的 prompt 分布，防御效果可能下降。

3. **假设条件限制**：
   - 假设 alignment 过程中条件词分布不变，仅重加权样本；若模型学会“伪装”而非真正改变分布，可能存在安全隐患。

4. **仅验证了两种攻击范式**：
   - 当前研究集中在 [Huang et al., 2025] 提出的 generation 和 shadow-model 攻击，未来可能出现更强攻击需进一步评估。

---

### 🔮 未来工作方向

1. **探索更细粒度的 alignment 机制**：
   - 如联合优化多个属性、考虑上下文敏感的 reward 设计。

2. **增强跨 prompt 分布的鲁棒性**：
   - 引入 prompt augmentation 或 domain adaptation 思路提升泛化能力。

3. **结合差分隐私或其他隐私机制**：
   - 构建 hybrid defense，兼顾个体级与分布级隐私保护。

4. **应用于其他模态或任务**：
   - 如图像生成模型中的属性推断防御。

5. **建立标准化的 Property Inference Benchmark**：
   - 推动社区对 LLM 分布级隐私风险的系统性评估。

---

> 💬 **总体评价**：  
> 本论文首次将 **alignment 技术用于防御属性推断攻击**，开辟了 LLM 隐私保护的新路径。其 post-training、无需修改数据的特点极具实用价值，尤其适合医疗、金融等高敏领域中已部署模型的安全加固。

</details>

---

### 7. [From Senses to Decisions: The Information Flow of Auditory and Visual Perception in Multimodal LLMs](https://arxiv.org/abs/2606.10147)

**Authors**: Wish Suharitdamrong, Muhammad Awais, Xiatian Zhu, Sara Atito  
**Category**: cs.AI  
**Published**: 2026-06-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.10147v1  

#### Abstract
Multimodal Large Language Models (MLLMs) can listen and see, but how do audio and visual signals actually travel through the network to shape an answer? Despite their growing role in research and real-world applications, the internal pathways through which audio and visual tokens influence the final...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Senses to Decisions: The Information Flow of Auditory and Visual Perception in Multimodal LLMs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文聚焦于**Audio-Visual Large Language Models (AVLLMs)** 内部的信息流动机制，特别是音频和视觉信号如何在模型中被处理、整合并最终影响决策。尽管 AVLLMs 在多模态任务中表现出色，但其内部的 **information flow**（信息流）路径尚不明确，尤其是在不同输入配置下（如单个音视频 vs 多个交错的音视频项）。

具体而言，论文试图回答以下开放性问题：
- 音频和视觉信息是否遵循与 VLMs 和 VideoLLMs 类似的流动路径？
- 不同模态的贡献如何随任务需求变化？
- 在多输入交错场景中，信息是如何路由的？
- 是否存在可以安全丢弃的冗余 token 以提升推理效率？

### 提出了什么新方法或新思路
论文采用 **mechanistic interpretability** 中的 **Attention Knockout** 技术，系统地追踪 AVLLMs 中 audio 和 visual 信息的流动路径。主要创新点包括：

- **首次对 AVLLMs 的信息流进行系统性分析**，覆盖两种典型输入配置：**audio-visual video** 和 **multiple interleaved audio-visual items**。
- 提出 **“信息聚合点”（aggregation point）** 的概念，即 late-positioned tokens（如 question 或 reference）作为信息汇聚中心。
- 发现 **token 可在信息传递完成后被丢弃**，为高效推理提供新思路。
- 揭示 **attention sinks** 是后期层 attention spikes 的主要原因，并非真正信息流动的标志。

### 相比现有方法的优势
| 方面 | 本文优势 |
|------|----------|
| **分析粒度** | 超越传统性能评测，深入到模型内部机制，揭示跨模态交互的具体路径 |
| **适用范围广** | 结果在多个模型（Qwen2.5-Omni, Video-SALMONN2Plus）、多种规模（3B, 7B）和多个数据集上一致成立 |
| **方法通用性强** | Attention Knockout 方法可推广至其他 MLLMs 的 interpretability 研究 |
| **实用价值高** | 发现 token discard 的可行性，直接支持更高效的 inference 设计 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
| 数据集 | 类型 | 主要用途 |
|--------|------|---------|
| **AV-SpeakerBench** | 单个 audio-visual video | 分析信息流路径，特别是 question-answer 场景下的跨模态推理 |
| **AV-Odyssey** | 多个交错的 audio-visual items | 研究 multi-input 场景中的并行信息流与匹配任务 |
| **WorldSense** | 音视频视频（真实世界场景） | 跨数据集泛化验证，增强结论普适性 |

### 实验设置和评估指标
#### 模型
- **主模型**：`Qwen2.5-Omni`（3B 和 7B）
- **对比模型**：`Video-SALMONN2Plus`（3B 和 7B）

#### 方法
- **Attention Knockout**：通过修改因果掩码（causal mask），阻断特定 token 之间的 attention 连接，测量预测概率的变化 $ \Delta p = (P_{\text{knockout}} - P_{\text{base}})/P_{\text{base}} $
- **滑动窗口分析**：使用大小为 $ k=7 $ 的滑动窗口定位信息流发生的网络深度
- **token discard 实验**：在信息转移完成后的指定层移除 audio/video/question tokens，观察对 accuracy 和 latency 的影响

#### 评估指标
- **相对概率变化 $ \Delta p $**：衡量某条路径的重要性
- **准确率（Accuracy）**：用于评估 token discard 后的性能保持情况
- **Prefill 延迟（Latency）**：评估推理效率提升程度

### 基线方法对比
本文未提出新的训练或架构方法，因此无传统意义上的“基线模型”。但所有实验均以 **原始完整模型（no knockout, no discard）** 作为 baseline，比较以下变体：
- 不同路径的 attention knockout
- 不同 token 类型的 discard（video/audio/non-option question/candidates/reference）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Token Discard 对准确率的影响（来自 Table 4）

| 配置 | Sp. Rec. | Vis. Rec. | Avg. Accuracy | Latency (ms) |
|------|----------|-----------|----------------|---------------|
| Baseline | 50.25 | 46.58 | — | 2288.65 |
| Discard Ques | 50.25 | 46.83 (+0.25) | ≈持平或略升 | ↓ 2279.97 |
| Discard Audio | 50.25 | 47.55 (+0.97) | ↑提升 | ↓ 2232.45 |
| Discard Video | 50.75 (+0.50) | 46.10 (-0.48) | ≈持平 | ↓↓ 2098.75 |
| **Discard All** | 49.75 (-0.50) | 46.59 (+0.01) | **基本不变** | **↓↓ 2089.47** |

> 💡 尽管部分任务略有下降，但总体 accuracy 几乎不受影响，甚至某些任务有轻微上升，而延迟显著降低。

#### ✅ 多输入场景下的 discard 效果（AV-Odyssey）

| 配置 | A→I | I→A | Latency (ms) |
|------|-----|-----|--------------|
| Baseline | 61.00 | 38.00 | 558.75 |
| Discard All | 63.00 | 38.00 | **530.62** |

> 📌 所有 token discard 后，准确率未降反升（尤其 A→I），延迟减少约 **5%**

---

### 与基线方法的对比结果
| 维度 | 本文发现 vs 基线认知 |
|------|------------------|
| **后期 attention ≠ 信息流** | 以往可能误将 attention spikes 当作信息流动证据；本文证明是 **attention sinks** 导致的 artifacts |
| **信息流路径** | 早期研究认为 deep layers 集成跨模态信息；本文发现实际集成发生在 **mid-layers**，deep layers 仅剩 sink artifacts |
| **token 必要性** | 传统做法保留所有输入 token；本文证明可在 transfer 完成后 discard，不影响甚至提升性能 |

---

### 消融实验结果
- **window size ablation**（附录 E）：测试不同 $ k $ 值（1–11），发现 $ k=7 $ 最佳，平衡了路径定位精度与噪声抑制。
- **per-task knockout**（附录 F/G/H/I）：在多个任务和模型上重复实验，结果高度一致，验证了结论的鲁棒性和泛化能力。
- **input order variation**（附录 D.3.1）：即使 prompt 中 media 顺序不同，信息仍通过 reference 聚合，说明 routing 具有结构性。

---

## 4. 关键结论和发现

### 论文的主要发现（Key Findings）

#### 🔹 Finding 1: Attention allocation is not a reliable indicator of information flow
- 后期层（如 layer 31）出现的 video attention spike 实际是由 **vision sink tokens** 引起的机械性 artifact。
- 这些 sink tokens 具有极高的 L2 norm 和特定维度激活模式，类似于已知的 **language sink**。
- **masking 实验表明**：屏蔽这些 attention 边缘不会降低 accuracy，说明它们不承载关键信息。

#### 🔹 Finding 2: 单音视频输入中，信息沿单一串行路径流动
- 流动路径为：**Modalities → Question → Last Token**
- 在 early-to-mid layers，audio 和 visual 信息通过 cross-frame 和 cross-modal attention 进行交互；
- mid layers，信息被转移到 **question tokens**（聚合点）；
- late layers，question 将信息传递给生成的第一个 token（answer letter）。

#### 🔹 Finding 3: 多交错输入中，信息通过两条并行路径流动
- 路径一：**Candidates + Question → Reference → Last**
- 路径二：**Candidates → Option Letters → Last**
- 两个路径分别在 **reference** 和 **correct option letter** 处聚合；
- 最终由 last token 综合两条路径的信息做出决策。

#### 🔹 Finding 4: Tokens can be discarded after information transfer
- 一旦 audio、visual 或 text token 的信息被成功转移至后续 token，即可安全丢弃；
- 每种 token 在不同 layer 完成 transfer（e.g., video ~L26, question ~L29）；
- discard 后 accuracy 基本不变或略有提升，**prefill latency 显著下降**（最高达 ~10%）。

---

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **仅适用于 MCQ 任务** | 实验基于 multiple-choice question answering，open-ended generation（如 captioning）可能涉及不同路径 |
| **依赖 Attention Knockout 的假设** | 该方法假设 attention 是信息流动的主要通道，忽略 MLP 或 residual stream 的作用 |
| **未考虑训练动态** | 分析为 post-hoc interpretability，未解释为何这些 flow structure 会自然形成 |
| **sink tokens 成因未完全揭示** | 虽识别出 sink 现象，但其形成机制仍需进一步研究 |

---

### 未来工作方向
1. **Efficient Inference via Internal Token Compression**
   - 利用 token discard 发现，在 LLM 内部实现动态 token pruning，超越现有的 input-level compression 方法（如 EchoingPixels, OmniZip）。

2. **Steering Modality Reliance**
   - 探索是否可通过干预信息流来引导模型平衡 audio 与 visual 的依赖，缓解 **visual bias** 问题。

3. **Understanding Bias Formation**
   - 将信息流分析扩展至 counterfactual 设置（如 audio-video mismatch），探究 bias 是在哪个 layer 开始主导的。

4. **Extending to Open-ended Generation**
   - 将当前框架应用于 captioning、dialogue 等自由生成任务，探索是否存在类似的 aggregation points。

5. **Model Architecture Design**
   - 基于 flow insights 设计新型 AVLLM 架构，例如显式引入 aggregation modules 或优化 token ordering。

---

> ✅ **总结一句话**：  
> 本论文首次系统描绘了 AVLLMs 中从感知到决策的信息流动图谱，揭示了串行与并行流动路径、attention sinks 的误导性以及 token discard 的可行性，为 MLLMs 的可解释性、设计优化与高效推理开辟了新方向。

</details>

---

### 8. [UniSVQ: 2-bit Unified Scalar-Vector Quantization](https://arxiv.org/abs/2606.10520)

**Authors**: Haoyu Wang, Haiyan Zhao, Xingyu Yu, Zhangyang Yao, Xu Han, Zhiyuan Liu, Maosong Sun  
**Category**: cs.CL  
**Published**: 2026-06-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.10520v1  

#### Abstract
Post-training quantization at the 2-bit level enables low-cost deployment and inference acceleration for large language models (LLMs). Scalar quantization (SQ) and vector quantization (VQ) are two primary quantization methods, however, the former suffers from significant performance degradation, and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：UniSVQ: 2-bit Unified Scalar-Vector Quantization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前大语言模型（LLMs）在进行 **2-bit 后训练量化（Post-Training Quantization, PTQ）** 时面临以下挑战：
- **Scalar Quantization (SQ)**：虽然计算高效，但因量化粒度粗、对异常值敏感，导致严重性能下降（如在零样本任务中性能下降超30%）。
- **Vector Quantization (VQ)**：虽能提升精度，但引入额外的码本（codebook）存储开销，并增加解码复杂度，影响推理吞吐。

### **提出的新方法与新思路**
作者提出了 **UniSVQ**，一种统一的 2-bit 量化框架，融合了 SQ 和 VQ 的优势：
- 引入 **仿射变换参数化的线性约束量化网格（linear-constrained quantization grid）**，将码本结构化为整数格点的仿射变换形式：  
  $$
  \text{dequant}(w) = A \cdot w_{\text{int}} + B
  $$
  其中 $A$ 是仿射矩阵，$B$ 是偏置向量。
- 这种结构既保留了 VQ 的灵活性（可拟合权重分布），又具备 SQ 的简洁性（兼容优化的整数量化 Matmul 内核）。
- 提出 **分块自适应微调策略**，通过反向传播直接最小化重建误差，进一步提升量化质量。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **性能** | 显著优于 SOTA SQ 方法，接近甚至超越先进 VQ 方法 |
| **效率** | 仅需每权重矩阵 **20 个额外浮点参数**（$A \in \mathbb{R}^{4\times4}, B \in \mathbb{R}^4$），码本存储减少约 **64倍** |
| **部署友好性** | 可复用高度优化的 SQ 推理内核，无需复杂解码流程 |
| **扩展性** | 支持不同 bit-width（如 3-bit），且无需重构码本设计 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **校准/微调数据**：从 RedPajama 数据集中随机采样 1,024 条序列（长度为 2,048）
- **评估数据集**：
  - **PPL（困惑度）**：WikiText-2、C4
  - **Zero-shot QA 准确率**：ARC-Easy、ARC-Challenge、BoolQ、HellaSwag、PIQA、WinoGrande

### **实验设置与评估指标**
- **模型家族**：Qwen-3（4B~32B）、Llama-3-8B-Instruct
- **量化位宽**：主实验为 2-bit，部分验证 3-bit 扩展性
- **量化维度 $d$**：默认设为 4（即每次处理 4 个连续权重）
- **评估指标**：
  - PPL（越低越好）
  - Zero-shot 平均准确率（AVG↑）
  - 相对于 FP16 模型的性能比率（PER.）
  - 推理吞吐（tokens/s）、峰值 GPU 内存（GMem）

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Scalar Quantization (SQ)** | GPTQ、QuIP、SpinQuant、OSTQuant |
| **Vector Quantization (VQ)** | AQLM（聚类法）、QuIP#（基于 E8 lattice） |
| **其他 VQ 对比**（附录） | QTIP（trellis-coded） |

所有方法使用相同校准数据以确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（以 Qwen-3-32B 为例）**
| 方法 | PPL (WikiText) | AVG (Zero-shot) | PER. |
|------|----------------|------------------|-------|
| FP16 | 7.61 | 78.01 | 1.00 |
| GPTQ (SQ) | 1.38E4 | 36.09 | 0.46 |
| OSTQuant (SQ) | 14.79 | 68.29 | 0.88 |
| QuIP# (VQ) | 9.04 | 76.30 | 0.98 |
| **UniSVQ (Ours)** | **9.26** | **76.15** | **0.98** |

> ✅ UniSVQ 在 PPL 和 Zero-shot 性能上全面超越所有 SQ 方法，与最强 VQ 方法（QuIP#）相当。

### **与其他模型规模的结果一致性**
- 在 Qwen-3-4B 到 32B 上，UniSVQ 均显著优于 SQ 方法（平均 PER. 提升 0.4~0.5）
- 在多个模型上达到或超过 AQLM 和 QuIP# 的性能
- **特别发现**：2-bit UniSVQ-Qwen-3-32B 的平均 QA 准确率 **超过 FP16-Qwen-3-4B**，表明更大规模的低比特模型更具实用价值

### **消融实验结果**

#### **(1) 是否微调（Fine-tuning）**
| 设置 | AVG (Qwen-3-8B) | PER. |
|------|------------------|-------|
| 完整 UniSVQ | 67.95 | 0.92 |
| 无微调（w/o fine-tuning） | 66.99 | 0.90 |

> 微调带来约 **+1% 绝对准确率提升**，证明其有效性。

#### **(2) 初始化方式对比**
| 初始化方式 | AVG | PER. |
|------------|-----|------|
| 随机正交矩阵（Proposed） | 67.95 | 0.98 |
| D4 lattice 生成矩阵 | 61.00 | 0.82 |

> 尽管 D4 lattice 数学上更优，但在线性约束下反而表现差，说明 **随机正交初始化更适合该框架**。

#### **(3) 是否使用 RHT（Randomized Hadamard Transform）**
| 方法 | PPL (WikiText) | AVG |
|------|----------------|------|
| UniSVQ（完整） | 20.04 | 63.42 |
| UniSVQ w/o RHT | 8314.12 | 39.06 |

> 移除 RHT 导致性能崩溃至近随机水平，验证其作为 **必要预处理步骤的重要性**。

#### **(4) 向量维度 $d$ 影响**
| $d$ | Fine-tuning | AVG |
|-----|-------------|-----|
| 4 | 否 | 66.99 |
| 8 | 否 | 67.34 |
| 4 | 是 | **67.95** |

> 增加维度带来的增益有限，远小于微调带来的收益；综合考虑效率，选择 $d=4$ 更合理。

---

## **4. 关键结论和发现**

### **主要发现**
1. **线性约束量化网格是连接 SQ 与 VQ 的桥梁**：
   - 既能获得 VQ 级别的灵活性，又能保持 SQ 的高效推理结构。
2. **极低辅助参数即可实现高性能**：
   - 每层仅需 20 个额外参数，却能达到与复杂 VQ 方法相媲美的性能。
3. **推理效率显著提升**：
   - 在 Llama-3-8B 上，UniSVQ 实现 **1.68× 推理加速** 和 **>75% GPU 内存节省**，优于 AQLM 和 QuIP#。
4. **跨架构泛化能力强**：
   - 在 Qwen 和 Llama 架构上均取得一致优异表现。

### **方法的局限性**
- 当前仅支持 **weight-only quantization**，未涉及激活值（activation）或 KV Cache 的量化，限制端到端推理优化潜力。
- 虽然兼容现有 GEMM 内核，但 **仿射变换与高度优化 kernel 的协同效应仍待深入探索**。
- 依赖 RHT 预处理，增加了前向计算的一次性开销。

### **未来工作方向**
- 扩展至 **activation 和 KV-cache 的联合量化**
- 探索 **动态仿射参数学习机制**
- 研究更高维度（如 $d=8$）结合稀疏化或分组策略的可能性
- 将 UniSVQ 思路应用于 **训练时量化（QAT）**

---

> 📌 **总结一句话**：  
> **UniSVQ 成功弥合了 Scalar 与 Vector Quantization 的鸿沟，在几乎不增加部署成本的前提下，实现了 2-bit 量化的性能飞跃，是迈向高效大模型部署的重要一步。**

</details>

---

### 9. [Bellman-Taylor Score Decoding for Markov Decision Processes with State-Dependent Feasible Action Sets](https://arxiv.org/abs/2606.10979)

**Authors**: Yi Chen (Lucy), Rushuai Yang (Lucy), Qiang Chen (Lucy),  Dongyan (Lucy),  Huo  
**Category**: cs.AI  
**Published**: 2026-06-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.10979v1  

#### Abstract
Many Markov decision processes (MDPs) in operations research have feasible actions that are state dependent and defined implicitly by various operational constraints. These features make it difficult to use standard deep reinforcement learning (DRL) algorithms, whose action interfaces typically assu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Bellman-Taylor Score Decoding for Markov Decision Processes with State-Dependent Feasible Action Sets*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
许多运筹学（OR）中的**Markov Decision Process (MDP)** 问题具有**状态依赖的可行动作集**（state-dependent feasible action sets），这些动作通常由容量、兼容性和整数约束等隐式定义，且维度高、组合性强。标准的**Deep Reinforcement Learning (DRL)** 算法难以直接应用，因为它们通常假设动作空间是固定的有限集合或简单的欧几里得向量空间。

### 提出的新方法：Bellman-Taylor Score Decoding
作者提出了一种新的动作接口框架——**Bellman-Taylor Score Decoding (BTSD)**，其核心思想是：
- 将策略学习从原始复杂的自然动作空间转移到一个**连续的欧几里得分数空间**（latent score space）。
- 引入一个**action decoder**，将学到的分数向量映射为满足所有约束的可行自然动作。该 decoder 基于对最优 action-value 函数 $ Q^*(s,a) $ 的**Taylor 展开近似**，其中分数 $ z $ 被解释为系统后置状态（post-action configuration）对未来价值影响的边际值梯度。

具体流程如下：
1. 策略网络输出一个**连续的分数向量 $ z \in \mathbb{R}^d $**；
2. 通过优化解码器 $ \mathcal{I}(s,z) = \arg\max_{a \in \mathcal{A}(s)} \left\{ w_s(a) + \langle z, \phi_s(a) \rangle \right\} $ 得到最终动作；
3. 在训练过程中，**不需要对 decoder 进行微分**（non-differentiable），仅需在前向传播中调用求解器。

### 相比现有方法的优势
| 方法类别 | 局限性 | BTSD 的优势 |
|--------|------|-----------|
| **Action Embedding / Masking** | 需要显式枚举动作或预定义掩码，不适用于大规模/隐式约束 | 不依赖动作枚举，decoder 可处理任意复杂约束 |
| **Differentiable Optimization Layers** | 需要可微优化层或代理梯度，难以处理整数/组合动作 | decoder 不参与反向传播，避免了不可微问题 |
| **Custom Architectures / Decomposition** | 高度依赖问题定制设计，通用性差 | 提供标准化接口，可复用于多种 OR 问题 |
| **Value-based with MIP** | 每步需解复杂 MIP，计算开销大 | 分离学习与可行性，decoder 可灵活实现 |

> ✅ **核心优势**：实现了**off-the-shelf DRL 算法**（如 PPO）与**复杂约束 MDP**之间的“即插即用”桥接，无需修改 DRL 算法本身。

---

## 2. 核心实验方法和设置

### 实验场景
论文在两个典型运筹学问题上验证方法有效性：

#### （1）库存控制问题（Sanity Check）
- **模型**：多地点库存系统，支持跨地调拨（transshipment），存在接收损耗（congestion effect）。
- **状态**：各位置库存水平 $ s_i $
- **动作**：补货量 $ u_i $ 和调拨矩阵 $ y_{ij} $，受库存上限和整数约束
- **目的**：验证 BTSD 能否处理状态依赖整数动作

#### （2）排队网络控制问题（主案例研究）
- **模型**：多类客户、多服务池的并行服务系统（multi-class, multi-pool queueing network）
- **状态**：队列长度 $ q_{it} $ 和正在服务的数量 $ h_{ijt} $
- **动作**：调度矩阵 $ a_{ijt} \in \mathbb{Z}_+ $，表示将多少类 $ i $ 客户分配给池 $ j $
- **约束**：不能超过当前队列数量和服务器容量
- **目标**：最小化折扣总成本（holding cost + overflow cost）

### 实验设置
- **算法实现**：
  - 主算法：**BTSD-PPO** —— 使用 PPO 学习 latent score
  - 对比算法见下文
- **评估指标**：
  - **Optimality Gap**：相对于最优策略的成本差距
  - **Cost / Expected Discounted Cost**：平均累积成本
  - **Bellman Regret**：$ \mathbb{E}[Q^*(s,a_\pi(s)) - V^*(s)] $
  - **Action Agreement Rate**：与最优动作一致的比例
- **消融实验**：测试不同 DRL backbone（PPO/SAC/DQN）配合 BTSD 的效果

### 基线方法对比
| 类型 | 方法名称 | 描述 |
|-----|---------|------|
| **经典启发式** | cu rule, modified cu, max-weight, modified max-weight | 手工设计的状态依赖索引规则 |
| **RL-based** | Atom-PPO | 将组合动作分解为原子动作序列决策（autoregressive policy） |
| **ADP-based** | MIP-based lookahead | 结合 value approximation 与数学规划进行每步优化（Harsha et al., 2025） |
| **Vanilla DRL** | Vanilla PPO/SAC/DQN + masking | 直接参数化策略并使用 action masking 处理可行性 |

---

## 3. 主要实验结果和性能指标

### （1）小规模排队系统（2×2） vs 最优解
| 设置 | Exact Optimal | BTSD-PPO | Gap |
|------|--------------|----------|-----|
| 平衡负载 $ (\rho_1=0.9,\rho_2=0.9) $ | ~1210–1285 | ~1226–1317 | **≤2.5%** |
| 不平衡负载 $ (\rho_1=0.95,\rho_2=0.85) $ | ~1185–1265 | ~1198–1278 | **≤1.3%** |

✅ **结论**：在可计算最优的小系统中，BTSD-PPO 接近全局最优，验证了方法的有效性。

---

### （2）中等规模排队系统（5×5） vs 各类基线（Table 3）
共测试 9 个实例（3 种流量模式 × 3 成本配置），结果摘要如下：

| 方法 | 平均成本范围 | 改进幅度（vs 最佳基线） |
|------|-------------|------------------------|
| **Best Heuristic (e.g., mod. cu)** | 233 – 954 | — |
| **Atom-PPO** | 289 – 758 | 明显劣于 BTSD |
| **MIP-based** | 272 – 904 | 表现不稳定 |
| **BTSD-PPO** | **196 – 548** | **+4.0% ~ +23.7% 更优** |

📌 **关键发现**：
- BTSD-PPO 在所有 9 个案例中均取得最低成本；
- 改进最大达 **23.7%**（Case T3）；
- 固定启发式（如 cu-rule）表现随参数变化剧烈，而 BTSD 自动学习适应性索引更鲁棒。

---

### （3）消融实验（Ablation Study, Table 4）
测试 BTSD 是否提升不同 DRL 算法的表现：

| DRL Backbone | Vanilla 版本成本 | BTSD 版本成本 | 成本下降 |
|--------------|------------------|---------------|----------|
| PPO          | 397.2           | 222.2         | **↓44.1%** |
| SAC          | 417.0           | 230.2         | **↓44.8%** |
| DQN          | 393.1           | 228.9         | **↓41.8%** |

✅ **结论**：**BTSD 是性能增益的关键来源**，而非特定 DRL 算法的选择；它显著提升了各类 backbone 的表现，并超越最佳启发式。

---

### （4）高阶扩展（Higher-Order Decoder）在库存问题上的表现（Table 1）
当系统动态非线性强时（接收效率递减），高阶 decoder 更有效：

| 接收损耗参数 $ p $ | First-order Optimality Gap | Second-order Optimality Gap |
|--------------------|-------------------------------|------------------------------|
| 0.0                | 1.0%                          | 0.2%                         |
| 0.5                | 10.5%                         | 0.4%                         |
| 0.75               | 11.4%                         | 0.6%                         |

✅ **结论**：第一阶 decoder 在近线性系统中足够高效；第二阶 decoder 显著改善强非线性系统的性能，体现框架的灵活性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **BTSD 成功桥接了标准 DRL 与复杂 OR-MDP**：通过引入 score-decoding 机制，使 PPO 等通用算法能直接应用于状态依赖、组合性动作问题。
2. ✅ **decoder 本质是学习一种状态依赖的 index rule**：所学分数 $ z(s) $ 相当于为每个 (class, pool) 对动态生成优先级索引，优于固定公式的手工索引。
3. ✅ **性能优越且稳定**：在小系统接近最优，在大系统显著优于经典启发式、原子动作 RL 和 MIP-based ADP。
4. ✅ **模块化设计带来泛化能力**：同一框架可用于库存、排队等多种问题，且兼容多种 DRL 算法。

### 方法的局限性
- ⚠️ **额外计算开销**：每次动作选择需调用一次优化求解器（如 MIP/IP），增加推理延迟。
- ⚠️ **表达能力受限**：score-decoding policy class 是原 policy space 的子集，无法表示所有可能策略。
- ⚠️ **依赖结构化建模**：需要明确定义 post-action configuration $ \phi_s(a) $ 和 transition function $ \mathcal{T}_s(\cdot,\cdot) $，不适合纯黑箱环境。
- ⚠️ **理论误差分解依赖理想假设**：performance bound 中的 residual oscillation 项依赖于最优值函数的知识，实际难以精确估计。

### 未来工作方向
1. 🔄 **扩展至 Average-Cost MDP**：将 discounted continuation value 替换为 relative value function，以适应长期平均成本场景。
2. 🔍 **结合 Offline RL**：利用历史轨迹数据训练，提高样本效率，适用于仿真昂贵或现实探索受限的系统。
3. 🧠 **学习 Post-action Representation**：当前 $ \phi_s(a) $ 需人工设计，未来可尝试从数据中自动学习有效的中间状态表示。
4. 💡 **探索更高效的 decoder 架构**：例如使用近似求解器加速 inference，或结合 warm-start 提升实时性。

---

> **总结一句话**：  
> **Bellman-Taylor Score Decoding 提供了一个强大而通用的接口，让标准 DRL 算法能够“看懂”运筹学问题中的复杂动作约束，并通过学习动态索引实现接近最优甚至超越传统方法的决策性能。**

</details>

---

### 10. [Small Data, Big Noise: Adversarial Training for Robust Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2606.10610)

**Authors**: Eitan Cohen, Idan Simai, Uri Shaham  
**Category**: cs.CL  
**Published**: 2026-06-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.10610v1  

#### Abstract
Parameter-Efficient Fine-Tuning (PEFT) has become essential for adapting foundation models to downstream NLP tasks. However, current PEFT methods often struggle with robustness to noise and performance degradation on limited training data. We propose SDBN (Small Data Big Noise), a unified framework ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Small Data, Big Noise: Adversarial Training for Robust Parameter-Efficient Fine-Tuning**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Parameter-Efficient Fine-Tuning (PEFT)** 方法（如 LoRA、Adapter、BitFit）虽然在计算和存储上高效，但在以下两个现实场景中表现脆弱：
- **低资源训练数据**（small data）：当训练样本有限时，模型容易过拟合，泛化能力差。
- **输入噪声**（big noise）：真实文本常含拼写错误、格式不一致、方言变异等，导致 tokenization 被破坏或语义偏移。

这些问题在医疗、航空航天、小语种等领域尤为突出，而现有 PEFT 方法并未专门针对这些挑战进行优化。

---

### 🚀 提出的新方法：SDBN 框架
作者提出 **SDBN (Small Data Big Noise)** ——一个将 **Adversarial Training** 与 **PEFT** 结合的统一框架，旨在提升模型在低资源和噪声环境下的鲁棒性。

#### 主要创新点：
1. **首次系统地将对抗训练引入 PEFT 范式**  
   尽管对抗训练在全量微调中已有应用，但在 PEFT 设置下研究较少。SDBN 成功将其适配至参数高效的设定中，且**不增加可训练参数**。

2. **三种不确定性集合（uncertainty sets）设计**：
   - **SDBN**：基于嵌入空间的连续扰动（`l∞` norm ball），适用于一般噪声。
   - **SDBN-h**：基于**离散字符级编辑**的对抗变体选择，解决 tokenization-breaking 错误（如删字母导致分词失败）。
   - **SDBN-p**：利用 **LLM 生成语义保持但具挑战性的变体**，专为生成任务设计。

3. **无需额外标注数据即可应对域迁移（domain shift）**  
   通过对抗训练扩展输入的“不确定性区域”，使模型能泛化到未见过的目标领域（如从 Yelp 迁移到 Amazon）。

---

### 🔍 相比现有方法的优势
| 对比维度 | SDBN 优势 |
|--------|----------|
| **鲁棒性** | 显著优于 vanilla PEFT 和随机噪声注入方法（如 NEFTune、EDA） |
| **参数效率** | 不引入额外可训练参数，保持原有 PEFT 的轻量化特性 |
| **适用范围广** | 支持分类与生成任务，兼容多种 PEFT 方法（LoRA, BitFit, Adapter, QLoRA） |
| **实际部署友好** | 内存开销极小（<0.15%），运行时间可控（约增加 1.3–1.4×） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### 分类任务：
- **BANKING77**：银行客服意图识别
- **TREC**：问题分类
- **20NEWSGROUPS**：新闻主题分类
- **IMDB**：情感分析
- **BLESS**：词义关系分类（用于字符级噪声测试）

#### 生成任务：
- **SQuAD**：抽取式问答
- **TweetQA**：社交媒体问答

#### 域迁移任务：
- **ArSarcasm-v2**：阿拉伯语方言情感分类（跨方言迁移）
- **NLI (Cross-genre)**：自然语言推理任务中的跨领域泛化（小说 → 电话对话）

---

### ⚙️ 实验设置

| 项目 | 设置说明 |
|------|---------|
| **预训练模型** | BERT-base, DeBERTa-v3, LLaMA-3.2-1B, LLaMA-2-7B, Qwen-2.5-7B |
| **PEFT 方法** | LoRA, BitFit, Adapter, QLoRA |
| **对抗训练策略** | SDBN（默认）、SDBN-h（字符级）、SDBN-p（LLM生成） |
| **基线方法** | Vanilla PEFT, NEFTune, EDA, FreeLB, SMART |
| **训练数据比例** | 5% ~ 100%，模拟低资源场景 |
| **噪声类型** | 见下表（Table 6） |
| **评估指标** | 准确率（Accuracy）、Exact Match（EM）、F1 Score |
| **重复实验** | 多次随机种子运行（5~10次），报告均值±标准差 |

#### 测试阶段施加的噪声类型（Table 6）：
| 噪声类型 | 示例 |
|--------|------|
| `Delete Word`, `Swap Words`, `Replace Word` | 删除/交换/替换词语 |
| `Homophone`, `Cyrillic`, `Keyboard-char` | 同音词、西里尔字母混淆、键盘邻键错 |
| `Delete Char`, `Insert Char`, `Phonetic Replacement` | 字符级扰动 |
| `Case`, `Slang`, `Pronoun` | 大小写、俚语、代词替换 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ 在低资源 + 噪声环境下显著领先
- **图4 & 图7**：在 BANKING77 上使用 1,000 样本训练，SDBN 在各种 word-level 噪声下平均比 vanilla LoRA 高 **15–20% 绝对准确率**。
- **表1（BLESS 数据集）**：在字符级噪声下，SDBN-h 相比 vanilla 提升达 **+4% ~ +7%**，尤其在 `Delete-char` 和 `Swap-char` 上效果明显。
- **表2（生成任务）**：
  - **SQuAD**：SDBN-p 在 `Swap-Word` 下 EM 达 **35.08%**（vanilla: 32.44%）
  - **TweetQA**：SDBN-p 在 `Delete-Char` 下 F1 达 **65.55%**（vanilla: 51.56%），提升超 **14个百分点**

#### ✅ 清洁数据上也保持甚至提升性能
- **图3**：随着训练数据减少（从 10,000 到 500），SDBN 相对于 baseline 的增益持续扩大，在仅 500 样本时可达 **+10%~15%**。
- 表明对抗训练不仅增强鲁棒性，还提升了泛化能力。

#### ✅ 跨域迁移能力强
- **图14（ArSarcasm-v2）**：在未见阿拉伯语方言上的分类准确率，SDBN 明显优于 vanilla 和 NEFTune。
- **图15（NLI 跨领域）**：即使只用 77 个源域样本，SDBN 在目标域仍保持高性能，验证其对未知域偏移的有效防御。

---

### 🔬 消融实验结果

#### （1）对抗训练对 PEFT 更有效 than 全量微调
- **表3**：在低资源 BANKING77 上：
  - SDBN 对 LoRA 提升 **+19.6% 平均增益**
  - 对 Full Fine-Tuning 仅提升 **+0.9%**
- **解释**：PEFT 参数受限，使得对抗信号更集中、不易过拟合；而全量微调易记住对抗样本，反而损害泛化。

#### （2）不同扰动范数比较（`l₁`, `l₂`, `l∞`）
- **表4 & C.6**：`l∞` norm（ε=1e-4）表现最佳，因其模拟现实中均匀分布的小幅扰动。
- `l₁` 和 `l₂` 效果较差，说明稀疏或方向性强的扰动不适合 NLP 场景。

#### （3）扰动位置影响巨大
- **表5**：只有在 **embedding layer** 注入噪声才有效。
- 若在 encoder 中间层添加扰动，性能急剧下降（<7% 准确率），因为破坏了高层语义表示。

#### （4）SDBN-p 动态重采样更优
- 在 TweetQA 中每轮重新生成 LLM 变体，比固定变体获得更强鲁棒性，表明多样性对抗监督的重要性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **对抗训练是提升 PEFT 鲁棒性的强有力工具**  
   特别是在低资源和噪声共存的实际场景中，SDBN 显著优于传统方法。

2. **梯度引导的对抗扰动 > 随机数据增强**  
   NEFTune 和 EDA 使用随机噪声，虽有一定帮助，但缺乏针对性；SDBN 通过最大化损失方向构造“最坏情况”样本，迫使模型学习更稳定的决策边界。

3. **tokenization-breaking 字符错误需特殊处理**  
   嵌入空间的连续扰动无法覆盖此类极端变化，因此 SDBN-h 引入离散字符编辑并用梯度选择最危险变体，成功弥补这一短板。

4. **LLM 生成的对抗变体适合生成任务**  
   SDBN-p 利用 GPT-5.2 生成多样化、语义合理但具挑战性的输入，极大增强了生成模型面对 paraphrase、typo 等复杂扰动的能力。

5. **SDBN 泛化至未见域迁移**  
   即使没有目标域数据，也能通过对抗训练隐式覆盖潜在语义空间，实现零样本域适应。

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **计算开销增加** | 每 batch 多一次前向-反向传播，runtime 增长约 30–40%（见表11） |
| **ε 选择敏感** | 扰动幅度需仔细调参，过大破坏语义，过小无效；目前依赖经验或简单搜索 |
| **离线生成成本高** | SDBN-p 需预先用 LLM 为每个训练样本生成多个变体，前期耗时较长 |
| **未探索更多结构化扰动** | 如句法改写、逻辑反转等高级语义扰动尚未纳入框架 |

---

### 🔮 未来工作方向

1. **自动化 ε 调整机制**：开发自适应扰动强度算法，避免手动调参。
2. **更高效的离散对抗搜索**：结合强化学习或检索技术加速 worst-case variant 发现。
3. **扩展至多模态任务**：将 SDBN 应用于 Vision-Language 模型中的 PEFT（如 AdvLoRA 的 NLP 方向延伸）。
4. **动态在线对抗生成**：在训练过程中实时生成更具挑战性的对抗样本，而非依赖静态集合。
5. **理论分析 PEFT + 对抗训练的协同效应**：为何在小参数空间中对抗信号更有效？是否与 implicit regularization 有关？

---

## 总结

> **SDBN 是首个将对抗训练深度整合进 PEFT 范式的鲁棒训练框架**，它在不牺牲参数效率的前提下，显著提升了模型在低资源、噪声输入和域迁移下的稳定性与泛化能力。其提出的 **SDBN-h** 和 **SDBN-p** 两种离散不确定性构建策略，分别解决了字符级断词难题和生成任务多样性需求，具有很强的实用价值和推广潜力。

该项目代码已开源：[GitHub - shaham-lab/SDBN](https://github.com/shaham-lab/SDBN)

</details>

---

### 11. [MODIP: Efficient Model-Based Optimization for Diffusion Policies](https://arxiv.org/abs/2606.10825)

**Authors**: Zakariae El Asri, Philippe Gratias-Quiquandon, Nicolas Thome, Olivier Sigaud  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.10825v1  

#### Abstract
Diffusion policies (DPs) have emerged as expressive policy representations for robot learning, often used with imitation learning methods such as behavioral cloning (BC). However, while their success has largely been confined to BC, direct reinforcement learning (RL) fine-tuning remains challenging ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MODIP: Efficient Model-Based Optimization for Diffusion Policies

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **扩散策略（Diffusion Policies, DPs）在直接强化学习（RL）微调时面临挑战**：由于DP通过多步去噪过程生成动作，直接应用RL进行优化会导致训练不稳定、计算成本高。
- **现有模型预测控制（MPC）方法在结合DP时推理效率低**：传统混合规划器使用 $Q(s, \pi(s))$ 形式的终端估计，当 $\pi$ 是DP时需反复执行昂贵的去噪过程，显著增加推理时间。

### 🚀 提出的新方法：MODIP
MODIP（**MOdel-based Distillation for ImProvement of Diffusion policies**）是一种用于**从离线到在线微调扩散策略**的高效框架，其核心思想是：
> 不直接对DP进行RL优化，而是利用一个基于世界模型（World Model, WM）的MPC规划器生成高质量轨迹，并将这些轨迹作为监督目标，通过行为克隆（Behavioral Cloning, BC）方式蒸馏回DP中。

该方法结合了**模型预测控制**与**监督学习**的优点，在保持DP训练稳定性的同时实现性能提升。

### 🔍 创新点
1. **Hybrid DP-guided MPC Planning**  
   - 将DP用作MPC中的**多模态动作序列先验**，引导采样过程。
   - 在潜在空间中使用Model Predictive Path Integral (MPPI) 进行轨迹优化，融合DP生成的候选动作与随机探索。

2. **MPC-to-Policy Distillation**  
   - 使用MPC规划器生成改进后的轨迹，再以标准的去噪损失（denoising loss）对DP进行BC式微调。
   - 实现“更强的规划器 → 更好的数据分布 → 更优的策略”闭环迭代。

3. **Efficient Value-based Trajectory Scoring with $V(s)$**  
   - 改用**终端状态值函数 $V(z)$** 替代传统的 $Q(z, \pi(z))$，避免在每个规划步骤中调用DP进行去噪。
   - 显著降低推理延迟（实测达 **2.9× 加速**），同时提升性能。

4. **Policy-Independent Critic Learning**  
   - 批评家（critic）更新时采用目标网络计算的 $V(z')$ 构建TD目标，而非依赖当前DP输出的动作。
   - 避免频繁查询DP，减少训练开销（总训练时间缩短约 **1.6×**）。

### ⚖️ 相比现有方法的优势
| 方面 | MODIP优势 |
|------|----------|
| **训练稳定性** | 保留BC的稳定监督目标，避免RL带来的方差和不稳定性 |
| **计算效率** | 使用 $V(s)$ 和 policy-independent critic 大幅降低推理与训练成本 |
| **表达能力** | 利用DP的多模态特性作为强先验，优于Gaussian MLP策略 |
| **通用性** | 可兼容多种任务类型（稠密/稀疏奖励、长周期操作等） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验涵盖三大类基准任务：
- **D4RL/MuJoCo**：连续控制任务，如 `HalfCheetah`, `Walker2d`, `Hopper`（稠密奖励）
- **D4RL/Kitchen**：长周期机器人操作任务，要求完成多个子任务（稀疏奖励）
- **RoboMimic**：基于状态的精确多阶段操作任务，如 `Lift`, `Can`, `Square`

所有任务均使用**离线数据集初始化 + 在线微调**范式。

### 🎯 实验设置与评估指标
| 设置项 | 描述 |
|-------|------|
| **训练阶段** | 分两阶段：<br>1. 离线预训练（DP + 世界模型 + 批评家）<br>2. 在线微调（MPC收集轨迹 + 混合回放缓冲区更新） |
| **数据混合策略** | 采用 RLPD 的动态混合机制：逐步增加在线样本比例，防止早期过拟合噪声数据 |
| **评估指标** |<ul><li>MuJoCo：累计回报（Return）</li><li>Kitchen / RoboMimic：任务成功率（Success Rate）</li></ul> |
| **计算预算** | 固定环境步数（通常为 $10^6$ 步），确保公平比较 |

### 🆚 对比的基线方法
| 类型 | 基线名称 | 简要说明 |
|------|--------|---------|
| **纯BC方法** | BC | 行为克隆训练的扩散策略（初始策略） |
| **DP + RL 微调** | DQL, DPPO, DSRL | 将DP与Q-learning、PPO或潜空间RL结合 |
| **批评家引导优化** | PA-RL | 通过critic优化动作后蒸馏回策略 |
| **强模型基线** | TD-MPC2 | 当前最先进的潜空间MPC方法，使用Gaussian MLP策略 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Benchmark | Task | BC | DPPO | PA-RL | TD-MPC2 | **MODIP (Ours)** |
|----------|------|----|------|--------|---------|------------------|
| D4RL/MuJoCo | HalfCheetah | 5108 | 4290 | 14254 | 12054 | **13775** |
|             | Walker2d    | 5721 | 2848 | 5361  | 5433  | **6081** |
|             | Hopper      | 3050 | 1612 | 2855  | 2129  | **3281** |
| D4RL/Kitchen | Complete   | 0.41 | 0.88 | 0.85  | 0.65  | **0.94** |
|              | Partial    | 0.32 | 0.67 | 0.93  | 0.55  | **0.98** |
| RoboMimic     | Lift       | 0.95 | 0.99 | 0.99  | 0.00  | 0.98 |
|               | Can        | 0.87 | 0.93 | 0.90  | 0.00  | **0.92** |
|               | Square     | 0.62 | 0.73 | 0.81  | 0.00  | **0.85** |

> ✅ MODIP在多数任务上达到**最优或次优表现**，尤其在MuJoCo和Kitchen任务中显著领先。

### 🔍 与基线方法的对比分析
- **相比DP直接RL微调方法（DQL, DPPO, DSRL）**：
  - MODIP全面超越，特别是在HalfCheetah上高出近 **3倍以上**。
  - 表明**间接蒸馏路径比直接RL优化更有效且稳定**。
- **相比PA-RL（同属“优化+蒸馏”范式）**：
  - MODIP在HalfCheetah和Kitchen任务上略胜一筹，说明**MPC比critic-based优化更能生成高质量轨迹**。
- **相比TD-MPC2（最强模型基线）**：
  - MODIP使用DP作为prior显著优于其使用的MLP策略，在所有MuJoCo任务中均取得更高回报。
  - 证明**DP作为表达力更强的动作先验能提升MPC效果**。

### 🔧 消融实验结果（Table 2 & Table 3 & Table 4）

#### ✅ 组件消融（Table 2）
| 变体 | HalfCheetah | Kitchen-Complete |
|------|-------------|------------------|
| Full MODIP | **13775** | **0.94** |
| No MPC refinement (policy-only) | 1178.7 | 0.77 |
| Use $Q(s,\pi(s))$ instead of $V(s)$ | 7113 | 0.55 |
| Replace DP prior with MLP policy | 12413 | 0.45 |
| Without offline pretraining | 11736 | 0.74 |

> 结果表明：**MPC refinement、$V(s)$ 设计、DP prior 和 offline init 均对最终性能至关重要**。

#### ⏱ 推理效率对比（Table 3）
| 终端估计方式 | 推理时间（秒 / $10^5$ 环境步） |
|------------|-------------------------------|
| $Q(z, \pi(z))$ | 85.75 |
| $V(z)$ (**MODIP**) | **29.54** |

> ✅ 使用 $V(s)$ 实现 **2.9× 推理加速**，极大提升实用性。

#### ⏳ 训练效率对比（Table 4）
| 变体 | 总训练时间 |
|------|-----------|
| Policy-coupled critic | 40h 16m |
| **Policy-independent critic (MODIP)** | **25h 16m** |

> ✅ 减少约 **1.6× 训练时间**，主要节省在MPC预训练和fine-tuning阶段。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MPC-to-Policy Distillation 是一种高效且稳定的扩散策略改进机制**  
   - 无需改变DP的监督训练形式，即可借助更强的规划器持续提升性能。
   
2. **DP作为MPC的动作先验具有显著优势**  
   - 相比Gaussian MLP策略，DP能提供更丰富、多模态的动作建议，提升搜索效率。

3. **使用 $V(s)$ 而非 $Q(s, \pi(s))$ 至关重要**  
   - 不仅提升推理速度，还改善了轨迹评分质量，带来性能增益。

4. **解耦critic学习与当前策略可大幅提升训练效率**  
   - 避免频繁调用DP进行action sampling，使整个系统更具可扩展性。

5. **MODIP在多种任务上达到SOTA水平**  
   - 在MuJoCo、Kitchen、RoboMimic三大基准上均表现出色，验证了其泛化能力。

### ⚠ 局限性
- **目前仅适用于状态输入（state-based）任务**，尚未扩展至图像输入（pixel-based）。
- **依赖高质量的世界模型建模**，若潜空间动力学不准可能导致规划失败。
- **仍需要一定量的在线交互**，不适合完全无交互场景。

### 🔮 未来工作方向
1. **扩展至视觉输入（pixels）**：结合vision encoder，应用于真实机器人视觉控制。
2. **进一步压缩MPC计算开销**：探索轻量化MPPI或替代优化算法。
3. **集成更多探索机制**：增强在线阶段的探索能力，适应更复杂任务。
4. **研究完全离线版本**：尝试在不进行在线交互的情况下完成策略提升。

---

> 💬 **总结一句话**：  
> **MODIP提出了一种高效、稳定且高性能的方式，通过MPC规划与BC蒸馏相结合，成功实现了对扩散策略的离线到在线微调，解决了DP难以直接RL优化的问题，并在效率与性能之间取得了优异平衡。**

</details>

---

### 12. [Flash-GMM: A Memory-Efficient Kernel for Scalable Soft Clustering](https://arxiv.org/abs/2606.10896)

**Authors**: Gal Bloch, Ariel Gera, Matan Orbach, Ohad Eytan, Assaf Toledo  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.10896v1  

#### Abstract
We present \textbf{Flash-GMM}, a fused Triton kernel for efficient computation of Gaussian Mixture Models (GMMs) over large-scale data in a single GPU pass. By eliminating the need to materialize the full responsibility matrix in GPU memory, Flash-GMM achieves a \textbf{20$\times$} speedup over exis...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Flash-GMM: A Memory-Efficient Kernel for Scalable Soft Clustering

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Gaussian Mixture Model (GMM)** 在大规模数据上的训练受限于两个关键瓶颈：
- **内存占用过高**：标准实现需要在 GPU HBM 中显式存储 $N \times K$ 的 *responsibility matrix*（责任矩阵），当 $N$ 达到千万级时，该矩阵可轻易超过单卡 GPU 内存容量（如 >80GB）。
- **计算效率低下**：频繁访问 HBM 导致大量 IO 开销，成为性能瓶颈。

这使得 GMM 难以应用于大规模场景（如 ANN 搜索中的 IVF 粗量化器），而目前主流仍依赖硬聚类方法如 **k-means**。

---

### 提出了什么新方法或新思路
作者提出 **Flash-GMM** —— 一个基于 **Triton** 编写的融合内核（fused kernel），其核心思想借鉴自 **FlashAttention** 的 IO-aware tiling 策略，并适配到 GMM 的 EM 算法中。

#### 主要技术创新包括：
- **不显式构建责任矩阵**：通过分块（tiling）策略，在每个 tile 上在线计算 log-likelihood 和 responsibilities，并直接累积 sufficient statistics（$N_k$, $M_k$, $Q_k$），避免将完整的 $N \times K$ 矩阵写入 HBM。
- **仅需 $O(KD)$ 工作内存**：峰值内存与数据量 $N$ 无关，仅随组件数 $K$ 和维度 $D$ 增长，从而支持任意规模的数据训练。
- **最小化 HBM 访问次数**：整个 EM 迭代过程中只读取一次数据 $X$（$ND$ 元素）和两次参数（$2KD$），总访问为 $O(ND)$，相比传统实现减少约 $3\times$ 以上的内存流量。

此外，作者展示了 Flash-GMM 在 **IVF 粗量化器** 中的应用潜力，提出了基于 GMM 责任值的 **soft multi-assignment** 策略，进一步提升 ANN 搜索质量。

---

### 相比现有方法的优势
| 维度 | Flash-GMM | TorchGMM | SciPy/CPU |
|------|-----------|----------|-----------|
| **GPU 内存使用** | 极低（~MB 级） | 高（随 $N$ 线性增长，OOM at $N>10^6$） | 不适用 |
| **运行速度** | 快（比 TorchGMM 快 19–32×，比 SciPy 快 ~1700×） | 中等 | 极慢 |
| **可扩展性** | 支持 $N=10^8$ | 最多支持 $N\sim10^6$ | $N<10^5$ 即极慢 |
| **功能扩展性** | 支持 soft assignment、multi-assignment | 仅基础 GMM | 无 |

> ✅ Flash-GMM 实现了 **memory-efficient + scalable + high-performance** 的 GMM 训练，首次使 soft clustering 成为生产级 ANN 系统的可行选择。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在以下三个标准 ANN benchmark 数据集上进行：
- **SIFT1M**：$10^6$ 个 SIFT 特征向量，$D=128$
- **Deep10M**：前 $10^7$ 个 deep-image 描述符，$D=96$
- **GloVe-100**：约 1.18M 个词嵌入向量，$D=100$，使用 angular distance

所有数据均经过 L2 归一化处理（尤其对 angular metric 场景）。

---

### 实验设置和评估指标

#### 评估任务：IVF 粗量化器替换实验
目标是验证 Flash-GMM 是否可以作为 k-means 的“drop-in replacement”，并在搜索质量和成本之间取得更好权衡。

#### 对比方法：
1. **K-Means**（FAISS 实现）：工业标准 baseline，hard assignment
2. **GMM single**：Flash-GMM + 单分配（每个向量分配至最大 responsibility 的 cluster）
3. **GMM multi**：Flash-GMM + 多分配（若某 cluster 的 responsibility $r_{ik} > 1/K$，则加入对应 posting list，最多两个）

#### 评估指标：
- **Recall@10 (R@10)**：查询返回结果中包含真实最近邻的比例
- **Distance Computation Operations (DCO)**：每查询的距离计算次数，反映实际搜索开销
- **Index Build Time**：索引构建耗时（离线开销）
- **Average List Multiplicity ($m$)**：平均每个向量被插入多少个 posting list

#### 参数设置：
- $K = 1024$（cluster 数量）
- $nprobe \in \{1,4,8,16,32,48\}$
- 所有方法使用相同随机种子
- GMM 初始化采用 warm-start（先跑 10 轮 k-means 再接 90 轮 soft EM）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ⏱️ 运行时间与可扩展性（Table 1 & Table 4）
| $N$ | Flash-GMM (30 EM iter) | TorchGMM | SciPy |
|-----|------------------------|----------|-------|
| $10^6$ | 3.76s | 22× slower | ~1,700× slower |
| $10^8$ | 74.3s | **OOM** | **OOM** |

> Flash-GMM 可处理高达 **1亿样本** 的 GMM 训练，而 TorchGMM 在 $N>10^6$ 后即 OOM。

#### 💾 GPU 内存占用（Table 2）
| $N$ | Flash-GMM kernel mem | TorchGMM kernel mem |
|-----|------------------------|---------------------|
| $10^5$ | 0.9 MB | ~2.1 GB |
| $10^6$ | 4.5 MB | ~21 GB |

> Flash-GMM 内核自身内存仅为 **几 MB**，而 TorchGMM 超过 **20GB**，相差近 **5000×**

#### 🔍 搜索性能（Table 3 & Figure 1）
以 **GloVe-100**, $nprobe=16$ 为例：
| Method | R@10 | DCO ($\times10^3$) |
|--------|------|------------------|
| K-Means | 0.85 | 18.4 |
| GMM single | 0.86 | 18.4 |
| GMM multi | **0.92** | 32.8 |

> 尽管 DCO 上升，但 GMM multi 在更小的 $nprobe$ 下即可超越更高 $nprobe$ 的 K-Means。例如：
- GMM multi @ $nprobe=16$: R@10=0.92, DCO=32.8K
- K-Means @ $nprobe=32$: R@10=0.90, DCO=36.9K  
→ **同时实现更高 recall 与更低 DCO**

#### 📈 总体收益总结
- **Recall 提升**：在相同 DCO 下，GMM multi 可带来 **+2 至 +12 pp** 的 Recall@10 提升
- **效率增益**：达到相同 recall 目标时，GMM multi 最多可节省 **1.7× 的 DCO**
- **index inflation 控制良好**：平均 $m \in [1.49, 1.78]$，远低于盲目 multi-assignment 的 $m=2.0$

---

### 消融实验结果（Ablation Studies）

#### （1）Multi-assignment 阈值分析（Section 4.6）
测试不同阈值 $T = \alpha / K$：
- $T = 0.5/K$：posting list 更长 → DCO ↑，但 recall 未提升
- $T = 2/K$：过滤太严 → recall ↓ 1–4 pp
- **最优为 $T = 1/K$**：平衡 recall 与 DCO

#### （2）是否 multi-assignment 自身带来的收益？（Table 8）
对比 **K-Means hard top-2**（强制分配最近两个 centroid）：
- 虽然 $m=2.0$，但 recall 提升有限，且 DCO 比 single assignment **高出 1.8×**
- 表明：**multi-assignment 本身不足以带来收益，必须结合 principled responsibility 信号**

#### （3）初始化方式比较（Table 6）
- **warm-start**（k-means init） vs **kmeans++**
- 两者 recall 几乎一致
- warm-start 训练时间快 **3–4×**
→ 推荐使用 warm-start 作为默认初始化

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Flash-GMM 是首个真正 scalable 的 GMM 实现**：
   - 内存复杂度 $O(KD)$，支持 $N=10^8$ 规模训练
   - 比现有 GPU 实现快 20×，比 CPU 实现快上千倍

2. ✅ **GMM 可作为 k-means 的 drop-in 替代方案用于 IVF**：
   - 输出格式兼容（centroids 相同）
   - 支持无缝集成进 FAISS 等系统

3. ✅ **soft multi-assignment 显著改善 ANN 搜索质量**：
   - 利用 responsibility 分布将边界向量分配至多个 cluster
   - 实现 **pareto-optimal 改进**：在相同 DCO 下获得更高 recall，或在相同 recall 下降低 DCO

4. ✅ **性能增益来源于 GMM centroids + responsibility 的联合优化**：
   - 单纯 multi-assignment（如 hard top-2）效果差
   - GMM centroids 本身已优于 k-means，再叠加 multi-assignment 效果更强

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **训练速度较慢** | Flash-GMM 比 k-means 慢 2.5–3.3×，不适合频繁 re-indexing 场景 |
| **索引体积增大** | multi-assignment 导致 posting list 总量增加 $m$ 倍（实测 1.5–1.8×），对内存敏感部署构成压力 |
| **当前仅支持 isotropic covariance** | 未支持 full covariance，因在典型 IVF 规模下易出现 overfitting 和矩阵奇异问题（见 Appendix A） |
| **超大规模（$N \geq 10^9$）仍需外存支持** | 输入数据无法放入单卡 GPU，需引入 SSD streaming 或分布式训练 |
| **未探索更大 $K$ 设置的影响** | 当前实验集中在 $K \in \{256,1024,4096\}$，更大 $K$ 下 multi-assignment 效果未知 |

---

### 未来工作方向
1. **H100 架构优化**：利用 Tensor Memory Accelerator 和 async warp MMA 指令重构 Flash-GMM，有望实现类似 FlashAttention-3 的新一代加速。
2. **与 fine-quantizer 方法结合**：将 Flash-GMM 应用于 **IVF-PQ** 或 **IVF-PQfs** 流程，探索 coarse + residual 两级优化的叠加效应。
3. **扩展至其他模型**：Flash-GMM 的核心模式（log-sum-exp + weighted accumulation）也适用于 **Fisher Vector Encoding**、**Kernel Density Estimation** 等任务，具有通用加速潜力。
4. **支持 full covariance 或 diagonal covariance**：在适当正则化或降维前提下尝试更复杂的协方差结构。
5. **探索与 RAIRS 等几何启发式方法的融合**：研究是否可在 Flash-GMM centroids 上应用 AIR heuristic 实现进一步增益。

---

> 🔗 **开源地址**：https://github.com/IBM/Flash-GMM  
> 作者已将 Flash-GMM 发布为独立库，multi-assignment 仅需简单阈值操作即可启用，极大降低了 soft clustering 在科研与生产系统的应用门槛。

</details>

---

### 13. [Self-Distillation Policy Optimization via Visual Feedback: Bridging Code and Visual Artifacts](https://arxiv.org/abs/2606.10334)

**Authors**: Haoyu Dong  
**Category**: cs.AI  
**Published**: 2026-06-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.10334v1  

#### Abstract
Code-generating large language models (LLMs) increasingly produce visual artifacts such as charts, web pages, and slides by writing programs that are executed by non-differentiable renderers, committing to code before observing the render. As a result, otherwise executable code often yields artifact...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Self-Distillation Policy Optimization via Visual Feedback: Bridging Code and Visual Artifacts*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Code-generating Large Language Models (LLMs)** 在生成可视化产物（如图表、网页、幻灯片）时，存在一个“**代码-视觉鸿沟**”（code-vision gap）：
- 模型必须在**未看到渲染结果前就提交代码**；
- 即使语法正确，生成的代码也常导致视觉缺陷，例如：
  - 元素重叠（overlap）
  - 文本被裁剪（clipping）
  - 对齐错误（misalignment）
  - 颜色对比度低（low contrast）
  - 内容溢出（overflow）

现有方法存在以下局限：
- **Inference-time visual reflection**（如 render-critique-revise）需要多次调用 renderer 和 VLM，推理成本高；
- **Visual-reward RL**（如 GRPO/DPO）仅提供**稀疏的标量奖励**，缺乏对具体缺陷位置的定位能力。

---

### 🚀 提出的新方法：Visual-SDPO
作者提出 **Visual-SDPO**（Visual Self-Distillation Policy Optimization），一种结合**视觉反馈自蒸馏**的策略优化框架，核心思想是：
> 利用渲染后的视觉反馈作为“特权上下文”（privileged context），由共享权重的“教师模型”吸收该信息，并通过**知识蒸馏**指导“学生模型”改进代码生成。

#### 主要创新点：

1. **Visual-Feedback Self-Distillation**
   - 教师模型（teacher）接收原始输入 + 渲染图像/结构化诊断（visual diagnostic）；
   - 学生模型（student）只接收原始输入；
   - 教师不重新生成代码，而是对学生的输出进行**重新打分**（rescore），形成 token-level 的 KL 蒸馏目标。

2. **Visual-Grounded Code Credit Weighting**
   - 引入**空间信用分配机制**，将检测到的视觉缺陷反向映射回生成这些元素的代码语句；
   - 为每个 token 分配加权系数 $ w_t = 1 + (\alpha - 1) \cdot \text{resp}(s) $，其中 $\text{resp}(s)$ 是语句 $s$ 对缺陷的责任得分（基于 IoU 和严重性）；
   - 实现**梯度聚焦于真正导致缺陷的代码部分**，避免无关 token 干扰训练。

3. **联合优化目标：Token-level KD + Sequence-level GRPO**
   - 结合密集的 token-level 蒸馏信号与稀疏但全局的 GRPO 奖励；
   - 二者互补：KD 提供局部可解释监督，GRPO 锚定整体执行成功与质量。

---

### ⚖️ 相比现有方法的优势

| 方法 | 缺陷 |
|------|------|
| Inference-time reflection | 推理开销大，需多轮 renderer/VLM 调用 |
| Scalar visual reward (GRPO/DPO) | 反馈稀疏，无法定位具体问题代码 |
| Standard SFT/OPSD | 依赖参考代码，泛化性差 |

✅ **Visual-SDPO 的优势**：
- **无需额外推理成本**：训练时利用 renderer 和 defect detector，推理阶段完全移除；
- **反馈更精细**：不仅能判断“好不好”，还能指出“哪里不好”、“谁造成的”；
- **样本效率更高**：相比纯 GRPO，达到相同性能所需 rollout 数量减少约 **71%**（平均仅需 29% 的 rollout 预算）；
- **统一架构支持多任务**：在 chart、web/UI、slide 上均有效。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 领域 | 训练数据集 | 评估基准 |
|------|-----------|----------|
| **Chart** | Chart2Code-160K（过滤可执行子集） | ChartMimic (Direct Mimic) |
| **Web/UI** | WebCode2M + WebSight | Design2Code |
| **Slide** | AeSlides-7k（instruction-only） | AeSlides verifiable metrics |

> 所有领域独立训练，不跨域混合数据。

---

### 🔬 实验设置

- **主干模型**：统一使用 `Qwen3-VL-8B-Instruct`；
- **视觉反馈通道**：
  - **Image Channel**：教师接收渲染图像截图；
  - **Rubric Channel**：轻量级预训练模块提取结构化缺陷表（JSON 形式），如 CLIP 相似度、文本匹配、颜色等；
- **缺陷检测与映射**：
  - **Region-to-code mapping** 采用两种方式：
    1. **Runtime introspection**：运行时插桩记录每条语句生成的区域（用于 matplotlib、python-pptx）；
    2. **VLM-based mapping**：用 Qwen3-VL 自动识别责任语句（通用性强，适用于未插桩工具链）；
- **训练目标**：
  $$
  \mathcal{L}(\theta) = \mathcal{L}_{\text{VSDPO}} + \beta \cdot \mathcal{L}_{\text{GRPO}}
  $$
  - $\mathcal{L}_{\text{VSDPO}}$: 加权 token-level KL 损失；
  - $\mathcal{L}_{\text{GRPO}}$: 序列级奖励 $ r(y) = r_{\text{exec}}(y) \cdot r_{\text{vis}}(y) $

---

### 📊 评估指标

| 领域 | 主要指标 |
|------|--------|
| **ChartMimic** | Overall = (Low + High)/2<br>- Low: 四项低级指标 F1 均值（Text, Layout, Chart-Type, Color）<br>- High: GPT-4o judge 得分 |
| **Design2Code** | Overall = 五项低级指标均值（CLIP, Block-Match, Text, Position, Color） |
| **AeSlides** | Avg = 四项可验证规则均值（Aspect Ratio, Whitespace, Collision, Imbalance） |

---

### 🆚 基线方法对比

所有方法共享同一 backbone 和训练数据集：

| 基线 | 描述 |
|------|------|
| **Zero-shot** | 原始 Qwen3-VL-8B-Instruct，无微调 |
| **SFT** | 监督微调（Supervised Fine-Tuning） |
| **OPSD** | 使用参考代码作为特权信息的自蒸馏（Reference-code privileged teacher） |
| **GRPO (visual reward)** | 使用视觉相似性奖励进行 GRPO 优化 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### Table 1: ChartMimic 结果
| Method | Exec (%) | Low | High | **Overall** | △Over Base |
|--------|---------|-----|------|------------|-------------|
| Zero-shot | 81.7 | 62.9 | 72.9 | **67.9** | — |
| +SFT | 90.3 | 69.9 | 77.2 | 73.6 | +5.7 |
| +OPSD | 92.3 | 73.9 | 80.1 | 77.0 | +9.1 |
| +GRPO | 92.9 | 72.6 | 79.7 | 76.2 | +8.3 |
| **+Visual-SDPO (Ours)** | **92.7** | **74.8** | **82.3** | **78.6** | **+10.7** |

> ✅ 在图表生成中，**Visual-SDPO 显著优于所有单信号方法**，尤其提升 High-level judge score。

---

#### Table 2: Design2Code 结果
| Method | CLIP | Block | Text | Position | Color | **Overall** | △Over Base |
|--------|------|-------|------|----------|-------|-------------|-------------|
| Zero-shot | 85.4 | 50.2 | 78.1 | 71.9 | 74.7 | **72.1** | — |
| +SFT | 87.1 | 59.6 | 82.3 | 77.2 | 78.8 | 77.0 | +4.9 |
| +OPSD | 87.4 | 62.3 | 82.9 | 78.4 | 81.8 | 78.6 | +6.5 |
| +GRPO | 88.3 | 64.5 | 83.3 | 78.7 | 85.1 | 80.0 | +7.9 |
| **+Visual-SDPO (Ours)** | **89.2** | **69.7** | **84.9** | **82.4** | **86.8** | **82.6** | **+10.5** |

> ✅ 最大增益来自 **Position (+3.7)** 和 **Block-Match (+5.2)**，说明 region-to-code mapping 对布局类缺陷特别有效。

---

#### Table 3: AeSlides 结果
| Method | **Avg** | △Over Base |
|--------|--------|-------------|
| Zero-shot | 49.5 | — |
| +SFT | 52.8 | +3.3 |
| +GRPO | 58.2 | +8.7 |
| **+Visual-SDPO (Ours)** | **60.7** | **+11.2** |

> ✅ 即使 GRPO 已直接优化评估指标，**Visual-SDPO 仍带来 +2.5 的额外增益**，表明 token-level credit assignment 仍有价值。

---

### 🔍 消融实验分析（Ablation Analysis）

- **Chart Domain**：
  - SFT → OPSD 提升最大（+3.4），说明参考代码对语法结构建模强；
  - GRPO 表现略低于 OPSD，说明**标量视觉奖励不如 token-level 结构监督**；
  - **Visual-SDPO 同时融合两者优势，实现全面超越**。

- **Web/UI Domain**：
  - GRPO 显著提升 Color 和 Block-Match；
  - **Visual-SDPO 进一步大幅提升 Position 和 Block-Match**，体现空间映射的有效性。

- **Slide Domain**：
  - GRPO 收益最大（+5.4 over SFT），因其奖励函数与评测指标一致；
  - **Visual-SDPO 仍能进一步提升 +2.5**，说明即使奖励对齐，细粒度 credit assignment 依然重要。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **视觉反馈可用于自蒸馏，显著提升代码生成质量**
   - 将渲染图像或结构化诊断作为“特权信息”，可有效引导学生模型学习视觉一致性。

2. **均匀蒸馏会引入噪声，应聚焦于“责任语句”**
   - 大多数代码 token 不直接影响视觉输出（如变量声明、控制流）；
   - **Visual-Grounded Code Credit Weighting 成功将梯度集中在关键语句上**，提高训练效率与效果。

3. **Token-level 与 Sequence-level 信号互补**
   - 密集的 token-level KD 提供局部修正信号；
   - GRPO 提供全局执行与质量锚点；
   - 二者结合优于任一单一信号。

4. **更高的样本效率**
   - Visual-SDPO 仅需 GRPO **约 29% 的 rollout 数量**即可达到相近甚至更优性能，大幅降低训练成本。

---

### ⚠️ 局限性

1. **依赖 renderer 和 defect detector 的可用性**
   - 当前方法需在训练期间访问非可微分 renderer，限制其在某些封闭环境中的应用；
   - 缺陷检测器目前为启发式规则或简单分类器，可能漏检复杂美学问题。

2. **mapping 准确性影响 credit assignment**
   - 若 region-to-code mapping 不准（尤其是 VLM-based 方式），可能导致错误归责；
   - 插桩方式虽精确，但需修改执行环境，扩展性受限。

3. **静态产物假设**
   - 当前方法针对静态渲染产物（如单张图表、页面）；
   - 对动态交互式 UI 流程的支持有限。

---

### 🔮 未来工作方向

1. **构建大规模 chart-RL 数据集**
   - 推动视觉反馈训练进入“大规模 RL 范式”。

2. **探索可微分渲染器作为第三通道**
   - 引入近似梯度传播路径，实现端到端优化。

3. **扩展 region-to-code mapping 至动态流程**
   - 支持多步交互式 UI 生成，追踪事件驱动下的视觉变化来源。

4. **更智能的缺陷诊断系统**
   - 使用更强的 VLM 或专用模型进行细粒度美学评估，替代手工定义 rubric。

---

> 💡 **总结一句话**：  
> **Visual-SDPO 开创性地将视觉反馈融入自蒸馏框架，通过“责任感知”的 token 加权机制，实现了高效、精准、无推理开销的代码生成优化，在 chart、UI、slide 多任务上全面超越零样本与主流 RL 方法。**

</details>

---

### 14. [ReasonAlloc: Hierarchical Decoding-Time KV Cache Budget Allocation for Reasoning Models](https://arxiv.org/abs/2606.11164)

**Authors**: Wenhao Liu, Hao Shi, Yunhe Li, Weizhi Fei, Xiangyuan Wang, Mengzhe Ruan, Hanxu Hou, Peisong Wang, Linqi Song, Shuang Qiu  
**Category**: cs.AI  
**Published**: 2026-06-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11164v1  

#### Abstract
Long chain-of-thought (CoT) trajectories in large language model (LLM) reasoning cause severe inference bottlenecks due to rapid key-value (KV) cache growth. Current decoding-time compression methods mitigate this issue via token eviction, but typically assume a uniform budget distribution across al...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**ReasonAlloc: Hierarchical Decoding-Time KV Cache Budget Allocation for Reasoning Models**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLM）在进行复杂推理任务时，通常生成长链的 **Chain-of-Thought (CoT)** 轨迹，导致解码阶段的 **Key-Value (KV) cache** 快速增长，引发严重的内存瓶颈和推理吞吐下降。

现有 **decoding-time KV cache 压缩方法**（如 SnapKV、R-KV）普遍采用**均匀预算分配策略**（uniform budget allocation），即在所有网络层（layers）和注意力头（heads）之间平均分配缓存容量。然而，这种假设忽略了不同层和头在推理过程中对 KV cache 的实际需求差异。

此外，虽然已有非均匀分配方法（如 PyramidKV），但它们主要针对 **prompt prefill 阶段** 设计，依赖静态规则（如单调递减），无法适应自回归生成中动态变化的上下文重要性。

---

### 🚀 提出的新方法与创新思路
作者提出 **ReasonAlloc** —— 一种**无需训练、即插即用**的分层 KV cache 预算分配框架，将 decoding-time KV 压缩重新定义为一个**层次化资源分配问题**。

#### 核心思想：两层级联动态分配
1. **离线层间预分配（Layer-wise Preallocation）**
   - 在推理前通过轻量级探针分析模型各层的 KV 需求。
   - 发现并建模了一种架构驱动的非线性 KV 需求模式，称为 **“Reasoning Wave”**：
     - **浅层**：高需求 → 全局语义感知
     - **中间层**：低且振荡 → 局部逻辑推导
     - **深层**：需求激增 → 输出前的整体验证机制
   - 该分布具有**跨任务稳定性**，主要由模型架构决定，而非输入任务。

2. **在线头内重分配（Head-wise Dynamic Routing）**
   - 在解码过程中每 Δ 步动态调整每层内部各 attention head 的预算。
   - 基于实时计算的 token utility（重要性 + 冗余度）得分，识别当前“信息丰富”的 head。
   - 引入 **robustification operator** 防止某些 head 因短期评分下降而被永久“饿死”。

> 🔧 **优势**：ReasonAlloc 是通用模块，可无缝集成到任何基于 token eviction 的压缩策略之上（如 R-KV、SnapKV），仅负责预算分配，不改变底层评分机制。

---

### ⚖️ 相比现有方法的优势
| 方面 | 传统方法（如 R-KV） | ReasonAlloc |
|------|--------------------|-----------|
| 预算分配方式 | 统一均匀分配 | 分层异构分配（layer + head） |
| 是否动态 | 否（静态或固定规则） | 是（head-level 动态刷新） |
| 是否任务自适应 | 否 | 是（head 路由随生成过程变化） |
| 是否架构适配 | 否 | 是（“Reasoning Wave”因架构而异） |
| 推理开销 | 极低 | 几乎无额外开销（仅每128步一次向量化操作） |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **MATH-500**：从 MATH 数据集中选取的 500 道数学题，涵盖多种难度。
- **AIME 2024**：美国数学邀请赛级别的竞赛题目，挑战性强，需长程推理。
- 辅助分析还使用了 **LCB**（代码合成）、**LongBench**（长文本理解）用于验证跨任务一致性。

### 🧪 模型
在以下三种主流推理模型上进行测试：
- **DeepSeek-R1-Distill-Llama-8B**（简称 R1-Llama-8B）
- **DeepSeek-R1-Distill-Qwen-14B**（R1-Qwen-14B）
- **AceReason-14B**

均为经过强化学习蒸馏的 CoT 推理专用模型。

### 📊 评估指标
- **Pass@1**：在温度 0.6、top-p 0.95 下采样 8 次，取平均 pass@1 得分。
- **系统效率**：
  - 吞吐量（tokens/s）
  - 最大可持续 batch size
  - 解码时间（seconds）

### 🔁 实验设置
- **全局 KV cache 预算 B**：控制保留 token 数量（如 128–3072）
- **刷新间隔 Δ = 128**：head-wise 分配每 128 步更新一次
- **注意力质量阈值 p = 0.93**：用于确定每层最小需保留 token 数（见附录 E 敏感性分析）
- 所有方法共享相同的 token scoring 策略（R-KV 的 `αI + (1−α)R`）

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **FullKV** | 无压缩 | 不删除任何 token，作为上限参考 |
| **SnapKV** | 静态压缩 | 基于 attention 重要性选择 token |
| **R-KV** | 解码时压缩 + uniform alloc. | 当前 SOTA，使用重要性-冗余联合打分，但预算均匀分配 |
| **Pyramid-RKV** | 静态非均匀分配 | 将 PyramidKV 的单调递减预算应用于 R-KV，模拟 prefill-centric 方法迁移至 decoding 的效果 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2 和 Figure 3）

#### 在 **MATH-500** 上的表现（R1-Llama-8B）
| 方法 \ 预算 | 512 | 1024 | FullKV |
|------------|-----|------|--------|
| SnapKV     | 63.62% | 75.68% | 89.20% |
| R-KV       | 76.48% | 81.64% | — |
| Pyramid-RKV| 79.40% | 83.64% | — |
| **ReasonAlloc (Ours)** | **82.50%** | **86.20%** | — |

> ✅ 在 **512 token 预算下提升近 6 个百分点**（vs R-KV）

#### 在 **AIME 2024** 上的表现（R1-Llama-8B）
| 方法 \ 预算 | 256 | 1024 | FullKV |
|------------|-----|-------|--------|
| SnapKV     | 1.25% | 26.25% | 50.42% |
| R-KV       | 10.42% | 45.42% | — |
| Pyramid-RKV| 9.17% | 39.17% | — |
| **ReasonAlloc (Ours)** | **20.00%** | **49.17%** | — |

> ✅ 在极小预算（256）下实现 **翻倍精度提升**；接近 FullKV 表现（49.17% vs 50.42%）

> 💡 **趋势总结**：预算越紧张，ReasonAlloc 的优势越明显！

---

### 🔍 消融实验结果（Ablation Study on AIME 2024）

| 配置 | 256 | 512 | 1024 | 2560 |
|------|-----|-----|------|-------|
| R-KV（Baseline） | 10.42% | 28.33% | 45.42% | 51.67% |
| + Layer-wise Only | 13.32% | 30.83% | 42.50% | **55.00%** |
| + Head-wise Only | 17.50% | 31.66% | 45.82% | 52.50% |
| **ReasonAlloc (Full)** | **20.00%** | **32.08%** | **49.17%** | 53.34% |

#### 结论：
- **Head-wise 动态路由** 对小预算最关键（↑7.5% @256）
- **Layer-wise 预分配** 更利于中等以上预算（↑显著 @2560）
- **两者结合** 取得最稳定全面的收益，体现协同效应。

---

## 4. 关键结论和发现

### 🎯 主要发现
1. **KV cache 需求存在显著异质性**：
   - 层间需求呈非线性 “**Reasoning Wave**” 模式，非单调衰减。
   - 头内需求差异巨大：某些 head 需 >500 tokens，相邻 head <50 tokens（见 Figure 6）。
   
2. **统一或静态分配是次优的**：
   - 均匀分配浪费资源于冗余 head，饿死关键路径。
   - Pyramid-RKV 显示：直接迁移 prefill 规则到 decoding 会损害性能。

3. **ReasonAlloc 实现高效精准资源调度**：
   - 离线捕捉架构共性（Reasoning Wave）
   - 在线响应生成动态（head utility 波动）
   - 性能显著优于 uniform 与 static non-uniform 方法

4. **几乎零推理开销**：
   - 在 16K 序列长度下，ReasonAlloc 吞吐达 **218.82 tokens/s**，相比 FullKV 提升 **5.52×**
   - 与 R-KV 相比无速度损失（≈217–218 tok/s），说明其高效性。

---

### ⚠️ 方法的局限性
- **依赖高质量 token scoring 机制**：本身不提供 scoring，需配合 R-KV 等先进策略才能发挥最大效用。
- **离线校准假设任务不变性**：虽在 CoT 模型中成立，但在通用 LLM 或高度任务敏感场景可能失效（为此提供了 fully dynamic fallback，见 Appendix C）。
- **参数敏感性未充分探索**：如 p=0.93 是手动选定，虽有合理性，但缺乏端到端调优。

---

### 🔮 未来工作方向
1. **扩展至多模态推理模型**：探索视觉-语言模型中的跨模态 KV 分配策略。
2. **引入轻量学习机制**：设计可微分预算控制器，实现端到端优化。
3. **支持更细粒度分配**：如 token-level 或 position-level 动态预算调整。
4. **构建通用 KV 编排器**：将 ReasonAlloc 推广为推理引擎的标准组件，适配更多压缩算法。

---

> ✅ **一句话总结**：  
> **ReasonAlloc 通过“离线层波 + 在线头流”的双层动态预算分配，在几乎不增加开销的前提下，显著提升了长链推理模型在有限 KV cache 下的准确率，尤其在极端压缩条件下表现卓越，为高效服务大型推理模型提供了实用解决方案。**

</details>

---

### 15. [WebChallenger: A Reliable and Efficient Generalist Web Agent](https://arxiv.org/abs/2606.10423)

**Authors**: Jayoo Hwang, Xiaowen Zhang, Vedant Padwal  
**Category**: cs.CL  
**Published**: 2026-06-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.10423v1  

#### Abstract
Autonomous web navigation remains challenging for LLM agents, and the strongest generalist systems rely on proprietary reasoning models whose inference cost is prohibitive for the repetitive tasks where such agents would be most useful. We argue this gap stems not from insufficient model capability ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# WebChallenger: A Reliable and Efficient Generalist Web Agent 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **LLM** 的 **web agent** 在自主网页导航任务中表现不佳，且最先进的通用系统依赖于推理成本高昂的专有模型（如 GPT-4o、GPT-5），难以在需要重复执行的任务中大规模应用。作者认为，这一差距并非源于模型能力不足，而是现有 **agent 架构** 未能复现人类在网页浏览中的三大认知优势。

### 提出的新方法与创新思路
论文提出了 **WebChallenger**，一个高效可靠的通用 **web agent** 框架，其核心是通过架构设计而非模型规模来弥补上述差距。该框架围绕一种名为 **PageMem** 的新型结构化页面表示构建。

#### 核心创新组件：
- **PageMem**：一种从 **DOM** 确定性构造的结构化页面表示。它将每个网页分解为语义区域的层次结构（如导航栏、产品列表、评论表单等），并为每个区域生成简短摘要。这使得 **agent** 可以像浏览目录一样快速“浏览”页面，并仅对相关区域进行详细处理。
- **分而治之的观察管道 (Divide-and-Conquer Observation Pipeline)**：模仿人类的**选择性注意 (selective attention)**。**agent** 首先浏览 **PageMem** 中各部分的摘要，筛选出与任务相关的区域，然后只从这些选定区域提取细节，从而生成信息密集的观察结果，避免处理整个冗长的页面。
- **轻量级探索与记忆系统 (Lightweight Exploration and Memory System)**：模仿人类的**持久记忆 (persistent memory)**。在任务开始前，系统会进行一次离线探索，遍历目标网站，构建一个可重用的 **WebsiteMem**。这个内存记录了网站的页面、导航路径以及交互元素的行为（例如点击后展开的下拉菜单），使得 **agent** 在后续任务中无需重新学习网站结构。
- **复合动作工作流 (Compound Action Workflows)**：模仿人类的**程序性熟练度 (procedural fluency)**。将常见的多步交互模式（如搜索、下拉菜单选择、表单提交）封装成单一的 **agent** 动作。这些工作流内部自动处理中间状态变化（如下拉菜单展开），无需 **agent** 对每一步都进行重新观察和推理。

### 相比现有方法的优势
- **高性能与低成本**：使用开源、未经微调的模型（如 **GLM-4-32B** 和 **Qwen2.5-VL-7B**）即可达到接近前沿专有系统的性能，但推理成本仅为后者的几分之一。
- **通用性强**：所有机制均建立在 **PageMem** 这一共享抽象之上，因此框架可以泛化到不同网站，无需为特定站点编写适配器。
- **效率高**：通过减少不必要的上下文处理和合并多步操作，显著降低了 **token** 消耗和决策步骤数。

## 2. 核心实验方法和设置

### 使用的数据集
论文在四个开放式的 **web navigation** 基准测试集上进行了评估：
- **WebArena**：包含 812 个模拟环境中的任务，旨在模仿常见网站类型（如论坛、维基）。
- **VisualWebArena**：基于 **WebArena** 的基础设施，但包含 910 个需要视觉推理的任务。
- **Online-Mind2Web**：包含来自 136 个真实世界网站的 300 个任务。
- **WorkArena**：包含 330 个企业相关的复杂 **UI** 导航任务。

### 实验设置和评估指标
- **模型**：使用 **GLM-4-32B-0414** 作为 **LLM** 控制器，**Qwen2.5-VL-7B-Instruct** 作为辅助的视觉模型（用于图像描述）。在 **VisualWebArena** 上则使用 **Qwen3-VL-4B-Instruct**。
- **训练**：所有实验均采用 **zero-shot** 方式，未对模型进行任何微调。
- **评估指标**：主要指标为任务成功率（success rate %）。
- **流程**：首先对基准测试的所有网站进行离线探索以构建 **WebsiteMem**，然后在独立的任务上运行推理。

### 基线方法对比
- **专有模型基线 (Proprietary Models)**：包括 **GenericAgent** (GPT-4o, Claude 3.5 Sonnet, GPT-5)、**WALT**、**IBM CUGA**、**OpenAI CUA**、**ScribeAgent** 等。
- **开源模型基线 (Open-Source Models)**：包括经过微调的 **Agent-as-Annotators**、**Mobile-Agent-v3.5**、**WebDreamer** 等，以及其他 **zero-shot** 方法如 **Tree Search**。

## 3. 主要实验结果和性能指标

### 关键性能数据
WebChallenger 在四个基准测试上均取得了当前使用开源模型的最佳性能（SOTA）：
- **WebArena**: **56.3%**
- **VisualWebArena**: **48.7%**
- **Online-Mind2Web**: **51.0%**
- **WorkArena**: **70.9%**

### 与基线方法的对比结果
- 在 **WebArena** 上，**56.3%** 的成绩超过了最强的微调开源模型基线 **Mobile-Agent-v3.5 (48.4%)** 近 8 个百分点，并优于 **ScribeAgent (53.0%)**。
- 在 **WorkArena** 上，**70.9%** 的成绩远超下一个最佳的 **zero-shot** 开源模型，并超过了 **Claude 3.5 Sonnet (56.4%)** 和 **GPT-4o (45.5%)** 等专有模型作为骨干的结果。
- 整体来看，WebChallenger 的性能已非常接近甚至在某些方面超越了部分专有系统，证明了优秀的架构设计可以极大地弥补小模型与大模型之间的差距。

### 消融实验结果
为了验证各组件的贡献，在 **WebArena-lite** 子集上进行了消融实验：

| 方法 | 平均成功率 (Avg) | 相对于完整系统的下降 (△) |
| :--- | :--- | :--- |
| **WebChallenger (完整系统)** | **58.8%** | — |
| - 记忆系统 (Memory) | 51.2% | -7.6 |
| - 复合动作 (Compound Actions) | 49.1% | -9.7 |
| - 观察管道 (Observation Pipeline) | 41.2% | **-17.6** |

- **结论**：移除**观察管道**导致的性能下降最大（-17.6%），其次是**复合动作**（-9.7%），最后是**记忆系统**（-7.6%）。这表明，能够聚焦于相关信息的分层观察机制是该框架最核心的创新。

## 4. 关键结论和发现

### 主要发现
1.  **架构即能力**：当前的 **LLM** 已经具备完成许多常见网页任务所需的智能，但标准框架未能通过适当的“脚手架”（即观察、记忆和行动的设计）来有效利用这种智能。
2.  **人类认知的启示**：通过在架构层面实现**选择性注意**、**持久记忆**和**程序性熟练度**这三种人类认知优势，可以显著提升 **web agent** 的性能和效率。
3.  **低成本高性能**：使用小型、通用的开源模型，通过精心设计的架构，可以在多个基准测试上达到接近专有系统的性能，为 **web agent** 的实际应用铺平了道路。

### 方法的局限性
1.  **依赖手工设计**：框架依赖于手工设计的组件（如 **DOM** 分区规则、可点击元素识别启发式方法、固定的复合动作工作流），在面对严重偏离常规模式的网站时，性能可能会下降。
2.  **顺序调用开销**：虽然总 **token** 数可能更少，但该框架使用了更多的顺序 **LLM** 调用，增加了任务的墙钟时间（wall-clock time），如果使用昂贵的专有模型API，运行成本会很高。
3.  **内存实例化简单**：本文采用了最小化的内存实例化方式，更高级的记忆机制（如在线工作流学习）有待未来探索。
4.  **安全性未知**：所有评估均在良性任务上进行，系统对对抗性网页内容的鲁棒性尚未得到表征。

### 未来工作方向
- 探索更高级的内存机制，如 **online workflow learning** 或 **synthetic-data generation**。
- 研究如何使框架对非标准或对抗性网页更具鲁棒性。
- 优化框架以减少顺序 **LLM** 调用次数，提高实时性能。

</details>

---

### 16. [Continual LLM Upcycling: A Predictor-Gated Bank-Wise Sparsity Training Recipe for Dense-to-Sparse LLMs](https://arxiv.org/abs/2606.10722)

**Authors**: Ruixuan Huang, Jinyuan Shi, Hantao Huang, Yifan Huang, Ziyi Guan, Hao Zeng, Ian En-Hsu Yen, Minghui Yu  
**Category**: cs.CL  
**Published**: 2026-06-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.10722v1  

#### Abstract
We study dense-to-sparse continual training as a way to construct channel-sparse large language models from dense checkpoints. Starting from a Qwen2.5-8B dense backbone, we continue training at 32K context and introduce a predictor-gated sparse SwiGLU FFN in the 32K stage. For each token and layer, ...

---

### 17. [Pushing the Limits of LLM Tool Calling via Experiential Knowledge Integration and Activation](https://arxiv.org/abs/2606.10875)

**Authors**: Yupu Hao, Zhuoran Jin, Huanxuan Liao, Kang Liu, Jun Zhao  
**Category**: cs.CL  
**Published**: 2026-06-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.10875v1  

#### Abstract
Large language models (LLMs) rely on tool use to act as autonomous agents, yet often fail in multi-step execution due to insufficient tool-related knowledge and ineffective knowledge activation. Therefore, we present a systematic study on how knowledge influences tool-use performance, covering the s...

---

### 18. [ActiveMem: Distributed Active Memory for Long-Horizon LLM Reasoning](https://arxiv.org/abs/2606.10532)

**Authors**: Yunhan Jiang, Wenbin Duan, Shasha Guo, Liang Pang, Xiaoqian Sun, Huawei Shen  
**Category**: cs.AI  
**Published**: 2026-06-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.10532v1  

#### Abstract
Memory is essential for enabling large language model (LLM) agents to handle long-horizon reasoning tasks. Existing memory mechanisms are largely centralized, typically organizing retrieved information and interaction history within a single model context. This design imposes a fundamental trade-off...

---

### 19. [Hasse Diagrams for Attention: A Partial Order Framework for Designing Transformer Masks](https://arxiv.org/abs/2606.09951)

**Authors**: Chentao Li, Han Guo  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.09951v1  

#### Abstract
During the training of large Transformer models, attention masks regulate the scope and direction of information flow across a sequence. Numerous mask variants exist, and operators such as FlexAttention already support arbitrary attention masks. Nevertheless, a systematic formal analysis of the info...

---

### 20. [Role-Agent: Bootstrapping LLM Agents via Dual-Role Evolution](https://arxiv.org/abs/2606.10917)

**Authors**: Xucong Wang, Ziyu Ma, Shidong Yang, Tongwen Huang, Pengkun Wang, Yong Wang, Xiangxiang Chu  
**Category**: cs.AI  
**Published**: 2026-06-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.10917v1  

#### Abstract
Although Large Language Model (LLM) agents have demonstrated strong performance on complex tasks, their learning is often limited by inefficient interaction feedback and static training environments, which hinder broader generalization. To address these limitations, this paper introduces Role-Agent,...

---

### 21. [Hyperparameter Learning for Latent Factorization of Tensors for Representation Learning to Large-scale Dynamic Weighted Directed Network](https://arxiv.org/abs/2606.09880)

**Authors**: Yaqian Zhan, Jialan He, Tianzhu Chen  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.09880v1  

#### Abstract
Large-scale dynamic weighted directed networks (DWDNs) are widely used to model time-varying interactions among nodes. Latent factorization of tensors (LFT) extracts target knowledge from DWDNs via low-rank embedding. However, similar to many machine learning models, the performance of LFT heavily d...

---

### 22. [SinkRec: Mitigating Semantic State Sink in Long Sequence Recommendation with Memory-Conditioned Gated Delta Networks](https://arxiv.org/abs/2606.09888)

**Authors**: Zhuang Zhuang, Zhipeng Wei, Ji Dai, Jie Chen, Fei Pan, Peng Jiang, Kun Gai  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.09888v1  

#### Abstract
Linear attention provides an efficient backbone for long-sequence recommendation by avoiding the quadratic cost of standard Transformers, but its compressed recurrent state can be dominated by repetitive behavior patterns. We identify this phenomenon as semantic state sink, where recurring semantics...

---

### 23. [Integrating Out, Twice:The Open-System Case That Neural-Network Ensemble Theory Is Missing](https://arxiv.org/abs/2606.09950)

**Authors**: Jin Lei  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.09950v1  

#### Abstract
Averaging a neural network over its random parameters and marginalizing a Gaussian sector are the same operation, the Schur complement of the eliminated block, and when that block is closed it returns a covariance and its inverse. That is all a network ensemble produces, the closed case. The open ca...

---

### 24. [PL-KKT-hPINN: Enforcing Nonlinear Equality Constraints on Neural Networks via Piecewise-Linear Projection](https://arxiv.org/abs/2606.10682)

**Authors**: Fateme Mohammad Mohammadi, Hector Budman, Joshua L. Pulsipher  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.10682v1  

#### Abstract
While physics-informed neural networks (PINNs) have shown strong potential for process modeling, physical equations are only enforced as soft constraints during training, and thus, they do not guarantee constraint satisfaction at inference. We propose a framework, called piecewise-linear Karush--Kuh...

---

### 25. [Less Context, Better Agents: Efficient Context Engineering for Long-Horizon Tool-Using LLM Agents](https://arxiv.org/abs/2606.10209)

**Authors**: Abhilasha Lodha, Mahsa Pahlavikhah Varnosfaderani, Abir Chakraborty, Abhinav Mithal  
**Category**: cs.AI  
**Published**: 2026-06-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.10209v1  

#### Abstract
Large language models deployed as autonomous agents for enterprise workflows face a key challenge: verbose tool responses from enterprise systems can cause context overflow, stale-state errors, and high inference cost. We study this problem in automated expense itemization in Microsoft Dynamics 365 ...

---

### 26. [Early-Token Confidence Predicts Reasoning Quality in Multi-Agent LLM Debate](https://arxiv.org/abs/2606.10307)

**Authors**: Ali Keramati, Justin Cheok, Jacob Horne, Mark Warschauer  
**Category**: cs.CL  
**Published**: 2026-06-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.10307v1  

#### Abstract
Evaluating reasoning quality in multi-agent LLM systems is challenging, especially for open-ended tasks without reference answers. We investigate whether intrinsic confidence signals, token-level log-probabilities from decoding, can predict reasoning quality as assessed by LLM-as-judge evaluation. U...

---

### 27. [The Order Matters: Sequential Fine-Tuning of LLaMA for Coherent Automated Essay Scoring](https://arxiv.org/abs/2606.10327)

**Authors**: Ali Keramati, Mark Warschauer  
**Category**: cs.CL  
**Published**: 2026-06-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.10327v1  

#### Abstract
Automated Essay Scoring (AES) systems must judge interdependent discourse elements (e.g., lead, claim, evidence, conclusion), yet most approaches treat these in isolation, harming coherence and generalization. We investigate task-aware fine-tuning of LLaMA-3.1-8B for AES using parameter-efficient Lo...

---

### 28. [REAL: A Reasoning-Enhanced Graph Framework for Long-Term Memory Management of LLMs](https://arxiv.org/abs/2606.10694)

**Authors**: Keer Lu, Liwei Chen, Guoqing Jiang, Zhiheng Qin, Yunhuai Liu, Wentao Zhang  
**Category**: cs.CL  
**Published**: 2026-06-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.10694v1  

#### Abstract
Large Language Models (LLMs) are increasingly expected to interact with users over long time horizons. However, due to their finite context window, LLMs cannot retain all past interactions, making long-term memory management essential for storing, updating, and retrieving historical information beyo...

---

### 29. [TENP: Trapezoidal Expert Neuron Pruning For Mixture-of-Experts](https://arxiv.org/abs/2606.09885)

**Authors**: Jiangyang He, Shaolin Zhu, Deyi Xiong  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.09885v1  

#### Abstract
Mixture-of-Experts large language models (LLMs) scale efficiently through sparse activation, yet their deployment is fundamentally constrained by the large static parameter footprint of experts. Existing compression approaches either remove entire experts, disrupting routing topology and harming per...

---

### 30. [DUET -- Dual User Embedding Transformers for Offsite Conversion Prediction](https://arxiv.org/abs/2606.10243)

**Authors**: Reazul Hasan Russel, Mingwei Tang, Rostam Shirani, Xinlong Liu, Navid Madani, Leo Ding, Yawen He, Xiangyu Wang, Mustafa Acar, Ashish Katiyar, Yuhai Li, Alan Yang, Metarya Ruparel, Derek Qiang Xu, Rupert Wu, Rui Yang, Liang Tao, Xinyi Zhao, Larry Zhang, Sri Reddy, Rob Malkin  
**Category**: cs.LG  
**Published**: 2026-06-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.10243v1  

#### Abstract
Offsite conversion rate (OCVR) prediction is an important ranking problem in computational recommendation systems. This task presents a modeling challenge: click signals are abundant and exhibit short temporal horizons, whereas conversion signals are inherently sparse, long-delayed, and frequently u...

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

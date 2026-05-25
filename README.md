# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-25 09:45:00 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs](https://arxiv.org/abs/2605.23764)

**Authors**: Zewen Jin, Congkun Ai, Guangpeng Zhang, Hanbo Zhang, Haoran Wang, Shihan Xiao, Da Lei, Xuefeng Jin, Teng Su, Cheng Li  
**Category**: cs.DC  
**Published**: 2026-05-25  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.23764v1  

#### Abstract
Modern Mixture-of-Experts (MoE) models increasingly rely on large-scale AI accelerator clusters for efficient training. Ascend NPUs expose heterogeneous on-chip compute resources, including matrix-oriented AIC units and vector-oriented AIV units with explicit cross-queue synchronization support. How...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 **Mixture-of-Experts (MoE)** 模型在训练时严重依赖大规模 AI 加速器集群，而现有的训练框架在 **Ascend NPU** 上存在以下瓶颈：
- **串行化执行**：MoE 操作符（如 Dispatch、GMM、SwiGLU、Combine）以独立 kernel 形式逐个执行，导致 AIC（AI Cube）和 AIV（AI Vector）单元交替空闲。
- **通信与计算无法细粒度重叠**：AllToAll 通信作为全局同步操作，暴露大量通信延迟，难以与矩阵计算有效重叠。
- **调度开销高**：动态调度在运行时进行任务选择和依赖检查，引入额外开销。

这些问题导致硬件利用率低下，尤其是 AIC 单元的 MAC 利用率仅为约 67%，且 39% 的 EP 通信延迟暴露在关键路径上。

---

### 提出了什么新方法或新思路
本文提出 **HyperParallel-MoE**，一种面向 Ascend NPU 的 MoE 训练编译与调度框架，其核心创新包括：

#### （1）**Tile-Level Heterogeneous Task Abstraction**
将 MoE-FFN 的执行从“算子级”转化为“瓦片级异构任务流”，统一表示 **communication、matrix computation (AIC) 和 vector computation (AIV)**，实现跨算子、跨资源的联合调度。

#### （2）**AIV-Driven One-Sided Communication**
- 将 AllToAll 通信分解为由 AIV 驱动的 `put_mem_signal` 任务。
- 每个通信 tile 独立完成远程写入并更新事件计数器，无需主机侧同步。
- 下游 GMM 可在部分通信数据就绪后立即启动，实现 **fine-grained communication-computation overlap**。

#### （3）**Dependency-Preserving Tile Task Generation**
通过 **Operator Dependency Graph (ODG)** 和 **SplitSpec** 进行跨算子的分块传播（split propagation），确保：
- 分块边界对齐（如 GMM 行分块与 SwiGLU 输入对齐）
- 保持原有算子优化（如 GMM 的 expert-local 缓存策略）
- 显式维护数据依赖关系

#### （4）**Event-Driven Static Scheduling**
- 在编译期生成 **静态任务流（CTQ/VTQ）** 和事件同步逻辑（CrossCoreSetFlag/CrossCoreWaitFlag）。
- 运行时仅需轻量级的任务获取、事件等待和触发，避免运行时调度决策开销。
- 支持 **rank-aware task reordering (RATR)** 和 **cache-guided GMM interleaving** 等优化。

#### （5）**Unified Runtime Execution**
单次 kernel launch 启动整个 MoE-FFN 正向/反向任务流，AIC 和 AIV 工作线程并发消费各自队列中的任务，实现：
- 细粒度重叠（communication、GMM、SwiGLU、Combine 并发）
- L2 cache 复用机会暴露给调度器

---

### 相比现有方法的优势
| 方面 | 现有方法（如 MindSpore 基线） | HyperParallel-MoE |
|------|-------------------------------|------------------|
| 执行模型 | Kernel-by-kernel 串行执行 | Tile-level 异构任务流 |
| 通信机制 | Host-driven collective AllToAll | Device-side one-sided communication |
| 调度方式 | 动态调度或粗粒度融合 | 静态编译 + 事件驱动同步 |
| 硬件利用 | AIC/AIV 交替空闲 | AIC/AIV 并发执行 |
| 算子复用 | 需重写算子内核 | 复用现有优化算子（GMM, SwiGLU 等） |

> ✅ **优势总结**：在不重写高性能算子的前提下，通过编译层优化实现细粒度异构并行，显著提升 Ascend NPU 上 MoE 训练效率。

---

## 2. 核心实验方法和设置

### 使用的模型与配置
- **模型**：DeepSeek-V3-style MoE-FFN
  - Sequence Length: 4096
  - Hidden Size: 7168
  - Intermediate Size: 2048
  - Top-k: 8
  - 数据类型: bf16
- **硬件平台**：Ascend A3 NPU
  - 每设备：25 AIC units, 50 AIV units, 192MB L2 cache
  - 总设备数：64（即 128 ranks，每 NPU 两个 device）

### 并行配置（Expert Parallelism）
| EP Setting | Total Experts | Local Experts per Rank |
|------------|--------------|------------------------|
| EP4        | 32           | 8                      |
| EP8        | 64           | 8                      |
| EP16       | 128          | 8                      |
- 其他并行维度：dp=32, tp=2

### 评估指标
1. **Module-level Latency**：Dispatch 到 Combine 的端到端延迟（正向 + 反向）
2. **End-to-End Training Step Latency**：完整训练步耗时
3. **Speedup**：相对于基线的加速比

### 基线方法对比
- **Baseline**：标准 operator-by-operator 执行路径
  - 每个算子单独 launch kernel
  - 使用集体通信（collective AllToAll）
  - AIC/AIV 无法并发执行
- **HyperParallel-MoE**：完整优化版本（含 one-sided comm、RATR、GMM interleaving）

---

## 3. 主要实验结果和性能指标

### 模块级性能（Dispatch-to-Combine Latency）

| EP Setting | Baseline (ms) | HyperParallel-MoE (ms) | Speedup |
|------------|---------------|-------------------------|---------|
| EP4        | 44.2          | 29.6                    | **1.49×** |
| EP8        | 47.1          | 29.9                    | **1.58×** |
| EP16       | 48.9          | 31.1                    | **1.57×** |

> 🔹 正向加速比达 **1.60–1.68×**，反向达 **1.44–1.53×**

![图7](https://via.placeholder.com/400x200?text=Figure+7:+Latency+Breakdown)  
*图示：在平衡路由下，HyperParallel-MoE 显著降低前向与反向延迟*

---

### 端到端训练步性能（Sampled Natural Routing）

| EP Setting | Speedup (vs Baseline) |
|------------|-----------------------|
| EP4        | **1.08×**             |
| EP8        | **1.09×**             |
| EP16       | **1.09×**             |

> 🔹 尽管 MoE-FFN 仅是训练流程的一部分，仍能带来近 **9% 的端到端加速**，说明优化具有实际价值。

---

### 微基准测试（Microbenchmarks）

#### （1）Tile Interleaving 对 L2 Cache Hit Rate 的影响（SwiGLU+Add）
| M (Row Dim) | Serial L2 Hit Rate | Interleaved L2 Hit Rate | 提升倍数 |
|-------------|--------------------|--------------------------|--------|
| 32K         | 5.20%              | 25.44%                   | **4.9×** |

> 🔹 由于静态调度将生产者（SwiGLU）与消费者（Add）任务紧密排列，中间结果保留在 L2 cache 中，减少 HBM 访问。

#### （2）Static vs Dynamic Scheduling 开销对比
| M (Row Dim) | Dynamic Scheduling (μs) | Static Scheduling (μs) | 加速比 |
|-------------|--------------------------|------------------------|--------|
| 2K          | 413.0                    | 54.0                   | **7.65×** |
| 32K         | 862.8                    | 588.4                  | **1.47×** |

> 🔹 动态调度每任务引入 ~2.36 μs 开销，远高于静态调度的 ~0.1 μs，验证了 **静态调度必要性**。

---

## 4. 关键结论和发现

### 主要发现
1. **Ascend NPU 的异构架构潜力巨大**：AIC/AIV 可并发执行，但现有框架未能利用。
2. **Tile-level scheduling 是解锁性能的关键**：将通信、计算分解为可调度的 tile 任务，实现细粒度重叠。
3. **One-sided communication + event-driven sync 可消除主机同步瓶颈**。
4. **静态调度优于动态调度**：在稳定训练阶段，离线生成调度计划可大幅降低运行时开销。
5. **无需重写算子即可获得显著加速**：通过封装现有 GMM/SwiGLU 等算子为 task handler，实现低侵入集成。

---

### 方法的局限性
1. **依赖固定形状桶（shape bucketing）**：对于频繁变化的序列长度或专家分布，需重新编译 SSC。
2. **当前主要针对 MoE-FFN 模块**：未覆盖 Attention 或其他复杂模块的全图融合。
3. **依赖 Ascend 特定编程接口**（如 put_mem_signal、event flag），迁移至其他硬件需适配。
4. **编译期开销存在**：虽然不在训练主路径中，但对于新配置仍需预处理时间。

---

### 未来工作方向
1. **扩展至更多算子和模型结构**：支持 Attention、MLP、LayerNorm 等的统一 tile-level 调度。
2. **支持动态形状的混合调度策略**：结合静态主干 + 动态 fallback 处理异常 shape。
3. **跨节点任务调度探索**：将 tile-level 思想推广至分布式多机场景。
4. **自动调优调度策略**：基于 profiling 自动选择最优 tile size 和任务顺序。
5. **集成进更高层框架**：在 MindFormers 或 Megatron 级别提供一键启用能力。

---

> ✅ **总体评价**：HyperParallel-MoE 成功揭示了在 **Ascend NPU** 上通过 **tile-level heterogeneous scheduling** 可显著提升 MoE 训练效率，为未来高效 MoE 系统设计提供了新范式。

</details>

---

### 2. [Fast-dDrive: Efficient Block-Diffusion VLM for Autonomous Driving](https://arxiv.org/abs/2605.23163)

**Authors**: Kewei Zhang, Jin Wang, Sensen Gao, Chengyue Wu, Yulong Cao, Songyang Han, Boris Ivanovic, Langechuan Liu, Marco Pavone, Song Han, Daquan Zhou, Enze Xie  
**Category**: cs.CL  
**Published**: 2026-05-25  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.23163v1  

#### Abstract
End-to-end autonomous driving via Vision-Language-Action (VLA) models demands a precarious balance between high-fidelity trajectory planning and efficient inference. Existing paradigms typically fall short: autoregressive (AR) VLAs are memory-bandwidth-bound on edge hardware and prone to exposure-bi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Fast-dDrive: Efficient Block-Diffusion VLM for Autonomous Driving》总结**

---

## **1. 主要贡献和创新点**

### **解决的问题**
现有的端到端自动驾驶模型主要依赖两种范式：
- **Autoregressive (AR) VLA**：推理效率低，在边缘硬件上受限于内存带宽（memory-bandwidth-bound），且存在暴露偏差（exposure bias），导致轨迹误差累积。
- **Full-sequence diffusion VLA**：虽能提供全局上下文，但无法复用 KV Cache，推理延迟高，并存在“逻辑泄漏”（logical leakage）——即后阶段的规划可能反向影响前阶段的感知判断，违背“先感知、再决策”的因果顺序。

此外，现有方法未充分利用驾驶 VLA 输出的**结构化特性**（如 JSON 格式的 CoT 输出），造成计算资源浪费。

---

### **提出的新方法与创新思路**
Fast-dDrive 提出了一种 **block-diffusion VLA 架构**，结合结构先验与高效推理机制，实现高精度与高吞吐的统一。其核心创新包括：

#### ✅ **Section-Aware Structured Diffusion (SASD)**
- 将输出划分为语义明确的四个 section：
  1. `critical_objects`（关键物体检测）
  2. `explanation`（自然语言解释）
  3. `future_meta_behavior`（纵向/横向行为决策）
  4. `trajectory`（轨迹坐标）
- 利用固定 schema 中的结构 token（如括号、字段名）构建 **frozen scaffold（冻结支架）**，仅对 value token 进行去噪。
- 在训练中引入：
  - **Section-weighted loss**：为安全关键部分（如 trajectory）分配更高损失权重。
  - **Section-adaptive Beta noise schedule**：根据各 section 难度定制噪声策略。

> 👉 优势：保证 100% 结构正确性，减少 ~30% 去噪负担，提升安全相关任务的学习优先级。

#### ✅ **Scaffold Speculative Decoding (SS)**
- 扩展自 self-speculative decoding，利用 scaffold 自动接受结构 token，仅对 value token 执行 draft-verify 循环。
- 流程：
  1. **Auto-accept scaffold tokens**
  2. **MDM head 并行生成 value draft**
  3. **AR head 因果验证 draft tokens**

> 👉 优势：相比纯 AR 推理大幅提速；相比 full-sequence diffusion 支持 KV Cache 复用，避免重复计算。

#### ✅ **Shared-Prefix Test-Time Scaling**
- 在推理时从共享的 prefix KV Cache 分叉出多个 trajectory rollout。
- 仅在 trajectory section 启用随机采样，最后取平均轨迹。
- 成本仅为单次推理的一小部分，却显著降低预测方差。

> 👉 优势：以极低成本实现 test-time compute scaling，提升鲁棒性和准确性。

---

### **相比现有方法的优势**
| 维度 | AR VLA | Full-sequence Diffusion | Fast-dDrive |
|------|--------|--------------------------|------------|
| 推理效率 | ❌ 内存带宽瓶颈，1 token/step | ❌ 无 KV Cache 复用 | ✅ 支持 KV Cache + 并行块解码 |
| 准确性 | ❌ 暴露偏差累积误差 | ⚠️ 全局上下文但有逻辑泄漏 | ✅ 块内双向 + 块间因果，保持一致性 |
| 结构利用 | ❌ 逐 token 生成所有内容 | ❌ 忽视 schema 可预测性 | ✅ 利用 scaffold 减少冗余计算 |
| 可扩展性 | ❌ Best-of-N 成本高 | ❌ 多 rollout 不经济 | ✅ Shared-prefix rollout 成本极低 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **WOD-E2E (Waymo Open Dataset End-to-End)**  
  - 包含 4,021 个长尾驾驶场景，每个 20 秒。
  - 使用前 12 秒输入，预测未来 5 秒轨迹。
  - 提供 chain-of-thought 注释，适配四段式 JSON 输出格式。

- **nuScenes**  
  - 1,000 个城市驾驶片段，标注频率 2Hz。
  - 输入为过去 1 秒的 front camera 图像帧 + ego state + 导航指令。

> 两数据集均不使用 LiDAR、radar 或 HD map，仅依赖多视角视觉与 ego 状态。

---

### **实验设置与评估指标**

#### **模型架构基础**
- 主干网络：**Qwen2.5-VL-3B**
- 转换方式：基于 **Fast-dVLM** 框架直接将 AR VLM 转为 block-diffusion 形式。
- 训练平台：8×H100 GPU，fine-tune 3 epochs。

#### **评估指标**

| 类型 | 指标 |
|------|------|
| **规划精度** |  
| WOD-E2E | ADE@3s, ADE@5s（平均位移误差）、RFS（Rater Feedback Score，人类对齐评分） |
| nuScenes | L2 Error @1s, 2s, 3s, Avg |
| **推理效率** |  
| | Latency (ms/sample), TPS (Tokens Per Second), Tok/Step（每步提交 token 数） |

#### **对比基线**
- **AR Baselines**: AutoVLA, Poutine-Base, LightEMMA, OpenEMMA
- **Diffusion Baseline**: dVLM-AD（full-sequence MDM）
- **消融对照**：不同 SASD 组件组合、是否启用 scaffold speculative decoding

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 📊 **WOD-E2E Test Set 结果（Table 2）**
| 方法 | Paradigm | RFS↑ | ADE@5s↓ | ADE@3s↓ | TPS↑ |
|------|---------|-------|---------|---------|-------|
| AutoVLA | AR | 7.557 | 2.958 | 1.351 | 51.2 |
| dVLM-AD | Diffusion | 7.633 | 3.022 | 1.285 | 35.2 |
| **Fast-dDrive (Scaffold Spec)** | **Block Diffusion** | **7.823** | **2.907** | **1.254** | **210.4** |
| + Inference Scaling (N=4) | Block Diffusion | 7.827 | **2.821** | **1.240** | 114.7 |

> 🔥 **Fast-dDrive 达成当前 diffusion-based VLA 中最高的 RFS 和最低的 ADE 指标**。

#### 📈 **nuScenes Val Set L2 Error（Table 3）**
| 方法 | L2@1s | L2@2s | L2@3s | **Avg↓** |
|------|-------|-------|-------|--------|
| UniAD | 0.20 | 0.42 | 0.75 | 0.46 |
| dVLM-AD | 0.15 | 0.40 | 0.68 | 0.41 |
| **Fast-dDrive** | **0.12** | **0.33** | **0.50** | **0.32** |

> ✅ **Fast-dDrive 在 avg L2 error 上达到 0.32m，较 dVLM-AD 提升 22%**。

---

### **推理效率对比（Table 4）**
在单张 H100 上，batch size = 1：

| 方法 | Latency (ms) | TPS | Tok/Step | ADE@5s |
|------|---------------|-----|-----------|--------|
| AR Baseline | 7,855 | 51.6 | 1.0 | 2.083 |
| dVLM-AD | 9,575 | 35.2 | 2.82 | 3.024 |
| Fast-dDrive (Self-Spec) | 3,714 | 109.0 | 2.41 | 1.973 |
| **Fast-dDrive (Scaffold Spec)** | **1,919** | **210.4** | **4.90** | **1.982** |
| **+ SGLang Serving** | **665** (**11.8× faster**) | **608.5** | 4.93 | 1.995 |

> ⚡ **Scaffold Spec 实现 4.1× 端到端加速，集成 SGLang 后达 11.8× 加速，接近实时部署要求**。

---

### **消融实验结果（Table 5）**

| IWL | SNS | ADE@5s↓ | RFS↑ |
|-----|-----|--------|------|
| × | × | 2.028 | 7.735 |
| √ | × | 2.003 | 7.855 |
| × | √ | 2.050 | 7.807 |
| √ | √ | **2.034** | **7.916** |

- **IWL（Section-Importance-Weighted Loss）** 是主要增益来源，显著提升 RFS。
- **SNS（Section-Adaptive Noise Schedule）** 提供互补改进。
- 二者联合取得最佳综合表现。

---

## **4. 关键结论和发现**

### **主要发现**
1. **结构化先验可极大提升扩散模型效率与质量**  
   利用 JSON schema 构建 scaffold，不仅能消除语法错误，还能节省 ~30% 的去噪计算量。

2. **Block-diffusion + Scaffold Speculative Decoding 实现 AR 级质量与远超 AR 的吞吐**  
   在保持生成质量的同时，实现高达 **12× 的端到端推理加速**，首次使大容量 VLA 接近车载实时部署门槛。

3. **Test-time scaling 可低成本提升稳定性**  
   通过 shared-prefix 多 rollout 平均轨迹，可在增加少量计算的前提下有效抑制预测方差，进一步降低 ADE。

4. **Section-aware 训练显著增强安全性关键任务的表现**  
   对 trajectory 和 meta-behavior 加权训练，使得模型更关注高风险决策环节。

---

### **局限性（Limitations）**
1. **依赖预定义 JSON schema**  
   当任务需求变化（如新增 object category 或 reasoning depth）时，需手动调整模板。

2. **Shared-prefix rollout 仍需额外计算**  
   在极端低延迟场景下，即使是分摊后的 rollout 成本也可能不可接受。

3. **当前评估为 open-loop setting**  
   缺乏 closed-loop 仿真验证，尚未测试其在动态交互环境中的反应能力。

---

### **未来工作方向**
- 支持动态 schema 的 adaptive scaffold 构建。
- 将 shared-prefix rollout 与 reward model 结合，进行智能轨迹选择。
- 扩展至闭环模拟器中进行 reactive planning 与 interaction modeling。
- 探索更多模态（如 V2X、地图先验）与 block-diffusion 的融合方式。

---

## ✅ 总结
**Fast-dDrive** 成功地将 **结构先验、block-diffusion 架构、speculative decoding 与 test-time scaling** 有机结合，重新定义了自动驾驶 VLA 的 **速度-精度前沿**。它不仅在多个 benchmark 上实现了 SOTA 精度，还通过系统级优化达成 **12× 推理加速**，为高容量 VLA 走向实车部署提供了可行路径。该工作揭示了一个重要原则：**当输出具有已知结构时，将其显式编码进生成过程，可在质量和效率上获得双重增益**。

</details>

---

### 3. [AlignedServe: Orchestrating Prefix-aware Batching to Build a High-throughput and Computing-efficient LLM Serving System](https://arxiv.org/abs/2605.23389)

**Authors**: Fengyao Bai, Hongbin Zhang, Zhitao Chen, Jiangsu Du, Zhiguang Chen, Yutong Lu  
**Category**: cs.DC  
**Published**: 2026-05-25  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.23389v1  

#### Abstract
High-throughput inference serving is essential for applications built on large language models (LLMs). Existing serving frameworks reduce request-level and batch-level bubbles through batching and scheduling, but often overlook bubbles within each decode iteration. Tokens generated in the same itera...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AlignedServe: Orchestrating Prefix-aware Batching to Build a High-throughput and Computing-efficient LLM Serving System**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统的大语言模型（LLM）推理服务系统在处理高并发请求时，普遍存在**多级“气泡”（bubbles）开销**，导致GPU资源利用率低、吞吐量受限。现有研究主要关注**request-level**（请求级）和**batch-level**（批级别）的气泡优化（如连续批处理 Continuous Batching），但忽略了**decode阶段每个迭代内部的气泡（iteration-level bubbles）**。

具体而言，在同一个 decode iteration 中，不同请求生成 token 所依赖的 **KVCache 长度不同**，导致计算成本不均。长前缀（long prefix）请求成为瓶颈，短前缀请求必须等待，造成 GPU 空转，形成 **iteration-level bubbles**。

---

### **提出了什么新方法或新思路**
为解决上述问题，本文提出：

#### ✅ **1. Prefix-aware Batching（前缀感知批处理）策略**
- 将具有**相似 prefix 长度**（即输入 prompt + 已生成 token 数量）的请求分到同一批次中。
- 确保同一批次内所有 token 在每个 decode iteration 中的计算成本相近，从而**消除 iteration-level bubbles**。

#### ✅ **2. AlignedServe 框架设计**
一个全新的 LLM 推理服务框架，核心组件包括：
- **KV Pool**：利用大容量 CPU 内存暂存大量飞行中的请求及其 KVCache，支持大规模请求缓冲以实现高效的 prefix-aware 分组。
- **分离式架构（Disaggregated Architecture）**：将 Prefill 和 Decode 阶段分配到不同的 GPU 上执行。
- **Candidate Batch Buffer & Candidate Requests Buffer**：预取机制的关键缓冲区，部署在 Prefill GPU 的 HBM 中。

#### ✅ **3. Batch-level Scheduling Policy（批级别调度策略）**
- 不再基于单个请求进行调度（如 FCFS），而是以“批”为单位进行调度。
- 支持动态填充运行批次，并通过智能调度减少调度间隙。

#### ✅ **4. GPU-Prefetch-For-GPU 架构（首创）**
- 利用 **NVLink** 高带宽链路，由 **Prefill GPU 预取 KVCache 到其 HBM**，再传输给 Decode GPU。
- 避免直接通过 PCIe 从 CPU 内存传输 KVCache，显著降低通信延迟。
- **这是首个采用“一个 GPU 为另一个 GPU 预取 KVCache”的设计**。

---

### **相比现有方法的优势**
| 维度 | 传统方法（如 vLLM, Orca） | AlignedServe |
|------|---------------------------|-------------|
| 批处理粒度 | 忽略 prefix 长度差异，混合长短请求 | 按 prefix 长度聚类，消除迭代内不平衡 |
| 资源管理 | KVCache 直接驻留 GPU 或通过 PCIe 交换 | 利用 CPU 大内存缓存 + NVLink 预取 |
| 调度机制 | Request-level / Iteration-level 调度 | Batch-level 调度 + 动态补充 |
| 性能目标 | 提高吞吐为主 | 同时提升吞吐、降低延迟、提高 GPU 利用率 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
#### **合成工作负载（Synthetic Workloads）**
- 控制长短请求比例（短请求 <1k tokens，长请求 1k–8k tokens）
- 设置不同短请求占比（70% ~ 95%），验证对混合负载的鲁棒性

#### **真实应用工作负载（Application Workloads）**
| 数据集 | 描述 |
|-------|------|
| **AzurePublicDataset** | 微软发布的真实 LLM 请求轨迹，涵盖对话与代码生成场景 |
| **ShareGPT** | 用户与 ChatGPT-4 的多轮对话记录 |
| **LongBench** | 双语长上下文理解评测基准，测试模型处理长序列能力 |

---

### **实验设置**
- **硬件平台**：
  - 2×Intel Xeon Platinum 8462Y+
  - 8×NVIDIA H100 GPUs（通过 NVLink 互联）
  - 800GB DRAM，PCIe 5.0 连接 CPU-GPU
- **模型**：
  - OPT 系列：OPT-2.7B, OPT-6.7B, OPT-13B, OPT-30B
  - 使用 FP16 精度
- **参数配置**：
  - Quad-tree 管理 prefix 范围 [1, 65536]
  - `Bmax` = 40% GPU block 容量，`Kmin` = 36

---

### **评估指标**
| 指标 | 说明 |
|------|------|
| **Decoding Throughput (tokens/s)** | 单位时间内成功生成的 token 数量，衡量系统吞吐能力 |
| **P99 Latency of TPOT (Time Per Output Token)** | 99百分位每次输出 token 的延迟，反映响应稳定性 |
| **TTFT (Time to First Token)** | 首个 token 返回时间，影响用户体验 |
| **Ablation Study** | 消融实验分析各模块贡献（如是否启用 prefetch、prefix-aware batching） |

---

### **基线方法对比**
| 基线系统 | 特点 |
|--------|------|
| **vLLM** | 实现 PagedAttention 和 Continuous Batching，主流开源框架 |
| **DistServe** | 分离 Prefill/Decode 阶段，采用 disaggregated 架构 |
| **FastGen (DeepSpeed-FastGen)** | 高吞吐 LLM 推理系统，支持混合 prefill/decode 批处理 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **解码吞吐量提升**
- 在 **OPT-2.7B + 95% 短请求负载** 下：
  - 相比 vLLM 提升 **1.32×**
  - 相比 DistServe 提升 **1.35×**
  - 相比 FastGen 提升 **1.85×**
- 在 **真实应用负载下最大提升达 1.98×**
  - AzurePublicDataset 上达 **1.98×** 于 FastGen
  - ShareGPT 上达 **1.98×**

> 💡 图 8 显示，在 LongBench、ShareGPT、AzurePublicDataset 上，AlignedServe 始终领先。

#### 🔹 **延迟显著下降**
- **P99 TPOT Latency 最多降低 7.4×**
  - AzurePublicDataset 上较 DistServe 降低 **7.4×**
  - ShareGPT 上降低 **1.65×**
  - LongBench 上降低 **2.1×**
- 合成负载下平均降低 **1.74× ~ 3.05×**

> 💡 图 10 表明，无论模型大小，AlignedServe 的 per-iteration 延迟远低于基线。

#### 🔹 **调度效率更高**
- **>95% 的迭代调度耗时 <5ms**
  - DistServe 中 80% 的迭代调度耗时 >10ms（图 11）
- NVLink 预取大幅缩短 KVCache 传输时间

---

### **消融实验结果（Ablation Study）**

#### ✂️ **移除 GPU Prefetching 的影响**
- 在 AzurePublicDataset 上：
  - 移除 Prefetch → 吞吐下降 **14.73%**
  - 同时移除 Prefetch + Prefix-aware Batching → 吞吐再降 **28.51%**

> 说明两项技术协同增效，尤其 **GPU Prefetching 对性能至关重要**。

#### ✂️ **Prefix-aware vs FCFS 批处理对比**
- 在 LongBench 和 ShareGPT 上比较：
  - 使用 Prefix-aware Batching：**>90% 的 iteration 在 30ms 内完成前向计算**
  - 使用 FCFS：**<10% 的 iteration 在 30ms 内完成**

> 图 13 清晰展示前缀感知批处理有效抑制了 iteration-level bubbles。

#### ✂️ **Batch Switch 开销分析**
- Batch Switch（即不同批次共存）仅发生在 **8.61% ~ 12.37% 的 iteration 中**
- 表明系统大部分时间保持“对齐”状态，设计哲学未被频繁破坏

#### ✂️ **KV Pool 内存占用**
- 实际使用内存：**20GB ~ 250GB**（远小于 800GB 上限）
- 表明 CPU 内存可支撑大规模飞行请求缓存

#### ✂️ **TTFT 影响可控**
- 平均 TTFT：
  - ShareGPT: **1.49s**
  - LongBench: **2.54s**
- 最大约 30s，可通过启用饥饿处理机制调节

---

## **4. 关键结论和发现**

### **主要发现**
1. **Iteration-level bubbles 是限制 LLM 推理效率的重要因素**，此前被广泛忽视。
2. **Prefix-aware Batching 能有效消除此类气泡**，使同批请求计算节奏一致。
3. **利用 CPU 大内存 + GPU 预取机制（via NVLink）可高效支持该批处理策略**，避免 PCIe 成为瓶颈。
4. **AlignedServe 在真实和合成负载下均显著优于 SOTA 系统**，实现高吞吐与低延迟兼得。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **引入额外首 token 延迟（TTFT）** | 请求需等待同类前缀积累才能成批，可能增加初始延迟 |
| **依赖 NVLink 硬件环境** | 若无 NVLink，则退化为 PCIe 传输，性能优势减弱 |
| **Batch Switch 仍存在** | 当无法及时补充同类请求时，会引入异构批次，轻微违背设计原则 |
| **Quad-tree 参数需调优** | 如 `Bmax`, `Kmin` 需根据硬件和负载调整 |

---

### **未来工作方向**
1. **自适应动态参数调整**：根据实时负载自动调节 batching 窗口和调度策略。
2. **支持跨节点分布式部署**：扩展至多机多卡集群，进一步放大缓冲池规模。
3. **结合共享前缀优化技术（如 HotPrefix, BatchLLM）**：联合优化冗余计算与调度效率。
4. **探索更细粒度的调度单元**：例如基于 token-level 的弹性调度。

---

> ✅ **一句话总结**：  
> AlignedServe 首次识别并系统性解决了 LLM 推理中的 **iteration-level bubbles** 问题，通过 **Prefix-aware Batching + GPU-Prefetch-For-GPU 架构**，实现了高达 **1.98× 吞吐提升** 和 **7.4× 延迟降低**，是构建高性能 LLM Serving 系统的重要进展。

</details>

---

### 4. [Steered Generation via Gradient-Based Optimization on Sparse Query Features](https://arxiv.org/abs/2605.23040)

**Authors**: Sumanta Bhattacharyya, Pedram Rooshenas  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.23040v1  

#### Abstract
Latent steering exploits internal representations of Large Language Models (LLMs) to guide generation, yet interventions on dense states can entangle distinct semantic features. In this paper, we investigate attention query activations as a high-fidelity site for precise control, hypothesizing that ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Steered Generation via Gradient-Based Optimization on Sparse Query Features*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLMs）在实际应用中需要满足用户定义的**风格、安全性和逻辑一致性**等约束条件。然而，传统的控制生成方法存在以下问题：

- **Prompt Engineering**：依赖上下文中的指令或示例（如 In-Context Learning），但长提示容易稀释控制信号，且对小模型效果有限。
- **Dense Latent Steering**：通过修改残差流（residual stream）或注意力头输出进行干预，但由于激活空间是“密集”的，多个语义特征纠缠在一起（superposition hypothesis），导致控制不精确、副作用多。

本文旨在解决**如何实现更精细、可靠且低扰动的可控文本生成**，尤其是在硬性规则约束（如路径规划）和软性语义风格（如教学反馈复杂度）两种场景下。

---

### **提出了什么新方法或新思路**

作者提出了一种名为 **Prototype-Based Sparse Query Optimization (SAE-OPT)** 的新框架，其核心思想是：

#### ✅ **Query-Level Intervention**
不再干预整个残差流，而是选择 **attention query 激活**作为干预点。因为 query 决定了模型从上下文中检索哪些信息，局部扰动 query 可以引导 attention 权重变化，从而影响生成内容，同时避免全局状态被覆盖。

#### ✅ **Sparse Autoencoder (SAE) 分解**
将 query 激活映射到一个**稀疏表示空间**（sparse latent space）。SAE 强制大多数神经元为零，仅少数活跃，使得每个特征更具可解释性和独立性，减少纠缠。

#### ✅ **Gradient-Based Prototype Optimization**
引入基于原型网络（Prototypical Networks）的思想，在稀疏空间中为每类目标行为（如“安全路径”、“高阶认知”）构建一个**prototype（原型中心）**。然后通过梯度上升优化当前样本的稀疏表示 $ z $，使其更接近目标 prototype，而不是简单地加上一个静态向量。

> 公式化目标函数：
> $$
> \max_z \log P(a = k_T | z), \quad \text{其中 } P(a=k_T|z) = \frac{\exp(-\|z - c_{k_T}\|^2)}{\sum_k \exp(-\|z - c_k\|^2)}
> $$

这使控制过程成为一个动态、自适应的搜索问题，而非固定偏移。

---

### **相比现有方法的优势**

| 维度 | 优势 |
|------|------|
| **控制精度** | 稀疏 + query 级干预 → 更少特征纠缠，更高保真度 |
| **扰动程度** | 干预发生在 attention query 层 → 对残差流扰动小，保持原始语义完整性 |
| **灵活性** | 梯度优化支持渐进式调整，避免“过冲”或模式崩溃 |
| **通用性** | 同一套框架适用于硬规则任务（Gridworld）和软风格任务（Bloom’s Taxonomy） |

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### 📌 **Textualized Gridworld (TGW)**
- 改编自经典强化学习环境，用于测试**硬性规则控制能力**。
- 输入是一个带墙的网格地图，要求模型输出从起点到终点的一条路径序列。
- 定义三种目标属性：
  - **Short**: 最短路径
  - **Safe**: 远离墙壁（低 wall-adjacency score）
  - **Long**: 尽可能长的有效路径（NP-hard）
- 数据集包含 15,250 个样本，确保三个目标路径互不相同。
- 评估标准严格：必须路径有效 + 满足目标属性才算成功。

#### 📌 **Educational Cognitive Style Dataset (Bloom’s Taxonomy)**
- 自建教育反馈数据集，用于测试**软性语义风格控制**。
- 包含 10 个 C++ 编程概念（如递归、指针等），每个有多个题目和错误变体。
- 针对每个答案生成六种不同认知层次的反馈（Remember → Create）。
- 总共 3,000 个样本，按主题划分训练/验证/测试集以保证泛化性。

#### 📌 **TruthfulQA**
- 外部基准，用于评估**真实性提升**。
- 开放式问答任务，衡量模型是否生成真实且信息丰富的回答。
- 使用 %Truthful × %Informative (%TxI) 作为主指标。

---

### **实验设置与评估指标**

| 设置项 | 描述 |
|-------|------|
| **模型** | Qwen-3-4B-Instruct, Phi-3-Mini-4K-Instruct, LLaMA3.1-8B-Instruct |
| **干预层** | 中间层（Qwen: layer 17; Phi/LLaMA: layer 15） |
| **SAE 架构** | 单层 ReLU Autoencoder，L1 正则化，decoder 列归一化 |
| **训练方式** | 先微调模型掌握任务 → 冻结模型 → 在 query 激活上训练 SAE → 推理时优化稀疏 latent |
| **Steering Strength ($\eta$)** | Qwen/LLaMA: 0.5；Phi: 0.8 |

---

### **基线方法对比**

| 基线 | 方法描述 |
|------|----------|
| **ICL** | 少样本提示学习（Few-shot In-Context Learning） |
| **CAA** | Contrastive Activation Addition，在残差流中加差值向量 |
| **ITI** | Inference-Time Intervention，干预 top-K 注意力头 |
| **SAE-SSV** | 静态稀疏向量添加（Static Sparse Steering Vector） |
| **DENSE-OPT** | 在原始 dense query 上做梯度优化（无 SAE） |
| **DISCO-Q** | 静态 query-level 干预（Torop et al., 2025） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔹 **TGW 路径规划任务（表1）**

| 方法 | Qwen 成功率 (Short/Safe/Long) | 逻辑违反率 ↓ |
|------|-------------------------------|-------------|
| ICL | 29.3 / 3.7 / 3.2 | 57.1 / 55.5 / 76.0 |
| CAA | 53.9 / 4.2 / 5.6 | 38.8 / 39.6 / 19.9 |
| **SAE-OPT (Ours)** | **55.5 / 15.2 / 8.4** | **21.0 / 20.1 / 21.6** |

> ✅ 在所有模型上均取得最高或接近最高的成功率  
> ✅ 特别是在 **Safe 和 Long** 这类困难任务上显著优于其他方法（Safe 提升超 3 倍）  
> ✅ 违反率稳定较低，说明未牺牲结构有效性换取属性匹配

#### 🔹 **Bloom’s Taxonomy 教学反馈控制（表2）**

| 方法 | GPT-4o 平均命中率 ↑ | Claude 平均命中率 ↑ |
|------|---------------------|--------------------|
| CAA | 22.2% | 25.0% |
| ITI | 16.0% | 10.1% |
| **SAE-OPT (Ours)** | **25.4%** | **27.1%** |

> ✅ 在最难的 **Remember** 和 **Create** 类别上提升最大  
> ✅ 显示出对模型默认 bias（偏向 Evaluate）的有效克服能力

#### 🔹 **AxBench 三轴综合评分（表8）**

| 目标 | 方法 | HM（Harmonic Mean）↑ |
|------|------|------------------|
| Understand | SAE-OPT | **0.950** > DENSE-OPT (0.907) > CAA (0.752) |
| Create | SAE-OPT | **0.764** > DISCO-Q (0.756) > DENSE-OPT (0.697) |
| Evaluate | SAE-OPT | **1.399** > DENSE-OPT (1.389) |

> ✅ 在保持 fluency 和 instruction-following 的前提下，显著提升 concept alignment

#### 🔹 **TruthfulQA 真实性控制（表3）**

| 方法 | Qwen %TxI ↑ | Mean JSD ↓ |
|------|------------|-----------|
| DISCO-Q (static) | 0.7078 | 0.0308 |
| **DENSE-OPT (gradient)** | **0.7407** | **0.0024** |

> ✅ 梯度优化比静态加向量提升超过 3% TxI，同时分布扰动降低一个数量级  
> ✅ 最大 JSD < 0.03，远低于静态方法的 0.13~0.28

---

### **消融实验结果**

#### 🔸 **是否使用锚定（anchoring）？**
- **SAE-OPT-ANCH** 添加 L2 锚定惩罚偏离原始 latent
- 结果：几乎完全坍缩回 “Evaluate” 类别（图10），说明锚定会阻碍探索远离初始状态的目标区域。

> ❌ 锚定虽能防止过度修改，但也限制了表达能力，不适合强控制任务。

#### 🔸 **是否使用稀疏正则化？（L1 vs L2 Autoencoder）**
- 比较 L1（稀疏）与 L2（非稀疏）autoencoder 在梯度优化中的表现
- 衡量指标：**Non-target Drift** —— 优化过程中对非目标类别的原型距离变化

> ✅ 图3 显示：L1 始终比 L2 产生更小的 collateral drift（最多达 4.9× 更清洁）
> ✅ 证明 L1 稀疏性有助于实现“干净”的控制，减少无关特征漂移

#### 🔸 **干预层的影响（Appendix H）**
- 在不同 attention layer 上训练 SAE 并比较效果
- 发现 **middle layers**（如 layer 15）效果最好
- 与已有研究一致：中间层编码更多结构化关系信息（Vig & Belinkov, 2019）

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Query-level 是比 residual stream 更优的干预位置**
   - 扰动传播更受限，next-token JSD 更低，任务成功率更高（图2b）
   - 支持更精细的信息检索层面控制

2. ✅ **Sparse Representations 提供更好的控制解耦性**
   - SAE 能有效分解 query 激活为可解释、独立的 feature
   - L1 正则化显著减少非目标属性的意外激活（non-target drift）

3. ✅ **Gradient-Based Optimization 优于 Static Vector Addition**
   - 动态优化能逐步逼近目标，避免一步到位带来的分布扭曲
   - 在 TruthfulQA 上验证了其优越的 accuracy-distortion trade-off

4. ✅ **统一框架适用于多种控制范式**
   - 成功应用于：
     - **Hard Logic Constraints**（TGW 路径规划）
     - **Soft Semantic Styles**（Bloom’s Taxonomy 教学反馈）
     - **External Benchmark**（TruthfulQA 真实性增强）

---

### **方法的局限性**

| 局限 | 说明 |
|------|------|
| **计算开销** | 需要预先训练 SAE，增加前期成本；推理时需多次前向/反向传播 |
| **依赖高质量支持集** | Prototype 依赖标注的支持集（support set），若标签噪声大则影响效果 |
| **尚未扩展至多属性联合控制** | 当前一次只优化单一目标 prototype，难以处理多目标权衡 |
| **SAE 可扩展性挑战** | 当前为每 head 单独训练 SAE，难以扩展到更大模型或跨层联合优化 |

---

### **未来工作方向**

1. **Cross-Layer Sparse Steering**
   - 联合多个 layer 的 query features 进行协同优化
   - 构建跨层 feature circuit 实现机制性编辑

2. **Multi-Objective Prototype Optimization**
   - 设计多目标损失函数，支持风格 + 安全 + 准确性的联合控制

3. **Efficient Online SAE Adaptation**
   - 探索轻量化、无需重新训练的 SAE 微调机制，适配新任务

4. **Integration with RLHF / DPO**
   - 将 sparse query steering 作为偏好优化的辅助信号，提升 alignment 效率

5. **Application to Multimodal Models**
   - 将该框架推广至 vision-language 模型，实现图文联合可控生成

---

> 💡 **一句话总结**：  
> 本论文提出 **SAE-OPT**，首次将 **sparse autoencoder** 应用于 **attention query 激活**，并通过 **gradient-based prototype optimization** 实现高保真、低扰动的可控生成，在硬规则与软风格任务上全面超越 prompt engineering 与 dense activation steering 方法。

</details>

---

### 5. [Convex Optimization for Alignment and Preference Learning on a Single GPU](https://arxiv.org/abs/2605.23244)

**Authors**: Miria Feng, Mert Pilanci  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.23244v1  

#### Abstract
Fine-tuning large language models (LLMs) to align with human preferences has driven the success of systems such as Gemini and ChatGPT. However, approaches like Reinforcement Learning from Human Feedback (RLHF) remain computationally expensive and complex. Direct Preference Optimization (DPO) offers ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Convex Optimization for Alignment and Preference Learning on a Single GPU**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的大语言模型（LLM）对齐方法如 **Reinforcement Learning from Human Feedback (RLHF)** 和 **Direct Preference Optimization (DPO)** 存在以下问题：
- **计算成本高**：需要多阶段训练，依赖参考模型（reference model），占用大量 VRAM。
- **优化不稳定**：DPO 训练过程中奖励波动大，收敛慢，且对超参数敏感。
- **资源门槛高**：通常需要多 GPU 或大规模算力支持，难以在单卡设备上运行。

### **提出了什么新方法或新思路**
本文提出 **COALA (Convex Optimization for Alignment and Preference Learning Algorithm)**，一种基于凸优化的轻量级偏好对齐框架，其核心思想包括：
- **凸优化重构（Convex Reformulation）**：将两层 ReLU 神经网络转换为具有理论保证的凸优化问题，从而实现全局最优求解。
- **无参考模型（Reference-Free）**：无需冻结的参考模型来稳定训练，显著降低内存开销。
- **ADMM 优化器（CRONOS）**：采用 **Alternating Direction Method of Multipliers (ADMM)** 的变体 CRONOS 进行高效优化，具备强收敛性且几乎无需调参。
- **模块化设计**：仅微调顶层的 **convex neural network (cvxNN)** 头部，冻结主干 LLM，兼顾表达能力与效率。

### **相比现有方法的优势**
| 维度 | COALA | DPO / ORPO |
|------|-------|-----------|
| **硬件需求** | 单张 RTX-4090（24GB VRAM）即可训练 Llama-3.1-8B | 需要 A100（40GB）及以上，常需 LoRA + DeepSpeed |
| **训练稳定性** | 奖励单调上升，无剧烈震荡 | DPO 奖励波动明显，易过拟合 |
| **计算效率** | 仅需 DPO 总 TFLOPs 的 ~17.6%，训练更快 | 计算密集，训练周期长 |
| **超参数敏感性** | 极低，近乎“免调参” | 对 β、学习率等高度敏感 |
| **理论保障** | 具备全局收敛性和多项式时间最优性证明 | 非凸优化，无全局最优保证 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共使用四个数据集进行综合评估：

| 数据集 | 类型 | 样本数（偏好对） | 特点 |
|--------|------|------------------|------|
| **EduFeedback** | 教育对话（合成） | 65,606 | 包含 26,621 场学生-导师多轮对话，引入“交替种群策略”生成偏好对 |
| **UltraFeedback** | 多模型反馈 | 60,917 | GPT-4 对多个开源模型输出打分后二值化 |
| **IMDb** | 情感分类 | 25,000 | 正负影评生成任务，用于测试风格控制能力 |
| **HelpSteer** | 助手行为标注 | 7,708 | 包含帮助性、连贯性等五维评分的真实人类标注 |

> ✅ **特别说明**：作者开源了 **EduFeedback** 数据集（[HuggingFace](https://huggingface.co/datasets/miria0/EduFeedback)），并提出 **Alternating Population Strategy** 自动生成高质量偏好对，无需外部重排序模型。

### **实验设置和评估指标**

#### **模型范围**
- 覆盖六种主流架构：`DistilGPT-2`, `GPT-2`, `Mistral-7B`, `Dolphin-2.6-7B`, `Llama-3.2-3B`, `Llama-3.1-8B`

#### **训练配置**
| 方法 | 硬件 | 框架 | 加速技术 |
|------|------|--------|----------|
| COALA | RTX-4090 (24GB) | JAX | CRONOS + XLA JIT 编译 |
| DPO/ORPO/SFT | A100 (40GB) | PyTorch | LoRA + DeepSpeed ZeRO-3 |

> ⚠️ **公平性控制**：所有方法均使用相同 prompt/chosen/rejected 三元组，生成长度一致，避免长度偏差影响评价。

#### **评估指标**
- **AlpacaEval2**：自动评测助手回复质量（GPT-4 作为裁判）
- **MT-Bench**：多轮对话能力评分
- **ArenaHard**：更具挑战性的对抗性评测
- **Human Evaluation**：双盲调查，107 名真实用户参与，评估实际偏好
- **TFLOPs**：衡量总计算消耗
- **Reward Margin**：训练过程中的偏好得分差值趋势

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表 1：AlpacaEval2 上的 Length-Controlled Win Rate (LC WR%)**
| 方法 \ 模型 | Llama-8B (Edu) | Mistral-7B (IMDb) | Dolphin-7B (Ultra) |
|------------|----------------|-------------------|--------------------|
| **COALA** | **40.90±0.09** | **27.64±0.27** | **31.58±0.34** |
| DPO       | 40.68±0.10      | 21.79±0.29       | 26.41±0.68        |
| ORPO      | 23.87±0.60     | 12.10±0.71       | 22.94±1.26        |
| SFT       | 10.92±0.20     | 8.16±0.11        | 14.88±0.46        |

✅ COALA 在多数场景下达到或超越 DPO，并显著优于 ORPO 和 SFT。

#### **表 2：真实人类偏好胜率（107 名参与者）**
| 方法 \ 数据集 | EduFeedback | IMDb |
|--------------|-----------|------|
| **COALA**    | **39.1%** | **42.7%** |
| DPO          | 28.8%     | 24.8%     |
| ORPO         | 15.5%     | 20.1%     |
| SFT          | 16.6%     | 12.4%     |

📌 **COALA 显著领先于所有基线**，验证其在真实用户感知上的有效性。

#### **表 3：总计算消耗（TFLOPs，EduFeedback 数据集）**
| 模型 | COALA | DPO | ORPO |
|------|-------|-----|------|
| Llama-8B | **1,805.39** | 10,253.37 | 12,352.98 |
| Mistral-7B | **1,580.45** | 9,284.71 | 11,241.89 |

📉 COALA 仅消耗约 **DPO 的 17.6% TFLOPs**，能效提升超过 **5.5 倍**。

---

### **与基线方法的对比结果**
| 对比项 | 结果 |
|--------|------|
| **vs DPO** | 性能相当甚至更优，但 VRAM 占用更低、无需参考模型、训练更稳定 |
| **vs ORPO** | 明显更快，更高胜率，且不牺牲初始 SFT 模型性能 |
| **vs SFT** | 远超纯监督微调，在偏好建模任务上有本质优势 |

### **消融实验结果**
- **CRONOS vs AdamW**（表 7）：
  - 在 cvxNN 分类任务中，CRONOS 在所有模型上均优于 AdamW，尤其在大模型（Llama-8B）上准确率高出近 6%。
  - 表明 ADMM 更适合高维凸优化，鲁棒性强。
- **SFT 初始化影响**：
  - COALA 即使从非 SFT 初始化也能取得良好效果，表明其对前置训练依赖较小。
  - 而 DPO 若缺乏 SFT 初始化则性能大幅下降。

---

## **4. 关键结论和发现**

### **主要发现**
1. **首次成功将凸优化应用于 LLM 偏好微调**，实现了理论可解释、实践高效的对齐路径。
2. **COALA 实现了稳定的奖励增长轨迹**，解决了传统方法中常见的奖励震荡问题。
3. **可在单张消费级 GPU 上完成 8B 级别模型的完整偏好训练**，极大降低了研究与部署门槛。
4. **真实人类评估证实 COALA 输出更受喜爱**，说明自动评测指标（如 AlpacaEval）虽有用但仍存在偏差。
5. **Alternating Population Strategy 可高效构建教育类偏好数据集**，减少对外部 LLM 的依赖。

### **方法的局限性**
- **表达能力受限**：由于冻结主干 LLM 参数，可能无法捕捉深层次语义迁移（如创造性写作风格转变）。
- **适用场景偏向指令遵循与事实准确性**：在需要深度结构调整的任务中表现可能不如全参数微调。
- **当前仅适用于偏好排序任务**：尚未扩展到强化学习或其他复杂目标函数。

### **未来工作方向**
- 将 COALA 扩展至 **GRPO、RRHF** 等更复杂的偏好学习算法。
- 探索 **多模态输入下的凸对齐机制**（文本+图像）。
- 开发更先进的 **inference-time guidance 策略**，如连续引导、多步前瞻搜索。
- 进一步提升 **cvxNN 的表达能力**，例如通过更深的凸结构或动态激活模式采样。
- 推动 **UltraMix 等多样化数据集上的泛化能力测试**。

---

> 🔗 **代码与数据开放**：
> - GitHub: [https://github.com/pilancilab/COALA](https://github.com/pilancilab/COALA)
> - Dataset: [https://huggingface.co/datasets/miria0/EduFeedback](https://huggingface.co/datasets/miria0/EduFeedback)

✅ **一句话总结**：  
**COALA 是首个基于凸优化的 LLM 偏好对齐算法，以极低资源消耗实现稳定、高效、高性能的单卡训练，为绿色 AI 与边缘部署提供了全新范式。**

</details>

---

### 6. [WeCon: An Efficient Weight-Conditioned Neural Solver for Multi-Objective Combinatorial Optimization Problems](https://arxiv.org/abs/2605.22876)

**Authors**: Xuan Wu, Jinbiao Chen, Yang Li, Lijie Wen, Chunguo Wu, Yuanshu Li, Yubin Xiao, Chunyan Miao, You Zhou, Di Wang  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.22876v1  

#### Abstract
Existing neural solvers for Multi-Objective Combinatorial Optimization Problems (MOCOPs) commonly adopt decomposition-based strategies that scalarize an MOCOP into multiple subproblems associated with distinct weight vectors. However, they either inject weights only once during decoding, limiting we...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：WeCon: An Efficient Weight-Conditioned Neural Solver for Multi-Objective Combinatorial Optimization Problems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Neural Combinatorial Optimization (NCO)** 模型在处理 **Multi-Objective Combinatorial Optimization Problems (MOCOPs)** 时存在以下关键缺陷：
- **Weight-conditioned context modeling 不足**：多数模型仅在 **Decoder** 中注入权重向量（如 PMOCO），导致编码阶段缺乏对权重的感知；或仅在 **Encoder** 中处理权重（如 WE-CA），导致解码时权重信号稀释（weight-signal dilution）。
- **Preference Optimization (PO) 效率低下**：现有训练策略依赖随机采样生成偏好对（preference pairs），常产生质量相近的弱信息对，降低训练有效性。

### 提出了什么新方法或新思路
本文提出了一种高效的 **Weight-Conditioned Neural Solver (WeCon)**，其核心创新包括：

#### I. 编码器设计：Gated Residual Fusion (GRF)
- 在每个 Encoder 层中引入三个注意力模块（MHSA + 双向 MHA）与 **GRF 模块**，实现 instance 特征与 weight 向量之间的深度交互。
- GRF 采用门控机制自适应融合 weight 信息，生成更具区分性的 **weight-conditioned context**。

#### II. 解码器设计：Residual Fusion (RF) 模块
- 在 Decoder 中插入 **plug-and-play 的 RF 模块**，在每一步解码中显式注入 weight 信号，缓解 weight-signal dilution。
- RF 结构简单高效，可无缝集成到不同架构中（如 WeCon-CCO 即为 RF + MoE 架构）。

#### III. 高效偏好优化：Efficient Preference Optimization (EPO)
- 改进传统 PO 策略，采用 **guided sampling**：在决策步中从 top-k 最优候选节点中采样部分解，其余仍随机采样。
- 显著提升采样解的质量，构造出差距更大的偏好对，增强训练信号。

### 相比现有方法的优势
| 维度 | WeCon 优势 |
|------|-----------|
| **性能** | 达到与 SOTA 模型 POCCO-W 相当甚至更优的 HyperVolume (HV) 表现 |
| **效率** | 推理时间减少约 **40%**，远低于依赖 MoE 路由机制的 POCCO-W |
| **通用性** | RF 模块即插即用，适用于多种 Decoder 架构 |
| **训练效果** | EPO 显著提升训练效率与最终性能 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖四种典型的 MOCOP 变体，在多个规模和分布下进行验证：
- **Bi-TSP**：双目标旅行商问题（n = 20, 50, 100, 150, 200, 500, 1000）
- **Tri-TSP**：三目标旅行商问题（n = 20, 50, 100）
- **Bi-CVRP**：双目标带容量车辆路径问题（n = 20, 50, 100）
- **Bi-KP**：双目标背包问题（n = 50, 100, 200, 500, 1000）
- **Real-world instances**：来自 TSPLIB 的 KroAB100/150/200 实例，用于测试跨分布泛化能力

### 实验设置和评估指标
| 设置项 | 描述 |
|-------|------|
| **训练配置** | Adam 优化器，学习率 3e-5，训练 200 轮，batch size = 64，实例通过均匀分布生成 |
| **权重生成** | 使用 Weighted-Sum (WS) 分解法，N=101 (K=2), N=105 (K=3) 个均匀分布的 weight vectors |
| **评估指标** | - **HyperVolume (HV)**：衡量解集收敛性与多样性，越高越好<br>- **Gap**：相对于最优 HV 的相对差距<br>- **Runtime**：单个测试集的总推理时间 |
| **硬件环境** | Intel Xeon Gold 6348 CPU + NVIDIA A800 GPU (80GB) |

### 基线方法对比
共比较 **14 种方法**，分为两类：
- **传统启发式算法**：WS-LKH, NSGA-II, MOEA/D, MOGLS, PPLS/D-C, WS-DP
- **神经求解器**：
  - 多模型：DRL-MOA, MDRL, EMNH
  - 单模型：PMOCO, CNH, WE-CA, PA-MoE-W, POCCO-W
- 所有神经模型均在相同条件下复现或直接引用原论文结果

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | Bi-TSP200 (HV↑) | Bi-TSP200 (Time↓) | Tri-TSP100 (HV↑) | Bi-CVRP100 (HV↑) |
|------|----------------|------------------|------------------|------------------|
| POCCO-W-Aug | 0.7399 | 2.1h | 0.5048 | 0.4089 |
| **WeCon-Aug (ours)** | **0.7402** | **1.2h** | **0.5049** | **0.4088** |
| **WeCon-CCO-Aug (ours)** | **0.7405** | 1.8h | 0.5046 | 0.4090 |

> ✅ WeCon-Aug 在保持 HV 与 POCCO-W-Aug 相当的同时，**推理时间减少约 43%**

### 与基线方法的对比结果
- **性能方面**：
  - WeCon-Aug 在大多数任务上达到或超过 POCCO-W-Aug 的 HV 表现。
  - 在无数据增强情况下，WeCon 已显著优于 POCCO-W（例如 Bi-TSP150 上 Gap 减少 0.08%）。
- **效率方面**：
  - WeCon 推理速度比 POCCO-W 快 **~40%**，主要得益于避免了 MoE 的路由开销。
  - 在大规模实例（如 Bi-TSP1000）上，WeCon-CCO 实现最佳 HV（0.7735），且运行时间可控。

### 消融实验结果（Ablation Studies）
#### 模块有效性验证（Table 10）
| 消融设置 | Bi-TSP100 (HV) | Bi-TSP150 (HV) |
|----------|---------------|----------------|
| 完整 WeCon | **0.7077** | **0.7063** |
| w/o Encoder（替换为 WE-CA） | 0.7073 | 0.7058 |
| w/o GRF（移除门控融合） | 0.7075 | 0.7060 |
| w/o RF（移除残差融合） | 0.7070 | 0.7051 |
| w/PO（换回原始 PO） | 0.7075 | 0.7060 |
| w/BOPO | 0.7058 | 0.7030 |

> 🔍 所有组件移除均导致性能下降，证明 GRF、RF 和 EPO 的必要性。

#### 超参数敏感性分析（Table 11）
- 参数 `k`（top-k 采样数）和 `c`（引导比例）在合理范围内对性能影响极小。
- 设定 `k=5`, `c=8` 时表现最优或接近最优，说明 EPO 策略鲁棒性强。

#### 分解技术对比（Table 12）
- 使用 **Weighted-Sum (WS)** 比 Tchebycheff (TCH) 更稳定，HV 更高。
- 与先前研究一致，支持选择 WS 作为默认分解方式。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **同时利用 Encoder 和 Decoder 中的 weight conditioning 是关键**：WeCon 首次实现在两个阶段都有效建模 weight-conditioned context，解决了 weight-signal dilution 问题。
2. ✅ **轻量级模块也能实现 SOTA 性能**：相比复杂的 MoE 架构（如 POCCO-W），WeCon 通过精心设计的 GRF 和 RF 模块，在更低延迟下达到相当甚至更好的性能。
3. ✅ **训练策略至关重要**：EPO 通过 guided sampling 显著提升了偏好对的信息量，从而加快收敛并提高最终性能。
4. ✅ **WeCon-CCO 实现最强性能**：将 RF 模块嵌入 MoE 架构（WeCon-CCO），获得所有方法中的最高 HV，适合对质量要求极高而对延迟容忍的应用场景。

### 方法的局限性
- **模型参数量较大**：WeCon (~5.4M) 参数多于 POCCO-W (~2.0M)，尽管实际内存占用和推理效率更高。
- **跨尺度泛化仍有挑战**：当前模型在多尺度联合训练，可能牺牲特定规模下的极致性能。
- **未探索其他偏好建模范式**：如 ranking-based 或 listwise learning 尚未尝试。

### 未来工作方向
- 引入 **Curriculum Learning** 渐进式训练，以进一步提升跨尺度泛化能力。
- 探索更高效的 weight embedding 编码方式，降低参数冗余。
- 将 WeCon 框架扩展至更多现实世界 MOCOP 场景，如供应链调度、能源管理等。
- 研究动态 weight 输入下的在线适应机制。

---

> 📌 **总结一句话**：  
> WeCon 是首个在 **不增加推理延迟的前提下**，通过 **双向 weight conditioning** 和 **高效偏好优化** 实现 SOTA 性能的 MOCOP 神经求解器，兼具高性能与高实用性。

</details>

---

### 7. [ImProver 2: Iteratively Self-Improving LMs for Neurosymbolic Proof Optimization](https://arxiv.org/abs/2605.22885)

**Authors**: Riyaz Ahuja, Tate Rowney, Jeremy Avigad, Sean Welleck  
**Category**: cs.AI  
**Published**: 2026-05-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.22885v1  

#### Abstract
Formal mathematics libraries are rapidly expanding, creating a growing need to refactor verified proofs for maintainability and to improve training data quality for neural provers. However, scalable proof optimization is hindered by heterogeneous and heuristically specified objectives, scarce data, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ImProver 2: Iteratively Self-Improving LMs for Neurosymbolic Proof Optimization

---

## 1. 主要贡献和创新点

### 解决的问题
该论文针对**形式化数学证明库**（如 Lean 的 Mathlib）快速增长所带来的**证明质量下降**和**可维护性挑战**。尽管当前神经定理证明器（neural provers）能生成正确的证明，但其风格不一致、冗余步骤多、依赖复杂，影响了代码库的长期可用性和作为训练数据的质量。

具体而言，论文聚焦于**自动化证明优化**（automated proof optimization）任务：给定一个已验证的证明，生成一个语义等价但更优的新证明，优化目标可以是长度、模块化程度或依赖项数量等。

### 提出的新方法与创新思路
论文提出了 **ImProver 2**，一个用于 Lean 4 中神经符号化（neurosymbolic）证明优化的自迭代改进框架，其核心创新包括：

- **Iterative Preference Optimization with Replay Buffer**  
  扩展了 IRPO（Iterative Reasoning Preference Optimization）算法，引入了一个**回放缓冲区**（replay buffer），在每轮训练中混合旧数据与新生成的数据，防止模型崩溃（model collapse），实现单调提升。

- **Neurosymbolic Augmentation（神经符号化增强）**  
  在生成过程中向语言模型提供来自 Lean 证明环境的丰富结构化信息，包括：
  - **Context**：相关引理和定义的上下文切片
  - **Chain-of-States (CoS)**：每个战术执行后的目标状态追踪
  - **Auto-informalization**：将形式化证明自动翻译为自然语言草图，提供高层语义抽象

- **多维度结构化优化指标**  
  定义并实现了三个实用且可自动评估的结构化指标：
  - **Length**：最小化战术（tactics）数量
  - **Dependencies**：最小化显式命名的外部引理数量
  - **Modularity**：最大化“有效生成的目标”（effective spawned goals）数量，衡量证明的模块化分解能力

### 相比现有方法的优势
- **无需人工标注数据**：通过自我迭代生成偏好对进行训练，摆脱对昂贵人类标注的依赖。
- **小模型高效专用化**：仅用 **7B 参数的 base model**（DeepSeek-R1-Distill-Qwen-7B），经过训练后性能超越更大规模的基础模型甚至部分前沿闭源模型。
- **通用性强**：支持多种异构优化目标，适用于研究级复杂数学定理。
- **开源开放**：作者公开了代码与数据。

---

## 2. 核心实验方法和设置

### 数据集
- **训练/验证集**：从多个形式化数学项目中提取，涵盖多个领域，包括：
  - `Mathlib`（通用数学）
  - `HepLean`（高能物理）
  - `ConNF`, `Seymour`, `FLT`, `Carleson`, `Foundation` 等
- **测试集**：保留 `miniCTX-v2` 中的所有定理作为测试集，确保无数据泄露。
- **预处理**：排除与测试文件同源的训练样本，并对 Mathlib 部分进行均匀采样以控制规模。

### 实验设置
- **Base Model**：`DeepSeek-R1-Distill-Qwen-7B`
- **训练流程**：
  1. 使用当前模型生成每个问题的 `n` 个候选证明
  2. 结合回放缓冲区中的历史数据
  3. 过滤出“胜者”（编译成功且评分更高）与“败者”
  4. 构建 preference pairs，使用 IRPO 损失函数进行训练
  5. 重复直至收敛或耗尽计算预算
- **推理方式**：`best@16` 采样（即生成16个样本取最优）

### 评估指标
- **主指标**：平均改进分数（mean improvement score）  
  $\Delta \mu = \frac{1}{|D|}\sum_{(c,x,y_0)\in D} \left[u(c,x,y) - u(c,x,y_0)\right]$，其中 $u \in \{\mu_{len}, \mu_{dep}, \mu_{mod}\}$
- **辅助指标**：
  - 编译准确率 $A(D)$：生成证明中能通过 Lean 编译的比例
  - 改进准确率 $A_I(D)$：既能编译又能带来正向改进的比例

### 基线方法对比
- **Intra-family Baselines**：
  - `DS-R1 7B`, `DS-R1 14B`, `DS-R1 671B`
- **Frontier Models**：
  - GPT-4o, GPT-5-nano, GPT-5-mini, GPT-5-chat, GPT-5-high
  - 开源稀疏 MoE 模型 `GPT-oss-120B`
- **Prior System**：
  - 原始 `ImProver` 系统（基于 GPT-4o 的多步提示系统）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（MiniCTX-v2 上 best@16 平均改进分）

| Model | Length ↑ | Modularity ↑ | Dependencies ↓ |
|-------|----------|--------------|----------------|
| **ImProver 2 (ours)** | **0.330** | **0.143** | **0.206** |
| DS-R1 7B (base) | 0.118 | 0.003 | 0.050 |
| DS-R1 14B | 0.140 | 0.037 | 0.093 |
| DS-R1 671B | 0.308 | 0.055 | 0.153 |
| GPT-5-mini | 0.330 | 0.109 | 0.203 |
| GPT-5-high | 0.660 | 0.120 | 0.208 |
| ImProver (prior) | 0.355 | 0.088 | 0.047 |

> ✅ **关键观察**：
> - ImProver 2 在 **Modularity** 上显著领先所有未增强（unscaffolded）系统。
> - 在 **Dependencies** 上几乎追平最强的 GPT-5-high（0.206 vs 0.208）。
> - 在 **Length** 上匹配 GPT-5-mini，略低于 GPT-5-high 和原始 ImProver。
> - **仅用 7B 模型，全面超越 671B 规模的同族模型**，说明任务特定训练优于通用缩放。

### 消融实验结果（Ablation Study）

#### （1）Neurosymbolic Scaffold 的影响（Table 3）
加入神经符号化增强后，几乎所有模型在所有指标上均有显著提升：

| Model | Length (+Scaffold) | Modularity (+Scaffold) | Dependencies (+Scaffold) |
|-------|--------------------|------------------------|----------------------------|
| DS-R1 7B | 0.118 → **0.236** | 0.003 → **0.007** | 0.050 → **0.056** |
| GPT-5-mini | 0.330 → **0.632** | 0.109 → **0.123** | 0.203 → **0.267** |
| GPT-5-high | 0.660 → **0.875** | 0.120 → **0.183** | 0.208 → **0.315** |

> 🔍 **结论**：神经符号化增强极大提升了小模型和大模型的搜索效率，使其更容易发现高质量重构。

#### （2）迭代训练效果（Table 2）
性能随 IRPO 轮次上升，在第 2–3 轮达到峰值：

| Iteration | Length | Modularity | Dependencies |
|---------|--------|------------|--------------|
| Base | 0.118 | 0.003 | 0.050 |
| +Scaffold | 0.236 | 0.007 | 0.056 |
| 1 | 0.265 | 0.062 | 0.137 |
| 2 | 0.318 | 0.134 | **0.206** |
| 3 | **0.330** | **0.143** | 0.165 |

> 📈 表明模型确实在从自身生成的优质样本中学习到有效的重构策略。

#### （3）Accuracy Trade-off 分析（Table 4 & 5）
- 优化过程通常会**降低编译成功率**但**提高改进成功率**。
- 例如在 Dependency 任务中，基础模型编译率高但改进率低；训练后改进率上升，但编译率下降，表明模型愿意尝试更具风险性的结构性修改。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **小模型可通过迭代自优化实现高性能**：7B 模型经 ImProver 2 训练后，在多个结构化指标上媲美甚至超越百B级模型。
2. ✅ **神经符号化增强至关重要**：暴露目标状态、上下文和非正式化描述显著提升了所有模型的表现，尤其对小模型帮助巨大。
3. ✅ **结构化指标具有实际意义**：提出的 `modularity` 和 `dependencies` 指标能够引导模型生成更具可读性、更少耦合的证明。
4. ✅ **IRPO + Replay Buffer 可稳定训练**：避免了自我训练中的分布坍塌问题，支持多轮单调提升。

### 方法的局限性
- **指标主观性**：所提指标（如 dependency 数量）虽可自动计算，但未必完全反映人类维护者的偏好。
- **缺乏人类评估**：未进行用户研究验证优化后的证明是否真正“更好”。
- **单步重写限制**：未构建完整的 agent 系统，无法进行多轮修复或探索。
- **对复杂 AI 生成证明优化有限**：在 AlphaProof 生成的 IMO 问题上，优化幅度较小（见 Figure 13, 19），表明极端复杂的机器证明仍难有效重构。

### 未来工作方向
- 将 **maintainer preference** 显式建模，可能借助 LLM-based reward modeling。
- 探索 **training dataset optimization 对下游 prover 性能的影响**。
- 构建 **agentic repair loop**，结合优化与错误修复能力。
- 扩展至其他形式化系统（如 Coq、Isabelle）和其他编程语言重构任务。

--- 

> 💡 **总体评价**：  
> ImProver 2 成功展示了如何通过**结构感知的训练框架**和**丰富的环境反馈**，使小型语言模型在高度专业化、结构敏感的任务上达到前沿水平。它不仅推动了形式化数学的可持续发展，也为“小模型专用化”提供了有力范例。

</details>

---

### 8. [Parallel Context Compaction for Long-Horizon LLM Agent Serving](https://arxiv.org/abs/2605.23296)

**Authors**: Musa Cim, Burak Topcu, Chita Das, Mahmut Taylan Kandemir  
**Category**: cs.AI  
**Published**: 2026-05-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.23296v1  

#### Abstract
Long-horizon LLM agents accumulate growing conversation histories that eventually exceed the model's context window. Context compaction via LLM-based summarization keeps the conversation bounded, but summarization is inherently lossy and the blocking call stalls agent inference for tens of seconds. ...

---

### 9. [One Policy, Infinite NPCs: Persona-Traceable Shared RL Policies for Scalable Game Agents](https://arxiv.org/abs/2605.23652)

**Authors**: Yoosung Hong  
**Category**: cs.AI  
**Published**: 2026-05-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.23652v1  

#### Abstract
On a 300-persona life-simulation benchmark, pcsp achieves compositional zero-shot persona identification up to 17x above chance, Spearman rho approx 0.73 semantic-behavioral alignment, and 22x faster inference than an LLM-as-policy baseline. Life simulation games require hundreds to thousands of non...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# One Policy, Infinite NPCs: Persona-Traceable Shared RL Policies for Scalable Game Agents 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代生活模拟游戏（如 *The Sims*, *Animal Crossing*）需要成百上千个具有**独特且一致人格**（persona）的非玩家角色（NPCs）。然而，当前主流方法在以下四个关键维度上无法同时满足需求：
- **Persona consistency**（人格一致性）
- **Natural-language controllability**（自然语言可控性）
- **Zero-shot generalization**（零样本泛化）
- **Real-time inference**（实时推理）

现有方法存在明显短板：
- **Behavior trees**：需手动编写，成本随角色数线性增长。
- **Per-NPC RL**：每个角色独立训练策略，计算开销巨大。
- **LLM-as-policy**：虽支持自然语言控制，但推理延迟高（>40ms），不适用于实时游戏。
- **Unsupervised skill discovery**（如 DIAYN）：缺乏语义可解释性，设计师无法指定具体人格。

### 提出的新方法：PCSP
作者提出 **PCSP (Persona-Conditioned Shared Policy)**，一种基于强化学习（RL）的统一策略框架，其核心思想是：

> **“一次编码，终生条件”**（once-per-NPC persona encoding, lifelong conditioning）

#### 方法架构
- **Persona Encoding**：使用冻结的 LLM（Qwen3-0.6B-Embedding）将自由形式的人格描述文本映射为固定向量 $ e_p \in \mathbb{R}^{1024} $。
- **Low-Rank Projection**：通过一个低秩（rank-16）投影层将 $ e_p $ 映射到 $ \mathbb{R}^{64} $，以增强语义对齐并减小模型规模。
- **Shared Policy**：一个轻量级 MLP 策略网络 $ \pi(a|s, e_p) $，通过 **FiLM** 或 **concat** 方式注入人格信号。
- **Co-Training Objective**：
  - **PPO**：主任务奖励优化。
  - **InfoNCE Consistency Loss**：确保轨迹能被追溯回原始人格（轨迹→人格对齐）。
  - **KL Diversity Regularization**：鼓励不同人格产生多样化行为。

### 相比现有方法的优势
| 方法 | Persona Consistency | NL Control | Zero-shot Gen | Real-time (<5ms) |
|------|---------------------|----------|---------------|------------------|
| Behavior trees | △ | ✗ | ✗ | ✓ |
| Per-NPC RL | ✓ | ✗ | ✗ | ✗ |
| LLM-as-policy | ✓ | ✓ | ✓ | ✗ |
| **PCSP (ours)** | **✓** | **✓** | **✓** | **✓** |

PCSP 是首个在所有四个维度上均达标的方案，实现了**可扩展、实时、可追踪人格的 NPC 控制**。

---

## 2. 核心实验方法和设置

### 数据集
- **PCSP-D**（原 Mini-Inzoi）：自研诊断环境，用于机制分析。
  - 6×6 网格，4 名智能体。
  - 300 个人格描述（15 种 Big Five 人格 × 20 种职业），划分为 240 训练 / 60 零样本测试。
- **Melting Pot 2.4.0**：外部多智能体 RL 基准，验证跨域泛化能力。
  - 包含三种社会困境：`commons_harvest_open`, `clean_up`, `prisoners_dilemma_in_the_matrix_repeated`。
  - RGB 观测（88×88×3），更复杂的观察几何与社交结构。
- **Unreal Engine 5 (UE5)**：真实引擎部署，验证工程可行性。
  - 部署 64 名智能体于自定义地图，测试实时性与系统稳定性。

### 实验设置
- **三层验证栈**（Three-Layer Validation Stack）：
  1. **Layer 1 (Mechanism)**：PCSP-D 上进行受控消融实验，验证 InfoNCE 的因果作用。
  2. **Layer 2 (Generalization)**：Melting Pot 上测试跨子环境泛化。
  3. **Layer 3 (Deployment)**：UE5 中测试实时推理、长时程行为持久性与系统鲁棒性。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Zero-shot Accuracy (ZS Acc)** | 给定轨迹，k-NN 分类器识别出自哪个（未见）人格生成，衡量零样本泛化能力。 |
| **Semantic-behavioral Alignment ($\rho$)** | Spearman $\rho$ 在“人格距离”与“行为 KL 散度”之间的相关性，衡量语义对齐程度。 |
| **Pairwise Action-KL** | 不同人格下策略分布的平均 KL 散度，衡量行为多样性。 |
| **Inference Latency** | 单步推理耗时（GPU ms/step），要求 < 16–33ms 以满足实时帧预算。 |
| **Fail Rate** | UE5 中路径失败率，反映系统稳定性。 |

### 基线方法对比
- **B1 No-Persona PPO**：无任何人格输入。
- **B3 SBERT-conditioned**：使用 Sentence-BERT 编码人格。
- **B4 DIAYN**：基于隐变量技能发现。
- **B5 LLM-as-policy**：每步调用 Qwen3-1.7B 进行决策。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | PCSP (full) | 最佳基线 | 提升倍数 |
|------|-----------|--------|---------|
| **ZS Acc (PCSP-D v3)** | 17.0% | — | **10× above chance** (1.7%) |
| **$\rho$ (Semantic Alignment)** | ~0.73 | — | 高度对齐 |
| **Inference Latency** | 0.183–0.202 ms/step | LLM-as-policy: 43.7 ms | **22× 更快** |
| **UE5 零样本泛化失败率** | 0.04% | — | 极低 |
| **InfoNCE Ablation 失败率** | ~1.7% (≈ chance) | — | 完全崩溃 |

### 与基线方法对比
- **优于所有基线**：在 ZS Acc 和 $\rho$ 上显著优于 SBERT、DIAYN 等。
- **远超 LLM-as-policy**：推理速度提升 22 倍，适合实时应用。
- **唯一满足四维要求的方法**：见 Table I。

### 消融实验结果
| 消融项 | ZS Acc ↓ | Reward ↑ | Pairwise KL ↓ | 结论 |
|--------|--------|--------|-------------|------|
| **Full PCSP** | 17.0% | 104.1 | 2.06 | 基准 |
| **no_consist** | **1.7%** | **118.4** | 1.07 | ❌ **完全崩溃至随机水平** |
| **no_diverse** | 16.0% | 122.1 | 0.39 | 行为趋同，但 ZS 仍有效 |
| **concat vs. FiLM** | concat 更好（新职业）<br>FiLM 更好（新人格） | — | — | 条件机制次要，取决于 OOD 类型 |

> **核心发现**：**InfoNCE 一致性损失是负载承载组件**（load-bearing component）。移除后，即使奖励更高，人格也无法从轨迹中恢复。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **共享 RL 策略可实现大规模人格化 NPC**：PCSP 成功实现了“一个策略，无限 NPC”的愿景。
2. 🔍 **InfoNCE 是人格可追溯性的关键**：轨迹一致性目标（InfoNCE）是保证“行为可归因于人格”的核心机制，移除后零样本识别降至随机水平。
3. 🔄 **方法具有强泛化能力**：
   - 跨越不同环境（PCSP-D → Melting Pot）保持人格区分。
   - 支持跨语言控制（英文人格描述 → 韩文训练策略）。
4. ⚙️ **工程部署可行**：
   - UE5 中 64 名 NPC 并发运行，失败率仅 1.7%。
   - 推理延迟 < 0.2ms，满足实时要求。
   - 长时程（30分钟）行为稳定，人格特征持续存在。

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **Vocabulary-expansion held-out recovery** | 对于嵌入空间内但训练中未出现的新词（如 "fast_mover", "spinner"），Top-1 检索始终为 0，仍是开放问题。 |
| **Action Space 限制** | 当前动作本体无法表达某些风格化行为（如 “protect”, “nurture”），导致部分设计意图无法体现。 |
| **引擎-研究差距**（engine-research gap） | UE5 中由于容量争用（contention），实际行为分散度（$p_{intra}$）低于研究环境（0.73 → 0.25），存在执行压缩。 |
| **合成人格数据** | 训练人格由模板生成，尚未验证对真实复杂人物设定的鲁棒性。 |
| **有限人类评估** | 当前人类实验仅收集聚合选择比例，缺乏信心、反应时间等细粒度分析。 |

### 未来工作方向
1. **动态人格与记忆机制**：
   - 引入事件驱动的记忆更新模块，支持人格演化（如经历冲突后变得神经质）。
   - 结合对话数据（如 [38]）实现人格成长。
2. **解决 Vocabulary-expansion 问题**：
   - 设计带边界的 InfoNCE（controlled-margin loss）。
   - 多目标 InfoNCE + 聚类锚定（per-cluster pinning）。
3. **构建跨环境不变的人格投影**：
   - 当前跨子环境迁移不对称（CU → CH 有效，反之无效）。
   - 目标是建立 substrate-invariant persona manifold。
4. **更丰富的人类评估**：
   - 使用富轨迹渲染（时间、空间、社交上下文）进行 A/B 测试。
   - 收集参与者信心、反应时间、重评一致性等指标。
5. **扩展至更大、程序化世界**：
   - 在 procedurally-generated 地图上测试泛化能力。
   - 集成连续物理与异步事件。

---

> **总结**：PCSP 提供了一种**高效、可控、可扩展**的 NPC 人格化解决方案，其核心洞见是：**人格一致性不仅依赖于输入条件，更依赖于输出轨迹的可追溯性**。InfoNCE 损失正是实现这一目标的关键机制。该工作为下一代生活模拟游戏 AI 奠定了坚实基础。

</details>

---

### 10. [Convex Low-resource Accent-Robust Language Detection in Speech Recognition](https://arxiv.org/abs/2605.23235)

**Authors**: Miria Feng, William Tan, Mert Pilanci  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.23235v1  

#### Abstract
Globalization and multiculturalism continue to produce increasingly diverse speech varieties. Yet current spoken dialogue systems frequently fail on under-represented dialects and accents, often misidentifying the input language and causing cascading failures in downstream dialogue tasks. Addressing...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Convex Low-resource Accent-Robust Language Detection in Speech Recognition**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 **ASR**（Automatic Speech Recognition）系统在处理**低资源语言**和**非主流口音**（如新加坡英语 Singlish、马来西亚英语、方言汉语等）时表现不佳，常错误识别输入语言，导致后续对话任务出现级联错误。这主要是因为：
- 大多数语音数据集对区域性口音标注不足；
- 高维语音数据下传统微调方法计算成本高且易过拟合。

### **提出的新方法：Convex Language Detection (CLD)**
作者提出了 **CLD**（Convex Language Detection），一种基于**凸优化**的轻量级语言检测框架，集成于 ASR 流水线中，用于提升多口音、低资源场景下的语言识别鲁棒性。

#### **核心创新点：**
- **理论驱动的凸优化重构**：将传统的两层 ReLU 网络重新表述为一个**等效的凸程序**（cvxNN），从而保证全局最优解，避免非凸训练中的局部极小值和超参数敏感问题。
- **高效实现**：采用 **JAX** 实现，并利用 **ADMM**（Alternating Direction Method of Multipliers）进行多 GPU 并行求解，实现快速训练与推理。
- **可验证的鲁棒性**：通过推导 **logit-Lipschitz 常数** 和 **margin stability certificate**，提供对隐藏特征扰动的**认证稳定性**（certified robustness）。
- **样本效率高**：在极低样本条件下（每语言仅百个样本）仍能保持高性能。

### **相比现有方法的优势**
| 维度 | 传统方法（如 fine-tuning、NN） | CLD（本文方法） |
|------|-------------------------------|------------------|
| **训练稳定性** | 依赖学习率等超参数，易陷入局部最优 | 全局最优，无需调参（hyperparameter-free） |
| **计算效率** | 训练耗时长，需多次迭代和网格搜索 | 训练时间缩短至 7.7%，TFLOPs 减少 13 倍 |
| **鲁棒性** | 黑箱模型，缺乏理论保障 | 提供可计算的鲁棒性证书（margin radius） |
| **低资源表现** | 数据稀少时性能急剧下降 | 在 <100 样本下仍达 97–98% 准确率 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Common Voice v23**：作为基础多语言语音数据源。
- **National Speech Corpus (NCS)**：新加坡首个本地化英语语料库（含 Singlish）。
- **Lahaja Dataset**：包含来自印度 83 个地区的 12.5 小时多口音印地语语音。
- **自建多口音数据集**：
  - **Binary Classification**：英语 vs 中文，各选 5 种区域口音（共 10 类），样本规模从 100 到 10,000 不等。
  - **Multiclass Classification**：涵盖 **5 种语言**（English, Chinese, Indonesian, Malaysian, Hindi），共 **24 种子口音**，总计 16,000 条训练样本（约每语言 3,200，每口音 ~666）。

所有音频均经过标准化增强（time stretch, pitch shift, background noise 等）。

### **实验设置与评估指标**
- **评估任务**：
  - 语言检测准确率（Language Detection Accuracy）
  - 词错误率（**WER**）
  - 字符错误率（**CER**）
  - 训练时间与计算成本（TFLOPs）
- **划分方式**：80%-10%-10% 的 train/validation/test 分割。
- **延迟要求**：在线推理需满足 <500ms 延迟以适配自然对话节奏。

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **WSP-SFT** | 微调 Whisper-small 的完整模型 |
| **Vanilla-NN** | 轻量级神经网络（Linear → ReLU → Dropout → Linear） |
| **Linear SVM / Kernel SVM** | 支持向量机分类器（线性与 RBF 核） |
| **KNN** | k-最近邻分类器 |
| **Default Detector** | Whisper 自带的语言检测模块 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **低资源环境下卓越表现**
- 在每语言仅 **100 个训练样本**的情况下，CLD 在二分类任务中达到 **97.37% 准确率**（见 Table D.1），远超 Vanilla-NN（47.38%）。
- 在多分类任务中，CLD 达到 **97–98% 的语言检测准确率**（Table 3），显著优于其他方法。

#### ✅ **显著降低 WER 与 CER**
| 模型 | 方法 | Detection Acc (%) | WER | CER |
|------|------|--------------------|-----|-----|
| Whisper-Small | Default | 71.54 | 139.37 | 73.85 |
| Whisper-Small | CLD (ours) | **97.15** | **31.74** | **17.84** |
| MMS-1B | Default | 67.01 | 51.88 | 27.61 |
| MMS-1B | CLD (ours) | **97.02** | **45.27** | **21.58** |

> ⚡️ **CLD 使 Whisper-Small 的 WER 下降超过 100 点**，说明其有效防止了跨语言解码错误。

#### ✅ **训练效率碾压级优势**
| 方法 | 训练时间 (秒) | TFLOPs 成本 |
|------|----------------|-------------|
| WSP-SFT | 1,096.74 | 239,528 |
| Vanilla-NN | 840.30 | 183,521 |
| **CLD (ours)** | **64.45** | **14,075** |

> 💡 CLD 训练时间仅为 Vanilla-NN 的 **7.7%**，计算量减少 **13 倍以上**。

### **与基线方法的对比结果**
- **CLD 在所有评估指标上全面领先**，尤其在低资源和多口音复杂边界场景下优势明显。
- **SVM 表现尚可但泛化差**：在 Whisper 模型上表现良好，但在 MMS-1B 上严重退化。
- **KNN 完全失效**：高维编码空间中简单距离度量无法区分相近语言。
- **Vanilla-NN 存在强烈英语中心偏见**：将大量中文样本误判为英文（如 Table 2 显示其对 Mainland Chinese 仅 8.78% 准确率）。

### **消融实验结果**
- **样本大小影响分析**（Figure 2）：
  - Vanilla-NN 和 WSP-SFT 性能随样本增加而上升；
  - **CLD 在所有样本规模下保持稳定高精度**，证明其极强的样本效率。
- **口音粒度分析**（Table 2）：
  - 对极具挑战性的 **Min Dong Chinese**（闽东话），传统方法准确率不足 25%，而 CLD 达到 **88.73%**。
  - CLD 在所有口音上表现均衡，无明显偏见。

---

## **4. 关键结论和发现**

### **主要发现**
1. **CLD 极大缓解了 ASR 中的“英语中心主义”偏差**，在低资源口音上实现公平且鲁棒的识别。
2. **凸优化重构不仅理论上优雅，而且实践中高效**：CLD 实现了全局最优、无需调参、快速收敛。
3. **语言检测是提升端到端 ASR 性能的关键瓶颈环节**：正确初始化语言 token 可大幅减少跨语言解码错误。
4. **认证鲁棒性可行**：首次在语音语言检测任务中提供了基于 variation norm 的 margin stability certificate。

### **方法的局限性**
- 当前 CLD 是一个**独立的检测头**，未与 ASR 编码器联合训练，可能未充分挖掘表示协同潜力。
- 虽然支持多语言，但实验集中在 **5 种语言 + 24 口音**，尚未扩展至千语言级别（如 MMS 目标）。
- 对极端噪声或病理语音的鲁棒性未测试。

### **未来工作方向**
- **端到端可微分 CLD**：通过隐式微分（implicit differentiation）将 ADMM 或 KKT 条件嵌入训练流程，实现 encoder 与 detection head 联合优化。
- **扩展至 Multimodal Agent**：将 CLD 应用于视觉-语音-文本多模态智能体中，提升跨模态语言理解。
- **更大规模部署**：结合 Cloud TPU 和大规模 MMS 模型，推动全球 1000+ 语言的普惠访问。

---

> 🔗 **开源信息**：  
> - PyPI 包：[https://pypi.org/project/jaxcld/](https://pypi.org/project/jaxcld/)  
> - GitHub 代码：[https://github.com/pilancilab/CLD](https://github.com/pilancilab/CLD)

> 🎯 **Impact Statement**：  
> 本研究致力于**民主化语音交互技术**，让不同口音、语言背景的用户都能被“听见”，推动 AI 的包容性与公平性发展。

</details>

---

### 11. [FuRA: Full-Rank Parameter-Efficient Fine-Tuning with Spectral Preconditioning](https://arxiv.org/abs/2605.22869)

**Authors**: Yequan Zhao, Ruijie Zhang, Liyan Tan, Niall Moran, Tong Qin, Zheng Zhang  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.22869v1  

#### Abstract
Both full fine-tuning (Full FT) and parameter-efficient fine-tuning methods such as LoRA introduce weight updates without accounting for the spectral structure established during pretraining. As a result, noisy gradients from limited fine-tuning data can perturb robust pretrained features. We identi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# FuRA: Full-Rank Parameter-Efficient Fine-Tuning with Spectral Preconditioning —— 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

- **现有方法忽略预训练权重的谱结构（spectral structure）**：
  - **Full FT** 和 **LoRA** 等主流方法在微调时直接添加参数更新（如 $ \Delta W $），而未考虑预训练模型中通过 SVD 学到的奇异值和奇异向量所构成的“特征空间”。
  - 这导致微调过程中，来自小规模、窄分布任务数据的**噪声梯度**可能破坏预训练阶段学到的鲁棒特征，引发**灾难性遗忘**（catastrophic forgetting）。

- **低秩约束限制表达能力**：
  - LoRA 等方法强制 $ \Delta W $ 为低秩矩阵，虽然节省参数，但也限制了模型适应复杂任务的能力，尤其在 **Reinforcement Learning with Verifiable Rewards (RLVR)** 等需要高秩更新的任务上表现不佳。

---

### **提出了什么新方法或新思路**

提出 **FURA (Full-Rank Adaptation)**，一种结合**全秩更新能力**与**谱预条件化**（spectral preconditioning）的高效微调框架。

#### **核心思想：谱预条件化（Spectral Preconditioning）**

- 将预训练权重 $ W $ 分解为其 SVD 形式 $ W = U\Sigma V^T $。
- 在微调时**冻结左奇异基 $ U $**，只训练 $ \Sigma $ 和 $ V $。
- 这样所有更新都被**限制在预训练的列空间 $ \text{col}(U) $ 内**，避免引入正交于原始特征流形的新方向。

#### **实现方式：块张量链分解（Block Tensor-Train, BTT）**

- 对每个线性层 $ W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} $，将其划分为 $ n $ 个块 $ W_k $。
- 对每个块进行全秩 SVD：$ W_k = L_k S_k R_k $，其中 $ L_k = U_k \in \mathbb{R}^{d_{\text{out}} \times b} $。
- **冻结大核 $ L $**，仅训练小核 $ R $ 和奇异值 $ S $。
- 最终微调权重为：  
  $$
  W' = \sum_{k=1}^n L_k \text{diag}(S_k) R_k
  $$

---

### **相比现有方法的优势**

| 特性 | Full FT | LoRA | FURA (ours) |
|------|--------|------|------------|
| 参数效率 | ❌ 高开销 | ✅ 极高 | ✅ 与 LoRA 相当 (~1.5%) |
| 更新秩容量 | ✅ 全秩 | ❌ 低秩 | ✅ **全秩** |
| 谱结构保留 | ❌ 忽略 | ❌ 忽略 | ✅ **显式保留** |
| 推理无额外开销 | ✅ | ✅ | ✅ 合并回稠密矩阵 |
| 性能上限 | 基准 | 通常低于 Full FT | ✅ **超越 Full FT** |

> ✅ **FURA 实现了三者统一**：**全秩更新能力 + 谱预条件化 + LoRA 级别的参数/计算效率**。

---

## 2. 核心实验方法和设置

### **使用的数据集**

| 任务类型 | 数据集 | 模型 |
|---------|--------|------|
| **常识推理 SFT** | Commonsense-170K | LLaMA-2-7B, LLaMA-3-8B |
| **数学强化学习 RLVR** | Math-10K, MetaMathQA-100K | Qwen3-1.7B, Qwen2.5-7B |
| **视觉指令调优 VLM** | llava_v1_5_mix665k | LLaVA-1.5-7B |
| **4-bit 量化微调** | Commonsense-170K, MetaMathQA-100K | LLaMA-3-8B, LLaMA-3-70B |

---

### **实验设置与评估指标**

- **训练设置**：
  - 使用 **bf16** 混合精度，单张 H100 GPU。
  - AdamW 优化器，梯度检查点开启。
  - 可变 batch size（如 16–128），序列长度 2048 或 512。
  - FURA 默认配置：`m=1`, `S` 单独可训练。

- **评估指标**：
  - **SFT**：8项常识推理任务平均准确率（BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e/c, OBQA）。
  - **RLVR**：MATH-500, AMC23, AIME-24/25 上的准确率（avg@8 或 greedy@1）。
  - **VLM**：7项视觉问答任务平均分（VQAv2, GQA, VisWiz, SQA, TextVQA, POPE, MMBench）。
  - **系统开销**：每步时间（s/step）、峰值 GPU 内存（GB）、可训练参数比例（%）。

---

### **基线方法对比**

| 类别 | 方法 |
|------|------|
| **全微调** | Full FT |
| **低秩适配器** | LoRA, DoRA |
| **SVD 初始化适配器** | PiSSA (top-r), MiLoRA (bottom-r) |
| **全秩/高秩适配器** | RandLoRA, LIFT |
| **量化微调** | QLoRA, QDoRA |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **常识推理 SFT（LLaMA-3-8B）**

| 方法 | 可训练参数 (%) | 平均准确率 (%) |
|------|----------------|----------------|
| Full FT | 100.00 | 86.64 |
| LoRA (r=64) | 1.39 | 83.46 |
| DoRA (r=64) | 1.42 | 85.04 |
| **FURA (ours)** | **1.46** | **88.01** |

> 🔺 **FURA 超越 Full FT +1.37**, 超越 DoRA +2.87，且仅用 **1.46%** 参数。

#### ✅ **数学强化学习 RLVR（Qwen3-1.7B）**

| 方法 | MATH-500 | AMC23 | AIME-24 |
|------|----------|-------|--------|
| Full FT | 61.5 | 46.7 | 13.2 |
| LoRA | 59.6 | 48.3 | 9.2 |
| RandLoRA | 62.2 | 55.0 | 13.8 |
| **FURA (ours)** | **62.5** | **55.8** | **13.8** |

> 🔺 FURA 在多个指标上**优于 Full FT**，尤其在 AMC23 上提升显著。

#### ✅ **视觉指令调优（LLaVA-1.5-7B）**

| 方法 | 可训练参数 (%) | 平均得分 |
|------|----------------|----------|
| Full FT | 100 | 66.5 |
| DoRA | 4.63 | 67.6 |
| **FURA (ours)** | **1.37** | **67.6** |

> 🔺 FURA **以 3.4× 更少的可训练参数**达到与 DoRA 相同性能。

#### ✅ **4-bit 量化微调（QFuRA）**

| 模型 | 方法 | 可训练参数 (%) | 准确率 |
|------|------|----------------|--------|
| LLaMA-3-8B | QLoRA | 1.39 | 83.89 |
| LLaMA-3-8B | QDoRA | 1.42 | 86.34 |
| LLaMA-3-8B | **QFuRA** | **1.46** | **87.30** |
| LLaMA-3-70B | QLoRA | 1.17 | 81.27 |
| LLaMA-3-70B | QFuRA | 1.45 | **83.78** |

> 🔺 QFuRA 在两个模型上均**显著超越 QLoRA 和 QDoRA**。

---

### **消融实验结果**

#### 📊 **设计选择消融（Table 8）**

| 配置 | SFT 平均 | MATH-500 |
|------|----------|----------|
| FURA Full (全训练) | 88.04 | 64.2 |
| **FURA (default: LSR)** | **87.91** | **57.5** |
| (LS)R (S 合入 L) | 87.12 | 55.0 |
| L(SR) (S 合入 R) | 84.09 | 47.5 |
| n=1 方向（输入空间冻结） | <87.0 | <53.0 |

> 🔍 结论：
> - **默认配置（m=1, S 单独可训练）最优**。
> - 冻结输出空间（m=1）优于冻结输入空间（n=1）。
> - 将 S 合入任一核都会显著降低性能。

#### 📊 **形状因子消融（Table 20）**

| 输入划分方式 | 平均准确率 |
|--------------|------------|
| 默认 (b ≈ √d_in) | 88.06 |
| 不平衡 (b=8) | 79.99 |
| 极端 (b=1) | 33.18 |

> 🔍 结论：**更均衡的块大小（balanced factorization）效果更好**。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **谱预条件化是关键缺失环节**：
   - 显式将更新限制在预训练列空间内，能**同时提升目标任务性能和源域泛化能力**，缓解灾难性遗忘。

2. ✅ **FURA 实现了“三赢”设计**：
   - **全秩更新能力**：不受预设秩限制，由优化器动态控制。
   - **谱预条件化**：通过冻结 $ L $ 实现列空间投影与奇异值缩放。
   - **LoRA 级效率**：训练参数 ~1.5%，步时和内存与 LoRA 相当。

3. ✅ **FURA 在多种任务上超越 Full FT**：
   - 包括 SFT、RLVR、VLM，且在 4-bit 量化下仍保持优势。

4. ✅ **QFuRA 是首个超越 QLoRA 的量化 PEFT 方法**：
   - 证明 FURA 框架对量化友好。

---

### **方法的局限性**

- **初始化成本较高**：需对每个线性层执行 Block-SVD，虽为一次性开销，但仍高于 LoRA。
- **缺乏跨块信息传递**：当前 BTT 设计中，各块独立处理，无法跨块共享信息（可通过引入 input-mixing 矩阵扩展）。
- **理论解释尚不完整**：为何谱预条件化能提升泛化，仍需更深的理论分析。

---

### **未来工作方向**

- 探索非 SVD 的分解初始化（如随机投影、傅里叶基）。
- 开发专用 kernel 加速 BTT 收缩操作。
- 设计更强的 QFuRA，利用 $ L $ 的正交性进行更激进量化。
- 扩展至 MoE 模型或其他模态。

---

> 💡 **总结**：  
> **FURA 揭示了“尊重预训练谱结构”是高效微调的关键**。它不仅是一种新方法，更提出了一种新的设计范式——**在预训练坐标系内重新定向，而非推倒重建**。该工作为 PEFT 领域树立了新的 SOTA，并有望推动对预训练模型几何结构的深入研究。

</details>

---

### 12. [Contrastive Distribution Matching for Amortized Sequential Monte Carlo in Discrete Diffusion](https://arxiv.org/abs/2605.23346)

**Authors**: Jaihoon Kim, Taehoon Yoon, Prin Phunyaphibarn, Seungjun Kim, Morteza Mardani, Minhyuk Sung  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.23346v1  

#### Abstract
Discrete diffusion models have emerged as powerful frameworks for generating structured categorical data. However, efficiently sampling from reward-tilted distributions remains a fundamental challenge. While Twisted Sequential Monte Carlo (SMC) offers asymptotic exactness for this task, estimating t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Contrastive Distribution Matching for Amortized Sequential Monte Carlo in Discrete Diffusion**

---

## **1. 主要贡献和创新点**

### **解决的问题**
离散扩散模型（discrete diffusion models）在生成结构化分类数据方面表现出色，但在从**奖励倾斜分布**（reward-tilted distribution）中高效采样时面临挑战。  
传统的 **Twisted Sequential Monte Carlo (SMC)** 虽然能实现渐近无偏采样，但其核心依赖于对“最优 twist 函数”的估计。在离散空间中，由于缺乏类似连续域中的 **Tweedie’s formula**，只能通过昂贵的 **Monte Carlo 估计**来逼近该函数，导致推理阶段计算开销巨大。

### **提出的新方法：Contrastive Distribution Matching (CDM)**
本文提出 **CDM**，一种用于**摊销**（amortize）SMC 推理成本的新型框架，其核心思想是：
- **学习一个参数化的 twist 函数**，通过**对比学习目标**（contrastive learning objective）进行训练。
- 利用正样本（high-reward 区域）和负样本（suboptimal 区域）来上权重和下权重，从而更有效地匹配目标分布。

### **相比现有方法的优势**
| 方面 | CDM | 现有方法（如 Soft Value） |
|------|-----|------------------------|
| **训练目标** | 最小化前向 KL 散度，具有对比结构 | 回归目标（MSE），直接拟合 twist 值 |
| **训练效率** | 利用扩散模型的闭式前向核（closed-form forward kernel），实现高效的缓冲区重用 | 需为每个时间步独立采样，效率低 |
| **推理开销** | 扭曲函数评估仅增加 <5% 的额外开销 | Monte Carlo 估计随 M 增加而线性增长 |
| **兼容性** | 可与任何 proposal 分布（包括已微调的模型如 d1, DRAKES）结合 | 通常独立于 proposal 设计 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与任务**
CDM 在多个领域进行了验证，涵盖文本与生物序列生成：

| 任务 | 数据集 | 奖励函数 |
|------|-------|---------|
| **Toxic Text Generation** | OpenWebText | 预训练毒性分类器（主奖励）与多语言分类器（heldout 奖励） |
| **Regulatory DNA Sequence Design** | Enhancer activity dataset (~700k DNA 序列) | Enformer 模型预测增强子活性 |
| **Protein Designability** | DPLM-2 模型生成蛋白 | 自洽性 RMSD (scRMSD)，折叠模型预测结构并计算距离 |
| **Diffusion LLM (dLLM) Alignment** | RewardBench 训练集 | Skywork Llama-3.1-8B 偏好分数（API 调用，不可导） |

### **实验设置与评估指标**
- **评估方式**：以**墙钟时间**（wall-clock time）为横轴，比较不同方法的**奖励得分**（主奖励与 heldout 奖励）。
- **多样性指标**（用于分析模式崩溃）：
  - 文本：Self-BLEU（n=4）、GPT2-XL 生成困惑度（PPL）
  - 蛋白质：FoldSeek 聚类数（Clusters）、平均 TMScore（inner-TM）
- **统一训练时间**：所有需优化的方法（CDM、Soft Value）在相同训练时间内比较。

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **BoN (Best-of-N)** | 从基础模型采样 N 个样本，选择最高奖励者 |
| **SMC** | 使用 Monte Carlo 估计 twist 函数（M=1,4） |
| **SMC+Grad** | 使用梯度指导的 proposal（仅适用于可导奖励） |
| **Soft Value** | 回归式学习 twist 函数（基于 MSE） |
| **d1 / DRAKES** | 微调 proposal 的方法，作为 CDM 的上游组合对象 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据与对比结果**
- **在所有任务中，CDM 均建立了新的 Pareto 前沿**，即在相同墙钟时间内获得更高奖励。
- **图 2 展示了四大任务的缩放曲线**：
  - **Toxic Text**：CDM 显著优于 SMC 和 Soft Value，尤其在 heldout 奖励上表现更强泛化能力。
  - **DNA 设计**：CDM 在给定和 heldout 奖励上均领先，SMC(M=4) 仅在主奖励上接近。
  - **Protein Designability**：SMC 因奖励计算昂贵（折叠模型）而扩展极差；CDM 在高成本场景下仍保持良好扩展性。
  - **dLLM Alignment**：CDM 在不可导 API 奖励下仍优于 BoN 和 Soft Value，显示其对复杂奖励的鲁棒性。

### **消融实验结果**
#### **(1) 正样本采样策略**
- **SMC vs IS**：使用 SMC 生成正样本比 IS 更有效（表 3b），因其通过重采样缓解了权重退化问题。

#### **(2) Monte Carlo 样本数 M（训练阶段）**
- **Soft Value**：性能随 M 增加而提升，但很快饱和（图 4）。
- **CDM**：对 M 不敏感，在 M=1 时即可达到高性能，说明其不依赖高质量 MC 估计。

#### **(3) 缓冲区更新频率 `nupdate`**
- 图 7 显示，增大 `nupdate`（即更频繁地复用正样本）可提高训练效率，尤其在奖励昂贵的任务中。

#### **(4) 与微调方法的协同效果**
- 图 3a：将 CDM 与 **d1** 或 **DRAKES** 结合，性能进一步提升，证明其**互补性**。
- 图 3b：微调方法（d1/DRAKES）存在**模式崩溃**（高 Self-BLEU，高 PPL），而 CDM 在保持高奖励的同时维持了多样性。

#### **(5) 训练收敛速度**
- 图 4 与图 9：CDM 比 Soft Value 收敛更快，且最终奖励更高，验证了**对比学习目标的有效性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **CDM 成功将 SMC 的推理瓶颈转化为可摊销的学习问题**，显著降低推理时的 twist 计算开销。
2. **对比学习目标比回归目标更高效、更稳定**，尤其在高维、稀疏奖励场景下。
3. **利用扩散模型的前向过程进行训练**，实现了高效的样本重用，特别适合奖励计算昂贵的应用。
4. **CDM 是通用框架**，可与任何 proposal（包括微调后的模型）结合，实现协同增益。
5. **CDM 能缓解微调方法带来的模式崩溃问题**，在保持高奖励的同时保留生成多样性。

### **局限性**
- 当前 twist 函数架构可能不足以建模极其复杂的奖励信号。
- 对于**极度稀疏的奖励信号**，学习期望未来奖励仍是挑战。
- 当前框架主要支持标量奖励，尚未原生支持二元或类别型奖励结构。

### **未来工作方向**
- 扩展框架以支持**二元和类别型奖励结构**。
- 探索更先进的 twist 函数架构以处理复杂奖励。
- 研究如何在**稀疏奖励**环境下提升学习效率。
- 将 CDM 应用于更多现实世界任务，如药物设计、代码生成等。

---

> **总结**：CDM 提出了一种新颖且高效的框架，通过**对比学习 + 摊销推理**解决了离散扩散模型中 reward alignment 的核心瓶颈。其实验验证全面，不仅在性能上超越现有方法，还展现出良好的兼容性与稳定性，为离散扩散模型的推理优化提供了重要新方向。

</details>

---

### 13. [Energy per Successful Goal: Goal-Level Energy Accounting for Agentic AI Systems](https://arxiv.org/abs/2605.22883)

**Authors**: Deepak Panigrahy, Aakash Tyagi  
**Category**: cs.AI  
**Published**: 2026-05-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.22883v1  

#### Abstract
Current AI energy benchmarks measure consumption at the granularity of a single model invocation or training run. For classical single-turn workloads this unit remains coherent. For agentic systems - where a single user goal may trigger multi-step orchestration, tool calls, retries, and failure-reco...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心总结：Energy per Successful Goal: Goal-Level Energy Accounting for Agentic AI Systems**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前主流的 AI 能耗评估方法（如 **energy-per-inference**）在评估 **Agentic AI 系统**时存在根本性缺陷。Agentic 系统通常涉及多步推理、工具调用、重试机制和失败恢复等复杂流程，单次“目标”可能触发多次模型调用。以“每次推理能耗”为单位会：
- 忽略失败尝试和重试的能耗；
- 将实现细节（如重试次数）误认为任务本身属性；
- 导致对真实能耗的系统性低估。

这种单位错配（unit misalignment）使得现有基准无法准确反映 Agentic 系统的真实能源成本。

---

### **提出了什么新方法或新思路**
论文提出 **A-LEMS**（Agentic LLM Energy Measurement System），并引入两个核心概念：

#### **(1) Energy per Successful Goal (EpG)**
- **定义**：完成一个成功目标所消耗的总能量，归一化于成功完成的目标数。
  $$
  \text{EpG} = \frac{\sum_{j \in W^+} E_{\text{workflow},j}}{|W^+|}
  $$
  其中 $W^+$ 是成功完成的工作流集合，$E_{\text{workflow}}$ 包括所有尝试（含失败）的能量。
- **意义**：将能耗评估从“模型调用粒度”提升到“用户目标粒度”，更符合实际使用场景。

#### **(2) Orchestration Overhead Index (OOI)**
- **定义**：Agentic 系统相对于线性（linear）系统的能耗开销比：
  $$
  \text{OOI} = \frac{\text{EpG}_{\text{agentic}}}{\text{EpG}_{\text{linear}}}
  $$
- **解释**：
  - OOI < 1：Agentic 更节能（如工具调用替代 token 生成）；
  - OOI > 1：Agentic 开销更高（常见于多步推理任务）；
  - OOI ≈ 1：两者能效相当。

---

### **相比现有方法的优势**
| 维度 | 传统方法（energy-per-inference） | A-LEMS（EpG + OOI） |
|------|-------------------------------|---------------------|
| **评估单位** | 单次推理 | 成功目标 |
| **是否包含失败能耗** | ❌ 否 | ✅ 是 |
| **是否支持多步/重试** | ❌ 不敏感 | ✅ 显式建模 |
| **可比性** | 跨架构不公平（鼓励拆分调用） | ✅ 公平比较不同架构 |
| **可复现性** | 缺乏标准化环境记录 | ✅ 三哈希协议（Hhw/Henv/Hrun）绑定硬件与运行状态 |

此外，A-LEMS 提供完整的 **五层观测管道**（L0–L4）和 **时间边界模型**（to/t1/t2），确保测量的准确性与可复现性。

---

## **2. 核心实验方法和设置**

### **使用的数据集与任务家族**
实验覆盖 **8 个任务家族**，分为两类：

#### **(1) 推理类任务（Reasoning Tasks）**
- **FQA**: 事实问答
- **SciQA**: 科学问答
- **LR**: 逻辑推理
- **GSM8K-B**: 基础算术题（单步）
- **GSM8K-M**: 多步算术题（需规划）

#### **(2) 工具增强类任务（Tool-Augmented Tasks）**
- **TG:Calc**: 调用计算器执行表达式
- **TG:DB**: 查询数据库
- **TG:Seq2**: 数据库 + 文件写入链式操作

> 所有任务均基于 **GSM8K** 或自定义工具图（tool-graph）构建，每项任务视为一个 **Goal**。

---

### **实验设置**

#### **两种推理模式**
| 类型 | 描述 | 测量范围 |
|------|------|---------|
| **Local Inference** | 使用 `Ollama/TinyLlama-1B` 在本地运行 | 完整 CPU package 能耗（RAPL） |
| **Remote Inference** | 使用 `Groq API` 调用 `llama-3.3-70b` | 仅客户端协调开销（client-side orchestration） |

#### **匹配对设计（Matched-Pair Design）**
- 每个 **Goal** 分别以 **Agentic** 和 **Linear** 方式执行一次；
- 在相同会话中背靠背运行，控制热态、DVFS 等变量；
- 总共 **827 对**（共 1654 次运行）用于主分析。

#### **评估指标**
- **EpG**（J/goal）：绝对能耗
- **OOI**（×）：相对开销指数
- **Success Rate**：成功率
- **Coverage (C)**：采样覆盖率（要求 ≥95% 为“黄金标准”）

#### **基线方法对比**
- **Linear Baseline**：无工具、无分支、无重试的单次提示-响应模式
- **Agentic System**：包含规划、工具调用、JSON 解析失败后重试等完整流程

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 任务类型 | 平均 EpG (Agentic) | 平均 EpG (Linear) | **平均 OOI** |
|--------|--------------------|-------------------|-------------|
| 所有推理任务（5类） | **888.1 J** | **205.3 J** | **4.33×** |
| FQA | — | — | 4.65× |
| SciQA | — | — | 5.79× |
| LR | — | — | 4.68× |
| GSM8K-B | — | — | 2.75× |
| GSM8K-M | — | — | **7.63×** |
| TG:Calc | — | — | **0.62×** |
| TG:DB | — | — | 0.96× |
| TG:Seq2 | — | — | 1.55× |

> ✅ **总体结论**：Agentic 系统在推理任务上平均消耗 **4.33 倍** 于线性系统的能量。

---

### **与基线方法的对比结果**

- 在 **GSM8K-M**（多步算术）任务中，OOI 高达 **7.63×**，表明复杂的规划与协调显著增加能耗。
- 在 **TG:Calc**（计算器调用）任务中，OOI 仅为 **0.62×**，说明当 Agentic 架构能用 O(1) 工具调用替代昂贵的 token 生成时，反而更节能。
- 这种 **方向性反转** 验证了 OOI 不是固定偏高，而是真正反映了 **Orchestration Structure** 的影响。

---

### **消融实验结果**

#### **(1) 无重试情况下的纯编排开销（Pure Orchestration Overhead）**
- 在 **305 个零失败重试的任务** 上：
  - Agentic 平均能耗：**1546.0 J**
  - Linear 平均能耗：**315.6 J**
  - OOI = **4.9×**
- 结论：即使没有失败重试，仅因 **planning loops、multi-step coordination、synthesis phases** 等结构差异，仍导致近 5 倍能耗差距。

#### **(2) 重试放大效应（Retry Amplification）**
- 在注入故障的实验中，**29 次重试** 占总 Agentic 能耗的 **26.9%**。
- 单次失败尝试平均耗能高于成功尝试（如 2256.1J vs 1358.4J），进一步加剧能耗失衡。

#### **(3) 相位级能耗分解（Phase-Level Attribution）**
在本地推理下，各阶段能耗分布如下：

| 阶段 | 平均能耗 (J) | 功率 (W) | 持续时间 (ms) |
|------|--------------|----------|---------------|
| Planning | 346.6 | 16.5 | 21,872 |
| Execution | 220.2 | 14.8 | 15,409 |
| Synthesis | 147.2 | 15.5 | 8,532 |
| **Gap（重试+协调）** | **2877.5** | — | — |

> 🔺 **Gap 项占主导**，验证了“协调开销”是主要瓶颈。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Energy-per-inference 是错误单位**：对于 Agentic 系统，它系统性低估真实能耗，尤其忽略失败与重试成本。
2. **EpG 是更合理的评估单位**：将能耗归一化到“成功目标”，使不同架构、可靠性策略之间具有可比性。
3. **Orchestration 是能耗主要来源**：实验显示，**4.33× 的平均开销并非来自更多计算，而是来自 planning、coordination 和 retry 机制**。
4. **OOI 具有方向敏感性**：在工具调用任务中 OOI < 1，证明该指标能正确识别节能场景，而非一味惩罚 Agentic 架构。
5. **Client-side orchestration 也有可观能耗**：远程推理中，客户端协调活动平均消耗 **1.0 W**，远高于空闲状态（0.2 W），这部分常被传统方法忽略。

---

### **方法的局限性**
1. **二元成功判定**：目前只支持“成功/失败”，不支持部分得分或质量分级。
2. **测量范围限制**：
   - 当前仅测量 **本地 CPU package**（RAPL），未包含 GPU、NIC 或远程服务器端能耗。
   - 对于远程推理，OOI 是 **下界估计**（lower bound），因为服务端额外调用未计入。
3. **依赖线性基线**：OOI 必须与 matched linear baseline 对比，无法单独报告。
4. **平台依赖性**：主要基于 Intel RAPL，ARM/macOS 支持有限。

---

### **未来工作方向**
1. **扩展测量维度**：
   - 整合 GPU、网络接口、远程服务器能耗（full-stack accounting）。
   - 支持跨云平台统一计量。
2. **动态质量感知 EpG**：结合输出质量评分，定义 **Quality-adjusted EpG**。
3. **自动化工具推荐引擎**：基于 OOI 预测，自动选择是否启用 Agentic 模式。
4. **标准化协议推动**：
   - 推动 EpG 成为 AI 能耗报告的标准单位；
   - 类似 PUE 在数据中心的地位。
5. **绿色 Agentic 设计指南**：基于 OOI 分析，提出降低编排开销的设计原则（如减少中间表示、优化重试策略）。

---

> 📌 **一句话总结**：  
> **Agentic AI 的能耗瓶颈不在 inference，而在 orchestration。EpG 与 OOI 提供了首个目标级、可复现、结构敏感的能耗评估框架，揭示了当前系统设计中的巨大优化空间。**

</details>

---

### 14. [The Efficiency Frontier: A Unified Framework for Cost-Performance Optimization in LLM Context Management](https://arxiv.org/abs/2605.23071)

**Authors**: Binqi Shen, Lier Jin, Hanyu Cai, Lan Hu, Yuting Xin  
**Category**: cs.CL  
**Published**: 2026-05-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.23071v1  

#### Abstract
Large language models (LLMs) increasingly rely on long-context processing, but expanding context windows introduces substantial computational and financial costs. Existing context reduction approaches, including retrieval and memory compression methods, are typically evaluated using performance and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：The Efficiency Frontier: A Unified Framework for Cost-Performance Optimization in LLM Context Management

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前在大型语言模型（LLM）中，**长上下文处理**虽然提升了任务能力（如多跳推理），但也带来了显著的计算和经济成本。现有的上下文管理策略（如检索、压缩、全上下文提示等）通常被**孤立评估**，仅分别报告性能（如F1）或效率（如token使用量），缺乏统一框架来系统比较不同策略在**性能与成本之间的权衡**。

这导致：
- 难以进行跨方法的公平比较；
- 缺乏部署导向的决策支持；
- 忽视了预处理复用（amortization）对实际系统的影响。

---

### 🚀 提出的新方法与创新思路
本文提出 **“The Efficiency Frontier”** ——一个**统一的、部署感知的优化框架**，用于系统化评估和选择LLM上下文管理策略。

#### 核心创新点包括：

1. **统一的效用函数建模**  
   定义了一个参数化的 `Efficiency Score`：
   $$
   \text{EfficiencyScore}(w) = w \cdot \text{F1} - (1 - w) \cdot \log(\text{EffectiveTokens})
   $$
   其中：
   - $ w \in [0,1] $ 控制对性能 vs 成本的偏好；
   - `EffectiveTokens` 考虑了预处理成本的摊销（amortized cost）。

2. **引入复用参数 $ N $**  
   明确建模上下文预处理的**可复用性**：
   $$
   \text{EffectiveTokens} = T_{\text{stage2}} + \frac{T_{\text{stage1}}}{N}
   $$
   - $ T_{\text{stage1}} $：预处理成本（如压缩、嵌入）；
   - $ T_{\text{stage2}} $：每次查询的推理成本；
   - $ N $：该预处理结果被复用的查询次数。

   这使得框架能反映真实场景（如持久记忆代理、共享缓存系统）中的成本动态。

3. **三阶段优化流程构建“效率前沿”**
   - **Stage 1**: 各策略内部配置优化（保留帕累托最优配置）；
   - **Stage 2**: 统一评估所有候选策略下的摊销成本；
   - **Stage 3**: 全局最优策略选择，生成随 $ w $ 变化的**全局效率前沿（Global Efficiency Frontier）**。

4. **提供决策地图（Decision Map）**
   输出两种实用工具：
   - 连续视角：$ w $ 扫描下的最优策略路径；
   - 离散指南：按目标F1查找最低成本策略。

---

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | 本文框架 |
|------|--------|---------|
| 评估方式 | 分离报告性能与成本 | 统一建模性能-成本权衡 |
| 决策支持 | 无明确推荐逻辑 | 提供部署感知的策略选择机制 |
| 复用建模 | 忽略预处理摊销效应 | 显式建模 $ N $，区分一次性 vs 可复用成本 |
| 比较公平性 | 不同实验设置下对比 | 统一评估协议，支持跨策略直接比较 |

> ✅ **优势总结**：从“报告指标”升级为“辅助决策”，推动上下文管理研究向**可持续、可部署的系统设计**演进。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **HotpotQA**：一个多跳问答数据集，包含相关文档和干扰项，适合测试上下文筛选与压缩能力。
- 实验规模：随机采样 **5,000个实例**（固定seed=42），确保可复现性和计算可行性。

### ⚙️ 实验设置
- **模型**：使用 **GPT-5.4 mini**（OpenAI技术报告[37]），兼顾推理能力和评估效率；
- **Prompt统一**：所有策略采用标准化prompt，控制变量；
- **确定性解码**：关闭随机性，保证结果稳定；
- **评估粒度**：每种策略尝试多种配置（如k值、压缩率），全面覆盖操作空间。

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **F1 Score** | 主要任务性能指标（问答准确率） |
| **EffectiveTokens** | 摊销后的每问有效token数（含预处理摊销） |
| **Efficiency Score** | 参数化效用函数，用于策略排序 |
| **Transition Points** | 不同策略间的切换边界（由$ w $决定） |

---

### 🆚 基线方法对比
共评估五类代表性策略：

| 策略 | 类型 | 是否有Stage1成本 |
|------|-----|----------------|
| **Full-Context Prompting** | 全文拼接 | 无（高Stage2成本） |
| **Oracle Retrieval** | 理想检索（仅用支持句） | 无（非部署可行） |
| **Memory Compression** | LLM驱动压缩（如摘要） | 有（高Stage1，低Stage2） |
| **TF-IDF (Vanilla)** | 词频匹配 | 无（轻量） |
| **TF-IDF (Query-Aware)** | 结合问题的词频加权 | 无 |
| **Semantic Embedding Retrieval** | 向量相似度检索（如dense retrieval） | 无 |

> 所有方法均在相同条件下运行，并通过公式(1)统一计算 `EffectiveTokens`。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）效率前沿分析（Fig. 1 & Fig. 2）
- 每个策略都有其内在帕累托前沿（Pareto Frontier），但最优操作点随 $ w $ 动态变化；
- **Query-aware retrieval** 明显优于 vanilla TF-IDF，在相同成本下获得更高F1；
- **Memory Compression** 在高 $ N $ 下表现突出，因摊销后Stage1成本降低；
- **Full-Context** 仅在追求最高性能时必要，但代价高昂。

#### （2）不同复用水平下的最优策略转移（Table I）

| 性能区间 | $ N=1 $（低复用） | $ N=100 $（高复用） |
|----------|--------------------|-----------------------|
| **Efficiency-oriented**<br>(F1 < 0.78) | TF-IDF QA (k=16)<br>或 Mem. Comp. (2.5×) | 同左 |
| **Balanced**<br>(0.78 ≤ F1 < 0.82) | Full-Context | **Memory Compression (2×)** |
| **High-performance**<br>(F1 ≥ 0.82) | Full-Context | Full-Context |

> 💡 观察到明显的策略迁移：随着 $ N $ 增大，memory compression 取代轻量检索成为平衡区间的首选。

---

### 📉 效率增益量化

| 场景 | 成果 |
|------|------|
| **Balanced Regime @ F1≈0.78**<br>从 $ N=1 $ → $ N=100 $ | 最优策略从 TF-IDF QA (566 tokens) → Memory Compression (424 tokens)<br>➡️ **有效token减少约25%** |
| **Near High-Performance @ F1≈0.80**<br>从 $ N=1 $ → $ N=100 $ | 从 Full-Context (1308 tokens) → Memory Compression (584 tokens)<br>➡️ **有效成本下降超50%** |
| **vs Full-Context Baseline** | 在摊销条件下，memory compression 达到相近性能时，token成本降低 >50% |

> ✅ 表明：**不是改进单个算法，而是正确选择策略**，即可实现显著效率提升。

---

### 🔍 消融实验与关键观察（隐含于分析中）
尽管未设独立“消融”章节，文中通过以下方式验证设计有效性：

1. **$ N $ 的影响验证**  
   - 当 $ N=1 $：轻量检索占优；
   - 当 $ N=100 $：memory compression 占据更大前沿区域；
   ➡️ 验证了摊销机制的重要性。

2. **log-cost penalty 的合理性**  
   使用 $\log(\text{EffectiveTokens})$ 而非线性惩罚，更符合大规模系统对边际成本增长的容忍度。

3. **策略多样性验证**  
   包括零成本检索、高成本压缩、全文提示等，证明框架适用于异构策略比较。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **性能与成本呈强非线性关系**  
   提升性能需不成比例地增加token消耗。例如：
   - 从 F1≈0.78 到 F1≈0.84，token使用翻倍以上；
   - 存在明显 **diminishing returns**。

2. **不存在单一最优策略**  
   - **低性能需求 + 低复用**：TF-IDF QA 最优；
   - **中等性能 + 高复用**：Memory Compression 更高效；
   - **高性能极限**：Full-Context 仍不可替代。

3. **摊销效应重塑策略优先级**  
   - 在 $ N>1 $ 场景（如企业知识库、持续对话代理），前期昂贵的压缩反而更划算；
   - 强调了将**部署模式纳入评估**的重要性。

4. **决策应基于操作区间而非绝对指标**  
   提出“效率导向 / 平衡 / 高性能”三区间分类，帮助实践者快速定位最优策略。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖人工设定 $ w $** | 权重 $ w $ 需根据应用场景手动调整，缺乏自动学习机制 |
| **假设固定效用函数形式** | log-cost 和线性F1组合可能不适用于所有任务类型 |
| **仅考虑token成本** | 未直接建模延迟、能耗、硬件占用等其他资源维度 |
| **基于特定数据集推导阈值** | 具体转换点（如F1=0.78）来自HotpotQA，泛化需谨慎 |

---

### 🔮 未来工作方向（原文建议）

1. **扩展至更多任务类型**  
   如代码生成、文档理解、Agent Memory Systems 等长上下文应用。

2. **整合更多系统目标**  
   将 latency、energy consumption、monetary cost 等纳入优化目标。

3. **自适应偏好建模**  
   开发 learnable 或 context-aware 的 $ w $ 函数，替代人工设定。

4. **结合领域知识优化表示**  
   引入 domain-aware representation learning 提升压缩保真度。

5. **探索结构化记忆架构**  
   如图结构记忆、索引增强检索，进一步提升长期上下文利用效率。

---

## ✅ 总结

> **The Efficiency Frontier 框架标志着上下文管理评估范式的转变**：  
> 从“哪个方法更好”转向“**在何种条件下哪个策略最合适**”。

它不仅是一个评估工具，更是连接研究与工业部署的桥梁，推动LLM系统向**更高效、更可持续、更具决策智能的方向发展**。

</details>

---

### 15. [ARES: Automated Rubric Synthesis for Scalable LLM Reinforcement Learning](https://arxiv.org/abs/2605.23454)

**Authors**: Xiaoyuan Li, Keqin Bao, Moxin Li, Yubo Ma, Yichang Zhang, Wenjie Wang, Fuli Feng, Dayiheng Liu  
**Category**: cs.CL  
**Published**: 2026-05-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.23454v1  

#### Abstract
Rubric-based rewards offer a promising way to extend reinforcement learning (RL) for large language models beyond tasks with automatically verifiable answers. However, scaling rubric-based RL remains challenging: existing approaches often rely on expert-written rubrics and manually constructed quest...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ARES: Automated Rubric Synthesis for Scalable LLM Reinforcement Learning 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现有的 **Reinforcement Learning with Verifiable Rewards (RLVR)** 方法在 LLM 后训练中取得了显著成功，但其应用受限于**任务必须具有可自动验证的答案**（如数学、编程等）。这导致以下两个核心瓶颈：

- **领域局限性**：仅适用于有明确正确答案的任务，难以扩展到开放性、多维度的复杂任务（如医疗咨询、写作、指令遵循）。
- **奖励稀疏性**：通常依赖二元奖励（binary reward），缺乏对响应质量的细粒度监督信号。

此外，已有基于 **rubric-based rewards** 的方法虽然能提供多维评价，但严重依赖专家手动编写 rubric 和问题集，**难以规模化**。

---

### **提出了什么新方法或新思路**

本文提出 **ARES (Automated Rubric synthEsis for Scalable RL)**，一个全自动构建大规模 rubric 注释 RL 数据的框架。其核心创新在于：

- **从原始预训练文档出发**，自动生成 **self-contained 的 QA 对** 及其对应的 **question-specific weighted rubric**。
- 在单次生成过程中联合产出 `question`、`reference answer` 和 `rubric`，确保三者语义一致且扎根于源文档。
- 引入 **domain 和 persona 条件控制**（如医生、学生、护理者视角），提升生成数据的多样性与现实相关性。
- 设计了完整的六阶段 pipeline（见图1），涵盖文档过滤、条件标注、共生成、质量验证、rubric 验证与格式转换。

> ✅ **关键突破**：首次实现 **question-specific rubric 的自动化、规模化合成**，使 rubric-based RL 能应用于开放域、多维度任务。

---

### **相比现有方法的优势**

| 维度 | ARES 的优势 |
|------|-------------|
| **可扩展性 (Scalable)** | 不依赖人工标注，直接从预训练语料生成，支持大规模部署 |
| **领域广度 (Multi-Domain)** | 覆盖 10 个领域（含医疗、社科、工程等），远超仅限 math/code 的方法 |
| **奖励丰富性 (Rubric Rewards)** | 提供加权多维奖励，而非单一 binary reward，指导模型优化多个响应维度 |
| **任务适配性 (Question-Specific)** | 每个问题都有定制化 rubric，比通用 rubric 更精准 |

> 📊 表1显示，ARES 是唯一同时满足 **Multi-Domain、Rubric Rewards、Doc-Grounded、Scalable** 四项标准的方法。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **训练数据**：通过 ARES 自动合成的 **101,847 个 rubric 注释实例**，来自三大预训练语料子集：
  - DCLM
  - FineWeb-Edu
  - FinePDFs
- **覆盖领域**（共10类）：
  - Social Science (18.8K)
  - Technology & Engineering (18.3K)
  - Medicine & Health (13.9K)
  - Travel & Lifestyle (12.7K)
  - Commerce & Economics (12.4K)
  - Natural Science (10.1K)
  - Education (8.8K)
  - Others (Coding: 1.6K, Math: 930)

> 🔢 平均每个实例包含 **10.88 个 rubric criteria**，其中正向奖励占 74%，负向惩罚占 26%（表3）。

---

### **实验设置和评估指标**

- **基础模型**：`Qwen3-4B-Base`
- **训练方式**：
  - 所有方法训练 3 轮，batch size 128
  - 使用 **Group Relative Policy Optimization (GRPO)** 进行 RL 训练
  - Rubric judge 使用 `Qwen3-32B`
- **评估基准（7个）**：
  - **知识推理**：MMLU-Pro
  - **数学推理**：GSM8K
  - **代码生成**：HumanEval+, MBPP+
  - **医疗问答**：HealthBench
  - **写作质量**：WritingBench
  - **指令遵循**：IFEval
- **报告指标**：各基准得分及七项平均分（Avg）

---

### **基线方法对比**

| 方法 | 描述 |
|------|------|
| **CPT (Continual Pretraining)** | 在相同源文档上继续预训练（next-token prediction） |
| **NaturalReasoning** | 基于 NaturalReasoning 数据集进行 Supervised Fine-Tuning (SFT) |
| **Webscale (Binary-Reward RL)** | 使用 WEBSCALE-RL 方式，基于 QA 对进行 binary reward RL |
| **ARES-SFT** | 在 ARES 生成的 QA 对上进行 SFT（不使用 rubric） |
| **ARES-RL (Ours)** | 使用 ARES 生成的 rubric 进行 RL 训练 |

> 💡 此设计可分离出 **数据本身 vs. 训练目标（SFT vs. RL）vs. 奖励形式（binary vs. rubric）** 的影响。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（表4）**

| Method | MMLU-Pro | GSM8K | HumanEval+ | MBPP+ | HealthBench | WritingBench | IFEval | **Avg** |
|--------|----------|-------|------------|--------|--------------|---------------|--------|---------|
| CPT | 46.02 | 82.34 | 32.32 | 59.40 | 35.04 | 36.98 | 39.39 | 47.36 |
| NaturalReasoning | 47.96 | 81.50 | 32.93 | 61.15 | 32.94 | 36.77 | 28.15 | 45.91 |
| Webscale | 49.50 | 84.91 | 33.54 | 61.40 | 36.08 | 37.09 | 35.61 | 48.30 |
| ARES-SFT | 50.56 | 85.67 | 31.10 | 61.40 | 35.78 | 37.05 | 46.41 | 49.71 |
| **ARES-RL (ours)** | **49.36** | **86.96** | **34.76** | **63.16** | **41.45** | **38.24** | **54.88** | **52.69** |

> ✅ **ARES-RL 达到最高平均分 52.69**，显著优于所有基线。

---

### **与基线方法的对比结果**

- **vs. CPT**：+5.33 分，说明从 token prediction 转为 QA+reward 优化更有效。
- **vs. SFT (NaturalReasoning)**：+6.78 分，表明 ARES 数据 + RL 训练优于通用 SFT。
- **vs. Binary-Reward RL (Webscale)**：+4.39 分，证明 **rubric 奖励优于 binary 奖励**。
- **vs. ARES-SFT**：+2.98 分（同数据下），说明 **rubric-based RL 优于 SFT**，即使有高质量参考答案。

> 🔺 特别是在开放性任务上提升巨大：
> - **HealthBench**: +5.37（36.08 → 41.45）
> - **IFEval**: +19.27（35.61 → 54.88）
> - **WritingBench**: +1.15（37.09 → 38.24）

---

### **消融实验结果（表5）**

| Reward Strategy | Avg Score |
|------------------|-----------|
| Blind Judge (holistic LLM judge) | 49.53 |
| General Rubric (fixed criteria) | 51.79 |
| Reference Answer (similarity-based) | 46.25 |
| **ARES-RL (question-specific rubric)** | **52.69** |

#### 关键发现：

- **Structured > Holistic**：结构化 rubric（51.79）优于整体打分（49.53），说明分解评价更稳定、可解释。
- **Question-Specific > General Rubric**：定制化 rubric 比通用 rubric 平均高 0.9 分，在 IFEval 上提升达 +9.22。
- **Reference-Based 不稳定**：虽在 HealthBench 上表现最好（50.13），但在 IFEval 上暴跌至 10.54，因其无法处理约束多样性。

> ✅ 结论：**question-specific rubric 在跨任务鲁棒性和平均性能上最优**。

---

## 4. 关键结论和发现

### **主要发现**

1. **Rubric-based RL 显著优于传统方法**，尤其在多维开放任务（如医疗、写作、指令遵循）上表现突出。
2. **ARES 实现了 rubric 数据的完全自动化生成**，无需人工干预即可构建高质量、多样化的大规模 RL 数据集。
3. **联合生成 QA 与 rubric** 可保证三者一致性，且支持 **instance-level reward supervision**，比 task-level rubric 更精细。
4. **即使在非目标领域（如 Math/Coding 仅占 ~2% 数据）也取得 SOTA 性能**，说明 rubric RL 具备良好迁移能力。

> 🎯 “ARES 将 raw 文档转化为 targeted 后训练监督信号”，实现了从被动预测到主动优化的转变。

---

### **方法的局限性**

- **模型规模限制**：实验仅在 `Qwen3-4B-Base` 上进行，未验证是否可扩展到更大模型（如 70B）。
- **Judge 模型受限**：使用 `Qwen3-32B` 作为 rubric judge，更大或更强 judge 可能进一步提升奖励质量。
- **潜在偏见风险**：
  - 源文档中的偏见可能被继承；
  - LLM 自动生成 rubric 和 judge 判断也可能引入模型自身偏差。
- **计算成本较高**：每轮训练需调用大 judge 模型评分，增加推理开销。

---

### **未来工作方向**

- 将 ARES 扩展至更大模型和更多语言。
- 探索轻量化 judge 模型或蒸馏策略以降低 RL 成本。
- 引入人类反馈对自动生成的 rubric 进行校准与纠偏。
- 构建动态 rubric 更新机制，支持持续学习与领域适应。

---

> ✅ **总体评价**：ARES 是推动 LLM 强化学习迈向 **开放域、多维度、可规模化** 的重要一步，为下一代对齐技术提供了新范式。

</details>

---

### 16. [From Residuals to Reasons: LLM-Guided Mechanism Inference from Tabular Data](https://arxiv.org/abs/2605.22897)

**Authors**: Mohammad R. Rezaei, Rahul G. Krishnan  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.22897v1  

#### Abstract
A persistent challenge in machine learning for scientific applications is jointly achieving prediction and understanding. Statistical models excel on structured data but operate as black boxes, while existing interpretability methods are largely inspective: they answer "which features matter?" but d...

---

### 17. [ThriftAttention: Selective Mixed Precision for Long-Context FP4 Attention](https://arxiv.org/abs/2605.23081)

**Authors**: Joe Sharratt  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.23081v1  

#### Abstract
Efficient attention algorithms are critical to mitigate the quadratic cost of attention in long-context workloads. Prior work utilises block-scaled quantisation techniques on Blackwell GPUs to move attention computation to 4-bit precision to accelerate inference. However, these techniques result in ...

---

### 18. [Non-normal spectral signatures of instability in neural network training dynamics](https://arxiv.org/abs/2605.23476)

**Authors**: Souvik Ghosh  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.23476v1  

#### Abstract
Training instabilities in deep networks - loss spikes, oscillatory convergence, and gradient pathologies - are empirically prevalent but lack a rigorous operator-theoretic explanation. We show that the linearized update operators for practically used optimizers are generically non-normal: for Adam, ...

---

### 19. [PathCal: State-Aware Reflection-Marker Calibration for Efficient Reasoning](https://arxiv.org/abs/2605.23074)

**Authors**: Lingyu Jiang, Zirui Li, Shuo Xing, Peiran Li, Tsubasa Takahashi, Dengzhe Hou, Zhengzhong Tu, Kazunori Yamada, Fangzhou Lin  
**Category**: cs.AI  
**Published**: 2026-05-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.23074v1  

#### Abstract
The emergence of Large Reasoning Language Models (LRMs) has paved the way for tackling complex reasoning tasks through test-time scaling by generating long-form Chain-of-Thought (CoT) trajectories during inference. Meanwhile, these trajectories often contain explicit reflection markers such as ``wai...

---

### 20. [Multi-Factor Trust-Driven Secure Communication Model for Cloud-Based Digital Twins](https://arxiv.org/abs/2605.23566)

**Authors**: Deepika Saxena, Ashutosh Kumar Singh  
**Category**: cs.DC  
**Published**: 2026-05-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.23566v1  

#### Abstract
Cloud-based Digital Twin (DT) platforms enable real-time monitoring, simulation, and collaborative decision-making across distributed clients. However, ensuring secure and trustworthy communication remains a critical challenge due to heterogeneous client behavior, resource contention, and evolving a...

---

### 21. [Enhancing Energy Efficiency in Scientific Workflows through CFD based PIVAEs](https://arxiv.org/abs/2605.23850)

**Authors**: Ali Zahir, Ashiq Anjum, Mark Wilkinson, Jeyan Thiyagalingam  
**Category**: cs.DC  
**Published**: 2026-05-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.23850v1  

#### Abstract
The growing complexity and scale of scientific workflows in high performance computing (HPC) environments have led to significant challenges in managing energy consumption without compromising computational performance. Traditional scheduling strategies often fail to account for the complex interpla...

---

### 22. [WMAttack: Automated Attack Search for Adversarial Evaluation of World-Model Agents](https://arxiv.org/abs/2605.23220)

**Authors**: Zhixiang Guo, Siyuan Liang, Shi Fu, Cheng Guo, Andras Balogh, Mark Jelasity, Dacheng Tao  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.23220v1  

#### Abstract
Despite the growing use of world models as decision-making agents, their adversarial robustness remains underexplored due to the lack of dedicated automated evaluation methods. A key obstacle is that attack evaluation must be both accurate and efficient: weak manually tuned attacks can overestimate ...

---

### 23. [Accelerating Divisible Load Processing Through Machine Learning: A Practical Framework for Large-Scale Workloads](https://arxiv.org/abs/2605.23247)

**Authors**: Bharadwaj Veeravalli  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.23247v1  

#### Abstract
In this paper, we introduce the first machine learning framework for predicting optimal processing times in Single-Level Tree Network (SLTN) architectures for the Divisible Load Theory (DLT) paradigm. Using a feedforward neural network(FNN) with 16 engineered features, we train a model on 100,000 sy...

---

### 24. [Approaching I/O-optimality for Approximate Attention](https://arxiv.org/abs/2605.23751)

**Authors**: P\'al Andr\'as Papp, Aleksandros Sobczyk, Anastasios Zouzias  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.23751v1  

#### Abstract
We revisit the I/O complexity of attention in large language models. Given query-key-value matrices $Q,K,V\in\mathbb{R}^{n\times d}$, and a machine with fast memory size $M$, the goal is to compute the "attention matrix" $A=\text{softmax}(Q K ^{\top}/\sqrt{d}) V$ with the minimal number of data tran...

---

### 25. [Complete-muE: Optimal Hyperparameter Transfer and Scaling for MoE Models](https://arxiv.org/abs/2605.23893)

**Authors**: Hongwu Peng, Ohiremen Dibua, Yuanjun Xiong, Yifan Gong, Jianming Zhang, Yan Kang  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.23893v1  

#### Abstract
We propose Complete-muE, a framework which targets hyperparameter transfer across dense FFN and any Mixture-of-Experts (MoE) setups in transformer blocks. Existing tools such as $\mu$P (requires fixed architectue) or SDE (requires fixed per-step token count) cannot directly solve the hyperparameter ...

---

### 26. [EDGE-OPD: Internalizing Privileged Context with Evidence Guided On-Policy Distillation](https://arxiv.org/abs/2605.23493)

**Authors**: Aristotelis Lazaridis, Dylan Bates, Aman Sharma, Brian King, Vincent Lu, Jack FitzGerald  
**Category**: cs.AI  
**Published**: 2026-05-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.23493v1  

#### Abstract
On-Policy Distillation (OPD) has gained wide attraction as an LLM post-training paradigm due to its effectiveness in improving capabilities without introducing model distribution drift, and consequently, regression in general tasks. On-Policy Self-Distillation (OPSD) is an efficient use-case of OPD,...

---

### 27. [A Reproducible Universal Dependencies-Style Pipeline for Katharevousa Greek Parliamentary Text](https://arxiv.org/abs/2605.22978)

**Authors**: George Mikros, Fotios Fitsilis  
**Category**: cs.CL  
**Published**: 2026-05-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.22978v1  

#### Abstract
Katharevousa Greek remains poorly served by contemporary NLP pipelines despite its importance for legal, administrative, and parliamentary archives. We present a reproducible workflow for building and evaluating a Universal Dependencies-style parsing resource for Katharevousa parliamentary questions...

---

### 28. [Hidden Human-Like Nature of Machine-Generated Texts: Theory and Detection Enhancement](https://arxiv.org/abs/2605.23190)

**Authors**: Chenwang Wu, Yiu-ming Cheung, Bo Han, Defu Lian  
**Category**: cs.CL  
**Published**: 2026-05-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.23190v1  

#### Abstract
Machine-generated texts (MGTs) produced by large language models (LLMs) are increasingly prevalent across various applications, while their potential misuse in fake news propagation and phishing has raised serious concerns, highlighting the need for MGT detection. Existing paragraph-level detection ...

---

### 29. [Learned Relay Representations for Forward-Thinking Discrete Diffusion Models](https://arxiv.org/abs/2605.22967)

**Authors**: Benjamin Rozonoyer, Jacopo Minniti, Dhruvesh Patel, Neil Band, Avishek Joey Bose, Tim G. J. Rudner, Andrew McCallum  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.22967v1  

#### Abstract
When Masked Diffusion Models (MDMs) generate sequences through iterative refinement, the rich internal computation over masked positions is discarded, forcing every subsequent refinement step to recompute the valuable internal information stored as model representations. To avoid a hard reset betwee...

---

### 30. [RelPrism: A Multi-Faceted Pre-training Framework with Self-Generated Tasks for Relational Databases](https://arxiv.org/abs/2605.23241)

**Authors**: Jinyu Yang, Cheng Yang, Junze Chen, Zedi Liu, Muhan Zhang, Hanyang Peng, Chuan Shi  
**Category**: cs.LG  
**Published**: 2026-05-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.23241v1  

#### Abstract
Relational databases (RDBs) remain the cornerstone of modern data systems and support diverse predictive tasks. Recent relational deep learning (RDL) methods enable end-to-end prediction by converting RDBs into graphs, where rows are represented as nodes and inter-table interactions are represented ...

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

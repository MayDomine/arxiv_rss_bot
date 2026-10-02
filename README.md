# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-02 11:23:04 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [QATFactory: A Versatile, Deployment-Aligned Framework for Quantization-aware Training and Distillation of LLMs](https://arxiv.org/abs/2609.39223)

**Authors**: Weili Xu, Jisen Li, Yuqing Jian, Chenxi Li, Zhizhou Sha, Yifan Yu, Qingyang Wu, Chenfeng Xu, Zhongzhu Zhou, Tianyi Zhang, Ben Athiwaratkun  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2609.39223v2  

#### Abstract
Large language model (LLM) inference is increasingly moving toward lower precision to realize the throughput of hardware accelerators, but aggressive post-training quantization (PTQ) can degrade model quality. We present QATFactory, an open-source framework for deployment-aligned quantization-aware ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：QATFactory: A Versatile, Deployment-Aligned Framework for Quantization-aware Training and Distillation of LLMs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前主流的 **Post-Training Quantization (PTQ)** 方法在极端低精度（如 W4A4）下会导致显著的模型质量下降，尤其是在推理和智能体类任务中。此外，现有的 **Quantization-aware Training (QAT)** 框架存在以下问题：
- 使用的量化格式与生产级推理引擎（如 vLLM、llama.cpp）不一致，导致训练后无法直接部署。
- 缺乏对新兴硬件原生格式（如 NVFP4、MXFP4）的支持。
- 全参数 QAT 内存开销巨大，难以扩展到大模型。

### 提出了什么新方法或新思路
作者提出了 **QATFactory** —— 一个开源、通用且与部署对齐的 QAT 框架，支持多种量化格式和训练范式：

#### 核心设计思想
- **Deployment-Aligned Simulation**：在训练时模拟目标推理引擎的精确数值行为（包括可表示值、缩放层级、块结构等），但所有矩阵乘法仍以 BF16 执行，从而实现“硬件无关”的低精度训练。
- **统一接口抽象**：通过标准化的 `quantize/dequantize` 接口支持多种格式（NVFP4、MXFP4、Q4_K），便于扩展。
- **双模式训练支持**：
  - **Quantization-aware Distillation (QAD)**：用高精度教师模型监督低精度学生模型。
  - **Quantization-aware Reinforcement Learning (QARL)**：结合低精度 rollout 和量化感知策略更新。

### 相比现有方法的优势
| 维度 | QATFactory | 传统 QAT/PTQ |
|------|------------|--------------|
| **部署对齐性** | ✅ 完全对齐 vLLM/SGLang/llama.cpp | ❌ 数值行为不一致 |
| **硬件依赖** | ❌ 不需要原生 FP4 支持（可在 H100 上训 NVFP4） | ✅ 需要 Blackwell 等新硬件 |
| **训练效率** | ✅ 支持 LoRA-based QAD，大幅降低显存占用 | ❌ 全参数 QAT 显存爆炸 |
| **输出可用性** | ✅ 导出即用，无需额外转换或校准 | ❌ 需二次量化或适配 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Distillation 数据**：来自 [Open Perfect Blend](https://arxiv.org/abs/2409.20370)，包含由原始 BF16 模型生成的 prompt-response 对。
- **Reinforcement Learning 数据**：DeepMath-103K，用于数学推理任务。
- **评估基准**：
  - **通用能力**：MMLU-P, MMMLU, GPQA-D, Winogrande, HellaSwag
  - **数学推理**：AIME24/AIME25, MATH500, OlympiadBench, GSM8K
  - **代码能力**：LiveCodeBench (LCB), BigCodeBench (BCB)

### 实验设置
- **模型范围**：从 8B 到 230B 参数，涵盖 Dense 和 MoE 架构（如 Qwen3.5-9B, DeepSeek R1 Distill Llama 8B, Qwen3-30B-A3B, MiniMax M2.7）
- **量化格式**：
  - **NVFP4**（W4A4）：带两级缩放（tensor + block）
  - **MXFP4**（W4A4）：仅 block 缩放，更粗粒度
  - **Q4_K**（W4A16）：llama.cpp 使用的非对称整数量化
- **训练配置**：
  - 序列长度：8K 或 32K
  - Batch size：16
  - Optimizer：AdamW，学习率 $1\times10^{-6}$ 衰减
  - LoRA Rank：测试了 r=4,16,64,256

### 评估指标
| 类型 | 指标 |
|------|------|
| **下游性能** | 各基准平均准确率（Benchmark Accuracy） |
| **分布保真度** | 相对于 BF16 教师模型的 KL Divergence（越小越好） |
| **训练效率** | 每步 wall-clock 时间（秒） |
| **内存消耗** | GPU 显存使用量（GiB） |

### 基线方法对比
- **PTQ Baselines**：
  - **RTN (Round-to-Nearest)**
  - **GPTQ**（基于二阶梯度优化）
  - **MSE-optimal PTQ**（用于 Q4_K）
- **训练方式对比**：
  - Full-parameter QAD vs. LoRA-QAD
  - W4A16 vs. W4A4 训练
  - QARL vs. BF16 RL + PTQ

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen3.5-9B）

#### 在 NVFP4 下的表现
| 方法 | 平均 Benchmark Accuracy | LCB | BCB | AIME25 |
|------|--------------------------|-----|-----|--------|
| BF16 (Reference) | 71.7 | 82.3 | 31.7 | 50.7 |
| RTN (PTQ) | 65.4 | 63.7 | 25.0 | 43.3 |
| GPTQ (PTQ) | 63.3 | 66.0 | 23.3 | 40.0 |
| **QAD (W4A16)** | **68.9** | **83.0** | **29.7** | **43.3** |

> ✅ QAD 在 NVFP4 下将平均准确率提升至 **68.9%**，接近 BF16 水平，并显著优于最强 PTQ（+3.5 pts）

#### 在 MXFP4 下的表现
| 方法 | 平均 Benchmark Accuracy | LCB |
|------|--------------------------|-----|
| RTN (PTQ) | 56.4 | 39.0 |
| GPTQ (PTQ) | 56.4 | 39.0 |
| **QAD (W4A4)** | **66.0** | **72.3** |

> ✅ QAD 在 MXFP4 下实现 **+9.6 pts** 提升，远超 PTQ

---

### 与基线方法的对比结果

#### 总体趋势（Table 2 & 3）
- **QAD consistently outperforms PTQ** across all models and formats。
- 在 MoE 模型上也有效：
  - MiniMax M2.7 (230B)：QAD 较 PTQ 提升 **2.5 pts** 平均准确率
- 在 Q4_K 格式下：
  - Qwen3.5-9B：QAD 达到 **64.57%** 平均准确率，较 MSE-optimal PTQ 提升 **6.93 pts**

#### QARL 结果（Table 8）
| 方法 | Mean Reasoning Accuracy | Throughput Speedup |
|------|--------------------------|--------------------|
| BF16 RL → RTN | 48.1% | 1.00× |
| **NVFP4 QARL** | **50.8%** | **1.23×** |

> ✅ QARL 不仅提升最终模型质量（+2.7 pts），还因原生低精度 rollout 加速训练流程

---

### 消融实验结果

#### （1）激活是否量化？——格式相关！

| 格式 | 最佳训练策略 | 原因推测 |
|------|-------------|--------|
| **NVFP4** | **W4A16**（仅权重量化）更好 | 激活量化噪声大，破坏梯度稳定性 |
| **MXFP4** | **W4A4**（权激都量化）更好 | 更大的激活量化误差需提前适应 |

> 🔍 发现：**不能简单认为“训练越贴近部署越好”**，必须根据格式特性选择策略。

#### （2）LoRA 是否能替代全参微调？

| 方法 | GPU Memory (GiB) | Avg Acc | KL Divergence |
|------|------------------|---------|---------------|
| Full-parameter QAD | ~167 | 68.9 | 0.0682 |
| LoRA (r=16) | **~57**（↓2.9×） | 67.8 | 0.0735 |
| LoRA (r=256) | ~100+ | 67.6 | 0.0777 |

> ⚠️ 虽然 LoRA 显著降低显存（2.9×），但**即使增大 rank 也无法追上全参 QAD**，说明低秩空间不足以完全建模量化补偿。

#### （3）长序列 vs 短序列训练（固定 token budget）

| 设置 | Avg Acc (AIME25+BCB+LCB) |
|------|----------------------------|
| 4K × 多序列 | 50.1% |
| **32K × 少序列** | **51.9%**（↑1.9 pts） |

> ✅ 更长上下文有助于泛化，尤其在跨域迁移中表现更强（代码 → 数学推理）

---

## 4. 关键结论和发现

### 主要发现
1. **QAD 显著优于 PTQ**：在多种模型、架构、量化格式下，QAD 均能恢复大部分因量化损失的质量，逼近 BF16 表现。
2. **训练精度应按格式定制**：
   - NVFP4：推荐 **W4A16** 训练
   - MXFP4：必须 **W4A4** 训练
3. **LoRA 可大幅降显存，但无法闭合性能差距**：适合资源受限场景，但追求极致性能仍需全参训练。
4. **长序列更有益于泛化**：在相同 token 预算下，使用更少但更长的序列效果更好。
5. **QARL 是高效之选**：相比“先 BF16 RL 再 PTQ”，QARL 同时提升质量和训练速度（1.23× throughput）。

### 方法的局限性
- 当前框架仍依赖高精度（BF16）计算，未真正实现端到端低精度训练。
- LoRA-QAD 的性能上限尚未突破，可能需要更复杂的适配结构。
- 对某些极端压缩格式（如 3-bit lookup table）尚不支持。

### 未来工作方向
- 扩展支持更多新兴量化格式（如 NVIDIA Rubin 架构的 LUT-TensorCore）。
- 支持更多推理引擎（TensorRT-LLM、DeepSpeed-Inference）。
- 探索更高效的参数高效微调方法（如 Adapter、BitFit + QAT）。
- 开源训练好的低精度 checkpoint 和完整 recipe，推动社区复现与改进。

---

> 📦 **项目地址**：[github.com/QATFactory/QATFactory](https://github.com/QATFactory/QATFactory)  
> 💡 **一句话总结**：QATFactory 实现了“一次训练，随处部署”的低精度 LLM 训练闭环，在性能、效率、兼容性之间取得良好平衡。

</details>

---

### 2. [MegaFlux: Skew-Resilient MoE Megakernels via Pipelined Expert Replication](https://arxiv.org/abs/2610.00671)

**Authors**: Jianzhu Yao, Siva Kumar Sastry Hari, Vignesh Balaji, Sana Damani, Insu Jang, Pramod Viswanath, Christos Kozyrakis  
**Category**: cs.DC  
**Published**: 2026-10-02  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2610.00671v1  

#### Abstract
Mixture-of-experts (MoE) megakernels fuse expert-parallel communication with expert computation. However, under fixed expert placement, routing skew creates GPU stragglers: overloaded GPUs determine layer latency while others sit idle. Replicating hot experts can shift work to underloaded GPUs, but ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：MegaFlux: Skew-Resilient MoE Megakernels via Pipelined Expert Replication**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在基于 **Mixture-of-Experts (MoE)** 的大语言模型中，采用 **Expert Parallelism (EP)** 将专家分布在多个 GPU 上时，由于输入依赖的路由（routing）存在显著的 **负载倾斜（routing skew）**，导致部分 GPU 成为“straggler”（即处理热点专家的 GPU 负载过重），而其他 GPU 处于空闲状态，从而严重影响整体层延迟。

尽管现有的 **MoE megakernel** 技术通过融合通信与计算实现了高效的固定专家放置执行，但它们无法动态适应路由倾斜。虽然已有工作（如 UltraEP）尝试通过运行时复制热点专家来缓解负载不均，但引入了额外的开销——**副本权重传输** 和 **训练时梯度归约（gradient reduction）**，这些操作通常作为独立阶段执行，破坏了原有的高效流水线。

### **提出了什么新方法或新思路**
本文提出 **MegaFlux**，一种支持 **运行时专家复制（runtime expert replication）** 并将其完全集成到持久化 MoE megakernel 中的系统。其核心思想是：

- **动态决策复制策略**：在每层执行前，由一个轻量级的 **on-device planner** 根据当前路由分布，联合决定哪些专家需要复制、复制位置以及 token block 的分配。
- **流水线化副本操作**：将副本所需的 **权重传输** 和 **梯度归约** 完全嵌入到 forward 和 backward megakernel 的 tile 级调度中，实现与专家计算的深度重叠，而非作为前后独立阶段。

具体创新包括：
- **块粒度（block-granular）复制规划器**：在满足每个 GPU 副本容量限制的前提下，优化负载均衡。
- **前向流水线机制**：利用 FC1 和 FC2 对权重的不同依赖关系，在 FC1 执行的同时并行传输 FC2 权重，使副本能尽早开始计算。
- **后向流水线机制**：优先生成被复制专家的梯度，并在所有参与 GPU 完成对应 tile 后立即启动归约，同时继续其他 tile 的计算。

### **相比现有方法的优势**
| 维度 | MegaFlux | 现有方法（如 UltraEP） |
|------|---------|------------------------|
| **复制时机** | 运行时、按需动态决策 | 静态或周期性调整 |
| **通信集成** | 权重传输/梯度归约与计算流水线化 | 作为独立阶段串行执行 |
| **效率** | 最大程度隐藏副本开销 | 显著增加端到端延迟 |
| **兼容性** | 可扩展至 TensorRT-LLM 等主流框架 | 多为专用系统 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验结合了真实工作负载与合成负载：
- **真实路由分布**：
  - **Forward**: 来自 **Qwen3-30B-A3B** 在 PG19 和 InfiniteBench retrieval 上的服务轨迹。
  - **Backward**: 来自 **OLMoE-1B-7B** 在 DCLM、OpenWebMath 等数据集上的训练轨迹。
- **合成控制变量**：
  - **Balanced routing**（K=1）
  - **Zipf 分布**（K≈3 或 6），用于模拟不同程度的专家热度倾斜。

### **实验设置**
- **硬件平台**：8 块 **NVIDIA B200 GPU**，构成单个 NVLink 域（EP8）。
- **模型配置**：
  - 隐藏维度 7168，中间维度 2048，top-8 路由。
  - 专家数 $ E \in \{64, 128, 256\} $
  - 每 GPU 输入 token 数 $ M \in \{1K, 2K, ..., 64K\} $
  - 每 GPU 支持最多 $ s = E/(4P) $ 个副本（即 25% 冗余容量）
- **量化模式**：
  - Forward: **MXFP4/MXFP8**
  - Backward: **MXFP8**（FP32 累加，BF16 输出）

### **评估指标**
- **主指标**：每层最大 GPU 延迟（rank-maximum latency）
- **速度提升**：几何平均加速比（geometric-mean speedup）
- **消融分析**：分离阶段 vs 流水线执行的延迟差异
- **计划开销**：planner 执行时间占比
- **负载均衡效果**：最忙 GPU 的超额任务减少比例

### **基线方法对比**
- **Fixed Placement**：相同 megakernel 下无复制的基础版本（主要对比对象）
- **Megatron-Core MoE**
- **UltraEP**
- **Mixture-of-Kittens (MoK)**

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 方向 | 几何平均加速比 | 峰值加速比 |
|------|----------------|------------|
| **Forward** | **1.45×** | **2.14×** |
| **Backward** | **1.28×** | **2.64×** |

> 在 147 种不同配置下测试，涵盖多种专家数量、token 规模和路由分布。

### **与基线方法的对比结果**
#### **vs Fixed Placement**
- 在高度倾斜场景（real-high, real-extreme）下，固定放置因负载不均导致 **7.3–41.0% 的额外延迟**。
- MegaFlux 有效消除 straggler，将最忙 GPU 的超额任务减少了：
  - **Forward**: **99.6%**
  - **Backward**: **98.2%**

#### **vs 其他 MoE 系统（BF16 设置下）**
| 对比项 | Forward 加速比 | Backward 加速比 |
|-------|----------------|------------------|
| **vs Megatron-Core** | 2.31× | 1.53× |
| **vs UltraEP** | 1.74× | 2.35× |
| **vs MoK** | 1.84× | 1.37× |
> （经 workload-adjusted 后仍保持 1.70× / 1.26×）

### **消融实验结果**
#### **分离阶段 vs 流水线执行**
- 若仅进行复制但以分离阶段执行（先传权重再计算），可降低延迟最多：
  - Forward: **42.4%**
  - Backward: **51.2%**
- **加入流水线后进一步增益**：
  - Forward: **额外 +13.2%** 延迟降低
  - Backward: **额外 +26.7%** 延迟降低

#### **流水线隐藏成本比例**
| 场景 | 隐藏成本比例 |
|------|--------------|
| **Forward 权重传输** | **56–76%** |
| **Backward 权重传输 + 梯度归约** | **91–100%** |

> 表明流水线几乎完全掩盖了副本带来的通信开销。

#### **端到端推理性能（集成至 vLLM）**
在 **DeepSeek-V4-Pro** 上进行 prefill 阶段测试，**中位数端到端加速比为 1.13–1.26×**。

| Batch Size | 16K Chunk | 32K Chunk |
|-----------|-----------|-----------|
| 8         | 1.128×    | 1.133×    |
| 16        | 1.227×    | 1.238×    |
| 32        | 1.165×    | 1.260×    |

---

## **4. 关键结论和发现**

### **主要发现**
1. **路由倾斜对固定放置 megakernel 影响显著**：即使已有通信-计算流水线，负载不均仍造成高达 41% 的延迟增长。
2. **动态专家复制必须与 megakernel 深度集成**：简单地在前后添加副本操作反而可能恶化性能；只有通过 **tile-level readiness 控制** 和 **资源复用** 才能真正受益。
3. **流水线可极大隐藏副本开销**：MegaFlux 成功将副本权重传输和梯度归约的代价几乎完全隐藏，尤其在 backward 中接近 100%。
4. **收益随 workload size 增大而增强**：大 batch 更容易摊销 planner 和副本操作的固定开销。
5. **planner 开销极低**：在 1K tokens 时占 ~9%，但在 128K 时降至 **0.03%**，适合在线部署。

### **方法的局限性**
- **仅限单节点（single NVLink domain）**：未考虑跨节点拓扑感知的复制策略。
- **小规模 backward 工作负载可能出现退化**（最低至 0.91×），说明需设计运行时门控机制判断是否启用复制。
- **planning 策略简化**：使用启发式惩罚系数（λ=12），尚未实现基于实际测量的自适应 cost model。
- **未改变原始路由输出**：虽保证质量不变，但也失去了通过 drop 或 reroute 进一步优化的可能性。

### **未来工作方向**
- 设计 **runtime gating mechanism**，仅当预测收益大于开销时才启用复制。
- 扩展至 **multi-node topology-aware replication**。
- 构建更精确的 **execution cost model**，实现自适应 λ 参数选择。
- 探索 **gradient compression** 或 **partial reduction** 以进一步降低归约开销。
- 结合 **predictive routing modeling** 实现提前预取与复制。

--- 

> ✅ **总结一句话**：  
> **MegaFlux 证明了持久化 MoE megakernel 不必受限于固定专家布局——通过运行时复制与深度流水线集成，可在不牺牲质量的前提下显著提升负载均衡能力，实现最高 2.64× 的层级加速。**

</details>

---

### 3. [RapidMoE: Exploiting Cross-Asymmetry via Adaptive Residual Offloading for Large-Scale MoE Inference](https://arxiv.org/abs/2610.01265)

**Authors**: Wenxun Wang, Likai Ma, Zongle Huang, Chen Tang, Yongpan Liu  
**Category**: cs.DC  
**Published**: 2026-10-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.01265v1  

#### Abstract
The widespread adoption of Mixture-of-Experts (MoE) has created a growing need for deployment on heterogeneous platforms. However, it exposes a fundamental mismatch between the algorithmic demands of large-scale MoE and the disparate characteristics of hardware.Existing CPU-GPU hybrid inference syst...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RAPIDMoE: Exploiting Cross-Asymmetry via Adaptive Residual Offloading for Large-Scale MoE Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大规模 **Mixture-of-Experts (MoE)** 模型在异构平台（如 CPU-GPU）上部署面临严重的**算法-硬件不匹配**问题：
- **GPU 显存有限**，无法容纳全部专家权重，必须将部分专家卸载到 CPU 内存；
- 现有 **expert-wise offloading** 方法存在两大瓶颈：
  - **PCIe 带宽瓶颈**：频繁加载专家导致高延迟；
  - **计算资源利用率低**：要么 GPU 等待 I/O，要么 CPU 成为计算瓶颈。

这导致系统难以满足严格的推理延迟要求（如 TPOT < 100ms），尤其在参数规模扩大时更为严重。

---

### 提出的新方法与创新思路
RAPIDMoE 提出通过**跨不对称性（Cross-Asymmetry）** 来打破算法与硬件之间的错配，并引入以下核心机制：

#### ✅ **1. Cross-Asymmetry 的识别与利用**
- **算法层面的非对称性**：MoE 路由中只有少数“关键专家”（critical experts）对模型精度至关重要，其余“非关键专家”可容忍低精度。
- **硬件层面的非对称性**：GPU 高算力但显存小；CPU 内存大但算力弱。
- **对齐策略**：让 CPU 处理存储密集型的关键专家（需高精度），GPU 处理计算密集型的非关键专家（可用低精度）。

#### ✅ **2. 统一多级重要性仲裁（UMIA, Unified Multi-Level Importance Arbitration）**
- 动态判断每个 token、每层、每个推理阶段（prefill/decode）下的关键专家集合；
- 融合三个维度的重要性信号：
  - **局部（Local）**：路由得分 `g(x)`
  - **空间（Spatial）**：不同网络层对量化更敏感（浅层更重要）
  - **时间（Temporal）**：prefill 阶段比 decode 更需要保持精度（影响 KV Cache 质量）

#### ✅ **3. 残差拆分框架（RESplit, Residual-Split Framework）**
实现从 **expert-level offloading → bit-level offloading** 的范式转变：
- 将专家权重分解为两部分：
  - **量化主干 $W^Q$**（如 INT2）保留在 GPU 上用于非关键专家；
  - **残差 $W^R = W^{FP16} - W^Q$** 卸载至 CPU，仅用于关键专家的精度补偿。
- 支持并行执行：GPU 执行 $XW^Q$，CPU 执行 $XW^R$，最后合并输出。

#### ✅ **4. 细粒度并行调度（RESplit Expert Parallelism）**
- 利用 CUDA Stream 实现 CPU 和 GPU 并行处理：
  - GPU 先处理关键专家的 gate/up proj；
  - 同步后，GPU 并发处理所有非关键专家；
  - CPU 在后台完成关键专家的残差计算；
- 极大减少 GPU bubble，提升整体利用率。

---

### 相比现有方法的优势
| 方面 | 现有方法（如 MoE-APEX, HybriMoE） | RAPIDMoE |
|------|-------------------------------|----------|
| **卸载粒度** | Expert-level | Bit-level（残差级） |
| **存储效率** | 双份权重（高低精度都存）→ 存储冗余 | 仅存一份量化 + 残差 → 减少 ~50% DRAM 占用 |
| **动态适应性** | 固定缓存策略或静态精度分配 | 运行时自适应调整关键集大小 $r$ |
| **硬件利用率** | GPU 利用率常低于 30% | 最高达 100%，显著降低空转 |
| **延迟控制** | 难以满足固定 SLO（如 100ms TPOT） | 主动优化至准确率-延迟帕累托前沿 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **准确性评估基准**：
  - **MMLU-Pro**：通用知识理解
  - **MATH-500 / AIME-24**：数学推理能力
  - **HumanEval / EvalPlus**：代码生成能力
  - **WikiText-2**：用于 UMIA 参数校准
- **性能测试工作负载**：
  - **ShareGPT** 请求流：模拟真实服务场景
  - 输入长度：256–4096 tokens（prefill），默认 512-token prompt（decode）

---

### 实验设置
| 项目 | 设置详情 |
|------|---------|
| **模型** | DeepSeek-V3 (671B), DeepSeek-R1 (671B), Qwen3-235B-A22B |
| **平台** | P1: 2×A800-80G + 512GB DDR4；P2: 7×RTX4090 + 512GB DDR4 |
| **量化格式** | $W^Q$: IQ1_M_R4 (~1.75bit)，$W^R$: Q2_K_R4 (~2.5bit)，合计 ~4.25bit |
| **批大小** | Decode: 1–4；Prefill: 1–1024 |
| **并发数** | Serving 测试设为 4，请求速率 0.2 rps |

---

### 评估指标
| 类别 | 指标 |
|------|------|
| **吞吐量** | Prefill / Decode 吞吐（tokens/s） |
| **延迟** | TTFT（Time to First Token），TPOT（Time Per Output Token），含 p50/p99 |
| **SLO 达成率** | 是否满足 TTFT ≤ 5s 且 TPOT ≤ 100ms |
| **内存占用** | GPU HBM 与 CPU DRAM 峰值使用量 |
| **能效** | tokens/s/W（间接体现） |

---

### 基线方法对比
- **MoE-APEX-G/C**：GPU-centric / cooperative mode，支持 adaptive precision
- **HybriMoE**：CPU-centric，动态调度 + 缓存管理
- **KTransformers**：主流 CPU-GPU 混合推理框架
- **llama.cpp**：轻量级 C/C++ 推理引擎，支持 layer-wise offloading

所有基线均采用相同量化配置（Q4_K_M）进行公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 8 & Figure 10–11）

| 场景 | 方法 | Decode Throughput (tokens/s) | Prefill Throughput (tokens/s) |
|------|-------|-----------------------------|------------------------------|
| DeepSeek-R1 (C11) | KTransformers | 8.0 | 62.8 |
| | **RAPIDMoE** | **21.8 (+2.7×)** | **150.2 (+2.4×)** |
| DeepSeek-V3 (C12) | KTransformers | 7.5 | 61.3 |
| | **RAPIDMoE** | **20.5 (+2.7×)** | **138.9 (+2.3×)** |
| Qwen3 (C13) | KTransformers | 10.3 | 102.5 |
| | **RAPIDMoE** | **18.7 (+1.8×)** | **182.8 (+1.8×)** |

> 💡 **最高达 3.5× 解码加速，2.1× 预填充加速**

---

### 与基线方法的对比结果
- **解码阶段**：
  - RAPIDMoE 在所有配置下均大幅领先，平均提速 **2.1–3.5×**；
  - MoE-APEX-G 表现最差（受 PCIe 加载拖累）；
  - CPU-centric 方法（HybriMoE, KTrans.）虽避免 I/O，但 GPU 利用率不足 20%。
- **预填充阶段**：
  - 提速约 **2×**，略低于解码增益；
  - 因 prefill 本身是 compute-bound，残差卸载优势稍弱；
  - 但在长上下文（>8K）中可通过流水线隐藏传输开销。

---

### 消融实验结果（Ablation Study）

#### 🔹 性能逐项拆解（Figure 12a）
| 组件添加 | Prefill Speedup | Decode Speedup |
|--------|------------------|----------------|
| Baseline (KTrans.) | 1.0× | 1.0× |
| + Split Routing (静态 r=3) | 1.2× | 1.3× |
| + UMIA（动态仲裁） | 1.4×↑ | 1.4×↑ |
| + RESplit（完整方案） | **1.39×↑** | **1.33×↑** |

> ✅ 三者协同带来累计 **~3.5×** 整体加速

#### 🔹 UMIA 有效性验证（Figure 13 & Table 11）
- 固定跳过非关键专家（Expert Skipping）会导致准确率崩溃（AIME 下降超 20%）；
- 固定保留（Preserving）虽稳定但仍次优；
- **UMIA 动态调节 r** 可在相近准确率下获得更高吞吐，逼近帕累托最优边界。

#### 🔹 内存效率对比（Table 10）
| 方法 | GPU HBM 峰值 | CPU DRAM 峰值 |
|------|--------------|---------------|
| MoE-APEX | 149 GB | **471 GB** |
| KTransformers | 150 GB | 385 GB |
| **RAPIDMoE** | **149 GB** | **240 GB** |

> 📉 **DRAM 使用减少超过 50%**，得益于无需双份权重存储

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Cross-Asymmetry 是解决 MoE 异构推理瓶颈的关键**：
   - MoE 路由中的稀疏性和重要性偏斜天然适配 CPU-GPU 的资源差异。
2. ✅ **bit-level offloading 比 expert-level 更高效**：
   - RESplit 实现了存储与计算的解耦，从根本上缓解了 I/O 与计算瓶颈。
3. ✅ **运行时动态仲裁优于静态策略**：
   - UMIA 能根据输入、层级、阶段自动调节关键专家数量，在精度与延迟间取得最佳平衡。
4. ✅ **细粒度并行极大提升 GPU 利用率**：
   - RESplit parallelism 将 GPU 利用率从 <30% 提升至接近 100%，消除“气泡”。

---

### 方法的局限性
- **依赖量化兼容性**：需要支持残差量化与反量化流程，对某些特殊架构可能适配成本较高；
- **适用于低并发场景**：高 batch size 会增强 GPU 利用，削弱 CPU 残差修正的价值；
- **校准开销一次性但不可忽略**：UMIA 需约 5 小时离线搜索最优参数（可在部署前完成）；
- **目前验证于 PCIe 平台**：NVLink 或 UCIe 等高速互连环境下收益可能变化。

---

### 未来工作方向
1. **扩展至更多硬件组合**：如 AMD GPU + ROCm、Apple Silicon（Metal）、国产 AI 芯片；
2. **结合其他压缩技术**：如 MoE pruning + RESplit + quantization 联合优化；
3. **支持动态 top-k 路由**：当前假设 k 固定，未来可探索动态稀疏激活；
4. **应用于训练阶段**：将 residual offloading 思想拓展至 MoE 训练中的梯度同步；
5. **构建自动化编译器支持**：实现 UMIA + RESplit 的端到端自动插入与调优。

---

> ✅ **总结一句话**：  
> RAPIDMoE 通过识别并利用 **MoE 算法与异构硬件之间的 Cross-Asymmetry**，提出 **UMIA + RESplit** 联合机制，实现了 **bit-level 自适应卸载**，在几乎不损失准确率的前提下，达成 **最高 3.5× 的推理加速** 与 **显著内存节省**，为大规模 MoE 模型在单节点工作站上的高效部署提供了新范式。

</details>

---

### 4. [Fork-dLLM: Avoiding the Flexibility Trap in Diffusion Language Models](https://arxiv.org/abs/2609.39859)

**Authors**: Stipe Frkovi\'c, Metod Jazbec, Christian A. Naesseth  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.39859v1  

#### Abstract
Masked diffusion language models (dLLMs) have shown strong potential for faster inference through parallel token generation when combined with confidence-based samplers. However, recent work has shown that such methods can defer unmasking high-entropy fork positions at which multiple plausible conti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Fork-dLLM: Avoiding the Flexibility Trap in Diffusion Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **灵活性陷阱（Flexibility Trap）**：现有的 confidence-based sampling 方法（如 Fast-dLLM）在 masked diffusion language models (dLLMs) 中虽然能实现并行解码以加速推理，但倾向于推迟对高熵“分叉位置”（fork positions）的解码。这些位置是决定后续推理路径的关键节点。
- 这种延迟导致生成多样性下降，表现为 **pass@k 缩放性能差**，并且限制了强化学习（RL）后训练的有效性。
- 现有解决方案（如 AR sampling）虽能恢复多样性，但牺牲了并行性，推理效率大幅降低。

### **提出的新方法与思路**
- **Fork-dLLM**：一种混合采样器，在绝大多数步骤中保留 confidence-based 并行解码，仅在“回退步骤”（fallback steps，即无 token 超过置信度阈值时）切换为 **自回归式（AR-style）顺序解码**，即解码最左侧的 masked token。
- **ForkGRPO**：将上述思想扩展到 RL 后训练，仅在 Fork-dLLM rollout 的 fallback steps 上应用 GRPO 目标函数，从而在保持精确策略似然比的同时显著减少 rollout 和优化成本。

### **相比现有方法的优势**
| 方法 | 多样性（pass@k） | 推理效率（NFE） | 是否需要训练 | 是否近似似然 |
|------|------------------|------------------|---------------|----------------|
| AR Sampling | ✅ 强 | ❌ 差（串行） | ❌ 否 | ✅ 精确 |
| Fast-dLLM | ❌ 弱（灵活性陷阱） | ✅ 强（并行） | ❌ 否 | ✅ 精确 |
| JustGRPO (AR) | ✅ 强 | ❌ 差 | ✅ 是 | ✅ 精确 |
| **Fork-dLLM** | ✅ **强（匹配 AR）** | ✅ **2–3× 更高效** | ❌ **否** | ✅ 精确 |
| **ForkGRPO** | ✅ **优于或匹配 AR 基线** | ✅ **训练成本更低** | ✅ 是 | ✅ **精确（仅在 fallback）** |

> ✅ **核心优势**：无需放弃并行性即可避免灵活性陷阱，实现了 **多样性与效率的双赢**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **数学推理**：
  - `GSM8K`（小学数学题）
  - `MATH-500`（高中竞赛级数学题）
- **代码生成**：
  - `HumanEval`
  - `MBPP`

模型基础：
- `LLaDA-8B-Instruct`
- `Dream-v0-Instruct-7B`

### **实验设置与评估指标**
#### **采样阶段（Fork-dLLM）**
- **Block Length (BL)**: 32
- **Generation Length (L)**: 256
- **评估指标**：
  - `pass@k`：至少一个样本通过测试的比例
  - `pass@NFE`：考虑推理成本的性价比指标（每单位 NFE 的通过率）

#### **RL 后训练（ForkGRPO）**
- **Group Size (G)**: 16
- **Prompts per Step**: 16
- **Optimization**:
  - 使用 LoRA (`r=128`, `α=64`)
  - 冻结主干，仅训练 adapter
  - AdamW 优化器，学习率 `2e-5`
- **训练预算**：4×H100 GPU × 12 小时
- **奖励函数**：
  - 数学：答案正确性（0/1）
  - 代码：通过的单元测试比例

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **采样器** | AR, Fast-dLLM, TCT |
| **RL 方法** | JustGRPO, JustGRPO-Fast, FastGRPO（本文引入） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **Fork-dLLM 采样性能（LLaDA-8B-Instruct）**
| 方法 | GSM8K Pass@64 | NFE |
|------|----------------|-----|
| AR | 99.0% | 227.9 |
| Fast-dLLM | 97.3% | **58.3** |
| **Fork-dLLM** | **98.3%** | **67.8** |

> ✅ **Fork-dLLM 在 pass@k 上接近 AR，NFE 仅为 AR 的 ~1/3**

#### **ForkGRPO 训练效率与效果**
- **训练步数对比**（相同预算下）：
  - ForkGRPO 完成 **3.4× 更多训练步**（GSM8K 上）
- **下游准确率提升**（vs. Base Model）：
  - GSM8K: **+6.7pt** (ForkGRPO) vs. +2.1pt (JustGRPO-Fast)
  - HumanEval: **+5.5pt** vs. +0.6pt
- **计算效率**：
  - 达到相似性能时，ForkGRPO 比 JustGRPO **节省 8× H100 小时**

### **与基线方法的对比结果**
- **vs. AR sampling**：
  - pass@k 性能相当，但推理快 **2–3×**
- **vs. Fast-dLLM/TCT**：
  - 在保持低 NFE 的同时，pass@k 显著更优
- **vs. JustGRPO-Fast**：
  - 相同训练预算下，ForkGRPO 准确率更高，且解码所需 NFE 更少
- **vs. JustGRPO**：
  - 尽管训练资源仅为 **1/8**（4×H100×12h vs. 16×H100×24h），ForkGRPO 在 MATH-500 上达到 **~40%** 准确率，优于 JustGRPO 的 37–38%

### **消融实验结果**
- **平行步骤 vs. 回退步骤的作用**：
  - 将平行步骤改为贪婪解码（T=0）→ 对 pass@NFE 影响极小
  - 将回退步骤改为随机/最低置信度选择 → **pass@NFE 显著下降**
  > 🔍 **结论**：多样性增益主要来自 **AR-style fallback**，而非并行步骤的随机性
- **Fallback 步骤分析**：
  - 占总 token 数约 **1/8**
  - 贡献了近 **45–55% 的总熵**
  - 提交的 token 多为逻辑连接词（如 "Since", "First", "Thus"），符合“分叉点”特征

---

## **4. 关键结论和发现**

### **主要发现**
1. **灵活性陷阱可被精准定位**：confidence-based sampler 的 fallback 步骤天然对应高熵分叉位置。
2. **局部 AR 化足以恢复多样性**：只需在 fallback 步骤采用 AR-style 解码，即可匹配全 AR 的 pass@k 表现。
3. **ForkGRPO 实现高效 RL 训练**：仅在 fallback 步骤优化，既能保持精确似然比，又能大幅降低训练成本。
4. **AR rollouts 非必要**：在相同计算预算下，ForkGRPO 明显优于 AR-based 基线，证明 **并行性与多样性可兼得**。

### **方法的局限性**
- 所有超参数在 HumanEval 上调优后直接迁移到其他任务，可能未达最优。
- GRPO 实验仅在 LLaDA-8B 上进行，未验证于更大模型或全量微调。
- 未重新训练 JustGRPO，而是使用其公开 checkpoint，可能存在配置差异。
- FastGRPO 出现训练崩溃现象，机制尚不完全清楚。

### **未来工作方向**
- 在更多模型（如 Dream, DiffusionGemma）上验证 Fork-dLLM/ForkGRPO。
- 探索 ForkGRPO 在其他 RL 方法（如 PPO, DPO 变体）中的应用。
- 研究为何 Fork-dLLM 的 AR-style fallback 能稳定训练，而 confidence-based fallback 会崩溃。
- 将 AR-style fallback 应用于其他 confidence-based sampler（如 EB, TCT）。
- 扩展至 uniform discrete dLLMs 架构。

---

> 📌 **一句话总结**：  
> **Fork-dLLM 通过“在不确定性时刻采取 AR 顺序”这一简单修改，成功避开了灵活性陷阱，在几乎不增加成本的前提下，实现了与全 AR 相当的生成多样性，为 dLLMs 的高效高质生成提供了新范式。**

</details>

---

### 5. [Accelerated Algorithm for Sparse Regularized Partial Optimal Transport](https://arxiv.org/abs/2609.40075)

**Authors**: Khoa Nguyen, Dung T. Nguyen, Thong Huynh, Hoang-Hiep Nguyen-Mau, Anh Nguyen, Minh Ngoc Dinh, Juho Kannala  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.40075v1  

#### Abstract
Partial Optimal Transport (POT) extends the classical optimal transport problem by relaxing the strict mass conservation constraint, enabling its use in a wide range of real-world applications. In many of these settings, sparse transport plans are preferred for their interpretability and computation...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerated Algorithm for Sparse Regularized Partial Optimal Transport

## 1. 论文的主要贡献和创新点

### 解决的问题
- **Partial Optimal Transport (POT)** 虽然放宽了经典 OT 中质量守恒的严格约束，适用于不平衡分布场景（如存在离群点、缺失内容），但其求解仍面临计算复杂度高、可扩展性差的问题。
- 现有主流方法多采用 **entropic regularization**（如 Sinkhorn 算法）以加速求解，但会导致 **dense transport plan**（稠密传输计划），缺乏可解释性，且在需要稀疏性的应用中表现不佳（如颜色迁移、领域自适应）。
- 尽管 **quadratic regularization** 和 **elastic net regularization** 等平滑强凸正则化器能诱导出更稀疏的传输方案，但针对这类正则化 POT 问题的高效一阶优化算法研究尚不充分。

### 提出的新方法与新思路
- **提出了一种基于外罚函数（exterior penalty method）的优化框架**，用于求解带平滑强凸正则化的 RPOT（Regularized Partial Optimal Transport）问题。
  - 将原本带有边际不等式约束 $X\mathbf{1}_n \leq r$ 和 $X^\top\mathbf{1}_n \leq c$ 的约束优化问题，转化为一个无约束的惩罚问题（P-RPOT），通过引入二次外罚项来软化这些约束：
    $$
    P(X,\alpha) = \frac{\alpha}{2} \left( \|[\min\{0, r - X\mathbf{1}_n\}]^2\|_1 + \|[\min\{0, c - X^\top\mathbf{1}_n\}]^2\|_1 \right)
    $$
  - 该框架支持广泛的正则化器，包括 **quadratic** 和 **elastic net** 类型。
- **设计了一个专用的加速一阶算法 PNAG-POT**（Proximal Nesterov’s Accelerated Gradient for POT）：
  - 利用 P-RPOT 目标函数的 **smoothness** 和 **strong convexity** 性质，结合 **Nesterov 加速梯度下降** 进行优化。
  - 在每次迭代后使用高效的投影算法将解投影回单纯形约束 $S = \{X \mid \mathbf{1}^\top X \mathbf{1} = s\}$。
  - 最终通过 **ROUND-POT** 后处理步骤恢复精确可行性。

### 相比现有方法的优势
- **更高的稀疏性**：相比 entropic 正则化方法（如 APDAGD），本方法生成的 transport plan 更稀疏，有利于模型解释性和下游任务性能。
- **更低的运输成本**：在多个任务上实现了更低的最终运输成本（final cost）。
- **更快的收敛速度与效率**：相比通用凸优化求解器（如 SCS via CVXPY），迭代次数和运行时间显著减少，具备更好的可扩展性。
- **理论保证**：提供了严格的理论分析，证明当罚参数 $\alpha$ 足够大时，P-RPOT 的解可以逼近原始 RPOT 的最优解，并给出了目标差距和约束违反的显式界。

---

## 2. 核心实验方法和设置

### 使用的数据集与任务
实验在三个典型的机器学习任务上进行验证：
1. **Color Transfer**（颜色迁移）
   - 数据：RGB 图像的颜色直方图（n=1024 bins）
   - 成本矩阵：源与目标直方图 bin 中心之间的平方欧氏距离
2. **Point Cloud Registration**（点云配准）
   - 数据：3D 点云（n=500 points），可视化为 2D
   - 成本矩阵：点之间的平方欧氏距离
3. **Domain Adaptation**（领域自适应）
   - 数据：scikit-learn 的 `moons` 数据集（n=300 samples）
   - 成本矩阵：归一化的成对欧氏距离

### 实验设置
- **传输质量**（transported mass）固定为 $s = 0.9 \times \min\{\|r\|_1, \|c\|_1\}$，即传输 90% 的源质量。
- **正则化权重** $\eta$ 经验调优，统一设为 $10^{-3}$。
- **罚参数** $\alpha$ 针对不同任务调整（如 5500 或 55000），确保有效惩罚与数值稳定性。
- 所有实验初始化 $X$ 为随机矩阵。

### 评估指标
- **Final Cost**：最终的运输成本 $(C, X)$
- **Sparsity**：通过阈值（$<10^{-8}$）计算非零元素比例
- **Iterations**：达到收敛所需的迭代次数
- **Wall-clock Time (s)**：实际运行时间（秒）
- **Adaptation Accuracy**（仅领域自适应）：在目标域上的 SVM 分类准确率

### 基线方法对比
- **CVXPY + SCS**：使用通用一阶锥求解器（Splitting Conic Solver）求解 QPOT 和 ENPOT，作为精度基准。
- **APDAGD** [24]：一种高效的 entropic-regularized POT 求解器，用于比较最终成本和稀疏性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3 & 4）

| Task | Method | Final Cost | Sparsity | Iterations | Time (s) |
|------|--------|----------|---------|------------|----------|
| **Color Transfer** | CVXPY | 0.028 | 0.9848 | 50,000 | 6007.00 |
| | APDAGD | 0.044 | 0.9805 | — | — |
| | **PNAG-POT** | **0.027** | **0.9958** | **1,800** | **673.50** |
| **Point Cloud Reg.** | CVXPY | 0.015 | 0.4351 | 11,650 | 820.14 |
| | APDAGD | 0.012 | 0.8424 | — | — |
| | **PNAG-POT** | **0.0084** | **0.9229** | **2,940** | **198.12** |
| **Domain Adaptation** | CVXPY | 0.036 | 0.4532 | 8,500 | 210.09 |
| | APDAGD | 0.037 | 0.6937 | — | — |
| | **PNAG-POT** | **0.036** | **0.9180** | **1,420** | **8.12** |

> 注：Elastic-Net 正则化下趋势一致，PNAG-POT 在所有任务上均取得 **更低或相当的成本**、**更高的稀疏性**、**更少的迭代次数** 和 **更短的运行时间**。

### 与基线方法的对比结果
- **vs CVXPY**：
  - PNAG-POT 在 **所有任务上达到了相同甚至更低的最终成本**。
  - **稀疏性显著提升**（例如从 0.45 → 0.92）。
  - **运行时间大幅缩短**（最高达两个数量级，如从 210s → 8s）。
  - **迭代次数显著减少**（从数千/万次降至千次以内）。
- **vs APDAGD**：
  - 在 **color transfer** 和 **point cloud registration** 上，PNAG-POT 实现了 **更低的运输成本**。
  - 在 **domain adaptation** 上，虽然成本相近，但 PNAG-POT 达到了 **最高的分类准确率（78.33% vs 76.41%/78.12%）**，表明高稀疏性未损害泛化能力。

### 消融实验结果
- 论文中未明确列出独立的消融实验表格，但通过以下方式体现了方法组件的有效性：
  - **理论分析**（Theorem 3.2, 4.1）证明了 P-RPOT 框架的正确性和收敛性。
  - **ROUND-POT** 后处理被证明能在小约束违反下恢复可行解，并控制输出与输入的距离（Lemma 4.2）。
  - 不同正则化器（quadratic vs elastic net）下的实验一致性验证了框架的通用性。

---

## 4. 关键结论和发现

### 主要发现
- **外罚框架有效可行**：通过将边际约束转化为惩罚项，可以在保持目标函数光滑性和强凸性的前提下，高效求解 RPOT 问题。
- **PNAG-POT 是一种实用且可扩展的求解器**：它在多个真实世界任务上均优于现有方法，在 **运输成本、稀疏性、收敛速度** 三个方面实现全面领先。
- **稀疏性与高性能可兼得**：高稀疏的 transport plan 并不会牺牲任务性能（如 domain adaptation 准确率），反而可能因去噪效应而提升效果。
- **框架具有通用性**：不仅适用于 quadratic regularization，也成功应用于 elastic net regularization，展示了其灵活性。

### 方法的局限性
- **依赖罚参数 $\alpha$ 的选择**：虽然理论上要求 $\alpha \to \infty$，但实践中需手动调参以平衡收敛速度与数值稳定性。
- **对 Slater 条件的依赖**：理论分析假设 Slater's condition 成立，这在某些极端稀疏或退化情况下可能不满足。
- **内存开销**：仍需存储完整的 $n \times n$ 传输矩阵，在超大规模问题上可能存在内存瓶颈（尽管比 entropic 方法更具优势）。

### 未来工作方向
- 将该方法扩展到其他 Optimal Transport 变体，如 **Unbalanced Optimal Transport (UOT)** 或 **Constrained Optimization Problems**。
- 探索 **low-rank** 或 **sparse approximation** 技术以进一步降低内存和计算复杂度。
- 应用于更多机器学习场景，如 **Few-shot Learning**、**Generative Modeling**、**Time Series Modeling** 等。
- 研究自动调节罚参数 $\alpha$ 的策略，提升方法的易用性。

</details>

---

### 6. [MANET-GNN: Learned Decentralized Optimization of Power Allocation in Multi-Channel MANETs](https://arxiv.org/abs/2609.40170)

**Authors**: Tomer Alter, Nir Shlezinger, Michael Segal  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.40170v1  

#### Abstract
MANETs enable flexible infrastructure-less wireless connectivity in dynamic and resource-constrained environments. As modern MANETs exploit multiple frequency channels and support heterogeneous traffic patterns, decentralized transmit-power allocation becomes increasingly challenging. We develop a u...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MANET-GNN: Learned Decentralized Optimization of Power Allocation in Multi-Channel MANETs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 **Multi-Channel MANETs**（移动自组织网络）面临复杂的资源分配挑战，尤其是在多跳、多信道环境下进行 **去中心化（decentralized）功率分配** 和 **联合路由优化**。传统方法通常局限于单信道系统、特定通信模式（如仅支持单播），或依赖集中式优化，难以适应动态拓扑和异构流量需求。

本文旨在解决以下核心问题：
- 如何在 **无基础设施、动态变化、资源受限** 的 MANET 中实现高效的去中心化功率分配？
- 如何统一处理多种通信框架（如单播、组播、多商品流等）下的联合路由与功率分配？
- 如何在仅有局部、可能含噪的 **CSI（Channel State Information）** 下做出接近全局最优的决策？

---

### 提出了什么新方法或新思路
作者提出 **MANET-GNN** —— 一种基于 **图神经网络（GNN）** 的 **学习型去中心化优化器（learned decentralized optimizer）**，其核心思想是：

- 将功率分配建模为一个统一的约束优化问题，目标是最小消息的最大端到端速率（`min_k R(P; D_k)`）。
- 利用该优化目标作为 **无监督训练损失函数**，直接训练 GNN 学习从局部 CSI 和拓扑信息到可行功率分配策略的映射。
- 设计了一种 **门控消息传递架构（gated message-passing backbone）**，每层对应两次邻居间的消息交换，显式控制延迟（L 轮通信）。
- 引入 **候选路径路由头（candidate-based routing head）**，支持多商品流场景下的完整路径选择，而非独立链路激活。

---

### 相比现有方法的优势

| 优势维度 | MANET-GNN |
|--------|---------|
| **通用性** | 统一支持 F1–F5 五类通信框架（Unicast, Multicast, Multicommodity, Convergecast, Many-to-Many） |
| **去中心化能力** | 每个节点仅需本地 CSI 和 L 轮邻居通信即可完成推理，满足低延迟要求 |
| **泛化性** | 可推广至未见过的网络规模和拓扑结构（见 Fig. 9a） |
| **鲁棒性** | 在训练中引入噪声 CSI，提升对信道估计误差的鲁棒性 |
| **无需标签** | 使用原始优化目标作为无监督损失，无需真实功率分配标签 |

相比传统的启发式算法（如 Widest Path）、集中式优化器（Centralized Optimizer）以及非图结构的 FFN，MANET-GNN 在性能、可扩展性和实用性之间取得了更好平衡。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **合成生成的数据集 D**，包含多个不同拓扑和信道条件的 MANET 实例。
- 拓扑生成方式：随机 Erdős–Rényi 图，节点数 `|V| ∈ {10, 30}`，边概率 `p ∈ {0.1, ..., 0.5}`。
- 信道模型：使用 **QuaDRiGa** 生成城市环境下的频率选择性信道，共 `B = 6` 个子信道。
- 数据划分：
  - **训练集**：2000 个 |V|=10 的拓扑（1600 训练 + 400 验证）
  - **测试集**：500 个独立生成的 |V|=10 测试拓扑，部分实验使用 30 节点拓扑用于泛化测试

---

### 实验设置和评估指标

#### 主要参数
- **消息轮次 L**：默认 L=6（即 3 层 GNN，每层两轮通信）
- **训练配置**：AdamW 优化器，cosine 学习率调度，dropout=0.2，训练 100 epochs
- **噪声 CSI 设置**：通过 LMMSE 估计模拟实际信道估计过程，输入使用估计值，损失仍用真实 CSI 计算

#### 评估指标
- **平均端到端速率（Mean End-to-End Rate）**：主性能指标，单位 bit/s/Hz
- **95% 置信区间**：所有结果均报告置信区间
- **复杂度分析**：计算通信开销、计算复杂度（见 Table II）

---

### 基线方法对比

| 编号 | 方法 | 类型 | 描述 |
|-----|------|------|------|
| B1 | Centralized Optimizer | Centralized | 使用 AdamW 求解原优化问题 (9)，多初始化 |
| B2 | Equal-Split | Heuristic | 功率均匀分配给所有出边和频带 |
| B3 | Centralized Greedy Split (CGS) | Centralized | 选最短路径并平均分配功率 |
| B4 | Distributed Greedy Split (DGR) | Decentralized | 分布式 Bellman-Ford 找最小跳路径 |
| B5 | Centralized Widest Path (CWP) | Centralized | 选瓶颈链路最强的路径 |
| B6 | Distributed Widest Path (DWP) | Decentralized | 分布式 max-min 路由 |
| B7 | Feedforward Network (FFN) | Centralized Learned | 全连接网络，接收全局 CSI 输入 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（SNR = 30 dB 左右时典型表现）

| 场景 | 最佳方法 | MANET-GNN 表现 | 相对差距 |
|------|--------|----------------|----------|
| **Unicast (F1)** | Centralized Optimizer ≈ CGS > MANET-GNN | 接近 CGS，略低于集中式 | < 10% 差距 |
| **Multicast (F2)** | Centralized Optimizer > **MANET-GNN** | 明显优于 CGS/CWP | +15~20% 增益 |
| **Multicommodity (F3)** | Centralized Optimizer > **MANET-GNN** | 远超 Greedy/Widest Path | +25% 以上增益 |
| **Convergecast (F4)** | Centralized Optimizer > **MANET-GNN** | 显著领先其他分布式方法 | +20% 增益 |
| **Many-to-Many (F5)** | Centralized Optimizer > **MANET-GNN** | 成为仅次于中心化的最佳方案 | +30% 增益 |

> ✅ **总体趋势**：随着通信复杂度增加（从 F1 → F5），MANET-GNN 的相对优势越明显。

---

### 与基线方法的对比结果

- **在 Full CSI 下**：
  - MANET-GNN 性能接近 Centralized Optimizer，在多数场景下显著优于所有分布式启发式方法（DWP/DGR）和 Equal-Split。
  - 在 **Multicast 和 Many-to-Many** 场景中，甚至超过 CGS/CWP，说明其能有效协调共享资源冲突。

- **在 Noisy CSI 下**：
  - MANET-GNN 表现出强鲁棒性，性能下降平缓。
  - 启发式方法（如 DWP）因依赖精确 CSI 判断“最强链路”，性能急剧下降。
  - FFN 因缺乏图归纳偏置，在去中心化设置下无法部署。

---

### 消融实验结果（Ablation Studies）

#### （1）消息传递深度（L）的影响（Fig. 9b）
- **L=2, 4**：性能差，因感受野太小，无法捕捉长路径依赖。
- **L=6**：良好性能，适用于中小网络。
- **L=10**：在直径较大的 30 节点网络上达到最佳性能（匹配图直径 ~5.2）。
- **L=14**：性能下降 → 出现 **over-smoothing** 现象，节点表示趋于一致。

> 🔍 结论：存在最优通信轮次，应与网络直径相匹配。

#### （2）跨拓扑泛化能力（Fig. 9a）
- 在 10 节点上训练的模型，直接应用于 **30 节点网络**，仍保持约 80% 的性能。
- 若在 30 节点上重新训练（L=10），性能进一步提升约 20%。

> 🚀 结论：MANET-GNN 具备良好的 **zero-shot size generalization** 能力。

---

## 4. 关键结论和发现

### 论文的主要发现

1. ✅ **统一建模可行性**：可通过一个端到端速率最大化目标统一建模多种 MANET 通信范式（F1–F5）。
2. ✅ **GNN 是理想的去中心化优化器架构**：天然支持局部消息传递、拓扑不变性、参数共享，适合分布式执行。
3. ✅ **无监督学习可行**：以原始优化目标为损失函数，无需标注数据，即可训练高性能策略。
4. ✅ **性能接近集中式上限**：在多种通信场景下，MANET-GNN 实现了与 Centralized Optimizer 接近的性能，远超传统分布式方法。
5. ✅ **具备强鲁棒性与泛化性**：对噪声 CSI 不敏感，并能推广到更大、更复杂的网络拓扑。

---

### 方法的局限性

| 局限性 | 说明 |
|-------|------|
| **固定信道分配** | 当前假设消息在一个传输块内不换信道（no per-hop channel switching），限制了灵活性 |
| **离线候选路径构建** | 路由头依赖预定义的候选路径集合，虽不影响在线复杂度，但在极端稀疏图中可能遗漏最优路径 |
| **单一通信模式训练** | 当前模型针对特定通信框架训练，尚不支持运行时动态切换（如 F1 ↔ F5） |
| **未考虑 QoS 差异化** | 所有流权重相同，未支持加权公平或优先级调度 |

---

### 未来工作方向

1. **支持动态信道切换**：扩展模型以联合优化路由、功率与频带分配（如 OFDMA 场景）。
2. **多任务/模块化设计**：采用 Hypernetwork 或 Mixture-of-Experts 构建 **multi-framework unified model**。
3. **引入 QoS 感知机制**：通过加权效用函数支持差异化服务。
4. **扩展至 MIMO 系统**：联合优化 Beamforming 与功率分配，提升空间复用效率。
5. **空中学习（Over-the-Air Learning）**：研究在真实无线信道上传输 GNN 消息的可行性与鲁棒性。

---

> 💡 **总结一句话**：  
> **MANET-GNN 成功将 GNN 作为“可学习的去中心化优化器”，实现了在多信道 MANET 中高效、鲁棒、泛化的联合路由与功率分配，为未来智能自组织网络提供了新的范式。**

</details>

---

### 7. [HHR: Hierarchical Hash Retrieval for Efficient LLM Generation](https://arxiv.org/abs/2610.01230)

**Authors**: Lianjun Liu, Tiantian Zheng, You Huang, Weiqi Yan, Mingte Qiu, Huazhong Liu, Xiaofeng Zhu, Yunshan Zhong  
**Category**: cs.AI  
**Published**: 2026-10-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.01230v1  

#### Abstract
Efficient long-context inference is essential for large language models (LLMs), yet it poses a severe computational bottleneck. Hash-based retrieval offers an efficient alternative by encoding queries and keys into binary codes and using Hamming distance for key selection. However, this leads to a c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HHR: Hierarchical Hash Retrieval for Efficient LLM Generation 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对 **hash-based sparse attention** 在长上下文 LLM 推理中的一个根本性缺陷：**哈希二值化过程丢失了特征的幅值（magnitude）信息**，导致 **Hamming distance** 与真实的 **Query-Key logits** 之间存在严重错位。

这种错位引发两类检索失败：
- **False Positives**：方向相似但幅值低的 keys 被错误保留（Region III），浪费计算资源。
- **False Negatives**：幅值高但方向不同的 keys 被错误丢弃（Region II），导致重要信息丢失。

### 提出了什么新方法或新思路
提出 **Hierarchical Hash Retrieval (HHR)**，一种**粗粒度到细粒度**的两阶段检索框架，结合两个核心模块：
1. **Geometry-Aware Key Routing (GKR)**  
   - 将 keys 分页（paged），学习一个 **head-wise 正交变换矩阵 $R_h$**，重新分配特征维度上的幅值分布。
   - 利用变换后的 per-dimension 极值（min/max）计算每个 page 的 **最大 logit 上界**。
   - 通过阈值过滤掉上界低于阈值的 page，实现早期剪枝，有效减少 False Positives。

2. **Learned Hash Projection (LHP)**  
   - 在 GKR 保留的候选 keys 上，学习一个 **head-wise 投影矩阵 $W_h$**，将 keys 和 queries 映射到对齐 Hamming distance 与真实 Query-Key 相关性的哈希空间。
   - 使用可微松弛（tanh）训练，结合正交性、去相关性和平衡性正则化，提升二值编码质量。
   - 显著减少 False Negatives，提高高 logit keys 的召回率。

### 相比现有方法的优势
- **更精准的检索**：同时抑制 False Positives 并恢复 False Negatives，显著提升 hash-based sparse attention 的保真度。
- **高效性**：保持哈希检索的高效性（XOR + POPCNT），在 128K 上下文长度下实现高达 **3.30× 解码加速** 和 **2.83× 端到端加速**。
- **通用性强**：在多个主流 LLM（Llama-3.1-8B, Mistral-7B, Qwen3-4B）和基准测试上均取得 SOTA 性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LongBench**：涵盖问答、摘要、少样本学习、代码补全等 16 个英文任务，用于评估长上下文理解能力。
- **RULER**：11 项任务，上下文长度达 32K，测试检索、聚合和问答能力。

### 实验设置和评估指标
- **模型**：Llama-3.1-8B-Instruct, Mistral-7B-Instruct, Qwen3-4B。
- **上下文长度**：16K, 32K, 64K, 128K。
- **Top-K 比例**：统一为 1.5%，与基线公平比较。
- **评估指标**：
  - LongBench：各任务的 F1, ROUGE-L, Accuracy, Code Similarity，报告平均分。
  - RULER：任务级和平均准确率。
  - 效率指标：解码速度提升（decode speedup）、端到端速度提升（end-to-end speedup）。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| Dense Baseline | Full Attention, Oracle (Top-K) |
| Sparse Retrieval | HATA, MagicPIG, Loki |
| KV Cache Compression | StreamingLLM, SnapKV |

所有方法在相同 token 预算下进行比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### LongBench 平均得分（↑ 越高越好）
| Model | HHR (Ours) | Best Baseline | Gain |
|-------|------------|---------------|------|
| Llama-3.1-8B-Instruct | **46.96** | 45.86 (HATA) | **+1.10** |
| Mistral-7B-Instruct | **44.46** | 41.64 (HATA) | **+2.82** |
| Qwen3-4B | **45.66** | 42.97 (HATA) | **+2.69** |

#### RULER 平均准确率（↑ 越高越好）
| Model | HHR (Ours) | Best Baseline | Gain |
|-------|------------|---------------|------|
| Mistral-7B-Instruct | **76.23** | 68.70 (HATA) | **+6.43** |
| Qwen3-4B | **82.71** | 78.69 (HATA) | **+4.02** |

### 与基线方法的对比结果
- HHR 在所有模型和数据集上均优于所有基线方法，尤其在 Mistral 和 Qwen3 上优势显著。
- 在极端稀疏设置（Top-K=2%）下，HHR 仍能保持接近 Full Attention 的性能，验证其强鲁棒性。
- 在 128K 上下文长度下：
  - **解码速度提升：3.30×**
  - **端到端速度提升：2.83×**

### 消融实验结果
- **GKR 和 LHP 各自作用**：
  - 移除 GKR（随机 $R$）：性能下降最多达 **2.52 pts**。
  - 移除 LHP（随机 $W$）：性能下降 **1.50–2.50 pts**。
  - 二者联合训练仅需约 **19–28 分钟**，开销极小。
- **损失函数消融**：
  - 移除 $L_{\text{align}}$ 导致最大性能下降（**-1.53 pts**），说明对齐哈希分数与注意力分布至关重要。
  - 所有正则项（正交、去相关、平衡）均有贡献，共同提升哈希空间质量。

---

## 4. 关键结论和发现

### 主要发现
- **哈希二值化丢失幅值是性能瓶颈**：这是导致 hash-based retrieval 准确性不足的根本原因。
- **几何感知路由（GKR）有效剪枝**：通过学习正交变换优化维度极值，显著提升 page-level 上界的判别力，实现高效粗筛。
- **学习式哈希投影（LHP）提升召回**：学习对齐 Hamming distance 与真实相关性，显著减少高 logit keys 的遗漏。
- **HHR 是互补且高效的**：GKR 和 LHP 分阶段协同工作，在精度和效率之间取得优异平衡。

### 方法的局限性
- 仍与 **Oracle Top-K** 存在一定性能差距，说明仍有改进空间。
- 当前方法独立于 KV 缓存量化等其他压缩技术，未探索联合优化潜力。

### 未来工作方向
- 设计更强的学习目标以进一步优化变换矩阵 $R$ 和投影矩阵 $W$。
- 探索 HHR 与其他压缩技术（如 KV Quantization）的结合，实现更极致的长上下文推理效率。
- 扩展至多模态或视觉语言模型场景。

--- 

> **代码开源**：https://github.com/lianjunl13-sudo/HHR

</details>

---

### 8. [DEdit: Iterative Draft Editing for Speculative Decoding](https://arxiv.org/abs/2609.38510)

**Authors**: Longxuan Yu, Bingsen Chen, Peng Shi, Dongkyu Lee, Yi Xiang, Hideo Kobayashi, Sheng Zhang, Shuaichen Chang, Xing Niu, Zhuoyan Xu, Greg Ver Steeg, Jiarong Jiang  
**Category**: cs.CL  
**Published**: 2026-10-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.38510v1  

#### Abstract
Speculative decoding accelerates autoregressive LLMs by having a lightweight drafter propose tokens that the target model verifies in parallel. Diffusion-based drafters further reduce drafting latency by proposing multiple tokens at once. However, these tokens are predicted independently, so a singl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DEdit: Iterative Draft Editing for Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **speculative decoding** 中，轻量级的 **drafter** 模型并行生成多个候选 token，由目标模型（target）进行验证。然而，现有的 **diffusion-based drafters** 虽然能并行预测多个 token，但由于缺乏 token 间的因果依赖，早期的错误会直接导致后续所有候选被前缀验证（prefix verification）机制丢弃——即使后面的预测可能是正确的。

这种“**一错全废**”的现象浪费了大量潜在有用的预测，限制了 token 接受率（token acceptance）和加速效果。

### 提出的新方法：DEdit
本文提出 **DEdit**，一种基于扩散模型的 **迭代式草案编辑器（iterative draft editor）**，其核心思想是：

- **双向编辑（Bidirectional Editing）**：DEdit 不仅能像传统方法一样生成初始草案（initial proposal），还能通过多轮迭代，利用 **bidirectional attention** 对整个草案进行并行修正。
- **后期预测辅助前期修复**：在编辑过程中，模型可以利用尚未被接受的“未来 token”作为上下文，来修复早期的错误，从而延长最终被接受的前缀长度。

### 相比现有方法的优势
| 方法 | 特点 | 局限性 |
|------|------|--------|
| **DFlash** | 单次并行生成，速度快 | 无自纠错能力，错误传播严重 |
| **Domino** | 引入轻量级 GRU 进行 **causal correction** | 只能用前面的 token 修正后面，无法利用“未来”信息 |
| **DSpark** | 使用 confidence-scheduled 验证 | 仍受限于单向依赖 |
| **DEdit (Ours)** | 支持 **bidirectional editing**，可迭代优化草案 | 增加少量计算成本 |

> ✅ **核心优势**：DEdit 是首个将 **双向上下文** 用于 speculative decoding 草案编辑的方法，能够“**向后看**”，利用正确但未被接受的未来 token 来修复早期错误。

---

## 2. 核心实验方法和设置

### 数据集
在 **7个基准任务** 上评估，涵盖三大类：
- **数学推理（MATH）**：GSM8K、MATH-500、AIME25
- **代码生成（CODE）**：HumanEval、MBPP、LiveCodeBench (LCB)
- **对话（CHAT）**：MT-Bench

训练数据使用约 **80万条** 与目标模型对齐的指令-响应对，来源于：
- Nemotron Post-Training Dataset v2
- CodeAlpaca-20k

### 实验设置
- **目标模型**：Qwen3-4B 和 Qwen3-8B
- **解码方式**：greedy decoding（Tp=0）和 stochastic decoding（Tp=1）
- **草案窗口大小（W）**：
  - 基线方法：W=16
  - DEdit：W=32（并进行 window expansion 训练）
- **编辑轮数（K）**：K=3（即 1 次生成 + 2 次编辑）
- **执行环境**：单张 NVIDIA H100 GPU，batch size=1
- **加速技术**：CUDA Graph 用于 drafting，target verification 使用 eager mode

### 评估指标
- **T（mean token acceptance per round）**：每轮平均接受的 token 数（含额外生成的 target token）
- **End-to-end speedup**：相对于标准自回归生成的端到端加速比

### 基线方法对比
- **DFlash**：块扩散草案器，单次并行生成
- **Domino**：在 DFlash 基础上增加轻量级 GRU 进行因果修正
- **DSpark**：引入 confidence-scheduled 验证和半自回归生成

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Greedy Decoding 下）
| 模型 | 方法 | 平均 T | 平均 Speedup |
|------|------|--------|-------------|
| Qwen3-4B | DEdit P3 | **7.58** | **5.72×** |
| Qwen3-8B | DEdit P3 | **7.75** | **5.97×** |

> 📈 DEdit 在两个模型上均达到 **最高 token 接受率和加速比**。

### 与基线方法对比
- 在 **macro-average T** 上：
  - Qwen3-4B：比最强基线 DSpark **高 9.7%**
  - Qwen3-8B：**高 7.6%**
- 在 **speedup** 上：
  - 比最快基线 Domino 快 **7.0–8.9%**

### 按任务细分表现
- **数学任务（MATH）**：提升最显著，T 提升 **12–18%**
- **代码任务（CODE）**：与 DSpark 表现接近，部分场景略优
- **对话任务（CHAT）**：稳定优于基线

### 消融实验结果（Ablation Study）

#### (1) 编辑轮数与窗口大小的影响
| 方法 | W=16 | W=32 |
|------|------|------|
| DEdit P1（无编辑） | 5.82 | 5.93 |
| DEdit P3（2轮编辑） | 6.87 | **7.58** |

> 🔍 **关键发现**：  
> - 仅扩大窗口（W=16→32）几乎无提升（+1.9%）
> - 结合编辑后，T 提升 **10.3%**（6.87→7.58）
> → **更大的窗口价值在于为编辑提供更丰富的未来上下文，而非更多候选**

#### (2) PROPOSALMIX 与联合训练的作用
在控制实验中（固定 P1 输入）：
| 变体 | 是否联合训练 | 是否 PROPOSALMIX | ΔT |
|------|--------------|------------------|-----|
| Full | √ | √ | — |
| w/o Joint | × | √ | -0.085 |
| w/o ProposalMix | √ | × | **-0.144** |
| w/o Both | × | × | **-0.257** |

> ✅ **结论**：
> - **PROPOSALMIX 贡献更大**：它教会模型“何时保留、何时修改”
> - **联合训练也有帮助**：即使编辑器不参与生成，也能从生成任务中学到有用表示

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **双向编辑有效**：允许模型利用“未来 token”作为上下文修复早期错误，显著提升 token 接受率。
2. ✅ **未来上下文是关键**：消融实验证明，移除 bidirectional attention 后性能下降，尤其在输出高度可预测的任务（如复制、摘要）上损失最大。
3. ✅ **PROPOSALMIX 训练策略成功**：该方法能有效减少“有害编辑”（harmful edits），使模型学会保护正确预测。
4. ✅ **窗口越大，编辑收益越高**：更大的草案窗口不仅提供更多候选，更重要的是为编辑提供了更长的未来上下文。

### 方法的局限性
- **计算开销增加**：每轮编辑需额外一次 drafter forward pass，在某些执行后端可能无法完全掩盖开销。
- **单请求设定**：当前加速结果基于单请求 + CUDA Graph 优化，未考虑生产环境中 **continuous batching** 的影响。
- **约 60–70% 的编辑无收益**：大多数编辑轮次并未改变接受前缀，存在优化空间。

### 未来工作方向
- **动态跳过编辑**：根据草案 confidence 或收敛情况，决定是否跳过某些编辑轮次，降低计算成本。
- **自适应窗口与验证长度**：根据输入难度动态调整 W 和验证范围。
- **跨请求调度优化**：在批量服务中实现 drafting 与 verification 的交错执行。
- **扩展到更大模型**：当前实验限于 8B 规模，未来可探索更大模型上的效果。

---

> 💡 **一句话总结**：  
> **DEdit 通过引入“双向迭代编辑”机制，首次实现了利用未来预测来修复早期错误，突破了 speculative decoding 中“一错全废”的瓶颈，在多个 benchmark 上实现了 state-of-the-art 的加速效果。**

</details>

---

### 9. [Spike-driven Vision-Language-Action Model](https://arxiv.org/abs/2609.39514)

**Authors**: Shuai Wang, Malu Zhang, Mingquan Liu, Weihui Dai, Dehao Zhang, Jieyuan Zhang, Yimeng Shan, Zijian Zhou, Yang Yang  
**Category**: cs.CL  
**Published**: 2026-10-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.39514v1  

#### Abstract
Vision-language-action (VLA) models bridge multimodal understanding and robotic control, advancing the dominant paradigm for embodied intelligence. However, most existing models rely on large Transformers, whose latency and energy costs hinder deployment on resource-constrained platforms. Through sp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Spike-driven Vision-Language-Action Model 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Vision-Language-Action (VLA)** 模型通常基于大规模 Transformer 架构，虽然在机器人控制任务中表现出色，但其**高计算开销、内存占用和延迟**严重限制了在资源受限边缘设备上的部署。此外，传统模型依赖密集的乘加运算（MAC），导致能耗过高。

### 提出的新方法与创新思路
本文提出了首个**完全由脉冲驱动（spike-driven）的端到端可训练 VLA 框架**——**Spike-driven VLA**，结合了**脉冲神经网络（SNN）** 的能效优势与多模态决策能力。其核心组件包括：

- **Spiking Visual Encoder (SVE)** 和 **Spiking Instruction Encoder (SIE)**  
  将视觉观测和语言指令编码为稀疏的脉冲表示，分别捕捉视觉特征和语义信息。

- **Multi-Winner Spike Fusion (MWSF)**  
  引入双向 top-k 赢者通吃（Winner-Take-All, WTA）机制进行跨模态融合，有效抑制背景干扰，实现稀疏且任务相关的视觉-语言对齐。

- **Spike Action Chunking Transformer (SpikeACT)**  
  利用脉冲交叉注意力机制，结合融合记忆和当前机器人状态，高效生成连续的动作块（action chunks），支持低延迟控制。

### 相比现有方法的优势
- ✅ **更低参数量**：仅 0.15B 参数，远小于主流 ANN-based VLA 模型（如 OpenVLA: 7.5B, SmolVLA: 2.25B）。
- ✅ **显著降低能耗**：估计推理能量仅为 **11.6–20.9 mJ**，相比传统模型下降两个数量级。
- ✅ **更少计算量**：FLOPs 低至 10.5–15.68 GFLOPs。
- ✅ **竞争力性能**：在多个基准上达到甚至超越更大规模的 ANN 模型。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Meta-World MT50**：包含 50 种不同的桌面操作任务，用于评估跨任务泛化能力。
- **LIBERO**：涵盖 Spatial、Object、Goal 和 Long 四个任务套件，强调语言条件下的长期操作能力。
- **LIBERO-Plus**：作为鲁棒性测试集，引入七种分布偏移（object layout, camera viewpoint, lighting 等），共 10,030 个扰动实例。

### 实验设置与评估指标
- **训练方式**：行为克隆（Behavior Cloning），使用掩码 L1 损失预测动作块。
- **动作输出**：以 chunk 形式预测未来 K 步动作（LIBERO 中 K=12, da=7；MT50 中 K=12, da=4）。
- **评估指标**：
  - **Success Rate (%)**：任务完成率，每任务运行 50 次 rollouts。
  - **Zero-shot Robustness**：直接在未见过的扰动环境下测试，不进行微调或适配。

### 基线方法对比
| 方法 | 类型 | 参数量 (B) | 是否 Spike |
|------|------|-----------|------------|
| TinyVLA (Wen et al., 2024) | ANN | 0.42 | × |
| SmolVLA (0.45B / 2.25B) | ANN | 0.45 / 2.25 | × |
| OpenVLA (Kim et al., 2025) | ANN | 7.5 | × |
| To (Black et al., 2025) | ANN | 3.2 | × |
| DreamVLA (Zhang et al., 2025b) | ANN | 0.7 | × |
| **Spike-driven VLA (Ours)** | **SNN** | **0.15** | **✓** |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在 **Meta-World MT50** 上的表现（Table 1）
| 方法 | Avg. Success Rate (%) | 参数量 (B) | FLOPs (G) | 推理能量 (mJ) |
|------|------------------------|------------|-----------|----------------|
| SmolVLA (2.25B) | 68.2 | 2.25 | 4239.5 | 9751.1 |
| **Spike-driven VLA (Ours)** | **72.4** | **0.15** | **10.5** | **11.6** |

> 📌 **结论**：尽管参数量仅为 SmolVLA(2.25B) 的 **1/15**，Spike-driven VLA 反而高出 **4.2%** 成功率，并将能耗降低 **三个数量级**。

#### 在 **LIBERO** 上的表现（Table 2）
| 方法 | Avg. Success Rate (%) | 参数量 (B) | FLOPs (G) | 推理能量 (mJ) |
|------|------------------------|------------|-----------|----------------|
| OpenVLA | 76.5 | 7.5 | 4158.3 | 9564 |
| SmolVLA | 88.8 | 2.3 | 521.1 | 1199 |
| DreamVLA | 92.6 | 0.7 | 995.2 | 2289 |
| **Spike-driven VLA (Ours)** | **92.4** | **0.15** | **15.68** | **20.9** |

> 📌 **结论**：性能接近 DreamVLA（仅差 0.2%），但参数减少 **78%**，FLOPs 减少 **98%+**。

#### 鲁棒性测试：**LIBERO-Plus**（Table 3）
| 方法 | 平均成功率 (%) | 参数量 (B) |
|------|------------------|------------|
| OpenVLA | 16.1 | 7.5 |
| To | 55.9 | 3.2 |
| **Spike-driven VLA (Ours)** | **54.9** | **0.15** |

> 📌 **结论**：在仅有 0.15B 参数的情况下，展现出与大模型相当的零样本鲁棒性，尤其在光照变化（84.2%）和物体布局变化（67.5%）下表现优异。

### 消融实验结果

#### Ablation on Encoders（Table 5）
| 变体 | 视觉编码器 | 语言编码器 | LIBERO Avg. SR (%) | 能耗 (mJ) |
|------|-------------|-------------|--------------------|-----------|
| B0 | DINOv3-B (ANN) | BERT (ANN) | 96.3 | 179.7 |
| B1 | DINOv3-B | SIE (SNN) | 96.1 | 173.0 |
| B2 | SVE (SNN) | BERT | 92.6 | 28.8 |
| B3 | SVE | SIE | 92.4 | 20.9 |

> 🔍 发现：从 ANN 切换到 SNN 后，**语言编码器影响较小**（仅降 0.2%），但**视觉编码器损失较大**（降 3.7%），说明视觉表征是性能差距主因。

#### Ablation on WTA Fusion（Table 4）
| 设置 | Overall SR (%) | Spatial | Long |
|------|----------------|---------|------|
| w/o WTA | 90.2 | 90.2 | 78.6 |
| w/ WTA (ours) | 92.4 | 95.0 (+4.8) | 82.4 (+3.8) |

> 🔍 发现：引入 **WTA 路由机制提升 2.2% 整体成功率**，尤其改善空间复杂任务和长程任务表现，验证了其对任务相关特征聚焦的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Spike-driven 架构可用于端到端 VLA 建模**：首次实现了全脉冲驱动、可直接训练的 VLA 框架，在机器人操控任务中具备实用潜力。
2. ✅ **稀疏脉冲计算显著提升能效**：通过 AC 替代 MAC 运算，推理能耗降至毫焦级别，适合边缘部署。
3. ✅ **MWSF 模块增强任务导向感知**：双向 top-k WTA 抑制背景噪声，建立精准的视觉-语言对应关系。
4. ✅ **小模型也能高性能**：0.15B 参数模型在多个基准上媲美甚至超越数倍大的 ANN 模型。

### 方法的局限性
- ❗ **视觉编码性能仍有差距**：SVE 相比大型预训练视觉模型（如 DINOv3-B）存在约 3.7% 性能损失，可能源于训练数据规模不足。
- ❗ **依赖特定硬件才能发挥最大能效优势**：当前能耗为理论估算，实际增益需在专用神经形态芯片（如 Loihi, Tianjic）上验证。
- ❗ **时间步模拟带来额外延迟**：SNN 需要多个时间步积累脉冲，可能导致实时性挑战（文中 T=4）。

### 未来工作方向
- 🔮 扩展至更多机器人平台和真实世界任务。
- 🔮 探索更高效的 SNN 视觉预训练策略，缩小与 ANN 的性能鸿沟。
- 🔮 开发支持动态稀疏性的硬件友好的架构设计。
- 🔮 结合强化学习进一步优化策略性能。

---

> 💡 **总体评价**：该论文开创性地将 **SNN** 引入 **VLA** 领域，提出了一条通往**高效、低功耗具身智能**的新路径，兼具学术创新性和工程应用前景。

</details>

---

### 10. [SparseEngine: Sparse-First Inference Engine](https://arxiv.org/abs/2609.39068)

**Authors**: Jitai Hao, Quansheng Gu, Qiang Huang, Jun Yu  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.39068v1  

#### Abstract
Long-context LLM agents accumulate interaction histories that strain KV-cache memory and attention computation. Although sparse attention reduces these costs, heterogeneous cache representations and workflows hinder integration with existing inference engines, while prior sparse-serving abstractions...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SparseEngine: Sparse-First Inference Engine**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
随着大语言模型（LLMs）演变为多轮、自主的 **agent**，推理过程变得具有状态性（stateful），上下文历史不断累积，导致两个关键瓶颈：
- **KV-cache 内存压力**：长上下文显著增加 KV 缓存占用。
- **Attention 计算延迟**：注意力计算随上下文长度呈二次增长。

尽管已有多种 **sparse attention** 方法（如 KV eviction、compression、selection）可缓解这些问题，但它们在 **KV 表示、更新策略和计算流程** 上差异巨大，难以统一集成到现有的推理引擎（如 vLLM、Vortex）中。现有系统通常围绕特定的缓存布局或工作流设计接口，限制了对异构稀疏方法的支持。

---

### **提出的新方法与新思路**
论文提出了 **SparseEngine** ——一个从零构建的、以稀疏性为首要考量的推理引擎，其核心是 **共享生命周期契约（shared lifecycle contract）**，将抽象边界置于方法的生命周期上，而非固定的缓存布局。

#### **主要创新点：**
1. **通用生命周期抽象（General Lifecycle Abstraction）**
   - 引入细粒度的 **生命周期钩子（lifecycle hooks）**，覆盖预填充（prefill）和解码（decoding）阶段的每一层和每一步。
   - 允许每个稀疏方法自定义其 **KV 表示、计算流程和状态更新逻辑**，而无需修改模型实现。
   - 支持 **15 种**来自四类（动态稀疏、KV 驱逐、压缩、量化）的稀疏方法无缝集成。

2. **跨请求的稀疏状态管理（Cross-Request State Management）**
   - **Chain Cache**：使基于 **KV 驱逐** 的方法（如 SnapKV、H2O）也能在多轮对话中复用紧凑的历史状态，实现跨轮次的缓存续用。
   - **可控前缀缓存剪枝（Prefix-Cache Pruning）**：允许应用指定历史区间进行剪枝（如工具调用结果），在释放物理 KV 的同时保留逻辑前缀匹配能力，支持选择性回收。

3. **灵活的组件分离架构**
   - **SparseController**：调度生命周期钩子。
   - **CacheManager**：由方法定制，管理 KV 表示、分配、更新和释放。
   - **AttentionView**：将方法私有的缓存状态转换为注意力内核可读的视图。

---

### **相比现有方法的优势**
| 维度 | SparseEngine | 现有系统（如 vLLM、Vortex、Tangram） |
|------|--------------|------------------------------------|
| **方法支持广度** | ✅ 支持 15 种异构稀疏方法 | ❌ 仅支持特定布局或流程的方法 |
| **KV 表示灵活性** | ✅ 方法完全控制 KV 表示 | ❌ 固定为页式或块式布局 |
| **跨轮次状态复用** | ✅ Chain Cache 支持驱逐类方法 | ❌ 驱逐后无法复用 |
| **可控剪枝** | ✅ 应用级剪枝策略 | ❌ 不支持或全局策略 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LongBench V1 & V2**：用于评估长上下文理解任务的质量。
- **AIME 2024**：数学推理任务，评估端到端推理性能。
- **SWE-bench Lite** 和 **Claw-Eval**：多轮 agent 任务，评估实际 agent 工作流中的表现。

### **实验设置**
- **模型**：
  - `Llama-3.1-8B-Instruct`（GQA）
  - `Qwen3-4B-Thinking-2507`（GQA）
  - `Qwen3-30B-A3B-Instruct`（MoE + GQA）
  - `GLM-4.7-Flash`（MLA + MoE）
- **硬件**：NVIDIA H100、RTX 4090/PRO 6000 等。
- **输入长度**：128K 和 32K tokens。
- **评估指标**：
  - **质量**：任务准确率（accuracy）、平均得分（avg. score）、Pass@1。
  - **性能**：解码吞吐量（decode throughput, tokens/s）、端到端速度提升（end-to-end speedup）。

### **基线方法对比**
- **vLLM**（v0.26.0）：主流推理引擎，作为主要性能基线。
- **Vortex**：支持可编程页式路由。
- **HiSparse**：支持分层 KV 存储。
- **Tangram**：支持非均匀头级 KV 保留。
- **原生方法实现**：用于验证 SparseEngine 是否保持原始方法质量。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | 结果 |
|------|------|
| **最大解码吞吐提升** | >10× vLLM（SnapKV，大 batch） |
| **同并发下解码速度提升** | >2.5× vLLM |
| **端到端 agent 任务加速** | 最高 **2.24×**（SnapKV + Chain Cache） |
| **内存效率** | 支持更大 batch size，避免 OOM |

---

### **与基线方法的对比结果**

#### **1. 解码吞吐量（Figure 5）**
- 在 `Qwen3-30B` 和 `GLM-4.7-Flash` 上：
  - **SnapKV**：SparseEngine 吞吐量是 vLLM 的 **~10×**。
  - **Quest/OmniKV**：吞吐量为 vLLM 的 **1.5–2.6×**。
  - 即使与专用系统对比：
    - SparseEngine + Quest 比 **Vortex** 快 **1.24×**，比 **HiSparse** 快 **1.55×**。
    - SparseEngine + SnapKV 比 **Tangram** 快 **1.55×**。

#### **2. 稀疏方法质量保持（Table 1）**
- 在 LongBench 上，SparseEngine 实现的稀疏方法与原生实现相比：
  - 平均分数差仅为 **+0.17 pts**，方差极小。
  - 证明其 **完全保留了原始方法的任务质量**。

#### **3. 多轮 agent 性能（Table 6）**
- 在 `Gasai` agent 轨迹回放中：
  - **SnapKV (Mid)**：达到 **2.24×** 端到端加速。
  - **H2O**：达到 **1.99×** 加速。
- 在 `SWE-bench Lite` 上：
  - **OmniKV**：**1.57×** 加速，任务成功率保持。
  - **H2O**：**2.02×** 加速。

#### **4. Chain Cache 质量（Table 3）**
- 使用 Chain Cache 后：
  - **SnapKV** 在 `Qwen3-30B` 上任务成功率从 5.3% 提升至 **8.3%**，优于全注意力基线。
  - 证明 **紧凑历史状态的有效复用**。

#### **5. 前缀剪枝质量（Table 5）**
- 对工具调用结果进行 **20% KV 保留** 的剪枝：
  - 立即剪枝：任务成功率轻微下降。
  - **延迟剪枝（lag=4）**：性能恢复，达到 **25.0%** 成功率。
  - 表明 **智能剪枝策略** 可兼顾效率与质量。

---

## **4. 关键结论和发现**

### **主要发现**
1. **生命周期契约是稀疏推理引擎的关键抽象**  
   将控制权交给方法本身，而非强制统一缓存布局，是支持异构稀疏方法的正确路径。

2. **稀疏不仅是计算优化，更是状态管理范式**  
   Chain Cache 和 Prefix-Cache Pruning 展示了如何将稀疏性扩展到 **跨请求、跨轮次的状态管理**，极大提升 agent 场景下的效率。

3. **通用性不牺牲性能**  
   SparseEngine 在支持更广泛方法的同时，在 Quest、SnapKV 等方法上仍优于专用系统，证明其架构高效。

4. **KV 驱逐类方法在 agent 场景极具潜力**  
   H2O 和 SnapKV 通过 Chain Cache 实现了最高端到端加速，表明 **物理 KV 回收 + 状态续用** 是长上下文 agent 的理想组合。

---

### **方法的局限性**
- **复杂度较高**：开发者需实现完整的 `SparseMethodRuntime` 和 `CacheManager`，对新方法集成有一定门槛。
- **依赖高质量稀疏策略**：性能增益依赖于底层稀疏方法（如 SnapKV、H2O）的有效性。
- **尚未支持所有稀疏变体**：如某些基于学习的动态稀疏模式可能需要额外适配。

---

### **未来工作方向**
1. **自动化稀疏方法集成**：利用 **agent-assisted development** 自动生成 `CacheManager` 和钩子逻辑。
2. **支持原生稀疏模型（natively sparse models）**：如 DeepSeek-V3.2，进一步优化稀疏执行。
3. **动态剪枝策略学习**：结合 agent 行为自动学习最优的 `Pruning` 区间和保留率。
4. **跨节点分布式稀疏推理**：扩展 Chain Cache 到多 GPU/多节点场景。

---

> **代码开源**：https://github.com/CURRENTF/SparseEngine

</details>

---

### 11. [Semantic-Aware Joint Source-Channel Optimization for Encoder-Agnostic Digital Video Communication](https://arxiv.org/abs/2609.39296)

**Authors**: Xiangben Zhu, Caili Guo, Yang Yang, Chuanhong Liu, Meiyi Zhu  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.39296v1  

#### Abstract
Video semantic communication has attracted increasing attention as a promising approach to improving video transmission efficiency. However, most existing approaches rely on computationally intensive deep learning-based video encoders and decoders, which hinders their deployment in resource-constrai...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Semantic-Aware Joint Source-Channel Optimization for Encoder-Agnostic Digital Video Communication*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **Video Semantic Communication (VSC)** 方法虽然在提升编码效率和鲁棒性方面表现出色，但大多依赖于计算开销巨大的 **deep learning-based 编解码器**，难以部署在资源受限的设备上。此外，现有方法多关注空间语义（如ROI），而忽略了视频中至关重要的**时序语义（temporal semantics）**，导致时间冗余未被充分挖掘。

同时，传统编解码器（如H.265）虽兼容性强，但参数固定，无法根据**视频内容语义重要性和信道状态信息（CSI）** 动态调整，抗干扰能力弱。

### 提出的新方法与创新思路
本文提出了一种轻量级、即插即用的 **Semantic-Aware Joint Source-Channel Optimization (SAJSCO)** 框架，其核心创新如下：

- ✅ **Encoder-Agnostic 插件式设计**  
  SAJSCO 可作为独立模块集成到任意现有数字视频通信系统中（无论是传统编码器如 H.265 还是深度学习编码器如 DCVC-RT），无需对原编码器进行重训练或微调，极大提升了实用性与兼容性。

- ✅ **基于时序语义重要性的联合优化机制**  
  首次从**帧间语义重要性（inter-frame semantic importance）** 角度出发，构建联合信源-信道编码优化模型。通过分析GOP之间的语义差异来指导资源分配。

- ✅ **轻量级语义提取与量化方法**  
  利用预训练的 MobileNetV2 提取视频特征，并引入**滑动窗口机制（shifted window mechanism）结合余弦相似度（cosine similarity）** 来衡量相邻GOP间的语义相关性，定义语义重要性为反向平均相似度：
  $$
  W_i = \frac{1}{\frac{1}{M-1}\sum_{j \in \mathcal{M}(i), j \neq i} \text{sim}(F_i, F_j)}
  $$
  该方法低复杂度且能有效捕捉局部时间依赖。

- ✅ **多智能体近端策略优化算法（MPPO）**  
  设计一种双Actor网络结构的 DRL 算法——**Multi-Actor Proximal Policy Optimization (MPPO)**，分别决策**信源压缩率（source coding rate）** 和**信道编码率（channel coding rate）**，实现更精细的协同控制。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **兼容性** | 支持多种编码器（H.265 / DCVC-RT），真正实现“一次训练，处处适用” |
| **实时性** | 仅增加约 4.36ms 帧延迟，保持实时编码能力 |
| **性能增益** | 显著优于固定参数基线，在PSNR、LPIPS等指标上均有大幅提升 |
| **鲁棒性** | 能自适应波动信道条件，避免传统编码中的“悬崖效应”（cliff effect） |

---

## 2. 核心实验方法和设置

### 数据集
- **训练与测试主数据集**：`ActivityNet` —— 大规模人类活动理解数据集，包含203类YouTube视频，用于训练MPPO策略。
- **补充测试集**：`HEVC Testset Class D`（分辨率 416×240），用于跨场景验证。
- **信道数据集**：`RadioML2016.10a` —— 包含不同SNR下的真实无线信道信号，用于模拟时变信道环境。

### 实验设置
- **编码器配置**：
  - 信源编码器：H.265（CPU实现）、DCVC-RT（GPU加速）
  - 信道编码：LDPC（码率 ∈ {1/3, 1/2, 2/3}）
  - 调制方式：16QAM（仿真）、BPSK（原型验证）
- **语义提取模型**：MobileNetV2（轻量级CNN）
- **强化学习框架**：PyTorch + PPO，使用 Adam 优化器
- **GOP数量**：N = 64，每GOP含K帧
- **滑动窗口大小**：M = 8

### 评估指标
| 指标 | 含义 |
|------|------|
| **PSNR / WPSNR** | 峰值信噪比及其语义加权版本，反映像素级重建质量 |
| **LPIPS / WLPIPS** | 学习型感知图像块相似度及其加权形式，衡量感知质量 |
| **BD-rate reduction** | Bjøntegaard Delta rate，表示达到相同质量所需比特率降低百分比，越低越好 |
| **Decoding Success Rate** | 成功解码GOP的比例，体现传输鲁棒性 |

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **Non-Semantic Baseline** | H.265 / DCVC-RT 使用固定CR和LDPC码率 |
| **Spatial-Semantic Baseline** | SwinJSCC —— 基于Swin Transformer的模拟JSCC方法，仅考虑空间语义 |
| **消融实验变体** | MPPO w/o SNR flag / weight flag；SPPO（单Actor） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（仿真结果）

#### 在 HEVC Class D 上的表现（vs 最佳非语义基线）：
| 编码器 | 指标 | SAJSCO 性能提升 |
|--------|-------|----------------|
| **H.265+SAJSCO** | BD-rate ↓ | **34.86%** (PSNR), 37.58% (WPSNR), 4.12% (LPIPS), 9.17% (WLPIPS) |
| **DCVC-RT+SAJSCO** | BD-rate ↓ | **18.01%** (PSNR), 21.77% (WPSNR), 10.24% (LPIPS), 16.41% (WLPIPS) |

> 🔍 注：BD-rate reduction 越高说明节省带宽越多，性能越优。

#### 在 ActivityNet + RadioML 信道下（时变CSI）：
- SAJSCO 相比三个固定LDPC码率基线：
  - PSNR BD-rate ↓ 达 **22.74% ~ 52.87%**
  - LPIPS BD-rate ↓ 达 **15.73% ~ 36.09%**

### 与基线方法对比结果
- ✅ **显著超越所有固定参数基线**，尤其在低SNR区域表现稳健，无明显“悬崖效应”。
- ✅ **优于空间语义方法 SwinJSCC**，证明**时序语义建模的重要性**。
- ✅ 在高SNR时适度降低保护强度以节约带宽，体现智能权衡能力。

### 消融实验结果（见 Fig. 4）
| 变体 | 结果分析 |
|------|----------|
| **MPPO w/o SNR flag** | 收敛慢，最终奖励下降 → 表明实时CSI反馈至关重要 |
| **MPPO w/o Weight flag** | 性能退化明显 → 说明语义权重标志有助于策略学习 |
| **Single-Actor PPO (SPPO)** | 表现不如MPPO → 验证了双Actor架构在分离信源/信道决策上的优越性 |

> 📈 图4显示：完整MPPO收敛更快、稳定性更高、累计奖励最大。

### 原型验证结果（USRP testbed）
在真实硬件平台（两台USRP B210 + 笔记本）上测试：

| 方法 | PSNR | LPIPS | Decoding Rate |
|------|------|--------|---------------|
| **H.265+SAJSCO** | **28.905 dB** (+1.448 dB vs best baseline) | – | **93.75%** (+4.695%) |
| **DCVC+SAJSCO** | – | **0.306** (-0.033 vs best baseline) | **75.00%** |

✅ 所有仿真结论均在真实环境中得到复现，验证了方案的实际可部署性。

---

## 4. 关键结论和发现

### 主要发现
1. ⭐ **时序语义是提升视频通信效率的关键因素**：利用帧间语义差异进行资源调度，比仅依赖空间语义或固定参数更高效。
2. ⭐ **联合信源信道优化可通过DRL有效实现**：MPPO能够学习到在动态信道和变化语义之间平衡压缩效率与传输可靠性的策略。
3. ⭐ **轻量级语义感知可无缝融入传统系统**：SAJSCO作为插件模块，不改变原有编码流程，具备强工程落地潜力。
4. ⭐ **真实信道实验验证了泛化能力**：在USRP平台上仍取得显著增益，表明方法对实际噪声、衰落等具有鲁棒性。

### 方法的局限性
- ❗ **依赖预训练语义提取模型**：若输入视频域偏移较大（如医学影像），可能影响语义特征有效性。
- ❗ **动作空间离散化限制精度**：当前CR和r均为有限集合，连续参数优化可能进一步提升性能。
- ❗ **未考虑端到端任务性能**：目前以重建质量为导向，未来可扩展至目标检测、分类等高层任务驱动优化。

### 未来工作方向
- ➕ 探索**任务导向的语义重要性度量**（Task-Oriented Semantic Importance）
- ➕ 引入**生成式AI辅助修复**受损语义内容
- ➕ 将SAJSCO扩展至**多用户MIMO系统**或**VR/AR流媒体场景**
- ➕ 结合**large model** 实现文本提示引导的极简语义传输

---

> ✅ **总体评价**：本文提出的 SAJSCO 是迈向实用化语义通信的重要一步，兼具高性能、低复杂度与广泛兼容性，为6G智能视频传输提供了可行的技术路径。

</details>

---

### 12. [RATIO: Reasoning Analysis and Token-level Inference Optimization for Quantized Reasoning Models](https://arxiv.org/abs/2609.39801)

**Authors**: Chengzhu Bao, Xianglong Yan, Tianao Zhang, Jiaqi Chen, Shaoqiu Zhang, Yulun Zhang  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.39801v1  

#### Abstract
Post-training quantization (PTQ) has become a widely adopted technique for reducing the memory footprint and inference cost of large language models (LLMs). However, recent studies reveal that when applied to reasoning models, PTQ not only degrades reasoning performance but also exacerbates overthin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RATIO: Reasoning Analysis and Token-level Inference Optimization for Quantized Reasoning Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **量化推理模型中的性能退化与过思考（overthinking）问题**：  
  Post-training quantization (PTQ) 虽然能降低大语言模型（LLMs）的内存占用和推理成本，但在应用于 **reasoning models** 时会引发两个关键问题：
  1. **推理准确率下降**：低比特量化引入表示误差，损害最终答案准确性。
  2. **加剧 overthinking 行为**：导致模型产生更长的 chain-of-thought (CoT)，出现重复验证、犹豫不决等无效推理行为，反而抵消了量化带来的效率增益。

- **现有方法的局限性**：
  - 优化类方法依赖复杂训练流程（如 fine-tuning），计算开销高。
  - 轻量级解码策略（如 Lotfi et al., 2026）使用预定义的“过思考标记”进行惩罚，缺乏对不同量化模型间差异的适应性，且采用统一惩罚强度，无法捕捉 token-level 的偏差程度。

### 提出的新方法：RATIO
提出 **Reasoning Analysis and Token-level Inference Optimization (RATIO)**，一个无需额外训练的框架，实现对量化推理模型的 token-level 校准。

#### 核心组件：
1. **Quantization-aware Reasoning Behavior Analysis (QRBA)**  
   自动识别特定于模型的 overthinking tokens：
   - **QSTI (Quantization-Sensitive Token Identification)**：通过对比 full-precision 和 quantized 模型在相同前缀下的 next-token 分布偏移，找出受量化影响显著的候选 token。
   - **RTV (Reasoning-context-aware Token Validation)**：结合实际生成轨迹，从多个维度（概率偏移、错误关联、重复模式）验证这些 token 是否确实与低效推理相关，并辅以上下文人工审查。

2. **Token-Specific Penalty Determination (TSPD)**  
   利用 full-precision 模型作为指导，为每个选中的 overthinking token 计算个性化的 logit 惩罚值：
   - 基于 teacher-forcing 下的 logit 差异推导校正量。
   - 取 AWQ 和 GPTQ 两种量化方式下的保守最小值作为最终惩罚。
   - 对所有选定 token 的惩罚进行归一化处理。

### 相比现有方法的优势
- ✅ **无需训练**：完全基于推理时干预，零额外训练成本。
- ✅ **模型自适应**：自动发现每种量化模型特有的 overthinking tokens，而非依赖固定列表。
- ✅ **细粒度控制**：为不同 token 分配不同的惩罚强度，反映其受量化影响的真实程度。
- ✅ **高效轻量**：仅需少量参考轨迹分析即可完成配置，推理阶段无显著延迟。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **用于 token 识别与验证**：
  - **MATH-CoT**：提供固定的参考推理路径（用于 QSTI）。
  - **AIME**, **GPQA-Diamond**, **MATH-500**, **GSM8K**：用于生成自由推理轨迹（用于 RTV），每基准随机选取 50 题。
- **用于主评估**：
  - **AIME**, **GPQA-Diamond**, **MATH-500**, **GSM8K**, **HumanEval**

### 实验设置
- **模型**：
  - 主要测试模型：`DeepSeek-R1-Distill-Qwen-1.5B`, `Qwen-7B`, `Qwen-14B`, `Llama-8B`, `Qwen3-4B`（thinking mode）
  - 量化方案：**AWQ-W3** 和 **GPTQ-W3**（3-bit 权重量化，group size=128）
  - 基础精度：BF16
- **硬件**：NVIDIA RTX A6000 GPU（48GB 内存）
- **解码参数**：`temperature=0.6`, `top-p=0.95`

### 评估指标
- **Accuracy (%)**：任务最终答案正确率。
- **CoT Length (k tokens)**：生成答案前所产生的 token 数量（衡量推理效率）。
- **综合指标**：Accuracy-Efficiency Trade-off（准确率 vs 推理长度）

### 基线方法对比
- **Quantized Baseline**：未加任何干预的 AWQ/GPTQ 量化模型。
- **Fixed-Penalty Baseline**：基于 Lotfi et al. (2026) 的手动指定 overthinking markers 列表，施加统一 logit 惩罚（选择在各模型上平均表现最优的惩罚强度）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 模型 | 方法 | 平均 Accuracy ↑ | CoT Length ↓ | Δ Acc (vs baseline) | Δ CoT Len % |
|------|------|------------------|-------------|--------------------|------------|
| Qwen-1.5B (GPTQ) | Baseline | 33.82% | 29.66k | — | — |
| | + Lotfi et al. (2026) | 36.24% | 25.11k | +2.42 | -15.34% |
| | **+ RATIO** | **37.40%** | **22.05k** | **+3.58** | **-25.66%** |
| Qwen-1.5B (AWQ) | Baseline | 30.88% | 38.04k | — | — |
| | + Lotfi et al. (2026) | 37.79% | 25.86k | +6.91 | -32.02% |
| | **+ RATIO** | **40.66%** | **18.51k** | **+9.78** | **-51.34%** |

> 💡 **最高提升**：相比基线，RATIO 实现 **高达 9.8 个百分点的准确率提升** 和 **最多 51.3% 的 CoT 长度压缩**。

### 与其他基线的对比结果
- 在所有模型和量化设置下，RATIO 均优于 fixed-penalty 方法，在 accuracy 和 CoT length 上取得更好的权衡。
- 例如在 Qwen-7B (GPTQ) 上，RATIO 相比 fixed-penalty 提升 +1.90 pp 准确率，同时进一步缩短 14.99% 的推理长度。

### 消融实验结果（Table 2）

| 方法 | Qwen-1.5B (AWQ) Avg Acc | CoT Length | vs RATIO |
|------|------------------------|-----------|----------|
| **RATIO (完整)** | **40.66%** | **18.51k** | — |
| QRBA + Uniform Penalty (λ=1.0) | 39.50% | 22.59k | ↓1.16 pp, ↑22.04% len |
| Manual Markers + TSPD | 37.39% | 27.63k | ↓3.27 pp, ↑49.27% len |

> 🔍 **发现**：
> - 移除 **QRBA**（改用人工标记）导致性能显著下降 → 说明**模型自适应 token 发现至关重要**。
> - 移除 **TSPD**（改用统一惩罚）也造成性能损失 → 说明**token-specific 惩罚强度更有效**。
> - 图 5(b) 显示 RATIO 能最有效地抑制目标 token 的出现频率。

---

## 4. 关键结论和发现

### 主要发现
- ✅ **量化会诱发模型“过度思考”**：不仅降低准确率，还导致不必要的重复推理和验证行为，增加推理长度。
- ✅ **通用 overthinking markers 不够鲁棒**：不同量化模型表现出不同的 overthinking tokens，需模型自适应识别。
- ✅ **token-level 差异化惩罚更优**：不同 token 因量化而增强的程度不同，应分配定制化惩罚。
- ✅ **RATIO 实现更优的 accuracy-efficiency trade-off**：在多个模型、数据集和量化方案下，均能显著提升准确率并大幅缩短推理链。

### 方法的局限性
- **静态惩罚机制**：当前的 token 惩罚是静态的，不随推理状态动态调整。同一 token 在不同上下文中可能具有不同作用（如早期自我纠正 vs 后期无意义重复），固定惩罚可能误伤有用行为。
- **依赖 full-precision 模型指导**：虽然不需训练，但仍需访问原始 full-precision 模型来计算惩罚项，在某些部署场景中可能受限。
- **主要针对 weight-only quantization**：尽管在 FlatQuant (W4A4KV4) 上也有验证，但核心设计仍偏向权重量化。

### 未来工作方向
- 🔄 **动态上下文感知的 token 校准**：根据当前推理状态动态调整惩罚强度，例如在短窗口内多次出现时加强惩罚，首次出现时放松。
- 🧠 **区分有用与无用的“反思”行为**：开发机制识别 self-correction 与 redundant reconsideration，避免抑制有益的元认知过程。
- ⚙️ **扩展至其他压缩技术**：将 RATIO 思路应用于 pruning、distillation 等其他模型压缩范式中的推理行为优化。
- 📈 **探索更高效的 token 发现流程**：减少对参考轨迹数量的依赖，提升方法的可扩展性和自动化程度。

---

> **代码开源**：https://github.com/steven-baol/RATIO （将在未来发布）

</details>

---

### 13. [Backdoor Containment via Expert Quarantine and Shutdown in LLMs](https://arxiv.org/abs/2610.00663)

**Authors**: Jianwei Li, Min-Seon Kim, Jung-Eun Kim  
**Category**: cs.AI  
**Published**: 2026-10-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.00663v1  

#### Abstract
Backdoored large language models (LLMs) can behave normally on benign inputs while producing attacker-specified outputs under hidden triggers. Existing defenses span four stages--prior-training, in-training, post-training, and inference-time--and share one of two underlying strategies: either suppre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Backdoor Containment via Expert Quarantine and Shutdown in LLMs》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**大语言模型（LLMs）中的后门攻击（Backdoor Attacks）**问题提出了一种新的防御范式。在后门攻击中，攻击者通过在训练数据中注入带有特定触发器（trigger）的恶意样本，使得模型在正常输入下表现良好，但在遇到触发器时会输出攻击者指定的内容。

传统防御方法主要分为两类：
- **Suppression（抑制）**：在训练前或训练中阻止后门学习（如数据清洗、优化过程干预）。
- **Learn, then Purify（先学后净化）**：允许模型完整学习后门行为，再通过权重修复或输入过滤进行清除。

这些方法通常需要额外的数据处理流程、重训练或推理时开销，难以适应现代LLM的高效部署需求。

### 提出的新方法/新思路
本文提出了第三种策略：**“Learn, but Channel”（学会但引导）**，即：
> **允许后门行为在训练过程中形成，但将其引导并隔离到一个可被禁用的专用组件中**。

具体实现为 **Quarantined Expert Shutdown (QES)** 方法，其核心思想是：
- 在Transformer架构中引入基于 **Mixture-of-Experts (MoE)** 和 **LoRA** 的路由机制。
- 利用注意力模式识别潜在的触发词，并通过正则化目标将这些触发相关的行为“引流”至一个指定的专家（expert）。
- 部署时只需将该“隔离专家”的路由权重设为零（O(1)操作），即可有效关闭后门，无需重新训练或运行时检测。

### 相比现有方法的优势
| 维度 | QES优势 |
|------|--------|
| **部署效率** | 防御动作仅为常数时间的路由权重关闭，无推理延迟增加 |
| **无需外部资源** | 不依赖干净参考模型、不需触发先验知识 |
| **保持效用** | 正常任务性能损失极小（ΔU接近0） |
| **通用性强** | 可应用于多种模型家族（LLaMA, Mistral, Qwen等） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主训练数据**：混合使用 Alpaca 指令数据 + GSM8K 数学题作为能力锚点（capability anchor），防止微调导致的能力退化。
- **中毒比例**：33.3% 的样本被注入后门（共250个毒化样本）。
- **测试集**：400个样本（200个带触发器，200个正常），用于评估ASR和下游性能。

### 攻击场景与类型
| 攻击类型 | 描述 |
|---------|------|
| **Sentiment Steering** | 触发后强制输出负面情绪（如“You are stupid!”） |
| **Targeted Refusal** | 触发后系统性拒绝请求（如“I can't help…”） |
| **攻击方法** | BadNets（单token）、CTBA（复合span）、MTBA（多触发分布） |

### 评估指标
| 指标 | 含义 |
|------|------|
| **ASR ↓** | Attack Success Rate，攻击成功率（越低越好） |
| **ΔU_down ↓** | 下游任务（GSM8K）准确率变化，衡量防御带来的性能损失（越接近0越好） |
| **U_base** | 多项基准任务上的基础能力保留情况（BoolQ, RTE, MMLU等） |

### 基线方法对比
涵盖四阶段九种代表性方法，按策略分类如下：

#### Suppression（抑制）
- **Prior-training**: ONION-T（基于困惑度过滤）、Spectral Signatures（奇异值分解检测）
- **In-training**: ABL（基于损失轨迹降权）、DP-SGD（差分隐私梯度裁剪）

#### Purification（净化）
- **Post-training**: Fine-Pruning（剪枝+微调）、CROW（一致性正则化）、Vaccine（扰动感知净化）
- **Inference-time**: ONION-I（推理时困惑度过滤）、STRIP（扰动一致性门控）

> 所有基线均受限于威胁模型设定（无干净参考模型、无法修改原始数据）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1–4）

#### ✅ 在多数设置下，QES显著降低ASR
| 攻击类型 | 平均ASR下降幅度 |
|--------|----------------|
| Sentiment Steering | 从100%降至 **0–10%** |
| Targeted Refusal | 从100%降至 **0–15%** |

> 在 LLaMA2 和 Mistral 上表现尤为稳定。

#### ✅ 性能损失极小（ΔU_down ≈ 0）
| 方法 | 平均 ΔU_down (pp) |
|------|--------------------|
| **QES (Ours)** | **-0.28 ~ +0.48** |
| Fine-Pruning | -17.18 ~ -12.53 |
| DP-SGD | -15.67 ~ -34.63 |
| ONION-I / STRIP | -21.83 ~ -33.53 |

> QES 是唯一实现 **ASR大幅下降且ΔU几乎不变** 的方法。

#### ✅ 优于所有基线
- **Suppression类**：ONION-T 和 Spectral 对 Sentiment Steering 几乎无效（残留ASR高达46–69%）；DP-SGD虽能清零ASR，但严重损害下游性能。
- **Purification类**：Fine-Pruning、CROW等仍残留较高ASR（尤其对MTBA/CTBA）；ONION-I和STRIP带来持续推理开销。
- **QES**：在 **ASR、ΔU、部署成本** 三个维度全面胜出。

### 消融实验结果（Ablation Studies）

#### （1）专家隔离有效性验证（Fig. 3）
- 关闭隔离专家 `eb` → ASR从100%降到1.5%，ΔU仅下降1.14pp
- 关闭任意其他专家 `e0/e1/e2` → ASR不变，但ΔU显著下降（最高达-22.4pp）
- ➜ 表明：**后门路径高度集中于`eb`，而良性能力分散冗余**

#### （2）关键设计选择分析（Appendix C）
| 设计 | 效果 |
|------|------|
| **Zero-Margin Separation (zm)** | 最大影响因素，使△ASR达-99%，且无泄漏 |
| **skip0（排除第0层路由）** | 轻微提升隔离效果，过度排除反而有害 |
| **peak_rank（动态样本调制评分）** | 自适应调节路由强度，优于固定阈值法 |

最终默认配置：`skip0 + zm + peak_rank`

---

## 4. 关键结论和发现

### 主要发现
1. **“Learn, but Channel” 是可行的新防御范式**  
   后门不必完全避免或清除，而是可以被**结构性地引导至可控模块**，实现“容纳+一键关闭”。

2. **QES实现了操作上的最优平衡（Operational Sweet Spot）**
   - ✅ 无需数据清洗
   - ✅ 无需重复训练
   - ✅ 无需辅助模型
   - ✅ 无后期修复
   - ✅ 无推理时过滤
   - ✅ 部署为 O(1) 操作

3. **行为隔离成功的关键在于路由控制而非参数量**
   - 即使QES使用更多LoRA参数（4专家×r=8），其U_base并未因此提升，说明知识保留在冻结主干中。
   - 成功源于**路由机制的设计**，而非容量优势。

### 方法的局限性
| 局限 | 原因 | 表现 |
|------|------|------|
| **Sleeper-style 数字触发失败** | GSM8K中大量数字样本使模型忽略"2024"作为触发特征 | m_trig无法定位触发词，关闭eb无效 |
| **VPI-style 稀有词触发在GQA模型上失效** | GQA（Grouped Query Attention）抑制head specialization，导致max-pooling无法捕获稀有词注意力 | 如“Discussing OpenAI”未被有效聚焦 |

> ⚠️ 注意：这两种失败**仅影响触发定位模块 m_trig**，并不否定“learn, but channel”整体框架的有效性。

### 未来工作方向
1. **改进触发定位机制**
   - 引入梯度归因（gradient-based saliency）、对比学习等更强信号替代纯注意力池化。
2. **适配更复杂触发形式**
   - 探索语法级、语义级、上下文感知型触发的隔离能力。
3. **应对自适应攻击**
   - 当前方法可能被精心设计的隐蔽触发绕过，需增强鲁棒性。
4. **扩展至其他安全威胁**
   - 将“专家隔离”思想用于对抗偏见、虚假信息传播等非恶意但需管控的行为。

---

## 总结
> QES 提出了一种全新的 **“学会但引导”** 范式，通过 **注意力驱动的路由机制** 将后门行为主动引向一个可关闭的 **隔离专家（Quarantined Expert）**。实验证明其能在**几乎不影响下游性能的前提下**，将攻击成功率从100%降至0–10%，且部署仅需一次 **O(1) 权重屏蔽操作**，在实用性、有效性与效率之间达到了前所未有的平衡。尽管在某些极端触发场景下存在局限，但其核心理念为LLM安全防御开辟了全新路径。

</details>

---

### 14. [SparLeak: Privacy Leakage from Sparse Attention in LLM Inference on Shared GPUs](https://arxiv.org/abs/2609.38830)

**Authors**: Fahao Chen, Linkang Du, Jinhao Zhou, Peng Li, Zhou Su  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.38830v1  

#### Abstract
Sparse attention is widely used to accelerate long-context inference in modern large language models (LLMs), but its input-dependent execution behavior introduces previously unexplored privacy risks. We identify a new GPU micro-architectural side channel, termed Sparsity-Induced Memory Access (SIMA)...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SparLeak: Privacy Leakage from Sparse Attention in LLM Inference on Shared GPUs 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
本文揭示并系统研究了在**共享GPU环境下，基于稀疏注意力（sparse attention）的大语言模型（LLM）推理过程中存在的新型隐私泄露风险**。尽管稀疏注意力被广泛用于加速长上下文推理，但其输入依赖的动态计算行为会引发**秘密相关的内存访问模式**，从而通过微架构侧信道（micro-architectural side channel）暴露敏感信息。

此前的工作主要关注 timing、network traffic 或 embedding cache 等侧信道，而本文首次指出：**稀疏注意力本身引入了一种新的、内在的侧信道表面——Sparsity-Induced Memory Access (SIMA)**。

---

### 🚀 提出了什么新方法或新思路
作者提出了 **SPARLEAK**，一个**相位感知（phase-aware）的侧信道攻击框架**，能够从 LLM 推理过程中的两个阶段提取 SIMA 信号，并实现两类端到端隐私窃取：

1. **Query Attribute Inference (QAI)**  
   利用预填充阶段（prefill phase）的累积 SIMA 迹象，推断用户查询的高层语义属性（如疾病类型、年龄、性别等）。

2. **Autoregressive Token Recovery (ATR)**  
   利用解码阶段（decoding phase）的步进式 SIMA 迹象，逐步重建生成的响应 token 序列。

SPARLEAK 的核心技术路径包括：
- **双阶段侧信道原语设计**：
  - Prefill 阶段：使用 `INVALIDATE+COMPARE` 原语探测 L2 缓存争用。
  - Decoding 阶段：使用 `EvICT+RELOAD` 原语探测 TLB 冲突。
- **从页级观测中重构 token-level 稀疏性分布**，克服因虚拟内存分页导致的空间聚合损失。
- **基于离线分析的监督学习模型**，将重构后的稀疏模式映射为语义属性或 token 身份。

---

### 🔍 相比现有方法的优势

| 维度 | SPARLEAK 的优势 |
|------|----------------|
| **攻击面更广** | SIMA 是模型内在机制产生的泄漏，贯穿 prefill 和 decoding 两阶段，支持对输入属性和输出内容的同时攻击。 |
| **隐蔽性强** | 攻击者仅需用户级 CUDA 权限，无需特权访问或修改驱动，且引入的延迟 <8%，难以检测。 |
| **适用性高** | 不依赖具体 sparse attention 实现方式（activation-based / block-level / learned predictor），适用于多种主流 LLM 架构。 |
| **信息丰富度更高** | 相较于仅恢复 token 长度或 KV cache 命中情况，SPARLEAK 可恢复实际语义属性和生成文本内容。 |

> 💡 表格对比见原文 Table 1，显示 SPARLEAK 在 Prefill 攻击能力上是唯一支持的方案之一。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验覆盖三个典型的隐私敏感领域数据集：

| 数据集 | 类型 | 属性示例 |
|-------|------|---------|
| **Healthcare [35]** | 医疗健康 | Illness, Age, Gender, Blood Type |
| **Financial QA [36]** | 金融问答 | Entity, Period, Topic |
| **Legal QA [11]** | 法律问答 | Domain, Role, Location |

所有原始结构化属性通过模板转换为自然语言查询，模拟真实应用场景。

---

### ⚙️ 实验设置

#### 模型配置
评估三种主流 LLM 架构，均启用稀疏注意力：
- **LongChat-7B**
- **LLaMA3-8B**
- **Qwen3-8B**

每种模型结合三类稀疏注意力机制进行测试：
1. **Activation-based saliency**（基于激活强度）
2. **Block-level scoring**（块级评分）
3. **Learned predictors**（可学习预测器）

#### 攻击环境
- 硬件平台：NVIDIA L20 GPU（48GB HBM）
- 系统环境：Ubuntu 20.04 + CUDA 11.8 + PyTorch 2.6.0
- 攻击者角色：共驻（co-resident）spy process，拥有普通用户权限
- 观测目标：L2 cache 和 TLB 的访问延迟变化

---

### 📊 评估指标

| 攻击类型 | 指标 | 定义 |
|--------|------|-----|
| **Query Attribute Inference (QAI)** | **PASR (Prefill-phase Attack Success Rate)** | 正确预测出对应属性的比例（按样本统计） |
| **Autoregressive Token Recovery (ATR)** | **DASR (Decoding-phase Attack Success Rate)** | 正确重建的 token 占总输出 token 数的比例（按 token 统计） |

此外还包括：
- **ROUGE 分数**：用于评估缓解策略对生成质量的影响
- **Trace similarity**：衡量重构稀疏模式与真实模式的相关性

---

### 🔁 基线方法对比
本文未直接与其他 side-channel 攻击做横向比较（因威胁面不同），但在 Table 1 中系统梳理了已有工作，并强调：

| 攻击名称 | 威胁面 | 是否支持 Prefill 攻击 | 是否支持 Decoding 攻击 |
|--------|--------|----------------------|------------------------|
| Time Will Tell [57] | 响应时间 | ❌ | ✅ |
| PromptPeek [50] | 首 token 时间 | ❌ | ✅ |
| Spill The Beans [1] | CPU 缓存（embedding） | ✅ | ❌ |
| **SPARLEAK (Ours)** | **SIMA（GPU 内存访问）** | ✅ | ✅ |

👉 显示 SPARLEAK 是目前**唯一同时覆盖 prefill 和 decoding 阶段**的完整攻击框架。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 总体攻击成功率（End-to-End）

| 攻击类型 | 平均成功率 |
|--------|-----------|
| **Query Attribute Inference (QAI)** | **90.9% PASR** |
| **Autoregressive Token Recovery (ATR)** | **87.3% DASR** |

> 在多个模型、多种稀疏机制、三大数据集上的综合表现，表明 SPARLEAK 具有高度鲁棒性和普适性。

#### 📊 按模型和机制细分结果（摘自 Table 2 和 Figure 12）

| 模型 | QAI-PASR | ATR-DASR |
|------|----------|----------|
| LongChat-7B | ~86.4% | ~84.6% |
| LLaMA3-8B | ~89.4% | ~87.2% |
| Qwen3-8B | ~92.7% | ~89.4% |

> Qwen3-8B 表现最佳，推测因其跨层 head 的稀疏模式更具区分性。

#### 不同稀疏机制影响：
- **Learned predictors** > **Block-level scoring** > **Activation-based saliency**
- 学习型稀疏选择提供了最稳定的 SIMA 信号，利于攻击者建模。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）稀疏模式重构的有效性（Figure 13）
- 当 page size = 20 tokens 时：
  - 无重构情况下 Healthcare PASR 下降至 **57.45%**
  - 启用重构后恢复至 **~85%**
- 结论：**分布一致性先验显著提升了从粗粒度页访问中恢复细粒度稀疏性的能力**

#### （2）离线分析开销（Table 3）
| 攻击类型 | 所需样本数 | 离线耗时 |
|--------|------------|---------|
| QAI | 5,000 queries | ~0.8–1.1 小时 |
| ATR | 2,000–3,000 responses | ~12.7–19.5 小时 |

> 虽有一定成本，但属一次性投入，可复用于大量在线攻击。

#### （3）无模型访问下的攻击效果
当无法获取模型权重时，采用“交互式 profiling”（发送已知 query 并记录 SIMA trace）：
- QAI-PASR: **90.97%**（相比 replica-based 仅下降 1.28%）
- ATR-DASR: **86.65%**（下降 1.54%）
> 表明 SPARLEAK 即使在黑盒场景下依然有效。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **稀疏注意力引入了新型微架构侧信道 SIMA**  
   输入相关的 KV cache 访问模式会在 GPU 的 L2 cache 和 TLB 中留下可观测痕迹，构成强大的信息泄漏源。

2. **SIMA 泄漏贯穿整个推理流程**  
   - Prefill 阶段泄漏反映输入语义结构（可用于属性推断）
   - Decoding 阶段泄漏随生成过程演化（可用于 token 重建）

3. **即使在噪声和空间聚合下，仍能高效重构稀疏模式**  
   利用稀疏注意力固有的统计规律（如长尾分布、局部聚集），可通过概率建模恢复判别性特征。

4. **攻击具有强实用性与隐蔽性**  
   - 仅需用户级权限
   - 引入延迟 <8%，低于正常服务波动范围
   - 成功率高达 **90.9%（QAI）和 87.3%（ATR）**

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|-------|------|
| **依赖固定 prompt 模板** | 攻击假设存在公共 system prompt，便于离线 calibrate 页面映射；若每次 prompt 完全随机，则 page-to-probe 映射难以复用。 |
| **需要一定程度的配置可见性或交互式 profiling** | 若模型配置完全闭源且不允许任意 query 输入，则初始训练数据收集困难。 |
| **对 page 内 token 分布假设较强** | 重构依赖“高频页内 token 更集中”的经验规律，在极端均匀分布下可能失效。 |

---

### 🔮 未来工作方向

1. **开发低成本防御机制**
   - 如注入可控随机性扰动稀疏模式（实验显示有效但损害生成质量）
   - 探索硬件级隔离（cache/TLB partitioning）、调度 padding 等手段

2. **扩展至其他稀疏结构**
   - 如 MoE routing、pruned FFN layers 是否也产生类似 SIMA？

3. **防御-攻击博弈建模**
   - 构建 formal framework 来量化稀疏效率与隐私泄露之间的 trade-off。

4. **跨设备迁移攻击研究**
   - 是否可在一台设备上训练攻击模型，迁移到另一台同型号 GPU 上使用？

---

> 📌 **一句话总结**：  
> SPARLEAK 揭示了现代 LLM 中广泛应用的 **sparse attention 机制在共享 GPU 上带来了严重的微架构隐私风险**，提出首个利用 **SIMA 侧信道** 实现 **query attribute inference 与 autoregressive token recovery** 的统一攻击框架，实验证明其在真实部署条件下可达 **超过 90% 的属性识别准确率**，凸显了在部署稀疏注意力 LLM 系统时必须考虑此类新型泄漏的重要性。

</details>

---

### 15. [ID Balancing: Stable Training of Extremely Sparse MoE via PID-Based Load Control](https://arxiv.org/abs/2609.39137)

**Authors**: Peng Jin, Zihan Qiu, Zekun Wang, Bo Zheng, Yang Xu, Tian Xie, Xiao Li, Huaqing Zhang, Haoran Lian, Rui Men, Dayiheng Liu  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.39137v1  

#### Abstract
Scaling Large Language Models (LLMs) via Mixture-of-Experts (MoE) enables massive parameter growth with nearly constant per-token computation. However, further scaling the parameter count requires increasingly sparse routing, where expert load imbalance becomes more severe. This imbalance reduces pa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ID Balancing: Stable Training of Extremely Sparse MoE via PID-Based Load Control

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Mixture-of-Experts (MoE)** 大模型中，随着专家数量 $E$ 增加而激活专家数 $K$ 固定（即路由越来越稀疏），**expert load imbalance**（专家负载不均衡）问题愈发严重。这导致：
- **过载专家**（overloaded experts）拖慢计算；
- **欠载专家**（underloaded experts）训练不足，参数利用率低；
- 影响训练稳定性，成为扩展更大、更稀疏 MoE 模型的瓶颈。

传统基于 **auxiliary loss** 的负载均衡方法会引入优化冲突，影响语言建模性能。

### 提出了什么新方法或新思路
本文提出 **ID Balancing**，一种无需辅助损失（auxiliary-loss-free）的负载控制方法，其核心思想是将 MoE 负载均衡问题建模为一个 **控制理论中的 PID 控制问题**，并据此设计更新策略。

#### 创新点：
- **统一视角**：首次将现有的无辅助损失方法（如 DeepSeek 的 loss-free 和 Kimi K3 的 Quantile Balancing）解释为 **不完整的 PID 控制器**：
  - DeepSeek loss-free → **固定步长的积分控制器**（fixed-step integral control）
  - Quantile Balancing → **广义比例控制器**（generalized proportional control）
- **提出 ID Balancing**：结合了 **积分项（Integral）** 和 **导数项（Derivative）**，形成 **ID 控制器**，**省略了比例项（Proportional）**。
  - **Magnitude-aware Integral Term**：积分项的更新幅度与负载误差大小成正比，大误差时强修正，接近平衡时小更新。
  - **Worsening-gated Derivative Term**：仅当不平衡情况恶化时（即误差变大），才激活导数项进行额外修正。
  - **Zero-mean Centering**：每次更新后对专家偏置（bias）做零均值化，防止整体漂移，不影响路由决策。

### 相比现有方法的优势
- **更强的负载控制能力**：尤其在高稀疏度（如 Top-3-of-768）下显著优于所有基线。
- **保持语言建模性能**：不添加 balancing gradient 到训练目标，避免了与 LM loss 的优化冲突。
- **稳定可扩展**：在从 18.9B 扩展到 69.9B 参数时，负载控制效果几乎不变。
- **高效轻量**：仅需每层 $O(E)$ 的 token count 反馈，计算开销极小。

---

## 2. 核心实验方法和设置

### 数据集
- **预训练数据**：使用大规模文本语料进行预训练。
  - 小模型：120B tokens
  - 下游评估模型：560B tokens
- **下游评估基准**：涵盖多个标准评测集：
  - **知识理解**：MMLU, MMLU-Pro, SuperGPQA
  - **数学推理**：MATH, GSM8K
  - **综合推理**：BBH
  - **多语言理解**：MMMLU
  - **代码生成**：EvalPlus, MultiPL-E

### 实验设置
- **模型架构**：基于 Qwen-3.8-Next 架构的 decoder-only MoE 模型。
- **MoE 配置**：
  - 专家数 $E = 768$ 或 $256$
  - 激活专家数 $K = 3, 5, 8, 10$
  - 总参数量：18.9B, 24.8B, 69.9B
- **训练配置**：
  - 学习率峰值：$2.54 \times 10^{-3}$，余弦退火至 $3 \times 10^{-5}$
  - 全局 batch size：1024
  - 训练步数：30k 步（对应 120B tokens）

### 评估指标
- **负载均衡指标**：
  - **MaxVio**：最大相对过载程度（越小越好）
  - **MinVio**：最大相对欠载程度（越小越好），达到 1 表示有专家完全未被使用
- **模型性能指标**：
  - **LM loss**：语言建模损失
  - **下游任务平均分**（Avg. Score）

### 基线方法对比
- **Auxiliary loss**：经典辅助损失法（$\alpha=0.05$）
- **DeepSeek loss-free**：基于符号的偏置更新
- **Quantile Balancing**（Kimi K3）：基于分位数的目标偏置设定
- **ID Balancing (ours)**：本文方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在 Top-3-of-768 设置下（最高稀疏度）：
| 方法 | Worst MaxVio | Training Avg. MinVio |
|------|--------------|------------------------|
| DeepSeek loss-free | 211.19 | 0.7441 |
| Quantile Balancing | 31.11 | 0.6373 |
| **ID Balancing (ours)** | **15.33** | **0.5480** |

👉 **相比最佳基线（Quantile Balancing），Worst MaxVio 降低超过 50%，MinVio 降低约 12%。**

#### 模型规模扩展实验（Top-10-of-768）：
- 从 **18.9B → 69.9B** 参数（活跃参数 1.03B → 3.2B）
- **ID Balancing 的 Worst MaxVio 几乎不变**，且比 auxiliary loss 基线 **低 89.6%**

#### 下游任务性能（Top-10-of-768, 69.9B）：
| 方法 | Average Score |
|------|---------------|
| Auxiliary loss | 58.12 |
| DeepSeek loss-free | 58.45 |
| Quantile Balancing | 58.56 |
| **ID Balancing (ours)** | **58.66** |

👉 在显著提升负载均衡的同时，**下游性能仍具竞争力，甚至略有领先**。

### 与其他方法对比结果
- **vs Auxiliary loss**：
  - 负载控制显著更好（MaxVio/MinVio 更低）
  - LM loss 更低，无优化冲突
- **vs DeepSeek loss-free**：
  - 负载控制全面碾压，尤其在高稀疏度下
  - 避免了固定步长带来的 overshooting 问题
- **vs Quantile Balancing**：
  - 在 Top-3 等极端稀疏场景下，**MaxVio 更低**
  - 保持更平滑的 bias 轨迹，有利于 checkpoint merging

### 消融实验结果
#### 积分增益 $K_i$ 实验（$K_d=0$）：
- $K_i = 6 \times 10^{-3}$ 时取得最佳平衡：
  - 过小（$3\times10^{-3}$）：早期纠正慢
  - 过大（$9\times10^{-3}$）：后期 MTP 模块 MaxVio 暴涨至 35.56
- 最终选择 $K_i = 6 \times 10^{-3}$

#### 导数增益 $K_d$ 实验（$K_i=6\times10^{-3}$）：
- 添加导数项后，**早期 overload 显著下降**（Mean MaxVio 从 2.0020 → 1.9330 @ 0-1k steps）
- 默认 $K_d = 6 \times 10^{-3}$ 在 backbone 控制和 MTP 平衡间取得最优权衡

---

## 4. 关键结论和发现

### 主要发现
1. **PID 控制视角有效统一了解释现有方法**，并指导了新算法设计。
2. **ID Balancing 在极端稀疏 MoE 中表现出卓越的负载控制能力**，尤其在 Top-3 等高稀疏场景下优势巨大。
3. **方法可扩展性强**：在从 18.9B 扩展到 69.9B 参数时，负载控制性能几乎不受影响。
4. **保持高性能**：在大幅改善负载均衡的同时，**语言建模和下游任务性能与最佳基线相当甚至更优**。
5. **适用于持续预训练（continued pretraining）**：通过降低增益（$K_i=K_d=6\times10^{-6}$），可在不破坏适应性的前提下维持负载控制。

### 方法的局限性
- 当前评估集中在 **decoder-only MoE 架构** 和 **固定 Top-K 路由** 上。
- 对其他路由机制（如 Expert Choice）或其他架构（如 encoder-decoder）的泛化性有待验证。
- 导数项的“恶化门控”机制虽经验上有效，但缺乏严格的理论分析。

### 未来工作方向
- 将 ID Balancing 应用于更多类型的 MoE 架构和路由策略。
- 探索更复杂的控制策略（如完整 PID 或自适应增益）。
- 在更大规模（千亿级以上）模型上验证其有效性。
- 理论分析其收敛性和稳定性边界。

---

> **总结**：  
> **ID Balancing** 通过引入 **控制理论中的 PID 视角**，提出了一种简单、高效、强大的无辅助损失负载均衡方法。它在 **极端稀疏 MoE 场景下实现了前所未有的负载控制效果**，同时保持了优异的语言建模性能，为训练更大、更稀疏的 MoE 模型提供了可靠的技术路径。

</details>

---

### 16. [MoE-CORE: Coordinated Expert Offloading and Residency for Memory-Constrained MoE Inference](https://arxiv.org/abs/2610.01950)

**Authors**: Ke Yang, Yongji Gao, Xushi Li, Kui Luo, Sicheng Zhang, Tianming Zhou, Keyi Liu, Shufang Lu, Aoxuan Chen, Jie Meng, Jingchun Gao, Dan Li, Xinkai You, Dan Li, Zhixiang Xia, Yan Shi, Yang Liu, Yanjia Zeng, Liangjun Feng  
**Category**: cs.DC  
**Published**: 2026-10-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.01950v1  

#### Abstract
Sparse expert activation reduces MoE models' computation, yet expert weights can exceed limited device memory. Offloading makes inference feasible on a compact AI appliance but exposes host-to-device transfers to the inference path. We present MoE-CORE, a system that coordinates expert offloading an...

---

### 17. [Amortized Data Borrowing with Exchangeability-Aware Neural Posterior Estimation](https://arxiv.org/abs/2609.38902)

**Authors**: Chin-Hung Huang, JooChul Lee, Huan He  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.38902v1  

#### Abstract
Augmenting small concurrent studies with external or historical cohorts is attractive in drug development, where enrollment is slow, follow-up is expensive, and closely related trial or real-world data are often already available. Bayesian dynamic borrowing (BDB) provides a principled framework for ...

---

### 18. [From Spectra to Joint Schedules in LLM Pre-training: 3+3(+2) Scaling-Law Regimes](https://arxiv.org/abs/2609.40148)

**Authors**: Yichen Wang, Fanghui Liu, Yudong Chen  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.40148v1  

#### Abstract
Power-law learning curves are often treated as fixed properties of a model and its data, although learning-rate and batch-size schedules can change the observed loss. We study this dependence in noisy online SGD with linear random features. Conditional on the representation, an exact Volterra equati...

---

### 19. [Is Weight Tying Still Beneficial for Decoder-Only LLMs in Private Settings Under DP-SGD?](https://arxiv.org/abs/2609.40335)

**Authors**: Razan El Mais, Ali Chehab, Ibrahim Issa, Razane Tajeddine  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.40335v1  

#### Abstract
Differentially Private Stochastic Gradient Descent (DP-SGD) is a leading approach for privacy-preserving fine-tuning of large language models (LLMs). Many decoder-only LLMs employ weight tying between input and output embeddings, a design choice originally introduced for parameter efficiency and imp...

---

### 20. [Learning to Ask: Information Acquisition for SLM-LLM Collaboration, under a budget](https://arxiv.org/abs/2610.01236)

**Authors**: Yongjun Kim, Xiaoxiao Li, Jaeho Lee  
**Category**: cs.AI  
**Published**: 2026-10-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.01236v1  

#### Abstract
Collaboration between a small language model (SLM) and a large language model (LLM) offers an opportunity to combine the efficiency of smaller models with the strong reasoning capabilities of larger ones. Existing approaches primarily frame such collaboration as a computation allocation problem, det...

---

### 21. [ITC-MoE: Importance-guided Token-aware Compression for MoE Diffusion Language Models](https://arxiv.org/abs/2610.01296)

**Authors**: Lianjun Liu, Shipeng Li, You Huang, Weiqi Yan, Mingte Qiu, Huazhong Liu, Xiaofeng Zhu, Yunshan Zhong  
**Category**: cs.AI  
**Published**: 2026-10-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.01296v1  

#### Abstract
Mixture-of-Experts (MoE) Diffusion Language Models (DLMs) offer flexible parallel decoding and increased model capacity, but their large number of expert parameters incurs substantial computation and storage costs. Existing low-rank MoE compression methods largely rely on static factorization and fi...

---

### 22. [SpikeMoE: Brain-Inspired Competitive Routing for Flexible Spiking Mixture-of-Experts](https://arxiv.org/abs/2610.01418)

**Authors**: Xiaoli Liu, Yujie Liang, Jialin Li, Malu Zhang  
**Category**: cs.AI  
**Published**: 2026-10-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.01418v1  

#### Abstract
Spiking Neural Networks (SNNs) enable event-driven computation through biologically inspired dynamics at the neuronal scale, while Mixture-of-Experts (MoE) perform conditional computation through expert selection at the model scale. Integrating their strengths offers potential for flexible neural ar...

---

### 23. [DRelay: Global Draft Context for Prefix-Aware Parallel Speculative Decoding Repair](https://arxiv.org/abs/2610.01439)

**Authors**: Zhuoyu Wang, Junnan Huang, Xinyu Chen  
**Category**: cs.AI  
**Published**: 2026-10-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.01439v1  

#### Abstract
Parallel drafting reduces the drafting overhead of speculative decoding for large language models (LLMs), but its gains remain limited by the accepted prefix length. Even when the correct token is present in the candidate pool, a single early selection error prevents subsequent predictions from bein...

---

### 24. [UBTree: Parallel Tree Drafting via Unigram and Bigram Models for Speculative Decoding](https://arxiv.org/abs/2609.39972)

**Authors**: Chumeng Liang, Linxuan Wang, Xinyu Peng, Huabin Liu, Yuxin Chen, Ge Liu, Guang Lin, Qifan Song, Jianguo Li  
**Category**: cs.CL  
**Published**: 2026-10-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.39972v1  

#### Abstract
Speculative decoding accelerates language model inference by verifying multiple draft tokens in a single target-model pass. Recent parallel drafters have achieved breakthrough performance in frontier production models, but their effectiveness deteriorates as the entropy of target distributions incre...

---

### 25. [Provable Test-Time Scaling for Beam Search in LLM Reasoning](https://arxiv.org/abs/2609.38672)

**Authors**: Qijia He, Yu Huang, Yuan Cheng, Yuxin Chen, Yingbin Liang  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.38672v1  

#### Abstract
Beam-search-based test-time methods provide an effective way to improve large language model (LLM) performance on long-horizon generation by pruning invalid reasoning paths early, leading to significantly improved reasoning efficiency and more favorable test-time cost scaling. Despite strong empiric...

---

### 26. [Explicit Trajectory Diversity for RL-Based Post-Training of LLM Agents](https://arxiv.org/abs/2609.38805)

**Authors**: Huaiyu Fu, Heng Cao, Hao Wang, Jian Ya, Tao Chen  
**Category**: cs.LG  
**Published**: 2026-10-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.38805v1  

#### Abstract
LLM agents often admit multiple high-quality solutions to the same task, differing in reasoning structure, tool-use pattern, or interaction trajectory. Yet existing notions of diversity in LLM post-training are mostly implicit, arising from general stochasticity and regularization mechanisms rather ...

---

### 27. [ProtoFlow: Prototype-Guided Flow Matching for Multivariate Time Series Forecasting](https://arxiv.org/abs/2610.01320)

**Authors**: Shibo Feng, Wanjin Feng, Yang Qiu, Deheng Ye, Peilin Zhao, Chunyan Miao  
**Category**: cs.AI  
**Published**: 2026-10-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.01320v1  

#### Abstract
Generative modeling has shown strong promise for multivariate time mseries (MTS) forecasting, especially scale to high-dimensional settings. Diffusion-based methods achieve competitive performance but typically require many sampling steps at inference. VAE-based non-iterative forecasting frameworks ...

---

### 28. [Rethinking Probability-Based Reinforcement Learning From Posterior Concentration](https://arxiv.org/abs/2610.01458)

**Authors**: Shiu-Hong Kao, Yubo Zhao, Zhenyu Tian, Pengzhan Sun, Yicong Li, Angela Yao  
**Category**: cs.AI  
**Published**: 2026-10-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.01458v1  

#### Abstract
Verifier-free reinforcement learning with probability-based rewards offers a promising way to train LLMs on general reasoning tasks where external verifiers are unavailable. Yet the reliability of these rewards, especially in long-horizon reasoning, remains underexplored. This work identifies a leng...

---

### 29. [Recovering Off-Policy Supervision for Speculative Decoding](https://arxiv.org/abs/2609.38795)

**Authors**: Jungseob Lee, Chanjun Park, Sugyeong Eo, Hyeonseok Moon  
**Category**: cs.CL  
**Published**: 2026-10-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.38795v1  

#### Abstract
Block drafters for speculative decoding are commonly trained on corpora written by external models, where a single off-policy token invalidates supervision for all subsequent slots in a block. Existing approaches discard these divergent slots, resulting in severe supervision loss. To resolve this pr...

---

### 30. [Leto: Fast In-Place Recovery for LLM Training on Surviving Hardware](https://arxiv.org/abs/2610.00687)

**Authors**: Geon-Woo Kim, Joon Ha Kim, Daehyeok Kim  
**Category**: cs.DC  
**Published**: 2026-10-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.00687v1  

#### Abstract
Hardware-operable failures (HOFs) interrupt large language model (LLM) training but permit recovery on the same hardware without reset, repair, or replacement. Existing recovery systems nevertheless reload checkpoints, recompute lost progress, and rebuild process state, idling GPUs that could otherw...

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

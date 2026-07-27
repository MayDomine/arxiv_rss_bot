# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-27 09:04:23 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters](https://arxiv.org/abs/2607.22432)

**Authors**: Zhiwen Mo, Yu Cheng, Lei Wang, Zhengju Tang, Lei Xu, Guoyu Li, Yuqi Dong, Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, Luo Mai, Zhi Yang, Wayne Luk, Hongxiang Fan  
**Category**: cs.DC  
**Published**: 2026-07-27  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2607.22432v1  

#### Abstract
Recent GPU programming frameworks such as Triton, TileLang, and CUDA Tile adopt tiles as first-class primitives, making tile-centric programming the prevailing approach for high-performance GPU kernels. Performance-analysis tooling has not followed: programmers still rely on coarse roofline bounds, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TILESIGHT**

## **1. 主要贡献和创新点**

### **解决的问题**
现代 GPU 编程框架（如 Triton、TileLang、CUDA Tile）已广泛采用 **tile** 作为核心编程原语，用于构建高性能内核（kernels）。然而，现有的性能分析工具（如 Roofline 模型、ML 预测器、Profiler）存在严重抽象不匹配：
- **Roofline 模型** 只能提供粗粒度瓶颈估计，无法区分 L2 缓存未命中与共享内存冲突。
- **ML-based 预测器**（如 NeuSight、PipeWeave）需要大量训练数据，缺乏可解释性。
- **Profiler 工具**（如 Nsight Compute）是后验的，依赖实际运行，无法预测未执行配置的性能。

这导致开发者在优化融合内核（fused kernels）和分布式推理时，只能依赖试错或黑盒调优，效率低下。

---

### **提出的新方法**
论文提出了 **TILESIGHT** —— 一个从第一性原理出发的、以 tile 为中心的 **统一分析型 GPU 性能模型**，将 tile 从编程原语提升为 **分析原语**。

#### **核心思想**
利用 tile 的三大特性进行建模：
1. **Deterministic**：给定 tile 配置（形状、流水线深度、内存布局），其资源使用完全确定，无需模拟。
2. **Composable**：tile 信息可分层组合（intra-tile → inter-tile → cross-device）。
3. **Portable**：tile 抽象跨 GPU 架构通用（NVIDIA A100/H200/B200/B6000, AMD MI210）。

#### **三层统一建模框架**
| 层级 | 功能 | 关键机制 |
|------|------|----------|
| **Intra-tile** | 单个 tile 内部资源分解 | 将每个 tile 表达为 `Resource Vector`：`<TC, CUDA, SFU, TMEM, SMEM, L1.5, L2, DDR, Net>`，基于操作、footprint 和 placement 计算各硬件管道时间 |
| **Inter-tile** | tile 间调度与缓存建模 | 基于 producer-consumer 依赖关系进行拓扑排序搜索，寻找最优合法流水线重叠；提出 **Tile Reuse Distance** 分析多级缓存命中率 |
| **Cross-device** | 跨设备通信建模 | 将远程 tensor 访问视为 placement 扩展，通过 α-β 模型计算通信阶段成本，注入 `Net` 资源维度 |

---

### **相比现有方法的优势**
| 特性 | TILESIGHT | Roofline | ML-based (NeuSight/PipeWeave) | Profiler |
|------|---------|--------|-----------------------------|--------|
| ✅ 无需训练/运行 | ✔️ | ✔️ | ❌ | ❌ |
| ✅ 流水线感知 | ✔️ | ❌ | ❌ | ✔️（后验） |
| ✅ 缓存命中预测 | ✔️（L2/L1.5） | ❌ | ❌ | ✔️（后验） |
| ✅ 支持融合内核 | ✔️ | ❌ | ⭕（有限） | ✔️（后验） |
| ✅ 支持分布式 | ✔️ | ❌ | ⭕ | ✔️（后验） |
| ✅ 计算-通信重叠 | ✔️ | ❌ | ❌ | ✔️（后验） |
| ✅ 可解释性 | ✔️（白盒） | ✔️ | ❌（黑盒） | ✔️ |

> **TILESIGHT 是首个在无需运行或训练的前提下，同时支持单卡到集群、融合内核、缓存与通信建模的白盒性能模型。**

---

## **2. 核心实验方法和设置**

### **实验平台**
- **GPU 架构**：NVIDIA A100, H200, B200, B6000, H200-NVL, AMD MI210
- **互联技术**：NVLink, PCIe, InfiniBand, NVSwitch
- **软件栈**：CUDA 12.9/13.1, ROCm 6.2, Triton, TileLang, vLLM 0.19.0

---

### **工作负载（Workloads）**
| 类别 | 具体任务 | 数量 |
|------|--------|------|
| **单算子** | BF16/FP16 Tensor-Core GEMM | 703 种形状 |
| **持久化内核** | GEMM 缓存行为扫描 | 4,680 个案例 |
| **分布式通信** | AllGather, AllReduce, ReduceScatter, All-to-All | 152 个纯集体通信 |
| **融合计算-通信** | AllGather+GEMM, GEMM+ReduceScatter, Ulysses Attention | 152 个融合内核 |
| **端到端服务** | vLLM 上 Qwen, Llama, DeepSeek 系列模型解码 | 166 个配置（含 MoE） |

---

### **评估指标**
- **MAPE**（Mean Absolute Percentage Error）：单算子延迟预测误差
- **wMAPE**（Weighted MAPE）：加权平均绝对百分比误差（用于分布式和端到端）
- **L2 Hit Rate MAE**：L2 缓存命中率预测的平均绝对误差（单位：百分点）
- **诊断准确率**：能否正确识别瓶颈并指导优化
- **成本模型有效性**：在调度搜索中剪枝 95% 候选者后仍能达到接近最优性能

---

### **基线方法对比**
| 基线 | 类型 | 是否支持训练 | 是否支持分布式 | 是否可解释 |
|------|------|------------|--------------|-----------|
| **Roofline** | 分析模型 | 否 | ❌ | ✔️ |
| **NeuSight** | ML-based | 是（需 per-arch 训练） | ❌ | ❌ |
| **PipeWeave** | 混合模型 | 是 | ✔️（有限） | ❌ |
| **GenZ** | 分析模型 | 否 | ✔️ | ✔️ |

> 注意：PipeWeave 在 H200/B200 上无原生支持，使用 H800 模型近似。

---

## **3. 主要实验结果和性能指标**

### **单算子 GEMM 延迟预测（703 个形状）**
| 方法 | A100 | B200 | B6000 | H200 | MI210 | **Pooled MAPE** |
|------|------|------|-------|------|-------|----------------|
| **TILESIGHT** | 14.9% | 18.7% | 23.4% | 21.9% | 23.4% | **12.35%** |
| PipeWeave | 31.7% | 16.5% | 14.3% | 24.5% | 25.5% | 21.97% |
| NeuSight | 28.1% | 44.9% | 42.2% | 17.8% | 26.4% | 32.95% |
| Roofline | 33.4% | 33.7% | 45.7% | 33.8% | 38.8% | 33.85% |
| GenZ | 34.9% | 22.4% | 46.3% | 48.4% | 40.4% | 34.89% |

> ✅ **TILESIGHT 在所有架构上均显著优于基线，尤其在新架构（B200/B6000/H200）上优势明显**  
> ✅ **无需训练即可实现跨架构迁移，而 NeuSight 在 A100 上因训练数据过拟合表现好，但在新架构上退化**

---

### **L2 缓存命中率预测（4,680 个持久化 GEMM）**
| GPU | MAE (pp) | MAPE (%) |
|-----|----------|----------|
| **A100** | 1.46 | 2.33% |
| **H200** | 0.88 | 1.50% |
| **B200** | 1.05 | 1.52% |
| **B6000** | 0.78 | 1.09% |

> ✅ **命中率预测误差稳定在 ~1 个百分点以内，远超 Roofline 等无法预测缓存行为的方法**

---

### **分布式内核预测（H200×8 / B200×8）**
| 任务 | TILESIGHT (wMAPE) | GenZ | PipeWeave |
|------|-------------------|------|-----------|
| **纯集体通信** | **12.22%** | 20.82% | 65.72% |
| **融合计算-通信** | **14.83%** | ❌ | ❌ |

> ✅ **TILESIGHT 在纯通信和融合内核上均大幅领先，且唯一支持融合内核建模**

---

### **端到端 vLLM 解码吞吐预测（166 配置）**
| 方法 | **Overall wMAPE** | MoE 支持 |
|------|------------------|----------|
| **TILESIGHT** | **13.52%** | ✔️ |
| PipeWeave | 31.84% | ❌ |

> ✅ **TILESIGHT 在密集和 MoE 模型上均表现优异，而 PipeWeave 不支持 MoE 且在大 batch 场景下因超出训练范围而失败**

---

### **消融与应用实验**
#### **(1) 成本模型剪枝能力（TileLang）**
- 保留预测前 5% 的调度方案
- 平均达到穷举搜索最优性能的 **99.66%**
- 剪枝率高达 **95%**

> ✅ 可高效集成至编译器调度器，替代昂贵的 autotuning

#### **(2) 性能诊断能力**
| 内核 | 问题 | 优化 | 加速比 |
|------|------|------|--------|
| ReLU (MI210) | Indirect addressing | Unroll addr | 1.27× |
| Avg_Pool (MI210) | 未重叠 + 间接寻址 | 小 tile + 解除寻址 | 2.00× |
| MLA (MI210) | 寄存器分配、SMEM 冲突 | Register alloc + 更大 tile | **8.97×** |

> ✅ TILESIGHT 能精准定位瓶颈并指导优化，实现显著加速

---

## **4. 关键结论和发现**

### **主要发现**
1. **Tile 是统一性能建模的理想抽象单元**：其 deterministic、composable、portable 特性使得从单核到集群的统一建模成为可能。
2. **白盒模型可媲美甚至超越黑盒 ML 模型**：TILESIGHT 无需训练即实现高精度预测，且具备完全可解释性。
3. **缓存与通信建模必须与调度耦合**：传统 flat 带宽假设失效，**Tile Reuse Distance** 和 **α-β 通信分解** 是关键。
4. **跨设备通信应视为 placement 扩展**：统一的 `Resource Vector` 框架自然支持计算-通信重叠建模。

---

### **局限性**
1. **假设 tile 均匀执行**：忽略 SM 间失步，导致大 K GEMM 的 L2 命中率预测偏乐观。
2. **不支持数据依赖控制流**：仅适用于规则 tile 结构程序。
3. **未建模指令级细节**：如 warp 调度、寄存器分配等由编译器决定的行为。
4. **多芯片架构细节缺失**：如 B200 的 SM-HBM 亲和性未建模。

---

### **未来工作方向**
1. 引入更精细的 SM 执行模型以提升 L2 命中率预测准确性。
2. 扩展至非规则、动态 tile 程序的支持。
3. 与 Triton、TileLang 等 DSL 编译器深度集成，实现自动优化闭环。
4. 探索在训练场景中的应用，支持更复杂的并行策略建模。

---

> **总结**：TILESIGHT 展示了“从第一性原理出发”的 tile-centric 建模范式的强大潜力——它不仅实现了高精度、可解释、跨架构、跨尺度的性能预测，更为下一代 GPU 编程与优化工具链提供了坚实的理论基础。该工具将在发表后开源。

</details>

---

### 2. [RIS-Kernel: A Model-Agnostic Architecture for Long-Context LLM Inference via Sparse Attention](https://arxiv.org/abs/2607.21927)

**Authors**: Anderson R. Santos  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.21927v1  

#### Abstract
Full self-attention in large language models scales as O(N^2), which limits long-context document analysis to 65,536 tokens and requires costly GPU clusters. The Reduced Interaction Sampling (RIS) inference engine addresses this constraint as a model-agnostic architecture. Without modifying weights,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RIS-Kernel: A Model-Agnostic Architecture for Long-Context LLM Inference via Sparse Attention

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）中的 **full self-attention** 计算复杂度为 $O(N^2)$，在处理长上下文（如超过 32k 或 65k tokens）时面临严重的内存和计算瓶颈，导致需要昂贵的 GPU 集群支持。这限制了大多数研究机构对长文本进行深度分析的能力。

本文旨在解决以下核心问题：
- 如何在不修改模型权重的前提下，实现高效、低成本的 **long-context LLM inference**？
- 如何突破硬件限制，在普通 CPU 服务器上完成 65k+ token 的推理任务？

### 提出的新方法与思路
作者提出了 **RIS-Kernel** —— 一种 **model-agnostic** 的推理架构，基于 **Reduced Interaction Sampling (RIS)** 机制，通过稀疏注意力（sparse attention）降低计算复杂度至 $O(N \log N)$。

其核心思想是：
- 在运行时动态注入稀疏注意力掩码（mask），而非修改模型结构或重新训练。
- 引入两种采样模式：
  - **RIS-Stochastic Mode**：全局随机抽样，适用于高密度下的噪声正则化。
  - **RIS-Structural Mode**：基于 block-clique 结构的局部社区保留采样，优先保证关键实体邻域的完整覆盖。

此外，提出三项关键技术组件：
1. **Streaming Mask Generation**：流式生成稀疏掩码，避免一次性分配大张量造成 OOM。
2. **Pre-Fusion Unified Softmax (PFUS)**：统一归一化所有选中 token（无论来自局部窗口还是随机采样），防止信号衰减。
3. **Dynamic RoPE Scaling**：支持在推理时动态切换 RoPE 扩展策略（如 linear interpolation 和 YaRN），适配超长序列的位置编码需求。

### 相比现有方法的优势
| 方面 | RIS-Kernel | 其他方法（如 BigBird、Longformer） |
|------|-----------|-------------------------------|
| **是否需重训练** | ❌ 不需要（model-agnostic） | ✅ 通常需要微调或特定结构设计 |
| **灵活性** | ✅ 支持任意模型（已验证于 Qwen2、TinyLlama） | ⚠️ 固定几何结构（strided/block-local）易漏检 |
| **硬件依赖** | ✅ 可在无 GPU 的 CPU 上运行（16–128GB RAM） | ⚠️ 多数依赖 GPU 加速 |
| **抗噪能力** | ✅ 稀疏本身作为正则器，过滤序列级噪声 | ⚠️ 密集注意力易受干扰 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- 自建合成数据集：将四篇不同生物学领域的科学论文拼接成一个长文档：
  - `ajinshanensis`：Acetilactobacillus jinshanensis 基因组特征
  - `aom`：厌氧海洋甲烷氧化
  - `genppi`：蛋白质相互作用网络预测
  - `meta`：Jatai 蜂幼虫食物宏基因组学
- 移除标题页、摘要等元信息，仅保留正文内容。

### 实验设置
#### 模型
- 主要测试模型：**Qwen2-1.5B-Instruct**
- 辅助验证模型：**TinyLlama-1.1B**

#### 上下文长度
- **Experiment A (控制实验)**：32,768 tokens（Qwen2 原生位置上限）
- **Experiment B (可扩展性实验)**：65,536 tokens（超出原生限制 2×）

#### 评估协议
- 构造两个 QA 数据集：
  - 平衡的 32-question 集合（用于 32k 实验）
  - 扩展的 64-question 集合（用于 64k 实验）
- 所有问题均为选择题（A–E），答案明确存在于上下文中。
- 采用 **discriminative logit analysis**：直接比较各选项的 log-probability，取最高者为预测结果，消除生成偏差。

#### 基线方法对比
| 基线类型 | 描述 |
|--------|------|
| **Full Dense Attention** | 在 32k 下可用，作为性能上界（upper bound） |
| **Zero-Context Baseline (w=0)** | 输入为空，衡量模型参数记忆（parametric memory）的影响 |
| **Linear Interpolation (RoPE)** | 传统位置外推方式，作为失败对照 |
| **YaRN Scaling (RoPE)** | 当前先进的位置扩展技术，用于对比有效性 |

#### 硬件环境
全部实验均在 **无 GPU 加速的 CPU 服务器** 上完成：
- **bioinfo 服务器**：Intel Core i7-3770，4核8线程，16GB DDR4 RAM（代表普通桌面级设备）
- **ibteci 服务器**：双路 Xeon，20核40线程，128GB DDR4 RAM（代表高性能学术节点）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Experiment A: 32k 控制实验（有 dense baseline）
| 方法 | 密度 | Seeds | 准确率 | 上下文恢复率 |
|------|-----|-------|--------|-------------|
| Zero-Context (w=0) | – | – | 59.38% | 0% |
| Full Dense Baseline | 100% | 1 | **71.88%** | 100% |
| **RIS-Stochastic** | **1%** | **70–80** | **75.00%** | **125.0%** |
| RIS-Stochastic | 5% | 10 | 71.88% | 100.0% |
| **RIS-Structural** | **1%** | **10** | **68.75%** | **75.0%** |

> 📌 **亮点**：  
> - **RIS-Stochastic 在 1% 密度 + 70 seeds 下准确率达到 75.00%，超越 full attention 基线（71.88%）！**
> - 表明稀疏注意力具有 **正则化效应（regularizer）**：低密度可滤除噪声，反而提升性能。

#### ✅ Experiment B: 64k 可扩展性实验（dense baseline 不可行）
| 方法 | 密度 | Seeds | 准确率 | 对比零上下文增益 |
|------|-----|-------|--------|------------------|
| Zero-Context (w=0) | – | – | 51.56% | – |
| RIS-Stochastic (YaRN) | 5% | 40–60 | 62.50% | +10.94 pp |
| **RIS-Structural (YaRN)** | **1%** | **40** | **65.62%** | **+14.06 pp** |

> 📌 **亮点**：
> - 在 65,536 tokens 下，RIS 成功实现推理，而 dense attention 触发 OOM 错误。
> - RIS-Structural 在极端稀疏（1%）条件下达到最高单次准确率（65.62%），显著优于其他配置。
> - McNemar 检验显示 RIS-Structural 提升 **边际显著**（p = 0.078 < 0.10）。

#### 🔍 消融实验与关键发现
- **稀疏密度与种子数的关系**：
  - 低密度（1%）+ 多种子（>70）→ 更好性能（因覆盖更广且去噪）
  - 高密度（5%）+ 少种子 → 易引入噪声，性能持平 dense baseline
- **Union Coverage 公式验证**：
  $$
  U = 1 - (1 - d)^N
  $$
  解释了为何增加种子会饱和：当 $d=0.01, N=70$ 时，$U \approx 50.5\%$，足以捕获关键锚点并去除一半干扰项。

- **RoPE Scaling 影响巨大**：
  | 方法 | 1% 密度准确率 | 是否有效 |
  |------|---------------|----------|
  | Linear Interpolation | 15.6% | ❌ 完全失效（位置混淆） |
  | **YaRN** | **57.8%** | ✅ 显著恢复位置一致性 |

- **TinyLlama 失败案例说明边界条件**：
  - TinyLlama-1.1B（训练限 2k）在 8k 以上完全崩溃（~6–12% 准确率 ≈ 随机猜测）
  - 所有 seed 和 density 设置下性能不变 → 表明 **位置编码系统已彻底失效**，RIS 无法补救。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **稀疏注意力可在极低密度（1%）下恢复甚至超越 full attention 性能**，尤其在多种子集成下表现出 **正则化优势**。
2. ✅ **RIS-Kernel 是真正 model-agnostic 的推理引擎**，无需修改权重即可插入任意支持 attention layer 的模型。
3. ✅ **长期上下文推理可在纯 CPU 上完成**，使用标准学术硬件（16–128GB RAM），打破对 GPU 集群的依赖。
4. ✅ **RIS-Structural 在极端资源受限场景下表现最优**：它通过 block-clique 结构确保关键 token 邻域始终被保留，适合“proximal anchor recovery”任务。
5. ✅ **位置编码质量决定上限**：即使 RIS 再强大，若 RoPE 扩展不当（如 linear interpolation），也无法恢复语义；**YaRN 是必要前提**。

### 方法的局限性
- **依赖有效的 positional encoding**：对于训练长度远小于推理长度的小模型（如 TinyLlama），RIS 无效。
- **吞吐瓶颈仍在内存带宽**：尽管算法复杂度降至 $O(N \log N)$，但 matmul_av 阶段仍受限于 DDR 到缓存的数据移动成本。
- **未测试更大规模模型**：目前仅验证于 sub-2B 模型，是否适用于 7B+ 待进一步研究。
- **静态稀疏结构**：当前 mask 在预填充阶段固定，未探索动态调整机制。

### 未来工作方向
1. 探索 **adaptive density scheduling**：根据输入内容自动调节稀疏程度。
2. 将 RIS 扩展至 **multi-modal models** 和 **decoder-only streaming generation** 场景。
3. 结合 **quantization** 与 **sparsity** 进一步压缩内存占用。
4. 开发 **lightweight checkpointing mechanism** 以支持超长文本的持续生成。
5. 推动 **standardized long-context benchmarks**，避免 question-set bias。

---

> 💡 **一句话总结**：  
> **RIS-Kernel 证明了“少即是多”——通过精心设计的稀疏注意力机制，可以在消费级 CPU 上实现媲美甚至超越 full attention 的 long-context 推理效果，同时揭示了 positional encoding 在长文本理解中的决定性作用。**

</details>

---

### 3. [Integrated Order Dispatching and Routing for Last-Mile Pickup via Deep Reinforcement Learning](https://arxiv.org/abs/2607.22356)

**Authors**: Yida Xu, Zhaofang Mao, Yuheng Miao, Jiaxin Zhang, Yiting Sun  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.22356v1  

#### Abstract
In recent years, the growing complexity of last-mile pickup operations has increased the need for fast and accurate decision-making on logistics platforms. This challenge is fundamentally driven by two key and tightly coupled decision-making processes: order dispatching and routing. Solving them sep...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**last-mile pickup**（最后一公里取件）中的**集成订单调度与路径规划问题**（integrated order dispatching and routing）。该问题在物流平台中具有高度动态性和复杂性，涉及两个紧密耦合的决策过程：
- **Order dispatching**：将新到达的订单分配给合适的骑手。
- **Courier routing**：为每个骑手生成最优取件路径。

传统方法通常将这两个子问题**分离求解**，忽略了其内在依赖关系，导致次优解；而端到端的深度强化学习（DRL）方法则面临奖励稀疏、训练不稳定、难以扩展至大规模实例等问题。

---

### 提出的新方法与思路
作者提出了一种**集成优化框架**（integrated optimization framework），其核心思想是“**学习型路由预言机 + 启发式调度**”（learned routing oracle + dispatching heuristics）：

1. **分层建模**：
   - 将原问题分解为两个子问题，并分别建立**混合整数线性规划**（MILP）模型，考虑**时变旅行时间**（time-dependent travel times）。

2. **DRL 路由预言机**（DRL Routing Oracle）：
   - 针对 routing 子问题，设计了一个基于深度强化学习的策略网络 **DR-LaCPNet**。
   - 采用 **rollout-baseline policy gradient** 算法进行训练，提升稳定性。

3. **路由感知的调度启发式算法**（Routing-aware Dispatching Heuristic）：
   - 提出 **PP-Greedy-LS** 算法（Position-pool Greedy with Local Search）。
   - 利用训练好的 DRL 路由模型作为“预言机”，快速评估候选骑手的边际成本（marginal cost），实现高质量且实时的调度决策。

---

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|---------|
| **系统级性能** | 首次通过 DRL 实现 dispatching 与 routing 的有效协同，避免了分离优化带来的次优性。 |
| **可扩展性与实时性** | 不直接训练上层调度网络，而是使用轻量级启发式 + DRL 预言机，显著降低计算开销，适用于大规模、高频率调度场景。 |
| **鲁棒性与泛化能力** | DR-LaCPNet 在不同规模实例上均保持稳定推理时间（<0.1秒），且能处理骑手异质性（heterogeneity）、时间窗约束等现实因素。 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用来自 **Cainiao Logistics** 的真实世界数据集 **LaDe-P**。
- 包含超过 **1060万包裹** 和 **61.9万骑手轨迹**，覆盖中国五个城市（杭州、上海、重庆、吉林、烟台），时间跨度六个月。
- 数据包含丰富的时空特征：GPS位置、时间戳、订单类型、区域兴趣（AOI）、天气、交通状况等。

---

### 实验设置
#### 模拟器设计
- 构建了一个**滚动更新的 dispatch instance simulator**（Algorithm 5.1 & 5.2），用于生成可控规模的训练与测试实例。
- 支持动态调整每波次的 `|O_new|`（新订单数）和 `K`（候选骑手数），模拟不同供需比（SDR）下的运行环境。

#### 评估方式
- **Offline Evaluation**：静态评估各方法在固定实例上的表现。
- **Online Rolling-Horizon Simulation**：连续模拟12个调度周期（共1小时），评估长期系统性能。

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Avg. Obj** | 平均目标函数值（加权总时间 + 时间窗惩罚） |
| **Avg. TWVR** | 时间窗违反率（Time Window Violation Rate） |
| **Avg. TWP / Max. TWP** | 平均/最大时间窗惩罚 |
| **Avg. TT** | 平均旅行时间 |
| **Avg. ST** | 平均求解时间（solution time） |
| **Avg. WL** | 工作负载不均衡度（workload imbalance ratio） |

---

### 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **MILP** | 数学规划 | 使用 Gurobi 求解 TDTSPTW-PU 模型，设最大求解时间为300秒 |
| **Greedy** | 启发式 | 最近邻插入法 |
| **TS** | 元启发式 | Tabu Search 算法 |
| **DeepRoute / Graph2Route** | 深度学习 | 基于 Transformer 或 GNN 的路线预测模型 |
| **DMCA** | DRL + 优化 | 基于 DRL 边际成本的最小费用二分图匹配 |
| **PP-Greedy** | 启发式 | 本文提出的无局部搜索版本 |
| **TS-Dispatching** | 元启发式 | 基于 DRL 成本评估的 Tabu Search 调度器 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 3–8）

#### ✅ DRL 路由预言机性能（Table 3）
- 在小规模实例（|O_new| ≤ 8）上，**DR-LaCPNet** 与 MILP 解质量相当，但求解时间仅需 **0.03–0.09 秒**（vs. MILP 的数十至上百秒）。
- 在大规模实例（|O_new| ≥ 9）上，MILP 无法在时限内找到可行解（“-”），而 **DR-LaCPNet 仍能在 <0.3 秒内提供高质量解**，目标值优于所有基线（最高提升达 **15.2%**）。
- 所有规模下，推理时间基本稳定，具备良好可扩展性。

#### ✅ 调度算法离线性能（Table 4）
- **PP-Greedy-LS** 在多数城市和场景下取得最低目标值（如上海 N=27 时从 8.77 降至 8.57，杭州 N=24 时从 10.17 降至 8.78）。
- 虽然 TS-Dispatching 在个别情况下略优，但其求解时间高达 **420秒以上**，远超实际可用范围（通常要求 <60秒）。
- PP-Greedy-LS 在 **质量和效率之间实现了最佳平衡**。

#### ✅ 在线滚动仿真结果（Table 5）
| 方法 | Avg. Obj ↓ | Avg. TWVR ↓ | Avg. TWP ↓ | Avg. ST ↑ |
|------|-----------|------------|------------|----------|
| DMCA | 7.38–38.45 | 0.42–0.84 | 2.14–10.32 | 6.32–13.12 |
| Greedy | 6.58–35.12 | 0.38–0.62 | 1.86–9.44 | 2.22–3.56 |
| PP-Greedy | 6.32–32.68 | 0.31–0.42 | 1.74–8.64 | 3.28–5.12 |
| TS-Dispatching | 6.11–30.24 | 0.22–0.31 | 1.58–7.54 | 62.45–115.42 |
| **PP-Greedy-LS** | **5.82–28.56** | **0.12–0.22** | **1.34–6.84** | **18.56–28.36** |

✅ **PP-Greedy-LS 全面领先**：
- 平均目标值下降 **5–10%**
- 时间窗违反率降低 **40–70%**
- 最大时间窗惩罚显著减少（尾部风险控制更好）
- 工作负载更均衡（WL 下降约 30–50%）

---

### 消融实验结果（Ablation Study, Table 7）

验证了 DR-LaCPNet 各组件的有效性：

| 变体 | 相对性能下降（Obj Imp.） |
|------|------------------------|
| **LaCPNet**（无 DRGAN 编码器） | 最高下降 36.07% |
| **DR-Net**（无 LaCP 解码器） | 最高下降 34.43% |
| **DR-CPNet**（无 Look-Ahead） | 最高下降 33.00% |
| **DR-LaNet**（无 Courier-Personalized） | 最高下降 27.34% |

📌 **关键发现**：
- 所有模块均有贡献，尤其在大规模实例上更为明显。
- **Look-Ahead 机制** 对处理时变交通至关重要。
- **Courier-Personalized 机制** 有效捕捉骑手个体差异（如速度、完成时间偏好）。

---

## 4. 关键结论和发现

### 主要发现
1. **集成优于分离**：dispatching 与 routing 必须联合考虑，否则会因忽略反馈循环而导致系统性能下降。
2. **“预言机 + 启发式”范式有效**：相比端到端 DRL 或纯优化方法，**学习型路由预言机 + 轻量调度启发式** 是解决大规模动态调度问题的实用路径。
3. **PP-Greedy-LS 显著提升服务质量**：
   - 显著降低时间窗违反率和惩罚。
   - 更好地平衡骑手工作负载，防止过载。
4. **Supply-to-Demand Ratio（SDR）影响显著**（Table 6 & Figure 7）：
   - 当 SDR < 1.0 时，增加骑手数量可大幅改善服务指标（边际收益高）。
   - 当 SDR > 1.4 后，进一步增加供给带来的改进趋于平缓。
   - 支持“高峰保供给、低峰优协调”的运营策略。

---

### 方法的局限性
1. **依赖高质量训练数据**：DRL 路由模型需要大量历史轨迹数据进行训练，在冷启动或新区域部署时可能受限。
2. **未完全端到端联合训练**：虽然避免了训练难度，但也限制了全局最优潜力。
3. **假设订单可拆分性有限**：当前模型未考虑订单打包、合并取件等更复杂的操作逻辑。

---

### 未来工作方向（原文 Section 6）
1. 探索 **Hierarchical Deep Reinforcement Learning**，以缓解奖励稀疏问题，并增强跨规模泛化能力。
2. 将 **Look-Ahead** 和 **Courier-Personalized** 机制推广至其他类似问题，如 **last-mile delivery**、**drone routing** 等。
3. 引入 **Heterogeneous Graphs** 来区分订单节点与骑手节点，提升特征表达能力。
4. 结合 **multi-agent RL** 进一步建模骑手间的竞争与协作行为。

</details>

---

### 4. [Scaling Native Multimodal Pre-Training From Scratch](https://arxiv.org/abs/2607.22043)

**Authors**: Haoyuan Wu, Aoqi Wu, Hai Wang, Jiajia Wu, Jinxiang Ou, Bei Yu  
**Category**: cs.CL  
**Published**: 2026-07-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.22043v1  

#### Abstract
Although large language models (LLMs) exhibit remarkable reasoning capabilities, their reliance on text-only pre-training restricts the perception of the multimodal physical world. Native multimodal pre-training avoids this limitation by training models from scratch on multimodal inputs, thereby ach...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Scaling Native Multimodal Pre-Training From Scratch》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的多模态预训练模型（如 CLIP、Flamingo）普遍采用 **late-fusion** 范式，即先分别在文本和视觉上进行独立预训练，再通过轻量级模块对齐。这种范式存在以下问题：
- **优化不对称性**：语言和视觉模态在不同数据分布、目标和优化路径下学习，限制了深层跨模态融合。
- **依赖单模态先验**：严重依赖预训练好的 LLM 和 vision encoder，难以实现真正的“从零开始”统一表征学习。

本文系统研究了 **native multimodal pre-training**（原生多模态预训练）的可扩展性规律，在固定计算预算下探索最优资源分配策略。

---

### 提出的新方法与新思路
1. **解耦式 scaling law 分析框架**  
   首次将语言目标 $L_{\text{text}}$ 和多模态目标 $L_{\text{mm}}$ 分开建模，分析其各自的 compute-optimal scaling behavior，揭示二者遵循不同的分配律。

2. **构建语言-多模态 Pareto 前沿**  
   在统一计算预算 $C_{\text{total}}$ 下，通过调节多模态数据比例 $r$，构建了一个 **language-multimodal Pareto frontier**，为实际训练提供明确配置指导（模型大小、token 数、数据混合比）。

3. **验证 native 多模态训练的正向迁移能力**  
   发现 native 多模态预训练不仅能提升多模态任务表现，还能增强纯文本的空间推理能力（spatial reasoning），并支持 robust 的 multimodal in-context learning。

---

### 相比现有方法的优势
| 维度 | Late-Fusion 方法 | Native Pre-Training（本文） |
|------|------------------|----------------------------|
| 架构设计 | 模块化拼接，两阶段训练 | 单一模型，端到端联合训练 |
| 表征融合深度 | 浅层对齐 | 深层共享参数空间 |
| 可扩展性理论支持 | 成熟的语言 scaling law | 新提出的双目标 compute-optimal law |
| 跨模态迁移 | 有限 | 显著正向迁移至文本任务 |
| 训练效率 | 利用已有 checkpoint 快速启动 | 更高计算成本但更优长期潜力 |

> ✅ 本文填补了 native 多模态模型缺乏系统 scaling law 指导的空白，提供了可预测、可复现的训练基础设施。

---

## 2. 核心实验方法和设置

### 数据集
- **文本数据**：250B tokens，来自网页、书籍、学术论文等。
- **多模态数据**：75B multimodal tokens，由网络爬取的 image-text pairs 和交错图文文档构成。
  - 图像被转换为连续 patch embeddings 输入。
  - 多模态数据占比 $r \in \{0, 0.1, 0.2, 0.3\}$ 进行消融。

---

### 模型架构
- **MoE-based decoder-only Transformer**，无传统 vision encoder。
- 使用单一 patch embedding 层直接将图像映射为序列。
- 不使用 auxiliary loss（如路由损失外的额外监督）。
- 模型规模覆盖从 **71M 到 3B active parameters**。

---

### 实验设置
- **训练方式**：原生多模态预训练，所有参数从头训练。
- **上下文长度**：最大 4K tokens，每张图最多分配 1536 视觉 tokens（few-shot 场景降至 512）。
- **学习率调度**：warmup-stable schedule，global batch size 16M。
- **loss 计算**：仅基于 text tokens 计算训练 loss（vision tokens 被 mask）。

---

### 评估指标
#### 文本能力（16 benchmarks）
- **综合理解**：MMLU-Redux, MMLU-Pro, AGIEval-en, SuperGPQA
- **编程**：HumanEval+, MBPP+
- **数学**：GSM8K, MATH
- **逻辑推理**：BBH, SpatialEval（text-only 子任务）
- **常识推理**：Hellaswag, SIQA, PIQA, WinoGrande
- **知识问答**：NaturalQuestions, TriviaQA

#### 多模态能力（23 benchmarks）
- **综合评测**：MMStar, MMMU, MMMU-Pro, MME, MMBench-en
- **视觉问答（VQA）**：VQAv2, TextVQA
- **STEM 推理**：MathVista, MathVerse, ScienceQA
- **文档理解**：HallusionBench, LogicVista, AI2D, ChartQA
- **计数能力**：CountBench, CountQA
- **空间推理**：RealWorldQA, CV-Bench, OmniSpatial, SEAM, SpatialEval

#### 评估协议
- **选择题**：按 perplexity 最低选答案 → 报告 accuracy。
- **开放生成**：exact-match 或 Pass@1。
- **few-shot 设置**：使用 development set 构造模板，以 interleaved image-question-answer 形式拼接输入。

---

### 基线方法对比
- **Baseline**：纯文本训练模型（$r=0$），用于比较是否因引入多模态而损害语言能力。
- **Late-Fusion 对照组未显式列出**，但文中多次引用 Radford et al. (2021), Liu et al. (2023a) 等作为背景对比。
- 主要对比维度是 **不同 $r$ 下的表现差异** 和 **scaling behavior 的变化趋势**。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）语言性能保持稳定（Fig. 8）
- 固定文本预算 250B tokens，改变 $r$（0 → 0.3）：
  - 所有模型尺度下，**平均文本准确率波动 <1%**。
  - 表明：加入多模态数据不会干扰核心语言能力的学习。

#### （2）空间推理显著提升（Fig. 9）
- 在 SpatialEval 的纯文本抽象空间推理子任务上：
  - $r=0.3$ 模型在 A3B 尺度下比 $r=0$ 提升 **+5.5 pts**。
  - 小模型差距小，大模型优势明显 → **跨模态迁移随 scale 放大**。

#### （3）多模态 in-context learning 成功涌现（Fig. 10–13）
- 在 0-/1-/3-shot 设置下测试：
  - A71M/A128M：few-shot 几乎无增益甚至下降。
  - A3B：3-shot 相较 0-shot 平均增益达 **+2.43 pts**。
  - 性能增益随训练 token 数增加而上升（Fig. 13）→ **in-context learning 是逐步习得的能力**。

#### （4）多模态任务全面领先（Table 8）
- 在 $r=0.3$, A3B 模型上：
  - 多模态平均得分从 $r=0$ 的 ~26 → 提升至 **37.49**（↑ >11 pts）。
  - 特别是在 counting、spatial reasoning 上提升显著。

---

### 与基线方法的对比结果
| 指标 | $r=0$（纯文本） | $r=0.3$（多模态） | 提升幅度 |
|------|------------------|--------------------|----------|
| 文本平均准确率 | 44.77 | 46.03 | +1.26 pts |
| 多模态平均准确率 | — | 37.49 | 显著优于 baseline（见 Table 6–8） |
| SpatialEval（text-only） | 35.95 | 41.43 | ↑ **+5.48 pts** |
| 3-shot gain (A3B) | N/A | +2.43 pts | 从近零增长到显著增益 |

> 🔺 结果表明：native 多模态训练不仅没有牺牲语言能力，反而实现了双向增强。

---

### 消融实验结果
#### （1）数据组成 $r$ 对 scaling law 的影响（Fig. 3 & Fig. 6）
| 目标 | 是否受 $r$ 影响 | 发现 |
|------|------------------|------|
| $L_{\text{text}}$ | ❌ 否（composition-invariant） | 最优 $N_{\text{opt}}, D_{\text{opt}}$ 几乎不随 $r$ 变化 |
| $L_{\text{mm}}$ | ✅ 是（composition-variant） | $r$ 越高，越倾向于 **更多数据、更少参数扩展**（parameter scaling exponent ↓） |

#### （2）compute-optimal allocation 曲线（Fig. 7）
- 当 $r=0.1$：$N_{\text{opt}} \propto C^{0.69}$
- 当 $r=0.3$：$N_{\text{opt}} \propto C^{0.665}$，$D_{\text{opt}} \propto C^{0.335}$ → 更偏向数据扩展

> 📌 结论：高密度多模态数据要求更大的 token 预算来有效学习。

---

## 4. 关键结论和发现

### 主要发现
1. **语言与多模态目标具有根本不同的 scaling behavior**
   - 语言目标 scaling 律对数据组成鲁棒；
   - 多模态目标则高度敏感，需动态调整资源配置。

2. **存在一条语言-多模态 Pareto 前沿**
   - 给定总计算预算，可通过调节 $r$ 得到一组最优配置 $(N, D_{\text{text}}, D_{\text{mm}})$。
   - 该前沿可用于指导大规模 native 多模态模型的设计。

3. **native 多模态训练带来正向跨模态迁移**
   - 学习到的空间结构可泛化到纯文本任务（如 SpatialEval）。
   - 支持有效的 multimodal in-context learning，且随 scale 增强。

4. **text-heavy 数据仅在大模型下才高效**
   - 小模型中过多文本无法充分利用；
   - 大模型才能吸收 text-heavy 数据中的丰富语义。

---

### 方法的局限性
1. **模型规模受限**：最大仅训练到 3B 参数，尚未验证超大规模（10B+）下的 scaling behavior。
2. **数据多样性不足**：仅使用一种 image-text 数据家族，未涵盖视频、音频等其他模态。
3. **评估代理指标依赖 loss**：缺乏独立 validation metric，依赖 smoothed training loss 作为 proxy。
4. **未包含 encoder-decoder 架构**：全部基于 decoder-only MoE 设计，结论可能不适用于其他结构。

---

### 未来工作方向
1. **扩展至更大模型与更多模态**（video, audio, sensor data）。
2. **探索动态数据混合策略**（curriculum learning style）而非固定 $r$。
3. **建立多模态专用 validation set** 以更好估计泛化误差。
4. **结合 instruction tuning 进一步释放 in-context learning 潜力**。
5. **研究跨语言多模态 scaling law**，推动 truly universal foundation models。

---

> 💡 **总体评价**：本文是 native multimodal pre-training 领域的重要奠基性工作，首次建立了系统的 compute-optimal scaling framework，揭示了多模态 scaling 的独特规律，并实证展示了其在跨模态迁移和上下文学习方面的巨大潜力。

</details>

---

### 5. [Unified Static-Dynamic Pruning for Efficient LLM Inference](https://arxiv.org/abs/2607.21985)

**Authors**: Jinhyeok Kim, Yejoon Lee, Jaeyoung Do  
**Category**: cs.DC  
**Published**: 2026-07-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.21985v1  

#### Abstract
The increasing deployment of large language models (LLMs) has magnified the computational and memory bottlenecks of autoregressive decoding, where low compute intensity and bandwidth-bound kernels dominate inference cost. Weight pruning offers a promising remedy,

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Unified Static-Dynamic Pruning for Efficient LLM Inference》论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代大语言模型（LLMs）在推理阶段面临严重的**计算与内存瓶颈**，尤其是在自回归解码（autoregressive decoding）过程中，由于低计算强度（compute intensity）和带宽受限的算子主导，导致推理效率低下。现有的权重剪枝（weight pruning）方法存在以下局限：
- **静态剪枝（Static Pruning, SP）**：永久移除冗余权重，虽能减少内存占用，但缺乏对输入的适应性。
- **动态剪枝（Dynamic Pruning, DP）**：根据输入动态跳过不重要的激活，具有上下文感知能力，但引入运行时不规则性和调度开销。

两者难以协同，系统层面存在格式、执行路径和硬件利用率的不兼容。

### 提出了什么新方法或新思路
本文提出 **SPDP** ——一个统一的稀疏推理框架，首次将**非结构化静态剪枝（SP）** 与**输入自适应动态剪枝（DP）** 在GPU上高效融合。

其核心是**联合设计的格式-内核 co-design**：
- **Tiled-Column-wise Bitmap Compressed (Tiled-CBC)** 格式：一种列优先的分块压缩格式，支持静态压缩的同时保留对动态激活稀疏性的直接寻址能力。
- 两个互补的GPU内核：
  1. **CUDA-core spMspV kernel**：用于 decode 阶段，采用 Hybrid Activation-aware Dynamic Shared-Memory Bitmap Decoding (HAD-SMBD)，实现细粒度运行时激活跳过。
  2. **Tensor-Core SpMM kernel**：用于 prefill 阶段，复用同一 Tiled-CBC 格式，通过共享内存中的布局对齐支持高效的 Tensor Core 运算。

### 相比现有方法的优势
| 维度 | SPDP优势 |
|------|---------|
| **算法效率** | 同时利用SP（减少内存访问）和DP（减少计算量），理论计算强度（CI）显著提升 |
| **系统兼容性** | 单一格式支持 decode 和 prefill 两阶段，避免重复存储与格式转换 |
| **硬件适配性** | decode 使用 CUDA core 实现高并行稀疏向量乘法；prefill 利用 Tensor Core 保持高吞吐 |
| **性能增益** | 在匹配困惑度（perplexity）下，达到更高稀疏率（+25%），同时获得显著加速 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **模型质量评估**：
  - **WikiText**：用于评估 perplexity（越低越好）
  - **HumanEval**：代码生成 pass@1 准确率
  - **GSM8K**：数学推理 exact-match 准确率
  - **CoQA**：对话问答 token-level F1
  - **MMLU**：通用知识 multiple-choice accuracy
- **校准数据集**：
  - **C4 dataset**：用于静态剪枝方法（如 Wanda）的统计估计（128个序列）
  - **Alpaca dataset**：用于动态剪枝（TEAL）构建激活幅值的经验累积分布函数（ECDF）

### 实验设置和评估指标
- **模型**：
  - 主要测试模型：`Llama-2-7B-hf`
  - 代码生成补充测试：`Qwen3-32B`
- **剪枝配置**：
  - SP 方法：Wanda（非结构化静态剪枝）
  - DP 方法：TEAL（基于幅值阈值的动态激活剪枝）
  - 联合剪枝策略：SP + DP，总稀疏率为 $1 - (1-sp)(1-dp)$
- **硬件平台**：
  - NVIDIA A10G、L4、L40S GPU
- **评估指标**：
  - **kernel-level**：TFLOP/s（归一化至 cuBLAS TC）
  - **end-to-end**：Time Per Output Token (TPOT)
  - **硬件分析**：SM Busy（活跃SM比例）、Max Bandwidth（DRAM带宽利用率）、Bank Conflicts（共享内存冲突）
  - **模型质量**：Perplexity、下游任务准确率

### 基线方法对比
| 基线 | 类型 | 说明 |
|------|------|------|
| **cuBLAS / cuBLAS_TC** | Dense baseline | 密集矩阵运算基准 |
| **cuSPARSE, Sputnik, SparTA** | 通用稀疏库 | 支持 SpMV/SpMM，但未针对LLM优化 |
| **Flash-LLM** | Load-as-sparse, compute-as-dense | 引入 tile reconstruction 思路 |
| **SpInfer** | State-of-the-art sparse inference | 使用 TCA-BME 格式 + Tensor Core SpMM，当前最优静态稀疏方案 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 内核级性能（spMspV kernel）
- 在 **A10G GPU** 上：
  - 平均比 **SpInfer** 快 **1.24×**
  - 最高速度达 **1.70×**
  - 比 **cuBLAS TC** 快 **1.88×**（平均），最高达 **3.32×**
- 在 **L4 GPU** 上：
  - 平均比 **SpInfer** 快 **1.37×**
  - 最高达 **2.51×**
  - 比 **cuBLAS TC** 快 **2.11×**（平均），最高达 **3.52×**

> 图8显示，在典型LLM投影矩阵形状下（如 OPT, LLaMA, Qwen 等架构），SPDP-spMspV 在所有测试场景中均优于基线。

#### 端到端推理性能（TPOT）
- 在多种GPU上，SPDP 实现了 **1.24×–1.37× 的平均 TPOT 加速**，峰值可达 **2.51×**
- 在相同 perplexity 下，SPDP 可以支持 **高达 25% 更高的稀疏率**，意味着更少的数据搬运和更高的能效

#### 硬件利用率分析（图9）
- **SM Busy**：SPDP-spMspV 达到最高，表明计算与内存重叠更好
- **Bandwidth**：与 SpInfer 相当，说明有效维持高内存带宽
- **Bank Conflicts**：显著低于 SpInfer，得益于 Tiled-CBC 的列对齐设计

### 与基线方法的对比结果
| 对比项 | 结果 |
|-------|------|
| vs. SpInfer（静态稀疏） | 在相同稀疏率下，SPDP 实现更低 TPOT 和更低 perplexity；在相同 perplexity 下，可使用更高稀疏率 |
| vs. TEAL（仅DP） | SPDP 显著降低内存带宽需求，而纯DP无法减少权重加载成本 |
| vs. Dense (cuBLAS) | 在 moderate sparsity（30%-50%）区间，SPDP 明显超越密集基线，尤其在 decode 阶段 |

### 消融实验结果（隐含于文中分析）
虽然没有明确列出“ablation study”章节，但通过多组对比揭示了关键设计价值：
- **Tiled-CBC 格式必要性**：传统CSR/CSC格式无法支持高效的HAD-SMBD解码；SpInfer的行主序格式不适用于列稀疏的DP
- **HAD-SMBD机制有效性**：相比原始SMBD，新增的 activation-aware 列跳过机制使 inactive column 完全跳过，减少无用访存
- **异步流水线设计**：三组异步操作（ColInfo、XTile、GTile）最大化隐藏延迟，提高ILP
- **Layout Alignment for Prefill**：padding-based transpose 方案有效避免 bank conflict，仅引入可忽略开销

---

## 4. 关键结论和发现

### 主要发现
1. **静态与动态剪枝本质上是正交且互补的**：
   - SP 沿权重维度剪枝（行方向）
   - DP 沿激活维度剪枝（列方向）
   - 二者重叠极少，联合稀疏度接近乘积效应，显著提升 compute intensity

2. **统一格式是实现端到端加速的关键**：
   - Tiled-CBC 成功桥接了 SP 和 DP 的系统鸿沟
   - decode 使用 CUDA core 处理 spMspV，prefill 复用同一格式跑 Tensor Core SpMM，无需重构

3. **roofline模型预测与实测一致**：
   - 实验验证了联合剪枝可在 decode 阶段突破 memory-bound 瓶颈，逼近 compute peak

4. **SPDP 推进了推理效率-质量的 Pareto 前沿**：
   - 在相同质量下更快，或在相同速度下质量更高
   - 尤其适合小批量、交互式、长序列生成等 memory-bound 场景

### 方法的局限性
1. **主要面向 decode 阶段优化**：
   - Prefill 阶段因输出宽度大（high N），常处于 compute-bound 区域，稀疏收益有限（见图12）
   - 当前 prefill 路径主要用于格式统一，而非极致加速

2. **不适用于极端稀疏场景**：
   - 如 SuiteSparse 中 >95% 稀疏的科学计算矩阵，此时 CSR/CSC 更优
   - SPDP 定位为 **moderate-sparsity LLM decode kernel**

3. **量化兼容性有限**：
   - 元数据（Bitmap, ColInfo）仍占较大比例，低比特量化下压缩增益减弱
   - 未来需探索更粗粒度静态稀疏模式以摊销元数据开销

4. **暂未支持 batched dynamic pruning**：
   - 当前 decode 为逐 token 处理，batched DP 需额外算法与内核扩展

### 未来工作方向
1. 扩展至 **batched dynamic pruning**，支持连续批处理（continuous batching）
2. 探索 **multi-GPU sparsity-aware scheduling**，实现跨设备稀疏负载均衡
3. 与 **quantization co-design** 深度结合，研究混合精度下的元数据压缩
4. 支持更多类型的 **dynamic pruning policies**（如基于mask的方法）
5. 集成进主流 LLM serving engine（如 vLLM, TensorRT-LLM）

---

> ✅ **开源信息**：  
> 项目代码、数据及 artifact 已公开于 GitHub：[https://github.com/AIDASLab/SPDP](https://github.com/AIDASLab/SPDP)

</details>

---

### 6. [PRISM: Evaluating POSIX Storage Systems for AI Research Workflows](https://arxiv.org/abs/2607.21746)

**Authors**: Adithya Kumar, Aditya Basu, Jacob Kahn, Parth Malani, Leo Huang, Kalyan Saladi  
**Category**: cs.DC  
**Published**: 2026-07-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.21746v1  

#### Abstract
The rapid advancement of AI research is driven by massive investments in GPU clusters, yet the critical role of storage systems in enabling efficient research workflows is often overlooked. Unlike traditional HPC workloads, AI research prioritizes researcher productivity and ease of iteration. Pract...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《PRISM: Evaluating POSIX Storage Systems for AI Research Workflows》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前 AI 研究集群在大规模部署 GPU 的同时，**存储系统的重要性常被忽视**。传统存储基准（如 `fio`、`IOR`、`MLPerf Storage`）主要针对生产环境中的高吞吐、顺序 I/O 场景，无法准确反映 AI **研究工作流**中常见的**动态性、突发性、小文件密集、元数据操作频繁**等特征。

这导致：
- 存储系统选型缺乏科学依据；
- 实际运行中出现性能瓶颈（如 checkpoint 加载延迟飙升）却难以定位；
- 研究人员因低效 I/O 而降低迭代效率。

---

### ✅ 提出了什么新方法或新思路

作者提出了 **PRISM**（**P**OSIX **R**esearch **I**nfrastructure **S**torage **M**easurement），一个专为 AI 研究场景设计的存储评估框架。

#### 核心创新点：
1. **以真实 AI 工作流为基础建模**  
   PRISM 不依赖合成 I/O 模式，而是复现了 AI 研究全生命周期的关键阶段：
   - 开发环境搭建（`git clone`, `tar/untar`）
   - 数据准备与管理（`create_files`, `move_files`）
   - 数据加载（`md5_check` 支持随机访问）
   - 模型 checkpointing（支持 DDP/FSDP 分布式保存与恢复）
   - 合成数据生成（可配置 arrival/size/data 分布）

2. **融合真实 ML 框架操作**  
   直接集成 **PyTorch** 和 **HuggingFace Transformers**，执行真实的 `torch.save()`、`torch.load()`、模型下载与缓存同步等操作，确保测试负载贴近实际。

3. **支持分布式执行与扩展性验证**  
   利用 SLURM/GPU 集群实现多 rank 并行测试，评估系统在 **8 ~ 256 GPUs** 规模下的表现，并自动处理 barrier 同步、rank 协调等问题。

4. **提供模块化插件架构**  
   用户可通过简单接口添加新的 benchmark 模块，极大提升了框架的可维护性和适应性。

---

### ✅ 相比现有方法的优势

| 特性 | PRISM | 传统工具（如 fio, IOR, MLPerf） |
|------|-------|-------------------------------|
| **是否反映真实 AI 工作流** | ✅ 是 | ❌ 否（偏重生产训练） |
| **是否覆盖开发/调试场景** | ✅ 是（env setup, metadata ops） | ❌ 否 |
| **是否支持分布式 checkpointing** | ✅ 完整支持 DDP/FSDP | ⚠️ 部分支持 |
| **是否使用真实 PyTorch 操作** | ✅ 是 | ❌ 多为模拟 |
| **是否衡量尾延迟（Tail Latency）** | ✅ 报告 p50/p99 | ⚠️ 常只关注平均值 |
| **是否可用于 CI/CD 回归检测** | ✅ 支持前后版本对比 | ❌ 通常静态测试 |

> 🔍 **关键优势总结**：PRISM 弥合了“理想化存储性能”与“现实研究体验”之间的鸿沟，是首个面向 **AI Research Workflows** 的端到端 POSIX 存储评估体系。

---

## 2. 核心实验方法和设置

### ✅ 使用了哪些数据集

PRISM 主要采用**合成数据集 + 真实模型权重**相结合的方式：

- **合成文件集**：
  - 文件大小从 **8KB 到 1GB 可配**
  - 支持多种分布：固定、指数、均匀、lognormal
  - 内容生成方式多样：`random`, `deterministic (SHA256)`, `zero`, `pattern`

- **真实模型来源**：
  - 集成 **HuggingFace Hub**，支持加载 GPT-2、LLaMA、BERT 等主流模型用于 checkpoint 测试
  - 所有 rank 共享缓存目录，仅 rank 0 下载，其余等待 barrier

- **典型测试规模**：
  - 文件数量达 **800K 小文件（32KiB each）**
  - Checkpoint 模型参数量级达百亿以上

---

### ✅ 实验设置和评估指标

#### 实验平台：
- 最大使用 **8K H100 GPUs** 集群
- 对比两种主流 POSIX 存储后端：
  - **Lustre_SYS**：基于 AWS FSx for Lustre 或类似云托管方案
  - **NAS_SYS**：商用 NAS Appliance（如 NetApp, VAST, PureStorage）
- 所有测试均通过 **SLURM** 进行 gang scheduling，保证同步性

#### 评估维度：
| 类别 | 具体 Benchmark | 关键指标 |
|------|----------------|----------|
| **Checkpointing** | `ddp_save/load`, `fsdp_save/load`, `rl_load` | Save/Load 延迟（mean, p99） |
| **Data Loading** | `md5_check`（seq/random/strided） | 读取延迟、带宽 |
| **Metadata Ops** | `create/list/move/delete_files`, `folder_bench` | 每项操作耗时（ms） |
| **Dev Environment** | `git_clone`, `run_tar`, `run_untar` | 克隆/打包时间 |
| **Synthetic Workload** | `create_synthetic_workload` | 可控 arrival/size/data pattern |

#### 性能指标重点：
- **Latency Percentiles**（p50, p99） > 平均延迟
- **Scalability Trends**：随 task 数增加的表现变化
- **Cache Effects**：启用/禁用 page cache（via `posix_fadvise(DONTNEED)`）
- **Failure Detection**：能否暴露 lock contention、bug 等异常

---

### ✅ 基线方法对比

PRISM 并非单纯对比算法，而是作为评估工具，用于比较不同 **storage architectures** 在相同 workload 下的表现：

| 存储类型 | 示例 | 是否完全 POSIX |
|--------|------|---------------|
| **Lustre-based** | AWS FSx Lustre, Azure Lustre | ✅ 完全兼容 |
| **NAS-based** | NetApp, VAST, PureStorage | ⚠️ 仅 NFS v3/v4 接口，有限 POSIX 语义 |

此外，还对比了不同软件栈的影响：
- **Native POSIX access** vs. **fsspec** 接口性能差异
- 升级前 vs. 升级后的性能回归检测

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### 📌 Checkpointing 性能（Fig. 6）
| 操作 | NAS_SYS vs. Lustre_SYS 表现 |
|------|------------------------------|
| `ddp_save` | Lustre_SYS 快 9–35%（得益于条带化写入） |
| `fsdp_save` | 小任务时 Lustre 更优（快 26%），大任务时 NAS 反超（快 2×） |
| `ddp_load` | NAS_SYS 快最多 **20%** |
| `fsdp_load` | NAS_SYS 快最多 **3×** |

> 💡 **说明**：NAS 在并行读取方面优化更好，尤其适合 FSDP 场景下多个 rank 同时读取各自 shard。

---

#### 📌 Dataloading 性能（Fig. 7）
- 测试：800K 个 32KiB 文件，随机读取
- 结果：
  - **未分片（flat dir）**：Lustre_SYS 比 NAS_SYS **慢超过 3×**
  - **分片后（sharded dirs）**：Lustre 性能提升近 3×，但仍落后于 NAS

> ⚠️ **根本原因**：Lustre 元数据服务器对单目录大量文件处理能力弱，必须人工分片缓解瓶颈。

---

#### 📌 Metadata 操作性能（Fig. 8）
| 操作 | NAS_SYS vs. Lustre_SYS |
|------|-------------------------|
| `list_files`（256 clients） | NAS_SYS 快 **80×** |
| `create_files` | NAS_SYS 快约 4.8× |
| `move_files` | NAS_SYS 快约 12× |
| `delete_files` | NAS_SYS 显著更快 |

> 🔥 **结论**：NAS 在元数据密集型操作上具有压倒性优势，这对 AI 研究中频繁的脚本调试、日志查看、环境重建至关重要。

---

#### 📌 性能回归检测案例（Fig. 9）
- 事件：NAS vendor 发布一次例行更新
- 影响：`fsdp_load` 延迟上升 **4–8×**
- 根因：`openat` 系统调用存在锁竞争，每次阻塞约 5 秒
- PRISM 作用：
  - 自动捕获性能退化
  - 提供可复现证据提交给厂商
  - 验证补丁有效性（修复后延迟下降 2.3×）

> ✅ **体现价值**：成为 CI/CD 中不可或缺的一环，防止“无声降级”。

---

#### 📌 fsspec vs. Native POSIX 性能对比（Fig. 10）
| 操作 | fsspec 开销 |
|------|------------|
| `list_files`（5000 files） | **15× 慢**（1651ms vs. 104ms） |
| `delete_files` | 明显更慢 |
| `md5_check` | 差异较小（<15%） |

> 🧩 **启示**：虽然 `fsspec` 提供跨后端抽象便利，但在元数据操作路径上开销巨大，不适合性能敏感场景。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **没有“万能”的最佳存储方案**  
   - **Lustre** 更适合大文件、高吞吐场景（如大规模 checkpoint 写入）
   - **NAS** 更适合小文件、高并发、元数据密集型场景（如代码克隆、日志分析、FSDP 读取）

2. **AI Research Workflows ≠ Production Training**  
   - 研究强调 **flexibility, interactivity, low friction**
   - 需要 POSIX 接口来支持 `bash` 脚本、`grep`、`ls`、`conda`、`git` 等生态工具
   - 忽视这一点会导致“GPU 空转等数据”

3. **元数据性能是隐形杀手**  
   - Metadata IOPS 占总 IOPS 的很大比例（Fig. 1c）
   - 传统 benchmark 忽略此问题，但现实中严重影响用户体验

4. **尾延迟决定感知性能**  
   - 即使平均延迟良好，p99 延迟过高也会导致 job 卡顿
   - PRISM 明确报告 p99，帮助识别 straggler 问题

5. **抽象层带来代价**  
   - `fsspec` 虽然方便，但 metadata 操作开销高达 15×
   - 应谨慎用于高频目录遍历场景

---

### ✅ 方法的局限性

1. **仍聚焦于 POSIX 层面**  
   - 未深入评估底层硬件（如 NVMe vs HDD）、网络拓扑影响
   - 对对象存储（S3/GCS）仅间接通过 `fsspec` 支持

2. **依赖特定框架生态**  
   - 当前重度绑定 PyTorch/HuggingFace，对 JAX/TensorFlow 支持有限

3. **尚未开源全部组件**  
   - 文中提到已在 Meta 内部广泛使用，但完整代码未公开（截至论文时间）

4. **成本较高**  
   - 需要在数千 GPU 集群上运行，中小团队难以复现

---

### ✅ 未来工作方向

1. **扩展支持更多 ML 框架**  
   - 如 JAX、TensorFlow、DeepSpeed 等 checkpoint 格式

2. **引入 trace replay 功能**  
   - 基于真实集群 I/O trace 构建更精确 workload 模型

3. **增强自动化决策能力**  
   - 结合 PRISM 输出，构建推荐引擎：给定 workload profile → 推荐最优 storage backend

4. **轻量化版本适配云实验室**  
   - 支持在 8~64 GPU 小集群快速运行 smoke test

5. **开放基准即服务（Benchmark-as-a-Service）**  
   - 为云厂商提供标准化认证流程，推动行业形成统一评估标准

---

## ✅ 总结一句话

> **PRISM 填补了 AI 研究场景下存储评估的空白，首次将“研究人员的真实体验”纳入评测体系，推动存储系统从“跑得快”走向“用得好”。**

</details>

---

### 7. [Duet: Co-Optimizing P2P Message Propagation and Rotating-Leader Consensus](https://arxiv.org/abs/2607.22209)

**Authors**: Yifeng Ye, Rongji Huang, Gerui Wang, Mingchao Wan, Yuxing Duan, Jingjing Zhang, Shengyun Liu  
**Category**: cs.DC  
**Published**: 2026-07-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.22209v1  

#### Abstract
In blockchain systems, peer-to-peer (P2P) overlay networks play a crucial role in providing reliable, scalable and efficient message-delivery services to upper layers. However, the consensus layer and the underlying P2P network remain mutually opaque in existing blockchains, waiving the opportunity ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Duet: Co-Optimizing P2P Message Propagation and Rotating-Leader Consensus

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在现有的区块链系统中，**共识层**（consensus layer）与底层的 **P2P overlay 网络**通常是相互独立设计的，彼此“互为黑盒”。这种解耦虽然简化了系统架构，但也导致无法利用跨层信息进行协同优化。例如：
- P2P 网络采用随机拓扑和基于 **gossip** 的广播机制，造成高带宽消耗和传播延迟；
- 共识协议中的 **leader rotation** 过程未考虑网络地理分布，影响整体吞吐量。

这限制了高性能区块链系统的可扩展性和效率。

### 提出的新方法与思路
作者提出 **Duet** ——一种**跨层协同优化框架**，通过将 P2P 网络状态（如拓扑、延迟）记录在链上，并结合 rotating-leader consensus 协议（如 Tendermint），实现以下三项核心优化：

#### （1）加速 Proposer Rotation 序列
- 利用链上维护的节点间 RTT 信息，构建一个**低延迟的 proposer 轮换环**（proposer ring）。
- 将轮换顺序建模为 **Traveling Salesman Problem (TSP)** 的变体，使用贪心算法最小化连续 proposer 之间的通信延迟。
- **优势**：显著减少区块提议传递时间，提升 pipeline 效率。

#### （2）引入两阶段可靠广播机制（Reliable Broadcast）
- 第一阶段（Best-effort）：使用**树形结构**（tree-based dissemination）快速分发 PROPOSAL 消息，降低冗余流量。
- 第二阶段（Fallback）：当某些节点未收到消息时，利用 **PREVOTE 投票作为 ACK**，触发基于 gossip 的重传。
- 引入 `tmr_TREE` 定时器控制从树播到 gossip 回退的切换时机。
- **优势**：正常情况下节省大量带宽；故障下仍能保证可靠性。

#### （3）构建多因素感知的传播树（Multi-Factor-Aware Tree）
- 构造每个 proposer 对应的 **dissemination tree**，综合考虑：
  - 地理延迟（latency）
  - 节点出度限制（fanout/bandwidth）
  - 后续 proposer 的优先级（proposer rotation awareness）
- 使用加权边选择策略，在稀疏图中构造高质量生成树。
- **优势**：优化消息路径，加快前缀交付（prefix delivery），支持更深的流水线。

> ✅ 所有上述结构均可由全局一致的拓扑信息 **确定性地本地生成**，无需额外协调。

---

## 2. 核心实验方法和设置

### 实验平台与部署环境
- 在 **Amazon EC2** 上部署最多 **300 个节点**，分布在 **10 个地理区域**（美国、欧洲、亚太）。
- 节点实例类型：`m4.xlarge`（4 vCPU, 16 GiB RAM）。
- 网络拓扑基于真实测量的跨区 RTT 构建。

### 数据集与工作负载
- 并非传统意义上的“数据集”，而是模拟大规模分布式区块链运行场景。
- 工作负载：固定大小交易（每笔 1KB），批量提交（batch size 可调至饱和）。
- 主要评估不同规模（N=10~300）下的性能表现。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Throughput (tps)** | 系统峰值吞吐量（transactions per second） |
| **Latency** | 从提案发出到被提交的时间（commit latency） |
| **Pipeline Depth** | 已提出但尚未提交的高度差，反映流水线利用率 |
| **Per-region Latency** | 不同地区节点的平均与尾部延迟（p95） |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Gossip** | 使用 libp2p 的 GossipSub，默认 mesh fanout=6，标准全冗余广播 |
| **K-ary Tree** | 每个 proposer 构造平衡 K 叉树（K=3 或 5），用于 proposal 分发，代表现有树型 BFT（如 Kauri）的设计 |
| **Duet-LimitedView** | Duet 的变种，仅暴露部分链路信息以缓解隐私风险 |

所有方法共享相同的共识逻辑（pipelined Tendermint）、投票传播方式和缓冲机制，仅 **PROPOSAL 分发策略不同**，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 规模 | 方法 | 吞吐量 (tps) | 延迟 (ms) | Pipeline Depth |
|------|------|-------------|-----------|----------------|
| N=10 | Duet | 14.07k | 1264 | ~3 |
| N=50 | Duet | 16.57k | 2125 | ~6 |
| N=100 | Duet | **21.09k** | 3080 | **8** |
| N=300 | Duet | **20.50k** | ~4000 | ~7 |

> 📈 随着节点数增加，Duet 表现出更强的可扩展性。

### 与基线方法的对比结果

#### 吞吐量提升（N=300）
- **Duet vs Gossip**：**7.26× 更高的吞吐量**
- **Duet vs K-ary Tree**：约 **4.88× 提升**

> 💡 图7显示，Gossip 和 K-ary Tree 在 N>100 后性能急剧下降，而 Duet 维持接近峰值水平。

#### 流水线深度对比（N=100, batch=10k）
| 方法 | Pipeline Depth | 吞吐量 |
|------|----------------|--------|
| Duet | **8** | 21.09k tps |
| K-ary Tree | 2 | 4.70k tps |
| Gossip | 1 | 3.44k tps |

> 🔍 深流水线是 Duet 高吞吐的关键驱动力，得益于更快的前缀交付。

#### 延迟表现
- 尽管 Duet 单个区块延迟较高（因深流水线），但单位时间内处理更多事务。
- 在相同吞吐量下（~13–14k tps），Duet 的延迟低于基线（见图10）。

### 消融实验结果（Ablation Study, N=100）

| 配置 | 吞吐量 (tps) | 提升倍数 |
|------|-------------|----------|
| Gossip + Random Order | 3.44k | 1.00× |
| Gossip + Greedy Order | 3.48k | 1.01× |
| K-ary Tree + Random | 4.70k | 1.37× |
| K-ary Tree + Greedy | 4.88k | 1.42× |
| Duet Tree + Random | 6.89k | 2.00× |
| **Duet Tree + Greedy** | **21.09k** | **6.13×** |

> 🔬 结论：**只有同时启用“智能排序”和“多因素树构造”才能释放最大性能潜力**。

### Timer 敏感性测试
- `tmr_TREE` 设置对性能极为敏感：
  - 若设为 500ms 或 1s：过早进入 gossip 回退，引发不必要的重传，吞吐降至 12k~13.5k。
  - 推荐设置为 **2–3 秒**，可在正常情况避免回退，保持高吞吐。

---

## 4. 关键结论和发现

### 主要发现
1. **共识与 P2P 网络可以且应当协同优化**  
   区块链天然具备 state machine 抽象能力，可用于可信地记录网络元数据，从而赋能跨层优化。

2. **树形广播 + 投票 ACK 机制可大幅降低带宽开销**  
   正常情况下消除 gossip 冗余，仅在必要时 fallback，兼顾效率与可靠性。

3. **地理感知的 proposer 轮换显著提升流水线效率**  
   减少 leader handoff 延迟，使后续 proposer 更快获取前序区块，推动 pipeline 深度增长。

4. **Duet 在大规模部署中展现出卓越的可扩展性**  
   即使在 300 节点跨洲部署下，仍能达到 **20.5k tps**，远超传统方案。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖链上拓扑可见性** | 需要公开节点连接与延迟信息，可能带来 **拓扑推断攻击**（topology inference）或定向 DoS 风险 |
| **TREE 阶段引入额外等待时间** | 每轮需等待 `tmr_TREE` 超时，轻微拖慢恢复速度，尤其在 leader crash 场景 |
| **对定时器配置敏感** | `tmr_TREE` 需精细调优，否则易误触发回退，损害性能 |
| **适用于静态或缓慢变化的网络** | 频繁的 membership change 会增加 reconfiguration 开销 |

> ⚠️ 实验表明，即使只暴露部分链路（LimitedView），Duet 仍优于基线，说明有一定隐私-性能折衷空间。

### 未来工作方向
1. **扩展至 multi-leader 或 DAG-based consensus 协议**  
   如 Narwhal + Tusk、Avalanche 等，这些协议并发提案更多，带宽压力更大，更需要高效传播机制。

2. **动态自适应树与 proposer 序列调整**  
   根据实时网络状况（如拥塞、故障）动态重构 dissemination tree 和 proposer ring。

3. **应用于 permissionless 区块链主干网**  
   通过质押机制（staking）让节点参与骨干网络建设，形成高性能 backplane。

4. **探索轻量级加密保护拓扑隐私**  
   如使用零知识证明或安全多方计算隐藏部分拓扑细节，同时允许正确构造传播树。

5. **集成更先进的网络模型**  
   如结合 FRING、Erlay 等压缩与 reconciliation 技术，进一步降低传播成本。

---

> ✅ **总结一句话**：  
> **Duet 通过将 P2P 网络状态“上链”，实现了共识层与网络层的协同进化，首次在 rotating-leader BFT 中系统性优化了 proposer rotation 与 proposal dissemination，实现在 300 节点规模下高达 7.26× 的吞吐提升，为下一代高性能区块链提供了新的设计范式。**

</details>

---

### 8. [Neural Feature Governance: Extending Atom Prevalence](https://arxiv.org/abs/2607.21671)

**Authors**: Idris Karel Seunda Ekwe, Patrick Tenga Shako, Ernest Parfait Fokou\'e  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.21671v1  

#### Abstract
Neural network compression and interpretability remain open challenges in modern deep learn- ing, where billion-parameter architectures deliver impressive accuracy at the cost of trans- parency, computational efficiency, and reliable uncertainty quantification. This paper introduces Neural Atom Prev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Neural Feature Governance: Extending Atom Prevalence

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代深度学习模型（尤其是大规模神经网络）面临以下核心挑战：
- **高参数量导致的低可解释性**：模型被视为“黑箱”，难以理解其决策过程。
- **计算效率低下**：数十亿参数的模型在训练和推理时消耗大量资源。
- **不确定性量化不可靠**：传统模型缺乏对预测不确定性的有效建模，尤其是在区分数据噪声（aleatoric uncertainty）和模型无知（epistemic uncertainty）方面。

本文旨在解决如何在保持甚至提升模型准确率的同时，实现**结构稀疏化、高可解释性、稳定性和可靠的不确定性量化**这一多目标优化难题。

### 提出了什么新方法或新思路
论文提出了 **Neural Atom Prevalence (NAP)** ——一种基于贝叶斯框架的节点级模型选择方法，将 Fokoué (2008) 的 **Atom Prevalence** 原则首次成功扩展到神经网络领域。

#### 核心创新点：
- **Neural Atom 定义**：将神经元（或激活单元）定义为基本分析单位（`h_j^l = σ(w_j^l ⋅ h^{l-1} + b_j^l)`），实现了从权重空间到节点空间的降维，使模型选择更具可解释性。
- **四阶段混合流水线 (Hybrid Pipeline)**：
  1. **Bayesian Lottery Ticket (BLT)**：通过 Iterative Magnitude Pruning (IMP) 找到一个高性能的稀疏子网络（“中奖彩票”）作为起点。
  2. **Soft SS-IG 训练**：在“中奖彩票”架构上应用 **Spike and Slab Independent Gaussian (SS-IG)** 先验进行软变分训练，避免早剪枝（premature pruning），并计算每个神经元的后验包含概率 `π_j^l`。
  3. **Poisson-Binomial (PB) 最优层大小选择**：利用动态规划递归计算每层活跃节点数 `K_v^l` 的后验分布，并选择最可能的层大小 `K_opt^l = argmax_k P(K_v^l = k)`。
  4. **贝叶斯微调 (Bayesian Fine-tuning)**：保留 `K_opt^l` 个包含概率最高的神经元，构建最终稀疏网络，并从初始权重重新开始微调，确保稳定性。

### 相比现有方法的优势
| 方面 | NAP | 传统方法（如 SS-IG, LTH） |
|------|-----|------------------------|
| **稀疏性控制** | 显式、数据驱动的最优层大小选择（PB），压缩率更高且更稳定。 | 依赖固定阈值或平滑收缩，压缩率较低或不稳定。 |
| **可解释性** | 以神经元为原子单位，直接识别出“最常出现”的关键特征表示。 | 多在权重级别操作，难以解释。 |
| **稳定性** | 四阶段设计避免了早剪枝；BLT提供良好初始化；微调从头开始。 | 无结构剪枝易导致收敛不稳定。 |
| **不确定性量化** | 实现了高度校准的不确定性分解，epistemic uncertainty 仅占总方差的 3-4%。 | Mean-Field VI 倾向于低估后验方差，导致过度自信。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了回归与分类任务，包括模拟和真实世界数据集：
- **Simulated Nonlinear Regression**：`Y = 2 + sin(x_3 x_4) + 2x_5 + ε`，用于已知 ground truth 下的精确评估。
- **UCI Regression Datasets**：
  - **Concrete**: 预测混凝土抗压强度 (1,030 样本, 8 维特征)。
  - **YearPredictionMSD**: 预测歌曲发行年份 (515,345 样本, 90 维特征)。
- **Image Classification Dataset**：
  - **MNIST**: 手写数字分类 (60,000 训练, 10,000 测试)。

### 实验设置和评估指标
- **模型架构**：
  - 回归任务：FNN (如 Concrete: 8→50→1)。
  - 分类任务：MLP (784→400→400→10)。
- **激活函数**：统一使用 Swish。
- **优化器**：Adam，梯度裁剪至范数 10.0。
- **评估指标**：
  - **性能**：测试集 RMSE (回归) 或 Accuracy (分类)。
  - **稀疏性**：相对于原始稠密架构被移除的节点比例 (`% pruned nodes`)。
  - **不确定性量化 (UQ)**：
    - **Aleatoric vs Epistemic Uncertainty**：分解总预测方差。
    - **校准性 (Calibration)**：95% 预测区间的实际覆盖率 (Coverage)，Expected Calibration Error (ECE)。
  - **稳定性**：多次运行的 RMSE/Accuracy 方差。

### 基线方法对比
- **SS-IG (Baseline)**：Jantre et al. (2023) 提出的原生 SS-IG 方法，在完整稠密网络上训练。
- **VBNN (Dense)**：未剪枝的变分贝叶斯神经网络，作为性能上限参考。
- **Bayesian Lottery Ticket (BLT)**：作为 NAP 流水线的第一阶段，也单独评估其性能。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 数据集 | NAP 性能 | 稀疏性 (% pruned) | Epistemic / Total Uncertainty |
|-------|----------|-------------------|-------------------------------|
| **Simulated** | RMSE ≈ 1.20 | 70% | ~3% |
| **Concrete** | RMSE = **6.317 ± 0.443** | **75.0% ± 1.8%** | 23.4% |
| **YearPredictionMSD** | RMSE = **8.80** | 56% | ~4.1% |
| **MNIST** | Accuracy = **97%** | **92%** (Layer 1: 95%) | ~3-4% (entropy-based) |

### 与基线方法的对比结果
- **稀疏性**：NAP 在所有任务上均实现了**显著更高的稀疏性**。例如，在 MNIST 上仅保留 **8%** 的原始节点，远超 SS-IG 的 ~20%。
- **准确性**：
  - 在 **Concrete** 数据集上，NAP (**6.317**) 不仅大幅优于 SS-IG (7.92)，甚至优于未剪枝的 VBNN (7.34)，同时实现了更高的稀疏性。这表明 NAP 的流水线具有**隐式正则化效应**，提升了泛化能力。
  - 在其他任务上，NAP 性能与 SS-IG 相当，仅略低于 VBNN，证明了其在极端压缩下的鲁棒性。
- **不确定性量化**：
  - **高度校准**：在模拟数据上，95% 预测区间实现了 **93.4%** 的实际覆盖率（接近理想值），MACE = 0.0131。
  - **可靠分解**：在回归任务中，epistemic uncertainty 仅占总方差的 **3-4%**，表明模型对自身参数非常确定，预测误差主要来自数据本身的噪声。
  - **过置信问题**：在 MNIST 分类任务上，NAP 表现出系统性过置信（ECE=0.1005），这是 Mean-Field VI 的固有局限。

### 消融实验结果
虽然文中未明确列出独立的消融研究，但整个 NAP 流水线的设计本身就是一种隐式的消融验证：
- **BLT 初始化的重要性**：采用“中奖彩票”而非随机初始化，是实现高精度压缩的关键。
- **Soft Training 的必要性**：在整个训练过程中保留所有神经元（软选择），避免了早剪枝导致的信息丢失。
- **PB 选择的有效性**：相比固定阈值，基于 Poisson-Binomial 分布选择最优层大小，提供了更原则性的压缩策略。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **成功扩展**：Fokoué 的 Atom Prevalence 原则可以被成功地从线性模型扩展到非线性的深度神经网络，形成了 **Neural Atom Prevalence (NAP)** 框架。
2. **高效稀疏化**：NAP 能够生成**高度稀疏**（最高达 92% 节点移除）且**性能优越**的模型，尤其在 Concrete 数据集上超越了所有基线。
3. **可靠不确定性**：该方法产生了**高度校准且可信的不确定性估计**，能够清晰地区分数据噪声和模型无知，这对于科学和安全关键应用至关重要。
4. **稳定训练**：四阶段混合流水线（BLT → Soft SS-IG → PB → Fine-tune）保证了训练过程的**稳定性**，避免了传统剪枝方法常见的收敛问题。

### 方法的局限性
- **Mean-Field VI 的质量**：使用的 Mean-Field 变分近似会系统性地低估后验方差，导致在 OOD (out-of-distribution) 场景下出现**过度自信**的风险。
- **Out-of-Distribution 泛化**：激进的剪枝使模型对已知数据的 epistemic uncertainty 极低，但无法有效检测未知输入，存在**结构性脆弱性**。
- **架构限制**：目前仅适用于 **Feedforward Neural Networks (FNN)**，尚未扩展到 CNN、Transformer、RNN 等现代主流架构。
- **计算成本高**：四阶段流水线（特别是 IMP 和多次训练）比单次变分训练**昂贵得多**。

### 未来工作方向
- **改进变分推断**：探索更强大的变分族（如 Structured VI, Normalizing Flows）来替代 Mean-Field，以获得更准确的后验近似。
- **增强 OOD 检测**：为 NAP 模型集成专门的 OOD 检测模块，以应对现实世界中的未知输入。
- **架构扩展**：将 NAP 框架推广到 **Convolutional Neural Networks (CNN)** 和 **Transformers** 等复杂架构。
- **降低计算开销**：研究更高效的 IMP 策略或端到端的联合优化方案，以减少训练时间和资源消耗。

</details>

---

### 9. [Energy Manifold Natural Gradient Descent: Riemannian Optimization for Neural PDE Solvers](https://arxiv.org/abs/2607.22004)

**Authors**: Zhangyong Liang, Huanhuan Gao  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.22004v1  

#### Abstract
Energy natural gradient descent (ENGD) aligns parameter updates with the curvature of an underlying function-space energy, but existing formulations assume an unconstrained Euclidean parameter domain. We introduce \EMNGDfull{}, a manifold optimization framework for physics-informed and variational n...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Energy Manifold Natural Gradient Descent: Riemannian Optimization for Neural PDE Solvers**

---

## 1. **论文的主要贡献和创新点**

### **解决的问题**
现有的 **Energy Natural Gradient Descent (ENGD)** 方法假设参数空间是无约束的欧几里得空间（Euclidean space），但在许多神经 PDE 求解器中，模型参数可能受到硬性约束（如权重归一化、方向-尺度分解等），其可行域构成一个**黎曼流形（Riemannian manifold）**。直接在欧氏空间执行 ENGD 并投影回流形会破坏能量诱导的二次模型最优性，导致更新方向失真。

此外，传统一阶优化方法（如 SGD、Adam）在训练神经 PDE 求解器时面临**残差刚性（residual stiffness）和病态条件数（ill-conditioning）**，收敛缓慢且精度受限。

### **提出的新方法：EMNGD**
本文提出了 **Energy Manifold Natural Gradient Descent (EMNGD)** ——一种将 ENGD 推广到**参数流形上的黎曼优化框架**。

#### **核心思想**
- 将函数空间的能量曲率通过微分映射拉回到参数流形的切空间上，构建**内蕴的（intrinsic）能量诱导度量**。
- 在切空间中求解阻尼后的自然梯度方向，并通过**retraction** 映射回流形，确保每一步都保持参数可行性。
- 不改变原始 PDE 能量目标，仅调整优化几何。

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **几何一致性** | 保留了 ENGD 的函数空间解释，同时兼容参数约束；当流形退化为欧氏空间时，EMNGD 精确还原为 ENGD。 |
| **理论保证** | 证明了未阻尼 EMNGD 方向的前向传播是函数空间 Newton 向量在当前模型切空间中的最佳近似（在能量度量下）。 |
| **数值稳定性** | 引入阻尼项使系统正定，提升鲁棒性；支持不精确求解仍能保持下降性质。 |
| **可扩展性** | 对于二次残差能量，利用 **Woodbury identity** 将大规模参数空间求解转移到小规模样本空间（sample space），显著降低计算成本。 |
| **方向控制** | 结合 **Nyström 近似**，提供两种模式：<br>• **Sketch-and-solve**：低秩近似方向，速度快<br>• **Preconditioned Krylov**：加速迭代求解，最终恢复精确方向 |

---

## 2. **核心实验方法和设置**

### **使用的 PDE 问题（Benchmark 数据集）**
实验涵盖多个经典偏微分方程，维度从 1D 到 5D：

- **Poisson 方程**（二维与五维）
  - $-\Delta u = f$，零边界条件
  - 解析解已知，便于误差评估
- **Heat 方程**（一维）
  - $\partial_t u = \partial_{xx} u$，初值与边值给定
- **Nonlinear equation**（非线性测试案例）
- **Helmholtz 方程**（二维）

网络结构多采用 `tanh` 激活函数的全连接网络（如 2-32-1、width-64 等）。

### **实验设置与评估指标**

#### **评估指标**
- **Relative $L^2$ error**: $\|u_\theta - u^*\|_{L^2}/\|u^*\|_{L^2}$
- **Relative $H^1$ error**（若可计算导数）
- **Training loss**（PINNs 残差损失）
- **Runtime / Time per update**
- **Memory consumption**

#### **基线方法对比**
| 方法 | 类型 |
|------|------|
| SGD | 一阶优化 |
| Adam | 自适应一阶优化 |
| BFGS / L-BFGS | 拟牛顿法 |
| ENGD | 能量自然梯度（欧氏） |
| Hessian-free / KFAC | 二阶梯度近似 |
| SPRING | 动量增强的自然梯度 |
| EMNGD (proposed) | 本文方法（含 Woodbury 和 Nyström 变体） |

#### **流形设置**
- 使用 **方向-尺度分解（direction-scale decomposition）** 构造参数流形：
  - 权重矩阵 $W = \text{Diag}(e^p) Q$，其中 $Q \in \text{Ob}(m,n)$ 是单位列范数矩阵（oblique manifold）
  - 参数流形为乘积流形：$\prod_l \text{Ob}(n_{l-1}, n_l) \times \mathbb{R}^{n_l} \times \mathbb{R}^{n_l}$

#### **求解策略**
- **Exact Woodbury solve**：用于小样本场景，直接求解样本空间系统
- **Nyström-preconditioned CG**：用于大样本，加速收敛
- **Armijo backtracking line search**：全局收敛保障

---

## 3. **主要实验结果和性能指标**

### **关键性能数据汇总**

| 方法 | Problem | Steps | Rel. $L^2$ Error | Runtime |
|------|--------|-------|------------------|---------|
| **EMNGD** | Poisson2D ($D=8,577$) | 20 | **6.778×10⁻⁹** | 25.42s |
| ENGD | Poisson2D | 50 | 4.639×10⁻¹ | 43s |
| Adam | Poisson2D | 200k | 9.321×10⁻⁴ | ~1h |
| **EMNGD-Woodbury** | 5D Poisson | – | **1.094×10⁻⁸** | 128.5s |
| ENGD (full) | 5D Poisson | – | 6.186×10⁻⁸ | 999.3s |
| **EMNGD** | Heat-1D | 1999 | **2.469×10⁻⁹** | 199.03s |
| Hessian-free | Heat-1D | 998 | 6.195×10⁻² | 1000.46s |

> 注：所有实验均在固定时间或步数预算下进行，EMNGD 在更少步数内达到更高精度。

### **与基线方法的对比结果**
- **收敛速度**：EMNGD 在 **几十到几百步内** 即可将误差降至 $10^{-8} \sim 10^{-9}$，而一阶方法即使运行数十万步也难以突破 $10^{-4}$。
- **最终精度**：在所有测试任务中，**EMNGD 达到了最低的相对误差**，显著优于 ENGD、Adam、BFGS 等。
- **效率优势**：得益于 Woodbury 转换，EMNGD 在高维参数下仍保持高效。例如在 5D Poisson 中，**EMNGD-Woodbury 仅用 128.5 秒即完成优化**，而其他方法耗时约 1000 秒仍未达同等精度。
- **流形适应性验证**：在方向-尺度分解的流形上，EMNGD 成功收敛，表明其对非欧参数结构的有效支持。

### **消融实验与分析**
- **Woodbury 正确性验证**：
  - Primal（参数空间）与 dual（样本空间）方向一致，相对误差达 $8.91×10^{-9}$，验证实现正确。
- **Nyström 近似影响**：
  - Rank-900 Nyström preconditioning 几乎完全跟随 Woodbury 曲线，最终误差仅略高（$1.45×10^{-8}$ vs $1.09×10^{-8}$）。
  - Sketch-and-solve 若秩不足会导致方向偏差，但 preconditioning 可逐步修正。
- **残差数量影响（N=1k → 10k）**：
  - Woodbury/Nyström 依然有效，在更大样本下仍能快速收敛。
  - SPRING 等随机方法波动增大，稳定性下降。
- **参数维度扩展测试（D ≈ 10⁴–10⁵）**：
  - EMNGD 在不同参数规模下均能在 $10^3$ 步内将误差压至 $10^{-7} \sim 10^{-8}$，表现出良好可扩展性。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **EMNGD 统一了参数约束与函数空间几何**：首次将 ENGD 推广至参数流形，实现了**约束优化下的能量自然梯度更新**，兼具理论严谨性与实践有效性。
2. ✅ **Woodbury identity 实现精确降维**：对于二次残差能量，样本空间求解**完全等价于参数空间求解**，不损失方向信息，极大提升了可扩展性。
3. ✅ **Nyström 提供灵活权衡机制**：可通过 sketch rank 控制“精度-速度”权衡——低秩 sketch 加速预处理，迭代后仍可恢复精确方向。
4. ✅ **实验全面验证优越性**：在多个 PDE 任务上，EMNGD 实现了**更快收敛、更高精度、更低运行时间**，尤其适合高维、病态问题。

### **方法的局限性**
- **适用范围限制**：目前主要针对**二次残差能量或 GGN 近似**的情形才能启用 Woodbury 加速；一般非线性能量需额外处理。
- **样本空间瓶颈**：当残差点数 $N$ 极大时（如 $N > 10^4$），样本核 $K \in \mathbb{R}^{N\times N}$ 存储和分解仍昂贵，需依赖 matrix-free 或 subsampling 技术。
- **实现复杂度高**：需实现流形上的切空间操作（projector, retraction）、伴随算子、Woodbury 重构等，工程门槛较高。
- **阻尼敏感性**：过小的阻尼可能导致病态，需 careful tuning 或自适应策略。

### **未来工作方向**
1. **拓展至更多流形结构**：探索权重共享、稀疏性、群不变性等结构对应的流形优化。
2. **开发 matrix-free & distributed solver**：应对超大规模残差系统的内存挑战。
3. **自适应阻尼与 Nyström rank selection**：动态调整以平衡精度与效率。
4. **结合物理先验与几何结构**：进一步融合 PDE 对称性、守恒律等进入流形设计。
5. **应用于真实科学计算场景**：如流体力学、量子系统、生物建模等复杂 PDE 系统。

---

> **总结一句话**：  
> EMNGD 为神经 PDE 求解器提供了**首个兼具几何一致性、理论保证与高效实现的流形自然梯度优化框架**，在精度与效率上全面超越现有方法，是迈向高保真物理模拟的重要一步。

</details>

---

### 10. [Latent PDE mapping for efficient physics-informed learning across geometries with limited data](https://arxiv.org/abs/2607.22215)

**Authors**: Ingvild Askim Adde, Mary M. Maleckar, Gabriel Balaban  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.22215v1  

#### Abstract
In this study, we introduce latent PDE mapping, a broadly applicable physics-informed learning technique designed to enable efficient geometric generalization with sparse training data. Latent PDE mapping pulls back geometry-specific PDE residuals and boundary conditions to a predefined latent geome...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Latent PDE Mapping for Efficient Physics-Informed Learning Across Geometries with Limited Data*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Physics-Informed Machine Learning (PIML)** 方法（如 PINNs）在面对**不同几何形状**时缺乏泛化能力，通常需要为每个新几何重新训练模型，严重削弱了其“快速求解参数化 PDE”的优势。尤其是在**训练数据稀疏**（limited data）和**几何多样性高**的场景下，这一问题尤为突出。

此外，现有方法在计算物理损失（physics loss）时，忽略了**形状梯度**（shape gradient）——即 PDE 解对几何变化的敏感性，导致优化过程中梯度信息不完整，影响模型在未见几何上的表现。

### 🚀 提出的新方法：Latent PDE Mapping (LPM)
作者提出了一种名为 **Latent PDE Mapping (LPM)** 的新框架，其核心思想是：
- 将一个预定义的**潜空间几何**（latent geometry, $\Omega_0$）作为所有物理几何 $\Omega(s)$ 的参考。
- 利用从有限元法（FEM）和非线性固体力学中借鉴的**形变梯度**（deformation gradient $F$）和**雅可比行列式**（Jacobian $J$），将物理几何上的 PDE 残差（residual）和边界条件（BC）残差**拉回**（pull back）到潜空间中。

这样，原本依赖于具体几何 $\Omega(s)$ 的 PDE 被转换为在固定潜空间 $\Omega_0$ 上定义的等效形式，从而实现了：
- **自动且准确地计算形状梯度**（$\partial \mathcal{L}_{\text{phys}} / \partial s$），解决了传统方法中因忽略边界运动项而导致的梯度缺失问题。
- 将几何相关的学习问题转化为**参数化的学习问题**，显著提升了模型在未见几何上的泛化能力。

### 🔍 相比现有方法的优势
| 方法类别 | 局限性 | LPM 的改进 |
|--------|------|----------|
| 传统 PINNs | 在不同几何上需重新训练，无法共享表示 | 统一在潜空间训练，支持跨几何泛化 |
| 基于特殊架构的方法（如 Graph NN, PointNet） | 需要复杂网格处理，推理慢，难以推广到新 PDE | 架构无关（architecture-agnostic），可集成到任意网络（PINN, DeepONet 等） |
| 基于 Autoencoder 的潜空间方法 | 需额外预训练，潜空间解释性差 | 显式数学映射，无需训练映射网络，更具可解释性 |
| 参数感知 PINN (PA-PINN) | 仅将几何参数作为输入，未修正物理损失中的梯度 | 不仅输入参数，更通过 LPM 改进物理损失本身的梯度计算 |

> ✅ **核心优势总结**：LPM 是一种**通用、高效、可解释性强**的 PIML 框架，特别适用于**小样本 + 多几何**的学习任务。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用开源 FEM 求解器 **openCARP** 生成合成的心脏电生理数据。
- 求解的是具有挑战性的 **Aliev-Panfilov PDE**（非线性、时间相关、存在尖锐梯度）。
- 几何体分为 **2D 和 3D** 两类，均基于一个基础潜几何（2D 正方形，3D 立方体）进行参数化变形。

#### 几何族（Geometry Families）
| 类型 | 变形方式 | 训练/测试划分 |
|------|--------|-------------|
| `g_exp` / `H_rot` | 扩张（affine scaling） | 内部范围训练（50个），外部范围测试（35个） |
| `g_shear` | 剪切变形 | 同上 |
| `g_nonlin` | 非线性二次变形 | 同上 |
| `g_rot` / `H_rot`, `H_y^*`, `H_z^*` | 旋转（绕轴） | 同上 |

> ⚠️ 测试集包含**分布外**（out-of-distribution）几何，用于评估泛化能力。

### 🧪 实验设置
- **神经网络架构**：
  - **PINN**：全连接网络，8层×64神经元。
  - **PI-DeepONet (PI-DON)**：分支-主干结构，用于学习算子。
- **几何描述符**（Geometric Descriptors）：
  - **显式参数**：仿射/剪切矩阵元素。
  - **隐式参数**：通过 PCA 对采样点降维得到的前两个主成分。
- **训练策略**：
  - 每个几何族仅使用 **10 个训练样本**（20%），强调“limited data”设定。
  - 损失函数为混合损失：$\mathcal{L} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{phys}} + \mathcal{L}_{\text{bc}} + \mathcal{L}_{\text{ic}}$。
  - 使用 Adam 优化器，共训练 5000 轮。

### 📈 评估指标
- **相对 L2 误差**（relative $L^2$ error）：
  $$
  e_{L^2} = \frac{\|V_{\text{pred}} - V_{\text{FEM}}\|_2}{\|V_{\text{FEM}}\|_2}
  $$
  报告所有测试几何上的平均值及标准差。
- **统计显著性检验**：使用 **paired Wilcoxon signed-rank test**（5% 显著性水平）判断性能差异是否显著。

### 🆚 基线方法对比
| 模型 | 是否使用潜几何 | 是否使用 LPM | 输入参数 |
|------|----------------|---------------|---------|
| **LPM-PINN / LPM-DON** | ✅ | ✅ | ✅ |
| **LG-PINN / LG-DON** | ✅ | ❌ | ✅ |
| **PA-PINN** | ❌ | ❌ | ✅ |
| **Basic-PINN** | ❌ | ❌ | ❌ |

> - **LPM** 是本文提出的方法。
> - **LG** 表示仅将数据映射到潜空间，但**未拉回 PDE 残差**。
> - **PA-PINN** 是典型参数感知模型，直接在物理空间操作。

---

## 3. 主要实验结果和性能指标

### 📉 关键性能数据（来自 Table 3 & 4）

#### 在 2D 外部测试集（external families）上的表现（部分关键结果）：

| Family | 最佳模型 ($e_{L^2}$) | 第二佳模型 ($e_{L^2}$) | 提升倍数 |
|-------|--------------------|----------------------|--------|
| `g_rot*` (rotation) | **LPM-PINN**: 0.060 | LG-PINN: 0.307 | **~5.1×** |
| `g_nonlin*` | **LPM-PINN**: 0.105 | LG-PINN: 0.121 | ~1.15× |
| `g_shear*` | **LPM-PINN**: 0.137 | LG-PINN: 0.172 | ~1.25× |
| `g_exp*` | **PA-PINN**: 0.113 | LPM-PINN: 0.125 | ——（LPM 无优势） |

#### 在 3D 外部测试集上的表现：

| Family | 最佳模型 ($e_{L^2}$) | 第二佳模型 ($e_{L^2}$) | 提升倍数 |
|-------|--------------------|----------------------|--------|
| `H_y^*` | **LPM-PINN**: 0.054 | LG-PINN: 0.273 | **~5.1×** |
| `H_z^*` | **LPM-PINN**: 0.093 | LG-PINN: 0.363 | **~3.9×** |
| `H_x^*` | **LPM-PINN**: 0.076 | LG-PINN: 0.079 | ~1.04×（提升小） |

> ✅ **最大提升出现在旋转类几何**，尤其是改变了纤维方向（fiber orientation）的情况。

### 🔁 与基线方法的对比结果
- **LPM-PINN** 在 16 个 2D 实验中 **11 次优于基线**，其中 **9 次达到统计显著**。
- **LPM-DON** 在 16 个 2D 实验中 **14 次优于基线**，其中 **13 次显著**。
- 在 3D 实验中，LPM-PINN 在全部 12 个任务中均表现最佳，且 11 次显著优于第二名。

### 🔍 消融实验结果
- **LPM vs LG**：证明了**仅仅将数据映射到潜空间是不够的**，必须同时拉回 PDE 残差才能获得显著收益。
- **LPM vs PA-PINN**：表明当几何变化简单（如扩张、剪切）时，PA-PINN 已足够；但在复杂变形（如旋转）下，LPM 因提供了正确的梯度信息而大幅领先。
- **使用 PCA 描述符 vs 显式参数**：LPM 在两种描述符下均有效，说明其**不依赖于显式的变形参数**，具备实际应用潜力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Latent PDE Mapping 显著提升了几何泛化能力**，尤其在训练数据稀少的情况下。
2. **最大的性能增益出现在几何变化引起显著边界运动和 PDE 解剧烈变化的场景**（如旋转导致各向异性方向改变）。
3. **传统 PIML 方法忽略了重要的边界形状梯度项**，LPM 通过数学变换恢复了这部分信息，使梯度更准确。
4. LPM 是**架构无关**的，成功应用于 PINN 和 PI-DeepONet，并保持了良好的推理效率。
5. 即使使用 **PCA 等低维几何描述符**，LPM 依然有效，增强了其在真实场景中的适用性。

### ⚠️ 方法的局限性
1. **依赖已知的形变梯度 $F$ 和雅可比 $J$**：当前假设形变映射已知，而在真实医学图像中，需从点云或表面估计 $F$，这可能引入误差。
2. **尚未验证于更复杂的几何形变**：目前仅测试了参数化变形，未来需扩展到统计形状模型（SSM）或 B-spline 等更灵活的表示。
3. **未系统研究潜几何的选择影响**：潜几何的选取可能影响性能，但文中未深入探讨。
4. **仅在一个 PDE（Aliev-Panfilov）上验证**：尽管该 PDE 很具挑战性，但仍需在更多类型的 PDE（如 Navier-Stokes, elasticity）上验证通用性。
5. **未评估大训练集下的表现**：LPM 的优势在小数据下明显，但在大数据下是否仍具优势尚不清楚。

### 🔮 未来工作方向
1. 开发从离散几何数据（如点云、mesh）中**数值估计 $F$ 和 $J$** 的方法，以适应真实世界数据。
2. 将 LPM 与**统计形状模型**（Statistical Shape Models）结合，用于患者特异性建模。
3. 探索更灵活的**局部非线性变形参数化方法**（如 B-spline, RBF）。
4. 研究 LPM 在**时间依赖几何**（如跳动的心脏）中的应用。
5. 进行更系统的**多随机种子实验**，以评估结果的鲁棒性。

---

> 💡 **总体评价**：  
> 本文提出的 **Latent PDE Mapping** 是一个理论严谨、实用性强的创新框架，它通过引入固体力学中的形变映射思想，从根本上改进了 PIML 中的梯度计算机制。实验证明其在小样本、多几何场景下能带来**数量级的误差下降**，为构建真正可泛化的物理机器学习模型提供了重要路径。

</details>

---

### 11. [IFCLoRA: Topology-Aware Rank Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2607.22251)

**Authors**: Wei Zhang, Xinwu Liu, Yihang Cheng  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.22251v1  

#### Abstract
Low-Rank Adaptation (LoRA) is a widely used parameter-efficient fine-tuning method for large language models, but its performance depends strongly on how a fixed rank budget is distributed across Transformer modules. Existing adaptive-rank methods usually rely on local gradient statistics collected ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# IFCLoRA: Topology-Aware Rank Allocation for Parameter-Efficient Fine-Tuning 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在参数高效微调（**Parameter-Efficient Fine-Tuning, PEFT**）中，**LoRA**（Low-Rank Adaptation）是一种广泛使用的低秩适配方法，其标准实现对所有Transformer模块分配相同的**rank**（即低秩矩阵的维度）。然而，在低预算（low-rank regime）下，不同模块对任务的重要性存在显著差异，统一的rank分配无法有效利用有限的适应容量。

现有自适应方法（如 **AdaLoRA**, **EVA**）虽然尝试动态调整rank，但通常依赖于训练过程中的局部梯度或激活统计量，引入额外计算开销，且忽略了**任务条件下的全局信息流拓扑结构**在rank分配中的作用。

### 提出的新方法：IFCLoRA
本文提出 **IFCLoRA**（**Information-Flow Centrality LoRA**），一种**拓扑感知的、预微调阶段一次性完成的rank分配框架**。其核心思想是将LoRA可插入模块视为一个**任务条件下的稀疏交互图**（interaction graph）中的节点，并基于该图的全局信息流结构来指导rank分配。

#### 创新点：
1. **任务条件的交互图构建**（Task-Conditioned Interaction Graph）：
   - 在**冻结的预训练模型**上，使用一个小的校准集（calibration set）进行轻量级追踪（lightweight tracing）。
   - 通过**零消融**（zero ablation）干预源模块，观察目标模块的响应变化，量化模块间的**影响强度**，从而构建有向加权图。

2. **信息流中心性**（Information-Flow Centrality, IFC）：
   - 提出一个新的重要性评分 **IFC**，它结合了：
     - **拓扑先验**（Topology Prior）：基于图的前向可达性和后向可达性，衡量一个节点是否位于从输入到输出的关键传播路径上。
     - **局部敏感性**（Local Sensitivity）：基于小样本上的梯度信息，衡量模块对当前任务的局部敏感度。
   - 采用乘法融合形式 `I = (f + ε)^(1-ρ) * (g + ε)^ρ`，确保高分节点必须同时具备**结构性重要性**和**任务相关性**。

3. **一次性、预算约束的rank分配**（One-shot, Budget-Constrained Allocation）：
   - 在微调开始前，基于IFC分数，通过带温度的softmax和残差补偿机制，一次性为每个模块分配整数rank，满足总rank预算约束。

### 相比现有方法的优势
- **更高效**：相比AdaLoRA等训练时动态调整的方法，IFCLoRA的额外开销仅存在于预处理阶段，训练时成本与标准LoRA相当。
- **更有效**：利用了**全局信息流拓扑**这一新的信号源，超越了仅依赖局部统计量的方法。
- **可解释性强**：学习到的非均匀rank分布可以追溯到图结构和任务条件下的扰动响应，提供了对“为何某些模块更重要”的直观理解。

---

## 2. 核心实验方法和设置

### 数据集
- **数学推理**：**GSM8K**（小学数学应用题）
- **通用语义理解**：**SuperGLUE**（包含BoolQ, CB, ReCoRD, RTE, WSC.fixed等多个子任务）

### 模型
在三种主流大模型上进行实验：
- **LLaMA3-8B**
- **Qwen3-8B**
- **Qwen3-14B**

### 实验设置
- **LoRA目标模块**：注意力投影层（`q`, `k`, `v`, `o`）和MLP投影层（`gate`, `up`, `down`）。
- **Rank预算**：主要研究低秩场景 `r=4` 和 `r=8`（指平均或总rank预算）。
- **训练细节**：
  - 学习率：1e-4
  - Epochs：3
  - Batch size：16
  - 优化器：默认配置
  - 随机种子：重复3次取均值±标准差。
- **校准集**：使用128个训练样本进行预处理追踪。

### 评估指标
- **GSM8K**：精确匹配准确率（exact-match accuracy）
- **SuperGLUE**：各子任务官方指标（如准确率、F1等）

### 基线方法对比
- **LoRA**：标准均匀rank分配。
- **AdaLoRA**：训练时动态调整rank的代表方法。
- **EVA**：基于下游激活方差进行初始化和重分配的方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### GSM8K 数学推理任务
在所有三个模型和两个rank预算下，**IFCLoRA均取得最佳性能**。

| 模型 | 方法 | r=4 | r=8 |
| :--- | :--- | :--- | :--- |
| **LLaMA3-8B** | LoRA | 64.14 | 64.37 |
| | **IFCLoRA** | **65.50** (+1.36) | **66.19** (+1.82) |
| **Qwen3-8B** | LoRA | 80.97 | 81.73 |
| | **IFCLoRA** | **82.33** (+1.36) | **82.41** (+0.68) |
| **Qwen3-14B** | LoRA | 84.84 | 84.69 |
| | **IFCLoRA** | **85.44** (+0.60) | **85.67** (+0.98) |

- **平均提升**：相比LoRA，IFCLoRA在GSM8K上平均提升 **1.13个百分点**。
- **最大提升**：在LLaMA3-8B上达到 **1.82个百分点**的增益。

#### SuperGLUE 语义理解任务
IFCLoRA表现**任务依赖但总体有利**：
- 在 **Qwen3-14B** 上，IFCLoRA在所有报告的SuperGLUE子任务上均优于或持平于基线。
- 在 **Qwen3-8B** 和 **LLaMA3-8B** 上，多数任务表现最佳或具有竞争力，尤其在 **BoolQ, CB, WSC.fixed** 等任务上优势明显。

#### 与AdaLoRA的对比
- **IFCLoRA consistently outperforms AdaLoRA** on GSM8K across all settings.
- 表明**预微调的拓扑感知分配**比**训练时的动态调整**在低秩数学推理任务中提供了更一致、更有效的信号。

### 消融实验结果（Ablation Study）
在 **Qwen3-8B (GSM8K, r=4)** 上进行消融，验证各组件必要性：

| 方法 | 准确率 (%) |
| :--- | :--- |
| **IFCLoRA (Full)** | **82.33** |
| w/o Gradient (无梯度校准) | 81.88 |
| w/o Topology (无拓扑先验) | 81.13 |
| w/o zero ablation (无零消融) | 81.56 |
| **LoRA (Baseline)** | 80.97 |

- **关键发现**：
  - 移除**拓扑先验**（w/o Topology）导致性能下降最多（-1.2），说明全局信息流结构至关重要。
  - 移除**梯度校准**也有显著影响（-0.45），表明任务特定的局部敏感性不可或缺。
  - 两者结合的**互补性**得到证实。

---

## 4. 关键结论和发现

### 主要发现
1. **非均匀rank分配更优**：在低预算PEFT中，将有限的适应容量集中在关键模块上，比均匀分配更有效。
2. **全局信息流拓扑是重要先验**：任务条件下的模块间交互图及其拓扑结构（特别是source-to-sink的可达性）为rank分配提供了强而有效的信号。
3. **IFCLoRA的有效性**：通过结合**拓扑先验**（IFC）和**局部敏感性**，IFCLoRA实现了更高效的容量利用，在多个模型和任务上持续超越LoRA、AdaLoRA和EVA等基线。
4. **可解释的分配模式**：IFCLoRA倾向于将更高rank分配给中后段的FFN模块和部分注意力头，这与先前关于FFN作为“key-value memory”的发现一致，支持了其合理性。

### 方法的局限性
1. **启发式设计**：IFC分数是一个启发式度量，缺乏理论最优性保证。
2. **架构假设**：图构建依赖于Decoder-only模型的前向因果顺序，其在Encoder-Decoder或MoE架构上的适用性尚待验证。
3. **评估范围有限**：实验集中在数学推理和语义理解任务，未覆盖代码生成、长文本推理等其他领域。
4. **计算开销**：尽管训练时开销低，但预处理阶段的图构建需要额外时间（约5分钟）。

### 未来工作方向
1. **降低图构建成本**：探索更高效的电路追踪方法。
2. **自适应再校准**：在微调过程中根据需要动态更新IFC分数。
3. **更强的因果验证**：使用更严格的因果干预来验证IFC识别出的瓶颈模块。
4. **扩展应用场景**：将IFCLoRA应用于代码生成、长上下文推理、指令微调、多模态适应等任务。
5. **探索其他PEFT设计**：将拓扑感知思想用于指导LoRA的插入位置（insertion location）或模块类型选择。

</details>

---

### 12. [J-CoT: Chain-of-Thought in J-Space](https://arxiv.org/abs/2607.21981)

**Authors**: Junde Wu, Jiayuan Zhu, Fengling Liu, Minhao Hu, Jiazhen Pan  
**Category**: cs.CL  
**Published**: 2026-07-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.21981v1  

#### Abstract
Chain-of-thought prompting improves language-model reasoning by carrying intermediate states across successive computation steps. However, relying on natural language as the only recurrent interface is overly restrictive, since many transient computations do not need to be fully verbalized. Existing...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：J-CoT: Chain-of-Thought in J-Space

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Chain-of-Thought (CoT)** 推理依赖自然语言作为中间状态的唯一传递接口，要求每一步推理都必须被完整地“说出”为流畅的文本。这带来了以下限制：
- 强制模型花费计算资源在语法、连贯性和显式表达上；
- 过早将未成熟的内部表示固化为特定语言解释，造成**过早承诺（premature commitment）**；
- 无法有效保留分布式或多路径的潜在假设。

另一方面，**latent-reasoning 方法**（如 Coconut）虽然通过直接传递 dense hidden states 避免了语言序列化，但其传递的是整个隐藏向量，缺乏对信息的选择与组织机制，导致噪声传播和无效路径持续存在。

### 提出的新方法或新思路
本文提出 **J-CoT**（J-space Chain-of-Thought），一种基于 **J-space** 的新型循环推理框架。其核心思想是：
- 在每个推理周期中，模型仍使用完整的 hidden space 进行自由计算；
- 在周期边界处，仅传递一个**词汇索引的系数状态（vocabulary-indexed coefficient state）**，称为 **J-thought**；
- J-thought 不是一个解码后的句子，也不是完整的 hidden vector，而是以模型词表为坐标的稀疏激活模式。

该方法的关键创新在于引入了 **J-space** 作为一种**模型原生（model-native）的读写接口**：
- 利用 J-space 的跨层共享词汇索引特性，实现不同 Transformer 层之间的状态可迁移性；
- 使用非负弹性网络分解（nonnegative elastic-net decomposition）从 carrier 位置提取 J-thought；
- 通过字典重建将其重新注入模型进行下一轮推理。

### 相比现有方法的优势
| 方法类型 | 优点 | 缺点 |
|--------|------|-------|
| **Explicit CoT** | 可读性强，易于调试 | 强制语言化，易过早承诺 |
| **Dense Latent Reasoning** | 保留非语言信息 | 无选择机制，噪声累积 |
| **J-CoT（本文）** | ✅ 语言可解释性 + ✅ 信息选择 + ✅ 跨层传输稳定性 | —— |

- **无需训练即可生效（J-CoT-Zero）**：固定组件即可超越最强的 latent baseline；
- **支持可学习扩展（J-CoT-Train）**：优化 carrier embeddings 和 read gate 后进一步提升性能；
- **更高效的推理控制**：J-thought 天然支持对活跃概念的选择与更新。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖四大类任务，共 **8 个基准数据集**：

| 类别 | 数据集 | 说明 |
|------|--------|------|
| 数学推理 | GSM8K, MATH-500, AIME 2024 | 数学应用题与竞赛题 |
| 科学问答 | GPQA-Diamond | 高难度研究生级别科学问题 |
| 代码生成 | HumanEval+, MBPP+ | 函数级代码补全 |
| 代码执行推理 | LiveCodeBench, CRUXEval | 包含输入输出预测与程序行为理解 |

### 实验设置和评估指标
- **主干模型**：Qwen3-8B-Base（36层，hidden dim=4096）
- **推理配置**：
  - 添加 8 个 non-linguistic **carrier positions**；
  - J-thought 读入层：layer 12，写出层：layer 28；
  - 最大循环次数：8 次；
  - 自适应终止条件：当连续两次 J-thought 变化小于阈值（0.02）时停止。
- **评估指标**：
  - 数学/科学任务：Exact Match Accuracy
  - 代码生成：pass@1
  - CRUXEval：input/output prediction accuracy 的 macro-average
- **所有方法共享相同的初始 checkpoint 和 prompt template**

### 基线方法对比
| 类型 | 方法 | 描述 |
|------|------|------|
| 显式语言推理 | CoT, PS+ | 标准 CoT 与 Plan-and-Solve 提示 |
| 隐空间推理 | Coconut, CODI, SIM-Coconut | 使用 continuous latent states 进行递归推理 |
| 本文方法 | **J-CoT-Zero**, **J-CoT-Train** | 分别对应零训练和可训练版本 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 方法 | 平均得分（8项平均） |
|------|------------------|
| CoT | 45.8 |
| PS+ | 46.8 |
| Coconut | 45.0 |
| CODI | 46.5 |
| SIM-Coconut（最强 latent baseline） | **47.5** |
| **J-CoT-Zero** | **47.9** ✅ |
| **J-CoT-Train** | **50.2** ✅✅ |

> ✅ J-CoT-Zero 在无额外训练的情况下已超过最强 latent baseline  
> ✅✅ J-CoT-Train 进一步显著领先，提升 **+2.7 pts**

#### 单项最佳表现举例：
- **MATH-500**: J-CoT-Train 达到 **54.0**（vs. SIM-Coconut 的 50.4），**+3.6 pts**
- **HumanEval+**: **62.6** vs. 59.8，**+2.8 pts**
- **CRUXEval**: **59.8** vs. 56.6，**+3.2 pts**
- **GSM8K**: **86.1** vs. 84.0，**+2.1 pts**

### 与基线方法的对比结果
- **J-CoT-Zero** 在所有 8 个 benchmark 上均 **匹配或超越 SIM-Coconut**；
- **J-CoT-Train** 在每一项任务上都取得 **最高分**；
- 改进效果在更具挑战性的任务（如 MATH-500、CRUXEval）中更为明显；
- 性能增益来自更有效的中间状态管理，而非单纯增加计算量。

### 消融实验结果（Ablation Study）

| 消融变体 | 平均准确率（4项平均） |
|----------|---------------------|
| 完整 J-CoT | **53.9** |
| w/o J-Thought Reading | 49.5 ❌ |
| 使用 Learnable Latent State 替代 J-space | 50.9 ❌ |
| 使用 Learnable Transport Adapter | 51.5 ❌ |
| 单个 Carrier | 51.8 ❌ |
| 无 Carrier 间注意力 | 52.6 ❌ |
| 固定大小 J-thought (k=6) | 53.0 ❌ |

> 结论：多 carrier 设计、inter-carrier attention、adaptive coefficient extraction 均对性能有正向贡献；**J-space 的固定结构优于可学习替代方案**，表明预训练模型中已存在可用的语义坐标系统。

---

## 4. 关键结论和发现

### 主要发现
1. **J-space 是一个有效的中间推理接口**：
   - 提供了一种介于 explicit CoT 与 dense latent recurrence 之间的“黄金中点”；
   - 支持语言可解释性的同时避免强制语言化。

2. **J-thought 比 decoded rationale 更高效，比 dense state 更可控**：
   - 实验证明，在接口谱系插值实验中（Figure 3），**λ=0.5（即 J-CoT 配置）达到峰值性能 88.8%**，高于两端（explicit λ=1: 79.0%，dense λ=0: 84.0%）；
   - 表明 J-thought 成功平衡了“保留多样性”与“抑制无效路径”的矛盾。

3. **良好的可扩展性**：
   - 随着 backbone 规模增大（从 7B 到 405B），J-CoT 的收益持续增长；
   - 更深的 reasoning depth（Heavy 模式）带来更大提升，尤其在大模型上；
   - 显示 J-CoT 能有效利用未来更大容量的 LLM。

4. **无需训练也能工作良好（J-CoT-Zero）**：
   - 表明预训练模型中已经隐含了可用的 J-space 结构；
   - 降低了部署门槛，适合快速迁移。

### 方法的局限性
- 当前 J-thought 的提取依赖于 **J-lens 字典**，构建需要估计 Jacobian，有一定计算开销；
- 多 carrier 的设计增加了实现复杂度；
- 尚未探索与其他 reasoning 架构（如 Tree of Thoughts）结合的可能性；
- 对 extremely long-horizon reasoning 的有效性尚待验证。

### 未来工作方向
- 探索 J-thought 的可视化与解释性分析，用于模型诊断；
- 将 J-CoT 与 verification 或 self-correction 机制结合；
- 扩展至多模态 reasoning 场景；
- 研究如何动态调整 read/write layers 以适应不同任务；
- 探索更轻量化的 J-lens 构建方式，便于小模型部署。

---

> **总结一句话**：  
> J-CoT 提出了一种**语言锚定但非句子化**的中间状态表示——**J-thought**，通过 **J-space** 实现跨层稳定的循环推理，在不牺牲计算自由度的前提下，实现了比显式 CoT 和稠密隐状态更优的推理效率与性能，代表了下一代 reasoning 接口的重要方向。

</details>

---

### 13. [Encoding Invisible Causation for Bridge Diagnostic Agents: Triple-Guided Retrieval-Augmented Fine-Tuning with QLoRA](https://arxiv.org/abs/2607.21680)

**Authors**: Takato Yasuno  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.21680v1  

#### Abstract
Bridge infrastructure deteriorates gradually, yet its root causes---salt intrusion, freezing, fatigue cracking, and others---remain invisible to the naked eye. Expert diagnosis relies on tacit knowledge built over years of practice. We address the challenge of automating this latent causal reasoning...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题**: *Encoding Invisible Causation for Bridge Diagnostic Agents: Triple-Guided Retrieval-Augmented Fine-Tuning with QLoRA*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
桥梁基础设施在长期服役过程中会因盐蚀、冻融、疲劳开裂等**不可见的潜在原因**导致可见损伤（如裂缝、剥落）。尽管表面损伤可通过视觉检测识别，但其**根本成因难以自动判断**，依赖专家多年积累的**隐性知识（tacit knowledge）**。现有AI系统多聚焦于损伤检测与分类，缺乏对“**为何发生损伤**”的因果推理能力。

本文旨在解决这一“**可见损伤 → 不可见成因**”的诊断鸿沟，实现自动化、可解释的根因推断。

---

### 🚀 提出的新方法与新思路

提出 **Triple-Guided Retrieval-Augmented Fine-Tuning** 框架，结合以下三大组件：

1. **Knowledge Triple Extraction**  
   利用大语言模型（LLM）从15–35份桥梁诊断PDF手册中提取结构化因果三元组（triples），形式为 `(damage cause, relation, explanation)`，例如：  
   `(rebar corrosion, caused_by, chloride ion concentration exceeding 1.2kg/m³)`。共提取 **6,745条因果三元组**，并建立 FAISS 向量索引。

2. **Retrieval-Augmented Context**  
   在训练和推理阶段，通过检索与当前损伤描述最相关的三元组，将其作为显式上下文拼接到输入中，将隐性领域知识转化为模型可理解的显式提示。

3. **Systematic Fine-tuning Comparison**  
   对比 LoRA、QLoRA 和 QA-LoRA 三种参数高效微调策略，在统一设置下评估其性能、速度、内存消耗与泛化能力。

此外，构建了一个高质量的 **Golden Testset**（116个样本），具备分层抽样、去重、难度标注等特点，作为可复现的基准测试集。

---

### 🔍 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **知识利用方式** | 将非结构化的PDF专家知识转化为结构化 triple，并通过 RAG 注入模型，优于纯文本微调或端到端训练 |
| **模型效率** | 使用 QLoRA 实现 **4-bit量化 + LoRA适配器**，大幅降低GPU内存需求（↓72%）且不损失精度 |
| **泛化能力** | QLoRA 在多样化未见输入上表现显著优于 LoRA（+13个百分点），表明量化具有**隐式正则化作用** |
| **部署可行性** | 支持在消费级硬件（如 RTX 4060 Ti, 16GB GPU）上运行，适合边缘部署 |

---

## 2. 核心实验方法和设置

### 📚 数据集

| 版本 | 来源PDF数 | 三元组数 | 训练样本数 | 类别不平衡比（IR） | 特点 |
|------|-----------|----------|-------------|---------------------|------|
| v0.2 (原始) | 15 | 4,186 | 428 | 35.1 | 极度不平衡（Salt类占65.7%） |
| v0.3 (平衡) | 15 | 4,186 | 388 | 7.7 | 应用采样控制，提升小类覆盖 |
| v0.4 (扩展) | 35 | 6,745 | 642 | 7.69 | 扩展桥型类型（涵洞、拱桥等） |
| **Golden Testset** | — | — | **116**（测试集） | — | 分层、去重、难度标签（Easy/Medium/Hard） |

> Golden Testset 是本文的重要贡献之一，确保不同方法间的公平比较。

---

### ⚙️ 实验设置

- **基础模型**: `cl-tohoku/bert-large-japanese-v2`（344M参数）
- **任务**: 多类别分类（10类损伤成因，见 Table 1）
- **输入格式**: `[CLS] S [SEP] o₁·o₂·...·oₖ [SEP]`，其中 $S$ 为损伤描述，$o_i$ 为 retrieved triple 的 object 字段
- **微调策略对比**:
  - **LoRA**: 标准低秩适配（FP32 backbone）
  - **QLoRA**: 4-bit NF4 量化 backbone + LoRA（bfloat16 adapters）
  - **QA-LoRA**: 加入 L2 正则以补偿量化误差（本文近似实现）

- **训练配置**:
  - Batch size: 8 × 4（梯度累积）
  - Epochs: 20
  - Optimizer: AdamW, lr=2e-4
  - Loss: Weighted Cross-Entropy（按类别频率加权）

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 主要性能指标 |
| **Weighted F1 Score** | 考虑类别不平衡的综合评价 |
| **Inference Speed (ms)** | 单次预测延迟 |
| **GPU Memory Usage (GB)** | 显存占用 |
| **Generalization on Diverse Inputs** | 在100个多样化未见样本上的准确率（跨位置、机制、结构元素） |

---

### 🆚 基线方法对比

直接对比三种 PEFT 方法：
- LoRA（全精度基线）
- QLoRA（推荐方案）
- QA-LoRA（尝试纠正量化误差）

所有方法共享相同 LoRA 配置（r=16, α=32）、训练超参和数据集划分。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Golden Testset, 116 samples）

| 方法 | Accuracy (%) | F1w | 推理速度 (ms) | GPU 内存 (GB) |
|------|--------------|-----|----------------|----------------|
| **LoRA** | **87.07** | 0.870 | 44.0 | 1.45 |
| **QLoRA** | **87.07** | 0.869 | **39.2** (-11%) | **0.40** (↓72%) |
| **QA-LoRA** | 85.34 | 0.854 | 41.5 | 0.42 |

> ✅ **QLoRA 在保持完全相同的测试准确率前提下，推理快11%，显存仅需0.4GB**

---

### 🔄 泛化能力对比（100个多样化未见输入）

| 方法 | 准确率 |
|------|--------|
| **QLoRA** | **47.0%** |
| LoRA | 34.0% |
| QA-LoRA | 30.0% |

> 💡 尽管在 Golden Testset 上精度相同，QLoRA 在分布外样本上**领先13个百分点**，说明 **4-bit量化噪声起到了隐式正则化作用**，增强泛化。

---

### 🔬 消融实验与关键发现

#### （1）数据平衡的影响（v0.2 vs v0.3）

| 设置 | LoRA Acc | QLoRA Acc | ΔAcc (QLoRA - LoRA) |
|------|---------|----------|--------------------|
| 不平衡 (IR=35.1) | 91.86% | 90.70% | -1.16% |
| 平衡 (IR=7.7) | 83.33% | 85.90% | **+2.57%** |

> 当类别平衡时，QLoRA 反超 LoRA，验证了“**量化作为正则化器**”的有效性。

#### （2）语料规模扩展（v0.3 → v0.4）

- 扩展至35份PDF后，训练样本增至642，类别更均衡。
- QLoRA 准确率从 **85.90% → 91.47%**，接近原始不平衡数据的最佳水平，同时保持良好泛化。

#### （3）失败模式分析（Persistent Failures）

在多样化输入中，以下类别几乎全部失败：

| 类别 | QLoRA 准确率 | LoRA 准确率 | 问题原因 |
|------|-------------|------------|---------|
| **Water Accumulation** | 0% | 0% | 缺乏判别性训练样本 |
| **Soil Liquefaction** | ≤10% | ≤10% | 描述稀疏，特征模糊 |
| **ASR (Alkali-Silica Reaction)** | ≤20% | ≤20% | 表面症状与其他裂缝高度相似 |

> 这些是未来**数据增强的重点目标**。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **Triple-Guided RAG 有效编码隐性因果知识**  
   通过从PDF手册中提取因果三元组并在推理时动态注入，使 BERT 模型能够模拟专家的深层诊断逻辑。

2. **QLoRA 是最优微调策略**  
   在 Golden Testset 上达到与 LoRA 相同精度（87.07%），但具备：
   - 更低显存（0.40GB vs 1.45GB）
   - 更快速度（↓11%）
   - 更强泛化能力（+13% on diverse inputs）

3. **量化不仅是压缩工具，更是正则化手段**  
   在类别平衡（IR < 10）的数据上，QLoRA 表现优于 LoRA，揭示了 **4-bit量化噪声抑制过拟合、促进泛化**的作用。

4. **Golden Testset 成为可复现评估标准**  
   提供一个标准化、难度分级的测试集，推动该领域的公平比较。

---

### ⚠️ 局限性

| 问题 | 说明 |
|------|------|
| **Triple Extraction 质量不稳定** | LLM 提取偶尔出现格式错误或语义偏差，需人工过滤 |
| **部分类别严重欠拟合** | Water Accumulation、Soil Liquefaction、ASR 因样本少且特征模糊，模型无法学习有效判据 |
| **仅支持日语文本输入** | 当前模型局限于日本桥梁文档体系，尚未扩展至多语言或多模态（图像+文本） |
| **LLM Filter 效率低** | 使用 Qwen2.5-7B 进行相关性过滤，增加推理延迟，未来可用轻量分类器替代 |

---

### 🔮 未来工作方向

1. **替换 LLM-based relevance filter** 为可训练的小型分类器，提升推理效率。
2. **开展针对性数据增强**，特别是针对 Water Accumulation、Soil Liquefaction 和 ASR 三类。
3. **探索多模态融合**：结合图示、表格块中的视觉信息，构建 Vision-Language 损伤诊断模型。
4. **横向迁移至其他领域**：应用于隧道衬砌诊断、路面退化评估、机械设备故障检测、医疗症状归因等场景。
5. **验证 IR-based Quantization Guideline**：提出“当 Imbalance Ratio < 10 时优先使用 QLoRA”的经验法则，需在更多领域验证普适性。

---

> 🏁 **最终意义**：本研究证明了在资源受限环境下，通过 **Retrieval-Augmented + QLoRA** 的组合，可在消费级硬件上实现高精度、可泛化的根因诊断，为基础设施智能运维提供了切实可行的技术路径。

</details>

---

### 14. [DWT-Fusion: A Signal-Based Framework for Training-Free LLM-Generated Text Detection](https://arxiv.org/abs/2607.22026)

**Authors**: Mehmet Batuhan \"Ozda\c{s}, Murat Osmano\u{g}lu  
**Category**: cs.CL  
**Published**: 2026-07-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.22026v1  

#### Abstract
Detecting LLM-generated text remains challenging under zero-shot and training-free conditions, especially when detectors must generalize across datasets, domains, and unseen generators. While existing training-free approaches exploit language-model statistics as detection signals, they typically cha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DWT-Fusion: A Signal-Based Framework for Training-Free LLM-Generated Text Detection

## 1. 论文的主要贡献和创新点

### 解决的问题
当前在**zero-shot**和**training-free**条件下检测大语言模型（LLM）生成文本仍具挑战性，尤其是在需要跨数据集、领域和未见生成器泛化时。现有方法多依赖全局统计量（如平均 log-likelihood、rank），忽略了 token-level 可预测性的局部和多尺度变化，导致潜在判别信息未被充分利用。

### 提出的新方法与思路
本文提出 **DWT-Fusion**，一种基于信号处理的训练免费（training-free）框架，用于检测 LLM 生成文本。其核心思想是将由代理因果语言模型（proxy causal language model）生成的 token-level log-probability 序列视为一维信号，并应用**离散小波变换**（Discrete Wavelet Transform, DWT）进行多分辨率分析。

具体流程如下：
1. **信号提取**：从 proxy model 获取每个 token 的条件 log-probability，形成序列。
2. **信号预处理**：对序列进行均值中心化（mean-centering），消除全局偏移。
3. **DWT 分解**：将信号分解为不同尺度的近似系数（approximation）和细节系数（detail coefficients）。
4. **多尺度评分**：从细节系数中提取三种可解释的标量检测分数：
   - **First-level detail energy**：捕捉短尺度波动。
   - **Multilevel detail energy**：聚合多个尺度的局部变化能量。
   - **Window-energy variability**：衡量小波能量在局部区域间的变异性。
5. **无监督集成**：提出四种无需训练的投票融合策略（equal-weight / calibration-weighted hard & soft voting），组合多种小波配置（不同 proxy model、wavelet family、score 定义）以提升性能。

### 相比现有方法的优势
- **保留序列结构**：不同于传统方法将 token 统计压缩为单一全局值，DWT-Fusion 利用信号顺序信息，捕获局部和多尺度动态。
- **真正 training-free**：不训练分类器、不微调模型、不构建参考数据库、不学习元分类器，仅依赖 calibration split 进行方向校正和阈值选择。
- **更强的信号表示能力**：相比基于 DFT 的全局频谱能量，DWT 能定位波动发生的位置和尺度，提供更丰富的判别信息。
- **可解释性高**：提出的三种 wavelet-domain score 具有明确的物理意义，便于理解模型决策依据。

---

## 2. 核心实验方法和设置

### 使用的数据集
在三个基准数据集上进行评估，覆盖不同难度和多样性：
- **HC3**：人类 vs. ChatGPT 回答，相对简单。
- **M4**：多生成器、多领域、多语言的黑盒检测任务，更具异质性。
- **MAGE**：真实场景下的机器生成文本检测，最具挑战性。

### 实验设置
- **Proxy Language Models**：GPT-Neo-2.7B、GPT-J-6B、Falcon-7B、LLaMA-3-8B。
- **Wavelet Families**：db1, db2, db4, sym2, coif1。
- **输入长度**：统一截断至最大 512 tokens。
- **数据划分**：采用 **30% calibration / 70% held-out test** 协议。calibration split 仅用于：
  - 分数方向校正（direction correction）
  - 阈值选择（F1 最大化）
  - 归一化参数计算
  - 确定性加权（如 AUROC-based weighting）
- **投票集成范围**：涵盖 score-level、wavelet-family、proxy-model 和 full configuration ensemble（共 60 种组合）。

### 评估指标
- **Threshold-independent**：
  - **AUROC**（Area Under ROC Curve）
  - **AUPRC**（Area Under Precision-Recall Curve）
- **Threshold-dependent**：
  - **Accuracy**
  - **F1 Score**
- **低误报率操作点**：
  - **TPR@0.1%FPR**, **TPR@1%FPR**, **TPR@5%FPR**

### 基线方法对比
- **零样本统计基线**：mean log-likelihood, mean rank, mean log-rank, mean entropy, LRR。
- **信号处理基线**：DFT-based spectral energy（dft_total_energy）。
- **其他相关方法**：SpecDetect（DFT-based）、WAVEDETECT（监督式 CWT）等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（held-out test split 上的最佳结果）

| 数据集 | 方法 | AUROC | AUPRC | F1 | TPR@1%FPR |
|-------|------|--------|--------|-----|------------|
| **HC3** | Calibration-weighted hard voting | **0.9919** | **0.9914** | 0.9700 | 0.9425 |
| **M4** | Calibration-weighted hard voting | **0.8477** | 0.8603 | 0.7747 | 0.2618 |
| **MAGE** | Calibration-weighted hard voting | **0.7471** | **0.5556** | 0.6111 | 0.0113 |

#### 单一最佳小波配置性能：
- **HC3**: 0.9872 AUROC (multilevel_energy, GPT-Neo, db1)
- **M4**: 0.8185 AUROC (multilevel_energy, LLaMA-3, db2)
- **MAGE**: 0.7138 AUROC (energy_norm, GPT-Neo, db4)

### 与基线方法的对比结果
- 在所有三个数据集上，**最佳单一 DWT 配置**均优于或媲美最强统计基线（如 mean_logrank）。
- **DWT-Fusion 集成方法**显著超越所有基线：
  - 在 HC3 上，AUROC 从 0.9876（best baseline）提升至 **0.9919**。
  - 在 M4 和 MAGE 上优势更为明显，尤其在异构环境下表现稳健。
- **vs. DFT 基线**：DWT 方法全面超越 DFT 总能量：
  - HC3: +0.1741 AUROC
  - M4: +0.0140 AUROC
  - MAGE: +0.1307 AUROC  
  表明**局部多尺度分析**远优于**全局频谱能量**。

### 消融实验与敏感性分析
- **投票机制有效性**：calibration-weighted voting 显著优于 equal-weight voting，说明基于 calibration 性能的加权策略有效。
- **Proxy Model 敏感性**：
  - HC3：GPT-J-6B 表现最好（平均 AUROC 0.8832）
  - M4：LLaMA-3-8B 明显领先（0.7730）
  - MAGE：GPT-Neo-2.7B 最优（0.6742）
  → Proxy model 选择对性能影响显著，无单一最优模型。
- **Wavelet Family 敏感性**：
  - db2 和 sym2 在 HC3 和 M4 上表现稳定。
  - coif1 在 MAGE 上略优。
  → 相比 proxy model，wavelet family 影响较小。

---

## 4. 关键结论和发现

### 主要发现
1. **局部多尺度信号分析有效**：token-level log-probability 的局部和多尺度波动是区分人类与 LLM 文本的重要信号，DWT 能有效提取此类信息。
2. **DWT 优于 DFT**：相比全局频谱分析，DWT 的局部定位能力使其在所有数据集上均取得更好性能，验证了“波动位置”比“总能量”更重要。
3. **集成显著提升性能**：通过 calibration-guided voting 融合多种小波配置，可在不引入监督训练的前提下进一步提升 AUROC 和 AUPRC，尤其在复杂数据集（M4/MAGE）上效果显著。
4. **不同分数适用不同场景**：
   - **Multilevel detail energy** 在 HC3/M4 上最优。
   - **First-level detail energy** 在最复杂的 MAGE 上表现最好，可能因其对噪声更鲁棒。
   - **Window-energy variability** 虽整体较弱，但在 MAGE 上达到最高的 TPR@5%FPR，表明其在特定操作点仍有价值。

### 方法的局限性
1. **依赖 Proxy Model**：性能受 proxy model 与真实生成器匹配度影响，存在分布偏移风险。
2. **非完全 calibration-free**：虽不训练模型，但仍需 calibration split 进行方向校正、阈值选择和加权，属于“calibration-guided”而非完全无数据依赖。
3. **输入长度限制**：固定 512 token 截断可能丢失长文本后半部分的判别信息。
4. **低 FPR 下性能有限**：在 MAGE 上，即使最优方法在 TPR@0.1%FPR 下也极低（~0.0003），难以满足实际部署中极低误报需求。
5. **现实场景未覆盖**：未测试编辑、改写、人机协作等复杂写作模式。

### 未来工作方向
- 探索**跨数据集 calibration** 或**无监督 calibration** 策略。
- 设计更稳定的低 FPR 优化方法。
- 引入**滑动窗口**或**长上下文模型**以支持更长文本分析。
- 构建**混合检测器**，结合 wavelet-domain 信号与统计、结构或其他互补特征。
- 在**真实世界写作场景**（如改写、混合创作）下评估方法鲁棒性。
- 探索**自适应分解深度**以适配不同长度和结构的输入信号。

</details>

---

### 15. [Enough is as good as a feast: A Comprehensive Analysis of How Reinforcement Learning Mitigates Task Conflicts in LLMs](https://arxiv.org/abs/2607.22039)

**Authors**: Zixuan Ren, Jinliang Lu, Junhong Wu, Yang Zhao, Dai Dai, Hua Wu, Haifeng Wang, Chengqing Zong  
**Category**: cs.CL  
**Published**: 2026-07-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.22039v1  

#### Abstract
Model merging plays a crucial role in consolidating multiple specialized models into a single, unified model, especially in the era of large language models (LLMs). Recent research has primarily focused on developing strategies to enhance merging performance with the trained models, while the impact...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Enough is as Good as a Feast: A Comprehensive Analysis of How Reinforcement Learning Mitigates Task Conflicts in LLMs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文聚焦于**大语言模型（LLMs）在模型合并（model merging）过程中面临的任务冲突（task conflicts）问题**。当多个针对不同任务微调的模型被合并为一个统一模型时，参数更新之间可能产生干扰，导致某些任务性能严重下降。

尽管已有大量研究关注模型合并策略（如 Task Arithmetic、TIEs、DARE），但这些工作普遍假设模型是通过 Supervised Fine-Tuning（SFT）训练的，而忽略了**不同的后训练范式（post-training paradigm）本身对任务冲突的影响**。

### ✅ 提出的新方法/新思路
本论文并未提出新的合并算法，而是首次系统性地揭示并论证了一个关键发现：

> **Reinforcement Learning（RL）训练的 LLMs 天然更适合模型合并，能显著缓解任务冲突。**

作者从理论和实证两个层面分析了这一现象背后的三大机制：
1. **On-policy training data**：RL 使用模型自身生成的数据进行训练，使得梯度更新幅度更小、更稳定，降低覆盖其他任务知识的风险。
2. **“Enough is as good as a feast” 的优化特性**：随着模型收敛，RL 的优势函数（advantage）趋于零，从而自动衰减参数更新强度，避免过度调整。
3. **正负样本联合优化**：RL 同时利用正例（高奖励输出）和负例（低奖励输出）进行学习，引导模型进入更鲁棒、无偏的任务特定参数子空间，减少参数冲突。

### ✅ 相比现有方法的优势
- **无需修改合并算法**：该优势独立于具体的 merging 方法（如 Averaging、TIEs、DARE），具有广泛适用性。
- **跨模型通用性强**：在 Llama-3.1-8B、Llama-3.2-3B、Mistral-Small-24B 等多种 base model 上均成立。
- **跨 RL 算法稳健**：PPO、GRPO、REINFORCE++ 等不同 RL 算法均表现出优于 SFT 的合并兼容性。
- **提供新设计原则**：建议未来构建通用 LLM 时优先采用 RL 进行 post-training，以提升可合并性和多任务集成能力。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验涵盖五个支持自动评估的任务：
| 任务 | 训练数据 | 测试基准 |
|------|--------|---------|
| **Math** | OpenMathInstruct-2 子集 | GSM8K, MATH-500 |
| **Code** | OpenCodeInstruct 子集 | HUMANEVAL, MBPP |
| **Instruction Following (IF)** | Tulu-3-SFT 指令子集 | IFEVAL, LIVEBENCH 指令子集 |
| **Logical Puzzle** | Knights and Knaves 合成数据（模板生成） | 同任务测试集（in-domain 和 OOD） |
| **Ranking** | Rankl 数据集 | NEVIR（基于 MTEB 协议） |

### ⚙️ 实验设置
- **Base Models**：Llama-3.1-8B、Llama-3.2-3B、Mistral-Small-3-24B
- **训练方式对比**：
  - **SFT**：标准监督微调，最小化负对数似然。
  - **RL**：使用 GRPO（默认）、PPO、REINFORCE++ 等算法，最大化期望奖励。
- **模型合并策略**（共四种）：
  1. **Averaging**：直接平均模型权重
  2. **Task-Arithmetic (Arithmetic)**：基于 task vector 加法
  3. **TIEs**：解决合并中的干扰问题
  4. **DARE**：结合稀疏化与任务向量融合
- **评估方式**：两两合并（pairwise merging），报告每个任务与其他任务合并后的平均性能。

### 📊 评估指标
- 主要指标：各任务上的准确率（Accuracy）或 Pass@1
- 性能变化：相对于原始未合并模型的相对性能下降百分比（↓%）
- 辅助分析指标：
  - 参数更新范数（update norm）
  - 冲突范数（conflict norm）
  - 参数符号冲突比例（sign conflict ratio）
  - 损失景观可视化（loss landscape visualization）

---

## 3. 主要实验结果和性能指标

### 🔢 关键性能数据（来自 Table 1）

| 方法 | Math | Code | IF | Puzzle | Rank | **Average** |
|------|------|------|-----|--------|-------|------------|
| **SFT（原模型）** | 61.9 | 60.5 | 63.9 | 86.2 | 52.8 | **61.5** |
| → 经 **Averaging** 合并 | ↓16% | ↓7.4% | ↓23% | ↓65% | ↓2.3% | **↓22%** |
| → 经 **TIEs** 合并 | ↓8.3% | ↓4.1% | ↓25% | ↓58% | ↓2.7% | **↓19%** |
| → 经 **DARE** 合并 | ↓6.1% | ↓4.1% | ↓27% | ↓56% | ↓6.7% | **↓19%** |

| 方法 | Math | Code | IF | Puzzle | Rank | **Average** |
|------|------|------|-----|--------|-------|------------|
| **RL (GRPO)（原模型）** | 64.6 | 65.6 | 90.0 | 85.2 | 55.7 | **72.2** |
| → 经 **Averaging** 合并 | ↓3.9% | ↓5.9% | ↓6.2% | ↓56% | ↓2.3% | **↓17%** |
| → 经 **TIEs** 合并 | ↓2.0% | ↓2.0% | ↓0% | ↓24% | ↓4.7% | **↓7.1%** |
| → 经 **DARE** 合并 | ↓1.7% | ↓2.1% | ↓0.1% | ↓24% | ↓4.7% | **↓7.1%** |

> 💡 **关键观察**：  
> - SFT 模型合并后平均性能下降 **18–22%**，尤其在 Puzzle 上高达 **65%** 下降；
> - RL 模型即使使用最简单的 Averaging，也仅下降 **17%**；使用 TIEs/DARE 可控制在 **7.1%** 以内；
> - 在 IF 任务上，RL 模型合并后几乎无损（↓0.1%）！

### 🔁 不同 RL 算法与 Base Model 的泛化性（Figures 2 & 3）
- **图2（不同 RL 算法）**：PPO、GRPO、REINFORCE++ 在 TIEs 合并下，性能下降均远低于 SFT（SFT ↓28.7%，RL 最多 ↓2.6%）。
- **图3（不同 base model）**：在 Llama-3.2-3B 到 Mistral-24B 上，RL 始终显著优于 SFT，且在更大模型上优势更明显（如 Mistral 上 RL 几乎无损，SFT 下降达 35.6%）。

### 🔍 消融实验与深入分析
#### （1）损失景观分析（Figure 4）
- 对随机扰动（random perturbation），SFT 与 RL 模型都表现鲁棒；
- 但对任务诱导更新方向（task-induced Δθ），**SFT 模型性能迅速下降**，表明存在强烈任务干扰；
- **RL 模型在此方向上保持稳定**，说明其学到的参数更新更具正交性（task-orthogonal）。

#### （2）参数更新与冲突范数（Figures 5 & 6）
- RL 的参数更新范数增长缓慢，远小于 SFT；
- RL 的 conflict norm 显著更低，证明其减少了跨任务的对抗性更新。

#### （3）正负样本作用验证（Figure 7）
- 设计对照实验 RL-Pos（将负样本优势设为0）：
  - 单任务性能：RL > RL-Pos > SFT
  - 合并后性能下降：RL < RL-Pos ≪ SFT
- 表明：**负样本的存在有助于提升单任务性能，并进一步降低任务冲突**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **RL 训练天然抑制任务冲突**：相比 SFT，RL 更适合用于需要后续合并的 LLM 后训练。
2. **三大机制共同作用**：
   - On-policy data 导致更小、更稳定的梯度更新；
   - RL 的优化目标随收敛自动衰减更新强度（“够了就好”）；
   - 正负样本联合优化使参数更新更均衡、更鲁棒。
3. **该优势具有强泛化性**：不依赖特定 merging 方法、base model 或 RL 算法。

### ⚠️ 局限性
- 当前分析集中在**可验证任务**（verifiable tasks），对于开放式生成任务（如创意写作）是否同样有效尚待验证。
- 实验主要基于**单一阶段 RL 微调**，未考虑多轮迭代或复杂奖励建模的影响。
- 虽然解释了“为什么 RL 更好”，但尚未提出如何**主动设计更利于合并的 RL 目标函数**。

### 🔮 未来工作方向
- 探索将 RL 的“自适应更新”机制引入 SFT 中（例如模拟优势衰减）；
- 设计专门面向模型合并的 RL 框架（Merge-aware RL）；
- 扩展到多模态模型（MLLMs）和其他参数高效微调方法（如 LoRA merging）；
- 构建支持动态增量合并的 RL 基础模型架构。

---

## 📌 总结一句话
> **本文揭示了 RL 不仅是一种对齐人类偏好的工具，更因其“够了就好”的优化本质，成为构建可合并、可扩展通用 LLM 的理想后训练范式 —— “Enough is as good as a feast” 不仅是一句谚语，更是 RL 缓解任务冲突的核心哲学。**

</details>

---

### 16. [MEUSLI: a Multilingual Projector for LLM-based ASR and Beyond](https://arxiv.org/abs/2607.22100)

**Authors**: Lorenzo Concina, Seraphina Fong, Marco Matassoni, Alessio Brutti  
**Category**: cs.CL  
**Published**: 2026-07-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.22100v1  

#### Abstract
Lightweight projectors are an established way to connect pre-trained speech encoders with large language models (LLMs), mapping acoustic features into token-level embeddings for tasks like ASR and spoken question answering. Existing systems, however, typically only support a few languages and are of...

---

### 17. [Variance-Reduced Q-Learning over Static and Time-Varying Networks](https://arxiv.org/abs/2607.21876)

**Authors**: Sreejeet Maity, Feng Zhu, Aritra Mitra, Robert W. Heath Jr  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.21876v1  

#### Abstract
We investigate a decentralized reinforcement learning problem involving multiple agents that interact with the same Markov Decision Process (MDP). The agents can exchange information over a network to collectively learn the optimal state-action value function. For this setting, we introduce a novel ...

---

### 18. [Pretraining EHR Foundation Models with Patient-Aware Sampling](https://arxiv.org/abs/2607.22114)

**Authors**: Joshua Placidi, Yuxuan Liu, Jinpei Han, Marek Rei, A. Aldo Faisal  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.22114v1  

#### Abstract
Autoregressive foundation models for electronic health records (EHRs) typically inherit pretraining methods from language modeling, where patient trajectories are concatenated into a single token stream and windows are sampled from that stream. In EHR data, this choice is consequential: windows may ...

---

### 19. [Benchmarking Fine-tuning and Retrieval Strategies for a Multimodal Language Model on the NRC Reactor Operator Licensing Examination](https://arxiv.org/abs/2607.22067)

**Authors**: Isak Hwang, Yoon Pyo Lee  
**Category**: cs.CL  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.22067v1  

#### Abstract
The integration of large language models (LLMs) into the nuclear power industry requires outputs grounded in domain-specific knowledge. This study evaluates a 31-billion-parameter open-weight multimodal model (Gemma 4 31B-IT) on its capacity to apply nuclear knowledge by benchmarking eight model-ret...

---

### 20. [Agentic CPU-GPU Scheduling for Heterogeneous AI Workloads](https://arxiv.org/abs/2607.22242)

**Authors**: Tianxi Lu, Sherief Reda  
**Category**: cs.DC  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.22242v1  

#### Abstract
Agentic AI systems compose heterogeneous tool workloads on shared GPU/CPU infrastructure, yet existing frameworks assign all GPU-capable tools to the GPU by default. We profile 19 AI tools across GPU and CPU and find that 11 are GPU-preferred, 4 are ambiguous, 1 is CPU-preferred due to PCIe transfer...

---

### 21. [A Drift Stable Quantum Federated Learning for Intelligent Services](https://arxiv.org/abs/2607.21647)

**Authors**: Shanika Iroshi Nanayakkara, Shiva Raj Pokhrel  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.21647v1  

#### Abstract
Quantum federated learning enables distributed clients to train quantum neural networks without sharing local data, making it promising for privacy-aware intelligent services. Intelligent services in this context refer to privacy-sensitive distributed decision systems, such as fraud detection and ge...

---

### 22. [CARNet Cycle-Conditioned Core Aggregation and Redistribution for Multivariate Time Series Forecasting](https://arxiv.org/abs/2607.21681)

**Authors**: Awsaf Tausif Adib, Md. Shahria Sarker Shuvo, Md. Estehaar Ahmed Emon, Mustafa Kamal, Fuad Rahman, Shafin Rahman, Nabeel Mohammed  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.21681v1  

#### Abstract
Accurately modeling cross-variate dependencies remains a key challenge in multivariate time series forecasting, particularly in the presence of strong periodic patterns. Many existing approaches rely on attention-based mechanisms that incur quadratic complexity and scale poorly with increasing numbe...

---

### 23. [A Defense of the Quadratic Model](https://arxiv.org/abs/2607.21716)

**Authors**: Alexandru Meterez, Pranav Ajit Nair, Depen Morwani, Cengiz Pehlevan, Sham Kakade, Alex Damian  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.21716v1  

#### Abstract
Due to the complexity of neural network loss landscapes, optimization theory is forced to rely on idealized models, and there is generally a tradeoff between how theoretically tractable the model is, and how accurately it describes the true optimization dynamics. In this work, we stress test the sim...

---

### 24. [Parameter-free Adaptive Sparse Attention via Compression-Based Content Selection](https://arxiv.org/abs/2607.21752)

**Authors**: Debarshi Kundu, Swaroop Ghosh, Vasant Honavar  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.21752v1  

#### Abstract
Data-adaptive sparse attention masks substantially outperform fixed patterns (e.g., BigBird and Longformer) and can even exceed dense attention on long sequences. Existing adaptive approaches---including SBM-Transformer, Dynamic Mask Attention, and NSA---typically require additional learnable parame...

---

### 25. [Smart predict-then-robustly-optimize](https://arxiv.org/abs/2607.21773)

**Authors**: Aakil Caunhye, Xuefei Lu, Belen Martin-Barragan  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.21773v1  

#### Abstract
In this paper, we propose and study a robust variant of the smart predict-then-optimize approach that accounts for prediction shifts due to disturbance in the covariate feature space. While traditional integrated-learning-and-optimization models assume that side information is perfectly revealed, em...

---

### 26. [Scaling Laws for Classical Machine Learning on Tabular Data: A Benchmark Study](https://arxiv.org/abs/2607.21866)

**Authors**: Kaihua Ding  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.21866v1  

#### Abstract
Prior classical-ML learning-curve work fits power laws to tree, linear, and kernel models on tabular data, but at small scale: typically one curve, one team, a handful of cells. We present a distributed classroom-scale replication: 127 students each ran a fixed protocol on 3 assigned datasets, drawn...

---

### 27. [Class-Balanced Softmax: A Bayes Theory-Based Method for Long-Tailed Recognition](https://arxiv.org/abs/2607.22258)

**Authors**: Yi-Hang Zhu, Rajeev Raman, Shiqi Su, Jianyuan Sun, Xinyu Yang, Nan Xing, Huiyu Zhou  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.22258v1  

#### Abstract
Deep learning models using traditional softmax classifiers have achieved remarkable success in various classification tasks. However, their performance degrades significantly on imbalanced datasets. Although Balanced Softmax is widely adopted as a state-of-the-art rebalancing method, it possesses in...

---

### 28. [IQ-JEPA: A Joint-Embedding Predictive Architecture with a Hermitian Vision Transformer for Sound Speed and Attenuation Estimation from Ultrasound IQ Data](https://arxiv.org/abs/2607.22351)

**Authors**: Masashi Sode, Gianmarco Pinton  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.22351v1  

#### Abstract
The speed of sound in tissue is a prerequisite for well-focused imaging and has diagnostic value, but recovering it from raw pulse-echo channel data is fundamentally a nonlinear inverse problem. Learned solvers are fast yet label hungry. Simulated sound-speed labels are expensive, while abundant rea...

---

### 29. [Complexity Bounds and Approaches to Learning Projected Gradient Descent Solver Iterates](https://arxiv.org/abs/2607.22467)

**Authors**: Anjian Li, Ryne Beeson  
**Category**: cs.LG  
**Published**: 2026-07-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.22467v1  

#### Abstract
Data scarcity poses a fundamental challenge in training generative models to produce initial guesses for parametric optimization problems that are otherwise numerically expensive to solve. We therefore study a $k$-neighborhood data collection strategy that augments datasets of converged solutions wi...

---

### 30. [Adversarial Style Optimization: Enhancing VLM Jailbreaks by GRPO-based Stylistic Triggers Optimization](https://arxiv.org/abs/2607.21619)

**Authors**: Bingjun Luo, Jialin Guo, Yue Yao, Xinpeng Ding  
**Category**: cs.CL  
**Published**: 2026-07-27  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2607.21619v1  

#### Abstract
Multimodal Large Language Models (MLLMs) have achieved impressive performance, but their safety alignment remains vulnerable to jailbreak attacks. Existing content-based jailbreaks are often inconsistent and show unsatisfying performance against the rapidly evolving MLLMs, failing to exploit non-con...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-03 10:31:04 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [KForge: LLM-Driven Cross-Platform Kernel Generation for AI Accelerators](https://arxiv.org/abs/2606.02963)

**Authors**: Taras Sereda, Burak Bartan, Ankita Nayak, Tom St. John, Natalie Serrino, Zain Asgar  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2606.02963v1  

#### Abstract
Production inference increasingly targets a heterogeneous mix of accelerators. Agentic pipelines interleave reasoning, tool calls, and multi-agent coordination, each with distinct compute and memory profiles. For optimal efficiency, each stage should run on the accelerator best suited to it. This cr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《KForge: LLM-Driven Cross-Platform Kernel Generation for AI Accelerators》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 AI 推理流水线（agentic pipelines）涉及多种计算模式（如推理、工具调用、多智能体协作），其不同阶段具有差异显著的计算与内存特征。因此，生产级部署倾向于将各阶段分配到最适合的 AI 加速器上执行，以实现最优性能。这带来了**跨平台高性能内核（kernel）生成**的巨大挑战：

- 不同硬件后端（NVIDIA、AMD、Intel、Apple）使用不同的编程模型（CUDA、Triton、SYCL、Metal 等）。
- 手动编写高性能 kernel 耗时且依赖专家经验，难以扩展。
- 现有 LLM 自动生成 kernel 的方法在低级代码正确性和跨平台泛化能力上仍存在不足。

### 提出了什么新方法或新思路
本文提出 **KForge**，一个基于大语言模型（LLM）驱动的**跨平台自动程序合成框架**，其核心创新在于：

- **双代理协同迭代优化架构**：
  - **Generation Agent**：负责生成候选 kernel，并根据编译、运行和数值正确性反馈进行功能修正。
  - **Performance-Analysis Agent**：专门分析性能剖析数据（如 Nsight、Xcode Instruments 输出），识别瓶颈并生成具体的优化建议（如提升 memory bandwidth 利用率、warp occupancy），指导下一轮合成。
- **闭环迭代精炼流程**：
  - 分为两个阶段交替进行：
    1. **Functional Pass**：确保 kernel 编译通过、无运行错误、输出数值正确。
    2. **Optimization Pass**：在功能正确的前提下，利用性能反馈持续优化执行效率。
- **统一接口支持多平台多模型**：
  - 支持 **4 大厂商**（NVIDIA、AMD、Intel、Apple）和 **6 种编程模型**（CUDA、Triton、CuTe DSL、HIP、SYCL、Metal）。
  - 可灵活接入不同 LLM（通过 model registry），具备良好的可扩展性。

### 相比现有方法的优势
| 维度 | 现有方法（如 CUDA-LLM、KernelBlaster） | KForge |
|------|----------------------------------------|--------|
| 架构设计 | 单一 agent 或简单反馈循环 | **双 agent 分工协作**，职责分离更高效 |
| 性能反馈 | 多为标量指标（如运行时间） | 支持**原始 profiling 数据 + GUI 截图等复杂输入**，提取细粒度优化信号 |
| 平台覆盖 | 主要集中于 NVIDIA/CUDA 生态 | **真正跨平台**，支持异构硬件统一生成 |
| 评估视角 | 多为单 kernel 微基准测试 | 注重对 **end-to-end 模型推理性能的影响** |

---

## 2. 核心实验方法和设置

### 使用的数据集 / 工作负载
- **Case Study 1 (NVIDIA B200)**：
  - 模型：`gpt-oss-20b`（MoE 架构）
  - 优化目标：decode-path 中的关键 kernel，包括：
    - `Fused Add + RMSNorm`
    - `MoE finalize`
    - `Bias + RoPE + KV update`
  - 来源：从 TensorRT-LLM v1.3.0rc9 中提取的开源 kernel 实现。
- **Case Study 2 (Intel Arc B580)**：
  - 基准套件：**KernelBench Level 2** 中的 **37 个 GEMM + tail-ops** 工作负载。
  - 内容：涵盖常见融合模式，如归约（reduction）、归一化（GN）、激活函数链等。

### 实验设置和评估指标
#### 共同设置
- **LLM 后端**：Claude Opus 4.6（high-effort mode）
- **生成策略**：generate-refine loop，每轮最多 5 次迭代。
- **验证机制**：
  - 功能正确性：通过 PyTorch reference model 的 `forward()` 输出对比（shape 和数值 tolerance）。
  - 编译 & 运行：集成编译器和运行环境自动化测试。

#### Case Study 1（NVIDIA B200）
- **评估方式**：
  - **Micro-benchmarking**：独立测量每个 kernel 在不同 batch size 下的执行时间。
  - **End-to-end evaluation**：使用 `trtllm-bench` 脚本测试完整推理吞吐量。
- **工作负载配置**：
  - 请求数量：512
  - Prefill 长度：1024 tokens
  - Decode 长度：8192 tokens
  - Batch size：8
- **控制变量**：
  - 禁用 autotuner（因其选择不稳定）
  - 锁定 GPU clock 至 1500 MHz，保证可复现性

#### Case Study 2（Intel Arc B580）
- **基线方法**：取 `torch.compile` 和 `PyTorch eager mode` 中更快者作为 baseline。
- **评估指标**：几何平均加速比（geometric mean speedup）。

### 基线方法对比
| 平台 | 基线方法 |
|------|----------|
| NVIDIA B200 | **TensorRT-LLM**（厂商高度优化的推理库） |
| Intel Arc B580 | **torch.compile / PyTorch eager**（通用编译器方案，无专用 hand-tuned kernels） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### Case Study 1：NVIDIA B200 vs. TensorRT-LLM
| 指标 | Baseline | KForge | 提升幅度 |
|------|---------|--------|----------|
| **Throughput (tok/s)** | 2601.55 | 2656.61 | **+2.12%** ✅ |
| **Wall-clock time (s)** | 1612.24 | 1578.82 | **-2.07%** ✅ |

- **微基准结果（Table I）**：
  - `Fused Add + RMSNorm`：平均加速 ~1.11×
  - `MoE finalize`：小 batch 加速 ~1.06–1.10×，大 batch（128）达 **1.43×**
  - `Bias + RoPE + KV update`：除 batch=8 外全面优于 baseline，最大加速 1.05×
- 尽管绝对增益不高，但在 **TensorRT-LLM 这样成熟的生产级 runtime 上实现净收益**，表明 KForge 能发现超越传统优化路径的新实现。

#### Case Study 2：Intel Arc B580 上的 Triton Kernel 生成
| 指标 | 结果 |
|------|------|
| **Geometric Mean Speedup** | **5.13×** ✅ |
| 最高单任务加速 | 达 **6.5×**（Problem 22） |

- **代表性问题性能（Table III）**：
  | Problem | t.comp (ms) | KForge (ms) | Speedup |
  |-------|------------|-------------|---------|
  | 37_Matmul_Swish_Sum_GN | 23.50 | 5.73 | 4.1× |
  | 22_Matmul_Scale LSE Mish | 9.86 | 1.51 | 6.5× |
  | 88_Gemm_GN_Swish_Mul_Swish | 10.00 | 1.65 | 6.1× |
  | 62_Matmul_GN_LReLU_Sum | 10.00 | 1.83 | 5.5× |

- **成功原因分析**：
  - **Layer-wise fusion**：将 GEMM 后续操作融合进单个 kernel，避免中间张量写回全局内存。
  - **Mixed-precision execution**：在 XMX 单元中使用 FP16 计算，FP32 累加，兼顾速度与精度。
  - **Single-pass reductions**：采用流式在线算法（类似 FlashAttention softmax）替代传统两遍归约。
  - **Occupancy-aware tiling**：根据 group size 自适应选择分块策略（sequential loop / 2D tile / per-row streaming）。

### 消融实验结果（文中未明确提供系统性消融研究）
- 文中未报告标准的消融实验（ablation study）来量化各组件（如 performance-analysis agent、cross-platform translation 等）的独立贡献。
- 但通过两个极端案例间接体现价值：
  - 在 **已有强 baseline 的平台（NVIDIA）** 上仍能提效 → 显示其优化能力。
  - 在 **缺乏 hand-tuned kernel 的新兴平台（Intel Arc）** 上实现大幅加速 → 显示其“bring-up”能力。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 驱动的 kernel 生成已具备实用价值**：
   - 不仅能在成熟平台（NVIDIA）上超越厂商优化库（TensorRT-LLM），还能在新兴平台（Intel Arc）快速构建高性能 kernel。
2. **双 agent 架构有效解耦功能与性能优化**：
   - 分离 concerns 提升了系统的模块化与稳定性，尤其适合处理复杂的 profiling 输入。
3. **跨平台知识迁移成为可能**：
   - 利用 CUDA 等主流生态中的实现作为参考，可在 Metal/SYCL 等平台上生成高质量 kernel，缓解训练数据偏差问题。
4. **端到端性能提升需全局考量**：
   - 局部 kernel 加速不一定带来整体收益；KForge 强调在真实推理上下文中验证效果，避免“虚假优化”。

### 方法的局限性
1. **依赖源码级 reference implementation**：
   - 当前流程需要一个可用的 PyTorch 实现作为起点，无法直接从行为规范（behavioral spec）或闭源二进制逆向生成 kernel。
2. **尚未支持 JAX 等其他 DL 框架**：
   - 目前仅验证于 PyTorch 生态，框架通用性有待扩展。
3. **缺乏形式化验证机制**：
   - 正确性依赖数值测试，未引入 differential testing 或 formal equivalence checking，对极端输入鲁棒性存疑。
4. **局部优化可能破坏全局调度**：
   - 更快的 kernel 可能改变 register/shared memory 占用，干扰 runtime autotuner 对邻近 kernel 的选择。

### 未来工作方向
1. **从行为规范生成 kernel**：
   - 针对只有 cubin 或 vendor BLAS 的场景，尝试通过黑盒测试 + LLM 推断实现逻辑。
2. **目标扩展至低级虚拟 ISA（如 PTX）**：
   - 释放更多底层优化潜力（如精确控制指令调度），但牺牲可移植性。
3. **联合优化 kernel block 而非单个 kernel**：
   - 在 attention block 或 MoE MLP 层面进行整体优化，协调资源分配与数据流。
4. **集成更先进的 agentic planning 能力**：
   - 引入 long-horizon reasoning，让 agent 自主决定优化顺序、探索策略与终止条件。

---

> **总结一句话**：  
> **KForge 展示了 LLM-based agentic 方法在跨平台高性能 kernel 生成上的巨大潜力——既能“锦上添花”于成熟平台，也能“雪中送炭”于新兴硬件，是迈向自动化 AI 系统优化的重要一步。**

</details>

---

### 2. [Qift: Shift-Friendly No-Zero W2 Post-Training Quantization for Rotated W2A4/KV4 LLM Inference](https://arxiv.org/abs/2606.02823)

**Authors**: Chi-Wei Huang, Chia-Chi Tsai  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.02823v1  

#### Abstract
Two-bit weight quantization is attractive for memory-efficient LLM inference, but the standard W2 level set {-2,-1,0,+1} often collapses under aggressive W2A4/KV4 settings. We study the scalar level-set geometry of two-bit weights in a Hadamard-rotated quantization pipeline. Conventional asymmetric ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Qift: Shift-Friendly No-Zero W2 Post-Training Quantization for Rotated W2A4/KV4 LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **W2A4/KV4 极低比特量化中的性能崩溃问题**：标准的对称整数量化器（如 `SYM-INT`，level set 为 `{-2, -1, 0, +1}`）在 **W2A4/KV4**（两比特权重、四比特激活和 KV 缓存）设置下表现极差，甚至导致模型完全失效（例如 LLaMA-3.1-8B 的 PPL 超过 3000）。
- 作者指出，这不仅是 **bit-width 不足** 的问题，更是 **scalar reconstruction level set 设计不当** 所致。

### 🚀 提出的新方法与思路
- **提出 Qift**：一种无需训练、无零点、固定 level set 的 **no-zero W2 量化方案**，专为 **Hadamard 旋转后的 LLM 权重分布** 设计。
- **核心思想**：
  - 旋转后权重近似 **零中心且高斯状分布**，因此最优的 W2 量化应避免将一个重建等级设为 `0`（即 mid-tread），而应采用 **mid-rise 结构**，将两个内层质心置于 `±0.5` 附近。
  - 提出两种具体 level set：
    - **Qift-MNZ**: `{±0.5, ±1.5}`（等价于 `{±1, ±3}` 在半尺度下）
    - **Qift-PoT-MNZ**: `{±1, ±4}`，支持 **sign-and-shift 解码**，硬件友好。
- **设计原则**：保持简单、模块化、硬件对齐：
  - 无 QAT（Quantization-Aware Training）
  - 无 learned codebook
  - 无 group-wise grid
  - 无 asymmetric zero-point
  - 仅改变 code-to-level 映射

### 🔍 相比现有方法的优势
| 特性 | Qift | RCP / QuIP# / AQLM 等 |
|------|------|------------------------|
| 是否需要 QAT | ❌ 否 | ✅ 是（RCP）或复杂训练 |
| 是否有 learned codebook | ❌ 无 | ✅ 有（向量/非均匀码本）|
| 是否 group-free | ✅ 是 | ❌ 否 |
| 是否 zero-point-free | ✅ 是 | ❌ 多数需要 |
| 是否硬件友好 | ✅ 是（整数/幂次级别） | ❌ 通常需查表 |
| 部署复杂度 | 极低（drop-in 替换） | 高 |

> ✅ Qift 是首个在 **不增加任何元数据或训练开销** 的前提下，显著提升 W2A4 性能的方法。

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **模型**：
  - `LLaMA-2-7B`
  - `LLaMA-3.1-8B`
- **校准数据集**：
  - `WikiText-2`，128 个样本，序列长度 2048
- **评估任务**：
  - **Perplexity (PPL)**：WikiText-2
  - **下游任务准确率**：ARC-C, ARC-E, HellaSwag, PIQA, WinoGrande（平均 Accuracy）

### ⚙️ 实验设置
- **量化配置**：
  - **W2A4/KV4**：所有实验均启用 Hadamard 旋转、weight、activation 和 KV-cache 量化
  - **Per-channel scaling**：基于 clip-based MSE 搜索 scale
  - **Activation & KV**：A4 对称量化，KV-cache 使用不对称量化
- **补偿机制**：
  - 支持 `RTN`, `GPTQ`, `GPTAQ`
- **混合精度设置（Mixed Precision）**：
  - `L=16`：将最敏感的 16 层升级为 W4A4，其余保持 W2A4，实现平均 `b_avg = 3`，与 W3A4 公平比较

### 🆚 基线方法对比
| 方法 | 类型 | 是否 QAT | 是否有 codebook | 是否 zero-point |
|------|------|----------|----------------|----------------|
| `SYM-INT` (`{-2,-1,0,+1}`) | 标准对称整数 | ❌ | ❌ | ❌ |
| `W-ASYM` | 常规非对称量化 | ❌ | ❌ | ✅ |
| `Lloyd-Max` | 高斯最优标量量化 | ❌ | ✅（固定表） | ❌ |
| `NF2` | NormalFloat 启发 | ❌ | ✅（固定表） | ❌ |
| `FAR-MNZ` (`{±1,±2}`) | 无零但内层太远 | ❌ | ❌ | ❌ |
| `RCP` | 学习非均匀量化 | ✅ | ✅ | ✅ |
| `Qift-MNZ` / `Qift-PoT-MNZ` | 本文方法 | ❌ | ❌ | ❌ |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 4 & 5 & 6）

#### ✅ **纯 W2A4 表现（GPTAQ）**

| 模型 | 方法 | PPL | 下游平均 Accuracy |
|------|------|-----|------------------|
| LLaMA-2-7B | `SYM-INT` | 12.118 | 0.4211 |
| | `Qift-MNZ` | **9.294** | **0.4794** |
| | `Qift-PoT-MNZ` | 9.577 | 0.4766 |
| | `Lloyd-Max` | 9.265 | 0.4793 |

| LLaMA-3.1-8B | `SYM-INT` | 29.695 | 0.3683 |
| | `Qift-MNZ` | **19.515** | **0.4064** |
| | `Qift-PoT-MNZ` | 20.150 | **0.4140** |

> 💡 **Qift 将 LLaMA-2-7B 的 PPL 降低 23%，LLaMA-3.1-8B 降低 34%**，并显著恢复下游任务能力。

#### ✅ **混合精度 L=16 vs W3A4（iso-bit 比较）**

| 模型 | 方法 | PPL | 下游平均 Accuracy |
|------|------|-----|------------------|
| LLaMA-2-7B | `W3A4 GPTQ` | 6.897 | 0.6200 |
| | `L=16 SYM-INT` | 7.825 | 0.5922 |
| | `L=16 Qift-MNZ` | **7.318** | **0.6157** |

| LLaMA-3.1-8B | `W3A4 GPTQ` | 10.954 | 0.5972 |
| | `L=16 SYM-INT` | 12.744 | 0.5431 |
| | `L=16 Qift-PoT-MNZ` | **11.499** | **0.5619** |

> ✅ **Qift 在 L=16 设置下缩小了约 50% 的 W3A4 性能差距**，同时保留一半参数为 W2，极具部署价值。

### 🔬 消融实验结果

#### （1）**RTN 重建误差诊断（Table 13）**
- `SYM-INT` 总重建误差：`1.2890×10⁵`
- `Qift-MNZ`：`1.0228×10⁵`（↓20.7%）
- `Qift-PoT-MNZ`：`1.0363×10⁵`（↓19.6%）
- 且 `MNZ` 的误差分配更均衡，`SYM-INT` 的 `+1` 桶承担了超过 55% 的误差。

#### （2）**Scale-Invariant Ratio 敏感性分析（Figure 8）**
- 定义 `r = inner / outer` 质心比
- 最优区间：`r ∈ [0.25, 0.33]`
- Qift-MNZ (`r=1/3≈0.333`) 和 Qift-PoT-MNZ (`r=0.25`) 正好落在该区间
- `FAR-MNZ` (`r=0.5`) 表现差，说明“去零”本身不够，**几何比例必须合理**

#### （3）**GPTQ 残差分析（Table 15）**
- `Qift-MNZ` 的累积残差仅为 `SYM-INT` 的 **77.8%**
- 残差越低，PPL 趋势越好，验证了 Qift 的重建优势能被 GPTQ 利用

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **W2A4 失败主因是 level set 设计不当**，而非单纯 bit-width 不足。
2. **Hadamard 旋转使权重标准化后更接近高斯分布**，且保持近似零中心（Figure 3, Table 3）。
3. **最优 W2 应采用 no-zero mid-rise 结构**，两个内层质心应在 `±0.5` 附近，而非 `0`。
4. **Qift 的 level set 在多个维度一致优于 baseline**：
   - 更低 PPL
   - 更高下游 Accuracy
   - 更小重建误差
   - 更少 GPTQ 残差
5. **硬件友好变体 Qift-PoT-MNZ (`{±1,±4}`) 表现依然强劲**，适合 sign-and-shift 架构。

### ⚠️ 方法的局限性
- **依赖 Hadamard 旋转**：Qift 假设输入是旋转后的权重，不适用于原始权重直接量化。
- **未探索动态或分组 level set**：虽然强调 simplicity，但在极端场景下可能不如 learned non-uniform quantizer（如 RCP）极限性能高。
- **仅针对 W2**：未扩展到其他极低比特（如 W1）。

### 🔮 未来工作方向
- 探索 **自适应 inner/outer ratio** 的轻量级方法
- 将 Qift 思想推广至 **W2A2 或 W1A4**
- 与 **learned rotation**（如 SpinQuant）结合
- 在真实边缘设备上验证 **decode-time throughput 提升**

---

## 总结

> ✅ **Qift 揭示了一个被忽视的关键事实：在极低比特量化中，level set 的几何设计比是否“去零”更重要。**
>
> 它以极简的设计（无训练、无元数据、固定 level）实现了对标准 W2 的全面超越，是 **post-training quantization 中“source-aware design”的典范**，为高效 LLM 推理提供了简单、实用、可部署的新路径。

</details>

---

### 3. [GreenGNN: Energy-Aware Windowed Communication Optimization for Distributed GNN Training](https://arxiv.org/abs/2606.02916)

**Authors**: Arefin Niam, Tevfik Kosar, M. S. Q. Zulkar Nine  
**Category**: cs.DC  
**Published**: 2026-06-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.02916v1  

#### Abstract
Large-scale graph neural network (GNN) training often requires distributed clusters because graph structure and feature tensors no longer fit in a single node's memory. In sampling-based training, each mini-batch expands into a receptive field that spans partitions and triggers thousands of remote f...

---

### 4. [SIGMA: A Versatile Streaming Graph Partitioner for Vertex- and Edge-Balanced Distributed GNN Training](https://arxiv.org/abs/2606.03519)

**Authors**: Barbara Hoffmann, Shai Dorian Peretz, Adil Chhabra, Ahmet Kadir Yalcinkaya, Ruben Mayer, Christian Schulz  
**Category**: cs.DC  
**Published**: 2026-06-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.03519v1  

#### Abstract
Distributed Graph Neural Network (GNN) training depends critically on how the underlying graph is partitioned across compute resources. Existing graph partitioners focus either on vertex partitioning or edge partitioning and typically optimize only a single communication objective (edge cut or verte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SIGMA: A Versatile Streaming Graph Partitioner for Vertex- and Edge-Balanced Distributed GNN Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的图划分器（graph partitioner）通常只专注于**vertex partitioning** 或 **edge partitioning**，并且仅优化单一通信目标（如 edge cut 或 replication factor），同时施加单一平衡约束（vertex balance 或 edge balance）。然而，在分布式 **GNN training** 中，计算既依赖于顶点中心（vertex-centric）操作（如聚合），也依赖于边中心（edge-centric）操作（如消息传递），因此需要同时考虑**顶点和边的负载均衡**以及**通信开销最小化**。

此外，传统方法将 vertex 和 edge 划分视为两个独立问题，导致研究重复、工程复杂度高。

### 提出的新方法：SIGMA
本文提出了 **SIGMA**（Streaming Integrated Graph Partitioning with Multi-objective Awareness），一个统一的、支持多目标、多约束的**流式图划分框架**，其核心创新如下：

- ✅ **统一框架支持两种划分范式**：
  - 支持 **vertex-streaming**（分配顶点以最小化 edge cut）
  - 支持 **edge-streaming**（分配边以最小化 vertex replication）
  - 可在同一个算法框架下灵活切换，无需为不同系统重新设计划分器。

- ✅ **多目标、多约束联合优化**：
  - 同时优化 **edge cut / replication factor**
  - 同时保证 **vertex balance** 和 **edge balance**
  - 在动态容量缩放机制下处理多个硬性/软性约束，避免早期过载导致后期无法平衡。

- ✅ **基于聚类的预处理阶段**（Clustering-Based Preprocessing）：
  - 使用 **CluStRE** 进行流式图聚类，捕捉全局社区结构。
  - 将聚类结果通过 **makespan scheduling** 映射到分区块，实现“先粗后精”的划分策略。
  - 预先分配结构一致且满足平衡条件的顶点或边，提升划分质量而不牺牲流式效率。

- ✅ **开源实现**：
  - 已公开代码：[https://github.com/bab-si/SIGMA](https://github.com/bab-si/SIGMA)

### 相比现有方法的优势
| 维度 | SIGMA | 现有方法（如 Fennel, HDRF, METIS 等） |
|------|-------|-------------------------------|
| **目标耦合性** | 联合优化 edge cut + replication + vertex/edge balance | 单一目标 + 单一约束 |
| **适用性** | 支持 vertex 和 edge 划分 | 通常只能支持一种 |
| **可扩展性** | 流式处理，内存友好，适合大图 | In-memory 方法（如 METIS）难以扩展 |
| **实际效果** | 更好地平衡训练负载、降低通信与内存消耗 | 常因忽略某一维度而导致瓶颈 |

---

## 2. 核心实验方法和设置

### 数据集
使用了六个广泛使用的基准图数据集，覆盖多种领域和规模：

| 图数据集 | 类型 | #Vertices | #Edges |
|--------|------|----------|--------|
| amazon computers | 电商 | 13.7k | 491.7k |
| flickr | 社交网络 | 89.2k | 899.7k |
| twitch | 社交网络 | 168.1k | 6.7M |
| ogbn-arxiv | 引用网络 | 169.3k | 1.2M |
| reddit | 社交网络 | 233.0k | 114.6M |
| ogbn-products | 商品共购图 | 2.4M | 61.9M |

这些数据集具有不同的密度、规模和结构特性，确保评估的全面性和鲁棒性。

### 实验设置
- **硬件环境**：两台服务器，每台配备 AMD EPYC 9454P（48核）、768GB DDR5 内存、双 NVIDIA L40S GPU（各 48GB）、SSD 存储。
- **分区数量 k**：测试了从 2 到 32 不等的分区数。
- **GNN 模型**：Two-layer GraphSAGE，GCN aggregator，hidden dim=16，ReLU + Dropout(0.5)，Adam(lr=0.003)。
- **训练配置**：
  - **DistDGL**（vertex-partitioned）：mini-batch training，batch size=1024，fanout=[25,25]
  - **DistGNN**（edge-partitioned）：full-batch training

### 评估指标
| 类别 | 指标 |
|------|------|
| **划分质量** | Edge-cut ratio（vertex partitioning）、Replication Factor（edge partitioning） |
| **负载均衡** | Vertex Balance、Edge Balance（越接近 1 越好） |
| **系统效率** | Partitioning Time、Average Training Time per Epoch |
| **资源消耗** | Peak GPU Memory（DistDGL）、Peak RAM（DistGNN） |

### 基线方法对比
#### 对于 **Edge Partitioning**：
- 流式方法：HDRF、Random、DBH、HeiStreamE、2PS
- In-memory 方法：FSM、HEP（t=100 接近全内存）

#### 对于 **Vertex Partitioning**：
- 流式方法：FENNEL、LDG、Cuttana、HeiStream、BuffCut、Random
- In-memory 方法：METIS、KaHIP

所有基线均使用默认或推荐参数运行。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）**Edge Partitioning 结果**

| 指标 | SIGMA 表现 |
|------|-----------|
| **Replication Factor** | 在所有流式方法中表现最佳。例如在 `reddit` 上（k=32），相比 HDRF 降低约 **82%**；相比 Random 降低约 **87%**。接近 FSM 和 HEP 的水平（差距 < 0.6）。 |
| **Vertex Balance** | 显著优于 FSM 和 HEP（最大 imbalance 从 2.09 降至 1.05），相对理想值 1.0 的偏差减少 **~51%**。 |
| **Edge Balance** | 控制在 1.10 以内（允许 10% imbalance），满足设定约束。 |
| **Partitioning Time** | 略高于 HDRF/DBH，但显著低于 FSM/HEP（尤其在大图上），保持流式优势。 |

> 💡 **结论**：SIGMA 在 edge partitioning 下实现了**最低的复制因子和最优的顶点平衡**，显著优于其他流式方法，并逼近 in-memory 方法的质量。

#### （2）**Vertex Partitioning 结果**

| 指标 | SIGMA 表现 |
|------|-----------|
| **Edge-cut Ratio** | 属中上水平，优于 Random、FENNEL、LDG。虽不及 METIS/KaHIP（后者能访问完整图进行全局优化），但在流式方法中竞争力强。例如在 `flickr`（k=32）上达到 0.642，优于 FENNEL 的 0.663。 |
| **Vertex Balance** | 极佳！范围为 **1.00–1.09**，远优于 FENNEL（最高达 2.40），改善约 **94%**。 |
| **Edge Balance** | 同样优秀，范围为 **1.01–1.18**，多数情况下优于或媲美其他方法。 |
| **Partitioning Time** | 与 FENNEL/Cuttana 相当，略慢于 HeiStream/BuffCut，但远快于 KaHIP。 |

> 💡 **结论**：尽管 edge-cut 不是最优，但 SIGMA 通过**卓越的负载均衡能力**弥补了这一点，整体划分质量更稳定可靠。

#### （3）**端到端训练性能**

##### ✅ **DistGNN（edge-partitioned）**
- SIGMA 在所有数据集上取得**最低或接近最低的每轮训练时间**。
- 相比 HDRF 平均提速 **~62%**，相比 HEP 提速 **~25%**。
- 内存占用（RAM）也最低之一，例如在 `flickr` 和 `twitch` 上分别仅为 **22.6GB** 和 **42.7GB**。

##### ✅ **DistDGL（vertex-partitioned）**
- 训练时间与 KaHIP/METIS 接近，优于大多数流式方法（如 FENNEL、LDG）。
- 在 `products` 上略有落后（因该图本身易平衡，edge-cut 成主导因素），但总体仍具竞争力。
- GPU memory 使用稳定，host RAM 消耗较低。

#### （4）消融实验（隐含分析）
虽然未显式列出 ablation study，但从设计可看出以下关键组件的作用：
- **Clustering-based preprocessing**：显著提升了划分质量，尤其是在社区结构明显的图（如 social networks）上。
- **Multi-objective scoring function**：引入 replication-aware penalty 有效减少了 ghost vertex 数量。
- **Dynamic capacity scaling**：防止早期某维度过载，保障最终 balance。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **统一框架可行且高效**：SIGMA 成功在一个流式框架内支持 vertex 和 edge 划分，打破了二者长期割裂的局面。
2. ✅ **多目标联合优化带来综合收益**：即使在某些单项指标（如 edge-cut）上不如专用方法，但通过**更好的负载均衡**和**更低的复制率**，最终实现更优的训练效率和资源利用率。
3. ✅ **流式方法可以媲美 in-memory 方法**：借助聚类预处理和智能评分函数，SIGMA 在划分质量和训练性能上接近甚至超越部分 in-memory 方法（如 HEP、FSM），同时保留了流式的低内存、高可扩展优势。
4. ✅ **划分质量直接影响训练性能**：replication factor ↓ → communication ↓ → training time ↓；balance ↑ → straggler ↓ → hardware utilization ↑。

### 方法的局限性
- ❗ **对高度均匀图增益有限**：如 `ogbn-products` 在 k=2 时多数方法都能很好平衡，此时 edge-cut 成主导因素，SIGMA 的多目标优势未能充分发挥。
- ❗ **预处理增加一定开销**：clustering 和 cluster-to-block mapping 增加了常数级时间成本，虽不影响渐近复杂度，但在极小图上可能不划算。
- ❗ **参数敏感性**：如 γ（imbalance penalty）、τ（replication penalty）需调优，在极端不平衡场景下可能需自适应调整。

### 未来工作方向
- 🔮 支持 **动态图划分**（dynamic/streaming graphs）
- 🔮 引入 **学习型 scoring 函数**（learned assignment policy）
- 🔮 扩展至 **异构设备环境**（GPU/CPU混合部署）
- 🔮 探索 **自动参数调优机制**（auto-tuning for γ, τ 等）

---

## 总结
SIGMA 是首个将 **vertex partitioning** 与 **edge partitioning** 统一于同一**流式多目标框架**下的图划分器。它不仅解决了分布式 GNN 训练中通信、计算、内存之间的复杂权衡问题，还通过**聚类预处理**提升了流式方法的传统短板——全局结构感知能力。实验证明，SIGMA 在多种真实图数据和主流 GNN 系统（DistDGL/DistGNN）上均能实现**优异的划分质量、良好的负载均衡、低通信开销和高效的端到端训练性能**，是迈向通用、高效、可扩展图划分的重要一步。

</details>

---

### 5. [DECA: Decentralizing Block-Wise Adam for Efficient LLM Full-Parameter Fine-Tuning on Non-IID Data](https://arxiv.org/abs/2606.03209)

**Authors**: Yunsheng Yuan, Shaowei Li, Kai Wang, Zhongyuan Sun, Zheng Zhang, Kai Han, Jun Luo, Feng Li  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.03209v1  

#### Abstract
Fine-tuning large language models (LLMs) in privacy-sensitive and resource-constrained environments remains challenging. Since training data are often distributed across multiple clients, decentralized fine-tuning offers a natural paradigm for collaborative adaptation without a central server. Howev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DECA: Decentralizing Block-Wise Adam for Efficient LLM Full-Parameter Fine-Tuning on Non-IID Data

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在隐私敏感和资源受限的环境中，对大语言模型（LLMs）进行微调面临以下挑战：
- **通信开销高**：传统去中心化方法需要交换完整的模型参数，对于数十亿参数的LLM而言，通信成本极高。
- **内存限制**：Full-Parameter Fine-Tuning（FPFT）要求存储所有参数的梯度和优化器状态，超出单个客户端的GPU内存容量。
- **非独立同分布（non-IID）数据导致的客户端漂移**：各客户端数据分布差异大，局部更新容易偏离全局目标，导致训练不稳定甚至发散。

现有方法多依赖于Parameter-Efficient Fine-Tuning（PEFT），如LoRA或Adapter，虽然提升了效率，但牺牲了下游任务性能。

---

### 提出的新方法：DECA
DECA（**Decentralized Block-wise Adam**）是一种面向non-IID数据的轻量级去中心化FPFT框架，其核心创新包括：

#### （1）**分块式Adam优化（Block-Wise Adam Optimization）**
- 将模型参数划分为 $ B $ 个不相交的块（block），每次只激活一个块进行协同优化。
- 在每轮全局迭代中，按顺序依次优化每个块，显著降低单次通信和计算负载。

#### （2）**块级动量近似（Block-wise Moment Approximation, BMA）**
- 引入第一阶和第二阶的块级动量估计，结合：
  - **新鲜的本地梯度统计信息**（local gradient statistics）
  - **共识衍生的差异信号**（consensus-derived discrepancy signal）
- 动态修正本地更新方向，使其更接近网络范围内的共识，缓解客户端漂移。

#### （3）去中心化架构设计
- 完全去除中央服务器，仅通过邻居节点间的P2P通信完成协作训练。
- 支持异构设备和受限带宽环境下的部署。

---

### 相比现有方法的优势
| 维度 | DECA | Dec-LoRA / Dec-Adapter | DeCAF |
|------|------|------------------------|-------|
| 微调方式 | **FPFT**（完整参数微调） | PEFT（低秩适配） | PEFT（TSVD增强LoRA） |
| 性能潜力 | ✅ 高（保留全部表达能力） | ❌ 有限（受限于低秩假设） | ❌ 有限 |
| 资源效率 | ✅ 高（分块处理） | ✅ 高 | ⚠️ 极高计算开销（TSVD分解） |
| 稳定性 | ✅ 强（BMA抑制漂移） | ⚠️ 中等 | ⚠️ 波动较大 |
| 适用场景 | 资源受限 + 高性能需求 | 仅高效优先 | 效率较低 |

> ✅ **DECA首次实现了在去中心化环境下高效且稳定的FPFT**，填补了该领域的空白。

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 分类任务：
- **NWGI**（News Writer Genre Identification）：新闻写作风格分类
- **AGNEWS**：四类新闻主题分类（World, Sports, Business, Sci/Tech）
- **TFNS**（Twitter Financial News Sentiment）：金融推文情感分析（Bullish/Bearish）
- **MNLI**（Multi-Genre Natural Language Inference）：自然语言推理任务

#### 生成任务：
- **Alpaca**：包含52,000条指令-响应对，用于评估模型遵循复杂指令的能力

---

### 实验设置
- **模型规模**：涵盖从1.5B到8B参数的主流LLM：
  - Qwen2-1.5B, Qwen2.5-3B-Instruct
  - Llama-2-7B, Llama-3.1-8B-Instruct
- **网络拓扑**：采用Erdős-Rényi（ER）图构建8个客户端的去中心化网络
- **数据划分**：使用Dirichlet分布（$ \alpha=0.25 $）模拟non-IID数据分布
- **训练配置**：
  - 总通信轮数 $ T=4 $
  - 每块本地更新步数 $ R=48 $
  - 学习率 $ \gamma = 5\times10^{-5} $
  - Adam超参：$ \alpha_1=0.9, \alpha_2=0.999 $

---

### 评估指标
| 任务类型 | 指标 |
|---------|------|
| 分类任务 | Accuracy（Acc.）、F1 Score |
| 生成任务 | Vicuna（VIC.）、MT-Bench（MT.）评分（基于GPT-4自动评判） |

---

### 基线方法对比
选取三种最先进的去中心化LLM微调方法作为基线：
1. **Dec-Adapter**：基于Adapter的PEFT去中心化实现
2. **Dec-LoRA**：去中心化的LoRA微调方法
3. **DeCAF**：结合TSVD分解的去中心化LoRA方案，强调共识聚合

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1 & Table 2）

#### 分类任务表现（平均Accuracy / F1）
| 方法 | Qwen2.5-3B | Llama-3.1-8B |
|------|------------|--------------|
| Dec-Adapter | 76.37 / 70.76 | 23.28 / 13.07 |
| Dec-LoRA | 75.52 / 68.92 | 80.42 / 75.55 |
| DeCAF | 62.15 / 54.12 | 64.59 / 54.56 |
| **DECA** | **79.32 / 74.12** | **82.30 / 77.18** |

> ✅ DECA在两个模型上均取得最优性能，尤其在Llama-3.1-8B上大幅领先Dec-LoRA约2个百分点。

#### 生成任务表现（MT-Bench综合得分）
| 方法 | Qwen2-1.5B | Llama-2-7B |
|------|-----------|------------|
| Dec-Adapter | 3.41 | 3.27 |
| Dec-LoRA | 3.69 | 3.59 |
| DeCAF | 4.06 | 3.65 |
| **DECA** | **4.10** | **3.72** |

> ✅ DECA在两项生成任务中均达到最佳或接近最佳水平。

---

### 与基线方法的对比结果
- **相比Dec-LoRA/Dec-Adapter**：
  - DECA在分类任务中平均提升 **2–6%** 的准确率；
  - 在生成任务中MT.评分提高 **0.3–0.4分**，表明更强的泛化能力和指令遵循能力。
- **相比DeCAF**：
  - 尽管DeCAF在个别指标略优，但其计算开销高达 **DECA的230倍以上**（见Appendix D.4），不具备实用性。
  - DECA以极低代价实现了可比甚至更优的整体性能。

---

### 消融实验结果（Ablation Study）

#### 设置
比较三种变体：
- **w/o BMA**：移除BMA模块
- **w/ trivial BMA**：仅用共识信号替代本地梯度（类似QG-Momentum）
- **DECA（完整版）**：融合本地梯度与共识信号

#### 结果（Qwen2.5-3B分类任务）
| 方法 | 平均Accuracy | 平均F1 |
|------|-------------|--------|
| w/o BMA | 73.09 | 66.90 |
| w/ trivial BMA | 65.19 | 61.69 |
| **DECA** | **79.32** | **74.12** |

> 🔍 发现：
- 移除BMA后性能下降明显（-6.2% Acc），说明BMA对稳定训练至关重要。
- “trivial BMA”反而性能最差，说明**过度依赖共识会破坏本地优化路径**。
- DECA成功平衡了“本地适应”与“全局一致性”。

---

## 4. 关键结论和发现

### 主要发现
1. **DECA是首个支持去中心化FPFT的有效框架**，打破了PEFT在去中心化场景中的主导地位。
2. **分块Adam + BMA机制有效缓解了non-IID数据带来的客户端漂移问题**，理论证明其收敛速率为 $ O(1/\sqrt{TBR}) $，匹配最优去中心化算法。
3. **实验验证DECA在多种模型（1.5B–8B）、任务（分类/生成）、拓扑结构下均优于现有基线**，兼具高性能与高稳定性。
4. **资源消耗显著低于传统FPFT，同时优于PEFT方法的端到端延迟**：
   - 正向传播时间减少 **45–56%**
   - 后向传播减少 **20–25%**
   - 端到端延迟降低最多达 **90%**（vs DeCAF）

---

### 方法的局限性
1. **分块粒度影响性能**：越细的分块（granularity=1）效果越好，但可能增加调度复杂度。
2. **仍需一定通信带宽**：虽低于全参数同步，但仍高于LoRA类方法（传输的是块级完整参数）。
3. **理论假设较强**：收敛性分析依赖Lipschitz平滑性和有界方差等常见假设，在极端non-IID下可能退化。

---

### 未来工作方向
1. **动态分块策略**：根据层重要性自适应调整块大小，进一步优化效率。
2. **异步版本DECA**：支持不同客户端异步更新，提升系统鲁棒性和吞吐量。
3. **扩展至多模态模型**：将DECA应用于视觉-语言模型的去中心化微调。
4. **隐私保护集成**：结合DP（Differential Privacy）或安全聚合（Secure Aggregation）实现端到端隐私保障。

---

> 📌 **总结一句话**：  
> **DECA为去中心化环境下的LLM全参数微调提供了首个高效、稳定且高性能的解决方案，在保持资源效率的同时显著超越了现有的PEFT方法，推动了边缘智能与联邦学习的发展边界。**

</details>

---

### 6. [TreeFlash: Parallel AR-Approximation for Faster Speculative Decoding](https://arxiv.org/abs/2606.03819)

**Authors**: Peer Rheinboldt, Fr\'ed\'eric Berdoz, Roger Wattenhofer  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.03819v1  

#### Abstract
One-shot block drafters for speculative decoding generate the full draft in a single forward pass, achieving strong throughput by eliminating sequential token generation. However, they predict each draft token conditioned only on the prefix context, with no dependence on previously drafted tokens. T...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TreeFlash: Parallel AR-Approximation for Faster Speculative Decoding**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **one-shot block drafters**（如 DFlash）虽然通过单次前向传播生成整个 draft 块，实现了高吞吐量，但由于其预测每个 draft token 时仅依赖于前缀上下文 $x_{<t}$，而**不依赖已生成的 draft tokens**，导致其分布为**非自回归（non-autoregressive）**。这种设计在树状 drafting（tree-based drafting）中尤为成问题：
- 随着 draft 深度增加，drafter 分布与 verifier 的真实 autoregressive 分布之间的差异（TVD）显著增大；
- 在共享前缀的不同分支中，后续 token 被迫共享相同的边际分布，损害了树的质量。

### **提出的新方法**
本文提出了 **TreeFlash**，一种结合轻量级 **AR-approximation** 机制的单次前向传播 drafter，旨在：
- 保留 one-shot drafting 的 $O(1)$ 解码时间复杂度；
- 引入对先前 draft token 的依赖，以逼近 autoregressive 分布。

#### **核心创新点**：
1. **引入 AR-approximator 层**：
   - 使用一个轻量级的 **SwiGLU 层**，将当前隐藏状态 $h_{t+i}$ 与前一个 token 的嵌入 $e_{t+i-1}$ 结合，生成修正后的分布 $q'(x_{t+i}|x_{<t}, x_{t+i-1})$。
   - 公式：  
     $$
     h_{t+i}^+ = h_{t+i} + \text{SwiGLU}(u(h_{t+i}), e_{t+i-1})
     $$

2. **两阶段树构建机制（Two-stage approximation）**：
   - 第一阶段：使用原始 DFlash 分布构建一个宽但浅的 $M$-ary 树（控制分支因子 $M$）；
   - 第二阶段：在该树节点上并行应用 AR-approximator，计算修正后的 token 分布；
   - 最后使用 **OPT-Tree** 算法选择最优的 $B$ 个候选节点构成最终 draft tree。
   - 保证总计算仍为 $O(1)$，维持高效性。

### **相比现有方法的优势**
| 方法 | 类型 | 是否 AR 条件 | 时间复杂度 | 树质量 |
|------|------|-------------|------------|--------|
| EAGLE-3 | 自回归 drafter | ✅ | $O(y)$ | 中等 |
| DFlash | One-shot 平行 drafter | ❌ | $O(1)$ | 差（边际分布） |
| DDTree | DFlash + OPT-Tree | ❌ | $O(1)$ | 较好（树结构优化） |
| **TreeFlash** | **One-shot + AR-approx** | ✅（近似） | $O(1)$ | **最优** |

- **优势总结**：
  - 显著优于纯边际分布方法（DFlash、DDTree）；
  - 在保持 $O(1)$ 复杂度的同时，逼近 autoregressive 分布；
  - 尤其在大 draft budget $B$ 和高接受率任务中表现更优。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
涵盖多种任务类型，共 64 个样本/数据集用于评估：
- **数学推理**：`MATH-500`, `GSM8K`
- **代码生成**：`HumanEval`, `MBPP`
- **通用指令遵循**：`MT-Bench`

### **实验设置**
- **目标模型（Target Models）**：
  - Qwen3 4B (+125M)
  - Qwen3 8B (+251M)
  - Qwen3 Coder 30B A3B（MoE 架构，+63M）
- **drafter 模型**：基于 DFlash 初始化，TreeFlash 进行微调
- **训练数据**：来自 Nemotron Post-Training Dataset V2 和 CodeAlpaca 的合成数据子集（100k 样本）
- **训练参数**：
  - Batch size: 128
  - 学习率：cosine decay，峰值 $10^{-4}$
  - 优化器：AdamW
  - 损失函数：forward KL divergence（vs. verifier 分布）
  - 序列长度限制：3072 tokens

### **评估指标**
| 指标 | 定义 | 说明 |
|------|------|------|
| **Speedup** | 相比 vanilla autoregressive decoding 的吞吐提升 | 受实现影响较大 |
| **Block Efficiency (T)** | 每轮 draft-verify 迭代平均接受的 token 数 | 主要评价指标，反映 drafter 质量 |
| **TVD (Total Variation Distance)** | drafter 与 verifier 分布的距离 | 衡量分布校准程度 |
| **Top-K Coverage** | drafter top-K token 在 verifier 分布下的累计概率 | 衡量候选 token 质量 |

### **基线方法对比**
- **EAGLE-3**：小型自回归 drafter，串行生成
- **DFlash**：单次扩散式前向传播生成序列 draft
- **DDTree**：在 DFlash 输出上应用 OPT-Tree 构建候选树
- 所有方法共享相同 backbone 和 checkpoint（除 EAGLE-3 外）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1 & 2）**

#### **在 Qwen3 4B / 8B 上的结果（B=16）**
| 方法 | 平均 Block Efficiency (T) | 平均 Speedup |
|------|--------------------------|-------------|
| DFlash(16) | ~4.5 | ~6.1 |
| DDTree(16) | ~4.9 | ~6.7 |
| **TreeFlash(16)** | **~5.0** | **~7.2** |

- TreeFlash 相比 DFlash：
  - **T ↑ +1.35 (+24.8%)**
  - **Speedup ↑ +0.69× (+17.1%)**
- 相比 DDTree：
  - **T ↑ +0.50 (+7.5%)**
  - **Speedup ↑ +0.19× (+3.9%)**

#### **在更大预算下（B=64）增益更明显**
| 方法 | 平均 Block Efficiency (T) | 平均 Speedup |
|------|--------------------------|-------------|
| DDTree(64) | ~5.7 | ~7.7 |
| **TreeFlash(64)** | **~6.2** | **~8.7** |

- TreeFlash 相比 DDTree：
  - **T ↑ +0.95 (+12.6%)**
  - **Speedup ↑ +0.50× (+9.0%)**
- 在 `MATH-500` 上甚至达到 **+13.3% T 提升**

#### **在大型 MoE 模型上的迁移能力（Qwen3 Coder 30B A3B）**
| 方法 | HumanEval T | MBPP T |
|------|-------------|--------|
| DDTree(64) | 9.66 | 8.84 |
| **TreeFlash(64)** | **10.57** | **10.22** |

- **AR-approximation 单独贡献 +1.01 T 提升**
- 验证了方法在大规模 MoE 模型上的有效性

### **消融实验结果（Ablation Study, Table 3）**
在 $B=64$ 设置下进行消融，结果表明：

| 变体 | Block Efficiency (T) | 说明 |
|------|---------------------|------|
| w/o AR-approximation | 7.6 | 无 AR 层，性能未提升 |
| w/ Linear layer | 6.8 | 线性层效果差，验证需非线性（SwiGLU） |
| **w/ Frozen backbone** | **7.4** | 固定 DFlash 主干，仅训练 AR-head → 性能接近完整 TreeFlash |
| w/ 2-prev tokens | 7.4 | 输入前两个 token 无额外收益 |
| w/ Cross-Entropy loss | 7.4 | KL 效果略优，但差距小 |
| w/o Loss Scaling | 7.4 | 影响较小，早期 token 加权非关键 |

> ✅ **结论**：性能增益主要来自 **AR-approximator 本身**，而非主干微调或其他训练技巧。

---

## **4. 关键结论和发现**

### **主要发现**
1. **AR-conditioning 是 one-shot drafting 的关键瓶颈**：
   - 单纯依赖边际分布会导致 TVD 随深度急剧上升（DFlash 在 depth=15 时 TVD=0.81）；
   - TreeFlash 通过 AR-approximation 将 TVD 控制在 0.62，且在 depth > 9 后优于“真实边际分布”。

2. **AR-approximation 显著提升 top-K 覆盖率**：
   - 在 depth=15，TreeFlash 的 top-1 覆盖 ≈ DFlash 的 top-5 覆盖；
   - 意味着更少的候选即可获得相同接受概率，提高资源利用率。

3. **TreeFlash 在所有配置下均 SOTA**：
   - 跨模型（4B → 30B）、跨任务（math/code/general）、跨解码模式（sampling/greedy）一致领先；
   - 增益随 $B$ 增大而增强，适合高吞吐场景。

4. **两阶段构造有效平衡效率与性能**：
   - 利用 $M$-ary 树控制 AR-approximator 的并行评估数量（$M \cdot y$），避免串行开销；
   - 实验建议 $M \leq 32$ 以最大化 throughput（图6）。

### **方法的局限性**
1. **依赖 DFlash 初始化**：
   - 未从零预训练，可能限制潜力；
   - 长期微调可能破坏初始候选分布。

2. **存在 exposure bias**：
   - 训练时 AR-approximator 使用 ground-truth 前序 token；
   - 推理时使用自身生成的 token，可能导致误差累积。

3. **未在生产级系统中验证**：
   - 所有实验基于单 batch + SDPA attention；
   - 实际服务中的量化、paged attention、多 batch 等优化可能影响大 tree 的可行性。

4. **仅在 Qwen 系列模型上验证**：
   - 泛化到其他架构（如 Llama、Phi）尚待验证。

### **未来工作方向**
- **端到端训练 TreeFlash**：联合优化主干与 AR-head，探索更适合 tree generation 的预训练目标。
- **缓解 exposure bias**：引入课程学习或强化学习策略，使模型适应自回归生成环境。
- **适配生产系统**：研究如何与 quantization、multi-batch decoding、custom kernels 协同优化。
- **扩展至更多模型族**：验证在 Llama、Mixtral、Gemma 等架构上的迁移能力。
- **探索更复杂的条件机制**：如引入局部注意力或路径感知 embedding。

---

> 🔚 **总结一句话**：  
> **TreeFlash 通过轻量级 AR-approximation，在不牺牲 one-shot 并行性的前提下，显著缩小了 drafter 与 verifier 的分布差距，成为当前 tree-based speculative decoding 的 SOTA 方案。**

</details>

---

### 7. [StepFinder: A Temporal Semantic Framework for Failure Attribution in Multi-Agent Systems](https://arxiv.org/abs/2606.03467)

**Authors**: Taiyu Zhu, Yifan Wu, Weilin Jin, Ying Li, Gang Huang  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.03467v1  

#### Abstract
LLM-based multi-agent systems exhibit remarkable collaborative capabilities in complex multi-step tasks. However, these systems are highly sensitive to single-step execution errors that can propagate through agent interactions and lead to cascading failures. To understand the causes of failure and i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：StepFinder: A Temporal Semantic Framework for Failure Attribution in Multi-Agent Systems**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
在基于 **Large Language Model (LLM)** 的 **Multi-Agent System (MAS)** 中，尽管多智能体协作在复杂任务中表现出色，但系统对单步执行错误极为敏感，这些错误会通过交互传播并引发级联失败（cascading failures）。因此，**failure attribution**（失败归因）——即自动识别导致任务失败的根因步骤（root cause step）——成为提升系统可靠性的关键。

然而，现有的归因方法主要依赖 LLM 对原始执行轨迹进行推理，存在以下问题：
- **高推理成本和延迟**（high inference cost and latency）
- **受冗余和噪声日志干扰**，导致 LLM 难以准确识别真正的根因步骤

### **提出了什么新方法或新思路**
本文提出 **StepFinder**，一种轻量级的 failure attribution 框架，其核心思想是将 failure attribution 重构为一个**结构化的时序建模问题**，而非直接依赖 LLM 进行端到端推理。

#### **核心创新点**：
1. **仅在特征构建阶段使用 LLM**  
   使用 Qwen3 Embedding 将每一步的执行内容（content）和代理身份（agent identity）编码为**时序语义序列**（temporal semantic sequence），避免在推理阶段调用 LLM 生成文本。

2. **参数高效的深度学习架构**  
   结合 **BiLSTM**（捕捉时序依赖）与 **Agent-Aware Attention**（建模跨步依赖与代理间交互），实现对轨迹演化过程的精细建模。

3. **精细化的异常评分机制**  
   引入两个关键机制优化最终的 step-level error score：
   - **Multi-Scale Differencing**：捕捉不同时间尺度下的状态波动，增强对渐进或突发错误的敏感性。
   - **Position Bias**：引入线性衰减的位置先验，优先考虑早期步骤作为潜在根因，符合“早期决策决定成败”的直觉。

4. **联合损失函数设计**  
   - 主任务：监督分类损失（Cross-Entropy），定位根因步骤。
   - 辅助任务：自监督的 **Temporal Consistency Loss**，预测下一步隐藏状态，增强模型对正常执行流的理解。

### **相比现有方法的优势**
| 维度 | StepFinder | LLM-based 方法 |
|------|-----------|----------------|
| 推理效率 | ⚡️ 极高（无文本生成） | ❌ 极低（需多次调用 LLM） |
| 成本 | ✅ 低（仅编码阶段使用 LLM） | ❌ 高（推理也依赖 LLM） |
| 准确性 | ✅ 更高（尤其在复杂轨迹上） | ⚠️ 易受噪声和上下文长度影响 |
| 可扩展性 | ✅ 支持大规模轨迹分析 | ❌ 难以部署于实时诊断 |

---

## 2. 核心实验方法和设置

### **使用的数据集**
- **Who&When benchmark** [47]：当前唯一标准化的 MAS failure attribution 数据集。
  - 包含两类失败轨迹：
    - **Algorithm-Generated (Alg)**：126 条测试轨迹（自动化系统生成）
    - **Hand-Crafted (HC)**：58 条测试轨迹（人工构造，更复杂）
- **训练数据**：通过 LLM 重生成策略扩充，共生成：
  - Alg 子集：1,564 条训练轨迹
  - HC 子集：2,604 条训练轨迹
- 训练与测试任务级别严格分离，确保评估可靠性。

### **实验设置**
- **硬件环境**：NVIDIA RTX 3080 Ti GPU
- **实现框架**：Python 3.12 + PyTorch 2.3.0 + CUDA 12.1
- **编码器**：Qwen3-Embedding-0.6B
- **模型结构**：
  - Temporal Feature Extraction：BiLSTM（2层，hidden dim=64）
  - Agent-Aware Interaction：2头注意力，带 agent-aware bias 和 gating
  - 输出层：非线性投影 + multi-scale differencing + position bias
- **优化器**：AdamW，weight decay=1e-5，梯度裁剪，早停机制

### **评估指标**
#### （1）**Attribution Precision**
- **Accuracy**：预测得分最高的步骤是否匹配真实根因。
- **Tolerance Accuracy**：允许 ±δ 步误差（δ ∈ {1,…,5}），衡量容错能力。

#### （2）**Ranking Quality**
- **Acc@K**：真实根因出现在前 K 名候选中的比例。
- **MRR@3**（Mean Reciprocal Rank@3）：衡量排名质量，越靠前得分越高。

### **基线方法对比**
| 类别 | 方法 |
|------|------|
| **Random** | 随机选择步骤作为根因 |
| **LLM-based Attribution** | 使用 GPT-4o 的三种策略：<br>• All-at-Once<br>• Step-by-Step<br>• Binary Search<br>（部分实验提供 Ground Truth 辅助） |
| **Sequential Models** | 替换 StepFinder 模块：<br>• BiGRU（RNN-based）<br>• TCN（CNN-based）<br>• Transformer（Attention-based） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **主结果：Accuracy 对比（Table 2）**
| 方法 | Alg (%) | HC (%) |
|------|--------|-------|
| Random | 15.08 | 4.60 |
| Best LLM-based (Step-by-Step-G) | 20.11 | 6.90 |
| Best Sequential (BiGRU) | 23.28 | 12.64 |
| **StepFinder (Ours)** | **29.63** | **22.99** |

- 在 Alg 子集上比最强 baseline 提升 **+9.52%**
- 在 HC 子集上提升 **+10.35%**（相对提升超 200%）

#### ✅ **容忍精度（Tolerance Accuracy）**
- 如图 3 所示，在所有容忍阈值下均显著领先。
- 在 Alg 上，当 δ=1 时，StepFinder 比最佳 LLM 方法高出约 **15%**，说明其预测高度集中于真实根因附近。

#### ✅ **排名质量（Ranking Quality）**
| 指标 | Alg | HC |
|------|-----|-----|
| **Acc@3** | 60.31% (**SOTA**) | 30.46% (**远超其他**) |
| **MRR@3** | ~50% (**最高**) | ~25% (**领先约 10%**) |

- 在 HC 子集上，StepFinder 的 **Acc@1 已超过多数基线的 Acc@3**，显示其极强的排序能力。

#### ✅ **效率分析（Table 6）**
| 方法 | Inference Time (Alg) | Output Tokens |
|------|------------------------|---------------|
| All-at-Once | 2.92s | 138.23 |
| Step-by-Step | 26.18s | 1,626.20 |
| **StepFinder** | **0.61s** | **0.00** |

- **推理速度提升近 5 倍**（相比最快 LLM 方法）
- **减少 79% 推理时间**
- **零文本生成开销**（no text generation overhead）

#### 🔍 **消融实验（Ablation Study, Table 5）**
| 移除组件 | Alg (%) ↓ | HC (%) ↓ | 发现 |
|---------|----------|----------|------|
| w/o TFE（BiLSTM） | 25.13 (-4.5) | 12.64 (-10.35) | 时序建模至关重要 |
| w/o ASI（Agent-Aware Attention） | 27.51 (-2.12) | 18.97 (-4.02) | 跨步交互建模有效 |
| w/o AI（Agent Identity） | 28.04 (-1.59) | 19.54 (-3.45) | 代理身份信息有增益 |
| w/o MsDiff（Multi-Scale Diff） | 29.37 (-0.26) | 19.54 (-3.45) | 对复杂轨迹更重要 |
| w/o PB（Position Bias） | 26.98 (-2.65) | 20.11 (-2.88) | 有助于纠正后期偏差 |
| w/o TCLoss（Temporal Consistency） | 25.13 (-4.5) | 16.67 (-6.32) | 自监督任务显著提升稳定性 |

> ✅ 完整模型在所有子集上表现最优，各模块均有贡献。

---

## 4. 关键结论和发现

### **主要发现**
1. **Failure attribution 应被视为时序语义建模问题**，而非纯语言推理任务。
2. **StepFinder 通过轻量级神经网络 + LLM 编码**，实现了精度与效率的双重突破。
3. **Agent identity、multi-scale dynamics、position bias** 是精准定位根因的关键信号。
4. **自监督的 future-step prediction 损失** 显著提升了模型对正常执行流的感知能力。
5. 在复杂、长程、交互密集的轨迹中（如 HC 子集），StepFinder 明显优于 LLM-based 方法，因其不易被冗余上下文干扰。

### **方法的局限性**
- 当前模型假设代理按轮询顺序执行（round-robin），未处理动态调度或并发行为。
- 依赖高质量的失败轨迹标注（尤其是根因步骤标签），在实际系统中获取成本较高。
- 对极端短轨迹（<3步）或高度非结构化日志可能泛化能力受限。

### **未来工作方向**
- 扩展至支持 **并发执行** 和 **异步通信** 的 MAS 架构。
- 探索 **few-shot 或 unsupervised setting** 下的 failure attribution。
- 将 StepFinder 集成到 MAS runtime 中，实现实时监控与自动修复建议。
- 结合 causal inference 方法进一步验证归因结果的可解释性。

---

> 📌 **一句话总结**：  
> StepFinder 通过将 LLM 用于**语义编码**而非**推理主体**，结合**时序建模**与**代理感知注意力**，实现了高效、精确、鲁棒的 step-level failure attribution，在 Who&When benchmark 上全面超越 LLM-based 与传统序列模型，为 MAS 的可维护性提供了实用解决方案。

</details>

---

### 8. [Experience-Driven Dynamic Exits for LLMs with Reinforcement Learning](https://arxiv.org/abs/2606.03113)

**Authors**: Yanyu Zhu, Hoilam Pao, Niu Hu, Wei Guo, Shaoxiong Zhan, Boyu Lai, Zitai Wang, Yongqin Zeng, Hai-Tao Zheng  
**Category**: cs.CL  
**Published**: 2026-06-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.03113v1  

#### Abstract
Large Language Models suffer from slow autoregressive inference. While self-speculative decoding accelerates this process, its efficiency is hampered by static configurations like fixed exit layers and speculation lengths. We reframe this optimization as a \textbf{Markov Decision Process} and propos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Experience-Driven Dynamic Exits for LLMs with Reinforcement Learning**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题
大型语言模型（LLMs）在自回归推理过程中存在显著延迟，因为每个 token 都需通过完整的 Transformer 层堆栈。虽然 **Self-Speculative Decoding (SSD)** 利用模型早期层作为“draft”来并行生成候选 token 以加速推理，但其效率受限于**静态配置**——如固定的 exit layer 和 speculation length。这种刚性策略无法适应不同上下文下的 token 预测难度差异（即 contextual sparsity），导致 draft 质量与计算成本之间的权衡次优。

### 🚀 提出了什么新方法或新思路
本文提出 **LEDE (Learning-based Dynamic Exit)**，首次将 SSD 中的动态控制问题建模为 **Markov Decision Process (MDP)**，并通过 **offline Reinforcement Learning (RL)** 学习一个策略，用于在每一步推理中动态选择最优的：
- **Exit Layer $ l^* $**：决定从哪一层开始进行 speculative draft；
- **Speculation Length $ d $**：自适应地确定 draft 的长度。

该策略基于当前生成序列的局部上下文状态（如 token confidence、entropy 等）做出决策，实现对计算深度和 speculation 效率的联合优化。

### 🔍 相比现有方法的优势
| 方法类型 | 缺陷 | LEDE 的改进 |
|--------|------|------------|
| 固定配置（如 LayerSkip） | 对所有输入使用相同 exit layer 和 speculation length，泛化能力差 | 动态适配上下文复杂度，提升整体效率 |
| 启发式规则（如 LITE、DV） | 依赖预设阈值或简单 heuristics，灵活性有限 | 使用 RL 学习更精细、鲁棒的控制策略 |
| 外部 draft 模型（传统 SD） | 需额外训练小模型，部署复杂 | 完全利用目标模型自身结构，无需外部组件 |

> ✅ **核心优势**：LEDE 是首个将 offline RL 应用于 SSD 控制策略学习的工作，实现了经验驱动的、上下文感知的动态退出机制。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集
实验覆盖多种任务和领域，验证方法的通用性：
- **指令跟随（Instruction Following）**：Alpaca, TOPv2
- **语言建模与摘要（Summarization）**：CNN/DailyMail (CNN/DM)
- **代码生成（Code Generation）**：HumanEval

### ⚙️ 实验设置
- **模型范围**：LLaMA-3.2-1B, LLaMA-2-7B, LLaMA-2-13B, CodeLLaMA-7B, CodeLLaMA-34B
- **基础架构**：所有模型均基于 LayerSkip 进行持续预训练，支持中间层输出
- **解码方式**：
  - Language Modeling：sampling decoding（temperature=0.6）
  - 其他任务：greedy decoding
- **Acceptance 策略**：speculative sampling [32]
- **硬件环境**：NVIDIA A100 (80GB), CUDA 12.2, PyTorch 2.6.0
- **实现基础**：基于 LayerSkip 开源代码库构建

### 📊 评估指标
| 指标 | 描述 |
|-----|------|
| **Average Speculation Length ($ \bar{d} $)** | 平均每次 speculation 生成的 draft token 数量 |
| **Acceptance Rate (Acc. Rate)** | draft token 被接受的比例，反映 draft 质量 |
| **Average Exit Layer ($ \bar{E} $)** | 实际用于 drafting 的平均 exit layer，越低表示计算越少 |
| **Speedup** | 相对于 autoregressive decoding 的端到端加速比 |
| **Rouge-L (R-L)** | 衡量生成质量是否受损 |

### 🆚 基线方法对比
| 基线 | 特点 |
|------|------|
| **Autoregressive (AR)** | 标准非加速解码，基准（1.00×） |
| **LayerSkip (LS)** | 固定 exit layer 和 speculation length 的 SSD 方法 |
| **LITE** | 基于 confidence 阈值的 rule-based 动态早退机制 |
| **Draft & Verify (DV)** | 固定 exit layer + 自适应 speculation length 控制 |

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（来自 Table 1 & 2）

| 模型 | 方法 | Speedup | Acc. Rate | $ \bar{E} $ | $ \bar{d} $ |
|------|------|---------|-----------|-------------|--------------|
| LLaMA-3.2-1B | LEDE | **2.04× ~ 2.28×** | 0.867 ~ 0.924 | 3.96 ~ 6.88 | 4.70 ~ 6.84 |
| LLaMA-2-7B | LEDE | **2.58× ~ 2.72×** | 0.893 ~ 0.981 | 6.82 ~ 8.72 | 6.20 ~ 9.40 |
| LLaMA-2-13B | LEDE | **2.51× ~ 2.57×** | 0.928 ~ 0.969 | 9.36 ~ 12.73 | 4.63 ~ 8.06 |
| CodeLLaMA-7B | LEDE | **2.18×** | 0.748 | 8.35 | 5.93 |
| CodeLLaMA-34B | LEDE | **2.07×** | 0.864 | 14.57 | 4.41 |

> 💡 **最高提速达 2.72×**，相比 autoregressive decoding 提速超过两倍以上。

### 🔁 与基线方法对比结果
- 在所有模型和任务上，**LEDE 显著优于所有 baseline**：
  - 相比 **LayerSkip**，获得 **额外 17% 的加速增益**
  - 相比 **DV 和 LITE**，在 acceptance rate 更高或相当的情况下，实现更高 speedup
- **R-L 分数几乎无损**，说明生成质量未受影响

### 🔍 消融实验结果（Ablation Study on LLaMA2-7B）
| 配置 | Speedup | Acc. Rate | 结论 |
|------|--------|-----------|------|
| **LEDE (Full)** | **2.7×** | 0.858 | 完整模型表现最佳 |
| w/o Adaptive Drafting | 2.04× | 0.700 | 移除动态 speculation length 导致性能大幅下降 |
| w/o Dynamic Exit | 1.99× | 0.690 | 移除动态 exit layer 同样严重削弱效果 |

> ✅ **双控机制协同作用明显**：动态 exit layer 和 adaptive drafting length 缺一不可，共同贡献高效能。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **上下文相关性至关重要**：并非所有 token 都需要深层推理，LEDE 成功能利用 contextual sparsity 实现“按需计算”。
2. **RL 可有效学习动态控制策略**：通过 offline RL 从历史推理轨迹中学习，能自动发现比人工设计规则更优的 exit 策略。
3. **动态联合优化优于单一维度调整**：同时控制 exit layer 和 speculation length 比仅优化其中一个维度带来更大收益。
4. **高性能且无损生成质量**：在大幅提升推理速度的同时，保持 Rouge-L 等指标稳定，证明方法实用性强。

### ⚠️ 方法的局限性
- 当前训练依赖于 **experience replay buffer**，需要先收集大量探索数据（约 200 episodes），可能增加前期开销。
- 所有实验集中在 **<34B 规模模型**，尚未验证在 70B+ 超大规模模型上的可扩展性。
- offline RL 政策一旦训练完成较难在线更新，缺乏实时反馈适应能力。

### 🔮 未来工作方向
- 将 LEDE 扩展至 **70B 及以上规模的 LLMs**
- 探索 **online fine-tuning 或 continual learning** 机制，使策略能随输入分布变化而演进
- 结合 **其他 dynamic computation 技术**（如 layer skipping、k-NN retrieval）进一步压缩冗余计算

---

> 🏁 **总结一句话**：  
> LEDE 通过引入 **offline RL 驱动的动态 exit 策略**，实现了对 SSD 过程的经验化、智能化控制，在多个 LLM 上达成 **高达 2.7× 的推理加速**，是迈向高效、自适应 LLM 推理系统的重要一步。

</details>

---

### 9. [Perceive Before Reasoning: A Pre-Reasoning Perception Framework for Efficient and Reliable Proactive Mobile Agents](https://arxiv.org/abs/2606.03236)

**Authors**: Zhijie Ding (HyperAI Team, Xiaomi Corporation, Zhongnan University of Economics and Law), Weinan Hong (HyperAI Team, Xiaomi Corporation, Jilin University), Zicheng Zhu (HyperAI Team, Xiaomi Corporation, The Chinese University of Hong Kong, Shenzhen), Lei Li (HyperAI Team, Xiaomi Corporation), Dezhi Kong (HyperAI Team, Xiaomi Corporation), Hao Wang (HyperAI Team, Xiaomi Corporation), Peng Zhou (HyperAI Team, Xiaomi Corporation), Xuchu Jiang (HyperAI Team, Xiaomi Corporation), Jiaming Xu (HyperAI Team, Xiaomi Corporation)  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.03236v1  

#### Abstract
Multimodal large language models (MLLMs) have substantially advanced mobile agents, yet proactive mobile assistance remains challenging because agents must decide \emph{when} to intervene before determining \emph{how} to assist. Existing systems often implement these two decisions within a unified M...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Perceive Before Reasoning: A Pre-Reasoning Perception Framework for Efficient and Reliable Proactive Mobile Agents  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前基于 **Multimodal Large Language Models (MLLMs)** 的主动型移动智能体（proactive mobile agents）面临两个核心挑战：

- **目标错位（Goal Misalignment）**：在统一的 MLLM 框架中，同时处理“何时干预”（when to intervene）和“如何协助”（how to assist）导致目标冲突。“何时”需要保守、高判别力的决策以避免误触发；而“如何”则需开放、复杂的多模态推理。两者耦合难以兼顾。
- **推理效率低下（Inference Inefficiency）**：即使系统应保持沉默（no intervention），仍会启动完整的 VLM 推理流程，造成不必要的计算开销，尤其在移动端资源受限场景下不可接受。

### ✅ 提出的新方法与新思路

作者提出 **Pre-Reasoning Perception Framework (PRPF)**，其核心思想是 **“感知先于推理”（Perceive Before Reasoning）**，通过架构级解耦实现高效可靠的主动服务。

#### 主要组件：
- **Multimodal Proactive Perceptor (MPP)**：一个轻量级的多模态融合编码器，作为前置感知模块，负责：
  - **Intervention Gating**：判断是否需要触发推荐（减少 false triggers）
  - **Context Compression**：将上下文压缩为 Top-K 场景候选函数，缩小后续推理空间
- **Proactive Agent Reasoner (PAR)**：仅在 MPP 判断需干预时才激活，执行深度、聚焦的“如何协助”推理。

该框架采用 **两阶段流水线设计**：
1. **Stage 1 (When)**：由 MPP 快速过滤非干预样本
2. **Stage 2 (How)**：由 PAR 对保留样本生成可执行 function-call 序列

### ✅ 相比现有方法的优势

| 维度 | PRPF 优势 |
|------|----------|
| **可靠性** | 显著降低 False Trigger Rate (FTR)，提升用户体验 |
| **效率** | 减少冗余推理，大幅降低计算量（TFLOPs）和端到端延迟 |
| **灵活性** | MPP 可插拔（plug-and-play），适配不同 backbone 的 reasoner |
| **准确性** | 分离目标后，各模块专注优化自身任务，提升整体 Success Rate |

---

## 2. 核心实验方法和设置

### 📚 数据集

- **ProactiveMobile Benchmark** (Kong et al., 2026)：本文主要评测平台
  - 包含 **14 类高阶意图场景**（如旅行、餐饮、金融等）
  - 输入包括：用户画像（U）、设备状态（D）、世界信息（W）、交互轨迹（I）
  - 轨迹形式分为：
    - **Multimodal**：连续 GUI 截图序列
    - **Text**：文本化的行为轨迹描述
  - 输出：可执行的 function-call 序列（来自 63 个复合 API 函数池）
  - 总计约 8,876 条训练样本，测试集 3,660 条

### 🎯 评估指标

遵循 ProactiveMobile 设定三大核心指标：

| 指标 | 含义 |
|------|------|
| **Type-Acc** ↑ | 预测的 function-name 序列与真实值完全匹配（顺序+长度） |
| **Success Rate (SR)** ↑ | 综合功能等价性判断（由 Gemini-2.5-Pro 作为 LLM Judge 评估参数语义一致性） |
| **False Trigger Rate (FTR)** ↓ | 不需要推荐却被错误触发的比例 |

此外还报告：
- 推理效率：每样本计算量（TFLOPs）、峰值 GPU 内存（GB）、端到端延迟（ms）

### 🔁 基线方法对比

共三类 baseline 进行比较：

| 类型 | 模型列表 |
|------|--------|
| **Closed-source** | GPT-5.5, o3, Gemini-3.1-Pro, Claude-Opus-4.7, GLM-4.6V, Kimi-K2.5, MiMo-2.5v |
| **Open-source** | TongUI-7B, Qwen3.5-9B |
| **Proactive Intelligence Models** | ProactiveMobile(7B), UI-TARS-7B-DPO+Proactive, Qwen3.5-9B+Proactive（SFT 微调版） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1）

| 模型 | Type-Acc (%) | SR (%) | FTR (%) |
|------|-------------|--------|---------|
| **Qwen3.5-9B+Proactive** (best baseline) | 49.51 | 34.56 | 13.49 |
| **PRPF (Ours)** | **55.00** | **41.15** | **7.21** |

> ✅ **相比最强微调基线，PRPF 提升显著：**
- **SR 提升 +6.59 pts**
- **FTR 下降 -6.28 pts**
- **Type-Acc 提升 +5.49 pts**

在 **Text 设置下，FTR 低至 1.75%**，显示极强的静默控制能力。

### ⚖️ 与基线对比亮点

- 封闭模型如 GPT-5.5 虽强大，但 **FTR 高达 23.84%**，说明缺乏针对性训练易频繁打扰用户。
- PRPF 在 **所有指标上均优于所有 closed/open-source 模型**，验证了架构设计的有效性。

### 🔍 消融实验结果（见 Table 2）

| 消融配置 | SR (%) | FTR (%) | 结论 |
|--------|-------|--------|------|
| Full PRPF (Ours) | 41.15 | 7.21 | — |
| -w/o MPP | 38.33 | 10.38 | 移除 MPP 导致效率下降、误触发上升 |
| -w/o PAR | 33.44 | 18.46 | PAR 是高质量输出的关键 |
| -w/o Slow Channel | 39.73 | 9.38 | 长期偏好建模重要 |
| -w/o Fast Channel | 40.41 | 7.32 | 短期动态捕捉也有贡献 |
| -w/o Compression | 40.55 | 7.78 | 上下文压缩对效率有帮助 |
| -w/o Recommend (仅压缩) | 39.29 | 9.74 | **干预门控比压缩更重要** |
| -w/o GRPO | 39.67 | 15.56 | GRPO 显著稳定无干预决策 |
| -w/o SFT | 32.51 | 18.61 | SFT 构建基础生成能力 |

> 💡 **关键发现：**
- MPP 的 **trigger gating 功能最为关键**，直接决定系统可靠性
- PAR 的 **GRPO 优化有效抑制 false positives**
- “慢通道”（长期行为）比“快通道”（短期界面）影响更大

### ⚡ 效率分析（见 Figure 3）

| 指标 | PRPF vs ProactiveMobile(7B) |
|------|----------------------------|
| **Expected Compute (TFLOPs)** | ↓ **69.3%** |
| **End-to-End Latency (ms)** | ↓ **60.1%** |
| **Peak Memory (GB)** | ↑ 12.0%（因运行更强的 9B PAR） |

尽管 PAR 更大，但由于 MPP 提前过滤了大量无需干预样本，整体推理成本大幅下降。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **“感知先于推理”范式有效**：将 **when** 与 **how** 解耦，能更好平衡保守性与生成能力，解决传统统一框架的目标冲突。
2. **MPP 实现高效预筛**：轻量级 MPP 可过滤约 **77% 无需干预样本**，同时保留超 90% 的真实需求，极大提升系统可用性。
3. **显著提升性能与效率**：PRPF 在 ProactiveMobile 上实现 **41.15% SR 和 7.21% FTR**，并节省近七成计算资源。
4. **MPP 具备良好泛化性**：可在不同 backbone（如 GLM-4.6V、Qwen3.5-9B）上即插即用，带来一致收益（见 Table 3）。

### ⚠️ 局限性

1. **MPP 依赖预定义场景分布**：若迁移到新领域或扩展 API 池，需重新训练 MPP 以学习新的 intent-scenario 分布。
2. **多模态理解仍是瓶颈**：
   - 多模态任务绝对成功率仅 **17.19% SR**
   - 即使最强模型也难突破 18%，表明当前 VLM 对 GUI 的细粒度理解不足
3. **失败集中在特定场景**：Travel & Lodging、Shopping 等生活类场景中，“off-scene” 错误较多，说明意图混淆问题仍存。

### 🔮 未来工作方向

- 探索 **continual learning** 或 **prompt-based adapter** 技术，降低 MPP 在新领域的迁移成本
- 强化 **视觉 grounding 能力**，如更高分辨率 GUI 编码器、更精细的区域感知机制
- 扩展 **function set 规模** 并优化 Top-K 过滤策略，缓解 distractor 注入问题（见 B.4）
- 改进 **reward shaping** 与 **candidate scoring**，提升 PAR 在复杂场景下的路由准确率

---

> 🧭 **总结一句话**：  
> PRPF 通过引入轻量级 **MPP** 实现“感知先行”，成功分离“是否干预”与“如何协助”两大任务，在大幅提升 **可靠性（↓FTR）** 与 **效率（↓Compute）** 的同时，增强了 **整体成功率（↑SR）**，为构建实用化的主动式移动智能体提供了新范式。

</details>

---

### 10. [Calibrating Urban Traffic Simulation from Sparse Road Observations via Genetic Optimization](https://arxiv.org/abs/2606.03823)

**Authors**: Hunter Sawyer, Jesse Roberts, Simon Matei  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.03823v1  

#### Abstract
Urban traffic simulation is a critical tool for infrastructure planning, including the placement of electric vehicle charging stations. However, realistic traffic simulation across many cities is hindered by two fundamental data limitations: detailed real-world traffic measurements are available for...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Calibrating Urban Traffic Simulation from Sparse Road Observations via Genetic Optimization*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文针对城市交通仿真在实际部署中面临的两大核心数据挑战：
- **稀疏的道路观测数据**：大多数城市的道路仅有极小部分路段具备真实交通流量测量数据。
- **缺乏高分辨率就业分布数据**：传统仿真依赖详细的OD（Origin-Destination）需求建模，尤其是通勤出行，但这类数据通常难以获取或分辨率不足。

这些问题严重限制了交通仿真模型在多样化城市中的可扩展性和实用性。

### 🚀 提出的新方法与创新思路
提出了一种基于 **Genetic Algorithm (GA)** 的新型框架，用于从**稀疏道路观测数据**中校准城市交通仿真，其核心创新在于：
- **优化空间活动分布而非驾驶行为参数**：不同于多数现有研究聚焦于校准微观驾驶行为（如car-following、lane-changing模型），本文首次将GA用于直接优化**job distribution（就业岗位的空间分布）** 和 **gate-traffic parameters（进出城流量）**。
- **无需真实就业数据进行训练**：模型仅依赖少量已知路段的交通流数据，通过遗传算法反向推断最可能的工作岗位布局，从而生成符合现实的通勤模式。
- **端到端联合优化**：同时优化三个关键参数组构成“基因组”（genome）：
  - 工作岗位在道路段上的分布（job distribution）
  - 总体进出城车辆数量（total gate traffic）
  - 各入口门（gates）之间的流量分配比例（gate-traffic distribution）

### ⚖️ 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 数据依赖 | 需要完整OD矩阵、详细人口统计、高密度传感器数据 | 仅需稀疏道路流量 + 开放数据（OpenStreetMap + Census住房数据） |
| 可扩展性 | 低，难以迁移到数据匮乏城市 | 高，适用于全球多数城市 |
| 优化目标 | 微观驾驶行为或路径选择 | 宏观空间需求结构（activity-based demand） |
| 泛化能力 | 易过拟合局部路段 | 在未见路段上表现良好 |

> ✅ **核心优势总结**：实现了“data-light”且“scalable”的仿真校准范式，显著降低了城市级交通建模的数据门槛。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
| 数据类型 | 来源 | 描述 |
|--------|-----|------|
| 路网数据 | OpenStreetMap | 导入Greensboro, NC区域的完整道路网络至SUMO |
| 住宅分布 | U.S. Census Bureau (2010 TAZ) | 将住户按最近TAZ均匀分配到各道路段 |
| 真实交通流量 | North Carolina DOT (2024) | 提供部分道路段的Annual Average Daily Traffic (AADT)，覆盖极少比例路段 |
| 就业真实分布（用于验证） | U.S. Census Bureau 就业密度图 | 用于RQ4中与模拟结果定性对比 |

### 🔧 实验设置
- **仿真平台**：  
  - 主体工具：**SUMO**（microscopic traffic simulation）
  - 流量生成器：**ActivityGen**（基于活动的需求建模，支持work/school/free-time trip generation）
- **初始设定**：
  - 模拟1000人分布在500个家庭中（远小于真实人口，但用于相对模式比较）
  - 住房分布固定，工作岗位分布由GA动态调整
- **GA配置**：
  - 种群大小：16个个体（children）
  - 迭代代数：200代
  - 选择机制：每代选出最优个体作为“parent”
  - 交叉（crossover）与变异（mutation）策略针对三类参数分别设计
  - “基因组”维度随时间动态调节突变率（通过sin函数控制，避免陷入局部最优）

### 🎯 评估指标
| 研究问题 | 评估方式 |
|--------|--------|
| RQ1: 是否提升对齐度？ | 计算**simulated vs. real traffic**在训练路段上的**Pearson相关系数（correlation）** |
| RQ2/RQ3: 是否泛化到未见路段？ | 分别计算**随机排除**和**地理区域排除**路段的相关性 |
| RQ4: 恢复的job distribution是否合理？ | 与Census就业热力图进行**视觉对比（heatmap overlay）** |

### ❌ 基线方法对比
本文**未直接对比传统手动校准或其他算法基线**，而是强调：
- 与现有GA方法的本质区别：已有工作多集中于driver behavior calibration，而本文是首个专注于**spatial demand structure optimization**的工作。
- 强调“zero-shot”性质：从未使用真实就业数据训练，却能恢复出类似结构。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ RQ1: 全量训练集校准效果
- 经过200代进化后，**平均correlation达到0.64~0.72之间**（见Fig. 1）
- 所有5次运行均显示快速上升趋势 → 表明GA有效收敛
- **结论**：遗传算法显著提升了仿真与真实交通的对齐程度

#### ✅ RQ2 & RQ3: 对未见路段的泛化能力
| 排除类型 | 结果表现 |
|--------|--------|
| **随机排除（15%-75%）** | 多数情况下，held-out路段的相关性**不低于甚至高于训练集**，尤其在中等排除比例下表现更优 |
| **地理三角区排除（Triangle Zones）** | 表现不一致：<br> - Triangles 1&2：明显过拟合<br> - Triangles 3,4,5：训练与排除路段同步改善 |

> 💡 特别发现：随着排除比例增加（即训练集变小），某些条件下held-out性能反而更好 —— 支持“维度简化有助于捕捉宏观结构”的假设。

#### ✅ RQ4: 恢复的job distribution质量
- 图4显示：模拟生成的job density heatmap与Census真实数据在以下方面存在**显著重叠**：
  - 最高就业密度区域（如市中心）
  - 最低就业密度区域（郊区/农村）
- 尽管未参与训练，仍能恢复出合理的空间聚类结构

> 📌 这表明：**traffic signal中蕴含可恢复的城市结构信息**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **稀疏观测足以驱动高质量仿真**：即使只有少数道路具有真实流量数据，也能通过GA优化job和gate参数，使整体仿真交通流与现实高度相关（corr > 0.6）。
2. **良好的跨路段泛化能力**：在随机排除的路段上，仿真性能普遍提升，说明模型学习到了**城市尺度的交通组织规律**，而非记忆特定路段。
3. **空间覆盖优于数据密度**：当缺失数据呈地理聚集时（如整个城区无数据），泛化效果下降；因此建议部署时优先保证**观测点的空间分布广度**。
4. **隐含城市结构可被还原**：从未见过真实就业数据的情况下，恢复的job分布与Census数据呈现定性一致，证明**交通流是城市功能结构的强代理信号（proxy signal）**。

### ⚠️ 方法的局限性
- **定性验证为主**：job distribution的评估依赖视觉对比，缺乏定量指标（如KL散度、MAE等）。
- **单一城市验证**：实验仅在Greensboro开展，尚未验证跨城市迁移能力。
- **简化的人口规模**：1000人的模拟可能忽略复杂拥堵反馈机制。
- **静态日均流量假设**：使用AADT而非小时级动态数据，限制了对高峰时段的精细刻画。

### 🔮 未来工作方向
1. **多城市定量验证**：在不同规模、结构的城市中测试该框架的普适性。
2. **引入动态流量数据**：结合浮动车数据（FCD）或GPS轨迹，实现时间维度上的校准。
3. **融合多源异构数据**：整合手机信令、POI、遥感影像等辅助信息进一步增强约束。
4. **构建标准化benchmark**：推动建立“sparse-data traffic calibration”领域的公开评测基准。
5. **应用于EV基础设施规划**：利用此框架预测电动车充电需求空间分布，支撑电网协同规划（GreenEVT场景延伸）。

---

## ✅ 总结一句话
> 本文提出一种基于**Genetic Algorithm**的轻量级交通仿真校准框架，仅需**稀疏道路观测**即可有效恢复城市级交通流模式与潜在就业空间结构，为数据稀缺环境下的智能交通系统建模提供了可扩展的新路径。

</details>

---

### 11. [InfoMem: Training Long-Context Memory Agents with Answer-Conditioned Information Gain](https://arxiv.org/abs/2606.03329)

**Authors**: Tiancheng Han, Yong Li, Wuzhou Yu, Qiaosheng Zhang, Wenqi Shao  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.03329v1  

#### Abstract
Long-context tasks require LLMs to identify and preserve answer-relevant information from large contexts. Chunk-wise memory agents address this issue by sequentially reading document chunks, updating a compact memory, and generating the final answer from the accumulated memory. However, existing RL-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# InfoMem: Training Long-Context Memory Agents with Answer-Conditioned Information Gain —— 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **long-context 任务** 中，LLM 需要从超长文本中识别并保留与答案相关的关键信息。现有的 **chunk-wise memory agents** 虽然通过分块阅读和维护紧凑记忆来处理长上下文，但其训练过程存在以下问题：

- **稀疏奖励（sparse outcome reward）**：仅依赖最终答案是否正确作为反馈，无法区分不同成功轨迹中“记忆质量”的差异。
- **局部中间奖励（lexical intermediate rewards）**：如基于词重叠的记忆更新奖励，关注的是表面相似性而非语义支持。

这些问题导致模型难以学习到真正有助于生成正确答案的高质量最终记忆（final memory）。

---

### ✅ 提出的新方法：InfoMem
作者提出 **InfoMem**，一种基于 **answer-conditioned information gain** 的奖励机制，用于指导 chunk-wise memory agent 的强化学习（RL）训练。

#### 核心思想：
一个有用的最终记忆 $ M_K $ 应该能**提升模型对 ground-truth answer $ y^* $ 的预测置信度**。为此，InfoMem 定义了一个点态（pointwise）的信息增益奖励：

$$
r_{\text{gain}}(x, M, y^*) = \log P_\theta(y^* | x, M) - \log P_\theta(y^* | x, \varnothing)
$$

即比较在有/无最终记忆条件下，模型对真实答案的 **per-token average log-likelihood** 差值。

---

### ✅ 相比现有方法的优势

| 方面 | InfoMem | 传统方法 |
|------|--------|---------|
| **监督信号粒度** | 直接评估最终记忆对答案的支持能力 | 仅评估最终答案是否正确（outcome-only），或中间步骤的词重叠（ReMemR1） |
| **训练稳定性** | 仅在成功轨迹上应用信息增益，并进行归一化（normalization） | 原始奖励尺度波动大，易导致优化不稳定 |
| **语义理解能力** | 鼓励保留语义支持证据，而非简单复制查询或关键词 | 易陷入“query repetition”陷阱，记忆中堆砌无关内容 |

> InfoMem 是首个将 **information-theoretic 视角** 引入 long-context memory agent 训练的工作，强调“记忆效用”应以它对目标答案的条件支持程度来衡量。

---

## 2. 核心实验方法和设置

### 📚 数据集

| 数据集 | 描述 |
|-------|------|
| **RULER-HotpotQA** | 主训练数据集，由 200 个文档组成的长上下文 QA 数据，混合相关与干扰文档。每个样本约 1M tokens。为控制变量，作者从中采样 **512 个训练样本** 和完整的 128 个验证样本。 |
| **SQuAD** | 用于构建合成诊断任务（synthetic hallucinated-evidence），测试奖励能否区分真实支持上下文与语义相似但事实错误的幻觉上下文。 |

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| **Base Model** | Qwen2.5-1.5B-Instruct |
| **算法框架** | GRPO（Group Relative Policy Optimization） |
| **Rollouts per prompt** | 8 |
| **Chunk Size** | 5000 tokens |
| **Max Memory Length** | 1024 tokens |
| **Training Steps** | 120 |
| **InfoMem coefficient $ \beta $** | 0.2 |
| **GPU 资源** | ~440 GPU-hours on 16 NVIDIA H20 GPUs |

---

### 📊 评估指标与基准测试集

| Benchmark | 任务类型 | 评估方式 | 子集规模 |
|----------|--------|--------|--------|
| **MRCR-8needle** | 多针检索（multi-needle retrieval） | Sequence Match | 100 examples (from 800) |
| **RULER synthetic QA** | 稀疏问答 | F1 Score | 262K-token subset |
| **CorpusQA** | 文档集合级推理 | LLM-as-judge (Kimi-K2.6) | 128K-token subset |
| **LongMemEval** | 对话记忆追踪 | LLM-as-judge (Kimi-K2.6) | LongMemEval-S (115K tokens) |

> 所有方法使用相同模型、解码配置和评估协议，确保公平比较。

---

### 🔁 基线方法对比

| 基线 | 描述 |
|-----|------|
| **Initial Model** | 未经过 RL 微调的原始 Qwen2.5-1.5B-Instruct |
| **Outcome-only GRPO** | 仅使用最终答案正确性的稀疏奖励（$ R_{\text{outcome}} $） |
| **ReMemR1 (Shi et al., 2026)** | 使用词级召回率作为中间记忆奖励的 RL 方法，引入 callback retrieval 机制 |

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（Table 2）

| Model | CorpusQA (%) | LongMemEval (%) | MRCR-8needle (%) | RULER synthetic QA (%) |
|-------|--------------|------------------|------------------|------------------------|
| Initial Model | 14.590 | 5.600 | 0.260 | 13.308 |
| Outcome-only GRPO | 16.413 | 10.000 | **0.063** | 34.735 |
| **InfoMem (Ours)** | **19.453** | **12.800** | **0.279** | **36.848** |
| ReMemR1 | 1.520 | 6.200 | 0.092 | 28.919 |

> ✅ InfoMem 在所有四个 long-context benchmark 上均取得最佳表现。

#### 关键观察：
- **InfoMem 显著优于 Outcome-only GRPO**：说明单纯优化答案正确性不足以提升记忆质量。
- **Outcome-only GRPO 在 MRCR-8needle 上严重退化**（从 0.26 → 0.063），表明稀疏奖励可能导致灾难性遗忘或检索行为劣化。
- **ReMemR1 表现不佳**：可能因其奖励基于词重叠，鼓励“关键词堆砌”而非语义整合。

---

### 🔍 合成诊断实验（Table 1）

| 方法 | MRR | Z-score SNR |
|------|-----|-------------|
| Embedding (BGE-M3) | 0.719 | 0.316 |
| ATTN-TOP1 | 0.792 | 0.577 |
| **Tgain (Ours)** | **0.977** | **2.960** |

> InfoMem 的 $ r_{\text{gain}} $ 在区分真实 vs 幻觉上下文方面远超 embedding、attention 等传统方法，且信号更稳定（高 SNR），适合作为训练奖励。

---

### 🔧 消融实验（Ablation Study, Table 3）

| Model | CorpusQA | LongMemEval | MRCR-8needle | RULER synthetic QA |
|-------|----------|-------------|---------------|---------------------|
| InfoMem | 19.453 | 12.800 | 0.279 | 36.848 |
| w/o Tgain normalization | 16.109 | 11.600 | 0.029 | 35.371 |
| w/ QueryPMI（query-conditioned） | 18.237 | 8.000 | 0.117 | 26.163 |

#### 发现：
1. **不归一化（no normalization）** → 性能显著下降，尤其在检索任务上（MRCR-8needle 降为 0.029），说明原始 $ r_{\text{gain}} $ 尺度变化剧烈，需标准化以避免奖励失衡。
2. **使用 QueryPMI 替代 answer-conditioning** → 性能全面下降，且训练中出现大量 **query repetition**（见 Figure 4），说明模型学会“让 query 更容易被预测”，而非保留答案支持证据。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **有效的最终记忆奖励应具备三大特性**：
   - ✅ **Success-side supervision**：只在成功轨迹上施加信息增益奖励，避免失败轨迹中的噪声干扰。
   - ✅ **Pre-composition normalization**：在组合奖励前对 $ r_{\text{gain}} $ 进行组内归一化，保证奖励尺度一致。
   - ✅ **Answer-conditioning**：奖励必须基于 ground-truth answer，而非仅 query，否则会诱导模型重复提问而非提取证据。

2. **InfoMem 提供了一种原则性的记忆监督范式**：
   - 从信息论角度定义“有用记忆”——减少模型对答案的不确定性。
   - 使用可计算的点态似然增益作为代理信号，实现高效训练。

3. **实验证明 InfoMem 可持续提升 long-context memory agent 性能**，优于 outcome-only 和 intermediate lexical reward 方法。

---

### ⚠️ 局限性（Limitations）

1. **模型与数据规模受限**：
   - 使用较小的 Qwen2.5-1.5B 模型和仅 512 个训练样本，主要目的是验证奖励设计的有效性，尚未扩展至更大模型和数据集。

2. **适用范围有限**：
   - 当前方法专为 **chunk-wise memory agent** 设计，不适用于 full-context attention 或纯 retrieval-based 系统。

3. **仅在最终步定义奖励**：
   - 虽然 GRPO 将优势传播回整个轨迹，但奖励本身只作用于最终记忆和答案，未考虑中间记忆状态的逐步优化。

4. **潜在风险**：
   - 若 ground-truth answer 错误或文档有偏见，InfoMem 可能放大误导性信息。
   - 模型可能过度优化“与答案强关联”的记忆，而忽略原文中的重要限定条件。

---

### 🔮 未来工作方向

- 将 answer-conditioned information gain 扩展到 **intermediate memory states**，实现 step-wise 过程监督。
- 探索在 **larger models** 和 **real-world long-document applications**（如法律、医疗文档分析）中的应用。
- 结合 **human feedback** 或 **verifier models** 来缓解错误答案引导的风险。
- 探索与其他 memory architecture（如 MemGPT、Block-Recurrent Transformers）的结合。

---

> 💡 **一句话总结**：  
> InfoMem 提出了一种基于 **answer-conditioned information gain** 的新型奖励机制，首次从信息论视角直接监督 long-context memory agent 的最终记忆质量，在多个 benchmark 上显著超越现有方法，并揭示了 success-side supervision、reward normalization 和 answer conditioning 三项关键设计原则。代码已开源：[https://github.com/GenSouKa1/InfoMem](https://github.com/GenSouKa1/InfoMem)。

</details>

---

### 12. [Topology-Aware Gaussian Graph Repair for Robust Graph Neural Networks](https://arxiv.org/abs/2606.03462)

**Authors**: Anubha Goel, Juho Kanniainen  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.03462v1  

#### Abstract
Graph neural networks have achieved strong performance on graph-structured data, but their effectiveness depends heavily on the quality of the observed graph. In real applications, graph topology is often imperfect: noisy edges may connect unrelated nodes, while missing edges may prevent useful info...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Topology-Aware Gaussian Graph Repair for Robust Graph Neural Networks**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
图神经网络（GNNs）在图结构数据上表现出色，但其性能高度依赖于输入图拓扑的质量。然而，在现实应用中，观察到的图通常存在**结构不完整性**：
- **噪声边（noisy edges）**：连接无关节点的虚假边传播误导信息；
- **缺失边（missing edges）**：本应存在的语义相关节点之间缺乏连接，阻碍有用信息传播。

现有方法如**边剪枝（edge removal）** 只能缓解噪声边问题，却无法恢复缺失连接；而**图结构学习（graph structure learning）** 虽可重构图，但常引入复杂的优化过程和额外参数，影响效率与稳定性。

---

### **提出了什么新方法或新思路**
本文提出 **Topology-Aware Gaussian Repair (TAGR)** ——一种轻量级、模块化的图修复框架，用于提升 GNN 在结构受损图上的鲁棒性。

#### **核心思想：图修复（Graph Repair）而非完全重建**
TAGR 不是学习一个全新的稠密图，而是通过两个互补机制对原始图进行**稀疏修复**：
1. **Adaptive Gaussian Feature-Neighborhood Repair（自适应高斯特征邻域修复）**
   - 基于节点特征相似性构建辅助边，恢复潜在的语义邻居。
   - 使用局部自适应带宽的高斯核计算权重，避免全局阈值敏感性。
   - 仅选择 top-k 特征相近且无原始边的候选对，保持稀疏性。

2. **Topology-Aware Residual Reweighting（拓扑感知残差重加权）**
   - 对原始边进行保留但动态重加权，依据局部特征一致性（cosine similarity）和结构证据（Jaccard overlap, common neighbors, clustering coefficient, degree imbalance）。
   - 引入有界残差乘子 `τ_ij ∈ [0.5, 1.5]`，防止过度放大或抑制。

最终修复后的邻接矩阵为：
```
Â = A ⊙ R + A_GF_N
```
其中第一项是重加权的原始图，第二项是新增的特征邻域边。

---

### **相比现有方法的优势**
| 维度 | TAGR | 现有方法（如 Pro-GNN, GAug, RS-GNN） |
|------|------|-------------------------------|
| **复杂度** | 低：无需可训练图生成器，无双层优化 | 高：需联合训练图结构模块 |
| **稀疏性** | 高：仅添加 O(kn) 辅助边，保留原始边支持集 | 多数为稠密图学习或全对评分 |
| **兼容性** | 强：修复后图可直接接入标准 GNN 架构（如 GCN/GAT） | 弱：常需定制化模型设计 |
| **解释性** | 高：修复基于明确的特征与拓扑信号 | 低：图生成过程黑箱化 |

> ✅ **核心优势总结**：TAGR 实现了**轻量、高效、可插拔**的图修复，在不改变 GNN 架构的前提下显著提升鲁棒性。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
在四个标准引用网络基准上进行评估：

| Dataset | Nodes | Edges | Features | Classes |
|--------|-------|-------|----------|---------|
| **Cora** | 2,708 | 5,278 | 1,433 | 7 |
| **Citeseer** | 3,312 | 4,536 | 3,703 | 6 |
| **Cora-ML** | 2,995 | 8,158 | 2,879 | 7 |
| **Pubmed** | 19,717 | 44,324 | 500 | 3 |

此外，在附录中补充了 **Flickr** 和 **Cornell** 用于机制分析。

---

### **实验设置和评估指标**

#### **任务**
半监督节点分类（semi-supervised node classification）

#### **图扰动协议**
- **Clean Graph**：原始图
- **Edge Addition (Add X%)**：随机插入等于原边数 X% 的非真实边 → 测试抗噪能力
- **Edge Deletion (Del X%)**：随机删除 X% 的原始边 → 测试补全能力
- 扰动比例覆盖广泛范围（Add: 10%-90%，Del: 5%-50%），绘制鲁棒性曲线

#### **评估指标**
- 主要：**测试准确率（test accuracy %）**
- 辅助：平均排名（average rank）、homophily ratio 分析

#### **实现细节**
- 所有方法共享相同的数据划分、特征、标签、扰动图
- TAGR 超参数在干净图上选定后固定，不在扰动图上调参
- 结果取 **5 次随机种子均值 ± 标准差**

---

### **基线方法对比**
分为三类：

| 类别 | 方法 | 说明 |
|------|------|------|
| **标准 GNN** | GCN, GAT, GraphSAGE | 直接使用原始图作为基线 |
| **图修复变体** | G-GCN（仅高斯修复）, TAGR-GCN | 分离验证各组件作用 |
| **先进对比方法** | RS-GCN, JNSGSL | 代表性的学习型图修复与结构学习方法 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（来自 Table 2）**

| Setting | GCN | G-GCN | **TAGR-GCN** | RS-GCN | JNSGSL |
|--------|-----|-------|--------------|--------|--------|
| **Cora, Add 90%** | 67.9±1.1 | 72.3±1.0 | **72.3±1.2** | 74.8±1.5 | 70.1±2.1 |
| **Cora, Del 50%** | 72.7±1.6 | 74.9±1.4 | **75.2±1.5** | 74.4±1.5 | 74.4±2.1 |
| **Citeseer, Add 90%** | 56.5±0.9 | 60.5±1.1 | **61.3±1.0** | 66.4±1.1 | 57.3±1.6 |
| **Citeseer, Del 50%** | 61.9±1.4 | 66.4±1.4 | **66.4±1.4** | 64.8±1.3 | 64.9±1.4 |
| **Pubmed, Del 50%** | 73.1±0.6 | 73.2±1.4 | **73.5±0.8** | 65.5±2.4 | 73.0±1.9 |

> 🔍 观察：
> - TAGR-GCN 在多数设置下优于或持平于 vanilla GCN 和 G-GCN；
> - 尽管 RS-GCN 在某些高噪声场景更强（如 Cora Add 90%），但 TAGR 更稳定；
> - 在大规模 Pubmed 上，TAGR 显著优于 RS-GCN。

---

### **与基线方法的对比结果**

- **vs. Vanilla GNNs**：TAGR 在所有扰动设置下均带来一致增益，尤其在 GCN 上最明显（因 GCN 完全依赖邻接矩阵归一化）。
- **vs. RS-GCN**：
  - RS-GCN 在小图 + 高边添加噪声时表现最佳（显示学习型修复的强大拟合能力）；
  - 但在边删除或大图（如 Pubmed）上表现下降，且计算成本更高；
  - TAGR 以更简单方式达到接近甚至超越的性能。
- **vs. JNSGSL**：
  - JNSGSL 在干净图上表现良好，但在严重扰动下不稳定（如 Citeseer Add 90% 下降至 57.3%）；
  - TAGR 表现出更强的泛化性和鲁棒性。

---

### **消融实验结果**

#### **G-GCN vs. TAGR-GCN**
- **G-GCN（仅高斯修复）** 已能显著提升性能，表明**特征邻域修复是鲁棒性的主要来源**。
- **加入 topology-aware residual 后（即完整 TAGR）**，进一步提升了稳定性，尤其是在图不完整（Del 设置）时效果更明显。
  - 如 Cora-ML Del 25%：从 80.6% → 81.3%
  - Pubmed Del 50%：从 73.2% → 73.5%

> ✅ 发现：**高斯修复主导性能增益，残差重加权提供“稳定器”作用**

#### **鲁棒性曲线分析（Figure 1–3）**
- **Edge Addition 曲线**：随着噪声边增加，GCN 性能急剧下降；TAGR-GCN 和 G-GCN 明显缓解退化。
- **Edge Deletion 曲线**：随着边被删减，TAGR 仍能维持较高性能，说明其有效补偿了缺失连接。
- **平均排名（Figure 3）**：
  - TAGR-GCN 在 **edge addition 和 edge deletion 场景下的平均排名均为第一**，显示出跨扰动类型的综合最优鲁棒性。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **图修复优于单纯去噪或重构**  
   单纯删除边不能解决缺失连接问题，而完全学习新图又过于复杂。TAGR 提出“修复”视角——**既补充缺失语义边，又合理调整原始边权重**，实现了平衡。

2. ✅ **特征邻域修复是鲁棒性的核心驱动力**  
   自适应高斯机制能有效识别并连接特征相似但无边相连的节点，显著增强消息传递的语义一致性。

3. ✅ **拓扑感知残差具有稳定作用，且在特定图中可成主因**  
   - 在特征主导图（如 Flickr）中，残差起微调作用；
   - 在结构敏感图（如 Cornell）中，残差重加权本身即可大幅提升性能（见 Appendix），说明其不仅是“稳定器”，也可成为主动修复信号。

4. ✅ **轻量稀疏修复可媲美复杂结构学习方法**  
   TAGR 无需可训练图模块、无双层优化、无稠密评分，但仍能在多种扰动下与 RS-GCN、JNSGSL 等先进方法竞争，证明了**简洁设计的有效性**。

---

### **方法的局限性**
1. **依赖高质量节点特征**  
   若特征本身含噪或与任务无关，则高斯修复可能引入错误连接。

2. **残差重加权采用手工设计信号**  
   当前使用的 Jaccard、clustering 等统计量未经过端到端学习，可能无法捕捉复杂任务相关的拓扑模式。

3. **修复图在训练过程中固定**  
   无法根据训练动态调整图结构，缺乏自适应性。

4. **未考虑异质图、时序图等复杂场景**  
   当前框架适用于静态同质图，扩展至其他图类型需进一步研究。

---

### **未来工作方向**
- 探索 **adaptive 和 task-aware 的 TAGR 变体**，让修复过程参与梯度更新。
- 研究在 **feature noise 下的鲁棒性**。
- 扩展至 **inductive、heterogeneous、temporal 和 large-scale graph learning** 场景。
- 结合 **self-supervised learning 或 contrastive learning** 进一步提升修复质量。

---

> 📌 **总体评价**：  
> TAGR 是一项简洁而有力的工作，它重新定义了“图鲁棒学习”的范式——从“替换图”转向“修复图”。其实验充分、分析深入，提出的轻量级稀疏修复策略为实际部署提供了高性价比解决方案，具有良好的理论价值与工程意义。

</details>

---

### 13. [HiSE: A Lightweight Hierarchical Semantic Explainer for Heterogeneous Graph Neural Networks](https://arxiv.org/abs/2606.03495)

**Authors**: Zongrui Li, Yuhang Zhao, Ying Zhao, Yuanzhao Guo, Qiang Huang, Yuan Tian  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.03495v1  

#### Abstract
Heterogeneous graph neural networks (HGNNs) have demonstrated remarkable performance in modeling complex relational data, however their interpretability in high-stakes applications remains a critical challenge. Existing explanation methods suffer from two major limitations: on the one hand, the gene...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HiSE: A Lightweight Hierarchical Semantic Explainer for Heterogeneous Graph Neural Networks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Heterogeneous Graph Neural Networks (HGNNs)** 解释方法存在两大核心缺陷：
- **缺乏语义层次建模**：无法反映 HGNN 内部在 *semantic level* 和 *cross-semantic level* 的分层决策机制，导致解释与模型真实推理过程不一致。
- **计算效率低下且不稳定**：多数方法依赖复杂的搜索或扰动策略（如特征掩码、子图采样），在高维噪声环境下易失效，且计算开销大。

### 提出的新方法：HiSE
提出 **HiSE (Hierarchical Semantic Explainer)** ——一种轻量级、面向特征的层级语义解释框架，其核心思想是**镜像 HGNN 自身的分层建模结构**进行解释。

#### 创新架构
HiSE 包含两个核心模块：
- **Semantic Sparse Proxy (SSP)**  
  在每个 *semantic view*（由 meta-path 或 attention head 定义）上构建基于 **LASSO** 的局部稀疏代理模型，学习该语义视角下的特征重要性 $ s_k \in \mathbb{R}^n $，实现语义内稀疏表示。
  
- **Cross-Semantic Aggregative Inversion (CAI)**  
  使用 **KL 散度** 衡量单语义子图预测分布与全图预测分布之间的差异，将其负值作为相似性得分，并通过 **Softmax** 归一化为各语义的权重 $ w_k $，最终加权融合所有 $ s_k $ 得到统一的跨语义解释向量 $ c = \sum_k w_k \cdot s_k $。

### 相比现有方法的优势
| 维度 | HiSE 的优势 |
|------|-----------|
| **保真性 (Fidelity)** | 更准确地还原 HGNN 的分层决策路径，解释更贴近模型内部机制 |
| **鲁棒性 (Robustness)** | 不依赖输入扰动，避免破坏原始数据分布，在噪声下表现稳定 |
| **效率 (Efficiency)** | 无需迭代搜索或组合优化，运行速度快 2–3 个数量级 |
| **可解释性深度** | 同时提供语义内（per-semantic）和语义间（cross-semantic）双层解释 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 |
|-------|------|
| **ACM** | 学术网络，包含 Paper、Author、Subject 节点；任务：论文分类（3类）；特征为关键词词袋 |
| **MAG** | 来源于 `ogbn-mag` 的异构学术子图，包含 Paper、Author、Field-of-Study 节点；任务：论文分类；特征为 128 维 word2vec |

> ✅ 详细统计见附录 Table V：包括节点数、边数、元路径定义及密度等。

### 实验设置
- **目标模型（被解释的 HGNN）**：
  - **Meta-path-based**: HAN, MAGNN
  - **Relation-based**: HetSANN, HGT
- **噪声增强设置**：向原始特征添加 30% 随机噪声特征（二值或均匀分布），用于测试鲁棒性和可用性。
- **超参数**：K-hop 邻域大小、LASSO 正则化强度 $\lambda$ 等均经过调优。

### 评估指标
| 指标 | 定义 | 目标 |
|------|------|------|
| **Fidelity** | 用解释选出的 Top-K 特征重新训练模型后，预测一致性比例 | ↑ 越高越好 |
| **Robustness** | 在 Top-K 解释中选中的噪声特征数量 $N$ | ↓ 越低越好；辅以 Cliff’s delta 效应量分析 |
| **Usability** | 是否能区分“好模型”（准确率 ≥85%）与“坏模型”（准确率 ~70%）对噪声的依赖程度 | ↑ 成功率越高越好 |
| **Cross-Semantic Capability** | HiSE 推断的语义权重 vs. 原始 HGNN 提供的真实权重的 **Cosine Similarity (CS)**；分配给噪声子图的权重 **Noise Weight (NW)** | CS↑, NW↓ |
| **Computational Efficiency** | 平均每节点解释耗时（秒） | ↓ 越短越好 |

### 基线方法对比
分为三类：
1. **基础方法**：
   - Random, Greedy
2. **同质图解释器（适配后使用）**：
   - GraphLIME, GNNExplainer, ZORRO
3. **异质图专用解释器**：
   - HGExplainer, HENCE-X

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ Fidelity 结果（Table II）
- HiSE 在几乎所有设置下取得 **最高且最稳定** 的保真度。
- 示例：HAN on ACM ($K=20$) → HiSE 达 **0.89**，显著优于第二名 GNNExplainer (0.84)。
- 在 Relation-based 模型（如 HetSANN）上优势更为明显，说明其对复杂聚合机制更具适应性。

#### ✅ Robustness 结果（Fig. 4 & Fig. 7）
- HiSE 所选 Top-K 特征中噪声数量 **远低于其他方法**，尤其在 MAG 数据集上表现突出。
- **Cliff’s delta 分析显示**：HiSE 对噪声抑制能力极强（effect size 接近 -1.0），而 HENCE-X、HGExplainer 甚至不如随机选择（正效应量）。

#### ✅ Usability 结果（Fig. 5 & Fig. 6）
- HiSE 在识别高性能模型方面成功率 **持续高于 80%**，远超基线。
- GraphLIME 表现最差，接近随机水平，表明其难以捕捉模型质量差异。

#### ✅ Cross-Semantic Explanation 能力（Table III）
- **Cosine Similarity (CS)**：HiSE 与 HAN/MAGNN 提供的真实语义权重高度一致，**CS > 99%**（HAN 上）。
- **Noise Weight (NW)**：HiSE 分配给注入的噪声子图的总权重普遍 **< 5%**，证明其有效识别并抑制无关语义。

#### ✅ Computational Efficiency（Table IV）
- HiSE 运行时间 **全面领先**，平均仅需 **0.1~3.8 秒/节点**。
- 相比 HENCE-X 实现 **约 100–1000 倍加速**。
- 即使与轻量级方法相比，也快一个数量级以上（如比 GNNExplainer 快 10–30 倍）。

> 📌 总结：HiSE 在 **保真性、鲁棒性、可用性、跨语义解释能力和效率** 五大维度均显著优于现有方法。

---

## 4. 关键结论和发现

### 主要发现
1. **层级语义建模是提升解释质量的关键**  
   HiSE 通过模仿 HGNN 的两阶段结构（语义内聚合 + 跨语义融合），实现了与模型内在机制一致的解释，从而获得更高保真性和稳定性。

2. **非扰动式代理建模优于扰动策略**  
   基于 LASSO 的局部线性拟合避免了特征扰动带来的分布偏移问题，在噪声环境中表现出更强鲁棒性。

3. **KL 散度能有效反演语义贡献权重**  
   利用单语义子图与全图输出分布的一致性来推断语义重要性，是一种简单但高效的方法，能精准恢复原始模型的注意力偏好。

4. **轻量化设计带来巨大效率优势**  
   分解为独立的 per-semantic 建模 + 轻量级聚合，使得 HiSE 可扩展至大规模异构图场景。

### 方法的局限性
- 当前依赖预定义的 **meta-path 或 attention head** 作为语义划分依据，若这些先验不合理，则会影响解释效果。
- 尚未支持动态或自适应语义划分机制。
- 主要关注 **node classification** 场景，对 link prediction 或 graph-level 任务的支持有待验证。

### 未来工作方向
- 设计 **细粒度或自适应的语义分割机制**，减少对人工定义 meta-path 的依赖。
- 探索 **跨层语义交互建模**，进一步揭示不同层级间的语义传递规律。
- 扩展至更多类型的 HGNN 架构和下游任务（如推荐系统、知识图谱补全）。
- 结合因果推理框架，增强解释的因果可信度。

--- 

> 💡 **总体评价**：HiSE 是首个明确提出“**层级语义解释范式**”的工作，不仅解决了现有 HGNN 解释方法的结构性失配问题，还通过简洁高效的架构实现了性能与速度的双重突破，为 XAI 在复杂异构图上的应用提供了新思路。

</details>

---

### 14. [FOLD: Fuzzy Online Deduplication for Very Large Evolving Datasets via Approximate Nearest Neighbor Search](https://arxiv.org/abs/2606.03001)

**Authors**: Nelson Bore, Pritish Mishra, Constantin Adam, Eyal de Lara, Oana Balmau  
**Category**: cs.DC  
**Published**: 2026-06-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.03001v1  

#### Abstract
Fuzzy deduplication is key to constructing large language model training corpora. However, classic Locality-Sensitive Hashing pipelines scale poorly as corpora grow and are ill-suited to continuous ingestion. We present FOLD (Fuzzy Online Deduplication), an online fuzzy deduplication system that del...

---

### 15. [BlobShuffle: Cost-Effective Repartitioning in Stream Processing Systems via Object Storage Exemplified with Kafka Streams](https://arxiv.org/abs/2606.03364)

**Authors**: S\"oren Henning, Otmar Ertl, Adriano Vogel  
**Category**: cs.DC  
**Published**: 2026-06-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.03364v1  

#### Abstract
Shuffling or repartitioning data streams is an essential operation of state-of-the-art stream processing frameworks to support stateful workloads in a large-scale, distributed setting. In today's cloud deployments, however, shuffling can become a major cost driver due to substantial network traffic ...

---

### 16. [MOSAIC: Efficient Mixture-of-Agent Scheduling via Adaptive Aggregation and Inference Concurrency](https://arxiv.org/abs/2606.03014)

**Authors**: Saptarshi Mitra, Yifan Zhang, Rachid Karami, Phyo Pyae Moe Aung, Nazmul Takbir, Sreetama Sarkar, Souvik Kundu, Sitao Huang  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.03014v1  

#### Abstract
Mixture-of-Agents (MoA) systems improve reasoning accuracy by routing each query to multiple expert LLMs and aggregating their outputs. Efficiently executing this workload on limited GPU resources has bottlenecks. Skill-based routing creates skewed expert demand, and combining instruction-tuned LLMs...

---

### 17. [Traj-Evolve: A Self-Evolving Multi-Agent System for Patient Trajectory Modeling in Lung Cancer Early Detection](https://arxiv.org/abs/2606.02812)

**Authors**: Sihang Zeng, Matthew Thompson, Ruth Etzioni, Meliha Yetisgen  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.02812v1  

#### Abstract
Modeling patient trajectories from longitudinal electronic health records (EHRs) requires reasoning over sparse, noisy, and long-context multimodal sequences. Existing LLM-based multi-agent systems address context length but process patients in isolation, failing to mirror how clinicians leverage ac...

---

### 18. [Don't Gamble, GAMBLe: An Analytical Framework for AI-Driven Research Systems](https://arxiv.org/abs/2606.02863)

**Authors**: Marquita Ellis, Paul Castro  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.02863v1  

#### Abstract
AI-Driven Research Systems (ADRS) -- systems coupling LLMs with automated evaluation to discover algorithms, proofs, and designs -- are being optimized and adopted across domains, but the tools to analyze them have not kept pace. ADRS performance depends on component interactions that are poorly und...

---

### 19. [WISE-HAR: A Generalizable Ensemble Deep Learning Framework for WiFi-Based Human Activity Recognition](https://arxiv.org/abs/2606.02974)

**Authors**: Maheen Arshad, Qindeel E Zahra, Muhammad Khuram Shahzad  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.02974v1  

#### Abstract
Human Activity Recognition (HAR) using WiFi signals has emerged as a transformative technology for smart homes, healthcare monitoring, security systems, and ambient assisted living. Unlike traditional camera-based systems that raise significant privacy concerns and fail in low-light conditions, or w...

---

### 20. [TriEval: A Resource-Efficient Pipeline for LLM Bias, Toxicity, and Truthfulness Assessment](https://arxiv.org/abs/2606.03036)

**Authors**: Akshatha Srikantha, Manpreet Singh, Yash Jajoo, Shyamal Lakhanpal  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.03036v1  

#### Abstract
LLMs have evolved from basic chatbots to the backbone of the AI ecosystem, now widely used in healthcare, schools, and government services. The domain-wide adoption of LLMs necessitates continuous evaluation to ensure their safety and fairness. Common issues encountered after deploying LLMs include ...

---

### 21. [What Makes Interaction Trajectories Effective for Training Terminal Agents?](https://arxiv.org/abs/2606.03461)

**Authors**: Sidi Yang, Chaofan Tao, Jierun Chen, Tiezheng Yu, Ruoyu Wang, Yuxin Jiang, Yiming Du, Wendong Xu, Jing Xiong, Taiqiang Wu, Lifeng Shang, Xiaohui Li, Ngai Wong, Haoli Bai  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.03461v1  

#### Abstract
Stronger code agents are commonly assumed to be superior teachers for post-training, yet this assumption remains poorly disentangled from task difficulty, harness design, and student capacity. We investigate this pedagogical link using Terminal-Lego, a scalable pipeline that transforms multi-domain ...

---

### 22. [DMF: A Deterministic Memory Framework for Conversational AI Agents](https://arxiv.org/abs/2606.03463)

**Authors**: Matteo Stabile, Enrico Zimuel  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.03463v1  

#### Abstract
Conversational AI agents require memory systems that are both scalable and semantically coherent across long interaction horizons. Existing approaches rely predominantly on large language model (LLM)-based summarisation at write time, which introduces non-determinism, escalating token costs, and opa...

---

### 23. [From Prompt to Service: An SLM-Based Agent Orchestration Gateway for AI-Driven Virtual Worlds](https://arxiv.org/abs/2606.03557)

**Authors**: Louis Nisiotis, Aimilios Hadjiliasi  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.03557v1  

#### Abstract
As generative AI capabilities expand, AI-driven virtual worlds face a growing architectural challenge. Users interact through in-world interfaces in multimodal ways, yet their requests demand fundamentally different AI backend models and computational resources. Embedding these capabilities directly...

---

### 24. [The Ghost Annotator: a Framework to Explore Human Label Variation in Content Moderation through Conformal Prediction](https://arxiv.org/abs/2606.02911)

**Authors**: Mirko Lai, Alessandra Urbinati, Simona Frenda, Fabiana Vernero, Marco Antonio Stranisci  
**Category**: cs.CL  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.02911v1  

#### Abstract
Current research primarily focuses on model performance, while comparatively less attention has been devoted to uncertainty estimation, particularly in settings where LLMs are increasingly used to generate annotated data. We introduce a framework combining conformal prediction with Collaborative Fil...

---

### 25. [Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility](https://arxiv.org/abs/2606.03291)

**Authors**: Chaoyi Xiang, Olga Ohrimenko, Benjamin I. P. Rubinstein, Lea Frermann  
**Category**: cs.CL  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.03291v1  

#### Abstract
Large language models (LLMs) can memorize sensitive facts, motivating unlearning methods that remove targeted knowledge without costly retraining. However, unlearning research remains heavily English-centric. We study multilingual unlearning by extending the TOFU benchmark to five languages, and fin...

---

### 26. [HybridThinker: Efficient Chain-of-Thought Reasoning via Compressed Memory and Transient Thought Steps](https://arxiv.org/abs/2606.03768)

**Authors**: Xin Liu, Runsong Zhao, Xinyu Liu, Junhao Ruan, Pengcheng Huang, Shichao Dong, Chunyang Xiao, Chenglong Wang, Changliang Li, Jingbo Zhu, Tong Xiao  
**Category**: cs.CL  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.03768v1  

#### Abstract
Extended chain-of-thought (CoT) traces improve LLM reasoning but incur substantial computational and memory costs. While existing CoT compression methods mitigate this by condensing thought steps into compact representations via memory tokens and retaining only these representations at inference tim...

---

### 27. [Pruning Deep Neural Networks via the Marchenko--Pastur Distribution](https://arxiv.org/abs/2606.02608)

**Authors**: Leonid Berlyand, Theo Bourdais, Houman Owhad, Yitzchak Shmalo  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.02608v1  

#### Abstract
We study a Marchenko--Pastur (MP) random-matrix approach to pruning deep neural networks with very small post-pruning fine-tuning budgets. The main practical contribution is accuracy retention under short calibration and fine-tuning schedules, rather than a long post-pruning reoptimization pipeline....

---

### 28. [RRISE: Robust Radius Inference via a Surrogate Estimator](https://arxiv.org/abs/2606.02876)

**Authors**: Jong-Ik Park, Shreyas Chaudhari, Carlee Joe-Wong, Jos\'e M. F. Moura  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.02876v1  

#### Abstract
Randomized smoothing (RS) uses a smoothed classifier to provide architecture-agnostic certificates of $\ell_2$ classification robustness, but its dependence on per-input Monte Carlo (MC) sampling undermines its use in real-time systems. We argue that this cost is structural rather than fundamental, ...

---

### 29. [Learning Temporal Causal Structure via Smooth Differentiable Optimization](https://arxiv.org/abs/2606.03227)

**Authors**: Tong Zhao, Ce Guo, Wayne Luk, Emil Lupu, Ray Dipojjwal  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.03227v1  

#### Abstract
Causal discovery with instantaneous effects in multivariate time series is challenging, as the instantaneous structure must be acyclic. Prior methods enforce this by either separating instantaneous and lagged estimation into multi-stage pipelines or imposing algebraic acyclicity constraints via comp...

---

### 30. [Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](https://arxiv.org/abs/2606.03698)

**Authors**: Sangeun Park, Minhae Kwon  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.03698v1  

#### Abstract
A central goal of large language model (LLM) research is to build agentic systems that can plan, act, and adapt through sustained interaction with dynamic environments. While recent LLM-based agents exhibit impressive contextual reasoning, their long-horizon decision-making remains fragile, often su...

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

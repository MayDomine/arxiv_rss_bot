# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-21 08:54:13 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs](https://arxiv.org/abs/2605.20315)

**Authors**: Haiquan Lu, Zigeng Chen, Gongfan Fang, Xinyin Ma, Xinchao Wang  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2605.20315v1  

#### Abstract
LLM agents have recently emerged as a powerful paradigm for solving complex tasks through planning, tool use, memory retrieval, and multi-step interaction. However, these agentic workflows often introduce substantial input-side overhead, making the compute-intensive prefilling stage a key bottleneck...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLM）代理（Agentic LLMs）在执行复杂任务时（如工具调用、记忆检索、多步交互），通常需要反复处理**长上下文输入**。这导致推理过程中的 **prefilling 阶段成为计算瓶颈**，尤其是在输入远长于输出的场景下。

然而，现有的统一量化策略（如对整个推理流程应用 W4A4 量化）虽然能加速 prefilling，但会因**解码阶段的误差累积**（error accumulation）严重损害生成质量，尤其在多轮、长轨迹的 agentic 任务中表现不稳定。

### 🚀 提出的新方法：Mix-Quant
作者提出 **Mix-Quant** —— 一种**相位感知**（phase-aware）的混合精度量化框架，其核心思想是：

> **“量化 prefilling，保留 decoding”**  
> 即：对计算密集型的 **prefilling 阶段使用高吞吐的 NVFP4 量化**，而对误差敏感的 **autoregressive decoding 阶段保持 BF16 高精度**。

该方法实现了：
- **算法层面**：解耦 prefilling 加速与 decoding 质量控制；
- **硬件层面**：利用 NVIDIA Blackwell 架构原生支持的 **NVFP4** 格式实现高效低比特计算。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | Mix-Quant 的改进 |
|------|-------|------------------|
| **Uniform Quantization**（如全模型 W4A4） | 解码误差累积，任务性能显著下降 | 仅 prefilling 量化，避免生成路径扰动 |
| **Weight-only Quantization**（如 GPTQ, AWQ） | 仅优化内存带宽，对 compute-bound 的 prefilling 加速有限 | 引入 **weight-and-activation quantization**，直接降低 prefilling 计算成本 |
| **通用 Prefill-Decode 分离架构**（如 DistServe） | 缺乏对不同阶段精度需求的建模 | 与之兼容且互补，进一步引入**相位感知量化策略** |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验覆盖三类典型任务，全面评估 agentic 推理能力：

#### （1）**Agentic Benchmarks**（代理任务）
- **BFCL v4**：评估工具调用与函数调用能力
- **LongMemEval**：测试长期交互记忆管理
- **T2-bench**：评估状态化对话中的 agent 行为

#### （2）**Long-Context Benchmarks**（长上下文理解）
- **LongBench-V2**
- **AA-LCR**

#### （3）**Reasoning Benchmarks**（数学推理）
- **MATH500**
- **AIME24 / AIME25**

> 所有任务均涉及长输入、多轮交互或复杂推理，符合 agentic workflow 特征。

---

### ⚙️ 实验设置与评估指标

| 项目 | 设置说明 |
|------|----------|
| **模型** | Qwen3-8B, Qwen3.5-9B, Gemma-4-26B-A4B-it, Gemma-4-31B-it |
| **上下文长度** | 最长达 256K–262K tokens（部分通过 YaRN 扩展） |
| **硬件平台** | NVIDIA RTX 5090 和 B200 GPU（支持 Blackwell NVFP4 加速） |
| **推理框架** | vLLM + FlashInfer + NIXL-based KV-cache transfer |
| **部署方式** | Prefill-Decode disaggregation：prefill worker 使用 NVFP4，decode worker 使用 BF16 |
| **评估指标** | 任务准确率（Accuracy）、平均得分（Avg. Score）、端到端 prefill 延迟加速比 |

---

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **BF16** | 原始高精度模型，作为性能上限 |
| **Uniform NVFP4** | 全流程 W4A4 量化，作为效率上限但质量受损基线 |
| **P16D4**（Prefill-BF16, Decode-NVFP4） | 反向策略：保留 prefill 精度，量化 decoding |
| **Mix-Quant**（本文方法） | Prefill-NVFP4, Decode-BF16 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### （1）**Agentic Benchmark 性能对比**（Table 1）
| Model | Avg. Score (BF16) | Uniform NVFP4 | **Mix-Quant** |
|-------|--------------------|----------------|----------------|
| Qwen3-8B | 42.85 | 38.64 (-4.21) | **41.45** (-1.40) |
| Qwen3.5-9B | 77.31 | 70.37 (-6.94) | **74.68** (-2.63) |
| Gemma-4-26B-A4B-it | 66.07 | 55.95 (-10.12) | **61.67** (-4.40) |
| Gemma-4-31B-it | 77.63 | 76.21 (-1.42) | **77.14** (-0.49) |

✅ **结论**：Mix-Quant 显著恢复了 uniform NVFP4 的性能损失，接近甚至逼近 BF16 水平。

---

#### （2）**Reasoning & Long-Context Performance**（Table 2）
| Model | BF16 Avg. | Uniform NVFP4 | **Mix-Quant** |
|-------|-----------|----------------|----------------|
| Qwen3.5-9B | 72.04 | 63.26 (-8.78) | **70.59** (-1.45) |
| Gemma-4-26B-A4B-it | 71.94 | 66.31 (-5.63) | **71.93** (-0.01) ✅ |
| Gemma-4-31B-it | 82.36 | 78.26 (-4.10) | **81.39** (-0.97) |

✅ **结论**：Mix-Quant 在数学推理和长文本理解任务上同样有效，验证其泛化能力。

---

#### （3）**Prefilling 阶段加速效果**（Figure 4）
- 在 RTX 5090 上，Mix-Quant 实现：
  - **2–3× 的 prefill 延迟加速**
  - 最高达 **3.74× speedup**（Qwen3-8B, seq len=32K）
- 加速效果随序列长度增加而提升，表明对长上下文更友好。

---

#### （4）**消融实验**（Ablation Study, Table 3）
比较三种 phase-wise 策略：

| 策略 | Qwen3-8B Avg. | Gemma-4-26B Avg. |
|------|---------------|------------------|
| Uniform NVFP4 | 33.59 | 53.34 |
| P16D4（Decode 量化） | 36.74 | 59.85 |
| **Mix-Quant**（Prefill 量化） | **38.32** | **60.18** |

🔍 **发现**：
- 量化 **decoding** 比量化 **prefilling** 更有害；
- Mix-Quant 明显优于 P16D4，说明 **decoding 对误差更敏感**；
- 尽管 prefill 量化也会扰动 KV Cache，但由于注意力集中现象（attention mass concentration），影响可控。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Agentic workflows 是 input-heavy 的**：输入 token 数量远超输出，使得 prefilling 成为主要瓶颈。
2. **Prefilling 具有量化冗余性**：长上下文中存在大量低注意力 token，量化误差被加权衰减，因此可安全压缩。
3. **Decoding 极其敏感于量化误差**：token 级错误会通过自回归机制传播，引发“雪球效应”（snowball effect），破坏任务完成路径。
4. **Mix-Quant 实现了最优权衡**：在 prefilling 阶段获得近 3× 加速的同时，几乎完全保留了 BF16 的任务性能。

> 如 Figure 1 所示：Mix-Quant 在速度与精度之间找到了理想平衡点。

---

### ⚠️ 方法的局限性
1. **依赖硬件支持**：当前依赖 NVIDIA Blackwell 架构的 **NVFP4 原生支持**，在其他硬件上难以复现同等加速比。
2. **KV Cache 类型需对齐**：prefill worker 输出的 KV Cache 必须与 decode worker 兼容，可能引入额外转换开销（尽管文中通过设计规避）。
3. **不适用于极短上下文场景**：若 prefilling 本身不占主导，则加速收益有限。

---

### 🔮 未来工作方向
1. **扩展至更多量化格式**：探索 INT4、MXFP4 等其他低比特格式在 phase-aware 框架下的表现。
2. **动态相位切换策略**：根据上下文长度、任务类型自动决定是否启用 Mix-Quant。
3. **结合稀疏注意力优化**：将 Mix-Quant 与 FlashPrefill、Minference 等动态稀疏 attention 方法结合，进一步降低 prefill 成本。
4. **跨设备协同部署**：在 heterogeneous edge-cloud 场景中部署 Mix-Quant，实现能效与延迟的联合优化。

---

## ✅ 总结
**Mix-Quant** 是首个明确提出并验证 **“prefilling 可激进量化，decoding 应保持高精度”** 的相位感知量化框架。它不仅解决了 agentic LLM 推理中的核心效率瓶颈，还揭示了不同推理阶段的本质差异，为未来的 **algorithm-hardware co-design** 提供了重要范式。

> **代码已开源**：[https://github.com/haiquanlu/Mix-Quant](https://github.com/haiquanlu/Mix-Quant)

</details>

---

### 2. [Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization](https://arxiv.org/abs/2605.20295)

**Authors**: Jinghe Zhang, Daliang Xu, Chenghua Wang, Weikai Xie, Tao Qi, Yun Ma, Mengwei Xu, Gang Huang  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2605.20295v1  

#### Abstract
Large language models (LLMs) are increasingly deployed on mobile devices, where Neural Processing Units (NPUs) necessitate fully static quantization for optimal inference efficiency. However, existing post-training quantization (PTQ) methods predominantly rely on dynamic activation quantization, ren...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Post-Training Quantization (PTQ)** 方法大多基于 **dynamic activation quantization**（动态激活量化），依赖运行时重新计算量化参数（如 min/max）。然而，移动设备上的 **Neural Processing Units (NPUs)** 为了实现高效推理，通常要求 **fully static quantization**（全静态量化）——即所有量化参数在编译前固定，以避免动态计算带来的高开销。

因此，现有 PTQ 方法虽然精度高，但无法直接部署到 NPU 上，导致从训练到部署存在严重不匹配，造成显著的 **accuracy collapse**（精度崩溃）。

---

### 🚀 提出的新方法：Quant.npu
Quant.npu 是一个专为移动 NPU 设计的 **integer-only 全静态量化框架**，其核心思想是将学习型量化参数与旋转矩阵联合优化，并通过系统级设计确保端到端兼容性。

#### 主要创新点：
1. **Rotation-and-bit-width-aware Initialization**  
   - 针对不同类型的激活分布（rotated vs. unrotated）和目标 bit-width，采用不同的初始化策略：
     - 对于经过旋转的平滑分布（Gaussian-like），使用 **Mean-based 初始化**，提升低比特下的量化精度。
     - 对于未旋转的重尾分布（heavy-tailed），使用 **Max-Min 初始化** 并配合更高 bit-width（8/16-bit），保证动态范围覆盖。
   - 这种感知旋转与比特宽度的初始化显著提升了优化稳定性。

2. **Distribution-aware Selective Optimization（两阶段量化流水线）**
   - 将量化参数分为两类进行差异化处理：
     - **Stage One（梯度优化）**：仅对输入激活、权重等“易优化”的 rotated 分布联合优化量化参数和旋转矩阵。
     - **Stage Two（静态校准）**：对输出激活、KV Cache 等难优化且敏感的 unrotated 分布直接使用静态校准，避免不稳定梯度影响整体收敛。
   - 有效解耦复杂性，提升优化效率与模型性能。

3. **Sensitivity-guided Adaptive Mixed-Precision Scheme**
   - 引入基于 **relative quantization error** 的敏感度指标：
     $$
     \text{ratio} = \frac{\sum |\text{dequantized} - \text{original}|}{\sum |\text{original}| + 10^{-8}}
     $$
   - 自适应地将部分高度敏感的 `down_proj` 输入激活提升至 16-bit，其余保持 8-bit。
   - 在极小代价下恢复因移除在线旋转（R4）造成的精度损失。

4. **硬件友好设计**
   - 仅引入两个可离线融合的旋转矩阵 R1 和 R2，消除运行时浮点开销。
   - 采用 **per-tensor activation quantization** 和 **per-channel weight quantization**，契合 NPU 的 systolic array 架构，最大化吞吐量。

---

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **部署兼容性** | 完全静态量化，无需运行时动态操作，完美适配 NPU 编译约束 |
| **推理效率** | 整数运算为主，减少内存带宽和计算延迟；相比 per-block 更快 |
| **精度表现** | 在同等延迟下优于 SOTA 方法，在 W4A8 设置下接近 FP32 性能 |
| **通用性** | 支持多种模型架构（LLaMA、Qwen、SmolLM）、规模（1B–8B） |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **校准数据集**：WikiText-2（用于量化参数初始化与校准）
- **评估数据集**：
  - **语言建模任务**：C4（评估 Perplexity, PPL）
  - **零样本推理任务**：PIQA、Winogrande、HellaSwag、ARC-E、ARC-C、LAMBADA
  - **指令跟随能力**：AlpacaEval 2.0
  - **真实场景负载测试**：HellaSwag、Persona-Chat、DroidTask（用于端到端延迟测量）

---

### ⚙️ 实验设置
- **模型**：
  - 主要：Llama-3.2-3B-Instruct
  - 扩展验证：Qwen2.5-3B-Instruct、SmolLM2-1.7B-Instruct、Qwen3-1.7B、Llama3-8B
- **量化配置**：
  - 权重：4-bit / 8-bit per-channel
  - 激活：8-bit / 16-bit per-tensor
  - KV Cache 固定为 8-bit，SiLU 等非线性函数保留 16-bit
- **训练细节**：
  - 使用 4×NVIDIA A40 GPU
  - 优化器：SGD + cosine decay
  - 学习率：rotation matrices 0.1，quantization parameters 0.01
  - 局部量化误差损失（local error loss）用于前 128 步稳定训练

---

### 🆚 基线方法对比
| 方法 | 类型 | 是否支持静态量化 | 特点 |
|------|------|------------------|------|
| **ExecuTorch** | 工业级框架 | 是（但 per-block 动态） | 默认支持 W4A16 per-block，延迟较高 |
| **QuaRot** | 旋转法（随机矩阵） | 否（假设动态量化） | 不适用于 NPU 静态部署 |
| **SpinQuant** | 可学习旋转 + 动态量化 | 否 | 优化时模拟动态量化，转静态后性能骤降 |
| **MobileQuant** | 移动端专用量化 | 是 | 基于 SmoothQuant，精度较弱 |

> 所有方法均在 ExecuTorch 框架上统一实现，仅替换量化模块，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Llama-3.2-3B-Instruct）

| 方法 | W-A-KV | PPL (C4) ↓ | Avg Accuracy ↑ | 推理延迟（SM8650） |
|------|--------|------------|----------------|--------------------|
| FP32 | — | 16.85 | 0.6130 | — |
| ExecuTorch (W4A16) | 4-16-8 | 18.19 | 0.5935 | 基准 |
| SpinQuant (W4A8) | 4-8-8 | 28.78 | 0.4623 | 较慢 |
| **Quant.npu (W4A8)** | **4-8-8** | **19.16** | **0.5827** | **最高（提速 up to 15.1%）** |
| **Quant.npu (W8A8)** | **8-8-8** | **16.42** | **0.6165** | 接近 FP32，精度反超 |

> ✅ **Quant.npu-W4A8 相比 SpinQuant 提升 12.04% 准确率，PPL 下降 9.62**

---

### 🔁 与 ExecuTorch-W4A16 的对比
尽管 Quant.npu 使用更低精度（W4A8 vs W4A16），但由于更优的静态量化设计：
- **平均准确率差距仅 2.58%**
- **PPL 仅增加 1.23**
- **推理速度提升高达 15.1%**

👉 表明 Quant.npu 成功实现了 **accuracy-efficiency trade-off 的突破**。

---

### 🔍 消融实验结果（Ablation Study）

| 组件 | PPL (C4) ↓ | Avg Acc ↑ | 贡献说明 |
|------|-----------|-----------|----------|
| Baseline（joint opt） | 69.65 | 0.2969 | 初始不稳定 |
| + Rotation-aware Init | 36.42 | 0.3796 | 显著加速收敛 |
| + Selective Opt（Two-stage） | 30.89 | 0.4015 | 解耦优化提升鲁棒性 |
| + Adaptive Mixed-Precision | **22.09** | **0.4733** | 恢复 outlier 敏感层精度 |

> ✅ 最终方案已逼近 FP32 性能（Avg: 0.4939），证明各组件协同增效。

---

### 📈 自适应混合精度效果（Figure 4）
- 仅将 `down_proj` 中 **top 10% 最敏感的激活** 升级为 16-bit，
- 即可使 PPL 从 30.89 降至 22.09，
- 相当于 **全 16-bit 的 90% 效果，但成本大幅降低**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **全静态量化中的优化稳定性高度依赖初始化与选择性优化**  
   - 不恰当的初始化会导致优化陷入局部最优或发散。
   - 联合优化所有量化参数反而会破坏收敛，尤其是对于 unrotated heavy-tailed 分布。

2. **旋转矩阵与量化参数必须联合优化才能对齐训练与部署**  
   - 若先固定量化参数再优化旋转，或反之，都会导致性能下降。
   - Quant.npu 通过 STE + gradient scaling 实现端到端联合优化。

3. **硬件约束驱动算法设计至关重要**  
   - 移除 R3/R4 在线旋转虽牺牲一定平滑性，但换来完全整数流水线，总体收益更大。
   - 自适应混合精度精准补偿关键瓶颈，实现“花小钱办大事”。

4. **Quant.npu 具备强泛化能力**
   - 在 Qwen、SmolLM、TinyLlama 等不同架构上均取得 SOTA 表现。
   - 支持从 1.7B 到 8B 的大模型，扩展性强。

---

### ⚠️ 方法的局限性
1. **仍需保留部分 16-bit 操作**  
   - 如 SiLU、Gate Projection 输出等非线性层仍用 16-bit，限制了完全低比特执行。
   - 未来需探索更激进的低比特非线性近似。

2. **缺乏对校准数据集的主动选择机制**  
   - 当前依赖 WikiText-2，若 domain shift 明显可能影响性能。
   - 可研究基于敏感度的数据采样策略。

3. **未支持 ultra-low bit (<4-bit) 量化**
   - 当前最低为 W4A8，尚不能满足极端资源受限场景需求。

---

### 🔮 未来工作方向
- 开发 **fully low-bit execution pipeline**，消除所有 16-bit 残留。
- 设计 **principled calibration set construction** 方法，提升跨域鲁棒性。
- 探索 **quantization + pruning + sparsity** 联合压缩方案。
- 将 Quant.npu 扩展至 **vision-language models** 和 **on-device fine-tuning** 场景。

---

## ✅ 总结
**Quant.npu** 是首个真正意义上兼顾 **高精度、高效率、高兼容性** 的移动端 LLM 静态量化框架。它通过 **算法-系统协同设计**，解决了动态量化与 NPU 部署之间的根本矛盾，在真实 NPU 上实现了 **SOTA 精度 + 最高推理速度** 的双重突破，为推动 **on-device LLMs** 的普及提供了坚实的技术基础。

</details>

---

### 3. [torchtune: PyTorch native post-training library](https://arxiv.org/abs/2605.21442)

**Authors**: Mark Obozov, Maxime Griot, Joseph Cummings, Evan Smothers, Felipe Mello, Rafi Ayub, Philip John Bontrager, Salman Mohammadi, Ariel Kwiatkowski, Nathan Azrak, Mircea Mironenco  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.21442v1  

#### Abstract
Modern LLMs typically require multistage training pipelines to achieve strong downstream performance, with post-training serving as the main interface for adapting open-weight models. We introduce torchtune, a PyTorch-native library designed to streamline the post-training lifecycle of LLMs, enablin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*torchtune: PyTorch native post-training library*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大型语言模型（LLMs）的**post-training**（后训练）阶段——包括监督微调（SFT）、偏好优化（DPO）、知识蒸馏、强化学习对齐等——已成为适配开源模型到下游任务的关键环节。然而，现有的微调框架存在以下问题：

- **依赖复杂**：基于 `transformers` 的框架继承了庞大的依赖栈，影响部署、调试和可复现性。
- **耦合性强**：模型构建、训练逻辑、分布式策略、适配器插入等被封装在抽象层中，难以进行细粒度修改。
- **性能路径不透明**：通用实现未能充分利用 PyTorch 的最新性能特性（如 FSDP2、DTensor、`torch.compile`）。
- **多任务支持碎片化**：SFT、DPO、PPO、KD 等任务分散在不同库或模块中，难以统一比较。
- **分布式组合性差**：多节点、张量并行、上下文并行等支持不一致。

### 提出的新方法与思路
提出 **torchtune** ——一个**原生 PyTorch 的 post-training 库**，其设计围绕以下核心理念：

- **模块化组件架构**：将模型、数据、目标函数、优化器、日志等解耦为独立可替换的组件。
- **YAML 驱动的 Recipe 系统**：受 Hydra 启发，通过配置文件定义训练流程，支持命令行覆盖，便于消融实验。
- **轻量级、高透明度**：不引入新的训练抽象，保持代码贴近实际执行的 PyTorch 逻辑，提升“hackability”（可修改性）。

### 相比现有方法的优势
| 特性 | torchtune | Axolotl / HuggingFace | Unsloth |
|------|---------|------------------------|---------|
| **透明性** | ⭐⭐⭐⭐⭐（直接访问 PyTorch 模块） | ⭐⭐（高度抽象） | ⭐（内核黑盒） |
| **灵活性** | ⭐⭐⭐⭐⭐（组件自由替换） | ⭐⭐⭐（模板驱动） | ⭐⭐（专注 LoRA/QLoRA） |
| **性能效率** | ⭐⭐⭐⭐（融合优化 + 编译） | ⭐⭐ | ⭐⭐⭐⭐⭐（定制 CUDA 内核） |
| **扩展性** | ⭐⭐⭐⭐⭐（支持 MoE、多模态、RL） | ⭐⭐⭐ | ⭐⭐ |

> torchtune 定位介于高层自动化框架（如 Lightning）和底层高性能内核（如 Unsloth）之间，强调**可复现研究**与**工程效率**的平衡。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Alpaca**：用于 SFT 和常规微调任务的标准指令数据集。
- **Anthropic’s helpful/harmless RLHF 数据子集**：用于 DPO 偏好优化任务。
- **合成长序列数据集**：用于测试 Context Parallelism，由 Alpaca 样本拼接而成，最长达百万 token。

### 实验设置
- **硬件环境**：
  - 单卡：单个 H100 GPU
  - 多卡：8×H100（单节点），使用 FSDP2 + Tensor Parallelism + Loss Parallelism
- **模型规模**：
  - Qwen3 系列：0.6B ~ 32B
  - Llama3.1 / Llama3.3：8B ~ 70B
- **超参数**：
  - 序列长度：2048
  - 微批次大小（micro-batch size）：2（Optim Bwd 时为 16）
  - 梯度累积步数：8（Optim Bwd 时为 1）

### 评估指标
- **内存占用（Memory）**：峰值 GPU 显存使用（GB/GPU）
- **吞吐量（Throughput）**：每秒处理 token 数（tokens/s/GPU）
- **可行性（Feasibility）**：是否发生 OOM（Out-of-Memory）

### 基线方法对比
- **Axolotl**：社区流行的 YAML 配置驱动微调框架
- **Unsloth**：专注于 LoRA/QLoRA 加速的高效框架（使用 Triton 内核）
- **Baseline torchtune**：逐步启用各项优化的对照组

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 2）

#### 表 1：多 GPU 设置下 Llama3.3 70B 与 Qwen3 32B 性能对比（8×H100）

| 方法 | Qwen3 32B 内存 (GB) | Qwen3 32B 吞吐 (tok/s) | Llama3.3 70B 内存 (GB) | Llama3.3 70B 吞吐 (tok/s) |
|------|---------------------|------------------------|------------------------|----------------------------|
| torchtune (baseline) | 67.78 | 465.79 | OOM | OOM |
| +AC（激活检查点） | 42.93 | 405.95 | 74.75 | 122.55 |
| +LCE（Linear CE） | 38.43 | 398.56 | OOM | OOM |
| +Compile | 44.17 | 433.12 | 75.22 | 128.57 |
| **+Optim Bwd** | **60.41** | **581.63** | **74.89** | **352.11** ✅ |
| Axolotl | 40.9 | 218 | OOM | OOM |

> 💡 **关键发现**：仅靠 `Optimizer in Backward` 就使 Llama3.3 70B 在 8×H100 上从 OOM 变为可行，并提升吞吐 **2.7×**！

#### 表 2：单 GPU 下 Qwen3 全参数微调与 LoRA 对比（1×H100）

| 方法 | Qwen3 8B 内存 (GB) | Qwen3 8B 吞吐 (tok/s) | LoRA 8B 内存 (GB) | LoRA 8B 吞吐 (tok/s) |
|------|--------------------|------------------------|-------------------|-----------------------|
| torchtune (全参) | OOM | OOM | 17.2 | 2745 |
| +AC | 64.04 | 3037 | – | – |
| **+Optim Bwd** | **51.79** | **3773** | – | – |
| **+AdamW8Bit** | **31.07** | 3066 | – | – |
| Axolotl | 76.6 | 1903 | 27.9 | 1609 |
| Unsloth | – | – | **16.8** | **1836** |

> 🔥 **亮点**：
> - torchtune 在 LoRA 场景下内存优于 Unsloth，吞吐更高。
> - 全参数微调下，`AC + Optim Bwd` 组合使原本 OOM 的 Qwen3 8B 成功运行。

#### 表 4：DPO 任务性能对比（单 GH200，96GB HBM3）

| 方法 | Llama3.1 8B 内存 (GB) | 吞吐 (tok/s) |
|------|------------------------|-------------|
| DPO (torchtune) | 81.82 | 745.0 |
| DPO (Axolotl) | OOM | – |
| DPO (Axolotl + 8-bit) | 67.64 | 249.2 |

> 🎯 **结论**：torchtune 能用标准 AdamW 成功运行 DPO，而 Axolotl 必须降级使用 8-bit 优化器才能避免 OOM。

---

## 4. 关键结论和发现

### 主要发现
1. **优化技术是互补的**：
   - `torch.compile` 是中小模型的主要**吞吐杠杆**。
   - **内存优化技术**（AC、Optim Bwd、AdamW8Bit）决定大模型能否运行。
   - `Linear Cross-Entropy` 有效降低损失计算时的峰值内存。

2. **in-backward optimizer fusion 是突破性技术**：
   - 将 optimizer step 融入 backward pass，显著缩短梯度生命周期。
   - 在 Llama3.3 70B 上实现了从 OOM 到 **352 tok/s** 的飞跃。
   - 是目前唯一能在 8×H100 上完成该规模全参微调的开源方案之一。

3. **模块化设计支持快速迭代**：
   - 通过 YAML 配置即可切换 LoRA、QLoRA、量化等策略，无需重写训练循环。
   - 支持 SFT、DPO、GRPO、KD、MoE、多模态等多种 post-training 范式。

4. **异步 GRPO 提升端到端利用率**：
   - 通过 Ray 构建生成（rollout）与训练的流水线，解耦 phase，提高 GPU 利用率。
   - 支持同步、准同步（bounded lag）等多种调度模式。

### 方法的局限性
- **不支持梯度累积（K > 1）下的 Optim Bwd**：因会提前触发参数更新。
- **对 ZeRO 类优化器集成有限**：Optim Bwd 与全局 optimizer step 假设有冲突。
- **编译开销较高**：首次启动时间较长，适合长时间训练任务。
- **当前生态仍较小**：相比 HuggingFace 生态，预训练权重和工具链仍在建设中。

### 未来工作方向
- 扩展对更多 MoE 架构的支持（如 DeepSeek-MoE）。
- 改进 Optim Bwd 对梯度累积的支持（例如通过延迟更新机制）。
- 增强与 Hugging Face Hub 的互操作性（自动转换权重格式）。
- 探索更高效的 long-context 训练策略（结合 Ring Attention 与 PagedAttention）。
- 开放更多 benchmark 和 recipe 示例，推动社区共建。

---

> ✅ **总体评价**：  
> **torchtune** 不是一个追求极致吞吐的“加速器”，而是一个面向**研究者**的**可复现、可扩展、可理解**的 post-training 基础设施。它填补了从“易用但封闭”到“高效但黑盒”之间的空白，有望成为 LLM 后训练领域的重要开源力量。

</details>

---

### 4. [Projecting Latent RL Actions: Towards Generalizable and Scalable Graph Combinatorial Optimization](https://arxiv.org/abs/2605.19721)

**Authors**: Franco Terranova (UL, LORIA, Inria), Guillermo Bernardez (UC Santa Barbara), Albert Cabellos-Aparicio (UPC), Nina Miolane (UC Santa Barbara), Abdelkader Lahmadi (LORIA, UL, Inria)  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.19721v1  

#### Abstract
Graph combinatorial optimization (GCO) has attracted growing interest, as many NP-hard problems naturally admit graph formulations, yet their combinatorial explosion renders exact methods computationally intractable. Recent advances in Reinforcement Learning (RL) combined with Graph Neural Networks ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Projecting Latent RL Actions: Towards Generalizable and Scalable Graph Combinatorial Optimization 总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**图组合优化（Graph Combinatorial Optimization, GCO）**中基于强化学习（Reinforcement Learning, RL）的方法所面临的两大挑战：

- **泛化能力差**：传统离散动作空间方法依赖于图实例特定的索引编码，难以在不同规模或分布的图之间迁移。
- **可扩展性低**：现有基于潜变量（latent action）的迭代式方法（如S2V-DQN）需要对每个候选动作进行独立的Q值评估，导致推理时间随动作空间大小线性甚至超线性增长，在大规模或复杂决策场景下不可行。

此外，现实世界中的GCO任务（如流量工程、网络安全路径预测）通常涉及**多变量、结构化的决策空间**（例如选择“边+权重”、“源节点+目标节点+漏洞”），其动作空间呈超线性增长，加剧了上述问题。

### 提出的新方法与思路
作者提出了一种名为 **Projection Agents** 的新型RL-GCO框架，其核心思想是将决策过程从离散动作选择转移到连续的**潜动作空间（latent action space）**中进行：

- **潜动作嵌入（Latent Action Embedding）**：利用图神经网络（GNN）为每一个可能的动作（如节点、边、路径、子图等）生成一个语义丰富的向量表示，构成一个连续且有几何结构的潜动作空间。
- **单次前向投影（Single Forward Pass Projection）**：策略网络（policy network）不再输出离散动作的概率分布，而是直接输出一个“期望”的潜动作向量（proto-action）。
- **最近邻解码（Nearest Neighbor Decoding）**：通过在预构建的潜动作空间中执行最近邻搜索（如k-NN），将连续的潜动作向量解码为一个有效的离散动作。

该方法结合了**端到端学习**与**高效检索**的优点，实现了“一次前向传播 + 快速检索”的决策范式。

### 相比现有方法的优势
| 特性 | 离散方法 (Discrete) | 迭代潜变量方法 (Iterative) | 投影代理 (Projection, 本文) |
| :--- | :--- | :--- | :--- |
| **推理速度** | 快（单次推理） | 慢（需遍历所有动作） | **极快**（单次推理 + 子线性检索） |
| **泛化能力** | 差（依赖索引） | 中等（语义嵌入） | **强**（共享嵌入空间，支持跨图迁移） |
| **动作空间支持** | 简单（节点级） | 复杂但效率低 | **复杂且高效**（支持结构化、超线性动作空间） |
| **计算复杂度** | O(1) 推理 | O(|A|) 推理 | O(1) 推理 + O(log \|A\|) 检索 |
| **支持插值** | 否 | 否 | **是**（潜空间连续，支持动作间平滑过渡） |

此外，作者还发布了 **LaGCO-RL** 开源库，自动化潜动作空间构建，并统一了观测与动作的嵌入空间，促进公平比较和复现。

---

## 2. 核心实验方法和设置

### 数据集（Benchmark Environments）
实验在七个GCO基准上进行，分为两类：
- **经典问题（线性动作空间）**：
  - **TSP** (Traveling Salesman Problem)
  - **MinVertex** (Minimum Vertex Cover)
  - **MaxCut**
- **应用驱动问题（超线性/结构化动作空间）**：
  - **Placement** (虚拟机放置)
  - **Cyber-Path** (网络攻击路径预测)
  - **OSPF** (OSPF路由权重调整)
  - **Traffic** (流量工程路径分配)

这些任务更贴近现实，动作空间复杂度高（如Traffic中为O(|V|²2^[])）。

### 实验设置与评估指标
- **训练策略**：采用四种迁移学习策略评估泛化能力：
  - **S/M/L**：在小/中/大图上训练，在其余图上测试。
  - **V (Varied)**：K折交叉式训练，使用多个不同图实例轮换训练。
- **评估方式**：每种配置重复5次，报告100个未见实例上的性能。
- **主要指标**：
  - **归一化得分（Normalized Score）**：将解的质量归一化到[0,1]，0为最差，1为最优。
  - **Interquartile Mean (IQM)**：取最好100个分数的中间四分位均值，作为最终性能指标。
  - **Train-Test Gap (△)**：衡量泛化能力，gap越小越好。
  - **推理延迟（Inference Latency）**：测量动作选择时间随图规模的增长趋势，拟合幂律模型 $T(n) = c \cdot n^\alpha$，指数 $\alpha$ 越小，可扩展性越好。

### 基线方法对比
- **P-Discrete / G-Discrete**：传统离散方法，分别使用填充（padding）和GNN池化作为观测编码。
- **P-Discrete-M / G-Discrete-M**：带动作掩码（action masking）的版本。
- **Iterative**：迭代式潜变量方法（类似S2V-DQN），对每个有效动作评估Q值。
- **Projection (Ours)**：本文提出的投影代理，使用PPO算法，k=1的最近邻解码。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比
#### 泛化性能（IQM 归一化得分）
| 方法 | TSP | MinVertex | MaxCut | Placement | Cyber-Path | OSPF | Traffic |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P-Discrete-M** | 0.48 | 0.50 | 0.90 | 0.65 | 0.67 | 0.73 | 0.78 |
| **Iterative** | **0.99** | 0.92 | 0.82 | 0.15 | 0.00 | 0.00 | 0.04 |
| **Projection (Ours)** | 0.71 | **0.95** | **0.94** | **0.91** | **0.88** | **0.83** | **0.83** |

- 在**经典问题**（TSP, MinVertex, MaxCut）上，`Projection` 表现优异，仅次于或接近 `Iterative`。
- 在**应用驱动问题**（Placement, Cyber-Path, OSPF, Traffic）上，`Projection` **全面大幅领先**，性能远超 `Iterative` 和离散方法，最高提升达 **40%**。

#### 可扩展性（推理延迟幂律指数 α）
| 方法 | TSP | Placement | Cyber-Path | Traffic |
| :--- | :--- | :--- | :--- | :--- |
| **Iterative** | 0.25 | 1.81 | 1.46 | **4.68** |
| **Projection (Ours)** | **0.17** | **0.22** | **0.09** | **0.73** |

- `Projection` 的推理延迟增长指数 **显著低于** `Iterative`，尤其在超线性动作空间（如Traffic, α=0.73 vs 4.68）上优势巨大。
- 推理速度最高可达 `Iterative` 方法的 **16.2倍**。

### 消融实验与分析
- **训练策略影响**：在应用问题上，“Varied (V)”训练策略效果最佳，表明暴露于多样化的图实例有助于提升泛化。
- **嵌入空间分析**：UMAP可视化显示，真实世界任务的动作空间更密集、重叠更多，对精确解码要求更高，而`Projection`仍能保持高性能。
- **解码策略**：即使使用最简单的k=1最近邻解码，`Projection` 也能取得优越性能，证明了其鲁棒性。

---

## 4. 关键结论和发现

### 主要发现
1. **投影代理（Projection Agents）在泛化性和可扩展性上取得了卓越平衡**：它解决了传统RL-GCO方法在处理大规模、多样化图时的根本瓶颈。
2. **连续潜动作空间是应对超线性复杂决策的有效范式**：通过将复杂动作映射到连续向量空间，可以利用高效的几何操作（如NN搜索）进行快速决策。
3. **单次前向传播 + 检索的架构极具潜力**：这种方法不仅速度快，而且由于共享的GNN嵌入空间，天然支持跨图泛化和动作插值。
4. **LaGCO-RL 库促进了研究的标准化和复现性**：提供了一个模块化框架，简化了新GCO任务的开发和现有方法的公平比较。

### 局限性
1. **编码与解码策略固定**：当前设计仅使用单一编码方式和k-NN解码，未探索更复杂的解码机制（如学习评分函数）。
2. **嵌入未微调**：GNN嵌入是无监督预训练得到的，未在下游RL任务中联合微调，可能限制了峰值性能。
3. **评估范围有限**：实验集中在RL方法内比较，未涵盖所有非RL方法（如监督学习、混合方法）。
4. **训练成本较高**：相比离散方法，`Projection` 的训练时间更长（Table 5 显示其训练耗时最高）。

### 未来工作方向
- 探索更先进的**解码策略**（如基于学习的排序、多步细化）。
- 研究**嵌入空间的联合优化**（representation learning + policy learning end-to-end）。
- 将该框架应用于更广泛的**现实世界GCO问题**，如芯片布局、供应链优化等。
- 结合**混合方法**，例如用`Projection`生成候选集，再用轻量级启发式精炼。
- 优化训练效率，降低计算开销。

---

</details>

---

### 5. [Spectral Souping: A Unified Framework for Online Preference Alignment](https://arxiv.org/abs/2605.20408)

**Authors**: Yinlam Chow, Guy Tennenholtz, Ted Yun, James Harrison, Arthur Gretton, Andre Barreto, Bo Dai  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.20408v1  

#### Abstract
Reinforcement Learning from Human Feedback (RLHF) effectively aligns Large Language Models (LLMs) with aggregate human preferences but often fails to address the diverse and conflicting needs of individual users. To overcome this issue, we introduce Spectral Souping, a unified framework for efficien...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Spectral Souping: A Unified Framework for Online Preference Alignment**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前主流的 **Reinforcement Learning from Human Feedback (RLHF)** 和 **Direct Preference Optimization (DPO)** 虽然能有效对齐大语言模型（LLM）与人类偏好，但其依赖于聚合反馈生成的单一奖励函数，无法满足**个体用户的多样化、甚至冲突的偏好需求**。为每个用户单独微调模型成本高昂且不可扩展。

本文旨在解决 **在线个性化偏好对齐（online preference alignment）** 的挑战——如何在不进行昂贵在线重训练的前提下，动态适应不同用户的细粒度偏好。

---

### **提出的新方法与新思路**
作者提出了 **Spectral Souping**，一个统一的、高效的在线偏好对齐框架，其核心思想基于以下理论发现：

- **通用谱表示（Universal Spectral Representation）**：  
  在语言马尔可夫决策过程（Language MDP）中，LLM策略的logits并非存在于任意空间，而是位于由MDP动力学决定的**低维结构化潜空间**中。该空间可通过参考LLM的特征 $ p(s) $ 构成的“谱表示”来参数化最优Q函数。

- **两阶段方法论**：
  1. **离线阶段（Offline Phase）**：预先训练一组专注于不同偏好维度（如帮助性、诚实性等）的**专业化策略（specialized policies）**，作为“基础策略”。
  2. **在线阶段（Online Phase）**：通过“汤化（souping）”机制，在推理时动态组合这些基础策略，形成针对特定用户的定制化策略，无需重新训练。

- **两种Souping方式**：
  - **显式Souping（Explicit Souping）**：直接线性组合各策略输出的logits。
  - **隐式Souping（Implicit Souping）**：通过拒绝采样（rejection sampling）从参考策略中生成动作，并依据Q函数差值接受或拒绝。

---

### **相比现有方法的优势**
| 维度 | Spectral Souping | 传统方法（如RLHF） |
|------|------------------|--------------------|
| **效率** | 仅需学习低维混合权重 $ \lambda $，避免全模型微调 | 需要完整的在线微调，计算开销大 |
| **可扩展性** | 支持零样本泛化到新用户 | 每个用户都需要独立训练 |
| **理论保障** | 提供**次优性界（sub-optimality bounds）**，证明性能接近完全微调的策略 | 多为启发式方法，缺乏理论支持 |
| **灵活性** | 可灵活增减基础策略以应对新偏好维度 | 模型结构固定，难以扩展 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验在三个真实世界的LLM个性化任务上进行：

1. **UltraFeedback Dataset**  
   - 包含提示词及响应对，标注了四维属性：**helpfulness, honesty, instruction-following, truthfulness**。
   - 合成多样化的用户偏好向量 $ w \in \mathbb{R}^4 $ 来模拟不同用户。
   - 过滤出偏好冲突的“争议性样本”，共 **23,614训练 / 401测试** 示例。

2. **Text-to-Image (T2I) Generation (PASTA Framework)**  
   - 5轮交互式图像生成任务，使用 **Stable Diffusion XL** 生成图像，**Gemini 1.5 Flash** 扩展提示。
   - 用户偏好关注美学、一致性等长期体验。
   - 离线阶段基于32种用户行为模式生成超30万条轨迹。

3. **Sleep Coaching (LifeSnaps Dataset)**  
   - 基于68位真实个体的睡眠健康档案构建合成用户。
   - 偏好与**Big Five人格特质**（extraversion, agreeableness等）关联。
   - 使用Gemini 1.5 Flash作为自动评分器评估对话质量。

---

### **实验设置与评估指标**
- **模型架构**：基于 **Gemma-V3** 系列（4B, 1B, 270M），使用 **LoRA** 在中间层进行适配。
- **离线训练**：每个基础策略在对应偏好的数据集上独立训练。
- **在线适应**：利用少量用户反馈实时更新混合权重 $ \lambda $。
- **评估方式**：
  - **Test-time Training Performance**：在线学习过程中的表现。
  - **Evaluation Performance**：最终收敛后的性能。
  - 使用 **Bradley-Terry Loss** 进行偏好优化。

---

### **基线方法对比**
| 基线方法 | 类型 | 描述 |
|--------|------|------|
| **Bespoke RLHF** | 上限基准 | 对每位用户单独微调，性能上限但成本高 |
| **P-SOUPS (Personalized Soups)** | 参数融合 | 加权平均多个专家模型的参数 |
| **PAD / PAD-SF** | 解码时对齐 | 利用奖励向量引导解码；SF版本引入successor features |
| **RLHF** | 强化学习微调 | 标准流程：奖励建模 + PPO微调 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
在所有三个领域中，Spectral Souping 显著优于现有方法：

| 方法 | UltraFeedback | T2I Generation | Sleep Coaching |
|------|---------------|----------------|----------------|
| **Spectral Souping (显式)** | **83%** of RLHF | **88%** of RLHF | **72%** of RLHF |
| P-SOUPS | ~75% | ~80% | ~65% |
| PAD-SF | ~70% | ~75% | ~60% |

> 注：“% of RLHF” 表示相对于完全微调方法达到的性能比例。

- **显式Souping略优于隐式Souping**，因其更精确地组合logits，避免采样偏差。
- 即使只用部分基础策略（K较小），仍能取得良好效果，体现鲁棒性。

---

### **与基线方法的对比结果**
- **显著超越P-SOUPS**：说明在**谱空间内合并策略**比在原始参数空间中加权更有效。
- **优于PAD/PAD-SF**：表明基于**最优Q函数权重**的指导比仅用奖励权重或advantage approximation更稳定高效。
- **逼近RLHF上限**：尽管免去了微调，性能仍可达其80%以上，验证了理论界的实用性。

---

### **消融实验结果**
通过逐步减少基础策略数量 $ K $ 观察性能变化（见图3）：

- **存在最小基础集阈值**：
  - UltraFeedback: $ K=7 $
  - T2I: $ K=5 $
  - Sleep Coaching: $ K=13 $
  - 低于此阈值性能急剧下降，说明需要足够覆盖偏好空间。

- **模型规模影响鲁棒性**：
  - 更大的模型（如Gemma 4B）即使减少基础策略，性能下降更平缓。
  - 推测原因：更大模型具有更丰富、更具表达力的谱表示。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **谱表示的存在性被实证验证**：LLM策略的logits确实存在于一个可被线性组合的低维谱空间中。
2. ✅ **Spectral Souping实现了高效在线个性化**：仅需调整少量混合权重即可实现接近全微调的性能。
3. ✅ **理论与实践结合紧密**：首次为“模型汤化”类方法提供了**形式化的次优性界**，填补了理论空白。
4. ✅ **方法具备强泛化能力**：在文本、图文、医疗等多个复杂场景下均表现出色。

---

### **方法的局限性**
1. **依赖基础策略的质量与多样性**：若基础策略未能覆盖关键偏好维度，则无法准确拟合新用户。
2. **线性组合假设限制表达能力**：非线性交互可能被忽略，未来可探索非线性融合机制。
3. **负权重处理带来额外误差项**：定理1中的惩罚项提示负权重会影响性能边界。
4. **谱表示是“发现”而非“学习”**：目前依赖预训练LLM内部特征，尚未端到端学习最优谱空间。

---

### **未来工作方向**
1. **在预训练阶段嵌入谱表示**：让基础模型天生具备可分解的偏好适应能力。
2. **开发非线性Souping算法**：如神经网络融合、meta-learning等方式快速推断 $ \lambda $。
3. **扩展至多模态场景**：应用于个性化 **text-to-image**, **audio generation** 等跨模态任务。
4. **主动偏好探测（Active Preference Elicitation）**：设计策略主动提问以更快识别用户偏好向量 $ w $。

---

> **总结一句话**：  
> **Spectral Souping 将LLM个性化从“昂贵微调”转变为“智能拼装”，通过发现并利用内在的谱结构，实现了高效、可解释、有理论保证的在线偏好对齐新范式。**

</details>

---

### 6. [PulseCol: Periodically Refreshed Column-Sparse Attention for Accelerating Diffusion Language Models](https://arxiv.org/abs/2605.20813)

**Authors**: Yanyi Lyu, Letian Chen, Futing Sun, Miao Zhang, Weili Guan, Liqiang Nie  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.20813v1  

#### Abstract
Inference in diffusion large language models (dLLMs) is computationally expensive, as full self-attention must be repeatedly executed at each step of the denoising process without KV cache. Recent sparse attention methods for dLLMs mitigate this cost via block-sparse computation, which is applied on...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PulseCol: Periodically Refreshed Column-Sparse Attention for Accelerating Diffusion Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Diffusion Large Language Models (dLLMs) 在推理过程中需要在每一步去噪阶段重复计算全量 self-attention，且无法像自回归模型那样复用 KV Cache，导致计算成本极高。尽管已有稀疏注意力方法（如 SparseD）尝试通过**块级稀疏（block-sparse）** 来加速，但这些方法通常仅在后期去噪步骤中应用稀疏化，以避免早期稀疏化对生成质量造成显著影响，因此加速效果有限。

### **提出的新方法与新思路**
本文提出了 **PulseCol**，一种**周期性刷新的列稀疏注意力（periodically refreshed column-sparse attention）** 方法，用于高效加速 dLLM 推理。其核心思想包括：

- **细粒度列稀疏结构（column-sparse attention）**：  
  替代传统的 block-sparse，直接识别并保留重要 key 列（即 attention 集中在少数 key 上），实现更精细、更匹配实际 attention 分布的稀疏模式。

- **早期稀疏化 + 周期性刷新机制**：  
  在**早期去噪步骤**即可安全引入稀疏计算，通过在初始阶段构建稀疏模式，并在后续少量中间步骤中**周期性刷新稀疏索引**，以适应 attention 模式的动态演化。

- **优化的 GPU 内核支持**：  
  设计了高效的 **column-sparse attention kernel**，避免显式构造完整 attention 矩阵，利用 tile-based 计算和 on-chip memory 保持内存效率。

### **相比现有方法的优势**
- ✅ **更早启用稀疏化**：突破了“早期必须全注意力”的限制，扩大了可加速的时间窗口。
- ✅ **更高精度保留**：列稀疏比块稀疏更能保留关键 attention 连接，尤其在高稀疏率下表现更鲁棒。
- ✅ **更强的实际加速**：结合算法稀疏性和定制 kernel，实现了高达 **1.95× 的端到端速度提升**，优于 FlashAttention 和 SparseD。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **GSM8K**：数学推理任务（4-shot）
- **HumanEval**：代码生成能力评估（0-shot）
- **RULER**：长上下文理解能力测试（0-shot，涵盖 4K 和 8K 上下文长度）

### **实验设置与评估指标**
- **模型**：
  - LLaDA-1.5
  - Dream
- **稀疏率设置**：主要测试 50% 和 80% 稀疏率下的性能
- **评估指标**：
  - **准确率（Accuracy）**：各任务上的得分
  - **端到端延迟（End-to-end Latency）**：总推理时间
  - **Speedup**：相对于 FlashAttention 的加速比
  - **Attention Recall**：oracle 模式下 top-k 注意力位置的召回率（图1a）

### **基线方法对比**
| 方法 | 类型 |
|------|------|
| Full Attention (FlashAttention) | 全注意力基准 |
| Sliding Window | 局部窗口稀疏 |
| StreamingLLM | Sink token + 局部缓存 |
| SparseD [24] | Block-sparse，跳过前 20% 步骤 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **准确性结果（平均得分）**

| 方法 | LLaDA-1.5 (80%) | LLaDA-1.5 (50%) | Dream (80%) | Dream (50%) |
|------|------------------|------------------|-------------|--------------|
| Full Attention | 67.58 | 67.58 | 73.56 | 73.56 |
| SparseD | 63.89 | 67.27 | 49.85 | 70.85 |
| **PulseCol (Ours)** | **66.58** | **67.47** | **72.83** | **74.08** |

> 💡 **观察**：
> - 在 80% 稀疏率下，PulseCol 仅比全注意力低约 1 point，而 SparseD 下降明显（Dream 下降超 23 points）。
> - 在 50% 稀疏率下，PulseCol 甚至**超过原始全注意力模型**（Dream 达 74.08 vs 73.56），表明其稀疏模式更优。

#### **延迟与加速比（64K context, 1024 steps）**

| 方法 | LLaDA-1.5 加速比 | Dream 加速比 |
|------|--------------------|---------------|
| SparseD (80%) | 1.64× | 1.65× |
| **PulseCol (80%)** | **1.95×** | **1.95×** |

> 🔥 **最高达 1.95× 端到端加速**，显著优于 SparseD（相对提速 ~20%）。

#### **内核级加速（Column-Sparse Kernel）**
- 在 90% 稀疏率、128K 上下文下，**单 kernel 最高可达 10.54× 速度提升**（见图5）。
- 加速效果随 context length 增加而增强，体现其对长序列的友好性。

---

### **消融实验结果**

| 配置 | LLaDA-1.5 (GSM8K/HE) | Dream (GSM8K/HE) | 平均下降 |
|------|------------------------|------------------|----------|
| **CS + PR（完整 PulseCol）** | 76.19 / 38.41 | 74.07 / 57.32 | — |
| Block-sparse + PR | 58.00 / 33.54 | 72.40 / 49.39 | ↓8.16 |
| Column-sparse + Skip | 69.14 / 34.15 | 70.28 / 52.44 | ↓5.00 |

> 📌 **结论**：
> - **列稀疏（CS）** 是性能保障的关键，块稀疏会严重损害早期注意力建模。
> - **周期性刷新（PR）** 明显优于“跳过早期”策略（Skip），说明动态更新比静态跳过更有效。

#### **超参数分析**
- **查询组大小（Group Size）**：最佳为 32，过大则共享过度，过小则开销高。
- **刷新次数（Refresh Count）**：LLaDA-1.5 最佳为 16，Dream 为 8，过多反而不稳定。
- **刷新窗口（Refresh Window）**：集中在前 30% 步骤内刷新即可，后期无需更新。

---

## **4. 关键结论和发现**

### **主要发现**
1. **早期去噪 attention 存在显著列稀疏性**：并非真正“密集”，而是集中在少数关键 key 上，这为早期稀疏化提供了理论基础。
2. **Block-sparse 不适配早期 attention 结构**：其粗粒度会导致误删关键连接或保留无用交互，是现有方法不敢在早期稀疏的根本原因。
3. **列稀疏 + 周期刷新 = 高效且稳定**：既能从第一步开始加速，又能通过少量刷新适应 attention 动态变化。
4. **算法与系统协同设计至关重要**：专用 column-sparse kernel 将算法稀疏转化为真实硬件加速，否则稀疏可能“只减 FLOPs 不减 latency”。

### **方法的局限性**
- **固定刷新调度**：当前使用预设的均匀刷新策略（uniform schedule），未根据 attention 变化动态调整，可能非最优。
- **依赖外部稀疏模式构建**：仍需在刷新步运行一次 full attention 来收集重要列，带来额外开销。
- **通用性待验证**：目前仅在 LLaDA-1.5 和 Dream 上验证，是否适用于所有 dLLM 架构有待进一步研究。

### **未来工作方向**
- **自适应刷新机制**：基于 attention 差异度自动决定是否刷新，减少不必要的 full attention 调用。
- **学习式稀疏预测**：训练轻量网络预测重要 key 列，替代运行时 score collection。
- **扩展至其他模态**：将 PulseCol 思路应用于 diffusion-based 图像、音频等多模态生成模型。

---

> ✅ **总结一句话**：  
> **PulseCol 通过“列稀疏 + 周期刷新”打破了 dLLM 早期不能稀疏的瓶颈，在几乎不损失性能的前提下实现了高达 1.95× 的端到端加速，是 diffusion language model 高效推理的重要进展。**

</details>

---

### 7. [LT2: Linear-Time Looped Transformers](https://arxiv.org/abs/2605.20670)

**Authors**: Chunyuan Deng, Yizhe Zhang, Rui-Jie Zhu, Yuanyuan Xu, Jiarui Liu, T. S. Eugene Ng, Hanjie Chen  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.20670v1  

#### Abstract
Looped Transformers (LT) have emerged as a powerful architecture by iterating their layers multiple times before decoding the final token. However, pairing them with full attention retains quadratic complexity, making them computationally expensive and slow. We introduce LT2 (Linear-Time Looped Tran...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LT2: Linear-Time Looped Transformers

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Looped Transformers (LT)** 虽然通过权重共享实现参数高效推理，但由于其依赖 **quadratic softmax attention**，在训练和推理时存在严重的计算瓶颈。随着序列长度和循环次数 $T$ 的增加，**attention FLOPs 和 KV-cache 内存开销呈平方增长**，导致长上下文场景下效率低下，难以扩展。

### 提出了什么新方法或新思路
本文提出 **LT2 (Linear-Time Looped Transformers)**，一个将 subquadratic token mixer 引入 looped 架构的新家族，旨在解决上述效率瓶颈。具体包括：

- **LT2-linear**：用 **linear attention**（如 GDN、KDA）替代 full attention。
- **LT2-sparse**：用 **sparse attention**（如 DSA、NSA）替代 full attention。
- **LT2-hybrid**：混合多种 attention 变体，在循环中交错使用不同类型的 mixer（如 GDN + DSA 或 Full + GDN），以平衡效率与性能。

此外，作者还提出了 **从预训练 LT 模型蒸馏为 LT2-hybrid 模型的方法**，无需从头训练即可获得线性时间效率。

### 相比现有方法的优势
- **效率显著提升**：避免了 quadratic attention，实现 **linear-time decoding**，尤其在长上下文（如 8k–32k tokens）下吞吐量大幅提升（最高达 5.7×）。
- **性能不降反升**：
  - LT2-hybrid (Full+GDN) 在零样本任务上平均得分比标准 Looped Transformer 高 **+2.1 分**（61.4% vs 59.3%）。
  - LT2-hybrid (GDN+DSA) 完全不含 full attention，却能匹配 full-attention LT 的质量，同时保持线性复杂度。
- **可迁移性强**：可通过轻量级蒸馏将已有 LT 模型转换为 LT2，仅需约 1B tokens 微调即可保留原模型能力并继承速度优势。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主预训练数据**：FineWeb-Edu（100B tokens）
- **下游评估基准**：
  - 零样本评测套件：HellaSwag, ARC-E/C, Winogrande, MMLU, GSM8K, PIQA, OBQA, BoolQ 等
  - 长上下文检索任务：SWDE, SQuAD, FDA, TriviaQA, Natural Questions, DROP
  - “大海捞针”测试（Needle-in-a-Haystack, NIAH）用于评估极端长程依赖建模能力

### 实验设置和评估指标
- **模型规模**：0.6B 和 1.3B 参数
- **循环次数 $T$**：固定为 4 次迭代
- **评估指标**：
  - **Perplexity (PPL)**：语言建模能力
  - **Zero-shot Accuracy (%)**：多个常识推理与问答任务的平均分
  - **Decode Throughput (tok/s)**：批大小=8，上下文长度=8k 时的解码速度
  - **OOM 边界**：最大支持的上下文长度（如 32k）
  - **Training Stability**：梯度范数、loss 曲线平滑性

### 基线方法对比
| 基线模型 | 描述 |
|--------|------|
| Standard Transformer | 非循环标准 Transformer |
| Looped Transformer (ref) | 权重共享、带 full attention 的循环架构（如 Ouro） |
| RetNet / Mamba2 / HGRN2 | 典型 linear attention 架构 |
| Sliding Window / NSA / DSA | 稀疏 attention 变体 |
| Jamba / Olmo-Hybrid | 已有的 hybrid attention 模型 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### ✅ 语言建模性能（1.3B 模型，avg. zero-shot）
| 模型 | PPL ↓ | 平均准确率 ↑ |
|------|-------|-------------|
| Looped Transformer (ref) | 9.87 | 59.27% |
| **LT2-hybrid (Full+GDN)** | **9.12** | **62.89%** (+3.6 pt) |
| **LT2-hybrid (GDN+DSA)** | 9.50 | 60.73% |
| Looped GDN | 9.75 | 59.92% |

> 💡 LT2-hybrid (Full+GDN) 不仅性能更强，且仍保持约 **5× 更高的 decode throughput**

#### ✅ 推理效率（decode throughput @ 8k context, bs=8）
| 模型 | 解码速度 (tok/s) |
|------|------------------|
| Looped Transformer | 22 tok/s |
| **LT2-hybrid (GDN+DSA)** | **125 tok/s** (**~5.7× 加速**) |
| **LT2-hybrid (Full+GDN)** | ~110 tok/s (**~5× 加速**) |

> 📈 在 32k 上下文中，标准 LT 在 bs=8 下已 OOM，而 LT2 可稳定运行

#### ✅ 长上下文检索（NIAH-4096，extrapolation）
| 模型 | 准确率 |
|------|--------|
| Looped Transformer | 0.0% |
| Looped Hybrid (GDN+DSA) | 60.3% |
| **Looped Hybrid (Full+GDN)** | **63.7%** |

> 🔍 表明 LT2 在未见过的超长上下文中具备更强外推能力

### 与基线方法的对比结果
- **LT2-linear/sparse**：在不使用 full attention 的情况下，性能接近甚至超过标准 Looped Transformer。
- **LT2-hybrid (GDN+DSA)**：完全线性时间成本下，达到 full-attention LT 的质量水平。
- **LT2-hybrid (Full+GDN)**：小比例 full attention（仅 20% 层）带来显著性能增益，同时维持高效率。

### 消融实验结果
#### （1）混合比例（Full : GDN）
| 比例 | PPL | Avg Acc |
|------|-----|---------|
| 1:0 (Full-only) | 9.87 | 59.27% |
| 1:4 (**最优**) | **9.31** | **61.39%** |
| 1:6 | 9.36 | 61.07% |
| 1:12 | 9.74 | 59.51% |
| 0:1 (GDN-only) | 10.02 | 58.42% |

> ✅ 存在一个 **inverse-U 形关系**，1:4 是最佳权衡点

#### （2）混合模式（Pattern）
| 模式 | 平均准确率 |
|------|-----------|
| Interleave（交错） | 61.39% |
| Bookend（首尾 full） | 61.52% |
| Front-loaded（前段集中） | 60.61% |
| Back-loaded（后段集中） | 60.43% |

> ✅ 分散分布优于集中分布；bookend 略优，表明输入编码与输出读出阶段都需要 full attention

#### （3）混合层级（Level）
| 方式 | 平均准确率 |
|------|-----------|
| Depth-level（层间交错） | 61.39% |
| Loop-level（循环间切换） | ≤61.10% |
| Random-sample + voting | 61.55%（但需 5× 推理代价） |

> ❌ loop-level 切换无明显收益，推荐固定 depth-level 交错

---

## 4. 关键结论和发现

### 主要发现
1. **Looping 与 subquadratic attention 天然协同**：
   - 对于 **linear attention**，looping 实现了 **rank-T memory update**，增强了状态追踪能力。
   - 对于 **sparse attention**，looping 扩展了 **effective receptive field 至 $O(Tw)$**，弥补局部视野缺陷。

2. **Hybrid 架构打开新的 Pareto frontier**：
   - **LT2-hybrid (GDN+DSA)**：实现了 **full-quality at linear-cost**
   - **LT2-hybrid (Full+GDN)**：实现了 **better-than-full at near-linear speed**

3. **训练更稳定**：
   - GDN 等具有 **data-dependent gating** 和 **delta rule** 的 mixer 在循环中表现出更小的梯度波动和更平滑的 loss 曲线。
   - 引入 **SDPA output gate** 可有效缓解 attention sink 在循环中的累积效应。

4. **可高效蒸馏已有模型**：
   - 将 Ouro-1.4B 蒸馏为 **Ouro-hybrid-1.4B** 后，仅用 1B tokens 微调即达到：
     - 匹配工业级 1B 模型性能
     - 接近 4B 模型表现
     - 保持线性时间推理优势

### 方法的局限性
- **未探索 full loop-level hybridization**：目前只尝试简单 schedule，未让每个循环动态选择不同 mixer 类型。
- **缺乏跨循环状态传递机制**：当前 state 在每轮重新初始化，未设计显式的跨 loop state carry。
- **ACT（Adaptive Computation Time）尚未稳定应用**：由于优化不稳定和 ragged halting 问题，目前采用固定 $T=4$。

### 未来工作方向
- 设计更灵活的 **loop-level adaptive mixer selection**
- 引入 **explicit cross-loop state propagation**
- 开发稳定的 **adaptive loop termination** 机制（如改进 ACT）
- 探索 LT2 在 vision、multimodal 等领域的扩展应用

---

> 🏁 **总结一句话**：  
> **LT2 成功将 looped transformers 的参数效率优势与 linear-time attention 的计算效率结合，通过 hybrid 架构突破了传统 full-attention LT 的性能-效率权衡边界，为高效小型语言模型的发展提供了新路径。**

</details>

---

### 8. [ShapeBench: A Scalable Benchmark and Diagnostic Suite for Standardized Evaluation in Aerodynamic Shape Optimization](https://arxiv.org/abs/2605.20763)

**Authors**: Shaghayegh Fazliani, Krissh Chawla, Jack Guo, Yiren Shen, Matthias Ihme, Madeleine Udell  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.20763v1  

#### Abstract
Rapid progress in aerodynamic shape optimization (ASO) has outpaced currently-available standardized evaluation frameworks. Fair comparison requires a unified benchmark spanning diverse shape classes, objective formulations, and matched-budget state-of-the-art baselines. We introduce ShapeBench, an ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ShapeBench: A Scalable Benchmark and Diagnostic Suite for Standardized Evaluation in Aerodynamic Shape Optimization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前在 **Aerodynamic Shape Optimization (ASO)** 领域存在以下关键挑战：

- **缺乏统一的基准测试框架**：现有研究多基于狭窄任务（如二维翼型优化），难以进行跨方法、跨几何形状的公平比较。
- **评估不一致**：不同研究使用不同的仿真工具、预算设定和接口，导致结果不可复现、不可比。
- **忽略保真度差距（fidelity gap）**：广泛使用 surrogate 模型加速搜索，但缺乏对高保真 CFD 验证的支持，易产生“虚假最优”设计。
- **新兴 LLM 方法缺乏标准化评估**：尽管 LLM-driven 优化器开始应用于科学领域，但其在 ASO 中的表现尚无统一评估标准。

### 提出了什么新方法或新思路

作者提出了 **ShapeBench**——一个开源、可扩展的 ASO 统一基准与诊断套件，具备以下核心特性：

- **103 个 ASO 任务**，覆盖 **8 类气动外形**（如 2D Airfoil、3D BWB、Delta Wing、Passenger Car、CCA 等），涵盖单目标/多目标、单点/多点飞行条件、连续/混合变量等多种优化范式。
- **统一 API 接口**：提供标准化的 `Python API`，支持所有优化器通过相同接口调用任务，实现公平比较。
- **配对仿真环境**：每个任务配备 **预训练 surrogate 模型**（用于快速搜索）和 **高保真 CFD 验证管道**（用于最终验证），支持 fidelity-gap 分析。
- **诊断套件（Diagnostic Suite）**：引入两阶段诊断流程：
  1. **确定性检查**：参数边界、网格质量、物理一致性等；
  2. **LLM 综合判断**：结合设计图、流场可视化生成审计报告，识别如 `GEOMETRY_DEFORMATION_EXCESSIVE`、`SURROGATE_DOMAIN_RISK` 等失败模式。
- **标准化基线与新方法**：
  - 提供经典优化器（Adjoint, L-BFGS-B, PSO, CMA-ES, Bayesian Optimization）的可复现配置。
  - 引入通用 LLM 优化器（OpenEvolve, ShinkaEvolve）作为对比。
  - 提出 **ShapeEvolve**：一种专为 ASO 设计的 LLM 进化方法，融合 **ASO 结构化提示**、**流场反馈** 和 **代码生成能力**，实现更高效的局部搜索。

### 相比现有方法的优势

| 特性 | ShapeBench | ADODG / AFBench / EngiBench |
|------|-----------|-----------------------------|
| 多几何支持 | ✅ 覆盖 8 类 | ❌ 通常仅限 Airfoil 或 Wing |
| Surrogate-to-CFD 验证 | ✅ 支持 | ❌ 缺乏 |
| 统一预算协议 | ✅ 明确评估次数 | ❌ 不一致 |
| 混合变量支持 | ✅ CERAS, STA | ❌ 有限 |
| 诊断能力 | ✅ 内置 Diagnostic Suite | ❌ 无 |
| LLM 方法集成 | ✅ 支持并提出 ShapeEvolve | ❌ 未涉及 |

---

## 2. 核心实验方法和设置

### 使用的数据集与任务

- **任务总数**：103 个 ASO 任务，分布在 8 个类别中。
- **主要数据来源**：
  - **公开数据集**：AFBench (Airfoil), BlendedNet (BWB), DrivAerStar (Car), SuperWing (Swept Wing), VortexNet (Delta Wing), NeuralFoil (Airfoil)。
  - **新增数据集**：**COCOANet**：作者构建的 **Collaborative Combat Aircraft (CCA)** 数据集，含 3,570 个高保真 CFD 仿真样本，填补该领域公共数据空白。
- **仿真层级**：
  - **Surrogate**：用于快速评估（毫秒至秒级）。
  - **High-Fidelity CFD**：用于最终验证（小时级），如 SU2、FUN3D、STAR-CCM+、Flow360。

### 实验设置和评估指标

- **评估协议**：
  - 所有优化器在 **相同评估预算** 下运行（以函数评估次数为单位）。
  - 使用 **归一化目标值** 和 **中位排名轨迹** 进行跨任务比较。
  - 报告 **中位数、四分位距（IQR）、最佳设计**。
- **关键指标**：
  - **Objective Value**：如 $ L/D $、$ C_D $、Fuel Mass。
  - **Rank Stability**：计算优化器在不同任务上的排名相关性（Spearman ρ）。
  - **Fidelity Gap**：Surrogate 与 CFD 对同一设计的目标值差异。
  - **Physical Credibility**：通过 Diagnostic Suite 判断设计是否合理。

### 基线方法对比

| 优化器 | 类别 | 实现方式 |
|--------|------|---------|
| Adjoint (IPOPT) | Gradient-based | 基于 CasADi 自动微分 |
| L-BFGS-B | Gradient-based | 有限差分梯度估计 |
| Bayesian Optimization (BO) | Surrogate-based | BoTorch + GP |
| PSO | Derivative-free | 粒子群算法 |
| CMA-ES | Derivative-free | 协方差矩阵自适应进化策略 |
| OpenEvolve | LLM-driven | LLM 生成代码进行优化 |
| ShinkaEvolve | LLM-driven | Bandit-guided 操作符演化 |
| **ShapeEvolve** | **LLM-driven (Proposed)** | **ASO 专用提示 + 流场反馈 + LLM 代码生成** |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）优化器排名不稳定（Rank Instability）
- 在不同任务上，优化器的相对表现差异巨大。
- **平均成对 Spearman 相关系数仅为 0.013**，表明 **单任务结论无法泛化到其他任务**。
- 例如：
  - **Bayesian Optimization** 在 CERAS 和 Delta Wing 上表现最好，但在 CCA 上最差。
  - **LLM 方法**（如 ShapeEvolve）在 CCA 上显著优于传统方法。

#### （2）ShapeEvolve 性能突出
- 在 **3D CCA 任务**（最大化 $ L/D $）中：
  - **ShapeEvolve 达到 $ L/D = 84.7 $**，远超第二名 Bayesian Opt. ($ L/D = 43.2 $)。
  - 传统方法如 L-BFGS-B、CMA-ES 表现较差。
- 在 **2D Airfoil 多点阻力最小化** 中：
  - ShapeEvolve 收敛最快，在约 2,000–3,000 次评估内接近最优。
  - 最终设计经 XFOIL 验证，$ C_D $ 误差 < 0.5%，证明 surrogate 准确。

#### （3）保真度差距分析（Fidelity Gap）
- **NeuralFoil vs XFOIL**：
  - 多点任务下，最佳设计的 $ C_D $ 差异最大为 **2.67%**。
  - 单点升阻比最大化时，$ L/D $ 保真度差距达 **~9%**，极端案例高达 **~18%**，源于 surrogate 在边缘区域预测不稳定。
- **DrivAerStar vs CFD**：
  - 由于无直接 CFD 接口，依赖 Transolver surrogate，需警惕 surrogate exploitation。

#### （4）消融实验与诊断结果

##### Surrogate Exploitation 检测（3D Car 设计）
- 所有优化器均将 $ C_D $ 降至 ~0.065，看似极优。
- 但 **Diagnostic Suite 发现严重问题**：
  - **几何变形过度**：后部离地间隙几乎坍塌（rear droop）。
  - **80% 参数处于边界**（G001）。
  - **综合角度应力过高**（G002）。
- **LLM 报告结论**：
  - `Primary failure`: `GEOMETRY_DEFORMATION_EXCESSIVE`
  - `Physical credibility`: `LOW`
  - `Recommended mitigation`: 添加后部离地间隙硬约束。

##### Blended Wing Body (BWB) 局部最优陷阱
- 使用 $ C_{fx} $（摩擦阻力代理）为目标时：
  - BO/PSO/CMA-ES 可收敛至全局最优（Corner A）。
  - 但若直接优化 $ L/D_{proxy} $，多数方法陷入局部最优（Corner B）。
- **Warm-start 实验**：从 Corner A 初始化，所有方法均可成功收敛，证明是 **探索问题而非收敛问题**。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **单任务评估具有误导性**：优化器在不同任务上的排名高度不稳定（Spearman ρ ≈ 0.013），**不能仅凭单一案例推断方法普适性**。
2. ✅ **LLM 方法展现潜力**：特别是 **ShapeEvolve** 在复杂高维任务（如 CCA）上显著超越传统方法，表明 LLM 在 ASO 中具有竞争力。
3. ✅ **Surrogate 存在严重 exploitation 风险**：即使 surrogate 经过良好训练，优化过程仍可能产生物理上不合理的设计（如边界坍塌、低可信度区域）。
4. ✅ **需要显式有效性检查**：仅看目标值会掩盖失败模式，必须结合 **Diagnostic Suite** 进行可信度评估。
5. ✅ **保真度差距不可忽视**：Surrogate 与 CFD 之间存在系统性偏差，尤其在设计空间边缘，需进行最终验证。

### 方法的局限性

- ❌ **并非所有任务都支持 CFD 验证**：如 **3D Passenger Car** 缺乏可运行的 CFD 接口，只能依赖 surrogate。
- ❌ **部分 surrogate 本身存在 exploitable 缺陷**：如 DrivAerStar 和 BlendedNet 在边界处易被利用，调整目标仅改变被利用的角落，无法根除问题。
- ❌ **计算资源门槛高**：虽然 surrogate 快速，但 CFD 验证成本高昂，限制了大规模部署。
- ❌ **当前侧重概念与速度模型**：尚未充分纳入制造性、鲁棒性、不确定性等现实约束。

### 未来工作方向

- 🔜 **扩展高保真验证覆盖范围**：为更多任务（尤其是汽车）添加可执行 CFD 回放功能。
- 🔜 **增强不确定性建模**：在 surrogate 中引入置信度校准机制，避免向低置信区域探索。
- 🔜 **强化诊断能力**：增加更丰富的物理一致性检查（如力矩平衡、诱导阻力合理性）。
- 🔜 **支持更复杂的现实约束**：如制造工艺限制、结构强度、噪声、多学科耦合等。
- 🔜 **推动社区共建**：支持插件式任务扩展，降低新任务集成成本，保持基准语义一致性。

---

> **总结**：  
> **ShapeBench** 是首个面向 **多几何、多范式、多保真度** 的标准化 ASO 基准平台。它不仅提供了丰富的任务与基线，更重要的是揭示了当前评估中的盲区——**排名不稳定** 与 **surrogate exploitation**。通过引入 **Diagnostic Suite** 和 **ShapeEvolve** 等工具，该工作为未来更可靠、更通用的 ASO 方法发展奠定了坚实基础。

</details>

---

### 9. [Llamas on the Web: Memory-Efficient, Performance-Portable, and Multi-Precision LLM Inference with WebGPU](https://arxiv.org/abs/2605.20706)

**Authors**: Reese Levine, Rithik Sharma, Nikhil Jain, Abhijit Ramesh, Zheyuan Chen, Neha Abbas, James Contini, Tyler Sorensen  
**Category**: cs.DC  
**Published**: 2026-05-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.20706v1  

#### Abstract
Running language models in the browser presents a unique opportunity to build efficient, private, and portable AI applications, but requires contending with constrained memory availability and heterogeneous hardware targets. To realize this opportunity, we present Llamas on the Web (LlamaWeb), a Web...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Llamas on the Web: Memory-Efficient, Performance-Portable, and Multi-Precision LLM Inference with WebGPU

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于浏览器的 **Large Language Model (LLM)** 推理框架面临三大挑战：
- **内存效率低下**：动态分配 GPU 内存、低效的模型加载导致内存占用高，易触发浏览器内存限制而崩溃。
- **缺乏性能可移植性（Performance Portability）**：WebGPU 提供功能可移植性，但不同硬件（如 NVIDIA、AMD、Apple Silicon）上的性能差异大，现有框架缺乏针对不同设备的优化机制。
- **量化格式支持有限**：为适应边缘设备，模型常采用多种量化格式（如 q4_k_m、q1_0），但现有框架支持的格式较少，限制了模型兼容性和灵活性。

### 提出的新方法
本文提出 **LlamaWeb**，一个为 `llama.cpp` 设计的 **WebGPU 后端**，旨在实现高效、可移植且多精度的 LLM 浏览器内推理。其核心创新包括：

- **静态内存规划（Static Memory Planning）**  
  在启动时一次性分配所有所需内存（包括中间状态和参数缓冲区），避免运行时动态分配，显著降低内存开销并防止内存泄漏。

- **可调优的内核库（Tunable Kernel Library）**  
  构建了一个模板化的 GPU 内核库，支持：
  - 根据设备特性（如 tile size、subgroup 支持）生成专用着色器。
  - 高效的内核调度与批处理，减少 CPU-GPU 同步开销。

- **量化感知的着色器设计（Quantization-Aware Shader Design）**  
  开发了支持多种 `llama.cpp` 量化格式（如 K-quants、I-quants、q1_0）的通用模板内核，将反量化操作直接集成到计算中，提升效率并增强扩展性。

### 相比现有方法的优势
| 维度 | LlamaWeb | WebLLM / Transformers.js |
|------|---------|--------------------------|
| **内存使用** | 减少 29–33% | 较高，存在内存泄漏风险 |
| **解码吞吐量（Decode Throughput）** | 提升 45–69% | 较低 |
| **支持的量化格式数量** | **23 种** | 6–7 种 |
| **可用模型数量** | **177,691 个**（GGUF 格式） | 400–41,632 个 |

---

## 2. 核心实验方法和设置

### 数据集与模型
- **模型集合**：共测试 **10 个 LLM**，涵盖不同架构与规模：
  - Transformer：Llama3.2、Qwen3、Gemma3
  - Hybrid SSM：LFM2.5
  - 1-bit 模型：Bonsai-1.7B
- **量化格式**：使用 `q4_k_m` 为主，部分实验覆盖 `q2_k`、`q8_0`、`f16` 和 `q1_0`。

### 实验平台
- **硬件**：在 **16 台设备**上进行测试，覆盖 **8 个厂商**的 GPU：
  - NVIDIA（RTX 5080）、AMD（RX 7900 XT）、Intel（Arc B580）
  - Apple（M4 Pro、M3、M2）、Qualcomm（Snapdragon X Elite）
  - 移动端：iPhone 17 Pro Max（iOS）、Galaxy S24（Android）
- **软件环境**：
  - 浏览器：Chrome（主）、Safari（iOS）
  - 后端对比：`llama.cpp` 的原生后端（CUDA、Metal、HIP、SYCL、Vulkan）

### 评估指标
| 指标 | 定义 |
|------|------|
| **Peak Memory Usage** | 推理过程中的峰值内存消耗（通过系统工具监控） |
| **Decode Throughput** | 解码阶段每秒生成的 token 数（tok/s） |
| **Prefill Throughput** | Prefill 阶段处理 prompt 的速度（tok/s） |
| **Geometric Mean Speedup** | 跨设备/配置的几何平均加速比，避免被极端值主导 |

### 基线方法对比
- **浏览器框架**：
  - **WebLLM**：基于 MLC-LLM + TVM + WebGPU
  - **Transformers.js**：基于 ONNX Runtime + WebGPU
- **原生框架**：
  - `llama.cpp` 的 CUDA、Metal、HIP、SYCL、Vulkan 后端

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **内存效率**：
  - LlamaWeb 的峰值内存使用比 WebLLM 低 **29–33%**，比 Transformers.js 低 **41%**。
  - 在 Safari 上，Transformers.js 出现严重内存泄漏，内存增长至 10GB 并被终止；LlamaWeb 无此问题。

- **解码吞吐量（Decode Throughput）**：
  - 在四款 GPU（NVIDIA RTX 5080、AMD RX 7900 XT、Intel Arc B580、Apple M4 Pro）上：
    - LlamaWeb 比 WebLLM 快 **54–69%**
    - 比 Transformers.js 快 **69%**
  - 在高端设备上，小模型可达 **>100 tok/s**，低端移动设备仍可达 **4–17 tok/s**。

- **Prefill 性能**：
  - 当前 LlamaWeb 的 Prefill 性能落后于 WebLLM（仅达其 **49%**）和 Transformers.js（**79%**），主要因缺乏 **kernel fusion** 和未启用 `subgroup matrix` 特性。

- **跨设备可移植性**：
  - 在 16 台设备上成功运行全部 10 个模型（除内存受限设备外）。
  - 通过 k-means 聚类将设备分为 high/mid/low 三档，LlamaWeb 在各档均表现稳定。

- **量化格式性能分析**：
  - **Decode 阶段**：从 `f16` 到 `q8_0` 提速明显（mid cluster 提升 53%），但继续压缩至 `q2_k` 反而导致性能下降（因反量化开销过大）。
  - **Prefill 阶段**：`f16` 表现最优，表明当前量化主要作用是 **节省内存而非提升性能**。

### 与原生后端对比
- **相比原生后端**：
  - 在 NVIDIA GPU 上，原生 CUDA 后端仍领先（Prefill 快 10×，Decode 快 2.5×）。
  - 但在某些配置下，LlamaWeb **优于 Vulkan 后端**（如 AMD 上 Decode 快 38%）。
  - 在 Intel GPU 上，LlamaWeb 的 Prefill 性能甚至 **超过原生 SYCL 后端 23%**。

- **安全检查影响**：
  - WebGPU 的运行时安全检查导致平均 **14–23% 的性能损失**，表明有进一步优化空间。

---

## 4. 关键结论和发现

### 主要发现
1. **LlamaWeb 实现了高效的浏览器内 LLM 推理**，在内存使用和解码吞吐量上显著优于现有框架。
2. **静态内存管理对浏览器环境至关重要**，有效避免了内存泄漏和 OOM 崩溃。
3. **性能可移植性可通过可调优内核库实现**，无需为每个设备编写专用代码。
4. **当前量化主要用于内存压缩**，而非性能提升；未来需更紧密的软硬协同设计以释放性能潜力。
5. **LlamaWeb 已具备与部分原生后端竞争的能力**，尤其在 Intel 和 AMD 平台上表现亮眼。

### 方法的局限性
- **Prefill 性能不足**：缺少 kernel fusion 和 subgroup matrix 支持，限制了 Prefill 效率。
- **部分硬件特性未充分利用**：如 `bf16`、`nvfp4` 等原生低精度格式无法直接支持。
- **浮点一致性问题**：不同设备间 f16 转换行为差异可能导致输出不一致。
- **移动端资源受限**：iOS Safari 内存上限 <500MB，限制了大模型部署。

### 未来工作方向
1. **引入 Kernel Fusion**：减少小内核频繁调用的开销，提升 Prefill 性能。
2. **Auto-tuning 与动态优化**：结合 GPTune 等框架，实现设备自适应参数调优。
3. **支持更多量化格式与 MoE 模型**：扩展对混合专家模型和新型量化方案的支持。
4. **推动 WebGPU 规范演进**：
   - 支持 `u16`、`u8` 等更细粒度类型以提升内存效率。
   - 引入更灵活的运行时参数传递机制（替代当前 slot 缓冲区设计）。
   - 优化安全检查机制，减少性能开销。

> **总结**：LlamaWeb 为浏览器内高效、可移植的 LLM 推理提供了坚实基础，不仅提升了性能与内存效率，还极大扩展了模型兼容性，是推动本地化、隐私保护 AI 应用落地的重要一步。

</details>

---

### 10. [Instant GPU Efficiency Visibility at Fleet Scale](https://arxiv.org/abs/2605.20799)

**Authors**: Connor Pedersen, Dong H. Ahn, Michel Migdal, Collin Neale, Nik Konyuchenko  
**Category**: cs.DC  
**Published**: 2026-05-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.20799v1  

#### Abstract
We present Overall FLOP Utilization (OFU), a hardware-level, precision-agnostic GPU efficiency metric for AI workloads on HPC systems, derived from two on-chip performance counters: Tensor Pipe Activity and SM clock frequency. OFU requires no application instrumentation and works across GPU generati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Instant GPU Efficiency Visibility at Fleet Scale*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在大规模 AI 高性能计算（HPC）系统中，**缺乏一种可扩展、无侵入、高精度的 GPU 利用率监控手段**。现有的 GPU 利用率测量方法存在以下局限：
- **Profiling 工具**（如 Nsight Compute、PyTorch Profiler）：需要应用级插桩，带来运行时开销，仅适用于单次分析，无法用于持续的全集群监控。
- **框架级 MFU（Model FLOPs Utilization）**：依赖手动推导的 FLOPs 公式，对新型模型架构（如 MoE、Mamba、多模态）极易出错且难以维护。
- **硬件计数器**：虽有潜力，但缺乏系统性的准确性验证与误差建模。

因此，**如何实现“即时、全量、精准”的 GPU 效率可见性**成为大规模 AI 基础设施管理的关键挑战。

### 提出了什么新方法或新思路
本文提出 **Overall FLOP Utilization (OFU)** —— 一种基于硬件性能计数器的、架构无关、精度无关的 GPU 浮点利用率度量方法。

**核心思想**：
- OFU 由两个硬件计数器直接计算得出：
  ```
  OFU = Tensor Pipe Activity × (SM Clock / Max SM Clock)
  ```
- **Tensor Pipe Activity**：衡量 GPU 执行 Tensor Core 指令的周期占比。
- **SM Clock 归一化**：将当前 SM 时钟频率归一化到最大频率，以反映时间维度上的实际吞吐损失。

该方法完全**无需修改应用程序或软件栈**，可部署于任何支持 DCGM 的 NVIDIA GPU 上。

### 相比现有方法的优势
| 维度 | Profiling 工具 | 框架级 MFU | **OFU（本文）** |
|------|----------------|-----------|----------------|
| **侵入性** | 高（需插桩） | 中（需集成） | **无（纯硬件）** |
| **覆盖范围** | 单任务 | 特定框架 | **所有任务、所有框架** |
| **精度可靠性** | 高（但依赖正确插桩） | 低（易因架构演进而出错） | **高（经实证验证）** |
| **部署成本** | 高 | 中 | **极低（DCGM 自动采集）** |
| **适用场景** | 性能调优 | 报告用途 | **持续监控、自动化优化** |

---

## 2. 核心实验方法和设置

### 使用的数据集与工作负载
- **控制实验**：使用 **GEMM（矩阵乘法）** 作为基准算子，在 H100 和 GB200 GPU 上测试不同精度（FP16、TF32、FP8、NVFP4）下的行为。
  - 矩阵尺寸从 128 到 16384，包含对齐与非对齐尺寸。
  - 使用 PyTorch `torch.matmul`、`torch._scaled_mm` 及内部 NVFP4 测试工具。
- **生产环境验证**：收集 **608 个真实训练作业**（H100 集群），涵盖多种配置（8–5888 GPUs）、用户（26人）和模型架构（包括 MoE、混合架构等）。

### 实验设置和评估指标
- **硬件平台**：NVIDIA H100 SXM、GB200 NVL。
- **数据采集**：
  - 使用 **DCGM** 收集 `PIPE_TENSOR_ACTIVITY` 和 `SM_CLOCK`。
  - 采样间隔为 30 秒（符合 DCGM 最佳实践）。
- **评估指标**：
  - **OFU**：硬件推导值。
  - **App MFU**：应用层报告的 MFU（来自 Megatron-LM + OneLogger）。
  - **相关性**：Pearson 相关系数 `r`。
  - **误差**：绝对误差（pp）、Mean Absolute Error (MAE)。
  - **覆盖率**：在 ±2pp、±5pp 内的样本比例。

### 基线方法对比
- **基线 1**：应用层 MFU（App MFU）
- **基线 2**：未校正的 OFU
- **改进版**：**Adjusted OFU** —— 引入 tile quantization 校正因子：
  ```
  OFU_adj = OFU × (2MNK / FLOPS_profiled)
  ```

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）控制实验（GEMM）结果
| GPU | 精度 | 方法 | MAE (pp) | ≤2pp 覆盖率 | ≤5pp 覆盖率 |
|-----|------|------|----------|------------|------------|
| H100 | FP16 | OFU | 1.90 | 64% | 96% |
| H100 | FP16 | **Adj OFU** | **0.06** | **100%** | **100%** |
| H100 | TF32 | OFU | 3.46 | 44% | 86% |
| H100 | TF32 | **Adj OFU** | **0.50** | **99%** | **99%** |
| GB200 | NVFP4 | OFU | 1.21 | 87% | 98% |
| GB200 | NVFP4 | **Adj OFU** | **1.15** | **95%** | **100%** |

✅ **结论**：经过 tile quantization 校正后，**OFU 可预测 App MFU 至 ≤2 pp 以内**，平均误差 < 0.6 pp。

#### （2）生产作业验证（608 jobs）
- **整体 Pearson r = 0.53**
- **排除异常作业后 r = 0.78**
- 平均绝对误差：6.2%
- **79.4% 的作业误差 < 10%**
- 大规模作业（>768 GPUs）误差普遍 < 5%

> 📌 **特别发现**：OFU 成功揭示了两个框架级 FLOPs 计算错误：
> 1. **MoE 下投影未计入** → 报告 MFU 54.27%，实际 OFU 25.58%（相对误差 112.2%）
> 2. **混合架构（Mamba + MoE）误按 Attention 计算** → 报告 MFU 24.51%，实际 OFU 15.56%（相对误差 57.5%）

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **OFU 是一个实用、可靠、可部署的硬件级 MFU 代理指标**：
   - 在控制实验中误差 ≤2 pp。
   - 在生产环境中与 App MFU 达成 r=0.78 的强相关性。
2. ✅ **应用层 MFU 易出错且不可靠**：
   - 框架公式难以跟上架构演进，导致严重高估。
   - OFU 可作为“真相来源”来检测并纠正这些错误。
3. ✅ **OFU 支持跨代、跨精度监控**：
   - 对 FP16、TF32、FP8、NVFP4 均有效。
   - 可用于追踪混合精度训练中的利用率变化。
4. ✅ **已产生实际运营价值**：
   - 在 **embodied agent 训练**中检测到 `TORCH_DISTRIBUTED_DEBUG` 导致的 2.5× 效率下降。
   - 在 **6,144 GPU 预训练任务**中，OFU 与 MFU 时间序列相关性达 r=0.977（按作业平均）。
   - 在 **world foundation model** 中发现激活重计算未被计入的问题。

### 方法的局限性
- **Tile Quantization Overhead**：小矩阵（N < 512）下误差可能超过 50%，但大模型训练中罕见。
- **TF32 特殊性**：cuBLAS 可能将其映射到 FP16 pipeline，导致更高 overhead。
- **SM Clock Sampling Noise**：瞬时采样引入微小噪声，但长期平均可忽略（< ±0.22 pp @ 30s 间隔）。
- **Non-Tensor Undercounting**：忽略 CUDA-core 运算（如 Softmax），但其占比通常 < 0.2%，可忽略。

### 未来工作方向
- 将 OFU 集成至更广泛的 **fleet-wide resilience 与 goodput 服务**，实现自动报警与优化建议。
- 推动硬件设计改进，例如提供 **hardware-averaged clock counter** 或 **native FLOPs counter**。
- 扩展至 **inference 场景** 和 **异构加速器**（如 TPUs、AMD GPUs）。
- 结合 OFU 与其他硬件信号（如 Memory Bandwidth、NVLink Utilization）构建更全面的效率诊断系统。

---

> **总结一句话**：  
> **OFU 提供了一种“即插即用”的方式，让大规模 GPU 集群获得即时、准确、无偏的浮点利用率可见性，是迈向高效 AI 基础设施运维的关键一步。**

</details>

---

### 11. [PlexRL: Cluster-Level Orchestration of Serviceized LLM Execution for RLVR](https://arxiv.org/abs/2605.20863)

**Authors**: Yiqi Zhang, Fangzheng Jiao, Tian Tang, Boyu Tian, Hangyu Wang, Qiaoling Chen, Guoteng Wang, Zhen Jiang, Peng Sun, Ping Zhang, Xiaohe Hu, Ziming Liu, Menghao Zhang, Yanmin Jia, Yang You, Siyuan Feng  
**Category**: cs.DC  
**Published**: 2026-05-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.20863v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has recently unlocked strong reasoning capabilities in large language models (LLMs), triggering rapid exploration of new algorithms and data. However, RLVR training is notoriously inefficient: long-tailed rollouts, tool-induced stalls, and asymme...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《PlexRL: Cluster-Level Orchestration of Serviceized LLM Execution for RLVR》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前基于 **Reinforcement Learning with Verifiable Rewards (RLVR)** 的大语言模型（LLM）训练存在严重的**集群级资源利用率低下**问题。尽管已有多种优化策略（如同步流水线、异步Rollout、共置部署等），但在单个任务内部仍存在大量因以下原因导致的**空闲时间（idle time）**：
- **长尾Rollout行为**（如工具调用延迟）
- **Rollout与Training阶段的资源需求不对称**
- **相位交替执行**（phase alternation）造成的设备闲置

这些问题源于现有系统将算法控制与模型执行紧密耦合，并以“任务隔离”方式管理资源，无法跨任务复用空闲周期。

---

### **提出了什么新方法或新思路**
提出 **PlexRL** —— 一种面向 RLVR 工作负载的**集群级统一 LLM 服务化执行运行时系统**，其核心思想是：

#### ✅ **解耦算法控制与模型执行**
- 将 Rollout 和 Training 抽象为远程服务调用（remote service），由中心化的调度器统一管理。
- 用户只需编写 RL 控制逻辑，无需关心底层模型部署细节。

#### ✅ **构建共享的 LLM 执行服务层**
- 多个 RLVR 任务共享一组大型 LLM 副本（Worker Process Group, WPG）。
- 利用不同任务之间的**空闲周期反相关性**，在时间维度上对多个任务进行**多路复用（multiplexing）**，填补彼此的空闲间隙。

#### ✅ **支持细粒度调度与状态管理**
- 引入 **Scheduler** 进行集群范围内的放置与调度决策。
- 设计 **StateManager** 统一管理模型状态在 GPU HBM、Host DRAM 和 NVMe 之间的迁移与驻留，降低上下文切换开销。

---

### **相比现有方法的优势**
| 方面 | 现有方法局限 | PlexRL 改进 |
|------|--------------|------------|
| **Split Deployment** | 各阶段独占设备池，交替执行导致严重空转 | 共享训练资源池，跨任务填充空闲期 |
| **Colocated Deployment** | Rollout 被迫使用训练所需的大规模 DP，造成低效长尾 | 可独立配置小规模 DP Rollout，提升 MFU |
| **Asynchronous Rollout** | 仅能部分缓解阶段不匹配，引入策略滞后（policy lag）风险 | 不依赖强异步即可实现高利用率，开发者可选择保守同步策略 |
| **系统灵活性** | 框架绑定执行流程，难以扩展新算法 | 开放接口，支持灵活组合新型 RL 流程 |

> ✅ **核心优势总结**：  
> PlexRL 在不牺牲算法灵活性的前提下，通过**集群级资源复用**显著提升了有效集群容量，降低了 GPU 小时成本。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 使用一个**专有的数学问题数据集**，难度相当于 AIME，包含约 45,000 个样本。
- 用于端到端 RLVR 训练验证。

---

### **模型与架构**
实验覆盖三种不同规模和结构的模型：
| 模型 | 类型 | 参数量 | 并行策略 |
|------|------|--------|----------|
| Qwen2.5-7B-Instruct | Dense | 7B | TP=2, PP=4, DP=2 |
| Qwen3-30B-A3B-Thinking | MoE | 30B | EP=8, TP=2, PP=8, DP=8 |
| Qwen3-235B-A22B-Instruct | MoE | 235B | EP=8, TP=8, PP=12, DP=1 |

> 所有实验均启用 ZeRO-2；235B 模型额外启用 ZeRO-offload。

---

### **实验设置**
- **硬件平台**：2048-GPU 集群，基于 Kubernetes 管理。
- **部署模式对比**：
  1. **Colocated Baseline**：Rollout 与 Training 共置在同一组 GPU 上。
  2. **Split-Async Baseline**：分离部署 + 异步执行（允许一步滞后）。
  3. **PlexRL**：Rollout 使用私有 GPU，Training GPU 构成共享池，两个任务共享训练资源（time-sliced）。

---

### **评估指标**
| 指标 | 定义 | 目标 |
|------|------|------|
| **GPU-hours per effective training step** | 总消耗 GPU 时间 / 完成的训练步数 | 越低越好 |
| **MFU (Model FLOPs Utilization)** | 实际计算利用率 | 越高越好 |
| **Decoding Throughput per GPU** | 解码吞吐率 | 越高越好 |
| **Queueing Delay & Makespan** | 作业排队延迟与总完成时间 | 越短越好 |
| **Bubble Ratio** | 非计算时间占比（空泡比例） | 越低越好 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **端到端 GPU 小时节省**
在相同训练步数下，PlexRL 相比 Split-Async 基线显著减少 GPU 消耗：

| 模型 | GPU 小时节省 |
|------|---------------|
| 7B | **31.36%** |
| 30B | **30.10%** |
| 235B | **37.58%** ✅（最高）|

> 💡 **说明**：模型越大，传统共置部署中 Rollout 的 DP 规模越不合理，PlexRL 的收益越高。

---

#### 🔹 **解码吞吐效率对比（235B 模型）**
| 部署方式 | 解码吞吐效率（AUC ratio） |
|---------|--------------------------|
| Colocated (Large DP) | 52.74% |
| PlexRL (Small DP) | **75.03%** |

> ⬆️ 提升超过 22 个百分点，表明 PlexRL 避免了大规模 DP 导致的**计算未饱和**问题。

---

#### 🔹 **气泡比例分析（Bubble Ratio）**
| 模型 | Bubble Ratio |
|------|--------------|
| 7B | 80.10% |
| 30B | 70.67% |
| 235B | 81.11% |

> 表明即使在先进系统中，**超过 70% 的周期处于非计算状态**，凸显了资源复用的巨大潜力。

---

#### 🔹 **集群级调度仿真结果（Trace-driven）**
在三个月真实作业轨迹回放中比较四种调度策略：

| 策略 | 总完成时间（相对 Isolated） | 优势 |
|------|----------------------------|------|
| Isolated（基线） | 100% | ❌ 排队延迟重 |
| Pack | ~70% | 中等改善 |
| Spread | ~65% | 减少相位冲突 |
| **Spread+Backfill** | **56.0%** ✅ | 最优，压缩延迟最明显 |

> 相当于**同等集群可承载约 1.8 倍的工作负载**。

---

### **消融实验（隐含分析）**
虽然未明确列出消融表，但文中通过设计分析揭示了关键机制的作用：
- **Affinity-aware placement** 显著减少模型迁移频率。
- **HRRS 调度算法** 有效缓解 Head-of-Line 阻塞与频繁上下文切换。
- **StateManager 的分层驻留与预取机制** 成功将状态操作移出关键路径。

---

## **4. 关键结论和发现**

### **主要发现**
1. **RLVR 的低效本质是结构性而非局部性问题**  
   单任务内不可避免的空闲周期，在集群视角下具有**跨任务互补潜力**。

2. **服务化 + 集群调度是解锁高效训练的关键路径**  
   将 LLM 执行抽象为共享服务，使原本“浪费”的空闲时间变为**全局可调度资源**。

3. **无需牺牲算法灵活性即可获得高性能**  
   PlexRL 在保持用户编程自由的同时，实现了接近最优的资源利用。

4. **大模型尤其受益于跨任务复用**  
   模型越大，共置部署带来的 Rollout 浪费越严重，PlexRL 的增益越显著。

---

### **方法的局限性**
- **冷启动开销**：首次部署仍需完整加载模型，短期任务可能难以受益。
- **状态一致性复杂性**：跨任务共享模型需精细管理状态版本与同步。
- **调度延迟敏感场景受限**：若某任务对延迟极度敏感，可能不适合被抢占。
- **依赖高质量执行轨迹预测**：Warm Start 依赖历史 trace，新类型任务需先经历 Cold Start。

---

### **未来工作方向**
1. **动态自适应调度策略**：根据实时负载自动调整 Packing 强度。
2. **支持更多并行范式融合**：如结合 Pipeline Parallelism 的细粒度切片。
3. **引入 SLO-aware 调度**：为不同优先级任务提供服务质量保障。
4. **扩展至其他多阶段训练场景**：如 Pretrain → SFT → RLHF 全流程协同调度。
5. **探索更激进的状态共享机制**：如参数子空间复用、轻量化影子副本等。

---

## ✅ **总结一句话**
> **PlexRL 通过将 LLM 执行服务化并实施集群级细粒度调度，成功将 RLVR 训练中的“不可避免空闲”转化为“可复用资源”，在不影响算法灵活性的前提下，最高降低 37.58% 的 GPU 小时成本，为大规模推理与智能体训练提供了高效的系统基础。**

</details>

---

### 12. [Optimized Federated Knowledge Distillation with Distributed Neural Architecture Search](https://arxiv.org/abs/2605.21322)

**Authors**: Chaimaa Medjadji, Sylvain Kubler, Yves Le Traon, Guilain Leduc, Sadi Alawadi, Feras M. Awaysheh  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.21322v1  

#### Abstract
Federated Learning (FL) enables collaborative model training without centralizing data. However, real-world deployments must simultaneously address statistical heterogeneity across client data (non-IID), system heterogeneity in device capabilities, and communication efficiency. Existing FL approache...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimized Federated Knowledge Distillation with Distributed Neural Architecture Search

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Federated Learning (FL)** 方法在面对现实世界部署时面临三大挑战的权衡：
- **统计异质性 (Statistical heterogeneity)**：客户端数据通常是非独立同分布（non-IID），导致模型漂移（client drift）和不稳定收敛。
- **系统异质性 (System heterogeneity)**：客户端设备在计算能力、内存、能耗等方面差异巨大。
- **通信效率 (Communication efficiency)**：传输完整模型参数在带宽受限或功耗敏感的场景下成本高昂。

现有方法大多假设客户端使用**固定的统一模型架构**，这限制了对不同数据复杂度和硬件约束的适应性，导致准确率与效率之间的次优权衡。

### 提出的新方法：FedKD-NAS
本文提出了 **FedKD-NAS**，一个将**客户端侧神经网络架构搜索 (Neural Architecture Search, NAS)** 与**服务器协调的知识蒸馏 (Knowledge Distillation, KD)** 相结合的新型 FL 框架。

其核心思想是将**模型架构**本身视为一个首要的优化变量，而不仅仅是模型参数。

#### 主要创新点
1.  **去中心化的 NAS 机制 (Decentralized NAS)**：
    - 每个客户端在每一轮训练中，从一个预定义的轻量级搜索空间中，自主选择一个最适合其本地数据分布和资源约束（如 CPU、内存）的模型架构。
    - 避免了全局架构同步，降低了开销，提高了可扩展性。

2.  **基于预测的协作协议 (Prediction-based Collaboration)**：
    - 客户端不共享模型权重，而是仅向服务器上传其在公共参考数据集 `DPub` 上的软预测（soft predictions/logits）。
    - 这消除了异构模型间的参数共享问题，实现了功能对齐。

3.  **稳定的知识转移机制 (Stabilized Knowledge Transfer)**：
    - 服务器聚合所有客户端的预测，并与一个固定的教师模型（Teacher Model）的输出进行融合，生成一个更稳定、偏差更小的“共识”目标。
    - 该目标经过指数移动平均（EMA）平滑处理后，广播给所有客户端作为下一轮蒸馏的监督信号，有效缓解了非 IID 数据下的客户端漂移。

4.  **统一的效率框架 (Unified Efficiency Framework)**：
    - 提出了四个复合评估指标（RES, PQS, CES, UES）来综合衡量模型的准确性、通信效率和资源消耗，为部署决策提供全面视角。

### 相比现有方法的优势
- **更高的 Pareto 效率**：在准确率、通信开销和客户端资源消耗之间取得了更好的平衡。
- **更强的鲁棒性**：在非 IID 条件下表现显著优于基线，准确率下降幅度最小。
- **更低的通信开销**：通过 logits 交换，通信量相比参数聚合方法减少了高达 44×。
- **更低的客户端资源消耗**：通过自适应的轻量化架构，CPU 使用率降低约 28%。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在六个数据集上进行，涵盖了图像、时间序列等多种模态：
- **图像分类**：MNIST, FMNIST, EMNIST, CIFAR-10, CIFAR-100
- **时间序列（传感器）**：CASA (Human Activity Recognition)

每个数据集都模拟了三种数据划分方式以测试异质性：
- **IID**：独立同分布
- **Dirichlet-based non-IID (α=0.1)**：基于狄利克雷分布的标签偏斜
- **Shard-based extreme label skew**：极端标签偏斜

### 实验设置和评估指标
- **客户端模型**：根据数据集特性选用不同的轻量级骨干网络（如 LeNet5, ResNet18, MobileNetV2, ShuffleNetV2, DeepConvLSTM）。
- **服务器教师模型**：在 `DPub` 上预训练的固定模型。
- **轮数**：100 轮（CIFAR-100）或 30 轮（其他数据集）。
- **评估指标**：
    - **原始指标**：准确率 (Acc)、损失 (Loss)、CPU 使用率、内存 (RAM)、通信量 (Comm)。
    - **复合指标**：
        - **RES (Resource Efficiency Score)**：综合 CPU 和内存的资源效率，越低越好。
        - **PQS (Performance Quality Score)**：综合准确率和损失的性能质量，越高越好。
        - **CES (Communication Efficiency Score)**：通信效率，越高越好。
        - **UES (Unified Efficiency Score)**：整合 PQS、CES 和 RES 的最终统一效率得分，越高越好。

### 基线方法对比
与六种代表性的 FL 基线方法进行了对比：
- **参数聚合类**：FedAvg, Ditto, Local-KD
- **知识蒸馏类**：FedMD, FedDF, FedDistill

这些基线覆盖了 FL 的主要范式，确保了比较的全面性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
FedKD-NAS 在绝大多数配置下均取得了最优或接近最优的性能，尤其是在非 IID 场景下优势显著。

- **准确率提升**：在非 IID 条件下，相比最强的蒸馏基线（如 FedDistill），准确率提升高达 **7-11%**；相比中位数基线，提升 **18-23%**。
- **通信效率**：通信开销相比 FedAvg 减少了 **44×**（MobileNetV2）和 **7×**（ShuffleNetV2）。
- **资源消耗**：客户端 CPU 使用率降低约 **28%**（从 ~47% 降至 ~34%）。
- **统一效率 (UES)**：在 35 个评估配置中，有 31 个配置下 UES 排名第一，且优势明显（例如，在 CIFAR-10 MobileNetV2 上比第二名高 1.5-4×）。

### 与基线方法的对比结果
- **与 FedAvg 对比**：在 IID 下，FedAvg 准确率略高；但在非 IID 下，FedKD-NAS 准确率反超，同时在通信和资源效率上全面占优。
- **与蒸馏基线 (FedDistill, FedMD) 对比**：在相同的通信效率（CES）下，FedKD-NAS 通过自适应架构获得了显著更高的准确率（PQS）和更低的资源消耗（RES），从而实现了更高的 UES。
- **鲁棒性**：FedKD-NAS 的准确率从 IID 到非 IID 的下降幅度最小（仅约 3-4%），而 Ditto 和 Local-KD 等方法下降超过 35-43%。

### 消融实验与关键发现
虽然没有明确的消融实验表格，但讨论部分通过分析揭示了各组件的贡献：
- **NAS 控制器**：是获得更高 PQS 的主要原因，它使模型能适应本地数据。
- **知识蒸馏 (KD)**：是实现通信效率（高 CES）的关键。
- **服务器教师模型 (Teacher Anchor)**：提供了稳定性，防止在非 IID 下性能崩溃。
- **EMA 平滑**：控制了蒸馏目标的波动，增强了收敛稳定性。

---

## 4. 关键结论和发现

### 主要发现
1.  **架构灵活性至关重要**：将模型架构作为优化变量，而非固定不变，是解决 FL 中统计和系统异质性问题的根本途径。
2.  **异质性可以成为优势**：FedKD-NAS 的性能会随着数据异质性的增加而增强。多样化的客户端数据为 NAS 提供了更丰富的信号，有助于发现泛化能力更强的架构。
3.  **通信效率具有条件性**：基于 logits 的蒸馏并非总是通信最优。当类别数 `C` 很大而模型很小（如 EMNIST）时，logits 的大小可能超过模型参数，此时参数聚合反而更高效。这是一个重要的实践启示。
4.  **综合评估的必要性**：单一指标（如准确率）无法反映真实部署的全貌。提出的 UES 指标能够有效区分不同方法在实际场景中的综合表现。

### 方法的局限性
1.  **依赖公共数据集 (DPub)**：需要一个与私有数据域匹配的小型公共参考数据集用于蒸馏。如果存在严重的域不匹配，性能可能会下降。
2.  **高类别数场景的通信劣势**：如 EMNIST 所示，当类别数很大时，logits 通信的带宽优势会消失甚至逆转。
3.  **隐私风险**：尽管只传输 logits，但仍可能受到成员推断攻击等威胁。
4.  **拜占庭鲁棒性**：框架对恶意客户端提交虚假预测缺乏形式化的防御保证。
5.  **绝对能耗未测量**：实验使用代理指标，缺乏直接的硬件能耗和碳排放测量。

### 未来工作方向
1.  **无数据依赖**：用合成数据或生成模型替代公共参考数据集 `DPub`。
2.  **高类别数优化**：结合稀疏化、量化等技术，恢复在高类别数任务上的通信优势。
3.  **增强隐私保护**：引入差分隐私（Differentially Private）机制来保护共享的预测。
4.  **提升鲁棒性**：集成抗拜占庭攻击的聚合规则（如裁剪均值）。
5.  **绿色AI评估**：使用 CodeCarbon 等工具进行精确的碳足迹和能耗核算。
6.  **动态架构扩展**：研究在任务复杂度高时动态扩大架构容量的策略。

</details>

---

### 13. [EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design](https://arxiv.org/abs/2605.19743)

**Authors**: Gioele Molinari, Florian Felten, Soheyl Massoudi, Mark Fuge  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.19743v1  

#### Abstract
Large Language Model (LLM) agents are increasingly applied to engineering design tasks, yet existing evaluation frameworks do not adequately address multi-agent systems that combine simulation, retrieval, and manufacturing preparation. We introduce a benchmark suite with three evaluation dimensions:...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前的 **LLM agents** 在工程设计任务中的应用日益增多，但现有的评估框架（如 AgentBench、ToolSandbox）主要关注通用工具调用或多轮对话能力，**无法有效评估多智能体系统在复杂工程流程中的表现**，尤其是在以下方面存在不足：
- 缺乏对 **simulation、retrieval 和 manufacturing preparation** 等多阶段协同的支持；
- 未涵盖 **条件分支（conditional branching）、语义消歧（semantic disambiguation）、工作记忆（working memory）** 等认知挑战；
- 无法隔离 **Retrieval-Augmented Generation (RAG)** 对参数选择的实际贡献；
- 缺少对 **High Performance Computing (HPC) 集群上端到端 ML 训练编排** 的评估。

### **提出了什么新方法或新思路**
本文提出两个核心贡献：

#### **(1) ENGIAI：一个基于 LangGraph 的 Multi-Agent System (MAS) 参考实现**
- 采用 **分层 supervisor 架构**，由一个中央 supervisor agent 路由请求至七个专业化 agent；
- 各 agent 分别负责：拓扑优化（Engineering）、文档检索（RAG/ArXiv/Search）、HPC 作业提交（HPC）、本地命令执行（CLI）、3D 打印控制（Prusa）等；
- 支持状态保持、条件路由、人机交互中断和检查点持久化，提升可扩展性和鲁棒性。

#### **(2) 一套全新的 Benchmark Suite**
该套件包含三个维度的评估任务，覆盖工程设计全流程：
| 维度 | 内容 |
|------|------|
| **Workflow Benchmark** | 设计七种 prompt style，测试不同认知需求下的任务完成率（如直接工具使用、语义消歧、条件分支、派生计算、多导出跟踪） |
| **RAG Benchmark** | 引入 **gated scoring 机制**，仅当 agent 显式调用 `search_documents` 工具时才为参数准确性赋分，防止模型依赖先验知识“猜中”答案 |
| **HPC Benchmark** | 测试 agent 是否能在 SLURM 集群上完成完整的 ML 训练流水线（生成脚本 → 提交 → 监控 → 评估），共四步 |

### **相比现有方法的优势**
| 方面 | 优势说明 |
|------|----------|
| **综合性更强** | 是首个同时覆盖 **仿真、检索、制造准备、HPC 编排** 的 LLM agent benchmark |
| **评估更精细** | 提出 **gated RAG scoring**，能准确衡量 retrieval 的独立贡献，避免“记忆 vs 检索”的混淆 |
| **贴近真实场景** | 包含长周期、多步骤、高成本的 HPC 训练任务，暴露出现有 agent 在长期指令跟随上的退化问题 |
| **模块化设计** | ENGIAI 框架支持通过添加新 agent 或 tool API 扩展功能，便于社区复用与迭代 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 基于 **EngiBench** 提供的两个标准工程问题：
  - **Beams2D**：二维悬臂梁拓扑优化，最小化 compliance，约束体积分数（volfrac）、力作用位置（forcedist）、滤波半径（rmin）
  - **Photonics2D**：二维光子器件逆向设计，最大化电磁场重叠（total_overlap），跨物理域迁移测试

### **实验设置**
- **LLM Backends（4个）**：
  - Proprietary: `gpt-5-mini`, `gemini-3-flash`
  - Open-source (via Ollama): `qwen3-4b`, `qwen3.5-4b`（均为 4B 参数量级）
- **每组配置运行 15 次**（3 种随机种子 × 5 个样本），确保统计稳健性
- 所有调用设置 `temperature=0` 并固定 seed 以增强可复现性

### **评估指标**
#### **(1) Workflow Evaluation**
综合得分公式如下：
$$
S_{\text{workflow}} = 0.65 \cdot S_{\text{design}} + 0.20 \cdot S_{\text{tool}} + 0.15 \cdot S_{\text{completion}}
$$
其中：
- $S_{\text{design}}$：设计质量（65%权重），包含 IoU、像素准确率、目标匹配、约束满足、连通性、水密性等子项
- $S_{\text{tool}}$：工具调用效率（20%），正确调用数 / max(最优, 实际)
- $S_{\text{completion}}$：任务完成率（15%），是否成功调用所有必需工具且参数符合要求

#### **(2) RAG Evaluation**
引入 **gated scoring**：
- 仅当 agent 调用 `search_documents` 时，其参数提取才能获得分数；
- 若未调用 retrieval 工具，即使猜对也得分为 0；
- 设置三种条件：
  - **RAG-on**：正常检索
  - **RAG-off**：移除检索工具
  - **Empty RAG**：索引为空，测试是否盲目信任空结果

#### **(3) HPC Training Evaluation**
主要指标为加权复合得分：
$$
S_{\text{HPC}} = 0.70 \cdot S_{\text{step}} + 0.15 \cdot S_{\text{config}} + 0.15 \cdot S_{\text{eval}}
$$
- $S_{\text{step}}$：四个步骤（生成、提交、监控、评估）的完成比例
- $S_{\text{config}}$：训练配置是否正确
- $S_{\text{eval}}$：能否成功提取并报告评价指标（MMD, DPP, RVC, IOG/COG/FOG）

### **基线方法对比**
- **Proprietary vs Open-source LLMs**：比较闭源大模型与开源小模型的表现差异
- **跨代对比**：`qwen3-4b` vs `qwen3.5-4b`，观察同规模下代际进步
- **Prompt Style 对比**：分析不同认知难度 prompt 下的任务完成率变化

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) Workflow Performance（Beams2D）**
| Prompt Style | GPT-5-mini (TC) | Gemini-3-Flash (TC) | Qwen3-4B (TC) | Qwen3.5-4B (TC) |
|------------|------------------|--------------------|---------------|----------------|
| FULL       | 1.00             | 0.93               | 0.00          | 0.73           |
| NATURAL    | 0.87             | 1.00               | 0.00          | 0.33           |
| W-RAND     | 1.00             | 1.00               | 1.00          | 1.00           |
| W-DERIVED  | 1.00             | 1.00               | 0.47          | 0.85           |
| W-DISTRACT | 1.00             | 1.00               | 1.00          | 0.93           |
| **W-COND** | **0.93**         | **0.87**           | **0.40**      | **0.60**       |
| W-MULTI    | 0.93             | 1.00               | 1.00          | 1.00           |
| **Average TC** | **0.96**     | **0.97**           | **0.55**      | **0.78**       |

> ✅ **闭源模型平均任务完成率达 96–97%，而开源 4B 模型为 55–78%**  
> 🔺 **条件分支（W-COND）是最难的任务，性能下降最显著**

#### **(2) Photonics2D 泛化能力**
| Prompt Style | Best Model (TC) |
|-------------|-----------------|
| W-RAND      | 1.00            |
| W-DISTRACT  | 1.00            |
| **W-COND**  | **0.53 (Gemini)** |

> ⚠️ 即使是最好的模型，在 Photonics2D 上的 W-COND 任务中也只能达到 **53% 完成率**，远低于 Beams2D 的 87–93%，表明 **条件推理能力受领域熟悉度影响极大**

#### **(3) RAG Evaluation**
| Condition       | Score (Avg across prompts) |
|----------------|----------------------------|
| **RAG-on**      | ~0.95–1.0                   |
| **RAG-off**     | 0.0                         |
| **Empty RAG**   | <0.2                        |

> ✅ **验证了 gated scoring 的有效性**：只有真正使用 retrieval 才能得分  
> 🔺 表明 agent 几乎完全依赖检索内容，而非凭记忆作答（除个别常见值如 volfrac=0.35 外）

#### **(4) HPC Orchestration**
| Model           | Step Completion Rate (Explicit) | Step Completion Rate (Natural) |
|----------------|----------------------------------|----------------------------------|
| **Gemini-3-Flash** | 100%                            | 100%                            |
| **GPT-5-mini**     | 70%                             | 50%                             |

> ⚠️ GPT-5-mini 在自然语言描述下，最终评估步骤失败率高达 50%，显示其 **长期指令跟随能力不稳定**

---

## **4. 关键结论和发现**

### **主要发现**
1. **RQ1 (Workflow Performance)**：
   - 当前主流闭源 LLM（GPT-5-mini, Gemini-3-Flash）在结构化工程流程中表现优异（平均 TC 96–97%）
   - 开源 4B 模型仍有明显差距，但 **qwen3.5-4b 相比 qwen3-4b 有显著代际提升**（55% → 78%）
   - **条件分支（conditional branching）是最大瓶颈**，尤其在陌生领域（如 Photonics2D）中 TC 降至 20–53%

2. **RQ2 (Model Robustness)**：
   - 闭源模型间表现稳定；开源模型则表现出较大波动，提示对 prompt 敏感
   - 代际改进可在一定程度上弥补规模劣势

3. **RQ3 (Tool Usage Efficiency)**：
   - 尽管部分模型能完成任务，但存在 **冗余工具调用**（如 Qwen3-4B 多次重复调用 `simulate_design`）
   - 过多调用会降低综合得分（CO），尽管设计质量不变 → **效率直接影响评分**

4. **RQ4 (RAG Improvement)**：
   - **RAG 是工程参数选择的关键**，无检索时得分趋近于零
   - gated scoring 成功隔离了 retrieval 的贡献，证明该机制有效

5. **RQ5 (HPC Orchestration)**：
   - **Gemini-3-Flash 能 100% 完成整个 HPC 流程**
   - GPT-5-mini 存在严重退化，特别是在自然语言指令下，说明其 **缺乏稳定的长期状态追踪能力**

### **方法的局限性**
- **问题范围有限**：仅测试 Beams2D 和 Photonics2D，未覆盖 EngiBench 全部问题
- **LLM 数量受限**：仅评测 4 个模型，且因成本未测试更大规模开源模型（如 70B）
- **缺少人类干预研究**：未进行真实工程师参与的人类反馈实验
- **HPC 实验仅限闭源模型**：因耗时过长，未将开源模型纳入 HPC 测试
- **未提供单 agent 基线**：缺乏对 supervisor 架构本身的消融分析

### **未来工作方向**
1. **扩大 benchmark 覆盖面**：
   - 加入更多 EngiBench 问题、更大规模开源模型（Llama3, Mistral）
   - 探索跨 family 的泛化能力

2. **增强 RAG 评估难度**：
   - 构建更大、噪声更多的文档集合
   - 引入对抗性 retrieval（多个冲突来源），测试证据整合能力

3. **改进 agent 架构与训练**：
   - 使用 benchmark 产生的 trace 数据微调小型模型
   - 引入 structured chain-of-thought 提升条件推理能力
   - 设计参数计划解耦机制，提升语义消歧能力

4. **深化 HPC 自主性**：
   - 从“编排预定义脚本”升级为“自主编写并调试训练代码”
   - 引入显式状态检查点机制，缓解长流程指令遗忘

5. **探索工具生态扩展**：
   - 结合 MCP 协议接入更多工程软件
   - 研究 agent 性能在工具数量增长时的可扩展性

---

> 📌 **总结一句话**：  
> 本文构建了首个面向 **LLM-driven 工程设计全链路** 的多智能体框架与基准测试体系，揭示了当前 LLM 在 **条件推理、长期编排、检索依赖** 等方面的关键瓶颈，并为下一代智能设计系统提供了可量化的发展路径。

</details>

---

### 14. [Parallel LLM Reasoning for Bias-Resilient, Robust Conceptual Abstraction](https://arxiv.org/abs/2605.20194)

**Authors**: Aisvarya Adeseye, Jouni Isoaho, Adeyemi Adeseye  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.20194v1  

#### Abstract
Large language models (LLMs) have been increasingly used to analyze text. However, they are often plagued with contextual reasoning limitations when analyzing long documents. When long documents are processed sequentially, early or dominant concepts can overshadow less visible but meaningful interpr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Parallel LLM Reasoning for Bias-Resilient, Robust Conceptual Abstraction**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对 **Large Language Models (LLMs)** 在长文本分析中的两大结构性缺陷：
- **Cumulative Analytical Bias（累积分析偏差）**：当文本被顺序分块处理时，早期或主导概念会持续影响后续推理，导致次要但重要的主题被忽略（即“**lost in the middle**”现象），造成遗漏错误（omission error）、过度泛化和位置主导效应。
- **Ungrounded Synthesis（无依据综合）**：在分块独立推理后合并结果时，若缺乏严格的证据约束，容易引入冗余、概念漂移（conceptual drift）和不支持的断言（unsupported claims），降低可追溯性和可信度。

### **提出了什么新方法或新思路**
提出了一种名为 **Parallel Evidence-Constrained Independent Inference (PECII)** 的结构化框架，其核心思想是：
- **并行独立推理（Parallel Independent Inference）**：将长文本划分为语义连贯的 chunks，并**并行地、互不依赖地**对每个 chunk 进行 LLM 推理，从而消除顺序执行带来的累积偏差。
- **证据锚定整合（Evidence-Anchored Consolidation）**：在合并阶段，强制要求每个提取的概念必须有明确的文本引证（quote），并通过语义对齐、多样性约束和去重机制进行验证，确保输出可追溯、可靠且一致。

### **相比现有方法的优势**
- **优于顺序处理（Sequential Chunking）**：避免了早期 chunk 对后续推理的“锚定效应”，显著减少遗漏和主导偏差。
- **优于端到端全文本处理（Full-Transcript Execution）**：克服了上下文窗口限制，同时通过结构化设计提升整体分析质量。
- **优于简单合并策略**：通过显式的证据验证和整合规则，大幅降低幻觉（hallucination）和概念泄漏（cross-theme leakage）。
- **方法论优先于模型规模**：研究表明，良好的推理架构设计（如 PECII）能显著缩小小模型与大模型之间的性能差距，使小模型也能达到接近大模型的可靠性。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **数据来源**：来自 82 名参与者的半结构化访谈转录本（semi-structured interview transcripts）。
- **主题**：组织环境中引入游戏化机制（gamification）所引发的隐私担忧。
- **文本长度**：每份转录本约 8,000–13,000 词，远超多数 LLM 的上下文窗口。
- **预处理**：完全匿名化，去除所有个人身份信息，并由两名独立研究者使用 NVivo 辅助编码，构建人类专家金标准（human-derived gold standard）。

### **实验设置**
- **三种执行策略对比**：
  1. **Direct Full-Transcript Execution**：整篇文档一次性输入（受限于 token 上限）。
  2. **Sequential Chunk Execution**：分块顺序处理，前一块输出作为下一块输入的一部分。
  3. **Parallel Chunk Execution (PECII)**：边界感知分块 + 并行独立推理 + 证据锚定整合。
- **模型选择**：共测试六种 LLM，涵盖不同架构与参数规模：
  - 小型模型：LLaMA-1B, Qwen-1.5B
  - 中型模型：LLaMA-3B, Qwen-4B, LLaMA-8B
  - 大型商用模型：ChatGPT 5.2
- **统一设置**：所有模型使用相同的 chunking 策略、相似度阈值、证据约束和整合流程，**不做模型特定提示调优**，以公平比较结构影响。

### **评估指标**
| 指标 | 缩写 | 含义 |
|------|------|------|
| 遗漏错误率 | Eom (%) | 未被提取的真实概念占比 |
| 早期块主导指数 | ECDI | 来自前段 chunk 的提取概念比例 |
| 解释新颖性评分 | INS (1–5) | 主题深度与创造性（专家盲评） |
| 证据可追溯性得分 | ETS (%) | 具备多源充分证据的主题比例 |
| 不支持断言率 | UCR (%) | 无有效引用或语义对齐的断言占比 |
| 主题压缩比 | TCR | 原始候选数 / 最终主题数（反映抽象效率） |
| 跨主题泄漏率 | CTL (%) | 被分配至多个主题的片段比例 |
| 合并理由质量 | MJQ (1–5) | 合并逻辑清晰度（专家盲评） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **表1：不同执行策略下的偏差与解释质量（部分摘要）**
| 模型 | 执行方式 | 遗漏错误 ↓ | ECDI ↓ | 解释新颖性 ↑ |
|------|----------|------------|--------|--------------|
| LLaMA-1B | Baseline | 36.8% | 0.51 | 3.1 |
| LLaMA-1B | Sequential | 23.5% (-36%) | 0.32 (-37%) | 3.8 (+22.6%) |
| LLaMA-1B | **PECII (Parallel)** | **5.9% (-84%)** | **0.09 (-82%)** | **4.7 (+51.6%)** |

> ✅ **平均而言，PECII 相比基线减少遗漏错误约 84%，降低早期主导效应超过 80%。**

#### **表2：证据锚定整合效果（部分摘要）**
| 模型 | 证据可追溯性 ↑ | 不支持断言率 ↓ | 主题压缩比 ↑ | 合并理由质量 ↑ |
|------|----------------|----------------|---------------|------------------|
| LLaMA-1B | 41% → **95% (+131.7%)** | 34% → **3% (-91.2%)** | 6.2 → 14.8 | 2.8 → 4.8 |
| ChatGPT | 63% → **96% (+52.4%)** | 9% → **2% (-77.8%)** | 8.6 → 15.3 | 4.1 → 4.8 |

> ✅ **证据锚定使 traceability 提升最高达 130%，unsupported claims 减少高达 91%。**

### **与基线方法的对比结果**
- **与 Full-Transcript 相比**：
  - PECII 在所有模型上均显著优于直接处理完整文本，尤其在小模型上优势更明显。
- **与 Sequential Chunking 相比**：
  - PECII 在 omission error、ECDI、ETS、UCR 等关键指标上全面领先。
  - Sequential 方法虽优于 baseline，但仍存在明显的顺序依赖和累积偏差。
- **跨模型一致性增强**：
  - 在 baseline 下，模型间性能差异较大（如 omission error 差 14%）；
  - 在 PECII 下，模型间方差下降 >80%，**小模型性能大幅提升，逼近大模型水平**。

### **消融实验结果**
虽然未单独列出“消融实验”章节，但从多维度分析中可视为隐式消融：
- **仅 chunking（Sequential）**：带来约 36–40% 的遗漏减少，但仍有显著偏差。
- **加入并行推理（PECII）**：进一步将遗漏减少至 ~84%，证明**并行性是偏差缓解的关键**。
- **加入证据锚定**：traceability 和 UCR 显著改善，说明**证据约束是抑制幻觉的核心机制**。
- **最终整合策略**：TCR 提高、CTL 下降、MJQ 上升，表明结构化合并提升了抽象质量和边界清晰度。

---

## **4. 关键结论和发现**

### **主要发现**
1. **推理结构比模型规模更重要**：
   - 方法论设计（如并行性、证据锚定）对提升 LLM 分析可靠性的影响，**超过单纯增加模型参数**。
   - 结构优化可使小型 LLM 接近甚至媲美大型模型的表现。

2. **并行推理有效打破累积偏差**：
   - 通过隔离 chunk 的推理过程，彻底切断 autoregressive conditioning 的递归依赖，显著减少“early dominance”和“omission”。

3. **证据锚定实现可审计的综合**：
   - 强制引用 + 语义对齐 + 多样性约束，使得最终输出具备高度 traceability 和 auditability，极大降低 hallucination。

4. **结构收敛现象（Structural Convergence）**：
   - 在 PECII 框架下，不同架构和规模的模型在多项指标上趋于一致，表明该框架具有**强正则化作用（regularization effect）**。

### **方法的局限性**
- **计算资源需求更高**：并行执行带来更高的并发内存消耗和调度开销。
- **严格过滤可能误删罕见解释**：过于严苛的证据验证可能排除一些真实但引用不佳的边缘观点。
- **依赖高质量 chunking**：若语义分块不合理（如割裂关键论述），仍会影响局部推理质量。
- **人工干预成本**：尽管自动化程度高，但在复杂领域仍需 human-in-the-loop 审核关键合并决策。

### **未来工作方向**
1. **扩展至更多领域**：测试 PECII 在政策、法律、临床叙事等领域的适用性。
2. **改进证据验证机制**：引入基于 entailment 的 verifier 或专用 SML 检查器，提高验证精度。
3. **动态 chunking 策略**：研究自适应分块策略，根据不同文本风格调整 chunk 大小与边界。
4. **人机协同变体**：开发 human-in-the-loop 版本，在整合阶段允许专家审核证据链接。
5. **资源优化研究**：量化并行化部署的成本效益，探索适用于资源受限环境的轻量级版本。

---

> 📌 **一句话总结**：  
> **PECII 通过“并行独立推理 + 证据锚定整合”的结构化设计，显著提升了 LLM 在长文本概念抽象中的可靠性、可追溯性和抗偏能力，证明了方法论架构在 LLM 应用中比单纯扩大模型规模更具决定性作用。**

</details>

---

### 15. [Cloud-Native Operation of Roadside Infrastructure Enabling Demand-Driven Collective Perception via V2X](https://arxiv.org/abs/2605.21145)

**Authors**: Lukas Zanger, Fabian Thomsen, Guido Linden, Jean-Pierre Busch, Lennart Reiher, Lutz Eckstein  
**Category**: cs.DC  
**Published**: 2026-05-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.21145v1  

#### Abstract
Intelligent roadside infrastructure is a key enabler for cooperative intelligent transport systems (C-ITS), supporting vehicles equipped with automated driving systems (ADS), e.g., through enhanced environment perception. With a growing number and an expanding functional scope of roadside units, sca...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Cloud-Native Operation of Roadside Infrastructure Enabling Demand-Driven Collective Perception via V2X*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
随着智能道路基础设施（如配备传感器的 **roadside units**）在 **Cooperative Intelligent Transport Systems (C-ITS)** 中的广泛应用，其部署规模不断扩大，带来了以下挑战：
- **资源浪费**：计算密集型应用（如基于机器学习的感知服务）若持续运行，会造成不必要的能源消耗、硬件磨损和通信信道拥塞。
- **运维复杂性**：大量分布式设备需要统一的软件部署、监控与生命周期管理。

传统“始终开启”的模式难以满足能效与可扩展性的需求。

---

### 🚀 提出的新方法与创新思路
本文提出了一种 **云原生（cloud-native）架构 + 需求驱动（demand-driven）编排策略**，用于高效运营路边基础设施：

#### （1）**云原生架构设计**
- 构建一个跨多个 **sRISUs**（stationary Roadside ITS Station Units）和中心服务器的 **Kubernetes 集群**。
- 使用轻量级发行版 **K3s** 实现边缘环境下的容器编排。
- 所有 C-ITS 应用以 **containerized microservices** 形式部署，支持自动化发布、配置、监控与故障恢复。

#### （2）**需求驱动的服务编排机制**
- 只有当附近有联网车辆（connected vehicle）进入指定区域时，才动态部署高负载服务（如 lidar 驱动、object detection）。
- 触发条件为接收到车辆通过 **ITS-G5** 发送的 **CAM（Cooperative Awareness Message）**。
- 编排框架由 **event detector** 和 **application manager** 组成，实现事件检测 → 部署请求 → Helm Chart 安装的闭环控制。

#### （3）代表性应用场景：V2X-based Collective Perception
- 当车辆接近路口时，启动 **Live Digital Twin** 流水线：
  - 多个 sRISU 融合感知对象 → 生成 **CPM（Collective Perception Message）** → 广播给车辆。
- 无车经过时，系统保持休眠状态，显著降低能耗。

---

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| **部署方式** | 固定部署、常驻运行 | 动态按需部署（on-demand） |
| **资源利用** | 持续占用 GPU/CPU 资源 | 仅在有需求时激活 |
| **运维效率** | 分散管理，OTA 更新困难 | 统一集群管理，支持自动扩缩容 |
| **能效表现** | 高功耗、高信道负载 | 显著节能，减少 channel congestion |
| **可扩展性** | 扩展成本高 | 支持大规模市政级部署 |

> ✅ **核心优势**：将成熟的 **cloud-native 技术栈**（K8s + Helm + Prometheus/Grafana）成功迁移至 **C-ITS 边缘场景**，并结合 **V2X 事件触发机制** 实现绿色、高效的基础设施运营。

---

## 2. **核心实验方法和设置**

### 📊 数据集与数据来源
- **真实世界 V2X 数据记录**：
  - 在亚琛工大测试场 **RITA（Roadside Infrastructure Testfield Aachen）** 连续采集一周（2026年2月）的 **ITS-G5 CAM 消息**。
  - 共计 **69,610 条 CAM 消息**，来自 **714 个唯一 station ID**（即不同车辆）。
- **实验验证数据**：
  - 使用研究车辆 **karl.** 主动发送 CAM，触发系统响应。
  - 同步记录 **ROS 2 bag 日志、容器日志、GPU 功耗、时间戳**等用于延迟分析。

---

### ⚙️ 实验设置
- **硬件部署**：
  - 4 个 sRISU 布置于城市交叉口，配备 lidar、PTRZ camera、V2X 单元及带 GPU 的计算单元。
  - 所有节点通过 **5G VPN** 互联，构成分布式 K8s 集群。
  - 中心服务器作为 **control plane** 节点，运行 Kubernetes 控制组件与编排框架。
- **软件栈**：
  - K3s 作为 Kubernetes 发行版。
  - Helm 管理微服务部署模板。
  - ROS 2 + Zenoh 作为中间件进行跨节点通信。
  - Grafana + Prometheus + Loki 实现监控与日志聚合。

---

### 🎯 评估指标
| 指标类别 | 具体指标 |
|--------|--------|
| **功能性能** | End-to-end latency（从 CAM 发送到 CPM 接收） |
| **系统开销** | Deployment latency（Pod 创建、服务启动）、各阶段延迟分解 |
| **通信性能** | ITS-G5 CAM/CPM 传输延迟 |
| **能效潜力** | GPU 额外功耗、每日非活跃时间、年化节能量估算 |
| **可行性** | 是否能在车辆到达前完成部署（geofence 设计依据） |

---

### ❌ 基线方法对比
本文未直接与其他编排框架（如 Docker Swarm 或裸机部署）进行横向对比，而是采用 **“always-on” vs “on-demand”** 的对照逻辑来凸显节能效果：

- **Baseline（隐含）**：假设所有感知服务始终运行。
- **Proposed Method**：仅在检测到 CAM 后才启动相关 microservices。

通过实际测量两种模式下的 **GPU 功耗差异** 和 **服务空闲时长**，间接证明所提方法的优越性。

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据

#### （1）端到端延迟（End-to-End Latency）
| 实验轮次 | 总延迟（s） | 部署延迟占比 |
|--------|------------|-------------|
| Run 1  | 12.346     | ~80%        |
| Run 2  | 12.457     | ~77%        |
| Run 3  | 12.602     | ~81%        |

> ✅ 结果稳定，平均约 **12.5 秒**，满足城市交通场景需求。

#### （2）部署延迟分解（Run 1）
| 步骤 | 延迟（s） |
|------|----------|
| 创建 Object Detection Pod | 6.099 |
| 第一次检测回调 | 0.370 |
| 生成首个 object list | 3.374 |
| **合计（主路径）** | **9.843** |
| Lidar Driver Pod 创建（并行） | ~4.15 ×2 |
| Lidar 数据接收 | ~0.64 ×2 |

> 🔍 **瓶颈分析**：Kubernetes Pod 创建是最大延迟来源（冷启动开销），尤其是 **object detection 服务**（依赖大型 ML 模型）。

#### （3）ITS-G5 通信延迟（均值）
| 通信方向 | 平均延迟 | 最小 | 最大 |
|--------|--------|------|------|
| CAM (karl. → sRISU4) | 8.17 ms | 3.22 ms | 22.91 ms |
| CPM (sRISU4 → karl.) | 5.25 ms | 0.19 ms | 11.79 ms |

> ✅ 通信延迟极低，符合 **low-latency C-ITS 应用要求**。

---

### 💡 能效潜力估算（基于一周 V2X 记录）

| 参数 | 数值 |
|------|------|
| 每日平均 CAM 出现分钟数 | 18 min |
| 加上 1 分钟缓冲期后激活时间 | ~33 min/day（≈2% 时间） |
| 非活跃时间 | **1407 分钟/天** |
| 单个 sRISU 额外 GPU 功耗（运行时） | +45W |
| 四个 sRISU 总额外功耗 | +180W |
| 每分钟额外能耗 | 3 Wh |
| **每日可避免能耗** | **~4.22 kWh** |
| **年化节能量（测试场）** | **~1500 kWh/year** |

> 🌍 对比参考：相当于德国一个单人家庭的年用电量 [33]。

#### 规模化推演：
- 若部署 **100 个类似 sRISU**，在相同交通密度下：
  - 年额外能耗可达 **~37.5 MWh**。
  - 表明 **demand-driven 策略对大规模部署至关重要**。

---

### 🔍 消融实验（Ablation Study）
虽未明确命名“消融实验”，但文中进行了关键因素分析：

| 因素 | 发现 |
|------|------|
| **Pod 冷启动** | 是主要延迟来源，尤其 object detection > lidar driver |
| **时间同步误差** | 存在 ±20ms 不确定性，影响中间延迟精度 |
| **地理围栏（geofence）设计** | 当前实验未设 geofence，但可根据 13s 延迟 + 10s 规划窗口反推出合理范围（约 300 米半径） |

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **技术可行性已验证**：
   - 云原生架构可在真实道路环境中稳定运行。
   - Kubernetes 成功管理跨边缘节点的微服务部署与监控。

2. **需求驱动编排切实可行**：
   - 系统可在车辆进入通信范围后及时启动感知流水线。
   - **12.5 秒内完成从 CAM 接收到 CPM 广播**，足以支撑城市驾驶决策。

3. **节能潜力巨大**：
   - 在中等交通流量下，系统 **98% 时间处于空闲状态**。
   - 可避免数千 kWh 的年能耗，对可持续发展具有重要意义。

4. **通信能力支持该范式**：
   - ITS-G5 的 **800 米通信距离**远超所需触发距离（~300 米），具备充足安全裕度。

---

### ⚠️ 局限性
1. **冷启动延迟较高**：
   - 当前 Pod 启动耗时较长（尤其 ML 模型加载），限制了实时性。
   - 未使用预热（warm-up）、镜像优化或 Serverless 技术缓解此问题。

2. **依赖特定硬件配置**：
   - 节能估算基于高功耗 GPU 设备，若使用低功耗 SoC，收益可能下降。

3. **市场渗透率影响触发频率**：
   - 当前 **ITS-G5 车辆覆盖率有限**，导致实际需求较少；未来普及后价值更大。

4. **未考虑多应用并发调度**：
   - 当前仅处理单一应用（collective perception），复杂场景下需更精细的资源协调机制。

---

### 🔮 未来工作方向
1. **集成更多 roadside units** 到 Kubernetes 集群，提升覆盖范围与可扩展性。
2. **优化冷启动性能**：
   - 探索轻量化容器镜像、ROS 2 生命周期管理改进、函数即服务（FaaS）模型。
3. **开发更智能的 geofence 策略**：
   - 基于历史轨迹与预测算法动态调整激活区域。
4. **拓展其他 on-demand V2X 应用**：
   - 如 demand-driven data logging、事件触发式信号灯优化等。
5. **引入 dependability-aware 编排层**：
   - 如文献 [16] 所述，增强故障检测与服务连续性保障。

---

## ✅ 总结
本论文首次将 **cloud-native 架构** 与 **V2X 触发的 demand-driven 编排** 相结合，成功实现了路边基础设施的 **绿色、智能、可扩展运营**。实验证明该方法不仅技术可行，且在 **延迟可控的前提下大幅节省能源**，为未来大规模 C-ITS 部署提供了重要实践参考。

</details>

---

### 16. [Introspective X Training: Feedback Conditioning Improves Scaling Across all LLM Training Stages](https://arxiv.org/abs/2605.20285)

**Authors**: Brandon Cui, Ximing Lu, Jaehun Jung, Syeda Nahida Akter, Hyunwoo Kim, Yuxiao Qu, David Acuna, Shrimai Prabhumoye, Yejin Choi, Prithviraj Ammanabrolu  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.20285v1  

#### Abstract
We tackle the question of how to scale more efficiently across the many, ever-growing stages of current LLM training pipelines. Our guiding intuition stems from the fact that the dynamics of later stages of the pipeline, e.g. post-training, can be used to inform earlier stages such as pre-training. ...

---

### 17. [Distributed Direct Preference Optimization](https://arxiv.org/abs/2605.20696)

**Authors**: Zhanhong Jiang  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.20696v1  

#### Abstract
Preference-based reinforcement learning (RL) is a key paradigm for aligning policies with human judgments, yet its theoretical behavior in distributed settings where preference data are fragmented across heterogeneous users remains poorly understood. Direct Preference Optimization (DPO) avoids expli...

---

### 18. [Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption](https://arxiv.org/abs/2605.19593)

**Authors**: Mert Yildiz, Pietro Spadaccino, Alexey Rolich, Francesca Cuomo, Andrea Baiocchi  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.19593v1  

#### Abstract
Modern deployments of Large Language Models (LLMs) increasingly require serving multiple models with diverse architectures, sizes, and specialization on shared, heterogeneous hardware. This setting introduces new challenges for resource allocation, dispatching, and scheduling, particularly under GPU...

---

### 19. [CogScale: Scalable Benchmark for Sequence Processing](https://arxiv.org/abs/2605.19758)

**Authors**: Yannis Bendi-Ouis (Mnemosyne), Romain de Coudenhove (ENS-PSL), Xavier Hinaut (Mnemosyne)  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.19758v1  

#### Abstract
The ability to maintain and manipulate information over time is a fundamental aspect of living beings and Artificial Intelligence. While modern models have achieved remarkable success in tasks like natural language processing, evaluating the capacity of novel architectures to process sequential info...

---

### 20. [From SGD to Muon: Adaptive Optimization via Schatten-p Norms](https://arxiv.org/abs/2605.19781)

**Authors**: Thomas Massena (IRIT, DTIPG - SNCF, UT3), Corentin Friedrich (IRIT), Mathieu Serrurier (IRIT)  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.19781v1  

#### Abstract
Modern optimizers, like Muon, impose matrix-wise geometry constraints on their updates. These matrix-wise constraints can be unified under Linear Minimization Oracle (LMO) theory. However, all current methods impose fixed LMO geometries for the update rules, chosen by-design or empirically, which ar...

---

### 21. [Cross-lingual robustness of LLM-brain alignment and its computational roots](https://arxiv.org/abs/2605.21049)

**Authors**: Ni Yang, Rui He, Philipp Homan, Iris Sommer, Davide Staub, Wolfram Hinzen  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.21049v1  

#### Abstract
Large language models (LLMs) reliably predict neural activity during language comprehension and transformer depth has been interpreted as mirroring hierarchical cortical organization. However, it remains unclear whether such alignment extends to subcortical regions, overlaps spatially across languag...

---

### 22. [TextReg: Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization](https://arxiv.org/abs/2605.21318)

**Authors**: Lucheng Fu, Ye Yu, Yiyang Wang, Yiqiao Jin, Haibo Jin, B. Aditya Prakash, Haohan Wang  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.21318v1  

#### Abstract
Large language models (LLMs) are highly sensitive to the prompts used to specify task objectives and behavioral constraints. Many recent prompt optimization methods iteratively rewrite prompts using LLM-generated feedback, but the resulting prompts often become longer, accumulate narrow sample-speci...

---

### 23. [Decomposing MXFP4 quantization error for LLM reinforcement learning: reducible bias, recoverable deadzone, and an irreducible floor](https://arxiv.org/abs/2605.20402)

**Authors**: Xiaocan Li, Shiliang Wu, Zheng Shen  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.20402v1  

#### Abstract
MXFP4 arithmetic can dramatically accelerate reinforcement learning (RL) post-training of large language models (LLMs), yet the quantization error introduces severe accuracy degradation. Existing work treats the quantization error as a monolithic noise term, missing the distinct mechanisms upon inte...

---

### 24. [Unsupervised clustering and classification of upper limb EMG signals during functional movements: a data-driven](https://arxiv.org/abs/2605.20599)

**Authors**: L. F. Salazar \'Alvarez, D. Escobar-Saltar\'en, M. B. Salazar S\'anchez, S. C. Henao-Aguirre  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.20599v1  

#### Abstract
This study presents a comprehensive approach for the clustering and classification of upper-limb surface electromyography (sEMG) signals during functional reach and grasp movements. The methodology was applied to the NINAPRO DB4 dataset, which provides multichannel EMG recordings of 52 gestures. A f...

---

### 25. [Choose Wisely and Privately: Proactive Client Selection for Fair and Efficient Federated Learning](https://arxiv.org/abs/2605.20975)

**Authors**: Adda Akram Bendoukha, Heber Hwang Arcolezi, Nesrine Kaaniche, Aymen Boudguiga  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.20975v1  

#### Abstract
Federated Learning enables collaborative model training across decentralized data sources without data transfer. Averaging-based FL is limited by the presence of non-IID data, which negatively impacts convergence speed and final model accuracy. Conventional alternatives suffer from significant ineff...

---

### 26. [Efficient Learning of Deep State Space Models via Importance Smoothing](https://arxiv.org/abs/2605.21108)

**Authors**: John-Joseph Brady, Nikolas Nusken, Yunpeng Li  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.21108v1  

#### Abstract
Latent state space systems are ubiquitous in statistical modelling, arising naturally when a time series is observed through a noisy measurement function, however training deep state space models (DSSM) at scale remains difficult. Two largely distinct strategies and literatures have developed around...

---

### 27. [Divide-Prompt-Refine: a Training-Free, Structure-Aware Framework for Biomedical Abstract Generation](https://arxiv.org/abs/2605.20628)

**Authors**: Sylvey Lin, Joe Menke, Shufan Ming, Dongin Nam, Neil Smalheiser, Halil Kilicoglu  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.20628v1  

#### Abstract
Biomedical abstracts play a critical role in downstream NLP applications, such as information retrieval, biocuration, and biomedical knowledge discovery. However, a non-trivial number of biomedical articles do not have abstracts, diminishing the utility of these articles for downstream tasks. We pro...

---

### 28. [GraphRAG on Consumer Hardware: Benchmarking Local LLMs for Healthcare EHR Schema Retrieval](https://arxiv.org/abs/2605.20815)

**Authors**: Peter Fernandes, Ria Kanjilal  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.20815v1  

#### Abstract
Graph-based Retrieval Augmented Generation (GraphRAG) extends retrieval-augmented generation to support structured reasoning over complex corpora, but its reliability under resource-constrained, privacy-sensitive deployments remains unclear. In healthcare, where Electronic Health Record (EHR) data i...

---

### 29. [NanoCP: Request-Level Dynamic Context Parallelism for Data-Expert Parallel Decoding](https://arxiv.org/abs/2605.21100)

**Authors**: Jiefei Chen, Binbin Lin, Jinming Ma, Jiangfei Duan, Haojie Duanmu, Hao Liu, Qinxiu Cheng, Xiuhong Li, Zhilin Pei, Hui Wang, Xingcheng Zhang, Dahua Lin  
**Category**: cs.DC  
**Published**: 2026-05-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.21100v1  

#### Abstract
Modern serving systems for Mixture-of-Experts (MoE) models adopt hybrid data-expert parallelism: expert parallelism (EP) shards experts across GPUs to scale capacity, while data parallelism (DP) replicates attention layers across instances to process independent requests. Existing systems bind each ...

---

### 30. [Frontier: Towards Comprehensive and Accurate LLM Inference Simulation](https://arxiv.org/abs/2605.21312)

**Authors**: Yicheng Feng, Xin Tan, Yangtao Deng, Yimin Jiang, Yibo Zhu, Hong Xu  
**Category**: cs.DC  
**Published**: 2026-05-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.21312v1  

#### Abstract
Modern LLM serving is no longer homogeneous or monolithic. Production systems now combine disaggregated execution, complex parallelism, runtime optimizations, and stateful workloads such as reasoning, agents, and RL rollouts. Simulation is attractive for exploring this growing design space, yet exis...

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

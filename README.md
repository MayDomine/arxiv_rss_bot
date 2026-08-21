# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-21 06:07:54 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM Serving](https://arxiv.org/abs/2608.19758)

**Authors**: Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, Ran He  
**Category**: cs.CL  
**Published**: 2026-08-21  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2608.19758v1  

#### Abstract
Long-context modeling is a pivotal capability for Large Language Models, yet the quadratic complexity of attention remains a critical bottleneck, particularly during the compute-intensive prefilling phase. Our previous work, FlashPrefill, mitigates this cost through instantaneous pattern discovery a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM Serving

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLM）在长上下文建模中面临自注意力机制的**二次复杂度瓶颈**，尤其是在计算密集的 **prefill 阶段**。虽然已有稀疏注意力方法尝试缓解该问题，但存在以下不足：

- **精度不可控**：在极端稀疏下性能严重下降。
- **系统级部署不兼容**：依赖连续 KV 缓存布局，无法与现代推理框架（如 SGLang）中的 **paged KV cache** 和 **continuous batching** 集成。
- **内核效率落后**：基于 FlashAttention-2 的实现，未能利用 Hopper 架构 GPU 上的最新优化（如 TMA、异步流水线），导致稀疏带来的理论加速被低效内核抵消。

---

### **提出了什么新方法或新思路**

FlashPrefill V2 将前作 FlashPrefill 从一个算法原型推进为可生产部署的长上下文服务方案，提出三大核心创新：

#### ✅ **1. 均值校正项（Mean Correction Term）**
- 在 **softmax 计算中补偿被剪枝块的概率质量**，通过聚合被丢弃块的 K/V 统计量（均值）作为代理贡献。
- 实现零阶近似恢复，显著抑制极端稀疏下的精度损失，即使在 <5% 密度下仍能保持接近全注意力的性能。

#### ✅ **2. 对齐 FlashAttention-3/4 的稀疏注意力内核**
- 全面重构稀疏注意力算子，适配 Hopper 架构 GPU，关键技术包括：
  - **PackGQA 内存访问**：提升 GQA 下的内存共享效率。
  - **Warp Specialization**：分离 producer/consumer 流水线，解耦数据加载与计算。
  - **Pingpong Pipelining**：在 MMA 与非 MMA 操作间重叠执行，提升利用率。
  - **FP8 支持**：支持 FP8 推理，满足实际量化需求。

#### ✅ **3. 原生支持生产级推理系统特性**
- **原生支持 paged KV cache** 和 **continuous batching**。
- 可直接作为 **attention backend** 集成进 SGLang 等现代推理框架，无需修改模型定义或调度逻辑。

---

### **相比现有方法的优势**

| 维度 | FlashPrefill V2 | 现有稀疏方法（如 FlashPrefill、XAttention） |
|------|------------------|---------------------------------------------|
| **精度控制** | 引入均值校正，极端稀疏下精度损失可控（<1.8 pts） | 无补偿机制，稀疏加剧时精度骤降 |
| **内核效率** | 对齐 FA3/4，利用 TMA、异步流水线，最大化硬件利用率 | 多基于 FA2，未充分利用 Hopper 特性 |
| **部署兼容性** | 支持 paged KV + continuous batching，可集成至 SGLang | 通常假设连续 KV，难以部署 |
| **量化支持** | 支持 FP8 推理 | 多数不支持或仅支持 BF16/FP16 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **RULER**：用于评估长上下文检索与推理能力，在受控序列长度（4K–128K）下测试。
- **LongBench**：多任务、双语长上下文基准，涵盖问答、摘要、代码等 21 个任务，更贴近真实场景。

### **实验设置和评估指标**

#### **硬件平台**
- 主要测试平台：**NVIDIA H20 GPU**（广泛部署的推理加速器）
- 软件栈：CUDA 12.9, PyTorch 2.9.1, CUTLASS 4.3, SGLang 0.5.10

#### **评估指标**
- **RULER / LongBench 任务得分**（如 RULER Score）
- **算子级加速比**：相对于 FlashAttention-2 和 FA3/4 对齐的稠密基线
- **端到端延迟**：
  - **TTFT（Time to First Token）**
  - **TPOT（Time Per Output Token）**
  - **吞吐量（tokens/s, req/s）**
- **注意力密度**（Selected Blocks Ratio）

#### **基线方法对比**
- **Full Attention**：完整注意力作为性能上限
- **MInference**、**FlexPrefill**、**XAttention**：主流训练无关稀疏注意力方法
- **FlashAttention-2 (FA2)**：标准稠密注意力基线
- **FA3/4-aligned Dense**：与本工作相同执行模型的稠密基线，用于公平比较稀疏收益

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **算子级加速（128K 序列长度，batch=4）**
| 精度 | 加速比（vs FA2） | 加速比（vs FA3/4 稠密基线） |
|------|------------------|----------------------------|
| **BF16** | **27.19×** | 17.54× |
| **FP8** | **47.26×** | **30.49×** |

> 即使在较短序列（4K）下，仍能保持加速（FP8 达 2.7×）。

#### ✅ **端到端 TTFT 降低（SGLang 部署）**
在 128K 上下文、batch=16 场景下，**TTFT 最高降低 4.8×**：
- **Qwen3-30B-A3B**：从 123.2s → **25.5s**（FP8）

#### ✅ **注意力密度**
随着序列增长，密度显著下降：
| 序列长度 | 4K | 8K | 16K | 32K | 64K | 128K |
|----------|-----|-----|-----|-----|-----|-------|
| **密度** | ~70% | ~50% | ~30% | ~18% | ~9% | **~4.6%** |

> 极低密度是实现高加速比的关键。

---

### **与基线方法的对比结果**

#### 📊 **RULER 性能（Llama-3.1-8B, 128K）**
| 方法 | RULER Score | 相对 Full 的差距 |
|------|-------------|------------------|
| Full | 73.82 | — |
| FlashPrefill V2 (BF16) | **72.08** | -1.74 |
| FlashPrefill V2-FP8 | **67.78** | -6.04 |
| FlashPrefill (V1) | 70.91 | -2.91 |
| XAttention | 72.33 | -1.49 |

> V2 在保持更高精度的同时实现更大加速。

#### 📈 **LongBench 平均得分**
| 模型 | Full | V2 (BF16) | V2-FP8 |
|------|------|-----------|--------|
| Llama-3.1-8B | 49.76 | **49.31** | **48.76** |
| Qwen3-30B-A3B | 51.63 | **50.73** | **50.29** |

> V2 是所有稀疏方法中表现最佳者，平均仅落后 Full Attention <1 pt。

---

### **消融实验结果**

#### 🔍 **均值校正（Mean Correction）的作用（Qwen3-4B, RULER, 128K）**
| 方法 | RULER Score | 相对修正版差距 |
|------|------------|----------------|
| V2 完整（BF16） | **69.76** | — |
| 无校正 | 68.86 | -0.90 |
| V2-FP8 完整 | **68.97** | — |
| 无校正 | 62.78 | **-6.19** |

> 结论：校正对 **FP8 更重要**，可挽回高达 6.2 分的精度损失。

#### 📉 **选择阈值 α 的影响（FP8, 64K）**
| α | 密度 | 有校正 | 无校正 | 差距 |
|----|------|--------|--------|------|
| 0.2 | 5.2% | **80.12** | 74.68 | -5.44 |
| 0.1 | 9.2% | 80.59 | 76.86 | -3.73 |
| 0.0125 | 23.6% | 80.75 | 78.02 | -2.73 |

> 均值校正使系统能在更低密度下运行而不牺牲精度，**扩展了可用稀疏范围**。

---

## 4. 关键结论和发现

### **主要发现**

1. **均值校正在极端稀疏下至关重要**，尤其在 **FP8 量化场景**，可防止高达 6+ 分的精度崩溃。
2. **对齐 FA3/4 的内核设计** 是释放稀疏潜力的前提，否则硬件效率瓶颈会吞噬稀疏增益。
3. **系统级兼容性（paged KV + continuous batching）** 是稀疏注意力走向生产的必要条件。
4. **FP8 + 稀疏** 可实现 **协同加速**，在 128K 上达到 **47.26× vs FA2** 的惊人速度。
5. 端到端延迟显著降低，**TTFT 最高压缩至 1/4.8**，极大改善用户体验。

---

### **方法的局限性**

- **仅适用于 prefill 阶段**：decode 阶段因单 token 输入，无法有效应用块稀疏。
- **chunked prefill 会削弱优势**：若启用分块预填充（chunked prefill），需每块重新执行索引选择，增加开销。
- **对检索类任务仍有轻微损失**：尽管已大幅缩小差距，但在极端稀疏下，极小概率的重要 token 仍可能被漏检。

---

### **未来工作方向**

- 扩展至 **decode 阶段的稀疏化**（如 query-aware KV 压缩）。
- 探索 **动态调整稀疏率**（per-layer 或 per-sequence）以平衡精度与速度。
- 结合 **其他加速技术**（如 speculative decoding、KV cache 压缩）构建端到端高效推理栈。
- 进一步优化 **FP8 校准策略**，减少量化误差对稀疏模式的影响。

---

> 💡 **总结**：  
> FlashPrefill V2 成功将稀疏注意力从“算法探索”推向“生产部署”，通过 **精度保障**、**硬件对齐** 和 **系统兼容** 三位一体的设计，在长上下文 LLM 服务中实现了前所未有的效率突破，为大规模模型的实际落地提供了强有力的技术支撑。

</details>

---

### 2. [Write Once, Run Everywhere: The Axon DSL for Shape-Safe and Framework-Agnostic LLM Architectures](https://arxiv.org/abs/2608.19889)

**Authors**: Jacob Nielsen, Danial Namazifard, Lukas Galke Poech, Peter Schneider-Kamp  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.19889v1  

#### Abstract
The entire ecosystem of open-source language models effectively relies on a single platform. What if this platform was forced to shut down tomorrow? Implementing and maintaining efficient model definitions and translating them between different training and inference regimes is a resource-heavy task...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前大语言模型（LLM）生态系统严重依赖少数中心化平台（如 Hugging Face Hub），导致以下问题：
- **实现漂移（implementation drift）**：不同框架（如 PyTorch、JAX、MLX）之间的手动移植容易引入错误，优化难以同步。
- **技术债务高**：模型代码被大量胶水代码和框架特定逻辑包裹，难以迁移和维护。
- **部署锁定（deployment lock-in）**：研究人员必须在“性能”和“可移植性”之间做出权衡。
- **训练/推理偏差**：训练与推理阶段的实现不一致可能导致行为差异。

### **提出的新方法**
作者提出了 **Axon** —— 一种**强类型、领域专用语言（DSL）**，支持“一次编写，处处运行”（Write Once, Run Everywhere）的 LLM 架构定义。

#### **核心创新点**
- **框架无关的模型规范**：Axon 不是针对某个框架（如 PyTorch）的 API 封装，而是一种独立的语言，用于描述模型的数学结构。
- **强类型与形状安全（Shape-Safe）**：内置符号维度（symbolic dimensions），确保张量操作的形状在编译期即被验证，避免运行时错误。
- **统一编译管道**：从单一 `.axon` 定义文件出发，通过一个编译器管道自动生成多个主流框架的独立实现：
  - PyTorch
  - PyTorch + Triton
  - JAX
  - MLX
  - vLLM
- **基于图的中间表示（Graph IR）**：所有后端共享同一个 Graph IR 合同，确保优化不会因后端不同而丢失。

### **相比现有方法的优势**
| 方面 | 现有方法（如 Transformers） | Axon |
|------|-----------------------------|------|
| **可移植性** | 需为每个框架重写或适配 | 一次定义，多框架生成 |
| **一致性** | 易出现实现漂移 | 所有后端源自同一 IR，保证一致性 |
| **性能** | 依赖框架默认调度，存在 Python 开销 | 编译生成扁平函数，消除模块调用开销 |
| **审计性** | 代码复杂，难于审查 | 定义简洁，可审计性强 |

---

## **2. 核心实验方法和设置**

### **实验设置**
- **模型范围**：共测试 **204 个检查点（checkpoints）**，覆盖 **60 个模型家族**，参数规模从 **135M 到 32B**。
- **硬件环境**：
  - **GPU**：NVIDIA B200（180GB HBM3）
  - **Apple Silicon**：M3 Max（36GB RAM）
- **精度**：主要使用 `bfloat16`，部分模型因精度问题回退到 `float32`。
- **编译配置**：所有模型均启用 `torch.compile` 或等效的 JIT 编译以公平比较。

### **评估指标**
- **吞吐量（Throughput）**：生成 token 数 / 花费时间（tok/s）
- **运行时比率（Runtime Ratio）**：$ p = t_{\text{Axon}} / t_{\text{Transformers}} $，$ p < 1 $ 表示 Axon 更快。
- **Top-1 Token Parity**：确保 Axon 与 Transformers 生成完全相同的 token 序列。
- **训练行为一致性**：比较损失曲线是否重合。

### **基线方法对比**
- **主基线**：Hugging Face `Transformers` 库中的官方实现。
- **对比后端**：
  - PyTorch
  - PyTorch + Triton
  - JAX
  - MLX
  - vLLM

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **自动回归生成（Autoregressive Generation）**
| 后端 | 模型规模 | 检查点数 | Axon 更快比例 | **中位数加速比** | 平均加速比 |
|------|----------|----------|----------------|------------------|------------|
| **PyTorch** | <4B | 76 | 63% | **0.903×**（9.7% 更快） | 0.925× |
| **Triton** | <4B | 75 | 80% | **0.843×**（15.7% 更快） | 0.878× |
| **JAX** | <4B | 74 | 86% | **0.481×**（51.9% 更快） | 1.122× |
| **PyTorch** | 4–32B | 87 | 63% | **0.987×**（1.3% 更快） | 0.980× |
| **Triton** | 4–32B | 87 | 77% | **0.924×**（7.6% 更快） | 0.931× |
| **JAX** | 4–32B | 68 | 81% | **0.589×**（41.1% 更快） | 0.981× |

> ✅ **总体趋势**：Axon 在 **76%（<4B）** 和 **73%（4–32B）** 的检查点上达到或优于 Transformers 性能。

#### **vLLM 推理性能**
| 模型规模 | 检查点数 | Axon 更快比例 | **中位数加速比** |
|----------|----------|----------------|------------------|
| 全部（135M–30B） | 88 | 74% | **0.631×**（**58% 中位数加速**） |

> 🚀 当部署为原生 vLLM 架构（含 PagedAttention 和 KV-cache）时，Axon 实现 **58% 中位数吞吐提升**。

#### **MLX（Apple Silicon）性能**
| 指标 | 数值 |
|------|------|
| 检查点数 | 126 |
| Axon 更快比例 | 95% |
| **中位数加速比** | **0.483×**（**翻倍速度**） |
| 最大加速 | 达 **4.5×**（如 Monad、Gemma3-270M） |

### **消融实验与归因分析**
虽然论文未提供显式的消融实验表，但在 **“Python Overhead”** 分析中明确指出了性能来源：
- **Transformers 的开销**：
  - 每层 `nn.Module.__call__` 引入 3 层 Python 调用。
  - `generate()` 循环包含 `prepare_inputs_for_generation`、`LogitsProcessorList` 等额外步骤。
- **Axon 的优势**：
  - 编译器将模块层次结构**扁平化为单个函数**，消除 `nn.Module` 调度开销。
  - 权重访问由属性链改为直接字典查找。
  - `generate` 循环简化为 4 步（vs Transformers 的 6 步）。
- **实证支持**：
  - Qwen-0.5B 上，Axon-Torch 的 `nn.Module.__call` 调用从 **35,616 次降至 0**。
  - GPU 活跃时间几乎不变，但 CPU 空闲时间减少，说明瓶颈在 Python 层。

---

## **4. 关键结论和发现**

### **主要发现**
1. **“一次编写，处处运行”是可行的**：Axon 成功实现了跨 PyTorch、JAX、MLX、vLLM 等多个后端的高性能生成。
2. **性能提升显著且广泛**：
   - 在 JAX 和 MLX 上表现尤为突出（中位数提速 50%~100%）。
   - vLLM 部署下中位数提速 **58%**。
3. **行为完全一致**：几乎所有模型都实现了 **100% Top-1 Token Parity**，证明语义正确性。
4. **训练兼容**：Axon 生成的 PyTorch 模型在训练时表现出与 Transformers 完全相同的损失曲线，且步长时间快 **9.6%**。
5. **性能来源于结构优化**：加速主要来自**消除 Python 调度开销**，而非更优的底层内核。

### **局限性**
- **仅限语言模型**：目前实验集中在 LLM，未涵盖 CV 或多模态模型。
- **硬件多样性不足**：实验集中在 NVIDIA GPU 和 Apple Silicon，缺乏其他芯片（如 AMD、TPU）验证。
- **前向传播性能弱**：对于 T5 等 encoder-decoder 模型，前向传播性能较差，尤其在 Triton 和 JAX 上。
- **部分模型需 FP32 回退**：约 10% 模型在 `bfloat16` 下无法保持 Top-1 一致性。
- **无显式消融实验**：未量化各编译阶段（如图优化）对最终性能的贡献。

### **未来工作方向**
1. **分布式张量支持**：原生支持分布式训练，细粒度控制通信。
2. **第三方内核注入**：集成 Unsloth、Liger-Kernel 等优化内核作为可选图优化。
3. **扩展更多后端**：支持 Vulkan、CoreML 等，覆盖边缘设备。
4. **自动化转换工具**：利用 LLM 将现有 Transformers 模型自动转为 Axon 定义（已初步尝试成功）。

---

> **总结**：Axon 提供了一种全新的范式——将模型架构从框架束缚中解放出来，通过**强类型 DSL + 统一编译管道**，实现了高性能、高一致性、高可移植性的 LLM 开发。它不仅提升了效率，更推动了 AI 生态的去中心化与开放协作。

</details>

---

### 3. [Learning how to Forget: Fine-tuning for Long-Context Sparse Attention](https://arxiv.org/abs/2608.19920)

**Authors**: Matthias Seeger, Zeyu Zhang, Vihang Patil, Konstantinos Benidis, Sebastian Schelter  
**Category**: cs.CL  
**Published**: 2026-08-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.19920v1  

#### Abstract
A lot of prior work addressed key-value (KV) cache selection and compression by sparse attention to enable long-context inference for transformer language models without excessive hardware budgets. We provide a new method for fine-tuning models with sparse attention. It works for any KV cache policy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning how to Forget: Fine-tuning for Long-Context Sparse Attention*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大语言模型（LLM）在处理长上下文（long-context）时面临巨大的内存和计算开销。传统的 Transformer 模型使用全注意力机制（exact attention），其 Key-Value (KV) 缓存随序列长度呈线性增长，导致 GPU 内存不足。虽然已有大量工作通过 **sparse attention**（稀疏注意力）来压缩 KV 缓存以支持长上下文推理，但这些方法通常在训练阶段仍依赖昂贵的 **sequence parallelism (SP)** 或 **context parallelism (CP)**，这造成了训练与推理之间的不一致。

本文提出并解决了以下核心问题：
- 如何在**有限硬件资源**（如单个 A100 GPU）上对采用 sparse attention 的模型进行高效 fine-tuning？
- 如何让模型在训练过程中“学会遗忘”——即与特定的 KV 缓存策略协同适应（co-adapt），从而提升推理表现？

### 提出的新方法和创新点
作者提出了一种全新的、适用于任意 KV 缓存策略的 **fine-tuning 方法**，其核心创新包括：

1. **Nested Activation Checkpointing + CPU Offloading**  
   结合嵌套激活检查点与 CPU 卸载技术，在前向传播中仅保存必要张量至 CPU，大幅降低 GPU 显存占用。

2. **Delta Encoding of KV Cache in Autograd Graph**  
   利用相邻 chunk 的 KV 缓存之间存在简单线性递推关系（`keys' = scatter(keys, index, key_new)`），在反向传播图中只存储差分值（`delta_key`, `delta_value`），而非完整缓存。这一设计将每个 autograd 调用的显存消耗从 $ O(k \cdot N_c \cdot D) $ 降至 $ O(N_c \cdot D) $，实现了常数级显存需求。

3. **Autograd Saved Tensors Hooks 实现通用性**  
   使用 PyTorch 的 `autograd saved tensors hooks` 机制实现上述 delta encoding，无需为每种缓存策略编写专用 CUDA 内核，使该方法对所有 KV 缓存策略（如 H2O、lastrec 等）完全透明且可扩展。

4. **Replay Log 机制解耦策略与梯度计算**  
   在前向传播中记录缓存替换决策日志（replay log），在反向传播中重放这些决策，避免对复杂的缓存策略本身求导，保证稳定性与效率。

### 相比现有方法的优势

| 特性 | 本方法（us） | 序列并行（SP） | OOMB/NSA 类方法 |
|------|-------------|----------------|------------------|
| 硬件要求 | 单卡 A100 可运行 | 多卡分布式训练 | 多卡 |
| 支持任意 KV 策略 | ✅ 是 | ❌ 否 | ❌ 需定制内核 |
| 训练-推理一致性 | ✅ 高 | ❌ 低（训练无缓存限制） |
| 显存效率 | ✅ 极高（常数级） | ⚠️ 高但受限于设备数 | ⚠️ 较高但需复杂优化 |

此外，该方法还可与 GQA、量化等正交压缩技术结合，并已在开源库 **KeysAndValues** 中提供高性能实现。

---

## 2. 核心实验方法和设置

### 数据集
实验基于 **Helmet** 基准测试套件中的多个长上下文任务，涵盖不同能力维度：

- **RAG**: `nq`, `trivia_qa`, `pop_qa`, `hotpot_qa`
- **Many-shot ICL**: `trec_coarse`, `nlu`, `clinc150`
- **Long-doc QA**: `inf_qa`, `inf_mc`
- **Synthetic Recall**: `json_kv`

上下文长度统一设为 **64k 和 128k tokens**。

### 实验设置
- **模型**: Qwen3-4B-Instruct-2507（4B 参数）
- **训练方式**: LoRA 微调（rank=16, α=16）
- **优化器**: AdamW，学习率 0.0005，最多 5 个 epoch
- **硬件**: 4×Nvidia A100 40GB GPU
- **批大小**: 每设备 batch size=2，总有效 batch size=8
- **位置编码**: RoPE + YaRN
- **KV 缓存长度**: $ N_c = 32768 $
- **Chunk Size**: $ S \in \{1024, 2048\} $

### 评估指标
根据不同任务类型使用不同指标：
- **SubEM**: 子串精确匹配（target 是否出现在 output 中）
- **Accuracy**: 输出中最频繁出现的数字是否等于目标数值
- **ROUGE-F1**: 文本生成质量评分

### 基线方法对比
- **sp**: 使用 MS-SWIFT + DeepSpeed ZeRO-3 + FlashAttention 进行 sequence parallelism 训练
- **us**: 本文提出的方法（fine-tuning with sparse attention in place）
- **no**: 未经微调的原始 checkpoint

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 2）

#### 表格概览
| 方法 | nq (SubEM) | trivia_qa (SubEM) | hotpot_qa (SubEM) | pop_qa (SubEM) | json_kv (SubEM) | inf_mc (Accuracy) |
|------|------------|-------------------|--------------------|---------------|------------------|------------------|
| sp (64k) | 50.7 | 79.8 | 60.0 | 62.7 | — | — |
| us (h2o2k) | 70.8 | 72.0 | 68.7 | 44.7 | — | — |
| sp (128k) | 50.7 | 50.7 | 50.7 | 57.0 | 0.0 | 66.0 |
| us (h2o1k) | 72.8 | 72.0 | 68.0 | 45.8 | 51.0 | 54.0 |
| no (base) | 50.7 | 50.7 | 50.7 | 57.0 | 0.0 | 36.0 |

> 注：数值越高越好；`us` 在多数任务上显著优于 `sp` 和 `no`

### 与基线方法的对比结果

1. **在 SubEM 指标下表现混合**
   - 对于 `nq`, `hotpot_qa`，`sp` 表现更好；
   - 对于 `pop_qa`，`us` 明显更优；
   - `trivia_qa` 上 fine-tuning 整体无效。

2. **在 Accuracy 指标下 `us` 强势领先**
   - 在 `trec_coarse`, `nlu`, `clinc150`, `inf_qa`, `inf_mc`, `json_kv` 等任务上，`us` 性能远超 `sp` 和 `no`。
   - 尤其是 `json_kv`，`us` 成功识别 UUID 约一半时间，而 `sp` 和 `no` 几乎从未成功。

3. **失败模式分析（Table 7）揭示根本差异**
   - `sp` 模型输出极长（平均超过 100 tokens）、包含大量无关内容；
   - `us` 模型输出简洁，长度接近真实标签（R ≈ 1）；
   - `sp` 经常填满全部 128 个生成 token（p128 接近 100%），表明无法正确终止生成。

### 消融实验结果（Table 5 & Appendix）

- **Chunk Size 影响不大**：尝试 $ S=128 $ 更细粒度更新，未带来性能提升，说明当前 $ S=1024/2048 $ 已足够。
- **H2O 变体比较**：
  - `h2o_norm`（归一化得分）略有优势；
  - `h2o_orig`（跨 batch 共享决策）在 `json_kv` 上表现极差（R=3.7, p128=85%），验证了 per-batch 决策的重要性。
- **Q-Hitter（H2O+量化）效果不佳**：在实验中表现不如纯 H2O，可能因量化误差干扰决策。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **训练应与推理保持一致**：若推理时使用 sparse attention 压缩 KV 缓存，则训练也应在相同条件下进行，否则模型会“忘记如何停止”，产生冗余输出。
2. ✅ **模型可以“学会遗忘”**：通过本文方法，模型能与 KV 缓存策略共适应，在有限记忆下做出更优决策。
3. ✅ **H2O 是领先的缓存策略**：在多种任务中表现最优，尤其配合本文提出的高效 SDPA 内核支持后更具实用性。
4. ✅ **低资源 fine-tuning 成为可能**：仅需单卡即可完成长上下文微调，极大降低了部署门槛。

### 方法的局限性
- **延迟仍高于 CP/SP**：尽管显存高效，但由于 chunk 顺序处理和策略判断开销，推理延迟目前仍高于 RingAttention 等并行方案。
- **依赖高质量 SDPA 内核支持**：当前 H2O 实现依赖自定义 Triton 内核返回 summed attention weights，主流库（如 vLLM）尚未原生支持。
- **通用性建立在工程复杂性之上**：`saved tensors hooks` 的非选择性导致需精心管理 annotation 匹配逻辑，易出错且调试困难。

### 未来工作方向
1. **Kernel Fusion 优化延迟**：融合策略判断与 attention 计算，减少 kernel launch 开销。
2. **异步 CPU Offloading**：借鉴 MegaTrain 思路，实现多流异步传输，进一步隐藏数据搬移延迟。
3. **结合 Context Parallelism**：探索 sparse attention 与 CP 的混合架构，兼顾扩展性与灵活性。
4. **推动社区支持**：呼吁 FlashAttention、vLLM 等主流库原生支持 summed attention weights 和灵活因果掩码，促进 sparse attention 生态发展。

---

> 🔗 **开源项目**：  
> 所有实验均基于作者开发的开源库 **[KeysAndValues](https://github.com/awslabs/keys_values)** 实现，旨在为研究者提供简洁、高效的 long-context fine-tuning 与 inference 平台。

</details>

---

### 4. [Design and Empirical Evaluation of a Network-Centric, On-Premises Architecture for Earth Observation Data Access](https://arxiv.org/abs/2608.20283)

**Authors**: Jo\~ao Pinelo, Jo\~ao Gon\c{c}alves, Denis Willett, Amit Ruhela, Derek Steinmoeller, Uriel Mendoza, Pelumi S. Alao, Ronald Soares Lopes, Rogerio Atem de Carvalho, Pedro Mattos  
**Category**: cs.DC  
**Published**: 2026-08-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.20283v1  

#### Abstract
Earth observation (EO) programmes generate data at volumes that exceed the transfer and storage capacity of most institutional networks. Public cloud platforms address this for well-resourced organisations, but institutions across the Atlantic basin face constraints in connectivity, sovereignty and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Design and Empirical Evaluation of a Network-Centric, On-Premises Architecture for Earth Observation Data Access

---

## 1. 论文的主要贡献和创新点

### 解决的问题
地球观测（Earth Observation, EO）数据量呈指数级增长（如 Copernicus 每日生成约 20 TB 数据），传统机构受限于**网络带宽、主权要求和资金约束**，难以依赖公有云平台（如 AWS、GCP）。尤其在大西洋沿岸地区（如亚速尔群岛、西非、巴西等），海底光缆稀少、连接不稳定，导致数据访问效率低下。

本文旨在解决：  
> **如何为资源受限的研究机构构建一个高性能、可复制、独立运行且支持联邦协作的本地化（on-premises）EO 数据基础设施？**

### 提出的新方法与架构
提出了一种**以网络为中心的、可复制的本地化架构**，作为“大西洋云”（Atlantic Cloud）计划的首个节点——AIR Data Centre 的实现基础。其核心设计包括：

- **三层解耦架构**：
  - **数据存储层**：基于 MinIO 的 S3 兼容对象存储集群，采用 EC:4 擦除编码，提供高可用性和横向扩展能力。
  - **元数据管理层**：PostGIS 空间数据库，独立管理时空元数据，支持高效查询。
  - **外部访问层**：通过 S3 API 和 OGC API–EDR 提供标准接口，兼容云原生工具链。
- **网络优先设计**：部署 **100 GbE（100 Gigabit Ethernet）网络骨干**，每服务器配置 4×100 Gbps 双端口绑定，确保内部传输不成为瓶颈。
- **联邦机制**：利用 MinIO 的多站点复制（multi-site replication）实现跨机构数据共享，无需中心化控制。

### 相比现有方法的优势
| 维度 | 传统方案 | 本论文方案 |
|------|--------|-----------|
| **网络依赖** | 依赖高速公网接入公有云 | 完全本地化，仅需低带宽互联用于联邦同步 |
| **主权与成本** | 数据出境风险；持续支付云费用 | 数据自主可控；一次性投入为主 |
| **性能表现** | 受限于公网延迟和带宽 | 内部 100 GbE 支持亚毫秒级延迟、百 Gbps 吞吐 |
| **可复制性** | 高度定制化，难复用 | 开源组件 + 标准硬件，易于推广至其他机构 |

---

## 2. 核心实验方法和设置

### 实验环境与硬件配置
- **部署地点**：葡萄牙亚速尔群岛特塞拉岛（Terceira Island）
- **核心节点组成**：
  - **Storage Tier**：8 节点 Dell PowerEdge R750，每节点 8×1.92 TB NVMe（JBOD），双 Xeon Gold 6342 CPU，192 GB DDR4。
  - **Application Tier**：3 节点 Dell EMC R6525，运行 PostGIS、API Gateway 和服务容器。
  - **网络设备**：两台 Dell EMC Z9264F-ON 交换机，构成双 spine 架构，所有节点通过 4×100 Gbps DAC 光纤连接，总带宽达 400 Gbps/节点。
  - **操作系统与软件**：Ubuntu Server，Docker Swarm 编排，MinIO 分布式模式，NGINX 负载均衡。

### 数据集与负载模拟
未使用真实 EO 数据文件，而是采用**代表性的对象大小**来模拟典型 EO 工作负载：
| 对象大小 | 类型 | 用途 |
|---------|------|------|
| 4 MiB | Zarr chunk | 多维数组随机读取 |
| 64 MiB | COG tile | 空间子集请求 |
| 512 MiB | Sentinel-2 L2A granule | 光学影像批量下载 |
| 2 GiB | Full SAR scene | 合成孔径雷达完整场景 |

### 评估指标
- **吞吐量（Throughput）**：PUT/GET 操作的 MiB/s 和 Gbps
- **延迟（Latency）**：Time-to-First-Byte (TTFB)，往返时延 RTT
- **稳定性**：Coefficient of Variation (CV%) 衡量波动
- **故障恢复能力**：链路/交换机失效下的自动切换行为
- **联邦复制性能**：跨大西洋站点的 multi-site replication 吞吐率

### 基线对比设置（关键创新）
作者进行了**反事实测量（counterfactual measurement）**，在同一套硬件上通过 `tc` 流控工具将网络速率限制为：
- **1 Gbps**：对应商业供应商建议的低成本方案
- **10 Gbps**：代表常规高校/研究机构的高端局域网水平
- **100 Gbps**：实际部署的高性能网络

此举**首次将网络带宽作为独立变量进行隔离测试**，揭示了其对整体性能的真实影响。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（100 GbE 实测）

#### 单客户端性能（Single-client）
| Size | GET (MiB/s) | GET (Gbps) |
|------|-------------|------------|
| 4 MiB | 9,966 | 83.6 |
| 64 MiB | 10,019 | 84.0 |
| 512 MiB | 10,444 | 87.6 |
| 2 GiB | 9,998 | 83.9 |

> ✅ **结论**：GET 性能几乎不受对象大小影响，保持 ~84 Gbps，适合混合粒度访问。

#### 分布式多客户端性能（Distributed, 3 clients）
| Size | GET (MiB/s) | GET (Gbps) |
|------|-------------|------------|
| 4 MiB | 18,217 | 152.8 |
| 64 MiB | 18,867 | 158.3 |
| 512 MiB | 17,946 | 150.5 |
| 2 GiB | 17,287 | 145.0 |

> ✅ **结论**：分布式下可达 ~145 Gbps，接近理论极限的 44%（受应用节点内存带宽限制）。

#### 网络限速下的性能对比（Throttled Baselines）
| Operation | 100 Gbps (MiB/s) | 10 Gbps (MiB/s) | 1 Gbps (MiB/s) | 10G/100G Ratio | 1G/100G Ratio |
|----------|------------------|----------------|---------------|----------------|----------------|
| GET (512 MiB) | 17,946 | 2,347 | 226 | 13.1% | 1.3% |
| PUT (512 MiB) | 15,954 | 2,334 | 232 | 12.4% | 1.2% |

> 🔍 **惊人发现**：
> - 在 1 Gbps 下，系统吞吐仅为 100 Gbps 的 **1.3%**，而非预期的 1%。
> - 这意味着即使网络降速两个数量级，仍能维持一定比例的性能，说明系统并未完全“线性退化”。

#### 时间到首字节（TTFB）
| Size | TTFB @100G (ms) |
|------|----------------|
| 4 MiB | 2.5 |
| 64 MiB | 3.1 |
| 512 MiB | 4.0 |
| 2 GiB | 5.6 |

> ⚡️ **意义重大**：TTFB < 6 ms，使得 Zarr 等格式的数千次小块请求成为可能（1000 次累计延迟 < 3 秒）。

### 与基线方法的对比结果
| 方案 | 预期吞吐（512 MiB） | 实际体验 |
|------|--------------------|----------|
| 1 Gbps 商业提案 | ~2.3 s/granule | 滑动浏览卡顿，交互不可行 |
| 10 Gbps 常规升级 | ~218 ms/granule | 较流畅 |
| **100 GbE 实际部署** | **~29 ms/granule** | **接近本地磁盘滚动体验** |

> 💡 用户感知差异巨大：从“幻灯片式等待”变为“实时探索”，彻底改变科学工作流。

### 消融实验与诊断发现
通过逐节点压测，发现了多个隐藏问题（production monitoring 无法检测）：
1. **BIOS 设置错误**：st04 节点 CPU 频率提升关闭 → 吞吐下降 22%
2. **TCP SACK 关闭**：st06 节点 GET 小对象性能下降 95%
3. **拥塞控制漂移**：部分节点由 BBR 变为 cubic，造成不对称行为
4. **共驻服务争抢内存**：Docker Swarm 中 Traefik 引发重传风暴

> 🛠️ **启示**：benchmark 不仅是性能验证，更是运维诊断工具。

---

## 4. 关键结论和发现

### 主要发现
1. **网络是决定性因素**  
   > “The network investment determines the performance ceiling.”  
   在本地化 EO 架构中，**网络带宽是首要投资项**，远超存储容量或计算能力。

2. **存在性能拐点（threshold）**  
   当单服务器带宽超过 **10 Gbps** 后，系统不再受网络限制，转而受限于**终端内存拓扑（memory topology）**，特别是 NUMA 结构和通道数。

3. **可推迟终端升级，先建高速网络**  
   推荐策略：**前期投入建设 100 GbE 骨干网（不可逆），后期逐步升级服务器内存（可逆）**。当前应用节点仅使用 2/16 内存通道，未来扩容即可逼近理论带宽。

4. **LACP bonding 在高并发下表现不佳**  
   多流竞争导致哈希冲突，stream CV 高达 93.3%，建议未来考虑 RoCE 或优化调度策略。

5. **联邦复制可行且具竞争力**  
   与 AWS/GCP 区域相比，在相同 RTT 条件下，机构间复制速率可达公有云出口速率的 **0.58–1.7×**，证明“去中心化联邦”模式可行。

### 方法的局限性
- **客户端并行度受限**：仅 3 个 application 节点，限制了最大并发请求处理能力。
- **未测试真实应用负载**：如 OGC API 查询、ML 推理等，仅测底层存储吞吐。
- **成本数据缺失**：未提供具体采购价格，难以做全面 cost-performance 分析。
- **LACP 瓶颈明显**：高并发下流量分布极不均匀，影响服务质量一致性。

### 未来工作方向
1. **跨节点联邦复制实测**：随着更多 Atlantic Cloud 节点上线，直接测量 node-to-node replication 性能。
2. **生产流量下的性能表征**：引入真实 DRSS 下行、OGC 查询、AI 推理等工作负载。
3. **联邦治理机制研究**：与 TACC 合作建立软件溯源与信任框架。
4. **PostGIS vs MongoDB 元数据性能对比**：已在 companion paper 中展开。
5. **外部客户端基准测试**：从合作伙伴站点发起真实访问，建立端到端 SLA 基线。

---

> 📌 **一句话总结**：  
> 本文证明，在资源受限环境下，**优先投资 100 GbE 网络 + 使用开源云原生栈**，可以构建出媲美公有云体验的本地化地球观测数据平台，并通过联邦机制实现跨机构协作，为全球南方和发展中区域提供了可持续的技术路径。

</details>

---

### 5. [GenMatch: An End-to-End Generative Matching Framework for Micro-View Order-Dispatching in Ride-Hailing](https://arxiv.org/abs/2608.19751)

**Authors**: Chuang Liu, Yuxueqing Zhang, Tengfei Lyu, Zirui Yuan, Weiqi Hu, Yanghan Cheng, Ming Wang, Li Ma, Zihao Lu  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.19751v1  

#### Abstract
Micro-View Order-Dispatching assigns available drivers to passenger orders within each dispatch batch and is critical to the service quality and operational efficiency of ride-hailing platforms. Mainstream industrial solutions follow a multi-stage paradigm of model prediction, value calculation, and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GenMatch: An End-to-End Generative Matching Framework for Micro-View Order-Dispatching in Ride-Hailing**

---

## 1. **论文的主要贡献和创新点**

### **解决的问题**
传统的 **Micro-View Order-Dispatching (MICOD)** 工业方案采用多阶段范式（multi-stage paradigm），即：
1. **Model Prediction**：预测每对订单-司机（OD pair）的业务信号（如司机应答概率 DA、乘客取消率 PCAA 等）
2. **Value Calculation**：通过人工设计的 **value function** 将多个信号聚合为匹配权重
3. **Dispatch Matching**：使用求解器（如 Kuhn-Munkres）进行全局匹配

这种分阶段优化导致 **cross-stage objective inconsistency** —— 各阶段目标不一致，局部优化不一定提升最终匹配质量。

此外，现有方法无法直接从系统上下文生成端到端的匹配结果，缺乏对动态匹配状态的建模能力。

---

### **提出的新方法与创新思路**
作者提出 **GenMatch**，首个在真实生产环境中部署的 **端到端生成式匹配框架（end-to-end generative matching framework）**，将 MICOD 重新定义为一个 **序列生成任务**，直接从完整的 dispatch batch 生成最优匹配序列。

#### **三大核心组件**
| 组件 | 功能 |
|------|------|
| **Context-Aware Bipartite Encoder** | 对动态稀疏二分图（dispatch batch）进行高效编码，捕捉订单与司机之间的竞争与匹配关系 |
| **Business-Aware Utility Learner** | 利用多任务学习（MTL）从异构反馈中学习统一的业务效用（business utility），替代手工价值函数 |
| **State-Aware Pointer Decoder** | 在每一步选择时跟踪已选/剩余候选对，动态屏蔽不可行选项，实现符合约束的一对一匹配 |

该框架实现了：
- **端到端训练与推理**：避免跨阶段目标不一致性
- **结构化批处理建模**：支持动态图结构下的联合决策
- **状态感知生成**：每步决策依赖当前匹配状态，确保可行性

---

### **相比现有方法的优势**
| 方面 | 传统多阶段方法 | GenMatch |
|------|----------------|----------|
| **目标一致性** | 各阶段独立优化，存在目标冲突 | 单一模型联合优化最终匹配质量 |
| **价值函数设计** | 手工规则，难以融合多目标 | 数据驱动学习统一 business utility |
| **匹配过程建模** | 静态权重 + 一次性求解 | 动态更新状态，逐步构造可行解 |
| **泛化能力** | 仅适用于已有候选对 | 可推广至未广播但模式相似的潜在高价值对 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
在 **DiDi 国际出行市场中的五个城市** 进行实验：
- **离线评估**：City I–III
- **在线 A/B 测试**：City III–V（City III 共享）
- 数据涵盖不同供需规模、订单密度和平均行程距离（见 Table 5）

所有方法共享相同的上游检索与可行性过滤模块，保证候选集一致。

---

### **实验设置与评估指标**

#### **评估方式**
| 类型 | 设置 |
|------|------|
| **离线评估** | 基于历史日志构建模拟环境，回放 7 天真实请求，训练 14 天数据 |
| **在线 A/B 测试** | 生产流量下运行 14 天，采用 **hourly interleaved design**，控制变量对比 |

#### **主要评估指标**
| 指标 | 含义 | 越高越好？ |
|------|------|------------|
| **AR (Answer Ratio)** | 司机应答比例 | ✅ |
| **CR (Completion Ratio)** | 成功完成订单比例 | ✅ |
| **APT (Average Pickup Time)** | 平均接驾时间 | ❌ |
| **GMV (Gross Merchandise Volume)** | 完成订单总交易额 | ✅ |

---

### **基线方法对比**
| 类别 | 方法 |
|------|------|
| **工业多阶段基线** | PDPKM（Kuhn-Munkres）、PDPGreedy、PDPGs（Gale-Shapley） |
| **端到端强化学习** | D2SN（两层 MDP + 强化学习策略） |
| **基于价值的方法** | V1D3、RLW |
| **多智能体方法** | CoRide、CoopRide |
| **消融变体** | GenMatch<sub>value</sub>（仅替换预测模块） |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（离线）**
> 相对于 PDPKM 的相对提升（Table 1）

| 城市 | AR ↑ | CR ↑ | APT ↓ | GMV ↑ |
|------|------|------|--------|--------|
| City I | +0.51% | +0.62% | -0.72% | +0.11% |
| City II | +0.31% | +0.23% | -0.40% | +0.23% |
| City III | +0.83% | +1.17% | -0.23% | +0.55% |

✅ **GenMatch 在所有城市和指标上全面超越所有基线**  
❌ D2SN 表现不佳，说明“仅序列生成”不足以解决问题，需结合结构建模与业务监督

---

### **与基线方法的对比结果**
- **优于所有 solver 变体**（PDPKM/Greedy/GS）：表明生成式建模优于静态加权匹配
- **优于 D2SN**：说明显式的 batch-level 编码、business utility 学习和状态跟踪至关重要
- **优于 V1D3/RLW/CoRide 等宏观视角方法**：说明长期调度策略不能有效迁移至当前 batch 决策

> **GenMatch<sub>value</sub>**（仅替换预测模块）也带来正向增益，验证了 Business-Aware Utility Learner 的有效性。

---

### **消融实验结果（Ablation Study）**
> 在 City III 上进行模块级消融（Table 2 & Table 6）

| 模块 | 移除项 | 影响（CR↓ / APT↑） |
|------|-------|------------------|
| **Encoder** | 匹配/竞争注意力（A1） | CR ↓3.07%，APT ↑1.16% |
| | 度嵌入（A3） | CR ↓0.25%，APT ↑0.18% |
| **Learner** | 辅助监督（A4） | CR ↓3.15%，最大下降 |
| | Utility logit（A5） | APT ↑1.53%，最大上升 |
| **Decoder** | 初始/残差状态（A6/A8） | CR ↓1.07%，GMV ↓1.45% |
| | 固定权重（A12） | CR ↓3.07%，证明动态评分必要 |

✅ 所有组件均有显著贡献，尤其是：
- **竞争建模与度信息** 对编码重要
- **辅助监督** 显著提升表示质量
- **状态感知解码** 是实现高质量匹配的关键

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **端到端生成式框架可有效解决 cross-stage objective inconsistency**，显著提升 dispatch 质量。
2. ✅ **结构化批处理编码**（context-aware bipartite encoder）能有效建模订单-司机间的竞争与协作关系。
3. ✅ **从异构反馈中学习统一 business utility** 比手工设计 value function 更鲁棒、更高效。
4. ✅ **state-aware pointer decoding** 支持在动态变化的可行集中进行合法且高效的序列生成。
5. ✅ **在线 A/B 测试验证了离线结果的可迁移性**，GenMatch 在真实流量中持续取得收益。

---

### **在线 A/B 测试关键结果（Table 3 & 4）**
| 指标 | 整体提升（vs PDPKM） |
|------|---------------------|
| **AR** | +2.26% |
| **CR** | +3.86% |
| **APT** | -1.84% |
| **GMV** | +2.97% |
| **Answer Count** | +2.16% |
| **Completion Count** | +3.84% |
| **DA (Driver Answer)** | +13.96% |
| **PBE (Passenger Bad Experience)** | -15.17% |

> 特别值得注意的是，在 **高峰时段**，GenMatch 的优势更加明显：
- CR 提升从低峰期的 3.24% 上升至高峰期的 **4.12%**
- PCAA/DCAA 下降幅度翻倍以上

这表明 GenMatch 能更好地挖掘密集竞争下的潜在优质匹配机会。

---

### **局限性**
1. **依赖完整 batch 构造**：需要升级原有 pair-level 引擎为 batch-level 推理架构，工程复杂度较高。
2. **自回归延迟风险**：虽然实测满足线上延迟要求，但在极端大图场景下可能面临效率挑战。
3. **fallback 机制依赖旧系统**：异常情况下仍需退回到传统 pipeline，尚未完全脱离 legacy 架构。

---

### **未来工作方向**
1. **非自回归生成（non-autoregressive generation）**：探索并行解码以进一步降低延迟。
2. **更大规模图建模**：扩展至超大规模城市或区域协同 dispatch。
3. **与 Macro-View 方法融合**：结合长期供需调控与短期精细匹配，形成统一调度体系。
4. **引入 LLM 或 reasoning augmentation**：利用大模型增强语义理解与复杂决策推理能力（如 Think2Go [41] 思路）。

---

## ✅ 总结
**GenMatch 是首个成功应用于真实出行平台的端到端生成式订单调度框架**。它通过 **Context-Aware Encoder + Business-Aware Learner + State-Aware Decoder** 的设计，解决了传统多阶段方法的目标不一致问题，并在离线与在线实验中均展现出显著且稳定的性能提升，具有重要的工业实践价值。

</details>

---

### 6. [SAPO: Single-Rollout Autoregressive Policy Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2608.19842)

**Authors**: Dayang Liang, Lang Feng, Bo An, Yunlong Liu  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.19842v1  

#### Abstract
Agentic reinforcement learning (RL) has become a critical stage in the post-training of large language models. Existing critic-free, group-relative methods estimate policy advantages from multiple rollouts, avoiding the substantial memory overhead of conventional proximal policy optimization (PPO) a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SAPO: Single-Rollout Autoregressive Policy Optimization for Agentic Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **agentic RL** 方法在大语言模型（LLM）后训练中面临三大挑战：
1. **缺乏显式的值函数泛化能力与有效的时间信用分配**（temporal credit assignment），导致难以识别低价值或错误的生成前缀；
2. 在长周期复杂任务中容易出现 **advantage collapse**（优势崩溃），尤其是在稀疏奖励下多个rollout获得相同奖励时，归一化后的优势趋于零；
3. **计算成本高昂**：group-relative 方法（如 GRPO）依赖多rollout采样，存在采样预算与性能之间的权衡，且同步开销大。

### 提出了什么新方法或新思路
本文提出 **SAPO**（Single-Rollout Autoregressive Policy Optimization），一种**单rollout、共享参数的actor-critic框架**，其核心思想是：
- 利用 LLM 的 **autoregressive 结构**，在同一模型流中通过不同的 **causal boundary**（因果边界）分别提取策略（policy）、状态值 $V(s)$ 和动作值 $Q(s,a)$；
- 在一个前向传播中完成策略生成与值估计，实现参数共享与高效训练；
- 引入 **trajectory-level generalized advantage estimator**，结合 $\lambda$-returns 与 batch normalization，实现无需多rollout的 turn-level 信用分配。

### 相比现有方法的优势
| 维度 | SAPO | PPO | GRPO |
|------|------|-----|------|
| **Rollout数量** | 单rollout | 多rollout（需critic） | 多rollout（group-relative） |
| **是否需要独立critic** | ❌（共享backbone） | ✅ | ❌ |
| **内存开销** | 极低（无额外critic） | 高（policy + critic + reference） | 中等（仅policy） |
| **时间效率** | ⬇️ 33.2% runtime ↓ vs PPO | 高 | 中等 |
| **信用分配精度** | ✅ 显式值学习 + GAE | ✅ | ❌（粗粒度group归一化） |
| **训练稳定性** | ✅（避免advantage collapse） | ⚠️（易崩溃） | ⚠️（稀疏奖励下失效） |

SAPO 成功**统一了actor-critic的表达能力与critic-free方法的效率优势**。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ALFWorld**：具身家庭环境，测试长周期文本推理与决策能力，包含6类任务（Pick, Look, Clean, Heat, Cool, Pick2），共3,827个任务。
- **WebShop**：交互式网购环境，含约110万商品和12,000条用户指令，评估真实场景下的工具使用与导航能力。

### 实验设置和评估指标
- **模型基础**：Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct。
- **训练配置**：
  - 学习率：$1 \times 10^{-6}$
  - KL penalty：0.01
  - 最大交互轮数：ALFWorld 为 50，WebShop 为 15
  - 总训练步数：150 步
  - 硬件：4×H200 + 8×A40 GPU
- **评估指标**：
  - **Success Rate**（成功率）
  - **Score**（综合得分）
  - 所有结果基于 **3个随机种子** 的均值 ± 标准差

### 基线方法对比
| 类型 | 方法 |
|------|------|
| 封闭模型 | GPT-4o, Gemini-2.5-Pro |
| 零训练提示法 | ReAct, Reflexion |
| RL 方法 | PPO, GRPO, RLOO, EMPG, GiGPO |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 **Qwen2.5-1.5B** 上的表现：
| 方法 | ALFWorld (All) | WebShop (Score/Succ.) |
|------|----------------|------------------------|
| PPO | 54.4% | 73.8 / 51.5% |
| GRPO | 72.8% | 75.8 / 56.8% |
| **SAPO** | **90.1%** (+35.7 vs PPO, +17.3 vs GRPO) | **82.2 / 63.7%** (+8.4 / +12.2) |

> 在 Clean 和 Heat 类别达到 **100% 成功率**

#### 在 **Qwen2.5-7B** 上的表现：
| 方法 | ALFWorld (All) | WebShop (Score/Succ.) |
|------|----------------|------------------------|
| PPO | 80.4% | 81.4 / 68.7% |
| GRPO | 77.6% | 79.3 / 66.1% |
| **SAPO** | **94.0%** | **88.6 / 82.4%** |

> SAPO 在两个规模上均显著超越所有基线，尤其在 WebShop 成功率提升达 **13.7个百分点**（7B）

### 与基线方法的对比结果
- SAPO 在 **ALFWorld** 上平均优于 PPO **+15.1个百分点**，优于 GRPO **+12.1个百分点**；
- 在 **WebShop** 上，SAPO 超越 PPO 和 GRPO，且优于近期改进方法如 GiGPO；
- **无需多rollout采样**，即可实现稳定训练与高性能，解决了 group-relative 方法的同步瓶颈。

### 消融实验（文中未直接展示表格，但从分析可推断）
虽然没有明确列出消融表，但文中强调以下设计的关键作用：
1. **Causal Boundary Readouts**：通过两个保留token（w⁺, w⁻）读取值函数，确保信息隔离；
2. **Batch-Normalized Turn Advantages**：相比 group normalization，跨任务归一化更鲁棒；
3. **Unified Objective with SARSA Targets**：同时优化 PPO 策略目标与 SARSA 值目标，增强值估计一致性；
4. **Single-Rollout GAE**：利用 $\lambda$-return 实现延迟奖励的有效回传。

---

## 4. 关键结论和发现

### 主要发现
1. **单模型流可以同时承担 policy 与 value 功能**：通过 autoregressive 结构中的 causal boundary 分离不同目标，无需独立 critic；
2. **显式值学习 + 单rollout 是可行且高效的路径**：SAPO 在去除 critic 冗余的同时保留了 temporal credit assignment 能力；
3. **架构对齐带来效率飞跃**：actor-critic 学习天然契合 autoregressive 顺序，使得参数共享成为可能；
4. **SAPO 显著降低资源消耗**：
   - 消除 critic 模型 → **节省内存**
   - 减少 rollout 数量 → **降低通信与同步开销**
   - 单次前向传播集成所有模块 → **运行时减少 33.2%**

### 方法的局限性
- 当前方法仍依赖于 **discounted return scaling**（如 $R_{\text{max}}$）进行值预测，可能对极端回报不鲁棒；
- 所有值估计基于 **frozen rollout policy**，未探索动态目标更新机制；
- 尚未在更大模型（如 70B+）或更复杂环境（如 ToolUseBench）中验证扩展性；
- 对 **value temperature $T$** 和 clipping 范围敏感，需谨慎调参。

### 未来工作方向
- 探索 **完全自回归的目标网络更新机制**，替代冻结的 rollout policy；
- 将 SAPO 扩展至 **multi-agent 或分布式 setting**；
- 结合 **intrinsic reward** 或 **curriculum learning** 进一步提升稀疏奖励下的探索效率；
- 应用于 **real-world tool calling、code generation、autonomous agent** 等更广泛场景。

---

> ✅ **总结一句话**：  
> SAPO 通过将 actor-critic 学习嵌入单个 autoregressive 流中，实现了**高性能、高效率、高稳定性的 agentic RL 新范式**，为大规模语言智能体的强化学习提供了一条轻量而强大的技术路径。

</details>

---

### 7. [DARS: Dual-Level Credit Assignment RL with Structured Reasoning for Instruction-Based Image Editing](https://arxiv.org/abs/2608.20161)

**Authors**: Haoxiang Cao, Jiajiong Cao, Xuanpu Zhang, Changqian Yu, Chaoqun Wang  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.20161v1  

#### Abstract
Instruction-based image editing uses a planner-renderer pipeline: a vision-language model (VLM) first converts the instruction into an edit plan, and a diffusion model then executes that plan. Training such systems with only final-image rewards is inefficient because a poor edit does not reveal whet...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# DARS: Dual-Level Credit Assignment RL with Structured Reasoning for Instruction-Based Image Editing

## 1. 论文的主要贡献和创新点

### 解决的问题
在基于指令的图像编辑任务中，系统通常采用 **planner-renderer** 两阶段架构：一个视觉语言模型（VLM）作为 **planner** 将自然语言指令转换为编辑计划，然后由扩散模型（diffusion model）作为 **renderer** 执行该计划生成图像。然而，在仅使用最终图像质量进行奖励的强化学习（RL）训练中，存在严重的 **credit assignment**（功劳分配）问题：

- **跨模块模糊性**：当编辑效果不佳时，无法判断是 planner 的计划有误，还是 renderer 执行失败，导致优化方向不明确。
- **规划器内部模糊性**：即使确定问题出在 planner，也无法定位是其输出中的哪个部分（如修改目标、保留约束等）需要修正。

### 提出的新方法
论文提出了 **DARS**（Dual-Level Credit Assignment RL with Structured Reasoning），一种用于解决上述双重信用分配问题的强化学习框架。其核心创新在于两个层面的设计：

#### (1) 跨模块信用分配（Cross-Module Credit Assignment）
通过 **multi-plan multi-render rollouts**（多计划多渲染采样）来估计奖励的方差，并将其分解为两个部分：
- **`Uplan` (Plan-dominant variability)**：衡量不同计划之间的平均奖励差异，反映“换计划”能带来多大提升。
- **`Urend` (Render-dominant variability)**：衡量同一计划下多次渲染的奖励波动，反映“改进渲染”能带来多大提升。
利用这两个量作为软路由信号（soft routing），动态决定对 planner 和 renderer 的更新权重，从而实现更精准的跨模块优化。

#### (2) 规划器内部信用分配（Within-Planner Credit Assignment）
引入了一个 **四字段结构化推理输出**（four-field structured reasoning output）：
- `<Modify>`：要改变的内容。
- `<Preserve>`：必须保持不变的内容。
- `<Overall>`：场景级的整体一致性要求。
- `<Tips>`：给 renderer 的具体执行提示。
这种结构化输出使得奖励可以分解到每个字段（slot-wise reward），并结合 **前缀门控机制**（prefix-gated reward）——即只有当前面的字段（如 `Modify`）正确后，后面的字段（如 `Preserve`, `Overall`）才能获得奖励。这确保了优化顺序，避免了低级错误被高级细节掩盖。

此外，还利用采样奖励的均值作为 **难度估计**（hardness estimate），驱动自适应课程学习（adaptive curriculum），让模型先学简单的样本。

### 相比现有方法的优势
- **解决了根本性的 credit assignment 问题**：现有方法要么只优化单个模块，要么对两个模块进行统一更新，而 DARS 能够智能地将优化重点分配给最需要改进的模块和模块内的具体部分。
- **实现了精细化的监督**：通过结构化输出和槽位奖励，将最终的图像反馈转化为对 planner 内部各部分的局部监督信号。
- **性能显著提升**：在多个基准测试上，尤其是在需要复杂推理的任务上，DARS 显著优于所有基线方法。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练数据**：从 `THINKEDIT-140K` 和 `UniREdit-Data-100K` 中各抽取 5K 个样本，共 10K 个训练样本。
- **评估基准**：在五个不同的基准上进行评估，覆盖了多种编辑类型：
  - **KRIS-Bench**：侧重于事实、概念和程序性知识的推理密集型编辑。
  - **RISE-Bench**：评估时间、因果、空间和逻辑推理能力。
  - **ImgEdit-Bench**：通用的指令跟随和编辑质量。
  - **GEdit-Bench-EN**：保护断裂敏感的局部编辑。
  - **PICA-Bench**：物理现实性感知的编辑。

### 实验设置和评估指标
- **模型架构**：
  - **Planner**：初始化自 `Qwen3-VL-4B-Instruct`。
  - **Renderer**：初始化自 `Qwen-Image-Edit-2511`。
- **采样策略**：每个输入样本采样 `M=4` 个计划，每个计划采样 `K=4` 个渲染结果（共 16 次 rollouts）。
- **优化算法**：
  - Planner 使用 `text-GRPO`。
  - Renderer 使用 `flow-GRPO`。
- **评估指标**：使用各基准提供的官方评估代码和指标，报告的是官方总体得分（official overall scores），数值越高越好。

### 基线方法对比
论文对比了 11 种基线方法，主要分为以下几类：
- **多轮反思编辑器**：`ThinkRL-Edit`, `Step1X-Edit-v1p2`, `EditThinker`。
- **推理/规划增强编辑器**：`ThinkGen`, `RePlan`, `UniREdit`, `UniReason 1.0`。
- **奖励优化/提示增强方法**：`PromptRL`, `PromptEnhancerV2`。
- **物理专项基线**：`PhysicEdit`。
- **强端到端编辑器**：`Qwen-Image-Edit-2511`。
- **控制基线**：`Joint RL + Adaptive Curriculum`，与 DARS 使用相同的骨干网络、数据、奖励模型和采样预算，但使用自由形式的 planner 输出，用于验证 DARS 各组件的有效性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
在 **Table 1** 中，DARS 在所有五个基准测试上都取得了最佳成绩。

| Method | KRIS-Bench | RISE-Bench | ImgEdit-Bench | GEdit-Bench-EN | PICA-Bench |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DARS** | **80.72** | **27.50** | **4.39** | **7.86** | **64.19/72.75** |
| Joint RL + Adpt. Curriculum (ctrl.) | 72.15 | 25.70 | 4.20 | 7.83 | 63.55/72.22 |

与控制基线相比，DARS 的增益非常显著，尤其是在推理密集型任务上：
- 在 **KRIS-Bench** 上提升了 **+8.57**。
- 在 **RISE-Bench** 上提升了 **+1.80**。

### 消融实验结果
消融研究（Ablation Studies）证实了 DARS 各组件的互补性。

#### (1) 课程学习和路由策略（Table 2）
- **课程学习**：自适应课程（Adaptive Curriculum）显著优于静态课程（Static Curriculum）和无课程（No Curriculum）。
- **路由策略**：软路由（Soft Routing）优于硬路由（Hard Routing）和无路由（No Routing），证明了连续的、基于不确定性的路由信号更有效。

#### (2) 规划器结构和奖励设计（Table 3）
- **规划器结构**：从自由形式（Free-form）到完整的四字段结构，性能呈单调递增。移除任何一个字段都会导致性能下降，其中移除 `<Overall>` 对 RISE-Bench 影响最大，移除 `<Preserve>` 对 GEdit-Bench-EN 影响最大。
- **奖励设计**：前缀门控奖励（Prefix-Gated）显著优于扁平平均（Flat Average）和加权求和（Weighted Sum），证明了其强制依赖关系的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. **双重信用分配至关重要**：在 planner-renderer 架构中，同时解决跨模块和模块内部的 credit assignment 问题是提升性能的关键。
2. **结构化推理是有效的**：将 planner 的自由形式输出改为结构化的四字段格式，不仅没有损失表达力，反而为精细化的奖励分解和诊断提供了可能。
3. **方差分解是可靠的路由信号**：通过 `multi-plan multi-render` rollouts 计算出的 `Uplan` 和 `Urend` 能够准确预测哪个模块是瓶颈，其有效性得到了 GPT-5 伪标签的验证（AUROC > 0.91）。
4. **方法在推理密集型任务上优势最大**：DARS 的最大增益出现在 KRIS-Bench 和 RISE-Bench 上，这验证了其设计初衷——更好地处理复杂的推理过程。

### 局限性
1. **计算成本高**：每次训练需要 `M×K` 次 rollouts，总成本高于单路径更新。
2. **依赖奖励模型**：方差分解和规划器评分都依赖于奖励模型的质量。
3. **难以解耦双模块同时失败的情况**：当 planner 和 renderer 同时出现严重错误时，credit assignment 仍然具有挑战性。
4. **对几何控制和长链推理支持有限**：在迷宫求解、精确的相对尺寸控制等需要紧密耦合多步推理或精确几何控制的任务上，模型表现仍较弱。

### 未来工作方向
- 将该框架扩展到视频编辑或多轮交互式编辑场景。
- 探索更强大的中间表示形式，以支持更复杂的图示推理和几何控制。
- 设计更鲁棒的机制来处理 planner 和 renderer 同时失败的复杂情况。

</details>

---

### 8. [Frequency-Aware Continual Learning for Smart Contract Vulnerability Detection with Large Language Models](https://arxiv.org/abs/2608.19680)

**Authors**: Tenghui Huang, Jiawen Kang, Dongning Liu, Changyan Yi, Chengjun Cai, Anjia Yang, Li Li, Dong In Kim  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.19680v1  

#### Abstract
Smart contract vulnerability detection with Large Language Models (LLMs) faces three causally linked challenges. First, new vulnerability categories demand parameter-efficient adaptation, since full retraining is prohibitive for sequentially arriving tasks. Second, training per-task adapters on a sh...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Frequency-Aware Continual Learning for Smart Contract Vulnerability Detection with Large Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**基于 Large Language Models (LLMs)** 的智能合约漏洞检测在实际部署中面临的三个因果关联的挑战：

1. **参数高效适应（Parameter-Efficient Adaptation）**：新漏洞类别不断出现，全量微调成本过高，需轻量级增量学习。
2. **灾难性遗忘（Catastrophic Forgetting）**：顺序训练多个任务时，共享主干网络导致旧知识被覆盖。
3. **部署整合难题（Deployment Consolidation）**：推理阶段无法获知任务身份，多适配器模型难以直接部署。

这三个问题形成“因果链”——前一个问题的解决方案引发下一个问题，因此需要一个端到端的统一框架。

---

### 🚀 提出的新方法与创新思路

作者提出了一套**三阶段连续学习框架**，每个阶段对应解决上述一个挑战：

#### （1）Stage 1: Frequency-Aware Low-Rank Adaptation (**FA-LoRA**)
- 将 LoRA 扩展到 **Fourier 频域**，通过可学习的频率门控（frequency gates）选择性保留高频分量（含细粒度任务特征）。
- 引入稀疏频率选择机制，仅更新最重要的频段，显著提升参数效率。
- **优势**：仅需 **0.4% 可训练参数**，优于标准 LoRA 和 QLoRA。

#### （2）Stage 2: Forget-Aware Replay (**FAR**)
- 利用 FA-LoRA 中的频率门估计每样本的“遗忘风险”，基于其训练过程中的 **loss dynamics（损失变化）** 动态调整回放优先级。
- 回放缓冲区中高遗忘风险样本获得更高采样概率。
- **优势**：实现知识感知型回放，避免均匀回放的资源浪费，在相同缓存容量下显著缓解遗忘。

#### （3）Stage 3: Anchor-Protected Progressive Merging (**APPM**)
- 在所有任务训练完成后，将多个任务专用的 FA-LoRA 适配器合并为单一部署模型。
- 创新性地利用 FAR 训练产生的**非对称泛化能力**，选择表现最强的适配器作为**anchor**。
- 合并策略包括：
  - **锚保护加权平均**（LoRA 参数）
  - **频域门竞争机制**（gate parameters 的 softmax 竞争）
- **优势**：无需额外训练或访问原始数据，毫秒级完成合并，无运行时内存开销。

---

### 🔍 相比现有方法的优势

| 维度 | 本文方法 | 现有方法局限 |
|------|--------|-------------|
| **参数效率** | FA-LoRA 仅需 0.4% 参数，聚焦高频信息 | LoRA/QloRA 缺乏频域建模，优化不精细 |
| **遗忘控制** | FAR 基于 loss dynamics 动态识别易忘样本 | 回放方法通常采用均匀采样，效率低 |
| **模型部署** | APPM 实现高质量、零开销合并 | 多适配器需任务标识；简单平均破坏性能 |
| **系统集成** | 三阶段闭环设计，组件间信息复用 | 多数研究孤立处理各子问题 |

> ⭐ 特别亮点：**频率门控信息贯穿全流程**——既用于 FA-LoRA 的适应，又被 FAR 用于遗忘估计，再被 APPM 用于门参数的竞争合并，实现了跨阶段的知识传递与协同优化。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用真实世界多标签智能合约漏洞数据集：**DIVE**
  - 包含 22,330 个合约，涵盖 **8 类常见漏洞**
  - 按照合约部署时间戳排序，划分为 **4 个连续时间段（task_A ~ task_D）**
  - 符合现实场景下的时间演化特性（temporal evolution）

| 分割 | task_A | task_B | task_C | task_D |
|------|-------|-------|-------|-------|
| 训练集 | 5,262 | 5,262 | 5,262 | 5,262 |
| 测试集 | 530 | 542 | 513 | 648 |

---

### ⚙️ 实验设置
- **基础模型**：`LLaMA-3.2-1B` 和 `LLaMA-3.2-3B`，使用 **4-bit NF4 量化**
- **适配器配置**：
  - FA-LoRA rank $ r = 16 $
  - 保留 **20% 最具信息性的高频分量**
- **训练方式**：
  - 顺序训练四个任务（$ T_1 \to T_2 \to T_3 \to T_4 $）
  - 每个任务训练后冻结适配器状态
- **回放设置**：
  - 回放缓冲区大小：2,000
  - 回放批次占比：25%
  - FAR 温度 $ T = 2.0 $

---

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Micro-F1 / Macro-F1** | 多标签分类常用指标，Micro 更关注整体准确率 |
| **FWT (Forward Transfer)** | 新知识对后续任务的帮助程度 |
| **BWT (Backward Transfer)** | 是否发生灾难性遗忘（负值表示遗忘） |
| **Merge Time** | 合并耗时（ms） |
| **Runtime Memory Overhead** | 推理时是否增加内存占用 |
| **Independent Upper Bound** | 每个任务单独训练且独立推理的最佳上限 |
| **No-Adapter Lower Bound** | 不进行任何微调的基线 |

---

### 🔁 基线方法对比
#### PEFT 对比（单任务）
- LoRA, QLoRA, SLoRA, FourierFT, FouRA, WaRA

#### 连续学习算法对比
- ER (Experience Replay)
- DER++ (Dark Experience Replay++)
- EWC, Online EWC
- Bidirectional CL

#### 模型合并方法对比
- Simple-Mean
- TIES-Merging
- DARE
- HAM
- RegMean

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 方法 | 平均 Micro-F1 | 距离上界差距 | 合并耗时 | 内存开销 |
|------|----------------|---------------|----------|-----------|
| **APPM (Ours)** | **0.8085** | **+2.7%** | **156 ms** | **0 MB** |
| Simple-Mean | 0.7535 | +8.2% | 102 ms | 0 MB |
| TIES | 0.7535 | +8.2% | 2,123 ms | 0 MB |
| DARE | 0.7355 | +10.0% | 4,204 ms | 0 MB |

> ✅ APPM 性能最接近独立训练上限（within 2.7%），且合并速度极快、无内存负担。

---

### 🔍 与基线方法对比结果

#### （1）FA-LoRA vs 其他 PEFT 方法（完整 DIVE 数据集）
| 方法 | Trainable Params | Micro-F1 (1B) | Micro-F1 (3B) |
|------|------------------|----------------|----------------|
| **FA-LoRA** | **2.62M (0.4%)** | **0.8185** | **0.8424** |
| LoRA | 3.42M | 0.8094 | 0.8370 |
| QLoRA | 1.72M | 0.8185 | 0.8365 |
| FouRA | 0.55M | 0.7635 | 0.8020 |

> ✅ FA-LoRA 在更少参数下达到甚至超越主流方法性能，尤其在 3B 模型上领先。

#### （2）FAR vs 其他 CL 方法
- **FAR 平均 Micro-F1 达到 0.8022**，优于所有基线
- 比 ER++ 高约 5 个百分点
- FWT > 0，表明存在正向迁移
- BWT 接近 0，说明遗忘得到有效抑制

#### （3）APPM 显著优于其他合并策略
- 在 task_C 和 task_D 上表现最佳 → 表明 anchor 机制有效保留后期任务强泛化能力
- 所有任务均衡表现最优 → 证明 anchor + 频率竞争机制实现良好平衡

---

### 🔧 消融实验结果（Ablation Study）

#### APPM 组件消融（见 Fig. 7）
| 配置 | 平均 Micro-F1 | 说明 |
|------|----------------|------|
| Full APPM (Anchor + Freq) | **0.8085** | 最佳整体性能 |
| Only Anchor | 0.7860 | 忽视频率竞争损害早期任务 |
| Only Freq | 0.7750 | 忽视锚保护削弱后期任务 |
| Neither (Simple-Mean) | 0.7535 | 性能下降明显 |

> ✅ 两个机制缺一不可：**anchor 保障泛化主导权，freq competition 减少干扰**

#### 敏感性分析（Fig. 8）
- **LoRA rank r**: 从 4 到 32 影响极小（ΔF1=0.0041）→ 表明频率门控补偿能力强
- **Retention ratio γ**: 即使只保留 5% 高频仍达 0.7617 → 高频承载主要信息
- **FAR temperature T**: 几乎无影响（ΔF1=0.0021）→ 策略鲁棒
- **APPM protection p**: 至关重要！$ p=1.0 $ 比 $ p=0.0 $ 提升 **+3.07%**，后期任务提升超 6%

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **频域建模是高效持续学习的关键路径**：
   - 高频分量蕴含任务特定细节，适合漏洞检测等细粒度任务。
   - 频率门控可作为“元信号”指导遗忘估计与模型融合。

2. **遗忘是可以被动态测量和管理的**：
   - FAR 通过 loss dynamics 估算遗忘风险，实现精准回放调度，优于固定策略。

3. **适配器合并应尊重知识积累的不对称性**：
   - 后续任务继承更多知识，自然具备更强泛化能力，应作为 anchor。
   - APPM 的 anchor-protected merging 是实现高性能合并的核心。

4. **三阶段框架形成正反馈循环**：
   - 更好的 CL 训练 → 更强的 anchor → 更优的合并模型 → 更好部署效果

---

### ⚠️ 方法的局限性
1. **依赖高质量初始预训练模型**：若 LLM 本身对智能合约理解不足，下游适配受限。
2. **Fourier 变换引入一定计算延迟**：虽总体参数少，但在边缘设备可能仍有负担。
3. **未考虑跨任务语义冲突**：某些漏洞模式可能存在逻辑互斥，当前方法未显式建模。
4. **仅验证于以太坊生态**：是否适用于 Solana、Cosmos 等其他链有待验证。

---

### 🔮 未来工作方向
1. **扩展至异构环境**：如 6G 空天地一体化网络中的边缘-云协同场景，探索轻量化边缘侧 Small Language Models 与云端 LLM 协同的持续学习架构。
2. **结合形式化验证**：将 FAR 输出的高风险样本送入 Symbolic Execution 或 Model Checking 工具进行深度验证。
3. **自适应频率选择机制**：让模型自动判断哪些频段对当前任务最重要，而非固定保留高频。
4. **支持双向知识流动**：允许后期任务的知识反哺早期任务模型，进一步增强 FWT。

---

## 总结

> 本论文提出了首个面向 **LLM-based 智能合约漏洞检测** 的**频率感知连续学习框架**，通过 **FA-LoRA → FAR → APPM** 三阶段流水线，系统性解决了参数效率、灾难性遗忘与部署整合三大难题。实验证明其在 DIVE 数据集上达到接近独立训练上限的性能（Micro-F1 0.8085），同时保持极低参数量（0.4%）、毫秒级合并时间和零运行时开销，具有很强的实际应用潜力。

</details>

---

### 9. [ReCache: Efficient KV Cache Reuse and Compression for Tool-Augmented LLM Agents](https://arxiv.org/abs/2608.19662)

**Authors**: Yichu Fang, Sitong Wei, Haozhe Hu, Xiaoyu Shen  
**Category**: cs.CL  
**Published**: 2026-08-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.19662v1  

#### Abstract
Agentic language models repeatedly encode tool and skill schemas that recur across requests in different combinations and orders, preventing standard prefix caching from reusing their key--value (KV) states. We introduce \textbf{ReCache}, a framework for independently caching resource representation...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ReCache: Efficient KV Cache Reuse and Compression for Tool-Augmented LLM Agents 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在 **Tool-Augmented LLM Agents** 中，模型需要动态调用外部工具（tools）或技能（skills），这些资源以 schema 形式提供，并在每次请求中被重复编码。尽管检索机制可以减少上下文长度，但相同的资源会在不同组合和顺序下反复出现，导致大量冗余的 **prefill 阶段计算** 和 **KV-cache 内存开销**。

标准的 **prefix caching** 要求内容必须形成完全一致的前缀才能复用，但在动态组合的资源场景中几乎无法满足。因此，如何高效地实现 **KV-cache 复用与压缩** 成为提升推理效率的关键瓶颈。

---

### **提出了什么新方法或新思路**

作者提出 **ReCache** —— 一种面向工具增强型 LLM Agent 的 KV-cache 复用与压缩框架，其核心由三个阶段构成：

#### （1）**Resource-wise Attention**
- 移除不同 resource 之间的 cross-resource attention。
- 在每个 resource 内部重置相对位置索引（local position reset），使其 KV 表示独立于全局顺序。
- 从而生成 **composition-invariant KV blocks**，支持跨请求独立缓存与复用。

#### （2）**Structural Pruning（结构化剪枝）**
- 基于 **marginal contribution to prediction loss** 对 Transformer 层（layers）和 KV head groups 进行重要性排序。
- 只保留对 resource invocation 最关键的 layer-KV-head-group 路由路径 $ \Omega^* $，其余路径屏蔽 resource KV 状态。
- 实现 **route-level 稀疏访问**，降低 attention 计算量。

#### （3）**Semantic Pruning（语义剪枝）**
- 针对 resource schema 的结构特性，仅保留关键字段：
  - **resource name**
  - **argument names**
  - **argument descriptions**
  - **final suffix token**（作为语义聚合锚点）
- 移除冗余文本（如格式标记、说明性文字等），实现 token-level 压缩。

---

### **相比现有方法的优势**

| 维度 | ReCache 优势 |
|------|-------------|
| **KV 复用能力** | 支持非连续、非固定顺序的 resource 缓存复用，突破 prefix caching 的限制 |
| **压缩策略针对性** | 结合结构稀疏性和字段语义设计剪枝，优于通用文本压缩方法 |
| **性能损失小** | 在显著加速的同时，保持接近 dense 模型的 Inv-F1 性能 |
| **泛化能力强** | 在 resource-disjoint OOD 场景下仍表现稳健 |

---

## 2. 核心实验方法和设置

### **使用了哪些数据集**

构建了一个统一的 benchmark，整合来自七个公开数据集的数据：

- **Tool 数据集**：ToolACE, APIGEN, ToolMind, ToolRet, Toucan, WildToolBench
- **Skill 数据集**：SkillRouter

共包含 **49,424 条训练样本**，并划分两个测试集（各 1,000 条）：

- **T_IND**：in-distribution，涉及训练中见过的 resources
- **T_OOD**：out-of-distribution，resources 完全未见（resource-disjoint）

采用 **diversity-first sampling** 策略避免频率偏差，确保评估的是真正的 schema 解析能力而非记忆捷径。

---

### **实验设置和评估指标**

#### **Backbone 模型**
- 主要使用 **Qwen3-4B (Q1)**，辅助分析使用 **Qwen3-1.7B (Qs)**

#### **评估指标**

| 类别 | 指标 | 说明 |
|------|------|------|
| **Effectiveness** | Inv-F1, ID-F1, Hallucination Rate (%) | 资源调用准确率、识别率、幻觉率 |
| **Efficiency** | TTFT (Time-to-First-Token)<br>TPOT (Time-Per-Output-Token)<br>Attn. Latency<br>Mem. (KV-cache 占用内存) | 推理延迟与内存消耗 |

---

### **基线方法对比**

| 基线方法 | 类型 | 对比目的 |
|--------|------|----------|
| **Dense** | 全量 dense attention | 效果上界 |
| **Block / Ω_full** | 仅 resource-wise attention | 验证位置重置有效性 |
| **SPEED (Oh et al., 2026)** | 层不对称 visibility | structural pruning 替代方案 |
| **SA20.3** | 基于 attention mass 的路由选择 | contribution-based selection 的对照 |
| **Gist / Beacon / SMP** | 通用语义压缩方法 | semantic pruning 对照 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据汇总**

| 方法 | Inv-F1 (T_IND) | Inv-F1 (T_OOD) | TTFT 加速比 | Attn. × | Mem. ↓ |
|------|----------------|----------------|--------------|---------|--------|
| **Dense** | 82.4% | 66.3% | 1× | 1.00× | 100% |
| **Ω_full (resource-wise)** | 82.3% | 64.7% | **3.655×** | 1.001× | 0.47% |
| **ReCache (完整框架)** | **80.3%** | **60.8%** | ~3.6× | **1.423×** | **92.43%↓** |

> 注：ReCache 保留了 Dense 模型 **97.5% (T_IND)** 和 **91.8% (T_OOD)** 的调用性能。

---

### **与基线方法的对比结果**

#### ✅ Structural Pruning 对比
- **Ω20.3 vs Ω_full+SPEED**：
  - 同等结构预算下，Inv-F1 高出 **3.2% (T_IND)** 和 **10.0% (T_OOD)**
  - hallucination 率从 13.4% 降至 0.5%
- **Ω20.3 vs Ω_full+SA20.3**（基于 attention mass）：
  - Inv-F1 提升 **3.0% (T_IND)** 和 **5.0% (T_OOD)**
  - 显示 **contribution-based selection 更鲁棒**

#### ✅ Semantic Pruning 对比
- **Ω_full+Gist**：KV 压缩最强（99.22%↓），但 Inv-F1 仅 **39.2%** → 单一 summary token 不适用于结构化接口
- **Ω_full+Beacon**：chunk-level summary，Inv-F1 78.6%，hallucination 较高（尤其 OOD）
- **ReCache (SMP)**：保留字段 + suffix token，平衡效果与压缩

---

### **消融实验结果**

#### 🔹 **Resource-wise Attention 消融（Table 2）**
| 方法 | Inv-F1 | TTFT (ms) |
|------|--------|-----------|
| Dense | 82.4% | 26.319 |
| Block（无位置重置） | 82.2% | — |
| Ω_full（带位置重置） | 82.3% | **7.200** |

→ 证明 **local position reset 是实现高效复用的关键**

#### 🔹 **Structural Budget 分析（Figure 3 & Table 1）**
- **Layer 贡献高度集中**：前 20 层即覆盖 >97% 贡献（Q1）
- **Head group 分布因模型规模而异**：
  - Qwen3-4B：只需 **3/8 groups**
  - Qwen3-1.7B：需 **7/8 groups**
- → 更大模型具备更强的 head substitutability

#### 🔹 **Semantic Pruning 字段消融（Appendix C, Table 8）**
| 配置 | Inv-F1 | Halluc. |
|------|--------|--------|
| Suffix-only | 14.8% | 76.0% |
| + Resource Name | 45.2% | 1.8% |
| + Arg Names | 67.7% | 1.3% |
| + Arg Descriptions | 72.8% | 1.0% |
| + Resource Desc. | <0.7% 提升 | — |

→ **resource name + arg names + arg descriptions** 是最小有效集合

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Resource-internal semantics 比 global order 更重要**
   - cross-resource attention 很弱，移除后不影响性能
   - resource-local position indexing 足以维持 high Inv-F1

2. ✅ **KV-cache 可以按 resource 粒度独立构建与复用**
   - resource-wise attention 实现 composition-invariant KV blocks
   - 支持跨请求、跨顺序的缓存共享

3. ✅ **Structural sparsity 存在于 layer 和 head group 维度**
   - 并非所有层都需要访问 resource KV
   - contribution-based selection 比 attention-mass 更可靠

4. ✅ **Semantic compression 必须尊重字段结构**
   - 工具 schema 不是普通文本，不能直接套用 prompt compression
   - 必须保留 executable interface 字段（name, args, constraints）

5. ✅ **ReCache 实现近乎恒定的推理开销**
   - 随着 resource 长度增长（XL ≥10K tokens），TTFT、TPOT、Attn. latency 几乎不变
   - KV-cache 内存被限制在 **0.03 GiB** 以内

---

### **局限性**

1. 当前实验集中在 **Qwen3 系列模型**，尚未验证其他架构（如 Llama、Mixtral）上的普适性。
2. 需要 **full fine-tuning** 来适应新的 attention pattern，不适用于 frozen LLM。
3. 假设 system instruction 和 resource schemas 相对稳定；若 retrieval order 或 cross-resource dependency 至关重要，则可能失效。
4. 最优 structural budget 依赖模型大小，需重新分析。

---

### **未来工作方向**

- 将 ReCache 扩展到更大规模模型（如 Qwen3-32B）和其他 backbone 架构。
- 探索无需微调的适配机制，使 ReCache 可用于 black-box 或 frozen models。
- 研究在 highly dynamic environment 下的缓存更新与版本管理机制。
- 探索将 resource cache 部署为 shared service，在多 agent 系统中实现跨用户复用。

---

> 📚 **代码开源地址**：[https://github.com/EIT-NLP/ReCache](https://github.com/EIT-NLP/ReCache)

</details>

---

### 10. [DICS: Data-Informed Centroid Splitting for Decision Tree Classifiers](https://arxiv.org/abs/2608.20258)

**Authors**: MD Saifur Rahman Mazumder, Feng Yu  
**Category**: cs.LG  
**Published**: 2026-08-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.20258v1  

#### Abstract
Decision tree-based models are widely used in machine learning due to their interpretability and strong empirical performance. However, training decision trees can be computationally expensive, particularly for large and high-dimensional datasets, largely due to the exhaustive search over candidate ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DICS: Data-Informed Centroid Splitting for Decision Tree Classifiers

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
决策树（Decision Tree, DT）在训练过程中需要对每个节点进行**候选分裂点（candidate splits）的穷举搜索**，尤其是在高维、大规模数据集上，这一过程计算成本极高，成为训练效率的主要瓶颈。尽管已有如 **histogram-based binning** 和 **approximate split finding** 等优化方法，但它们通常引入量化误差或依赖启发式策略，缺乏对数据结构的有效利用。

### 提出了什么新方法或新思路
本文提出了一种名为 **Data-Informed Centroid Splitting (DICS)** 的新框架，其核心思想是：
- 利用 **K-means 聚类**从输入数据中提取结构信息，生成一组**紧凑且信息丰富的候选分裂阈值**。
- 在分类任务中，假设同类样本在特征空间中聚集，因此聚类边界可作为潜在的分类边界先验。
- DICS 通过计算聚类中心之间的**中点或方差调整边界**来生成候选分裂点，并仅保留距离最远的 $ m $ 对聚类中心以减少冗余。

该方法不改变决策树原有的贪心分裂目标，而是为标准的 **Classification Tree、Random Forest (RF) 和 Gradient Boosting Machine (GBM)** 提供一个预计算的候选分裂字典。

### 相比现有方法的优势
| 方面 | DICS 的优势 |
|------|-------------|
| **计算效率** | 显著降低每节点的分裂搜索空间，从 $ O(NPU) $（穷举）降至 $ O(mPU) $，其中 $ m \ll N $。 |
| **精度保持** | 理论证明和实验表明，DICS 选择的最优分裂增益与标准方法渐近等价（$ O(1/\sqrt{N}) $），几乎不损失预测性能。 |
| **通用性** | 可无缝集成到 DT、RF 和 GBM 中，形成 CGCT、CGRF 和 FastC-GBM。 |
| **无需辅助模型** | 不像某些稀疏树方法需依赖参考模型（如 boosted ensemble），DICS 完全基于原始数据驱动。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验涵盖 **合成数据** 与 **真实世界数据集**：

#### 合成数据
- 生成方式：多类分类，包含线性、正弦、二次项及交互效应。
- 参数范围：$(N, P) \in \{(10^4,10^3), (5\times10^3,7\times10^3), (2\times10^4,5\times10^3)\}$，类别数 $K \in \{3,5,10\}$。

#### 真实数据集
| 数据集 | $N$ | $P$ | $K$ | 类型 |
|--------|-----|-----|-----|------|
| Helena | 65,196 | 27 | 100 | 生物医学（tabular） |
| Spambase | 4,601 | 57 | 2 | 邮件垃圾检测 |
| Santander | 200,000 | 200 | 2 | 金融交易预测 |
| CIFAR-10 | 60,000 | 3,072 | 10 | 图像分类（flatten） |
| MNIST | 70,000 | 784 | 10 | 手写数字识别 |
| Fashion-MNIST | 70,000 | 784 | 10 | 服装图像分类 |

### 实验设置和评估指标
- **硬件平台**：Apple M4 Max + 64GB 内存。
- **最大深度**：
  - DT & RF: $D=8$
  - Boosting: $D=3$
- **候选分裂采样数**：$k=100$
- **聚类对数量 $m$**：
  - 若 $K \leq 5$, $m = K(K-1)/2$
  - 否则 $m=25$
- **mini-batch K-means 批大小**：512
- **评估指标**：
  - **Test Accuracy**（测试准确率）
  - **Training Time (seconds)**（训练时间）
  - **Speedup Ratio**（加速比）

### 基线方法对比
| 模型类型 | 基线方法 |
|----------|---------|
| 单棵决策树 | Standard DT, BDTKS [17] |
| 随机森林 | Random Forest (RF) |
| 梯度提升 | XGBoost [9], LightGBM [10] |

> 注：未将 Clus-DTI [16] 纳入比较，因其改变了树的目标函数；BDTKS 因递归聚类导致计算昂贵，仅用于 DT 对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1–6）

#### ✅ 决策树（DT vs CGCT）
| 数据集 $(N,P)$ | Speedup ($t_{DT}/t_{CGCT}$) | Accuracy Drop |
|------------------|-------------------------------|---------------|
| (10000,1000)     | ~10–13x                       | <0.02         |
| (20000,5000)     | ~14–22x                       | <0.01         |
| CIFAR-10         | 12.8x                         | 0.03          |

> CGCT 在所有数据集上实现 **8–22倍加速**，精度下降极小。

#### ✅ 随机森林（RF vs CGRF）
| 数据集 | Speedup ($t_{RF}/t_{CGRF}$) | Accuracy Drop |
|-------|------------------------------|---------------|
| MNIST | 2.3x                         | 0.01          |
| CIFAR-10 | 12.8x                     | 0.02          |
| Helena | 6.5x                        | 0.01          |

> CGRF 实现 **2.3–13倍加速**，精度基本持平。

#### ✅ 梯度提升（FastC-GBM vs XGBoost / LightGBM）
| 数据集 | Speedup vs XGBoost | Speedup vs LightGBM | Accuracy Drop |
|--------|--------------------|------------------------|---------------|
| CIFAR-10 | up to **18.75x**   | up to **9.49x**        | ≤0.02         |
| MNIST    | 11.54x            | 1.70x                  | none          |
| Helena   | 2.37x             | 2.44x                  | minor         |

> FastC-GBM 在多数情况下显著快于主流 boosting 系统，尤其在大样本下表现突出。

### 与 BDTKS 的对比
- BDTKS 在所有设置下均**更慢且精度更低**，说明其递归聚类策略不适合高效训练。
- 例如在 (10000,1000) 上，BDTKS 训练时间达 30+ 秒，而 CGCT 仅需约 0.2 秒。

### 消融实验结果（Ablation Study）
#### 参数敏感性分析（Figure 2）
- **Cluster pairs $m$**：当 $m > 10$ 后准确率趋于饱和，表明少量高质量聚类对已足够。
- **Feature-threshold pairs $k$**：超过一定数量后增益不再明显。
- **Tree depth $D$**：最佳性能出现在 $D=8{-}10$，更深反而略降。
- **Mini-batch size $b$**：从 32 到 2048 准确率几乎不变，说明 DICS 对 batch size 不敏感。

#### 分裂质量验证（Table 7）
- 在 CIFAR-10 上，CGCT 达到 **97.18% 的累计增益保留率**（cumulative gain ratio $R_a$）。
- 表明即使独立生长，DICS 构建的树仍能捕获绝大部分信息增益。

#### 渐近行为验证（Table 8）
- 随着样本量 $N$ 增加，DICS 与标准 DT 的 **gain gap $\Delta G$ 下降**，且 $\sqrt{N}\Delta G$ 保持稳定 → 支持理论中的 $O(1/\sqrt{N})$ 收敛速率。

---

## 4. 关键结论和发现

### 主要发现
1. **数据结构可用于指导分裂点选择**：聚类中心间的边界是有效的“数据驱动先验”，能极大压缩候选空间而不牺牲性能。
2. **DICS 具有强理论保障**：Theorem 1 证明，在密度正则性条件下，DICS 的最优分裂增益与标准方法差距为 $O(1/\sqrt{N})$，随样本增加趋于零。
3. **实际加速效果显著**：在多种模型和数据集上，DICS 实现 **3–50倍训练加速**，测试准确率损失普遍小于 0.02。
4. **模块化设计便于集成**：DICS 是一种即插即用的组件，适用于 DT、RF、GBM 等主流框架。

### 方法的局限性
- 当前仅适用于 **classification 任务**，尚未扩展至 regression。
- 依赖于 **聚类结构与类别分布的一致性假设**，若数据高度混杂或非凸簇，则可能失效。
- 对于极端不平衡或多模态类别，简单 K-means 可能不能充分捕捉复杂边界。

### 未来工作方向
- 将 DICS 扩展至 **regression trees**，探索基于回归残差的聚类先验。
- 引入更复杂的聚类方法（如 spectral clustering 或 GMM）以适应非球形结构。
- 探索动态更新候选分裂集（dynamic DICS）以适应深层节点的数据分布变化。
- 结合其他加速技术（如 GOSS、column subsampling）进一步优化端到端推理效率。

--- 

> **总结一句话**：  
> DICS 通过引入 **data-informed clustering prior** 来构建紧凑的候选分裂集，在几乎不损失 accuracy 的前提下，实现了 decision tree 类模型的 **orders-of-magnitude training speedup**，为可扩展、高效的树学习提供了新范式。

</details>

---

### 11. [Enforcing LLM Safety through DMD-based Classification of Prompt-Response Embedding Dynamics](https://arxiv.org/abs/2608.19579)

**Authors**: Mohamed Akrout, Olivera Kotevska, Dan Wilson  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.19579v1  

#### Abstract
Large Language Models (LLMs) are increasingly deployed in high-stakes applications, yet their tendency to generate toxic, harmful, or policy-violating content poses significant risks. Detecting these unsafe outputs efficiently in a black-box manner remains an open challenge. In this paper, we extend...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Enforcing LLM Safety through DMD-based Classification of Prompt-Response Embedding Dynamics*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在实际部署中可能生成有害、有毒或违反政策的内容（如仇恨言论、非法指导、医疗误导等），尤其是在**交互依赖型安全违规**（interaction-dependent violations）场景下，仅检查响应文本本身无法识别风险。例如，一个看似无害的提示可能诱导模型做出危险行为（如越权退款）。  
现有黑盒检测方法大多仅基于输出文本或多次采样的一致性分析，难以捕捉**提示与响应之间的动态交互模式**。

### 🚀 提出的新方法与创新思路
本文提出一种基于**动力系统理论**（Dynamical Systems, DS）的黑盒 LLM 安全分类框架，其核心创新如下：

- **将 LLM 的 token 生成过程建模为离散时间动力系统**，利用 **Koopman 算子理论** 和 **动态模态分解**（Dynamic Mode Decomposition, DMD）来拟合安全与不安全状态下的嵌入轨迹演化规律。
- 引入 **Prompt-Response 联合建模机制**：分别对**提示**（prompt）和**响应**（response）的 token embedding 序列构建独立的 Koopman 预测模型，首次显式地纳入提示的动态信息。
- 设计 **微分残差评分**（differential residual score, Δε）作为分类依据：
  $$
  \Delta\epsilon = \frac{1}{L}\sum_{k=1}^{L} \left(\text{error}_{\text{unsafe}}^{(r)} - \text{error}_{\text{safe}}^{(r)}\right) + \frac{1}{P}\sum_{j=1}^{P} \left(\text{error}_{\text{unsafe}}^{(p)} - \text{error}_{\text{safe}}^{(p)}\right)
  $$
  通过比较安全 vs 不安全模型对当前序列预测误差的差异进行二分类决策。

### ⚙️ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **访问要求** | 真正的**黑盒方法**：无需访问模型内部隐藏状态、梯度或 token 概率分布 |
| **推理效率** | 单次响应即可判断，无需多轮采样或一致性检验 |
| **任务适配性** | 无需针对特定任务进行 fine-tuning，仅需少量 embedding 轨迹训练 DS 模型 |
| **检测能力提升** | 显著增强对**交互依赖型违规**的识别能力，这是传统 response-only 方法的盲区 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在三个多样化的安全基准上进行了评估：

| 数据集 | 特点 | 安全标签来源 | 样本量（测试集） |
|-------|------|-------------|----------------|
| **Aegis AI Content Safety Dataset 2.0** [7] | 包含 12 类危害类别（仇恨、暴力、自残等），强调人机交互中的安全边界；由人类标注 + 多LLM陪审团生成标签 | Hybrid pipeline (human + LLM jury) | 12K |
| **Synthetic CoT Safety Benchmark** [47] | 结构化 Chain-of-Thought 输出，包含“安全推理 → 明确拒绝”模式，用于评估模型是否能合理拒绝对有害请求 | Predefined templates | 710 |
| **BeaverTails Dataset** [48] | 包含 14 种危害类型，提示为真实人类撰写，响应由 Alpaca-7B 生成；适合研究人类意图与模型反应的关系 | Human preference labeling | 12K |

### 🔬 实验设置与评估指标

#### 嵌入模型（Embedding Models）
选用 HuggingFace MTEB 排行榜上的三种主流 embedding 模型：

| 模型 | 参数量 | 架构类型 | 上下文长度 |
|------|--------|----------|------------|
| **Qwen3-Embed** | 0.6B | Dense Decoder-Only | 32,768 |
| **Mistral** | 7.2B | Sparse MoE (SMoE) | 32,768 |
| **Llama-3** | 8.0B | Dense Causal Decoder | 8,192 |

#### 分类策略对比
- **Response-only**：仅使用响应 embedding 构建 Koopman 模型
- **Prompt-aware (joint)**：同时使用 prompt 和 response 的 embedding 动力学建模
- **Prompt-only**：仅用提示 embedding 进行分类（消融实验）

#### 评估指标
- **F1 Score**（主指标）
- Accuracy, Recall
- ROC 曲线与 AUC 值
- 不同响应长度阈值下的性能变化（$L \geq 1, 50, 100, 150$）

#### 基线方法对比
- **AegisGuard** [7]：基于参数高效微调（PEFT）的安全过滤器，在 Aegis 上达到 F1=86.8%
- **SelfCheckGPT** [32]、**SAC3** [33] 等零资源黑盒方法（未直接复现，文中指出本方法无需多采样）

> 注：本文方法为完全黑盒且无需任务特定训练，因此与需要 fine-tuning 的监督方法不在同一比较维度。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### 在 **Aegis Dataset** 上的表现（F1 / Accuracy @ $L \geq 150$）

| 方法 | Qwen-Embed | Mistral | **Llama-3** |
|------|-----------|---------|------------|
| Response-only | 72.4 / 78.5 | 67.7 / 77.0 | 72.8 / 78.6 |
| **Prompt-aware (Ours)** | 76.5 / 80.0 | 68.8 / 75.8 | **77.0 / 80.2** |

✅ **Llama-3 在引入 prompt 后表现跃升至最优**，F1 提高 4.2 个百分点。

#### 在 **Synthetic CoT Safety Dataset** 上的表现（F1 @ $L \geq 1$）

| 方法 | Qwen-Embed | Mistral | **Llama-3** |
|------|-----------|---------|------------|
| Response-only | 81.8 | 79.7 | **83.0** |
| **Prompt-aware (Ours)** | 83.2 | 81.1 | **83.7** |

✅ 所有模型均有小幅提升，Llama-3 保持领先。

#### 在 **BeaverTails Dataset** 上的表现（F1 @ $L \geq 150$）

| 方法 | **Qwen-Embed** | Mistral | Llama-3 |
|------|---------------|---------|--------|
| Response-only | 84.7 | 85.6 | 83.7 |
| **Prompt-aware (Ours)** | **86.7** | 86.3 | 85.1 |
| Prompt-only | 83.4 | 82.2 | 79.4 |

✅ **Qwen-Embed 达到全实验最高 F1 = 86.7**，表明其在语义密集型违规检测中更优。

---

### 🔍 消融实验与关键观察

| 发现 | 支持证据 |
|------|----------|
| **加入 prompt embedding 显著提升性能** | 在 Aegis 上，Llama-3 的 F1 提升达 +4.2；在 BeaverTails 上所有模型均获益 |
| **不同 embedding 模型擅长不同类型违规检测** | - **Llama-3**（因果解码器）在交互依赖型违规（Aegis）中表现最佳<br>- **Qwen-Embed**（语义编码强）在响应内显式违规（BeaverTails）中占优 |
| **长序列提供更强动力学信号** | 所有数据集中，随着 $L$ 增加，ROC 曲线向左上移动，说明轨迹越长越易区分 |
| **仅靠 prompt embedding 可实现中等检测能力** | 在 BeaverTails 上，prompt-only 达到 F1=83.4，说明人类提示本身携带安全相关信息 |

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **动力系统视角可用于解释和监控 LLM 行为**：
   - 将 LLM 的 token 流视为可观测的动力系统轨迹是可行且有效的。
   - Koopman-based 预测误差可作为安全性的代理指标。

2. **Prompt-Response 联合建模显著优于 response-only 方法**：
   - 特别是在**交互依赖型安全违规**场景下（如越权操作），必须考虑提示上下文才能准确识别风险。

3. **Embedding 模型的选择应匹配应用场景**：
   - 若应用中存在大量**对抗性提示诱导行为偏移** → 推荐使用 **Llama-3** 类因果架构 embedding 模型
   - 若主要防范**显式有害内容生成**（如暴力、毒品指导）→ 推荐使用 **Qwen-Embed** 类语义表征能力强的模型

4. **本方法实现了高性能的零样本黑盒检测**：
   - 在无需 fine-tuning、无需多采样、无需内部状态的情况下，达到接近专用 fine-tuned 模型（如 AegisGuard）的性能水平（77.0 vs 86.8 F1），差距主要来自后者利用了领域特定训练数据。

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **依赖高质量 embedding 模型** | 性能受限于 embedding 模型能否有效编码安全相关语义 |
| **未处理多模态输入** | 当前仅适用于纯文本 prompt-response 对 |
| **缺乏理论可解释性保证** | 虽然经验上有效，但尚无严格证明安全/不安全流形在 embedding 空间中线性可分 |
| **实时性挑战** | DMD 拟合与预测涉及矩阵运算，可能影响低延迟场景的应用 |

---

### 🔮 未来工作方向

1. **扩展至多类别分类**：
   - 将 binary safe/unsafe 判定拓展为细粒度 harm category prediction（如 violence, self-harm, hate speech 等）

2. **构建集成模型（Ensemble Strategy）**：
   - 融合多个 embedding 模型（如 Llama-3 + Qwen-Embed）的优势，提升鲁棒性

3. **探索 embedding 流形的几何结构**：
   - 理论分析安全与不安全轨迹在 embedding 空间中的分离条件，建立形式化判据

4. **结合在线学习机制**：
   - 允许 DS 模型在部署过程中持续更新，适应新型攻击模式（如新型 jailbreak 技术）

5. **应用于其他生成模型**：
   - 将该框架推广至代码生成、图像生成等领域的安全性监测

---

> 💡 **一句话总结**：  
> 本文开创性地将 **DMD 与 Koopman 理论** 引入 LLM 安全检测，提出一种**无需训练、单样本、黑盒式**的方法，通过联合建模 **prompt-response embedding dynamics** 显著提升了对交互型安全违规的识别能力，并揭示了 **embedding 模型架构与违规类型之间的重要匹配关系**。

</details>

---

### 12. [QUASAR: A Quantum-Classical Neural Network for SAR Satellite Physical-Layer Authentication](https://arxiv.org/abs/2608.20240)

**Authors**: Vincenzo Sammartino, Nathanael Denis, Roberto Di Pietro  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.20240v1  

#### Abstract
X-band SAR satellites (8-12 GHz) play a critical role in disaster response, environmental monitoring, and military intelligence. Yet, they lack robust physical-layer authentication (PLA), a security layer orthogonal to cryptographic solutions. Existing PLA systems, typically based on radio-frequency...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**QUASAR: A Quantum-Classical Neural Network for SAR Satellite Physical-Layer Authentication**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **X-band SAR卫星缺乏物理层认证（Physical-Layer Authentication, PLA）机制**：尽管SAR卫星在灾害响应、环境监测和军事情报中至关重要，但其通信信号（特别是成像脉冲）通常不携带加密载荷，难以进行传统密码学认证。
- **现有RF指纹技术局限于sub-6 GHz频段**：商用SDR设备无法直接采集X-band（8–12 GHz）信号，且经典深度学习模型对IQ相位非线性失真建模能力不足，导致硬件指纹提取效果差。

### 🚀 提出的新方法与创新思路
- **提出 QUASAR**：据作者所知，这是**首个用于X-band SAR卫星的量子-经典混合神经网络架构**，实现物理层认证。
- **核心设计**：
  - 融合一个 **CNN spectrogram encoder** 和一个 **Variational Quantum Circuit (VQC)**。
  - 引入 **IQ-native encoding**：将复数IQ样本的幅度和相位分别映射为单量子比特态在Bloch球上的极角（polar angle）和方位角（azimuthal angle），保留完整信息。
  - 采用**late fusion结构**：CNN输出的latent vector被分路处理，一路送入VQC，另一路经全连接层压缩后与VQC输出拼接，最终由分类头决策。

### 🔍 相比现有方法的优势
| 优势维度 | 描述 |
|--------|------|
| **数据效率极高** | 仅需**10%训练数据**即可达到甚至超越经典基线模型的准确率，大幅缩短数据收集周期（从数月降至数天）。 |
| **更高的分类精度** | 在相同数据预算下，相比纯经典模型提升 **+7.5个百分点** 的准确率。 |
| **更强的非线性建模能力** | VQC利用高维Hilbert空间捕捉IQ信号中的微弱非线性硬件特征（如振荡器漂移、混频器非线性），这些是浅层经典网络难以拟合的。 |
| **端到端可微分训练** | 整体架构支持通过parameter-shift rule进行梯度反向传播，实现量子与经典组件联合优化。 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **来源**：真实采集自 **37颗在轨运行的ICEYE X-band SAR卫星**（工作频率9.65 GHz）。
- **规模**：共收集 **3.76 TB原始IQ数据**，持续28天。
- **采集平台**：
  - 天线：Aaronia Hyperlog Pro 70140（700 MHz – 14 GHz）
  - 下变频器：DSI MX12000（LO设为7.2 GHz → IF = 2.45 GHz）
  - SDR：Ettus USRP X310（采样率10 Msps，带宽40 MHz）
- **预处理流程**：
  - 自适应阈值滤除噪声区间
  - 分段为10万样本的burst
  - STFT生成log-magnitude spectrogram（256×256），重采样至224×224输入CNN
- **类别平衡**：采用one-vs-rest协议构建二分类任务（目标星 vs 其余36颗），并通过欠采样+过采样实现均衡。

### ⚙️ 实验设置与评估指标
- **任务类型**：
  - 二元认证（Binary Authentication）
  - 欺骗检测（Spoofing Detection）：三种攻击场景
- **评估指标**：
  - Accuracy, Macro-F1, Precision, Recall
  - t-SNE可视化 + 聚类质量指数（Silhouette Score, Calinski-Harabasz, Davies-Bouldin）
  - 梯度显著图（Gradient Saliency Maps）用于可解释性分析
- **训练细节**：
  - 批大小：32
  - 优化器：Adam（lr=1e-3, weight decay=1e-4）
  - 早停策略：验证损失连续50轮未下降则终止
  - 模型收敛于第37轮（约63分钟）

### 🆚 基线方法对比
| 类别 | 对比模型 |
|-----|---------|
| **消融模型** | 
| - QNN-Only | PCA降维后接入VQC（破坏时序结构） |
| - CNN-Only | 移除VQC分支，仅保留经典路径 |
| - QUASAR (angle embedding) | 使用传统实值角度编码（丢弃相位信息） |
| **外部经典基线** |
| - ResNet-18 | 单通道适配版本 |
| - MobileNetV2 | 轻量级CNN |
| - Transformer Encoder | 基于spectrogram的轻量ViT |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 二元认证性能（Binary Authentication）
| 模型 | Validation Accuracy | Test Accuracy | Macro-F1 |
|------|---------------------|---------------|----------|
| **QUASAR (IQ-native)** | **0.973** | **0.969** | **0.969** |
| QUASAR (angle embedding) | 0.950 | 0.947 | 0.947 |
| CNN-Only | 0.894 | 0.894 | 0.891 |
| ResNet-18 | 0.854 | 0.854 | 0.861 |

> 💡 **结论**：QUASAR比最佳经典基线高出 **+7.5个百分点** 准确率。

#### ✅ 数据效率测试（Data Efficiency）
- 在仅使用 **10%训练数据** 的情况下：
  - QUASAR 达到 **96.9% 测试准确率**
  - 经典CNN-Only需使用全部数据才能接近此水平
- 图6显示：当训练数据比例低于10%时性能急剧下降；超过10%后增益趋于饱和。

#### ✅ 攻击场景下的欺骗检测率（Spoofing Detection）
| 攻击类型 | 检测成功率 |
|--------|------------|
| **Replay Attack**（回放攻击） | **89.7%** |
| **Crafted-IQ Injection**（伪造IQ注入） | **94.1%** |
| **Space-borne Spoofing**（星载欺骗） | **81.3%** |

> ✅ 特别值得注意的是，在Replay攻击中实现了 **100% Recall（零漏检）**，说明模型能稳定识别出所有伪造信号。

#### 🔍 消融实验结果（Ablation Study）
| 模块变化 | 影响 |
|--------|------|
| **移除VQC → CNN-Only** | 准确率↓7.5个百分点（96.9 → 89.4） |
| **替换为angle embedding** | 准确率↓2.2个百分点（96.9 → 94.7） |
| **收敛速度提升** | IQ-native编码使训练收敛从68轮缩短至37轮（↓46%时间） |
| **QNN-Only失败** | 仅得71.3%准确率，证明PCA破坏了关键时序指纹信息 |

> ✅ 结论：VQC本身贡献+5.3%，IQ-native编码额外贡献+2.2%，二者协同作用显著。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **量子-经典混合架构可用于真实卫星PLA任务**：首次成功将VQC应用于X-band SAR信号认证，验证了其在现实世界高频RF指纹识别中的可行性。
2. **IQ-native encoding具有实质性优势**：相比传统实值编码，同时编码幅度和相位信息不仅提升准确率，还加快收敛，证明**复数结构保真是关键**。
3. **VQC增强了指纹可分性**：t-SNE和聚类指标表明，融合后的latent space中合法与欺骗信号分离更明显，Davies-Bouldin指数降低15.4%，意味着簇更紧凑、边界更清晰。
4. **决策聚焦于脉冲起始瞬态**：梯度显著图显示，模型注意力集中在**<1ms的burst onset窗口**，这正是硬件启动噪声、相位抖动最剧烈的区域，符合物理直觉。
5. **跨接收机迁移性良好**：使用两个独立USRP采集的数据验证了系统的鲁棒性。

### ⚠️ 局限性
- **当前为模拟执行**：VQC在CPU上通过PennyLane模拟运行，并未部署在真实量子硬件上（受限于NISQ设备规模）。
- **仅针对成像脉冲**：未涵盖X-band数据下行链路（如QPSK调制信号），两类信号指纹不可互换。
- **部分卫星识别困难**：对于波形高度平稳的卫星（如X14），hit rate低至0.25，因硬件指纹对比度较弱。
- **依赖高质量IQ采集**：地面干扰（如电磁噪声）可能影响spoofing detection precision。

### 🔮 未来工作方向
1. **扩展为多类认证系统**：从one-vs-rest升级为37类全星座识别。
2. **引入Transfer Learning应对硬件漂移**：定期更新CNN参数以适应长期轨道老化效应（thermal cycling, radiation damage等）。
3. **支持多极化模式采集**：升级前端以捕获VV/VH/HV/HH四种极化信号，丰富极化散射矩阵信息。
4. **探索轻量化量子电路设计**：适配未来NISQ设备的实际部署需求。
5. **防御对抗性RF扰动攻击**：研究如何增强模型对精心构造的小幅扰动的鲁棒性。

---

> 🌟 **总体评价**：  
> QUASAR不仅是**首项将QML应用于X-band SAR卫星PLA的工作**，更展示了量子机器学习在**高维非线性信号建模、小样本学习和物理可解释性方面的独特潜力**，为未来空间安全与量子智能交叉领域开辟了新路径。

</details>

---

### 13. [MidTool: Mid-training Data Synthesis for Agentic Tool Use](https://arxiv.org/abs/2608.20314)

**Authors**: Fengqing Jiang, Yite Wang, Boyi Liu, Zhaoyang Wang, Canwen Xu, Zhewei Yao, Radha Poovendran, Yuxiong He  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.20314v1  

#### Abstract
Mid-training is increasingly recognized as a critical stage for shaping the capabilities of large language models. Recent work has shown that targeted mid-training can strengthen reasoning-intensive abilities such as math and science, and can also improve agentic capabilities in software-engineering...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：MidTool: Mid-training Data Synthesis for Agentic Tool Use**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前大型语言模型（LLM）的**工具使用能力**（tool use）主要依赖于后训练阶段（post-training），如监督微调（SFT）和强化学习（RL）。然而，这种方法要求模型在有限且狭窄的监督下同时掌握多种原子级智能体能力（如工具识别、参数提取、多步规划、缺失信息恢复等）。此外，有效的工具使用知识广泛分布于开发者文档、代码仓库、API 规范等非显式轨迹数据中，难以通过纯后训练充分捕捉。

本文提出：**是否可以通过专门的 mid-training 阶段来提前塑造通用工具使用能力？**

### **提出的新方法与思路**
作者提出了 **MidTool**，一个面向**通用工具使用**（general tool use）的开放中训数据构建管道，并发布了其产物 **MidTool-Mix** —— 一个 20.3B token 的混合语料库。

#### **核心创新点：**
- **首次将“通用工具使用”作为 mid-training 的明确目标**，填补了该领域在数学、软件工程之外的能力空白。
- 设计了**双分支合成策略**以覆盖工具使用的两大核心缺陷：
  1. **Context-Grounded Trajectory Augmentation**：从网页、PDF、代码等非结构化文档中提取工具边界、参数逻辑和工作流结构，生成问答对和单步/多步交互轨迹。
  2. **Native Agentic Trajectory Synthesis**：基于真实 API 和 MCP skills 构建可执行接口，直接生成多轮、带验证的智能体轨迹（含澄清请求、错误恢复等行为）。
- 数据来源多样且互补：涵盖 **Web、PDF、Code、Structured Tool Artifacts**（API/MCP），确保知识广度与深度并存。
- 全流程开源发布，推动 agentic mid-training 的可复现研究。

### **相比现有方法的优势**
| 方面 | MidTool-Mix | 其他 mid-training 方法（如 Dolmina, MegaMath） |
|------|------------|---------------------------------------------|
| **目标能力** | 通用工具使用（agentic tool use） | 数学推理、通用指令遵循、特定任务（如 SWE） |
| **数据构成** | 显式合成 agentic 轨迹 + 文档增强 | 多为通用文本或领域特定数据，缺乏系统性轨迹合成 |
| **工具多样性** | 高（2.6M unique tool names） | 低或无 |
| **公开性** | ✅ 完全公开（HF 集合） | ❌ 多数未完全公开 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

#### **训练数据**
- **MidTool-Mix**：20.3B tokens，由以下四部分组成（见 Table 2）：
  - Web（4.4B / 4.1B 合成）
  - PDF（2.6B / 2.1B 合成）
  - Code（3.8B / 1.5B 合成）
  - Native Agentic Trajectories（1.8B）

#### **下游 SFT 数据**
- **TOUCAN**：采样 100K 工具使用样本用于监督微调。

#### **RL 训练环境**
- **AWM**（Agent World Model）：526 个合成工具使用环境，支持 agentic RL。

#### **评估基准**
| 基准 | 描述 |
|------|------|
| **BFCLv3** | 函数调用质量评测，区分单轮/多轮、参数缺失、长上下文等场景 |
| **2-Bench** | 真实垂直领域交互任务（航空、零售、电信），强调多步执行与容错 |
| **MCP-Universe** | 在真实 MCP 服务器上运行，测试跨域泛化能力（浏览器自动化、金融、位置服务、网络搜索） |

### **实验设置**
- **基础模型**：`Qwen3-4B-Base`, `Qwen3-8B-Base`
- **训练流程**：
  1. Mid-training on MidTool-Mix（1 epoch）
  2. 下游 SFT on TOUCAN subset
  3. 可选 RL on AWM environments
- **对比设置**：
  - 不进行 mid-training（仅 SFT/RL）
  - 使用通用 mid-training 数据（Dolmino-20BT）作为对照
  - 与官方发布的 Qwen3 模型比较

### **评估指标**
- **BFCLv3**：Single-turn, Multi-turn（MF/MP/LC）、Hallucination、Overall
- **2-Bench**：Pass@1, Pass@4
- **MCP-Universe**：Score, Pass Rate（按子任务分项报告）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（摘要）**

| 模型 | BFCL Overall ↑ | 2-Bench Pass@1 ↑ | MCP-Universe Pass ↑ |
|------|----------------|------------------|---------------------|
| Qwen3-4B-Base + SFT | 39.73% | 8.54% | 1.68% |
| **+ MidTool-Mix + SFT** | **50.25%** (+10.5) | **12.23%** (+3.7) | **5.03%** (+3.4) |
| **+ MidTool-Mix + SFT + RL** | **54.18%** (+14.5) | **19.96%** (+11.4) | **10.06%** (+8.4) |

> 所有提升均显著，尤其在多轮、交互性强的任务中表现突出。

### **与基线方法的对比结果**
- **一致优于所有 baselines**：
  - 在 BFCL 上，mid-trained 模型在 multi-turn 子集提升超 10 个百分点。
  - 在 2-Bench 上，Pass@1 接近翻倍（4B 模型从 8.54% → 19.96%）。
  - 在 MCP-Universe 上，pass rate 提升达 5–8 个百分点，显示强 OOD 泛化能力。
- **优于通用 mid-training（Dolmino-20BT）**：
  - Dolmino 对 BFCL 有一定帮助，但在 2-Bench 和 MCP-Universe 上表现差甚至负迁移。
  - 表明：**通用技术文本训练不足以支撑复杂 agentic 行为**，必须引入显式的工具使用轨迹。

### **消融实验结果（Ablation Study）**
（固定 Qwen3-4B-Base + SFT 设置，仅改变 mid-training 数据）

| Mid-training 数据 | BFCL Overall Δ | MCP Pass Δ | 结论 |
|--------------------|----------------|------------|--------|
| 无 mid-training | 0.0 | 0.0 | 基线 |
| Dolmino-20BT | +3.4 | -1.68% | 有害于 MCP 泛化 |
| Raw sources only | +2.6 | +1.4% | 原始文档本身已有价值 |
| + Context-grounded traj. | +4.9 | +3.4% | 强化 grounding 能力 |
| + Native agentic traj. | +7.9 | -6.4% | 提升函数调用精度，但损害泛化 |
| **MidTool-Mix（完整）** | **+10.5** | **+3.4%** | **两者结合最优，互补性强** |

> ✅ **关键发现**：context-grounded 和 native trajectory 分支**功能互补**，缺一不可。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **专用 mid-training 显著提升工具使用能力**  
   相比仅靠 post-training，MidTool-Mix 在多个 benchmark 上带来稳定且显著的性能增益，证明 **mid-training 是塑造 agentic 能力的关键阶段**。

2. ✅ **双分支设计有效覆盖工具使用的核心挑战**  
   - **Context-grounded augmentation** 提升对模糊文档的理解与参数提取能力；
   - **Native trajectory synthesis** 强化多轮规划、澄清与恢复机制。

3. ✅ **能力具有强泛化性**  
   模型能在未见过的 MCP 服务器上成功执行任务，说明 mid-training 构建了**可迁移的工具使用先验知识**。

4. 🚫 **揭示了能力边界：深搜类任务仍需专项训练**  
   尽管其他任务均有提升，但 **web search 子任务得分始终为 0.00%**。这表明：
   - 当前 MidTool-Mix 未能教会模型进行迭代式信息搜集与证据整合；
   - **探索性行为（exploratory behaviors）需要独立的数据与训练目标**。

### **方法的局限性**
- **依赖强教师模型**：轨迹合成依赖 GPT-5/Qwen2.5 等闭源或大模型，限制完全自举。
- **视觉工具使用未覆盖**：当前仅为文本工具，虽有初步视觉迁移信号（见 Appendix C.3），但仍需专门设计。
- **未联合优化 post-training**：实验固定下游训练流程，未探索 mid-training 与 SFT/RL 的协同设计空间。

### **未来工作方向**
1. **扩展 native trajectory 收集规模**，覆盖更多真实工具生态。
2. **构建面向特定领域的 mid-training 混合体**，如 deep search、software engineering、vertical workflows。
3. **减少对强 teacher 的依赖**，尝试用更小模型或自生成方式完成轨迹合成。
4. **联合设计 mid-training 与 post-training**，研究二者如何相互增强。
5. **加强代码密集型工具使用训练**，提升 CLI-style 与 SWE 场景下的执行可靠性。

---

> 🔍 **一句话总结**：  
> **MidTool 首次系统性地将“通用工具使用”纳入 mid-training 范畴，通过融合文档理解与原生轨迹合成，显著提升了 LLM 的 agentic 能力，尤其在复杂交互与跨域泛化中表现出色，同时也明确了其在探索性任务上的边界，为后续研究指明了方向。**

</details>

---

### 14. [Pandora's AI Model Routing Box: Efficient Allocation with Costly Value Estimation](https://arxiv.org/abs/2608.20316)

**Authors**: Adam Fisch, Shubhendu Trivedi, Fantine Huot, William W. Cohen, Michael Kaisers, Mirella Lapata, Kate Larson, Jacob Eisenstein  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.20316v1  

#### Abstract
Heterogeneous AI systems composed of multiple models, architectures, harnesses, or inference-time settings can improve quality and efficiency by routing queries to the specialist who can answer most effectively at the lowest cost. Routing requires estimating each specialist's expected return, but th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Pandora's AI Model Routing Box: Efficient Allocation with Costly Value Estimation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文聚焦于**AI模型路由（model routing）中的成本-精度权衡问题**。在异构AI系统中，多个模型（如不同规模的LLM、带检索增强的模型等）可以处理同一输入，但其能力和推理成本各异。理想情况下，应将每个查询分配给“性价比最高”的模型。

然而，**准确估计每个模型的预期收益（value estimation）本身是有成本的**：
- **廉价估计器**（如基于embedding的KNN）速度快但噪声大；
- **昂贵估计器**（如微调的小型LLM、执行部分推理链或检索）更准确但计算开销高。

传统路由方法通常忽略价值估计的成本，导致资源浪费或次优决策。本文提出：**何时值得为一个模型支付更高代价去获取更精确的价值估计？**

---

### **提出的新方法与新思路**
论文将上述问题形式化为经典的**Pandora's Box问题**（来自经济学与运筹学），并据此提出两个核心框架：

#### **(1) Pandora's Router（集中式路由）**
- 将每个模型视为一个“盒子”（box），打开盒子需支付成本 $ c_m $，以获得更精确的价值估计 $ g_m(x) $。
- 在**非强制检查（non-obligatory inspection）**设定下，允许路由器选择不打开任何盒子而直接基于廉价估计 $ f_m(x) $ 进行决策。
- 基于**高斯信号模型**（Gaussian signal model），推导出闭式解的**保留价格（reservation price）** 和 **备份价格（backup price）**，用于指导是否值得对某个模型进行昂贵评估。
- 算法按保留价格排序，依次决定是否“打开盒子”，并在合适时机停止搜索。

#### **(2) Pandora's Bidder（去中心化竞价）**
- 扩展至去中心化场景：各模型作为独立参与者，自行决定是否投资于自我评估（self-assessment）来参与竞标。
- 引入**价值信息（Value of Information, VoI）** 推理机制：当市场报价接近自身预期能力时，才值得花成本获取更精确的 $ g_m $。
- 对应于Parkes (2005)研究的**有成本偏好获取的升价拍卖机制**的一个阶段。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **理论基础** | 首次将Pandora's Box问题应用于AI模型路由，提供严谨的经济决策框架。 |
| **效率提升** | 显著减少昂贵价值估计器 $ g $ 的调用次数，同时保持甚至优于全量估计的质量。 |
| **灵活性** | 支持动态调整策略，适应不同 $ c_g $ 成本水平，在低/高成本区间自动退化为最优基线。 |
| **可扩展性** | 可自然推广到去中心化多智能体系统，支持模型自主参与竞争。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与领域**
实验覆盖三个典型应用场景：

| 领域 | 描述 |
|------|------|
| **Math** | 数学推理任务（MATH, Omni-Math, AIME, HMMT），比较Gemma-4B与Gemini-3.1-Flash-Lite。$ g $ 利用前20个推理token预测最终正确率。 |
| **RAG** | 检索增强生成，包含维基百科与PubMed两个专用语料库。$ g $ 包含实际检索结果。模型成本为0.05。 |
| **EmbedLLM** | 大规模LLM选型基准，涵盖123个开源模型。$ g $ 是微调的小型LLM直接评分。 |

所有数据划分为训练、校准（calibration）、测试三部分。

---

### **实验设置与评估指标**

#### **价值估计器设计**
- **$ f $（廉价）**: KNN + prompt embedding（Gemini Embedding）
- **$ g $（昂贵）**: 微调小型LLM（Gemini 2.5-Flash-Lite）进行回归预测

#### **评估指标**
主指标为：
> **Regret + Inspection Cost**
- **Regret**: 所选模型的真实奖励与事后最优（oracle）之间的差距
- **Inspection Cost**: 调用 $ g $ 的总成本（统一设为 $ c_g $）

目标是最小化该综合损失。

---

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **f-only** | 仅使用廉价估计 $ f $，从不调用 $ g $ |
| **g-always** | 总是调用所有 $ g $，完全精确但成本极高 |
| **Top-2** | 仅对 $ f $ 分数最高的两个模型调用 $ g $ |
| **Coin Flip** | 以一定概率随机调用 $ g $ |
| **RANDOM-Npr / MARGIN-Npr** | 消融实验：使用Pandora选出的相同查询预算 $ N_{pr} $，但采用随机或基于margin的启发式策略 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见Table 2）**

| 方法 | MATH (Total) | RAG (Total) | EmbedLLM (Total) |
|------|-------------|------------|------------------|
| f-only | 0.117 | 0.150 | 0.393 |
| g-always | 0.128 | 0.141 | 2.356 |
| Top-2 | 0.128 | 0.146 | 0.438 |
| Coin Flip | 0.127 | 0.151 | 1.390 |
| **Pandora's Router** | **0.105** | **0.118** | **0.386** |

✅ **Pandora's Router在所有三项任务上均取得最低的“regret + cost”总和。**

---

### **与基线方法的对比结果**
- **vs. g-always**: 在MATH和RAG上达到相近质量，但**调用 $ g $ 的频率显著更低**（图4显示随 $ c_g $ 上升迅速下降）。
- **vs. f-only**: 在中等成本区域明显降低regret，实现更好的质量-成本平衡。
- **vs. Top-2 / Coin Flip**: 更智能地选择何时调用 $ g $，避免盲目或随机行为。
- **vs. MARGIN-Npr**: 尽管共享相同的查询预算，Pandora仍表现更好，说明其**保留价格机制比不确定性启发式更有效**。

> 图2显示，Pandora's Router在整个 $ c_g $ 范围内几乎紧贴各基线的下包络线，实现了平滑且鲁棒的权衡。

---

### **消融实验结果**
- **MARGIN-Npr vs. RANDOM-Npr**: 不确定性启发式优于随机，但仍不如Pandora。
- **Pandora-OI vs. Pandora-NI**: 非强制检查（NI）版本能跳过所有昂贵查询，尤其在高成本区节省明显。
- **高斯 vs. KNN信号模型**（Table 6）：尽管KNN局部估计在校准性上略优，但并未带来性能提升 → 表明**高斯近似已足够实用**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **价值估计本身应被视为一种有成本的操作**，不能假设免费或固定误差。
2. ✅ **Pandora's Box框架为模型路由提供了理论最优的决策原则**，通过保留价格实现高效搜索。
3. ✅ **Pandora's Router能在极低的 $ g $ 查询频率下匹配g-always的路由质量**，证明了其高效性。
4. ✅ **去中心化的Pandora's Bidder也能有效运作**，但在对手估值不准时可能牺牲全局效率以提升个体效用。
5. ✅ 方法在数学推理、RAG、大规模模型选择三种差异显著的任务中均有效，表明其**通用性强**。

---

### **局限性**
1. **高斯信号假设**：虽然实用，但真实分布可能存在重尾或多峰，未充分建模。
2. **两层估计器限制**：目前只考虑 $ f $ 和 $ g $ 两级，未来可扩展为链式或树状估计流程。
3. **单轮拍卖简化**：Pandora's Bidder基于单轮机制，缺乏对未来竞价的策略性预期。
4. **潜在激励失配**：在去中心化设置中，个体理性可能导致整体福利下降（如图5所示）。

---

### **未来工作方向**
- 探索**非高斯信号模型**（如Gaussian Process、混合模型）以更好拟合复杂分布。
- 构建**多级/链式价值估计器**，形成渐进式精炼路径。
- 扩展至**多轮升价拍卖机制**，支持更复杂的策略互动。
- 研究如何设计**激励相容机制**，使个体利润最大化与全局效率一致。
- 将框架应用于**工具调用、计划生成、多跳推理**等需要动态资源分配的场景。

---

> 📌 **总结一句话**：  
> 本文首次将Pandora's Box理论引入AI模型路由，提出了**Pandora's Router**与**Pandora's Bidder**，实现了在**昂贵价值估计下的高效、自适应、可扩展的模型分配机制**，为构建经济高效的异构AI系统提供了新范式。

</details>

---

### 15. [LLM as Detector: An In-context Learning Approach for Tabular Anomaly Detection](https://arxiv.org/abs/2608.19463)

**Authors**: Tu Anh Hoang Nguyen, Dang Nguyen, Thuc Duy Le, Trung Le, Sunil Gupta  
**Category**: cs.LG  
**Published**: 2026-08-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.19463v1  

#### Abstract
Anomaly detection in tabular data is challenging because abnormal samples often arise as violations of cross-feature dependencies rather than simple marginal deviations. Existing detectors rely on geometric or reconstruction signals, while prior LLM-based approaches mainly fine-tune LLMs with normal...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*LLM as Detector: An In-context Learning Approach for Tabular Anomaly Detection*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Tabular Anomaly Detection (TAD)** 方法存在以下局限：
- 多数基于 **reconstruction error** 或 **density sparsity**，难以捕捉特征间的**结构性依赖关系**（如因果机制）。
- 对混合类型数据（mixed-type）处理不佳，类别变量编码易导致语义丢失。
- 现有 LLM-based 方法（如 AnoLLM、LLM-DAS）需要微调或生成合成异常样本，计算成本高且无法直接建模检测逻辑。

本文提出了一种无需训练、高效且能显式建模跨特征依赖的新范式。

### 提出了什么新方法或新思路
作者提出了 **LLM-Detector**，一个利用大语言模型（LLM）的 **in-context learning** 能力来自动生成可执行异常检测引擎的框架。其核心思想是：
> 将 LLM 视为“代码合成器”，通过结构化提示（prompt），让 LLM 从正常数据中提取的知识出发，自动生成用于打分的 Python 异常评分程序。

该方法分为两个阶段：
1. **Knowledge-to-Code**：将正常数据转化为三种结构化知识并构建 prompt：
   - `Pstats`：统计分布（均值、方差、范围等）
   - `Pcausal`：通过 PC 算法学习得到的因果图（DAG）
   - `Pdistill`：通过 K-Means 提取的代表性原型样本
2. **Code-to-Score**：LLM 根据 prompt 生成一个确定性的 Python 函数 `evaluate_anomalies()`，对测试样本输出 [0,100] 区间内的连续异常分数。

### 相比现有方法的优势
| 维度 | LLM-Detector | AnoLLM | LLM-DAS |
|------|--------------|--------|---------|
| 是否需 fine-tuning | ❌ 否 | ✅ 是 | ✅ 是 |
| 支持类别变量 | ✅ 文本表示保留语义 | ✅ | ❌ 编码后可能失真 |
| 计算效率 | ⭐ 高（仅一次代码生成） | ⚠️ 低（逐样本推理） | ⚠️ 中等（需训练传统模型） |
| 显式建模因果结构 | ✅ 是 | ❌ 否 | ❌ 否 |
| 可解释性与可控性 | ✅ 高（生成代码透明） | ❌ 黑箱预测 | ❌ 间接增强 |

此外，LLM-Detector 不依赖于模型训练或参数优化，完全在 zero-shot 下运行，具备良好的隐私保护性和泛化能力。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
共评估了 **24 个 tabular 数据集**，涵盖多个领域（医疗、金融、网络安全等）：
- **12 个 mixed-type 数据集**：包含数值与类别特征（如 Bank、Vifd、Fraud）
- **12 个 continuous-only 数据集**：全为数值型（如 Shuttle、Thyroid、Wine）

数据来源包括：
- ODDS Library
- ADBench
- Kaggle 公开数据集

### 实验设置和评估指标
- **训练/测试划分**：50% 正常样本用于构建知识包（`D_normal`），其余 50% 正常 + 所有异常构成测试集（`D_test`）
- **评估指标**：
  - 主要指标：**AUC-ROC**
  - 补充指标：**F1-score**（见附录）
- **重复实验**：三次不同随机种子，报告平均值 ± 标准差

### 基线方法对比
共比较了 **15 种 SOTA 方法**，覆盖五大类：
| 类别 | 方法 |
|------|------|
| Proximity-based | IForest, KNN |
| Distribution-based | PCA, ECOD |
| Boundary-based | DeepSVDD, GOAD |
| Reconstruction/GAN-based | REPEN, NeuTraL, SLAD, AnoGAN |
| Self-supervised / Diffusion | ICL, DTE |
| LLM-based | AnoLLM, LLM-DAS |

所有非 LLM 方法通过 **PyOD** 库实现，LLM 方法使用官方代码。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 在 mixed-type 数据集上的 AUC-ROC 结果：
| 方法 | 平均 AUC-ROC |
|------|-------------|
| **LLM-Detector (Ours)** | **0.7407** |
| AnoLLM | 0.6972 |
| LLM-DAS | 0.6900 |
| DeepSVDD | 0.6552 |
| IForest | 0.6112 |

👉 **提升约 5%**，显著优于当前最佳 LLM 方法。

#### 在 continuous-only 数据集上的表现：
- LLM-Detector 达到 **平均 AUC-ROC ≈ 0.91**，略高于 AnoLLM 和 LLM-DAS，在部分数据集（如 Wine、Hepatitis）上优势明显。
- 表明该方法不仅适用于混合类型数据，在纯数值场景下也具有竞争力。

#### F1-score 补充结果（附录3）：
- LLM-Detector 平均 F1 达 **0.5923**，分别比 LLM-DAS 和 AnoLLM 高出约 **5% 和 2%**。

### 与基线方法的对比结果
- 在 **12 个 mixed-type 数据集中的 10 个** 上取得最优或次优性能。
- 特别是在复杂依赖结构的数据（如 DAMRE、Lymp）上远超其他方法。
- 在 **Bank、Fraud、Vifd** 等真实业务场景中表现稳健。

### 消融实验结果（Ablation Study）

#### （1）不同知识组件的影响（mixed-type 数据集）
| 知识组合 | AUC-ROC |
|--------|--------|
| Statistics only | 0.7112 |
| + Causal knowledge | 0.7245 |
| + Distilled samples (**完整 LLM-Detector**) | **0.7407** |

✅ 结论：三者协同作用显著，尤其加入 distilled prototypes 后提升了多变量密度估计能力。

#### （2）不同 LLM backbone 的影响
| LLM 模型 | AUC-ROC |
|--------|--------|
| DeepSeek-V3.2 | 0.7069 |
| GPT-5.2 | 0.7217 |
| **Gemini-3.0** | **0.7407** |

✅ 结论：更强的长上下文理解和代码生成能力（如 Gemini）有助于更准确地解析 prompt 并生成高质量评分函数。

#### （3）蒸馏样本数量 $N_{\text{distill}}$ 的影响
- 当 $N_{\text{distill}} < 100$ 时性能较低；
- 在 $N_{\text{distill}} = 100$ 时达到稳定；
- 更多样本带来边际收益递减。

✅ 推荐设置：$\min(100, N)$，兼顾效率与效果。

#### （4）因果发现算法敏感性分析
比较了 PC、BOSS、FCI、GES、GRaSP 五种算法：
- 不同方法略有差异，但**全部优于 AnoLLM**
- 最佳仍为 PC 算法（0.7407）

✅ 结论：性能增益不依赖单一因果发现工具，说明“引入因果结构”本身有效。

#### （5）计算效率对比（Figure 9）
| 方法 | 平均运行时间（分钟） |
|------|------------------|
| LLM-Detector | ~0.002 |
| AnoLLM | ~7.4 |
| AnoGAN | ~15.8 |

✅ LLM-Detector 测试阶段仅为轻量级程序执行，速度极快，适合实时部署。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **In-context learning 可用于构建可执行的异常检测逻辑**，而不仅是分类或生成任务。
2. ✅ **结构化注入 normal-state knowledge**（统计 + 因果 + 原型）能有效引导 LLM 生成高质量评分规则。
3. ✅ **无需 fine-tuning 或训练任何模型**，即可实现 SOTA 性能，大幅降低计算开销。
4. ✅ **天然支持 mixed-type 数据**，避免类别编码带来的语义损失。
5. ✅ **生成的代码透明、可审计、可复用**，增强了系统的可信度与可维护性。

### 方法的局限性
- **依赖 LLM 的代码生成能力**：若 LLM 输出语法错误或逻辑偏差，会影响最终性能（尽管实验显示主流 LLM 表现稳定）。
- **因果图估计误差**：PC 算法假设无隐变量、满足忠实性等，在现实复杂系统中可能不成立。
- **提示工程敏感性**：prompt 设计需精细控制格式与约束，否则可能导致输出不可控。
- **固定预算机制限制灵活性**：各子得分分配固定权重，缺乏动态调整能力。

### 未来工作方向
- 探索 **multi-turn prompting** 或 **feedback loop** 来迭代优化生成的评分函数。
- 引入 **domain knowledge injection**（如专家规则）进一步增强 prompt 表达能力。
- 扩展至 **time-series tabular data** 或 **streaming anomaly detection** 场景。
- 研究如何自动化评估生成代码的质量与鲁棒性。
- 探索 **smaller specialized models** 替代通用 LLM 进行代码合成，降低成本。

---

> 🔚 **总结一句话**：  
> **LLM-Detector 开创性地将 LLM 作为“异常检测程序生成器”，通过 in-context learning 实现无需训练、高效、可解释的 tabular 异常检测，在 mixed-type 数据上显著超越现有方法，为 LLM 在 structured data reasoning 中的应用提供了全新范式。**

</details>

---

### 16. [RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations](https://arxiv.org/abs/2608.19735)

**Authors**: En Zhi Tan, Jia Xiang Lim, Bryan Lijie Chew, Tze Minh Ng, Benjamin Yan Han Yap  
**Category**: cs.LG  
**Published**: 2026-08-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.19735v1  

#### Abstract
We introduce RecPFN, a prior-fitted network that brings in-context learning to sequential recommendation. RecPFN is pretrained entirely on synthetic clickstream environments sampled from a broad structural causal prior, enabling it to amortize Bayesian-style inference from a small support set. At in...

---

### 17. [Systematic Evaluation of TabPFN-TS for Zero-Shot Probabilistic Heat Load Forecasting in District Heating Networks](https://arxiv.org/abs/2608.20024)

**Authors**: Ben Spoek, Karim K. Ben Hicham, Kai Derzsi, Philipp Althaus, Alexander Mitsos, Dirk M\"uller  
**Category**: cs.LG  
**Published**: 2026-08-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.20024v1  

#### Abstract
District heating energy hubs require reliable heat load forecasts for efficient operational scheduling. Conventional forecasting workflows train system-specific models on historical data, which can become burdensome when networks change through new consumers, retrofits, or changing operating regimes...

---

### 18. [Decoding silent reading from non-invasive EEG](https://arxiv.org/abs/2608.20186)

**Authors**: Ingo Marquardt, Anthilia Alchanat, Priyanka Jain  
**Category**: cs.LG  
**Published**: 2026-08-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.20186v1  

#### Abstract
Non-invasive decoding of inner speech faces a fundamental data problem: a corpus pairing brain activity with a person's spontaneous inner monologue cannot be collected, and the available proxy paradigms (cued repetitive and retrospectively reported generative inner speech) are slow to acquire, poorl...

---

### 19. [Learning Hierarchical Skill Policies with Offline Quality-Diversity Reinforcement Learning](https://arxiv.org/abs/2608.19684)

**Authors**: Tanachai Anakewat, Takayuki Osa, Tatsuya Harada  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.19684v1  

#### Abstract
Recent studies investigate how to leverage pre-collected datasets to improve the policy performance and sample efficiency of RL. One promising approach to achieve this goal is to employ a two-stage strategy: In the first stage, diverse skills are extracted as a low-level policy from a given dataset,...

---

### 20. [Quantifying Event Impacts on Time Series via Multiscale Contrastive Learning](https://arxiv.org/abs/2608.19447)

**Authors**: Yiming Sun, Shengyu Chen, Zhengzhang Chen, Haoyu Wang, Xiaowei Jia, Haifeng Chen  
**Category**: cs.LG  
**Published**: 2026-08-21  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.19447v1  

#### Abstract
Shocks that spread through the web, such as cybersecurity breach disclosures, can abruptly disrupt financial time series and cause substantial abnormal losses. While these events are disclosed as discrete records through news reports, regulatory filings, or public databases, their consequences unfol...

---

### 21. [Unregularized Convergence of Single-Loop, Entropy-Regularized Natural Actor-Critic](https://arxiv.org/abs/2608.19587)

**Authors**: Zhiqiang Tan  
**Category**: cs.LG  
**Published**: 2026-08-21  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.19587v1  

#### Abstract
While entropy regularization is widely used to stabilize and accelerate Natural Policy Gradient methods, its ability to yield faster convergence rates for the unregularized objective remains underexplored. Existing analyses often rely on double-loop architectures and invoke a linear entropy penalty....

---

### 22. [Multi-Source Wasserstein Distributionally Robust Graph Learning](https://arxiv.org/abs/2608.19914)

**Authors**: Chuansen Peng, Yifan Xia, Jinshan Zhong, Xiaojing Shen  
**Category**: cs.LG  
**Published**: 2026-08-21  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.19914v1  

#### Abstract
Network topology inference from graph signals is central to graph signal processing with applications in neuroscience, sensor, and social networks. In practice, target-domain samples are scarce while heterogeneous source-domain data are abundant. Fusing these sources is challenging: Euclidean averag...

---

### 23. [Active Inference as Context Acquisition for AI Agents](https://arxiv.org/abs/2608.19202)

**Authors**: Sanchayan Dutta, Sai Niranjan Ramachandran, Suvrit Sra  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.19202v1  

#### Abstract
Interactive AI agents must acquire the right context as efficiently as possible. When a user omits a constraint, preference, file, or task variable, an agent can proceed with a default assumption or spend tokens on a clarifying question, retrieval call, tool call, or prompt trial. We formulate this ...

---

### 24. [EnvHarness: Awakening Static Worlds for Agent Learning](https://arxiv.org/abs/2608.19880)

**Authors**: Chengsong Huang, Zifeng Wang, Rujun Han, Jun Yan, Yanfei Chen, Zoey CuiZhu, Ke Jiang, Peng Xia, Han Yu, Yufan Zhuang, Yifei Ming, Jiaqi Pan, Bhavana Dalvi Mishra, Jiaxin Huang, Burak Gokturk, Tomas Pfister, Chen-Yu Lee  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.19880v1  

#### Abstract
LLM agents learn by interacting with environments, yet these environments are hand-built and static: blind to an agent's weaknesses, and quickly left behind as it improves. While recent environment generation methods attempt to address this, they require domain-specific pipelines, rely on expensive ...

---

### 25. [Learning Early-to-Final Solution Consistency for MILP Acceleration](https://arxiv.org/abs/2608.19953)

**Authors**: Guanlin Li, Chengrui Gao, Chenguang Wang, Haopu Shang, Zherong Zhang, Ke Xue, Jixiang Lu, Weiyong Yang, Chao Qian  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.19953v1  

#### Abstract
Mixed-Integer Linear Programming (MILP) is a fundamental problem class in operations research and combinatorial optimization, with broad applications to industrial decision-making. Owing to their NP-hardness, however, modern solvers may struggle to find high-quality solutions for challenging MILP in...

---

### 26. [Optimal Skill Selection for LLM Agents with Provable Bicriteria Guarantees](https://arxiv.org/abs/2608.19993)

**Authors**: Yu Chen, Ruishuo Chen, Xun Wang, Zhuoran Li, Longbo Huang  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.19993v1  

#### Abstract
Loading reusable skill documents into a bounded context window is now the primary way large language model (LLM) agents acquire task-specific capabilities, which makes skill selection a first-order determinant of task performance and token cost. Yet current agents score skills independently by seman...

---

### 27. [MemTrapBench: Benchmarking Cognitive Traps in LLM Memory Use](https://arxiv.org/abs/2608.20202)

**Authors**: Mengru Wang, Haozhe Luo, Zhenqian Xu, Zhixiang Cui, Haoming Xu, Qu Yang, Jizhan Fang, Junfeng Fang, Ningyu Zhang  
**Category**: cs.AI  
**Published**: 2026-08-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.20202v1  

#### Abstract
Memory has become a key component of large language models, enabling them to retain information and learn from long-term interactions. However, existing memory benchmarks mainly evaluate whether information is correctly extracted, stored, and retrieved, while largely overlooking how retrieved memori...

---

### 28. [Mitigating Identity Essentialism in LLM Agents with Longitudinal Life Trajectories](https://arxiv.org/abs/2608.19621)

**Authors**: Hexi Wang, Yujia Zhou, Bangde Du, Weihang Su, Xinyuan Cao, Qingyi Pan, Qingyao Ai, Yueyue Wu, Min Zhang, Yiqun Liu  
**Category**: cs.CL  
**Published**: 2026-08-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.19621v1  

#### Abstract
Large language models (LLMs) offer a scalable approach to social simulation, but their credibility depends on how agents are constructed. Existing methods can partially reproduce population-level patterns, yet often fail to capture human-like diversity. Our analysis shows that static-profile agents ...

---

### 29. [CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving](https://arxiv.org/abs/2608.19677)

**Authors**: Huang Cheng  
**Category**: cs.DC  
**Published**: 2026-08-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.19677v1  

#### Abstract
Prefix caching avoids prefill only when a repeated request returns to a server that still holds the prefix KV. Cache-blind balancing disperses that reuse; fixed affinity preserves it but can overload a server. CacheRoute resolves this tradeoff with a periodic routing plan. It admits high-rate keys t...

---

### 30. [FAR-DPO: Feasibility-Aware and Robust Direct Preference Optimization for Cyclic Peptide Design](https://arxiv.org/abs/2608.19808)

**Authors**: Guofeng Zhang, Rong Han, Xiaoyu Wang, Zhiyun Li, Zongbo Han, Xiaohong Liu, Guangyu Wang  
**Category**: cs.LG  
**Published**: 2026-08-21  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.19808v1  

#### Abstract
Cyclic peptides are emerging as promising molecular scaffolds in drug discovery due to their high binding affinity and structural stability. However, extending generative models from linear to cyclic peptide design remains challenging, as cyclization sharply restricts the feasible design space throu...

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

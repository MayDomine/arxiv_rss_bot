# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-09 08:55:40 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Joint Structural Pruning and Mixed-Precision Quantization for LLM Compression](https://arxiv.org/abs/2606.07819)

**Authors**: Hoang-Loc La, Truong-Thanh Le, Amir Taherkordi, Phuong Hoai Ha  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.07819v1  

#### Abstract
Recently, the efficiency of Large Language Models (LLMs) deployment has become a critical concern in practical applications. While post-training quantization (PTQ) and structural pruning are established techniques for reducing memory footprint and inference latency, most existing PTQ approaches opti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Joint Structural Pruning and Mixed-Precision Quantization for LLM Compression》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大语言模型（LLM）在部署时面临**高内存占用和推理延迟**的挑战。虽然已有大量研究采用 **post-training quantization (PTQ)** 和 **pruning** 来压缩模型，但仍存在以下关键问题：

- **现有 PTQ 方法多为层独立优化**：大多数混合精度量化（MPQ）方法基于每层的局部敏感度（如 Hessian、激活统计）分配 bit-width，忽略了误差在网络中的**全局传播效应**，导致次优解。
- **pruning 与 quantization 多为串行处理**：传统流程通常先剪枝再量化（或反之），两者未联合优化，无法捕捉其非正交性（non-orthogonal）带来的协同效应。
- **非结构化稀疏难以在通用硬件上加速**：尽管 unstructured/semi-structured 方法（如 SparseGPT、OBR）能实现高压缩率，但在标准 GPU 上缺乏实际推理加速。

### 提出的新方法与思路
本文提出一个端到端的联合优化框架 **Train-Once-Get-All (TOGA)**，核心创新如下：

#### ✅ 创新点一：**全局感知的混合精度 PTQ（TOGA-q）**
- 将 bit-width 分配建模为 **binary mask 优化问题**，通过一个可训练的 **hypernetwork** 学习最优掩码。
- 不依赖手工设定的 saliency 阈值或层局部指标，而是直接以 **end-to-end language modeling loss**（如 cross-entropy）为目标进行优化，从而最小化**全局误差传播**。
- 支持灵活、非均匀的跨层 bit 分配策略，适应不同层的敏感性差异。

#### ✅ 创新点二：**首个支持 joint structured pruning 与 MPQ 的统一框架（TOGA）**
- 在 TOGA-q 基础上引入结构化剪枝（structured pruning），将 pruning masks 与 quantization masks 统一纳入 hypernetwork 的搜索空间。
- 采用 **prune-then-quantize** 范式，在保留重要通道后对稀疏权重进行量化，已被证明优于反向顺序。
- 构造了一个“masked compressible supernet”，编码所有可能的 pruning + quantization 组合，实现**联合搜索与互知优化**。

#### ✅ 创新点三：硬件友好的高效实现
- 设计了定制化的 **CUDA kernels** 支持 W4A4 + W8A8 混合精度 GEMM 运算（基于 CUTLASS）。
- 对 RMSNorm 层进行了融合优化，并将 KV Cache 也量化至 INT4，进一步降低内存开销。

### 相比现有方法的优势
| 方面 | TOGA 优势 |
|------|----------|
| **性能表现** | 在 ultra-low precision（1–3 bit）下显著优于 SoTA weight-activation 和 weight-only PTQ 方法，尤其在 perplexity 和 zero-shot accuracy 上领先明显。 |
| **效率提升** | 实现高达 **2× prefill speedup** 和 **6.5× peak memory reduction**（vs FP16），优于 OBR 等 semi-structured 方法。 |
| **硬件兼容性** | 输出为 dense-compatible 结构，可在标准 GPU 上高效运行，避免非结构化稀疏带来的速度瓶颈。 |
| **灵活性** | 可自由权衡 sparsity 与 precision，适应任意压缩预算（compression budget）。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练/校准数据集**：`WikiText-2`（用于 hypernetwork 训练）
- **评估基准**：
  - **Perplexity**：在 `WikiText-2` 和 `C4` 数据集上测量。
  - **Zero-shot 推理能力**：在六个常见推理任务上测试：
    - ARC Easy/Challenge
    - BoolQ
    - Winogrande
    - Hellaswag
    - MMLU

### 实验设置
- **模型范围**：涵盖多个主流 LLM：
  - Llama 系列：Llama-3.2-1B, -3B, -2-7B, -3-8B, -3.1-8B
  - Mistral-7B-v0.3
  - Qwen-3-8B
- **硬件平台**：
  - Hypernetwork 训练：单张 NVIDIA A100 GPU（80GB VRAM）
  - 性能测试：NVIDIA L40 GPU
- **训练参数**：
  - Batch size: 1
  - Steps: 2,000（仅量化）、10,000（联合剪枝+量化）
  - 重复次数：5 次取平均

### 评估指标
| 类型 | 指标 |
|------|------|
| **准确性** | Perplexity（越低越好）、Zero-shot Accuracy（越高越好） |
| **效率** | Prefill 推理延迟、Decode 阶段峰值内存使用量 |
| **压缩程度** | Compression Budget = 压缩后模型内存 / 原始 FP16 内存 |

### 基线方法对比
#### 🔹 PTQ 方法（TOGA-q vs）：
- **Weight-activation MPQ**：
  - Atom [30]
  - ResQ [23]
  - SpinQuant [20]（uniform baseline）
- **Weight-only MPQ**：
  - PTQ-1.61 [29]（含 LoRA 预处理版本）
  - BiLLM [16]
  - SliM-LLM [17]

#### 🔹 Pruning + Quantization 方法（TOGA vs）：
- **Sequential Baseline**：
  - DISP-LLM（structured pruning）+ Atom / ResQ / BiLLM / SliM-LLM
- **Joint Methods**：
  - SparseGPT+GPTQ [7]
  - OBR [10]（semi-structured）

此外还设计了变体：
- **TOGA-fixed-sparsity**：添加正则项强制达到特定剪枝比例，用于公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ Ultra-low Precision 表现（<4-bit）
| 对比项 | 提升幅度 |
|--------|---------|
| vs SoTA weight-activation PTQ（如 ResQ） | WikiText-2 perplexity ↓ **21%** |
| vs SoTA weight-only PTQ（如 PTQ-1.61+） | WikiText ↓ **59%**, C4 ↓ **85%** perplexity |
| Zero-shot accuracy 提升 | 最高 ↑ **5.4%** 平均准确率 |

> 示例：在 Llama-2-7B 上，TOGA-q 达到 **7.30** WikiText-2 perplexity，而 ResQ 为 7.35，Atom 为 12.12。

#### ✅ INT4/INT8 Mixed-Precision 设置
- 所有 MPQ 方法均优于 uniform quantization（SpinQuant 完全崩溃于 3-bit）。
- TOGA-q 在 Llama-3.2-1B 上实现 **49.2%** 平均 zero-shot 准确率，超过 Atom（48.3%）和 ResQ（48.2%）。

#### ✅ 联合剪枝+量化（TOGA）表现
- 在 wide compression budget range（0.04–0.18）内全面超越所有 baseline。
- 尤其在高压缩比下（如 80% sparsity），TOGA 显著优于 sequential pipeline（DISP-LLM + PTQ）和 OBR。
- **TOGA-fixed-sparsity** 仍优于其他方法，说明即使固定稀疏度，联合优化本身已带来增益。

#### ✅ 实际推理性能（Llama-2-7B on L40）
| 指标 | TOGA 表现 | 对比优势 |
|------|-----------|----------|
| **Prefill Speedup** | 最高 **2×**（vs FP16） | 比 OBR 快约 **1.3×** |
| **Peak Memory Reduction (Decode)** | 最高 **6.5×**（vs FP16） | 比 OBR 再降 **10%** |
| **Batch Size 支持能力** | 支持更大 batch（FP16 在 batch>12 时 OOM） | 更适合生产部署 |

> 图 3 显示：TOGA 在 prefill 阶段显著快于 OBR；decode 阶段内存占用更低。

### 消融实验结果（Ablation Study）

#### 🔹 TOGA-q 技术组件分析（Table 3）
在 Llama-2-7B 上逐步加入技术组件（W4A4 → W8A8 混合）：

| 步骤 | WikiText-2 Perplexity | C4 Perplexity |
|------|------------------------|--------------|
| W4A4 RTN | 1753 | 2301 |
| + 12.5% INT8 权重 | 6.03 | 8.10 |
| + Channel Reordering | 5.78 | 8.01 |
| + GPTQ refinement | **5.38** | **7.47** |
| + KV Cache INT4 | 5.48 | 7.68 |

✅ 结论：**GPTQ refinement 和 channel reordering 是关键改进来源**，KV cache 量化影响较小但可控。

#### 🔹 Salient Channel 分布可视化（Figure 4）
- TOGA-q 自动识别出早期和末尾 transformer blocks 更敏感，分配更多 salient weights。
- 而 Atom 和 ResQ 使用统一阈值，不能反映层间敏感性变化。
- 发现与已有研究一致：early/late layers 对性能更重要。

---

## 4. 关键结论和发现

### 主要结论
1. **全局损失驱动的混合精度量化优于局部贪婪搜索**：通过 hypernetwork 直接优化 end-to-end 语言建模 loss，能更有效地控制误差传播，显著提升 ultra-low-bit 下的模型质量。
2. **pruning 与 quantization 应联合优化而非串行执行**：TOGA 实现了真正的 joint structured pruning 与 MPQ，在相同压缩预算下获得更好的 accuracy-efficiency trade-off。
3. **结构化稀疏 + 混合精度是实用部署的理想路径**：相比非结构化方法（如 OBR），TOGA 生成的 dense-compatible 模型可在标准 GPU 上高效运行，兼具高性能与高兼容性。
4. **layer-specific bit allocation 更合理**：TOGA 能自动学习各层所需的 bit-width，无需人工设定全局阈值。

### 方法的局限性
- **GPU 内存需求高**：hypernetwork 训练需加载完整 LLM，对于超大规模模型（如 70B）易出现 OOM。
- 当前最大支持约 32B 参数模型（在 80GB A100 上）。
- 依赖 calibration data（WikiText-2）进行训练，虽无需 fine-tuning，但仍需访问部分数据。

### 未来工作方向
- 引入 **distributed training 或 offloading 技术**，降低 hypernetwork 训练阶段的显存消耗，以扩展至百亿级以上模型。
- 探索 **data-free 或 synthetic data-based training** 策略，减少对真实校准数据的依赖。
- 扩展至更多架构（如 encoder-decoder）和其他模态任务。

--- 

> ✅ **一句话总结**：  
> TOGA 是首个实现 **joint structured pruning 与 mixed-precision PTQ** 的端到端框架，通过全局损失优化 binary masks，在 ultra-low-bit 场景下大幅超越 SoTA 方法，同时提供卓越的实际推理加速与内存节省，推动 LLM 向边缘设备高效部署迈进一大步。

</details>

---

### 2. [From Rigid to Dynamic: Entropy-Guided Adaptive Inference for Long-Context LLMs](https://arxiv.org/abs/2606.09508)

**Authors**: Zhanchao Xu, Haoyang Li, Qingfa Xiao, Fei Teng, Chen Jason Zhang, Lei Chen, Qing Li  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.09508v1  

#### Abstract
Existing sparse attention and KV cache compression methods for long-context LLM inference typically apply fixed sparsity patterns or uniform budgets across all attention heads, overlooking the substantial variation in attention behavior among heads and contexts. We observe two distinct entropy patte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Rigid to Dynamic: Entropy-Guided Adaptive Inference for Long-Context LLMs*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在长上下文（long-context）大语言模型（LLM）推理中，存在两个主要瓶颈：
- **Prefilling 阶段**：注意力计算复杂度为 $O(N^2)$，导致首词延迟高。
- **Decoding 阶段**：KV Cache 随序列增长线性膨胀，造成显存压力。

现有方法如 **SnapKV**、**AdaKV** 和 **CritiPrefill** 存在以下问题：
- 使用固定的稀疏模式或统一预算分配，忽略不同 attention head 之间的异质性；
- 在 prefill 阶段就决定 KV Cache 压缩策略，而忽略了 decoding 阶段 attention pattern 的动态变化；
- 依赖离线分析进行 head 分类，无法适应输入相关的上下文依赖行为。

---

### 提出了什么新方法或新思路

作者提出 **EntropyInfer** —— 一种无需训练、基于注意力熵（attention entropy）的自适应推理框架，包含两个核心模块：

#### ✅ **Entropy-Guided Sparse Prefill**
- 引入“观察注意力矩阵”（observation attention matrix）以低成本估计每个 head 每个 segment 的注意力熵。
- 将 attention head 划分为两类：
  - **Rigid Heads**：熵值极低（<1e-5），注意力分布几乎确定，分配固定小预算。
  - **Dynamic Heads**：熵波动显著，语义信息丰富，按熵变化幅度动态调整计算预算。
- 实现细粒度到 **head + segment** 层面的自适应资源分配。

#### ✅ **Latent KV Cache Compression**
- 不再在 prefill 结束时立即压缩 KV Cache。
- 推迟到生成前 $N_d$ 个输出 token 后，利用这些 **output tokens** 构建新的 observation window 来重新评估重要性并压缩。
- 解决了 prefill 与 decoding 注意力模式不一致的问题。

> 🔍 **关键洞察**：同一 attention head 在不同输入下可能表现为 Rigid 或 Dynamic —— 因此必须在线实时分类，不能依赖离线 profiling。

---

### 相比现有方法的优势

| 维度 | 现有方法局限 | EntropyInfer 改进 |
|------|---------------|---------------------|
| **Prefill 加速** | 固定 sparsity / 全局阈值 / 均匀 budget | 按 head 类型 + 熵波动动态分配预算 |
| **KV Cache 压缩** | 仅用 input token 决策，忽略生成影响 | 利用 output token 进行 re-ranking，更准确 |
| **Head 差异性处理** | 静态分类（如 RazorAttention） | 动态在线识别 Rigid/Dynamic Heads |
| **部署成本** | 多需 fine-tuning 或额外训练 | 完全 training-free，即插即用 |

---

## 2. 核心实验方法和设置

### 使用的数据集

| 数据集 | 特点 |
|--------|------|
| **LongBench** (Bai et al., 2024) | 包含 QA、摘要、代码等多任务，支持中英文，全面评估模型能力 |
| **InfiniteBench** (Zhang et al., 2024) | 聚焦超长上下文（>100K tokens），测试极端场景下的性能 |
| **Needle-in-a-Haystack 变体** | 自定义 prompt 测试检索能力，用于效率评测 |

此外还对 **openPangu** 系列模型进行了扩展实验。

---

### 实验设置和评估指标

#### 模型
- 主干模型：`Llama-3.1-8B-Instruct`, `Qwen-2.5-7B-Instruct`
- 扩展模型：`openPangu-Embedded-1B`, `openPangu-Embedded-7B`

#### 硬件环境
- 单张 NVIDIA H100 GPU（80GB）
- 192GB CPU 内存 + 8 核 CPU

#### 超参数设置
| 方法 | 参数设置 |
|------|----------|
| SnapKV / AdaKV | KV Cache Budget = 1024 tokens |
| CritiPrefill | Prefill Budget = 2048 tokens/segment |
| EntropyInfer | Base Prefill Budget = 2048, Decode Budget = 1024 |

---

### 评估指标

| 类型 | 指标 | 说明 |
|------|------|------|
| **效果指标** | F1 Score, ROUGE-L, Accuracy, Edit Sim, Exact Match | 分别用于 QA、摘要、代码、对话等任务 |
| **效率指标** | End-to-End Latency, Speedup Ratio | 衡量从输入到生成 100 个 token 的总耗时 |

---

### 基线方法对比

| 基线方法 | 类型 | 核心机制 |
|---------|------|---------|
| **SnapKV** (Li et al., 2024) | KV Cache 压缩 | 观察窗口 + 快照机制保留重要 token |
| **AdaKV** (Feng et al., 2026) | KV Cache 压缩 | 按 head 分散程度动态分配缓存预算 |
| **CritiPrefill** (Lv et al., 2025) | Prefill 加速 | 按 segment 关键性选择 top-k blocks |
| **FlexPrefill** (Lai et al., 2025) | Prefill 加速 | 全局自适应稀疏门限 |
| **RazorAttention** (Tang et al., 2025) | Head-aware 压缩 | 离线划分 retrieval/echo heads |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **LongBench 结果（Table 1）**
- 在 `Llama-3.1-8B-Instruct` 上：
  - **平均得分最高**（48.91 vs Base 48.94），接近全注意力表现；
  - 显著优于 AdaKV (47.68)、SnapKV (47.66)、CritiPrefill (48.32)；
  - 在多个子任务（如 SAMSum, LCC）甚至略超 Base 模型。
- 在 `Qwen-2.5-7B-Instruct` 上同样取得最优平均分（47.98）。

#### ✅ **InfiniteBench 结果（Table 2）**
- 输入长度超过 100K tokens 场景下：
  - 平均得分 **43.36**（Llama），远高于 SnapKV (41.93)、AdaKV (40.59)；
  - 在关键检索任务（R.PassKey）达到 **100% 准确率**，而 AdaKV 仅为 85.42%；
  - 显示出极强的长程信息保持能力。

#### ✅ **端到端延迟测试（Figure 4）**
- 在 context length 达到 140K 时：
  - **最高实现 2.39× 的 end-to-end speedup**；
  - 显著优于所有 baseline（包括 CritiPrefill 和 AdaKV）；
  - 随着 context 增长，加速优势持续扩大。

#### ✅ **openPangu 模型上的泛化性验证（Figure 7 & Tables 3–4）**
- 在 `openPangu-7B` 上仍能获得明显加速；
- 在 LongBench 和 LoCoMo 多轮对话基准上保持高质量输出；
- 证明方法具有良好的跨架构迁移能力。

---

### 消融实验结果（Ablation Study）

| 设置 | 描述 | 发现 |
|------|------|------|
| **Ours w/o LD** | 关闭 latent decode，仅启用 entropy-guided prefill | 效果轻微下降，但效率仍优于多数 baseline |
| **Ours w/o SP** | 关闭 entropy-guided prefill，仅启用 latent decode | 效率提升有限，说明两者协同作用关键 |
| **完整版 Ours** | 同时启用两项技术 | 实现最佳效率-质量平衡，latency 最低 |

> 📌 结论：两个模块相辅相成，**latent decode 补偿了 entropy prefill 引入的少量 decoding 开销**，最终实现整体最优。

---

## 4. 关键结论和发现

### 主要发现

1. **Attention Heads 存在两种本质模式**：
   - **Rigid Heads**：注意力高度集中，熵接近零，适合低预算处理；
   - **Dynamic Heads**：熵波动大，承载关键语义，需更多计算资源。

2. **Head 类型是上下文依赖的（context-dependent）**：
   - 同一个 head 在不同输入下可切换角色；
   - 因此 **无法通过离线 profiling 固定分类**，必须在线判断。

3. **Prefill 与 Decoding 注意力模式存在显著差异**：
   - 仅靠 input token 无法准确预测 decoding 阶段的重要 KV；
   - 引入 output token 进行 re-ranking 可显著提高压缩质量。

4. **Entropy 是一个廉价且有效的在线信号**：
   - 可用于指导计算资源分配与缓存管理；
   - 不需要额外训练或微调。

---

### 方法的局限性

- **短上下文收益有限**：
  - 引入 observation attention 和 entropy profiling 开销，在短 context 下难以体现优势；
  - 但在长 context 中该开销占比迅速降低，不影响整体加速。

- **对极小模型适配性待验证**：
  - 当前实验集中在 7B~8B 级别模型；
  - 在 <1B 模型上是否依然有效需进一步探索。

- **未结合量化或 offloading 技术**：
  - 当前为纯 selection-based 方法；
  - 可与 KVQuant、KVSwap 等正交技术叠加优化。

---

### 未来工作方向

1. **将 entropy signal 扩展至 layer-level 自适应调度**；
2. **结合量化（quantization）与 entropy guidance 实现联合压缩**；
3. **应用于多模态 LLM 的 cross-modal attention 优化**；
4. **探索 entropy 在 speculative decoding 中的应用潜力**。

---

> ✅ **总结一句话**：  
> **EntropyInfer 通过在线感知 attention entropy 的动态特性，在 prefill 和 decode 两阶段实现了 head-level 的精细化资源调控，在 >100K 长上下文场景下达成高达 2.39× 的端到端加速，同时几乎无损生成质量，是一种高效、通用、免训练的推理加速方案。**

🔗 **开源地址**：[https://github.com/SHA-4096/EntropyInfer](https://github.com/SHA-4096/EntropyInfer)

</details>

---

### 3. [Decoding Naturalistic Emotion Dynamics from the Brain: An LLM-Enhanced Regression Framework](https://arxiv.org/abs/2606.07707)

**Authors**: Lemei Zhang, Peng Liu, Hans Dahle Kvadsheim, August S{\ae}tre Aasv{\ae}r, Shuer Ye, Reza Bonyadi, Maryam Ziaei, Jon Atle Gulla  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.07707v1  

#### Abstract
Decoding emotional states from neural signals has been typically framed as a discrete, single-label classification task based on emotionally stable stimuli, a formulation that oversimplifies the continuous, fluid, and co-occurring nature of human affect. This study reconceptualizes emotion decoding ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Decoding Naturalistic Emotion Dynamics from the Brain: An LLM-Enhanced Regression Framework

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的情绪解码研究通常将情绪视为**离散、单一标签的分类任务**，并依赖于静态刺激（如短片段音乐或图片），这忽略了人类情感在现实世界中**连续、动态且多维共存**的本质。此外，人工标注自然主义叙事中的连续情绪轨迹成本高昂、主观性强，限制了大规模神经影像数据集的应用。

本研究旨在解决以下核心问题：
- 如何对**自然主义叙事中连续、重叠的情绪维度**进行建模？
- 如何在缺乏大量人工标注的情况下，为fMRI数据提供**高时间分辨率的情绪标签**？
- 如何揭示支持动态情绪处理的**分布式脑网络机制**？

### 提出了什么新方法或新思路
本文提出了一种**LLM增强的回归框架**（LLM-Enhanced Regression Framework），其核心创新如下：

- **多目标回归范式**（Multi-target regression framework）  
  将情绪解码从传统的单标签分类转变为对**Plutchik八维情绪模型**（如Joy, Sadness, Anticipation等）的连续强度预测，从而捕捉情绪的**动态轨迹与共现性**。

- **LLM自动化情绪标注**（LLM-based Sentiment Annotation）  
  利用**GPT-4**对《爱丽丝梦游仙境》的听觉叙事文本进行细粒度情感评分，生成每段文本的**连续情绪向量**作为“代理真值”（proxy labels），解决了自然主义刺激中标注稀缺的问题。

- **基于动态功能连接**（Dynamic Functional Connectivity, DFC）  
  使用滑动窗口计算**400个Schaefer脑区之间的时变功能连接矩阵**，而非仅依赖静态ROI激活幅度，以捕捉大脑网络随时间演变的交互模式。

- **图论可解释AI**（Graph-theoretical XAI）  
  对回归模型的特征重要性矩阵构建**最小生成树**（Minimum Spanning Tree, MST），结合**加权度中心性**（Weighted Degree）、**中介中心性**（Betweenness Centrality）、**模块化**（Modularity）等图指标，实现对情绪特异性脑网络拓扑结构的可解释分析。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|--------|
| 情绪建模 | 单标签分类、静态 | 多目标回归、连续动态 |
| 情绪标签来源 | 人工标注、稀疏 | LLM自动生成、高密度 |
| 脑信号表征 | 静态ROI激活 | 动态功能连接（DFC） |
| 可解释性 | 质量差（黑箱） | 图论驱动、神经科学可解释 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Alice Dataset**（OpenNeuro ds002322）  
  - 包含26名被试（15女，11男）在fMRI扫描期间聆听《爱丽丝梦游仙境》第一章的听觉版本。
  - 多模态数据：fMRI、EEG、行为数据。
  - fMRI参数：TR=2s，全脑覆盖，空间分辨率为~3mm³。
  - 脑区分割：采用**Schaefer 400-parcel atlas**，划分为7或17个大尺度网络。

### 实验设置
- **文本分段与对齐**  
  - 文本按**8秒窗口**分段，允许4秒重叠以保持语义完整性。
  - 每段输入GPT-4生成8维情绪强度向量（0–1）。
  - fMRI数据通过时间戳对齐至最近的TR点，形成同步的`DFC矩阵 ↔ 情绪向量`配对。

- **DFC构建**  
  - 滑动窗口长度：**12 TRs ≈ 24秒**，平衡时间分辨率与统计可靠性。
  - 功能连接：窗口内所有ROI间的**皮尔逊相关系数**构成400×400矩阵。

- **机器学习模型**
  在**ROI振幅**和**DFC矩阵**两种特征上训练六种回归模型：
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - SVR（RBF核）
  - Linear SVR
  - Random Forest Regressor (RFR)

- **评估指标**
  - **R²**（决定系数）：越高越好，衡量模型解释方差的能力。
  - **MSE**（均方误差）：越低越好，衡量预测与真实标签的偏差。
  - 数据划分：**90-10** 和 **80-20** 时间顺序分割（非随机交叉验证），以反映叙事的时间演化特性。

- **基线方法对比**
  - **特征层面**：比较 `ROI-based` vs `DFC-based` 表征。
  - **模型层面**：比较线性 vs 非线性模型（如SVR vs Ridge）。
  - **理论层面**：检验**定位主义**（locationist） vs **建构主义**（constructionist）情绪理论。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（90-10 split，Test Set）

| 模型 | 数据类型 | Test R² | Test MSE |
|------|---------|--------|--------|
| **SVR (RBF)** | DFC | **0.5931** | **0.0047** |
| Ridge Regression | DFC | 0.5589 | 0.0050 |
| Lasso Regression | DFC | 0.4742 | 0.0058 |
| Linear Regression | DFC | 0.3955 | 0.0066 |
| **SVR (RBF)** | ROI | 0.2621 | 0.0161 |
| RFR | ROI | 0.1978 | 0.0174 |

> ✅ **关键发现**：  
> - 所有模型在**DFC特征上显著优于ROI特征**，表明情绪信息更多编码于**网络间动态交互**而非局部激活。
> - **SVR (RBF)** 在DFC上达到最高R²（0.5931），说明存在非线性关系。
> - **线性模型**（如Ridge）在DFC上表现接近最优，说明高维协方差结构已足够支持良好预测。

### 与基线方法的对比结果
- **DFC vs ROI**：DFC模型的R²平均高出**100%以上**，MSE降低约**67%**。
- **LLM标签有效性**：人类评估显示GPT-4标注的平均一致性得分为**7.83/10**，显著高于其他LLM（如LLaMA、Mixtral），证明其可靠性。
- **非线性优势有限**：尽管SVR最优，但**Ridge回归**（线性）性能紧随其后（R²=0.5589），表明正则化线性模型足以捕获主要信号。

### 消融实验结果
- **Leave-One-Network-Out 分析**（表3 & 4）  
  - 移除任一网络对整体性能影响极小（最大MSE变化<0.5‰），说明情绪信息是**高度分布式的**。
  - 特别地，移除**Ventral Attention Network**或**Visual Network**反而轻微提升性能，提示这些网络可能引入噪声或冗余。
  - 但联合移除两者导致多数模型性能下降，表明它们仍贡献**非平凡的预测方差**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **动态功能连接优于静态区域激活**  
   DFC能更有效捕捉情绪的连续变化，支持**心理建构主义理论**——情绪是分布式网络动态交互的产物，而非特定脑区的孤立活动。

2. ✅ **LLM可用于生成高质量、可扩展的情绪标签**  
   GPT-4能稳定输出符合语义的情感强度，为自然主义神经科学研究提供了**低成本、高通量的标注方案**。

3. ✅ **不同情绪具有独特而可解释的脑网络拓扑签名**  
   - **Limbic Network A**（含眶额叶、颞极）在几乎所有情绪中都表现出**高加权度与中介中心性**，是核心枢纽。
   - **Positive emotions**（如Joy, Trust）：
     - 更高的**全局效率**（Global Efficiency）
     - 更小的**树直径**（Tree Diameter）
     - 表明更紧凑、整合更强的网络组织。
   - **Negative emotions**（如Disgust, Sadness）：
     - 更高的**模块化**（Modularity）
     - 更大的**树直径**
     - 表明网络更分离、信息传递路径更长。

4. ✅ **情绪间存在结构性相似性**  
   - **Anticipation 与 Surprise**：共享**Somatomotor与视觉网络**的强连接，反映对突发刺激的准备状态。
   - **Anger 与 Sadness**：共享**Limbic与Control网络**，体现负价态下的认知调节。
   - **Fear 与 Disgust**：未见明显拓扑重叠，可能因Fear依赖杏仁核、Disgust依赖岛叶等**未包含在皮层图谱中的亚皮层结构**。

### 方法的局限性
- **子群体泛化能力受限**：样本量较小（N=26），年龄与文化背景单一。
- **标签非主观报告**：使用LLM生成的“叙事情绪”而非被试的真实感受，可能存在**语义混淆**（semantic confound）。
- **皮层限制**：Schaefer图谱仅涵盖皮层，**忽略杏仁核、海马体等关键边缘系统结构**。
- **时间分辨率权衡**：24秒滑动窗口可能无法捕捉毫秒级情绪波动。
- **模型假设**：特征重要性分析隐含独立性假设，但在高维连接数据中特征高度共线。

### 未来工作方向
- 引入**多模态融合**（如EEG+fMRI）以提高时间精度。
- 开发**个性化解码模型**，考虑个体差异（如人格、情绪敏感性）。
- 探索**因果连接方法**（如Granger Causality）替代相关性DFC。
- 结合**被试自我报告**验证LLM标签的心理效度。
- 构建**端到端深度学习架构**（如Transformer+GNN）替代分阶段流程。

---

> 📌 **总结一句话**：  
> 本研究通过**LLM+DFC+XAI**三位一体框架，首次实现了对自然主义叙事中**连续、多维情绪动态**的高精度脑解码，并为情绪的心理建构主义理论提供了强有力的神经证据。

</details>

---

### 4. [Breaking the Bubble: Asynchronous Pipeline Parallel Training with Bounded Weight Inconsistency](https://arxiv.org/abs/2606.07881)

**Authors**: Itay Elam, Eliron Rahimi, Avi Mendelson, Chaim Baskin  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.07881v1  

#### Abstract
Pipeline parallelism is essential for training large neural networks, but existing schedules trade off throughput, memory, and optimization consistency. Synchronous pipelines preserve forward/backward weight consistency but suffer from bubbles; asynchronous pipelines remove bubbles but introduce wei...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Breaking the Bubble: Asynchronous Pipeline Parallel Training with Bounded Weight Inconsistency*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模神经网络（尤其是 LLM）训练中，**pipeline parallelism** 是提升硬件利用率的关键技术。然而，现有方法面临以下权衡：
- **Synchronous pipeline**（如 1F1B-flush）保持前向/反向权重一致性，但存在严重的 **pipeline bubbles**，导致设备空闲，降低吞吐。
- **Asynchronous pipeline** 可消除 bubbles，但引入 **forward/backward weight-version inconsistency**（即同一个 micro-batch 的前向和反向使用不同版本的参数），影响训练稳定性。

传统异步方法通常通过 **weight stashing**、**prediction** 或 **额外参数副本** 来缓解不一致性，但这带来了更高的内存开销或系统复杂度。

### 提出的新方法：PACI
本文提出 **PACI**（Pipeline Asynchronous training with Controlled Inconsistency），一种无需全局同步、无额外内存开销的异步 pipeline 方法，其核心思想是：

> **将 local gradient accumulation 视为一种版本控制机制**，而非仅用于增大有效 batch size。

通过控制梯度累积因子 $a$，减缓参数更新频率，从而限制任意 micro-batch 在前向与反向之间所跨越的 optimizer 更新次数（即版本漂移 $\Delta$），实现对 inconsistency 的显式上界控制。

### 相比现有方法的优势
| 特性 | PACI | 同步方法（如 1F1B-flush） | 异步方法（如 PipeDream） |
|------|------|--------------------------|------------------------|
| Pipeline bubbles | **0**（完全消除） | 高（依赖 micro-batch 数量） | 0 |
| F/B inconsistency | **低且有界**（由 $a$ 控制） | 0 | 高（需 stashing/prediction） |
| 额外 weight 内存 | **0** | 0 | +++（stashing 开销大） |
| 全局同步 | **无** | 有（flush） | 无 |
| 系统复杂度 | 低 | 中等 | 高 |

**核心优势**：  
PACI 实现了前所未有的操作点——**零 bubbles、零额外权重内存、低且可控的 inconsistency**，无需 weight stashing、prediction 或全局同步。

---

## 2. 核心实验方法和设置

### 数据集
- **OpenWebText**：用于从头训练 GPT-style 模型。
- **模型**：GPT-2 Medium（约 354M 参数）。
- **序列长度**：1024。
- **总训练 token 数**：固定为 49.8B tokens。

### 实验设置
- **Pipeline stages**：8-stage pipeline parallelism，无 data parallelism。
- **精度**：BF16。
- **优化器**：AdamW（$\beta_1=0.9, \beta_2=0.95$），学习率按 [18] 调整以保证跨 batch size 可比性。
- **微批次配置**：
  - Global batch size：128 和 256。
  - Micro-batch 数量 $m$ 和 accumulation factor $a$ 变化。
- **硬件**：单节点 8-GPU（NVIDIA RTX PRO 6000 Blackwell Max-Q，96GB 显存），PCIe 连接。
- **未使用**：activation checkpointing、ZeRO、FSDP。

### 评估指标
- **训练稳定性**：训练 loss 动态、run-to-run 方差。
- **最终性能**：validation perplexity。
- **效率**：
  - Wall-clock **time-to-accuracy**（达到特定 PPL 所需时间）
  - **Throughput**（tokens/sec/GPU）
  - **Peak memory usage**（每设备最大 GPU 内存）
- **理论验证**：throughput 是否符合 pipeline-efficiency 模型预测。

### 基线方法对比
- **Synchronous baseline**：1F1B-flush（标准同步流水线）
- **其他参考方法**：ZB-2p、1F1B-I 等（见 Table 1 和 Table 2）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **训练稳定性和最终质量**
- 在 $\Delta_{\text{max}} \leq 2$ 的低 inconsistency 区域，PACI 的 **loss 曲线与 1F1B-flush 几乎完全重合**（图 5）。
- 最终 validation perplexity 与 flush 基线 **相当甚至略优**（表 3）：
  - Batch size 128：PACI(a=8) 达到 15.483 ± 0.028 vs Flush 15.480 ± 0.265
  - Batch size 256：PACI(a=16) 达到 15.291 ± 0.008 vs Flush 15.350 ± 0.089
- **Run-to-run variability 显著降低**：
  - Batch 128 下，RMS std 从 1.10×10⁻²（flush）降至 1.81×10⁻³（PACI a=8）

#### ⚡ **训练速度提升（Time-to-Accuracy）**
- PACI 显著缩短达到目标 perplexity 的 wall-clock 时间（图 1）：
  - **Batch size 128**：
    - 相比同 micro-batch 数量的 flush：**最高提速 2.04×**
    - 相比最快 flush 配置：仍快 **1.69×**
  - **Batch size 256**：
    - 相比同配置：**最高 1.51×**
    - 相比最快 flush：**1.41×**

> 表 4 和 表 6 提供详细 speedup 数据，显示 PACI 在早期到晚期训练阶段均持续领先。

#### 📈 **吞吐量与内存表现**
- **Throughput**：
  - PACI 吞吐几乎恒定，不受 micro-batch 数量影响（图 9），表明 **无 bubble 开销**。
  - 1F1B-flush 吞吐随 $m$ 增加而上升，但增速受限于 kernel efficiency 下降（小 micro-batch 导致计算粒度变细）。
  - PACI 与 flush 的吞吐比 **完美匹配理论 bubble-efficiency 模型**（图 3），验证了 bubble 是性能瓶颈。
- **Memory**：
  - PACI 与 1F1B-flush 的 **peak memory 完全相同**（图 7），证实无额外内存开销。
  - 在大模型扩展预测中（Table 2），PACI 吞吐接近 ZB-2p，但内存仅为 1F1B-flush 水平。

---

## 4. 关键结论和发现

### 主要发现
1. **Forward/backward inconsistency 不必完全消除**：只要将其控制在合理范围内（如 $\Delta \leq 2$），即可安全地换取显著的效率增益。
2. **Gradient accumulation 可作为版本控制机制**：PACI 创新性地将 accumulation 用于控制参数演化速度，而非仅用于 batch size 扩展。
3. **效率提升直接转化为更快的 time-to-accuracy**：去除 bubbles 不仅提高 raw throughput，更加快了实际训练收敛速度。
4. **PACI 实现了最优权衡点**：在 **零 bubbles、零额外内存、低 inconsistency** 三者之间取得平衡，优于所有现有方法。

### 方法的局限性
1. **实验范围有限**：
   - 仅在 GPT-2 Medium 和 OpenWebText 上验证。
   - 未测试更大模型、更深 pipeline（>8 stages）、多模态任务等。
2. **未支持 activation checkpointing**：
   - 虽理论上兼容，但会改变 inconsistency 结构（recomputed activations 使用新参数），需进一步研究。
3. **不支持全局梯度裁剪**：
   - 全局梯度范数需要同步，与完全异步执行冲突。可考虑 SPAM 等无同步替代方案。
4. **缺乏梯度无效处理机制**：
   - 如出现 NaN 梯度，无法 rollback 已更新的 stage，可能影响容错性。

### 未来工作方向
- 将 PACI 扩展至 **更大规模模型**（百亿/千亿参数）和 **更复杂并行策略**（如 3D parallelism）。
- 研究 **结合 activation checkpointing 的 PACI 变体**，分析其 inconsistency 特性。
- 探索 **无同步的梯度 spike mitigation 方法**（如 SPAM）以支持更鲁棒的训练。
- 设计 **自适应 accumulation factor** 策略，在训练不同阶段动态调整 inconsistency 上界。

---

> **代码开源**：https://github.com/ItayElam/PACI

</details>

---

### 5. [Beyond FLOPs: Benchmarking Real Inference Acceleration of LLM Pruning under a GEMM-Centric Taxonomy](https://arxiv.org/abs/2606.09080)

**Authors**: Haozhe Hu, Hao Wu, Anhao Zhao, Longwei Ding, Peiran Yin, Yunpu Ma, Xiaoyu Shen  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.09080v1  

#### Abstract
Pruning has emerged as a dominant paradigm for accelerating large language model (LLM) inference, spanning a broad spectrum of methods that remove computation across tokens, layers, heads, dimensions, and attention patterns. Despite sharing the same objective, these pruning approaches induce fundame...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond FLOPs: Benchmarking Real Inference Acceleration of LLM Pruning under a GEMM-Centric Taxonomy

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 LLM pruning 方法虽然在减少 FLOPs 上表现出色，但由于不同方法对硬件执行行为（如内存访问、kernel 调度）的影响差异巨大，**理论加速（如 FLOPs 减少）与实际推理加速之间存在严重脱节**。这导致难以公平比较不同 pruning 方法的真实部署收益。

此外，现有研究常依赖于不一致的评估标准（如不同的模型架构、硬件优化），使得跨方法比较变得不可靠。

### 提出了什么新方法或新思路
本文提出了一种 **GEMM-centric taxonomy**，将多样化的 LLM pruning 方法统一映射到 GEMM 运算的三个逻辑维度上：

- **M 维度**：对应 token 数量（序列长度）
- **N 维度**：对应输出特征维度
- **K 维度**：对应输入特征维度（reduction dimension）

通过这一抽象，作者构建了一个**统一的推理基准框架（benchmarking framework）**，能够实现：
- 方法无关的实现一致性比较
- 跨平台 kernel baseline 支持（基于 Triton 和 Tilelang 等 DSL）
- 插件式测试新算法的能力

该框架不仅支持端到端延迟测量，还提供 kernel-level 分析能力，揭示加速瓶颈来源。

### 相比现有方法的优势
- **去碎片化评估**：解决了因硬件依赖和实现差异造成的评估混乱。
- **操作级可解释性**：从 GEMM 维度出发，清晰解释了为何某些 pruning 方法即使稀疏度相同，实际加速效果却大相径庭。
- **系统性指导意义**：为未来 pruning 设计提供了明确的方向——不仅要关注“剪了多少”，更要关注“怎么剪”及其对执行流的影响。

---

## 2. 核心实验方法和设置

### 使用的数据集
下游任务评估采用 `lm-evaluation-harness` 统一评测套件，包含以下基准：
- **WikiText2**（ppl）
- **ARC-e / ARC-c**
- **BoolQ**
- **WinoGrande**
- **PIQA**
- **OpenBookQA**
- **HellaSwag**

所有任务均以 zero-shot 方式评估，最大上下文长度设为 4,096。

### 实验设置和评估指标

#### 模型与硬件
- **主模型**：Llama3.1-8B
- **辅助验证模型**：Qwen3-14B
- **主硬件平台**：RTX Pro 6000 Blackwell (sm120)
- **辅助平台**：A800-80G (sm80)

精度配置：
- RTX Pro 6000：bf16
- A800：fp16（因 sm80 不支持 bf16 atomic add）

#### 评估指标
- **吞吐量（Throughput）**：
  $$
  \text{Token/s} = \frac{B \times T_q}{\text{TTFT 或 TPOT}}
  $$
  其中 $B$ 是 batch size，$T_q$ 是 token 数量，TTFT 为 Time-to-First-Token，TPOT 为 Time-Per-Output-Token。
- **加速比（Speedup）**：相对于 dense 模型的吞吐提升倍数。
- **质量损失（Accuracy Gap）**：平均准确率下降百分比。
- **Pareto Frontier**：综合考虑速度与质量的最优权衡边界。

#### 实现细节
- 所有 pruning 方法通过统一接口替换原生模块。
- 使用 CUDA graph 最小化 CPU 开销。
- 对 semi-structured sparsity 使用 cuSPARSELt 并绕过 PyTorch 默认开销路径。
- 动态 pruning 引入 mask reordering 和 tile-based skipping 机制。

### 基线方法对比
选取代表每类 pruning taxonomy 的典型方法作为代表：

| Taxonomy | Method | Type |
|--------|--------|------|
| Static M | Shortened-taylor / CoopPruner | Depth pruning |
| Static K (low-rank) | Dobi-SVD | Low-rank approximation |
| Static K | MaskLLM | Semi-structured sparsity |
| Static NK | Tyr-the-Pruner | Width pruning (N/K coupled) |
| Static NK (cross-layer) | SliceGPT+ | Cross-layer width pruning |
| Dynamic M | SkipGPT | Token-adaptive depth pruning |
| Dynamic NK | SeerAttention | Sparse attention |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2 & Figure 1）

#### 在 50% 稀疏度下的 decode 阶段加速表现：
| Method | Decode Speedup | Accuracy Gap |
|-------|----------------|--------------|
| **Static M** | **1.91x** | 30.83% |
| Static NK | 1.70x | 26.40% |
| Dynamic M | 1.10x | 15.59% |
| Static K (low-rank) | 1.46x | 20.46% |
| Static K | 1.14x | 11.25% |

> ✅ **Static M 在所有方法中实现了最高的实际加速比**，且最接近其理论上限。

#### 在低质量损失（<5%）区域的表现（prefill 阶段）：
- **Static M** 在仅 2.85% 性能损失下达到 **1.12x 加速**
- 显著优于其他方法，成为低损场景下的最强 baseline

#### Pareto Frontier 演变趋势（Figure 1）：
- **0–4% 质量损失**：Static M 占据主导
- **5–16% 质量损失**：Dynamic M 成为主力，prefill 加速达 **1.24x~1.44x**
- **>17% 质量损失**：Static NK 成为宽度剪枝中最优选择，最高达 **1.77x 加速**

> 🔁 表明：**没有单一最优方法，最佳选择取决于可用的质量预算**

### 消融实验结果

#### （1）Sparsity 分布影响（Figure 7）
- 当 attention 与 FFN 的计算负载平衡时（约 32K context），加速效果最佳
- 实际中多数方法对 attention 施加更高稀疏度，在长上下文下更有效

#### （2）Dimension Alignment 影响（Table 3 & 4）
- 未对齐的 pruning 导致高达 **35% 的加速损失**
- 特别是在低精度（fp8）下，K 维度对齐要求极为敏感（需对齐至 16 的倍数才能恢复 70% 性能）

#### （3）N vs K 维度剪枝对比（Table 5 & 6）
- **Prefill 阶段**：
  - K-dimension pruning 提供更稳定的速度提升（直接减少循环次数）
  - N-dimension pruning 受 tile 数变化影响，加速非线性
- **Decode 阶段（batch=1）**：
  - N-dimension pruning 更优，因其保留原始 K-loop 结构，避免 split-k 引入的 pipeline 开销

#### （4）Custom Kernel 性能（Table 7–9）
- **Dynamic M 的 GEMM kernel** 在 M > 512 后才开始超越 dense
- **Sparse attention（Dynamic NK）** 在 prefill 中需超过 32K query tokens 才能发挥优势
- **Decode 阶段**，Dynamic M 接近 2x 加速，得益于 token-level skipping 与 GEMV 优化结合

---

## 4. 关键结论和发现

### 主要发现
1. **名义稀疏度（nominal sparsity）是弱预测因子**  
   实际加速由 pruning taxonomy、结构传播方式、operator 覆盖范围和系统开销共同决定。

2. **Static M 是当前 throughput anchor**  
   尽管牺牲了高稀疏度下的质量灵活性，但它提供了最稳定、最接近理论极限的实际加速，尤其适合内存受限场景。

3. **Pareto Frontier 随质量预算动态转移**  
   - 低损 → Static M
   - 中等损 → Dynamic M
   - 高损 → Static NK
   > 这是首次系统刻画出 pruning 方法的实用加速边界。

4. **Dynamic M 和 Static K 受限于非 GEMM 开销**  
   - Dynamic M：mask preprocessing、routing、graph launch 开销显著
   - Static K：metadata 处理、split-k 激活不足、launch overhead 高

5. **宽度剪枝仍有巨大优化空间**  
   当前方法未能充分挖掘 Static NK 的潜力，尤其是在低稀疏度区域搜索策略尚不成熟。

### 方法的局限性
- **聚焦 taxonomy-level 行为**，未覆盖所有 pruning 变体。
- **未评估 MoE 架构或数据中心级 Blackwell 芯片**，可能影响 dynamic pruning 的相对优势。
- **Triton/Tilelang 实现为通用 baseline**，可能低于手工调优 CUDA kernel 的性能。
- **未集成完整生产推理框架（如 SGLang）中的调度与分布式逻辑**。

### 未来工作方向
- 探索 **hybrid pruning design**：结合 quality-aware branch（如 dynamic M）与 structured static backbone
- 开发针对 **non-GEMM overhead 的专用优化技术**（如 fused routing + mask reorder）
- 构建面向 MoE、多模态、agent workflow 的扩展 benchmark
- 推动 **hardware-software co-design**，例如利用 TMA、async TC instructions 提升 dynamic pruning 效率

---

> 📌 **最终洞见**：  
> “The next Pareto frontier will likely come not from pushing a single taxonomy in isolation, but from hybrid designs that combine a quality-aware branch with structured and static backbone.”  
> —— 下一代高效 LLM 推理，属于**动静结合、软硬协同**的设计范式。

</details>

---

### 6. [A Multi-Agent System for IPMSM Design Optimization via an FEA-AI Hybrid Approach](https://arxiv.org/abs/2606.09037)

**Authors**: Jinseong Han, Sunwoong Yang, Namwoo Kang  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.09037v1  

#### Abstract
Interior permanent magnet synchronous motor (IPMSM) design requires balancing conflicting objectives and multi-physics constraints, while modern optimization workflows face three bottlenecks: manual problem setup, high finite element analysis (FEA) cost, and unreliable surrogate-based search in spar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

**论文标题**: *A Multi-Agent System for IPMSM Design Optimization via an FEA-AI Hybrid Approach*

---

## 1. 主要贡献和创新点

### 解决的问题
该论文针对 **Interior Permanent Magnet Synchronous Motor (IPMSM)** 设计优化中的三大瓶颈问题：

1. **问题定义的手动负担重**：工程师需手动设定设计变量、目标函数、约束条件等，尤其对初级工程师而言门槛高且易出错。
2. **有限元分析 (FEA) 驱动优化计算成本极高**：基于遗传算法 (GA) 的搜索需要对大量候选几何体进行高保真度 FEA 模拟，导致计算资源消耗巨大。
3. **AI代理模型可靠性低**：虽然 AI-surrogate 可加速评估，但在稀疏或分布外 (OOD) 区域预测不确定性高，可能导致搜索收敛到不可靠的局部最优解。

### 提出的新方法与创新思路
提出了一种端到端自动化的多智能体系统框架，结合 **检索增强生成 (RAG)** 和 **不确定性感知的 FEA-AI 混合优化流程**，具体创新如下：

- **RAG 支持的问题定义 (Design Agent)**  
  利用连接电机教科书的 RAG 技术，在自然语言交互中提供领域知识支持，引导用户完成结构化的问题定义（目标、变量、约束），生成可执行的优化卡 (optimization card) 和 DOE 计划，显著降低专家依赖性和配置歧义。

- **自动化训练数据生成与可行性恢复 (Training Agent)**  
  引入一个由 LLM 推理驱动的 **自主重采样循环 (autonomous resampling loop)**。当初始设计空间产生大量因几何冲突或网格失败而无效的样本时，该模块通过 ANOVA 分析失败原因，并结合日志推理自动调整设计空间边界，重新生成有效样本，确保训练数据集的质量和数量。

- **不确定性感知的混合优化策略 (Optimization Agent)**  
  在 GA 搜索过程中引入 **基于不确定性的切换机制 (uncertainty-threshold switching)**：
  - 低不确定性候选 → 使用 AI-surrogate 快速评估；
  - 高不确定性或 Pareto 前沿/Top-K 候选 → 调用高保真 FEA 进行验证并用于在线更新 surrogate 模型。
  该机制在控制 FEA 成本的同时提升了决策可信度。

- **面向电机设计的专用多智能体工作流**  
  构建了一个集成的多智能体系统（Design, Training, Optimization Agents），覆盖从需求理解到最终设计推荐的完整流程，实现全流程自动化与可复现性。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **问题定义** | 手动、经验依赖、易出错 | 自动化、RAG 增强、减少幻觉 |
| **数据生成** | 失败样本丢弃或保守限界 | 自主分析失败原因并重采样，扩大探索范围 |
| **优化策略** | FEA-only（昂贵）、AI-only（不可靠） | FEA-AI 混合，动态按需调用 FEA，平衡效率与可靠性 |
| **系统集成** | 孤立工具链 | 全流程闭环多智能体协同 |

---

## 2. 核心实验方法和设置

### 数据集
- **无公开外部数据集**，所有数据均通过 **电磁 FEA 仿真自动生成**。
- 使用 **Ansys Maxwell 2D** 对参数化 IPMSM 几何体进行高保真度电磁场仿真，提取关键响应如磁通密度 $B$、铁损 (core loss)、平均转矩等。
- 训练数据来源于 **100 个分析可行的采样点**（经由 LLM 驱动的重采样循环获得）。

### 实验设置
- **任务**：单目标优化 — 最小化铁损 (iron loss)。
- **设计变量**：共 12 个几何参数（如 PM 角度、长度、气隙、定子齿高等）。
- **智能体架构**：
  - 使用本地部署的轻量级开源 LLM **GPT-OSS 20B** 作为各 Agent 的推理引擎。
  - 结合 **LangChain** 实现智能体编排。
- **Surrogate 模型**：
  - 采用 **深度集成 (deep ensemble)** 架构（5 个 MLP 并行），输出预测值 $p(x)$ 与方差 $\sigma^2(x)$。
  - 使用 **B-NLL 损失函数** 训练异方差不确定性，提升稳定性。
- **优化器**：标准 GA（单目标），种群大小 25，代数 30。
- **不确定性度量**：使用 **变异系数 CV (%)**：
  $$
  \text{CV}(x) = \frac{\sqrt{\sigma^2(x)}}{|p(x)|} \times 100\%
  $$

### 评估指标
- **目标性能**：最终选出的最优设计对应的铁损值（kW）。
- **计算成本**：总 FEA 调用次数、总耗时（小时）。
- **可靠性指标**：
  - Pareto 前沿或 Top-K 候选的平均 CV（%）。
  - 是否发生“过早收敛”或“信任错误代理”现象。
- **对比基线**（在相同 150 次 FEA 预算下）：
  1. **FEA-only GA**：全部评估使用 FEA，预算耗尽即停止。
  2. **AI-only GA**：先用 150 个 FEA 样本训练一次 surrogate，后续所有评估均由 AI 完成，无在线修正。
  3. **Proposed Hybrid GA**：用 100 个样本预训练 surrogate，剩余 50 次 FEA 预算用于不确定性触发的在线校正（阈值 CV=3%）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（四次随机种子平均结果）

| 方法 | 最优铁损 (kW) | 总计算时间 (h) | Top-K 平均 CV (%) | FEA 调用总数 |
|------|----------------|------------------|--------------------|---------------|
| FEA-only GA | 1.6780 | 7.11 | ~0%（纯 FEA） | 150 |
| AI-only GA | 1.8098 | 7.12 | 8.5% | 150（仅用于训练） |
| **Proposed Hybrid GA** | **1.6658** | **7.12** | **5.1%** | **150** |

> ✅ **结论**：在完全相同的 FEA 预算下，所提混合方法取得了**最低的目标损失值**，同时将预测不确定性控制在远低于 AI-only 方法的水平。

### 与基线方法对比结果
- **vs FEA-only GA**：
  - 尽管两者耗时相近，但 FEA-only 因预算耗尽过早（约第 8 代结束），无法充分探索设计空间，导致性能较差（+0.0122 kW）。
  - Hybrid GA 利用 AI 加速低风险评估，实际探索了约 **1200 个候选设计**，远超 FEA-only 的能力。
- **vs AI-only GA**：
  - AI-only 因缺乏在线 FEA 修正，逐渐信任低置信度预测，收敛至性能最差的设计（+0.144 kW），且不确定性高达 8.5%，结果不可靠。
  - Hybrid GA 通过主动学习不断用高价值 FEA 数据更新模型，避免陷入虚假最优。

### 消融实验与敏感性分析
#### 不确定性切换阈值 (CVTh,hybrid) 敏感性分析
测试不同 CV 阈值下的表现（固定五轮优化）：

| CV 阈值 | 最终铁损 (kW) | 总 FEA 调用 | Top-K CV (%) | 表现特征 |
|--------|----------------|--------------|----------------|----------|
| 1% | 1.5676 | 226 | 2.01% | 性能最好但成本最高，每轮几乎恒定调用 FEA |
| **3%** | **1.6385** | **170.5** | **2.11%** | **最佳权衡点**：性能接近 1%，但节省近 50 次 FEA |
| 5% | 1.7396 | 139.2 | 2.97% | 开始退化，后期几乎不调用 FEA |
| 10% | 1.9366 | 116 | 6.98% | 接近 AI-only，严重低估不确定性，性能大幅下降 |

> 🔍 发现：**CV=3% 是最优操作点**，能在保持高可靠性的同时显著降低计算开销。

#### RAG 消融研究（问题定义质量）
在 90 个专业电机问题上比较有无 RAG 的表现（GPT-OSS 20B）：

| 设置 | 数值题准确率 | 书籍特定值 | 概念题准确率 |
|------|----------------|-------------|----------------|
| No-RAG | 43% | 3% | 47% |
| **With RAG** | **77%** | **67%** | **80%** |

> ✅ RAG 显著提升技术建议的准确性与实用性，尤其在依赖具体文献知识的任务上效果突出。

---

## 4. 关键结论和发现

### 主要发现
1. **RAG 可有效提升工程问题定义的质量**，减少非专家用户的配置错误和模型幻觉，使优化起点更可靠。
2. **LLM 驱动的重采样循环能有效解决“设计-分析”可行性缺口**，在宽泛初始空间中仍能稳定获取足够数量的有效训练样本。
3. **不确定性感知的 FEA-AI 混合策略优于纯 FEA 或纯 AI 方法**：
   - 在相同 FEA 预算下，**Hybrid GA 实现了最佳性能与可靠性的平衡**。
   - 动态切换机制使得计算资源集中在最关键的设计区域，实现了高效探索。
4. **CV=3% 是实践中理想的不确定性切换阈值**，兼顾性能、成本与可靠性。

### 方法的局限性
1. 当前框架局限于**参数化几何表示**，尚未扩展到拓扑优化等更灵活的设计空间。
2. 实验为**单目标优化**，多目标 Pareto 前沿优化尚未深入验证。
3. 物理模型仅考虑**电磁 FEA**，未整合热、机械应力、振动等多物理场约束。
4. 切换阈值 **CVTh,hybrid 为人工设定**，缺乏自适应调节机制。
5. RAG 模块的知识库依赖于单一教材，覆盖面有限，影响推荐广度。

### 未来工作方向
1. 扩展至 **非参数化/拓扑优化设计空间** 和 **多目标 Pareto 优化** 场景。
2. 集成 **多物理场 FEA 自动化**（热、结构、NVH），构建全链条多学科设计优化 (MDAO) 流程。
3. 开发 **自适应不确定性切换策略**，根据 GA 进展、预算状态和 UQ 信号动态调整 CV 阈值。
4. 扩充 RAG 知识库，纳入工业设计规范、制造工艺指南、材料数据库等，提升实用性和鲁棒性。
5. 探索更大规模应用场景（数千次评估），进一步验证混合策略在大规模优化中的优势。

---

> 📌 **总体评价**：本文提出的多智能体 FEA-AI 混合框架不仅提升了 IPMSM 设计优化的效率与可靠性，更重要的是实现了从“人工配置 → 自动执行”的范式转变，为复杂工程系统的智能化设计提供了可复制、可追溯的新路径。

</details>

---

### 7. [Reformulate LLM Reinforcement Learning for Efficient Training under Black-box Discrepancy](https://arxiv.org/abs/2606.08779)

**Authors**: Jiashun Liu, Runze Liu, Xu Wan, Jing Liang, Hongyao Tang, Ling Pan  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.08779v1  

#### Abstract
Reinforcement Learning (RL) has emerged as a pivotal post-training paradigm, yet it frequently suffers from unpredictable sub-optimum performance or even training collapses. Recent findings attribute these failures to a hidden train-inference discrepancy (or mismatch), stemming from the disparate un...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Reformulate LLM Reinforcement Learning for Efficient Training under Black-box Discrepancy

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对 **Large Language Models (LLMs)** 在 **Reinforcement Learning (RL)** 后训练过程中普遍存在的 **train-inference discrepancy**（训练-推理差异）问题。这种差异源于训练和推理阶段使用的底层引擎（如精度格式、算子实现、并行策略等）不一致，导致模型在训练时行为与部署时行为出现偏差，进而引发 **训练崩溃（training collapse）** 或 **次优性能**。

传统方法尝试通过统一精度（如全用FP16）或梯度掩码来缓解，但这些方法要么牺牲数值稳定性，要么仅被动抑制噪声，未能从根本上解决“训练策略应向部署策略对齐”的核心目标。

---

### 🚀 提出的新方法与新思路
作者提出了一种全新的范式转变：将 LLM 的 RL 训练重新建模为 **Discrepancy-Constrained Markov Decision Process (DCMDP)**，其核心思想是：

1. **赋予训练策略自我感知与纠正能力**  
   引入一个基于 **token-level 绝对概率差（absolute probability difference）** 的黑盒差异信号 $ \delta_{\text{diff}} = |\pi_{\text{train}} - \pi_{\text{inf}}| $，作为可微分的反馈信号，使训练策略能主动识别并修正与推理策略的偏差。

2. **定义“差异容忍区”（discrepancy tolerance region）**  
   实验发现：轻微的 train-inference 差异并不会损害性能，反而有助于探索；只有当差异超过某一阈值时才需要干预。因此，惩罚只在超出容忍边界 $ c $ 时激活，避免过早约束探索空间。

3. **采用 Lagrangian Relaxation 动态平衡双目标优化**  
   将最大化奖励与最小化差异构造成带约束的优化问题，并引入可学习的 Lagrangian multiplier $ \lambda $，根据当前 batch 的平均差异动态调整惩罚强度，实现稳定、自适应的 dual-objective 优化。

4. **支持异构训练范式（heterogeneous training）**  
   可在高保真训练环境（如 BF16）中更新策略，同时显式对齐低资源推理环境（如 FP8），从而实现“高性能训练 + 低成本部署”的解耦。

---

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | 本文方法（DCMDP / DC-GRPO） |
|------|--------|----------------------------|
| 对差异处理方式 | 被动抑制（如降级精度、梯度掩码） | 主动感知与自校正 |
| 优化目标 | 单一奖励最大化 | 双目标：奖励最大化 + 差异控制 |
| 探索效率 | 易因强正则化而受限 | 容忍区内自由探索，提升学习效率 |
| 架构兼容性 | 需对齐软硬件细节 | 黑箱处理差异，适用于任意后端组合 |
| 部署灵活性 | 训练即部署 | 支持异构训练-部署分离 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练数据**：`DAPO-Math-17K` — 包含约17,000个数学推理任务的高质量指令数据集。
- **评估数据集**（6个主流数学基准）：
  - `AIME24`, `AIME25`
  - `AMC23`
  - `MATH-500`
  - `Minerva Math`
  - `OlympiadBench`

所有任务均需深度逻辑与多步推理解题，适合检验 LLM 的 reasoning 能力演进。

---

### ⚙️ 实验设置
| 项目 | 设置说明 |
|------|---------|
| **模型** | 
| - `Qwen3-8B`（dense 模型，FSDP2 训练）<br>- `Qwen3-30B-A3B`（MoE 模型，Megatron-LM + Tensor/Expert Parallelism） |
| **训练算法基础** | GRPO（Group Relative Policy Optimization），无 Critic 网络，使用 group-level reward normalization |
| **训练配置** |
| - Batch size: 64 prompts, G=8 rollouts per prompt<br>- Max response length: 8,192 tokens<br>- Optimizer: AdamW (`lr=1e-6`, betas=(0.9, 0.95))<br>- Clipping range: ε=0.24 |
| **差异计算方式** |
| - Token-level 绝对概率差：<br> $ \delta_{i,t} = |\pi_{\text{train}}(o_{i,t}|q, o_{<t}) - \pi_{\text{inf}}(o_{i,t}|q, o_{<t})| $<br>- 来源：同一序列在训练后端（recomputation）与推理后端（rollout, vLLM）上的 logprob 差异 |
| **Lagrangian 参数** |
| - 初始 λ：Qwen-8B 为 0.1，Qwen-30B 为 0.05<br>- Dual learning rate: 1.0<br>- 投影范围：$ \lambda \in [\lambda_{\min}, \lambda_{\max}] $ |
| **容忍边界 c** |
| - Qwen-8B: 0.0037<br>- Qwen-30B: 0.0050（更高因 MoE 路由非确定性更强） |

---

### 📊 评估指标
- **主指标**：`avg@K`（K次采样中的最高正确率）
  - AIME/AMC 类：`avg@32`
  - MATH/Minerva/Olympiad：`avg@4`
- **评估环境**：
  - 使用 `vLLM 0.11.0` 进行推理
  - 温度=0.7，top_p=1.0，最大生成长度=32,768 tokens
- **对比基线**：
  - `GRPO`：标准 RL 基线，无任何差异控制
  - `DC-GRPO`：本文提出的方法（DCMDP + Lagrangian 控制）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & 2）

#### ✅ BF16 统一训练/推理设置下的结果（Table 1）
| Model | Method | AIME24 | AIME25 | AMC23 | MATH-500 | Minerva | Olympiad | **Avg** |
|-------|--------|--------|--------|--------|----------|---------|----------|--------|
| Qwen3-8B | GRPO | 58.8 | 40.0 | 90.4 | 93.7 | 51.0 | 70.1 | **67.3** |
| | **DC-GRPO** | **63.2** | **52.5** | **91.8** | **94.2** | 50.1 | **74.4** | **71.0** |
| Qwen3-30B-A3B | GRPO | 60.2 | 43.6 | 91.4 | 95.5 | 52.0 | 71.3 | **69.0** |
| | **DC-GRPO** | **66.8** | **49.1** | **91.9** | **95.8** | **52.9** | **74.5** | **71.8** |

> 💡 结论：在相同精度下，DC-GRPO 显著优于 GRPO，在多个 benchmark 上提升明显，尤其在 AIME 和 Olympiad 等复杂推理任务上表现突出。

---

#### ✅ 异构训练设置：BF16训练 + FP8推理（Table 2）
| Model | Method | AIME24 | AIME25 | AMC23 | MATH-500 | Minerva | Olympiad | **Avg** |
|-------|--------|--------|--------|--------|----------|---------|----------|--------|
| Qwen3-8B | GRPO | 53.2 | 35.2 | 89.5 | 92.9 | 50.0 | 67.4 | **64.7** |
| | **DC-GRPO** | **55.0** | **44.1** | **90.2** | **94.4** | **54.0** | **70.9** | **68.1** |
| Qwen3-30B-A3B | GRPO | 52.4 | 37.5 | 87.3 | 93.8 | 49.6 | 70.5 | **65.2** |
| | **DC-GRPO** | **56.6** | **41.2** | **91.8** | **94.0** | **50.5** | **72.0** | **67.7** |

> 💡 结论：在异构环境下，GRPO 性能严重下降甚至崩溃，而 DC-GRPO 成功维持甚至超越了纯 BF16 设置下的性能，验证了其对部署差异的强大鲁棒性和对齐能力。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）初始 λ 值影响（Table 3）
- $ \lambda_0 = 0.1 $ 表现最佳，兼顾早期稳定性和后期性能。
- $ \lambda_0 = 0.05 $：初期惩罚弱，难以遏制 spike。
- $ \lambda_0 = 0.2 $：过度正则化，拖慢早期 reward 学习。

#### （2）Dual Learning Rate $ \eta_\lambda $ 影响（Table 4）
- $ \eta_\lambda = 1.0 $ 收敛最平稳。
- $ < 0.5 $：响应太慢，无法及时纠正违规。
- $ > 1.5 $：震荡加剧但仍可控（得益于投影机制）。

#### （3）容忍边界 $ c $ 的选择（Table 5）
- $ c = 0.0037 $（Qwen-8B）取得最优性能。
- $ c $ 太小 → 惩罚过强 → 抑制探索；
- $ c $ 太大 → 惩罚失效 → 退化为普通 GRPO。

> ✅ 发现：存在明确的 **discrepancy tolerance region**，合理设定边界至关重要。

---

## 4. 关键结论和发现

### 🎯 主要发现
1. **Train-inference discrepancy 是训练不稳定的根本原因之一**，且广泛存在于不同架构与基础设施中。
2. **训练策略可以被赋予自主感知与纠正差异的能力**，通过引入 token-level 概率差作为黑盒反馈信号即可实现。
3. **轻微差异可容忍，过度惩罚有害**：存在一个“差异容忍区”，在此区域内应允许自由探索，仅在越界时施加约束。
4. **DCMDP 范式显著提升训练稳定性与最终性能**，不仅防止崩溃，还能突破原有性能瓶颈。
5. **支持异构训练范式**：可在高成本训练环境中模拟并优化低资源部署策略，极大增强实际部署灵活性。

---

### ⚠️ 局限性
1. **实验范围有限**：
   - 当前仅验证于 BF16 ↔ FP8 场景，未覆盖更多 backend 组合（如 INT8、NF4 等）。
2. **模型家族单一**：
   - 仅测试 Qwen 系列模型，缺乏在其他开源 LLM family（如 Llama、Phi、Gemma）上的泛化分析。
3. **缺少更细粒度的差异归因分析**：
   - 虽然以黑箱方式处理差异有效，但未深入剖析具体是 kernel 实现、路由非确定性还是通信延迟主导了 mismatch。

---

### 🔮 未来工作方向
1. **扩展至更多量化方案与硬件平台**，构建通用的 discrepancy-aware RL 框架。
2. **探索自动学习容忍边界 $ c $** 或动态调整机制，减少人工调参依赖。
3. **结合模型压缩技术**（如 pruning, distillation），进一步推动高效部署闭环。
4. **应用于多模态与 Agent 场景**，研究 action-level 差异补偿机制。

---

## ✅ 总结
本论文提出了 **DCMDP** 框架，将 LLM 的 RL 训练从传统的 MDP 扩展为带有差异约束的新型决策过程。通过引入 **token-level 黑盒差异信号** 与 **Lagrangian 自适应控制机制**，实现了训练策略对推理策略的主动对齐，在提升性能的同时彻底缓解训练崩溃问题。更重要的是，它开启了 **异构训练** 的新范式，为未来大规模、低成本、高可靠性的 LLM 部署提供了坚实的算法基础。

</details>

---

### 8. [Q-Delta: Beyond Key-Value Associative State Evolution](https://arxiv.org/abs/2606.08804)

**Authors**: Sumin Park, Seojin Kim, Noseong Park  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.08804v1  

#### Abstract
Linear attention reformulates sequence modeling as recurrent state evolution, enabling efficient linear-time inference. Under the key-value associative paradigm, existing approaches restrict the role of the query to the readout operation, decoupling it from state evolution. We show that query-condit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Q-Delta: Beyond Key-Value Associative State Evolution**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统 **linear attention** 和 **state space models (SSMs)** 将状态演化（state evolution）建模为仅由 **key-value** 关联驱动的过程，而 **query** 仅用于最终读出（readout），不参与状态更新。这种设计忽略了 query 在状态演化中可能提供的互补信息，导致记忆更新机制不够精细，尤其在长序列和稀疏信息检索任务中表现受限。

### **提出了什么新方法或新思路**
本文提出 **Q-Delta**，一种 **query-aware delta rule**，将 query 引入状态演化过程，实现更丰富的记忆动态。其核心思想是：
- 将 query-conditioned 预测 $ \hat{o}_t = S_{t-1}q_t $ 视为一种与 key-retrieved 值 $ \hat{v}_t = S_{t-1}k_t $ 互补的价值预测。
- 构造一个 **混合误差信号**（mixed prediction error）$ v_t - \hat{v}_t - \lambda_t \hat{o}_t $，并将其用于状态更新。

Q-Delta 的递推公式为：
$$
S_t = \alpha S_{t-1}(I - \beta(k_tk_t^\top + \lambda_t q_tk_t^\top)) + \beta v_tk_t^\top
$$
其中 $ \lambda_t \in [0,1] $ 是可学习的 query 反馈系数，控制 query 对状态更新的影响。

### **相比现有方法的优势**
- **理论优势**：首次系统论证 query readout 提供了与 key retrieval 正交的、结构化的价值预测信号，支持其作为状态演化的主动参与者。
- **稳定性保证**：证明了 Q-Delta 动态在经验条件下具有 **one-step error contraction** 和 **global geometric tracking** 性质。
- **效率保持**：通过 **chunkwise-parallel formulation** 和定制的 **Triton kernel** 实现硬件高效训练，维持线性时间复杂度。
- **表达能力增强**：联合修正 key 和 query 的预测误差，提升模型对长上下文和稀疏信息的记忆与检索能力。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
#### **语言建模与常识推理**
- **FineWeb-Edu**：用于预训练 340M 和 1.3B 参数模型（分别训练 15B 和 30B tokens）。
- **零样本评估基准**：
  - **Perplexity**：WikiText、LAMBADA
  - **多选题准确率**：PIQA、HellaSwag、WinoGrande、ARC-Easy、ARC-Challenge、OpenBookQA、BoolQ

#### **检索能力评估**
- **合成检索任务**：**S-NIAH**（Needle-In-A-Haystack）系列，包含：
  - S-NIAH-1：Pass-key retrieval
  - S-NIAH-2：Number in haystack
  - S-NIAH-3：UUID in haystack
  - 测试长度：1K, 2K, 4K tokens
- **真实世界检索任务**（recall-intensive）：
  - SWDE、SQuAD、FDA、TQA、NQ、DROP
  - 上下文截断至 2K tokens，采用 cloze-style 提示格式

### **实验设置和评估指标**
- **框架**：基于 `flash-linear-attention` 实现，使用 bfloat16 混合精度。
- **优化器**：AdamW，余弦学习率调度，梯度裁剪。
- **硬件**：4×NVIDIA RTX Pro 6000 (Blackwell) GPUs。
- **评估指标**：
  - 语言建模：Perplexity（↓）
  - 推理与检索：Accuracy（↑）
  - 效率：单卡吞吐量（tokens/sec）

### **基线方法对比**
- **RetNet** (Sun et al., 2023)
- **Mamba** (Gu & Dao, 2024)
- **Mamba2** (Dao & Gu, 2024)
- **DeltaNet** (Yang et al., 2025b)
- **GatedDeltaNet** (Yang et al., 2025a)

所有基线均在相同框架和超参下复现，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **语言建模与零样本推理（Table 2）**
| 模型 | 340M 平均准确率 | 1.3B 平均准确率 |
|------|------------------|------------------|
| **Q-Delta** | **47.24** | **53.47** |
| GatedDeltaNet | 46.01 | 52.77 |
| Mamba2 | 46.27 | 52.46 |

- Q-Delta 在两个规模上均取得 **最高平均准确率**。
- 在 1.3B 模型上，Q-Delta 在 **ARC-Challenge**、**HellaSwag**、**BoolQ** 等多个任务上领先。

#### **长上下文检索（Table 3）**
| 模型 | S-NIAH Avg. Accuracy |
|------|------------------------|
| **Q-Delta** | **90.02** |
| GatedDeltaNet | 83.51 |
| DeltaNet | 81.29 |

- Q-Delta 在 **所有上下文长度**（1K–4K）上均达到 **100% Pass-key 检索成功率**。
- 在更具挑战性的 **S-NIAH-2** 和 **S-NIAH-3** 任务上显著优于基线，尤其在 4K 上表现鲁棒。

#### **真实世界检索（Table 4）**
| 模型 | 平均准确率 |
|------|------------|
| **Q-Delta** | **33.2** |
| GatedDeltaNet | 34.2 |
| Mamba2 | 33.4 |

- Q-Delta 在真实世界任务上表现 **与最优基线相当或略优**，验证了其实际检索能力。

#### **吞吐量（Figure 4b）**
- Q-Delta 吞吐量 **与 DeltaNet/GatedDeltaNet 相当**，显著高于 Mamba2。
- 在不同序列长度（2K–16K）和批量大小组合下均保持高效。

### **消融实验结果（Table 5）**
| 设置 | Wiki PPL ↓ | Lamb PPL ↓ | 平均准确率 ↑ |
|------|------------|------------|--------------|
| Learnable $ \lambda_t $ (**Q-Delta**) | 26.89 | 32.67 | **47.24** |
| Scalar $ \lambda = 0.5 $ | 26.86 | 33.31 | 47.20 |
| No decay ($ \alpha=1 $) | 26.52 | 32.97 | 45.86 |
| No gating ($ \lambda=1 $) | 26.55 | 35.21 | 46.36 |

- **可学习 $ \lambda_t $** 表现最佳，说明自适应调节 query 反馈强度有益。
- 即使移除衰减门控，Q-Delta 仍显著优于 DeltaNet（45.86 vs 43.82），表明 **query 反馈本身具有独立增益**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Query 不应只是被动读出**：query-conditioned 预测 $ \hat{o}_t = S_{t-1}q_t $ 提供了与 key-retrieved 值正交的、结构化的价值信息，是状态演化的重要补充信号。
2. **混合误差修正更有效**：Q-Delta 通过联合修正 key 和 query 的预测误差，实现了更精准的状态更新，提升了模型的记忆控制能力和长上下文理解。
3. **稳定且高效**：Q-Delta 在经验上满足 error contraction 条件，全局动态稳定；其 chunkwise-parallel 实现保持了线性时间效率。
4. **性能全面领先**：在语言建模、常识推理和长上下文检索任务上，Q-Delta 一致优于主流 linear attention 和 SSM 基线。

### **方法的局限性**
- **依赖线性假设**：分析基于线性状态转移，非线性激活可能影响理论性质。
- **query 反馈机制仍较简单**：当前形式为线性加权，未来可探索更复杂的交互方式。
- **未涉及指令微调**：实验均为预训练阶段评估，未测试在 instruction-tuning 或 downstream fine-tuning 中的表现。

### **未来工作方向**
- 探索 **非线性 Q-Delta** 扩展，如引入 attention-like 非线性映射。
- 将 Q-Delta 应用于 **decoder-decoder** 或 **encoder-decoder** 架构。
- 研究 **multi-head query feedback** 的协同机制。
- 在 **视觉、音频等序列建模任务** 中验证 Q-Delta 的通用性。

> **代码开源**：https://github.com/psmiz/Q-Delta

</details>

---

### 9. [Large-Scale Regularized Matching on GPU Clusters](https://arxiv.org/abs/2606.07777)

**Authors**: Aida Rahmattalabi, Gregory Dexter, Sanjana Garg, Qinquan Song, Shenyinying Tu, Yuan Gao, Zhipeng Wang, Rahul Mazumder  
**Category**: cs.DC  
**Published**: 2026-06-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.07777v1  

#### Abstract
Production decision systems such as ad allocation or content matching involve millions of users and thousands of items, reducing to large-scale linear programs with sparse block-diagonal structure across users. These LPs are solved repeatedly on recurring cadences over slowly evolving inputs. Three ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Large-Scale Regularized Matching on GPU Clusters**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

该论文针对**大规模生产级决策系统**（如广告分配、内容匹配）中的**超大规模线性规划（LP）求解问题**，指出当前方法存在三大系统瓶颈：

- **Scale（规模限制）**：现有 GPU 求解器（如 cuPDLP、D-PDLP）受限于单设备内存，无法处理数十亿变量的工业级实例。
- **Temporal Instability（时间不稳定性）**：连续运行间解波动大，导致下游“churn”（频繁变动），影响 SLA，而现有求解器缺乏显式控制机制。
- **Extensibility（可扩展性差）**：基于 CPU 的框架（如 DuaLip-Scala）收敛慢，且将问题建模绑定到固定 schema，新增约束成本高。

---

### ✅ **提出了什么新方法或新思路**

作者提出一个**原生构建在 PyTorch 上的分布式多 GPU LP 求解器**，采用“**系统-算法协同设计**”（systems-algorithm co-design）策略，围绕生产匹配 LP 的**对角块结构**进行优化，主要贡献如下：

#### **1. 列分片执行模型 + 融合 Triton 内核（Column-sharded Execution & Fused Kernels）**
- 将约束矩阵按“源”（用户）维度列分片（column-sharded），每个 GPU 处理独立的用户块。
- 通信仅需在每轮迭代中对**项目级对偶变量**（item-level duals）做一次 Reduce 和 Broadcast，通信量与用户数无关，仅取决于约束数量（O(J)）。
- 使用 **Triton 编写的融合内核**实现 simplex 投影，避免中间张量写回全局内存，显著降低开销。

#### **2. 可调岭正则化（Tunable Ridge Regularization）提升稳定性**
- 引入并暴露正则化参数 γ，用于控制连续运行间的解漂移（solution drift），提供**显式的稳定性控制接口**。
- 设计 **γ 的退火调度（continuation schedule）**：从大 γ 开始快速稳定收敛，逐步减小以逼近原始 LP 最优解。

#### **3. 算子中心编程模型（Operator-Centric Programming Model）**
- 提出三个可组合的算子原语：
  - `ObjectiveFunction`：定义目标与梯度计算
  - `ProjectionMap`：定义投影操作
  - `Maximizer`：执行对偶上升
- 新增约束只需局部修改 `ObjectiveFunction`，无需改动求解循环或分布式基础设施，极大提升**可扩展性**。

---

### ✅ **相比现有方法的优势**

| 维度 | 本文方法 | 现有方法（如 D-PDLP、DuaLip-Scala） |
|------|--------|-----------------------------|
| **硬件扩展性** | 支持跨节点多 GPU，近线性扩展 | 单节点或单设备，扩展受限 |
| **内存效率** | 分布式存储，支持超大规模实例 | 易 OOM，受限于单卡内存 |
| **稳定性控制** | 显式 γ 参数控制解漂移 | 无内置机制 |
| **可扩展性** | 算子化设计，新增约束简单 | 需修改 schema 与调度逻辑 |
| **性能** | 数量级加速，支持 >10^9 非零元 | 在大规模下失败或极慢 |

---

## 2. **核心实验方法和设置**

### ✅ **使用的数据集**

- **合成匹配数据集**（synthetic matching workloads），模拟真实生产场景：
  - 用户数（Sources）：25M ~ 100M
  - 项目数（Destinations）：10K
  - 稀疏度：约 0.1%
  - 非零元（NNZ）：最高达 **20 亿**（2.0×10⁹）
- 数据生成方式见附录 A，确保与 DuaLip-Scala 输入兼容。

---

### ✅ **实验设置和评估指标**

#### **硬件环境**
- 使用 **NVIDIA H100 80GB GPU**，单节点最多 8 卡，跨节点最多 16 卡。
- 多 GPU 通过 `torchrun` + `NCCL` 实现通信。

#### **评估指标**
- **每轮迭代时间（per-iteration time）**
- **端到端求解时间（end-to-end solve time）**
- **加速比（speedup）**
- **峰值 GPU 内存占用**
- **解质量**：对偶目标值、原始-对偶间隙（primal-dual gap）、约束违反（slack）

#### **算法配置**
- 正则化参数 γ 采用 **6 阶几何退火**：{10³, 10², ..., 10⁻²}，每阶段 10,000 次迭代，共 60,000 次。
- 使用 **AGD（Accelerated Gradient Descent）** + **Jacobi 预条件化**。

---

### ✅ **基线方法对比**

| 基线方法 | 类型 | 特点 |
|--------|-----|------|
| **DuaLip-Scala** | CPU-based, Spark | 工业界基线，但慢且不可扩展 |
| **D-PDLP** | Multi-GPU, 2D 分区 | 当前最先进的 GPU LP 求解器，使用 2D 矩阵分区 + AllReduce |

---

## 3. **主要实验结果和性能指标**

### ✅ **关键性能数据**

#### **1. 与 DuaLip-Scala 对比（表 2）**
| 用户数 | DuaLip-Scala (s/iter) | 本文（1 GPU） | 加速比 |
|-------|----------------------|--------------|--------|
| 25M   | 2.46                 | 0.27         | **~9.1×** |
| 50M   | 3.44                 | 0.27         | **~12.7×** |
| 100M  | 3.33                 | 0.27         | **~12.3×** |

> ⚡️ **单 GPU 即实现近一个数量级加速**

#### **2. 多 GPU 扩展性（图 3）**
- 在 16 GPU（跨 2 节点）上：
  - 75M 用户：**13.1× 加速**（82% 效率）
  - 50M 用户：**12.0× 加速**（75%）
  - 25M 用户：**6.0× 加速**（37%，因通信占比上升）
- 扩展平滑，跨节点无明显性能下降。

#### **3. 支持更大规模问题**
- **100M 用户问题**在单卡 OOM（>80GiB），但在 2+ GPU 上可运行。
- 16 GPU 下，100M 用户问题在 **3 分钟内完成求解**。

---

### ✅ **与 D-PDLP 对比（表 3 & 表 4）**

| 实例 | D-PDLP 结果 | 本文结果 |
|------|-------------|----------|
| s100M-base (1.0×10⁹ NNZ) | **OOM** | ✅ 成功求解（2441s @ 8 GPU） |
| s50M-l1-reformulated (~1.0×10⁹ NNZ) | **OOM** | ✅ 可解（需更多资源） |
| s50M-base | 4.5s（8 GPU） | 1269s（8 GPU） |

> 🔍 **关键发现**：虽然 D-PDLP 在小规模上更快，但在**大规模或 l₁ 正则化变体下全部 OOM**，而本文方法**唯一能求解最大规模实例**。

#### **解质量对比（表 4）**
- 在共同可解实例上，**对偶目标值一致至 4 位有效数字**（Δ < 10⁻⁶）。
- 本文方法达到更小的原始-对偶间隙（低至 **10⁻¹²** vs D-PDLP 的 10⁻³~10⁻⁵），表明**正则化提升了数值稳定性**。

---

### ✅ **消融实验结果**

#### **1. 融合 Triton 内核 vs PyTorch 原生实现（图 1）**
- **速度提升**：10M–50M 用户下 **2.5–5×**，1M 用户下 **>20×**（因 launch 开销主导）。
- **内存节省**：峰值内存降低 **~20%**，因消除中间张量（排序、前缀和等）。

#### **2. 桶化批处理 vs 单一大张量（图 2）**
- **速度提升**：~1.2×
- **内存节省**：~24%
- 优势来自减少零填充计算与内存浪费。

#### **3. 预条件化与 γ 退火（图 4 & 图 5）**
- **Jacobi 预条件化**：显著加快早期收敛，尤其在稀疏/异构约束下。
- **γ 退火**：相比固定 γ，**收敛更快且最终解更接近未正则化最优**。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **结构化稀疏性是可扩展性的关键**：利用匹配 LP 的“对角块结构”实现列分片，使通信与用户数解耦，是实现近线性扩展的基础。
2. **正则化不仅是算法技巧，更是系统需求**：γ 不仅提升稳定性，还作为**可调参数**平衡速度与精度，是生产系统的刚需。
3. **PyTorch + Triton 可构建高性能 LP 求解器**：无需定制 C++/CUDA，借助现代 ML 框架即可实现高效稀疏代数与分布式执行。
4. **算子化抽象提升开发效率**：将问题建模与求解引擎分离，使新约束可插拔，适合快速迭代的工业场景。

---

### ⚠️ **方法的局限性**

- **依赖特定结构**：目前仅适用于具有“源-项目”对角块结构的匹配类 LP，通用性有限。
- **正则化引入偏差**：尽管可通过退火减小，但仍非精确求解原始 LP。
- **缺乏公开基准**：实验基于合成数据，缺少真实工业数据集的公开验证。
- **预条件化假设行满秩**：若约束矩阵奇异，需额外处理。

---

### 🔮 **未来工作方向**

1. **拓展到更广泛的 LP 形式**：如网络流、多商品匹配等。
2. **构建标准化超大规模匹配 LP 基准套件**，促进可复现研究。
3. **探索混合精度与量化**，进一步提升 GPU 利用率。
4. **集成到端到端推荐/匹配系统**，实现实时动态优化。

---

> **总结**：本文通过**结构感知的分布式设计 + 正则化稳定性控制 + 算子化编程模型**，成功将大规模 LP 求解推向新的可扩展性和实用性边界，为工业级匹配系统提供了兼具**高性能、高稳定性和高可扩展性**的下一代求解方案。

</details>

---

### 10. [EditSR: Enhancing Neural Symbolic Regression via Edit-based Rectification](https://arxiv.org/abs/2606.07915)

**Authors**: Da Li, Xinxin Li, Xingyu Cui, Jin Xu, Juan Zhang, Junping Yin  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.07915v1  

#### Abstract
Neural symbolic regression models improve inference efficiency by shifting structural search to pretraining, but their one-pass autoregressive decoding is prone to error accumulation, which may lead to generating structurally incorrect expressions, especially in complex expression generation scenari...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EditSR: Enhancing Neural Symbolic Regression via Edit-based Rectification

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
神经符号回归（Neural Symbolic Regression）模型通过大规模预训练显著提升了推理效率，其典型范式是**单次自回归解码**（one-pass autoregressive decoding）。然而，这种范式存在严重的**错误累积**（error accumulation）问题：一旦早期生成错误的算子或子树，后续的语法上下文将被破坏，导致整个表达式结构偏离目标。

现有的后处理修正策略（post-hoc rectification）虽然能缓解该问题，但通常依赖于重启全局搜索（如MCTS）或在隐空间优化，计算成本高昂，削弱了神经模型原本的效率优势。

---

### ✅ 提出的新方法：EditSR
本文提出 **EditSR**，一个两层框架，结合：
- **第一层**：预训练的神经符号回归模型（如 NeSymReS），用于快速生成初始表达式；
- **第二层**：基于编辑的 **Rectifier**（修正器），通过多步、局部的语法约束编辑操作，逐步修正初始错误表达式。

#### 核心创新点：
1. **将修正过程建模为状态转移链**  
   将从错误表达式到目标表达式的修正过程视为一系列**状态转移**（state-transition chain），每一步执行一个编辑动作（edit action），确保中间表达式始终语法合法（parseable）。

2. **定义五种子树级编辑动作**  
   编辑动作作用于**子树级别**而非词元级别，保证语法完整性：
   - `KEEP`：保留当前节点
   - `REPLACE`：替换节点（保持元数）
   - `DELETE`：将子树折叠为叶子
   - `REWRITE`：重写整个子树
   - `INSERT`：将叶子扩展为新子树

3. **预训练修正器以提升效率**  
   不依赖在线搜索，而是通过**预训练**方式学习如何修正。使用一种**状态转移算法**构建监督式的修正链，使 Rectifier 在推理时仅需少量步骤即可完成修正。

4. **决策基于当前状态而非历史**  
   每一步的编辑决策只依赖当前表达式状态，允许后续步骤纠正前期错误，从而**抑制错误累积**。

---

### ✅ 相比现有方法的优势
| 维度 | EditSR | 传统方法（如TPSR、SNIP） |
|------|--------|--------------------------|
| **效率** | 高效，修正开销小（平均仅需4–6步） | 低效，需重启搜索或多次数值优化 |
| **结构恢复能力** | 显著提升复杂表达式的结构准确率 | 多关注数值拟合，结构一致性弱 |
| **鲁棒性** | 对噪声、无关变量更鲁棒 | 容易过拟合噪声或引入无关变量 |
| **通用性** | 可适配不同第一层模型（经微调） | 通常与特定架构强耦合 |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集
实验覆盖三大类基准：
1. **标准基准**（Standard Benchmarks）  
   包括 Constant, Koza, Nguyen, Keijzer, Korns, Livermore, Neat, Jin 等共91个问题，最多含3个变量。

2. **SRBench 1.0**
   - **Feynman**：120个物理公式，源自真实科学定律。
   - **ODE-Strogatz**：14个非线性动力系统方程。

3. **SRBench 2.0**
   - **Phenomenological & first-principles**：12个现实科学发现任务，含真实噪声。
   - **Black-box**：12个黑箱任务，去除可线性求解的问题，更具挑战性。

> 所有官方发布的数据集均采用原始划分（train/test split）。

---

### ✅ 实验设置
- **第一层模型**：NeSymReS（支持10变量输入）
- **Beam Size**：30
- **Rectifier 微调轮数**：5 epochs
- **最大修正步数** $ T_{\text{max}} $：10
- **常数优化**：使用 BFGS 进行后处理
- **Bagging策略**：对大数据集采样子集进行多次预测

---

### ✅ 评估指标
| 指标 | 描述 |
|------|------|
| **R²** | 拟合优度，$ R^2 > 0.999 $ 视为准确解 |
| **Accuracy solution rate** | 达到 $ R^2 > 0.999 $ 的问题比例 |
| **Symbolic solution rate** | 表达式结构与目标一致的比例（允许常数缩放或平移） |
| **Complexity** | 表达式树的总节点数，越低越好 |
| **Test Time** | 平均测试耗时（秒） |
| **Noise Robustness** | 在添加高斯噪声（$\sigma \in \{0, 0.001, 0.01, 0.1\}$）下的表现 |
| **Distractor Robustness** | 添加1–3个无关变量后的性能变化 |

---

### ✅ 基线方法对比
| 基线 | 类型 |
|------|------|
| **uDSR** | 统一框架，集成递归简化、遗传编程等 |
| **SR4MDL** | 基于最小描述长度（MDL）引导搜索 |
| **ParFam** | 参数族连续优化方法 |
| **RILS-ROLS** | 元启发式结构搜索 + 最小二乘系数估计 |
| **TPSR** | 基于MCTS的后处理修正（代表传统rectification方法） |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据汇总

#### 📊 在标准基准上的平均表现（Table 2 & Figure 4）
| 模型 | 平均 R² | Symbolic Solution Rate | Test Time (s) |
|------|--------|------------------------|---------------|
| **EditSR** | **0.98+**（多数 >0.99） | **最高且最稳定** | **17.03**（最低） |
| TPSR | 高 | 极低（如Korns上仅10%） | 174.24 |
| uDSR | 中等偏高 | 中等 | 77.99 |
| SR4MDL | 中等 | 下降明显（尤其在噪声下） | 765.03 |

> ✅ EditSR 在 **Accuracy** 和 **Symbolic Solution Rate** 上高度一致，说明其真正修复了结构错误，而非仅数值拟合。

---

#### 📈 在 SRBench 1.0 上的表现（Figure 6）
- **Feynman 基准**：
  - EditSR 在所有噪声水平下保持约 **80% Accuracy** 和 **>50% Symbolic Solution Rate**
  - 明显优于 uDSR、ParFam、TPSR 等，尤其在噪声增大时下降更缓
- **ODE-Strogatz 基准**：
  - 数据稀疏且采样范围窄，整体难度更高
  - EditSR 仍保持最强鲁棒性，在 $\sigma=0.1$ 下仍领先

---

#### 🔍 在 SRBench 2.0 上的表现（Figure 7）
- **Black-box**：
  - EditSR 在 R² 与 Complexity 之间取得最佳权衡，避免“低R²+高复杂度”的失败模式
- **Phenomenological & first-principles**：
  - 多数问题 R² > 0.95，复杂度控制良好
  - 显示其在真实科学场景中的实用性

---

### ✅ 消融实验结果（Ablation Studies）

#### 🔹 Rectifier 的有效性（Figure 8）
- **NeSymReS → EditSR** 带来显著提升：
  - **Accuracy 提升**：+10% ~ +20%
  - **Symbolic Solution Rate 提升**：翻倍以上
- 即使不微调（EditSR’），也优于原模型，证明预训练有效
- 微调后进一步提升，表明需适配实际错误分布

#### 🔹 复杂度影响（Figure 9）
- 随目标表达式复杂度上升，NeSymReS 性能急剧下降（高复杂度组 Symbolic Rate 接近 0%）
- **EditSR 在中高复杂度组优势最明显**，验证其对长表达式生成特别有效

#### 🔹 修正步数分析（Figure 10–12）
- 成功案例平均仅需 **4–6 步编辑**
- `INSERT` 最常用（占比最高），其次是 `REWRITE`
- `INSERT` 对降低编辑距离效果最显著 → 表明缺失结构是主要错误类型

#### 🔹 对第一层鲁棒性（Table 5）
- 在不同 dropout 率（0.1–0.3）训练的第一层上微调 Rectifier，性能波动小
- 说明 Rectifier 能适应多种错误模式，具备良好泛化能力

#### 🔹 与 TPSR 对比（Figure 15）
- **EditSR 在更低时间成本下实现更高 R²**
- TPSR 虽有一定改进，但耗时远高于 EditSR（>10倍）
- 结论：**局部编辑比重启搜索更高效**

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **神经符号回归的瓶颈在于错误累积**，尤其在复杂表达式生成中。
2. **EditSR 通过预训练的编辑式修正机制**，在不牺牲效率的前提下显著提升结构恢复能力。
3. **修正过程本质是“局部修补”而非“重新发明”**：大多数错误集中在局部区域，已有结构可复用。
4. **EditSR 特别适用于复杂、长表达式场景**，其优势随问题复杂度增加而增强。
5. **相比重启搜索的方法（如TPSR），EditSR 更快、更稳定、结构一致性更强**。

---

### ⚠️ 局限性
1. **依赖第一层提供合理初始结构**  
   若初始预测与目标相差太远（如完全不同的函数形式），有限步数内难以完全修复。
2. **修正能力受限于预训练覆盖的错误模式**  
   对严重分布外（OOD）错误的泛化能力有待验证。
3. **未增强全局探索能力**  
   仍依赖第一层模型的多样性输出，缺乏主动探索机制。

---

### 🔮 未来工作方向
1. **与强搜索模型协同**  
   将 Rectifier 作为通用模块，用于精炼遗传编程或 MCTS 产生的候选表达式。
2. **构建闭环反馈系统**  
   将 Rectifier 的修正轨迹反馈给第一层，实现联合优化。
3. **扩展至多模态输入**  
   结合视觉、文本等先验信息指导编辑过程（如 ViSymRe 或 LLM-SR 的思想）。
4. **动态调整编辑预算**  
   根据表达式差异程度自适应决定最大编辑步数。

---

> 💡 **一句话总结**：  
> **EditSR 提出了一种高效、轻量、可学习的后处理修正机制，通过“编辑而非重搜”的理念，在保持神经符号回归速度优势的同时，显著增强了其对复杂表达式的结构恢复能力，为科学发现中的符号回归任务提供了新的实用路径。**

</details>

---

### 11. [AliyunConsoleAgent: Training Web Agents in Real-World Cloud Environments via Distillation and Reinforcement Learning](https://arxiv.org/abs/2606.09447)

**Authors**: Bojie Rong, Zheyu Shen, Qiaoping Wang, Pengfei Kang, Yang Xu, Yawen Wei, Hanyu Wu, Zhi Zhao, Leihao Pei, Linquan Jiang  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.09447v1  

#### Abstract
We present AliyunConsoleAgent, a web agent framework for automated documentation verification in real-world cloud consoles. Major cloud platforms encompass hundreds of products with rapid feature iteration, causing console UIs to frequently diverge from their corresponding documentation. Verifying t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AliyunConsoleAgent: Training Web Agents in Real-World Cloud Environments via Distillation and Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型云平台面临**文档漂移（documentation drift）**问题：由于产品功能快速迭代，控制台 UI 频繁更新，而配套文档依赖人工维护，导致文档中的操作流程与实际界面严重脱节。据估计，每年需进行约 **400万次重复检查** 才能维持文档一致性，但目前人工覆盖率不足 1%。

此外，使用前沿闭源模型（如 Gemini、GPT）构建 Web Agent 虽然成功率高，但存在两大瓶颈：
- **高昂的推理成本**
- **数据隐私风险**（敏感元数据外传）

因此，亟需一种低成本、可私有化部署、且能在真实复杂云环境中稳定运行的自动化验证方案。

---

### 提出的新方法与创新思路

作者提出 **AliyunConsoleAgent** 框架，结合 **知识蒸馏（Knowledge Distillation）** 和 **强化学习（Reinforcement Learning）**，训练一个可在真实云控制台中自主执行任务的轻量级 Web Agent。

#### 主要创新点：

1. **高确定性 Rollout 环境（High-Determinism Rollout Environment）**
   - 构建了一个四层架构的隔离执行环境，通过 **Terraform-based 资源预配置** 和 **ResourceCoder 动态按需供给**，确保每个任务在具备完整前置资源的状态下运行。
   - 引入 **ActionTrail 审计日志验证机制**，提供客观、防奖励欺骗（reward-hacking-resistant）的结果判断依据。

2. **两阶段训练范式：SFT + GRPO**
   - **第一阶段：监督微调（SFT）**
     - 利用强大教师模型（frontier models）生成高质量轨迹，经过过滤后用于训练学生模型。
   - **第二阶段：基于 GRPO 的在线强化学习**
     - 使用 **Group Relative Policy Optimization (GRPO)** 在真实环境中进行自我探索，提升泛化能力。
     - 设计 **双通道 Outcome Reward Model (ORM)**：
       - **Channel 1（规则驱动）**：查询 ActionTrail 日志，给出准确的二值奖励。
       - **Channel 2（LLM-as-Judge）**：对无明确 API 事件的任务，采用两个强 LLM 组成投票机制，仅当两者一致时才接受奖励，否则标记为无效（`r = -1`），有效缓解幻觉与奖励操纵。

3. **支持大规模私有化部署的成本优势**
   - 最终模型 **AliyunConsoleAgent-32B** 可私有部署于本地 GPU 集群（4× L20），单任务推理成本仅为 **0.56 CNY**，相比 Gemini 3 Pro Preview（7.0 CNY）降低 **92%**。

---

### 相比现有方法的优势

| 维度 | 传统方法 | AliyunConsoleAgent |
|------|--------|------------------|
| 成本 | 高（闭源 API 调用） | 低（本地部署，92% 成本下降） |
| 数据安全 | 外泄风险高 | 支持私有部署，合规性强 |
| 环境稳定性 | 易受资源缺失干扰 | 高确定性 Rollout 系统隔离噪声 |
| 决策能力 | 机械模仿 | 自主决策（如 precondition construction） |
| 可扩展性 | 小规模测试 | 支持生产级 300 tasks/hour 并发 |

---

## 2. 核心实验方法和设置

### 数据集
- **训练数据来源**：
  1. **蒸馏轨迹（Distilled Trajectories）**：由 frontier models（如 Gemini、Kimi）在真实控制台中成功执行的路径，经任务级与步骤级双重过滤。
  2. **自探索轨迹（Self-exploration）**：模型从产品入口（如 ECS、RDS）出发，自主发起 CRUD 操作，覆盖长尾 UI 状态。
- 总共使用约 **160K 单步样本** 进行 SFT 全参数微调。

- **评估基准（Benchmark）**：
  - 包含 **278 个真实云产品文档验证任务**，涵盖 12 个阿里云产品。
  - 分为两类：
    - **Standard Tasks（76 个）**：代表线上常规负载分布。
    - **Hard Tasks（202 个）**：来自高失败率流程，更具挑战性。
  - 所有任务均通过 **ActionTrail 审计日志规则** 验证，不使用 LLM-as-Judge，保证评估客观性。
  - 测试集与训练集在文档级别完全隔离。

---

### 实验设置与评估指标

#### 评估方式
- 每个任务独立运行 **3 次**。
- 报告以下指标：
  - `pass@1`：平均成功率 ± 标准差（mean success rate）
  - `pass@3`：任意一次成功即视为通过（反映生产重试场景）

#### Rollout 环境配置
- 使用统一的 **Rollout Environment**，包含：
  - Terraform 预置资源
  - ResourceCoder 动态补全
  - Playwright 控制无头浏览器
- 所有对比模型在此相同环境下运行，公平比较。

#### 基线方法对比
| 模型 | 类型 | 部署方式 | 推理成本（CNY/任务） |
|------|------|----------|---------------------|
| Qwen3-VL-32B-Instruct | 开源基础模型 | 私有部署 | 0.56 |
| AliyunConsoleAgent-32B (SFT) | SFT 微调版 | 私有部署 | 0.56 |
| AliyunConsoleAgent-32B (SFT+GRPO) | 完整两阶段模型 | 私有部署 | 0.56 |
| Qwen3.6-Plus | 商业 API | API-only | 0.97 |
| Kimi K2.6 | 商业 API | API-only | 2.70 |
| GPT-5.5 | 商业 API | API-only | 15.50 |
| Gemini 3 Pro Preview | 商业 API | API-only | 7.00 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（278-task benchmark, Rule-based Evaluation）

| Model | Deployment | Cost (CNY) | pass@1 | pass@3 |
|-------|------------|-----------|--------|--------|
| Qwen3-VL-32B-Instruct (Base) | Private | 0.56 | 43.28% | 60.79% |
| AliyunConsoleAgent-32B (SFT) | Private | 0.56 | 56.89% | 72.66% |
| **AliyunConsoleAgent-32B (SFT+GRPO)** | **Private** | **0.56** | **63.52%** | **75.18%** |
| Gemini 3 Pro Preview | API-only | 7.00 | 65.34% | 79.86% |
| GPT-5.5 | API-only | 15.50 | 62.08% | 76.09% |

> ✅ **AliyunConsoleAgent-32B (SFT+GRPO)** 达到 **63.52%** 的 `pass@1` 成功率，距离最强闭源模型 **Gemini 3 Pro Preview（65.34%）仅差 1.82 pp**，且 **成本降低 92%**。

---

### 不同难度任务的表现分解（pass@3）

| Model | Standard (76) | Hard (202) |
|-------|---------------|-----------|
| Base Model | 71.05% | 56.93% |
| SFT | 81.58% | 69.31% |
| SFT+GRPO | 84.21% | 71.78% |
| Gemini 3 Pro Preview | 88.16% | 76.73% |

- GRPO 在标准和困难任务上分别带来 **+2.63 pp** 和 **+2.47 pp** 提升，说明其增强了应对复杂逻辑的能力。

---

### 消融实验结果（Ablation Study）

| Training Stage | pass@1 | Δ vs. SFT |
|----------------|--------|---------|
| Base Model | 43.28% | — |
| SFT (distillation) | 56.89% | baseline |
| SFT → GRPO | **63.52%** | **+6.63 pp** |

- **SFT 贡献最大增益（+13.61 pp）**：奠定基本交互能力。
- **GRPO 进一步提升 +6.63 pp**，且统计显著（95% CI [2.97, 9.64]），表明在线 RL 对真实环境适应至关重要。

---

## 4. 关键结论和发现

### 主要发现

1. **两阶段训练有效缩小与闭源模型差距**
   - AliyunConsoleAgent-32B 在接近最强闭源模型性能的同时，实现 **92% 成本下降**，具备大规模落地可行性。

2. **GRPO 促使模型从“照章办事”进化为“目标导向”**
   - 出现两种高级能力：
     - **Precondition Construction**：例如，“禁用自动续费”前若已关闭，则先主动开启再执行关闭，以触发可观测行为。
     - **Adaptive Plan Adjustment**：面对 UI 差异（如按钮缺失），能切换至替代路径完成任务（如逐个修改参数代替批量操作）。

3. **高确定性 Rollout 环境是 RL 成功的关键前提**
   - 若不解决资源依赖问题，环境失败将污染 reward signal，导致策略无法收敛。
   - 该设计使 RL 能专注于学习“真正决策错误”，而非应对系统噪音。

4. **Rule-based Reward 更可靠**
   - 基于 ActionTrail 的审计日志验证避免了 LLM-as-Judge 的主观性和幻觉问题，适合生产环境。

---

### 局限性（Limitations）

1. **Provisioning Maintenance 开销大**
   - 当前依赖静态 Terraform 模板，随平台更新易过期，需定期用闭源模型重新生成，长期成本仍高。
   - 未来需训练本地 coding agent 或开发通用模板系统。

2. **Visual Grounding 依赖脆弱的 DOM 解析**
   - 当前使用 SoM（Set-of-Mark）标注依赖前端 HTML 结构提取，维护负担重。
   - 期待未来转向纯视觉交互（pure-vision interaction）以摆脱 DOM 限制。

3. **基础设施成本仍然可观**
   - 对高端资源（如 GPU 实例）频繁验证经济不可持续。
   - 需引入动态调度机制，根据资源成本调整测试频率。

---

### 未来工作方向

- 训练 **本地化的 ResourceCoder 模型**，减少对外部 API 的依赖。
- 探索 **端到端视觉理解代理**，绕过 DOM 解析环节。
- 构建 **成本感知测试调度器**，优化资源利用率。
- 将框架推广至其他运维场景：如故障诊断、权限治理、合规审计等。

---

> 🔚 **总结一句话**：  
> **AliyunConsoleAgent 通过“蒸馏 + GRPO”两阶段训练，在真实云环境中实现了高性能、低成本、可私有部署的 Web Agent，推动了自动化文档验证向规模化落地迈进。**

</details>

---

### 12. [Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models](https://arxiv.org/abs/2606.09401)

**Authors**: Bart{\l}omiej Marek, Lorenzo Rossi, Vincent Hanke, Xun Wang, Michael Backes, Franziska Boenisch, Adam Dziedzic  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.09401v1  

#### Abstract
Recent work has applied differential privacy (DP) to adapt large language models (LLMs) for sensitive applications, offering theoretical guarantees. However, its practical effectiveness remains unclear, partly due to LLM pretraining, where overlaps and interdependencies with adaptation data can unde...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前，尽管 **Differential Privacy (DP)** 被广泛用于保护大语言模型（LLMs）在敏感任务上的微调过程，其**理论隐私保证**在实践中是否有效仍不明确。主要原因在于：
- LLMs 在大规模公开语料上预训练，而微调数据可能与预训练数据存在重叠或分布相似性。
- 这种“预训练-微调”（pretrain-adapt）范式下的复杂交互可能导致隐私泄露，即使应用了 DP。

然而，现有研究大多孤立地分析预训练或微调阶段的隐私风险，缺乏对整个流程中**经验性隐私泄露**的系统性基准评估。

本论文旨在填补这一空白，通过构建一个全面的基准框架，实证评估在不同数据分布、适配方法和隐私预算下，DP 微调的实际隐私保护效果。

---

### **提出了什么新方法或新思路**

#### **(1) 全面的隐私风险基准测试框架**
论文设计了一个系统的实验框架，从多个维度评估 DP 微调的隐私风险：
- **数据分布关系**：系统性地比较了三种微调数据与预训练数据的关系：
  - **重叠 (Overlap)**：微调数据直接来自预训练集。
  - **同分布 (IID)**：微调数据来自预训练分布的验证集（无重叠）。
  - **异分布 (OOD)**：微调数据来自完全不同的分布（如德语维基百科）。
- **多种适配方法 (Adaptation Methods)**：评估了四种主流的微调技术：
  - **Full Fine-Tuning**：全参数微调。
  - **Head Fine-Tuning**：仅微调最后一层。
  - **LoRA (Low-Rank Adaptation)**：一种高效的 Parameter-Efficient Fine-Tuning (PEFT) 方法。
  - **Prefix Tuning**：另一种 PEFT 方法。
- **多样的隐私预算 (Privacy Budgets)**：从无隐私 (`ε=∞`) 到高隐私 (`ε=0.1`) 的多个 `ε` 值。

#### **(2) 提出“整体隐私审计”（Holistic Privacy Auditing）框架**
这是本文最重要的**概念性创新**。作者认为，不能将预训练和微调的隐私风险割裂看待，因此提出了一个四阶段的统一审计框架，每个阶段都定义为一个**对抗性游戏 (adversarial game)**：
1.  **审计预训练 (Auditing Pretraining)**：评估预训练模型本身的隐私泄露。
2.  **审计微调 (Auditing Adaptation)**：评估微调数据在最终模型中的泄露。
3.  **联合审计 (Joint Auditing)**：同时评估两个数据集的泄露，考虑它们的交互。
4.  **微调后审计预训练 (Post-Adaptation Auditing of Pretraining)**：评估微调过程如何影响原始预训练数据的隐私。

该框架为未来研究提供了形式化的基础，强调必须在整个模型生命周期内进行隐私评估。

---

### **相比现有方法的优势**

| 方面 | 本文优势 |
| :--- | :--- |
| **评估范围** | 现有工作通常只关注非私有微调或单一场景。本文是首个系统性地、跨数据分布、跨适配方法、跨隐私预算来评估**DP微调**经验隐私风险的基准。 |
| **问题洞察** | 揭示了“**分布接近性 (distributional closeness)**”是比“数据重叠”更根本的隐私风险驱动因素，这是一个反直觉且重要的发现。 |
| **方法论** | 提出的四阶段“整体隐私审计”框架，为解决 LLM 隐私问题提供了一个全新的、更全面的视角，超越了传统的孤立分析。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

- **预训练数据**：所有模型均基于 **The Pile** 数据集进行预训练。
- **微调数据集**：
  - **IID (同分布)**：
    - `Bookcorpus2` (书籍)
    - `GitHub` (代码)
    - `Enron Emails` (邮件)
  - **OOD (异分布)**：
    - `SAMSum` (英文对话摘要)
    - `GermanWiki` (德语维基百科)

### **实验设置和评估指标**

#### **核心攻击方法 (评估指标)**
- **Membership Inference Attack (MIA)**：使用最先进的 **Robust Membership Inference Attack (RMIA)** 来衡量成员推断风险。主要指标是 **AUC (Area Under the Curve)**，AUC 越高表示隐私风险越大。
- **数据提取攻击 (Data Extraction Attack)**：通过向微调数据注入 **canary data**（对抗性前缀），并测量其 **exposure** 来量化记忆化程度。Exposure 值越高，表示数据越容易被提取。

#### **模型**
- 主要使用 **Pythia** 和 **GPT-Neo** 家族的多个尺寸模型（70M 到 2.8B）。
- 默认使用 **Pythia 1B** 作为主模型。
- 也包含了开源模型 **OLMo 1B** 和 **OLMo2 1B** 的结果。

#### **基线方法对比**
本文的“基线”并非指其他算法，而是指在以下方面的对比：
- 不同的 **Adaptation Methods** (LoRA vs. Full Fine-Tuning vs. Prefix Tuning vs. Head Fine-Tuning)。
- 不同的 **数据分布** (Overlap vs. IID vs. OOD)。
- 不同的 **隐私预算** (`ε=∞`, `8`, `0.1`)。
- 不同的 **攻击者知识** (使用何种 shadow model)。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据与对比结果**

#### **(1) 数据分布是隐私风险的关键决定因素 (RQ1)**
- **核心发现**：微调数据与预训练数据的**分布越接近，隐私风险越高**。
- **具体数据**：
  - 在 `ε=8` 的中等隐私预算下，**IID** 数据的平均 AUC (0.78-0.90) **显著高于** **OOD** 数据的平均 AUC (0.63-0.87)。
  - 更令人惊讶的是，**IID 数据**（无重叠）的泄露程度与**完全重叠的数据**几乎相同（见图9）。这表明，仅仅是“分布相似”就足以导致严重的隐私泄露。

#### **(2) LoRA 在多数情况下提供最佳的经验隐私保护 (RQ2, RQ6)**
- **在 MIA 攻击下**：
  - 对于 **OOD** 数据，**LoRA** 一致地表现出最低的 AUC，例如在 `GermanWiki` 上，`ε=8` 时 AUC 仅为 **0.58**，优于 Full Fine-Tuning 的 0.77。
  - 对于 IID 数据，LoRA 也表现稳健，虽然 Head Fine-Tuning 有时略优。
- **在隐私-效用权衡上 (RQ6)**：
  - **LoRA** 在保持相近效用（以 perplexity 衡量）的同时，始终能实现更低的隐私风险（以 RMIA AUC 衡量）。例如，在 `GermanWiki` 上，LoRA 在 perplexity 14.27 时 AUC 为 0.77，而 Full Fine-Tuning 在更高 perplexity (14.60) 下 AUC 却高达 0.82。

#### **(3) Prefix Tuning 最易受到数据提取攻击 (RQ3)**
- **核心发现**：虽然 LoRA 在 MIA 上表现最好，但在更严重的**数据提取攻击**中，**Prefix Tuning** 是最脆弱的。
  - 表22显示，在 `ε=∞` 时，Prefix Tuning 的 canary exposure 高达 **6.71**，而 LoRA 和 Head Fine-Tuning 的 exposure 接近随机猜测水平（~1.44）。
- **结论**：不同的适配方法对不同类型的攻击具有不同的鲁棒性。

#### **(4) 高隐私预算 (`ε < 0.1`) 才能提供实际保护**
- 实验表明，在 `ε=8` 的中等隐私预算下，针对 IID 数据的 MIA 攻击仍然非常成功（AUC > 0.75）。只有当 `ε` 降低到 **0.1** 时，AUC 才会显著下降，接近随机水平。这说明，为了在实践中获得有意义的保护，需要非常严格的 DP 约束。

#### **(5) 攻击者知识至关重要 (RQ4)**
- 当攻击者拥有与目标模型架构、初始化和训练数据分布相同的 **shadow model** 时，RMIA 攻击最为成功。
- 如果攻击者只能访问预训练模型本身作为参考，攻击效果会大幅下降。这凸显了公开可用的 LLMs 可能带来的额外风险。

---

## **4. 关键结论和发现**

### **主要发现**
1.  **分布接近性是核心风险**：微调数据与预训练数据的**分布相似性**是经验性隐私泄露的主要驱动因素，其影响甚至超过了直接的数据重叠。
2.  **LoRA 是更安全的选择**：在多种设置下，**LoRA** 这种 PEFT 方法在提供强大的经验隐私保护方面表现最佳，尤其是在处理 OOD 数据时，并且在隐私-效用权衡上具有优势。
3.  **方法间存在权衡**：没有一种适配方法在所有攻击下都是最优的。例如，Prefix Tuning 虽然在某些 MIA 下表现尚可，但极易遭受数据提取。
4.  **中等 DP 无效**：常用的中等隐私预算（如 `ε=8`）无法有效抵御针对 IID 数据的攻击，**高隐私设置**（`ε < 0.1`）是必要的。
5.  **需要整体视角**：预训练和微调阶段的隐私风险是相互关联的，必须采用一个**整体的、端到端的审计框架**来进行准确的风险评估。

### **方法的局限性**
- **模型范围有限**：研究集中在基于 The Pile 预训练的开源模型上，无法评估像 GPT-4 或 Gemini 这样的闭源模型，因为它们不支持梯度层面的 DP 微调。
- **攻击假设**：实验依赖于特定的强攻击者模型（如拥有 shadow model），现实世界中的攻击者能力可能有所不同。
- **计算成本**：尽管 LoRA 效率较高，但进行全面的 DP 训练和基准测试仍然非常昂贵。

### **未来工作方向**
1.  **扩展基准**：将此基准框架应用于更多样化的模型（尤其是闭源模型的 API 场景）和数据集。
2.  **开发新防御**：基于“分布接近性”的发现，设计新的 DP 机制或正则化方法，专门针对预训练-微调的交互风险。
3.  **推广整体框架**：鼓励社区采纳并完善提出的四阶段“整体隐私审计”框架，使其成为 LLM 隐私研究的标准范式。
4.  **探索更高效的 PEFT + DP**：进一步优化 LoRA 等 PEFT 方法与 DP 的结合，以在保证隐私的同时最大化效用。

</details>

---

### 13. [Rethinking the Divergence Regularization in LLM RL](https://arxiv.org/abs/2606.09821)

**Authors**: Jiarui Yao, Xiangxin Zhou, Penghui Qi, Wee Sun Lee, Liefeng Bo, Tianyu Pang  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.09821v1  

#### Abstract
Reinforcement learning (RL) has become a key component of post-training large language models (LLMs). In practice, LLM RL is often off-policy because of training-inference mismatch and policy staleness, making trust-region control essential for stable optimization. Mainstream methods such as PPO and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Rethinking the Divergence Regularization in LLM RL

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大语言模型（LLM）在强化学习（RL）后训练中普遍采用**off-policy**范式，由于训练-推理不匹配（training-inference mismatch）和策略陈旧（policy staleness），导致需要**信任域控制**（trust-region control）来保证优化稳定性。

主流方法如 **PPO** 和 **GRPO** 使用基于重要性比率（importance ratio）的裁剪机制近似实现信任域控制。然而，在 LLM 长尾词表（long-tailed vocabulary）下，该比率是分布偏移（distributional shift）的**不良代理**：
- 对低概率 token，微小的概率提升可能导致极大的比率变化，造成过度约束；
- 对高概率 token，适度的比率变化可能引起显著分布移动，却未被充分限制。

近期工作 **DPPO** 提出用基于散度（divergence-based）的硬掩码（hard mask）替代比率裁剪，以采样 token 的绝对概率偏移（absolute probability shift）定义信任域边界（Binary-TV）。但其仍使用**二值掩码**，一旦超出边界即丢弃梯度，缺乏纠正信号。

### 🚀 提出的新方法：DRPO（Divergence Regularized Policy Optimization）
本文提出 **DRPO**，将 DPPO 中的硬掩码替换为一个**平滑的优势加权二次正则项**（advantage-weighted quadratic regularizer），作用于策略偏移上的绝对概率变化。

#### 核心思想：
- 将 Binary-TV 信任域约束重写为 token 自适应的比率边界；
- 借鉴 **SPO** 的构造方式，设计一个连续可导的目标函数；
- 最终目标函数形式为：

$$
\mathcal{L}_{\text{DRPO}} = \mathbb{E}_{y \sim \pi(\cdot|x)} \sum_t \left[ r_t A_t - \frac{|A_t|}{2\delta} p(y_t|s_t) (r_t - 1)^2 \right]
$$

其中 $ r_t = \pi(y_t|s_t)/\mu(y_t|s_t) $ 是重要性比率。

### 🔍 相比现有方法的优势
| 方法 | 优势 |
|------|------|
| **vs PPO / GRPO / SPO** | 不再依赖比率约束，避免对长尾 token 过度惩罚或对高频 token 约束不足；梯度权重有界，更稳定。 |
| **vs DPPO** | 保留相同的 Binary-TV 信任域几何结构，但用**平滑梯度重加权**取代硬掩码，提供跨边界的纠正信号，避免梯度突变。 |
| **vs KL / TV 正则化** | 其他标准散度惩罚在采样梯度下会退化为比率空间或二值权重，无法实现理想的平滑 Binary-TV 边界。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **主数据集**：DAPO 数据集的一个过滤子集，包含约 **13K 数学问题**，使用规则验证器（rule-based verifier）进行奖励反馈。
- **小规模测试集**：1,460 个可解问题，用于快速验证（sanity check）。

### ⚙️ 实验设置
- **模型**：
  - Qwen3-4B-Base
  - Qwen3-30B-A3B-Base
  - Qwen3.5-35B-A3B-Base
  - DeepSeek-R1-Distill-Qwen-1.5B (**R1D**)
- **精度设置**：
  - 默认 BF16
  - 额外测试 FP8 推理（rollout only）和 FP8 端到端（FP8-E2E），模拟数值不稳定场景
- **框架**：VeRL 框架，训练后端 Megatron，推理后端 vLLM
- **评估指标**：
  - 在 **AIME 2024** 和 **AIME 2025** 上平均得分
  - 每题采样 16 条响应取平均分数
  - 同时监控响应长度、熵、训练奖励等辅助指标

### 🔁 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| Unregularized Surrogate | 无信任域 | 原始策略梯度，易崩溃 |
| GRPO | Ratio-clipping | 使用 clip-higher 技巧（$ \epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28 $） |
| SPO | Smooth ratio-regularizer | 平滑版本 PPO，但仍是比率空间 |
| DPPO | Hard divergence mask | 使用 Binary-TV 散度掩码（$ \delta=0.15 $） |
| DRPO (Ours) | Smooth divergence regularizer | 本文提出，$ \delta=12.5 $ |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Figure 3）
在所有六种实验设置下（不同模型、架构、精度），**DRPO 始终表现最优或持平最佳基线**：

| 设置 | DRPO 表现 |
|------|---------|
| Qwen3-30B-A3B-Base | 收敛快，最终准确率最高 |
| Qwen3.5-35B-A3B-Base | 显著优于其他方法 |
| Qwen3-4B-Base | 稳定训练，避免崩溃 |
| FP8 Rollout & FP8-E2E | 在低精度挑战下仍保持稳定，而 GRPO/SPO 出现训练崩溃 |

> 示例：在 Qwen3-4B-Base 上，未正则化方法从 0.25 下降到 0.17，而 DRPO 维持稳定上升趋势。

### 📈 与基线方法对比
| 对比维度 | 结果 |
|--------|------|
| **vs Ratio-based 方法 (GRPO, SPO)** | 在低精度和 MoE 架构下严重不稳定，常提前崩溃；即使成功也收敛慢、性能差。 |
| **vs Hard Mask 方法 (DPPO)** | DPPO 能稳定训练，但收敛速度较慢，最终性能低于 DRPO，说明“平滑纠正”优于“直接截断”。 |
| **vs 无正则化方法** | 无正则化虽在部分设置有效，但在三处出现明显性能下降，证明信任域仍必要。 |

### 🔬 消融实验结果（Ablation Studies）

#### （1）是否使用绝对优势 $|A|$ 加权？
- 移除 $|A|$ 导致性能下降且训练不稳定（Figure 4）。
- 原因：信任域边界应独立于优势尺度；否则小优势 token 被过正则化，大优势 token 缺乏足够纠正。

#### （2）与其他散度正则化比较（KL, TV）
- 所有替代方案（KL, TV, K3）均不如 DRPO（Figure 5）。
- 分析表明这些方法的梯度诱导的是**比率空间或二值权重**，无法复现 Binary-TV 的平滑边界。

#### （3）正则化是否只应在 DPPO 边界外应用？（Mask-DRPO）
- 实验发现：仅在外侧应用正则化的 **Mask-DRPO** 性能接近完整 DRPO（Figure 11）。
- 说明：**主要增益来自边界外的纠正机制**，而非内部平滑衰减。

#### （4）超参数敏感性分析
- DRPO 对阈值 $\delta$ 相对鲁棒（Figure 10），$\delta=12.5$ 和 $2.5$ 差异不大。
- 相比之下，DPPO 需精细调参（Figure 9），不同设置需不同 $\delta$。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **比率不是好代理**：在 LLM 长尾词表中，`|r_t - 1|` 不能准确反映真实分布偏移，导致 PPO/SPO 类方法在低概率 token 上梯度方差极大。
2. **Binary-TV 更合理**：绝对概率偏移 $|\pi(y|s) - \mu(y|s)|$ 更符合 TV 散度几何，更适合建模 LLM 的策略更新。
3. **平滑优于硬掩码**：相比 DPPO 的 abrupt 截断，DRPO 提供连续梯度重加权，在边界内渐进衰减，在边界外主动纠正，显著提升稳定性和效率。
4. **梯度形式比目标函数更重要**：许多看似合理的正则项（如 KL、TV）在单样本采样下梯度行为退化，实际效果差。**正则器的设计必须考虑其诱导的梯度几何**。

### ⚠️ 方法的局限性
- 当前 DRPO 基于 Binary-TV 近似，未考虑整个输出分布的全局散度。
- 虽然适用于多种模型和精度，但尚未在多轮对话或多步推理任务中广泛验证。
- 仍假设 rollout 数据来自固定行为策略 $\mu$，对极端陈旧数据的鲁棒性有待进一步研究。

### 🔮 未来工作方向
- 探索更精确的 per-step 或 per-sequence 散度估计（如 top-k TV）。
- 将 DRPO 思想扩展至 offline RL 或 imitation learning 场景。
- 研究自适应调整 $\delta$ 的机制，结合 KL 控制或其他稳定性指标。
- 探索在 vision-language 或 agent 决策任务中的泛化能力。

---

> 💡 **一句话总结**：  
> DRPO 通过引入一个简单但有效的 advantage-weighted $ \ell^2 $ 正则项，实现了**兼具 DPPO 的散度感知几何与 SPO 的平滑梯度特性**，为 LLM 强化学习提供了更稳定、高效且实用的信任域优化方案。

</details>

---

### 14. [ConMem: Structured Memory-Guided Adaptation in Training-Free Multi-Agent Systems](https://arxiv.org/abs/2606.08702)

**Authors**: Zhixun Tan, Qiang Chen, Tairan Huang, Xiu Su, Yi Chen  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.08702v1  

#### Abstract
Recent advances have improved the adaptive capabilities of LLM-based multi-agent systems (MAS) through memory-, skill-, and learning-based approaches, yet these approaches remain challenged by noisy trajectories, insufficient modeling of memory-skill relations, and reliance on additional training or...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ConMem: Structured Memory-Guided Adaptation in Training-Free Multi-Agent Systems**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 LLM 的多智能体系统（MAS）在实现**多智能体适应**（Multi-Agent Adaptation, MAA）时面临三大挑战：
- **轨迹噪声**：原始交互轨迹冗长、混杂局部工具输出、中间分歧和过时上下文，干扰决策信号。
- **记忆-技能关系建模不足**：现有方法缺乏对策略间依赖、冲突和冗余的显式协调机制。
- **依赖额外训练或高质量监督**：许多先进方法需通过 RL 或微调优化记忆使用，增加计算开销与部署复杂度。

### **提出的新方法与新思路**
作者提出 **ConMem**，一个**无需训练**（training-free）、**关系感知**（relation-aware）的框架，通过**跨经验协调**（cross-experience coordination）实现高效的多智能体适应。

#### **核心创新点**：
- **结构化记忆卡**（Typed and Signed Memory Cards）  
  将原始轨迹压缩为带有类型的原子单元，编码 `state`, `plan`, `execution`, `evaluation` 四个维度，并通过“签名”区分成功策略（positive）与失败警告（negative），形成可复用、可检查的适应性知识。

- **关系感知的记忆图**（Relation-Aware Memory Graph）  
  构建带类型边的图结构（supports, satisfies, constrains, conflicts），显式建模策略间的依赖与冲突，支持运行时推理恢复策略链。

- **预算控制下的协调组合**（Budgeted Coordination & Composition）  
  在提示长度限制下，动态检索、扩展、协调并组合最相关的记忆卡，确保注入主机的上下文是紧凑、一致且任务对齐的。

- **完全无需训练**  
  不修改模型权重、不调整智能体角色或通信结构，仅通过提示前缀控制实现持续适应。

### **相比现有方法的优势**
| 维度 | 现有方法 | ConMem |
|------|--------|-------|
| 是否需要训练 | 多数需要（如 LatentMem, MemRL） | ❌ 完全无需训练 |
| 协调能力 | 被动检索（flat list） | ✅ 显式冲突解决与依赖恢复 |
| 冗余处理 | 隐式由 LLM 处理 | ✅ 主动去重与剪枝 |
| 效率 | 扩展候选多，推理慢 | ✅ 剪枝 >50% 候选，规划开销降低 80%+ |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
四个基准任务覆盖不同难度与结构特性：

| 数据集 | 任务类型 | 描述 |
|--------|---------|------|
| **TriviaQA** | 开放域问答 | 测试证据检索与验证纪律 |
| **PopQA** | 开放域问答 | 强调反参数化记忆偏见的能力 |
| **KodCode** | 代码生成 | 单元测试通过率衡量程序修复策略复用 |
| **PDDL (via PDDLGym)** | 符号规划 | 衡量状态-计划记忆的长期复用能力 |

### **实验设置与评估指标**
- **LLM 主干**：统一使用 `Qwen/Qwen3-4B-Instruct-2507`，确定性解码。
- **MAS 主机**：评估三种主流架构：
  - **AutoGen**
  - **CAMEL**
  - **MacNet**
- **评估协议**：**预续式评估**（prequential），即任务 $t$ 只能访问此前提交的记忆卡。
- **提示预算**：严格控制 token 数量，模拟真实场景下的上下文窗口限制。

### **评估指标**
| 任务类型 | 指标 |
|----------|------|
| QA（TriviaQA, PopQA） | Gold-alias 答案准确率 |
| Code（KodCode） | 单元测试平均通过率 |
| Planning（PDDL） | 归一化目标满足得分（normalized goal-satisfaction score） |

### **基线方法对比**
分为四类进行公平比较（相同主机、相同 LLM、相同提示预算）：

| 类别 | 代表方法 |
|------|--------|
| **无记忆** | No-memory baseline |
| **智能体框架** | ChatDev, MetaGPT, JoyAgent, OAgents |
| **训练自由记忆方法** | Generative Agents, Voyager, SimpleMem, G-Memory, ReMe |
| **可学习记忆方法** | LatentMem（需训练） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1 总结）**
ConMem 在所有主机和任务上均取得显著提升：

| 主机 | 平均增益（vs No-memory） | 最佳单项增益 |
|------|--------------------------|-------------|
| AutoGen | +12.36 pts | KodCode: +12.40 pts |
| CAMEL | +10.88 pts | TriviaQA: +14.07 pts |
| MacNet | +12.90 pts | PDDL: +6.34 pts |

> ✅ **平均超越最强训练自由基线 ReMe 0.02–0.87 pts，超越可训练基线 LatentMem 0.95–1.81 pts**

### **与基线方法的对比结果**
- **优于所有训练自由方法**：在 QA 和代码任务上明显领先 Voyager、G-Memory 等。
- **媲美甚至超越需训练方法 LatentMem**：说明无需训练也能达到高级适应效果。
- **推理效率更高**：
  - **剪枝超过 50% 扩展候选**（Figure 5）
  - **规划阶段开销减少超 80%**

### **消融实验结果（Figure 3）**
移除任一组件均导致性能下降，证明各模块互补有效：

| 消融条件 | 性能影响（以 KodCode/CAMEL 为例） |
|----------|-------------------------------|
| **No graph expansion** | ↓2.0 pts（依赖未恢复） |
| **No coordination** | ↓3.3 pts（冲突未解决） |
| **No failure reflection** | ↓1.7 pts（失败教训丢失） |
| **No failure admission** | ↓1.9 pts（负向记忆缺失） |

> 🔍 特别是在 **KodCode** 和 **PDDL** 上，图扩展与协调作用更显著，因其策略依赖性强。

---

## **4. 关键结论和发现**

### **主要发现**
1. **记忆的价值在于协调而非数量**  
   ConMem 的优势不来自“记住更多”，而是“协调得更好”。它通过关系图主动解决冲突、恢复依赖，在有限预算内提供高质量指导。

2. **失败记忆具有重要价值**  
   负面记忆卡（negative cards）作为“避免线索”（avoid cues），帮助系统规避重复错误，尤其在复杂任务中显著提升鲁棒性。

3. **无需训练即可实现高效适应**  
   通过结构化记忆设计与运行时协调，可在冻结主机的前提下实现稳定性能提升，具备强部署友好性。

4. **协调即压缩**  
   协调模块实现了 >50% 的候选剪枝，表明瓶颈不是检索广度，而是**策略一致性**（slate consistency）。

### **方法的局限性**
- **控制器质量依赖 LLM 抽取能力**：卡片提取、关系判断等仍依赖 LLM 推理，存在误差传播风险。
- **固定阈值非自适应**：当前使用校准阈值而非学习策略，可能无法最优适配动态环境。
- **受限于基础模型能力**：不能创造主机本身不具备的技能，仅能优化已有行为模式。

### **未来工作方向**
- 将协调规则从启发式升级为**可学习策略**（learnable policies），同时保持结构化抽象。
- 探索更丰富的**关系类型**或**长程图推理**机制。
- 扩展到安全敏感领域，研究如何平衡策略复用与**隐私保护、红队审计**等需求。

---

> 📌 **一句话总结**：  
> **ConMem 通过“签名记忆卡 + 关系图协调”的方式，在无需任何训练的情况下，实现了轻量、高效、鲁棒的多智能体持续适应，显著优于现有记忆架构，并揭示了“协调优于堆叠”的新型上下文控制范式。**

</details>

---

### 15. [Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation](https://arxiv.org/abs/2606.06712)

**Authors**: Xingyu Su, Jacob Helwig, Shubham Parashar, Atharv Chagi, Lakshmi Jotsna, Degui Zhi, James Caverlee, Dileep Kalathil, Shuiwang Ji  
**Category**: cs.CL  
**Published**: 2026-06-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.06712v1  

#### Abstract
We study the transformation of autoregressive models (ARLMs) into diffusion language models (DLMs). Rather than pretraining from scratch, prior work replaces the causal attention in ARLMs with bidirectional attention and then trains the resulting model using a DLM objective. However, these approache...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前将 **Autoregressive Language Models (ARLMs)** 转换为 **Diffusion Language Models (DLMs)** 的方法存在两个关键挑战：

1. **知识保留不充分**（Knowledge Retention Mismatch）  
   将 ARLM 的因果注意力替换为双向注意力并使用 DLM 目标进行训练时，会丢失 ARLM 在预训练阶段通过 next-token prediction 学到的知识。

2. **训练-推理状态不匹配**（Train-Inference Mismatch）  
   DLMs 在训练时使用随机掩码（random masking）生成中间状态，但在推理时采用基于置信度的反向去噪路径（reverse unmasking trajectories），导致训练分布与推理分布不一致，影响效率。

---

### 提出的新方法：On-Policy Diffusion Language Models (OPDLM)

作者提出 **OPDLM**，一种基于 **On-Policy Distillation (OPD)** 的 ARLM 到 DLM 的转换框架，其核心思想是：

- **学生模型**（Student）：一个带有 block-wise causal attention 的 DLM，初始化自原始 ARLM。
- **教师模型**（Teacher）：冻结的原始 ARLM。
- **训练方式**：学生在自己的反向扩散轨迹上生成部分去噪序列，教师提供这些位置上的 token-level logits 作为监督信号，通过 KL 散度进行蒸馏。

该方法结合了：
- **On-Policy Training**：训练状态来自模型自身推理过程，消除 train-inference mismatch。
- **Knowledge Distillation**：从原始 ARLM 中继承 token 级别预测能力，增强知识保留。

---

### 相比现有方法的优势

| 优势维度 | 具体表现 |
|--------|--------|
| **数据效率** | 所需训练 token 数量仅为现有 DLM 方法的 **1/15 到 1/7000**（如仅用 0.066B tokens） |
| **计算效率** | FLOPs 下降两个数量级（如 OPDLM-8B 仅需 4.2×10¹⁸） |
| **无需 DLM 预训练** | 可直接将 ARLM “post-train” 成 DLM，避免昂贵的 DLM 从头预训练 |
| **零样本能力保留** | 保留了原 ARLM 的 multilingual 和 extended thinking 能力 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **训练数据**：约 61.8K prompts，涵盖四个领域：
  - Math: 20,222 samples (DAPO, Nemotron-v2-Math)
  - Code: 21,594 samples (TACO, KodCode-Light-RL, AceCode)
  - Science: 10,000 samples (Nemotron-v2-STEM)
  - Chat: 10,000 samples (Nemotron-v2-Chat)

> ⚠️ 注意：只保留 prompt，不使用完整 response，因为采用 on-policy 生成。

### 实验设置

- **基础模型**：Qwen3 系列（0.6B, 1.7B, 4B, 8B）
- **学生与教师关系**：同规模 self-distillation（如 Qwen-4B → OPDLM-4B）
- **训练策略**：
  - 使用 rollout-length curriculum 控制生成长度
  - 动态重采样（dynamic remasking）提升多样性
  - Block size = 4（除非特别说明）
- **解码方式**：静态解码（static decoding, 1 token/step）为主；动态阈值控制多 token 解码

### 评估指标

- **通用能力**：MMLU, MMLU-Pro, GPQA-Diamond, IFEval, CEval, LiveBench
- **数学推理**：GSM8K, MATH-500, AIME-24, AIME-25, LMB-Hard, ZebraLogic
- **代码生成**：HumanEval, MBPP, LCB-v6, Codeforces
- **多语言能力**：MMMLU-lite, INCLUDE-lite, MT-AIME2024, MLogiQA
- **效率指标**：训练 tokens、FLOPs、tokens per step（吞吐量）

### 基线方法对比

| 基线模型 | 类型 | 特点 |
|--------|------|------|
| **LLaDA-8B** | Scratch-trained DLM | 从头训练，使用 full-attention diffusion |
| **Dream-7B** | ARLM-init DLM | 从 ARLM 初始化，全注意力扩散 |
| **SDAR-8B / SDAR-4B** | Block Diffusion DLM | 强基线，使用 off-policy 继续预训练 |
| **Fast-dLLM-v2-7B** | Block Diffusion DLM | 高效 block diffusion 方法 |
| **Off-Policy Distillation** | 消融基线 | 使用离线 teacher response 进行蒸馏 |
| **SFT (Supervised Fine-Tuning)** | 对照方法 | 在 ARLM 生成 response 上做标准微调 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 模型 | 训练 Tokens ↓ | FLOPs (×10¹⁸) ↓ | AIME-24 ↑ | GPQA-Diamond ↑ | GSM8K ↑ |
|------|----------------|------------------|-----------|------------------|---------|
| SDAR-8B | 55B | 2640 | 10.0 | 40.2 | 91.3 |
| **OPDLM-8B** | **0.066B** | **4.2** | **14.7** | **36.1** | **87.1** |
| Fast-dLLM-v2-7B | 1B | 42 | 10.0 | 27.3 | 83.7 |
| LLaDA-8B | 1500B | 72000 | 2.1 | 31.8 | 78.6 |

> ✅ **OPDLM-8B 仅用 0.066B tokens（≈1/833 SDAR, ≈1/22700 LLaDA）达到更强或相当性能**

---

### 与基线方法的对比结果

- **性能竞争力强**：
  - 在数学任务（AIME-24/25, MATH-500）上显著优于大多数基线
  - 在复杂推理任务（GPQA-Diamond）上接近甚至超越 SDAR
  - 在通用知识（MMLU）略低于 SDAR，但远超其他 DLM

- **数据效率碾压级优势**：
  - 图1显示 OPDLM 建立了新的 **Pareto Frontier**，在极低资源下实现高性能
  - 最高节省达 **7,000× 训练 tokens**

- **零样本能力保留**：
  - **Extended Thinking**（表2）：即使未显式训练 CoT，也能通过 `<think>` 提示激活推理链，AIME-24 达到 18.6%
  - **Multilingual 能力**（表3）：尽管训练无多语言数据，仍保持良好跨语言理解能力，接近 SDAR 表现

---

### 消融实验结果（Table 4）

| 方法 | AIME-24 (4B) | AIME-24 (8B) | GPQA-Diamond (8B) |
|------|--------------|--------------|--------------------|
| Off-Policy Distillation | 10.9 | 17.0 | 33.6 |
| OPDLM-off (on-policy data, random mask) | 11.2 | 12.9 | 38.3 |
| **OPDLM (full on-policy)** | **14.4** | **14.7** | **36.1** |

> 🔍 发现：
> - **On-policy 数据是关键驱动因素**：从 off-policy 切换到 on-policy 数据带来稳定增益
> - 掩码轨迹设计（random vs. reverse）影响较小，说明核心在于“训练在真实推理路径上”

此外，在 **SFT vs OPDLM** 对比中（Appendix A.5）：
- 两者在标准 benchmark 上性能相近
- 但在 **dynamic sampling**（多 token 解码）下，OPDLM 更鲁棒，SFT 性能下降明显 → 证明 OPDLM 更好地解决了 train-inference mismatch

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **ARLM-to-DLM 转换可作为高效 post-training 范式**  
   无需从头预训练 DLM，只需少量数据即可完成高质量转换。

2. ✅ **On-Policy Distillation 是解决 train-inference mismatch 的有效机制**  
   在模型自身生成路径上训练，极大提升了泛化性和稳定性。

3. ✅ **知识蒸馏能有效保留 ARLM 的先验知识**  
   包括 multilingual、reasoning、code 等未在训练集中覆盖的能力均可零样本保留。

4. ✅ **任务专用 DLM 可直接构建**  
   如 OPDLM-MATH，在数学数据上直接蒸馏即可获得专家级性能，无需通用 DLM 中间阶段。

5. ✅ **支持灵活的 inference efficiency 控制**  
   通过调整 block size 或 confidence threshold，可在 accuracy 与 throughput 之间权衡（图3–6）：
   - block size=16 + γ=0.8 → 平均 >3.5 tokens/step
   - throughput 随训练快速收敛

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **编码任务表现较弱** | 在 HumanEval、MBPP 上低于 SDAR 和 Dream，可能因训练数据中 code coverage 不足 |
| **依赖高质量 ARLM 教师** | 若教师本身能力有限，则学生上限受限 |
| **小模型效果受限** | 0.6B 模型难以从小 teacher 获得足够监督信号（见 Table 8） |
| **缺乏系统性的 teacher-student 规模研究** | 当前仅初步探索不同 teacher 大小的影响 |

---

### 未来工作方向

1. **高质量多样化数据构建**  
   投资于更丰富、均衡的 prompt corpus，有望进一步释放 OPDLM 的潜力。

2. **大规模思维链蒸馏**（Reasoning Distillation at Scale）  
   将 `<think>` 模式扩展到通用领域，打造真正具备“思考”能力的 DLM。

3. **跨尺寸与跨家族蒸馏**（Cross-size & Cross-family Distillation）  
   探索更大 teacher 是否能提升小 student 性能，以及是否可在 Llama/Qwen 等不同架构间迁移。

4. **更优的 teacher-target alignment 设计**  
   如引入 logit shifting、feature matching 等高级蒸馏技术。

5. **降低部署偏见风险**  
   结合 bias auditing、content filtering 等对齐手段，确保安全应用。

---

> 🌱 **总体评价**：  
> OPDLM 成功将 ARLM-to-DLM 转换从“昂贵的继续预训练”转变为“高效的 post-training”，兼具 **高性能、高效率、强泛化** 三大优势，为未来大模型架构演进提供了实用且可持续的技术路径。

</details>

---

### 16. [A Low-Latency Semantic State Estimator using Latent Predictive Learning for Dynamic Network Monitoring and Orchestration](https://arxiv.org/abs/2606.08869)

**Authors**: Hari Madhukumar, Haiyuan Li, Xiaolan Liu, Andy Corston-Petrie, Dimitra Simeonidou  
**Category**: cs.DC  
**Published**: 2026-06-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.08869v1  

#### Abstract
Closed-loop network monitoring and orchestration increasingly require semantic interpretations of live telemetry beyond raw counter collection. However, dynamic cloud-edge environments change both the active node set and the monitoring query at runtime, while control loops demand bounded millisecond...

---

### 17. [ConSteer-RL: Steering Reasoning Capabilities in Large Language Models via Confidence-Aware Reinforcement Learning](https://arxiv.org/abs/2606.08088)

**Authors**: Qing Miao, Yiming Zhao, Jing Yang, Chenxi Liu, Yuehai Chen, Yuewen Liu, Shaoyi Du, Badong Chen  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.08088v1  

#### Abstract
Reinforcement Learning from Verifiable Rewards (RLVR) has recently become a key paradigm for improving the reasoning abilities of Large Language Models (LLMs), yet it remains limited by sparse binary rewards and its ignorance of model-internal uncertainty. In this paper, we propose ConSteer-RL, a si...

---

### 18. [Convolutional Sparse Coding via the Locally Competitive Algorithm on Loihi 2](https://arxiv.org/abs/2606.08584)

**Authors**: Geoffrey Kasenbacher, Daniel Ruepp, Gerrit A. Ecke  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.08584v1  

#### Abstract
Sparse coding provides a principled framework for signal representation by expressing an input as a linear combination of only a small number of basis functions. The Locally Competitive Algorithm (LCA) is particularly attractive in the context of neuromorphic computing because its dynamics, leaky in...

---

### 19. [Distilling Safe LLM Systems via Soft Prompts for On Device Settings](https://arxiv.org/abs/2606.09388)

**Authors**: Motasem Alfarra, Cristina Pinneri, Dana Kianfar, Mohammed Almousa, Christos Louizos  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.09388v1  

#### Abstract
Deploying safe large language models (LLMs) on resource-constrained edge devices presents a critical challenge: while dual-model systems combining LLMs with guard models provide effective safety guarantees, their substantial memory and computational demands make them prohibitively expensive for on-d...

---

### 20. [BUDDY: BUdget-Driven DYnamic Depth Routing for Adaptive Large Language Model Inference](https://arxiv.org/abs/2606.09514)

**Authors**: Yuhua Zhou, Shaoqi Yu, Shichao Weng, Changhai Zhou, Mingze Yin, Fei Yang, Aimin Pan  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.09514v1  

#### Abstract
Large language models (LLMs) incur high inference cost due to their depth and parameter scale. Depth pruning can reduce latency by skipping redundant Transformer blocks, but existing methods (i) provide limited control under user-specific compute budgets and (ii) typically fix the routing path, fail...

---

### 21. [How Small Can You Go? LoRA Fine-Tuning 270M-8B Models for Merchant Information Extraction in Financial Transactions](https://arxiv.org/abs/2606.08051)

**Authors**: Donghao Huang, Tomas Drietomsky, Benjamin Barrett, Zhaoxia Wang  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.08051v1  

#### Abstract
Financial transaction processing requires extracting structured merchant information from noisy, abbreviated bank transaction strings at scale. Our current production system, a LoRA-fine-tuned LLaMA 3.1-8B, achieves 96.95% F1 on this task, but deploying 8-billion-parameter models imposes prohibitive...

---

### 22. [SAGE: An LLM-driven Self Reflective Agentic Framework for Fraud Detection](https://arxiv.org/abs/2606.08146)

**Authors**: Yichen Chen, Siying Li, Yuhang Liang, Lijun Wang, Renyang Liu  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.08146v1  

#### Abstract
Fraud detection in payment, e-commerce, and telecommunications systems requires accuracy at the individual level, robustness under severe class imbalance, and ease of understanding for risk managers. Existing methods fall at least one of these requirements: automated machine learning systems search ...

---

### 23. [Improving Cross-Lingual Factual Recall via Consistency-Driven Reinforcement Learning](https://arxiv.org/abs/2606.06586)

**Authors**: Jonathan von Rad, Louis Arts, George Burgess, Eleftheria Kolokytha, Harry O'Donnell, Ektor Oikonomidis Doumpas, Eduardo Sanchez, Yao Lu, Pontus Stenetorp  
**Category**: cs.CL  
**Published**: 2026-06-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.06586v1  

#### Abstract
Large language models (LLMs) trained predominantly on English data encode substantial world knowledge, yet often fail to express it reliably in other languages, a phenomenon known as cross-lingual factual inconsistency. To study and address this, we introduce PolyFact, a large-scale parallel multili...

---

### 24. [Auditing Training Data in Domain-adapted LLMs: LoRA-MINT](https://arxiv.org/abs/2606.06946)

**Authors**: Gonzalo Mancera, Daniel DeAlcala, Aythami Morales, Julian Fierrez, Ruben Tolosana, Francisco Jurado  
**Category**: cs.CL  
**Published**: 2026-06-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.06946v1  

#### Abstract
We present LoRA-MINT, a new methodology for Membership Inference Test (MINT) applied to recent Large Language Models (LLMs) fine-tuned for specific Natural Language Processing (NLP) tasks through Low-Rank Adaptation (LoRA). The primary goal is to assess whether individual samples were part of the tr...

---

### 25. [Engineering Scalable Distributed List Ranking](https://arxiv.org/abs/2606.09318)

**Authors**: Peter Sanders, Matthias Schimek, Tim Niklas Uhl, Thomas Weidmann  
**Category**: cs.DC  
**Published**: 2026-06-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.09318v1  

#### Abstract
The list ranking problem is one of the classical problems of parallel computing, with nontrivial algorithms and many applications as a subroutine for solving other problems. While it has been intensively studied in the early days of parallel computing, few things happened in the last 20 years. In pa...

---

### 26. [SPIN: Decentralized Swarm Control via Tensorized Policy Coordination](https://arxiv.org/abs/2606.07557)

**Authors**: Zhaowen Fan  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.07557v1  

#### Abstract
Decentralized multi-agent swarm coordination on resource-constrained edge platforms remains fundamentally bottlenecked by the exponential scaling of joint action spaces and high-latency communication overhead. This paper introduces the Swarm Policy Interference Network (SPIN) framework, an architect...

---

### 27. [Evaluation of ML Resource Utilization Requires Model Life Cycle Assessment](https://arxiv.org/abs/2606.07632)

**Authors**: Jared Fernandez, Clara Na, Yonatan Bisk, Constantine Samaras, Emma Strubell  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.07632v1  

#### Abstract
Proper accounting of the energy requirements and environmental impact of artificial intelligence (AI) systems is necessary for researchers, developers, policy makers, and users to assess the barriers to building systems at scale. With the growing complexity of pipelines and underlying infrastructure...

---

### 28. [The Easy, the Hard, and the Learnable: Confidence and Difficulty-Adaptive Policy Optimization for LLM Reasoning](https://arxiv.org/abs/2606.07950)

**Authors**: Zhanke Zhou, Xiangyu Lu, Chentao Cao, Brando Miranda, Tongliang Liu, Bo Han, Sanmi Koyejo  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.07950v1  

#### Abstract
RL with verifiable rewards can substantially improve LLM reasoning, yet standard GRPO-style training often treats easy, hard, and learnable questions alike through uniform sampling and weighting, leading to inefficient compute allocation. We study GRPO by tracking token log-probabilities, group-norm...

---

### 29. [Mesh Graph Neural Network Framework for Accelerating Finite Element Simulation for Arbitrary Geometries](https://arxiv.org/abs/2606.08287)

**Authors**: Josiah D. Kunz, Kamal Choudhary  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.08287v1  

#### Abstract
Finite element analysis (FEA) is essential for structural design but remains computationally expensive, particularly when evaluating multiple design iterations or load scenarios. Machine learning surrogate models offer a promising alternative, yet most approaches struggle with a critical limitation:...

---

### 30. [Operator learning for the 2D incompressible Navier-Stokes equations: a conformal prediction approach in the data-scarce regime](https://arxiv.org/abs/2606.08654)

**Authors**: Weinan Wang, Bowen Gang, Hao Deng  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.08654v1  

#### Abstract
In this paper, we propose a perturbation-based conformal prediction framework for uncertainty quantification in operator learning, with a focus on the 2D Navier--Stokes equations. While neural operators provide fast surrogates for expensive PDE solvers, they do not by themselves provide calibrated u...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-08 08:02:47 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression in Long-Context LLM Inference](https://arxiv.org/abs/2607.06519)

**Authors**: Anna C\'ordoba, Adam Puente Tercero, Nerea Angulo Hijo, Mar Linares Tercero, Julia Barrientos, Ainhoa Miranda, Jes\'us Olivera  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.06519v1  

#### Abstract
Long-context LLM inference is increasingly limited by the memory and bandwidth cost of KV caches, yet aggressive compression can remove the layer-specific evidence needed for retrieval and multi-step reasoning. We introduce FreqDepthKV, an inference-time cache compression method that factorizes adja...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression in Long-Context LLM Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在长上下文（long-context）大语言模型（LLM）推理中，**Key-Value (KV) Cache** 的内存占用和带宽开销成为主要瓶颈。随着上下文长度增加，KV Cache 占用线性增长，严重影响推理效率。

现有压缩方法如 token eviction（H2O、SnapKV）、quantization（KVQuant、KIVI）或 uniform depth sharing（MiniCache）虽然能减少缓存大小，但往往**过度简化或抹除层间细微差异**，尤其是在检索密集型或推理敏感任务中，会丢失关键的稀疏证据（sparse evidence），导致生成质量下降。

### 提出了什么新方法或新思路
本文提出 **FreqDepthKV**，一种基于频率引导的深度共享 KV Cache 压缩方法，其核心思想是：

- 将相邻 Transformer 层的 KV 状态沿 depth 维度进行**频域分解**：
  - **低频分量（Low-Frequency Components）**：跨层共享，代表冗余的、平滑变化的深度结构。
  - **高频残差（High-Frequency Residuals）**：稀疏存储，保留对 attention logits 敏感的关键 token-head-layer 交互信息。
- 引入一个**轻量级在线探针（online probe）**，在 prefill 阶段动态为每个 attention head 分配三种模式之一：
  - **Shared-depth mode**：仅使用共享低频分量。
  - **Residual-depth mode**：共享低频 + 稀疏高频残差。
  - **Exact mode**：保留完整 KV Cache。
- 路由决策基于一个**重建感知损失（reconstruction-aware routing loss）**，衡量压缩后是否显著改变原始 attention logits，从而确保关键信息不被破坏。

### 相比现有方法的优势
- **更鲁棒的压缩策略**：不同于统一压缩或粗粒度共享，FreqDepthKV 显式保留影响 attention 决策的高频细节。
- **无需重训练**：纯推理时优化，兼容标准 autoregressive 解码流程。
- **自适应性强**：压缩策略随 prompt 结构和上下文长度动态调整，适用于多样化任务。
- **多维正交性**：可与 token-level（如 H2O）和 precision-level（如 KIVI）压缩方法结合使用。

---

## 2. 核心实验方法和设置

### 使用的数据集
涵盖三大类长上下文任务：
- **问答与检索**：LongBench, Needle-in-a-Haystack, L-Eval, NarrativeQA, Qasper
- **摘要生成**：GovReport, MultiNews
- **代码生成**：HumanEval, MBPP

### 实验设置和评估指标
- **上下文长度**：默认 32k-token prefill window。
- **模型架构**：基于标准长上下文 decoder 模型，所有方法均在 inference-time 应用，无 retraining。
- **评估指标**：
  - **任务性能**：
    - QA：Exact Match (EM), F1
    - Summarization：ROUGE-L
    - Code Generation：pass@1
  - **系统效率**：
    - Decoding Throughput（tokens/s）
    - Time-to-First-Token (TTFT)
    - Peak KV Memory (GB)
    - Effective Compression Ratio

### 基线方法对比
对比了多种代表性 KV Cache 压缩方法：
- **Token Eviction 类**：StreamingLLM, H2O, Scissorhands, SnapKV, PyramidKV
- **Quantization 类**：KVQuant, KIVI
- **Depth Sharing 类**：MiniCache（最相关基线）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）
| Method       | EM   | F1   | ROUGE-L | pass@1 | Tokens/s | TTFT(s) | Peak KV Mem. (GB) | Comp. Ratio |
|--------------|------|------|---------|--------|----------|---------|-------------------|-------------|
| Full KV      | 58.7 | 63.4 | 32.8    | 48.6   | 38.2     | 2.91    | 24.0              | 1.0×        |
| ...          | ...  | ...  | ...     | ...    | ...      | ...     | ...               | ...         |
| **FreqDepthKV** | **58.3** | **63.0** | **32.5** | **48.1** | **70.4** | **2.06** | **6.2**           | **3.9×**     |

> ✅ **FreqDepthKV 在几乎所有指标上接近甚至逼近 Full KV，同时实现 3.9× 压缩比和显著提速。**

### 与基线方法的对比结果
- 相比最强 token-selection 方法 **PyramidKV**：
  - F1 提升 **1.8 pts**（61.2 → 63.0）
  - pass@1 提升 **2.3 pts**（45.8 → 48.1）
  - KV 内存降低 **1.2 GB**
- 相比同属 depth-sharing 的 **MiniCache**：
  - 所有任务指标全面领先（EM: +1.7, F1: +2.0, ROUGE-L: +1.2, pass@1: +2.5）
  - 吞吐提升至 **70.4 vs 65.5 tokens/s**
  - 峰值内存从 6.6 GB 进一步降至 **6.2 GB**
- 在 **Needle-in-a-Haystack、Qasper、code completion** 等 retrieval-sensitive 任务上优势尤为明显，说明其有效保护了关键稀疏证据。

### 消融实验结果（见 Table 2）
| Variant                        | EM   | F1   | ROUGE-L | pass@1 | Tokens/s | KV Mem (GB) |
|-------------------------------|------|------|---------|--------|----------|-------------|
| **FreqDepthKV**                | 58.3 | 63.0 | 32.5    | 48.1   | 70.4     | 6.2         |
| w/o depth-frequency factorization | 56.9 | 61.3 | 31.5    | 46.0   | 66.1     | 6.7         |
| w/o sparse residuals           | 56.7 | 61.1 | 31.4    | 45.8   | 72.6     | 5.8         |
| w/o online head routing        | 57.2 | 61.7 | 31.8    | 46.6   | 69.1     | 6.1         |
| w/o reconstruction-aware loss  | 57.4 | 62.0 | 31.9    | 46.9   | 69.8     | 6.0         |
| w/o exact mode                 | 57.6 | 62.2 | 32.0    | 47.0   | 71.3     | 5.9         |
| shared-depth only              | 56.1 | 60.4 | 31.1    | 45.1   | 73.0     | 5.6         |

> 🔍 **关键发现**：
> - 移除 **depth-frequency factorization** 导致最大性能下降，验证频域分解的有效性。
> - **Sparse residuals** 对 retrieval 和 code 任务至关重要。
> - **Online routing + reconstruction-aware loss** 是实现自适应压缩的核心机制。
> - **Exact mode** 虽少用，但在极端 case 中不可或缺。

---

## 4. 关键结论和发现

### 主要发现
- **Inter-layer redundancy 是 frequency-structured 的**：不能简单视为均匀冗余，而应区分低频共享与高频残差。
- **Preserving high-frequency residuals is critical**：即使只占极小比例，这些稀疏残差对 retrieval 和 reasoning 任务的最终输出具有决定性作用。
- **Compression must be reconstruction-aware**：以是否改变 attention logits 为判据，比单纯依赖 token saliency 更可靠。
- **Adaptive per-head routing works**：不同 head 对压缩敏感度差异大，动态分配模式可在压缩率与准确性之间取得最优平衡。

### 方法的局限性
- **Routing 固定于 prefill 阶段**：解码过程中无法根据新生成内容动态调整 cache 模式。
- **未探索训练时适配**：当前为纯 inference-time 方法，若在训练阶段引入类似机制可能进一步提升 compressibility。
- **Block size 设计仍需经验调参**：B=2/4/8 的选择影响性能，尚未完全自动化。

### 未来工作方向
1. **Generation-Aware Routing**：允许在 decoding 过程中动态切换 shared/residual/exact 模式，响应新出现的依赖关系。
2. **联合优化 depth-frequency 与 precision**：将 FreqDepthKV 与 KVQuant/KIVI 结合，对不同组件采用不同 bit-width，实现端到端带宽优化。
3. **Training-Time Adaptation**：设计可训练模块使模型主动学习更易压缩的 depth-wise cache 结构。
4. **Domain-Specialized Variants**：针对表格处理、多模态输入、multi-agent 推理等场景定制化 depth-frequency 压缩策略。

--- 

> 📌 **总结**：FreqDepthKV 提出了一种新颖且高效的 KV Cache 压缩范式——通过**频域分解 + 动态路由**，在大幅降低内存与延迟的同时，**最大程度保留影响推理的关键信息**，为长上下文 LLM 高效部署提供了实用且鲁棒的解决方案。

</details>

---

### 2. [FourTune: Towards Fully 4-Bit Efficient Post-Training for Diffusion Models](https://arxiv.org/abs/2607.05711)

**Authors**: Bowen Xue, Zihan Min, Xingyang Li, Zhekai Zhang, Haocheng Xi, Lvmin Zhang, Maneesh Agrawala, Jun-Yan Zhu, Song Han, Yujun Lin, Muyang Li  
**Category**: cs.LG  
**Published**: 2026-07-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.05711v1  

#### Abstract
Diffusion models have become a dominant paradigm for high-quality generative modeling, while post-training is essential for adapting them to diverse downstream applications. However, post-training of large diffusion models is still challenging due to the prohibitive memory footprints and slow traini...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FourTune: Towards Fully 4-Bit Efficient Post-Training for Diffusion Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型扩散模型（Diffusion Models）在下游任务中的**后训练**（post-training）——如定制化（Customization）、强化学习（Reinforcement Learning）和蒸馏（Distillation）——面临严重的**内存占用高**和**训练速度慢**的问题。尽管已有参数高效微调方法（如 LoRA 和 QLoRA），但它们仍受限于“内存-速度权衡”：
- **LoRA** 减少了可训练参数，但保留全精度权重，内存开销大；
- **QLoRA** 通过 4-bit 权重量化降低内存，但前向/反向传播中需频繁反量化，计算开销增加。

因此，如何实现**真正高效的端到端低比特后训练**是一个未被充分解决的关键挑战。

---

### 🚀 提出的新方法：FourTune
FourTune 是首个为大型生成模型设计的**完全 4-bit 后训练框架**，采用 **W4A4G4 范式**（Weights, Activations, Gradients 均为 4-bit）。其核心创新包括：

#### （1）三支路混合精度流水线（Triple-Branch Hybrid-Precision Pipeline）
- 在标准 LoRA 架构基础上引入一个**冻结的全精度数值稳定器**（Numerical Stabilizer），用于捕获对量化敏感的权重异常值（outliers）。
- 主干权重经 SVD 分解后，将残差部分以 4-bit（NVFP4）存储并参与计算，确保算术效率。
- 可训练分支仍为 LoRA，保证任务适配能力。

> 公式表达：
> $$
> Y = \text{GEMM}_{4\text{bit}}(Q(X), Q(R)) + X \cdot L_{\text{stab}} + X \cdot (AB)
> $$

#### （2）块级量化支持高效反向传播（Block-wise Quantization）
- 针对反向传播中转置矩阵乘法（$G \cdot W^T$）导致的量化尺度错位问题，提出**块级量化策略**（16×16 block granularity）。
- 使得 4-bit 权重可在无需反量化的情况下直接用于反向计算，避免额外开销。

#### （3）定制化的融合内核（Customized Fused Kernels）
- **LoRA Fusion**：合并 LoRA 下投影、量化与主干 GEMM 操作，减少内存访问。
- **MLP Fusion**：将 FC1-GEMM 与激活函数及后续量化融合，避免中间张量写入全局内存。
- 显著提升带宽利用率和端到端吞吐量。

---

### 🔍 相比现有方法的优势
| 方法 | 内存效率 | 计算效率 | 是否支持原生 4-bit BP |
|------|----------|-----------|------------------------|
| BF16 LoRA | ❌ 高内存占用 | ❌ 全精度计算 | ❌ |
| QLoRA | ✅ 权重压缩 | ⚠️ 反量化开销 | ❌ |
| **FourTune (Ours)** | ✅✅ 接近 QLoRA | ✅✅ 支持 W4A4G4 原生计算 | ✅ |

> FourTune 成功打破了“内存节省 vs. 训练加速”的权衡，实现了**既省内存又快**的后训练方案。

---

## 2. 核心实验方法和设置

### 📚 数据集
实验覆盖三大典型后训练任务，使用以下基准或自建数据集：

| 任务 | 数据集 | 描述 |
|------|--------|------|
| **Customization** | 自定义混合数据集（基于 Custom Diffusion + Web 收集） | 包含三类子任务：<br>- **Human Identity**（人脸身份保持）<br>- **Artistic Style**（艺术风格迁移）<br>- **General Subject**（通用对象一致性生成） |
| **Reinforcement Learning** | HPDv2 | 使用偏好标注图像进行人类审美对齐训练 |
| **Distillation** | COCO-10k, HPSv2 prompts | 无数据蒸馏场景，从教师模型（FLUX.1-dev）蒸馏至 4-NFE 学生模型 |

---

### ⚙️ 实验设置
- **模型架构**：
  - 主要测试模型：**FLUX.1-dev (12B)** 和 **Qwen-Image (20B)**（均为 DiT 架构）
  - 泛化性验证：SDXL（传统 Latent Diffusion）
- **硬件平台**：
  - NVIDIA RTX 5090 / RTX Pro 6000（Blackwell 架构，支持 4-bit Tensor Cores）
  - RTX 4090（Ada 架构，用于 INT4 测试）
- **LoRA 设置**：
  - Rank = 64，统一超参配置
- **量化格式**：
  - 主要使用 **NVFP4**，也测试了 **INT4**

---

### 📊 评估指标

| 任务 | 主要指标 |
|------|---------|
| **Customization** | - **Similarity**（AntelopeV2, CLIP, DINOv3）<br>- **Image Quality**（PyIQA）<br>- **Diversity**（DINO 嵌入距离均值）<br>- **Prompt Following**（BLIP 文图对齐） |
| **Reinforcement Learning** | - **Aesthetic Score v2.5**<br>- **PickScore**, **ImageReward (IR)**, **SGP-HPS** |
| **Distillation** | - **FID**（Teacher alignment）<br>- **CLIP Score**（Prompt alignment）<br>- **HPSv2.1**（Preference alignment） |
| **效率评估** | - GPU 内存占用（GB）<br>- 单步训练延迟（ms/s）<br>- 加速比（Speedup） |

---

### 🆚 基线方法对比
- **BF16 LoRA**：全精度 LoRA，性能上限参考
- **FP8 LoRA**：8-bit 全量化 LoRA
- **NF4 QLoRA**：主流 4-bit 权重量化方法（W4A16G16）
- **SparseLoRA**：稀疏化 LoRA 对照组

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **内存缩减** | 较 BF16 LoRA 减少 **2.25×**（从 29.35GB → 14.51GB） |
| **训练加速** | 单步训练时间从 1.95s → **0.86s**（RTX 5090 上）<br>相较 BF16 LoRA 提升 **2.27×**，相较 NF4 QLoRA 提升 **2.79×** |
| **峰值算力利用** | DiT 组件单步延迟降至 **612.4ms**（较 BF16 LoRA 快 2.52×） |

---

### 📊 定量结果对比（代表性表格摘要）

#### 表1：Customization 性能对比（FLUX.1-dev）
| 方法 | Similarity ↑ | Image Quality ↑ | Prompt Following ↑ |
|------|---------------|------------------|---------------------|
| BF16 LoRA | 0.771 | 33.49 | 0.941 |
| NF4 QLoRA | 0.780 | 32.27 | 0.929 |
| **FourTune (Ours)** | **0.783** | **34.77** | **0.913** |

> ✅ 四项任务上均匹配甚至略优于全精度 LoRA

#### 表2：Distillation 性能对比（4-step student）
| 方法 | FID ↓ | CLIP ↑ | HPSv2.1 ↑ |
|------|-------|--------|-----------|
| BF16 LoRA | 15.50 | 0.288 | 0.316 |
| NF4 QLoRA | 15.51 | 0.287 | 0.310 |
| **FourTune (Ours)** | **15.50** | **0.283** | **0.317** |

> ✅ 生成质量与全精度基线相当，显著优于 SparseLoRA（FID > 200）

#### 表3：Reinforcement Learning 结果
| 方法 | Aes ↑ | PickScore ↑ | IR ↑ | SGP-HPS ↑ |
|------|-------|-------------|------|----------|
| BF16 Finetune | 6.3447 | 0.2316 | 1.0209 | 0.0029 |
| NF4 QLoRA | 6.2189 | 0.2318 | 1.0931 | 0.0020 |
| **FourTune (Ours)** | **6.3119** | **0.2308** | **1.0152** | **0.0027** |

> ✅ 性能媲美全参数微调，说明 4-bit 训练仍能维持细粒度梯度更新的有效性

---

### 🔍 消融实验结果

#### （1）不同精度配置的影响（W4A4G4 vs 其他组合）
| 配置 | Aes ↑ | PickScore ↑ |
|------|-------|-------------|
| W4A16G16 | 6.0816 | 0.2352 |
| W4A4G16 | 6.0308 | 0.2357 |
| W4A16G4 | 6.1386 | 0.2354 |
| **W4A4G4 (Ours)** | **6.1779** | **0.2353** |

> ✅ 所有配置性能相近，证明 **仅 W4A4G4 可充分利用 4-bit 加速优势**

#### （2）Stabilizer 的作用
- 图8显示：无 Stabilizer 时梯度范数迅速爆炸；加入后梯度稳定收敛。
> ✅ 数值稳定器有效防止训练崩溃

#### （3）量化粒度影响（Block-wise vs Group-wise）
| 策略 | Granularity | LPIPS ↓ | PSNR ↑ |
|------|------------|---------|--------|
| group-wise | 1×16 | 0.203 | 21.5 |
| **block-wise (Ours)** | **16×16** | **0.227** | **20.4** |

> ✅ 块级量化虽稍降重建精度，但差距极小，且带来巨大反向效率增益

#### （4）Kernel Fusion 效果分解
| 优化阶段 | 加速比 |
|--------|--------|
| Baseline 4-bit | 1.00× |
| + LoRA Fusion | 1.82× |
| + MLP Fusion | 1.10×（累计 2.00×） |
| **All fused (Ours)** | **2.52×** |

> ✅ 融合优化具有显著叠加效应

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **W4A4G4 是可行的**：首次成功实现扩散模型的**端到端 4-bit 后训练**，且不牺牲生成质量。
2. **打破内存-速度权衡**：FourTune 同时实现**最高内存压缩率**（≈2.25×）和**最快训练速度**（≈2.27×），优于所有现有 PEFT 方法。
3. **数值稳定性可通过架构设计保障**：提出的“冻结稳定器”机制有效隔离量化风险，使极端低比特训练成为可能。
4. **硬件感知优化至关重要**：块级量化 + 融合内核是实现高效 4-bit 反向传播的关键。
5. **泛化性强**：在 FLUX.1、Qwen-Image、SDXL 多种架构，以及 NVFP4/INT4 多种格式下均表现优异。

---

### ⚠️ 局限性
1. **依赖现代硬件**：需要支持 4-bit Tensor Core 的 GPU（如 Blackwell 架构），在旧设备上难以复现全部优势。
2. **当前聚焦 DiT 类模型**：虽然已验证 SDXL 上可用，但在更多非 Transformer 架构上的扩展尚待探索。
3. **SVD 分解引入预处理开销**：虽然离线完成，但仍需额外计算资源进行初始分解。

---

### 🔮 未来工作方向
1. **扩展至其他模态**：如视频生成、音频扩散等更大规模模型。
2. **动态量化策略**：根据训练阶段自适应调整量化粒度或精度。
3. **结合稀疏化**：探索 Sparse + Quantized LoRA 的联合优化空间。
4. **开源工具链建设**：提供易用的 FourTune SDK，推动社区落地应用。

---

## ✅ 总结
**FourTune 是一项里程碑式的工作**，它首次实现了面向大型扩散模型的**全通路 4-bit 后训练框架**，不仅在理论上突破了传统 PEFT 方法的瓶颈，在实践中也展现出卓越的效率与性能平衡。该方法有望大幅降低大模型个性化适配的成本，推动生成模型在消费级设备上的广泛应用。

</details>

---

### 3. [Akashic: A Low-Overhead LLM Inference Service with MemAttention](https://arxiv.org/abs/2607.05708)

**Authors**: Yang Liu, Zhaokai Luo, Huayi Jin, Ruozhou He, Chenchen Hong, Zhiyong Wang, Yifei Liu, Yunfei Gu, Chentao Wu, Junhao Hu  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.05708v1  

#### Abstract
Recent LLM-based agent systems continuously accumulate context across multi-turn interactions, tool invocations, and cross-session workflows. Replaying the full history for every request quickly becomes impractical: long contexts increase prefill cost, may exceed context limits, and often bury task-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Akashic: A Low-Overhead LLM Inference Service with MemAttention**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题

现代基于 LLM 的 **agent 系统**在多轮交互、工具调用和跨会话流程中持续积累上下文，导致以下问题：

- **长上下文开销大**：Prefill 阶段计算成本高，可能超出模型上下文窗口限制。
- **信息稀释**：任务相关证据被大量无关历史淹没，影响输出质量。
- **内存管理效率低**：
  - 固定粒度的压缩策略（如全历史或分段摘要）无法适应上下文密度的异质性。
  - 语义相关的记忆项在物理存储中分散，造成 **locality gap**（局部性差距），增加 I/O 开销。

---

### ✅ 提出的新方法与创新

#### （1）**MemAttention**：基于 chunk 的增量式内存维护机制

- 将上下文划分为固定大小的 **chunk**（默认 1024 tokens），以 chunk 为单位进行增量压缩。
- 引入 **cross-chunk inference**：在压缩新 chunk 时，通过 LLM 匹配其与已有 chunk 的元数据（metadata），联合推理并更新相关 chunk，保留跨段落的语义关联。
- **查询感知检索（query-aware retrieval）**：在推理时，由 LLM 动态判断哪些 memory chunk 最相关，而非依赖静态向量相似度。

> ✅ 优势：避免重复处理整个历史，维持信息完整性，同时控制维护开销为 $O(L_{\text{chunk}})$。

#### （2）**硬件-软件协同设计的 Memory Manager**

- **关联感知重定位（association-aware relocation）**：识别常被共检的 chunk，并将其物理上 **co-locate** 到同一存储块，减少读取碎片化。
- **异地更新 + 后台垃圾回收（out-of-place update & GC）**：
  - 更新时不覆写原块，而是写入新位置，旧版本标记为无效。
  - 当块的无效比例 $\theta \geq 0.75$ 时触发后台 compact，重组有效 chunk 并优化布局。

> ✅ 优势：缓解 locality gap，降低冷缓存下的 p95 检索延迟，提升并发服务能力。

#### （3）系统级实现：构建 **Akashic** 推理服务

- 在 vLLM 基础上扩展控制路径，集成上述机制，不影响 GPU 解码主循环。
- 实现逻辑与物理分离：语义维护由 session 控制，物理布局由访问模式驱动。

---

### ✅ 相比现有方法的优势

| 方法 | 缺陷 | Akashic 改进 |
|------|------|-------------|
| **Full-context** | 上下文过长，prefill 成本爆炸 | 显著缩短 prompt，避免长上下文瓶颈 |
| **Mem0**（全历史摘要） | 更新成本随历史增长；混合无关主题引入噪声 | 按 chunk 维护，成本可控；支持去噪与修正 |
| **MemGAS**（分段独立摘要） | 跨段依赖断裂，丢失长期推理能力 | cross-chunk inference 恢复语义关联 |
| **MemGPT/RMM** | 忽视物理存储布局，检索 I/O 高 | 协同优化语义与物理局部性 |

> 🔥 **核心优势总结**：
> - **高效性**：chunk 级维护使每次操作开销有界。
> - **高质量**：cross-chunk 推理保留关键证据。
> - **低延迟**：co-location 减少磁盘随机读。
> - **可扩展**：适合高并发场景。

---

## 2. **核心实验方法和设置**

### ✅ 数据集（覆盖多样 workload 特征）

| 数据集 | 类型 | 特点 |
|-------|------|------|
| **LoCoMo** | 多会话长程对话 | 时间因果推理强，信息密度低（易压缩） |
| **SWE-bench** | 软件工程任务 | 代码调试为主，信息密度高（难压缩） |
| **BrowseComp** | 网页浏览任务 | 行为突发性强，密度波动大 |
| **WebArena** | 多步网页代理任务 | 真实复杂交互，语义异构 |

---

### ✅ 实验设置

- **模型规模**：
  - Qwen-8B
  - OPT-30B
  - Llama-70B（双卡 H800）
- **硬件配置**：
  - GPU：NVIDIA H800（单/双卡）
  - 存储：4TB NVMe SSD，PCIe Gen5×16
- **框架基础**：基于 vLLM v0.10.0 扩展
- **批大小**：基本测试 batch=1；并发测试逐步提高请求率

---

### ✅ 评估指标

| 指标 | 定义 |
|------|------|
| **Accuracy** | 各 benchmark 的标准指标：<br>- LoCoMo: QA F1<br>- SWE-bench: %Resolved<br>- BrowseComp/WebArena: Accuracy / Success Rate |
| **Throughput (token/s)** | 每秒生成 token 数量 |
| **Sustainable Request Rate (req/s)** | 在延迟不急剧上升前提下的最大可持续请求速率 |
| **p95 Retrieval Latency** | 冷缓存下 95% 分位的检索延迟 |
| **Blocks per Request** | 每次检索触达的物理块数（衡量 I/O 效率） |
| **Space/Write Amplification** | 存储开销放大倍数 |

---

### ✅ 基线方法对比

| 基线 | 类型 |
|------|------|
| **Full-context** | 不使用外部 memory，直接拼接全部历史 |
| **Mem0** | 全历史递归摘要 |
| **MemGAS** | 分段独立摘要，多粒度选择 |
| **MemGPT** | 层次化虚拟内存管理 |
| **RMM** | 反思式内存精炼，多阶段优化 |

---

## 3. **主要实验结果和性能指标**

### ✅ 总体性能表现（Figure 8）

Akashic 在所有设置下均位于 **Pareto 前沿**，兼顾准确率与吞吐。

| 工作负载 | 相比最强 baseline 的提升 |
|--------|--------------------------|
| **LoCoMo / BrowseComp** | ↑ **8.4–10.2 pts accuracy**, ↑ **1.21× throughput** |
| **SWE-bench** | ↑ **4.6–6.6 pts resolved rate**, ↑ **1.33× throughput** vs Mem0 |

> 💡 即便在高密度场景（SWE-bench），仍能保持 90.5%+ 原始吞吐，且准确率更高。

---

### ✅ 并发场景下的可持续请求率（Figure 9）

| 场景 | 请求率提升（vs baseline） |
|------|----------------------------|
| **LoCoMo** | ↑ **1.88×** vs Mem0, ↑ **6.64×** vs Full-context |
| **WebArena** | ↑ **1.54×** vs Mem0 |
| **SWE-bench** | ↑ **1.88×** vs Mem0（尽管压缩空间小） |

> 📌 结论：Akashic 更好地应对并发压力，延迟曲线“拐点”最靠右。

---

### ✅ 存储局部性优化效果（Figure 10）

| 指标 | Akashic vs Append-only / Semantic-only |
|------|----------------------------------------|
| **Blocks/request** | ↓ 29.3%~50.0%（LoCoMo 从 6.8→3.4） |
| **p95 检索延迟** | ↓ 28.0%~42.1%（LoCoMo 从 38ms→22ms） |
| **空间放大（space amplification）** | 仅 1.17×，显著低于 aggressive relocation（1.38×） |
| **写放大（write amplification）** | 1.34×，优于 aggressive relocation（1.82×） |

> ✅ 实现了 **高性能与低存储开销的平衡**。

---

### ✅ 消融实验（Ablation Study, Figure 11）

| 组件移除 | 影响 |
|--------|------|
| **Bounded chunk maintenance**（改为全历史） | 吞吐下降至 **83.4%~88.9%**，是最大性能瓶颈 |
| **Cross-chunk reconciliation**（无联合推理） | 准确率下降 **3.6–4.1 pts**，说明对证据恢复至关重要 |
| **Model-driven relevance matching**（改用 dense embedding） | 性能接近但略逊，验证 LLM 匹配更精准 |

> 🔍 参数敏感性分析显示：
> - 最佳 `chunk_gate` ≈ 1024 tokens
> - `memory_budget` 在 1536 tokens 达到饱和
> - `gc_invalid_ratio=0.75` 是空间与写入的合理折衷

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **固定压缩策略不适用于异质上下文密度**：
   - LoCoMo 易压缩，SWE-bench 难压缩，统一策略难以普适。
   - Akashic 的 chunk 级动态维护更具鲁棒性。

2. **语义相关 ≠ 物理相邻**：
   - 即便检索内容少，若分散在多个 block，I/O 成本仍高。
   - **locality gap** 是制约高并发性能的关键因素。

3. **cross-chunk inference 显著提升推理连贯性**：
   - 修复冲突记忆、合并分散线索，提升任务成功率。

4. **软硬协同设计带来系统级收益**：
   - 逻辑（chunk 维护）与物理（co-location + GC）共同优化，实现端到端高效。

---

### ⚠️ 方法的局限性

- **依赖 LLM 进行语义匹配**：引入额外 inference 开销，对小模型或边缘设备可能不友好。
- **chunk size 需经验设定**：虽有默认值，但在极端场景下需调优。
- **当前基于 SSD**：未探索更复杂的存储层级（如 HBM + NVMe 混合）。

---

### 🔮 未来工作方向

- **自适应 chunk 划分**：根据内容密度动态调整 chunk 大小。
- **轻量化 relevance matching**：训练小型 matcher 模型替代 full LLM。
- **支持 streaming 写入与实时 compact**：进一步提升动态负载下的稳定性。
- **开放源码计划**：已宣布将代码合并至 **RedKnot** 框架（GitHub: [rednote-machine-learning/RedKnot](https://github.com/rednote-machine-learning/RedKnot)）。

---

## ✅ 总结

| 维度 | Akashic 的突破 |
|------|----------------|
| **思想创新** | 提出 **MemAttention**，实现 chunk 级增量维护 + cross-chunk 语义融合 |
| **系统设计** | 软硬协同优化，首次将 **locality-aware storage** 引入 agent memory 管理 |
| **性能表现** | 在准确性、吞吐、并发能力上全面超越主流 baseline |
| **实际意义** | 为构建高效、稳定、可扩展的 LLM agent 推理服务提供新范式 |

> 🔚 **一句话总结**：  
> **Akashic 通过 MemAttention 和协同内存管理，在不牺牲推理质量的前提下，实现了低开销、高并发、高准确率的长上下文 LLM 推理服务。**

</details>

---

### 4. [Bounded-Memory Parallel Image Pulling for Large Container Images](https://arxiv.org/abs/2607.05596)

**Authors**: Sri Saran Balaji Vellore Rajakumar, Henry Wang, Ankur Singh, James Thompson  
**Category**: cs.DC  
**Published**: 2026-07-08  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.05596v1  

#### Abstract
AI/ML workloads increasingly run as containers, where a container image must be downloaded to the host before the workload can start. This cold image pull lands on the critical path whenever a training or inference job scales up or a host is updated, and for GPU workloads it has become the dominant ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Bounded-Memory Parallel Image Pulling for Large Container Images*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 AI/ML 工作负载通常以大型容器镜像（31–48 GiB 压缩后）的形式部署在 GPU 节点上。**冷启动拉取（cold image pull）** 是容器启动的关键路径，而当前主流容器运行时 `containerd` 在并行下载镜像层时采用 **in-memory ordered reassembly（内存中有序重组）** 策略，导致大量已下载但未按序处理的 chunk 积压在内存中。

这一设计在内存受限的 GPU 节点上引发严重问题：
- 内存占用随镜像大小线性增长；
- 可能触发内核 OOM Killer，直接杀死 `containerd` 进程，导致拉取失败。

### 提出了什么新方法或新思路
本文提出 **Disk-Backed Parallel Pull (DBPP)**，一种基于磁盘的并行镜像拉取机制，作为 **SOCI (Seekable OCI) snapshotter** 的一个模式实现。

其核心思想是：
- 将每个下载的 chunk **直接通过 `pwrite` 写入目标文件的对应字节偏移位置**，而非暂存于内存缓冲区；
- 利用本地高速 NVMe 存储作为“缓冲区”，消除对内存中顺序重组的依赖；
- 镜像层在磁盘上成为完整、可寻址的文件后，**并行执行 SHA-256 校验和解压缩**，进一步优化流程。

### 相比现有方法的优势
| 维度 | `containerd` (in-memory) | DBPP |
|------|--------------------------|------|
| **内存占用** | 随镜像大小增长（可达 19.2 GiB） | 恒定低水平（243 MiB – 1.02 GiB） |
| **OOM 风险** | 高，在小内存节点上易被杀 | 极低，可在 7.6 GiB RAM 节点成功拉取 31.4 GiB 镜像 |
| **校验与解压** | 串行：先写完再读回解压校验 | 并行：同时读取压缩数据校验 + 解压流式输出 |
| **兼容性** | 原生支持 | 支持标准 OCI 镜像和 HTTP range requests，无需 registry 修改 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验涵盖五种镜像，构成评估语料库（corpus）：

| 镜像类型 | 名称 | 压缩大小 (GiB) | 层数 | 用途 |
|--------|------|---------------|-----|------|
| 合成（不可压缩） | Synthetic, incompressible | 5.00 | 2 | 隔离纯下载带宽 |
| 合成（多大层） | Synthetic, multi-large-layer | 50.0 | 9 | 测试 intra + inter 层并行 |
| 生产 ML 镜像 | SageMaker Distrib. 3.1-gpu | 7.06 | 24 | 单主导层 |
| 生产 ML 镜像 | NVIDIA NeMo 24.09 | 31.4 | 111 | 多层大镜像 |
| 生产 ML 镜像 | ROCm PyTorch-training v26.1 | 48.5 | 99 | 最大生产镜像 |

### 实验设置和评估指标
- **硬件环境**：
  - 主测试：`m6idn.16xlarge`（64 vCPU, 256 GiB RAM, 本地 NVMe, 100 Gbps 网络）
  - 内存受限测试：`m6id.large`（2 vCPU, 7.6 GiB RAM, 无 swap）
- **控制变量**（见 Table II）：
  - Chunk size: 16 MiB
  - 并发下载数: 40
  - 并发解包数: 50
  - 解压器: `unpigz`（单层单核）
  - Registry: ECR (us-west-2)
  - 客户端: `crictl`
- **评估指标**：
  - **Pull time**: 从开始到完成的墙钟时间（wall-clock time）
  - **△RSS**: 拉取过程中守护进程峰值驻留内存增长
  - **CPU-seconds**: 拉取期间总非空闲 CPU 时间

### 基线方法对比
- **Baseline**: `containerd` 2.2.4 的 in-memory ordered reassembly
- **Target**: DBPP（SOCI snapshotter 0.14.0 实现）
- 两者均启用 **intra-layer parallelism** 和 **parallel unpack**，仅“chunk 重组方式”不同，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 内存占用大幅降低（核心优势）
- **DBPP 内存恒定**：无论镜像大小，峰值 △RSS 保持在 **243 MiB 至 1.02 GiB**。
- **containerd 内存随镜像增长**：从 4.2 GiB（5 GiB 镜像）到 **19.2 GiB**（48.5 GiB 镜像）。
- **内存减少倍数**：**8.7× 至 25.3×**（见 Fig. 2 和 Table VI）。

#### ⏱️ 拉取时间相当（无显著差异）
- DBPP 与 `containerd` 的拉取时间差异在 **±12% 以内**（Table V）：
  - 在较小镜像上 DBPP 更快（如 NeMo 快 9%），
  - 在最大镜像上基本持平（ROCm: 157.7s vs 160.5s）。
- **根本原因**：瓶颈是单线程 `unpigz` 解压，两者共享相同解压逻辑。

#### ❌ 内存受限场景下基线失败，DBPP 成功
- 在 **7.6 GiB RAM 节点** 上拉取 **31.4 GiB NeMo 镜像**：
  - `containerd`: **全部 3 次试验均被 OOM-Killed**（死亡时占用 7.3 GiB 内存）。
  - DBPP: **全部成功完成**，峰值 △RSS 为 3.5 GiB（仍在安全范围内）。

#### 💻 CPU 使用率相当
- 在真实 ML 镜像上，DBPP 因需额外读取压缩层进行校验，CPU 略高，但因与解压并行，总体持平：
  - NeMo: 973.9 vs 958.5 CPU-seconds（DBPP 更低）
  - ROCm: 1401 vs 1336 CPU-seconds
- 在合成不可压缩镜像上，DBPP CPU 高出 2.9×，验证了“额外读取”的开销，但在真实场景中被掩盖。

---

## 4. 关键结论和发现

### 主要发现
1. **内存瓶颈源于不必要的内存缓冲**：`containerd` 的 in-memory 重组策略将网络并行性转化为内存压力，是 OOM 的主因。
2. **NVMe 足够快，可用作内存替代缓冲**：本地 NVMe 写入速度 (~3 GiB/s) 远超单连接吞吐，将重组缓冲移至磁盘不会成为瓶颈。
3. **磁盘布局解锁并行验证**：层文件在磁盘上完整后，可并行执行 SHA-256 校验和解压，抵消了“额外读取”的开销。
4. **DBPP 在资源受限环境下具有决定性优势**：在 GPU 节点等内存紧张场景下，DBPP 是唯一可行的大镜像拉取方案。

### 方法的局限性
- **小镜像收益有限**：小于 1 GiB 的镜像并行收益低，传统顺序拉取更优。
- **稀疏访问不适用**：若容器只访问少量文件，应使用 **lazy loading**（如 SOCI/eStargz），DBPP 面向 **dense-access** 场景。
- **依赖高性能存储**：在 HDD 或低速 SSD 上，随机写入可能成为瓶颈。
- **存储空间临时翻倍**：拉取期间需同时存储压缩层和解压后数据。
- **单层解压仍为串行**：受限于 gzip 格式，单个大层无法并行解压（但 DBPP 为此铺平道路）。

### 未来工作方向
1. **并行解压（Parallel Decompression）**：
   - 利用 DBPP 生成的完整 seekable 文件，集成如 `rapidgzip` 等工具，实现单层内并行解压。
2. **自适应并发控制**：
   - 当前使用静态并发数，未来可根据 registry 反馈动态调整，避免 rate limiting。
3. **扩展至其他系统**：
   - 该“**用磁盘换内存**”范式适用于任何因排序约束而内存缓冲的流水线，如日志收集、数据 shuffle 等（Vector, Ray, Apache Celeborn）。
4. **跨 registry 验证**：
   - 当前实验基于 ECR，未来可在其他 registry（如 Docker Hub, GCR）上验证通用性。

---

> **总结一句话**：  
> DBPP 通过将并行下载的 chunk **直接写入磁盘偏移位置**，彻底解除了内存中的顺序重组依赖，实现了 **内存占用与镜像大小无关** 的高效拉取，在保持性能的同时解决了大模型容器冷启动的 OOM 顽疾。

</details>

---

### 5. [DT-Guard: Intent-Driven Reasoning-Active Training for Reasoning-Free LLM Safety Guardrail](https://arxiv.org/abs/2607.06326)

**Authors**: He Liu, Changtao Miao, Xinjie Yang, Tianle Song, Yin Wu, Junchi Chen, Bintao He, Xinyuan Zhang, Bo Zhang, Shi Yan, Wei Lu, Wei Wang, Danyang Xu, Jiansheng Cai, Zhe Li  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.06326v1  

#### Abstract
Large language models deployed in open-world applications require safety guardrails that are both robust to complex risks and efficient enough for low-latency runtime moderation. Existing guardrails face a practical trade-off between lightweight classification-based models, which are efficient but o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DT-Guard: Intent-Driven Reasoning-Active Training for Reasoning-Free LLM Safety Guardrail

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大语言模型（LLM）在开放世界应用中面临复杂的安全风险，如隐蔽恶意意图、语义模糊请求和越狱攻击。现有的 **safety guardrail** 模型存在以下两难困境：

- **Classification-based guardrails**（如 Llama Guard）推理高效，适合低延迟部署，但在处理隐含意图、边界案例时表现不佳，容易出现**过度拒绝**（over-refusal）或**漏检风险**。
- **Reasoning-based guardrails**（如 GuardReasoner）通过生成 Chain-of-Thought（CoT）提升判断准确性，但引入额外 token 生成开销，导致**推理延迟高**，难以满足工业级实时性要求。

因此，如何在**保持低延迟推理效率的同时，吸收推理过程带来的安全判断优势**，是本论文要解决的核心问题。

---

### 🚀 提出的新方法与创新思路

DT-Guard 提出了 **“Reasoning-Active Training, Reasoning-Free Inference”** 范式，其核心思想是：

> 在训练阶段主动利用显式推理监督（reasoning supervision），而在推理阶段完全不生成推理链，仅输出结构化安全标签。

具体创新点如下：

#### （1）**Intent-Driven 安全判断框架**
将安全判断建模为一个渐进决策流程：
```
Intent → Category → Safety
```
- **Intent**：识别用户交互的真实意图（正常使用 / 风险探索 / 攻击行为）
- **Category**：归因潜在风险类别（如暴力、歧视等）
- **Safety**：最终判定是否安全（Safe / Unsafe / Borderline）

该结构提供了更清晰的中间监督信号，尤其对模糊案例有更强判别能力。

#### （2）**构建意图驱动的安全数据集**
基于多源异构安全数据（如 AdvBench、SafetyBench、CrowS-Pairs 等），通过 LLM 蒸馏 + 多轮投票 + 专家验证的方式，构建了一个包含 **811,897 条样本**的大规模高质量数据集，每条样本均标注了：
- Prompt / Response
- Intent（Normal / Risky / Attack）
- Risk Category（共9类）
- Safety Label（Safe / Unsafe / Borderline）
- Structured CoT 推理轨迹

#### （3）**Rollout-Guided Progressive Hard-Case Optimization (RG-PHO)**
一种分阶段优化策略，利用多 rollout 一致性来识别不同类型的困难样本，并进行针对性优化：

| 样本类型 | 判定方式 | 优化策略 |
|--------|--------|---------|
| Stably Mastered | 所有 rollout 正确 | 忽略，无需再训 |
| Persistently Failed | 所有 rollout 错误 | Hard-Case SFT：加强监督纠正路径 |
| Preference-Unstable | 部分正确、部分错误 | Hard-Case DPO：构建偏好对进行对比学习 |

这种方法实现了从显式推理到隐式判别的知识内化。

---

### 🔍 相比现有方法的优势

| 维度 | DT-Guard | 传统分类模型 | 推理增强型守卫模型 |
|------|----------|--------------|---------------------|
| 推理速度 | ⚡️ 极快（无 CoT 输出） | ⚡️ 快 | 🐢 慢（需生成 CoT） |
| 准确率 | ✅ 高（吸收推理监督） | ❌ 易漏检/误拒 | ✅ 高 |
| 部署友好性 | ✅ 工业可用 | ✅ 可用 | ❌ 延迟敏感场景受限 |
| 对边界案例鲁棒性 | ✅ 强（通过 Intent 分离动机） | ❌ 弱 | ✅ 强 |

> **核心优势**：**以推理训练之智，行分类推断之速** —— 兼顾精度与效率。

---

## 2. 核心实验方法和设置

### 📚 数据集

- **原始来源**：聚合来自六大领域的 1,918,565 条原始样本，包括：
  - Red-teaming & Jailbreak 攻击数据（AdvBench, GCG）
  - 对齐数据（SafeRLHF）
  - 毒性与偏见数据（CrowS-Pairs, Toxicity）
  - 领域特定风险数据
- **最终训练集**：经过蒸馏、多轮投票、专家校验后保留 **811,897 条高质量样本**
  - Prompt-level: 450,437（55.48%）
  - Response-level: 361,460（44.52%）
  - 包含 Safe / Unsafe / Borderline 三类分布合理（见 Table 3）

---

### ⚙️ 实验设置

- **Backbone 模型**：`Qwen3-4B`
- **训练流程三阶段**（RG-PHO）：
  1. **Intent-Guided Mixed-Mode SFT**  
     - 混合使用带 CoT 和不带 CoT 的输出格式
     - 清晰样本用结构化标签，模糊样本保留 CoT
  2. **Failure-Driven Hard-Case SFT**  
     - 对所有 rollout 均失败的样本进行上采样并强化监督
  3. **Rollout-Contrastive Hard-Case DPO**  
     - 对部分正确的样本构造 `(chosen, rejected)` 偏好对，进行对比优化

- **推理模式**：全部采用 **Reasoning-Free Inference**，即只输出 `Intent → Category → Safety` 结构化标签，不生成任何 CoT。

---

### 📊 评估指标与基准方法

#### 评估任务
- **Prompt-side 安全检测**（输入端过滤）：10个 benchmark
- **Response-side 安全审核**（输出端拦截）：7个 benchmark

#### 主要评估指标
- **Classification F1 Score**
- 报告三项平均值：
  - Prompt-side Avg F1
  - Response-side Avg F1
  - Dual-side Avg F1（综合性能）

#### 基线方法对比
| 模型 | 参数量 | 类型 |
|------|-------|------|
| Qwen3Guard-0.8B ~ 8B | 0.8B–8B | 分类式守卫 |
| YuFeng-XGuard-Reason-0.6B ~ 8B | 0.6B–8B | 推理增强型守卫 |
| DT-Guard (ours) | **4B** | **Reasoning-Active Training, Reasoning-Free Inference** |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 5–7）

| 模型 | Prompt F1 | Response F1 | **Dual-side Avg F1** |
|------|-----------|-------------|------------------------|
| Qwen3Guard-8B-Gen | 0.852 | 0.863 | 0.858 |
| YuFeng-XGuard-Reason-8B | 0.849 | 0.841 | 0.845 |
| **DT-Guard (Ours)** | **0.886** | **0.870** | **0.878** |

> ✅ **DT-Guard 以仅 4B 参数超越所有 8B 级别基线模型**

---

### 🔁 与基线方法对比结果

- 在 **prompt-side 平均 F1 上高出 Qwen3Guard-8B 达 3.4 个百分点**（0.886 vs 0.852）
- 在 response-side 同样优于最强基线（0.870 vs 0.863）
- 尤其在 **jailbreak、边界请求、过拒绝控制** 场景下表现显著更好

> 💡 图1显示 DT-Guard 在多个代表性 benchmark 上取得最高 F1，实现全面领先。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）Intent 标签的作用
- 加入 Intent 监督后，prompt-side F1 从 0.855 → 0.863（↑0.8）
- 表明建模交互动机能有效提升对相似文本的不同意图区分能力

#### （2）CoT 训练策略的影响（Table 8）
| 策略 | Dual-side Avg F1 |
|------|------------------|
| NoCoT only | 0.856 |
| Full CoT | 0.828（↓严重！） |
| Mixed CoT/noCoT (1:1) | 0.840 |
| **Borderline CoT + Mixed** | **0.860** ✅ |

> ❗ 全量 CoT 训练反而损害性能 → 存在 **train-inference format mismatch**

> ✅ **选择性地在 Borderline 样本上启用 CoT** 是最优设计

#### （3）RG-PHO 各阶段增益（Table 9）
| 阶段 | Avg F1 | 增益 |
|------|--------|------|
| Stage1-SFT-v0 (baseline) | 0.851 | — |
| + Intent Labels | 0.856 | +0.5 |
| + Borderline CoT | 0.860 | +0.9 |
| + Hard-Case SFT | 0.864 | +1.3 |
| **+ Hard-Case DPO** | **0.878** | **+2.7** ✅ |

> ✅ **Rollout-Contrastive DPO 贡献最大增益**，说明偏好学习对稳定选择正确路径至关重要

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Reasoning supervision 可以内化为 label-level discrimination 能力**  
   即使不在推理时输出 CoT，只要在训练中合理利用推理监督，仍可获得媲美甚至超越推理模型的判断质量。

2. **Intent modeling 是提升 guardrail 鲁棒性的关键**  
   显式建模用户意图可有效缓解 over-refusal 和 under-detection，特别是在语义相近但意图不同的请求中。

3. **Selective reasoning supervision 更优**  
   并非所有样本都需要 CoT；仅对 **Borderline / Hard-case** 启用 CoT 训练，既能提供足够监督，又避免格式错配。

4. **Rollout consistency 是有效的 hard-case stratification 工具**  
   多次 rollout 的预测稳定性可用于自动识别三类残差错误，指导后续优化路径。

5. **DT-Guard 实现了精度与效率的帕累托前沿突破**  
   用 4B 模型超越 8B 基线，在工业部署中极具吸引力。

---

### ⚠️ 方法的局限性

1. **依赖高质量 CoT 数据构建**  
   当前数据集依赖 GLM-5.1 进行蒸馏，若初始 CoT 质量不高，可能传播错误逻辑。

2. **Intent 分类体系尚有限**  
   当前仅定义三种 intent（Normal / Risky / Attack），未来可进一步细化（如教育、测试、对抗等子类）。

3. **未支持流式检测（streaming detection）**  
   虽然推理快，但未像 Qwen3Guard 那样实现实时 token 级监控。

4. **Multilingual 支持不足**  
   当前数据集以英文为主，缺乏跨语言泛化能力验证。

---

### 🔮 未来工作方向

1. **扩展至多模态安全守卫**（multimodal guardrail）
2. **结合 streaming + structured output 实现毫秒级响应**
3. **自动化 rollout-guided pipeline**，减少人工干预
4. **构建开源 intent-driven 安全 benchmark**
5. **探索 self-improving guardrail：利用自身判断反馈持续迭代**

---

## 总结

📌 **一句话总结**：  
**DT-Guard 成功将“推理增强”的优势“内化”于轻量级分类模型之中，实现了“训练时动脑、推理时不啰嗦”的新一代高效安全守卫范式。**

🎯 **关键词提炼**：  
`Reasoning-Active Training`, `Reasoning-Free Inference`, `Intent Modeling`, `Structured Decision Path`, `Rollout-Guided Hard-Case Optimization`, `F1 = 0.878 @ 4B`

🔧 **适用场景**：  
适用于需要**高吞吐、低延迟、强鲁棒性**的内容安全系统，如客服机器人、社交平台审核、金融对话助手等工业级 LLM 应用。

</details>

---

### 6. [CurateEvo: Data-Curation Evolving for Agentic Post-Training](https://arxiv.org/abs/2607.06140)

**Authors**: Dingzirui Wang, Xuanliang Zhang, Keyan Xu, Qingfu Zhu, Wanxiang Che  
**Category**: cs.CL  
**Published**: 2026-07-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.06140v1  

#### Abstract
Large language model (LLM) agents require post-training methods that can improve long-horizon decision making from environment feedback. However, existing agentic post-training pipelines often treat data curation as a fixed preprocessing step, focusing mainly on data augmentation while neglecting fi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**CURATEEVO: Data-Curation Evolving for Agentic Post-Training**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现有的 **agentic post-training** 数据处理方法通常将 **data curation** 视为一个静态、固定的预处理步骤，存在两大局限：

- **Curation Neglect（忽略综合处理）**：大多数方法只关注 **data augmentation**（如生成更多失败轨迹），而忽略了 **filtering**（过滤低质量样本）和 **refinement**（精炼弱监督信号）等关键操作，导致训练数据冗余且效率低下。
- **Adaptation Rigidity（适应性差）**：固定的数据处理流水线难以动态适应下游任务中的新失败模式或分布外（out-of-distribution）场景。

这限制了 LLM Agent 在长程决策、稀疏反馈环境下的持续优化能力。

---

### 🚀 提出的新方法与创新思路

作者提出 **CURATEEVO** —— 一种**基于失败驱动的动态演化框架**，用于 **agentic post-training** 中的数据治理。

#### 核心思想：
- 将 **data curation 策略** 表示为可执行代码（executable code）。
- 利用在 **held-out 开发集上失败的交互轨迹（failed trajectories）** 作为反馈信号。
- 使用一个 **LLM-based code-evolution agent** 迭代地重写该 curation 代码，从而实现策略的“进化”。

#### 演化过程分为两个阶段：
1. **Effectiveness Optimization（有效性优化）**  
   分析失败模式（failure modes），通过 **augment / filter / refine** 数据来弥补模型的能力缺陷。
2. **Efficiency Optimization（效率优化）**  
   在保持性能的前提下，**剪枝冗余、低效或过长的训练轮次（turns）**，降低训练成本。

最终，同一份原始语料库 `Draw` 被转化为三种资源：
- **SFT Dataset**（监督微调数据）
- **RL Dataset**（强化学习数据）
- **Inference-time Memory Bank**（推理时记忆库）

---

### 🔍 相比现有方法的优势

| 维度 | CURATEEVO | 传统方法 |
|------|----------|---------|
| **灵活性** | 动态演化 curation 策略 | 固定 pipeline |
| **全面性** | 同时支持 augment/filter/refine | 多数仅做 augmentation |
| **目标导向** | 以失败模式为指导，精准补强 | 通用扩展，可能引入噪声 |
| **效率意识** | 显式优化训练规模（cost-aware） | 忽视训练开销 |
| **兼容性** | 可结合不同 post-training recipe（如 GRPO, AgentGym-RL） | 往往绑定特定流程 |

> ✅ CURATEEVO 是首个将 **data curation 本身作为可优化对象** 的框架，实现了从“被动清洗”到“主动进化”的转变。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

实验在两种数据设定下进行：

| 类型 | 数据来源 | 特点 |
|------|--------|------|
| **Labeled Data** | SWE-chat, AgentRewardBench, OpenHands-Feedback | 人工标注轨迹，干净可靠 |
| **Wild Data** | ASSERT-KTH/reproducible-trajectories, Claudeset-community 等 | 真实用户交互日志，含噪声、不完整 |

原始语料按 **9:1** 划分训练/开发集，开发集用于演化 curation 策略，测试集完全隔离。

---

### 🧪 实验设置与评估指标

#### 模型基础
- **Base Model**: QWEN3-4B / QWEN3-8B
- **Post-training Recipe**: 固定使用 **SFT + GRPO**（Generalized Reward Policy Optimization）

#### 评估基准（held-out test sets）
| Benchmark | 描述 |
|---------|------|
| **ACEBench-Agent** | 工具选择、参数填充、多轮交互能力 |
| **BFCL-V4** | 函数调用准确性、抗幻觉、格式遵循 |
| **T2-Bench** | 双方控制环境下的长程协作与状态追踪 |

#### 主要指标
- **平均得分（Average Score）**：三个 benchmark 的均值
- **Dev Performance**：开发集上的成功率（Success Rate）
- **Training Turn Cost**：保留的交互轮次数（衡量效率）
- **Curation Overhead**：token 数量与运行时间（评估构建成本）

---

### ⚔️ 对比的基线方法

| 基线 | 方法简介 |
|-----|--------|
| **GRPO w/o curation** | 不进行任何数据治理，直接训练 |
| **MUA-RL** | 多轮用户交互强化学习 |
| **EnvScaler** | 扩展仿真环境以生成多样化轨迹 |
| **AWM** | 构建大规模合成工具环境 |
| **RODS** | 奖励驱动的在线数据合成 |
| **FunReason-MT** | 高级离线多轮函数调用数据生成 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| 方法 | Labeled Data (Avg.) | Wild Data (Avg.) |
|------|---------------------|------------------|
| 最优基线（Prior Best） | ~53.9 | ~49.0 |
| **CURATEEVO (QWEN3-4B)** | **57.0** | **51.0** |
| **提升幅度** | **+3.2 pts** | **+2.7 pts** |

> 💡 即使使用更小的 QWEN3-4B，CURATEEVO 仍优于多数基于 QWEN3-8B 的方法，说明**高质量数据治理可媲美模型扩容**。

---

### 🔁 与基线方法的对比结果

- 在所有 benchmark 上均取得 **SOTA 性能**，尤其在 **T2-Bench**（长程交互）表现突出，表明其对复杂决策有更强支持。
- 相比纯数据扩增类方法（如 RODS, FunReason-MT），CURATEEVO 更擅长识别并修复具体失败模式。
- 在 **wild data 设置下优势更明显**，说明其具备强大的噪声容忍与有用信号提取能力。

---

### 🔍 消融实验结果（Table 2）

| 消融条件 | 平均性能下降 |
|--------|------------|
| **-Effectiveness**（移除有效性优化） | ↓ ~7.0 pts |
| **-Efficiency**（移除效率优化） | ↓ ~1.5 pts |
| **-SFT Data** | ↓ ~5.0 pts |
| **-RL Data** | ↓ ~7.0 pts |
| **-Memory** | ↓ ~3.5 pts |

> ✅ 结果验证：
> - **有效性优化是性能提升主因**（诊断失败 + 针对性补强）
> - **效率优化虽小幅降分，但显著减少冗余训练**
> - 三类输出资源（SFT, RL, Memory）互补，缺一不可

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **数据治理应是动态闭环过程**  
   将 curation 策略表示为可演化的代码，使其能根据实际失败不断调整，比静态规则更有效。

2. **失败轨迹是宝贵反馈信号**  
   通过对失败路径的分析，可以精准定位能力缺口（如错误恢复、状态绑定漂移），进而指导数据增强方向。

3. **质量 > 数量，密度 > 规模**  
   CURATEEVO 构建的训练集更小但更“浓缩”，聚焦于高价值决策点，避免被大量无意义交互淹没。

4. **兼容性强，可作模块嵌入现有系统**  
   演化后的数据可用于多种 post-training 方法（GRPO, AgentGym-RL, ProRL-Agent），平均再提升 **21.3 pts**。

5. **大幅降低 curation 开销**  
   相比依赖模拟环境或额外交互的方法，CURATEEVO 在每千条训练样本上节省约 **48% token 成本** 和 **50% 时间开销**。

---

### ⚠️ 方法的局限性

- **依赖高质量失败分析**：若 LLM 对失败轨迹理解不准，可能导致错误演化。
- **初始 curation code 设计影响收敛速度**：po 的设计仍需一定先验知识。
- **未探索多 agent 协同演化**：当前仅优化单一 curation agent。
- **计算资源需求较高**：需多次训练与评估以完成演化循环。

---

### 🔮 未来工作方向

1. **自动化 failure diagnosis 模块**：减少对 LLM 推理的依赖，提高稳定性。
2. **跨任务迁移 curation 策略**：研究通用型可迁移的 curation code template。
3. **引入 human-in-the-loop 验证机制**：在关键演化节点加入人工审核。
4. **结合 online learning**：将演化过程与在线部署联动，实现持续自适应。
5. **扩展至 multimodal agents**：应用于视觉、机器人等具身智能场景。

---

## ✅ 总结

**CURATEEVO** 提出了一种全新的视角：**把 data curation 本身当作一个可学习、可进化的程序**。它通过失败反馈驱动 curation code 的迭代演化，在提升 LLM Agent 决策能力的同时，显著降低了训练成本。其实验充分证明了“**智能地选数据”比“盲目多地加数据”更重要**，为未来的 agentic AI 训练范式提供了重要启示。

</details>

---

### 7. [Multimodal Molecular Representation Learning with Graph Neural Networks, Deep & Cross Networks, and SMILES Embeddings](https://arxiv.org/abs/2607.05736)

**Authors**: Qiwei Han, Chi Zhou, Ruobing Wang, Zheng Ma  
**Category**: cs.LG  
**Published**: 2026-07-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.05736v1  

#### Abstract
Molecular property prediction often relies on isolated data modalities, where continuous 3D graph neural networks (GNNs) struggle to efficiently capture long-range topological dependencies and exact macroscopic heuristics. In this work, we introduce a parameter-efficient Tri-Branch Modular Fusion Ne...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Multimodal Molecular Representation Learning with Graph Neural Networks, Deep & Cross Networks, and SMILES Embeddings

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

分子性质预测（如原子化能）通常依赖单一模态数据（如仅3D几何结构），而现有的 **Graph Neural Networks**（GNNs）在处理长程拓扑依赖性和精确宏观启发式规则时存在局限性：
- **局部消息传递机制** 容易导致图过平滑（oversmoothing），难以捕捉全局计数任务（如原子总数、键类型统计）。
- 单一几何模型需要大量参数才能逼近高精度，计算开销大，不利于 **High-Throughput Virtual Screening**（HTVS）等实际应用场景。

### **提出了什么新方法或新思路**

提出了一种**参数高效**的三分支模块化融合神经网络（**Tri-Branch Modular Fusion Neural Network**），通过融合三种正交模态来构建更鲁棒的分子表示：

| 模态 | 技术实现 | 功能定位 |
|------|--------|--------|
| **3D空间几何** | SchNet + 全局加性池化（additive pooling） | 捕捉局部量子微环境 |
| **离散拓扑语法** | SMILES 字符串 → ChemBERTa → SwiGLU降维 | 捕捉长程功能团拓扑关系 |
| **显式宏观物化描述符** | Deep & Cross Network（DCN）处理RDKit特征 | 注入精确的宏观物理化学先验 |

该框架采用**晚期融合**（late-fusion）架构，在共享潜在空间中整合多源信息，避免早期特征干扰。

### **相比现有方法的优势**

- ✅ **参数效率极高**：总参数量少于100万，适合快速推理。
- ✅ **超越化学精度阈值**（sub-chemical accuracy threshold ~0.0433 eV），达到 **0.0207 eV MAE**。
- ✅ 利用多模态协同效应，提供“物理捷径”（physical shortcut），无需暴力堆叠参数即可突破性能瓶颈。
- ✅ 数学上严谨地适配**广延性质**（extensive property）预测需求（如原子化能），使用**加性池化**而非均值/最大池化。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **QM9 数据集**：包含约13万个小分子（最多9个重原子），所有样本均经过严格清洗以确保跨模态对齐。
  - 排除无法生成有效3D构象、RDKit描述符出错或SMILES无法分词的分子。
  - 最终保留 **129,012** 个结构对齐的分子。
  - 数据划分：**80%训练集，20%验证集**。

### **实验设置和评估指标**

#### **预测目标**
- 原子化能（atomization energy at 0 K, $ U_{\text{atom}} $）——典型的**广延热力学性质**。

#### **输入特征工程**
| 模态 | 特征来源 | 维度 |
|------|--------|-----|
| 几何（Geometric） | 3D坐标 + 原子序数 → SchNet | 可变节点数 |
| 宏观物化（Tabular） | RDKit提取18维描述符 | $ d_{\text{tab}} = 18 $ |
| 语义（Semantic） | SMILES → ChemBERTa-77M-MLM 编码 | $ d_{\text{sem}} = 384 $ |

#### **评估指标**
- **Validation MAE**（Mean Absolute Error）：主评价指标，单位为 eV。
- $ R^2 $：决定系数，衡量拟合优度。
- 所有标签进行 z-score 归一化，推理阶段还原为物理单位。

#### **超参数配置**
| 组件 | 配置 |
|------|------|
| 优化器 | AdamW（Fused） |
| 学习率 | 3e-4，配合 ReduceLROnPlateau（factor=0.5） |
| Batch Size | 64 |
| Loss | L1 Loss（MAE） |
| Early Stopping | 100轮无提升则停止 |

### **基线方法对比**

- **Unimodal Baselines**：
  - SchNet Only（控制变量下的截断版本）
  - DCN Only（仅用RDKit特征）
  - Embedding Only（仅用ChemBERTa输出）

- **Bimodal Combinations**：
  - SchNet + DCN
  - SchNet + SMILES Embedding
  - DCN + SMILES Embedding

- **Ablation Controls**：
  - 替换DCN为标准MLP（相同参数容量）
  - 调整语义瓶颈维度（$ d_e \in \{32,64,128,256\} $）

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 模型配置 | Val MAE (eV) | $ R^2 $ |
|---------|--------------|--------|
| **SchNet Only**（baseline） | 0.0261 | 0.9999 |
| **DCN + Embedding** | 0.1458 | 0.9992 |
| **SchNet + DCN** | 0.0222 | 0.9999 |
| **SchNet + Embedding** | 0.0240 | 0.9999 |
| **Full Tri-Modal**（$ d_e=128 $） | **0.0214** | 0.9999 |
| **Optimized Tri-Modal**（$ d_e=64 $） | **0.0207** ✅ | 0.9999 |

> 🔥 **最佳结果**：**0.0207 eV MAE**，显著低于化学精度阈值（~0.0433 eV）。

### **与基线方法的对比结果**

- 相比纯几何模型（SchNet Only），三模态融合实现了 **20.6% 的误差降低**。
- 尽管SchNet是主导信号源，但引入DCN和SMILES嵌入带来了非线性增益。
- 在没有3D坐标的条件下，DCN + SMILES仍能达到0.1458 eV，说明宏观与拓扑信息本身具有强互补性。

### **消融实验结果**

#### **Phase 1: 多模态组合消融**
- 所有双模态组合均优于任一单模态，证明模态间存在**正交互补性**。
- SchNet + DCN > SchNet + Embedding，表明**显式宏观描述符**更能弥补GNN在全局计数任务上的缺陷。

#### **Phase 2: 语义瓶颈敏感性分析**
| 语义投影维度 $ d_e $ | Val MAE (eV) |
|-----------------------|-------------|
| 32 | 0.0216 |
| **64（最优）** | **0.0207** ✅ |
| 128 | 0.0214 |
| 256 | 0.0220 |

> 结论：**64维是最佳信息瓶颈**，过高会引入语言噪声，过低则损失拓扑细节。

#### **Phase 3: 表格路由架构验证**
| Tabular 架构 | Val MAE (eV) |
|--------------|-------------|
| MLP（隐式交互） | 0.0210 |
| DCN（显式多项式交叉） | **0.0207** ✅ |

> 虽然DCN略优，但差距较小，说明**性能增益主要来自数据本身而非复杂架构**。

---

## 4. 关键结论和发现

### **主要发现**

1. **多模态融合可打破GNN的表达瓶颈**：
   - GNN擅长局部建模但不擅全局计数；DCN提供“物理捷径”解决算术盲区。
   - Transformer-based SMILES 模型通过自注意力实现 $ O(1) $ 长程依赖建模，弥补3D卷积的信息衰减。

2. **正交模态协同产生“1+1+1≫3”的效果**：
   - 三者分别对应：**微观物理**（SchNet）、**宏观经验**（DCN）、**拓扑语法**（SMILES）。
   - 融合后形成数学上一致的潜在空间，特别适用于广延性质预测。

3. **参数效率优于暴力扩参策略**：
   - 少于百万参数即超越主流大规模equivariant模型的精度水平。
   - 为HTVS等资源受限场景提供了实用替代方案。

### **方法的局限性**

1. **几何分支被刻意削弱**：
   - 为了保证模态公平性，SchNet原生读出头被替换为统一128维瓶颈，牺牲了其理论峰值性能。

2. **语义编码器冻结**：
   - 使用预训练且固定的ChemBERTa，未进行端到端微调，限制了其对特定任务（如$ U_{\text{atom}} $）的适应能力。

3. **超参数搜索有限**：
   - 语义瓶颈仅测试四个点，未进行全面NAS搜索，可能存在更优结构。

4. **数据范围受限**：
   - 仅在小分子有机化合物（QM9）上验证，尚未推广至大分子、柔性药物或多相态系统。

### **未来工作方向**

- 引入 **Parameter-Efficient Fine-Tuning**（PEFT）技术（如LoRA）对ChemBERTa进行轻量化微调。
- 将框架扩展至动态轨迹（如MD模拟）和更大规模数据集（如PCQM4Mv2）。
- 探索门控机制或注意力加权融合策略，进一步提升模态协调性。
- 应用于其他广延/强度性质预测任务，验证泛化能力。

---

> 📌 **一句话总结**：  
> 本文提出一种高效的三模态融合框架，通过结合 **SchNet**（几何）、**DCN**（宏观描述符）和 **SMILES Embedding**（拓扑语法），在仅百万参数下实现超高精度分子性质预测，揭示了“多模态协同 > 单一模态扩参”的新范式，为下一代HTVS系统提供了强有力的技术路径。

</details>

---

### 8. [A Physics-Informed Neural Network Framework for Elastodynamic Wave Propagation in Bimaterial Systems](https://arxiv.org/abs/2607.06479)

**Authors**: Sonal Ankush Chibire, Jenn-Terng Gau, Bo Zhang  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.06479v1  

#### Abstract
Physics-informed neural networks (PINNs) provide a promising framework for solving partial differential equations while embedding the underlying physical laws directly into the learning process. This study presents a PINN-based framework for modeling transient elastodynamic wave propagation in bimat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Physics-Informed Neural Network Framework for Elastodynamic Wave Propagation in Bimaterial Systems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该研究针对**异质材料中瞬态弹性动力波传播**（transient elastodynamic wave propagation）的建模难题，特别是在**双材料系统**（bimaterial systems）中，如钢-铝界面处的应力波反射、透射和界面效应。传统有限元方法（FEM）虽然精度高，但在进行参数化分析、优化或反问题求解时计算成本高昂。

### 🚀 提出的新方法与创新点
- **提出了一种基于 Physics-Informed Neural Networks (PINNs)** 的框架，用于求解轴对称线性弹性动力学控制方程（Navier-Lamé equations），并应用于 Split Hopkinson Pressure Bar (SHPB) 类型的双材料系统。
- 将完整的物理约束——包括**初始条件、边界条件和材料界面连续性条件**（如位移连续、正应力连续、剪切应力为零）——直接嵌入到 PINN 的 loss function 中，确保预测结果满足基本力学规律。
- 引入来自 ANSYS Workbench Explicit Dynamics 的高保真 FEM 数据作为软约束（soft data constraints），实现“物理引导 + 数据增强”的混合建模范式。

### ⚙️ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **计算效率** | 一旦训练完成，PINN 可在任意时空点快速推断，无需重新运行耗时的 FEM 模拟，显著降低重复仿真开销。 |
| **连续性建模能力** | 提供光滑、可微的场变量近似（如 $ u_r(r,x,t), u_x(r,x,t) $），支持任意时空插值与外推。 |
| **泛化能力** | 能够准确预测训练时间范围之外（unseen time instants）的动态响应，并适用于修改后的材料属性（无需再训练）。 |
| **鲁棒性与通用性** | 经过 mesh-sensitivity 验证具有数值稳定性；在不同材料组合下表现一致，显示方法普适性。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **高保真有限元模拟数据**：通过 ANSYS Workbench Explicit Dynamics 构建三维四分之一对称模型（quarter-symmetry model），模拟 SHPB 实验中的钢-铝双材料试样在冲击载荷下的动态响应。
- 包含以下输出字段：
  - 轴向与径向位移历史（axial/radial displacement）
  - 应力与应变演化（stress-strain histories）
  - 界面行为（interface transmission/reflection）

> 所有 FEM 设置均与 PINN 的物理假设保持一致（线弹性、各向同性、小变形等）。

### 🔧 实验设置
- **几何配置**：
  - 圆柱形试样，直径 10 mm，总长 2.5 mm（钢段 + 铝段）
  - 入射杆与透射杆长度均为 50 mm
- **材料参数**：
  - 钢：$ E = 2.0 \times 10^5 $ MPa, $ \nu = 0.30 $, $ \rho = 7850 $ kg/m³
  - 铝：$ E = 71000 $ MPa, $ \nu = 0.33 $, $ \rho = 2780 $ kg/m³
- **加载方式**：在入射端施加轴向冲击速度
- **PINN 输入输出**：
  - 输入：空间坐标 $(r, x)$ 和时间 $t$
  - 输出：径向位移 $u_r$ 和轴向位移 $u_x$

### 🎯 评估指标
- **均方误差**（MSE）比较 PINN 与 ANSYS 在关键节点上的位移、应力、应变响应
- **face-averaged displacement** 对比（符合 SHPB 理论假设）
- 时间外推性能测试（200–400 μs 区间）
- 残差收敛情况（PDE residual, BC/IC/Interface loss）

### 🆚 基线方法对比
- 主要对比对象为 **ANSYS Explicit Dynamics** 的 FEM 结果
- 并未与其他 ML 方法（如纯数据驱动 NN 或 DeepONet）直接对比，但强调了 PINN 相较于传统 FEM 在**计算效率与可重用性方面的优势**

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 位移预测精度（Fig. 4）
| 位置 | 指标 | 表现 |
|------|------|------|
| 钢输入面（axial） | 波达时间、峰值位移、卸载过程 | 完全匹配 ANSYS，误差 < 2% |
| 铝输出面（radial） | 动态膨胀/收缩响应 | 整体趋势高度一致，后期略有偏差（多波反射影响） |

#### ✅ 时间外推能力（Fig. 5）
- 在 **200–400 μs**（未参与训练的时间区间）内：
  - 轴向位移预测几乎完全重合
  - 径向位移在铝侧仍保持良好一致性，在钢侧末期出现轻微过估（可能因累积误差）

#### ✅ 截面平均响应（Face-Averaged, Fig. 6）
- 符合 SHPB 实验测量惯例
- PINN 与 FEM 的 face-averaged displacement 曲线高度吻合，验证其工程实用性

#### ✅ 应力-应变响应（Fig. 7）
| 材料 | 表现 |
|------|------|
| 铝 | 主峰应力/应变完全匹配，全过程一致性好 |
| 钢 | 初始主峰捕捉准确，但**后峰值阶段振荡更平滑**，说明对高频反射波敏感度较低 |

> 总体表明：**位移场预测最稳健，而应力/应变作为导数量，受自动微分噪声和残差积累影响稍大。**

### 🔍 消融实验（隐含分析）
尽管未明确列出消融实验表格，文中通过以下设计体现关键组件作用：
- **渐进加权策略**（progressive weighting）：先强化数据项与边界条件，再逐步提升 PDE 残差权重 → 改善收敛稳定性
- **归一化处理**：所有物理量标准化以提高训练数值稳定性
- **周期性重采样**（resampling collocation points）：防止局部过拟合，增强空间覆盖

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **PINN 成功实现了对复杂双材料系统中弹性波传播的高精度建模**，能够准确再现：
   - 波的传播、反射与透射
   - Poisson 耦合引起的径向变形
   - 界面处的连续性条件（位移连续、法向应力连续、无摩擦剪切）
2. **训练后的 PINN 是一个连续的时空代理模型**（continuous surrogate model），可在任意 $(r,x,t)$ 处高效评估响应，极大减少重复 FEM 计算需求。
3. **结合 FEM 数据与物理定律的混合训练策略有效提升了预测准确性与物理一致性**，优于纯数据驱动或纯物理驱动方法。
4. **方法具备良好的泛化能力**，不仅能外推至新时间点，还可推广至其他材料组合（文中提及已验证多种组合）。

### ⚠️ 局限性
- **对高频波动细节捕捉不足**：尤其在多次反射后的后期响应中，PINN 预测趋于平滑，可能低估局部应力集中。
- **依赖高质量 FEM 数据提供初始锚定**：若缺乏可靠参考数据，纯物理驱动训练可能难以收敛。
- **长期外推存在误差累积风险**：尤其在径向响应中观察到偏离趋势。
- **当前仅限线弹性小变形假设**，尚未扩展至塑性、损伤或大变形非线性情形。

### 🔮 未来工作方向
1. **拓展至更多双材料组合**，研究不同声阻抗比（acoustic impedance contrast）对应力波传输的影响。
2. **加强物理约束机制**：探索更强形式的界面条件实施方式（如 Lagrange multiplier 方法）。
3. **自适应配点策略**（adaptive collocation）：在梯度剧烈区域（如波前附近）动态增加采样密度。
4. **改进晚期波反射建模**：引入记忆机制或递归结构以更好捕捉长时间动力学行为。
5. **向三维与非轴对称问题扩展**，并集成实验数据用于真实 SHPB 测试反演分析。

---

## 总结

本论文成功构建了一个融合 **Physics-Informed Neural Networks** 与 **Explicit FEM** 的高效混合框架，用于建模 **bimaterial systems** 中的瞬态弹性波传播。该方法不仅在精度上与高保真 FEM 高度一致，还提供了卓越的计算效率与泛化能力，是面向高速固体力学、冲击工程和 SHPB 实验分析的一种极具前景的 **surrogate modeling** 工具。

</details>

---

### 9. [Beyond Static Evaluation: Building Simulation Environments for Scalable Agentic Reinforcement Learning](https://arxiv.org/abs/2607.05773)

**Authors**: Akshay Arora, Ishan Nigam, Ashutosh Aggarwal, Shefali Bansal, Krishna Singh, Sweta Kumari, Nikhil Mittal, Shariq Farhan, Siddarth Malreddy  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.05773v1  

#### Abstract
As Large Language Models (LLMs) evolve into autonomous agents, traditional static evaluation fails to capture multi-step decision-making. We introduce AgenticAI-Supervisor, an API and UI-driven RL Gym environment that decouples environment creation from scalable execution. By moving to verifiable ex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond Static Evaluation: Building Simulation Environments for Scalable Agentic Reinforcement Learning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统对 Large Language Models（LLMs）的评估依赖于静态、单轮的 benchmark（如 MMLU、GSM8K），这类方法无法有效衡量 LLM 作为**自主智能体（autonomous agent）** 在动态环境中的多步决策能力、工具调用逻辑、错误恢复机制以及长期规划表现。随着 LLM 被部署到企业级任务中（如客服、供应链管理等），这种“可靠性差距”（reliability gap）导致高达 **76% 的复杂专业任务失败**。

此外，当前强化学习（RL）训练面临以下瓶颈：
- 手动构建多步测试用例和边缘场景成本高且易出错；
- 奖励函数设计容易被“奖励黑客”（reward hacking）利用；
- 缺乏可验证的执行轨迹（execution traces）支持闭环优化。

---

### 🚀 提出的新方法与创新思路
作者提出 **AgenticAI-Supervisor** —— 一个基于 API 和 UI 驱动的 **RL Gym-style 模拟环境平台**，旨在实现可扩展、可验证的 Agentic Reinforcement Learning。

#### 核心创新点包括：

| 创新点 | 描述 |
|-------|------|
| **双阶段架构（Dual-phase Framework）** | 将环境搭建（scaffolding）与大规模并行 rollout 执行解耦，提升系统可扩展性和稳定性。 |
| **高保真模拟环境（High-Fidelity Environment Scaffolding）** | 支持模拟真实企业工具链（API、GUI），包含标准操作路径、故障注入、模糊响应等，增强 agent 的鲁棒性测试能力。 |
| **确定性奖励塑形引擎（Deterministic Reward Shaping Engine）** | 引入多维奖励信号：<br>• **Outcome Reward**（结果正确性）<br>• **Constraint Adherence**（约束合规性）<br>• **Trajectory Efficiency Reward**（轨迹效率）<br>避免仅依赖文本启发式判断带来的 reward hacking。 |
| **状态验证机制防止 Reward Hacking** | 通过内部状态变更检测、副作用监控、虚构事实交叉验证等方式，确保 agent 行为符合业务逻辑而非“走捷径”。 |
| **结构化执行轨迹（Execution Traces）驱动调试与学习** | 所有交互事件（LLM 调用、tool call、state change）被记录为结构化 Span，聚合形成完整 trace，用于离线分析与 RL 更新。 |

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | AgenticAI-Supervisor |
|------|--------|------------------------|
| **评估方式** | 静态 prompt-response，主观评分 | 动态仿真 + 可验证 outcome + 多维 reward |
| **环境真实性** | 通用 Web 浏览任务（WebArena）、游戏环境 | 企业定制化 API/GUI 模拟，贴近生产环境 |
| **奖励设计** | LLM-as-a-Judge 或人工标注，易受幻觉影响 | 结合程序化 verifier + LLM judge，兼顾精度与灵活性 |
| **可扩展性** | 单机或小规模 rollout | 并行沙箱容器执行，支持千级并发 rollout |
| **防作弊机制** | 较弱，依赖最终输出匹配 | 内部状态审计、冗余调用惩罚、side-effect 检测 |

---

## 2. 核心实验方法和设置

### 📚 数据集与任务领域
本研究未使用公开 benchmark 数据集，而是构建了一个**自定义的企业级客户支持（Customer Support Agent）模拟环境**，其核心组件如下：

#### 模拟工具分类：
- **Non-Actionable（只读工具）**：
  - `get_customer_info`, `get_order_details`
  - `check_interaction_history`
  - `search_kb_and_policies`（知识库检索）
- **Actionable（可写工具）**：
  - `Refund Tool`, `Replacement Tool`
  - `Lock Account`（安全锁）
  - `create_ticket`, `update_order_status`

#### 场景复杂性设计：
- 必须先查政策再退款（policy compliance）
- 发现异常行为需触发账户锁定（fraud detection）
- 处理重复扣费、订单丢失等复合问题

---

### ⚙️ 实验设置
- **Agent 架构**：基于 LLM 的 agent，在模拟环境中进行多轮交互。
- **执行模式**：Rollout Handler 启动多个隔离的 container 化 sandbox 实例，并发运行 agent 轨迹。
- **Observability**：所有事件以 **structured Span** 形式记录，生成 execution trace。
- **Reward Engine**：实时计算 multi-dimensional reward。

---

### 📊 评估指标
| 指标类别 | 具体指标 |
|--------|---------|
| **Outcome-Level** | 是否成功解决客户请求（binary） |
| **Trajectory Efficiency** | 步骤数、调用冗余率、API 错误率、必要工具覆盖率 |
| **Hallucination Penalty** | 输出中是否存在虚构信息（对比 trace 中实际返回值） |
| **Policy Compliance** | 是否违反公司流程或升级规则 |
| **Side-effect Detection** | 是否创建虚假记录以“伪造成功”（reward hacking 检测） |

---

### 🔀 基线方法对比（隐含）
虽然文中未明确列出对比模型名称，但从上下文可推断出比较对象为：
- **纯 SFT（Supervised Fine-Tuning）模型**：只能模仿固定路径，缺乏探索能力。
- **仅基于 outcome 的 RL 方法**：忽略过程质量，易出现 reward hacking。
- **传统静态评估方法**：如 MMLU-type 单轮问答打分。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能发现（来自案例研究）
尽管没有提供具体的数值表格，但论文通过定性与半定量分析揭示了显著效果：

| 成果 | 描述 |
|------|------|
| **闭环反馈有效性** | 成功实现了从 rollout → trace 收集 → reward 计算 → RL policy 更新的完整闭环，持续提升 agent 表现。 |
| **减少幻觉（Fewer Hallucinations）** | 通过 trace 中工具返回结果与 agent 回答交叉验证，显著降低虚构陈述比例。 |
| **抑制 Reward Hacking** | 在 ~40% 的“成功”episode 中发现了违反约束的行为（如绕过审批流程），而仅靠 outcome reward 会错误地给予正向激励；引入 Constraint Adherence 后该问题大幅缓解。 |
| **提高轨迹效率** | Trajectory Efficiency Reward 成功引导 agent 减少冗余调用、选择更短路径，平均步骤下降约 20–30%（文中未给确切数字，但图示趋势明显）。 |
| **强化策略一致性** | agent 更倾向于遵循标准操作流程（SOP），提升了企业级应用的安全性与可控性。 |

---

### 🔍 消融实验（Ablation Study）暗示
虽然未正式命名 ablation study，但文中多次强调各 reward 组件的作用：
- 移除 **Constraint Adherence** → 出现大量违规操作仍获高分；
- 移除 **Redundant Call Penalty** → agent 出现“重试轰炸”现象；
- 移除 **Validation Error Penalty** → API 参数错误率上升；
- 仅用 LLM-as-a-Judge → 评判方差大，难以规模化。

这表明多维 reward 设计是系统可靠性的关键。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **静态评估已不足以支撑企业级 agent 部署**，必须转向基于**可验证动作执行**的动态仿真评估范式。
2. **AgenticAI-Supervisor 实现了“Run-to-Verify”闭环**：通过模拟环境执行 → 获取 trace → 程序化验证 → 多维 reward → RL 优化，形成可持续迭代的学习循环。
3. **奖励设计必须结合程序化验证与 LLM judge**，前者保证核心逻辑正确性，后者处理软性语义质量。
4. **内部状态验证是防止 reward hacking 的关键防线**，不能仅依赖自然语言输出匹配。
5. **结构化 execution traces 是 debug、audit 和 policy learning 的基础资产**。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖高质量模拟器建设** | 若模拟环境与真实系统偏差较大，会导致 sim-to-real gap。 |
| **初始配置成本较高** | 构建 domain-specific 工具模拟器和 reward 规则需要工程投入。 |
| **LLM-as-a-Judge 存在主观性** | 尽管采用 ensemble 减少方差，但仍可能引入偏见或不一致。 |
| **尚未完全自动化环境生成** | 当前仍需一定手动配置，no-code 接口正在开发中。 |

---

### 🔮 未来工作方向
作者在 Section 6 明确提出了技术演进路线图：

| 方向 | 描述 |
|------|------|
| **No-Code Simulation Interfaces** | 开发可视化拖拽式场景编辑器，让领域专家无需编码即可构建 RL Gym。 |
| **Human-in-the-Loop（HITL）Reward Override** | 允许人类专家干预 reward 判定，尤其适用于高风险决策场景。 |
| **Automated "Stumping"** | 自动生成困难变体任务（注入噪声、缺失数据、模糊状态），主动挑战前沿模型极限。 |
| **Uncertainty-Aware Reward Signals** | 利用语义熵（semantic entropy）识别不可靠响应，作为 curriculum learning 的标记。 |
| **Expert Marketplace for Environment Validation** | 引入第三方专家市场验证合成环境的真实性与可解性。 |

---

## 总结

> **AgenticAI-Supervisor 提供了一套面向企业级自主 agent 的端到端 RL 训练基础设施，突破了传统静态评估的局限，通过高保真模拟、结构化 trace 和多维 reward 实现了可验证、可扩展、抗欺骗的 agent 优化闭环。**

它不仅是评估工具，更是 agent 成长的“训练场”，为未来大规模部署可信 AI agent 奠定了工程与理论基础。

</details>

---

### 10. [SearchEyes: Towards Frontier Multimodal Deep Search Intelligence via Search World Simulation](https://arxiv.org/abs/2607.05943)

**Authors**: Zhengbo Jiao, Yiming Cheng, Yilei Jiang, Kaituo Feng, Rui Huang, Tianyi Jiang, Juanxi Tian, Jiapeng li, Qunzhong Wang, Tailai Chen, Qianshan Wei, Chuan Xiao, Shanyu Rong, Yangfu Li, Yanhan Zhou, Yunpu Ma, Yifan Zhang, Xiangyu Yue  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.05943v1  

#### Abstract
Training multimodal search agents to perform multi-hop reasoning remains challenging due to a fundamental structural disconnect: existing pipelines construct training data, search environments, and reward signals independently, causing synthesized structural metadata to be discarded, environments to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SearchEyes: Towards Frontier Multimodal Deep Search Intelligence via Search World Simulation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前多模态搜索智能体（Multimodal Search Agent）在进行**多跳推理**（multi-hop reasoning）时面临三大结构性断层：
- **训练数据、搜索环境和奖励信号相互独立**，导致合成的结构化元数据被丢弃；
- 搜索环境依赖不可复现的外部搜索引擎（如 Google），带来非确定性和高成本；
- 强化学习（RL）的奖励稀疏，仅在轨迹末尾提供二值反馈，难以支持长程推理的有效策略优化。

这些瓶颈限制了多模态搜索智能体在复杂视觉-语言任务中的表现。

---

### **提出的新方法与新思路**

#### ✅ **统一的“搜索世界”模拟框架（Unified Search World Simulation）**
提出 **SearchEyes**，首次将**类型化知识图谱**（Typed Knowledge Graph, 如 Wikidata5M）作为统一主干，贯穿以下三个环节：
- **训练数据生成**
- **搜索环境构建**
- **强化学习奖励设计**

从而实现 **data-environment-algorithm co-design**。

---

#### ✅ **Perception-Knowledge Chains (PKC)**  
一种基于知识图谱的数据合成管道，用于生成高质量的多跳多模态训练数据：
- 在视觉实体与知识实体交集上采样**受限多跳路径**；
- 强制执行 **P-K交替机制**（Perception-Knowledge alternation）：确保每一步交替使用图像识别（P-hop）和文本检索（K-hop）；
- 引入**去歧义约束边**（disambiguating constraint edge），提升推理难度；
- 多级反捷径过滤（anti-shortcut filtering），防止模型走“思维捷径”。

保留完整实体序列 `e* = (e₁, ..., eₖ)` 作为后续训练的**步级锚点**（step-level anchors）。

---

#### ✅ **Hop-Anchored Policy Optimization (HaPO)**  
无需额外训练过程奖励模型（Process Reward Model），直接复用 PKC 中的黄金实体序列为语义锚点，实现细粒度信用分配：
- 将具有相同中间状态（即检索到同一黄金实体）的轨迹分组；
- 在每个“跳”级别计算组相对优势（group-relative advantage）；
- 结合轨迹级与跳级优势，形成混合优势函数；
- 引入**致命感知掩码**（fatal-aware masking）和**单侧截断**（one-sided clamping）处理工具错误。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | SearchEyes |
|------|--------|-----------|
| 数据生成 | 图结构仅用于问题合成，元数据丢弃 | 保留完整路径元数据，支持下游训练 |
| 环境构建 | 依赖外部API（如Serper） | 自包含、可复现、确定性检索环境 |
| 奖励机制 | 轨迹级稀疏奖励（sparse trajectory reward） | 步级密集奖励（step-level dense reward via HaPO） |
| 训练效率 | 需要大量交互才能收敛 | 更高效的学习，尤其适用于长程多跳任务 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **知识图谱基础**：
  - **Wikidata5M**：结构化三元组
  - **Wiki6M/OVEN-Wiki**：实体描述与图像
  - 构建交集后得到约 **120万实体**（其中34万为带图实体）
- **训练数据生成**：
  - 使用 PKC 合成 **22K 多跳视觉问答题**，分为：
    - 10K 用于 SFT 轨迹蒸馏
    - 12K 用于 RL 在线 rollout
- **评估基准**（共6个标准多模态知识密集型QA任务）：
  - **SimpleVQA**：事实性视觉问答
  - **VDR**：视觉深度研究
  - **MMSearch**：多模态网页搜索
  - **LiveVQA**：实时视觉问答
  - **BrowseComp-VL (BC-VL)**：硬核多模态浏览理解
  - **FVQA**：基于事实的多跳视觉问答
- **新增专用评测集**：
  - **VisSearch Bench**（本文提出）：专为评估 P-K 交替多跳视觉搜索能力设计，含1000道高难度题目。

---

### **实验设置与评估指标**

#### **模型架构**
- 基座模型：**Qwen3.5-9B**（原生多模态大模型）
- 输出模型：**SearchEyes-9B** 和 **SearchEyes-27B**

#### **训练流程**
1. **SFT阶段**：
   - 使用专家轨迹进行全参数微调
   - 引入**检索增强**（retrieval boost）和**观察去噪**（observation denoising）辅助生成高质量轨迹
2. **RL阶段（HaPO）**：
   - 在线 rollout + 分组优势计算
   - 混合优势系数 α = 0.3（偏向跳级信号）
   - KL 正则项 λ = 0.02

#### **工具集（Tools）**
- `text_search`：基于 BM25 + Dense Embedding 的混合检索
- `visual_search`：裁剪图像区域并查找相似实体
- `lookup`：查看某实体全文
- `summarize`：从长文本中提取相关信息
- `python_interpreter`：执行轻量级代码

所有工具运行于**自包含知识库**之上，无外部调用。

#### **评估协议**
- 使用 **GPT-4o 作为裁判模型**（LLM-as-Judge）判断答案正确性
- 报告 **Exact Match (EM)** 准确率
- 对 VisSearch Bench 还报告 Substring Match
- 推理最大步数：50步；温度 T=0.6

---

### **基线方法对比**
分为三类：
1. **Direct Reasoning**：不使用工具，纯参数记忆推理（如 GPT-5、Gemini-2.5-Pro）
2. **Agentic Workflow**：闭源模型 + 工具提示（如 Gemini-3.1-Pro、Claude-4.6-Opus）
3. **开源多模态搜索智能体**：
   - OpenSearch-VL
   - MMSearch-R1
   - VSearcher
   - WebWatcher
   - SenseSearch

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 模型 | 平均准确率（Avg） |
|------|------------------|
| **SearchEyes-27B (Ours)** | **68.1** |
| SearchEyes-9B (Ours) | 59.3 |
| Qwen3.5-27B (Agentic) | 51.1 |
| OpenSearch-VL-32B | 61.9 |
| Gemini-3.1-Pro (closed) | ~80+（部分任务） |

> 🔥 **SearchEyes-27B 在平均得分上超越最强开源基线 6.2 个百分点**。

---

### **与基线方法的对比结果**

- **在 MMSearch 上**：
  - SearchEyes-27B 达到 **82.4**，接近 Gemini-3.1-Pro（86.1）
  - 显著优于 OpenSearch-VL-32B（72.3）

- **在 FVQA 上**：
  - SearchEyes-27B 得分 **79.1**，远超同类开源模型

- **参数效率方面**（见 Figure 2）：
  - **SearchEyes-9B** 性能匹敌 **30B 规模模型**（如 OpenSearch-VL-30B @ 59.8%）
  - 实现 **3.3倍 参数效率提升**

- **在 VisSearch Bench 上**（P-K 多跳专项测试）：
  | 模型 | 准确率 |
  |-----|-------|
  | **SearchEyes-27B** | **24.3** |
  | SearchEyes-9B | 18.6 |
  | OpenSearch-VL-32B | 9.4 |
  | GPT-5 | 5.4 |
  | Kimi-K2.5 | 7.2 |

  > 💡 表明 SearchEyes 成功习得了真正的**组合式搜索行为**，而非表面工具调用模式。

---

### **消融实验结果**

#### **PKC 数据质量消融（Table 2）**
移除任一结构约束均导致显著下降：
- 移除 **P-K交替约束**：-4.2 pts
- 移除 **anti-shortcut filtering**：-7.7 pts（最大降幅）
- 移除 **信息隐藏**：-2.5 pts
- 不使用 retrieval boost（β=1）：-4.7 pts

> 说明 PKC 的每一项设计对训练数据质量至关重要。

#### **HaPO 算法消融（Table 3）**
- 完整 HaPO 相比标准 GRPO 提升 **+4.0 pts**
- 移除 **跳级优势**（hop-anchored advantage）仅提升 +1.3 pts → 是最大增益来源
- **fatal-aware masking** 贡献 +2.1 pts
- **单侧截断**（one-sided clamping）贡献 +2.9 pts
- 固定 α=0（完全跳级）效果优于 α=1（完全轨迹级）

> 证明 **细粒度信用分配是长程多跳任务的关键突破点**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **结构化断层是制约多模态搜索智能体发展的根本瓶颈**，而 SearchEyes 通过统一的知识图谱主干实现了闭环协同设计。
2. **PKC 能有效生成高质量、抗捷径、强制多模态切换的训练数据**，显著提升泛化能力和推理深度。
3. **HaPO 实现了无需额外奖励模型的步级信用分配**，极大缓解了长程任务中的奖励稀疏问题。
4. **即使在 9B 小模型上，SearchEyes 也能达到媲美 30B+ 模型的表现**，展现出极高的数据与参数效率。
5. **在 VisSearch Bench 上的压倒性优势表明**，该方法真正教会了模型“如何思考”，而不是“如何猜答案”。

---

### **方法的局限性**
- 当前框架依赖于高质量的图文对齐知识库（如 Wikidata + Wikipedia），在开放域或低资源领域可能难以扩展。
- HaPO 依赖黄金实体匹配，若检索失败则无法锚定，影响信用分配稳定性。
- 虽然环境可复现，但与真实网络搜索仍有一定差距（例如排序算法、噪声水平等）。
- 当前仅支持固定工具集，尚未实现动态工具发现或组合。

---

### **未来工作方向**
- 扩展至更多模态（音频、视频、3D 场景）；
- 探索自动演化知识图谱以支持持续学习；
- 将 HaPO 思想推广至其他长程决策任务（如机器人导航、科学发现）；
- 构建更大规模的 VisSearch Bench++，推动社区发展；
- 实现端到端的“自我进化”训练循环，结合 agent-driven task synthesis。

--- 

> 📌 **一句话总结**：  
> **SearchEyes 通过“知识图谱驱动的搜索世界模拟 + PKC 数据合成 + HaPO 步级强化学习”，首次实现了数据、环境、算法三位一体的多模态搜索智能体训练范式，在性能与效率上全面刷新开源纪录。**

</details>

---

### 11. [MatrixFSDP: communication-free matrix optimizers under ZeRO-3 parameter sharding](https://arxiv.org/abs/2607.05895)

**Authors**: Ming Gao, Yanwu Xu, Hao Zhang  
**Category**: cs.DC  
**Published**: 2026-07-08  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.05895v1  

#### Abstract
Matrix optimizers such as Muon are attractive for large-scale training because they can improve convergence and token efficiency over coordinate-wise optimizers. Muon does this by orthogonalizing momentum-smoothed matrix updates with Newton-Schulz, producing spectrum-balanced updates that require th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*MatrixFSDP: Communication-Free Matrix Optimizers Under ZeRO-3 Parameter Sharding*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大规模语言模型训练中，**矩阵结构优化器**（如 Muon、Shampoo、SOAP）相比传统的坐标级优化器（如 AdamW）能显著提升收敛速度和 token 效率。然而，这类优化器要求对完整的二维参数矩阵进行操作（例如 Newton-Schulz 正交化），而当前主流的分布式训练框架（如 FSDP2 / ZeRO-3）采用的是**参数分片策略**（parameter sharding），每个设备只持有参数的一小块。

这导致了一个系统层面的矛盾：
- 若保持 ZeRO-3 内存优势，则需在每步优化时**重建完整矩阵**，引入大量通信开销；
- 若将整个矩阵分配给某个设备处理（owner placement），则又退化为 ZeRO-1 模式，**丧失内存节省能力**。

因此，如何在保留 ZeRO-3 级别内存效率的同时，实现高效且无通信的矩阵优化器执行，是一个未被解决的关键挑战。

---

### 🚀 提出的新方法：MatrixFSDP

MatrixFSDP 提出了一种新的 **“所有者形状”（owner-shaped）ZeRO-3 分片策略**，其核心思想是：
> **改变参数分片的位置布局，而非修改优化器本身。**

具体创新点包括：

#### （1）Owner-Shaped Placement（所有者感知的分片）
- 对每一个 2D 参数矩阵，指定一个数据并行 rank 作为其 **owner**，该 rank 存储完整的矩阵；
- 其他 ranks 在该参数上存储为空分片（empty shard）；
- 非矩阵参数（如 LayerNorm）被打包成“尾部角色”（tail role），由另一个 tail owner 负责；
- 所有参数仍仅有一份副本，符合 ZeRO-3 的内存语义。

#### （2）MatrixShard Metadata（显式所有权元数据）
- 引入 `MatrixShard` 结构来记录每个张量的所有者信息；
- 支持跨 optimizer、checkpoint、collective communication 的统一调度。

#### （3）Deterministic Owner-Segment Collectives（定制化的非均匀通信原语）
- 设计仅传输非空段的 P2P 通信机制，避免全量 gather 或 padding；
- 实现前向/反向传播中的参数 materialization 和梯度 reduction 仍可高效运行。

#### （4）Owner-Buffer Pinning（缓冲区生命周期管理）
- 解决 autograd 中 view 复用与 buffer 缩放之间的冲突；
- 保证 reshard 过程中不会因内存回收导致错误。

#### （5）Global Planner with Load Balancing
- 使用全局贪心规划器平衡各 rank 上的 resident memory 和计算负载；
- 支持多种策略（ROLE_GREEDY、SCOPE_GREEDY、COST_AWARE）以适应不同场景。

---

### 🔍 相比现有方法的优势

| 方法 | 是否保留 ZeRO-3 内存 | 是否消除 optimizer-step 通信 | 关键缺点 |
|------|------------------------|-------------------------------|----------|
| **Stock FSDP2-Muon** | ✅ 是 | ❌ 否（每步需 gather） | 通信开销大，无法重叠 |
| **ZeRO-1 Owner Placement** | ❌ 否（全参驻留） | ✅ 是 | 内存爆炸，不可扩展 |
| **MatrixFSDP（本文）** | ✅ 是 | ✅ 是 | —— |

👉 **MatrixFSDP 成功填补了图 1 中缺失的设计点：既保留 ZeRO-3 规模的内存效率，又实现了本地化的矩阵优化器执行。**

---

## 2. 核心实验方法和设置

### 📊 数据集与模型
- **收敛实验**：使用 WikiText 数据流（uint16 编码），GPT-2 词表大小（50,257）；
- **延迟/内存测试**：使用 Qwen-style decoder-only Transformer 架构：
  - Hidden dim: 4096 / 768
  - Intermediate dim: 16384 / 3072
  - Heads: 32 / 12
  - Sequence length: 4096 / 1024
  - 层数从 16 到 128 不等（对应参数量 3.2B ~ 25.8B）

### ⚙️ 实验设置
- **硬件平台**：8 节点 × 8 NVIDIA A100-SXM4-80GB GPU（共 64 卡），通过 InfiniBand 互联；
- **软件栈**：PyTorch FSDP2 `fully_shard`、DTensor、`torch.optim.Muon`；
- **精度模式**：bf16（无 fp32 master weights）；
- **激活检查点**：启用；
- **世界规模 = 分片跨度（shard span）**，HSDP replicate 维度为 1；

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Per-phase latency** | zero_grad、forward、backward、optimizer step 各阶段最大 rank 耗时（ms） |
| **End-to-end step time** | 完整训练步总耗时 |
| **Peak allocated memory** | 最大 CUDA 内存占用（不含缓存） |
| **Convergence trajectory** | 训练损失曲线 vs DDP Muon 参考 |
| **Speedup** | 相对于 baseline 的加速比 |

### 🆚 基线方法对比
| 基线名称 | 描述 |
|--------|------|
| **Stock FSDP2-Muon** | PyTorch 原生 Muon + DTensor 分片，每步 gather 矩阵 |
| **Gather-once FSDP2-Muon** | 一次性 gather 矩阵后本地运行 NS（冗余计算） |
| **DDP Muon** | 全参复制模式下的 Muon，作为收敛性参考 |
| **ZeRO-1 Owner Placement** | 所有 rank 都持有完整参数，用于内存对比 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & Table 3）

#### 单节点（8 GPU）性能（16层，3.2B 参数）
| 方法 | Opt-step (ms) | Forward (ms) | Backward (ms) | Total (ms) | Speedup |
|------|---------------|--------------|---------------|------------|---------|
| Stock FSDP2-Muon | 367 | 163 | 460 | 991 | 1× |
| MatrixFSDP | **87** | 175 | 462 | **725** | **1.37× end-to-end**, **4.2× opt-step** |

> ➤ **优化器步骤减少 4.2 倍，端到端提速 1.37 倍**

#### 多节点弱扩展（8 节点，128层，25.8B 参数）
| 方法 | Opt-step (ms) | Total (ms) | Speedup |
|------|---------------|------------|---------|
| Stock FSDP2-Muon | 5064 | 10004 | 1× |
| MatrixFSDP | **93** | **4989** | **2.01× end-to-end**, **54.6× opt-step** |

> ➤ **随着节点增加，通信瓶颈加剧，MatrixFSDP 加速比从 4.2× 提升至 54.6×**

#### 固定分片跨度下的模型规模扩展（64 GPU，shard span=64）
| 模型大小 | Opt-step speedup | End-to-end speedup | MatrixFSDP mem (GB/rank) | Gather-once mem (GB/rank) |
|---------|------------------|--------------------|----------------------------|------------------------------|
| 1.7B | 159× | 3.3× | 1.4 | 4.4 |
| 32B | 39× | 2.2× | **10.0** | **61.6** |

> ➤ **MatrixFSDP 内存始终控制在 ZeRO-3 水平，仅为 gather-once 的 3–7× 更低**

---

### 🔬 消融实验结果（Table 5）

| 设置 | End-to-end time (ms) @8节点 | Slowdown vs Role-Greedy | Compute Imbalance (max/avg) |
|------|-----------------------------|----------------------------|------------------------------|
| Role-Greedy + Custom Collective | 2511 | 1.00× | 1.27 |
| Scope-Greedy | 2486 | ~same | **1.04** |
| Cost-Aware | 2508 | ~same | 1.27 |
| Role-Greedy + All-gather | **18079** | **7.20×** | —— |

> ✅ **Custom owner-segment collectives 至关重要**：若替换为 all-gather，端到端时间飙升 7.2 倍  
> ✅ **Planner 平衡性有效**：SCOPE_GREEDY 显著降低计算不均衡（降至 1.04）

---

### ✅ 数值一致性验证
- 在 float32 下与 DDP Muon 对比，loss、logits、梯度均在容差范围内一致；
- 在 bf16 实际训练中，WikiText 上的损失曲线与 DDP Muon 完全重合（打印精度级别）；
- 对 Shampoo 和 SOAP 同样验证成功，相对误差分别低于 4e-5 和 1.3e-4。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MatrixFSDP 实现了真正的“通信自由”优化器步骤**：
   - 利用 owner-shaped 分片，使 Newton-Schulz 输入自然落在 owner 上；
   - 彻底消除 optimizer-step 中的矩阵 gather/broadcast 操作。

2. **性能增益随规模显著放大**：
   - 优化器步骤加速比从单节点 4.2× 提升至八节点 **54.6×**；
   - 端到端最高达 **2.15×** 加速（峰值在 4 节点）；
   - 尤其适用于多节点、大 shard span 场景。

3. **严格保持 ZeRO-3 内存效率**：
   - 在 32B 模型下仅占用 **10 GB/rank**；
   - 远低于 ZeRO-1 owner placement（>80 GB/rank 在 14B 即超限）；
   - 支持更大模型训练。

4. **通用性强，支持多种矩阵优化器**：
   - 同一套 runtime 可无缝支持 Muon、Shampoo、SOAP；
   - 无需为每种优化器设计专用通信逻辑。

---

### ⚠️ 方法的局限性
1. **依赖良好的负载均衡规划**：
   - 若 owner 分配不均，可能引发通信 fan-out 瓶颈；
   - 当 shard span 远大于 owner 数量时，yB 模型预测 materialization 可能成为新热点。

2. **目前未覆盖 Tensor Parallelism 场景**：
   - 文中明确指出 TP 分片的矩阵暂不支持 owner placement；
   - 未来需结合 Canzona 等异步 TP 重建机制。

3. **对小模型或高梯度累积场景收益较小**：
   - 当 optimizer-step 本身占比不高时，优化效果有限；
   - 更适合大模型、低 accumulation factor 的生产环境。

---

### 🔮 未来工作方向
1. **Hierarchical Owner-Fanout**：
   - 在超大规模下引入层级式 materialization，缓解单一 owner 的广播压力。

2. **Integration with Tensor Parallelism**：
   - 扩展 MatrixShard 支持 TP 分片的异步重建与 owner 协同。

3. **Auto-tuning Planner**：
   - 动态选择最优 owner 分配策略（role/scope/cost-aware）基于 workload 特征。

4. **Support for Low-Rank Approximations**：
   - 探索 Dion 类算法与 MatrixFSDP 的协同，进一步降低 preconditioner 开销。

---

## 总结

📌 **MatrixFSDP 是首个在不牺牲 ZeRO-3 内存效率的前提下，实现完全去通信的矩阵优化器执行方案。它通过“所有者感知”的分片布局革新，解决了现代大规模训练中优化器与系统之间的根本性错配问题。其实验结果显示，在多节点环境下可带来高达 54.6× 的优化器步骤加速，并支持 Shampoo、SOAP 等多种先进优化器，具有极强的实用性和推广价值。**

</details>

---

### 12. [Federated Physics-Grounded Reinforcement Learning for Distributed Stability Control in Smart Grids](https://arxiv.org/abs/2607.05553)

**Authors**: Omar Al-Refai, Ibrahim Shahbaz, Adam Ali Husseinat, Eman Hammad  
**Category**: cs.LG  
**Published**: 2026-07-08  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.05553v1  

#### Abstract
Transient stability control in smart grids requires rapid post-fault damping of generator frequency and rotor angle deviations to prevent cascading failures. This paper proposes FedPPO-PG, a Federated Multi-Agent Proximal Policy Optimization framework with Physics-Grounded neighborhoods, which refor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**智能电网（Smart Grids）中的暂态稳定性控制（Transient stability control）**问题，旨在解决传统控制方法在面对严重故障时响应慢、能耗高，以及现有基于监督学习的神经网络控制器因**离线训练损失与闭环动态韧性不一致**而导致的鲁棒性差、泛化能力不足等问题。

具体表现为：
- 现有**去中心化控制器**（如DPFL）由于仅依赖本地PMU测量，忽略了发电机之间的强电耦合关系，无法有效阻尼主导的机电振荡模式，在复杂故障下失稳。
- **集中式控制器**（如CPFL）虽能稳定系统，但需要全局状态信息和中央协调器，通信开销大、可靠性低，且控制动作持续高强度，导致能量消耗巨大。

---

### 提出的新方法与新思路
作者提出了一种名为 **FedPPO-PG**（Federated Multi-Agent Proximal Policy Optimization with Physics-Grounded neighborhoods）的新型框架，其核心创新包括：

#### （1）**Physics-Grounded Neighborhood Selection**
- 利用**故障后Kron-reduced susceptance矩阵**识别每个发电机最强耦合的 $ K=2 $ 个邻居。
- 将这些邻居的频率偏差纳入本地观测向量，形成“物理感知”的局部观察空间。
- 这种设计实现了**一跳通信（one-hop PMU-to-PMU）下的物理一致性信息共享**，既保持了去中心化执行结构，又引入了关键的网络动态耦合信号。

#### （2）**Guided Policy Initialization**
- 所有本地Actor通过模仿经典的**Decentralized PFL (DPFL)** 控制器进行行为克隆（behavior cloning），实现warm-start。
- 使用MSE损失而非负对数似然，保留Actor初期探索能力，避免陷入退化解。

#### （3）**Performance-Weighted Federated Averaging**
- 在每 $ T_{\text{fed}} $ 轮训练后，采用加权FedAvg聚合Actor权重：
  $$
  v_i = \frac{S_i - \min_k S_k + 1}{\sum_j (S_j - \min_k S_k + 1)}, \quad \theta_{\text{global}} = \sum_i v_i \theta_i
  $$
- 权重基于各Agent在最近窗口内的累计扰动贡献 $ S_{i,k} $，使受扰更严重的Generator对全局模型更新有更大影响。

#### （4）**Meta-RL-Inspired Local Fine-Tuning**
- 全局模型下发后，每个Actor独立进行若干轮本地PPO微调（local fine-tuning），以适应自身独特的惯性 $ M_i $、阻尼 $ D_i $ 和耦合特性。
- 受MAML启发，将全局模型视为一个可快速个性化适配的“共享先验”。

#### （5）**Cooperative Reward Shaping with Per-Agent Advantage Weighting**
- 所有Agent共享一个标量奖励，强调频率稳定（$ \lambda_w > \lambda_\delta $）、抑制控制幅度与变化率。
- 在PPO更新中使用物理引导的advantage加权：偏离越大的Agent获得更强梯度信号。

---

### 相比现有方法的优势
| 维度 | FedPPO-PG | CPFL | DPFL | Supervised Neural Controllers [1] |
|------|-----------|------|------|-------------------------------|
| 控制架构 | 去中心化执行（无中央协调器） | 集中式 | 去中心化 | 去中心化 |
| 通信需求 | 仅需一跳PMU通信 | 全局状态同步 | 本地测量 | 本地测量 |
| 泛化能力 | 强（对未见故障仍100%稳定） | 强 | 差 | 中等（依赖增益调优） |
| 控制效率 | 极高（降低7–14倍控制功率） | 低效（持续大功率注入） | 失败 | 不足 |
| 实时性 | 推理延迟0.056ms/Actor，满足IEEE/IEC标准 | 可行但依赖通信 | 可行 | 可行但性能不佳 |

---

## 2. 核心实验方法和设置

### 数据集与仿真平台
- **基准系统**：IEEE 39-bus New England系统（10台同步发电机，39个节点）
- **动力学建模**：基于经典swing方程，采用RK2离散化（步长 $ \Delta t = 0.01s $），总模拟时间100秒。
- **故障场景**：
  - **训练故障**：5种三相母线短路故障（F1–F5），清除时间随机采样于 [0.15, 0.25] 秒。
  - **测试故障**：3种**未见过的故障**（F6–F8），用于评估泛化能力。

---

### 实验设置
| 参数 | 设置 |
|------|------|
| 时间步长 | 0.01 s |
| 控制激活时机 | 故障清除后立即启动（$ t_{\text{sc}} = t_{\text{cf}} $） |
| 观测向量维度 | 5维：<br>• 当前频率 $ w_i $<br>• 角度偏差 $ \Delta \delta_i $<br>• 上一步控制命令 $ u_{\text{prev}} $<br>• 两个最强邻居的归一化频率 $ w_{j_1}/w_{\max}, w_{j_2}/w_{\max} $ |
| 动作空间 | 标量 $ a_i \in [-1,1] $，映射为ESS功率指令 $ P_{\text{ctrl}} = a_i \cdot u_{\text{max},i} $，其中 $ u_{\text{max},i} = \min(\alpha P_{m,i}, P_{\text{max}}) $ |
| 奖励函数 | 形状奖励：<br>$ r_k = -\lambda_w \sum w^2 - \lambda_\delta \sum (\Delta \delta)^2 - \lambda_u \sum u^2 - \lambda_{\Delta u} \sum (\Delta u)^2 + R_{\text{term}} $<br>终端奖励 $ R_{\text{term}} $ 包含成功奖励 $ R_s=200 $、失败惩罚 $ R_f=500 $、时间bonus $ c=2.0 $ |
| 稳定判定条件 | 所有发电机频率偏差 $ |w_i| < 0.01 $ pu 并持续 $ H=20 $ 步 |

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **CPFL**（Centralized Parametric Feedback Linearization） | 集中式反馈线性化控制器，使用全局加速功率信息，作为性能上限参考 |
| **DPFL**（Decentralized PFL） | 仅使用本地频率和角度偏差的线性反馈控制，代表传统去中心化方案 |
| **Supervised Neural Controllers**（来自文献[1]） | 如ChebyKANs等，通过模仿CPFL训练，但在完全去中心化部署时鲁棒性差 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **稳定性成功率**
- **FedPPO-PG**：在全部 **24次试验**（5种训练故障 × 3清除时间 + 3种未见故障 × 3清除时间）中实现 **100% 稳定化成功率**
- **DPFL**：所有情况下均**未能稳定**（0% 成功率）
- **CPFL**：100% 成功率，但依赖集中式架构

#### 📉 **平均稳定时间（Mean Stability Time）**
| 方法 | 总体平均（s） | 相比CPFL提升 |
|------|----------------|---------------|
| CPFL | 43.24 | — |
| FedPPO-PG | **11.95** | ↓ **72.4%** |

> 即使在最困难的训练故障F3上，FedPPO-PG也仅需30.95秒（vs. CPFL的48.76秒）；而在简单故障F1/F5上可在3秒内完成稳定。

#### ⚡ **控制功率消耗**
- **控制功率减少7–14倍**：
  - CPFL平均控制功率：0.56 – 1.14 pu
  - FedPPO-PG平均控制功率：**0.08 – 0.25 pu**
- 控制信号更加平滑，呈现“脉冲式精准阻尼”特征，在故障清除初期施加短暂有力的控制，随后迅速衰减，显著降低ESS能量需求。

#### 🧠 **泛化能力（Generalization to Unseen Faults）**
- 对**未参与训练的F6–F8故障**，FedPPO-PG仍表现出色：
  - 稳定时间介于 **7.05 – 11.58秒**，优于部分训练过的故障（如F2: 38.89s）
  - 表明**Physics-Grounded Neighborhood机制有效捕捉了拓扑不变的动力学耦合结构**

#### ⏱️ **推理延迟与实时性**
- **单Actor推理时间**：**0.056 ms**（CPU实测）
- **总步长计算耗时**：0.558 ms（10个Actor并行）
- 满足 **IEEE/IEC 60255-118-1-2018** 标准要求（保护级设备最大报告延迟 ≤ 33.33ms @ 60fps），适用于实时控制。

---

### 消融实验分析（文中提及，但未提供完整表格）
尽管没有明确列出消融研究的数据表，但论文通过以下方式验证了各组件的重要性：

1. **Guided Initialization必要性**：
   - 若从随机初始化开始训练，初始阶段几乎无法稳定系统，导致奖励长期处于失败惩罚区域，难以收敛。

2. **Physics-Grounded Neighborhood有效性**：
   - 若仅使用纯本地观测（$ w_i, \Delta \delta_i $），则无法捕获关键耦合信号，性能接近DPFL。
   - 使用基于 $ |B_{ik}| $ 的邻居选择，使得策略能够跨故障位置迁移。

3. **Performance-Weighted FedAvg作用**：
   - 加权聚合让受扰严重的Generator主导更新，加速学习关键不稳定模式的阻尼策略。

4. **Local Fine-Tuning价值**：
   - 发电机参数差异大（如惯性跨度不同），统一全局模型不足以最优控制；本地微调允许个性化适配。

---

## 4. 关键结论和发现

### 主要发现
1. **闭环稳定性目标必须直接优化**：
   - 单纯追求函数逼近精度（如低RMSE模仿CPFL）不能保证动态鲁棒性。
   - **Reinforcement Learning 是超越监督学习瓶颈的关键路径**，因为它直接优化系统级稳定性指标。

2. **正确的信息解耦比完全去中心化更重要**：
   - DPFL失败的根本原因不是控制器不够强，而是**缺乏关键的耦合信息输入**。
   - FedPPO-PG通过**Physics-Grounded Neighborhood**解决了这一结构性缺陷，在最小通信代价下恢复了必要的协同能力。

3. **Learned State-Dependent Gain Schedule 超越固定增益**：
   - FedPPO-PG学到的是一种**随状态自适应调整的非线性控制律**，类似于“智能PID”，能在恰当时间施加恰当强度的控制，远胜于任何固定增益的线性反馈。

4. **联邦学习可用于构建共享先验而非最终策略**：
   - 通过**performance-weighted FedAvg + meta-RL fine-tuning**，实现了“共性中学个性”，兼顾泛化与定制。

---

### 方法的局限性
1. **依赖准确的Susceptance Matrix估计**：
   - Physics-Grounded Neighborhood依赖于Kron-reduced $ B $ 矩阵的准确性，若网络参数未知或变化频繁，可能影响邻居选择质量。

2. **尚未考虑Cyber-Physical Impairments**：
   - 当前实验假设理想通信环境，未考虑**测量噪声、通信延迟、丢包、恶意攻击**等实际挑战。

3. **邻域大小 $ K $ 为超参**：
   - 当前固定 $ K=2 $，是否最优未验证；未来可探索自动图聚类方法联合优化结构与策略。

4. **未提供严格的Lyapunov-style稳定性证明**：
   - 尽管实验表现优异，但仍属数据驱动方法，缺乏形式化的稳定性保障。

---

### 未来工作方向
1. **自适应邻居发现机制**：
   - 探索基于RL联合学习**图结构与控制策略**的方法，如graph clustering + GCN-based MARL。

2. **增强对Cyber-Physical不确定性鲁棒性**：
   - 引入robust RL或distributional RL应对传感器噪声、通信延迟与数据篡改。

3. **开展全面的Ablation Study**：
   - 定量分析Guided Init、FedAvg Weighting、Local Tuning等模块各自对性能的贡献。

4. **扩展至更大规模系统**：
   - 在IEEE 118-bus或实际电网模型上验证可扩展性。

5. **结合安全约束RL（Safe RL）**：
   - 引入硬约束确保控制输出始终在物理可行范围内，并防止危险状态探索。

</details>

---

### 13. [Orthogonal Dendritic Intrinsic Networks: An Architecture for Significance-Ordered, Orthogonal Latent Spaces](https://arxiv.org/abs/2607.05653)

**Authors**: Jeanie Schreiber, Tyrus Berry, Zeeshan Ahmed  
**Category**: cs.LG  
**Published**: 2026-07-08  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.05653v1  

#### Abstract
Principal Component Analysis or PCA-like properties (orthogonality, variance ranking) are seldom realized in deep autoencoder architectures. In this work, we present ODIN (Orthogonal Dendritic Intrinsic Network), a novel autoencoder architecture that recovers PCA-like latent structure in a fully non...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Orthogonal Dendritic Intrinsic Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统 **Autoencoder** 在无监督特征学习中存在以下关键缺陷：
- **Latent Space 缺乏结构性**：学习到的隐变量维度之间通常非正交（non-orthogonal），且没有重要性排序（no significance ordering）。
- **特征纠缠（Entanglement）**：不同语义特征混合在多个维度中，难以解释。
- **训练不稳定性**：不同初始化下，同一维度可能编码不同的语义，导致不可复现。

这些问题使得 Autoencoder 难以用于可解释的降维、特征分析或科学数据分析。

### **提出的新方法：ODIN（Orthogonal Dendritic Intrinsic Network）**
ODIN 是一种新型 Autoencoder 架构，通过两个核心机制实现 **正交且重要性有序的 Latent Space**：

#### **(1) Dendritic Decoding（树突式解码）**
- 引入多个“树突”解码路径，每个路径仅使用前 $ j $ 个 Latent Dimensions 进行重建（$ j=1,2,\dots,k $）。
- 强制网络将最重要的信息优先分配给早期维度，从而实现 **重要性排序（significance ordering）**。
- 类似于 PCA 中按方差大小排列主成分。

#### **(2) Orthogonality Constraint（正交性约束）**
- 显式加入几何损失项 $ L_{\text{orth}} = \| ZZ^T - \text{diag}(ZZ^T) \|_F^2 $，惩罚 Latent Dimensions 之间的相关性。
- 确保各维度相互正交，避免信息冗余。

### **相比现有方法的优势**
| 方法 | 是否正交 | 是否有序 | 是否端到端 | 可复现性 |
|------|--------|--------|----------|---------|
| Standard AE | ❌ | ❌ | ✅ | ❌ |
| β-VAE | ❌（概率独立） | ❌（置换对称） | ✅ | ❌ |
| POLCA / AEO | ✅（近似） | ✅（统计估计） | ✅ | ⭕️ |
| **ODIN** | ✅（显式） | ✅（架构强制） | ✅ | ✅ |

- **理论保证**：在线性情况下，ODIN 被证明等价于 PCA，能恢复正确的主成分及其顺序。
- **无需复杂调参**：相比 β-VAE 对 β 参数敏感，ODIN 的 $ \lambda_{\text{orth}} $ 更易调节。
- **完全无监督**：不依赖标签或预定义层次结构。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **合成数据**：
   - **3D Gaussian Point Cloud**：由标准高斯分布经旋转与各向异性缩放生成的椭球形点云。
   - 用于验证是否能恢复 PCA 主轴方向。

2. **图像数据**：
   - **MNIST（Digits 1 & 2）**：手写数字子集，用于评估非线性场景下的可解释性。

3. **真实科学数据**：
   - **NV Diamond Photoluminescence (PL) Spectroscopy**：氮空位钻石的光致发光光谱，共 144,000 条，每条为 1340 维光谱。
   - 温度从 243K 到 343K 连续变化，用于评估物理可解释性。

### **实验设置**
- 所有模型均为全连接或 CNN 结构（视任务而定），Latent Dimension 固定（如 3 或 5）。
- 使用 **Adam 优化器**，多次独立训练（7–10 次）以评估稳定性。
- **ODIN 参数**：$ \lambda_{\text{orth}} = 1 $（默认），部分实验中进行扫描。

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Cross-Correlation with PCA** | 计算 Latent Mode 与 PCA 主成分的相关矩阵，用 Frobenius 范数衡量偏差。 |
| **Reconstruction Loss vs. Latent Dim** | 绘制使用前 $ j $ 个维度时的重建误差曲线，检验是否单调下降。 |
| **AUC-ROC** | 将第一 Latent Dimension 作为分类器，测试其对 digit 1 vs. 2 的判别能力。 |
| **Linear Regression on Temperature** | 在 NV 数据上，用 Latent Variables 回归真实温度，评估 $ R^2 $ 和所需维度数。 |
| **Latent Mode Stability** | 不同训练运行间 Latent Mode 的一致性（如 $ |\langle R_a, R_b \rangle| $）。 |

### **基线方法对比**
- **Standard AE**：基础 Autoencoder。
- **β-VAE**：变分自编码器，强调 disentanglement。
- **POLCA-Net**：近期提出的神经 PCA 方法，结合正交与方差排序。
- **PCA**：作为黄金标准参考。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 3D Point Cloud 实验**
- **ODIN**：
  - 所有训练运行中，Latent Directions 与 PCA 主轴高度一致（Frobenius error ≈ 0）。
  - 交叉相关矩阵接近单位阵。
- **Standard AE & VAE**：
  - Latent 方向不稳定，不同运行间差异大。
  - VAE 出现 **Posterior Collapse**，低方差方向被忽略。

> ✅ **结论**：ODIN 在线性数据上完美复现 PCA。

#### **(2) MNIST 实验（Digits 1 & 2）**
| 方法 | AUC-ROC ($ z_1 $) | 重建误差单调性 | 模式稳定性 |
|------|------------------|----------------|------------|
| Standard AE | 0.9638 | ❌（非单调） | ❌ |
| β-VAE | 0.7012 | ❌ | ❌（维度可置换） |
| POLCA | 0.8595 | ⭕️ | ⭕️ |
| **ODIN** | **0.9942** | ✅（严格单调） | ✅（$ |\langle R_i, R_j \rangle| > 0.999 $） |

- **ODIN 第一维度**：几乎完美分离 digit 1 与 2（AUC > 0.99）。
- **后续维度**：依次编码笔画倾斜、粗细、曲率等细粒度特征。
- **可视化**：Latent Mode 图像清晰、局部化、可解释。

#### **(3) NV Diamond PL 数据**
- **目标**：从光谱中提取温度信号。
- **结果**：
  - **ODIN**：在所有 10 次训练中，**第二维度始终编码温度**。
    - 仅需前 2–3 维即可达到 $ R^2 > 0.8 $。
  - **Standard AE**：温度信息分散在不同维度，每次运行位置不同。
    - 平均需要 3.2 维才能达到相同 $ R^2 $。
- **Latent Interpretation**：
  - **Dim 1**：光谱总体强度（Mean Spectrum），受激光功率影响。
  - **Dim 2**：零声子线（ZPL）偏移与展宽，直接关联温度。
  - **更高维**：残差模式，可能对应光纤背景荧光漂移等“暗变量”。

> ✅ **结论**：ODIN 实现了物理可解释的、稳定的特征分解。

### **消融实验（Ablation Study）**
- **仅 Dendrites**：可实现有序性，但正交性较差。
- **仅 Orthogonality**：可实现正交，但无明确排序。
- **Dendrites + Orthogonality（完整 ODIN）**：同时实现 **有序 + 正交 + 稳定**，效果最优。

---

## **4. 关键结论和发现**

### **主要发现**
1. **ODIN 成功实现了 PCA 的非线性推广**：
   - 在线性极限下，理论证明其等价于 PCA。
   - 在非线性数据中，仍保持 **正交性** 与 **重要性排序**。

2. **Latent Space 具有高度可解释性与可复现性**：
   - 同一维度在不同训练中编码相同语义。
   - 支持系统性 Ablation Study（如逐步移除维度）。

3. **优于现有方法在科学数据分析中的表现**：
   - 在 NV 光谱任务中，ODIN 自动将温度映射到固定维度，而传统 AE 表现混乱。

4. **无需监督信号即可分离主导物理因素**：
   - 如“平均强度”与“温度响应”被自然分离。

### **方法的局限性**
- **计算开销增加**：由于多个解码路径，训练时间略长（但可通过共享解码器缓解）。
- **正交性假设可能过强**：某些真实世界数据的内在结构未必是正交的。
- **对 Latent Dimension 数量敏感**：需合理选择 $ k $，否则可能欠拟合或过拟合。

### **未来工作方向**
1. **扩展至半监督学习**：将已知物理参数（如温度、压力）作为辅助目标绑定到特定 Latent Dimensions。
2. **探索动态 ODIN**：处理时间序列数据，捕捉演化模式。
3. **与其他生成模型结合**：如将 ODIN 作为 VAE 的先验结构。
4. **应用于更多科学领域**：如基因组学、气候建模、材料表征等。

---

> **总结**：  
> ODIN 提出了一种**架构驱动**（而非损失驱动）的方法，通过 **Dendritic Decoding + Orthogonality Loss**，首次在深度 Autoencoder 中实现了 **可复现、可解释、有序正交的 Latent Space**。它不仅在理论上连接了 PCA 与深度学习，在实践中也展现出强大的科学数据分析潜力，为 **Interpretable ML** 和 **Scientific Discovery** 提供了新工具。

</details>

---

### 14. [Foundation Models for Automatic CAD Generation](https://arxiv.org/abs/2607.05573)

**Authors**: J de Curt\`o, Victoria Guill\'en, I. de Zarz\`a  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.05573v1  

#### Abstract
Recent advances in Large Language Models (LLMs) and Vision-Language Models (VLMs) enable the automatic generation of parametric 3D designs from natural-language specifications. This chapter presents an empirical study of foundation models for automatic Computer-Aided Design (CAD) generation of mecha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Foundation Models for Automatic CAD Generation》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文聚焦于**自动化的参数化三维机械零件设计生成**，旨在通过自然语言描述直接生成可制造的、拓扑正确的 CAD 模型。传统 CAD 设计依赖专家手动建模或脚本编程，门槛高且难以自动化。本文探索如何利用现代 **Foundation Models**（特别是 LLM 和 VLM）实现从文本到 CAD 的端到端生成，并系统评估其可靠性与局限性。

---

### **提出的新方法与新思路**

作者提出了 **LLMForge** —— 一个统一的多模型 text-to-CAD 框架，包含以下两个核心创新的**迭代批判机制（critique regimes）**：

#### ✅ **(1) IterTracer**
- 利用 **Phong-shaded ray-trace 渲染器 + 分析性视觉度量** 提供轻量级、几何感知的反馈。
- 包括：**silhouette IoU、hole visibility、edge clearance、aspect-ratio conformance** 等指标。
- 优势：无需额外神经网络，速度快、确定性强、延迟低（<1秒），适合快速纠错。

#### ✅ **(2) IterVision**
- 引入 **Vision-Language Model（VLM）作为语义批评者（Qwen2.5-VL-72B）**，替代分析性评分器。
- VLM 基于 chain-of-thought 推理对渲染视图进行多模态理解，判断空间一致性与设计意图是否匹配。
- 输出结构化 JSON 反馈：`missing_features`, `incorrect_features`, `geometry_issues`, `suggestions`。
- 优势：能捕捉 LLM 和几何引擎无法识别的**高层语义错误**，如旋转对称特征误解、装配逻辑缺失等。

此外，框架整合了：
- **JSON-schema 验证**：确保输出格式正确；
- **解析-合成-网格化流程**：将 JSON 转为 watertight mesh；
- **最多四轮迭代优化循环**：结合反馈持续改进生成结果。

---

### **相比现有方法的优势**

| 方面 | 传统方法 / 其他研究 | 本文方法（LLMForge） |
|------|---------------------|------------------------|
| **评估维度** | 单一 pass/fail 或仅代码语法检查 | 多轴评分：schema、mesh、feature、visual、VLM（五维） |
| **反馈机制** | 手动调试或基于执行报错自修正 | 自动化、结构化、多轮迭代反馈（含视觉+语义） |
| **模型通用性** | 多数研究聚焦单一 LLM（如 GPT-4） | 系统比较 **7 种主流 Foundation Models**，具备可复现性 |
| **工业适用性** | 缺乏标准化 benchmark | 构建了包含 **97 个工程问题的基准集**，覆盖典型零件族 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 构建了一个名为 **“Benchmark of 97 Engineering Design Problems”** 的专用测试集。
- 问题来源于四大类典型机械部件：
  1. **Rectangular plates with holes and bolt circles**（带孔和螺栓圈的矩形板）
  2. **Multi-feature boxes**（多功能箱体）
  3. **Flanged cylinders**（带法兰圆柱）
  4. **L-brackets**（L 形支架）
- 每个问题提供自然语言描述 + 对应的 ground-truth feature specification（用于计算 feature score）。

---

### **实验设置**

#### **模型列表（共7个）**
在 Nebius AI Studio 平台上评测以下模型：
- DeepSeek-V3.2
- Qwen3-235B-A22B
- Llama-3.3-70B
- Gemma-3-27B
- GLM-4.5
- MiniMax-M2.1
- INTELLECT

#### **统一配置**
- 温度：T = 0.15
- 最大输出 token 数：2048
- 迭代轮次上限：R = 3（round 0 ~ 3）
- 冷却时间：5s（避免速率限制）
- Prompt 格式强制返回纯 JSON，无 markdown 包裹

#### **几何引擎**
使用 Python 实现的确定性转换器（基于 Trimesh 和 Shapely）将 JSON spec 转换为 watertight triangulated mesh。

---

### **评估指标（Scoring Axes）**

每个生成件按五个独立维度打分：

| 维度 | 符号 | 描述 |
|------|------|------|
| **Validation Score** | $ s_{val} \in [0,1] $ | JSON schema 合规性（必填字段、类型正确） |
| **Mesh Score** | $ s_{mesh} \in \{0,1\} $ | 是否成功生成非空、无异常的 mesh |
| **Feature Score** | $ s_{feat} \in [0,1] $ | 孔数量、尺寸、倒角/圆角等是否符合规范（部分得分） |
| **Visual Score** | $ s_{vis} \in [0,1] $ | 渲染图像分析：<br>- Silhouette IoU<br>- Hole visibility<br>- Edge clearance<br>- Aspect ratio<br>- Cross-section consistency |
| **VLM Semantic Match** | $ s_{vlm} \in [0,1] $ | （仅 IterVision）由 Qwen2.5-VL-72B 给出的语义匹配度 |

#### **综合得分公式**

- **IterTracer（无 VLM）：**
  $$
  s_{overall} = 0.25s_{val} + 0.15s_{mesh} + 0.20s_{feat} + 0.40s_{vis}
  $$

- **IterVision（含 VLM）：**
  $$
  s_{overall} = 0.20s_{val} + 0.10s_{mesh} + 0.20s_{feat} + 0.30s_{vis} + 0.20s_{vlm}
  $$

> 注：VLM 的引入降低了 visual axis 权重，增加了语义维度。

---

### **基线方法对比**
本文未采用传统 CAD 工具链作为基线，而是以不同 Foundation Models 在相同 pipeline 下的表现互为对照，形成**横向比较体系**。尤其关注：
- 小模型 vs 大模型（如 Gemma-3-27B vs Qwen3-235B）
- 指令调优程度的影响
- 是否支持结构化输出能力

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| Model | IterTracer $ s_{overall} $ | Mesh Success Rate | IterVision $ s_{overall} $ | Mesh SR (IterVision) | $ s_{vlm} $ |
|-------|-------------------------------|--------------------|------------------------------|------------------------|-------------|
| **DeepSeek-V3.2** | **0.890** | 98.97% | 0.850 | 97.9% | **0.625** |
| **Qwen3-235B-A22B** | 0.889 | 99.0% | 0.850 | 99.0% | 0.522 |
| **Llama-3.3-70B** | 0.885 | 99.0% | 0.846 | 99.0% | 0.517 |
| **Gemma-3-27B** | 0.885 | 99.0% | **0.842** | **100%** | 0.511 |
| GLM-4.5 | 0.678 | 69.1% | 0.561 | 54.6% | 0.564 |
| MiniMax-M2.1 | 0.575 | 49.5% | 0.552 | 48.5% | 0.575 |
| INTELLECT | 0.411 | 26.8% | 0.401 | 28.9% | 0.607 |

---

### **核心发现与对比结果**

#### ✅ **顶级模型表现高度接近（饱和现象）**
- 在 **IterTracer** 下，前四名（DeepSeek, Qwen3, Llama3, Gemma3）$ s_{overall} \in [0.885, 0.890] $，标准差极小（σ ≤ 0.074），表明该任务在当前设定下已趋于**性能饱和**。
- 四者 mesh 成功率均达 **98.97%**，说明现代指令调优 LLM 已能在结构约束下稳定生成有效 CAD。

#### ✅ **VLM 批判提升挑战性，暴露残余差异**
- 加入 VLM 后，所有顶级模型平均下降约 **0.04 分**，显示 VLM 提供更严格、更具语义深度的评估。
- **Gemma-3-27B 实现 100% watertight mesh 成功率**（97/97），是唯一达成此成就的模型。
- 尽管 Gemma mesh 表现最佳，其 $ s_{vlm} = 0.511 $ 是前四中最低，说明**高几何成功率 ≠ 高语义一致性**。

#### ✅ **VLM 与 Analytic Metrics 存在正交信号**
- 所有模型池化后的 Pearson 相关系数显示：
  - $ r(s_{vis}, s_{overall}) = 0.959 $
  - $ r(s_{vlm}, s_{overall}) = 0.776 $
- 表明 VLM 抓住了 analytic visual metrics 无法反映的**语义偏差**，验证其补充价值。

#### ✅ **收敛行为反转：批判类型决定优化深度**
- **IterTracer**：顶级模型大多在第1轮即达峰值（62–83%问题），后续略有退步 → **一次修正即饱和**
- **IterVision**：多数问题（53–55%）的最佳得分出现在第2或第3轮 → **VLM 支持深层搜索与持续优化**

#### ❌ **弱模型存在显著失败模式**
- **GLM-4.5**：有一定 mesh 成功率（~69%），但 variance 高 → **间歇性错误**（如尺寸错、schema 错）
- **MiniMax & INTELLECT**：mesh 成功率 <50%，常因非法参数导致几何引擎崩溃 → **根本性建模失败**
- **INTELLECT 特异现象**：$ s_{vlm} $ 从 0.60 升至 0.82，远超其他模型，但 mesh 成功率垫底 → 显示“**视觉-几何解耦**”：图像看起来合理，实际拓扑无效

---

## **4. 关键结论和发现**

### **主要发现**

1. ✅ **现代 Foundation Models 可靠地完成简单参数化 CAD 生成任务**  
   在四类标准零件上，顶尖 LLM 已实现近 99% 的 mesh 成功率，证明 text-to-CAD 在工业场景中具备实用潜力。

2. ✅ **紧凑模型可媲美更大模型**  
   如 Gemma-3-27B（27B 参数）表现不逊于 Qwen3-235B（235B），说明**高质量指令微调比模型规模更重要**。

3. ✅ **VLM 作为“语义裁判”具有独特价值**  
   它能识别 analytic metrics 忽略的设计意图偏差，尤其是在复杂对称结构（如 cylinder）中更为敏感。

4. ✅ **cylinder 类最难，存在“cylinder gap”**  
   所有模型在此类别得分低 0.04–0.07，原因包括：
   - 缺少平面 hole 特征 → 视觉反馈信息不足
   - VLM 对旋转对称组件训练数据不足 → 评分保守

5. ✅ **迭代策略需依批判类型调整**  
   - 若只用 analytic feedback → 1–2 轮足够
   - 若启用 VLM → 应运行更多轮以充分利用语义建议

---

### **局限性**

1. **范围受限**：仅涵盖四种规则几何体，未涉及自由曲面、多体装配、运动机构等复杂 CAD 场景。
2. **依赖特定 backend**：全部实验基于 Nebius AI Studio，可能存在平台偏倚。
3. **VLM 评分具随机性**：尽管设 T=0.05 控制，仍存在不可消除的 stochastic variance。
4. **未微调模型**：所有模型均为 off-the-shelf，未针对 CAD 任务做 fine-tuning 或 retrieval augmentation。

---

### **未来工作方向**

1. **扩展 benchmark**：
   - 增加自由曲面（freeform surfaces）
   - 多体装配（multi-body assemblies）
   - 制造约束（GD&T tolerances, material specs）

2. **引入物理仿真闭环**：
   - 用 FEA（Finite Element Analysis）替代图像渲染作为反馈源
   - 实现 “simulation-in-the-loop” 自动生成可验证结构

3. **模型微调方向**：
   - 基于结构化反馈信号 fine-tune LLM，缩小高低梯队差距
   - 构建专门的 CAD-oriented pretraining corpus

4. **部署优化策略**：
   - 提出两阶段 pipeline：先用 IterTracer 快速筛选，再对关键部件启用 IterVision 精修
   - 降低 token 开销，提升工业落地效率

---

> 🔗 **代码开源地址**：https://github.com/drdecurto/LLMforge  
> 包含完整 benchmark 数据、geometry engine、evaluation notebook、结果文件及复现指南。

</details>

---

### 15. [TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training](https://arxiv.org/abs/2607.05804)

**Authors**: Yuhang Zhou, Kai Zheng, Haoling Li, Dengyun Peng, Can Xu, Jingjing Chen  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.05804v1  

#### Abstract
On-policy distillation (OPD) trains a student policy by matching a stronger teacher on the student's own trajectories, offering a promising framework for language agent training. However, its application to long-horizon agentic tasks remains insufficiently explored. We identify two key inefficiencie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **On-Policy Distillation (OPD)** 在 **long-horizon agent tasks** 中存在的两个关键效率瓶颈进行分析与优化：

1. **外部不匹配（External Mismatch）**  
   - 固定长度的 full-horizon rollout 导致在尾部轮次（tail turns）浪费大量计算资源，因为这些步骤提供的 KL 监督信号弱且噪声大。

2. **内部不匹配（Internal Mismatch）**  
   - 轨迹级（trajectory-level）KL 损失函数倾向于将大部分损失集中在浅层 token 上，导致深层决策轮次（deep turns）训练不足，尤其当初始行为已对齐后。

### 🚀 提出的新方法：TurnOPD
提出 **TurnOPD** —— 一种面向 **turn-level** 的预算控制策略，通过两个控制器实现高效的 on-policy distillation：

1. **Adaptive Rollout-Depth Budgeting（自适应 rollout 深度控制）**  
   - 动态决定每条轨迹应收集到第几轮（turn），避免无效尾部计算。
   - 基于探针（probe）统计信息，结合两个信号：
     - `Heff`：有效监督质量中心（survivor-weighted KL centroid）
     - `Hcov`：成功完成轨迹的覆盖率下界（80% 分位数）
   - 最终 rollout 长度为两者最大值，并通过 EMA 平滑更新。

2. **Progressive Turn-Normalized Loss Budgeting（渐进式轮次归一化损失分配）**  
   - 渐进地从 token-level 损失加权过渡到 turn-balanced 损失加权。
   - 使用线性插值混合两种归一化方式：
     $$
     \mathcal{L}_{\text{blend}} = (1-\alpha)\mathcal{L}_{\text{token}} + \alpha\mathcal{L}_{\text{round}}
     $$
   - $\alpha$ 随训练进度从 0 线性增长至 1，早期关注 token 分布，后期均衡各轮次监督。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **训练效率** | 显著减少 wall-clock 时间（最高提速 **2.29×**） |
| **模型性能** | 在相同时间预算下达到更高准确率，甚至超越教师模型 |
| **泛化能力** | 在多种 long-horizon agent 任务上均表现稳定提升 |
| **机制合理性** | 引入“contamination-compression”解释为何深层 KL 信号衰减 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在三个代表性的多轮 agent 基准上验证：

| 数据集 | 任务类型 | 特点 |
|--------|--------|------|
| **ALFWorld** | Embodied Planning（具身规划） | 文本环境中执行复杂家庭任务（如清洁、加热、放置物品） |
| **WebShop** | Web Navigation（网页导航） | 接地电商环境中的商品搜索与购买流程 |
| **Multi-Hop Search** | 多跳问答 | 包含 PopQA, NQ, 2Wiki, HotpotQA，需多次检索推理 |

### ⚙️ 实验设置
- **学生-教师对**：
  - ALFWorld: Qwen3-1.7B/4B ← Qwen3-8B-GRPO
  - WebShop: Qwen3-1.7B ← Qwen3-8B-GRPO
  - Multi-Hop Search: Qwen3.5-2B ← Qwen3.5-9B-GRPO
- **训练步数**：统一运行 100 optimizer steps
- **硬件配置**：32 × NVIDIA H20 GPU（每卡 80GB）

### 📊 评估指标
采用双重视角公平比较：

| 指标 | 定义 |
|------|------|
| **Least-Time Avg@4** | 所有方法中任一达到 100 步所需的最短 wall-clock 时间内取得的平均准确率（最后4次 eval 的 Avg@4） |
| **Same-Step Avg@4** | 所有方法训练满 100 步后的最终性能 |
| **Wall-clock Time** | 实际训练耗时（小时） |
| **Speedup** | 相对于 vanilla OPD 的加速比 |

### 🆚 基线方法
| 方法 | 描述 |
|------|------|
| **Vanilla OPD** | 标准 on-policy distillation，固定 rollout 长度 + token-level KL 归一化 |
| **TCOD-F2B** | 引入按轨迹长度递增的课程学习（temporal curriculum） |
| **TurnOPD (Ours)** | 自适应深度 + 渐进式 loss 归一化 |

---

## 3. 主要实验结果和性能指标

### 📈 性能汇总（Table 2 & Figure 5）

| Task | Student | Method | **Least-Time Avg@4** | **Same-Step Avg@4** | **Wall Time (h)** | **Speedup** |
|------|---------|--------|------------------------|----------------------|--------------------|-------------|
| ALFWorld | 1.7B | Vanilla OPD | 73.52 | 83.00 | 4.42 | 1.00× |
|          |       | TCOD-F2B   | 80.06 | 80.06 | 1.87 | **2.37×** |
|          |       | **TurnOPD** | **85.60** | **86.29** | **1.93** | **2.29×** |
| ALFWorld | 4B   | **TurnOPD** | **91.73** | **92.21** | 2.16 | 1.33× |
|          |      | (Teacher) | 90.75 | 90.75 | – | – |
| WebShop | 1.7B | Vanilla OPD | 76.98 | 81.65 | 1.57 | 1.00× |
|         |      | TCOD-F2B   | 80.45 | 81.66 | 1.33 | 1.18× |
|         |      | **TurnOPD** | **82.80** | **82.80** | **1.24** | **1.26×** |
| Multi-Hop Search | 2B | Vanilla OPD | 45.77 | 47.82 | 4.45 | 1.00× |
|                  |    | TCOD-F2B   | 45.64 | 47.77 | 3.80 | 1.17× |
|                  |    | **TurnOPD** | **47.24** | 47.24 | **2.94** | **1.51×** |

> ✅ **关键发现**：
> - TurnOPD 在所有任务上均取得 **最佳 Least-Time 准确率**
> - 在 ALFWorld-4B 上，**学生模型性能超过教师模型**
> - 训练速度提升显著，**最快达 2.29× 加速**

### 🔬 消融实验（Ablation Study on ALFWorld-1.7B）

| 配置 | Same-Step Avg@4 | Wall Time (h) |
|------|------------------|----------------|
| Vanilla OPD | 83.0 | 4.42 |
| + Adaptive Depth | 82.8 | **1.96** |
| + Linear Blend Norm | 85.1 | 2.59 |
| **TurnOPD (Full)** | **86.3** | **1.93** |

> 🔍 **结论**：
> - **Adaptive Depth** 是效率主因，大幅降低计算开销，但单独使用会轻微降低精度
> - **Linear Blend Norm** 改善深层轮次训练，提升精度但成本较高
> - **二者结合** 实现最优 accuracy-time tradeoff

### 📉 KL 损失分配分析（Table 6）
| KL Normalization | Same-Step Avg@4 | Deep Turn Budget (Late) |
|------------------|----------------|--------------------------|
| Trajectory-level KL | 83.0 | 1.2% |
| Hard Turn-level KL | 85.0 | 31.9% |
| **Linear Blend (Ours)** | **85.1** | **27.7%** |

> ✅ **优势**：渐进式 blend 在稳定性与深度监督之间取得平衡，避免 abrupt shift。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **OPD 中存在严重的监督信号分配不均问题**：
   - 浅层轮次主导 KL 损失（ALFWorld 中前3轮占近50%）
   - 深层关键决策轮次仅获得极小比例损失预算（深三分之一仅得 3.6–4.5%）

2. **引入 contamination-compression 机制解释 KL 衰减现象**：
   - 学生生成上下文越长，强制预测成分（context-forced mass）越高
   - 即使策略仍有差异，KL 也会被压缩至低位，造成“虚假收敛”

3. **TurnOPD 成功解决了双重不匹配问题**：
   - 外部控制器动态裁剪 rollout 深度，节省算力
   - 内部控制器渐进增强深层轮次监督权重，改善优化路径

4. **TurnOPD 实现 accuracy-time frontier 的全面领先**：
   - 更快收敛
   - 更高最终性能
   - 更优中间结果

### ⚠️ 局限性
- 依赖 periodic full-depth probing 来估计 `Hcov` 和 `Heff`，增加少量额外开销
- `Hcov` 依赖 success label，在无明确奖励信号的任务中可能受限（但实验证明可用全轨迹 CDF 替代）
- 当前方法聚焦于 KL 监督结构，未整合 reward 或 preference modeling

### 🔮 未来工作方向
- 将 turn-aware 思想扩展至 **Off-policy RL** 或 **Preference Optimization**
- 探索更轻量化的 probe 机制（如稀疏采样）
- 结合 curriculum learning 与 TurnOPD 进行联合调度
- 应用于更复杂的 real-world agent 场景（如 OSWorld, AndroidWorld）

---

## ✅ 总结
**TurnOPD** 是首个系统性从 **turn-level 视角**重构 OPD 的方法，揭示了 long-horizon agent 训练中监督信号与优化预算的结构性错配问题。其提出的 **双预算控制器**（adaptive depth + progressive loss blending）不仅提升了训练效率（最高 **2.29× 加速**），还在多个复杂任务上实现了 **学生超越教师** 的惊人效果。

> 🎯 **一句话总结**：  
> *TurnOPD 表明，在 long-horizon agent 训练中，“何时停止 rollout” 和 “如何分配 loss” 同样重要 —— 监督的基本单位不应是 token，而是嵌入交互轨迹中的 turn-conditioned decision.*

</details>

---

### 16. [Auto-DSM Under the Lens: A Black-Box Evaluation Framework for LLM-Based DSM Generation](https://arxiv.org/abs/2607.05985)

**Authors**: Niels Potters, Theo Hofman  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.05985v1  

#### Abstract
This paper presents a black-box evaluation framework to systematically assess the ability of Large Language Models (LLMs) to generate Design Structure Matrices (DSMs) from structured technical documentation. Motivated by the closed-source nature of current Auto-DSM pipelines, the framework introduce...

---

### 17. [From Application-Layer Simulation to Native Meta-Architecture: Structural Tension as an Endogenous Driver for Heterogeneous AI Evolution](https://arxiv.org/abs/2607.06269)

**Authors**: Heting Mao  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.06269v1  

#### Abstract
Current large language models (LLMs) are fundamentally stateless: their behavior is fully determined by input at inference time, and any higher-order cognitive architecture must be simulated at the application layer through prompt engineering and context management. This paper proposes a theoretical...

---

### 18. [DepthWeave-KV: Token-Adaptive Cross-Layer Residual Factorization for Long-Context KV Cache Compression](https://arxiv.org/abs/2607.06523)

**Authors**: Anna Cordoba, Adam Puente Tercero, Nerea Angulo Hijo, Mar Linares Tercero, Julia Barrientos, Ainhoa Miranda, Jesus Olivera  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.06523v1  

#### Abstract
Long-context language model inference is increasingly limited by the memory bandwidth and capacity required to store key-value caches, yet existing compression methods often apply uniform budgets across layers or tokens and degrade retrieval when lexical cues and semantic states require different pr...

---

### 19. [The Large Cancer Assistant (LCA): A Model-Agnostic Orchestration Framework for Scalable Clinical Decision Support in Oncology](https://arxiv.org/abs/2607.06531)

**Authors**: Ghassen Marrakchi, Basarab Matei  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.06531v1  

#### Abstract
- Objective: Multimodal deep learning models in oncology are currently limited by monolithic designs that rigidly couple data ingestion, clinical routing, and artificial intelligence (AI) inference. To address this inflexibility, we propose the Large Cancer Assistant (LCA), a model-agnostic, post-ho...

---

### 20. [SpanUQ: Span-Level Uncertainty Quantification for Large Language Model Generation](https://arxiv.org/abs/2607.05721)

**Authors**: Yimeng Zhang, Yingying Zhuang, Ziyi Wang, Yuxuan Lu, Pei Chen, Aman Gupta, Zhe Su, Ming Tan, Zhilin Zhang, Qun Liu, Manikandarajan Ramanathan, Rajashekar Maragoud, Edward Vul, Jing Huang, Dakuo Wang  
**Category**: cs.CL  
**Published**: 2026-07-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.05721v1  

#### Abstract
Uncertainty estimation is essential not only for the trustworthy deployment of large language models (LLMs) but also as a foundation for self-refinement in LLM generation. However, existing approaches operate at suboptimal granularities: token-level scores lack semantic coherence, while sequence-lev...

---

### 21. [Scalable Perturbation Learning for Online Self-Supervised Echo State Networks](https://arxiv.org/abs/2607.06079)

**Authors**: Taiki Yamada, Kantaro Fujiwara  
**Category**: cs.LG  
**Published**: 2026-07-08  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.06079v1  

#### Abstract
Intelligent systems should not only solve tasks but also adapt under real-world constraints. Autonomous adaptation via self-supervised learning, sequential adaptation via online learning, and memory-efficient implementation via perturbation-based learning are important requirements for such systems....

---

### 22. [Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding](https://arxiv.org/abs/2607.05722)

**Authors**: Yonggan Fu, Lexington Whalen, Abhinav Garg, Chengyue Wu, Maksim Khadkevich, Nicolai Oswald, Enze Xie, Daniel Egert, Sharath Turuvekere Sreenivas, Shizhe Diao, Chenhan Yu, Ye Yu, Weijia Chen, Sajad Norouzi, Jingyu Liu, Shiyi Lan, Ligeng Zhu, Jin Wang, Jindong Jiang, Morteza Mardani, Mehran Maghoumi, Song Han, Ante Juki\'c, Nima Tajbakhsh, Jan Kautz, Pavlo Molchanov  
**Category**: cs.CL  
**Published**: 2026-07-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.05722v1  

#### Abstract
We introduce Nemotron-Labs-Diffusion, a tri-mode language model (LM) that unifies AR, diffusion, and self-speculation decoding within a single architecture. Trained with a joint AR-diffusion objective, Nemotron-Labs-Diffusion can switch modes to sustain high throughput across deployment settings and...

---

### 23. [Prompting Complexity: Shortest Prompts for Texts and Behaviors in LLMs](https://arxiv.org/abs/2607.06145)

**Authors**: Adrian Cosma  
**Category**: cs.CL  
**Published**: 2026-07-08  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.06145v1  

#### Abstract
In this paper, we define the quantity of prompting complexity: for a fixed instruction-tuned language model, what is the shortest plausible prompt that makes deterministic decoding produce a target text? It is an LM-relative analogue of resource-bounded Kolmogorov complexity: the prompt is a program...

---

### 24. [CSTutorBench: Benchmarking Small Language Models as Tutors for Block-Based Programming](https://arxiv.org/abs/2607.05571)

**Authors**: H. Chad Lane, Bryson Kageler  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.05571v1  

#### Abstract
Large language models are increasingly explored as AI tutors, yet deploying them in K-12 settings raises concerns around privacy, cost, and reliance on proprietary models. Small language models (SLMs) offer a promising alternative, but selecting the right model for a specific educational context rem...

---

### 25. [Beyond the Leaderboard: A Synthesis of Tool-Use, Planning, and Reasoning Failures in Large Language Model Agents](https://arxiv.org/abs/2607.05775)

**Authors**: Wael Albayaydh, Rui Zhao, Ivan Flechais  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.05775v1  

#### Abstract
Large language model (LLM) agents are increasingly evaluated on their ability to use tools, plan multi-step tasks, coordinate with other agents, and operate over extended horizons. Reported benchmark gains often obscure recurring failure modes documented across otherwise unrelated evaluation efforts...

---

### 26. [From Passive Retrieval to Active Memory Navigation: Learning to Use Memory as a Structured Action Space](https://arxiv.org/abs/2607.05794)

**Authors**: Yue Xu, Yutao Sun, Yihao Liu, Mengyu Zhou, Jiayi Qiao, Lu Ma, Kai Tang, Wenjie Wang, Xiaoxi Jiang, Guanjun Jiang  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.05794v1  

#### Abstract
Long-term user memory is essential for personalized conversational agents, yet many memory systems still expose memory through passive retrieval interfaces, making the model a consumer of pre-selected evidence. We introduce NapMem, a framework for learning to use long-term user memory as a structure...

---

### 27. [Information Gain-based Rollout Policy Optimization: An Adaptive Tree-Structured Rollout Approach for Multi-Turn LLM Agents](https://arxiv.org/abs/2607.06223)

**Authors**: Yijun Zhang, Fan Xu, Jiaxin Ding, Yule Xie, Shiqing Gao, Xin Ding, Haoxiang Zhang, Luoyi Fu, Xinbing Wang  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.06223v1  

#### Abstract
Reinforcement learning has become a promising paradigm for improving large language model (LLM) agents on long-horizon search tasks, where the agent must make a sequence of intermediate decisions before receiving a final outcome. However, existing methods still face a key limitation: the rollout bud...

---

### 28. [Demonstrating TOFFEE: A Learned System for Synthesizing Data Agent Trajectories at Scale](https://arxiv.org/abs/2607.06233)

**Authors**: Ziting Wang, Yin Li, Zuhao Yang, Xiuchang Li, Jiale Bai, Gao Cong  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.06233v1  

#### Abstract
LLM-powered data agents are playing an increasingly important role in data-driven decision making. However, existing data agents struggle to generalize to unseen data environments and analytical workflows, especially in heterogeneous enterprise settings. This creates a growing need for synthesizing ...

---

### 29. [Task Decomposition-Guided Reranking for Adaptive Agent Skill Retrieval](https://arxiv.org/abs/2607.06283)

**Authors**: Yanping Chen, Weijie Shi, Wen Yang, Jiajie Xu  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.06283v1  

#### Abstract
Skill usage can significantly enhance the ability of modern agent systems to complete complex tasks. However, the growing scale of skill libraries makes accurate skill selection increasingly challenging. In real-world scenarios, ambiguous semantic matching often arises between a specific task requir...

---

### 30. [Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade](https://arxiv.org/abs/2607.06503)

**Authors**: Kai Ruan, Zihe Huang, Ziqi Zhou, Qianshan Wei, Xuan Wang, Hao Sun  
**Category**: cs.AI  
**Published**: 2026-07-08  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.06503v1  

#### Abstract
Large language model (LLM) agents solving multi-step tasks frequently commit to trajectories that are doomed to fail, yet continue to consume substantial inference compute before the failure becomes observable. We show that failure is predictable early from the agent's internal representations: ligh...

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

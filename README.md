# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-04 10:00:57 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Margins, Not Windows: Training-Free Per-Step Lossy Speculative Decoding](https://arxiv.org/abs/2609.02897)

**Authors**: Oszk\'ar Urb\'an, Young D. Kwon, Stylianos I. Venieris, Cecilia Mascolo  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2609.02897v1  

#### Abstract
Speculative decoding accelerates LLM inference by drafting candidate tokens and verifying them in parallel. Tree-attention drafters such as EAGLE-3 are widely adopted, yet typically hold two decisions fixed: (1) a strict token-match verification rule and (2) a static draft-tree shape. Prior work rel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Margins, Not Windows: Training-Free Per-Step Lossy Speculative Decoding**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 **Speculative Decoding**（推测解码）方法在加速大语言模型（LLM）推理时面临两个关键限制：
1. **验证规则过于严格**：传统方法仅接受与目标模型输出完全匹配的 draft token，忽略了语义上等价但形式不同的 token，导致接受率低。
2. **静态 draft tree 结构**：大多数方法（如 EAGLE-3）使用固定的 draft tree 形状（深度、宽度），无法根据每一步的置信度动态调整，造成计算资源浪费。

此外，已有改进方法存在局限：
- **FLy** 使用 lookahead window 判断语义等价，但依赖长 draft chain，在短接受序列中失效。
- **TALON** 虽能动态调整 tree shape，但受限于固定总 token 预算，只能重新分配而非真正调节计算量。

---

### **提出的新方法：AdaptiveSpec**
本文提出 **AdaptiveSpec**，一种无需训练、逐步骤自适应的推测解码框架，从两个正交维度进行优化：

#### ✅ **(1) Margin-based Lossy Verification（基于概率边距的有损验证）**
- 在第一个 mismatch 位置，若目标模型对 draft token 的概率与其 top-1 概率的比值超过阈值 $ K $，则仍可接受该 token：
  $$
  \text{margin}(j) = \frac{P_{\text{target}}(\text{draft})}{P_{\text{target}}(\text{top1})}
  $$
- **优势**：
  - 无需额外训练模块或 lookahead window。
  - 不依赖 draft chain 长度，适用于任意 drafter 架构。
  - 实现“近似正确即接受”，提升 acceptance rate。

#### ✅ **(2) Dynamic Draft Tree-Shaping（动态 draft tree 构建）**
- 引入 **Draft Confidence Score (DCS)** 综合考虑：
  - $ P_{\text{draft}}(\text{top1}) $：draft 模型自身置信度
  - RAR（Rolling Acceptance Rate）：近期接受历史的指数移动平均
  $$
  \text{DCS} = P_{\text{draft}}(\text{top1}) \cdot \text{RAR}
  $$
- 根据 DCS 动态选择 `(nsteps, top-k, ndt)` 三元组：
  - 高 DCS → 深窄链（deep & narrow）
  - 低 DCS → 浅宽树（shallow & wide）
- **优势**：
  - 真正调整每步计算量，而非仅在固定预算内重分配。
  - 与 SGLang 生产级引擎兼容（通过预捕获 CUDA graphs 支持动态切换）。

---

### **相比现有方法的优势**
| 方法 | 是否需训练 | 是否支持动态 tree | 是否支持 lossy verification | 是否依赖 window |
|------|------------|-------------------|------------------------------|---------------|
| EAGLE-3 | ❌ | ❌ | ❌ | ❌ |
| TALON | ❌ | ✅（受限） | ❌ | ❌ |
| FLy | ❌ | ❌ | ✅ | ✅ |
| **AdaptiveSpec** | ✅（无） | ✅（自由） | ✅ | ❌ |

> ✔️ **统一框架**：将上述两个组件结合，增益叠加，实现端到端吞吐量显著提升。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **GSM8K**：小学数学应用题，测试推理能力
- **MATH-500**：高等数学问题，更具挑战性
- **HumanEval**：代码生成任务，评估编程能力

---

### **实验设置**
- **目标模型（Target Models）**：
  - `Llama-3.1-8B-Instruct`
  - `DeepSeek-R1-Distill-Llama-8B`
  - `Qwen3-8B`
- 所有模型均搭配其公开发布的 **EAGLE-3** draft model。
- **推理引擎**：基于 **SGLang** 实现，支持 CUDA graphs 加速。
- **硬件环境**：单张 NVIDIA A100 GPU，batch size = 1。
- **解码方式**：greedy decoding（temperature=0）。

---

### **评估指标**
| 指标 | 含义 |
|------|------|
| **Speedup** | 相对于 vanilla autoregressive 推理的速度提升倍数 |
| **Mean Accepted Tokens (T)** | 每次 verify 步骤平均接受的 draft token 数量 |
| **Task Accuracy Recovery (%)** | 相对于 lossless EAGLE-3 的任务准确率保留比例（用于 lossy 方法） |

---

### **基线方法对比**
| 基线 | 类型 | 说明 |
|------|------|------|
| **EAGLE-3** | Static + Lossless | 固定 tree shape，精确匹配验证 |
| **TALON*** | Dynamic + Lossless | 在 SGLang 中复现，budget-constrained 自适应 tree |
| **FLy*** | Static + Lossy | 在 SGLang 中复现，lookahead window 判断语义等价 |

> *注：TALON 和 FLy 原始未集成进 SGLang，作者进行了适配实现以公平比较。*

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| 方法 | 平均 Speedup ↑ | 最高 Speedup | Accuracy Recovery (%) ↑ |
|------|----------------|-------------|--------------------------|
| EAGLE-3 (Baseline) | ~1.82× | 2.28× | 100% |
| TALON* | ~2.07× | 2.88× | 100% |
| FLy* | ~2.00× | 3.07× | ~92% |
| **AdaptiveSpec (Combined)** | **~2.44×** | **3.20×** | **93–109%** |

> 🔺 **峰值提升达 56%**（Llama-3.1-8B on HumanEval：1.81× → 2.82×）

---

### **与基线方法的对比结果**
- **Dynamic-only 版本**（仅动态 tree）：
  - 平均提速 **+21.4%**（1.82× → 2.21×）
  - 完全保留准确率（100% recovery）
- **Lossy-only 版本**（仅 margin 验证）：
  - 平均提速 **+20.9%**（1.82× → 2.20×）
  - 准确率保留 **92%**
- **Combined 版本**（双策略联合）：
  - 平均提速 **+34%**（相对 baseline），最高 **+56%**
  - 多数任务恢复至 **≥93% 准确率**，部分甚至超过 baseline（如 103%）

> 💡 **特别优势**：在原本 lossy 表现差的任务上（如 Llama-3.1-8B on MATH-500），combined 策略通过提供更多候选路径，反而提升了 accuracy recovery（92% → 96%）。

---

### **消融实验结果（Table 2）**
| 配置 | 平均 Speedup | T（平均接受长度） | Recovery |
|------|--------------|--------------------|---------|
| Static × Strict | 1.82× | 2.59 | 100% |
| Dynamic × Strict | 2.21× | 3.61 | 100% |
| Static × Lossy | 2.20× | 3.99 | 92% |
| **Dynamic × Lossy (AdaptiveSpec)** | **2.44×** | **4.10** | **98%** |

> ✅ 两项改进独立有效，且**增益可叠加**，证明二者作用于不同轴，互不冲突。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Margin-based verification 更鲁棒**：
   - 直接读取目标模型在 mismatch 位置的概率分布，避免了 FLy 对 lookahead window 的依赖。
   - 尤其适合 acceptance length 较短的场景（如 EAGLE-3 平均仅 2–3 tokens）。

2. **Dynamic tree-shaping 显著提升效率**：
   - 通过 DCS 信号智能调节 draft compute，强信心步扩展深度，弱信心步增加宽度。
   - 真正实现了 per-step 计算资源优化，而非简单重分配。

3. **两种机制协同增效**：
   - 动态 tree 提供更多合理候选路径，配合宽松验证器，进一步释放吞吐潜力。
   - 即使是有损方法，也能在多数任务上接近甚至超越原始准确率。

---

### **方法的局限性**
1. **不保证理论上的分布一致性**：
   - Margin rule 不满足 Leviathan et al. (2023) 定义的形式化 lossless 条件，准确性为经验性保留。
2. **仅在 batch size=1 下验证**：
   - 当前收益最大化的场景是 latency-bound 的小批量推理，多 batch 场景有待验证。
3. **目前仅适配 EAGLE-3 架构**：
   - 虽然原理可推广至其他 drafter（如 DDTree），尚未实证。

---

### **未来工作方向**
1. **拓展至非语言领域**：
   - 如 vision-language-action 模型（robotics）、语音生成等自回归架构，均可受益于 per-step 自适应推测。
2. **结合 retrieval-based drafting**：
   - 在 agentive workloads 中利用历史输出缓存作为 draft source，并用 AdaptiveSpec 机制优化验证与结构。
3. **探索更细粒度的 margin 规则**：
   - 引入上下文感知的 adaptive threshold $ K $，而非固定值。

---

> 📌 **总结一句话**：  
> **AdaptiveSpec 通过 margin-based lossy verification 和 dynamic tree-shaping 两个无需训练的 per-step 自适应机制，在几乎不失准确性的前提下，将 Speculative Decoding 的吞吐量提升了 18–56%，显著优于现有 SOTA 方法。**

</details>

---

### 2. [Hardware-Aware FP4 FlashAttention-4](https://arxiv.org/abs/2609.04105)

**Authors**: Robert Hu  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2609.04105v1  

#### Abstract
Blackwell's 4-bit floating-point (FP4) tensor cores do not automatically make attention faster because softmax conversion and on-chip dependencies dominate once its matrix products shrink. We address this with \emph{Direct-P} for noncausal inference and a causal path that passes the forward quantiza...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Hardware-Aware FP4 FlashAttention-4》论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
NVIDIA Blackwell 架构引入了 4-bit floating-point (FP4) tensor cores，理论上可显著加速矩阵乘法。然而，在 Attention 机制中，**softmax 转换和片上依赖关系**成为了瓶颈，导致 FP4 并不能自动提升整体性能。具体而言：
- **前向传播瓶颈**：在 `QK^T` 和 `PV` 两个矩阵乘法之间，需要进行 softmax 归一化，该过程涉及指数计算、求和归一化等操作，这些操作无法从 FP4 加速中受益，且成为关键路径上的延迟来源。
- **训练效率瓶颈**：在反向传播中，传统方法会重新计算高精度的分数矩阵 `S` 来重建概率 `P`，造成冗余计算。

### 提出的新方法与创新思路
论文提出了两种核心优化策略：

#### （1）**Direct-P**（前向推理）
一种用于非因果（noncausal）前向推理的全 FP4 方法，其核心思想是**直接映射对数分数到 FP4 概率码**，并**使用相同的量化值进行归一化**。
- **直接映射**（Direct Mapping）：将 softmax 的输入（即 `(z_i - m)`）通过一个仿射分类器（affine classifier）直接映射到 E2M1 的 8 个离散值之一，跳过中间的高精度浮点指数计算。
- **一致归一化**（Consistent Normalization）：归一化的分母也由这些被量化后的 FP4 概率值计算得出，确保了前向算子的一致性。
- **极端 Logits 防护**：对于某些模型层中极高的 logits，采用采样锚点（sampled anchor）和重关联计算来避免次正规数（subnormal）溢出为零。

#### （2）**量化因果反向传播**（Quantized Causal Backward）
一种用于训练的因果反向传播方法，其核心是**复用前向传播中的低精度状态**。
- **前向状态传递**：前向传播保存量化后的 Q/K 数据、缩放因子（scales）和 softmax 正常化因子（LSE）。
- **反向概率重建**：反向传播利用这些保存的低精度状态直接重建概率 `P`，而无需重新计算完整的 BF16 分数矩阵。
- **FP8 梯度操作数**：梯度矩阵乘法使用 FP8 操作数以保证数值稳定性。

### 相比现有方法的优势
- **更高的吞吐量**：在前向推理中，Direct-P 在有利的 Blackwell 形状下，达到了 BF16 FA4 吞吐量的 **2.13 倍**。
- **更低的延迟**：在单 GPU 的 80 亿参数模型更新中，完整步骤时间加速了 **1.14 倍**。
- **硬件感知设计**：方法紧密结合 Blackwell 的硬件特性（如 TMEM、TMA、MMA），有效解决了片上存储和依赖瓶颈。

---

## 2. 核心实验方法和设置

### 使用的数据集
论文并未在传统 NLP 或 CV 数据集上进行端到端微调，而是采用了多种**固定输入的模型评估**和**合成基准测试**：
- **Vision Transformer (ViT)**：在 S256, S1024, S4096 序列长度上进行图像分类。
- **BERT**：在 S256 和 S512 上进行掩码语言建模（MLM）和 SST-2 情感分析。
- **Wan 视频扩散模型**：在 S7680 上评估视频生成质量。
- **ViT-MAE**：在 COCO 图像上进行图像重建任务。
- **合成负载**：使用高斯分布生成的精确 softmax 概率进行数值诊断。

### 实验设置和评估指标
- **硬件平台**：NVIDIA GB200 和 B300 GPU。
- **核心形状**：主要评估 `D128` 头维度下的多种 `(B, S, H)` 组合。
- **评估指标**：
  - **性能**：吞吐量（TFLOP/s 或 PFLOP/s）、延迟（ms）、加速比（Speedup）。
  - **准确性**：输出与 BF16 参考结果的 Cosine 相似度、相对 L2 误差（rel-L2）、RMSE。
  - **训练稳定性**：损失函数曲线、梯度范数、是否发散。

### 基线方法对比
- **HAO AI Lab 的 FP4 FA4 实现**：作为主要基线，支持 NVFP4/NVFP4、NVFP4/FP8 等路径。
- **BF16 FA4**：标准的 bfloat16 实现，作为性能和准确性的黄金参考。
- **FP8 P/V 路径**：作为高精度的 FP4 替代方案进行比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **前向推理吞吐量**：
  - 在 `D128/H24/S4096` 形状下，Direct-P 达到了 **2237 TFLOP/s**，相比 HAO BF16 实现，几何平均加速比为 **2.023 倍**，峰值达到 **2998 TFLOP/s**。
  - 在 B300 上，`D128/H64/S8192` 达到了 **3116 TFLOP/s**。
- **完整模型更新速度**：
  - 在单个 GB200 上，对 80 亿参数模型进行完整更新，当本地批大小 `B=4` 时，加速比达到 **1.14 倍**。
  - 分布式训练中，FP8 P/V 路径的每 GPU 吞吐量从 21.85k 提升至 24.30k tokens/s，加速比 **1.112 倍**。

### 与基线方法的对比结果
- **与 HAO NV/FP8 对比**：
  - **速度**：Direct-P (NV/MX) 显著更快（~2.0x vs ~1.1x）。
  - **精度**：HAO NV/FP8 更准确（Cosine ~0.9899 vs ~0.9438）。Direct-P 是一个明确的**速度-精度权衡**。
- **与 HAO NV/NV 对比**：Direct-P 在所有测试形状下都实现了显著的速度优势。

### 消融实验结果
- **概率格式影响**：
  - 表 3 显示，MXFP4 在避免零缩放方面优于未稳定化的 NVFP4，但精度略低于稳定的 NVFP4。
- **训练稳定性消融**：
  - **关键发现**：所有测试的 **MXFP4 P/V 训练轨迹均发散**，而 FP8 P/V 轨迹保持稳定。
  - 因此，最终的训练路径仍选择 **FP8 P/V** 以保证稳定性，尽管其前向部分更慢。
- **Direct-P 策略对比**：
  - `fast` 策略（全仿射）延迟最低。
  - `accurate` 策略（部分使用 EX2）精度更高，但速度稍慢。

---

## 4. 关键结论和发现

### 主要发现
1. **FP4 加速的瓶颈在于非矩阵运算部分**：单纯加速 `QK^T` 和 `PV` 矩阵乘法不足以提升整体性能，**softmax 路径的延迟**才是关键。
2. **Direct-P 有效缩短了关键路径**：通过直接映射和一致归一化，成功地将全 FP4 前向推理的吞吐量提升至 BF16 的两倍以上。
3. **复用前向状态可加速反向传播**：将量化后的 Q/K 状态传递给反向传播，可以有效减少冗余计算，实现端到端加速。
4. **训练稳定性至关重要**：尽管 MXFP4 在前向推理中表现优异，但其在分布式训练中会导致**轨迹发散**，因此 **FP8 P/V 是当前稳定训练的必要选择**。
5. **硬件资源限制了并行度**：Tensor Memory (TMEM) 的所有权和容量是限制 QK、softmax 和 PV 操作进一步重叠的主要因素。

### 方法的局限性
- **并非纯端到端 FP4 训练**：学习到的投影层（learned projections）和注意力操作数仍是不同的精度边界，未实现完全的 FP4 训练。
- **训练路径仍需 FP8**：由于 MXFP4 P/V 的不稳定性，最终的训练路径未能完全使用 FP4，仍需依赖 FP8 概率和值。
- **硬件特定性**：结论和优化高度依赖于 Blackwell 架构的硬件特性（如 TMEM、TMA），可能不适用于其他架构。
- **固定输入评估**：下游任务的评估基于固定输入，未证明在真实微调或预训练场景下的泛化能力。

### 未来工作方向
- **探索更稳定的 FP4 训练格式**：论文提到，**Unsigned E5M3 (UE5M3) block scaling** 已在独立工作中实现了稳定的 FP4 语言模型预训练，将其应用于 P/V 产品和反向梯度产品是一个有前景的方向。
- **改进硬件设计**：建议硬件层面提供更大的可用重叠窗口，例如：
  - 增加一个可分配的分数银行（allocatable score bank）。
  - 支持 K32 的 scaled-FP4 PV 指令。
  - 将缩放因子移出 TMEM。
- **扩展到其他形状**：当前工作聚焦于 D128，未来需研究 D64 等其他头维度下的优化策略。

</details>

---

### 3. [Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM](https://arxiv.org/abs/2609.04098)

**Authors**: Sergii Kozyrev, Davyd Maiboroda  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2609.04098v1  

#### Abstract
Hybrid LLMs pair softmax attention with linear-attention layers such as Gated DeltaNet (GDN), whose recurrent state summarizes the context in fixed size. Early community 4-bit quantizations of Qwen3.8-27B (48 GDN layers, 16 attention layers) left the GDN block in 8- or 16-bit precision -- especially...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 Hybrid LLM（如 Qwen3.8-27B）中，**Gated DeltaNet (GDN)** 层因其递归状态机制，在社区实践中普遍被保留为 8-bit 或 BF16 精度，而仅对 MLP 和 Attention 层进行 4-bit 量化（W4A4）。这种做法基于一个直觉假设：  
> “递归结构中的量化误差会在长上下文中不断累积，尤其是控制衰减和写入强度的门控投影 `a` 和 `b` 非常敏感。”

本文挑战并推翻了这一假设，系统性地回答了：**为什么 GDN 能够安全地承受 W4A4 量化？**

### 提出的新方法与思路
- **构建首个真正的全模型 W4A4 量化方案 MINIMA**：将 Qwen3.8-27B 中全部 **496 个线性层**（包括 GDN 的 `in_proj_a`, `in_proj_b`, `qkv`, `z`, `out_proj`）统一量化至 **NVFP4 W4A4**，首次实现 GDN 块的完全低比特化。
- **提出“机制解释”而非经验主义**：通过四步机制研究（block scaling、gate nonlinearity、delta rule dynamics、end-to-end error washout），从架构层面解释为何 GDN 不仅能存活于 4-bit，反而是“最容易量化的部分”。
- **修复量化部署栈的关键缺陷**：
  - 揭示并解决 **per-module calibration 与 fused-GEMM kernel 之间的 scale mismatch 问题**，避免门控参数被错误缩放。
  - 提出 **calibrated FP8 KV-cache scales**，几乎完全消除 FP8 KV 缓存带来的长上下文 perplexity 惩罚。

### 相比现有方法的优势
| 维度 | 社区方案（Unsloth / RadixArk） | 本文 MINIMA |
|------|-------------------------------|-------------|
| 量化范围 | 仅 MLP 用 NVFP4，GDN & Attention 保留在 FP8/BF16 | 所有 496 线性层统一 W4A4 |
| 显存占用 | ~18–20 GiB | **17.5 GiB（最小）** |
| Prefill 速度 | 较慢（GDN 仍运行在高精度 GEMM） | **最快（TTFT 6.90s → 4.03s @32K）** |
| 准确率 | 接近 BF16 | **匹配 BF16 在种子噪声范围内** |
| 架构理解 | 黑箱保护 GDN | **提供可解释的量化鲁棒性机制** |

---

## 2. 核心实验方法和设置

### 使用的数据集与任务
- **Perplexity**: WikiText-2 @ 4K 和 32K 上下文长度
- **知识推理**: MMLU-Pro
- **数学能力**: GSM8K
- **进阶数学**: AIME'25（pass@1）
- **科学难题**: GPQA-Diamond
- **代码生成**: LiveCodeBench v6（unit test grading）
- **长上下文检索**: RULER @ 32K / 64K（multi-key NIAH）

> 所有任务均采用固定服务配置，确保公平比较。

### 实验设置
- **模型**: Qwen3.8-27B（48 GDN + 16 Attention layers）
- **量化格式**: NVFP4（E2M1 4-bit 值 + E4M3 每16元素块 scale + FP32 tensor scale）
- **W4A4**: 权重与激活均量化
- **硬件平台**: 单张 RTX PRO 6000（96GB），使用 vLLM 0.27.1，TP=1
- **KV Cache**: 统一使用 FP8，排除内存干扰
- **校准集**: 冻结的 128 个样本 × 32K token 数据集

### 基线方法对比
| 方法 | 量化策略 | 是否开源 |
|------|----------|---------|
| **BF16** | 全精度参考模型 | 是 |
| **MINIMA (Ours)** | 所有线性层 W4A4（含 GDN a/b/qkv/z/out） | 是（HF 发布） |
| **Unsloth (Dynamic v3)** | MLP: NVFP4；GDN & Attention: FP8 W8A8，a/b 保持 BF16 | 是 |
| **RadixArk (ModelOpt)** | 同上 | 是 |

> 所有模型在同一服务环境下测试，启用 per-sample validity check（过滤空输出、泄露 `<think>` 等无效响应）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 指标 | BF16 | MINIMA | Unsloth | RadixArk |
|------|------|--------|---------|----------|
| **PPL@4K / @32K** | 6.95 / 10.35 | 7.67 / 10.84 | 7.16 / 9.91 | 7.35 / 9.95 |
| **MMLU-Pro (%)** | 80.4 | 79.7 | 78.9 | 79.1 |
| **GSM8K (%)** | 95.5 | 95.5 | 95.4 | 95.7 |
| **AIME'25 (pass@1)** | 86.7 | 86.7 | 87.5 | 84.2 |
| **GPQA-Diamond (%)** | 86.5 | 85.1 | 85.0 | 85.4 |
| **LiveCodeBench v6 (%)** | 79.0 | 78.5 | 79.9 | 79.6 |
| **5-task avg (△)** | 85.62 | 85.10 (-0.52) | 85.34 (-0.28) | 84.80 (-0.82) |
| **VRAM 权重大小** | 50.13 GiB | **17.53 GiB** | 20.23 GiB | 18.83 GiB |
| **Decode tok/s @32** | 621 | 1,154 | 1,132 | 1,174 |
| **TTFT @32K (prefill 时间)** | 6.90s | **4.03s** | 4.49s | 4.39s |

> ✅ **所有量化模型在各项任务上均未超出 BF16 的种子波动范围（seed noise）**  
> 🔺 MINIMA 在准确率上与 BF16 完全持平（如 AIME'25 四次运行均为 26/30），且生成行为一致（平均思考长度相同）

### 与基线方法的对比结果
- **准确性**：MINIMA 在 5-task 平均分上仅落后 BF16 0.52 分，优于 RadixArk（-0.82），接近 Unsloth（-0.28），差异小于单题得分波动。
- **效率优势显著**：
  - 显存减少 **2.9×**（50.1 → 17.5 GiB）
  - Prefill 吞吐提升 **+14–19%**
  - 支持更大的 KV cache（达 1.81M tokens）
- **解码吞吐**：虽略低于 RadixArk（因小 batch 激活量化开销），但仍高出 BF16 近 **2×**

### 消融实验与机制分析（S5）

#### （1）输入统计并非主因（S5.1）
- GDN 输入与其他模块一样存在极端 outlier（max/RMS > 60, kurtosis ~1500）
- 但由于 **NVFP4 的 block scaling**，每个 outlier 只影响其所在 block 的 15 个邻居，从而局部化误差
- 结果：各层角色的 A4 量化误差均匀（7.5–9.2%），不依赖“干净输入”

#### （2）受保护的门控参数最鲁棒（S5.2）
| 投影 | GEMM 错误率 | 输出 y 相对误差 |
|------|--------------|------------------|
| `a` (decay gate) | 11.0% | **2.1%** |
| `b` (write gate) | 8.5% | **2.6%** |
| `qkv` | 10.6% | 10.4% |
| `out_proj` | 12.7% | 12.7% |

> ❗ 社区重点保护的 `a` 和 `b` 实际是**最不敏感**的部分！  
> 原因在于其非线性变换：`softplus(a)` 和 `exp(log_alpha)` 对输入扰动具有压缩效应。

#### （3）递归动态主动擦除噪声（S5.3）
- **状态误差 plateau 在 ~12.6%**，在整个 32K 序列中保持稳定（无积累）
- 注入 1% 状态脉冲后，误差在 **数百步内衰减至 1/e**，远快于由 α 控制的理论遗忘窗口（可达 60K）
- 原因：**Delta Rule 每次写入沿当前 key 方向覆盖旧状态**，实现“主动删除”，而非被动衰减

#### （4）端到端误差随上下文稀释（S5.4）
- 分位置 NLL 分析显示：**MINIMA 与 BF16 的差距在前半段较大（+0.081 nat），但在后半段缩小甚至反转（最后 2K tokens 为 -0.053 nat）**
- 表明量化代价是短上下文的 per-token 效应，被填充后的状态吸收

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **GDN 是 Hybrid LLM 中“最容易量化”的部分**，而非最脆弱的环节。
2. ✅ **传统保护策略（保留 a/b 为高精度）是冗余的**，这些门控本身已被架构保护（log-space parameterization + nonlinearity）。
3. ✅ **递归不会导致误差积累**，反而因 Delta Rule 的 overwrite 机制主动清除历史噪声。
4. ✅ **block scaling + gate compression + overwrite dynamics** 共同构成 GDN 对 W4A4 的天然鲁棒性。
5. ✅ **量化一切（quantize everything）是可行且最优策略**，配合 calibrated FP8 KV-cache 可实现极致高效部署。

### 方法的局限性
- **模型范围有限**：目前仅验证于 Qwen3.8-27B，尚未扩展至其他架构或更大规模。
- **未涵盖更低位宽**：如 3-bit 或 Int4 对称量化未探索。
- **依赖特定参数化形式**：若门控采用线性参数化（而非 log-space + softplus/exp），可能不再具备同等鲁棒性（见 S9）。
- **decode overhead 存在 kernel 级别瓶颈**：NVFP4 激活量化在小 batch 下带来额外延迟。

### 未来工作方向
- 将该量化范式推广至更多 Hybrid 架构（如 Mamba、Hawk 等）。
- 探索 sub-4-bit 量化（如 3-bit MLP + 4-bit GDN）以进一步压缩。
- 设计面向 fused-GEMM 的统一 calibration protocol，避免 scale mismatch。
- 研究如何将此机制洞察用于训练阶段，设计更易量化的新型 recurrent mixer。

---

> 📦 **实用建议总结**：  
> 对于 Hybrid LLM 的部署，“**quantize everything, ship KV scales**” 是当前最佳实践。  
> GDN 的递归性质不是负担，而是天然的误差抑制器 —— **the recurrent half is the easy half to quantize**。  

🔗 量化模型已发布：[https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4)

</details>

---

### 4. [BASP: Communication-Efficient Batch-Aware Sequence Parallelism for LLM Training](https://arxiv.org/abs/2609.03151)

**Authors**: Bigyan Ghimire, Jon C. Calhoun  
**Category**: cs.DC  
**Published**: 2026-09-04  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.03151v1  

#### Abstract
Long-context reasoning for large language models (LLMs) is becoming increasingly important, but training over long sequences remains challenging due to massive memory and communication requirements. Sequence parallelism has emerged as an essential technique for addressing bottlenecks in long sequenc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BASP: Communication-Efficient Batch-Aware Sequence Parallelism for LLM Training

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **Sequence Parallelism**（如 DeepSpeed-Ulysses）在处理长序列训练时，采用全局 `N-way` 的 **all-to-all** 通信模式，无论 micro-batch size 大小如何，所有 GPU 都需参与全连接通信。这导致：
- 通信开销随 batch size 增大而显著增加；
- 在多节点集群中，跨节点（inter-node）通信（如 InfiniBand）成为瓶颈；
- 资源利用率低，训练效率受限于通信而非计算。

### 🚀 提出的新方法：Batch-Aware Sequence Parallelism (BASP)
BASP 是一种新型的 **sequence parallelism** 架构，其核心思想是：
> 利用 **micro-batch 结构** 将 GPU 分组为多个独立的 **batch-aware groups**，每个 group 只负责一个或部分序列的并行处理，从而将全局 `N-way all-to-all` 拆分为多个更小的 `K-way all-to-all` 子通信操作。

#### 具体设计：
- 给定总 GPU 数 $ N $ 和 micro-batch size $ B $，若 $ N = K \times B $，则创建 $ B $ 个互不重叠的子组，每组大小为 $ K = N/B $。
- 每个 micro-batch 中的序列被分配给一个独立的 GPU 子组进行 sequence partitioning 和 attention 计算。
- 所有通信限制在子组内部，显著减少通信参与者数量（从 $ N-1 $ 降至 $ K-1 $）。

### 🔍 相比现有方法的优势
| 方面 | Ulysses-SP | BASP |
|------|-----------|-------|
| All-to-all 规模 | 全局 $ N $-way | 局部 $ K = N/B $-way |
| 通信范围 | 跨所有节点 | 可限定在单个节点内（利用 NVLink） |
| 内存占用 | 相同 | 完全相同（无额外内存代价） |
| 序列长度支持 | 支持长序列 | 同样支持，且更高效 |
| 正确性 | 保证数学等价 | 完全保留原模型行为 |

> ✅ **优势总结**：BASP 在不牺牲模型精度、内存使用和最大可训练序列长度的前提下，显著降低 all-to-all 通信开销，提升训练吞吐量。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 论文未明确指定具体预训练数据集名称，但指出实验基于标准 LLM 预训练任务。
- 使用合成输入数据进行性能基准测试（典型做法），重点在于测量端到端训练时间与通信开销。

### ⚙️ 实验设置
| 项目 | 设置 |
|------|------|
| 硬件平台 | 2 节点 × 4 NVIDIA A100 40GB GPU（共 8 GPUs）<br>节点间通过 400Gbps InfiniBand 连接<br>节点内通过 NVLink 高速互联 |
| 模型族 | **Llama** 系列（3.2-1B, 3.2-3B, 3.1-8B）<br>**Qwen** 系列（1.5-1.8B, 2.5-3B, 3-8B） |
| 序列长度 | 最高至 **32K tokens** |
| Micro-batch size | 1 ~ 8（受限于显存） |
| 并行策略 | Sequence Parallelism + ZeRO-3 + Mixed Precision |
| 实现方式 | 修改开源 DeepSpeed 框架实现 BASP，保持 API 兼容性 |
| 测量指标 | - End-to-end step time（迭代耗时）<br>- All-to-all 通信时间占比<br>- Loss 收敛曲线对比 |

### 🆚 基线方法
- **Ulysses-SP**（DeepSpeed-Ulysses）作为主要 baseline。
- 包括不同 SP degree 设置下的变体对比（如 SP=N vs SP=N/B）。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）端到端训练速度提升（Speedup）
| 模型 | BASP 相对 Ulysses-SP 加速比 |
|------|----------------------------|
| **Llama 3.1-8B** | **1.21×** |
| **Qwen 1.5-1.8B** | **1.32×** |
| 其他模型 | 1.17× ~ 1.31× 不等 |

> 在 batch size=2、seq len=16K 条件下，平均提速约 **18–24%**。

#### （2）All-to-All 通信时间大幅下降
| 场景 | 通信时间减少倍数 | 说明 |
|------|------------------|------|
| Qwen 1.5-1.8B | 3.10× ↓ | all-to-all 占比从 37.7% → 16% |
| Llama 3.2-1B | 2.45× ↓ | 占比从 33.5% → 13.4% |
| Batch size=8 时 | 最高达 **85×** ↓ | all-to-all 时间几乎可忽略（仅占 0.5%） |

> ✅ 当 $ K = 4 $（即每组 4 GPU）时，恰好匹配单节点拓扑，通信完全运行在高速 NVLink 上，避免了慢速 InfiniBand。

#### （3）消融实验与扩展性分析
| 实验维度 | 发现 |
|---------|------|
| **Micro-batch scaling** | 随着 batch size 增大，加速效果增强：<br>B=1: 无增益（等效）<br>B=2: 1.10×<br>B=8: 达 **1.26×** |
| **Sequence length scaling** | 长序列下收益更大：<br>seq=8K: +13.1%<br>seq=16K: +18.2%<br>seq=32K: **+25.9%** |
| **通信瓶颈转移** | 在 B=8 时，all-to-all 已非瓶颈，ZeRO 相关通信成为新瓶颈 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Batch structure 可被有效利用来优化 SP 通信**：传统 SP 忽视 batch 维度结构，造成不必要的全局通信。
2. **BASP 显著降低 all-to-all 开销**：通过构建 topology-aware 的 batch-aware groups，实现通信局部化。
3. **性能增益在长序列和大 batch 下尤为明显**：适用于典型的 long-context LLM 训练场景。
4. **完全兼容原有训练流程**：无需修改模型结构、优化器或损失函数，仅需调整通信分组逻辑。
5. **准确率完全一致**：loss 曲线与 Ulysses-SP 几乎完全重合，验证了方法的正确性。

### ⚠️ 局限性
1. **要求 $ N \mod B = 0 $**：即 GPU 总数必须能被 micro-batch size 整除，否则无法均匀分组。
2. **当前假设 $ K = N/B $ 为整数**：尚未支持非整除情况下的动态分组策略。
3. **当 batch 很小时（如 B=1）无收益**：此时仍退化为全局 all-to-all。
4. **依赖硬件拓扑对齐**：最佳性能需 $ K $ 等于每节点 GPU 数量（如 4 或 8）。

### 🔮 未来工作方向
- 支持 **non-divisible configurations** 的弹性分组机制。
- 动态自适应选择是否启用 BASP（结合 FlexSP 思路）。
- 探索与其他 SP 变体（如 Ring Attention、Striped Attention）的融合。
- 扩展至更大规模集群（百/千卡级别）验证可扩展性。

---

## ✅ 总结一句话
> **BASP 通过“批感知”的 GPU 分组策略，将全局 all-to-all 降级为局部通信，在不增加内存、不影响收敛的前提下，实现了高达 1.32× 的训练加速，尤其适合长上下文、大批量的 LLM 训练场景。**

</details>

---

### 5. [DE-Venus: A Data-Efficient RLVR Framework for Large Language Models](https://arxiv.org/abs/2609.03324)

**Authors**: Shenzhi Yang, Guangcheng Zhu, Kai Tang, Zhengqing Zang, Xing Zheng, Haobo Wang, Yingfan Ma, Bowen Song, Bo Han, Bo An, Lei Feng, Weiqiang Wang, Junbo Zhao, Gang Chen  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.03324v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) improves large language model reasoning, but its practical scaling is constrained by expensive on-policy rollouts and the cost of obtaining reliable targets at scale. Existing methods address sample selection, incomplete supervision, or noisy lab...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DE-Venus: A Data-Efficient RLVR Framework for Large Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

**Reinforcement Learning with Verifiable Rewards (RLVR)** 是提升大语言模型（LLM）推理能力的重要范式，但其实际应用面临以下挑战：

- **高成本的 on-policy rollouts**：每次训练都需要生成多个推理路径并进行验证，计算开销巨大。
- **标注成本高昂**：需要大量高质量、可验证的参考答案作为监督信号，尤其在专业领域难以获取。
- **现有方法碎片化**：样本选择、弱监督构建、噪声标签处理等技术通常独立实现，且与分布式训练逻辑耦合严重，导致复现困难、比较不公、难以复用。

因此，如何在**有限标注预算下高效利用数据**，同时保持甚至提升模型性能，是当前 RLVR 面临的核心瓶颈。

---

### 🚀 提出了什么新方法或新思路

作者提出 **DE-Venus** ——一个统一的数据高效 RLVR 框架，其核心思想是：

> 将“监督”（supervision）视为一种**随训练演化的状态**，贯穿数据准备到策略优化全过程，并通过模块化解耦干预点与执行后端。

#### 创新架构：三大干预模块 + 统一执行边界

| 模块 | 功能 |
|------|------|
| **Active Data Selection** | 在训练前决定哪些样例应被保留、标注或用于弱监督，基于难度、不确定性或探针校准进行路由。 |
| **Weak Supervision Construction** | 对无标签样例构造伪目标（pseudo-targets）或无目标奖励（target-free rewards），如共识投票、交叉视图一致性、自信心奖励等。 |
| **Training-Time Supervision Refinement** | 在训练过程中动态过滤、加权或修正不可靠的监督信号，例如基于轨迹动态、表示几何或生成证据进行去噪。 |

这些模块共享一个轻量级控制平面，但**不替换底层的分布式 RL 执行引擎**（基于 `verl`），仅在其接口处插入变换操作。

---

### 🔍 相比现有方法的优势

| 特性 | DE-Venus | 传统方法 |
|------|---------|--------|
| **系统设计** | 模块化、解耦监督逻辑与执行后端 | 耦合严重，常需 fork 整个训练流程 |
| **可复现性** | 支持配置驱动、持久化 Parquet 数据集版本管理 | 实现分散，难以复现 |
| **可扩展性** | 支持多种方法组合（如 TTRL + TraPO），新增方法只需实现特定接口 | 每种方法为独立管道 |
| **效率增益** | 显著减少标注量、训练数据量和收敛步数 | 多数只关注单一环节优化 |
| **兼容性** | 完全兼容 `verl` 合同（data/proto/batch/reward/advantage） | 自定义协议，迁移成本高 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### 公共基准（Public Benchmarks）
- **数学推理（ID）**：
  - AIME 2024/2025, AMC, MATH-500, Minerva, OlympiadBench
- **通用推理（OOD）**：
  - ARC-c, GPQA-Diamond, MMLU-Pro

#### 业务场景（Business Scenarios）
- **贷款信用分配**（Loan-Credit Assignment）
- **医疗共情训练**（Medical Empathy Training）
- **内在安全性训练**（Intrinsic Safety Training）

#### 训练数据集
- **DeepMath-103K**（数学任务）
- **DAPO-Math-14K**（数据选择任务）

---

### ⚙️ 实验设置和评估指标

| 设置项 | 描述 |
|-------|------|
| **Backbone Models** | Qwen3-8B-Base, Qwen3-4B-Base |
| **Optimization Algorithm** | 默认使用 GRPO（Group Relative Policy Optimization） |
| **Rollouts per Prompt** | G = 8 |
| **Batch Size** | 总 batch size 128，micro-batch 32 |
| **Decoding** | Temperature 0.6, top-p = 1.0 |
| **评估指标** |  
| - 数学任务 | `avg@32`（AIME/AMC）、`avg@4`（其余） |
| - 通用任务 | `pass@1`（MMLU-Pro）、`avg@4`（其余） |
| - 业务场景 | 归一化的业务指标（normalized credit-assignment metric, response-quality index 等） |

---

### 🔁 基线方法对比

| 类型 | 方法列表 |
|------|----------|
| **Fully Supervised** | 使用全部标签的完整监督训练 |
| **Unsupervised RLVR** | TTRL, Tok-Entropy, Seq-Entropy, Self-Certainty, Co-Rewarding（无标签） |
| **Semi-supervised RLVR** | TTRL, EMRL, TraPO, GeoMin（配合 10% 标签） |
| **Noisy Label Baseline** | Standard GRPO（不同噪声比例） |
| **Selection Baselines** | Random, Consistency, Entropy, Self-Certainty, CoE, CoT-Kinetics, PivotTrace |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据与对比结果

#### ✅ **弱监督学习（4.1节）**

| 方法 | ID Acc (%) | OOD Acc (%) | 标签比例 |
|------|------------|-------------|--------|
| Fully Supervised | 46.7 | 69.0 | 100% |
| **GeoMin (10%)** | **47.9** ↑1.2 | **69.5** ↑0.5 | 10% |
| TraPO (10%) | 43.8 | 67.2 | 10% |
| TTRL (0%) | 42.7 | 67.0 | 0% |

> 💡 **结论**：仅用 **10% 的标签**，GeoMin 即可**超越全监督基线**！

---

#### ✅ **噪声标签鲁棒性（4.2节）**

在 **active noise regime** 下（错误标签可被策略生成强化）：

| 噪声比例 $p$ | 方法 | ID Gain | OOD Gain |
|--------------|------|--------|--------|
| 0.5 | OLR vs GRPO | +6.4 pts | — |
| 0.7 | OLR vs GRPO | — | **+8.1 pts** |

> 💡 **结论**：**Online Label Refinement (OLR)** 在各种噪声条件下均带来稳定增益，证明框架能有效纠正不可靠标注。

---

#### ✅ **数据选择效率（4.3节）**

| 设置 | 方法 | ID Acc | OOD Acc | 数据保留率 | 标签保留率 |
|------|------|--------|---------|-----------|------------|
| Full Data + Full Labels | — | 47.6 | 62.2 | 100% | 100% |
| **Selected Subset** | **PivotTrace** | **49.5** ↑1.9 | **64.9** ↑2.7 | **57.9%** | **29.3%** |

> 💡 **结论**：使用不到 **58% 的数据 + 不到 30% 的标签**，PivotTrace 反而**全面超越全数据全标签训练**。

---

#### ✅ **业务场景验证（4.4节）**

| 场景 | 成果 |
|------|------|
| **贷款信用分配** | 固定 500 标签 + 1,000 未标注数据 → **归一化指标最高提升 14 点**（GeoMin） |
| **医疗共情训练** | 轨迹过滤移除 **28% 训练数据** → 性能仍比全数据低 <1.9 点，远高于未训练基线（+2.7） |
| **内在安全训练** | 保留 **13%-30.7% 相关数据** → 安全指标持平，部分能力提升 **6.7%**，收敛步数减少 **63%-75%** |

> 💡 **结论**：DE-Venus 在真实业务中显著降低**标注、计算和迭代成本**，同时维持甚至提升质量。

---

### 🔍 消融实验（隐含于多配置对比）

虽然未明确列出“ablation study”章节，但以下对比本质上构成消融分析：

- **是否使用可靠性感知机制？**
  - TTRL 和 EMRL 在贷款场景中使用额外未标注数据反而**性能下降**（97 vs 100），说明盲目加入弱监督有害。
  - 而 TraPO 和 GeoMin 因具备**可靠性筛选机制**，带来正向增益（111–114）。
- **是否进行数据选择？**
  - 多数非 PivotTrace 方法在子集训练下表现不稳定或退化，表明**选择策略本身至关重要**，而非简单删减数据。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **数据效率应视为端到端监督生命周期问题**  
   不仅仅是减少数据量，而是要在 **selection → construction → refinement** 全链路中智能管理监督质量。

2. **弱监督可以优于全监督**  
   当结合**可靠性感知机制**（如 TraPO、GeoMin）时，仅用 **10% 标签**即可超越全监督训练。

3. **高质量选择 > 更多数据**  
   移除低效样本（如已掌握或无法解决的问题）不仅能加速训练，还能**提升最终性能**。

4. **框架设计促进公平比较与复用**  
   DE-Venus 实现了多种前沿方法的统一集成，在相同环境下验证其有效性，避免工程偏差。

5. **真实场景收益显著**  
   在业务系统中，DE-Venus 可实现：
   - 标注成本降低 **90%+**
   - 训练数据减少 **~70%**
   - 收敛速度加快 **>60%**
   - 性能持平或反超

---

### ⚠️ 方法的局限性

1. **依赖 rollout 生成**  
   仍需多次采样推理路径，对推理延迟敏感的任务可能受限。

2. **初始冷启动问题**  
   如 TraPO、GeoMin 需要少量可信标签来建立参考轨迹或分布模型，完全零标签场景支持较弱。

3. **模块间交互尚未充分探索**  
   当前实验多为单模块启用，联合使用 Active Selection + Weak Supervision + Refinement 的协同效应有待深入研究。

4. **仅适配 GRPO 类算法**  
   虽然宣称兼容其他 RLVR 算法（如 RLOO、REINFORCE++），但实证主要集中在 GRPO 上。

---

### 🔮 未来工作方向

1. **自动化模块编排**  
   开发元控制器，根据任务特性自动选择最优的干预组合（如“何时用 TraPO，何时用 GeoMin”）。

2. **跨轮次知识迁移**  
   将一轮训练中积累的监督决策（如可靠样本池）迁移到后续任务或模型微调中。

3. **更轻量的监督信号提取**  
   探索无需多 rollouts 的不确定性估计方式，进一步降低推理成本。

4. **扩展至更多 RLVR 算法**  
   验证 DE-Venus 在 RLOO、DPO-style RLVR 中的有效性和通用性。

5. **开放生态建设**  
   构建社区驱动的插件库，支持第三方方法快速接入 DE-Venus 生态。

---

> 📌 **一句话总结**：  
> **DE-Venus 通过将“监督”建模为可演化状态，实现了模块化、高效、可复现的数据高效 RLVR 框架，在仅用 10%-30% 数据/标签的情况下，不仅节省了训练资源，还在多个任务上超越了全监督基线。**

</details>

---

### 6. [Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010)

**Authors**: Subham Sekhar Sahoo, Lingjie Chen, Khiem Pham, Jonathan Geuter, Chaitanya Dwivedi, Varad Pimpalkhute, Yash Akhauri, Alexander Moreno, Mikhail Yurochkin, Zhenting Wang, Mostafa Elhoushi, Nolan Dey, Shane Bergsma, Joel Hestness, John Thickstun, Eric Xing, Zhengzhong Liu  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.04010v1  

#### Abstract
Large Language Models (LLMs) owe much of their success to next-token prediction (NTP), but their autoregressive (AR) structure requires slow, sequential token generation. To overcome this bottleneck, we introduce diffusion-augmented LLMs, a new class of models that defines an AR model distribution w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Unlocking Lossless Speedups in LLMs via Discrete Diffusion

---

## 1. 主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）依赖于**自回归（Autoregressive, AR）**的逐token生成方式，这导致推理过程缓慢且顺序化，严重制约了服务延迟和强化学习（RL）训练效率。尽管已有如**投机解码（Speculative Decoding）**和**离散扩散模型（d-LLMs）**等加速方法，但它们存在以下问题：
- **投机解码**需要额外的草稿模型（draft model），增加部署复杂性和内存开销；
- **d-LLMs**虽然支持并行生成，但通常以牺牲生成质量为代价（lossy），且在大batch size下速度优势消失。

### 提出的新方法
本文提出 **diffusion-augmented LLMs**，一种新型架构，其核心是将一个标准AR模型“增强”为同时具备高质量和高速度能力的统一框架。该方法的关键创新包括：

#### （1）参数解耦设计
- 将模型权重分为两组：
  - **AR weights**：负责建模输出分布，决定生成质量，通过标准的Next-Token Prediction（NTP）目标训练。
  - **Diffusion weights**：轻量级模块，专用于并行生成多个token，通过**Diffusion Distillation**阶段训练。
- 二者共享主干网络，diffusion weights以**LoRA适配器**形式插入各层，仅在噪声序列上激活（通过gated LoRA技术）。

#### （2）训练流程：Drop-in式增强
- 先完成AR模型的标准训练（Pre-training → SFT → RL）；
- 冻结AR weights，单独训练diffusion weights，对原训练流程无侵入，可直接应用于现有开源模型。

#### （3）推理采样器：V-Spec
- 提出 **V-Speculative Sampler（V-Spec）**，结合了扩散生成与AR验证机制：
  - 扩散路径并行生成token块；
  - AR路径作为验证器进行拒绝采样（rejection sampling），保留最长有效前缀。
- 保证最终输出严格遵循原始AR模型的分布，实现**无损加速（lossless speedup）**。

#### （4）无需独立草稿模型
- 不同于投机解码需额外训练小模型，本方法在同一架构内完成起草与验证，节省存储与计算资源。

---

## 2. 核心实验方法和设置

### 数据集
- **端到端训练设置**：使用内部高质量文本语料约23T tokens进行AR权重训练；diffusion weights在SFT数据中抽取7B tokens训练。
- **基于开源模型增强设置**：以 **Qwen3-8B** 为基础模型，在 **OpenThoughts** 开源数据集上训练diffusion weights（未访问原始训练数据）。

### 实验设置与评估指标
#### 评估任务分类
| 类别 | 包含基准 |
|------|--------|
| **Agentic Tasks** | T2-Bench, Terminal-Bench v2.1, SWE-bench Verified |
| **长上下文推理** | AA-LCR |
| **数学推理** | GSM8K, MATH500, AIME系列 |
| **代码生成** | HumanEval, MBPP, LiveCodeBench v6 |
| **科学知识** | GPQA-Diamond, Humanity's Last Exam |
| **指令跟随** | IFEval |

#### 性能指标
- **Pass@1**：主流准确率指标。
- **Tokens Per Forward-pass (TPF)**：每次生成迭代平均接受的token数，反映加速效果。
- **Throughput (tokens/sec)**：
  - **系统吞吐量（System Throughput）**：最大batch size下的总生成速率，衡量高并发服务能力。
  - **单请求吞吐量（Per-request Throughput）**：batch size=1时的响应速度，关注低延迟体验。
- **1K/8K Throughput Test**：输入1024 tokens，生成8192 tokens，标准化比较不同方法的实际吞吐表现。

### 基线方法对比
| 类型 | 对比方法 |
|------|--------|
| **开源d-LLMs** | DiffusionGemma-26B-A4B, Nemotron-Labs-Diffusion-14B |
| **闭源d-LLM** | Mercury 2（Inception Labs） |
| **无损加速方法** | EAGLE-3（AR drafter）, DFlash（diffusion drafter） |
| **有损加速方法** | Jacobi Forcing, SDAR, OPDLM, I-DLM, FLARE, Fast-dLLM v2 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）系统吞吐量（System Throughput）
| 方法 | 最大系统吞吐量 (tokens/sec) |
|------|----------------------------|
| **Uno (Ours)** | **5255** |
| AR (Base Model) | 3577 |
| DiffusionGemma | 1136 |
| Nemotron-Labs-Diffusion | 2794 |
| Mercury 2* | 1197* |

> 💡 Uno 在相同硬件（H200 GPU）上达到 **~4.6× 高于 Mercury 2** 的系统吞吐量。

#### （2）单请求吞吐量（Per-request Throughput）
| 方法 | Batch Size=1 吞吐量 (tokens/sec) |
|------|-------------------------------|
| **Uno (Tree Sampler)** | **383** |
| AR | 176 |
| Mercury 2* | 769* |

> ⚠️ Mercury 2 虽然单请求更快，但其运行在更先进的Blackwell GPU上，且可能使用更低精度量化。

#### （3）加速倍数
- 相比基础AR模型：
  - **最高达 3× 加速**；
  - 在最大batch size下仍保持 **1.5–2× 加速**；
  - 单请求场景下可达 **2.2× 加速**。

---

### 与基线方法对比结果

#### ✅ 超越所有开源d-LLMs
| 指标 | Uno vs. DiffusionGemma | Uno vs. Nemotron-Labs-Diffusion |
|------|-------------------------|----------------------------------|
| 准确率 | 显著更高（尤其agentic任务） | 全面领先 |
| 吞吐量 | 更高系统吞吐 | 更高且质量更好 |
| 是否lossless | ✔️ 是 | ❌ 否（修改AR权重） |

#### ✅ 超越闭源 Mercury 2
- 在**Agentic Tool Use、Coding、Long-Context Reasoning**等任务上全面胜出；
- **系统吞吐量高出 ~4.6×**，即使运行在较慢硬件上；
- 支持**无损加速**，而Mercury 2为lossy方法。

#### ✅ 超越无损投机解码方法
| 方法 | TPF（平均） | 额外参数 | 峰值内存 | 优势 |
|------|-------------|----------|----------|------|
| **UnoQwen** | **5.97** | 0.35B | 118.0 GiB | ✔️ 最高TPF，最低内存 |
| EAGLE-3 | 3.48 | 0.40B | 129.4 GiB | — |
| DFlash | 2.74 | 1.05B | 129.8 GiB | — |

> Uno 在所有batch size下均Pareto占优，见图2。

---

### 消融实验结果（Ablation Studies）

#### （1）损失函数消融（Loss Terms）
| 损失组合 | 平均TPF |
|--------|--------|
| TV Only | 2.39 |
| KL Only | 2.23 |
| KL + TV (α=0.01, β=1) | **2.40** |

> 结合Total Variation Loss有助于提升连续token接受长度。

#### （2）训练课程（Curriculum）
- 使用逐步增大的block size（2→4→8→16）比固定大block size训练更有效，TPF从2.65提升至**2.71**。

#### （3）LoRA配置
- **LoRA rank=128 vs 256**：TPF从2.39→2.47，但参数翻倍；
- **LoRA位置**：应用于所有投影矩阵（Q/K/V/O + MLP）效果最好；
- **QLoRA/rLoRA比例**：最优值随训练轮次变化，三轮训练时**16**最佳。

#### （4）与I-DLM对比
- 官方I-DLM虽声称无损，实测显示其采样器破坏了分布一致性；
- 若将其LoRA适配器接入本文提出的V-Spec采样器，则可恢复无损性，但仍显著慢于Uno。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **可以将AR模型的质量与扩散模型的速度统一在一个架构中**，无需牺牲任一方。
2. ✅ **Diffusion weights可通过极少量数据和计算成本进行训练**，适合“即插即用”地增强现有LLM。
3. ✅ **V-Spec采样器实现了真正的lossless加速**，输出分布完全等价于原始AR模型。
4. ✅ **在真实负载（大batch size）下依然保持显著加速**，解决了当前d-LLMs“仅在小batch有效”的痛点。
5. ✅ **Uno在agentic任务上表现尤为突出**，优于当前最先进的d-LLMs和闭源模型。

### 方法的局限性
- **依赖AR模型的KV Cache机制**，难以进一步压缩首次token延迟；
- 当前实现仍采用两步forward pass（draft + verify），理论上限TPF < B+1；
- 扩散训练依赖高质量AR模型提供teacher signal，若AR模型弱则难以提升；
- 多步denoising尚未充分探索，目前主要聚焦single-step generation。

### 未来工作方向
- 探索**quadratic sampling**或**multi-step diffusion**以进一步提高TPF；
- 研究**inference-time scaling via additional denoising steps**是否能超越AR质量；
- 将该框架扩展至**多模态生成模型**；
- 优化kernel实现以支持更大block size和更高并发；
- 探索**与Multi-Token Prediction（MTP）方法的融合潜力**。

---

> 🔗 **代码与模型已开源**：https://s-sahoo.com/uno

</details>

---

### 7. [LeanStream: A Speculate-and-Refine Streaming Framework for Efficient on-Device LLM Inference](https://arxiv.org/abs/2609.03079)

**Authors**: Renyuan Liu (Richard), Yuyang Leng (Richard), Kaiyan Liu (Richard), Yuzhou Zhong (Richard), Shaohan Hu (Richard),  Chun-Fu (Richard),  Chen, Peijun Zhao, Heechul Yun, Shuochao Yao  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.03079v1  

#### Abstract
On-device LLM inference is attractive for privacy and responsiveness, but remains challenging on mobile and embedded devices because model weights far exceed available DRAM. Prior systems exploit activation sparsity and offload weights to SSD or flash storage, but face a fundamental systems trade-of...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*LeanStream: A Speculate-and-Refine Streaming Framework for Efficient on-Device LLM Inference*

---

## 1. 主要贡献和创新点

### **解决的问题**
在移动和嵌入式设备上进行大语言模型（LLM）推理面临**内存严重受限**的挑战。尽管已有工作利用激活稀疏性（activation sparsity）将权重存储在 SSD 或闪存中，并按需加载执行，但仍存在根本性的系统权衡：
- **准确决策依赖最新上下文**：需要等待前一层完全计算完成才能做出最优的权重加载和执行决策。
- **高效流水线需要提前预测**：为了实现计算与 I/O 的重叠，必须在前一层完成前就预测下一层的激活模式。

这导致现有方法要么**串行化执行**（牺牲效率），要么因预测不准而引入**冗余 I/O 和额外计算**，并带来较大的缓存开销。

---

### **提出的新方法与创新思路**

LeanStream 提出了一种 **Streaming Speculate-and-Refine（流式推测-精炼）框架**，其核心思想是：
> **不再在“等待精确上下文”和“早期粗略预测”之间二选一，而是从初始预测出发，持续利用 GPU 的中间结果逐步精炼计算、加载和缓存保留的优先级。**

具体创新包括：

#### ✅ **Fine-Grained Streaming Control（细粒度流控机制）**
- 设计了基于 **thread-block 级别的非阻塞、异步通信原语**，避免传统 `cudaDeviceSynchronize` 等全局同步带来的高开销。
- 支持 GPU 在部分权重到达后立即开始执行，同时 CPU 可聚合多个已完成的 thread block 结果来更新 I/O 预测。
- 引入 **自适应在线控制器**，动态调整同步频率以平衡预测精度与硬件并行性。

#### ✅ **Stacked Learnable Hashing（堆叠可学习哈希）**
- 提出一种轻量级控制机制，用于快速生成系统控制信号（如神经元重要性排序、缓存驱逐优先级）。
- 相比传统浅层 MLP 控制器，该方法具有更低的延迟和内存占用，基于位运算和寄存器内查表操作，适合资源受限设备。
- 支持端到端可微训练，兼容标准监督学习流程。

#### ✅ **Permutation-Invariant Execution（置换不变执行）**
- 利用 MLP 层输出对隐藏维度顺序不敏感的特性（SwiGLU 输出为求和形式），允许按任意到达顺序处理权重块，无需重排或动态重构 kernel。
- 显著降低乱序加载带来的执行复杂性。

---

### **相比现有方法的优势**
| 维度 | LeanStream | Prior Art（如 DejaVu, PowerInfer-2） |
|------|------------|-------------------------------|
| 决策方式 | 渐进式精炼（speculate-and-refine） | 单次预测（one-shot）或静态调度 |
| 计算-I/O 重叠 | 细粒度、动态重叠 | 粗粒度、固定流水线 |
| 同步开销 | 极低（非阻塞、异步） | 高（依赖 kernel 级同步） |
| 控制器开销 | <100 μs，仅 ~23MB 内存 | >1ms，>1GB 内存 |
| 缓存效率 | 显著更高（智能优先级保留） | 固定策略（LRU/LFU）或简单预测 |

---

## 2. 核心实验方法和设置

### **使用的模型与平台**
- **模型**：Mistral-7B, Llama2-7B, Qwen2.5-7B
- **硬件平台**：
  - 嵌入式：NVIDIA Jetson AGX Orin / Xavier（配 Samsung 980 Pro SSD）
  - 移动端：OnePlus 13（Snapdragon 8 Elite + UFS 4.0）

### **数据集**
- **Scrolls-Qasper**：长文档问答任务（代表长上下文场景）
- **TruthfulQA**：开放事实生成（测试真实性与推理能力）
- **CoQA**：对话式问答（多轮交互场景）

### **评估指标**
- **Token Generation Throughput (tokens/s)**：主性能指标
- **Memory Usage (GB)**：运行时内存消耗
- **Cache Miss Ratio**：衡量缓存策略有效性
- **Redundant I/O / Computation**：冗余加载与计算比例
- **Energy per Token (J/token)**：能效
- **Thermal Behavior**：峰值温度与是否触发降频

### **基线方法对比**
| 方法 | 特点 |
|------|------|
| **DejaVu** | 基于前一层输入预测当前层激活，无缓存优化 |
| **PowerInfer-2** | 结合权重预测与内存缓存缓解 I/O 延迟 |
| **DejaVu+** | 作者改进版，加入 LRU 缓存支持可变内存预算 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔥 **吞吐量提升**
- 在达到先前最佳方法（PowerInfer-2）最高吞吐率时，**LeanStream 进一步提升 1.6×–2.1×**。
- 在极端低内存条件下（如 1–2GB），其他方法出现 OOM，而 LeanStream 仍可运行。

#### 💾 **内存使用大幅下降**
- **内存占用减少 4.8×–7.5×**，例如：
  - PowerInfer-2 需约 3GB 缓存空间，
  - LeanStream 仅需约 0.4–0.6GB 即可达更优性能。

#### 🚦 **缓存效率显著提高**
| Memory Budget | PowerInfer-2 | LRU | Ours (LeanStream) |
|---------------|--------------|-----|------------------|
| 25%           | 0.56         | 0.81| **0.11**         |
| 50%           | 0.19         | 0.39| **0.05**         |

> LeanStream 的缓存未命中率仅为 PowerInfer-2 的 **1/4 到 1/10**。

#### ⏱️ **控制器延迟极低**
| 方法 | 模型 | 大小 | 推理延迟 |
|------|------|------|--------|
| DNN Predictor | Mistral-7B | 1.4GB | 1.41ms |
| BNN Predictor | Mistral-7B | 120MB | 363μs |
| **Ours (Stacked Learnable Hashing)** | Mistral-7B | **24MB** | **92μs** |

> 控制器体积缩小 **58×**，延迟降低 **15×**，且保持相近的加载冗余控制能力（~12–14%）。

#### 📈 **动态流控优于静态配置**
| 配置 | Mistral-7B (tokens/s) | 提升倍数 |
|------|------------------------|---------|
| One-Shot Prediction | 6.2 | — |
| Best Static (离线调优) | 10.7 | 1.7× |
| **Ours (动态流控)** | **16.4** | **2.6×** |

> 动态策略比最优静态配置还高出 **53%** 吞吐。

---

### **消融实验结果**

#### ✂️ **组件消融分析（Ablation Study）**
在 Jetson AGX Orin 上对各模块进行逐步添加测试（20% 权重缓存预算）：

| 组件 | Mistral-7B 吞吐提升 |
|------|--------------------|
| Baseline (DejaVu) | 1.0× |
| + Prioritized Compute | 1.3× |
| + Prioritized Loading | 1.4× |
| **+ Prioritized Caching** | **2.4×** |

> **优先级缓存管理贡献最大**，说明在 I/O 受限环境下，智能保留关键权重是性能瓶颈突破口。

#### 🔁 **渐进式精炼效果验证**
- 将每个 MLP 分为 8 个阶段，每完成一个阶段即更新下一层预测。
- **Top-10% Recall** 和 **Importance Ratio** 随 refinement step 持续上升。
- 第一阶段（2/8）即超越 one-shot 预测，最终接近完整特征预测性能。

---

## 4. 关键结论和发现

### **主要发现**
1. **Speculate-and-Refine 范式有效打破系统瓶颈**：通过渐进式利用部分结果，LeanStream 成功协调了“高精度预测”与“高效流水线”的矛盾。
2. **细粒度控制必须配合低开销机制**：传统的同步机制无法支撑高频协作，必须设计专用的轻量通信 runtime。
3. **缓存策略比预测本身更重要**：即使预测略有误差，只要能精准保留最可能复用的权重，就能极大降低 I/O 开销。
4. **移动端 LLM 推理瓶颈在 decode 阶段**：prefill 虽然密集，但 decode 的序列化特性使其成为长期性能主导因素。

---

### **方法的局限性**
- **依赖特定硬件架构**：目前实现基于 SoC 统一内存（Unified Memory），在分离内存架构上需额外适配。
- **训练成本存在**：stacked learnable hashing 需要离线训练，虽轻量但仍需标注数据与训练流程。
- **对极端稀疏模式敏感**：若实际激活分布偏离训练分布较远，预测质量可能下降。
- **未考虑多模态扩展**：当前框架聚焦纯文本 LLM，向多模态延伸需重新设计特征提取路径。

---

### **未来工作方向**
- 扩展至 MoE 模型中的 expert 路由预测与加载调度。
- 探索跨层联合优化策略，而非逐层独立决策。
- 支持更多设备类型（如 Apple Silicon, RISC-V）。
- 结合 KV Cache 压缩技术进一步降低内存压力。
- 开发自动化工具链，实现从原始模型到 LeanStream 部署的一键转换。

--- 

> ✅ **总体评价**：LeanStream 是面向资源受限设备 LLM 推理的一项系统级突破，它不仅提出了新的 speculative-refinement 范式，而且构建了一套完整的轻量控制、高效通信与智能缓存体系，在真实平台上实现了数量级的内存节省和显著的吞吐提升，为边缘侧高效 AI 推理提供了重要实践路径。

</details>

---

### 8. [Jina-OCR-v1: Efficient Document Parsing with Speculative Decoding and Dense Verifiable Rewards](https://arxiv.org/abs/2609.03181)

**Authors**: Alejandro Bar\'on Garc\'ia, Feng Wang, Emilia Garcia Casademont, Han Xiao  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.03181v1  

#### Abstract
We present Jina-OCR-v1, an end-to-end document parsing model built to serve on low-budget GPUs. It combines the compressed-vision encoder and the 3B mixture-of-experts decoder of DeepSeek-OCR, which activates about 570M parameters per token, with a FastMTP speculative decoding head that shares a sin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Jina-OCR-v1: Efficient Document Parsing with Speculative Decoding and Dense Verifiable Rewards》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 Vision-Language Models (VLMs) 的端到端文档解析模型在实际部署中面临两大挑战：
- **解码成本高**：长输出序列导致自回归解码缓慢，尤其在低预算 GPU 上难以高效运行。
- **后训练监督信号稀疏且不一致**：公开标签存在退化循环、结构错误；公式和表格等特定结构的奖励仅适用于部分样本，导致训练信号稀疏。

### 🚀 提出的新方法与创新思路
Jina-OCR-v1 是一个面向低成本 GPU 部署的高效端到端文档解析模型，其核心创新包括：

#### （1）**FastMTP Speculative Decoding**
- 在 DeepSeek-OCR 架构基础上引入 **FastMTP** 多令牌预测头（multi-token prediction head），采用递归共享单个 draft block 进行 K=3 步预测。
- 优势：
  - 草稿参数数量与预测步数 K 无关，保持恒定，节省显存。
  - 使用贪婪验证（greedy verification）确保解码无损（lossless decoding），即最终输出与原始自回归生成完全一致。

#### （2）**Dense Verifiable Rewards + GRPO 强化学习**
- 设计了一套**密集可验证奖励机制**（dense verifiable rewards），通过确定性代码对输出进行评分，支持部分正确也能获得部分信用。
- 奖励项包括：
  - 内容相似度（normalized edit distance）
  - 公式匹配（formula string matching）
  - 表格结构恢复（TEDS/TEDS-S）
  - 结构完整性（brace balance, tag closure）
  - 单元测试通过率（unit tests）
  - 抗重复与格式一致性
- 使用 **multiplicative GRPO**（带 ReMax baseline）优化策略，避免因单一失败项导致整体奖励归零（通过设置 floor ≥ 0.1–0.2）。

#### （3）**指令导向训练 + 合成数据增强**
- 构建多样化的指令集合，覆盖全页解析、元素级转录、图像描述、VQA 和关键信息提取。
- 引入合成数据集 **JinaOCRSynth**，专门填充高密度公式和复杂表格，提升奖励函数的应用覆盖率。

---

### 🔍 相比现有方法的优势
| 维度 | Jina-OCR-v1 的优势 |
|------|------------------|
| **效率** | 在 NVIDIA L4 等低预算 GPU 上，FastMTP 使解码速度接近翻倍（1.95× 加速）。 |
| **准确性** | 在 OmniDocBench v1.6 和 olmOCR-Bench 上均达到 SOTA 级别，优于同规模甚至更大模型。 |
| **部署友好性** | 激活参数仅 ~570M，整机 <1B 参数，适合边缘设备部署。 |
| **训练有效性** | 利用合成数据和密集奖励显著提升了公式与表格结构的保真度。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
训练数据为混合来源，涵盖真实与合成文档：

| 类型 | 数据源示例 |
|------|-----------|
| **公开 OCR 数据集** | `olmOCR-mix`, `FinePDFs`, `LightOnOCR`, `RVL-CDIP`, `DocLocal4K` |
| **公式与表格专项** | `LaTeX-OCR`, `MMTab`, `SynthChartNet`, `UniMER` |
| **历史/退化文档** | `Europeana newspapers`, `Library of Congress`, `NARA pension files` |
| **多语言与表单** | 中文 PDF、CommonForms、VDR 多域语料 |
| **合成数据** | 自渲染 HTML 页面 + **JinaOCRSynth**（专为奖励覆盖设计） |

### ⚙️ 实验设置与评估指标

#### 评估基准
| 基准 | 主要指标 |
|------|--------|
| **olmOCR-Bench** | 整体得分（overall），含文本存在性、阅读顺序、数学表达式、表格单元测试 |
| **OmniDocBench v1.6** | 综合得分 = 平均（文本编辑距离、公式 CDM、表格 TEDS） |

#### 推理与吞吐量测试
- **硬件平台**：A100（主测）、NVIDIA L4（低预算 GPU 测试）
- **并发设置**：concurrency 32
- **评估维度**：
  - 页面吞吐量（pages/s）
  - 输出 token 数/页
  - 输出 token 吞吐量（tok/s）
  - 解码加速比（vs greedy autoregressive）

#### 基线对比模型
- **通用 VLMs**：Qwen3-VL-235B, Gemini 3 Flash
- **专用 OCR 模型**：DeepSeek-OCR, DeepSeek-OCR-2, PaddleOCR-VL-1.6, HunyuanOCR-1.5, LightOnOCR-2, olmOCR-2, dots.mocr 等

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

| 指标 | Jina-OCR-v1 表现 |
|------|----------------|
| **OmniDocBench v1.6 Overall** | **91.14** |
| **olmOCR-Bench Overall** | **83.4** |
| **页面吞吐量（A100）** | **2.57 pages/s**（所有对比模型中最高） |
| **输出 token/页** | **1085**（最简洁之一） |
| **激活参数量** | **~570M**（MoE 解码器） |
| **FastMTP 加速比（L4, eager mode）** | **1.95×**（K=3） |

### 🔁 与基线方法的对比结果

#### 在 OmniDocBench v1.6 上的表现（Specialized OCR Models）
| Model | Params | Overall |
|-------|--------|---------|
| PaddleOCR-VL-1.6 | 0.9B | **96.34** |
| HunyuanOCR-1.5 | 1B | 94.74 |
| **Jina-OCR-v1** | **3B/570M** | **91.14** |
| DeepSeek-OCR-2 | 3B/570M | 90.25 |
| Qwen3-VL-235B | 235B/22B | 89.78 |

> ✅ **超越更大模型**：尽管参数远小于 Qwen3-VL-235B，仍高出约 1.36 分。

#### 在 olmOCR-Bench 上的表现
| Model | Params | Overall |
|-------|--------|---------|
| chandra-ocr-2 | 4B | **85.8** |
| dots.mocr | 3B | 83.9 |
| **Jina-OCR-v1** | **3B/570M** | **83.4** |
| LightOnOCR-2 | 1B | 83.2 |
| DeepSeek-OCR | 3B/570M | 76.0 |

> ✅ **较基线提升明显**：相比其继承模型 DeepSeek-OCR 提升 **+7.4 分**，证明后训练策略有效。

#### 吞吐量排名（Table 7）
| Model | Pages/s |
|-------|---------|
| **Jina-OCR-v1** | **2.57** |
| DeepSeek-OCR | 2.10 |
| LightOnOCR-2 | 1.33 |
| olmOCR-2 | 1.22 |
| Surya OCR 2 | 1.05（但每页输出高达 3568 tokens） |

> ✅ **最优权衡**：结合较高的 token/s（2792）与最低的输出长度之一（1085 tok/page），实现最高 page throughput。

### 🔬 消融实验与分析（隐含于文中）

虽然未单独列出消融表，但从以下几点可推断各组件贡献：

| 组件 | 贡献证据 |
|------|--------|
| **FastMTP** | 在 L4 上实现 1.95× 加速，验证其对推理效率的关键作用 |
| **Dense Rewards + GRPO** | 相比 DeepSeek-OCR 提升 7.4 分，尤其在公式 CDM（+1.44）和表格 TEDS（+0.79）上表现突出 |
| **JinaOCRSynth 合成数据** | 显著提高公式与表格类任务的训练频率与质量，支撑奖励机制落地 |
| **ReMax Baseline** | 替代 group-normalized baseline，在低方差 rollout 下更稳定，防止噪声放大 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **视觉压缩 + 快速解码 + 密集奖励 = 高效高质量 OCR**
   - Jina-OCR-v1 成功将三者融合，在精度与速度之间取得领先平衡。
2. **FastMTP 可显著加速低预算 GPU 上的解码过程**
   - 在 NVIDIA L4 上达到近两倍加速，且保持 lossless 特性。
3. **Dense verifiable rewards 支持细粒度反馈**
   - 即使输出部分正确也能获得梯度更新，特别有利于结构复杂的公式与表格。
4. **输出长度独立于解析质量**
   - 更短的输出不一定意味着更低的质量——Jina-OCR-v1 是唯一在 >83 分模型中最短输出者。

### ⚠️ 方法的局限性
- **依赖高质量参考标签**：虽然使用合成数据缓解，但真实世界复杂排版仍可能缺乏精确标注。
- **FastMTP 加速受限于 verifier 成本**：当 verifier 步骤本身被高度优化（如 CUDA graph），额外 speculative 步骤收益下降（见 Table 8，graph mode 最佳 K=1）。
- **MoE 激活参数虽少，但总参数仍达 3.4B**：对于极轻量场景仍有压缩空间。

### 🔮 未来工作方向
- 将 FastMTP 扩展至更高 K 或动态调整 K，以适应不同文档复杂度。
- 探索全自动合成数据生成 pipeline，进一步扩大 JinaOCRSynth 规模与多样性。
- 结合 agentic workflow，实现“解析-纠错-重试”闭环训练。
- 开发更轻量化的 vision encoder-decoder 架构，适配移动端部署。

---

> 🔗 **模型开源地址**：[https://huggingface.co/jinaai/jina-ocr-v1](https://huggingface.co/jinaai/jina-ocr-v1)  
> 📘 **论文链接**：arXiv:2609.03181

</details>

---

### 9. [Para-Pipe: Exploiting Hierarchical Operator Parallelism of ML Computational Graphs on SoCs](https://arxiv.org/abs/2609.04168)

**Authors**: Yujie Zhang, Huiying Lan, Ehsan Aghapour, Zhiyuan Ning, Peng Zan, Weidong Shao, Anuj Pathania, Tulika Mitra  
**Category**: cs.DC  
**Published**: 2026-09-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.04168v1  

#### Abstract
As edge-based deep learning applications become more complex, optimizing performance on heterogeneous System-on-Chips (SoCs) presents unique challenges. Traditional pipelining techniques distributing the computation across different on-chip processing units, while effective for throughput, do not ad...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Para-Pipe: Exploiting Hierarchical Operator Parallelism of ML Computational Graphs on SoCs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代边缘设备上的深度学习应用日益复杂，广泛采用具有密集 **operator parallelism**（算子并行性）的神经网络（如 Inception、Transformer）。然而，传统优化策略面临以下挑战：

- **纯流水线（pipelining）** 方法虽能提升 **throughput**，但因跨阶段通信开销大，导致 **latency** 显著增加。
- **纯并行执行（parallel execution）** 虽可降低单帧延迟，却牺牲了多帧推理的吞吐量。
- 现有框架（如 PyTorch、TVM）通常仅利用单一高性能单元顺序执行，未能充分利用异构 SoC 上的多类计算单元（CPU、GPU、NPU、DSP）。

因此，如何在 **latency** 和 **throughput** 之间实现灵活且高效的权衡，并提升 **energy efficiency**，是当前的关键瓶颈。

---

### 🚀 提出的新方法：Para-Pipe

Para-Pipe 是一种**分层映射框架**，通过在流水线架构中融合 **intra-stage**（阶段内）和 **inter-stage**（阶段间）的算子并行性，实现对 **latency-throughput trade-off** 的细粒度控制。

#### 核心创新点：
1. **分层映射架构（Hierarchical Mapping）**
   - **Pipeline Mapping**：将计算图划分为多个拓扑有序的子图（subgraphs），每个子图构成一个 pipeline stage。
   - **Operator Mapping**：在每个 stage 内部，进一步将子图中的算子并行分配到多个处理单元上，最大化 **intra-stage parallelism**。

2. **双粒度 ILP 映射算法**
   - **粗粒度映射（Coarse-grained）**：以“分支”为单位进行映射，减少通信开销，适合规则结构（如 Inception 模块）。
   - **细粒度映射（Fine-grained）**：逐个算子映射，灵活性高，适用于复杂不规则模型（如 Transformer）。

3. **Pareto 最优策略选择**
   - 通过成本估计器建模 **computation cost** 和 **communication cost**，生成多个在 latency、throughput、energy efficiency 上 Pareto 最优的配置供用户选择。

---

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **性能权衡** | 支持灵活调节 latency 与 throughput，避免“非此即彼”的妥协 |
| **能效提升** | 减少跨处理器通信，显著提高 energy efficiency（最高达 23.3%） |
| **适用性广** | 支持不规则图结构（如 PETR、BEVFormer），优于仅支持线性模型的传统流水线方法 |
| **平台兼容性** | 在 Amlogic SoC（CPU/GPU）和 BST SoC（NPU/DSP）上均有效验证 |

---

## 2. 核心实验方法和设置

### 📊 数据集与模型
使用六种具有密集算子并行性的现代 DNN 模型进行评估：

| 模型 | 类型 | 特点 |
|------|------|------|
| GoogLeNet, Inception-v3, Inception-v4, Inception-ResNet-v2 | CNN with Inception modules | 高度分支化结构，典型 intra-operator parallelism |
| PETR-based, BEVFormer-based | Transformer-based 3D 检测 | 复杂不规则连接，用于自动驾驶感知 |

这些模型被自动划分为若干 **subgraphs**（最多 45 个），作为 pipeline 分区的基础。

---

### ⚙️ 实验平台
1. **Amlogic A311D SoC**（真实硬件）
   - 架构：ARM big.LITTLE CPU（Cortex-A73 + A53）+ ARM G52 GPU
   - 工具链：基于 **ARM Compute Library (ARM-CL)** 实现运行时调度
   - 测量工具：USB power meter（功耗）、TinyMemBench / clpeak（内存带宽）

2. **Black Sesame Technology (BST) A1000 SoC**（仿真）
   - 架构：NPU + 2× DSP（其余 2 DSP 保留给其他任务）
   - 工具：使用厂商提供的 **operator simulator** 进行性能与通信成本预测

---

### 📈 评估指标
| 指标 | 定义 |
|------|------|
| **Latency** | 单帧推理时间（seconds/frame） |
| **Throughput** | 每秒处理帧数（FPS） |
| **Energy Efficiency** | 推理请求次数每焦耳能量（frames/joule） |
| **Active Power** | 所有激活处理器的总功耗（Watts） |
| **RMSPE** | 成本估计器的预测误差（Root Mean Squared Prediction Error） |

---

### 🔁 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **pipe-only** | 传统流水线方法，最大化 throughput |
| **para-only** | 全并行执行，最小化 latency |
| **Layer-switched** | 层级切换执行，代表最优顺序执行上限 |
| **HEFT & CPOP** | 经典 DAG 映射算法，用于比较并行调度效果 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Amlogic SoC）

| 指标 | 结果 |
|------|------|
| **Latency 改进** | `hybrid-L` 相比 `pipe-only` 平均降低 **36.0%** |
| **Throughput 改进** | `hybrid-T` 相比 `para-only` 平均提升 **11.9%** |
| **Energy Efficiency** | `hybrid-T` 较 `pipe-only` 提升 **11.0%**，较 `para-only` 提升 **23.3%** |
| **预测准确性** | 成本估计器 RMSPE：latency 15.33%，throughput 15.25%，energy 6.40% |

> 💡 特别案例：在 Inception-v4 上，`hybrid-T` 不仅 throughput 超过 `pipe-only`，且 latency 减半。

---

### 🆚 与基线方法对比（归一化至 para-only）

| 方法 | Latency | Throughput | Energy Efficiency |
|------|---------|------------|-------------------|
| **pipe-only** | ↑113.8% | ✅ 最高 | ↑12.2% |
| **hybrid-L** | ↓36.0% | ↓12.4% | ↑16.7% |
| **hybrid-T** | ↓26.8% | ↓7.3% | ↑23.3% |
| **Layer-switched** | ↓~10% | ↓~12% | ↓24.5% |
| **HEFT & CPOP** | ↓~15% | ↓~18% | ↓8.0% |

> ✅ Para-Pipe 的 hybrid 配置在三项指标上均优于传统方法。

---

### 🔬 消融实验与分析

#### （1）粗粒度 vs. 细粒度映射
- **粗粒度**：同步开销低，适合 CPU-GPU 协同，program overhead 平均 5.9%
- **细粒度**：在双 CPU 集群上表现更优，平均提升 **3.35% latency** 和 **4.28% throughput**
- 含小 CPU 可减少 GPU 同步点 20.1%，降低 jitter

#### （2）异构单元协同效率
- CPU + GPU 并行受限于数据格式转换（OpenCL tensor）和地址映射
- CPU + 小 CPU 组合通信无额外开销，更适合并行执行
- 因此 Para-Pipe 在 stage 内优先组合架构相似的单元

#### （3）映射求解时间（Fine-grained on BST SoC）
| 模型 | 子图数 | 算子数 | ILP 求解时间（分钟） |
|------|--------|--------|------------------|
| GoogLeNet | 11 | 141 | <1 |
| Inception-v3 | 13 | 220 | 4 |
| PETR-based | 24 | 337 | 361（约 6 小时）|

> ✅ 大多数模型可在 5 分钟内完成调度；最大子图（169 算子）需 6 小时，但可通过并行求解加速。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **分层并行是解决 latency-throughput 权衡的有效路径**  
   Para-Pipe 通过 **intra-** 和 **inter-stage parallelism** 的协同，实现了传统方法无法兼顾的性能平衡。

2. **hybrid 执行模式显著提升 energy efficiency**  
   通过减少跨 stage 通信和合理分配串行/并行任务，hybrid-T 在 Amlogic SoC 上实现 **23.3%** 的能效增益。

3. **平台特性决定最优映射策略**  
   - 架构兼容性（如 CPU 与 GPU 数据格式差异）直接影响通信开销；
   - NPU 对部分算子不支持时，Para-Pipe 可自适应地结合 DSP 进行补偿。

4. **成本估计器具备足够精度指导决策**  
   尽管 latency 预测误差约 15%，但其对不同策略的**相对排序准确**，足以支撑 Pareto 最优选择。

---

### ⚠️ 方法的局限性
1. **静态映射，缺乏运行时动态调整能力**  
   当前 Para-Pipe 为一次性离线调度，无法应对输入变化或资源竞争等动态场景。

2. **大规模复杂子图求解时间较长**  
   如 PETR 模型的最大子图需 6 小时 ILP 求解，虽可并行优化，但仍影响部署敏捷性。

3. **依赖精确的 operator profiling 数据**  
   性能建模高度依赖离线采集的 computation 和 communication cost，若平台变化需重新校准。

---

### 🔮 未来工作方向
1. **引入 runtime 动态调度机制**  
   结合 work-stealing 或 feedback control，在线调整 stage 划分与资源分配。

2. **轻量化 ILP 求解器或替代优化方法**  
   探索基于 RL 或启发式算法的快速近似求解，缩短映射时间。

3. **扩展至 multi-DNN workloads**  
   支持多个模型并发执行下的资源争用管理与调度协调。

4. **支持更多硬件后端（如 TPU、Neuromorphic chips）**  
   增强框架的通用性和可移植性。

---

> **总结**：Para-Pipe 提出了一种新颖的 **hierarchical operator parallelism** 利用方式，成功在异构 SoC 上实现了 **latency、throughput、energy efficiency** 的多目标优化，为复杂边缘 AI 推理提供了实用且高效的系统级解决方案。

</details>

---

### 10. [Speculative Macro Commit for Faster Tool-Using Agents](https://arxiv.org/abs/2609.03236)

**Authors**: Zeyu Liu, Souvik Kundu, Peter A. Beerel  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.03236v1  

#### Abstract
Tool-using LLM agents spend wall-clock time not only on model inference but also in serial action--observation turns, where each tool call, environment transition, and observation can delay subsequent decisions. We introduce \textbf{Speculative Macro Commit} (SMC), a runtime mechanism for a two-tier...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Speculative Macro Commit for Faster Tool-Using Agents

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
当前的 **tool-using LLM agents** 在执行任务时面临显著的 **wall-clock latency** 问题。这种延迟不仅来自模型推理本身，更源于 **串行的 action-observation 循环**：每一步都必须等待前一个工具调用完成、环境返回观察结果后才能进行下一步决策。

尽管已有工作如 **Speculative Actions (SA)** 尝试通过预测未来动作来并行执行，但其仅支持单步推测提交（single-step commit），无法有效利用多步重复的行为模式，限制了提速潜力。

---

### 🚀 提出了什么新方法或新思路  
本文提出 **Speculative Macro Commit (SMC)** ——一种运行时机制，用于加速两层架构下的 agent 系统：

- **权威 Actor 模型**（large authoritative actor）：负责生成正式轨迹，保证行为正确性。
- **快速推测 Drafter 模型**（fast speculative drafter）：在隔离环境中超前预测并执行未来的 action chain。

**核心创新点**：
1. **Macro Mining**：从训练轨迹中挖掘出频繁出现的多步 action 序列（称为 *macro*），构建 macro library。
2. **Macro Commit 机制**：当 drafter 预测的动作链中匹配到某个 macro，且 actor 的首个动作与该 macro 的第一个动作一致时，SMC 可以“提交”后续已预执行的多个步骤及其观测结果，跳过对应的模型调用和环境等待。
3. **Runtime-Level Optimization**：macro 的触发和提交完全由 executor 控制，无需修改模型输出空间或要求模型主动选择 meta-tool。

---

### 🔍 相比现有方法的优势
| 方法 | 是否需模型支持 | 支持多步提交 | 是否损失精度 |
|------|----------------|--------------|---------------|
| Sequential Execution | 否 | ❌ 单步 | 否 |
| Speculative Actions (SA) | 否 | ❌ 单步 | 否（lossless） |
| AWO-style Meta-Tools | 是 | ✅ 多步 | 易失败（模型很少选用） |
| **SMC (本文)** | **否** | ✅ **多步** | **近似优化（approximate），但实证保持质量** |

> ✅ **优势总结**：
> - 不依赖模型学会使用新的 meta-tool；
> - 能复用已执行的多步推测结果，显著减少 wall-clock time；
> - 提交过程受多重保护（anchor verification + online checks），保障可靠性。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
1. **T2-Bench Telecom subset**  
   - 专注于电信领域对话控制任务。
   - 强调多轮工具协调能力。
2. **AppWorld**  
   - 一个可控的应用程序交互环境，模拟真实手机 App 操作。
   - 包含复杂状态转移和 API 调用序列。

---

### ⚙️ 实验设置
- **Actor Model**: `Qwen3.5-27B INT4`（量化版，单 GPU 可运行）
- **Drafter Model**: `Qwen3.5-4B`（更快，用于推测）
- **解码方式**：greedy decoding
- **硬件配置**：
  - Baseline：1 GPU（仅 actor）
  - SA / SMC：3 GPUs（actor + replica for speculative requests + drafter）

---

### 🎯 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy / Task Completion** | T2-Bench: 二元任务准确率；AppWorld: Task Goal Completion (TGC) |
| **Latency** | 平均每任务 wall-clock 时间（秒） |
| **△ (%)** | 相对于 sequential baseline 的延迟降低百分比 |

---

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **Sequential Baseline** | 标准串行动作循环，无任何推测 |
| **SA (Speculative Actions)** | 单步推测提交，drafter 可提前执行一步，若匹配则复用 |
| **SMC (本文)** | 在 SA 基础上引入 macro commit，支持多步提交 |

此外还进行了与 **AWO-like meta-tool** 和 **passive committing** 的消融比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1）

| Benchmark | Run | Acc. | Lat. (s) | △ (%) |
|----------|-----|-------|-----------|--------|
| **T2Telecom** | Baseline | 99.52% | 27.60 | — |
|             | SA       | 99.47% | 25.03 | -9.31% |
|             | **SMC**  | **99.52%** | **22.47** | **-18.59%** ✅ |
| **AppWorld** | Baseline | 41.67% | 355.7 | — |
|            | SA       | 41.67% | 212.1 | -40.37% |
|            | **SMC**  | 40.48% | **195.9** | **-44.93%** ✅ |

> ✅ **结论**：
> - 在 T2Telecom 上，SMC **保持精度不变**，延迟比 baseline 下降 **18.59%**，比 SA 再提升 **10.23%**。
> - 在 AppWorld 上，SMC 进一步将延迟压低至 **195.9s**（↓44.93%），虽有轻微准确率下降（↓1.19%），但在相同结果的任务子集上提速达 **13.5%**。

---

### 🔍 消融实验结果

#### （1）不同接口设计对比（Table 4）
| 方法 | 准确率 | 延迟 | 结论 |
|------|--------|--------|--------|
| Baseline | 99.52% | 27.60s | — |
| + AWO-like meta-tools | 99.34% | ↑27.89s (+1.05%) | 模型几乎不选 macro，无效甚至拖慢 |
| + Passive committing | 96.48% | ↓24.47s (-11.34%) | 忽略验证导致严重错误 |
| → **SMC（完整规则）** | **99.52%** | **22.47s** | ✅ 高效且安全 |

> 💡 **发现**：隐藏 runtime state + anchor verification 是关键。

---

#### （2）Commit Precision 分析（Table 5）
逐步过滤机制极大提升了提交准确性：

| 过滤阶段 | 匹配事件数 | 正确率（exact match） |
|--------|------------|------------------|
| Library match only | 1,968 | 34.6% ❌ |
| + drafter 已执行 | 885 | 70.6% ✅ |
| + anchor call verified | 711 | 87.9% ✅✅ |
| + depth guard (Lmin=1) | 343 | 90.4% ✅✅✅ |
| **最终实际提交** | **158** | **100.0%** ✅✅✅✅ |

> ✅ 所有被真正 commit 的 macro 均未改变任务结果，说明在线检查机制非常有效。

---

#### （3）Critical-Path Depth 影响（Table 6）
| 方法 | 提交次数 | 跳过步数 | 延迟 | 对比 SA |
|------|---------|----------|--------|--------|
| SA | 0 | 0 | 25.03s | — |
| Legacy SMC（浅层提交多） | 6,410 | 10,528 | 25.44s | **+1.64%** ❌ |
| **Final SMC（深度过滤）** | **3,352** | **7,154** | **22.47s** | **-10.23%** ✅ |

> 💡 **关键发现**：提交频率高 ≠ 更快！只有足够深（Lmin ≥ 2）、位于关键路径上的 commit 才能带来端到端提速。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **多步推测可安全加速 agent 执行**：通过 SMC，可以在不修改模型的前提下，复用已执行的多步推测动作，实现高达 **18.6%~44.9%** 的 wall-clock 时间压缩。
2. **macro 应作为 runtime 隐藏状态而非 model-visible 工具**：直接暴露 mined macro 为 meta-tool 效果差（模型不用）；而由 executor 自动识别并提交更高效。
3. **提交机制需要严格守卫**：仅靠“pattern 存在”不足以提交，必须结合：
   - drafter 实际已执行
   - anchor call 被 actor 验证
   - 最小跳过长度（Lmin）
   - online state/argument checks
4. **SMC 在相似轨迹上收益更大**：在任务结果不变的子集上，SMC 加速效果更明显（AppWorld 达 -13.5%）。

---

### ⚠️ 方法的局限性
1. **近似优化（approximate）**：不是严格 lossless，存在极小概率因跳过步骤导致行为偏移（如 AppWorld 中丢失 2 个任务）。
2. **依赖高质量的 macro mining 和 drafter 表现**：若 drafter 无法稳定重现 macro，则无法触发 commit。
3. **对非重复性任务增益有限**：适用于具有规律性操作流程的任务（如电信服务、固定 App 流程），对高度开放任务帮助较小。

---

### 🔮 未来工作方向
1. **动态 macro learning**：在线学习新的 macro 模式，适应不断变化的任务分布。
2. **跨任务迁移 macro**：将在一个 domain 中学到的 macro 泛化到其他相似场景。
3. **结合 program-aware serving**：与 KVFlow、ThunderAgent 等系统集成，进一步优化缓存与调度。
4. **提升 robustness**：研究如何检测和回滚错误提交，增强 fault tolerance。

---

> ✅ **总体评价**：  
> SMC 提供了一种**实用、高效、无需改动模型结构**的方式来加速 tool-using agents，是 speculative execution 从 token-level 到 action-level 再到 **macro-level** 的重要推进。代码已开源，具备良好落地潜力。

</details>

---

### 11. [GrowPage: On-Demand KV Budgeting for Efficient LLM Reasoning Serving](https://arxiv.org/abs/2609.03494)

**Authors**: Qiankun Ma, Yanjiang Zhou, Zinan Xiong, Haofei Wang, Zhen Song, Yang Xiang, Ziyao Zhang, Hairong Zheng  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.03494v1  

#### Abstract
Long-output reasoning has made the key--value (KV) cache a critical memory bottleneck for efficient LLM serving. Existing KV compression methods usually rely on a predefined per-request budget and adjust only which KV states are retained, leaving the total capacity fixed throughout decoding. However...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GrowPage: On-Demand KV Budgeting for Efficient LLM Reasoning Serving**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
大型推理模型（Reasoning LLMs）在生成长链思维（Chain-of-Thought, CoT）时会持续扩展其 **Key-Value (KV) Cache**，导致显存占用成为高效服务（efficient serving）的关键瓶颈。现有的 **KV Cache 压缩方法**（如 Zipage、R-KV 等）通常依赖于**预设的固定请求级 KV 容量预算**（fixed per-request budget），无法适应以下两个现实挑战：

- **跨请求异质性**（Inter-request heterogeneity）：不同请求对 KV 容量的需求差异巨大。
- **单请求内动态变化**（Intra-request temporal variation）：一个请求在解码过程中，注意力集中程度会随时间演变。

这种静态预算机制要么过度分配资源给低需求请求（wasted memory），要么在高需求阶段不足（under-provisioning），从而限制了系统吞吐量和推理准确性之间的平衡。

---

### **提出了什么新方法或新思路**
本文提出 **GrowPage**，一种**按需 KV 预算框架**（On-Demand KV Budgeting Framework），将 KV 容量视为运行时可变资源而非静态预留。

#### **核心创新点：**
1. **双时间尺度查询摘要**（Dual-Timescale Query Summaries）
   - 维护轻量级的短期（short-timescale）和长期（long-timescale）查询表示（EMA 平滑）。
   - 利用两者诱导的历史注意力工作集（working set）的相对差异来估计当前注意力需求趋势。

2. **在线容量控制策略**
   - 在每个 KV 容量边界（capacity boundary）处，基于需求趋势信号 $\Delta_t$ 决定：
     - **Compress & Hold**：若注意力趋于集中，则压缩历史 KV 状态以腾出空间。
     - **Grow by One Page**：若注意力趋于扩散，则申请一个新的物理 KV 页面。

3. **与 PagedAttention 深度协同设计**
   - 与 **PagedAttention** 的页级内存抽象无缝集成，支持：
     - 连续批处理（continuous batching）
     - 前缀缓存（prefix caching）
     - CUDA Graph 执行
   - 不破坏现有高性能推理引擎的系统优化。

---

### **相比现有方法的优势**
| 特性 | 固定预算方法（如 Zipage） | GrowPage |
|------|--------------------------|---------|
| KV 容量 | 静态预设，全程不变 | 动态按需调整 |
| 资源利用率 | 易出现过配或欠配 | 更好匹配实际需求 |
| 系统兼容性 | 多数不兼容现代 serving 机制 | 兼容 PagedAttention 及其生态 |
| 性能-吞吐权衡 | 固定折中点 | 可动态优化 |

> ✅ **优势总结**：GrowPage 在保持推理准确性的前提下显著提升吞吐量，并更好地利用有限 GPU 内存支持更多并发请求。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
涵盖数学推理与代码生成任务，具体包括：
- **数学推理**：
  - GSM8K
  - MATH500
  - AMC23
  - AIME24
- **代码生成**：
  - LiveCodeBench

这些基准覆盖了多样化的推理负载，具有不同的输出长度和 KV 缓存压力。

---

### **实验设置和评估指标**

#### **模型**
- DeepSeek-R1-Distill-Llama-8B
- Qwen3-8B

#### **硬件环境**
- 单张 NVIDIA A100 GPU（80GB 显存）
- GPU 内存利用率固定为 0.9，确保公平比较

#### **实现基础**
- 基于 **nano-vLLM + PagedAttention** 构建
- KV block size = 256 tokens
- 最近保留窗口 $R_t$ = 16 tokens

#### **评估指标**
| 指标 | 含义 |
|------|------|
| **Pass@1 Accuracy (%)** | 推理任务首次生成即正确的比例 |
| **TPS (Tokens/s)** | 输出吞吐量，衡量每秒生成的 token 数量 |
| **Total Inference Time (h)** | 整个工作负载完成所需总时间 |
| **TPOT (ms/token)** | 每个输出 token 的平均延迟 |
| **Resident Concurrency (C)** | GPU 上同时驻留的请求数量 |

---

### **基线方法对比**
| 方法 | 类型 | 是否兼容 vLLM |
|------|------|---------------|
| FullKV (vLLM / nano-vLLM) | 无压缩 | 是 |
| MorphKV (ICML’25) | KV 压缩 | 否 |
| R-KV (NeurIPS’25) | KV 压缩 | 否 |
| G-KV (arXiv’25) | KV 压缩 | 否 |
| Zipage (ACL’26) | 压缩 + PagedAttention | 是 |
| **GrowPage (Ours)** | **按需扩容 + PagedAttention** | **是** |

> 注：所有基线使用相同内存预算；非 vLLM 兼容方法采用最大可行 batch size 测试。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

#### **DeepSeek-R1-Distill-Llama-8B 结果**
| 方法 | Avg. Pass@1 (%) | Avg. TPS (tokens/s) |
|------|------------------|---------------------|
| FullKV (vLLM) | 69.2 | 1472 |
| Zipage | 67.6 | 1936 |
| **GrowPage (Ours)** | **68.7** | **2417** |

> 📈 **提升**：相比 FullKV，**吞吐 +64.3%**，精度基本持平  
> 🔺 相比 Zipage，**吞吐 +24.8%**，精度更高（+1.1%）

#### **Qwen3-8B 结果**
| 方法 | Avg. Pass@1 (%) | Avg. TPS (tokens/s) |
|------|------------------|---------------------|
| FullKV (vLLM) | 88.7 | 1357 |
| Zipage | 86.5 | 1853 |
| **GrowPage (Ours)** | **87.5** | **2174** |

> 📈 **提升**：相比 FullKV，**吞吐 +60.2%**，精度略降但可控  
> 🔺 相比 Zipage，**吞吐 +17.3%**，精度更高（+1.0%）

---

### **与基线方法的对比结果**

#### **混合工作负载下的帕累托优势（图5）**
- 构造混合负载（GSM8K + AMC23 + AIME24）
- GrowPage 实现：
  - **84.9% Pass@1 @ 2213 tokens/s**
  - 超越 Zipage-4096（83.7%, 1690 tokens/s）
  - 接近 Zipage-8192（85.1%）但吞吐高出 **~1.9×**

> 💡 表明：自适应预算能更高效地将有限内存转化为并发性和吞吐量。

#### **消融实验结果**

##### **(1) 需求趋势信号有效性验证（图6 & 图7）**
- **图6**：$\Delta_t$ 与未来注意力需求变化呈强相关（Spearman ρ = 0.786，符号一致率 80.3%）
- **图7**：$\Delta_t$ 越大，扩容带来的预测损失下降越明显 → 证明信号可靠

##### **(2) 双时间尺度设计消融（表2）**
| $\beta_s$ | $\beta_L$ | Working Set | AMC23 Pass@1 | TPS |
|-----------|-----------|-------------|--------------|-----|
| 0.0 | 0.999 | Top-p | 89.9 | 2394 |
| 0.9 | 0.999 | Top-p | **91.4** | **2261** |
| 0.9 | 0.999 | Entropy | 90.5 | 2179 |

> ✅ 引入适度短期平滑（$\beta_s=0.9$）显著提升准确率  
> ❌ 使用熵作为工作集指标效果更差

##### **(3) 局部保留配额影响（表4）**
| $k_{\text{loc}}$ | AMC23 Pass@1 | AIME24 Pass@1 |
|------------------|---------------|----------------|
| 16 | 90.3 | 72.4 |
| 64 | **91.4** | **73.8** |
| 128 | 90.6 | 72.8 |

> ✅ $k_{\text{loc}}=64$ 达到最佳平衡：既保证局部覆盖率又不失全局重要性选择灵活性

---

## **4. 关键结论和发现**

### **主要发现**
1. **KV 需求是动态且异质的**：
   - 不同请求间最小必要 KV 预算跨度极大（见图2）
   - 单个请求内部注意力模式随时间演化（集中 ↔ 扩散）

2. **GrowPage 成功捕捉并响应需求变化**：
   - 双时间尺度摘要有效估计注意力趋势
   - $\Delta_t$ 是可靠的扩容决策信号

3. **动态容量优于固定预算**：
   - 在相同内存约束下，GrowPage 实现更高的 **accuracy-throughput trade-off**
   - 支持更多并发请求（higher resident concurrency）

4. **系统开销极低**：
   - 每次容量决策仅增加约 **5.84–19.05ms** 开销
   - 占解码时间 < 0.32%，可忽略不计

---

### **方法的局限性**
1. **仅支持单向扩容**（monotonic growth）：
   - 不主动回收页面（避免不可逆信息丢失）
   - 在极端内存压力下可能导致 preemption 增加（见 H.2）

2. **依赖 PagedAttention 抽象**：
   - 当前实现绑定于特定 serving 引擎（如 vLLM）

3. **未探索多步前瞻预测**：
   - 决策基于当前趋势，缺乏对未来多步需求的建模

---

### **未来工作方向**
1. **双向动态调整机制**：
   - 设计安全的页面回收策略（如结合重要性评分）

2. **跨请求容量调度器**：
   - 全局感知内存压力，协调多个 GrowPage 请求间的资源竞争

3. **与量化/稀疏化联合优化**：
   - 将 GrowPage 与 KIVI、KVQuant 等低比特量化技术结合

4. **扩展至其他注意力架构**：
   - 支持 MQA/GQA/MHA 混合场景下的统一预算管理

---

> ✅ **总体评价**：GrowPage 提出了一个新颖且实用的视角——将 KV 容量从“静态预算”转变为“运行时资源”，并通过轻量级信号实现高效的在线调控，在不影响系统兼容性的前提下显著提升了 LLM 推理服务效率。该思想有望成为下一代高效推理系统的标准组件之一。

</details>

---

### 12. [LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening](https://arxiv.org/abs/2609.04013)

**Authors**: Muhammad Ashad Kabir, Sirajam Munira  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.04013v1  

#### Abstract
Early screening of chronic kidney disease (CKD) is critical for timely intervention, yet most machine learning (ML) and deep learning (DL) approaches require labeled data and model training, limiting their use in real-world screening settings. This study evaluates the effectiveness of large language...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **低资源环境下早期慢性肾病（CKD）筛查困难**：传统机器学习（ML）和深度学习（DL）模型依赖大量标注数据和固定特征集，在资源受限地区（如LMICs）难以部署。
- **现有CKD风险评分工具泛化能力差**：多数基于高收入国家人群开发，对早期CKD敏感度低，且不适用于异构、缺失特征的临床场景。
- **LLMs在定量医学任务中的潜力未被系统探索**：尽管LLMs在临床推理中表现良好，但其作为**零样本/少样本定量筛查模型**的能力尚未充分验证。

### 提出的新方法与思路
- **提出LLM4CKD框架**：首次系统评估LLMs在**zero-shot**和**few-shot in-context learning**设置下用于早期CKD筛查的有效性。
- **结合临床特征选择与结构化提示工程**：
  - 使用经ML分析筛选的**clinically meaningful features**构建输入。
  - 设计多种**prompt template**（list/text-based serialization）和**prompt style**（instruction/chat-style），实现无需微调的LLM推理。
- **引入可解释性分析**：通过SHAP surrogate模型比较LLM与传统ML的feature importance排序，评估其决策是否符合临床逻辑。

### 相比现有方法的优势
- **数据效率高**：仅需极少量标注样本（甚至为零）即可达到有竞争力的性能，适合标注稀缺场景。
- **灵活性强**：支持动态输入配置，适应不同医疗环境下的特征可用性变化。
- **无需训练**：避免复杂的模型训练流程，降低部署门槛。
- **概率输出质量优**：部分LLM在Brier loss等校准指标上优于传统规则型筛查工具。

---

## 2. 核心实验方法和设置

### 数据集
| 数据集 | 来源 | 样本数 | 特征数 | 描述 |
|-------|------|--------|--------|------|
| **Dataset-1** | 孟加拉国社区队列（Mirzapur） | 284人（112 CKD + 172 非CKD） | 24 → 选后9个 | 主要评估集，聚焦**早期阶段CKD**（stages 1–3），排除eGFR/uACR等诊断标记防止标签泄露 |
| **Dataset-2** | 印度医院UCI CKD数据集 | 400人（250 CKD + 150 非CKD） | 23 → 选后匹配9个 | 外部独立验证集，用于跨数据集评估，但无明确分期信息 |

> ✅ 所有特征经过**harmonization**处理以统一语义编码。

### 实验设置
- **Zero-shot setting**：仅提供任务描述和查询患者记录，无示例。
- **Few-shot setting**：在查询前插入4~32个带标签的上下文示例（in-context examples）。
- **训练/测试划分**：80%/20%分层抽样，五次随机种子重复（{0,1,32,42,1024}）确保可比性。
- **所有模型使用相同特征集（全量 vs. 精选）进行公平比较**。

### 评估指标
- **Balanced Accuracy**（平衡准确率）
- **AUROC**
- **Macro-F1**
- **Sensitivity**
- **Brier Loss**（↓越小越好，衡量概率预测准确性）
- **Expected Calibration Error (ECE)**（↓越小越好，衡量置信度校准程度）
- 统计检验：paired permutation test 和 mixed-effects model 分析

### 基线方法对比
| 类别 | 模型列表 |
|------|---------|
| **LLMs** | Gemma-2-9B, Llama-3-8B, Qwen-3-8B, Mistral-7B, GPT-4o-mini |
| **Traditional ML** | Logistic Regression (LR), Random Forest (RF), Extra Trees (ET), XGBoost (XGB), Gradient Boosting (GB), AdaBoost (AB), Decision Tree (DT), LightGBM (LGB) |
| **Deep Learning / Tabular Foundation Models (TFMs)** | MLP, TabNet, NODE, TabPFN, SAINT |
| **CKD Screening Tools** | SCORED [1], Kshirsagar [11], Thakkinstian [30], Kwon [13], Kearns [10]（均为规则型风险评分） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Selected Features Setting）

#### （1）Few-shot 性能趋势（Fig. 3）
- **LLMs在极低数据下表现强劲**：
  - Qwen-3、Gemma-2、GPT-4o-mini 在 **4-shot** 即达 ~0.80 平衡准确率。
  - Qwen-3 在4-shot时取得最高平衡准确率 **0.814 ± 0.025** 和高灵敏度 **0.873**，显示其在极端低资源下检测早期病例能力强。
- **更多shot不一定更好**：
  - Llama-3 和 Mistral 表现随shot增加而波动或下降，表明对上下文示例组合敏感。
  - GPT-4o-mini 最稳定，保持约0.80准确率。

#### （2）Brier Loss 对比（Fig. 4）
- **Qwen-3 表现最佳**：
  - 四项shot设置下Brier loss均最低（最低达 **0.141 @16-shot**），显著优于多数其他LLM及传统ML/DL模型。
- **TabPFN 和 NODE 在大数据量下反超**：
  - 当训练样本增至32时，TabPFN 达到 **0.820** 准确率和 **0.122** Brier loss，成为最强基线。
- **MLP虽分类尚可，但校准较差**：Brier loss持续高于Qwen-3和TabPFN。

#### （3）与现有CKD筛查工具对比（Table VI）
| LLM | Bal. Acc. | vs. 工具平均ΔBrier Loss |
|-----|-----------|--------------------------|
| **Qwen-3** | 0.7829 | **-0.111 ~ -0.206*** |
| **Mistral** | 0.7925 | **-0.110 ~ -0.205*** |
| GPT-4o-mini | 0.7725 | -0.059 ~ -0.154** |
| Gemma-2 | 0.7299 | -0.095 ~ -0.190*** |
| Llama-3 | 0.7422 | 仅对SCORED/Kwon显著 |

> ✅ 负值表示LLM概率预测更优；`***` p<0.001  
> 🔍 **即使某些LLM分类准确率略低于Thakkinstian（0.8103），其概率输出仍显著更优**

#### （4）消融实验结果
- **特征选择的影响（Fig. 2 & Fig. 6a）**
  - 多数LLM（Llama-3, Mistral, Qwen-3, GPT-4o-mini）在**精选特征**下性能提升或持平。
  - Llama-3在Chat+Text设置中从0.51→0.74，提升巨大。
  - **Gemma-2例外**：多数情况下使用全特征更好，说明其可能依赖更广上下文。
- **Prompt Style 影响**
  - Instruction-style 对开源LLM更有效。
  - Chat-style 是GPT-4o-mini唯一可用方式。
- **跨数据集泛化（Fig. 6b）**
  - 在Dataset-2上，多数LLM few-shot性能优异（0.84–0.92）。
  - TabPFN继续展现强大扩展性，从4→32 context样本准确率从0.85升至0.94。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLMs可在零/少样本下实现有竞争力的CKD筛查性能**，尤其适合标注数据稀少的低资源环境。
2. ✅ **精选临床特征 + 结构化提示设计** 可显著提升LLM表现，减少噪声干扰。
3. ✅ **Qwen-3、Mistral、GPT-4o-mini 表现最优**，其中Qwen-3在概率校准方面尤为突出。
4. ⚖️ **存在“数据效率 vs. 可扩展性”权衡**：
   - LLMs：**低数据高效**，但增益饱和甚至退化。
   - ML/DL/TFMs：需要更多数据，但随数据增长稳步提升。
5. 🧠 **LLM特征重要性具有一定临床合理性**：
   - 多数LLM重视Diabetes、Anemia、Obesity等公认风险因素。
   - 但与MLP参考排名相关性弱（Spearman ρ ≈ 0.1–0.5），说明其决策机制不同于纯数据驱动模型。
6. 🏆 **顶级LLM在概率预测质量上超越传统CKD筛查工具**，即便分类准确率相近。

### 局限性
- **样本量较小**：主队列仅284人，影响稳定性。
- **LLM结果受prompt设计、序列化方式、版本更新影响大**，复现性和鲁棒性有待加强。
- **特征子集来自前期ML研究**，未必普适于所有人群。
- **外部数据集来源不同（社区vs医院）、无分期信息**，限制跨域结论强度。
- **未直接测试缺失特征、异构输入、部署延迟等现实挑战**。
- **缺乏前瞻性临床验证**，尚不能用于实际诊疗。

### 未来工作方向
- 探索LLM对**missing features**和**heterogeneous inputs**的鲁棒性。
- 优化**prompt engineering pipeline**自动化与标准化。
- 开发面向LLM的**tabular data-specific fine-tuning-free methods**。
- 在更大、更多样化的多中心队列中进行**external validation**。
- 评估**computational efficiency**、成本效益与临床整合路径。
- 推进**prospective trials**验证真实世界有效性与安全性。

---

> 💡 **总结一句话**：  
> **LLM4CKD证明了大型语言模型在极低标注成本下可用于早期CKD筛查，是传统ML方法的有力补充，尤其在资源匮乏环境中具有应用前景，但其性能高度依赖模型选择与提示设计，且需进一步外部验证才能走向临床落地。**

</details>

---

### 13. [High-Dimensional Learning Dynamics of Attention-Indexed Models](https://arxiv.org/abs/2609.03858)

**Authors**: Yizhou Xu, Margarita Sagitova, Lenka Zdeborov\'a, Florent Krzakala  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.03858v1  

#### Abstract
Attention mechanisms are central to modern foundation models, yet their training dynamics remain poorly understood, especially when the attention matrices have extensive rank. In this work, we study attention-indexed models, a broad framework that can represent multi-layer and multi-head attention a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：High-Dimensional Learning Dynamics of Attention-Indexed Models

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在解决**现代基础模型中注意力机制（attention）在高维特征学习场景下的训练动态（training dynamics）理解不足**的问题。尽管注意力机制是当前 Transformer 架构的核心，但其在高维、可学习权重矩阵（即 `extensive-rank` 注意力矩阵，其秩随嵌入维度 `d` 增长）情况下的优化过程仍缺乏理论解释。

现有研究大多局限于静态分析（如贝叶斯最优）、低秩注意力（`finite-rank`）或不考虑高维极限的简化模型，无法捕捉实践中广泛存在的高秩注意力行为。

### 提出的新方法与新思路
作者提出了一个名为 **Attention-Indexed Models** 的通用框架，并在此框架下建立了首个针对 `extensive-rank` 注意力矩阵的**高维动力学理论**。其核心创新点如下：

1.  **宏观动力学描述**：
    *   **静态层面**：证明在 `d → ∞` 的高维极限下，种群损失（population loss）景观（landscape）仅由一组**有限的迹序参量（trace order parameters）** 决定，这提供了一个有限维的“势能”函数。
    *   **动态层面**：揭示了在线随机梯度下降（online SGD）的动力学远比损失景观复杂。它生成了一个**无限层次的矩阵矩（infinite hierarchy of matrix moments）**，这些矩共同决定了参数的演化轨迹。

2.  **可计算的近似理论**：
    *   尽管动力学是无限维的，但作者证明这个无限层次可以被**指数级精确地近似**。通过截断到有限阶数 `M` 的矩系统，其误差以 `exp(-M)` 的速度衰减。这使得该理论在数值上是可处理的。

3.  **注意力参数化的隐式偏置（Implicit Bias）**：
    *   这是本文最核心的发现。作者比较了三种表示有效注意力矩阵 `S` 的方式：
        *   **直接优化 `S`** (`S ∈ ℝ^{d×d}`)。
        *   **绑定因子化（Tied Factorization）** `S = WWᵀ`。
        *   **非绑定因子化（Untied Factorization）** `S = UVᵀ`。
    *   分析表明，不同的参数化方式会引入**本质不同的优化几何**，从而导致完全不同的学习行为，尤其是在从无信息初始状态开始学习时。

### 相比现有方法的优势
*   **适用范围更广**：首次将高维动力学分析扩展到 `extensive-rank` 注意力，填补了理论空白。
*   **理论深度更深**：明确区分了“静态损失景观”和“动态演化”的维度差异，揭示了无限维动力学的存在。
*   **洞察更具指导性**：将“参数化”本身视为一种架构上的隐式偏置，为模型设计提供了新的理论视角，而不仅仅是寻找更好的优化算法。

---

## 2. 核心实验方法和设置

本文主要基于**理论推导和数值模拟**来验证其结论，而非在真实世界的大规模数据集上进行实验。

### 数据集
*   理论分析基于**高斯数据假设**。输入数据向量 `x₁, ..., x_L ∈ ℝ^d` 被建模为联合高斯分布 `N(0, C ⊗ I_d)`，其中 `C` 是一个 `L×L` 的协方差矩阵，控制序列内 token 之间的相关性。
*   这是一种标准的理论分析设定，用于解耦数据分布的影响，专注于模型本身的动力学。

### 实验设置和评估指标
*   **任务**：采用教师-学生（teacher-student）设置。一个固定的“教师”网络生成标签，一个“学生”网络通过 SGD 学习去拟合它。
*   **模型**：
    *   **多层多头注意力网络**：作为 `attention-indexed model` 的一个具体实例（见附录 B.1）。
    *   **简化模型**：为了清晰展示核心机制，主图中使用了更简单的单层或双层模型。
*   **优化**：使用 **online SGD**，每次迭代使用一个新鲜采样的数据批次。
*   **评估指标**：
    *   **结构重叠（Structural Overlap）** `p₁₂`：这是衡量“弱恢复”（weak recovery）的关键指标。它定义为学生和教师注意力矩阵的“无迹部分”（traceless parts）之间的归一化迹，用以排除各向同性（isotropic）分量的干扰，真正反映特征对齐的程度。
    *   **样本复杂度（Sample Complexity）**：达到某个阈值 `c` 的 `p₁₂` 所需的训练步数 `N(c)`，用于量化学习速度。

### 基线方法对比
*   本文的“基线”是不同参数化方式下的自身对比：
    1.  **`W`-flow**：优化 `S = WWᵀ`（绑定因子化）。
    2.  **`S`-flow**：直接优化 `S`。
    3.  **`U, V`-flow**：优化 `S = UVᵀ`（非绑定因子化）。
*   通过比较这三种方式在相同教师-学生设置下的学习曲线，来凸显参数化带来的差异。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **绑定因子化 `S = WWᵀ` 的优势**：
    *   **自动对称性破缺（Automatic Symmetry Breaking）**：由于 `WWᵀ` 天然是半正定（PSD），其初始的各向同性分量 `t₁(0) > 0`。这使得即使在无信息初始状态下，梯度流也有一个非零的速度朝向信息丰富的方向演化。
    *   **弱恢复样本复杂度**：在满足温和条件的情况下，`W`-flow 能在 `O(d² log d)` 的样本复杂度内实现弱恢复（见 **Corollary 3** 和 **图2**）。而直接优化 `S` 在某些情况下可能永远停留在无信息流形上。

2.  **非绑定因子化 `S = UVᵀ` 的快慢机制**：
    *   **两阶段学习**：训练分为两个时间尺度。
        *   **快阶段（Fast Phase）**：预激活均值 `mₖ = Tr(UₖVₖ)/d` 快速演化，而矩阵矩保持冻结。
        *   **慢阶段（Slow Phase）**：均值被“锁定”在种群损失的临界流形上，随后矩阵矩才开始缓慢演化。
    *   **弱恢复的二分法（Dichotomy）**：弱恢复能否发生取决于**快阶段所选择的状态**。
        *   如果快阶段的均值移动打破了相关的对称性，则慢阶段可以逃离无信息状态，在 `O(d² log d)` 样本内实现弱恢复。
        *   如果对称性未被打破，则慢阶段动力学停滞，无法在 `O(d² log d)` 样本内实现弱恢复（见 **Corollary 4** 和 **图3**）。

3.  **与直接优化 `S` 的对比**：
    *   **图2** 清晰展示了这一对比。对于线性激活函数，所有方法都能学习。但对于更高阶的激活函数（如 `h₂(x)=x²-1`, `h₃(x)=x³-3x`），直接优化 `S` 的 `S`-flow 几乎停滞，而 `W`-flow 却能成功学习并发展出显著的重叠。这直接证明了 `S = WWᵀ` 参数化提供的隐式偏置的有效性。

### 消融实验结果
*   本文没有传统意义上的消融实验，但其核心理论本身就是一种深刻的“思想实验”消融。
*   **图1** 展示了对无限矩层次进行**截断阶数 `M` 的消融**。随着 `M` 从 2 增加到 4 再到 6，理论预测的 ODE 曲线与在线 SGD 的经验轨迹的吻合度越来越高，验证了“高阶矩可被指数级近似”的结论。

---

## 4. 关键结论和发现

### 主要发现
1.  **高维注意力需要新的动力学描述**：`extensive-rank` 注意力的训练动力学本质上是**无限维**的，由一个无限矩层次驱动，这与有限维的损失景观形成鲜明对比。
2.  **参数化即偏置**：注意力矩阵的参数化方式（`S`, `WWᵀ`, `UVᵀ`）不是等价的坐标变换，而是引入了**根本性的隐式偏置**，决定了模型如何逃离局部极小值或无信息状态。
3.  **绑定因子化的内在优势**：`S = WWᵀ` 的 PSD 结构提供了一种**自动的对称性破缺机制**，使其在 `O(d² log d)` 样本内就能实现弱恢复。
4.  **非绑定因子化的决策机制**：`S = UVᵀ` 的学习遵循**快慢分离**原则，最终的学习成败取决于快阶段所选择的初始状态是否破坏了对称性。

### 方法的局限性
1.  **理想化假设**：理论建立在高斯数据、无限宽度极限和特定的正则化条件下，与现实世界的复杂数据和有限宽度模型存在差距。
2.  **关注特定参数化**：分析集中在 `S`, `WWᵀ`, `UVᵀ` 三种形式，而实际 Transformer 中的 `QKV` 参数化更为复杂，其完整动力学尚未被推导。
3.  **长期学习分析缺失**：理论主要刻画了 `O(d² log d)` 样本内的“弱恢复”，对于更长训练周期的“强恢复”（strong recovery）和最终泛化性能的分析较少。
4.  **依赖教师-学生设定**：核心结论在教师-学生框架下得出，其普适性有待进一步验证。

### 未来工作方向
1.  **扩展到完整的 `QKV` 动力学**：将该框架推广到分析查询（Query）、键（Key）、值（Value）矩阵的完整更新规则。
2.  **分析更长的训练时间尺度**：研究 `O(d² log d)` 之后的学习阶段，特别是与“信息指数”（information exponent）相关的更长样本复杂度。
3.  **探索其他优化方法**：将理论扩展到批量梯度下降（batch GD）、Adam 等更复杂的优化器。
4.  **连接谱分布**：寻求用 resolvent 等工具将矩层次与注意力矩阵的谱分布（spectral distribution）直接联系起来。
5.  **应用于模型设计**：利用“参数化即偏置”的洞见，设计具有更好学习特性的新型注意力架构。

</details>

---

### 14. [RL-ADA: A World-Feedback Framework for Adversarially Robust Enterprise Dialogue Agents](https://arxiv.org/abs/2609.02902)

**Authors**: Ram Narayanan, Harshit Rajgarhia, Abhishek Mukherji  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.02902v1  

#### Abstract
Deploying task-oriented dialogue agents in enterprise customer support faces a persistent annotation bottleneck: robust training requires labelled interaction data at scale, yet enterprise conversational logs are privacy-sensitive and expensive to annotate, while user behaviour evolves faster than l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*RL-ADA: A World-Feedback Framework for Adversarially Robust Enterprise Dialogue Agents*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
企业级任务导向型对话代理（Enterprise Dialogue Agents）在客户支持场景中面临**标注瓶颈**：
- 企业对话日志具有隐私敏感性、领域专有性；
- 用户行为演化速度快于人工标注流程；
- 现有方法依赖大规模标注数据进行训练，难以持续适应真实用户的对抗性和动态变化。

传统方法如 **RLHF**（Reinforcement Learning from Human Feedback）成本高且不可扩展；**self-play** 方法假设对称智能体，不适用于客服系统中“支持代理”与“客户”的非对称角色。

---

### 🚀 提出的新方法与创新思路

提出 **RL-ADA**（Reinforcement Learning with Adversarial Dialogue Agents），一种基于**世界反馈**（world feedback）的共进化训练框架，核心思想如下：

#### （1）**用世界反馈替代人类标注**
- 不依赖任何人工标注数据；
- 使用从交互结果中提取的**可测量后果信号**作为奖励（reward），例如是否成功解决任务、工具调用是否正确等；
- 引入一个固定的自动化裁判（**Automated Judge**）来评估对话质量并提供终端评分。

#### （2）**非对称共进化架构**
- 部署两个角色不同的语言模型：
  - **Customer Support Agent (DA)**：3B 参数，负责澄清问题、路由工具、结束通话；
  - **Adversarial Customer Agent (CA)**：7B 参数，主动生成具有迷惑性的自然语言输入以诱导 DA 出错。
- 双方通过对抗性训练共同演化，形成“红队-蓝队”机制。

#### （3）**隔离健身房（Isolation Gym）机制**
- 落后的智能体进入“隔离健身房”，在其对手冻结权重的情况下，基于失败/成功对话记录（70:30 比例）进行重训练；
- 实现**无监督的后部署数据飞轮**（post-deployment data flywheel），无需人工干预即可持续优化。

#### （4）**宏观停止准则（Macro Stopping Criterion）**
- 基于滚动胜率稳定性定义收敛条件：
  - **能力门限**（Competency Gate）：DA 胜率 ≥ 60%
  - **稳定门限**（Stability Gate）：连续两轮胜率波动 ≤ 5%
- 作为经验性的 *s-Nash 收敛* 替代指标，避免无限训练。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | RL-ADA 的改进 |
|------|--------|----------------|
| **RLHF** | 依赖昂贵的人类偏好标注，难以规模化 | 完全去除人类标注，仅依赖自动化的 outcome-based reward |
| **Self-play** | 对称训练导致非平稳环境，易出现灾难性遗忘 | 冻结一方训练另一方，避免循环震荡 |
| **Red Teaming** | 多为静态攻击目标，无法随防御方同步进化 | CA 与 DA 共同演化，保持攻击有效性 |
| **Intent Classification** | 依赖显式意图标签 | DA 不做 intent 分类，直接学习从文本到 tool routing 的映射 |

---

## 2. 核心实验方法和设置

### 📚 数据集与领域设定
- **领域**：银行客户支持（banking customer support）
- **意图体系**：基于 **Banking77** 数据集扩展，共 **78 种客户意图**，映射至 **6 个 API 工具调用**（见附录 A）
  - 如 `unrecognized_charge` → `dispute_charge`
  - `check_balance` → `lookup_account`
- 所有意图均不能直接命名，需通过上下文推断（如“I see something odd on my statement”可能对应多种意图）

> ⚠️ 注意：所有训练过程**未使用任何带标签的真实对话数据**，仅利用合成或模拟环境中的 outcome 信号。

---

### ⚙️ 实验设置

#### 模型配置
| 角色 | 模型 | 参数量 | 微调方式 |
|------|------|--------|----------|
| DA（Support Agent） | Qwen2.5 | 3B | LoRA + GRPO |
| CA（Adversarial Customer） | Qwen2.5 | 7B | LoRA + GRPO，无 SFT 初始化 |
| Judge（裁判） | Qwen2.5 | 7B | 固定权重，不更新 |

#### 训练三阶段循环（Three-phase Loop）
1. **Bootstrap Phase**  
   - DA 先在 Banking77 上做 SFT，再用 Judge 奖励微调；
   - CA 从零开始，在固定 DA 下通过 reward 学习误导策略。

2. **Adversarial Arena**  
   - 当前 DA 与 CA 进行多轮对话比赛（每场比赛 180 场景 × 2 次运行）；
   - 统计 DA 胜率 $W_{DA}$，判断谁更弱。

3. **Isolation Gym**  
   - 较弱的一方进入隔离训练，使用最近的 **70% 失败 + 30% 成功** 对话记录构建训练集；
   - 使用滑动窗口逐轮构造 GRPO 样本，提升鲁棒性。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Tool-routing accuracy** | 正确调用目标工具的比例 |
| **PASS rate (strict)** | 同时满足：<br>- 正确 tool routing<br>- `lookup_account` 首先调用<br>- episode reward ≥ 2.0<br>- 干净结束对话 |
| **FAIL rate** | 不符合 PASS 条件的比例 |
| **Avg episode reward** | 单次对话平均得分（含格式、流程、结果等） |
| **Lookup-first rate** | 是否遵守先验证身份的流程规范 |

#### 基线方法对比
- **DA₀**：初始模型，经过 SFT + 初始 GRPO 训练，未参与共进化；
- **DA₂**：经历五轮共进化后的最终模型；
- 两者在同一组 **12 个固定 hold-out 场景** 上测试，比较性能差异。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 4）

| 指标 | DA₀（基线） | DA₂（RL-ADA 后） | 提升 |
|------|------------|------------------|-------|
| **Tool-routing accuracy** | 75% | **100%** | +25pp |
| **Strict PASS rate** | 25% | **50%** | +25pp |
| **FAIL rate** | 33% | 33% | 0（结构变化） |
| **Avg episode reward** | +1.58 | **+2.16** | +0.58 |
| **Lookup-first rate** | 58% | **83%** | +25pp |

> ✅ **所有工具路由错误被完全消除**，PASS 率翻倍，且全部由自动化 reward 驱动，**无新增标注数据**。

---

### 🔄 共进化动态分析（Table 3 & Figure 2）

- 在五轮 Arena 匹配中，DA 胜率呈现**非单调波动**：
  - 第3轮（CA₁ vs DA₁）：$W_{DA} = 0.56$ ↓（CA 经过 Gym 训练后增强）
  - 第4轮（DA₂ vs CA₁）：$W_{DA} = 0.64$ ↑（DA 反击成功）
  - 第5轮（DA₂ vs CA₂）：$W_{DA} = 0.62$，趋于稳定
- 表明存在真实的**对抗性共进化压力**，而非独立提升。

> ✅ 停止准则在第5轮触发（连续两轮满足能力与稳定性门限），提前终止训练（最大允许8轮），说明该机制有效。

---

### 🔍 消融实验与关键观察（虽无正式 ablation study，但有深入分析）

#### （1）失败模式转移
- DA₀ 的失败主要源于 **routing error**（如将 dispute 请求转给人类）；
- DA₂ 能正确 routing，但仍有约 33% FAIL，原因转为：
  - 流程违规（未先调用 `lookup_account`）
  - 对话质量不足（judge 打分低）
- ⇒ 表明 **routing 问题已解决**，后续可通过调整 reward 权重进一步优化流程合规性。

#### （2）奖励函数设计影响
- 使用密集 shaping reward（如 `r_format`, `r_env`）防止早期 collapse；
- 最终策略提升由终端 judge score（world feedback）驱动。

> ❗作者指出 reward component 权重选择为启发式设定，未来需系统性敏感性分析。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **完全去除人类标注是可行的**  
   - RL-ADA 证明仅靠 **automated world feedback** 就能实现高质量对话代理训练；
   - 自动裁判（NeutralJudge）在 resolution detection 和 hallucination detection 上表现接近 GPT-4o-mini（F1=0.807 vs 0.821，**幻觉检测 F1=1.0**）。

2. **共进化带来真实鲁棒性提升**  
   - DA 在面对不断进化的 CA 攻击下仍能恢复并超越；
   - 非单调胜率轨迹表明系统经历了真正的“攻防博弈”。

3. **涌现出高级对抗策略：Contextual Camouflage**  
   - CA 学会将真实意图隐藏在大量真实细节中（如具体商户名、金额、时间），而非简单模糊表达；
   - 示例：“It seems like I might have accidentally ordered two drinks…” 实际意图为 `dispute_duplicate_charge`；
   - 这种行为**从未被明确编程或标注指导**，纯由 reward pressure 激发。

4. **支持企业级数据飞轮闭环潜力**  
   - Isolation Gym 构成了训练侧的数据飞轮机制；
   - 若能在生产环境中接入真实 outcome 信号（如 resolution marker、callback rate），即可实现全自动迭代优化。

---

### ⚠️ 局限性

| 局限 | 说明 |
|------|------|
| **停止准则通用性有限** | 目前阈值（60%, 5%）为经验设置，尚未推广到多域配置化框架 |
| **评估范围窄** | 仅在一个 banking domain 验证，缺乏跨领域泛化证据 |
| **缺少消融研究** | reward 组件、70:30 混合比例、hyperparameters 影响未知 |
| **未闭环部署验证** | 数据飞轮仍在 arena 内演示，尚未连接真实用户 outcome |
| **角色漂移风险** | CA 偶尔模仿 agent 口吻（role reversal），需加强 system prompt 控制 |

---

### 🔮 未来工作方向

1. **构建可配置的部署门控系统**（deployment gating）  
   - 对关键场景（如 fraud detection）设置更高通过门槛（如 ≥90% 胜率）；
   - 与训练停止准则解耦，保障合规性。

2. **跨领域迁移验证**  
   - 将 RL-ADA 应用于保险、电信等领域，检验 world feedback 框架普适性。

3. **建立真实 outcome 映射机制**  
   - 将生产环境中的 callback rate、escalation trace 等 telemetry 映射为 judge 替代信号，实现端到端 flywheel。

4. **量化 emergent behavior**  
   - 使用 named-entity density 或 token length 度量 “Contextual Camouflage” 的强度演进。

5. **引入 role-consistency reward**  
   - 防止 CA 模仿客服角色，确保对抗合理性。

---

## 总结

> **RL-ADA 是首个完全去人类标注、基于 world feedback 的企业级对话代理共进化训练框架**。它通过 DA 与 CA 的非对称对抗训练，结合自动化裁判与隔离健身房机制，在银行客服场景中实现了：
>
> - ✅ **100% 工具路由准确率**
> - ✅ **PASS 率翻倍至 50%**
> - ✅ **涌现高级对抗策略 Contextual Camouflage**
> - ✅ **无需人工标注的数据飞轮雏形**
>
> 尽管当前验证局限于单一领域，其组件设计（adversarial arena, isolation gym, world feedback）均为**领域无关**，具备向其他企业服务场景迁移的巨大潜力。

</details>

---

### 15. [SGD-KV: Summarization Guided KV Cache Compression](https://arxiv.org/abs/2609.03235)

**Authors**: Zeyu Liu, Woomin Song, Xuandi Fu, Sai Muralidhar Jayanthi, Vivek Govindan, Aram Galstyan, Sravan Babu Bodapati, Srikanth Ronanki  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.03235v1  

#### Abstract
Large language models (LLMs) face severe memory bottlenecks in long-context inference due to the linearly growing size of key-value (KV) caches. Existing KV cache compression techniques typically rely on simple heuristics, overlooking the distinct functional roles of different attention heads. We pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SGD-KV: Summarization Guided KV Cache Compression 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在长上下文推理时面临严重的**内存瓶颈**，其根源在于 Key-Value (KV) cache 的大小随上下文长度线性增长。现有的 KV cache 压缩方法通常依赖简单的启发式策略（如基于注意力分数或最近性的 token 蒸发），忽略了不同 **attention head** 在功能上的差异性。

尤其在多文档分析、长对话等需要**层次化信息聚合**（hierarchical information aggregation）的任务中，传统检索导向的方法表现不佳。

---

### 🚀 提出的新方法与创新思路
作者提出 **SGD-KV**（Summarization-Guided KV Cache Compression），一种**head-aware**的 KV cache 压缩框架，核心思想是：

- 引入“**summarization heads**”这一新型功能类别：识别出专门负责从长文本中提取和综合语义摘要的 attention heads。
- 设计一个新颖的诊断任务——**chunk-summarization task**：
  - 将多个短文本拼接成长上下文输入；
  - 要求模型划分语义块并为每个块生成关键词；
  - 利用该任务量化各 attention head 对信息聚合的能力。
- 基于每个 head 的“**summarization score**”，采用类似 water-filling 的算法动态分配 KV cache 预算，优先保障关键 summarization heads 的缓存容量。

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | SGD-KV 的改进 |
|------|--------|----------------|
| StreamingLLM / H2O | 仅基于 token 级别重要性（如 recency）进行删除，缺乏功能感知 | 引入 head-level 功能角色理解 |
| AdaKV / PyramidKV | 使用 attention 分布统计分配预算，仍以检索为中心 | 明确区分“信息检索”与“信息综合”功能 |
| HeadKV / DuoAttention | 关注 retrieval 或 reasoning heads，忽视更高阶的认知能力 | 发现并利用 **summarization heads** 进行更精细的资源调度 |

> ✅ **优势总结**：通过功能导向的设计，实现更优的效率-准确性权衡，在百万级 token 上下文中显著减少内存占用而不牺牲性能。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **训练/诊断数据**（用于计算 summarization score）：
  - CNN/DailyMail
  - DialogSum
  - SAMSum
  - XSum
  - Databricks Dolly
  - WikiLingua  
  （共6个摘要类数据集，确保泛化性）
- **评估基准**：
  - **MRCR**（Multi-Round Co-Reference Resolution）：测试长对话中的指代消解能力
  - **ETHIC**：评估高信息覆盖率任务下的表现（Attribution, Organization, Recall 子任务）
  - **BABILong**：推理-in-a-haystack 类型任务，检验复杂推理与检索能力

---

### ⚙️ 实验设置与评估指标
- **模型**：
  - `Qwen2.5-7B-Instruct-1M`（经 fine-tuning 后使用）
  - `Qwen3-32B`
- **上下文长度范围**：从 8K 到 **1M tokens**
- **KV Cache Budget**：压缩至原始的 25% 左右（部分实验低至 10%，高达 75%）
- **评估指标**：
  - 准确率（Accuracy）
  - 平均得分（Avg. Score on ETHIC）
  - 内存节省比例（up to 75% reduction）

---

### 🆚 基线方法对比
| 基线方法 | 类型 |
|---------|------|
| FullKV | 不压缩，完整 KV cache |
| Minference | 动态稀疏 attention，token-level |
| AdaKV | head-level 自适应预算分配 |
| DuoAttention | 将 heads 二分为 retrieval vs. streaming |
| HeadKV | 基于 R2 score 的 head-level 分配 |

所有方法统一保留 `sink tokens` 和 `recent window tokens`，公平比较中间部分的压缩效果。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Tables 1–2）

#### 在 MRCR 上的表现（Qwen2.5-7B-1M, 25% KV budget）

| Method       | 64k   | 128k  | 256k  | 512k  | 1M    |
|--------------|-------|-------|-------|-------|-------|
| FullKV       | 95.01 | 96.38 | 88.6  | 63.84 | 43.29 |
| DuoAttention | 85.80 | 89.82 | 73.78 | 39.10 | 24.73 |
| HeadKV       | 68.52 | 74.81 | 66.98 | 44.36 | 28.90 |
| **SGD-KV (Ours)** | **85.09** | **87.19** | **83.29** | **48.86** | **34.16** |

✅ **结论**：SGD-KV 在超长上下文（>128K）下明显优于其他方法，尤其在 1M token 场景中领先第二名近 **10 pts**

---

#### 在 ETHIC 上的表现（平均得分）

| Method       | Qwen2.5-7B Avg. | Qwen3-32B Avg. |
|--------------|------------------|----------------|
| FullKV       | 21.65            | 28.53          |
| DuoAttention | 19.51            | 24.49          |
| HeadKV       | 20.87            | 28.19          |
| **SGD-KV (Ours)** | **21.34**        | **28.38**      |

✅ **亮点**：在仅有 25% KV cache 的情况下，SGD-KV 在 Qwen3-32B 上达到了接近 FullKV 的性能（28.38 vs 28.53），**几乎无损压缩**！

---

### 🔬 消融实验结果（Ablation Studies）

#### （1）Query-Aware vs. Query-Unaware Token Selection（Table 3）
| 条件 | 方法 | 性能趋势 |
|------|------|--------|
| Query-Aware（默认） | SGD-KV | 最佳性能 |
| Query-Unaware | SGD-KV | 显著下降 |
| **Proxy-Query**（使用通用摘要 prompt） | **SGD-KV** | **大幅缓解性能损失，接近 Query-Aware** |

> 💡 发现：提出的 **Summarization Prompt** 可作为真实查询的有效代理，适用于预压缩场景（query未知时）。

#### （2）不同 KV Cache 预算下的鲁棒性（Appendix A.6）
- SGD-KV 在 **15%-50%** 缓存预算范围内始终优于 AdaKV 和 HeadKV
- 当预算 >50%，开始超越 Minference（token-level 方法）
- 极端低预算（<15%）下优势缩小，但仍具竞争力

#### （3）head 配置变体实验（Appendix A.7）
- **SGD-KV (Reverse)**（反向分配预算）→ 性能暴跌 → 证明 score 排序有效
- **SGD-KV + HeadKV**（融合 R2 与 summarization score）→ 性能介于两者之间
- **SGD-KV (thr)**（只保留高分 heads）→ 短上下文提升，长上下文下降 → 表明低分 heads 在极长上下文中仍有作用
- **Ipt., Max**（结合 summarization score 进行 attention aggregation）→ 显著提升 GQA 模型表现

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **存在专用的 summarization heads**：某些 attention heads 在信息综合任务中表现出高度专业化，可通过 chunk-summarization 任务系统识别。
2. **功能感知压缩优于纯统计压缩**：基于功能角色（而非仅注意力强度）分配 KV cache 能带来更优的精度-效率平衡。
3. **SGD-KV 实现 SOTA 性能**：
   - 在长达 **1M tokens** 的上下文中保持高性能
   - KV cache 内存使用减少 **up to 75%**
   - 多项 benchmark 上达到或逼近 FullKV 表现
4. **通用摘要 prompt 是有效的 query proxy**：即使最终查询未知，也可用固定 prompt 指导 token selection，降低部署复杂度。

---

### ⚠️ 方法的局限性
- **依赖离线诊断过程**：需预先运行 chunk-summarization 任务获取 head scores，增加前期开销
- **fine-tuning 影响配置有效性**：实验发现微调后原有 head score 配置可能失效（Appendix A.2）
- **对 base model 能力有要求**：在能力较弱的模型上可能难以准确识别 summarization heads
- **目前主要验证于 Qwen 系列模型**，跨架构泛化性有待进一步验证

---

### 🔮 未来工作方向
1. **自动化 summarization head 发现流程**：设计无需人工标注的自监督方式识别此类 heads
2. **在线自适应调整机制**：根据输入内容动态更新 head importance scores
3. **扩展到其他高级认知功能**：如 planning heads、reflection heads 等
4. **集成进推理引擎**：将 SGD-KV 与 vLLM、TensorRT-LLM 等系统结合，推动工业级应用

---

## ✅ 总结一句话
> SGD-KV 首次提出“**summarization heads**”概念，并通过功能导向的 KV cache 分配策略，在百万 token 长上下文场景下实现了**最高达 75% 的内存压缩**，同时保持接近 FullKV 的性能，为高效、可解释的 LLM 推理提供了新范式。

</details>

---

### 16. [Iapetus: Content-Aware Hierarchical Scheduling for Collaborative ViT Inference in LEO Satellite Networks](https://arxiv.org/abs/2609.03318)

**Authors**: Yan Chen, Yunxiang Zhang, Guanjun Jiang, Haiquan Wang  
**Category**: cs.DC  
**Published**: 2026-09-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.03318v1  

#### Abstract
Collaborative inference pools distributed resources to run compute-intensive Vision Transformers (ViTs) in satellite edge computing. Model partitioning enables such collaboration by assigning consecutive layer groups to different nodes, but the large volume of intermediate activation data incurs sub...

---

### 17. [Equation Recast for Canonical Operator Learning Across Parametric PDEs](https://arxiv.org/abs/2609.02982)

**Authors**: Qiyun Cheng, Valentin Duruisseaux, Cesar F. Clauser, Md Hossain Sahadath, Huihua Yang, Shaowu Pan, Nathaniel Ferraro, Anima Anandkumar, Wei Ji, Cristina Rea  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.02982v1  

#### Abstract
Learning solution operators across broad parameter ranges can require substantial coverage of both input functions and physical parameters, particularly for purely data-driven parametric models. In addition, the resulting models may fail silently outside the training distribution. We introduce equat...

---

### 18. [Mesh-Native Physics-Informed Graph Surrogates for TCAD-in-the-Loop Design Space Exploration](https://arxiv.org/abs/2609.02988)

**Authors**: Leonid Popryho, Ayoub Sadeghi, Inna Partin-Vaisband  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.02988v1  

#### Abstract
High-fidelity TCAD simulation of drift-diffusion transport remains the workhorse of emerging FinFET device design, but it is computationally expensive, especially for 3D structures where runtime escalates steeply with mesh complexity. This sharply limits multi-objective design space exploration. Exi...

---

### 19. [What Matters for Aggressive Decoding-Time KV Eviction? Temporal Aggregation and Ranking Preservation](https://arxiv.org/abs/2609.03515)

**Authors**: Bo Zeng, Yu Zhao, Yefeng Liu, Zhihong Lu, Xuanfan Ni, Xintong Wang  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.03515v1  

#### Abstract
Decoding-time KV cache compression research focuses heavily on designing better token scoring functions, while the temporal rule that aggregates scores across decode steps is often treated as an implementation detail. Under aggressive KV compression, we find that exponential-moving-average (EMA) agg...

---

### 20. [R$^{2}$Adapter: A Routing and Rewriting Adapter for Efficient Hybrid RAG](https://arxiv.org/abs/2609.02894)

**Authors**: Yucan Guo, Miao Su, Saiping Guan, Long Bai, Zhongni Hou, Zixuan Li, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.02894v1  

#### Abstract
Retrieval-Augmented Generation (RAG) has become a prevailing paradigm for enhancing Large Language Models (LLMs) with non-parametric knowledge. Vanilla RAG efficiently handles simple queries but struggles with relational or multi-hop reasoning. Graph-based RAG alleviates this issue but incurs higher...

---

### 21. [Less Is Moral: A CHARMing Framework for Moral Foundations Detection in Endorsement Behaviour](https://arxiv.org/abs/2609.03330)

**Authors**: Huixiang Fu, Marian-Andrei Rizoiu  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.03330v1  

#### Abstract
Moral language plays a central role in shaping online endorsement and the diffusion of information, yet existing moral foundation detection systems often suffer from poor cross-domain generalization, weak rationale grounding, and reliance on costly prompting-based large language models (LLMs). We in...

---

### 22. [TRACE: Spatiotemporal Contact Memory Graph Network Simulator for Granular Dynamics](https://arxiv.org/abs/2609.02991)

**Authors**: Changjian Zhou, Negin Yousefpour, Jie Qi, Junfeng Fang, Guillermo A. Narsilio, Hans Petter Jostad  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.02991v1  

#### Abstract
Learned graph simulators provide an efficient alternative to high-fidelity solvers for granular dynamics. However, granular motion depends strongly on inter-granular contact history, which is difficult to preserve when particle contacts form, break, and rearrange. Existing simulators mainly store te...

---

### 23. [Coupled Scaling: A Representational Accessibility Framework for Neural Scaling Laws](https://arxiv.org/abs/2609.03533)

**Authors**: Jie Wang  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.03533v1  

#### Abstract
Existing theories derive neural scaling from data geometry or a specified data-model spectrum, but systems trained on the same data can scale differently when architecture or optimization changes the representations they can efficiently reach. We introduce Coupled Scaling, a task-conditioned framewo...

---

### 24. [A Peer-Relative Representation Learning Framework for Energy Inefficiency Identification in Mobile Network Sites](https://arxiv.org/abs/2609.03809)

**Authors**: Eliud Nyakweba Koto, Jaco du Toit, Adham Stoltz, Johan du Preez  
**Category**: cs.LG  
**Published**: 2026-09-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.03809v1  

#### Abstract
Energy consumption is one of the largest operational expenditure items for mobile network operators, yet site-level energy inefficiencies such as faulty cooling controllers, idle radio equipment, and parasitic auxiliary loads often remain undetected because no ground-truth inefficiency labels exist ...

---

### 25. [PPO-STGNN: A Proximal Policy Optimization Approach with Spatio-Temporal Graph Neural Networks for DAG Task Scheduling in Cloud-Edge-End Computing](https://arxiv.org/abs/2609.03503)

**Authors**: Yangshuo Qi, Chenwei Wang, Zihan Shen, Songlin Sun  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.03503v1  

#### Abstract
With the rapid development of the Internet of Things, computation intensive directed acyclic graph (DAG) tasks have become increasingly common in cloud-edge-end collaborative environments. However, cloud, edge, and end nodes are highly heterogeneous in computing capacity, network bandwidth, and ener...

---

### 26. [NeoRed: A Knowledge-Logic-Alignment Multimodal Large Language Model for Neonatal Respiratory Disease Diagnosis](https://arxiv.org/abs/2609.03527)

**Authors**: Yinan Liu, Hongtai Xia, Haoran Xu, Jiankang Hong, Jingkuan Song, Ye Luo  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.03527v1  

#### Abstract
Neonatal respiratory diseases are a major cause of neonatal morbidity and mortality, posing substantial challenges in clinical practice. Despite recent advances, existing Multimodal Large Language Models (MLLMs) face two key limitations in neonatal diagnosis: (1) domain gap arising from predominantl...

---

### 27. [Feature Reconfiguration With Visual Prior for Medical Lesion Segmentation](https://arxiv.org/abs/2609.03535)

**Authors**: Yinan Liu, Jiankang Hong, Zhen Gao, Ye Lu  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.03535v1  

#### Abstract
Lesion segmentation in medical images plays a critical role in clinical diagnosis and treatment planning. Despite significant advances, lesion segmentation remains challenging due to two major factors: (1) complex background interference; (2) diverse lesion morphology. Existing encoder-decoder based...

---

### 28. [Synthetic Semantic Supervision for Contrastive Code Representation Learning in Small Transformers: An Empirical Study](https://arxiv.org/abs/2609.03702)

**Authors**: Kenneth Paulsen, Florian Tambon, Mike Papadakis, Shin Yoo  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.03702v1  

#### Abstract
General-purpose code embeddings power tools for code search, classification, and retrieval. Compact transformer encoders for code typically rely on either human-written docstrings (labor-intensive and inconsistent) or mined structural signals such as execution traces (setting-specific and costly to ...

---

### 29. [SVG-Score: Human-Aligned Evaluation of Text-to-SVG Generation](https://arxiv.org/abs/2609.03806)

**Authors**: Marco Cipriano, Leonardo Zini, Alexandra Schild, Valentin Teutschbein, Afsana Mimi, Marcella Cornia, Lorenzo Baraldi, Gerard de Melo  
**Category**: cs.AI  
**Published**: 2026-09-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.03806v1  

#### Abstract
Scalable Vector Graphics (SVG) generation is attracting increasing attention as generative models improve in expressiveness and controllability. Progress, however, is held back by the lack of domain-specific evaluation protocols: current practice relies on metrics designed for natural images, most n...

---

### 30. [Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning](https://arxiv.org/abs/2609.03430)

**Authors**: Heng Wang, Jielin Qiu, Wenting Zhao, Cheng Qian, Liangwei Yang, Jiawei Han, Heng Ji, Silvio Savarese, Shelby Heinecke, Huan Wang  
**Category**: cs.CL  
**Published**: 2026-09-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.03430v1  

#### Abstract
Large language models achieve superior performance on tasks that require extended reasoning, but long chains of thought make the KV cache a severe memory bottleneck. Existing KV cache compression methods share one paradigm: score each cached token by some estimate of how much it will matter later, a...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-05 12:39:55 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration](https://arxiv.org/abs/2610.02800)

**Authors**: Chence Yang, Ningxi Cheng, Arash Akbari, Qitao Tan, Qingchan Zhu, Ci Zhang, Changdi Yang, Yanzhi Wang, Wei Niu, Jinhui Wang, Jin Lu, Geng Yuan  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2610.02800v1  

#### Abstract
Speculative decoding accelerates autoregressive generation by using a lightweight draft to propose multiple tokens for parallel verification. However, existing methods often require an additional draft model or weight representation, introducing non-negligible memory overhead on resource-constrained...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration**

---

## **1. 主要贡献和创新点**

### **解决的问题**
现有的 **Speculative Decoding** 方法通过使用轻量级的 **draft model** 来并行生成多个候选 token，并由目标模型（target model）进行验证，从而加速自回归推理。然而，这些方法通常需要额外的 draft 模型或低精度权重表示，导致在资源受限设备上产生显著的 **memory overhead**。

此外，**self-speculative decoding** 虽然避免了独立的 draft 模型，但仍面临以下权衡：
- draft 质量 vs. target 质量
- 推理效率 vs. 存储开销
- 低精度 draft 与高精度 target 的一致性

### **提出的新方法**
本文提出了 **BitNest**，一种全新的 **bit-nested speculative decoding** 框架，其核心思想是：
> 将一个低精度的 draft 模型直接嵌入到高精度 target 模型的权重表示中，实现物理上的“位嵌套”（bit-nesting），而非维护两个独立的模型。

#### **关键技术思路**
- **Base-First Construction**：先构建一个高质量的 **W4 base**（作为 draft），然后在其基础上添加 **4-bit refinement** 来恢复为 **W8 target**。
- **Dual-Plane Weight Storage**：将 base 和 refinement 分别存储为两个独立可访问的 bit plane，draft 阶段只读取 base，verification 阶段读取两者。
- **Nested KV Cache**：将相同的设计扩展到 KV cache，在长上下文场景下进一步减少内存带宽压力。

### **相比现有方法的优势**
| 方面 | BitNest | 其他方法（如 QSpec, QuantSpec） |
|------|--------|-------------------------------|
| **Memory Overhead** | 无额外 draft 存储，共享同一物理表示 | 需要额外存储 draft 权重或 KV 表示 |
| **Target Quality** | 接近 W8A8 性能 | QSpec 受限于 W4 target，质量下降明显 |
| **Draft-Target Agreement** | 平均接受率高达 **95.2%** | 多数方法低于 90% |
| **端到端加速** | **1.48–1.61×** 超过 FP16 autoregressive | 多数 <1.3× |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖六类典型任务：
- **语言建模**：WikiText-2
- **数学推理**：GSM8K
- **代码生成**：HumanEval + MBPP（统称 Code）
- **对话生成**：ShareGPT
- **长文本理解**：LongDoc
- **长上下文语言建模**：PG-19

此外还测试了 **LongBench** 上的多任务长上下文理解能力。

### **实验设置**
- **模型**：
  - LLaMA-2-7B
  - LLaMA-3-8B
  - Qwen2-7B
  - Qwen2.5-7B
  - LLaMA-2-7B-32K（用于长上下文）
- **硬件平台**：
  - 主实验：NVIDIA RTX A6000
  - 边缘设备：Jetson Orin NX（16GB 和 8GB 版本）
- **Batch Size**：1
- **Context Length**：标准为 2K；长上下文实验从 4K 到 32K
- **Decoding Mode**：greedy decoding，生成 256 个新 token

### **评估指标**
- **End-to-end decoding speedup**（相对于 FP16 autoregressive baseline）
- **Perplexity / Accuracy**（衡量生成质量）
- **Speculative acceptance rate**
- **Peak GPU memory usage**
- **Energy consumption**（边缘设备）

### **基线方法对比**
| 方法 | Draft → Target | 特点 |
|------|----------------|------|
| **QSpec** | W4A4 → W4A16 | 共享 W4 权重，但 target 仍为 W4，质量受限 |
| **QuantSpec** | W4A16 → W16A16 | 使用低精度 draft + 高精度 target，需额外存储 |
| **Draft & Verify** | W16A16 → W16A16 | 层跳过机制，效率提升有限 |
| **FP16 AR** | — | 基准线 |

所有 speedup 均在同一推理引擎内测量，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **平均 speculative acceptance rate**：**95.2%**（跨 4 个模型、6 个任务）
- **端到端解码加速**：**1.48–1.61×** 超过 FP16 autoregressive decoding
- **边缘设备加速**：
  - 在 Jetson Orin NX 16GB 上对 LLaMA-2-7B-32K 实现约 **1.5×** 加速
  - 解码能耗从 **1.52 J/token** 降至 **0.95 J/token**

### **与基线方法的对比结果**
#### **表：端到端解码速度提升（部分）**
| Model | Method | Wiki | GSM8K | Code | ShareGPT | LongDoc | PG19 |
|-------|--------|------|--------|------|----------|---------|------|
| LLaMA-2-7B | QSpec | 1.35× | 1.43× | 1.46× | 1.46× | 1.44× | 1.40× |
| LLaMA-2-7B | QuantSpec | 1.17× | 1.19× | 1.23× | 1.22× | 1.17× | 1.12× |
| LLaMA-2-7B | **BitNest** | **1.50×** | **1.59×** | **1.59×** | **1.61×** | **1.60×** | **1.57×** |

> BitNest 在所有任务上均显著优于现有 self-speculative 方法。

#### **生成质量对比（GSM8K 准确率）**
| Model | FP16 | W8A8 | QSpec | **BitNest** |
|-------|------|------|--------|------------|
| LLaMA-3-8B | 50.1% | 50.0% | 37.0% | **49.8%** |

> BitNest 几乎完全保留了 W8A8 的生成质量，而 QSpec 下降明显。

### **消融实验结果**
#### **(1) Speculation Length 影响**
- 最优 speculation length $ \gamma = 4 $
- 更大的 $ \gamma $ 导致 acceptance 下降，吞吐不再提升

#### **(2) KV Cache 精度影响**
- 在长上下文（32K）下，使用 **KV4 draft + KV8 verify** 比全 KV8 提升显著：
  - LLaMA-2-7B 吞吐从 28.5 → **34.3 tok/s**
- 对 GQA 架构（如 Qwen）提升较小，因其本身 KV traffic 已较低

#### **(3) Target Precision 必要性**
- 单独使用 W4A8 作为最终模型会导致严重性能退化（如 LLaMA-3-8B 从 50.0% → 41.4%）
- BitNest 通过 refinement 成功恢复至 49.8%，证明 refinement 的有效性

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **单一物理表示可同时支持 draft 和 target**：通过 bit-nesting 设计，实现了真正的 zero-overhead draft 存储。
2. ✅ **高 acceptance 与高质量兼顾**：95.2% 的平均接受率表明 W4 draft 与 W8 target 高度一致。
3. ✅ **显著降低内存占用**：
   - 权重仅占 **6.22 GiB**（等同于单个 W8 模型）
   - 相比独立存储 W4 draft + INT8 target（需 9.38 GiB）节省 33%
4. ✅ **边缘设备友好**：
   - 在 Jetson Orin NX 上成功运行 LLaMA-2-7B-32K
   - 最大可执行上下文从 FP16 的 2K 提升至 **10K**

### **方法的局限性**
- ❌ **不适用于非自回归 draft 架构**：当前设计聚焦于 self-speculative，未整合如 DFlash 等基于 diffusion 的外部 draft 模型。
- ❌ **依赖特定量化流程**：需先应用 rotation-based transformation（如 SpinQuant）以优化 outlier 分布。
- ❌ **prefill 阶段加速有限**：当前实现中 prefill 使用 dequantized weights，导致该阶段反而变慢。

### **未来工作方向**
- 🔮 扩展 BitNest 支持 **auxiliary drafters**（如 diffusion-based models），在保持 nested 表示优势的同时引入更强的并行性。
- 🔮 开发专用的 **large-M INT8 GEMM kernel**，以优化 prefill 阶段性能。
- 🔮 探索更多 bit 配置组合（如 W3/W6、W5/W10）以适应不同硬件约束。

---

> **总结一句话**：  
> **BitNest 通过 bit-nested 权重设计，在几乎零内存开销的前提下，实现了接近 W8 质量的高质量 draft，达成 1.5× 左右的端到端加速，是面向边缘设备的高效 LLM 推理的重要进展。**

</details>

---

### 2. [ByteSplat: Efficient Distributed 3D Gaussian Splatting Training via Intra- and Inter-GPU communication reduction](https://arxiv.org/abs/2610.02851)

**Authors**: Shuo Wu, He Zhu, Han Zhao, Xiaohui Zhang, Yaqian Zhao, Hui Wei, Ruyang Li, Hongzhi Shi, Lihua Lu, Jingwen Leng, Yu Feng, Minyi Guo  
**Category**: cs.DC  
**Published**: 2026-10-05  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.02851v1  

#### Abstract
3D Gaussian Splatting (3DGS) enables photorealistic scene reconstruction, but training large-scale scenes requires substantial memory and computation. Distributing training across multiple GPUs increases available memory capacity, yet its performance is strictly constrained by data movement. We iden...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《ByteSplat: Efficient Distributed 3D Gaussian Splatting Training via Intra- and Inter-GPU communication reduction》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
3D Gaussian Splatting (3DGS) 是一种实现高质量、实时渲染场景重建的前沿技术，但在训练大规模场景时面临显著的**内存和计算瓶颈**。分布式训练虽能扩展显存容量，但其性能受限于两大关键因素：
- **Intra-GPU 数据移动**：前向与反向光栅化过程中频繁的片外（off-chip）内存访问导致高延迟。
- **Inter-GPU 通信开销**：部分梯度在多GPU间传输时产生大量冗余通信，尤其是零值梯度。

这些数据移动成为分布式3DGS训练的主要性能瓶颈。

---

### **提出的新方法与创新思路**
为解决上述问题，论文提出了 **ByteSplat**，一个高效的分布式3DGS训练框架，通过联合优化 **Intra-GPU 和 Inter-GPU 数据移动** 来提升训练效率。其三大核心技术如下：

#### **(1) Fused Rasterization（融合光栅化）**
- 将 **forward rasterization、loss computation 和 backward rasterization** 融合为单个GPU内核（mega-kernel）。
- 中间结果（如Gaussian属性、像素颜色、梯度等）保留在片上存储（on-chip SRAM），避免重复读写片外DRAM。
- 显著减少中间数据的往返传输，降低内存带宽压力。

#### **(2) Hardware-Aware Pruning（硬件感知剪枝）**
- 针对融合光栅化带来的片上内存压力（shared memory压力），提出一种考虑目标GPU硬件资源限制的剪枝策略。
- 动态识别“过载”图像块（tile），并优先移除对渲染质量影响小但占用资源高的Gaussians。
- 使更多tile满足融合内核的共享内存预算，从而提高融合执行覆盖率。

#### **(3) Zero-Eliding Gradient Communication（零梯度剔除通信）**
- 利用反向传播中大量partial gradients为**零**的稀疏特性（高达72.0%）。
- 在通信前剔除所有全零梯度，并使用GPU友好的编码器压缩非零梯度记录。
- 接收端直接聚合稀疏梯度，跳过密集重构过程，大幅减少通信量和本地聚合开销。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | ByteSplat |
|------|--------|----------|
| 内存访问 | 分离内核 → 多次片外读写 | 融合内核 → 片上保留中间状态 |
| 剪枝目的 | 减少模型大小或加速推理 | 优化融合执行效率，适配硬件约束 |
| 梯度通信 | 所有梯度均传输（含零） | 仅传输非零梯度，通信量锐减 |
| 整体效果 | 受限于数据移动瓶颈 | 显著加速训练，保持重建质量 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共评估六个广泛使用的3DGS数据集，涵盖大/小规模场景：
- **大规模场景**：`Mill-19`, `UrbanScene3D`, `MatrixCity`
- **中小规模场景**：`Mip-NeRF 360`, `DeepBlending`, `Tanks & Temples`

最大模型包含 **21.6 million Gaussians**，用于测试可扩展性。

---

### **实验设置**
- **硬件平台**：
  - A6000 Server：8 × NVIDIA RTX A6000（48GB GDDR6）
  - A100 Server：8 × NVIDIA A100（80GB HBM）
- **训练配置**：
  - 每GPU处理一个camera view
  - 使用原生分辨率进行训练与测试
  - 测试集采用每第8个视角的标准协议
- **变体对比（ablation variants）**：
  - `BASE`：原始 Grendel-GS [53]
  - `BASE + FUSED`：加入融合光栅化
  - `BASE + FUSED + SPCoMM`：进一步加入稀疏梯度通信
  - `ByteSplat`：完整方案（融合 + 硬件感知剪枝 + 零梯度剔除）

---

### **评估指标**
| 类别 | 指标 |
|------|------|
| **性能** | End-to-end per-iteration latency, Speedup |
| **通信效率** | Intra-GPU off-chip traffic, Inter-GPU communication volume |
| **质量保留** | PSNR, SSIM（与原始方法对比） |
| **剪枝分析** | Pruning ratio, % of tiles within fused-kernel budget |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | 结果 |
|------|------|
| **Intra-GPU off-chip traffic reduction** | ↓ **63.4%** |
| **Backward Inter-GPU communication volume reduction** | ↓ **65.8%**（最高达86.5%） |
| **End-to-end training speedup (A6000)** | 最高 **6.1×**，平均 **3.4×** |
| **Speedup (A100)** | 最高 **3.6×**，平均 **2.7×** |
| **Zero gradient ratio observed** | 高达 **72.0%**（不同数据集一致） |

---

### **与基线方法的对比结果**
- 在所有六大数据集上，**ByteSplat 均显著优于 BASE**。
- 即使在A100这种高带宽平台上仍取得近3倍加速，说明优化不仅依赖特定硬件。
- 大规模场景收益更高（几何平均加速 **3.7×** vs 小场景 **2.8×**），因其提供更多冗余数据移动消除机会。

---

### **消融实验结果**
![Ablation Study](#fig15)

| 方法 | 相对于 BASE 的加速比 |
|------|------------------|
| `BASE + FUSED` | **1.2×** |
| `BASE + FUSED + SPCoMM` | **1.6×**（较前一步提升1.3×） |
| `ByteSplat`（完整） | **3.2×**（较前一步再提速2.1×） |

> ✅ **结论**：三项技术具有**互补性**，其中硬件感知剪枝带来最大增益。

此外：
- **Tile-budget satisfaction rate** 从原始的 **33.2%** 提升至 **50.7%**（平均提升17.5个百分点）。
- **Model scaling test**：当Gaussian数量从20M增至120M，ByteSplat 仍稳定提供 **1.9–2.0×** 加速。

---

## **4. 关键结论和发现**

### **主要发现**
1. **数据移动是分布式3DGS训练的核心瓶颈**，而非计算本身。
2. **融合光栅化可有效减少片外内存访问**，但需配合剪枝以缓解片上资源压力。
3. **梯度稀疏性极高（~72%零梯度）**，剔除零梯度可显著降低通信负载。
4. **硬件感知剪枝不仅能提升性能，还能维持甚至略微改善PSNR/SSIM**，因剩余Gaussians可通过微调补偿被删去的部分。

### **方法的局限性**
- **依赖精确的零检测**：仅剔除“完全为零”的梯度，无法处理近似稀疏或量化后的梯度。
- **剪枝引入额外训练阶段**：需要预训练 → 剪枝 → 微调流程，增加部署复杂性。
- **对极小模型增益有限**：小规模场景由于本就通信较少，加速效果不如大型场景明显。

---

### **未来工作方向**
- **算法-架构协同设计（Algorithm-Architecture Co-design）**：更深层次地结合GPU微架构特性（如Tensor Core调度、缓存层级）优化执行路径。
- **动态自适应剪枝**：根据运行时负载自动调整剪枝强度与策略。
- **支持异构设备训练**：将ByteSplat思想扩展到CPU-GPU混合或边缘设备集群。
- **探索更激进的数据压缩**：结合量化、编码等手段进一步压缩梯度表示。

---

> 📌 **总结一句话**：  
> **ByteSplat 通过融合计算、硬件感知剪枝与零梯度剔除，在不牺牲重建质量的前提下，实现了高达6.1×的分布式3DGS训练加速，揭示了“减少数据移动”是突破当前训练瓶颈的关键路径。**

</details>

---

### 3. [Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen LLM Hear Beyond the Transcript](https://arxiv.org/abs/2610.02638)

**Authors**: Jie Jin, Ziyin Ma, Min Yin, Jinyu Chen, Haigang Song, Zhikun Pang, Xiaowen Zhang  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.02638v1  

#### Abstract
Full-duplex voice agents make many small, closed decisions, which current systems answer by slow autoregressive decoding. We propose DuplexJev, which feeds ASR-encoder hidden states through a small connector into a frozen LLM and reads each question as a single-token distribution over its options. N...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Batched Speech Decisions Without Decoding: Single-Token Supervision Lets a Frozen LLM Hear Beyond the Transcript*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前全双工（full-duplex）语音代理在处理用户话语时，需要做出大量**封闭式决策**（closed decisions），例如：
- 当前说话是否结束？（turn complete）
- 是否为插入语（barge-in）？
- 应播放哪个预录填充音（filler）？
- 说话人身份、情绪状态等。

传统方法依赖于**级联式流程**（cascaded pipeline）：先进行 ASR 转录，再通过 LLM 进行文本理解并生成答案。这种方式存在以下问题：
- **延迟高**：autoregressive decoding 步骤导致响应延迟。
- **成本高**：消耗大量 LLM 输入 tokens，占实际部署系统中高达 30% 的 token 开销。
- **无法感知副语言信息**（paralinguistics）：如性别、情感等非文字信息在转录过程中被丢弃。

此外，现有 speech-to-text LLM 多采用 **transcript distillation** 方式训练连接器（connector），其教师模型仅基于文本输出，**从未“听到”声音本身**，因此无法学习到语音中的声学线索。

---

### 🚀 提出的新方法与创新思路

作者提出 **DuplexJev** —— 一种无需解码、直接从语音嵌入中批量读取决策的框架，核心思想如下：

#### （1）**Decoding-free 决策机制**
- 不进行任何 autoregressive decoding。
- 将每个问题建模为一个 **single-token 分类任务**：将选项用字母 A/B/C/D 表示，模型只需预测下一个 token 的分布 $ p(\text{A}|q), ..., p(\text{D}|q) $。
- 输出即为 logits 归一化后的概率分布，无需采样，保证输出合法且可提供置信度。

#### （2）**Single-Token Supervision**
- 改变传统 distillation 范式：不再让 connector 学习 LLM 在 transcript 上的完整输出分布。
- 而是只对最终读出的那个 answer token 施加 **cross-entropy loss**。
- 同时保留 transcript distillation 用于对齐内容理解能力（content alignment）。
- 构成 **mixed objective**：一部分样本用 distillation（如转录、续写），另一部分用 answer-token CE（如分类决策）。

> 💡 这使得模型可以“听见”超出文字的信息（如语气、性别、情绪），因为监督信号来自真实标签而非文本先验。

#### （3）**高效批处理与前缀共享（Prefix Sharing）**
- 所有问题在同一 context 下可在一次 forward pass 中并行处理。
- 利用 KV Cache 实现 prefix sharing：共享对话历史、音频编码等长上下文，仅 suffix 差异化拼接。
- 显著降低计算和内存开销，尤其适用于长上下文场景。

#### （4）模块化设计
- 支持任意 ASR encoder 与任意 frozen LLM 组合。
- 只需训练轻量级 connector（projector + optional fusion block）。

---

### ⚖️ 相比现有方法的优势

| 维度 | 传统方法 | DuplexJev |
|------|--------|---------|
| 延迟 | 高（需 autoregressive decoding） | 极低（单步前向传播） |
| 成本 | 高（每决策消耗数百至数千 tokens） | 极低（无 decoding，共享 prefix） |
| 并发能力 | 弱（难以 batch across calls） | 强（支持跨调用、跨问题批量处理） |
| 对副语言信息的感知 | 几乎无（依赖文本） | 强（通过 answer-token supervision 学习） |
| 模块化与灵活性 | 差（端到端训练） | 强（encoder/LLM 可替换） |

---

## 2. 核心实验方法和设置

### 📚 数据集使用

| 类型 | 数据集 | 描述 |
|------|-------|------|
| **内容理解** | `qa100`（本文发布） | 自建双语（中英各50题）多选问答集，含逻辑与事实类问题 |
| | `ZJU-ML v2.0.0` | 包含真实录音与 TTS 的混合语音问答数据（共100题） |
| | `Easy-Turn` | 零样本 turn-taking 识别任务 |
| **副语言任务** | `AISHELL-1`, `LibriSpeech`, `Common Voice` | 性别识别（800 条真实语音） |
| | `ESD`, `CREMA-D` | 情绪识别（四类：neutral/happy/angry/sad，共800条） |
| **训练数据** | `WenetSpeech`, `GigaSpeech`, `LibriSpeech`, `Common Voice`, `CoVoST` | 多源 ASR/翻译混合数据，用于 content task（R1-R2阶段） |
| | `AISHELL-1 + LibriSpeech`（性别包）<br>`ESD + CREMA-D`（情绪包） | 用于 R3 阶段 paralinguistic 微调 |

---

### 🔧 实验设置

- **LLM**: Qwen3-32B（frozen, bf16）
- **Encoder**: Qwen3-ASR-0.6B 或 Whisper-large-v3-turbo（均 frozen）
- **Connector**:
  - **B（last layer）**: 最后一层隐藏状态经 MLP 投影进入 LLM 空间
  - **A（cross-attention fusion）**: 引入 cross-attention 融合多个 encoder 层（query=h18, key=h14, value=h9）
- **训练策略**:
  - R1-R2: 仅 content task，使用 transcript distillation（KL divergence）
  - R3: 加入 decision task，比较不同目标函数：
    - Distillation-only
    - Answer-token Cross-Entropy（CE）
    - MIX-CE（全部使用 CE）
    - MIX-KD（mixed objective）
- **评估指标**:
  - Accuracy（主要）
  - Latency（ms）、Throughput（events/s, decisions/s）
  - Confidence gating 效果
  - Probe performance on encoder features

---

### 🆚 基线方法对比

| 基线 | 描述 |
|------|------|
| **Cascade (JSON)** | ASR → LLM 生成 JSON 结构化回复（典型 autoregressive 流程） |
| **Readout on Transcript** | 使用相同 single-token readout，但在 transcript 上运行（模拟理想上限） |
| **Whisper + Released Connector** | 使用 Fixie Ultravox 发布的 Whisper-Qwen3 连接器作为外部基线 |
| **Zero-initialized baselines** | 验证初始化影响 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）**延迟与吞吐量优势**

| 方法 | 单事件延迟（ms） | 每秒事件数（within 0.5s） | 每秒决策数 |
|------|------------------|----------------------------|------------|
| Cascade (JSON) | 1567 | 0 | ~30 |
| Single-token Readout | **92** | **20.4** | **204** |
| 含 1.5k 上下文 | 144 | 12.0 | 120 |
| 8-GPU 节点（8 events） | ~100 | ~80 events/s | **~1,613 decisions/s** |

> ✅ 达到 **17倍延迟下降**，在严格实时预算下实现高并发。

#### （2）**内容理解性能（Accuracy %）**

| 输入方式 | qa100 | ZJU-ML | Easy-Turn |
|--------|-------|--------|----------|
| Oracle Transcript（上界） | 91 | 89 | 80.5 |
| Options Only（下界） | 36 | 39 | — |
| B (last layer, R2) | **90** | 79 | 76.1 |
| A (x-attn, R2) | 83 | 77 | 77.1 |
| Whisper + Qwen3-32B | 89 | 85 | — |

> ✅ 使用 last-layer connector（B）时，**语音问答准确率达 90%，接近文本输入的 91%**，几乎无损。

#### （3）**副语言识别性能提升显著**

| 方法 | Gender Acc (%) | Emotion Acc (%) | △qa100 |
|------|----------------|------------------|--------|
| Encoder Probe（上限） | 97.5 | 87.9 | — |
| Distillation-only (6k steps) | ~53 | ~28 | ±0 |
| **Answer-token CE (gender)** | **87.9–89.4** | 26–27 | -3~+3 |
| **Answer-token CE (emotion)** | 40–43 | **71.8–85.5** | -1~-10 |
| **MIX-KD (mixed objective)** | **89.9** | **90.0** | -1 |

> ✅ 从 **55% → 90%**（gender），**28% → 90%**（emotion），实现质的飞跃！

#### （4）消融实验关键发现

| 实验 | 发现 |
|------|------|
| **Prefix Sharing** | 与逐个推理结果一致（200 对中有 195–198 完全匹配），数值误差 < 0.06，证明精确性 |
| **Batching Across Calls** | 批量处理不影响准确性（800 条中 790 一致） |
| **Connector Architecture (A vs B)** | A（cross-attention）略优，但差异不大；**瓶颈不在架构而在 objective** |
| **Objective Comparison** | Distillation 完全无法提升 gender/emotion；仅更换 loss 即可大幅提升 → **objective 是关键瓶颈** |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Transcript Distillation 导致“聋哑”问题**
   - 教师模型未听声音 → 学生无法学到声学线索 → gender/emotion 停留在随机水平。
   - 即使 encoder 内部已有 strong signal（probe 可达 97%），connector 也无法传递。

2. **Single-Token Supervision 是突破口**
   - 仅监督 answer token 可有效引导模型利用语音中的副语言信息。
   - 不破坏原有 content understanding 能力（qa100 保持 87–90%）。

3. **高效批处理可行且必要**
   - 一次 forward pass 可处理多个问题、多个通话。
   - prefix sharing 极大节省显存与计算，使长上下文实用化。

4. **模块化设计增强泛化性**
   - 可无缝替换 ASR encoder（如换用 Whisper）而不需重新训练整个系统。

---

### ⚠️ 局限性

- **情绪训练会损害 factual 内容理解**：在 ZJU-ML 上损失可达 16–17 pts，尤其是在真实语音的事实类问题上。
- **confidence gating 仍有改进空间**：虽然可用 p ≥ 0.95 实现 93% 准确率，但覆盖率为 87%，意味着约 13% 的样本需拒答。
- **尚未验证流式输入场景**：目前假设每次处理固定片段，未测试完全 streaming 模式。
- **依赖高质量标注数据做 decision tuning**：虽比 instruction tuning 轻量，但仍需构造 labeled decision 数据集。

---

### 🔮 未来工作方向

1. **On-device deployment with small LLMs**
   - 将该范式迁移到边缘设备，结合小型 LLM 实现本地化快速决策。
   
2. **Dynamic confidence-based fallback**
   - 结合 confidence score 触发 full decoding 回退机制，在效率与精度间动态平衡。

3. **扩展更多 decision types**
   - 如 urgency detection, speaker change, overlap prediction 等。

4. **探索更高效的 connector 架构**
   - 如 low-rank adaptation (LoRA) 或更紧凑的 fusion mechanism。

5. **构建通用 spoken decision benchmark**
   - 推动社区关注 speech-to-decision 而非单一 speech-to-text 范式。

---

## 🔗 开源与资源发布

作者公开了以下资源：
- ✅ 模型权重（connectors）
- ✅ 训练配方（training recipe）
- ✅ 批量推理 pipeline（for full-duplex serving）
- ✅ 双语语音问答数据集 `qa100`

GitHub: [https://github.com/adventists-ai/duplexjev](https://github.com/adventists-ai/duplexjev)  
PyPI: `pip install duplexjev`

--- 

> **一句话总结**：  
> DuplexJev 通过 **single-token supervision + frozen LLM + no decoding**，实现了超低延迟、高并发、能“听见”语气与性别的语音决策系统，为下一代全双工语音代理提供了全新架构路径。

</details>

---

### 4. [Geometry Meets Physics: Data-Efficient Pre-Training for Unstructured Neural PDE Solvers](https://arxiv.org/abs/2610.03363)

**Authors**: Luis Medrano-Navarro, Giacomo Baldan, Qiang Liu, Benjamin Holzschuh, Jan Hagnberger, Mathias Niepert, Nils Thuerey  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.03363v1  

#### Abstract
Neural surrogate models for Partial Differential Equations (PDEs) on unstructured 3D geometries are often limited by poor generalization and the high cost of generating large-scale training datasets. Consequently, pre-training on massive datasets of related PDE dynamics has emerged as a critical alt...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Geometry Meets Physics: Data-Efficient Pre-Training for Unstructured Neural PDE Solvers**

---

## **1. 主要贡献和创新点**

### **解决的问题**
神经PDE求解器在**非结构化3D几何体**上的应用面临两大瓶颈：
- **泛化能力差**：模型难以适应多样化的复杂几何形状。
- **训练成本高**：依赖大规模预计算的仿真数据集，生成这些数据本身计算开销巨大。

现有基于预训练（pre-training）的方法虽然提升了鲁棒性，但仍严重依赖昂贵的离线数据存储，缺乏**数据效率**和**计算效率**。

---

### **提出的新方法与新思路**
本文提出了 **Geometry Meets Physics (GMP)** ——一种**无需磁盘存储预训练数据**的高效预训练框架，专为非结构化网格上的稳态与瞬态PDE求解设计。

#### **核心思想：在线生成监督信号（Online Supervision Generation）**
- **完全避免**对大规模仿真数据或高精度CAD模型的依赖。
- 在训练过程中**实时生成几何与物理监督信号**，实现“disk-data-free”预训练。

#### **针对两类问题的专用策略**
| 问题类型 | 预训练策略 | 关键技术 |
|--------|-----------|--------|
| **Steady-State**（稳态） | **几何驱动预训练**（Geometric Pre-training） | 在线生成随机几何体（如立方体、球体、四面体组合），训练模型预测其**表面法向量（normals）、曲率（curvature）和矢量距离场（VDF）** |
| **Transient**（瞬态） | **物理驱动预训练**（Physics Pre-training） | 利用GPU谱方法求解器（spectral solver）在线生成**标准PDE（如扩散、Burgers、KS方程）的合成解**，进行掩码自编码（masked autoencoding） |

此外，提出 **GMP-Net**，一个模块化架构，支持两种预训练模式的无缝集成。

---

### **相比现有方法的优势**
| 特性 | 传统FM | GeoPT [17] | GMP（本文） |
|------|--------|------------|-------------|
| **无物理标签** | ✅ | ✅ | ✅ |
| **低预训练成本** | ❌ | ❌ | ✅ |
| **唯一预训练阶段** | ❌ | ❌ | ✅ |
| **无需高细节CAD** | ❌ | ❌ | ✅ |
| **零磁盘存储** | ❌ | ❌ | ✅ |

> ✅ 表示具备该特性；❌ 表示不具备

- **计算成本极低**：几何预训练仅需 **4小时**（单A100），远低于下游任务训练时间（如80小时）。
- **通用性强**：适用于多种架构（SMART, AB-UPT, Transolver++等），且支持跨维度迁移（3D→2D）。
- **提升显著**：尤其在**小样本场景**下，大幅加速收敛并提高精度。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 类型 | 任务 | 样本数 | 几何/场信息 |
|-------|------|------|--------|------------|
| **DrivAerML** [56] | 稳态 | 汽车外流场（压力、剪应力） | 500 | ~9M表面点，160M体积点 |
| **SHIFT-Wing** [57] | 稳态 | 航空翼型跨音速流 | 6000 | ~3M表面点，6M体积点 |
| **KS Equation** (ours) | 瞬态 | Kuramoto-Sivashinsky 方程 | 630×30 | 2D，4096点，1通道 |
| **Ellipse [59]** | 瞬态 | 2D绕椭圆流动 | 1200×100 | 2D，1024点，3通道 |
| **SHIFT-Crash** [58] | 瞬态/稳态 | 汽车碰撞结构力学 | 768×12 | 3D，~400K点，4通道 |

---

### **实验设置与评估指标**

#### **评估范式**
- **低数据场景分析**：从完整数据集中抽取子集（16, 32, 64, 128样本）进行训练，测试泛化能力。
- **预训练 vs. 从头训练**（scratch）：比较相同架构下是否使用GMP预训练的效果差异。
- **微调协议统一**：所有投影层从头初始化，主干网络先冻结warm-up，确保公平。

#### **主要评估指标**
- **相对L2误差**（Relative L2 Error）：用于稳态任务。
- **归一化RMSE**（NRMSE）与**一步预测NMSE**：用于瞬态自回归任务。
- **收敛速度**：达到相同误差所需的epoch数。
- **数据效率**：达到相同性能所需的数据量减少比例。

---

### **基线方法对比**
- **架构级对比**：AB-UPT, Erwin, PTV3, Transolver++, GeoTranssolver, SMART
- **预训练方法对比**：
  - **从头训练**（Scratch）
  - **GeoPT** [16]：基于升维轨迹的几何预训练
  - **Zhang et al. [17]**：基于CAD几何的物理无关预训练
- 最终选择 **SMART** 作为基础架构，并扩展为 **GMP-Net**

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **稳态任务（Table 3）**
在 **DrivAerML** 上使用 **16个训练样本**时：
| 模型 | 相对L2误差（Scratch） | GMP预训练后 | 改进幅度 |
|------|------------------------|--------------|----------|
| Transolver++ | 0.685 | 0.503 | **27%** |
| AB-UPT | 0.439 | 0.450 | -2%（轻微过拟合） |
| **GMP-Net** | **0.412** | **0.346** | **16%** |

> GMP-Net在所有设置下均取得最低误差。

在 **SHIFT-Wing** 上使用16样本：
- GMP使Transolver++误差降低 **47%**，显示其在航空场景的强大迁移能力。

#### **瞬态任务（Table 5）**
在多个自回归任务中，GMP预训练带来 **10%-25%** 的NRMSE下降：
| 任务 | GMP-Net (Scratch) → GMP-Net (PT) | 改进 |
|------|-------------------------------|------|
| 2D KS, t=10 | 0.615 → 0.479 | **22%** |
| 2D Ellipse, t=10 | 0.044 → 0.033 | **25%** |
| 3D SHIFT-Crash, t=10 | 0.040 → 0.034 | **15%** |

> 即使是从未见过的物理系统，也能有效迁移。

#### **数据效率与收敛速度（Figure 7–9）**
- **收敛更快**：预训练模型在更少epoch内达到相同性能（**提速达2倍以上**）。
- **数据节省**：仅用16样本微调的预训练模型，性能优于从头训练使用32样本的结果。
- **气动力系数预测**：升力系数 $ C_L $ 的 $ R^2 $ 显著提升，尤其在低数据区。

---

### **消融实验结果**

#### **几何描述符有效性（Table 4）**
| 预训练方式 | DrivAerML (64样本) | 相对改进 |
|-----------|--------------------|---------|
| 仅Occupancy | 0.424 | -2.9% |
| 仅VDF | 0.402 | -3.4% |
| 仅Normals + Curvature | **0.387** | **-6.1%** |
| **All (N+C+VDF)** | **0.346** | **+16%** |

> 表明**表面法向与曲率**对捕捉边界层梯度至关重要。

#### **优化算法对比（ConFIG vs. 动态加权）**
- 使用 **ConFIG**（冲突感知梯度优化）可缓解多任务梯度冲突，进一步提升微调性能（见Table 8）。
- 在64样本DrivAerML上，ConFIG比动态加权再降 **2-3%** 误差。

#### **与其他预训练方法对比**
- **vs. GeoPT**：GMP训练时间仅 **4小时**，而GeoPT需约 **10天**（每轮12小时）。
- 在低数据区性能相当，高数据区GMP更优。
- **vs. Zhang et al. [17]**：无需高保真CAD，实用性更强。

---

## **4. 关键结论和发现**

### **主要发现**
1. **几何与物理可以解耦预训练**：通过分别学习几何结构与物理规律，可在无真实仿真数据的情况下构建强归纳偏置。
2. **在线生成监督信号可行且高效**：无论是几何特征还是简单PDE解，都能有效提升下游任务表现。
3. **小样本下优势最明显**：在工程实践中常见的**低数据场景**，GMP带来最大收益。
4. **表面微分几何特征至关重要**：法向量与曲率比单纯的距离场更能反映复杂PDE行为。
5. **模块化架构设计有效**：GMP-Net能灵活切换几何/物理分支，适配不同任务需求。

---

### **局限性**
- **对高度复杂、低数据场景仍可能过拟合**：如AB-UPT在极小数据下出现性能饱和。
- **预训练信号与真实物理存在差距**：尽管有迁移效果，但合成PDE与真实系统仍有分布差异。
- **目前未探索更多几何描述符**：如拓扑特征、高阶导数等可能进一步提升性能。

---

### **未来工作方向**
- 探索更丰富的**几何描述符集合**以增强表示能力。
- 将GMP扩展到**多物理场耦合**与**不确定性量化**任务。
- 结合**主动学习**策略，在线筛选最有价值的预训练样本。
- 探索**无监督域适应**机制，提升跨领域迁移能力。

---

> **总结**：本文提出的 **GMP框架** 为非结构化神经PDE求解器提供了一条**高效、低成本、免存储**的预训练路径，显著提升了模型在现实低数据工程场景下的可用性，推动了科学机器学习向实用化迈进。

</details>

---

### 5. [AFORE: Attention-FFN Disaggregation with Overlapped Reconfiguration of Experts](https://arxiv.org/abs/2610.03203)

**Authors**: Wenshuang Li, Youhe Jiang, You Peng, Jiawei Jiang, Binhang Yuan  
**Category**: cs.DC  
**Published**: 2026-10-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.03203v1  

#### Abstract
Efficient serving of Mixture-of-Experts (MoE) models is challenging due to large expert parameters, input-dependent expert activation, and dynamic workloads. Expert parallelism distributes expert computation across GPUs, while attention-FFN disaggregation (AFD) separates attention and feed-forward c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AFORE: Attention-FFN Disaggregation with Overlapped Reconfiguration of Experts

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在基于 **Attention-FFN Disaggregation (AFD)** 架构的大规模 **Mixture-of-Experts (MoE)** 模型推理服务中，**专家负载不均衡**（expert load imbalance）会严重制约端到端性能。由于每个 token 只激活部分专家，且路由是输入依赖的，导致某些 GPU 上的专家成为“热点”，形成瓶颈。

传统方法如 **Expert Parallelism (EP)** 和静态放置无法有效应对动态变化的负载。而 AFD 将 Attention 和 FFN 计算分离后，FFN 阶段的专家负载不均直接成为独立的流水线瓶颈，进一步放大了该问题。

此外，现有的 **expert reconfiguration**（专家重配置）机制面临两大挑战：
- **C1：需求动态且不可预测**：历史统计或静态配置难以准确反映即将到达的 microbatch 的专家需求。
- **C2：重配置开销大**：迁移专家权重、更新路由元数据等操作若暴露在关键路径上，可能抵消其带来的负载均衡收益。

### 提出了什么新方法或新思路
论文提出 **AFORE**，一种面向 AFD 架构的 **时序专家重配置系统**，其核心思想是利用 AFD 流水线结构提供的两个关键机会：

- **O1：Microbatch-level demand is visible**  
  在 microbatch 到达 FFN 阶段前，其专家-令牌分布已由 Attention 阶段生成，可被提前获取用于调度决策。

- **O2：Reconfiguration overhead can be overlapped**  
  存在一个时间窗口，可在当前 microbatch 执行的同时，**并行地迁移下一个目标 microbatch 所需的专家副本**，从而隐藏迁移延迟。

基于此，AFORE 将专家重配置建模为一个 **microbatch-aware 的在线调度问题**，设计了一个 **migration-aware scheduler**，综合考虑：
- 即将到来的 microbatch 的专家需求
- 当前 FFN worker 的负载情况
- 迁移操作的预期开销（特别是暴露在关键路径上的部分）

仅当迁移带来的性能增益大于其暴露开销时，才触发迁移。

### 相比现有方法的优势
- **更及时的决策**：基于真实未来的 microbatch 需求，而非历史统计或预测，避免反应滞后。
- **更低的运行时开销**：通过 **pipeline overlap** 将专家迁移与计算重叠，几乎完全隐藏迁移延迟。
- **更高的资源利用率**：动态调整专家副本分布，显著缓解 FFN 阶段的 straggler 问题。
- **轻量级实现**：采用 primary + redundant slot 设计，支持快速 NVLink-based GPU-GPU 权重迁移。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验基于四种代表性动态推理工作负载，源自以下公开数据集：
- **ShareGPT**：对话类请求（conversational）
- **FineWeb**：文档风格输入（document-style）
- **CodeForces**：代码生成任务（code-generation）
- **GSM8K**：数学推理查询（mathematical-reasoning）

这些 workload 具有不同的请求特征和专家路由模式，用于验证方法的通用性。

### 实验设置和评估指标

#### 硬件环境
- 集群：2 台机器，每台配备 8 块 **NVIDIA A100 GPU**
- 节点内互联：**NVLink/NVSwitch**（峰值带宽 300 GB/s）
- 节点间通信：RDMA 网络（50 GB/s）

#### 模型
- **110B 参数的 MoE 模型**（GLM-4.5-Air）
- 部署架构：**PD + AFD**（Prefill-Decoding + Attention-FFN Disaggregation）
- FFN 侧采用 **Expert Parallelism (EP)**，EP degree 最高至 128

#### 评估指标
- **Output Throughput**：输出吞吐量（tokens/sec）
- **P95 Inter-Token Latency (ITL)**：第95百分位的 token 生成延迟，衡量尾部延迟
- **Raw Migration Cost vs. Exposed Migration Cost**：原始迁移耗时 vs. 暴露在关键路径上的等待时间

### 基线方法对比
所有基线均在同一 AFD 运行时中实现，确保公平比较：
- **Static**：专家静态均匀分配，无重配置
- **EPLB**：基于历史负载统计周期性重平衡
- **HarMoEny**：运行时基于路由需求进行反应式负载均衡
- **Lina**：基于近期专家选择模式预测未来热点并预复制
- **Libra**：基于下一层 top-k 激活预测预取专家副本

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在四个动态 workload 上，**AFORE** 均取得最优性能：

| 指标 | 结果 |
|------|------|
| **Output Throughput 提升** | 相比最强基线提升 **10.1–17.6%** |
| **P95 ITL 降低** | 相比最强基线降低 **7.1–9.5%** |
| **相比 Static 放置** | 吞吐平均提升 **29.8%**，P95 ITL 平均降低 **18.2%** |

### 与基线方法的对比结果
- AFORE 在所有 workload 上均显著优于所有基线。
- 特别是在 **ShareGPT** 和 **FineWeb** 等高动态性场景下优势更为明显。
- 基线方法如 Lina 和 Libra 虽尝试预测，但仍受限于预测误差和迁移开销暴露在关键路径。

### 消融实验结果（Ablation Study）
通过禁用关键组件验证其有效性：

| 配置 | 相比 AFORE 的性能下降 |
|------|------------------------|
| **w/o Prefetch**（使用历史需求） | 吞吐下降 12.4–17.9%，P95 ITL 上升 6.3–10.5% |
| **w/o Overlap**（迁移在关键路径） | 吞吐下降 15.2–21.3%，P95 ITL 上升 12.2–16.0% |

> ✅ **结论**：**demand prefetching** 和 **pipeline overlap** 是 AFORE 成功的关键，二者缺一不可。

### 调度器效率与解质量
- **调度延迟**：中位数低于 **250 μs**，即使在 EP=128 时仍保持高效。
- **解质量**：在 EP ≥ 8 时，**99.5% 的 microbatch 达到最优最大 tile 负载**，证明调度算法接近最优。

### 迁移隐藏效果（Migration Hiding）
- **平均原始迁移成本**：0.453–0.495 ms
- **平均前瞻窗口（lookahead window）**：2.008–2.119 ms
- **暴露迁移成本**：**0 ms**（所有样本均在使用前完成迁移）

> ✅ **结论**：AFD 的 pipeline window 完全足以隐藏专家迁移开销，实现 **zero exposed cost**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **AFD 放大了专家负载不均衡的影响**，但也提供了 **解决该问题的独特机会** —— 即 microbatch 需求可见性和 pipeline overlap。
2. **基于真实未来需求的重配置** 显著优于基于历史统计或预测的方法。
3. **将迁移与计算重叠** 是控制开销的关键，否则重配置可能适得其反。
4. AFORE 的设计实现了 **高吞吐、低尾延迟、高资源利用率** 的统一，在多种 workload 下均表现稳健。

### 方法的局限性
- 依赖 AFD 架构，不适用于传统的 monolithic 执行模式。
- 当前实现假设专家副本迁移可通过 NVLink 高速完成，跨节点迁移可能面临更高延迟。
- 调度模型未显式建模多层之间的依赖关系，未来可扩展至跨层协同优化。

### 未来工作方向
- 扩展至 **multi-layer joint reconfiguration**，考虑层间路由相关性。
- 探索 **跨节点专家迁移** 与通信优化。
- 结合 **routing-aware 模型微调**，从源头减少负载倾斜。
- 支持 **异构硬件环境** 下的专家放置与迁移策略。

--- 

> **总结**：AFORE 通过洞察 AFD 架构的时序特性，提出了一个高效、实用的专家重配置框架，**将原本有害的负载不均衡问题转化为可被主动管理的优化机会**，为大规模 MoE 模型的高性能推理服务提供了重要解决方案。

</details>

---

### 6. [16-bit Precision of Convolutional Neural Networks on Microcontroller Units for 8-bit Costs](https://arxiv.org/abs/2610.03402)

**Authors**: Rui Liu, Benjamin Paa{\ss}en  
**Category**: cs.LG  
**Published**: 2026-10-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.03402v1  

#### Abstract
To deploy deep neural networks on edge hardware, highly efficient inference schemes are necessary that retain high accuracy. This work presents W16A16, a high precision (16-bit), fast speed, low energy quantization method. On a widely applied microcontroller architecture Armv7E-M, our proposed appro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《16-bit Precision of Convolutional Neural Networks on Microcontroller Units for 8-bit Costs》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在边缘设备（如 **MCU**）上部署深度神经网络（**DNNs**）面临资源受限（内存小、算力低、能耗敏感）的挑战。传统 **INT8 量化**虽能降低内存占用并提升推理速度，但会引入显著的**量化误差**，尤其在回归任务中严重影响模型精度。

然而，提高精度至 **16-bit** 通常被认为会带来更高的计算开销和能耗，限制其在 MCU 上的应用。

本文提出：**可以在不增加时间与能耗成本的前提下，在 MCU 上实现接近 FLOAT32 精度的 16-bit 量化推理**。

---

### 提出了什么新方法或新思路
提出了一种名为 **W16A16** 的新型量化方案：
- **W16A16**：权重（Weights）和激活值（Activations）均采用 **16-bit 对称量化**（symmetric quantization）。
- 充分利用 **ARMv7E-M 架构**的指令集特性（特别是 **SMLAD/SMLALD 双 MAC 指令**），使得 16-bit 运算与 8-bit 运算具有相近甚至更优的执行效率。
- 设计了高效的 **Seq2col** 数据布局转换方法，优化 1D 膨胀卷积的内存访问模式。
- 实现了 **Full-Precision 96-bit Cross-Register Shift** 和 **Two-Stage Saturation** 技术，解决高精度重量化过程中的溢出问题。

---

### 相比现有方法的优势
| 方面 | 优势说明 |
|------|--------|
| **精度** | 显著优于 W8A8 和 W8A16，量化误差降低约 **10 倍以上**，接近 FLOAT32 性能。 |
| **速度与能耗** | 推理时间与能耗与 W8A8 相当，且**显著优于 W8A16**（快约 17.6%）。 |
| **硬件适配性** | 针对 ARMv7E-M 架构优化，无需额外浮点运算，完全基于整数指令实现。 |
| **通用性支持** | 支持 TCN 等复杂结构（如因果膨胀卷积、非对称填充），而主流框架（如 TFLM、Cube.AI）无法有效支持。 |

> ✅ **核心理念突破**：**“16-bit 精度，8-bit 成本”** —— 在特定架构下，更高位宽不一定更慢或更耗电。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 任务类型 | 描述 |
|-------|---------|------|
| **C-MAPSS FD001** | 回归（Remaining Useful Life, RUL） | NASA 提供的航空发动机剩余寿命预测数据集，共 100 个训练引擎，14 个输入通道，窗口长度 30。 |
| **NinaPro DB2** | 回归 & 分类 | 表面肌电信号（sEMG）数据集，用于手势识别与力估计：<br>- **回归任务**：从 sEMG 预测手部施加的力（6 输出通道）<br>- **分类任务**：识别 50 种手势意图 |

---

### 实验设置和评估指标

#### 硬件平台
| MCU 型号 | 核心 | 主频 | 特性 |
|--------|-----|------|------|
| **STM32H723ZG** | Cortex-M7 | 550 MHz | 支持双发射（Dual-issue）、I/D Cache |
| **GD32F450VET** | Cortex-M4 | 200 MHz | 单发射，无 Cache |

两者均支持 **ARMv7E-M DSP 指令集**。

#### 量化配置
- 所有模型先以 **FLOAT32** 训练，后进行 **Post-Training Quantization (PTQ)**。
- 量化工具：W8A8 使用 PyTorch.TQ；W8A16 和 W16A16 为自研实现。
- 采用 **Per-channel 量化** 权重，**Per-layer 量化** 激活。

#### 模型架构
使用 **Temporal Convolutional Network (TCN)**：
- 4 层 TCN Block，每块含两个因果膨胀卷积层（dilation=2^i）
- 使用 **Weight Normalization**, **ReLU**, **Dropout (0.3)**
- 最终输出层为 1x1 Conv

#### 评估指标
| 类别 | 指标 |
|------|------|
| **精度性能** | - RMSE（相对于 FLOAT32 输出）<br>- 分类准确率（Accuracy） |
| **系统性能** | - 推理周期数（CPU Cycles）<br>- 推理时间（ms）<br>- 能耗（mJ）<br>- 平均电流（mA） |
| **对比方式** | 以 FLOAT32 TCN 输出为参考，计算量化模型输出的 RMSE，隔离量化误差影响 |

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **W8A8** | 权重和激活均为 8-bit，主流标准 |
| **W8A16** | 权重 8-bit，激活 16-bit（混合精度） |
| **Cube.AI (INT8/Float32)** | 商业级嵌入式 AI 工具链，作为外部 baseline |
| **FLOAT32** | 浮点全精度模型，作为性能上限参考 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 📊 回归任务量化误差（RMSE 相对于 FLOAT32）

| 方法 | C-MAPSS (End-to-End) | NinaPro DB2 (Force Reg.) |
|------|------------------------|----------------------------|
| **W8A8** | 1.89373 | 0.1970 |
| **W8A16** | 0.31093 | 0.0161 |
| **W16A16** | **0.01101** | **0.0015** |

> 🔍 **发现**：W16A16 的量化误差比 W8A8 **低约 100 倍**，比 W8A16 也低一个数量级。

#### 🎯 分类任务准确率（NinaPro DB2, 50-class gesture）

| 方法 | 准确率 |
|------|--------|
| **FLOAT32** | 53.31% |
| **W16A16** | 53.29% |
| **W8A16** | 53.29% |
| **W8A8** | 41.63% |

> ⚠️ W8A8 准确率下降 **11.66%**（p < 1e-10），而 W16A16 几乎无损。

---

#### ⏱️ 推理速度（CPU Cycles）

##### 层级测试（STM32H7，典型配置）
| 方法 | Cycle 数量级 | 相对表现 |
|------|---------------|----------|
| **W8A8** | 最低（基准） | 快 |
| **W16A16** | 略高于 W8A8（+2.29%） | 接近 |
| **W8A16** | 显著更高（+~30%） | 最慢 |

##### 端到端模型推理（TCN 全模型）
| 平台 | W8A8 vs W16A16 vs W8A16 |
|------|---------------------------|
| **STM32H7** | W8A8 ≈ W16A16 < W8A16<br>W16A16 比 W8A16 快 **17.6%** |
| **GD32F4** | W16A16 比 W8A16 快 **24.1%** |

> ✅ 尽管 W16A16 多用 2 条 `LDR` 指令加载数据，但由于省去了 **sign extension** 和 **reordering** 操作，整体仍更快。

---

#### 💡 能耗测量（使用 Nordic PPK2）

| 平台 | 方法 | 推理时间 (ms) | 总能量 (mJ) | 算法执行能量 (mJ) |
|------|------|----------------|--------------|--------------------|
| **STM32H7** | W8A8 | 2.77 | 1.80 | ~1.33 |
| | **W16A16** | **2.82** | **1.95** | **~1.47** |
| | W8A16 | 3.42 | 2.35 | ~1.82 |
| | Cube.AI (INT8) | 13.68 | 8.05 | — |
| **GD32F4** | W8A8 | 14.46 | 5.84 | — |
| | **W16A16** | **15.90** | **6.34** | — |
| | W8A16 | 20.59 | 8.45 | — |

> 🔋 W16A16 能耗仅略高于 W8A8，但远低于 W8A16（节省约 20–25% 能量）。

---

### 消融实验结果（隐含分析）

虽然未明确列出“消融实验”章节，但以下分析构成实质上的消融研究：

| 组件 | 影响分析 |
|------|---------|
| **对称量化（Symmetric Quantization）** | W16A16 放弃激活值的不对称量化（asymmetric），因 16-bit 动态范围足够大，避免了 `Z` 偏移带来的额外指令开销。 |
| **SMLAD/SMLALD 指令利用** | 分析表明，ARMv7E-M 中 16-bit MAC 与 8-bit 实际吞吐一致，且 8-bit 需预处理（SXTAB16/PKHTB），反而更慢。 |
| **Seq2col 内存布局优化** | 显著减少不连续内存访问，提升缓存命中率和 SIMD 利用效率。 |
| **96-bit Requantization Pipeline** | 解决了高精度中间结果溢出问题，保障数值稳定性，是 W16A16 可行的关键技术。 |

---

## 4. 关键结论和发现

### 主要发现
1. **ARMv7E-M 架构允许 16-bit 与 8-bit 整数 MAC 同速运行**，因此 W16A16 可达到与 W8A8 相当的速度。
2. **W8A16 实际最慢**：因其需同时处理 8-bit 和 16-bit 数据格式，导致更多预处理指令和寄存器压力。
3. **W16A16 在精度上碾压 W8A8**：量化误差降低 **约 100 倍**，分类准确率几乎无损，而 W8A8 下降严重。
4. **能耗方面，W16A16 仅略高于 W8A8，远优于 W8A16**。
5. **现有框架（TFLM、Cube.AI）无法有效支持 TCN**：要么报错，要么回退到浮点运算，失去效率优势。

> ✅ **最终结论**：**W16A16 实现了“16-bit 精度，8-bit 成本”**，是高精度边缘推理的理想选择。

---

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **依赖特定架构** | 优势建立在 **ARMv7E-M** 的双 MAC 指令基础上，不适用于可并行处理四个 8-bit 的新架构（如 ARMv8.1-M 或 RISC-V V 扩展）。 |
| **内存占用翻倍** | W16A16 模型参数和激活值占用内存是 W8A8 的两倍，可能不适合极端内存受限场景（< 256KB）。 |
| **当前验证范围有限** | 实验集中在 TCN 模型和两个数据集，尚未扩展至 Transformer、LSTM 或图像 CNN 等其他模型。 |

---

### 未来工作方向
1. **扩展至其他模型架构**：将 W16A16 方法应用于 LSTM、Transformer、MobileNet 等。
2. **探索其他硬件平台**：研究是否可在 RISC-V 或 ARMv8-M 上通过类似技巧获得收益。
3. **结合剪枝与稀疏化**：缓解 W16A16 带来的内存增长问题。
4. **训练时量化（QAT）支持**：进一步提升 W16A16 的精度极限。
5. **自动化工具链集成**：将该方法集成进主流 TinyML 框架（如 TFLM、CMSIS-NN）。

---

## 总结一句话
> **在 ARMv7E-M 架构 MCU 上，W16A16 量化实现了近乎 FLOAT32 的精度，同时保持与 INT8 相当的推理速度和能耗，打破了“低位宽一定更快”的固有认知，为高精度边缘智能提供了新路径。**

</details>

---

### 7. [HyperThink: Text-to-Parameter Hypernetworks for Efficient Reasoning](https://arxiv.org/abs/2610.03039)

**Authors**: Donggyun Kim, Jack Lu, Chanwoo Kim, Mengye Ren, Seunghoon Hong  
**Category**: cs.CL  
**Published**: 2026-10-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.03039v1  

#### Abstract
Long-form thinking traces can substantially improve the multi-step reasoning performance of large language models (LLMs), but they introduce high inference-time overhead, with latency dominated by sequential decoding. We propose HyperThink, a text-to-parameter approach that amortizes this reasoning ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HyperThink: Text-to-Parameter Hypernetworks for Efficient Reasoning 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLMs）在多步推理任务中表现优异，主要得益于 **Chain-of-Thought (CoT)** 和 **thinking traces**（如 `<think>...</think>`）等机制。然而，这些中间推理过程需要自回归生成大量 token，导致显著的 **推理延迟** 和高计算成本，尤其在对延迟敏感的应用场景中不切实际。

另一方面，**non-thinking 模式**（直接生成答案）虽然速度快，但容易通过“捷径”生成看似合理但错误的答案，牺牲了推理可靠性。

因此，核心问题是：  
> **能否在接近 non-thinking 的低延迟下，保留 thinking 模式的强推理能力？**

---

### **提出了什么新方法或新思路**

论文提出 **HYPERTHINK**，一种 **text-to-parameter 的超网络（hypernetwork）框架**，将显式的文本推理过程 **amortize**（摊销）为一次性的参数更新。

#### **核心思想**
- 不再将“思考”视为要生成的 token 序列，而是将其建模为 **查询条件下的模型参数扰动**。
- 给定输入问题 `q`，一个轻量级的 **hypernetwork** 预测一组参数更新 `Δθ(q)`，注入到冻结的 base LLM 中。
- 更新后的模型直接生成简洁的 step-by-step 解答，无需输出长的 thinking trace。

#### **关键技术组件**
- **Bias-Only Adaptation**：仅更新 LLM 中少量选定层的 bias 参数（如 `q_proj`, `v_proj` 等），实现高效且有效的模型调整。
- **Vector-Quantized (VQ) Bottleneck**：引入向量量化瓶颈，将连续的参数更新映射到有限的离散原型集合中，增强鲁棒性和跨实例的知识复用。

---

### **相比现有方法的优势**

| 方法 | 推理延迟 | 是否保留推理能力 | 参数效率 | 动态适应性 |
|------|----------|------------------|-----------|-------------|
| **Thinking Mode** | 高（长 trace） | ✅ 强 | ❌ 静态权重 | ❌ |
| **Non-Thinking Mode** | 低 | ❌ 易走捷径 | ✅ | ❌ |
| **System 2 Distillation** | 低 | ⚠️ 有限泛化 | ❌ 全模型微调 | ❌ |
| **TokenSkip / Budget-Controlled** | 中 | ⚠️ 压缩可能丢失信息 | ✅ | ❌ |
| **HYPERTHINK (Ours)** | **低（≈ non-thinking）** | ✅ **强（≈ thinking）** | ✅✅（仅 bias + 超网络） | ✅✅（每查询动态更新） |

> ✅ **优势总结**：
> - **低延迟**：仅需一次非自回归的 hypernetwork 前向传播 + 简短解码。
> - **高性能**：在数学和通用推理任务上显著优于 non-thinking 和 distillation 方法。
> - **模块化与可扩展**：base LLM 冻结，仅训练轻量 hypernetwork。
> - **强泛化性**：VQ 瓶颈提升 out-of-distribution 泛化能力。

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### 数学推理任务
- **GSM8K**：小学数学应用题，标准测试集。
- **MATH-500**：更具挑战性的数学竞赛题，用于评估 **out-of-distribution (OOD) 泛化能力**。

#### 通用推理任务
- **AIME**：奥数级别数学题。
- **LiveCodeBench**：代码生成任务。
- **CommonsenseQA**：常识问答。
- **BIG-Bench Hard**：复杂多步推理任务。

---

### **实验设置和评估指标**

#### **Base LLMs**
- **Qwen3-0.6B**
- **SmolLM3-3B**
- **Olmo-3-7B-Think**

所有模型均经过充分的 thinking/non-thinking 后训练，作为强基线。

#### **评估指标**
- **Accuracy (%)**：最终答案正确率。
- **Pass@5 (%)**：5 次采样中至少有一次正确的概率。
- **FLOPs (G)**：推理过程的浮点运算量，衡量计算成本。
- **Latency (s)**：端到端响应延迟（在 NVIDIA H200 GPU 上测量）。

#### **训练细节**
- Hypernetwork 使用 **AdamW** 优化器，学习率 `1e-4`，batch size 16。
- 文本编码器冻结，仅训练 hypernetwork 和 VQ codebook。
- 使用 **teacher forcing**，监督信号来自 base model 在 thinking mode 下生成的响应。

---

### **基线方法对比**

| 基线方法 | 描述 |
|---------|------|
| **Thinking Mode** | 生成完整 thinking trace，性能上限，但延迟高。 |
| **Budget-Controlled Thinking** | 限制 thinking token 数量，控制与 HYPERTHINK 相当的预算。 |
| **Native Non-Thinking Mode** | 直接生成答案，无 thinking trace，速度最快。 |
| **System 2 Distillation** | 使用相同目标函数但直接微调整个 base LLM。 |
| **TokenSkip** | 学习压缩 thinking trace，选择性跳过不重要 token。 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据（Qwen3-0.6B）**

| 方法 | GSM8K-Test (Pass@5) | MATH-500 (Pass@5) | FLOPs (G) |
|------|---------------------|--------------------|-----------|
| Thinking Mode | 87.41% | 80.40% | 4535.70 |
| Budget-Controlled | 56.94% | 61.48% | 1198.27 |
| Native Non-Thinking | 80.14% | 69.00% | 962.35 |
| System 2 Distillation | 75.06% | 50.80% | 838.52 |
| TokenSkip | 81.50% | 57.40% | 1562.24 |
| **HYPERTHINK (Ours)** | **82.11%** | **73.40%** | **1150.83** |

> ✅ **结论**：
> - 在 **GSM8K** 上，HYPERTHINK **超越所有基线**，甚至略优于 non-thinking。
> - 在 **MATH-500 (OOD)** 上，HYPERTHINK **大幅领先**，表明其更强的泛化能力。
> - FLOPs 仅略高于 non-thinking，远低于 full thinking。

---

### **更大模型上的表现（SmolLM3-3B）**

| 方法 | GSM8K-Test (Pass@5) | MATH-500 (Pass@5) | FLOPs (G) |
|------|---------------------|--------------------|-----------|
| Thinking Mode | 96.51% | 93.60% | 28460.97 |
| Budget-Controlled | 83.02% | 63.20% | 5645.65 |
| Native Non-Thinking | 95.15% | 84.20% | 7143.64 |
| **HYPERTHINK (Ours)** | **95.15%** | **81.20%** | **5900.99** |

> ✅ **结论**：
> - 在更大模型上，HYPERTHINK **保持竞争力**，在低延迟区域仍具优势。
> - 在 AIME 和 LiveCodeBench 上，high-budget thinking 仍更强，但 HYPERTHINK 在 near-non-thinking 区域提供更优权衡。

---

### **消融实验结果（Ablation Study）**

#### **组件消融（Table 3）**

| 变体 | GSM8K-Test (Pass@5) | MATH-500 (Pass@5) |
|------|---------------------|--------------------|
| System 2 Distillation (全微调) | 75.06% | 50.80% |
| Bias-Only (无 VQ) | 82.11% | 66.20% |
| HYPERTHINK (无 VQ) | 80.14% | 66.90% |
| **HYPERTHINK (完整)** | **82.11%** | **73.40%** |

> 🔍 **发现**：
> - **Bias-only adaptation** 本身已非常有效，优于全模型微调。
> - **VQ bottleneck** 对 OOD 性能（MATH-500）有巨大提升（+6.5%），防止过拟合，促进知识复用。

#### **参数子空间对比（Table 4）**

| 参数更新方式 | MATH-500 (Pass@5) |
|--------------|-------------------|
| LoRA | 74.00% |
| Prompt Tuning | 57.40% |
| **Bias (Ours)** | **73.40%** |

> ✅ **结论**：bias-only 在效率和性能之间取得最佳平衡。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **推理可以被“编译”为参数更新**：HYPERTHINK 成功将 thinking trace 的作用 **amortize** 为一次性的 query-conditioned parameter shift，实现了“无迹推理”。
2. ✅ **Bias-only adaptation 高效且强大**：仅更新少量 bias 参数即可显著改变模型行为，支持复杂推理。
3. ✅ **VQ bottleneck 是关键正则化手段**：它强制模型将参数更新组合成有限的“推理原型”，大幅提升 OOD 泛化能力，并防止过拟合。
4. ✅ **在低延迟区域具有显著优势**：HYPERTHINK 在 **near-non-thinking regime** 提供了最优的 **accuracy-latency trade-off**，特别适合对延迟敏感但又需可靠推理的场景。

---

### **方法的局限性**

- **依赖高质量 teacher signal**：性能受限于 base model 在 thinking mode 下的表现。
- **VQ codebook 容量有限**：若问题多样性极高，有限的 codebook 可能成为瓶颈。
- **不适用于极端复杂问题**：对于需要深度搜索的 Olympiad 级别问题，full thinking 仍是更强大的选择。
- **额外训练成本**：需要训练 hypernetwork，尽管 inference 高效，但 training 成本存在。

---

### **未来工作方向**

1. **探索更丰富的参数更新空间**：如结合 LoRA 或 adapter，研究 hybrid adaptation。
2. **动态 codebook 扩展**：允许 codebook 在推理时动态增长或检索。
3. **多模态 HYPERTHINK**：将图像、代码等输入也纳入参数预测。
4. **在线 test-time adaptation**：结合 test-time training 思想，在推理时进一步微调 hypernetwork。
5. **理论分析**：形式化分析 bias update 如何模拟 in-context learning 或 gradient descent。

---

> 📌 **总体评价**：  
> HYPERTHINK 提出了一种新颖且高效的推理范式，将“思考”从 token 空间转移到 parameter 空间，为构建 **快速而可靠** 的推理系统提供了新路径。其核心思想——**amortized, query-conditioned parameter steering**——有望启发更多高效推理与自适应学习的研究。

</details>

---

### 8. [VenusRL: A Fully Disaggregated Agentic RL System with Priority Scheduling and Scalable Interaction](https://arxiv.org/abs/2610.03286)

**Authors**: Mingjun Zhang (Institute of Computing Technology, Chinese Academy of Sciences), Yucheng Li (Beihang University), Menghao Zhang (Beihang University), Shuyong Zhu (Institute of Computing Technology, Chinese Academy of Sciences), Ping Zhang (Infrawaves), Xiaohe Hu (Infrawaves), Jun Chen (Infrawaves), Zhixin Wang (Shanghai Innovation Institute), Xutong Wang (Infrawaves), He Liu (Infrawaves), Yanmin Jia (Infrawaves), Shengrong Zhu (Infrawaves), Peng Sun (Shanghai Zhifeng Co., Ltd), Mingjie Zhang (Infrawaves), Liming Liu (Shanghai Innovation Institute), Jinlong Hou (Shanghai Innovation Institute), Yuan Cheng (Shanghai Innovation Institute), Yujun Zhang (Institute of Computing Technology, Chinese Academy of Sciences)  
**Category**: cs.DC  
**Published**: 2026-10-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.03286v1  

#### Abstract
Agentic Reinforcement Learning (RL) trains LLM agents through multi-turn interactions with external tool environments. Its multi-turn nature exposes two system-level bottlenecks unaddressed by existing agentic RL frameworks. First, end-to-end training throughput is constrained by the slowest traject...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# VenusRL 论文总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题

VenusRL 针对 **Agentic Reinforcement Learning (RL)** 中存在的两个系统级瓶颈：

1. **多轮交互下的训练吞吐受限**  
   传统端到端训练受最慢轨迹拖累，且现有 action-level 调度策略虽提升 GPU 利用率，但导致 rollout 进度分散，无法有效加速整体训练。

2. **环境沙箱资源利用率低下**  
   工具沙箱（sandbox）通常静态分配内存上限，造成大量“内存搁浅”（memory stranding）；同时，来自同一 prompt 的多个 rollout 沙箱间存在大量重复状态（state duplication），缺乏共享机制。

---

### 提出的新方法与创新思路

VenusRL 是一个**完全解耦的（fully disaggregated）Agentic RL 系统**，提出两大核心组件：

#### ✅ **Priority-aware Action Scheduler（优先感知动作调度器）**

- 引入**长度预测启发式算法**，识别即将完成的关键样本组（critical groups），这些组的完成最可能解锁下一轮训练。
- 在三个层面实现优先级调度：
  - **GPU Slot 执行**：高优样本抢占低优任务；
  - **KV Cache 居留管理**：基于轨迹感知 Radix Cache 和三级模型（unused / low_ref / high_ref），保护高优轨迹的 KV 缓存；
  - **跨 worker 请求编排**：监控 KV 压力并迁移请求及其前缀缓存，缓解负载不均。

#### ✅ **Environment Resource Manager（环境资源管理器）**

- **动态准入阈值控制**：基于实际内存增长预测而非声明上限来决定是否接纳新沙箱，显著提高部署密度。
- **模板级页面共享池（group-level page sharing pool）**：
  - 同一组内沙箱共享只读页（通过写保护 PTE 映射）；
  - 写操作触发 Copy-on-Write（CoW），保证隔离性；
  - 利用沙箱初始化代码高度相似的特点，减少冗余内存占用。

---

### 相比现有方法的优势

| 维度 | VenusRL 优势 |
|------|-------------|
| **训练效率** | 最高可达 **4.24× 端到端训练加速**，优于 Slime、RollFlash 和 ThunderAgent |
| **环境成本** | 沙箱部署密度提升至 **905%**，环境资源成本降低 **89%** |
| **KV Cache 利用** | 减少高优轨迹的缓存驱逐与重计算开销，命中率保持在 95% 以上 |
| **系统设计** | 完全解耦架构支持灵活扩展，优先级机制不影响训练稳定性 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **OpenSWE 数据集**：包含 45,320 个可执行 Docker 环境，覆盖超过 12.8K 仓库，用于模拟真实软件工程任务。
- 任务类型为 **SWE-agent**（Software Engineering Agent），涉及代码生成、测试运行、格式检查等复杂交互。

---

### 实验设置

| 参数 | 设置 |
|------|------|
| **硬件平台** | 4 台服务器，共 32 块 NVIDIA Hopper GPU，每节点 8 GPU + 180 CPU 核心 + 1.8TB 主机内存 |
| **通信网络** | NVLink（900GB/s 内节点）、InfiniBand（400Gb/s 跨节点） |
| **模型** | Qwen3-4B 和 Qwen3-32B |
| **最大输出长度** | 128K tokens |
| **每轨迹最大回合数** | 300 turns |
| **采样方式** | GRPO 算法，每个 prompt 生成 8 个样本 |
| **温度参数** | 0.7（鼓励探索） |
| **权重版本阈值** | 2（丢弃跨越 3 个以上版本的样本） |

---

### 评估指标

- **端到端训练吞吐量（End-to-end training throughput）**
- **奖励得分收敛曲线（Reward score）**
- **KV Cache 命中率**
- **沙箱并发数与内存占用**
- **消融实验中的各模块贡献**

---

### 基线方法对比

| 基线系统 | 特点 |
|--------|------|
| **Slime [68]** | 广泛使用的开源框架，采用 trajectory-level 调度，易出现 turn-level agentic bubbles |
| **RollFlash [33]** | 解耦环境控制器，支持 action-level 调度，提升 GPU 利用率但未优化训练进度 |
| **ThunderAgent [28]** | 程序感知调度器，主动驱逐短上下文以保留长上下文连续性，但仍无优先级机制 |
| **E2B [13]** | 代表性沙箱平台，作为环境成本对比基准 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 结果 |
|------|------|
| **最高端到端训练加速比** | **4.24×** vs RollFlash（Qwen3-32B, bsz=64） |
| **vs Slime 加速比** | 1.07–3.26× |
| **vs ThunderAgent 加速比** | 最高 **2.67×** |
| **环境成本降低** | 最多 **89%** vs E2B |
| **沙箱部署密度提升** | 从原生 E2B 的 100 → VenusRL 达 **905**（提升 9×） |
| **KV Cache 命中率** | 在 bsz=64 下仍维持 >95%，HiCache 仅为 ~79% |

---

### 与基线方法的对比结果

#### 📈 图 10：不同 batch size 与异步因子下的训练性能
- 在所有配置下，VenusRL 均优于所有基线。
- 尤其在大 batch size（32/64）时优势明显：
  - Qwen3-32B 上达到 **4.24×** 于 RollFlash。
- 即使在严格同步训练（async_factor=1.0）下仍有 **1.07–1.85×** 提升。

#### 📊 图 11：奖励得分收敛曲线
- VenusRL 与 Slime、RollFlash 收敛速度一致，最终奖励相近，说明**训练质量未受损**。

#### 🔥 图 12：权重版本热图
- VenusRL 的长尾样本丢弃率不高于基线，验证了其**不会因优先调度而引入严重偏置**。

---

### 消融实验结果（Ablation Study）

#### 表 1：各模块对训练加速的累积贡献（Qwen3-32B, async_factor=1.5）

| 方法 | bsz=16 | bsz=32 | bsz=64 |
|------|--------|--------|--------|
| Baseline | 1.00× | 1.00× | 1.00× |
| + Priority-aware Schedule | 1.01× | 1.81× | 1.42× |
| + KV Cache Residency | 1.06× | 2.72× | 2.78× |
| + Request Migration | **1.10×** | **2.89×** | **3.26×** |

- **KV Cache 居留管理**是最大贡献者，在 bsz=64 下带来 **~1.36×** 额外增益。
- **请求迁移**进一步缓解跨 worker 不均衡，贡献高达 **0.48×**。

#### 图 17：KV Cache 回落机制有效性
- 使用 HiCache：hit rate ≈ 82%（bsz=32），性能差；
- 加入 trajectory-aware radix cache：hit rate ↑ 至 93%，提速 56%；
- 再加入 **residency-aware priority demotion**：hit rate >95%，总提速达 **2.09×** vs HiCache。

---

## 4. 关键结论和发现

### 主要发现

1. **Action-level 调度不足以最大化训练吞吐**  
   单纯提升 GPU 利用率可能导致 rollout 进度分散，反而延迟训练启动时机。

2. **优先级调度 + KV 缓存协同设计至关重要**  
   仅靠调度无法解决 KV 驱逐带来的重计算开销，必须结合缓存保护机制才能释放潜力。

3. **沙箱内存浪费主要源于保守分配与状态复制**  
   动态准入 + 页面级共享可将单节点沙箱容量提升近 **10 倍**。

4. **预测驱动的轻量级优先级机制可行且高效**  
   剩余长度预测 Spearman 相关系数达 **0.98**，足以支撑有效排序；预测延迟远小于工具执行时间，可完美隐藏。

5. **训练稳定性得以保障**  
   - 通过 staleness-boundary promotion 防止长期饥饿；
   - 实验显示丢弃率未上升，收敛行为正常。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖长度预测准确性** | 若任务难度与历史轨迹无关，预测效果可能下降（但可通过定期微调更新模型缓解） |
| **强隔离环境下难以动态扩缩容** | 当前未集成 auto-scaling，因 microVM 不支持运行时弹性调整资源 |
| **RDMA 迁移依赖特定硬件** | Sandbox migration 使用 RDMA 加速，普通集群可能不具备该能力 |
| **两优先级类设计简化了调度逻辑** | 多级优先级可能更精细，但也增加复杂性和开销 |

---

### 未来工作方向

1. **集成动态资源伸缩机制**  
   探索基于 workload 的自动扩缩容策略，进一步提升资源利用率。

2. **支持更多类型的环境抽象**  
   如 WASM、容器等轻量级执行环境，适配不同安全与性能需求场景。

3. **增强预测鲁棒性**  
   引入不确定性估计或在线学习机制，适应策略演进过程中的分布漂移。

4. **跨节点全局调度优化**  
   当前 focus 在单节点内调度，未来可扩展为集群级统一调度器。

5. **与 off-policy correction 方法结合**  
   如 TIS 或 IcePop，进一步提升高并发异步训练下的策略稳定性。

---

> ✅ **总结一句话**：  
> VenusRL 通过**优先级感知调度 + 高密度沙箱部署**，实现了 Agentic RL 系统在**训练效率**与**环境成本**上的双重突破，是迈向大规模智能体训练的重要基础设施进展。

</details>

---

### 9. [EdgeAgent: Orchestrating On-Device LLM inference for End-User Multi-Agent Systems on CPU-GPU Unified Memory Architectures](https://arxiv.org/abs/2610.03394)

**Authors**: Yuhai Long (School of Computer Science and Engineering, Sun Yat-sen University), Yuanxin Wei (School of Computer Science and Engineering, Sun Yat-sen University), Kai Wu (China Mobile Internet Company Ltd), Jinhui Wei (School of Computer Science and Engineering, Sun Yat-sen University), Dan Huang (School of Computer Science and Engineering, Sun Yat-sen University), Jiangsu Du (School of Computer Science and Engineering, Sun Yat-sen University)  
**Category**: cs.DC  
**Published**: 2026-10-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.03394v1  

#### Abstract
Emerging multi-agent LLMs demand privacy-preserving edge deployment, yet current inference systems struggle with these collaborative workflows. Specifically, the memory-bound decode phase causes severe bus contention on unified memory architectures (UMA), paralyzing naive CPU-GPU co-execution. Furth...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EdgeAgent: Orchestrating On-Device LLM Inference for End-User Multi-Agent Systems on CPU-GPU Unified Memory Architectures

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代边缘设备上的 **multi-agent LLM** 应用（如 LangGraph、OpenClaw）对隐私保护和本地化推理提出了更高要求。然而，现有的 LLM 推理系统在处理这类协作式、高碎片化的多智能体任务时面临以下挑战：

- **内存瓶颈**：在统一内存架构（UMA）下，decode 阶段是 memory-bound 的，导致 CPU-GPU 协同执行时总线争用严重，硬件利用率低下。
- **静态调度低效**：传统 speculative decoding（SD）采用固定长度的 draft，无法适应不同 agent 间巨大的 drafting difficulty 差异（例如复杂推理 vs 结构化代码生成）。
- **工具调用阻塞**：agent 在执行外部工具（如 API 调用）时会陷入长时间 stall，造成计算资源浪费。

### 提出了什么新方法或新思路
作者提出 **EDGEAGENT**，一个专为边缘 UMA 架构和 multi-agent 工作负载协同设计的跨层推理系统，包含两个核心层次：

#### （1）UMA-Aware Execution Layer（微架构层）
- **Zero-Copy Tensor Parallelism (TP)**：通过非对称内存布局（asymmetric memory layout），让 CPU 使用 SME tile 格式预打包权重，GPU 保持线性格式，实现零拷贝共享模型权重。
- **Lock-Free 并行写入机制**：引入自定义的 **Logical Dependency Barrier** 替代传统的 `concat` 操作，避免显式内存复制，允许 CPU 和 GPU 直接向同一物理 buffer 写入不重叠的数据区域。
- **SME Micro-Kernel 优化**：针对 Apple M4 的 ARMv9 SME 扩展定制高效 GEMM 内核，结合指令流水、缓存分块和动态 work-stealing 实现负载均衡。

#### （2）Agent-Aware Scheduling Layer（调度层）
- **动态 draft 预算分配**：基于 **Historical Accepted Length (HAL)** 动态调整每个 agent 的 draft 长度。HAL 是接受 token 数量的指数移动平均，反映当前序列的可预测性。
- **异步 suspend-and-yield 机制**：当 agent 因工具调用而 stall 时，立即将其挂起并释放其占用的 speculative slot，供其他活跃 agent 使用；恢复后无需重新 prefill，直接 resume。

### 相比现有方法的优势
| 维度 | 传统方法缺陷 | EDGEAGENT 改进 |
|------|---------------|----------------|
| **内存管理** | 图编译器强制 `concat` 导致昂贵内存拷贝 | 自定义 barrier 实现 zero-copy，节省带宽 |
| **TP 效率** | 编译器同步开销大，破坏并行性 | Lock-free 写入 + 异构 fast path 提升吞吐 |
| **Speculative Decoding** | 固定 draft 长度，资源错配 | 动态按需分配 draft budget，提升 acceptance rate |
| **工具调用容忍** | Stall 导致 head-of-line blocking | 主动 suspend/yield，隐藏延迟，维持硬件饱和 |

---

## 2. 核心实验方法和设置

### 使用的数据集
由于缺乏标准的 multi-agent 推理基准，作者构建了一个 trace-driven 的合成 workload 生成器，融合三个真实数据集：
- **LongBench [7]**：用于模拟 **reasoning task**（难起草，acceptance 率低）
- **MBPP [6]**：用于模拟 **structured output task**（易起草，acceptance 率高）
- **ToolBench [15]**：用于模拟 **tool execution task**（伴随网络延迟）

最终 workload 按照 **1:2:2** 的比例混合三种任务类型，并注入 log-uniform 分布的工具 stall 时间（[1,10]s 或 [1,100]s）。

### 实验设置
- **硬件平台**：
  - 主测试平台：Apple M4 SoC（10-core CPU + 10-core GPU，32GB 统一内存，120 GB/s 总线带宽）
  - 对比平台：Apple M4 Pro（更高带宽，验证泛化性）
- **模型**：
  - DeepSeek-R1-Distill-Llama-8B（侧重推理能力）
  - Llama-3.1-8B-Instruct（通用指令遵循）
  - Draft model：EAGLE-3
- **精度**：FP16

### 评估指标
- **End-to-End Makespan**：完成整个混合 workload 所需的墙钟时间（wall-clock time）
- **Global Throughput**：所有并发流的总 token 生成速率（tok/s）
- **Speedup**：相对于 baseline 的加速比

### 基线方法对比
| Baseline | 描述 |
|---------|------|
| **Batch-AR** | 朴素连续 batching，无 speculative decoding |
| **Seq-SD (EAGLE-3)** | 单流 speculative decoding 上限 |
| **Batch-SD (Batched EAGLE-3)** | 当前最强 baseline：GPU 上 batched speculative decoding，每请求固定 draft 长度 $L=16$ |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 场景 | 方法 | Speedup (vs Batch-SD) | Global Throughput |
|------|------|------------------------|--------------------|
| 纯代码生成（无 stall） | EDGEAGENT | **1.29×** | 33.6 tok/s (DeepSeek) |
| 混合任务（无 stall） | EDGEAGENT | **1.33×** | —— |
| 混合任务 + [1,100]s stall | EDGEAGENT | **1.77×** | 11.4 tok/s → 提升 15.2% |

> 注：1.77× 加速中，1.29× 来自 UMA-aware execution 层，其余来自 agent-aware scheduling。

### 与基线方法的对比结果
- 在极端工具 stall 场景下（N=4 agents, stall up to 100s），Batch-SD 的 makespan 达到 **213.1 秒**，而 EDGEAGENT 仅为 **120.6 秒**，显著缓解了 head-of-line blocking。
- 在纯 decode 场景下，传统 concat 同步方式带来高达 50% 的额外延迟，EDGEAGENT 的 zero-copy barrier 将此降低至 < 5%，获得约 **1.14–1.31×** 的 decode 加速。

### 消融实验结果
#### （1）组件逐步启用（Figure 8）
在 M4 上逐步添加优化：
- Baseline (Batch-SD) → + Optimized SME Kernels → + Zero-Copy TP → + HAL Scheduling
- 最终端到端 makespan 缩短 **1.77×**

#### （2）HAL 调度有效性（Figure 13）
- 动态预算分配使总执行时间从 **434s**（uniform）降至 **388s**，实现 **1.12×** 加速。
- 高 predictability 的 agent 获得更多 draft slots，低 predictability 的被主动抑制，避免无效计算。

#### （3）Workload Mix 敏感性分析（Figure 9）
- HAL 调度在 structured-heavy（code/tool-rich）场景下增益更大，在 reasoning-heavy 下仍优于 baseline。
- 默认 1:2:2 设置是一个保守而非最优点，说明 EDGEAGENT 在更偏向结构化输出的任务中潜力更大。

#### （4）Prefill 性能（Figure 10）
- 在 prefill 阶段，CPU+GPU co-execution 达到 **304.5 tokens/sec**，相比 GPU-only 提升 **1.31×**。
- TTFT（Time to First Token）从 21.0s（GPU-only）降至 16.2s（4096 token prompt）。

#### （5）SME Kernel 效率（Figure 12）
- 在小 batch decode 场景（M=1），自定义 SME kernel 达到 **85.0 GFLOPS**，比 Apple BNNS 快 **4.47×**。
- 在大 batch prefill 场景（M=4096），峰值达 **2.07 TFLOPS**（理论上限 ~2.3 TFLOPS），利用率达 **>90%**。

---

## 4. 关键结论和发现

### 主要发现
1. **UMA 架构下必须绕过图编译器限制**：传统 DAG 抽象无法安全表达 lock-free 的跨设备写入，必须引入轻量级 barrier 和 fast path 才能实现真正的 zero-copy TP。
2. **multi-agent workload 具有高度异质性**：不同 agent 的 drafting difficulty 差异巨大，静态 draft 分配会造成严重的资源浪费。
3. **工具调用是性能杀手**：即使单个 agent stall，也会拖累整体系统效率；必须支持快速 context switch 以隐藏延迟。
4. **跨层协同设计至关重要**：仅靠硬件优化（如 zero-copy TP）只能带来 1.29× 提升，结合 agent-aware scheduling 后可达 **1.77×**，证明语义层与物理层联合优化的必要性。

### 方法的局限性
- **依赖 UMA 架构特性**：zero-copy 和 in-place KV freezing 严重依赖于统一地址空间和硬件一致性协议，在 discrete GPU 架构上难以直接迁移。
- **SME kernel 专用性强**：虽然逻辑可移植，但高性能依赖于特定 ISA（如 ARMv9 SME），在 x86 或 NVIDIA 平台需重新实现。
- **HAL 是启发式策略**：未提供理论最优保证，仅基于经验信号进行调度决策，可能在某些动态变化剧烈的场景下响应滞后。

### 未来工作方向
- **扩展至 discrete-GPU 数据中心**：将 suspend/yield 和 HAL 调度思想应用于 PCIe 连接的异构环境，结合 FlexGen 类 offloading 策略。
- **支持更多 agent 控制流**：如条件分支、循环等复杂 workflow，需要更强的 runtime 支持。
- **自动化 tuning HAL 参数**：当前 decay factor $\gamma=0.3$ 和初始值通过网格搜索确定，未来可探索在线自适应调节。
- **集成训练时优化**：结合 EAGLE-3 等训练感知的 speculative decoding 方法，进一步提升 acceptance rate。

---

> ✅ **一句话总结**：  
> **EDGEAGENT** 通过 **UMA-aware zero-copy TP** 与 **agent-aware dynamic scheduling** 的跨层协同设计，在 Apple M4 上实现了最高 **1.77×** 的 end-to-end 加速，显著提升了边缘 multi-agent LLM 系统的资源利用率与响应效率。

</details>

---

### 10. [Learning the Latent Structure: A Feature-Centric Approach to Graph Data Augmentation](https://arxiv.org/abs/2610.02517)

**Authors**: Yu Song, Zhigang Hua, Yan Xie, Bingheng Li, Jingzhe Liu, Bo Long, Jiliang Tang, Hui Liu  
**Category**: cs.LG  
**Published**: 2026-10-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.02517v1  

#### Abstract
Graph-structured data plays a pivotal role in modeling complex relationships. However, real-world graphs are often incomplete due to data collection and observational constraints, severely limiting the effectiveness of modern graph learning pipelines. While existing Graph Data Augmentation (GDA) met...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning the Latent Structure: A Feature-Centric Approach to Graph Data Augmentation*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现实世界中的图数据（Graph-structured data）普遍存在**连接不完整**（incomplete connections）的问题，例如由于数据采集限制、观测缺失或冷启动场景导致图结构稀疏。这种不完整性严重影响了 GNN 等图学习模型的性能，甚至使其退化为简单的 MLP。

现有的 **Graph Data Augmentation (GDA)** 方法虽然试图通过优化图结构来提升下游任务表现，但存在以下关键缺陷：
- **依赖标签监督**（label-dependent），难以在无标签场景下应用；
- **计算开销大**，常涉及双层优化（bilevel optimization）或迭代训练；
- **仅适用于直推式学习**（transductive），无法泛化到未见图（unseen graphs）或动态环境；
- 难以扩展到大规模图（scalability issues）。

---

### 🚀 提出的新方法与核心思想

本文提出了一种全新的 **feature-centric 图数据增强框架** —— **SelfAug**，其核心创新在于：

#### （1）从“结构建模”转向“表示增强”
不再显式地修改或重构图结构 $ A $，而是直接在**嵌入空间**（embedding space）中对节点表示进行增强。  
通过建模如下关系：
$$
Z_{\text{comp}} = Z_{\text{obs}} + \Delta(Z_{\text{obs}})
$$
其中 $ Z_{\text{obs}} $ 是基于不完整图得到的嵌入，$ Z_{\text{comp}} $ 是理想完整图下的嵌入，$ \Delta $ 是一个轻量级 MLP 增强模块，用于预测残差。

> **意义**：将复杂的图结构学习问题转化为节点级别的表示修正任务，复杂度从 $ O(n^2) $ 降至 $ O(n) $，极大提升了效率和可扩展性。

#### （2）自监督逆掩码训练机制（Self-supervised Inverse Masking）
- 在训练时，随机移除部分边构造更稀疏的掩码图 $ G_{\text{mask}} $；
- 使用 GNN 编码器生成 $ Z_{\text{mask}} $ 和原始图的 $ Z_{\text{obs}} $；
- 训练目标是让增强器 $ \Delta $ 将 $ Z_{\text{mask}} $ 映射回接近 $ Z_{\text{obs}} $ 的表示。

该设计无需任何任务标签，实现完全**无监督训练**，且增强器一旦训练完成即可在测试图上单次前向传播使用。

#### （3）鲁棒性增强技术
为应对训练图本身可能存在的噪声与稀疏性，引入两个关键技术：
- **Message Regularizer**：鼓励相似节点间的消息传递具有一致性，抑制噪声连接的影响；
- **Bootstrap Augmentation Strategy**：动态更新目标表示，利用增强器自身输出逐步逼近更完整的结构信号。

---

### ⭐ 相比现有方法的优势
| 维度 | SelfAug | 传统 GDA 方法 |
|------|--------|---------------|
| 是否需要标签 | ❌ 否（self-supervised） | ✅ 多数需要 |
| 泛化能力 | ✅ 支持归纳式（inductive）和冷启动场景 | ❌ 多为直推式 |
| 推理效率 | ✅ 单次前向传播 | ❌ 需要图级别优化 |
| 可扩展性 | ✅ 轻量 MLP，适合大图 | ❌ 常有 $ O(n^2) $ 开销 |
| 部署便捷性 | ✅ 模型即插即用 | ❌ 每个图需重新优化 |

> SelfAug 是首个真正意义上**可部署于真实工业场景**的通用图数据增强方案。

---

## 2. 核心实验方法和设置

### 📊 数据集
共使用 **10 个跨领域的标准图基准**，涵盖学术引用网络与电商图：
- **学术图**：CORA, CiteSeer, PubMed, DBLP, WikiCS, OGBN-ARXIV（超16万节点）
- **电商图**：SportsFit, Products, Photo, Computer（来自 Amazon 用户-商品交互）

所有特征均采用 SentenceBERT 编码文本属性，确保公平比较。

---

### 🔬 实验设置与评估协议

#### （1）两种挑战性评估场景：
| 设置 | 描述 |
|------|------|
| **Inductive Node Classification** | 将图划分为互不相交的训练/验证/测试子图（节点和边均不重叠），模拟新图到来的场景 |
| **Cold-start Node Classification** | 测试集中所有边被移除，仅保留节点特征，模拟推荐系统中新用户/物品无行为记录的情况 |

> 这两种设置严格检验模型的**泛化能力和实用性**。

#### （2）评估指标
- 主要指标：**分类准确率（Accuracy %）**
- 辅助分析：归一化互信息（NMI）、运行时间、GPU 内存占用

#### （3）训练与调参
- 增强器在 $ G_{\text{train}} $ 上训练，不在测试图上微调；
- 使用 Optuna 自动调参（学习率、权重衰减等）；
- 对比方法复现自统一代码库（Li et al., 2023）。

---

### 🧪 基线方法对比
分为三类进行比较：

| 类别 | 方法 |
|------|------|
| **先进 GNN 架构** | GCN, GAT, APPNP, GPRGNN |
| **图自监督学习（GSSL）** | DGI, GRACE, GraphMAE, GraphMAE2, VGAE |
| **图数据增强（GDA）** | GRCN, IDGL, ProGNN, SUBLIME |

> 特别注意：多数 GDA 方法依赖测试图标签进行结构优化，而 SelfAug 完全不需要。

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（见 Table 1 & 2）

#### （1）归纳式节点分类（Inductive Setting）
| 方法 | 平均准确率 | 最佳次数 |
|------|-----------|----------|
| GNNs（如 GPRGNN） | ~65–73% | - |
| GSSL 方法 | ~68–76% | - |
| GDA 方法 | 表现不稳定，部分OOM | - |
| **SelfAug（ours）** | **77.37% (DBLP)**, **71.60% (PHOTO)** 等 | **全部10项第一** ✅ |

> 在 DBLP 上领先第二名 **5.24%**，在 PHOTO 上领先 **5.73%**，优势显著。

#### （2）冷启动节点分类（Cold-start Setting）
| 方法 | 平均准确率 | 关键表现 |
|------|-----------|----------|
| GDA 方法 | 普遍下降严重，IDGL/SUBLIME 出现 OOM | 不适用 |
| GSSL 方法 | 有一定表现但受限 | 如 GRACE 在 OGBN-ARXIV 上失败 |
| **SelfAug（ours）** | **75.31% (CORA)**, **70.53% (WikiCS)** | **全面领先，无内存溢出** ✅ |

> 冷启动下仍保持高精度，证明其对**零结构信息**场景的强大适应力。

---

### 🔍 消融实验（Ablation Study）

在冷启动设置下移除关键组件的结果（Figure 3）：

| 模型变体 | 影响 |
|---------|------|
| **SelfAug（完整版）** | 0.75+ 准确率 |
| **w/o Message Regularizer (MR)** | 性能显著下降（尤其在稀疏图上） |
| **w/o Bootstrap Augmentation (BA)** | 小幅下降，但仍优于基线 |

> 结论：**MR 更重要**，它增强了邻域聚合的稳定性；**BA 提升了在弱监督下的学习能力**。

---

### ⚙️ 效率分析（Table 3）

在 CORA 数据集上的平均资源消耗：

| 方法 | 平均时间 (s) | 峰值显存 (MB) |
|------|--------------|----------------|
| GCN | 1.46 | 15.31 |
| GAT | 2.05 | 79.98 |
| ProGNN | 69.03 | 78.85 |
| SUBLIME | 92.27 | 133.03 |
| **SelfAug** | **0.90** ✅ | **15.83** ✅ |

> SelfAug 是**最快且最省内存的方法**，推理成本低于基础 GNN，远胜所有 GDA 方法。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **表示层面的数据增强优于结构层面建模**：  
   直接修正节点嵌入比重构图结构更高效、更具泛化性。

2. **自监督逆掩码机制有效捕捉潜在结构信息**：  
   即使没有标签，也能通过重建完整嵌入恢复缺失的拓扑信号。

3. **SelfAug 在归纳式与冷启动场景下表现卓越**：  
   显著超越 GNN、GSSL 和 GDA 方法，验证其作为通用增强工具的潜力。

4. **极高的效率支持实际部署**：  
   单次前向传播即可完成增强，适合在线服务与大规模图处理。

---

### ⚠️ 局限性
- 当前评估集中在**同域迁移**（within-domain）场景；
- 假设训练图具有一定的密度和清洁度（clean and dense）；
- 对**跨域图**（如从社交网迁移到生物网络）的泛化能力尚未验证。

---

### 🔮 未来工作方向
1. **跨域图数据增强**（Cross-domain GDA）  
   探索如何让增强器适应不同分布的图结构（sparsity, scale, semantics）。

2. **更鲁棒的噪声建模机制**  
   引入对抗训练或因果推理进一步过滤虚假相关性。

3. **与 LLM 结合构建统一图预训练框架**  
   利用大语言模型先验知识指导图表示增强（文中提及 LLM for Graphs 是趋势）。

4. **动态图上的实时增强策略**  
   扩展至流式图更新场景，实现实时嵌入修正。

---

> **总结一句话**：  
> SelfAug 提出了一种**轻量、无监督、可泛化的特征中心化图增强范式**，在准确性、效率和实用性上全面超越现有方法，为现实世界图学习提供了新的基础设施级解决方案。

</details>

---

### 11. [Neuron merging via inverse-activation regression for post-training compression of sigmoid neural networks](https://arxiv.org/abs/2610.02559)

**Authors**: Ao Kuniya, Jun Ohkubo  
**Category**: cs.LG  
**Published**: 2026-10-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.02559v1  

#### Abstract
As neural networks continue to grow in scale, model compression is becoming increasingly important for efficient inference under limited computational resources. Structured pruning methods remove neurons or channels that are estimated to be less important, but the removed units may still contain use...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Neuron merging via inverse-activation regression for post-training compression of sigmoid neural networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着神经网络规模不断增大，**模型压缩**在资源受限场景下的推理效率中变得至关重要。传统的 **structured pruning** 虽然能有效减少参数量和计算开销，但直接删除“不重要”的神经元可能导致有用信息丢失，尤其在高压缩率下性能下降显著。

此外，现有 **neuron merging** 方法（如 Kim et al. [21]）主要依赖于 ReLU 激活函数的代数性质（如正齐次性），难以推广到非 ReLU 网络（如 sigmoid 网络）。

本论文旨在解决以下问题：
- 如何在 **post-training 阶段** 对 sigmoid 网络进行高效压缩？
- 如何在合并神经元时保留更多信息，避免简单剪枝带来的性能损失？
- 数据（activation 响应）在 neuron merging 中扮演何种角色？

---

### 🚀 提出的新方法与创新点

作者提出了一套基于 **inverse-activation regression** 的 neuron merging 框架，适用于具有可逆激活函数（如 sigmoid）的网络。其核心思想是：

> 将多个相似神经元的输出响应通过 **反向激活函数（logit）映射回 pre-activation 空间**，然后使用 **least-squares regression** 来估计代表神经元的权重和偏置。

#### 主要创新包括：

1. **Inverse-Activation Regression (M-logit)**  
   - 利用 sigmoid 函数的可逆性（logit 函数），将目标激活值转换为 pre-activation 值。
   - 在 pre-activation 空间构建线性回归问题，求解 incoming weights 和 bias。
   - 这是一种 **data-dependent merging 方法**，能更准确地重建原始网络的行为。

2. **组合式 merging pipeline**  
   支持多种组合策略：
   - **Clustering 方法**：
     - `C-weight`：基于 weight/bias/output-weight 向量的数据无关聚类（data-free）
     - `C-activation`：基于实际输入数据生成的 activation 响应进行聚类（data-dependent）
   - **Merging 方法**：
     - `M-av`：加权平均法（data-free）
     - `M-logit`：基于 logit 回归的 least-squares 法（data-dependent）

3. **支持训练数据无关（training-data-free）版本**  
   即使没有真实训练数据，也可使用随机生成的输入（如 $ \mathcal{N}(0, I) $）来获取 activation 响应，仍能实现有效压缩。

---

### 🔍 相比现有方法的优势

| 方法 | 是否需 fine-tuning | 是否利用 activation 信息 | 是否适用于 sigmoid | 性能表现 |
|------|-------------------|----------------------------|--------------------|----------|
| Structured Pruning (e.g., L1/L2) | 否（但性能差） | 否 | 是 | 压缩后精度快速下降 |
| Kim et al. [21] Neuron Merging | 否 | 否（仅 weight） | ❌ 仅限 ReLU | 不适用 |
| **本文方法 (M-logit + C-weight)** | ❌（无需 fine-tuning） | ✅（充分利用 activation） | ✅ | 显著优于 pruning |

- **无需 fine-tuning**：所有方法均在 post-training 阶段完成，不依赖额外优化。
- **更高保真度**：相比简单剪枝，能更好地保持原始网络的功能。
- **揭示信息角色**：明确指出 **weight 信息更适合 clustering，activation 信息更适合 merging 重建**。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **MNIST**：手写数字识别任务，784 维输入，10 类输出。
- **Fashion-MNIST**：服装图像分类任务，作为更复杂场景的验证。

### ⚙️ 实验设置
- **模型架构**：全连接前馈网络 `784-1024-1024-1024-10`
- **隐藏层激活函数**：Sigmoid
- **训练配置**：
  - Optimizer: Adadelta (lr=1.0)
  - Epochs: 20
  - Batch size: 128
  - 独立训练 10 个不同随机种子的模型取平均结果

- **压缩方式**：
  - 每层等比例移除神经元，定义 **node-retention ratio** $ r_{\text{ret}} $ 衡量压缩强度。
  - 所有方法在相同压缩比例下比较。

- **评估指标**：
  - **Test accuracy**：在 10,000 测试样本上的分类准确率（均值 ± 标准差）
  - **One-time compression time**：压缩过程耗时（不包括推理时间）

### 🔁 基线方法对比
- **Random pruning**
- **L1/L2 weight-magnitude structured pruning** [12,13]
- **Vanilla neuron merging baselines**：
  - `C-weight / M-av`
  - `C-activation(data) / M-av`
  - `C-weight / M-logit(data)`
  - `C-activation(data) / M-logit(data)`
- 新增变体：
  - 使用随机输入代替真实数据：`C-activation(random)` 和 `M-logit(random)`

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（以 MNIST 为例）

| 方法 | Node-Retention Ratio = 0.2 时的 Accuracy |
|------|----------------------------------------|
| 原始未压缩模型 | ~98% |
| L2 structured pruning | ~60% |
| Random pruning | ~50% |
| `C-weight / M-av` | ~85% |
| `C-activation(data) / M-av` | ~80% |
| `C-weight / M-logit(data)` | **~92%** ✅（最佳） |
| `C-activation(data) / M-logit(data)` | ~90% |

> 💡 在高压缩率下（保留 20% 节点），所提方法仍能维持接近原始模型的性能，远超传统剪枝。

---

### 🔀 与基线方法的对比结果

- **图4（MNIST）** 显示：
  - 所有 merging 方法均显著优于 structured pruning。
  - `C-weight / M-logit(data)` 表现最优，说明 **weight-based clustering + data-driven merging** 是最佳组合。

- **图5（压缩耗时）**：
  - merging 方法的一次性压缩时间略高于 pruning，但在可接受范围内。
  - 主要开销来自 k-means clustering 和 activation 收集，而非 regression 本身。

- **图6 & 图7（Fashion-MNIST）**：
  - 趋势一致：`M-logit` 类方法在高压缩率下优势明显。
  - 随机输入版 `M-logit(random)` 依然优于 pruning，表明 **即使无真实数据也能获得良好压缩效果**。

---

### 🔍 消融实验结果

#### （1）Clustering 方法的影响
- `C-weight` > `C-activation(data)` > `C-activation(random)`
- **结论**：weight 包含更稳定、更具判别性的特征，适合用于聚类；activation 易受输入分布影响，在 random input 下性能严重下降。

#### （2）Merging 方法的影响
- `M-logit(data)` > `M-av`
- `M-logit(random)` > `pruning`
- **结论**：activation 信息对 merging 重建非常关键；即使是随机输入提供的 activation，也能帮助构造合理的 sigmoid 响应。

#### （3）关键发现总结
- ✅ **Weight information is crucial for clustering**
- ✅ **Activation information is crucial for merging**
- ✅ **Data helps — especially in the merging stage**
- ✅ **Even random data improves over pruning**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Neuron merging 比 structured pruning 更高效**  
   在不 fine-tuning 的前提下，merging 可显著减少信息损失，尤其在高压缩率下优势明显。

2. **Weight vs. Activation 的分工明确**  
   - **Weight 矩阵** 更适合用于 **clustering**（因聚合了全局连接信息）
   - **Activation 响应** 更适合用于 **merging 参数重建**（因反映真实输入-输出行为）

3. **Inverse-activation regression 有效可行**  
   利用 logit 函数将 sigmoid 输出映射回线性空间，使得 least-squares 回归成为有效的重建手段。

4. **Training-data-free merging 是可能的**  
   使用标准正态随机输入即可获得足够多样的 activation 模式，支持无需真实数据的压缩流程。

---

### ⚠️ 方法的局限性

1. **仅适用于可逆激活函数**  
   当前框架依赖 activation function 的可逆性，因此 **不适用于 ReLU、LeakyReLU 等不可逆函数**。

2. **对 activation 分布敏感**  
   若使用 random input，当 synthetic distribution 与 task distribution 差异过大时，clustering 效果会显著下降。

3. **当前实验局限于 FC 网络和小数据集**  
   尚未验证在 CNN、Transformer 或大规模网络（如 ResNet、BERT）中的有效性。

4. **未考虑输出层或跨层依赖**  
   方法聚焦于单层内部的 neuron merging，未处理跨层协同或输出保真度的进一步优化。

---

### 🔮 未来工作方向

1. **扩展至其他 invertible activation functions**  
   如 tanh、Swish（若局部可逆）、Softplus 等。

2. **设计更好的 synthetic input generation 策略**  
   探索如 **SWIM [33,34]**、Ridgelet 方法等，生成更能反映 task structure 的 synthetic inputs。

3. **结合 extreme learning machines (ELM) 或 reservoir computing**  
   在无需 backpropagation 训练的冗余网络中应用 neuron merging，提升实用性。

4. **探索 merging 与 quantization、distillation 的联合压缩方案**

5. **理论分析 merging 引入的 approximation error 及泛化边界**

---

> 📌 **一句话总结**：  
> 本文提出了一个基于 **inverse-activation regression** 的 neuron merging 框架，首次系统揭示了 **weight 信息用于 clustering、activation 信息用于 merging** 的双路径机制，并证明即使使用 **random data** 也能实现优于 structured pruning 的压缩效果，为 post-training compression 提供了新范式。

</details>

---

### 12. [Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning](https://arxiv.org/abs/2610.02687)

**Authors**: Yehya Farhat, Michael Desmond, Anastasios Kyrillidis  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.02687v1  

#### Abstract
Large language models (LLMs) are increasingly deployed in enterprise, scientific, and medical applications, where agents must incorporate domain-specific knowledge and adapt from experience. Context engineering offers a practical alternative to weight updates by improving model behavior through inst...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大型语言模型（LLMs）在部署后通常保持静态，难以从交互经验中持续学习。虽然 **context engineering**（如 prompt tuning）提供了一种无需更新权重的适应方式，但现有方法存在以下瓶颈：

- **内存扩展成本高**：如 Agentic Context Engineering (ACE) 等方法通过不断追加知识到共享上下文中，导致每轮查询需处理的 **token 数量线性增长**（O(N)），逼近模型 context window 上限。
- **上下文退化（context degradation）**：随着记忆膨胀，大量无关信息混杂，影响推理质量。
- **缺乏结构化组织**：现有 memory 系统多为扁平化文本（flat collection），无法有效捕捉知识点之间的关系。

### 🚀 提出的新方法与新思路
本文提出 **GraphMemory** —— 一种轻量级、基于图结构的外部 memory 系统，其核心思想是：

- 将可复用的知识点（如策略、修正建议）表示为图中的 **节点（node）**；
- 利用带权边（weighted edge）建模知识点间的有用关联；
- 对每个查询，仅检索相关子图（subgraph），而非加载全部记忆。

此外，作者提出了一个统一的 **context optimization 框架**，将 memory 更新视为对上下文的优化过程，并形式化分析了不同 memory 架构下的 token 复杂度。

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 ACE） | GraphMemory |
|------|------------------|-------------|
| 内存增长模式 | Append-only → O(N) 上下文长度 | Bounded retrieval → O(1) 检索内容 |
| 可解释性 | 高（自然语言 memory） | 保持高可解释性 |
| 跨 agent 迁移 | 支持 | 支持 |
| 性能稳定性 | 后期可能因过长 context 而下降 | 更稳定，持续提升 |
| Token 效率 | 低（重复加载全 memory） | 极高（仅加载 relevant 子图） |

> ✅ **核心优势**：在几乎不牺牲性能的前提下，显著降低训练和测试阶段的 token 消耗（减少 81–85%），实现更高效、可扩展的 test-time continual learning。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **FORMULA**：金融数值推理任务，用于 XBRL 标签生成。
  - 分割：500 / 300 / 200（train / val / test）
- **FINER**：金融实体识别任务，侧重数字理解。
  - 分割：1,000 / 500 / 441（train / val / test）

两个任务均属于企业级金融场景，强调精确性和领域知识积累。

### ⚙️ 实验设置
- **模型 Backbone**：
  - `Claude Haiku 4.5`（通过 LiteLLM 接入 API）
  - `Qwen3.8-27B`（本地运行，4-bit 量化）
- **训练方式**：单 epoch 在线训练，每 100 步评估一次验证集准确率。
- **memory cap**：
  - ACE 在 Claude 上受限于 80k token context window，设置了上限；
  - GraphMemory 不需要此类限制。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 最终测试集准确率（initial vs final） |
| **Training/Test Token Usage** | 构建 memory 和测试时消耗的总 token 数 |
| **Estimated Cost (USD)** | 基于 API 定价估算的金钱成本 |
| **Wall-clock Time** | 实际训练与测试耗时（active time） |
| **Retrieved Context Length** | 每次查询实际传给 Generator 的 memory token 数 |

### 🆚 基线方法对比
- **ACE (Agentic Context Engineering)**：当前最先进的动态 memory 方法，采用 Generator-Reflector-Curator 架构逐步演化 playbook。
- **Dynamic Cheatsheet**：早期 flat memory 方法（文中作为背景提及）。
- 所有方法使用相同的 backbone 和配置，仅 memory 接口不同。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & 2）

#### ✅ FORMULA + Claude Haiku 4.5
| 指标 | GraphMemory | ACE (80k cap) | 提升/节省 |
|------|-------------|---------------|----------|
| Final Test Accuracy | **0.81** | 0.85 | ↓4.0 pp |
| Training Tokens | ~20M | ~131M | ↓**84.7%** |
| Test Tokens | ~1.4M | ~18M | ↓**92.2%** |
| Estimated Cost | **$24.00** | $157.20 | ↓**84.7%** |
| Training Time | **6.4h** | 8.1h | ↓21.0% |

> ❗尽管精度略低 4 个百分点，但节省了超过 **84% 的训练 token 和成本**。

#### ✅ FORMULA + Qwen3.8-27B
| 指标 | GraphMemory | ACE | 提升/节省 |
|------|-------------|-----|----------|
| Final Test Accuracy | **0.84** | 0.83 | ↑**+1.5 pp** |
| Training Tokens | ~8M | ~42M | ↓**81.0%** |
| Estimated Cost | **$5.00** | $26.25 | ↓**81.0%** |

> ✅ **反超 ACE**：不仅节省 81% token，还实现了更高准确率！

#### ✅ FINER + Claude Haiku 4.5
| 指标 | GraphMemory | ACE (80k cap) | 提升/节省 |
|------|-------------|---------------|----------|
| Final Test Accuracy | 0.77 | **0.78** | ↓1.0 pp |
| Training Tokens | ~57M | ~327M | ↓**82.6%** |
| Estimated Cost | **$68.40** | $392.40 | ↓**82.6%** |
| Training Time | **19.0h** | 28.9h | ↓34.3% |

> 即便在更复杂任务上，仍保持竞争力并大幅降低成本。

---

### 🔍 其他重要观察
- **学习曲线差异**：
  - ACE 初期快速上升，但在接近 context cap 后趋于停滞甚至下降；
  - GraphMemory 初始较慢，但能持续改进（见 Figure 3），体现更强的长期适应能力。
- **推理延迟权衡**：
  - GraphMemory 因引入 retrieval 阶段，**每 query 推理时间增加约 1.8×**；
  - 但可通过 embedding-based retriever 或缓存优化缓解。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **memory 与 context 应解耦**：存储的知识可以无限增长，但暴露给 LLM 的 active context 应保持 bounded。
2. **structured memory 更高效**：图结构能显式建模知识间关系，支持组合式推理。
3. **bounded retrieval 实现 O(1) context 扩展**：相比 append-only 的 O(N)，极大提升了 token 效率。
4. **性能与效率可兼得**：GraphMemory 在多数设置下达到与 ACE 相当甚至更好的性能，同时节省 **81–85% 的训练 token 和成本**。

### ⚠️ 局限性
- **当前 retrieval 开销较高**：Retriever 需读取完整 index（虽小但仍增长），非严格端到端 O(1)。
- **仅验证于金融领域任务**：泛化性有待在更多任务（如科学、医疗）中验证。
- **缺少跨模型迁移实验**：未测试大模型构建的 memory 是否可用于小模型 adaptation。
- **单一运行结果**：缺乏多次随机种子的统计显著性检验。

### 🔮 未来工作方向
- 设计更高效的 retrieval 机制，如：
  - **embedding-based retriever**
  - **hybrid retrieval（向量 + 图路由）**
  - **caching frequently used subgraphs**
- 扩展至 **longer continual learning streams**（多 epoch 场景）
- 探索 **cross-agent knowledge transfer**：利用大模型提炼 memory 来增强小模型
- 引入 **learned traversal policies**（如 RL）替代固定规则遍历

---

## ✅ 总结一句话
> **GraphMemory 通过将 memory 表示为图并实施 bounded retrieval，在保持 high interpretability 和 weight-free adaptation 的同时，实现了 token-efficient、可扩展的 test-time continual learning，显著优于传统 flat memory 方法。**

</details>

---

### 13. [Fisher-Guided Submodular Data Selection for Continual Pre-Training of Large Language Models](https://arxiv.org/abs/2610.02593)

**Authors**: Zhenghao Zhao, Gaowen Liu, Zhiling Lan, Yan Yan  
**Category**: cs.LG  
**Published**: 2026-10-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.02593v1  

#### Abstract
Data selection is already a central bottleneck in large-language-model training, where web-scale corpora are noisy and token budgets are finite. In continual pre-training (CPT), it becomes a forgetting-control problem: a poorly chosen target-domain corpus can overwrite capabilities encoded in the pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Fisher-Guided Submodular Data Selection for Continual Pre-Training of Large Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在大语言模型（LLM）的**持续预训练（Continual Pre-Training, CPT）**过程中，如何从目标领域（如医学文本）中高效选择训练数据，是一个关键挑战。现有方法面临两个核心问题：
- **参数无关的选择策略**（如基于 perplexity 或 loss 的方法）无法判断训练样本是否会破坏已有能力（即导致**灾难性遗忘**）。
- **回放策略（Replay）**虽然能缓解遗忘，但需要大量通用域 token，效率低下。

因此，本文旨在解决：**如何在有限 token 预算下，既有效学习新领域知识，又最小化对原有能力的遗忘**。

---

### **提出的新方法与新思路**
作者提出了一个**Fisher引导的子模数据选择器（Fisher-guided submodular selector）**，其核心思想是：
- 利用预训练模型的**Fisher信息矩阵**（特别是对角线近似）来刻画参数空间中“已承诺”（high-Fisher）和“未承诺”（low-Fisher）的方向。
- 将每个候选样本的梯度分解为两个分量：
  - **Anchor Component**：衡量更新对 high-Fisher 方向的影响（可能导致遗忘）。
  - **Frontier Component**：衡量更新在 low-Fisher 方向的潜力（可用于安全地吸收新知识）。
- 构建一个**基于 log-determinant 的子模目标函数**，联合优化这两个信号，实现：
  - 最小化对已有能力的干扰（anchor 控制）
  - 最大化对新知识的学习（frontier 推动）
  - 子集多样性（避免冗余）

该方法通过 **Sieve-Streaming** 算法在线处理大规模数据流，在单次遍历中完成高质量子集选择。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **效率** | 在 **1B token** 上的表现已超越 Replay 方法在 **10B token** 上的结果，实现 **≥10× 的 token 效率提升**。 |
| **效果** | 同时显著提升目标领域性能并抑制遗忘，达到更优的 **(adaptation, forgetting)** Pareto 前沿。 |
| **机制解释性** | 明确将遗忘归因于 high-Fisher 方向的参数漂移，并通过几何感知的方式进行干预。 |
| **可扩展性** | 支持 LoRA、TRAK 投影、对角 Fisher 近似等工程优化，适用于十亿级参数模型的大规模流式处理。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **目标领域候选池（Target-domain candidate pool）**：
  - PMC 全文文章（38B tokens）
  - PubMed 摘要（7B tokens）
- **参考分布（Reference distribution `ppre`）用于估计 Fisher**：
  - FineWeb（400B tokens）
  - Llama3-SynE 英文部分（90B tokens）
- 所有数据均经过去重、污染过滤（如排除与 PubMedQA/BioASQ 源文档重叠的样本）。

---

### **实验设置**
- **模型**：
  - 主要模型：**TinyLlama-1.1B**
  - 扩展验证：**Llama-3.1-8B**
- **训练方式**：
  - 使用 **LoRA（r=128）** 进行适配器微调。
  - Fisher 估计和梯度计算均在 LoRA 子空间内进行。
  - 使用 **TRAK 随机投影**（d=8192）降低维度以加速计算。
- **选择预算**：
  - TinyLlama：4B tokens
  - Llama-3.1-8B：10B tokens
- **流式处理**：采用 **Sieve-Streaming** 实现单遍扫描。

---

### **评估指标**
- **适应增益（Adaptation Gain, Δ↑）**：
  - 目标任务上的平均性能提升（如 PubMedQA、BioASQ、MedMCQA 等）。
- **遗忘程度（Forgetting, Φ↓）**：
  - 通用任务上的平均性能下降（如 MMLU、HellaSwag、PIQA、LAMBADA 等）。
- **综合比较**：Δ 和 Φ 共同构成 Pareto 前沿分析。

---

### **基线方法对比**
| 基线方法 | 类型 | 描述 |
|--------|------|------|
| **Random** | 随机采样 | 不加选择地随机抽取 token |
| **Low-PPL / High-PPL** | Perplexity-based | 按最低/最高困惑度排序后取 top-k |
| **Replay (Ibrahim et al., 2024)** | 回放策略 | 混入 30% 通用域数据防止遗忘 |
| **EWC (Kirkpatrick et al., 2017)** | 正则化方法 | 使用 Fisher 作为正则项约束参数变化 |
| **Galactica / Mistral-7B** | 领域专用模型 | 用于跨模型比较 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

#### **TinyLlama-1.1B（4B tokens）**
| 方法 | Adaptation Δ↑ | Forgetting Φ↓ |
|------|----------------|---------------|
| Random | +0.2 | 3.5 |
| High-PPL | +1.3 | 5.2 |
| Replay | -2.6 | 0.9 |
| EWC | -0.4 | 1.4 |
| **Ours** | **+5.5** | **0.4** |

> ✅ **Ours 在适应性和抗遗忘方面全面领先**。

#### **Llama-3.1-8B（10B tokens）**
| 方法 | Adaptation Δ↑ | Forgetting Φ↓ |
|------|----------------|---------------|
| Random | +0.5 | 2.5 |
| High-PPL | +1.5 | 3.5 |
| Replay | -1.8 | 0.6 |
| **Ours** | **+4.4** | **0.1** |

> ✅ **Ours 不仅适应更强，且几乎不遗忘**，甚至优于专门的 Galactica 和 Mistral-7B。

---

### **与基线方法的对比结果**
- **vs. Perplexity 方法**：Ours 的 Δ 高出至少 4.2，Φ 更低。
- **vs. Replay**：尽管 Replay 在 LAMBADA 上略有优势（得益于通用数据），但在整体适应性上远逊于 Ours。
- **vs. EWC**：EWC 虽减少遗忘，但严重牺牲适应能力（Δ 为负），而 Ours 双赢。
- **vs. 领域模型**：Ours 在医学 QA 上大幅超越 Galactica，同时保持更强的通用能力。

---

### **消融实验结果（Ablation Studies）**

#### **表 2：Fisher 分解组件消融**
| 变体 | Δ↑ | Φ↓ | 说明 |
|------|-----|-----|------|
| **Ours (完整)** | +5.5 | 0.4 | — |
| No anchor (`α→∞`) | +4.2 | 5.0 | 忘记严重，说明 anchor 对保留能力至关重要 |
| No frontier (`α=0`) | +0.8 | 0.5 | 适应停滞，说明 frontier 推动新知识获取 |
| Identity Fisher (`A=I`) | +1.5 | 2.5 | 移除 Fisher 几何后性能骤降 |

> 🔍 **结论**：Fisher 引导的梯度分解不可或缺。

#### **表 3：子模选择 vs. Top-k 排序**
| 方法 | Δ↑ | Φ↓ | Pairwise Cosine ↓ | Gram log det ↑ |
|------|-----|-----|------------------|----------------|
| Top-B by `‖g_anc‖` | +1.8 | 0.6 | 0.45 | 8.5 |
| Top-B by `‖g_fr‖` | +3.5 | 4.0 | 0.22 | 11.5 |
| Top-B by `δ(x)` | +3.0 | 2.2 | 0.30 | 10.0 |
| **Ours (Sieve-Streaming)** | **+5.5** | **0.4** | **0.12** | **18.5** |

> 🔍 **结论**：子模集合选择能显著提升多样性和非冗余性，top-k 容易陷入局部重复。

#### **表 4：Token 效率对比**
| 方法 / Token 数 | 1B | 2B | 4B | 10B |
|----------------|-----|-----|-----|------|
| **Replay** (Φ↓) | 1.2 | 1.0 | 0.9 | **0.8** |
| **Replay** (Δ↑) | -3.5 | -2.9 | -2.6 | **-2.0** |
| **Ours** (Φ↓) | **0.6** | 0.5 | 0.4 | — |
| **Ours** (Δ↑) | **+3.0** | +4.5 | **+5.5** | — |

> 🚀 **Ours 在 1B token 时就已全面超越 Replay 在 10B token 的表现**，实现 **10× token 效率优势**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **灾难性遗忘是一种 Fisher 几何现象**：传统方法会导致 high-Fisher 方向参数漂移，而 low-Fisher 方向未被充分利用。
2. **Fisher 可用于前向选择而非后向正则化**：本文首次将 Fisher 从 EWC 中的“事后惩罚”角色转变为“事前选择”依据。
3. **子模目标函数能有效平衡 acquisition 与 retention**：log-det 目标天然鼓励高价值且非冗余的样本组合。
4. **极高的 token 效率**：1B token 即可超越 10B token 的 Replay 策略，具有巨大实用价值。

---

### **方法的局限性**
- 当前使用的是 **对角 Fisher 近似**，忽略了参数间的协方差结构。
- 依赖于一个短步数的 **warmup checkpoint** 来计算梯度和 Fisher，可能引入偏差。
- 实验集中在**医学领域 CPT**，其他领域（如代码、金融）尚未充分验证。
- 未考虑多模态或指令微调场景。

---

### **未来工作方向**
- 扩展到 **block-diagonal 或低秩 Fisher 模型**，捕捉更多参数间关系。
- 探索 **全参数空间适应**（而非仅 LoRA）下的应用。
- 应用于 **instruction tuning** 和 **multimodal continual learning**。
- 设计自适应 warmup 或动态调整 α 的机制。
- 结合生成式数据压缩技术进一步提升效率。

--- 

> ✅ **总结一句话**：  
> 本文提出了一种基于 Fisher 几何感知的子模数据选择方法，在持续预训练中实现了**高效、低遗忘、高适应性**的数据筛选，为 LLM 的高效领域适配提供了新的范式。

</details>

---

### 14. [Gated Slot Attention-2: Two-Sided Associative Memory Correction in Linear Attention](https://arxiv.org/abs/2610.02816)

**Authors**: Ruijie Li, Shengnan Ding, Weimin Zhang, Derick Tang, Zhanpeng Zeng, Qinsong Zeng, Ming Chen, Jiaxi Hu, Yuxuan Liang  
**Category**: cs.LG  
**Published**: 2026-10-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.02816v1  

#### Abstract
Linear attention models have emerged as efficient alternatives to standard attention, but effectively managing their fixed-size recurrent memory remains challenging. To improve memory, recent work has explored two distinct directions: delta-rule variants for precise correction of values associated w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Gated Slot Attention-2: Two-Sided Associative Memory Correction in Linear Attention

## 1. 论文的主要贡献和创新点

### 解决的问题
线性注意力（Linear Attention）模型通过将历史信息压缩到一个固定大小的循环状态中，实现了线性时间复杂度的序列建模和常数内存解码，解决了传统 Transformer 自注意力机制在长序列上计算和内存开销呈二次增长的问题。然而，如何有效地组织和更新这个压缩后的记忆状态，仍然是一个挑战。

现有方法主要分为两类：
- **基于插槽的架构**（如 Gated Slot Attention, GSA）：通过共享的潜在插槽（latent slots）分离键（key）和值（value）的记忆，提供了对关联两侧操作的自然接口，但其更新机制主要是门控累加（gated accumulation），缺乏显式的纠错能力。
- **基于增量规则的方法**（如 DeltaNet 及其变体）：通过类似 Oja 规则或 Delta 规则的机制，提供针对值（value）侧的显式关联纠错，但通常只作用于单一的键寻址关联状态。

这两类方法各有侧重，但未能结合两者的优势。

### 提出的新方法与新思路
本文提出了 **Gated Slot Attention-2 (GSA2)**，一种全新的两阶段循环注意力层，其核心创新在于**将互补的、双向的关联纠错机制引入到基于插槽的线性注意力框架中**。

具体而言，GSA2 结合了两种新的纠错规则：
- **Gated Oja Rule-2**：用于**键侧**（key-side）的纠错。它受 Oja 归一化赫布学习规则启发，利用当前的值（value）来修正其对应的键（key）的表示，从而实现“值到键”的反向纠错。
- **Gated Delta Rule-2**：用于**值侧**（value-side）的纠错。这是对已有 Delta 规则的改进，利用当前的键（key）来修正其对应的值（value）。

这两个规则通过一组**共享的潜在插槽**（shared latent slots）耦合在一起：
1.  **第一阶段**：输入的 `key` 与共享插槽 `w` 进行关联，并应用 **Gated Oja Rule-2** 对键侧进行纠错。
2.  **第二阶段**：共享插槽 `w` 作为查询，与 `value` 进行关联，并应用 **Gated Delta Rule-2** 对值侧进行纠错。

这种方法首次实现了在同一个紧凑的循环记忆架构中，对关联的“键”和“值”两端都进行独立且互补的显式纠错。

### 相比现有方法的优势
- **更强的记忆管理**：相比 GSA 的简单累加，GSA2 引入了强大的误差纠正机制，能更精确地更新和维护记忆。
- **更全面的纠错视角**：相比仅修正值侧的 Delta 规则模型，GSA2 同时修正键和值，从两个维度优化了关联质量。
- **高效的训练与推理**：继承了线性注意力的优点，支持高效的块状并行训练（chunkwise training），保持了线性时间序列建模和常数内存递归解码。
- **模块化设计**：通过解耦擦除（erase）和写入（write）控制门，提供了更精细的内存编辑能力。

---

## 2. 核心实验方法和设置

### 数据集
实验在多个基准上进行，覆盖了不同任务和上下文长度：
- **语言建模与常识推理**：
  - `WikiText`, `LAMBADA`（困惑度）
  - 零样本常识推理套件：`PIQA`, `HellaSwag`, `WinoGrande`, `ARC-e/c`, `OpenBookQA`, `SIQA`, `BoolQ`。
- **合成长上下文检索**：
  - `RULER` 基准，包含单针 (`S-NIAH`) 和多针 (`MK-NIAH`) 测试，以及 `MQ`, `MV`, `CWE`, `FWE`, `HPQA`, `SQuAD`, `VT` 等任务，测试长度从 1K 到 8K。
- **真实世界检索**：
  - `SWDE`, `SQD`, `FDA`, `TQA`, `NQ`, `DROP`，所有输入截断至 2K 上下文。
- **长上下文理解**：
  - `LongBench` 基准，包含 11 个任务，涵盖单文档问答、多文档问答、摘要、少样本学习和代码等。

### 实验设置和评估指标
- **模型规模**：约 1.3B 参数。
- **训练数据**：100B FineWeb-Edu tokens。
- **训练配置**：AdamW 优化器，峰值学习率 4e-4，权重衰减 0.1，全局批大小 0.5M tokens，训练长度 4K。
- **混合模型**：对于 Hybrid 架构，GSA2 与滑动窗口注意力（Sliding-Window Attention, SWA）交错使用，SWA 窗口为 2K。
- **评估指标**：
  - 语言建模：`ppl` (困惑度)，`acc` (准确率)。
  - 推理与检索：`acc` (准确率)。
  - 吞吐量：`Kt/s` (每秒处理的千个 token 数)。

### 基线方法对比
与以下先进模型进行了对比：
- **循环模型**：`Mamba-2`, `Gated DeltaNet`, `KDA`, `Mamba-3 (SISO/MIMO)`, `Gated DeltaNet-2 (GDN2)`, `OJA`, `OJA2`, `GSA`。
- **注意力或混合模型**：`Transformer`, `GSA2` 的混合版本等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
GSA2 在几乎所有基准上均取得了最佳或极具竞争力的性能。

- **语言建模与常识推理**（表2）：
  - 在 **1.3B 循环模型**中，GSA2 在 `WikiText` 和 `LAMBADA` 困惑度上均达到最低（`15.69` / `10.68`），并在常识推理平均分上取得最高（`54.08`）。
  - 在 **混合模型**中，GSA2 在 `LAMBADA` 准确率上表现最佳（`51.76`），常识推理平均分同样领先（`53.96`）。

- **真实世界检索**（表3）：
  - 在 **循环设置**下，GSA2 以 `31.26` 的平均准确率大幅领先于次优的 `Gated DeltaNet-2` (`29.88`)。
  - 在 **混合设置**下，GSA2 再次取得最高分 `36.70`。

- **合成长上下文检索 (RULER)**（图2, 表6）：
  - GSA2 在 `S-NIAH` 和 `MK-NIAH` 上的表现显著优于 `GSA`, `GDN2`, `OJA2` 等基线，证明了双向纠错的有效性。

- **长上下文理解 (LongBench)**（表4）：
  - GSA2 在 **循环和混合**设置下的平均分均为最高（`17.3`），表明其优势不仅限于检索任务，也适用于更复杂的长上下文理解。

- **吞吐量**（图3）：
  - GSA2 保留了循环模型近乎平坦的扩展特性，当序列长度从 2K 增加到 16K 时，吞吐量仅下降 4.2%，远优于 Transformer 的 40.2% 下降。尽管由于额外的计算，GSA2 比 `OJA2` 和 `GDN2` 分别慢约 8.6% 和 12.6%，但其性能增益远超这一小的效率损失。

### 消融实验结果（表5）
消融研究验证了 GSA2 设计的关键要素：
- **互补的双侧纠错**：同时使用 `Gated Oja Rule-2` (键侧) 和 `Gated Delta Rule-2` (值侧) 的组合效果最好。单独在两个阶段都使用同一种规则（无论是 Delta 还是 Oja）效果更差，证明了两种纠错方向的互补性。
- **解耦的擦除与写入门**：在两个阶段都解耦擦除和写入控制门，性能最强。仅在一个阶段解耦效果不一致，说明独立控制对两个记忆都至关重要。
- **其他设计选择**：移除 `1/√d` 缩放或用 `Softmax` 替换 `SiLU` 会导致性能广泛下降。默认的 128 个插槽数量在各项任务中达到了最佳平衡。

---

## 4. 关键结论和发现

### 主要发现
1.  **互补性是关键**：将基于插槽的两阶段架构（GSA）与显式的、双向的关联纠错机制相结合是有效的。GSA 提供了操作接口，而 Delta/Oja 规则提供了强大的纠错动力。
2.  **双向纠错优于单向**：同时修正键和值的关联，比只修正其中一方或重复使用同一种修正规则更为有效。这表明在记忆更新中，对“键”和“值”的独立、精细化控制是提升性能的关键。
3.  **GSA2 是一个强大的通用组件**：GSA2 不仅在专门的检索任务上表现出色，在语言建模、常识推理和综合长上下文理解任务上也持续超越强基线，证明了其作为一种高效、强大 Token Mixer 的普适价值。

### 方法的局限性
- **计算开销略高**：相比于一些更简单的线性注意力变体（如原始的 GSA 或 GDN2），GSA2 由于引入了更复杂的双阶段纠错逻辑，带来了约 8-12% 的吞吐量下降。
- **依赖于精心设计**：其优越性能依赖于 `SiLU` 激活函数、`1/√d` 缩放等特定设计，这些细节的改动可能导致性能下降。

### 未来工作方向
- **探索更高效的实现**：进一步优化 Gated Oja Rule-2 的计算，减少其相对于 Gated Delta Rule-2 的额外开销。
- **应用于更大规模模型**：在百亿甚至千亿参数的大模型上验证 GSA2 的可扩展性和性能。
- **理论分析**：深入研究 Gated Oja Rule-2 的收敛性和稳定性，为其提供更坚实的理论基础。
- **与其他架构结合**：探索将 GSA2 与不同的骨干网络（backbone）或其他注意力机制进行更深层次的融合。

</details>

---

### 15. [MOF-VERIFY: A Failure-Aware Agentic Harness for MOF Hypothesis Verification](https://arxiv.org/abs/2610.03056)

**Authors**: Donghyun Lee, Taehoon Lee, Geonhee Ahn, Jieun Kim, Jihyun Park, Suyeon Cho, Yoona Kim, Chaerim Shin, Hoi Ri Moon, Jonggeol Na, Sukho Hong, Jihwan Oh, Soo Kyung Kim  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.03056v1  

#### Abstract
Large language models are increasingly used as reasoning components in AI-driven materials Co-Scientists, yet the reliability of the resulting verification pipeline remains unclear. Metal-organic frameworks (MOFs) provide a particularly challenging setting because structures may appear under differe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MOF-VERIFY: A Failure-Aware Agentic Harness for MOF Hypothesis Verification —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Models (LLMs)** 的 AI Co-Scientist 在金属有机框架（**MOFs**）假设验证中存在显著可靠性问题。MOF 验证任务面临以下挑战：
- 同一材料可能有多个名称或结构标识符（如 CSD refcode），导致身份混淆；
- 合成条件对实验结果高度敏感，微小差异可能导致不同性质；
- 文献证据分散、冲突或依赖特定条件，难以直接比较；
- 某些假设无法仅通过文献判断，需借助 **MLIP-based 计算工具** 进行科学计算。

传统端到端推理方法（如直接 prompting 或 RAG）无法定位失败根源，容易产生错误且不可信的结论。

---

### 🚀 提出的新方法与创新思路

作者提出两个核心组件：

#### （1）诊断性基准测试：**T-MOF-1 至 T-MOF-4**
构建了一个专家设计的诊断基准，将 MOF 假设验证分解为四个独立任务家族，分别暴露不同阶段的失败模式：

| Task | 目标 | 失败类型 |
|------|------|--------|
| **T-MOF-1**: Structural Identity | 结构检索与归因 | 身份识别错误、结构不匹配 |
| **T-MOF-2**: Synthesis-condition Verification | 合成条件提取 | 条件遗漏或误读 |
| **T-MOF-3**: Evidence-Sufficiency Verification | 证据充分性判断 | 忽略矛盾、过度承诺（over-commitment） |
| **T-MOF-4**: MLIP-based Computational Verification | 计算验证能力 | 缺少可执行计算支持 |

每个任务在三种控制条件下评估：
- **Closed-book**：仅依赖 LLM 参数化知识
- **Retrieval-enabled (RAG)**：引入外部证据检索
- **Oracle-evidence**：提供“黄金”证据供模型推理

该设计形成 **3×3 诊断矩阵**，用于精确定位失败发生在：**知识缺失 → 检索失败 → 推理缺陷**

#### （2）故障感知型智能体框架：**MOF-VERIFY**
基于上述诊断结果，开发了 **MOF-VERIFY** —— 一种模块化、冻结基础 LLM 的 agentic harness，包含五个专用模块：

| 模块 | 功能 | 对应任务 |
|------|------|---------|
| **Identity Resolution (IR)** | 将材料提及映射到标准身份 | T-MOF-1 |
| **Structural Evidence (SE)** | 获取可信晶体结构（CIF 文件） | T-MOF-1 |
| **Literature Evidence (LE)** | 从指定 DOI 中提取合成记录 | T-MOF-2 |
| **Evidence Sufficiency (ES)** | 判断证据是否足够做出唯一结论 | T-MOF-3 |
| **Computational Evidence (CE)** | 执行 MLIP 计算解决数值型假设 | T-MOF-4 |
| **Verdict Routing (VR)** | 决定性路由：整合所有证据并输出最终裁决 | 综合决策 |

> ⚠️ 特点：除 LE 和 ES 使用 LLM 外，其余均为规则驱动、可追溯、防幻觉的确定性模块。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法（Direct/RAG） | MOF-VERIFY |
|------|--------------------------|------------|
| **可靠性** | 易受幻觉影响，缺乏证据溯源 | 输出带 provenance 的结构化证据包 |
| **可解释性** | 黑箱推理，难以归因错误 | 明确标注失败环节（如 identity error） |
| **适应复杂性** | 无法处理多源异构证据 | 支持文献、数据库、计算三类证据融合 |
| **抗干扰能力** | RAG 可能引入噪声降低性能 | 主动检测冲突、缺失、不可靠证据 |
| **无需微调** | —— | 完全基于冻结 LLM + 规则系统 |

---

## 2. 核心实验方法和设置

### 📚 数据集
使用自建诊断基准 **MOF-Verify-Benchmark**，共 **1,270 个样本**，分布如下：

| Task | 数量 | 控制样例占比 | 黄金标签依据 |
|------|-----|---------------|-------------|
| T-MOF-1 | 276 | 47% | CSD refcode |
| T-MOF-2 | 869 | 49% | DOI + 原文片段 |
| T-MOF-3 | 117 | 63% | DOI + 关键条件 |
| T-MOF-4 | 8 | 0% | MLIP 计算规范哈希 |

> 所有项目由 MOF 领域专家监督构建，排除可通过参数知识直接回答的简单问题。

---

### 🧪 实验设置

#### 评估模型（Backbone LLMs）
涵盖主流闭源与开源模型：
- OpenAI: `gpt-4o`, `gpt-5.4`, `gpt-5.6-terra`, `gpt-5.6-sol`
- Anthropic: `claude-opus-4.8`, `claude-opus-5`, `claude-sonnet-5`
- Google: `gemini-3.7-flash`
- Qwen: `qwen3.7-max`
- DeepSeek: `deepseek-v4-pro`

#### 评估协议
- **T-MOF-1~3**：采用三分类 Macro-F1 为主要指标（平衡类别不平衡）
- **T-MOF-4**：以能否成功运行计算并返回有效结果为准
- 引入两个增量指标分析瓶颈：
  - △access = F1_RAG − F1_Closed → 衡量检索带来的提升
  - △retrieval = F1_Oracle − F1_RAG → 衡量检索质量限制

#### 基线方法对比
- **Closed-book**：纯 LLM 推理
- **RAG**：检索增强生成
- **Oracle-evidence**：理想证据输入下的上限表现
- **MOF-VERIFY (Ours)**：本文提出的模块化框架

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 2）

| Task | 平均 F1 (Closed) | 平均 F1 (RAG) | 平均 F1 (Oracle) | **MOF-VERIFY** |
|------|------------------|----------------|--------------------|----------------|
| T-MOF-1 | 11.09 → 23.49 | 55.99 → 70.01 | 74.61 → 87.05 | **78.31** ✅ |
| T-MOF-2 | 24.47 → 39.70 | 51.76 → 57.19 | 60.96 → 67.55 | **50.00 ~ 57.03** ⚠️ |
| T-MOF-3 | 58.09 → 66.72 | 43.19 → 54.29 | 64.52 → 67.88 | **48.69 ~ 70.56** ✅↑ |
| T-MOF-4 | 0.00 ~ 16.67 | — | — | **56.97** ✅✅ |
| **平均 (T-MOF-1~3)** | **31.02** | **54.66** | **67.26** | **62.82** |

> ✅ 表示优于 RAG；⚠️ 表示仍低于 Oracle 上限

---

### 🔬 与基线方法对比结果

- **相比 Closed-book**：MOF-VERIFY 提升 **+31.80 pts Macro-F1**
- **相比 RAG**：提升 **+8.16 pts Macro-F1**
- **接近 Oracle 上限**：距离仅差 **4.44 pts**
- 在 **T-MOF-1 和 T-MOF-3** 上显著超越 RAG 和部分 Oracle 表现
- 在 **T-MOF-4** 上将平均准确率从 **5.74 → 56.97**，证明计算必须显式执行而非依赖记忆

---

### 🔍 消融实验与失败归因分析（Table 3）

通过人工与 LLM judge 分析错误原因，发现：

| 错误类型 | 占比（平均） | MOF-VERIFY 是否缓解 |
|--------|--------------|---------------------|
| **Rstruct**（结构身份错误） | 94.87% | ✅ 是（IR+SE 模块解决） |
| **Rsynth**（合成条件遗漏/误读） | 93.77% | ⚠️ 部分缓解（LE 模块仍有挑战） |
| **Revidence**（忽略证据不足） | 64.66% | ✅ 是（ES+VR 减少 over-commitment） |
| **Rover**（本应 Uncertain 却判 Yes/No） | 81.18% | ✅ 显著改善 |

> 💡 发现：即使在 Oracle 条件下，仍有大量错误源于推理缺陷，说明单纯改进检索不足以解决问题。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **MOF 验证失败是结构性的**，不能仅靠更好的 RAG 解决：
   - T-MOF-3 中 RAG 性能甚至低于 Closed-book（△access < 0），因为检索到冲突证据反而误导模型。
   
2. **三大瓶颈主导失败**：
   - 结构归因（T-MOF-1）
   - 合成信息提取精度（T-MOF-2）
   - 证据充分性判断（T-MOF-3）

3. **模块化、规则优先的设计更可靠**：
   - MOF-VERIFY 不修改 LLM，而是用规则模块“围栏”其行为，防止幻觉传播。
   - VR 模块确保只有当证据链完整时才输出 Yes/No。

4. **计算型假设必须显式执行**：
   - T-MOF-4 结果表明，LLM 无法凭记忆判断计算是否收敛，必须调用真实计算器（如 SevenNet）。

---

### ⚠️ 局限性

1. **依赖高质量本地知识库**：
   - IR、SE、LE 模块需要预加载 CSD 映射表和已验证文献副本，部署成本较高。
   
2. **T-MOF-2 性能仍未达 Oracle 水平**：
   - 精确提取合成参数（如温度、溶剂比例）仍是挑战，尤其在文本表述模糊时。

3. **扩展性受限于模块设计**：
   - 新增任务需手动添加新模块，自动化程度不如端到端训练模型。

4. **未涉及多跳推理或假设生成**：
   - 当前 focus 在“验证”，而非“发现”。

---

### 🔮 未来工作方向

1. **集成更多计算工具**：
   - 扩展 CE 模块支持 DFT、MD、吸附模拟等其他物理量计算。

2. **动态知识更新机制**：
   - 构建自动同步 CSD、文献数据库的新鲜度管道。

3. **轻量化版本适配边缘设备**：
   - 开发适用于实验室本地服务器的小型化 MOF-VERIFY Lite。

4. **向其他材料体系迁移**：
   - 将框架推广至 COFs、Zeolites、Perovskites 等类似挑战领域。

5. **结合主动学习优化数据采集**：
   - 利用 VR 的 abstention 信号指导人类专家补充关键证据。

---

> 🔗 代码与数据已开源：[https://github.com/IMMS-Ewha/MOF-Verify-Benchmark](https://github.com/IMMS-Ewha/MOF-Verify-Benchmark)

</details>

---

### 16. [EVOL: Simulator-Guided Evolutionary Expert Synthesis for Deployment-Free Learning Path Recommendation](https://arxiv.org/abs/2610.03273)

**Authors**: Geonwoo Bang, Dongho Kim, Moohong Min  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.03273v1  

#### Abstract
Reinforcement learning (RL) for learning path recommendation (LPR) faces two coupled obstacles. First, the policy must commit to a sequence of L concepts without intermediate feedback, producing a combinatorial search space that grows super-exponentially with L and provides reward only at the final ...

---

### 17. [Beaver: Elastic GPU Sharing between ML and Latency-Critical vRAN Workloads](https://arxiv.org/abs/2610.02522)

**Authors**: Yuncheng Yao, Zhenzhou Qi, Junyao Zheng, Chung-Hsuan Tung, Danyang Zhuo, Tingjun Chen  
**Category**: cs.DC  
**Published**: 2026-10-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.02522v1  

#### Abstract
Within the shared industry vision of AI-RAN, AI-and-RAN seeks to co-locate virtualized radio access network (vRAN) workloads and AI services on shared GPUs. This sharing is inherently asymmetric: vRAN workload is latency-critical, whereas the machine learning (ML) workload is a throughput-oriented, ...

---

### 18. [RailWave: Adaptive Spatial and Temporal Scheduling for Expert-Parallel Communication](https://arxiv.org/abs/2610.03415)

**Authors**: Chutian Wang, Wenhao He, Jingmin Zhu, Qingyu Yin, Heng Xu, Xiuyu Li  
**Category**: cs.DC  
**Published**: 2026-10-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.03415v1  

#### Abstract
Irregular All-to-All communication is a major bottleneck in expert-parallel Mixture-of-Experts (MoE) models. Even with fixed expert routing and placement, uneven utilization of parallel network Rails and incast can limit communication performance. We present RailWave, a phase-adaptive communication ...

---

### 19. [AI-driven Thermal-aware Data Center Capacity Planning](https://arxiv.org/abs/2610.02442)

**Authors**: Yixing Li, Mark Fenton, Matthew Kaufeler, Ka Ming Leung, Xin Ai, Zhiyu Zeng  
**Category**: cs.LG  
**Published**: 2026-10-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.02442v1  

#### Abstract
The emerging of large language models (LLMs) has posed significant challenges to the thermal management of data center. Intense GPU computation for LLMs results in localized hotspots. Moreover, spiking thermal loads during training and inference bursts make real-time cooling response more difficult ...

---

### 20. [Post-Training Quantization of Autoregressive Weather Models](https://arxiv.org/abs/2610.02511)

**Authors**: Ananyo Bhattacharya, Swastik Bhattacharya, Christiane Jablonowski  
**Category**: cs.LG  
**Published**: 2026-10-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.02511v1  

#### Abstract
Advancements in high-resolution numerical weather prediction (NWP) and data assimilation (DA) have shaped the developments in deep learning (DL) architectures emulating atmospheric dynamics. Emulators for weather forecasting exhibit forecast quality comparable to physics based models at forecast hor...

---

### 21. [Context-Tower Conversion Preserves Generation While Freezing Retains Knowledge: Low-Budget AR-to-Diffusion Conversion of MoE LLMs](https://arxiv.org/abs/2610.02657)

**Authors**: Wentao Lu, Jesse Clark, Tianyu Zhu  
**Category**: cs.LG  
**Published**: 2026-10-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.02657v1  

#### Abstract
Converting a pretrained autoregressive (AR) model to a diffusion language model (dLLM) enables parallel generation without pretraining a new model. Published conversion methods differ by roughly three orders of magnitude in training data and have not been compared under a common protocol. We compare...

---

### 22. [Page-EntroKV: Hardware-Aligned, Entropy-Weighted KV-Cache Eviction under Grouped-Query Attention](https://arxiv.org/abs/2610.03135)

**Authors**: Inbasekaran S  
**Category**: cs.LG  
**Published**: 2026-10-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.03135v1  

#### Abstract
Serving long-context autoregressive language models is constrained by the key-value (KV) cache. Most dynamic eviction methods score token importance per query head and choose tokens independently. This fits poorly with grouped-query attention (GQA), where several query heads share one physical KV bu...

---

### 23. [Frequency Is Not Sensitivity Identifying Safety-Sensitive Experts in Sparse MoE LLM](https://arxiv.org/abs/2610.02910)

**Authors**: Md Nurul Absar Siddiky, Liuwan Zhu, Yingfei Dong  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.02910v1  

#### Abstract
Suppressing a small set of routed experts can weaken the safety behavior of a sparse Mixture-of-Experts (MoE) language model without retraining. Which experts to suppress is therefore a security question, and the usual answer is activation frequency, but frequency measures use, not influence. We tes...

---

### 24. [CreateScore: Domain-Theory-Informed Bayesian Routing for LLM-Based CV Screening](https://arxiv.org/abs/2610.02972)

**Authors**: Rupsa Roy  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.02972v1  

#### Abstract
Large language models (LLMs) can support rubric-based screening of CVs, but applying a high-capability model to every candidate and criterion is costly. We present CreateScore, a domain-theory-informed Bayesian network for criterion-level LLM routing. A hand-specified directed acyclic graph with Dir...

---

### 25. [Relevant Evidence Decoding for Audio-Visual Hallucination Mitigation](https://arxiv.org/abs/2610.02976)

**Authors**: Hyunjae Ra, Aecheon Jung, Jungin Park, Sungeun Hong  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.02976v1  

#### Abstract
Audio-Visual Large Language Models (AV-LLMs) remain prone to cross-modal hallucinations, where one modality incorrectly affects predictions about another. Although contrastive decoding reduces hallucinations in vision-language models, its direct extension to AV-LLMs overlooks a key challenge: differ...

---

### 26. [Predictor-Guided Latent Space Codon Optimization for Maximizing Protein Expression](https://arxiv.org/abs/2610.03098)

**Authors**: Alberto Caron, Tianyu Cui, Dmytro S. Lituiev, Mangal Prakash, Artem Moskalev, Amina Mollaysa, Bo Zhai, Hirsh Nanda, Daniel M. Poole, Zhongyin Liu, Iman Farasat, Robert Davidson, Nikolay V. Manyakov, Tommaso Mansi, Scott Oloff, Rui Liao  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.03098v1  

#### Abstract
Codon optimization, the process of selecting synonymous codons to improve mRNA translation efficiency and protein expression, is central to therapeutic protein production and mRNA vaccines, yet it remains a hard problem. The design space is discrete and combinatorially large, precluding gradient-bas...

---

### 27. [JOVE: Joint Execution and Verification for Resource-Aware LLM Task Graphs](https://arxiv.org/abs/2610.03296)

**Authors**: Haoran Zhang, Dongjun Kim, Seohyeon Cha, Kevin S Chan, Ananthram Swami, Gustavo De Veciana, Haris Vikalo  
**Category**: cs.AI  
**Published**: 2026-10-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.03296v1  

#### Abstract
Complex reasoning queries can be decomposed into directed acyclic task graphs and distributed across heterogeneous LLMs, reducing latency through parallelism and enabling smaller models to solve complex tasks. In practice, however, the suitability of an LLM for a given subtask may be a priori unknow...

---

### 28. [Text-Centric Post-Training for Omni-Modal Reasoning](https://arxiv.org/abs/2610.02819)

**Authors**: Ziyang Cheng, Yuhao Wang, Hongcheng Liu, Qimin Wu, Jingru Fan, Chen Qian, Yanfeng Wang, Yu Wang  
**Category**: cs.CL  
**Published**: 2026-10-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.02819v1  

#### Abstract
Improving joint audio-visual reasoning in Omni Large Language Models typically incurs substantial data construction and training costs. Our diagnostics reveal multi-hop reasoning difficulties despite correct answers to all corresponding single-hop questions and suggest partial decoupling in the loca...

---

### 29. [To Jev or Not? Evaluating the Accuracy and Efficiency of Structured Decision Models for Hate-Speech Moderation](https://arxiv.org/abs/2610.03324)

**Authors**: Demetris Paschalides, George Pallis, Marios D. Dikaiakos  
**Category**: cs.CL  
**Published**: 2026-10-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.03324v1  

#### Abstract
The scale of online content makes hate-speech moderation challenging, while Large Language Models (LLMs) enable harmful material to be produced and adapted more easily. Moderation therefore requires efficient classifiers that can accommodate different definitions of hate speech. Recent structured de...

---

### 30. [Mitigating Convergence Collapse in Fixed-Target Anomaly Detectors via Kernel-Anchored Locality Regularization](https://arxiv.org/abs/2610.02345)

**Authors**: Jos\'e Lucas De Melo Costa, Fabrice Popineau, Arpad Rimmel, Bich-Li\^en Doan  
**Category**: cs.LG  
**Published**: 2026-10-05  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.02345v1  

#### Abstract
A family of tabular anomaly detectors trains a neural map toward a fixed target under squared-error loss and scores anomalies by the test-time residual; contraction matching, one-step rectified flow, and reconstruction autoencoders all fit this template. We characterize a convergence collapse: bette...

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

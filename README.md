# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-21 10:37:05 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [E8-QAT: Efficient 2-Bit Quantization-Aware Training with Implicit Lattice Codebooks](https://arxiv.org/abs/2610.27699)

**Authors**: Xinyu Lei, Hong Huang, Jiebin Zhang, Hongyu Li, Jinlong Song, Decheng Wu, Guanghua Yu, Jianchen Zhu, Sujian Li  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.27699v1  

#### Abstract
Low-bit quantization reduces the memory footprint of large language models, but preserving accuracy with efficient training remains challenging. Scalar quantization encodes weights independently, while explicit-codebook vector quantization introduces costly search during training. We propose E8-QAT,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：E8-QAT: Efficient 2-Bit Quantization-Aware Training with Implicit Lattice Codebooks**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
低比特量化（low-bit quantization）是压缩大语言模型（LLM）内存占用的关键技术，但在极低位宽（如2-bit）下，**如何在保持高精度的同时实现高效的量化感知训练（QAT）** 仍是一个挑战。现有方法面临以下瓶颈：
- **Scalar Quantization**（标量量化）独立编码每个权重，缺乏跨维度联合建模能力，限制了码字排列的灵活性。
- **Explicit-codebook Vector Quantization**（显式码本向量量化）虽能联合编码权重组，但训练时需进行最近邻搜索，带来巨大计算开销，且码本大小随维度指数增长，难以扩展。

### **提出的新方法与创新思路**
本文提出 **E8-QAT**，一种基于 **E8 lattice** 的隐式八维码本的 **2-bit 量化感知训练框架**，其核心创新如下：

- ✅ **无搜索联合量化（Search-free Joint Quantization）**  
  利用 **E8 lattice** 的数学结构，在前向传播中通过**算术投影**（rounding, parity correction, candidate comparison）将每组8个权重直接映射到最近的格点，**无需存储或搜索显式码本**，将每组计算复杂度从 $O(dK)$ 降至 $O(d)$，与码本名义大小无关。

- ✅ **精确导出机制（Exact Export via Sparse Extensions）**  
  引入“**2-bit核心编码 + 稀疏扩展记录**”策略：
  - 大多数投影落在有限核心（finite core, 2¹⁶ 个低范数点）内，用16-bit索引表示。
  - 少数超出核心的点（escapes）通过存储其位置和双倍坐标（doubled coordinates）精确保留。
  - 实验表明稀疏记录仅增加 **<0.1%** 的索引存储开销，实现**零额外量化误差的导出**。

- ✅ **隐式码本设计**  
  E8 lattice 是一个经典数学结构，而非从数据中学得，避免了码本学习开销，同时其优良的覆盖几何（covering geometry）优于传统标量网格。

### **相比现有方法的优势**
| 方面 | E8-QAT | LC-QAT / PV-Tuning |
|------|--------|---------------------|
| **训练效率** | 无码本搜索，算术投影高效 | 需码本查找或复杂更新 |
| **量化精度** | 联合编码 + 最优格结构 | 受限于标量结构或搜索近似 |
| **存储开销** | 2-bit + <0.1% 扩展 | 显式码本存储 |
| **灵活性** | 支持无限投影空间 | 受限于有限码本 |

---

## **2. 核心实验方法和设置**

### **使用的模型与数据集**
- **基础模型**：
  - `Qwen3.6-35B-A3B`（MoE 架构）
  - `Llama-3.1-8B`（稠密架构）
- **训练数据**：
  - 使用 **UltraData** 集合中的长序列文本（共约5080万 token/epoch），包含代理对话（agentic conversations）。
- **校准数据**：
  - 用于第一阶段 PTQ 的 1,680 条 16,384 长度序列。

### **实验设置**
- **量化配置**：
  - 2-bit 权重量化，分组维度 $d=8$。
  - 使用 **Hadamard 预处理** 混淆权重分布，使其更接近高斯，适配 E8 lattice 结构。
- **训练流程**：
  - 第一阶段：基于 E8 lattice 的 PTQ 初始化。
  - 第二阶段：E8-QAT 微调，使用 **identity STE**（Straight-Through Estimator）反向传播。
  - 学习率调度：warmup-stable-decay，峰值学习率 $10^{-4}$（Qwen），按权重尺度归一化（Llama）。
- **硬件**：8块GPU并行训练。

### **评估指标**
- **任务性能**（zero-shot）：
  - `ARC-C`, `HellaSwag`, `PIQA`, `WinoGrande`, `BoolQ`, `ARC-Easy`
  - `LiveCodeBench`（代码生成 pass@1）
  - `BFCL`（工具调用准确率）
  - `IFEval`（指令遵循严格准确率）
  - `GPQA`（科学问答准确率）
  - `AIME25 pass@3`
- **推理效率**：
  - 峰值内存（peak memory）
  - 最大可行 batch size
  - 解码吞吐量（decoding throughput）
- **重建质量**：
  - 层级输出失真（output distortion）

### **基线方法对比**
- **Post-Training Quantization (PTQ)**：
  - RTN, GPTQ, UniSVQ, AQLM, QuIP#, E8-PTQ
- **Quantization-Aware Training (QAT)**：
  - PV-Tuning, LC-QAT（state-of-the-art 对比基线）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Qwen3.6-35B-A3B）**
| 指标 | E8-QAT | LC-QAT | BF16（全精度） |
|------|--------|--------|---------------|
| **四任务平均分**（ARC-C, HelS, PIQA, WinG） | **73.93** | 73.12 | 73.93 |
| **LiveCodeBench pass@1** | **80.09** | 78.58 (+1.51%) | 81.42 |
| **IFEval** | **82.99** | 80.96 (+2.03%) | 85.40 |
| **六任务平均分**（+BoolQ, ARC-E） | **76.95** | 75.97 | 75.49 |
| **AIME25 pass@3** | 90.00 | 90.00 | 90.00 |

> ✅ **E8-QAT 在相同训练数据、调度和比特宽度下，全面超越 LC-QAT，并达到与 BF16 相当甚至更高的任务平均分。**

### **与基线方法的对比**
- 相比 **LC-QAT**：
  - IFEval ↑ **2.03%**
  - LiveCodeBench ↑ **1.51%**
  - 六任务平均 ↑ **0.98%**
- 相比其他 2-bit 方法（如 GPTQ, AQLM, UniSVQ）均显著领先。
- 在 `Llama-3.1-8B` 上同样取得最佳性能，平均提升 **2.99%**。

### **消融实验与关键验证**
- **第二阶段 QAT 的必要性**：
  - E8-PTQ → E8-QAT 在 LiveCodeBench 上提升 **24.36%**，证明 QAT 对恢复长文本生成能力至关重要。
- **初始化优势**：
  - E8-PTQ 的层均输出失真比 UniSVQ 低 **20.4%**，比 AQLM 低 **19.4%**，验证 E8 lattice 提供更优起点。
- **精确导出开销极低**：
  - 在 750 步训练终点，逃逸率（escape rate）为 $8.556 \times 10^{-5}$。
  - 稀疏记录仅增加 **0.086%** 的 2-bit 索引负载，总存储成本 ≈ **2.0017 bits/weight**。
- **训练效率**：
  - 在 375 步实验中，E8-QAT 比 LC-QAT 快 **4.5%**。

### **推理效率**
- 相比 BF16：
  - 峰值内存 ↓ **80%**
  - 支持 batch size ↑ **4×**
  - 在最大可行 batch 下，解码吞吐量达 BF16 的 **1.97×**

---

## **4. 关键结论和发现**

### **主要发现**
1. **E8 lattice 作为隐式码本，可实现高效且高精度的联合量化**，解决了传统 VQ-QAT 中“搜索开销 vs. 编码精度”的权衡问题。
2. **算术投影 + identity STE** 支持端到端训练，无需码本存储或搜索，显著提升训练效率。
3. **稀疏扩展机制** 以极小代价（<0.1%）实现了对所有训练选择格点的**精确导出**，避免了剪裁带来的性能损失。
4. 在多个任务和模型上，E8-QAT 不仅**匹配甚至超越全精度模型性能**，还大幅提升了推理吞吐。

### **方法的局限性**
- **评估范围有限**：目前实验集中在 Qwen 和 Llama 两个模型，尚未在更多架构（如纯 Transformer、Vision-Language 模型）上验证。
- **仅权重量化**：激活（activations）和 KV Cache 仍使用高精度表示，未实现端到端低比特推理。
- **部署依赖定制核函数**：推理性能依赖 fused decode kernels，通用性有待加强。

### **未来工作方向**
- 扩展至 **activation quantization** 和 **KV-cache quantization**，实现全模型低比特部署。
- 探索其他高效 lattice 结构（如 D₄, Aₙ）在不同维度下的适用性。
- 优化稀疏记录的压缩方式（如 root packing），进一步降低存储开销。
- 在更多应用场景（如长上下文、多模态）中验证方法泛化能力。

---

> **总结**：E8-QAT 通过引入 **E8 lattice** 作为隐式码本，实现了**无需搜索的高效联合量化**，结合**稀疏扩展精确导出**，在 2-bit 条件下达到了与全精度模型相当甚至更优的性能，同时大幅提升推理效率，为超低位宽 LLM 部署提供了新的有效路径。

</details>

---

### 2. [TeCoP: Accelerating Diffusion Large Language Model Decoding via Tentative Context Pipelining](https://arxiv.org/abs/2610.27583)

**Authors**: Jui-Chien Tsou, Hsuan-Chi Liu, Wei-Chin Wang, Yu-Syuan Xu, Chun-Yi Lee  
**Category**: cs.CL  
**Published**: 2026-10-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.27583v1  

#### Abstract
Diffusion large language models (dLLMs) predict multiple token positions per forward pass, yet block-wise decoders serialize generation and repeatedly recompute stable predictions. We attribute this inefficiency to coupling a prediction's readiness for irreversible commitment with its utility as con...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TeCoP: Accelerating Diffusion Large Language Model Decoding via Tentative Context Pipelining —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

**问题背景**：  
Diffusion Large Language Models (dLLMs) 具备并行预测多个 token 的能力，但现有的 **block-wise semi-autoregressive** 解码策略严重限制了这一潜力。这些方法将输出划分为连续块，按顺序解码，并且仅当某个位置的置信度超过高阈值时才提交（commit），导致以下问题：

- 大量已稳定但未达提交阈值的预测被反复重新计算（recomputation waste）；
- 后续 block 必须等待前序 block 完全提交后才能开始，造成严重的串行化瓶颈；
- 整体解码效率低下，接近自回归模型。

**根本原因**：  
现有方法将一个预测的两个属性耦合在一起：
- **Commitment Readiness（提交就绪性）**：是否足够可靠以永久固定；
- **Contextual Utility（上下文可用性）**：是否足够准确以作为其他位置的生成上下文。

这导致即使某些 token 已高度可信，只要未提交，就不能用于指导后续生成。

---

### ✅ 提出的新方法：Tentative Context Pipelining (TeCoP)

**核心思想**：  
提出 **Tentative Context Pipelining (TeCoP)** —— 一种无需训练的新型解码器，通过 **解耦“提交”与“用作上下文”** 来提升 dLLM 解码效率。

#### 创新机制：
1. **Block-Parallel Pipelining（块并行流水线）**
   - 维持一个滑动窗口内的多个 active block 并行运行；
   - 当前最前端 block 提交达到一定比例（activation threshold α）时，立即激活下一个 block；
   - 实现跨 block 的重叠执行，打破严格左到右依赖。

2. **Draft-and-Verify（草案-验证机制）**
   - 将置信度高于 draft threshold $ T_d $ 的未提交预测视为 **tentative draft（暂定草案）**；
   - 在下一轮中将其作为 **provisional context（临时上下文）** 输入模型；
   - 只有满足以下任一条件才正式提交：
     - **Safe-commit**：置信度 ≥ commit threshold $ T $
     - **Verify-commit**：草案在多轮间保持一致且当前置信度 ≥ verify threshold $ T_v $
     - **Fallback**：若无任何 token 达标，则提交最高置信度的位置以保证进度

> 💡 关键洞察：**稳定 ≠ 已提交；可复用 ≠ 已确认**

---

### ✅ 相比现有方法的优势

| 特性 | TeCoP | Fast-dLLM / DSB | D2F (Wang et al.) |
|------|-------|------------------|-------------------|
| 是否需要再训练 | ❌ No | ❌ No | ✅ Yes（需蒸馏） |
| 能否跨 block 复用上下文 | ✅ 是 | ❌ 否 | ✅ 是 |
| 是否利用未提交但稳定的预测 | ✅ 是 | ❌ 否 | ✅ 是 |
| 是否适用于冻结模型 | ✅ 是 | ✅ 是 | ❌ 否 |
| 性能增益 | 最高 | 中等 | 高（但需训练） |

> ✅ **TeCoP 是首个在不修改模型的前提下，实现跨 block 上下文复用的 training-free 方法**

---

## 2. 核心实验方法和设置

### 📚 数据集

- **GSM8K**：数学推理任务，5-shot exact match
- **Minerva MATH**：复杂数学问题，4-shot math_verify
- **MBPP**：Python 编程生成，3-shot pass@1

所有任务均使用 `lm-eval` 进行评估。

---

### ⚙️ 实验设置

- **模型**：
  - LLaDA 系列：LLaDA-8B-Instruct, LLaDA-1.5
  - Dream 系列：Dream-v0-Base-7B, Dream-v0-Instruct-7B
- **硬件**：单张 NVIDIA H200 GPU，batch size = 1
- **生成长度**：默认 256，部分实验扩展至 512
- **评估指标**：
  - **Accuracy (%)**：任务准确率
  - **Number of Forward Evaluations (NFE)**：每样本平均前向传播次数
  - **Throughput (TPS)**：tokens per second（仅统计答案部分）
  - **Speedup (Spd.)**：相对于原始 block-wise 解码（无缓存）的 TPS 加速比

---

### 🔁 基线方法对比

| 方法 | 描述 |
|------|------|
| **Original** | 标准 block-wise 解码，逐块推进，保守提交 |
| **Fast-dLLM** | 提交所有置信度超过阈值的位置，支持 KV caching（DualCache） |
| **DSB** | 动态滑动块调度，training-free，支持其专用缓存 |
| **TeCoP (Ours)** | 本文方法，采用 pipelined scheduling + draft-and-verify，使用 Fast-dLLM 的 prefix KV cache |

> 所有方法均测试 **有/无 KV caching** 场景，以分离缓存与算法优化的影响。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 2）

| 模型 | 方法 | Speedup (Spd.) | Accuracy Δ | NFE ↓ |
|------|------|----------------|------------|--------|
| LLaDA-8B-Instruct | TeCoP + cache | **7.87×** | -0.5 pp | ↓75% |
| LLaDA-1.5 | TeCoP + cache | **8.20×** | +0.3 pp | ↓76% |
| Dream-v0-Base-7B | TeCoP + cache | **4.91×** | -0.9 pp | ↓52% |
| Dream-v0-Instruct-7B | TeCoP + cache | **7.93×** | -2.1 pp | ↓76% |

> ✅ **最大加速达 11.77×（相对无缓存原方法）**

---

### 🔍 与基线方法对比结果

| 对比维度 | 结果 |
|---------|------|
| **TPS 表现** | TeCoP 在 **全部 12 种 model-benchmark 设置中均优于 Fast-dLLM 和 DSB**（无论是否启用缓存） |
| **NFE 降低** | NFE 最多减少 **87%**（从 256 → 33） |
| **缓存效率** | 即使只使用简单的 prefix KV cache（不如 DualCache 强），TeCoP 仍比使用 DualCache 的 Fast-dLLM 快 **1.26–1.58×** |
| **平均准确率损失** | 所有缓存设置下，平均 accuracy 下降 < **1 percentage point**，显著优于 Fast-dLLM |

> 📈 **几何平均加速（geometric-mean speedup）**：
> - TeCoP + cache: **6.65×**
> - DSB + cache: 4.72×
> - Fast-dLLM + cache: 4.50×

---

### 🔬 消融实验结果（Ablation Study）

#### （1）Pipeline Depth $ K $

| $ K $ | Speedup | Accuracy | 说明 |
|-------|--------|----------|------|
| 1（无流水线） | 3.63× | 78.2% | 相当于关闭跨 block 并行 |
| 2 | **4.26×** | 76.7% | 默认设置，性价比最优 |
| 4 | 6.30× | 74.3% | 吞吐更高，但精度下降明显 |

> ✅ 推荐 $ K=2 $：兼顾速度与稳定性

#### （2）Draft Threshold $ T_d $

| $ T_d $ | Speedup | Accuracy | 说明 |
|--------|--------|----------|------|
| ≥ T（禁用草案） | 3.94× | 77.4% | 退化为传统模式 |
| 0.85 | **4.26×** | 76.7% | 默认值，平衡性能与质量 |
| 0.6 | 6.20× | 75.2% | 准确率不稳定，风险高 |

> ✅ 设计合理：过低的 $ T_d $ 导致噪声传播，验证机制有效过滤错误草案

#### （3）验证机制诊断（Table 6）

| 输入类型 | 被接受率 | 被拒绝率 |
|--------|----------|----------|
| 真实草案（genuine draft） | 97.31% | 2.69% |
| 随机 token 替换 | 16.46% | **83.54%** |

> ✅ **验证机制不是简单复制输入**，而是基于模型一致性判断，具有强纠错能力

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **解耦“提交”与“上下文使用”是提升 dLLM 解码效率的关键突破口**；
2. **大量 token 在提交前已趋于稳定，可安全地作为临时上下文复用**；
3. **TeCoP 通过 block-parallel pipelining 和 draft-and-verify，在无需训练的情况下实现了高达 11.77× 的吞吐提升**；
4. **尽管引入推测性上下文，平均 accuracy 损失控制在 1% 以内，优于主流基线**；
5. **效率增益在更长序列（512 tokens）上依然成立，甚至更强（如 LLaDA-1.5 达 8.44×）**

---

### ⚠️ 局限性

- **尚未在开放域、长文本生成任务上验证**（如文章写作、对话延续），高熵输出可能导致草案不稳定；
- **错误传播风险存在**：一旦早期草案出错并被广泛使用，可能影响后续生成；
- **对模型置信度校准敏感**：依赖 confidence 作为 drafting 和 committing 的信号，若模型 calibration 差，效果可能下降；
- **GPU 内存开销略有增加**（约 +1 GiB on Dream models），虽可控但仍需注意

---

### 🔮 未来工作方向

1. **研究跨领域、跨任务的 draft stability 泛化能力**；
2. **探索动态调整 $ T_d $、$ T $、$ K $ 的自适应策略**；
3. **结合 soft prompting 或 latent space editing 进一步增强 tentative context 的表达力**；
4. **应用于 code generation、multi-step reasoning 等需长期依赖的任务中，评估 error propagation 控制机制**；
5. **设计轻量级 error correction module 用于在线修正 tentative context**

---

## ✅ 总结

> **TeCoP 是一项简洁而强大的 training-free 解码优化方案，首次在无需模型重训练的前提下，实现了 dLLM 中“暂定预测”的上下文复用。它打破了 block-wise 解码的串行瓶颈，释放了 diffusion 模型本应具备的并行潜力，在保持精度几乎不变的同时，带来数量级级别的吞吐提升，为高效推理提供了新范式。**

</details>

---

### 3. [Efficient Edge-aware Attention Network for Graph Generation](https://arxiv.org/abs/2610.28618)

**Authors**: Vincent Jung, Alba Carballo-Castro, Yiming Qin, Lonneke van der Plas, Pascal Frossard  
**Category**: cs.LG  
**Published**: 2026-10-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.28618v1  

#### Abstract
Graph generative models commonly use graph-specific Transformer architectures to jointly update node and edge features. However, these architectures do not directly benefit from recent hardware-aware attention implementations, limiting their scalability. The adaptation is, however, non-trivial: edge...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Edge-aware Attention Network for Graph Generation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的图生成模型（如 DiGress、DeFoG 等）广泛采用 **Graph Transformer (GT)** 架构，该架构通过联合更新节点、边和图级特征，在生成质量上表现出色。然而，这类模型存在严重的**计算与内存瓶颈**：
- 维护全连接的成对边状态（dense pairwise edge states），导致注意力计算和边表示更新均呈 $ O(N^2) $ 复杂度；
- 无法利用现代硬件优化的 **fused attention kernels**（如 FlashAttention、FlexAttention），限制了其在大规模图上的可扩展性。

### **提出的新方法与创新思路**
本文提出了 **FlexAttentionGT** —— 一种高效且表达力强的 **edge-aware attention layer**，兼容硬件融合注意力机制，同时保留原有架构的关键归纳偏置（inductive bias）。主要创新包括：

1. **从 element-wise 外积到标量点积注意力**  
   将原始 GT 中每对节点间 $ d_f $ 维的外积注意力张量 $ \mathbf{Y}_{ij,h} \in \mathbb{R}^{d_f} $ 替换为标准的标量点积注意力：
   $$
   s_{ij,h} = \frac{\mathbf{q}_{i,h} \cdot \mathbf{k}_{j,h}}{\sqrt{d_f}}
   $$
   使注意力计算可被 **FlexAttention** 编译为 fused kernel。

2. **Head-wise 结构化 FiLM 调制（Head-wise structural FiLM layer）**  
   利用 FlexAttention 提供的 `score_mod` 钩子，在预 softmax 阶段注入边和全局特征：
   $$
   s'_{ij,h} = s_{ij,h}(1 + M_{ij,h}) + B_{ij,h}
   $$
   其中 $ M, B $ 是由边特征 $ E $ 和图级特征 $ y $ 投影得到的 head-wise 标量调制项，避免写入完整的 $ O(N^2) $ 张量至显存。

3. **单次内核调用实现节点与边同步更新（Single-pass node and edge update via IN concatenation）**  
   通过将单位矩阵 $ \mathbf{I}_N $ 拼接到 value 矩阵中：
   $$
   \text{FlexAttn}(Q, K, [V \| I_N]) = [\alpha V \| \alpha]
   $$
   在一次 fused kernel 调用中同时获得聚合后的节点表示和归一化的注意力图 $ \alpha $，用于后续边更新，显著减少访存开销。

4. **轻量级残差边更新机制**  
   使用注意力图 $ \alpha $ 和全局特征进行 FiLM 调制来更新边：
   $$
   E' = \text{Linear}(\alpha \odot \phi(y)) + E
   $$
   保持边更新的依赖性，同时避免复杂的中间张量存储。

### **相比现有方法的优势**
| 方面 | FlexAttentionGT 的优势 |
|------|------------------------|
| **效率** | 支持 fused attention kernels，大幅降低峰值显存并提升吞吐量 |
| **表达能力** | 完整保留了原始 GT 的结构感知能力（edge-aware, global-aware） |
| **兼容性** | 可直接集成进主流框架（PyTorch + FlexAttention） |
| **可扩展性** | 为构建 **graph foundation models** 提供实用路径 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Planar**: 平面图，固定大小 64 节点
- **Tree**: 树形图，固定大小 64 节点
- **SBM**: 随机块模型图，节点数范围 44–187
- **SBM-Large**（附加测试）: 扩展版 SBM，节点数 401–599，用于验证大图性能

### **实验设置**
- **框架基础**：基于 **DeFoG (Qin et al., 2025a)** 的训练流程
- **模型配置**：保持所有超参数一致，仅替换 attention 层
- **硬件平台**：
  - 主要实验：V100 / GH200 GPU
  - 性能测试：A100 GPU
- **训练步数**：
  - Planar: 34,000 步
  - Tree: 16,500–26,000 步
  - SBM: 28,000 步

### **评估指标**
| 指标 | 含义 |
|------|------|
| **Validity, Uniqueness, Novelty (V.U.N.)** | 衡量生成图的有效性、多样性与新颖性 |
| **Ratio** | 衡量生成图与真实图之间统计距离相对于训练-测试差距的比例，越接近 1 越好 |
| **Inference Speed** | 推理速度（samples/sec） |
| **Training Speed** | 训练速度（samples/sec） |
| **Peak GPU Memory** | 训练过程中的最大显存占用（MB） |

### **基线方法对比**
- **GT (Original)**：原始 DeFoG 使用的 dense edge-aware Graph Transformer
- **GT (Scaled Down)**：缩小容量以匹配显存的简化版本
- **GT (Light Edge)**：移除 per-edge 注意力调制的变体，用于消融分析

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### ✅ **生成质量（Table 1）**
| Model | V.U.N.↑ | Ratio↓ |
|-------|--------|--------|
| **GT (Original)** | 99.5±1.0 (Planar), 97.0±3.7 (Tree), 85.0±5.3 (SBM) | 2.47–3.39 |
| **FlexAttentionGT (Ours)** | **100.0±0.0**, **97.0±1.9**, **89.5±5.7** | **2.21–3.70** |

> 📌 **结论**：生成质量持平甚至略有提升，尤其在 SBM 上 Validity 显著提高（+4.5%），Ratio 差异在一标准差内，说明结构保真度相当。

#### ⚡ **效率提升（Figure 1 & Table 2）**
| 指标 | 平均提升幅度 |
|------|-------------|
| **推理速度（Inference Throughput）** | **+94.4%**（最高达 116%） |
| **训练速度（Training Throughput）** | **+44.8%**（平均） |
| **峰值 GPU 显存消耗** | **↓65%**（降至原版约 1/3） |

> 示例（Planar 数据集）：
> - 原始 GT：显存 9,580 MB → FlexAttentionGT：**3,323 MB**
> - 训练速度：132.6 → **217.9 samples/sec**

#### 🔍 **消融实验结果**
| 变体 | 性能表现 | 发现 |
|------|----------|------|
| **GT (Scaled Down)** | 学习更慢，最终性能下降 | 减少模型容量不是有效替代方案 |
| **GT (Light Edge)** | **完全无法学习**（V.U.N. ≈ 0） | **per-edge attention modulation 是关键组件** |
| **FlexAttentionGT (Unnormalized)** | 收敛更慢 | 使用 softmax 归一化的注意力系数有助于稳定边特征学习 |
| **Two-pass vs Single-pass（大图）** | Two-pass 更快 | 对 >256 节点图，分两步计算 attention 输出和 map 更优（见 Table 3） |

> 📌 **特别发现**：即使在仅 50 generation steps 的低预算设置下，FlexAttentionGT 仍能稳定生成高质量图，而其他变体失败，表明其具有更强的样本效率和鲁棒性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **原始 GT 的高表达力可以不牺牲地迁移到 fused attention 框架中**  
   通过 head-wise FiLM 和 `score_mod` 设计，成功将结构信息注入 fused kernel，无需放弃 dense attention。

2. ✅ **边更新机制至关重要**  
   移除 per-edge attention modulation 导致训练崩溃，证明其是维持生成能力的核心。

3. ✅ **效率提升显著且通用**  
   在多个数据集上一致实现 **~65% 显存节省** 和 **~45–94% 速度提升**，适用于不同规模图。

4. ✅ **优化动态未改变**  
   训练损失曲线几乎重合（Figure 3），说明性能差异源于更好的归纳偏置而非优化路径变化。

### **方法的局限性**
- **仍为 $ O(N^2) $ 边更新**：虽然注意力部分已优化，但边表示更新本身仍是二次复杂度。
- **SRAM 占用随图增大增加**：IN concatenation 技巧在大图（>256 节点）时需降 block size，影响 kernel 效率。
- **依赖 FlexAttention 支持**：目前仅在支持自定义 attention 修改的系统中可用（如 PyTorch + Triton）。

### **未来工作方向**
1. **定制化 kernel 实现 block-wise edge update**  
   设计可在 tile 内完成 attention 计算与边更新的 custom kernel，避免将完整 attention map 写回 global memory。

2. **探索线性或稀疏化扩展**  
   结合 sparse graph structure 或 low-rank approximation 进一步降低复杂度。

3. **推广至其他图任务**  
   将 head-wise conditioning 与 single-pass 更新机制应用于 **graph classification、link prediction、regression** 等任务。

4. **适配更大规模图 foundation models**  
   推动该架构在 billion-node 级图上的部署，助力 **graph foundation models** 发展。

---

> 💡 **总体评价**：  
> 本论文提供了一条**实用且高效的路径**，使得原本难以扩展的 dense edge-aware Graph Transformer 能够充分利用现代硬件加速技术，在不损失生成质量的前提下实现显著的资源节约。这对于推动图生成迈向大规模、基础模型时代具有重要意义。

</details>

---

### 4. [Stratified Sparse Attention: Reducing the Sparse-to-Dense Gap at Extreme Sparsity](https://arxiv.org/abs/2610.28441)

**Authors**: Sahil Joshi, Prithvi Dixit, Aryan Katiyar, Agniva Chowdhury, Joseph E. Gonzalez, Anshumali Shrivastava, Aditya Desai, Kumar Krishna Agrawal  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.28441v1  

#### Abstract
Large Language Model inference over long contexts is increasingly bottlenecked by the cost of attending to a growing Key-Value (KV) cache. Sparse attention reduces this cost by accessing only a fraction of the cache, but existing methods face a trade-off between deterministic retrieval, which captur...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Stratified Sparse Attention: Reducing the Sparse-to-Dense Gap at Extreme Sparsity

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在长上下文场景下，**Large Language Model (LLM)** 的推理过程受到 **Key-Value (KV) cache** 规模急剧增长的限制。传统的 **dense attention** 需要对整个 KV cache 进行计算，导致内存带宽成为瓶颈。虽然 **sparse attention** 通过只访问部分 cache 来降低开销，但现有方法面临以下挑战：
- **Deterministic retrieval（如 top-k）**：仅保留高分 token，忽略了残差中的注意力质量（residual attention mass），导致精度损失。
- **Sampling-based methods**：虽能覆盖更广范围，但引入较高方差，尤其在极稀疏条件下表现不佳。
- **Hybrid methods（如 vAttention）**：结合 top-k 和均匀采样，但仍存在与 dense attention 的显著差距。

### 提出的新方法：Stratified Sparse Attention (SSA)
SSA 是一种统一的稀疏注意力框架，将 token 分为两个“层”（strata）进行处理：
1. **Deterministic stratum（确定性层）**：使用代理分数（proxy scores）**确定性地选择高置信度的 heavy hitters（top-k 类似）**。
2. **Stochastic stratum（随机层）**：对剩余 token 使用 **score-guided importance sampling**，即根据代理分数指导采样概率，并通过逆概率加权校正偏差。

该方法的关键思想是：**不仅保留高分 token，还智能探索可能携带残差注意力质量的中低分区域**。

### 相比现有方法的优势
- **更高的准确性**：相比纯 top-k、uniform sampling 或 hybrid 方法，在相同稀疏度下显著缩小与 dense attention 的差距。
- **理论保障**：提供了严格的误差界分析，证明其能以高概率逼近 dense attention 输出。
- **灵活性与通用性**：框架可退化为 top-k、top-p、hybrid top-k+uniform sampling 等特例，兼容多种 proxy score 方法（如 PQCache）。
- **实际加速能力**：配合优化的 CUDA/Triton 内核，实现了端到端的解码速度提升。

---

## 2. 核心实验方法和设置

### 数据集
实验在多个长上下文基准上进行，涵盖不同任务类型：
- **RULER**：评估模型的真实上下文长度能力（32K–128K tokens）
- **LongBench**：多语言、多任务长文本理解（10K–16K tokens）
- **LOFT**：测试检索增强生成（RAG）、SQL 执行等复杂任务（32K–128K tokens）
- **AIME 2025**：数学推理任务，生成长度达 32K tokens

### 实验设置与评估指标
- **模型范围**：从 3B 到 27B 参数的多种 LLM，包括：
  - `Llama-3.1-8B-Instruct`
  - `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.5-27B`
  - `Ministral-3-3B-Instruct`, `Ministral-3-8B-Instruct`
- **稀疏度（Sparsity）**：测试极端稀疏条件，如 **10×, 20×, 50×, 100×**（即仅访问 10%, 5%, 2%, 1% 的 KV cache）
- **评估指标**：
  - **Accuracy / Avg@16**：任务级准确率（如 RULER-HARD, AIME）
  - **Attention error (L2)**：输出向量与 dense attention 的 L2 距离
  - **Decode latency / Speedup**：解码延迟及相对于 FlashInfer 的加速比

### 基线方法对比
- **PQCache**：基于 product quantization 的近似 top-k 检索
- **vAttention (PQ)**：结合 top-k 与 uniform sampling 的混合方法
- **Oracle-top-k**：使用真实 attention scores 进行 top-k 选择（非实用，作为上界参考）
- **MagicPig**, **HashAttention**, **Double Sparsity**：其他代表性稀疏方法（见附录）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 在 RULER-HARD 上的表现（50× sparsity）
| Model | Method | 32K Acc | 128K Acc |
|-------|--------|---------|----------|
| **Ministral-3-8B** | Dense | 91.95 | 89.45 |
| | PQ | 81.83 | 62.99 |
| | vAttn(PQ) | 78.69 | 63.78 |
| | **SSA** | **84.17** | **69.25** |

- 在 **32K 上，SSA 将准确率从最强实用基线（PQ）的 81.8% 提升至 84.2%**。
- 在 **128K 上，从 63.8% 提升至 69.3%**。
- 分别恢复了 dense attention 性能的 **94% 和 85%**。

#### 在 AIME 2025 上的表现（Qwen3.5-9B, 50× sparsity）
| Method | Accuracy @16 |
|--------|----------------|
| Dense | 66.3 |
| Oracle-top-k | 48.3 |
| PQ | 36.7 |
| vAttn(PQ) | 35.8 |
| **SSA** | **59.8** |

- SSA 显著优于所有基线，在极端稀疏下仍保持接近 dense 的性能。

#### 解码速度提升（1M context, batch size 16）
| Sparsity | SSA vs FlashInfer Speedup |
|----------|----------------------------|
| 50× | **2.96×** |
| 20× | 2.17× |
| 10× | 1.55× |

- 即使在 batch size=1 时，SSA 在 1M 上下文下也能实现 **1.79×** 加速。

### 与基线方法的对比结果
- 在所有模型、数据集和稀疏度下，**SSA 均显著优于 PQCache 和 vAttention(PQ)**。
- 在轻度稀疏时，各方法差距较小；但在 **50× 及以上极端稀疏时，SSA 优势极为明显**。
- 在 Qwen3.5 系列上，SSA 几乎达到 dense performance，远超其他 sparse 方法。

### 消融实验结果（Ablation Study）
#### 不同 heavy fraction $ h $ 的影响（AIME 任务）
| Heavy Fraction $ h $ | 20× Acc | 10× Acc |
|----------------------|--------|--------|
| 0.0 (Pure sampling) | 40.2 | 42.5 |
| 0.2 | 61.0 | 63.8 |
| **0.4** | **63.5** | **64.0** |
| 0.6 | 62.7 | 62.7 |
| 0.8 | 63.8 | 63.1 |
| 1.0 (Pure top-k / PQ) | 49.4 | 53.8 |

- **纯 top-k ($ h=1 $) 和纯采样 ($ h=0 $) 表现均较差**。
- 最佳性能出现在 **$ h \in [0.2, 0.8] $** 区间，验证了 **exploitation + exploration** 结合的有效性。
- 最优配置约为 **60% 预算用于采样**，说明残差探索至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **Score-guided sampling 比 uniform sampling 更有效**：残差注意力质量分布高度不均，使用 proxy scores 指导采样能更高效地捕获残差质量。
2. **Stratification 是关键**：将预算分配给 **high-confidence heavy hitters** 和 **score-guided exploration** 两个互补目标，显著缩小了 sparse-to-dense gap。
3. **SSA 是通用框架**：可涵盖 top-k、top-p、hybrid 等方法，具有良好的扩展性和兼容性。
4. **理论与实践一致**：理论误差界与实验结果吻合，且优化内核实现了实际加速。

### 方法的局限性
- **依赖 proxy score 质量**：若 proxy scores 与真实 attention scores 差异较大，会影响采样效率。
- **采样开销**：相比纯 top-k，增加了采样和去重的计算开销，尽管通过 hierarchical sampler 优化后影响较小。
- **当前实现使用固定 budget split**：理论算法自适应分配预算，但当前 GPU 实现采用固定比例，仍有优化空间。

### 未来工作方向
- **Verifiable SSA**：引入用户可控的质量保证机制，例如提供误差上界承诺。
- **Adaptive budget allocation**：实现实时动态调整 heavy/sampling 预算分配。
- **Integration with KV compression**：与 SnapKV、KVZip 等 KV cache 压缩技术结合，进一步降低内存占用。
- **Extension to training**：探索 SSA 在长序列训练中的应用潜力。

> **开源代码**：https://github.com/skylight-org/SSA-Paper-Code

</details>

---

### 5. [Evolving Semantic Indexes for Generative Retrieval](https://arxiv.org/abs/2610.28568)

**Authors**: Jinming Nian, Jyun-Yu Jiang, Henry Tsai, Yi Fang, Zhirong Liang  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.28568v1  

#### Abstract
Generative retrieval (GR) casts search and recommendation as autoregressive generation of discrete semantic identifiers (SIDs). SID design largely determines GR quality, but its discrete, interdependent components (SID construction, collision handling, decoding, and training) cannot be optimized end...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Evolving Semantic Indexes for Generative Retrieval**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
生成式检索（**Generative Retrieval, GR**）将检索任务建模为自回归地生成文档的语义标识符（**Semantic Identifier, SID**）。然而，SID 的设计涉及多个离散且相互依赖的组件（如索引构建、冲突处理、训练目标、解码策略），这些组件难以通过梯度反向传播进行端到端优化，通常需要人工设计。

本文提出了一种自动化框架，用于在不改变下游生成器训练流程的前提下，自动优化 SID 的设计。

---

### 🚀 提出的新方法与创新思路

1. **LLM 驱动的进化搜索框架（LLM-driven Evolutionary Search）**
   - 首次将 LLM 驱动的程序级进化搜索应用于 **Generative Retrieval** 任务。
   - 利用一个本地可部署的开源大模型（**gemma-4-31b-it**, 31B 参数）作为“突变器”（mutator），对 SID 构建代码进行 `SEARCH/REPLACE` 式编辑，探索更优的 SID 设计方案。

2. **双层代理评估机制（Two-Tiered Proxy Evaluation）**
   - 引入一个极小的代理生成器（**1M 参数 GPT-2**）快速评估候选 SID 程序的性能，避免每次评估都训练完整的 1B 参数目标生成器（耗时约 5 小时）。
   - 该机制使进化搜索在计算上变得可行（评估时间降低约 15 倍）。

3. **完全开源与去中心化实现**
   - 整个框架不依赖任何专有 LLM API（如 GPT-4），所有组件均可在单张 GPU 上运行，提升了透明性和可复现性。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| **SID 设计方式** | 手工设计（如 RQVAE + TIGER） | 自动进化搜索 |
| **优化粒度** | 模型参数级（backpropagation） | 程序逻辑级（code-level mutation） |
| **评估成本** | 每次评估需训练完整生成器（高成本） | 使用小型代理模型快速反馈（低成本） |
| **系统开放性** | 可能依赖闭源 LLM | 完全基于开源模型（gemma） |

---

## 2. 核心实验方法和设置

### 📚 数据集

| 数据集 | 类型 | 文档数 | 查询数 | 特点 |
|-------|------|--------|--------|------|
| **NQ320K** | 开放域问答 | ~110K | 7,830 | 基于 Natural Questions，标准 GR 测试集 |
| **MSMARCO300K** | 信息检索 | ~320K | 808 | 基于 MS MARCO，广泛使用的 IR 基准 |
| **Multilingual E-commerce** | 多语言电商 | 数百万商品 | 25K 购买查询 | 私有数据集，query 与 item 不一定文本匹配 |

> 所有实验均固定使用 Contriever 或 mContriever 编码器提取文档/查询 embedding，仅优化 SID 构建过程。

---

### ⚙️ 实验设置

- **目标生成器（Target Generator）**：`google/gemma-3-1b-pt`（1B 参数），训练 3 轮。
- **代理生成器（Proxy Generator）**：两层 GPT-2，约 1M 非嵌入参数，训练 10 轮。
- **训练数据划分**：
  - `R_proxy`（2/3）：用于训练代理生成器。
  - `R_feedback`（1/3）：用于评估候选程序（fitness score）。
- **突变器模型**：`gemma-4-31b-it`，通过 vLLM 在单张 H200 上服务。
- **搜索算法**：基于 **AdaEvolve** 的岛式进化算法（3 islands）。
- **约束条件**：
  - SID 程序必须输出唯一行（无重复 SID）。
  - 单个程序执行时间 ≤ 2 小时。
  - SID 冲突率 > 50%（SID-only）或 > 5%（Joint）则被拒绝。

---

### 📊 评估指标

- **主指标**：
  - `Recall@1`, `Recall@5`, `Recall@10`, `Recall@100`
  - `MRR@10`
- **代理评估指标**：`Recall@100` on `R_feedback`
- **最终评估**：在保留测试集上报告目标生成器的性能。

---

### 🔁 基线方法对比

| 方法 | 描述 |
|------|------|
| **BM25** | 经典词法检索模型 |
| **Contriever** | 稠密检索基线（dense retrieval） |
| **Seed GR** | 基线 GR 模型，使用 RQVAE + TIGER 冲突处理 |
| **SID Evolution** | 本文提出的进化后 SID 程序 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

#### **NQ320K 结果**

| Method | R@1 | R@5 | R@10 | R@100 | MRR@10 |
|--------|-----|-----|------|-------|--------|
| Seed GR | 55.82 | 71.12 | 75.27 | 85.82 | 62.50 |
| **SID Evolution** | **59.28** (+3.46) | **76.44** | **81.05** | **90.57** (+4.75) | **66.74** |
| Contriever | 36.18 | 62.45 | 71.37 | 89.41 | 47.36 |

> ✅ 进化后的 SID 程序显著超越 Seed GR 和 Contriever，在 Recall@1 上提升近 **3.5%**，Recall@100 提升 **4.75%**。

#### **MSMARCO300K 结果**

| Method | R@1 | R@5 | R@10 | R@100 | MRR@10 |
|--------|-----|-----|------|-------|--------|
| Seed GR | 19.43 | 40.97 | 51.49 | 79.70 | 28.61 |
| **SID Evolution** | **20.48** (+1.05) | **41.96** | **55.01** | **82.55** (+2.85) | **30.09** |

> ✅ 在 MSMARCO 上也取得稳定增益，尤其在 Recall@10 和 MRR@10 上表现更好。

#### **多语言电商数据集（相对提升）**

| Method | Gain over BM25 (R@100) |
|--------|------------------------|
| Seed GR | +49.01 |
| **SID Evolution** | **+50.66** (+1.65) |

> ✅ 在真实电商场景中依然有效，表明方法具有跨领域泛化能力。

---

### 🔬 消融实验结果（Table 2 & Section 5.2）

| 搜索范围 | Proxy R@100 | Target R@100 | Δ vs Seed |
|----------|-------------|--------------|-----------|
| **Joint SID-Collision** | 64.39 | **90.57** | **+4.75** ✅ |
| **SID-only** | 61.47 | 87.40 | +1.58 ✅ |
| **Sequential SID-Collision** | 61.30 | 87.25 | +1.43 ✅ |
| **Collision-only** | 52.69 | 85.72 | -0.10 ❌ |
| **Objective Evolution** | 53.80 | 84.46 | -1.36 ❌ |
| **Full-Pipeline Evolution** | 62.85 | **76.12** | **-9.70** ❌ |

> 💡 **关键发现**：
> - **仅优化 SID 构造**（含联合优化冲突处理）能带来稳定正向迁移。
> - **修改生成器训练目标或完整 pipeline** 会导致代理评估与真实性能**负相关**（Spearman ρ = -0.80），即“越优化越差”。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **SID 设计是 GR 性能的关键瓶颈**
   - 仅通过改进 SID 构建程序（不改动生成器架构或训练配方），即可大幅提升 Recall@100 达 **4.75%**。
   - 表明当前 GR 的潜力尚未被充分挖掘，**语义索引的设计空间远大于已有认知**。

2. **LLM 驱动的进化搜索是有效的自动化设计工具**
   - 使用开源 LLM（gemma-31b）即可发现优于人工设计的 SID 算法。
   - 发现的改进包括：
     - **方向性残差量化**（cosine similarity 替代 Euclidean）
     - **基于查询对齐的训练损失**（alignment loss）
     - **几何感知的冲突处理**（按 embedding 相似度排序后缀）

3. **代理评估的有效性取决于优化对象**
   - ✅ 对 **SID 构造** 的优化，代理与目标系统高度一致（ρ = 0.90）。
   - ❌ 对 **训练目标或解码逻辑** 的优化，代理可能误导搜索方向。
   - ➡️ **建议：优先在索引层面进行自动化优化，谨慎扩展至训练目标。**

4. **联合优化 SID 与冲突处理效果最佳**
   - 分步优化（先 SID 后 collision）不如联合优化。
   - 因为冲突组由 SID 前缀决定，二者存在强耦合关系。

---

### ⚠️ 方法的局限性

1. **搜索空间受限于程序接口**
   - 当前只能优化 `generate_sids()` 函数内部逻辑，无法改变输入输出格式。
   - 更复杂的架构变更（如 variable-length SID）需重新定义接口。

2. **代理评估并非万能**
   - 当优化目标涉及模型规模敏感的训练动态时（如 loss design），小模型代理无法准确预测大模型行为。

3. **计算资源要求仍较高**
   - 尽管使用代理加速，整体搜索仍需数百 GPU 小时，不适合轻量级研究者。

4. **未达到 SOTA GR 性能**
   - 作者明确声明：本工作不追求 state-of-the-art，而是验证**自动化设计范式的可行性**。

---

### 🔮 未来工作方向

1. **开发更可靠的跨尺度代理评估器**
   - 探索能预测不同规模模型行为的 meta-evaluator。

2. **扩展至其他 GR 组件的协同进化**
   - 如联合进化 SID + 解码策略，但需解决信用分配问题（credit assignment）。

3. **引入多智能体协作进化**
   - 如 CORAL 或 SwarmResearch 中的多 agent 协同，提升搜索效率。

4. **应用于推荐系统等生成式排序任务**
   - 如论文中提到的 BONSAI、GLEN 等方向，拓展至 generative recommendation。

---

## 总结

> 🌟 **一句话总结**：  
> 本文首次实现了**无需闭源 LLM API** 的、基于 **LLM 驱动进化搜索** 的生成式检索系统自动优化框架，证明了通过自动化手段改进 **Semantic Index** 可显著提升 GR 性能，且其收益可通过小型代理模型可靠预测——为下一代检索系统的自动化设计开辟了新路径。

</details>

---

### 6. [LEAF: LUT-aware Efficient Attention on FPGAs](https://arxiv.org/abs/2610.27078)

**Authors**: Arianna Cox, Chang Sun, Lauri Laatu, Benedikt Maier, Aadeesh Sharma, Leo Rozanov, Wayne Luk, Maria Spiropulu, Alexander Tapper, Zhiqiang Que  
**Category**: cs.LG  
**Published**: 2026-10-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.27078v1  

#### Abstract
We introduce Lookup-table (LUT)-aware Efficient Attention on FPGAs (LEAF). LEAF is the first nanosecond-scale LUT-aware transformer for FPGAs. Deploying transformers that are large enough to perform well, small enough to fit on an FPGA, and fast enough to meet the sub-microsecond timescales required...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LEAF: LUT-aware Efficient Attention on FPGAs

## 1. 论文的主要贡献和创新点

### 解决的问题
在高能物理实验（如CERN的CMS和ATLAS）中，硬件触发系统需要在**亚微秒级延迟**内完成事件筛选，这对部署在FPGA上的机器学习模型提出了极高的**低延迟、低资源消耗**要求。尽管Transformer因其强大的表达能力被用于jet tagging等任务，但其标准Attention机制计算复杂、资源开销大，难以满足FPGA上的实时性约束。

传统LUT-aware训练方法（如HGQ-LUT）通过将权重固定的矩阵运算转换为查找表（LUT）来压缩模型并提升推理速度，但**Attention机制中的query-key乘法涉及两个激活值相乘**，无法直接应用LUT映射，因此此前的LUT-aware方法仅支持Dense和Conv层，不适用于Attention。

### 提出的新方法与创新点
本文提出 **LEAF (LUT-aware Efficient Attention on FPGAs)**，是首个实现**LUT-aware Attention**的Transformer架构，主要创新如下：

- **LUT-aware Attention设计**：识别Attention模块中存在**常量权重的矩阵-向量乘法（CMVM）操作**（如query/key/value投影层和输出前馈层），将其转化为基于LUT的表示形式。
- **混合计算架构**：保留query-key和softmax后的value乘法等**变量-变量乘法**以算术方式执行，其余部分采用LUT查表，形成“**部分LUT化 + 部分算术**”的高效混合结构。
- **端到端优化流程**：结合HGQ-LUT进行LUT-aware训练，并使用Alkaid编译器生成bit-exact RTL代码，实现从训练到部署的完整闭环。

### 相比现有方法的优势
- 在相同输入规模下，相比之前的state-of-the-art sub-microsecond Transformer [8]：
  - **准确率提升1.8个百分点**
  - **LUT使用减少2.9倍**
- 实现了**纳秒级延迟**（最低达35 ns）
- 是第一个支持LUT-aware训练的Transformer和Linformer层，已开源至[HGQ](https://github.com/arianna-cox/LEAF_transformer)工具库，推动社区发展。

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **hls4ml LHC Jet Dataset** [10]，包含最多150个粒子的喷注（jet），每个粒子有三个特征：
  - 横向动量 $p_T$
  - 伪快度 $\eta$
  - 方位角 $\phi$
- 共5类粒子来源：gluon, light quark, W boson, Z boson, top quark
- 输入序列按$p_T$降序排列，分别测试使用前8、16、32、64个粒子的情况

### 实验设置与评估指标
| 类别 | 设置 |
|------|------|
| **模型架构** | Encoder-only Transformer 和 Linformer，单头Attention |
| **量化方法** | HGQ-LUT：支持每权重/激活独立可训练bitwidth，引入EBOPs作为资源代理损失项 |
| **RTL生成** | Alkaid编译器：支持bit-exact转换，内置对HGQ的支持，进一步优化分布式算术 |
| **硬件平台** | Xilinx XCU250 FPGA，使用Vivado 2025.1综合与布局布线 |
| **评估指标** | Accuracy (%)、Latency (ns)、LUT数量 (k)、DSP数量、Initiation Interval (clk) |

### 基线方法对比
- **Transformer [8]**：基于MAC架构，使用da4ml后端
- **Transformer [7]**：基于MAC架构，使用Alkaid后端
- **Linformer [8], [7]**：不同后端下的Linformer实现
- **JEDI-linear [16]**：当前领先的FPGA高效图神经网络
- **MLP Mixer [11]**、**Deep Sets [12]**：其他轻量级替代方案

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1）

#### 最优性能点（64输入粒子）
| 模型 | Acc. (%) | Latency (ns) | LUTs (k) | DSP |
|------|----------|--------------|-----------|-----|
| **LEAF (Linformer)** | **80.4** | **35** | **28** | 0 |
| LEAF (Transformer) | 80.9 | 74 | 162 | 0 |
| JEDI-linear [16] | 81.8 | 78 | 164 | 0 |
| Linformer [8] | 79.8 | 78 | 202 | 1 |

> ✅ LEAF-Linformer在仅28k LUT和35ns延迟下达到80.4%准确率，显著优于同类Transformer实现。

#### 32输入粒子对比（关键比较点）
| 模型 | Acc. (%) | LUTs (k) | Latency (ns) |
|------|----------|-----------|--------------|
| **LEAF (Transformer)** | **78.8** | **62** | 62 |
| Previous SOTA [8] | 77.0 | 180 | 83 |
| Alkaid-only (no LEAF) [7] | 77.0 | 141 | 80 |

> 🔍 分析表明：
> - 改用Alkaid后端带来约1/3的LUT节省（180k → 141k）
> - 引入LEAF的LUT-aware Attention再带来额外2/3的压缩（141k → 62k）并提升精度1.8%

### 消融实验分析（隐含于对比中）
虽然未明确列出消融实验表格，但作者通过对以下因素的分离分析实现了等效消融：

- **后端影响**：比较da4ml vs Alkaid → 显示Alkaid本身可降低LUT用量（如32粒子Transformer从180k→141k）
- **LUT-aware训练影响**：固定Alkaid后端，比较是否启用LEAF → 显示LEAF额外带来显著压缩和精度增益
- 结论：**性能提升主要归功于LUT-aware Attention设计，而非单纯后端优化**

---

## 4. 关键结论和发现

### 主要发现
- ✅ **Attention可以部分LUT化**：尽管完整的Attention不能全表化，但其中的CMVM子模块完全可以转换为LUT，从而大幅节省资源。
- ✅ **LUT-aware训练可用于Transformer/Linformer**：首次实现了对Attention-based模型的LUT-aware训练，打破了该技术仅限于Dense/Conv层的局面。
- ✅ **LEAF显著改善Pareto前沿**：在accuracy-resource trade-off上，LEAF使Transformer首次进入与JEDI-linear相当的竞争区间，在低LUT区域表现尤为突出。
- ✅ **端到端流程可行且高效**：HGQ-LUT + Alkaid组合实现了训练与部署之间的bit-exact一致性，适合实际部署。

### 方法的局限性
- 并非所有Attention操作都能LUT化：query-key和softmax(value)等依赖双激活的操作仍需算术单元处理，限制了极致压缩潜力。
- 当前仅针对jet tagging验证，通用性有待在更多HepPhysics任务中检验（如anomaly detection）。
- 对超大规模Transformer扩展性尚未展示。

### 未来工作方向
- 将LEAF扩展至更复杂的Transformer变体（如multi-head, encoder-decoder）
- 探索更多LUT-friendly的注意力近似结构（如low-rank, sparse attention）
- 应用于其他高能物理任务，如track reconstruction、event classification
- 构建LUT-aware foundation models for HEP
- 进一步集成到CERN真实触发系统中进行在线测试

---

> 📌 总结：**LEAF是FPGA上高效Attention机制的重要突破**，它通过巧妙地识别并LUT化Attention中的常量路径，在保持高精度的同时实现了前所未有的资源压缩和超低延迟，为在资源受限环境中部署先进ML模型开辟了新道路。

</details>

---

### 7. [How Fine-Tuning Objectives Shape Layer-Wise Information in LLM Hallucination Detection](https://arxiv.org/abs/2610.27711)

**Authors**: Kaiyang Wan, Forrest Sheng Bao, Amin Ahmad, Yuxia Wang  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.27711v1  

#### Abstract
Under matched conditions, classifier-head fine-tuning (\textsc{cls}) paradigm consistently outperforms the LLM-as-Judge sequence-to-sequence (\textsc{s2s}) on binary hallucination detection. We trace this gap to the representations induced by the two objectives. Static analysis shows that \textsc{cl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：How Fine-Tuning Objectives Shape Layer-Wise Information in LLM Hallucination Detection

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文研究了在**二元幻觉检测**（binary hallucination detection）任务中，为何 **Classifier-head fine-tuning (CLS)** 范式在匹配条件下持续优于 **LLM-as-Judge sequence-to-sequence (S2S)** 范式。尽管 S2S 更自然地利用预训练语言模型头（language-modeling head），理论上应具备更强的归纳偏置，但实证结果却相反。

该研究旨在从**表示学习机制**的角度解释这一“性能反转”现象，并探索如何基于此设计更高效的 LLM 微调策略。

---

### 提出的新方法与新思路

#### （1）揭示了两种微调目标对表示空间的塑造差异
- **CLS**：通过一个随机初始化的二分类头进行监督，迫使模型学习一个**任务特定的判别方向**（discriminative direction），将隐藏状态组织成清晰的双簇几何结构（two-cluster geometry），显著提升类别可分性和线性可解码性。
- **S2S**：依赖于预训练的语言建模头生成“是/否”判断，对表示空间的重构压力较小，因此其隐藏状态更接近原始预训练骨干（pretrained backbone）。

#### （2）发现了 CLS 中的“信息窗口”（InfoWindow）动态机制
- 在训练过程中，任务相关信息并非在整个网络中均匀增长，而是集中在**中上层的狭窄层带**（如 L16–L19）快速积累，作者将其命名为 **InfoWindow**。
- InfoWindow 是信号首次形成的关键区域；而上层则在后期才同步对齐该信号。

#### （3）提出了基于 InfoWindow 的早期退出（early-exit）规则
- 利用 InfoWindow 和上层冗余性的发现，提出一种**静态截断策略**：保留底层基底（substrate）、InfoWindow 及其后的少量稳定层，移除上方冗余层。
- 这种方法无需为每个输入动态决策，而是为整个任务设定一个固定的退出深度。

---

### 相比现有方法的优势

| 方面 | 优势说明 |
|------|----------|
| **效率高** | 截断 31% 的网络层数后，准确率仍在种子噪声范围内保持不变，延迟降低 **1.4×**。 |
| **通用性强** | 所得的深度比例（如保留 ~69% 层）作为先验，在未分析的模型（如 Llama-3.2-3B、Mistral-7B-v0.3）上也能成功迁移并取得良好效果。 |
| **机制驱动** | 不同于启发式或基于置信度的 early exit，本方法由训练动态和表示分析驱动，具有更强的可解释性和理论基础。 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### 微调数据（训练集）
- **RAGTruth**：用于构建可信检索增强生成系统的幻觉语料。
- **ClaimVerify**：事实性验证数据集。
- **FactCheck-GPT**：GPT 自动生成的事实核查标注数据。
- 共计 **22,827 条样本**，混合后按 95%/5% 划分为训练/验证集。

#### 测试数据（评估集）
- **In-distribution (ID)**：
  - RAGTruthtest
  - ClaimVerifytest
  - FactCheck-GPTtest
- **Out-of-distribution (OOD)**：
  - AggreFact：摘要事实性
  - FaithBench：现代 LLM 摘要中的多样化幻觉
  - FavaBench：细粒度幻觉类型检测

所有测试集均经过去重处理，并进一步划分为 **probe-half**（用于表示分析）和 **eval-half**（用于最终性能报告）。

---

### 实验设置

#### 模型骨干
- 主要分析模型：**Qwen3-4B**, **Qwen3-8B**（均为 36 层）
- 泛化性测试模型：**Llama-3.2-3B**（28 层）、**Mistral-7B-v0.3**（32 层）

#### 微调配置
- 使用 **LoRA**（rank=32, α=64）进行参数高效微调。
- 优化器：AdamW，学习率 1e-4，warmup 3%，训练 3 个 epoch。
- Batch size：8，上下文长度 2048。
- 所有对比实验在完全相同的训练条件下进行，仅改变头部结构和损失函数。

#### 头部与损失函数定义
| 范式 | 头部 | 损失函数 |
|------|------|---------|
| **S2S** | 预训练的 `language modeling head` | 最大化生成 “Yes” 或 “No” 的概率 |
| **CLS** | 新初始化的 `linear classifier head`（2×D） | 在最后一层隐藏状态上进行二分类交叉熵 |

---

### 评估指标

- **主指标**：**macro-F1**（对正负类平等加权）
- 报告形式：三次不同随机种子（42, 43, 44）下的均值 ± 标准差
- 分别计算：
  - In-distribution 平均
  - Out-of-distribution 平均
  - Overall 平均

此外还使用以下诊断工具辅助分析：
- **Fisher score**：衡量沿判别方向的类别分离度
- **V-information & linear probing**：量化每层线性可解码的标签信息量
- **Partial Information Decomposition (PID)**：分析层间信息的独特性、冗余性与协同性
- **Bootstrap confidence intervals**：检验统计显著性

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Backbone | Method | ID Macro-F1 (%) | OOD Macro-F1 (%) | All Macro-F1 (%) | △(CLS−S2S) |
|---------|--------|------------------|-------------------|------------------|------------|
| Qwen3-4B | S2S LoRA | 77.8 ± 1.0 | 50.1 ± 2.1 | 64.0 ± 0.9 | — |
|          | **CLS LoRA** | **81.0 ± 1.0** | **70.8 ± 0.4** | **75.9 ± 0.5** | **+11.9** |
| Qwen3-8B | S2S LoRA | 78.4 ± 0.5 | 64.5 ± 3.0 | 71.4 ± 1.3 | — |
|          | **CLS LoRA** | **81.3 ± 0.7** | **70.7 ± 2.4** | **76.0 ± 0.9** | **+4.6** |
| Llama-3.2-3B | S2S LoRA | 75.4 ± 0.8 | 49.1 ± 2.8 | 62.3 ± 1.6 | — |
|            | **CLS LoRA** | **79.1 ± 0.8** | **52.3 ± 7.4** | **65.7 ± 3.3** | **+3.5** |

> ✅ **结论**：在所有骨干模型上，CLS 均显著优于 S2S，尤其在 OOD 上差距更大。

---

### 与基线方法的对比结果

- 即使采用更强的 S2S 变体（如 likelihood-ratio scoring、verbalizer ensemble、per-corpus threshold calibration、adapted LM head），CLS 仍保持领先（+2.2 至 +10.4 F1 点）。
- 将两个范式的头部都丢弃，统一用一个新的 linear probe 去拟合各层特征时，CLS 特征的表现依然远超 S2S（高出 0.13–0.15 F1）。
> ➠ 表明性能差距源于**表示本身的不同**，而非头部能力差异。

---

### 消融实验与关键发现

#### （1）表示几何分析（Section 4）
- **Fisher Score** 显示 CLS 更早且更强地提升了类别分离度。
- **Linear probing** 显示 CLS 的 label information 快速上升并在中上层饱和，而 S2S 提升有限。

#### （2）上层平台区冗余性分析（Section 5.1–5.2）
- **Bootstrap plateau test** 发现：
  - Qwen3-4B：第 **22 层**起即可达到最终层的读出性能（占总层数 **61%**）
  - Qwen3-8B：第 **23 层**起等效（**64%**）
- **PID 分析** 显示上层信息以**冗余为主**（redundant），独特信息（unique info）趋近于零，协同信息（synergy）< 0.06 nats。
- **Fusion probing** 实验显示融合多个上层并不能提升性能，甚至略有下降。

#### （3）训练动态分析（Section 5.3）
- 发现三个动态阶段：
  1. **Lower substrate (L1–L15)**：几乎无信息增益，维持预训练水平。
  2. **InfoWindow (L16–L19)**：信息快速增长，形成可解码信号。
  3. **Upper plateau (L20–L36)**：后期同步对齐 InfoWindow 输出，表现为冗余载体。

#### （4）早期退出实验（Section 6）

| 模型 | 退出位置 | 移除层数 | 准确率变化（△All） | 延迟降低 |
|------|----------|----------|--------------------|---------|
| Qwen3-4B | L25 | 31% | -0.5 ± 1.4 | **1.4×** |
| Qwen3-8B | L25 | 31% | +0.8 ± 1.1 | **1.4×** |
| Llama-3.2-3B | L20（保守） | 29% | -1.7 ± 3.8 | — |
|               | L17（激进） | 39% | +1.7 ± 1.9 | — |
| Mistral-7B-v0.3 | L23（保守） | 28% | +5.6 ± 5.9 | — |
|                   | L19（激进） | 41% | +2.6 ± 2.7 | — |

> ✅ **关键发现**：只要保留 InfoWindow 和适当缓冲，即使大幅截断上层，性能也不受影响。

---

## 4. 关键结论和发现

### 主要发现

1. **CLS 优于 S2S 的根本原因在于表示重塑能力**：
   - CLS 引导模型学习判别性方向，形成清晰的双簇结构；
   - S2S 因复用预训练 head，对表示改动小，保留更多原始语义结构。

2. **任务信号在训练中呈现“自下而上”的动态演化**：
   - 信号首先在 **InfoWindow**（中上层窄带）集中形成；
   - 上层只是后期复制和传播该信号，构成**统计上可互换的冗余平台区**。

3. **可以安全地移除上层冗余模块而不影响性能**：
   - 提出基于 InfoWindow 的**静态 early-exit 规则**，实现高效推理。

4. **该结构具有跨模型和跨任务泛化性**：
   - 在 Llama、Mistral 上观察到相同 substrate / InfoWindow / plateau 结构；
   - 在 QNLI 和 SST-2 上也重现该模式，尽管 CLS 优势缩小（因任务较简单）。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **任务范围限制** | 当前聚焦于**二元幻觉检测**，是否适用于多标签或细粒度分类尚待验证。 |
| **CLS 优势非普适** | 在 QNLI/SST-2 等简单任务上，CLS 优势极小（F1 差 < 0.03），因其预训练表示已足够好。 |
| **静态退出 vs 动态适应** | 使用固定退出点，无法像动态 early exit 那样根据不同输入调整计算量。 |
| **单 token verdict 控制变量** | 限制 S2S 仅输出一个 verdict token，排除了 rationale-based judge 的比较。 |

---

### 未来工作方向

1. **扩展至更复杂的检测任务**：
   - 如细粒度幻觉类型识别、多跳推理错误定位等。

2. **结合动态 early exit 与 InfoWindow 先验**：
   - 利用 InfoWindow 提供的结构先验，指导 adaptive exit 决策。

3. **探索其他压缩技术的整合**：
   - 将 InfoWindow 截断与知识蒸馏（distillation）、剪枝（pruning）结合，进一步提升效率。

4. **深入分析特征级信息分布**：
   - 当前关注层间分布，未来可探究**同一层内哪些神经元或注意力头承载关键信号**。

5. **设计直接塑造 InfoWindow 的新型目标函数**：
   - 开发能主动引导信息在特定层带聚集的微调目标。

---

> 📌 **总结一句话**：  
> 本文揭示了 CLS 在幻觉检测中胜出的本质是其强大的**表示重构能力**，并通过发现 **InfoWindow** 这一关键训练动态，提出了一种**机制驱动、高效可迁移的 early-exit 设计原则**，为任务专用 LLM 的轻量化提供了新路径。

</details>

---

### 8. [From Comparison to Selection: Enhancing Cross-Scale Generalization of Pointer Networks Trained with Reinforcement Learning for Capacitated Drone Delivery Routing](https://arxiv.org/abs/2610.27936)

**Authors**: Sofiene Lassoued, Devara Tertia, Stefan Lier, Andreas Schwung  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.27936v1  

#### Abstract
Unmanned aerial vehicles (UAVs) enable fast, flexible last-mile and emergency delivery, but their routing remains challenging as the number of clients varies. Classical approaches based on mixed-integer programming and heuristics can scale poorly or require problem-specific design, motivating learni...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Comparison to Selection: Enhancing Cross-Scale Generalization of Pointer Networks Trained with Reinforcement Learning for Capacitated Drone Delivery Routing*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

该论文聚焦于**神经组合优化**（Neural Combinatorial Optimization, NCO）中一个关键瓶颈：**Pointer Networks 在跨规模泛化**（cross-scale generalization）。具体而言，当训练好的模型应用于不同于训练规模的问题实例时（如从 CVRP20 迁移到 CVRP100 或反之），其性能显著下降，表现为“上采样”（upsampling）和“下采样”（downsampling）均存在性能退化。

这一问题在无人机配送等实际场景中尤为关键，因为客户数量每日变化，无法为每个规模单独训练模型。

---

### 🚀 提出的新方法与新思路

论文系统性地分析了性能退化的三个根本原因，并提出针对性改进：

#### （1）**诊断驱动的头对头比较框架**
- 首次将多种已有和新提出的改进方法统一在相同框架下进行公平比较，涵盖三个层级：
  - **Architecture-wide**: 如 attention scaling
  - **Encoder-side**: 如 GraphSAGE、Set Transformer
  - **Decoder-side**: 如图上下文聚合机制
- 明确指出：**上采样与下采样需要不同的优化策略**。

#### （2）**提出的新型图上下文聚合机制**（Proposed）
- **Weighted-average context**：通过一个 scoring network 学习节点权重，替代原始的 uniform mean pooling。
- **Set2Set context**：采用 Set2Set 模型作为 decoder 的图级 context，实现更丰富的、顺序不变的聚合。
- 二者均为**可学习的、排列不变**（permutation-invariant）的聚合方式，能缓解“图上下文稀释”（graph-context dilution）问题。

#### （3）**提出 Approach Selector**（核心创新）
- 不依赖单一最优架构，而是构建一个**选择器**（selector），根据训练规模 $a$ 和目标测试规模 $n$，动态选择表现最佳的模型。
- 选择依据是基于少量已知规模上的成本拟合一个二次函数来估计未知规模的成本：
  $$
  \hat{z}_{k,a}(n) = \alpha_{k,a}n^2 + \beta_{k,a}n + \gamma_{k,a}
  $$
- 最终选择 $\arg\min_k \hat{z}_{k,a}(n)$ 对应的方法。

---

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **通用性** | 所有方法均保持 size-agnostic，无需重新训练即可迁移 |
| **有效性** | 在最难的 CVRP100→CVRP20 下采样任务中，Set2Set 和加权平均分别提升 **0.92pp 和 0.91pp** |
| **互补性利用** | Selector 成功整合不同方法的优势，在所有9个已见和8/9未见规模对上均优于基准 |
| **无额外训练开销** | Selector 无需训练，仅需已有模型的测试成本进行拟合 |

---

## 2. 核心实验方法和设置

### 📊 数据集

- 使用合成生成的 **CVRP** 实例作为无人机配送的代理任务（proxy problem）。
- 客户坐标服从单位正方形内的均匀分布，需求为整数 $\{1,\dots,9\}$，车辆容量固定为 **Q=30**。
- 规模包括：
  - **训练与验证规模**：CVRP20, CVRP50, CVRP100
  - **未见测试规模**（用于评估泛化）：CVRP35, CVRP75, CVRP200
- 所有实例通过 `RL4CO` 框架生成，确保可复现性。

---

### ⚙️ 实验设置

| 组件 | 设置 |
|------|------|
| **框架** | RL4CO（基于 PyTorch） |
| **硬件** | NVIDIA RTX 3070 Ti GPU |
| **训练算法** | REINFORCE with rollout baseline |
| **训练方式** | 每种方法独立从零开始训练，共150轮，batch size=512 |
| **解码策略** | 训练时采样，测试时 greedy decode |
| **超参数一致性** | 所有变体共享相同超参（dim=128, heads=8, layers=3 等） |

---

### 📈 评估指标

定义两个核心指标：

#### （1）**Cross-size Optimality Gap (%)**
$$
\text{Gap}_{a \to b} = \frac{z_{a \to b} - z_{b \to b}}{z_{b \to b}} \times 100
$$
- $z_{a \to b}$：在规模 $a$ 上训练、在 $b$ 上测试的平均成本
- $z_{b \to b}$：在 $b$ 上训练并测试的“原生”模型成本
- 越小越好，反映跨规模泛化能力

#### （2）**Gap Gain (Δ)**（主要比较指标）
$$
\Delta_{a \to b} = \text{Gap}'_{a \to b} - \text{Gap}_{a \to b}
$$
- 表示某方法相对于 reference 的 gap 改善（正值表示更好）

此外报告绝对成本（cost）以衡量真实收益。

---

### 🔁 基线方法对比

| 方法 | 类型 | 来源 |
|------|------|------|
| **Reference (Attention Model)** | Baseline | Kool et al. [6] |
| **ESF (Entropy-based Scaling Factor)** | Architecture-wide | Xiao et al. [17] |
| **LSF (Learnable Scaling Factor)** | Architecture-wide | 改进自 [17] |
| **GraphSAGE Encoder** | Encoder-side | Hamilton et al. [18] |
| **Set Transformer Encoder** | Encoder-side | Lee et al. [19] |
| **Weighted Average Context** | Decoder-side | 本文提出 |
| **Set2Set Context** | Decoder-side | 本文提出 |
| **GRU Context** | Decoder-side（非排列不变）| 本文提出（消融用） |

---

## 3. 主要实验结果和性能指标

### 📌 关键性能数据（来自 Table 3 和 Table 4）

#### （1）Reference 模型的泛化瓶颈（Table 2）

| Train → Test | CVRP20→CVRP100 | CVRP100→CVRP20 |
|--------------|----------------|----------------|
| **Optimality Gap** | 5.23% | **7.77%** |

👉 发现：**下采样比上采样更难**，且差距随规模差异增大而加剧。

---

#### （2）各方法的 Gap Gain（Table 3，单位：percentage points）

| Transfer | ESF | Set Transformer | Weighted Avg. | **Set2Set** |
|---------|-----|------------------|----------------|------------|
| **20→100 (up)** | **+1.41** | +0.49 | +0.41 | +0.60 |
| **100→20 (down)** | +0.24 | -1.08 | **+0.91** | **+0.92** |

- **ESF** 在最难的上采样任务（20→100）中表现最好（+1.41pp）
- **Set2Set / Weighted Avg.** 在最难的下采样任务（100→20）中表现最佳（~+0.91pp）
- **GraphSAGE** 上采样增益最大（+3.07pp on 20→100），但损害下采样性能
- **GRU context** 因破坏排列不变性，全面劣于其他方法

---

#### （3）Approach Selector 性能（Table 4）

Selector 在以下方面全面超越 reference：

| 场景 | 最大成本降低 |
|------|-------------|
| **已见规模对**（9组） | **全部优于 reference**，最高达 **1.07%**（CVRP100→CVRP20） |
| **未见规模对**（9组） | **8/9 优于 reference**，最高达 **3.2%**（CVRP20→CVRP200） |

👉 特别是在最大外推任务 CVRP20→CVRP200 中，成本从 50.04 降至 **48.43**，提升显著。

---

#### （4）消融实验结果

- **Scoring Network 结构影响**：
  - 使用单一线性层的 weighted average 在下采样中反而恶化（-1.30pp），说明**表达能力足够**的 scoring network 是关键。
- **Permutation Invariance 必要性**：
  - GRU context（顺序依赖）在所有转移中表现差，验证了**排列不变性是设计必要条件**。
- **Same-size Quality**：
  - 多数方法（除 GraphSAGE 外）保持与 reference 相近的原生性能（Table 5），说明改进不牺牲同规模质量。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **跨规模泛化退化源于三大机制**：
   - **Node Density Shift**：规模变化导致客户密度和所需路径数变化
   - **Attention Dispersion**：softmax attention 随节点增多趋于平坦
   - **Graph-context Dilution**：mean-pooling 在大规模下趋于收敛至同一向量，丧失判别力

2. **上采样与下采样需不同策略**：
   - **上采样**（small → large）受益于 encoder 改进（如 Set Transformer）和 attention scaling（如 ESF）
   - **下采样**（large → small）最受益于 decoder 端的**高级图上下文聚合**（如 Set2Set）

3. **Approach Selector 可有效整合互补优势**：
   - 不强行寻找“万能架构”，而是根据任务动态选择最优模型
   - 在已见和未见规模上均实现一致增益，最高达 **3.2%**

---

### ⚠️ 方法的局限性

- **Selector 仅基于问题规模决策**，未考虑实例内部特征（如 demand 分布、空间聚集度）
- **Set2Set 和 Weighted Average 增益集中在下采样**，对上采样帮助有限
- 当前实验仍基于 CVRP 抽象模型，尚未集成电池、空域限制等真实无人机约束
- Selector 依赖少量已知规模的成本拟合，若分布偏移可能失效

---

### 🔮 未来工作方向（作者提出）

1. **联合建模最强组件**：
   - 将 attention scaling（如 LSF）与 learned context aggregation 结合，训练多规模混合模型。

2. **增强 Selector 智能性**：
   - 构建基于 instance features（如密度、负载率）的小型 selection network，而非仅依赖规模。

3. **扩展至完整无人机路由模型**：
   - 引入 battery-aware cost functions、air corridor constraints 和 multi-drone coordination
   - 在更大规模和真实城市布局中验证

4. **探索在线适应机制**：
   - 结合 test-time adaptation 或 meta-learning 实现动态调整

---

> **总结一句话**：  
> 本论文揭示了 Pointer Networks 跨规模泛化中的不对称挑战，提出通过 **learned graph-context aggregation** 缓解下采样退化，并首创 **approach selector** 动态选择最优模型，在已见与未见规模上均实现显著性能提升，为构建真正实用的 size-agnostic 无人机路径规划系统提供了新范式。

</details>

---

### 9. [RACE: RoPE-Aligned Attention Compression Based on Query-Key Activation Energies](https://arxiv.org/abs/2610.28385)

**Authors**: Irene Tenison, Abdulrahman Alabdulkareem, Daniel Karl I. Weidele, Nima Dehmamy, Dan Gutfreund  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.28385v1  

#### Abstract
Head-dimension compression can reduce the memory required by the key-value (KV) cache, but existing methods often reconstruct full-dimensional representations during inference, adding significant decoding FLOPs overhead. We formalize the design considerations for converting pretrained attention into...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《RACE: RoPE-Aligned Attention Compression Based on Query-Key Activation Energies》核心总结

---

## 1. 主要贡献和创新点

### 解决的问题
自回归语言模型在长上下文推理时面临 **Key-Value (KV) Cache** 内存瓶颈。传统 **head-dimension compression** 方法虽然能减少 KV Cache 存储，但在推理时需要对压缩后的表示进行 **runtime reconstruction**（如恢复 full-dimensional keys 以应用 RoPE），这引入了额外的计算开销（FLOPs），限制了效率提升。

本文旨在解决以下核心矛盾：
- 如何在不重建原始维度的情况下，直接在低维空间执行 **RoPE-based attention**？
- 如何实现真正的 **native reduced-dimensional execution**，同时兼容预训练的 RoPE 结构？

### 提出的新方法与创新思路
作者提出了 **RACE (RoPE-Aligned Attention Compression based on activation Energy)**，一种无需重建的后训练注意力压缩方法，其核心思想包括：

#### （1）将注意力压缩建模为带约束的近似矩阵乘法（Constrained AMM）
- 将 Q/K 压缩视为保持 `post-RoPE` 查询-键乘积 $ Q\tilde{K}^T $ 的近似问题。
- 强调应使用 **共享的投影映射** $ P_{QK} $ 同时作用于查询和键，以保留它们之间的交互。

#### （2）提出 RoPE 兼容性条件（RoPE Compatibility）
- 要使降维操作能在 **RoPE 之前执行**（即 native execution），必须满足一个结构性约束：**只能完整地保留或丢弃整个 rotary pair**。
- 理论证明：对于坐标选择型压缩，RoPE 兼容当且仅当所选维度构成完整的 rotary pairs。

#### （3）基于联合激活能量的选择准则（Joint Query-Key Activation Energy）
- 定义每个 rotary pair $ m $ 的能量为：
  $$
  E_m = \|Q_m\|_F + \|K_m\|_F
  $$
- 保留能量最高的 $ r_k/2 $ 个完整 rotary pairs，该准则最小化了一个可分离的、依赖校准数据的误差上界。

#### （4）端到端折叠（Folding into Weights）
- 将选定的 $ P_{QK} $ 和值子空间 $ P_v $ 直接融合进原始的 attention 投影权重中（$ W_Q^{fold}, W_K^{fold}, W_V^{fold} $）。
- 推理时完全避免运行时重建或专用 kernel，支持标准 PyTorch 操作。

### 相比现有方法的优势
| 特性 | RACE | EigenAttention | PALU | Thin Keys | RAP |
|------|------|----------------|------|-----------|-----|
| 无需重建 (Reconstruction-free) | ✅ | ❌ | ❌ | ✅ | ✅ |
| 无定制 kernel | ✅ | ✅ | ❌ | ✅ | ✅ |
| 共享 Q/K 映射 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 联合 Q/K 信号 | ✅ | ❌ | ❌ | ❌ | ❌ |
| RoPE 兼容结构 | ✅ | ❌ | ❌ | ✅ | ✅ |
| 原生低维执行 | ✅ | ❌ | ❌ | ✅ | ✅ |

> RACE 是首个同时满足 **共享映射、联合信号、RoPE 兼容、原生执行** 四大特性的方法。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准数据集 (Calibration Set)**：FineWeb-Edu（32,768 tokens），用于收集激活统计量。
- **评估任务**：
  - **语言建模**：WikiText-2（perplexity）
  - **零样本能力**：15 项任务平均准确率（如 HellaSwag, PIQA, ARC, MMLU 等）
  - **长文本理解**：LongBench（16 英文任务平均得分）

### 实验设置
- **模型家族**：
  - Llama-3.2-3B-Instruct
  - Qwen2.5-7B
  - Granite-4.1-3B
- **压缩比例**：50%, 75%, 87.5% KV Cache 保留
- **硬件平台**：单张 NVIDIA H100 80GB GPU
- **精度模式**：BF16 推理，FP32 校准
- **系统指标**：
  - 总推理 FLOPs
  - Time to First Token (TTFT)
  - 吞吐量（Prefill / Decode / Overall）
  - GPU 内存占用

### 基线方法对比
- **Reconstruction-based**:
  - **EigenAttention**: 使用低秩子空间，需重建 keys
  - **PALU**: 分解 key/value 投影，需重建
- **Native (Reconstruction-free)**:
  - **Thin Keys**: 只压缩 keys，values 不变
  - **RAP** (并发工作): 基于 Fisher 评分的 structured pruning，也实现 native 执行

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 Llama-3.2-3B-Instruct 在 50% KV Cache Retention 下为例）

| 指标 | RACE vs. 原始模型 |
|------|------------------|
| **总推理 FLOPs** | ↓ **47.8%** |
| **吞吐量 (Throughput)** | ↑ **1.79×** |
| **TTFT (Time to First Token)** | ↓ **43.2%** |
| **GPU 内存分配** | ↓ **28.1%** |
| **KV Cache 大小** | ↓ 50% |

> 在 256K 长上下文下效果更显著。

### 与基线方法的对比结果

#### （1）质量表现（Quality）
- 在 **50% KV retention** 下：
  - RACE 的 **WikiText perplexity 最低**，优于所有基线。
  - **零样本准确率** 与 PALU 相当，明显优于 RAP 和 Thin Keys。
- 在 **75% retention** 下：
  - RACE 保留了原始模型 **96.5%~98.5%** 的零样本准确率。

#### （2）效率表现（Efficiency）
- RACE 与 RAP、Thin Keys 一样实现了 **native reduced-dimensional execution**，因此 decode FLOPs 仅为 full attention 的 **0.5×**。
- 相比之下，PALU 和 EigenAttention 因需重建，decode FLOPs 分别高达 **21.92×** 和 **11.42×**。

#### （3）组合压缩能力
- RACE 可与 token-level 压缩（如 H2O, SnapKV）和量化（如 KIVI）结合：
  - 组合后总 KV Cache 可降至 **37.5% 或更低**
  - 虽然性能略有下降，但 **RACE + KIVI** 表现最稳健，说明与量化互补性强。

### 消融实验结果

#### （1）选择准则对比（Table 6, Figure 7b）
| 方法 | 50% Retention 准确率 | Perplexity |
|------|------------------------|----------|
| **Joint Q/K Energy (RACE)** | **0.54** | **16.02** |
| Q-only Energy | 0.52 | 21.01 |
| K-only Energy | 0.52 | 20.32 |
| Gradient Fisher | 0.48 | 17.80 |
| Random | 0.41 | 26.09 |

> **联合 Q/K 能量准则最优**，且优于更复杂的梯度类方法。

#### （2）层间误差分析（Figure 12）
- 使用联合能量选择的 QK 乘积近似误差 **平均最低（0.105）**，且分布均匀。

#### （3）校准数据影响（Figure 6）
- 使用任务相关数据校准可带来最多 **+5.4%** 的性能增益。
- 但 FineWeb-Edu 作为通用校准源仍表现稳健，适合未知下游任务场景。

---

## 4. 关键结论和发现

### 主要发现
1. **RoPE 兼容性是实现真正 native 压缩的关键**：必须保留完整的 rotary pairs 才能保证降维操作与 RoPE 可交换。
2. **联合 Q/K 信号优于单侧信号**：仅用 query 或 key 的统计信息会破坏交互保真度。
3. **简单的前向激活能量是最有效的选择准则**：相比基于梯度或注意力分数的方法，它在性能和成本之间达到最佳平衡。
4. **RACE 实现了质量与效率的双重领先**：
   - 质量上媲美 reconstruction-based 方法（如 PALU）
   - 效率上等同于最优 native 方法（如 RAP），但选择机制更优。

### 方法的局限性
- **固定结构压缩**：RACE 生成的是静态的压缩结构，无法像 MatryoshkaKV 等方法那样动态调整 rank。
- **依赖校准数据分布**：性能受校准数据影响，若与目标领域差异大可能次优。
- **未处理异构压缩**：各层统一压缩比例，未探索 per-layer 或 per-head 差异化策略。

### 未来工作方向
- **更灵活的压缩结构设计**：结合 input-adaptive 或 dynamic rank selection。
- **跨模态扩展**：将 RACE 应用于视觉 Transformer 或多模态模型中的 KV Cache 压缩。
- **预训练阶段协同设计**：如第 4.6 节所示，**fewer but wider heads** 架构更适合 post-training compression，未来可在预训练时考虑部署压缩需求。
- **与其他高效推理技术集成**：如与 speculative decoding、token eviction 等进一步结合。

---

> **总结**：RACE 提供了一种理论严谨、实践高效的 KV Cache 压缩方案，在保证语言模型性能的同时大幅降低推理资源消耗，是迈向 **高效长上下文推理** 的重要一步。

</details>

---

### 10. [SPEM: Structure-Aware Procedural Memory for Efficient Skill Use in LLM Agents](https://arxiv.org/abs/2610.26995)

**Authors**: Ran Zhou, Zehui Kong, Fanyu Lin, Jiaheng Liu, Zhen Zhang, Yuxiang Ren  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.26995v1  

#### Abstract
LLM-based agents increasingly rely on reusable skills. However, scaling skill libraries makes efficient and accurate skill use difficult. Full skill loading incurs high token and runtime overhead. In contrast, flat semantic retrieval treats skills as independent text entries, ignoring dependencies, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《SPEM: Structure-Aware Procedural Memory for Efficient Skill Use in LLM Agents》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在基于 **Large Language Model (LLM)** 的智能体系统中，随着可复用技能（skills）数量的增长（可达数百甚至上千），如何高效且准确地检索、组织和重用这些技能成为关键挑战。现有方法存在以下问题：

- **Full skill loading** 导致极高的 **token 开销** 和运行时延迟。
- **Flat semantic retrieval** 忽略了技能之间的依赖关系、执行顺序、状态转移和数据流约束。
- 成功的技能组合未被保存为可复用的程序性记忆（procedural memory），导致每次任务都需要重新规划。
- 效率问题日益突出：重复的模型调用造成不必要的 token 消耗和计算开销。

### **提出的新方法与创新思路**
作者提出了 **SPEM**（**Structure-Aware Procedural Memory**），一个轻量级、结构感知的程序性记忆框架，用于提升 LLM Agent 在大规模技能库下的技能使用效率。

#### **核心创新点包括：**

- **双层记忆架构（Dual-Layer Procedural Memory）**
  - **Atomic Skill Memory**：存储单个技能的语义嵌入、接口信息和执行反馈（如成功率、频率）。
  - **Skill-DAG Memory**：从成功执行轨迹中提取并存储可复用的技能组合模板（以有向无环图 DAG 形式表示），支持直接复用已有流程。

- **Skill-GNN 模块**
  - 构建任务条件下的候选技能图（task-conditioned candidate graph）。
  - 利用图神经网络进行 **state- and structure-aware skill selection**，综合考虑任务需求、当前环境状态及技能间依赖关系。

- **Trajectory-DAG Extraction**
  - 将成功的执行轨迹压缩为紧凑的 Skill-DAG 模板，并写回 Skill-DAG Memory。
  - 支持程序性记忆随时间积累和演化。

- **冷启动支持（Cold-Start Deployment）**
  - 在缺乏历史轨迹时，通过技能库元数据构建弱关系图实现初始化，逐步过渡到基于经验的选择。

### **相比现有方法的优势**
| 维度 | 现有方法局限 | SPEM 的优势 |
|------|---------------|-------------|
| **效率** | 加载全部技能或大量检索结果导致高 token 和 runtime 成本 | 只加载少量原子技能或完整 Skill-DAG 模板，显著降低上下文负担 |
| **结构感知** | 多数方法仅基于语义相似性检索，忽略依赖与顺序 | 显式建模技能间的依赖、状态转换和数据流 |
| **可复用性** | 成功路径不保留，需重复推理 | 自动提取 Skill-DAG 模板供后续任务复用 |
| **适应性** | 静态图结构难以应对新任务 | Skill-GNN 具备归纳能力，可在动态构造的图上泛化 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ALFWorld**  
  - 文本型具身交互环境，模拟家庭场景中的多步任务（如“把苹果放进微波炉”）。
  - 测试长视野依赖、状态转移和跨任务技能复用能力。
  - 包含 140 个开发集 episode，最大步数限制为 30。

- **SkillsBench**  
  - 技术类任务基准，涵盖代码生成、API 调用、文件操作等。
  - 使用大型技能库（200 个技能），测试大规模技能检索与执行能力。
  - 采用 Docker 环境 + Harbor 执行器 + 官方验证器，最多两次尝试每项任务。

### **实验设置与评估指标**

#### **主干模型（Backbone LLMs）**
- `MiniMax-M2.7`：较弱模型，测试方法鲁棒性
- `GPT-5.2-Codex`：较强模型，测试效率差异

#### **评估指标**

| 数据集 | 指标 | 含义 |
|--------|------|------|
| **ALFWorld** | Success (↑) | 任务成功率 |
| | Steps (↓) | 平均交互步数 |
| | Tokens (↓) | 平均 token 消耗 |
| | Runtime (↓) | Agent 执行时间（秒） |
| **SkillsBench** | Reward R (↑) | 官方验证器打分 |
| | Input Tokens T (↓) | 输入 token 数量 |
| | Runtime S (↓) | Agent 执行时间（不含环境搭建） |

> 注：所有方法共享相同的技能库、agent 接口、执行环境和验证协议，确保公平比较。

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Vanilla Skills** | 加载全部技能描述作为上下文 |
| **Vector Skills** | 基于嵌入相似性的 top-k 检索（flat retrieval） |
| **GoS (Graph of Skills)** | 基于预构建技能依赖图的静态结构检索 |
| **SkillDAG** | 当前 SOTA，自演化的类型化技能图，支持依赖感知选择 |
| **SPEM (Ours)** | 提出的方法：双层记忆 + Skill-GNN + DAG 提取 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

#### **ALFWorld 结果**
| 方法 | Success (%) | Steps | Tokens | Runtime (s) |
|------|-------------|-------|--------|-----------|
| **MiniMax-M2.7** | | | | |
| Vanilla Skills | 59.3 | 21.24 | 152,242.1 | 55.95 |
| Vector Skills | 63.9 | 18.36 | 55,816.6 | 55.71 |
| GoS | 68.6 | 20.26 | 62,033.9 | 54.22 |
| SkillDAG | 63.3 | 16.37 | 296,512.4 | 137.38 |
| **SPEM (Ours)** | **70.7** | 18.29 | **45,785.7** | **51.56** |
| **GPT-5.2-Codex** | | | | |
| Vanilla Skills | 84.9 | 14.65 | 99,023.9 | 84.57 |
| Vector Skills | 83.7 | 16.18 | 47,357.3 | 80.43 |
| GoS | 85.3 | 16.46 | 48,805.6 | 71.23 |
| SkillDAG | 85.7 | 11.73 | 105,449.9 | 169.10 |
| **SPEM (Ours)** | **87.1** | 15.07 | **47,409.1** | **66.01** |

> ✅ **SPEM 在两个模型下均取得最高成功率，同时 token 和 runtime 最低。**

#### **SkillsBench 结果**
| 方法 | Reward (%) | Input Tokens | Runtime (s) |
|------|------------|-------------|------------|
| **MiniMax-M2.7** | | | |
| Vanilla Skills | 21.61 | 798K | 469.3 |
| Vector Skills | 13.07 | 722K | 439.5 |
| GoS | 23.50 | 735K | 399.4 |
| SkillDAG | 27.09 | 979K | 476.1 |
| **SPEM (Ours)** | **25.14** | **721K** | **360.5** |
| **GPT-5.2-Codex** | | | |
| Vanilla Skills | 28.87 | 2,785K | 686.8 |
| Vector Skills | 23.44 | 1,047K | 763.4 |
| GoS | 37.50 | 1,162K | 705.2 |
| SkillDAG | 34.38 | 1,370K | 861.1 |
| **SPEM (Ours)** | **38.38** | **980K** | **618.7** |

> ✅ **SPEM 在 GPT-5.2-Codex 下达到最高的 38.38% 奖励，且输入 token 减少 15.7%，runtime 降低 12.3% 对比 GoS。**

### **与基线方法的关键对比**
- 相比 **SkillDAG**（当前 SOTA）：
  - 在 `MiniMax-M2.7` 上，SPEM **减少 26.3% token 和 24.3% runtime**，奖励接近。
  - 在 `GPT-5.2-Codex` 上，SPEM **提升奖励至 38.38%**，同时大幅降低资源消耗。
- 相比 **GoS**：
  - 在 ALFWorld 上成功率更高（87.1% vs 85.3%），且 runtime 更低（66s vs 71s）。
  - 在 SkillsBench 上奖励提升明显（+0.88pp），且更高效。

### **消融实验结果（Ablation Study on ALFWorld）**
| 方法变体 | Success (%) | Steps | Runtime (s) | Tokens |
|---------|------------|--------|--------------|--------|
| w/o Skill-DAG | 56.8 | 23.0 | 55.4 | 67.5K |
| w/o Skill-GNN | 58.3 | 20.3 | 54.7 | 53.6K |
| w/o DAG Extraction | 63.0 | 18.9 | 57.8 | 56.2K |
| **Full SPEM** | **70.7** | **18.3** | **51.6** | **45.8K** |

> 🔍 发现：
> - 移除 **Skill-DAG Memory** 导致成功率下降最严重（-13.9%），说明可复用流程至关重要。
> - 移除 **Skill-GNN** 后性能仍差，表明状态感知选择不可或缺。
> - 不进行 **DAG 提取**（保留完整轨迹）会增加冗余，影响效率和成功率。

---

## **4. 关键结论和发现**

### **主要发现**
1. **结构感知优于纯语义检索**  
   单纯依赖 embedding 相似性无法捕捉技能间的依赖与顺序，容易引入无关或冲突技能。

2. **可复用的程序性记忆显著提升效率**  
   Skill-DAG 模板使得 Agent 能“记住”过去成功的流程，在相似任务中直接复用，避免重复规划。

3. **紧凑指导比大上下文更有效**  
   尽管更强的 LLM（如 GPT-5.2-Codex）能处理更多上下文，但 **SPEM 通过结构化精简输入实现了更高成功率和更低开销**，证明“质量 > 数量”。

4. **Skill-GNN 具备良好的可扩展性**  
   如 Figure 3(c)(d) 所示，在技能库扩大到 2000 个时，Skill-GNN 仍能保持较高的 Hit@K 和 MRR，远超仅靠检索的方法。

5. **DAG 模板具有强压缩性和实用性**  
   - 轨迹平均从 13.4 步压缩至 6.0 步（↓54.9%）。
   - 在 ALFWorld 中，模板命中率达 **60.71%（GPT-5.2-Codex）**，命中任务的成功率为 **85.88%**。
   - 精确模板引导可将 steps ↓45.2%、tokens ↓49.8%、runtime ↓25.5%。

### **方法的局限性**
- **依赖高质量的历史成功轨迹**：若初始阶段失败较多，Skill-DAG Memory 难以快速建立。
- **Skill-GNN 训练需要监督信号**：虽支持冷启动，但在早期阶段性能受限。
- **对异构性强的任务（如 SkillsBench）模板复用有限**：因技术任务多样性高，Exact-DAG 复用机会较少。

### **未来工作方向**
- 引入 **failure-aware memory update**：不仅记录成功路径，也分析失败原因以优化未来决策。
- 扩展至 **multi-agent 场景**：支持团队协作下的共享程序性记忆。
- 探索 **zero-shot DAG 初始化**：利用先验知识图谱或仿真数据预填充 Skill-DAG Memory。
- 结合 **memory forgetting mechanism**：自动清理低频或过时的 Skill-DAG 模板，防止记忆膨胀。

---

> 📌 **总结一句话**：  
> **SPEM 通过“原子技能检索 + 结构化流程复用 + 图神经网络状态感知选择”的三重机制，在保证高任务成功率的同时，显著降低了 LLM Agent 的 token 消耗和运行时间，是迈向高效、可积累、可进化智能体的重要一步。**

</details>

---

### 11. [LatBridge: Efficient and Stable Model Replacement in Latent-Communication Multi-Agent Systems](https://arxiv.org/abs/2610.27813)

**Authors**: Jiajun Zhu, Haobo Li, Zixin Chen, Huamin Qu, Wenshuo Zhang  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.27813v1  

#### Abstract
Multi-agent systems rely on communication to complete tasks, and latent communication passes continuous representations to reduce text generation and re-encoding. However, multi-agent systems must reconfigure agent models, whereas existing latent-communication methods are built for fixed model combi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LatBridge: Efficient and Stable Model Replacement in Latent-Communication Multi-Agent Systems**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **接口失效问题**：现有的 **latent-communication** 方法（如 Interlat）在多智能体系统中依赖于固定的发送和接收模型组合来训练通信接口。当某个 agent 的模型被替换时，原有的接口无法直接复用，导致通信性能严重下降。
- **缺乏对动态配置的支持**：多智能体系统需要根据任务需求、资源限制等动态调整 agent 模型，但现有方法未解决模型更换后的接口兼容性和稳定性问题。

### **提出了什么新方法或新思路**
提出 **LatBridge**，一种支持高效且稳定模型替换的 latent-communication 框架，其核心思想包括：
- **共享通信协议（Shared Communication Protocol）**：通过一个固定的 **Bridge model** 作为所有 agent 的公共通信枢纽，将端到端的表示映射分解为两个独立的接口（input interface 和 output interface），分别连接源 agent 和目标 agent 到 Bridge。
- **接口重用机制**：当某一方 agent 模型被替换时，仅需查找或训练与新模型匹配的新接口，而另一侧已有的接口可被复用，无需重新训练整个通信路径。
- **加速模式（Accelerated Mode）**：避免 Bridge model 进行自回归生成（autoregressive generation），而是利用预构建的词汇映射（vocabulary mapping）和嵌入查找（embedding lookup）直接构造目标端连续输入，显著降低延迟。

### **相比现有方法的优势**
| 维度 | LatBridge | 现有方法（如 Interlat） |
|------|---------|------------------|
| **模型可替换性** | 支持灵活更换 agent 模型，接口可部分复用 | 接口绑定固定模型对，更换后性能骤降 |
| **训练效率** | 仅训练缺失接口，其余组件冻结 | 需要重新训练或微调整个通信链路 |
| **通信延迟** | 加速模式下几乎无额外处理延迟（~0.3ms） | 自回归生成带来高延迟（>500ms） |
| **通用性设计** | 模块化接口设计便于跨配置复用 | 缺乏标准化通信中间层 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MATH**：数学推理数据集，评估最终答案的 **exact match** 准确率。
- **ALFWorld**：具身环境中的任务规划数据集，评估下一步动作预测的 **normalized exact match**。

### **实验设置**
- **模型规模**：
  - 小模型：`Qwen2.5-0.5B-Instruct` 和 `Llama-3.2-1B-Instruct`
  - 大模型：`Qwen2.5-7B` 和 `Llama-3.1-8B-Instruct`
- **配置数量**：每种规模下共 8 种不同的 source/Bridge/target 组合。
- **通信预算（Communication Budget）**：设定为 $ L = 128 $，控制传递的状态长度。
- **接口结构**：采用 **low-rank adaptation**（LoRA 风格）实现轻量级接口，rank=32，参数量远小于完整模型。

### **评估指标**
- **任务准确率（Task Accuracy）**：MATH 和 ALFWorld 上的答案/动作匹配率。
- **通信延迟（Communication Latency）**：从 Bridge 输入到目标端连续输入准备完成的时间（GPU 同步计时）。
- **消融实验**：验证接口模块、训练策略、消息内容的影响。
- **表示分析**：通过线性探针（linear probing）、最近邻检索等方式分析表示空间的信息保留能力。

### **基线方法对比**
由于没有现成支持“模型替换”的 baseline，作者采用以下方式比较：
- **Interlat**：作为 latent communication 的代表性方法，在原始配置下提供性能参考。
- **No Communication**：移除 Bridge 输出，仅保留目标任务指令，用于衡量通信增益。
- **Pilot Study**：展示直接复用原接口在模型替换后的性能崩溃，证明问题存在。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **基本模式（Basic Mode）平均准确率**
| 数据集 | 小模型均值 | 大模型均值 |
|--------|------------|-----------|
| MATH | **14.8%** | 28.7% |
| ALFWorld | **27.3%** | 28.0% |

> 表明即使在小模型上，使用匹配接口仍能恢复有效通信。

#### ✅ **加速模式（Accelerated Mode）性能提升**
| 数据集 | 小模型均值 | 大模型均值 |
|--------|------------|-----------|
| MATH | **18.0%** | 50.5% |
| ALFWorld | **36.6%** | 44.8% |

> 所有配置下，加速模式不仅大幅降低延迟，还普遍提升了任务准确率。

#### ✅ **通信延迟对比（Table 5）**
| 规模 | 任务 | 基本模式（ms） | 加速模式（ms） | **加速倍数** |
|------|------|----------------|----------------|--------------|
| Small | MATH | 1,262.53 | 0.334 | **~3,777×** |
| Small | ALFWorld | 559.84 | 0.335 | **~1,669×** |
| Large | MATH | 2,256.96 | 0.367 | **~6,149×** |
| Large | ALFWorld | 1,181.89 | 0.354 | **~3,335×** |

> 加速模式将通信处理时间从数百毫秒降至亚毫秒级别，接近即时传输。

### **与基线方法的对比结果**
- 在 **pilot study** 中，若直接复用原接口进行模型替换：
  - MATH 上仅保留原始通信增益的 **1.5%-10%**
  - ALFWorld 上通信增益完全消失（接近 no communication）
- LatBridge 通过接口选择/训练机制，成功恢复通信效能，证明其接口复用机制的有效性。

### **消融实验结果（Ablation Study）**
| 变体 | MATH 准确率 ↓ | ALFWorld 准确率 ↓ | 结论 |
|------|---------------|--------------------|------|
| 完整 LatBridge | 11.0% | 22.4% | — |
| 无 input interface | 2.5% | 0.0% | 输入接口至关重要 |
| 无 output interface | 0.5% | 1.5% | 输出接口不可或缺 |
| 不解耦输出训练 | 3.0% | 0.0% | 分阶段训练更优 |
| 联合优化接口 | 2.5% | 33.6% | 效果不稳定 |
| 更新 Bridge 参数 | 2.5% | 22.4% | 冻结 Bridge 更好 |

> 结果表明：**分离接口 + 解耦两阶段训练 + 冻结 Bridge** 是关键设计。

---

## **4. 关键结论和发现**

### **主要发现**
1. **接口不可复用是真实存在的挑战**：实验证明，现有 latent communication 方法在模型替换后性能急剧下降，说明必须显式设计支持接口迁移的机制。
2. **LatBridge 实现了高效的接口复用**：通过引入共享 Bridge 和模块化接口，实现了“换模型不换全部通信链路”，大幅减少再训练成本。
3. **加速模式兼具高效与高性能**：跳过自回归生成不仅极大降低延迟，还能提升任务准确率，可能因减少了信息损失。
4. **通信内容携带样本特异性信息**：消融实验显示，替换 Bridge 输出中的状态会显著降低准确率，说明传递的是有意义的任务相关表示。
5. **表示空间差异影响读取效果**：不同模型对接口输出的解读能力存在差异，强调了适配目标模型的重要性。

### **方法的局限性**
- **依赖固定 Bridge model**：若 Bridge 模型本身也被替换，则原有两侧接口均需更新，当前框架不支持 Bridge 更换。
- **接口覆盖范围有限**：目前实验集中在 Qwen 和 Llama 两个家族，尚未验证对更多架构（如 Mistral、Phi）的泛化能力。
- **静态通信拓扑假设**：实验主要针对单跳通信关系，多跳、动态拓扑下的累积误差和协调机制尚待研究。
- **离线评估为主**：使用的是 offline 下一动作预测，未测试在交互式环境中长期决策与错误恢复的能力。
- **表示可解释性不足**：虽然表示保留了信息，但难以解释具体语义含义，缺乏对 latent message 的可视化或干预工具。

### **未来工作方向**
- 扩展至 **multi-hop communication** 场景，研究多轮转发中的误差传播与校正机制。
- 探索 **dynamic Bridge switching** 或 **meta-learning 接口适配器**，以支持 Bridge 模型的动态更换。
- 构建 **universal interface standard**，推动跨模型族、跨模态（vision-language）的 latent communication 协议。
- 引入 **interpretability 工具** 来分析 latent message 的语义结构，辅助调试与安全审计。
- 在 **real-world interactive environments** 中部署并评估系统的鲁棒性与安全性。

--- 

> **总结一句话**：  
> **LatBridge 提出了一种模块化、可复用的 latent communication 架构，首次系统性解决了多智能体系统中模型替换带来的接口失效问题，在保持高任务准确率的同时，将通信延迟降低了三个数量级以上。**

</details>

---

### 12. [Explore More, Validate Less: Efficient Reflective Prompt Optimization](https://arxiv.org/abs/2610.27367)

**Authors**: Xiaofan Zhou, Lu Cheng  
**Category**: cs.CL  
**Published**: 2026-10-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.27367v1  

#### Abstract
Reflective prompt optimization improves instructions using task feedback, but coupling candidate generation with full-validation-based selection can make search costly and sequential. We propose a two-stage framework that decouples reflective candidate-pool construction from validation allocation. D...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Explore More, Validate Less: Efficient Reflective Prompt Optimization**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
现有的 **reflective prompt optimization** 方法（如 GEPA）在优化提示时存在两个主要瓶颈：
- **串行依赖性强**：每次生成候选提示后必须进行完整的 validation 才能决定是否保留，导致搜索过程高度串行化，效率低下。
- **资源浪费严重**：即使父提示在训练集上表现完美（all-correct minibatch），仍需重复计算；被拒绝的子提示也消耗了验证成本却未扩展候选池。

这限制了在有限预算下对多样化提示路径的探索能力。

---

### ✅ **提出了什么新方法或新思路**
本文提出一个**两阶段框架**（two-stage framework），将 **candidate-pool construction** 与 **validation-based selection** 显式解耦：

#### 🔹 阶段一：**基于训练反馈的候选池构建（Construction Stage）**
- 多个 parent prompts 并行采样，每个 parent 在小规模 training minibatch 上生成响应并保存反馈。
- 利用这些共享反馈，在多个并行分支中进行 reflection 和 revision，生成 child prompts。
- 子提示仅通过 **local screening**（在相同 minibatch 上比较得分提升）判断是否保留，无需 full validation。
- 构建大小为 $ N $ 的冻结候选池 $ \mathcal{P} $，允许池子远大于后续可验证数量。

#### 🔹 阶段二：**基于固定预算的验证选择（Selection Stage）**
- 冻结候选池后，使用 exploration policy 从池中选择最多 $ B $ 个候选进行 full validation（$ B \ll N $）。
- 最终返回 validation score 最高的 prompt。
- 支持多种 selection 策略（如随机选择、学习预测 + 相似性探索），且可在同一池上复用。

---

### ✅ **相比现有方法的优势**
| 特性 | 本方法 | GEPA / MIPROv2 |
|------|--------|----------------|
| **并行性** | 多 parent + 多 minibatch 并行反射 | 单 proposal 或受限并行 |
| **验证开销控制** | Full validation 仅用于最终筛选，不参与生成决策 | 每次接受候选都需 full validation |
| **资源利用率** | 可保留无完整验证的中间提示支持进一步演化 | 被拒候选完全浪费 |
| **灵活性** | 池大小 $ N $ 与验证预算 $ B $ 可独立设置 | 通常 $ B $ 决定搜索深度 |
| **探索广度** | 更深更宽的搜索树（higher depth & branching） | 搜索较浅窄 |

> 💡 核心思想：**“Explore More, Validate Less”** —— 允许更广泛的探索，把昂贵的 full validation 留给最有希望的少数候选。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
在五个具有代表性的 benchmark 上评估：
| 数据集 | 任务类型 | 评分方式 |
|-------|--------|---------|
| **HotpotQA** | 多跳问答（multi-hop QA） | normalized exact match |
| **SuperGPQA** | 研究生级别多选题 | accuracy |
| **LiveBench-Math** | 数学推理 | 自定义答案检查器 |
| **IFBench** | 指令遵循能力 | constraint satisfaction ratio |
| **BIG-Bench Extra Hard (BBEH)** | 复杂推理任务 | binary answer matching |

所有数据集划分为固定的 train/validation/test 各 150/300/300（LiveBench-Math 为 122/122/122）。

---

### ⚙️ **实验设置和评估指标**

#### **模型**
- **Qwen3.5-9B**
- **Llama-3.1-8B-Instruct**

> 所有任务中，同一模型负责 solving、reflection 和 revision，参数冻结。

#### **关键超参**
- 候选池大小 $ N = 100 $
- 验证预算 $ B = 20 $
- 每轮最多采样 $ P = 5 $ 个 parents
- 每个 parent 回答 $ M = 15 $ 个训练样本
- 每 parent 生成 $ K = 5 $ 个 minibatch（每 batch $ b = 3 $ 例）
- 温度设置：task solving 用 temp=0，reflection/revision 用 temp=0.7

#### **评估流程**
- 最终测试使用三个 inference seed 取平均。
- 测试时不进行任何 reflection 或 prompt 修改，仅固定最优 prompt 推理。

---

### 🔁 **基线方法对比**
| 方法 | 类型 | 是否更新参数 |
|-----|------|-------------|
| **Initial solver** | 初始提示 | ❌ |
| **MIPROv2** | 联合优化 instruction + demo | ❌ |
| **GEPA** | 基于 Pareto 的 reflective evolution | ❌ |
| **DPO** | 参数微调（adapter） | ✅ |
| **Ours (Random)** | 本文方法 + 随机选择 | ❌ |
| **Ours (MLP + similarity)** | 本文方法 + 学习选择 | ❌ |

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据（Test Score %, Higher is Better）**

| Dataset | Initial | MIPROv2 | GEPA | DPO | Ours (Rand) | Ours (MLP+sim) |
|--------|--------|--------|------|-----|------------|---------------|
| **Llama-3.1-8B-Instruct** | | | | | | |
| SuperGPQA | 17.00 | 18.67 | 16.67 | 16.33 | **18.89** | **21.44**↑ |
| LiveBench-Math | 7.38 | 8.47 | **10.66** | 7.38 | 10.38 | 9.02 |
| IFBench | 31.00 | 31.50 | 30.78 | **32.00** | 28.11↓ | 31.67 |
| BBEH | 9.33 | 9.33 | 7.33 | 8.33 | **10.00**↑ | 9.67 |
| HotpotQA | 32.33 | **44.89** | 41.00 | 41.00 | 42.44 | 42.22 |
| **Qwen3.5-9B** | | | | | | |
| SuperGPQA | 48.44 | 50.00 | 50.56 | 49.33 | 51.33 | **52.78**↑ |
| LiveBench-Math | 53.28 | 51.64 | 52.05 | 52.46 | 56.28 | **56.83**↑ |
| IFBench | 39.28 | 31.33 | 40.39 | 36.83 | **41.67**↑ | 41.22 |
| BBEH | 25.22 | 22.00 | 21.11 | 21.33 | 24.56 | **25.78**↑ |
| HotpotQA | 40.67 | 49.33 | 51.00 | 40.00 | **51.89**↑ | 51.78 |

> ✅ **亮点总结**：
- 在 **10 项任务-模型组合中，我们的方法在 7 项取得最高分**。
- **Qwen 上全面领先**：全部 5 项任务均优于所有 baseline。
- **最大增益达 +11.22 pp**（Qwen HotpotQA：40.67 → 51.89）。
- **Learned selection 在所有 10 项中均优于初始 solver**，而 random selection 在 8 项中胜出。

---

### 🔍 **与基线方法的对比结果**

| 对比维度 | 结果 |
|--------|------|
| **vs. GEPA** | 我们的方法在 8–9/10 项任务中优于 GEPA，尤其在数学和复杂推理任务上优势明显（如 Qwen LiveBench-Math +4.78 pp）。 |
| **vs. MIPROv2** | 在 Qwen 上全面超越；但在 Llama 上部分任务落后（如 HotpotQA）。 |
| **vs. DPO** | 在 9/10 项中优于 DPO，说明无需参数更新也能实现更强 prompt 表达力。 |
| **两种 selection 策略之间** | 无一致优劣：MLP-based 在某些任务领先（如 SuperGPQA），random 在其他任务更好（如 IFBench），表明仍有改进空间。 |

---

### 🔧 **消融实验与额外分析**

#### ✅ **运行时间分析（RQ2）**
| 方法 | 平均 build 时间 | 平均 validation 时间 | 总时间更低？ |
|------|------------------|------------------------|--------------|
| **Ours** | 较低（除 Qwen SuperGPQA） | 相近 | ✅ 8/10 更快 |
> - 尽管构造阶段耗时较长，但由于避免频繁 full validation，总体时间更具竞争力。
> - 特别是在 Llama 上，总时间显著低于 GEPA-20。

#### ✅ **搜索树结构分析（RQ3）**
| 统计量 | Ours | GEPA | 发现 |
|-------|------|------|------|
| **Mean Depth** | 更高（9/10） | 较低 | 演化路径更长，支持深层改进 |
| **Mean Branching Factor** | 更高（8/10） | 较低 | 更广泛地探索替代方案 |

> 表明该框架确实实现了“更广更深”的搜索行为。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **解耦 construction 与 validation 是有效的**：
   - 允许在不消耗昂贵 full validation 的前提下构建大规模候选池，显著提升探索效率。
2. **“Explore More, Validate Less” 提升性能上限**：
   - 即使使用简单的随机选择策略，也能在多数任务上超越已有方法。
3. **learned selection 策略稳定优于初始 prompt**：
   - 表明 learned score prediction + novelty exploration 是可行方向。
4. **无需参数更新即可媲美甚至超越 DPO**：
   - 在大多数任务上优于 DPO，说明高质量 prompt 设计本身极具潜力。

---

### ⚠️ **局限性**
1. **learned selection policy 较简单**：
   - 当前使用的是基于 prompt embedding 的 MLP，未建模演化路径（trajectory）信息。
2. **仅在开源模型上验证**：
   - 实验限于 Qwen 和 Llama，未测试闭源模型（如 GPT、Claude）。
3. **未考虑语义多样性**：
   - 搜索树统计关注结构而非语义差异，可能存在冗余提示。
4. **total cost 并非绝对下降**：
   - 虽然验证更高效，但 construction 成本上升，端到端节省需权衡。

---

### 🔮 **未来工作方向**
1. **设计更强大的 exploration policy**：
   - 引入 RNN/GNN 建模 prompt 演化轨迹，利用历史 reflection 信息预测性能。
2. **扩展至 black-box API 场景**：
   - 将本框架应用于 GPT 等闭源模型，结合 soft prompt 或指令优化。
3. **引入多样性正则化机制**：
   - 在 construction 阶段鼓励语义多样化的 prompt 分支。
4. **动态调整 construction 与 validation 预算分配**：
   - 根据任务难度自适应设置 $ N $ 和 $ B $。

---

## ✅ **总结一句话**
> 本文提出 **“先广泛探索、再精准验证”** 的两阶段 prompt 优化框架，通过解耦 candidate construction 与 full validation，在保持参数不变的前提下显著提升了自动 prompt engineering 的效率与性能，验证了 **“Explore More, Validate Less”** 的有效性。

</details>

---

### 13. [DStill: Trajectory Consistency Distillation for Accelerated One-Step Drafting in Block Diffusion Speculative Decoding](https://arxiv.org/abs/2610.27379)

**Authors**: Shreyas Kulkarni, Uttam Gurram, Madhav Kanda, Krishna Teja Chitty Venkata, Shichu Chen, Kartik Balasubramaniam, Supriyo Chakraborty, Sambit Sahu  
**Category**: cs.CL  
**Published**: 2026-10-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.27379v1  

#### Abstract
Block-diffusion speculative decoding accelerates large language model inference by drafting multiple tokens in parallel. However, one-step drafting in discrete diffusion is limited by token factorization error: each token is predicted from the same masked block, without conditioning on the tokens se...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**DSTILL: Trajectory Consistency Distillation for Accelerated One-Step Drafting in Block Diffusion Speculative Decoding**

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

在基于 **Block Diffusion** 的 **Speculative Decoding** 中，**One-step drafting** 存在一个关键缺陷：**Token Factorization Error**。

- 在离散扩散模型中，每个 token 是独立预测的，没有考虑同一 block 内其他位置已选 token 的上下文。
- 这导致生成的 draft 序列可能不一致（例如，出现逻辑冲突的词对），降低 **acceptance length**（接受长度），从而削弱加速效果。
- 虽然 **multi-step refinement** 可以通过逐步 unmasking 来提升一致性，但会增加推理延迟，违背了加速初衷。

因此，核心问题是：**如何在保持 one-step drafting 低延迟的同时，获得 multi-step refinement 的高 draft 一致性？**

---

### ✅ **提出了什么新方法或新思路**

作者提出 **DSTILL** —— 一种两阶段的 **trajectory consistency distillation** 框架，将多步 refinement 的能力“蒸馏”到单步 drafting 模型中。

#### **核心思想：**
1. **Stage I：训练一个 Partial-Mask Teacher**
   - 训练一个目标条件化的 drafter，使其能够从部分可见的 block（即部分 token 已揭示）中完成剩余 token 的预测。
   - 这模拟了 multi-step refinement 中每一步的输入状态。

2. **Stage II：轨迹一致性蒸馏（Trajectory Consistency Distillation）**
   - 固定 teacher，离线收集其在完整 refinement 过程中的 **denoising trajectories**（即每一步的中间状态和最终输出）。
   - 训练一个与 teacher 架构相同的 **student**，要求它能从任意中间状态（包括初始 root state）直接预测出 teacher 的最终输出（endpoint）。
   - 引入 **curriculum learning**：训练初期更多使用中间状态提供上下文，后期逐渐增加 root state 的采样概率，使 student 最终适应 one-step 推理。

#### **理论基础：**
- 借鉴了 **Consistency Models** 的思想：学习从扩散路径上任意点直接映射到最终输出。
- 在离散语言建模中实现“一步到位”的高质量生成。

---

### ✅ **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **性能** | 显著提升 acceptance length 和 end-to-end 速度，最高达 **6.45×** 加速（vs. autoregressive）。 |
| **效率** | 推理时仅需 **one-pass**，无额外延迟，避免了 multi-step 或 tree search 的开销。 |
| **通用性** | 可应用于不同架构的 drafter（如 DFlash、DSpark），具有扩展潜力。 |
| **零测试时开销** | 蒸馏完成后，student 与 baseline 架构完全相同，部署成本不变。 |

---

## 2. 核心实验方法和设置

### ✅ **使用的数据集**

- **训练数据**：`OpenPerfectBlend` 数据集，涵盖对话、数学、代码等任务。
- **评估基准**：
  - **GSM8K**（小学数学）
  - **MATH-500**（高等数学）
  - **HumanEval & MBPP**（代码生成）
  - **MT-Bench**（多轮对话）
  - **SWE-Bench**（真实 GitHub 问题修复）

---

### ✅ **实验设置和评估指标**

#### **目标模型（Target LLMs）**
- `GPT-OSS-120B`（MoE）
- `Gemma-4-31B`（Dense）
- `Gemma-4-26B-A4B`（MoE）
- `Nemotron-3.5-30B-A3B`（Hybrid Mamba-Attention MoE）

#### **drafter 配置**
- Block size: `B=16`（1 anchor + 15 draft tokens）
- 所有方法使用相同架构和参数量，确保公平比较。

#### **评估指标**
| 指标 | 定义 |
|------|------|
| **Acceptance Length (T)** | 平均每次验证步骤中被接受的 draft token 数量（含 anchor）。越高越好。 |
| **Speedup (S)** | 相对于标准 autoregressive 解码的速度提升倍数：$ S = t_{AR} / t_{spec} $。 |
| **Recovery (%)** | one-step student 恢复 multi-step teacher 性能提升的百分比。 |

#### **推理设置**
- 温度：0 和 1
- 请求并发数（Concurrency）：C ∈ {1, 8, 16, 32}
- 使用 `vLLM` 实现，8×H100 GPU

---

### ✅ **基线方法对比**

| 方法 | 描述 |
|------|------|
| **DFlash** | 当前主流的 one-step block diffusion drafter，作为主要 baseline。 |
| **Multi-step Teacher (8-pass)** | 用于衡量上限性能，但推理慢。 |
| **Autoregressive Decoding** | 速度参考基准（S=1×）。 |

---

## 3. 主要实验结果和性能指标

### ✅ **关键性能数据**

| 模型 | 方法 | Acceptance Length (T) ↑ | Speedup (S) ↑ |
|------|------|------------------------|---------------|
| **Gemma-4-31B** | DFlash | 6.78 | 5.85× |
| | **DSTILL (Ours)** | **7.48** (+10.3%) | **6.45×** (+10.3%) |
| **GPT-OSS-120B** | DFlash | 6.30 | 4.81× |
| | **DSTILL (Ours)** | **6.79** (+7.8%) | **5.17×** (+7.5%) |
| **Nemotron-3.5-30B-A3B** | DFlash | 5.84 | 4.49× |
| | **DSTILL (Ours)** | **6.94** (+18.8%) | **5.36×** (+19.4%) |

> ✅ **总体提升**：
> - **Acceptance length 最高提升 28%**
> - **Wall-clock speedup 最高提升 25%**
> - **最高达到 6.45× 加速**（vs. autoregressive）

---

### ✅ **与基线方法的对比结果**

- 在所有模型和任务上，**DSTILL 全面优于 DFlash**，尤其在 **reasoning 和 code generation** 上提升显著。
- 即使在高并发（C=32）下，DSTILL 仍能保持 **2.7–3.9×** 的加速，说明其在实际服务场景中依然有效。
- 在 **non-thinking 模式** 下，Nemotron 上 DSTILL 达到 **3.88×** 加速（temp=0, C=1）。

---

### ✅ **消融实验结果**

#### **(1) 蒸馏增益来源分析（Table 4）**
| 模型 | 8-pass Teacher 提升 | DSTILL 恢复比例 |
|------|--------------------|------------------|
| GPT-OSS-120B | +18.0% | **35.8%** |
| Nemotron-3.5-30B-A3B | +17.9% | **28.6%** |

> ✅ 表明 DSTILL 成功将多步 refinement 的大部分收益迁移到了单步模型中。

#### **(2) 继续训练 vs. 蒸馏（Table 5）**
- 将 DFlash 训练从 5 轮延长至 10 轮，性能仍低于默认设置的 DSTILL。
> ✅ 证明性能提升来自 **trajectory distillation**，而非更长训练。

#### **(3) 状态采样策略（Table 6）**
| 策略 | Average T |
|------|---------|
| Root-only (proot=1) | 4.01 |
| **Cosine schedule (0.5→0.85)** | **4.09** (+0.08) |

> ✅ 渐进式增加 root state 采样概率的 curriculum 设计更优。

#### **(4) 监督信号组合（Table 7）**
- **Endpoint CE + Top-k KD (α=0.2)** 效果最佳，结合了 token 准确性和分布匹配。

#### **(5) 轨迹质量加权（Table 8）**
- 对 teacher 输出与 reference 不一致的位置降权，可进一步提升 T（+0.08）。

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **Trajectory Consistency Distillation 是有效的**：
   - 成功将 multi-step refinement 的能力压缩到 one-step model 中。
   - 学生模型学会从根状态直接预测“精炼后”的输出，避免了 token factorization error。

2. **性能提升显著且通用**：
   - 在三种不同架构的目标模型上均取得一致提升。
   - 覆盖 reasoning、math、code、chat 多种任务。

3. **无需测试时开销**：
   - 蒸馏后 inference 仍为 one-pass，适合生产部署。

4. **可扩展性强**：
   - 成功应用于 DSpark，acceptance length 提升 **4.5%**，验证了方法的通用性。

---

### ⚠️ **局限性**

1. **更高的训练成本**：
   - 包含 teacher 训练、轨迹收集、学生蒸馏三个阶段，训练开销更大。
   - 适用于大规模推理场景，小规模使用性价比不高。

2. **继承 teacher 错误**：
   - 如果 teacher 的 endpoint 本身错误，student 也会学习并复制这些错误。

3. **输出仍是 factorized**：
   - 尽管内部学习了一致性，但最终输出仍是独立采样，无法建模复杂的联合依赖。

4. **泛化范围有限**：
   - 实验仅覆盖特定模型、块大小和配置，对其他架构或 block size 的迁移需进一步验证。

---

### 🔮 **未来工作方向**

1. **扩展到更大的 block size 或 variable-length drafting**。
2. **结合 dynamic early exiting 或 adaptive refinement depth**。
3. **探索 teacher ensemble 或 self-correction 机制以减少错误传播**。
4. **应用于非语言模态（如语音、图像）的 speculative decoding**。
5. **设计更高效的 trajectory replay 机制以降低存储成本**。

---

> **总结**：  
> **DSTILL** 提出了一种优雅而实用的方法，通过 **offline trajectory distillation** 将多步 refinement 的“思考过程”内化为单步 drafting 的“直觉”，在不增加推理延迟的前提下显著提升了 speculative decoding 的效率和质量，是迈向高效 LLM inference 的重要一步。

</details>

---

### 14. [Adaptive Replay Refinement: Allocating Test-Time Compute Between Revision and Resampling](https://arxiv.org/abs/2610.28294)

**Authors**: Ahsan Bilal, Muhammad Ahmed Mohsin, Muhammad Umer, Ayesha Mohsin, Sarmad Bilal, Dean F. Hougen  
**Category**: cs.CL  
**Published**: 2026-10-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.28294v1  

#### Abstract
Test-time scaling allocates additional inference compute to refining existing solutions or generating new ones. This raises a fundamental question: Should I keep improving this solution, or stop working on it and try a completely new solution? Existing methods make this decision using fixed refineme...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Adaptive Replay Refinement: Allocating Test-Time Compute Between Revision and Resampling*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在推理时可以通过增加计算量来提升性能，主要有两种方式：
- **并行采样**（如 Majority Voting）：生成多个独立解，通过聚合答案提高鲁棒性。
- **顺序修订**（如 Self-Refine）：对单个解进行多次迭代修正。

然而，这两种策略面临一个根本性的**资源分配问题**：  
> “我应该继续改进当前的解决方案，还是放弃它去尝试一个全新的解？”

现有方法通常采用**固定修订轮数**、**自验证机制**或**学习型控制器**来决定何时停止修订。这些方法存在以下问题：
- 固定调度可能浪费算力（当解已收敛后仍继续修订）；
- 自验证信号不可靠（LLM难以准确判断自身答案是否正确）；
- 学习型方法需要额外训练和调参。

---

### 提出的新方法：Adaptive Replay Refinement (ARR)

ARR 是一种**无需训练**（training-free）的方法，旨在动态地将测试时计算资源在“修订”与“重采样”之间进行最优分配。

#### 核心思想
- 利用 **replay decoding** 高效检测修订过程中的**不动点**（fixed point），即当 `F(y) = y` 时，表示进一步修订不会改变输出。
- 一旦检测到不动点，立即终止对该路径的修订，并将其计算资源释放给一个新的独立采样（new root）。
- 这实现了从“深度修订”向“广度采样”的自适应转移。

#### 技术亮点
- **Replay Decoding**：将当前解作为 draft，利用并行验证机制加速修订过程，在块级别（block-level）验证 token 是否被保留，显著减少串行前向传递次数。
- **Exact Fixed-Point Detection**：通过 replay decoding 精确识别 `F(y) = y`，这是一个确定性信号，不依赖于外部反馈或置信度评分。
- **Compute Reallocation**：不动点不仅是停止信号，更是资源再分配的触发器。

---

### 相比现有方法的优势
| 维度 | ARR 的优势 |
|------|-----------|
| **无需训练** | 不需要额外训练 verifier 或 controller，完全基于推理时信号决策。 |
| **高效低延迟** | 利用 replay decoding 大幅缩短关键路径上的串行 passes，降低推理延迟。 |
| **资源利用率高** | 避免在已收敛的解上浪费计算资源，转而用于探索新解。 |
| **通用性强** | 在多种模型架构（full attention, sliding window, hybrid linear attention）上均有效。 |

---

## 2. 核心实验方法和设置

### 数据集
共使用 **7 个基准**，涵盖数学推理与一般推理任务：
- **数学推理**：
  - MATH500（500题）
  - AIME24 / AIME25（各30题）
  - AMC23（40题）
  - OlympiadBench（674题，双语科学竞赛题）
  - GSM8K（1,319题，饱和控制集）
- **非数学推理**：
  - LiveBench（Web of Lies, Zebra Puzzles, Spatial Reasoning，各50题）

---

### 实验设置
- **主模型**：Qwen2.5-Math-7B-Instruct
- **其他模型**（泛化性验证）：
  - Qwen3-8B, Ministral-8B, Gemma-4-E4B, Qwen3.5-9B/2B, Qwen2.5-7B
- **预算单位**：每题分配 **200–800 TFLOPs**（按模型规模缩放）
- **宽度参数 N**：默认为 4 个并发活跃根（active roots）
- **块大小 W**：默认为 32
- **解码方式**：greedy decoding for revision, nucleus sampling (T=0.7) for sampling

---

### 评估指标
| 指标 | 含义 |
|------|------|
| **Accuracy** | 最终预测答案的准确率（多数投票） |
| **Serial Forward Passes (S)** | 关键路径上的串行前向传播次数 → 决定**延迟** |
| **TFLOPs Spent** | 总处理的浮点运算量 → 衡量**总计算成本** |
| **Roots per Question** | 每道题生成的有效解数量 |
| **Net Repair Rate** | `(W→C - C→W)`：修订带来的净修复比例（错误→正确 减去 正确→错误） |

---

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Majority Voting** | 并行采样 | 所有预算用于独立采样，无修订 |
| **Self-Refine** | 固定修订 | 固定 D=4 轮修订，不重用槽位 |
| **SETS** | 显式自验证 | 两阶段 critique-and-correction，需“correct” verdict 才停止 |
| **Sequential Revision** | 随机延续 | 非确定性采样延续，无固定点概念 |
| **ConMA** | 置信度控制 | 基于 token-level confidence 动态聚合与终止 |
| **ARRAR / ARRPLD / ARRDepth** | 控制变量变体 | 分别替换 decoding 方式或禁用重采样 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen2.5-Math-7B-Instruct, B=800 TFLOPs）

| 方法 | Accuracy (%) | Serial Passes | TFLOPs Used | Notes |
|------|-------------|----------------|--------------|-------|
| **ARR (ours)** | **84.6** | **7,877** | 789 | 匹配 Majority Voting 精度 |
| **Majority Voting** | 84.6 | 13,049 | 781 | 需 2.4× 更多串行 passes |
| **ARRDepth** | 81.5 | 3,810 | 381 | 未重用槽位，精度下降 3.1 pts |
| **Self-Refine** | 81.7 | 2,680 | 268 | 固定 4 轮，效率低 |
| **SETS** | 83.1 | — | 793 | 显式验证反而更差 |
| **ConMA** | 75.1–77.4 | 644 | 185 | 节省算力但精度损失大 |

> ✅ **ARR 在相同计算量下达到 Majority Voting 的最高精度，仅需 41% 的串行 passes**

---

### 与基线方法的对比结果

#### ✅ **优于固定修订类方法**
- ARR 比 **ARRDepth** 高 **3.1 pts**（MATH500），说明“深度→广度”的资源再分配是关键增益来源。
- 比 **Self-Refine** 高 **2.9 pts**，且使用更多计算资源（789 vs 268 TFLOPs），表明其更充分挖掘了预算潜力。

#### ✅ **优于显式自验证方法**
- **SETS** 尽管有“correct” verdict 机制，但在 MATH500 上比 ARR 低 **1.5 pts**，说明自验证信号不可靠。
- SETS 有更高固定点率（69% vs 23%），但净修复更少，说明很多“稳定”解仍是错的。

#### ⚠️ **与 Majority Voting 的权衡**
- **等 FLOPs 下**：ARR 在 **MATH500、AMC23、AIME24** 上匹配 Majority Voting 精度，但在 **AIME25 和 OlympiadBench** 上落后最多 7.3 pts。
- **原因**：ARR 收集的 votes 数量约为 Majority Voting 的 **1/6**（16 vs 97）。当修订本身修复能力弱时，更多独立样本更有利。
- **等延迟下**：ARR 更优；**等算力下**：Majority Voting 更优。

#### ✅ **Replay Decoding 加速效果**
- ARR 比 ARRAR（普通 autoregressive decoding）减少 **~30% 串行 passes**。
- 在 MATH500 上，ARR 输出 token 数仅为 Majority Voting 的 **~40%**，因大量文本被复用。

---

### 消融实验结果

| 变体 | 结果分析 |
|------|----------|
| **ARRDepth**（无重采样） | 精度下降 3.1–4.0 pts，证明“释放资源给新样本”至关重要 |
| **ARRAR**（autoregressive decoding） | 精度几乎一致，但串行 passes ↑30%，验证 replay 不影响输出质量 |
| **ARRPLD**（prompt lookup decoding） | 速度更快（仅需 55% 串行 passes），说明 realignment 可进一步优化 |
| **不同 stopping rule**：<br>- `answer stability`<br>- `one revision only` | 前者有时略优，但导致更多串行 passes；<br>**固定点检测仍是最低延迟方案** |
| **不同 N**（宽度）：<br>- N=2 → N=8 | N=8 时精度提升 **1.9 pts**，串行 passes ↓36%，显示可扩展性 |
| **不同 W**（块大小） | W∈[8,64] 对精度影响 <0.5%，对串行 passes 影响 ≤5% |

---

## 4. 关键结论和发现

### 主要发现
1. **修订很快饱和**：
   - 超过 **75% 的解在 4 次修订内达到不动点**（图1a）；
   - 继续修订只会浪费算力。

2. **不动点是有效的停止信号**：
   - `F(y) = y` 是一个**精确、无需训练、无需标签**的信号，表明修订已耗尽。
   - 它不保证答案正确，但保证继续修订无效。

3. **资源再分配优于固定调度**：
   - 将已收敛路径的资源重新用于新采样，能显著提升整体性能（+3.1 pts）。

4. **修订质量决定方法成败**：
   - 当修订操作 `F` 的 **net repair rate > 0.25%** 时，ARR 优于 Majority Voting；
   - 否则，独立采样更优（如 LiveBench 上 ARR 全面落后）。

5. **Replay decoding 极大提升效率**：
   - 通过并行验证 draft token，大幅压缩串行路径，降低延迟。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖内部访问权限** | 需要访问 logits 和 decoding loop，无法用于 black-box API |
| **不提升单条路径质量** | 不能解决“模型根本不会做某类题”的问题 |
| **依赖修订算子质量** | 若 `F` 本身破坏正确答案（如 Qwen3.5-2B），ARR 性能会严重下降 |
| **votes 数较少** | 在需要大量采样的场景（如高方差任务）可能不如 Majority Voting |

---

### 未来工作方向
1. **更强的修订算子设计**：
   - 引入外部工具、执行反馈或多步反思机制增强 `F` 的修复能力。
2. **支持 realignment 的 decoder**：
   - 如 prompt lookup 或 PLD+，可进一步加速。
3. **结合 verifier 或 reward model**：
   - 在不动点基础上引入外部信号，实现更智能的终止与重采样。
4. **动态宽度调整（N）**：
   - 根据问题难度或早期表现自适应调整并发路径数。
5. **应用于其他序列模型任务**：
   - 如代码生成、规划、对话系统中的“反思-重试”机制。

---

> 🔚 **总结一句话**：  
> **ARR 提出了一种无需训练的自适应机制，通过检测修订不动点来动态将计算资源从“精修旧解”转向“探索新解”，在保持精度的同时显著降低推理延迟，揭示了“何时停止修订”这一基本问题可通过确定性信号而非复杂判据来解决。**

</details>

---

### 15. [KPage: Cross-Block Page Compaction for Hardware-Efficient Sparse Attention](https://arxiv.org/abs/2610.28096)

**Authors**: Junwei Feng, Qiaorui Chen, Chenyu Liang, Zhen Guo, Jingyi Yang, Xiaoling Zhou, Jiahao Zhang, Dongliang Huang, Yiqiao Liao, Chengru Song  
**Category**: cs.DC  
**Published**: 2026-10-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.28096v1  

#### Abstract
Sparse attention reduces the computational cost of long-sequence video Transformers, but three-dimensional token layouts often leave selected blocks partially occupied, wasting physical tile capacity. We present KPage, a block-sparse attention scheme that packs valid 16-row K/V pages across logical-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# KPage: Cross-Block Page Compaction for Hardware-Efficient Sparse Attention —— 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现代视频扩散模型（Video Diffusion Models）依赖于长序列的 **Transformer** 架构，其自注意力机制具有 $O(n^2)$ 的计算复杂度，成为训练和推理的主要瓶颈。为缓解此问题，**sparse attention** 被广泛采用，通过结构化稀疏模式减少计算量。

然而，现有 **block-sparse attention** 实现（如 NVIDIA BSA）在处理三维视频布局（T×H×W）时存在显著效率损失：  
- 尽管逻辑上某些 **blk128** 块被选中用于 attention，但由于三维几何边界未对齐，这些块内部可能仅部分行有效（valid），导致大量 **padding**。
- 当前系统将每个选中的逻辑块独立映射到物理 tile，即使多个部分填充块可合并，也无法跨块压缩，造成 **物理 tile 容量浪费**。

> **核心问题**：如何在不改变稀疏图结构、不牺牲大 tile 高效执行的前提下，消除三维布局下部分填充块（partial blocks）带来的内部 padding 浪费？

---

### **提出了什么新方法或新思路**

论文提出 **KPage**，一种面向硬件高效的 **page-compacted 执行方案**，核心思想如下：

#### ✅ **K/V Page Compaction（跨块页压缩）**
- 将每个选中的 **K/V blk128 块** 分解为 **Page16** 单元（每页 16 行）。
- 将来自不同逻辑块的有效 Page16 页面 **跨块打包** 到同一个物理 attention tile 中。
- 打包过程在 **K/V 加载阶段直接完成于 shared memory**，无需在 HBM 中显式构建紧凑张量。

#### ✅ **Query-Adaptive Tiling（查询维度自适应分块）**
- 对于查询（Q）侧，根据有效行数选择两种 tile 形状：
  - `M64×N256`：适用于 ≤64 有效行的 Q 块
  - `M128×N128`：适用于 >64 有效行的 Q 块
- 两者均提供 16,384 个 Q-K 位置，保持 Tensor Core 高效利用率，同时减少 Q 侧 padding。

#### ✅ **Exact Semantics Preservation（精确语义保留）**
- 使用 **exact score masking** 在 softmax 前屏蔽无效分数，确保数学语义不变。
- 支持 **独立 Q/K 分区**（如 SVG2 的语义聚类），允许 Q 和 K/V 使用不同的块边界。

#### ✅ **Backward 兼容设计**
- 构建 **K/V-owned paged layout**，支持反向传播中多个 Q 块对同一 K/V 块梯度更新。
- 引入 **page-pair adjacency metadata**，恢复原始稀疏邻接关系，避免错误连接。

---

### **相比现有方法的优势**

| 方面 | 现有方法（如 BSA） | KPage |
|------|---------------------|--------|
| **K/V 压缩粒度** | 以完整 blk128 为单位，无法跨块合并 | 按 Page16 细粒度打包，跨逻辑块压缩 |
| **内存效率** | 存在大量内部 padding，利用率低 | 显著提升物理 tile 利用率 |
| **执行效率** | 固定 tile 大小，难以适配部分块 | 自适应 Q 路径 + 动态打包，更灵活高效 |
| **语义正确性** | 支持 valid-length，排除 padding 影响 | 同样支持 exact masking，且 backward 可逆 |
| **通用性** | 通常假设 Q/K 共享分区 | 支持独立 Q/K 分区（如 SVG2） |

> ✅ **核心优势**：**在不缩小 compute tile 的前提下实现细粒度 padding 移除，兼顾算法稀疏性和硬件效率。**

---

## 2. 核心实验方法和设置

### **使用的数据集与工作负载**

实验涵盖两类场景：

#### 🔹 控制变量实验（Controlled Workloads）
- 视频长度：5s, 10s, 15s, 30s
- 两种布局：
  - **无 padding**：所有逻辑块完全填满（理想情况）
  - **3D padded**：模拟真实三维布局边界，引入 20.8%-22.8% 的 padding
- 目标：验证 KPage 是否仅在存在部分块时带来增益

#### 🔹 代表性稀疏注意力案例（5 个）
- 序列长度：470K ~ 629K tokens
- 不同 sparsity 水平与部分块比例
- 覆盖多种稀疏模式分布

#### 🔹 开源系统重建布局（8+2 个配置）
从以下开源系统重建稀疏布局进行泛化测试：
- **SSTA**（HunyuanVideo 1.5）
- **STA**（Sliding Tile Attention）
- **VSA**（Video Sparse Attention）
- **SVG2**（Sparse VideoGen2，含独立 Q/K 分区）

---

### **实验设置和评估指标**

| 项目 | 设置 |
|------|------|
| **硬件平台** | 单卡 GPU，CUDA 13.2 |
| **精度** | BF16 输入，FP32 验证参考 |
| **软件栈** | CUTLASS DSL 实现 KPage；NVIDIA cuDNN Frontend 1.28.0 提供 BSA |
| **计时范围** | **bare attention kernel 执行时间**（不含元数据构建） |
| **重复次数** | 多次 warm-up + median 报告（10–20 次迭代） |
| **评估指标** |
| - 正向延迟（forward latency）<br>- 正反向总延迟<br>- 相对于 BSA 的 speedup<br>- 物理 query-key 对容量与利用率<br>- 数值正确性（LSE error, cosine similarity） |

---

### **基线方法对比**

| 基线 | 描述 |
|------|------|
| **NVIDIA BSA** | 主要对比基线，支持 valid-length 的通用 block-sparse attention |
| **FA4 (FlashAttention-4)** | 密集 attention 参考 |
| **VSA-CUDA / VSA-Triton** | FastVideo 提供的专用稀疏 kernel |
| **FlexAttention** | 支持灵活稀疏模式的编译器级 attention 引擎 |
| **SVG2 Variable-Block Kernel** | 官方发布的动态块大小 kernel（基于 FlashInfer） |

> ⚠️ 所有方法使用相同稀疏图与 valid-row 元数据，公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 📊 **控制实验结果（Table 1）**

| 场景 | 几何平均 speedup (vs BSA) |
|------|----------------------------|
| **无 padding（全满块）** | **1.03×**（基本持平） |
| **3D padded（部分块）** | **2.29×** |

> ✅ 表明 KPage 的加速效果**专门针对部分块碎片化问题**，而非通用优化。

#### 📈 **五种代表性 workload 结果**

- **正向 kernel speedup**：1.35× ~ 2.06×
- **几何平均 speedup**：**1.77× vs NVIDIA BSA**

#### 📉 **物理计算量分析**

| 项目 | 数值 |
|------|------|
| 有用 interaction 数量 | 160.6G valid Q-K pairs |
| 原始物理容量（blk128） | 334.1G positions |
| Page16 K/V 压缩后 | ↓ 199.7G |
| + Q64/Q128 routing 后 | ↓ **177.5G** |
| **利用率提升** | **48.1% → 90.5%** |

> 💡 接近理想稀疏界限的 **76–79%**，仍有约 1.3× 提升空间。

---

### **与基线方法的对比结果**

#### ✅ 在 **8 个共享分区布局** 上：
- KPage 在所有情况下取得最低 forward 时间
- 相比最强 exact baseline，提速 **1.13× ~ 4.25×**
  - SSTA 平均：**1.28×**
  - VSA 平均：**2.58×**
  - STA（几乎全满）：**1.13×**（接近平局）

#### ✅ 在 **SVG2 独立分区配置** 上：
- 支持独立 Q/K 聚类，无需修改 kernel
- Wan 1.3B/480p：**1.93× faster**
- **Wan 14B/720p（paper-matched）**：
  - 比最强 exact baseline 快 **1.49×**
  - 比 NVIDIA BSA 快 **1.57×**

#### ✅ 正反向联合性能（training 场景）
- 在 3D padded 布局下，**forward+backward 几何平均 speedup 达 1.37×**
- 在 VSA 配置中最高达 **2.71×**

---

### **消融实验与数值正确性**

#### 🔍 **数值正确性验证**
- 正向输出误差与 BSA 相当（~1e-4）
- LSE error < 2e-6
- 反向梯度 cosine similarity ≥ 0.99999
> ✅ 证明 KPage **完全保留原始稀疏 attention 语义**

#### 🔍 **元数据复用性**
- page table、validity mask、route assignment 等可在稀疏图不变时复用
- 无额外运行时开销

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **部分块填充是当前 block-sparse attention 的主要效率瓶颈**，尤其在三维视频布局中普遍存在。
2. ✅ **KPage 成功实现了“细粒度去 padding”与“大 tile 高效执行”的统一**：
   - 不改变稀疏图
   - 不引入中间 compacted tensor
   - 不牺牲 Tensor Core 利用率
3. ✅ **加速效果高度依赖于部分块比例**：
   - 全满块时与 BSA 持平（1.03×）
   - 高碎片化时可达 2.29× 以上
4. ✅ **方法具有强泛化能力**：
   - 在 SSTA、STA、VSA、SVG2 多种稀疏模式下均表现最优
   - 支持独立 Q/K 分区，兼容语义聚类等高级稀疏策略

---

### **方法的局限性**

| 局限 | 说明 |
|------|------|
| **依赖稀疏图静态性** | 若每帧稀疏图变化频繁，则需重新构建元数据（但可缓存） |
| **增加元数据复杂度** | 需维护 page table、validity mask、adjacency bitmaps |
| **对极小块不敏感** | 若大部分块已接近满，收益有限（如 STA 场景仅 1.13×） |
| **尚未集成至主流框架** | 当前为研究原型，需等待 cuDNN Frontend 合并 |

---

### **未来工作方向**

1. **支持动态稀疏图的在线 page packing**
   - 结合运行时 profiling 实现 adaptive compaction
2. **扩展至其他模态**（如 3D 点云、医学影像）
   - 探索非视频领域的多维稀疏结构优化
3. **与 FlexAttention / Triton 编译器集成**
   - 实现自动化的 page-aware sparse kernel 生成
4. **探索更小 page granularity（如 Page8）**
   - 权衡压缩效率与 metadata 开销
5. **开放源码与生态推广**
   - 作者承诺将开源 KPage 并提交 PR 至 NVIDIA cuDNN Frontend

---

## ✅ 总结一句话

> **KPage 通过跨块 page 压缩与 query 自适应分块，在不改变稀疏图和牺牲硬件效率的前提下，显著提升了三维视频稀疏 attention 的执行效率，在典型场景下实现高达 2.29× 的正向加速，且完全保持语义精确性。**

</details>

---

### 16. [Weight Placement Is Not Enough: Synchronization in Multi-Node CPU Decoding of Ternary LLMs](https://arxiv.org/abs/2610.28620)

**Authors**: Runguo Li, Dezhen Lu  
**Category**: cs.DC  
**Published**: 2026-10-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.28620v1  

#### Abstract
Ternary LLMs read only a few gigabytes per decoded token, yet on dual-socket servers with four NUMA nodes the fastest ternary runtime we found uses less than half of the memory bandwidth. Placing each weight on the node of the thread that reads it is necessary but not enough: in a decode engine with...

---

### 17. [Beyond the Point Mass: Acceptance-Derived Weights and Distributional Losses for Speculative Decoding](https://arxiv.org/abs/2610.28355)

**Authors**: Tianyu Wu, Zhuohan Wang, Jonathan Geuter, Yu Yao, Weifan Jiang, Chengxi Zhang, Ju Li, Minlan Yu  
**Category**: cs.LG  
**Published**: 2026-10-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.28355v1  

#### Abstract
In speculative decoding, a lightweight draft model proposes tokens that the target model verifies in parallel, speeding up generation without changing the target distribution. D-PACE (Wu et al., 2026) trains such drafters with a loss that weights each position by its contribution to the expected acc...

---

### 18. [PaPi: Parallel Picard-Duhamel Integration for Fast Neuro-Spectral Architectures](https://arxiv.org/abs/2610.28591)

**Authors**: Arthur Bizzi, M\'arcio Marques, Kleyton Da Costa, Vitor Balestro, Tiago Novello, Jo\~ao M. Pereira, Daniel Yukimura  
**Category**: cs.LG  
**Published**: 2026-10-21  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.28591v1  

#### Abstract
Neuro-spectral architectures (NeuSA) are a class of physics-informed neural networks (PINNs) that have proven effective for solving time-dependent PDEs in the frequency domain. However, training and inference often rely on costly numerical integration of the learned vector field, typically using seq...

---

### 19. [M-Plan: Comprehensive Planning For LLM Agents with Memory Arrangement](https://arxiv.org/abs/2610.27032)

**Authors**: Ziyang Liu, Jiakang Yuan, Jiayuan Fan, Tao Chen  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.27032v1  

#### Abstract
Large language model (LLM)-based agents have recently demonstrated outstanding performance in single-turn tasks, yet they continue to struggle with long-horizon tasks, particularly in environments without prior training. A key limitation of single LLMs is their difficulty in processing long contexts...

---

### 20. [Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs](https://arxiv.org/abs/2610.27103)

**Authors**: Zijian Liu, Chaoyou Fu, Yan Yang, Caifeng Shan, Ran He  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.27103v1  

#### Abstract
Recent Omni LLMs (OLLMs) have demonstrated remarkable capabilities in joint audio-visual reasoning, yet cross-modal hallucinations remain a major challenge. Existing mitigation methods mainly operate at the output level, overlooking the entanglement of grounding evidence and learned priors within fu...

---

### 21. [Sparse RL Relaxation for LLM Reasoning](https://arxiv.org/abs/2610.27220)

**Authors**: Jiabin Fan, Matthias Lindemann, Zita Marinho, Andr\'e FT Martins, Lili Mou  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.27220v1  

#### Abstract
Recent work has explored continuous representations in place of discrete tokens during Chain-of-Thought (CoT) reasoning, often referred to as soft CoT. This is motivated by the idea that a mixture of token embeddings can represent a superposition of multiple reasoning possibilities within a single s...

---

### 22. [Quantum-JEPA: Quantum Born Priors for Latent-Action JEPA World Models](https://arxiv.org/abs/2610.27921)

**Authors**: Zhiyuan Liu, Sicheng Liu, Haiyu Wu  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.27921v1  

#### Abstract
A single observation can admit several plausible futures, making uncertainty an essential part of world modelling. However, quantum generative models remain underexplored for this task. We introduce Quantum-JEPA, a latent-action JEPA framework that learns distributions of future representations usin...

---

### 23. [Robust Representation Learning for Spiking Neural Networks based on Conditional Information Bottleneck](https://arxiv.org/abs/2610.28110)

**Authors**: Yuli Zha, Wang Fan, Yixiao Dai, Lingjie Wang, Bang Hu, Wei Cao, Fengzhe Zhang, Fan Zhang  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.28110v1  

#### Abstract
Spiking Neural Networks (SNNs) have shown great potential for low-power neuromorphic computing due to their event-driven processing and spatiotemporal sparsity, but they remain vulnerable to input perturbations, hardware-intrinsic noise, and adversarial attacks in real-world deployment. We attribute...

---

### 24. [ProVe: Proposer-Verifier Decoding](https://arxiv.org/abs/2610.27815)

**Authors**: Ziqi Jin, Weiwen Xu, Jiaxi Li, Yujie Zhong, Qi She, Aixin Sun  
**Category**: cs.CL  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.27815v1  

#### Abstract
Speculative decoding accelerates autoregressive (AR) language models by letting a small model draft tokens serially and a large model verify them in parallel. We reverse these roles. Starting from a standard AR checkpoint, we fine-tune the large model into a proposer that predicts a block of future ...

---

### 25. [A Hybrid TFT-NBEATS Framework for Robust Multi-Horizon Photovoltaic Power Forecasting](https://arxiv.org/abs/2610.27155)

**Authors**: Shafi Bin Sultan, Sabik Bin Sultan, Mohammad Tajrian Arafat Alvi, Rayhanul Islam Sony, Abdulah Al Numan  
**Category**: cs.LG  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.27155v1  

#### Abstract
Accurate multi-horizon photovoltaic (PV) power forecasting is essential for grid stability, energy scheduling, and the efficient operation of PVbatterygrid systems. However, existing models often fail to jointly capture short-term variability and long-term temporal patterns, while lacking unified mu...

---

### 26. [Sample-efficient learning with Dense Associative Memories](https://arxiv.org/abs/2610.27384)

**Authors**: Andrew Lucas, Dmitry Krotov  
**Category**: cs.LG  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.27384v1  

#### Abstract
We analytically study the evolution of the energy function of Dense Associative Memories (DenseAMs) as they train on randomly sampled codewords of random expander low-density parity-check codes (a.k.a. solutions of random XORSAT problems). We identify two distinct regimes and a transition between th...

---

### 27. [Foundation-Style Pretraining for Scalable and Transferable Distributed PV Forecasting](https://arxiv.org/abs/2610.27763)

**Authors**: Dayin Chen, Xiaodan Shi, Xin Xu, Zhongyue Su, Shibo Zhu, Ruizhe Deng, Dongxiao Zhang, Yuntian Chen, Jinyue Yan  
**Category**: cs.LG  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.27763v1  

#### Abstract
The rapid growth of distributed photovoltaic (PV) generation is increasing the demand for accurate forecasting across large numbers of geographically dispersed stations. Existing approaches are commonly developed for individual sites or specific forecasting horizons, requiring repeated training and ...

---

### 28. [Observation Assimilation Transformer: Probabilistic Weather Analysis from Sparse Station Observations](https://arxiv.org/abs/2610.28327)

**Authors**: Christian Donner, Michele Cattaneo, Maxim Samarin, Francesco Zanetta, Ic\'iar Llor\'ens Jover, Garbiela Aznar Siguan, Michele Volpi  
**Category**: cs.LG  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.28327v1  

#### Abstract
We present the Observation Assimilation Transformer (OAT), a geometry-aware neural process that corrects and refines a prior numerical weather prediction (NWP) field into a high-resolution posterior field by assimilating sparse observations. OAT is an encoder-decoder transformer that considers a set...

---

### 29. [Convergence of stochastic gradient descent for learning linear neural networks](https://arxiv.org/abs/2610.28561)

**Authors**: Gabin Maxime Nguegnang, Bubacarr Bah, Holger Rauhut, Ulrich Terstiege  
**Category**: cs.LG  
**Published**: 2026-10-21  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.28561v1  

#### Abstract
Stochastic gradient descent (SGD) is the cornerstone of modern deep learning, yet its theoretical convergence properties remain poorly understood in non-convex settings. To address this gap, we study SGD dynamics in deep linear neural networks - simplified models that retain key non-convex character...

---

### 30. [Pareto-Optimal RTL Code Generation via Multi-Objective Reinforcement Learning with Large Language Models](https://arxiv.org/abs/2610.26904)

**Authors**: Koki Takeshita, Kenji Uebayashi, Hikaru Tomonari, Masaya Tsunokake, Koichiro Yawata, Tatsuya Sasaki, Yusuke Ohtsubo, Kota Dohi, Yasuhiro Sogawa  
**Category**: cs.AI  
**Published**: 2026-10-21  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.26904v1  

#### Abstract
Recent studies have proposed reinforcement learning with verifiable rewards (RLVR) methods that enable LLMs to generate high-quality RTL circuits from natural-language specifications. Existing approaches optimize circuit PPA (Power, Performance, Area) by aggregating multiple objectives into a single...

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

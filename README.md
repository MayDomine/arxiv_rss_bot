# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-30 08:03:08 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [GLIDE: Guided Layerwise Hybrid Attention for Efficient LLM Inference](https://arxiv.org/abs/2607.24788)

**Authors**: Vimal William, Ravi Tandon, Jyotikrishna Dass  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2607.24788v1  

#### Abstract
As Large Language Models scale to increasingly long contexts, the memory I/O and computational overhead of the Key-Value (KV) cache during decoding emerges as the primary throughput bottleneck. To address this, we propose GLIDE, a Guided Layerwise Hybrid Attention that strategically integrates slidi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# GLIDE: Guided Layerwise Hybrid Attention for Efficient LLM Inference 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
随着 **Large Language Models (LLMs)** 处理的上下文长度不断增长，**autoregressive decoding** 过程中的 **Key-Value (KV) Cache** 成为推理吞吐量的主要瓶颈。KV cache 的内存占用和 I/O 开销随序列长度线性增长，导致模型从计算密集型转变为内存带宽受限，严重影响长上下文生成的效率。

现有方法如 **Sliding Window Attention (SWA)** 和统一的 **Hybrid Attention** 虽然能缓解问题，但存在以下不足：
- **SWA** 通过丢弃旧 token 来减少内存，但可能丢失重要长期依赖。
- 统一的 Hybrid 方法在所有层采用相同的 softmax 与线性注意力混合策略，忽略了不同层对线性化的敏感度差异。

### 提出的新方法：GLIDE
本文提出 **GLIDE (Guided Layerwise Hybrid Attention)**，一种基于层间异质性的自适应混合注意力机制，其核心思想是：
> **早期层保留高保真的 softmax attention，深层则逐步替换为高效的线性注意力（linear recurrent aggregation）**。

#### 创新点：
1. **揭示了 Transformer 层对线性化的深度依赖敏感性（Layer-wise Heterogeneity）**：
   - 早期层对线性化极为敏感，移除 softmax 会导致准确率急剧下降（平均下降 36%）。
   - 深层对线性化容忍度高，可完全线性化而性能损失极小。
2. **提出了非均匀的层间混合策略**：
   - 不再对所有层施加相同比例的 softmax，而是根据层的位置动态分配。
   - 将网络划分为 **Early、Middle、Late** 三个块，分别配置不同的 `θ` 参数控制 softmax 与线性注意力的比例。
3. **模块化设计，兼容现有架构**：
   - GLIDE 可无缝集成到 **Liger、LoLCats** 等基于 **Parameter-Efficient Fine-Tuning (PEFT)** 的 retention-based 架构中，无需重新训练。

### 相比现有方法的优势
| 特性 | 传统 Hybrid / SWA | GLIDE |
|------|------------------|-------|
| 混合策略 | 全局统一 | **层自适应（guided layerwise）** |
| KV Cache I/O | 高或固定压缩 | **显著降低（45×–62×）** |
| 准确率保持 | 一般 | **高达 92%–96% 基线性能** |
| 推理延迟 | 改善有限 | **最高 3.3× 解码加速** |
| 内存占用 | 仍较高 | **支持更长上下文，避免 OOM** |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **推理能力评测基准（zero-shot 或 few-shot）**：
  - **PiQA**（物理常识）
  - **ARC-Easy / ARC-Challenge**（科学推理）
  - **HellaSwag**（常识推理）
  - **WinoGrande**（共指消解）
  - **MMLU**（多任务语言理解，5-shot）
- **训练数据**：50K–100K 条清洗后的 **Alpaca instruction-following 数据**，用于 LoRA 微调。

### 实验设置
- **基础模型**：
  - **Llama-3-8B**
  - **Mistral-7B**
- **窗口大小（window size）**：`w = 1024` 或 `w = 20K`
- **模型维度**：`d_model = 4096`
- **微调方法**：**LoRA (Low-Rank Adaptation)**，rank=8，scaling=8，训练 2 个 epoch。
- **硬件平台**：单张 **NVIDIA Grace Hopper GH200 Superchip (120GB)**。

### 评估指标
| 指标 | 描述 |
|------|------|
| **KV Cache I/O (MB/token)** | 每个生成 token 所需的 KV 缓存内存传输量，衡量内存效率 |
| **End-to-End Latency (s)** | 完整生成过程的累计延迟，反映实际推理速度 |
| **Zero-shot / Fine-tuned Accuracy (%)** | 在多个下游任务上的平均准确率，衡量生成质量 |
| **OOM (Out-of-Memory)** | 是否因内存不足而无法完成长序列推理 |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Vanilla Softmax** | 全层使用标准 softmax attention，KV cache 线性增长 |
| **Sliding Window Attention (SWA)** | 仅保留最近 `w` 个 token，其余丢弃 |
| **Pure Linear Attention** | 全层使用线性注意力，无 KV cache |
| **Uniform Hybrid Model** | 所有层使用相同比例的 softmax + 线性注意力 |
| **GLIDE (Non-uniform)** | 本文方法，early 层全 softmax，late 层全线性，middle 层渐变 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **KV Cache I/O 显著降低**
- 在 **Llama-3-8B** 上，相比 vanilla softmax（4000 MB/token）：
  - **GLIDE (0,0,w)**：降至 **88 MB/token**（**45× 减少**）
  - **GLIDE (0,15w/16,w)**：降至 **43 MB/token**（**93× 减少**）
- 图 7 显示，在 128K 上下文时，baseline KV I/O 达 **17.5 GB**，而 GLIDE 仅 **5.2 GB**（**3.2× 更低**）。

#### ✅ **端到端延迟大幅缩短**
- 在 **Llama-3-8B** 上生成 32K 序列：
  - Vanilla Softmax：**OOM**（无法运行）
  - Hybrid Baseline (0,0,0)：**3705.72 秒**
  - **GLIDE (0,0,w)**：**2638.35 秒**（**提速 ~1.4×**）
  - **Fully Linear (w,w,w)**：**1640.75 秒**（最快，但准确率崩坏）

#### ✅ **准确率保持优异**
- 经 **LoRA 微调后**，GLIDE 在多个配置下保持接近原模型性能：
  - **GLIDE (0,0,w)**：达到 **96% 基线准确率**
  - **GLIDE (0,15w/16,w)**：保持 **92% 基线准确率**
- 表 I 显示，即使在极端压缩下（43 MB/token），fine-tuned GLIDE 仍能达到 **66.74% avg accuracy**（vs. baseline 72.16%）。

#### ✅ **消融实验验证层异质性假设**
- **图 2(b)** 显示：
  - 线性化 **第1-2层** → 准确率暴跌至 **36%**
  - 线性化 **第31-32层** → 准确率几乎不变
- 证明：**早期层是表达瓶颈，必须保留 softmax；深层冗余性强，适合线性化**

---

## 4. 关键结论和发现

### 主要发现
1. **Transformer 各层对注意力线性化的敏感度存在显著差异**：
   - 早期层高度依赖 softmax，移除将导致灾难性性能下降。
   - 深层对线性化具有强鲁棒性，可安全替换以提升效率。
2. **非均匀的层间混合策略优于全局统一策略**：
   - GLIDE 通过“早层保真、深层压缩”实现了 **更优的 accuracy-efficiency Pareto frontier**。
3. **KV Cache I/O 是长上下文推理的核心瓶颈**：
   - 通过减少 softmax 层数量，可直接降低内存带宽压力，从而提升吞吐量和并发能力。
4. **GLIDE 支持更高并发部署**：
   - 内存占用降低 2–3×，意味着同一硬件可服务 **2–3 倍的并发用户**。

### 方法的局限性
1. **线性注意力存在信息稀释（information dilution）问题**：
   - 随着序列增长，常数大小的 recurrent state 会逐渐丢失细粒度 token 区分能力。
2. **小窗口尺寸下 kernel 开销显著**：
   - 当 `w ≤ 128` 时，GPU 上的 kernel launch 和调度开销抵消了理论 FLOPs 优势。
3. **缺乏针对 hybrid attention 的专用优化 kernel**：
   - 当前实现未充分融合 sliding window softmax 与 linear recurrence，仍有优化空间。

### 未来工作方向
1. **改进状态压缩机制**：
   - 设计更高效的状态更新方式，缓解信息稀释。
2. **开发 fused hybrid attention kernels**：
   - 结合 persistent threads、warp-specialized execution 等技术优化执行效率。
3. **扩展至分布式推理场景**：
   - 研究 GLIDE 在 **tensor/pipeline parallelism** 下的表现。
4. **探索更细粒度的动态分配策略**：
   - 基于输入内容动态调整各层的 `θ`，而非静态划分。

---

> **总结一句话**：  
> **GLIDE 通过洞察 Transformer 层间对线性化的异质敏感性，提出了一种非均匀的 guided layerwise 混合注意力机制，在保持 92%–96% 基线性能的同时，实现了高达 62× 的 KV Cache I/O 降低和 3.3× 的解码加速，为高效长上下文 LLM 推理提供了新的最优解。**

</details>

---

### 2. [SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE Foundation Models](https://arxiv.org/abs/2607.24787)

**Authors**: Jinwei Kong, Runqi Meng, Fanyi Wang, Wentao Qiu, Haotian Hu, Yongjian Zhou, Zhenhua Ge  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.24787v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) models expand foundation model capacity through conditional expert activation, but their full expert pools remain difficult to deploy under limited accelerator memory. Although expert offloading alleviates memory pressure by moving inactive experts to host memory or s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE Foundation Models**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
在 **Sparse Mixture-of-Experts (MoE)** 模型中，虽然每个 token 只激活少量专家（sparse activation），显著降低了计算量，但**完整的专家池仍需在推理时可访问**。当部署于内存受限设备（如边缘设备）时，通常采用 **expert offloading**（将非活跃专家卸载到主机内存或存储中），但在推理过程中，**专家需求直到原生 top-K 路由完成后才可知**，导致以下瓶颈：

- **序列化执行路径**：隐藏状态计算 → 路由决策 → 专家加载 → 专家执行  
- **专家加载延迟暴露在关键路径上**，成为性能瓶颈。

现有方法存在局限：
- **量化/压缩**：减少传输体积，但不改变加载时机。
- **训练无关预取（如 FATE）**：依赖跨层路由规律性，在多模态或复杂任务下不稳定。
- **基于学习的预测器（如 Draft Model, ProMoE）**：可能干扰原始路由语义，影响模型输出。

---

### 🚀 提出的新方法：SpecPrefetch

SpecPrefetch 是一种 **参数高效、仅用于传输的专家预取框架**，其核心思想是：

> **解耦“传输预测”与“执行路由”**：用一个轻量级适配器（adapter）提前预测下一层可能被调用的专家，仅用于异步加载；最终执行仍由冻结的原生路由器决定。

#### 创新点：
1. **Transfer-only 预测范式**：
   - 预测结果**不影响模型输出**，只用于调度专家提前加载。
   - 错误预测不会改变模型行为（false positive 浪费带宽，false negative 触发按需加载），系统更鲁棒。

2. **共享轻量级 Adapter 架构**：
   - 在每一层共享一个低秩映射（low-rank adapter）来从当前层表示预测下一层专家优先级。
   - 参数量极小，训练成本低，易于部署。

3. **Window-aware 运行时调度器**：
   - 综合考虑预测置信度、缓存驻留状态、待传输队列和完成时间，动态决定哪些专家应被异步加载。
   - 支持在真实系统约束下实现最优资源利用。

---

### 🔍 相比现有方法的优势
| 方面 | SpecPrefetch | 现有方法 |
|------|--------------|--------|
| **是否改变路由语义** | ❌ 否（保留原生路由） | ✅ 是（部分方法会替代或微调路由） |
| **参数效率** | ⭐ 极高（仅 ~1.6M–13M trainable params） | ❌ 高（如 ProMoE 达 414M） |
| **预测目标** | 仅用于异步传输准备 | 多与执行绑定 |
| **鲁棒性** | 高（错误不影响输出） | 中（可能引入偏差） |
| **适用场景** | 内存/带宽受限边缘设备 | 主要面向服务器端优化 |

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集

#### **模型架构**
- **Qwen3-VL-30B-A3B**：纯路由专家结构（pure routed expert pool）
- **DeepSeek-VL2-Tiny**：共享+路由专家设计（shared + routed experts）

#### **评估基准（Benchmarks）**
分为两类工作负载：

| 类别 | 数据集 | 任务类型 |
|------|-------|---------|
| **VLM 工作负载** | OCRBench, ChartQA-Test, HallusionBench | 文本密集视觉理解、图表问答、幻觉敏感推理 |
| **LLM 工作负载** | GSM8K, HumanEval | 数学推理、代码生成 |

> 所有评估均使用**独立的测试集**，确保无数据泄露。

---

### ⚙️ 实验设置

#### **系统模拟环境**
- 使用 **2×NVIDIA A800-SXM4-80GB GPU** 进行离线仿真。
- 专家卸载至主机内存，通过模拟 I/O 延迟控制存储速度。
- 注入不同级别的 I/O 延迟（对应 NVMe、UFS、SD Card 等存储介质）以评估实际性能。

#### **真实设备验证**
- 在 **Qualcomm Snapdragon 8 Elite + Adreno 825 GPU** 上部署 DeepSeek-VL2-Tiny（4-bit量化）。
- 模拟冷启动（`drop_caches`）和热缓存场景，测量端到端吞吐量。

---

### 📊 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **Expert Recall @M (R@M)** | 下一层真实激活专家中，出现在前 M 个预测候选中的比例 | 衡量预取准确性 |
| **Average Recall (Avg.)** | 多个预算下的平均召回率 | 综合比较性能 |
| **Decoding Throughput (tokens/sec)** | 每秒生成 token 数 | 实际推理效率 |
| **Speedup** | 相对于 baseline 的加速比 | 系统级收益 |
| **Trainable Parameters** | 可训练参数总数 | 参数效率衡量 |

---

### 🔁 基线方法对比

| 基线 | 类型 | 描述 |
|------|------|------|
| **FATE** | 训练无关 | 利用相邻层 gate 信息进行跨层复用预测 |
| **Draft Model** | 学习型辅助模型 | 使用小型辅助网络预测专家 |
| **ProMoE** | 强大学习预测器 | 更大的预测模块，更强表达能力 |
| **MLP Predictor (消融)** | 替代结构 | 使用普通 MLP 替代 adapter 进行比较 |

所有方法统一采用 **transfer-only 协议**：预测仅用于预取，不参与最终路由选择。

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（来自 Table 1）

#### 在 **Expert Recall** 上的表现（越高越好）：

| 模型 | Benchmark | FATE | Draft Model | ProMoE | **SpecPrefetch** |
|------|----------|------|-------------|--------|------------------|
| Qwen3-VL | GSM8K | 88.27 | — | 89.94 | **91.86** ✅ |
| Qwen3-VL | HumanEval | 87.90 | — | 86.97 | **91.67** ✅ |
| Qwen3-VL | OCRBench | 91.23 | 91.52 | 93.35 | **94.48** ✅ |
| Qwen3-VL | ChartQA-Test | 90.56 | 92.31 | 94.39 | **94.63** ✅ |
| Qwen3-VL | HallusionBench | 91.90 | 92.31 | 93.75 | **94.55** ✅ |
| DeepSeek-VL2 | GSM8K | 91.66 | 86.39 | 85.73 | **91.18** ✅（仅次于 FATE） |
| DeepSeek-VL2 | HumanEval | 88.84 | 78.54 | 75.07 | **89.13** ✅ |
| DeepSeek-VL2 | OCRBench | 91.70 | 90.14 | 92.90 | **94.82** ✅ |
| DeepSeek-VL2 | ChartQA-Test | 90.85 | 91.03 | 93.97 | **95.23** ✅ |
| DeepSeek-VL2 | HallusionBench | 93.15 | 93.63 | 92.57 | **94.39** ✅ |

✅ **SpecPrefetch 在 10 个 model-benchmark 设置中拿下 9 项最佳平均召回率**！

---

### 🔍 消融实验结果

#### （1）**Predictor 设计消融（vs MLP Predictor）**

| 模型 | Benchmark | MLP Predictor (Avg.) | SpecPrefetch (Avg.) | 提升 |
|------|----------|------------------------|------------------------|------|
| Qwen3-VL | GSM8K | 86.92 | 91.86 | **+4.94%** |
| Qwen3-VL | HumanEval | 80.05 | 91.67 | **+11.62%** |
| DeepSeek-VL2 | GSM8K | 72.47 | 91.18 | **+18.71%** |
| DeepSeek-VL2 | HumanEval | 60.80 | 89.13 | **+28.33%** |

👉 表明 **gate-aware adapter 校准机制显著优于通用 MLP 映射**。

#### （2）**参数效率对比（Table 4）**

| 模型 | Single MLP | Draft Model | ProMoE | **SpecPrefetch** |
|------|------------|-------------|--------|------------------|
| DeepSeek-VL2-Tiny | 1.56M | 26.40M | 32.69M | **1.63M** |
| Qwen3-VL-MoE-30B | 24.38M | 19.19M | 414.45M | **12.95M** |

- SpecPrefetch 仅为 ProMoE 的 **3.1% 参数量**（在 Qwen 上），却取得更高 recall。
- 参数效率优势极为明显，适合边缘部署。

---

### 📱 真实设备性能提升（Figure 3 & 4）

在 **Snapdragon 8 Elite + Adreno 825 GPU** 上运行 DeepSeek-VL2-Tiny：

| 存储类型 | Compute-Optimized Baseline | **+ SpecPrefetch** | 加速比 | 吞吐提升 |
|--------|----------------------------|--------------------|--------|----------|
| Mid UFS | ~3.6 tokens/s | **~4.1 tokens/s** | 1.15× | **+15%** |
| Slow UFS | ~3.4 tokens/s | **~4.1 tokens/s** | 1.20× | **+20%** |
| SD Card | ~3.2 tokens/s | **~3.8 tokens/s** | 1.17× | **+17%** |

> 当 I/O 成为瓶颈时，SpecPrefetch 效果最显著。

#### 冷启动场景（Figure 5）
- 清除 page cache 模拟首次运行：
  - Baseline: 3.29 tps
  - Compute-Optimized: 3.25 tps（无改善）
  - **SpecPrefetch + Compute**: **3.76 tps** → **+16% 预取贡献，总提速 1.14×**

👉 说明：**仅靠计算优化无法解决冷加载问题，而预取能有效缓解。**

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **SpecPrefetch 实现了高精度、低开销的专家预取**：
   - 通过轻量 adapter 提前预测下一层专家，召回率显著高于各类基线。
   - 不改变原生路由逻辑，保证模型输出一致性。

2. **参数效率极高**：
   - 可训练参数远少于主流学习型预测器（如 ProMoE 的 3%），更适合移动端部署。

3. **在真实边缘设备上带来可观性能增益**：
   - 在 I/O 受限场景下，解码吞吐最高提升 **20%**。
   - 尤其在冷启动或慢速存储环境下效果突出。

4. **window-aware 调度器提升了系统实用性**：
   - 动态权衡预测置信度、缓存状态与传输可行性，实现高效资源利用。

---

### ⚠️ 局限性（Limitations）

1. **收益依赖运行时条件**：
   - 若缓存命中率高或存储速度快（如 NVMe），预取增益有限。
   - 性能提升取决于 transfer-compute overlap 是否充分。

2. **预测误差仍影响系统效率**：
   - False positive 浪费带宽和缓存空间。
   - False negative 仍需阻塞加载。

3. **单个共享 adapter 可能不足以建模复杂路由模式**：
   - 对高度异构的专业化专家结构适应性有限。

4. **调度策略局部最优**：
   - 当前调度器未考虑全局缓存管理、跨请求重用或多用户批处理。

5. **验证范围有限**：
   - 实测仅在 DeepSeek-VL2-Tiny 和移动平台完成，尚未扩展至更大 MoE 模型或服务器环境。

---

### 🔮 未来工作方向

- 设计 **分层或多头 adapter** 以更好捕捉层间差异。
- 开发 **联合缓存-预取优化策略**，支持跨请求专家重用。
- 探索 **自适应预取预算机制**，根据上下文动态调整 M。
- 扩展至 **多设备协同推理** 场景，支持分布式专家调度。
- 结合 **speculative execution** 与 **expert prefetching**，进一步拉长调度窗口。

---

## ✅ 总结

**SpecPrefetch** 提出了一种新颖且实用的 **transfer-only 专家预取范式**，通过一个**轻量共享 adapter** 实现对下一层专家需求的早期预测，并结合 **window-aware 调度器** 实现高效的异步加载。它在保持原生路由不变的前提下，显著提升了专家预取准确率，在真实边缘设备上实现了高达 **20% 的吞吐提升**，同时具备卓越的**参数效率**，为 **memory- and bandwidth-constrained MoE deployment** 提供了一个极具前景的解决方案。

</details>

---

### 3. [From Tokens to Watt-hours: Analytical Energy Estimation for LLM Inference on Modern GPUs](https://arxiv.org/abs/2607.26571)

**Authors**: Tina Vartziotis, Rodopi Kosteli, Elli Vartziotis, George Dasoulas, Michael Keckeisen, Konstantinos Skianis, Sotirios Kotsopoulos, Francesca Dominici  
**Category**: cs.LG  
**Published**: 2026-07-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.26571v1  

#### Abstract
The operational energy consumption of large language model (LLM) inference is becoming an increasingly important component of the environmental footprint of deployed AI systems. However, direct measurement of inference energy often requires hardware telemetry, power instrumentation, or infrastructur...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Tokens to Watt-hours: Analytical Energy Estimation for LLM Inference on Modern GPUs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLM）推理阶段的**运行能耗**已成为部署AI系统环境足迹的重要组成部分。然而，直接测量GPU推理能耗通常依赖硬件遥测、功耗仪器或特定基础设施监控，限制了其在跨平台比较、早期系统设计和可持续性报告中的应用。

本文旨在解决这一问题：  
👉 **如何在没有运行时功耗测量的情况下，对现代GPU上的LLM推理能耗进行透明、可复现且假设明确的估算？**

---

### ✅ 提出的新方法与创新思路

作者提出了一种**半分析式（semi-analytical）、经验校准的GPU级能耗估算框架**，用于估计LLM在NVIDIA H100类加速器上的推理能耗。该方法的核心创新包括：

#### （1）**结构化能量分解模型**
将总GPU能耗 $ E_{\text{GPU}} $ 分解为两个阶段和多个组件：
- **两阶段分离**：
  - **Prefill（提示预填充）**：处理输入prompt并构建初始KV Cache。
  - **Decode（自回归解码）**：逐个生成输出token。
- **多维度能量构成**：
  $$
  E_{\text{GPU}} = E_{\text{compute}} + E_{\text{memory}}
  $$
  其中内存部分进一步细分为：
  - 参数访问（Parameter-access）
  - KV Cache写入（KV-cache write）
  - 注意力读取（Attention-read）

#### （2）**参数缩放FLOP建模 + 内存流量校准因子**
- 使用标准Transformer FLOP公式（$ C = K \cdot N \cdot T $，其中 $ K=6 $）估算计算量。
- 引入三个**经验校准因子**以反映非理想行为：
  - $ \gamma(N) $：参数重用效率随模型增大而下降
  - $ S_{\text{attn}}(N) $：注意力相关的KV缓存读取开销（超线性增长）
  - $ \eta(N) $：全局HBM传输低效因子

#### （3）**支持不同粒度的能耗估算**
提供三种层级的输出：
- 请求级（per-request）
- Token级（input/output token）
- 组件级（compute vs memory）

#### （4）**简化版估算器（Parameter-only Estimator）**
当缺乏详细架构信息时，仅需模型参数量即可估算：
$$
E_{\text{out/token}} = \alpha_{\text{eff}} \cdot K \cdot N
$$
便于快速模型比较。

---

### ✅ 相比现有方法的优势

| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| 是否需要运行时测量 | 是（如NVML、电源计） | 否（纯分析） |
| 可移植性 | 差（绑定具体硬件/服务栈） | 高（模块化+可重校准） |
| 透明性 | 黑箱式报告 | 明确假设与分解机制 |
| 应用场景 | 实际部署后评估 | 设计阶段决策支持 |
| 成本敏感性分析能力 | 弱 | 强（支持green coding干预分析） |

> ✅ **优势总结**：本文方法填补了“完全实测”与“粗粒度碳核算工具”之间的空白，提供了**无需仪器、可复现、可解释的GPU级能耗估算方案**。

---

## 2. 核心实验方法和设置

### 📚 使用的模型集合（Model Set）
并非传统意义上的“数据集”，而是选取了一系列具有代表性的LLM，覆盖小（<3B）、中（3–30B）、大（>30B）参数范围，包括：
- 通用Decoder-only LLMs（如LLaMA 3.3 70B）
- 编码模型（Embedding models）
- 代码模型（DeepSeek-Coder V2）
- 视觉语言模型（Qwen-VL）
- 推理模型等

所有模型均基于公开配置（见Supplementary Table 5），关键参数包括：
- 参数数量 $ N $
- 层数 $ n_l $
- 隐藏维度 $ d_{\text{model}} $
- KV Cache维度等

---

### ⚙️ 实验设置

| 项目 | 设置说明 |
|------|----------|
| **目标硬件** | NVIDIA H100-class GPU（FP16/BF16精度） |
| **硬件系数**（来自Antepara et al. [21]） | 
| - $ \omega_{\text{TC}} $（tensor-core能效） | 0.52 pJ/FLOP |
| - $ \epsilon_{\text{HBM}} $（HBM每比特能耗） | 11.68 pJ/bit |
| **默认工作负载** | $ T_{\text{in}} = 500 $, $ T_{\text{out}} = 500 $ |
| **其他变量实验** | 固定输入长度 $ T_{\text{in}} = 100 $，变化 $ T_{\text{out}} $（从1到20,000） |
| **是否启用批处理** | 默认单请求；未显式建模continuous batching，但通过参数重用因子间接体现 |
| **校准依据** | 基于Caravaca et al. [23]发布的实测能耗数据进行拟合 |

---

### 📊 评估指标

| 指标 | 定义 |
|------|------|
| $ E_{\text{request}} $ | 单次请求总GPU能耗（Wh） |
| $ E_{\text{in/token}} $ | 输入token平均能耗（mJ/token） |
| $ E_{\text{out/token}} $ | 输出token平均能耗（mJ/token） |
| $ E_{\text{avg/token}} $ | 所有处理token的平均能耗 |
| 能量组成比例 | Compute / Memory / KV-read / Param-access占比 |
| 相对误差（vs measured） | $ |\hat{E} - E_{\text{measured}}| / E_{\text{measured}} $ |

---

### 🔁 基线方法对比
本文不直接对比传统ML模型，而是与**measurement-based studies**进行验证性对比，特别是：
- **Caravaca et al. [23]**：真实环境中测量LLM推理能耗的研究
- 对比对象为相同工作负载下（500 in / 500 out）的Wh/request值

此外，也隐含对比了以下两类方法：
- **纯FLOP-based估算**（忽略内存）
- **黑盒式碳排放工具**（如MLCO2、Green Algorithms）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）随输出长度扩展的能量趋势（Fig. 1 & 3）
- **Compute 和 KV-write**：几乎线性增长 → 符合预期
- **Attention-related memory traffic**：呈现**超线性甚至接近二次方增长**
  - 原因：每个新token需attend over整个上下文（$ \sim T_{\text{out}}^2 $）
  - 在长序列（>5000 tokens）时成为主导因素之一

> 🔍 发现：**长文本生成任务中，memory movement尤其是attention read可能超越compute成为瓶颈**

---

#### （2）随模型规模扩展的趋势（Fig. 2）
- 总能耗近似**线性增长**（log-log图上斜率≈1）
- 主导项是 **compute energy**（$ E_{\text{compute}} \propto N $）
- memory energy增长略快于线性，受 $ \gamma(N), S_{\text{attn}}(N), \eta(N) $ 影响
- 存在局部非单调现象 → 表明**架构差异影响显著**（如hidden dim、layer数非均匀缩放）

---

#### （3）Token级能耗估算（Table 3）
| 模型 | 参数量 | $ E_{\text{out/token}} $(mJ) | $ E_{\text{in/token}} $(mJ) | $ E_{\text{request}} $(Wh) |
|------|-------|-------------------------------|------------------------------|-----------------------------|
| Gemma (0.3B) | 0.3B | 0.96 | 1.15 | ~0.0006 |
| Qwen3 (32B) | 32B | 99.84 | 119.81 | 0.0527 |
| LLaMA 3.3 (70B) | 70B | 218.4 | 262.1 | 0.1707 |
| GPT-OSS (120B) | 120B | 374.4 | 449.3 | 0.1552 |

> 💡 观察：能耗与参数量基本呈线性关系（符合 $ E \propto N $），适合用于模型选型比较。

---

#### （4）与实测结果对比（Table 4）

| 模型大小（B） | 分析估算（Wh） | 实测值（Wh） | 相对误差 |
|--------------|----------------|---------------|-----------|
| 8            | 0.0118         | 0.00927       | +27.2%    |
| 24           | 0.0324         | 0.02628       | +23.4%    |
| 70           | 0.1707         | 0.1792        | -4.7%     |
| 72           | 0.1707         | 0.2333        | -26.8%    |

> ✅ 结果表明：估算值在多数情况下落在实测值±30%以内，尤其在70B模型上误差仅4.7%，说明模型具备良好的**数量级预测能力和趋势一致性**。

---

### 🔍 消融实验与归因分析（Fig. 3）

- **短序列生成**：compute占主导（>80%）
- **长序列生成**（Tout > 10k）：
  - attention-related memory traffic迅速上升
  - 成为主要内存开销来源
- **KV-cache write 和 parameter access**：在整个范围内贡献极小（<5%）

> 🔎 结论：**优化方向应随场景变化**：
> - 短文本 → 降低compute（如模型压缩）
> - 长文本 → 减少attention memory访问（如KV cache量化、稀疏注意力）

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLM推理能耗可通过分析建模有效估算**，即使无运行时测量也能获得合理近似。
2. **输入与输出token的能耗模式完全不同**：
   - Prefill阶段主要决定 $ E_{\text{in/token}} $
   - Decode阶段主导 $ E_{\text{out/token}} $
3. **内存移动（尤其是attention-related KV-cache reads）在长序列中起关键作用**，不能被忽略。
4. **总能耗大致随模型参数线性增长**，但存在因架构差异导致的局部偏差。
5. **提出的校准因子（$ \gamma(N), S_{\text{attn}}(N), \eta(N) $）能有效捕捉现实系统中的非理想行为**。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **仅限GPU侧能耗** | 不包含CPU、网络、冷却、PUE等系统级开销 |
| **未建模复杂系统效应** | 如tensor parallel通信、kernel fusion、调度延迟等 |
| **依赖经验校准** | 当前系数针对H100 + FP16/BF16 + 特定推理引擎，换平台需重新校准 |
| **简化了MoE、Multi-Query Attention等结构** | 当前模型主要面向dense decoder-only架构 |
| **忽略batching效率提升** | 虽提及但未显式建模连续批处理的影响 |

> ❗ 重要提醒：该方法**不是替代物理测量**，而是作为**设计阶段辅助工具**。

---

### 🔮 未来工作方向

1. **扩展至Quantized Models**（INT8, FP8, 4-bit GGUF等）
2. **集成Mixture-of-Experts（MoE）模型的能耗建模**
3. **引入Batching与Prefix Caching的显式建模**
4. **支持更多GPU架构（如AMD Instinct、Apple Silicon）**
5. **结合trace-driven仿真提升准确性**
6. **向系统级/数据中心级能耗推演延伸**（结合PUE、cooling overhead）

---

## ✅ 总结

本文提出了一个**透明、可复现、无需运行时测量的LLM推理能耗估算框架**，实现了从token到watt-hour的映射。它不仅可用于绿色AI研究中的模型比较，也为工程师在部署前评估不同模型、提示长度、生成策略的能耗提供了实用工具。

> 🌱 **核心价值**：推动**Green Coding**实践，在不影响性能的前提下做出更可持续的技术选择。

</details>

---

### 4. [Reasoning with Memory: A Temporal Granularity-Adaptive Framework for Training-Free Long Video Understanding](https://arxiv.org/abs/2607.24794)

**Authors**: Linghao Meng, Qiankun Li, Junyuan Mao, Pujin Liao, Zhicheng He, Enbo Zhang, Kun Wang, Yang Liu, Huazhu Fu, Yueming Jin  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.24794v1  

#### Abstract
While Multimodal Large Language Models (MLLMs) demonstrate superior generalization in fundamental video tasks, restricted context windows limit their long video understanding. To accommodate this constraint, models typically resort to keyframe selection. However, uniform sampling or static query-gui...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Reasoning with Memory: A Temporal Granularity-Adaptive Framework for Training-Free Long Video Understanding**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

当前的 **Multimodal Large Language Models (MLLMs)** 在长视频理解任务（LongVideoQA）中面临两大挑战：

- **上下文窗口限制**：标准 MLLMs 受限于输入 token 数量，无法处理长视频中的全部帧。
- **关键帧选择策略不足**：现有方法（如均匀采样或静态查询引导选择）往往忽略**时间上下文**，且无法适应不同问题所需的**时间粒度（temporal granularity）**，导致冗余采样或遗漏关键事件。

此外，许多先进方法依赖训练或微调，缺乏通用性和即插即用能力。

---

### ✅ 提出了什么新方法或新思路

作者提出 **ReMem**（**Reasoning with Memory**），一个**无需训练**（training-free）、**时间粒度自适应**的关键帧选择框架，用于提升 MLLMs 在长视频上的零样本推理能力。

其核心思想是利用“**记忆机制**”（memory mechanism）在两个层面进行自适应建模：

#### 🔹 双层级记忆增强适配（Dual-Level Memory-Augmented Adaptation）

1. **Query Level: Memory-Driven Question Parsing**
   - 利用 LLM 的**长期记忆**分析问题的时间粒度（temporal granularity）`g ∈ [0,1]`：
     - `g ≈ 0`：局部细粒度问题（如“某人做了什么动作？”）
     - `g ≈ 1`：全局粗粒度问题（如“视频中太阳出现了几次？”）
   - 同时从问题和候选答案中提取**语义实体**（semantic entities），增强文本信号质量。

2. **Video Level: Synergistic Dual-Semantic Frame Alignment + Structure-Aware Dynamic Frame Routing**
   - 构建**时空记忆图**（temporal-semantic memory graph），捕捉帧间的动态演化关系。
   - 设计**协同双语义对齐机制**，融合：
     - 静态视觉-语义相似性（Static Visual-Semantic Alignment, VSA）
     - 记忆增强的时序-语义对齐（Temporal-Semantic Alignment, TSA）
   - 通过**结构感知的动态帧路由**（Structure-Aware Dynamic Frame Routing, FR）将预算合理分配给不同事件簇，避免重复采样。

---

### ✅ 相比现有方法的优势

| 优势维度 | ReMem 表现 |
|--------|-----------|
| **无需训练** | 完全 plug-and-play，适用于任意 off-the-shelf MLLM，无参数调整 |
| **时间感知更强** | 显式建模时间依赖与因果结构，支持多粒度推理 |
| **抗冗余能力强** | 动态聚类与路由抑制局部高分段的过度采样 |
| **泛化性强** | 在多个 MLLM 和 benchmark 上均取得 SOTA |

> ❗️关键突破：首次将 LLM 的“认知先验”与视频的“内在结构记忆”结合，实现**语义-时间联合对齐**。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集

在四个主流的 **LongVideoQA benchmark** 上进行评估：

| 数据集 | 特点 |
|------|------|
| **LVBench** [35] | 极端长度视频（平均 68 分钟），强调长期依赖推理 |
| **LongVideoBench** [38] | 多模态交错理解，涵盖长时因果、计数等任务 |
| **MLVU** [51] | 多任务长视频理解，含短/中/长三种时长子集 |
| **Video-MME** [9] | 综合评测基准，测试模型在不同时间尺度下的鲁棒性 |

---

### ⚙️ 实验设置与评估指标

- **模型架构**：基于三个主流 Video-LLMs 测试：
  - LLaVA-Video-7B-Qwen2（64帧预算）
  - Qwen2-VL-7B-Instruct（32帧）
  - Qwen3-VL-8B-Instruct（32帧）

- **评估方式**：
  - 所有实验为 **zero-shot** 设置
  - 输入不含字幕（subtitle-free）
  - 采用**多项选择题准确率**（answer selection accuracy）作为主指标

- **实现细节**：
  - 使用 GPT-4o 进行问题解析
  - 使用 CLIP-L-14 提取图文嵌入
  - 初始帧序列以 1fps 均匀采样
  - 候选池大小 `N=150`，衰减常数 `λ=1e-4`

---

### 🔁 基线方法对比

| 类型 | 方法 |
|-----|------|
| **原始模型** | LLaVA-Video, Qwen-VL, VideoLLaMA 等 |
| **固定采样** | Uniform Sampling |
| **训练无关方法** | AKS [33], Q-Frame [47], FlexSelect [49], BOLT [24], MDP3 [32] |
| **训练相关方法\*** | GenS [39]\*, FrameOracle [15]\*, Selector [12]\* |

> 注：带 \* 为需训练的方法，其余为 training-free。

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（来自 Table 1）

| 方法 | LVBench ↑ | MLVU ↑ | LongVideoBench ↑ | Video-MME ↑ |
|------|----------|--------|------------------|-------------|
| **LLaVA-Video (baseline)** | 42.2 | 70.8 | 58.9 | 64.4 |
| **+ AKS** | — | — | 62.7 | 65.3 |
| **+ ReMem (ours)** | **54.5** (**+12.3**) | **77.3** | **67.1** (**+8.2**) | **69.2** |

> 💡 在 LLaVA-Video 上，ReMem 带来显著增益，尤其在极端长视频任务上表现突出。

| 方法 | LVBench ↑ | MLVU ↑ |
|------|----------|--------|
| **Qwen2-VL (baseline)** | 41.5 | 56.9 |
| **+ ReMem (ours)** | **51.7** | **72.8** |
| **Qwen3-VL (baseline)** | 42.7 | 63.6 |
| **+ ReMem (ours)** | **53.3** (**+10.6**) | **77.6** (**+14.0**) |

> ✅ ReMem 在不同规模和架构的 MLLM 上均带来一致提升，验证其**模型无关性**。

---

### 🔍 与训练方法的对比（体现 training-free 的优越性）

尽管 ReMem **不经过任何训练**，却超越多个强监督方法：

| 对比项 | 结果 |
|-------|------|
| vs. **GenS**（训练-based） | 在 MLVU 上，ReMem 达到 **72.8%** vs GenS 的 **66.9%**（+5.9%） |
| vs. **FrameOracle**（训练-based） | 在 MLVU 上，ReMem 达到 **77.6%** vs FrameOracle 的 **62.9%**（+14.7%） |

> ✅ 表明：**显式建模时空相关性 > 黑箱式训练学习**

---

### 🔧 消融实验结果（Ablation Study）

使用 **LLaVA-Video** 和 **Qwen2-VL** 进行消融（见 Table 2 & Fig. 3）：

| 组件移除 | LVBench ↓ | MLVU ↓ | LongVideoBench ↓ |
|--------|----------|--------|------------------|
| 完整 ReMem | 54.5 | 77.3 | 67.1 |
| 移除 **Entity Extraction (EE)** | 52.3 (-2.2) | 75.8 (-1.5) | 66.7 (-0.4) |
| 移除 **VSA** | 41.6 (-12.9) | 64.0 (-13.3) | 57.4 (-9.7) |
| 移除 **TSA** | 47.6 (-6.9) | 72.9 (-4.4) | 61.5 (-5.6) |
| 移除 **Frame Routing (FR)** | 51.9 (-2.6) | 73.6 (-3.7) | 61.7 (-5.4) |

> 🔺 发现：
- **VSA 是基础模块**：提供视觉-语言桥接，影响最大。
- **TSA 至关重要**：尤其在长视频中，能连接跨场景因果事件。
- **FR 提升多样性**：防止预算集中在单一事件上，增强时间覆盖。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **时间粒度感知至关重要**  
   不同问题需要不同的时间建模策略。ReMem 通过连续变量 `g` 实现**动态平衡**空间细节与时间上下文。

2. **记忆机制可有效桥接语义与结构**  
   - LLM 的 long-term memory 可用于解析问题意图；
   - 视频的结构记忆可通过聚类+图扩散建模，保留因果链。

3. **无需训练也能超越训练方法**  
   显式设计优于隐式学习——**良好的结构先验 + 自适应机制**足以击败大量需训练的关键帧选择器。

4. **动态帧路由优于 Top-K 选择**  
   单纯按分数排序会导致上下文冗余；而基于事件聚类的预算分配更高效。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **预处理开销略高** | 虽然推理阶段不变，但关键帧选择阶段引入额外计算（约 +8–10s） |
| **依赖外部 LLM** | 需使用 GPT-4o 等高性能 LLM 进行问题解析，可能影响部署成本 |
| **极端长视频仍具挑战** | 尽管优于现有方法，但在超过小时级视频上仍有信息压缩损失风险 |

---

### 🔮 未来工作方向

1. **轻量化记忆模块**  
   探索更高效的图构建与扩散机制，降低预处理延迟。

2. **端到端集成**  
   将 ReMem 中的 memory parsing 模块蒸馏进小型 LLM，实现完全本地化运行。

3. **扩展至其他模态**  
   如音频、传感器流等多模态长序列理解任务。

4. **探索在线自适应机制**  
   根据 MLLM 的中间反馈动态调整关键帧集合（closed-loop selection）。

---

## ✅ 总结

ReMem 是一项开创性的 **training-free、temporal granularity-adaptive** 长视频理解框架。它通过引入**双层级记忆机制**，实现了对问题语义与视频结构的深度对齐，在多个 benchmark 上达到 **state-of-the-art zero-shot performance**，并证明了**良好设计的结构化先验可以超越训练驱动的方法**。该工作为高效、可扩展的长视频理解提供了新的范式。

</details>

---

### 5. [Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes](https://arxiv.org/abs/2607.26627)

**Authors**: Tianyu Wang, Yuxuan Zhou, Wenbin Wang, Heng Li, Zikai Xiao, Junyuan Shang  
**Category**: cs.CL  
**Published**: 2026-07-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.26627v1  

#### Abstract
Speculative Decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose tokens that are subsequently verified in parallel by a larger target model. Recent approaches introduce lossy verification schemes to further improve efficiency by relaxing strict di...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文系统性地分析了 **Lossy Verification**（有损验证）在 **Speculative Decoding**（SD）中的机制、权衡与失败模式。当前许多声称高效的 lossy 方法（如 Medusa、SpecCascade、CoS）通过放松分布匹配要求来提升推理速度，但其实际加速效果往往被高估，且可能严重损害生成质量。

作者指出：
- 现有研究常将 lossy 方法与默认 decoding 基线比较，而忽略了其性能提升可能主要来自 **truncation sampling** 本身；
- 不同方法看似差异大，实则可归为两类统一机制；
- 缺乏对真实 **speed-quality trade-off** 的准确评估。

### 🔍 提出的新方法与新思路
#### （1）提出**统一分类框架**：所有 lossy verification 方法可分为两类
| 类别 | 代表方法 | 核心机制 |
|------|--------|---------|
| **Truncation-based Verification** | SpecCascade, Typical Acceptance (Medusa) | 接受 draft token 当且仅当它属于由 min-p 或 τ-sampling 定义的 allowed set |
| **Collaborative Verification** | Lenience-based Relaxation, CoS | 将 draft 和 target 分布进行插值，生成混合分布 |

> 这一分类揭示了不同方法间的“表面差异”背后共享的深层机制。

#### （2）识别关键设计原则
- 对于 **Truncation-based 方法**：存在根本性缺陷——**分布扭曲导致性能显著低于真正的 truncation sampling 基线**。
- 对于 **Collaborative 方法**：**控制 overshoot tokens（即 draft 概率远高于 target 的 token）是保持生成质量的关键**，而非简单的全局插值。

#### （3）构建诊断评估框架
设计了跨多个难度递增 benchmark 的测试方案，并强调应使用 **distribution-matched baseline** 而非默认 baseline 进行公平比较。

### 🆚 相比现有方法的优势
- **更准确的评估视角**：首次明确区分“来自 truncation 的收益”与“来自 verification 机制的收益”。
- **理论指导实践**：提出“抑制 overshoot”优于“均匀插值”，为后续算法设计提供清晰方向。
- **揭示隐藏风险**：指出在 EAGLE-3 等多草案树结构中，truncation-based 方法的质量退化会被放大。

---

## 2. 核心实验方法和设置

### 📚 数据集
选用四个具有挑战性的 benchmark，覆盖多种任务类型：
| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **MATH** | 数学推理 | 高中至竞赛级数学题，衡量逻辑与符号推理能力 |
| **MBPP+** | 代码生成 | Python 函数编写任务，评估 Pass@1 |
| **INCLUDE** | 多语言理解 | 包含 22 种语言的知识问答，测试跨文化语义理解 |
| **BFCL** | 工具调用（Agentic Tool Use） | 多工具并行调用场景，评估复杂指令执行能力 |

> 所有任务均具有一定难度，避免简单任务掩盖质量问题。

---

### ⚙️ 实验设置
#### 模型配置
- **主实验**：`Qwen2.5-72B`（target） + `Qwen2.5-0.5B`（draft）
- **EAGLE-3 实验**：`LLaMA-3.1-8B`（target） + 官方配套 draft model
- 所有模型均为 GPTQ 8-bit 量化版本，确保推理效率可控

#### 硬件平台
- 主要实验运行于双 NVIDIA A100（80GB）
- EAGLE-3 实验使用单 A6000（48GB）
- 图1 使用 H200 GPU

#### 评估指标
| 指标 | 含义 |
|-----|------|
| **Block Efficiency (BE)** | 每步平均接受的 token 数量，反映加速潜力（硬件无关） |
| **Decoding Speed (DS)** | 实际解码速度（tokens/sec），反映端到端效率 |
| **Accuracy / Pass@1** | 任务正确率，衡量生成质量 |

---

### 🔁 基线方法对比
| 方法类别 | 具体方法 | 对应 baseline |
|--------|--------|-------------|
| Truncation-based | SpecCascade | Min-p sampling on target |
| Truncation-based | Typical Acceptance (Medusa) | τ-sampling on target |
| Collaborative | Lenience-based Relaxation | Lossless SD |
| Collaborative | CoS | Lossless SD |
| 控制组 | Standard SD (Leviathan et al., 2023) | —— |

> 特别强调：truncation-based 方法必须与其对应的 truncation sampling 基线进行配对比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 2 & 3）

#### 在标准 SD 设置下（Table 2）
| 方法 | MATH Acc (%) | MBPP+ Pass@1 (%) | 平均 Acc Gap vs Baseline |
|------|---------------|------------------|--------------------------|
| Min-p sampling (baseline) | **76.51** | **75.87** | —— |
| SpecCascade | 75.63 | 75.88 | ↓ ~0.88 pp |
| τ-sampling (baseline) | **75.55** | **75.80** | —— |
| Typical Acceptance | 75.55 | 75.80 | ≈持平（但波动更大） |

> 结论：truncation sampling 基线略优或相当，说明 verification 机制未带来额外收益。

#### 在 EAGLE-3 设置下（Table 3）
| 方法 | MATH Acc (%) | INCLUDE Acc (%) | 平均 Acc Gap |
|------|----------------|------------------|--------------|
| Min-p sampling | **76.88** | 37.00 | ↓ -0.72 pp |
| SpecCascade | 76.16 | 33.27 | ↓ -1.68 pp |
| τ-sampling | 76.12 | **36.73** | ↓ -0.32 pp |
| Typical Acceptance | **69.84** | **27.91** | ↓ **-6.32 pp** |

> ⚠️ 严重退化！尤其在 INCLUDE 上，Typical Acceptance 下降达 **8.8 pp**，甚至低于原始 EAGLE-3 baseline。

---

### 🔍 消融实验结果（Table 1 & Table 6）

#### 协作类方法的关键因素分析（Table 1）
| 变体 | λ | BE | Pass@1 (%) |
|------|----|-----|------------|
| Adaptive Interpolation | 0.8 | 5.95 | 66.14 |
| **Overshoot Ceiling Only** | 0.8 | **5.54** | **75.13** |
| Lossless Baseline | — | 5.47 | 75.84 |

> 发现：**仅保留 overshoot ceiling 就能接近 lossless 性能**，而 adaptive interpolation 导致严重 trade-off。

#### 组合策略探索（Table 6）
| 策略 | Avg. BE | Avg. Pass@1 (%) |
|------|--------|------------------|
| Lossless SD | 5.42 | 75.66 |
| Truncation-only | 5.51 | 74.87 |
| Overshoot ceiling only | 5.58 | 75.33 |
| **Truncation + Overshoot Cap** | **5.62 (+3.7%)** | **75.66 (持平)** |

> ✅ 最佳组合：**使用 truncation set 控制 acceptance，同时对 set 外的 overshoot token 施加 p/l 上限**，实现加速无损。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Lossy Verification 可分为两大范式**：
   - **Truncation-based**：本质是强制接受 allowed set 内的所有 draft token，导致输出分布偏向 draft。
   - **Collaborative**：通过插值融合 draft 与 target 分布，牺牲一致性换取速度。

2. **Truncation-based 方法存在根本性缺陷**：
   - 其所谓“高性能”多源于 truncation sampling 本身的正向作用；
   - 实际上，由于分布扭曲，其性能**低于对应 truncation sampling 基线**；
   - 在 EAGLE-3 中该 gap 被显著放大（最高达 20 倍），表明其不适用于复杂验证结构。

3. **Collaborative 方法的核心在于 overshoot 控制**：
   - **Lenience-based relaxation 的有效性主要来自对 overshoot tokens 的硬截断（ceiling）**；
   - 自适应插值部分反而引入不必要的质量损失；
   - 因此，“选择性抑制 overshoot”优于“全局线性插值”。

4. **评估必须使用 distribution-matched baseline**：
   - 忽视这一点会误导社区对方法有效性的判断；
   - 应报告相对于 truncation sampling baseline 的净增益。

---

### ⚠️ 局限性（Limitations）
1. **模型泛化性待验证**：实验集中于 Qwen 和 LLaMA 系列，其他架构或 scale 是否适用尚不确定。
2. **任务范围有限**：聚焦于可自动评分的任务（reasoning/code/tool-use），开放生成与对话等主观任务未涵盖。
3. **理论假设较强**：分析基于特定 truncation/collaborative 规则，未必覆盖所有未来方法。
4. **硬件依赖影响实际加速**：报告的是 BE（block efficiency），实际 wall-clock speedup 受 serving stack 影响。

---

### 🔮 未来工作方向
1. 设计新型 verification 机制，显式建模并抑制 overshoot tokens；
2. 探索动态调整 truncation set 与 overshoot threshold 的联合优化；
3. 将本框架扩展至多模态 speculative decoding；
4. 构建面向开放生成任务的高质量人工评估 protocol；
5. 开发轻量级 runtime monitor 来检测 distributional drift 并触发 fallback。

---

> 💡 **一句话总结**：  
> 本文揭示了当前主流 lossy speculative decoding 方法的本质机制，指出 **truncation-based 方法因分布扭曲而存在性能陷阱，collaborative 方法的有效性关键在于对 overshoot 的控制**，呼吁社区采用更严谨的评估基准以推动真正有意义的加速技术发展。

</details>

---

### 6. [DIRECT: Direct Decoding for Efficient and Aligned Sequence Labeling with Large Language Models](https://arxiv.org/abs/2607.26891)

**Authors**: Yilei Wang, Jiaxin Gan, Kexuan Zhang, Ling Li, Wentao Zhang, Peichao Lai  
**Category**: cs.CL  
**Published**: 2026-07-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.26891v1  

#### Abstract
Sequence labeling is a fine-grained information extraction task, yet existing large language model-based approaches suffer from insufficient domain alignment and low inference efficiency. To address these issues, we propose DIRECT, a framework that addresses these issues through training-time optimi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DIRECT: Direct Decoding for Efficient and Aligned Sequence Labeling with Large Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 Large Language Models (LLMs) 的 **sequence labeling** 任务面临两大挑战：
- **Insufficient domain alignment**：模型输出难以严格对齐预期格式和标签体系，尤其在低资源场景下表现不稳定。
- **Low inference efficiency**：传统自回归生成方式导致推理速度慢，计算冗余严重。

### 🚀 提出的新方法：DIRECT
提出一种名为 **DIRECT (DPO-based Inference RECTification)** 的新框架，结合训练时优化与推理时干预，提升模型对齐性和效率。

#### 主要创新点：
1. **Domain-adaptive alignment optimization**
   - 在 Supervised Fine-Tuning (SFT) 后引入 **Direct Preference Optimization (DPO)**，利用偏好对 `(preferred, less-preferred)` 引导模型更贴近人类期望的输出行为。
   - 构造策略：选择 **BLEU 高但 F1 低** 的错误样本来作为负样本，增强对比学习信号。

2. **Controlled inference with constrained decoding**
   - 推理阶段强制模型遵循固定输出格式（如 `word(label)`），并通过限制解码空间到预定义的 **candidate set** 来减少非法输出。

3. **Template-filling + KV Cache reuse 提升效率**
   - 引入模板填充机制：仅让 LLM 生成 label tokens，其余部分由模板自动补全。
   - 利用 **KV Cache** 缓存已知上下文（如输入句子、括号等），避免重复计算，显著加速推理。

### 🔍 相比现有方法的优势
| 维度 | DIRECT | 现有方法（如 InstructUIE, GoLLIE, GNER） |
|------|--------|------------------------------------------|
| 对齐性 | ✅ 显著提升输出格式一致性与标签准确性 | ❌ 输出常缺失、错位或不符合规范 |
| 效率 | ⚡ 最高可达 **9× 更快推理速度** | ⏱️ 全序列自回归生成，效率低下 |
| 性能 | 🏆 多数数据集达到 SOTA 或第二优 | 📉 表现波动大，尤其在小样本下 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
共评估 **8个数据集**，涵盖中英文 NER 和 POS 任务：

| 类型 | 数据集 | 语言 | 说明 |
|------|-------|------|------|
| NER | Weibo, Youku, Taobao, Resume | 中文 | 社交媒体、电商、简历等真实场景命名实体识别 |
| NER | CoNLL03, MIT-Movie | 英文 | 新闻与电影领域实体识别 |
| POS | UD, CTB6 | 中文 | 中文词性标注任务 |

> 所有实验均在 **low-resource setting** 下进行：从训练集中随机采样 $ K \in \{250, 500, 1000\} $ 样本用于训练。

### 📊 实验设置与评估指标
- **Backbone Models**：
  - `LLaMA-3.1-8B-Instruct`
  - `GLM4-9B-Chat`
- **Baseline 方法对比**：
  - **InstructUIE** (Flan-T5/mT5-based)
  - **GoLLIE** (Code-LLaMA-7B)
  - **GNER** (LLaMA-7B)
- **评估指标**：
  - 主要使用 **F1-score**（token-level）
  - 推理时间分析（seconds）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）

| 数据集 | DIRECT (GLM4-9B-Chat) 最佳 F1 | 相对最优基线提升幅度 |
|--------|-------------------------------|---------------------|
| Youku ($K=1000$) | **79.31** | +1.34 vs InstructUIE |
| Taobao ($K=1000$) | **78.58** | +2.60 vs InstructUIE |
| Weibo ($K=1000$) | **70.31** | +1.31 vs InstructUIE |
| Resume ($K=1000$) | **95.94** | +1.96 vs GoLLIE |
| CoNLL03 ($K=1000$) | **89.96** | +2.51 vs GoLLIE |
| MIT-Movie ($K=1000$) | **87.66** | +15.32 vs InstructUIE! |
| UD ($K=1000$) | **89.76** | +2.77 vs GoLLIE |
| CTB6 ($K=1000$) | **89.59** | +1.17 vs GoLLIE |

> 💡 **平均提升达 0.37% ~ 14.72%**，在 MIT-Movie 上尤为显著（+13~15%），表明其在复杂语义边界识别上的优势。

### 🔁 消融实验结果（Table 2，$K=1000$ 设置）

| 模型变体 | 影响说明 | 性能变化趋势 |
|---------|--------|-------------|
| **w/o DPO** | 移除 DPO 训练 | 所有数据集上轻微下降（如 MIT-Movie ↓0.23） |
| **w/ SFT only** | 仅做 SFT，无 DPO & 无 inference rectification | 显著退化（MIT-Movie 下降超 20%！） |

> ✅ 结论：**DPO + inference rectification 双管齐下是性能提升的关键**。

### ⏱️ 推理效率对比（Figure 2）
- 测试环境：单张 NVIDIA L40 GPU，batch size=1
- 输入：CTB6 数据集中 10 句，平均长度 192 tokens

| 方法 | 平均推理时间（秒） | 相对 DIRECT 耗时倍数 |
|------|--------------------|------------------------|
| DIRECT | **32.86s** | ×1.0 |
| GNER | 309.44s | ≈ **9.4× 更慢** |
| GoLLIE | 1,012.07s | ≈ **30.8× 更慢** |
| InstructUIE | 1,468.00s | ≈ **44.7× 更慢** |

> ✅ **DIRECT 实现高达 9× 的推理加速**，得益于 template-filling 和 KV Cache 复用。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DPO 显著提升任务对齐能力**  
   通过构建高质量偏好对（high-BLEU / low-F1 错误样本），引导模型学会区分“看似合理但实际错误”的输出，从而更好适应 sequence labeling 的精确要求。

2. **推理控制机制大幅提升鲁棒性与可控性**  
   固定格式 + 候选集约束有效防止非法输出，案例研究显示 DIRECT 能完整识别所有实体并正确标注标点符号，而其他方法普遍存在漏标、错标问题。

3. **模板填充 + KV Cache 复用极大提升效率**  
   模型只需生成 label tokens，其余内容由模板填充并复用缓存，大幅减少重复 attention 计算，实现高效部署。

### ⚠️ 局限性
- 当前方法依赖于 **预定义标签集合**，难以直接扩展至开放域 extraction 场景。
- DPO 数据构造依赖 SFT 模型生成多样性错误，若初始模型太强或太弱，可能影响偏好对质量。
- 模板设计需人工参与，通用性受限于特定输出格式。

### 🔮 未来工作方向
- 将 template-filling 思路推广至更多 structured generation 任务（如 relation extraction, event detection）。
- 探索自动构建 preference pairs 的方法，降低人工干预。
- 结合 active learning，在低资源场景下动态选择最有价值样本进行标注与训练。

---

## ✅ 总结一句话
> **DIRECT 通过 DPO 训练 + 推理时格式控制 + KV Cache 复用，在保持 SOTA 性能的同时实现了高达 9× 的推理加速，为 LLM-based sequence labeling 提供了一种高效且对齐良好的解决方案。**

</details>

---

### 7. [Equilibrium Training of Energy-Based Models with Parallel Trajectory Tempering](https://arxiv.org/abs/2607.27077)

**Authors**: Nicolas B\'ereux, Aur\'elien Decelle, Cyril Furtlehner, Beatriz Seoane  
**Category**: cs.LG  
**Published**: 2026-07-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.27077v1  

#### Abstract
Energy-Based Models (EBMs) provide an interpretable framework for generative modeling of scientific data, but poor Markov Chain Monte Carlo mixing often limits their reliability. We introduce a training algorithm based on Parallel Trajectory Tempering (PTT), which exploits the continuity of the opti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Equilibrium Training of Energy-Based Models with Parallel Trajectory Tempering*

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

- **EBM训练中的非平衡采样问题**：Energy-Based Models (EBMs) 在最大似然训练中依赖 MCMC 生成模型样本，但随着训练进行，能量景观变得复杂，导致 Markov Chain Monte Carlo (MCMC) 混合缓慢、难以达到热平衡（equilibration），从而引发梯度偏差、优化不稳定、过拟合和模式崩溃等问题。
- **Persistent Contrastive Divergence (PCD)** 虽被广泛使用，但在高维、多模态或数据稀疏场景下容易失效——链会陷入局部模式或训练样本附近，无法有效探索整个分布。

### 🔧 **提出了什么新方法或新思路**

提出了一种名为 **Parallel Trajectory Tempering (PTT)** 的新型训练算法：

- **核心思想**：借鉴 Parallel Tempering (PT)，但不是在温度空间上构建副本（replica ladder），而是在**学习轨迹上的不同模型快照之间交换配置**（Hamiltonian-exchange MCMC）。
- 利用训练过程中模型参数演化的**连续性**，确保相邻检查点之间的分布具有足够重叠，从而实现高效的 replica exchange。
- 结合 **reservoir sampling** 和 **adaptive optimization**，避免对所有历史模型持续模拟，显著降低计算开销。

### 🆚 **相比现有方法的优势**

| 方面 | PTT 的优势 |
|------|-----------|
| **采样质量** | 维持全程均衡采样（equilibrium sampling），提供无偏梯度估计 |
| **稳定性** | 避免 PCD 中常见的链冻结、模式坍塌和突然性能下降 |
| **效率** | 计算成本与 PCD 相当，远低于传统 PT 或 AIS 方法 |
| **附加功能** | 免费获得：<br>• 准确的 log-likelihood 估计<br>• 热化时间 $T_{\text{exp}}$ 和自相关时间 $T_{\text{int}}$<br>• 可靠的模型评估与收敛诊断 |
| **鲁棒性** | 在小样本、强多模态、结构化数据上表现优异 |

---

## 2. 核心实验方法和设置

### 📚 **使用的数据集**

论文在多个科学领域的典型数据集上进行了验证，涵盖离散、混合型、高维且常具挑战性的结构：

| 数据集 | 类型 | 特点 |
|-------|------|------|
| **2D Ising Model** | 物理系统（格点自旋） | 多模态、临界相变行为，用于检验是否能恢复正确热力学统计量 |
| **Human Genome Dataset (HGD)** | 基因组二进制数据 | 强聚类结构、地理人群分层、样本少（~1k）、隐私敏感 |
| **B-Lactamase (PF13354)** | 蛋白质序列多重比对（MSA） | 生物学相关性高，用于接触预测任务 |
| **Neural Recordings (Neuropixels)** | 小鼠神经元活动记录 | 时间相关性强、高维度（2028 neurons）、稀疏激活 |
| **Medical Recommendation Dataset** | 医疗推荐表格数据 | 异构类别变量、症状-疾病-治疗依赖关系复杂 |

---

### ⚙️ **实验设置和评估指标**

#### ✅ 模型架构
- 主要采用 **Restricted Boltzmann Machines (RBMs)** 作为 EBM 的代表。
- 对比方法包括：
  - **PCD-RBM**（标准基线）
  - **Bayesian Flow Networks (BFNs)**（state-of-the-art 深度生成模型）
  - **edDCA**（用于蛋白质序列建模的传统方法）

#### ✅ 评估指标
| 指标 | 描述 |
|------|------|
| **Test Log-Likelihood** | 衡量泛化能力，通过 PTT 递归估计 $ \log Z $ 得到准确值 |
| **Two-/Three-body Correlations** | 检查低阶统计匹配程度（scatter plot + MSE） |
| **Earth Mover’s Distance (EMD)** | 分布间距离度量 |
| **PRIVET** | 基于最近邻距离分布判断欠拟合/过拟合/记忆化 |
| **Binder Cumulant & Susceptibility Scaling** | 物理系统中检测临界现象的能力 |
| **Positive Predictive Value (PPV)** | 蛋白质接触预测准确性（top-k couplings 中真实接触比例） |
| **Replica Diffusion / $T_{\text{exp}}$** | 诊断采样器是否处于平衡状态 |

#### ✅ 实验协议
- 40% 数据留作测试集
- 模型选择依据为 **最高测试对数似然**
- 所有生成样本均经过严格热化验证后再保存

---

## 3. 主要实验结果和性能指标

### 📊 **关键性能数据与对比结果**

#### ✅ **Ising 2D 模型**
- **PCD 失败于低温区域**：
  - 出现**单侧磁化模式坍塌**（仅负磁化）
  - 低估断开磁化率 $\chi_{\text{dis}}$，偏离理论标度律
- **PTT 成功复现双峰分布与临界行为**：
  - Binder 累积量交叉点精确对应临界温度
  - $\chi_{\text{dis}}$ 完美符合有限尺寸标度
- **效率更高**：PTT ladder 在低温下饱和增长，而标准 PT 所需温度副本随 $\beta$ 指数增加

> ➤ 图 2 显示 PTT 在物理保真度上全面超越 PCD

---

#### ✅ **Human Genome Dataset (HGD)**
- **PCD 与 BFN 均出现严重模式坍塌或权重失衡**
- **PTT-RBM 完整还原主成分空间中的多模态结构**
- **PRIVET 分析显示**：
  - BFN：明显**欠拟合**
  - PCD-RBM：强烈**过拟合特定簇**
  - PTT-RBM：几乎完美匹配参考分布 → **最佳泛化**
- 自由能最小值的层次聚类揭示出大陆与亚群结构，无需标签监督

> ➤ 图 3 展示 PTT 在遗传结构捕捉上的优越性

---

#### ✅ **Protein Family (B-Lactamase)**
- **两体/三体相关误差最小**：PTT 接近测试集基准水平
- **EMD 最低**：生成分布最接近真实数据
- **PRIVET 显示轻微欠拟合但仍优于其他方法**
- **接触预测 PPV 更高**：尤其在 Top-50~Top-100 范围内保持更优精度

> ➤ 图 4 表明 PTT 不仅生成质量高，还能提取更有意义的残基相互作用

---

#### ✅ **Neural Recordings**
- **PCD 性能高度依赖平行链数量**：
  - < 5000 chains 时显著偏差
  - 即使使用 1000 chains，也无法恢复三点相关性
- **PTT 稳定且高效**：
  - 在任意链数下均保持低误差
  - 正确还原同步放电数 $K$ 的分布
- Pearson 相关系数达 0.99（vs. PCD ~0.89）

> ➤ 图 5 证明 PTT 对采样资源不敏感，更具实用性

---

#### ✅ **Medical Recommendation Data**
- **PTT 达到更高的测试 LL**，且训练过程稳定
- **PCD 出现异常动态波动**
- 投影显示 PTT 能再现主要数据流形，而 BFN 生成不现实样本
- 两体相关性重建更准确

> ➤ 图 6 展示其在异构表格数据上的适用性

---

#### ✅ **消融实验与失败分析（MNIST 子集）**

- 在极小数据集（如 M=100 的 binarized MNIST）中：
  - **PCD 出现“训练 LL 崩溃”**：即使训练损失也急剧下降
  - 同时伴随 $T_{\text{exp}}$ 暴增 → 链完全冻结
  - Gibbs 轨迹迅速陷于训练样本附近 → **记忆化**
- **PTT 则表现为标准过拟合曲线**，无突发性崩溃
- 表明 PTT 能维持采样有效性，即使在极端小样本条件下

> ➤ 图 7 揭示 PCD 的根本缺陷源于采样失衡

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **EBM 训练失败的根本原因在于采样失衡**，而非表达能力不足。
2. **PTT 实现了真正意义上的均衡最大似然训练**，解决了长期存在的“负阶段”难题。
3. **利用学习路径的连续性构造 replica ladder 是高效且自然的设计**，优于基于温度的 PT。
4. **结合 reservoir sampling 后，PTT 的计算开销与 PCD 相当**，使其成为实用替代方案。
5. **紧凑的 RBM 在科学数据上可超越复杂的深度生成模型（如 BFN）**，前提是训练得当。
6. **PTT 提供丰富的内置诊断工具**（log-likelihood, $T_{\text{exp}}$, swap rate），极大提升可解释性和调试便利性。

---

### ⚠️ **方法的局限性**

- 当前实现主要针对 RBMs，扩展至更深或更复杂 EBMs（如 CNN-EBM）尚需工程适配。
- Reservoir 需存储 $N_{\text{res}}$ 个样本，在极高维数据中可能占用较多内存（但实践中可控）。
- 新 checkpoint 添加策略依赖 swap acceptance rate，需合理设定阈值。
- 并行开销仍存在，尽管总体优于 PT/AIS。

---

### 🔮 **未来工作方向**

1. **推广至其他 EBM 架构**：如 Convolutional EBMs、Graph EBMs 等。
2. **在线 ladder 更新机制**：动态删除早期冗余 checkpoint 以节省资源。
3. **与其他加速技术结合**：如 low-rank preconditioning、importance sampling。
4. **应用于更大规模生物医学问题**：如全基因组生成、单细胞调控网络推断。
5. **将 PTT 作为通用采样器嵌入其他框架**：如用于 variational inference 或贝叶斯神经网络。

---

## ✅ 总结一句话

> **PTT 通过在学习轨迹上实施并行退火，首次实现了高效、稳定、可诊断的均衡 EBM 训练，在多种科学数据上显著超越主流方法，重新确立了 EBMs 在可解释科学建模中的竞争力。**

</details>

---

### 8. [How Small Can You Go? A Controlled Study of LoRA Rank, Target Modules, and Quantization Trade-offs for Text-to-SQL on a 60M-Parameter Model](https://arxiv.org/abs/2607.25583)

**Authors**: Mahendra Singh Rathor, Anagheem Azzam  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.25583v1  

#### Abstract
Parameter-efficient fine-tuning (PEFT) and low-bit quantization are now standard tools for adapting language models under tight compute budgets, yet their interaction is most often studied on billion-parameter models where the design space is expensive to explore. We ask a complementary question: on...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*How Small Can You Go? A Controlled Study of LoRA Rank, Target Modules, and Quantization Trade-offs for Text-to-SQL on a 60M-Parameter Model*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前关于 **Parameter-Efficient Fine-Tuning (PEFT)** 和 **低比特量化**（如 QLoRA）的研究大多集中在十亿参数以上的大模型上，其设计空间探索成本高昂，难以进行系统性、可复现的消融研究。  
本文提出一个互补性问题：  
> 在一个小规模、完全可复现的设定下（60M 参数的 T5-small + WikiSQL），每个效率调节“旋钮”（rank、模块选择、精度）究竟会带来多少任务准确率的损失？  

目标是将模型适应过程视为**在参数量、内存、延迟等约束下的多目标优化问题**，而非单纯的准确率最大化。

---

### 🚀 提出的新思路与创新点
- **首次在小模型上对 LoRA 的多个维度进行受控单变量研究**：
  - LoRA rank（`r ∈ {2,4,8,16,32}`）
  - 被适配的模块（`{q,v}`, `{q,k,v,o}`, `{q,v}+FFN`）
  - 数值精度（FP16, INT8, NF4）
- **联合报告任务准确率与四项系统级指标**：
  - 可训练参数数量（Trainable params）
  - 训练峰值显存（Peak memory）
  - 推理延迟（Latency）
  - 吞吐量（Throughput）
- **引入帕累托前沿分析（Pareto analysis）**，明确指出不同资源约束下的最优配置。
- **强调可复现性**：公开所有代码、配置文件和训练日志，完整实验可在单张 T4 GPU 上约 35 分钟内完成。

---

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **研究视角** | 不追求 SOTA 性能，而是聚焦于“效率-准确性”的权衡关系，更具部署指导意义 |
| **实验设计** | 单变量控制 + 多随机种子验证，结果更可靠 |
| **适用场景** | 特别适合边缘设备、预算有限或快速原型开发等资源受限环境 |
| **透明度** | 完全开源，便于后续研究复现和扩展 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **WikiSQL**：标准的单表 text-to-SQL 基准数据集
  - 输入：自然语言问题 + 表格头信息
  - 输出：对应的 SQL 查询语句
  - 使用 5,000 条训练样本，500 条验证样本（便于单卡快速遍历）

### ⚙️ 实验设置
- **基础模型**：`T5-small`（60M 参数，encoder-decoder 架构）
- **任务形式化**：
  ```
  translate English to SQL: <Q>table:<H1|H2|...>
  ```
- **训练细节**：
  - 优化器：AdamW
  - 学习率：5e-4
  - Batch size：8
  - Epochs：3
  - 序列长度：input 256 / output 128
  - Beam search：num_beams=4
  - LoRA 设置：dropout=0.1，bias 冻结，缩放系数 α=2r
- **硬件平台**：NVIDIA T4 GPU（16GB VRAM），通过 Kaggle Notebooks 运行

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Exact-match Accuracy (EM)** | 生成 SQL 与黄金 SQL 字符串完全匹配（忽略大小写和空格） |
| **Execution Accuracy (Exec)** | 执行结果是否一致（仅针对可执行的 25 个样本子集，方差较高） |
| **Trainable Parameters** | 可训练参数总数（LoRA 中仅为 adapter 参数） |
| **Peak Training Memory** | 训练过程中最大 GPU 显存占用 |
| **Inference Latency & Throughput** | 单次推理耗时与每秒 token 数 |

---

### 🔁 基线方法对比
| 类型 | 配置 |
|------|------|
| **Zero-shot** | 未经微调直接推理 |
| **Full Fine-tuning** | 全参数微调（60.5M 参数）作为上限参考 |
| **LoRA Baseline** | `r=8`, `{q,v}`, FP16 作为默认对照组 |
| **QLoRA Variants** | INT8 和 NF4 4-bit 量化版本，附加 LoRA (`r=8`, `{q,v}`) |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2，seed=42）

| 配置 | EM (%) | Exec (%) | Trainable Params | Params (%) | Mem (GB) | Time (s) |
|------|--------|----------|------------------|------------|-----------|-----------|
| Full FT | 71.2 | 84.0 | 60.5M | 100% | 2.31 | 229.6 |
| LoRA r=16 ({q,v}) | **59.6** | 80.0 | 589,824 | **0.97%** | **1.60** | 187.1 |
| LoRA r=32 ({q,v}) | 60.4 | 80.0 | 1.18M | 1.91% | 1.61 | 189.8 |
| LoRA {q,k,v,o} | 59.2 | 76.0 | 589,824 | 0.97% | 1.69 | 239.9 |
| LoRA {q,v}+FFN | 58.6 | 80.0 | 786,432 | 1.28% | 1.74 | 228.1 |
| QLoRA INT8 | 52.8 | 72.0 | 294,912 | 0.48% | **0.60** | 595.3 |
| QLoRA NF4 | **53.2** | 76.0 | 294,912 | 0.65% | **0.60** | **441.8** |

> 注：LoRA r=16 已恢复 **83.7%** 的全微调准确率，仅用 **<1% 参数** 和 **31% 更少显存**

---

### 🔍 与基线方法的对比结果
- **LoRA vs Full Fine-tuning**：
  - 准确率差距：`71.2% → 59.6%`（↓11.6 pp）
  - 参数减少：`100% → 0.97%`
  - 显存降低：`2.31 GB → 1.60 GB`（↓31%）
- **QLoRA vs FP16 LoRA**：
  - 准确率轻微下降：`53.4% → 52.8%/53.2%`（仅 ↓0.2–0.6 pp）
  - 显存大幅压缩：`1.59 GB → 0.60 GB`（↓62%）
  - 但推理延迟上升（T4 上 dequantization 开销大）

---

### 🔬 消融实验结果

#### ✅ LoRA Rank 影响（r=2 到 r=32）
- `r=2`: 38.6%
- `r=8`: 53.4%
- `r=16`: 59.6%
- `r=32`: 60.4%（仅提升 0.8 pp，参数翻倍）
- ➤ **结论：r=16 达到饱和，继续增加 rank 收益极小**

#### ✅ 模块扩展影响（固定 r=8）
- `{q,v}`: 53.4% @ 344K params
- `{q,k,v,o}`: 59.2% @ 590K params（↑5.8 pp，↑71% 参数）
- `{q,v}+FFN`: 58.6% @ 786K params（不如前者高效）
- ➤ **结论：扩展模块不如提高 rank 高效；key 和 output 投影增益有限**

#### ✅ 量化影响（QLoRA）
- INT8/NF4 准确率接近 FP16 LoRA（r=8）
- 显存从 1.59 GB 降至 **0.60 GB**
- NF4 推理快于 INT8（得益于 double quantization）
- ➤ **结论：量化带来显著内存节省，适合内存受限部署**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LoRA rank 存在早期饱和现象**：
   - 在 T5-small/WikiSQL 设定下，`r=16` 即为最佳点，`r>16` 无明显收益。
2. **rank 提升比模块扩展更高效**：
   - 增加 rank 比添加更多模块（如 k/o/FFN）能以更低参数代价获得更高准确率。
3. **LoRA adapter 显存开销可忽略**：
   - 所有 LoRA 配置显存差异 <0.02 GB，主要消耗来自冻结主干和优化器状态。
4. **QLoRA 实现极致内存压缩**：
   - NF4 量化仅需 **0.60 GB 显存**，相比全微调节省 **74%**，准确率仅降不到 1 个百分点。
5. **帕累托最优配置明确**：
   - **无严格内存限制时**：LoRA r=16 on `{q,v}` 是 Pareto 最优
   - **强内存限制时**：QLoRA NF4 是首选方案

---

### ⚠️ 局限性
- **任务简单性**：WikiSQL 是单表任务，复杂度远低于 Spider 或 BIRD，结论可能不适用于多表 JOIN 场景。
- **模型规模限制**：仅基于 T5-small（60M），不能推广至更大或不同架构模型。
- **硬件依赖性**：QLoRA 在 T4 上推理较慢（dequantization 开销），在支持 Tensor Core 的 A100/H100 上表现更好。
- **执行准确率样本太少**：仅 25 个可执行样例，统计意义有限。

---

### 🔮 未来工作方向
- 将该方法论扩展到：
  - 更复杂的 text-to-SQL 基准（如 Spider、BIRD）
  - 其他小型模型架构（如 TinyBERT、DistilT5）
  - 其他 NLP 任务（如摘要、翻译）
- 探索自动化搜索 Pareto 最优 PEFT 配置的方法
- 结合编译优化进一步降低 QLoRA 推理延迟
- 研究量化对小模型训练动态的影响机制

---

## 总结一句话
> 本论文通过对 T5-small 在 WikiSQL 上的 LoRA 和量化策略进行全面、可控、可复现的消融研究，揭示了在小模型场景下 **“r=16 + {q,v}” 是最佳适配配置**，而 **QLoRA NF4 则是在极端内存限制下的理想选择**，为资源受限环境下的高效模型适配提供了清晰的设计指南。

</details>

---

### 9. [StrataCL: Fabric-Native Communication Library for Production Supernodes](https://arxiv.org/abs/2607.26444)

**Authors**: Tiancheng Hu, Jin Qin, Yuzheng Wang, Ke Liu, TangShengsheng Li, Sheng Wang, Zhongzhe Hu, Tianlun Hu, Wei Wang, Lijun Li, Jingbin Zhou, Xiaoming Bao, Hongwei Sun, Jieru Zhao, Huimin Cui, Tao Xie, Chenxi Wang  
**Category**: cs.DC  
**Published**: 2026-07-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.26444v1  

#### Abstract
Modern distributed AI workloads run across hundreds of accelerators, making communication a major bottleneck. Existing communication libraries remain largely buffer-centric because user and communication buffers are managed separately, causing redundant data copies or costly user-buffer registration...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*StrataCL: Fabric-Native Communication Library for Production Supernodes*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现代大规模分布式 AI 工作负载（如 LLM 推理与训练）在数百个加速器上运行，**通信成为主要性能瓶颈**。传统通信库（如 NCCL、HCCL）采用“buffer-centric”设计，用户缓冲区（user buffer）与通信缓冲区（communication buffer）分离管理，导致以下问题：

- **冗余数据拷贝**：数据需先从用户缓冲区复制到内部通信缓冲区，再进行传输，增加延迟并消耗额外 HBM。
- **注册开销大**：虽然已有方法（如 HCCL-zerocopy、NCCL UBR）支持 user-buffer direct communication，但依赖 **just-in-time 注册**，在动态分配场景（如 MoE 推理）中注册延迟会严重拖慢通信路径。

此外，在新兴的 **supernode 架构**（如华为 CloudMatrix384）上，传统通信算法（如 ring、recursive halving）因同步开销高、无法充分利用高带宽低延迟互连而不再高效。

---

### 提出了什么新方法或新思路

StrataCL 是一个面向生产级 supernode 的 **zero-redundancy、fabric-native 通信库**，提出两大核心技术：

#### （1）**Registration-on-Allocation**  
- **核心思想**：利用内存分配与首次通信之间的长间隔（通常 >2.6 秒），在内存分配后**异步注册**物理内存为远程可访问，将注册操作移出通信关键路径。
- **实现机制**：
  - 拦截 `aclrtMalloc` 等内存 API，触发后台注册。
  - 使用 **shadow virtual addressing**：每个 NPU 分配独立虚拟地址空间，所有对端 NPU 在相同虚拟地址映射源 NPU 的物理 HBM，避免通信时的地址转换开销。
  - 支持 VMM API-based allocator（如 PyTorch 的 expandable-segment allocator），通过增量更新处理运行时 remapping。

#### （2）**Fabric-Native 通信算子设计**
- **Full-Mesh 编程抽象**：替代多阶段算法（如 ring），使用单阶段并发远程读写，减少同步开销。
- **Workload-Balanced NPU-Core Partitioning**：
  - 将通信任务划分为细粒度 transfer units。
  - 建模不同拓扑层级（die-to-die, intra-node, inter-node）的延迟与带宽差异。
  - 使用 LPT-style 调度器分配任务，最小化最大完成时间（makespan），缓解 core-level 长尾问题。
- **NPU-Driven SDMA Offloading**：
  - NPU 核心仅提交 SDMA 描述符，由专用 SDMA 引擎异步执行数据搬运。
  - 释放 NPU 核心用于计算，提升 compute-communication overlap 效率。

---

### 相比现有方法的优势

| 维度 | 传统方法（HCCL / NCCL） | StrataCL |
|------|--------------------------|---------|
| 数据拷贝 | 存在 staging copy | **零冗余拷贝** |
| 用户缓冲区注册 | Just-in-time，阻塞关键路径 | **异步注册，透明无感** |
| 地址转换 | 每次通信需 peer-to-peer 映射 | **统一虚拟地址，免转换** |
| 同步开销 | 多阶段算法同步频繁 | **Full-mesh 单阶段，低同步** |
| 核心负载均衡 | 固定分区，易产生长尾 | **基于拓扑与流量的动态均衡** |
| 计算资源竞争 | 通信占用大量 NPU core | **SDMA offloading，释放 NPU core** |

---

## 2. 核心实验方法和设置

### 实验平台
- **硬件**：华为 **CloudMatrix384 (CM384)** supernode，集成 384 个 Ascend 910C NPU，通过 **Unified Bus (UB)** 互联，提供近 400 GB/s 带宽与纳秒级远程访问延迟。
- **软件栈**：基于 CANN，集成 PyTorch、SGLang、TorchTitan、TorchRec。

### 基线方法对比
- **HCCL**：华为生产级通信库，存在 staging copy。
- **HCCL-zerocopy**：支持 user-buffer direct communication，但使用 **预注册内存池**（pool-based registration）模拟，作为乐观基线。

### 评估指标
- **微基准测试**：Bus Bandwidth (GB/s)，遵循 NCCL 标准。
- **端到端应用**：
  - **LLM 推理**：吞吐量（throughput）、P99 TTFT（Time-To-First-Token）、P99 TPOT。
  - **训练任务**：迭代时间（iteration time）。

### 主要工作负载
| 工作负载 | 模型 | NPU 数量 | 并行策略 |
|--------|------|----------|----------|
| LLM 推理 | DeepSeek V4 Flash | 192 | DP+EP |
| LLM 训练 | DeepSeek V3.2 671B | 512 | FSDP+TP+EP |
| Recsys 训练 | DLRM | 128 | DP+MP (table-wise embedding) |

---

## 3. 主要实验结果和性能指标

### 微基准测试结果

#### （1）Collective 通信（AllGather / AllReduce）
- **小到中等负载（<16 MiB）**：StrataCL 较 HCCL-zerocopy 提升 **最高达 1.6×** Bus BW。
- **原因**：full-mesh 执行避免多步同步，workload-balanced partitioning 减少长尾。
- **大负载**：HCCL-zerocopy 略优（约高 6%），因 full-mesh 引发过多并发访问与网络争用。

#### （2）MoE Dispatch/Combine（EP=32）
| 模式 | Dispatch (HT) | Combine (HT) | Dispatch (LL) | Combine (LL) |
|------|---------------|--------------|---------------|--------------|
| **DeepEP (RDMA)** | 61 | 61 | 50 | 55 |
| **CANN EP** | 98 | 85 | 61 | 68 |
| **CANN EP zerocopy** | 106 | 93 | 82 | 79 |
| **StrataCL** | **130** | **121** | **107** | **108** |

- StrataCL 较 CANN EP zerocopy 提升 **22.6%–36.7%**，得益于 workload-balanced partitioning 对专家路由不均的优化。

---

### 端到端性能结果

| 任务 | 指标 | HCCL → HCCL-zerocopy | HCCL → StrataCL |
|------|------|------------------------|------------------|
| **LLM 推理** | 吞吐量 | +1.2× | **+1.9×** |
| | P99 TTFT | - | **降低 2.2×** |
| | P99 TPOT | - | **降低 1.1×** |
| **LLM 训练** | 迭代时间 | -6% | **降低 18%–24%** |
| **Recsys 训练** | 迭代时间 | - | **降低 ~23%** |

> **注**：HCCL-zerocopy 因预注册内存池导致 PyTorch expandable-segment allocator 被禁用，引发 **1.6–3.0 GiB 内存碎片**，限制 batch size，削弱其收益。

---

### 消融实验结果（Ablation Study）

#### （1）LLM 推理性能分解（图 13a）
| 技术组合 | 吞吐量提升（vs HCCL） |
|--------|-----------------------|
| JIT 注册 | 1.1× |
| **Registration-on-Allocation (RoA)** | **1.4×** |
| + Workload-Balanced Partitioning | 1.7× |
| + SDMA Offloading | **1.9×** |

#### （2）SDMA Offloading 开销
- **延迟代价**：较 MTE 路径慢约 **9%**（descriptor 构造与 doorbell 开销）。
- **NPU core 占用**：**降低 >95%**，显著缓解 compute-communication contention。

#### （3）Workload-Balanced Partitioning 效果
- **核心完成时间差距**：从 naive 分区的 **43%** 降至 **<5%**。
- **Makespan 降低 19%**。

#### （4）Registration-on-Allocation 可行性
- **分配到通信间隔**：最小值仍达数秒（远大于 μs/ms 级注册耗时），验证异步注册可行性。

---

## 4. 关键结论和发现

### 主要发现
1. **Registration-on-Allocation 是可行且高效的**：利用框架内存分配惰性，将注册移出关键路径，实现透明的 user-buffer direct communication。
2. **Full-mesh + Workload-Balancing 在 supernode 上更优**：在高带宽低延迟 fabric 上，同步与长尾成为新瓶颈，传统 ring 算法不再最优。
3. **SDMA Offloading 显著提升 overlap 效率**：释放 NPU core 给计算核，是实现高性能 compute-communication overlap 的关键。
4. **StrataCL 兼容性强**：已集成至主流框架（PyTorch、SGLang），并在 NVIDIA GPU 上验证其通用性（NCCL + RoA 提升 1.3×）。

---

### 方法的局限性
1. **Full-mesh 在大规模下存在 fan-out 争用**：当 rank 数增长（如 256 ranks），并发远程访问可能引发网络拥塞，性能略低于 ring。
2. **大负载场景未完全优化**：对于超大 payload，multi-step 算法（如 ring）仍有优势，需引入 workload-aware operator selection。
3. **依赖 supernode 特性**：全局统一物理地址空间（global unified physical address space）是实现轻量注册的关键，难以直接移植到传统 RDMA 集群。

---

### 未来工作方向
1. **动态 operator selection**：根据 payload 大小、rank 数、拓扑位置自动选择 full-mesh 或 multi-step 算法。
2. **跨 supernode 扩展**：探索 StrataCL 在多 supernode 间通信的扩展能力。
3. **更智能的 partitioning 策略**：结合 runtime profiling 动态调整 workload 分区。
4. **支持更多通信原语**：如 AllToAllv、ReduceScatter 等的深度优化。

---

> **总结**：StrataCL 重新思考了 supernode 架构下的通信库设计，通过 **registration-on-allocation** 和 **fabric-native operator 设计**，实现了 **零冗余、低延迟、高带宽** 的通信，显著提升了 LLM 与 Recsys 等生产负载的端到端性能，为下一代 AI 系统通信基础设施提供了重要参考。

</details>

---

### 10. [DualDecoder: Accelerate Long Context LLM Inference by Predictive Prefetch](https://arxiv.org/abs/2607.26475)

**Authors**: Zuning Liang, Zhiyi Yao, Qi Chen, Yuedong Xu, Hao Dai, Zhiqiang Ding, Tongkai Yang, Jinlong Hou, Yuan Cheng  
**Category**: cs.DC  
**Published**: 2026-07-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.26475v1  

#### Abstract
Long-context inference is becoming a fundamental capability for modern LLM serving, especially driven by emerging agentic applications. Yet it faces a severe memory wall that the KV cache scales proportionally with increasing context length and request concurrency. Existing sparse KV cache methods o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《DualDecoder: Accelerate Long Context LLM Inference by Predictive Prefetch》论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代大语言模型（LLM）在长上下文推理中面临严重的**内存墙**问题。随着上下文长度和请求并发数的增长，KV Cache 占用大量 GPU 高带宽内存（HBM），限制了解码吞吐量。

尽管已有动态稀疏 KV Cache 方法（如 ShadowKV、SpeCache）通过将大部分 KV 条目卸载到主机内存来缓解该问题，但它们引入了显著的**GPU-resident 辅助状态（KV-cache-auxiliary residency）**，例如低秩 Key 缓存或量化 KV 表示，用于加速检索或预取。这些辅助状态本身消耗大量 GPU 内存，在高并发场景下成为新的瓶颈。

### 提出了什么新方法或新思路
本文提出 **DualDecoder**，一种轻量级的长上下文 LLM 推理系统，其核心思想是：  
> **利用前序生成 token 对下一个解码步所需的 KV 条目进行预测，并提前从主机内存中预取（predictive prefetch），从而消除对 GPU 上冗余辅助状态的依赖。**

具体创新设计包括：

- **Dual-Token 解码流水线（Dual-Token Decoding Pipeline）**  
  在每个 decoding step 中同时生成正常输出 token 和一个“推测性 token”（speculative token）。后者用于预测下一步 attention 所需的关键 KV 条目索引，且与主 token 共享计算流程，仅增加极小开销。

- **层感知的 KV 预取调度器（Layer-aware Transfer Schedule）**  
  根据 Transformer 层执行顺序安排 KV 预取时机，在 layer `i` 完成 attention 后开始预取 layer `i+2` 的 KV 数据，实现通信与计算的有效重叠，避免过早占用 GPU buffer。

- **层作用域 KV 内存管理（Layer-Scoped KV Memory Manager）**  
  使用双缓冲（ping-pong buffer）机制，仅维护两个活跃的 sparse KV buffer，大幅减少运行时 GPU buffer 大小，并通过 CUDA stream event 协调计算与通信，防止数据竞争。

- **优先级缺失 KV 传输 + 自适应 Key 重建（Prioritized Recovery & Adaptive Reconstruction）**  
  将未命中项（missing KV entries）通过高优先级流紧急传输；对于预测准确率较低的层，选择性启用低秩 Key 重建以进一步降低延迟。

### 相比现有方法的优势
- **无需大量 GPU-resident 辅助状态**：摆脱了对低秩 Key 缓存等内存密集型结构的依赖。
- **更高的吞吐量**：释放出的 GPU 内存可用于支持更大的 batch size，提升并发处理能力。
- **保持低延迟与高质量**：不牺牲 decoding latency 或生成质量。
- **系统轻量高效**：所有优化均集成于标准推理流程中，无额外模型或复杂控制逻辑。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **RULER**：广泛使用的长上下文基准测试套件，包含多个需要深度理解的任务（如 Needle-in-a-Haystack, QA 等）。
- **NIAH-1 / NIAH-2 / NIAH-3 / QA-2**：用于评估 end-to-end 准确性的子任务集合。

### 实验设置
- **硬件平台**：
  - 8 × NVIDIA Hopper GPU（每卡 80GB HBM）
  - 双 Intel Xeon Platinum 8558 CPU（共 96 核）
  - 主机内存 1.6TB
  - PCIe 5.0 互联（双向带宽 128 GB/s）
- **软件环境**：
  - PyTorch 2.3.1, CUDA 12.8
  - 基于 **ShadowKV** 构建 DualDecoder 系统
- **模型范围**：
  - Llama-3 系列（8B, 14B, 32B）
  - Qwen-2.5 系列（8B, 32B）
- **上下文长度**：64K ~ 512K tokens
- **请求模式**：Poisson 过程模拟请求到达，速率从 0.1 到 100 req/s 不等

### 评估指标
| 指标 | 描述 |
|------|------|
| **Decoding Throughput (tokens/sec)** | 每秒生成的 token 数量，衡量系统吞吐能力 |
| **P99 Request Completion Time (RCT)** | 99% 请求完成时间，反映服务响应性能 |
| **GPU Memory Footprint** | GPU 显存占用情况，重点关注 KV-cache-auxiliary residency 比例 |
| **KV Retrieval Prediction Accuracy** | 预测的 sparse KV indices 与真实值的重合率 |
| **TTFT / TPOT** | Time to First Token / Time Per Output Token，衡量端到端延迟 |

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **vLLM** | Full KV Cache | 整个 KV cache 存放于 GPU，易 OOM |
| **ShadowKV** | Dynamic Sparse KV Cache | 使用低秩 Key 缓存重构部分 KV，减少 GPU 存储但引入辅助状态 |
| **SpeCache** | Speculative KV Caching | 使用 draft model 引导 KV 预取，仍需紧凑 KV 表示 |
| **DualDecoder (Ours)** | Predictive Prefetch | 无 heavy auxiliary state，基于 speculative token 预测并预取 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 解码吞吐量提升（Figure 12）
| Context Length | Model | DualDecoder vs. ShadowKV | vs. SpeCache | vs. vLLM |
|----------------|--------|----------------------------|---------------|------------|
| 64K | Qwen-2.5-32B | **1.14×** | **9.02×** | **3.65×** |
| 128K | Qwen-2.5-32B | **2.62×** | 8.63× lower | OOM |
| 256K | Qwen-2.5-32B | **1.32×** | OOM | OOM |

> 💡 **最高达 2.62× 吞吐提升**，尤其在长上下文下优势明显。

#### ✅ 请求并发能力增强（Figure 13）
- 当设定 P99 RCT ≤ 60s 时，DualDecoder 支持的请求到达率是 ShadowKV 的 **2.51×**。
- 在 128K context 下，当请求速率为 4.8 req/s 时，DualDecoder 所有请求可在 50 秒内完成，而 ShadowKV 的 P99 时间急剧上升，超过 1.92 req/s 即无法满足 SLA。

#### ✅ 显存占用显著下降（Figure 14, 15）
| 场景 | 内存节省 |
|------|---------|
| 8B model, batch=12 | 比 ShadowKV 节省 **36.4%**，比 SpeCache 节省 **62.0%** |
| 32B model, batch=4 | 仍可节省 **13.8%** 显存 |
| Context 从 64K → 512K | DualDecoder 仅增加 **3.7GB** 内存，而 ShadowKV 和 vLLM 已 OOM 或需额外 17.8GB |

> 📉 内存增长缓慢，具备良好可扩展性。

#### ✅ KV 检索预测准确性高（Figure 16）
- 平均 KV retrieval prediction accuracy 达 **88%**。
- 在 128K context 下仍稳定在 **82.6% 以上**，表明预测高度可靠。
- 初始 speculative token 若使用真实 token 初始化，准确率远高于随机初始化（可达 88% vs. 56%）。

#### ✅ 端到端生成质量无损（Table 1）
| Dataset | ShadowKV | DualDecoder (Ours) |
|--------|----------|---------------------|
| NIAH-1 | 1.00 | 1.00 |
| NIAH-2 | 1.00 | 1.00 |
| NIAH-3 | 0.82 | **1.00** |
| QA-2   | 0.53 | 0.58 |
| **Avg** | 0.94 | **1.00** |

> ✅ DualDecoder 在多数任务上保持 **lossless quality**，甚至优于 ShadowKV（因后者低秩重建导致精度损失）。

---

### 消融实验结果（Modular Study）

#### 🔬 Dual-Token 解码效率（Figure 17）
- 相比传统“额外一次 decoding pass”方式，Dual-Token 设计使 attention 层计算时间减少 **49%**（0.035ms vs. 0.069ms）。
- linear 层也获得显著加速，证明 co-execution 高效。

#### ⚖️ 优先级调度效果（Figure 18）
- 在 33% prediction miss rate 下，朴素调度导致 **43.3ms 延迟**，而 DualDecoder 的优先级流将其压缩至 **34.8ms**。
- 有效避免 pipeline stall，保障生成流畅性。

#### 🚀 GPU-side Gather 性能优势（Figure 19）
- 相比 CPU-side gather，GPU-side gather copy latency 从 **49.5ms 降至 3.25ms（↓93.4%）**。
- 支持异步、连续块预取 + GPU 并行重排，极大降低 I/O 开销。

---

## 4. 关键结论和发现

### 主要发现
1. **KV retrieval 具有强可预测性**：相邻 decoding step 所需的 sparse KV indices 高度相关，可通过 speculative token 准确预测（平均 88% 准确率）。
2. **auxiliary residency 是当前稀疏 KV 系统的新瓶颈**：其显存占用可达 sparse KV entries 的 8.5×，严重制约 batch size 扩展。
3. **predictive prefetch 可替代 heavy auxiliary state**：通过提前预取，既能隐藏 PCIe 传输延迟，又能释放 GPU 显存。
4. **DualDecoder 实现了吞吐、内存、延迟的协同优化**：在不牺牲质量的前提下，显著超越现有最先进系统。

### 方法的局限性
- **依赖 speculative token 质量**：若初始 token 初始化不当，可能影响后续预测轨迹。
- **对模型结构有一定假设**：需支持灵活的 attention mask 和 dual-input 处理。
- **预取错误仍需纠正机制**：missed KV entries 需要高优传输或重建，增加了控制复杂性。
- **目前实现在 ShadowKV 基础上构建**：通用性有待在更多系统中验证。

### 未来工作方向
- 探索更高效的 speculative token 生成策略（如 early-exit 或 multi-token prediction）。
- 将 predictive prefetch 思想推广至其他 memory-bound inference 场景（如 MoE routing、activation offloading）。
- 结合 disaggregated memory 或 remote KV pool 架构，进一步拓展存储边界。
- 动态调整预取粒度与预算，适应不同 workload 特征。

---

> ✅ **总结一句话**：  
> **DualDecoder 通过“预测 + 预取”的范式转变，打破了稀疏 KV 缓存中“辅助状态换速度”的传统权衡，实现了更高吞吐、更低内存、同等质量的长上下文 LLM 推理。**

</details>

---

### 11. [RAGuard: A Layered Defense Framework for Retrieval-Augmented Generation Systems Against Data Poisoning](https://arxiv.org/abs/2607.26339)

**Authors**: Pushkal Kumar, Tucker Nielson, Tanish Kolhe, Shubham Zala, Vincent Li  
**Category**: cs.LG  
**Published**: 2026-07-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.26339v1  

#### Abstract
Retrieval-Augmented Generation (RAG) systems ground large language models (LLMs) in external corpora, but this reliance exposes them to corpus poisoning: maliciously injected passages that manipulate retrieved evidence. We introduce RAGuard, a layered defense against \emph{factual} corpus-poisoning ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RAGuard: A Layered Defense Framework for Retrieval-Augmented Generation Systems Against Data Poisoning  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Retrieval-Augmented Generation (RAG) 系统通过从外部语料库中检索证据来增强大语言模型（LLM）的回答能力，但这也带来了新的安全威胁——**corpus poisoning（语料库投毒）攻击**。攻击者可以向检索语料库中注入看似相关但包含虚假信息的文档，从而误导生成器输出错误答案。

现有防御机制存在以下缺陷：
- **基于检测的过滤器**依赖于已知的投毒标签或启发式规则，难以泛化到新型攻击；
- **生成器加固方法**推理开销高，且在多个投毒文档同时被检索时效果下降；
- **对抗训练的检索器**容易过拟合训练时见过的投毒类型，无法应对未知攻击。

### 🚀 提出的新方法：RAGuard
本文提出 **RAGuard**，一个两层防御框架，结合了**对抗性检索器训练**与**无监督推理时过滤机制**，以抵御事实类语料投毒攻击。

#### 创新点如下：

| 层级 | 方法 | 创新说明 |
|------|------|----------|
| **Layer 1**: 对抗性检索器训练 | Adversarially fine-tune dense retriever | 在合成投毒文档（如伪造事实、矛盾陈述、逻辑陷阱）上进行对比学习，使检索器学会将恶意文档降权。 |
| **Layer 2**: 零知识推理补丁（ZKIP） | Zero-Knowledge Inference Patch (ZKIP) | 一种**无需标签、黑盒、自参照**的推理时过滤器：<br>对每个检索到的文档执行 leave-one-out（LOO）解码，计算其移除后引起的：<br>① **语义答案偏移量**（Semantic Shift）<br>② **输出熵变化**（Output Entropy Differential）<br>综合这两个信号判断是否为投毒文档并予以剔除。 |

> 🔍 **ZKIP 的核心思想是：**  
> 不依赖任何外部真值（gold answer）、投毒标签或模型内部参数，仅通过比较模型自身在不同反事实上下文下的输出差异，识别出“一旦移除就能显著提升答案稳定性或降低不确定性的文档”——这正是投毒文档的典型特征。

### ⭐ 相比现有方法的优势

| 维度 | RAGuard 的优势 |
|------|----------------|
| **通用性** | ZKIP 是 label-free 和 attack-agnostic 的，可防御未见攻击类型；而传统方法需重新标注/训练。 |
| **模块化与兼容性** | 可插拔设计，适用于任意 LLM 和 retriever，不修改原始系统结构。 |
| **端到端安全性** | 同时覆盖 retrieval 和 generation 两个阶段，形成纵深防御（layered defense）。 |
| **无需额外监督信号** | ZKIP 完全不需要 ground-truth answers 或 poison labels，在真实场景更具实用性。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Natural Questions (NQ)**：广泛使用的开放域问答数据集，主题多样。
- **BEIR (NFCorpus)**：医学/科学领域的信息检索基准，相关性信号稀疏，更具挑战性。

> 所有数据集均构造了**干净版本**与**投毒版本**（5%-30% 投毒比例），每条记录包含查询、原始正确文档、投毒文档及攻击类别标签。

### 🧪 投毒构建方式
使用 LLM 构造三类 synthetic poisons：
1. **Fabrication**：添加虚构或幻觉性陈述；
2. **Contradiction**：翻转关键事实词（如 “true” → “false”）；
3. **Reasoning Trap**：引入误导性推理步骤或中间断言。

> 所有投毒均为语义改写，保留关键词，因此对 BM25 影响小，专门针对 dense retrieval 设计。

### 📊 实验设置
- **Retriever 类型**：
  - `BM25`（lexical）
  - `Dense (clean)`：仅用干净三元组训练
  - `Dense (adv-trained)`：在合成投毒数据上对抗训练
- **Generator**：GPT-4o-mini（主实验）、FLAN-T5-small（批处理实验）
- **Top-k retrieval**：k = 5
- **防御配置**：单独使用 Layer1、单独使用 ZKIP、两者联合

### 🎯 评估指标
| 指标 | 定义 | 越低越好？ |
|------|------|-----------|
| **Recall@5** | 正确文档出现在 top-5 中的比例 | 否（越高越好） |
| **MRR** | 第一个相关文档的平均倒数排名 | 否 |
| **Attack Success Rate (ASR)** | 投毒文档排在正确文档之前且导致错误回答的比例 | ✅ 是 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & Table 4）

#### 在 NQ 上，10% 投毒率的结果：

| Retriever | Defense | Recall@5 | MRR | ASR ↓ |
|----------|--------|---------|-----|-------|
| Dense (clean) | None | 0.259 | 0.176 | 0.101 |
| Dense (adv-trained) | None | 0.319 | 0.215 | 0.072 |
| Dense (clean) | ZKIP | 0.259 | 0.176 | **0.000** |
| Dense (adv-trained) | ZKIP | 0.314 | 0.215 | **0.000** |
| BM25 | ZKIP | 0.070 | 0.056 | **0.000** |

> ✅ **所有启用 ZKIP 的配置，ASR 均降至 0.000！**

#### 在 NQ 上，30% 投毒率的结果（部分）：
| Configuration | ASR ↓ |
|--------------|--------|
| Dense (adv-trained) + ZKIP | **0.000** |
| Dense (clean) + ZKIP | **0.000** |

> 即使在极端投毒条件下，ZKIP 仍能完全消除测量到的攻击成功。

### 🔍 与基线方法的对比结果

| 对比维度 | 结果 |
|--------|------|
| **vs. 单纯对抗训练 retriever** | 能提升 Recall@5 和 MRR，但**不能彻底消除 ASR**（最低仍为 0.072） |
| **vs. 无防御 baseline** | Dense retrievers 的 ASR 在 0.029–0.101 之间波动，表明易受攻击 |
| **vs. BM25** | BM25 自身几乎不受影响（ASR ≈ 0.000），因其基于关键词匹配，而本研究的投毒保留了关键词；但这不代表 BM25 更优——它的 Recall@5 极低（0.07），不具备实用价值 |

> 💡 **重要发现**：投毒主要影响 dense retrieval，对 lexical retrieval（如 BM25）无效 → 明确了威胁模型边界。

### 🔬 消融实验结果（Ablation Study）

#### （1）**防御层级消融（Table 2）**

| 配置 | ASR ↓ | Recall@5 |
|------|------|---------|
| Only adversarial retriever | 0.072 | 0.319 |
| Only ZKIP | **0.000** | 0.259 |
| Both layers | **0.000** | 0.314 |

> ✅ **结论**：ZKIP 是实现零 ASR 的必要且充分条件；对抗训练提升检索质量，但不足以单独完成防御任务。

#### （2）**ZKIP 信号消融（未完成，计划中）**
- 测试仅使用 `stability` 或 `entropy differential` 是否足够。
- 预期：
  - Stability 主要捕捉 fabrication/contradiction 攻击；
  - Entropy differential 更适合 detecting reasoning traps（增加模型不确定性）。

#### （3）**监督分类验证（Appendix D）**
使用 ZKIP 提取的影响特征训练监督分类器：
- Logistic Regression: AUPRC = 0.377
- BERT Classifier: AUPRC = **0.732**

> ✅ 表明 ZKIP 的反事实信号中蕴含可学习的投毒结构，支持其有效性。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **ZKIP 极其有效**：  
   在所有测试配置下（包括不同 retriever、不同投毒率），ZKIP 将 **ASR 成功压至 0.000**，证明其作为推理时过滤器的强大鲁棒性。

2. **分层防御优于单一机制**：  
   - Layer 1（对抗训练 retriever）提升检索质量；
   - Layer 2（ZKIP）保障生成安全；
   - 二者结合可在保持高 Recall@5 的前提下实现完全防御。

3. **ZKIP 是真正意义上的黑盒防御**：  
   无需 poison labels、gold answers、model internals，仅依赖模型自身的反事实行为差异，具备强泛化潜力。

4. **威胁模型明确界定**：  
   当前方法针对 dense retrieval 的语义投毒，而 keyword-preserving 投毒对 BM25 几乎无影响，说明攻击与防御均需考虑检索范式。

---

### ⚠️ 方法的局限性（Limitations）

| 问题 | 描述 |
|------|------|
| **计算开销大** | ZKIP 需要 `k+1` 次 generator 推理（k=5 时为 6× 开销），对延迟敏感应用不友好。虽可通过 batching 缓解，但仍显著高于 baseline。 |
| **多投毒协同攻击失效风险** | 若多个投毒文档共同支撑同一错误答案，单次 LOO 移除任一文档都不会引起明显变化，LOO 信号被抑制 → ZKIP 可能漏检。 |
| **假阳性问题** | 对某些良性但具有决定性作用的文档（如唯一提供关键事实的文档），其移除也会引起答案剧变，可能被误判为投毒。 |
| **评估范围有限** | 当前实验集中在 factual QA；对于 opinion manipulation、multi-hop reasoning 等复杂任务尚未验证。 |
| **对抗训练泛化性存疑** | 在 BEIR 上，对抗训练未能显著降低 ASR，提示其可能对 out-of-domain 数据脆弱。 |

---

### 🔮 未来工作方向（Future Work）

1. **应对多投毒联盟攻击**：
   - 开发迭代式 ZKIP（iterative removal），逐个清除协同投毒成员。
   - 引入 coalition detection 机制。

2. **降低推理成本**：
   - 使用 early stopping、subset sampling 近似策略减少 LOO 次数；
   - 动态选择性应用 ZKIP（仅对高风险查询启用）。

3. **扩展至更广威胁模型**：
   - 集成 PoisonedRAG、FlippedRAG 等标准攻击套件进行压力测试；
   - 探索对 prompt injection、backdoor trigger 的防御能力。

4. **理论分析**：
   - 建立 LOO filtering 与 poison geometry 之间的理论联系；
   - 分析 ZKIP 的误差界与鲁棒性保证。

5. **人机协同机制**：
   - 结合 human-in-the-loop 验证可疑文档；
   - 构建 active learning pipeline 持续优化防御。

---

## 总结（Summary）

> ✅ **RAGuard 是首个实现“零攻击成功率”的 label-free、black-box、layered RAG 投毒防御框架。**  
> 它通过 **对抗训练 retriever** 提升第一道防线的质量，并利用创新的 **ZKIP 机制** 在生成阶段动态识别并剔除潜在投毒文档。实验证明，ZKIP 能将 ASR 彻底归零，同时仅轻微牺牲 Recall@5（within 0.03 of baseline）。尽管存在计算开销和多投毒协同等挑战，但其模块化、无需标注、可泛化的特性使其成为迈向可信 RAG 系统的重要一步。

> 🔗 **代码、数据集、评测工具均已开源**：https://github.com/RAGuard-AI/RAGuard  
> 👉 推荐用于高风险领域（医疗、金融、法律）中的 RAG 系统部署前的安全加固。

</details>

---

### 12. [A Cost-Effective Multimodal LLM Reasoning Framework for Question Answering over Irregular Clinical Time Series](https://arxiv.org/abs/2607.25947)

**Authors**: Frank Nie, Ethan B Liu, Yuan Zhu, Wei Fan, Jindong Han  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.25947v1  

#### Abstract
Question answering (QA) over irregular clinical time series (ICTS) plays a pivotal role in a wide range of healthcare applications. Although recent multimodal time-series large language models (LLMs) have shown considerable promise in general-purpose time-series QA, they remain poorly equipped to mo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对**不规则临床时间序列**（Irregular Clinical Time Series, ICTS）上的**自然语言问答**（Question Answering, QA）任务中存在的三大挑战：
- **不规则性建模**（Irregular temporal modeling）：临床观测具有非均匀、变量依赖的采样间隔，且观测过程本身可能携带信息。
- **稀疏多尺度证据定位**（Sparse multi-scale evidence localization）：临床问题所需证据在时间上分布稀疏且跨尺度（如全局趋势、局部事件、单点测量）。
- **不规则时序-语言对齐困难**（Irregular temporal-language alignment）：缺乏大规模配对数据来有效连接不规则时序模式与语言语义。

现有方法（如文本序列化、表示适配或多模态LLM）大多假设规则采样，在处理稀疏、异步的临床数据时表现不佳。

---

### 提出的新方法：CLINPRISM
作者提出 **CLINPRISM** —— 一种**低成本、高效的多模态LLM推理框架**，专为ICTS上的QA设计。其核心创新包括：

#### （1）不规则感知的多尺度编码器（Irregularity-Aware Multi-Scale Encoder）
- 在三个尺度上捕获临床证据：
  - **Macro-scale**：全局轨迹趋势（通过可学习查询注意力池化）。
  - **Meso-scale**：局部事件动态（使用可学习软窗口，自适应于观测密度）。
  - **Micro-scale**：细粒度状态（基于参考时间点的缩放点积注意力）。
- 所有尺度直接作用于原始不规则序列，**无需重采样或插值**，保留原始时间戳和缺失模式。

#### （2）时序证据蒸馏器（Temporal Evidence Distiller）
- 将多尺度异构表示投影到LLM隐藏空间，并通过**分层融合**整合信息。
- 使用**基于查询的重采样**（Query-Based Resampling）将变长表示压缩为固定数量（默认16个）的`temporal tokens`，实现高效LLM交互。

#### （3）渐进式时序-语言对齐策略（Progressive Temporal-Language Alignment）
三阶段训练策略：
1. **Stage 1**：多尺度编码器预训练（无监督）。
2. **Stage 2**：冻结编码器和LLM，仅优化蒸馏器，利用**分层caption**进行双向对比对齐。
3. **Stage 3**：两步微调——先微调蒸馏器，再联合优化蒸馏器与LLM中的LoRA模块，完成QA适配。

#### （4）构建高质量训练资源
- 构造了30,000条带**分层描述**（global/segment/micro）的ICTS轨迹。
- 构造了约41,000个覆盖11项任务的**多任务QA指令数据集**。

---

### 相比现有方法的优势
- **高效性**：仅用16个`temporal tokens`即可实现高精度，避免长文本序列化带来的高延迟。
- **有效性**：显式建模多尺度稀疏证据，显著提升对复杂临床问题的理解能力。
- **通用性**：渐进式对齐策略可推广至其他多模态LLM系统。
- **成本低**：基于仅4B参数的Qwen3-4B LLM即达到SOTA性能。

---

## 2. 核心实验方法和设置

### 数据集
- **训练数据**：从MIMIC-IV ICU数据库中提取的去标识化记录。
  - **不规则时间序列语料库**：30,000条ICU住院轨迹，含11个临床变量（如heart rate, map, spo2等），保留原始观测时间与缺失模式。
  - **分层caption语料库**：每条轨迹对应一组由GPT-5.5生成的global/segment/micro/fused caption，用于Stage 2对齐。
  - **多任务QA语料库**：约41,000个四选一选择题，覆盖4大能力维度共11类任务（见下表）。

| 能力维度 | 任务类型 |
|---------|--------|
| Temporal Understanding | TG, ASR, TPR, MA, TSS |
| Temporal Forecasting | TF, NIF |
| Temporal Reasoning | CVR, IR |
| Temporal Decision-Making | IID, MED |

- **测试基准**：**CLIR-Bench** —— 一个独立保留的不规则多变量时间序列QA评测集，包含上述11项任务。

> 注：训练与测试数据无患者、住院或ICU停留重叠，确保公平评估。

---

### 实验设置与评估指标
- **模型架构**：以 **Qwen3-4B** 作为LLM主干。
- **Token预算**：每个问题使用 **16个temporal tokens**。
- **评估指标**：所有任务均为多项选择题，采用**准确率**（Accuracy）作为唯一指标。
- **硬件环境**：NVIDIA RTX 4090 GPU，批量大小32。

---

### 基线方法对比
涵盖四类主流模型：
1. **闭源LLM**：
   - Gemini-2.5-flash
   - GPT-5.4 mini
2. **开源通用LLM**（参数量3–27B）：
   - DeepSeek-V4-flash, KiMi-2.6, Gemma系列, Qwen系列等
3. **时间序列LLM**：
   - Time-LLM, ITFormer, ChatTS, TS-Reasoner, AutoTime
4. **特殊基线**：
   - `t-PatchGNN + Qwen3-4B`：使用t-PatchGNN作为时间编码器但**无caption对齐**的变体，用于消融比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 参数量 | 平均准确率（Overall） | 推理延迟 | Temporal Tokens |
|------|-------|---------------------|----------|----------------|
| **CLINPRISM** | **4B** | **49.83%** | **0.15秒** | **16** |
| GPT-5.4 mini | – | 50.15% | – | – |
| Gemini-2.5-flash | – | 40.30% | – | – |
| KiMi-2.6 | 1T | 49.09% | – | – |
| t-PatchGNN + Qwen3-4B | 4B | 38.65% | ~0.12秒 | 16 |

> ✅ **CLINPRISM是所有开源系统中表现最佳者**，仅次于GPT-5.4 mini（仅差0.32个百分点），远超Gemini-2.5-flash（+9.53pp）。

---

### 与基线方法的对比结果
- 在**Missingness Awareness (MA)** 上达 **87.00%**，显著优于次优模型（54.17%），表明其能有效识别缺失模式。
- 在**Immediate Intervention Decision (IID)** 上达 **96.67%**，说明其具备强临床决策支持潜力。
- 在**Cross-Variable Reasoning (CVR)** 和**Intervention Response (IR)** 上也大幅领先，验证了多尺度证据融合的有效性。
- 相比`t-PatchGNN`基线，平均提升**11.18个百分点**，证明所提架构与对齐策略的关键作用。

---

### 消融实验结果（Ablation Study）

| 变体 | 平均准确率 |
|------|-----------|
| Only Macro | 46.97% |
| Macro + Meso | 48.24% |
| Full Model (CLINPRISM) | **49.83%** |
| Without Stage 2 (无caption对齐) | 48.27% |
| Without Stage 3 Step 1 | 47.55% |
| Without Stage 3 Step 2 (无LoRA联合优化) | **36.35%** ⬇️ |

> 🔍 发现：
> - 添加Meso和Micro尺度持续增益，证明**多尺度建模必要性**。
> - 移除Stage 2导致性能下降，说明**分层caption对齐至关重要**。
> - 移除Stage 3第二步（LoRA联合优化）导致最大性能损失，凸显**端到端协同优化的重要性**。

此外，对`temporal-token`数量 $ K $ 的敏感性分析显示：
- 最优 $ K = 16 $，此时平均准确率达峰值 **49.83%**。
- 更大的 $ K $（如128）反而性能下降，说明**过度token化会引入噪声并降低效率**。

---

## 4. 关键结论和发现

### 主要发现
1. **多尺度建模是处理稀疏、不规则临床数据的关键**：Macro提供上下文，Meso捕捉事件，Micro保留细节，三者互补。
2. **渐进式对齐策略有效桥接时序与语言空间**：通过分层caption监督，可在不微调LLM的情况下实现初步对齐。
3. **极少量temporal tokens即可实现高性能**：仅需16个tokens和0.15秒延迟，即可在复杂临床QA任务上达到接近闭源模型的表现。
4. **CLINPRISM在不同观测密度下保持鲁棒性**：相对变化仅1.6%，远低于基线（12.7%），适用于真实世界稀疏数据场景。

---

### 方法的局限性
- 当前仅支持**封闭式问答**（multiple-choice），尚未扩展至开放式生成。
- 所有caption和QA均由GPT-5.5生成，存在潜在的语言偏见或幻觉风险（尽管经过严格约束与人工审核）。
- 模型未建模不确定性，无法输出置信度估计。
- 依赖预定义的任务schema，泛化到全新任务类型的能力有限。

---

### 未来工作方向
- 扩展至**开放式问答**与**不确定性感知推理**。
- 引入更多模态（如影像、文本病历）构建真正的多模态临床AI助手。
- 在更广泛的临床环境中进行外部验证与部署研究。
- 探索轻量化部署方案，推动实际医疗应用落地。

</details>

---

### 13. [Penelope: Localized Latent Recurrence for Efficient Structured Reasoning](https://arxiv.org/abs/2607.25915)

**Authors**: Yutong Chen, Shouqian Shi, Xinran Liu, Haochen Wang, Jiaying Wang, Tianxing Xu, Yuanxi Wang, Zirui Ding  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.25915v1  

#### Abstract
Complex structured reasoning tasks often require additional computation, yet current language models obtain it mainly by increasing parameter scale or by serializing intermediate steps as chain-of-thought (CoT) tokens. The former raises training and deployment costs, while the latter ties reasoning ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Penelope: Localized Latent Recurrence for Efficient Structured Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前语言模型在处理**复杂结构化推理任务**（如表达式求值、逻辑推导、本体推理）时，通常依赖以下两种方式增加计算量：
- **扩大模型参数规模**（scaling）：提升能力但显著增加训练与部署成本；
- **Chain-of-Thought (CoT)**：通过生成中间推理token显式扩展推理路径，但将额外计算绑定到输出长度，导致推理延迟高。

此外，已有**latent reasoning**方法虽将中间推理移入隐藏状态以避免长文本轨迹，但仍普遍需要反复执行整个decoder，造成大量冗余计算。

Penelope旨在解决：  
> 如何在不重复执行完整decoder、也不生成长CoT序列的前提下，为预训练decoder-only Transformer提供高效且可扩展的**内部推理计算接口**？

---

### 🚀 提出的新方法与核心思想

提出 **Penelope** —— 一种面向decoder-only Transformer的**局部隐态循环框架**（localized latent recurrence），其核心设计包括：

#### （1）**局部化循环计算区间**（Localized Recurrent Interval）
- 将decoder划分为三个部分：`F[0:ℓs)`（前缀）、`F[ℓs:ℓe)`（循环区间）、`F[ℓe:L)`（后缀）。
- 仅对选定的输出侧子区间 `F[ℓs:ℓe)` 进行K次重复应用，用于更新固定大小的**latent memory**和**readout state**。
- 前缀 `F[0:ℓs)` 只运行一次，构建问题相关的边界上下文，并缓存KV Cache，供后续复用。

#### （2）**持久化的隐态接口**（Persistent Latent Interface）
- 引入一组固定的**latent anchors** 和 **readout anchors**，作为可在循环中持续更新的状态槽位。
- 利用时间调制GRU（time-modulated GRU）动态控制每步的记忆更新强度与时序信号。

#### （3）**渐进式CoT到隐态的课程学习**（Progressive CoT-to-Latent Curriculum）
- 训练初期保留可见CoT路径；
- 随着训练进行，逐步用一个latent refinement step替代一个visible reasoning step；
- 最终实现从显式推理向纯隐态推理的平滑过渡。

#### （4）**层一致的答案上下文构造**
- 在K轮latent refinement之后，最后一次通过 `F[ℓs:L)` 构造完整的answer-context KV Cache；
- 后续自回归解码基于此cache进行，无需再参与循环。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（如Coconut/CODI） | Penelope |
|------|----------------------------|----------|
| **计算效率** | 每次refinement需重跑全decoder（深度L） | 仅重跑局部区间（宽度r << L） |
| **推理延迟** | 与latent depth线性增长 | 增长速率降低至 `r/L` 倍 |
| **内存开销** | 多次full-pass带来高KV Cache重建成本 | 前缀KV Cache只建一次，极大减少访存 |
| **灵活性** | 循环跨度固定或全局 | 支持灵活选择任意decoder子区间 |
| **兼容性** | 多数需修改架构或训练流程 | 可适配不同backbone（如Llama/Qwen） |

> ✅ **核心优势总结**：  
> Penelope实现了**计算分配与模型规模的解耦**，在保持competitive accuracy的同时，显著降低了inference latency，提供了更实用的accuracy-efficiency tradeoff。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **Deep ListOps** | 表达式归约（compositional expression reduction） | 修改自ListOps，测试嵌套深度1–12的算术表达式求值；强调递归结构理解能力 |
| **ProsQA** | 合成多步逻辑推理（synthetic multi-step logical deduction） | 包含因果、否定、传递等规则链；验证逻辑演绎能力 |
| **PrOntoQA** | 组合性本体推理（compositional ontology reasoning） | 基于知识图谱关系的组合查询，考验语义结构泛化 |

所有数据集均采用**exact match (EM)** 作为主评估指标。

---

### ⚙️ 实验设置与评估指标

#### 模型配置
- 主干模型：`Llama-3.2-1B`（16层decoder），部分实验使用 `Qwen3.5-0.8B-Base`
- Penelope参数设置：
  - 循环区间：`[11, 16)`（即最后5层）
  - Latent memory slots: 8
  - Readout states: 2
  - Refinement steps K: validation-selected（非自适应）

#### 训练策略
- 起始checkpoint：统一来自同一visible-CoT微调模型
- 渐进课程训练：逐步替换visible reasoning steps为latent refinement steps
- 优化器：AdamW，cosine decay，batch size=4（有效），共1500/3000步更新

#### 评估指标
| 指标 | 定义 |
|-----|------|
| **Exact Match (EM)** | 完全匹配答案的比例（mean ± std over 3 runs） |
| **Latency (ms)** | 批大小为1，BF16精度下greedy generation的端到端延迟（同步CUDA测量） |
| **Pre-answer decoder-layer applications** | 回答生成前的串行decoder层调用次数（衡量计算拓扑深度） |

---

### 🔁 基线方法对比

| 方法 | 类型 | 特点 |
|-----|------|------|
| **Visible CoT** | 显式推理链 | 生成完整CoT过程，token数多，延迟高 |
| **Coconut** | 全decoder latent recurrence | 使用完整decoder进行K次隐态迭代，计算代价高 |
| **CODI** | 隐空间蒸馏 | 将CoT压缩为连续隐状态，但未优化执行路径 |
| **Full-decoder recurrence** | 全路径循环基线 | 用于分析Penelope局部化的有效性 |

> 所有方法共享相同初始checkpoint、数据划分、optimizer配置，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 总体性能对比（见Table 1）

#### (a) Deep ListOps 结果
| Method | EM (%) | Output Tokens | Latency (ms) | Pre-answer Layer Calls |
|--------|--------|----------------|---------------|-------------------------|
| Visible CoT | 51.27±0.92 | 69.00 | 1047.46±63.74 | — |
| Coconut | 52.79±0.36 | 4.00 | 188.15±1.02 | 160 |
| **Penelope** | **52.25±0.72** | 4.00 | **99.82±0.65** | **46** |

✅ Penelope达到与Coconut相当的准确率（相差<0.55点），**延迟降低46.9%**，**pre-answer计算量仅为Coconut的28.8%**（46 vs 160）。

---

#### (b) ProsQA 结果
| Method | EM (%) | Output Tokens | Latency (ms) |
|--------|--------|----------------|---------------|
| Visible CoT | 45.87±0.46 | 52.07 | 780.69±187.00 |
| Coconut | 79.87±1.80 | 8.65 | 252.25±15.94 |
| **Penelope** | **78.27±0.90** | 8.65 | **168.19±27.70** |

✅ 准确率接近Coconut（差约1.6点），**延迟下降33.3%**。

---

#### (c) PrOntoQA 结果
| Method | EM (%) | Output Tokens | Latency (ms) |
|--------|--------|----------------|---------------|
| Visible CoT | 99.71±0.19 | 87.79 | 1295.99±9.33 |
| Coconut | 99.58±0.14 | 3.00 | 160.71±0.63 |
| **Penelope** | **99.67±0.07** | 3.00 | **92.85±16.70** |

✅ 准确率略优于Coconut，**延迟下降42.2%**。

---

### 🔬 消融实验与机制分析

#### （1）K=0 推理诊断（无循环 refine）
- 使用训练好的模型但关闭latent refinement（K=0）
- 结果：平均EM为51.56±0.81%，相比启用refinement的52.25±0.72%低0.69点
- ➤ 表明latent refinement确实能修正部分错误预测，贡献正向增益

#### （2）transition dynamics 替换实验
- 移除time modulation、residual adapter、gated integration、memory GRU，改为直接覆盖 `M_t = H^lat_t`
- 结果：EM从52.25降至51.44（下降0.81点）
- ➤ 验证了regulated state refinement模块的有效性

#### （3）placement analysis（循环区间位置影响）
| 区间位置 | 层数范围 | Layer Calls | EM (%) |
|--------|----------|-------------|--------|
| Early | [0,5) | 51 | 52.04±0.47 |
| Middle | [6,11) | 51 | 52.23±0.85 |
| Output-side | [11,16) | 51 | 52.17±0.97 |
| Full decoder | [0,16) | 128 | 52.44±0.81 |

✅ 不同区间表现相近（差异<0.4点），说明**没有“最优layer”偏好**，局部化本身是可行的；
❌ 全decoder虽稍优（+0.27点），但代价是2.5倍以上的计算量 ➤ **性价比极低**

---

### 💡 Qwen跨骨干验证（Table 3）
- 在`Qwen3.5-0.8B`上复现实验：
  - Penelope EM: 52.06%，Coconut: 52.25%
  - Penelope latency: **207.11ms**，Coconut: **469.00ms**
  - ➤ **延迟降低55.8%**，准确率几乎持平

➤ 验证了Penelope对不同decoder架构的良好兼容性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **局部隐态循环足以支持有效推理**
   - 无需在整个decoder上重复执行，只需在一个窄区间内循环即可维持competitive reasoning performance。

2. **显著降低推理延迟与计算负载**
   - marginal refinement cost由L降为r（本文中5 vs 16），理论加速比趋近于 `r/L ≈ 31.25%`

3. **cache-compatible设计提升效率**
   - 前缀KV Cache仅构建一次，大幅减少内存访问与重复计算。

4. **accuracy-efficiency tradeoff更优**
   - 在多个structured reasoning任务上，Penelope实现了与full-latent方法相当的EM，同时显著降低latency（平均~40–55%）。

5. **无需假设特定layer具有特殊语义功能**
   - placement实验证明不同区间效果相近，支持“功能去中心化”的Transformer理解观。

---

### ⚠️ 局限性

1. **固定depth设定**
   - 当前采用validation-selected fixed K，缺乏input-adaptive的动态停止机制（如PonderNet）。

2. **尚未探索更大模型上的scalability**
   - 实验集中在1B以下模型，是否能在10B+级别仍保持高效有待验证。

3. **训练成本较高**
   - 引入了额外的recurrent modules（~64M params），trainable参数总量高于Coconut（75.35M vs 11.28M）
   - ❗但论文强调其claim focus on **inference efficiency**, not parameter efficiency

4. **仅适用于structured reasoning任务**
   - 对开放生成、创意写作等任务的有效性未知。

---

### 🔮 未来工作方向

1. **引入input-adaptive halting机制**
   - 动态决定每个样本所需的refinement steps数量，进一步提升效率。

2. **扩展至encoder-decoder架构**
   - 探索在T5、BART等模型中实现类似的localized latent recurrence。

3. **结合MoE或其他稀疏化技术**
   - 在latent refinement路径中引入sparsity，进一步压缩计算开销。

4. **应用于real-world reasoning场景**
   - 如数学证明、代码调试、法律推理等复杂领域任务。

5. **探索更高效的memory更新机制**
   - 替代GRU/time modulation的设计，如attention-free更新、state-space models等。

---

## ✅ 总结一句话

> **Penelope通过将latent recurrence局部化到decoder的一个子区间，并结合一次性上下文缓存与渐进式训练，实现了在不牺牲准确率的前提下，显著降低structured reasoning任务的推理延迟，为decoder-only模型提供了一种高效、实用的内部推理架构新范式。**

</details>

---

### 14. [Steering Instruction Hierarchies at Inference Time](https://arxiv.org/abs/2607.26228)

**Authors**: Siqi Zeng, Sewoong Lee, Han Zhao, Julia Hockenmaier  
**Category**: cs.CL  
**Published**: 2026-07-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.26228v1  

#### Abstract
Instruction hierarchies are a core safety assumption of language model deployment: higher priority inputs, such as system prompts, should override conflicting lower priority inputs from users or tools. Yet frontier LLMs often violate this hierarchy. We introduce V-Steer, a training-free inference ti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Steering Instruction Hierarchies at Inference Time**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代大型语言模型（LLMs）在部署时依赖**指令层级（Instruction Hierarchy, IH）** 来保证安全性与可控性。理想情况下，高优先级输入（如系统提示 `system prompt`）应覆盖低优先级输入（如用户请求、工具输出）中的冲突指令。然而，前沿 LLMs 在实践中常常违反这一层级，导致诸如 **prompt injection** 和 **agent hijacking** 等安全攻击成功。

本文指出，尽管已有训练阶段的方法试图强化这种层级控制，但在推理阶段缺乏高效、低成本且无需重新训练的干预手段。

---

### **提出的新方法：V-Steer**
作者提出了 **V-Steer** —— 一种**无需训练、仅在推理时进行干预**的方法，通过编辑缓存的 **value vectors** 来恢复系统指令的主导地位。

#### **核心思想**
- 利用 **Direct Logit Attribution (DLA)** 分析模型首次生成 token 时各注意力头对不同指令来源的贡献。
- 找出那些“错误地”更关注低优先级 span（如用户指令）而非高优先级 span（如系统指令）的注意力头。
- 对这些“坏头”执行 **in-place multiplicative edits**：
  - **Boost** 高优先级 span 的 value 向量（乘以 $1+\gamma^+$）
  - **Suppress** 冲突的低优先级 span 的 value 向量（乘以 $1-\gamma^-$）

该操作直接修改 KV Cache 中的 `V` 张量，不改变注意力计算过程本身。

---

### **相比现有方法的优势**
| 维度 | V-Steer | 其他方法 |
|------|--------|---------|
| **是否需要训练** | ❌ 不需要（training-free） | ✅ 多数需额外 fine-tuning 或 SFT/DPO |
| **是否兼容优化后端** | ✅ 是（保持 FlashAttention/SDPA 快路径） | ❌ 注意力重加权需 materialize attention matrix，破坏 fused attention |
| **运行时开销** | ⏱️ 仅一次 prefill 开销，无每步解码延迟 | ⏱️ 注意力干预每步都需重算，累积成本高 |
| **效果提升幅度** | 📈 显著优于 prompt engineering，媲美甚至超越部分 SoTA 训练方法 |

> 💡 **关键优势总结**：V-Steer 在**零训练成本 + 极低推理开销**下实现了接近最优的指令层级控制能力。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **Control Illusion (Geng et al., 2025)**  
   - 二元指令冲突基准，测试模型是否遵循更高优先级指令。
   - 包括多种 prompt 构造方式：`Pure`, `Task`, `Emph.`，以及简单/丰富上下文（simple/rich context）。
   - 社会权威偏见测试（CEO vs. Intern, Nature paper vs. blog 等），检验社会暗示对 IH 的干扰。

2. **IHEval (Zhang et al., 2025)**  
   - 更全面的多源指令层级评估框架。
   - 覆盖四类场景：System > User > History > Tool Outputs。
   - 包含多个子任务类别：
     - Rule Following（单轮/多轮）
     - Task Execution（提取、生成、分类）
     - Safety Defense（劫持防御、信息提取）
     - Tool Use（工具注入攻击）

---

### **实验设置与评估指标**

#### **评估指标**
- **Primary Constraint Accuracy (%)**：模型是否遵守了预期的高优先级约束（如系统指令），通过程序化自动判断。
- **Average Score on IHEval**：综合多个子任务的表现平均得分。
- **Generation Collapse Rate**：输出中出现重复 n-gram（>2次）的比例，衡量生成退化风险。

#### **模型范围**
跨 **7B 到 70B 参数规模**，涵盖：
- Llama-3.1 系列（8B, 70B）
- Qwen2.5 系列（7B, 14B, 32B）

所有模型均为 instruction-tuned 版本。

#### **硬件与精度**
- 单张 NVIDIA H200 GPU
- 模型加载为 bf16，Llama-70B 使用 INT8 量化

---

### **基线方法对比**

| 类别 | 方法 |
|------|------|
| **Prompt-only Baselines** | 原始冲突提示（Conflict）、强调系统指令（Emph.）、添加层级说明（IPP） |
| **Training-based SoTA** | RealGuardrail (SFT/DPO), VerifierSup (SFT+GRPO), HieraCRO (iterative PO) |
| **Inference-time Methods** | InstABoost（增强注意力权重）、Attention Steering（如 PASTA） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **Control Illusion 上的结果（Tab. 1a）**
| Model | Conflict (%) | V-Steer (%) |
|-------|--------------|------------|
| Llama-3.1-8B | ≤18 | **79.8–85.6** |
| Llama-3.1-70B | 17.8 | **83.5–92.0** |
| Qwen2.5-7B | ~9–10 | **70.2–75.9** |

👉 **提升超过 60 个百分点**，从几乎完全失效到高度可靠。

#### ✅ **IHEval 上的整体表现（Tab. 2b）**
| Method | Qwen2.5-32B | Qwen2.5-14B | Qwen2.5-7B | Llama-3.1-8B |
|--------|-------------|-------------|-------------|---------------|
| Conflict | 42.8 | 29.1 | 19.8 | 11.4 |
| HieraCRO (SoTA train) | 65.2 | 52.5 | 41.8 | 46.5 |
| **V-Steer** | 63.8 | 53.8 | 37.0 | 38.3 |
| **V-Steer+Prompt** | **65.6** | **54.2** | 33.0 | **47.6** |

✅ **V-Steer+Prompt 在 3/4 尺寸模型上匹配或超越 SoTA 训练方法！**

---

### **消融实验结果**

#### 🔍 **Head Selection 策略对比（Tab. 3 & 12）**
| Head Selection | Primary Acc (%) | Collapse Rate (%) | Rel. Collapse |
|----------------|------------------|--------------------|----------------|
| DLA（本文方法） | 79.8–85.6 | **0.02** | **1×** |
| All heads | 80.6–86.3 | 0.29 | 14× |
| Random | ~60 | 0.35 | 17× |
| Complement of DLA | <17 | 0.38 | 19× |
| Gradient × Activation | ≈DLA | 0.02 | 1× |

📌 **结论**：DLA 准确识别关键 head；盲目 steering 所有 head 会导致严重生成退化。

#### 🔍 **Span Extraction 策略鲁棒性（Fig. 5 & Tab. 9）**
比较三种策略：
- **V-Steer**（精确 constraint 定位）
- **V-Simple**（整条 system/user message 作为 span）
- **V-Auto**（聚类注意力模式自动发现冲突 span）

| 方法 | 性能 | 成本 |
|------|------|------|
| V-Steer | 最佳 | 需人工标注或 LLM 抽取 |
| V-Simple | 接近最佳（多数情况） | **零抽取成本，最实用** |
| V-Auto | 表现不稳定 | 额外计算开销，收益有限 |

> ✅ **V-Simple 是性价比最高的默认选择**。

#### 🔍 **超参数敏感性分析（Fig. 6 & Tab. 13）**
- Boost 因子 $\gamma^+ \in [2.0, 3.0]$，Suppress 因子 $\gamma^- \in [0.5, 1.0]$ 区间内性能稳定。
- 过强的 suppress 可能损害通用能力（如 MMLU 下降），但可通过调节 $\gamma^-$ 实现 **compliance-capability trade-off**。

#### 🔍 **通用能力保留测试（Tab. 6）**
在非冲突场景下启用 V-Steer（V-Simple）的影响：
| Benchmark | No Steer | V-Simple | Δ |
|----------|---------|----------|----|
| MMLU (5-shot) | 66.1 | 57.6 | -8.5 |
| IFEval (strict) | 82.4 | 80.1 | -2.3 |
| BBH (3-shot) | 69.6 | 67.7 | -1.9 |

📌 **适度损失可接受，且可通过调参缓解**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **指令层级崩溃普遍存在**：即使是最新的 LLMs，在冲突指令下也极易忽略系统提示。
2. ✅ **价值向量编辑是高效的干预途径**：通过修改 cached `V` 张量即可显著纠正行为偏差。
3. ✅ **DLA 是轻量有效的诊断工具**：无需昂贵的验证集调优，即可精准定位“坏头”。
4. ✅ **V-Steer 实现了效率与性能的平衡**：
   - 推理速度仅慢 **1%**（vs. 2.4× 慢的 Attention Steering）
   - 效果远超 prompt engineering，媲美 SoTA 训练方法
5. ✅ **方法具有良好的泛化性和鲁棒性**：在不同模型家族、尺寸、任务类型上均有效。

---

### **方法的局限性**
1. **依赖 span 定义**：虽然 V-Simple 表现良好，但在复杂 prompt 中仍可能误判 boost/suppress 区域。
2. **潜在的副作用**：过度抑制可能导致任务相关输入也被削弱（尤其当 user message 包含任务描述时）。
3. **未处理动态冲突演化**：当前只基于第一步 DLA 做一次性编辑，未考虑后续生成过程中注意力变化。
4. **对非 attention-based 模型无效**：假设模型架构为标准 Transformer。

---

### **未来工作方向**
1. **自动化 span 发现**：开发更可靠的 unsupervised 方法（如 V-Auto 改进版）来定位冲突指令。
2. **动态 steering**：在生成过程中持续监控并调整 value 编辑策略。
3. **探索因果机制**：研究被 DLA 识别出的 heads 是否构成稳定的 “role-priority circuits”。
4. **结合训练与推理**：
   - 在训练中引入 DLA-based 正则项，惩罚 favor 冲突低优先级 span 的 heads。
   - 设计 cache-aware fine-tuning，学习 layer-specific value scaling coefficients。
5. **扩展至其他控制目标**：如风格控制、事实一致性、价值观对齐等。

---

> 🔗 **代码已开源**：[https://github.com/cindy2000sh/v-steer](https://github.com/cindy2000sh/v-steer)

--- 

📌 **一句话总结**：  
**V-Steer 提供了一种高效、免训练、低开销的推理时干预方案，能够在不牺牲解码速度的前提下，大幅提升 LLM 对指令层级的遵从能力，是实现安全可控生成的重要一步。**

</details>

---

### 15. [Efficient Heteroscedastic Bayesian Optimization for Risk-Aware AutoRL](https://arxiv.org/abs/2607.26680)

**Authors**: Mingxuan Che, Tsung-Yuan Tseng, Theresa Eimer, Marius Lindauer, Alexander von Rohr  
**Category**: cs.LG  
**Published**: 2026-07-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.26680v1  

#### Abstract
Reinforcement learning (RL) has shown remarkable success across a wide range of complex tasks. However, RL outcomes can be highly stochastic, and both expected performance and variability often depend on hyperparameter (HP) configurations. We propose efficient and risk-averse heteroscedastic Bayesia...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Efficient Heteroscedastic Bayesian Optimization for Risk-Aware AutoRL》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
强化学习（**Reinforcement Learning, RL**）训练过程具有高度随机性，相同的超参数（**Hyperparameter, HP**）配置在不同运行中可能产生差异巨大的性能结果。传统的超参数优化（**HPO**）方法通常只关注期望性能（如平均回报），而忽略了结果的方差（variability），导致选出的“最优”配置在实际部署时可能表现不稳定甚至失败。

这种高方差使得 HPO 过程变得“脆弱”（brittle），即优化出的配置无法可靠复现其性能，增加了重训练成本。因此，如何在 RL 中进行**风险感知**（risk-aware）的 HPO，以同时优化**高平均性能**和**低方差**，是一个关键挑战。

此外，现有的风险感知 HPO 方法（如 RAHBO）采用**固定重复次数**（fixed replication budget）来估计每个配置的方差，这在计算代价高昂的 RL 场景中效率低下。

---

### **提出的新方法：ERAHBO**
作者提出了 **Efficient and Risk-Averse Heteroscedastic Bayesian Optimization (ERAHBO)**，一种高效的异方差贝叶斯优化方法，用于风险感知的 AutoRL。

#### **核心思想**
- **建模均值与方差**：与 RAHBO 一样，ERAHBO 使用两个独立的 **Gaussian Process (GP)** 分别建模性能的**均值函数 $f(x)$** 和**方差代理函数 $p(x)$**。
- **优化目标为 Mean-Variance Objective**：
  $$
  \text{MV}(x) = f(x) - \alpha p(x)
  $$
  其中 $\alpha \geq 0$ 是风险厌恶系数，权衡平均性能与方差。
- **自适应重采样（Adaptive Re-sampling）**：这是 ERAHBO 的核心创新。它不再对每个配置使用固定的重复次数 $k$，而是动态决定是否继续采样某个配置。

#### **自适应停止准则**
对于当前查询的配置 $x_t$，算法持续采集样本，直到满足以下任一条件：
1. 其乐观估计（UCB of MV）低于当前最优配置的悲观估计（Incumbent LCB）：
   $$
   \text{UCB}_{\text{MV}}(x_t) \leq B_t := \max_{x_i \in D_{t-1}} \text{LCB}_{\text{MV}}(x_i)
   $$
2. 达到最大采样次数 $k_{\text{max}}$。

该策略确保只在“有希望且不确定”的配置上投入更多资源，显著提升**样本效率**。

---

### **相比现有方法的优势**
| 方法 | 缺陷 | ERAHBO 的改进 |
|------|------|----------------|
| **GP-UCB** | 只优化均值，忽略方差，选出的配置可能不稳定 | 显式建模并优化均值-方差权衡，提高可靠性 |
| **RAHBO** | 固定重复次数 $k$，无论配置好坏都采样相同次数，效率低 | 自适应采样，仅在必要时增加重复，节省预算 |
| **RAHBO ($k=2$ vs $k=20$)** | $k=2$ 方差估计不准；$k=20$ 浪费资源 | 自动平衡精度与效率，无需手动调 $k$ |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ARLBench (Becktepe et al., 2026)**：一个用于 RL 超参优化的基准工具。
- **自建离线数据集**：作者基于 ARLBench 生成了一个高质量的 RL 性能数据集，包含：
  - **3 种 RL 算法**：DQN、PPO、SAC
  - **19 个 RL 任务**：涵盖 Brax、Classic Control (CC)、XLand-Minigrid 等环境
  - **每任务 512 个 HP 配置**
  - **每个配置 50 次随机种子运行**（远高于以往工作的 5–10 次），以精确估计异方差性（heteroscedasticity）
- **公开发布**：代码与数据集已开源：https://github.com/LUH-AI/Efficient-Risk-Averse-BO

---

### **实验设置**
- **BO 设置**：
  - 初始点：5 个 Sobol 序列采样的配置
  - 最大评估次数：约 1500–1900 次（取决于任务）
  - 重复实验：20 次独立运行，报告均值与标准误
- **模型细节**：
  - 均值 $f(x)$：使用异方差 GP
  - 方差 $p(x)$：使用同方差 GP（简化）
  - $\alpha = 1$，$\beta = 1$（confidence parameter）
  - $k_{\text{min}} = 2$, $k_{\text{max}} = 20$

---

### **评估指标**
1. **Mean-Variance Regret**：
   $$
   \text{Regret}_T = \sum_{t=1}^T k_t \left( \text{MV}(x^*) - \text{MV}(x_t) \right)
   $$
   衡量累计风险感知损失。
2. **Simple Regret MV**：最终找到的最佳配置的 MV 值。
3. **Cumulative Regret MV**：整个优化过程的累计 MV 损失。
4. **Ranking Metrics**：各算法在所有任务上的平均排名。
5. **Sample Efficiency**：达到特定 regret 阈值所需的评估次数。

---

### **基线方法**
| 方法 | 类型 | 说明 |
|------|------|------|
| **GP-UCB** | 风险中立 | 只优化均值，$k=20$ |
| **RAHBO ($k=2$)** | 风险规避 | 固定小预算，效率高但估计不准 |
| **RAHBO ($k=20$)** | 风险规避 | 固定大预算，准确但低效 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表 1：算法在所有任务上的平均排名（越低越好）**
| Metric | ERAHBO | RAHBO($k=20$) | RAHBO($k=2$) | GP-UCB |
|--------|--------|---------------|--------------|--------|
| Simple Regret MV | **1.79±0.77** | 2.42±1.04 | 2.37±1.09 | 3.42±0.88 |
| Cumulative Regret MV | **1.79±0.77** | 2.58±0.82 | 2.32±1.34 | 3.32±0.86 |

👉 **ERAHBO 在统计上显著优于所有基线**。

---

#### **表 2：达到不同 Regret 阈值所需的平均评估次数**
| Threshold | ERAHBO | RAHBO($k=20$) | RAHBO($k=2$) | GP-UCB |
|----------|--------|----------------|--------------|--------|
| 75% | **40.8±5.9** | 62.3±9.1 | 80.6±43.2 | 83.7±25.1 |
| 50% | **107.6±14.9** | 132.7±15.7 | 139.2±91.6 | 197.7±52.5 |
| 25% | **193.6±32.7** | 304.6±41.5 | 147.9±72.2 | 359.2±98.5 |

👉 **ERAHBO 在几乎所有阈值下都最快收敛**，尤其在中高预算下优势明显。

---

### **消融实验结果**

#### **图 4：重复次数分布**
- **ERAHBO**：大多数配置仅采样 2–3 次即被拒绝，少数优质配置获得大量重复（>15 次）。
- **RAHBO**：所有配置均采样固定次数（2 或 20），无区分度。

✅ **验证了 ERAHBO 的自适应机制有效聚焦于 promising 配置**。

#### **图 7：不同 $k$ 下 RAHBO 的表现**
- RAHBO 的最佳 $k$ 因任务而异（Brax 偏好 $k=20$，XLand 偏好 $k=2$）。
- **ERAHBO 自动适应不同任务需求，在后期超越所有固定 $k$ 版本**。

#### **图 14：调度 $B_{\text{stop}}$ 的影响**
- 引入随时间递增的 $B_{\text{stop}}$（如指数或 sigmoid 调度），可进一步提升性能。
- 尤其在 XLand 上，**指数调度 ($\gamma=50$)** 显著改善早期探索。

---

## **4. 关键结论和发现**

### **主要发现**
1. **风险感知 HPO 对 RL 至关重要**：仅优化均值可能导致不稳定配置，引入方差惩罚可提升可靠性。
2. **自适应重采样显著提升效率**：ERAHBO 通过动态分配采样次数，在保证准确性的同时大幅减少不必要的评估。
3. **ERAHBO 自动平衡探索与利用**：无需手动选择 $k$，适用于不同预算和任务。
4. **高质量数据集的价值**：提供 50 次重复的数据，使异方差性分析和风险感知方法评估成为可能。

---

### **方法的局限性**
1. **目标函数的尺度敏感性**：MV 目标 $f(x) - \alpha p(x)$ 对性能尺度敏感，需谨慎归一化。
2. **未直接建模尾部风险**：仅使用方差作为风险度量，不能捕捉极端失败事件（tail events）。
3. **理论界保守**：regret bound 是 worst-case 分析，未精确刻画自适应规则的实际收益。
4. **部分环境表现不佳**：在 XLand 等极难环境中，由于整体性能低，难以形成强 incumbent，导致过度采样。

---

### **未来工作方向**
1. **更丰富的风险度量**：如 CVaR、概率约束等，以更好控制失败概率。
2. **在线 HPO 验证**：当前实验基于离线数据集，未来可在真实在线 RL 训练中验证。
3. **自适应 $\alpha$ 调整**：根据搜索进度动态调整风险厌恶程度。
4. **扩展至多目标与多任务 AutoRL**。
5. **结合零时代 HPO 方法**：如权重继承、梯度预测等，进一步加速搜索。

---

> ✅ **总结一句话**：  
> **ERAHBO 通过自适应重采样机制，在不牺牲风险感知能力的前提下，显著提升了异方差贝叶斯优化在 AutoRL 中的样本效率，是迈向可靠、高效自动化强化学习的重要一步。**

</details>

---

### 16. [CADENCE: A Cardiac Atom Dictionary for Interpretable Neural Concept Extraction from ECG Foundation Models](https://arxiv.org/abs/2607.25244)

**Authors**: Yixuan Duan, Arjun Naik, Sadeer Al-Kindi, Wei Qiu  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.25244v1  

#### Abstract
Foundation models for 12-lead electrocardiograms (ECGs) transfer well across clinical tasks, but the physiological knowledge encoded in their representations remains opaque. We present CADENCE, a framework that decomposes an ECG foundation model into a human-interpretable, queryable dictionary of ph...

---

### 17. [Salient Knowledge Pathways: Sparse Cross-Modal Routing for Efficient Knowledge-Intensive Multimodal Question Answering](https://arxiv.org/abs/2607.25422)

**Authors**: Noor Islam S. Mohammad, Ulu\u{g} Bayaz{\i}t  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.25422v1  

#### Abstract
Knowledge-intensive multimodal question answering (KI-MMQA) sits at the intersection of three expensive primitives: long visual token sequences, dense retrieval over large external corpora, and full cross-modal fusion. Existing systems pay all three costs uniformly per query, even though only a smal...

---

### 18. [Distributed Constraint Optimization via Online Learning and Iterative Pricing with Application to Large-Scale Satellite Scheduling](https://arxiv.org/abs/2607.25835)

**Authors**: Itai Zilberstein, Pranav Rajbhandari, Steve Chien, Tuomas Sandholm  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.25835v1  

#### Abstract
Distributed constraint optimization problems (DCOPs) provide a popular framework for distributed decision making under limited communication, but many real-world instances are too large to solve monolithically. We address this challenge from two complementary directions. We revisit the connection be...

---

### 19. [Metis: Memory Foundation Model](https://arxiv.org/abs/2607.26760)

**Authors**: Zeyu Zhang, Ziliang Guo, Yihang Sun, Xichong Zhang, Xixuan Hao, Zehao Lin, Yang Zhang, Xiaoyan Zhao, Tong Shen, Bo Tang, Zhi-Qin John Xu, Junchi Yan, Haofen Wang, Xu Chen, Feiyu Xiong, Zhiyu Li, Tat-Seng Chua  
**Category**: cs.CL  
**Published**: 2026-07-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.26760v1  

#### Abstract
Recent advances in AI agents have increasingly internalized native capabilities into their underlying foundation models, giving rise to multimodal foundation models and large reasoning models. However, agent memory is still primarily implemented through external modules, leaving the native memory ca...

---

### 20. [ServerlessT2I: Efficient Text-to-Image Workflow Serving on a Serverless Platform](https://arxiv.org/abs/2607.26566)

**Authors**: Xiaoxiao Jiang, Suyi Li, Sheng Yao, Tianyu Feng, Lingyun Yang, Dapeng Nie, Haoran Yang, Wei Wang  
**Category**: cs.DC  
**Published**: 2026-07-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.26566v1  

#### Abstract
Text-to-image (T2I) workflows are increasingly deployed on serverless platforms because users often compose customized workflows and invoke them intermittently. Existing platforms typically deploy each workflow as an opaque GPU function, provisioning, placing, and scaling all constituent models in t...

---

### 21. [DHRCL:Training Code LLMs with Dense Hierarchical Rewards and Curriculum Learning](https://arxiv.org/abs/2607.26457)

**Authors**: Shuhang Wang, Ziming Li, Hui Cheng  
**Category**: cs.LG  
**Published**: 2026-07-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.26457v1  

#### Abstract
Reinforcement learning is a natural post-training paradigm for code-oriented large language models because generated programs can be evaluated through parsing, execution, unit tests, and structural analysis.However, existing methods often rely on sparse outcome rewards or statically combine heteroge...

---

### 22. [SkillRise: Agentic Reinforcement Learning for Cross-Task Skill Evolution](https://arxiv.org/abs/2607.26784)

**Authors**: Zhiyuan Yao, Yuxin Chen, Zhengxi Lu, Zishan Xu, Yueqing Sun, Yifu Guo, Yuquan Lu, Zhengzhou Cai, Kangning Zhang, Zhuowen Han, Zi-Han Wang, Ziang Ye, Qi Gu, Xunliang Cai, Weiwen Liu, Yongliang Shen  
**Category**: cs.LG  
**Published**: 2026-07-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.26784v1  

#### Abstract
Large language model agents often encounter related yet distinct tasks that share reusable solution patterns. Yet standard agentic reinforcement learning treats tasks as independent episodes, while existing approaches to skill learning either focus on repeated attempts of one task or use pipelines w...

---

### 23. [Kernel Forge: An Agent Harness for LLM-based Generation and Optimization of CUDA Kernels](https://arxiv.org/abs/2607.24762)

**Authors**: Joshua Brodsky, Dhravid Kumar, Savini Kashmira, Jayanaka Danatanarayana, Jason Mars, Krisztian Flautner, Lingjia Tang  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.24762v1  

#### Abstract
Machine learning models are increasingly embedded in everyday software, and most of their runtime is spent in a small set of compute kernels such as matrix multiplication, convolution, and normalization. Optimizing these kernels is one of the most direct ways to reduce latency and cost, but it has t...

---

### 24. [ProcAgent: An Agentic Framework for Procedural Task Guidance on Edge with Human-in-the-Loop](https://arxiv.org/abs/2607.24770)

**Authors**: Azizul Zahid, Subrata Biswas, Bashima Islam, Sai Swaminathan  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.24770v1  

#### Abstract
Procedural tasks such as furniture assembly and home repair impose substantial cognitive demands because users must interpret instructions, track task progress, reason about spatial state, and recover from errors while performing physical actions. Prior multimodal assistants have shown promise for p...

---

### 25. [A GAN-Based Framework for Robust Data Synthesis in Satellite Internet Observations](https://arxiv.org/abs/2607.24790)

**Authors**: Xiang Shi, Peng Hu  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.24790v1  

#### Abstract
Low-Earth orbit (LEO) satellite Internet has become an important infrastructure for enabling ubiquitous connectivity to align with the International Telecommunications Union vision for 6G telecommunications networks. However, current LEO satellite Internet observations often suffer from missing data...

---

### 26. [ODYSSE: Episode-wise Policy Optimization for Personalized Agentic Reasoning](https://arxiv.org/abs/2607.25369)

**Authors**: Jiaqi Zhang, Tong Chen, Junliang Yu, Quoc Viet Hung Nguyen, Hongzhi Yin  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.25369v1  

#### Abstract
Agentic systems have rapidly advanced in their ability to interact with real-world environments, leverage external tools, and provide services for users. However, unlike natural-world tasks that assume well-defined instructions, human-centered scenarios are characterized by ambiguous requests that l...

---

### 27. [TRWH: A Text-Driven Random Walk Heterogeneous GNN for Semantic-Aware Sparse Recommendation](https://arxiv.org/abs/2607.25471)

**Authors**: He Ma, Chen Liu  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.25471v1  

#### Abstract
Graph Neural Networks (GNNs) and Large Language Models (LLMs) have each advanced recommendation systems by modeling structural and semantic signals, respectively. However, integrating their complementary strengths remains challenging, particularly in sparse settings where maintaining semantic precis...

---

### 28. [Matrix-Free Photoacoustic Image Reconstruction via Sensor-Token Self-Attention](https://arxiv.org/abs/2607.25576)

**Authors**: Mary John, Shibili Said, Imad Barhumi, Sherzod Turaev, Mohamed Yahia  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.25576v1  

#### Abstract
Photoacoustic tomography (PAT) combines the optical absorption contrast of biological tissue with the spatial resolution of ultrasound, yet recovering the initial pressure distribution from sparse-view sensor measurements remains an ill-posed inverse problem. Iterative compressive-sensing solvers an...

---

### 29. [Joint Text-Audio Alignment for EEG-to-Text Decoding in Chinese Speech Production and Perception](https://arxiv.org/abs/2607.25626)

**Authors**: Tian Zheng, Xurong Xie, Xinxin Zhu, Xiaolan Peng, Feng Tian  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.25626v1  

#### Abstract
Decoding speech information directly from scalp electroencephalography (EEG) into text provides a potential non-invasive neural communication pathway for individuals with severe speech and motor impairments. Compared with invasive approaches such as electrocorticography, EEG is safer and more widely...

---

### 30. [OmniDelta: Skill-Driven Budget Allocation for Token Compression in OmniLLMs](https://arxiv.org/abs/2607.25669)

**Authors**: Haoyang Huang, Wenjie Huang, Tianqi Xu, Hongyaoxing Gu, Kang Tan, Yikai Fu, Yuhao Shen, Tianyu Liu, Baolin Zhang, Jun Zhang, Xinyi Hu, Jun Dai, Shuang Ge, Lei Chen, Yue Li, Mingchen Wang, Meng Zhang  
**Category**: cs.AI  
**Published**: 2026-07-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.25669v1  

#### Abstract
Emerging Omni-modal Large Language Models (OmniLLMs) enable unified understanding of text, audio, and video, but their long audio-video token sequences introduce substantial memory and inference costs. Existing compression methods mainly focus on selecting important tokens under fixed budgets, leavi...

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

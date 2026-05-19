# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-19 08:51:07 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Roll Out and Roll Back: Diffusion LLMs are Their Own Efficiency Teachers](https://arxiv.org/abs/2605.16941)

**Authors**: Fanqin Zeng, Feng Hong, Geng Yu, Huangjie Zheng, Xiaofeng Cao, Ya Zhang, Bo Han, Yanfeng Wang, Jiangchao Yao  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.16941v1  

#### Abstract
Diffusion Large Language Models (DLLMs) promise fast parallel generation, yet open-source DLLMs still face a severe quality-speed trade-off: accelerating decoding by revealing multiple tokens often causes substantial quality degradation. We attribute this dilemma to a train-inference mismatch amplif...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Roll Out and Roll Back: Diffusion LLMs are Their Own Efficiency Teachers*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本文针对 **Diffusion Large Language Models (DLLMs)** 在开放源码模型中普遍存在的 **质量-速度权衡**（quality-speed trade-off）问题。尽管DLLMs理论上支持并行生成以实现高速推理，但在实践中，加速解码（如一次生成多个token）往往导致显著的质量下降。

作者指出，这一困境源于 **训练-推理不匹配**（train-inference mismatch）和 **不可逆解码**（irreversible decoding）：
- **训练阶段**：模型从随机掩码状态重建文本，恢复顺序是随机的。
- **推理阶段**：需要一个自适应的去噪顺序——简单token应早揭示，依赖上下文的token应延迟揭示。
- 标准解码一旦揭示token就无法修改，导致早期错误被固化和传播。

---

### ✅ 提出的新方法与新思路

为解决上述问题，作者提出两个互补的方法：

#### （1）**WINO**：一种无需训练的可撤销解码算法（Revokable Decoding）
- **核心思想**：“宽进窄出”（Wide-In, Narrow-Out），即在每一步**激进地生成多个token**（Wide-In），然后利用增强的全局上下文**验证并重新掩码不可靠的token**（Narrow-Out）。
- **机制**：采用“起草-验证-回退”（draft-verify-fallback）流程，在单次前向传递中完成。
- **优势**：
  - 打破了解码的不可逆性，允许后期修正早期错误。
  - 支持更激进的并行生成，提升效率而不牺牲质量。
  - **无需额外训练**，可直接应用于现有DLLMs。

#### （2）**WINO+**：基于轨迹注入的训练框架（Trajectory Injection）
- **核心思想**：让DLLM成为自己的“效率教师”。利用WINO在推理时发现的可靠去噪顺序，将其**蒸馏到模型参数中**。
- **机制**：
  - 离线运行WINO，提取每个token的“最终化步骤”（finalization step）。
  - 构建按此顺序逐步揭示token的训练样本。
  - 训练模型遵循该顺序进行去噪，而非随机重建。
- **优势**：
  - 对齐训练与高效推理过程。
  - 减少对在线回退的依赖，进一步提升推理速度和质量。

---

### ✅ 相比现有方法的优势

| 方面 | 现有方法 | 本文方法（WINO / WINO+） |
|------|--------|--------------------------|
| **解码灵活性** | 固定顺序或固定数量并行生成 | 动态调整，可撤销错误 |
| **训练目标** | 随机掩码重建 | 学习最优去噪顺序 |
| **是否需训练** | — | WINO无需训练；WINO+通过轻量微调实现 |
| **性能增益** | 加速常伴随质量下降 | **加速同时提升质量** |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集

#### 语言任务（基于 **LLaDA-8B-Instruct**）：
- **GSM8K**：数学应用题
- **MATH-500**：复杂数学推理
- **HumanEval & MBPP**：代码生成
- **Countdown, Sudoku**：逻辑推理
- **ARC-E & ARC-C**：常识推理

#### 多模态任务（基于 **MMaDA-8B-MixCoT**）：
- **Flickr30K**：图像描述生成（captioning）
- **AI2D**：图表理解
- **MATH-Vision, MathVista**：视觉数学推理
- **MMMU, ScienceQA**：多学科多模态推理

---

### ✅ 实验设置与评估指标

| 设置项 | 描述 |
|------|------|
| **解码策略** | 半自回归（semi-autoregressive）采样，块长度128，生成长度256 |
| **评估方式** | 零样本（zero-shot），除Sudoku为4-shot |
| **评估指标** | - **准确率（Accuracy）**：用于大多数任务<br>- **CIDEr**：用于Flickr30K图像描述 |
| **效率指标** | - **解码步数（Decoding Steps）**<br>- **Tokens Per Second (TPS)**<br>- **步数缩减倍数（Step Reduction）** |
| **基线方法** | - 标准解码（1 token/step）<br>- Naive Parallel Sampling（固定多token/step） |

---

### ✅ 基线方法对比

| 方法 | 特点 | 缺陷 |
|------|------|------|
| **Standard Decoding** | 逐个token生成 | 效率低，未发挥DLLM并行潜力 |
| **Naive Parallel Sampling** | 每步生成M个token | 质量严重下降，尤其当M增大时 |
| **Entropy-Bounded Sampler [28]** | 基于熵动态解码 | 仍缺乏纠错机制，易受早期错误影响 |
| **本文方法（WINO/WINO+）** | 可撤销 + 学习最优顺序 | 显著优于所有基线 |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自Table I 和 Table II）

#### 🔹 在 **GSM8K** 上的结果：
| 方法 | 准确率 | 解码步数 | 步数缩减 | TPS | TPS加速 |
|------|--------|-----------|------------|-----|---------|
| LLaDA（标准） | 73.24% | 256 | 1.00× | 17.76 | 1.00× |
| **WINO** | **75.82%** (+2.58) | **41.93** | **6.10×** | **100.53** | **5.66×** |
| **WINO+** | **76.58%** (+3.34) | **37.47** | **6.83×** | **121.86** | **6.86×** |

> ✅ **结论**：**既提速又提质**

#### 🔹 在 **Flickr30K** 上的结果：
| 方法 | CIDEr | 解码步数 | 步数缩减 | TPS | TPS加速 |
|------|--------|-----------|------------|-----|---------|
| MMaDA（标准） | 53.67 | 256 | 1.00× | 6.41 | 1.00× |
| **WINO** | 53.83 | 25.47 | 10.05× | 55.11 | 8.60× |
| **WINO+** | **63.38** (+9.71) | **15.78** | **16.22×** | **106.07** | **16.55×** |

> ✅ **CIDEr大幅提升，且推理速度快16倍以上**

#### 🔹 在其他任务上的典型提升：
- **ARC-E**：准确率从59.13% → **84.97%**，步数减少10.3倍
- **ScienceQA**：准确率从30.89% → **53.84%**，TPS加速11.02×
- **Countdown**：准确率从24.21% → **48.05%**

---

### ✅ 消融实验结果

#### （1）**移除验证模块（Only Draft）**
- 若仅保留起草（T1=0.6），虽步数少但质量下降明显（GSM8K: 70.28% vs 75.82%）
- 表明**验证机制对维持高质量至关重要**

#### （2）**不同轨迹来源对比（Table VI）**
| 轨迹类型 | GSM8K 准确率 | 步数 | TPS |
|--------|-------------|------|-----|
| Random trajectory | 72.63% | 46.69 | 96.62 |
| **WINO trajectory** | **76.58%** | **37.47** | **121.86** |
> ✅ **证明WINO发现的顺序具有实际价值，优于随机顺序**

#### （3）**损失函数组件消融（Table VII）**
| 组件 | GSM8K 准确率 | 步数 |
|------|-------------|------|
| 仅 `L_tok` | 73.16% | 42.28 |
| + `L_defer` | 75.59% | 39.60 |
| + `L_sharp`（完整） | **76.58%** | **37.47** |
> ✅ 三个损失缺一不可：揭示正确token、抑制错误提前揭示、增强置信度

#### （4）**GPU内存开销（Fig. 7）**
- **WINO**：因引入shadow block，内存略增（+2.4%）
- **WINO+**：**内存低于原始模型**，因无需在线验证
> ✅ **WINO+实现了更高效的推理架构**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **DLLMs可以作为自身的“效率教师”**：
   - 通过可撤销解码（WINO）发现可靠的去噪顺序。
   - 再通过轨迹注入（WINO+）将该顺序内化为模型能力。
   - 形成“Roll Out（推理优化）→ Roll Back（训练反馈）”的闭环。

2. **打破传统质量-速度权衡**：
   - 不再是“提速必降质”，而是**提速的同时提升质量**。
   - 在多个任务上实现 **6–16倍的步数缩减**，同时准确率/CIDEr显著上升。

3. **验证机制的价值**：
   - 允许模型在获得更多信息后修正早期预测，有效缓解错误传播。

4. **学习“何时揭示”比“如何生成”更重要**：
   - WINO+的成功表明，**去噪顺序本身是一种强先验知识**，可用于指导训练。

---

### ⚠️ 方法的局限性

1. **依赖高质量的离线轨迹**：
   - WINO+需要在特定任务上运行WINO生成可靠轨迹，可能限制泛化性。
2. **计算开销集中在训练端**：
   - 虽然推理高效，但轨迹收集和LoRA微调仍需资源。
3. **当前主要验证于LLaDA/MMaDA架构**：
   - 是否广泛适用于其他DLLM架构有待验证。

---

### 🔮 未来工作方向

1. **扩展到更多模态与任务**：
   - 如语音、视频生成等更复杂的diffusion场景。
2. **探索在线自适应轨迹学习**：
   - 让模型在推理过程中实时更新去噪策略，形成持续学习机制。
3. **结合强化学习优化揭示策略**：
   - 将“揭示哪个token”建模为决策问题，进一步自动化。
4. **降低轨迹构建成本**：
   - 设计更高效的轨迹采样或合成方法，避免大量离线运行WINO。

---

## 总结

> **DLLMs can serve as their own efficiency teachers.**

本文提出了 **WINO** 和 **WINO+**，首次实现了 **在加速DLLM推理的同时提升生成质量**。其核心在于：
- 利用 **可撤销解码** 发现最优去噪顺序；
- 通过 **轨迹注入** 将该顺序内化为模型知识。

实验表明，该方法在语言与多模态任务上均取得显著突破，为下一代高效大模型提供了全新范式。

🔗 **代码地址**：[https://github.com/Feng-Hong/WINO-DLLM/tree/WINO-plus](https://github.com/Feng-Hong/WINO-DLLM/tree/WINO-plus)

</details>

---

### 2. [KVDrive: A Holistic Multi-Tier KV Cache Management System for Long-Context LLM Inference](https://arxiv.org/abs/2605.18071)

**Authors**: Jian Lin, Jiazhi Mi, Zicong Hong, Haodong Wang, Qianli Liu, Haodyue Zhang, Peng Li, Song Guo  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.18071v1  

#### Abstract
Supporting long-context LLMs is challenging due to the substantial memory demands of the key-value (KV) cache. Existing offloading systems store the full cache in host memory and selectively fetch critical entries during decoding, but this strategy quickly hits a ceiling: sparsity cannot be pushed f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# KVDRIVE: A Holistic Multi-Tier KV Cache Management System for Long-Context LLM Inference —— 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在长上下文（long-context）大语言模型（LLM）推理中，**Key-Value (KV) cache** 的内存开销随序列长度和批处理大小线性增长，迅速超出 GPU 内存容量。现有 offloading 系统将 KV cache 存储于主机内存（DRAM），但在解码阶段仍面临以下瓶颈：
- **频繁的数据迁移**：每步重新加载 critical KV entries，导致大量冗余 I/O。
- **流水线停顿**（pipeline stalls）：选择、获取、计算三个阶段串行执行，造成 GPU 空闲。
- **存储层级单一**：仅依赖 DRAM，无法应对超长上下文或大批次场景下的内存压力。

### 提出的新方法与核心创新
KVDRIVE 是一个**端到端的多级 KV cache 管理系统**，覆盖 GPU HBM、主机 DRAM 和 SSD，从系统层面协同优化缓存管理、流水线调度与跨层协调。其三大核心技术为：

#### （1）Attention-Based Cache Management（基于注意力的缓存管理）
- **滑动窗口复用机制**：维护一个包含多个最近 token 的 critical KV entries 滑动窗口，利用 temporal locality 实现增量更新，减少重复加载。
- **前向预判驱逐策略**（Lookahead Eviction）：根据当前 attention 分数预测未来重用概率，优先淘汰低分项，提升缓存命中率。
- **2D 层头动态缩放**（2D Window Scaling）：针对不同 Transformer 层和 attention head 的局部性差异，离线建模并分配最优窗口大小，最大化内存效率。

#### （2）Elastic Pipeline Scheduling（弹性流水线调度）
- **SFC 解耦设计**：将 Selection（选择）、Fetching（获取）、Computation（计算）解耦为独立可调度阶段。
- **微批处理并行化**（micro-batching）：通过细粒度 micro-batch 实现各阶段重叠执行，隐藏 I/O 和计算延迟。
- **联合参数调优**：自动调节 index 大小、cache 容量、micro-batch 尺寸以平衡准确率、吞吐与延迟。

#### （3）Coordinated Multi-Tier KV Storage（协调式多级 KV 存储）
- **重要性引导预热**（Importance-Guided Warm-Up）：在 prefill 阶段基于末尾 observation window 的 attention 权重对 prefix KV entries 进行重要性评分，并据此分级放置至 HBM、DRAM 或 SSD。
- **SSD 感知布局规划**（SSD-Aware Layout）：采用语义连续打包（semantic-contiguity packing）和层头分区（layer-head partitioning），将随机访问转化为顺序 I/O，提升 SSD 吞吐。
- **并行稀疏同步**（Parallel Sparse Synchronization）：按需拉取特定 KV 块，结合异步预取与 pinned memory 缓冲池，降低跨层传输开销。

### 相比现有方法的优势
| 维度 | 现有方法（如 Quest, ShadowKV, RetroInfer） | KVDRIVE |
|------|----------------------------------------|---------|
| **Caching** | 使用 LRU/LFU 等通用策略，未考虑 attention 行为 | 注意力感知缓存，支持滑动窗口与前向驱逐 |
| **Scheduling** | 串行流水线，存在严重 GPU stall | 弹性解耦调度，实现 fine-grained overlap |
| **Tiering** | 仅 DRAM offloading，难以扩展 | 支持 HBM-DRAM-SSD 协同管理，突破内存墙 |

---

## 2. 核心实验方法和设置

### 数据集
- **LongBench**：双语长文本理解基准，涵盖问答、摘要、推理等任务。
- **RULER**：评估检索、多跳推理、聚合与 QA 能力的综合性长上下文测试集。

### 模型
- Llama-3-8B-1048K（1M 上下文）
- Qwen-3-8B / Qwen-3-14B（128K 上下文）
- Phi-4-Mini-128K（128K 上下文）

### 硬件环境
| 类型 | 配置 |
|------|------|
| 成本敏感服务器 | NVIDIA L20 GPU (48GB), 100GB DDR5, NVMe SSD |
| 高端服务器 | NVIDIA H20 GPU (96GB), 200GB DDR5 |
| 工作站 | RTX 4090 (24GB), 120GB DDR5 |

### 基线方法对比
- **Original**：全 KV cache 保留在 GPU
- **FlexGen**：整层 offloading 到 DRAM
- **Quest**, **ShadowKV**, **RetroInfer**, **PQCache**, **MagicPIG**：主流 KV offloading 方法
- 所有 baseline 在统一框架下复现，确保公平比较

### 评估指标
- **生成吞吐量**（Generation Throughput, tokens/s）
- **缓存命中率**（Hit Rate）
- **Prefill 与 Decoding 延迟**
- **GPU 内存占用**
- **准确率**（RULER / LongBench 得分）
- **成本效益分析**

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **最高达 1.74× 吞吐提升**：相比当前最优系统（如 ShadowKV），KVDRIVE 在多种配置下平均提升约 70%，峰值达到 **1.74×**。
- **维持原始精度**：在 RULER 和 LongBench 上的表现与 full-attention baseline 几乎一致，无显著精度损失。
- **高缓存命中率**：得益于 lookahead eviction 和滑动窗口，critical KV 复用率高达 **80% 以上**。
- **低 Prefill 开销**：index 构建与 offloading 不引入额外延迟，prefill 时间与原生模型持平。

### 与基线方法对比结果
| 指标 | KVDRIVE vs Baseline |
|------|---------------------|
| **吞吐量** | 达到 ShadowKV 的 **1.7×**，FlexGen 的 **>100×**（后者因频繁重载几乎不可用） |
| **GPU 内存使用** | 显著低于 Quest 和 ShadowKV（得益于高效索引），略高于 FlexGen 但避免了其 I/O 瓶颈 |
| **可扩展性** | 支持 batch size=8、context=360k 场景，而多数 baseline 因 OOM 失败 |
| **SSD 利用能力** | 在 DRAM 不足时仍能保持 **60% 的 DRAM-only 性能**，远优于 naïve offloading |

### 消融实验结果
#### （1）Lookahead Eviction 效果（Table 3）
| 方法 | 平均命中率提升（vs LRU） |
|------|--------------------------|
| Quest | +0.9% ~ +1.3% |
| ShadowKV | +2.2% ~ +2.9% |
| RetroInfer | +1.3% ~ +3.9% |
| **KVDRIVE** | **+1.5% ~ +3.0%** |

表明该策略具有广泛适用性，在不同架构上均有效。

#### （2）2D Window Scaling（图 15）
- 相比 uniform window allocation，2D scaling 在相同 GPU 缓存预算下进一步降低 **15–30% 的数据传输量**。
- 特别是在局部性强的层/头上分配更大窗口，显著提升了资源利用率。

#### （3）Window Size 影响（图 16）
- 小 batch（BS=1）适合较小窗口（size=2），避免 lookup 开销过大；
- 大 batch（BS=4）受益于更大窗口（size=4），缓解 I/O 压力；
- 说明需动态调整以取得最佳平衡。

#### （4）Chunk Size 与 Centroid 数量（图 17–19）
- 最佳 chunk size = 4：太小 → I/O 频繁；太大 → 冗余数据多。
- Centroid 数量应与 context 长度成比例（如 120k context → 8192 centroids）。
- **减少 centroid 至 1/4 可节省 75% index 空间而不影响 accuracy**，证明聚类机制鲁棒。

---

## 4. 关键结论和发现

### 主要发现
1. **Temporal locality 在 long-context 中普遍存在**：相邻 token 的 attention pattern 高度相关，支持滑动窗口复用。
2. **传统串行流水线是性能瓶颈根源**：Selection 与 Fetching 阶段阻塞 GPU，必须通过解耦与重叠来消除 stall。
3. **SSD 可用于长上下文推理**：只要配合合理的布局与调度，SSD 能作为有效的第三级存储，突破 DRAM 容量限制。
4. **系统级协同优化优于算法级稀疏化**：KVDRIVE 不依赖更激进的 sparsity，而是通过 cache、pipeline、tiering 的联合设计实现高效。

### 方法的局限性
- **依赖离线 profiling**：2D window scaling 需预先运行 profiling 获取各 layer-head 的收益曲线。
- **实现复杂度较高**：涉及多级存储协调、异步调度、pinned memory 管理，部署难度大于纯软件方案。
- **对硬件带宽敏感**：若 PCIe/NVMe 带宽不足，SSD 层优势可能受限。

### 未来工作方向
1. **扩展至多模态模型**：探索视觉-语言模型中的 KV cache 访问模式差异。
2. **集成 Processing-in-Memory（PIM）硬件**：将部分 selection 或聚类操作下沉至存储端，进一步减少数据移动。
3. **混合精度存储**（Tiered Mixed-Precision Storage）：
   - HBM 中 hot blocks 使用 FP16，
   - SSD 中 cold blocks 使用 INT4 量化，
   - 在保证质量前提下提升有效带宽与容量。

---

> ✅ **总结一句话**：  
> KVDRIVE 通过 **attention-aware caching + elastic pipeline + coordinated multi-tier storage** 的系统级协同设计，在不牺牲精度的前提下，实现了高达 **1.74× 的吞吐提升**，首次使消费级 GPU（如 RTX 4090）能够高效服务百万级上下文 LLM 推理任务。

</details>

---

### 3. [JanusPipe: Efficient Pipeline Parallel Training for Machine Learning Interatomic Potentials](https://arxiv.org/abs/2605.18404)

**Authors**: Hongyu Wang, Weijian Liu, Hongtao Xu, Yan Wang, Mingzhen Li, Weile Jia, Guangming Tan  
**Category**: cs.DC  
**Published**: 2026-05-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.18404v1  

#### Abstract
Discovering atom-level phenomena requires molecular dynamics (MD) simulations with ab initio accuracy. Machine learning interatomic potentials (MLIPs) enable stable, high-accuracy MD simulations, and their models exhibit scaling-law trends similar to large language models. However, the lack of scala...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# JanusPipe: Efficient Pipeline Parallel Training for Machine Learning Interatomic Potentials 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 **Machine Learning Interatomic Potentials (MLIPs)**，尤其是**保守型MLIPs**（conservative MLIPs），在分子动力学（MD）模拟中表现出接近第一性原理（ab initio）的精度，且计算复杂度接近线性 $O(N)$，极大加速了大规模原子系统的模拟。然而，这类模型在训练时具有独特的 **double-backward 执行模式**，包含四个阶段：  
- **Forward Energy (FE)**  
- **Forward Force (FF)**  
- **Backward Force (BF)**  
- **Backward Energy (BE)**  

这种四阶段依赖关系与传统的 **Pipeline Parallelism (PP)** 调度机制（如 1F1B）不兼容，导致以下问题：
- **冗余计算**：FF 阶段需要重算 FE 的激活值以重建计算图。
- **参数复制**：FE 和 FF 分布在不同设备上，需跨设备同步梯度。
- **流水线气泡（pipeline bubbles）**：各阶段执行时间不均衡（$t_{FE} < t_{FF} < t_{BE} < t_{BF}$），破坏了流水线的高效重叠。

因此，**缺乏高效的分布式训练系统限制了保守型MLIPs的可扩展性**。

---

### 提出了什么新方法或新思路
作者提出 **JanusPipe**，一个专为保守型 MLIPs 设计的 **3D-parallel 分布式训练系统**，整合了 PP、DP 和 GP，并引入两个核心调度优化组件：

#### （1）SymFold
- 将传统的一阶 PP 调度转换为适用于二阶（four-phase）MLIPs 的指令列表。
- 通过“对称折叠”将 FE 和 FF 共置于同一物理设备上，**避免 FE 激活值的跨设备传输和重复计算**。
- 消除冗余通信（如 SAE/RAE），减少内存占用和同步开销。

#### （2）WaveK
- 针对四阶段执行时间不平衡的问题，提出一种自适应调度策略。
- 将多个 micro-batches 组织成 **WaveK 单元**，每个单元包含前向波（WaveK-F: FE+FF）和后向波（WaveK-B: BF+BE）。
- 通过**重叠相邻单元边界**，消除因阶段时间差异导致的流水线空闲周期。
- 支持离线调优选择最优的 `k`（每单元 micro-batch 数量），在吞吐量和显存之间取得平衡。

#### （3）GARS（Graph-Aware Re-Scheduling）
- 一种轻量级微批次重打包模块，缓解由于图大小分布长尾引起的负载不均衡。
- 在全局批次内重新排序并打包原子图，提升 PP、DP、GP 多维度并行下的负载均衡性。

---

### 相比现有方法的优势
| 特性 | JanusPipe | 传统 PP（如 1F1B/Hanayo） |
|------|----------|------------------------|
| 冗余计算 | ✅ 消除 FE 重算 | ❌ 必须在 FF 阶段重算 FE |
| 参数复制 | ✅ 减少跨设备同步 | ❌ 多设备复制参数 |
| 流水线效率 | ✅ 显著减少 bubbles | ❌ 四阶段失衡导致大量空转 |
| 显存控制 | ✅ 可控内存占用（via `k`） | ❌ 显存波动大，易 OOM |
| 适用性 | ✅ 专为 conservative MLIPs 定制 | ❌ 仅支持 first-order 模型 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **混合数据集**，由以下三个组成，按等概率采样：
  - **ODAC23**（Direct Air Capture）
  - **OMat24**（无机材料）
  - **OMol25**（有机分子）
- 每个训练迭代处理 **12,800 atoms** 的全局批次，划分为 **400 atoms / micro-batch**。

### 实验设置
- **硬件平台**：ARMv8 CPU + NVIDIA A100-40GB GPU × 32（8节点，每节点4卡）
- **软件环境**：CUDA 12.4, PyTorch 2.6
- **并行配置维度**：
  - **PP**（Pipeline Parallelism）：`P ∈ {4,8}`
  - **GP**（Graph Parallelism）：`G ∈ {2,4}`
  - **DP**（Data Parallelism）：`D ∈ {2,4}`

### 评估指标
- **端到端训练吞吐量**（throughput）：**atoms/sec**
- **峰值 GPU 显存占用**
- **强/弱扩展性效率**
- **消融实验**：逐步启用 SymFold、WaveK、GARS 分析性能增益

### 基线方法对比
- **1F1B-2nd**：基于 Megatron-LM 的 1F1B 调度，适配为支持四阶段训练。
- **Hanayo-2nd**：基于 Hanayo 的 wave-style 调度，同样进行二阶扩展。
- 所有基线均采用贪婪打包 micro-batch，**不包含 SymFold/WaveK/GARS**。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **32 GPUs** 上测试，JanusPipe 的平均性能提升如下：

| 指标 | JanusPipe vs. 1F1B-2nd | JanusPipe vs. Hanayo-2nd |
|------|------------------------|--------------------------|
| **平均吞吐量提升** | **1.51×** | **1.45×** |
| **峰值显存降低** | 最多 **20.56%** | 最多 **42.70%** |
| **OOM 缓解能力** | 成功运行 eSEN-220M（baseline OOM） | 同上 |

> 示例：在 `UMA-2.3B (P=8,G=2,D=2)` 下，JanusPipe 达到约 **1089 atoms/sec**，而 1F1B-2nd 仅为 **666 atoms/sec**，提速达 **1.64×**。

---

### 与基线方法的对比结果
- **所有配置下 JanusPipe 均显著优于基线**，尤其在高参数量模型（如 UMA-2.3B）和高并行度场景中优势更明显。
- **Hanayo-2nd 虽然能减少部分 bubbles，但由于其 wave 结构延长了激活值生命周期，反而导致更高显存占用**。
- **JanusPipe (k=P)** 已优于所有基线；使用离线调优选择 `k=Best` 进一步释放性能潜力。

---

### 消融实验结果
从 `1F1B-2nd` 出发，逐步添加组件的结果显示各模块互补：

| 组件 | 吞吐量相对提升 | 主要作用 |
|------|----------------|---------|
| + SymFold | ↑ 最多 **23%** | 消除冗余计算与参数复制 |
| + WaveK | ↑ 额外 **18%** | 改善四阶段重叠，减少 bubbles |
| + GARS | ↑ 平均 **11%**（最高 23%） | 缓解图大小不均导致的负载失衡 |

> 图表显示三者叠加后达到最高性能，验证了设计的正交性和有效性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **保守型 MLIPs 的 double-backward 执行模式是现有 PP 系统的瓶颈**，必须专门设计调度策略。
2. **SymFold 通过共置 FE/FF 实现零冗余计算**，是高效训练的基础。
3. **WaveK 利用四阶段的时间偏序关系（partial order）实现深度流水线重叠**，有效压缩 bubbles。
4. **GARS 在数据层面优化负载均衡**，进一步释放硬件利用率。
5. **JanusPipe 实现了真正的 3D-parallel（PP/DP/GP）训练框架**，支持大规模 MLIP 扩展。

---

### 方法的局限性
- **依赖离线调优确定最优 `k`**：虽然默认 `k=P` 表现良好，但仍需一次性的搜索过程。
- **目前仅针对 conservative MLIPs 设计**，非保守型 MLIPs 或其他科学模型需另行适配。
- **未集成 kernel-level 加速技术**（如 FlashTP、NequIP），仍有进一步优化空间。

---

### 未来工作方向
- 自动化 `k` 的在线动态调整，适应训练过程中变化的 workload。
- 将 JanusPipe 推广至更多类型的物理守恒模型（如电荷、动量预测）。
- 结合编译器优化与自动并行调度（如 Alpa），实现全栈自动化分布式训练。
- 探索在更大规模集群（百卡以上）上的部署与容错机制。

---

> **总结**：JanusPipe 是首个专为 **conservative MLIPs** 设计的高效 **Pipeline Parallel** 训练系统，解决了 double-backward 模式带来的冗余计算与流水线低效问题。其实验表明，在 32 GPU 上即可实现 **1.5× 以上的端到端吞吐提升**，并显著降低显存压力，为 MLIP 社区推动 scaling laws 研究提供了坚实基础设施。

</details>

---

### 4. [Scalable Knowledge Editing for Mixture-of-Experts LLMs via Tensor-Structured Updates](https://arxiv.org/abs/2605.16686)

**Authors**: Roman Maksimov, Vladimir Aletov, Dmitry Bylinkin, Daniil Medyakov, Vladimir Solodkin, Aleksandr Beznosikov  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.16686v1  

#### Abstract
Knowledge editing (KE) provides a lightweight alternative to repeated fine-tuning of LLMs. However, most existing KE methods target dense feed-forward layers, while modern LLMs increasingly adopt Mixture-of-Experts (MoE) architectures for their superior memory footprint and inference efficiency. Thi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scalable Knowledge Editing for Mixture-of-Experts LLMs via Tensor-Structured Updates

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代大型语言模型（LLMs）越来越多地采用 **Mixture-of-Experts (MoE)** 架构以提升参数规模与推理效率，而现有的 **Knowledge Editing (KE)** 方法大多针对传统的密集型（dense）前馈网络（FFN）设计。这导致在 MoE 模型上缺乏高效、可扩展的知识编辑工具。

现有方法如 **MoE-Edit** 虽然尝试解决该问题，但其依赖 **Block Coordinate Descent (BCD)** 的迭代优化过程，在编辑批量大或专家数量多时存在严重的计算瓶颈（逐层激活收集 + 逐专家更新），难以实现规模化应用。

### 提出了什么新方法或新思路
本文提出了一种名为 **MoTE (MoE Tucker Editor)** 的新框架，用于在 MoE 架构的 LLM 上进行高效、闭式（closed-form）的知识编辑。其核心创新包括：

- **利用 Woodbury Identity 进行低秩逆矩阵加速**  
  将 MoE 层的编辑目标重构为一个可应用 **Woodbury identity** 的形式，避免对大规模堆叠专家权重矩阵（$E \times d_{\text{hidden}}$）进行显式构造或求逆，仅需对大小为 $T \times T$ 的小矩阵求逆（$T$ 为编辑批次大小）。

- **引入 Tucker Decomposition 结构化更新**  
  将每个专家的参数更新 $\Delta W_j$ 视为一个三阶张量 $\Delta W \in \mathbb{R}^{E \times d_{\text{model}} \times d_{\text{hidden}}}$，并用 **Tucker 分解** 对其进行低秩参数化：
  $$
  \Delta W = \mathcal{G} \times_1 U_e \times_2 U_{\text{out}} \times_3 U_{\text{in}}
  $$
  其中因子矩阵 $U_e, U_{\text{out}}, U_{\text{in}}$ 在预训练权重上通过 HOSVD 提取，捕捉专家间的共享结构。

- **单次求解 + 多层传播机制**  
  仅在最顶层关键层执行一次完整的编辑求解，然后将压缩后的核心张量 $\mathcal{G}$ 按照 MEMIT 风格的残差衰减策略传播到下层，显著减少重复计算。

### 相比现有方法的优势
| 维度 | MoE-Edit | MoTE（本文） |
|------|---------|------------|
| 更新方式 | BCD 迭代优化 | 闭式解析解（closed-form） |
| 计算复杂度 | $O((E d_{\text{hidden}})^3)$ | $O(T^3 + T E d_{\text{hidden}})$ |
| 是否需要反向传播 | 是（多次 backward pass） | 否（backward-pass-free） |
| 批处理能力 | 弱（序列化处理） | 强（完全批量化） |
| 编辑速度 | 慢（数百秒级） | 快（最高提速 **6×**） |
| 编辑质量 | 最优或接近最优 | 接近 MoE-Edit，略低但可接受 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **COUNTERFACT**：单跳反事实编辑数据集，用于测试模型能否成功修改特定事实（如“谁是某国总统”）。
- **ZsRE (Zero-shot Relation Extraction)**：零样本关系抽取数据集，评估编辑后模型在未见过提示下的泛化能力。

### 实验设置和评估指标
#### 模型
在三个主流 MoE 模型上进行实验：
- **Qwen3-30B-A3B**（128 专家，top-8）
- **GPT-OSS-20B**（32 专家，top-4）
- **Qwen3.6-35B-A3B**（256 专家，top-8 + 1 共享专家）

#### 评估指标
- **Efficacy（有效性）**：编辑提示下是否正确输出目标对象。
- **Generalization（泛化性）**：在改写/同义提示下是否仍能正确回答。
- **Specificity（特异性）**：在无关邻近提示下是否保持原行为不变（衡量局部性）。
- **Utility**：前三项的平均值，综合评价编辑平衡性。

#### 基线方法对比
- **Fine-tuning (FT)** 和 **FT-L**（带 L∞ 约束）
- **AdaLoRA**：参数高效微调方法
- **UnKE**：基于外部记忆的编辑方法
- **MoE-Edit**：当前唯一专为 MoE 设计的 KE 方法（主要对比基准）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | Method | Eff. (%) | Gen. (%) | Spe. (%) | Utility (%) |
|-------|--------|----------|-----------|-----------|--------------|
| Qwen3-30B | MoE-Edit | **97.90** | **86.50** | **83.45** | **89.28** |
|           | MoTE     | 97.00    | 82.60     | 79.57     | 86.39        |
| GPT-OSS-20B | MoE-Edit | 96.90    | 38.80     | 80.93     | 72.21        |
|             | MoTE     | **94.60**| 37.55     | **81.85** | **71.33**    |
| Qwen3.6-35B | MoE-Edit | 93.40    | 78.55     | 77.99     | 83.25        |
|             | MoTE     | **89.20**| **68.44** | **75.33** | **77.64**    |

> ✅ MoTE 在所有设置中均取得**第二好成绩**，部分场景（如 GPT-OSS 上的 specificity）甚至优于 MoE-Edit。

### 与基线方法的对比结果
- MoTE 显著优于 FT、AdaLoRA、UnKE 等通用方法，在 efficacy 和 specificity 上优势明显。
- 相比 MoE-Edit，MoTE 的编辑质量略有下降（约 0.9–4.2 个百分点），但**换来了高达 6× 的速度提升**。
- 在 **GPT-OSS-20B** 上，MoTE 实现了更高的 **specificity**，说明其对原始模型行为的干扰更小。

### 消融实验结果（Table 2 & Appendix C）

#### （1）Tucker vs 无 Tucker（Table 2）
| Method | Eff. (%) | Time (s) |
|--------|----------|----------|
| MoE-Edit | 97.90 | 477.0 |
| Global + speedup（无 Tucker） | 94.80 | 81.6 |
| MoTE（含 Tucker） | 97.00 | 76.8 |

👉 表明：即使使用 Woodbury 加速，若不引入 Tucker 结构先验，编辑效果仍显著劣于 MoTE，验证了**结构化更新的重要性**。

#### （2）Whitening 消融（Table 5）
- 使用 **in-whitening**（基于隐藏状态协方差对齐输入空间）效果最佳。
- 不使用 whitening 导致性能大幅下降（如 Qwen3-30B 上从 97% → 63.5%）。
- 添加 out-whitening 无增益，甚至轻微降低性能。

#### （3）Null-space Projection 消融（Table 6）
- 移除 null-space 投影后，模型退化为随机预测器（efficacy ≈ 50%，utility ≈ 49–50%）。
- 证明该组件对于**防止知识干扰、维持 specificity** 至关重要。

#### （4）Routing Shift 分析（Table 4）
- MoTE 引起的路由分布变化（Routing Similarity 下降）小于 MoE-Edit，表明其对 MoE 路由机制的影响较小，更具稳定性。

---

## 4. 关键结论和发现

### 主要发现
1. **MoE 架构的知识编辑可以高效实现闭式求解**  
   通过结合 **Woodbury identity** 与 **Tucker decomposition**，首次实现了在 MoE 模型上的 scalable、backward-pass-free 知识编辑。

2. **结构先验至关重要**  
   单纯将 MoE 视为“扁平化”的专家集合会导致编辑失效；必须建模专家之间的**低多线性秩结构**才能获得高质量更新。

3. **速度与精度的良好权衡**  
   MoTE 牺牲少量编辑质量（<5%），换取高达 **6× 的加速**，使其更适合实际部署中的大规模连续编辑任务。

4. **Null-space projection 是稳定性的关键保障**  
   缺少此机制会导致模型行为崩溃，强调了保留原有知识的重要性。

### 方法的局限性
1. **大批次编辑时仍面临挑战**  
   当 $T > 1000$ 时，$T \times T$ 矩阵求逆成本上升，可能限制超大批量编辑效率。

2. **依赖 HOSVD 预处理**  
   需要在每个目标层上预先计算 Tucker 因子，增加了初始化开销，且对资源有限场景不友好。

3. **未支持动态路由变化建模**  
   假设路由在编辑前后基本稳定，未显式建模编辑对 router 的潜在影响。

### 未来工作方向
- 将 MoTE 扩展至 **MHSA 子层** 或 **全模型联合编辑**。
- 探索 **在线 Tucker 更新机制**，适应持续学习场景。
- 结合 **memory-based editing** 思路，构建 hybrid 编辑系统。
- 研究如何进一步降低对高维激活存储的需求，推动完全轻量化的 MoE 编辑。

---

> 📌 **总结一句话**：  
> MoTE 成功将闭式知识编辑从 dense 架构推广至现代 MoE LLMs，通过 **tensor-structured updates + Woodbury acceleration** 实现了**高效、可扩展、结构感知**的编辑方案，为未来 MoE 模型的动态维护提供了实用路径。

</details>

---

### 5. [HPC-LLM: Practical Domain Adaptation and Retrieval-Augmented Generation for HPC Support](https://arxiv.org/abs/2605.16347)

**Authors**: Nourin Shahin, Izzat Alsmadi  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.16347v1  

#### Abstract
Modern scientific research increasingly depends on High-Performance Computing (HPC) infrastructures, yet many researchers face significant operational barriers when interacting with cluster environments, job schedulers, GPU resources, and parallel computing frameworks. General-purpose large language...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《HPC-LLM: Practical Domain Adaptation and Retrieval-Augmented Generation for HPC Support》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代科学研究高度依赖 **High-Performance Computing (HPC)** 系统，但研究人员在使用集群环境、作业调度器（如 Slurm）、GPU 资源管理和并行计算框架时面临显著的操作障碍。通用大语言模型（**LLMs**）虽能提供基础编码帮助，但缺乏 HPC 领域所需的**领域特定操作知识**，例如集群策略、节点配置、文件系统层级等。

此外，HPC 文档通常分散于多个来源（官网、手册、软件指南），用户需手动整合信息，效率低下。

### 提出的新方法与思路
本文提出 **HPC-LLM** —— 一个面向 HPC 支持的**检索增强生成（RAG）与领域适配结合的语言助手框架**，其核心设计包括：

- **自动化文档摄取**：通过 Crawler Agent 自动爬取 80+ 所大学及机构的公开 HPC 文档（如 TACC、SDSC、Harvard RC 等）。
- **密集向量检索（Dense Retrieval）**：使用 `BGE-large-en-v1.5` 对查询进行嵌入，在 ChromaDB 中执行基于 HNSW 的相似度搜索。
- **轻量级领域适配（Domain Adaptation）**：采用 **QLoRA** 技术对 `Llama 3.1 8B-Instruct` 模型进行微调，构建 HPC 特定指令遵循能力。
- **模块化本地推理架构**：支持完全本地部署，无需依赖云 API，适用于资源受限或隐私敏感场景。

### 相比现有方法的优势
| 维度 | HPC-LLM 的优势 |
|------|----------------|
| **部署成本** | 只需 5GB VRAM 即可运行（RTX 3090 级别），远低于大型通用模型（如 Qwen 72B 需 38GB 4-bit） |
| **响应速度** | 推理延迟低（平均 ~5–9 秒），比同级别模型更快 |
| **准确性与相关性** | 结合 RAG 与领域微调，提升事实一致性，减少幻觉 |
| **可扩展性与可维护性** | 支持自动更新文档库、反馈闭环学习（feedback loop）、增量训练 |
| **开放性与复现性** | 全流程开源，提供完整训练与部署脚本 |

> ✅ **核心创新**：首次将 **QLoRA 微调 + RAG 检索 + 本地部署** 集成到 HPC 运维支持中，实现了“小模型 + 强领域知识”的高效解决方案。

---

## 2. 核心实验方法和设置

### 数据集构建
HPC-LLM 使用自建的 **HPC 指令数据集**，共约 **9,000–24,000 个问答对**，来源如下：

| 来源 | 数量 | 描述 |
|------|------|------|
| LLM-generated (Qwen 2.5 14B) | ~9,000–21,000 | 基于爬取文档块生成多样化 Q&A |
| Curated expert pairs | 30 | 人工编写的关键主题（Slurm、MPI、NUMA、Singularity 等） |
| GPU Advisor pairs | ~50 | H100/A100/V100 规格、VRAM 估算、多 GPU 并行范式等 |
| Template-based | ~500–2,000 | 模板填充生成的标准命令类问答 |
| Prompt datasets | ~1,000 | 外部收集的 HPC 相关 prompt |

> 📌 数据预处理包括去重（MD5）、清洗短文本、分块（350词+50重叠）、SHA-256 内容哈希防重复摄入。

### 实验设置
- **平台**：JetStream2 基础设施
- **基准测试集**：`hpc_1000_prompts.txt`（从 5,000 条中抽取），覆盖 11 类任务：
  - Job scheduling, MPI, GPU computing, Filesystems, Modules, Containers, Data transfer, Cluster access, Debugging, Policy, Workflows
- **运行模式**：
  - **Run 1 (V3)**：HPC-LLM vs. Phi-2, Phi-3, Mistral Nemo 12B, TinyLlama（均启用 RAG）
  - **Run 2**：HPC-LLM vs. Qwen 2.5 14B Instruct（未微调通用模型）等

### 评估指标
| 指标 | 说明 |
|------|------|
| **BERTScore F1** | 主要质量指标，衡量语义相似性（经 baseline rescaling） |
| **ROUGE-L** | 最长公共子序列 F1，反映表面词汇重叠（预期较低） |
| **Cosine Similarity** | 使用 BGE 嵌入比较 prompt 与 response |
| **HPC Domain Score** | 回答中包含的 HPC 术语比例（>50 个关键词） |
| **RAG Relevance** | 回答与 top-3 检索文档的余弦相似度 |
| **Latency (s)** | 端到端响应时间 |
| **Response Length** | 输出 token 数量（反映简洁性） |

### 基线方法对比
- **Phi-2 (2.7B)**、**Phi-3 (14B)**：微软小型高性能模型
- **Mistral Nemo 12B**：高效闭源训练模型
- **TinyLlama (1.1B)**：极小规模基线
- **Qwen 2.5 14B Instruct**：作为“更大通用模型”对照组

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 4 & 5）

#### Run 1: HPC-LLM vs. 小型模型（RAG 启用）
| Model | Size | Latency (s) | BERTScore F1 | Resp. Length |
|-------|------|-------------|---------------|--------------|
| Phi-2 | 2.7B | 5.71 | **0.846** | 138.1 |
| Phi-3 | 14B | 7.30 | 0.841 | 125.0 |
| Mistral Nemo | 12B | 6.12 | 0.841 | 123.5 |
| **HPC-LLM (Ours)** | **8B** | **5.22** | **0.808** | **90.7** |

> 🔍 分析：虽然 BERTScore 略低于 Phi-2，但 HPC-LLM 是第二快且输出最简短的模型，适合命令行交互。

#### Run 2: HPC-LLM vs. Qwen 2.5 14B（直接对比）
| Model | Size | Latency (s) | BERTScore F1 | VRAM Requirement |
|--------|------|-------------|---------------|------------------|
| Qwen 2.5 14B | 14B | 12.11 | 0.832 | ~28 GB (BF16) |
| **HPC-LLM (Ours)** | **8B** | **9.27** | **0.831** | **~5 GB (4-bit)** |

> ✅ **核心发现**：
- HPC-LLM 的 **BERTScore F1 仅落后 Qwen 0.001（0.831 vs 0.832）**
- 响应速度快 **23%（9.27s vs 12.11s）**
- 显存需求仅为 **1/3 左右**

### 消融分析（隐含结论）
尽管文中未明确列出消融实验表格，但从设计逻辑可推断以下有效性验证：

| 组件 | 贡献 |
|------|------|
| **RAG 检索** | 提升事实准确性，缓解模型参数记忆不足问题 |
| **QLoRA 微调** | 显著提升领域术语理解与命令生成能力（HPC Domain Score 达标） |
| **本地部署优化** | Flash Attention 2、torch.compile、auto-quantization 提升推理效率 20–40% |
| **反馈闭环机制** | 用户好评的回答自动加入训练集，实现持续学习 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **轻量级模型可通过领域适配媲美更大通用模型**  
   经 QLoRA 微调的 **8B 模型性能接近 14B 通用模型（Qwen 2.5）**，证明领域知识注入可弥补参数规模差距。

2. ✅ **RAG + 微调双路径优于单一策略**  
   检索提供动态知识支撑，微调增强指令遵循与术语理解，二者协同提升整体表现。

3. ✅ **本地化部署完全可行且高效**  
   在 **单张 RTX 3090（24GB VRAM）上即可部署完整系统**，满足科研机构对隐私、可控性和低成本的需求。

4. ✅ **输出更简洁实用**  
   HPC-LLM 输出长度显著短于其他模型（Run1: 90.7 words），更适合终端用户快速获取命令。

### 局限性
| 问题 | 说明 |
|------|------|
| 缺乏专家验证基准 | 当前评估依赖自动指标（BERTScore），缺少人工标注真值或可执行命令验证 |
| 合成数据风险 | 部分训练数据由 LLM 自动生成，存在潜在幻觉传播风险 |
| 检索质量依赖文档质量 | 若原始文档陈旧或结构混乱，会影响检索效果 |
| 安全性未充分评估 | 未测试生成命令的安全性（如误删文件、越权操作） |
| 多集群泛化能力有限 | 当前系统偏向美国高校 HPC 架构，跨区域适应性待验证 |

### 未来工作方向
1. **构建专家验证的 HPC Benchmark Dataset**  
   包含标准问题、正确答案、可执行命令验证脚本。

2. **引入可执行验证机制（Executable Verification）**  
   在沙箱环境中运行建议命令，检测是否达成目标。

3. **加强安全性与权限控制**  
   加入命令风险识别模块，防止生成危险操作。

4. **改进检索鲁棒性**  
   探索 chunking 策略优化、混合检索（vector + keyword）、知识图谱融合。

5. **集成 Tool Use 与 Scheduler Interaction**  
   实现真正意义上的“AI Agent”，不仅能回答问题，还能提交作业、监控状态、自动调试。

6. **开发 Retrieval-Aware Fine-Tuning（如 RAFT）**  
   在微调阶段显式利用检索上下文，进一步提升一致性。

---

## 总结
> **HPC-LLM 成功展示了“小而精”的 AI 助手在专业运维领域的可行性**。它通过 **QLoRA 领域微调 + RAG 检索增强 + 本地高效推理** 的组合拳，在仅需 **5GB VRAM** 的条件下，达到了与 **14B 级通用模型相当的语义质量**，同时具备更低延迟、更高安全性和更强可定制性。

该工作为 **科学计算基础设施中的 AI 辅助系统** 提供了一个可复现、可扩展、可部署的范例，推动了 LLM 在 HPC 社区的实际落地应用。

</details>

---

### 6. [CoX-MoE: Coalesced Expert Execution for High-Throughput MoE Inference with AMX-Enabled CPU-GPU Co-Execution](https://arxiv.org/abs/2605.17889)

**Authors**: Mu-Young Son, Yi Chen, Seungjae Yoo, Soongyu Choi Joo-Young Kim  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.17889v1  

#### Abstract
The Mixture-of-Experts (MoE) architecture improves computational efficiency via sparse expert activation, but throughput-oriented inference faces substantial GPU memory pressure due to a significant parameter size and intermediate data. Prior works attempt to mitigate this using expert offloading wi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CoX-MoE: Coalesced Expert Execution for High-Throughput MoE Inference with AMX-Enabled CPU-GPU Co-Execution

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Mixture-of-Experts (MoE) 模型虽然在计算效率上优于密集型大语言模型（dense LLMs），但由于其巨大的参数量和中间激活数据，在**高吞吐量推理场景下面临严重的 GPU 显存（VRAM）压力**。现有方法如 micro-batching 和专家卸载（expert offloading）存在以下问题：

- **micro-batching 导致操作强度（operational intensity）降低**，使专家计算变为内存瓶颈；
- CPU 卸载受限于 PCIe 带宽瓶颈，且传统 CPU 指令集（如 AVX）无法有效支持 GEMM 密集型任务；
- 缺乏对专家激活模式的静态分析，导致频繁的动态数据传输和负载不均衡。

### 提出的新方法与创新思路
本文提出 **CoX-MoE**，一种基于 **AMX-enabled CPU-GPU 协同执行系统**，通过两个核心机制优化 MoE 推理：

#### （1）Coalescing-Aware Orchestration Policy（融合感知调度策略）
- **摒弃 micro-batch 执行专家计算**，改为在整个 batch 上进行 **coalesced expert execution（融合专家执行）**，显著提升操作强度。
- 引入 **attention offloading** 策略：将 prefill 阶段中产生大量中间数据的 attention 运算卸载到 CPU，释放 GPU 显存用于保留更多专家权重。
- 联合优化非 MoE 操作（如 QKV 投影、attention）和专家计算的设备分配，实现资源高效利用。

#### （2）Expert-Aware Stratification (EAS) —— 专家感知分层预部署
- 在推理前对输入数据进行聚类采样，识别高频激活的“热”专家；
- 将这些专家**静态预加载至 GPU 显存**，减少运行时 PCIe 数据搬运开销；
- 实现 CPU 与 GPU 之间的负载平衡，避免因低效卸载造成性能倒退。

### 相比现有方法的优势
| 维度 | 传统方法（如 FlexGen, MoE-Lightning） | CoX-MoE |
|------|-------------------------------|--------|
| 批处理方式 | Micro-batching 分割 batch | 对专家使用 full batch 执行 |
| 卸载对象 | 主要卸载专家权重或激活 | 卸载 attention 中间数据以腾出显存 |
| CPU 利用 | 使用 AVX，GEMM 性能有限 | 启用 AMX，提供高达 ~144 TFLOPs BF16 性能 |
| 专家管理 | 动态加载/卸载，无先验知识 | 基于数据分布静态部署高频专家 |
| 系统利用率 | 显存碎片化严重，利用率低 | 显著提高 GPU/CPU 并行效率 |

---

## 2. 核心实验方法和设置

### 使用的模型
评估了三种具有代表性的 MoE 模型：
- **Mixtral-8x7B-Instruct (Mixtral)**：较少但更大的专家
- **DeepSeek-V2-Lite (DeepSeek)**：较多小专家
- **Qwen3-30B-A3B (Qwen3)**：混合架构，适合作为通用测试基准

### 实验平台（见 Table 2）
| 配置 | CPU | GPU | 内存/显存 | PCIe 版本 |
|------|-----|-----|----------|-----------|
| System I | Intel Xeon Platinum 8452Y (36核, 支持 AMX) | RTX 6000 Ada (48GB) | 512GB DDR5 | PCIe 4.0 |
| System II | 同上 | A100 (80GB) | 同上 | PCIe 4.0 |
| System III | 同上 | H100 (80GB) | 同上 | PCIe 5.0 |

> 所有系统均启用 AMX 加速 CPU 端矩阵运算。

### 评估指标
- **端到端推理吞吐量（Throughput）**：单位为 **Tokens/s**
- **每层延迟（Per-layer latency）**
- **专家命中率（Expert Hit Ratio）**：GPU 上驻留专家被调用的比例
- **消融研究（Ablation Study）**：各组件对性能的贡献分解

### 基线方法对比
- **FlexGen**：基于 micro-batch 和 zig-zag 调度的经典卸载框架
- **MoE-Lightning**：当前最先进的 batch 推理系统，采用 CGO 流水线与分页专家权重管理

两者均将 decode 阶段的 attention 卸载至 CPU。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（图 7）
在 **batch size = 1024** 下测试不同输入长度（Lin）和输出长度（Lout）的表现：

| 场景 | 相比 MoE-Lightning 提升 | 相比 FlexGen 提升 |
|------|------------------------|------------------|
| 平均吞吐提升 | **1.7× ~ 2.4×** | **3.4× ~ 7.1×** |
| 最佳情况（Mixtral, Lin=800, Lout=256） | **2.4×** | **7.1×** |

> CoX-MoE 在 prefill-heavy 和 decode-heavy 工作负载下均保持稳定优势。

### 专家命中率表现（图 8）
- 当仅能缓存部分专家时（如 DeepSeek 最多 30 个，Qwen3 最多 50 个），EAS 方案相比随机选择：
  - **专家命中率高出约 40%**
  - 命中率达到 **~80% 时，吞吐提升达 1.47–1.50×**

### 消融实验结果（表 3）
在 Qwen3 上进行组件拆解分析（System I, B=512）：

| 技术组合 | 微批次大小 (m) | 吞吐量 (Tokens/s) | 相对提升 |
|---------|----------------|--------------------|-----------|
| Baseline (MoE-Lightning) | 256 | 16.8 | — |
| + Coalesced execution + AMX co-execution | 256 | 25.5 | **1.51×** |
| + Attention offloading | 512 | 32.3 | 1.26× |
| + 80% expert hit ratio (EAS) | 512 | 34.1 | 1.05× |

> 结果表明：
> - **coalesced 执行是最大性能驱动力（+51%）**
> - attention 卸载允许更大 micro-batch，进一步释放潜力
> - EAS 虽然只作用于 prefill 阶段，仍带来可观增益

---

## 4. 关键结论和发现

### 主要发现
1. **micro-batching 是 MoE 推理中的隐性性能杀手**：它降低了专家计算的操作强度，使其陷入内存瓶颈，反而加剧了延迟。
2. **AMX 极大地提升了 CPU 的 GEMM 能力**（~144 TFLOPs），使得 CPU 可以承担 prefill 阶段的 attention 计算，不再是“弱辅助”。
3. **战略性的 attention 卸载比专家卸载更优**：因为 attention 产生的中间数据远大于专家参数本身，卸载前者可大幅释放 GPU 显存。
4. **专家激活高度倾斜（skewed）**：少数专家承担大部分计算负载，因此可通过静态部署显著减少动态传输开销。
5. **CoX-MoE 实现接近最优的系统资源利用率**，平均达到 SOTA 方法的 **2.0× 吞吐量提升**。

### 方法的局限性
- **依赖 AMX 支持的 CPU**：目前仅限第4代及以后的 Intel Xeon 处理器，限制了部署范围。
- **EAS 需要预分析阶段**：适用于已知 workload 的离线批处理场景，难以应用于完全动态的在线请求流。
- **未考虑多 GPU 扩展**：设计聚焦于单 GPU + CPU 协同，尚未探索分布式 MoE 场景下的扩展性。

### 未来工作方向
- 将 EAS 扩展为**在线自适应版本**，结合 runtime profiling 动态调整专家部署；
- 探索 **AMX + GPU tensor core 更细粒度的任务划分**，例如 layer-level 或 token-level 协同；
- 支持 **multi-GPU + multi-CPU 架构下的协同调度框架**；
- 将 CoX-MoE 思路推广至其他稀疏模型架构，如 Dynamic Networks 或 Sparse Attention 模型。

--- 

> ✅ **总结一句话**：  
> CoX-MoE 通过 **融合专家执行 + 注意力卸载 + 专家静态分层部署**，充分利用 AMX 强大的 CPU 矩阵能力，在单 GPU 场景下实现了高达 **7.1× 的吞吐提升**，为高吞吐 MoE 推理提供了全新的系统级优化范式。

</details>

---

### 7. [Latent Action Reparameterization for Efficient Agent Inference](https://arxiv.org/abs/2605.18597)

**Authors**: Wenhao Huang, Qingwen Zeng, Qiyue Chen, Zijie Guo, Yu Sun, Cheng Yang, Siru Ouyang, Jiri Gesi, Fang Wu, Jiayi Zhang, Huaming Chen, Bang Liu, Xiangru Tang, Chenglin Wu  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.18597v1  

#### Abstract
Large language model (LLM) agents often rely on long sequences of low-level textual actions, resulting in large effective decision horizons and high inference cost. While prior work has focused on improving inference efficiency through system-level optimizations or prompt engineering, we argue that ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Latent Action Reparameterization for Efficient Agent Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Large Language Model (LLM)** 的智能体（agent）在执行复杂任务时，通常依赖于长序列的低层次文本动作（low-level textual actions），导致**有效决策步数（effective decision horizon）过长**，从而引发以下问题：
- 推理延迟高（high inference latency）
- 计算成本高昂（prohibitive computational cost）
- 难以部署和实现实时交互

尽管已有研究通过系统优化、prompt engineering 或硬件加速来提升效率，但这些方法并未触及根本——即**动作空间本身的表示方式过于细粒度**。

### 提出了什么新方法或新思路
本文提出 **Latent Action Reparameterization (LAR)**，一种全新的框架，其核心思想是：
> 将原始的 token-level 动作空间重新参数化为一个紧凑的**潜在动作空间（latent action space）**，其中每个 latent action 对应一个多步语义行为（multi-step semantic behavior）。

#### 核心机制：
- **学习可执行的潜在动作**：从 agent 轨迹中自动识别出重复出现且具有“状态转移等价性”（transition-equivalent）的行为片段。
- **保留高熵内容显式输出**：仅压缩低熵、结构性强的部分（如工具调用模板、系统提示），而将高熵、参数绑定的内容（如搜索查询、具体数值）保留在显式空间，确保**executability**（可执行性）。
- **端到端集成**：latent actions 被作为新的 vocabulary tokens 加入模型，并通过 **trajectory-level distillation** 进行训练，无需运行时解码扩展。

### 相比现有方法的优势
| 方法类型 | 代表技术 | 局限性 | LAR 的优势 |
|--------|--------|------|-----------|
| Token-level 控制 | TokenSkip, ConciseHint | 可能破坏推理逻辑，稳定性差 | 不干预生成过程，保持语义完整性 |
| Context 压缩 | ACON | 压缩历史记忆，不减少决策步数 | 直接缩短**有效决策步数** |
| 手工宏指令 | Macros / Hierarchical Controllers | 需人工设计，泛化能力弱 | 完全自动学习，具备跨任务泛化能力 |

> ✅ **LAR 的本质突破在于：将“动作表示学习”视为与模型架构同等重要的建模选择**。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖三类典型 LLM agent 场景：

| 数据集 | 类型 | 特点 |
|-------|-----|------|
| **TriviaQA** | 多跳问答（Multi-hop QA） | 强调推理链与检索结合，结构重复性中等 |
| **KodCode** | 代码生成 | 高度结构化的语法模式，适合抽象 |
| **Mind2Web** | Web 工具使用 | 包含大量协议级 scaffolding（HTML、tool invocation） |

此外，在 **Musique**, **HumanEval**, **MBPP** 上进行零样本迁移测试，验证泛化性。

### 实验设置和评估指标

#### 主干模型（Backbone Models）
- `Qwen3-8B`
- `Llama-3.1-8B-Instruct`

#### 评估指标
| 指标 | 描述 |
|-----|------|
| **Task Success Rate / Accuracy** | 核心性能指标（如 TriviaQA 准确率） |
| **Action Token Reduction (%)** | 相比 vanilla 的动作 token 数量下降比例 |
| **Wall-clock Inference Time** | 实际推理耗时 |
| **Token Throughput (TT)** | 每秒处理 token 数 |
| **Peak GPU Memory (PG)** | 显存峰值占用 |
| **Reparameterization Rate** | 被替换为 latent action 的片段占比 |

#### 基线方法对比
| 基线 | 类型 | 说明 |
|-----|-----|------|
| **Vanilla** | 基准 | 原始 LLM agent |
| **CoT / ReAct** | 推理增强 | 标准思维链或反思式 agent |
| **TokenSkip** | Token 控制 | 跳过冗余 token 生成 |
| **ConciseHint** | Prompt 优化 | 引导更简洁推理 |
| **ACON** | Memory 压缩 | 压缩上下文历史 |

所有方法均在同一硬件（8×H200 GPU）、相同 decoding 设置下评测。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| Backbone | Method | TriviaQA ↑ | KodCode ↑ | Mind2Web ↑ | Action Tokens ↓ |
|---------|--------|------------|-----------|-------------|----------------|
| Qwen3-8B | Vanilla | 67.40 | 34.44 | 36.73 | — |
| Qwen3-8B | ReAct | 77.84 | 53.64 | — | — |
| Qwen3-8B | **LAR** | **80.09** (+2.25) | **54.30** (+0.66) | **39.84** (+3.11) | **-27.1%** |
| Llama-3.1 | Vanilla | 73.63 | 31.13 | 24.40 | — |
| Llama-3.1 | ReAct | 59.88 | 33.11 | — | — |
| Llama-3.1 | **LAR** | **72.46** (-1.17) | **35.10** (+1.99) | **28.30** (+3.90) | **-23.3%** |

> 🔍 注：LAR 在多数情况下不仅显著降低 token 数量，还**维持甚至提升任务成功率**。

### 系统级效率提升（见 Table 8）

| Model | Method | Token Throughput ↑ | Peak GPU Memory ↓ |
|-------|--------|--------------------|-------------------|
| Qwen | ReAct → LAR | +17.5% | -2.2 GB |
| Llama | ReAct → LAR | +0.4% ~ +7.5% | -0.1 ~ -0.4 GB |

✅ **LAR 推理无额外开销**：latent action 符号像普通 token 一样处理，直接带来 prefill 计算、KV-cache 和端到端延迟的节省。

### 消融实验结果

#### （1）零样本迁移能力（Held-out Generalization，Table 2）
| Backbone | Method | Musique | HumanEval | MBPP |
|---------|--------|--------|-----------|-------|
| Qwen3-8B | ReAct | 27.61 | 89.63 | 74.17 |
| Qwen3-8B | **LAR** | **26.57** | **91.46** | **75.50** |

➡️ 表明 learned latent actions 具备**领域级通用性**（如代码骨架、格式规范），而非数据集特异性。

#### （2）渐进式抽象消融（Progressive Abstraction Ablation，Fig. 3）
揭示三个阶段：
- **Phase I（适度抽象）**：性能随压缩率上升而提高（去除冗余结构）
- **Phase II（抽象边界）**：达到最优性能点
- **Phase III（过度抽象）**：一旦开始压缩高熵参数内容（如搜索词），性能急剧崩溃（categorical breakdown）

> 📌 发现：存在明确的“**抽象边界**”，由 next-token entropy 自然界定。

#### （3）动作等价性分析（Action Equivalence，Table 3）
引入 **LAR-PT**（padding to match length）：
- LAR > LAR-PT > ReAct
➡️ 证明性能增益来自**动作抽象本身**，而非单纯长度压缩。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **动作表示学习是提升 LLM agent 效率的关键路径**  
   改变动作空间的粒度比优化单个 token 生成更能从根本上缓解长程推理的成本问题。

2. ✅ **LAR 实现了高效且安全的抽象**  
   通过 entropy-based filtering 保证 only transition-equivalent, executable segments 被压缩，避免破坏环境交互。

3. ✅ **效率与性能可兼得**  
   平均减少 **~20–30% 动作 token**，同时**提升或持平任务成功率**，尤其在结构丰富任务（如 KodCode、Mind2Web）上表现突出。

4. ✅ **具备良好泛化性和可扩展性**
   - 跨任务迁移有效（HumanEval, MBPP）
   - 支持多任务联合训练（A.9 Unified Model）
   - 可扩展至更大模型（Qwen3-32B 上仍有效，A.10）
   - 可无缝接入工业框架（OpenClaw，A.14）

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖轨迹中的重复模式** | 若任务动作高度自由、缺乏结构性，则可压缩空间有限（如开放对话） |
| **无法处理动态变化的接口** | latent actions 假设工具调用格式稳定；若 protocol 频繁变更需重新训练 |
| **压缩率受限于 entropy 分布** | 必须牺牲部分压缩潜力以保障 executability，不能追求极致压缩 |

### 未来工作方向
1. **动态 latent action 学习**：在线识别新出现的可复用行为模式
2. **跨 agent 共享 latent action 字典**：构建通用 action abstraction library
3. **与 multi-token prediction 结合**：进一步加速推理
4. **应用于 RLHF 或 GRPO 中的 policy learning**：利用更稳定的 rollout 提升训练效率（文中已初步验证学习稳定性提升）

---

> 💡 **总结一句话**：  
> **LAR 通过学习可执行的 latent actions，将 LLM agent 的决策单位从“token”升级为“语义行为”，实现了推理效率的本质性跃迁，同时开辟了“动作表示学习”这一新研究方向。**

</details>

---

### 8. [TriAxialKV: Toward Extreme Low-Precision KV-Cache Quantization for Agentic Inference Tasks](https://arxiv.org/abs/2605.17170)

**Authors**: Hanzhang Shen, Haoran Wu, Yiren Zhao, Robert Mullins  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.17170v1  

#### Abstract
Agentic workloads have emerged as a major workload for LLM inference. They differ significantly from chat-only workloads, requiring long-context processing, the ability to handle multimodal inputs, and structured multi-turn interactions with tool calling capabilities. As a result, their context exhi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**TriAxialKV: Toward Extreme Low-Precision KV-Cache Quantization for Agentic Inference Tasks**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Agentic Inference**（代理型推理）任务中，LLM 需要处理多轮、多模态、长上下文的复杂交互（如工具调用、系统提示、图像输入等），导致 **KV Cache** 快速膨胀，成为内存瓶颈。传统 KV Cache 压缩方法（如均匀量化或单维度混合精度）忽略了不同 token 在 **时间、模态、语义角色** 上的异质性，导致压缩后精度显著下降。

### 🚀 提出的新方法：**TriAxialKV**
提出一种新型的 **混合精度 KV Cache 量化框架 TriAxialKV**，其核心创新在于：

- **三轴标签体系（Triaxial Tagging）**：为每个 token 分配一个三维标签：
  - **Temporal Axis（时间轴）**：`older`, `turn_m2`, `turn_m1`, `current`
  - **Modal Axis（模态轴）**：`text`, `image`
  - **Semantic Axis（语义轴）**：`inst`, `user`, `assistant`, `reasoning`, `tool_call`, `obs`, `delim`
- **基于标签的敏感度校准**：通过离线校准测量每种标签组合下 token 对 INT2/INT4 量化的敏感度（以 attention output MSE 衡量）。
- **动态比特分配器**：在固定平均 bitwidth 预算下，优先将 4-bit 分配给高敏感度标签组，其余使用 2-bit。
- **端到端系统实现**：
  - 支持 INT2/INT4 混合存储的 **分页内存池（Paged Memory Pool）**
  - 自定义 **融合 Triton 解码内核（Fused Triton Decode Kernel）**，支持实时解量化与 attention 计算

### 🔍 相比现有方法的优势
| 方法 | 缺陷 | TriAxialKV 的改进 |
|------|------|------------------|
| **Uniform Quantization**（如 KIVI, FP4） | 所有 token 同等对待，无法适应异构敏感度 | 引入三轴结构感知，差异化分配 bitwidth |
| **Single-Axis Methods**（如 PM-KVQ 时间轴, VL-Cache 模态轴） | 只考虑单一维度，忽略交叉影响 | 联合建模三个正交维度，更精细控制 |
| **Token Eviction 类方法** | 丢弃 token 可能丢失关键信息 | 不丢弃 token，仅压缩表示，保留完整上下文 |

> ✅ **核心优势**：在极低平均 bitwidth（~2.5–2.7）下仍能保持接近 BF16 的任务准确率，同时大幅提升吞吐量。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **BFCL Memory**：文本为主的函数调用任务，强调对 tool schema 和参数的精确记忆。
- **OSWorld**：多模态计算机使用任务，包含截图观察、GUI 操作、多轮交互，上下文长度可达 ~100K tokens。

### ⚙️ 实验设置
- **模型**：
  - 文本模型：Qwen3-14B/32B/235B, Falcon3-10B
  - 多模态模型：Qwen3-VL-8B/32B-Thinking, InternVL3.5-38B
- **硬件平台**：
  - NVIDIA B200（180GB HBM3e）
  - NVIDIA H100（80GB HBM3）
- **评估指标**：
  - **Task Accuracy (%)**：任务完成率
  - **End-to-End Throughput (tokens/sec)**：端到端生成速度
  - **Concurrent Requests**：最大并发请求数
  - **KV Cache Size Reduction / Expansion**

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **SGLang BF16** | 全精度 KV Cache，作为无损参考 |
| **SGLang FP4** | 统一 FP4 量化，代表 uniform low-bit 浮点方案 |
| **KIVI [26]** | 非对称 2-bit KV 量化，代表性低比特方案 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ **任务准确率（Accuracy）**
| 方法 | Qwen3-32B (BFCL) | Qwen3-VL-32B (OSWorld) |
|------|------------------|------------------------|
| SGLang BF16 | 25.78 | 39.20 |
| SGLang FP4 | 20.22 (-5.56) | 40.54 (+1.34) |
| KIVI | 21.56 (-4.22) | 41.49 (+2.29) |
| **TriAxialKV Mixed (Ours)** | **25.11 (-0.67)** | **40.59 (+1.39)** |

> ✔️ 在 BFCL 上接近 BF16（仅差 0.67 pts），远优于 FP4/KIVI  
> ✔️ 在 OSWorld 上甚至略超 BF16（+1.39 pts），说明合理压缩可避免噪声干扰

#### ✅ **端到端吞吐量（Throughput）**
在 Qwen3-VL-32B + OSWorld 上：
| 平台 | SGLang BF16 | **TriAxialKV** | 提升倍数 |
|------|-------------|---------------|---------|
| B200 | — | **1.32×** | ↑32% |
| H100 | — | **1.52×** | ↑52% |

> 💡 吞吐提升主要来自：
> - 更大的 batch size（并发请求提升 3.4–4.0×）
> - 更高效的 decode kernel 利用带宽

#### ✅ **KV Cache 容量扩展**
- 在相同内存预算下，**KV Cache 容量提升 4.5×**
- 支持更长上下文与更高并发

---

### 🔬 消融实验结果

#### 📉 **轴向消融（Ablation on Axes）**
| 方法 | Qwen3-14B (BFCL) | Qwen3-32B (BFCL) |
|------|------------------|------------------|
| Full (三轴) | 24.22 | 25.11 |
| No Temporal | 22.00 | 24.00 |
| No Semantic | 18.00 | 20.89 |

> 🔍 发现：
> - **Semantic Axis 最重要**：移除后准确率下降 >6 pts，因其保护了 `inst`（系统提示/工具签名）这类关键 token
> - **Temporal Axis 也有贡献**：允许对旧轮次进行更强压缩

#### 📏 **内存预算扫描（Memory Budget Sweep）**
在 Qwen3-14B 上调整平均 bitwidth：
| Average Bitwidth B | 2.5 | 2.6 | 2.7 |
|--------------------|-----|-----|-----|
| Accuracy | 16.22 | 19.56 | 24.22 |

> ⚠️ 结论：性能对 bitwidth 极其敏感，**必须通过校准选择最优操作点**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Agentic Prefill 存在结构性异质性**：token 的 KV 量化敏感度差异超过一个数量级，且主要由 **Temporal、Modal、Semantic** 三轴决定。
2. **联合建模三轴可实现极端低比特压缩**：在平均 ~2.6-bit 下仍能保持 BF16 级准确率。
3. **Semantic Role 是最关键维度**：系统提示（`inst`）和工具调用参数对量化极其敏感，需优先保护。
4. **系统级优化至关重要**：定制化的 **混合精度内存管理 + 融合 Triton kernel** 是实现高性能的关键。

### ⚠️ 局限性（Limitations）
1. **非零样本迁移**：需要 per-model/per-workload 的离线校准，不能直接迁移到全新任务。
2. **依赖 Chat Template 结构**：若 prompt 没有标准 role marker 或 bracket（如 `<think>`），tagger 可能失效。
3. **仅支持 INT2/INT4**：未探索 INT3 或更细粒度 bitwidth，存在进一步优化空间。

### 🔮 未来工作方向
- 扩展至更多 bitwidth（如 INT3, FP8）并设计通用 kernel
- 探索在线自适应 calibration，减少人工校准成本
- 将三轴思想应用于 **权重量化** 或 **注意力稀疏化**
- 支持非模板化 prompt 的自动语义解析 tagging

---

## 📌 总结一句话
> **TriAxialKV 通过联合建模 Temporal、Modal、Semantic 三轴结构，实现了面向 Agentic Inference 的极致低比特 KV Cache 压缩，在几乎不损失准确率的前提下，将吞吐提升 30–50%，KV 容量扩大 4.5 倍，是迈向高效长上下文代理系统的重要一步。**

</details>

---

### 9. [Skim: Speculative Execution for Fast and Efficient Web Agents](https://arxiv.org/abs/2605.16565)

**Authors**: Mike Wong, Kevin Hsieh, Suman Nath, Ravi Netravali  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.16565v1  

#### Abstract
Skim is a speculative execution framework for web agents that exploits the predictable structure of purpose-built websites. Today's web-agent expense is not intrinsic to the tasks but a property of how agents are composed: frontier-model inference, browser rendering, and ReAct-style planning are app...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Accio: Speculative Execution for Fast and Efficient Web Agents》核心总结

> **注意**：您提供的论文标题为 *“Skim”*，但实际文档内容是关于 **Accio** 的研究。因此以下总结基于正确论文标题 **《Accio: Speculative Execution for Fast and Efficient Web Agents》** 进行。

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

Web agents（基于 LLM 的网页代理）虽然能通过直接浏览真实网站完成复杂任务（如多步查询、登录页面导航、结构化信息提取），但其高昂的 **每任务成本（cost）** 和 **延迟（latency）** 严重限制了实际应用。这些问题并非源于任务本身，而是由当前通用组件的设计缺陷导致：

- 使用 **frontier LLMs**（如 GPT-4o）进行每一步推理；
- 依赖完整的 **headless browser** 渲染所有页面；
- 采用统一的 **sequential ReAct 执行框架**，无论步骤是否需要复杂决策。

这些组件本为更广泛的问题设计，在大多数常规 Web 任务中造成了资源浪费。

---

### 🚀 提出了什么新方法或新思路

提出 **Accio** —— 一种用于 Web agents 的 **speculative execution（推测执行）框架**，其核心思想是：

> 利用网站固有的结构性规律，在离线阶段构建可复用的导航路径模板，并在运行时尝试走“快速通道”，仅在失败时回退到完整 agent。

#### 核心机制包括：

1. **Offline Profiling（离线剖析）**
   - 对每个目标站点进行一次性的结构分析，生成 **site profile**，包含：
     - URL templates（例如搜索页、详情页的参数化 URL）
     - Search semantics（如何构造查询）
     - Pagination behavior（翻页逻辑）
     - Answer schemas（答案格式）
     - Capability metadata（是否需 JS 渲染、HTTP 可访问性等）

2. **Online Speculative Cascade（在线推测级联）**
   - 接收到任务后，尝试通过 profile 合成目标 URL 并发起轻量级获取（HTTP fetch + 轻模型处理）；
   - 使用一个小型 **judge model** 验证输出是否符合 query 和 schema；
   - 若验证失败，则将控制权交还给原始 ReAct agent，**以推测路径最终到达的 URL 作为 warm start**，避免重复探索前缀路径。

3. **两种部署模式**
   - **Accelerate Mode**：节省下来的计算资源直接体现为更低的成本与延迟。
   - **Aggregate Mode**：利用节省的预算并行运行多个 speculative trials，再由 verifier 选出最佳结果，提升准确率。

---

### 🔍 相比现有方法的优势

| 维度 | Accio 的优势 |
|------|-------------|
| **效率** | 显著降低 median cost（↓1.9×）和 latency（↓33.4%） |
| **准确性** | 不牺牲 end-to-end accuracy，错误推测会安全回退 |
| **通用性** | 支持任意自然语言任务描述，无需硬编码规则 |
| **鲁棒性** | 即使推测失败也能保留导航进度（warm start） |
| **自动化程度高** | 自动识别可加速的任务类型，无需人工编写脚本 |

相比 hand-crafted 优化程序，Accio 实现了接近同等性能的自动化版本，而传统 ReAct agent 完全无法利用此类结构规律。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **WebVoyager Benchmark**  
  包含来自 15 个真实世界网站的任务，涵盖开放式的导航、检索与结构化抽取任务。
  
- **WebShop**  
  基于 Amazon 商品数据构建的购物环境，专注于多步产品搜索与选择任务。

> 实验从两个 benchmark 中随机选取 **300+ 代表性任务** 进行评估。

---

### ⚙️ 实验设置和评估指标

#### 基线方法（Backbone Agents）
- **WebVoyager**：多模态 agent，基于截图进行视觉理解，代表强泛化能力的 baseline。
- **AgentOccam**：强调轻量推理与动作效率，使用文本 DOM 观察，注重成本控制。
- **BrowserUse**：生产导向的浏览器自动化框架，紧密耦合 LLM 与浏览器操作。

#### Accio 部署方式
- 在上述三种 agent 上层集成 Accio 框架，保持底层 agent 不变。
- 使用 **Qwen2.5-14B-Instruct** 作为 fast path 中的轻量模型，GPT-4o 用于 full ReAct agent。

#### 评估指标
| 指标 | 描述 |
|------|------|
| **End-to-end Latency** | 完整任务耗时（秒） |
| **Cost per Task ($)** | 总 API 成本（主要来自 LLM 调用） |
| **Task Success Rate / Accuracy** | 正确完成任务的比例 |
| **CDF 分布图** | 展示不同百分位下的性能分布 |
| **Warm Start vs Cold Start Fallback** | 回退时是否继承中间状态的影响 |

---

### 🆚 基线方法对比

| 方法 | 特点 | 是否被超越 |
|------|------|-----------|
| Off-the-shelf ReAct agents | 统一使用 full browser + frontier LLM + sequential loop | ✅ 被 Accio 显著超越 |
| Hand-engineered programs | 手写 URL 模板 + 直接抓取 + 小模型提取 | 接近性能上限，Accio 向其逼近 |
| Stateless retrieval (e.g., Perplexity) | 仅依赖搜索引擎摘要，不访问原网页 | ❌ 功能不同，不可比 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

| 指标 | 结果 |
|------|------|
| **Median Cost Reduction** | ↓ **1.9×** |
| **Median Latency Reduction** | ↓ **33.4%** |
| **End-to-End Accuracy** | 与 baseline 相当（within noise），无显著下降 |
| **Fast Path Success Rate** | 12.6–45.3% 的任务成功走完 fast path |
| **Fallback Improvement via Warm Start** | 明显优于 cold start，尾部延迟大幅压缩 |

> 数据来源：Figure 15–18, Table 2

---

### 🔁 与基线方法的对比结果

| 对比项 | Accio 表现 |
|--------|----------|
| vs ReAct-only | 所有 agent 后端均实现显著加速，尤其在 median 性能上 |
| vs Hand-optimized Programs | 达到 66.7–94.9% 的速度提升和 17.7–100.7× 的成本降低，接近手工优化水平 |
| vs Naive Substitution（直接替换小模型+HTTP） | 后者平均任务成功率下降 **60%**，Accio 通过验证机制避免此问题 |

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）**Warm Start 的影响（Figures 17–18）**
- 即使 speculative execution 失败，warm start 仍能显著减少 fallback 的延迟。
- 因为大多数任务共享导航前缀（如搜索 → 过滤 → 排序），推测路径即使未达终点也已推进至接近目标页。

#### （2）**Verifier 成本 vs 准确性（Section 5.3）**
- 使用 **lightweight verifier（Qwen-based）** 比 frontier-model verifier **便宜 11.5×**
- 仍能达到：
  - Precision: 82.0%
  - Recall: 86.2%
  - F1: 0.84
  - Accuracy: 86.9%

> 设计原则：偏向保守拒绝（bias toward rejection），确保不会提交错误答案。

#### （3）**Offline Profiling 开销（Figure 21）**
- 单个站点的 profiling 时间：
  - ~40% 站点 < 10 秒
  - ~60% 站点 < 12 秒
  - 最长约 16 秒（复杂 JS 或 bot detection 场景）
- 成本一次性摊销，不影响在线延迟。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Web sites 具有高度可预测的结构规律**
   - URL templates、search patterns、answer schemas 在同一 task type 下稳定存在。
   - 这些规律可以被离线捕获并用于加速。

2. **Speculative execution 是可行且高效的加速手段**
   - 多数 read-only 任务可通过 direct URL + HTTP fetch 快速解决。
   - 错误推测可通过 cheap verification 检测，并安全回退。

3. **Trajectory prefixes are highly shared**
   - 不同任务常共享相同的导航前缀（如“搜索商品”），使得 warm start 极具价值。

4. **Verification 可以非常廉价**
   - 利用 schema check + compressed state summary，即可实现高效可靠的判断，无需 full DOM + frontier model。

5. **节省的预算可用于 accuracy 提升（Aggregate Mode）**
   - 在相同成本下运行多个 speculative trials，accuracy 最高可提升 **16.7 pp（oracle selection）**，多数投票也可提升 **4.2 pp**。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **仅适用于 read-dominant workloads** | 当前不支持 purchase、form submission 等 stateful actions，因涉及更高安全要求 |
| **对 JS-heavy 或 anti-bot 强的站点效果有限** | 如 heavily dynamic UI、验证码机制，会导致 HTTP fetch 失效或验证频繁失败 |
| **Profile 更新机制被动响应** | 虽然能检测 drift 并自动 re-profile，但仍依赖运行时反馈，存在短暂窗口期风险 |
| **URL synthesis 依赖高质量 template** | 若 site 的 search semantics 对 query phrasing 敏感，则合成可能不准 |

---

### 🔮 未来工作方向

1. **扩展至 stateful workflows**
   - 引入 authentication management、transaction safety、side-effect validation 支持购买类任务。

2. **动态自适应 profiling**
   - 主动监控 site changes，提前更新 profile，而非等待失败触发。

3. **跨站迁移学习**
   - 利用相似站点（如同类电商）之间的共性，减少新站 profiling 成本。

4. **增强 multimodal speculative paths**
   - 对需视觉理解的任务（如 CAPTCHA、图表解读），探索轻量视觉模型 + screenshot 的 speculative 路径。

5. **更智能的 trial scheduling in Aggregate Mode**
   - 基于历史成功率动态调整 speculative trials 数量与策略，最大化 ROI。

---

## 总结一句话

> **Accio 通过“离线建模 + 在线推测 + 验证回退”的范式，首次将 speculative execution 成功应用于 Web agents，实现了接近手工优化程序的性能，同时保持全自动与高可靠性，在不牺牲 accuracy 的前提下将 median cost 降低 1.9×、latency 降低 33.4%。**

</details>

---

### 10. [NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning](https://arxiv.org/abs/2605.16757)

**Authors**: Haoran Lu, Luyang Fang, Wenxuan Zhong, Ping Ma  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.16757v1  

#### Abstract
Multi-agent language systems are often built as hand-designed workflows, where agents are assigned semantic roles and communication protocols are specified in advance. We propose NeuroMAS, a method that first treats a multi-agent language system as a trainable and scalable neural-network-like archit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning  
—— 核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Multi-Agent Language Systems** 多依赖于人工设计的工作流（hand-designed workflows），例如为每个 Agent 分配固定角色（如 planner、solver、critic）并预设通信协议。这种设计方式存在以下问题：
- **缺乏可扩展性**：组织结构是静态的，难以随任务复杂度增长而动态扩展。
- **设计成本高**：需要大量人力进行角色定义与流程编排。
- **优化割裂**：通常只优化 Agent 内部提示（prompt）或拓扑结构，而非将“组织”本身作为可训练对象。

该论文提出应将多智能体系统的“组织架构”视为一个**可学习、可扩展的计算结构**，而非仅靠工程设计。

---

### 🚀 提出的新方法：NeuroMAS

**NeuroMAS**（Neural Multi-Agent System）是一种将多智能体语言系统建模为**类神经网络的可训练架构**的新框架，其核心思想包括：

| 创新维度 | 具体实现 |
|--------|---------|
| **架构视角** | 将 LLM Agents 视为节点（nodes），中间文本消息为边（edges），构成一个**文本值（text-valued）的神经网络架构**。 |
| **角色自由（Role-Free）** | 节点不被赋予语义角色（如 planner 或 verifier），而是通过结构位置和训练过程自然演化出功能分工。 |
| **结构感知（Structure-Aware）** | 每个节点知道自己的层级、位置及输出格式要求，从而理解信息流动路径。 |
| **联合强化学习训练** | 所有节点共享最终任务奖励，通过 **REINFORCE-style policy gradients** 进行端到端联合训练。 |
| **渐进式生长（Progressive Growth）** | 支持从小规模训练好的系统逐步扩展至更大拓扑，提升大系统训练稳定性。 |

> 💡 类比：就像 CNN 中卷积层自动学习边缘检测器一样，NeuroMAS 中的节点在无监督角色分配下，通过奖励信号自发形成协作模式。

---

### 🔍 相比现有方法的优势

| 对比维度 | 传统方法 | NeuroMAS |
|--------|--------|----------|
| **Agent 角色设定** | 手动指定（planner/solver/critic） | 无预设角色，由训练动态生成 |
| **组织是否可训练** | 固定拓扑或离线搜索 | 拓扑+策略联合优化 |
| **训练目标一致性** | 各 Agent 可能有独立目标 | 所有节点共享同一终端奖励 |
| **可扩展性** | 难以规模化增长 | 支持宽度、深度、连接性的灵活扩展 |
| **参数效率** | 单一大模型需更多参数 | 模块化结构更适应分层任务，理论更高效 |

> ✅ **核心优势总结**：从“workflow engineering”转向“architecture design”，使多智能体系统具备类似神经网络的**可缩放、可训练、自组织能力**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验涵盖推理与代码生成六大基准任务，覆盖多种认知类型：

| 数据集 | 任务类型 | 描述 |
|------|--------|------|
| **ARC-Challenge** | 科学问答 | 复杂科学选择题，测试常识与推理能力 |
| **BBH-Navigate** | 空间推理 | BIG-Bench Hard 子集，判断导航指令是否可达终点 |
| **MMLU-Abstract Algebra** | 数学推理 | 抽象代数领域多选题 |
| **MMLU-College Physics** | 物理推理 | 大学水平物理知识 |
| **MMLU-Professional Medicine** | 医学知识 | 专业医学考试题 |
| **HumanEval** | 代码生成 | 函数级 Python 编程挑战，执行通过率评估 |

> ⚠️ 实验设定在资源受限环境下：所有 Agent 共享一个小的冻结 backbone（如 Qwen3-0.6B 或 Gemma-3-1B-IT），仅通过 LoRA adapter 微调。

---

### 🧪 实验设置与评估指标

| 设置项 | 说明 |
|------|------|
| **Backbone Models** | Qwen3-0.6B（主实验）、Gemma-3-1B-IT（鲁棒性验证） |
| **Adapter 方法** | 所有节点使用 **node-specific LoRA adapters**，backbone 参数冻结 |
| **训练算法** | **REINFORCE** with baseline（score-function gradient） |
| **评估方式** | Greedy decoding，max_new_tokens=200 |
| **准确率计算** | 输出标准化后匹配（exact-match），HumanEval 使用官方 unit test 执行验证 |
| **Topology 表示法** | `NeuroMAS-c`：表示每前向传播调用 c 次 LLM（含 output node）<br>例：NeuroMAS-3 = [1,1] 层结构（2 hidden + 1 output） |

---

### 🆚 基线方法对比

分为两类基线：

#### （1）**Frozen Backbone Methods**（不更新模型参数）
| 方法 | 类型 |
|-----|------|
| Direct Prompting | 零样本提示 |
| Self-Refine | 自反馈迭代改进 |
| Self-Check | 自检推理步骤 |
| MoA (Mixture-of-Agents) | 多 Agent 投票机制 |
| GoA (Graph of Agents) | 图结构协作 |
| GPTSwarm | 优化 agent graph 与 prompt |
| AgentNet | 去中心化协调网络 |

#### （2）**Trained Backbone Methods**（允许参数更新）
| 方法 | 说明 |
|-----|------|
| Single-LLM RL | 单 Agent 强化学习（等价于 NeuroMAS-1） |
| MALT | Generator-Verifier-Refiner 流水线训练 |
| CoLLM-CC | 多智能体 Actor-Critic 框架，集中式 critic |
| NeuroMAS-3/5/7 | 本文方法不同规模变体 |

> ✅ 特别强调与 **Single-LLM RL** 的对比：二者使用相同 backbone 和相近 trainable parameter budget（如 6.9M LoRA 参数），唯一区别在于是否引入分布式组织结构。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Method | ARC | Nav | Alg | Phy | Med | HEval | Trainable Params | LLM Calls |
|-------|-----|-----|-----|-----|-----|--------|------------------|-----------|
| **Direct Prompting** | 24.5 | 40.5 | 22.0 | 17.6 | 24.0 | 11.0 | 0 | 1 |
| **Self-Refine** | 37.5 | 40.5 | 25.0 | 24.5 | 37.5 | 21.3 | 0 | 2 |
| **GPTSwarm** | 46.0 | 40.5 | 25.0 | 25.5 | 43.5 | 12.8 | 0 | 11 |
| **Single-LLM RL** | 46.0 | 42.2 | 29.0 | 24.5 | 43.0 | 17.7 | 6.9M | 1 |
| **NeuroMAS-3** | **56.5** | **45.5** | **39.0** | **44.1** | **48.0** | **30.5** | 6.9M | 3 |
| **NeuroMAS-5** | 54.0 | 48.0 | 39.0 | 39.0 | 41.5 | 29.9 | 11.5M | 5 |
| **NeuroMAS-7** | 53.5 | 51.0 | **42.0** | 39.5 | 43.5 | **31.7** | 16.1M | 7 |

> ✅ **NeuroMAS-3 在所有任务上均优于最强 fixed-backbone 基线**，平均提升显著（如 Physics 提升 18.6 pts）。

---

### 🔁 与 Single-LLM RL 的参数匹配对比（关键控制实验）

尽管两者拥有相同的 **6.9M LoRA 参数预算**，NeuroMAS-3 仍全面超越：

| 任务 | NeuroMAS-3 vs Single-LLM RL 提升 |
|------|-------------------------------|
| ARC | +10.5 pp |
| Navigate | +3.3 pp |
| Algebra | +10.0 pp |
| Physics | +19.6 pp |
| Medicine | +5.0 pp |
| HumanEval | +12.8 pp |

> ✅ **结论**：性能增益并非来自更多参数，而是源于**可学习的组织结构带来的计算分解优势**。

---

### 🔍 消融实验：渐进式生长（Progressive Growth）

在 **Navigate** 任务上的消融研究（Table 3）揭示了组织扩展的关键发现：

| Method | From Scratch | Progressive Growth |
|--------|--------------|--------------------|
| NeuroMAS-3 | 45.5 | 45.5 |
| NeuroMAS-5 | 41.0 | **48.0** (+7.0) |
| NeuroMAS-7 | 40.5 | **51.0** (+10.5) |

> ❗ **重要发现**：更大的拓扑并不自动带来更好性能；**从零训练的大系统反而退化**，而通过 **progressive growth**（继承已有适配器 + 继续训练）则显著提升效果。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **组织即架构（Organization as Architecture）**
   - 多智能体系统可以像神经网络一样被设计为可训练、可扩展的模块化架构。
   - 不需要手动指定角色，**功能专业化可通过结构+奖励联合训练自然涌现**。

2. **参数效率优势**
   - 当任务具有层次分解结构时，模块化组织比单一大模型更具**参数效率**（compositional efficiency）。
   - 理论分析表明，在满足局部可学习性和稳定组合假设下，multi-agent 架构能达到更低的误差下界。

3. **组织扩展具有路径依赖性（Path-Dependent Scaling）**
   - 直接训练大规模 NeuroMAS 容易失败。
   - **渐进式生长（Progressive Growth）** 是有效扩展策略：从小系统出发，保留已有知识，逐步增加宽度或深度。

4. **NeuroMAS 是新的扩展轴（Scaling Axis）**
   - 除了扩大模型本身（scale up model），还可以通过**扩展组织结构**（scale up organization）来增强能力。
   - 这提供了一条低成本、高效益的能力增长路径，尤其适用于小 backbone 场景。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **推理开销高** | 每次前向需多次 LLM 调用（如 NeuroMAS-7 需 7 次），延迟和成本上升。 |
| **依赖任务可分解性** | 若任务无法有效分层（如全局直觉判断），则模块化优势消失。 |
| **拓扑设计仍需先验** | 当前实验采用全连接分层结构，最优拓扑搜索尚未完全自动化。 |
| **理论未保证收敛** | 理论分析基于容量假设，不保证实际训练一定能找到最优解。 |

---

### 🔮 未来工作方向

1. **自动化拓扑搜索（Neural Architecture Search for MAS）**
   - 结合 NAS 思想，自动发现最优 agent 拓扑结构。

2. **异构 Agent 与动态路由**
   - 引入不同类型 backbone 或动态决定信息流向（dynamic routing）。

3. **减少推理冗余**
   - 探索 early-exit、sparse activation 等机制降低 inference cost。

4. **应用于真实世界任务**
   - 如软件开发流水线、科研辅助、教育辅导等长期交互场景。

5. **结合 memory 与 stateful agents**
   - 当前为 stateless 文本传递，未来可引入持久状态以支持长程记忆。

---

## 📌 总结一句话

> **NeuroMAS 将多智能体系统重新定义为一种可训练、可扩展的“文本神经网络”架构，证明了“组织结构”本身可以成为 LLM 能力增长的新维度，且其有效性源于结构诱导的功能分化与联合强化学习的协同演化。**

</details>

---

### 11. [Unleashing LLMs in Bayesian Optimization: Preference-Guided Framework for Scientific Discovery](https://arxiv.org/abs/2605.17976)

**Authors**: Xinzhe Yuan, Zhuo Chen, Jianshu Zhang, Huan Xiong, Nanyang Ye, Yuqiang Li, Qinying Gu  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.17976v1  

#### Abstract
Scientific discovery is increasingly constrained by costly experiments and limited resources, underscoring the need for efficient optimization in AI for science. Bayesian Optimization (BO), though widely adopted for balancing exploration and exploitation, often exhibits slow cold-start performance a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Unleashing LLMs in Bayesian Optimization: Preference-Guided Framework for Scientific Discovery*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Bayesian Optimization (BO)** 在科学发现中面临两大挑战：
- **Cold-start 问题**：初始阶段缺乏有效数据，导致探索效率低下。
- **高维可扩展性差**：在高维参数空间中性能下降，难以应对复杂科学任务。

尽管已有研究尝试引入 **Large Language Models (LLMs)** 来辅助 BO（如 warm-start 初始化或候选生成），但这些方法仅将 LLM 作为一次性或辅助工具，未能**持续、系统地融合 LLM 的语义推理能力**到优化循环中。

### 🆕 提出的新方法：LLM-Guided Bayesian Optimization (LGBO)
作者提出了 **LGBO** ——首个将 LLM 的偏好（preferences）**连续嵌入**到 BO 循环中的框架，其核心机制是 **Region-Lifted Preference**。

#### 创新机制：Region-Lifted Preference
- LLM 不再仅提供建议点，而是输出两种格式之一：
  1. `[point, [...], ccc]`：建议一个具体点
  2. `[region, [[lb], [ub]], ccc]`：建议一个超矩形区域 + 置信度 `ccc ∈ [0,1]`
- 这些偏好被转化为对 **Gaussian Process (GP)** 代理模型的**均值偏移（mean shift）**，而不改变协方差结构。
- 偏移强度由 LLM 的置信度动态校准，确保在不确定性高的区域给予更强引导，在已知区域避免过度干预。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | LGBO 的改进 |
|------|--------|-------------|
| **标准 BO (GPBO)** | 冷启动慢，依赖随机初始化 | 引入 LLM 语义先验加速搜索 |
| **LLAMBO / ADO-LLM** | 仅用于 warm-start 或候选生成，后续仍由 acquisition function 主导 | **持续集成 LLM 偏好**，每轮更新 surrogate model |
| **ColaBO 类人类专家方法** | 假设偏好稳定且一次性给出，不适用于 LLM 动态推理 | 支持**迭代更新偏好**，适应 LLM 推理演化 |

> ✅ **核心优势**：  
> - **稳定性强**：通过均值偏移而非直接修改 acquisition function，保持 BO 的统计严谨性。  
> - **加速收敛**：当 LLM 偏好与目标一致时，显著提升样本效率。  
> - **鲁棒性强**：即使 LLM 给出误导性建议，理论证明其最坏情况性能不会显著劣于标准 BO。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验涵盖 **4 个干实验（dry benchmarks）** 和 **1 个湿实验（wet-lab）**，覆盖多个科学领域：

| 数据集 | 领域 | 参数维度 | 目标 |
|-------|------|----------|------|
| **LNP3** | 脂质纳米颗粒药物递送 | d=5 | 最大化载药量、包封率，最小化粒径 |
| **Cross-barrel** | 3D打印结构设计 | d=4 | 最大化机械韧性 |
| **Concrete** | 混凝土配比优化 | d=7 | 最大化抗压强度 |
| **HPLC** | 高效液相色谱参数调优 | d=6 | 最大化峰面积 |
| **Fe-Cr 电解液（湿实验）** | 氧化还原液流电池 | d=3 | 优化离子浓度以提升综合性能 |

> 所有干实验基于真实数据构建黑箱 oracle；湿实验为真实实验室环境下的在线优化。

### ⚙️ 实验设置与评估指标
- **Surrogate Model**：统一使用 **GP + Matérn-5/2 kernel**
- **Acquisition Function**：log-qEI
- **初始化**：
  - GPBO：Sobol 序列
  - LLAMBO / LGBO：共享相同的 LLM 建议点（公平比较）
- **LLM 模型**：Intern-S1-241B（科学领域预训练）
- **评估指标**：
  - 收敛速度（达到高价值解的速度）
  - 最终目标值（best observed value）
  - 多次运行的方差（稳定性）
  - 达到 90% 最优值所需轮数

### 🆚 对比的基线方法
| 基线 | 描述 |
|------|------|
| **GPBO** | 标准贝叶斯优化，无 LLM 参与 |
| **LLAMBO** | 使用 LLM 进行 warm-start 和候选生成，但决策仍由 acquisition function 控制 |
| **ColaLLM** | 将 ColaBO 框架适配至 LLM 场景，支持偏好建模 |
| **BOPRO** | 利用 LLM 作为隐式先验进行上下文学习 |
| **CAKE** | 使用 LLM 构造任务自适应 kernel 函数 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 干实验总体表现（图 2–4, 22–26）
- 在所有 4 个干实验中，**LGBO 均实现最快收敛和最高最终性能**。
- 特别是在 **LNP3** 上，LGBO 在前几轮迅速超越其他方法，并保持低方差。
- 在 **HPLC**（高噪声场景）下，LGBO 依然表现出更强鲁棒性和更高平均性能。

#### ✅ 湿实验突破性结果（Fe-Cr 电解液优化）
> **关键指标**：
- **LGBO 在第 6 轮即达到最佳观测值的 90%以上**
- 而 **GPBO 和 LLAMBO 需要超过 10 轮**

> 这表明 LGBO 在真实、昂贵、数据稀缺的实验环境中具有巨大潜力。

#### 📊 性能对比摘要
| 方法 | 收敛速度 | 最终性能 | 稳定性（方差） |
|------|----------|-----------|----------------|
| **LGBO** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ |
| **LLAMBO** | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ | ⭐⭐☆☆☆ |
| **GPBO** | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ |
| **BOPRO / CAKE / ColaLLM** | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ |

> LGBO 在早期阶段优势尤为明显，后期也能持续逼近最优。

### 🔍 消融实验结果（Ablation Studies）

#### （1）不同 LLM Backbones 的影响（图 5）
- 使用更大或科学领域微调的 LLM（如 Qwen3-235B-Instruct）可进一步提升性能。
- “Thinking” 模式模型虽最终性能略高，但收敛更慢，说明指令遵循能力更重要。
> ✅ 结论：**LGBO 框架对 LLM 类型不敏感，但更强的 LLM 能带来额外收益**。

#### （2）随机区域提升（Random Region Lifting）
- 若将 LLM 建议替换为**相同大小和置信度的随机区域**，性能大幅下降。
> ✅ 结论：**性能增益来自 LLM 提供的“有意义语义信号”，而非 lifting 机制本身**。

#### （3）初始化消融
- 即使 LGBO 和 LLAMBO 使用相同的 LLM 初始化点，**LGBO 仍显著优于 LLAMBO**。
> ✅ 结论：**优势主要来自“持续偏好集成”，而非“warm-start”效应**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 的语义推理可以被有效转化为数学上可处理的偏好信号**，并通过 **region-lifted preference** 安全融入 BO。
2. **持续集成 LLM 偏好比一次性 warm-start 更有效**，尤其在冷启动和高维场景下。
3. **LGBO 在理论上有保障**：
   - 当偏好对齐时，regret bound 显著收紧；
   - 当偏好错误时，性能退化可控，不会发散。
4. **在真实湿实验中验证成功**：仅用 6 轮即达 90% 最优，远超现有方法。

### ⚠️ 方法的局限性
- **依赖 LLM 输出格式的严格约束**：需精心设计 prompt 保证输出合规（如 `[region, [...]]` 格式）。
- **对 LLM 领域知识有一定依赖**：若 LLM 缺乏相关背景知识，引导效果可能有限。
- **当前仅支持静态 kernel**：未动态调整 kernel（如 CAKE），未来可结合。

### 🔮 未来工作方向
1. **扩展至多模态 LLM**：结合图像、分子结构等输入，增强引导能力。
2. **动态 kernel + 动态 mean 同时优化**：融合 CAKE 与 LGBO 思路。
3. **应用于更多真实自驱动实验室（Self-driving Lab）系统**。
4. **研究 LLM 自我修正机制**：让 LLM 根据反馈迭代优化自身偏好策略。

---

## ✅ 总结
**LGBO 是首个将 LLM 偏好“持续、稳定、可证安全”地嵌入 BO 框架的方法**。它不仅解决了传统 BO 的冷启动和高维瓶颈，还为 LLM 在科学发现中的深度整合提供了新范式。实验证明其在多种科学任务中均显著优于现有方法，尤其在真实实验中展现出强大的加速潜力，是迈向高效、智能科研自动化的重要一步。

</details>

---

### 12. [AutoVecCoder: Teaching LLMs to Generate Explicitly Vectorized Code](https://arxiv.org/abs/2605.17978)

**Authors**: Shangzhan Li, Xinyu Yin, Xuanyu Jin, Ye He, Yuxin Zhou, Yuxuan Li, Xu Han, Wanxiang Che, Qi Shi, Ting Liu, Maosong Sun  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.17978v1  

#### Abstract
Vectorization via Single Instruction, Multiple Data (SIMD) architectures is a cornerstone of high-performance computing. To fully exploit hardware potential, developers often resort to explicit vectorization using intrinsics, as compiler-based auto-vectorization frequently yields suboptimal results ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**AUTOVECCODER: Teaching LLMs to Generate Explicitly Vectorized Code**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代高性能计算依赖于 **SIMD（Single Instruction, Multiple Data）** 架构进行向量化以提升性能。然而：
- **编译器自动向量化（auto-vectorization）** 因保守的静态分析常无法生成最优代码；
- **显式向量化（explicit vectorization）** 需要开发者手动使用底层 **intrinsic** 指令，学习成本高、易出错且可移植性差；
- 当前 **Large Language Models (LLMs)** 在通用代码生成上表现优异，但在生成高效、语义正确的显式向量化代码方面能力有限，主要受限于：
  - 缺乏高质量的向量化训练数据；
  - 对低级硬件指令的严格约束理解不足。

### 🚀 提出的新方法与创新思路
作者提出 **AUTOVECCODER** 框架，首次系统性地将 LLMs 用于自动化显式向量化任务，其核心由两个模块构成：

#### （1）**VECPROMPT**：基于知识增强的数据合成管道
- 利用 **Retrieval-Augmented Generation (RAG)** 技术从官方 SIMD 文档中检索相关 intrinsic 定义，注入领域知识；
- 自动构建大规模、高质量的标量 → 向量化代码平行语料库；
- 包含合成模板（synthetic schemata）与真实代码片段（real-world collection），覆盖多种算子、数据类型和控制流复杂度。

#### （2）**VECRL**：性能驱动的强化学习算法
- 引入 **Correctness-Gated Performance Reward** 机制：
  - 只有功能正确的代码才获得性能奖励；
  - 性能增益通过 `tanh` 映射进行归一化，防止极端值主导训练；
- 使用 **Group Relative Policy Optimization (GRPO)** 进行策略优化，结合执行反馈动态调整生成策略，使模型不仅“写对”，还能“写快”。

### 🔍 相比现有方法的优势
| 维度 | 现有方法 | AUTOVECCODER |
|------|--------|-------------|
| 数据来源 | 手工标注 / 小规模数据 | 大规模自动合成 + RAG 注入权威知识 |
| 训练目标 | 功能正确性为主 | 联合优化：**正确性 + 执行效率** |
| 优化方式 | 监督微调（SFT）或零样本提示 | SFT + **性能感知的 RL 微调** |
| 性能潜力 | 接近或略优于 `-O3` 编译器 | **超越 `-O3` 优化结果**，达到 SOTA |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **SimdBench**（主评测基准）：
  - 多架构（x86, ARM, RISC-V）、多指令集（SSE, AVX）下的 SIMD 代码生成评测套件；
  - 本文聚焦于 **SSE 和 AVX 子集**；
- **MBPP** 和 **XLCoST**：用于提取真实世界 C/C++ 函数作为 seed program；
- 自建知识库：基于 **Intel SIMD Intrinsics Guide** 构建，供 RAG 检索使用。

### ⚙️ 实验设置
- **基础模型**：Qwen3-8B；
- **训练流程**：
  1. **SFT 阶段（VECPROMPT）**：在 7,685 个高质量样本上进行监督微调；
  2. **RL 阶段（VECRL）**：在 3,988 个样本上使用 GRPO 进行性能导向强化学习；
- **执行环境**：
  - 硬件：Intel Xeon Platinum 8374C CPU @ 2.70GHz；
  - 编译器：`clang++`，启用 `-O3` 优化；
  - 性能测量工具：Google Benchmark；
  - 沙箱系统：基于 ZeroMQ 实现资源隔离与并发执行控制。

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **SpeedUp** | $ \frac{T_{\text{scalar}}}{T_{\text{vector}}} $，即向量化版本相对于标量版本的速度提升倍数 |
| **Corr (Correctness)** | 功能等价的样本占比 |
| **fast₁** | 正确且 SpeedUp > 1 的样本比例（即确实更快） |
| **P50 / P75** | 在正确样本上的 SpeedUp 中位数与第75百分位数 |

### 🆚 基线方法对比
涵盖多个先进闭源与开源模型，在 **零样本（zero-shot）** 设置下统一测试：
- **DeepSeek-V3**, **DeepSeek-R1**
- **Qwen3-Coder-480B**, **Qwen3-Coder-Plus**
- **Gemini-2.5-Pro**
- **Claude-4-Sonnet**
- **GPT-5**
- **Grok4-Fast**

> ❗所有模型均未在推理时接入 RAG，仅 AUTOVECCODER 在训练阶段利用 RAG 构建数据。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）

| 模型 | AVX: Corr / fast₁ / P50 / P75 | SSE: Corr / fast₁ / P50 / P75 |
|------|-------------------------------|------------------------------|
| **AUTOVECCODER-8B (Ours)** | **76.76% / 47.35% / 0.99 / 2.74** | **77.35% / 53.53% / 1.02 / 2.22** |
| 第二名（Gemini-2.5-Pro） | 63.97% / 39.71% / 0.88 / **2.84** | 61.76% / 47.06% / 0.93 / **2.35** |

> ✅ AUTOVECCODER-8B 在 **正确率（Corr）** 和 **实用加速率（fast₁）** 上全面领先，尤其在 SSE 表现最佳。

### 🔁 与基线方法的对比结果
- 尽管部分大模型（如 Gemini）能达到更高的峰值速度（P75），但其 **正确率显著偏低**，说明存在大量无效或错误实现；
- **AUTOVECCODER-8B 是唯一一个在正确性和性能之间取得良好平衡的模型**；
- 其 8B 参数量远小于多数对比模型（如 Qwen3-480B），却实现了 **state-of-the-art 性能**，证明了训练范式的有效性。

### 🔍 消融实验结果

#### （1）**VECPROMPT 的作用（Table 2）**
| 设置 | AVX fast₁ | SSE fast₁ |
|------|----------|----------|
| 无 VECPROMPT（直接 RL） | 7.35% | 9.56% |
| 有 VECPROMPT（SFT + RL） | **47.35%** | **53.53%** |

> ✅ VECPROMPT 显著提升了初始正确率，缓解了 RL 中的稀疏奖励问题，并缩小了搜索空间，使 RL 更专注于性能优化。

#### （2）**奖励函数设计的影响（Table 3）**
| 奖励机制 | AVX fast₁ | SSE fast₁ |
|--------|----------|----------|
| Naive SpeedUp Reward (NSR) | 36.03% | 46.32% |
| VECRL（本文设计） | **41.18%** | **47.79%** |

> ✅ 分层奖励机制（correctness-gated + tanh-scaled）更稳定，避免了早期过度探索高风险模式导致后期性能下降的问题。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **性能感知训练比参数规模更重要**：
   - 即使是 8B 规模的小模型，通过 **VECPROMPT + VECRL** 联合训练，也能超越数十亿甚至数千亿参数的通用 LLM；
2. **执行反馈是突破编译器瓶颈的关键**：
   - AUTOVECCODER 成功生成了 **优于 GCC `-O3` 优化结果** 的代码，特别是在以下场景：
     - **Mask-based 控制流转换**
     - **非确定性循环迭代处理**
     - **指针别名/内存依赖的安全绕过**
     - **不规则内存访问模式重构**
3. **RAG 在低级编程中至关重要**：
   - 显著减少对 intrinsic 语义的“幻觉”；
   - 示例显示，未使用 RAG 时遗漏 `_mm256_min_ps` 导致逻辑错误，而 RAG 成功召回该指令并修复代码。

### ⚠️ 局限性
- **架构泛化能力有限**：
  - 当前主要验证于 x86 平台（SSE/AVX）；
  - 对 ARM NEON 或 RISC-V Vector 支持尚未充分测试；
- **向量化深度未被精细建模**：
  - 当前奖励仅关注端到端执行时间，未区分“纯内存优化”与“真正计算并行”；
  - 存在“浅层向量化”现象（如仅用于批量加载）；
- **适用范围仍局限于循环结构**：
  - 尚未扩展至复杂非结构化代码或 DSL（如 CUDA kernel）；

### 🔮 未来工作方向
1. 扩展至更多 SIMD 架构（NEON, SVE, RVV）；
2. 设计更细粒度的奖励信号以鼓励深层向量化；
3. 结合编译器反馈（如 LLVM IR 分析）进一步提升可靠性；
4. 探索自动迁移至不同硬件平台的能力（cross-architecture vectorization）；
5. 将框架推广至其他高性能 DSL（如 Triton, Halide）。

---

> 💡 **总结一句话**：  
> **AUTOVECCODER 证明了“小模型 + 领域知识注入 + 执行反馈强化学习”可以战胜“大模型 + 零样本提示”范式，在显式向量化这一关键 HPC 场景中实现质的飞跃。**

</details>

---

### 13. [InfoFlow: A Framework for Multi-Layer Transformer Analysis](https://arxiv.org/abs/2605.17930)

**Authors**: Penghao Yu, Haotian Jiang, Zeyu Bao, Qianxiao Li  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.17930v1  

#### Abstract
While the approximation properties of single-layer Transformer architectures have been studied in recent works, a rigorous theoretical understanding of the multi-layer setting remains limited. In this work, we establish that multi-layer Transformers possess fundamentally different approximation capa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# InfoFlow: A Framework for Multi-Layer Transformer Analysis 论文总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文旨在解决当前对**multi-layer Transformer**架构近似能力（approximation capability）缺乏系统理论理解的问题。尽管已有研究分析了单层Transformer的表达能力，但对于多层Transformer为何更强大，以及其内部信息传播机制如何影响模型效率，尚无统一的理论框架。

具体而言，作者关注以下核心问题：
- 多层Transformer相比单层在逼近特定任务时是否存在根本性的效率优势？
- 如果有，这种优势源于何种结构性机制？
- 如何建立一个可预测、可解释且与训练后行为一致的抽象框架来指导模型设计？

### 提出了什么新方法或新思路
作者提出了 **InfoFlow** —— 一种用于分析多层Transformer的**粗粒度理论框架**（coarse-grained framework），其核心思想是：

- **用离散的信息集（information set）替代连续的隐藏状态表示**：每个token在每一层都关联一个`I(t, l)`，表示该位置当前可访问的输入位置集合。
- **定义三种信息传播机制**，并赋予每种机制相应的参数代价律（parameter cost law）：
  1. **Max-position retrieval**：通过softmax attention高效检索注意力得分最高的token。
  2. **Global information aggregation**：将整个序列信息聚合到一个token中。
  3. **Specific position aggregation**：利用位置编码选择固定位置进行聚合。
- 引入 **Number of Comparison** 作为衡量目标任务复杂度的关键量，从而将任务需求与模型容量联系起来。

### 相比现有方法的优势
| 方面 | InfoFlow 的优势 |
|------|----------------|
| **理论深度** | 揭示了深度分离现象（depth separation）：某些任务单层需指数级参数，而两层仅需线性参数即可实现相同精度。 |
| **抽象层次** | 类似物理中的“有效理论”（effective theories），不追踪具体数值，而是捕捉主导计算模式，使复杂系统变得可分析。 |
| **预测能力** | 能够在训练前预测模型能否有效学习某类任务，并给出所需head数等超参建议。 |
| **一致性验证** | 不仅恢复已知理论边界，还与真实训练网络的行为高度一致（如argmax信息保留）。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
本研究主要基于**合成任务（synthetic tasks）** 构造数据集，以精确控制任务复杂度并排除语言先验干扰：

1. **Intrinsic Dimension Phenomenon 实验**
   - **目标函数**：$ F(X_T) = \sum_{i=1}^D \max_{1\leq s,t \leq T} x(s)^T A_i x(t) $
   - **输入分布**：$ x(t) \sim \mathcal{N}(0, I_d), d=4, T=64 $
   - **目的**：验证InfoFlow关于“内在维度D决定最小head数”的预测。

2. **Triangle-Center Task 实验**
   - **目标函数**：$ F(X_T) = \min_{1\leq t_1,t_2,t_3 \leq T} \|x(t_1)+x(t_2)+x(t_3)\| $
   - **输入分布**：$ x(t) \sim \mathcal{N}(0, 2I_2), \text{clamp norm } [0.2, 4] $
   - **目的**：验证高阶比较任务（Order of Comparison > 2）随长度增长难以逼近的现象。

3. **Max-position Retrieval 验证实验**
   - **任务**：训练模型完成广义D-retrieval任务。
   - **测量方式**：拟合线性映射 $ g: \mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^d $，评估是否能从post-attention state中恢复argmax位置的原始token。

4. **Specific Position Aggregation 实验**
   - **目标函数**：$ F(X_T) = x(1) + x(2) + x(3), T=10 $
   - **输入分布**：$ x(t) \sim \mathcal{N}(0, I_2) $
   - **目的**：验证模型能否仅依赖位置编码实现固定位置选择。

### 实验设置和评估指标
| 实验 | 模型结构 | Optimizer | Batch Size | Epochs | 评估指标 |
|------|----------|-----------|----------|--------|----------|
| Intrinsic Dim | 2-layer Transformer, $ h_1,h_2 \in \{D-1,D,D+1\} $ | AdamW ($10^{-3}$), cosine decay | - | 多种子采样 | Best validation NMSE |
| Triangle-Center | 同上，三种规模配置 | AdamW ($10^{-3}$) | - | - | Best validation NMSE |
| Max-pos Retrieval | 2-layer, varying $(h_1,h_2)$ | AdamW | - | 收敛为止 | Recovery Error Ratio (Trained/Initialized) |
| Specific Pos | 1-layer, 1-head | AdamW ($10^{-3}$) | - | 2000 | Validation MSE, Attention Weights |

> **NMSE（Normalized Mean Square Error）说明**：值接近0表示完美逼近；值接近1表示模型退化为常数预测器。

### 基线方法对比
本文未直接与其他模型架构（如RNN、CNN）对比，而是聚焦于**不同配置下的Transformer自身比较**，形成“内部基线”：
- 不同head数量组合：$(D,D)$ vs $(D-1,D)$ vs $(D,D-1)$
- 不同embedding dimension：24 vs 48
- 是否使用positional encoding

这些对比用于验证InfoFlow框架的预测能力。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）Intrinsic Dimension 实验结果（T=64）
| D | (D,D) NMSE | (D-1,D) NMSE | (D,D-1) NMSE |
|----|------------|--------------|---------------|
| 2  | $5.89\times10^{-5}$ | $5.89\times10^{-2}$ | $3.67\times10^{-2}$ |
| 3  | $6.37\times10^{-5}$ | $2.64\times10^{-2}$ | $2.07\times10^{-2}$ |
| 4  | $6.92\times10^{-5}$ | $1.39\times10^{-2}$ | $1.30\times10^{-2}$ |
| 5  | $4.69\times10^{-4}$ | $4.46\times10^{-3}$ | $7.19\times10^{-3}$ |
| 6  | $1.99\times10^{-3}$ | $6.26\times10^{-3}$ | $7.67\times10^{-3}$ |

> ✅ **结论**：当且仅当 $ h_1=h_2=D $ 时，模型能达到 $10^{-5} \sim 10^{-3}$ 级别的低误差，其余配置差2~3个数量级。

#### （2）Triangle-Center 实验结果（随T变化）
| T | (2,2), E=24 | (4,4), E=24 | (4,4), E=48 |
|----|-------------|-------------|-------------|
| 8  | 0.086       | 0.072       | 0.080       |
| 16 | 0.843       | 0.853       | 0.856       |
| 32 | 0.957       | 0.958       | 0.959       |
| 64 | 0.985       | 0.985       | 0.985       |

> ✅ **结论**：所有配置的NMSE均随T迅速上升至接近1，表明模型无法有效逼近该任务，且增加head或dimension无显著改善。

#### （3）Max-position Retrieval 实验
- 所有配置下，**argmax位置信息**的恢复误差在训练后大幅下降（最多降低92%，ratio=0.08）。
- 相反，**第二大的注意力位置信息**恢复误差几乎不变（trained/init ≈ 1.0），证明softmax只集中于最大值。

#### （4）Specific Position Aggregation 实验
- 模型成功将CLS token的注意力权重集中在位置1、2、3，各占1/3，其余位置为0。
- 该模式稳定存在于不同样本之间，与token内容无关，证实了位置编码支持的内容无关选择机制。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **深度分离存在**：对于某些任务（如全局最小化），单层Transformer需要参数随序列长度$T$指数增长才能达到精度$\epsilon$，而两层Transformer仅需$O(\epsilon^{-1})$参数即可。
2. **Softmax Attention 的本质限制**：只能高效检索注意力得分最高的token；检索第$k$大（$k\geq2$）需至少$\Omega(\epsilon^{-(k-1)/n})$参数，成本呈指数增长。
3. **信息解码代价高昂**：聚合来自$|I|$个token的信息，其解码所需的FFN参数为$O(\epsilon^{-|I|d/E})$，随信息集大小指数增长。
4. **InfoFlow 成功预测现象**：
   - 内在维度$D$决定了最小head数：必须$h_1 \geq D$且$h_2 \geq D$。
   - 高阶比较任务（如triangle-center）随$T$增长变得不可逼近，即使增加head或dimension也无效。

### 方法的局限性
1. **适用范围有限**：目前主要适用于“retrieval-type”任务，其中active index set大小不随$T$增长。
2. **未考虑attention矩阵秩的影响**：参数代价律未纳入query/key矩阵低秩特性的影响（见Amsel et al., 2025）。
3. **传播模式基于合成任务**：三大机制由人工构造任务验证，在真实NLP/CV任务中的普适性有待进一步检验。
4. **忽略动态交互细节**：作为抽象框架，牺牲了对中间层动态过程的精细刻画。

### 未来工作方向
1. **扩展到更广泛的任务类别**：如生成任务、逻辑推理任务等。
2. **整合更多传播机制**：例如feed-forward network的非线性变换路径、跨头协作等。
3. **结合实际训练动力学**：将InfoFlow与梯度流、优化轨迹相结合，构建端到端可微分的代理模型。
4. **指导架构搜索**：利用InfoFlow自动推荐最优的$L, h_l, E_l$组合以匹配目标任务复杂度。
5. **应用于其他架构变体**：如RetNet、Mamba、State Space Models等新型序列模型。

> 💡 **总体评价**：InfoFlow提供了一个强有力的**原则性工具**，让我们能够像分析电路一样“阅读”Transformer的计算流程，标志着向可解释、可预测的深度学习理论迈出了重要一步。

</details>

---

### 14. [S2Aligner: Pair-Efficient and Transferable Pre-Training for Sparse Text-Attributed Graphs](https://arxiv.org/abs/2605.18579)

**Authors**: Yuhan Wang, Haopeng Zhang, Yibo Ding, Jiaqi Yu, Xinyu Zhao, Yuhang Liu, Ziwei Zhang, Xiao Wang, Ruijie Wang  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.18579v1  

#### Abstract
Pre-training on text-attributed graphs (TAGs) is central to building transferable graph foundation models, where LLM-as-Aligner methods align graph and text representations through the semantic knowledge of large language models. However, these methods usually assume that node texts provide sufficie...

---

### 15. [NGM: A Plug-and-Play Training-Free Memory Module for LLMs](https://arxiv.org/abs/2605.16893)

**Authors**: Yuwen Qu, Wenhui Dong, Chenyang Si, Caifeng Shan  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.16893v1  

#### Abstract
Recent studies introduce conditional memory modules that decouple knowledge storage from neural computation, enabling more direct knowledge access. Compared to MoE, which relies on dynamic computation paths, explicit lookup provides a more efficient knowledge retrieval mechanism. However, these appr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NGM: A Plug-and-Play Training-Free Memory Module for LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的大型语言模型（LLMs）在处理语言时，将**动态组合计算**与**静态局部模式复用**混合在一起。对于诸如命名实体、重复标识符、公式化短语等局部、固定的语言模式，标准 Transformer 缺乏原生的“查找”机制（lookup primitive），导致模型必须通过注意力和前馈网络在推理时重新构建这些信息，效率低下。

现有基于显式记忆模块的方法（如 Engram、MeKi、L3 等）虽然引入了 N-gram 查找机制，但通常依赖：
- 额外训练的记忆嵌入表（learned memory embeddings）
- 专用的检索管道或存储架构
- 额外的可训练参数

这限制了其灵活性和即插即用能力。

### 🚀 提出的新方法：NGM（N-gram Memory）
作者提出 **NGM** —— 一种**无需训练、即插即用**（plug-and-play, training-free）的记忆增强模块，直接注入到已训练好的冻结 LLM 中。

#### 核心组件：
1. **Causal N-Gram Encoder（因果 N-gram 编码器）**
   - 不学习新的 N-gram 嵌入，而是**直接对主干模型预训练的 token embeddings 进行平均池化**，构造多尺度的 N-gram 表示。
   - 使用因果滑动窗口（trailing window），保证自回归生成的一致性。
   - 无需额外内存表或训练。

2. **Cosine-Gated Memory Injector（余弦门控记忆注入器）**
   - 使用非参数化的 **cosine 相似度 + ReLU 门控** 来衡量当前 decoder hidden state 与 N-gram 表示之间的匹配程度。
   - 只有正相关的记忆信号才会通过缩放残差连接注入到隐藏状态中。
   - 完全无参数（non-parametric），无需训练。

### 🔍 相比现有方法的优势
| 特性 | NGM | Engram / MeKi / L3 类方法 |
|------|-----|--------------------------|
| 是否需要训练 | ❌ 否（training-free） | ✅ 是（需训练 memory embeddings） |
| 是否新增可训练参数 | ❌ 否 | ✅ 是 |
| 是否依赖外部检索系统 | ❌ 否 | ✅ 是（如哈希表、CPU offload） |
| 是否即插即用 | ✅ 是（可随时启用/禁用） | ❌ 否（需集成进训练流程） |
| 实现复杂度 | 低（仅 avg pool + cosine） | 高（需设计路由、索引、存储） |

> ✅ **核心创新**：首次证明——**预训练 embedding 空间本身即可作为轻量级本地记忆源**，无需额外训练或基础设施。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集（共 8 个基准）
涵盖数学、代码、知识、对齐四大类任务：

| 类别 | 数据集 | 说明 |
|------|--------|------|
| **Math** | GSM8K, MATH500 | 数学应用题求解 |
| **Code** | HumanEval, LiveCodeBench v5 | 代码生成能力 |
| **Knowledge** | MMLU-Redux, GPQA-Diamond | 多学科知识问答（GPQA 为研究生级别难题） |
| **Alignment** | IFEval (strict-prompt), TruthfulQA (MC2) | 指令遵循与事实性判断 |

此外还测试了**多模态场景**：
- **Qwen3-VL-2B-Instruct** 上的 MM-Bench, MMStar, OCRBench, TruthfulQA, MMLU-R

### ⚙️ 实验设置
- **模型系列**：Qwen3 系列（0.6B, 1.7B, 4B, 8B, 14B），全部使用公开开源版本。
- **骨干模型状态**：所有 backbone 参数保持 **frozen（冻结）**，不进行微调。
- **NGM 配置**：
  - N-gram 尺寸：默认 `N = {2, 3}`
  - 注入层：选择早期和中间若干 decoder 层（如第 2 和第 15 层），具体见附录 Table 7
  - 输出缩放因子 λ：固定值（如 0.1）
  - 门控方式：使用 ReLU 抑制负相关更新
- **评估框架**：统一使用 EvalScope 进行评测。
- **解码参数**：temperature=0.7, top-p=0.8, top-k=20（除特定任务外一致）

### 🔁 基线方法对比
- **Base Model**：原始 Qwen3 模型（无任何修改）
- **+NGM**：相同模型 + NGM 模块注入（零参数增加）
- 所有比较均在同一 checkpoint 下完成，确保公平性。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Table 1）

| Model | Avg Score (Base) | Avg Score (+NGM) | ΔAvg |
|-------|------------------|-------------------|------|
| Qwen3-0.6B | 35.65 | 36.86 | **+1.21** |
| Qwen3-1.7B | 54.68 | 55.16 | **+0.48** |
| Qwen3-4B | 63.53 | 64.11 | **+0.58** |
| Qwen3-8B | 71.35 | 72.17 | **+0.81** |
| Qwen3-14B | 73.77 | 74.49 | **+0.72** |

> ✅ 在所有五个规模上，NGM 均带来稳定提升，平均增益 **+0.5 到 +1.2 分**。

### 🔝 特定任务显著提升
| 任务 | 模型 | 提升幅度 |
|------|------|---------|
| **LiveCodeBench** | Qwen3-14B | **+3.0** |
| **GPQA-Diamond** | Qwen3-14B | **+3.03** |
| **MMStar**（多模态） | Qwen3-VL-2B | **+1.53** |

> 💡 表明 NGM 在**代码生成**和**知识密集型任务**中效果尤为突出。

### 🔍 消融实验结果（Ablation Studies）

#### （1）N-gram 尺寸影响（Table 3）
- 单一尺寸（如仅 n=2 或 n=3）表现不稳定。
- 多尺度组合 `{2,3}` 效果最佳（平均得分最高）。
- 加入 `n=4` 并未进一步提升整体性能。

#### （2）ReLU 门控作用（Table 4）
- **移除 ReLU** 导致平均分从 72.17 降至 70.38（↓1.79）。
- 最大下降出现在 LiveCodeBench（-5.4 分），说明抑制负对齐信号至关重要。

#### （3）融合方式对比：Stack vs Concat（Table 5）
- **Stack（默认）**：每个 n-gram 尺寸独立计算门控 → 更灵活，平均得分更高（72.17 vs 71.07）
- **Concat**：拼接后统一门控 → 性能较差，表明多尺度应分别建模。

#### （4）压缩 tokenizer 影响（Table 6）
- 引入 Engram 风格的 Compressed Tokenizer 对某些任务（如 HumanEval）有益。
- 但**整体平均性能不如默认配置**，故未采用。

> ✅ 结论：**多尺度构造 + ReLU 门控 + Stack 融合**是关键成功因素。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **预训练 embedding 空间天然具备局部记忆结构**
   - 实验证明：聚合后的 N-gram embedding 与 decoder hidden states 存在显著几何对齐（cosine similarity 显著高于随机控制组）。
   - 支持了“embedding space 可作 memory source”的假设。

2. **NGM 是有效的训练免费增强手段**
   - 无需训练、无需额外参数、无需外部知识库，即可实现稳定性能提升。
   - 特别适用于**短程模式稳定性强的任务**（如代码、术语复用、算术链）。

3. **记忆交互具有强局部性**
   - cross-position 分析显示：cosine 相似度集中在对角线附近（diagonal-dominant），说明有用记忆信号主要来自邻近位置。
   - 支持了因果滑动窗口的设计合理性。

4. **在多模态模型中也有效**
   - 将 NGM 仅应用于 Qwen3-VL-2B 的语言解码器部分，即可在多个多模态基准上取得提升（如 MMStar +1.53）。
   - 表明该机制具有良好的泛化性和模块化潜力。

### ⚠️ 方法的局限性
1. **Bag-of-Embeddings 近似忽略词序**
   - 当前 N-gram 表示采用平均池化，无法捕捉顺序敏感的短语（如 “not good” vs “good not”）。
   - 可能注入误导性信号。

2. **固定注入规则缺乏上下文适应性**
   - 使用固定的 λ 和 ReLU 门控，不能根据不同任务或生成阶段动态调整。
   - 导致在 IFEval 等指令敏感任务上有时性能下降。

3. **仅强化短程规律，不解决长程推理或外部知识检索**
   - NGM 不替代 long-context 或 retrieval-augmented 方法。
   - 应视为**互补机制**，而非通用解决方案。

### 🔮 未来工作方向
- 探索轻量级可学习门控机制（如少量 adapter 参数），实现任务自适应注入。
- 结合 positional encoding 或 attention pattern 改进 order-insensitive 问题。
- 扩展至 encoder-decoder 架构或多模态联合记忆建模。
- 与 long-context 或 retrieval systems 联合使用，形成层次化记忆体系。

---

> 🔗 **代码地址**：[https://github.com/PioneerQyw/NGM](https://github.com/PioneerQyw/NGM)  
> 📄 **原文链接**：[https://arxiv.org/abs/2605.16893](https://arxiv.org/abs/2605.16893)

</details>

---

### 16. [A Practical Noise2Noise Denoising Pipeline for High-Throughput Raman Spectroscopy](https://arxiv.org/abs/2605.18511)

**Authors**: David Martin-Calle (ILM,UCBL,CNRS), Cesar Alvarez Llamas (ILM,UCBL,CNRS), Vincent Motto- Ros (ILM,UCBL,CNRS), Christophe Dujardin (ILM,UCBL,CNRS,IUF), J\'er\'emie Margueritat (ILM,UCBL,CNRS), David Rodney (ILM,UCBL,CNRS)  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.18511v1  

#### Abstract
A lightweight and reproducible denoising pipeline for high-throughput Raman spectroscopy is presented. The approach relies on a one-dimensional convolutional autoencoder trained using a Noise2Noise strategy, requiring neither external spectral libraries nor high signal-to-noise reference spectra for...

---

### 17. [CompactAttention: Accelerating Chunked Prefill with Block-Union KV Selection](https://arxiv.org/abs/2605.16839)

**Authors**: Jiwon Song, Dongwon Jo, Beomseok Kang, Jae-Joon Kim  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.16839v1  

#### Abstract
Chunked prefill has become a widely adopted serving strategy for long-context large language models, but efficient attention computation in this regime remains challenging. Existing sparse attention methods are primarily designed for one-shot prefill and do not translate efficiently to chunked prefi...

---

### 18. [HexAGenT: Efficient Agentic LLM Serving via Workflow- and Heterogeneity-Aware Scheduling](https://arxiv.org/abs/2605.16637)

**Authors**: You Peng, Youhe Jiang, Wenshuang Li, Xu Xu, Ke Zhou, Jiawei Jiang, Chen Wang, Binhang Yuan  
**Category**: cs.DC  
**Published**: 2026-05-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.16637v1  

#### Abstract
Agentic LLM applications increasingly execute user requests as multi-step workflows involving planning, tool use, branching, refinement, and synthesis. In such settings, users experience the end-to-end latency of an entire workflow, not the latency of any single LLM call. In this paper, we study how...

---

### 19. [World Model-Enabled Causal Digital Twins for Semantic Communications in Physical AI Systems](https://arxiv.org/abs/2605.16547)

**Authors**: Lingyi Wang, Tingyu Shui, Walid Saad, Pascal Adjakple  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.16547v1  

#### Abstract
Semantic communication has emerged as a promising paradigm for enabling goal-oriented networking. However, most existing semantic communication solutions are tailored to one-shot tasks and optimize instantaneous performance. Hence, they cannot be used to support closed-loop dynamic systems with phys...

---

### 20. [HydroAgent: Closing the Gap Between Frontier LLMs and Human Experts in Hydrologic Model Calibration via Simulator-Grounded RL](https://arxiv.org/abs/2605.17792)

**Authors**: Zhi Li, Songkun Yan, Jie Cao, Mofan Zhang, Anjiang Wei, Jinwoong Yoo, Yang Hong  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.17792v1  

#### Abstract
Calibrating distributed hydrologic models is a critical bottleneck across operational water resources management - streamflow prediction, reservoir operation, drought monitoring, infrastructure design, and flood forecasting all depend on it. Each basin demands an expert to translate hydrograph signa...

---

### 21. [Dual-Rate Diffusion: Accelerating diffusion models with an interleaved heavy-light network](https://arxiv.org/abs/2605.18190)

**Authors**: Grigory Bartosh, David Ruhe, Emiel Hoogeboom, Jonathan Heek, Thomas Mensink, Tim Salimans  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.18190v1  

#### Abstract
Diffusion models achieve state-of-the-art generative performance but suffer from high computational costs during inference due to the repeated evaluation of a heavy neural network. In this work, we propose Dual-Rate Diffusion, a method to accelerate sampling by interleaving the execution of a heavy ...

---

### 22. [PAIR: Prefix-Aware Internal Reward Model for Multi-Turn Agent Optimization](https://arxiv.org/abs/2605.17877)

**Authors**: Wonjoong Kim, Yeonjun In, Sangwu Park, Dongha Lee, Chanyoung Park  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.17877v1  

#### Abstract
A significant hurdle for current LLMs is the execution of complex, multi-stage tasks. Group Relative Policy Optimization (GRPO) has been emerging as a leading choice, but its reliance on sparse outcome rewards severely limits credit assignment across intermediate steps. Existing remedies such as run...

---

### 23. [TeleCom-Bench: How Far Are Large Language Models from Industrial Telecommunication Applications?](https://arxiv.org/abs/2605.18025)

**Authors**: Jieting Xiao, Yun Lin, Huizhen Qiu, Rui Ma, Chen Zhong, Dongyang Xu, Xiao Long, Chaoyu Zhang, Qiaobo Hao, Ding Zou, Zhiguo Yang, Yanqin Gao, Fang Tan  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.18025v1  

#### Abstract
While Large Language Models have achieved remarkable integration in various vertical scenarios, their deployment in the telecommunications domain remains exploratory due to the lack of a standardized evaluation framework. Current telecom benchmarks primarily focus on static, foundational knowledge a...

---

### 24. [Retrieval-Based Multi-Label Legal Annotation: Extensible, Data-Efficient and Hallucination-Free](https://arxiv.org/abs/2605.16767)

**Authors**: Li Zhang, Jaromir Savelka, Kevin Ashley  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.16767v1  

#### Abstract
Multi-label legal annotation requires assigning multiple labels from large, evolving taxonomies to long, fact-intensive documents, often under limited supervision. Parametric encoders typically require task-specific training and retraining when the label set changes, while prompting generative large...

---

### 25. [E-PMQ: Expert-Guided Post-Merge Quantization with Merged-Weight Anchoring](https://arxiv.org/abs/2605.16882)

**Authors**: Wenjun Wang, Yanggan Gu, Shuo Cai, Yuanyi Wang, Pengkai Wang, Jianmin Wu, Hongxia Yang  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.16882v1  

#### Abstract
Low-resource deployment constraints have made model quantization essential for deploying neural networks while preserving performance. Meanwhile, model merging has become an increasingly practical low-resource strategy for integrating multiple task- or domain-specialized experts into a single model ...

---

### 26. [Learning Transferable Topology Priors for Multi-Agent LLM Collaboration Across Domains](https://arxiv.org/abs/2605.17359)

**Authors**: Taolin Zhang, Zijie Zhou, Jiuheng Wan, Tingyuan Hu, Chengyu Wang, Xiaofeng He, Richang Hong  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.17359v1  

#### Abstract
Large language model (LLM)-based multi-agent systems have shown strong potential for complex reasoning by coordinating specialized agents through structured communication. However, existing topology-evolution methods typically construct or optimize a collaboration topology for each query from scratc...

---

### 27. [EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL](https://arxiv.org/abs/2605.18703)

**Authors**: Minrui Xu, Zilin Wang, Mengyi DENG, Zhiwei Li, Zhicheng Yang, Xiao Zhu, Yinhong Liu, Boyu Zhu, Baiyu Huang, Chao Chen, Heyuan Deng, Fei Mi, Lifeng Shang, Xingshan Zeng, Zhijiang Guo  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.18703v1  

#### Abstract
Equipping LLMs with tool-use capabilities via Agentic Reinforcement Learning (Agentic RL) is bottlenecked by two challenges: the lack of scalable, robust execution environments and the scarcity of realistic training data that captures implicit human reasoning. Existing approaches depend on costly re...

---

### 28. [Lever: Speculative LLM Inference on Smartphones](https://arxiv.org/abs/2605.16786)

**Authors**: Tuowei Wang, Fengzu Li, Yanfan Sun, Wei Gao, Ju Ren  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.16786v1  

#### Abstract
Large language models (LLMs) are increasingly needed for interactive mobile applications, but high-quality models exceed the limited DRAM available on smartphones. Flash storage can hold larger models, yet flash-backed inference is slow because autoregressive decoding repeatedly invokes the target m...

---

### 29. [Differentiable Optimization Layers for Guaranteed Fairness in Deep Learning](https://arxiv.org/abs/2605.17118)

**Authors**: David Troxell, Noah Roemer, Guido Mont\'ufar  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.17118v1  

#### Abstract
Differentiable optimization layers are traditionally integrated in predict-then-optimize frameworks where a neural model estimates parameters that subsequently serve as fixed inputs to downstream decision-making optimization problems. In this work, we introduce the concept of a "fairness layer": a d...

---

### 30. [TriOpt: A Scalable Algorithm for Linear Causal Discovery](https://arxiv.org/abs/2605.17465)

**Authors**: Rafat Ashraf Joy, Elena Zheleva  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.17465v1  

#### Abstract
Learning causal relations from observational data is challenging because the graph search space grows super-exponentially with the number of variables. Ordering-based methods reduce this space by first identifying the topological ordering, whereas continuous optimization methods explore most likely ...

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

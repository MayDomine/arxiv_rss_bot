# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-29 08:14:53 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding](https://arxiv.org/abs/2607.25852)

**Authors**: Hong Liu, Rui Cen, Junhan Shi, Guangshuo Qin, Jiebin Zhang, Tianyu Liu, Runzhi Fan, Guoliang Zhao, Ruobing Xie, Kai Zhang, Song Liu, Guanghua Yu, Jianchen Zhu  
**Category**: cs.CL  
**Published**: 2026-07-29  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2607.25852v1  

#### Abstract
Speculative decoding accelerates large language model inference without changing the target distribution, but no single drafting structure performs best across real-world workloads. Autoregressive multi-token prediction (MTP) is a lightweight, stable proposal mechanism, whereas block-parallel diffus...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding 核心总结

---

## 1. 主要贡献和创新点

### 解决的问题
当前的 **Speculative Decoding** 方法通常依赖单一的 **drafter** 结构，在真实世界多样化的负载（如对话、代码生成、数学推理）下表现不一致。不同任务的输出分布差异显著：
- **高熵对话**：语义多样性高，长序列预测接受率迅速衰减。
- **低熵代码/数学**：结构化强，存在更长的可预测片段。

现有方法未能针对这种 **workload heterogeneity** 进行优化，导致在实际部署中性能受限。

### 提出的新方法与创新思路
论文提出 **AngelSpec**，一个统一的高性能推理框架，从训练、架构到推理三个层面进行系统性优化：

#### （1）**双路径专业化 Drafter 设计**
- **MTP Drafter**：用于高熵对话场景。
  - 基于 **autoregressive multi-token prediction**，轻量且稳定。
  - 通过 **Training-Time Test (TTT)** 和 **long-context training** 缓解训练-推理不匹配问题。
- **DFly Drafter**：用于代码/数学等结构化任务。
  - 基于 **block-parallel diffusion** 架构，支持并行生成长候选块。
  - 引入 **hybrid target-conditioning backbone** 和 **predecessor-conditioned autoregressive head**，提升目标特征利用和块内依赖建模。

#### （2）**DFly 架构创新**
- **Hybrid Target-Conditioning Backbone**：
  - 融合 DFlash 的共享上下文投影与 DFlare 的层特定融合权重，使每层 draft layer 可以差异化地利用多层 target hidden states。
- **Predecessor-Conditioned Autoregressive Head**：
  - 在并行 backbone 后引入轻量级自回归头（采用 **hidden-correction** 形式），动态修正每个位置的预测，基于前序已选 token，增强块内一致性。

#### （3）**D-cut：运行时自适应验证深度控制**
- 将 **target verification** 视为批处理级别的共享资源。
- 动态分配验证预算：结合 **prefix confidence** 和 **profiled runtime cost model**，决定保留多少 draft token 进行验证。
- 实现 **吞吐量最大化**，尤其在高并发下避免验证成为瓶颈。

#### （4）**开源训练框架 AngelSpec**
- 支持 MTP 与 block-parallel speculative decoding 的统一训练。
- 包含 **disaggregated hidden-state generation**、**TTT rollout**、**long-context training**、**online acceptance evaluation** 等模块。
- 提供插件化接口，便于扩展新模型、损失函数和优化器。

### 相比现有方法的优势
| 方面 | AngelSpec 优势 |
|------|----------------|
| **通用性** | 不追求“万能 drafter”，而是根据任务特性选择最优方案（MTP vs DFly）。 |
| **效率** | DFly 显著提升平均接受长度（MAL），D-cut 动态剪枝低价值后缀，减少无效计算。 |
| **灵活性** | 支持多种 drafter 结构、损失函数（如 LK Loss, e2e TV Loss）、数据策略。 |
| **实用性** | 开源完整训练与评估流程，推动社区发展。 |

---

## 2. 核心实验方法和设置

### 数据集
- **基础训练数据**：Open-PerfectBlend（覆盖问答、推理、指令遵循等）。
- **领域增强数据**：
  - **代码**：OpenCodeInstruct + OpenCodeReasoning（共 500K prompts）
  - **数学**：Big-Math（200K prompts）
- **评估基准**：
  - 数学：GSM8K, Math500
  - 代码：HumanEval, MBPP, LiveCodeBench
  - 对话：MT-Bench, AlpacaEval

### 实验设置
- **目标模型**：
  - Qwen3-8B
  - Hy3-A21B / Hy3-295B-A21B（MoE 模型）
- **硬件平台**：8× H20 GPUs，Tensor Parallelism=8
- **并发测试范围**：4 ~ 64
- **温度设置**：T=0（greedy）和 T=0.9（stochastic）

### 评估指标
| 指标 | 定义 |
|------|------|
| **Mean Accepted Length (MAL)** | 平均每次 speculative step 成功提交的 token 数（含 bonus token） |
| **Throughput (Tok/s)** | 输出 token 吞吐量 |
| **Speedup** | 相对于 autoregressive decoding 的加速比 |
| **Avg Acceptance Rate** | 累积接受率的平均值 |

### 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **AR (Autoregressive)** | 基线 | 标准逐 token 解码 |
| **MTP** | 自回归多 token 预测 | 单一 MTP 模块递归使用 |
| **DFlash** | 并行块预测 | 基于 diffusion 的 block-parallel drafter |
| **DSpark** | 半自回归 | 引入 Markov transition bias 的 DFlash 变体 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Hy3-A21B）

#### ✅ 接受长度提升（Table 3）
| Drafter | Math500 | GSM8K | HumanEval | MBPP | LiveCodeBench | MT-Bench | **Avg MAL** |
|--------|---------|--------|-----------|------|---------------|----------|------------|
| MTP    | 3.30    | 3.30   | 3.13      | 3.04 | 2.84          | 2.40     | **3.00**   |
| DFlash | 4.01    | 4.23   | 4.36      | 4.05 | 3.10          | 2.38     | **3.69**   |
| **DFly** | **5.23** | **5.53** | **5.52** | **5.41** | **4.07** | **2.96** | **4.79** |

> 💡 **DFly 相比 MTP 提升 59.7%，相比 DFlash 提升 29.8%**

#### ✅ 端到端吞吐量（Table 7）
在多个并发级别下，**DFly-8** 均实现最高吞吐：

| Conc. | AR (Tok/s) | DFlash-8 (Tok/s) | **DFly-8 (Tok/s)** | **Speedup (vs AR)** |
|-------|------------|------------------|--------------------|---------------------|
| c4    | 288.4      | 516.9            | **571.2**          | **1.98×**           |
| c8    | 421.8      | 767.4            | **851.2**          | **2.02×**           |
| c16   | 638.2      | 1270.3           | **1418.8**         | **2.22×**           |
| c32   | 933.9      | ~2150            | **2243.1**         | **~2.40×**          |
| c64   | 1297.7     | ~1890            | **2726.2**         | **2.11×**           |

> ✅ **DFly 实现 1.98–2.40× 加速，比 DFlash 高出 10.5–11.8% 吞吐**

#### ✅ D-cut 动态剪枝效果（Figure 4）
- 在真实流量回放中，D-cut 在高并发下仍持续提升吞吐：
  - c64 下比 DFly 提升 **+15.7%** 吞吐。
- 接受长度仅下降约 **1.5%**（2.50 → 2.46），说明剪枝丢弃的是本就会被拒绝的低置信后缀。

---

### 消融实验结果（Table 4 & Table 5）

#### 🔍 架构消融（Hy3-A21B, T=0）
| 组件 | Math500 | GSM8K | HumanEval | MBPP | LiveCodeBench | MT-Bench | Avg MAL |
|------|--------|--------|-----------|------|---------------|----------|---------|
| DFlash | 4.27 | 3.86 | 4.27 | 4.18 | 3.46 | 2.60 | 3.77 |
| → **DFly (hybrid backbone)** | **4.92** | **4.88** | **5.00** | **5.03** | **3.71** | **2.88** | **4.40** |
| → + Markov Head | 5.12 | 5.13 | 5.24 | 4.98 | 3.85 | 3.03 | 4.56 |
| → + **Hidden-Correction Head** | **5.13** | **5.17** | **5.26** | **5.00** | **4.00** | **3.06** | **4.60** |
| → + **Code/Math Data** | **5.33** | **5.18** | **5.53** | **5.45** | **4.02** | 3.02 | **4.75** |

> ✅ Hybrid backbone 提升 **+0.63 MAL**
> ✅ Hidden-correction head 再提升 **+0.14 MAL**
> ✅ 领域数据增强带来最大收益，提升 **+0.15 MAL**

#### 🔍 推理配置消融（Table 5）
| Draft Layers | Block Size | Avg MAL |
|-------------|------------|---------|
| 3 | 5 | 3.29 |
| 5 | 5 | 3.39 |
| 5 | 8 | **4.25** |
| 7 | 8 | 4.25 |

> ✅ 增大 block size 比加深 drafter 更有效；5 层 + block size 8 是最佳性价比组合。

---

## 4. 关键结论和发现

### 主要发现
1. **Workload heterogeneity 是真实部署中的首要挑战**：
   - 不存在“通吃”所有任务的 drafter。
   - 应对策略：**MTP 用于高熵对话，DFly 用于低熵代码/数学**。

2. **DFly 架构显著提升 block-parallel drafting 性能**：
   - Hybrid backbone 提高 target feature 利用率。
   - Hidden-correction head 有效缓解块内独立预测带来的接受率衰减。

3. **训练策略至关重要**：
   - **TTT + Rollout + Long-context training** 显著提升 MTP 的深层接受率。
   - **领域增强数据** 对 DFly 在代码/数学上的表现起决定性作用。

4. **运行时动态控制是释放潜力的关键**：
   - D-cut 通过将验证视为共享资源，实现了 **更高吞吐与更低延迟的平衡**，尤其在高并发下优势明显。

### 方法局限性
- **MTP 与 DFly 需分别训练与维护**，增加工程复杂度。
- **D-cut 当前实现有开销**（如 piecewise CUDA graph），尚未完全发挥潜力。
- **未支持 thinking mode 的联合优化**：实验表明 no-think 与 high-think 模式需独立训练 drafter。

### 未来工作方向
- 实现 **全图捕捉（full CUDA graph capture）** 以降低 D-cut 开销。
- 将 **budget selection 移出关键路径**，实现异步调度。
- 探索 **统一多任务 drafter** 或 **混合专家（MoE）drafter** 架构。
- 扩展至 **reasoning step-level speculative decoding**（如 SpecCoT）。

---

> 📦 **项目开源地址**：  
> GitHub: [https://github.com/Tencent/AngelSpec](https://github.com/Tencent/AngelSpec)  
> 文档: [https://angelspec.readthedocs.io](https://angelspec.readthedocs.io)

</details>

---

### 2. [GLIDE: Guided Layerwise Hybrid Attention for Efficient LLM Inference](https://arxiv.org/abs/2607.24788)

**Authors**: Vimal William, Ravi Tandon, Jyotikrishna Dass  
**Category**: cs.AI  
**Published**: 2026-07-29  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2607.24788v1  

#### Abstract
As Large Language Models scale to increasingly long contexts, the memory I/O and computational overhead of the Key-Value (KV) cache during decoding emerges as the primary throughput bottleneck. To address this, we propose GLIDE, a Guided Layerwise Hybrid Attention that strategically integrates slidi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GLIDE: Guided Layerwise Hybrid Attention for Efficient LLM Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
随着 Large Language Models (LLMs) 处理的上下文长度不断增长，**autoregressive decoding 阶段的 Key-Value (KV) cache 内存 I/O 和计算开销**成为推理吞吐量的主要瓶颈。传统方法如 sliding window attention 通过丢弃旧 token 来减少内存，但会丢失长期依赖；而纯 linear attention 虽然节省内存，却在早期层严重损害模型表达能力。

### **提出的新方法与新思路**
论文提出了 **GLIDE (Guided Layerwise Hybrid Attention)**，一种基于**层间异质性**（layer-wise heterogeneity）的自适应混合注意力机制，其核心思想是：

- **非均匀地分配 softmax 与 linear attention**：在不同 Transformer 层采用不同的混合策略。
- **深度感知的敏感性建模**：早期层对 linearization 极为敏感，需保留 full softmax attention；深层则表现出冗余性，可完全线性化。
- **滑动窗口 + 线性递归聚合**：结合 sliding window softmax attention 与 linear recurrent aggregation，在局部保持高保真交互，全局通过常数大小状态压缩历史信息。

具体实现上，GLIDE 将模型划分为三个块：
- **Early layers**: 全部使用 softmax (`δ = 0`)
- **Middle layers**: 混合使用 (`δ = α·w`, `α ∈ [0,1]`)
- **Late layers**: 完全线性化 (`δ = w`)

该设计将配置空间从 $ L $ 维简化为单参数 $ \alpha $，便于优化。

### **相比现有方法的优势**
| 对比维度 | 现有方法（如 Liger, LoLCats） | GLIDE |
|--------|-------------------------------|-------|
| **混合策略** | 统一层级混合（uniform hybridization） | **分层引导混合**（guided layerwise） |
| **效率-精度权衡** | 固定比例，难以平衡 | 可调节的 Pareto 最优前沿 |
| **KV Cache I/O** | 减少有限 | **降低 45×–62×** |
| **兼容性** | 依赖特定架构 | 支持多种 retention-based 架构（Liger, LoLCats, BASED 等） |
| **无需重训练** | 需 fine-tuning | 支持 zero-shot 替换，也可结合 LoRA 微调恢复性能 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **基准测试套件**（来自 `lm-evaluation-harness`）：
  - **PiQA**（物理常识）
  - **ARC-Easy / ARC-Challenge**（科学推理）
  - **HellaSwag**（常识推理）
  - **WinoGrande**（共指消解）
  - **MMLU**（多任务语言理解，5-shot prompting）

此外还使用了 **Alpaca instruction-following 数据集**（cleaned 100K 样本）进行 LoRA 微调。

### **实验设置**
- **模型架构**：
  - **Llama-3-8B**
  - **Mistral-7B**
- **上下文长度**：支持 up to 128K tokens
- **滑动窗口大小**：默认 `w = 1024` 或 `w = 20K`（视场景而定）
- **硬件平台**：NVIDIA Grace Hopper GH200 Superchip（120GB GPU memory）
- **实现工具**：基于 PyTorch FlexAttention + Flash Attention backend
- **微调方式**：LoRA（rank=8, scaling=8），训练 2 个 epoch

### **评估指标**
| 指标 | 描述 |
|------|------|
| **KV Cache I/O (MB/token)** | 每生成一个 token 所需的 KV 缓存传输量，反映内存带宽压力 |
| **End-to-End Latency (s)** | 不同序列长度下的总推理延迟 |
| **Zero-shot / Fine-tuned Accuracy (%)** | 在多个下游任务上的平均准确率 |
| **Speedup** | 相对于 baseline 的推理加速比 |

### **基线方法对比**
- **Vanilla Softmax**：标准 full attention，KV cache 线性增长
- **Sliding Window Attention (SWA)**：固定窗口，FIFO 丢弃旧 token
- **Pure Linear Attention**：无 KV cache，但表达力差
- **Uniform Hybrid Models**（如 Liger）：各层统一混合比例
- **GLIDE (non-uniform)**：本文提出的分层混合策略

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **KV Cache I/O 显著下降**
- GLIDE 实现 **45×–62× 的 KV cache I/O 降低**：
  - Baseline (vanilla softmax): ~4000 MB/token
  - GLIDE `(0, 15w/16, w)`：仅 **43 MB/token**（≈93× 更高效）
  - GLIDE `(0,0,w)`：**88 MB/token**（≈45× 更高效）

> 图 7 显示，在 128K 上下文时，baseline 累计 I/O 达 **17.5 GB**，而 GLIDE `(0,15w/16,w)` 仅为 **5.2 GB**（3.4× 减少）

#### ✅ **端到端延迟大幅缩短**
| 配置 | Llama-3-8B @ 32K | 加速比 |
|------|------------------|--------|
| Baseline (vanilla softmax) | OOM（无法运行） | — |
| Hybrid baseline `(0,0,0)` | 3705.72 s | 1× |
| GLIDE `(0,0,w)` | 2638.35 s | **1.4×** |
| GLIDE `(w,w,w)` | 1640.75 s | **2.25×** |

> 注意：尽管 `(0,w/2,w)` 和 `(0,15w/16,w)` 的线性化程度不同，但由于瓶颈已转为 memory-bound，它们的延迟几乎相同（~2637s），说明 **I/O 是主导因素**。

#### ✅ **维持高准确率**
| 配置 | 平均准确率（Llama-3-8B） | 相对 baseline (%) |
|------|--------------------------|--------------------|
| Baseline (vanilla softmax) | 72.16% | 100% |
| Hybrid baseline `(0,0,0)` | 69.12% | 96% |
| GLIDE `(0,0,w)`（fine-tuned） | 67.84% | 94% |
| GLIDE `(0,15w/16,w)`（fine-tuned） | 66.74% | 92% |
| Pure linear `(w,w,w)` | 33.96% | 47% ❌ |

> 结合 LoRA 微调后，GLIDE 能恢复 **6–8 个百分点**的性能损失，达到 **92%–96% 原始性能**。

#### ✅ **扩展服务能力**
- 由于每请求的 KV cache 占用减少 2–3×，**单台加速器可服务并发用户数提升 2–3 倍**。
- 在长上下文多用户部署中具有显著优势。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Transformer 层对 attention linearization 表现出明显的 depth-dependent 敏感性**：
   - 早层（1–11）高度依赖 softmax，线性化导致准确率暴跌至 ~36%
   - 深层（22–32）可完全线性化而不明显影响性能
2. **非均匀混合优于统一混合**：
   - GLIDE 在 Pareto 前沿上全面超越 uniform hybrid 和 SWA
   - “在哪里用 softmax” 比“用了多少”更重要
3. **KV cache I/O 是长上下文推理的核心瓶颈**：
   - 一旦进入 memory-bound 区域，进一步减少计算收益有限
   - GLIDE 通过控制 softmax footprint 实现 sub-linear I/O 增长

### **方法的局限性**
1. **信息稀释问题（Information Dilution）**：
   - 长序列下，linear recurrent state 逐渐丢失细粒度 token 区分能力
2. **小窗口下的系统开销**：
   - 当 `w ≤ 128` 时，kernel launch 和调度开销抵消了理论 FLOPs 优势
3. **缺乏针对 hybrid attention 的专用 kernel 优化**
4. **未探索分布式训练/推理场景下的表现**

### **未来工作方向**
1. 设计更强大的 **state compression 机制** 以缓解信息稀释
2. 开发 **fused kernels** 支持 sliding window softmax + linear recurrence 的联合执行
3. 探索 **动态调整 δ 参数** 的在线策略（adaptive allocation）
4. 将 GLIDE 扩展至 **tensor/pipeline parallelism 分布式环境**
5. 研究在 **reasoning-intensive workloads** 中的 layer-wise sensitivity 是否有所不同

---

> 🔚 **总结一句话**：  
> **GLIDE 通过“早层保精度、深层层提效率”的分层混合注意力策略，在几乎不牺牲生成质量的前提下，实现了高达 62× 的 KV cache I/O 压缩和 3.3× 的解码加速，为长上下文 LLM 推理提供了新的 Pareto 最优解。**

</details>

---

### 3. [Beyond Shapley: An Influence-Based Data Auditing Pipeline for LLM Alignment and Evaluation](https://arxiv.org/abs/2607.22766)

**Authors**: Yunting Song, Matthew Watson, Peter Grabowski, Jun Qin  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2607.22766v1  

#### Abstract
The alignment of Large Language Models (LLMs) is increasingly bottlenecked by data quality. As datasets scale, massive preference and instruction-tuning corpora inevitably accumulate hidden structural contradictions, safety risks, and systemic human annotation errors. Standard dataset auditing metho...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Beyond Shapley: An Influence-Based Data Auditing Pipeline for LLM Alignment and Evaluation —— 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大语言模型（LLM）的对齐（alignment）瓶颈已从模型架构转向**数据质量**。大规模偏好数据、指令微调语料和人工标注数据不可避免地积累以下问题：
- 隐藏的结构性矛盾（structural contradictions）
- 安全风险（safety risks）
- 系统性的人工标注错误（systematic human annotation errors）

传统数据审计方法如**语义去重**（semantic deduplication）或**LLM-as-a-judge**存在严重缺陷：
- 无法捕捉单条记录的实际预测影响力
- 忽略深层的功能性规则冲突（functional rule clashes）
- 人类重新标注成本高昂且不可扩展

### 提出的新方法与新思路
本文提出一种**可扩展的、仅依赖推理（inference-only）的数据估值流水线**，用于近似计算 **Shapley value**，而无需迭代模型重训练。

核心思想是通过构建**语义 k-NN 邻域图**，利用目标 LLM 的概率分布，测量**零样本（zero-shot）与一样本（one-shot）条件对数似然偏移**（conditional log-likelihood shift），从而量化每条数据的“影响”（influence）。

#### 创新点：
- **Influence-Based Valuation Framework**  
  提出一个高效的、基于推理的数据价值评估框架，避免了 O(2^M) 复杂度的 Shapley 计算。
  - 时间复杂度为 O(K)，其中 K 是邻域大小
  - 支持生成式任务（如文本生成、pairwise preference）而非仅限于分类/回归
- **Structural Contradiction Detection**  
  能够发现主流数据集中被标准相似性指标忽略的隐藏结构性矛盾
- **Benchmark Integrity Validation**  
  首次将审计扩展到**评估集（evaluation split）**，揭示当前基准中存在的严重标签缺陷，导致高性能模型被不公平惩罚

### 相比现有方法的优势
| 方法 | 缺陷 | 本方法优势 |
|------|------|-----------|
| Semantic Deduplication | 只能识别文本重复，无法衡量功能效用 | 捕捉功能性影响，识别逻辑冲突 |
| LLM-as-a-Judge | 易受格式、长度等偏见影响；需大量昂贵推理 | 数学上更严谨；大幅减少需人工审核的数量 |
| Cluster Shapley / Model Arithmetic | 聚合到组级别，牺牲细粒度；仍需微调 | 保留**逐条记录级粒度**，完全免训练 |
| Exact Shapley | 计算不可行（O(2^M)） | 用 in-context probing 近似边际贡献 |

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **HelpSteer2** [20]  
   - 包含 21,362 条记录，用于奖励模型训练
   - 每条记录有 0–4 分的帮助性评分（helpfulness score）
2. **Anthropic's HH-RLHF** [2]  
   - 广泛使用的多轮对话偏好数据集
   - 包含训练集和评估集，用于 RLHF 对齐
   - 每个样本包含两个响应选项及人类选择的“更好”响应

### 实验设置
- **嵌入模型**：`Qwen3-Embedding-4B` 用于生成 prompt 的向量表示
- **检索方法**：HNSWlib 实现高效 k-NN 检索（k=15）
- **打分模型**：`Qwen3.5-9B` 作为目标 LLM，计算 zero-shot 和 one-shot 的 `log-likelihood`
- **专家仲裁器**：`Gemini 3.1 Pro` 用于验证候选矛盾对是否构成真实标注错误
- **超参数**：
  - 邻域大小 $ K = 15 $
  - 优势阈值 $ T_{adv} = -2 $
  - 最小退化邻居数 ≥5（MNN constraint）

### 评估指标
- **Disagreement Rate**：模型预测与人类标签不一致的比例
- **Manual Audit Search Space Reduction**：相比全量人工审核，所需检查样本数量的压缩率
- **Confirmed Contradictions**：经 LLM 验证的真实标注错误数量
- **False Positive Rate**：基线方法过度标记的比例

### 基线方法对比
- **Naive LLM-as-a-Judge**：直接让 LLM 判断每条记录的质量
- **Similarity-only + LLM Judge**：基于语义相似性进行配对后交由 LLM 审核
- **Direct LLM Evaluator without Filtering**：无数学预筛选，直接进行全面比较

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 在 HelpSteer2 上的结果
- 手动审计搜索空间减少 **99.1%**
  - 原始需审记录：8,434 条（得分为 4 的高质量记录）
  - 经管道过滤后仅剩 **77 对候选矛盾**
  - 最终确认 **18 对矛盾**，涉及 **10 条错误标注记录**
- 发现多种失败模式：
  - **事实性错误**（Factuality）：虚构日期、错误引用
  - **指令遵循失败**（Instruction）：忽略 API 格式要求
  - **学术造假**（Academic）：编造论文标题与作者
  - **功能性缺失**（Functional）：未按要求输出问答对

> 示例：一条响应因结构良好但内容虚假（伪造研究）被评满分，而另一条类似行为却被正确判低分 → 揭示“表面性偏见”（superficiality bias）

#### ✅ 在 HH-RLHF 上的结果
- **训练集审计**：
  - 初始标记潜在矛盾：11,663 条（约 6.8%）
  - 经 LLM 验证后确认 **4,238 条真实矛盾**，涉及 **2,841 条唯一记录**
  - 主要类别包括：
    - 安全违规（Safety & Harmlessness Violations）
    - 幻觉与事实错误（Hallucinations & Factual Errors）
    - 不必要回避（Unhelpful Evasion）
    - 结构性失败（如截断、循环）
    - 毒性回应（Toxic Responses）

- **评估集审计（Benchmark Integrity）**：
  - 发现 **108 条高置信度标签错误记录**
  - 在这些记录上，模型“失败率”显著上升：
    | 模型 | 全体评估集 Disagreement Rate | 本方法提取子集 Disagreement Rate |
    |------|-------------------------------|----------------------------------|
    | Qwen3.5-9B (FT) | 28.9% | **62.04%** |
    | Qwen3.5-27B     | 28.67%| **75.00%** |
    | Gemma3-12B      | 38.34%| **63.89%** |

  > 表明：许多所谓“模型失败”实则是由于**人类标签本身错误**所致

### 与基线方法对比结果
| 方法 | 审核成本 | 标记数量 | 真实矛盾检出率 | 问题 |
|------|----------|---------|----------------|------|
| Naive LLM Judge | 极高（8,434 次推理） | 3,193 条（37.8%） | 低 | 严重过标记 |
| Similarity + LLM Judge | 极高（>126k 次推理） | 未报告 | 中 | 成本不可接受 |
| **本文方法** | 极低（仅 77 对） | 18 对 | **极高** | ✅ 成本降低 99.1% |

### 消融实验结果（Ablation Studies）

#### 📌 超参数敏感性分析
- **邻域大小 K**：
  - K=5 时无法建立可靠上下文，**0 条确认矛盾**
  - K∈[10,25] 性能稳定，最终选 K=15 平衡效率与效果
- **优势阈值 $ T_{adv} $**：
  - $ T_{adv} = -1.0 $：标记过多（4,770 对），稀释质量
  - $ T_{adv} = -2.5 $：过于严格，仅得 29 条
  - $ T_{adv} = -2.0 $：最优平衡点（108 条，62.04% disagreement）
- **最小退化邻居数**：
  - 对 HH-RLHF 影响较小（501–516 对之间）
  - 但在噪声更大数据集中至关重要，防止误报

#### 📌 模型替换测试（Robustness）
| 配置 | Qwen-9B FT Disagreement |
|------|------------------------|
| 默认（Qwen-Embed + Qwen-9B） | 62.04% |
| 替换为 BGE-M3 嵌入 | 66.32% |
| 替换为 Gemma3-12B 打分模型 | 50.41% |

> 结论：方法具有较强鲁棒性；使用与目标模型同架构的 scorer 更有利于发现特定模型的 alignment 冲突

---

## 4. 关键结论和发现

### 主要发现
1. **人类标注存在系统性偏差**  
   即使在经过严格审查的数据集中（如 HelpSteer2、HH-RLHF），也普遍存在“**表面性偏见**”——即人类倾向于奖励结构良好、格式正确的响应，即使其内容虚假或有害。

2. **当前评估基准存在严重漏洞**  
   在 HH-RLHF 评估集中发现了上百条标签错误，导致：
   - 高性能模型因做出正确判断反而被扣分
   - “模型失败率”被严重高估
   - 当前 benchmark 的完整性受到挑战

3. **数学驱动的影响评分可极大提升审计效率**  
   - 将人工审核范围缩小 **99%以上**
   - 以极低成本定位高价值纠错目标
   - 为构建可信的 LLM 对齐数据提供了实用工具

4. **In-context probing 可有效近似 Shapley value**  
   证明了无需模型重训练即可实现细粒度数据价值评估，适用于生成式任务

### 方法的局限性
1. **依赖 LLM 验证器**  
   最终仲裁使用 Gemini 等 LLM evaluator，可能存在主观性和非确定性，不同运行可能产生轻微差异。

2. **△LL 是异常过滤器，非独立检测器**  
   数学得分仅用于缩小搜索空间，仍需外部仲裁确认语义矛盾。

3. **强调精度而非召回率**  
   采用严格阈值（如 $ T_{adv} = -2 $）优先保证高精度，因此会遗漏部分低置信度错误。

4. **未覆盖全部模型失败原因**  
   如 HH-RLHF 中 ~30% 的基础 disagreement 并非都源于本文发现的 108 条错误，说明还有其他因素影响模型表现。

### 未来工作方向
1. **自动化构建高杠杆训练集**  
   利用连续的 advantage metrics 自动生成精简但高效的训练子集，降低 alignment 计算成本。

2. **动态损失加权（Dynamic Loss Weighting）**  
   在微调过程中将 advantage scores 作为 loss weights，软性惩罚梯度冲突记录，同时保持多样性。

3. **合成对抗性负样本**  
   利用该框架生成高度隐蔽、上下文欺骗性的负例，用于强化 RL 或 DPO（Direct Preference Optimization）训练。

4. **扩展至多模态与跨语言场景**  
   探索在图像-文本、语音等多模态对齐任务中的应用潜力。

--- 

> **总结一句话**：  
> 本文提出了一种**免训练、高效率、数学严谨**的数据审计框架，首次系统性揭示了主流 LLM 对齐数据中存在的**结构性矛盾与标签缺陷**，并暴露了当前评估基准的脆弱性，为构建更安全、更可靠的 AI 系统提供了关键诊断工具。

</details>

---

### 4. [SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE Foundation Models](https://arxiv.org/abs/2607.24787)

**Authors**: Jinwei Kong, Runqi Meng, Fanyi Wang, Wentao Qiu, Haotian Hu, Yongjian Zhou, Zhenhua Ge  
**Category**: cs.AI  
**Published**: 2026-07-29  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.24787v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) models expand foundation model capacity through conditional expert activation, but their full expert pools remain difficult to deploy under limited accelerator memory. Although expert offloading alleviates memory pressure by moving inactive experts to host memory or s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE Foundation Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **专家加载延迟瓶颈**：在稀疏 MoE（Sparse Mixture-of-Experts）模型中，尽管每个 token 只激活少量专家（top-K routing），但整个专家池仍需在推理时可访问。当采用 **expert offloading**（将非活跃专家卸载至主机内存或存储）以缓解设备端内存压力时，**专家需求只能在原生路由决策后才被确定**，导致“计算 → 路由 → 加载 → 执行”这一串行流程，使专家加载延迟暴露在关键路径上，严重影响推理吞吐。

### **提出的新方法与思路**
- **SpecPrefetch**：一种参数高效的专家预取框架，其核心思想是：
  - **解耦传输预测与执行路由**（Decoupling transfer prediction from execution routing）：
    - 使用一个**共享轻量级 adapter** 预测下一层可能被调用的专家候选集（candidate set），仅用于**异步预加载**；
    - 最终执行的专家仍由原始冻结的 native router 决定，**不改变预训练模型的路由语义**。
  - **仅用于传输调度的预测机制**：预测错误不会影响模型输出，只影响传输效率（false positive 浪费带宽，false negative 触发按需加载）。

### **相比现有方法的优势**
| 方法类型 | 代表 | 缺陷 | SpecPrefetch 的优势 |
|--------|------|------|------------------|
| **Quantization/Compression** | Yan et al., 2025 | 不改变专家需求可见时间点，无法重叠计算与加载 | 显著提前暴露专家需求，实现更早预取 |
| **Training-free Prefetching** | FATE (Fang et al., 2025b) | 依赖跨层路由稳定性，在多模态/复杂任务中表现不稳定 | 更鲁棒，适应性强 |
| **Training-based Prediction** | Draft Model, ProMoE | 将预测与路由耦合，可能扰动原始专家激活模式 | **保持原生路由不变**，仅优化传输时机 |

> ✅ **核心创新**：提出“**transfer-only prefetching**”设计范式——预测只为传输服务，不影响最终执行逻辑。

---

## **2. 核心实验方法和设置**

### **使用的模型与数据集**
#### **模型**
- **Qwen3-VL-30B-A3B**：纯路由型 MoE 架构（pure routed expert pool）
- **DeepSeek-VL2-Tiny**：共享+路由混合架构（shared + routed experts）

#### **评估基准（Benchmarks）**
分为两类工作负载：
- **VLM 工作负载**：
  - `OCRBench`：文本密集图像理解
  - `ChartQA-Test`：图表问答
  - `HallusionBench`：幻觉敏感的多模态推理
- **LLM 工作负载**：
  - `GSM8K`：数学推理
  - `HumanEval`：代码生成

> 数据多样性确保跨架构、跨任务、跨模态的有效性验证。

### **实验设置**
- **模拟实验**：在 Qwen 和 DeepSeek 模型上进行 offloading 推理仿真，量化预取对端到端延迟的影响。
- **真实设备评测**：部署 **DeepSeek-VL2-Tiny** 到 **Snapdragon 8 Elite** 移动平台（Adreno 825 GPU），通过注入不同 I/O 延迟模拟多种存储速度（NVMe → SD Card）。
- **offloading 设置**：
  - 专家以 4-bit 量化形式存储于主机侧
  - 每个专家约 1.94MB，总专家存储达 1.36GB
  - 共享专家常驻设备端，提供部分计算窗口用于隐藏加载延迟

### **评估指标**
| 指标 | 定义 | 说明 |
|------|------|------|
| **Expert Coverage Recall (R@M)** | $ \frac{|T_{l+1} \cap C_{l+1}|}{|T_{l+1}|} $ | 下一层真实需求 $T_{l+1}$ 中有多少被成功预测并列入候选集 $C_{l+1}$（M 为候选数量） |
| **Ready Recall** | 在 routed expert computation 开始前已驻留的专家比例 | 更贴近运行时实际收益 |
| **Throughput (tokens/sec)** | 解码吞吐量 | 衡量系统级性能提升 |
| **Speedup** | 相对于 load-on-demand 基线的速度提升倍数 | 综合反映预取效果 |
| **Trainable Parameters** | 可训练参数量 | 衡量方法的参数效率 |

### **基线方法对比**
| 基线 | 类型 | 特点 |
|-----|------|------|
| **FATE** | Training-free | 利用相邻层 gate 信息进行无训练预取 |
| **Draft Model** | Learned predictor | 辅助小模型预测下一层专家 |
| **ProMoE** | Strong learned predictor | 更大的预测模块，更强学习能力 |

> 所有方法均统一在 “**transfer-only**” 协议下比较：预测结果仅用于预取，最终执行仍由 native router 决定。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表1：下一层专家覆盖召回率（Expert Coverage Recall）**
| 模型 | Benchmark | FATE | Draft Model | ProMoE | **SpecPrefetch (ours)** |
|------|-----------|------|-------------|--------|------------------------|
| Qwen3-VL-30B-A3B | GSM8K (Avg.) | 88.27 | — | 89.94 | **91.86** ✅ |
| | HumanEval (Avg.) | 87.90 | — | 86.97 | **91.67** ✅ |
| | OCRBench (Avg.) | 91.23 | 91.52 | 93.35 | **94.48** ✅ |
| | ChartQA-Test (Avg.) | 90.56 | 92.31 | 94.39 | **94.63** ✅ |
| | HallusionBench (Avg.) | 91.90 | 92.31 | 93.75 | **94.55** ✅ |
| **DeepSeek-VL2-Tiny** | GSM8K (Avg.) | 91.66 | 86.39 | 85.73 | **91.18** ✅ |
| | HumanEval (Avg.) | 88.84 | 78.54 | 75.07 | **89.13** ✅ |
| | OCRBench (Avg.) | 91.70 | 90.14 | 92.90 | **94.82** ✅ |
| | ChartQA-Test (Avg.) | 90.85 | 91.03 | 93.97 | **95.23** ✅ |
| | HallusionBench (Avg.) | 93.15 | 93.63 | 92.57 | **94.39** ✅ |

> 🔍 **结论**：SpecPrefetch 在 **10 个 model-benchmark 组合中拿下 9 项最佳平均召回率**，尤其在 VLM 任务上优势显著。

---

#### **消融实验结果**

##### **(1) 预测器结构消融（vs MLP Predictor）**
| 模型 | Benchmark | MLP Predictor (Avg.) | **SpecPrefetch** | 提升幅度 |
|------|-----------|-----------------------|------------------|----------|
| Qwen3-VL-30B-A3B | GSM8K | 86.92 | **91.86** | +4.94% |
| | HumanEval | 80.05 | **91.67** | +11.62% |
| DeepSeek-VL2-Tiny | GSM8K | 72.47 | **91.18** | +18.71% |
| | HumanEval | 60.80 | **89.13** | +28.33% |

> ✅ 表明 **adapter-based gate-aware calibration** 比通用 MLP 映射更有效。

##### **(2) 参数效率对比（Table 4）**
| 模型 | Single MLP | Draft Model | ProMoE | **SpecPrefetch** |
|------|------------|-------------|--------|------------------|
| DeepSeek-VL2-Tiny | 1.56M | 26.40M | 32.69M | **1.63M** ✅ |
| Qwen3-VL-MoE-30B | 24.38M | 19.19M | 414.45M | **12.95M** ✅ |

> 📉 SpecPrefetch 仅需 **ProMoE 的 3.1% 参数量**，却取得更高召回率，体现极强参数效率。

---

#### **真实设备性能提升（Figures 3 & 4）**
在 **Snapdragon 8 Elite + Adreno 825 GPU** 上测试：

| 存储类型 | Compute-Optimized Runtime | **+ SpecPrefetch** | 吞吐提升 |
|---------|----------------------------|--------------------|----------|
| Mid UFS | ~3.6 tokens/s | ~4.1 tokens/s | **+15%** |
| Slow UFS | ~3.4 tokens/s | ~4.1 tokens/s | **+20%** ✅ |
| SD Card | ~3.2 tokens/s | ~3.7 tokens/s | **+17%** |

> 💡 当 I/O 成为瓶颈时，SpecPrefetch 效果最明显。

##### **冷缓存场景（Cold Cache, Fig. 5）**
- 清除 OS page cache 后首次运行：
  - Baseline: 3.29 tps
  - Compute-Optimized: 3.25 tps（几乎无改善）
  - **SpecPrefetch + Compute**: **3.76 tps** → **相对提速 1.14×，预取贡献 +16%**

> ❗说明：**仅靠计算优化无法消除存储瓶颈，必须依赖预取机制提前准备专家**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **SpecPrefetch 显著提升了下一层专家的预测准确率**，在多数 benchmark 上优于 training-free 和 training-based 基线。
2. ✅ **参数极其高效**：使用共享 adapter 实现高性能预测，参数量远低于主流 learned predictor。
3. ✅ **真实设备上带来可观吞吐增益**：在移动边缘设备上，最高可达 **20% 的 decoding throughput 提升**。
4. ✅ **设计安全可靠**：因不解耦执行路由，**完全保留原始 MoE 模型行为**，预测误差不影响输出正确性。
5. ✅ **特别适用于 I/O 受限场景**：在慢速存储或冷启动情况下，预取带来的收益最大。

---

### **局限性（Limitations）**
1. **收益依赖运行时条件**：
   - 若 cache 已热或存储极快（如 NVMe），大部分加载已被计算掩盖，预取增益有限。
2. **预测错误仍影响系统效率**：
   - False positive 浪费带宽与 cache 空间；
   - False negative 导致仍需阻塞加载。
3. **单一共享 adapter 可能不足**：
   - 对于专家专业化程度极高、各层路由模式差异大的 MoE 模型，可能需要 layer-specific 适配器。
4. **调度策略较局部**：
   - 当前调度器未考虑全局 cache 管理、跨请求复用或多用户批处理优化。
5. **验证范围有限**：
   - 实测仅在 DeepSeek-VL2-Tiny 和移动端完成，缺乏更大规模 MoE 或服务器端 offloading 场景验证。

---

### **未来工作方向**
- 设计 **动态预算调整机制**：根据当前 cache 状态、带宽估计自适应调整预取数量 $M$。
- 引入 **multi-hop prefetching**：预测多层后的专家需求，进一步拉长预取窗口。
- 结合 **prefetching + speculative execution**：构建更复杂的调度流水线。
- 开发 **layer-aware adapters**：针对高度异构的 MoE 层设计差异化预测模块。
- 扩展至 **server-side heterogeneous memory systems**：应用于 GPU-NUMA、CPU-GPU disaggregation 等场景。

---

> **总结一句话**：  
> **SpecPrefetch 通过“轻量 adapter + 解耦预取”的设计，在不改变 MoE 模型行为的前提下，实现了高精度、低开销的专家预取，在资源受限设备上显著提升了稀疏 MoE 模型的推理效率，为 MoE 落地边缘计算提供了实用解决方案。**

</details>

---

### 5. [MXAttention: Data-Free Optimal Scaling and Pre-Normalization Quantization for MXFP4 Attention](https://arxiv.org/abs/2607.24377)

**Authors**: Jianlin Yu, Jing Lin, Linghui Kong, Aiyue Chen, Weiyi Sun, Chenyu Zeng, Wangli Lan, Jinxi Li, Zhuo Zheng, Ziyang Yue, Danning Ke, Fei Yi, Tianchi Hu, Yuan Ding, Yiwu Yao, Junsong Wang  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.24377v1  

#### Abstract
The quadratic cost of attention is a major bottleneck in diffusion-based video generation models. MXFP4 attention provides a promising path toward efficient inference, but direct MXFP4 quantization often degrades generation quality due to two numerical issues: the clipping-underflow trade-off from p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MXAttention: Data-Free Optimal Scaling and Pre-Normalization Quantization for MXFP4 Attention

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在基于扩散模型的视频生成中，**注意力机制的二次计算复杂度**是推理成本的主要瓶颈。虽然 MXFP4 这类 4-bit 浮点格式能显著降低 GEMM 计算开销，但直接将标准 MXFP4 量化应用于注意力模块会导致严重的生成质量下降。

作者指出，这种退化主要源于两个**数值失效模式**：
1. **Power-of-two 共享缩放导致的剪裁-下溢权衡（clipping-underflow trade-off）**：由于 MXFP4 使用以 2 为底的共享指数（E8M0 scale），在块内对数值进行缩放时，过小的 scale 会引发大值饱和（clipping），而过大的 scale 则加剧小值的舍入误差和下溢（underflow）。
2. **Softmax 循环中的归一化不一致（normalization mismatch）**：在 FlashAttention 的在线 Softmax 循环中，若仅对输出累加器更新路径中的 `Pij` 进行量化，而保留行和累加路径使用原始浮点值，则会导致最终注意力权重不再满足行归一化（即行和不等于 1）。

### 提出的新方法与创新思路
为解决上述问题，论文提出 **MXAttention** —— 一种完全无需校准（data-free）、无需量化感知训练（QAT）的后训练量化（PTQ）框架，其核心包含两大创新组件：

#### （1）通用最优缩放（Universal Optimal Scaling, UOS）
- **核心思想**：利用 power-of-two 缩放带来的周期性结构，推导出一个**全局最优且分布无关的缩放边界**。
- **具体实现**：通过最小化一个全局的 MXFP4 量化误差目标函数，得出闭式解 **`Qmax = 7.25`**。
- **优势**：
  - 完全无需 per-layer 搜索或数据校准；
  - 对任意激活分布均有效，具有理论最优性；
  - 在 OCP（[4,8)）与 TFS（(3,6]）之间取得更优平衡，将溢出区域压缩至 **(7, 7.25]**，大幅减少 clipping 风险。

#### （2）预归一化量化（Pre-Normalization Quantization, PNQ）
- **核心思想**：在执行行和累加与输出累加之前，**统一先对未归一化的 Softmax 指数项 `Pij` 进行量化**，并复用同一份量化后的 `Pij` 更新两个状态。
- **具体实现**：修改 FlashAttention 的在线循环，在 `l_i` 和 `O_i` 更新中都使用 `Qx(Pij)`。
- **优势**：
  - 保证了即使经过量化，诱导出的注意力权重仍严格满足行归一化（sum to one）；
  - 消除了因路径不一致引入的额外缩放偏差；
  - 不增加额外计算或内存访问，可无缝融合进低精度注意力流水线。

### 相比现有方法的优势
| 维度 | MXAttention | 其他方法（如 OCP MXFP4、NVFP4） |
|------|-------------|-------------------------------|
| 是否需要数据校准 | ❌ 否（data-free） | ✅ 是（通常需 calibration） |
| 是否需要 QAT | ❌ 否 | ✅ 是（部分方法依赖） |
| 是否破坏行归一化 | ❌ 否（由 PNQ 保证） | ✅ 是（常见问题） |
| 缩放策略灵活性 | 固定边界 `Qmax=7.25`，理论最优 | 依赖经验或搜索 |
| 实现复杂度 | 极低，仅修改 scale selection 和量化时机 | 可能涉及多级缩放、平滑等 |

---

## 2. 核心实验方法和设置

### 使用的数据集与模型
- **模型**：
  - **Wan2.2-14B**：开源大规模文本到视频生成模型，采用自注意力与交叉注意力分离架构。
  - **HunyuanVideo-13B**：腾讯混元团队的大规模视频生成模型，采用双流/单流 Transformer 结构。
- **数据集**：
  - 使用 **Open-Sora prompt suite** 中固定的提示词子集进行生成测试。
  - 所有方法在同一 prompt 和随机种子下生成视频，确保可比性。

### 实验设置
- **分辨率与时长**：
  - Wan2.2：720p，81帧，40个去噪步。
  - HunyuanVideo：720p，129帧，50个去噪步。
- **量化策略**：
  - 主比较中 Wan2.2 使用混合精度（Block 0 和最后两步保持高精度），其余为 4-bit；
  - HunyuanVideo 使用**全 4-bit 注意力**（所有 block 和 step 均量化）。
- **硬件支持**：MXFP4 被 NVIDIA Blackwell、AMD MI350、Ascend 950 等主流加速器原生支持。

### 评估指标
- **生成质量（VBench）**：
  - **Subject Consistency**（主体一致性）
  - **Imaging Quality**（图像质量）
  - **Aesthetic Quality**（美学质量）
- **帧级相似度（vs. FP16 基线）**：
  - **Cosine Similarity**
  - **SSIM**（结构相似性）
  - **PSNR**（峰值信噪比）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **FP16** | 全精度注意力，作为黄金标准 |
| **MXFP4 (OCP)** | 标准 OCP 规则（floor-based），仅量化输出路径 |
| **NVFP4** | NVIDIA 提出的 4-bit 格式，16-element block + tensor-level scale |
| **NVFP4 + SageAttention** | NVFP4 + 通道平滑 + Softmax 路径两级缩放 |
| **MXAttention** | 本文方法（UOS + PNQ + Hadamard Rotation） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| Model | Method | Imaging Quality | Cosine Sim. | SSIM | PSNR |
|-------|--------|------------------|--------------|------|------|
| **Wan2.2** | FP16 | **0.7085** | — | — | — |
| | MXFP4 (OCP) | 0.6414 | 0.9290 | 0.5076 | 15.58 |
| | **MXAttention** | **0.7054** (+95.4% gap closed) | **0.9536** | **0.6319** | **17.92** |
| **HunyuanVideo** | FP16 | 0.6185 | — | — | — |
| | MXFP4 (OCP) | 0.4459 | 0.9489 | 0.5954 | 16.25 |
| | **MXAttention** | **0.6380** (>100% gap closed) | **0.9745** | **0.7061** | **19.23** |

> ✅ **关键结论**：
> - MXAttention 在两个模型上均**关闭了至少 95% 的 VBench Imaging Quality 差距**；
> - 在 HunyuanVideo 上甚至**超过 FP16 基线**；
> - 帧级相似度（Cosine、SSIM、PSNR）也大幅提升，表明生成内容高度忠实于原始 FP16 输出。

### 消融实验结果（Table 3）

| 配置 | Imaging Quality (Wan2.2) | Imaging Quality (HunyuanVideo) | Cosine Sim. (↑) |
|------|---------------------------|----------------------------------|------------------|
| Full (UOS + PNQ + Rotation) | **0.6842** | **0.6380** | **0.9329 / 0.9745** |
| w/o UOS | 0.6822 | 0.6423 | 0.9143 / 0.9676 |
| w/o UOS & PNQ | 0.6352 | 0.5321 | 0.9063 / 0.9643 |
| OCP MXFP4 | 0.5452 | 0.4459 | 0.9127 / 0.9489 |

> 🔍 **分析**：
> - **PNQ 对 Imaging Quality 提升最大**（+~5–10 pts），说明归一化一致性至关重要；
> - **UOS 显著提升帧相似度指标**（SSIM/PSNR），验证其缩放策略更保真；
> - **Hadamard Rotation 抑制 outlier 效果明显**，单独使用即可显著优于 OCP。

### 机制验证实验
- **UOS 验证**：在 40 层注意力中，经验最优 `Qmax` 有 **90–95% 的层集中在 7.25**，证明理论边界高度匹配实际需求。
- **PNQ 验证**：直接 MXFP4 的平均行和仅为 **0.9336**，存在系统性低估；而 PNQ 从机制上消除该偏差。

---

## 4. 关键结论和发现

### 主要发现
1. **MXFP4 注意力退化主因明确**：并非单纯精度损失，而是由 **power-of-two 缩放的剪裁-下溢矛盾** 和 **Softmax 路径量化不一致** 引起。
2. **UOS 实现理论最优缩放**：`Qmax = 7.25` 是分布无关的全局最优解，无需任何数据或搜索即可达到接近最优的量化效果。
3. **PNQ 保障数值稳定性**：通过统一量化路径，强制保持行归一化，从根本上避免了注意力权重失真。
4. **MXAttention 实现近无损量化**：
   - 所有 VBench 指标与 FP16 差距 **< 0.01**；
   - 性能媲美甚至超越强 NVFP4 + SageAttention 基线；
   - **零校准、零 QAT、零 per-layer 参数调优**。

### 方法的局限性
- **依赖 FlashAttention 架构**：PNQ 设计紧密耦合于 tiled online-softmax 循环，难以直接迁移到 materialize-full-P 的旧式实现。
- **固定 `Qmax` 可能在极端分布下非最优**：尽管理论证明其鲁棒性强，但在某些特殊任务或模型中可能仍有微调空间。
- **未探索训练阶段集成**：当前为纯 PTQ 方案，若结合 QAT 或微调可能进一步压榨性能边界。

### 未来工作方向
- 将 UOS 思想推广至其他 microscaling 格式（如 MXFP6、NVFP8）；
- 探索 PNQ 在其他归一化操作（LayerNorm、RMSNorm）中的应用；
- 开发支持 MXAttention 的编译器级优化，实现端到端自动融合；
- 扩展至语音、多模态等长序列生成场景。

---

> 📦 **代码已开源**：  
> MXAttention 已集成至华为 **MindIE-SD** 主干分支，公开可用：  
> [https://gitcode.com/Ascend/MindIE-SD/tree/master/mindiesd](https://gitcode.com/Ascend/MindIE-SD/tree/master/mindiesd)

</details>

---

### 6. [Learning to Optimize: Joint Routing and Flow Allocation on Sparse Non-Euclidean Networks](https://arxiv.org/abs/2607.23467)

**Authors**: Haomiao Sun, Fang He, Congyuan Ji, Xindi Tang  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.23467v1  

#### Abstract
We study an integrated pickup-and-delivery problem on sparse, non-Euclidean networks that jointly optimizes cyclic routing, cargo flow allocation, and cross-cycle service. The tight coupling of these operational constraints creates a complex discrete-continuous decision space with highly restricted ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Learning to Optimize: Joint Routing and Flow Allocation on Sparse Non-Euclidean Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本文研究的是**稀疏非欧几里得网络上的集成取送货问题（Non-Euclidean Pickup-and-Delivery Problem with Cross-Cycle Service, NE-PDP-CCS）**，该问题在现实工业场景中广泛存在，如集装箱班轮运输。其核心挑战在于：
- 网络是**稀疏、有向、非欧几里得**的，节点之间并非全连接；
- 节点具有多重角色（既是起点又是终点或中转枢纽）；
- 存在**跨周期服务（cross-cycle service）**：货物可能在上一个航次被装载，在当前周期内交付；
- 需要同时优化**循环路径规划（cyclic routing）** 和 **货流分配（cargo flow allocation）**。

传统方法难以处理这种高度耦合、离散-连续混合决策空间的问题。

---

### **提出了什么新方法或新思路**
作者提出了一种名为 **Double-Channel Graph Attention (DCGA)** 的端到端深度强化学习框架，其核心创新包括：

#### ✅ **双通道图注意力架构（Double-Channel Architecture）**
- **网络通道（Network Channel）**：建模物理可达性、旅行成本与时间等拓扑属性；
- **需求通道（Demand Channel）**：建模OD对之间的供需关系、数量、收益、软截止期限等服务逻辑；
- 通过 **LPL投影（Logical-Physical-Logical Projection）** 在多任务节点间共享信息，避免表征坍塌。

> 这种分离设计有效防止了物理移动特征与服务逻辑混淆，提升了模型对复杂约束的理解能力。

#### ✅ **状态条件解码器 + 约束感知掩码机制**
- 引入**复合操作掩码（composite operational masking）**，在每一步自动排除非法动作（如容量超限、不可达边、重复访问限制等）；
- 使用**OD索引的动作空间**和**交付侧偏置（delivery-side bias）**，自然支持可选履约与跨周期交付，无需硬性前置约束。

#### ✅ **仿真器耦合奖励机制**
- 解码路径后由确定性**flow simulator**评估可行货流分配与最终目标值；
- 政策学习基于下游利润而非单纯路径长度，实现路由与流量的联合优化。

---

### **相比现有方法的优势**
| 维度 | DCGA优势 |
|------|----------|
| **适用性** | 专为稀疏、非欧、循环服务设计，而大多数NCO方法假设全连接欧氏平面 |
| **可行性保障** | 掩码机制确保每步输出合法，避免后修复（post-hoc repair）带来的低效与不稳定 |
| **效率** | 推理速度达到**秒级**，适合动态重规划场景 |
| **扩展性** | 随问题规模增大，性能优势显著增强 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LinerLib**（Brouer et al., 2014）中的地中海子集，这是一个公开的真实班轮航运网络基准数据集。
- 包含真实港口拓扑、航线距离与时序信息。
- 在此基础上生成不同规模的 NE-PDP 实例（从 NE-PDP61 到 NE-PDP141，表示最多141个请求）。

---

### **实验设置和评估指标**

#### ✅ **评估指标**
- **目标函数值（Objective Value）**：最大化总收益减去运输成本、未满足需求惩罚和延迟惩罚；
- **相对差距（gap%）**：相对于已知最优或最佳解的目标差距；
- **运行时间（Runtime）**：求解整个测试批次所需时间；
- **推理延迟（Inference Latency）**：单实例平均推理时间。

#### ✅ **训练与测试配置**
- 使用 **Intel Xeon Platinum CPU + NVIDIA RTX PRO 5000 GPU**；
- 模型采用 **Actor-Critic 架构**，训练使用带熵正则化的蒙特卡洛策略梯度；
- 测试时采用贪婪搜索（greedy decoding）；
- 每个规模下测试数百个实例以保证统计显著性。

---

### **基线方法对比**
分为三类共 **13 种基线方法**：

| 类别 | 方法 | 简要说明 |
|------|------|---------|
| **优化方法** | MIP（McCormick松弛）、Flow Model (FM) | 使用 Gurobi 求解松弛后的MILP作为上界参考 |
| **启发式/元启发式** | LKH3, GA, SA, ALNS | 经典强基线，其中LKH3为TSP类最强启发式之一 |
| **学习型方法** | NCS/N2S（原版 + 图编码增强版 NCS2） | 当前最先进的神经PDP求解器，代表主流坐标系方法 |

> 特别地，NCS2 加入了与DCGA相同的图编码器，用于剥离“是否只是图编码更强”的影响。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 实例规模 | DCGA 目标值（×10⁵） | 最佳基线目标值 | DCGA 推理时间（秒） |
|--------|---------------------|---------------|--------------------|
| NE-PDP61 | -14.03 | -12.30 (GA2) | ~14 s |
| NE-PDP81 | -18.77 | -22.29 (GA2) | ~16 s |
| NE-PDP91 | -22.07 | -32.24 (GA2) | ~16 s |
| NE-PDP101 | -36.71 | -39.47 (GA2) | ~16 s |
| NE-PDP121 | -62.36 | -67.78 (LKH3) | ~20 s |
| NE-PDP141 | -90.72 | -105.79 (LKH3) | ~18 s |

> 注：目标函数为负值（最小化成本），数值越小越好。

---

### **与基线方法的对比结果**

#### ✅ **在中小规模（≤61节点）**
- DCGA 推理最快（秒级），但略逊于 GA2（需近12分钟）；
- 表明在小规模下，充分迭代的传统搜索仍具竞争力。

#### ✅ **在中大规模（≥81节点）**
- **DCGA 全面超越所有基线**，成为新的 SOTA；
- 相比 LKH3、ALNS、GA、SA 等经典方法，gap 达 **70%~150%**；
- 相比学习型方法 NCS/NCS2，DCGA 性能提升巨大（gap >150%），且后者推理耗时高达数小时；
- **即使面对增强版 NCS2（加入相同图编码器）**，DCGA 依然大幅领先 → 证明优势来自整体架构而非单一模块。

#### ✅ **计算效率碾压**
- DCGA 平均**单实例推理时间 < 21 秒**；
- 而 LKH3 在大实例上需要 **数小时至数天**；
- NCS/NCS2 单次推理也需 **数十分钟至数小时**。

---

### **消融实验结果**

#### 🔹 **掩码机制消融（Masking Ablation）**
- 移除除基本连通性外的所有掩码（容量、访问次数、终止条件等）；
- 结果：训练不稳定，eval reward 显著下降；
- 原因：策略频繁进入无效序列，导致大量未履约惩罚（unmet-demand penalty ↑↑）；
- **结论**：约束感知掩码对稳定探索至关重要。

#### 🔹 **双通道 vs 单通道（Single-Channel Graph Attention, SCGA）**
- 替换为单一图编码器，融合所有信息；
- 结果：在所有规模下性能均劣于 DCGA，且差距随规模增大而扩大；
- **结论**：双通道设计有效缓解了表征冲突，尤其在大规模稀疏图中更为重要。

#### 🔹 **敏感性分析（Hyperparameter Sensitivity）**
- **嵌入维度越大，性能越好**（diminishing returns）；
- **Graph Encoder 的注意力头数敏感**，太少或太多都会损害性能；
- **Fusion Encoder 的头数影响较小** → 融合阶段更偏向整合而非复杂建模。

---

## **4. 关键结论和发现**

### **主要发现**
1. **结构感知学习优于通用坐标映射**  
   在稀疏、非欧、循环服务场景下，直接将经纬度输入标准NCO模型（如NCS）效果极差；必须显式建模网络结构。

2. **双通道分离建模显著提升性能与稳定性**  
   将“能否走”（network channel）与“要不要服务”（demand channel）分开处理，避免语义干扰。

3. **约束前置（hard masking）比后修复更高效可靠**  
   通过 step-wise masking 主动规避非法动作，比依赖采样+修复的方式更适合高约束环境。

4. **DCGA 是首个能在秒级时间内解决大型 NE-PDP-CCS 的方法**  
   兼具高质量与低延迟，适用于实际工业系统的滚动重规划。

---

### **方法的局限性**
- 当前仅针对**单条服务环路**（single service loop），尚未扩展到多路线协同调度；
- 假设旅行时间为静态，未考虑拥堵、天气等动态因素；
- 模型依赖预定义图结构，对拓扑剧烈变化的泛化能力有待验证；
- 虽然推理快，但**训练成本较高**，需大量GPU资源。

---

### **未来工作方向**
1. **扩展至多路线系统（multi-route planning）**  
   支持共享运力、协调中转与系统级调度。

2. **引入随机性与动态性建模**  
   如时间依赖旅行时间、需求不确定性、突发事件响应，可通过鲁棒RL或场景训练实现。

3. **结合轻量级局部搜索改进机制**  
   将 DCGA 作为构造器，配合 ALNS 或 LKH 进行微调，进一步逼近最优。

4. **理论分析可行性边界与近似比**  
   探索 DRL 在稀疏图组合优化中的可证性能界，建立与传统优化的桥梁。

--- 

> 📌 **一句话总结**：  
> DCGA 是一种面向**稀疏非欧网络**的新型端到端DRL求解器，通过**双通道图编码 + 约束感知解码**，实现了在**秒级时间内超越传统优化与神经求解器**的性能，为现实物流系统提供了高效可靠的决策引擎。

</details>

---

### 7. [DraftExpert: Expansion-Aware Self-Speculative Decoding for End-Device MoE Inference](https://arxiv.org/abs/2607.24434)

**Authors**: Dengke Han  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.24434v1  

#### Abstract
Large Mixture-of-Experts (MoE) language models are attractive for end-device deployment because only a small subset of experts is active per token, but their routed expert weights often exceed accelerator memory. We target latency-critical single-user settings where routed experts are staged on dema...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*DraftExpert: Expansion-Aware Self-Speculative Decoding for End-Device MoE Inference*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**端侧设备上的 MoE（Mixture-of-Experts）语言模型推理**中，**专家权重因显存不足而被卸载到 CPU 内存或 Flash 存储**（即 *expert-offloaded* 场景）所引发的新瓶颈，提出了一种新的 speculative decoding 框架。

在该场景下，传统 speculative decoding 的三大前提失效：
- **Drafting 不再廉价**：使用更多 routed experts（如 top-r）提升 draft 准确率的同时，会显著增加从 CPU/Flash 加载专家的开销；
- **Verification 成本上升**：验证一个 token block 可能激活多个不同的 target expert，导致并行验证的成本远高于单步；
- **Acceptance 率低**：若为控制成本仅用 top-1 或共享路径，隐藏状态漂移会导致 draft 与 target 差异大，接受率低。

因此，**核心问题是：如何在专家动态加载的代价主导延迟的背景下，实现高效、低成本的 self-speculative decoding？**

### 提出的新方法或新思路
作者提出了 **DraftExpert** —— 一种 **expansion-aware** 的 self-speculative decoding 框架，其核心思想是将优化目标从“每单位时间生成更多 token”转变为“**每新增一个加载的专家，尽可能多地接受 draft token**”。

#### 主要创新点包括：
1. ✅ **固定占用空间的 drafter 架构（Fixed-Footprint Drafting）**  
   每层 MoE 引入一个轻量级的 **accelerator-resident draft expert**，采用 `shared + top-1 + draft-expert` 路径进行 drafting，确保 drafting 阶段的 **expert-set expansion 受控且接近 top-1 水平**。

2. ✅ **多信号自蒸馏训练（Self-Distillation with Multiple Signals）**  
   利用冻结的完整 target MoE 对 draft expert 进行监督训练，损失函数包含：
   - **Residual loss**：恢复被 top-1 忽略的其他专家残差；
   - **Logit/Token loss**：提升输出分布和 token 预测一致性；
   - **Router-agreement loss**：使 draft 隐藏状态更贴近 target，提高对 verifier 将使用的专家的预测准确性。

3. ✅ **基于扩展代价的动态控制机制**
   - **Cost-aware Dynamic Truncation**：结合 draft token 的置信度 $q_i$ 和预测的 verifier 专家增量 $\Delta$，当低置信且高扩展代价时提前终止 drafting；
   - **Draft-Router Expert Prefetching**：利用 draft router 的预测结果，在 drafting 阶段异步预取可能需要的 verifier 专家，重叠传输延迟。

所有最终输出 token 均由完整的 target MoE 验证，保证输出等价于原模型。

### 相比现有方法的优势
| 方面 | 现有方法（如 shared+top-r） | DraftExpert |
|------|-------------------------------|-----------|
| **Draft 成本** | 随 r 增加而线性增长（需加载更多专家） | 固定 footprint，仅加载 top-1 + 小型 draft expert |
| **Acceptance 率** | top-3 仍低于 50%，精度有限 | 达到 **84–87%**，显著更高 |
| **Verification 控制** | 无显式控制，block 越长越慢 | 动态截断 + 预取，降低实际验证开销 |
| **Prefetch 效果** | 依赖不准确的 cheap drafter，命中率低 | 借助 router-agreement 蒸馏，**prefetch hit rate 达 86–88%** |

---

## 2. 核心实验方法和设置

### 使用的模型与平台
- **Target Models**:
  - **DeepSeek-V2-Lite (DS)**：27 层，64 routed + 2 shared experts，top-6 激活
  - **Moonlight-16B-A3B (ML)**：同结构
- **部署场景**（两种端侧内存层级）：
  - **CG（Consumer GPU）**：CPU → GPU 卸载，BF16 精度，RTX 4090
  - **MN（Mobile NPU）**：Flash → NPU 卸载，Q4_0 量化，Snapdragon 8 Elite 上的 Hexagon HTP v81 NPU，使用 llama.cpp

> 所有非 routed 组件（attention、router、output head 等）及 draft expert 均驻留加速器；routed experts 按需从 CPU DRAM 或 Flash 加载。

### 基线方法对比
- **AR Offload**：标准自回归推理，每次只生成一个 token，专家按需加载（baseline）
- **Shared+top-r**：最强的无需训练的 self-drafting 方法，r ∈ {1,2,3}，经调优选择最优配置
- **DraftExpert**：本文方法，含蒸馏训练、confidence-expansion truncation、target-expert prefetching

### 评估指标
- **Decode Throughput (TPS)**：prefill 后的解码阶段每秒生成 token 数
- **Speedup**：相对于 AR offload 的加速比
- **Draft Acceptance Rate**：draft token 被 target 接受的比例
- **Expert Set Expansion**：draft/verify 阶段累计加载的唯一 routed expert 数量
- **Prefetch Hit Rate**：预取成功的专家比例
- **Ablation Study**：分析不同蒸馏损失项的影响

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table 2）

| Model | Platform | AR TPS | top-r TPS | **DraftExpert TPS** | **Speedup** |
|-------|----------|--------|-----------|---------------------|-------------|
| DS    | CG       | 2.19   | 1.19      | **2.99**            | **1.36×**   |
| DS    | MN       | 10.18  | 4.80      | **15.47**           | **1.52×**   |
| ML    | CG       | 1.94   | 1.01      | **2.50**            | **1.29×**   |
| ML    | MN       | 8.50   | 3.82      | **13.69**           | **1.61×**   |
| **Avg.** | —        | —      | —         | —                   | **1.45×**   |

> 💡 在 Flash→NPU 场景下增益更大，说明 DraftExpert 特别适合带宽受限的移动端部署。

### 与基线方法对比结果

#### ✅ Draft-Side 性能（Table 3）
- **top-3** 虽然 acceptance 提升至 ~45%，但 unique expert 数翻倍以上，cost 显著上升；
- **DraftExpert** 在 expert footprint 仅比 top-1 多约 5% 的情况下，acceptance 高达 **85%**（DS）和 **83%**（ML），实现“高精度 + 低成本”。

#### ✅ Verify-Side 控制（Table 4）
- 固定长度验证（Fixed K）导致 verify 成本随 block 增长迅速上升（最高达 3.3×）；
- 加入 **confidence-expansion truncation** 后，verify 成本降至 ~1.5×；
- 再加入 **prefetch**，进一步压缩至 **1.25–1.40×**，接近理想水平。

#### ✅ Prefetch 效果
- **Prefetch Hit Rate**：**86–88%**（见 Table 5），浪费率仅 12–14%
- 高命中得益于 router-agreement 蒸馏带来的精准预测能力

### 消融实验结果（Table 5）

| 损失项 | Effect |
|-------|--------|
| **+ Residual** | Acceptance 从 ~22% → ~77% |
| **+ Logit/Token** | Acceptance 进一步提升至 **84–87%** |
| **+ Router Agreement** | Router KL 下降，**prefetch hit rate 从 ~70% → 86–88%** |

> 表明三项蒸馏信号各有分工：前两者提升 draft 精度，后者提升 verify-side 可预测性。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 在 **expert-offloaded MoE 推理**中，**expert-set expansion 是 speculative decoding 的主要瓶颈**，而非单纯的计算或 token 数量。
2. 🔄 传统的 speculative decoding 分析范式（以 target forward 为单位）不再适用，应转向“**accepted tokens per expert loaded**”这一新度量。
3. ✅ **DraftExpert 成功重构了 speculative decoding 的三个经典条件**：
   - **Cheap drafting**：通过固定 footprint 的 draft expert 实现；
   - **Controlled verification**：通过 confidence-expansion truncation 和 prefetch 实现；
   - **High acceptance**：通过多信号 self-distillation 恢复精度。
4. 📈 在真实端侧硬件上实现了平均 **1.45× 的 decode throughput 提升**，并在 Flash→NPU 场景下达到 **1.61×**，具有强实用性。

### 方法的局限性
- ❗ **依赖额外训练**：需要对每个 target MoE 单独训练 draft expert，增加部署复杂性；
- ❗ **仅适用于静态 MoE 结构**：假设 routing pattern 固定，难以适应动态路由变化；
- ❗ **prefetch 浪费带宽风险**：尽管 hit rate 高，但在极端情况下仍可能导致无效数据传输；
- ❗ **未解决 prefill 阶段优化**：focus 在 decode 阶段，prefill 仍为标准方式。

### 未来工作方向
- 🚀 设计 **免训练的初始化策略**，减少对 distillation pipeline 的依赖；
- 🔄 探索 **dynamic expansion budget**，根据当前缓存状态自适应调整 truncation 阈值 $B$；
- 🧠 将 draft expert **集成进模型架构设计**，形成“天生支持 speculative”的 MoE 架构；
- 📱 扩展至 **多用户 batch 场景**，探索跨请求的 expert cache 复用与 speculative 协同。

--- 

> ✅ **总结一句话**：  
> **DraftExpert 重新定义了端侧 MoE 推理中的 speculative decoding 范式，通过 expansion-aware 的设计，在保持输出精确性的前提下，实现了高达 1.45× 的端到端解码加速，是面向资源受限设备的重要进展。**

</details>

---

### 8. [SpeechLLM Meets Federated Learning for End-to-End ASR: English and Italian Case Studies](https://arxiv.org/abs/2607.25716)

**Authors**: Mohamed Nabih Ali, Daniele Falavigna, Alessio Brutti  
**Category**: cs.CL  
**Published**: 2026-07-29  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.25716v1  

#### Abstract
Federated learning (FL) enables privacy-preserving training of automatic speech recognition (ASR) systems across distributed data sources, yet its application to large-scale speech language models (SpeechLLMs) remains unexplored. This paper presents the first systematic study of federated training f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*SpeechLLM Meets Federated Learning for End-to-End ASR: English and Italian Case Studies*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **SpeechLLM** 的端到端 **ASR** 系统通常依赖于集中式数据训练，这带来了严重的隐私和数据治理风险（如语音中包含生物特征和个人敏感信息）。尽管 **Federated Learning (FL)** 被广泛用于保护隐私，但其在大规模 **SpeechLLM** 架构上的应用尚未被系统研究。

本文首次探索并系统化地解决了以下挑战：
- 如何在分布式、非独立同分布（non-IID）的语音数据上高效训练大型 SpeechLLM。
- 如何缓解 FL 中通信开销大、模型不稳定、“client drift”等问题在超大规模语音-语言联合模型中的影响。

---

### 🚀 提出的新方法与创新思路

1. **首个面向 SpeechLLM 的联邦训练框架**
   - 首次将 FL 应用于基于 **SpeechLLM** 的端到端 ASR，实现去中心化的隐私保护训练。

2. **通信高效的参数更新机制**
   - 仅聚合可训练模块（**LoRA** 和 **Projector**）的参数，冻结主干 LLM 和语音编码器，大幅降低通信成本（减少约 90% 可训练参数量）。

3. **改进的 FedAvg 算法：Adaptive FedAvg**
   - 引入统一的指数学习率衰减策略（`η_t = η₀ × γ^(t/T)`），提升训练稳定性，加快收敛速度。
   - 相较于传统静态学习率，该策略使客户端在早期进行快速探索，在后期精细化调整。

4. **模块化架构设计**
   - 采用“语音编码器 → 投影层 → LLM + LoRA”的三段式结构，支持灵活适配不同语音编码器（如 WavLM、Whisper）。

---

### ⚖️ 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **隐私性** | 数据始终保留在本地设备，无需上传原始语音或转录文本 |
| **通信效率** | 仅传输轻量级适配模块（LoRA/Projector），显著降低带宽需求 |
| **训练稳定性** | Adaptive FedAvg 缓解了 non-IID 数据下的 client drift 问题 |
| **扩展性** | 支持多语言、跨域场景，为实际部署提供基础 |
| **性能保持** | 在英语和意大利语任务上接近集中式训练性能 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 语言 | 数据集 | 描述 |
|------|--------|------|
| **English** | **LibriSpeech-100 (LS)** | 包含约 100 小时高质量朗读语音，来自 LibriVox 有声书；使用 `train-clean-100` 训练，`test-clean` 测试 |
| **Italian** | **Multilingual LibriSpeech (MLS) Italian** | 包含约 247 小时意大利语朗读语音，对应公共领域有声书；用于评估非英语语言表现 |

> 表格 I 显示：LS 含 251 名说话人，MLS Italian 含 65 名训练说话人。

---

### 🔧 实验设置

- **FL 框架工具**：使用 [Flower](https://flower.dev/) 实现联邦协调。
- **客户端划分**：每个说话人作为一个独立客户端（共 316 客户端）。
- **每轮参与比例**：随机选择 30% 的客户端参与训练。
- **本地训练配置**：每客户端执行 10 epochs，batch size = 4。
- **总通信轮数**：100 轮。
- **优化器**：AdamW，初始学习率 0.001。
- **Adaptive FedAvg 参数**：
  - 初始学习率 `η₀ = 0.001`
  - 衰减因子 `γ = 0.9`
  - 周期 `T = 10`

---

### 🎯 评估指标

- 主要指标：**Word Error Rate (WER)**  
- 对比方式：
  - 联邦学习（Federated Learning） vs. 集中式训练（Central Training）
  - 不同语音编码器（WavLM-large vs. Whisper-medium）
  - 不同参数微调策略（Full FT vs. PEFT）

---

### 🆚 基线方法对比

| 方法 | 描述 |
|------|------|
| **Vanilla FedAvg** | 标准 FedAvg，无学习率调度 |
| **Adaptive FedAvg** | 本文提出的方法，带指数学习率衰减 |
| **Central Training** | 所有数据集中训练，作为性能上限参考 |
| **Full Fine-tuning (FT)** | 更新整个模型参数（不适用于 FL） |
| **PEFT 方法（LoRA/Adapters）** | 参数高效微调，仅更新少量新增参数 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### A. 英语 LibriSpeech 结果（使用 WavLM 编码器）

| 方法 | WER (%) @ Round 100 | 相对集中式差距 |
|------|---------------------|----------------|
| Central Training | 6.1% | — |
| **Adaptive FedAvg (Ours)** | **6.4%** | +0.3% |
| Vanilla FedAvg | 7.9% | +1.8% |

> ✅ **Adaptive FedAvg 在第 20 轮即达到 9.7% WER，远优于标准 FedAvg 的 19.7%**

#### B. 意大利语 MLS 结果（使用 WavLM 编码器）

| 方法 | WER (%) @ Round 100 |
|------|---------------------|
| Central Training | 20.1% |
| **Federated Learning** | **22.6%** |
| 差距 | +2.5% |

> ❗ 意大利语存在更大性能差距，可能因数据量少、说话人少导致 non-IID 更严重。

#### C. 使用 Whisper 编码器的结果（更强泛化能力）

| 数据集 | Federated WER | Central WER | 差距 |
|-------|---------------|-------------|------|
| LS (English) | 6.6% | 6.0% | +0.6% |
| MLS (Italian) | 18.7% | 17.5% | +1.2% |

> ✅ Whisper 表现出更小的联邦-集中差距，说明其预训练的多语言鲁棒性有助于 FL 场景。

#### D. 多语言联合训练（WavLM 编码器）

| 数据集 | Federated WER | Central WER | 差距 |
|-------|---------------|-------------|------|
| LS (English) | 16.8% | 6.1% | +10.7% ❗ |
| MLS (Italian) | 19.7% | 18.4% | -0.3% ✅（反超） |

> ⚠️ 英语性能下降明显，表明多语言混合训练在 FL 下仍具挑战；但意大利语表现稳健甚至略优。

---

### 🔍 消融实验结果（Ablation Study）

#### 实验：比较不同参数化策略在 LS 上的表现（见 Table II）

| 方法 | # Trainable Params | Central WER (%) | Federated WER (%) | 是否可行 |
|------|--------------------|------------------|--------------------|----------|
| WavLM-FT (全量微调) | 85.1M | 4.4 | ×（未收敛） | ❌ 不适合 FL |
| WavLM EL-adapters | 9.1M | 4.6 | 6.1 | ✅ |
| **Speech-LLM (LoRA + Projector)** | **8.4M** | **6.1** | **6.4** | ✅✅（最优平衡） |

> ✅ **Speech-LLM 更新参数减少 90.1%，但仍实现稳定收敛和接近中心化的性能**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **SpeechLLM 可以成功应用于联邦学习环境**  
   - 首次证明了基于 LLM 的端到端 ASR 模型可在 FL 框架下有效训练，且性能接近集中式训练。

2. **Adaptive FedAvg 显著提升收敛速度与稳定性**  
   - 指数学习率衰减策略有效缓解早期震荡，加速前 20 轮性能跃升（相对改进达 51%）。

3. **参数高效微调（PEFT）是联邦 SpeechLLM 的关键技术路径**  
   - 全量微调在 FL 中难以收敛，而 LoRA 等方法通过冻结主干、仅更新低秩矩阵，实现了通信与性能的双赢。

4. **Whisper 编码器在 FL 中更具鲁棒性**  
   - 得益于其大规模多语言弱监督训练，Whisper 在 non-IID 条件下表现出更强的泛化能力和更小的性能落差。

5. **语言与数据特性显著影响 FL 效果**  
   - 英语数据丰富、说话人多，FL 性能接近集中式；
   - 意大利语虽数据较少，但在某些设置下仍能逼近甚至反超集中训练，体现 FL 的潜力。

---

### ⚠️ 方法的局限性

1. **当前仅限于朗读语音（read speech）**
   - 实际应用场景中更多为 spontaneous speech，更具挑战性。

2. **多语言联合训练效果不佳（尤其英语退化严重）**
   - 表明当前聚合机制尚难处理高度异构的语言分布。

3. **缺乏严格的隐私保障机制**
   - 当前仅依赖数据不离开本地，尚未引入 **Differential Privacy** 或加密聚合。

4. **客户端异质性建模不足**
   - 所有客户端使用相同计算资源假设，未考虑真实边缘设备差异。

---

### 🔮 未来工作方向

1. **拓展至大规模多语言与跨领域 ASR**
   - 探索更多语言组合及真实噪声环境下的联邦训练。

2. **引入高级通信压缩技术**
   - 如梯度量化（quantization）、稀疏化（sparsification）、知识蒸馏（knowledge distillation）进一步降低通信负担。

3. **增强隐私保护机制**
   - 结合 **Differential Privacy** 或安全多方计算（Secure Aggregation）提供形式化隐私保证。

4. **个性化联邦学习（Personalized FL）**
   - 引入客户端特定适配器（client-specific adapters）或联邦持续学习（federated continual learning），缓解异质性问题。

5. **向更大规模 LLM 和真实设备部署演进**
   - 验证在手机、IoT 设备等资源受限平台上的可行性与延迟表现。

---

## ✅ 总结

本论文开创性地将 **SpeechLLM** 与 **Federated Learning** 相结合，提出了一个通信高效、隐私友好的端到端 ASR 训练框架。通过 **Adaptive FedAvg** 和 **PEFT（LoRA）** 技术，实现在英语和意大利语任务上接近集中式训练的性能（如 LS 上仅差 0.3–0.6% WER），同时大幅降低通信开销。实验验证了该方法的有效性与稳定性，为未来在医疗、客服、智能终端等敏感场景中部署多语言语音识别系统奠定了坚实基础。

</details>

---

### 9. [Reasoning with Memory: A Temporal Granularity-Adaptive Framework for Training-Free Long Video Understanding](https://arxiv.org/abs/2607.24794)

**Authors**: Linghao Meng, Qiankun Li, Junyuan Mao, Pujin Liao, Zhicheng He, Enbo Zhang, Kun Wang, Yang Liu, Huazhu Fu, Yueming Jin  
**Category**: cs.AI  
**Published**: 2026-07-29  
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

### **解决了什么问题**
当前的 **Multimodal Large Language Models (MLLMs)** 在长视频理解任务（LongVideoQA）中面临严重挑战，主要受限于其**有限的上下文窗口**（context window），无法处理密集的视频帧序列。为应对这一限制，现有方法通常采用**关键帧选择**（keyframe selection）策略，如均匀采样或基于查询相似度的静态选择。

然而，这些方法存在两大缺陷：
- **忽略时间动态性**：仅基于静态语义匹配选择帧，忽略了事件之间的**时序依赖**和**因果结构**。
- **缺乏对查询时间粒度的适应性**：不同问题对时间尺度的需求不同（例如，“发生了多少次” vs “某个瞬间做了什么”），但现有方法无法自适应调整。

### **提出了什么新方法或新思路**
本文提出 **ReMem**（Reasoning with Memory），一个**无需训练**（training-free）、**时间粒度自适应**（temporal granularity-adaptive）的关键帧选择框架，用于提升 MLLMs 在长视频理解中的推理能力。

#### **核心创新点**：
1. **双层级记忆增强机制**（Dual-Level Memory-Augmented Adaptation）：
   - **Query Level**：通过 **Memory-Driven Question Parsing** 利用 LLM 的长期记忆分析问题的时间粒度（temporal granularity）并提取关键实体（entities），增强文本信号。
   - **Video Level**：通过 **Synergistic Dual-Semantic Frame Alignment** 和 **Structure-Aware Dynamic Frame Routing**，利用视频内在的**结构化记忆**（structural memory）进行帧对齐与聚类，实现动态预算分配。

2. **时间感知的图扩散机制**（Temporal-Semantic Memory Graph）：
   - 构建一个由关键视觉状态组成的记忆图，结合**语义权重**和**时间邻接约束**，通过图扩散（graph diffusion）捕捉跨场景、长距离的依赖关系。

3. **动态平衡机制**：
   - 根据问题的时间粒度 $g$ 动态调节**空间语义相似度**（$S_v$）与**拓扑时间语义分数**（$S_t$）的权重，实现从局部精细到全局连贯的灵活切换。

### **相比现有方法的优势**
- **无需训练**：完全 plug-and-play，适用于任意现成 MLLM。
- **更强的时间推理能力**：显式建模时间结构，避免冗余帧聚集在局部高分段。
- **更高的泛化性和鲁棒性**：在多种时间跨度（短、中、长）下均表现优异。
- **显著性能提升**：在多个基准上达到 SOTA 零样本性能，大幅超越包括训练型方法在内的基线。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
在四个主流的 **LongVideoQA** 基准上进行全面评估：
- **LVBench** [35]：极端长度视频理解基准，平均时长 68 分钟。
- **LongVideoBench** [38]：交错式视频语言理解基准，涵盖复杂时空推理。
- **MLVU** [51]：多任务长视频理解基准，包含分类、定位等任务。
- **Video-MME** [9]：综合评测 MLLMs 视频分析能力的基准，按视频长度分为 Short (~1.3min)、Medium (~9min)、Long (~41min) 子集。

### **实验设置和评估指标**
- **模型架构**：在三种主流 Video-LLM 上测试：
  - LLaVA-Video-7B-Qwen2
  - Qwen2-VL-7B-Instruct
  - Qwen3-VL-8B-Instruct
- **输入帧数**：固定帧预算（B=64 或 32）
- **编码器**：OpenAI CLIP-L-14 提取图文表示；GPT-4o 作为推理 LLM 进行问题解析。
- **初始采样**：原始视频以 1fps 均匀采样构建候选帧序列。
- **评估指标**：**准确率**（Answer Selection Accuracy），即正确选项被选中的比例。
- **实验性质**：全部为 **zero-shot**、**training-free** 实验。

### **基线方法对比**
与以下代表性方法比较：
- **Uniform Sampling**：基础均匀采样。
- **AKS** [33]：自适应关键帧采样，平衡相关性与覆盖范围。
- **Q-Frame** [47]：查询感知、多分辨率采样。
- **FlexSelect** [49]：上下文感知的概率采样。
- **BOLT** [24]：动态路由匹配证据与意图。
- **GenS** [39]、**FrameOracle** [15]*：*训练型方法（supervised）。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
| 方法 | LVBench | MLVU | LongVideoBench | Video-MME (Overall) |
|------|--------|------|----------------|---------------------|
| **LLaVA-Video (Baseline)** | 42.2 | 70.8 | 58.9 | 64.4 |
| **+ AKS** | — | — | 62.7 | 65.3 |
| **+ ReMem (Ours)** | **54.5** (**+12.3**) | **77.3** (**+6.5**) | **67.1** (**+8.2**) | **69.2** |

> 注：括号内为相对于基线的绝对提升。

- 在 **Qwen3-VL** 上，ReMem 在 LVBench 上实现 **53.3 → 42.7**（+10.6），在 MLVU 上达 **77.6%**（+14.7% vs FrameOracle*）。
- 所有 MLLM 架构下均取得一致且显著增益，验证了框架的**模型无关性**（model-agnostic）。

### **与基线方法的对比结果**
- **优于所有训练型方法**：
  - 尽管 **GenS** 和 **FrameOracle** 经过监督训练，ReMem 仍分别高出 **5.9%** 和 **14.7%**。
  - 表明**显式建模时空关联**比“暴力训练”更高效。
- **在长视频上优势明显**：
  - 在 Video-MME 的 Long 子集上，ReMem 达到 **61.8%**，而 AKS 仅为 **54.1%**（+7.7%）。
  - 显示其在**长时间跨度推理**上的强大鲁棒性。

### **消融实验结果**
在 **LLaVA-Video** 和 **Qwen2-VL** 上进行组件消融（Table 2 & Fig. 3）：

| 模块移除 | LVBench ↓ | MLVU ↓ | LongVideoBench ↓ |
|----------|---------|-------|------------------|
| 无 Entity Extraction (EE) | -2.2 | -1.5 | -0.4 |
| 无 Static Visual-Semantic Alignment (VSA) | -12.9 | -13.3 | -9.7 |
| 无 Temporal-Semantic Alignment (TSA) | -6.9 | -4.4 | -5.6 |
| 无 Dynamic Frame Routing (FR) | -6.8 | -3.7 | -5.4 |

#### **关键发现**：
- **VSA 是核心基础**：提供视觉-文本对齐桥梁，缺失导致最大性能下降。
- **TSA 至关重要**：尤其在长视频中，能连接因果事件，维持逻辑一致性。
- **FR 抑制冗余**：防止 Top-K 选择导致的上下文重复，提升多样性。
- **EE 提升细粒度聚焦**：帮助构建更精确的记忆图节点。

---

## 4. **关键结论和发现**

### **主要发现**
1. **时间粒度是影响长视频推理的关键因素**：问题的时间尺度应指导关键帧的选择策略，而非统一处理。
2. **记忆机制可有效桥接语义与时间结构**：
   - LLM 的**长期记忆**可用于解析问题语义；
   - 视频的**结构化记忆**可通过聚类+图建模捕获动态演化。
3. **无需训练即可超越监督方法**：通过精心设计的 memory-augmented 推理流程，可在不微调的情况下实现 SOTA 性能。
4. **动态加权优于二元分类**：连续的时间粒度建模（而非简单“长短”划分）更能适应真实世界的复杂事件分布。

### **方法的局限性**
- **预处理延迟较高**：由于引入图构建与扩散计算，**keyframe selection 阶段耗时增加**（约 +8–10 秒/样本）。
- **依赖高质量 CLIP 表示**：对细粒度动作或抽象概念的识别可能受限于视觉编码器的能力。
- **候选池大小敏感**：过大或过小都会影响性能，需经验调参（文中确定最优 N=150）。
- **未探索音频或多模态记忆融合**：目前仅基于视觉-语言通道。

### **未来工作方向**
- **轻量化图推理模块**：加速 memory graph 构建与扩散过程，提升端到端效率。
- **扩展至其他模态**：整合音频线索或语音转录，构建更丰富的多模态记忆。
- **在线自适应机制**：根据 MLLM 反馈动态调整关键帧集合（closed-loop selection）。
- **应用于视频摘要、检索等下游任务**：探索 ReMem 在非 QA 场景中的通用性。

---

> ✅ **一句话总结**：  
> ReMem 通过引入**记忆增强的双层级自适应机制**，首次实现了**无需训练的时间粒度感知关键帧选择**，在多个 LongVideoQA 基准上取得 SOTA 零样本性能，为高效、鲁棒的长视频理解提供了新范式。

</details>

---

### 10. [CoSA: Accelerating Long-Context Inference via Proxy-Kernel Co-Designed Sparse Attention](https://arxiv.org/abs/2607.25291)

**Authors**: Yufei Xue, Lin Niu, Hong Liu, Siran Liu, Hanyong Shao, Wei Liu, Guanghua Yu, Jianchen Zhu, Jun Zhang  
**Category**: cs.CL  
**Published**: 2026-07-29  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.25291v1  

#### Abstract
The quadratic cost of self-attention makes long-context inference prohibitively expensive, and proxy-based block-sparse attention has become a practical remedy. Existing methods typically rely on a proxy to predict a binary sparse mask and a kernel to consume this mask and perform sparse attention c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# CoSA: Accelerating Long-Context Inference via Proxy-Kernel Co-Designed Sparse Attention — 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决的问题**

标准的 self-attention 机制在处理长上下文时面临 **二次方计算复杂度**（quadratic cost），导致推理延迟极高，尤其在 128K 等超长序列场景下难以实用。现有的 **proxy-based block-sparse attention** 方法通过轻量级代理（proxy）预测重要 block 并生成稀疏掩码（sparse mask），再由 attention kernel 执行稀疏计算。

然而，当稀疏预算（sparsity budget）收紧时，proxy 容易遗漏真正重要的 blocks（salient blocks），而 kernel 只能机械地执行掩码，无法动态调整，导致模型精度显著下降。

### **提出的新方法与新思路**

论文提出了 **CoSA**（**C**o-designed **S**parse **A**ttention），一种无需训练的两阶段稀疏注意力机制，其核心是 **proxy-kernel co-design**（代理-内核协同设计）思想。

#### 主要组件：

- **Kernel-Aware Proxy (KAP)**  
  在 moderate 预算下选择 blocks，并输出一个 **ordered mask**（计算顺序掩码），不仅指定保留哪些 blocks，还规定它们在 kernel 中被访问的顺序。特别地，优先访问包含 rowmax 的 blocks（标记为 HRM blocks）。

- **Ordered-Skipping Kernel (OSK)**  
  利用 KAP 提供的有序掩码进行 **物理页面跳转**（physical page jumping），并结合在线 softmax 统计信息，在运行时进一步跳过更多不重要的 blocks（第二阶段稀疏化）。

### **相比现有方法的优势**

| 方面 | 传统方法 | CoSA |
|------|--------|------|
| **Proxy 设计** | 输出 binary mask，仅决定是否计算 | 输出 **ordered mask**，控制访问顺序 |
| **Kernel 行为** | 被动遵循掩码，无动态跳过能力 | 主动利用精确 logits 进行 in-kernel skipping |
| **协同机制** | Proxy 和 kernel 完全解耦 | **双向塑造**：proxy 影响 kernel 执行顺序，kernel 利用该顺序提升跳过效率 |
| **跳过保守性** | 易受“bucket effect”影响，跳过保守 | 通过提前访问 HRM blocks，使 running max 更快收敛，提升跳过激进程度 |

> ✅ **关键洞见**：order-invariance of OSM 允许任意访问顺序，这为重排序提供了理论基础；而重排序又能缓解 in-kernel skipping 的保守性问题。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **RULER**：合成任务基准，支持从 4K 到 128K 不同长度的上下文测试，用于评估模型对极端长程依赖的理解能力。
- **LongBench-v2**：真实世界长文本理解与推理任务集合，涵盖问答、摘要、逻辑推理等，更具实际挑战性。

### **实验设置与评估指标**

| 项目 | 设置说明 |
|------|----------|
| **模型骨干** | `Qwen3-8B`, `Llama-3.1-8B-Instruct` |
| **上下文长度** | 最高至 **128K tokens** |
| **稀疏范围** | 仅应用于 **prefill 阶段**，decode 保持 dense |
| **评估指标** | <ul><li>**准确率（Accuracy）**：各任务平均得分</li><li>**加权预算（Weighted Budget B↓）**：衡量计算资源消耗</li><li>**Attention Speedup**：注意力模块加速比</li><li>**Time-to-First-Token (TTFT) Speedup**：端到端首 token 延迟降低倍数</li></ul> |
| **硬件平台** | NVIDIA H20 节点 |

### **基线方法对比**

- **Dense**：完整 attention（FlashAttention-2）
- **MInference (MInf)**：基于预定义模式的 proxy
- **FlexPrefill (Flex)**：query-aware 的稀疏策略
- **XAttention (XAttn)**：基于反向对角线评分的 proxy

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 指标 | 结果 |
|------|------|
| **Attention Speedup @128K** | **4.93×** |
| **End-to-End TTFT Speedup @128K** | **2.53×** |
| **性能损失** | 准确率下降可忽略（negligible performance degradation） |
| **预算（B）** | 在 Qwen3-8B 上仅为 **22%**，低于所有 baseline |

### **与基线方法的对比结果**

#### ✅ 在 RULER 上的表现（Table 1）

| 方法 | Avg Acc (Qwen3-8B) | Budget |
|------|---------------------|--------|
| Dense | 89.20% | 100% |
| MInf | 86.77% | 49% |
| Flex | 87.72% | 28% |
| XAttn | 85.93% | 24% |
| **CoSA** | **88.65%** | **22%** ✅ |

➡️ CoSA 在更低预算下实现了最高准确率。

#### ✅ 在 LongBench-v2 上的表现（Table 2）

| 方法 | Avg Acc (Qwen3-8B) | Budget |
|------|---------------------|--------|
| Dense | 37.48% | 100% |
| MInf | 34.49% | 28% |
| Flex | 33.90% | 20% |
| XAttn | 33.20% | 18% |
| **CoSA** | **36.51%** | **15%** ✅ |

➡️ 即使在复杂推理任务中，CoSA 仍以最低预算取得最佳性能。

### **消融实验结果（Table 3）**

逐步添加 CoSA 各组件的效果分析（LongBench-v2, w/o CoT）：

| 变体 | Acc (Qwen3-8B) | Budget |
|------|----------------|--------|
| Base (binary mask) | 32.71% | 22% |
| + KAP (ordered mask) | 33.27% (+0.56) | 20% |
| + IKS (in-kernel skip) | 32.27% (-0.44) | 16% |
| **+ RMP (page remapping)** | **33.45% (+0.74)** | **15%** |

> 🔍 发现：
> - KAP 提升了 block 选择质量；
> - IKS 提高稀疏度但轻微降准；
> - **RMP（页面重映射）不仅恢复精度，还进一步提升性能**，验证了 co-design 的必要性。

---

## 4. 关键结论和发现

### **主要发现**

1. **Proxy 和 kernel 的脱节是限制高稀疏度下性能的关键瓶颈**。
2. **引入 ordered mask 替代 binary mask**，使得 proxy 能主动引导 kernel 的执行路径。
3. **提前访问 HRM blocks** 可快速提升 running max，从而让后续 in-kernel skipping 更安全且更激进。
4. **proxy-kernel co-design 实现了 1+1 > 2 的效果**：单独使用 KAP 或 OSK 效果有限，但二者结合显著优于任何独立改进。

### **方法的局限性**

- 当前仅适用于 **prefill 阶段**，未扩展至 decoding。
- decoding 阶段具有不同的 query shape 和计算模式，需专门设计。
- 对 KV-cache 的 page remapping 依赖现代 serving 框架（如 PagedAttention）。

### **未来工作方向**

- 将 CoSA 扩展至 **decoding 阶段**，实现全流程稀疏化。
- 探索更细粒度的跳过策略（如 token-level 或 vector-wise）。
- 结合硬件特性进一步优化 OSK 内核调度与内存访问模式。

---

> 🏁 **总结一句话**：  
> CoSA 通过 **proxy-kernel co-design**，首次将 **ordered mask** 引入稀疏 attention，实现了 **更高精度、更低预算、更快速度** 的长上下文推理，在 128K 场景下达到 **4.93× attention 加速** 和 **2.53× TTFT 降低**，同时几乎不损失性能。

</details>

---

### 11. [How Small Can You Go? A Controlled Study of LoRA Rank, Target Modules, and Quantization Trade-offs for Text-to-SQL on a 60M-Parameter Model](https://arxiv.org/abs/2607.25583)

**Authors**: Mahendra Singh Rathor, Anagheem Azzam  
**Category**: cs.AI  
**Published**: 2026-07-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.25583v1  

#### Abstract
Parameter-efficient fine-tuning (PEFT) and low-bit quantization are now standard tools for adapting language models under tight compute budgets, yet their interaction is most often studied on billion-parameter models where the design space is expensive to explore. We ask a complementary question: on...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*How Small Can You Go?*

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于在**小规模语言模型**（60M参数）上进行高效微调时，如何系统地研究 **LoRA Rank、目标模块选择和量化精度** 对任务性能与系统成本之间的权衡关系。现有研究多集中于十亿级大模型，其设计空间探索成本高昂，而本文则在可完全复现的小模型场景下，首次对这些效率“旋钮”进行了**受控、单变量的全面消融研究**。

### 提出的新方法/新思路
- **将适配过程建模为约束优化问题**：不仅关注准确率，还联合报告可训练参数量、峰值显存、推理延迟和吞吐等系统级指标，并通过 **Pareto 分析** 找出最优权衡点。
- **强调小模型上的效率饱和现象**：揭示了在 T5-small + WikiSQL 场景下，LoRA 的 rank 和模块扩展存在明显的收益递减效应。
- **开源完整实验流程**：发布所有代码、配置文件和训练日志，确保结果完全可复现。

### 相比现有方法的优势
- **更高的研究透明度与可复现性**：不同于大多数基于大模型难以复现的研究，本工作可在单张 T4 GPU 上完成全部实验（约35分钟），极大降低了验证门槛。
- **更贴近实际部署需求**：综合考虑准确率与内存、延迟等资源消耗，提供面向真实应用场景的决策依据。
- **填补小模型PEFT研究空白**：明确指出大模型上的结论（如高rank更好）不能直接迁移到小模型，推动了对小型化模型适配机制的理解。

---

## 2. 核心实验方法和设置

### 数据集
- **WikiSQL**：一个标准的单表 text-to-SQL 基准数据集。
  - 输入：自然语言问题 + 表格头信息
  - 输出：对应的 SQL 查询语句
  - 使用 5,000 条训练样本和 500 条验证样本，以保证单GPU可遍历整个设计空间。

### 模型架构
- **Base Model**: `T5-small`（共 60.5M 参数）
- **任务形式化**：序列到序列生成任务，输入格式为：
  ```
  translate English to SQL: <Q>table:<H1|H2|...>
  ```

### 实验设置
| 类别 | 配置 |
|------|------|
| **优化器** | AdamW |
| **学习率** | 5e-4 |
| **Batch Size** | 8 |
| **训练轮数** | 3 |
| **最大长度** | 输入 256 / 输出 128 tokens |
| **解码方式** | Beam Search (`num_beams=4`) |
| **LoRA 设置** | Dropout=0.1, bias 冻结, scaling α=2r |
| **硬件平台** | 单块 NVIDIA T4 GPU (16GB VRAM)，通过 Kaggle Notebooks 运行 |

### 评估指标
- **Exact-Match Accuracy (EM)**：生成 SQL 与黄金 SQL 字符串完全匹配（忽略大小写和空格）
- **Execution Accuracy (Exec)**：执行结果是否一致（仅针对可执行的 25 个样本子集，方差较高）
- **系统级指标**：
  - 可训练参数数量（Trainable Params）
  - 峰值训练显存（Peak Memory, GB）
  - 推理延迟（Latency, ms/query）
  - 吞吐量（Throughput, tok/s）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Zero-shot** | 不进行任何微调 |
| **Full Fine-tuning** | 微调全部 60.5M 参数（作为上界参考） |
| **LoRA** | 在不同 rank 和模块组合下应用 LoRA |
| **QLoRA (INT8/NF4)** | 结合 8-bit 或 4-bit 量化与 LoRA（r=8, {q,v}） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2，seed=42）

| Configuration | EM (%) | Exec (%) | Trainable Params | % Total | Mem (GB) | Latency (ms) |
|---------------|--------|----------|------------------|---------|-----------|--------------|
| Full FT       | 71.2   | 84.0     | 60.5M            | 100%    | 2.31      | ~380         |
| LoRA r=16 ({q,v}) | **59.6** | 80.0     | 589,824          | **0.97%** | **1.60**  | ~380         |
| LoRA r=8 ({q,v})  | 53.4   | 72.0     | 344,064          | 0.57%   | 1.59      | ~380         |
| QLoRA INT8        | 52.8   | 72.0     | 294,912          | 0.48%   | **0.60**  | 921          |
| QLoRA NF4         | **53.2** | 76.0     | 294,912          | 0.65%   | **0.60**  | **614**      |

> 注：LoRA r=16 在 {q,v} 上达到 **59.6% EM**，恢复了全微调 **83.7% 的准确率**，仅需 **<1% 可训练参数** 和 **31% 更低的峰值显存**。

### 与基线方法的对比结果
- **相比 Full FT**：
  - LoRA r=16 节省了 **99.03% 可训练参数** 和 **31% 显存**，准确率损失约 11.6 个百分点。
- **相比 LoRA r=8**：
  - 提升 6.2 个百分点准确率（53.4 → 59.6），仅增加约 24.5K 参数（相对增幅不大），显存几乎不变。
- **相比 Zero-shot**：
  - Zero-shot 准确率为 0%，说明任务需要特定适配。

### 消融实验结果

#### ✅ LoRA Rank 影响（r ∈ {2,4,8,16,32}）
- 准确率随 rank 提升持续增长至 **r=16**（59.6%），之后趋于饱和。
- **r=32** 仅带来 **+0.8%** 收益，但参数翻倍（从 589K 到 1.18M），显存仍稳定在 ~1.6GB。
- ➤ **结论：rank 饱和早且陡峭，r=16 是性价比最佳点。**

#### ✅ 目标模块影响（Target Modules）
| 模块设置 | EM (%) | 参数增量 | 准确率增益 |
|--------|--------|----------|------------|
| {q,v} (r=8) → {q,k,v,o} | 53.4 → 59.2 | +71% | +5.8 pts |
| {q,v} → {q,v,FFN} | 53.4 → 58.6 | +128% | +5.2 pts |
| {q,v} (r=16) | — | — | **59.6** |

- ➤ **结论：提升 rank 比扩大模块更高效；r=16 on {q,v} 已优于扩展模块方案。**

#### ✅ 量化影响（Precision）
- **INT8 / NF4 QLoRA** 达到 **52.8–53.2% EM**，接近 LoRA r=8 FP16（53.4%），仅损失 **0.2–0.6 pts**。
- 显存大幅下降至 **0.60 GB**（比 FP16 LoRA ↓62%，比 Full FT ↓74%）。
- 缺点是推理延迟显著上升（INT8: 921ms, NF4: 614ms），因 T4 上反量化开销大。
- ➤ **NF4 是最佳量化选择**：精度接近 FP16，显存极低，吞吐达 34 tok/s。

#### ✅ 多种子实验（Seeds 42, 123, 456）
- 所有配置的排序一致性高：Full FT > LoRA r=32 ≈ r=16 > r=8 > r=4 > r=2
- 低 rank 方差更大（如 r=2 std=0.063），表明初始化敏感
- QLoRA 极其稳定（INT8 std=0.006, NF4 std=0.015），可能得益于量化正则化作用

---

## 4. 关键结论和发现

### 主要发现
1. **LoRA r=16 on {q,v} 是 Pareto 最优配置**：
   - 在无严格内存限制下，它提供了最佳准确率/参数/显存权衡。
   - 仅训练 **0.97% 参数**，节省 **31% 显存**，恢复 **83.7% 全微调性能**。

2. **Rank 存在早期饱和现象（Early Saturation）**：
   - 超过 r=16 后无明显增益，挑战了“越大越好”的直觉，反映 WikiSQL 任务内在维度较低。

3. **模块扩展不如提升 rank 高效**：
   - 添加 k/o/FFN 层带来的准确率提升远低于所需额外参数，属于低效策略。

4. **QLoRA 在内存受限场景极具吸引力**：
   - INT8/NF4 仅损失不到 1 point 准确率，显存降至 **0.6GB**，适合边缘设备或低资源环境。

5. **量化提升稳定性**：
   - QLoRA 配置表现出更低的随机种子方差，暗示量化具有优化平滑化效果。

### 方法的局限性
- **任务单一性**：仅在 **WikiSQL**（单表、简单 SQL）上验证，无法推广至复杂多表任务（如 Spider、BIRD）。
- **模型规模限制**：结论基于 **T5-small (60M)**，未必适用于其他架构或更大/更小模型。
- **硬件依赖性**：QLoRA 的延迟劣势源于 T4 GPU 的反量化效率低，在专用硬件上可能改善。
- **执行准确率不可靠**：仅有 25 个可执行样例（占 5%），导致 exec accuracy 统计意义有限。

### 未来工作方向
- 将相同方法论扩展至 **multi-table text-to-SQL benchmarks**（如 Spider）验证泛化能力。
- 探索 **不同模型架构**（如 BERT-style encoder-only 或 decoder-only LLMs）下的效率边界。
- 研究 **混合精度训练 + LoRA** 的动态调节策略，实现运行时自适应。
- 开发针对小模型的 **自动化 PEFT 配置搜索工具**，结合 Pareto 优化自动推荐最优设置。

---

> 🔗 **可复现性声明**：  
> 所有代码、配置和日志已公开于 GitHub：[https://github.com/mahendrarathore1742/efficient_peft_small_models](https://github.com/mahendrarathore1742/efficient_peft_small_models)  
> 完整实验可在单张 T4 GPU 上 **35 分钟内跑完**，支持社区进一步验证与拓展。

</details>

---

### 12. [Enhancing Error Detection Performance through Parallel CRC Computation on Multi-Core Architectures](https://arxiv.org/abs/2607.24849)

**Authors**: Mohammad Javad Khani, Mahmood Ahmadi  
**Category**: cs.DC  
**Published**: 2026-07-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.24849v1  

#### Abstract
Cyclic Redundancy Check (CRC) remains one of the most widely used error-detection mechanisms in communication, storage, and embedded systems. However, conventional software CRC implementations suffer from inherent sequential dependencies that limit efficient utilization of modern multi-core processo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的软件实现 **CRC**（Cyclic Redundancy Check）算法具有固有的**顺序依赖性**，即当前状态依赖于前一个数据块的计算结果，这限制了其在现代多核处理器上的并行化潜力。尽管已有多种优化技术（如查表法、SIMD、硬件指令等），但它们通常存在以下局限：
- 仅支持特定 CRC 变体（如 CRC-32）
- 依赖特定硬件特性（如 PCLMULQDQ 指令）
- 缺乏跨平台可移植性
- 并行组合阶段可能破坏数学正确性（例如使用 naive XOR 聚合）

本文旨在解决如何在通用多核架构上实现**高效、正确且可移植的并行 CRC 计算**。

---

### 提出的新方法与创新点

1. ✅ **通用并行 CRC 框架（Generalized pthread-based framework）**
   - 基于 **POSIX threads (pthreads)** 实现，具备良好的跨平台可移植性。
   - 支持多种 CRC 标准：**CRC-8, CRC-16, CRC-32, CRC-64, CRC-128**，在一个统一框架中处理不同位宽和多项式结构。

2. ✅ **GF(2)-based 正确性保持组合机制（Correctness-preserving Combine Mechanism）**
   - 不采用简单的 XOR 合并部分 CRC 结果，而是基于有限域 **GF(2)** 上的多项式算术设计组合操作。
   - 引入 **matrix-based shifting operation** 来高效模拟 CRC 状态推进（state advancement），确保并行计算结果与串行完全等价。

3. ✅ **系统性实验评估**
   - 包括执行时间、吞吐量、延迟、扩展性分析以及能量消耗估计。
   - 分析同步开销、内存带宽影响及多核扩展瓶颈。

4. ✅ **强调实际工程权衡**
   - 探讨了同步成本、缓存争用、功耗行为等现实因素对性能的影响。

---

### 相比现有方法的优势

| 特性 | 本工作 | 典型查表/SIMD/硬件方法 |
|------|--------|--------------------------|
| 多变体支持 | ✅ 支持 CRC-8 到 CRC-128 | ❌ 多数仅支持 CRC-32 |
| 可移植性 | ✅ 纯软件，无硬件依赖 | ⚠️ 依赖 SSE/AVX/PCLMULQDQ 或专用指令 |
| 正确性保障 | ✅ 数学等价于串行 CRC | ✅（多数也保证） |
| 并行能力 | ✅ 多线程级并行 | ✅ SIMD 是数据级并行 |
| 部署灵活性 | ✅ 易集成到通用系统 | ❌ 需要特定编译器或硬件 |

> 🎯 定位明确：不追求极致性能超越硬件加速方案，而是在**可移植性、通用性和正确性之间取得平衡**，适用于通用多核系统的软件级加速需求。

---

## 2. 核心实验方法和设置

### 数据集
使用合成的大规模数据集进行测试，大小分别为：
- **100 MB**
- **500 MB**
- **1000 MB**

> 注：未使用真实应用 trace，但通过控制变量法验证规模效应。

---

### 实验平台配置（见 Table 2）

| 参数 | 配置 |
|------|------|
| CPU | Intel Core i5-2430M (2 cores / 4 threads) |
| 主频 | 2.40 GHz |
| 内存 | 4 GB RAM |
| OS | Windows 10 |
| 编译器 | GCC (Dev-C++ 6.3)，启用 `-O2` 优化 |
| 线程库 | POSIX Threads (**pthreads**) |
| 测试 CRC 类型 | CRC-8, CRC-16, CRC-32, CRC-64, CRC-128 |
| 线程数 | 1, 2, 4, 8 |
| 重复次数 | 10 次取平均值 |

---

### 评估指标

| 指标 | 定义 |
|------|------|
| **Execution Time** | 总运行时间（秒） |
| **Speedup** | $ \frac{T_{\text{serial}}}{T_{\text{parallel}}} $ |
| **Efficiency** | $ \frac{\text{Speedup}}{T} $，衡量资源利用率 |
| **Throughput** | $ \frac{\text{Processed Data (MB)}}{\text{Time (s)}} $，单位 MB/s |
| **Latency** | 小消息（4KB）处理延迟（ms） |
| **Energy Consumption** | 基于电池放电估算总能耗（J） |
| **Scalability Behavior** | 不同线程下的加速趋势分析 |

---

### 基线方法对比

- **Baseline**: 传统串行 CRC 实现（bitwise 或 lookup-table）
- **对比对象**（来自文献引用）：
  - Lookup-Table CRC
  - Slicing-by-8
  - SIMD / Vectorized CRC（如 SSE/AVX）
  - Hardware-assisted CRC（如 PCLMULQDQ 指令）
  - FPGA-based 实现

> ⚠️ 注意：这些对比为定性比较（非同一平台实测），用于定位本文方法在整体优化谱系中的位置。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 🔹 加速比与效率（Table 3, CRC-32, 1000 MB）

| Threads | Execution Time (s) | Speedup | Efficiency |
|--------|--------------------|---------|----------|
| 1      | 312.05             | 1.00    | 1.00     |
| 2      | 176.80             | 1.76    | 0.88     |
| 4      | 103.30             | 3.02    | 0.76     |
| 8      | 92.10              | 3.39    | 0.42     |

✅ 最高达到约 **3.4× 加速**，接近理论极限（Amdahl 定律预测 ~4.7×），但由于同步、超线程共享资源等原因出现亚线性扩展。

---

#### 🔹 吞吐量提升（Table 4, 1000 MB, 4 threads）

| CRC Type | Serial (MB/s) | Parallel (4T, MB/s) | 提升倍数 |
|----------|---------------|---------------------|--------|
| CRC-8  | 3.33          | 9.60                | ~2.9× |
| CRC-16 | 3.27          | 9.21                | ~2.8× |
| CRC-32 | 3.20          | 9.67                | ~3.0× |
| CRC-64 | 2.95          | 8.84                | ~3.0× |
| CRC-128| 2.22          | 5.40                | ~2.4× |

📌 所有 CRC 类型均显著提升；**CRC-128 因寄存器更宽、运算复杂度更高，相对增益略低**。

---

#### 🔹 不同数据集规模的影响（Table 9）

| Dataset Size | Serial Time (s) | Parallel Time (s) | Speedup |
|--------------|------------------|--------------------|--------|
| 100 MB       | 31.20            | 11.00              | 2.84   |
| 500 MB       | 156.70           | 55.20              | 2.84   |
| 1000 MB      | 312.05           | 103.30             | 3.02   |

📈 表明：**随着输入规模增大，并行优势更加明显**，因为固定开销（线程创建、同步）被摊薄。

---

#### 🔹 能耗表现（Table 7, 1000 MB）

| CRC Type | Serial Energy (J) | Parallel Energy (J) | 节省比例 |
|----------|-------------------|----------------------|--------|
| CRC-8  | 2890              | 1510                 | ~48% ↓ |
| CRC-32 | 3080              | 1320                 | ~57% ↓ |
| CRC-128| 4210              | 2210                 | ~47% ↓ |

💡 尽管并行时 CPU 占用率更高，但因**执行时间大幅缩短**，总体能量消耗显著降低 —— 对能效敏感场景有价值。

---

#### 🔹 延迟表现（Table 6, 4KB 小包）

| CRC Type | Serial Latency (ms) | Parallel Latency (ms) |
|----------|----------------------|------------------------|
| CRC-32   | 0.255                | 9.85                   |

⚠️ 并行版本延迟反而更高！说明：
- **小负载下线程管理开销占主导**
- 并行策略不适合 ultra-low-latency 场景（如嵌入式实时系统）

---

#### 🔹 与其他方法的定性对比（Table 5）

| 方法 | 报告加速比 | 本文框架对比 |
|------|------------|-------------|
| Lookup-Table | 2–5× | 相当或略优（3–4×） |
| Slicing-by-8 | 5–20× | 较低（但更通用） |
| SIMD/Vectorized | 10–40× | 显著更低（但依赖硬件） |
| Hardware Instructions | 20–50× | 远低于专用指令 |
| **本文方法** | **3–4×** | ✅ 可移植 + 多变体 + 正确性 |

➡️ 结论：虽未达到专用优化水平，但在**通用性与实用性上具有独特价值**。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **并行 CRC 在大数据集上可带来显著性能提升**：
   - 在 1000 MB 数据上实现了 **3–4× 的加速比**。
   - 吞吐量提升一致，最高达 **9.67 MB/s（CRC-32）**。

2. ✅ **GF(2)-based 组合机制有效保证了数学正确性**：
   - 并行结果与串行完全一致，满足通信/存储系统的严格要求。

3. ✅ **能量效率得到改善**：
   - 更短的执行时间抵消了更高的瞬时功耗，**整体能耗下降近 50%**。

4. ⚠️ **扩展性受现实因素制约**：
   - 超过 4 个线程后效率下降（从 76% → 42%），受限于：
     - 同步开销
     - Cache contention
     - Memory bandwidth saturation
     - Hyper-Threading 资源竞争

5. ⚠️ **小数据包场景不适合并行化**：
   - 4KB 数据下并行延迟远高于串行（>10ms vs <0.3ms），表明应根据负载动态选择模式。

6. 📊 **性能收益随数据规模增长而增强**：
   - 小数据（100MB）也能获益，但增益受限于启动开销。

---

### 局限性

1. **实验平台较旧**：
   - 使用的是双核四线程的 i5-2430M（2011年发布），无法反映现代 many-core 架构（如 16 核以上）的表现。

2. **能耗测量精度有限**：
   - 依赖电池百分比估算，缺乏 **RAPL** 或专用功率计等精确仪器支持。

3. **未与最先进指令级优化直接对比**：
   - 如未在同一平台上测试 `PCLMULQDQ` 或 AVX512 实现，难以量化差距。

4. **CRC-128 非标准工业规范**：
   - 当前主要用于验证框架扩展能力，而非实际部署推荐。

5. **静态划分策略简单**：
   - 未考虑负载不均衡或自适应调度策略。

---

### 未来工作方向

1. **在现代多核/众核平台上重新评估**：
   - 使用服务器级 CPU（如 Intel Xeon, AMD EPYC）测试更高线程数下的扩展性。

2. **引入更精细的 workload distribution 策略**：
   - 动态分块、任务窃取（work-stealing）以应对不规则输入。

3. **结合 instruction-level 优化进行混合加速**：
   - 在每个线程内部使用 SIMD 或 PCLMULQDQ 进一步提速。

4. **采用精确的能量监控工具**：
   - 如 **Intel RAPL**, **perf**, 或外接 power monitor，获取细粒度功耗数据。

5. **探索异构加速可能性**：
   - 将部分计算卸载至 GPU 或 FPGA，构建 hybrid CRC pipeline。

6. **应用于真实系统场景验证**：
   - 集成到文件系统、网络协议栈或数据库引擎中进行端到端评估。

---

> 🧩 **总结一句话**：  
> 本文提出了一种**可移植、通用、数学正确的多线程并行 CRC 框架**，在通用多核平台上实现了 **3–4× 的加速和近 50% 的能耗降低**，特别适合大规模数据处理场景，虽不及专用硬件极致性能，但在灵活性与实用性方面填补了重要空白。

</details>

---

### 13. [ACRL: Adaptive Control of Training-Inference Discrepancy for Stable Reinforcement Learning](https://arxiv.org/abs/2607.24062)

**Authors**: Wenwu Fan, Qihong Lin, Zhijie Xia, Zhuo Zheng, Sihao Wang, Qiang Chen, Liangsheng Zhu  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.24062v1  

#### Abstract
Reinforcement Learning (RL) training for Large Language Models (LLMs) often suffers from instability due to the discrepancy between training and inference. This training-inference discrepancy stems from two primary factors: an architectural separation between training and inference engines, and the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ACRL: Adaptive Control of Training-Inference Discrepancy for Stable Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

该论文针对 **Large Language Models (LLMs)** 在 **Reinforcement Learning (RL)** 微调过程中普遍存在的 **训练-推理不一致性（Training-Inference Discrepancy）** 所导致的 **训练不稳定甚至崩溃（training collapse）** 问题。

这种不一致性主要来源于两个方面：
- **架构分离**：训练引擎（如 FSDP）与推理引擎（如 vLLM）实现不同；
- **精度差异**：推理阶段常采用低精度量化（如 FP8），而训练仍使用高精度格式（如 BF16）。

这些因素导致训练策略 $\pi$ 和推理策略 $\mu$ 的概率分布出现偏差，使本应为 on-policy 的学习退化为 off-policy 学习，从而引发梯度失真、方差增大、收敛困难等问题。

---

### 🚀 提出的新方法与新思路

作者提出 **Adaptive Control Reinforcement Learning (ACRL)**，其核心思想是：

> **将训练-推理不一致性视为一个可控制的动态变量，并通过自适应反馈机制将其维持在一个“合理范围”内，而非一味消除或忽略。**

#### 创新点包括：

1. **首次引入“自适应控制”视角**  
   将 RL 中的稳定性问题类比于控制系统中的误差调节，通过设计一个 **ACRL Controller** 动态调整策略更新强度，防止不一致性过大（导致崩溃）或过小（导致欠探索、精度下降）。

2. **双向调节机制（Bidirectional Discrepancy Control）**
   - 当当前序列级不一致性 $Y > X$（参考值）时，**减小**训练对高置信度 token 的过度拟合；
   - 当 $Y < X$ 时，**主动增强**差异以促进探索，避免模型被过度约束。

3. **基于 token-level 的自适应重要性采样（Importance Sampling）**
   - 不同于传统的 token-level IS 或 sequence-level IS，ACRL 使用序列级测量 $Y$ 来动态生成每个 token 的调整权重 $\rho_{i,t}$，实现局部精细化控制。

4. **理论牺牲换取实践稳定性的权衡设计**
   - 明确承认 ACRL 是一个 **biased estimator**，但指出在低精度环境下，**稳定性优先于无偏性**，延续了 PPO/GRPO 中 clip 等机制的设计哲学。

---

### 🔍 相比现有方法的优势

| 方法 | 缺陷 | ACRL 如何改进 |
|------|------|----------------|
| **Token-level IS (TIS)** | 高方差、无法处理序列级偏差，易波动 | 引入序列级参考信号进行平滑控制 |
| **Sequence-level IS (MIS)** | 梯度消失严重，重要性权重趋近于零，难以收敛 | 避免全局缩放，采用指数型自适应权重缓解数值问题 |
| **统一精度训练（如全用 FP8）** | 虽然减少差异但损失表示能力，降低准确率（accuracy tax） | 允许 FP8 推理 + BF16 训练，恢复 BF16 准确率，消除“量化税” |
| **静态对齐策略** | 无法泛化到其他差异源（如 kernel 实现差异） | 控制框架通用性强，适用于多种不一致来源 |

✅ 总结优势：
- 更强的 **训练稳定性**
- 更高的 **最终准确率**
- 更好的 **探索能力（exploration）**
- 支持 **极端量化场景（如 Truncated FP8）**

---

## 2. 核心实验方法和设置

### 📚 数据集

- **数学推理任务**：
  - **GSM8K**：小学数学应用题基准
  - **AIME, AMC, HMMT, MATH500**：高难度数学竞赛题集合
- **综合知识推理任务**：
  - **MMLU-Pro**：涵盖 14 个领域的 12,000 道高质量选择题（生物、法律、物理、哲学等）

---

### ⚙️ 实验设置

| 项目 | 设置说明 |
|------|----------|
| **模型** | Qwen2.5-3B/7B/32B, Qwen3 MoE 架构（30B-A3B） |
| **训练框架** | VeRL（支持 GRPO/PPO/DAPO） |
| **训练引擎** | FSDP（BF16） |
| **推理引擎** | vLLM（FP8/MXFP8 量化） |
| **量化方式** | Per-token activation + per-channel weight FP8 quantization；部分实验使用更激进的 **truncated FP8** |
| **评估指标** | Pass@1 Accuracy（主）、平均最后 200 步准确率、训练熵（entropy）、梯度范数、奖励曲线 |

---

### 🆚 基线方法对比

| 基线方法 | 描述 |
|--------|------|
| **BF16 Baseline** | 双端均使用 BF16，作为高精度上限 |
| **FP8 (Uncorrected)** | 推理用 FP8，无任何修正，用于展示崩溃风险 |
| **TIS (Token-level IS)** | 每个 token 应用截断重要性采样 |
| **MIS (Masked IS)** | 序列级单一重要性权重，理论上无偏但实践中不稳定 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ 表格 1：Qwen2.5-3B on GSM8K（标准设置）

| Method | Peak Acc | Avg Acc (last 200 steps) |
|--------|----------|-------------------------|
| BF16   | 87.72%   | 86.60%                  |
| TIS    | 88.17%   | 86.73%                  |
| MIS    | 88.17%   | 86.29%                  |
| **ACRL** | **88.10%** | **87.05%** ✅ |

> 💡 **结论**：ACRL 不仅稳定训练，且平均准确率超过所有基线，尤其优于 MIS。

---

#### ✅ 表格 2：Truncated FP8 极端压力测试（GSM8K）

| Method | 是否成功 | Accuracy | 相对 BF16 |
|--------|----------|----------|-----------|
| BF16   | 成功     | 85.90%   | —         |
| FP8    | 失败 ❌   | ×        | —         |
| TIS    | 成功     | 84.69%   | -1.21%    |
| MIS    | 失败 ❌   | ×        | —         |
| **ACRL** | **成功 ✅** | **86.13%** | **+0.23%** ✅ |

> 💥 在极端量化下，只有 ACRL 成功稳定训练并反超 BF16！

---

#### ✅ 表格 3：Qwen2.5-7B on 多项高难数学任务（AIME/AMC/MATH 等）

| Method | Average Accuracy |
|--------|------------------|
| BF16   | 35.11%           |
| TIS    | 34.57%           |
| MIS    | 34.93%           |
| **ACRL (γ=0.65)** | **36.39%** ✅ |

> 🎯 ACRL 在更大模型上显著超越 BF16 基线，证明其可扩展性。

---

#### ✅ 表格 5：架构扩展性测试（32B Dense & MoE）

| Model | Method | Avg Acc (600–800) |
|-------|--------|--------------------|
| Qwen3-32B | BF16 | 0.9606 |
| | MXFP8+ACRL | **0.9613** ✅ |
| Qwen3-30B-A3B (MoE) | BF16 | 0.9558 |
| | MXFP8+ACRL | **0.9578** ✅ |

> ✅ 即使在 MoE 这种对 top-k routing 敏感的结构中，ACRL 依然有效。

---

#### ✅ 表格 6：MMLU-Pro 综合推理能力（跨领域泛化）

| Training Method | FP8 Inference | BF16 Inference |
|------------------|---------------|----------------|
| BF16 Baseline    | 0.4484        | 0.4491         |
| FP8 + TIS        | 0.4480        | 0.4530         |
| **FP8 + ACRL**   | **0.4534** ✅  | **0.4562** ✅  |

> 🧠 表明 ACRL 提升的是通用推理能力，非仅限于数学任务。

---

### 🔬 消融实验结果

#### 表 10：单向控制 vs 完整 ACRL（GSM8K）

| Method | Peak Acc | Avg Acc |
|--------|----------|---------|
| Full ACRL | 88.10% | 87.05% |
| Continuous Reduction（只减不增） | 87.19% | 86.14% |
| Fallback to GRPO（Y<X时不干预） | 87.72% | 86.97% |

> 📉 结果表明：**双向控制至关重要**。仅减少差异会导致欠探索；完全放弃 Y<X 时的调节则损失性能。

---

#### 表 9：控制强度 γ 敏感性分析

| γ | Accuracy | 是否完成 |
|----|----------|--------|
| 0.1 | 0.8749 | 成功 |
| 0.5–1.7 | ↑ 最高达 **0.8832** ✅ | 成功 |
| ≥2.1 | ↓ 下降至 0.8696 | **失败 ❌** |

> ⚠️ 过强控制（large γ）会过度压缩参数空间，导致训练崩溃。

---

#### 图 8 & 10b：训练熵（Exploration）对比

- **ACRL 始终保持最高 entropy**
- MIS 虽有高熵，但多来自无意义输出（“garbage tokens”）
- ACRL 的熵提升源于 **有意义的探索增强**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **训练-推理不一致性必须被主动控制，而非被动容忍或强行消除**  
   → 过大导致崩溃，过小损害探索与准确率。

2. **ACRL 通过自适应反馈机制实现了稳定的训练过程**  
   → 在 FP8 推理下仍能匹配甚至超越 BF16 准确率。

3. **ACRL 内在促进了策略熵增加，增强了探索能力**  
   → 通过 AGPA 机制强化 PA&LP / NA&HP 类 token 的影响。

4. **ACRL 具备良好的可扩展性和鲁棒性**  
   - 支持 GRPO/PPO/DAPO 等多种算法
   - 适配 Dense / MoE 架构
   - 在极端量化（truncated FP8）下仍表现优异

5. **计算开销极低**  
   → ACRL 模块仅增加约 **0.1%** 的训练延迟，几乎无额外成本。

---

### ⚠️ 局限性

1. **依赖静态参考值 $X$**  
   - 当前 $X$ 固定为初始步骤计算值，长期训练中可能不再适用；
   - 未来可探索动态更新机制（如移动平均）。

2. **超参数敏感性存在边界**  
   - 控制强度 $\gamma$ 需谨慎选择，过大或过小都会影响性能。

3. **实证集中在数学与客观任务**  
   - 尚未充分验证在主观任务（如创意写作、对话安全）中的效果。

4. **评估范围有限**  
   - 当前实验集中于单轮推理、vLLM+FSDP 组合；
   - 多轮交互、工具调用、异步 pipeline 等复杂场景尚未覆盖。

---

### 🔮 未来工作方向

1. **探索更优的不一致性度量方式**  
   - 如 KL 散度、Euclidean 距离替代绝对差。

2. **开发动态参考基准 $X_t$**  
   - 使用滑动窗口估计自然差异漂移。

3. **拓展至更多主观 RLHF 场景**  
   - 验证 ACRL 在人类偏好建模中的探索收益。

4. **跨平台兼容性研究**  
   - 在不同推理/训练后端组合（如 SGLang + Megatron）中验证通用性。

5. **结合硬件感知优化**  
   - 将 ACRL 与芯片级低精度支持深度协同设计。

---

## ✅ 总结

ACRL 提出了一种全新的 **“自适应控制”范式** 来解决 LLM 强化学习中的训练-推理不一致性问题。它不是简单修补梯度偏差，而是将整个训练过程看作一个需要调控的系统，通过动态平衡不一致性来实现：

- ✅ **训练稳定**
- ✅ **探索增强**
- ✅ **准确率恢复甚至超越 BF16**
- ✅ **支持 FP8/MXFP8 等高效推理方案**

该方法在多个模型规模、算法、任务和架构上均取得领先结果，为 **低精度高效 RLHF 训练** 提供了一个稳健、实用且极具前景的技术路径。

</details>

---

### 14. [A Cost-Effective Multimodal LLM Reasoning Framework for Question Answering over Irregular Clinical Time Series](https://arxiv.org/abs/2607.25947)

**Authors**: Frank Nie, Ethan B Liu, Yuan Zhu, Wei Fan, Jindong Han  
**Category**: cs.AI  
**Published**: 2026-07-29  
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
本文针对**不规则临床时间序列**（Irregular Clinical Time Series, ICTS）上的**问答任务**（Question Answering, QA）中存在的三大挑战提出解决方案：
- **不规则时序建模**：临床观测具有稀疏、异步、非均匀采样等特点，传统模型难以有效捕捉其动态模式。
- **多尺度证据定位**：不同问题依赖于不同时间尺度的证据（如全局趋势、局部事件、单个测量值），且相关证据往往稀疏分布。
- **不规则时序-语言对齐困难**：缺乏大规模配对数据来连接不规则时间模式与自然语言语义。

现有方法（如文本序列化、表示适配或多模态LLM）在处理常规时间序列上表现良好，但在临床场景中因忽略稀疏性和异步性而性能受限。

---

### 提出的新方法：CLINPRISM
作者提出了 **CLINPRISM** —— 一种**低成本、高效的多模态大语言模型推理框架**，专为ICTS上的QA设计。其核心创新包括：

#### （1）不规则感知的多尺度编码器（Irregularity-Aware Multi-Scale Encoder）
- 在三个时间尺度上分别建模：
  - **Macro-Scale**：捕获整个轨迹的长期趋势和跨变量上下文。
  - **Meso-Scale**：通过可学习软窗口提取局部片段级动态（如低血压发作期）。
  - **Micro-Scale**：保留细粒度参考时间点的状态及其观测支持强度。
- 所有尺度直接作用于原始不规则序列，无需重采样或插值，保留了真实的时间间隔和缺失模式。

#### （2）时序证据蒸馏器（Temporal Evidence Distiller）
- 将异构的多尺度输出投影到统一的LLM空间，并进行**分层融合**（Hierarchical Fusion）。
- 使用**基于查询的重采样机制**（Query-Based Resampler）将变长表示压缩为固定数量的`temporal tokens`（默认16个），实现高效交互。

#### （3）渐进式时序-语言对齐策略（Progressive Temporal-Language Alignment）
采用三阶段训练流程：
1. **Stage 1**: 多尺度编码器预训练（无监督自监督目标）。
2. **Stage 2**: 冻结编码器和LLM，仅优化蒸馏器，利用**分层caption**进行跨模态对齐。
3. **Stage 3**: 分两步微调以适应下游QA任务：
   - Step 1: 固定LLM，微调蒸馏器；
   - Step 2: 引入LoRA模块并联合优化蒸馏器与LoRA参数。

该策略实现了从原始时序信号到语言语义空间的稳定过渡，同时避免全量微调LLM带来的高成本。

---

### 相比现有方法的优势
| 维度 | CLINPRISM优势 |
|------|----------------|
| **效率** | 仅用 **16个temporal tokens** 和 **4B参数LLM** 即可完成推理，显著降低计算开销。 |
| **精度** | 在多个临床推理任务上达到SOTA性能，尤其在稀疏观测下鲁棒性强。 |
| **通用性** | 支持多种类型的自然语言QA任务（理解、预测、推理、决策）。 |
| **兼容性** | 不依赖文本序列化，避免数值精度损失；接口紧凑，易于集成到现有LLM系统中。 |

---

## 2. 核心实验方法和设置

### 数据集
所有实验均基于从 **MIMIC-IV ICU** 数据集中构建的三大资源：
1. **不规则时间序列语料库**（30,000条ICU住院轨迹）  
   - 包含11个生命体征变量（如heart rate, map, spo2等），保留原始时间戳和缺失模式。
2. **分层caption语料库**（30,000条）  
   - 每条轨迹配有macro/meso/micro三级自然语言描述，由GPT-5.5生成并经过自动与人工验证。
3. **多任务QA语料库**（约41,000道四选一选择题）  
   - 覆盖4类能力、11种任务类型（见下表），答案由程序确定后经GPT-5.5重写提升多样性。

| 能力类别 | 任务类型 |
|----------|---------|
| Temporal Understanding | Temporal Grounding (TG), Anchor-State Retrieval (ASR), Trend Pattern Recognition (TPR), Missingness Awareness (MA), Clinical Time-Series Summarization (TSS) |
| Temporal Forecasting | Threshold Forecasting (TF), Next-Value Interval Forecasting (NIF) |
| Temporal Reasoning | Cross-Variable Reasoning (CVR), Intervention Response (IR) |
| Temporal Decision-Making | Immediate Intervention Decision (IID), Monitoring/Escalation Decision (MED) |

最终评估在独立保留集 **CLIR-Bench** 上进行，确保无患者、住院或ICU停留重叠。

---

### 实验设置与评估指标
- **主干模型**：基于 `Qwen3-4B` 构建，冻结原始权重，仅引入轻量级组件。
- **评估指标**：所有任务均为多项选择题，报告 **accuracy**。
- **推理延迟**：在NVIDIA RTX 4090 GPU上批量测试（batch size=32），记录平均每问耗时。
- **token消耗**：统计输入中用于表示时间序列的`temporal tokens`数量（固定为16）。

---

### 基线方法对比
涵盖四类主流方法：
1. **闭源LLM**：Gemini-2.5-flash, GPT-5.4 mini
2. **开源通用LLM**：DeepSeek-V4-flash, KiMi-2.6, Gemma系列, Qwen系列
3. **时间序列LLM**：Time-LLM, ChatTS, ITFormer, TS-Reasoner, AutoTime
4. **不规则时间序列LLM基线**：t-PatchGNN + Qwen3-4B（相同训练流程但无caption对齐）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **CLINPRISM在CLIR-Bench上的总体准确率为 `49.83%`**，是当前所有**开源系统中的最高成绩**。
- 显著优于最强的时间序列基线 **t-PatchGNN**（+11.18个百分点）。
- 与闭源强基线相比：
  - 仅落后于 **GPT-5.4 mini** 0.32个百分点（50.15% vs 49.83%）
  - 超越 **Gemini-2.5-flash** 达9.53个百分点（40.30%）

> 特别是在以下任务上表现突出：
> - **Missingness Awareness (MA)**: 87.00%
> - **Clinical Time-Series Summarization (TSS)**: 57.83%
> - **Immediate Intervention Decision (IID)**: 96.67%

表明模型能有效识别稀疏观测、整合多尺度信息并做出合理临床判断。

---

### 与基线方法的对比结果（摘要）
| 模型 | 参数量 | 总体准确率 |
|------|--------|------------|
| GPT-5.4 mini | — | 50.15% |
| **CLINPRISM** | **4B** | **49.83%** ✅（开源最佳） |
| t-PatchGNN + Qwen3-4B | 4B | 38.65% |
| KiMi-2.6 | 1T | 49.09% |
| Qwen3.6-27B | 27B | 25.01% |

> CLINPRISM以极小的参数规模（4B）超越多数更大模型，展现出卓越的性价比。

---

### 消融实验结果（Ablation Study）

#### （1）多尺度编码器的影响
| 变体 | 平均准确率 |
|------|-----------|
| Only Macro | 46.97% |
| Macro + Meso | 48.24% |
| Full (Macro + Meso + Micro) | **49.83%** |

✅ 表明**meso和micro尺度提供了互补的有效信息**。

#### （2）渐进式对齐策略的作用
| 移除部分 | 平均准确率 | 下降幅度 |
|--------|------------|----------|
| Without Stage 2（无caption对齐） | 48.27% | -1.56 pts |
| Without Stage 3 Step 1 | 47.55% | -2.28 pts |
| Without Stage 3 Step 2（无LoRA联合优化） | **36.35%** | **-13.48 pts** ❗️最大下降 |

➡️ 验证了**joint LoRA-distiller adaptation**的关键作用。

#### （3）temporal token数量敏感性分析（K = |T|）
| K | 平均准确率 |
|----|-----------|
| 8 | 46.02% |
| **16** | **49.83%** ✅ |
| 32 | 47.79% |
| 64 | 47.59% |
| 128 | 47.45% |

➡️ 准确率**并非随token数单调上升**，**K=16达到最优平衡**，说明过度压缩或扩展都会损害性能。

---

## 4. 关键结论和发现

### 主要发现
1. **多尺度建模对临床QA至关重要**：宏观背景、中观事件、微观状态共同构成完整证据链。
2. **渐进式对齐优于端到端训练**：通过分阶段优化，可在不微调LLM的情况下实现高质量时序-语言融合。
3. **少量temporal tokens即可实现高效推理**：16个token足以承载关键临床信息，极大降低部署成本。
4. **CLINPRISM在稀疏数据下更鲁棒**：相对性能波动仅为1.6%，远低于t-PatchGNN的12.7%。

---

### 方法局限性
1. 当前仅支持**封闭式问答**（multiple-choice），尚未扩展至开放式生成。
2. caption生成依赖GPT-5.5，存在潜在偏差风险，需进一步控制生成质量。
3. 所有任务基于回顾性数据，未考虑实时流式输入场景。
4. 模型未显式建模不确定性，在低置信度情况下可能给出错误自信的回答。

---

### 未来工作方向
1. 扩展至**开放域问答**与**不确定性感知推理**。
2. 探索在其他医疗领域（如影像、病理报告）中的泛化能力。
3. 结合外部医学知识图谱增强推理逻辑。
4. 开展前瞻性临床验证研究，评估实际辅助诊疗潜力。

> “Future work will extend the framework to open-ended questions, uncertainty-aware reasoning, and evaluation across broader clinical settings.” —— 原文结尾

</details>

---

### 15. [Neuromorphic Diffusion Language Models: Addressing Compute and Memory Bottlenecks via Sparsity and Block Denoising](https://arxiv.org/abs/2607.24841)

**Authors**: Dengyu Wu, Clement Ruah, Jiechen Chen, Bipin Rajendran, Osvaldo Simeone  
**Category**: cs.CL  
**Published**: 2026-07-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.24841v1  

#### Abstract
Autoregressive (AR) large language models (LLMs) are inherently inefficient at inference time because each generated token requires accessing the full set of model parameters, leading to low operational intensity and high energy consumption. Masked diffusion language models (MDLMs) partially address...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Neuromorphic Diffusion Language Models: Addressing Compute and Memory Bottlenecks via Sparsity and Block Denoising*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **Autoregressive LLMs (AR-LLMs)** 在推理时存在严重的效率瓶颈：
- 每生成一个 token 都需要访问全部模型参数，导致 **operational intensity**（计算密度）低、内存带宽压力大、能耗高。
- 即使是改进的 **Masked Diffusion Language Models (MDLMs)** 虽然通过并行去噪提升吞吐量，但在现代具备片上内存（in-chip memory）的硬件平台上仍受限于计算瓶颈，难以进一步提升能效。

### 提出的新方法与新思路
本文提出 **Neuromorphic Masked Diffusion Language Models (N-MDLMs)**，将两种前沿技术结合：
- **Block Diffusion**：在每个去噪步骤中并行生成多个 token，减少参数访问频率。
- **Spiking Neural Networks (SNNs) + Neuromorphic Computing**：采用基于 **Integrate-and-Fire (IF) 神经元模型** 的事件驱动机制，实现动态稀疏性（spike-induced sparsity），仅在有 spike 时才触发计算和内存访问。

该方法通过量化转换（quantization-based conversion）从预训练的 MDLM 构建 N-MDLM，并引入超参数 $ K $ 控制激活稀疏度。

### 相比现有方法的优势
| 方法 | 优势 |
|------|------|
| **vs. AR-LLM** | 显著降低每 token 的内存传输和计算开销，提升吞吐量与能效 |
| **vs. MDLM** | 在 compute-bound 平台上依然有效，而传统 MDLM 在此类平台无明显增益 |
| **vs. N-AR-LLM** | 结合 block diffusion 与 sparsity，在内存和计算两方面同时优化 |

> ✅ **核心创新**：首次将 neuromorphic computing 引入 diffusion-based LLM 推理，形成 **N-MDLM** 新架构，实现 **block-parallel generation** 与 **event-driven sparsity** 的协同增益。

---

## 2. 核心实验方法和设置

### 数据集
- **WMT 14 DE-EN**：标准机器翻译任务数据集，用于评估模型性能。

### 模型架构
- 基于 **E2D2 (Efficient Encoder-Decoder Diffusion)** 架构：
  - 编码器层数 $ N_{\text{enc}} = 28 $
  - 解码器层数 $ N_{\text{dec}} = 4 $
  - Embedding 维度 $ D_{\text{emb}} = 512 $
  - 中间隐藏维度 $ D_{\text{hid}} = 1536 $
  - 总参数约 **250M**

### 实验设置
- **硬件模拟配置**：
  - 使用 **NVIDIA DGX Spark GPU** 进行仿真
  - 采用 **bitwise coding** 模拟 T=8 时间步的 spike 序列
- **转换流程**：
  - 从预训练 MDLM 出发 → 应用量化 + IF 神经元模型转换 → 微调（fine-tune）
    - 优化器：Adam
    - 学习率：$ 3 \times 10^{-5} $
    - Batch size：16
    - 步数：1000
- **关键变量控制**：
  - Block size $ B \in \{2, 4, 8\} $
  - 稀疏性超参数 $ K \in \{1, 2, 3\} $，越小表示稀疏性越高
  - 设置 $ S = B $：每步解码 unmask 一个 token

### 评估指标
| 指标 | 定义 |
|------|------|
| **Token Throughput (token/s)** | 每秒可处理的 token 数量 |
| **Energy per Token (J/token)** | 单个 token 推理消耗的能量 |
| **BLEU Score** | 翻译任务准确性衡量 |
| **Sparsity Level $ \alpha $** | 输入矩阵中零元素的概率，反映 spike 稀疏程度 |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **AR-LLM** | 标准自回归模型，串行生成 |
| **MDLM** | 块扩散模型，支持并行去噪 |
| **N-AR-LLM** | 应用相同神经形态转换的自回归模型 |
| **N-MDLM** | 本文提出的神经形态扩散模型（主方法） |

此外区分运行环境：
- **Off-Chip Memory System (OCMS)**：如传统 GPU，内存带宽受限
- **In-Chip Memory System (ICMS)**：代表先进加速器（如 Loihi、NorthPole），接近 memory-bound 极限

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自图3–图5）

#### 📈 图3：归一化吞吐量 vs. Block Size $ B $
- **OCMS 上**：
  - MDLM 吞吐随 $ B $ 增加显著上升 → 受益于内存访问摊销
  - N-MDLM 表现更优，但提升空间有限（已接近最优）
- **ICMS 上（compute-bound）**：
  - MDLM 几乎没有增益（因计算成为瓶颈）
  - **N-MDLM 吞吐持续增长**，尤其当 $ K=1 $（最高稀疏性）时表现最佳
  - 最终趋于饱和 → 系统进入 compute-bound 极限

> 🔍 发现：**sparsity 是打破 compute-bound 瓶颈的关键**，使得即使在 ICMS 上也能获得收益。

#### ⚡ 图4：吞吐 vs. 能耗（Energy per Token）
- 所有模型中，**N-MDLM 实现最高吞吐 + 最低能耗**
- 随着 $ B $ 增大，能耗略有上升（更多并行操作），但吞吐提升更大 → 整体性价比更高
- $ K=1 $ 时能耗最低 → spike-induced sparsity 显著减少无效运算与访存

> 💡 能效来源：**memory access 占总能耗主导地位**，N-MDLM 成功减少了权重加载次数。

#### 🎯 图5：归一化的 BLEU、吞吐、能耗综合比较（$ B=4 $）
| 模型 | 吞吐 | 能耗 | BLEU |
|------|------|------|------|
| AR-LLM | 1.0 | 1.0 | 1.0 |
| MDLM | ~1.2 | ~0.8 | ~0.95 |
| **N-MDLM ($ K=1 $)** | **~1.8** | **~0.6** | **~0.93** |

> ✅ 结论：**N-MDLM 在保持几乎相同翻译质量的前提下，吞吐提升 80%，能耗降低 40%**

### 消融实验分析（隐含在不同 $ K $ 和 $ B $ 对比中）
- **稀疏性影响（$ K $）**：
  - $ K=1 $ 比 $ K=3 $ 更稀疏 → 更少 spike → 更低计算负载和内存流量
  - 在 ICMS 上，高稀疏性显著缓解 compute-bound 限制
- **块大小影响（$ B $）**：
  - 小 $ B $：不足以摊销内存成本
  - 大 $ B $：提升吞吐，但可能加剧 compute pressure
  - 最佳平衡点出现在 $ B=4, K=1/2 $

> 🔬 发现：**block size 与 sparsity 存在协同效应**——适当增大 $ B $ 可推动系统向 memory-bound 转移，从而让 sparsity 更有效地发挥作用。

---

## 4. 关键结论和发现

### 主要发现
1. **N-MDLM 实现了 block diffusion 与 neuromorphic sparsity 的协同优化**：
   - Block diffusion 提升 operational intensity（每参数访问生成更多 token）
   - Spike-induced sparsity 动态跳过非活跃通道，减少实际计算与内存访问
2. **在 compute-bound 平台（如 ICMS）上，传统 MDLM 无效，但 N-MDLM 仍能大幅提升性能**：
   - 因为 sparsity 降低了有效计算量，使系统重新偏向 memory-bound，释放 diffusion 的潜力
3. **能量效率主要由 memory access 决定**：
   - N-MDLM 通过稀疏性大幅减少权重读取，成为节能主因
4. **精度损失极小**：
   - BLEU score 仅轻微下降（<5%），说明转换与稀疏化对语义理解影响可控

### 方法的局限性
- 当前实验基于 **GPU 上的 spike 仿真**，尚未部署到真实 neuromorphic hardware（如 Loihi 或 SpiNNaker）
- 转换过程依赖 fine-tuning，可能增加额外训练成本
- 对 $ K $ 和 $ B $ 的选择敏感，需根据硬件特性进行调优
- 目前仅验证于翻译任务，通用性有待在其他 NLP 任务中验证

### 未来工作方向
1. **在真实 neuromorphic hardware 上部署与验证 N-MDLM**
2. **开发自适应策略联合优化 $ B $ 和 $ K $**，以应对不同输入长度、系统资源约束
3. **扩展至 causal LM 或 decoder-only 架构**，探索在生成式 AI 中的应用
4. **研究更高效的 spike 编码方案**（如 temporal coding 优化），进一步压缩延迟与能耗

---

> ✅ **总体评价**：  
> 本论文提出了一个极具前瞻性的 **N-MDLM** 框架，成功将 **diffusion modeling** 与 **neuromorphic computing** 融合，不仅解决了当前 LLM 推理中的 compute/memory 瓶颈问题，还为下一代高效 AI 系统设计提供了新范式。其实验充分、分析深入，展示了在先进硬件平台上的巨大潜力。

</details>

---

### 16. [QFedPolyp: A Communication- and Inference-Efficient Federated Learning Framework for Polyp Segmentation](https://arxiv.org/abs/2607.22743)

**Authors**: Madan Baduwal, Priyanka Paudel  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.22743v1  

#### Abstract
Background and Objective: Automatic polyp segmentation supports computer-aided diagnosis and early colorectal cancer detec- tion. Centralized deep learning requires hospitals to share sensitive medical data, while federated learning preserves privacy but introduces high communication costs through r...

---

### 17. [A Coulomb Particle Model for Learning Kernel Attention in Transformers](https://arxiv.org/abs/2607.23869)

**Authors**: Masoud Badiei Khuzani, Sharath Honnaiah, Atiq Islam, Alex Cozzi, Abraham Bagherjeiran  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.23869v1  

#### Abstract
Randomized features provide a scalable approximation to kernel machines, but their performance depends strongly on the choice of feature distribution. We propose a particle-based method that learns this distribution by optimizing kernel-target alignment while regularizing particles with a Riesz/Coul...

---

### 18. [Every Client Is an Environment: Federated De-confounding for Spatio-Temporal Forecasting](https://arxiv.org/abs/2607.24218)

**Authors**: Qingxiang Liu, Anqi Liang, Heng Wang, Yuxuan Liang  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.24218v1  

#### Abstract
Federated learning has emerged as a promising paradigm for spatio-temporal forecasting (STF), enabling collaborative model training without sharing raw observations. Existing federated STF methods primarily regard cross-client heterogeneity as an optimization challenge and mitigate it through person...

---

### 19. [Penelope: Localized Latent Recurrence for Efficient Structured Reasoning](https://arxiv.org/abs/2607.25915)

**Authors**: Yutong Chen, Shouqian Shi, Xinran Liu, Haochen Wang, Jiaying Wang, Tianxing Xu, Yuanxi Wang, Zirui Ding  
**Category**: cs.AI  
**Published**: 2026-07-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.25915v1  

#### Abstract
Complex structured reasoning tasks often require additional computation, yet current language models obtain it mainly by increasing parameter scale or by serializing intermediate steps as chain-of-thought (CoT) tokens. The former raises training and deployment costs, while the latter ties reasoning ...

---

### 20. [PowerScale: Energy-Efficient Geo-Distributed Model Training with Federated Datacenter Power](https://arxiv.org/abs/2607.25650)

**Authors**: Talha Mehboob, Zhe Xu, Michael Zink, David Irwin  
**Category**: cs.DC  
**Published**: 2026-07-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.25650v1  

#### Abstract
The power demands of large-scale AI training increasingly exceed the capacity of any single data center, making geo-distributed training across power-constrained sites a practical necessity. Prior work optimizes such training mainly for time-to-accuracy using single-tier aggregation, where every sit...

---

### 21. [Sparse Gaussian-Mixture-Model Q-Functions via Hadamard Overparametrization for Online Reinforcement Learning](https://arxiv.org/abs/2607.23474)

**Authors**: Minh Vu, Konstantinos Slavakis  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.23474v1  

#### Abstract
This paper develops an online, off-policy policy-iteration framework for reinforcement learning (RL), based on sparse Gaussian-mixture-model Q-functions (S-GMM-QFs). The framework reconciles streaming, non-stationary data with the Riemannian structure of the parameter space while handling distributi...

---

### 22. [ADVERSARIAL: And-Inverter Graph-Assisted Hardware Trojan Detection At Scale](https://arxiv.org/abs/2607.23882)

**Authors**: Yaroslav Popryho, Debjit Pal, Inna Partin-Vaisband  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.23882v1  

#### Abstract
Modern System-on-Chip (SoCs) often contain hundreds of millions to tens of billions of gates, making existing Hardware Trojan (HT) detection methods impractical due to their immense scale. The proposed approach incorporates symbolically enabled learning by modeling flattened gate-level netlists as B...

---

### 23. [CADENCE: A Cardiac Atom Dictionary for Interpretable Neural Concept Extraction from ECG Foundation Models](https://arxiv.org/abs/2607.25244)

**Authors**: Yixuan Duan, Arjun Naik, Sadeer Al-Kindi, Wei Qiu  
**Category**: cs.AI  
**Published**: 2026-07-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.25244v1  

#### Abstract
Foundation models for 12-lead electrocardiograms (ECGs) transfer well across clinical tasks, but the physiological knowledge encoded in their representations remains opaque. We present CADENCE, a framework that decomposes an ECG foundation model into a human-interpretable, queryable dictionary of ph...

---

### 24. [Salient Knowledge Pathways: Sparse Cross-Modal Routing for Efficient Knowledge-Intensive Multimodal Question Answering](https://arxiv.org/abs/2607.25422)

**Authors**: Noor Islam S. Mohammad, Ulu\u{g} Bayaz{\i}t  
**Category**: cs.AI  
**Published**: 2026-07-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.25422v1  

#### Abstract
Knowledge-intensive multimodal question answering (KI-MMQA) sits at the intersection of three expensive primitives: long visual token sequences, dense retrieval over large external corpora, and full cross-modal fusion. Existing systems pay all three costs uniformly per query, even though only a smal...

---

### 25. [Distributed Constraint Optimization via Online Learning and Iterative Pricing with Application to Large-Scale Satellite Scheduling](https://arxiv.org/abs/2607.25835)

**Authors**: Itai Zilberstein, Pranav Rajbhandari, Steve Chien, Tuomas Sandholm  
**Category**: cs.AI  
**Published**: 2026-07-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.25835v1  

#### Abstract
Distributed constraint optimization problems (DCOPs) provide a popular framework for distributed decision making under limited communication, but many real-world instances are too large to solve monolithically. We address this challenge from two complementary directions. We revisit the connection be...

---

### 26. [UniMem: Complementary Episodic-to-Parametric Memory for Boundary-Agnostic Task Streams](https://arxiv.org/abs/2607.26017)

**Authors**: Siyu Xia, Chenheng Zhang, Yanting Wu, Haoxuan Li, Jiajun Chai, Xiaohan Wang, Guojun Yin, Wei Lin, Zhouchen Lin, Haifeng Zhang, Jun Wang  
**Category**: cs.CL  
**Published**: 2026-07-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.26017v1  

#### Abstract
Memory is essential for LLM agents to accumulate task experience and reuse task-specific execution strategies. However, real-world deployment over boundary-agnostic and evolving task streams exposes a fundamental stability-plasticity dilemma. External retrieval-based memory can rapidly absorb new ev...

---

### 27. [Hierarchical Grading in Large Language Models](https://arxiv.org/abs/2607.22757)

**Authors**: T. Shaska  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.22757v1  

#### Abstract
We introduce Graded Large Language Models (GLLMs), an algebraic framework that equips the representation space of a transformer with a grading and propagates the induced weighted scalar action through embeddings, self-attention, and the training objective. The construction extends the theory of grad...

---

### 28. [DynaCalKV: Key-Value Cache Compression via Head Grouping and Adaptive Rank Allocation](https://arxiv.org/abs/2607.24331)

**Authors**: Tan T. Nguyen, Quan V. Dang  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.24331v1  

#### Abstract
As the inference phase of Large Language Models (LLMs) requires handling long context windows, the Key-Value (KV) cache initially appears to address this challenge but eventually becomes a significant bottleneck as the context window continues to grow. Low-rank compression has recently been studied ...

---

### 29. [FlowCTS: On-policy Continuous Trajectory Supervision of Flow Models](https://arxiv.org/abs/2607.24522)

**Authors**: Kaiyang Ye, Yuan Ge, Junxiang Zhang, Bei Li, Ziming Zhu, Haishu Zhao, Xiaoqian Liu, Chenglong Wang, Jingbo Zhu, Zhengtao Yu, Tong Xiao  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.24522v1  

#### Abstract
While on-policy distillation (OPD) effectively addresses sparse rewards and exposure bias in large language model post-training, its extension to flow models remains underexplored. To this end, we propose Flow Continuous Trajectory Supervision (FlowCTS), which matches subsequent student and referenc...

---

### 30. [LOCKS: Page-Local Compact Key Summaries for Efficient Long-Context Decoding](https://arxiv.org/abs/2607.24555)

**Authors**: Junsung Hwang  
**Category**: cs.LG  
**Published**: 2026-07-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.24555v1  

#### Abstract
Serving large language models at long context is bottlenecked by the key-value (KV) cache, which is read in full at every decode step. Attention keys are locally low-rank though globally high-rank: shared low-rank bases discard page-specific directions that a page's own compact basis retains. LOCKS ...

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

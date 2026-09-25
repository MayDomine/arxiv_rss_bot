# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-25 10:35:45 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Task-Aware Spectral Pruning: A Mixture-of-Masks Framework for Efficient LLM Inference](https://arxiv.org/abs/2609.29499)

**Authors**: Ibne Farabi Shihab, Fariya Afrin, Sanjeda Akter, Anuj Sharma  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2609.29499v1  

#### Abstract
Static pruning imposes one sparse structure on every prompt, even though reasoning, retrieval, generation, coding, and translation can depend on different parts of a language model. We introduce Task-Aware Spectral Pruning (TASP), a post-training framework that calibrates module-level spectral descr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Task-Aware Spectral Pruning: A Mixture-of-Masks Framework for Efficient LLM Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的静态剪枝（static pruning）方法对所有输入提示（prompt）施加**单一稀疏结构**，而不同任务（如推理、检索、生成、编码、翻译）依赖于语言模型的不同模块。这种“一刀切”的策略导致了所谓的 **versatility tax（通用性税）**——即一个为数学推理优化的掩码可能损害其在流畅生成上的表现。

此外，许多剪枝方法虽然能减少参数量，但由于未考虑硬件兼容性和执行一致性（如 KV Cache 有效性），难以转化为实际的推理加速。

### 提出的新方法：TASP（Task-Aware Spectral Pruning）
TASP 是一种**后训练（post-training）框架**，通过以下机制实现高效且任务感知的 LLM 推理：

- **任务感知的谱分析（Task-Aware Spectral Analysis）**  
  利用模块级权重矩阵的**谱描述符**（spectral descriptors，如尾指数、有效秩、谱范数等）结合轻量级激活敏感度统计（salience），预测每个模块在特定任务下的移除影响。

- **可拒绝的适用性门控（Applicability Gate）**  
  引入一个前置的“试点”阶段（pilot），通过交叉验证判断谱信号是否具有足够的任务区分能力。若不满足阈值，则直接放弃剪枝，避免无效计算开销。

- **依赖闭合的掩码构建（Dependency-Closed Mask Construction）**  
  在构建稀疏掩码时显式处理架构依赖关系：
  - **Grouped-Query Attention (GQA)**：只要任一查询头保留，对应的 KV 头也必须保留。
  - **SwiGLU Feed-Forward**：`W_gate`, `W_up`, `W_down` 的通道组需联合保留或删除。

- **一次路由、全程固定（Route Once, Execute Consistently）**  
  每个用户对话轮次（user turn）仅进行一次任务分类和掩码选择，该掩码在整个 prefill 和 autoregressive decoding 阶段保持不变，确保 **KV Cache 形状一致且可复用**。

- **混合掩码执行（Mixture-of-Masks Execution）**  
  构建多个任务专用的静态稀疏子网络（sparse subnetworks），共享同一个 INT8 权重存储，运行时根据任务动态调度。

### 相比现有方法的优势
| 维度 | TASP | 典型基线（如 SparseGPT, SliceGPT, ShortGPT） |
|------|------|---------------------------------------------|
| **任务适应性** | ✅ 多任务掩码，按需切换 | ❌ 单一全局掩码 |
| **硬件有效性** | ✅ 编译为静态 Triton kernel，支持实际加速 | ⚠️ 多为分析性 FLOP 减少，未必可执行 |
| **缓存一致性** | ✅ 同一轮次内掩码不变，KV Cache 可复用 | ⚠️ 动态剪枝可能导致 cache invalidation |
| **适用性控制** | ✅ 内置 pilot gate，防止不适用模型被错误剪枝 | ❌ 无此机制，盲目应用 |
| **部署灵活性** | ✅ 支持 confidence-based fallback 到 dense 路径 | ❌ 通常无 fallback 机制 |

---

## 2. 核心实验方法和设置

### 使用的数据集
涵盖五大类共 11 个基准任务：
- **Reasoning**: MMLU, GSM8K, MGSM
- **Generation**: AlpacaEval, MT-Bench
- **Retrieval**: NQ-Open, TriviaQA
- **Code**: HumanEval, MBPP
- **Translation**: WMT14 En-De, En-Fr

### 实验设置与评估指标

#### 模型
- 主要测试模型：**Llama-3-8B**, **Llama-3-70B**
- 对照模型：**Qwen2.5-1.5B**

#### 剪枝目标
- **Active FLOP 减少 43%**（即保留 57% 的 active FLOPs）

#### 评估环境
- 硬件：单张 **A100 80GB SXM**
- 量化配置：**INT8-weight / BF16-compute**
- 序列长度：2048-token prefill + 256-token decode
- 批大小：1

#### 评估指标
| 指标 | 描述 |
|------|------|
| `Ret(m; r)` | 宏平均相对保留率：<br>$ \frac{1}{|B|} \sum_{b \in B} \frac{\text{score}_{m,b}}{\text{score}_{r,b}} $ |
| Decode Latency | 每 token 解码延迟（ms/token） |
| Speedup | 相对于 dense INT8 的解码速度提升倍数 |
| AUROC | Pilot 阶段谱特征对任务敏感性的区分能力 |

#### 基线方法对比
- **Unstructured Pruning**: Magnitude, SparseGPT, Wanda
- **Structured Pruning**: LLM-Pruner, SliceGPT, ShortGPT, AlphaPruning, LLaMaFlex
- **Task-Aware Baselines**: TASP-Single（单全局掩码）、TASP-Oracle（理想路由）
- **近期对比方法**：PuDDing, IG-Pruning, Instruction-Following Pruning, TaBP, ShadowLLM

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Llama-3-70B）

#### 科学基准（BF16 Harness）下的质量保留
| 方法 | Full Retention |
|------|----------------|
| Dense BF16 | 100.0% |
| TASP-MoM | **97.7±0.2%** |
| TASP-Single | 93.3±0.3% |
| TASP-Oracle | 98.7±0.2% |
| 最强基线（LLaMaFlex） | 94.0±0.3% |

> ✅ TASP-MoM 比最佳基线高出 **3.7个百分点**

#### 部署匹配运行时（INT8/BF16）的实际加速
| 方法 | Decode Latency (ms/token) | Speedup | Retention vs. BF16 |
|------|----------------------------|---------|--------------------|
| Dense INT8 | 45.2±0.4 | 1.00× | 99.6±0.1% |
| ShortGPT (compiled) | 27.8±0.3 | **1.63×** | 91.5±0.4% |
| TASP-MoM (sparse path) | **31.3±0.4** | **1.44×** | **97.3±0.2%** |

> ✅ TASP 实现了 **质量-延迟帕累托改进**：虽略慢于 ShortGPT，但保留了多 **5.8个百分点** 的能力。

#### 混合策略下的综合表现（T=0.85）
- **76% 请求走稀疏路径**
- **综合延迟降至 34.6 ms/token**
- **综合加速 1.30×**
- **保留 99.3±0.1% 相对于 dense INT8**

---

### 消融实验结果（Ablation Study）

| 变体 | Retention vs. BF16 | 相对损失 |
|------|---------------------|--------|
| TASP-Oracle（理想路由） | 98.7±0.2% | — |
| TASP-MoM（学习路由） | 97.7±0.2% | -1.0 pt |
| 去除 salience 特征 | 95.5±0.3% | -2.2 pt |
| 去除 spectral 特征 | 93.6±0.4% | -4.1 pt |
| 使用单个全局掩码 | 93.3±0.3% | -4.4 pt |
| 使用随机掩码 | 85.6±0.5% | -12.1 pt |

> 🔍 结论：**谱特征 + 多掩码设计是互补增益**，缺一不可；单纯增加掩码数量无法替代有监督排序。

---

## 4. 关键结论和发现

### 主要发现
1. **谱信号的任务可分性是关键前提**  
   - Llama-3 系列（8B/70B）表现出显著的谱-任务分离性（AUROC > 0.66），可通过剪枝获益。
   - Qwen2.5-1.5B 表现接近随机水平（AUROC ≈ 0.53），不适合 TASP。
   - ✅ 提出的 **applicability gate** 成功识别出不适用模型。

2. **任务条件化掩码显著优于全局掩码**  
   - 尤其在 **GSM8K** 和 **HumanEval** 上差距最大，说明结构化生成任务需要专门保留关键模块。

3. **编译后的稀疏路径确实带来真实加速**  
   - 在 A100 上实现 **1.44× 解码加速**，证明从分析性 FLOP 减少到实际性能提升的闭环。

4. **mask 多样性 ≠ 任意多样性**  
   - 随机掩码导致严重性能下降（-12.1pt），说明必须基于任务敏感性进行排序。

5. **路由稳定性与 fallback 至关重要**  
   - 当 top-1 路由准确率从 100% 下降到 70%，配合 fallback 仍能维持 97.5%+ 保留率。

---

### 局限性（Limitations）
- **高离线成本**：Llama-3-70B 的完整流程耗时 **136 A100 GPU-hours**，远高于其他 post-training 方法（7–20 小时）。
- **任务定义需稳定**：要求任务类别长期稳定以摊销校准成本，不适合频繁变化的任务场景。
- **架构限制**：目前仅支持 decoder-only Transformer with GQA 和 SwiGLU；不适用于 encoder-decoder、MoE、SSM 或多模态架构。
- **泛化风险**：路由器在跨领域或语言迁移下可能出现过度自信，需依赖 fallback 保障安全。
- **平台依赖性**：实测延迟针对特定硬件（A100）、量化方案（INT8/BF16）和编译器（Triton），不能直接外推。

---

### 未来工作方向
1. **前瞻性验证适用性规则**  
   在更广泛的模型家族上验证 applicability gate 的普适性。
2. **探索因果解释机制**  
   结合 activation patching 或 causal mediation 分析，理解为何某些模块对特定任务敏感。
3. **编译器感知的掩码选择**  
   在 Rashomon set（多个等效高性能掩码）中选择更适合目标设备 tile shape 的版本。
4. **与其他压缩技术结合**  
   探索 TASP 与量化（quantization）、低秩近似（low-rank approximation）的联合优化。
5. **降低校准成本**  
   设计更高效的 label sampling 或 transfer learning 策略，减少对全量 ablation 的依赖。

---

> 📌 **总体结论**：  
> TASP 成功将 **spectral pruning** 重构为一个**部署决策流程**：先检验谱-任务可分性，再构建依赖闭合的硬件有效掩码，最后评估质量、延迟与摊销权衡。它不是简单的剪枝算法改进，而是提出了一套完整的 **“可剪枝性判定 → 任务感知压缩 → 编译执行 → 回退保障”** 的工业化 LLM 推理优化范式。

</details>

---

### 2. [FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates](https://arxiv.org/abs/2609.29812)

**Authors**: Wanqi Yang, Shiwei Liu  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2609.29812v1  

#### Abstract
Looped Transformers have attracted substantial attention as a parameter-efficient approach to increasing computational depth through repeated application of shared Transformer blocks. However, their practical advantages over conventional Transformers remain under debate: each additional loop incurs ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates*

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

**Looped Transformers**（循环式Transformer）通过重复应用共享的Transformer模块来增加计算深度，从而在不增加参数量的前提下提升模型能力，具有良好的**参数效率**（parameter efficiency）。然而，其推理开销巨大：

- 每一轮循环都需要重新执行所有Transformer层（导致FLOPs线性增长）；
- 每轮需缓存独立的 **KV Cache**，导致显存占用随循环次数线性膨胀；
- 尤其在长上下文（如32K tokens）和多轮循环下，推理延迟和内存需求远超同等性能的非循环模型。

这使得其“参数高效”难以转化为“推理高效”。

---

### ✅ **提出了什么新方法或新思路**

作者提出 **FlashLoop** —— 一种无需训练、即插即用的推理优化框架，基于对Looped Transformers中跨循环冗余性的系统观察，引入三项关键技术：

#### 🔹 **Token-Sparse Updates（令牌稀疏更新）**
- 观察到随着循环进行，只有少数token的状态发生显著变化（lazy updates）。
- 因此仅对“活跃token”进行前向计算并更新其KV状态，其余token复用之前的结果。
- 显著减少计算量和KV Cache写入。

#### 🔹 **Loop-aware Sparse Attention（循环感知稀疏注意力）**
- 注意力输出的变化集中在少数关键的 **key columns** 上，且这些列的重要性在相邻循环间高度稳定。
- 利用上一轮的注意力分布选择重要列，在当前轮只重算这些列的贡献。
- 引入 **global probability mass caching** 来保持原始softmax尺度，避免概率归一化偏差。

#### 🔹 **KV-Residual Quantization（KV残差量化）**
- 相邻循环间的KV状态差异（residuals）比完整KV更小、分布更集中，更适合低比特量化。
- 存储一个量化后的基础KV状态，并以低比特存储后续循环的残差更新。
- 大幅压缩KV Cache体积，尤其适用于深层循环。

> 所有组件均在GPU上实现硬件友好的融合内核（fused kernels），避免稀疏操作带来的性能损失。

---

### ✅ **相比现有方法的优势**

| 维度 | FlashLoop | 现有通用优化方法（如KV量化、Token压缩等） |
|------|-----------|-------------------------------|
| **针对性** | 针对Looped Transformers特有的跨循环冗余设计 | 主要面向非循环模型，未利用循环结构特性 |
| **是否需要训练** | ❌ 不需要任何微调或再训练（training-free） | 部分方法需额外训练（如Looped Latent Attention） |
| **兼容性** | 可直接部署于已训练好的循环模型 | 往往依赖特定架构修改 |
| **综合收益** | 同时降低FLOPs、KV内存、解码延迟 | 通常只能优化单一维度 |

---

## 2. 核心实验方法和设置

### 📚 **使用的模型与数据集**

#### ✅ **测试模型（均为Looped Transformers）**
- **Ouro系列**：Ouro-1.4B, Ouro-2.6B（含标准版与thinking版）
- **Huginn-3.5B**：支持多达32次循环

#### ✅ **评估基准（Benchmark）**
使用 [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness) 测试以下五个任务：
- **MATH-500**：数学推理
- **GSM8K**：小学数学题
- **ARC-Challenge**：科学问答
- **HellaSwag**：常识推理
- **WinoGrande**：代词消解

此外还进行了：
- **Long-context 质量评估**：WikiText-2 上的困惑度（PPL）与 Needle-in-a-Haystack 任务

---

### ⚙️ **实验设置与评估指标**

| 类别 | 设置说明 |
|------|--------|
| **上下文长度** | 最高至 **32K tokens** |
| **循环次数（R）** | Ouro: R=4；Huginn: R=32 |
| **量化配置** | 使用 **INT4** 非对称分组量化（group size=64），RoPE后key按通道量化，value按token量化 |
| **稀疏策略** | 前1–2轮全量计算（warm-up），之后逐步稀疏化（见Table 5） |
| **评估指标** | <ul><li>✅ 准确率（Accuracy %）</li><li>✅ 推理端到端速度提升（Speedup ×）</li><li>✅ KV Cache 内存缩减倍数（KV Memory ↓×）</li><li>✅ Prefill / Decode 延迟</li></ul> |

---

### 🔁 **基线方法对比**

| 方法 | 描述 | 是否专用 |
|------|------|---------|
| **Original** | 原始Looped Transformer实现 | 是 |
| **H2O (Zhang et al., 2023)** | 保留attention得分最高的25% tokens | 否（通用KV压缩） |
| **Last-step KV Reuse (Zhu et al., 2025)** | 仅保留最后一轮KV用于后续生成 | 是（循环相关） |
| **Per-loop KIVI4** | 对每轮KV独立进行INT4量化 | 否（通用量化） |

---

## 3. 主要实验结果和性能指标

### 📊 **关键性能数据（来自Table 1）**

| Model | 方法 | 平均准确率 (Avg.) | △Avg. vs Original | Speedup | KV Memory ↓ |
|-------|------|------------------|--------------------|----------|--------------|
| Ouro-1.4B | Original | 70.11% | — | 1.00× | 1.00× |
|             | **FlashLoop** | **70.48%** | **+0.37pp** | **1.59×** | **5.85×↓** |
| Ouro-2.6B | Original | 71.21% | — | 1.00× | 1.00× |
|             | **FlashLoop** | **71.08%** | **-0.13pp** | **1.64×** | **6.06×↓** |
| Huginn-3.5B | Original | 42.60% | — | 1.00× | 1.00× |
|              | **FlashLoop** | **42.77%** | **+0.17pp** | **1.52×** | **5.18×↓** |

> 💡 结论：**几乎无损精度** 下，实现高达 **1.64× 速度提升** 和 **6× KV Cache压缩**

---

### 🔍 **消融实验结果（Ablation Study）**

#### ✅ **各组件对准确率的影响（Table 2）**

| 方法 | Avg. Accuracy |
|------|---------------|
| Original | 68.18% |
| + Loop-aware Sparse Attention | 68.91% ↑ |
| + Token-Sparse Updates | 68.63% |
| + Full KV Quantization (INT4) | 67.12% ↓ |
| **Full FlashLoop** | **68.56%** ≈ Original |

> 表明：**稀疏注意力轻微增益，残差量化优于全KV量化**

#### ✅ **各组件对效率的贡献（Figure 4）**
- **Token-Sparse Updates** → 显著降低Prefill阶段FLOPs和KV存储
- **Sparse Attention** → 显著降低Decode阶段计算负担
- **KV-Residual Quantization (4-bit)** → 进一步压缩KV Cache大小，降低带宽压力

---

### 🔎 **与其他KV压缩策略对比（Table 3）**

| 方法 | Ouro-1.4B Avg Acc | Ouro-2.6B Avg Acc |
|------|-------------------|-------------------|
| Original | 70.11% | 71.21% |
| H2O (-75% KV) | 62.81% ↓ | 67.35% ↓ |
| Last-step KV Reuse | 68.33% ↓ | 69.96% ↓ |
| **FlashLoop** | **70.48%** ✅ | **71.08%** ✅ |

> FlashLoop 在大幅压缩KV Cache的同时，**保持接近原模型的性能**，而其他方法出现明显下降。

---

### 📈 **扩展性分析（Scaling Behavior）**

#### ✅ **随上下文长度增长（Figure 6）**
- 在 **32K context** 下：
  - KV Cache 减少 **20 GiB**（降幅达 **82.8%**）
  - 解码延迟降低约 **37.5%**
  - 端到端速度提升 **~1.6×**

#### ✅ **随循环次数增加（Figure 7）**
- 原始模型：FLOPs 和 KV Cache 随循环线性增长
- FlashLoop：后期循环因稀疏化，增长显著放缓
- 特别是在 **Huginn-3.5B（32 loops）** 中优势更加明显

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **Looped Transformers存在强烈的跨循环冗余性**：
   - Token级更新趋于稀疏（lazy updates）
   - Attention column重要性稳定可预测
   - KV residuals比完整KV更易量化

2. **FlashLoop能有效挖掘上述冗余**：
   - 实现 **lossless or near-lossless inference**
   - 达成 **最高1.64×加速 + 6× KV Cache压缩**
   - 效果随context length和loop depth增强

3. **优于通用KV压缩方法**：
   - H2O、Last-step Reuse等会丢失中间循环信息，造成性能下降
   - FlashLoop选择性保留关键信息，兼顾效率与质量

---

### ⚠️ **局限性**

1. **依赖预定义稀疏调度表**（sparsity schedule）：
   - 当前为静态配置（per-model calibrate once），缺乏动态自适应机制
2. **对“thinking-style”模型稍敏感**：
   - 如Ouro-Thinking版本略有性能下降（但仍可控）
3. **目前仅支持BF16激活下的推理优化**
4. **尚未集成到主流推理引擎（如vLLM、TensorRT-LLM）**

---

### 🔮 **未来工作方向**

1. **动态稀疏控制机制**：根据输入内容自动调整token/column保留比例
2. **结合MoE架构**：将FlashLoop思想应用于循环式MoE模型
3. **支持更多量化格式**：如INT2、FP8，进一步压缩残差
4. **扩展至训练阶段**：探索如何在训练中鼓励更早收敛以增强lazy update特性
5. **构建统一推理后端**：将FlashLoop集成进生产级LLM服务系统

---

## ✅ 总结一句话

> **FlashLoop首次系统揭示了Looped Transformers中的跨循环冗余现象，并提出一种无需训练、高效实用的推理框架，在几乎不损失精度的前提下，实现了高达1.64倍加速和6倍KV Cache压缩，极大提升了循环式模型的实际可用性。**

</details>

---

### 3. [Where Does the Energy Go? Profiling LLM Agent Inference on Blackwell GPUs](https://arxiv.org/abs/2609.29707)

**Authors**: Qi Luo, Kunlin Li, Ziwen Wang, Yun Chen  
**Category**: cs.DC  
**Published**: 2026-09-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.29707v1  

#### Abstract
LLM agents that iteratively reason, plan, and invoke tools create workload profiles fundamentally different from single-pass inference, yet how their energy consumption is distributed across hardware components and workload phases remains poorly understood. Characterizing these workloads therefore r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Where Does the Energy Go? Profiling LLM Agent Inference on Blackwell GPUs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前对大语言模型（LLM）推理能耗的研究大多集中在**单次前向推理**或**批量服务场景**，而忽略了**LLM Agent**这一新兴范式。LLM Agent 具有迭代推理、调用工具、上下文累积等特性，其工作负载模式与传统推理显著不同。然而，目前缺乏对这类**agent 工作负载**在全系统层面的细粒度能耗分析。

本文填补了这一空白，首次系统性地研究了 LLM Agent 在 Blackwell 架构 GPU 上的**全栈（full-stack）能量分布**。

### 🚀 提出的新方法与创新思路
- **全栈能量剖析框架**：结合多种硬件监控接口，实现组件级和系统级的同步能量测量：
  - **NVML**：用于 GPU 能耗采样
  - **Intel RAPL**：用于 CPU 和 DRAM 能耗
  - **IPMI**：获取整机系统功耗（包括风扇、PSU、主板等）
- 首次将上述多源传感器融合，应用于完整的 agent 推理轨迹（tool-interleaved trajectories），覆盖从推理到工具执行的全过程。
- 引入“**per-turn energy profiling**”分析方式，追踪 agent 在多轮交互中每一轮的能量消耗变化。

### 🔍 相比现有方法的优势
| 现有方法局限 | 本工作的改进 |
|------------|-------------|
| 多数研究仅依赖 GPU telemetry（如 `nvidia-smi`） | 揭示 GPU-only 测量会遗漏 **41–45%** 的系统能耗 |
| 数据中心级研究缺乏细粒度相位划分 | 支持按 **workload phase**（prefill/decode/tool call）进行能量归因 |
| 缺乏对 reasoning 和 tool-interleaved 行为的建模 | 明确量化了 extended thinking、context growth 对能耗的影响 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与工作负载
论文评估了三种代表性 workload 类型：

| 工作负载类型 | 数据集 | 特点 |
|------------|--------|------|
| **Agentic Coding** | SWE-bench Verified (50 tasks) | 多轮代码修复任务，涉及 shell command、file read/write、grep 等工具调用 |
| **Mathematical Reasoning** | AIME 2022–2024 (90 problems)<br>MATH-500 | 对比开启/关闭 thinking 模式的能耗差异 |
| **Continuous Serving** | ShareGPT prompts | 连续批处理基准，请求速率为 1–16 req/s，用于建立能效前沿 |

模型统一使用 **Qwen3.8-27B**，部署于 **vLLM** 框架，启用 chunked prefill 和 tensor parallelism。

### ⚙️ 实验平台配置
- **GPU**: 2× NVIDIA RTX PRO 6000 Blackwell（96GB GDDR7，最大功耗 600W）
- **CPU**: 2× Intel Xeon Platinum 8558（共 96 线程）
- **内存**: 256GB DDR5 ECC
- **互连**: PCIe 5.0 ×16
- **软件栈**: Ubuntu 26.04, CUDA 13.3, Driver 610.57

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Energy per token (mJ/tok)** | 总系统能量 / 输出 token 数量 |
| **Power (W)** | 组件级与系统级实时功率 |
| **Energy per task (kJ)** | 单个任务完成所需的总系统能量 |
| **Throughput (tok/s)** | 每秒生成的 token 数 |
| **Pothers** | $ P_{\text{system}} - P_{\text{GPU}} - P_{\text{CPU}} - P_{\text{DRAM}} $，包含 PSU、VRM、风扇等损耗 |

> 注：除非特别说明，所有能量指标均基于 **total system energy**

---

## 3. 主要实验结果和性能指标

### 🔢 关键性能数据汇总

#### （1）GPU-only 测量严重低估系统能耗
- 在所有 workload 中，**GPU 仅占系统总功耗的 55–59%**
- 非 GPU 组件合计贡献 **41–45%** 的能耗
- “Others”项（含 PSU、VRM、风扇等）高达 ~274W，在 SWE-bench 中占比达 19.5%

> 💡 结论：仅靠 GPU telemetry 会严重低估真实碳足迹，影响绿色 AI 决策。

#### （2）Agent 工作负载单位输出 token 能耗极高
- **Sequential agent coding (SWE-bench)**:
  - 平均能耗：**58,383 mJ/tok**
  - 是饱和连续批处理（16 req/s）的 **63×**
- 原因分析：
  - ❌ 无跨请求 batching
  - ⏱️ 上下文增长导致 prefill 时间延长
  - ⛔ 工具调用期间 GPU Idle，但 CPU/系统仍耗电

#### （3）Extended Thinking 显著增加输出长度，但不影响 per-token 效率
| 数据集 | Thinking 模式 | Tokens per problem | Energy per problem | Δ per-token energy |
|-------|---------------|--------------------|---------------------|------------------|
| AIME | Enabled (√) | 10,506 | 313.2 kJ | <0.4% 变化 |
| AIME | Disabled (×) | 5,991 | 178.2 kJ | — |
| MATH-500 | Enabled | 1,800 | 53.6 kJ | — |
| MATH-500 | Disabled | 1,488 | 44.2 kJ | — |

> ✅ 发现：额外能耗主要来自 **output volume 增加（+21%～75%）**，而非 per-token 推理效率下降。

#### （4）Continuous Batching 显著提升能效
| Req/s | Throughput (tok/s) | System Energy per tok (mJ/tok) | 能效提升倍数 |
|------|---------------------|-------------------------------|-------------|
| 1    | 401                 | 2,955                         | 1.0×        |
| 4    | 1,158               | 1,147                         | ~2.6×       |
| 16   | 1,494               | 920                           | **3.2×**    |

- GPU 功耗在 4 req/s 后趋于饱和（~800W），而吞吐持续上升 → 利用率提高
- 展现出明显的规模效应（scale-out efficiency）

#### （5）Power Capping 实验揭示 workload 特性
- **Reasoning workload（memory-bandwidth-bound）**:
  - 400–600W cap 下 throughput 几乎不变（~48 tok/s）
  - 降至 300W 时 throughput 下降 **3.1×**，energy per tok 上升 **2.5×**
  - 表明存在“性能悬崖”（performance cliff）
- **Serving workload**:
  - 300W cap 实现最低 energy per tok（1,167 mJ/tok），优于更高上限

> ✅ 启示：应根据 workload 类型动态调整 power cap 策略。

---

## 4. 关键结论和发现

### 🎯 主要发现
1. **GPU telemetry 不足以代表系统能耗**  
   忽略非 GPU 组件会导致 **41–45% 的能量被漏计**，尤其在高 CPU/IO 密集型 agent 场景中更为严重。

2. **Agent 能耗激增源于行为模式，而非低效推理**  
   - 每 token 能耗稳定（<1% 差异）
   - 能耗上升主因是：
     - 更长的上下文 → 更久的 prefill
     - 更多输出 token（especially with thinking）
     - 工具调用引入的 idle period

3. **Context growth 是 agent 能耗的关键驱动因素**  
   - 每轮（turn）的能耗随回合数递增（见 Figure 4）
   - 平均功率基本不变 → 能耗增长由执行时间拉长引起
   - 建议采用 **context summarization** 或 **selective retrieval** 来缓解

4. **Batching 是提升能效最有效的手段之一**  
   - 连续批处理使系统能效提升 **3.2×**
   - GPU 功耗 plateau 而 throughput 持续增长 → 利用率优化空间大

5. **Power capping 需 workload-aware 设计**  
   - 对 memory-bound workload（如 reasoning），过低的 cap 会导致剧烈性能退化
   - 应避免一刀切的节能策略

---

### ⚠️ 方法的局限性
- 实验仅基于 **单一服务器平台**（双 Blackwell GPU + 双 Xeon CPU），泛化性有待验证
- 缺少外部 PDU 测量进行 AC 输入校准
- “Others”类别包含估算成分（如 PSU overhead、VRM loss），存在一定不确定性
- 未涵盖更多 agent 框架（如 LangChain、AutoGen）或其他模型规模

---

### 🔮 未来工作方向
1. **开发面向 agent 的 energy-aware scheduler**
   - 动态控制 thinking depth
   - 自适应 context truncation / summarization
   - early termination for low-confidence trajectories

2. **构建 agent-specific energy benchmark suite**
   - 类似 ML.ENERGY 或 TokenPowerBench，但支持 tool-interleaved tracing

3. **探索异构计算卸载策略**
   - 将部分 tool execution offload 至边缘设备以降低主机能耗

4. **扩展至多节点分布式 agent 集群**
   - 分析通信开销与能耗的关系

5. **结合 carbon intensity signal 实现 green agent dispatching**
   - 在电网清洁时段运行高能耗 agent 任务

--- 

> 📌 **一句话总结**：  
> LLM Agent 的高能耗并非源于 per-token 推理低效，而是由 **context growth、output expansion 和 tool-induced idle** 所致；真正的节能突破口在于 **context management** 与 **continuous batching**，且必须基于 **full-stack energy profiling** 才能准确评估。

</details>

---

### 4. [Graph-Based Inference and Topology-Aware Multi-Agent Reinforcement Learning for Large-Scale Railway Network Management](https://arxiv.org/abs/2609.30150)

**Authors**: Giacomo Arcieri, Gregory Duth\'e, Christophe Muller, Konstantinos G. Papakonstantinou, Daniel Straub, Eleni Chatzi  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.30150v1  

#### Abstract
Modern infrastructure asset management constitutes a complex sequential decision-making problem, characterized by long planning horizons and system-level interactions, such as spatial deterioration correlations and economies of scale. While deep reinforcement learning has shown promise in optimizing...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Graph-Based Inference and Topology-Aware Multi-Agent Reinforcement Learning for Large-Scale Railway Network Management

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代基础设施资产管理是一个复杂的**sequential decision-making**问题，具有以下挑战：
- **长规划周期**与**系统级交互**（如空间退化相关性和规模经济）
- **传统方法局限性**：组件级策略忽略系统级依赖；集中式DRL难以扩展到大规模网络；去中心化方法缺乏协调机制。

### 提出的新方法与创新思路
本文提出一个**端到端的图学习框架**，整合环境建模与决策优化，核心创新如下：

#### （1）**Graph-based Environment Inference**
- 利用**Hierarchical Bayesian Model**结合**Gaussian Process on Graphs (GPG)** 从真实铁路监测数据中推断网络化的退化动态。
- GPG核基于图拉普拉斯算子的谱特性构建，显式编码拓扑依赖关系，支持跨不同图结构的泛化。

#### （2）**Topology-Aware MARL 架构**
- 将**Graph Neural Networks (GCN)** 和 **Graph Transformers (GT)** 引入Multi-Agent Reinforcement Learning (MARL)，使智能体能感知局部拓扑上下文进行协作决策。
- 智能体策略基于**局部邻域特征**而非固定节点ID，实现**参数共享与结构不变性**。

#### （3）**Zero-Shot Transfer Learning for Scalability**
- 首次在基础设施管理中实现**零样本迁移**：在小规模子图上训练的agent可直接部署到未见过的大规模网络，无需再训练。
- 成功绕过标准MARL在大规模系统中的计算瓶颈。

### 相比现有方法的优势
| 方法 | 局限性 | 本论文优势 |
|------|--------|------------|
| Heuristic Rules | 仅考虑局部状态，无法捕捉系统级协同机会 | 主动协调维护动作，显著降低生命周期成本 |
| Centralized DRL (CTCE) | 输入维度随网络增长爆炸，不可扩展 | 图结构提供归纳偏置，提升学习效率 |
| Standard CTDE-MARL | 依赖one-hot Agent ID，无法迁移到新网络 | 基于拓扑上下文的策略，支持zero-shot transfer |
| Decentralized Agents | 缺乏协调机制，错失规模经济效益 | 显式建模邻居影响，有效利用economies of scale |

---

## 2. 核心实验方法和设置

### 数据集
- **数据来源**：瑞士联邦铁路（Swiss Federal Railways, SBB）
- **地理范围**：苏黎世都市区约10年内的轨道监测数据
- **关键数据类型**：
  - **Monitoring Data**：每6个月采集一次轨道几何参数（纵向水平偏差），采样间隔0.25米。
  - **Condition Indicators**：
    - **D1 (3–25m)**：短波长缺陷，反映上部结构问题（如道砟压实不均）。
    - **D2 (25–70m)**：长波长缺陷，反映下部结构问题（如路基沉降）。
  - **Topology Data**：轨道连接关系（用于构建图$ G=(V,E) $）
  - **Maintenance Logs**：历史维修记录（tamping、renewal等）

### 实验设置
- **图建模**：将连续轨道划分为150米段，每段为一个node，物理连接定义edge。
- **时间步长**：Δt = 6个月，总时长T = 50步（25年）
- **动作空间** $ \mathcal{A} = \{a_0, a_1, a_2, a_3\} $：
  - $ a_0 $: Do-nothing
  - $ a_1 $: Tamping
  - $ a_2 $: Re-tamping（更换后一年内强制执行）
  - $ a_3 $: Track Renewal

### 评估指标
- **Life-Cycle Cost (LCC)**：总维护成本 + 条件惩罚（若超过安全阈值则施加高额罚款）
- **Reward Function**：$ r_t = -\sum_i (C_{\text{maint}}(a_{t,i}, k_{n,i}) + C_{\text{cond}}(s_{t,i})) $
  - 其中 $ k_{n,i} $ 表示与agent i同时执行相同动作的连通邻居数量，体现**economies of scale**

### 基线方法对比
| 类别 | 基线方法 | 描述 |
|------|---------|------|
| **Heuristic** | Only Tamping | 每步都进行tamping，作为性能下界 |
| | Optimized Threshold Policy | 基于网格搜索优化的条件触发规则 |
| **RL Baselines** | CTCE-PS (MAPPO) | 中心化训练+执行，输入为全局状态向量 |
| | CTDE-PS (MAPPO) | 中心化训练+去中心化执行，使用Agent ID区分节点 |
| | Fully Decentralized Agent | 单节点独立训练PPO，复制到全网，无协调能力 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见Table 2）
| Policy | Medium Scale (N=349) | Large Scale (N=1016) |
|--------|-----------------------|------------------------|
| Optimized Threshold Policy | -72,521 | -214,553 |
| Independent Agent (Single Node) | -72,380 | -214,900 |
| Topology-Aware GCN | **-62,691** | **-179,491** |
| Topology-Aware GT | **-61,671** | **-175,028** |

> 数值为累计reward（负值越小越好），即**Topology-Aware方法节省约15%-18%生命周期成本**

### 与基线方法对比结果
- 在**52节点训练环境**中（图14）：
  - GCN/GT agent快速收敛并优于CTCE-PS和Optimized Threshold Policy。
  - 尽管CTCE拥有完整全局信息，但由于缺乏图结构先验，难以从高维扁平向量中学习有效协调模式。

- 在**349节点零样本迁移测试**中（图15）：
  - GCN/GT agent（仅在52节点训练）表现媲美直接在该网络训练7天的CTDE-PS agent。
  - 实现“即时部署”，避免数天训练开销。

- 在**1016节点超大规模网络**中：
  - CTDE-PS因计算不可行而无法训练。
  - Topology-Aware agents仍保持高性能，远超Decentralized和Heuristic基线。
  - 证明其**极端可扩展性**。

### 消融实验与关键分析
- **GPG模型验证**：
  - 推断出的空间相关性随拓扑距离衰减（图8），符合实际物理规律。
  - 在未见的52节点网络上生成的退化轨迹与真实数据高度一致（图12），验证了**环境模型的泛化能力**。

- **Agent ID的影响**：
  - 使用Agent ID虽有助于稳定训练，但导致策略绑定特定图大小，丧失迁移性。
  - 本文通过**topology-aware encoding**替代ID，兼顾稳定性与可转移性。

- **GCN vs. GT**：
  - GT因使用Attention机制，在远距离依赖建模上略优（尤其在稀疏连接区域）。
  - 两者均显著优于传统架构，表明**图结构感知是关键增益来源**。

---

## 4. 关键结论和发现

### 主要发现
1. **显式建模拓扑依赖对环境推理至关重要**：
   - GPG成功捕获非欧几里得空间退化相关性，支持跨网络拓扑的仿真生成。
   
2. **Topology-Aware MARL显著提升协调效率**：
   - 智能体学会在连通区域内同步维护动作，最大化**economies of scale**带来的成本节约。

3. **Zero-Shot Transfer 是解决大规模基础设施管理的关键路径**：
   - 在小规模代表性子图上训练即可获得高质量策略，适用于城市级甚至国家级网络部署。
   - 训练时间从“数天”降至“数小时”，极大增强实用性。

4. **共同谱基础统一了推理与决策**：
   - GPG核与Graph Transformer均基于图拉普拉斯的特征分解（$ \Delta = U\Lambda U^T $），使得**环境动力学与决策机制在同一数学空间表达**，增强了系统一致性。

### 方法的局限性
- **假设局部拓扑相似性**：迁移效果依赖于训练与目标网络具有类似的度分布和连接模式（如同为线性主干+稀疏分支）。若拓扑差异过大（如从线性网转至密集网格），性能可能下降。
- **静态图假设**：当前模型假设网络拓扑不变，未处理动态重构或施工引起的临时断连。
- **奖励函数设计敏感性**：economies of scale的成本函数需准确估计，否则可能导致过度延迟维护。

### 未来工作方向
1. **跨领域迁移研究**：
   - 测试Topology-Aware MARL在其他基础设施系统（如桥梁群、风力发电场、道路网络）中的通用性。
   
2. **混合核建模**：
   - 结合**Graph Kernel + Spatial Kernel**，同时捕捉拓扑依赖与地理地质相关性。

3. **动态图与部分可观测扩展**：
   - 引入Dynamic GNNs处理拓扑变化，并结合POMDP框架应对传感器缺失场景。

4. **人机协同策略蒸馏**：
   - 将学到的复杂策略提炼为可解释的规则，辅助人工决策者理解AI建议逻辑。

--- 

> ✅ **总结一句话**：  
> 本文通过**图神经网络+多智能体强化学习+零样本迁移**，实现了对大规模铁路网络的高效、可扩展、系统级最优维护决策，为现实世界基础设施智能运维提供了新的范式。

</details>

---

### 5. [SpaFactor: Lightweight Spatial Context-Aware Gene Program Modeling for Histology-to-Transcriptomics Inference](https://arxiv.org/abs/2609.28563)

**Authors**: Shiting Ruan, Xitong Ling, Qiming He, Ziyou Yan, Huaitian Yuan, Tian Guan, Ying Xiao, Xu Guan, Yonghong He  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.28563v1  

#### Abstract
Spatial transcriptomics (ST) profiles gene expression within tissue architecture, but its cost and experimental complexity limit routine use. Predicting spatial expression from routinely available hematoxylin and eosin (HE) images therefore offers a scalable alternative. However, conventional method...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SpaFactor: Lightweight Spatial Context-Aware Gene Program Modeling for Histology-to-Transcriptomics Inference

## 1. 论文的主要贡献和创新点

### 解决的问题
- **空间转录组学**（Spatial Transcriptomics, ST）虽然能够提供基因表达的空间定位信息，但其高昂的成本、复杂的实验流程和平台依赖性限制了广泛应用。
- 当前基于 **H&E 图像到 ST 推断**（H&E-to-ST inference）的方法通常将高维基因输出作为独立目标进行建模，忽略了基因间的生物学协调关系，容易受到高维噪声和过拟合的影响。
- 现有方法常依赖计算密集的图网络（graph networks）或需要复杂辅助监督信号，难以实现高效、轻量化的预测。

### 提出的新方法与新思路
作者提出 **SpaFactor**，一种轻量级、高效的低秩形态-程序-基因分解框架（low-rank morphology-program-gene factorization），核心思想如下：

- **轻量级空间上下文建模**：
  - 引入**无参数、可预计算的空间聚合机制**，通过拼接中心点位的 GigaPath 嵌入与其局部（k=4）和区域（k=16）邻域的均值嵌入，构建具有空间感知能力的组织学表示。
  - 避免使用图消息传递（graph message passing）、跨点注意力（cross-spot attention）或对编码器微调，显著降低计算开销。

- **因子化解码的基因程序建模**（Factorized Gene-Program Prediction）：
  - 设计“形态 → 潜在程序 → 基因”三级映射结构。
  - 使用一个四层残差 MLP（ResMLP）从组织微环境学习非线性映射，得到低维潜在基因程序活动（latent program activities, K=256）。
  - 通过共享的基因载荷矩阵（shared gene loading matrix）将这些程序解码为多基因表达预测，使相关基因共享统计强度，增强模型正则化和生物一致性。

- **无需外部知识的结构化建模**：
  - 不依赖外部基因数据库、通路标签或表达基础模型（expression foundation model），完全由数据驱动学习基因间依赖关系。

### 相比现有方法的优势
- **高效轻量**：训练和推理仅需批处理稠密操作，下游模型仅含 2230 万参数，比直接解码小 62.1%。
- **高性能**：在多个公开数据集上取得最佳综合表现，尤其在空间变异基因恢复方面优势明显。
- **生物学保真度高**：能更准确地重建生物组织的空间模式和功能通路活性。
- **部署友好**：支持 H&E-only 推理路径，适合大规模回顾性病理队列分析。

---

## 2. 核心实验方法和设置

### 数据集
共评估五个公开空间转录组数据集，总计 **421 张切片、799,085 个 spot**：

| 数据集 | 组织类型 | 平台 | 切片数 | Spot 数 |
|--------|---------|------|-------|--------|
| Primary breast | 乳腺癌 | Spatial Transcriptomics | 108 | 45,306 |
| Visium breast | 乳腺癌 | Visium | 87 | 163,931 |
| Bowel | 肠道 | Visium | 73 | 205,642 |
| Brain | 大脑 | Visium | 91 | 322,572 |
| Skin | 皮肤 | Visium | 62 | 61,634 |

数据来自 **STimage-1K4M** 和 **HEST-1k** 两个大型基准数据集。

### 实验设置
- **五折交叉验证**：按切片级别划分训练/验证/测试集（~70%/10%/20%）。
- **特征提取**：
  - 使用预训练且冻结的 **Prov-GigaPath tile encoder** 提取每个 spot 的 H&E 图像嵌入（1536 维）。
  - 所有方法共享相同的冻结特征、基因面板、数据划分和评估代码，确保公平比较。
- **基因面板构建**：
  - 在每折训练集中选择最多 2000 个高变基因（HVG），并标准化处理。
- **模型配置**：
  - ResMLP 隐藏维度 1024，内部宽度 2048，4 层残差块，dropout=0.1。
  - 潜在因子数 K=256。
  - 优化器：AdamW（lr=1e-3, wd=1e-5），batch size=1024，最大 100 轮，早停基于 VR-PCC@200。

### 评估指标
- **Spot PCC**：所有 spot 内预测与真实基因谱之间的平均 Pearson 相关系数（衡量整体表达一致性）。
- **Gene PCC**：每个基因在所有 spot 上预测与真实的 Pearson 相关系数的平均值（衡量空间模式一致性）。
- **VR-PCC@K**：在 held-out 数据中变异最大的 K 个基因上的 Gene PCC 平均值（K ∈ {50,100,200}），强调对空间异质性基因的建模能力。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Primary Breast 数据集）
| 方法 | Spot PCC (%) | Gene PCC (%) | VR-PCC@200 (%) |
|------|---------------|----------------|--------------------|
| HE2RNA（最强基线） | 71.19 ± 3.37 | 33.86 ± 4.10 | 59.49 ± 6.69 |
| GigaPath-MLP（匹配控制） | 70.29 ± 3.60 | 33.10 ± 4.48 | 58.84 ± 7.01 |
| **SpaFactor** | **71.67 ± 3.29** | **36.36 ± 4.17** | **63.41 ± 6.68** |
| SpaFactor-Cal | 71.73 ± 3.28 | 36.33 ± 3.71 | 63.20 ± 6.23 |

> **相对提升**：相比 HE2RNA，SpaFactor 在 Gene PCC 上提升约 **7.3%**，在 VR-PCC@200 上提升高达 **6.5%**。

### 与其他方法的全面对比
- 在 **5 个数据集、20 项指标中，SpaFactor 或 SpaFactor-Cal 在 18 项上优于最强先前模型**。
- 特别是在 **VR-PCC 指标上优势最一致**，表明其在恢复空间有序表达模式方面显著优于现有方法。
- 在所有数据集上均超越匹配的 **GigaPath-MLP** 基线，在 25 对比中赢得 20–23 次（p < 0.05，bootstrap 检验）。

### 消融实验结果（Ablation Studies）
| 变体 | Gene PCC ↓ | VR-PCC@200 ↓ | 结论 |
|------|------------|----------------|------|
| Full SpaFactor | 36.36 | 63.41 | 基准 |
| △ Spot only（无邻域） | 33.99 (-2.37) | 59.61 (-3.80) | **空间上下文至关重要** |
| △ Regional only（仅区域 k=16） | 36.14 | 63.12 | 区域上下文主导性能 |
| △ Local only（仅局部 k=4） | 35.66 | 62.28 | 局部提供补充细节 |
| △ Plain MLP（无残差） | 16.74 | 31.49 | **残差连接稳定训练** |
| △ Direct decoder（无因子解码） | 35.74 | 62.91 | 因子化解码提升协调性 |
| △ Uniform HVG weights | 36.15 | 63.05 | 权重设计影响较小 |
| △ No Gene-PCC loss | 36.22 | 63.27 | 辅助损失非关键因素 |

> **结论**：空间上下文是主要性能来源；残差结构保障深层映射稳定性；因子化解码带来小幅但稳定的增益。

### 敏感性分析
- **因子数量 K**：在 K=64 至 1024 范围内性能稳定，K=256 已足够，具备良好泛化性。
- **残差深度 L**：L=4 即达高性能，更深模型（L=10/12）参数增加 2.1–2.5 倍，训练时间延长 38%，但性能未持续提升，验证了轻量化设计的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. **轻量级联合建模有效提升性能**：
   - 将空间组织上下文与基因程序联合建模，可在不引入复杂架构的前提下显著提高预测准确性。
   - **空间上下文**（尤其是区域级 k=16）是性能提升的主要驱动力。

2. **生物学保真度更高**：
   - SpaFactor 更好地恢复了已知生物标志物（如 COL1A1, GATA3, ERBB2, ESR1）的空间分布边界。
   - 在 **1,274 个滑片-通路对** 上，平均提升通路活性预测 PCC **6.11 个百分点**（95% CI [4.71, 7.48]），证明其能捕捉协调的生物学功能。

3. **效率与精度兼备**：
   - 模型轻量、训练快速、易于部署，适合大规模应用。
   - 所有改进均建立在固定的基础模型之上，说明下游结构设计本身即可带来显著收益。

### 方法的局限性
- 依赖高质量的 spot 级 H&E 与 ST 配准。
- 当前模型未显式建模细胞类型组成或单细胞分辨率信息。
- 虽然性能优越，但在极端稀疏或低质量图像下可能受限。

### 未来工作方向
- 扩展至 **super-resolution** 或 **single-cell level** 的表达推断。
- 结合临床表型进行端到端 biomarker discovery。
- 探索更多可解释的潜在程序（latent programs）与已知生物学通路的对应关系。
- 进一步压缩模型以适应边缘设备部署。

---

> ✅ **总结一句话**：  
> **SpaFactor 通过轻量级的空间上下文融合与因子化的基因程序建模，在不依赖复杂图网络或外部知识的情况下，实现了更准确、更具生物学意义的 H&E-to-ST 推断，为大规模空间基因组研究提供了高效实用的新工具。**

</details>

---

### 6. [A Rapid Pipeline for Training and Deploying ML Models on WeBe Band](https://arxiv.org/abs/2609.29084)

**Authors**: Ehsan Kourkchi, Asmita Asmita, Houman Homayoun, Mahdi Eslamimehr  
**Category**: cs.AI  
**Published**: 2026-09-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.29084v1  

#### Abstract
Developing optimized machine-learning algorithms for edge devices with limited computational and memory resources is challenging, time-consuming, and highly dependent on device-specific constraints. In this work, we streamline an edge ML workflow to enable rapid development, optimization, and deploy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《A Rapid Pipeline for Training and Deploying ML Models on WeBe Band》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对 **TinyML** 在边缘设备（尤其是可穿戴设备）上部署时面临的 **开发-部署割裂问题** 展开研究。具体挑战包括：
- 模型训练与嵌入式部署流程脱节，研究人员需同时掌握 ML 和 embedded systems 技术；
- 缺乏自动化工具链支持，导致从模型到固件的集成过程繁琐、易错且高度依赖硬件细节；
- 医疗健康等跨学科团队缺乏嵌入式开发能力，难以在真实硬件上验证模型性能。

### 🚀 提出的新方法与思路
作者提出一个 **端到端的自动化 ML 部署流水线（end-to-end pipeline）**，专为 WeBe Band 可穿戴平台设计，其核心创新如下：

- **系统级自动化部署框架**：  
  将开源的 **Piccolo AI** 生态系统与自定义后端集成，实现从数据上传 → 模型训练 → 固件编译 → OTA 部署的全流程自动化。
  
- **硬件感知的模型生成与部署**：  
  支持 AutoML、硬件感知量化（hardware-aware quantization）、以及 on-device profiling，确保生成的模型满足目标设备的延迟、内存和功耗约束。

- **无需干预的固件集成机制**：  
  用户只需上传数据和模型知识包（Knowledge Pack, KP），系统自动完成头文件生成、编译配置、二进制打包，并通过蓝牙 **OTA** 推送到 WeBe Band。

- **通用化架构设计**：  
  虽然以 WeBe Band 为例，但整个流程基于模块化设计，可扩展至其他 Cortex-M 类设备（如 STM32、ESP32、nRF53 等）。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 开发门槛 | 高（需熟悉 firmware、toolchain） | 低（仅需提供数据/KP） |
| 部署效率 | 手动修改代码、烧录固件 | 自动化生成 + OTA 更新 |
| 性能评估方式 | 离线仿真或理论估算 | **on-device profiling**（真实环境测量） |
| 可复现性 | 差（依赖人工操作） | 强（标准化 pipeline） |
| 跨平台潜力 | 通常绑定特定硬件 | 模块化设计支持迁移 |

> ⭐ 核心优势：**将 TinyML 实验的重点从“如何部署”转移到“如何优化模型”**，极大降低跨学科研究的技术壁垒。

---

## 2. 核心实验方法和设置

### 📊 数据集
- **任务类型**：手势识别（motion detection）
- **采集设备**：WeBe Band（采样率 25 Hz）
- **传感器输入**：三轴加速度计（accelerometer）
- **动作类别**：6 种空中书写手势 —— 字母 A, B, C, D, X, O
- **参与者**：2 名用户
- **数据格式**：CSV 文件，经预处理后用于训练
- **窗口设置**：每段 100 个样本（约 4 秒），滑动步长 20 样本（0.8 秒）

> 注：本研究不强调数据规模或泛化能力，而是聚焦于 **部署可行性与资源消耗分析**。

### ⚙️ 实验设置
- **目标平台**：WeBe Band（ARM Cortex-M4F @ 64 MHz，1MB RAM，外接 Flash）
- **模型训练平台**：Piccolo AI
- **部署方式**：通过蓝牙进行 OTA 更新
- **评估方式**：所有性能指标均在 **真实设备上直接测量（on-device profiling）**

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Latency (μs)** | 单次推理最大延迟（使用 DWT cycle counter 精确测量） |
| **Flash Memory (Bytes)** | 模型存储所需闪存空间 |
| **SRAM (Bytes)** | 运行时占用的静态随机存取内存 |
| **Inference Stability** | 是否影响连续数据采集（是否丢帧） |

### 🆚 基线方法对比
共评估五类模型：
1. **Random Forest (RF)** —— 经典集成学习
2. **Pattern Matching Engine (PME)** —— 基于原型的距离分类器
3. **Neural Networks (NN1–NN4)** —— 不同结构的轻量级全连接网络（参数量 ~3.9K–8.9K）

> 所有模型使用相同的 **66 维统计特征集** 和相同的数据划分，以公平比较架构差异带来的影响。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table I）

| Model | Architecture | #Params | Latency (μs) | Flash (Bytes) | SRAM (Bytes) |
|-------|--------------|---------|---------------|----------------|----------------|
| RF    | –            | –       | **8,520**     | 10,368         | 1,244          |
| PME   | –            | –       | 8,763         | **8,234**      | 1,364          |
| NN1   | 32-32-16-8   | ~3.9K   | 10,535        | 15,666         | 4,444          |
| NN2   | 64-64-32-16-8| ~8.9K   | 11,209        | 24,050         | 4,444          |
| NN3   | 64-32-16-8   | ~5.9K   | 10,769        | 18,930         | 4,444          |
| NN4   | 32-32-16-16-8| ~4.8K   | 10,832        | 16,754         | 4,444          |

> ✅ 最佳表现：**RF 和 PME 在延迟和内存方面全面优于 NN 模型**

### 🔬 与基线方法的对比结果
- **延迟方面**：
  - RF 和 PME 推理时间 < 9ms，适合实时应用；
  - NN 模型延迟 > 10.5ms，增加约 25% 以上。
- **内存占用**：
  - RF/PME 仅需 ~8–10KB Flash，而 NN 模型需要 15–24KB；
  - SRAM 占用：NN 模型是 RF/PME 的 **3.5 倍以上**（4,444 vs ~1,300 Bytes）。
- **运行稳定性**：
  - 所有模型在滑动窗口推理中未造成数据丢失；
  - RF/PME 表现出更稳定的 latency 曲线（图 3），NN 存在明显峰值波动。

### ❌ 消融实验（隐含分析）
虽然未明确列出消融实验，但通过不同 NN 架构的对比可得出以下结论：
- **层数越深、宽度越大 → 参数越多 → Flash 和 Latency 显著上升**
- **SRAM 使用几乎一致**：因中间激活缓冲区大小相近，表明内存瓶颈主要来自中间层输出
- **小规模 NN 已显著劣于经典模型**：即使是最简单的 NN1（~3.9K params），其延迟和内存仍远高于 RF/PME

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **经典 ML 模型更适合资源受限边缘设备**：
   - Random Forest 和 PME 在 **latency、memory footprint、稳定性** 上全面优于轻量级神经网络。
   - 对于简单分类任务（如手势识别），无需引入复杂 NN 架构。

2. **on-device profiling 至关重要**：
   - 实际硬件上的性能与离线估计存在偏差；
   - 本文采用 DWT cycle counter 实现精确测量，提升了结果可信度。

3. **自动化部署显著加速迭代周期**：
   - 研究人员可在数分钟内完成“模型更新 → 设备部署 → 性能测试”的闭环；
   - 支持快速探索模型-资源权衡（trade-off exploration）。

4. **系统设计具有良好的可扩展性**：
   - 流程不绑定特定硬件，可通过适配 KP 解析与固件抽象层迁移到其他 Cortex-M 平台；
   - 支持多种 model types（classical ML / NN）和 inference backends（TFLM / CMSIS-NN）。

### ⚠️ 方法的局限性
- **应用场景有限**：目前仅验证于手势识别任务，尚未覆盖复杂的多模态生理信号建模（如 ECG + PPG 融合）；
- **精度非重点**：未报告分类准确率，无法判断性能下降是否由简化模型引起；
- **用户样本少**：仅两名用户参与数据收集，泛化性存疑；
- **未支持动态模型切换**：当前 OTA 更新替换整个固件，不能热插拔多个模型。

### 🔮 未来工作方向
1. **扩展至更多生理模态**：整合 PPG、EDA、SpO₂ 等传感器，构建综合健康监测模型；
2. **支持更大规模数据集与跨用户泛化研究**；
3. **实现多模型共存与动态加载机制**（multi-model firmware）；
4. **进一步验证跨平台兼容性**：在 STM32、ESP32 等平台上复现该 pipeline；
5. **结合 energy profiling**：加入功耗监控，优化能耗-性能平衡。

---

> 💡 **总体评价**：  
> 本论文并非追求算法创新，而是推动 **TinyML 向“可部署优先”范式转变**。它强调：  
> > “**Deployability is a first-class design objective.**”  
> 通过构建一个高效、易用、硬件感知的自动化 pipeline，为医疗健康领域的跨学科协作提供了强有力的工程基础。

</details>

---

### 7. [PTC-Bias: Phoneme-Level Temporal Competition for Bias Retrieval and Post-Decoding Correction in Speech LLMs](https://arxiv.org/abs/2609.28727)

**Authors**: Zhiqi Ai, Han Cheng, Shiyi Mu, Yongjin Zhou, Shugong Xu  
**Category**: cs.CL  
**Published**: 2026-09-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.28727v1  

#### Abstract
Contextual biasing improves rare-word recognition in speech large language models (SpeechLLMs), but efficiently exploiting large bias lists remains challenging. We propose PTC-Bias, a two-stage framework based on phoneme-level temporal competition. At the prefill stage, PTC Retrieval performs frame-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PTC-Bias: Phoneme-Level Temporal Competition for Bias Retrieval and Post-Decoding Correction in Speech LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
SpeechLLMs 在识别**罕见词**（如人名、地名、技术术语）时表现不佳。虽然可通过引入上下文 bias list 来提升识别准确率，但直接将大规模 bias list 插入提示（prompt）会导致：
- 推理成本上升
- 引入词汇干扰（lexical interference）
- 难以精确定位语音片段的时间位置

此外，现有检索方法存在以下不足：
- 缺乏对**近音词**（near-homophones）之间的显式竞争建模
- 检索后无法纠正 SpeechLLM 输出中的错误（如分词错误、同音替换）

---

### 🚀 提出的新方法：PTC-Bias
提出一个两阶段框架 **PTC-Bias**，基于 **phoneme-level temporal competition** 实现高效的上下文偏置处理。

#### 创新点：
1. **PTC Retrieval（Prefill 阶段）**
   - 基于轻量级 phoneme-CTC 分支生成帧级 phoneme posteriors
   - 构建共享前缀树（prefix trie），进行帧同步的 phoneme 解码
   - 在候选发音之间引入**时间竞争机制**（temporal competition），抑制重叠区间内的弱匹配项
   - 输出紧凑的 bias-word 短列表及其对应的语音时间区间

2. **PTC Correction（Post-decoding 阶段）**
   - 复用 PTC Retrieval 中缓存的 phoneme posteriors 和时间区间
   - 对 SpeechLLM 初始输出中与检索结果不一致的文本段落进行局部竞争比较
   - 使用 CTC log forward probability 计算声学得分差（acoustic margin），决定是否替换
   - 支持边界检查与词法验证，避免误改正确转录

#### 核心优势：
- **无需额外 SpeechLLM forward pass**：两个阶段共享同一套 phoneme posteriors
- **高效且可扩展**：支持高达 2000 词的大规模 bias list
- **精准纠错能力**：不仅能检索，还能在解码后修正 near-homophone 和 word-segmentation 错误
- **低延迟 CPU 检索**：利用并行化实现快速搜索（如 N=2000 仅需 ~7.6ms）

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练集**：
  - LibriSpeech 的 460 小时 clean 子集（LS-460）
  - 完整的 960 小时训练集（LS-960）
- **测试集**：
  - `test-clean`（2,620 条）
  - `test-other`（2,939 条）

遵循 **Rare5k 协议**：
- 每条 utterance 包含 oracle rare words + N 个干扰项（distractors）
- 干扰项数量 $ N \in \{100, 500, 1000, 2000\} $
- 最多从列表中选择 K=10 个 bias words 输入模型

---

### ⚙️ 实验设置
- **SpeechLLM 主干模型**（冻结）：
  - Prompt-SLAM-ASR-7B
  - Qwen3-ASR-0.6B
- **Phoneme Front-ends 对比**：
  - DS-KWS（专用 KWS 模型）
  - WavLM（预训练语音编码器）
  - AuT（来自 Qwen3-ASR 的音频 Transformer）

- **PTC 参数配置**：
  - $ M = 100 $（保留最高 M 个事件）
  - $ K = 10 $（最终传递给 SpeechLLM 的 bias 数）
  - 温度 $ T = 0.25 $，阈值 $ \eta = 0.05 $
  - 局部窗口扩展 ±5 帧，声学 margin 阈值为 2.0

---

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **WER** | Overall Word Error Rate |
| **B-WER** | Biased-word WER（关注 bias list 中词语的错误率） |
| **U-WER** | Unbiased-word WER（非 bias 词的错误率） |
| **PER** | Phoneme Error Rate |
| **RecallB@99** | 达到 99% oracle recall 所需平均候选数 |
| **RecallB#50** | Top-50 中命中 oracle 的比例 |
| **RecallH#50** | Top-50 中误召回 near-homophone 的比例 |

---

### 🔁 基线方法对比
- **DB-NNLM** [1]
- **USTR-CT** [3]
- **CB-QwenAudio** [4]
- **CTC-Filter** [5]
- **Bias Retrieval** [7]
- **BR-ASR (Acoustic/Textual)** [7]

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（N=2000）

#### 在 Prompt-SLAM-ASR-7B 上的结果（Table 1）：
| 方法 | test-clean / test-other |
|------|------------------------|
| **CTC-Filter** | B-WER: 4.41%/10.02% |
| **+ PTC Retrieval** | B-WER: 3.85%/8.24% |
| **+ PTC Correction** | **B-WER: 3.38%/7.63%** |
| **相对提升** | ↓ **23.4% / 23.9%** vs CTC-Filter |
| **U-WER 变化** | 几乎不变（仅轻微波动） |

> ✅ 表明 PTC-Bias 显著降低 bias 词错误率，同时不影响通用词识别质量。

#### 在 Qwen3-ASR-0.6B 上的结果：
| 方法 | test-clean / test-other |
|------|------------------------|
| **+ PTC Correction** | B-WER: 4.45%/8.61% |
| 相比 CTC-Filter | ↓ 明显改善（尤其在 large bias list 下）

---

### 🔍 消融实验分析（Ablation Study）

#### （1）PTC Retrieval vs 其他检索方法（Table 3）
| 方法 | RecallB@99 | RecallB#50 | RecallH#50 |
|------|------------|-------------|------------|
| BR-ASR (Acoustic) | 42.2 | 99.7% | 69.3% |
| **PTC Retrieval** | **16.9** | **99.3%** | **22.0%** |

✅ 结论：
- 以更短的短列表（↓60%）达到相同 recall
- 近音干扰显著减少（↓47.3个百分点）

#### （2）PTC Correction 的增益（Nested Rows in Table 1）
- 在 Prompt-SLAM-ASR-7B 上：
  - PTC Retrieval → B-WER: 3.85%/8.24%
  - + PTC Correction → **3.38%/7.63%**
  - ⇒ 绝对下降约 0.5–0.6%，说明**后解码校正有效**

> 💡 即使正确 bias 被检索到，SpeechLLM 仍可能输出错误形式（如 “Pleas” → “Plaza”），需二次竞争判断。

#### （3）不同 phoneme front-end 的 PER 表现（Table 2）
| Front-end | test-clean / test-other |
|----------|------------------------|
| DS-KWS | 4.45% / 11.80% |
| AuT | 2.06% / 5.13% |
| **WavLM** | **1.13% / 2.27%** |

✅ 表明高质量 phoneme posteriors 显著影响整体性能，但即使使用 AuT 这类通用 SpeechLLM 编码器也能提供足够判别力。

---

### ⏱️ 效率分析（Figure 3）
- **CPU 检索延迟**（5.09 秒音频）：
  - $ N=2000 $：**7.61 ms**（4 workers）
  - $ N=50,000 $：从 391 ms（单核）降至 **34.74 ms**（16 核）

✅ 表明 PTC Retrieval 可通过并行高效扩展至超大 bias list，适合工业部署。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **PTC-Bias 显著降低 B-WER**：
   - 在 Prompt-SLAM-ASR-7B + 2000 bias 词下，相对 CTC-Filter 下降 **23.4%/23.9%**
   - U-WER 基本保持稳定，说明未破坏正常识别能力

2. **两阶段设计协同增效**：
   - PTC Retrieval 提供高精度、低歧义的短列表
   - PTC Correction 进一步修复 near-homophone 和分词错误
   - 二者复用同一 phoneme posteriors，零额外计算开销

3. **temporal competition 机制有效抑制干扰**：
   - 显式建模多个候选在时间上的冲突关系
   - 抑制共现区间的弱近音词，提高短列表纯净度

4. **兼容多种 front-end 和 SpeechLLM 架构**：
   - 在 WavLM、AuT、DS-KWS 上均有效
   - 支持 Prompt-SLAM 和 Qwen3-ASR 两种主流 SpeechLLM

---

### ⚠️ 局限性
1. **依赖 phoneme-CTC 模块的质量**：
   - 若 phoneme alignment 不准，会影响两个阶段效果
   - 当前需额外训练轻量 CTC 分支（尽管参数极少）

2. **无法区分完全同音词（exact homophones）**：
   - 如 “there” 和 “their” 无法通过声学建模区分
   - 当前仅在特定 rare-word 或 segmentation 场景下处理

3. **g2p 转换误差风险**：
   - 英语 grapheme-to-phoneme 转换（g2pE）可能存在错误发音映射

---

### 🔮 未来工作方向
1. **端到端集成 phoneme CTC 分支**
   - 将 phoneme-CTC 与主模型联合优化，进一步提升 alignment 质量

2. **结合 semantic context 进行 post-correction**
   - 当声学证据不足时，引入 LLM 的语义理解辅助决策

3. **支持 streaming 场景下的实时更新**
   - 动态添加新 bias words，并实现实时竞争检索

4. **跨语言扩展**
   - 探索多语言 phoneme inventory 下的通用性

---

> 📌 **总结一句话**：  
> **PTC-Bias 通过 phoneme-level temporal competition 实现了高效、精准、可扩展的上下文偏置机制，在大幅提升 rare-word 识别准确率的同时，保持低延迟与零额外 SpeechLLM 推理开销，是当前 SpeechLLM 偏置增强的一项重要进展。**

</details>

---

### 8. [Generative Atmospheric Super-Resolution from Heterogeneous In Situ Observations through Composable Interfaces](https://arxiv.org/abs/2609.29027)

**Authors**: Yang Xu, Dibyajyoti Chakraborty, Haiwen Guan, Sen Wang, Romit Maulik  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.29027v1  

#### Abstract
Atmospheric observations are sparse, heterogeneous, and unevenly distributed, whereas many generative atmospheric models learn distributions over regularly gridded multivariate states. Once pretrained, diffusion models can supply atmospheric priors that can be combined with observation-derived likel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Generative Atmospheric Super-Resolution from Heterogeneous In Situ Observations through Composable Interfaces

## 1. 论文的主要贡献和创新点

### 解决的问题
大气观测数据具有**稀疏性、异质性和不均匀分布**的特点，而现有的生成式大气模型通常在规则网格化的多变量状态上学习分布。如何将这些结构差异显著的**in situ 观测源**（如探空、飞机、地面站）有效地融合到一个预训练的生成先验模型中，是一个关键挑战。

传统方法往往需要为每种观测组合重新训练模型，缺乏灵活性和可扩展性。

### 提出的新方法与创新思路
本文提出了 **“可组合的观测接口”（composable observation interfaces）** 框架，其核心创新在于：

- **解耦先验与观测建模**：保留一个固定的、预训练的13变量大气扩散模型作为**通用先验**（atmospheric prior），不进行任何微调。
- **显式定义观测接口**：为每个观测源（R/A/S）设计独立的“接口”，明确指定以下四个组件：
  1. **保留的观测构造**（retained-observation construction）
  2. **状态映射**（state mapping）
  3. **残差计数规则**（residual counting）
  4. **似然加权**（likelihood weighting）
- **推理时组合**（inference-time composition）：在采样过程中，通过 **Diffusion Posterior Sampling (DPS)** 将各源的似然因子动态组合，无需重新训练模型。

该方法将异构观测融合问题转化为**推理阶段的模块化接口设计问题**。

### 相比现有方法的优势
- **无需重训练**：同一预训练模型可复用于不同观测源组合，极大提升效率。
- **模块化与可审计性**：每个观测源的影响路径清晰、可检查、可修改。
- **灵活性强**：易于集成新的观测源，只需定义其接口并校准参数。
- **零样本能力**：在2019年开发、2020年无调优评估，验证了泛化性。

---

## 2. 核心实验方法和设置

### 数据集
使用三种异构的 **in situ** 观测数据，时间跨度为2019–2020年：

| 观测源 | 数据产品 | 覆盖范围 | 主要变量 |
|--------|---------|----------|----------|
| **Radiosonde (R)** | IGRA profiles | 全球 | `t2m`, `u10`, `v10`, `t500`, `u500`, `v500`, `z500`, `q500`, `t850`, `u850`, `v850`, `z850`, `q850` |
| **Aircraft (A)** | MADIS ABO reports | CONUS区域 | `t500`, `u500`, `v500`, `t850`, `u850`, `v850` |
| **Surface Station (S)** | MADIS METAR reports | CONUS区域 | `t2m`, `u10`, `v10` |

**参考真值**：使用 **ERA5** 再分析数据作为评估基准。

### 实验设置
- **研究区域**：**CONUS domain**（24°–50°N, 125°–66°W）
- **先验模型**：基于 **EDM框架** 预训练的13变量大气扩散模型（来自Chakraborty et al., 2026）
- **采样方法**：**Diffusion Posterior Sampling (DPS)**
- **接口开发与评估分离**：
  - **开发阶段**：使用2019年24个预设案例选择接口设计与参数。
  - **独立评估**：在2020年723个匹配分析时刻进行无调优评估。

### 评估配置（conditioning configurations）
| 配置 | 描述 |
|------|------|
| **R-only** | 仅使用探空观测（基线） |
| **R+A** | 探空 + 飞机观测 |
| **R+S** | 探空 + 地面站观测 |
| **R+A+S** | 三者联合 |

### 评估指标
- **确定性指标**：
  - **RMSE**（均方根误差）及其相对变化（%）
  - 分组评估：所有13变量、地表目标变量、高空温风变量
- **概率性指标**：
  - **CRPS**（连续排序概率评分）
  - **Spread-skill ratio**, **rank histogram**, **5th–95th percentile coverage**
- **不确定性估计**：
  - 使用 **14天移动块自助法**（moving-block bootstrap）处理时间相关性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（CONUS域，2020年）

#### 表1：分组平均RMSE降低（相对于R-only）

| 配置 | 所有13变量 | 地表目标变量 | 高空温风变量 |
|------|-----------|-------------|--------------|
| **R+A** | -4.46% | -2.28% | **-6.74%** |
| **R+S** | -5.35% | **-13.31%** | -2.99% |
| **R+A+S** | **-9.24%** | **-14.17%** | **-9.30%** |

> ✅ R+A+S在所有变量组上均显著优于单一补充源。

#### 进一步增益（相对于更强的单源补充）：
- 在**所有变量**上，R+A+S比R+S额外降低 **3.89个百分点**。
- 在**高空温风变量**上，比R+A额外降低 **2.56个百分点**。

---

### 与基线方法对比
- **R+A+S vs R-only**：
  - **RMSE降低9.24%**（所有13变量）
  - **CRPS降低9.98%**，表明改进不仅限于均值，也提升了概率预测技能。
- **全局影响**：
  - 在**全球非CONUS区域**，RMSE变化接近零，说明改进集中在观测直接影响区，未引入虚假全局信号。

---

### 消融实验与接口设计选择

#### （1）飞机观测接口设计消融
- **报告流选择**：排除TAMDAR后性能更优；ACARS direct、MDCRS/ARINC、Canadian AMDAR组合最佳。
- **压力匹配窗口**：±25 hPa优于±5 hPa。
- **残差表示**：**等单元均值**（equal-cell mean residuals）显著优于逐报告计数。

#### （2）地面站接口设计消融
- 同样，**等单元均值**表现最优，避免密集站点区域过度加权。

#### （3）似然参数校准
- 通过2019年开发集选择最优参数：
  - `λ_A = 0.4`, `λ_S = 0.4`
  - 最终R+A+S配置在2019年开发集上实现**11.961%**的RMSE降低。

---

### 持留观测评估（held-out evaluation）
在24个季节分布的2020案例中，随机保留80%观测用于条件化，20%用于评估：

| 评估源 | 条件化比较 | RMSE降低 |
|--------|------------|----------|
| **Surface (S)** | R+A+S (含80%S) vs R+A | **-13.50%** |
| **Aircraft (A)** | R+A+S (含80%A) vs R+S | **-11.71%** |

> ✅ 表明新增观测源能有效**插值预测**到未见的同类观测点。

---

## 4. 关键结论和发现

### 主要发现
1. **可组合接口有效提升重建精度**：
   - R+A+S在CONUS域内显著降低RMSE（**-9.24%**）和CRPS（**-9.98%**）。
   - 飞机与地面观测提供**互补增益**：飞机改善高空场，地面站改善近地面场。

2. **接口设计至关重要**：
   - 并非所有观测都“越多越好”——TAMDAR的加入反而降低性能。
   - **等单元均值聚合**优于逐报告计数，防止高密度区域主导似然。

3. **无需重训练即可组合多源观测**：
   - 同一预训练扩散模型可通过接口灵活支持多种观测组合，验证了**模块化先验复用**的可行性。

4. **改进具有统计稳健性**：
   - 所有13个变量的RMSE降低均在**14天移动块bootstrap**下显著（95%区间全为负）。
   - 改进不是由少数极端事件驱动。

---

### 局限性
- **依赖单一先验模型**：结论受限于所用的13变量ERA5扩散模型。
- **垂直分辨率有限**：飞机观测仅映射到500/850 hPa两个层次。
- **似然模型简化**：使用加权平方残差，未显式建模跨源或空间相关误差。
- **非线性修正**：共享的梯度裁剪操作使最终修正呈非线性。
- **机制诊断基于单案例**：图示分析仅为单次反向轨迹，不代表年度统计规律。

---

### 未来工作方向
- 测试更多**垂直层次**和**新型观测源**（如卫星、雷达）。
- 引入**相关误差建模**的似然函数。
- 设计**重复或空间结构化**的持留评估方案。
- 构建**全球分布的接口系统**。
- 探索其他**大气先验与采样器**的兼容性。
- 与经典**data assimilation系统**进行受控对比。

---

## 总结
本文提出了一种**模块化、无需重训练**的方法，通过**可组合的观测接口**，成功将异构的in situ观测（探空、飞机、地面站）融合到一个预训练的大气扩散模型中。实验表明，该方法在**CONUS区域**显著提升了大气场重建精度（RMSE ↓9.24%，CRPS ↓9.98%），且飞机与地面观测提供**互补性改进**。该框架为构建灵活、可解释、可扩展的**生成式大气数据同化系统**提供了新范式。

</details>

---

### 9. [On the second-order optimization for spiking neural networks](https://arxiv.org/abs/2609.29379)

**Authors**: Ngoc Phu Doan, Ihsen Alouani  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.29379v1  

#### Abstract
Spiking Neural Networks (SNNs) offer an energy-efficient alternative to conventional neural networks by exploiting sparse, binary spikes, and event-driven computation. However, the training of SNNs remains challenging, as spiking activations create a sharp loss landscape that hinders training, and d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*On the Second-Order Optimization for Spiking Neural Networks*

## 1. 论文的主要贡献和创新点

### 解决了什么问题  
Spiking Neural Networks (SNNs) 虽然在能效方面优于传统人工神经网络（ANNs），但由于其**非连续、离散的脉冲激活机制**，导致损失函数景观（loss landscape）极为尖锐且不规则。这使得基于一阶导数的优化器（如 SGD）收敛缓慢，泛化能力差。而主流的自适应优化器（如 Adam 及其变体）虽然引入了对角形式的二阶信息（即梯度平方的移动平均），但忽略了参数间的**交叉曲率信息（off-diagonal curvature）**，难以有效应对 SNN 中复杂的几何结构。

此外，现有的二阶优化方法（如 K-FAC）是为静态、前馈网络设计的，无法直接应用于具有**时间递归性、稀疏性和代理梯度（surrogate gradient）机制**的 SNN。

### 提出了什么新方法或新思路  
本文提出 **SpiKFAX** —— 一种专为 SNN 设计的、基于 Kronecker-factored 近似的二阶优化方法，用于近似 Fisher Information Matrix (FIM)，从而实现更有效的梯度预处理（preconditioning）。

#### 核心创新点：
- **首次将 Kronecker-factored FIM 近似扩展到 SNN 架构中**，考虑了 SNN 的三大特性：  
  1. 时间递归动态（temporal recurrence）  
  2. 代理梯度反向传播（surrogate-gradient-based backpropagation）  
  3. 脉冲稀疏性与事件驱动计算
- 引入三项关键假设以实现可计算性：
  - **层间块对角近似**（Layer-wise block diagonality）
  - **激活与梯度独立性假设**（Independent activations and derivatives）
  - **输入二阶矩的时间同质性**（Temporal homogeneity of input second moment），即用脉冲发放率向量 $ r $ 替代逐时刻相关结构
- 推导出适用于共享权重的时间卷积层的 Kronecker 分解形式：  
  $$
  \mathbf{F}_W \approx \mathbf{A} \otimes \mathbf{G},\quad \text{其中 } \mathbf{A} = \mathbb{E}[rr^\top],\ \mathbf{G} = \mathbb{E}\left[\left(\sum_t \delta_t\right)\left(\sum_s \delta_s\right)^\top\right]
  $$
  并给出高效的更新规则：$\Delta W = -\eta\, G^{-1} (\nabla_W \mathcal{L}) A^{-1}$

### 相比现有方法的优势  
- 相比 Adam/AdamW：能够捕捉参数之间的交互作用，提升训练稳定性和泛化性能；
- 相比标准 K-FAC 或精确 FIM：通过结构化近似显著降低计算复杂度，使其适用于大规模 SNN；
- 在多种模型架构和数据集上均表现出更强的鲁棒性和更高的最终准确率。

---

## 2. 核心实验方法和设置

### 使用的数据集  
实验涵盖 **7 个数据集**，分为两类：

| 类型 | 数据集 |
|------|--------|
| **静态图像数据集** | MNIST, F-MNIST, CIFAR10, CIFAR100 |
| **神经形态数据集（event-based）** | N-MNIST, CIFAR10-DVS, DVS128 Gesture |

这些数据集覆盖了从简单手写数字识别到复杂动态视觉任务的广泛场景。

### 实验设置和评估指标  
- **模型架构**：共测试 **5 种 SNN 架构**：
  - S-MLP
  - S-LeNet5
  - S-VGG11
  - S-VGG16
  - S-ResNet18
- **训练细节**：
  - 每组实验重复 5 次，报告均值 ± 标准差；
  - 学习率进行调优，选择最优配置；
  - 使用 SNNTorch 框架实现；
  - 硬件平台：NVIDIA RTX A5000 GPU，64GB RAM。
- **评估指标**：
  - 主要指标：**测试准确率（Test Accuracy）**
  - 辅助分析：训练损失曲线、测试准确率随 epoch 变化、训练稳定性、迭代耗时

### 基线方法对比  
与以下主流优化器进行比较：
- **SGD**（带动量）
- **Adam**
- **AdamW**

所有优化器均在同一训练流程下运行，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Fig. 2）

| Dataset | Best Baseline | SpiKFAX Result | Improvement |
|--------|---------------|----------------|-------------|
| **N-MNIST** | AdamW: 97.35% | **99.92%** | **+2.57%** |
| **CIFAR10-DVS** | Adam: 53.78% | **56.15%** | **+2.37%** |
| **DVS128 Gesture** | Adam: 71.78% | **76.13%** | **+4.35%** |

> 注：原文 Table 1 中部分数值可能存在排版错误，但趋势明确显示 SpiKFAX 全面领先。

在静态图像数据集上的表现同样优异（见 Fig. 2）：
- 在 CIFAR10 上，S-VGG16 使用 SpiKFAX 达到约 **98.5%** 准确率，显著高于其他优化器；
- 所有五种架构中，**SpiKFAX 均取得最高测试精度**，优势普遍在 1–3% 以上。

### 与基线方法的对比结果  
- **准确性全面超越**：在全部 7 个数据集、5 种架构中，SpiKFAX 均达到最佳或接近最佳性能；
- **训练稳定性更高**：
  - 如 Fig. 3 所示，SpiKFAX 在早期波动后迅速稳定，而 Adam/AdamW 持续震荡；
  - 训练损失下降更快，尤其在第 75 步之后出现“陡降”现象；
- **收敛速度更快**：
  - 在 DVS128-Gesture 上，SpiKFAX 在 **第 15 个 epoch 即达峰值准确率**，而 Adam/AdamW 需更多轮次且未完全收敛。

### 消融实验结果  
#### （1）学习率敏感性分析（Fig. 4A）
- SpiKFAX 在不同学习率下表现稳健，即使在极端小/大学习率下仍保持较高准确率；
- 对比之下，Adam 和 SGD 对学习率更敏感，性能波动较大；
- 表明 SpiKFAX 的自适应预处理机制增强了优化过程的鲁棒性。

#### （2）训练时间开销（Fig. 4B）
- 每次迭代耗时比 SGD 高约 **1.4 倍**，比 Adam/AdamW 高 **1.2 倍**；
- 但远低于理论上的完整 FIM 计算（后者不可行，趋于无穷）；
- 总体性价比高：**小幅增加单步时间换取大幅加速收敛与更高精度**。

---

## 4. 关键结论和发现

### 主要发现  
1. **SNN 的尖锐损失景观可通过二阶优化有效缓解**：  
   尖锐性并非不可克服的根本障碍，而是可以通过合适的曲率建模加以利用。
   
2. **SpiKFAX 显著提升了 SNN 的训练效率与泛化能力**：  
   在多个基准上一致优于 SGD、Adam 和 AdamW，验证了显式建模跨参数曲率的重要性。

3. **结构化近似使二阶优化在 SNN 中变得可行**：  
   通过 Kronecker 分解与时间统计聚合，成功将原本不可行的 FIM 近似转化为实用算法。

4. **训练稳定性增强**：  
   不仅提高最终精度，还减少训练过程中的震荡，有助于部署于资源受限设备。

### 方法的局限性  
- **计算开销仍高于一阶方法**：尽管已优化，但在边缘设备上实时应用仍有挑战；
- **依赖代理梯度框架**：当前推导基于 surrogate gradient，若采用无梯度训练方法则不适用；
- **假设限制**：如时间同质性假设可能在高度动态输入中失效；
- **暂未支持脉冲稀疏正则化等高级训练技巧**，未来可结合其他 SNN 特定策略。

### 未来工作方向  
- 将 SpiKFAX 扩展至更深的 SNN 架构（如 S-Transformer）；
- 探索低秩更新或异步更新机制以进一步降低内存与计算负担；
- 结合在线学习与片上训练（on-chip training），推动其在 neuromorphic hardware 上的实际部署；
- 研究如何融合 sharpness-aware minimization 或 sparsity-inducing 正则项，形成统一优化框架。

---

> ✅ **总结一句话**：  
> 本论文填补了 SNN 与二阶优化之间的空白，提出了首个面向脉冲网络的 Kronecker-factored Fisher 近似方法 SpiKFAX，在保持计算可行性的同时显著提升了训练稳定性与模型性能，为高效训练高性能 SNN 提供了新的路径。

</details>

---

### 10. [SLCA-GRPO: Resolving Cross-Segment Credit Misattribution in Tool-Calling RL](https://arxiv.org/abs/2609.29050)

**Authors**: Yan Zhan, Shaobo Liu, Qiunan Liu, Yuanjun Shi, Siqi Xu, WeiYi Hou, Xiang Xu, Zekang Li, Weizhou Pan, Jiahong Yan  
**Category**: cs.AI  
**Published**: 2026-09-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.29050v1  

#### Abstract
Tool-calling agents produce heterogeneous outputs, interleaving structured tool invocations with user-facing natural language summaries. This output heterogeneity presents a structural failure mode in standard on-policy Reinforcement Learning (RL): algorithms like GRPO indiscriminately broadcast a h...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SLCA-GRPO: Resolving Cross-Segment Credit Misattribution in Tool-Calling RL**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
在 **Tool-calling RL**（工具调用强化学习）中，标准的 on-policy RL 算法（如 GRPO）存在一个结构性缺陷：它们将整个轨迹的统一优势值（trajectory-level scalar advantage）广播到所有 token 上，无论其属于 **tool 调用段** 还是 **自然语言摘要段**。

这导致了 **Cross-Segment Credit Misattribution**（跨段信用错分）问题：
- **摘要段的奖励波动**（例如流畅的回复）会污染 **tool 决策 token 的梯度更新**。
- 即使 tool 调用错误或低效，只要最终摘要正确，也可能被强化（反之亦然），造成训练不稳定和性能下降。

### **提出了什么新方法或新思路**
论文提出 **SLCA-GRPO**，其核心是 **Segment-Locked Credit Assignment (SLCA)**，即“段锁定信用分配”。

#### **SLCA 的核心机制**：
- 将输出轨迹 $ y $ 分解为两个语义段：
  - $ y_{tool} $：包含推理链和结构化工具调用。
  - $ y_{sum} $：用户可见的自然语言摘要。
- 引入 **Hierarchical Rewards (HierR)**：
  - $ R_{tool} $：密集的执行奖励，评估工具调用的正确性和效率。
  - $ R_{sum} $：终端偏好奖励，评估最终答案的质量。
- 在优化时，**分别归一化并路由优势值**：
  - $ A_{tool} $ 只路由给 $ y_{tool} $ 的 token。
  - $ A_{sum} $ 只路由给 $ y_{sum} $ 的 token。
- 通过 **gradient masking** 和 **自动段分解** 实现，无需额外 rollout 或模型拆分。

此外，构建了 **Schema-Guided LLM Simulator (SGLS)** 作为训练基础设施，以支持可扩展的探索和稳定的训练。

### **相比现有方法的优势**
| 方法 | 局限性 | SLCA-GRPO 的改进 |
|------|--------|------------------|
| **GRPO/VinePPO/GiGPO** | 沿时间轴分配信用，但仍广播统一优势值，无法解决跨段污染。 | 在**结构轴**上隔离信用，从根本上阻断摘要对 tool 的梯度污染。 |
| **ToolPO** | 添加局部工具奖励，但 summary 噪声仍可通过全局奖励影响 tool token。 | 完全隔离，确保 tool 更新仅依赖 $ R_{tool} $。 |
| **RLTR** | 将 planner 和 summarizer 拆分为两个独立模型，放弃统一骨干网络。 | 在**单一统一策略**下实现信用分离，更简洁且易于部署。 |

**SLCA 是一种与时间信用分配方法正交的补充机制**，可组合使用。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **Toucan-1.5M**：用于 SFT 初始化和 RL 后训练，包含单轮和多轮工具调用轨迹。
- **Berkeley Function-Calling Leaderboard (BFCL) V3**：评估跨 schema 泛化能力。
- **t2-Bench**：评估在动态双控环境（航空、零售、电信）中的协作鲁棒性。

### **实验设置和评估指标**
- **主干模型**：Qwen2.5-3B/7B-Instruct, Qwen3-8B-Base。
- **训练设置**：
  - 使用 **SGLS** 模拟器进行 rollout，避免真实 API 的高成本和不稳定性。
  - **rollout group size G=16**，一次 RL epoch。
  - 控制变量严格，确保公平比较。
- **评估指标**：
  - **Toucan-Test**：`Success@0.9`（基于 `Sprocess` ≥ 0.9）、`Name F1`, `ArgMatch`, `Process`。
  - **BFCL**：Overall Accuracy。
  - **t2-Bench**：Pass1（任务成功率）。

### **基线方法对比**
- **SFT**：监督微调基线。
- **SFT+GRPO**：标准 GRPO，使用统一优势值。
- **ToolPO**：添加局部工具奖励的基线。
- **RLTR**：planner-summarizer 两阶段管道。
- **w/o SLCA / w/o SGLS / w/o HierR**：消融实验。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（7B 主干模型）**
| 方法 | Toucan-Test (Success) | BFCL (Acc) | t2-Bench (Pass1) |
|------|------------------------|------------|-------------------|
| **SFT+GRPO** | 76.60 ± 1.27 | 68.41 ± 0.11 | 31.87 ± 2.97 |
| **SLCA-GRPO (Ours)** | **79.13 ± 1.05** | **69.77 ± 0.47** | **41.02 ± 1.01** |
| **提升 (Δ)** | **+2.53 pp** | **+1.36 pp** | **+9.15 pp** |

在 3B 和 8B 模型上也观察到一致提升。

### **与基线方法的对比结果**
- **优于 GRPO**：在所有指标上均显著超越，证明 SLCA 有效缓解了信用错分。
- **优于 ToolPO**：ToolPO 在 7B 上表现崩溃（Success 仅 20.00），而 SLCA-GRPO 稳定提升。
- **优于 RLTR**：RLTR 因粗粒度奖励和冻结 summarizer 表现受限，SLCA-GRPO 在统一框架下取得更好效果。

### **消融实验结果**
| 消融条件 | Toucan-Test (Success) | BFCL (Acc) | t2-Bench (Pass1) |
|----------|------------------------|------------|-------------------|
| **w/o SLCA** (即 GRPO) | 76.60 ± 1.27 | 68.41 ± 0.11 | 31.87 ± 2.97 |
| **w/o SGLS** | 78.02 ± 1.12 | 68.75 ± 0.48 | 36.19 ± 1.95 |
| **w/o HierR** | 73.44 ± 1.21 | 69.07 ± 0.46 | 34.50 ± 1.44 |
| **SLCA-GRPO (完整)** | **79.13 ± 1.05** | **69.77 ± 0.47** | **41.02 ± 1.01** |

- **w/o SLCA**：移除 SLCA 导致性能大幅下降，验证其核心作用。
- **w/o SGLS**：性能下降，说明 SGLS 对稳定探索和泛化至关重要。
- **w/o HierR**：性能最差，证明密集的分层奖励是成功的关键。

---

## 4. **关键结论和发现**

### **主要发现**
1. **Cross-Segment Credit Misattribution 是 Tool-calling RL 的结构性失败模式**，源于优势值污染而非梯度方向冲突。
2. **SLCA 通过结构化解耦优势估计**，在不增加 rollout 成本的情况下，实现了更稳定、高效的优化。
3. **SLCA-GRPO 显著加速收敛**，并在多个基准上取得 SOTA 性能，同时降低工具冗余和成本。
4. **梯度诊断显示**：标准 GRPO 存在梯度不稳定性尖峰，而 SLCA-GRPO 梯度范数更平滑，支持其稳定性主张。

### **方法的局限性**
- **段边界假设**：要求 tool 调用和自由文本有明确边界，不适用于内联代码生成等场景。
- **奖励偏差**：
  - $ S_{process} $ 依赖于与黄金轨迹的匹配，可能低估有效的替代工具序列。
  - $ A_{sum} $ 的归一化未考虑不同后工具状态，存在分组偏差。
- **时间信用分配**：SLCA 不区分 tool 段内的动作顺序（如 Action 1 vs Action 2），需与 VinePPO 等方法结合。
- **规模限制**：实验最大使用 8B 模型，更大规模和真实 API 的对比仍是未来工作。

### **未来工作方向**
- 探索 **SLCA 与其他时间信用分配方法（如 VinePPO）的组合**。
- 开发 **无监督或弱监督的段分割机制**，以适应更复杂的输出格式。
- 在 **更大规模模型（如 70B+）和真实 API 环境** 下验证 SLCA 的有效性。
- 研究 **更鲁棒的执行奖励**，减少对黄金轨迹的依赖。

---

</details>

---

### 11. [Decoding Imagined Speech: A Strictly Subject-Independent Approach Using EEG](https://arxiv.org/abs/2609.29820)

**Authors**: Frederik M{\o}llskov Trier, Xiaopeng Mao, Sadasivan Puthusserypady  
**Category**: cs.AI  
**Published**: 2026-09-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.29820v1  

#### Abstract
Imagined speech decoding from electroencephalography (EEG) has gained increasing attention as a potential communication pathway for individuals with severe motor impairments, yet reported performance often relies on evaluation protocols that do not clearly reflect cross-subject generalization. This ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Decoding Imagined Speech: A Strictly Subject-Independent Approach Using EEG*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
- 当前基于 EEG 的 **imagined speech decoding** 研究中，许多报告的高性能依赖于“泄露”的评估协议（如未明确说明数据划分方式），尤其是缺乏对 **cross-subject generalization**（跨被试泛化）能力的真实评估。
- 尽管 Kumar 等人提出的公开数据集被广泛使用，但已有研究大多采用 subject-dependent 或不透明的数据划分方式，导致模型在实际 BCI 应用中的可推广性存疑。

### ✅ 提出的新方法与新思路
- 首次在 Kumar 的 imagined speech EEG 数据集上实施 **严格 subject-independent evaluation 协议**，确保训练和测试完全在不同被试之间进行，真实反映模型的跨个体泛化能力。
- 构建了一个**透明且可复现的基准框架**，用于未来方法比较。
- 对比了两种特征提取 pipeline：
  - **Kumar Pipeline**：基于时域统计特征（time-domain statistical features）
  - **Spectral Pipeline**：基于频域谱功率特征（frequency-domain spectral band-power features）

### ✅ 相比现有方法的优势
- 强调评估协议的严谨性，避免因数据泄露导致的性能高估。
- 揭示了在真实跨被试场景下，**spectral features 显著优于传统 time-domain features**。
- 通过 forward feature selection 发现，仅需少数几个频段即可达到接近最优性能，为轻量化 BCI 系统设计提供依据。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集
- **Kumar’s EEG dataset**（公开可用）
  - 包含 23 名被试（S1–S23），年龄 15–40 岁
  - 采集设备：Emotiv EPOC+，14 通道（AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4）
  - 采样率：原始 2048 Hz → 下采样至 128 Hz（公开版本）
  - 实验任务：想象三类刺激（coarse-grained classes）：
    - Characters（字母：A, C, F, ..., Y）
    - Digits（数字：0–9）
    - Images（日常物体：apple, car, dog, ...）
  - 每个 trial 持续 10 秒，共 690 条记录（23 被试 × 30 类别）

> 注：本文聚焦于 **coarse-level classification**（三分类任务：字符 vs 数字 vs 图像）

---

### ⚙️ 实验设置与评估指标

| 组件 | 设置 |
|------|------|
| **预处理共通步骤** | 中心截取 10s 片段；1s 滑动窗口（75% 重叠）；subject-wise z-score normalization |
| **交叉验证** | 5-fold subject-wise CV（见 Fig. 3）<br>使用 `StratifiedGroupKFold`，保证每个 fold 中类别平衡且无被试重叠 |
| **分类器** | Random Forest（RF）<br>- 树数量：200<br>- 其他参数使用 scikit-learn 默认值<br>- 使用 Gini index 分裂准则 |
| **预测聚合策略** | Window-level prediction → Trial-level majority voting（MV）→ Fold-level accuracy |
| **最终性能指标** | 平均 trial-wise accuracy（五折平均） |

---

### 🔍 基线方法对比

| Pipeline | 特征类型 | 特征维度 | 关键处理步骤 |
|--------|---------|----------|--------------|
| **Kumar Pipeline** | Time-domain statistical features | 56-D（14 ch × 4 feat） | 移动平均滤波 + 提取 sd, rms, sum, energy + log 变换（sd） |
| **Spectral Pipeline** | Frequency-domain band power | 70-D（14 ch × 5 band） | Notch + Bandpass 滤波 + Periodogram PSD 估计 + 绝对功率积分（delta, theta, alpha, beta, gamma）+ log 变换 |

> 两者均使用相同 CV 和 evaluation 流程，确保公平比较。

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（Coarse-level 分类）

| Pipeline | Accuracy (%) | Std Dev |
|--------|---------------|---------|
| **Spectral Pipeline** | **49.03** | ±4.18 |
| **Kumar Pipeline** | 37.97 | ±3.79 |

- **性能提升显著**：Spectral Pipeline 比 Kumar Pipeline 高出约 **11 个百分点**
- 差异具有统计学意义：**paired Wilcoxon signed-rank test, p = 0.018**，效应量大（rank-biserial correlation = 0.57）

---

### 🧪 类别级准确率分析（Table II）

| Class | Kumar Pipeline (%) | Spectral Pipeline (%) |
|-------|--------------------|------------------------|
| **Character** | 48.90 ±9.19 | **56.30 ±8.94** |
| **Digit**     | 36.90 ±8.70 | **46.00 ±10.24** |
| **Image**     | 29.40 ±14.16| **48.50 ±13.94** |

> 所有类别中，Spectral Pipeline 均表现更优，尤其在 “image” 类上有巨大提升（+19.1%）

---

### 🔍 消融实验：Forward Feature Selection（频段重要性分析）

逐步添加频段以最大化 trial-wise accuracy：

| Step | Band Set | Accuracy (%) |
|------|--------|--------------|
| 1    | α      | 46.67 ±6.93  |
| 2    | α + δ  | **51.17 ±5.66** |
| 3    | α + δ + γ | 51.03 ±6.58 |
| 4    | α + δ + γ + θ | **52.83 ±4.64** |
| 5    | Full bands (α+δ+γ+θ+β) | 49.03 ±4.18 |

#### 发现：
- **Alpha band 单独表现最好**（46.67%），可能与闭眼想象任务相关（alpha rhythm 增强）
- 加入 **delta band 后性能跃升**，表明低频成分对分类至关重要
- **四频段组合（α+δ+γ+θ）达到最高精度（52.83%）**
- 完整五频段模型反而略低，且与其他组合无显著差异（经 Bonferroni 校正后）

> 结论：**beta band 贡献有限，甚至可能引入噪声**

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. 在严格的 subject-independent setting 下，reported accuracy 远低于以往研究（如 Kumar 报告 85.2%），说明 **常见评估协议存在严重过拟合风险**。
2. **Spectral features（特别是 band power）比 time-domain statistical features 更具跨被试可迁移性**，更适合 imagined speech decoding。
3. **并非所有 frequency bands 都同等重要**：
   - **Alpha 和 delta bands 是最具判别性的频段**
   - 仅需少量频段即可实现接近最优性能，支持未来开发高效、低维 BCI 系统
4. 不同类别的 discriminability 存在差异：
   - 字符最容易识别（可能因其抽象性强、神经激活模式更一致）
   - 图像最难（语义复杂度高？视觉表征多样性大？）

---

### ⚠️ 局限性
- **未进行 artifact removal**：未去除眼电、肌电等干扰，可能影响信号质量。
- **未做 channel selection**：使用全部 14 通道，而 prior work 表明某些 task-relevant channels（如 occipital 区域）更具信息量。
- **仅使用绝对 band power**：未考虑相对功率、相位同步、功能连接等高级 spectral features。
- **未探索深度学习模型**：仅使用 RF，未能验证现代 DL 架构在此 setting 下的表现。

---

### 🔮 未来工作方向
1. 引入更先进的 artifact rejection 和 noise reduction 技术（如 ICA、DL-based denoising）。
2. 探索 **channel selection 与 cortical source localization**，识别关键脑区。
3. 扩展 spectral feature 集合（如 relative power, coherence, entropy measures）。
4. 在相同 subject-independent protocol 下评估 CNN、Transformer 等 deep learning 模型。
5. 推广至 fine-grained classification（例如区分具体字母或图像），并建立统一的 benchmarking platform。

---

## ✅ 总结一句话
> 本研究建立了首个在 Kumar EEG 数据集上 **严格 subject-independent 的 imagined speech decoding 基准**，证明 **spectral band-power features（尤其是 alpha 和 delta）在跨被试泛化中显著优于传统 time-domain 特征**，为未来可实用化的 BCI 系统提供了可靠的方法论基础和性能参考。

</details>

---

### 12. [MILO: Efficient Many-shot In-Context Learning with Block-wise Low-rank Compression](https://arxiv.org/abs/2609.29913)

**Authors**: Youpeng Zhao, Tian Tan, Liqian Peng, Jun Wang, Alec Go  
**Category**: cs.CL  
**Published**: 2026-09-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.29913v1  

#### Abstract
Many-shot in-context learning (ICL) enables large language models (LLMs) to adapt to complex tasks by conditioning on thousands of demonstration examples, but this paradigm shifts the inference efficiency bottleneck to the key-value (KV) cache memory. Due to the linear scaling behavior of the KV cac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MILO: Efficient Many-shot In-Context Learning with Block-wise Low-rank Compression

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **KV Cache 内存瓶颈**：在 many-shot In-Context Learning（ICL）中，模型需处理数千个示例，导致 Key-Value（KV）Cache 随序列长度线性增长，极大占用显存（如 8B 模型存储 90k 示例需额外 11.1GB），严重制约在线服务和端侧部署。
- **现有压缩方法不足**：已有工作聚焦于稀疏注意力（如 DBSA）、KV 缓存复用（如 Adapshot）或量化（如 KIVI），但未系统解决 KV Cache 的**存储效率**问题，尤其缺乏对 many-shot 场景下跨示例低秩冗余的利用。

### 🚀 提出的新方法：MILO
提出一种名为 **MILO**（**M**any-shot **I**n-context **L**earning with b**O**ck-wise low-rank compression）的高效框架，核心创新如下：

#### （1）**Block-wise Low-rank Compression (BLC)**  
- 将 KV Cache 按“块”（block）进行分组压缩，每块包含多个 many-shot 示例。
- 相比于：
  - **全局压缩**（global-based）：易丢失细粒度信息，解压开销大；
  - **逐示例/逐 token 压缩**（example-based）：累积误差严重，性能下降明显；
- BLC 在**压缩率**与**表示保真度**之间取得更优平衡。

#### （2）**Dynamic Rank Allocation Based on Entropy**
- 观察到不同 block 的信息密度异质性强（某些 block 包含关键推理链 CoT，某些则高度冗余）。
- 提出基于**信息熵**（information entropy）动态分配压缩秩（rank budget）：
  - 高熵 block（信息丰富）→ 分配更高秩；
  - 低熵 block（冗余多）→ 强压缩。
- 公式化为带约束的优化问题，通过贪心策略实现高效求解。

#### （3）**系统级优化支持**
- 实现定制化的 **Triton fused kernel**、**CUDA Streams 并行重建** 和 **CUDA Graph 执行流**，显著降低重建延迟和 CPU 调度开销。

### 🔍 相比现有方法的优势
| 方法 | 类型 | 局限性 | MILO 改进 |
|------|------|--------|----------|
| **ASVD** | Example-based SVD | 细粒度压缩导致误差累积 | 块级压缩减少误差传播 |
| **Palu** | Global low-rank | 忽视局部语义差异 | 动态秩分配适应异质性 |
| **DBSA / Adapshot** | Sparse Attention / Reuse | 不直接压缩 KV 存储体积 | 显著降低内存占用 |

> ✅ MILO 是首个专门针对 **many-shot ICL 中 KV Cache 存储效率**设计的低秩压缩框架。

---

## 2. 核心实验方法和设置

### 📚 数据集
涵盖自然语言理解与数学推理任务：
- **分类任务**：`Banking77`, `Clinic150`, `TREC`, `NLU`
- **数学推理**：`MathQA`, `SVAMP`

### ⚙️ 实验设置
- **模型**：`Qwen2.5-3B` 和 `Qwen2.5-7B`
- **精度**：FP16
- **硬件**：单张 NVIDIA L4 GPU（24GB），Intel Xeon CPU + 512GB DRAM
- **上下文长度**：支持 up to 512-shot 输入
- **检索器**：使用标准 BM25 进行示例选择
- **框架基础**：基于 FlexGen、Transformers、PyTorch 和 Triton 实现

### 📊 评估指标
| 指标类别 | 具体指标 |
|---------|--------|
| **模型性能** | 准确率（Accuracy），平均 across 多个 benchmark |
| **系统性能** | KV Cache 内存占用、端到端吞吐量（Throughput）、延迟 |
| **压缩效果** | KV Cache 压缩比（Reduction Ratio） |

### 🆚 基线方法对比
| 基线 | 描述 |
|-----|------|
| **Full Cache** | 无压缩，作为准确率上限与效率下限 |
| **ASVD** (Yuan et al., 2023) | 按示例粒度进行激活感知 SVD 压缩 |
| **Palu** (Chang et al., 2024) | 全局低秩投影矩阵压缩 KV Cache |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & Figure 4）

#### （1）**准确率表现**
- 在 **50% 压缩率**下：
  - `Qwen2.5-3B` 上，MILO 平均准确率达 **0.647**，接近 Full Cache（0.677），优于 ASVD（0.605）和 Palu（0.637）
  - `Qwen2.5-7B` 上，MILO 达 **0.725**，显著优于 ASVD（0.680）和 Palu（0.700）
- **最大提升达 +4.5%** 相比 SOTA 低秩方法。

#### （2）**系统性能与吞吐量**
- **KV Cache 内存减少 50%**
- **端到端吞吐量提升最高达 1.8×**
- 在 KV offloading 设置下（缓存驻留 CPU 内存）：
  - 相比 Full Cache，平均提速 **1.6×**
  - 对 7B 模型 + 512-shot 输入，可达 **3.2× 吞吐提升**

#### （3）与基线方法对比
| 方法 | 吞吐优势 | 准确率优势 |
|------|--------|----------|
| vs. Palu | ↑ 最高 37% | ↑ 一致领先 |
| vs. ASVD | ↑ 最高 60% | ↑ 显著更优 |

> 💡 特别地，**模型越大，MILO 提升越显著**，因其 KV Cache 更大，内存压力更重。

### 🔬 消融实验结果（Figure 5）

#### （1）**压缩率影响**
- 所有方法随压缩率上升而准确率下降；
- MILO 在整个范围内始终优于 ASVD 和 Palu，验证其**更强的信息保留能力**。

#### （2）**块大小（Block Size）影响**
- 最优块大小约为 **32 示例/块**；
- 过小 → 无法捕获跨示例冗余；
- 过大 → 降低检索与压缩粒度，增加重建开销。

#### （3）**系统优化有效性**
| 优化项 | 加速比（vs. PyTorch baseline） |
|-------|-----------------------------|
| Fused Triton Kernel | 2.5× |
| + CUDA Streams | 3.3× |
| + CUDA Graph | **3.7×** |

#### （4）**秩分配策略对比**
- **Entropy-based > Uniform > Random**
- 自适应分配能更有效利用有限秩预算，在高 shot 数下仍保持领先。

#### （5）**与量化结合（Table 2）**
| 方法 | KV Reduction | Accuracy (`Qwen2.5-7B`) |
|------|--------------|------------------------|
| Full Cache | 1× | 0.753 |
| MILO | 2× | 0.725 |
| MILO + INT8 | 4× | 0.712 |
| MILO + KIVI | **5.3×** | 0.701 |

> ✅ 表明 MILO 可与量化正交结合，实现**叠加式内存压缩**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Many-shot ICL 中 KV Cache 具有强低秩特性**，尤其在 key cache 中更为显著（Figure 1 & 2）。
2. **更多示例带来更多信息冗余**，反而为低秩压缩提供了更大空间。
3. **信息密度在 block 间高度异质**，统一压缩策略次优，必须动态适配。
4. **块级压缩（BLC）是平衡效率与性能的理想粒度**。
5. **系统级优化至关重要**：仅算法改进不足以释放全部潜力，需融合 kernel fusion、并行重建等工程手段。

### ⚠️ 方法的局限性
- 当前依赖 SVD 分解，有一定计算开销（虽可在预编码阶段完成）；
- 动态秩分配依赖熵估计，可能对噪声敏感；
- 尚未在超大规模模型（如 70B+）或真实生产流量中验证稳定性；
- 对极端长上下文（>1M tokens）的支持有待进一步测试。

### 🔮 未来工作方向
- 探索免 SVD 的轻量级秩估计方法（如随机投影）；
- 结合稀疏注意力（如 H2O）与低秩压缩，构建混合压缩架构；
- 扩展至多模态 ICL 场景；
- 在边缘设备（Edge/IoT）上部署 MILO，推动 on-device many-shot learning。

---

## 总结
> **MILO 通过块级低秩压缩 + 动态秩分配 + 系统协同优化，首次系统性解决了 many-shot ICL 中 KV Cache 的存储瓶颈问题，在几乎无损准确率的前提下实现了 50% 内存压缩和高达 1.8× 的吞吐提升，为长上下文大模型的实际部署提供了高效可行的技术路径。**

</details>

---

### 13. [Resource-Aware Model Selection for Scalable Indoor Localization on HPC Platforms](https://arxiv.org/abs/2609.29402)

**Authors**: Fukuharu Tanaka, Hamada Rizk, Moustafa Youssef, Hirozumi Yamaguchi  
**Category**: cs.DC  
**Published**: 2026-09-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.29402v1  

#### Abstract
Large-scale indoor localization is increasingly needed in campuses, smart buildings, factories, and digital-twin infrastructures, where wireless conditions, access-point deployments, and spatial layouts evolve over time. Such systems must be accurate, extendable, and maintainable, allowing new build...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Resource-Aware Model Selection for Scalable Indoor Localization on HPC Platforms

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于深度学习的室内定位系统通常采用**单体模型（monolithic model）**，即用一个全局模型覆盖整个部署区域。这种设计在环境扩展（如新增建筑、楼层）或局部无线条件变化时存在严重缺陷：
- 需要对整个模型重新训练，成本高昂；
- 推理开销随空间规模增长而急剧上升；
- 缺乏可维护性和可扩展性。

此外，模块化方法虽然提升了可维护性，但引入了新的挑战：当环境中存在数百甚至上千个局部模型时，对每个查询进行**穷举推理（exhaustive inference）**会导致极高的计算、内存驻留、模型加载和调度开销。

### 🚀 提出的新方法与思路
本文提出了一种**资源感知的模块化推理框架（resource-aware modular inference framework）**，将大规模WiFi指纹室内定位建模为一个**大规模预训练模型集合上的模型选择问题**。其核心思想是：
- 将环境按**building-floor-spot**层次结构分解；
- 每个空间单元训练独立的**autoencoder模型**作为“异常检测器”；
- 利用**重构误差（reconstruction error）**衡量输入指纹与各模型的匹配程度；
- 在推理阶段通过两种轻量级剪枝策略动态减少需执行的候选模型数量。

#### 主要创新点：
1. **Hierarchical Candidate Pruning (HCP)**  
   自顶向下的粗到细模型选择机制：
   - 先预测最可能的 building；
   - 再在其下属 floors 中筛选；
   - 最后仅评估对应 spots 的局部模型。
   
2. **Trajectory-Aware Pruning (TAP)**  
   利用用户移动的**时间局部性（temporal locality）**，限制候选集为前一次估计位置的空间邻域，显著缩小搜索范围。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可扩展性** | 支持增量添加新建筑/楼层/房间，无需重训全局模型 |
| **可维护性** | 局部更新只需重新训练受影响的小范围模型 |
| **推理效率** | 显著降低模型执行数、内存占用和延迟 |
| **资源适配性** | 特别适用于HPC和分布式平台，优化了模型加载与并行调度 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用公开真实世界数据集：**UJIIndoorLoc** [37]
- 覆盖3栋建筑，约108,703 m²
- 包含933个参考点，由20+用户使用25台Android设备采集
- 输入特征：520维RSSI向量（来自WAPs）
- 输出目标：SpaceID（spot）、floor、BuildingID

> 注：原始验证集不含SpaceID，因此作者将原训练集以8:2比例划分为新的训练/测试集，并按spot级别划分以保证每类有足够的样本用于局部模型训练。

### ⚙️ 实验设置
- **模型架构**：全连接自编码器（AE），Encoder: 200-300-400-500；Decoder: 500-400-300-200
- **训练参数**：
  - Batch size: 256
  - Optimizer: Adam (lr=1e-3)
  - Dropout: 0.2
  - Early stopping patience: 8 epochs
- **推理流程**：
  1. Preprocessing → 2. Candidate Selection (HCP/TAP) → 3. Reconstruction-Based Localization

### 🎯 评估指标
| 类型 | 指标 |
|------|------|
| **准确性** | - Top-1 / Top-3 / Top-5 准确率<br>- 平均距离误差（m）<br>- Building-level 和 Floor-level 准确率 |
| **效率** | - 每次查询执行的模型数量<br>- 单次推理延迟（ms）<br>- 多节点CPU/GPU环境下的并行可扩展性 |

### 🔁 基线方法对比
- **Exhaustive Baseline**：评估所有735个spot模型，取最低重构误差者为预测结果
- **TAP的初始化**：首次预测仍使用exhaustive方式获取初始位置，后续利用轨迹信息剪枝

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 方法 | 执行模型数 | Top-1 Acc | 距离误差（m） | Floor Acc | Building Acc |
|------|------------|-----------|----------------|------------|---------------|
| Exhaustive | 735 | 0.631 | 5.182 | 0.9855 | ~1.0 |
| HCP | **67** | **0.638** | **4.04** | **0.9855** | ~1.0 |
| TAP | **10** | 0.615 | 5.11 | 0.9855 | ~1.0 |

> ✅ **HCP不仅精度略高于基线，且平均距离误差更低（4.04m vs 5.18m）**

### 🔻 与基线方法对比结果
- **模型执行数大幅下降**：
  - HCP：从735 → **67**（↓90.9%）
  - TAP：从735 → **10**（↓**98.6%**）
- **延迟显著降低**（Fugaku超算，16节点CPU）：
  - Exhaustive: 30.04 ms
  - HCP: **4.48 ms**
  - TAP: **1.07 ms**
- **受限内存环境下优势更明显**：
  - 当VRAM只能容纳100个模型时，频繁加载导致exhaustive方法延迟极高；
  - HCP/TAP因模型调用少，避免大量host-device传输，节省显著时间。

### 🔬 消融实验结果
#### （1）TAP中邻居数量的影响（Figure 7 & 8）
- 使用10个最近邻时达到最佳性价比：
  - 更少则易遗漏真值（准确率下降）；
  - 更多则收益递减，计算成本上升。
- 结论：**K=10 是短期跟踪中的理想平衡点**。

#### （2）递归推理鲁棒性测试（Figure 9 & 10）
- 随着递归步数增加，TAP精度逐渐下降（误差累积）；
- 使用更大邻域（如20–30个spots）可缓解退化，提高长期稳定性；
- 表明：**小邻域适合短时追踪，大邻域更适合长序列连续定位**。

#### （3）执行时间分解（Figure 11 & 12）
- **Model execution（exec）主导总延迟**，尤其在高并发场景下；
- HCP/TAP直接压缩exec环节，效果跨平台通用（GPU/CPU/HPC）；
- 多节点扩展性良好：随着节点数增加，延迟近似线性下降。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **模块化 + 资源感知剪枝 = 可扩展定位的新范式**
   - 将定位转化为“模型选择”而非“端到端分类”，兼顾精度与效率。
2. **HCP显式建模空间层级结构，提升垂直维度判别能力**
   - 在floor-level accuracy上表现最优，说明分层建模有效捕捉建筑立体特征。
3. **TAP充分利用人类移动的时间连续性，实现极致剪枝**
   - 仅需评估10个模型即可保持接近基线的精度，适用于实时跟踪任务。
4. **剪枝不仅省算力，还优化系统级资源使用**
   - 减少模型加载、缓存替换、通信开销，在HPC和边缘环境中均有巨大潜力。
5. **方法具备强并行可扩展性**
   - 在Fugaku超算上实现毫秒级延迟（低至1.07ms @16 nodes），适合大规模并发服务。

### ⚠️ 方法的局限性
- **依赖历史位置信息**：TAP要求有可靠的前序估计，冷启动或信号中断后需回退至HCP/exhaustive；
- **空间邻域定义依赖坐标先验**：需要已知spot centroid，若无精确标注则难以构建邻接关系；
- **未考虑多用户竞争资源的情况**：当前实验为单用户轨迹模拟，实际系统中可能存在资源争抢；
- **仅验证于WiFi RSSI**：虽具通用性，但在CSI或其他modalities上的迁移效果待验证。

### 🔮 未来工作方向
1. **多用户场景下的缓存管理策略研究**
   - 如何在共享GPU/HPC资源下高效缓存常用模型？
2. **动态资源分配机制**
   - 根据负载自动调整HCP/TAP参数或切换策略。
3. **结合联邦学习或增量学习进一步增强可维护性**
   - 支持在线更新模型而不影响整体服务。
4. **拓展至其他sensor modalities（如BLE, UWB, CSI）**
   - 验证框架在异构信号输入下的普适性。

---

> 💡 **一句话总结**：  
> 本论文提出了首个面向HPC平台的大规模模块化室内定位框架，通过**Hierarchical Candidate Pruning**和**Trajectory-Aware Pruning**两大轻量剪枝策略，在几乎不损失精度的前提下，将模型执行数从735降至67（HCP）甚至10（TAP），推理延迟压缩达98%以上，为构建**可扩展、可维护、高性能**的下一代智能建筑定位服务提供了可行路径。

</details>

---

### 14. [GBFRVFL: Granular-Ball Computing-Based Fuzzy Random Vector Functional Link Network](https://arxiv.org/abs/2609.29670)

**Authors**: A. Quadir, A. Rahaman, P. N. Suganthan, M. Tanveer  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.29670v1  

#### Abstract
In practical machine learning tasks, data are often contaminated with noise, outliers, and class imbalance, which can degrade the performance of conventional models. While random vector functional link (RVFL) networks offer fast training and strong generalization, they do not explicitly handle uncer...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GBFRVFL: Granular-Ball Computing-Based Fuzzy Random Vector Functional Link Network**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
在实际机器学习任务中，数据常受到**噪声（noise）**、**离群点（outliers）** 和**类别不平衡（class imbalance）** 的影响，这些因素会显著降低传统模型（如 RVFL）的性能。尽管 RVFL 具有训练速度快、泛化能力强的优点，但它对所有样本赋予相同权重，缺乏对不确定性建模的能力。

此外，现有的 granular-ball 方法虽然通过聚合样本来提升鲁棒性，但仍容易受极端样本干扰，且未有效结合自适应的成员度分配机制来处理模糊性和分布不确定性。

### 🚀 提出的新方法与新思路
本文提出了一种基于粒计算（Granular Computing）的新型随机向量函数链接网络框架——**GBFRVFL**（Granular-Ball Computing-Based Fuzzy Random Vector Functional Link Network），并设计了两种变体：

- **F-GBRVFL**：引入**模糊成员度（fuzzy membership）** 来衡量每个 granular ball 的可靠性。
- **SDAP-GBRVFL**：提出一种全新的**统计密度自适应毕达哥拉斯成员度（Statistical Density-Adaptive Pythagorean Membership, SDAPM）** 方案，动态调整 membership 与 non-membership 值。

SDAPM 综合考虑以下三个局部统计特性：
- 类内方差（class-wise variance）
- 局部稀疏性（local sparsity）
- granular ball 的紧凑性（compactness）

该机制使模型能更精细地区分核心球与边界球，增强抗噪能力。

### 🔍 相比现有方法的优势
| 特性 | 优势 |
|------|------|
| **鲁棒性强** | 利用 granular ball 抽象减少噪声和离群点的影响；通过自适应成员度加权进一步抑制异常区域贡献 |
| **不确定性建模能力强** | 引入模糊与毕达哥拉斯成员系统，显式建模 granular ball 的可信度与不可信度 |
| **高效性保留** | 继承 RVFL 的闭式求解机制，无需迭代优化，保持快速训练特性 |
| **适用于复杂场景** | 对 label noise、类别不平衡等挑战具有更强容忍力 |

---

## 2. **核心实验方法和设置**

### 📚 数据集
实验在 **37 个公开基准数据集** 上进行，涵盖：
- **UCI 数据集**（如 `breast cancer`, `ionosphere`, `heart` 等）
- **KEEL 数据集**（用于不平衡分类的经典集合）

并在部分实验中加入人工标签噪声（5%~40%），以验证模型在**噪声环境下的鲁棒性**。

### ⚙️ 实验设置
- **数据划分**：70% 训练，30% 测试
- **超参数调优**：采用五折交叉验证 + 网格搜索
- **正则化参数范围**：$\lambda \in \{10^{-5}, ..., 10^5\}$
- **隐藏节点数 $h$**：从 3 到 203 范围内选择
- **激活函数**：测试了 9 种，包括 SELU、ReLU、Sigmoid、Sine、Hardlim、Tribas、Radbas、Sign、Leaky ReLU

### 📊 评估指标
- **分类准确率（Accuracy, Acc）**
- **平均排名（Average Rank）**
- 非参数统计检验：
  - **Friedman 检验**（检测性能差异是否显著）
  - **Nemenyi 后验检验**（成对比较模型间差异）

### 🆚 基线方法对比
与以下六种主流模型进行比较：
1. **RVFL** [8]
2. **ELM** [9]
3. **GB-RVFL** [27]
4. **GE-GB-RVFL** [27]
5. **CRVFL** [30]
6. **ACRVFL** [30]

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（Clean Data）

| 模型 | 平均 Accuracy | 平均 Rank |
|------|----------------|------------|
| **SDAP-GBRVFL**（本文） | **84.46%** | **2.07** ✅（最优） |
| **F-GBRVFL**（本文） | 82.29% | 3.20 |
| RVFL | 81.55% | 4.51 |
| ELM | 80.81% | 5.22 |
| GB-RVFL | 79.62% | 4.86 |
| GE-GB-RVFL | 78.17% | 5.16 |
| CRVFL | 75.86% | 5.72 |
| ACRVFL | 76.62% | 5.26 |

> ✅ **SDAP-GBRVFL 在平均 Acc 和平均 Rank 上均排名第一**，显著优于所有基线模型。

### 📉 噪声环境下表现（Noisy Labels）

在添加 **5%-40% 标签噪声** 的五个代表性数据集上进行了鲁棒性测试（见 Supplementary Material Table S.I）：

| 模型 | Overall Average Acc (含噪声) |
|------|-------------------------------|
| **SDAP-GBRVFL** | **79.72%** ✅ |
| **F-GBRVFL** | 78.23% |
| GB-RVFL | 75.96% |
| RVFL | 75.92% |
| ELM | 74.18% |

> ✅ 即使在高达 40% 的标签错误下，**SDAP-GBRVFL 仍保持稳定性能**，尤其在 `cleve` 和 `ecoli` 数据集上远超其他模型。

### 🧪 统计显著性分析
- **Friedman 检验**：$F_F = 12.38 > 临界值\ 2.05$ → 拒绝零假设，表明模型间存在显著差异。
- **Nemenyi 检验**（C.D. = 1.72）：
  - SDAP-GBRVFL 与所有基线模型的 rank 差异均大于 C.D. → 性能提升**统计显著**
  - F-GBRVFL 显著优于除 RVFL 外的所有基线

### 🔍 消融实验与敏感性分析（Supplementary）
- **激活函数敏感性**：SDAP-GBRVFL 在不同激活函数下波动更小，表现出更强稳定性。
- **超参数敏感性（D 和 N）**：SDAP-GBRVFL 在较宽的正则化参数和隐藏节点范围内均保持高性能，说明其配置鲁棒性强。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **granular ball + 自适应成员度 = 更强鲁棒性**
   - 将 granular ball 抽象与 fuzzy / SDAP 成员度结合，可有效缓解噪声、离群点和类别不平衡带来的负面影响。
2. **SDAPM 是关键创新**
   - 动态利用类方差、局部稀疏性和球紧凑性构建 membership，使得模型能够智能区分高置信与低置信区域。
3. **SDAP-GBRVFL 综合性能最佳**
   - 不仅在干净数据上领先，在噪声环境中也展现出卓越的稳定性与泛化能力。
4. **无需牺牲效率换取性能**
   - 保持了 RVFL 的闭式求解优势，计算开销可控，适合大规模应用。

### ⚠️ 方法的局限性
- 当前仅应用于**浅层 RVFL 架构**，尚未扩展至深层或图结构网络。
- granular ball 的生成依赖于聚类过程（如 2-means 分裂），可能在高维空间中面临“维度灾难”。
- SDAPM 设计较为复杂，解释性略低于简单模糊策略。

### 🔮 未来工作方向
1. 扩展到 **deep GBFRVFL** 或 **ensemble GBFRVFL** 框架
2. 探索 **multi-granularity fusion** 策略，融合多尺度 granular ball 表示
3. 应用于 **real-world 应用场景**，如医疗诊断、金融风控、图像识别中的噪声标签学习
4. 开发 **可解释性模块**，可视化 membership 分布以辅助决策

---

> 🔗 **代码与补充材料开源地址**：  
> https://github.com/mtanveer1/GBFRVFL

--- 

✅ **总结一句话**：  
本文提出的 **GBFRVFL 框架**，特别是 **SDAP-GBRVFL** 模型，通过将 granular-ball computing 与新型自适应毕达哥拉斯成员度相结合，在不牺牲训练效率的前提下，实现了对噪声、离群点和类别不平衡的高度鲁棒性，是 randomized learning 与 granular computing 融合的一次成功探索。

</details>

---

### 15. [Pistis Technical Report](https://arxiv.org/abs/2609.28554)

**Authors**: Heyun Chen, Xiaohan Lan, Jiaxi Li, Zhilin Lu, Qi She, Weiwen Xu, Fei Yu, Yujie Zhong, Jinghuan Chen, Zijian Feng, Siyu Jiao, Yiheng Lin, Xinhao Wang, Sihan Yang, Jieyu You, Changbin Zhang, Hengyu Zhang, Xudong Zhang, Yunqing Zhao, Shuai Zheng  
**Category**: cs.AI  
**Published**: 2026-09-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.28554v1  

#### Abstract
We introduce the Pistis model family, comprising 27B- and 9B-parameter multimodal large language models built on Qwen3.6 and Qwen3.5, respectively, and developed through a general and scalable post-training framework. The framework first establishes a strong foundation through large-scale multimodal...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Pistis Technical Report》核心结论与实验结果总结

---

## 1. 主要贡献和创新点

### 解决的问题
当前多模态大语言模型（MLLMs）在后训练（post-training）阶段存在以下瓶颈：
- **监督微调（SFT）与强化学习（RL）分离**：传统流程将 SFT 和 RL 分为独立阶段，导致二者优势无法协同发挥。
- **策略熵崩溃（entropy collapse）**：纯 RL 训练容易导致输出多样性下降，限制探索能力。
- **长程任务信用分配困难**：对于需要多步推理、工具调用的 agentic 任务，仅依赖最终奖励难以准确归因中间步骤的有效性。

### 提出的新方法与新思路

#### （1）**Interleaved Distillation and Reinforcement Learning (IDRL)**  
一种新型后训练范式，**交替执行 on-policy distillation（OPD）与 RL 更新**，而非串行或静态加权联合优化。

- **核心机制**：
  - 在每个训练周期内交替进行 OPD 和 RL 梯度更新。
  - 利用强教师模型提供密集的 token-level 监督信号（OPD），同时通过 RL 优化任务级奖励。
  - 引入 **Positive-Advantage Suppression (PAS)** 技术，在长程轨迹中抑制无效中间动作获得正向信用。

- **优势**：
  - 避免梯度冲突：交替更新避免了联合损失函数中的目标干扰。
  - 维持策略熵：OPD 的“质量覆盖”特性防止 RL 导致的熵坍缩。
  - 更优信用分配：PAS 改善了对中间行为的学习效率。

#### （2）**Pistis-Auto-Harnessing (PAH)**  
一个系统级自动化框架，用于优化推理时的 agent harness（如提示词、状态管理、工作流等），而**不改变模型参数**。

- **核心机制**：
  - 由一个 Optimization Agent 在开发集上迭代提出、验证并接受 harness 修改。
  - 包括 **Candidate Ledger**（候选实体记录）、**Search Skills**（可复用检索策略）、自适应工作流等组件。
  - 最终冻结最优 harness 进行测试。

- **优势**：
  - 实现模型与系统的解耦优化。
  - 可迁移至其他模型（如 GPT-5.5）仍能提升性能。
  - 提升证据组织与决策一致性。

#### （3）**Pistis 模型家族**
基于 Qwen3.5 和 Qwen3.6 构建两个规模（9B 和 27B）的多模态模型，并衍生两种专业化变体：
- **Pistis-Thinking**：专注于深度多模态推理。
- **Pistis-Agentic**：额外引入 agentic 轨迹数据，支持长程规划、迭代推理与工具使用，尤其擅长 multimodal search。

---

## 2. 核心实验方法和设置

### 数据集

#### 多模态理解与推理基准（共 30 项）
| 类别 | 数据集 |
|------|-------|
| STEM/数学 | MathVista-mini, MathVerse-mini, Geo3K |
| 视觉问答 | MM-Vet, HallusionBench, AI2D-test |
| 文档理解 | DocVQA, InfoVQA, ChartQA, OCRBench |
| 空间定位 | RefCOCO+, Charades-STA |
| 视频理解 | MVBench, MLVU, TempComp |

#### Agentic 与搜索能力基准（共 18 项）
| 类别 | 数据集 |
|------|-------|
| 工具集成推理 | TreeBench, LogicVista |
| 多模态搜索 | MMSearch, BrowseComp-VL, VDR-testmini |
| 实时视觉问答 | LiveVQA |
| Claw-Style 交互 | PinchBench |
| 高分辨率感知 | HRBench4K/8K, MME-RealWorld-Lite |

#### 自建业务专用基准
- **Pistis Benchmark**：约 10,000 条高质量样本，聚焦内容安全与商业合规场景，涵盖图像与视频输入，强调政策对齐、细粒度语义判断与跨模态推理。

### 实验设置与评估指标

| 项目 | 设置说明 |
|------|--------|
| **模型基础** | Pistis-9B 基于 Qwen3.5-9B；Pistis-27B 基于 Qwen3.6-27B |
| **训练流程** | SFT → IDRL（仅 9B）或 Pure RL（27B） |
| **IDRL 参数** | `SoPD=5`, `SRL=5`, `S*=300`（9B Thinking）或 `100`（9B Agentic） |
| **评估环境** | 统一使用 PistisEvalKit，兼容 VLMEvalKit，控制推理配置一致 |
| **推理模式** | 温度 0.0，top_p 1.0，最小化采样方差 |
| **评估指标** | 准确率（Accuracy）、IoU、平均得分（Average Score） |

### 基线方法对比
- **基础模型**：Qwen3.5-9B, Qwen3.6-27B, Qwen3.8-27B
- **同类模型**：Step3-VL-10B, Keye-VL-1.5, InternVL3.5, Thyme-7B
- **消融对比**：
  - Pure RL
  - Pure OPD
  - Sequential OPD→RL
  - Joint RL+OPD（静态加权）
  - w/ vs w/o PAS

---

## 3. 主要实验结果和性能指标

### 性能总览（来自 Table 4 & 5）

| 模型 | Non-Grounding Avg. | Grounding Avg. | Overall Agentic Avg. |
|------|---------------------|----------------|------------------------|
| **Pistis-27B-Thinking** | 82.3 | **80.5** | — |
| Qwen3.8-27B | 82.4 | 71.9 | — |
| Qwen3.6-27B | 82.0 | 76.4 | — |
| **Pistis-9B-Thinking** | **80.6** | **80.1** | — |
| Qwen3.5-9B | 80.0 | 73.0 | — |
| **Pistis-27B-Agentic** | — | — | **78.3** |
| Qwen3.6-27B | — | — | 75.7 |
| **Pistis-9B-Agentic** | — | — | **74.7** |
| Qwen3.5-9B | — | — | 71.5 |

> ✅ Pistis 在 grounding 任务上显著领先，且 agentic 版本全面超越对应 base model。

### 关键单项性能提升（vs Base Model）

| 任务 | Pistis-27B-Agentic ↑ | Pistis-9B-Agentic ↑ |
|------|------------------------|----------------------|
| BrowseComp-VL | +8.8 pts | +8.4 pts |
| MMSearch | +3.3 pts | +8.0 pts |
| VDR-testmini | +3.2 pts | +3.2 pts |
| LiveVQA | +9.7 pts | +11.4 pts |
| PinchBench | +0.8 pts | +2.8 pts |
| TreeBench | +8.7 pts | +3.0 pts |

> 🔍 表明 **Pistis-Agentic 在多模态搜索与复杂交互任务上有显著优势**。

### Pistis-Auto-Harnessing (PAH) 效果

| Harness | VDR-testmini Acc. | Δ |
|--------|--------------------|----|
| Baseline Harness (15 interactions) | 26.8% | — |
| **Optimized Harness (same budget)** | **28.6%** | **+1.8 pts** |

- 平均交互次数几乎不变（8.68 vs 8.60），说明性能提升源于更高效的资源利用。
- **零样本迁移效果**（Same harness, no tuning）：
  - MMSearch: +0.4 pts
  - BrowseComp-VL: +1.7 pts
  - LiveVQA: +1.0 pts
  - Seed-2.1-turbo: +1.4 pts
  - GPT-5.5: **+4.0 pts**

> 🚀 显示 PAH 所学的 harness 结构具有良好的泛化能力。

### 消融实验结果（Table 8 & 9）

#### IDRL 消融（Pistis-9B-Agentic）

| 方法 | Overall Avg. |
|------|---------------|
| SFT | 73.7 |
| Pure RL | 74.0 |
| Pure OPD | 73.4 |
| Sequential OPD→RL | 74.1 |
| Joint RL+OPD | 74.1 |
| **IDRL (proposed)** | **74.7** ✅ |

> IDRL 在所有类别中均表现最佳，尤其在 Search-Oriented (+1.9 vs seq.) 上优势明显。

#### 正优势抑制（PAS）消融

| 方法 | Overall Avg. | Δ |
|------|--------------|----|
| IDRL (with PAS) | 74.7 | — |
| IDRL (w/o PAS) | 74.3 | -0.4 |

- 影响最大的是长程任务：
  - Search-Oriented: -1.2 pts
  - Multimodal Reasoning: -0.9 pts
  - PinchBench: -0.8 pts

> 证实 PAS 对改善长程信用分配至关重要。

---

## 4. 关键结论和发现

### 主要发现

1. **IDRL 是一种稳定且高效的多目标训练范式**：
   - 交替更新 OPD 与 RL 可有效维持策略熵，避免梯度冲突。
   - 相比 joint 或 sequential 方案，IDRL 在综合性能上达到最高。

2. **Pistis-Agentic 在多模态搜索任务中表现突出**：
   - 得益于 agentic trajectory SFT 与 IDRL 的结合，尤其在 MMSearch、BrowseComp-VL 等任务上大幅领先。

3. **系统级优化（PAH）可独立提升 agent 性能**：
   - 即使固定模型参数，通过优化 harness（如 Candidate Ledger、Search Skills）也能带来显著增益（+1.8 pts）。
   - 该优化具备跨模型迁移潜力。

4. **grounding 能力显著增强**：
   - Pistis-27B-Thinking 在 grounding 平均分达 **80.5**，超过 Qwen3.8-27B 达 **8.6 pts**，显示其在空间与时间定位上的强大能力。

### 方法的局限性

1. **过程层面仍存在弱点**（见 Section 4）：
   - **意图误解**：能识别视觉线索但错误解读问题意图。
   - **工具确认偏见**：使用工具验证预设假设，而非获取决策相关测量。
   - **检索与推理脱节**：即使检索到正确信息，也可能未绑定到最终答案。

2. **Harness 机制激活有限**：
   - 如 Search Skills 仅在 27% 的轨迹中被触发，多数情况下依赖已有策略。
   - 当前 Candidate Ledger 缺乏系统性的“挑战者测试”，可能过早锁定错误实体。

3. **PAH 泛化范围待验证**：
   - 当前仅在多模态搜索场景验证成功，是否适用于 coding、Claw-Style 等其他领域尚需研究。

### 未来工作方向

1. **引入过程级监督信号**：
   - 在 RL 阶段加入对 **intent verification**、**decision-relevant tool use**、**consistency between evidence and answer** 的奖励。

2. **增强检索-推理闭环设计**：
   - 设计机制确保不确定性自动触发检索，避免“talk itself out of search”。

3. **扩展 PAH 至更多 agent 场景**：
   - 将 auto-harnessing 应用于代码生成、复杂规划等任务，探索通用性。

4. **缩小 benchmark performance 与可靠推理之间的差距**：
   - 推动从“高分”向“可信、可解释”的 agent 发展。

--- 

> 💡 **总结**：Pistis 通过 **IDRL** 和 **PAH** 两条路径，分别在模型参数与系统架构层面实现了对 MLLMs 的高效增强。其实验表明，**共享基础 + 专业化后训练 + 系统级自动化优化** 是构建高性能多模态 agent 的可行范式。

</details>

---

### 16. [To Think or Not to Think: Allocating Reasoning Where It Helps](https://arxiv.org/abs/2609.29664)

**Authors**: Zhengdong He, Yunfan Zhou, Jianguo Yao, Haibing Guan, Xijun Li  
**Category**: cs.AI  
**Published**: 2026-09-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.29664v1  

#### Abstract
Reinforcement learning (RL) has proven effective in enhancing the reasoning performance of large language models (LLMs), particularly in complex mathematical and programming tasks. However, this capability comes with systematic \textit{length misallocation}, in which models devote excessive reasonin...

---

### 17. [Search-Aware Reinforcement Learning for Multi-Component Query Understanding in Roblox Game Search](https://arxiv.org/abs/2609.30177)

**Authors**: Nayoung Choi, Shengjian Chen, Xiaokai Wei, Wenzheng Zhang, Daiyao Yi, Rachit Pareek, Vincent Su, Michelle Gong, Jinho D. Choi  
**Category**: cs.AI  
**Published**: 2026-09-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.30177v1  

#### Abstract
Query understanding (QU) plays a critical role in production search systems, translating raw user queries into search execution plans that drive downstream retrieval and ranking. While large language models (LLMs) have enabled QU to be framed as a structured multi-task generation problem (e.g., inte...

---

### 18. [ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks](https://arxiv.org/abs/2609.29102)

**Authors**: Zeyu Michael Li, William Xingxu Chen, Bingshuo Qian, Jiayin Liu, Xiang Cheng  
**Category**: cs.CL  
**Published**: 2026-09-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.29102v1  

#### Abstract
Fully continuous diffusion language models (dLMs) denoise continuous representations without intermediate discretization, then decode all response tokens in parallel at the final step. Their performance on challenging reasoning tasks remains less established than that of autoregressive (AR) LLMs and...

---

### 19. [Encoded but Not Decoded: Layer-Localized Evidence for a Three-Level Gap in LLM Syntax](https://arxiv.org/abs/2609.29848)

**Authors**: Zhenyan Lu, He Wang, Xiaohui Huang  
**Category**: cs.CL  
**Published**: 2026-09-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.29848v1  

#### Abstract
A language model can fail a syntactic test in two distinct ways: by not encoding the relevant structure, or by encoding it but failing to use it at the output. Behavioral evaluation alone cannot tell these apart. We propose a three-level evaluation framework (behavioral deployment, LM-head readout, ...

---

### 20. [Spatio-temporally complementary feature propagation on graphs for longitudinal AADT estimation](https://arxiv.org/abs/2609.29906)

**Authors**: Linghang Sun, Qishen Zhou, Michail A. Makridis, Anastasios Kouvelas  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.29906v1  

#### Abstract
The estimation of Annual Average Daily Traffic (AADT) is vital for transportation planning and infrastructure maintenance, yet obtaining accurate values for an entire urban network across multiple years remains challenging due to the high cost and spatial sparsity of physical sensors. This research ...

---

### 21. [Grammatical "grandmother neurons" are rare in LLMs](https://arxiv.org/abs/2609.29328)

**Authors**: Linyang He, Nima Mesgarani  
**Category**: cs.CL  
**Published**: 2026-09-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.29328v1  

#### Abstract
Understanding how Large Language Models (LLMs) encode linguistic structures remains a fundamental challenge in interpretability research. While diagnostic classifiers (or "probes") are widely used for this task, they face significant methodological criticism: training auxiliary classifiers introduce...

---

### 22. [Confident but Wrong: A Constrained Decoding Diagnostic for Low-Resource Automatic Post-Editing](https://arxiv.org/abs/2609.29680)

**Authors**: Isuru Wijesiri, Nisansa de Silva, Kavindu Warnakulasuriya, Aloka Fernando, Surangika Ranathunga  
**Category**: cs.CL  
**Published**: 2026-09-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.29680v1  

#### Abstract
Automatic Post-Editing (APE) for low-resource languages (LRLs) often fails to improve Machine Translation (MT), and the score alone cannot say why: whether more training would help, or whether the training data is too inconsistent to learn from. We introduce a black-box, inference-time diagnostic th...

---

### 23. [Monitoring Urban Traffic Dynamics at Fine Spatiotemporal Resolution Using Distributed Acoustic Sensing and Deep Learning](https://arxiv.org/abs/2609.28793)

**Authors**: Hao Tian, Heng Cai, Xiaowei Chen, Yifan Yang  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.28793v1  

#### Abstract
Mapping the distribution of traffic dynamics at high spatiotemporal resolution is a fundamental question in transportation research. Distributed acoustic sensing (DAS), an innovative seismic observation tool, emerges as a promising solution for real-time urban traffic monitoring at high spatial and ...

---

### 24. [CounterRoute: Self-Routed Reasoning via Hierarchical Counterfactual Credit Assignment](https://arxiv.org/abs/2609.29109)

**Authors**: Ruochen Jiao, Besnik Fetahu, Zhenyu Shi, Priyanka Nigam  
**Category**: cs.AI  
**Published**: 2026-09-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.29109v1  

#### Abstract
Reasoning-capable language models often produce long chains of thought when direct answers suffice, wasting inference compute. Many dual-mode models leave this choice to users. Automating it is challenging because routing targets evolve with the policy, initial mode preferences destabilize explorati...

---

### 25. [Qwen-Planner-Agent: A Closed-Loop AI-for-AI Framework for Real-World Mobile Planner Agents](https://arxiv.org/abs/2609.29892)

**Authors**: Tingyu Qu, Weigao Sun, Yuecheng Liu, Yucheng Zhao, Yi Zhu, Yifeng Ding, Qiyi Wang, Sihan Cao, Pengkun Jiao, Hanlei Xie, Xiongwei Wu, Qichao Wang, Haodong Zhang, Jiajun Liu, Yuhao Wang, Yuqing Xie, Junpeng Zhao, Long Chen, Ming Ma, Sihan Yang, Ziwang Zhao, Yanhao Jia, Liangquan Gong, Feida Zhu, Yiran Zhong, Steven Hoi  
**Category**: cs.AI  
**Published**: 2026-09-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.29892v1  

#### Abstract
The rapid progression of large language models is extending AI from passive content generation into the active workflows of engineering and scientific discovery. This shift raises a compelling question: can AI be both the object of development and an active participant in building next-generation AI...

---

### 26. [A Living Benchmark for Information Retrieval from Electronic Health Records](https://arxiv.org/abs/2609.30205)

**Authors**: Jordan L. Cahoon, Chloe O. Stanwyck, Sulaiman Somani, Philip Chung, Kevin R Keet, Kameron C. Black, Andrea T. Fisher, Sarita Khemani, Jerry Liu, Stephen Ma, Saloni K. Maharaj, Rita M. Pandya, Eduardo Perez-Guerrero, Priyanka Pillai, Lisa Shieh, David J. H. Wu, James Xie, James C. McAvoy, Teresa Nguyen, Jessica Tran, Lucy Yin, Bridget Lin, Alison Callahan, Jason A. Fries, Nigam H. Shah, Emily Alsentzer  
**Category**: cs.AI  
**Published**: 2026-09-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.30205v1  

#### Abstract
Large language model (LLM)-based clinical assistants are increasingly being integrated into electronic health record (EHR) systems, transforming how clinicians retrieve and synthesize information from patient records. Their safety and utility depend on rigorous evaluation, yet existing benchmarks ar...

---

### 27. [SemMSA: Latent Semantic-Aided Robust Multimodal Sentiment Analysis with Incomplete Data](https://arxiv.org/abs/2609.30238)

**Authors**: Wenhao Li, Zhibin Wu, Chong Xiao, Qiangchang Wang  
**Category**: cs.CL  
**Published**: 2026-09-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.30238v1  

#### Abstract
Recent research on Multimodal Sentiment Analysis (MSA) has focused on learning from language, visual, and acoustic modalities with incomplete data to infer human sentiment. Most studies typically compensate for missing information by reconstructing modality features or designing complicated fusion m...

---

### 28. [CARE: Condition-Aware Representation Regularization for Diffusion Models](https://arxiv.org/abs/2609.28561)

**Authors**: Fengjia Guo, Zhuoyi Yang, Jie Tang  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.28561v1  

#### Abstract
Recent advances in diffusion models highlight the importance of representation regularization for improving sample quality and training efficiency. However, commonly used regularization methods often overlook the built-in conditions (such as labels or texts) which directly determine the generation t...

---

### 29. [SPADE-DFL: Communication-Efficient Decentralized Federated Learning via Derivative-Free Linearized ADMM](https://arxiv.org/abs/2609.29446)

**Authors**: Mengli Wei, Mengkai Zhu, Jiawen Chen, Wenwu Yu, Duxin Che  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.29446v1  

#### Abstract
Reducing communication in derivative-free decentralized learning requires controlling the disagreement accumulated over multiple local updates. This paper develops SPADE-DFL, a primal--dual method that allows the number of local function-value updates between neighbor exchanges to grow with the comp...

---

### 30. [An Analytical Theory of Auxiliary Learning](https://arxiv.org/abs/2609.29774)

**Authors**: Federico Milanesio, Alessandro Ingrosso, Matteo Osella  
**Category**: cs.LG  
**Published**: 2026-09-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.29774v1  

#### Abstract
Auxiliary learning is an optimization paradigm in which a neural network's performance on a target task is improved by jointly training it on additional tasks. However, the mechanisms behind this improvement remain poorly understood. We study this problem using a teacher-student framework and derive...

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

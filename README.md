# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-31 11:53:34 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Trajectory-Level Speculative Decoding for Diffusion Language Models](https://arxiv.org/abs/2608.27514)

**Authors**: Tianxiang Pan, Baitao Gong, Mo Guang, Hongwei Yong, Tianpeng Jiang, Yaqian Li, Zheng Cao, Kaiwen Long  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.27514v1  

#### Abstract
Diffusion-based language models (dLLMs) enable parallel token generation through iterative denoising, but existing decoding strategies collapse to single-token generation under low confidence, severely limiting throughput. Unlike autoregressive models where speculative decoding operates on token seq...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Trajectory-Level Speculative Decoding for Diffusion Language Models》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
**低置信度退化（low-confidence degeneration）** 是当前 **diffusion-based language models (dLLMs)** 并行解码中的根本瓶颈。尽管 dLLMs 具备并行生成多个 token 的潜力，但在模型对某些位置预测置信度较低时，现有策略（如阈值法或 top-k 选择）会退化为单 token 解码，严重限制吞吐量。

作者通过实证分析发现，在推理和代码生成任务中，超过 **80% 的解码步骤** 出现低置信度退化现象，导致实际每步生成 token 数远低于理论上限（如从 32 降至仅 1–3）。

### 🚀 提出的新方法：Trajectory-Level Speculative Decoding
针对上述问题，本文提出首个专为 dLLMs 设计的 **轨迹级推测解码框架（trajectory-level speculative decoding）**，其核心思想是将推测空间从传统的 token 序列提升到完整的 **denoising 轨迹（trajectory）** ——即包含 token、位置和 unmasking 顺序的多步更新路径。

#### 创新组件：
1. **树形草案轨迹构建（Tree-based draft trajectory construction）**
   - 构建一棵以 block 为节点的树，每个路径代表一条可能的 denoising 轨迹。
   - 在根节点进行 top-k 扩展以最大化多样性，在深层采用 top-1 扩展保持紧凑结构（hybrid expansion），控制复杂度在 $O(W \cdot D)$。

2. **块级并行验证（Blockwise parallel verification）**
   - 设计新的 **blockwise attention masking** 机制：各草案 block 可双向访问已缓存的 prefix/suffix tokens，但彼此隔离，防止干扰。
   - 单次前向传播即可并行验证所有分支，显著降低延迟。

3. **跨块推测（Inter-block speculation）**
   - 利用 dLLMs 的 **双向注意力结构**，在解码当前 block $B_t$ 时监控下一个 block $B_{t+1}$ 的早期置信度。
   - 若满足触发条件（如 $c(B_{t+1}[1]) > T$），则联合构造两个 block 的草案树并同步验证，实现 **跨块前瞻（cross-block lookahead）**。

### 🔍 相比现有方法的优势

| 维度 | 自回归模型的 speculative decoding | 本文方法（Trajectory-Level Speculation） |
|------|-------------------------------|----------------------------------------|
| 推测粒度 | Token-level（固定左到右顺序） | **Trajectory-level**（含位置与 unmasking 顺序） |
| 验证方式 | Causal masking，一次验证整条序列 | **Blockwise masking**，支持双向上下文与独立路径验证 |
| 并行能力 | 固定顺序，无法利用任意 unmasking 顺序 | 显式建模多种 unmasking 路径，提升鲁棒性 |
| 结构优势 | 不适用于 bidirectional 模型 | 充分利用 dLLMs 的双向结构，引入 **inter-block speculation** |
| 实现开销 | 较低（token 级别） | 合理可控（受限树结构 + 高效验证） |

此外，该方法与系统级优化（如 Fast-dLLM 的 dual-cache）正交且可叠加，形成算法-系统协同加速。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **数学推理任务**：
  - `GSM8K`：小学数学应用题
  - `MATH`：复杂多步数学推理（按难度分级）
- **代码生成任务**：
  - `HumanEval`：Python 函数补全
  - `MBPP`：基于描述的编程任务
- 补充测试：
  - `IFEval`：长文本指令遵循任务（用于验证低置信场景下的表现）

### ⚙️ 实验设置
- **模型**：
  - `LLaDA-Instruct-7B` 和 `Dream-Instruct-7B`（均为 dLLMs）
- **硬件平台**：
  - 主要使用 A800 和 H800 GPU
- **生成参数**：
  - 总生成长度固定为 512 tokens
  - Block size = 32 tokens
  - 默认树结构：`W2D2(3)`（宽度 2，深度 2，共 3 个草案 block）
- **评估指标**：
  - `TPS`（Tokens Per Second）
  - `End-to-end latency`
  - `Denoising steps`
  - `Tokens-per-step`
  - `Accuracy`（任务准确率）
  - `Speedup`（相对于 baseline 的加速比）

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| Vanilla dLLM | 基线 | 无任何优化 |
| Fast-dLLM (Single Cache) | 系统优化 | 引入 KV 缓存减少计算 |
| Fast-dLLM (Dual Cache) | 主要 baseline | 改进版 KV 缓存 + 并行解码，已是高效系统 |
| Spiffy (Agrawal et al., 2025) | 并发工作 | 块级推测 + 图校准保证 lossless 验证 |
| Self-Speculative Decoding (Gao et al., 2025) | 并发工作 | token-level 推测 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 LLaDA-7B 为例）

| Benchmark | 方法 | TPS ↑ | Denoising Steps ↓ | Tokens/Step ↑ | Speedup ↑ | Acc. (%) |
|----------|------|-------|------------------|---------------|-----------|----------|
| **GSM8K** | Vanilla | 4.1 | 512.0 | – | 1.0× | 76.6 |
|           | Fast-dLLM (DC) | 48.8 | 110.3 | 2.6 | 11.9× | 75.6 |
|           | **Ours** | **55.7** | **71.3** | **3.8–4.3** | **13.6×** | **75.5** |
| **HumanEval** | Fast-dLLM (DC) | 70.6 | 179.2 | 2.6 | 5.1× | 45.7 |
|               | **Ours** | **94.4** | **106.7** | **4.3** | **6.8×** | **46.3** |
| **MBPP** | Fast-dLLM (DC) | 48.5 | 132.8 | 2.6 | 10.1× | 13.6 |
|          | **Ours** | **56.4** | **81.7** | **4.3** | **11.8×** | **13.4** |
| **MATH** | Fast-dLLM (DC) | 60.7 | 167.1 | 2.6 | 8.4× | 35.7 |
|          | **Ours** | **73.5** | **96.8** | **4.3** | **10.2×** | **36.0** |

> ✅ **总体提升**：
> - 相比 Fast-dLLM (Dual Cache)，**减少 30–43% 的 denoising 步骤**
> - **Tokens-per-step 从 2.6 提升至 3.8–4.3**
> - 实现 **1.2–1.4× 的额外加速**
> - 相比 vanilla dLLM，达到 **7–14× 端到端加速**
> - 准确率变化小于 **1%**

### 🔬 消融实验结果（Ablation Study）

#### （1）不同树结构的影响（HumanEval, A800）

| 配置 | Tokens/Step | Steps | Latency (s) |
|------|-------------|--------|------------|
| Fast-dLLM (DC) | 2.61 | 179.2 | 6.6 |
| + Intra W1D1(1) | 3.51 | 134.5 | 5.6 |
| + Intra W2D2(3) | 3.83 | 124.0 | 5.3 |
| + Intra W3D3(6) | 3.95 | 120.1 | 5.5 |
| + Inter W2D2(3) | **4.28** | **106.7** | **5.0** |
| + Inter W3D3(6) | 4.52 | 105.1 | 5.6 |

> 💡 发现：
> - 更大的树能略微提升 acceptance rate，但 per-step 开销增加，**整体延迟反而上升**
> - **Inter-block speculation 贡献约 12% 的步数下降**

#### （2）跨硬件表现差异（A800 vs H800）
- 在算力更强的 H800 上，更大树结构（如 W3D3）仍可带来收益，说明 **最优树大小依赖于硬件能力**

#### （3）树构建开销分析（A100）
- 对于默认配置 `W2D2`，在序列长度 512 时，**树构建开销 < 2%**
- 随着序列增长，相对开销进一步下降，表明其轻量化设计有效

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **低置信度退化是 dLLMs 并行效率的核心瓶颈**，而非模型本身能力不足。
2. 即使在低置信位置，正确 token 往往仍在 top-k 中，因此可通过 **trajectory-level speculation** 来维持高并行度。
3. 所提方法可在 **几乎不损失精度（<1%）的前提下，实现 7–14× 加速于 vanilla dLLM，1.3× 优于 Fast-dLLM**。
4. **Theorem 1 形式化证明了方法在理想条件下是精确等价的（exact）**，偏差来源于“轨迹漂移”（trajectory drift），而非算法近似。
5. **Inter-block speculation 是一种独特优势**，源于 dLLMs 的双向结构，无法在 autoregressive 模型中实现。

### ⚠️ 局限性
1. **轨迹漂移（trajectory drift）不可避免**：当提高并行度时，可能会偏离原始解码路径，尤其影响需要精确多步推理的任务（如 MATH Level 4）。
2. 当任务整体置信度极高或极低时，增益有限：
   - 高置信：baseline 已高效，推测增益小；
   - 极低置信：难以构造有效草案路径。
3. 当前实现位于 Python 层，未融合 CUDA kernel，**per-step 验证开销尚未完全摊销**，仍有工程优化空间。

### 🔮 未来工作方向
1. **自适应树结构（adaptive tree construction）**
   - 引入学习型控制器动态调整树宽/深，基于历史接受率、当前置信分布、任务类型等。
2. **降低轨迹漂移风险**
   - 设计 confidence-aware verification 或 rollback 机制，在检测到漂移时回退。
3. **联合优化扩散调度与推测策略**
   - 将 speculation 与 noise schedule 耦合，设计更利于推测的 denoising 过程。
4. **扩展至更大规模模型与更长序列生成**
   - 探索在 13B+ 模型及 8k+ 上下文中的有效性。

---

> 📌 **总结一句话**：  
> 本文提出了首个面向 **diffusion language models** 的 **trajectory-level speculative decoding** 框架，通过构建树状 denoising 轨迹、块级并行验证与跨块前瞻机制，在几乎不失真的前提下大幅提升了解码效率，实现了 **7–14× 相对于 vanilla dLLM 的端到端加速**，为 dLLMs 的高效部署提供了重要算法基础。

</details>

---

### 2. [H-Scale: Hessian-Guided Scale Refinement for NVFP4 Sub-Byte LLM Inference](https://arxiv.org/abs/2608.28113)

**Authors**: Hao Yu, Zheng Li, Dayiheng Liu, Jianwei Zhang  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.28113v1  

#### Abstract
The NVIDIA Blackwell architecture, with native support for the ultra-fine-grained NVFP4 format, opens new opportunities for accelerating large language model (LLM) inference. NVFP4's micro-block design, such as a group size of 16, offers strong representational flexibility for capturing local weight...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# H-Scale: Hessian-Guided Scale Refinement for NVFP4 Sub-Byte LLM Inference —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **NVFP4 量化中的 per-group scale 敏感性问题**：  
  NVFP4 采用超细粒度分组（如 group size $g=16$），每个组使用独立的 E4M3 FP8 局部缩放因子（local scale）。传统方法（如 RTN）通过最大绝对值初始化 scale，但这种策略仅关注权重范围拟合，忽略了对模型输出的影响，导致量化误差较大。
- **现有 PTQ 方法的局限性**：  
  当前主流 PTQ 方法（如 GPTQ、SpinQuant）主要优化量化后的权重值（weight reconstruction），而对 scale 的选择缺乏精细化调整，尤其在 NVFP4 微缩放（microscaling）场景下效果有限。

### 🚀 提出的新方法：H-Scale
- **核心思想**：将 per-group scale 的选择从“最小化权重重建误差”转向“最小化层输出扰动”，引入基于对角 Hessian 的加权目标函数。
- **具体实现**：
  - 利用校准集（calibration set）上的激活值计算对角 Hessian 矩阵 $\mathbf{h} = \text{diag}(X^T X)$，作为各输入通道的重要性权重。
  - 对每个 weight group，在硬件支持的 E4M3 scale 邻域内进行离散搜索，选择使 Hessian 加权重建误差最小的 scale：
    $$
    \min_{s_{r,i}} \|(\mathbf{w}_{r,i} - \hat{\mathbf{w}}_{r,i}(s_{r,i})) \odot \sqrt{\mathbf{h}_i}\|^2
    $$
  - 最终仍保持原生 NVFP4 格式，不改变推理开销。

### 🔍 相比现有方法的优势
| 特性 | H-Scale | 传统方法（如 RTN/GPTQ） |
|------|--------|--------------------------|
| **优化目标** | 输出感知（output-aware） | 权重重建（weight-centric） |
| **是否引入推理开销** | ❌ 零额外开销 | ❌ 无（同为 PTQ） |
| **是否兼容现有 pipeline** | ✅ 可即插即用替换 RTN-style scale selection | — |
| **是否依赖训练/微调** | ❌ 完全后训练（post-training） | ❌ 同属 PTQ 范畴 |
| **适用性** | 广泛适用于各类 NVFP4 pipeline（RTN, GPTQ, 4over6, ArcQuant 等） | 方法特定 |

> 💡 **关键优势**：H-Scale 是一种轻量级、通用、零推理成本的 **scale refinement 插件模块**，显著提升多种 NVFP4 基线的精度表现。

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **校准数据（Calibration Set）**：
  - 使用 `FineWeb` 数据集，共 4096 个样本，序列长度 8192。
  - 少数消融实验中也测试了仅用 128 样本的效果（见 Table 12）。
- **评估模型**：
  - Qwen 系列：`Qwen3-4B-Instruct`, `Qwen3-30A3-Instruct`, `Qwen3-30A3-Thinking`
  - LLaMA 系列：`LLaMA-3.1-8B-Instruct`
- **任务类型覆盖广泛**：
  - 推理：AIME24/AIME25, GPQA
  - 知识理解：MMLU-R, C-Eval
  - 编码能力：LiveCodeBench (LCBench)
  - 数学：GSM8K
  - 综合推理：ARC-C, BBH

### 🧪 实验设置与评估指标
- **量化格式**：NVFP4，group size 固定为 16（Blackwell 架构原生支持）
- **评估协议**：
  - 所有生成任务使用固定解码参数：temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
  - Beam search：AIME 使用 beam_size=16，LiveCodeBench 使用 beam_size=10
  - 每个配置运行 **3 次独立推理**，报告平均得分以减少随机波动影响
- **评估指标**：
  - 主要指标：各项任务的准确率（accuracy），最终取多任务平均分（Avg.）
  - 辅助分析指标：layer output error, weight MSE, Hessian-weighted error

### 🆚 基线方法对比
H-Scale 被集成到以下主流 NVFP4 PTQ pipeline 中进行比较：
| 方法 | 类型 |
|------|------|
| **RTN** | Round-To-Nearest，基础 baseline |
| **GPTQ** | 基于 Hessian 的逐列重构 |
| **GPTAQ** | 改进版 GPTQ，处理非对称分布 |
| **4over6** | 自适应子块格式优化 |
| **ArcQuant** | 引入残差通道增强信息保留 |
| **MR-GPTQ** | 针对 microscaling 场景优化的 GPTQ 变体 |

> ⚠️ 不包含 AWQ、SmoothQuant 等激活迁移类方法，因已有研究表明其在 $g=16$ 下增益有限。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（来自 Tables 1–4）

| 模型 | 方法 | Avg. Score | +H-Scale 提升 |
|------|------|-----------|--------------|
| Qwen3-30A3-Thinking | GPTQ → GPTQ+H-Scale | 80.01 → **81.22** (+1.21) |
| Qwen3-30A3-Instruct | 多数 baseline | 平均提升 ~0.5–0.8 pts |
| Qwen3-4B-Instruct | GPTAQ → GPTAQ+H-Scale | 56.63 → **57.52** (+0.89) |
| LLaMA-3.1-8B-Instruct | GPTAQ → GPTAQ+H-Scale | 49.54 → **51.48** (+1.94) |

> ✅ **所有测试的 baseline 在加入 H-Scale 后均取得平均性能提升**

#### 典型结果亮点：
- 在 `Qwen3-30A3-Thinking` 上，`GPTQ + H-Scale` 达到 **81.22**，超过 BF16 基线（81.06），表明该组合已接近浮点精度。
- 在复杂推理任务（AIME24/AIME25）上提升尤为明显，说明 H-Scale 更好地保留了模型的推理能力。
- 在较小模型（4B）和跨家族模型（LLaMA）上依然有效，验证了泛化性。

### 🔬 消融实验结果（Ablation Study）

#### （1）Hessian 是否必要？（Table 5 & 11）
| 方法 | Variant | Avg. Score |
|------|--------|----------|
| 4over6 | Baseline | 64.25 |
| 4over6 | Scale-only (uniform weight) | 63.55 ↓ |
| 4over6 | +H-Scale (Hessian-weighted) | **64.53** ↑ |
| ArcQuant | +H-Scale | 64.24 vs. baseline 63.45 |
| GPTQ | +H-Scale | 64.50 vs. baseline 63.90 |

> ❗ 结论：单纯搜索 scale（without Hessian weighting）可能反而损害性能；**Hessian 加权是关键增益来源**。

#### （2）候选窗口宽度影响（Table 7）
- 默认设置：向上搜索 6 步（U=6），总预算 K=16（即向下最多 9 步）
- 实验发现：当 $U \geq 6$ 时，中位数恢复率达 **100%**（相对于枚举全部 E4M3 scale）
- 更宽窗口（如 K=24）收益极小 → 表明默认设置已足够高效

#### （3）运行时间开销（Table 6）
| 方法 | Runtime (原始) | +H-Scale |
|------|---------------|---------|
| 4over6 | 0.2h | **0.4h** |
| GPTQ | 9.6h | **9.9h** |

> ✅ H-Scale 增加的离线成本非常低（仅增加约 0.3h 或更少），且可并行加速。

#### （4）Scale 调整统计（Figure 2）
- 31.2% 的 group 保持原始 max-abs scale
- **47.9% 选择了更大的 scale**，20.9% 选择了更小的 scale
- 深度收缩（deep contraction, $\delta \leq -4$）虽罕见（0.11%），但贡献了 **8.3% 的总误差降低**

> 📌 发现：并非简单“缩小 scale”就好，而是需根据 Hessian 动态判断最优方向。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Per-group scale selection 是 NVFP4 量化中的关键瓶颈**，不应被忽视。
2. **H-Scale 显著优于传统的 weight-centric scale 搜索**，因其直接优化输出扰动代理目标。
3. **H-Scale 是通用、轻量、零推理开销的插件式改进**，可无缝集成至 RTN、GPTQ、4over6 等各类 NVFP4 pipeline。
4. **在多个模型规模（4B~30B）、多个模型家族（Qwen/LLaMA）、多种任务上均稳定提点**，具备强泛化能力。
5. **即使使用少量校准数据（128 samples）也能带来增益**（Table 12），显示其鲁棒性。

### ⚠️ 局限性
- 当前研究局限于文本-only LLM 和固定的 $g=16$ 设置。
- 未探索其他 group size（如 $g=32, 64$）或其他量化格式（如 INT4, NF4）下的适用性。
- Hessian 近似仅使用对角项，忽略通道间相关性（off-diagonal），理论上存在进一步优化空间。

### 🔮 未来工作方向
- 探索 H-Scale 在 **多模态模型**（vision-language models）中的应用。
- 研究动态 group size 或 hierarchical scaling 策略。
- 将 scale tuning 思路扩展至 **activation quantization** 或 **KV cache 压缩**。
- 探索在线自适应 scale 调整机制（尽管会牺牲零开销特性）。

---

> 🏁 **总结一句话**：  
> **H-Scale 通过引入 Hessian 指导的 per-group scale refinement，解决了 NVFP4 量化中 scale 敏感性的关键问题，在几乎无推理代价的前提下，显著提升了多种主流 LLM 在多种任务上的量化性能，是一种极具实用价值的轻量级后处理技术。**

</details>

---

### 3. [DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization](https://arxiv.org/abs/2608.27513)

**Authors**: Tao Zhang, Jianchao Tan, Pingwei Sun, Yanqi Yu, Zixu Jiang, Yuchen Xie, Xunliang Cai, Ziqian Zeng  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.27513v1  

#### Abstract
Softmax attention stores key and value vectors for every preceding token, causing inference memory to grow with sequence length. Recent language models incorporating Gated DeltaNet (GDN) or Kimi Delta Attention (KDA) reduce this cost by replacing the KV cache in most layers with fixed-size recurrent...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代大语言模型（LLMs）在长序列推理中面临 **KV Cache** 的内存瓶颈。为缓解此问题，新兴架构如 **Gated DeltaNet (GDN)** 和 **Kimi Delta Attention (KDA)** 使用固定大小的 **recurrent state** 替代传统的 KV Cache，显著降低内存增长。然而，这些 recurrent states 通常以 **FP32** 存储，占用大量 GPU 内存，并且其更新操作是 **memory-bandwidth bound**，成为解码延迟的重要来源。

本文首次研究了 **post-training quantization（PTQ）** 在 GDN 和 KDA 架构中的 **recurrent state** 上的应用，旨在压缩存储、加速更新，同时保持模型精度。

### **提出了什么新方法或新思路**
作者提出 **DAMP（Decay-Aware Mixed-Precision）**，一种针对 recurrent state 的混合精度量化方法，其核心思想是：
- **识别高风险通道（key channels）**：通过分析量化误差能量（quantization-error energy）和基于衰减的持久性（decay-based persistence），离线识别对精度影响最大的 state 通道。
- **静态混合精度布局**：将高风险通道以 **FP16** 存储，其余通道以 **INT8 + Hadamard 变换** 存储，形成固定的混合精度格式。
- **无需重训练或动态选择**：该布局在离线校准阶段确定，推理时复用，不增加额外计算开销。

### **相比现有方法的优势**
- **首次探索**：据作者所知，这是首个研究 GDN/KDA 中 recurrent state 量化的 PTQ 工作。
- **优于均匀量化**：相比统一的 INT8 或 FP8 量化，DAMP 显著减少精度损失，尤其在数学和代码生成等复杂任务上表现优异。
- **高效实现**：通过打包（packed）布局和融合内核（fused kernel），实现了高效的内存访问和计算，提升推理速度。
- **通用性强**：方法适用于 GDN（head-level decay）和 KDA（channel-level decay）两种架构。

---

## **2. 核心实验方法和设置**

### **使用的模型**
- **Qwen3.6-35B-A3B**：包含 30 层 GDN 和 10 层 full-attention。
- **Kimi-Linear-48B-A3B-Instruct**：包含 20 层 KDA 和 7 层 MLA。

### **数据集与基准测试**
在六项下游任务上评估模型性能：
- **数学推理（Mathematical Reasoning）**：
  - AIME 2026 Part I & II
  - HMMT February 2026
  - IMO-AnswerBench
- **通用推理（General Reasoning）**：
  - GPQA-Diamond
  - MMLU-Pro
- **代码生成（Code Generation）**：
  - LiveCodeBench-v6

### **实验设置**
- **推理框架**：基于 **SGLang** 实现。
- **批处理大小**：从 32 到 256 不等。
- **输入输出长度**：输入 256 tokens，输出 64 tokens。
- **校准数据**：使用 **The Pile** 数据集中 32 个未标注文档（来自数学、PubMed、GitHub、StackExchange），每 8 个 token 采样一次状态，共 1,024 个样本/层。

### **评估指标**
- **Accuracy**：各任务上的平均准确率。
- **Storage Cost**：每个 state value 的有效比特数（bits/value）。
- **Latency**：
  - Recurrent-state update operator 延迟
  - 全模型 **Time Per Output Token (TPOT)**

### **基线方法对比**
| 方法 | 精度格式 | Bits/Value |
|------|--------|------------|
| FP32 | 全精度 | 32 |
| FP16 / BF16 | 半精度 | 16 |
| FP8 (E4M3) | 浮点8位 | 9.0 |
| INT8 / INT4 | 整数量化 | 9.0 / 5.0 |
| INT8+Hadamard | Hadamard 变换后量化 | 9.0 |
| NVFP4 | NVIDIA FP4 格式 | 4.5 |
| **DAMP INT8** | **混合精度（FP16 + INT8+H）** | **9.9** |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | 方法 | Avg. Bits | AIME 2026 | HMMT Feb | IMO-Ans | GPQA-D | MMLU-Pro | LCB-v6 | 平均准确率 |
|------|------|----------|----------|----------|---------|--------|----------|--------|------------|
| Qwen3.6 | FP32 | 32.0 | 85.46 | 57.57 | 50.29 | 81.97 | 84.66 | 86.95 | — |
| | DAMP INT8 | **9.9** | **83.65** | **54.40** | **48.75** | **81.50** | **84.52** | **85.46** | **~FP32** |
| Kimi-Linear | FP32 | 32.0 | 62.34 | 38.02 | 28.04 | 63.16 | 68.48 | 61.17 | — |
| | DAMP INT8 | **9.9** | **63.72** | **38.39** | **28.56** | **64.64** | **68.47** | **61.02** | **~FP32** |

> ✅ DAMP 在仅 **9.9 bits/state value** 下，几乎完全保留了 FP32 的精度。

### **与基线方法的对比结果**
- **vs. 均匀量化（Uniform Quantization）**：
  - INT8+Hadamard：在 Qwen 上平均准确率下降超 20 pts。
  - FP8 / INT4 / NVFP4：严重崩溃，数学与代码任务接近零。
- **vs. 半精度（FP16/BF16）**：
  - DAMP 存储更省（9.9 vs 16 bits），精度更高或相当。
  - BF16 表现较差，说明 exponent 范围不如量化分辨率重要。

### **效率提升**
| 指标 | 提升幅度 |
|------|----------|
| **Recurrence-State 存储** | ↓ **69.1%** |
| **Recurrent-State Update Kernel 速度** | ↑ **up to 2.01×** |
| **全模型 TPOT** | ↓ **up to 10.9%**（batch=256） |

> ⚡ 随着 batch size 增大，DAMP 的优势更加明显，因内存带宽压力更大。

### **消融实验结果**

#### **(1) Key-Channel Selector 对比（KDA）**
| Selector | AIME 2026 | GPQA-D | LCB-v6 |
|---------|----------|--------|--------|
| Random | 55.31 | 62.75 | 58.68 |
| State Energy | 60.52 | 63.97 | 59.67 |
| Error Only (E) | 61.98 | 64.03 | 60.34 |
| Persistence Only (P) | 61.35 | 63.05 | 60.69 |
| **DAMP (E × P)** | **63.72** | **64.64** | **61.02** |

✅ **结合 error 和 persistence 的乘积评分效果最佳**。

#### **(2) 精度预算（Precision Budget）**
- 当保留 **16/128 = 12.5%** 的 key channels 为 FP16 时，性能趋于饱和。
- 此时总成本为 **9.875 ≈ 9.9 bits/value**，为最优工作点。

#### **(3) 校准稳定性**
- 在不同校准子集上，top-16 key channels 的重叠率达 **92.0% (KDA)** 和 **93.2% (GDN)**。
- Spearman 相关系数 > 0.98，表明 decay ordering 高度稳定。

---

## **4. 关键结论和发现**

### **主要发现**
1. **uniform quantization 不适用于 recurrent state**：即使 INT8 也会导致显著精度下降，INT4/NVFP4 几乎失效。
2. **量化误差具有通道集中性**：少量 key channels 承担大部分重建误差。
3. **learned decay 具有跨任务稳定性**：尽管 token 级 decay 动态变化，但 channel-level 的 retention strength 排序高度一致，可用于离线估计。
4. **error + persistence 是有效的风险指标**：两者的乘积能更好预测通道对最终输出的影响。
5. **DAMP 实现近无损压缩**：在 9.9 bits 下达到 FP32 级精度，同时大幅提升效率。

### **方法的局限性**
- **依赖特定实现**：当前评估基于 SGLang 和特定 GDN/KDA 检查点，硬件或框架差异可能影响收益。
- **低精度底层层未突破**：尝试使用 INT4 或 NVFP4 作为底层层未能恢复 FP32 精度。
- **简化了 error propagation 模型**：仅考虑对角线 decay 路径，未建模 full transition matrix 的复杂动态。

### **未来工作方向**
- 建模更丰富的 recurrent dynamics，如 cross-channel error redistribution。
- 开发更鲁棒的 **4-bit quantizer** 和混合布局。
- 将 DAMP 扩展到更多 SSM 架构（如 Mamba）和其他 serving system。
- 探索训练时感知的量化（quantization-aware training）以进一步压缩。

---

> 📌 **总结一句话**：  
> **DAMP 是首个针对 GDN/KDA recurrent state 的 post-training 混合精度量化方案，通过 decay-aware 风险建模，在 9.9 bits 下实现近乎无损压缩，显著降低存储与延迟，为长上下文 LLM 高效部署提供了新路径。**

</details>

---

### 4. [D-TAIA: Domain-Aware LLM Adaptation for Multi-Task Predictive Process Monitoring](https://arxiv.org/abs/2608.28236)

**Authors**: Sjoerd van Straten, Christine Jacob, Marwan Hassani  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.28236v1  

#### Abstract
Predictive Process Monitoring (PPM) enables organizations to forecast future process behavior, such as the next activity and remaining time of ongoing cases. In practice, three conditions cause existing methods to degrade, namely data scarcity, high process entropy and distributional shift. While Fo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：D-TAIA: Domain-Aware LLM Adaptation for Multi-Task Predictive Process Monitoring

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**预测性流程监控（Predictive Process Monitoring, PPM）**中的三大现实挑战，现有方法在以下条件下表现退化：
- **Data Scarcity（数据稀缺）**：事件日志样本少，尤其稀有活动难以学习；
- **High Process Entropy（高流程熵）**：流程变体多、不确定性高，导致预测困难；
- **Distributional Shift（分布偏移）**：测试时出现训练未见的新流程变体，违反i.i.d.假设。

尽管已有部分方法分别应对上述单个问题，但**尚无统一框架能同时处理这三个条件下的联合“下一活动（NA）”与“剩余时间（RT）”预测任务**。

---

### 🚀 提出的新方法：D-TAIA 框架
提出 **D-TAIA（Domain-aware Training and Attention-based Inference Architecture）**，一个基于 Foundation Model（FM）的参数高效微调（PEFT）框架，用于多任务 PPM。

#### 核心创新点：
1. **Domain-aware Triplet Loss (DATL) 预训练 + FAISS 检索机制**
   - 利用 DATL 构建领域无关的嵌入空间，使相似行为（如剩余时间模式）聚类，而非按领域特征聚类；
   - 在推理阶段通过 **FAISS-based nearest neighbor retrieval** 获取历史相似前缀，进行非参数化 RT 估计，缓解数据稀缺和高熵问题。

2. **TAIA 推理策略保留预训练序列推理能力**
   - 微调后推理时冻结 FFN 层的 LoRA 更新（`ΔW_FFN = 0`），仅保留 Attention 层更新；
   - 保留 LLM 在预训练中获得的通用序列建模能力，提升对分布外（OOD）样本的鲁棒性。

3. **文本序列化输入设计**
   - 将流程前缀结构化为文本格式（含 domain token 和 activity-feature 对），直接复用 LLM 分词器，避免数值特征离散化损失信息。

4. **双路径 RT 融合预测（FusionGate）**
   - 结合直接回归头输出 `rtdirect` 与检索估计 `rtretrieved`，加权融合得到最终 RT 预测：  
     `rt_final = β * rtdirect + (1−β) * rtretrieved`

---

### 🔍 相比现有方法的优势
| 方面 | D-TAIA | 现有方法（如 Oyamada et al. [12]） |
|------|--------|-------------------------------|
| 多任务支持 | ✅ 同时 NA + RT | ✅ 支持 |
| 数据稀缺处理 | ✅ DATL + Retrieval | ❌ 无专门机制 |
| 高熵鲁棒性 | ✅ 检索增强 + 行为聚类 | ❌ 回归头易失效 |
| 分布偏移鲁棒性 | ✅ TAIA 推理架构 | ❌ 依赖标准微调，泛化差 |
| 时间预测方式 | ✅ 非参数检索 + 参数回归融合 | ❌ 单一头直接回归 |

> ✅ D-TAIA 是首个**同时解决数据稀缺、高熵、分布偏移**三大挑战的**多任务 PPM 框架**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
来自 BPI Challenge 的四个真实世界事件日志，涵盖不同规模与复杂度：

| Dataset | #Cases | #Events | #Activities | Avg. Length | Avg. Duration (days) | Entropy |
|--------|--------|---------|-------------|--------------|------------------------|--------|
| **BPI2012** | 13,087 | 262K | 24 | 20.04 | 8.62 | 3.688 |
| **BPI2017** | 31,509 | 1.2M | 26 | 38.16 | 21.90 | 3.786 |
| **BPI2015_2** | 832 | 44K | 410 | 53.31 | 160.49 | **7.105** *(高熵)* |
| **BPI2020_DD** | 10,500 | 56K | 17 | 5.37 | 11.67 | **2.827** *(低熵)* |

> 特别关注极端情况：**BPI2015_2（小样本+高熵）** 和 **BPI2020_DD（近乎确定性流程）**

---

### 🧪 实验设置与评估指标

#### ✅ 评估指标
- **Macro-F1**：活动预测性能，对稀有类公平；
- **RT MAE**：剩余时间预测误差（单位：天）；
- **Wall-clock Runtime**：总运行时间（小时），衡量计算开销。

#### ✅ 数据划分
- 时间顺序划分（temporal split）：65% 训练 / 15% 验证 / 20% 测试；
- 所有案例不跨集合，防止泄露；
- 重复 5 次不同随机种子，报告均值 ± 95% 置信区间。

#### ✅ 基线方法对比
| Baseline | 描述 |
|--------|------|
| **FT-LLM** | 基于 Oyamada et al. [12] 的 LoRA 微调 LLM，相同 backbone 下比较，排除容量影响；不含 DATL、Retrieval、TAIA |
| **MT-RNN** | 经典多任务 LSTM 架构，共享编码器 + 双头输出，代表传统深度学习范式 |

#### ✅ Backbone 对比实验
使用三种不同规模的 LLM：
- **Tiny-LLM (10M)**：极小模型，验证轻量化可行性；
- **Qwen2.5 (500M)**；
- **Llama3.2 (1B)**；

---

## 3. 主要实验结果和性能指标

### 📊 性能对比结果（图 2–4）

#### ✅ 整体性能优势
- 在所有数据集和 backbone 上，**D-TAIA 在 Macro-F1 和 RT MAE 上均优于 FT-LLM 和 MT-RNN**；
- 尤其在 **BPI2015_2（高熵+小样本）** 上优势最显著。

| 指标趋势 | 观察结果 |
|--------|--------|
| **Macro-F1** | D-TAIA > FT-LLM > MT-RNN，差距在高熵日志更明显 |
| **RT MAE** | D-TAIA 显著低于 FT-LLM，置信区间不重叠（BPI2012/BPI2020_DD） |
| **运行效率** | 尽管引入检索模块，整体 runtime 仍可控，且小 backbone 下更具性价比 |

> 💡 **即使使用仅 10M 参数的 Tiny-LLM，D-TAIA 也能达到或超越更大模型的表现**，说明其架构有效性。

---

#### ✅ 数据量敏感性分析（图 3）
- 在 **20%~100% 训练数据比例下，D-TAIA 始终保持领先**；
- 在 **BPI2015_2 全量数据下优势最大且最稳定**；
- 在低熵日志（如 BPI2020_DD）上三者趋近，表明简单任务中增益有限。

> 👉 表明 D-TAIA 特别适用于**数据稀缺 + 高不确定性场景**。

---

#### ✅ 前缀长度敏感性分析（图 4）
- **短前缀（Bucket 1）时性能优势最大**：
  - Macro-F1 提升 3.0–6.0 个百分点；
  - RT MAE 显著更低；
- 随着前缀增长（→Bucket 5），各模型差距缩小；
- 但在 **BPI2015_2 上，即使长前缀仍有较大残差优势**。

> 👉 D-TAIA 在早期预测阶段最具价值——这正是实际业务中最需要干预的关键时刻。

---

### 🔍 消融实验结果（表 3）

在 **BPI2015_2（高熵）** 和 **BPI2020_DD（低熵）** 上进行组件消融：

| 变体 | Macro-F1 ↓ | MAE ↑ | 显著下降？ |
|------|-----------|-------|------------|
| **D-TAIA (full)** | .445±.032 | 53.85±2.18 | — |
| **-DATL** | .352±.048* | 68.45±3.95* | ✅ 是（全部指标） |
| **-FAISS** | .418±.036 | 61.85±2.88* | ✅ MAE 显著上升 |
| **-TAIA** | .425±.034 | 55.28±2.35 | ⚠️ 影响最小 |
| **-Domain ID** | .385±.042 | 58.22±2.65 | 中等影响 |

#### 关键发现：
1. **DATL 最关键**：移除后性能暴跌，且检索也因编码器未训练而失效；
2. **FAISS 对 RT 至关重要**：尤其在高熵环境下，非参数估计有效弥补回归头不足；
3. **TAIA 影响较小**：可能因当前 backbone 或任务特性限制了其潜力，但仍具理论意义；
4. **Domain Label 设计有效**：有助于控制嵌入空间结构。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **非参数时间估计优于直接回归**：  
   ➤ 基于 FAISS 的 nearest neighbor retrieval 显著提升了 RT 预测精度，尤其是在高熵、数据稀缺的日志中，解决了 autoregressive LLM 在连续数值预测上的结构性弱点。

2. **DATL + TAIA 可迁移至 PPM**：  
   ➤ 来自 NLP/CV 的技术（如 domain-aware triplet loss、attention/FFN 功能分离）可成功迁移到流程挖掘领域，仅需 **10M 参数 backbone** 即可实现 SOTA 表现。

3. **D-TAIA 实现全面领先**：  
   ➤ 在多个真实日志上，D-TAIA 在 Macro-F1 和 RT MAE 上均优于 FT-LLM 和 MT-RNN；
   ➤ 优势在 **短前缀、高熵、小样本** 场景最为突出，符合实际应用需求。

4. **架构增益独立于模型大小**：  
   ➤ 即使使用 Tiny-LLM，D-TAIA 仍能超越更大的标准微调模型，证明其组件设计的有效性远超单纯增加参数。

---

### ⚠️ 局限性
1. **TAIA 的实际效果有限**：  
   ➤ 消融实验显示其影响最小，可能因为当前任务或 backbone 不足以体现其 OOD 泛化优势；
   ➤ 缺乏显式的 distributional shift benchmark 来验证其设计初衷。

2. **FusionGate 权重固定**：  
   ➤ β=0.5 为手动设定，未根据上下文动态调整，可能不是最优策略。

3. **检索质量依赖 DATL 训练**：  
   ➤ 若 DATL 失败，整个 retrieval 路径崩溃；尚未验证是否可用 LLM 自身表示替代专用 DATLEncoder。

4. **扩展性挑战**：  
   ➤ FAISS 索引需存储所有训练嵌入，在超大规模日志中可能面临内存压力。

---

### 🔮 未来工作方向
1. **探索更强 backbone 上的 TAIA 效果**：  
   ➤ 如结合 GNN 编码流程拓扑结构，观察结构适应是否更能体现 TAIA 价值。

2. **动态 FusionGate 设计**：  
   ➤ 引入门控机制，根据前缀熵、检索置信度等动态调整 `β`。

3. **将检索前缀注入上下文窗口**：  
   ➤ 类似 in-context learning，直接将 top-k 相似前缀拼接进 prompt，测试是否比单独检索更优。

4. **构建标准 OOD Benchmark for PPM**：  
   ➤ 显式构造分布偏移场景，系统评估模型泛化能力。

---

> ✅ **总结一句话**：  
> **D-TAIA 成功将 NLP/CV 领域的先进思想整合到 PPM 中，提出了一种轻量、鲁棒、高效的多任务预测框架，在数据稀缺、高熵、分布偏移等现实挑战下展现出卓越性能，推动了 LLM 在流程智能中的落地应用。**

</details>

---

### 5. [Accelerating LLM Inference via Vector Index Based Output Embeddings](https://arxiv.org/abs/2608.27460)

**Authors**: Martin Loretz, Sepp Hochreiter  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.27460v1  

#### Abstract
Large output embedding matrices create a significant memory bandwidth bottleneck during autoregressive decoding, especially for compact LLMs with large multilingual vocabularies. We reformulate the output projection followed by top-k token selection as a maximum inner product search over token embed...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Accelerating LLM Inference via Vector Index Based Output Embeddings*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**自回归解码**（autoregressive decoding）过程中，尤其是对具有**大规模多语言词汇表**（large multilingual vocabularies, 100k–250k tokens）的紧凑型 LLM（如 Gemma、Llama3.2、Qwen3），输出层的 **dense output projection** 成为显著的内存带宽瓶颈。  
尽管模型主干较小，但每一步都需要将整个输出嵌入矩阵从内存中加载以计算 logits，造成不必要的开销。

### 🚀 提出的新方法
提出一种基于 **Vector Index** 的输出头机制，将传统的输出投影 + top-k 采样重构为一个 **Maximum Inner Product Search (MIPS)** 问题：
- 将 `W_out` 的每一行视为 token embedding 向量，构建一个 **HNSW**（Hierarchical Navigable Small World）向量索引。
- 在推理时，直接通过 HNSW 搜索与当前 hidden state `h` 内积最大的 top-k 个 token，避免全量矩阵乘法。
- 支持无缝集成到现有解码流程中：将检索到的 logits 散布（scatter）回一个稀疏的全词表大小张量中，兼容标准 logit processor（如 top-p、min-p）。

### 🔍 相比现有方法的优势
| 对比维度 | 本文方法 | 传统方法 / 其他近似方法 |
|--------|--------|---------------------|
| **无需重训练** | ✅ 完全 post-training，适用于已有模型 | ❌ 如 Adaptive Softmax 需预训练阶段修改架构 |
| **内存效率高** | ✅ 显著减少内存带宽使用（仅访问少量 embedding） | ❌ Dense GEMM 加载全部权重 |
| **可插拔设计** | ✅ “drop-in” 替换输出头，兼容主流框架 | ⚠️ 多数优化需定制实现 |
| **适合小批量/边缘部署** | ✅ 在 batch size = 1 场景下优势明显 | ❌ Batched GEMM 在大批次才高效 |

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **模型家族**：
  - **Gemma 3**（270M, 1B；vocab: 256k）
  - **Llama 3.2**（1B；vocab: 128k）
  - **Qwen 3**（0.6B, 1.7B；vocab: 152k）
- **输入数据**：
  - 微基准测试：真实 hidden states 提取自处理 Wikipedia 文章的模型中间状态。
  - 端到端生成任务：短问题 → 生成 128-token 回应。

### ⚙️ 实验设置
- **硬件平台**：Intel Core i7-10750H CPU（6核），32GB DDR4，频率锁定为 2.60GHz。
- **精度格式**：单精度 float32（未使用量化，聚焦算法加速）。
- **HNSW 参数**：
  - `M = 32`, `ef_construction = 5000`
  - 推理时 `ef = 200`（默认），部分实验尝试 `ef=100` 和 `400`
- **采样策略**：top-k sampling (`k=50`)
- **评估指标**：
  - **吞吐量**（Throughput, tokens/sec）
  - **相对加速比**（Speedup ratio vs baseline）
  - **Recall@2**：检索出真实 top-2 最高得分 token 的比例
  - **生成质量**：AlpacaEval 上的 **length-controlled win rate**（LLM-as-a-judge 范式，使用 GPT-5 Nano 作为裁判）

### 🆚 基线方法
- **Baseline**：标准 dense output projection（即 `z = W_out @ h`）
- 所有比较均在同一模型 checkpoint 下进行，仅替换输出头。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）输出投影层微基准（Figure 1）
- 在 **batch size = 1** 时：
  - **Gemma 模型输出层加速 >12×**
  - 加速效果随词汇量增大而增强（Gemma 256k > Qwen 152k > Llama 128k）
- 小 batch 下内存带宽是瓶颈，vector index 显著缓解该问题。

#### （2）端到端解码吞吐（Figure 2）
- **Gemma 3 270M 在 batch size = 1 时，端到端吞吐提升达 82%**
- 性能增益随模型参数占比中 output embedding 比例升高而增加（该模块占 Gemma 3 270M 总参数约 62%）
- 当 batch size > 32 时，优势减弱甚至反转 —— 因为多个查询导致缓存失效，密集 GEMM 更优。

#### （3）检索准确率（Table 1: Recall@2）
| Model | ef=100 | ef=200 | ef=400 |
|-------|--------|--------|--------|
| Gemma 3 270M | 97.3% | **99.1%** | 99.7% |
| Llama 3.2 1B | 97.9% | **99.5%** | 99.9% |
| Qwen3 1.7B | 96.3% | **99.0%** | 99.8% |

- `ef=200` 可实现 >99% 的 Recall@2，已足够维持生成质量。
- `ef=100` 虽然速度更快（吞吐翻倍），但略有质量下降。

#### （4）生成质量评估（Table 2: AlpacaEval Win Rate）
| Model | Length-Controlled Win Rate |
|-------|----------------------------|
| Gemma 3 270M | 48.1 ± 0.10% |
| Llama 3.2 1B | 49.1 ± 0.25% |
| Qwen3 1.7B | 46.9 ± 0.22% |

- 所有模型 win rate 接近 50%，说明 **judge 几乎无法区分优化模型与原模型**。
- 表明生成质量损失极小，可忽略不计。

#### （5）消融实验（Ablation）
- **special token 强制保留**：显式计算常用标点、结构词等特殊 token 的 logits，防止因近似搜索遗漏关键符号。
- **ef 参数权衡**：`ef=200` 是精度与速度的最佳平衡点；进一步提高带来边际收益递减。
- **profiler trace 分析**（Appendix Figure 3）显示：baseline 中 output projection 占主导延迟；vector index 极大缩短此阶段。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **大型输出嵌入矩阵是小型 LLM 的主要瓶颈**，尤其在边缘设备的小批量推理场景。
2. **将输出投影转化为 MIPS 问题是可行且高效的**，利用 HNSW 实现近似检索可在几乎无损质量的前提下大幅提升推理速度。
3. **vector index embedding 在 batch size = 1 时表现最佳**，最高实现 **82% 的端到端吞吐提升**。
4. **Recall@2 > 99% 即可有效保持生成质量**，证明 softmax 分布主要由前几个高分 token 决定。
5. 该方法完全兼容现有模型和解码 pipeline，具备良好的工程落地潜力。

### ⚠️ 局限性
- **依赖 CPU 执行**：HNSW 图遍历具有高度随机访存特性，在 GPU 上难以并行化，目前缺乏高效 GPU 实现。
- **静态内存开销略增**：由于存储图结构，output layer 存储占用增加约 5–10%（从 `|V|×4d` 到 `|V|×(4d + 8M)` 字节）。
- **大 batch 场景不适用**：当 batch size 较大时，密集 GEMM 的缓存友好性和并行性更优，本方法失去优势。
- **对重度量化的 baseline 提升有限**：若 baseline 已采用 GPTQ 等量化技术，相对增益可能缩小。

### 🔮 未来工作方向
1. 开发 **GPU-accelerated vector index**（如支持 CAGRA 或定制 CUDA kernel）以释放并行潜力。
2. 结合 **weight quantization** 进一步压缩索引内存占用。
3. 探索动态调整 `ef` 策略，根据上下文置信度自适应控制搜索深度。
4. 扩展至其他任务，如 **reranking、retrieval-augmented generation** 中的 token-level 检索。

---

> 🔗 **开源代码**：https://github.com/martinloretzzz/vector-index-embedding  
> 💡 **适用场景推荐**：移动端、车载系统、工业边缘设备等低延迟、小批量、资源受限环境下的 LLM 部署。

</details>

---

### 6. [TerraceMoE: A Cost Model for Hierarchical MoE All-to-All Communication](https://arxiv.org/abs/2608.27874)

**Authors**: Weicheng Xue, Bingqiang Wang, Li Yuan, Huihui Zhou, Yonghong Tian  
**Category**: cs.DC  
**Published**: 2026-08-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.27874v1  

#### Abstract
Hierarchical two-hop dispatch can reduce slow-fabric traffic in expert-parallel Mixture-of-Experts training, but it adds a second collective and an arrival-side operator chain. We present a cost model for screening that trade at the communication-call level, bounded by validation gates that withdraw...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TerraceMoE: A Cost Model for Hierarchical MoE All-to-All Communication

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Mixture-of-Experts (MoE)** 模型在 **expert-parallel** 分布式训练中通信开销过大的问题，特别是 **All-to-All** 通信成为训练瓶颈的现象。传统的 **flat dispatch** 在跨节点通信时会产生大量冗余流量。

为缓解此问题，已有研究提出 **Hierarchical Two-Hop Dispatch**（即先将token发送到目标组的代表rank，再在组内scatter），但其是否真正有效取决于硬件拓扑、实现细节和模型配置等复杂因素。然而，构建并实测两种路径成本高昂，且负向结论无法被验证。

因此，本文要解决的核心问题是：**能否在不实际构建系统的情况下，准确预测 hierarchical MoE dispatch 是否优于 flat dispatch？**

### 提出的新方法与思路
作者提出了 **TerraceMoE** —— 一个用于建模 hierarchical MoE All-to-All 通信成本的分析模型，并引入了 **validation gates** 机制来增强模型的可信度。

- **T-A2A**: 表示所提出的两跳 dispatch 方法。
- **T-Route**: 一种新的路由约束，要求每个token的专家仅分布在 $M$ 个组中，且在每个选中的组中恰好选择 $k/M$ 个专家，以保证两跳 dispatch 可表达（expressible）。
- **Cost Model**: 基于 Hockney 模型，对 one-hop 和 two-hop 的通信时间进行建模，显式计入：
  - 集体通信的固定开销（$\alpha$）
  - 带宽项（受半性能消息大小 $x_{1/2}$ 影响）
  - 到达端的处理开销（arrival chain cost $C_{\text{chain}} \cdot Tk$）
  - 拆分大小获取开销（$t_{\text{splits}}$）

### 相比现有方法的优势
- **可解释性与可问责性（Accountability）**：不同于以往仅报告平均误差的模拟器，本文采用 **validation gates**，若某个门限未通过，则直接移除某项能力（如不能进行端到端吞吐量预测），而非添加免责声明。
- **分离关注点**：将 **routing constraint** 的代价（如对模型质量的影响）与 **transport**（通信实现）的代价分开评估。
- **强调实现细节的重要性**：证明了 arrival chain 的实现开销对最终决策有决定性影响，远超互联带宽比本身。

---

## 2. 核心实验方法和设置

### 实验平台
使用两台机器进行校准与验证：

| Machine | Topology | R | B_fast | B_slow | B_fast/B_slow |
|---------|----------|----|--------|--------|----------------|
| A       | unified-fabric supernode | 8 | 122.6 GB/s | 119.5 GB/s | **1.03** |
| B       | same family, unified fabric | 8 | 111.9 GB/s | not separated | not measured |

> 注意：两台机器均未进入 hierarchical regime（即 $B_{\text{fast}}/B_{\text{slow}} \gg 1$），所有高比率场景均为合成推演。

### 数据集与微基准测试（Corpora）
共设计六组有明确目标的测试集（corpora）：

| ID | 内容 | 作用 |
|-----|------|-------|
| C1 | 直接测量 one-hop 和 two-hop dispatch 时间 | 用于拟合模型参数（Tier-1） |
| C2-C4 | All-to-All 大小扫描（不同机器/规模） | 验证模型泛化性（Tier-1b） |
| C5 | Machine A, world 8 的大小扫描 | 独立验证（Tier-1b） |
| G1 | 端到端 step time 测试（7种几何配置） | 验证端到端预测能力（Tier-2） |
| Q1 | 路由约束对模型质量的影响（13.14B 参数模型） | 评估 T-Route 对 validation loss 的影响 |

### 评估指标与门限（Gates）
- **Tier-1**: 通信级预测误差 ≤20%（中位数），最差情况 ≤35%
- **Tier-1b**: 更严格的误差控制（≤12%），用于跨机器/跨基准验证
- **Tier-2**: 端到端 step time 预测的 MAE ≤ 0.025，且至少 4/6 个 holdout 在 ±0.035 内
- 若 **Tier-2 失败**，则模型**禁止输出任何 step-level 预测**

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）通信级预测表现
- **Tier-1 (C1)**: 中位相对误差 **4.1%**，最差 **24.5%** → ✅ 通过
- **Tier-1b (C2-C4)**: 所有误差均 < 10% → ✅ 通过
- **Tier-1b (C5)**: 中位误差 **15.1%**，偏差 -9.1% → ❌ **失败**

> 尽管 C5 失败，但整体通信级模型仍被认为可用。

#### （2）端到端预测表现
- **Tier-2 (G1)**: MAE = **0.135**，0/6 在容忍范围内 → ❌ **失败**

> 因此，**论文明确声明不做任何 step-level 吞吐量预测**。

#### （3）breakeven hierarchy ratio（关键阈值）
在参考配置（16组×8 rank, $q=3$, $H=2048$, 4096 tokens/rank）下，两跳优于一跳所需的 hierarchy ratio 阈值为：

| 实现方式 | Effective Breakeven Ratio |
|----------|----------------------------|
| **PyTorch 到达链（实测）** | **3.98** |
| **假设融合核函数（hypothetical fused target）** | **1.49** |
| **零实现开销（理论极限）** | **1.10** |

> 这表明 **arrival chain 的实现效率是决定性因素**，其影响远超互联架构本身。

#### （4）T-Route 路由约束的质量代价
在 13.14B 模型上测试 T-Route 对 validation loss 的影响：

- **T-Route (+ group limit & equal quota)**: **+0.0034 nats**
- 仅为预设“无损失”边界（0.1 nats）的 **3.4%**，影响极小。

下游任务（HellaSwag, LAMBADA）显示等效性，但因估计器未完全公开，不可独立复现。

---

## 4. 关键结论和发现

### 主要发现
1. **Hierarchy ratio 不足以决定 dispatch 优劣**：传统观点认为只要 $B_{\text{fast}}/B_{\text{slow}} > r_{\text{break}}$ 即可，但本文证明 **arrival chain 的实现开销** 是更关键的因素。
2. **Arrival chain 是主导敏感性来源**：相比增加一次集体通信的开销，**arrival chain 的处理时间** 对 breakeven ratio 的影响大一个数量级以上。
3. **T-Route 约束代价极小**：所提出的路由约束对模型质量的影响微乎其微（+0.0034 nats），可在不显著损失性能的前提下启用两跳 dispatch。
4. **端到端预测不可靠**：由于通信与计算重叠（overlap）难以精确建模，当前方法无法可靠预测 step time，故 **Tier-2 门限失败**。

### 方法的局限性
- **未解决 overlap 建模问题**：通信与计算的重叠程度依赖调度策略，难以用单一参数刻画。
- **高 world size 外推不可靠**：唯一覆盖 256/512 ranks 的数据集存在低置信度问题，因此论文撤回了对大规模集群的性能增益声明。
- **Fast domain 必须均匀**：模型假设组内带宽一致，不适用于如 Frontier 这类组内拓扑非均匀的机器。
- **Artifact 不支持完整复现**：虽开源代码与常数，但未提供原始测量数据与完整估计器输入。

### 未来工作方向
- 构建能精确测量 **communication-compute overlap** 的工具。
- 在真实 hierarchical cluster（$B_{\text{fast}}/B_{\text{slow}} \gg 1$）上验证模型。
- 开发更高效的 arrival chain 实现（如 fused kernel），以降低 breakeven ratio。
- 建立更通用的非均匀 fast domain 建模框架。

---

> **总结**：TerraceMoE 并非一个“更快”的 MoE 通信系统，而是一个**可问责的成本模型**。它揭示了在部署 hierarchical dispatch 前必须测量的关键参数（尤其是 arrival chain 性能），并通过严格的 validation gates 避免做出不可靠的预测，为系统设计提供了坚实、透明的决策依据。

</details>

---

### 7. [Memory-efficient GPU pipelines for real-time non-line-of-sight reconstruction](https://arxiv.org/abs/2608.28183)

**Authors**: Alfonso L\'opez-Ruiz, Diego Royo  
**Category**: cs.DC  
**Published**: 2026-08-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.28183v1  

#### Abstract
Non-line-of-sight (NLOS) imaging reconstructs scenes hidden around a corner from indirect light recorded by a single-photon avalanche diode (SPAD). A single reconstruction is a large inverse problem: billions of photon timestamps must be binned, moved through memory, transformed and inverted. As SPA...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Memory-efficient GPU pipelines for real-time non-line-of-sight reconstruction

## 1. 论文的主要贡献和创新点

### 解决的问题
非视距成像（**Non-line-of-sight, NLOS**）通过分析从可见中继墙散射的间接光来重建隐藏场景。单次重建是一个巨大的逆问题，涉及数十亿光子时间戳的分箱、内存传输、变换和求逆。随着 **SPAD** 阵列提高采集吞吐量，**重建已成为限制性阶段**。本文旨在解决 **NLOS 重建的计算瓶颈**，特别是其高内存占用和低处理速度，使其无法满足实时或大规模重建的需求。

### 提出的新方法和新思路
作者为两种主流的基于波的 **NLOS** 算法——**f-k migration** 和 **phasor-fields**——重新设计了高效的 **GPU** 执行流水线，适用于流式（streaming）和离线（offline）处理。其核心创新在于一系列算法和系统级优化：

1.  **离线构建环形核（Offline Construction of Ring-and-Radius Kernels）**：
    *   针对 **phasor-fields** 算法，作者利用**环的傅里叶变换具有解析解**（即零阶贝塞尔函数 $J_0$）这一数学性质。
    *   他们将 **Jiang et al. (2022)** 提出的“环与半径”表示法进一步优化，**在预计算阶段一次性构建紧凑的径向核（radial kernel）**，并将其存储为 `[p, f, d]`（傅里叶半径、频率、深度）的数组。
    *   **关键优势**：在运行时，传播核不再以密集形式存在，而是通过查找表直接采样。这**彻底消除了运行时重建密集核的步骤**，显著降低了内存带宽和峰值显存占用。

2.  **系统级 GPU 流水线优化**：
    *   **融合内核（Fused Kernels）**：将多个小的 **CUDA** 内核合并，减少内核启动开销。
    *   **线程束级光子分箱（Warp-level Photon Binning）**：在 **phasor-fields** 的分箱阶段，一个 **warp** 负责一个光子，其内部线程覆盖不同的时间频率，减少了调度开销和内存访问冲突。
    *   **批处理变换（Batched Transforms）**：使用批处理的 **cuFFT** 操作，提高 **FFT** 效率。
    *   **CUDA Graph Replay**：将整个重建流程记录为 **CUDA Graph** 并重放，大幅降低重复执行相同序列时的调度开销。
    *   **选择性使用 FP16**：仅在能真正缓解瓶颈的地方（如存储 **phasor-fields** 的传播数据）使用 **FP16**，避免因转换开销或 **cuFFT** 对幂次尺寸的要求而得不偿失。

### 相比现有方法的优势
*   **速度更快**：在流式处理上，比 **Nam et al. (2021)** 的参考管道快 **42倍**；在全多平面体积重建上，比最快的已发表 **GPU** 基线 **Physics to the Rescue (PttR)** 快约一个数量级（7.7x 到 14.0x）。
*   **内存效率极高**：峰值显存占用仅为 **PttR** 的 **2.5%** 左右（例如，0.22-0.29 GB vs 8.9-12.7 GB），是参考 **phasor-fields** 流水线的 **8.2%**（0.27 GB vs 3.34 GB）。
*   **扩展性更强**：极低的内存占用使得在相同硬件上可以进行更大规模、更精细的重建，或者在更低端的消费级 **GPU** 上实现同等规模的重建。
*   **支持实时视频处理**：为下一代 **NLOS** 视频处理提供了充足的帧预算，从而启用了新的去噪策略。

---

## 2. 核心实验方法和设置

### 使用的数据集
*   **动态 SPAD 数据集**：来自 **Nam et al. (2021)** 的 `nlosbox1` 数据集，包含原始光子时间戳，用于评估**流式处理**性能。
*   **标准离线数据集**：来自 **Lindell et al. (2019)** 和 **Galindo et al. (2019)** 的公开数据集（如 `bike`, `teaser`, `statue`, `usaf`, `bunny`, `Z`, `office`），用于评估**离线重建**性能。
*   **模拟数据集**：一个受 **Liu et al. (2020)** 启发生成的 `office` 场景数据集，用于测试大体积重建。

### 实验设置和评估指标
*   **硬件平台**：Intel Core i7-14700KF CPU, **NVIDIA RTX 4080 SUPER GPU (16 GB VRAM)**, 64 GB RAM。
*   **评估模式**：
    *   **流式处理（Streaming）**：模拟实时采集，一边处理当前帧，一边采集下一帧。评估指标为**每秒帧数（FPS）** 和**峰值显存（Peak VRAM）**。
    *   **离线处理（Offline）**：所有瞬态数据均已可用。评估指标为**重建时间（Time）**、**吞吐量（Mvox/s，百万体素/秒）** 和**峰值跟踪内存（Max. tracked memory）**。
*   **主要评估指标**：
    *   **重建速度/吞吐量**：FPS 或 Mvox/s。
    *   **内存效率**：峰值显存（VRAM）或峰值跟踪内存（GB）。
    *   **等效性**：与参考实现的均方绝对误差（Mean Absolute Difference）、FLIP 误差。
    *   **去噪效果**：背景噪声水平、对比度（Contrast）、锐度（Sharpness）。

### 基线方法对比
*   **流式基线**：
    *   **Phasor-fields (Nam et al., 2021)**：实现了 5 FPS 的实时 **NLOS** 成像，是本文流式处理的主要对比对象。
*   **离线/高性能基线**：
    *   **Physics to the Rescue (PttR, Mu et al., 2025)**：当时最快、基于 **PyTorch** 的 **GPU** 基线，是本文在吞吐量上的主要比较目标。
    *   **f-k migration (Lindell et al., 2019)** 和 **Phasor-fields (Liu et al., 2019)** 的原始 **MATLAB** 实现：作为离线处理的慢速参考。
    *   **FPGA-accelerated (Liao et al., 2022)**：基于 **FPGA** 的加速器，提供硬件对比。
    *   **Backprojection (Arellano et al., 2017; Sun et al., 2026)**：作为传统方法的代表，因其速度慢而主要用于展示不同方法家族的扩展性差异。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
*   **流式处理性能**（在 `nlosbox1` 动态数据集上）：
    *   **我们的 f-k (无填充)**：高达 **919.9 FPS**，峰值显存 **0.30 GB**。
    *   **我们的 f-k (有填充)**：**81.4 FPS**，峰值显存 **1.08 GB**。
    *   **我们的 phasor-fields**：**448.5 FPS**，峰值显存 **0.27 GB**。
    *   **对比**：**Nam et al. (2021)** 的 **phasor-fields** 为 **10.7 FPS**，峰值显存 **3.34 GB**。

*   **离线处理性能**（在 `office` 场景上，360×260×512 体素）：
    *   **我们的 phasor-fields**：**17,745.27 Mvox/s**，峰值显存 **0.219 GB**。
    *   **PttR (Mu et al., 2025)**：**1,269.18 Mvox/s**，峰值显存 **12.707 GB**。
    *   **对比**：我们的方法比 **PttR** 快 **14.0倍**，显存占用仅为 **1.7%**。

*   **总体性能提升**：
    *   相比 **PttR**，在多平面重建上快 **7.7x - 14.0x**，显存占用降至 **2.5%**。
    *   相比 **Nam et al. (2021)** 的流式管道，速度快 **42x**，显存占用为 **8.2%**。
    *   在离线处理上，相比原始 **MATLAB** 实现，重建时间平均减少 **98.3% (f-k)** 和 **96.1% (phasor-fields)**。

### 与基线方法的对比结果
*   **速度**：在所有可比设置下，本文提出的方法在速度上均显著超越所有 **GPU** 和 **CPU** 基线。
*   **内存**：内存占用远低于所有基于密集核的 **phasor-fields** 实现（如 **PttR** 和 **Nam et al.**），差距可达两个数量级。
*   **可扩展性**：如 **Table 5** 所示，在最大填充配置（720×520×1024）下，**f-k** 和 **MoE** 因内存不足（OOM）而失败，而**我们的 phasor-fields** 仍能以 **4.53 GB** 显存成功完成，凸显了其卓越的内存效率。

### 消融实验结果
*   **CUDA Graph Replay**：带来 **1.3%**（f-k）到 **0.3%**（phasor-fields）的性能提升，证明了减少启动开销的有效性。
*   **Warp-per-photon 分箱**：相比逐光子处理，显著提升了高光子计数下的吞吐量。
*   **精度（FP16 vs FP32）**：
    *   对于 **phasor-fields**，**FP16** 有效降低了存储传播数据的显存。
    *   对于 **f-k**，**FP16** 可能因 **cuFFT Xt** 对幂次尺寸的要求和转换开销而**变慢**，表明精度优化需谨慎应用。
*   **填充（Padding）**：填充能减少 **FFT** 混叠伪影，但会增加时间和内存。本文提供了权衡方案。

---

## 4. 关键结论和发现

### 主要发现
1.  **内存是主要瓶颈**：对于现代 **NLOS** 重建，尤其是 **phasor-fields**，**内存带宽和容量是比计算本身更关键的瓶颈**。
2.  **算法表示至关重要**：通过**离线构建解析的径向核**，将密集核表示转化为紧凑的径向表示，是实现极致内存效率的核心。
3.  **系统级优化不可或缺**：**CUDA Graph**, **warp-level** 编程等系统级优化对于消除小内核带来的开销至关重要。
4.  **没有万能优化**：**FP16** 等技术并非总是有益，必须针对具体瓶颈应用。
5.  **高帧率赋能新应用**：极高的重建速度不仅提升了性能，还为**多帧聚合去噪**（如 **DDA**, **Coherence-weighted merging**, **MoE**）等新策略创造了可能。

### 方法的局限性
*   **径向近似引入微小误差**：紧凑的径向核是对理想密集核的近似，会丢失一些角向结构，导致重建结果与参考实现存在微小视觉差异（如 **Figure 16** 所示）。
*   **依赖特定硬件特性**：充分利用了 **NVIDIA GPU** 的 **CUDA** 特性（如 **warp**, **shared memory**, **CUDA Graph**），移植到其他架构可能需要调整。
*   **未解决信噪比（SNR）问题**：本工作专注于计算瓶颈，而非物理层面的低 **SNR** 问题。

### 未来工作方向
*   **集成到真实 **SPAD** 阵列**：将该流水线与实际的高速 **SPAD** 相机和数据接口集成，验证端到端性能。
*   **异步上传和数据移动优化**：在高光子计数下，数据移动和分箱仍是成本，需要更好的异步上传机制。
*   **更优的变换调度和传播表示**：探索更高效的 **FFT** 调度策略和更紧凑的传播算子表示。
*   **开发保留时序细节的去噪策略**：利用高帧率优势，设计既能降噪又能保留动态场景细节的算法。

</details>

---

### 8. [A Method for Layer Bit-Width Allocation in LLM Quantization via Performance Maximization Under a Quality-Degradation Constraint](https://arxiv.org/abs/2608.28003)

**Authors**: Artem Safronov  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.28003v1  

#### Abstract
This paper proposes a layer bit allocation method for Gemma-3-1B, formulating the problem as performance maximization (latency decrease) given a degradation budget constraint (allowable level of generation quality loss). This approach is different from time- and resource-consuming uniform layer quan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Method for Layer Bit-Width Allocation in LLM Quantization via Performance Maximization Under a Quality-Degradation Constraint

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**大语言模型（LLM）推理阶段的量化效率瓶颈**，特别是：
- 传统均匀量化（如 GPTQ、AWQ）虽能压缩模型，但对**推理延迟的加速有限**；
- 现有混合精度方法（如 MixLLM、TorchAO）缺乏对**实际硬件性能增益的实证支持**，且未系统分析量化带来的**dequantization/quantization（Q/DQ）开销**；
- 在低 batch size（如 batch=1）场景下，如何在控制生成质量退化的同时最大化吞吐量（tokens/sec）。

目标是：**在给定的质量损失预算内，最大化推理性能（最小化延迟）**。

---

### 🚀 提出的新方法与创新思路
作者提出了一种基于**层敏感度分析**的**分层混合精度量化策略**，其核心创新包括：

1. **基于 SA-PTQ 的层重要性度量进行 bit-width 分配**  
   利用前序研究中提出的 **SA-PTQ（Sensitivity-Aware Post-Training Quantization）** 方法构建模型各层的敏感度画像，识别出对量化鲁棒性强的模块（如 FFN 外层、lm_head），优先将其量化为 INT8。

2. **块级分组量化（Block-wise Grouping）与独立评估**  
   将模型划分为三大子系统分别处理：
   - **FFN blocks**（gate_proj, up_proj, down_proj）
   - **Attention projections**（qkv, dense）
   - **lm_head**（输出投影层）
   
   并采用三种分组策略进行实验：
   - `5+5`：仅量化第 0–4 和 21–25 层
   - `10+10`：量化第 0–9 和 16–25 层
   - `all26`：全部 26 层量化

3. **通过 TensorRT-LLM 实现 activation pass-through 模式下的真实性能测量**  
   不同于以往“假设”混合精度有益的做法，本文在 **TensorRT-LLM 1.2.1** 中实现了完整的 W8A8 量化流程，并使用 **Nsight Systems/Compute** 工具精确测量 Q/DQ 开销及其对 kernel 执行时间的影响。

4. **手动实现 SmoothQuant 对 lm_head 的支持**  
   发现官方工具链（ModelOpt + TensorRT-LLM）无法正确导出 Gemma 模型中 tied embedding 结构的 lm_head 量化权重，因此**自行实现了 SmoothQuant 流程**以绕过此限制。

---

### 🔍 相比现有方法的优势
| 方法 | 缺陷 | 本工作的改进 |
|------|------|---------------|
| **GPTQ / AWQ** | 统一量化所有层，忽略层间敏感性差异；无性能加速证明 | 引入敏感度感知分配，聚焦可加速模块 |
| **MixLLM / TorchAO** | 虽提混合精度，但未提供 per-layer 性能收益证据 | 实测每层 FLOPs 和 token/s，验证加速来源 |
| **通用 INT8 推理框架** | 忽视 Q/DQ wrapper 的额外开销 | 明确测量并揭示 overhead 成本，指出何时“得不偿失” |

> ✅ **核心优势**：首次将“**性能最大化 + 质量约束**”形式化为一个可操作的优化问题，并通过真实硬件测量验证了不同配置的实际效果。

---

## 2. 核心实验方法和设置

### 📊 数据集与校准集
- **校准数据集（Calibration Set）**：使用 **5 个文本样本** 进行 SmoothQuant 参数估计（α=0.5），用于计算 per-channel scaling factors。
- **测试提示（Prompt）**：固定输入 prompt，生成 100 或 50 个 token，用于性能 benchmark。
- **质量评估数据集**：
  - 一个 **181-token 的 held-out 文本集**（非重叠）
  - 另一个独立问答任务集（用于 Top-1 agreement 和 perplexity 测量）

---

### ⚙️ 实验设置
- **模型**：Gemma-3-1B（FP16 基线）
- **硬件平台**：NVIDIA RTX 5090（Blackwell 架构，sm_120）
- **软件栈**：
  - TensorRT-LLM 1.2.1
  - ModelOpt 用于量化校准
  - trtllm-build 编译引擎
- **批量大小**：`batch_size = 1`（关注低延迟场景）
- **上下文长度**：短上下文（input_length ≤ 512），decode 阶段为主
- **量化方案**：W8A8（Weight 8-bit, Activation 8-bit），对称 per-channel 量化

---

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **tok/s** | 每秒生成 token 数，主性能指标 |
| **Speedup** | 相对于 FP16 baseline 的加速比 |
| **SQNR（Signal-to-Quantization-Noise Ratio）** | 衡量某一层输出相对于 FP16 的保真度（dB） |
| **Top-1 Agreement** | 量化模型与原始模型预测最可能 token 的一致率 |
| **Perplexity Degradation** | 困惑度上升百分比，衡量语义一致性下降程度 |
| **Nsight Profiling** | 包括 kernel 执行时间、DRAM 吞吐、SM 占用率等底层指标 |

---

### 🔁 基线方法对比
| 基线 | 类型 | 是否参与比较 |
|------|------|--------------|
| **FP16 Baseline** | 全精度参考 | ✔️ 是 |
| **GPTQ / AWQ** | 统一 INT4/INT8 量化 | ❌ 未直接运行，作为背景讨论 |
| **MixLLM / TorchAO** | 混合精度推理框架 | ❌ 无实测数据，仅理论对比 |
| **Uniform W8A8** | 所有层统一 INT8 | ❌ 未单独列出，隐含在 all26 中 |

> 注：本文更侧重于**内部消融比较**而非跨框架横向评测。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（见 Table 6）

| 配置 | tok/s | Speedup | Top-1 Agreement | Perplexity Δ |
|------|--------|---------|------------------|-------------|
| **FP16 baseline** | 380.53 | 1.000× | 100% | 0% |
| **lm_head only** | 407.49 | 1.071× | 98.90% | +0.85% |
| **FFN 5+5** | 393.5 | 1.031× | — | — |
| **FFN 10+10** | 408.6 | 1.075× | — | — |
| **FFN all26** | 414.98 | 1.090× | — | — |
| **FFN 5+5 + lm_head** | 422.14 | **1.110×** | **98.90%** | **+0.85%** |
| **FFN all26 + lm_head** | 453.27 | **1.191×** | ~80% | 显著退化 |
| **Full INT8 (all + attn)** | 440.85 | 1.157× | 严重退化（重复崩溃） | — |

---

### 🔬 分项实验结果

#### ✅ FFN 量化结果
- **加速显著**：随着更多层被量化，速度持续提升。
- **最佳平衡点**：`5+5` 配置达到 **1.031× speedup**，同时保持高质量。
- **原因分析**：TRT-LLM 使用 `sm80_xmma_lgemm_i8i8` kernel 加速 INT8×INT8 GEMM，在 Blackwell 上可全层启用。

#### ⚠️ Attention 量化结果
- **反直觉现象**：尽管 Attention 权重占比高，但**量化后反而变慢**。
- **attn_ffn all26 vs FFN all26**：
  - 前者：407.28 tok/s（1.069×）
  - 后者：414.98 tok/s（1.090×）
  - ➜ **引入 Attention 量化导致 -7.7 tok/s 的性能倒退**
- **根本原因**：
  - FlashAttention 内核本身高度优化（FP16 fused kernel）
  - 添加 Q/DQ wrapper 后破坏融合路径，引入两个额外 CUDA kernel
  - 实际仅 qkv/dense GEMM 被加速，而核心 attention kernel 仍运行在 FP16
  - GQA 架构使 qkv 矩阵较小 → 加速收益不足以抵消 overhead

#### ✅ lm_head 量化结果
- **成功突破官方限制**：手动实现 SmoothQuant 支持 tied embedding 的 lm_head 量化
- **性能增益明显**：单独量化 lm_head 即带来 **1.071× speedup**
- **质量几乎无损**：Top-1 agreement 达 **98.90%**，perplexity 仅上升 **+0.85%**
- **SQNR 高达 45.6 dB**，说明该层极其鲁棒

#### 🔗 组合配置表现
- **最优组合**：`FFN 5+5 + lm_head`
  - **1.110× speedup（+11.0%）**
  - 生成质量“good”，无重复崩溃
- **最大加速组合**：`FFN all26 + lm_head`
  - **1.191× speedup（+19.1%）**
  - 代价是生成质量“degraded”（出现重复 token）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **FFN 和 lm_head 是 W8A8 量化的理想候选模块**  
   - 二者均受益于 INT8 计算加速，且 Q/DQ 开销可被完全补偿。
   - lm_head 特别鲁棒，SQNR 高达 45.6 dB。

2. **Attention 量化在当前实现下“适得其反”**  
   - 在 batch=1、短上下文场景中，**添加 Attention 量化会降低整体性能**。
   - 根本原因是：**Q/DQ wrapper 破坏了原本高效的 FP16 fused attention kernel**。

3. **混合精度并非自动带来加速，必须考虑工程实现细节**  
   - “理论上应该更快” ≠ “实际上真的更快”
   - **真正的瓶颈不在数值容忍度，而在 kernel 融合程度**

4. **非量化中间层具有误差补偿作用**  
   - 实验显示，保留中间若干层为 FP16 可缓解首尾层量化误差传播，提升整体 SQNR。

5. **Blackwell 架构对 INT8 更友好**  
   - 相比 Ampere（如 RTX 3070），Blackwell（RTX 5090）能稳定使用 `i8i8` kernel，即使全层量化也不降速。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖特定框架（TensorRT-LLM）** | 实验结果受限于 TRT-LLM 的 kernel 实现逻辑 |
| **未解决 Attention 内核融合问题** | 当前仅暴露问题，尚未提出新 kernel 实现 |
| **短上下文设定** | 长 context 下 KV-cache 成为主要瓶颈，结论可能变化 |
| **缺乏跨模型泛化验证** | 仅在 Gemma-3-1B 上验证，是否适用于其他架构待观察 |
| **官方工具链缺陷需手动绕过** | lm_head 量化需 hack，不利于部署自动化 |

---

### 🔮 未来工作方向
1. **开发 fused INT8 Attention Kernel**  
   - 类似 FlashAttention-int8 或 SageAttention，将 Q/DQ 融入 kernel 内部
   - 只有这样才能真正释放 Attention 量化的潜力

2. **KV-Cache Quantization**  
   - 将 KV-cache 从 FP16 压缩至 INT8/FP8，尤其在长 context 场景下意义重大
   - TRT-LLM 已支持 `kv_cache_quant_algo`，值得进一步探索

3. **Partial Attention Quantization（类比 FFN 5+5）**  
   - 仅量化外层 Attention 层（SQNR 更高），跳过中间敏感层
   - 可能实现“质量不变 + 少量加速”

4. **转向 FP8 精度**  
   - Blackwell 支持原生 FP8 tensor core，峰值吞吐高于 INT8
   - 实验表明 FP8 FFN 可达 **1.46× speedup**

5. **推动官方修复 tied embedding 量化 bug**  
   - 当前 lm_head 量化需手动实现，应提交 patch 至 NVIDIA 官方维护分支

6. **探索 calibration 策略差异化设计**  
   - Attention 投影（q/k/v/o）对误差更敏感，建议采用更保守的 calibration（如 AWQ-style channel weighting）

---

## ✅ 总结一句话
> 本文提出一种基于 **SA-PTQ 敏感度分析** 的分层 W8A8 量化方法，在 **Gemma-3-1B** 上实现了最高 **11.0% 延迟降低（1.110× speedup）且质量几乎无损** 的推理加速，关键在于：**只量化 FFN 和 lm_head，避免盲目量化 Attention**，并揭示了当前框架中 **Q/DQ wrapper 导致性能倒退的根本机制**，为后续高效 INT8/FP8 推理系统设计提供了重要指导。

</details>

---

### 9. [SOMTab: Set-Order Mamba for Efficient Tabular In-Context Learning](https://arxiv.org/abs/2608.27882)

**Authors**: Hao Wang, Siyu Zhang, Wei Ma  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.27882v1  

#### Abstract
Tabular foundation models based on in-context learning have recently emerged as strong alternatives to task-specific model fitting. However, the current performance frontier remains dominated by attention-heavy architectures, where attention is used throughout the modeling pipeline. This raises a na...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SOMTab: Set-Order Mamba for Efficient Tabular In-Context Learning**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题
当前主流的 **tabular foundation models**（如 TabPFNv2、TabICLv2）依赖于全注意力机制（Transformer-based attention），虽然在预测性能上表现优异，但在处理长上下文表格时面临显著的计算瓶颈：
- **Attention 的二次复杂度**（quadratic cost）导致推理速度慢、GPU 内存占用高。
- 表格数据本身具有**无序性**（rows 和 columns 无固定顺序），直接应用序列模型（如 Mamba）存在结构不匹配。

因此，核心问题是：  
> **是否必须在整个建模流程中都使用 attention？能否将高效的一维序列建模机制（如 Mamba）用于部分阶段以提升效率？**

---

### 🚀 提出的新方法与新思路

作者提出 **SOMTab**（Set-Order Mamba for Tabular In-Context Learning），其核心思想是：

#### （1）功能解耦：分离 *representation construction* 与 *query-context matching*
- **Representation Construction**（表征构建）：
  - 目标：从原始单元格值中提取列分布特征和行间交互信息。
  - 特点：对输入顺序不敏感（permutation-invariant）。
  - 方法：采用基于 **Mamba** 的状态空间模型进行高效混合。
- **Query-Conditioned Retrieval**（查询条件化检索）：
  - 目标：让测试样本根据标注上下文样例动态获取相关信息。
  - 特点：需要 query-dependent 路由能力。
  - 方法：保留 **attention-based ICL reader**。

> 🔑 **核心假设**：Attention 最有价值的是在最终预测阶段实现 query-context 匹配；而在前期表征学习中可用更高效的线性复杂度算子替代。

#### （2）Set-to-Order Mechanism
为解决表格无序性与 Mamba 序列扫描需求之间的矛盾，引入：
- **可学习的 latent seed slots** 将无序 token 集合映射为有序潜在序列。
- 使用 cross-attention 实现 set-to-slot 映射，保证排列不变性。
- 在 latent slots 上运行 Mamba 进行信息混合，再通过 readback 恢复 token 对齐表示。

#### （3）新型合成先验：DCH-TailMix
为增强 pretraining 任务多样性，提出新的 synthetic prior：
- **DCH**（Degree-Corrected Heterogeneity）：控制图结构异质性（稀疏/密集/模块化等）。
- **TailMix**：混合多种重尾分布（Student-t, Pareto），生成多样化的依赖强度模式。
- 结合二者生成更具现实代表性的因果图结构，提升迁移效果。

---

### ⚖️ 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **效率** | 推理速度快、内存占用低，尤其在长上下文场景下优势明显 |
| **精度** | 接近最强 Transformer 基线（如 TabICLv2），远超传统树模型 |
| **架构设计** | 首次系统论证“attention 不必处处使用”，提出分阶段算子分配原则 |
| **泛化性** | DCH-TailMix 提升 pretraining 多样性，有助于跨数据集迁移 |

---

## 2. **核心实验方法和设置**

### 📚 数据集
- 主要基准：**TALENT**（Ye et al., 2024）
  - 包含多个真实世界 tabular classification 数据集。
  - 官方 train/val 合并作为 context，test 作为 query。
- 辅助基准：**TabArena**（Erickson et al., 2025）
  - 更广泛的 tabular ML benchmark，用于综合比较。

---

### 🧪 实验设置与评估指标

| 类别 | 设置说明 |
|------|----------|
| **评估协议** | 所有方法在同一服务器（H200/A100）、相同 pipeline 下重跑 |
| **预处理** | 使用官方默认配置，不额外调参 |
| **评估指标** | - **Normalized Log-loss**（越小越好）<br>- **Accuracy**, **Macro F1**<br>- **Runtime**（fit+predict 时间 per 1K context samples）<br>- **Peak GPU Memory Usage** |
| **上下文长度扩展实验** | 固定 `ntest=1024`, `m=32`, `C=10`，变化 `ntr` 从 512 到 28,672，验证 scalability |

---

### 🔁 基线方法对比
| 类型 | 方法 |
|------|------|
| **Tree-based** | RandomForest, ExtraTrees, XGBoost, CatBoost |
| **Neural Baselines** | RealMLP |
| **Tabular Foundation Models (PFN-style)** | TabPFNv2, TabICL, TabICLv2（含单 estimator 与 ensemble 版本） |

> 所有 baseline 使用其官方实现和默认参数。

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（来自 Table 1 和 Figure 1）

| 方法 | Norm. LL ↓ | Accuracy ↑ | Macro F1 ↑ | Time/1K (s) ↓ |
|------|------------|-----------|-------------|----------------|
| **SOMTab** | **0.054** | **0.849** | **0.785** | **0.019** |
| TabICLv2 (8 est.) | 0.029 | 0.854 | 0.790 | 0.463 |
| TabICLv2 (1 est.) | 0.046 | 0.853 | 0.789 | 0.151 |
| TabPFNv2 | 0.100 | 0.849 | 0.779 | 0.675 |
| RealMLP | 0.498 | 0.830 | 0.757 | 4.379 |
| CatBoost | 0.309 | 0.826 | 0.755 | 0.508 |

> ✅ **SOMTab 在保持接近最优精度的同时，推理时间仅为 TabICLv2 的 ~4%（单 estimator）或 ~1%（ensemble）！**

---

### 🔍 与基线方法对比结果

#### （1）效率-精度权衡（Figure 1）
- SOMTab 明显位于 **Pareto 前沿**附近。
- 性能仅次于 TabICLv2 ensemble，但速度快一个数量级。
- 显著优于所有 tree-based 和 RealMLP 方法。

#### （2）上下文长度扩展性（Figure 3）
- **Runtime**：SOMTab 增长最缓慢，在大 context 下优势扩大。
- **Memory**：由于避免了 full attention，memory usage 增长接近线性，而 attention-based 方法呈二次增长。
- 即使在小 context 场景下略有内存开销（因 latent slots），但在大规模时全面反超。

#### （3）TabArena 综合表现（Figure 9–10）
- 在不同 GPU（A100/H200）上均表现出良好移植性和竞争力。
- 平均性能差距小，win-rate 分析显示在多数数据集上可击败多数 baseline。

---

### 🔬 消融实验结果（Ablation Study, Figure 4 & Table 6）

| 变体 | 修改内容 | 准确率趋势 |
|------|--------|----------|
| **SOMTab (DCH-TailMix + Attention Reader)** | 全模型（默认） | ✅ 最高准确率 |
| Cauchy Prior | 替换为 TabICLv2 式随机图先验 | ❌ 略差 |
| SCM-Tree Prior | 使用 TabICL 风格混合先验 | ❌ 更差 |
| **Mamba Reader** | 将 final attention 替换为 Mamba | ❌ 明显下降 |

> 💡 **关键发现**：
> - **DCH-TailMix** 显著优于已有 synthetic prior，证明多样化依赖结构的重要性。
> - **Final attention 是必要的**：用 Mamba 替代 final ICL reader 会损害性能，说明 attention 在 query-context matching 中不可替代。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **Attention 并非必需贯穿始终**：
   - 在 representation construction 阶段可用 **Mamba** 等线性复杂度模型有效替代。
   - 在 query-conditioned prediction 阶段，**attention 仍是最优选择**。

2. **Set-to-Order + Mamba 是可行路径**：
   - 通过 learnable seed slots 构造稳定 latent order，解决了表格无序性与序列建模的冲突。
   - 实现了 permutation-equivariant 表示更新。

3. **合成先验的设计至关重要**：
   - **DCH-TailMix** 通过结构异质性 + 重尾混合，提升了 pretraining 多样性，带来更好的泛化性能。

4. **SOMTab 实现了优秀的 speed-accuracy trade-off**：
   - 接近最强 Transformer 模型的精度。
   - 推理速度快 10–30 倍，内存占用更低，适合实际部署。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **仅适用于 ICL 范式** | 依赖 synthetic pretraining，无法像 XGBoost 那样即插即用 |
| **对 very small tables 效率增益有限** | 当 context 很小时，set-to-order 开销可能抵消收益 |
| **Mamba 参数量仍较大** | 虽然比 full Transformer 快，但仍需数千万参数（28.6M） |
| **未支持 regression 或 missing data** | 当前仅针对分类任务，扩展性有待验证 |

---

### 🔮 未来工作方向

1. **探索更多 hybrid 架构组合**：
   - 如在 column-level 使用其他 SSM（如 Hyena、Fourier Mixing）进一步优化。
   
2. **轻量化版本设计**：
   - 压缩 seed slot 数量、降低 hidden dim，适配边缘设备。

3. **拓展至 regression、multi-modal tables**：
   - 支持数值目标变量、文本特征嵌入等。

4. **改进 synthetic prior 的可控性**：
   - 引入 domain-specific prior（如金融、医疗）进行定向预训练。

5. **结合 retrieval-augmented ICL**：
   - 动态选择 relevant context samples，减少冗余 attention 计算。

---

> 📌 **一句话总结**：  
> **SOMTab 成功验证了“attention 用于 query-context matching，Mamba 用于 representation construction”的分阶段设计理念，在保持高性能的同时大幅提升了 tabular ICL 的推理效率，为下一代高效 tabular foundation models 提供了新范式。**

</details>

---

### 10. [Curvature-Conditioned Multiscale Momentum with Sphere Constraints for LLM Pretraining](https://arxiv.org/abs/2608.28442)

**Authors**: Shuchen Zhu, Yuxin Fang, Mingze Wang, Kun Yuan  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.28442v1  

#### Abstract
Pretraining accounts for a large fraction of the total computational cost in LLM training. However, noise-dominant gradients and the highly ill-conditioned loss landscape bring severe challenges. Although modern adaptive optimizers such as AdamW and Muon have achieved great success in large-scale pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Curvature-Conditioned Multiscale Momentum with Sphere Constraints for LLM Pretraining*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在大语言模型（LLM）的预训练阶段，优化过程面临两大挑战：
1. **噪声主导的梯度**：由于批量大小远小于总训练 token 数，随机梯度中的噪声远大于真实梯度信号。
2. **高度病态的损失景观**：损失函数的 Hessian 矩阵具有极不均衡的特征值分布，即存在少量“尖锐方向”（sharp directions）和大量“平坦方向”（flat directions）。最终的损失下降主要由在**平坦方向**上的缓慢进展驱动。

尽管现代自适应优化器（如 AdamW 和 Muon）通过梯度归一化缓解了部分病态问题，但在平坦方向上的训练动态仍然较慢，限制了整体收敛速度。

---

### 提出了什么新方法或新思路
本文提出了一种名为 **Curvature-Conditioned Multiscale Momentum with Sphere Constraints** 的新优化方法，其核心是 **Flat-Direction Multiscale Momentum**，并结合 **Sphere Constraints** 技术以稳定训练。

#### 主要创新点包括：

- **曲率条件化的多尺度动量（Curvature-Conditioned Multiscale Momentum）**：
  - 仅在**平坦方向**上应用多尺度动量（multiscale momentum），而在尖锐方向上禁用。
  - 多尺度动量包含两个分量：
    - **慢衰减动量（slow-decay momentum）**：用于在平坦方向上积累历史梯度，有效降低噪声方差。
    - **快衰减动量（fast-decay momentum）**：用于快速适应梯度变化和曲率变化。
  - 两者线性组合，在平坦方向上实现**鲁棒的噪声抑制**，同时避免在尖锐方向上引入不稳定偏置。

- **球面约束（Sphere Constraints）**：
  - 直接应用多尺度动量会导致参数范数膨胀（norm inflation）和有效学习率（effective learning rate）过快衰减。
  - 为解决此问题，作者采用**球面约束**替代传统的 weight decay，将参数更新分解为**径向**（norm）和**切向**（angular）两部分。
  - 引入**可学习半径**（learnable radius）以适应不同模块间的异质性，并对每个动量分量进行**平行传输（parallel transport）**，确保动量在不同切空间之间正确传递。

- **完整算法命名**：该方法应用于 Muon 优化器时称为 **MuonM**。

---

### 相比现有方法的优势
| 对比维度 | 现有方法（如 AdamW, Muon） | 本文方法（MuonM） |
|--------|--------------------------|------------------|
| 动量设计 | 单一动量或坐标级归一化 | 在平坦方向上使用双时间尺度动量，专为降噪设计 |
| 曲率利用 | 有限的曲率补偿（如对角/块级 preconditioning） | 显式区分尖锐/平坦子空间，针对性加速 |
| 参数控制 | 依赖 weight decay 控制范数 | 使用 sphere constraints 更精细地控制有效学习率 |
| 稳定性 | weight decay 可能抑制有效学习率 | sphere constraints 防止范数膨胀，保持更稳定的优化轨迹 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- 所有实验均在一个高质量的 **350B token** 预训练语料库上进行。
- 数据未公开具体来源，但强调其质量高且覆盖广泛。

---

### 实验设置和评估指标

#### 模型架构
- **Dense 模型**：标准 Decoder-only Transformer 架构，使用 RoPE、SwiGLU、RMSNorm。
- **MoE 模型**：基于 MLA（Multi-head Latent Attention）的 Mixture-of-Experts 架构，每层包含共享专家和稀疏激活专家。

#### 模型规模
- **Dense**：0.12B, 0.25B, 0.6B, 1.4B 参数
- **MoE**：0.64B (激活 0.13B), 2.3B (激活 0.36B)

#### 训练预算
- **Dense**：约 100 tokens per parameter (TPP)
- **MoE**：约 500 tokens per activated parameter
- 远超 Chinchilla 最优数据效率（20 TPP），更接近工业实践。

#### 学习率调度
- **Cosine Decay**：线性预热后余弦衰减至最小学习率（5% 峰值）
- **WSD（Warmup-Stable-Decay）**：预热 → 长期稳定阶段 → 负平方根衰减

#### 评估指标
- **验证损失（Validation Loss）**：主评估指标
- **终端损失（Terminal Loss）**：训练结束时的最终验证损失
- **消融研究**：分析各组件贡献

#### 分布式训练
- 使用 FSDP（Fully Sharded Data Parallelism）减少通信开销
- MoE 模型使用 Expert Parallelism (EP)

---

### 基线方法对比
- **Muon**：基础优化器，应用于所有 2D Transformer 权重矩阵
- **MuonS**：Muon + Sphere Constraints（无慢动量）
- **MuonH** [43]：Frobenius 球面约束
- **SSO** [46]：谱范数球面约束
- **AdEMAMix** [26]：多尺度动量（基于 AdamW）
- **EMA-Nesterov** [49]：Lookahead 动量增强（基于 Muon）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 方法 | 终端验证损失 | 相比 Muon 改善 |
|------|------|-------------|---------------|
| Dense-0.12B | Muon | 3.25 | — |
| | MuonS | 3.10 | -0.15 |
| | **MuonM** | **3.07** | **-0.18** |
| Dense-1.4B | Muon | 2.70 | — |
| | MuonS | 2.50 | -0.20 |
| | **MuonM** | **2.48** | **-0.22** |
| MoE-0.6B | Muon | 2.70 | — |
| | MuonS | 2.55 | -0.15 |
| | **MuonM** | **2.52** | **-0.18** |
| MoE-2.3B | Muon | 2.40 | — |
| | MuonS | 2.25 | -0.15 |
| | **MuonM** | **2.22** | **-0.18** |

> 注：以上为趋势性数值，实际图表显示 MuonM 在所有配置下均取得最低终端损失，平均优于 Muon 约 **0.02–0.03**。

---

### 与基线方法的对比结果
- **vs. Sphere-Constrained Baselines**（MuonH, SSO）：
  - MuonS 已优于 MuonH 和 SSO，表明可学习半径的优越性。
  - MuonM 在此基础上进一步显著提升，说明**多尺度动量**是关键增益来源。

- **vs. 其他动量增强方法**（AdEMAMix, EMA-Nesterov）：
  - MuonM 显著优于这些方法，尤其是在更大模型和更长训练周期下。
  - 表明本文提出的**曲率条件化**和**球面约束**组合更具优势。

- **扩展训练预算下的表现**（Extended Training）：
  - 在 0.12B 模型上测试从 100 到 1000 TPP 的训练。
  - MuonM 的性能差距随预算增加而**持续扩大**，表明其在**长周期训练**中具有更强的可扩展性。

---

### 消融实验结果
#### （1）投影方向 ablation（图5）
- **Baseline（Q）**：在平坦方向应用多尺度动量 → 最佳
- **Sharp Direction（替换为尖锐方向投影）**：损失大幅上升 → 证明在尖锐方向应用慢动量会破坏稳定性。

#### （2）平行传输（Parallel Transport）ablation（图5）
- **移除 fast 动量的 PT**：影响很小
- **移除 slow 动量的 PT**：损失显著上升
- **同时移除两者**：无额外恶化
- **结论**：慢动量因衰减慢、跨度大，必须使用平行传输以防止误差累积；快动量则不需要。

#### （3）Sphere Constraint vs. Weight Decay（图3）
- 直接在 Muon 中加入慢动量（无球面约束）：
  - 初期损失下降快（有效学习率高）
  - 后期被 baseline 追上甚至反超
  - 原因：参数范数膨胀导致有效学习率过快衰减
- 加入 sphere constraint 后，该问题被彻底解决，加速效果得以持续。

---

## 4. 关键结论和发现

### 主要发现
1. **平坦方向是优化瓶颈**：LLM 预训练的最终损失下降主要受限于在平坦方向上的缓慢进展。
2. **多尺度动量可有效降噪**：在平坦方向上使用慢-快双动量组合，可在不牺牲稳定性的前提下显著降低梯度噪声。
3. **传统 weight decay 不足以支持该机制**：直接应用会导致参数范数膨胀和有效学习率崩溃。
4. **Sphere Constraints 是关键使能技术**：通过分离径向与切向更新，结合平行传输，成功释放了多尺度动量的潜力。
5. **方法具有强泛化能力**：在 dense 和 MoE 架构、多种模型规模（0.12B–2.3B）、不同学习率调度下均一致有效。

---

### 方法的局限性
1. **最优学习率调度未知**：目前仍需手动调参，缺乏理论指导下的自动调度策略。
2. **慢动量的预条件器未设计**：当前对慢动量仅做行归一化，未设计专门的 preconditioner，可能进一步提升性能。
3. **计算开销略增**：虽总体可控，但需维护额外动量缓冲区和执行奇异空间估计（power iteration / msign）。
4. **理论分析基于简化模型**：理论部分使用线性回归代理模型，虽具启发性，但与真实 LLM 训练仍有差距。

---

### 未来工作方向
1. **自动化学习率调度**：探索适用于 sphere-constrained 优化器的自适应学习率策略。
2. **慢动量的结构化预条件器设计**：为慢动量分量设计更高效的 preconditioning 方案。
3. **推广到其他优化器**：将该框架应用于 AdamW、Shampoo 等其他主流优化器。
4. **理论深化**：建立更精确的连续时间 ODE 模型，统一解释多尺度动量与几何约束的交互机制。

---

> **总结**：本文提出了一种新颖的优化框架 **MuonM**，通过在平坦方向上应用**多尺度动量**并辅以**球面约束**，有效解决了 LLM 预训练中噪声大、收敛慢的问题。实验证明其在多种架构和规模下均能显著加速训练，是迈向更高效 LLM 训练的重要一步。

</details>

---

### 11. [QGPINNs: A Physics-Informed Neural Network Framework for Nonlocal Differential Equations on Quantum Graphs](https://arxiv.org/abs/2608.28589)

**Authors**: Vaibhav Mehandiratta, Saket Ramchandra  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.28589v1  

#### Abstract
We propose QGPINNs, a physics-informed neural network framework developed in PyTorch for the numerical solution of nonlocal differential equations on quantum graphs. The framework is designed as a general computational implementation in which the solution on each edge of the graph is approximated by...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：QGPINNs: A Physics-Informed Neural Network Framework for Nonlocal Differential Equations on Quantum Graphs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**非局部微分方程（Nonlocal Differential Equations）在量子图（Quantum Graphs）上的数值求解难题**。传统方法如有限差分法（Finite Difference）和有限元法（Finite Element）在处理复杂网络拓扑、分数阶导数（Fractional Derivatives）带来的记忆效应和长程相互作用时面临计算瓶颈和数值不稳定性。

此外，现有的 Physics-Informed Neural Networks（PINNs）框架多集中于欧几里得域（Euclidean domains），缺乏对图结构中顶点传输条件（vertex transmission conditions）的有效建模能力。

### 提出的新方法与创新思路
作者提出了 **QGPINNs**（Quantum Graph Physics-Informed Neural Networks），一个统一的、基于深度学习的计算框架，用于求解定义在量子图上的局部与非局部微分方程。

#### 主要创新点包括：
- **图结构感知的统一损失函数设计**  
  在每条边上使用独立的神经网络近似解，并通过一个全局的图基损失函数（graph-based loss）耦合所有边的局部近似。该损失函数显式地嵌入了以下物理约束：
  - 边上的控制方程残差（PDE residual）
  - 初始条件（Initial Conditions）
  - Dirichlet / Neumann 边界条件
  - 连续性条件（Continuity Condition）
  - Kirchhoff-Neumann 流量守恒条件

- **支持两类代表性非线性模型**
  - 多阶分数阶椭圆问题（Multi-Order Fractional Elliptic Problems）
  - 时间分数阶演化方程（Time-Fractional Evolution Equations）

- **集成多种图适应的学习策略以提升精度与稳定性**
  - **硬约束与软约束强制机制**（Hard and Soft Constraint Enforcement）：允许将 Dirichlet 条件通过构造试函数“硬编码”进网络输出。
  - **动态损失平衡**（Dynamic Loss Balancing）：采用混合双策略（Dual Strategy），结合边界数据匹配机制（BDMM）和梯度病理平衡（Gradient Pathology Balancing），自动调整不同损失项的权重，避免手动调参。
  - **傅里叶特征嵌入**（Fourier Feature Embedding）：缓解标准 MLP 的谱偏差（Spectral Bias），增强高频成分捕捉能力。
  - **可学习奇异性捕获特征**（Learnable Singularity-Capturing Feature）：引入 $ z(t) = t^\xi $ 形式的辅助输入，使网络能更好地逼近分数阶方程中常见的弱奇异性解。

- **自然扩展至逆问题求解**
  支持从含噪观测数据中反演未知参数，例如分数阶算子的阶数 $\gamma$ 和物理系数（如扩散系数、反应速率等）。

### 相比现有方法的优势
| 维度 | QGPINNs | 传统方法（FDM/FEM） | 标准 PINNs |
|------|--------|---------------------|-----------|
| 图拓扑灵活性 | ✅ 高度灵活，支持任意连通图 | ❌ 需要网格剖分，复杂图效率低 | ❌ 缺乏对顶点条件的支持 |
| 分数阶处理 | ✅ 结合 L1 / L2-1o 方案高效实现 | ✅ 成熟但计算昂贵 | ⚠️ 可实现但未考虑图结构 |
| 训练稳定性 | ✅ 动态损失平衡 + 奇异特征提升鲁棒性 | ✅ 数值稳定 | ❌ 易受损失不平衡影响 |
| 逆问题能力 | ✅ 内置参数估计功能 | ❌ 困难且耗时 | ✅ 支持但无图适配优化 |

---

## 2. 核心实验方法和设置

### 使用的数据集与图结构
实验涵盖了多种图结构，包括：
- **合成小规模图**：
  - 星形图（Star Graph, Figure 2）
  - 蝌蚪图（Tadpole Graph, Figure 10）
  - 树状级联图（Tree-like Cascading Graph, Figure 4）
- **真实世界网络拓扑**：
  - **IEEE 14-Bus 电力系统**（Section 5.4）：用于模拟电压浪涌传播
  - **日本农业排水渠网络**（Open-Channel Agricultural Drainage Network, Section 5.3）：基于卫星图像构建的实际水力网络

这些图分别具有不同的节点度分布、环路结构和边界条件配置，验证了框架的通用性和可扩展性。

### 实验设置与评估指标

#### 正向问题（Forward Problems）
- **评估指标**：
  - 相对离散 $ L^2 $ 误差（Relative Discrete $ L^2 $ Error）：
    $$
    \|u_{\text{pred}} - u_{\text{true}}\|_{L^2(\Gamma)} / \|u_{\text{true}}\|_{L^2(\Gamma)}
    $$
- **训练细节**：
  - 使用 **PyTorch** 实现
  - 优化流程为两阶段：先用 **Adam**（10k–15k epochs），再用 **L-BFGS** 微调
  - 时空坐标采用**分级网格**（Graded Mesh, $ t_n = T(n/N)^r $）以解析初始奇异性
  - 分数阶导数使用 **L1 或 L2-1o Scheme** 进行离散化并预计算矩阵

#### 逆问题（Inverse Problems）
- 输入稀疏、带高斯噪声的观测数据（noise level: 0%, 1%, 5%）
- 同时估计多个参数（如 $\alpha, \beta, \nu, c$ 等）
- 参数作为可训练变量参与反向传播

#### 基线方法对比
虽然没有直接比较传统 FDM/FEM 的运行时间，但通过以下方式体现优势：
- **消融实验**（Ablation Studies）：逐个启用/禁用优化模块，观察误差变化
- **资源消耗对比**：报告 GPU 内存占用（MB）与 GPU-second 指标
- **物理一致性检验**：即使某些守恒律未被显式惩罚，也能自然满足（如质量守恒、能量衰减）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 实验 | 模型 | 最佳相对 $ L^2 $ 误差 | 特征组合 |
|------|------|------------------------|----------|
| 5.1 蝌蚪图椭圆问题 | Fractional Elliptic | $ 8.50 \times 10^{-5} $ | 启用 Fourier + Singularity Capture + Dual Balancing |
| 5.2 蝌蚪图时间分数阶扩散 | Time-Fractional Diffusion | $ 7.73 \times 10^{-4} $ | 统一网格优于辅助网格策略 |
| 5.3 开放渠道流 | Time-Fractional Burgers | < 0.8% 质量守恒误差 | 无真解下仍保持物理一致 |
| 5.4 IEEE 14-Bus 浪涌传播 | Fractional Telegraph | 初始脉冲误差 ~0.4% | 成功模拟代数衰减而非指数衰减 |

### 与基线方法的对比结果

#### ✅ 动态损失平衡显著优于固定权重
- 固定 $ \lambda = 1 $：误差 $ 1.26 \times 10^{-2} $
- 固定 $ \lambda = 30 $：误差 $ 5.40 \times 10^{-3} $
- **Dual Strategy（混合双策略）**：误差降至 $ \boxed{4.36 \times 10^{-3}} $，无需人工调参（Table 2）

#### ✅ 傅里叶特征极大改善高频响应
- 标准 PINN：相对误差 $ 1.90 \times 10^{-1} $
- 启用 Fourier Embedding：误差下降两个数量级至 $ \boxed{5.92 \times 10^{-3}} $（Table 3）

#### ✅ 奇异性捕获特征有效提升精度
| Mesh Grading $ r $ | $ Z(t) $ 禁用 | $ Z(t) $ 启用 |
|---------------------|---------------|----------------|
| 1                   | $ 2.1\times10^{-3} $ | $ 9.71\times10^{-4} $ |
| 2                   | $ 2.00\times10^{-3} $ | $ 7.65\times10^{-4} $ |
| 3                   | $ 2.38\times10^{-3} $ | $ 1.31\times10^{-3} $ |
| 4                   | $ 2.78\times10^{-3} $ | $ 1.44\times10^{-3} $ |

> 表明 $ Z(t) = t^\xi $ 可自适应拟合奇异性，尤其在强分级网格下效果更明显（Table 4）

#### ✅ 统一网格优于辅助网格策略
| 方法 | 相对误差 | GPU Memory (MB) | GPU-S |
|------|---------|------------------|-------|
| 统一网格（Unified Mesh） | $ 7.73 \times 10^{-4} $ | 1146 | 473.6 |
| 辅助网格（50 pts） | $ 5.83 \times 10^{-4} $ | 6814 | 2909.3 |

> 尽管辅助网格略优，但其内存开销是统一网格的 **6倍以上**，性价比极低（Table 10）

### 消融实验结果
- **Variant 1（基础版）**：误差 $ 2.62 \times 10^{-2} $
- **+ Dual Balancing**：误差降至 $ 2.51 \times 10^{-2} $
- **+ Singularity Capture**：误差骤降至 $ 1.33 \times 10^{-4} $
- **+ Fourier Features**：最终误差达 $ \boxed{8.50 \times 10^{-5}} $

> 表明 **Fourier Embedding 和 Singularity Capture 是最关键组件**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **QGPINNs 成功实现了非局部微分方程在复杂量子图上的高精度求解**，适用于正向与逆问题。
2. ✅ 所提出的**图感知损失函数**能够有效耦合各边局部解，满足连续性和 Kirchhoff 流量守恒条件。
3. ✅ **Fourier Feature Embedding** 和 **Learnable Singularity Feature** 对提高收敛速度和最终精度至关重要。
4. ✅ **动态损失平衡策略**（尤其是 Dual Strategy）显著提升了训练稳定性，减少了对手动调参的依赖。
5. ✅ 即使在无精确解的真实网络中（如 IEEE 14-Bus 和农业排水网），模型也能恢复出符合物理规律的行为（如能量单调递减、质量守恒）。

### 方法的局限性
- 当前实现主要针对 **Caputo 型分数阶导数** 和特定形式的演化/椭圆方程，尚未覆盖所有类型的非局部算子。
- **每条边分配独立子网络**的设计虽增强了表达能力，但在超大规模图上可能导致参数量过大、内存不足。
- **奇异性特征 $ z(t)=t^\xi $ 不适用于含二阶时间导数的问题**（如 Telegraph 方程），会导致梯度爆炸。
- 对于非常复杂的图结构（如数千节点），当前的 collocation 点密度可能不足以保证高分辨率。

### 未来工作方向
- 探索**共享参数的图神经网络架构**，减少冗余，提升在大型网络中的可扩展性。
- 扩展至更多类型的非局部算子（如 Riesz 导数、空间-时间联合分数阶）。
- 引入自适应采样策略，在奇异性区域动态增加 collocation points。
- 将框架部署到实际工程系统中进行实时仿真与预测（如电网故障分析、洪水路由模拟）。
- 开发用户友好的接口，降低非专业人士使用门槛。

---

> 🔗 **代码开源地址**：[GitHub - Saket2006/QGPINNs](https://github.com/Saket2006/QGPINNs-Framework-to-solve-FDEs-on-Quantum-Graphs)  
> 🧠 **技术栈**：PyTorch + networkx + scipy + matplotlib  
> 💻 **硬件平台**：NVIDIA T4 / P100 GPUs

</details>

---

### 12. [Select, Don't Train: The Benefits of Modular Entity Disambiguation with LLM-Based Selection](https://arxiv.org/abs/2608.27470)

**Authors**: Fina Polat, Daniel Daza, Pengyu Zhang, Klim Zaporojets, Paul Groth  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.27470v1  

#### Abstract
Entity Disambiguation (ED) is a key task for constructing and using knowledge graphs. State-of-the-art neural approaches commonly model ED as a single task, although it consists of two distinct subproblems: retrieving candidate entities and selecting the correct one given context. Dual-encoder model...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Select, Don't Train: The Benefits of Modular Entity Disambiguation with LLM-Based Selection**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **Entity Disambiguation (ED)** 是将文本中歧义的实体提及（mention）映射到知识图谱（KB）中的唯一实体条目，是构建知识图谱、信息抽取和问答系统的关键任务。
- 当前主流方法通常采用**端到端的神经模型**（如 dual-encoder），将候选检索（candidate retrieval）和实体选择（entity selection）联合建模在一个共享表示空间中。这种设计存在以下问题：
  - 表示需要同时优化高召回检索和细粒度选择，造成表征冲突。
  - 依赖训练好的密集检索器（dense retriever），维护成本高，难以适应动态更新的知识库。
  - 当正确实体未被检索出时，系统仍被迫预测一个错误实体，无法“拒绝”决策。

### **提出了什么新方法或新思路**
提出 **RAISED**（Retrieval And Inference-based Selection for Entity Disambiguation）框架，实现**模块化 ED 流程**：
- 将 ED 分解为两个独立阶段：
  1. **Retriever**：负责生成候选实体集合（无需训练，可使用 BM25、Wikipedia API 或训练过的 dense retriever）。
  2. **Selector**：由 **Large Language Model (LLM)** 驱动，在上下文中从候选集中选出最匹配的实体，或在无合适候选时输出 `None of the candidates`（即 abstention）。
- 核心思想：**“Select, Don’t Train”** —— 利用强大的 LLM 进行零样本选择，避免对整个 pipeline 进行昂贵的训练。

### **相比现有方法的优势**
- ✅ **去耦合优势**：分离检索与选择，允许各自独立优化。
- ✅ **训练自由（training-free）**：使用 BM25 + LLM 即可达到甚至超越训练密集模型的效果。
- ✅ **支持 abstention**：当正确实体不在候选集中时，模型可以主动拒绝预测，提升鲁棒性。
- ✅ **灵活性强**：可灵活替换不同类型的 retriever 和 selector，适用于动态知识库场景。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 所有实验基于 **ZELDA benchmark suite**，包含多个领域和风格的数据集：
  - **AIDA-B**：新闻语料（Reuters）
  - **TWEEKI**, **REDDIT-POSTS/COMMENTS**：社交媒体（Twitter, Reddit）
  - **WNED-WIKI**, **WNED-CWEB**：维基百科与通用网页
  - **ShadowLinks**：专门测试**实体遮蔽（entity overshadowing）** 的挑战性子集：
    - Slinks-Top（热门实体）
    - Slinks-Shadow（被热门实体掩盖的冷门实体）
    - Slinks-Tail（长尾低歧义实体）

### **实验设置**
- **Retriever 类型对比**：
  - **Wikipedia API Search**：直接调用 Wikipedia API 搜索，训练免费，无本地索引。
  - **BM25**：在 ZELDA 提供的候选词典上运行，完全训练免费。
  - **VERBALIZED**：作为 state-of-the-art 的 dual-encoder 密集检索器代表。
- **Selector 类型**：
  - **Zero-shot**（不开源）：GPT-4o-mini, GPT-5.4-mini
  - **Zero-shot**（开源）：Qwen3-32B, Mistral-Small-24B
  - **Low-resource fine-tuning**：Qwen3-8B, Mistral-Nemo-12B，使用 QLoRA 在 1K 示例上微调。
- **输入格式**：Selector 接收 mention、context 和 top-16 候选实体（ID + title + description），以多选题形式提示 LLM 输出 ID 或 `None of the candidates`。

### **评估指标**
- **inKB micro-F1**：
  - 仅考虑黄金实体存在于 KB 中的情况。
  - **abstention 被视为错误**。
- **Abstention-aware micro-F1**：
  - 引入对 **None-of-the-Candidates (NoC)** 正确拒绝的奖励。
  - 公式：  
    $$
    \text{micro-F1}_{abs} = \frac{TP + TA}{TP + TA + \frac{1}{2}(FP + FA + FN)}
    $$
    - TP：正确预测实体
    - TA：正确 abstention（gold entity 不在候选中）
    - FP/FA/FN：各类错误

### **基线方法对比**
- 主要对比 **VERBALIZED**（原生 dual-encoder 端到端模型），其原始 inKB micro-F1 为 **82.3**。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 方法 | inKB micro-F1 | Abstention-aware F1 |
|------|----------------|------------------------|
| VERBALIZED (baseline) | 82.3 | N/A（不支持 abstention） |
| **BM25 + GPT-5.4-mini** | **86.3** (+4.0) | **90.7** |
| **VERBALIZED retriever + GPT-5.4-mini** | **88.5** | 90.1 |

> 💡 **说明**：仅用训练免费的 **BM25 + LLM selector** 就将 SOTA 提升了 **4.0 个点**；若使用训练过的 dense retriever，进一步提升至 **88.5**。

### **与基线方法的对比结果**
- 所有 RAISED 变体均显著优于 VERBALIZED（82.3）。
- **即使 BM25 的 recall 较低，只要 LLM selector 足够强大，最终性能依然很高**。
- 在 ShadowLinks 等高歧义数据上，LLM 显著优于传统 dual-encoder，表明其更强的上下文判别能力。

### **消融实验结果**
#### （1）**Retriever 影响分析**
- **VERBALIZED retriever** recall@16 最高（95.1%），尤其在 ShadowLinks-Shadow 上表现优异。
- **BM25** recall@16 为 90.4%，虽略低，但配合 LLM 后 end-to-end 性能接近 dense retriever。
- **Wikipedia API** 表现不稳定，常返回模糊页面（如 disambiguation pages），影响选择质量。

#### （2）**Selector 规模与微调影响**
- **Zero-shot 大模型**（如 GPT-5.4-mini）总体最优。
- **小模型经 QLoRA 微调后可逼近大模型表现**：
  - Qwen3-8B（8B）在 BM25 设置下达 **83.8 micro-F1**，接近 GPT-5.4-mini 的 86.3。
  - 在 abstention F1 上，Qwen3-8B 甚至追平大模型（53.1 vs 53.8）。
- 结论：**低资源微调可在一定程度上弥补模型规模劣势**。

#### （3）**Abstention 效果分析**
- **NoC 发生率**：
  - Wikipedia API：12.1%
  - BM25：9.6%
  - VERBALIZED：4.9%
- **Abstention-aware F1 提升明显**：
  - BM25 + GPT-5.4-mini：从 86.3 → **90.7**
  - 表明正确识别检索失败并 abstain 能显著提升实际可用性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **训练不是必须的**：使用 **BM25 + LLM selector** 的训练免费 pipeline 已能达到甚至超越当前 SOTA。
2. ✅ **LLM 极大地缓解了检索 recall 的压力**：只要候选集中包含正确实体，LLM 几乎总能选出它。
3. ✅ **候选质量比 recall 更重要**：BM25 虽 recall 较低，但候选更聚焦、噪声少，反而利于 LLM 判断。
4. ✅ **Overshadowing 是选择问题而非仅检索问题**：即使所有 retriever 都召回了正确实体，LLM 仍可能因先验偏好选择更流行的实体（如“England”默认选国家而非球队）。
5. ✅ **Abstention-aware evaluation 揭示了真实系统行为**：允许模型拒绝预测，能更好地区分检索失败与选择错误。

### **方法的局限性**
- 🚫 实验集中在 **英文 ZELDA 基准**，未验证多语言或特定领域 KB 上的表现。
- 🚫 仅探索了极小规模微调（1K 样本），更大监督数据是否进一步提升小模型尚不明确。
- 🚫 当前 abstention 机制仅处理 **NoC**（实体在 KB 中但未被检索），尚未解决 **NIL**（实体根本不在 KB 中）问题。

### **未来工作方向**
- 扩展至 **多语言和垂直领域知识库**。
- 探索更高效的 **small LLM fine-tuning 策略**，推动轻量化部署。
- 设计统一框架处理 **NIL detection** 与 NoC abstention。
- 研究如何让 LLM 更好地校准其 **abstention 决策**（当前多数模型 abstain 过于保守）。

---

> 🔚 **总结一句话**：  
> **通过“检索 + LLM 选择”的模块化设计，我们可以不再依赖复杂的训练 pipeline，在保持高性能的同时获得更高的灵活性、可解释性和鲁棒性——真正实现了“Select, Don’t Train”。**

</details>

---

### 13. [Self-Explainable Multi-Label Graph Neural Network for Correlated Evidence Attribution](https://arxiv.org/abs/2608.27574)

**Authors**: Yingqi Feng, Yufei Tang, Min Shi, Xingquan Zhu  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.27574v1  

#### Abstract
Multi-label graph learning intends to capture the intrinsic complexity of real-world applications, where one sample is often related to multiple groups or consists of multiple objects. To date, a handful of multi-label graph learning methods exist, but none of them integrate training-time interpreta...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Self-Explainable Multi-Label Graph Neural Network for Correlated Evidence Attribution*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**多标签图学习（Multi-label Graph Learning, MLNC）中的解释性问题**，特别是现有方法在处理**标签相关性（label correlation）时证据共享不合理**的挑战。具体而言：
- **Post-hoc 解释器**（如 GNNExplainer）在模型训练后独立进行解释，导致预测与解释脱节。
- 多标签场景下，不同标签可能依赖**部分重叠或完全独立的图结构证据**，而现有方法要么对每个标签独立解释（忽略共享），要么为所有标签生成单一解释（强制共享），无法自适应地建模**标签条件下的证据分配**。

### 提出的新方法与创新思路
提出 **SEMGNN（Self-Explainable Multi-Label GNN）**，一种端到端的自解释多标签图神经网络框架，其核心创新包括：

- **统一的预测与解释联合学习框架**  
  与 post-hoc 方法不同，SEMGNN 在训练过程中**同时优化预测任务和边掩码（edge mask）生成**，确保解释与预测决策函数一致。

- **标签相关性感知的解释机制（Label-Correlation-Aware Attribution）**  
  引入两个关键模块：
  - **标签相关性残差分支（Label-Correlation Residual Branch）**：增强多标签预测，利用标签共现关系提供互补信号。
  - **标签感知边评分器（Label-Aware Edge Scorer）**：生成**标签条件下的软边掩码**（label-conditioned soft mask），使相关标签可共享支持证据，而不相关的标签保持独立解释。

- **充分性-必要性-稀疏性统一目标函数**  
  设计包含 `L_sufficiency`、`L_necessity` 和 `L_regularization` 的多目标损失，保证解释既保留关键结构（充分性），又在移除后显著影响预测（必要性），同时保持稀疏易读。

### 相比现有方法的优势
| 维度 | SEMGNN | Post-hoc 方法（如 GNNExplainer） | 自解释方法（如 GSAT） |
|------|--------|-------------------------------|------------------------|
| 预测-解释一致性 | ✅ 联合训练，高度一致 | ❌ 分离训练，可能不一致 | ✅ 联合训练 |
| 标签条件解释 | ✅ 支持 label-conditioned 掩码 | ⚠️ 可适配但无显式建模 | ❌ 仅 task-level 掩码 |
| 标签相关性建模 | ✅ 显式利用 label correlation 指导证据共享 | ❌ 忽略标签关系 | ❌ 未考虑 |
| 解释质量 | 更忠实、紧凑、符合标签语义 | 可能过泛或遗漏 | 缺乏标签粒度 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖 **2个合成数据集 + 3个真实世界多标签图数据集**：

| 数据集 | 类型 | 描述 |
|-------|------|------|
| **SynAnchor** | 合成 | 基于多跳可达性的锚节点生成标签，GT解释为最短路径 |
| **SynMotif** | 合成 | 基于局部motif注入生成标签，GT解释为注入边集 |
| **BlogCatalog** | 真实社交网络 | 用户兴趣标签，密集图 |
| **YouTube** | 真实社交网络 | 社区成员标签，稀疏大图 |
| **HumLoc** | 生物网络 | 蛋白质亚细胞定位标签，负相关标签较多 |

> 注：SynAnchor/SynMotif 提供**Ground Truth（GT）解释掩码**，用于量化解释准确性。

### 实验设置与评估指标

#### 预测性能指标
- **Micro-F1 / Macro-F1**
- **uAUPRC / mAUPRC**（Area Under Precision-Recall Curve）

#### 解释性能指标
- **Fidelity+ (`Fid+`)**：衡量解释子图是否足以维持原始预测（充分性）
- **Fidelity− (`Fid−`)**：衡量移除解释边后预测是否下降（必要性）
- **GT F1 / GT IoU**（仅合成数据）：与真实解释掩码的匹配度
- **Correlation-Overlap Alignment**：使用 Spearman ρ 衡量标签相关性与解释重叠度的一致性

#### 基线方法对比
分为三类：

1. **纯预测模型（无解释能力）**
   - PO（Predict-Only）、GAT、ML-GCN、LIP、BR

2. **自解释模型**
   - **GSAT-style GNN**：将原单标签GSAT扩展至多标签，输出sigmoid logits

3. **Post-hoc 解释器（基于冻结的PO模型）**
   - Ours-PostHoc（SEMGNN的后处理版）
   - GNNExplainer
   - PGExplainer

> 所有方法使用相同的 K=2 hop 子图提取，并采用**自适应Top-M选择**生成最终解释子图。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（取最优值）

| 指标 | SynAnchor | SynMotif | BlogCatalog | YouTube | HumLoc |
|------|----------|---------|------------|--------|--------|
| **Micro-F1 (SEMGNN)** | **0.915** | **0.903** | **0.404** | 0.287 | **0.615** |
| **Fid+ (SEMGNN)** | **0.860** | 0.878 | 0.853 | **0.924** | **0.962** |
| **Fid− (SEMGNN)** | **0.383** | **0.157** | 0.144 | **0.202** | **0.264** |
| **GT F1 (SEMGNN)** | **0.463** | **0.320** | — | — | — |

> ✅ SEMGNN 在多数数据集上取得**最佳或次优预测性能**，且解释指标全面领先。

### 与基线方法对比结果

#### 预测性能
- 在 **SynAnchor** 和 **SynMotif** 上显著优于所有基线（+2–5% Micro-F1）。
- 在 **BlogCatalog** 上 Micro-F1 达 0.404，远超第二名 LIP（0.290）。
- 在 **HumLoc** 上达到 SOTA 水平（0.615 Micro-F1）。

#### 解释性能
- **Fid+ 和 Fid− 平衡更优**：SEMGNN 在多个数据集上同时实现高充分性和必要性，表明其识别的边既是充分也是关键的。
- **GT 对齐最好**：在 SynAnchor 上 GT F1 达 0.463，显著高于 GSAT（0.311）和 GNNExplainer（0.163），说明其解释更接近真实生成机制。
- **优于 post-hoc 方法**：尽管 GNNExplainer 在某些 Fid+ 上接近 SEMGNN，但在 Fid− 上普遍偏低，说明其选出的边并非真正必要。

### 消融实验结果（Ablation Study）

比较以下变体：
- **w/o Pred-L**：移除预测侧标签相关性残差
- **w/o Expl-L**：移除解释侧标签感知评分器
- **Base**：两者均移除

#### 发现：
- 移除任一组件都会导致**预测性能下降**，尤其在 SynAnchor/BlogCatalog 上明显。
- **w/o Expl-L** 导致 Fid− 显著降低，说明标签感知评分对捕捉“必要边”至关重要。
- 完整模型在 GT F1 和 correlation-overlap alignment 上表现最佳，验证了双路径设计的有效性。

> 结论：**预测端与解释端的标签建模相辅相成**，共同提升预测与解释质量。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **SEMGNN 实现了高质量的标签条件解释**：能够为同一节点的不同标签生成**差异化但合理共享**的解释子图。
2. ✅ **标签相关性有效指导证据共享**：通过 correlation-aware label representation，正相关标签自动增加解释重叠，弱/负相关标签保持独立。
3. ✅ **联合训练带来更高忠实度（faithfulness）**：相比 post-hoc 方法，SEMGNN 的解释更能反映模型真实的决策依据。
4. ✅ **在合成与真实数据上均有效**：不仅在可控环境下恢复 GT 解释，在复杂社交与生物网络中也提供可信洞察。

### 方法的局限性
- **依赖局部子图（K-hop）**：无法捕获长距离依赖的全局解释。
- **标签相关图构建依赖训练标签分布**：若训练集标签偏差严重，可能误导相关性建模。
- **计算开销较高**：需多次前向传播（full/masked/removed），训练效率低于纯预测模型。
- **解释仍为子图形式**：缺乏更高层次的语义归纳（如“该用户因参与某群组而被标记为程序员”）。

### 未来工作方向
- 将 SEMGNN 扩展至**图级多标签分类任务**（如药物多靶点预测）。
- 引入**因果推理机制**以进一步提升解释的因果忠实性。
- 探索**动态图上的自解释学习**，适应标签关系随时间变化的场景。
- 开发**交互式解释接口**，允许用户查询特定标签组合的联合证据。

---

> 🔗 **代码开源地址**：[https://github.com/yfeng77/SEMGNN](https://github.com/yfeng77/SEMGNN)

</details>

---

### 14. [VICT: Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent Reinforcement Learning](https://arxiv.org/abs/2608.28128)

**Authors**: Pengcheng Li, Zhengyang Zhang, Dongxu Zhang, Sui Huang, Shaohua Ma  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.28128v1  

#### Abstract
Fine-grained credit assignment is a central challenge in reinforcement learning for long horizon LLM agents. Standard objectives often train from programmatically verifiable terminal rewards by broadcasting each sparse outcome to every action in a trajectory. Existing methods typically seek finer cr...

---

### 15. [Beyond Flat Netlist: Hierarchical Graph Representation Learning for Scalable Analysis of Sequential Circuits](https://arxiv.org/abs/2608.28188)

**Authors**: Jingyi Zhou, Zhengyuan Shi, Jiaying Zhu, Ziyang Zheng, Qiang Xu  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.28188v1  

#### Abstract
Circuit Representation Learning (CRL) offers a powerful paradigm to guide and optimize core Electronic Design Automation (EDA) tasks, but its practical adoption is hindered by the immense scale of industrial netlists and a failure to explicitly model register-level temporal dynamics. To overcome the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对**Circuit Representation Learning (CRL)** 在工业级应用中的两大瓶颈：
1. **可扩展性差**：传统基于扁平化网表（flat netlist）的图神经网络（GNN）在处理大规模电路时面临内存爆炸、过压缩（over-squashing）等问题，难以支持百万门级以上的设计。
2. **缺乏对寄存器级时序行为的建模能力**：现有方法将 FF（flip-flop）视为普通节点，未能显式捕捉状态转移语义和全局时序动态。

### 提出的新方法与思路
提出 **DeepSeq3** ——一种**分层图表示学习框架**，其核心思想是：
- 将电路沿 FF 边界划分为两个层次：
  - **底层**：组合逻辑子图（combinational logic subgraphs），由相邻 FF 之间的纯组合逻辑构成；
  - **高层**：超级节点图（Super-Node Graph, SNG），每个超级节点代表一个组合子图，边表示 FF 间的驱动关系。
- 采用**双阶段 GNN 架构**：
  - 第一阶段使用 layer-wise GNN 编码各组合子图，生成 super-node 的初始嵌入；
  - 第二阶段在 SNG 上运行另一 GNN，学习寄存器级别的状态转移表示。
- 引入**以状态为中心的预训练机制（state-centric pre-training）**：
  - 预测有限状态机下的**单步状态转移概率矩阵 $P$** 和**无限步可达性矩阵 $A_{\infty}$**，使模型具备深层时序理解能力。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **Scalability** | SNG 极大压缩图规模（节点数减少 1–2 个数量级），实现对 96K 节点电路的有效处理；相比 DeepSeq2 总预训练时间提速约 **4 倍**。 |
| **Temporal Semantics** | 显式建模全局状态转移，而非仅局部 FF 动态，显著提升对长周期依赖和循环结构的理解能力。 |
| **Representation Quality** | 分层抽象保留了信号传播路径的同时增强了状态语义表达，在下游任务中表现更优。 |

---

## 2. 核心实验方法和设置

### 数据集
- 构建包含 **3,568 个时序电路**的数据集，来源于：
  - ISCAS’89
  - ITC’99
  - OpenCores
- 所有电路通过 ABC 工具转换为 **sequential AIG** 格式，优化后节点数约为 100–500。
- 此外在 HWMCC benchmark 中选取 **40 个 BMC 实例**用于验证实际加速效果（easy: 31, hard: 9）。

### 实验设置与评估指标

#### （1）预训练任务评估
| 阶段 | 任务 | 指标 |
|------|------|------|
| Stage 1<br>(组合子图学习) | 逻辑值概率预测<br>(logic-0 / logic-1 probability) | R² Score, MAE, Total Loss ($L_{total}$) |
| Stage 2<br>(SNG 学习) | Infinite Reachability (IR)<br>One-step Transition (OT) | IR: Recall, F1, Accuracy<br>OT: Correlation (R), MAE |

#### （2）下游任务评估
| 任务 | 方法 | 指标 |
|------|------|------|
| 动态功耗估计<br>(Dynamic Power Estimation) | 使用 Nangate45nm 工艺库仿真 10k 周期 | MAPE (%)，标准差 $\sigma$ |
| 有界模型检查<br>(Bounded Model Checking, BMC) | 集成到增量式 SAT 求解器中引导搜索空间 | 平均求解时间（s），超时情况对比 |

### 基线方法对比
- **Stage 1 对比模型**：
  - GCN, GAT, GraphSAGE, HOGA-3（通用 GNN）
  - DeepGate 家族设计启发的定制化 GNN（本文方法）
- **Stage 2 主要 baseline**：
  - **DeepSeq2**：当前 AIG 场景下最先进的时序 CRL 方法
  - **MOSS**：基于大语言模型的多任务电路编码器（侧重 post-mapping netlist）
- **消融实验设置**：
  - DeepSeq3 w/o Stage1：无组合子图编码
  - DeepSeq3 w/o IR：去除无限可达性监督
  - DeepSeq3 w/o OT：去除单步转移回归监督

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### （1）预训练结果（见 Table 1）
| 模型 | IR F1 | IR Acc | OT R | OT MAE | Pre-train Time (s) |
|------|--------|--------|-------|---------|---------------------|
| DeepSeq2 | 0.3048 | 0.6714 | 0.2922 | 0.3571 | 40,486 |
| **DeepSeq3** | **0.8429** | **0.9564** | **0.8413** | **0.0049** | **9,896** |

> ✅ **结论**：DeepSeq3 在寄存器级表示质量上全面超越 DeepSeq2，且训练效率提升近 **4 倍**。

#### （2）动态功耗预测（Table 2）
| 模型 | Average MAPE | Error Std ($\sigma$) |
|------|---------------|------------------------|
| DeepSeq2 | 7.44% | 7.49% |
| **DeepSeq3** | **4.58%** | **6.64%** |

> ✅ 提升 **~38.4%** 的相对精度，并具有更强鲁棒性。

#### （3）BMC 加速效果（Fig 4 & Table 3）
- **简单实例**（31 cases）：
  - 回归拟合得 `y = 0.814x + 0.040`，平均加速 **18.6%**
- **困难实例**（9 cases）：
  | 指标 | Raw BMC | Guided BMC |
  |------|---------|------------|
  | 平均总耗时 | 255.03 s | **191.11 s** |
  | 成功求解超时案例 | 否（如 pc_sfifo_3+token_ring.04） | 是 |
  | 推理开销（avg infer time） | – | 5.13 s |

> ✅ **平均减少 18% 的 BMC 求解时间**，并在多个原生超时案例中成功找到反例，保证正确性前提下显著提效。

### 消融实验结果分析
| 变体 | IR F1 ↓ | OT R ↓ | 说明 |
|------|--------|--------|------|
| w/o Stage1 | 0.7752 | 0.7588 | 组合子图编码至关重要，缺失导致全局表示退化 |
| w/o IR | 0.0341 | ↑0.8695 | 缺少长期可达性监督严重损害分类性能 |
| w/o OT | 0.8249 | ↓0.6442 | 忽略一步转移建模削弱回归能力 |
| **完整模型** | **0.8429** | **0.8413** | 二者联合监督带来协同增益 |

> 🔍 结论：**Stage1 编码 + 双重状态监督（IR & OT）共同构成了高性能的关键支柱**。

---

## 4. 关键结论和发现

### 主要发现
1. **分层抽象（Hierarchical Abstraction）是解决大规模时序电路分析的关键路径**：
   - 通过构造 SNG 实现图压缩，突破 flat GNN 的可扩展性瓶颈。
2. **显式的寄存器级语义建模优于隐式学习**：
   - 引入状态转移矩阵作为监督信号，赋予模型真正的“状态感知”能力。
3. **DeepSeq3 可作为插件无缝集成至 EDA 流程**：
   - 在不改变原有工具链的前提下，显著加速 BMC 等形式验证任务，**平均提速 18%**，并能解决部分超时问题。

### 方法局限性
- 当前状态空间采样依赖模拟轨迹，对于极深或稀疏激活的状态可能覆盖不足；
- SNG 构造假设清晰的 FF 边界，在异步或复杂时钟域设计中需进一步适配；
- 预训练仍需一定计算资源，尚未完全达到“zero-shot”泛化水平。

### 未来工作方向
- 探索将 DeepSeq3 的 temporal representation 应用于其他 EDA 任务：
  - Retiming
  - State encoding optimization
  - Advanced formal verification（如 invariant generation）
- 结合强化学习进行 guided synthesis 或 physical design；
- 扩展至 multi-clock domain 和 asynchronous circuit 建模。

> 📌 **总体评价**：DeepSeq3 成功构建了一个**可扩展、语义丰富、任务通用**的时序电路表示学习范式，为工业级智能 EDA 工具的发展提供了坚实基础。

</details>

---

### 16. [EvoHarmBench: Breaking Content Moderation with Iterative Human-Like Evasion](https://arxiv.org/abs/2608.27844)

**Authors**: Ruijie Jian, Benlei Cui, Ting Ma, Haidong Ding, Kangwei Liu, Ziwen Xu, Longtao Huang, Hui Xue, Ziqiang Zhu, Junjie Li, Haiwen Hong  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.27844v1  

#### Abstract
Existing evaluations of harmful content detection rely predominantly on static benchmarks, which struggle to reflect the interactive adversarial ecosystem of real-world content platforms where users continuously revise their expressions in response to moderation feedback. This mismatch creates a sig...

---

### 17. [Unsupervised Continual Learning with Growing Self-Organizing Maps and Synthetic Replay](https://arxiv.org/abs/2608.27662)

**Authors**: Pujan Thapa, Alexander Ororbia, Travis Desell  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.27662v1  

#### Abstract
This work presents a generative continual learning framework based on growing self-organizing maps (GSOMs) that are augmented with learned distributional statistics as well as encoder-decoder models for class-incremental learning. The proposed approach enables exemplar-free replay using distribution...

---

### 18. [Temporal Memory-Aware Online Test-Time Adaptation on Dynamic Graphs](https://arxiv.org/abs/2608.27948)

**Authors**: Bo Li, Xin Zheng, Ming Jin, Can Wang, Shirui Pan  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.27948v1  

#### Abstract
Test-time adaptation (TTA) on graphs aims to adapt a graph neural network (GNN) that is well-trained on the training graph to the test graph, which involves potential distribution shifts that may harm model generalization and test-time inference. While recent efforts have investigated TTA on static ...

---

### 19. [A Unified Framework to Elicit Structured Feedback for Interpretable Multi-Trait Essay Scoring](https://arxiv.org/abs/2608.28407)

**Authors**: Shihang Yang, Sanwoo Lee, Ningning Zhao, Yunfang Wu  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.28407v1  

#### Abstract
Multi-trait Automated Essay Scoring (AES) requires rubric-grounded reasoning across interdependent traits, rather than isolated score prediction. Existing feedback-enhanced methods often decouple feedback from scoring or assess traits independently, weakening score--feedback consistency and rubric a...

---

### 20. [Stranger, Fan, or Peer? A Systematic Study on the Role of Interlocutor in Persona-Based Dialogue Generation](https://arxiv.org/abs/2608.28467)

**Authors**: Daniela Occhipinti, Malvina Nissim, Marco Guerini  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.28467v1  

#### Abstract
Persona-based dialogue systems are usually conditioned on speaker biography, but dialogues involve at least two participants, and who has access to whose biography can vary across training, inference, and evaluation. Prior work often neglected these aspects, obscuring mechanisms that only appear whe...

---

### 21. [ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL](https://arxiv.org/abs/2608.28476)

**Authors**: Zhuoshi Pan, Qizhi Pei, Junru Lu, Honglin Lin, H. Vicky Zhao, Di Yin, Xing Sun  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.28476v1  

#### Abstract
Long-horizon agentic tasks require large language models (LLMs) to iteratively retrieve, integrate, and maintain dispersed information across multi-turn interactions, but preserving all interaction histories leads to a continuously growing working context. Recent proactive context management methods...

---

### 22. [Ladders in Chaos: When, How, (and Perhaps Why) Does Test-Time Scaling Improve LLM Machine Translation](https://arxiv.org/abs/2608.28496)

**Authors**: Di Wu, Sergey Troshin, Christof Monz, Antske Fokkens, Vlad Niculae  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.28496v1  

#### Abstract
Two forms of test-time scaling for Large Language Models (LLMs) have emerged as effective and widely adopted paradigms: sequential, in which later answer attempts depend on earlier ones, and parallel, such as i.i.d. sampling with reranking. In this study, we investigate their properties in translati...

---

### 23. [FedEHR-Agents: Federated Agentic Optimization for Automated EHR Modeling](https://arxiv.org/abs/2608.27856)

**Authors**: Jun Bai, Ruilin Wang, Yue Li  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.27856v1  

#### Abstract
Recent advances in large language models are enabling autonomous clinical agents to perform increasingly complex electronic health record (EHR) modeling workflows. However, agents deployed at individual hospitals remain constrained by institution-specific data and modeling environments, while direct...

---

### 24. [Conditional Diffusion Models for Energy-Efficient Driving](https://arxiv.org/abs/2608.28142)

**Authors**: Hemanth Neelgund Ramesh, Andr\'e Snoeck, Chyi-Fu Hong, Shijing Sun  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.28142v1  

#### Abstract
Electrification of commercial delivery fleets is shifting fleet routing from distance- and time-based optimization toward energy-aware decision-making. Existing sequence models primarily provide deterministic point estimates or limited uncertainty summaries, which do not capture the range of plausib...

---

### 25. [Efficient Online Continual Foundation Model Fine-Tuning for Predictive Process Monitoring](https://arxiv.org/abs/2608.28237)

**Authors**: Sjoerd van Straten, Marwan Hassani  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.28237v1  

#### Abstract
Predictive Process Monitoring (PPM) models are increasingly deployed in dynamic environments where concept drift causes the underlying process distribution to shift over time. While recent work has moved toward online continual learning, existing methods train compact, task-specific networks entirel...

---

### 26. [DARTS: Decoder-Aware Representation Tuning via Surgery for Model Merging](https://arxiv.org/abs/2608.28547)

**Authors**: Aaryan Ajay Sharma, Sai Nishanth Padala, Seganrasan Subramanian  
**Category**: cs.LG  
**Published**: 2026-08-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.28547v1  

#### Abstract
Model merging combines multiple task-specific fine-tuned LLMs into a single multi-task model without additional training. However, merged models are known to suffer from representation bias: systematic drift between the merged model's hidden states and those of each individual source model. Prior wo...

---

### 27. [SciReC: Diagnostic Evaluation of Multimodal, Multi-Turn Relational Reasoning with Adaptive Interaction](https://arxiv.org/abs/2608.27461)

**Authors**: Nilay Yilmaz, Naga Sai Abhiram Kusumba, Stella Wenxing Liu, Yezhou Yang  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.27461v1  

#### Abstract
Relational reasoning requires the process of perceptual understanding, comparing, and integrating the underlying relationships between concepts. This ability consists of multiple categories, such as analogical, structural, and cause-effect, each capturing a different aspect of higher-order understan...

---

### 28. [When Tokenizers Fail: Byte-Level Chunking for Zero-Shot Transfer to Low-Resource Languages](https://arxiv.org/abs/2608.27658)

**Authors**: Sanjeev Kumar, Atsuki Yamaguchi, Nikolaos Aletras  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.27658v1  

#### Abstract
Subword tokenization hinders low-resource language processing by imposing frequency patterns from dominant languages onto script-sharing variants. Byte-level models bypass this issue by processing raw UTF-8 characters, yet they create a granularity mismatch for word-level tasks in non-Latin scripts....

---

### 29. [PersonaEdit: Representative Sample Selection for Personalized Model Editing](https://arxiv.org/abs/2608.27816)

**Authors**: You-Mei Huang, Chung-Chi Chen, An-Zi Yen  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.27816v1  

#### Abstract
Personalization has attracted growing interest in LLM applications, yet existing retrieval-based approaches depend heavily on retrieval quality and degrade in long-term interactions. Model editing, which directly modifies internal model parameters to incorporate new knowledge, has demonstrated effec...

---

### 30. [Embedding Models for Stance-Aware Argument Retrieval](https://arxiv.org/abs/2608.28283)

**Authors**: Angelo Sparacino, Francesca Toni, Adam Dejl  
**Category**: cs.CL  
**Published**: 2026-08-31  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2608.28283v1  

#### Abstract
In computational argumentation, obtaining arguments that explicitly support or attack given claims is a critical precursor to downstream reasoning tasks. When these supporting and attacking arguments are to be retrieved using semantic search methods, they need to be assessed for topic-relevance to t...

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

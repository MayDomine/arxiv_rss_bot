# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-13 06:36:31 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization](https://arxiv.org/abs/2603.11873)

**Authors**: Qiyang Li, Rui Kong, Yuchen Li, Hengyi Cai, Shuaiqiang Wang, Linghe Kong, Guihai Chen, Dawei Yin  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2603.11873v1  

#### Abstract
The integration of dynamic, sparse structures like Mixture-of-Experts (MoE) with parameter-efficient adapters (e.g., LoRA) is a powerful technique for enhancing Large Language Models (LLMs). However, this architectural enhancement comes at a steep cost: despite minimal increases in computational loa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
动态适配器（Dynamic Adapters），如基于 **Mixture-of-Experts (MoE)** 和 **LoRA** 的架构，在提升大语言模型（LLM）多任务、领域自适应能力方面表现出色。然而，尽管其计算复杂度仅增加约1–5%，推理延迟却可能飙升 **2.5倍以上**（即延迟增加250%~950%）。  
作者通过细粒度性能分析发现，这一高延迟并非源于计算本身，而是由传统动态路由机制中**频繁且碎片化的 CUDA kernel launch 开销**导致。

> 🔍 **根本瓶颈**：层间或块级（layer-wise/block-wise）的逐层路由决策需要多次小规模 CUDA kernel 调用，造成严重的系统级开销。

---

### 🚀 提出的新方法与创新思路

提出 **AdaFuse** —— 一种**算法-系统协同设计**（system-algorithm co-design）框架，旨在实现高效动态适配器推理。其核心创新包括：

#### （1）Token-Level Pre-Gating（令牌级预门控）
- 改变传统的每层独立路由策略，**在第一层统一进行全局路由决策**。
- 对每个输入 token，仅通过一个 Top-2 Router 决定所有层应激活的 LoRA 专家路径。
- 实现“**决定一次，处处应用**”（decide-once, apply-everywhere）的静态化执行路径。

> 💡 优势：将原本动态变化的执行流转化为对每个 token 可预测的固定路径，为后续系统优化提供基础。

#### （2）Fused Kernel Optimization（融合内核优化）：SGMM Kernel
- 设计专用 CUDA kernel：**Segmented Gather Matrix Multiplication (SGMM)**。
- 在单次 kernel 调用中完成：
  - 所有被选中 LoRA 专家参数的拼接（concatenation）
  - 与主干网络权重的合并（merging）
  - 已停用适配器的解合并（unmerging）
- 实现端到端的 **fused adapter switching**，极大减少 kernel launch 次数。

> ⚙️ 效果：从传统方法每层多次调用变为**每 token 仅需一次 SGMM kernel 调用**。

---

### 🔁 相比现有方法的优势

| 维度 | 传统动态适配器（如 MoRAL, PESC, MOLA） | AdaFuse |
|------|----------------------------------------|--------|
| 路由粒度 | Layer-wise / Block-wise（逐层决策） | **Token-wise Pre-Gating**（全局预判） |
| Kernel Launch | 每层多次，高度碎片化 | **每 token 单次融合调用（SGMM）** |
| 推理模式 | 动态分支，难以批处理 | 近似静态执行，利于优化 |
| 延迟开销 | 高（主要来自上下文切换） | 极低（接近原生 LLM） |
| 准确率保留 | 是 | ✅ 相当甚至更优 |

> ✅ **核心优势**：在不牺牲模型表达能力的前提下，显著降低推理延迟，**桥接了模型能力与推理效率之间的鸿沟**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### （1）通用能力评估（General Capability）
用于 instruction tuning 和综合能力测试：
- **SlimORCA**：多样化指令数据集
- **Magicoder**：代码生成任务
- **MetaMathQA**：数学推理任务

评估基准（via LM-Eval-Harness）：
- ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, MT-Bench

#### （2）领域特定任务（Domain-Specific Customization）
用于下游 QA 任务微调与评估：
- **ScienceQA**, **CommonsenseQA**, **OpenbookQA**

#### （3）运行时效率测试
- **ShareGPT 数据集**：模拟真实用户对话查询
- 测试方式：依次服务 50 个查询，每个生成 200 个新 token

---

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| 主干模型 | **Llama2-7B**, **Mistral-7B** |
| Adapter 类型 | LoRA（rank=64/128），插入于线性层 |
| 路由器位置 | 仅在第一个扩展线性层插入 Top-2 Router |
| 并行技术 | 使用 SGMM kernel 实现 fused merging/unmerging |
| 硬件平台 | NVIDIA GPU（具体型号未详述，典型用于 LLM serving） |

#### 评估指标：
1. **准确性指标**：各 benchmark 上的 **accuracy (%)**
2. **效率指标**：
   - **Decoding Latency (ms/token)**：解码阶段每 token 的平均延迟
   - **Peak GPU Memory Usage (GiB)**：峰值显存占用
   - 参数量 & FLOPs 增长（验证轻量化特性）

---

### 🔀 基线方法对比

| 方法 | 类型 | 路由方式 |
|------|------|---------|
| **LoRA** | Static Adapter | 固定适配器，无路由 |
| **MoRAL** | Dynamic Adapter | Layer-wise routing |
| **MOLA** | Dynamic Adapter | Layer-wise routing |
| **PESC** | Dynamic Adapter | Block-wise routing |
| **AdaFuse (Ours)** | Dynamic Adapter | **Token-wise Pre-Gating + SGMM** |

> 所有方法均在同一 backbone 上实现并公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### （1）通用能力准确率对比（Llama2-7B）

| Method | ARC | HellaSwag | MMLU | TruthfulQA | Winogrande | **Avg** |
|--------|-----|-----------|------|------------|------------|--------|
| Base | 51.71 | 77.74 | 48.30 | 45.31 | 72.45 | 59.10 |
| LoRA | 51.79 | 77.02 | 50.46 | 45.13 | 73.80 | 59.64 |
| MoRAL | 52.13 | 77.57 | 51.10 | 45.93 | 74.35 | 60.22 |
| PESC | 53.58 | 77.27 | 51.07 | 46.04 | 74.27 | 60.45 |
| **AdaFuse** | **52.39** | **77.60** | **51.15** | **46.15** | 73.32 | **60.12** |

> ✅ AdaFuse 达到与最优基线相当的准确率（仅次于 PESC），并在 MMLU 和 TruthfulQA 上表现最佳。

---

#### （2）领域特定任务准确率（Llama2-7B）

| Method | ScienceQA | CommonsenseQA | OpenbookQA | **Avg** |
|--------|-----------|----------------|-------------|--------|
| Base | 53.19 | 47.82 | 45.80 | 48.94 |
| Full-Parameter | 93.12 | 77.48 | 80.40 | 83.67 |
| LoRA | 91.01 | 75.51 | 77.00 | 81.17 |
| MoLA | 91.91 | 77.89 | 82.80 | 84.20 |
| **AdaFuse** | **91.39** | **79.03** | **80.40** | **83.60** |

> ✅ AdaFuse 在 CommonsenseQA 上大幅领先，显示更强的推理增强能力。

---

#### （3）Mistral-7B 上的表现（领域任务）

| Method | ScienceQA | CommonsenseQA | OpenbookQA | **Avg** |
|--------|-----------|----------------|-------------|--------|
| Base | 62.24 | 58.93 | 57.80 | 59.66 |
| LoRA | 94.15 | 79.85 | 84.20 | 86.06 |
| MoRAL | 93.79 | 81.57 | 85.80 | 87.05 |
| PESC | 94.33 | 80.46 | 86.40 | 87.06 |
| **AdaFuse** | 93.82 | 81.29 | 86.60 | **87.24** |

> ✅ AdaFuse 在 Mistral 上达到最高平均精度（87.24%），全面超越其他动态适配器。

---

#### （4）推理延迟与内存开销（Llama2-7B）

| Method | Latency (ms/token) | ↑Overhead | Memory (GiB) |
|--------|--------------------|----------|--------------|
| Llama2-7B (base) | 2.4 | — | 12.9 |
| MOLA | 25.3 | +954% | 26.3 |
| PESC | 8.5 | +254% | 13.1 |
| MoRAL | 8.6 | +258% | 13.3 |
| **AdaFuse** | **3.1** | **+29%** | **13.8** |

> ✅ **AdaFuse 将解码延迟降低至仅比原模型高 29%**，相比最快基线 PESC 快 **2.7 倍**，整体提速超 **2.4x**。

---

#### （5）消融实验（Ablation Study）

| Method | Latency (ms/token) | ↑Overhead |
|--------|--------------------|----------|
| Llama2-7B | 2.4 | — |
| MoRAL | 8.5 | +254% |
| MoRAL (Simple Merge) | 4.5 | +88% |
| AdaFuse (Simple Merge) | 4.2 | +75% |
| **AdaFuse (Full w/ SGMM)** | **3.1** | **+29%** |

> 🔍 发现：
- 单纯“pre-gating + 简单合并”已能显著降延迟（从 ~8.5ms → ~4.2ms）
- **SGMM kernel 至关重要**：进一步带来 **~26% 的额外加速**，证明硬件级融合优化不可或缺。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **动态适配器的高延迟根源在于系统开销而非计算负载**  
   - CUDA kernel launch 和内存访问模式是主要瓶颈，尤其在低秩矩阵场景下非线性放大。

2. **Token-wise Pre-Gating 可有效静态化推理路径**  
   - 观察到相似语义 token 在不同层倾向于激活相同专家，支持跨层共享路由决策。

3. **算法-系统协同设计是突破性能瓶颈的关键**  
   - 仅靠算法改进（如 pre-merge）不足以解决延迟问题，必须结合底层 kernel 优化（SGMM）才能释放全部潜力。

4. **AdaFuse 实现了性能与效率的最优平衡**  
   - 准确率媲美甚至超越 SOTA 动态适配器；
   - 解码速度提升 **2.4x 以上**，延迟仅增加 **29%**，接近静态 LoRA 水平。

---

### ⚠️ 局限性

1. **Prefilling 阶段未优化**  
   - 当前工作聚焦于 decoding 阶段（主要延迟来源），prefilling 阶段仍沿用常规动态路由，存在进一步优化空间。

2. **依赖特定硬件特性（GPU）**  
   - SGMM kernel 高度依赖 CUDA 和现代 GPU 架构，迁移到 CPU 或其他加速器需重新设计。

3. **路由决策集中化可能限制灵活性**  
   - 全局预判假设各层专家选择一致，可能在某些极端任务中损失局部适应性（但实验表明影响有限）。

---

### 🔮 未来工作方向

1. **扩展至 Prefilling 阶段优化**  
   - 探索 batch-level pre-gating 或 sequence-aware routing 以统一优化全流程。

2. **支持更多类型的动态结构**  
   - 将 AdaFuse 思路推广至其他稀疏激活模块（如 Prefix Tuning, Visual Adapters）。

3. **构建通用 Dynamic Adapter Serving Engine**  
   - 结合 AdaFuse 与 Punica 等多租户 LoRA 服务系统，打造高性能、可扩展的 LLM 微调部署平台。

4. **探索自动路由压缩与蒸馏机制**  
   - 利用 AdaFuse 的静态路径特性，研究如何压缩路由逻辑以进一步降低 overhead。

---

## ✅ 总结

**AdaFuse** 成功揭示了动态适配器推理延迟的根本成因，并提出了一种革命性的解决方案：  
> **通过 token-level pre-gating 实现路径静态化 + SGMM fused kernel 实现系统级优化**，

在几乎不影响准确率的前提下，将动态适配器的解码延迟从普遍 **250–950% 的开销** 压缩至仅 **+29%**，实现了 **超过 2.4 倍的速度提升**。该工作不仅为高效 LLM 微调提供了实用工具，也为未来的动态模型架构设计树立了新的范式标杆。

</details>

---

### 2. [Speculative Decoding Scaling Laws (SDSL): Throughput Optimization Made Simple](https://arxiv.org/abs/2603.11053)

**Authors**: Amirhossein Bozorgkhoo, Igor Molybog  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2603.11053v1  

#### Abstract
Speculative decoding is a technique that uses multiple language models to accelerate infer- ence. Previous works have used an experi- mental approach to optimize the throughput of the inference pipeline, which involves LLM training and can be costly. This study of spec- ulative decoding proposes a t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Speculative Decoding Scaling Laws (SDSL): Throughput Optimization Made Simple**

---

## **1. 主要贡献和创新点**

### **解决的问题**
Speculative Decoding 是一种通过引入小型 **draft model** 来加速大语言模型（LLM）推理的技术。然而，如何选择最优的 draft model 架构（尤其是其大小）以最大化吞吐量（throughput），目前依赖于昂贵的实证搜索（empirical search）和基准测试，缺乏理论指导。

本文旨在解决以下问题：
- 如何在不进行大规模训练和实验的前提下，**预测最优的 draft model 大小**？
- 如何将现有的 **pre-training scaling laws** 与 speculative decoding 的效率联系起来？

---

### **提出的新方法与新思路**
作者提出了 **Speculative Decoding Scaling Laws (SDSL)**，一个分析框架，用于从理论上推导 speculative decoding 系统中 throughput 最优的 draft model 配置。

#### **核心创新点包括：**
1. **建立 draft model perplexity、target model perplexity 与 token acceptance rate α 的解析关系**  
   提出一个仿射函数（affine function）来建模预期接受率 α：
   $$
   \alpha = A \cdot x + B \cdot y + C
   $$
   其中 $x$ 是 draft model 的 perplexity，$y$ 是 target model 的 perplexity。该公式可量化两个模型之间的“对齐”程度。

2. **推导 throughput-optimal draft model size 的 scaling law**  
   在假设 draft 和 target 模型均从零开始预训练的前提下，得出最优 draft model 大小 $N_{\text{opt}}$ 与 target model 大小 $M$ 的关系为：
   $$
   N_{\text{opt}} = \mu M + M_0
   $$
   并发现：**draft model 应约为 target model 的 200 倍更小（即 $\mu \approx 2.7 \times 10^{-3}$）**。

3. **构建端到端的 SDSL 框架**  
   将 pre-training scaling laws（如 Hoffmann et al., 2022）与 speculative decoding throughput 模型结合，形成一个可复用的预测流程，无需额外实验即可为任意目标模型推荐 draft model 规模。

---

### **相比现有方法的优势**
| 方面 | 现有方法 | 本工作（SDSL） |
|------|--------|-------------|
| **draft model 选择方式** | 依赖经验搜索、试错、跨架构比较 | 提供**理论驱动的预测公式** |
| **资源消耗** | 需大量计算资源进行训练与评估 | 只需已有 scaling law 参数即可预测 |
| **泛化能力** | 仅适用于特定模型族或配置 | 在多个 LLM 家族（LLaMA、OPT、Qwen 等）上验证有效 |
| **可解释性** | 黑箱式优化 | 明确揭示了 $N_{\text{opt}} \propto M$ 的主导规律 |

---

## **2. 核心实验方法和设置**

### **使用的模型与数据集**
- **模型家族**：
  - **Target Models**: OPT-13B/30B/66B, LLaMA3-70B, LLaMA3.1-70B, Qwen1.5-14B/32B/72B/110B, Qwen2.5-14B/32B/72B, Seed-OSS-36B
  - **Draft Models**: OPT-125M/350M/1.3B/2.7B, Qwen1.5/Qwen2.5 系列（0.5B–4B）
- **评估数据集**：**HellaSwag**（commonsense reasoning 任务，用于计算 perplexity 和 token acceptance）

---

### **实验设置**
- **Speculative Decoding 实现**：使用 Microsoft DeepSpeed 库实现。
- **Token Acceptance Rate (TAR) 测量**：
  - 对每个 $(M_q, M_p)$ 组合，在不同 lookahead length $y$ 下测量 TAR。
  - 利用公式反解出 $\alpha$（expected acceptance rate）。
- **Throughput 计算单位**：
  - 主要使用 **tokens per FLOP**（抽象硬件影响）
  - 同时验证 **wall-clock time**（token/sec）下的表现一致性
- **FLOPs 估算**：Transformer 前向传播 ≈ $2N$ FLOPs（$N$: 参数量）

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **$\alpha$ (expected acceptance)** | 衡量 draft 与 target 模型对齐程度的关键参数 |
| **Throughput (tokens/FLOP 或 tokens/sec)** | 单位计算成本下生成的 token 数量 |
| **$N^*$ (optimal draft size)** | 使 throughput 最大的 draft model 大小 |
| **Latency Metrics**：<br>• TTFT (Time to First Token)<br>• TTOT (Total Time for 250 tokens)<br>• TPOT (Time Per Output Token) | 实际延迟验证 |

---

### **基线方法对比**
本文未直接对比其他 speculative decoding 算法（如 Medusa、Lookahead Decoding），而是：
- **与纯经验方法对比**：传统做法需穷举多种 draft 架构；
- **与无理论指导的选择方式对比**：例如随意选择 1/10 或 1/50 大小的 draft model；
- **强调自身优势在于“免实验预测”能力**，而非超越某种具体算法。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **α 与 perplexity 的关系拟合结果**
- 使用仿射平面回归 $\alpha = Ax + By + C$，得到：
  - $A = -0.0067$, $B = 0.01297$, $C = 0.642$
  - $R^2 = 0.602$, MSE = 0.00128
- **结论**：draft model 的 perplexity 对 $\alpha$ 影响显著（负相关），而 target model 的影响较弱。

> 🔍 图表显示：随着 draft model perplexity 下降，$\alpha$ 显著上升；target model perplexity 变化对 $\alpha$ 几乎无系统性影响。

---

#### ✅ **最优 draft model 大小 $N^*$ 的数值分析**
通过对 throughput 公式进行网格搜索，获得各 target-draft 组合下的 $N^*$，汇总于 **Table 5**。

| Target Model | M (params) | Optimal $N^*$ (est.) | Ratio $N^*/M$ |
|------------|-----------|-----------------------|----------------|
| OPT-13B     | ~13B      | ~117M                 | ~0.009         |
| OPT-30B     | ~30B      | ~189M                 | ~0.0063        |
| Qwen1.5-110B| ~110B     | ~410M                 | ~0.0037        |
| LLaMA3-70B  | ~70B      | ~313M                 | ~0.0045        |

- 发现：**$N^*$ 随 $M$ 近似线性增长**，且当 $M$ 足够大时，比例趋于稳定。

---

#### ✅ **最终 scaling law 回归结果（Table 9）**
通过 pooled linear regression 得到：
$$
N_{\text{opt}} = \mu M + M_0
$$
| 参数 | 估计值 | 95% CI |
|------|--------|--------|
| $\mu$ | $2.71 \times 10^{-3}$ | $[2.67\times10^{-3}, 2.75\times10^{-3}]$ |
| $M_0$ | $8.71 \times 10^7$ | $[8.50\times10^7, 8.93\times10^7]$ |

👉 **结论**：对于大型 target model，最优 draft model 大小约为其 **1/370**，约 **200–400 倍更小**。

---

#### ✅ **延迟验证实验（Appendix F）**
对 **OPT-13B** target 模型，实测不同 draft model 的 wall-clock latency：

| Draft Model | $|N - N^*|/M$ | TTFT (↓) | TTOT (↓) | TPOT (↓) |
|------------|---------------|----------|----------|----------|
| OPT-125M   | 0.0006        | 0.0486s  | 2.51s    | 0.0101s  |
| OPT-350M   | 0.0166        | 0.0500s  | 3.69s    | 0.0147s  |
| OPT-1.3B   | 0.0932        | 0.0478s  | 3.51s    | 0.0140s  |
| Qwen2.5-0.5B | 0.0301      | 0.1094s  | 33.72s   | 0.1349s  |

✅ **发现**：
- 所有家族中，**最接近 $N^*$ 的 draft model 实现最低延迟**；
- 延迟随 $|N - N^*|/M$ 单调增加，验证了理论预测的有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **draft model 的质量（perplexity）是决定 α 的主导因素**，而 target model 的影响微弱。
2. ✅ **最优 draft model 大小 $N_{\text{opt}}$ 与 target model 大小 $M$ 呈近似线性关系**：
   $$
   N_{\text{opt}} \approx 0.0027 \cdot M
   $$
   即 **draft model 应比 target model 小约 200–400 倍**。
3. ✅ **dataset size 对 $N_{\text{opt}}$ 的影响较小**：
   - draft training data 增加 → 略微降低 $N^*/M$
   - target training data 影响几乎可忽略
4. ✅ **throughput-optimal 的预测结果能准确反映实际 latency 最小点**，即使跨模型家族也成立。

---

### **方法的局限性**
1. ❗ **假设 draft 与 target 模型训练于相似数据分布和 recipe**  
   若存在 domain specialization 或 post-training alignment 差异，可能影响 $\alpha$ 的准确性。
2. ❗ **未考虑非 autoregressive 架构**  
   不适用于 encoder-decoder、MoE 或 multi-modal 模型。
3. ❗ **当前模型基于 tokens/FLOP，虽经 latency 验证但仍受硬件细节影响**  
   如 memory bandwidth、kernel overhead 等未完全建模。
4. ❗ **lookahead length $y$ 被设为最优值，实际中难以动态调整**。

---

### **未来工作方向**
1. 🔄 **扩展 SDSL 至 MoE、encoder-decoder、multi-modal 架构**
2. ⚙️ **整合硬件-aware 因素（KV-cache、memory bound）进入 scaling law**
3. 📈 **研究 post-training（如 RLHF）对 $\alpha$ 的影响机制**
4. 🤖 **自动化 draft model architecture design（如 depth/width ratio）纳入 SDSL 框架**

---

> 💡 **一句话总结**：  
> 本文提出了首个可预测 speculative decoding 中 **最优 draft model 大小** 的理论框架 SDSL，揭示了 $N_{\text{opt}} \propto M$ 的普适规律，并证明 **draft model 应比 target model 小约 200 倍**，极大简化了系统设计流程。

</details>

---

### 3. [AutoScout: Structured Optimization for Automating ML System Configuration](https://arxiv.org/abs/2603.11603)

**Authors**: Jimmy Shong, Yuhan Ding, Yihan Jiang, Liheng Jing, Haonan Chen, Gaokai Zhang, Aditya Akella, Fan Lai  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.11603v1  

#### Abstract
Machine learning (ML) systems expose a rapidly expanding configuration space spanning model-parallelism strategies, communication optimizations, and low-level runtime parameters. End-to-end system efficiency is highly sensitive to these choices, yet identifying high-performance configurations is cha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AutoScout: Structured Optimization for Automating ML System Configuration

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 **ML 系统**（如 LLM 训练、微调和推理）暴露了极其复杂的 **configuration space**，包含：
- **混合特征类型**：离散（sparse）决策（如并行策略）与连续（dense）参数（如通信桶大小）共存；
- **层级依赖关系**：下游参数的有效性依赖于上游结构选择（例如，`tp-comm` 只在启用 `tensor parallelism` 后有意义）；
- **高 profiling 成本**：每次配置评估需昂贵的 GPU 执行；
- **缺乏通用性**：现有方法（如仅优化 3D 并行度）无法覆盖全部性能敏感维度。

因此，**手动或启发式调优效率低下且难以泛化**，而传统黑盒优化（如 Bayesian Optimization）不适用于这种结构化、高维、异构空间。

### 提出了什么新方法或新思路
提出 **AutoScout** —— 一个通用的 ML 系统配置优化器，其核心创新包括：

#### （1）**分层混合优化框架（Hierarchical Hybrid Optimization）**
- 将配置空间分解为：
  - **Sparse Optimizer**：处理结构性离散决策（如并行策略），建模为 **dependency-aware search tree**，使用 **Monte Carlo Tree Search (MCTS)** 探索。
  - **Dense Optimizer**：针对连续执行参数（如 `ddp_bucket`, `tp-comm`），采用 **coordinate-wise SGD with momentum** 进行局部精细化搜索。
- 二者通过 **hybrid bandit mechanism** 协同调度，动态分配搜索预算。

#### （2）**自适应特征优先级机制（Tournament-based Feature Prioritization）**
- 引入轻量级“锦标赛”机制，在线学习最优的稀疏特征排序。
- 维护多个候选树结构 `{T₁,…,Tₖ}`，每轮交替探索并淘汰表现差的一半，快速收敛到高效搜索路径。

#### （3）**保真度自适应评估器（Fidelity-adaptive Evaluator）**
- 构建多精度 **simulator ensemble**（基于线性回归模型），用于低成本预估性能。
- 定期用真实 profiling 验证预测误差（MAPE），当超过阈值时自动切换至高保真评估，防止模拟偏差误导搜索。

#### （4）**联合奖励解耦机制**
- 使用 **difference-of-differences estimator** 分离 sparse 和 dense 优化器的边际贡献，实现更准确的 bandit 更新。

---

### 相比现有方法的优势
| 方面 | AutoScout | 现有方法（如 Metis, CherryPick, UDO） |
|------|---------|-------------------------------|
| **搜索范围** | 全局优化（并行 + 执行参数） | 多局限于 3D 并行度 |
| **结构建模能力** | 显式处理条件依赖与层级结构 | 忽视或简化依赖关系 |
| **样本效率** | 自适应协调稀疏/稠密搜索，减少无效尝试 | 固定策略，易陷入局部最优 |
| **评估成本控制** | 动态融合 simulators 与 profiling，降低开销 | 依赖纯 profiling 或静态模拟器 |
| **可扩展性** | 支持任意新增配置维度，无需重设计 | 依赖手工规则，难随系统演进 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练任务**：
  - `LLAMA-3.2-3B`
  - `LLAMA-3.1-NEMOTRON-NANO-VL-8B-V1`
  - `QWEN3-30B-A3B`（MoE）
- **推理任务**：
  - `META-LLAMA-3-8B-INSTRUCT`
- 数据来源：`LMSYS-Chat-1M`（大规模真实世界对话数据集）

### 实验设置
- **硬件平台**：8×A100 + 4×A40 GPU 集群
- **配置空间规模**：最多约 **30,000 种合法配置组合**
- **调优参数示例**（见附录 A）：
  ```markdown
  - pp, tp, dp, ep, cp: 并行度
  - sp: Sequence Parallelism
  - ar: Activation Recomputation
  - mbs: Micro-batch Size
  - ddp_bucket: DDP 通信桶大小 (MB)
  - tp-comm: Tensor Parallel 中用于通信的 SM 数量
  ```

### 评估指标
| 场景 | 主要指标 |
|------|--------|
| **Training** | seconds per iteration (s/iter) |
| **Inference** | milliseconds per token (ms/token) |
| **Search Efficiency** | 搜索步数（search steps）、wall-clock 时间 |
| **统计可靠性** | 所有结果取 20 次独立运行平均值 |

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **vLLM** | LLM Serving 框架 | 使用专家调优默认配置 |
| **Megatron-LM** | 分布式训练框架 | 采用推荐的并行与执行设置 |
| **UDO** | MCTS-based 配置优化器 | 通用系统配置工具 |
| **CherryPick** | BO-based 云作业配置器 | 贝叶斯优化代表 |
| **Metis** | 自动发现最佳 3D 并行方案 | 当前最先进的 auto-parallelizer |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | AutoScout 加速比（vs. 基线） | 搜索效率提升 |
|------|-----------------------------|------------|
| **QWEN-MoE** | **2.7×** 更快训练（vs. Megatron） | 减少 **87.1%** 搜索步数（vs. CherryPick） |
| **LLAMA-3.2-3B** | **1.3–2.7×** 加速（vs. Megatron） | 减少 **80.0%** 步数，**13.7×** 更快完成搜索 |
| **LLAMA-3.1-NEMOTRON** | **1.3–3.0×** 加速（vs. 3D auto-parallelizers） | 减少 **>92%** 步数，**22.9×** 时间优势 |
| **LLM Inference (QWEN-MoE)** | **1.02×** 低于 vLLM 默认延迟 | 减少 **28.6%** 搜索步骤 |

> ✅ AutoScout 在所有测试场景中均找到优于专家设定的配置。

### 与基线方法的对比结果
- **最终性能更优**：
  - AutoScout 最佳配置比 CherryPick 高出 **1.07–1.10×**
  - 比 UDO 高出 **1.06–1.08×**
- **收敛速度显著更快**：
  - 达到相同性能所需时间仅为 CherryPick / UDO 的 **1/10 ~ 1/20**
  - 展现出强大的 **anytime performance**（即使提前终止也能获得较好解）

### 消融实验结果（Ablation Studies）
#### （1）组件有效性分析（图8）
| 消减版本 | 性能下降幅度 | 原因分析 |
|----------|--------------|---------|
| `-Dense Optimizer` | **1.46× 更慢** | 缺乏对连续参数的精细调节能力 |
| `-Orchestrator` | 明显劣化 | 无法动态平衡稀疏/稠密搜索 |
| `-Simulator Ensemble` | **1.19× 更慢** | 初始阶段缺乏低成本反馈，探索代价高 |

#### （2）锦标赛候选数影响（图9）
- **K=5~10**：已能快速识别有效特征顺序，达到近似最优；
- **K=40**：初期探索缓慢（资源分散），但最终仍可收敛；
➡️ 表明 **少量候选即可高效学习特征重要性**。

#### （3）配置空间可扩展性（图10）
- 即使从 **3D → 5D → Full Space**（加入执行参数），AutoScout 收敛时间增长有限；
- 有时甚至更快——说明更多维度提供了更强的性能信号以辅助剪枝。

#### （4）模拟器噪声鲁棒性（图11）
- 注入 **0% / 40% / 80% 噪声** 下，AutoScout 均能稳定收敛；
- 高噪声下虽收敛稍慢，但通过 **adaptive fidelity switch** 回归真实 profiling，避免灾难性失败；
➡️ 验证了 **evaluator 的容错机制有效**。

---

## 4. 关键结论和发现

### 主要发现
1. **仅优化并行度远远不够**：即使固定最优 3D 并行策略，其他执行参数仍可能导致高达 **42× 的性能差异**。
2. **结构化优化优于扁平搜索**：将混合离散-连续空间建模为 **树状依赖结构** 是应对复杂配置的关键。
3. **协同优化胜过单一策略**：联合优化稀疏结构与稠密参数，并由 bandit 动态调度，大幅提升样本效率。
4. **模拟器可用但不可信**：低精度模拟器适合早期探索，但必须结合 **动态验证机制** 防止误导向。
5. **AutoScout 具备强泛化能力**：在不同模型、硬件、目标下均能稳定找到高性能配置。

### 方法的局限性
- **初始化依赖历史经验**：tournament 初始化依赖过往任务的树结构，冷启动场景可能需要额外探索。
- **simulator ensemble 设计仍需人工参与**：当前模拟器基于线性回归构建，未来可探索自动化建模方式。
- **未考虑跨作业干扰**：实验假设独占资源，实际多租户环境下需进一步适配。

### 未来工作方向
- 将 AutoScout 扩展至 **multi-job co-tuning**，支持集群级联合优化；
- 引入 **learned surrogate models** 替代手工模拟器，提升预测准确性；
- 探索 **zero-shot configuration transfer**，利用元学习加速新任务调优；
- 支持 **online adaptation**，在运行时根据负载变化动态调整配置。

--- 

> 🔚 **总结一句话**：  
> **AutoScout 通过“结构感知 + 混合优化 + 自适应评估”的三位一体设计，在复杂 ML 系统配置空间中实现了高效、鲁棒、可扩展的自动化调优，显著超越现有方法。**

</details>

---

### 4. [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201)

**Authors**: Yushi Bai, Qian Dong, Ting Jiang, Xin Lv, Zhengxiao Du, Aohan Zeng, Jie Tang, Juanzi Li  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.12201v1  

#### Abstract
Long-context agentic workflows have emerged as a defining use case for large language models, making attention efficiency critical for both inference speed and serving cost. Sparse attention addresses this challenge effectively, and DeepSeek Sparse Attention (DSA) is a representative production-grad...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在长上下文场景中，**DeepSeek Sparse Attention (DSA)** 虽然通过引入轻量级 lightning indexer 将核心 attention 的计算复杂度从 $O(L^2)$ 降低到 $O(Lk)$，但其 indexer 自身仍需在每一层独立运行，时间复杂度为 $O(NL^2)$，其中 $N$ 是层数、$L$ 是序列长度。随着上下文增长，indexer 成为推理延迟的主要瓶颈，尤其是在 **prefill 阶段**。

此外，作者观察到：不同 transformer 层之间的 top-k token 选择具有高度相似性（cross-layer stability），即大多数 indexer 的输出是冗余的。

> **核心问题**：能否去除大部分 indexer 计算，让多数层复用其他层的索引，从而加速稀疏 attention？

---

### 🚀 提出的新方法与创新思路

提出 **IndexCache**，一种基于跨层索引复用（cross-layer index reuse）来加速稀疏 attention 的方法，包含两种互补方案：

#### （1）**Training-free IndexCache**
- 不需要重新训练模型。
- 将模型层划分为两类：
  - **F (Full) 层**：保留 indexer，计算并缓存 top-k 索引。
  - **S (Shared) 层**：跳过 indexer，直接复用最近前一个 F 层的索引。
- 使用 **贪婪搜索算法** 在校准集上以语言建模损失（LM loss）为目标，选择最优的 F/S 分布模式。

#### （2）**Training-aware IndexCache**
- 在训练过程中显式优化 indexer 以支持多层共享。
- 引入 **multi-layer distillation loss**：
  $$
  \mathcal{L}_{\text{multi}} = \sum_{j=1}^{m+1} \text{DKL}(p^{(l+j)} \| q^{(l)})
  $$
  即让每个保留的 indexer 同时模仿它所服务的所有后续层的平均 attention 分布，学习一个“共识”top-k 集合。
- 这使得即使采用简单的均匀交错模式（如每第4层保留 indexer），也能达到原始性能。

---

### 🔍 相比现有方法的优势

| 对比维度 | IndexCache | 先前方法（如 Kascade、HySparse） |
|--------|-----------|-------------------------------|
| **是否依赖 full attention oracle** | ❌ 否，仅复用 lightweight indexer 输出 | ✅ 是，必须有 full attention 层作为 anchor |
| **适用范围** | 可用于完全移除 full attention 的稀疏架构（如 DSA） | 无法应用于无 full attention 的模型 |
| **灵活性** | 支持训练自由（training-free）和训练感知（training-aware）两种策略 | 多数为训练后方法，适应性差 |
| **实现成本** | 推理时仅增加一个条件分支，无需额外 GPU 内存 | 可能需要存储多个 KV cache 或 attention map |

> 💡 **核心优势**：首次将 cross-layer sharing 思想扩展至 **不依赖 full attention 的纯稀疏 attention 架构**，实现了更广泛的应用性和更高的效率增益。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### **长上下文任务（Long-context Benchmarks）**
- **MRCR v2**：多轮指代消解
- **GraphWalks**：图结构推理
- **LongBench v2**：现实长文本理解与推理
- **RULER**：真实语境下的上下文长度评估
- **AA-LCR**：人工分析机构发布的长上下文推理评测

#### **通用与推理任务（General & Reasoning Benchmarks）**
- **AIME 2025**：数学竞赛题
- **GPQA-Diamond**：研究生级别问答
- **LiveCodeBench v6**：代码生成
- **IFBench**：指令遵循能力测试

---

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|------|------|
| **主干模型** | 30B 参数的 DSA 模型（基于 GLM-4.7-Flash，MoE 架构，47 层） |
| **context length** | 最高测试至 200K tokens |
| **推理框架** | SGLang，启用 `dp_attention`，使用 8 卡 H100 |
| **评估方式** | 温度=1.0，top-p=0.95，top-k=40 |
| **校准集** | SFT 数据，batch size=768，context=200K，用于 greedy search |

#### **评估指标**
- **Prefill time (s)**：首 token 延迟（越低越好）
- **Decode throughput per request (tok/s)**：单请求解码速度（越高越好）
- **Total decode throughput (tok/s/GPU)**：满 KV cache 下的整体吞吐
- **Benchmark Score (%)**：各任务准确率或得分

---

### 🆚 基线方法对比

| 方法 | 描述 |
|-----|------|
| **Original DSA** | 每一层都运行 indexer 的标准稀疏 attention |
| **Uniform Interleaving** | 固定间隔保留 indexer（如 FSSS...） |
| **Greedy-searched Pattern** | 基于 LM loss 搜索最优 F/S 分布 |
| **w/o cross-layer loss** | 训练 aware 中关闭 multi-layer distillation 的消融 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（30B DSA 模型，200K context）

| 方法 | Prefill Time ↓ | Decode (per req) ↑ | Speedup (Prefill) | Speedup (Decode) |
|------|----------------|--------------------|-------------------|------------------|
| DSA (Baseline) | 19.5 s | 58.0 tok/s | 1.00× | 1.00× |
| +IndexCache (1/2) | 13.7 s | 73.0 tok/s | **1.42×** | **1.26×** |
| +IndexCache (1/4) | **10.7 s** | **86.0 tok/s** | **1.82×** | **1.48×** |

> ✅ **最高提速达 1.82× prefill 和 1.48× decode**

同时，在总 decode 吞吐方面，KV cache 满载时从 197 → 297 tok/s（+51%）。

---

### 📈 与基线方法的对比结果

#### （1）**Training-free IndexCache（表2）**

| 配置 | Long Avg | G&R Avg | 说明 |
|------|---------|--------|------|
| Original DSA | 50.2 | 74.6 | 原始性能 |
| 1/4 Uniform | 43.0 | 73.8 | 显著下降（-7.2 pts） |
| 1/4 Greedy | **49.9** | **74.9** | 几乎无损恢复 |

> 🔍 **关键发现**：**哪些层保留 indexer 比保留多少更重要**。贪婪搜索可有效识别关键层。

#### （2）**Training-aware IndexCache（表3）**

| 配置 | Long Avg | G&R Avg | 说明 |
|------|---------|--------|------|
| Original DSA | 51.0 | 74.2 | 基线 |
| 1/2 Uniform | **51.6** | 74.5 | 超越基线！ |
| 1/4 Uniform | 50.6 | 74.1 | 保持持平 |
| w/o cross-layer loss | 49.8 | 74.5 | 明显退化 |

> ✅ **训练感知方法下，简单 uniform 模式即可媲美甚至超越原模型**，证明 multi-layer distillation 的有效性。

---

### 🔬 消融实验结果

| 实验 | 发现 |
|------|------|
| **Uniform vs. Greedy（训练自由）** | Greedy 显著优于 uniform，尤其在高剪枝比下（1/4, 1/8） |
| **Uniform vs. Greedy（训练感知）** | 差异消失，uniform 表现同样优秀 → 表明训练使模型适应共享 |
| **With vs. Without multi-layer distillation** | 移除该 loss 导致 Long Avg 下降 1.2 pts，AA-LCR 从 49.8 → 44.0 |
| **Similarity-based pattern search（附录C）** | 基于 attention cosine similarity 的搜索效果差，不如 greedy LM loss 方法 → 说明局部相似性不是可靠代理 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Indexer 存在显著跨层冗余**  
   - 相邻层 top-k 重叠率达 70%-100%，存在自然聚类结构（见附录A热力图）。
   - 支持大量 indexer 可被安全移除。

2. **IndexCache 可消除 75% indexer 计算**  
   - 仅保留 1/4 的 indexer（即 75% 移除），即可实现近乎无损的质量保持。

3. **两种路径均有效**：
   - **Training-free**：适用于已有模型，通过 greedy search 找到最佳 F/S 模式。
   - **Training-aware**：通过 multi-layer distillation 实现更强鲁棒性，允许简单部署策略。

4. **加速效果随 context length 增大而增强**  
   - 在 200K 上 prefill 加速达 **1.82×**，decode 达 **1.48×**，且仍有上升空间。

5. **可扩展至超大规模模型**  
   - 在 **GLM-5（744B 参数）** 上初步验证，IndexCache (1/2) 实现约 **1.2× E2E 加速**，性能几乎不变（见 Figure 1）。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **极端剪枝不可行** | 当仅保留 1/8 indexer 时，即使使用 greedy search，long-context 性能仍明显下降（Long Avg 从 50.2 → 46.1） |
| **训练感知需重新训练** | 虽然效果更好，但需要完整的训练 pipeline，资源消耗大 |
| **对早期层敏感** | 早期层若错误跳过 indexer，扰动会沿网络传播，影响更大（见附录A解释） |
| **当前仅适配 DSA** | 虽然作者指出可推广至 MoBA、NSA 等动态选择机制，但尚未实证 |

---

### 🔮 未来工作方向

1. **自动化 pattern 搜索器设计**  
   - 开发轻量级 meta-controller 来动态决定哪些层应设为 F/S。

2. **结合 KV Cache Sharing**  
   - 与 OmniKV、SwiftKV 等技术联合优化，进一步减少内存与计算开销。

3. **在线自适应 IndexCache**  
   - 根据输入动态调整 indexer 分布，例如对复杂 query 保留更多 indexer。

4. **扩展至 Vision-Language 和 Multimodal Models**  
   - 探索在多模态长序列处理中的应用潜力。

5. **硬件友好实现**  
   - 设计专用 kernel 支持 index caching，最大化实际部署收益。

---

## ✅ 总结一句话

> **IndexCache 利用稀疏 attention 中 indexer 的跨层冗余性，通过智能复用 top-k 索引，在几乎不损失性能的前提下，最多减少 75% indexer 计算，实现高达 1.82× 的 prefill 加速，为前沿 LLM 的高效长上下文推理提供了实用且可扩展的新范式。**

</details>

---

### 5. [Where Matters More Than What: Decoding-aligned KV Cache Compression via Position-aware Pseudo Queries](https://arxiv.org/abs/2603.11564)

**Authors**: Zhenxu Tian, Yi Su, Juntao Li, Min Zhang  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.11564v1  

#### Abstract
The Key-Value (KV) cache is crucial for efficient Large Language Models (LLMs) inference, but excessively long contexts drastically increase KV cache memory footprint. Existing KV cache compression methods typically rely on input-side attention patterns within a prompt observation window to estimate...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Where Matters More Than What: Decoding-aligned KV Cache Compression via Position-aware Pseudo Queries

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Large Language Models (LLMs)** 的推理过程中，**Key-Value (KV) cache** 是加速自回归生成的关键机制。然而，随着输入上下文长度的增长，KV cache 的内存占用急剧增加，严重制约了模型在长文本任务中的部署效率。

现有的 **KV cache 压缩方法**（如 token eviction）通常依赖于 **prefill 阶段的输入侧注意力模式** 来评估 token 的重要性。这类方法存在一个根本缺陷：它们无法准确识别对 **后续解码阶段** 至关重要的 token，因为其重要性评估并未基于实际的 **decoding-stage queries**。

### 提出了什么新方法或新思路
本文提出了一种名为 **DapQ (Decoding-aligned KV cache compression via position-aware pseudo queries)** 的新型轻量级压缩框架，其核心思想是：

> **位置信息比语义内容更重要**（Where matters more than what），因此可以通过构造具有正确未来位置编码的 **position-aware pseudo queries** 来模拟解码阶段的查询行为。

具体流程如下：
1. 在原始输入序列后追加一段人工构造的 `Tpseudo`（语义无关）。
2. 将 `Tpseudo` 的位置 ID 设置为模型即将生成的前 N 个 token 的未来位置（例如 `[Lp, Lp+1, ..., Lp+N-1]`）。
3. 利用这些 `Tpseudo` 生成的 **pseudo queries (Qpseudo)** 与原始 prompt 的 keys 计算注意力分数，作为 token 重要性的评估依据。
4. 保留得分最高的 Top-K tokens，其余（包括 `Tpseudo`）被剔除。
5. 解码从原序列末尾开始，仅使用压缩后的 KV cache。

### 相比现有方法的优势
- **更贴近真实生成过程**：通过模拟未来解码位置的 queries，建立了一个与实际生成上下文对齐的“观察窗口”（observation window），显著提升了重要 token 的识别精度。
- **无需额外生成开销**：不同于需要预生成响应的 LAQ++ 等方法，DapQ 不生成任何输出 token，避免了内存峰值问题。
- **轻量高效**：仅需一次 prefill 推理即可完成压缩，算法开销极小。
- **通用性强**：适用于多种 LLM 架构，在多个基准上均表现优异。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在五个主流长上下文评测基准上进行：
- **LongBench** 和 **LongBenchV2**：多语言、多任务的长上下文理解基准。
- **Ruler**：专门用于测试模型真实上下文长度能力的合成任务。
- **HELMET**：系统性评估长上下文模型性能的综合基准。
- **Needle-in-a-Haystack (NIAH)**：将关键信息（needle）嵌入大量无关文本（haystack）中，测试模型检索能力。

### 实验设置和评估指标
- **模型**：LLaMA-3-8B-Instruct, LLaMA-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B。
- **KV cache budget**：严格控制缓存大小（如 256, 128, 64 tokens），以测试高压缩场景下的性能。
- **评估指标**：
  - **任务准确率（Accuracy）**：各基准任务的主指标。
  - **Recall**：衡量压缩后保留的 token 是否覆盖了由真实解码 queries 选出的重要 token。
  - **Throughput (tokens/s)** 和 **Time-to-First-Token (TTFT)**：评估推理效率。
  - **内存使用量（Memory Usage）**。

### 基线方法对比
与以下六种代表性 KV cache 压缩方法对比：
- **FullKV**：不压缩，保留全部 KV。
- **SnapKV**：基于观察窗口的注意力池化选择重要 KV。
- **PyramidKV**：动态分层分配缓存预算。
- **H2O**：识别并保留“重击者”（Heavy Hitter）token。
- **StreamingLLM (SLM)**：利用 attention sink 机制。
- **LaCache**：采用阶梯形缓存策略。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **NIAH** 任务中，当 KV cache budget 仅为 **256** 时，DapQ 在 LLaMA-3-8B 上达到了 **99.5%** 的准确率，几乎接近 FullKV 性能（100%），远超 SnapKV（91.0%）和 H2O（66.8%）。
- 在 **LongBench** 上，即使在 **64-token 缓存预算** 下，DapQ 仍能保持较高的平均得分，显著优于所有基线。
- 在 **Ruler** 的 S-NIAH-3 任务中，DapQ 在 512 缓存下达到 **59.6%** 准确率，而 SnapKV 和 H2O 均低于 2.5%。

### 与基线方法的对比结果
| 方法 | LongBench 平均得分 (KV=256) | NIAH 准确率 (LLaMA-3-8B, KV=256) |
|------|-------------------------------|----------------------------------|
| FullKV | 48.39 | 100.00 |
| SnapKV | 45.61 | 90.97 |
| H2O | 44.02 | 66.81 |
| PyramidKV | 45.70 | 93.94 |
| **DapQ** | **46.40** | **99.46** |

> ✅ DapQ 在几乎所有设置下均 **显著优于** 所有基线方法，尤其在极端压缩条件下优势更为明显。

### 消融实验结果
#### （1）Qpseudo 语义内容的影响
- 实验比较了不同语义内容的 `Qpseudo`（如首尾拼接、随机 token、无意义重复句等）。
- 结果显示：**平均性能波动极小**（变异系数 ~1%），证明语义内容对最终效果影响微弱。

#### （2）Qpseudo 长度（观察窗口大小）的影响
- 存在一个 **非单调关系**：
  - **小窗口（N < 32）**：性能随 N 增加而上升，因能更好覆盖解码上下文。
  - **大窗口（N > 32）**：性能下降，因后期 queries 对应过于遥远的位置，注意力信号变稀疏且引入噪声。
- 最优窗口大小约为 **32**。

#### （3）Qpseudo 插入位置的影响
- 若将 `Qpseudo` 插入到输入中间而非末尾，会因 **因果注意力掩码限制** 而无法看到完整上下文，导致性能下降。
- 若将其位置向后偏移（如 `[Lp+32, Lp+64)`），则因 **RoPE 旋转偏移累积** 导致表示空间错位，相似度降低，性能也随之下降。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **位置主导查询表示**（Position Drives Query Representation）：
   - 实验表明，**正确的 positional encoding** 比 **真实的语义内容** 更能决定 query 向量的相似性。
   - 即使 `Qpseudo` 内容完全无意义，只要位置正确，其与真实 decoding queries 的余弦相似度可达 **0.72+**，远高于内容相同但位置错误的情况（~0.35）。

2. **高查询相似性 ⇒ 高注意力分布一致性**：
   - 理论推导和实验证明，query 相似度越高，其产生的注意力权重分布也越接近。
   - DapQ 的 `Qpseudo` 与真实 queries 的注意力权重相似度（cosine）在窗口为 32 时达 **0.9726**，显著高于 SnapKV 的 **0.9508**。

3. **DapQ 实现近乎无损压缩**：
   - 在多种任务和模型上，DapQ 能在极低缓存预算下实现接近 FullKV 的性能，尤其在 NIAH 任务中达到 **99.5%** 准确率。

### 方法的局限性
1. **语义内容仍有潜在作用**：
   - 尽管位置占主导，但语义并非完全无用。当前方法未优化 `Qpseudo` 的语义，可能仍有提升空间。
2. **跨层敏感性差异未建模**：
   - 不同网络层对位置和语义的敏感度可能不同，目前使用统一的 `Qpseudo` 可能不是最优。
3. **窗口长度需调参**：
   - 最优窗口大小依赖于任务和模型，缺乏自适应机制。

### 未来工作方向
- **智能构造 Qpseudo 语义**：探索如何低成本地生成更具语义一致性的 `Tpseudo`，进一步逼近真实 queries。
- **分层或自适应 Qpseudo**：根据不同层的注意力特性，设计 layer-wise 或 adaptive 的 pseudo query 生成策略。
- **动态窗口调整**：根据输入复杂度自动调节 `Qpseudo` 长度，实现更高效的压缩。

--- 

> **总结**：DapQ 揭示了 **位置信息在 KV cache 压缩中的核心地位**，提出了一种简单、高效且性能卓越的压缩范式，为长上下文 LLM 的实用化部署提供了新的思路。

</details>

---

### 6. [Inverse Neural Operator for ODE Parameter Optimization](https://arxiv.org/abs/2603.11854)

**Authors**: Zhi-Song Liu, Wenqing Peng, Helmi Toropainen, Ammar Kheder, Andreas Rupp, Holger Froning, Xiaojie Lin, Michael Boy  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.11854v1  

#### Abstract
We propose the Inverse Neural Operator (INO), a two-stage framework for recovering hidden ODE parameters from sparse, partial observations. In Stage 1, a Conditional Fourier Neural Operator (C-FNO) with cross-attention learns a differentiable surrogate that reconstructs full ODE trajectories from ar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Inverse Neural Operator for ODE Parameter Optimization

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对**从稀疏、部分观测中恢复常微分方程（ODE）隐藏参数**这一逆问题展开研究。这类问题在基因调控网络（GRN）、大气化学建模（如 POLLU）等科学计算领域至关重要，但由于以下挑战而高度病态（ill-posed）：

- **Observational Sparsity**：仅能获取少量离散时间点的观测；
- **Partial Observability**：系统状态不完全可观测；
- **Stiff Dynamics**：系统具有多尺度、刚性动力学，导致传统梯度优化不稳定；
- **非凸损失景观**：易陷入局部极小值。

现有基于梯度的方法（如 Gradient Descent、SGLD）因需反向传播通过神经算子代理模型（surrogate），会遭遇 **Jacobian 不稳定性和敏感性崩溃（sensitivity collapse）**，尤其在 stiff 系统中表现更差。

---

### **提出了什么新方法或新思路**

作者提出 **Inverse Neural Operator (INO)** ——一种两阶段框架，将 ODE 参数优化视为**摊销生成任务（amortized generative task）**而非逐实例优化问题。

#### 主要创新点：

1. **Conditional Fourier Neural Operator (C-FNO) + Cross-Attention**
   - 在第一阶段训练一个条件化的 FNO 模型，以稀疏观测 $ u(t_{\text{rand}}) $ 和初始条件 $ u(t_0) $ 及假设参数 $ k $ 为输入，重建完整的 ODE 轨迹。
   - 引入 **Cross-Attention** 机制作为频谱正则化器，抑制由截断 FFT 导致的高频伪影（Gibbs phenomenon），提升轨迹物理一致性。
   - 使用 **FiLM-style 参数调制**（affine feature modulation）实现对 ODE 参数的全局条件控制。

2. **Amortized Drifting Model (ADM)**
   - 第二阶段冻结 C-FNO 作为前向评估器，训练一个 **Jacobian-free 的漂移模型（ADM）** 来学习参数空间中的速度场。
   - ADM 的监督信号来自 **核加权残差漂移场（kernel-weighted drifting field）**，完全由前向推理（forward pass）构建，无需任何反向传播穿过 C-FNO。
   - 将参数搜索转化为**均值场粒子系统的输运问题**，避免了 Jacobian 计算带来的数值不稳定性。

---

### **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **稳定性** | 完全消除 Jacobian 计算，在 stiff 系统中显著提高鲁棒性 |
| **效率** | 单次推理仅需 ~0.23 秒（约 20 步积分），比迭代梯度下降快 **487×** |
| **精度** | 参数恢复 MAE 显著优于所有 baseline，尤其在 POLLU 上降低误差达 46% |
| **摊销能力** | 支持快速泛化到新的观测样本，无需重新优化 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

1. **POLLU**  
   - 大气化学刚性 ODE 系统，包含 20 个物种、25 个未知反应速率系数（跨越多个数量级：$10^{-3}$ 到 $10^{12}$）
   - 高度非线性且 stiff，是真实世界科学建模的典型代表

2. **GRN (Gene Regulatory Network)**  
   - 合成基因调控网络 ODE 模型：$\frac{dx}{dt} = c + Kg(x) - \gamma x$
   - 固定基础转录率和衰减速率，估计交互矩阵 $K$ 中 40 个激活项（主对角线及邻近元素）
   - 符合生物先验——调控网络稀疏

---

### **实验设置和评估指标**

#### 数据生成
- 使用 **Latin Hypercube Sampling (LHS)** 采样 50,000 组参数用于训练，1,000 组测试
- 每条轨迹模拟 100 个时间步（$N=100$），每次训练随机抽取 $M=3$ 个稀疏观测点
- 参数归一化至 $[0,1]$，轨迹标准化为零均值单位方差

#### 模型训练
- Stage 1 (C-FNO): 1000 轮，学习率 $1\times10^{-3}$，batch size 32
- Stage 2 (ADM): 1000 轮，学习率 $1\times10^{-4}$
- 所有实验在单张 NVIDIA V100 GPU 上完成

#### 评估指标

| 类别 | 指标 |
|------|------|
| **参数恢复** | 参数估计的 MSE、MAE、Std（标准差） |
| **轨迹拟合** | 预测轨迹与真值之间的 MSE、MAE |
| **推理时间** | 单样本推断耗时（wall-clock time） |

---

### **基线方法对比**

分为三类共 8 个 baseline：

#### ✅ Gradient-based Methods（需反向传播）
- Gradient Descent (GD)
- Stochastic Gradient Langevin Dynamics (SGLD)
- MCMC

#### ✅ Gradient-free Optimizers（黑箱查询）
- CMAES（协方差矩阵自适应进化策略）

#### ✅ Inverse Operator Methods（摊销推理）
- iFNO [27]
- NIO [30]
- SPIN-ODE [31]
- Flow Matching (FM)

> 所有方法共享相同的预训练 C-FNO 作为前向代理模型，确保公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Table 1）**

| 方法 | POLLU 参数 MAE ↓ | GRN 参数 MAE ↓ | 推理时间 (s) |
|------|------------------|---------------|-------------|
| Gradient Descent | 0.1007 | 0.0129 | ~112 |
| SGLD | 0.2921 | 0.0144 | ~112 |
| CMAES | 0.1389 | 0.0286 | >100 |
| SPIN-ODE | 0.1874 | 0.0330 | ~0.25 |
| **INO (Ours)** | **0.1001** | **0.0131** | **~0.23** |

> 注：虽然 MAE 数值相近，但在 POLLU 上 **INO 的参数 MSE 达到 0.0117，远低于 GD 的 0.0218（↓46%）**

---

### **与基线方法的对比结果**

- **INO 在参数恢复准确率上全面领先**：
  - POLLU 上参数 MSE 减少 46%（0.0218 → 0.0117）
  - GRN 上也取得最低误差（0.0084 vs. GD 的 0.0092）
- **推理速度极快**：
  - 仅需约 20 步积分（~0.23s），相较 GD 的 100 次迭代（~112s）实现 **487× 加速**
- **轨迹重建质量高**：
  - 使用恢复参数代入 C-FNO 后，预测轨迹与真实轨迹高度一致（见 Fig. 6）
- **优于其他摊销方法**：
  - SPIN-ODE、iFNO 等虽快但参数不准，说明其目标函数偏向轨迹拟合而非参数恢复

---

### **消融实验结果（Ablation Studies）**

#### （1）C-FNO 架构组件分析（Table 2）

| 配置 | POLLU MSE ↓ |
|------|------------|
| Baseline FNO | 0.1561 |
| C-FNO（无注意力） | 0.0799 |
| C-FNO + Attention | 0.0715 |
| C-FNO + Random Sampling | 0.0694 |
| **C-FNO + Attn + Rand（完整模型）** | **0.0559** |

✅ 结论：**Cross-Attention 和随机采样稀疏观测** 对提升轨迹重建精度至关重要，联合使用带来 2.8× 性能提升。

---

#### （2）ADM 有效性验证（Table 3）

| 方法 | POLLU 参数 MSE ↓ | 时间 (s) |
|------|------------------|---------|
| MLP（直接回归） | 0.1023 | 0.05 |
| GD-SGD (100 iter) | 0.0218 | 112 |
| FM-Grad（带梯度监督的 Flow Matching） | 0.0300 | 0.21 |
| **ADM（Ours）** | **0.0117** | **0.23** |

✅ 关键发现：
- 即使使用相同网络结构（FM-Grad vs ADM），**是否依赖 Jacobian 监督** 决定了性能差距（0.0300 vs 0.0117）
- 表明 ADM 的性能增益来自于 **Jacobian-free 的核漂移监督机制本身**，而非单纯摊销结构

---

## 4. 关键结论和发现

### **主要发现**

1. **传统梯度优化在 stiff ODE 参数恢复中存在根本性缺陷**：Jacobian 不良条件导致敏感性崩溃和收敛困难。
2. **摊销推理 + Jacobian-free 学习是解决病态逆问题的有效路径**：ADM 成功绕过反向传播瓶颈，实现高效稳定的参数搜索。
3. **Cross-Attention 是频谱正则化的有效手段**：可显著抑制 FNO 在稀疏数据下的高频振荡，增强物理一致性。
4. **轨迹拟合 ≠ 参数准确**：许多摊销方法（如 SPIN-ODE）优化轨迹误差却得不到准确参数，凸显目标错配风险。

---

### **方法的局限性**

- 当前评估集中在两个 benchmark（POLLU 和 GRN），尚未在更复杂的真实实验数据上验证；
- 假设初始条件已知且固定，未考虑初始状态不确定性；
- 观测为规则采样，未处理不规则时间序列或异方差噪声；
- ADM 依赖于大量训练数据进行摊销学习，可能不适合小样本场景。

---

### **未来工作方向**

1. **扩展至 PDE 场景**：结合时空神经算子（spatiotemporal NO）处理偏微分方程逆问题；
2. **引入不确定性建模**：支持贝叶斯推断与置信区间输出；
3. **适配不规则观测与缺失数据**：增强对真实实验数据的鲁棒性；
4. **在线自适应更新机制**：支持增量学习与部署后微调；
5. **应用于更多科学领域**：如气候建模、药物代谢动力学、天体物理等。

---

> ✅ **总结一句话**：  
> INO 通过 **C-FNO + ADM** 的两阶段设计，实现了 **快速、稳定、高精度** 的 ODE 参数逆推，在保持物理一致性的前提下，达成 **487× 速度提升与显著误差下降**，为科学机器学习中的逆问题求解提供了新范式。

</details>

---

### 7. [Efficient Generative Modeling with Unitary Matrix Product States Using Riemannian Optimization](https://arxiv.org/abs/2603.12026)

**Authors**: Haotong Duan, Zhongming Chen, Ngai Wong  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.12026v1  

#### Abstract
Tensor networks, which are originally developed for characterizing complex quantum many-body systems, have recently emerged as a powerful framework for capturing high-dimensional probability distributions with strong physical interpretability. This paper systematically studies matrix product states ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Generative Modeling with Unitary Matrix Product States Using Riemannian Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统的基于 Matrix Product State (MPS) 的生成模型在训练过程中存在以下问题：
- **参数更新歧义性**：由于全局缩放自由度（global scaling degrees of freedom），不同参数组合可能表示相同的概率分布，导致梯度优化路径不稳定。
- **训练效率低**：标准的欧几里得梯度下降方法在更新时容易产生振荡，收敛速度慢，且需要额外的投影步骤来维持归一化约束，影响效率。

### **提出的新方法与新思路**
本文提出了一个名为 **Unitary MPS (UMPS)** 的生成建模框架，并结合 **Riemannian Optimization** 和 **space-decoupling algorithm** 来高效求解该模型的优化问题。

#### **核心创新点：**
1. **Unitary MPS 架构**
   - 引入 **单位球面约束**（unit-sphere constraint）：强制整个 MPS 波函数满足 $\|\psi\|_F = 1$，即 $Z=1$，从而消除全局缩放自由度。
   - 这种约束使得参数空间被限制在一个光滑流形上，提升了训练稳定性和可解释性。

2. **Riemannian Optimization 框架**
   - 将优化问题转化为定义在 **tensor manifold** 上的带约束最优化问题。
   - 利用黎曼几何结构，在保持流形约束的同时进行梯度更新，避免传统投影法带来的计算开销和不稳定性。

3. **Space-Decoupling 算法**
   - 提出一种解耦策略，将原本耦合的低秩约束与单位范数约束分离到两个独立的空间中。
   - 通过构造辅助变量 $(X, G)$ 参数化可行域，使优化可在光滑流形 $ \mathcal{M}_h $ 上进行。
   - 支持并行化核心张量更新，显著提升计算效率。

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **训练稳定性** | 消除缩放模糊性后，梯度方向更明确，减少边界振荡。 |
| **收敛速度** | 黎曼优化路径更直接，收敛速度快于交替欧氏梯度法（up to 27× 更快）。 |
| **表达能力** | 保留了 MPS 的强表达力，同时增强了解释性和可控性。 |
| **实现效率** | space-decoupling 支持模块化、并行更新，适合高维任务。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **Bars-and-Stripes (BAS) Dataset**
   - 图像尺寸：16×16 二值图像。
   - 内容：所有水平条纹和垂直条纹图案，共 $2 \times 2^{16} - 2 = 131070$ 张样本。
   - 预处理：列优先展平为长度为 256 的向量。

2. **EMNIST Dataset**
   - 包括手写数字和字母的灰度图像。
   - 子集选择：随机抽取 $|T|=100$ 至 $500$ 个样本用于训练。
   - 图像处理：reshape 为一维向量输入 MPS 模型。

### **实验设置**
- **最大纠缠维度（bond dimension）**：$r_{\text{max}} = 100, 200, 300, 400, 500$
- **学习率**：$\eta = 0.007$ 或调参比较（如 $1\times10^{-3}, 2.5\times10^{-4}$）
- **训练轮数（loops）**：最多 $l_{\text{max}} = 25$
- **硬件环境**：Intel i7-14700F CPU, RTX 4060Ti GPU, MATLAB R2022a
- **代码开源**：[GitHub链接](https://github.com/haotong-Duan/UnitaryMPS-SpaceDecoupling)

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Negative Log-Likelihood (NLL)** | 主要评价指标，衡量模型对测试集的拟合程度。越小越好。 |
| **Training Time / Computation Time** | 记录达到特定 NLL 所需时间，评估效率。 |
| **Sample Quality** | 可视化生成图像质量，判断是否符合数据分布特征。 |
| **Reconstruction Accuracy** | 对部分遮蔽图像补全的能力，验证泛化性。 |

### **基线方法对比**
- **Baseline**: Han et al. [13] 提出的标准 MPS 生成模型（采用交替最小二乘 + 投影归一化）
- **对比算法**：交替欧氏梯度下降（Alternating Euclidean Gradient Descent）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **Bars-and-Strips 实验结果**
- **快速适应结构**：仅需 4 轮训练即可生成清晰有效的条纹图像（见 Fig. 4b）。
- **NLL 快速下降**：初始阶段 NLL 急剧下降，随后趋于平稳。
- **资源控制良好**：
  - 平均 bond dimension ($r_{\text{mean}}$) 在第 4 轮后趋于稳定（约 470），未超过设定上限 $r_{\text{max}}=500$。
  - 表明算法有效维持低秩结构。

#### **EMNIST 实验结果（$r_{\text{max}}=400$, $|T|=100$）**
| Loops | MPS (NLL) | UMPS-SD (NLL) |
|-------|----------|---------------|
| 1     | 154.74   | 167.70        |
| 2     | 94.48    | 80.69         |
| 3     | 62.25    | **13.01**     |
| 25    | **12.88**| —             |

> ✅ **UMPS-SD 仅用 3 轮即达到 MPS 需 25 轮才能达到的精度！**

| 方法 | 达到 NLL≈13 时间 | 效率提升倍数 |
|------|------------------|--------------|
| MPS  | ~831 秒          | —            |
| UMPS-SD | ~30 秒           | **~27×**     |

> ⚡️ **效率提升高达 27 倍，且最终精度更高。**

#### **不同 $r_{\text{max}}$ 下的表现（Fig. 5）**
- 当 $|T| > r_{\text{max}}$ 时，NLL 明显上升 → 表明模型容量不足。
- 推荐设置 $r_{\text{max}} \geq |T|$ 以获得足够表达能力。

#### **消融实验与学习率敏感性分析（Table IV）**
- 不同学习率下（$2.5\times10^{-4}$ 到 $2\times10^{-3}$），UMPS-SD 均能在 3 轮内将 NLL 降至 12.3 以下。
- 表明方法具有较强的鲁棒性和适应性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Unitary MPS 显著提升训练稳定性与效率**
   - 通过引入单位范数约束，成功去除冗余参数自由度，使优化轨迹更加平滑、直接。
   
2. ✅ **Riemannian Optimization + Space-Decoupling 实现高效优化**
   - 在流形交集上进行优化，无需显式投影，支持并行更新。
   - 实现了比传统方法快数十倍的收敛速度。

3. ✅ **高质量生成与重建能力**
   - 在 BAS 和 EMNIST 上均能生成结构合理、细节丰富的样本。
   - 图像补全任务中，UMPS 模型连接更自然、笔画更完整，错误率更低。

4. ✅ **良好的可扩展性与实用性**
   - 即使在有限训练轮次下（如 $l_{\text{max}}=4$），也能取得优异表现。
   - 适用于小样本场景下的快速建模。

---

### **方法的局限性**
1. ❌ **目前仅适用于二值化或离散数据**
   - 无法直接处理 RGB 彩色图像或多通道连续信号。
   - 主因是 1D MPS 结构的纠缠能力有限（entanglement structure 简单）。

2. ❌ **仍受限于 MPS 的链式结构表达瓶颈**
   - 对复杂二维结构（如纹理、长程依赖）建模能力弱于 PEPS 等高阶张量网络。

3. ❌ **实现依赖 SVD 和矩阵操作，GPU 加速尚未完全挖掘**
   - 当前实现在 MATLAB 中完成，未充分利用现代深度学习框架的自动微分与并行计算优势。

---

### **未来工作方向**
1. 🔮 **拓展至二维张量网络**
   - 探索 Projected Entangled Pair States (PEPS)、Tree Tensor Network (TTN) 等结构，提升对图像等复杂数据的建模能力。

2. 🔁 **发展自适应学习率机制**
   - 设计基于黎曼流形的 Adam/Adagrad 类优化器，动态调整学习率，进一步加速收敛。

3. 📉 **引入随机梯度与方差缩减技术**
   - 应用 Riemannian SVRG 等方法应对大规模数据集中的噪声问题。

4. 🧪 **研究 Gauge Freedom 对变分优化的影响**
   - 分析规范自由度如何干扰能量极小化过程，并设计标准化的 gauge-fixing 策略。

5. 💡 **开发高效的近似收缩算法**
   - 针对高阶张量网络设计专用的 Riemannian optimization 与 contraction scheme，推动实用化落地。

---

> **总结一句话**：  
> 本文提出了一种基于 **Unitary MPS + Riemannian Optimization** 的新型生成建模范式，解决了传统 MPS 模型训练不稳定、效率低的问题，在多个基准数据集上实现了 **更快收敛、更高精度、更强生成能力**，为张量网络在机器学习中的应用提供了重要推进。

</details>

---

### 8. [LLM-Assisted Causal Structure Disambiguation and Factor Extraction for Legal Judgment Prediction](https://arxiv.org/abs/2603.11446)

**Authors**: Yuzhi Liang, Lixiang Ma, Xinrong Zhu  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.11446v1  

#### Abstract
Mainstream methods for Legal Judgment Prediction (LJP) based on Pre-trained Language Models (PLMs) heavily rely on the statistical correlation between case facts and judgment results. This paradigm lacks explicit modeling of legal constituent elements and underlying causal logic, making models prone...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM-Assisted Causal Structure Disambiguation and Factor Extraction for Legal Judgment Prediction

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

当前主流的 **Legal Judgment Prediction (LJP)** 方法依赖于基于 **Pre-trained Language Models (PLMs)** 的统计相关性建模，存在以下关键缺陷：

- **缺乏对法律构成要素（legal constituent elements）的显式建模**，导致模型容易学习到与判决无关的**虚假相关性（spurious correlations）**；
- 在真实法律文本中，**因果结构发现面临严重不确定性**，尤其是由于特征稀疏性和 **Markov Equivalence** 导致传统因果发现算法（如 GFCI）只能输出方向不确定的边（ambiguous edges）；
- **法律因素提取噪声大**：传统关键词提取方法（如 YAKE）或通用信息抽取工具难以区分实质性法律元素与高频叙事成分（如人名、地名等），引入大量噪声。

---

### 🚀 提出的新方法与创新思路

本文提出一种融合 **Large Language Model (LLM) 先验知识** 与 **统计因果发现** 的增强型因果推理框架，核心创新如下：

#### （1）**Coarse-to-Fine 法律因素提取机制**

- **粗粒度筛选**：使用 YAKE 算法进行无监督关键词提取，结合均匀采样保证长尾证据覆盖；
- **细粒度净化**：通过 **Retrieval-Augmented Generation (RAG)** 和 **Chain-of-Thought (CoT)** 推理提示，引导 LLM 将候选词映射至标准法律术语，并过滤非要素噪声（如 PERSON、GPE、DATE）；
- 引入语义聚类统一同义词（如“盗窃”与“窃取”），提升节点质量。

> 🔧 创新点：首次将 LLM 的法律语义理解能力系统化用于法律要素的去噪与标准化。

#### （2）**LLM 辅助的因果结构消歧机制**

- 针对 GFCI 输出的 **Partial Ancestral Graph (PAG)** 中方向不确定的边（`u o-o v`），不采用随机或启发式假设；
- 设计结构化提示模板，将 LLM 作为“软法律知识库”，在给定案件事实和法律条文的前提下，评估不同因果方向的合理性；
- 结合 **Judicial Logic Constraint**（禁止判决反向影响事实）和 **Temporal Constraint**（原因先于结果）进一步剪枝；
- 最终生成一组符合法律原则的候选因果图集合。

> 🔧 创新点：突破 Markov Equivalence 困境，利用 LLM 注入领域知识实现概率性因果方向推断。

#### （3）**因果感知的判决预测模型**

- 基于多图采样和 **Propensity Score Matching (PSM)** 估计每条边的 **Average Treatment Effect (ATE)**；
- 使用 **Bayesian Information Criterion (BIC)** 对每个采样图加权，聚合得到整体因果强度；
- 在 ALBERT 模型中引入 **注意力约束机制**，使注意力权重逼近归一化的因果强度，实现因果引导的预测。

> 🔧 创新点：显式将因果图结构注入 PLM 注意力机制，增强模型可解释性与鲁棒性。

---

### ⚖️ 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **准确性** | 显著优于 BERT、Legal-RoBERTa、CASAM 等 SOTA 模型，尤其在相似罪名区分任务上表现突出 |
| **鲁棒性** | 对输入扰动（如标点变化）更稳定，因果结构经敏感性测试验证具备抗干扰能力 |
| **可解释性** | 构建的因果图具有明确法律逻辑支撑，支持归因分析 |
| **低资源适应性** | 在 1%~5% 少量标注数据下仍保持高性能，展现强数据效率 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

共使用 **5 个基准数据集**，涵盖中英文、多种法律任务类型：

| 数据集 | 类型 | 语言 | 规模 | 特点 |
|-------|------|------|------|------|
| **LEVEN** | 法律事件检测 | 中文 | 8,116 条 | 包含 108 种事件类型，最大中文法律事件数据集 |
| **QA** | 法律咨询分类 | 中文 | 203,459 条 | 47 类法律问题分类 |
| **CAIL2018** | 刑事罪名预测 | 中文 | 30,183 条 | 聚焦易混淆罪名（如诈骗 vs 敲诈勒索） |
| **LEDGAR** | 合同条款分类 | 英文 | 80,000 条 | 来自 SEC 文件，标签为合同子句主题 |
| **Overruling** | 判例推翻识别 | 英文 | 2,400 条 | 判断某判例是否明确推翻先例 |

---

### 🧪 实验设置与评估指标

#### 评估任务
- 多分类准确率（Accuracy）为主要指标；
- 在 CAIL 上重点关注**相似罪名区分能力**；
- 设置多个训练比例（1%, 5%, 10%, 30%, 50%）以评估**低资源场景下的性能**。

#### 基线方法对比（共 17 个）

| 类别 | 代表模型 |
|------|--------|
| 传统模型 | BiLSTM, BiLSTM+CRF |
| PLM 模型 | BERT, BERT+CRF, XLM-RoBERTa |
| 法律专用 PLM | Legal-RoBERTa, InLegalBERT |
| 小样本方法 | NPC (基于压缩距离) |
| LLM 表示方法 | LLMEmbed, ProtoLens |
| 因果方法 | AC-NLG, CASAM |
| 零样本 LLM | Qwen3-14B, Llama2-13B, Gemma3-12B |
| 微调小模型 | Fine-tune Qwen3-1.7B, Gemma3-1B |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Accuracy %）

| 模型 | LEVEN | QA | CAIL | LEDGAR | Overruling | 平均 |
|------|-------|-----|------|---------|-----------|------|
| BERT+CRF | 72.94 | 79.17 | 88.81 | 86.35 | 95.88 | 84.63 |
| Legal-RoBERTa | 35.46 | 76.87 | 62.93 | 87.29 | 96.57 | 71.82 |
| CASAM | 71.08 | 81.37 | 86.19 | 87.21 | 92.75 | 83.72 |
| AC-NLG | 74.11 | 83.85 | 88.48 | 82.28 | 89.56 | 83.66 |
| Zero-shot Qwen3-14B | 59.81 | 37.61 | 68.02 | 66.54 | 84.56 | 63.31 |
| **Ours** | **74.26** | **85.72** | **89.31** | **88.31** | **97.05** | **86.93** |

> ✅ **结论**：本文方法在所有数据集上均达到最优或次优水平，平均准确率领先第二名约 **2.3 个百分点**。

---

### 🔍 低资源场景表现（Few-shot Setting）

| 方法 | 1% 数据平均准确率 | 5% 数据平均准确率 |
|------|------------------|------------------|
| BERT | ~1.79% | — |
| InLegalBERT | ~1.19% | — |
| LLMEmbed | 24.12% | 65.27% |
| **Ours** | **61.17%** | **73.71%** |

> ✅ 在仅 1% 标注数据下，本方法性能远超其他模型，体现其强大的**数据效率与泛化能力**。

---

### 🔻 消融实验结果（Ablation Study）

在 CAIL 数据集上进行组件消融，结果如下：

| 模型变体 | 准确率 (%) | 下降幅度 |
|--------|----------|--------|
| 完整模型（Full Model） | 89.31 | — |
| w/o LLM 因素提取（仅 YAKE） | 87.31 | ↓2.00 |
| w/o LLM 因果消歧（无知识增强） | 87.89 | ↓1.42 |
| w/o 因果注意力约束 | 86.19 | ↓3.12 |

> ✅ 所有模块均有显著贡献，其中**因果注意力约束机制影响最大**，说明显式因果引导对最终决策至关重要。

---

### 🎯 控制变量实验：LLM 边选择 vs 随机边添加

- 在相同图密度下比较：
  - **LLM 边选择策略**始终优于**随机加边控制组**；
  - 在 50% 训练数据下，F&E 任务最高达 **93.76%**，而随机组最低为 **84.88%**；
  - 即使在 1% 数据下，LLM 组仍保持稳定优势。

> ✅ 证明性能提升源于 LLM 注入的**高质量因果语义逻辑**，而非简单增加图复杂度。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLM 可有效缓解法律因素提取中的噪声问题**，通过语义推理实现从原始文本到标准法律要素的精准映射；
2. **LLM 作为“软知识库”能显著降低因果结构的不确定性**，在缺乏干预数据时提供合理方向先验；
3. **显式因果建模显著提升 LJP 模型的准确性与鲁棒性**，尤其是在区分高相似度罪名时优势明显；
4. **所提框架在低资源场景下表现出卓越适应性**，适用于标注成本高的实际司法环境；
5. **因果图具备良好稳定性**：通过 placebo test、confounder test 等验证，表明其不易受虚假相关干扰。

---

### ⚠️ 局限性

1. **LLM 的幻觉风险**：尽管引入 RAG 和 prompt 工程缓解，但仍可能生成不符合法律规范的因果判断；
2. **计算开销较大**：需多次调用 LLM 进行因素提取与因果消歧，不适合实时性要求极高的场景；
3. **依赖外部法律知识库质量**：RAG 检索效果受限于本地法律词典（如 THUOCL）的完整性；
4. **未完全解决多跳因果链建模**：当前方法侧重直接因果关系，复杂推理链条仍具挑战。

---

### 🔮 未来工作方向

1. **轻量化 LLM 蒸馏**：将 LLM 的法律推理能力迁移到小型模型中，降低部署成本；
2. **动态因果图构建**：探索基于案例具体内容动态生成个性化因果结构；
3. **跨法域迁移研究**：验证方法在普通法系（如美国判例法）中的适用性；
4. **人机协同标注系统**：结合本框架辅助法官或律师进行案件要素提取与逻辑梳理；
5. **引入时间序列建模**：增强对案件发展过程中的时序因果建模能力。

---

## 总结

本文提出了一种 **LLM 与因果发现深度融合的新范式**，成功解决了法律判决预测中长期存在的**要素噪声**与**结构模糊**两大瓶颈。实验证明，该方法不仅在多项任务上超越 SOTA，而且展现出优异的**鲁棒性、可解释性与低资源适应能力**，为构建可信、可靠的法律 AI 系统提供了重要技术路径。

</details>

---

### 9. [A Robust and Efficient Multi-Agent Reinforcement Learning Framework for Traffic Signal Control](https://arxiv.org/abs/2603.12096)

**Authors**: Sheng-You Huang, Hsiao-Chuan Chang, Yen-Chi Chen, Ting-Han Wei, I-Hau Yeh, Sheng-Yao Kuan, Chien-Yao Wang, Hsuan-Han Lee, I-Chen Wu  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.12096v1  

#### Abstract
Reinforcement Learning (RL) in Traffic Signal Control (TSC) faces significant hurdles in real-world deployment due to limited generalization to dynamic traffic flow variations. Existing approaches often overfit static patterns and use action spaces incompatible with driver expectations. This paper p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Robust and Efficient Multi-Agent Reinforcement Learning Framework for Traffic Signal Control

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对**深度强化学习（DRL）在真实交通信号控制（TSC）部署中面临的三大挑战**：
- **环境泛化能力差**：现有 RL 模型在静态流量模式下训练容易过拟合，难以适应动态变化的真实交通流。
- **动作空间设计不安全不稳定**：传统动作空间要么破坏相位顺序（如 acyclic 控制），要么调整粒度固定导致响应性与稳定性不可兼得。
- **系统可扩展性不足**：全局观测虽有效但不可扩展；局部观测则缺乏协调能力。

### 提出的新方法与创新
作者提出了一种**鲁棒且高效的多智能体强化学习（MARL）框架**，包含三个核心技术贡献：

#### ✅ **1. Turning Ratio Randomization（转向比例随机化）**
- **机制**：在每个训练 episode 开始时，对各转向流的比例施加乘性噪声并重新归一化，模拟非平稳交通条件。
- **优势**：增强 agent 对未见场景的鲁棒性，防止其记忆固定配时方案（open-loop policy），迫使模型基于状态观测进行决策。

#### ✅ **2. Exponential Phase Duration Adjustment（指数级相位持续时间调整）**
- **机制**：采用 `{0, ±1, ±2, ±4, ±8}` 这类指数增长的动作集合来微调下一周期绿灯时长。
- **优势**：实现“粗到细”（coarse-to-fine）控制——大步长快速应对突发拥堵，小步长维持稳态精度，兼顾**响应性与稳定性**。

#### ✅ **3. Neighbor-Based Observation with CTDE（基于邻居观测的集中训练分散执行）**
- **机制**：使用 MAPPO 算法，在训练阶段利用 centralized critic 获取全局信息以优化信用分配；执行阶段仅依赖本地 + 邻居交叉口的观测。
- **优势**：解决了 scalability 与 coordination 的矛盾，在保持分布式架构的同时逼近全局协作效果。

### 相比现有方法的优势
| 维度 | 本文方法 | 现有方法 |
|------|--------|---------|
| 泛化性 | ✅ 强（通过转向比例扰动提升鲁棒性） | ❌ 易过拟合静态流量 |
| 动作安全性 | ✅ 保留 cyclic 相序，符合驾驶预期 | ⚠️ 部分允许任意切换，存在安全隐患 |
| 控制精细度 | ✅ 指数调整平衡快慢调节需求 | ⚠️ 固定线性步长难以两全 |
| 可扩展性 | ✅ Neighbor-level + CTDE 实现高效协同 | ⚠️ Global 观测不可扩展，Local 缺乏协调 |

---

## 2. 核心实验方法和设置

### 使用的数据集与仿真环境
- **仿真平台**：PTV **Vissim** —— 工业界标准微观交通仿真器，采用 Wiedemann car-following 模型，具备高保真驾驶行为建模能力。
- **路网结构**：台湾桃园市中正东路五连信号交叉口数字孪生模型（短间距、强交互路段）。
- **真实交通数据**：
  - 基于实际检测器采集的 24 小时车流数据（图7）
  - 区分高峰时段（~4800 veh/h）与平峰时段（~1800 veh/h）
- **训练-测试分离策略**：
  - **训练仅用高峰数据** → 测试其在高峰和平峰下的表现，验证泛化能力
  - 避免同分布训练/测试带来的虚假性能

### 实验设置
- **智能体设置**：每个交叉口为一个独立 agent，构成 Dec-POMDP 框架
- **状态空间（Observation）**：
  - 当前相位 ID
  - 当前相位已运行时间
  - 各进口道车道车辆计数（有限范围内）
- **奖励函数**：加权组合：travel time ↓, waiting time ↓, average speed ↑, throughput ↑
- **算法基础**：Multi-Agent PPO (**MAPPO**)，支持 CTDE 范式

### 评估指标
| 指标 | 含义 |
|------|------|
| **ATT** (Average Travel Time) | 平均行程时间（秒/车） ↓ |
| **AWT** (Average Waiting Time) | 平均等待时间（秒/车） ↓ |
| **AD** (Average Delay) | 平均延误（秒/车） ↓ |
| **VC** (Vehicle Count) | 单位小时通过车辆数（辆/小时） ↑ |

### 基线方法对比
| 类别 | 方法 |
|------|------|
| 传统控制 | FixTime（绿波优化定时方案） |
| 启发式方法 | MaxPressure（经典压力最大化算法） |
| 标准 RL 方法 | M<sup>static</sup><sub>local/global/neighbor</sub>（无随机化训练） |
| 消融对照 | non-CTDE（IPPO）、Linear Adjustment 动作空间 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 方法 | 场景 | ATT (s) | AWT (s) | AD (s) | VC (vehs/h) |
|------|------|--------|--------|-------|------------|
| FixTime | Peak | 383.92 | 352.87 | 319.04 | 4015.87 |
| MaxPressure | Peak | 265.79 | 285.93 | 196.54 | 4223.80 |
| **M<sup>randomized</sup><sub>neighbor</sub> (Ours)** | **Peak** | **230.58** | **231.01** | **160.34** | **4416.53** |
| | Off-Peak | **124.37** | **44.09** | **53.44** | **1808.47** |

> ✅ 在高峰场景下，相比 MaxPressure，**ATT 降低超过 13%**  
> ✅ 在平峰场景下，**ATT 比 MaxPressure 更低（124.37 vs 126.57）**，显示极强泛化能力

### 与基线方法对比结果
- **相比 FixTime 和 MaxPressure**：
  - 在高峰和平峰均显著优于所有 baseline
  - 特别是在 ATT 和 AWT 上取得最大降幅（>10%）
- **相比标准 RL 方法（M<sup>static</sup>）**：
  - 所有 M<sup>static</sup> 方法在平峰场景性能严重退化（ATT >130s）
  - 表明**无随机化的训练极易过拟合**

### 消融实验结果

#### 🔹 CTDE vs Non-CTDE（Table 3）
| 方法 | ATT (Peak) | ATT (Off-Peak) |
|------|-----------|----------------|
| non-CTDE (IPPO) | 298.43 | 134.20 |
| **CTDE (MAPPO, Ours)** | **230.58** | **124.37** |

> ✅ 使用 centralized critic 显著提升性能，说明 CTDE 对协调至关重要

#### 🔹 动作空间比较（Table 4）
| 动作空间类型 | 设置 | ATT (Peak) | ATT (Off-Peak) |
|-------------|------|-----------|---------------|
| Linear Small-Scale | {0,±2,±4,±6,±8} | 263.11 | 158.10 |
| Linear Large-Scale | {0,±5,±10,±15,±20} | 283.56 | 144.96 |
| **Exponential (Base-2, Ours)** | **{0,±1,±2,±4,±8}** | **230.58** | **124.37** |

> ✅ 指数动作空间在两种场景下全面胜出  
> ✅ 线性方法在平峰时因过度调整导致震荡，性能下降明显

#### 🔹 Turning Ratio Randomization 的必要性
- 所有未使用 randomization 的模型在 off-peak 场景中性能骤降
- 而 M<sup>randomized</sup> 在两个场景中都保持稳定高性能
> ➤ 证明该策略是实现跨负载泛化的关键正则化手段

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Turning Ratio Randomization 是提升泛化性的核心机制**：能有效防止 agent 学习 open-loop 定时策略，促使其真正理解状态-动作映射关系。
2. ✅ **Exponential Action Space 实现了控制灵活性与稳定性的统一**：通过非均匀动作粒度设计，兼顾紧急响应与日常平稳运行。
3. ✅ **Neighbor-level Observation + CTDE 可实现近似全局协调的可扩展架构**：无需全局信息即可达成接近 global-agent 的性能（如 M<sup>randomized</sup><sub>neighbor</sub> 接近 M<sup>randomized</sup><sub>global</sub>）。
4. ✅ 所提框架在**未参与训练的低流量场景中仍表现优异**，验证了其面向真实世界部署的潜力。

### 方法的局限性
- 当前验证仅限于**单向主干道路段**（arterial road），尚未扩展至复杂 grid network。
- 转向比例扰动假设各 movement 独立扰动，未考虑更复杂的时空相关性（如上下游联动变化）。
- 依赖 VissimRL 框架集成，对其他仿真平台迁移需额外适配成本。

### 未来工作方向
- 将框架推广至 **grid-shaped 城市路网**，研究更大规模下的协调效率。
- 引入 **multi-modal traffic data**（如行人、自行车、公交车优先等）以支持综合交通管理。
- 探索 **real-world deployment pipeline**，结合在线自适应机制进一步缩小 sim-to-real gap。

--- 

> 📌 **总体评价**：本文提出了一套面向真实部署的、兼具**鲁棒性、稳定性与可扩展性**的 MARL-TSC 框架，通过三项精心设计的技术创新，在高保真仿真环境中实现了超越主流基线 10% 以上的性能增益，并展现出卓越的跨场景泛化能力，为 DRL 在智慧交通中的落地提供了重要实践路径。

</details>

---

### 10. [Try, Check and Retry: A Divide-and-Conquer Framework for Boosting Long-context Tool-Calling Performance of LLMs](https://arxiv.org/abs/2603.11495)

**Authors**: Kunfeng Chen, Qihuang Zhong, Juhua Liu, Bo Du, Dacheng Tao  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.11495v1  

#### Abstract
Tool-calling empowers Large Language Models (LLMs) to interact with external environments. However, current methods often struggle to handle massive and noisy candidate tools in long-context tool-calling tasks, limiting their real-world application. To this end, we propose Tool-DC, a Divide-and-Conq...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Try, Check and Retry: A Divide-and-Conquer Framework for Boosting Long-context Tool-Calling Performance of LLMs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Large Language Models (LLMs)** 在执行 **tool-calling** 任务时面临两大挑战：
- **长上下文推理困难**：当候选工具数量增多时，模型需处理更长的上下文，导致性能显著下降。
- **语义相似但参数不同的工具干扰**：大量功能相似但参数定义不同的工具容易引发参数填充错误（argument-filling errors）。

这些问题在真实场景中尤为突出，因为实际应用通常涉及数十甚至上百个候选工具，而现有基准（如 BFCL）中的候选工具数量普遍不足（<10），无法反映现实复杂性。

### 提出了什么新方法或新思路
作者提出 **Tool-DC**，一个基于 **Divide-and-Conquer** 范式的框架，通过 “**Try-Check-Retry**” 三阶段流程提升 LLMs 在长上下文下的 tool-calling 性能。

该框架包含两个变体：
- **Tool-DC (TF)**：训练免费（training-free），即插即用，适用于任意 LLM。
- **Tool-DC (TB)**：基于训练（training-based），将 Try-Check-Retry 的推理轨迹内化到模型参数中，提升推理效率。

#### 三阶段详解：
1. **Try（分组并并行推理）**  
   将所有候选工具划分为多个子集（subspaces），利用 **Strategic Anchor Grouping** 策略构建并行推理组，减少每轮推理的噪声和上下文长度。
   
2. **Check（模式验证）**  
   使用规则驱动的 **Consistency Validator** 对生成的 tool calls 进行严格校验，确保：
   - 函数名存在
   - 参数键正确且完整
   - 数据类型匹配
   
3. **Retry（全局聚合与重试）**  
   基于 Check 阶段筛选出的有效候选工具，重新组织上下文进行最终决策，充分利用 LLM 的自省能力（self-reflection）优化输出。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **灵活性** | Tool-DC (TF) 不依赖额外训练，可直接部署于任何 LLM，适配性强。 |
| **鲁棒性** | 显著缓解因候选工具数量增加带来的性能衰减，尤其对小规模模型效果明显。 |
| **高效性** | Tool-DC (TB) 将多步推理压缩为单次前向传播，推理延迟更低。 |
| **通用性** | 方法不依赖特定 retriever 或 prompt 设计，在多种 LLM 架构上均有效。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **BFCL (Berkeley Function-Calling Leaderboard)** (Patil et al., 2025)  
  包含 Non-Live（合成）和 Live（人工编写）两类任务，涵盖 single/multiple/parallel function invocation 场景。
- **ACEBench** (Chen et al., 2025a)  
  多领域工具调用评测集，覆盖技术、金融、健康等 8 大类，细分为 Single Turn、Multi Turn、Similar API 等子任务。

> ⚠️ 为模拟真实场景，作者引入 **Extended Setting**：将候选工具从标准的 <10 扩展至 **20 个**，其中包含大量无关工具以测试抗噪能力。

### 实验设置和评估指标
- **评估指标**：采用严格的 **AST exact-match accuracy**（抽象语法树完全匹配准确率）作为主指标。
- **模型范围**：
  - 开源模型：Qwen2.5 系列（1.5B/3B/7B）、Llama-3.1/3.2、Gemma-3-it
  - 闭源模型：GPT-4o-mini、DeepSeek-V3.2
- **实现细节**：
  - Retriever 使用 **BM25**（无监督稀疏检索）
  - Tool-DC (TF) 中分组数 $ K = \min(5, N) $
  - Tool-DC (TB) 基于 **xlam-function-calling-60k** 构建 CoT 数据集，并使用 LoRA 微调

### 基线方法对比
#### 训练免费方法（Training-free Baselines）：
- `GT_Funs`：仅提供真实工具（理想上限）
- `All_Funs`：提供全部候选工具（朴素 baseline）
- `Top-K`：使用 BM25 检索 top-K 工具
- `HiTEC-ICL`：基于手动设计的 error checklist 进行 ICL 推理
- `ToolGT (Prompting)`：课程式提示引导逐步调用

#### 训练基础方法（Training-based Baselines）：
- `Vanilla SFT`：直接监督微调，无推理链
- 多个专有模型（如 OpenAI o3、Claude-Haiku-4.5）用于横向比较

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | BFCL Overall (Qwen2.5-1.5B) | ACEBench Overall (Qwen2.5-1.5B) | 提升幅度 |
|------|-------------------------------|----------------------------------|---------|
| All_Funs (Standard) | 64.07% | 46.33% | — |
| Tool-DC (TF) (Standard) | **70.53%** | **49.08%** | +6.46% / +2.75% |
| All_Funs (Extended) | 36.96% | 22.00% | — |
| Tool-DC (TF) (Extended) | **63.07%** | **46.08%** | **+25.10% avg** |

> ✅ 在扩展设置下，Tool-DC (TF) 对 Qwen2.5-1.5B 带来高达 **+25.10% 的平均增益**，表明其在高噪声环境下极强的鲁棒性。

### 与基线方法的对比结果
- **在 Qwen2.5 全系列上，Tool-DC (TF) 均取得最优性能**，尤其在小模型上优势更大。
- **在 Llama/Gemma/GPT-4o-mini 上也一致提升性能**：
  - GPT-4o-mini 在 BFCL 上获得 **+5.3%** 增益
  - Llama-3.2-3B 提升 **+20.4%**
- **Tool-DC (TB) 微调后表现超越多个闭源模型**：
  - Qwen2.5-7B-w/Tool-DC(TB) 在 BFCL 上达到 **83.16% overall**
  - 超过 **OpenAI o3 (77.58%)**, **Claude-Haiku-4.5 (82.59%)**

### 消融实验结果（Ablation Study）
使用 Qwen2.5-3B-Instruct 在 Extended Setting 下进行消融分析：

| 方法 | Non-Live | Live | Overall |
|------|--------|------|--------|
| Full Tool-DC (TF) | 71.79 | 57.74 | **64.77** |
| w/o Try | 44.19 | 29.39 | 36.79 ↓ |
| w/o Check | 52.06 | 52.41 | 52.24 ↓ |
| w/o Retry | 7.33 | 3.18 | **5.26 ↓↓** |

> 🔍 发现：
> - 移除 **Retry** 导致性能崩溃（从 64.77 → 5.26），说明其是整合有效信号的关键。
> - **Try** 和 **Check** 分别负责降低推理难度和过滤幻觉，缺一不可。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **长上下文 tool-calling 是现实瓶颈**：即使是最新的长上下文模型（如 Qwen2.5-7B-1M），在候选工具增多时仍会严重退化。
2. **Divide-and-Conquer 有效缓解长上下文压力**：通过分组推理 + 自我验证机制，显著提升模型在大规模候选集下的准确性。
3. **Try-Check-Retry 范式释放了 LLM 的自省潜力**：Check 提供反馈信号，Retry 利用该信号进行自我修正，形成闭环优化。
4. **Tool-DC (TB) 可实现高性能与低延迟兼得**：将多步推理内化为模型能力，推理速度优于 TF，性能媲美甚至超越闭源模型。

### 方法的局限性
1. **依赖外部 retriever 初始化（TF 版本）**：虽然可通过枚举策略缓解，但仍可能遗漏关键工具。
2. **CoT 数据多样性有限**：目前仅基于 xlam-function-calling-60k 构建，缺乏多步嵌套调用场景。
3. **未支持 multi-step/nested tool-calling**：当前实验集中在单步调用任务，尚未验证在复杂代理流程中的表现。

### 未来工作方向
- 构建更具挑战性的 **大规模、高噪声、多步骤 tool-calling 数据集**
- 引入 **Reinforcement Learning**（如 GRPO）进一步优化决策过程
- 探索 **无需 retriever 的端到端分组机制**
- 将 Tool-DC 扩展至 **multi-agent** 和 **agent workflow** 场景

---

> 📌 **一句话总结**：  
> Tool-DC 通过“尝试→检查→重试”的分治策略，显著提升了 LLM 在长上下文、高噪声环境下的 tool-calling 能力，兼具即插即用的灵活性与训练优化的高效性，是迈向实用化 AI Agent 的重要一步。

</details>

---

### 11. [HPC Containers for EBRAINS: Towards Portable Cross-Domain Software Environment](https://arxiv.org/abs/2603.12044)

**Authors**: Krishna Kant Singh, Eric M\"uller, Eleni Mathioulaki, Wouter Klijn, Lena Oden  
**Category**: cs.DC  
**Published**: 2026-03-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.12044v1  

#### Abstract
Deploying complex, distributed scientific workflows across diverse HPC sites is often hindered by site-specific dependencies and complex build environments. This paper investigates the design and performance of portable HPC container images capable of encapsulating MPI- and CUDA-enabled software sta...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HPC Containers for EBRAINS: Towards Portable Cross-Domain Software Environment

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在高性能计算（HPC）环境中，复杂科学工作流的部署面临以下挑战：
- **软件可移植性差**：不同HPC站点存在依赖冲突、工具链不兼容、构建环境差异等问题。
- **性能损失风险**：传统容器化技术（如Docker）在HPC中常因网络和GPU加速器隔离导致通信性能下降。
- **可重复性不足**：科研成果难以复现，部分原因在于缺乏对完整计算环境（操作系统、库版本等）的有效封装。

该研究聚焦于如何在不牺牲裸金属（bare-metal）性能的前提下，实现跨异构HPC平台的**高性能、可移植、可验证的容器化部署方案**。

---

### 🚀 提出的新方法与创新思路
提出了一种基于 **PMIx 的混合式 Apptainer 容器化策略**，其核心思想是：

- **完全自包含的MPI栈**：容器内嵌完整的 Open MPI 5 栈（包括 PMIx、UCX、PRRTE、hwloc 等），无需依赖宿主机MPI安装。
- **动态利用宿主硬件资源**：通过 `--mpi=pmix` 启动协议，让容器内的MPI运行时通过PMIx客户端接口查询Slurm资源管理器，自动发现分布式端点并绑定底层高速互连（如InfiniBand）和GPU设备。
- **统一构建流程集成到EBRAINS Software Distribution (ESD)**：结合 Spack 包管理系统，设计了一个可在HPC上并行构建容器镜像的CI流程（dedal），支持自动化生成性能验证过的便携式HPC容器。

---

### 🔍 相比现有方法的优势
| 方面 | 传统做法 | 本文方法 |
|------|--------|---------|
| **MPI兼容性** | 需要现场编译或WI4MPI等翻译层 | 利用MPI-5 ABI标准 + PMIx桥接，无需重新编译 |
| **GPU支持** | 易受CUDA驱动版本限制 | 支持向后兼容（旧版CUDA容器可运行在新版驱动上） |
| **跨平台移植性** | 弱，需针对每个系统定制 | 单个镜像可在多个HPC集群直接运行 |
| **性能保障** | 缺乏系统性验证机制 | 微基准测试 + 应用级扩展性分析 + 日志诊断 |
| **运维效率** | 用户需具备HPC专业知识 | 自动化CI/CD流程降低使用门槛 |

> 💡 **关键优势总结**：实现了“一次构建、随处高效运行”（build once, run anywhere with native-like performance）的目标。

---

## 2. 核心实验方法和设置

### 🧪 实验平台
在两个生产级HPC集群上进行测试：

| 集群 | Karolina (Czech Republic) | JURECA-DC (Germany) |
|------|--------------------------|--------------------|
| CPU | AMD EPYC 7H12 (128核/节点) | AMD EPYC 7742 (128核/节点) |
| GPU | 8× NVIDIA A100 (NVLink12) | 4× NVIDIA A100 (NVLink4) |
| 网络 | InfiniBand HDR | InfiniBand HDR100 |
| 资源管理器 | Slurm + PMIx | Slurm + PMIx |
| 容器引擎 | Apptainer 1.4.x | Apptainer 1.4.x |

> 所有容器镜像均**仅构建一次**，未经修改即部署于两套异构系统。

---

### 📦 构建与部署流程
- 使用 **Spack** 管理约80个顶层软件包及其800+依赖项。
- 构建分为两个阶段：
  1. **Fetch阶段**：获取所有源码；
  2. **Build阶段**：在计算节点本地内存文件系统（`/dev/shm`）中完成编译，避免共享存储I/O瓶颈。
- 最终打包为两种Apptainer镜像：
  - CPU-only 镜像
  - GPU-accelerated 镜像（CUDA 12.2）

---

### 🎯 评估指标与方法
#### （1）微基准测试（Microbenchmarks）
| 测试类型 | 工具 | 指标 |
|--------|-----|-----|
| 初始化开销 | OSU_init | `MPI_Init()` 时间 |
| 通信延迟 | OSU_latency | 点对点消息延迟（intra-/inter-node） |
| GPU通信带宽 | NCCL-tests (`all_reduce_perf`) | AllReduce总线带宽 |

#### （2）应用级基准测试（Neuroscience Workloads）
| 模拟器 | 类型 | 测试模式 |
|-------|------|---------|
| **Arbor** | CPU/GPU | Strong & Weak Scaling |
| **NEURON** | CPU | Strong & Weak Scaling |

> 所有测试均对比 **原生执行（native）** 与 **容器化执行（Apptainer）** 性能。

---

### ⚖️ 基线方法对比
- **Baseline**: 原生模块加载方式（module load）下的本地部署
- **Target**: Apptainer容器化部署（相同软件栈，独立镜像）
- 不同之处在于是否经过容器隔离层，其余配置尽可能一致。

---

## 3. 主要实验结果和性能指标

### 📊 微基准测试结果

#### ✅ OSU_init（MPI初始化时间）
| 平台 | 结果 |
|------|------|
| **Karolina** | 容器比原生慢，且随节点数增加差距扩大（最大+~40% @256节点） |
| **JURECA-DC** | 容器反而快约 **50%**，表明容器环境可能绕过了冗余探测过程 |

> ❗说明：容器初始化性能**高度依赖宿主系统的PMIx启动路径优化程度**，非固定开销。

#### ✅ OSU_latency（通信延迟）
- **intra-node & inter-node** 延迟几乎无差异：
  - 小消息（≤1KB）额外延迟 < 0.2 μs
  - 中大消息段性能曲线完全重合
- **结论**：Apptainer未引入可观测的通信延迟开销。

#### ✅ NCCL-tests（GPU通信）
| 场景 | 性能偏差 |
|------|--------|
| 单节点（NVLink） | ≤1.3% 差异 |
| 双节点（RDMA over IB） | ≤0.1% 差异 |
| GPUDirect RDMA | 成功启用，带宽饱和 |

> ✅ 表明容器中GPU Direct RDMA正常工作，无性能损失。

---

### 📈 应用级性能结果

#### 🔹 Arbor（CPU）
- **强扩展性（Strong Scaling）**
  - Karolina：容器效率达原生的 **62.6%** vs 原生67.5%
  - JURECA：容器效率 **98.0%**，优于原生（95.9%）
- **弱扩展性（Weak Scaling）**
  - 两平台上容器与原生性能基本一致，波动在测量噪声范围内

> ✔️ 容器未影响CPU模拟器的并行效率。

#### 🔹 NEURON（CPU）
- 强/弱扩展性下，容器与原生曲线**几乎完全重叠**
- 在Karolina上实现从单节点250s到256节点1.5s的超线性加速
- 容器差异在运行间变异范围内

> ✔️ 容器对纯CPU神经模拟无显著影响。

#### 🔹 Arbor（GPU）
| 指标 | 结果 |
|------|------|
| **相对性能损失** | 容器比原生慢 **12%–19%** |
| **开销性质** | 固定相对开销（非随规模增长） |
| **弱扩展性** | 绝对延迟恒定（~+13s），比例稳定（~17%） |
| **强扩展性** | 开销占比随节点增多而减小（因每GPU负载下降） |

> ⚠️ 存在**可复现但来源未明的GPU特定开销**，推测与CUDA版本差异或初始化I/O有关。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **可行性已证实**：
   - 基于Apptainer + PMIx的容器策略可在真实HPC集群上实现接近裸金属性能的执行。
   - 支持MPI和CUDA应用的跨平台无缝迁移。

2. **通信性能无损**：
   - OSU和NCCL测试显示，**网络与GPU通信性能与原生一致**，证明UCX、InfiniBand Verb、NVLink、GPUDirect RDMA均可被正确调用。

3. **CPU应用近乎零开销**：
   - 对Arbor和NEURON等神经模拟器，容器化带来的性能差异在误差范围内。

4. **GPU存在轻微固定开销**：
   - 观察到 **12%-19% 的恒定性能下降**，但不影响扩展行为，属于可接受折衷。

5. **容器成为系统健康检测工具**：
   - 通过对比容器与原生性能，发现了JURECA-DC上的MPI启动延迟异常和集体通信带宽退化问题，反向促进系统调优。

---

### ⚠️ 局限性
1. **GPU性能开销尚未定位**：
   - 当前未深入分析CUDA用户态库（如`libcuda.so`）与内核驱动间的交互细节。
   - 缺少对不同CUDA版本组合的影响研究。

2. **生态系统级验证不足**：
   - 当前仅测试了Arbor、NEURON等少数组件，尚未覆盖全部80+ ESD软件包的耦合工作流。

3. **专家干预仍必要**：
   - 检测次优传输路径（如回落到TCP而非IB）仍需人工查看调试日志。

4. **构建流程尚未完全自动化**：
   - 当前dedal流程仍在整合中，尚未全面接入ESD的CI/CD流水线。

---

### 🔮 未来工作方向
1. **自动化日志解析与调优建议**：
   - 开发工具自动分析PMIx/UCX/NCCL日志，识别非最优路径并提示修复。

2. **扩展至复合工作流测试**：
   - 在CI中加入多工具串联的功能与性能回归测试。

3. **集成EESSI与Site-Optimized Libraries**：
   - 自动挂载站点预优化的通信库（如厂商定制UCX），进一步提升性能。

4. **支持“软件流式传输”（Software Streaming）**：
   - 借鉴CVMFS/EESSI模式，按需加载容器内容，减少初始I/O压力。

5. **推动临床子集发布**：
   - 创建轻量、安全的ESD子集容器，用于处理敏感医疗数据，确保行为一致性。

---

> 🏁 **最终目标**：建立一个**自动化、可验证、高性能、易用**的HPC容器交付管道，使研究人员能专注于科学问题本身，而非底层基础设施差异。

</details>

---

### 12. [NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication](https://arxiv.org/abs/2603.11438)

**Authors**: Yusheng Zheng  
**Category**: cs.DC  
**Published**: 2026-03-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.11438v1  

#### Abstract
NCCL is the de facto standard for collective GPU communication in large-scale distributed training, relying heavily on plugins to customize runtime behavior. However, these plugins execute as unverified native code within NCCL's address space, risking job crashes, silent state corruption, and downti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **NCCL 插件系统的安全性和可靠性问题**：当前 NCCL（NVIDIA Collective Communications Library）通过原生 `dlopen` 加载的插件（如 tuner、profiler、net plugin）以无验证的 native code 形式运行在主进程中，存在严重风险：
  - 内存错误（如空指针解引用、use-after-free）可导致训练任务崩溃；
  - 死循环或竞态条件可能造成集体通信静默失败；
  - 插件间缺乏结构化状态共享机制，无法实现闭环自适应策略；
  - 更新插件需重启作业，带来生产环境中的显著停机时间。

### 提出的新方法与思路
提出 **NCCLbpf** —— 一个将 **userspace eBPF 运行时嵌入 NCCL 插件接口** 的新型框架，无需修改 NCCL 源码即可实现：
- **Load-time 静态验证**：利用 eBPF 的 PREVAIL 验证器，在加载阶段检查内存安全、终止性、栈安全等属性，防止不安全代码执行；
- **结构化跨插件通信**：通过 typed eBPF maps 实现 tuner 与 profiler 等插件之间的安全、原子状态共享；
- **原子热更新（Atomic Hot-Reload）**：支持在线替换策略程序而不停止训练任务，确保零调用丢失。

### 相比现有方法的优势
| 维度 | 传统 NCCL Plugin | NCCLbpf |
|------|------------------|--------|
| 安全性 | 无验证，易崩溃 | Load-time 验证，杜绝 crash/hang |
| 可组合性 | 插件孤立，无法协同 | 支持跨插件闭环反馈（如 profiler → tuner） |
| 可维护性 | 更新需重启作业 | 支持原子热更新，<1.07μs 切换延迟 |
| 性能开销 | 原生性能 | <0.03% 集体通信延迟增加 |
| 兼容性 | 直接集成 | 不修改 NCCL，兼容现有 ABI |

> ✅ **核心创新**：首次将 eBPF 的“内核级安全扩展”范式成功迁移至 **GPU 集体通信库层**，实现了 **verified + composable + hot-swappable** 的政策执行模型。

---

## 2. 核心实验方法和设置

### 实验平台
- **硬件配置**：
  - 8× NVIDIA B300 SXM6 GPUs（Blackwell 架构，每卡 275GB）
  - 通过 **NVLink 5** 互连（带宽达 1.8TB/s/GPU）
- **软件环境**：
  - CUDA 13.0
  - NCCL 2.29.7
  - bpftime 用户态 eBPF 运行时（基于 LLVM JIT 和 PREVAIL 验证器）

### 评估指标
| 类别 | 指标 |
|------|------|
| **性能开销** | 单次策略决策延迟（P50/P99）、collective 通信吞吐量（GB/s） |
| **安全性** | 对非法行为的拦截能力（null deref, oob access, infinite loop 等） |
| **热更新能力** | reload 时间、是否丢弃调用、一致性保障 |
| **功能有效性** | 自定义策略对 AllReduce 吞吐提升效果 |
| **稳定性** | 多轮测试下的方差（CV）、异常值检测 |

### 基线方法对比
- **Native Baseline**：相同逻辑的 C++ 插件（开启 `-O2`），用于衡量 eBPF 层额外开销；
- **No-plugin Baseline**：关闭所有插件的原始 NCCL 行为；
- **Bad Policy**：人为构造低效策略（如强制单 channel）作为下界参考；
- **Static Environment Tuning**：使用 `NCCL_ALGO=...` 等全局变量设定算法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）eBPF 策略执行开销（CPU 微基准）
| Policy | P50 (ns) | AP50 (vs Native) |
|--------|----------|------------------|
| Native baseline | 20 ns | — |
| noop | 100 ns | +80 ns |
| size_aware_v2 | 100 ns | +80 ns |
| lookup_update | 140 ns | +120 ns |
| slo_enforcer | 150 ns | +130 ns |

> 🔹 **结论**：最复杂策略仅引入 **~130 ns 开销**，占典型 AllReduce 延迟（~394 μs @128MiB）的 **<0.03%**。

#### （2）端到端 GPU 通信性能（NVLink 上 AllReduce）
| 消息大小 | NCCL 默认 (NVLS) | NCCLbpf (eBPF Policy) | 提升幅度 |
|---------|------------------|------------------------|----------|
| 4 MiB   | 133.5 GB/s       | 148.1 GB/s             | +10.9%   |
| 8 MiB   | 196.3 GB/s       | 249.7 GB/s             | **+27.2%** |
| 16 MiB  | 278.8 GB/s       | 337.4 GB/s             | +21.0%   |
| 32 MiB  | 349.3 GB/s       | 402.4 GB/s             | +15.2%   |
| 64 MiB  | 425.2 GB/s       | 471.8 GB/s             | +11.0%   |
| 128 MiB | 596.9 GB/s       | 628.9 GB/s             | +5.4%    |
| ≥256 MiB | 更优           | 略低或持平             | -3.7% ~ -16.6% |

> 📈 **亮点**：eBPF 策略可根据消息大小动态选择 Ring + LL128/Simple 协议，在 **4–128 MiB 区间最高提升 27% 吞吐**。

#### （3）安全性与热更新表现
- **安全验证结果**：
  - 测试 14 个程序（7 安全 + 7 不安全）；
  - 所有不安全程序均被 **load-time 拒绝**，包括：
    - Null pointer dereference
    - Out-of-bounds memory access
    - Unbounded loop
    - Stack overflow
    - Division by zero
    - Illegal helper call
    - Input field write
- **热更新性能**：
  - Reload 总耗时：~9.4 ms（含验证 + JIT 编译）
  - 原子切换时间：**1.07 μs**
  - 在连续 400,000 次调用中 **零调用丢失**
  - 若新策略验证失败，旧策略继续运行（fail-safe）

#### （4）消融实验与案例研究
- **Profiler-to-Tuner 闭环控制**：
  - 初始保守设置 `nChannels=2`
  - Profiler 收集延迟并写入 shared map
  - Tuner 动态上调至 12 channels
  - 注入延迟后自动降回 2，恢复后再上升 → 成功验证 **跨插件协作闭环**
- **Net Plugin 扩展性验证**：
  - 封装 Socket 传输层，插入 eBPF hook 统计连接数与字节数
  - 开销 <2%，证明可在数据路径部署轻量监控策略

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **eBPF 能有效解决 NCCL 插件的安全隐患**：静态验证可在运行前捕获所有内存安全与终止性缺陷，避免训练中断。
2. ✅ **结构化 map 使跨插件协同成为可能**：首次实现 tuner 与 profiler 的闭环反馈，支持动态自适应策略。
3. ✅ **极低运行时开销**：每决策仅增加 80–130 ns，对整体通信性能影响可忽略（<0.03%）。
4. ✅ **真正的无缝热更新**：原子替换保证服务连续性，为生产环境快速迭代提供基础。
5. ✅ **实际性能增益显著**：基于消息大小的自适应策略在关键区间（4–128 MiB）提升 AllReduce 吞吐高达 **27%**。

### 方法的局限性
- ❗ **不防御语义错误**：验证仅保证内存安全和终止性，不能阻止逻辑错误（如 bad_channels 强制单通道仍可通过验证但严重降速）；
- ❗ **当前覆盖范围有限**：目前支持 tuner、profiler、net plugin；env plugin 尚未完全集成；
- ❗ **尚未验证大规模多节点场景**：实验限于单节点 8 GPU NVLink 环境，InfiniBand 多节点扩展待验证；
- ❗ **编程门槛较高**：需熟悉 eBPF 编程模型，对普通 ML 工程师不够友好。

### 未来工作方向
- 🔄 **构建高层 DSL**：开发专用于通信调优的领域特定语言（DSL），编译为 eBPF 字节码，降低使用门槛；
- 🌐 **扩展至多节点与 RDMA 场景**：支持 InfiniBand 和 RoCE 网络下的 net plugin 深度集成；
- ⚙️ **支持更多插件类型**：完整集成 env plugin，并探索对 MSCCL 或 AutoCCL 的增强支持；
- 🤝 **跨厂商适配**：将类似架构应用于 AMD 的 RCCL，推动行业标准化；
- 🧠 **结合机器学习进行智能调优**：利用 eBPF 收集的细粒度 telemetry 数据训练 RL 模型，实现自动化策略生成。

---

> 💡 **总体评价**：  
> NCCLbpf 是一次成功的系统抽象迁移实践——将 **eBPF 在操作系统内核的成功经验**（安全、可观测、可扩展）成功复用到了 **高性能 GPU 通信库** 中。它不仅解决了长期存在的插件安全隐患，还打开了 **可组合、可进化、可验证的分布式训练优化新范式**，具有重要的工程价值和研究启发意义。

</details>

---

### 13. [Beyond Barren Plateaus: A Scalable Quantum Convolutional Architecture for High-Fidelity Image Classification](https://arxiv.org/abs/2603.11131)

**Authors**: Radhakrishnan Delhibabu  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.11131v1  

#### Abstract
While Quantum Convolutional Neural Networks (QCNNs) offer a theoretical paradigm for quantum machine learning, their practical implementation is severely bottlenecked by barren plateaus -- the exponential vanishing of gradients -- and poor empirical accuracy compared to classical counterparts. In th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond Barren Plateaus: A Scalable Quantum Convolutional Architecture for High-Fidelity Image Classification*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对 **Quantum Convolutional Neural Networks (QCNN)** 在实际应用中的核心瓶颈——**Barren Plateaus（贫瘠高原）** 问题展开研究。  
Barren Plateaus 是指在训练深度 Parameterized Quantum Circuits (PQCs) 时，梯度方差随量子比特数指数级衰减，导致优化过程陷入平坦的损失景观，无法有效更新参数，从而严重限制了 QCNN 的可扩展性和分类性能。

此前的 QCNN 模型在 MNIST 等标准数据集上准确率仅约 52%，远低于经典 CNN 的 >99%，其根本原因正是未解决的 Barren Plateaus 导致训练失败。

---

### 🚀 提出的新方法与创新思路

作者提出了一种**新型可扩展 QCNN 架构**，通过以下两个关键技术组合，**从理论上和实证上解决了 Barren Plateaus 问题**：

#### （1）**Localized Cost Function（局部化成本函数）**
- 替代传统的全局可观测量（如 $ H_G = |0\rangle\langle0|^{\otimes m} $），改用对每个存活 qubit 测量局部 Pauli-Z 期望值。
- 成本函数定义为：
  $$
  C_L(\theta) = \frac{1}{m} \sum_{i=1}^m \left(1 - \langle Z_i \rangle_\theta \right)
  $$
- 根据 Cerezo et al. [5] 的理论，局部成本函数的梯度方差衰减由指数级 $ O(2^{-n}) $ 改善为多项式级 $ \Omega(1/\text{poly}(n)) $，从根本上避免了梯度消失。

#### （2）**Tensor Network Initialization (TNI) 参数初始化协议**
- 利用经典 **Matrix Product State (MPS)** 和 **Tree Tensor Network (TTN)** 对 QCNN 进行近似建模。
- 在经典计算平台上预训练 TTN 模型，获得高质量的初始参数种子 $ \theta_{\text{seed}} $。
- 将这些参数“热启动”（warm-start）导入量子电路，使优化起点直接位于收敛漏斗内，绕过平坦区域。

> 🔍 **创新本质**：将量子训练难题部分前移到经典高性能量子模拟器上处理，实现“经典预热 + 量子精调”的混合范式。

---

### ⚖️ 相比现有方法的优势

| 维度 | 传统 QCNN | 本文方法 (Scalable QCNN) |
|------|-----------|--------------------------|
| **成本函数设计** | 全局可观测量 → 易出现 Barren Plateaus | 局部可观测量 → 理论规避 BP |
| **参数初始化** | 随机初始化 → 易陷入局部极小 | TNI 预训练 → 快速进入收敛区 |
| **参数复杂度** | $ O(\log N) $，但不可训练 | $ O(\log N) $，且可高效训练 |
| **硬件友好性** | 深度大、需全连通 | 浅层树结构 + 局部测量 → 更适合 NISQ 设备 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **MNIST 手写数字数据集**
  - 任务：二分类（识别数字 '0' vs '7'）
  - 图像尺寸：28×28 → 展平为 784 维向量
  - 幅度编码（Amplitude Encoding）前填充至长度 1024 ($ 2^{10} $)，使用 **10 个 qubits**

---

### ⚙️ 实验设置

| 设置项 | 描述 |
|-------|------|
| **量子框架** | Cirq + TensorFlow Quantum (TFQ) |
| **状态准备** | Amplitude Encoding via `tfq.layers.AddCircuit` |
| **卷积单元** | Brick-layer pattern 的两量子比特块：<br>$ U_{\text{block}}(\phi) = (R_y(\phi_3)\otimes R_y(\phi_4)) \cdot CZ \cdot (R_y(\phi_1)\otimes R_y(\phi_2)) $ |
| **池化操作** | $ V_{\text{pool}}(\theta) = \text{CNOT}_{j,k} \cdot (I \otimes R_y(\theta)) \cdot \text{CNOT}_{j,k} $，随后对目标 qubit 进行 partial trace |
| **优化器** | Adam ($ \beta_1=0.9, \beta_2=0.999 $) |
| **学习率** | 初始 0.015，指数衰减（$ \gamma=0.9 $） |
| **Batch Size** | 32 |
| **训练轮次** | 150 epochs |
| **梯度计算** | Parameter-Shift Rule（避免 shot noise 影响） |

---

### 🧪 基线方法对比

| 模型 | 参数数量 | 是否存在 Barren Plateaus | 准确率 |
|------|---------|------------------------|--------|
| Classical CNN (ResNet-lite) | ~120,000 | 否（ReLU 缓解梯度消失） | 99.9% |
| Baseline QCNN (Global Cost) | 45 | 是（指数梯度衰减） | 52.32% |
| **Proposed Scalable QCNN** | **45** | **否（理论规避）** | **98.7%** |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

- **测试准确率：98.7% ± 0.1%**
  - 训练集准确率：98.7%
  - 验证集准确率：98.1%
- **参数效率极高**：
  - 仅需 **45 个可训练参数**
  - 相比经典 CNN（~1.2×10⁵ 参数），减少超过 **三个数量级**
- **收敛速度快且稳定**：
  - 无剧烈震荡，MSE 损失平滑下降
  - 在 150 轮内完成收敛

---

### 🔍 与基线方法的对比结果

| 指标 | Baseline QCNN | Proposed QCNN |
|------|---------------|----------------|
| Accuracy | 52.32% | **98.7%** (+46.38 pp) |
| Trainable? | ❌（陷入 Barren Plateau） | ✅（成功收敛至全局最优） |
| Gradient Variance @ n=10 | < 1e-4 | ~1e-2（保持有效信号） |
| Parameter Count | 45 | 45（相同规模下巨大提升） |

> 💡 **意义**：证明了即使参数极少，只要架构合理，QCNN 完全可以达到接近经典的高性能。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）TNI 初始化的影响
- **随机初始化**：
  - 初始损失高
  - 30% 以上概率陷入局部最小（最高准确率 ~88%）
- **TNI 初始化**：
  - 初始损失降低 **42%**
  - 几乎总是快速收敛到全局最优（>98%）
- ➤ 结论：**TNI 是实现高保真度的关键算法组件**

#### （2）局部 vs 全局成本函数的梯度方差比较（图5）
- **全局成本函数**：梯度方差随 qubit 数指数衰减（$ \sim O(2^{-n}) $）
- **局部成本函数**：梯度方差呈多项式衰减（$ \sim \Omega(1/\text{poly}(n)) $）
- 在 $ n=10 $（MNIST 所需）时，局部方案梯度强度高出 **两个数量级以上**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Barren Plateaus 可被系统性规避**：
   - 通过结合 **Localized Cost Function** 与 **QCNN 的树状结构**，可在理论上保证梯度不会指数消失。
   
2. **参数效率优势显著**：
   - 实现 $ O(\log N) $ 参数复杂度，远优于经典 CNN 的 $ O(N^2) $，具备潜在的**量子优势**（尤其在高维数据场景）。

3. **TNI 是实用化的关键**：
   - 即便有良好损失几何，非凸优化仍易陷于局部极小；TNI 提供确定性“热启动”，极大提高成功率。

4. **模型具有强噪声鲁棒性（NISQ 友好）**：
   - 在 depolarizing noise 达 1% 时，准确率仍保持在 **94.2%**
   - 在 5% 错误率下仍高于随机猜测（>60%）
   - 原因：局部测量 + 浅层结构 → 抗干扰能力强

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **当前仅验证于小规模图像** | MNIST（10 qubits），尚未扩展到 CIFAR-10 或 ImageNet 级别 |
| **依赖经典预训练资源** | TNI 需要高性能经典计算支持，可能成为瓶颈 |
| **物理部署挑战** | 需要 SWAP routing 应对硬件连接限制，增加深度和误差 |
| **任务限定为二分类** | 多类扩展需进一步验证 |

---

### 🔮 未来工作方向

1. **扩展至多类别图像分类任务**（如 MNIST 全集、Fashion-MNIST）
2. **在真实 NISQ 设备上部署**（如 IBM Quantum、IonQ）
3. **集成先进错误缓解技术**：
   - Zero-Noise Extrapolation (ZNE)
   - Probabilistic Error Cancellation (PEC)
4. **探索隐私保护应用场景**：
   - 如医疗影像分析中利用量子态天然模糊性进行数据脱敏
5. **与其他 VQA 架构融合**：
   - 如结合 QAOA 或 Hamiltonian Variational Ansatz 提升表达能力

---

## ✅ 总结

> 本文首次实现了 **高保真（98.7%）、可扩展、抗贫瘠高原的 QCNN 架构**，通过 **Localized Cost Function + Tensor Network Initialization** 的双重机制，突破了制约量子机器学习发展的核心障碍。实验不仅验证了理论预测，更展示了在参数效率上的压倒性优势，为未来在 NISQ 设备上实现真正有意义的 **Quantum Advantage in Computer Vision** 奠定了坚实基础。

</details>

---

### 14. [Deep Learning Network-Temporal Models For Traffic Prediction](https://arxiv.org/abs/2603.11475)

**Authors**: Yufeng Xin, Ethan Fan  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.11475v1  

#### Abstract
Time series analysis is critical for emerging net- work intelligent control and management functions. However, existing statistical-based and shallow machine learning models have shown limited prediction capabilities on multivariate time series. The intricate topological interdependency and complex ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Deep Learning Network-Temporal Models For Traffic Prediction》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- 针对**网络流量预测中的多变量时间序列（MTS）建模挑战**，特别是：
  - 复杂的**非平稳性、非正态分布、长短周期混合**的时间模式；
  - 网络拓扑结构带来的**空间依赖性和异构相关性**；
  - 现有统计模型（如SARIMAX）和浅层机器学习方法在大规模、高维、非线性网络数据上的预测能力有限。

### 🚀 提出的新方法与创新思路
1. **定制化的 Network-Temporal Graph Attention Network (NT-GAT)**  
   - 将图注意力机制（GAT）与LSTM结合，显式建模**网络拓扑依赖关系**和**时间动态特征**。
   - 支持多跳邻域聚合，灵活捕捉局部与全局网络影响。

2. **增强型多模态大语言模型框架：Cluster-CALF**
   - 基于 **CALF（Cross-Modal LLM Fine-Tuning）** 架构，首次将LLM应用于网络级MTS预测任务。
   - 引入 **Spearman相关性聚类预处理步骤**，形成 **Cluster-CALF** 框架：
     - 利用Spearman秩相关度量构建鲁棒的相似性矩阵；
     - 对高维时间序列进行聚类分组，缓解多重共线性并降低输入复杂度；
     - 每个簇独立送入CALF模型，提升训练效率与泛化能力。

### 🔍 相比现有方法的优势
| 方法 | 优势 |
|------|------|
| **NT-GAT** | 显式融合网络拓扑结构，减少不同时间序列间的预测方差；适合具有强空间依赖性的场景 |
| **Cluster-CALF** | 利用LLM强大的序列建模能力和zero/few-shot潜力，在整体预测精度上显著优于传统DL模型；通过聚类进一步优化性能分布 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用来自某互联网骨干网服务提供商的真实**小时级流量数据集**，持续一年；
- 包含近**一百条双向链路**（即约200个时间序列）；
- 数据具备典型网络特性：非平稳、多尺度季节性（日/周）、异常波动等。

### ⚙️ 实验设置
- **预测目标**：对未来多个时间步长（horizon）的流量值进行直接预测（direct forecasting），涵盖短至长时域：
  - 预测 horizon：`[1, 6, 12, 24, 48, 96]` 小时；
- **输入序列长度（history length）**：系统搜索最优范围（如12~336小时）；
- **训练/验证/测试划分**：按时间顺序切分，避免信息泄露；
- **硬件环境**：本地及SLURM高性能计算集群（GPU加速）用于超参调优与训练。

### 📏 评估指标
- 主要指标：**sMAPE（Symmetric Mean Absolute Percentage Error）**
  - 优点：无量纲，适用于跨量级时间序列比较；
- 辅助分析：**MAE、RMSE、sMAPE分布统计（均值、中位数、标准差）**；
- 特别关注：**所有时间序列上的性能分布差异**，而非单一平均值。

### 🆚 基线方法对比
| 模型 | 类型 | 说明 |
|------|------|------|
| **SARIMAX** | 统计模型 | 含外生变量的季节性自回归模型，作为传统方法代表 |
| **LSTM** | 深度学习 | 单层LSTM + Dropout + Dense输出，作为基础深度模型 |
| **NT-GAT** | 图神经网络 | GAT + 双层LSTM，本文提出的空间-时间联合模型 |
| **CALF / Cluster-CALF** | 多模态LLM | 跨模态微调+聚类增强的大语言模型架构，本文核心创新 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 模型 | 最佳 horizon (1h) sMAPE | 分布标准差 |
|------|--------------------------|------------|
| **LSTM** | **56.26%** | 42.73 |
| **NT-GAT** | ~58%（略差） | 更集中（更低离散度） |
| **CALF（无聚类）** | **33.0%**（↓41.31% vs LSTM） | ↓29% |
| **Cluster-CALF（7簇）** | **32.30%** | ↓3.41%（相对CALF） |

> 注：Cluster-CALF 在 horizon=6h 达到最大增益（mean sMAPE下降4.3%）

### 🔁 与基线方法对比结果
- **Cluster-CALF 显著优于所有其他模型**：
  - 相比LSTM，**最佳sMAPE降低41.31%**；
  - 相比NT-GAT，不仅绝对误差更低，且预测更稳定（分布更紧凑）；
- **NT-GAT虽平均表现不如LSTM**，但在以下方面表现优异：
  - 不同预测horizon间性能波动小；
  - 时间序列间的预测方差更小 → 表明其能有效利用图结构稳定预测行为。

### 🔍 消融实验结果
- **Hop数选择（NT-GAT）**：
  - 最优k-hop为2~5，过大引入噪声，过小无法捕获远距离依赖；
- **损失函数对比**：
  - **Huber Loss** 明显优于MSE/MAE，尤其在含异常值的数据中更稳健；
- **聚类数量影响（Cluster-CALF）**：
  - 聚类数为6~8时性能最佳；
  - **7个Spearman聚类达到最优平衡**，优于不聚类或其他聚类方式；
- **相关性度量选择**：
  - **Spearman > Pearson**：因对非线性、非正态、异常值更鲁棒；
  - 未采用DTW/SBC等时序感知聚类是出于计算成本考虑（未来方向）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM-based模型（Cluster-CALF）在整体预测性能上全面领先**：
   - 充分证明了**大型语言模型在时间序列预测中的可迁移性和强大表达能力**；
   - 通过跨模态对齐机制成功弥合文本token与数值时间序列之间的语义鸿沟。

2. **聚类预处理显著提升LLM模型效果**：
   - 揭示了现实MTS中“**异质相关性**”的重要性——并非所有时间序列都应被同等对待；
   - **基于Spearman的相关性聚类能有效识别冗余信息，减轻多重共线性问题**。

3. **NT-GAT虽未胜出，但展现出独特优势**：
   - 在**预测稳定性与方差控制方面优于LSTM**；
   - 适用于需要**低波动、一致性强的预测场景**（如网络容量规划）。

4. **评估需超越“平均指标”**：
   - 论文强调应同时考察**性能分布（如median vs mean sMAPE）**；
   - LSTM存在大量高误差outliers（mean远高于median），而Cluster-CALF分布更集中。

### ⚠️ 方法的局限性
- **Cluster-CALF训练开销大**：尽管使用LoRA参数高效微调，仍依赖GPU资源；
- **聚类方法简化**：当前仅使用静态Spearman相关性，未考虑动态时间对齐（如DTW）；
- **通用性待验证**：目前仅在一个骨干网数据集上验证，是否适用于边缘网络或无线场景尚不确定；
- **解释性不足**：LLM黑箱特性导致难以解释具体预测逻辑。

### 🔮 未来工作方向
1. 探索**动态聚类技术**（如滑动窗口DTW聚类）以适应时变相关性；
2. 研究**轻量化LLM蒸馏方案**，提升部署可行性；
3. 扩展至**多指标联合预测**（带宽、延迟、丢包率等）；
4. 结合**因果推断与图学习**，增强模型可解释性；
5. 探索**zero-shot迁移能力**：能否将在一个网络上学到的模式迁移到新网络？

--- 

> 💡 **总结一句话**：  
> 本论文系统比较了多种深度学习模型在网络MTS预测中的表现，提出并验证了 **Cluster-CALF** 这一融合**LLM+聚类+跨模态对齐**的新范式，在真实网络数据上实现了**最高精度与最强泛化能力**，为智能网络管理提供了新的AI驱动解决方案。

</details>

---

### 15. [FlexRec: Adapting LLM-based Recommenders for Flexible Needs via Reinforcement Learning](https://arxiv.org/abs/2603.11901)

**Authors**: Yijun Pan, Weikang Qiu, Qiyao Ma, Mingxuan Ju, Tong Zhao, Neil Shah, Rex Ying  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.11901v1  

#### Abstract
Modern recommender systems must adapt to dynamic, need-specific objectives for diverse recommendation scenarios, yet most traditional recommenders are optimized for a single static target and struggle to reconfigure behavior on demand. Recent advances in reinforcement-learning-based post-training ha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FlexRec: Adapting LLM-based Recommenders for Flexible Needs via Reinforcement Learning**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**

现代推荐系统通常被优化为单一静态目标（如点击率或购买转化），难以适应动态变化的用户意图和业务需求。例如，用户可能在不同场景下希望“最大化兴趣”、“探索小众内容”或“发现热门趋势”，而传统模型无法灵活切换行为。

此外，基于大语言模型（LLM）的推荐器虽具备强大的指令跟随能力，但在**推荐任务上的对齐不足**，且现有的强化学习方法（如GRPO）存在以下两个核心挑战：

1. **粗粒度信用分配（Coarse Credit Assignment）**：序列级奖励（sequence-level rewards）将整个排序列表视为一个整体，无法区分每个物品放置决策的优劣。
2. **稀疏且含噪的交互反馈（Sparse and Noisy Feedback）**：真实场景中大多数用户-物品交互未被观测，依赖代理模型（critic）预测缺失奖励时会引入误差，导致训练不稳定。

---

### **提出了什么新方法或新思路**

本文提出 **FlexRec**，一种用于后训练（post-training）LLM 推荐器的强化学习框架，通过以下两项关键技术解决上述问题：

#### ✅ **(1) Swap-based Item-level Reward（基于交换的逐项奖励）**

- 在自回归排序过程中，针对每一个位置 $k$ 上的物品 $a_k$，通过**反事实交换操作**（counterfactual swap）评估其边际贡献。
- 具体做法是：将 $a_k$ 与后续任意位置 $j > k$ 的候选物品进行交换，计算该操作对整体排序质量（如NDCG）的影响。
- 定义 item-level 改进量为：
  $$
  \Delta_k(y;x) = \mathbb{E}_{j \sim \text{Unif}(k+1:K)} [R_n(y^{(k\leftrightarrow j)};x) - R_n(y;x)]
  $$
- 最终得到的 item-level 奖励具有：
  - **因果性（Causality）**：仅考虑剩余候选池中的比较；
  - **可比性（Comparability）**：跨不同前缀和上下文仍可归一化；
  - **细粒度监督信号**：实现更高效的信用分配。

#### ✅ **(2) Uncertainty-aware GRPO（不确定性感知的GRPO更新）**

- 引入一个神经网络 **critic** 同时预测：
  - 用户-物品交互奖励的均值 $\hat{r}(a_k;x)$
  - 预测的不确定性（方差）$\text{Var}[r(a_k;x)]$
- 利用方差估计来加权优势函数（advantage），降低高不确定性样本的更新权重：
  $$
  \tilde{A}_i = \frac{1}{u_i + \epsilon} A_i
  $$
  其中 $u_i$ 是序列总奖励的累积方差。
- 这种机制有效缓解了因 critic 错误估计而导致的策略崩溃问题。

---

### **相比现有方法的优势**

| 维度 | FlexRec | 现有方法（如Rec-R1, Rank-GRPO） |
|------|--------|-------------------------------|
| 奖励粒度 | **逐项级（item-level）**，支持精细信用分配 | 序列级或排名级，信用分配粗糙 |
| 可比性 | 奖励设计满足因果性和可比性 | Rank-GRPO 中奖励不可跨 rollout 比较 |
| 对噪声鲁棒性 | 显式建模 reward 不确定性并动态降权 | 忽略预测置信度，易受错误信号误导 |
| 泛化能力 | 单一模型可泛化至未见需求 | 多需多模型或零样本表现弱 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**

| 数据集 | 类型 | 特点 |
|-------|-----|------|
| **KuaiRec** | 短视频推荐 | 全观测、密集交互，作者人为采样10%模拟稀疏反馈 |
| **MovieLens-1M (ML-1M)** | 电影评分推荐 | 显式评分（1–5星），长期偏好建模基准 |
| **ESCI (Amazon Product Search)** | 商品搜索排序 | 查询驱动的产品相关性标注（Exact/Substitute/Complement/Irrelevant） |

> 所有数据集均构造了**多样化的需求指令集**以支持多目标评估。

---

### **实验设置和评估指标**

#### **需求定义（Need Specification）**

构建三种典型推荐目标作为指令输入：

| 需求 | 描述 |
|------|------|
| **Maximizing Interest** | 根据历史行为预测最可能感兴趣的项目（如观看比例最高） |
| **Explore New Topics (Niche Discovery)** | 推荐符合兴趣但主题新颖的内容（未出现在近期历史中） |
| **Trend Promotion** | 平衡个性化与短期流行度，优先推荐近期高热度项目 |

#### **评估指标**

- **NDCG@K** ($K=5,10,30$): 衡量排序质量
- **Recall@5**: 覆盖正例的能力
- **MRR@5**: 排名首位的相关性强度

---

### **基线方法对比**

| 类别 | 方法 |
|------|------|
| **传统重排器** | BERT4Rec, STAR |
| **零样本LLM** | GPT-4o, Qwen2.5-3B-Instruct |
| **微调/后训练LLM** | TALLRec (SFT), Rec-R1 (GRPO), ConvRec-R1 / Rank-GRPO |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **单需求性能提升（Maximizing Interest）**

来自 **Table 1**（KuaiRec & ML-1M）:

| 模型 | NDCG@5 ↑ | Recall@5 ↑ | 相对提升 |
|------|----------|------------|---------|
| **FlexRec (ours)** | **0.597** | **0.335** | — |
| Qwen2.5-3B + TALLRec | 0.507 | 0.264 | — |
| → **相对增益** | **+59.2%** | **+109.4%** | ⬆️ |

> 在 MovieLens 上也取得显著领先（NDCG@5 +23.7%）。

---

#### 🔹 **跨需求零样本泛化能力（Generalization）**

来自 **Table 2 & H (Table 6)**：

| 测试需求 | FlexRec Recall@5 | 最佳基线 | 增益 |
|--------|------------------|-----------|------|
| Explore New Topics | **0.165** | 0.147 (Rec-R1) | **+17.9%** |
| Trend Promotion | **0.269** | 0.244 (TALLRec) | **+24.1%** |

> 表明 FlexRec 学到了通用的排序原则，而非过拟合单一目标。

---

#### 🔹 **联合训练下的通用推荐器表现**

- **Figure 2** 显示：一个在所有需求上联合训练的 FlexRec 模型，在各个独立需求下均保持高性能。
- 支持“**一个模型，多种用途**”范式，适合作为统一的 LLM 推荐引擎。

---

#### 🔹 **非序列任务有效性验证（Product Search）**

来自 **Table 3**（ESCI 数据集）:

| 模型 | NDCG@5 | Recall@5 |
|------|--------|----------|
| GPT-4o | 0.502 | 0.647 |
| **FlexRec (ours)** | **0.528** | **0.678** |
| → 增益 | +17.6% | +15.9% |

> 证明 FlexRec 不仅适用于序列推荐，也能有效处理查询驱动的静态排序任务。

---

### **消融实验结果（Ablation Studies）**

#### ✅ **Table 7：不同奖励形式对比（KuaiRec, Maximizing Interest）**

| Reward Formulation | NDCG@5 |
|--------------------|--------|
| Independent contribution | 0.461 |
| Non-causal swap | 0.607 |
| **Causal swap (ours)** | **0.621** |

> 因果性约束（仅与后续未选物品交换）带来额外增益，说明设计合理性。

---

#### ✅ **Table 8：不同reward来源与uncertainty机制效果**

| 方法 | NDCG@5 |
|------|--------|
| User-KNN CF | 0.410 |
| Item-KNN CF | 0.417 |
| Raw critic (no uncertainty) | 0.566 |
| **FlexRec (w/ uncertainty-aware)** | **0.595** |

> 显示：
> - Learned critic > CF heuristic
> - 加入 uncertainty-aware weighting 再提升近 3 个百分点

---

## 4. **关键结论和发现**

### **主要发现**

1. ✅ **细粒度 item-level 奖励显著优于序列级奖励**  
   - Swap-based 设计提供了更丰富、更准确的训练信号，加速收敛并提升最终性能。
   - 因果性保障了 credit assignment 的正确性。

2. ✅ **显式建模 reward 不确定性可大幅提升训练稳定性**  
   - 尤其在稀疏反馈场景下，避免 critic 的错误估计误导策略更新。

3. ✅ **FlexRec 具备强大泛化能力**  
   - 即使只在一个需求上训练，也能在其他需求上实现优异的 zero-shot transfer。
   - 支持构建**通用型 LLM 推荐器**，通过 prompt 动态切换行为模式。

4. ✅ **生成式推理过程更具可解释性**  
   - 如 **Appendix G.1** 所示，模型能根据需求生成合理的思考链（Chain-of-Thought），例如：
     - “这个用户喜欢美妆，但没看过育儿类视频 → 可尝试推荐相关 niche 内容”
     - “当前最火的是视频 #1429，播放量达6549次 → 应优先展示”

---

### **方法的局限性**

1. ❗ **封闭集合假设（Closed-set reranking）**  
   - 当前工作聚焦于固定候选池内的重排序，未涉及检索阶段（retrieval）。
   - 实际系统中 item space 是开放且不断演化的。

2. ❗ **标签来源于已有信号**  
   - 如 watch ratio、ratings 等，并非真正意义上的“新需求”标注。
   - 缺乏真实人类对“探索性”或“趋势性”的主观判断。

3. ❗ **计算开销增加**  
   - Item-level reward 需要 $O(K^2)$ 次反事实评估，在大规模候选集中可能成为瓶颈。

---

### **未来工作方向**

1. ➡️ **扩展到 retrieval-augmented 推荐流程**  
   - 结合 dense retrieval 与 FlexRec reranker，形成端到端生成式推荐 pipeline。

2. ➡️ **引入人类反馈（RLHF）或模拟用户（Simulated User）**  
   - 构建更真实的动态需求环境，支持在线持续适应。

3. ➡️ **轻量化 item-level reward 计算**  
   - 采用采样策略减少 swap 数量，或设计近似算法降低复杂度。

4. ➡️ **多模态推荐场景延伸**  
   - 将文本描述扩展至图像、音频等多模态输入，增强 LLM 的感知能力。

---

> **总结一句话**：  
> **FlexRec 通过因果感知的 item-level 强化学习 + 不确定性引导的策略优化，首次实现了 LLM 在多变推荐需求下的高效、稳定、可泛化的动态对齐，为构建“全能型”生成式推荐系统提供了坚实路径。**

</details>

---

### 16. [Reversible Lifelong Model Editing via Semantic Routing-Based LoRA](https://arxiv.org/abs/2603.11239)

**Authors**: Haihua Luo, Xuming Ran, Tommi K\"arkk\"ainen, Zhonghua Chen, Jiangrong Shen, Qi Xu, Fengyu Cong  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.11239v1  

#### Abstract
The dynamic evolution of real-world necessitates model editing within Large Language Models. While existing methods explore modular isolation or parameter-efficient strategies, they still suffer from semantic drift or knowledge forgetting due to continual updating. To address these challenges, we pr...

---

### 17. [TimeSqueeze: Dynamic Patching for Efficient Time Series Forecasting](https://arxiv.org/abs/2603.11352)

**Authors**: Sravan Kumar Ankireddy, Nikita Seleznev, Nam H. Nguyen, Yulun Wu, Senthil Kumar, Furong Huang, C. Bayan Bruss  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.11352v1  

#### Abstract
Transformer-based time series foundation models face a fundamental trade-off in choice of tokenization: point-wise embeddings preserve temporal fidelity but scale poorly with sequence length, whereas fixed-length patching improves efficiency by imposing uniform boundaries that may disrupt natural tr...

---

### 18. [Learning Transferable Sensor Models via Language-Informed Pretraining](https://arxiv.org/abs/2603.11950)

**Authors**: Yuliang Chen, Arvind Pillai, Yu Yvonne Wu, Tess Z. Griffin, Lisa Marsch, Michael V. Heinz, Nicholas C. Jacobson, Andrew Campbell  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.11950v1  

#### Abstract
Modern sensing systems generate large volumes of unlabeled multivariate time-series data. This abundance of unlabeled data makes self-supervised learning (SSL) a natural approach for learning transferable representations. However, most existing approaches are optimized for reconstruction or forecast...

---

### 19. [AGMARL-DKS: An Adaptive Graph-Enhanced Multi-Agent Reinforcement Learning for Dynamic Kubernetes Scheduling](https://arxiv.org/abs/2603.12031)

**Authors**: Hamed Hamzeh  
**Category**: cs.DC  
**Published**: 2026-03-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.12031v1  

#### Abstract
State-of-the-art cloud-native applications require intelligent schedulers that can effectively balance system stability, resource utilisation, and associated costs. While Kubernetes provides feasibility-based placement by default, recent research efforts have explored the use of reinforcement learni...

---

### 20. [High-resolution weather-guided surrogate modeling for data-efficient cross-location building energy prediction](https://arxiv.org/abs/2603.11121)

**Authors**: Piragash Manmatharasan, Girma Bitsuamlak, Katarina Grolinger  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.11121v1  

#### Abstract
Building design optimization often depends on physics-based simulation tools such as EnergyPlus, which, although accurate, are computationally expensive and slow. Surrogate models provide a faster alternative, yet most are location-specific, and even weather-informed variants require simulations fro...

---

### 21. [Relaxed Efficient Acquisition of Context and Temporal Features](https://arxiv.org/abs/2603.11370)

**Authors**: Yunni Qu (The University of North Carolina at Chapel Hill), Dzung Dinh (The University of North Carolina at Chapel Hill), Grant King (University of Michigan), Whitney Ringwald (University of Minnisota Twin Cities), Bing Cai Kok (The University of North Carolina at Chapel Hill), Kathleen Gates (The University of North Carolina at Chapel Hill), Aiden Wright (University of Michigan), Junier Oliva (The University of North Carolina at Chapel Hill)  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.11370v1  

#### Abstract
In many biomedical applications, measurements are not freely available at inference time: each laboratory test, imaging modality, or assessment incurs financial cost, time burden, or patient risk. Longitudinal active feature acquisition (LAFA) seeks to optimize predictive performance under such cons...

---

### 22. [The Unlearning Mirage: A Dynamic Framework for Evaluating LLM Unlearning](https://arxiv.org/abs/2603.11266)

**Authors**: Raj Sanjay Shah, Jing Huang, Keerthiram Murugesan, Nathalie Baracaldo, Diyi Yang  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.11266v1  

#### Abstract
Unlearning in Large Language Models (LLMs) aims to enhance safety, mitigate biases, and comply with legal mandates, such as the right to be forgotten. However, existing unlearning methods are brittle: minor query modifications, such as multi-hop reasoning and entity aliasing, can recover supposedly ...

---

### 23. [LLMs can construct powerful representations and streamline sample-efficient supervised learning](https://arxiv.org/abs/2603.11679)

**Authors**: Ilker Demirel, Larry Shi, Zeshan Hussain, David Sontag  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.11679v1  

#### Abstract
As real-world datasets become increasingly complex and heterogeneous, supervised learning is often bottlenecked by input representation design. Modeling multimodal data for downstream tasks, such as time-series, free text, and structured records, often requires non-trivial domain-specific engineerin...

---

### 24. [DocSage: An Information Structuring Agent for Multi-Doc Multi-Entity Question Answering](https://arxiv.org/abs/2603.11798)

**Authors**: Teng Lin, Yizhang Zhu, Zhengxuan Zhang, Yuyu Luo, Nan Tang  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.11798v1  

#### Abstract
Multi-document Multi-entity Question Answering inherently demands models to track implicit logic between multiple entities across scattered documents. However, existing Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) frameworks suffer from critical limitations: standard RAG's v...

---

### 25. [Few-for-Many Personalized Federated Learning](https://arxiv.org/abs/2603.11992)

**Authors**: Ping Guo, Tiantian Zhang, Xi Lin, Xiang Li, Zhi-Ri Tang, Qingfu Zhang  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.11992v1  

#### Abstract
Personalized Federated Learning (PFL) aims to train customized models for clients with highly heterogeneous data distributions while preserving data privacy. Existing approaches often rely on heuristics like clustering or model interpolation, which lack principled mechanisms for balancing heterogene...

---

### 26. [DeReason: A Difficulty-Aware Curriculum Improves Decoupled SFT-then-RL Training for General Reasoning](https://arxiv.org/abs/2603.11193)

**Authors**: Hanxu Hu, Yuxuan Wang, Maggie Huan, Jannis Vamvas, Yinya Huang, Zhijiang Guo, Rico Sennrich  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.11193v1  

#### Abstract
Reinforcement learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for eliciting reasoning capabilities in large language models, particularly in mathematics and coding. While recent efforts have extended this paradigm to broader general scientific (STEM) domains, the complex i...

---

### 27. [Beyond BFS: A Comparative Study of Rooted Spanning Tree Algorithms on GPUs](https://arxiv.org/abs/2603.11645)

**Authors**: Abhijeet Sahu, Srikar Vilas Donur  
**Category**: cs.DC  
**Published**: 2026-03-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.11645v1  

#### Abstract
Rooted spanning trees (RSTs) are a core primitive in parallel graph analytics, underpinning algorithms such as biconnected components and planarity testing. On GPUs, RST construction has traditionally relied on breadth-first search (BFS) due to its simplicity and work efficiency. However, BFS incurs...

---

### 28. [Differentiable Thermodynamic Phase-Equilibria for Machine Learning](https://arxiv.org/abs/2603.11249)

**Authors**: Karim K. Ben Hicham, Moreno Ascani, Jan G. Rittig, Alexander Mitsos  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.11249v1  

#### Abstract
Accurate prediction of phase equilibria remains a central challenge in chemical engineering. Physics-consistent machine learning methods that incorporate thermodynamic structure into neural networks have recently shown strong performance for activity-coefficient modeling. However, extending such app...

---

### 29. [Personalized Federated Learning via Gaussian Generative Modeling](https://arxiv.org/abs/2603.11620)

**Authors**: Peng Hu, Jianwei Ma  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.11620v1  

#### Abstract
Federated learning has emerged as a paradigm to train models collaboratively on inherently distributed client data while safeguarding privacy. In this context, personalized federated learning tackles the challenge of data heterogeneity by equipping each client with a dedicated model. A prevalent str...

---

### 30. [Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics](https://arxiv.org/abs/2603.12087)

**Authors**: Ming-Hong Chen, Kuan-Chen Pan, You-De Huang, Xi Liu, Ping-Chun Hsieh  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.12087v1  

#### Abstract
Cross-domain reinforcement learning (CDRL) is meant to improve the data efficiency of RL by leveraging the data samples collected from a source domain to facilitate the learning in a similar target domain. Despite its potential, cross-domain transfer in RL is known to have two fundamental and intert...

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

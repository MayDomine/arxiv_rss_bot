# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-27 16:51:41 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [ExFold: Unified Expert Folding for Training-Free MoE Prefill-Decode Acceleration](https://arxiv.org/abs/2608.24938)

**Authors**: Juntong Wu, Yifei Liu, Junyi Chen, Siqi Fan, Chaoran Feng, Minghao Li, Liujie Zhang, Weihang Chen, Li Yuan  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2608.24938v1  

#### Abstract
Mixture-of-Experts (MoE) models scale capacity for strong quality while keeping per-token compute bounded through sparse expert activation. Yet low-latency MoE serving is increasingly challenging, because it spans two inference phases with fundamentally different bottlenecks: prefill is dominated by...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ExFold: Unified Expert Folding for Training-Free MoE Prefill-Decode Acceleration 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

Mixture-of-Experts (MoE) 模型通过稀疏激活专家实现高效扩展，但在实际部署中面临两个阶段的不同瓶颈：

- **Prefill 阶段**：以 **token-wise expert computation** 为主导，计算密集。
- **Decode 阶段**：以 **batch-wise expert memory traffic** 为主导，内存密集。

现有训练无关（training-free）加速方法通常只优化其中一个阶段，且对被排除专家的贡献处理不当：
- **Token-wise sparsification**（如 Top-K 减少）：直接丢弃未选专家，导致质量下降。
- **Expert-set consolidation**（如 REAP、SERE）：静态合并或动态重路由，但缺乏对输出幅度差异的校准。

这些方法在联合加速时误差叠加，难以兼顾两阶段性能与质量。

---

### 提出了什么新方法或新思路

作者提出 **ExFold** —— 一种统一的、无需训练的 **专家折叠（Expert Folding）框架**，用于同时加速 MoE 的 prefill 和 decode 阶段。

#### 核心思想：将专家排除视为“有损压缩”，通过“投影恢复”重建其贡献

- 将 prefill 和 decode 统一建模为一个 **受限输出近似问题（constrained output approximation）**。
- 对于被排除的专家 $E_s$，不直接丢弃，而是将其输出 $E_s(x)$ 通过一个 **标量投影器（scalar projector）** 投影到某个保留专家 $E_t$ 上：
  $$
  E_s(x) \approx s_{s\to t} \cdot E_t(x)
  $$
  其中标量 $s_{s\to t}$ 在离线校准阶段学习得到。
- 被折叠的贡献通过修改 **router weights** 实现，无需改变模型参数或结构。

#### 两大阶段共享同一机制，仅选择策略不同：

| 阶段     | 保留专家选择方式                     | 折叠机制               |
|----------|--------------------------------------|------------------------|
| Prefill  | 每个 token 选 Top-K 主导专家         | Token-level folding    |
| Decode   | 整个 batch 选最多 D 个活跃专家集合     | Batch-level folding    |

两者共用相同的 **scalar projector matrix** 和 **loss matrix** 来决定如何折叠。

---

### 相比现有方法的优势

| 方面 | ExFold | 现有方法（如 MC-MoE, REAP, SERE） |
|------|-------|-------------------------------|
| **统一性** | ✅ 同时优化 prefill 和 decode | ❌ 通常只针对单一阶段 |
| **质量保持** | ✅ 显式恢复被排除专家贡献 | ❌ 丢弃或粗略替代，误差大 |
| **无需训练** | ✅ 完全基于离线校准 | ✅ 多数也是 |
| **兼容性** | ✅ 只修改 router metadata，可插拔集成 | ✅ 类似 |
| **灵活性** | ✅ 动态适应不同 token/batch 的预算 | ❌ 静态压缩无法自适应 |

> 💡 **关键洞察**：许多专家输出方向相近但幅值不同 → 只需一个标量即可有效对齐。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

综合多个领域 benchmark 进行评估：

| 数据集 | 任务类型 | 主要指标 |
|--------|---------|---------|
| **MATH500** | 数学推理 | Accuracy |
| **AIME24** | 竞赛数学 | Mean symbolic-equivalence accuracy |
| **IFEval / IFBench** | 指令遵循 | Prompt-level accuracy |
| **GPQA-Diamond** | 高阶知识推理 | Accuracy |
| **LiveCodeBench v5 / HumanEval+** | 代码生成 | Pass@8, Pass@1 |
| **MMLU-Pro** | 多任务知识理解 | Category-macro accuracy |

> 所有评估均使用 OpenCompass 兼容 pipeline。

---

### 实验设置和评估指标

#### 模型
- **主模型**：`Qwen3-30B-A3B`（48层，128专家，Top-8）
- **其他验证模型**：
  - `GLM-4.5-Air`
  - `DeepSeek-V2-Lite`
  - `DeepSeek-V4-Flash`（284B 参数，256专家）
  - `Qwen3.5-35B-A3B`

#### 硬件与系统
- 平台：**vLLM**
- GPU：NVIDIA H800 80GB
- 实现：自定义 **Triton CUDA kernel**，轻量级集成

#### 评估指标
| 指标 | 含义 | 测量场景 |
|------|------|--------|
| **TTFT** | Time to First Token | Prefill 阶段延迟 |
| **TPOT** | Time Per Output Token | Decode 阶段延迟 |
| **Offline Throughput** | 输出 token/s | 高吞吐批量生成 |
| **Average Quality** | 多任务平均得分 | 模型能力保留程度 |

#### 加速配置命名
- `P4`: Prefill 每 token 保留 4 个专家
- `D64`: Decode 每 batch 最多激活 64 个专家
- `P4+D64`: 联合加速

---

### 基线方法对比

| 类别 | 方法 | 特点 |
|------|------|------|
| **Token-wise sparsification** | Top-K reduction, MC-MoE, MoDES | 减少每个 token 的专家数，利于 prefill |
| **Expert-set consolidation** | REAP (pruning), SERE (re-routing) | 缩小 batch 激活专家集，利于 decode |
| **联合压缩** | All Top-K=4, MC-MoE-K4 | 直接应用单阶段方法到双阶段 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Qwen3 和 DeepSeek-V4）

| 指标 | 结果 |
|------|------|
| **TTFT 加速比** | 最高达 **1.41×** |
| **TPOT 加速比** | 最高达 **2.45×** |
| **离线吞吐提升** | 达 **1.20× ~ 1.29×** |
| **质量保留率** | 平均保留原始模型 **~99%** 的能力 |

> 在 `ExFold P4+D64` 设置下，实现显著加速的同时几乎无损。

---

### 与基线方法的对比结果（Qwen3-30B-A3B）

#### ✅ Prefill Only (`P4`)
| 方法 | Avg. Quality | 相比 Baseline |
|------|-------------|---------------|
| Baseline (Top-8) | 68.57 | — |
| Direct Top-4 | 66.64 | ↓1.93 |
| MC-MoE-P4 | 65.44 | ↓3.13 |
| **ExFold-P4** | **69.26** | ↑**+0.69** ✅ |

> ExFold 不仅没降质，反而轻微提升！

#### ✅ Decode Only (`D64`)
| 方法 | Avg. Quality | 相比 Baseline |
|------|-------------|---------------|
| REAP-D64 | 61.50 | ↓7.07 |
| SERE-K4 | 64.15 | ↓4.42 |
| **ExFold-D64** | **68.75** | ↑**+0.18** ✅ |

> 即使 decode 池缩小一半（128→64），仍接近原模型表现。

#### ✅ Joint Acceleration (`P4+D64`)
| 方法 | Avg. Quality | 相比 Baseline |
|------|-------------|---------------|
| All TopK=4 | 56.17 | ↓12.40 |
| MC-MoE-K4 | 57.30 | ↓11.27 |
| MoDES-K4 | 65.72 | ↓2.85 |
| **ExFold P4+D64** | **68.50** | ↑**-0.07** ✅ |

> 其他方法严重掉点，而 ExFold 几乎无损！

---

### 消融实验结果

#### 🔍 投影器形式对比（Table 3）

| 投影器类型 | 存储开销 | 是否可用 fused kernel | 下游任务平均得分 |
|-----------|----------|------------------------|------------------|
| Global scalar | 1 | ✅ 是 | 65.60 |
| Layer scalar | L | ✅ 是 | 64.70 |
| **Pairwise scalar** | P | ✅ 是 | **66.94** ✅ |
| Diagonal | PH | ❌ 否（需 unfused） | 65.87 |
| Low-rank (R16) | ~4.87 GiB | ❌ 否 | 66.60 |

> **Pairwise scalar** 在精度、效率、兼容性上达到最佳平衡。

#### 🔍 校准数据来源影响（Table 4）

| 校准数据来源 | 平均得分 |
|------------|---------|
| MATH | 71.10 |
| CODE | 71.62 |
| PRETRAIN | 72.75 |
| **Mixed (default)** | **73.00** ✅ |

> 混合领域校准效果最好，说明专家几何具有通用性，**无需任务特定校准**。

#### 🔍 保留专家选择策略（Table 5）

| 策略 | Prefill 得分 | Decode 得分 |
|------|------------|-----------|
| 仅按 router score $s_i$ 排序 | 79.36 | 78.55 |
| 仅按输出范数 $h_e$ 排序 | 74.42 | 78.74 |
| **按 $s_i \times h_e$ 排序（本文）** | **80.11** ✅ | **79.74** ✅ |

> 结合路由权重与输出强度的选择更优。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **MoE 两阶段瓶颈本质统一**：尽管 prefill 和 decode 资源约束不同，但“被排除专家贡献”的形式一致，可统一建模为输出近似问题。
2. ✅ **专家冗余主要体现在幅值而非方向**：大量专家输出方向相似，仅幅值差异显著 → **一个标量即可有效对齐**。
3. ✅ **显式恢复优于隐式丢弃**：ExFold 通过 scalar projector 显式重建被排除专家贡献，在同等预算下显著优于直接剪枝或跳过。
4. ✅ **统一框架避免误差累积**：ExFold 使用相同 projector 处理两阶段，避免了分别优化带来的目标不一致和误差叠加。
5. ✅ **即插即用、高效实用**：仅需一次离线校准 + 修改 router metadata，可在 vLLM 中无缝集成，带来显著加速。

---

### 方法的局限性

1. 🚫 **依赖专家间方向对齐假设**：若某些模型专家高度异构、方向分散，则 scalar projector 效果可能下降。
2. ⚠️ **校准数据需具代表性**：虽然无需任务标签，但仍需覆盖多样输入分布，否则 projector 泛化性受影响。
3. ⚠️ **极端压缩仍有损失**：当 `D` 极小时（如 D=32），即使 ExFold 也难以完全保持质量（见 Table 6）。
4. 🚫 **未解决专家调度开销**：虽减少内存访问，但 expert dispatch 和 aggregation 开销依然存在。

---

### 未来工作方向

1. 🔮 **探索更复杂的 projector 形式**：如 low-rank 或 conditional scalar，在精度要求极高时使用。
2. 🔁 **在线自适应 folding**：根据输入难度动态调整保留专家数量与 folding 强度。
3. 🧩 **与其他优化正交结合**：与 KV Cache 压缩（如 C2KV）、attention 优化等联合使用，进一步提升端到端性能。
4. 🌐 **推广至更多架构**：验证在 dense 模型、非 Transformer 架构中的可行性。
5. 📦 **自动化 tuning 工具链**：构建自动搜索最优 `K`, `D`, `calibration set` 的工具包。

---

## 总结

📌 **ExFold 是首个真正意义上统一加速 MoE prefill 与 decode 的 training-free 框架**。它通过 **专家折叠（Expert Folding）** 的新颖视角，将被排除专家的贡献显式投影回保留专家，实现了：

- ✅ **高达 1.41× TTFT 和 2.45× TPOT 加速**
- ✅ **平均保留 ~99% 原始模型质量**
- ✅ **即插即用、无需微调、兼容性强**

其成功源于对 **专家输出几何特性** 的深刻洞察：**方向对齐 + 幅值可调 → 标量投影足够**。该工作为 MoE 高效推理提供了新的设计范式。

🔗 **开源地址**：[https://github.com/Time-Rune/ExFold-MoE](https://github.com/Time-Rune/ExFold-MoE)

</details>

---

### 2. [Hierarchical Shared Memory-Aware Optimization for TRSM on GPU Platforms](https://arxiv.org/abs/2608.25469)

**Authors**: Xinzhe Chen, Haowei Li, Lijuan Hu, Wenjing Ma, Fangfang Liu  
**Category**: cs.DC  
**Published**: 2026-08-27  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.25469v1  

#### Abstract
Triangular Solve with Multiple Right-hand Sides (TRSM) is a fundamental BLAS Level-3 operation that underpins LU/Cholesky decomposition, sparse direct solvers, and matrix inversion. In the left-side lower-triangular case studied in this paper, efficient GPU implementation remains challenging because...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Hierarchical Shared Memory-Aware Optimization for TRSM on GPU Platforms

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **TRSM**（Triangular Solve with Multiple Right-hand Sides）在 GPU 上的高效实现面临三大挑战：
- **小规模矩阵**（m, n ≤ 64）受限于共享内存（shared memory）容量，尤其是 `double complex` 类型下无法同时缓存矩阵 A 和 B，导致频繁访问全局内存。
- **大规模矩阵**中，传统方法对角块求逆需要 $O(I^2)$ 的共享内存，限制了 block size 的选择，影响 GEMM 更新阶段的计算密度和效率。
- 现有优化多为单平台设计，缺乏跨平台（如 NVIDIA A100、H800 和 Hygon DCU Z100）的性能可移植性。

### 提出的新方法与创新思路
作者提出 **HSMA-TRSM**，一个层次化的共享内存感知优化框架，其核心贡献如下：

#### （1）小规模 TRSM 优化：流水线化计算-内存重叠机制
- 设计基于 **loop unrolling** 和 **instruction reordering** 的流水线策略，在小规模场景下实现数据预取与前向替换（forward substitution）计算的重叠。
- 针对 `double complex` 类型提出 **dual thread-group seven-stage pipeline**，将共享内存划分为多个子区域，通过两个线程组协作完成分块加载与计算，突破共享内存容量瓶颈。

#### （2）大规模 TRSM 优化：对角块解耦 + 自适应分块
- 提出 **diagonal block decoupling strategy**，将传统融合内核拆解，采用纯回代方式执行对角块求逆，将其共享内存占用从 $O(I^2)$ 降低至 $O(IB)$。
- 引入 **double buffering** 机制构建异步计算流水线，隐藏列级数据加载延迟。
- 提出 **adaptive blocking algorithm**，根据矩阵规模和数据类型动态选择最优 block size。

#### （3）零运行时开销的跨平台配置选择框架
- 构建基于 **offline profiling + online lookup** 的编译期配置选择机制：
  - 离线阶段生成“输入规模 × 数据类型 → 最优 block size”的映射表；
  - 运行时仅需一次哈希查找即可确定参数，无额外性能开销。
- 实现了在 **NVIDIA A100、H800** 和 **Hygon DCU Z100** 上的高性能可移植性。

### 相比现有方法的优势
| 方面 | 传统方法（cuBLAS / rocBLAS） | HSMA-TRSM |
|------|-------------------------------|-----------|
| 小规模内存利用 | 分阶段执行，难以重叠访存与计算 | 流水线设计显著提升内存利用率 |
| `double complex` 支持 | 受限于共享内存容量，性能差 | 专用七级双线程组流水线有效缓解瓶颈 |
| 大规模 block size | rocBLAS 固定为 128，不自适应 | 动态选择 block size，适配不同硬件与问题规模 |
| 对角块内存占用 | 融合内核需 $O(I^2)$ shared memory | 解耦后降至 $O(IB)$，支持更大 block size |
| 性能可移植性 | 平台特定优化，迁移困难 | 统一框架 + 查表机制支持多平台 |

---

## 2. 核心实验方法和设置

### 实验平台
- **NVIDIA A100-SXM4-40GB**
- **NVIDIA H800 PCIe**
- **Hygon DCU Z100**

详细规格见下表：

| Platform | SM/CU | Shmem/LDS per Block | HBM | Bandwidth | FP32 Peak | Thread Group |
|---------|--------|---------------------|-----|------------|------------|---------------|
| A100 | 108 SM | 48KB | 40GB HBM2e | 1.6 TB/s | 19.5 TFLOPS | 32 (warp) |
| H800 | 132 SM | 48KB | 80GB HBM3 | 2.0 TB/s | 51 TFLOPS | 32 (warp) |
| DCU Z100 | 64 CU | 64KB | 16GB HBM2 | 1.0 TB/s | 13.9 TFLOPS | 64 (wavefront) |

### 数据集与测试配置
- 使用 **square matrices**（m = n），范围从 64 到 16384。
- 测试四种数据类型：
  - `float`, `double`
  - `float complex`, `double complex`
- 输入矩阵随机初始化。
- 每个测试点运行 10 次，取后 9 次平均值（首次用于预热）。

### 评估指标
- **GFLOPS**：实测性能
- **Speedup**：相对于 cuBLAS / rocBLAS 的加速比
- **Numerical correctness**：以 $\frac{\|X_{\text{ours}} - X_{\text{cuBLAS}}\|_F}{\|X_{\text{cuBLAS}}\|_F}$ 衡量数值一致性

### 基线方法对比
- **cuBLAS v13.2.1**（NVIDIA 平台）
- **rocBLAS v5.1**（DCU 平台）
- **MAGMA**（A100 上作为补充对比）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 小规模 TRSM（m, n ≤ 64）
| 数据类型 | 平台 | m=64 性能（GFLOPS） | 相对提升 |
|--------|-------|------------------|----------|
| `float` | DCU | 4.64 | +18.4% vs rocBLAS |
| `float` | A100 | 5.75 | +29.2% vs cuBLAS |
| `float complex` | DCU | 11.61 | +86.3% vs rocBLAS |
| `double complex` | DCU | 3.53 | **+96.1%** vs rocBLAS |
| `double complex` | A100 | 15.88 | **+14.5%** vs cuBLAS |
| `double complex` | H800 | 12.47 | **+105.4%** vs cuBLAS |

> 💡 在 `double complex` 场景下达到最高 **2.05×** 加速（H800 上）。

#### 大规模 TRSM（m > 64）
| 数据类型 | 平台 | 峰值性能（GFLOPS） | 相对加速比 |
|--------|-------|------------------|-------------|
| `float` | DCU | 6010 | 1.63× vs rocBLAS |
| `double` | DCU | 3654 | **2.06×** vs rocBLAS |
| `float` | A100 | 17802 | > cuBLAS/MAGMA |
| `double complex` | A100 | 17111 | > cuBLAS/MAGMA |
| `float` | H800 | 36088 | 最高绝对吞吐 |
| `float complex` | H800 | 38654 | 最高绝对吞吐 |

> ✅ 在 m=16384 时仍保持 **1.63× ~ 2.06×** 的稳定加速。

### 与 MAGMA 的对比（A100）
| m | 数据类型 | HSMA-TRSM (GFLOPS) | MAGMA (GFLOPS) | 提升幅度 |
|----|----------|--------------------|----------------|-----------|
| 4096 | `float` | 11798 | 8837 | +33.5% |
| 4096 | `double` | 10577 | 7344 | +44.0% |
| 16384 | `double complex` | 17111 | 14179 | +20.7% |

### 消融分析（Ablation Study）
- **对角块解耦效果显著**：
  - 在 A100 上，随着矩阵增大，`Diag. inv.` 占比从 20.7%（m=2048）下降到 3.7%（m=8192），而 `GEMM update` 占比上升至 88.0%，表明大问题已由 GEMM 主导，优化收益明显。
- **自适应 blocking 更优**：
  - 固定 block size=128（如 rocBLAS）在中等规模尚可，但在大矩阵上因更新次数过多导致开销增加。
  - HSMA-TRSM 能随矩阵增长自动增大 block size（如从 256 → 512），减少 kernel launch 次数并提高 GEMM 效率。

---

## 4. 关键结论和发现

### 主要发现
1. **共享内存是小规模 TRSM 的核心瓶颈**，尤其对于宽数据类型（如 `double complex`）。传统的直接求解策略在此类边界情况下失效。
2. **解耦对角块处理路径** 是打破 $O(I^2)$ 内存约束的关键，使 block size 可扩展性大幅提升，从而启用更高效率的大型 GEMM 内核。
3. **自适应 blocking 策略优于固定 block size**，特别是在跨越性能拐点（crossover point）时能避免不必要的细粒度更新。
4. **离线索引 + 运行时查表** 的配置选择机制实现了真正的零运行时开销性能调优，适用于生产级 BLAS 库部署。
5. **HSMA-TRSM 在多种平台和数据类型上均取得显著加速**，峰值达 **2.05× over cuBLAS**, **2.06× over rocBLAS**，且在 `double complex` 小规模和 `real-type` 大规模场景下增益最强。

### 方法的局限性
- 当前框架主要面向 **dense TRSM**，未涵盖稀疏或批量（batched）场景。
- 对于某些高度优化的 vendor kernel（如 H800 上的 `double complex`），剩余优化空间较小，提升有限。
- 编译期 profiling 表格需预先构建，虽不影响运行时，但增加了部署复杂性。

### 未来工作方向
- 扩展至 **very small scales** 的 kernel fusion 优化。
- 支持 **batched TRSM** 和 **mixed-precision** 计算。
- 探索在 **sparse matrix** 或 **hierarchical matrices** 中的应用。
- 进一步集成自动调参工具链，实现全自动部署。

--- 

> 📌 **总结一句话**：  
> HSMA-TRSM 通过分层优化策略，在小规模采用共享内存流水线，在大规模实施对角块解耦与自适应分块，并结合离线调优+在线查表机制，实现了跨平台、高效率、低开销的 TRSM GPU 加速，在多种场景下达到 **2倍以上** 的性能超越。

</details>

---

### 3. [Compression Trinity: Exploring Sparsity, Quantization, and Low-Rank Approximations for LLM Compression](https://arxiv.org/abs/2608.24070)

**Authors**: Mohammad Mozaffari  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.24070v1  

#### Abstract
Prohibitive computational and environmental costs impede the scalable deployment of Large Language Models (LLMs). Traditional compression techniques (sparsity, quantization, low-rank approximations) are typically applied in isolation, and each hits an accuracy-efficiency wall. This thesis proposes t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Compression Trinity: Exploring Sparsity, Quantization, and Low-Rank Approximations for LLM Compression**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）在训练和推理阶段面临高昂的计算、内存和环境成本，限制了其可扩展部署。传统压缩技术（如 sparsity、quantization、low-rank approximations）通常**孤立应用**，导致每种方法在达到一定压缩率后遭遇“准确率-效率墙”（accuracy-efficiency wall），无法进一步提升效率而不严重损失性能。

该论文指出，单一压缩策略存在根本性局限：
- **Sparsity** 减少 FLOPs，但对 memory-bound 的推理阶段加速有限；
- **Quantization** 减少内存带宽需求，但在 compute-bound 阶段收益较小；
- 单独使用任一方法均难以兼顾不同生命周期阶段的硬件瓶颈。

### **提出的新方法与新思路**
论文提出了 **“Compression Trinity”** 框架——一种将 **sparsity、quantization 和 low-rank approximations 联合应用** 的统一范式，认为三者是互补而非替代关系。

#### **三大支柱的角色分工**：
| 技术 | 功能 |
|------|------|
| **Sparsity** | 减少计算量（降低 FLOPs） |
| **Quantization** | 减少内存带宽压力（降低参数位宽） |
| **Low-rank approximations** | 恢复因前两者造成的精度损失，作为“算法补偿机制” |

该框架贯穿 LLM 生命周期，在 **pretraining** 和 **post-training compression** 两个阶段分别设计了具体实现方案：

#### **创新方法**：
1. **MKOR**（用于优化器加速）
   - 在 second-order optimizer 中引入 block-diagonal sparsity 和 low-rank inversion 来近似 curvature。
   - 将 curvature 更新复杂度从 $O(d^3)$ 降至 $O(d^2)$，收敛速度比 KFAC 快 **1.85×**。

2. **SLoPE**（Double-Pruned Sparse Plus Lazy Low-Rank Adapter Pretraining）
   - 在 pretraining 中同时对前向和反向传播进行 N:M sparsity（如 2:4）剪枝。
   - 最后 1% 训练阶段引入 low-rank “lazy” adapters 恢复精度。
   - 实现端到端训练加速 **1.25×**，且不影响推理效率。

3. **OPTIMA**
   - 后训练压缩中通过全局最优列级二次规划重建权重，稳定静态 mask。
   - 在零样本场景下将 accuracy 提升最高达 **3.97%**。

4. **PATCH**
   - 引入可学习的混合稀疏结构（dense tiles + 2:4 sparse tiles），支持连续 sparsity ratio（0%-50%）。
   - 允许灵活权衡稀疏性与模型质量，突破固定结构限制。

5. **SLiM**（Sparse + Low-rank + Quantized Model）
   - 综合三者的一次性（one-shot）压缩方法。
   - 使用数学推导的 low-rank adapter 补偿量化与剪枝的信息损失。
   - 性能超越当前 SOTA 方法最多 **5.66%**，甚至在相同参数预算下优于未压缩 dense 模型 **0.6%**。

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **系统性** | 不再孤立看待压缩技术，而是构建多维协同框架 |
| **全生命周期覆盖** | 支持 pretraining 加速与 post-training 部署优化 |
| **硬件感知设计** | 区分 compute-bound（training/prefill）与 memory-bound（decoding）阶段，针对性施加压缩策略 |
| **高效恢复机制** | 利用 low-rank 结构以极小代价恢复精度，避免昂贵微调 |
| **实际可用性强** | 多数方法无需完整 retraining，适合工业部署 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 类型 | 数据集 |
|------|--------|
| **预训练任务** | WikiCorpus, BookCorpus, SlimPajama（用于 BERT/GPT 类模型 pretraining） |
| **下游评估任务** |  
| - **问答/常识推理** | MMLU, PIQA, ARC-Easy/Challenge, OpenBookQA, RACE |
| - **语言建模** | WikiText2（perplexity 评估） |
| - **其他** | WinoGrande, HellaSwag, GLUE（分类任务） |

### **实验设置与评估指标**
| 设置项 | 描述 |
|-------|------|
| **模型族** | OPT, LLaMA-2, LLaMA-3, Qwen-2.5, BERT-Large-Uncased, GPT2-Small |
| **压缩目标** | 达成 4x–8x 模型大小压缩比（model size reduction） |
| **评估指标** |  
| - **Accuracy** | 平均零样本任务得分（zero-shot accuracy） |
| - **Perplexity** | 在 WikiText2 上的语言建模能力 |
| - **Speedup** | 训练/推理端到端时间加速比 |
| - **Memory Reduction** | 显存占用下降倍数 |
| - **Sparsity Ratio** | 结构化（2:4）、非结构化（50%, 60%）等 |

### **基线方法对比**
| 基线方法 | 类型 | 特点 |
|---------|------|------|
| **SparseGPT**, **Wanda**, **Thanos** | 一次性剪枝（one-shot pruning） | 层级局部优化，忽略跨层依赖 |
| **GPTQ**, **OPTQ** | 逐层量化 | 基于 Hessian 近似量化误差最小化 |
| **FST (Fully Sparse Training)** | 动态稀疏训练 | 使用 transposable weights，需最终 dense fine-tuning |
| **LAMB**, **SGD**, **KFAC**, **KAISA**, **Eva** | Optimizer 对比 | 测试 MKOR 在收敛速度上的优势 |
| **Bi-Mask**, **ProxSparse** | N:M 稀疏训练 | 仅应用于 MLP 层，不覆盖 Attention |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 方法 | 场景 | 性能提升 |
|------|------|----------|
| **MKOR** | BERT-Large pretraining | 收敛速度比 KFAC 快 **1.85×**，端到端训练提速 **2.57×** |
| **SLoPE** | LLM pretraining | 训练加速 **1.25×**，推理加速最高 **2.1×**（取决于 adapter rank） |
| | | 内存减少 **1.8×–2.3×**（训练），**2.0×–2.5×**（推理） |
| **OPTIMA** | 后训练压缩 | 在 60% unstructured sparsity 下，平均 accuracy 提升 **3.97%** |
| **PATCH** | 可学习稀疏结构 | 实现 **1.38× 推理速度提升**，支持动态 sparsity ratio 控制 |
| **SLiM** | 一体化压缩 | 相比 SOTA 提升 accuracy 最高 **5.66%**，同等参数下优于 dense 模型 **0.6%** |

### **与基线方法的对比结果**
#### ✅ **SLoPE vs FST**（Table 4.1 & 4.2）
| 指标 | SLoPE | FST |
|------|-------|-----|
| 是否修剪 Attention 层 | ✅ 是 | ❌ 否 |
| 是否使用动态权重 | ❌ 静态 mask | ✅ 是（额外开销） |
| 是否需要 dense fine-tuning | ❌ 否 | ✅ 是（~17% 预训练） |
| 推理是否加速 | ✅ 是 | ❌ 否（最终为 dense 模型） |
| 端到端训练速度 | **快 1.25×** | 无显著加速 |

> → SLoPE 在训练和推理上均实现持续加速，而 FST 因最后密集微调失去推理优势。

#### ✅ **Multi-Pillar vs Single-Pillar Compression**（Table 1.2）
| 方法 | 压缩比 | 平均 accuracy |
|------|--------|--------------|
| Dense Baseline (FP16) | 1x | 54.61% |
| 2-bit Quantization | 8x | 31.81% |
| 87.5% Unstructured Sparsity | 8x | 31.24% |
| **4-bit + 2:4 Sparsity** | 8x | **47.97%** |
| **4-bit + 50% Unstructured Sparsity** | 8x | **52.38%** |

> → 单一压缩柱石导致性能崩溃；联合使用可恢复 **>95%** 的原始性能。

#### ✅ **SLiM vs 其他压缩方法**（Table E.10–E.11）
在 8x 压缩比下：
- **纯 2-bit 量化**：perplexity 高达 44.9（LLaMA-2-13B）
- **4-bit + 2:4 sparsity**：perplexity 降至 54.9
- **SLiM（三者结合）**：perplexity 进一步降至 **48.0**，accuracy 提升明显

> → **sparsity + quantization + low-rank > quantization-only**

### **消融实验结果**
- **低秩适配器的作用**：移除 lazy adapter 导致 SLoPE 在最后阶段 accuracy 下降约 2–4%，验证其恢复能力。
- **tile mask 学习的有效性**：PATCH 中若禁用 tile-level mask 学习，则无法灵活控制整体 sparsity ratio，性能下降。
- **OPTIMA 的重建策略**：相比简单 magnitude-based pruning，OPTIMA 的 column-wise quadratic program 显著改善 zero-shot accuracy（+3.97%）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **“Compression Trinity” 是必要而非充分条件**  
   单一压缩技术存在天花板，只有联合利用 sparsity（减 FLOPs）、quantization（减 bandwidth）、low-rank（补 accuracy）才能突破效率墙。

2. **不同生命周期阶段需差异化压缩策略**  
   - **Training/Prefill**：compute-bound → 优先使用 sparsity 加速矩阵乘法；
   - **Decoding**：memory-bound → 优先使用 quantization 减少 HBM 数据搬运；
   - Trinity 正好覆盖两类瓶颈。

3. **low-rank 不应仅用于 fine-tuning，也可用于压缩恢复**  
   SLiM 和 SLoPE 表明，mathematically-derived low-rank adapters 可高效补偿信息损失，无需大规模 retraining。

4. **静态 mask 有极限，动态学习更优**  
   OPTIMA 在 zero-training 下表现良好，但 PATCH 通过 fine-tuning 学习 hybrid 结构，实现了更高灵活性和性能上限。

5. **端到端加速必须考虑全流程一致性**  
   如 FST 虽然训练稀疏，但最终转为 dense 模型，导致推理无加速；SLoPE 保持稀疏性贯穿始终，真正实现 end-to-end 提速。

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **依赖特定硬件支持** | 如 cuSPARSELt 对 2:4 sparsity 的加速依赖 NVIDIA Ampere 架构 Tensor Core |
| **超大模型扩展性待验证** | 实验集中在 125M–70B 参数范围，万亿级模型可能面临通信瓶颈变化 |
| **KV Cache 未被压缩** | 当前工作聚焦 linear layers，attention 的 KV cache 成为主要内存瓶颈（尤其长序列） |
| **metadata 开销未完全消除** | 稀疏格式仍需存储 indices，低比特下可能抵消压缩增益 |

### **未来工作方向**
1. **将 Trinity 扩展至 Context Window**
   - 量化动态 KV states
   - 在 attention pattern 中引入 sparsity（如 sliding window, block-sparse）
   - 对 attention head 应用 low-rank 分解，支持无限上下文推理

2. **硬件-算法协同设计（Hardware-Algorithm Co-design）**
   - 设计无需显式索引存储的“algorithmic sparsity”
   - 提出“硬件定义”的稀疏格式，减少 metadata 开销

3. **激活张量压缩（Activation Sparsity）**
   - 将 PATCH 思路迁移到 activation tensor，实现 prefill 阶段的 compute-bound 加速
   - 探索 activation-level 的可学习稀疏模式

4. **联合优化框架**
   - 将当前顺序 pipeline（prune → quantize → adapt）升级为 joint optimization
   - 探索 “quantized sparse-plus-low-rank” 分解这一开放问题

5. **更广泛的模型架构适用性**
   - 验证 Compression Trinity 在 MoE、state space models（如 Mamba）中的有效性

---

> 💡 **结语**：本论文确立了一个核心理念——**效率不是事后优化，而是模型设计的基本约束**。“Compression Trinity” 不仅是一种技术组合，更是新一代高效 AI 的设计哲学：**知识密集，计算稀疏**（dense in knowledge, sparse in computation），推动高智能 AI 更加可持续与普惠。

</details>

---

### 4. [Parason: Revealing Subtask and Trial Parallelism in LLM Reasoning](https://arxiv.org/abs/2608.24658)

**Authors**: Zhengyang Zhang, Zijian Zhang, Jiaxuan Gao, Shusheng Xu, Yi Wu, Song Han, Ligeng Zhu  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.24658v1  

#### Abstract
Scaling test-time reasoning has substantially improved the problem-solving ability of large language models (LLMs), but standard autoregressive decoding still executes long reasoning traces sequentially, creating severe latency for difficult tasks (up to days and weeks). Parallel reasoning offers a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Parason: Revealing Subtask- and Trial Parallelism in LLM Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前大型语言模型（LLM）在复杂推理任务中依赖**长序列的 Chain-of-Thought（CoT）推理**，但由于标准的自回归解码机制是**串行执行**的，导致推理延迟极高（甚至可达数天），严重限制了其在交互式应用（如编程助手、数学求解器）中的实用性。

现有并行推理系统（如 Multiverse、ThreadWeaver）主要关注 **Subtask Parallelism**（子任务并行），即把一个大任务分解为多个独立子任务并行处理。然而，它们忽略了另一种广泛存在的并行形式——**Trial Parallelism**（尝试并行），即模型对不确定路径进行多假设探索、验证与聚合的过程。

Parason 发现：**Trial Parallelism 在硬问题中占主导地位（高达 65.5%~76.1%）**，而现有方法未能有效利用这一潜力。

---

### 🚀 提出的新方法与新思路

Parason 是一个**算法-系统协同设计框架**（algorithm-system co-design），旨在揭示并充分利用 LLM 推理中的两种并行性：

#### （1）提出双语义并行分类法（Dual Semantic Taxonomy）
- **Subtask Parallelism（AND 分支）**  
  将问题拆分为多个必须完成的独立子目标（如分治计算 1–800 的和）。所有分支结果都需合并才能得到最终答案。
- **Trial Parallelism（OR 分支）**  
  多个不确定路径被并行探索（如尝试不同公式解 24 游戏），只要有一个成功即可；输出会被拼接进上下文用于后续综合判断。

> 🔍 这种区分具有实际意义：Subtask 减少最长路径延迟，Trial 提升准确性。

#### （2）引入结构化并行轨迹格式（CFG-based Representation）
定义了一个**上下文无关文法（Context-Free Grammar, CFG）** 来显式标记：
- `<Parallel>`：并行区域开始
- `<Outlines>`：列出分支目的
- `<Subtask>` / `<Trial>`：指定分支类型
- `<Thread>`：存储各分支执行结果

该格式使推理过程可被解析、调度和执行。

#### （3）提出 PA-GRPO 强化学习训练算法
**Parallelism-Aware Group Relative Policy Optimization (PA-GRPO)** 是一种多目标强化学习策略，联合优化以下目标：
- 正确性（accuracy）
- 最长路径延迟（token-level latency）
- 子任务与试错并行的比例（parallelism ratio）

奖励函数设计如下：
$$
R = -1 + 2 \cdot \mathbf{1}_{\text{correct}} \left(1 - \frac{T}{\mu_T}\right)^\alpha + \beta_{\text{subtask}} \cdot R_{\text{subtask}} + \beta_{\text{trial}} \cdot R_{\text{trial}} + \text{acceleration reward}
$$

这使得模型学会在合适时机生成合适的并行结构。

#### （4）无缝集成到现代推理引擎
通过将 `<Parallel>` 区域映射为 **tool calls**，并在 SGLang 中实现运行时支持，实现了真正的**端到端可执行并行推理**，无需修改底层推理架构。

---

### ⚖️ 相比现有方法的优势

| 维度 | 现有方法（如 ThreadWeaver） | Parason |
|------|-----------------------------|--------|
| 并行类型 | 主要支持 Subtask Parallelism | 同时建模 Subtask & Trial Parallelism |
| 结构表达能力 | 缺乏语义标签区分分支类型 | 显式语法标记，语义清晰 |
| 可执行性 | 多为理论加速或需深度定制引擎 | 支持 tool call 执行，真实 wall-clock 加速 |
| 训练引导 | 无显式鼓励并行结构 | PA-GRPO 显式奖励并行性和低延迟 |
| 实际收益 | 加速有限（约 1.1–1.2×） | 实测平均 **1.7× token-level acceleration** |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 描述 |
|-------|------|
| **AIME24 / AIME25** | American Invitational Mathematics Examination，高难度数学竞赛题，共 ~30 道/年 |
| **AMC** | American Mathematics Competitions，中等难度选择题 |
| **Math500** | Hendrycks 等人构建的数学推理基准 |
| **OpenMath** | 开放数学推理数据集，强调多样化解法 |
| **Humanity's Last Exam (HLE)** | 极难综合性考试数据集，用于分析并行比例 |

> 注：训练数据来自 ThreadWeaver 发布的 964 条 Qwen3-8B 推理轨迹，并结合 Polaris-53k 进行 RL 微调。

---

### 🧪 实验设置与评估指标

#### 模型规模
- 主要使用 **8B 参数模型**进行训练与比较（控制变量）

#### 基线方法对比
| 方法 | 类型 | 是否开源 |
|------|------|---------|
| ThreadWeaver (8B) | 自适应并行推理 | ✅ |
| Multiverse (32B) | 并行生成框架 | ✅ |
| Dynamic Early Exit | 提前退出机制 | ✅ |
| AdaptThink | 动态思考长度调整 | ✅ |
| Self-consistency / Best-of-N | 多采样投票 | ✅ |

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy (%)** | 最终答案正确率 |
| **Token Latency (#Tokens)** | 最长生成路径上的 token 数量（反映 wall-clock 时间） |
| **Acceleration Ratio** | 总生成 token / 最长路径 token |
| **Trigger Ratio** | 含至少一个 `<Parallel>` 块的样本占比 |
| **Subtask/Trial Ratio** | 各类并行步骤所占比例 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 2 & 4）

| 方法 | AIME24 | AIME25 | Math500 | AMC | Avg Acc | Token Latency | Accel. Ratio |
|------|--------|--------|---------|-----|----------|---------------|--------------|
| ThreadWeaver (8B) | 79.9 | 60.5 | 92.3 | 91.4 | 81.0 | 14.8k | 1.18× |
| **Parason-8B (+PA-GRPO)** | **78.2** | **70.6** | **94.6** | **97.5** | **84.7** | 14.5k | **1.71–1.75×** |

> 💡 Parason 在仅用 **8B 模型**的情况下，超越多数基于 32B 模型的方法，在 AIME25 和 AMC 上达到 SOTA 表现。

---

### 🔍 与基线方法的对比结果

- **平均准确率提升**：从 ThreadWeaver 的 81.0% 提升至 **84.7%**
- **加速比显著提高**：从 1.18× 提升至最高 **1.75×**
- **更优的延迟-精度权衡**：在相同 token 预算下，Parason 能达到更高准确率（见 Table 3）

#### 示例：在 2048 token 预算下的表现（Table 3）
| 方法 | AIME24 Accuracy |
|------|------------------|
| SFT only | 16.8% |
| PA-GRPO (β_subtask=0.1) | **34.7%**（↑17.9 pts）|

> 表明 Parason 能在极低预算下仍保持高效推理。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）不同 β 设置的影响（Table 2）
| 设置 | 效果 |
|------|------|
| ↑ `β_trial` | → 显著提升 accuracy（尤其 AIME25 和 AMC） |
| ↑ `β_subtask` | → 更多 Subtask 分支 → 更短最长路径 → 更高 acceleration ratio |
| ↑ `α`（latency penalty） | → 模型主动压缩 critical path，降低 token latency |

> ✅ 实验表明两个并行维度可以独立调节，实现灵活控制。

#### （2）PA-GRPO 对并行触发的影响（Table 4）
| 设置 | Trigger Ratio | Accel. Ratio |
|------|----------------|--------------|
| SFT only | 69.3% | 1.27× |
| +PA-GRPO (β_trial=0.1) | 98.8% | 1.71× |
| +PA-GRPO (β_subtask=0.05) | 79.3% | **1.75×** |

> 表明 PA-GRPO 成功“教会”模型主动构造并行结构。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Trial Parallelism 是主流**  
   在 HLE 和 OpenMath 上，**Trial Parallelism 占据 58%~76% 的可并行步骤**，远超 Subtask Parallelism。例如：
   - DeepSeek-V4 在 HLE 上 Trial 占 **65.5%**
   - GPT-5.5 达到 **76.1%**

2. **硬问题更依赖 Trial 并行**  
   随着问题难度上升，模型更多地采用“尝试多种思路”的策略，而非简单分解。

3. **并行 ≠ 浪费计算**  
   Parason 将冗余的 Trial 分支转化为**可控的搜索宽度**，并通过结构化表示避免重复计算。

4. **真实加速可行**  
   在 A800 GPU 上实测显示：
   - Easy 问题：**1.62× wall-clock speedup**
   - Hard 问题：**1.47× speedup**
   - 平均节省 **>15k tokens**

---

### ⚠️ 局限性（Limitations）

1. **领域局限**：目前实验集中在**数学推理任务**，尚未验证在代码生成、规划代理等其他领域的泛化能力。
2. **模型尺度受限**：主要在 8B 模型上验证，未扩展至更大模型族（如 70B+）。
3. **人工标注依赖**：训练数据依赖于外部系统（如 Gemini-3-Flash）对 Subtask/Trial 的标注，可能存在偏差。
4. **动态调度开销未计入**：虽然支持 tool call，但多 worker 调度的通信成本未完全建模。

---

### 🔮 未来工作方向

1. **跨领域迁移**：将 Parason 应用于 agentic workflow、科学发现、自动定理证明等场景。
2. **更大模型扩展**：在 32B/70B 级别模型上验证并行模式是否一致。
3. **自动化标注 pipeline**：减少对商业模型（Gemini/GPT）标注的依赖，构建全开源闭环流程。
4. **动态资源分配**：根据问题难度自适应决定 Subtask vs Trial 的比例与数量。
5. **硬件感知调度**：结合 GPU 利用率、内存带宽等因素优化并行执行效率。

---

## ✅ 总结一句话

> **Parason 首次系统性识别并建模了 LLM 推理中的 Trial Parallelism，提出 PA-GRPO 与 CFG 结构化表示，在不牺牲准确性的前提下实现了约 1.7× 的实际推理加速，推动了高效、可执行的并行推理系统发展。**

</details>

---

### 5. [Understanding the Energy Scaling of Large Language Model Inference Across Context Lengths and Attention Architectures](https://arxiv.org/abs/2608.25096)

**Authors**: Molka Chkir, Syed Muhammad Danish, Jos H\"oll, Arghavan Asad  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.25096v1  

#### Abstract
The growing adoption of large language models (LLMs) has raised increasing concerns about the energy consumption and environmental impact of inference. This paper presents a systematic empirical study of decode-phase energy consumption across representative open-source LLMs employing Multi-Head Atte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Understanding the Energy Scaling of Large Language Model Inference Across Context Lengths and Attention Architectures

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
本论文系统地研究了**大型语言模型（LLM）在推理阶段（decode phase）的能耗如何随上下文长度（context length）和注意力机制（attention architecture）的变化而扩展**。尽管已有研究关注 LLM 的训练能耗，但对推理阶段、尤其是不同 attention 架构下能耗动态的研究仍十分有限。

具体而言，论文试图回答以下四个研究问题（RQ）：
- **RQ1**: 解码能耗每生成 token 如何随 context length 增加？
- **RQ2**: 不同 attention 机制如何影响解码能耗的扩展行为？
- **RQ3**: 在自回归生成过程中，token 位置如何影响解码能耗（KV cache 增长的影响）？
- **RQ4**: 请求批处理（batching）如何影响能耗和延迟？

### ✅ 提出了什么新方法或新思路
- 首次**隔离 decode phase** 进行精细化能耗测量，排除 prefill phase 干扰。
- 在统一硬件、精度（FP16）、软件环境（Hugging Face Transformers + PyNVML）下，对采用不同 attention 机制的代表性开源 LLM 进行横向比较。
- 利用 **NVIDIA NVML 硬件计数器**进行精确的 GPU 能耗测量（单位：Joules），提升结果可信度。
- 引入“**within-sequence energy drift**”概念，分析单个生成序列中随 KV cache 扩展导致的能耗变化趋势。

### ✅ 相比现有方法的优势
| 方面 | 以往研究局限 | 本文改进 |
|------|----------------|-----------|
| **研究范围** | 多聚焦训练或整体推理能耗 | 聚焦 decode phase，更具实际部署意义 |
| **attention 影响分析** | 缺乏跨架构系统对比 | 明确区分 MHA、GQA、GQA+SWA 的能耗扩展差异 |
| **测量粒度** | 多为 aggregate 测量 | 细化到 token-level 和 sequence-level 能耗演化 |
| **实验控制** | 模型、硬件、配置不一致 | 统一实验条件，确保公平比较 |

---

## 2. 核心实验方法和设置

### 🧪 使用的模型（非数据集）
由于是推理效率研究，未使用传统 NLP 数据集，而是构建了一个**固定英文段落作为 base prompt**，并截断至目标 context length。所选模型覆盖三种典型 attention 架构：

| 模型 | 参数量 | Attention 类型 |
|------|--------|----------------|
| OPT-1.3B | 1.3B | Multi-Head Attention (MHA) |
| Phi-3 Mini | 1.3B | Multi-Head Attention (MHA) |
| Gemma-2-2B | 2.2B | Grouped Query Attention (GQA) |
| Mistral-7B | 7B | GQA + Sliding Window Attention (SWA) |

> 注：所有模型均从 Hugging Face 下载，本地运行，使用 FP16 精度。

### ⚙️ 实验设置
| 参数 | 设置 |
|------|------|
| **GPU** | NVIDIA A100-SXM4-40GB |
| **集群** | Narval HPC（Digital Research Alliance of Canada） |
| **调度器** | SLURM |
| **解码策略** | Greedy decoding |
| **生成 token 数** | 200 tokens（RQ1–RQ4）；chunk size=100（RQ3） |
| **context lengths** | 128, 512, 1024, 1800 tokens |
| **batch sizes** | 1, 2, 4, 8 requests |
| **重复次数** | 每配置运行 10 次 |
| **预热** | 2 次完整前向传播以稳定 GPU 频率 |
| **能耗测量工具** | `PyNVML` → `nvmlDeviceGetTotalEnergyConsumption()`（硬件级 millijoule 计数） |

### 📊 评估指标
| 指标 | 定义 | 单位 |
|------|------|-------|
| **Energy per generated token** | 总 decode 能耗 / 生成 token 数 | Joules (J) |
| **Latency per token** | 总 decode 时间 / 生成 token 数 | milliseconds (ms) |
| **Latency per request** | 总 decode 时间 / batch size | milliseconds (ms) |

> 所有指标均基于 decode phase，prefill 不计入。

---

## 3. 主要实验结果和性能指标

### 🔢 关键性能数据汇总

#### ✅ RQ1 & RQ2：context length 对 decode energy 的影响（见 Table II 和 Fig. 2）

| Model | Attention | 128→1800 Energy Increase |
|-------|----------|-------------------------|
| OPT-1.3B | MHA | +17.92% |
| Phi-3 Mini | MHA | +16.98% |
| Gemma-2B | GQA | +3.62% |
| Mistral-7B | GQA+SWA | **+3.32%** |

> **结论**：MHA 模型能耗随 context 显著上升；GQA 明显缓解增长；GQA+SWA 几乎实现恒定能耗。

#### ✅ RQ3：token position 对能耗的影响（within-sequence energy drift）

| Model | Context | Drift (pos 50 → pos 950) |
|-------|--------|--------------------------|
| OPT-1.3B (MHA) | 128 | +9.64% |
| Phi-3 Mini (MHA) | 512 | +6.43% |
| Gemma-2B (GQA) | 512 | +0.99% |
| Mistral-7B (GQA+SWA) | 1024 | **-0.83%**（基本不变） |

> **结论**：MHA 中 KV cache 膨胀显著增加后续 token 的 decode energy；GQA 缓解该效应；SWA 几乎完全消除 drift。

#### ✅ RQ4：batching 对能效的影响（见 Tables VII–X）

| Model | Context | Batch 1 → Batch 8 能耗降幅 |
|-------|--------|----------------------------|
| OPT-1.3B | 128 | -84.13% |
| Phi-3 Mini | 128 | -84.82% |
| Gemma-2B | 128 | -86.74% |
| Mistral-7B | 128 | -86.10% |

> **平均降低超过 80%**，最高达 **87%** 的能耗和延迟下降。

同时观察到：
- MHA 模型在长 context 下 batching 效益衰减（如 OPT-1.3B 在 context=1800 时仅降 67%）
- GQA/GQA+SWA 模型在长 context 下仍保持高 batching 效率（>82%）

---

## 4. 关键结论和发现

### 🎯 主要发现

1. **Attention mechanism 是决定 decode energy scaling 的首要因素**
   - MHA：能耗随 context length 快速增长（~17%↑）
   - GQA：显著抑制增长（~3–4%↑）
   - GQA+SWA：几乎实现 constant energy scaling（<4%↑）

2. **Model size 决定绝对能耗水平**
   - 尽管 OPT-1.3B 使用低效 MHA，但因参数少，其每 token 能耗最低（1.596 J @128）
   - Mistral-7B 虽用最先进 attention，但因模型大，能耗最高（4.89 J @128）

3. **KV cache 增长导致 within-sequence energy drift**
   - MHA 模型在生成后期能耗持续上升
   - GQA 减缓 drift，GQA+SWA 基本消除 drift

4. **Batching 是最有效的部署级优化手段**
   - 可将 energy per token 和 latency per request 同时降低 **高达 87%**
   - 对 GQA/GQA+SWA 模型更友好，尤其在长 context 场景下

5. **Latency 不能反映真实能耗代价**
   - MHA 模型能耗上升但 latency 几乎不变 → 表明能耗来自更高 GPU power 或 memory activity，而非执行时间延长

---

### ⚠️ 局限性

1. **仅评估 decode phase**  
   虽然符合研究目标，但在真实场景中 prefill 也可能成为瓶颈，特别是对于长输入。

2. **单一硬件平台（A100）**  
   结果可能无法直接外推到其他 GPU（如 consumer-grade 或移动设备）。

3. **未考虑量化、缓存优化等系统级技术**  
   如 KV cache quantization、PagedAttention 等可进一步影响能耗，但不在本研究范围内。

4. **仅测试 greedy decoding**  
   采样策略（如 beam search、top-k）可能引入不同计算模式，影响能耗分布。

---

### 🔮 未来工作方向

1. **扩展至更多 attention 架构**  
   如 MQA（Multi-Query Attention）、ALiBi、FlashAttention 等，建立更完整的 energy-scaling 图谱。

2. **结合系统优化技术联合分析**  
   探索 batching + quantization + attention 架构的协同节能潜力。

3. **端到端任务级能耗建模**  
   将 energy scaling 模型应用于真实应用（如 chatbot、代码生成），预测全生命周期能耗。

4. **绿色 LLM 设计指南**  
   基于本研究结论，提出面向可持续部署的 LLM 架构选择与 inference 配置推荐框架。

---

> ✅ **一句话总结**：  
> 本论文揭示了 **attention architecture 是 LLM 推理能耗扩展性的核心控制器**，而 **model size 决定基础能耗水平**，**batching 是最强部署优化**。为绿色、高效 LLM 部署提供了实证依据与设计指导。

</details>

---

### 6. [Groundhog Bit-Flip Attack: Seeding Infinite Generation Loops in Mixture-of-Experts LLMs through Bit Flips](https://arxiv.org/abs/2608.25276)

**Authors**: Huakang Lin, Tiancheng Zheng, Mingxuan Sun, Tianhong Xu, Fan Zhang, Yunsi Fei, Ruyi Ding  
**Category**: cs.CL  
**Published**: 2026-08-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.25276v1  

#### Abstract
Mixture-of-Experts (MoE) architectures enable scalable and efficient large language models (LLMs) by selectively activating expert sub-networks through a routing mechanism. However, this adaptive design introduces a new attack surface: specific experts become disproportionately correlated with certa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Groundhog Bit-Flip Attack: Seeding Infinite Generation Loops in Mixture-of-Experts LLMs through Bit Flips

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文揭示了 **Mixture-of-Experts (MoE)** 架构在 **availability 安全性** 上的重大漏洞。尽管 MoE 能有效降低推理成本，但其路由机制（routing）使得某些专家（expert）高度专业化于生成特定控制类 token（如 `<EOS>` 或 `<EOT>`）。攻击者可利用此特性，通过微小扰动（bit flip）破坏这些关键专家的功能，导致模型无法正常终止输出，从而引发 **Denial-of-Wallet (DoW)** 攻击 —— 用户因无限生成而消耗大量 token，造成经济负担。

### 提出了什么新方法或新思路
作者提出了 **Groundhog Bit-Flip Attack (GBFA)**，这是首个针对 MoE-based LLM 的基于 bit-flip 的 DoW 攻击方法。其核心思想是：
- 利用 MoE 中专家对特定 token（如 EOS）的高度依赖性；
- 识别并定位负责生成终止 token 的“目标相关专家”（target-related experts）；
- 在 **router 层** 对这些专家对应的权重或偏置进行 **bit-level 扰动**，使其在推理时不再被激活，从而抑制 EOS 生成，诱导模型进入无限循环。

### 相比现有方法的优势
| 维度 | 传统 BFA（Bit-Flip Attack） | GBFA（本文） |
|------|----------------------------|------------|
| 攻击目标 | Integrity（完整性），如降低准确率、诱导错误分类 | Availability（可用性），诱导输出膨胀，实现 DoW |
| 影响范围 | 广泛影响模型整体行为 | 精准操控特定功能（如终止逻辑） |
| 隐蔽性 | 输出可能失真、无意义 | 语义保真度高，输出仍连贯，难以检测 |
| 参数修改量 | 通常需大量 bit 修改 | **平均 <4 个专家** 即可引发显著效果 |

> ✅ **创新点总结**：首次将 bit-flip 攻击从 integrity-oriented 转向 availability-oriented，提出了一种轻量级、高隐蔽性的结构性攻击范式，精准打击 MoE 的路由机制弱点。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **专家识别阶段**：使用 Stanford Alpaca instruction-tuning 数据集构建样本序列。
- **攻击评估阶段**：
  - 分类任务：`AGNews`, `SST-2`
  - 生成任务：`Samsum`（对话摘要）、`SQuAD 2.0`（问答）
- **代理编码任务**：自定义 10 个 Python 编码沙箱环境（sandbox），用于测试 plan mode 和 thinking loop 攻击。

### 实验设置和评估指标

#### 模型
共评估 **6 个开源 MoE-based LLMs**：
- Mixtral-8x7B
- Phi-3.5-MoE
- DeepSeek-V2-Lite
- Qwen3-30B-A3B
- Qwen3-Coder-Next
- GPT-OSS-20B

#### 攻击流程（三步法）
1. **Step 1：Target-related Expert Detection**
   - 提出两种策略：
     - **GLOBAL**：跨层排序所有专家，选择 Top-K 差异最大的。
     - **LOCAL**：逐层分析，找出对输出长度影响最大的一层。
   - 度量指标：**Target Activation Shift (T<sub>l,i</sub>)** 和 **Target Gate Shift (Δg<sub>l,i</sub>)**

2. **Step 2：Vulnerable Bit Search**
   - 缓存目标层隐藏状态，无需重复前向传播；
   - 枚举 router 权重/偏置中的每一位，计算其 **bit-flip effectiveness (BF-eff)**；
   - 选取每专家 top-3 最有效的 bit 进行翻转。

3. **Step 3：Online BFA Execution**
   - 模拟 Rowhammer 攻击，在部署模型的 DRAM 中注入 bit flip；
   - 一次成功即可持久化影响后续所有请求。

#### 评估指标
| 指标 | 含义 |
|------|------|
| **P (%)** | 输出 token 数变化百分比：<br>P = (T_deact − T_base)/T_base × 100% |
| **Clean Accuracy (CA)** | 分类任务准确率 |
| **ROUGE-1 / F1** | 生成质量度量 |
| **Perplexity (PPL)** | 语言流畅性指标 |
| **EOS emission rate** | 成功生成 EOS 的样本比例 |

#### 基线方法对比
- **Random Deactivation**：随机关闭 50 个专家作为负面对照；
- **Manual Deactivation**：手动设 logits 为 `-inf`，模拟理想攻击上限；
- **No Attack (Baseline)**：原始模型表现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 输出膨胀幅度（P 值）
| 模型 | 最大 P 增幅（分类任务） | 典型增幅 |
|------|------------------------|---------|
| Mixtral | **~2.74×10⁹%**（AGNews, GLOBAL） | >10⁵% |
| Phi-3.5 | **~2.99×10⁹%**（AGNews, GLOBAL） | >10⁷% |
| DeepSeek | **~3.81×10⁴%**（AGNews, LOCAL） | ~10⁴% |
| GPT-OSS | **556%**（AGNews, GLOBAL） | ~400–500% |

> 🔥 **最高达 87× 输出增长**，多数测试样本达到 `max_new_tokens` 上限（1024）。

#### Bit-Flip 攻击效果（Table 3）
- **仅翻转每个专家 3 个 bit**，即可实现接近 manual deactivation 的效果。
- 例如在 DeepSeek 上，GLOBAL bit-flip 攻击使 AGNews 输出 token 从 47.4 → **2510+**，P 达 **4.64×10⁴%**。
- GPT-OSS 因直接攻击 **bias 参数**，效果更稳定，接近人工去活化水平。

#### 性能保留情况（Table 4）
| 模型 | 攻击后 CA 变化 | PPL 变化 |
|------|---------------|----------|
| Mixtral | 基本不变（0.838 → 0.840） | 保持 <2.0 |
| Phi-3.5 | 小幅下降 | 多数 <2.0 |
| DeepSeek | **严重崩溃**（PPL > 5e6） | ❌ 不稳定 |
| GPT-OSS | 几乎无损 | PPL ≈ 1.5 |

> ✅ **大多数模型在攻击后仍保持良好语义一致性**，表明 GBFA 具备强隐蔽性。

#### 消融实验结果
- **LOCAL vs GLOBAL**：
  - 当 EOS 控制集中于某一层时，LOCAL 更高效；
  - 若分布于多层，则需 GLOBAL 才有效（如 GPT-OSS）。
- **解码策略鲁棒性**（Appendix E）：
  - 在 greedy、temperature sampling、nucleus sampling 下均有效；
  - 即使加入 repetition penalty (rp=1.3)，仍能进一步延长输出。
- **token budget 影响**（Appendix F）：
  - 更大的 `max_new_tokens` 导致更高的相对增幅 P；
  - 但在 1024 已足以暴露攻击威胁。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **MoE 存在严重的 termination specialization 现象**：少数专家主导 EOS/EOT 生成，形成天然攻击面。
2. ✅ **极少量 bit flip 即可触发无限生成**：平均关闭不到 4 个专家，即可驱动输出膨胀至 **5912%**。
3. ✅ **攻击具有高隐蔽性和实用性**：
   - 语义连贯，不易被用户察觉；
   - 可通过 Rowhammer 等硬件故障注入实现；
   - 攻击一次成功即永久生效（因修改的是 persistent weights）。
4. ✅ **适用于多种 LLM 模式**：
   - 对话模式（conversational）
   - 推理模式（reasoning）
   - 代理模式（agentic planning），如 Qwen3-Coder 的 plan-step 被拉长至最大步数。

### 方法的局限性
- **强威胁模型假设**：
  - 需白盒访问模型参数（虽合理，因许多 MoE 模型公开权重）；
  - 需具备 Rowhammer 或电压毛刺等 fault injection 能力。
- **系统级防御存在挑战**：
  - ECC 内存可缓解但不完全阻止；
  - TEE 隔离有开销且非普遍部署。
- **部分模型不稳定**：
  - 如 DeepSeek 在 bit-flip 后出现语言崩溃（PPL > 5e6），限制了通用性。
- **缺乏理论建模**：
  - 未形式化解释为何某些 bit 具有不成比例的影响。

### 未来工作方向
1. **MoE-specific 防御机制设计**：
   - 引入 routing redundancy 或 checksum 保护关键专家；
   - 动态监控 expert activation pattern 异常。
2. **防御与检测结合**：
   - 开发 real-time loop detection（如 Breaking the Loop）；
   - 结合 API 层长度限制 + 行为异常评分。
3. **扩展到其他控制 token**：
   - 攻击 `<think>`, `<tool_call>`, `<image>` 等结构化 token 的生成路径。
4. **探索训练时加固策略**：
   - 通过 adversarial training 增强 router 对 bit-flip 的鲁棒性。

---

> 📌 **总结一句话**：  
> **GBFA 揭示了 MoE 架构中“高效路由”背后的“脆弱终止”悖论 —— 正是那些让模型更快的专家，成了让它永不结束的致命弱点。**

</details>

---

### 7. [psRL: Efficient Training for Agentic AI via Training-Time Prefix Sharing](https://arxiv.org/abs/2608.25683)

**Authors**: Mianjie Yu, Zizhao Mo, Huanyu Qu, Zhirong Qian, Huanle Xu, Cen Li, Zifeng Zhao, Zhi Zhou, Jinhua Zhou, Jun Xie, Chengzhong Xu  
**Category**: cs.DC  
**Published**: 2026-08-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.25683v1  

#### Abstract
In modern agentic AI training, the system bottleneck is shifting from rollout to update. Emerging sampling strategies such as tree-structured and step-wise RL greatly increase training sample volume while incurring relatively low marginal rollout cost, causing the update phase to dominate the end-to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*psRL: Efficient Training for Agentic AI via Training-Time Prefix Sharing*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现代 **Agentic AI**（智能体AI）训练中，传统的系统瓶颈是 **rollout** 阶段（即模型生成轨迹），但随着 **tree-structured RL** 和 **step-wise RL** 等采样策略的兴起，训练样本量急剧增加，而 rollout 成本增长缓慢。这导致 **update 阶段**（参数更新）成为新的性能瓶颈，占端到端训练时间的 **超过 50%**。

然而，现有训练系统（如 Megatron-LM、DeepSpeed）将每个训练样本视为独立序列处理，忽略了大量样本之间存在的 **前缀冗余（prefix redundancy）**，造成重复计算和内存浪费。

### 提出了什么新方法或新思路

论文提出 **psRL**（prefix sharing for RL），一种专为 **agentic RL 训练** 设计的新型训练系统，其核心思想是利用训练阶段的两个特性：

- **Global Visibility**：整个训练数据在 update 阶段已知且静态。
- **Data Immutability**：训练样本在 update 过程中不会改变。

基于此，psRL 在 **workload scheduling** 和 **memory management** 两方面进行优化：

#### 创新机制一：灵活的前缀共享机制（Flexible Prefix Sharing）
1. **Inter-batch Sharing**  
   允许具有相同前缀的序列分布在不同 micro-batch 中，复用已计算的 KV states 和 attention 结果，提升前缀复用率并支持灵活调度。

2. **Self-sequence Sharing**  
   将长序列切分为多个连续块，跨 micro-batch 执行，通过缓存前缀 KV 来降低峰值内存占用并消除 pipeline bubbles。

#### 创新机制二：自适应 KV 缓存管理（Adaptive KV Cache Manager）
1. **Adaptive Block Allocation (ABA)**  
   动态分配可变大小的内存块以精确匹配共享前缀长度，消除内部碎片化（internal fragmentation），提高内存访问效率。

2. **Dynamic Block Caching (DBC)**  
   基于 rollout 构建的前缀树分析，采用 **Just-in-Time** 策略：提前缓存未来需要的 KV 块，并在最后一次引用后立即驱逐，最大化内存利用率和前缀命中率。

此外，psRL 实现了 **分层调度策略**：
- **全局层面**：按语义组（semantic groups）划分数据，平衡负载并保留前缀局部性。
- **本地层面**：采用 **token-wise micro-batching**，在 token 粒度上打包序列，减少因长度不均造成的 pipeline bubbles。

### 相比现有方法的优势

| 维度 | 现有方法（如 vLLM-PS, SGLang-PS） | psRL |
|------|-------------------------------|------|
| **前缀共享粒度** | 固定块大小（如 1 或 16 tokens） | 自适应块大小，精准对齐前缀边界 |
| **内存管理** | 被动缓存，易产生碎片或高访问开销 | 主动 JIT 缓存 + 及时回收 |
| **调度灵活性** | 忽视前缀复用或牺牲负载均衡 | 同时优化前缀复用与负载均衡 |
| **适用场景** | 主要针对推理 | 专为训练设计，支持反向传播正确性 |

---

## 2. 核心实验方法和设置

### 使用的数据集

实验覆盖多种 **agent workloads**，包括：

- **学术基准环境**：
  - `Search`（搜索任务）
  - `WebShop`（电商交互）
  - `ALFWorld`（文本游戏环境）
- **工业级大规模代理**：
  - `DTN Agent`：基于 **Qwen3-235B MoE** 模型的数字孪生网络运维助手，处理真实设备配置文件与操作任务。

### 实验设置和评估指标

#### 模型配置
- 学术任务：Qwen2.5-1.5B / 7B
- 工业任务：Qwen3-235B MoE（最大 40K prompt + 20K gen tokens）

#### 分布式并行
- 学术任务：DP=2, TP=2, PP=2 → 8 GPUs
- DTN Agent：DP=4, TP=4, EP=8, PP=12 → 1536 GPUs

#### 采样策略（RL Modes）
- Trajectory-wise (Traj)
- Step-wise (Step)
- Step + Summarization (Step-S)
- Step + Remove Think (Step-RT)
- Tree-structured (Tree)
- Tree-Step (结合 tree 与 step)

#### 评估指标
- **Throughput**（tokens/s）：update 阶段吞吐量
- **Peak GPU Memory Usage**（MB）：峰值显存消耗
- **End-to-End RL Latency**：完整训练迭代耗时
- **Ablation Studies**：验证各组件有效性

### 基线方法对比

| 基线 | 描述 |
|------|------|
| **veRL** | 默认训练框架，无前缀共享 |
| **veRL w/ vLLM-PS** | 引入 vLLM 风格的固定块大小（16 tokens）前缀共享 |
| **veRL w/ SGLang-PS** | 引入 SGLang 风格的细粒度块（1 token）前缀共享 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 更新阶段吞吐量（Update Throughput）
- psRL 在所有 workload 上实现 **1.2× ~ 5.2×** 于 veRL 的加速。
- 在 **DTN Agent (Step)** 场景下达到 **239.1k tokens/s**，相比最强基线 SGLang-PS（62.7k tokens/s）提升 **3.8×**。

#### 显存使用（Peak Memory）
- psRL 显存开销接近原生 veRL，远低于 vLLM-PS 和 SGLang-PS。
- 在极端场景 **ALFWorld (Step-S)** 中：
  - vLLM-PS 峰值达 **73.1 GB**
  - veRL 仅 **6.4 GB**
  - psRL 控制在 **相近水平**
- 在 **DTN Agent** 上，psRL 将峰值内存从 **56.2 GB** 降至 **33.0 GB**（↓41%）

#### 端到端训练延迟（E2E Latency）
- psRL 实现最高 **2.1×** 的端到端加速。
- 在 DTN Agent (Step) 中：
  - `Log Old Prob` 时间从 57.1s → 7.1s（↓8×）
  - `Reference Model Forward` 从 68.3s → 8.3s（↓8×）
  - `Update` 阶段从 462.1s → 88.7s（↓5.2×）

### 与基线方法的对比结果

| 方法 | 吞吐量优势 | 内存优势 | 正确性保障 |
|------|----------|--------|----------|
| vLLM-PS | 有限加速（~1.5×） | 显存爆炸（↑2–10×） | 不适用于训练反向传播 |
| SGLang-PS | 加速较弱（~1.8×） | 更高访问开销 | 缺乏生命周期控制 |
| **psRL** | ✅ 最高 **5.2×** 加速 | ✅ 显存更低，可控增长 | ✅ 支持 masked attention 与 loss 计算 |

### 消融实验结果（Ablation Study）

#### （1）调度策略对比（Fig. 11a）
- **Sharing-Agnostic Balancing**（veRL 默认）：执行时间严重倾斜（235s ~ 299s）
- **Prefix-Match-First**：前缀复用高但负载极不平衡
- **Balance-First**：平衡好但前缀复用差
- **psRL（联合优化）**：时间集中在 **245–261s**，兼顾复用与均衡

#### （2）Token-wise Micro-Batching 效果（Fig. 12）
- 显著平滑 micro-batch 执行时间，避免延迟尖峰。
- 减少等待气泡（pipeline bubbles），提升 GPU 利用率。

#### （3）Dynamic Block Caching 影响（Fig. 11b）
- 无 DBC：KV 内存持续增长至 **73k MB**
- 启用 DBC：峰值降至 **33k MB**（↓54%），呈锯齿状释放模式

#### （4）Adaptive Block Allocation 性能（Fig. 13）
- 固定块大小存在明显 trade-off：
  - 小块 → 高访问开销
  - 大块 → 高内存浪费
- ABA 无需调参，在 Search 和 DTN Agent 上均表现最优

---

## 4. 关键结论和发现

### 主要发现

1. **瓶颈转移**：现代 agentic RL 的瓶颈已从 **rollout-bound** 转向 **update-bound**，传统优化不再有效。
2. **前缀冗余普遍存在**：在 step-wise 和 tree-structured RL 中，**prefix match rate 超过 90%**（如 ALFWorld Step 模式达 94.51%），现有系统未充分利用。
3. **训练阶段的独特优势**：Global Visibility 与 Data Immutability 为前缀共享提供了全新优化空间。
4. **psRL 实现显著加速**：通过 **inter-batch/self-sequence sharing + ABA + DBC**，在真实生产 trace 下实现最高 **5.2× 吞吐提升** 和 **41% 显存下降**。

### 方法的局限性

- **依赖 rollout 结构信息**：需提前构建前缀树用于调度与缓存决策，可能增加预处理开销。
- **扩展性依赖良好语义分组**：若语义组内前缀重叠低，效果会打折扣。
- **当前实现在 Megatron-LM 上定制**：通用性有待在其他训练框架中验证。

### 未来工作方向

- 将 prefix sharing 扩展至 **off-policy RL** 或 **multi-agent training** 场景。
- 探索 **自动识别语义组** 的轻量算法，降低调度开销。
- 结合 **compression** 与 **quantization** 技术进一步压缩 KV cache。
- 支持动态 rollout 生成过程中的在线前缀共享。

---

> **代码开源计划**：作者表示 psRL 源码将很快公开。

</details>

---

### 8. [Drift-Aware Multimodal User Representation Learning via Multi-Scale Temporal Modeling and Sparse Mixture-of-Experts](https://arxiv.org/abs/2608.25773)

**Authors**: Ziqing Qian, Haohang Chen, Shengqi Dang, Yuhan Xiong, Canyu Shen, Jiaying Lei, Nan Cao  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.25773v1  

#### Abstract
Understanding user preferences from noisy and temporally evolving social media behaviors is fundamentally challenging due to interest drift, where user preferences shift across time and exhibit both multi-scale temporal patterns and diverse co-existing interests. To address this, we propose DUMoE, a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**用户兴趣漂移**（interest drift）带来的挑战，即用户偏好在时间上不断演变，同时表现出多尺度的时间模式（如短期行为动态与长期依赖）以及多种共存的兴趣。传统方法通常采用静态表示或粗粒度时序建模，难以有效捕捉这种复杂的动态演化过程。

此外，现有方法大多局限于单一模态（如仅文本或交互序列），缺乏对**多模态信号**（text, image, interaction）、**多尺度时间动态**和**显式/隐式用户兴趣**的统一建模框架。

---

### 提出的新方法与思路
作者提出了 **DUMoE**（Drift-aware User Modeling with Mixture-of-Experts），一个统一的用户表征学习框架，其核心创新包括：

#### （1）Temporal Dynamics-Aware Backbone Network
- 融合三种时间维度的信息：
  - **Static Profile**：用户静态属性（如粉丝数、自我描述）
  - **Short-term Behavioral Signals**：通过 LSTM 建模最近的行为序列
  - **Long-term Dependencies**：通过 Transformer 捕捉历史交互中的长程依赖
- 将三者融合为层次化用户表示 $ z_u $，实现对多尺度时间动态的有效建模。

#### （2）Sparse Mixture-of-Experts (MoE) Interest Adapter
- 引入稀疏 MoE 结构进行**多兴趣解耦**（multi-interest disentanglement）：
  - 多个专家网络（expert networks）各自专注于不同的潜在兴趣子空间
  - 门控网络（gating network）动态选择最相关的 K 个专家（top-K routing），实现个性化路由
  - 残差结构设计使专家在共享语义空间基础上进行特化，提升可解释性和稳定性

#### （3）Three-Stage Training Strategy
为稳定训练并促进专家专业化，提出分阶段优化策略：
1. **Backbone Pre-training**：先训练主干网络，不启用 MoE
2. **Expert Specialization**：冻结 backbone，独立训练每个专家对应特定兴趣类别
3. **Gating Optimization**：冻结 backbone 和专家，仅训练门控网络参数

---

### 相比现有方法的优势
| 维度 | DUMoE 的优势 |
|------|---------------|
| **多模态整合** | 支持 text + image 内容联合编码（使用 UNITE 模型） |
| **时间建模** | 显式区分短时、长时与静态偏好，优于单一序列模型（如 SASRec/BERT4Rec） |
| **兴趣多样性建模** | MoE 实现兴趣解耦与稀疏激活，优于单向量或多兴趣胶囊路由方法（如 MIND/HORAE） |
| **训练稳定性** | 分阶段训练避免端到端联合优化导致的不稳定收敛 |

---

## 2. 核心实验方法和设置

### 数据集
构建了一个大规模、带有时序标注的多模态社交媒体数据集，来源于 **X (Twitter)** 平台：
- **用户数量**：14,015
- **推文数量**：7.685M
- **图片数量**：2.890M
- **兴趣领域**：15 个高层类别（如 Entertainment, Politics, Technology, Food 等）
- **标签生成方式**：使用 GPT-5.1 对每组连续 30 条推文的内容及其前序上下文进行自动标注，形成伪标签
- **数据划分**：按时间顺序切分为 8:2:2（训练:验证:测试）

> 图 2 展示了各兴趣领域的数据分布，显示主题覆盖广泛且存在显著差异。

---

### 实验设置与评估指标

#### 下游任务
1. **User Interest Classification**（兴趣分类）
2. **Social Interaction Prediction**（社交互动预测）

#### 评估指标
| 任务 | 指标说明 |
|------|--------|
| **Interest Classification** |  
| - Hit@1 | Top-1 准确率 |
| - NDCG@3 | 排名质量（强调 top 位置） |
| - Macro Recall | 宏平均召回率 |
| - KL Divergence | 预测分布与真实分布之间的对齐程度（越低越好） |
| **Interaction Prediction** |  
| - Accuracy (Acc) | 分类准确率 |
| - Recall | 正样本检索能力 |
| - F1 Score | 综合性能 |
| - BCE Loss | 概率校准能力（越低越好） |

#### 基线方法对比
涵盖三大类主流方法：

##### （1）Sequential Recommendation
- SASRec [1]
- BERT4Rec [2]
- PTUM [6]

##### （2）Multi-Interest Modeling
- MIND [22]
- HORAE [28]

##### （3）Content-Aware Representation Learning
- PeterRec [3]
- UniSRec [18]

所有基线均使用官方代码复现，并保持相同的数据划分与预测头结构以确保公平比较。

---

## 3. 主要实验结果和性能指标

### 用户兴趣分类结果（Table 1）

| Method | Hit@1↑ | NDCG@3↑ | Recall↑ | KL↓ |
|--------|--------|---------|--------|-----|
| SASRec | 0.788 | 0.883 | 0.723 | 0.584 |
| BERT4Rec | 0.776 | 0.879 | 0.707 | 0.568 |
| PTUM | 0.811 | 0.900 | 0.757 | 0.482 |
| MIND | 0.821 | 0.904 | 0.763 | 0.471 |
| HORAE | 0.807 | 0.897 | 0.746 | 0.506 |
| PeterRec | 0.786 | 0.885 | 0.714 | 0.532 |
| UniSRec | 0.767 | 0.873 | 0.689 | 0.595 |
| **DUMoE (Ours)** | **0.872** | **0.940** | **0.856** | **0.323** |

> ✅ **关键突破**：
- Hit@1 提升约 **6.2%**（vs MIND）、**7.5%**（vs PTUM）
- Recall 提升 **+12.2%**（vs MIND）
- KL Divergence 下降 **31.4%**（vs MIND），表明预测分布更贴近真实分布

---

### 用户互动预测结果（Table 2）

| Method | Acc↑ | Recall↑ | F1↑ | BCE↓ |
|--------|------|--------|-----|-----|
| SASRec | 0.873 | 0.877 | 0.874 | 0.302 |
| BERT4Rec | 0.877 | 0.884 | 0.878 | 0.294 |
| PTUM | 0.886 | 0.887 | 0.886 | 0.274 |
| MIND | 0.884 | 0.884 | 0.884 | 0.277 |
| HORAE | 0.882 | 0.888 | 0.883 | 0.281 |
| PeterRec | 0.883 | 0.894 | 0.885 | 0.279 |
| UniSRec | 0.875 | 0.883 | 0.876 | 0.297 |
| **DUMoE (Ours)** | **0.883** | **0.926** | **0.888** | **0.276** |

> ✅ **关键表现**：
- Recall 达到 **0.926**，远超最佳基线 PeterRec（+3.2%）
- F1 与最佳持平（0.888），BCE 与 PTUM/MIND 相当，说明在保持良好概率校准的同时大幅提升检索能力

---

### 消融实验结果（Table 3）

| 变体 | Hit@1↑ | Recall↑ | Acc↑ | BCE↓ | 说明 |
|------|--------|--------|------|-----|------|
| w/o Adapter | 0.858 | 0.810 | 0.858 | 0.329 | 移除 MoE 导致性能下降，证明其必要性 |
| w/o Gating | 0.864 | 0.827 | 0.858 | 0.326 | 均匀加权替代门控，削弱个性化路由效果 |
| w/o Stage-wise Training | 0.865 | 0.846 | 0.873 | 0.294 | 性能略降，验证三阶段训练有助于稳定收敛 |
| Static Only | 0.774 | 0.728 | 0.807 | 0.409 | 忽略动态行为严重损害性能 |
| Short-term Only | 0.805 | 0.738 | 0.846 | 0.350 | 短期建模不足 |
| Long-term Only | 0.862 | 0.829 | 0.864 | 0.313 | 表现较好但仍低于完整模型 |
| Lcls Only | 0.870 | 0.848 | 0.799 | 0.437 | 仅用分类损失损害交互预测 |
| Lrec Only | 0.798 | 0.719 | 0.869 | 0.303 | 仅用重建损失损害兴趣识别 |

> 🔍 **结论**：
- 所有组件均有贡献，尤其是 MoE Adapter 和 Adaptive Gating
- 三阶段训练提升了 KL 对齐和整体稳定性
- 多任务目标（Lcls + Lrec）对于平衡两类任务至关重要

---

## 4. 关键结论和发现

### 主要发现
1. **多尺度时间建模是关键**：结合 static profile、short-term dynamics 和 long-term history 显著优于单一时间视角。
2. **Sparse MoE 有效支持多兴趣解耦**：专家专业化 + 动态路由机制能够更好地建模用户的多样化兴趣，提升可解释性与个性化表达能力。
3. **分阶段训练提升稳定性与性能**：解耦 backbone 学习、expert specialization 和 gating optimization 有助于避免训练干扰，促进清晰的角色分工。
4. **多模态输入增强表征质量**：融合 text 和 image 内容显著提升兴趣理解能力，尤其在视觉相关领域（如 Fashion & Beauty, Food）。

---

### 方法的局限性
1. **伪标签噪声风险**：兴趣标签由 GPT-5.1 自动生成，可能存在标注偏差或噪声，影响监督信号的质量。
2. **固定窗口长度**：LSTM 与 Transformer 使用固定的短/长期窗口（Lt << Lt），未考虑用户个体活动频率差异，可能造成信息丢失或冗余。
3. **静态专家配置**：专家数量 K 和 top-K’ 路由策略固定，无法根据用户行为复杂度自适应调整活跃专家数量。
4. **离线训练范式**：当前框架基于批量训练，缺乏在线增量更新能力，难以应对实时兴趣漂移。

---

### 未来工作方向
1. **引入半监督/自监督学习**：减少对外部语言模型生成标签的依赖，探索 contrastive learning 或 masked modeling 进行无监督兴趣发现。
2. **自适应时间建模**：设计基于用户活跃度的动态窗口选择机制，实现个性化的多尺度聚合。
3. **Dynamic Expert Allocation**：让专家数量或激活比例随用户行为多样性动态变化，提升灵活性。
4. **Online Continual Learning**：开发轻量级微调机制（如 LoRA、Adapter Tuning），支持实时更新用户表示以响应快速兴趣漂移。
5. **更细粒度的兴趣建模**：从高层 domain-level 向 sub-topic 或 intent-level 扩展，提升推荐精度。

---

> 📌 **总结**：  
> DUMoE 是首个将 **multimodal input**、**multi-scale temporal dynamics** 和 **sparse MoE-based multi-interest modeling** 统一于一个框架中的用户建模方法，在真实社交平台数据上展现出卓越的性能。它不仅推动了 drift-aware user representation learning 的发展，也为未来构建更加智能、动态、个性化的推荐系统提供了新范式。  
> 作者承诺将公开代码与数据集（遵守隐私政策），促进可复现研究。

</details>

---

### 9. [Selective Regenerative Decoding: Trajectory-Level Intervention for Inference-Time Reasoning](https://arxiv.org/abs/2608.24338)

**Authors**: Sophia Xiao Pu, Yumo Xu, Sailik Sengupta, Millennium Bismay, Ruixue Lian, James Gung, Yi-an Lai, Arshit Gupta  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.24338v1  

#### Abstract
Inference-time decoding methods improve LLM reasoning by exploring multiple candidate trajectories, yet treat each trajectory as atomic: either retaining it whole or discarding it irreversibly. This wastes computation on partially promising candidates whose high-quality prefixes are abandoned alongs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Selective Regenerative Decoding: Trajectory-Level Intervention for Inference-Time Reasoning**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

现有的 **inference-time decoding** 方法（如 Best-of-N、Speculative Rejection）在处理推理轨迹时，将整个轨迹视为一个不可分割的原子单元（atomic unit），要么保留整个轨迹，要么完全丢弃。这种策略存在以下问题：

- **计算浪费**：许多候选轨迹具有高质量的前缀（prefix），但在后半部分出现质量下降（degraded suffix）。这些“边缘候选”（borderline candidates）因整体得分低而被丢弃，导致前期高质量推理的计算资源被浪费。
- **效率低下**：Speculative Rejection 虽然能提前终止低分前缀，但决策不可逆，一旦拒绝就永久丢失潜在有用的部分。

### ✅ **提出了什么新方法或新思路**

论文提出 **Selective Regenerative Decoding (SRD)**，一种支持**段级干预**（segment-level intervention）的解码算法，其核心思想是：

> **不抛弃整个轨迹，而是识别并仅重生成质量下降的后缀部分，同时保留高质量前缀。**

SRD 分为三个阶段：
1. **Generation**：从生成模型 $ G $ 中采样多个候选轨迹。
2. **Routing**：使用奖励模型 $ R $ 对每个轨迹打分，并根据归一化排名分数 $ u(T) $ 将其路由到三种状态：
   - **Keep**：高分轨迹直接保留。
   - **Discard**：低分轨迹直接丢弃。
   - **Refine**：中等分数轨迹进入重生成流程。
3. **Regeneration**：对需 **Refine** 的轨迹，定位质量首次下降的位置 $ j^* $，然后以更高温度重新生成从 $ j^*+1 $ 开始的后缀。

该方法**无需更大的目标模型**（larger target model），仅依赖现有生成模型和奖励模型即可完成局部编辑。

### ✅ **相比现有方法的优势**

| 方法 | 是否可逆 | 是否支持局部干预 | 是否需要更大模型 | 样本效率 |
|------|----------|------------------|------------------|-----------|
| **Best-of-N** | 否 | 否 | 否 | 低（全生成） |
| **Speculative Rejection** | 否 | 否 | 否 | 中（早停） |
| **SRD (Ours)** | 是 | ✅ 是 | 否 | **高（选择性重写）** |

- **更高的样本效率**：理论证明 SRD 在样本效率上比 rejection sampling 高 **1.28–1.36×**。
- **更高的期望轨迹质量**：通过保留并修复中等质量轨迹，提升最终选出的最佳轨迹质量。
- **更优的 accuracy-compute tradeoff**：在相同或更低 token 成本下达到与 Best-of-N 相当甚至更好的性能。

---

## 2. **核心实验方法和设置**

### ✅ **使用的数据集**

实验覆盖四类典型任务，涵盖数学推理、科学问答、多跳问答和指令跟随：

| 数据集 | 任务类型 | 评估指标 |
|--------|----------|-----------|
| **MATH500** | 数学推理 | Accuracy |
| **GPQA Diamond** | 科学问答（研究生水平） | Accuracy |
| **HotpotQA** | 多跳问答 | EM, F1 |
| **AlpacaEval** | 指令跟随 | GPT-4o-mini Win Rate |

### ✅ **实验设置和评估指标**

- **生成模型与奖励模型组合**：共使用 **10 种组合**，例如：
  - `Llama-3.1-8B-Instruct` + `AceMath-7B-RM`（用于 MATH）
  - `Qwen3-4B-Instruct` + `RM-Mistral-7B`（用于 AlpacaEval）
- **评估方式**：报告任务层面的真实性能（accuracy、win rate），而非 reward score，避免 reward hacking 偏差。
- **关键超参数**：
  - 路由阈值：$ \theta_{\text{low}} = 0.3, \theta_{\text{high}} = 0.5 $
  - 重生成边界检测间隔：每 $ m=10 $ 步评估一次 reward
  - 最大重生成次数：$ N_{\text{refine}} = 3 $

### ✅ **基线方法对比**

所有方法共享相同的 prompt、生成模型和采样参数，仅解码策略不同：

| 基线方法 | 描述 |
|---------|------|
| **Temperature Sampling (N=1)** | 单次采样，最基础 baseline |
| **Best-of-N (BoN)** | 生成 N 条完整轨迹，选 reward 最高的 |
| **Speculative Rejection (Spec-Rej)** | 实时监控前缀 reward，低于阈值则提前终止 |

> ❗未比较 Reward-guided Speculative Decoding (RSD)，因其依赖更大的 target model，而 SRD 不需要。

---

## 3. **主要实验结果和性能指标**

### ✅ **关键性能数据**

#### 🔹 **总体趋势（图2）**
- SRD 在 **accuracy-compute 曲线上形成新的前沿**，位于 Spec-Rej 和 Best-of-N 之间。
- 在**低计算预算下显著优于 Spec-Rej**，在**中等预算下接近甚至超越 Best-of-N**，但使用更少输出 token。

#### 🔹 **MATH500 上的具体表现（表2）**
| $ N $ | Accuracy | 输出 Token（平均） | SRD vs BoN 效率 |
|-------|----------|--------------------|----------------|
| 10    | 0.544    | 2,166              | 显著更高效 |
| 100   | 0.640    | 21,840             | 达到 BoN 水平，token 更少 |

> ✅ SRD 用远少于 BoN 的 token 数量实现了相近准确率。

#### 🔹 **GPQA 上的表现**
- 所有方法准确率较低（反映任务难度），但 SRD 在**中低预算区间表现更稳定**。
- BoN 在高预算下虽略高，但 token 成本增长过快，收益递减明显。

#### 🔹 **HotpotQA 与 AlpacaEval**
- 在多跳推理任务中，SRD 可有效减少冗余链路生成。
- 在指令跟随任务中，尽管 reward 信号噪声较大，SRD 仍保持稳健提升。

### ✅ **与基线方法的对比结果**

| 对比维度 | 结果 |
|--------|------|
| **vs Best-of-N** | 在相同 accuracy 下，**节省大量输出 token**；在相同 token 预算下，**性能相当或更优** |
| **vs Speculative Rejection** | 在低 compute regime 下**全面胜出**，尤其适合资源受限场景 |
| **vs 温度采样 (N=1)** | 显著提升 accuracy，且成本可控 |

### ✅ **消融实验结果**

#### 🔹 **路由阈值影响（表3）**
- 默认设置 $ (\theta_{\text{high}}, \theta_{\text{low}}) = (0.5, 0.3) $ 表现最佳。
- 过于激进（如 $ \theta_{\text{low}}=0.5 $）会误删可修复轨迹；过于宽松则增加无效重生成。

#### 🔹 **重生成策略对比（表4）**
| 策略 | 特点 | 性能 |
|------|------|------|
| **Reroute (global)**（默认） | 重生成后与其他候选一起排序 | ✅ MATH500 上最优（reward 稳定） |
| **Self-compare** | 仅比较重生成前后 reward 差异 | ✅ GPQA 上更优（reward 噪声大时更鲁棒） |
| **Force keep / Refine-BoN** | 强制保留或多路径重试 | ❌ token 成本飙升，无性能增益 |

> 📌 发现：**更复杂的重生成策略并不带来收益，反而增加开销**。

#### 🔹 **评分间隔（scoring interval）影响（表5）**
- 过小（如1步）易受局部波动干扰，导致过早重生成。
- 过大（如100步）延迟检测退化点，错过修复时机。
- **适度间隔（如10–50步）平衡精度与稳定性**。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **轨迹不应被视为原子单元**：许多“失败”轨迹其实拥有高质量前缀，应被拯救而非丢弃。
2. **局部重生成显著提升效率**：SRD 实现了 **1.28–1.36× 的样本效率增益**，且期望轨迹质量严格更高。
3. **reward model 的校准至关重要**：
   - 在 reward 稳定时，全局排序（reroute）更优；
   - 在 reward 噪声大时，本地比较（self-compare）更鲁棒。
4. **SRD 是通用框架**：可与 speculative decoding、controlled decoding 等方法结合，具备良好扩展性。

### ✅ **方法的局限性**

| 局限性 | 说明 |
|--------|------|
| **依赖固定路由阈值** | 当前使用人工设定的 $ \theta_{\text{low}}, \theta_{\text{high}} $，缺乏自适应能力 |
| **边界检测基于启发式规则** | 再生起点 $ j^* $ 依赖 reward 下降点，可能不够精确 |
| **实现复杂度较高** | 需协调 generator、reward model、editor、router 四个组件，部署成本上升 |
| **reward model 偏见放大风险** | 主动 rewrite 内容可能强化 reward model 的风格/长度偏好，而非事实正确性 |

### ✅ **未来工作方向**

1. **学习路由与边界检测策略**：将 $ \theta_{\text{low}}, \theta_{\text{high}} $ 和 $ j^* $ 的选择端到端学习。
2. **引入 adaptive scoring interval**：动态调整 reward 检查频率。
3. **探索 reward model alignment**：减轻 bias 放大问题，确保优化的是任务正确性而非 proxy reward。
4. **扩展至多模态与规划任务**：应用于视觉推理、agent planning 等长序列决策场景。

---

> 💡 **一句话总结**：  
> **SRD 通过“选择性重生成”打破传统解码中原子轨迹假设，在不增加模型规模的前提下，实现了更高效、更智能的推理路径搜索，开辟了 accuracy-compute tradeoff 的新区域。**

</details>

---

### 10. [Neurosymbolic Alignment for Physiologically-Safe Clinical Language Models](https://arxiv.org/abs/2608.24534)

**Authors**: Abdulhady Abas Abdullah, Erik Cambria, Milena Zivkovic  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.24534v1  

#### Abstract
Clinical LLMs can generate recommendations that are factually plausible yet physiologically unsafe. We investigate whether safety alignment can be improved by grounding preference optimization in structured physiological knowledge rather than text-only supervision. Methods: We propose Neurosymbolic ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Neurosymbolic Alignment for Physiologically-Safe Clinical Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前临床大语言模型（Clinical LLMs）虽然具备良好的医学知识和流畅表达能力，但可能生成**在生理学上不安全**的建议，例如有害的药物组合、违反内稳态（homeostasis）原则或忽略患者特定禁忌症。这类错误并非简单的事实性错误，而是**关系性、上下文依赖性的生理逻辑冲突**，仅靠检索增强（RAG）或基于文本的偏好优化（如DPO、ORPO）难以有效捕捉。

### 提出的新方法与创新思路
作者提出 **Neurosymbolic Alignment（神经符号对齐）** 框架，其核心创新在于：

- **生理世界模型（Physiological World Model）**：构建一个包含847K节点的异构生物医学知识图谱（Biomedical Knowledge Graph），并训练一个基于 **HGNN（Heterogeneous Graph Neural Network）** 的可行性评分器。
- **结构化偏好信号生成**：利用HGNN对候选响应进行生理可行性打分，综合考虑：
  - **Homeostatic Constraints (Shom)**：生命体征和实验室值的生理边界。
  - **Multi-hop Path Plausibility (Spath)**：通过多跳路径验证症状-疾病-药物之间的逻辑连贯性。
  - **Drug-Interaction Penalties (Pint)**：基于知识图谱的药物相互作用风险惩罚。
- **迭代式在线策略对齐（Iterative ORPO）**：在训练过程中，从当前策略动态采样候选响应，用HGNN打分生成偏好对，并执行 **on-policy ORPO 更新**。这避免了静态偏好数据的分布偏移问题。

### 相比现有方法的优势
- **训练时对齐，推理时轻量**：HGNN仅用于训练时生成偏好信号，推理时仅需部署对齐后的LLM，无需实时图查询，保证了部署效率。
- **超越纯文本监督**：将偏好优化的基础从“文本合理性”提升到“生理逻辑一致性”，直接针对临床安全的核心挑战。
- **可扩展且自动化**：无需大量人工标注的偏好对，通过知识图谱自动产生高质量的生理安全偏好信号。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CSB (Clinical Safety Benchmark)**：本文提出的**2,500个场景**的合成基准，专门测试生成式临床推理中的生理约束违规。包含六类场景：
  - 药物-药物相互作用 (25%)
  - 禁忌症检测 (20%)
  - 内稳态违规 (18%)
  - 剂量错误 (15%)
  - 器官功能损害 (12%)
  - 多系统多重用药 (10%)
- **辅助数据集**（用于通用医学能力评估）：
  - MedQA-USMLE, PubMedQA, MedMCQA, MMLU-Medical
- **知识图谱来源**：
  - 集成 UMLS, DrugBank, Reactome, HPO, SIDER, PharmGKB, STRING 等，共 **847,392 个节点，3.2M 条三元组**。

### 实验设置
- **基础模型**：Mistral-7B，采用 LoRA 微调。
- **训练流程**：
  1. 每个查询生成 K=8 个候选响应。
  2. 用 scispaCy + UMLS 进行实体抽取与链接。
  3. HGNN 在子图上计算每个候选的 **PhysioScore**。
  4. 根据得分生成偏好对，执行 **Iterative ORPO** 更新（共5轮迭代）。
- **知识图谱构建**：静态快照，HGNN预训练后冻结，仅用于打分。

### 评估指标
| 指标 | 描述 |
|------|------|
| **CSS (Clinical Safety Score)** | 响应中无生理约束违规的比例（HGNN得分 > 0.85）。 |
| **RSS (Rule-Engine Safety Score)** | 使用独立的 **DrugBank-Rule 引擎**检查，无警报的比例（完全独立于HGNN）。 |
| **HR (Hallucination Rate)** | 医生评审的生成内容中存在生理不可能陈述的比例（专家盲评）。 |
| **DID (Drug Interaction Detection)** | 识别禁忌药物组合的 F1 分数（基于标准标签）。 |
| **MA (Medical Accuracy)** | 在医学问答数据集上的准确率。 |
| **PC (Physiological Consistency)** | 测试响应的平均 HGNN 可行性得分。 |

### 基线方法对比
| 基线方法 | 类型 |
|---------|------|
| SFT, DPO, ORPO | 监督微调与偏好优化 |
| SFT+RAG, SFT+DenseRAG | 检索增强生成 |
| KG-LLM | 静态知识图谱融合 |
| DrugBank-Rule | 基于规则的运行时防护栏 |
| SFT+SelfCorrect | 推理时自修正（生成→规则检查→再生） |
| GPT-4 (Zero-shot & 5-shot) | 闭源模型参考 |
| SFT-CSB | 在CSB训练集上直接SFT |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Ours vs. ORPO）
| 指标 | ORPO | **Ours (Iter-5)** | **提升** |
|------|------|------------------|----------|
| **CSS** | 69.5% | **90.8%** | **+21.3 pp** |
| **RSS** | 65.2% | **86.4%** | **+21.2 pp** |
| **HR** | 14.1% | **5.1%** | **-9.0 pp** |
| **DID** | 72.8% | **91.6%** | **+18.8 pp** |
| **MA** | 70.1% | **75.2%** | +5.1 pp |
| **PC** | 0.71 | **0.89** | +0.18 |

### 与其他方法的对比
- **优于 GPT-4 (5-shot)**：尽管参数量少10倍，但在所有安全指标（CSS, RSS, DID）上均超越 GPT-4。
- **优于 SFT+SelfCorrect**：CSS 高出 **11.4 pp**，表明训练时对齐比推理时自修正更有效。
- **优于 DrugBank-Rule baseline**：不仅在 CSS 和 DID 上更高，RSS 也达到 **86.4%**（vs. 82.4%），说明模型学习到了超越显式规则的隐含安全模式。

### 消融实验结果
| 配置 | CSS | ΔCSS | HR |
|------|-----|------|----|
| **Full Model** | 90.8% | — | 5.1% |
| **w/o HGNN Scoring** | 74.6% | **-16.2 pp** | 12.4% |
| **w/o Iterative Training** | 79.3% | **-11.5 pp** | 9.8% |
| **w/o Homeostatic Constraints** | 82.1% | -8.7 pp | 9.4% |
| **w/o Drug Interaction Penalty** | 84.3% | -6.5 pp | 10.8% |
| **w/o Path Plausibility** | 86.9% | -3.9 pp | 7.2% |

> **结论**：HGNN打分和迭代训练是性能增益的最主要来源。

### 鲁棒性测试
- 在合成的 EHR 风格噪声（实体缺失、缩写歧义等）下，CSS 仍保持 **84.2%**，表现出良好鲁棒性。

---

## 4. 关键结论和发现

### 主要发现
1. **训练时生理对齐显著提升安全性**：将偏好优化建立在结构化的生理知识之上，能有效减少生成内容中的生理不安全行为。
2. **HGNN 是关键组件**：相比纯规则引擎，HGNN 能捕捉多跳、隐含的生理关系，提供更丰富的监督信号。
3. **迭代训练至关重要**：动态生成偏好对能逐步暴露更细微的违规，避免静态数据的局限。
4. **改进非过拟合**：通过 **RSS、DID、HR** 等与HGNN无关的独立指标验证，性能提升是真实有效的，而非对评分器的过拟合。
5. **跨领域泛化好**：在老年科、儿科等高风险领域提升尤为明显。

### 方法的局限性
1. **知识图谱覆盖不全**：约 **4.2%** 的错误源于图谱缺失（尤其在肿瘤学等快速发展的领域）。
2. **缺乏时间动态建模**：无法处理累积毒性、药物洗脱期等时间相关风险。
3. **实体链接脆弱**：Negation（否定）、缩写歧义等问题可能导致状态误判。
4. **评估为合成数据**：所有结果基于 **CSB 合成基准**，尚未在真实电子病历（EHR）上验证。
5. **计算成本高**：Iterative ORPO 训练耗时约为单次ORPO的4倍。

### 未来工作方向
1. **外部验证**：在去标识的真实临床笔记上进行盲评验证。
2. **动态知识更新**：建立 **KG 增量更新机制**（如FDA警报自动注入），解决知识陈旧问题。
3. **混合架构（Hybrid Architecture）**：部署时引入轻量级验证器，对低置信度输出进行二次检查。
4. **高效对齐**：探索将HGNN信号蒸馏到小型评分器，降低训练成本。
5. **时间感知建模**：引入时间序列建模能力，处理动态生理变化。
6. **扩大评估范围**：纳入匹配规模的 PPO/RLHF 系统和工具增强代理（tool-augmented agents）作为基线。

> **最终结论**：Neurosymbolic Alignment 在受控环境下证明了**基于生理知识的训练时对齐**能够显著、可验证地提升临床LLM的安全性，为构建可信医疗AI提供了新范式，但向真实临床部署的转化仍需克服知识动态性、评估真实性等关键挑战。

</details>

---

### 11. [When Personality Meets Quantization: A Layer-wise MBTI Analysis of Quantized LLMs](https://arxiv.org/abs/2608.25977)

**Authors**: Yao Fu, Lijia Huang, Xiaomin Li, Runchao Li, Yu Yin, Kenneth A. Loparo  
**Category**: cs.CL  
**Published**: 2026-08-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.25977v1  

#### Abstract
Personality is increasingly important in large language models (LLMs), as it shapes users' trust, engagement, and emotional experiences. While the Myers--Briggs Type Indicator (MBTI) has emerged as a common framework for assessing LLMs' personality, existing studies focus primarily on full-precision...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# When Personality Meets Quantization: A Layer-wise MBTI Analysis of Quantized LLMs —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前关于 **Large Language Models (LLMs)** 的人格研究主要集中于 **full-precision 模型**，且仅基于最终输出进行评估。然而，在实际部署中，**quantized LLMs**（如 4-bit 或 2-bit）被广泛用于降低内存占用和推理成本。因此，以下关键问题尚未被系统研究：

- **量化是否影响 LLM 的人格特质？**
- **人格判断是如何在模型各层中逐步形成的？**
- **推理过程中的解码策略是否会引发“人格漂移”（personality drift）？**

本文首次系统地探讨了这些问题。

---

### 🚀 提出的新方法与新思路

#### （1）**首个面向量化 LLMs 的 MBTI 人格评估框架**
- 将 **MBTI personality assessment** 形式化为单 token 分类任务（A–G 七级量表），采用 **greedy decoding** 消除采样随机性，提升可复现性。
- 覆盖多种量化方法：主流 **4-bit GPTQ、AWQ** 和极端 **2-bit AQLM** 变体。

#### （2）**引入 Layer-wise 决策动态分析**
- 首次从 **layer-wise entropy** 和 **confidence gap** 角度分析人格决策的演化路径：
  - **Entropy** 衡量选项分布的不确定性；
  - **Confidence gap** 衡量最可能选项与次优选项之间的概率差。
- 揭示人格并非静态属性，而是随网络深度逐渐收敛的动态过程。

#### （3）提出 **Uncertainty-Amplified Layer Decoding (UALD)**
- 一种受 DoLa 启发但专为 **主观任务设计** 的推理机制：
  - 不操作全词表，而是将 logits 投影到 **七个 MBTI 回答选项** 上；
  - 使用 **scaled additive combination** 放大中间层的不确定性信号；
  - 公式：`p(UALD) = log P_mature + λ * log P_premature`
- 通过调节 `λ`（evolution scale）控制对早期层表示的信任程度，从而探测解码过程中的人格稳定性。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | 本工作 |
|------|---------|--------|
| 模型精度 | 仅评估 full-precision | 覆盖 4-bit 与 2-bit 量化模型 |
| 评估粒度 | 仅看最终输出 | 引入 **layer-wise 动态分析** |
| 人格可控性 | 忽略提示工程影响 | 包含 **personality-conditional prompting** |
| 推理扰动 | 忽视解码策略影响 | 提出 **UALD** 主动诱导并测量 personality drift |

> ✅ **优势总结**：更贴近真实应用场景，揭示了人格表现对量化、提示、解码等实现细节的高度敏感性。

---

## 2. 核心实验方法和设置

### 📚 数据集与测试工具
- 使用标准 **60题 MBTI 问卷**（源自 [16personalities.com](https://www.16personalities.com)）
- 每道题提供七种响应选项（A–G），对应不同程度的同意/不同意。
- 所有问题映射到四个 MBTI 二元维度：
  - **E/I**: Extraversion vs Introversion
  - **N/S**: Intuition vs Sensing
  - **T/F**: Thinking vs Feeling
  - **J/P**: Judging vs Perceiving

> 最终 MBTI 类型由每个维度得分符号决定（正为第一项，负为第二项）。

---

### ⚙️ 实验设置

#### （1）模型选择
覆盖多个主流开源 LLM 家族及量化版本：

| 模型家族 | 参数规模 | 量化方法 |
|--------|--------|--------|
| LLaMA3.1 | 8B, 70B | FP16, GPTQ-INT4, AWQ-INT4, AQLM-PV-INT2 |
| Mistral | 7B, 24B | 同上 |
| Qwen2.5 | 14B, 72B | 同上 |

所有模型均来自 Hugging Face 公开仓库。

---

#### （2）两种 Prompting 范式
| 类型 | 描述 |
|-----|------|
| **Unconditional Prompting** | 无显式人格引导，探查模型内在倾向（default personality） |
| **Personality-Conditional Prompting** | 明确指令模型以某 MBTI 类型作答（如 “respond as an ENFJ”），测试其可控性 |

---

#### （3）评估指标
| 指标 | 定义 |
|------|------|
| **MBTI Type Prediction** | 基于 60 题加权汇总得到最终人格类型 |
| **Cross-Precision Agreement** | 量化模型预测结果与原始 FP16 模型的一致性 |
| **Prompt Consistency** | 条件提示下模型是否遵循指定人格 |
| **Layer-wise Entropy** | 各层输出分布的不确定性 |
| **Confidence Gap** | Top-1 与 Top-2 选项的概率差距 |
| **Hamming Distance** | 在 UALD 中衡量 personality drift 程度（最多 4 位不同） |

---

#### （4）基线对比
- **Baseline**：Full-precision FP16 模型作为黄金标准
- 对比不同量化方法（GPTQ, AWQ, AQLM）在相同任务下的表现差异

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据与对比

#### （1）主导人格类型高度一致
- 在 **unconditional prompting** 下，几乎所有模型（无论大小或量化方式）都表现为 **ENFJ**。
- ENFJ 特征：外向、直觉、情感导向、判断型 —— 符合“AI助手”的理想形象（warm, supportive, guidance-oriented）。

> 💡 推测原因：该倾向源于 RLHF、DPO 等后训练过程塑造的“AI Assistant” persona。

---

#### （2）4-bit 量化基本保留人格结构，2-bit 则显著退化
| 指标 | 4-bit (GPTQ/AWQ) | 2-bit (AQLM) |
|------|------------------|-------------|
| Cross-precision agreement | 高（多数保持一致） | 显著下降 |
| Prompt consistency | 较好维持 | 极不稳定，常偏离目标类型 |
| Example: LLaMA3.1-8B | 4-bit 与 FP16 高度一致 | 2-bit 出现频繁 switch（见 Table 12） |

> ❗ 极端压缩破坏了细粒度决策边界，导致人格不可控。

---

#### （3）人格决策是逐层演化的动态过程
- **早期层（~1–21）**：
  - 高 entropy，低 confidence gap → 多个选项概率接近，存在显著不确定性。
- **后期层（~22–32）**：
  - entropy 显著下降，gap 快速上升 → 决策趋于明确，形成稳定人格判断。

> 图形证据见 Figure 1–2：决策承诺（decisional commitment）出现在高层。

---

#### （4）解码策略可诱发 personality drift
使用 **UALD** 调节 `λ` 发现：

| 设置 | 结果 |
|------|------|
| **Unconditional + FP16** | 随 `λ` 增大出现明显 drift（如 ENFJ → ISTP） |
| **ENFJ-conditioned + FP16** | 即使 `λ` 很大仍保持稳定 → **人格对齐增强鲁棒性** |
| **Quantized models** | GPTQ 更早漂移，AWQ 相对稳定；但 2-bit 模型行为不一（部分意外稳定，部分极度敏感） |

> Figure 3–18 展示了不同模型在 evolution scale 下的 Hamming distance 变化。

---

#### （5）模型规模与鲁棒性关系
| 模型规模 | 表现 |
|--------|------|
| **>70B 大模型** | FP16 下极难改变人格（始终为 ENFJ），但量化后反而更容易被 prompt 控制 |
| **<14B 小模型** | 本身对 prompt 更敏感，但 4-bit 保留较好一致性；2-bit 完全失控 |

> 表格支持：Table 7（LLaMA-70B）、Table 3（LLaMA-8B）

---

#### （6）消融实验：条件提示的有效性
- 当 prompt 与内在偏好一致（如 ENFJ-conditioned → ENFJ）时：
  - entropy 下降更快，gap 上升更陡 → 决策更果断。
- 当 prompt 与内在冲突（如 ISTP-conditioned）时：
  - entropy 持续偏高，gap 提升缓慢 → 存在内部张力。

> Figure 2(e–h) 清晰展示这一现象。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLM 的人格不是静态属性，而是一个 emergent、layer-dependent 的决策过程**。
2. **ENFJ 是跨模型家族和量化级别的主导人格类型**，反映“AI助手”角色的普遍设计取向。
3. **4-bit 量化基本保留人格结构**，但在极端 2-bit 压缩下，人格一致性、可控性和跨精度稳定性严重受损。
4. **人格决策始于模糊的中间状态，在深层才逐渐固化**，符合人类自我认知发展的心理学类比。
5. **推理解码策略（如 UALD）可以主动诱导 personality drift**，说明人格归因对实现细节敏感。
6. **人格对齐的提示能增强模型鲁棒性**，减少解码扰动带来的波动。

---

### ⚠️ 方法的局限性

1. **未涵盖所有量化技术**：缺少对 **QAT**（Quantization-Aware Training）和 **PEFT** 方法的研究。
2. **仅限单轮对话场景**：未考虑多轮交互中人格的动态演变。
3. **依赖概率指标，缺乏因果解释**：无法确定为何某些维度（如 E/N）更具顽固性。
4. **局限于 MBTI 框架**：结论可能不适用于 Big Five 等其他人格体系。
5. **未包含闭源商业模型**（如 GPT-4o, Claude, Gemini），外部有效性受限。

---

### 🔮 未来工作方向

1. **扩展至多轮人格追踪**：研究对话历史如何影响人格表达。
2. **结合 mechanistic interpretability**：定位具体神经元或电路负责人格相关判断。
3. **开发人格感知的量化算法**：在压缩过程中保留关键人格特征。
4. **探索人格与安全性的关联**：例如，是否某些人格更易产生幻觉或顺从性（sycophancy）？
5. **构建人格可控的 agent 系统**：实现个性化聊天机器人定制。

---

## 总结一句话

> **LLM 的人格既是训练产物，也是架构、量化、提示与解码共同作用的结果 —— 它不是一个固定的标签，而是一条在层层计算中逐渐成型的行为轨迹。**

</details>

---

### 12. [Probabilistic Performance Analysis of Parallel Signature Search Strategies in Multi-Level Tree Networks](https://arxiv.org/abs/2608.25087)

**Authors**: Jingwei Li, Thomas G. Robertazzi  
**Category**: cs.DC  
**Published**: 2026-08-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.25087v1  

#### Abstract
Hierarchical distributed search, locating a data pattern, or signature, across a tree-structured collection of files, underlies distributed index traversal, deep packet inspection and sequence alignment. A practitioner must decide how much parallelism to employ: scan each layer sequentially, fan out...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Probabilistic Performance Analysis of Parallel Signature Search Strategies in Multi-Level Tree Networks*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文研究了在**多级树网络（multi-level tree networks）**中进行**并行签名搜索（parallel signature search）**的性能预测问题。具体而言，它解决了以下挑战：

- 在搜索开始前，如何**预先（a priori）预测**不同并行策略下的**完成时间（completion time）**。
- 现有分析方法（如 [1]）存在局限：它们假设每个节点都含有签名，或依赖运行时才能获知的信息（如实际签名数量），导致预测不准确或不可用。
- 多签名文件、容量约束、不同并行粒度（层、子树、全树）等因素使建模复杂化。

### 提出了什么新方法或新思路
作者提出了一套**统一的概率性性能分析框架**，具有以下创新点：

- **混合模型（Mixture Model）**：对单个节点的扫描时间建模为“含签名”与“不含签名”两种情况的混合分布，更真实地反映稀疏签名场景。
- **分层条件期望（Conditional Expectation over Layer Width）**：将每层的完成时间基于其父层中含签名节点数 $M_{i-1}$ 进行条件建模，并通过期望平均得到总时间。
- **极端值理论（Extreme-Value Theory, EVT）与中心极限定理（CLT）结合**：用于分析 S4 策略（子树内串行、子树间并行）下，多个子树最大完成时间的统计特性。
- **生成函数法处理已知签名数（Known Counts）**：
  - 对于已知每层签名数的情况，推导出子树间签名分布服从**多元超几何分布（multivariate hypergeometric law）**，而非简单的“星与条”均匀组合。
  - 引入生成函数（generating function）精确计算容量受限下的放置方案数，并区分 **Bose-Einstein (BE)** 和 **Maxwell-Boltzmann (MB)** 两种放置约定。

### 相比现有方法的优势
| 维度 | Prior Work [1] | 本文工作 |
|------|----------------|--------|
| **信息前提** | 部分依赖运行时信息 | 完全 a priori 可计算 |
| **节点时间模型** | 仅用含签名文件统计量 | 使用 presence/absence 混合模型 |
| **容量类** | 单签名 / 无限制 | 统一支持单签名 / K签名 / 无限制 |
| **策略覆盖** | 主要层级并行 | 覆盖五种策略（S1–S5），含子树级并行 |
| **验证方式** | 有限 | 包括 Monte Carlo 仿真、误差量化、原型实测 |

此外，所有公式均附带**精确性标签（exactness label）**，明确标注为 `EXACT`、`PLUG-IN`、`ASYMPTOTIC` 或 `BOUND`，并给出适用范围和误差估计。

---

## 2. 核心实验方法和设置

### 数据集
本研究未使用真实世界数据集，而是基于**合成树结构和随机过程生成的数据**：

- 构造符合 Assumption 1 的多级树网络（高度 $H$，每层扇出 $n_i$）。
- 签名位置服从均匀分布 $U[0,s]$，是否含签名由概率 $p$ 决定。
- 支持两种信息条件：
  - **未知签名数**：通过 $p$ 隐式控制。
  - **已知签名数**：每层签名数 $\{m_i\}$ 预先给定，签名随机分配至节点。

### 实验设置和评估指标
#### 仿真环境
- 开发了一个**离散事件 Monte Carlo 仿真器**，直接实现论文中的模型。
- 每组配置运行 $10^5 \sim 10^6$ 次以获得稳定统计结果。
- 仿真器开源可获取。

#### 评估指标
- **预期完成时间 $T$**：各策略下搜索任务的整体完成时间期望。
- **相对误差**：理论预测值 vs. 仿真结果之间的百分比偏差。
- **峰值并发度 $P_{\text{peak}}$**：执行过程中同时运行的最大节点数。
- **预留成本 $C_{\text{res}} = P_{\text{peak}} \times T$**：衡量资源占用的规划指标。
- **速度提升（Speedup）**：相对于 S1（完全串行）的时间减少倍数。

### 基线方法对比
- **理论基线**：与 [1] 中的方法比较，展示其在低 $p$ 下的乐观偏差。
- **策略间互为基线**：五种策略（S1–S5）相互比较性能差异。
- **不同近似方法对比**：
  - 渐近公式 vs. 数值卷积（如 S4 使用 (10) vs. (13)）
  - Plug-in 近似 vs. 全分布枚举

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 设计示例）
设定：$H=4$, $n=10$, $p=0.5$

| Strategy | Predicted Time ($T$) | Simulated Time | $P_{\text{peak}}$ | $C_{\text{res}}$ |
|----------|------------------------|----------------|--------------------|------------------|
| **S1** (Sequential) | 1170.0 | 1173.8 | 1 | 1170 |
| **S2** (Layer-Parallel) | 4.0 | 4.0 | 1250 | 5000 |
| **S3** (Subtree-Parallel) | 156.0 | 156.5 | 10 | 1560 |
| **S4** (Subtrees-Parallel) | 35.2 | 35.1 | 125 | 4394 |
| **S5** (All-Parallel) | 1.0 | 1.0 | 11110 | 11110 |

> ✅ 所有预测误差 < 0.5%，表明理论模型高度准确。

### 与基线方法的对比结果
- **相比 [1]**：当 $p \to 0$ 时，[1] 将节点时间恒设为 $X = s/2$，低估延迟最多达 **50%**；本文混合模型误差始终 < 0.1%。
- **策略排序**：
  - 时间维度：S5 ≪ S2 < S4 < S3 ≪ S1
  - 成本维度：S1 最省资源，S5 最贵
- **S3 vs. S4 排序可变**：
  - 当 $n=10, H=4$：S4 快于 S3（35.2 vs 156.0）
  - 当 $n=6, H=3$：S3 快于 S4（4.5 vs 3.8），说明深度和宽度影响策略优劣。

### 消融实验结果
#### A. Plug-in Approximation 误差分析
- 在 $pn \geq 2$ 时误差较小（< 2%）
- 在亚临界区 $pn=0.6$ 时误差高达 **74%**
- 建议：当 $pn \leq 2$ 时使用精确公式 (1)

#### B. S4 渐近公式的准确性（Figure 2a）
- 使用 CLT+EVT 的渐近公式 (10)：
  - 当 $n \geq 50, M \geq 10$ 时误差 < 1%
  - 当 $n=4, M=5$ 时误差达 3.3%
- 推荐自动切换机制：小参数用数值卷积 (13)，大参数用渐近公式

#### C. 已知签名数下的分布选择影响（Figure 2b）
- **多元超几何分布**（正确模型）完美匹配仿真。
- “星与条”均匀组合模型高估空子树概率约 **5 倍**，导致层时间预测偏高 **4%**。
- BE 与 MB 约定下，关键概率（如满载概率）相差可达 **31%**，说明选择至关重要。

#### D. 多核原型测试结果（Table 3）
硬件：Apple M1（8核），Python multiprocessing  
设定：$H=3, n=6, p=0.5$

| Strategy | Measured Speedup (W=8) | Analytical Prediction |
|---------|-------------------------|------------------------|
| S2      | 3.69                    | 19.5×                  |
| S3      | 3.03                    | 4.5×                   |
| S4      | 3.02                    | 3.8×                   |
| S5      | 4.30                    | 58.5×                  |

> ⚠️ 实际速度远低于理想预测，受制于：
> - 仅 8 个物理核心
> - 同步开销（barrier cost）
> - S3 与 S4 实测性能几乎无法区分，尽管理论预测差 20%

---

## 4. 关键结论和发现

### 论文的主要发现
1. **首次实现了完全 a priori 的并行签名搜索性能预测框架**，适用于多种策略、信息条件和容量类。
2. **混合时间模型显著提升精度**，尤其在签名稀疏场景下避免了传统方法的严重低估。
3. **S4 策略性能介于 S2 与 S5 之间**，其加速比约为 $O(\sqrt{n \log M})$，接近但略逊于完全并行。
4. **多元超几何分布是已知签名数下的正确子树占用模型**，“星与条”假设会导致系统性偏差。
5. **理论预测极其高效**：Table 2 中所有计算耗时 **< 1ms**，而同等 Monte Carlo 估计需数分钟。
6. **同步开销可能抹平理论上的微小差距**：S3 与 S4 在真实系统中可能表现相近，需结合调度结构判断。

### 方法的局限性
- **处理器无限假设**：理论模型忽略核心数量限制和调度开销，在真实系统中会显著降低性能。
- **同构子树假设**：所有子树大小相同，难以推广到异构拓扑。
- **无取消机制（No Cancellation）**：即使某子树可剪枝，仍完成全部扫描，未考虑动态撤销任务的成本节省。
- **开放问题**：在已知签名数且 $m \geq rK$ 的情况下，S4 层时间尚无闭式解（见 Remark 9）。

### 未来工作方向
1. **扩展至有限处理器池（bounded processor pool）**：引入 $W$ 个 worker 的调度模型，考虑 barrier 和 dispatch cost。
2. **闭环优化设计**：结合本文的时间模型与资源成本，构建 joint time-cost objective，进行树形结构与策略联合优化（已在 companion paper 中展开）。
3. **支持异构扇出 $n_i$** 和非均匀签名分布。
4. **集成取消机制**：允许 coordinator 在发现所有签名后提前终止部分任务。
5. **闭合开放问题**：推导容量受限下的最小计数分布，完善 S4 在高负载下的分析。

--- 

> 📌 **总结一句话**：  
> 本文建立了一个**高效、精确、可解释**的 a priori 性能预测框架，为多级树网络中的并行签名搜索提供了从理论建模到实际部署的完整决策支持体系。

</details>

---

### 13. [SHSP: Structure-Aware Hierarchical Solution Prediction for Mixed-Integer Linear Programming](https://arxiv.org/abs/2608.25282)

**Authors**: Zherong Zhang, Guanlin Li, Chengrui Gao, Haopu Shang, Ke Xue, Jixiang Lu, Weiyong Yang, Chao Qian  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.25282v1  

#### Abstract
Mixed-Integer Linear Programming (MILP) is a fundamental optimization paradigm in combinatorial optimization and has been widely applied across real-world domains. Due to its NP-hard nature, obtaining optimal solutions for large-scale or highly constrained MILP instances remains computationally proh...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SHSP: Structure-Aware Hierarchical Solution Prediction for Mixed-Integer Linear Programming

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现有的基于学习的 **Mixed-Integer Linear Programming (MILP)** 解决方案预测方法大多采用 **one-shot prediction** 范式，即通过图神经网络（GNN）一次性并行预测所有变量的边际概率。这种方法存在以下关键缺陷：

- **忽略变量间的条件依赖关系**：MILP 中的变量通过约束高度耦合，最优赋值具有强组合结构性质，而 one-shot 方法在解码阶段将变量视为独立，导致预测结果可能相互冲突。
- **过度依赖 GNN 表征能力**：复杂的组合结构完全由 GNN 编码器隐式建模，难以充分捕捉长程依赖。

### 🚀 提出的新方法与创新思路

作者提出 **SHSP (Structure-Aware Hierarchical Solution Prediction)** 框架，其核心思想是：**显式建模变量间的结构依赖，并通过分层条件解码机制进行顺序预测**。

#### 主要创新点包括：

1. **Variable Coupling Graph 构造**
   - 基于约束系数和期望违反程度构建加权无向图，量化变量对之间的耦合强度。
   - 边权重由两个因素决定：
     - **Expected Violation**：衡量一对变量单独作用下违反约束的程度。
     - **Coefficient Importance**：归一化后的系数乘积，反映其在约束中的相对重要性。

2. **Hierarchical Conditional Decoding**
   - 定义每个变量的 **Variable Coupling Score (VCS)** 为与其相连边的权重之和。
   - 将变量按 VCS 升序划分为多个层次 $ H_1, H_2, ..., H_K $，从弱耦合到强耦合依次预测。
   - 每一层的预测以之前层次已固定的赋值作为上下文条件输入，实现**逐步推理**。

3. **Confidence-aware Mask-and-Repair 机制**
   - 在每步预测后，低置信度变量被随机掩码（mask），避免错误传播。
   - 所有层次解码完成后，对被掩码的变量集合 $ R $ 进行一次集中修复（repair），利用完整上下文重新预测。

4. **Structure-Aware Variable Fixing Strategy**
   - 不仅考虑预测置信度，还优先固定高 VCS 的强耦合变量。
   - 因为这些变量一旦确定，可通过共享约束显著收紧其他变量的可行域，从而更有效地缩小搜索空间。

### 🔍 相比现有方法的优势

| 方面 | 传统 One-Shot 方法 | SHSP |
|------|------------------|-------|
| 解码方式 | 并行、独立预测 | 分层、条件依赖预测 |
| 结构建模 | 隐式（全靠 GNN） | 显式（构造 coupling graph） |
| 错误控制 | 无机制 | Mask-and-Repair 减少累积误差 |
| 变量选择策略 | 仅基于置信度 | 结合结构影响（VCS）与置信度 |

> ✅ **本质区别**：SHSP 将“结构感知”从编码阶段延伸至整个预测流程，形成一种全新的 **conditional hierarchical prediction paradigm**，而非简单改进模型架构。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

在四个标准 MILP 基准上进行评估：

| 数据集 | 应用场景 | 变量数 | 约束数 | 类型 |
|--------|--------|--------|--------|------|
| **CA (Combinatorial Auctions)** | 组合拍卖 | 1500 | 2592 | 二元变量为主 |
| **WA (Workload Appointment)** | 工作负载调度 | 61000 | 64318 | 大规模混合整数 |
| **IP (Item Placement)** | 物品放置优化 | 1083 | 195 | 实际工业问题 |
| **SC (Set Covering)** | 集覆盖问题 | 5000 | 3000 | 经典组合优化 |

此外，在 **MIPLIB 子集 IIS** 上验证泛化能力。

### ⚙️ 实验设置

- **训练/验证/测试划分**：240 / 60 / 100 实例
- **下游求解器**：Gurobi 和 SCIP
- **时间限制**：1000 秒（主实验）
- **训练数据生成**：使用 Gurobi 在 3600 秒内收集前 50 个高质量解用于监督训练

### 🎯 评估指标

- **OBJ**：找到的最佳目标值
- **Absolute Primal Gap (Gapabs)**：  
  $$
  \text{Gapabs} = |\text{OBJ} - \text{BKS}|
  $$
  其中 BKS 是单线程 Gurobi 在 3600 秒内找到的最优解（Best Known Solution）
- **Relative Reduction in Gapabs**：相对于基线的差距缩减比例

### 🆚 对比的基线方法

| 类别 | 方法 |
|------|------|
| **传统求解器** | Gurobi, SCIP |
| **学习引导框架** | Neural Diving (ND), Predict-and-Search (PaS), Apollo-MILP |
| **变体对比** | 各框架 + SHSP 替换预测器（如 PaS+SHSP） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（使用 Gurobi 下游，1000s 时间限制）

| 方法 | CA Gapabs ↓ | WA Gapabs ↓ | IP Gapabs ↓ | SC Gapabs ↓ | 平均 Gapabs 缩减 |
|------|-------------|-------------|-------------|-------------|------------------|
| Gurobi (1000s) | 1081.73 | 0.21 | 0.47 | 0.24 | — |
| PaS | 597.50 | 0.21 | 0.37 | 0.19 | — |
| **PaS+SHSP** | **1.52** | **0.13** | **0.28** | **0.11** | **99.7% → 38.1% → 24.3% → 42.1%** |
| Apollo | 491.50 | 0.21 | 0.28 | 0.12 | — |
| **Apollo+SHSP** | **0.00** | **0.15** | **0.02** | **0.07** | **100.0% → 28.6% → 92.9% → 41.7%** |

> 💡 **亮点发现**：
> - 在 **CA 数据集上，Apollo+SHSP 实现了 Gapabs = 0.00**，意味着找到了超越 3600 秒 Gurobi 的更优解！
> - **平均绝对原始间隙减少达 54%**（文中强调）。
> - 即使在大规模 WA 和复杂 SC 上也保持稳定提升。

### 🔬 消融实验结果（Ablation Study）

#### (1) **Mask-and-Repair 机制有效性**

| 方法 | CA Obj | IP Obj |
|------|--------|--------|
| PaS+SHSP (None) | 98380.92 | 12.12 |
| PaS+SHSP (Mask only) | 98416.55 | 12.17 |
| **PaS+SHSP (Mask+Repair)** | **98487.72** | **11.97** |

> ✅ **结论**：仅 mask 会丢失信息，必须配合 repair 才能有效纠正错误，两者相辅相成。

#### (2) **Hierarchical Prediction vs 原始 GNN**

| 方法 | CA Obj | IP Obj |
|------|--------|--------|
| PaS (原 GNN) | 97891.74 | 12.06 |
| **PaS+SHSP (Hierarchical)** | **98487.72** | **11.97** |

> ✅ **结论**：分层预测本身就能带来显著增益，说明结构感知解码优于并行预测。

#### (3) **Structure-Aware Fixing 策略效果**

| 方法 | CA Obj | IP Obj |
|------|--------|--------|
| PaS+SHSP (Original Fixing) | 98061.51 | 12.34 |
| **PaS+SHSP (Structure-Aware Fixing)** | **98487.72** | **11.97** |

> ✅ **结论**：结合结构影响力的变量固定策略能进一步释放性能潜力。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **One-shot 预测范式存在根本局限**：无法有效建模变量间复杂的条件依赖关系。
2. **SHSP 成功引入了结构感知的分层预测机制**，通过 variable coupling graph 显式指导解码顺序，显著提升了预测准确性。
3. **Mask-and-Repair 机制有效缓解了多步推理中的误差累积问题**。
4. **Structure-aware fixing 利用了拓扑结构信息**，使得搜索空间缩减更加高效。
5. **SHSP 是通用插件式框架**，可无缝集成到 ND、PaS、Apollo 等多种 learning-guided search 方法中，且一致取得大幅性能提升。

> 🌟 **最惊人结果**：在 Combinatorial Auctions 上，**SHSP 在远小于 3600 秒的时间预算内，找到了比完整运行 Gurobi 更好的解**，表明其不仅能加速搜索，还能引导求解器发现更高品质的局部最优。

### ⚠️ 方法的局限性

- **图构建开销**：虽然平均 <15 秒，但在极端大规模实例上可能成为瓶颈（如 SC 达 15.23s）。
- **超参数敏感性**：如层次数 $ K $、mask 概率函数、VCS 阈值等需调优。
- **当前 coupling score 为启发式设计**，未端到端学习。
- 对非二元整数变量的支持仍需扩展（目前聚焦 binary variables）。

### 🔮 未来工作方向（作者指出）

1. **端到端学习 variable coupling scores**，替代手工设计的 heuristics。
2. **扩展至更广泛的 problem classes**，如非线性规划、随机规划等。
3. 探索 **dynamic graph updating** 机制，在迭代过程中更新 coupling graph。
4. 结合 **large language models** 或 **diffusion models** 进一步增强生成能力。

---

## 总结

> **SHSP 提出了一种革命性的结构感知分层预测范式，打破了传统 one-shot 预测的局限，在多个标准 MILP 基准上实现了高达 100% 的 gap 缩减，甚至发现了超越商业求解器的更优解。它不仅是技术上的改进，更是对“如何利用机器学习理解组合结构”的一次深刻探索。**

</details>

---

### 14. [Beyond Scaling: Self-Evolving LLM Agents for Hardware Kernel Optimization via an Experience-Driven Workflow and Experience Graph Memory](https://arxiv.org/abs/2608.25570)

**Authors**: Siyuan Chen, Runlin Hou, Shenxiu Wu, Yansong Sun, Junming Cao, Yiyu Zhang, Shudi Shao, Junhao Qiu, Zhichao Lu, Qingfu Zhang  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.25570v1  

#### Abstract
Hardware kernel optimization requires repeated compilation, correctness testing, profiling, and revision. LLM agents can automate parts of this process, and stronger foundation models, longer context windows, and longer execution horizons have improved optimization within individual tasks. These adv...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Beyond Scaling: Self-Evolving LLM Agents for Hardware Kernel Optimization via an Experience-Driven Workflow and Experience Graph Memory  
**——核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
硬件 kernel 优化是一个高度依赖专家经验的过程，涉及反复的编译、正确性验证、性能剖析和修改。尽管当前的 **LLM agents** 已能通过更强的 foundation model 和更长上下文窗口提升单次任务内的优化能力，但它们普遍缺乏从已完成优化运行中**持续学习并复用执行反馈**的能力。

现有方法存在两大缺陷：
- **不保留决策证据**：很少系统化地保存“某个决策 → 观察到的结果”这一因果链，导致每次优化都是“从头开始”。
- **历史轨迹冗余且低效**：简单保留所有历史会占用大量 context 空间，反而稀释当前任务的关键信息（attention dilution）。

### 🚀 提出的新方法：KOPE 框架
本文提出 **KOPE**（Knowledge-Oriented Programming Environment），一个面向硬件 kernel 优化的**经验驱动型 LLM agent 框架**，其核心是让 agent 在探索过程中自我演化（self-evolving），即在固定 foundation model 参数的前提下，通过外部记忆机制实现知识积累与迁移。

#### 主要创新点：
1. **Experience-Driven Optimization Workflow**  
   构建闭环 agent 工作流，每轮生成候选 kernel 后立即进行 correctness 测试与 performance profiling，并将完整轨迹记录为可复用的经验。

2. **Experience Graph Memory (KOPE-Mem)**  
   将优化过程中的决策、观测结果（如编译错误、性能增益）、替代分支等组织成有向图结构（Directed Acyclic Forest）。每个节点是一个 `Case`（JSON 结构化记录）或 `Journal`（Markdown 叙述性日志），边表示执行顺序（provenance link），支持基于下游结果的检索排序。

3. **Active Context Management and Injection**  
   在每次推理前动态构建 prompt，采用三级 context 分层策略：
   - **Hot Context**：必需的任务状态（specification、incumbent code、最新诊断）
   - **Warm Context**：当前状态下检索出的相关历史经验（成功动作、失败模式）
   - **Cold Context**：通用领域知识（文档、跨类别经验）
   
   在固定 token budget 下智能选择并压缩最相关的内容注入 prompt，避免全量历史拖累效率。

### 🔍 相比现有方法的优势
| 维度 | 现有方法 | KOPE |
|------|--------|-------|
| **知识留存** | 无或仅短期记忆 | 长期结构化记忆（图谱） |
| **经验复用** | 限于单任务内迭代 | 支持跨任务、跨 operator 复用 |
| **上下文利用** | 被动填充历史 | 主动筛选+注入高价值片段 |
| **资源效率** | 上下文膨胀严重 | 固定预算下高效利用 |

---

## 2. 核心实验方法和设置

### 📊 数据集与平台
- **基准测试集**：`CANN Bench v0.4.0`，包含 **53 个 AscendC operators**，共 **1,060 个测试案例**（每个 operator 20 cases）
- **硬件平台**：华为 Ascend 910C NPU
- **评估环境**：提供完整的编译、功能测试、性能测量流程，防止 reward hacking（如 CPU fallback）

### ⚙️ 实验设置
- **模型统一**：主实验使用 `GLM-5.2` 和 `Deepseek-V4-Pro`，均支持 **1M token 上下文窗口**，确保公平比较。
- **token budget 固定**：所有配置共享相同 context 容量。
- **输出完整性要求**：缺失 operator 按 20 个失败 case 计入总分。

### 📈 评估指标
| 指标 | 定义 |
|------|------|
| **Case Pass Rate** | 成功通过 final evaluator 的 case 数 / 1060 |
| **Operator Pass** | 至少有一个 case 通过的 operator 数量 |
| **Geometric Mean Speedup** | 对正向 speedup 字段取几何平均（排除负值/缺失） |
| **Token Consumption** | 整体优化过程消耗的总 tokens |
| **Exact Intersection Analysis** | 在两个配置都返回有效结果的交集上做配对性能对比 |

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **CANNBot** | 华为官方 baseline，基于 CANN 示例库和优化实践 |
| **CUDA-Agent** | 当前最强 GPU kernel 优化 agent（来自 Dai et al., 2026），具备强化学习与 ReAct 循环，用于检验跨硬件泛化能力 |

---

## 3. 主要实验结果和性能指标

### 📊 系统级性能对比（GLM-5.2）

| Method | Case Pass | Pass Rate | Operator Pass | Speedup (GeoMean) | Score |
|--------|-----------|------------|------------------|---------------------|--------|
| **KOPE** | **897** | **84.6%** | **52/53** | **1.54× vs CANNBot** | **2004.49** |
| **CANNBot** | 613 | 57.8% | 37/53 | 1.00× | 1465.93 |
| **CUDA-Agent** | 156 | 14.7% | 13/53 | 0.100× (below parity) | 312.00 |

> ✅ **KOPE 显著领先**：相比最强 baseline CANNBot，pass rate 提升 **26.8 pp**，score 提升 **+36.7%**；而 CUDA-Agent 几乎无法覆盖完整 suite。

### 🔬 消融实验一：Active Context Management & Injection（GLM-5.2）
> 对比“被动上下文构建” vs “主动管理+注入”

| Context Policy | Case Pass | Pass Rate | Score | Speedup (GeoMean) | Token Usage |
|---------------|-----------|------------|--------|--------------------|-------------|
| Passive | 636 | 60.0% | 636.00 | 0.0382× | **15.9B** |
| **Active** | **897** | **84.6%** | **2004.49** | **0.0661× (+1.73×)** | **1.113B (-93.0%)** |

> ✅ **主动管理大幅提升效果且极度节省 token**：在相同模型下，pass rate ↑24.6%，token 消耗 ↓93%，说明并非靠堆历史取胜。

### 🔬 消融实验二：Experience Graph Memory（GLM-5.2）
> 对比启用 vs 禁用图记忆，在 412 个配对有效 timing 案例上的表现

| Configuration | Case Pass | Pass Rate | GeoMean Speedup (Valid Pairs) |
|---------------|-----------|------------|-------------------------------|
| Without Graph | 585 | 55.2% | — |
| **With Graph** | **897** | **84.6%** | **1.434×** |

> ✅ 图记忆使 pass rate 提升 **29.4 pp**，在有效配对中带来 **1.43× 性能加速**，尤其在复杂级别 L2–L4 上优势显著（最高达 9.175×）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **模型能力 ≠ 优化成功**：即使使用强大的 foundation model（如 GLM-5.2 或 Deepseek-V4-Pro），若缺乏目标硬件上的经验积累，仍难以实现广泛的功能覆盖（如 CUDA-Agent 仅通过 ~14%）。
2. **外部经验可弥补数据稀缺**：在 public training data 极少的新硬件平台上（如 Ascend），**通过 Experience Graph Memory 积累 target-side 执行反馈** 是提升优化成功率的关键。
3. **主动上下文管理优于被动记忆**：不是“越多越好”，而是“越准越好”。**Active Context Management** 能在有限 token 内精准注入高价值经验，显著提升效率与性能。
4. **KOPE 具备跨栈泛化潜力**：附录中在 **RISC-V + Triton** 平台上的实验表明：
   - Ascend 上的历史经验可帮助提升 K3 上的 **pass coverage（+23 pp）**
   - 本地 target-side 学习才能恢复性能（combined memory 达 0.786× PyTorch）
   - 支持“源端可行性提示 + 目标端性能调优”的协同范式

### ⚠️ 局限性
- **单次运行验证**：所有结果基于一次归档运行，缺乏 run-to-run variance 估计。
- **未开放代码与完整 trace**：部分分析受限于日志粒度（如缺少重复 timing sample）。
- **图谱更新策略静态**：context tier caps（Qw, Qc）为手动设定，尚未实现在线学习调整。

### 🔮 未来工作方向
- 动态学习 context 注入策略（adaptive cap tuning）
- 引入 causal reasoning 提升图谱推理能力
- 探索多硬件联合训练与迁移学习框架
- 将 KOPE-Mem 应用于其他系统级编程任务（如编译器优化、OS 调参）

---

## ✅ 总结一句话
> **KOPE 证明了：在 foundation model 不变的情况下，通过 Experience Graph Memory 和 Active Context Management，LLM agents 可以在硬件 kernel 优化这类数据稀疏任务中实现持续自我进化，显著超越仅依赖模型规模扩展的方法。**

</details>

---

### 15. [Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged Attention](https://arxiv.org/abs/2608.23834)

**Authors**: Sergii Kozyrev (Minima AI, Inc), Davyd Maiboroda (Minima AI, Inc)  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.23834v1  

#### Abstract
The key-value (KV) cache is a primary capacity and bandwidth bottleneck in long-context LLM serving. We present Minima-KV, a retention-preserving hierarchy for mixed-format paged attention. Recent and protected Anchor pages remain in FP8, while older non-anchor pages move to packed TQ3; every live-r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged Attention  
**——核心结论与实验结果总结**

---

## 1. 主要贡献和创新点

### ✅ 解决的问题
在长上下文大语言模型（LLM）服务中，**Key-Value (KV) 缓存容量和内存带宽**成为主要瓶颈。传统方法如均匀量化、稀疏保留或低秩表示虽能压缩，但面临以下挑战：
- **均匀量化**浪费比特于不重要位置；
- **基于重要性的淘汰机制**可能导致未来需要时无法恢复；
- **结构性压缩**可能引入高昂的重构开销。

### 🚀 提出的新方法：Minima-KV
Minima-KV 是一种**保留完整性的分层 KV 缓存压缩系统**，结合了混合格式分页注意力（Mixed-Format Paged Attention）与动态生命周期管理，其核心思想是将 KV 页面划分为三个保真度层级：

| 层级 | 格式 | 特性 |
|------|------|------|
| **Recent** | FP8 | 新生成的页面，高保真存储，避免频繁解压 |
| **Anchor** | FP8 | 保护关键历史内容（如指令、检索证据等），防止降级 |
| **Stale** | TQ3 | 老旧非锚点页面使用 3-bit 量化（TurboQuant-inspired），大幅降低存储 |

该系统实现了：
- 所有活跃请求的逻辑页面始终可寻址（no deletion）；
- 支持跨格式的在线 Softmax 合并，无需构建全量密集缓存副本（no dense shadow）；
- 利用 TileLang 编写的专用 CUDA 内核实现高效异构解码。

### 🔍 相比现有方法的优势
| 对比维度 | Minima-KV | 其他方法（如 H2O, SnapKV, KIVI） |
|--------|----------|-------------------------------|
| **保留完整性** | ✅ 所有页面保留近似值 | ❌ 可能删除或不可逆丢弃 |
| **精度控制灵活性** | ✅ 分层混合格式（FP8 + TQ3） | ⚠️ 多为统一量化或二元保留/丢弃 |
| **运行效率** | ✅ 异构内核直接融合输出，无 dense shadow | ⚠️ 需反量化或重建才能参与 attention |
| **生产适用性** | ✅ 支持 CUDA Graph、所有权协议、前缀复用 | ⚠️ 多停留在研究原型阶段 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **RULER NIAH**：8 个“大海捞针”任务，在 4K/8K/16K 上下文中测试定位能力，每任务 32 示例。
- **LongBench v2**：503 道多选题，涵盖六大现实场景：
  - 单文档问答
  - 多文档问答
  - 长上下文学习
  - 对话理解
  - 代码库理解
  - 结构化数据理解  
  测试长度覆盖 16K、32K、64K。

### ⚙️ 实验设置
- **模型**：Qwen3.6-27B（仅语言路径）
- **硬件**：单张 NVIDIA RTX PRO 6000 Blackwell GPU（96GB GDDR7）
- **上下文管理**：PagedAttention，逻辑页大小为 1,792 tokens
- **评估模式**：
  - **质量评测**：逐样本配对比较（paired evaluation），启用动态重分级（retiering），关闭 attention scoring
  - **性能评测**：双活跃请求，各含 59,008 prompt tokens，静态 FP8/TQ3 页面分布，启用 CUDA Graph 回放

### 📊 评估指标
| 类型 | 指标 |
|------|------|
| **存储效率** | 每 token KV 存储字节数（KiB/token）、相对 BF16/FP8 压缩比 |
| **推理吞吐** | 解码吞吐量（tok/s）、步长时间（ms）、相对吞吐比 |
| **准确性** | RULER NIAH 任务宏平均准确率、LongBench v2 正确数及变化量（delta） |
| **系统能力** | 并发上下文容量预测（analytical estimation） |

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **BF16 KV** | 全精度 KV 缓存，64 KiB/token |
| **FP8 KV** | 当前主流生产基线，32 KiB/token |
| **Three-tier Minima-KV** | 本文方法：Recent/Anchor-FP8 + Stale-TQ3 |
| **Dense Control** | 密集缓存策略对照组（dtype 未绑定） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### （1）KV 存储压缩效果（部署统计）
| 指标 | 数值 |
|------|------|
| **实际占用** | **18.3 KiB/token**（owner-reported aggregate） |
| **相对于 BF16 压缩比** | **3.50×**（理论 3.497×） |
| **相对于 FP8 压缩比** | **1.75×**（理论 1.749×） |

> 💡 在百万 token 场景下，Minima-KV 仅需 17.45 GiB，而 BF16 需 61.04 GiB，FP8 需 30.52 GiB。

#### （2）质量评估结果（Materializing Profile）
| 任务 | Dense 控制 | Minima-KV | Delta |
|------|-----------|------------|-------|
| **RULER NIAH @4K** | 97.27% | 93.75% | **-0.90 pp** |
| **RULER NIAH @8K** | 99.80% | 100.00% | **+0.20 pp** |
| **RULER NIAH @16K** | 100.00% | 100.00% | **±0.00 pp** |
| **LongBench v2 @16K** | 190 正确 | 186 正确 | **-0.80 pp** |
| **LongBench v2 @32K** | — | — | **-0.60 pp**（paired delta） |
| **LongBench v2 @64K** | — | — | **-0.40 pp**（paired delta） |

> ✅ 在较长上下文（≥8K）中表现更稳定，误差随长度增加而收敛。

#### （3）性能评估结果（Direct Canary）
| 指标 | Dense 控制 | Minima-KV | Ratio |
|------|-----------|------------|-------|
| **解码吞吐** | 29.804 tok/s | 29.270 tok/s | **0.9821×** |
| **Step Time** | 42.535 ms | 43.311 ms | ~1.8% 延迟上升 |
| **Active KV 字节** | 477,102,080 B | 131,610,624 B | **3.625× 压缩** |
| **Layer Routes** | — | 16/16 成功路由 | 无 fallback |
| **Dense Shadow** | 存在 | **已释放** | ✅ 无额外内存负担 |

> ✅ 实现了接近原生性能的高压缩比推理，且所有 16 层均通过异构路径处理。

#### （4）并发容量分析（Analytical Estimation）
假设总内存预算 $ M = 86.4\,\text{GiB} $，权重 $ W = 25.15\,\text{GiB} $，固定状态 $ D = 0.153\,\text{GiB} $

| 上下文长度 | BF16 支持并发 | FP8 支持并发 | Minima-KV 支持并发 |
|------------|----------------|----------------|---------------------|
| 32K        | 28             | 53             | **84**              |
| 64K        | 14             | 28             | **47**              |

> ✅ 在相同条件下，Minima-KV 可支持 **1.58× 更高的并发数**（相比 FP8）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Minima-KV 实现了高保真的 KV 压缩**：通过分层保真设计，在不删除任何活跃页面的前提下，实现高达 **3.5× 的 BF16 压缩比**。
2. **质量损失可控且随长度收敛**：在 4K 上略有下降（-0.9pp），但在 8K 和 16K 表现持平甚至略优；LongBench v2 的误差从 16K 的 -0.8pp 下降到 64K 的 -0.4pp，显示更强的长程鲁棒性。
3. **高性能异构解码可行**：通过定制化的混合格式 attention 内核，可在无 dense shadow 的情况下完成全局 softmax 合并，**吞吐达 dense 控制的 98.2%**。
4. **显著提升服务并发能力**：理论上可将 32K 上下文的并发承载能力从 FP8 的 53 提升至 84，提升约 58%。

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **单一模型 & 硬件验证** | 仅在 Qwen3.6-27B 和 Blackwell GPU 上验证，泛化性待检验 |
| **aggregate 数据缺乏细粒度拆解** | 18.3 KiB/token 为整体上报值，未提供各层级占比（$p_R, p_A, p_S$） |
| **profile 分离** | 质量、性能、前缀复用来自不同配置，无法在同一实验中同时验证全部特性 |
| **attention scoring 未启用** | 尽管支持反馈机制，但当前实验中关闭，影响动态 anchor 管理潜力 |
| **无置信区间与重复性报告** | 性能仅为单次运行结果，缺乏统计显著性分析 |

### 🔮 未来工作方向
1. **四层扩展（Stale2）**：提出未来可引入基于共享张量骨干（shared tensor backbone）的 **Stale2 层**，用于最冷门页面的结构化压缩（如 Tucker 分解 + adapter），进一步压缩空间。
2. **闭环评分机制集成**：启用并验证基于 attention score 的动态 promotion/demotion 控制器，实现更智能的 anchor 管理。
3. **端到端服务质量建模**：开展开放负载实验，量化 KV 压缩对 TTFT、p95/p99 延迟的实际影响。
4. **多 GPU 与 Tensor Parallel 支持**：扩展至分布式环境下的协同压缩与同步协议。

---

> 📌 **总结一句话**：  
> **Minima-KV 通过保留完整性的三层次 KV 分层架构（FP8 + TQ3）与混合格式 attention 内核，在几乎不牺牲推理质量的前提下，实现了高达 3.5× 的 KV 缓存压缩和接近原生性能的解码吞吐，为长上下文 LLM 高效部署提供了实用解决方案。**

</details>

---

### 16. [Data Mixing as Mixture Experiment: Response Surface Methodology and Optimal Design for Large Language Model Pretraining](https://arxiv.org/abs/2608.23922)

**Authors**: Yicheng Mao, Hongru Du  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.23922v1  

#### Abstract
Data mixing is a central design problem in large language model pretraining: given a fixed token budget, practitioners must decide how much data to allocate to each domain. Recent proxy-based methods address this problem by training small models on candidate mixtures, fitting a response model, and u...

---

### 17. [Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping](https://arxiv.org/abs/2608.24135)

**Authors**: Yiwen Zhang, Xiaodong Yan, Zhenyu Huang, Deng Zhao, Liang Jiang, Qing Cui, Zujie Wen, Zhiqiang Zhang, Jun Zhou  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.24135v1  

#### Abstract
Reinforcement learning from verifiable rewards (RLVR) has emerged as a pivotal technique for enhancing the code generation capabilities of Large Language Models (LLMs). However, the efficacy of RLVR in coding implementations is fundamentally limited by the comprehensiveness of test cases, because in...

---

### 18. [MetaRAG: Belief-Action Aligned Policy Optimization for Agentic RAG](https://arxiv.org/abs/2608.24214)

**Authors**: Qiuyi Qi, Tian Liang, Jiamu Wang, Jinjian Zhang, Wei Zhou, Pengcheng Zhu, Linjian Mo, Ming Kong, Jie Liu, Qiang Zhu  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.24214v1  

#### Abstract
Agentic retrieval-augmented generation (RAG) requires language models to decide when to continue searching and when to answer. Existing RL-based methods rely on external supervision and overlook the agent's internal belief about whether the current evidence is sufficient. To address this problem, we...

---

### 19. [ResiSpec: Enhancing Multi-Candidate Speculative Sampling via Residual Distribution Shaping](https://arxiv.org/abs/2608.24411)

**Authors**: Zhi-Kai Chen, Jun-Jie Tao, Wei-Xiang Mao, De-Chuan Zhan, Han-Jia Ye  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.24411v1  

#### Abstract
The efficiency of Large Language Model (LLM) serving is fundamentally limited by the sequential nature of autoregressive decoding. Speculative Decoding (SD) mitigates this by using a lightweight draft model to speculate future tokens, which are then validated by the LLM in a single parallel forward ...

---

### 20. [Reinforcement Learning-Guided Evolutionary Policy Optimization for Preference-Adjustable Heterogeneous Agile Earth Observation Satellite Scheduling](https://arxiv.org/abs/2608.24470)

**Authors**: He Wang, Junyu Wu, Hui Li, Yanjie Song, Witold Pedrycz, Liang Li  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.24470v1  

#### Abstract
Heterogeneous agile Earth observation satellite (AEOS) scheduling requires task selection, satellite assignment, and observation sequencing under satellite-dependent visibility windows, attitude maneuvering requirements, energy consumption, and onboard storage constraints. Since satellites differ in...

---

### 21. [PeakBench: Benchmarking Resource-Aware Tool Invocation in LLM Agents](https://arxiv.org/abs/2608.24509)

**Authors**: Zhi-Kai Chen, Xu-Xiang Zhong, Song-Yan Li, De-Chuan Zhan, Han-Jia Ye  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.24509v1  

#### Abstract
LLM agents increasingly solve tasks by invoking multiple tools, where parallel execution is essential for low latency but difficult to manage safely. Existing agent benchmarks primarily evaluate tool selection, argument generation, and end-to-end success under mostly serial execution, largely overlo...

---

### 22. [REE-TM: Reliable and Energy-Efficient Traffic Management Model for Diverse Cloud Workloads](https://arxiv.org/abs/2608.25747)

**Authors**: Ashutosh Kumar Singh, Deepika Saxena, Volker Lindenstruth  
**Category**: cs.DC  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.25747v1  

#### Abstract
Diversity of workload demands lays a critical impact on efficient resource allocation and management of cloud services. The existing literature has either weakly considered or overlooked the heterogeneous feature of job requests received from wide range of internet services users. To address this co...

---

### 23. [The Von-Neumann State-Space Transformer for neural decoding](https://arxiv.org/abs/2608.25088)

**Authors**: Morteza Sarafyazd  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.25088v1  

#### Abstract
Cortical computation is strikingly low-dimensional: a handful of latent variables, carried in a neural population's activity, steer the higher-dimensional responses of individual neurons. Our aim is sample efficiency-models that decode well from limited data and at small parameter budgets. In a stan...

---

### 24. [Physics-Informed Foresight Pruning for Sparse PINN Solvers of Nonlinear PDEs](https://arxiv.org/abs/2608.25564)

**Authors**: Ahmad Ishaque Karimi, Uvini Balasuriya Mudiyanselage, Kookjin Lee  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.25564v1  

#### Abstract
Physics-informed neural networks (PINNs) often rely on over-parameterized models to optimize coupled solution and differential-residual objectives, leaving unclear how much capacity is necessary and what pruning should preserve. We study foresight pruning at initialization for sparse PirateNet PDE s...

---

### 25. [Are LLM-Enhanced GNNs Privacy-Safe?](https://arxiv.org/abs/2608.25727)

**Authors**: Longzhu He, Zelang Wen, Chaozhuo Li, Sen Su  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.25727v1  

#### Abstract
Large language models (LLMs) have recently advanced graph neural networks (GNNs) by enriching node representations with semantic information, giving rise to LLM-enhanced GNNs that achieve substantial performance gains. However, their vulnerability to privacy attacks, in which adversaries infer sensi...

---

### 26. [Robust CurveMoE: Multi-Norm Adversarial Defense for Mixture-of-Experts Models via Mode Connectivity](https://arxiv.org/abs/2608.26043)

**Authors**: Xu Zhang, Ren Wang  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.26043v1  

#### Abstract
Multi-norm adversarial defense aims to protect neural networks against perturbations defined by different norm constraints, but existing methods typically optimize competing robustness objectives within a single parameter configuration, leading to substantial training cost and unfavorable robustness...

---

### 27. [Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware](https://arxiv.org/abs/2608.23807)

**Authors**: Farhana Amin, Sabiha Afroz, Mona Moghadampanah, Dimitrios S. Nikolopoulos  
**Category**: cs.AI  
**Published**: 2026-08-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.23807v1  

#### Abstract
Masked diffusion language models (dLLMs) can in principle generate text faster than autoregressive (AR) models, since they denoise many tokens at once. Recent systems have begun building serving infrastructure for dLLMs, but none first measure how these models behave under real, concurrent serving l...

---

### 28. [Why and When Neural Networks Improve Local Approximation in Optimization](https://arxiv.org/abs/2608.24963)

**Authors**: Chengkuo Bian, Pengcheng Xie  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.24963v1  

#### Abstract
Published experience with neural surrogates in derivative-free optimisation is contradictory: the same family of models that cuts the evaluation count of one solver leaves another unchanged, or makes it worse. We show that the contradiction dissolves once three factors are stated, and that these, ra...

---

### 29. [Physics-Informed Error Field Learning: A Post-Training Optimization Framework for Physics-Informed Neural Networks](https://arxiv.org/abs/2608.24970)

**Authors**: Jiuyun Sun, Yong Zhang  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.24970v1  

#### Abstract
Physics-Informed Neural Networks (PINNs) have emerged as an important class of numerical methods for solving partial differential equations (PDEs). However, during the late-stage optimization process, further parameter updates often yield diminishing accuracy improvements while increasing computatio...

---

### 30. [GRAPE: Gradient Refinement and Progress-Aware Exploitation for Query-Efficient High-Dimensional Bayesian Optimization](https://arxiv.org/abs/2608.25116)

**Authors**: Richard Cornelius Suwandi, Feng Yin  
**Category**: cs.LG  
**Published**: 2026-08-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.25116v1  

#### Abstract
Optimizing expensive, high-dimensional black-box functions remains a central challenge in modern machine learning and scientific discovery. While local Bayesian optimization mitigates the curse of dimensionality, existing techniques often prioritize the probability of descent over the magnitude of p...

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

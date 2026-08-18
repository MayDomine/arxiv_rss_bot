# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-18 06:04:41 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FlashQuant: Sparse-Dense Fusion for Memory-Efficient Outlier-Aware LLM Inference](https://arxiv.org/abs/2608.15531)

**Authors**: Junqing Lin, Jingwei Sun, Zhengding Hu, Guangzhong Sun  
**Category**: cs.DC  
**Published**: 2026-08-18  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2608.15531v1  

#### Abstract
Low-bit quantization reduces the memory footprint and computational cost of large language model (LLM) inference. However, high-magnitude outlier weights can induce substantial quantization errors and degrade model accuracy. Outlier-aware quantization addresses this issue by retaining outliers in hi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《FlashQuant: Sparse-Dense Fusion for Memory-Efficient Outlier-Aware LLM Inference》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在大型语言模型（LLM）推理中，**低比特量化**（如 W4A16）被广泛用于降低内存占用和计算开销。然而，权重中的**高幅值离群值**（outliers）会导致严重的量化误差，从而损害模型精度。为此，**outlier-aware quantization** 被提出：将离群权重保留为高精度（如 BF16），其余权重进行低比特量化。

这种策略虽然提升了精度，但也引入了**混合计算模式**：  
- 低比特权重通过 **low-bit GEMM** 处理（密集路径）  
- 高精度离群值通过 **SpMM** 处理（稀疏路径）

现有系统通常将这两个路径分别用独立的 GPU kernel 执行，导致：
- **重复加载激活矩阵**（activations）
- **中间输出多次写回全局内存**
- **冗余的 global memory 访问**
- 在解码阶段（decoding）这类内存受限场景下，性能严重下降

---

### ✅ 提出的新方法与创新思路

论文提出了 **FlashQuant** —— 一种**统一的稀疏-稠密融合执行框架**，用于 outlier-aware 的 W4A16 解码推理。

其核心思想是：  
> 将原本分离的 **dense GEMM** 和 **sparse SpMM** 路径**融合到一个 GPU kernel 中**，实现对共享数据（如 activation tiles、output tiles）的片上复用（on-chip reuse），从而消除冗余内存访问。

#### 主要技术创新：

1. **Sparse-Dense Tiling（稀疏-稠密分块）**
   - 构建统一的 tile 层次结构，使稀疏 outlier 处理与 dense GEMM 分块对齐
   - 实现 CTA、warp、thread 三级协同调度
   - 支持 Stream-K 动态负载均衡

2. **Tile-COO 离群值编码格式**
   - 专为融合执行设计的稀疏存储格式
   - 按 tile 和 column-oriented bucket 组织数据
   - 支持向量化访问、减少 shared memory bank conflict
   - 存储开销低于传统 CSR 格式

3. **Pipelined Scheduling（流水线调度）**
   - 采用多阶段流水线 + 双缓冲机制
   - 重叠数据搬运（HBM → SHM → Register）、反量化、稀疏解码与计算
   - 有效隐藏内存延迟

---

### ✅ 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **内存效率** | 消除重复 activation 加载和部分输出写回，显著减少 global memory traffic |
| **计算效率** | 单 kernel 执行避免 launch overhead，提升硬件利用率 |
| **可扩展性** | 在不同 generation GPU 上均表现优异，尤其在高算力设备上增益更大 |
| **精度-效率平衡** | 在保持 outlier-aware 高精度的同时，大幅提升推理速度 |

---

## 2. 核心实验方法和设置

### 📚 数据集与模型

- 从多个主流 LLM 家族提取典型算子形状：
  - **LLaMA**, **Qwen**, **DeepSeek**, **OPT**
- 测试不同离群值密度：`{1.0%, 1.5%, 2.0%, 2.5%, 3.0%}`
- 总计构建 **340 个 INT4+BF16 outlier-aware 算子配置**
- 问题规模范围：`(4096,4096)` 到 `(75264,14848)`

---

### ⚙️ 实验设置

| 项目 | 设置 |
|------|------|
| **硬件平台** | NVIDIA GeForce RTX 3090 / 4090 / 5090（三代消费级 GPU） |
| **评估阶段** | 主要聚焦于 **autoregressive decoding** 阶段（内存受限） |
| **评估指标** | - 推理速度（Speedup）<br>- 内存访问量（Memory Access Overhead）<br>- 吞吐量（Throughput, tokens/s）<br>- 预处理时间与存储开销 |

---

### 🔁 基线方法对比

| 基线 | 描述 |
|------|------|
| **cuBLAS BF16** | 全精度 GEMM，作为高精度参考 |
| **Marlin** | 当前最优的 dense low-bit GEMM kernel（INT4） |
| **Sputnik / cuSPARSE** | 主流稀疏矩阵乘法库，用于处理 outlier SpMM |
| **Sputnik + Marlin / cuSPARSE + Marlin** | 强基线：**分离式执行方案**（dense 与 sparse 分别运行）<br>——这是当前实际部署中最常见的做法 |
| **Marlin-only** | 忽略 outlier 的理想上限（upper bound） |

> 注：目前尚无公开的“融合型”outlier-aware kernel，因此采用组合方式构建最强 unfused baseline。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 整体加速比（vs cuBLAS BF16）

- FlashQuant 实现 **2.74× ~ 4.18×** 的端到端加速
  - 主要得益于 INT4 量化带来的计算压缩

#### ✅ 对比最强 unfused 基线（Sputnik + Marlin）

- 在 RTX 3090 上：**1.03× ~ 1.16×**
- 在 RTX 4090 上：**1.12× ~ 1.37×**
- 在 RTX 5090 上：**1.23× ~ 1.53×**

> 💡 结论：随着 GPU 算力提升，FlashQuant 的优势更加明显，因为其减少了内存瓶颈的影响。

---

### 🔍 内存访问优化效果

- FlashQuant 相比 Sputnik 减少最多达 **45.4% 的内存访问量**
- 特别是在 batch size 较小时（如 8~16），内存复用收益最大

---

### 🧪 消融实验（Ablation Study）

在 RTX 4090 上逐步启用各项技术，几何平均提速如下：

| 技术 | 提速幅度 | 贡献说明 |
|------|----------|--------|
| **Sparse-Dense Tiling (SD)** | +6% ~ 16% | 实现 tile 对齐，启用 activation/output on-chip reuse |
| **Extra Warp Allocation (EW)** | +2% ~ 4% | 为稀疏路径分配额外 warp，改善内存延迟隐藏 |
| **Reordering (RE)** | +4% ~ 9% | 列重排平衡负载，元素重排减少 bank conflict |

> 三者协同带来显著累积增益。

---

### 🔄 端到端推理性能（vLLM 集成测试）

在 **LLaMA3-8B** 和 **Qwen2.5-14B** 上测试真实生成吞吐量（tokens/s）：

| Model | Batch Size | BF16 | Marlin | cuSPARSE | **FlashQuant** |
|-------|------------|------|--------|----------|----------------|
| LLaMA3 | 8 | 56.9 | 138.5 | 90.7 | **114.4** |
| LLaMA3 | 16 | 56.1 | 132.3 | 75.3 | **107.6** |
| Qwen2.5 | 8 | OOM | 81.4 | 52.5 | **68.3** |
| Qwen2.5 | 16 | OOM | 79.4 | 41.0 | **64.2** |

> - FlashQuant 实现 **2.01×** 端到端加速（vs BF16）
> - 虽然慢于纯 Marlin，但**精度更高**
> - 相比 cuSPARSE 实现 **~1.3× ~ 1.6×** 加速，且更稳定

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **冗余内存访问是 outlier-aware 推理的关键瓶颈**
   - 分离式 kernel 设计导致 activation 和 output 被反复读写
   - 在 decoding 场景下尤为致命

2. **content-sharing fusion 是有效的解决方案**
   - FlashQuant 通过单 kernel 融合 dense 与 sparse 路径，实现了真正的 intra-operator 数据复用

3. **定制化稀疏格式（Tile-COO）至关重要**
   - 传统 CSR/COO 不适配融合执行
   - Tile-COO 提供更好的局部性、更低的元数据开销和更高的并行度

4. **性能增益随硬件演进而放大**
   - 新一代 GPU（如 RTX 5090）算力更强，内存墙更突出，FlashQuant 的优化价值更大

---

### ⚠️ 方法的局限性

1. **仅支持 offline 预处理**
   - 需要在部署前完成 outlier detection、量化、Tile-COO 编码等步骤
   - 不适用于动态变化的 outlier 分布

2. **当前仅支持 W4A16 + BF16 outlier**
   - 尚未扩展至更低比特（如 INT2）或其他混合精度组合

3. **小 batch size 下存在填充开销**
   - 输入会 pad 到最小尺寸（如 8），影响极小 batch（1~2）时的效率

4. **依赖特定 GPU 架构特性**
   - 如 Tensor Core、shared memory bank 结构等，移植性需验证

---

### 🔮 未来工作方向

- 扩展至 **更低比特量化**（如 INT2）和 **weight-activation quantization**
- 支持 **online outlier prediction** 或自适应阈值调整
- 探索 **训练-感知的量化与融合策略**
- 将融合思想推广至其他混合计算模式（如 MoE、pruning + quantization）

---

## ✅ 总结

**FlashQuant** 是首个针对 **outlier-aware LLM 推理** 提出的**稀疏-稠密融合执行框架**。它通过 **sparse-dense tiling**、**Tile-COO 编码** 和 **pipelined scheduling**，成功将原本分离的 dense GEMM 与 sparse SpMM 路径融合进单一 kernel，实现了片上数据复用，大幅降低了内存访问开销。

实验表明，该方法在多种 GPU 上实现了 **最高 1.53×** 的加速（相比最强 unfused 基线），并在端到端任务中达到 **2.01×** 的吞吐提升，同时保持了 outlier-aware 量化带来的高精度优势。

> **一句话总结**：  
> FlashQuant 用“融合”打破“分离”，以系统级优化释放了 outlier-aware 量化的真正潜力。

</details>

---

### 2. [Every Expert Counts: ExactMoE for Memory-Efficient W4A16 Inference](https://arxiv.org/abs/2608.15383)

**Authors**: Amjad Saab  
**Category**: cs.LG  
**Published**: 2026-08-18  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2608.15383v1  

#### Abstract
Sparse mixture-of-experts (MoE) language models reduce arithmetic by activating only a small subset of experts per token, yet deployment still requires storing and moving the full expert bank. We present ExactMoE, an inference design that applies symmetric group-128 four-bit weight quantization only...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Every Expert Counts: ExactMoE for Memory-Efficient W4A16 Inference*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
稀疏混合专家模型（sparse MoE）虽然通过仅激活部分专家实现了高效的计算，但在部署时仍需存储完整的专家库，导致**GPU 显存占用高**。现有方法面临三大瓶颈：
- **GPU 容量限制**：所有专家必须驻留 GPU；
- **主机-设备传输开销**：频繁的专家加载带来延迟；
- **量化误差**：压缩专家权重可能损害模型质量。

本文旨在解决如何在**不牺牲专家完整性、保持路由逻辑不变**的前提下，实现**低显存、高性能的 W4A16 推理**。

---

### 提出的新方法：EXACTMoE
提出 **EXACTMoE**，一种面向内存高效的 MoE 推理系统设计，其核心思想是“**每个专家都重要**”（Every Expert Counts），确保所有专家均可被访问且按原路由机制执行。

#### 创新点：
- ✅ **Expert-only W4A16 量化**：仅对 routed expert 的权重进行对称 group-128 四比特（INT4）量化，其余组件（router、attention、embeddings、norm 层、LM head）保持 BF16 精度。
- ✅ **Kernel-Native 存储与传输**：量化后的专家以 MARLIN 格式打包并直接存储于 pinned host memory 中，GPU cache miss 时无需反量化或重新打包，直接异步拷贝。
- ✅ **可配置 GPU Slot Cache + Wave-Based 执行**：使用一个可调大小的 GPU 缓存存放活跃专家；当活跃专家数超过缓存容量时，将其划分为多个“wave”，每 wave 内通过 fused grouped MoE kernel 并行执行。
- ✅ **完整专家可用性保障（Completeness Guarantee）**：
  - 不剪枝任何专家；
  - 不替换或代理专家；
  - 所有选中专家均在 GPU 上执行，无 CPU fallback。

> “Exact” 指的是：**原始 router 不变、top-k 路由不变、所有专家均可执行**，而非数值上等同于 BF16 模型。

---

### 相比现有方法的优势
| 维度 | EXACTMoE | 其他方法（如 MoE-Infinity, SwapMoE, HOBBIT） |
|------|----------|---------------------------------------------|
| **专家完整性** | ✔️ 完全保留 | ❌ 可能剪枝、替换或虚拟化 |
| **执行位置** | ✔️ 所有专家在 GPU 执行 | ⚠️ 部分专家可能回退到 CPU |
| **表示一致性** | ✔️ Host/GPU 使用相同 MARLIN-packed 表示 | ⚠️ 可能需要运行时转换 |
| **执行效率** | ✔️ Fused grouped GEMM，避免逐专家调用 | ⚠️ 多为顺序或轻度并行执行 |

---

## 2. 核心实验方法和设置

### 使用的数据集与模型
- **主模型**：`allenai/OLMoE-1B-7B-0924-Instruct`
  - 16 个稀疏层，每层 64 个专家，top-8 路由；
  - 总参数 ~7B，其中 routed expert 参数达 **6.44B**；
  - 激活参数约 1B。
- **评估任务**：
  - **ARC-Easy**（570 题）
  - **PIQA**（1,838 题）
  - **HellaSwag**（10,042 题）
  - 合计 **12,450 道零样本多选题**
  - WikiText-2 和 Dolly-15k 用于 perplexity 测试

---

### 实验设置
- **硬件平台**：单张 **NVIDIA L4 GPU**（22.0 GiB 显存），53 GiB 主机内存
- **软件环境**：
  - PyTorch 2.11.0 + cu130
  - vLLM 0.26.0（使用其 fused MARLIN MoE primitive）
  - MARLIN-packed weights 实现基于 IST-DASLab/marlin
- **缓存配置（C）**：
  - `C ∈ {8, 16, 32, 64}` slots per layer
  - `C=64` 表示 fully resident（全驻留）
- **推理模式**：
  - Prefill + Decode 分离
  - Greedy decoding，生成 128 tokens
  - Persistent warm 运行（进程复用，cache 持久化）

---

### 评估指标
| 类别 | 指标 |
|------|------|
| **性能** | Decode throughput (tokens/s), TTFT (ms), Peak reserved GPU memory (GiB) |
| **传输开销** | Host-to-device traffic (H2D, GiB/request) |
| **缓存行为** | Hit rate (%) |
| **模型质量** | Normalized MC accuracy (%), Raw accuracy, Perplexity (WikiText/Dolly) |
| **正确性验证** | Bootstrap confidence intervals, McNemar test（配对预测差异检验） |

---

### 基线方法对比
- **BF16 Baseline**：原始全精度模型，作为性能与质量基准
- **Sequential W4 Reference**：内部实现的逐专家执行 W4 推理路径，用于消融实验
- *未在相同环境下复现其他 MoE offloading 系统（如 MoE-Infinity），故不做直接比较*

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1 & Figure 2）

| 方法 | Slots | Decode (tok/s) | TTFT (ms) | Reserved GPU (GiB) | H2D Traffic (GiB) | Hit Rate (%) |
|------|-------|----------------|-----------|---------------------|--------------------|--------------|
| BF16 | – | 21.66 | 79.03 | **14.17** | 0.00 | – |
| EXACTMoE | 16 | **17.73** | 290.42 | **1.84** | 21.44 | 56.3% |
| EXACTMoE | 64 (full) | **31.92** | 33.92 | **4.06** | 0.00 | resident |

#### 核心性能对比：
- **显存节省**：
  - 在 `C=16` 下，峰值 GPU 显存从 **14.17 GiB → 1.84 GiB**，降低 **87.04%**
- **吞吐表现**：
  - `C=16` 达到 BF16 的 **81.85% decode throughput**
  - `C=64`（全驻留）达到 **31.92 tokens/s**，比 BF16 快 **+47.4%**
- **传输代价**：
  - `C=16` 每请求平均传输 **21.44 GiB** 专家数据（H2D）
  - `C=32` 降至 9.13 GiB，命中率提升至 81.4%

---

### 模型质量结果（Table 2 & 3）

| 方法 | Wiki PPL | Dolly PPL | MC Acc. Norm (%) | MC Acc. Raw (%) |
|------|----------|-----------|------------------|-----------------|
| BF16 | 17.8185 | 11.1014 | **70.8996%** | 59.0602% |
| EXACTMoE W4A16 | 18.5047 | 10.9399 | **70.3534%** | 58.5542% |

#### 质量分析：
- **归一化准确率保留率**：  
  $ \frac{70.3534}{70.8996} = 99.23\% $
- **绝对下降**：-0.546 个百分点（95% CI: [-0.964, -0.137]，p=0.0112）
- **任务级差异**：
  - ARC-Easy: +0.35 pp（不显著）
  - PIQA: +0.05 pp（不显著）
  - HellaSwag: **-0.71 pp**（显著下降，p=0.0027）

> 尽管整体略有下降，但相对保留率极高，说明 W4 量化对 MoE 专家具有较强鲁棒性。

---

### 消融实验结果

#### （1）Fused vs Sequential Execution（匹配 16-slot 缓存）
- **Sequential W4 Reference**：9.36 tokens/s
- **Fused Grouped Execution**：**18.43 tokens/s**
- ➜ **加速 1.97×**

✅ 验证了 fused grouped GEMM 的巨大优势：减少 kernel launch 开销，合并计算。

#### （2）Kernel 正确性测试
- 对 materialized dequantized W4 输出的最大相对误差：< 0.00560
- fused MoE preflight 最大绝对误差：0.00098
- 最后一层 logits 差异最大为 0.421875（FP16 容差内）

✅ 证明实现无 bug，packing 与 kernel 执行正确。

#### （3）Batched Generation 扩展性（Table 4）
| Batch | BF16 (tok/s) | W4A16 (tok/s) | W4A16 Reserved (GiB) |
|-------|---------------|----------------|------------------------|
| 1     | 32.14         | 15.74          | 1.82                   |
| 8     | 65.25         | **90.02**      | 1.86                   |

- 在 batch=8 时，W4A16 吞吐**超过 BF16**，得益于更多 token 被聚合进 grouped GEMM，发挥 W4 加速潜力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **可在极低显存下运行大型 MoE 模型**：
   - 仅需 **1.84 GiB GPU 显存**即可运行 OLMoE-7B（原需 14.17 GiB），适合边缘或消费级设备。
2. ✅ **高质量保留**：
   - 归一化准确率保留 **99.23%**，表明 expert-only W4 量化对语义影响极小。
3. ✅ **性能-显存权衡可控**：
   - 通过调节 cache slot 数量（C），可在显存、带宽、吞吐之间灵活选择操作点。
4. ✅ **全专家可用性可行**：
   - 实现了“complete-expert”推理范式，无需剪枝、替换或 CPU fallback。
5. ✅ **Fused grouped execution 是关键优化**：
   - 相比逐专家执行提速近 **2×**，凸显 kernel fusion 的价值。

---

### 方法的局限性
1. **依赖 host-to-GPU 带宽**：
   - 若专家重用率低（poor locality），频繁 cache miss 会导致严重传输瓶颈。
2. **Host memory requirement**：
   - 所有 packed expert 必须能放入 host memory（本例中约 3.09 GiB），限制超大规模模型部署。
3. **非数值恒等性**：
   - W4 量化会轻微改变 hidden states，可能导致后续 routing 路径偏移，影响长序列一致性。
4. **硬件与格式依赖性强**：
   - 依赖 NVIDIA GPU、CUDA、MARLIN kernel 及特定 weight layout（如 group-128, concat order）。
5. **静态权重假设**：
   - 不支持动态更新专家（如 online learning），更新需重新打包。

---

### 未来工作方向
1. **动态精度分配**：结合 sensitivity-aware 或 activation-frequency-based 方法，在敏感专家上使用更高 bit-width。
2. **预取与缓存策略优化**：引入 request-level locality prediction 或 LRU+LFU hybrid policy 减少 miss。
3. **跨设备扩展**：支持 multi-GPU 分布式 cache 协调与 continuous batching。
4. **端到端非劣效性验证框架**：
   - 预注册任务加权方案与 non-inferiority margin，提升量化评估严谨性。
5. **绿色推理研究**：
   - 量化虽省显存，但大量 H2D 传输可能增加能耗，需综合评估生命周期效益。

---

> 📌 **一句话总结**：  
> **EXACTMoE 实现了“一个不少”的 MoE 推理——在仅占 13% 显存的情况下，以 99.2% 的准确率保留和接近原生速度运行 OLMoE-7B，揭示了一条实用的 memory-transfer-throughput 权衡前沿。**

</details>

---

### 3. [MoE Router-Guided Clustering for Heterogeneous Federated Instruction Tuning](https://arxiv.org/abs/2608.15311)

**Authors**: Ankita Sharma, Bahar Farahani, Sanaz Rahimi Moosavi, Amir Rrahmani, Farshad Firouzi, Krishnendu Chakrabarty  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2608.15311v1  

#### Abstract
Federated instruction fine-tuning enables Large Language Models (LLMs) to adapt to decentralized, privacy-sensitive data without requiring data sharing. Recent Mixture-of-Experts (MoE) LLMs are particularly attractive for federated learning because their sparse activation reduces computation and com...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：MoE Router-Guided Clustering for Heterogeneous Federated Instruction Tuning**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
在**异构联邦指令微调**（Heterogeneous Federated Instruction Tuning）场景中，不同客户端的数据分布差异显著（如专注于不同的任务：分类、问答、摘要等）。传统的 **Federated Averaging (FedAvg)** 方法对所有客户端进行全局参数聚合，容易导致**负迁移**（Negative Transfer），损害个性化性能。

此外，尽管 **Mixture-of-Experts (MoE)** 架构因其稀疏激活特性在联邦学习中具有通信效率优势，但现有方法大多仅将其用于模型扩展或个性化适配，**忽略了路由机制（router）本身所蕴含的语义信息**——即专家选择模式可以反映客户端数据分布特征。

---

### 🚀 提出的新方法：ClientMorpher
本文提出 **ClientMorpher**，一种**基于路由感知的个性化联邦指令微调框架**，其核心思想是：

> 利用预训练 MoE 模型中的 **routing signatures**（专家激活模式）作为轻量级信号，指导客户端协作分组，在聚合前识别“应合作”的客户端。

#### 两种互补的聚类策略：
| 方法 | 描述 |
|------|------|
| **ClientMorpher-C** | 直接基于客户端的 **expert activation profiles** 进行聚类（客户中心视角） |
| **ClientMorpher-E** | 先对专家按跨客户端使用模式聚类，再将客户端分配到其路由流量最集中的专家簇中（专家中心视角） |

这两种策略都利用了 MoE 路由行为作为结构性信号，实现更合理的协作分组。

---

### 🔍 相比现有方法的优势
| 维度 | ClientMorpher 的优势 |
|------|------------------------|
| **信息利用** | 首次系统性地利用 MoE 的 **routing behavior** 作为客户端相似性的代理信号，而非仅依赖梯度或参数 |
| **个性化能力** | 通过聚类避免不相关客户端间的无效甚至有害聚合，提升模型个性化表现 |
| **通信效率** | 保持 MoE 架构原有的稀疏性与低通信开销（仅更新 LoRA 参数），无额外通信成本 |
| **可扩展性** | 聚类发生在训练前（Phase 0），不影响训练流程，适用于大规模部署 |

---

## 2. **核心实验方法和设置**

### 📚 数据集
- 使用 **Databricks Dolly-15K** 数据集，包含多种指令遵循任务：
  - **CLS**: 文本分类
  - **QA**: 封闭域问答
  - **IE**: 信息抽取
  - **Summ**: 摘要生成
- 所有样本格式化为 Alpaca-style prompt，损失仅计算 response tokens。

---

### ⚙️ 实验设置
#### 客户端非独立同分布（Non-IID）划分方式：
1. **Pathological Split**  
   - 每个客户端只拥有单一任务类型的全部数据（极端异构）
2. **Dirichlet Partition**（浓度参数 α）
   - α = 0.1：高度异构
   - α = 1.0：相对均衡的非IID分布

#### 模型架构
- 基于 **MoE-LLM** 架构，冻结主干网络和专家参数
- 只微调少量 **LoRA** 参数（占比 0.0878%），确保通信高效
- Router 动态选择 top-k 专家处理每个 token

#### 评估指标
- **ROUGE-L**：衡量生成文本与参考答案的重叠程度（越高越好）
- 平均 ROUGE-L（Across Tasks）
- 各任务类别下的性能分解
- **通信开销**（Communication Cost %）：统一控制变量

---

### 🆚 对比的基线方法
| 方法 | 描述 |
|------|------|
| **Local Training** | 各客户端独立训练，无协作，零通信 |
| **MoE-FedAvg (LoRA)** | 标准联邦平均，所有客户端共享一个全局模型状态 |
| **ClientMorpher-C** | 基于客户端激活轮廓聚类后分组聚合 |
| **ClientMorpher-E** | 基于专家使用签名聚类，间接形成客户端协作组 |

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（来自 Table II & III）

#### ✅ 四客户端 Pathological Split 结果（Table II）
| Method | CLF | CQA | IE | SUM | **Avg. ROUGE-L** | Comm.% |
|--------|-----|-----|----|-----|------------------|---------|
| Local | 0.6036 | 0.4096 | 0.5087 | 0.4075 | **0.4824** | 0.00 |
| MoE-FedAvg | 0.6299 | 0.4232 | 0.5085 | 0.4090 | 0.4927 | 0.0878 |
| ClientMorpher-C | 0.5937 | 0.3785 | **0.5243** | 0.3960 | 0.4731 | 0.0878 |
| **ClientMorpher-E** | **0.6411** | **0.4254** | 0.5099 | **0.4181** | **0.4986** | 0.0878 |

> 💡 **结论**：ClientMorpher-E 在平均性能上优于所有方法（+0.59% vs FedAvg, +1.62% vs Local），且在多数任务上领先；ClientMorpher-C 在 IE 上最优，但在其他任务上略逊。

---

#### ✅ Dirichlet 分布下多客户端结果（Table III）

##### 当 α = 0.1（高异构）：
- MoE-FedAvg 平均得分：**0.4960**
- ClientMorpher-C：**0.4959**
- ClientMorpher-E：**0.4929**
- > 两者均接近 FedAvg，远超 Local（0.4391），说明路由引导有效缓解负迁移

##### 当 α = 1.0（较平衡）：
- ClientMorpher-E 平均达 **0.4936**，优于 FedAvg（未列出但趋势一致）
- ClientMorpher-C 在 Summarization 上仍具优势
- > 表明两种聚类策略适应不同异构程度

---

### 🔬 消融分析与关键观察
- **ClientMorpher-E 更稳健**：在分类和问答任务中表现稳定，尤其适合中等异构环境
- **ClientMorpher-C 更敏感**：当客户端任务边界清晰时（如病理分割），直接聚类效果更强
- **互补性发现**：client-centric 与 expert-centric 聚类捕捉了不同类型的任务相似性，联合使用可能进一步提升性能
- **通信成本相同**：所有联邦方法交换相同的 LoRA 参数量，性能提升完全归因于“更聪明的合作”

---

## 4. **关键结论和发现**

### ✅ 主要结论
1. **Routing Signatures 是有效的客户端相似性信号**  
   MoE 模型的 router 不仅是功能模块，更是反映输入语义分布的“观测器”，可用于指导联邦协作。

2. **ClientMorpher 显著提升个性化性能**  
   在高度异构环境下，相比传统 FedAvg 和本地训练，ClientMorpher 实现了更优的个性化生成质量（ROUGE-L ↑）。

3. **两种聚类策略具有互补性**  
   - **ClientMorpher-C**：适用于任务边界分明、客户端专业化强的场景
   - **ClientMorpher-E**：更适合分布渐变、专家共享频繁的设定，更具鲁棒性

4. **无需增加通信即可提效**  
   所有改进均在原有 MoE + LoRA 的低通信框架内完成，具备实际部署价值。

---

### ⚠️ 局限性
- 当前方法依赖**预训练 MoE 模型的稳定性**，若 router 本身不稳定或专家无明确语义分工，性能可能下降。
- 聚类过程假设客户端任务静态不变，未考虑动态加入/退出或任务漂移（Dynamic Task Shift）。
- 实验集中在文本生成任务，尚未验证在多模态或其他 NLP 任务中的泛化能力。
- 聚类数量 $ K $ 需预先设定，缺乏自动确定机制。

---

### 🔮 未来工作方向
1. **动态聚类机制**：支持客户端在线加入、任务演化下的实时重聚类
2. **自监督路由表征学习**：增强 routing signature 的判别能力，尤其在专家职责模糊时
3. **混合聚类策略融合**：结合 ClientMorpher-C 与 -E 输出，构建层次化协作图
4. **理论分析**：建立 routing similarity 与任务相似性之间的形式化联系
5. **扩展至更多 MoE 架构**：如 Switch Transformer、GLaM 等大规模稀疏模型

---

## ✅ 总结一句话
> **ClientMorpher 首次将 MoE 模型的 routing behavior 视为联邦学习中的结构先验，通过路由感知聚类实现了高效、个性化且通信友好的联邦指令微调，为稀疏大模型在去中心化场景下的应用提供了新范式。**

</details>

---

### 4. [KV-Pipe: On the Relation Between KV Sharing and Pipeline Parallel Efficiency in LLMs](https://arxiv.org/abs/2608.15943)

**Authors**: Maryam Dialameh, Hossein Rajabzadeh, Harish Krishnamoorthy Murali, Walid Ahmed, Weiwei Zhang, Hyock Ju Kwon  
**Category**: cs.DC  
**Published**: 2026-08-18  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2608.15943v1  

#### Abstract
Pipeline parallelism (PP) is widely used to scale large language model (LLM) training, but its efficiency is often limited by stage imbalance and pipeline bubbles. Meanwhile, cross-layer KV sharing has primarily been studied as a mechanism for reducing KV-cache costs during inference, without examin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：KV-Pipe: On the Relation Between KV Sharing and Pipeline Parallel Efficiency in LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **Pipeline Parallelism (PP)** 在大规模语言模型（LLM）训练中广泛使用，但其效率常受限于**阶段不平衡（stage imbalance）** 和 **流水线气泡（pipeline bubbles）**，导致设备利用率低下。
- 传统方法通常将模型每层的计算成本视为固定不变，忽略了**KV Cache布局**对流水线负载分布的影响。
- 现有的 **cross-layer KV sharing** 主要用于推理阶段减少 KV-cache 内存开销，未被系统性地用于优化训练时的流水线效率。

### 🚀 提出的新方法与创新思路
- **KV-Pipe**：一种**stage-aware 的跨层 KV 共享机制**，首次将 KV 共享从单纯的内存压缩技术提升为**流水线平衡的控制旋钮（control knob）**。
- 核心思想：通过有选择地将某些注意力层转换为 KV-sharing 层，**主动重塑各 stage 的计算负载**，从而降低瓶颈 stage 的 FLOPs，使整个流水线趋于均衡。
- 提出 **FLOPs Imbalance Ratio (FIR)** 作为衡量流水线不平衡程度的轻量级指标，并以此指导 KV-sharing 层的选择。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | KV-Pipe |
|------|--------|---------|
| **优化目标** | 固定模型结构，优化调度或分区 | 修改模型内部 KV 布局以改变 stage 成本 |
| **作用方式** | “隐藏”气泡（如通过调度填充空闲时间） | “消除”气泡源头（减少瓶颈 stage 时间） |
| **适用性** | 多依赖在线调优或复杂搜索 | **离线执行**，仅需 PP 分区和每层 FLOPs 预估，无运行时开销 |
| **双重收益** | 通常只针对训练或推理之一 | 同时提升 **训练 MFU** 和 **推理吞吐** |

---

## 2. 核心实验方法和设置

### 📊 数据集与模型
- 主要模型：**LLaMA2-7B**（32 层），并扩展至 LLaMA2-13B、LLaMA3-8B、Qwen2.5-14B。
- 序列长度（Sequence Length）：`4K`, `8K`, `16K`, `32K`, `64K`, `128K`。
- 不涉及传统 NLP 下游任务微调，而是基于真实训练/推理 workload 进行端到端性能测量。

### ⚙️ 实验设置
- **硬件平台**：
  - **NPU**：8× Huawei Ascend 910B（主实验）
  - **GPU**：8× NVIDIA V100（验证泛化性）
- **并行策略**：
  - 纯粹研究 **Pipeline Parallelism (PP)**，固定其他维度（不启用 TP/DP）。
  - PP Degree：`PP=2`, `4`, `8`。
- **调度器**：
  - 使用标准 **1F1B** 和改进的 **Seq1F1B** 调度进行对比。
- **KV-Pipe 变体**：
  - **UNIFORM**：均匀分配 KV-sharing 层
  - **SYMMETRIC BIPOLAR**：从中部和尾部对称分配
  - **ARCHITECTURE-BALANCED**（推荐）：优先处理当前瓶颈 stage

### 📈 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **MFU (Model FLOPs Utilization)** | $ \frac{\text{Throughput (iter/s)} \times \text{FLOPs/iter}}{\text{Peak FLOPs}} $ | 衡量训练效率的核心指标 |
| **Iteration Time (s)** | 单次迭代耗时 | 直接反映端到端速度 |
| **FLOPs Imbalance Ratio (FIR)** | $ \frac{\max_i(F_{\text{stage}_i})}{F_{\text{avg}}} $ | 诊断流水线不平衡程度 |
| **Inference Throughput (tokens/s)** | 解码吞吐率 | 推理性能评估 |

### 🆚 基线方法对比
- **Baseline**：原始 full-attention + 1F1B 调度
- **Seq1F1B**：更高效的流水线调度，减少 bubbles
- **Echo-style 25% sharing**：固定尾部 25% 层共享 KV（如 Dialameh et al., 2025）
- **DawnPiper/vPipe**：基于代价模型的自动分区方法（用于比较互补性）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Ascend 910B）

| 设置 | MFU 提升 | Iteration Time 减少 | FIR 改善 |
|------|----------|---------------------|-----------|
| **PP=2, S=4K** | +3.10% | -4.90% | 1.0182 → 1.0101 |
| **PP=4, S=8K** | +4.32% | -7.56% | 1.051 → 1.018 |
| **PP=8, S=8K** | **+9.17%** | **-9.80%** | **1.1194 → 1.0004** |

> ✅ **趋势明显**：随着 PP degree 增大，KV-Pipe 的增益显著增加，说明在高并行度下阶段不平衡问题更严重，KV-Pipe 的调节能力更强。

### 🔁 与基线方法对比
- 在所有配置下，**Architecture-Balanced** 策略均优于 Uniform 和 Symmetric Bipolar。
- 例如在 `PP=8, S=8K` 下：
  - **Uniform**：MFU +6.40%，Time -7.00%
  - **Symmetric Bipolar**：MFU +7.10%，Time -7.60%
  - **Architecture-Balanced**：MFU **+9.17%**，Time **-9.80%**

### 🔍 消融实验结果
#### （1）不同放置策略效果（Table 9）
| 策略 | MFU 提升 | Time 减少 | 说明 |
|------|----------|-----------|------|
| Uniform | +6.40% | -7.00% | 分散优化，无法精准打击瓶颈 |
| Symmetric Bipolar | +7.10% | -7.60% | 有一定集中性，但仍非最优 |
| **Architecture-Balanced** | **+9.17%** | **-9.80%** | 集中资源解决当前瓶颈，最有效 |

#### （2）停止容忍度 ε 敏感性测试
- ε ∈ {0.02, 0.05, 0.10} 对结果影响较小，MFU 提升稳定在 **8.6%–9.17%**，表明算法鲁棒性强。

#### （3）FIR vs 实测延迟引导
- 使用 FIR（分析预估） vs 最大 stage time（实测）作为优化信号：
  - FIR 引导：+9.17% MFU
  - 实测引导：+8.90% MFU
- 表明 **FIR 是一个有效的轻量级代理指标**，无需实际 profiling 即可实现高性能。

#### （4）组合性验证（KV-Pipe + Seq1F1B）
| 方法 | LLaMA2-7B MFU | Time (s) |
|------|---------------|----------|
| 1F1B | 39.0% | 305 |
| Seq1F1B | 47.0% | 265 |
| **Seq1F1B + KV-Pipe** | **50.0%** | **248** |

> ✅ KV-Pipe 与先进调度**正交且可叠加**，进一步提升效率。

#### （5）与 DawnPiper 对比（Table 11）
| 方法 | GPT-2 Speedup vs vPipe-AS | T5 Speedup |
|------|----------------------------|------------|
| DawnPiper-AS | 1.15× | 1.34× |
| **KV-Pipe** | **1.20×** | **1.41×** |

> ✅ KV-Pipe 在相同设置下超越 DawnPiper，证明**修改 per-layer 成本**是独立且有效的优化维度。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KV Layout 是一个被忽视的系统架构自由度**：
   - 通过有选择地引入 cross-layer KV sharing，可以**主动调控流水线各 stage 的计算负载**，而不仅是被动适应。
   
2. **KV-Pipe 实现“双倍收益”（Double Dividend）**：
   - **训练侧**：减少瓶颈 stage FLOPs → 平衡 stage 时间 → 提升 MFU（最高 +9.2%）
   - **推理侧**：减少 KV-cache 生长和冗余投影 → 提升长上下文解码吞吐（最高 +2.77×）

3. **存在最优共享预算（Non-monotonic MFU）**：
   - 并非越多 KV-sharing 越好。当过度减轻原瓶颈后，**早期 stage 可能成为新瓶颈**，导致 FIR 上升、MFU 下降。
   - KV-Pipe 自动找到这个**平衡点**（FIR ≈ 1），避免过矫正。

4. **FIR 是有效的离线诊断工具**：
   - 无需实际运行即可预测不平衡程度，并指导优化方向。

5. **KV-Pipe 正交于现有 PP 技术**：
   - 可与调度优化（如 Seq1F1B）、分区搜索（如 DawnPiper）等方法**组合使用**，带来额外增益。

---

### ⚠️ 方法的局限性
1. **质量假设依赖特定模型结构**：
   - 当前质量验证仅限于 **dense LLaMA-family 模型**，未验证 MoE、Hybrid（如 Mamba-Transformer）等架构下的表现。
   
2. **默认尾部优先策略可能不普适**：
   - 虽然实验中瓶颈始终位于尾部（因 LM Head），但在某些异构设计中，瓶颈可能出现在中间或前端，需配合安全层约束（safe-layer guardrail）使用。

3. **未探索全局最优解**：
   - KV-Pipe 使用贪心策略（greedy），不保证所有 layer-sharing 组合中的全局最优。

4. **实验规模有限**：
   - 最大 PP=8，尚未验证在 PP=16 或更高时的表现；也未覆盖超大规模模型（如 100B+）。

---

### 🔮 未来工作方向
1. **联合优化框架**：
   - 将 **pipeline partitioning**、**KV-sharing placement** 和 **quality constraint** 进行端到端联合优化。

2. **动态自适应 KV-Pipe**：
   - 在训练过程中根据实时负载变化动态调整 KV-sharing 层。

3. **扩展至 MoE 和 Hybrid 架构**：
   - 探索 KV-Pipe 在稀疏专家模型或状态空间模型中的应用潜力。

4. **支持更多 attention 变体**：
   - 如 Grouped-Query Attention (GQA)、Multi-Query Attention (MQA) 下的进一步优化空间。

5. **集成到主流训练框架**：
   - 将 KV-Pipe 打包为插件，供 Megatron-LM、DeepSpeed 等系统直接调用。

---

## 💡 总结
**KV-Pipe** 揭示了一个深刻洞见：**模型内部的 KV 结构不仅是功能组件，更是可编程的系统级优化杠杆**。它打破了“模型结构固定”的隐含假设，提出了一种**架构感知、阶段敏感、离线轻量**的流水线优化新范式，在不增加运行时复杂性的前提下，实现了训练效率与推理性能的双重提升，为 LLM 系统优化开辟了新的设计空间。

</details>

---

### 5. [DepTGL: A Parallel Framework for Memory-based TGNN Training with Adaptive Temporal Data Dependency Management](https://arxiv.org/abs/2608.16305)

**Authors**: Linfang Chen, Zhen Song, Lei Liu, Yu Gu, Yushuai Li, Yanfeng Zhang, Lizhen Cui, Ge Yu, Tianyi Li  
**Category**: cs.DC  
**Published**: 2026-08-18  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2608.16305v1  

#### Abstract
Memory-based Temporal Graph Neural Networks (M-TGNNs) maintain recursively updated node states to capture fine-grained temporal interactions. However, existing distributed frameworks lack effective mechanisms for managing the temporal data dependencies inherent in these models. As a result, they mus...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DepTGL: A Parallel Framework for Memory-based TGNN Training with Adaptive Temporal Data Dependency Management

---

## 1. 论文的主要贡献和创新点

### 解决的问题
**Memory-based Temporal Graph Neural Networks (M-TGNNs)** 在处理动态图时通过维护节点的时序状态来捕捉细粒度的时间交互。然而，现有的分布式训练框架在管理这些模型中的**时序数据依赖关系**方面存在严重不足，导致以下三大瓶颈：

- **严格的串行更新**：必须按时间顺序处理事件，限制了并行化。
- **高昂的远程同步开销**：频繁的跨worker缓存同步带来巨大通信成本。
- **严重的负载不均衡**：真实世界的时间流通常具有高度偏斜，导致straggler效应。

### 提出的新方法
作者提出了 **DepTGL**，一个面向内存型TGNN的可扩展并行训练框架，从**数据中心视角重构了时序依赖管理机制**，其核心创新包括：

#### （1）混合式时序依赖管理（Hybrid Temporal-Dependency Management）
- 引入**离线依赖扫描**（offline dependency scanner），提前识别目标事件所需的远程历史事件。
- 构建**worker-specific mixed batches**，将必要的远程历史事件作为辅助重放事件（auxiliary replay events）嵌入本地批次。
- 结合**轻量级远程缓存**与**选择性依赖驱动通信**，减少运行时对远程状态的请求。

#### （2）梯度感知的缓存同步策略（Gradient-aware Cache Synchronization）
- 利用**梯度范数的指数移动平均（EMA）** 作为模型优化稳定性的信号。
- 当模型进入收敛阶段、梯度变化较小时，**自适应跳过不必要的边界缓存刷新**，显著降低冗余同步。

#### （3）负载感知的时序剪枝策略（Load-aware Temporal Pruning）
- 检测到负载高峰时，**主动移除纯辅助性的重放事件**（pure auxiliary replay events），减轻计算压力。
- 保留所有目标任务事件，确保学习目标不变。

### 相比现有方法的优势
| 维度 | 传统方法缺陷 | DepTGL 改进 |
|------|-------------|------------|
| **通信效率** | 高频全量同步（如 Vanilla DDP）或全缓存（Full Cache）造成高开销 | 动态控制同步频率，通信占比从 >80% 降至 **6.6%** |
| **计算效率** | 大量远程访问阻塞训练流程 | 本地重放替代远程获取，减少细粒度请求 |
| **负载均衡** | 负载偏斜导致严重等待（straggler） | 自适应剪枝缓解热点负载 |
| **系统设计** | 各机制割裂（如仅优化缓存或采样） | **统一运行时控制器**联合调控依赖服务、同步与剪枝 |

---

## 2. 核心实验方法和设置

### 数据集
在六个真实世界的时序图数据集上进行评估，涵盖社交网络、在线社区、教育平台等场景：

| 数据集 | 节点数 | 边数 | 时间跨度 |
|--------|--------|------|----------|
| AskUbuntu | 159.3K | 964.4K | 2613天 |
| MathOverflow | 24.8K | 506.6K | 2350天 |
| Reddit | 11.0K | 672.4K | 30天 |
| MOOC | 7.0K | 411.7K | 30天 |
| lastfm | 2.0K | 1.29M | 1587天 |
| Wikipedia | 9.2K | 157.5K | 30天 |

### 模型
- **TGN**（Temporal Graph Networks）
- **JODIE**

两类典型的 M-TGNN 模型，均维护递归更新的节点记忆状态。

### 实验环境
- **Env 1**：真实分布式环境，5台机器（含 RTX 2080 Ti 和 Tesla P100 GPU）
- **Env 2**：高性能多GPU环境（8 × RTX 4090）

### 评估指标
| 指标 | 描述 |
|------|------|
| **Epoch Time** | 单轮训练耗时（秒） |
| **Speedup** | 相对于 Vanilla DDP 的加速比 |
| **Throughput** | 每秒处理的目标事件数（events/s） |
| **Test AUC / AP** | 链接预测任务下的准确率指标 |
| **Scaling Efficiency** | 多worker下的扩展效率 |

### 基线方法对比
| 基线 | 特点 |
|------|------|
| **Vanilla DDP** | 标准分布式数据并行，固定间隔同步记忆 |
| **DisTGL** | 基于时间划分与混合缓存优化 |
| **NeutronStream** | 滑动窗口事件处理 |
| **MemShare** | 热点节点状态共享以减少通信 |
| **Full Cache**（分析参考） | 完全本地化远程历史缓存 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **平均加速比**：DepTGL 在六大数据集上相比 SOTA 基线实现了 **4.99× 的平均 epoch-time 加速**。
- **最高加速比**：在 AskUbuntu 上达到 **14.06×**（JODIE）和 **6.17×**（TGN）。
- **吞吐量提升**：在 AskUbuntu 上达到 **2921 events/s**，是 Vanilla DDP 的 **6.27×**。

### 与基线方法对比结果
| 方法 | 平均 Speedup | 最大 Speedup | 通信占比 |
|------|---------------|--------------|-----------|
| Vanilla DDP | 1.00× | — | ~82.9% |
| DisTGL | ~1.2× | 1.35× | ~80.5% |
| NeutronStream | ~3.0× | 5.13× | ~67.5% |
| MemShare | ~2.9× | 3.78× | ~33.2% |
| **DepTGL** | **4.99×** | **14.06×** | **6.6%** |

> ✅ DepTGL 在所有数据集和模型上均取得最优 epoch time。

### 准确性保持
尽管进行了大量优化（如同步跳过、事件剪枝），**测试 AUC 和 AP 与基线持平甚至略有提升**，表明其未牺牲模型精度。

例如在 MOOC 上：
- Vanilla DDP: AUC=0.7865, AP=0.7700
- DepTGL: AUC=0.8268, AP=0.8158

说明：**目标事件始终保留，仅剪枝辅助事件，不影响监督目标**。

### 消融实验结果（Ablation Study）
逐步启用各组件的效果如下（以 Wikipedia 为例）：

| 配置 | Epoch Time (s) | Speedup vs Base |
|------|----------------|------------------|
| Base（关闭所有优化） | 408.58 | 1.00× |
| +Replay（启用本地重放） | 103.02 | 3.97× |
| +Sync（加入梯度感知同步） | 137.81 | — |
| Full（完整 DepTGL） | 55.83 | 7.32× |

> 🔍 三个模块互补：
> - `+Replay` 显著减少远程依赖；
> - `+Sync` 进一步削减同步开销；
> - `+Pruning` 缓解负载尖峰。

---

## 4. 关键结论和发现

### 主要发现
1. **时序依赖可以离线展开为本地重放事件**，从而将运行时的远程请求转移到预处理阶段。
2. **梯度稳定性可用于指导缓存同步决策**，避免在收敛期浪费通信资源。
3. **负载感知剪枝能有效缓解 straggler 效应**，尤其在事件流偏斜严重时效果显著。
4. **通信-缓存-计算三者需协同设计**，单一优化无法突破整体瓶颈。

### 方法的局限性
- **依赖精确的时间戳排序**：若输入事件乱序，可能影响状态重建正确性。
- **超参数敏感性**：同步窗口大小 $K$、阈值 $T_g$, $T_c$ 需合理调参（见 Table VII）。
- **主要用于 M-TGNN**：对 snapshot-based TGNN 或静态 GNN 不适用。

### 未来工作方向
- 将自适应机制扩展至更广泛的动态图学习场景（如异构图、多模态流）。
- 探索基于强化学习的动态调度策略，进一步提升运行时适应能力。
- 支持容错与弹性伸缩，在大规模集群中实现更高可用性。

---

> 📌 **总结一句话**：  
> DepTGL 通过**离线依赖展开 + 梯度感知同步 + 负载感知剪枝**三位一体的设计，在不损失准确性的前提下，将 M-TGNN 分布式训练速度提升了近 **5倍**，为高效时序图神经网络训练提供了新的系统范式。

</details>

---

### 6. [S2-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices](https://arxiv.org/abs/2608.15018)

**Authors**: Haochen Huang, Shengxuan Qiu, Meng Li  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2608.15018v1  

#### Abstract
Deploying large language models (LLMs) for inference on edge devices is challenging due to severe memory and bandwidth constraints. While speculative decoding and Mixture-of-Experts (MoE) have been proposed to improve inference efficiency, naively combining them often incurs excessive verification o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《S2-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在边缘设备上部署大规模语言模型（LLMs）面临严重的**内存带宽瓶颈**，尤其是对于采用 **Mixture-of-Experts (MoE)** 架构的模型。虽然 **speculative decoding (SD)** 被广泛用于提升推理效率，但将其直接应用于 MoE 模型时会遇到以下挑战：
- **验证开销过高**：每个 draft token 可能激活不同的专家（experts），导致大量冗余的参数访问。
- **专家重用率低**：被拒绝的 draft token 引发的专家激活无法复用，造成资源浪费。
- **draft 与 target 对齐差**：外部或简化 draft 模型预测质量不足，接受率低。

因此，**如何在保持高接受率的同时降低 MoE 模型 speculative decoding 的验证成本**，是本文要解决的核心问题。

---

### 提出了什么新方法或新思路
作者提出 **S2-MoE** ——一种面向边缘设备上 MoE 模型的高效 **self-speculative decoding** 框架，其三大核心技术组件为：

1. **Routing-aware Adaptive Speculative Expansion（路由感知自适应推测扩展）**
   - 动态选择是否扩展某个 draft token，基于一个综合考虑“接受概率”和“验证成本”的**效用得分（utility score）**。
   - 成本建模中引入了对新增专家数量的估计（`ΔC(i|S)`），避免选择那些虽有信心但会触发大量新专家的候选 token。

2. **Reuse-aware Expert Gating（重用感知专家门控）**
   - 在 draft 阶段有意引导门控机制优先选择已激活的专家，从而提高专家参数的重用率。
   - 这是一种可控的近似策略，在小幅度牺牲准确性的前提下显著减少内存访问。

3. **Context-aligned Self-speculative Decoding（上下文对齐的自推测解码）**
   - draft 和 target 共享相同的 **KV Cache** 上下文，防止因上下文不一致导致的错误累积，提升 draft 的预测准确性与接受率。

此外，S2-MoE 完全集成于 `llama.cpp` 中，实现了端到端优化。

---

### 相比现有方法的优势
| 方法 | 局限性 | S2-MoE 的改进 |
|------|--------|----------------|
| 外部 Draft Model (如 EAGLE-3) | 难以与 MoE 目标模型对齐；训练依赖强 | 无需额外训练，基于 self-speculative 实现天然对齐 |
| 简单剪枝/置信度驱动扩展 | 忽略专家级验证成本，无法控制冗余激活 | 显式建模专家激活成本，实现更优权衡 |
| 传统 self-speculative (如 ES/LS) | 缺乏参数重用机制，仍存在高验证开销 | 引入 reuse-aware gating 和共享 context，系统性降低开销 |

> ✅ 总体优势：**无需训练、轻量实现、高接受率、低验证成本、适合内存受限边缘场景**

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
根据不同模型的能力划分任务集合：

- **基础推理能力测试**：
  - GSM8K（数学推理）
  - HumanEval（代码生成）
  - ARC-E / ARC-C（常识推理）

- **高级推理能力测试**（针对更强模型如 Qwen3/GPT-OSS）：
  - GPQA Diamond（复杂问答）
  - AIME24 / AIME25 / HMMT（数学竞赛题）

- **长上下文与分布一致性测试**：
  - LongBench 数据集子集：GovReport, NarrativeQA, Qasper
  - WikiText-2：用于计算 KL 散度、logit shift、Top-1 准确率等输出一致性指标

---

### 实验设置和评估指标

#### 硬件平台
- **边缘设备**：NVIDIA Jetson Orin（16GB / 32GB / 64GB 内存配置）
- **高性能 GPU**：RTX 4090（模拟受限 GPU 内存环境）

> 注：当专家超出内存时，采用 SSD 或 CPU 内存进行 offload，符合实际边缘部署实践。

#### 主要评估指标
| 指标 | 含义 |
|------|------|
| **Speedup Ratio** | 相较于标准 autoregressive decoding 的加速比（核心指标） |
| **Tokens per second (Tok/s)** | 实际吞吐量 |
| **Acceptance Length** | 平均每次成功接受的 draft token 数量 |
| **Expert Reuse Ratio** | 已激活专家的重复利用率 |
| **Quality Metrics** | 包括任务准确率、PPL ratio、KL divergence、Top-1 agreement 等 |

---

### 基线方法对比
共比较五种 baseline：
1. **AR (Autoregressive)**：标准自回归解码（基准）
2. **ES (Expert Sparsity)**：通过减少每层激活专家数构建轻量 draft
3. **LS (Layer Sparsity)**：跳过部分 Transformer 层形成 draft
4. **EAGLE-3**：当前最先进的 speculative decoding 方法
5. **Cascade**：一种 MoE-aware 的动态启用 speculation 方法（基于 EAGLE-3 实现）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **Jetson Orin** 上：
  - **最高达 5.3× 加速**（平均约 2.0×）
  - 覆盖不同 MoE 模型（DeepSeek, OLMoE, Qwen3, GPT-OSS）和多种任务
- 在 **RTX 4090** 上：
  - 达到 **1.2× 至 2.9× 加速**

> 即使在 batch size=1 的极端内存受限条件下，S2-MoE 依然表现强劲。

---

### 与基线方法的对比结果
| 对比项 | 结果 |
|-------|------|
| vs. AR | 在所有模型和任务上均显著优于 AR，平均提速 >2× |
| vs. ES/LS | 自研 self-speculative 方法通常无法超越 AR（尤其在小内存下），而 S2-MoE 明显胜出 |
| vs. EAGLE-3 | 更稳定地取得更高加速比，不受特定模型家族影响（EAGLE-3 在某些模型上失效） |
| vs. Cascade | S2-MoE 使用细粒度 token-level 控制，避免了 Cascade 的历史决策延迟，效果更优 |

> 图7显示：S2-MoE 在各类设置下的 speedup 曲线全面领先。

---

### 消融实验结果
图8 展示了各模块的渐进贡献（ablation study）：

| 组件添加顺序 | 加速效果提升（以 Qwen3/GSM8K 为例） |
|-------------|-------------------------------|
| Self-Spec only | ~1.1× |
| + Context Alignment (CA) | ~1.5× |
| + Adaptive Expansion (AE) | ~1.8× |
| + Reuse-aware Gating (RG) → **S2-MoE** | **~2.5×** |

✅ 表明三个组件互补且必要。

#### 各组件具体影响：
- **Context Alignment**：将 acceptance rate 最高提升 **79%**（如 GSM8K 上的 DeepSeek）
- **Reuse-aware Gating**：expert reuse ratio 提升 **25–33%**（见 Table 4）
- **Adaptive Expansion**：有效抑制低性价比 draft 分支，降低冗余验证

---

## 4. 关键结论和发现

### 论文的主要发现
1. **MoE 模型中的 speculative decoding 成本主要来自冗余专家激活**，而非计算本身。
2. **token-level confidence 不足以指导 MoE 场景下的 speculative 扩展决策**，必须结合 routing-aware 成本建模。
3. **self-speculative decoding 是 MoE 推理的理想起点**，但需配合 context alignment 与 expert reuse 优化才能发挥最大潜力。
4. **S2-MoE 在无需训练的前提下，在多样化的 MoE 模型和边缘硬件上实现了稳定且显著的加速（1.3×~5.3×）**，同时保持输出质量基本不变。

---

### 方法的局限性
1. **Reuse-aware gating 引入了轻微路由偏差**，尽管实验证明其对最终任务性能影响极小（Top-1 Agreement >89%, PPL ratio ≈1.01），但仍属于有损近似。
2. 当前设计依赖于 draft-time 的 routing 预测信号来预估 target 验证成本，若 routing 不稳定可能导致成本估计不准。
3. 主要在 llama.cpp 框架中实现，尚未扩展至其他推理引擎（如 vLLM、TensorRT-LLM）。

---

### 未来工作方向
1. 将 S2-MoE 思路推广到更多类型的稀疏架构（如 block-wise sparsity、channel-wise MoE）。
2. 探索动态调整 reuse cap 的自适应机制，进一步提升鲁棒性。
3. 结合硬件感知调度（如专家预取、内存压缩）实现更深层次的系统协同优化。
4. 在真实边缘应用场景（如手机对话助手、车载 AI）中进行端到端部署验证。

---

> 🔗 开源地址：[https://github.com/angerybob/S2-MoE](https://github.com/angerybob/S2-MoE)

</details>

---

### 7. [Belayer: Efficient Fault Tolerance for LLM Agentic RL Training](https://arxiv.org/abs/2608.14635)

**Authors**: Jiecheng Zhou, Qinghao Hu, Peng Sun, Xingcheng Zhang, Weiming Zhang  
**Category**: cs.DC  
**Published**: 2026-08-18  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.14635v1  

#### Abstract
Large language model (LLM) agents are increasingly trained with reinforcement learning in long-horizon, sandboxed environments. Unlike conventional RL, agentic RL couples GPU-intensive rollout engines with stateful environment containers whose actions may produce visible side effects, such as file e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Belayer: Efficient Fault Tolerance for LLM Agentic RL Training**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **LLM Agentic RL Training** 中，训练过程涉及长时间、多轮次的 **LLM 推理** 与 **沙盒环境交互**（如文件修改、命令执行等），其轨迹状态跨越 LLM 上下文与容器运行时状态。传统容错机制（如全量重启）存在以下问题：

- **Rollout Engine 故障恢复开销大**：冷启动需重新初始化 GPU 状态，耗时数十秒。
- **环境故障导致轨迹不一致**：直接重启轨迹会丢失已完成的工作；若无对齐的检查点，则无法保证 LLM 上下文与环境状态的一致性。
- **现有系统缺乏高效且正确的恢复机制**。

### 🚀 提出的新方法与创新点

#### （1）**Selective GPU-State Reuse Protocol（选择性 GPU 状态复用协议）**
- 引入 **Shadow Worker（影子工作节点）**，预初始化以避免故障路径上的昂贵初始化。
- 设计细粒度状态隔离机制：
  - **保留可重用状态**：模型权重 `Weights` 和原始 KV-Cache 内存池（`KV Arena`）由独立的 `Weight Server` 和 `KV-Cache Server` 管理，在主 Worker 失败后可通过健康检查安全复用。
  - **重建易失状态**：请求级 KV 内容、CUDA Graph 缓冲区、调度器状态等在 Shadow Worker 中重建。
- 利用 **token prefix 日志** 重建中断请求的上下文。

> 🔍 *创新洞察*：基于对 vLLM、SGLang 等推理引擎真实故障分析发现，**模型权重和 KV Arena 很少被故障破坏**，而进程上下文和完整 KV 缓存则高度不可靠。

#### （2）**LLM-Response-Aware Environment Checkpointing（LLM 响应感知的环境检查点机制）**
- 在每次环境动作完成后、LLM 生成响应期间，利用“推理气泡”时间窗口进行 **full_checkpoint**。
- 提出 **adaptive risk-aware policy** 动态决策是否执行检查点：
  - 权衡：预期恢复收益 vs. 可能暴露于关键路径的延迟成本。
  - 公式化建模：  
    $$
    \text{Benefit } B_t = p_t \cdot C_t \quad (\text{失败概率} \times \text{再生代价}) \\
    \text{Visible Overhead } O_t = (1 - q(x; c)) \cdot c_t \quad (\text{未被隐藏的概率} \times \text{检查点耗时})
    $$
    当 $ B_t \geq O_t $ 时触发检查点。
- 使用 **Docker Overlay + CRIU** 实现原子化的文件系统与运行时状态联合快照（`full_checkpoint` / `full_restore`）。
- 检查点与 LLM 上下文绑定，确保 **prefix consistency**。

### ⚖️ 相比现有方法的优势

| 方面 | 现有方法 | Belayer |
|------|----------|---------|
| **Rollout 恢复速度** | 冷启动（~30–40s） | **~1s 内热切换**（提速达 42×） |
| **环境恢复一致性** | 重启整个轨迹或部分状态不一致 | **前缀对齐恢复**，保持轨迹语义正确 |
| **检查点开销控制** | 固定频率（如每步都 checkpoint） | **自适应策略**，96%+ 时间隐藏在 LLM 推理中 |
| **资源利用率** | Shadow Worker 占用高 GPU 显存 | 通过 CUDA VMM 解绑缓冲区，**显存仅增 1.84GB** |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Math Reasoning**：`dapo-17k-math` 数据集，测试数学推理能力。
- **Software Engineering (SWE)**：`SWE-Gym` 数据集，模拟代码修复任务。

### 🧪 实验设置
- **模型**：Qwen3 系列（4B, 8B, 32B 参数）
- **硬件集群**：
  - 4 节点，共 32 块 NVIDIA H200 GPU，RoCE 互联（400Gbps × 8）
  - 环境集群：Ubuntu 24.04，16 核 CPU，32GB RAM
- **并行配置**：
  - Rollout 阶段使用 TP（Tensor Parallelism）
  - Training 阶段使用 DP+TP
- **训练算法**：GRPO（Generalized Reward Policy Optimization）
- **Batch Size**：64 rollout batch，每 prompt 采样 8 次
- **最大回合数（SWE）**：20 rounds

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **End-to-End Training Time** | 完整训练步骤耗时，为主要性能指标 |
| **Recovery Time** | Worker 或环境故障后的恢复延迟 |
| **Throughput & Memory Overhead** | 故障自由场景下的性能影响 |
| **Regeneration Cost** | 因失败需重新生成的 token 数或时间 |
| **Checkpoint Exposure Time** | 检查点未能被 LLM 推理“气泡”掩盖的时间 |

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **Slime（无容错）** | 原始框架，作为性能基准 |
| **Cold Restart Baseline** | 失败后从头加载模型（优化版：通过高速网络拉取权重，约 5s 加载 Qwen3-32B） |
| **Restart-from-Scratch (Env)** | 环境失败后重启整个轨迹 |
| **Fixed Policy Checkpointing**：
  - `Always`：每个 action 后都 checkpoint
  - `Every-3`：每 3 个 action checkpoint 一次 | 用于比较自适应策略的有效性 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**Rollout Worker 恢复效率**
- **恢复时间**：
  - Belayer：平均 **~1 秒**
  - Cold Restart：**38.5–43 秒**
  - ➜ **加速高达 42×**
- **单次软件故障对训练时间的影响**：
  - Belayer：增加 **1.16%**
  - Baseline（冷启）：增加 **8.72%**
- **Token-Level Context Recovery 贡献**：额外减少 3.46% 时间开销。

#### （2）**Shadow Worker 开销极低**
- **GPU 显存占用**：仅 **+1.84GB**
- **吞吐量影响**：与无 Shadow Worker 场景基本一致（见 Figure 8）

#### （3）**环境恢复性能**
- **恢复时间对比**：
  - Belayer：**1.5× – 3.5× 快于基线**
- **端到端训练时间增长（含故障）**：
  - 基线（重启轨迹）：**+48.7%**
  - Belayer（自适应检查点）：仅 **+1.5% – 2.1%**
- **检查点隐藏率**：
  - 自适应策略下，**96.8% 的检查点时间被 LLM 推理气泡覆盖**

#### （4）**不同存储后端下的检查点开销（Table 5）**
| 存储 | 策略 | E2E 时间增长 | 总检查点时间 | 暴露时间 |
|------|------|----------------|----------------|------------|
| HDD | Always | 276.7% | 45,122.3s | 3,130.6s |
| HDD | Every-3 | 44.4% | 3,989.1s | 188.6s |
| HDD | **Ours (Adaptive)** | **21.6%** | **641.7s** | **20.3s** ✅ |
| SSD | **Ours** | **2.1%** | 967.0s | 14.0s ✅ |

> ✅ 表明 Belayer 的自适应策略能显著降低实际可见开销。

#### （5）**消融实验**
- **Shadow Worker Co-location**：验证了 idle shadow 不干扰主 worker 性能。
- **Adaptive Policy vs Fixed**：证明动态策略在不同故障率和 batch size 下均优于固定频率。
- **Scaling 测试**：
  - 随着故障频率上升，Belayer 训练时间增长平缓；
  - 基线因频繁冷启动导致性能急剧下降，尤其当失败节点超过可用健康节点时（无法远程加载权重）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **GPU 状态具有强可复用性**：模型权重和 KV Arena 在大多数推理故障中未受损，支持选择性复用。
2. **Shadow Worker + Selective Reuse 可实现近实时恢复**：将 rollout worker 恢复时间从数十秒压缩至 **1 秒内**。
3. **LLM 推理间隙是天然的检查点窗口**：合理利用可几乎完全隐藏环境检查点开销。
4. **自适应检查点策略至关重要**：静态策略在 HDD 上代价极高，而 Belayer 能根据风险动态调整，实现最优权衡。
5. **Prefix Consistency 是轨迹正确性的基础**：必须同时恢复 LLM 上下文、文件系统和运行时状态至同一合法前缀。

### ⚠️ 局限性
1. **依赖特定容器技术栈**：
   - 当前仅支持 Linux Containers、Docker Overlay 和 CRIU；
   - 不适用于 VM 或其他隔离机制。
2. **外部副作用无法恢复**：
   - 如 bind-mounts、host daemon、device state、live network connections 等不在恢复范围内。
3. **不处理硬件级损坏**：
   - CUDA context corruption、GPU 硬件故障、driver reset 等超出当前设计范围。
4. **Shadow Worker 假设健康检测可靠**：若 Shadow 自身也发生故障，可能引发 cascading failure。

### 🔮 未来工作方向
1. **扩展至更复杂的环境拓扑**：支持与外部服务交互的 agent 的容错恢复。
2. **Flush-Free KV-Cache Recovery**：探索无需清空 KV-Cache 的更轻量恢复方式。
3. **硬件故障应对机制**：
   - 引入 live migration、elastic rollout scaling 等弹性调度策略。
4. **更乐观的恢复策略**：尝试 partial recovery 而非全量重建。

---

> 💡 **总结一句话**：  
> **Belayer 通过 “选择性 GPU 状态复用” + “响应感知的自适应检查点” 实现了高效、一致的 LLM Agentic RL 容错训练，在几乎零开销的前提下，将恢复速度提升数十倍，并保障了训练轨迹的语义正确性。**

</details>

---

### 8. [RLCascadeRouter: Quality-Estimator-Free Cascade Routing via Reinforcement Learning](https://arxiv.org/abs/2608.15817)

**Authors**: Shihong Huang, Shengjie Wang, Hong Ma, Zhou Xu  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.15817v1  

#### Abstract
The growing ecosystem of large language models (LLMs) offers huge potential to optimize performance-cost trade-offs. However, their heterogeneous capabilities and inference costs make efficiently routing queries a significant challenge. Existing paradigms are inflexible: one-shot routers commit befo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**RLCascadeRouter: Quality-Estimator-Free Cascade Routing via Reinforcement Learning**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **LLM 路由（routing）** 和 **级联（cascading）** 方法存在以下限制：
- **One-shot routing**：在生成响应前就做出模型选择决策，无法根据实际输出动态调整。
- **Fixed-order cascading**：虽然可以动态停止，但模型调用顺序是固定的，不能跳过或重排序。
- **Predict-then-optimize 范式依赖质量估计器（quality estimator）**：通过预测响应质量来决定是否停止或继续，但这种“预测损失”与“决策损失”不一致，导致小的预测误差可能引发错误的路由决策。

### 🚀 提出的新方法
提出 **RLCascadeRouter**，一种基于强化学习（Reinforcement Learning, RL）的 **quality-estimator-free cascade routing 框架**，其核心思想包括：

#### （1）将 cascade routing 建模为 **Markov Decision Process (MDP)**
- **状态（State）**：包含查询 `q`、当前响应 `y`、已选模型历史 `H`、累计成本 `C`、当前深度 `t`。
- **动作空间（Action Space）**：每一步同时考虑 `"STOP"` 和所有未使用的候选模型，实现真正的动态决策。
- **奖励函数（Reward）**：直接优化 performance-cost trade-off，使用轨迹回报（trajectory return），避免中间预测环节。

#### （2）设计 **Cascade Policy Network (CPN)**
- **Complementarity Encoder (CE)**：建模剩余模型之间的能力互补性，用于更优的模型选择。
- **Value-Aware Stopper (VAS)**：联合学习“停止”与“继续”的价值，无需独立的质量评估器或阈值。

#### （3）完全消除对 quality estimator 的依赖
- 不再需要训练额外的 response quality predictor 或 calibrated stopping threshold。
- 决策直接从端到端的 RL 反馈中学习，解决了 **Prediction-Decision Mismatch** 问题。

### 🔍 相比现有方法的优势
| 特性 | 传统方法 | RLCascadeRouter |
|------|--------|------------------|
| 动态模型选择 | ❌ 固定顺序 | ✅ 支持任意顺序 |
| 动态停止机制 | ⚠️ 依赖 quality estimator | ✅ 学习式停止，无外部估计器 |
| 决策一致性 | ❌ 预测误差可能导致决策翻转 | ✅ 直接优化最终目标 |
| 泛化能力 | ❌ 新模型需重新训练 | ✅ 支持 unseen models 无需 retraining |

---

## 2. 核心实验方法和设置

### 📚 数据集
在 **LLMRouterBench [Li et al., 2026]** 上进行评估，涵盖 **10 个 benchmark**，覆盖多个任务领域：
- **数学推理**：AIME, LiveMathBench (LMB)
- **代码生成**：LiveCodeBench (LCB), SWE-bench (SWE)
- **知识问答**：GPQA, Humanity's Last Exam (HLE), MMLU-Pro, SimpleQA
- **指令跟随**：ArenaHard
- **工具使用**：T2-Bench (Tau2)

### 🧪 实验设置
- **候选模型池**：共 13 个主流 LLMs，如 GPT-5、Gemini-2.5-Pro、Qwen3-235B、DeepSeek 系列等。
- **最大调用次数**：每个查询最多调用 3 个模型。
- **训练配置**：
  - 使用 **PPO** 算法训练。
  - 网络结构：6 层 Transformer 编码器，隐藏维度 256，8 头注意力。
  - 训练 100 轮，每轮 2048 条轨迹。
  - 划分比例：60% 训练，10% 验证，30% 测试。

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Performance (宏观平均得分)** | 各数据集上最后一个模型输出的标准化分数的宏平均 |
| **Cost** | 整个 cascade 过程中所有模型调用的 API 成本总和 |

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **Single-model baselines** | GPT-5, Gemini-2.5-Pro, Qwen3-235B-Thinking 等 13 种独立模型 |
| **Routing baselines** | |
| &nbsp;&nbsp;• Random Router | 随机选择一个模型 |
| &nbsp;&nbsp;• HybridLLM | 二元路由，基于难度判断强弱模型切换 |
| &nbsp;&nbsp;• FrugalGPT | 学习质量评分器并设定接受阈值 |
| &nbsp;&nbsp;• GraphRouter | 图神经网络建模任务-查询-模型关系 |
| &nbsp;&nbsp;• Avengers-Pro | 基于聚类和统计的高性能路由策略 |

---

## 3. 主要实验结果和性能指标

### 📈 总体性能（见 Table 1）
| 方法 | Avg. Performance ↑ | Total Cost ↓ |
|------|--------------------|--------------|
| GPT-5 | 65.12 | 124.81 |
| Avengers-Pro | 67.84 | 135.65 |
| **RLCascadeRouter (Ours)** | **68.81** | **180.22** |

- 在 **平均性能上达到 68.81%**，优于最强 baseline **Avengers-Pro (+0.97%)** 和最强单模型 **GPT-5 (+3.69%)**。
- 尽管总成本较高，但在高性价比区域表现更优（见下图）。

### 💰 Performance-Cost Trade-off 分析（见 Figure 4）
- 在中高绩效区间，RLCascadeRouter 提供了 **更优的帕累托前沿（Pareto frontier）**：
  - 在 ~59% 性能时，成本仅为 **10.64**，相比 Avengers-Pro (**16.12**) 降低 **34.0%**。
  - 在 ~67% 性能时，以 **85.41** 成本达成 67.29%，优于 Avengers-Pro 的 67.22% @ 107.88（**节省 20.8% 成本**）。
  - 最高可达 **68.08% @ 108.07**，超过 Avengers-Pro 的上限（67.93%）。

> ✅ 表明 RLCascadeRouter 并非仅靠“堆模型”提升性能，而是在多种预算下均能提供更好权衡。

### 🔁 泛化能力测试（见 Table 2）
在 **unseen models** 场景下的表现：

| 设置 | 描述 | Avg. Perf | Cost | Unseen Selection Rate |
|------|------|---------|-------|------------------------|
| A | 原始模型池训练 + 原始测试 | 59.32 | 82.74 | 0.00% |
| B | 原策略 + 替换三个模型（未重训） | **53.97** | **72.76** | **98.18%** |
| C | 重训练新模型池 | 60.61 | 174.76 | 97.59% |

- **无需 retraining 的情况下保留了 89–91% 的性能**。
- **98.18% 的查询选择了新引入的模型**，说明模型描述有效引导了泛化路由。

### 🔍 消融实验（Ablation Study，见 Table 3）

| 变体 | θ=1.0 Perf/Cost | θ=0.5 Perf/Cost | θ=0.2 Perf/Cost |
|------|------------------|------------------|------------------|
| Full (完整模型) | **68.81 / 180.22** | **61.91 / 28.06** | **58.02 / 10.13** |
| w/o CE | 68.30 / 152.60 | 60.32 / 21.05 | 56.81 / 7.23 |
| w/o VAS | 67.43 / 191.56 | 60.67 / 15.80 | 58.62 / 9.97 |
| QE Stop | 67.70 / 135.76 | 61.23 / 27.43 | 58.02 / 10.13 |

#### 结论：
- **移除 CE（Complementarity Encoder）** → 性能下降，尤其在高 θ 下，说明模型间互补性建模重要。
- **移除 VAS（Value-Aware Stopper）** → 多步场景下性能显著下降，表明 action context 对 stopping 至关重要。
- **替换为 quality estimator-based stopping (QE Stop)** → 即使成本更低，性能仍低于端到端学习的 stopping 策略，验证了 **predict-then-optimize 的局限性**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Prediction-Decision Mismatch 是现有 cascade routing 的根本瓶颈**：即使预测准确率提高，也可能因边界穿越导致决策错误。
2. **将 cascade routing 建模为 MDP 并用 RL 直接优化 trajectory return 是有效的解决方案**。
3. **RLCascadeRouter 实现了更强的 performance-cost trade-off**，在多个 benchmark 上超越现有最强路由方法。
4. **无需 quality estimator 和 stopping threshold**，简化部署流程，增强鲁棒性。
5. **支持对 unseen models 的零样本迁移**，仅需提供文本描述即可纳入路由决策。

### ⚠️ 方法的局限性
- **总成本偏高**：为了追求更高性能，有时会调用更多模型，适合对延迟不敏感但对质量要求高的场景。
- **依赖模型描述 embedding**：若描述不准确或误导，可能影响路由效果。
- **训练开销较大**：需要大量 trajectory 数据和 RL 训练资源。

### 🔮 未来工作方向
- 扩展至 **online learning setting**，适应动态变化的模型池、成本、延迟分布。
- 探索 **token-level cascade**，而非整个模型级别的调用。
- 引入 **multi-agent coordination** 机制，支持更复杂的 agentic workflow 路由。
- 结合 **uncertainty estimation** 与 RL 策略，进一步提升 stopping 判断精度。

---

> **一句话总结**：  
> RLCascadeRouter 通过 **强化学习直接优化 cascade routing 的最终目标**，摆脱了传统方法对 quality estimator 的依赖，在性能、灵活性和泛化性方面实现了全面突破。

</details>

---

### 9. [Q-First: Attention and Feed-Forward Concurrency at the Smallest Change to the Block](https://arxiv.org/abs/2608.15473)

**Authors**: WenJie Fan  
**Category**: cs.DC  
**Published**: 2026-08-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.15473v1  

#### Abstract
Disaggregated LLM serving puts the KV-cache sweep on memory-optimised hardware and the projections and feed-forward on compute-optimised hardware, then inherits from the decoder block a dependency neither device wants: attention runs first and the feed-forward consumes its output, so within one sequ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Q-First: Attention and Feed-Forward Concurrency at the Smallest Change to the Block**

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在 **disaggregated LLM serving** 架构中，KV-Cache 被部署在内存优化硬件上，而注意力计算和 FFN（Feed-Forward Network）则运行在算力优化硬件上。然而，标准的 decoder block 结构要求 **attention 必须先于 FFN 执行**，导致两个设备在单个序列处理中交替空闲，无法并发执行，降低了硬件利用率。

传统解决方案是通过 **多序列并行（keeping multiple sequences in flight）** 来掩盖空闲时间，但这需要为每个序列维护一份驻留的 KV-Cache，违背了分离架构节省内存的初衷。

---

### 提出的新方法或新思路

论文提出 **Q-First 协议**，其核心思想是：

> **让 FFN 先运行，同时提前将 query 向量发送给 cache-side 设备，使其可以立即开始对历史 KV-Cache 进行 attention sweep，而无需等待当前 token 的 key 和 value。**

这一改变的关键在于：
- Cache-side 设备只需要 **query** 就能启动 attention 计算（因为 sweep 只依赖缓存中的历史 keys/values）。
- 当前 token 的 key 和 value 是作为写操作（write）异步到达 cache-side 的，不阻塞 sweep。
- 通过重新排序 block 内部子层（sub-layer reordering），使得 query 可以从 block 输入直接生成，早于 FFN 输出。

该方法被称为 “**最小改动（smallest change）**”，因为它：
- 不引入新的 operator 或 kernel；
- 不改变 tensor 形状或参数数量；
- 不修改模型结构图（computation graph）本身，仅调整读取顺序（reading order）；
- 完全兼容现有的 FlashAttention 等 fused kernel。

---

### 相比现有方法的优势

| 方面 | Q-First 优势 |
|------|-------------|
| **并发性** | 实现 attention sweep 与 FFN 计算真正并发，消除设备间等待 |
| **内存效率** | 无需额外维护多个序列的 KV-Cache 来填充空闲周期 |
| **兼容性** | 可运行在标准框架和 stock kernels 上，无需定制硬件或算子 |
| **训练影响小** | 实验证明 query 提前读取对训练轨迹扰动极小（< 0.0026 bpb） |
| **可扩展性强** | 支持 cache 分片到任意数量设备，并行 sweep 后合并结果 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- 主要训练数据：**WikiText-103**，共 **372M tokens**
- 训练时长：3 epochs，约 **11,355 optimiser steps**
- 序列长度：1,024
- 每步 token 数：32,768

> 注：性能验证部分还使用了合成张量和官方发布的 Qwen3-0.6B checkpoint 进行端到端测试。

---

### 实验设置

- **模型架构**：基于 **Qwen3-0.6B**（596.05M 参数）
- **训练配置**：
  - AdamW 优化器
  - Cosine 学习率调度，峰值 LR = 4×10⁻³
  - 使用 **bfloat16** 精度，normalization weights 保持 float32
  - **tied embedding 层冻结**（避免初始化偏差干扰比较）
  - 所有 runs 编译加速（compiled）
- **每组实验重复两次（two seeds）**，报告双种子结果而非均值，强调可复现性和扰动大小判断

---

### 评估指标

- **主要指标**：**bits per byte (bpb)**，衡量语言建模 loss
- **相对误差**：与原始 fused attention 对比的输出差异（relative error）
- **收敛轨迹分析**：绘制训练过程中各变体与 control 的差距变化
- **消融维度**：移动 query/key/value 的读取时机（read point）

---

### 基线方法对比

| 方法 | 描述 |
|------|------|
| **Standard** | 原始 attention-first block |
| **Swapped** | FFN 先执行，attention 后执行（control for Family A） |
| **Parallel Block** | Wang & Komatsuzaki [2021] 提出的并行 attention + FFN 结构（不同 computation graph） |
| **Shifted-Q** | 将 query 从前一层 FFN 输入处读取（half-sublayer shift） |
| **Single-Q / Shared Query** | 所有层的 query 都从网络输入处投影（极端前置） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Q-First 协议精度验证
- 在真实 Qwen3-0.6B checkpoint 上进行端到端验证
- 与 fused attention 输出对比，**相对误差仅为 `3.2×10⁻⁵`**
- 表明协议在数学上等价且数值稳定

#### ✅ 并发机制可行性
- 提前发送 query 不依赖新算子，可在 stock FlashAttention kernel 上实现
- attention-side 返回 `(o_<j, lse_<j)` 可被主设备用于最终组合
- 组合公式为 logistic 形式的插值（lerp），无溢出风险（经 [-800, 800] 区间验证）

---

### 与基线方法的对比结果

#### 📊 Family A：FFN 不消费本层 attention 输出（即 swapped 结构）

| 方法 | bpb (seed 0) | bpb (seed 1) | Δ vs Swapped |
|------|--------------|--------------|------------|
| Swapped (control) | 1.2388 | 1.2381 | — |
| Early-K | 1.2352 | 1.2365 | -0.0026 |
| Early-Q | 1.2369 | 1.2403 | +0.0001 |
| Early-QK | 1.2369 | 1.2377 | -0.0011 |
| Parallel | 1.2335 | 1.2400 | -0.0017 |

> 🔍 **所有变体之间的最大差异仅 0.0027 bpb，小于 run-to-run 差异（floor = 0.010 bpb）**

👉 结论：**移动 read point 几乎不影响训练结果，说明 Q-First 是“不可察觉”的改动**

---

#### 📊 Family B：标准 attention-first 结构

| 方法 | bpb (seed 0) | bpb (seed 1) | Δ vs Standard |
|------|--------------|--------------|-------------|
| Standard | 1.2998 | 1.3038 | — |
| Shifted-Q | 1.2862 | 1.2897 | **-0.0139** |
| Single-Q | 1.3976 | 1.4008 | **+0.0974** |

> ⚠️ **Single-Q 显著劣化（+0.0974 bpb）**，且超过预注册拒绝阈值 (+0.05)，被明确 refuted

👉 结论：**query 最多只能提前一个 FFN 层（即从上一层输入读取），不能从全局输入统一生成**

---

### 消融实验结果

| 发现 | 说明 |
|------|------|
| **Early-Q vs 控制组差异 < 0.0001 bpb** | 移动 query 读取点几乎无代价 |
| **Early-K 差异达 -0.0026 bpb** | 但仍在噪声范围内，不影响结论 |
| **Sub-layer exchange 差异达 0.025+ bpb** | 是 read-point 移动的 25 倍以上 → 仪器灵敏度足够 |
| **Shared Query 移动达 38 倍于 read-point 变化** | 证明测量系统能分辨显著扰动 |

> 💡 特别指出：**Family A 内部所有 read-point 变化都小于单个 arm 自身双种子差异（0.0066 bpb）**  
这意味着：“**模型无法感知这些变化**” —— 正是协议所需的理想属性。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Cache-side 设备只需 query 即可启动 attention sweep**，无需等待当前 key/value。
2. ✅ **将 FFN 提前、query 从 block 输入读取**，即可实现完全并发，且改动极小。
3. ✅ **该改动对训练质量无显著影响**（Δ < 0.0026 bpb），远低于训练随机性。
4. ✅ **attention 中间状态 `(o_<j, lse_<j)` 是 mergeable aggregate**，支持 cache 分布到任意数量设备并独立 sweep。
5. ✅ **最多允许 query 提前一个 FFN 层**；若从网络输入统一生成，则性能严重下降（+0.0974 bpb）。
6. ✅ **无需 custom kernel 或 operator**，可在 FlashAttention 等现有系统中实现。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| ❌ 未实测推理速度提升 | 实验在单卡完成，stream 会串行化所有操作；实际收益需双设备 + decode loop 验证 |
| ❌ 长上下文下收益有限 | 当 context > ~4.6K（Qwen3-0.6B）时，sweep 成为主导开销，无法再被隐藏 |
| ❌ 依赖 batching on cache-side | 若 cache-side 不 batch，Wq/Wo 投影成本过高（unbatched 达 2.3ms/token） |
| ❌ Family gap 未解释清楚 | Swapped 架构整体优于 Standard 约 -0.0634 bpb，但原因尚不确定（可能与 frozen embedding 有关） |
| ❌ 仅在一个 scale 和 corpus 上验证 | 是否泛化至更大模型或更多数据未知 |

---

### 未来工作方向

1. **构建真实 disaggregated 系统**，测量跨设备延迟与 FFN 时间比，验证实际吞吐增益。
2. **探索动态 batching 策略**，适配不同 context 长度请求混合场景。
3. **研究更激进的 read-point 调整**，例如局部共享 query 或 conditional early-read。
4. **解冻 embedding 层重新训练**，澄清 Family A 与 Standard 的性能差距是否真实存在。
5. **结合 GQA/MQA 架构进一步压缩通信量**，提升跨设备效率。

---

## 总结一句话

> **Q-First 通过最微小的 block 重构（仅移动 query 读取点），实现了 attention 与 FFN 的完全并发执行，在不牺牲精度的前提下解锁了 disaggregated serving 的硬件并行潜力，是一种“免费的午餐”级优化。**

</details>

---

### 10. [Adaptive Heterogeneous Compression for Resource-Efficient Federated Knowledge Distillation](https://arxiv.org/abs/2608.15660)

**Authors**: Chenwang Liu, Yijun Liu, Chang Liu, Xu Zhang, Pengchao Han  
**Category**: cs.DC  
**Published**: 2026-08-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.15660v1  

#### Abstract
Federated learning (FL) enables privacy-preserving distributed model training but faces challenges from heterogeneous model architectures and limited communication resources at the network edge. Federated knowledge distillation (FedKD) alleviates model heterogeneity by combining prototype-wise param...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Adaptive Heterogeneous Compression for Resource-Efficient Federated Knowledge Distillation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在 **Federated Knowledge Distillation (FedKD)** 中，客户端通常具有异构的模型架构（如不同深度/宽度的神经网络）和资源条件（计算能力、通信带宽）。尽管 FedKD 能缓解模型异构性，但其仍需频繁传输梯度，导致显著的 **communication overhead**。现有压缩方法（如 Top-K、Random-K）通常采用**统一策略**应用于所有客户端，忽略了客户端在模型大小、计算能力和通信环境上的差异，从而可能导致次优的效率-性能权衡。

### **提出的新方法与创新思路**
本文提出了 **ASCEND (Adaptive heterogeneouS Compression algorithm for fEderated kNowledge Distillation)**，一个面向 FedKD 的自适应异构压缩框架，其核心创新如下：

- **异构压缩框架**：允许每个客户端从候选集 $ \mathcal{S} = \{\text{Top-K}, \text{Random-K}, \text{Periodic-K}\} $ 中**独立选择**最适合自身条件的压缩策略。
- **MAB 建模**：将压缩策略选择问题建模为 **非平稳随机多臂赌博机 (non-stationary stochastic Multi-Armed Bandit, MAB)** 问题，其中每种压缩策略对应一个“臂”。
- **效率感知奖励函数**：设计了一个综合考虑以下因素的奖励：
  - **本地优化改进**（$ \Delta f_n $）
  - **全局知识对齐**（$ \Delta \mathcal{L}_p $）
  - **执行时间开销**（$ T(s) $）
  
  奖励定义为：  
  $$
  G(s) = \frac{\Delta \mathcal{L}_n(t)}{T(s)}
  $$
  通过归一化提升效率，鼓励快速收敛且低开销的策略。
- **EMA增强的ε-greedy策略**：采用指数移动平均（EMA）来平滑奖励估计，结合随训练轮次衰减的探索率 $ \epsilon_t $，实现探索-利用平衡。
- **回滚稳定性保障机制**：当检测到全局损失突增时，系统回滚至前一轮状态，并强制所有客户端暂时使用 Top-K 策略，以防止不稳定训练。

### **相比现有方法的优势**
- **个性化适配**：突破“一刀切”的压缩策略，实现客户端级别的动态策略选择。
- **动态适应性**：能随训练进程自动调整策略（早期偏好性能，后期偏好效率）。
- **理论保证**：提供了收敛性分析（$ O(1/\sqrt{T}) $）和亚线性后悔界（$ O(\sqrt{MT}) $），证明其长期有效性。
- **鲁棒性强**：在不同数据异构性（non-IID）、通信带宽下均表现稳定。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MNIST**：手写数字识别，用于轻量级模型测试。
- **CIFAR-10**：图像分类，更具挑战性。

数据划分采用 **Dirichlet 分布**模拟 non-IID 场景，参数 $ \alpha = 0.5 $（默认），$ \alpha \in \{0.1, 0.5, 1\} $ 用于测试数据异构性影响。

### **模型配置**
| 数据集     | 模型           | 特点描述 |
|------------|----------------|----------|
| MNIST      | LeNet5Half     | 浅层、紧凑，约 1.57万 参数 |
| MNIST      | LeNet5         | 更深，约 6.17万 参数 |
| CIFAR-10   | WResNet40-2    | 宽残差网络，约 224万 参数 |
| CIFAR-10   | ResNet18       | 深层残差网络，约 1118万 参数 |

### **实验设置**
- **真实平台**：10 台 Raspberry Pi 构成边缘设备集群。
- **仿真环境**：基于 NVIDIA RTX 3090 和 AMD Ryzen 7 4800H 进行时间建模。
- **压缩级别 K**：在多个数量级上测试（如 $ K=10^2, 10^3, 5\times10^3, 10^5, 2\times10^6 $）。
- **通信带宽**：测试 5 Mbps、50 Mbps、500 Mbps 不同网络条件。
- **评估指标**：
  - **测试准确率 (Accuracy)**
  - **训练时间 (Training Time)**
  - **通信开销 (Communication Overhead)**
  - **策略选择动态演化**

### **基线方法对比**
- **Uniform Compression**：
  - Top-K
  - Random-K
  - Periodic-K
- **自适应基线**：
  - **EXP3**：一种对抗性 MAB 算法，用于对比非平稳场景下的适应性。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 在 **MNIST** 上，ASCEND 在 $ K=10^4 $ 时达到最佳性能，**准确率接近 97%**，且收敛速度最快。
- 在 **CIFAR-10** 上，ASCEND 在多种 $ K $ 设置下均优于固定策略，尤其在中等压缩比下优势明显。

### **与基线方法的对比结果**
| 场景 | ASCEND 表现 |
|------|-------------|
| **不同压缩级别** | 在所有 $ K $ 下均优于或等于最佳固定策略（Top-K 或 Random-K 随 $ K $ 变化），而 ASCEND 自动适应并保持最优。 |
| **不同数据异构性 ($ \alpha $)** | 在 $ \alpha=0.1 $（高度 non-IID）下，ASCEND 仍能稳定收敛，性能优于所有基线。 |
| **不同通信带宽** | 在低带宽（5 Mbps）下，Top-K 占优，ASCEND 快速学习并倾向选择 Top-K；在高带宽（500 Mbps）下，Random-K 更高效，ASCEND 同样能正确识别并切换。 |
| **vs EXP3** | ASCEND 在多数场景下优于 EXP3，表明其针对优化动态设计的奖励机制更有效。 |

### **消融实验与策略选择分析**
- **策略选择动态**（见 Fig. 9–12）：
  - 小模型（LeNet5Half）倾向于选择 **Top-K**（排序开销小，精度高）。
  - 大模型（LeNet5, ResNet18）在低 $ K $ 时偏好 Top-K，在高 $ K $ 时转向 **Random-K**（降低选择开销）。
  - 验证了 ASCEND 能根据 **模型大小、压缩级别、系统开销** 自主决策。
- **敏感性分析**：
  - 权重参数 $ \rho=0.6, \beta=0.4 $（强调全局知识）表现最佳。
  - 探索超参 $ c=3 $ 在 ε-greedy 中取得最佳平衡。

---

## **4. 关键结论和发现**

### **主要发现**
1. **没有单一最优压缩策略**：Top-K、Random-K、Periodic-K 的优劣高度依赖于客户端的模型架构、资源能力和当前训练阶段。
2. **自适应选择至关重要**：ASCEND 能动态识别最优策略，实现 **通信效率、计算开销与模型精度的最佳平衡**。
3. **MAB 建模有效**：将策略选择建模为非平稳 MAB 是合理且高效的，EMA 增强的 ε-greedy 策略能快速响应变化。
4. **鲁棒性强**：在不同数据分布、网络带宽下均表现出色，适用于真实边缘环境。

### **方法的局限性**
- 当前仅支持预定义的三种压缩策略（Top-K, Random-K, Periodic-K），未涵盖量化、稀疏化等其他压缩技术。
- 奖励函数依赖服务器反馈（如全局损失），增加了协调复杂性。
- 理论分析基于 piecewise-stationary 假设，实际训练过程可能更复杂。

### **未来工作方向**
- **联合优化**：同时优化 **压缩策略选择** 与 **压缩级别 $ K $** 的动态调整。
- **扩展策略空间**：纳入更多压缩方法（如量化、混合压缩）。
- **去中心化实现**：减少对中央服务器反馈的依赖，提升可扩展性。
- **跨任务泛化**：验证 ASCEND 在 NLP、语音等任务中的有效性。

--- 

> **总结**：ASCEND 提出了一种面向 FedKD 的**自适应异构压缩框架**，通过 MAB 实现客户端级策略动态选择，在多个数据集和真实平台上验证了其在降低通信开销和训练时间的同时，**维持甚至提升模型准确率**的能力，为资源受限的联邦学习系统提供了高效解决方案。

</details>

---

### 11. [EcoVLA: Energy-Efficient Device-Edge Co-Inference for Vision-Language-Action Models under Real-Time Constraints](https://arxiv.org/abs/2608.15502)

**Authors**: Ao Zhou, Bo Dai, Le Yu, Xingyu Liu, Zeyu Hao, Lingkun Long, Chunming Hu, Jianlei Yang  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.15502v1  

#### Abstract
Vision-Language-Action (VLA) models have emerged as a promising foundation for Embodied AI, but their high inference cost poses significant challenges for deployment in robotic systems. In practice, on-device inference is constrained by limited compute capacity and energy budgets, struggling to simu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EcoVLA: Energy-Efficient Device-Edge Co-Inference for Vision-Language-Action Models under Real-Time Constraints

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Vision-Language-Action (VLA) 模型在具身智能（Embodied AI）中展现出强大能力，但其高昂的推理成本使其难以部署于资源受限的机器人设备上。现有方案面临以下挑战：
- **On-device inference**：受限于算力与能耗，难以满足实时控制需求（如 20Hz 控制频率）。
- **Edge-only offloading**：受网络波动影响大，延迟不可控，易违反 SLO（Service Level Objective）。
- **静态 co-inference 策略**：无法适应动态变化的网络带宽和边缘负载，导致能效次优甚至超时。

因此，亟需一种**兼顾实时性约束与系统级能量效率**的自适应 device-edge 协同推理框架。

---

### 🚀 提出的新方法与创新思路

作者提出 **EcoVLA**，是首个面向 VLA 模型、以最大化 **Actions/J**（单位能耗下的动作输出数）为目标的自适应 device-edge co-inference 框架。其核心创新包括：

#### （1）统一的 stage-level 抽象设计空间（Unified Stage-Level Abstraction）
- 将异构的 VLA 模型（autoregressive 与 diffusion-based）解耦为统一的有向无环图 $G=(V,U)$，其中每个节点 $v \in V$ 表示一个可独立部署的 inference stage。
- 定义标准化的中间状态接口（structured intermediate packet），封装输入、输出与增量上下文（incremental context），实现跨范式的执行、通信与调度解耦。
- 支持新模型快速接入，无需重写协同逻辑。

#### （2）联合的端到端延迟与能耗建模（Joint Latency-Energy Modeling）
- 构建细粒度预测模型，综合考虑：
  - 每个 stage 在不同设备上的执行时间 $t(v,d)$ 和功耗 $p(v,d)$
  - 跨设备传输开销（基于 packet size 与实时带宽探测）
  - 边缘队列延迟（multi-stage task queues + 请求到达模式分析）
- 支持毫秒级快速评估候选方案的 **end-to-end latency** 与 **energy efficiency**。

#### （3）SLO 约束下的能效优先动态调度（SLO-Constrained Energy-Priority Scheduling）
- 动态选择满足 $T(\pi) \leq T_{\text{SLO}}$ 的能量最优方案 $\pi^* = \arg\max \frac{L_{\text{act}}}{E(\pi)}$
- 若无可行方案，则退化为最小化延迟违规程度。
- 利用预留机制（reservation）显式管理节点占用区间，提升后续调度准确性。

#### （4）轻量级协同通信中间件（Lightweight Co-Inference Communication Middleware）
- 分离控制平面（Ray RPC）与数据平面（torch.distributed + Gloo）
- 同节点内 stage 间采用 inline passing 避免拷贝；跨节点使用 P2P 通信 + pinned buffer 减少固定开销
- 实现计算-通信重叠（computation-communication overlap）

---

### 🔍 相比现有方法的优势

| 维度 | 现有方法（如 Fixed Split, Edge-Only） | EcoVLA |
|------|-------------------------------|--------|
| **抽象层次** | Layer-level / Token-level | Stage-level（语义更清晰，支持异构模型） |
| **调度策略** | 静态划分（Static Partitioning） | 动态自适应（Runtime Adaptation） |
| **优化目标** | 最小化延迟或吞吐优化 | 最大化 **Actions/J**（系统级能效） |
| **环境感知** | 忽略网络/边缘负载波动 | 实时感知并响应变化 |
| **通信效率** | 通用 RPC 序列化开销高 | 结构化 packet + P2P + pinned memory |

---

## 2. 核心实验方法和设置

### 📚 使用的模型（非传统“数据集”，而是代表性 VLA 模型）
实验覆盖五类主流 VLA 模型，涵盖两种范式：
- **Autoregressive VLA**：
  - OpenVLA
- **Diffusion/Denoising-based VLA**：
  - To, To.5
  - SmolVLA
  - RDT-1b

所有模型均使用官方预训练权重与标准推理接口，确保公平比较。

---

### ⚙️ 实验设置
- **硬件平台**：
  - **Robot Device**：NVIDIA Jetson AGX Orin 32GB（低功耗边缘设备）
  - **Edge Server**：配备 NVIDIA RTX 4090 GPU
- **网络模拟**：
  - 使用 `tc` 工具注入带宽扰动（如从 200 Mbps 下降到 50 Mbps）
- **通信架构**：
  - 控制面：Ray RPC
  - 数据面：torch.distributed + Gloo backend
- **冷启动处理**：stage weights 预加载，避免 cold-start 干扰

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Energy Efficiency (Actions/J)** | 单位焦耳能量下可完成的有效动作数量，为核心优化目标 |
| **End-to-End Latency** | 单次推理总耗时 |
| **SLO Satisfaction** | 是否满足 $T \leq 1s$（对应 20Hz 输出频率） |
| **Prediction Accuracy** | 延迟与能耗预测的 MAPE（Mean Absolute Percentage Error） |
| **Adaptability** | 在动态网络/负载下维持 SLO 与高能效的能力 |

---

### 🆚 基线方法对比
- **Device-Only**：全部推理在 AGX Orin 上执行
- **Edge-Only**：完整模型卸载至 RTX 4090
- **Fixed Split**：静态设备-边缘划分策略（代表传统 co-inference 方法）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Fig. 3）

| Model | Baseline (Fixed Split) | EcoVLA | 提升幅度 | SLO 满足？ |
|-------|------------------------|--------|----------|-----------|
| To | 0.08 Actions/J | 0.126 Actions/J | **+58%** | ✅ |
| To.5 | 0.09 Actions/J | 0.166 Actions/J | **+84%** | ✅ |
| SmolVLA | 0.10 Actions/J | 0.20 Actions/J | **+100%** | ✅ |
| RDT-1b | 0.11 Actions/J | 0.31 Actions/J | **+182%** | ✅ |
| OpenVLA | 0.14 Actions/J | 0.47 Actions/J | **+236%** | ✅ |

> 💡 注：Device-Only 虽在部分模型上有较高能效，但在 4/5 模型上 **无法满足 SLO**，实用性受限。

---

### 🔄 动态环境适应性测试（见 Fig. 4）
在网络带宽从 200 Mbps 降至 50 Mbps 时：
- **Fixed Split**：延迟显著上升，接近或超过 SLO
- **Edge-Only**：因全模型传输负担加重，延迟增加约 40%
- **EcoVLA**：检测到带宽下降后自动切换为 **device-heavy plan**，将更多 stage 移回本地，稳定在 ~100ms 内，保持 **2.4 Actions/J**

> ✅ 在轻、中、重负载下，EcoVLA 相比 Fixed Split 平均降低延迟 **23.6%**，节省能量 **24.8%**

---

### 🎯 系统性能预测精度（见 Fig. 5）
- **Latency Prediction MAPE**: **6%**
- **Energy Prediction MAPE**: **3.8%**
- 在 ±10% 误差范围内，准确率 >80%
- 在 ±20% 误差范围内，准确率 >95%

> 高精度预测保障了调度决策的可靠性。

---

### ❌ 消融实验（未明确列出表格，但文中隐含验证）
虽然未提供正式消融表，但通过以下分析体现各模块作用：
- **无动态调度（即退化为 Fixed Split）**：在带宽变化下性能急剧下降
- **无 workload awareness**：无法有效预判边缘排队延迟，导致 SLO 违反
- **无 structured packet 设计**：通信开销更高，且难以建模

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **系统级能效不能仅靠直觉判断**：高功耗设备（如 RTX 4090）可能因极低延迟反而实现更高 **Actions/J**。
2. **静态划分不再适用**：在动态环境中，最优 split 点持续变化，必须在线调整。
3. **stage-level 抽象具有普适性**：成功统一 autoregressive 与 diffusion-based VLA 的协同推理流程。
4. **通信与计算需联合建模**：忽略任一部分都会导致调度失准。
5. **EcoVLA 可持续满足 SLO 并最大化能效**：在多种模型与动态条件下均表现稳健。

---

### ⚠️ 方法的局限性
1. 当前聚焦于 **single edge server + multiple robots** 场景，尚未扩展至 multi-edge 协作。
2. stage 划分依赖人工定义，未来可探索自动化 stage discovery。
3. 未集成 VLA-specific 的压缩技术（如 token pruning 或量化），仍有进一步优化空间。

---

### 🔮 未来工作方向（作者明确提出）
1. 扩展至 **multi-edge collaborative scenarios**
2. 将 **VLA-specific compression techniques**（如 EfficientVLA 中的方法）融入 stage-level co-inference 设计空间
3. 探索更细粒度的 adaptive compilation（如 stage-level torch.compile 优化）

---

## 总结

> **EcoVLA 是首个专为 VLA 模型设计的、以最大化 Actions/J 为目标的自适应 device-edge co-inference 框架。它通过 stage-level 抽象、联合性能建模与动态调度，在保证实时性的前提下，实现了高达 236% 的系统能效提升，显著优于现有方法，并展现出强大的环境适应能力。**

</details>

---

### 12. [An Agentic Framework Using Rules and LLMs for Embedding and Annotating Descriptive Document Layouts: A Plant Science Use Case](https://arxiv.org/abs/2608.14587)

**Authors**: Nicolas Turenne, Youcef Sklab, Eric Chenin, Jean-Daniel Zucker  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.14587v1  

#### Abstract
Background: Recent advances in information retrieval (IR) leverage both dense and sparse representations, large language models (LLMs), and specialized retrieval models to improve ranking accuracy, relevance, and cross-lingual performance. Complementary techniques such as passage indexing, document ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An Agentic Framework Using Rules and LLMs for Embedding and Annotating Descriptive Document Layouts: A Plant Science Use Case

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该研究针对**植物学文献中形态特征（morphological traits）提取困难**的问题。大量植物描述文本以扫描PDF形式存在，且语言多样、格式复杂，传统信息检索（IR）和知识提取方法难以高效、准确地从这些非结构化文本中提取标准化的植物性状数据。

此外，通用的LLM在处理此类任务时存在**幻觉（hallucination）、不一致性和缺乏可解释性**等问题，限制了其在科学领域的可信应用。

---

### 提出了什么新方法或新思路
作者提出了一种**模块化的Agentic框架**，结合规则系统与大型语言模型（LLMs），用于从植物学文本中进行布局分析、实体嵌入与性状标注。其核心创新包括：

- **模块化Agentic架构**：将OCR、文本分割、索引、规则解析和LLM增强等步骤建模为多个协同工作的“agent”，实现流程自动化与可追溯性。
- **混合式知识提取机制**：
  - 使用**基于规则的解析器（rule-based parsers）** 进行精确的trait-value对提取；
  - 引入**Object Class Learning (OCL)** 算法，利用LLM集成（ensemble of LLMs）迭代扩展领域术语词典，提升覆盖范围。
- **可解释的知识构建路径**：通过规则驱动保证透明性，同时借助LLM扩展词汇边界，兼顾准确性与泛化能力。

---

### 相比现有方法的优势
| 维度 | 本文方法 | 现有主流方法 |
|------|----------|-------------|
| **领域适应性** | 高度专业化于植物学文本，支持多语言、历史文献 | 多为通用IR或NLP模型，缺乏领域语义理解 |
| **可解释性** | 规则为基础，LLM仅用于术语扩展，全过程可审计 | 端到端LLM黑箱操作，难以验证结果来源 |
| **精度与覆盖率平衡** | 规则确保高精度，LLM增强显著提升覆盖率 | 单一LLM易产生幻觉；纯规则系统覆盖有限 |
| **结构化输出** | 输出标准化的trait-value三元组，便于构建知识图谱 | 多为自由文本生成或关键词抽取 |

> ✅ **关键优势总结**：在保持高可解释性的前提下，实现了大规模、跨区域植物性状数据的鲁棒提取。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **文本数据集**：
  - 来自三个地区的区域性植物志（floras）：
    - 新喀里多尼亚（New Caledonia）
    - 塞内加尔（Senegal）
    - 喀麦隆（Cameroon）
  - 共包含 **550+ 本专著（monographs）**，约 **36个地理区域**
  - 时间跨度：1845–2000年，均为扫描版PDF
  - 总计约 **127小时OCR处理时间**

- **辅助资源与字典**：
  - 合并三大权威数据库构建物种名称词典：
    - World Checklist of Vascular Plants (**WCVP**)
    - TRY Database
    - World Flora Online (**WFO**)
  - 最终词典含 **160万条目**，归一化后保留 **120万唯一物种名**

---

### 实验设置和评估指标

#### 主要流程阶段
1. **OCR转换**：使用 **Tesseract OCR v5.5.0**（基于LSTM）将PDF转为文本
2. **文本分段与索引**：
   - 按属（genus）切分文档块
   - 利用模式匹配识别每个物种描述段落起止位置
3. **物种名识别与归一化**：开发 **WordGen算法**，支持模糊匹配与拼写纠错
4. **规则驱动的Trait提取**：
   - 定义28个形态性状（如叶形、茎习性等）的正则规则
   - 每个性状对应一个独立的“parser agent”
5. **LLM驱动的术语扩展**：
   - 使用 **GPT-4、LLaMA、Mistral** 构成ensemble
   - 应用 **Object Class Learning (OCL)** 迭代发现新术语

#### 评估指标
- **Number of Trait Extractions**：总提取出的trait数量（按trait和总体统计）
- **Average Number of Correct Annotations per Species**：
  $$
  \text{Avg annotations/species} = \frac{\text{Total correct trait annotations}}{\text{Number of species}}
  $$

---

### 基线方法对比
- **单一单体LLM（No Agent）**：
  - 直接使用LLM进行问答式提取（QA Prompting）
  - 示例指令：“指出是否有花序存在？是否有果实？”
- **传统OCR工具对比**：
  - 对比 Tesseract vs. `pdf2txt` 工具在物种识别上的表现

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 成功从 **4,961个物种** 中提取了 **55,737条trait注释**
- 平均每种植物标注 **9.1个traits**
- 所有标注基于 **29个预定义形态性状类别**

---

### 与基线方法的对比结果

| 方法 | Inflorescence检测准确率 | Fruit存在判断准确率 |
|------|------------------------|--------------------|
| 单一LLM（Zero-shot QA） | 56% (5/9) | 67% (6/9) |

> ⚠️ 结果表明：即使是最先进的LLM，在无上下文引导的情况下仍表现出明显的不稳定性与低准确率，验证了引入结构化agent框架的必要性。

---

### 消融实验结果（Ablation Study）

#### LLM术语扩展的影响
- **75%的traits** 受到了LLM驱动的术语扩展影响
- 总注释数量平均提升了 **59%**
- 示例：原始规则仅能识别“erect stem”，经OCL扩展后新增识别：
  - `'ascending stem'`, `'prostrate stem'`, `'twining stem'` 等变体表达

#### OCR引擎比较（Error Propagation 分析）
- 使用 **Tesseract vs. pdf2txt**：
  - 在属级识别上，Tesseract 提升 **32%**
  - 在种级识别上，提升 **14%**
- 但最终整体annotation数量变化不大 → 表明后续pipeline具有较强的**容错性和鲁棒性**

> 🔍 发现：虽然OCR质量影响初期识别，但整个agent pipeline能够缓解误差传播，保障最终输出稳定。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **规则+LLM的混合范式优于纯LLM或纯规则方法**：
   - 规则提供高精度与可解释性
   - LLM通过OCL机制有效扩展领域词汇，显著提升覆盖率
2. **Agentic架构适合复杂科学文本处理**：
   - 各agent分工明确、依赖清晰，支持模块化调试与优化
   - 支持长期推理、工具调用与迭代学习
3. **系统具备良好的跨区域适用性与语言包容性**：
   - 在法语、英语、葡萄牙语等多种语言的历史文献中均取得稳定效果
4. **LLM术语扩展需人工干预筛选**：
   - 虽然GPT-4一轮即生成114个候选术语，但并非全部可用
   - 当前仍需研究人员审核后纳入规则库，限制完全自动化

---

### 方法的局限性
- **依赖高质量字典与本体支持**：若目标领域缺乏标准命名体系，则WordGen和OCL效果受限
- **自动化程度未达完全闭环**：LLM建议的新术语仍需人工确认才能加入规则，影响扩展速度
- **计算成本较高**：调用多个LLM API带来token消耗与经济成本压力
- **仅限形态性状提取**：当前框架聚焦morphological traits，尚未拓展至生理、生态等功能性状

---

### 未来工作方向
1. **实现全自动OCL闭环**：探索自动验证LLM生成术语的方法（如基于共现频率、上下文一致性）
2. **融合视觉信息**：结合PDF中的图像与表格内容，进一步丰富trait提取维度
3. **构建植物性状知识图谱（Plant Trait KG）**：将提取结果组织为RDF三元组，支持SPARQL查询与推理
4. **部署本地化LLM替代云端API**：使用 **pllama** 或其他植物科学专用LLM降低延迟与成本
5. **扩展至其他生物类群**：应用于真菌、昆虫等其他分类群的描述性文本挖掘

---

> 📌 **总结一句话**：  
> 本文提出了一种**可解释、可扩展、领域定制化的Agentic框架**，成功实现了从海量植物学扫描文献中自动化提取结构化性状数据，为构建全球植物表型知识库提供了可靠的技术路径。

</details>

---

### 13. [Domain-Agnostic Neural Topic Modeling with Contextual Token-Level Semantic Graph Representation](https://arxiv.org/abs/2608.16269)

**Authors**: Seung-Won Seo, Won Ik Cho, Yongmin Yoo  
**Category**: cs.CL  
**Published**: 2026-08-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.16269v1  

#### Abstract
Recent advances in neural topic models with pre-trained language models (PLMs) have achieved strong performance by leveraging general-domain pre-training, yet their topic interpretability often degrades on specialized corpora. This limitation primarily stems from the geometry of the embedding space,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Domain-Agnostic Neural Topic Modeling with Contextual Token-Level Semantic Graph Representation*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Pre-trained Language Models (PLMs)** 的神经主题模型在通用领域表现优异，但在**特定领域**（如生物医学、法律）中面临严重挑战。其根本原因在于：
- **Representation Degeneration**：PLM 在预训练阶段未见过的领域专有词（如“oxidative”、“nitrate”）被映射到嵌入空间中一个狭窄且难以区分的区域。
- 固定的 PLM 嵌入几何结构无法捕捉目标语料中的**上下文语义关系**，导致生成的主题不连贯、解释性差。

现有方法（如 domain-specific PLMs、graph-based models、prefix tuning）均存在局限：
- 领域专用 PLM（如 BioBERT）需要大规模再训练；
- 词级别图模型（word-level graphs）忽略上下文变化；
- 参数高效微调（如 prefix tuning）受限于原始编码器的表示能力。

### 🚀 提出的新方法：DARTOPIC
提出 **DARTOPIC**（**Domain-Agnostic neuRal Topic modeling**），一种无需微调 PLM 的轻量级框架，核心思想是：
> **在冻结的 PLM 嵌入之上，构建可学习的、语料库特定的 token-level 语义图，并通过 GNN 联合优化主题推断任务，从而重构嵌入空间的几何结构。**

#### 创新点：
1. **Token-Level Semantic Graph Construction**  
   - 不同于传统的词共现图或局部窗口图，DARTOPIC 构建的是基于 **token embedding 相似度**的语义图。
   - 每个节点是一个 token（而非单词），保留了文档内的上下文信息。
   - 边由余弦相似度决定：$ A_{ij} = \text{cos}(h_i, h_j) $ if ≥ 阈值 $ T $。

2. **Joint Optimization with Topic Objective**  
   - 使用两层 GCN 学习图上 token 表示，并通过 mean pooling 得到文档向量。
   - 将该向量输入 VAE 进行主题推断，整个系统端到端联合训练。
   - 图结构的学习直接受主题建模目标驱动，使表示更贴合主题发现需求。

3. **完全冻结 PLM，实现真正的 domain-agnostic 设计**  
   - 所有 PLM 参数冻结，仅训练 GNN 和 VAE 组件。
   - 实现对不同规模、不同类型 PLM 的鲁棒性，降低部署成本。

### 🔍 相比现有方法的优势
| 方面 | DARTOPIC | 其他方法 |
|------|----------|---------|
| 是否需微调 | ❌ 否（冻结 PLM） | ✅ 是（如 PVTM 使用 prefix tuning） |
| 图粒度 | ✅ Token-level（含上下文） | ❌ Word-level（静态） |
| 领域适应性 | ✅ 强（从目标语料学结构） | ⚠️ 弱（依赖预训练覆盖） |
| 效率 | ✅ 更快（无额外适配参数） | ❌ 较慢（如 PVTM 有 prefix overhead） |

---

## 2. 核心实验方法和设置

### 📚 数据集
在三个跨领域的基准数据集上进行验证：
| 数据集 | 领域 | # 文档 | 平均长度 | 标签数 |
|--------|------|--------|-----------|--------|
| **20NewsGroup (20NG)** | General | 16,309 | 48.02 | 20 |
| **BioASQ** | Biomedical | 19,448 | 7.44 | 20 |
| **BillSum (Bills)** | Legal | 18,945 | 76.28 | — |

> 注：所有文本均经过清洗和预处理（见 Appendix A）。

### 🧪 实验设置
- **PLM**: 默认使用 `all-MiniLM-L6-v2`（轻量级 sentence transformer）作为冻结编码器。
- **GNN**: 两层 GCN，用于传播 token 级语义。
- **Topic Model**: VAE 结构，输出 document-topic 分布和 topic-word 分布。
- **训练**: 使用 Adam 优化器（lr=1e-3），200 轮，单张 NVIDIA H100 GPU。
- **开源复现**: 所有基线统一使用相同环境和 PLM，确保公平比较。

### 📊 评估指标
#### 主题质量（Topic Quality）
| 指标 | 描述 |
|------|------|
| **NPMI** | Normalized Pointwise Mutual Information，衡量 top words 的语义一致性（越高越好） |
| **TU** | Topic Uniqueness，衡量主题间多样性（避免重复） |
| **TQ** | Topic Quality = NPMI × TU，综合指标 |

#### 文档主题分布质量（下游任务）
| 指标 | 描述 |
|------|------|
| **Purity** | 聚类纯度，衡量每个簇是否集中于单一真实类别 |
| **NMI** | Normalized Mutual Information，信息论角度衡量聚类与标签的一致性 |

---

## 3. 主要实验结果和性能指标

### 📈 主要性能对比（Table 2）

| 方法 | 20NG-NPMI | 20NG-TQ | BioASQ-NPMI | BioASQ-TQ | Bills-NPMI | Bills-TQ |
|------|------------|----------|--------------|-------------|------------|-----------|
| FASTopic | 0.2624 | 0.2313 | 0.0789 | 0.0323 | 0.2291 | 0.2226 |
| PVTM | 0.2707 | 0.2473 | 0.1596 | 0.1553 | 0.2609 | 0.2383 |
| **DARTOPIC (Ours)** | **0.2736** | **0.2561** | **0.1641** | **0.1602** | **0.2685** | **0.2481** |

✅ **结论**：
- DARTOPIC 在所有三个数据集上均取得 **最佳 NPMI 和 TQ 成绩**，尤其在 BioASQ 上大幅领先（+0.005 TQ over PVTM），说明其在领域偏移下仍能保持高主题连贯性。
- 在 TU 上也保持竞争力，表明主题多样性和去冗余能力强。

### 🎯 文档聚类性能（Table 4）
| 方法 | 20NG-Purity | 20NG-NMI | BioASQ-Purity | BioASQ-NMI |
|------|-------------|-----------|----------------|---------------|
| PVTM | 0.4927 | 0.3917 | 0.5174 | 0.3824 |
| **DARTOPIC** | **0.4905** | **0.3968** | **0.5235** | **0.3833** |

✅ **结论**：
- 在 BioASQ 上全面超越所有基线，显示其在领域术语密集场景下的强大判别力。
- 即便在通用领域（20NG）也达到 SOTA 水平。

### 🔬 消融与分析实验

#### (1) 对 PLM 的鲁棒性分析（Table 3 & 5）
| 方法 / PLM | MiniLM → BioBERT（BioASQ-TQ） |
|-----------|-------------------------------|
| FASTopic | 0.0323 → 0.1137 (+252%) |
| ZeroShotTM | 0.1251 → 0.1576 (+26%) |
| **DARTOPIC** | 0.1602 → 0.1638 (**+2.2%**) |

✅ **发现**：
- 多数方法性能高度依赖 PLM 类型；
- **DARTOPIC 性能几乎不受 PLM 变化影响**，即使使用轻量级 MiniLM 也能媲美 BioBERT 效果，体现其真正的 *domain-agnostic* 特性。

#### (2) 运行效率对比（Table 6）
| 方法 | 20NG-Train Time | Bills-Train Time |
|------|------------------|------------------|
| PVTM | 13.55s | 9.82s |
| **DARTOPIC** | **7.67s** | **8.67s** |

✅ **发现**：
- DARTOPIC 训练速度显著更快，尤其在长文档数据集（20NG）上优势明显。
- 因为没有 prefix tuning 的额外参数负担，架构更简洁高效。

#### (3) 图构建方式对比（Appendix D）
| 图类型 | 20NG-TQ | BioASQ-TQ |
|--------|---------|-----------|
| 1-hop | 0.2460 | 0.1520 |
| 2-hop | 0.2507 | 0.1620 |
| **Semantic (ours)** | **0.2561** | **0.1602** |

✅ **发现**：
- 语义图在长文档中表现更好（20NG），而局部 n-hop 图在短文本略优；
- 但总体而言，**语义相似性图更适合主题建模任务**。

#### (4) 超参数敏感性分析（Appendix E）
- 最佳阈值 $ T \in [0.2, 0.3] $，过高过低都会损害性能；
- 但整体趋势稳定，说明方法对超参不敏感。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **解耦主题质量与 PLM 容量**  
   DARTOPIC 成功证明：**即使使用冻结的小型 PLM，也能通过 token-level 语义图学习重建领域特定的语义结构**，打破传统对 domain-specific pre-training 或 fine-tuning 的依赖。

2. **Token-Level Graph 的优越性**  
   相较于 word-level 图，token-level 图能捕获同一词在不同上下文中的语义差异，更适合复杂主题建模。

3. **联合优化的有效性**  
   图表示学习直接由主题目标引导，使得学到的结构更具语义判别力，而非仅仅语法相关。

4. **高效且鲁棒的框架设计**  
   - 不依赖大模型或昂贵微调；
   - 在多种 PLM 下表现稳定；
   - 运行速度快，适合实际应用。

### ⚠️ 局限性
1. **语言限制**：目前实验仅在英文语料上验证，多语言适用性待检验。
2. **自动评估为主**：缺乏人工评估（human evaluation）来深入分析主题可解释性。
3. **图稀疏性控制**：依赖阈值 $ T $ 控制边密度，虽经调优但仍属启发式设定。

### 🔮 未来工作方向
- 扩展至 **multilingual topic modeling** 场景；
- 探索动态图构建机制（如 attention-based edge learning）；
- 引入 **human-in-the-loop** 评估以提升领域专家可用性；
- 应用于低资源或零样本领域迁移任务。

---

> 💡 **一句话总结**：  
> **DARTOPIC 通过在冻结 PLM 上叠加一个可学习的 token-level 语义图层，实现了无需微调的领域无关主题建模，在多个领域均达到 SOTA 性能，兼具高效率与强鲁棒性。**

</details>

---

### 14. [p-Spin Glass Network Efficient Single-Batch Continual Learning](https://arxiv.org/abs/2608.14774)

**Authors**: Vladimer Khasia  
**Category**: cs.LG  
**Published**: 2026-08-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.14774v1  

#### Abstract
Modern sequence models heavily rely on massive memory footprints and large-batch stochastic optimization, barriers that restrict sample efficiency and continual learning. We introduce the $p$-Spin Glass Network, a novel architecture that overcomes these limitations, structurally manages optimization...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：p-Spin Glass Network Efficient Single-Batch Continual Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代深度学习模型（尤其是基于 **Transformer** 架构的序列模型）严重依赖大规模内存占用和大批次（large-batch）随机优化，这带来了以下限制：
- **样本效率低下**：需要大量训练序列才能收敛。
- **连续学习困难**：在边缘设备上难以实现持续学习（continual learning），因为小批量甚至单样本训练时梯度方差高、优化不稳定。
- **内存瓶颈**：激活值存储随序列长度平方增长（$O(B \cdot T^2)$），参数量也巨大。

这些问题阻碍了低资源场景、在线学习和边缘 AI 的发展。

---

### 提出了什么新方法或新思路
作者提出了一种全新的架构——**p-Spin Glass Network (SGN)**，其核心思想融合了：
- **隐式深度学习（Implicit Deep Learning）**：通过固定点求解器（fixed point solver）定义隐藏状态，而非显式展开前向传播图。
- **物理启发的热力学吸引子（Thermodynamic Attractors）**：借鉴自 **Spin Glass 物理系统** 的能量最小化机制，构建高阶交互结构。
- **原生三元量化（Native Ternary Quantization）**：对内部投影矩阵 $W_{\text{ext}}, W_{\text{int}}$ 进行 $\{-1, 0, 1\}$ 量化，压缩参数并正则化梯度。

该网络将序列建模重构为一个动态寻找能量极小态的过程，并利用 **Implicit Function Theorem (IFT)** 推导精确梯度，从而解耦激活内存与迭代深度。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **Memory Efficiency** | 参数压缩达 **8×**（因三元量化）；激活内存严格控制在 $O(B \cdot T \cdot D)$，不受固定点迭代次数 $K$ 影响。 |
| **Sample Efficiency** | 在仅使用 **12.5% 训练数据**（8,000 vs 64,000 序列）下达到与标准 Transformer 相当甚至更优的性能。 |
| **Single-Batch Stability** | 支持 **micro-batch size = 1** 下平滑单调收敛，无需梯度累积，适合持续学习。 |
| **Modality Agnosticism** | 可无缝处理离散 subword tokens（|V|=49,152）和原始 raw byte 流（|V|=256），具备通用序列建模能力。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **HuggingFace FW/fineweb-edu dataset**（10BT 子集）
- 数据流式加载，shuffle buffer 大小为 500，随机种子固定为 1337。

---

### 实验设置
| 配置项 | 设置说明 |
|--------|----------|
| **Baseline Model** | 标准 autoregressive Transformer，batch size = 64，训练 1,000 步 → 共见 64,000 个序列。 |
| **Proposed Models** | p-Spin Glass Network 两种配置：<br>• Subword-level: |V|=49,152, T=1024<br>• Byte-level: |V|=256, T=2048<br>均在 **B=1** 下训练 8,000 步 → 共见 8,000 个序列（仅为 baseline 的 12.5%）。 |
| **模型规模** | 所有模型约 **60M 参数**，确保公平比较。<br>p-Spin 使用 L=4 层，H=8 attention heads，K=5 micro-steps，chunk size C=128。 |
| **优化器** | AdamW，weight decay=0.05，gradient clipping=1.0<br>学习率调度：前 500 步线性 warmup 至 $8\times10^{-4}$，后接余弦退火。 |

---

### 评估指标
| 模态 | 主要指标 |
|------|---------|
| **Subword** | Cross-Entropy Loss、Perplexity（PPL） |
| **Byte** | Bits-Per-Byte (BPB)，等效换算为 subword loss用于横向对比 |

---

### 基线方法对比
- **Transformer Baseline**：高度优化的标准 Transformer，在相同任务和数据上作为性能上限参考。
- 对比维度包括：最终 loss、训练稳定性、每步延迟、内存占用、样本效率。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| ARCHITECTURE | MODALITY | BATCH SIZE | SEQ. SEEN | TIME/STEP | FINAL LOSS (L) |
|--------------|----------|------------|-----------|-----------|----------------|
| TRANSFORMER BASE | Subword | 64 | 64,000 | 18.90s | 4.8025 |
| p-SPIN (Ours) | Subword | 1 | 8,000 | **0.44s** | **4.8004** |
| p-SPIN (Ours) | Byte | 1 | 8,000 | 1.53s | 5.5839 (equiv. subword ~5.5) |

> 注：Byte 模型的等效 subword loss 按公式 $L_{\text{sub}} \approx \text{BPB} \cdot \ln(2) \cdot \mu$，其中 $\mu=3.8$ 是平均 bytes-per-token 压缩比。

---

### 与基线方法的对比结果
1. **样本效率碾压式领先**
   - p-Spin Subword 模型仅用 **1/8 的训练序列**（8k vs 64k），达到了略优于 baseline 的验证损失（4.8004 vs 4.8025），实现 **asymptotic parity with higher data efficiency**。
   - 如 Figure 2 所示，p-Spin 很快就逼近 baseline 的渐近性能。

2. **极致推理效率**
   - 尽管是 B=1 单样本训练，p-Spin Subword 每步耗时仅 **0.44 秒**，远低于 baseline 的 18.9 秒（B=64），表明其计算结构高度优化。

3. **单批次稳定训练**
   - 如 Figure 3 所示，p-Spin 在 B=1 下展现出近乎完美的平滑、单调下降曲线，而传统 SGD 在如此小批量下通常剧烈震荡。
   - 证明其具有强大的 **variance-damping effect**，归功于固定点吸引子的热力学正则化作用。

4. **跨模态泛化能力强**
   - Byte-level p-Spin 达到 **2.12 BPB**，虽未超越 subword baseline，但趋势持续下降，验证其能在超长原始字节流中有效进行 temporal credit assignment。
   - 内存开销显著降低：Byte 模型峰值 VRAM 投影减少约 **192×**（从 49k 到 256 类别）。

---

### 消融实验结果（文中隐含分析）
虽然没有明确列出消融表，但从设计原理可推断关键组件的作用：

| 组件 | 功能 | 贡献 |
|------|------|------|
| **Ternary Quantization** | 将 $W_{\text{ext}}, W_{\text{int}}$ 量化至 $\{-1,0,1\}$ | 实现 **8× 参数压缩**，同时约束 Lipschitz 常数，抑制梯度爆炸 |
| **Implicit Function Theorem (IFT)** | 反向传播时不展开 K 步迭代图，而是用 Neumann 级数近似逆雅可比 | 激活内存恒定 $O(B \cdot T \cdot D_{\text{int}})$，不随 K 增加 |
| **SRAM-Fused Chunking** | 在 SRAM 中完成因果掩码计算并立即规约 | 避免 $O(T^2)$ 激活材料化，保持 HBM 内存线性增长 |
| **Higher-Order p-Spin Interactions** | 引入非线性吸引子动力学 | 增强表示能力，提升每步更新的信息密度 |

---

## 4. 关键结论和发现

### 论文的主要发现
1. **大批次不再是深度学习稳定的必要条件**  
   p-Spin Glass Network 成功实现了 **single-batch continual learning**，打破了“必须靠大 batch 或梯度累积来稳定训练”的范式。

2. **隐式物理吸引子 + 三元量化 = 高效且鲁棒的学习机制**  
   固定点搜索过程天然具备噪声过滤能力，结合量化带来的结构正则化，使模型在极低数据量和极小批量下仍能稳定收敛。

3. **真正的模态无关序列建模成为可能**  
   同一套架构可统一处理 tokenized 和 byte-level 输入，为未来 **tokenizer-free modeling** 提供可行路径。

4. **边缘 AI 的理想候选架构**  
   极低内存占用（如 Byte 模型总 footprint 仅 **41.37 MB**）、高样本效率、支持在线更新，非常适合部署于资源受限设备。

---

### 方法的局限性
1. **输出层仍是瓶颈（尤其 subword 模态）**  
   尽管内部高效，但 tied embedding 到大词汇表（|V|=49k）的投影仍造成显著计算和内存负担。

2. **byte-level 性能尚未超越 subword baseline**  
   当前配置下，raw byte 模型性能仍落后于成熟的 subword tokenizer 方案，需进一步调优或扩展上下文。

3. **硬件依赖性强**  
   SRAM-fused chunking 和 ternary arithmetic 的优势依赖特定 GPU 架构（如支持 Triton 编译优化）才能充分发挥。

---

### 未来工作方向
1. **扩展到更多模态**：图像 patch、音频帧、DNA 序列等，验证其作为通用序列引擎的能力。
2. **完全无 tokenizer 的端到端训练**：结合 byte-level modeling 与自监督目标，探索真正“zero-preprocessing”语言模型。
3. **嵌入式部署实测**：在 Raspberry Pi、Jetson 或手机芯片上实测能耗与延迟，推动边缘智能落地。
4. **理论深化**：建立 p-Spin 吸引子与 loss landscape 平坦性之间的形式化联系，提供更强的泛化保证。

---

> ✅ **一句话总结**：  
> 本文提出的 **p-Spin Glass Network** 通过融合物理启发的隐式吸引子与三元量化，首次实现了 **高效、稳定、单样本级别的持续学习**，在样本效率、内存压缩和跨模态适应性方面全面超越传统 Transformer，为下一代边缘 AI 和持续学习系统奠定了基础。  
> 🔗 代码开源地址：[https://github.com/VladimerKhasia/sgn](https://github.com/VladimerKhasia/sgn)

</details>

---

### 15. [PERO: Efficient Robust Post-Training Foundation Models for Encrypted Traffic Classification](https://arxiv.org/abs/2608.15504)

**Authors**: Wumei Du, Jiarong Wen, Kaiyu Zhang, Zi Yang, Yiqin Lv, Longfei Zhang, Dong Liang, Zheng Xie  
**Category**: cs.LG  
**Published**: 2026-08-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.15504v1  

#### Abstract
Encrypted traffic classification is vital for network security, yet real-world deployments are inherently sensitive to rare but high-loss errors such as misclassification of malicious traffic. The encrypted traffic foundation model, as a promising general-purpose technique, can achieve impressive ov...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PERO: Efficient Robust Post-Training Foundation Models for Encrypted Traffic Classification

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
加密流量分类（Encrypted Traffic Classification）在网络安全中至关重要，尤其是在识别恶意流量方面。然而，现有的 **foundation models** 虽然在平均性能上表现优异，但在高风险尾部样本（high-risk tail events）上的鲁棒性较差。标准的训练目标如 **Empirical Risk Minimization (ERM)** 忽视了这些罕见但损失极高的错误，导致模型在实际部署中可能漏检严重威胁。

此外，直接应用鲁棒优化方法（如 **CVaR**）进行 post-training 在大规模 foundation models 上计算开销巨大，因为需要对大量样本进行重复的损失评估和参数更新，难以满足实时性和资源受限场景的需求。

### 提出了什么新方法或新思路
本文提出了一种高效的鲁棒后训练框架：**Pre-Evaluation Robust Optimization (PERO)**。

其核心思想是：
- 引入一个**轻量级的预评估模块（pre-evaluation module）**，作为代理模型来估计每个样本的风险（即预测其在主模型上的交叉熵损失）。
- 利用该模块从候选池中筛选出潜在的高风险样本子集，仅对这些样本执行昂贵的主模型前向/反向传播。
- 实现“**风险估计与模型优化解耦**”（decoupling risk estimation from optimization），从而大幅降低计算成本。

### 相比现有方法的优势
| 维度 | PERO 的优势 |
|------|-------------|
| **效率** | 显著低于 MC-CVaR 等传统鲁棒方法的计算和内存开销，接近 ERM 的效率水平 |
| **鲁棒性** | 在尾部样本（tail-risk）上显著优于 ERM、Focal Loss、GroupDRO、TDRO 等方法 |
| **通用性** | 可适配不同 backbone（如 ET-BERT、YaTC），具有良好的扩展性 |
| **实用性** | 特别适用于持续学习（continual post-training）等资源敏感场景 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在四个典型的加密流量分类基准数据集上进行：
- **USTC-TFC**：包含 10 类恶意软件和 10 类良性应用，共 97,115 条流量记录
- **ISCX-VPN-Service**：基于 VPN 场景的服务级别分类，12 类，60,000 条样本
- **ISCX-VPN-App**：基于 VPN 的应用级别分类，17 类，77,163 条样本
- **CICIoT2022**：大规模物联网攻击流量数据集，用于验证泛化能力

### 实验设置和评估指标

#### 主干网络（Backbone）
- 主要使用 **ET-BERT** 作为基础模型
- 补充实验采用 **YaTC** 验证方法的通用性

#### 评估指标
- **整体性能**：Accuracy (AC)，Precision (PR)，Recall (RC)，F1-score（均为 macro-averaged）
- **尾部鲁棒性**：在最差的 `(1−α)` 比例样本上的性能，记为 `AC_α`, `F1_α` 等，其中 α ∈ {0.5, 0.7, 0.9}
  - 例如 `F1_0.9` 表示最差 10% 样本的平均 F1 分数
- **效率指标**：每轮迭代运行时间（Runtime/Iter.）和 GPU 内存占用

#### 训练设置
- 批大小（Batch Size）：默认 B=32，候选池大小 B'=64
- 优化器：主模型用 AdamW，预评估模块用 Adam
- 学习率：主模型 1e-6，预评估模块 5e-6
- 总迭代次数：非选择类方法训练 10 轮，选择类方法训练 20 轮以保持更新总量可比
- 结果取五次随机种子平均值

### 基线方法对比
对比了以下代表性方法：
| 方法 | 类型 | 说明 |
|------|------|------|
| **ERM** | 基准 | 标准经验风险最小化 |
| **Random Selection** | 控制组 | 随机选取子集更新 |
| **Focal Loss** | 损失重加权 | 自动关注难例 |
| **MC-CVaR** | 鲁棒优化 | Monte Carlo 近似 CVaR，直接选高损失样本 |
| **GroupDRO** | 分布鲁棒 | 最大化最差 group 的损失 |
| **OHTM** | 在线难样本挖掘 | 基于历史缓冲区选择困难任务 |
| **TDRO** | 平滑鲁棒优化 | 使用 LogSumExp 替代 max 操作 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 ISCX-VPN-Service 和 USTC-TFC 为例）

#### 表：ISCX-VPN-Service 上的结果（部分）
| Method | F1 (%) | F1_0.9 (%) |
|--------|--------|------------|
| ERM | 97.69 | 59.61 |
| MC-CVaR | 98.22 | 70.74 |
| **PERO (Ours)** | **98.60** | **78.03** |

> PERO 在整体 F1 上领先，在尾部 F1_0.9 上**超越第二名 MC-CVaR 达 7.29%**

#### 表：USTC-TFC 上的结果（尾部性能尤为突出）
| Method | F1 (%) | F1_0.9 (%) |
|--------|--------|------------|
| ERM | 97.04 | 31.29 |
| MC-CVaR | 98.12 | 75.73 |
| OHTM | 97.83 | 72.87 |
| **PERO (Ours)** | **98.26** | **87.63** |

> 在极端尾部风险下，PERO 将 F1_0.9 提升至 **87.63%**，远超其他所有方法（比 MC-CVaR 高 11.9%）

#### 效率对比（ISCX-VPN-Service）
| Method | Runtime/Iter. (s) | Memory Usage (GB) |
|--------|---------------------|--------------------|
| ERM | 0.864 | 11.59 |
| MC-CVaR | 1.568 | 26.18 |
| **PERO (Ours)** | **0.826** | **11.75** |

> PERO 的运行速度**快于所有鲁棒方法**，且内存消耗仅为 MC-CVaR 的 **45%**，几乎与 ERM 持平

---

### 与基线方法的对比结果
- **鲁棒性全面领先**：在三个主要数据集上，PERO 在 `F1_0.9` 指标上均取得最佳成绩，尤其在 USTC-TFC 上优势巨大
- **整体性能不妥协**：不仅提升尾部性能，同时保持甚至略微提升整体准确率
- **效率碾压传统鲁棒方法**：相比 MC-CVaR，PERO 减少了约 **50% 的训练时间和 55% 的显存占用**
- **优于其他采样策略**：显著优于 Random、OHTM、TDRO 等启发式或复杂采样方法

---

### 消融实验结果（Ablation Study）

在 ISCX-VPN-App 上进行了关键超参数分析：

#### （1）批大小 $ B $
- $ B=32 $ 时达到最优平衡
- $ B=16 $ 时尾部性能下降；$ B=64 $ 收益递减
- 结论：适度批量即可获得良好效果

#### （2）候选池大小 $ B' $ 与选择比例 $ r = B/B' $
- 更大的候选池（更小的选择比例）有助于探索更多潜在高风险样本
- $ r=1/4 $（即 $ B'=128 $）时尾部性能最佳
- 说明充分的候选空间对鲁棒性有益

#### （3）预评估模块中 MLP 的深度
- **3 层 MLP** 表现最好
- 2 层容量不足；4 层无进一步增益
- 结论：**中等复杂度的代理模型已足够有效**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **尾部风险可通过高效机制缓解**：无需在整个大模型上反复评估损失，通过轻量级代理即可实现高质量的风险排序。
2. ✅ **PERO 实现了鲁棒性与效率的双赢**：在不牺牲整体性能的前提下，显著提升了对高风险样本的分类能力，且计算开销极低。
3. ✅ **代理模块具备良好的风险预测保真度**：Pearson/Spearman 相关系数稳定在 0.5~0.6，Precision@k 达到 0.65 以上，证明其能有效识别真正高损失样本。
4. ✅ **方法具有 backbone 通用性**：在 ET-BERT 和 YaTC 上均表现出色，验证了其广泛适用性。

### 方法的局限性
1. **理论假设较强**：收敛性分析依赖于损失函数 Hessian 谱半径有界，这对大型 Transformer 模型可能不完全成立。
2. **代理模型为近似估计**：无法保证在分布偏移剧烈时仍能准确预测风险。
3. **引入额外超参数**：如候选池大小、代理模型结构等，需调优。
4. **未解决数据偏差问题**：虽然改善尾部性能，但不能消除训练数据本身存在的偏见。

### 未来工作方向
1. **自适应代理设计**：开发能自动调整结构或学习率的动态 pre-evaluation 模块
2. **扩展至 RL post-training**：将“代理预测轨迹价值 + 选择性更新”范式应用于 LLM 或强化学习微调
3. **多模态与跨域迁移**：探索在不同网络环境间的鲁棒迁移能力
4. **在线持续学习集成**：将 PERO 应用于流式数据场景，支持增量式鲁棒更新

--- 

> **总结一句话**：  
> **PERO 通过“轻量代理先行筛选 + 大模型聚焦更新”的机制，在几乎不增加计算负担的情况下，实现了加密流量 foundation models 的高效鲁棒 post-training，解决了传统 CVaR 方法不可扩展的根本瓶颈。**

</details>

---

### 16. [An Analytical-Prior Framework for Data-Efficient Prediction of Sound-Reduction Frequencies in Rectangular Side-Branch Helmholtz Resonators](https://arxiv.org/abs/2608.16873)

**Authors**: Jiaming Li  
**Category**: cs.LG  
**Published**: 2026-08-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.16873v1  

#### Abstract
High-fidelity finite-element simulations can provide accurate numerical predictions for side-branch resonators, but large simulation datasets are expensive to generate and purely data-driven surrogates may become unreliable when simulation-labelled data are scarce. This study develops an analytical-...

---

### 17. [SKILL: Self-correcting Knowledge-guided Iterative Large Language Model Agent for Logic Optimization](https://arxiv.org/abs/2608.14579)

**Authors**: Rui Yang  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.14579v1  

#### Abstract
Logic synthesis optimization poses significant challenges due to exponentially growing search spaces, sparse reward signals, and diverse logic structures. Traditional expert-designed flows lack adaptability, while reinforcement learning (RL) methods often suffer from low sample efficiency and limite...

---

### 18. [When Entropy Is Not Enough: Reclaiming Lost Semantics in LLM Output Length Prediction](https://arxiv.org/abs/2608.15592)

**Authors**: Feiyang Ren, Shengtao Wen, Lingbing Guo, Yu Tian, Yuanning Cui, Xiang Chen  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.15592v1  

#### Abstract
Efficient LLM serving is often bottlenecked by the need to pad sequences to a fixed maximum length, and this wastes compute and degrades throughput. Predicting output lengths in advance makes it possible to adopt length-aware scheduling, and this reduces the overhead. This advantage is especially pr...

---

### 19. [LOCAL: Enabling Learning On-device Contiguously for Agent LLMs](https://arxiv.org/abs/2608.15241)

**Authors**: Xinxin Liu, Jiaxin Li, Zibo Wang, Yun Ji, Zhangqi Zhu, Qing Hu, Zhibin Wang, Rong Gu, Sheng Zhong, Chen Tian  
**Category**: cs.DC  
**Published**: 2026-08-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.15241v1  

#### Abstract
On-device LLM agents interact repeatedly with users on local hardware, producing private traces that are valuable for adaptation but should not be sent to a remote trainer. Ideally, such agents would learn contiguously---adapting from every interaction without pausing or suspending user-facing infer...

---

### 20. [Funnel of Thoughts: Efficient Test-Time Scaling via Early Voting and Rollout Pruning](https://arxiv.org/abs/2608.15065)

**Authors**: Chanhee Park, Sungbin Han, Jeongho Yoon, Seongtae Hong, Heuiseok Lim  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.15065v1  

#### Abstract
Large Reasoning Models produce diverse, sometimes inconsistent answers across repeated queries on the same problem, so multi-sample inference is a prerequisite for reliable deployment. Majority voting at k rollouts is the standard solution and the de facto accuracy target for this regime, but it is ...

---

### 21. [DriveCache: Action-Aware Caching for Driving World Model Inference](https://arxiv.org/abs/2608.16354)

**Authors**: Jianchun Yang, Jian Liang, Xianda Guo, Pinhan Fu, Yanlun Peng, Conglang Zhang, Wenke Huang, Mang Ye  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.16354v1  

#### Abstract
Driving video generation models support autonomous-driving development by predicting controllable future scenes for simulation, planning evaluation, and offline data generation. Diffusion-based driving generators repeatedly evaluate large backbones across denoising steps, which limits generation thr...

---

### 22. [ParaTempo: Efficient Parallel Reasoning via Temporal Confidence](https://arxiv.org/abs/2608.16425)

**Authors**: Xuteng Zhang, Wenhao Zeng, Xiaodong Gu, Chao Hu, Haotian Lin, Yuling Shi, Min Wang, Beijun Shen  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.16425v1  

#### Abstract
Parallel reasoning improves the accuracy and robustness of large reasoning models by exploring multiple solution paths, but its computational cost grows with reasoning depth and branch count. Existing methods for managing these parallel paths typically rely on final-answer consensus, local token con...

---

### 23. [Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents](https://arxiv.org/abs/2608.15008)

**Authors**: Wei-Chieh Huang, Weizhi Zhang, Yuchen Wu, Yankai Chen, Eric Hanchen Jiang, Wooseong Yang, Yiwei Yang, Henry Peng Zou, Hanrong Zhang, Ying Nian Wu, Haolun Wu, Kai-Wei Chang, Philip S. Yu, Xue Liu, Aylin Caliskan  
**Category**: cs.CL  
**Published**: 2026-08-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.15008v1  

#### Abstract
Memory is becoming core infrastructure for long-horizon LLM agents, yet existing evaluations offer limited guidance on which memory substrate, namely the underlying medium in which memory is represented and stored, should be used under different operating regimes. We present a controlled harness eva...

---

### 24. [Architecture-Dependent Causal Transfer of Activation States Across Large Language Models](https://arxiv.org/abs/2608.16347)

**Authors**: Fernando Cardenas Piepereit  
**Category**: cs.CL  
**Published**: 2026-08-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.16347v1  

#### Abstract
Direct communication between AI systems relies on natural language as an intermediate layer, incurring encoding/decoding overhead, token cost, and latency. We ask whether internal activation states can instead be transferred causally between different large language model (LLM) architectures via a l...

---

### 25. [SubZero+: Efficient Zeroth-Order LLM Fine-Tuning via Large Learning Rates](https://arxiv.org/abs/2608.15665)

**Authors**: Ziming Yu, Shuyao Xiao, Xingyu Zhao, Sike Wang, Pan Zhou, Peiyu Zang, Xiangda Yan, Yongjie Yang, Jia Li  
**Category**: cs.LG  
**Published**: 2026-08-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.15665v1  

#### Abstract
Zeroth-order (ZO) optimization enables backpropagation-free fine-tuning of large language models, but existing ZO methods suffer from high-variance gradient estimators, making convergence unstable and highly sensitive to learning rates. We propose SubZero+, an improved SubZero framework that improve...

---

### 26. [CrevasseSeg: A Label-Efficient UAV Crevasse Segmentation Framework](https://arxiv.org/abs/2608.15790)

**Authors**: Steven Wallace, William D Harcourt, Richard Hann, Aiden Durrant, Somayajulu Sripada, Georgios Leontidis  
**Category**: cs.LG  
**Published**: 2026-08-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.15790v1  

#### Abstract
Crevasse mapping from uncrewed aerial vehicle (UAV) imagery matters for glaciological research and for field safety in glaciated terrain. Yet, pixel-level annotation of glacier surfaces is costly and requires domain experts. We introduce CrevasseSeg, a framework for binary segmentation over the term...

---

### 27. [Euclid-Omni : A Unified Neuro-Symbolic Framework for Plane Geometry](https://arxiv.org/abs/2608.14585)

**Authors**: Zhaoyu Li, Hangrui Bi, Youyuan Zhang, Wenjie Ma, Zenan Li, Zhaolei Zhang, Xujie Si, Kaiyu Yang  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.14585v1  

#### Abstract
Euclidean geometry is a compelling testbed for AI reasoning, as it demands the combination of intuitive diagram understanding, axiomatic deduction, and algebraic computation. Yet, existing approaches typically address only a subset of these abilities or struggle with competition-level problems. We i...

---

### 28. [CEDAR-GRPO: Process-Aware Reinforcement Learning for General Abductive Reasoning in LLMs](https://arxiv.org/abs/2608.14791)

**Authors**: Moein Salimi, Danial Parnian, Shaygan Adim, Amirmohammad Ebrahiminasab, Nima Alighardashi, Parsa Gholami, Sahand Akramipour, Mahdi Jafari Siavoshani, Mohammad Hossein Rohban  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.14791v1  

#### Abstract
Abductive reasoning, often characterized as inference to the best explanation, is central to explanation under uncertainty, from everyday sense-making and investigation to scientific discovery. Yet LLM research has mostly studied abduction through narrow, task-specific benchmarks, making it unclear ...

---

### 29. [LLM-Based Hierarchical Coordinated Control with Continuation-Aware Policy Learning](https://arxiv.org/abs/2608.15041)

**Authors**: Changhong He, Jinda Gao, Xinkuan Liu, Le Zhang, Xizi Luo, Yu Mei  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.15041v1  

#### Abstract
Coordinating multiple interacting units in complex engineering systems is challenging when system interactions are difficult to model, operational information is heterogeneous, and low-level actions must satisfy strict constraints. We propose an LLM-based hierarchical framework in which the LLM coor...

---

### 30. [Decentralized Federated Learning for Heterogeneous Multi-Task Semantic Communication](https://arxiv.org/abs/2608.15256)

**Authors**: Lin Yin, Tiejun Lv, Weicai Li, Xi Yu, Xiaoyu He  
**Category**: cs.AI  
**Published**: 2026-08-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.15256v1  

#### Abstract
Collaborative training in distributed semantic communication (DSC) networks typically relies on decentralized federated learning (DFL). However, pushing topology-agnostic aggregation into heterogeneous, multi-task environments creates a fundamental bottleneck: it drives negative transfer and overcon...

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

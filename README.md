# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-19 11:16:12 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment](https://arxiv.org/abs/2610.23006)

**Authors**: Wenhui Lv, Yunming Liao, Hongli Xu, Yang Xu, Junhao Cheng  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2610.23006v1  

#### Abstract
Mixture-of-Experts (MoE) models reduce inference computation through sparse expert activation, but their large expert memory footprint often exceeds GPU capacity, making expert offloading necessary for memory-constrained inference. However, existing systems typically handle GPU-missed experts by eit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Mixture-of-Experts (MoE) 模型虽然通过稀疏激活降低了推理计算量，但其专家参数的**巨大内存占用**常常超出 GPU 显存容量，导致必须将部分专家卸载到 CPU 内存中（expert offloading）。然而，现有方法在处理 GPU 缺失专家（GPU-missed experts）时存在以下瓶颈：

- **纯 GPU 执行**：将缺失专家从 CPU 传输至 GPU 再执行，受限于 PCIe 传输延迟，成为性能瓶颈。
- **传统 CPU-GPU 协同执行**：直接在 CPU 上运行缺失专家，但由于 CPU 计算能力远弱于 GPU（如 bfloat16 吞吐低 10× 以上），导致 CPU 成为关键路径。

因此，如何高效地决定“**哪些缺失专家应在 CPU 上执行，哪些应传回 GPU 执行**”是一个关键挑战。

---

### 提出的新方法：Apex
作者提出 **Apex** —— 一种基于**自适应专家分配**（Adaptive Expert Assignment）的 CPU-GPU 协同 MoE 推理系统。

#### 核心思想
- 不再对所有 GPU-missed experts 采用统一策略。
- 利用不同专家的 **token-load**（即被路由到该专家的 token 数量）差异进行差异化调度：
  - **高 token-load 专家** → 转移到 GPU 执行（尽管有传输开销，但节省大量计算时间）
  - **低 token-load 专家** → 在 CPU 上直接执行（避免不必要的权重传输开销）

#### 创新机制
1. **Exact Assignment（精确分配）**  
   基于当前层的实际路由结果、专家负载和实时观测的计算/传输速率，动态选择最优转移数量 $ p^* $，以最小化 CPU 和 GPU 路径的最大延迟：
   $$
   p^* = \arg\min_p \max\left(T_{\text{CPU}}(C(p)),\ T_{\text{transfer}}(G(p)) + T_{\text{GPU}}(R \cup G(p))\right)
   $$

2. **Speculative Assignment（推测性分配）**  
   利用当前层 attention 输出预测下一层可能激活的高负载专家，并提前开始传输，延长重叠窗口，提升 GPU 利用率。

3. **Rate-driven Online Balancing**  
   动态跟踪 CPU、GPU 和 transfer 的吞吐率（$ \mu_C, \mu_G, v_T $），用于更准确地估计路径延迟。

---

### 相比现有方法的优势
| 方法 | 局限性 | Apex 的改进 |
|------|--------|-------------|
| **Mixtral-Offloading** | 所有缺失专家都需传输，受 PCIe 带宽限制 | 只选择高负载专家传输，减少无效传输 |
| **Fiddler / Post-Fetch** | 所有缺失专家都在 CPU 执行，易造成 CPU 瓶颈 | 自适应分配，平衡双路径延迟 |

> ✅ **Apex 的核心优势在于实现了 CPU 计算路径 与 GPU 传输+计算路径之间的动态平衡，显著降低端到端推理延迟。**

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MMLU** 数据集的 validation split 用于推理测试
- training split 用于离线分析专家激活频率（如 Fiddler 的 profiling）

### 实验设置
- **硬件平台**：
  - GPU: NVIDIA RTX A6000 (48GB)
  - CPU: Intel Xeon Platinum 8358P
  - Host Memory: 512GB
  - 互联: PCIe 4.0 ×16 (理论带宽 32 GB/s)

- **实现框架**：
  - 基于 **KTransformers** 构建，支持 CPU/GPU 混合执行与专家卸载
  - 扩展支持 next-layer 预测、在线专家分配与 GPU 专家池管理

### 评估模型（见 Table 1）
| Model | Total Params | Activated Params | # Experts | K (per token) |
|-------|---------------|------------------|-----------|----------------|
| Qwen3.5-35B-A3B | 35B | 3B | 256 | 8 |
| Qwen3-Next-80B-A3B | 80B | 3B | 512 | 10 |
| GLM-4.5-Air | 106B | 12B | 128 | 8 |

> 所有模型均无法完全放入 48GB GPU 显存，必须启用 expert offloading。

### 评估指标
| 指标 | 定义 | 目标 |
|------|------|------|
| **TTFT** (Time to First Token) | 首个输出 token 的延迟 | 越小越好 |
| **Prefill Speed** | 每秒处理的 prompt tokens 数 | 越大越好 |
| **TPOT** (Time Per Output Token) | 平均生成每个输出 token 的时间 | 越小越好 |
| **Decode Throughput** | 每秒生成的 output tokens 总数 | 越大越好 |

### 对比的基线方法
| Baseline | 类型 | 关键策略 |
|---------|------|----------|
| **Mixtral-Offloading** | Pure GPU Execution | 下一层专家预测 + 权重预取 |
| **Fiddler** | Static CPU-GPU Collaboration | 离线 profiling，固定驻留热门专家 |
| **Post-Fetch** | Dynamic CPU-GPU Collaboration | 当前步 CPU 执行，异步回传供后续复用 |

> 所有方法在相同框架和 GPU 专家池容量下复现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 整体性能提升（Figures 5 & 6）
#### Prefill Stage
- **平均提升 prefill speed：31.2%**
- **最高提升达 51.9%**
- 尤其在 batch size ≥ 8 时优势明显，因更大 batch 导致更多专家被激活且 token-load 更不均衡

#### Decode Stage
- **平均提升 decode throughput：31.1%**
- **最高提升达 39.5%**
- Post-Fetch 在 decode 中优于 Fiddler（得益于跨步复用），但仍落后于 Apex

#### 不同模型表现差异
| Model | Avg. Improvement |
|-------|------------------|
| Qwen3-Next (512 experts) | +36.4% |
| Qwen3.5 (256 experts) | +34.1% |
| GLM-4.5-Air (fewer, larger experts) | +22.9% |

> 说明 Apex 在专家粒度细、激活稀疏的模型中效果更好。

---

### 不同 GPU 内存预算下的表现（Table 2）
- 测试 GPU 专家池容量从 0.5G₀ 到 1.5G₀（G₀ 为基准配置）
- 结果显示：
  - Apex 在所有内存配置下均优于最强 baseline
  - **平均提升 prefill speed 33.0%，decode throughput 39.2%**
  - 表明 Apex 的自适应机制在**显存紧张和宽松场景下均有效**

> 🔍 图 7 显示：随着 GPU 容量增加，Apex 减少传输数量，自动适配资源变化。

---

### 消融实验（Ablation Study, Table 3）
在 Qwen3.5-35B-A3B 上验证各组件作用（batch size=16）：

| Configuration | CPU Path (ms) | GPU Path (ms) | Critical Path (ms) | Decode Throughput (tokens/s) |
|--------------|----------------|----------------|---------------------|-------------------------------|
| No assignment (all on CPU) | 3.074 | 0.295 | 3.074 | 72.44 |
| SA only (speculative only) | 2.515 | 0.408 | 2.515 | 84.62 (+16.8%) |
| EA only (exact only) | 2.220 | 2.320 | 2.320 | 90.31 (+24.7%) |
| **Full Apex (SA + EA)** | **2.130** | **1.930** | **2.130** | **103.35 (+42.7%)** |

> ✅ **SA 和 EA 是互补的**：
> - SA 利用重叠窗口提前传输，减轻 CPU 负担
> - EA 精确控制转移数量，实现路径平衡
> - 两者结合才能达到最佳性能

---

## 4. 关键结论和发现

### 主要发现
1. **专家 token-load 高度不均衡**，尤其在大 batch 场景下，这为自适应调度提供了优化空间。
2. **统一策略不可取**：无论是全传还是全 CPU 执行，都会导致某一路经成为瓶颈。
3. **路径平衡是关键**：Apex 通过 runtime-driven 的 exact + speculative assignment，在每层动态平衡 CPU 与 GPU 路径，最大化并行效率。
4. **Apex 具有强鲁棒性和泛化性**：在多种模型规模、batch size 和 GPU 显存限制下均保持显著优势。

---

### 方法的局限性
1. **依赖准确的 token-load 观测**：需要在路由完成后才能做出 exact assignment，存在一定决策延迟。
2. **推测性预测精度有限**：虽然利用相邻层 hidden state 相似性进行预测，但仍有误差，可能导致预取浪费。
3. **对专家大小敏感**：当专家本身很大（如 GLM-4.5-Air）时，单次传输代价高，可调空间变小。
4. **未考虑多 GPU 或分布式场景**：目前仅针对单机 CPU-GPU 协同设计。

---

### 未来工作方向
1. **更精准的专家需求预测**：引入轻量级 predictor 或历史 trace 学习模式。
2. **支持多 GPU 架构下的协同调度**：扩展至 multi-GPU + CPU 混合部署。
3. **结合量化与压缩技术**：进一步降低传输开销，增强 Apex 在低带宽环境下的适用性。
4. **动态调整专家池容量**：根据请求负载自适应调节驻留专家集合。

---

## 总结
✅ **Apex 是首个提出“基于 token-load 的自适应专家分配”机制的 MoE 推理系统**，它打破了传统“全传”或“全 CPU 执行”的僵局，通过动态平衡 CPU 与 GPU 的执行路径，在多种真实 MoE 模型上实现了高达 **51.9% 的 prefill 加速** 和 **39.5% 的 decode 吞吐提升**，为内存受限的大规模 MoE 模型部署提供了高效可行的解决方案。

</details>

---

### 2. [FuTransformer: FP8 prefill and INT4 decode for Llama-style inference on Hopper](https://arxiv.org/abs/2610.23796)

**Authors**: Wei Wang  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.23796v1  

#### Abstract
Transformer prefill reuses weights across tokens and is dominated by matrix multiplication and activation data movement; single-token decode has limited reuse and is dominated by weight and KV-cache reads. We present FUTRANSFORMER, a CUDA implementation of a Llama-style Transformer block for NVIDIA ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**FUTRANSFORMER: FP8 PREFILL AND INT4 DECODE FOR LLAMA-STYLE INFERENCE ON HOPPER**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Transformer 推理在 **prefill** 和 **decode** 阶段面临不同的性能瓶颈：
- **Prefill**：计算密集，受限于矩阵乘法（GEMM）和激活值的数据移动。
- **Decode**：内存带宽受限，受限于权重和 KV-Cache 的读取，且数据复用率低。

现有框架（如 PyTorch）在这些阶段未能充分优化数据流和低精度执行，导致效率低下。

---

### 提出的新方法与创新点

FUTRANSFORMER 是一个为 **NVIDIA Hopper GPU** 设计的 CUDA 实现，针对 Llama-style 模型推理进行了端到端优化，提出以下四大核心创新：

#### ✅ **1. 推理专用的 GEMM-plus-Epilogue 架构**
- 延续 CODA 的 GEMM 主循环 + 可编程 Epilogue 思路，但专为 **inference** 定制。
- 在 Epilogue 中融合多种操作，避免中间张量写回 HBM：
  - **直接 KV-Cache 写入**：QKV 投影后直接写入 KV-Cache，跳过中间张量。
  - **RMSNorm Gain 折叠**：将固定的 RMSNorm 增益提前折叠进权重，消除归一化激活张量。
  - **延迟缩放 FP8 输出**：使用延迟缩放（delayed scaling）直接从 Epilogue 写出 FP8 激活。
  - **支持 packed prefill**：处理变长请求时跳过 padding，减少无效计算。

#### ✅ **2. 片段对齐的 INT4 Decode 路径（Fragment-aligned INT4 decode）**
- 使用 **INT4 权重** 和 **INT4 KV-Cache 副本** 显著减少内存带宽压力。
- 权重和 Key 数据按 **Tensor Core MMA fragment 的布局打包存储**，解量化在寄存器中完成，无需中间 tile。
- 解量化通过：
  - **查表法（lookup table）** 快速展开 INT4 cache。
  - **寄存器逻辑运算** 展开 INT4 权重（nibble → BF16），避免额外 GEMM。

#### ✅ **3. 抗异常值的缓存量化（Outlier-aware cache quantization）**
- 引入 **块对角哈达玛变换（block-diagonal Hadamard transform）** 预处理 Key 和 Value。
- 作用：分散通道中的异常值（outliers），提升 INT4 量化的鲁棒性。
- 变换在写入缓存前应用，在查询和输出时逆变换，保持注意力计算等价。

#### ✅ **4. 完整路径验证（Final-result evaluation）**
- 不仅对比单层算子，还评估 **完整 Llama-3.1-8B 模型** 的端到端性能。
- 同时报告 **吞吐、延迟、数值误差（perplexity）和任务准确率**，提供全面评估。

---

### 相比现有方法的优势

| 方面 | FUTRANSFORMER 优势 |
|------|------------------|
| **Prefill** | 比编译后的 PyTorch FP8 快 **1.48×**，达到 **1.29 PFLOP/s** |
| **Decode** | INT4 路径比 PyTorch BF16 快 **1.3–2.4×**，比其 INT4 kernel 快 **1.3–6.3×** |
| **内存效率** | INT4 decode 比 FP8 decode 快 **31–51%** |
| **灵活性** | 支持变长 packed prefill，最高提速 **5.07×** |
| **数值保真度** | Kernel 与参考实现误差 < 0.001 perplexity |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **合成数据**：用于单层性能测试（prefill/decode latency, FLOP/s）。
- **WikiText-2**：用于完整模型的 **perplexity** 评估。
- **HellaSwag & ARC-Challenge**：用于零样本多选任务的 **accuracy** 评估。

---

### 实验设置
- **硬件**：单块 **NVIDIA H100 80GB SXM5**，CUDA 13.0，PyTorch 2.14.1。
- **模型规模**：测试 **1B, 8B, 70B, 405B** 层形状（非完整模型）。
- **完整模型**：**Llama-3.1-8B**（32 层）。
- **量化方式**：
  - **INT4**：对称分组量化（group size=128），支持 round-to-nearest 和 GPTQ。
  - **FP8**：E4M3 格式，延迟缩放。
- **对比模式**：
  - Prefill：对比 **compiled PyTorch + CUDA graph + FP8**。
  - Decode：对比 **PyTorch eager/BF16、compiled BF16、INT4 weight-only kernel**。
  - 内部对比：FUTRANSFORMER 内部不同精度组合（BF16/FP8/INT4 + BF16/E4M3/INT4 cache）。

---

### 评估指标
| 类别 | 指标 |
|------|------|
| **性能** | Latency (μs/ms), Throughput (tokens/s), FLOP/s, Speedup |
| **质量** | Perplexity (WikiText-2), Accuracy (HellaSwag, ARC-Challenge) |
| **误差** | RMSE, Relative RMS Error, KL Divergence |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔹 Prefill 性能（Batch=1）
- **速度提升**：相比 PyTorch FP8，**1.48×** 最大加速（405B, S=2048）。
- **峰值算力**：达到 **1.29 PFLOP/s**（接近理论上限）。
- **FP8 发射效率**：延迟缩放 FP8 输出减少内存流量，提升吞吐。

#### 🔹 Decode 性能（8B/70B）
| 对比项 | 加速比 |
|--------|--------|
| vs. PyTorch compiled BF16 | **1.3–2.4×** |
| vs. PyTorch INT4 weight-only kernel | **1.3–6.3×**（batch 32 时反超） |
| vs. 自身 FP8 decode 路径 | **31–51% 更快** |

> 示例：8B 模型，B=1, T=1024，FUTRANSFORMER INT4 decode 仅需 **77 μs**，而 PyTorch 编译版需 172 μs。

#### 🔹 完整模型推理（Llama-3.1-8B）
| 配置 | 单步延迟 (ms) | 吞吐 (tokens/s) | 相比 PyTorch 加速 |
|------|----------------|------------------|--------------------|
| PyTorch compiled BF16 | 8.1 | ~123 | 1× |
| FUTRANSFORMER INT4/INT4 | **2.8** | **362** | **2.9×** |

- **Prefill 2048 tokens**：FP8 路径耗时 **26.6 ms**，比 PyTorch 快 **2×**。

---

### 消融实验结果（Ablation Study）

| 修改项 | 影响（8B, B=32, T=4096） |
|--------|--------------------------|
| K 存储为自然顺序（非 fragment 对齐） | **+16% 延迟** |
| 使用算术解量化（而非查表） | **+25% 延迟** |
| 分离缓存量化核（非内联） | **+1–2 μs/step** |
| 权重解量化使用 bias MMA | **+9–20% 延迟** |
| 使用旧版 FMA split-KV attention | **+2.4× 延迟** |

✅ 结论：**fragment 对齐 + 寄存器级解量化 + 内联量化** 是性能关键。

---

## 4. 关键结论和发现

### 主要发现
1. **Prefill 和 Decode 应采用不同优化策略**：
   - Prefill 优化数据流和融合操作（GEMM+Epilogue）。
   - Decode 优化内存带宽（INT4 + fragment 对齐）。

2. **INT4 在 decode 阶段极具潜力**：
   - 权重和 KV-Cache 同时 INT4 化可带来 **显著加速（>2×）**。
   - **Hadamard 旋转** 可有效缓解 INT4 量化误差，尤其对抗 outliers。

3. **系统级融合至关重要**：
   - 跳过中间张量、直接写入 KV-Cache、折叠 RMSNorm，每一项都能节省数百微秒。

4. **数值误差可控**：
   - 使用 GPTQ + Hadamard 旋转，INT4 全路径仅使 perplexity 从 **6.24 → 7.18（+0.94）**。
   - Kernel 与模拟量化参考误差 < 0.001 perplexity，说明实现精确。

---

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **架构依赖性强** | 高度依赖 Hopper 的 fragment 布局、TMA、异步拷贝，难以直接移植到 Ampere 或 Ada Lovelace。 |
| **仅支持 decode 阶段 INT4** | Prefill 不支持 INT4 权重（因 GEMM 尺寸大，FP8 更优）。 |
| **质量损失存在** | INT4 路径在 ARC-Challenge 上准确率下降 **4.9–7.6 pts**，需权衡性能与质量。 |
| **未覆盖分布式场景** | 实验基于单卡，未涉及 tensor/pipeline parallelism。 |

---

### 未来工作方向
1. **扩展至其他架构**：适配 Blackwell 或 AMD GPU，抽象 fragment 布局。
2. **支持更多量化格式**：如 INT2、FP6、稀疏量化。
3. **动态精度调度**：根据 layer 或 token 动态选择 INT4/FP8/BF16。
4. **集成到推理服务系统**：结合 continuous batching、PagedAttention 等技术。
5. **更广泛的下游任务评估**：如生成质量、指令遵循能力等。

---

## 总结

FUTRANSFORMER 是首个将 **GEMM-plus-Epilogue** 范式完整应用于 **Transformer 推理** 的高性能实现。它通过：
- **prefill 阶段的 FP8 融合计算**
- **decode 阶段的 INT4 + fragment 对齐执行**
- **抗 outlier 的 Hadamard 量化**

实现了 **2–3× 的端到端加速**，同时保持了可接受的数值误差。该工作为 LLM 推理系统设计提供了重要范式：**将低精度执行与硬件原语深度对齐，是突破内存墙的关键路径**。

</details>

---

### 3. [ScoutRoute: Efficient Mixture-of-Experts Routing Using Internal Language Model Signals](https://arxiv.org/abs/2610.23443)

**Authors**: Rajdeep Pandhare, Delzaan Bharucha  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.23443v1  

#### Abstract
Efficient expert routing is a fundamental challenge in Mixture-of-Experts (MoE) systems, particularly for heterogeneous collections of independently trained Large Language Mod els (LLMs). Existing routing approaches typically rely on jointly trained gating networks, prompt-based predictors, or compu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《ScoutRoute: Efficient Mixture-of-Experts Routing Using Internal Language Model Signals》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在 **Mixture-of-Experts (MoE)** 系统中，高效且准确的 **expert routing** 是核心挑战，尤其是在由多个独立训练的异构 **Large Language Models (LLMs)** 组成的系统中。传统 MoE 路由器依赖于与专家联合训练的门控网络（gating network），无法直接应用于已训练好的异构模型集合。而现有的后处理（post-hoc）路由方法通常依赖 prompt embeddings、手工设计的置信度指标或多模型推理，存在计算开销大或未能充分利用模型内部信号的问题。

### **提出了什么新方法或新思路**
本文提出 **ScoutRoute** —— 一种轻量级、无需联合训练的后处理路由框架，其核心思想是：
- 利用一个低成本的 **scout model** 对输入进行一次前向传播（single forward pass）；
- 从中提取 **hidden-state representations** 和 **logit-derived confidence** 两类内部语言模型信号；
- 使用轻量级的 **logistic probes** 基于隐藏状态预测各候选专家的胜任力（competence）；
- 结合 **domain-aware routing 策略**，根据不同任务领域选择最优路由信号（competence probe 或 mean log probability）进行专家选择。

该方法完全无需修改或重新训练底层专家模型，实现了高效的跨模型能力预测。

### **相比现有方法的优势**
- ✅ **高效性**：仅需一次 scout model 推理，避免多模型并行执行；
- ✅ **灵活性**：适用于任意独立训练的异构 LLM 集合；
- ✅ **无需联合训练**：不依赖与专家共同优化的路由器；
- ✅ **利用内部信号**：比基于 prompt embedding 或外部编码器的方法更充分地利用了模型推理过程中的信息；
- ✅ **更好的校准性（calibration）**：probe-based competence 估计比传统 logit confidence 更可靠（更低的 ECE）。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖四个代表性基准，涵盖多种推理类型：
- **GSM8K**：数学推理（mathematical reasoning）
- **HumanEval**：代码生成（code generation）
- **CommonsenseQA**：常识推理（commonsense reasoning）
- **MMLU**：多学科知识理解（multidisciplinary knowledge）

这些数据集提供了多样化的任务场景，用于评估在不同领域下的路由有效性。

### **实验设置和评估指标**

#### **候选语言模型（Expert Pool）**
共使用五个开源权重的 LLMs，包括通用与专用模型：
- `DeepSeek-R1-Distill-Llama-8B`
- `Llama-3.1-8B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Qwen2.5-Coder-7B`
- `Qwen2.5-Math-7B`

#### **Scout Models**
测试了多个模型作为 scout model：
- `Qwen2.5-1B`, `Phi-3 Mini`, `Qwen2.5-3B`, `Qwen2.5-Math`, `Qwen2.5-Coder`

#### **评估指标**
- **Routing Accuracy**：所选专家输出正确答案的比例；
- **AUROC**（Area Under ROC Curve）：衡量 competence probe 的判别能力；
- **ECE**（Expected Calibration Error）：衡量预测置信度与实际准确率的一致性，越低越好；
- **Oracle Router**：理想上限（已知哪个专家表现最好）；
- **Random Routing**：随机选择专家，作为下限。

#### **训练与划分**
- 每个数据集中 30% 的样本用于训练 competence probes；
- 剩余 70% 用于路由评估；
- 使用 PCA 进行 hidden-state 降维，再训练带类别平衡的 logistic 回归探针。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| **Routing Method**             | **Accuracy** |
|-------------------------------|-------------|
| Random Routing                | 35.36%      |
| Negative Entropy              | 63.47%      |
| Probe Score                   | 63.16%      |
| Mean Log Probability          | 65.02%      |
| **ScoutRoute (Domain-Aware)** | **67.80%**  |
| Oracle Router                 | 82.66%      |

👉 ScoutRoute 在所有实际可行方法中表现最佳，显著优于单一信号方法。

### **与基线方法的对比结果**
- ScoutRoute 比最强的单一路由信号（mean log probability）提升约 **2.8个百分点**；
- 明显优于 random、entropy、probe-only 等策略；
- 尽管未达到 oracle 上限，但在仅一次推理的前提下接近实用极限。

### **消融实验与分析结果**

#### **(1) Competence Probe 性能（AUROC）**
| Scout Model       | AUROC     |
|-------------------|----------|
| Qwen2.5-1B        | 0.8957   |
| Phi-3 Mini        | 0.8610   |
| Qwen2.5-3B        | 0.8231   |
| Qwen2.5-Math      | 0.7240   |
| Qwen2.5-Coder     | 0.6823   |

✅ 表明 compact general-purpose models（如 Qwen2.5-1B）比 specialist models 更适合作为 scout model。

#### **(2) 校准性分析（ECE）**
| Scout Model       | Log Prob. | Neg. Entropy | **Probe** |
|-------------------|-----------|--------------|-----------|
| Qwen2.5-1B        | 0.4691    | 0.4299       | **0.1456** |
| Phi-3 Mini        | 0.4723    | 0.3155       | **0.1313** |

✅ 所有模型上，**probe 的 ECE 显著低于 logit confidence 方法**，说明其置信度估计更可靠。

#### **(3) 不同 Scout Model 的最终路由效果**
| Scout Model       | Probe Router | **ScoutRoute** |
|-------------------|--------------|----------------|
| Qwen2.5-1B        | 65.33%       | 67.80%         |
| Phi-3 Mini        | 65.33%       | **68.42%**     |
| Qwen2.5-Math      | 58.51%       | 65.33%         |

✅ **Phi-3 Mini** 实现最高端到端准确率（68.42%），验证小而通用的模型可成为优秀 scout；
✅ domain-aware 策略提升了鲁棒性，缩小了不同 scout model 之间的差距。

#### **(4) Specialist Routing 分析**
| Domain      | Specialist       | Routed (%) | Specialist Acc. | Defection Acc. |
|-----------|------------------|------------|------------------|----------------|
| GSM8K     | Qwen2.5-Math     | 72.86%     | 86.27%           | 73.68%         |
| HumanEval | Qwen2.5-Coder    | 72.17%     | 67.47%           | 15.62%         |

✅ 数学任务中 specialist 表现优异，defection 合理；
❌ 但在 HumanEval 中，一旦离开 coder 专家，性能急剧下降 → 支持引入 **defection threshold** 控制切换行为。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **内部表示具有迁移性**：scout model 的 hidden states 可有效预测其他异构 LLM 的胜任力，表明 LLM 内部表征蕴含可迁移的认知信息。
2. ✅ **hidden-state + logit confidence 是互补信号**：结合二者并通过 domain-aware 策略融合，能显著提升路由准确性。
3. ✅ **轻量通用模型优于专家作为 scout**：`Phi-3 Mini` 和 `Qwen2.5-1B` 等小型通用模型比 specialist 更适合担任 scout，因其表征更具泛化性。
4. ✅ **probe 比传统 confidence 更优校准**：competence probe 提供更可靠的置信度估计（低 ECE），更适合高风险决策场景。

### **方法的局限性**
- ❌ 依赖高质量的 domain classifier（当前使用 TF-IDF + logistic regression，较简单）；
- ❌ competence probe 需要在目标数据集上有标注的“正确性”标签进行训练；
- ❌ 当前仅支持文本任务，尚未扩展至 multimodal 场景；
- ❌ 对非常见或分布外（OOD）查询的泛化能力有待验证。

### **未来工作方向**
- 🔮 扩展至更大规模的 expert pools 和更多样化的 reasoning domains；
- 🔮 改进 domain classification 模块（例如使用 fine-tuned encoder）；
- 🔮 引入在线学习机制，动态更新 competence probes；
- 🔮 探索将 ScoutRoute 应用于 **multimodal foundation models**；
- 🔮 研究 zero-shot competence probing，减少对 labeled data 的依赖。

---

> 📌 **总结一句话**：  
> **ScoutRoute 通过挖掘单次前向传播中的内部信号，实现了一种高效、灵活且无需联合训练的 MoE 路由机制，在保持极低计算成本的同时显著提升了异构 LLM 集合的路由精度与可靠性。**

</details>

---

### 4. [HydraLLM: Adaptive Width Slicing for Elastic Text and Vision-Language Models](https://arxiv.org/abs/2610.23455)

**Authors**: Oscar Kemme, Janek Haberer, Ali Hojjat, Olaf Landsiedel  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.23455v1  

#### Abstract
LLMs and LVLMs are increasingly deployed across heterogeneous hardware, ranging from data-center GPUs to resource-constrained edge devices. Their fixed-size architectures limit the ability to reuse a single model across deployment settings with different compute, latency, and model-size targets. We ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HydraLLM: Adaptive Width Slicing for Elastic Text and Vision-Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLMs）和视觉-语言模型（LVLMs）在从数据中心GPU到资源受限的边缘设备等异构硬件上部署时，面临**固定架构无法适应不同计算、延迟和模型大小需求**的问题。训练多个专用模型成本高昂且不现实。

现有弹性方法（如仅对FFN进行切片的MatFormer）通常只针对单一组件，缺乏端到端的宽度控制能力，限制了参数配置的灵活性和效率。

### **提出了什么新方法或新思路**

本文提出 **HydraLLM**，一种基于 **MQA（Multi-Query Attention）** 的弹性架构，支持**端到端自适应宽度切片（adaptive width slicing）**，其核心思想包括：

- **单头计数控制（Head-count Control）**：通过一个运行时参数 `k`（激活的query heads数量）统一控制整个解码器的有效宽度，包括：
  - Token Embeddings 维度
  - Attention 投影维度
  - FFN 层宽度
  - 输出头（output head）
  
  这种设计形成了一个**嵌套子网络家族（nested subnetworks）**，所有子网络共享同一组参数。

- **静态到弹性转换流程（Static-to-Elastic Conversion Pipeline）**：
  1. **重要性驱动权重重排序（Importance-driven Weight Reordering）**：在训练前对各组件（embeddings, heads, FFN neurons）按重要性重新排列，确保前缀切片保留最关键参数。
  2. **带知识蒸馏的随机子网络训练（Stochastic Subnetwork Training with Knowledge Distillation）**：每步随机采样一个 `k`，仅更新对应子网络，并利用教师模型进行logit级监督。

- **弹性多模态投影模块（Elastic Projection Module）**：扩展至LVLM场景，使视觉特征到文本空间的投影输出宽度 `dP_active` 动态跟随当前文本子网络的激活宽度 `dE_active`，实现跨模态的统一弹性轴。

### **相比现有方法的优势**

| 方面 | HydraLLM | 典型基线（如MatFormer） |
|------|----------|------------------------|
| 切片范围 | **端到端联合切片**（Embedding + Attention + FFN） | 通常仅切片 **FFN** |
| 参数覆盖范围 | 更广（最小可低至~49M参数） | 起始点更高（如~215M） |
| 弹性粒度 | 单一控制变量 `k`，保证一致性 | 多维独立控制，可能不一致 |
| 多模态支持 | 支持弹性视觉-语言接口 | 无相关设计 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### 文本任务：
- **预训练数据**：FineWeb-Edu 350BT 子集
- **校准/重排序数据**：Tiny Shakespeare 数据集（用于收集激活统计）
- **评估基准**：
  - **TriviaQA**（五次提示精确匹配准确率）
  - **ARC-Easy**（原始与答案长度归一化准确率）
  - **HellaSwag**（十次提示）
  - **Perplexity**：在莎士比亚戏剧样本上测量

#### 多模态任务：
- **训练数据**：
  - Captioning: PixMo-Cap, COCO Recaptioned, LLaVA Pretrain
  - QA: PixMo Ask Model Anything, PixMo Cap-QA, CoSyn-400K VQA
  - Code Generation: CoSyn-400K Code
  - Text-only: Wikipedia, FineWeb-Edu
- **对齐数据**：64k COCO Recaptioned + 64k PixMo-Cap
- **评估基准**：
  - **CLIPScore**（×100）：在COCO验证集上使用 `clip-vit-large-patch14`
  - **MMMU**：大规模多学科多模态理解基准
  - 定性生成示例分析

### **实验设置和评估指标**

| 设置项 | 描述 |
|-------|------|
| **基础模型** | Google Gemma 3 270M（H=4 heads） |
| **弹性配置** | `k ∈ {1,2,3,4}` 对应 H1–H4 子网络 |
| **训练策略** | 非均匀采样（full-width 加权×3），AdamW优化器，余弦学习率调度 |
| **知识蒸馏** | 教师模型为冻结的 Gemma 3 270M 或 Gemma 3 4B |
| **吞吐量测试** | 使用 `llama.cpp` 在 NVIDIA RTX PRO 4000 上测试：<br>• Prefill: 2048 tokens<br>• Decode: 512 tokens |
| **对比基线** | MatFormer（仅FFN切片）、原始Gemma 3 270M/4B |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ 文本任务表现（Gemma 3 270M 基础）：

| 指标 | HydraLLM (H3, 75% heads) | 基线（MatFormer W0.25） | 优势 |
|------|----------------------------|-------------------------|------|
| **TriviaQA 准确率** | **5.90%** | 3.69% | **↑2.21个百分点** |
| **参数量** | ~160M | ~188M | ↓17.25% 更少参数 |

> 💡 尽管参数更少，HydraLLM 在 H3 下仍显著优于更大尺寸的 MatFormer W0.25。

| 指标 | HydraLLM (H1, 25% heads) | Full Model (H4) | 提升 |
|------|----------------------------|--------------------|------|
| **Token Generation Throughput** | **1,057.0 t/s** | 603.0 t/s | ↑**75.3%** |

> ⚠️ 注意：Prefill 吞吐提升较小（约10%），主要收益来自 decode 阶段。

#### ✅ 多模态任务表现（结合 Gemma 3 4B 视觉塔）：

| 指标 | HydraLLM (H2, 50% heads) | Gemma 3 4B (7×更大) | 结果 |
|------|----------------------------|------------------------|------|
| **CLIPScore** | **66.9** | 65.3 | **反超 +1.6 分** |

> 🎉 表明即使只有半宽文本模型，也能达到甚至超过完整大模型的图像描述质量评分。

| 指标 | HydraLLM (H1, 25% heads) | Full Model (H4) | 提升 |
|------|----------------------------|--------------------|------|
| **Token Generation Throughput** | **1,057.0 t/s** | 603.0 t/s | ↑**75.3%** |

> 🔁 与纯文本一致，极小宽度下生成速度大幅提升。

### **与基线方法的对比结果**

| 对比维度 | HydraLLM vs MatFormer |
|--------|------------------------|
| **参数覆盖范围** | HydraLLM 可达 **49.34M → 270M**<br>MatFormer 仅 **215M → 270M** | ✅ 广得多 |
| **Accuracy-Compute Trade-off** | 在相似大小下，HydraLLM H3 显著优于 MatFormer W0.25 | ✅ 更高效 |
| **Throughput Scaling** | HydraLLM decode 速度提升 **75.3%**<br>MatFormer 仅 **11.17%** | ✅ 极大优势 |
| **Prefill Speed** | 两者均提升有限（<12%） | ❌ 共同瓶颈 |

### **消融实验结果**

#### （1）重要性重排序策略比较（表1 & 表2）

| 重排序策略 | H1 Perplexity ↓ | H2 Perplexity ↓ | 最佳组合 |
|-----------|----------------|----------------|----------|
| No Reordering | 123.9 | 77.4 | — |
| Mag-CETT-CETT | **106.6** | **71.6** | ✅ 最优 |
| Mag-Mag-Mag | 110.1 | 71.9 | 次优 |
| VxC-VxC-VxC | 118.1 | 71.7 | 差于其他 |

> ✅ **Embedding用Magnitude，Heads/FFN用CETT** 是最佳组合。

#### （2）知识蒸馏作用（图6及正文）

- 移除KD后，H3下的TriviaQA准确率下降 **97.3%**（相对值），说明**知识蒸馏对小模型性能至关重要**。

#### （3）多模态训练策略（表4）

| 训练方式 | H2 CLIPScore | H4 CLIPScore |
|---------|-------------|-------------|
| 仅训练Projection | 31.4 | 33.5 |
| Projection + Text Model | 53.9 | 58.8 |
| + Pre-alignment | **56.9** | **61.0** |
| + Full-parameter Train (16k steps) | **60.7** | **62.5** |

> ✅ **预对齐 + 联合训练 + 全参微调** 是最优路径。

---

## 4. 关键结论和发现

### **主要发现**

1. **端到端宽度切片可行且高效**：首次成功将HydraViT风格的头堆叠（head stacking）应用于decoder-only LLM，实现真正的运行时弹性推理。
2. **统一控制优于局部切片**：相比仅切片FFN的方法（如MatFormer），HydraLLM能覆盖更广的参数谱系，在相同或更低参数下取得更好性能。
3. **吞吐增益集中在decode阶段**：由于MQA机制禁止KV Cache压缩，prefill加速有限；但decode阶段因更小的矩阵运算获得高达 **75.3%** 的吞吐提升。
4. **弹性多模态接口有效**：通过让视觉投影宽度随文本宽度动态调整，可在较小子网络上复现接近大模型的CLIPScore表现。
5. **重排序 + KD是关键**：**Magnitude+CETT重排序** 和 **知识蒸馏** 是保障小模型性能的核心技术。

### **方法的局限性**

1. **生成质量不稳定**：尽管CLIPScore高，但定性分析显示模型存在严重**幻觉（hallucination）** 和**对象识别错误**（见图6），尤其在低k配置下。
2. **MMMU表现差**：在需要强推理的MMMU基准上仅为随机水平（~25%），远低于Gemma 3 4B（~39%），表明**视觉-语言对齐仍薄弱**。
3. **投影层容量不足**：小型单层投影难以充分对齐高维视觉特征与文本空间，成为瓶颈。
4. **MQA限制KV Cache优化**：无法通过减少key/value heads来节省内存或计算。

### **未来工作方向**

1. **增强投影模块**：替换为**更深或多层FFN结构**的投影器，以提升视觉特征对齐能力。
2. **多样化多模态训练**：引入更多非captioning任务（如VQA、推理）以改善接地能力和鲁棒性。
3. **融合细粒度控制**：结合MatFormer式的**逐层弹性控制**，实现更精细的accuracy-compute权衡。
4. **探索KV弹性机制**：设计支持key/value head动态调整的注意力变体，突破MQA限制。
5. **硬件协同优化**：针对弹性特性定制推理引擎，进一步释放吞吐潜力。

--- 

> 📌 **总结一句话**：HydraLLM通过**单头计数控制实现了LLM/LVLM的端到端弹性切片**，在保持竞争力的同时大幅扩展了可用参数范围，并在生成吞吐和部分多模态指标上超越更大模型，为异构部署提供了强大工具，但在生成忠实性和复杂推理方面仍有待加强。

</details>

---

### 5. [On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction](https://arxiv.org/abs/2610.22543)

**Authors**: Eliezer Shlomi, Eliron Rahimi, Avi Mendelson, Chaim Baskin  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.22543v1  

#### Abstract
Diffusion language models (DLMs) promise parallel text generation, but in practice face an unresolved speed-quality trade-off. Exact decoding fills in only one masked position per model pass, while faster multi-token decoding schemes are lossy. Speculative decoding is the standard lossless accelerat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction

## 1. 论文的主要贡献和创新点

### 解决了什么问题
扩散语言模型（**Diffusion Language Models, DLMs**）虽然理论上支持并行文本生成，但在实践中面临一个未解决的**速度-质量权衡**（speed-quality trade-off）。传统的精确解码（exact decoding）每次仅填充一个被掩码的位置，效率低下；而更快的多令牌解码方案则往往是**有损的**（lossy），会改变生成轨迹并降低质量。

为DLMs引入**推测解码**（speculative decoding）这一标准的无损加速技术面临根本性挑战。与自回归模型（AR-LMs）不同，DLMs在每个前向传递中必须同时决定“**生成什么**”（what token）和“**在哪里生成**”（where to generate），即需要预测**提交顺序**（commit order）。

### 提出了什么新方法或新思路
本文提出了 **SPOT (Speculating Positions Order Trajectory)** 方法，其核心思想是：**提交顺序预测是推测解码的关键瓶颈，而非令牌预测本身**。

- **核心观察**：研究发现，在推测解码的相关范围内，目标模型（target model）从上一轮得到的过时输出分布（stale output distributions）已经包含了非常可靠的令牌信息。其最高概率的预测（top-1 prediction）与最终提交的令牌匹配度高达92%-100%。然而，准确预测下一个将被提交的位置要困难得多。
- **新方法**：
  1. **学习提交顺序**（Learned Commit-Order Drafter）：SPOT 引入了一个轻量级的（0.62M参数）**顺序模型**（order model），该模型以目标模型的过时输出分布作为输入，专门用于预测下一个最有可能被提交的位置 `s`。它不负责预测令牌，而是专注于解决“**在哪里**”的问题。
  2. **构建状态树**（Tree of States）：基于预测的顺序，SPOT 构建一棵候选块状态的树（draft tree）。这棵树通过递归地查询顺序模型来生长，并优先考虑高概率的分支。
  3. **耦合噪声验证**（Coupled-Noise Match Verification）：为了解决在采样模式（sampling, T>0）下，独立随机抽样导致的状态等价验证失效的问题，SPOT 采用了一种**共享Gumbel噪声**（shared-Gumbel-noise）的方法。通过让草稿器（drafter）和目标模型共享相同的随机噪声，使得目标模型的采样过程在给定噪声的条件下变为确定性的，从而可以再次使用状态等价性进行验证。

### 相比现有方法的优势
- **无损且高效**：SPOT 在任何温度下都能**无损地**（losslessly）保留目标模型的原始生成分布，这是许多近似方法无法做到的。
- **性能显著提升**：相比现有的推测解码基线（如 Spiffy, SSD, BlockSpec），SPOT 能够大幅减少目标模型的前向传递次数（NFE），并显著提高吞吐量（throughput）。
- **可迁移性强**：训练好的SPOT草稿器可以在不同的DLM之间迁移，无需重新训练，表现出良好的泛化能力。

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在四个广泛使用的基准数据集上进行：
- **GSM8K**: 数学应用题数据集。
- **MBPP** (Mostly Basic Python Problems): Python编程问题数据集。
- **HumanEval**: 编程代码补全数据集。
- **MATH500**: 数学问题数据集。

### 实验设置和评估指标
- **目标模型**（Target Models）：主要在 **SDAR-8B** 上进行实验，并扩展到 **SDAR-4B**, **SDAR-1.7B**, **LLaDA-2.0-mini**，以及用于迁移测试的 **LLaDA-8B** 和 **Dream-7B**，共覆盖了六种不同的DLM。
- **评估指标**：
  - **NFE** (Number of Forward Evaluations)：目标模型的前向传递次数，衡量计算成本。
  - **Throughput Ratio**：吞吐量比率，即每秒生成的令牌数（tok/s）相对于标准无草稿解码（draft-free decoding）的倍数，衡量实际加速效果。
  - **Accuracy**：任务完成的准确率。
- **温度设置**：在贪婪解码（T=0）和采样（T=1）两种模式下均进行了评估。
- **预算**（Budget D）：控制一次验证中草稿状态的数量，实验中使用了 D=8, 12, 20 等不同预算。

### 基线方法对比
与以下几种代表性的DLM推测解码方法进行了对比：
- **Spiffy**：使用离线校准的静态草稿图（calibrated draft graph）。
- **SSD** (Self Speculative Decoding)：按置信度排序位置形成一条自草稿链（free chain）。
- **BlockSpec**：基于置信度构建草稿树（confidence tree）。
- **no draft (D0)**：标准的无草稿逐位解码，作为性能基准。

## 3. 主要实验结果和性能指标

### 关键性能数据
- **在 SDAR-8B 上，SPOT 将目标模型的前向传递次数（NFE）最多减少了 6.8 倍**。
- **吞吐量（throughput）最高达到了标准解码的 3.2 倍**。
- 在 **T=1** 的采样模式下，SPOT 依然保持了强大的性能优势。

### 与基线方法的对比结果
根据论文中的 **Table 2**，在 SDAR-8B 上的结果如下：

| 方法 | NFE (D8) | 吞吐量比率 (D8) | NFE (D20) | 吞吐量比率 (D20) |
| :--- | :--- | :--- | :--- | :--- |
| **SPOT (ours)** | **56.7** | **2.61** | **39.6** | **1.72** |
| Spiffy graph | 83.4 | 1.86 | 61.3 | 1.17 |
| SSD free chain | 84.7 | 1.80 | 76.1 | 0.91 |
| BlockSpec tree | 135.0 | 1.10 | 111.5 | 0.58 |

**结论**：
- SPOT 在所有任务和预算设置下，无论是 NFE 还是吞吐量比率，都**全面超越**了所有基线方法。
- 相比最强的基线 **Spiffy**，SPOT 减少了 **15-35%** 的 NFE，并实现了 **1.1-1.5 倍**的吞吐量提升。

### 消融实验结果
消融实验（Ablation Study）验证了SPOT各组件的有效性：
- **树结构 vs 链结构**（Tree vs Chain）：将树替换为链会使NFE显著增加（D8时增加7.3，D20时增加11.9），证明了**树形结构对于有效利用预算至关重要**。
- **SPOT-y**：这是一个改进版本，通过引入一个前缀质量下限（mass floor）来自适应调整验证预算。它在保持较低NFE的同时，使吞吐量比率对预算大小不敏感，实现了更稳定的性能。
- **验证方式**：在T>0时，**耦合噪声匹配**（coupled-noise match）相比独立匹配（uncoupled match）能显著提高接受率，尤其是在高温下，后者几乎无法接受任何提议。

## 4. 关键结论和发现

### 论文的主要发现
1.  **核心瓶颈是顺序，而非令牌**：对于DLM的推测解码，最大的信息瓶颈在于预测“**在哪里**”（提交顺序），而不是“**是什么**”（令牌身份）。目标模型的过时分布已经提供了足够好的令牌假设。
2.  **SPOT方法的有效性**：通过一个轻量级模型专门学习提交顺序，并结合耦合噪声验证，SPOT成功地将高效的推测解码技术应用于DLM，实现了巨大的加速。
3.  **强大的可迁移性**：在一个模型上训练的SPOT草稿器，可以直接迁移到其他完全不同的DLM（如LLaDA-8B和Dream-7B）上使用，并仍然优于这些模型上的校准基线，证明了所学知识的通用性。

### 方法的局限性
- **验证成本成为瓶颈**：当块宽度（block width）很大时，即使只接受一个提议，也需要对整个 `(1+D)L` 序列进行前向传递，此时**验证成本**（verification cost）会成为新的瓶颈。
- **低接受率任务表现不佳**：在一些接受率本身就比较低的任务上，SPOT的增益会缩小。
- **高温下的性能下降**：在 `T≥2` 的极高温度下，目标模型的质量急剧下降，且在T=0下训练的顺序模型会被能够感知噪声的排名策略超越。

### 未来工作方向
- **解决双向注意力下的精确树打包问题**：当前的树状草稿在验证时是作为独立副本处理的，未能充分利用双向注意力的特性。未来的工作可以探索如何在单次前向传递中更精确地打包和验证整棵草稿树。
- **优化验证机制**：设计更高效的验证协议，以降低宽块场景下的验证开销。
- **探索更复杂的草稿器架构**：虽然当前的0.62M模型已足够，但可以进一步探索更大容量的模型是否能在更复杂的任务上带来额外收益。

</details>

---

### 6. [CoTA++: Understanding the Repeat Curse in dMLLMs from an Information Flow Perspective](https://arxiv.org/abs/2610.23290)

**Authors**: Qiyan Zhao, Xiaofeng Zhang, Xu-Yao Zhang, Shijie Han, Zhepeng Li, Shunzhi Zhu, Da-Han Wang  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.23290v1  

#### Abstract
Diffusion-based multimodal large language models (dMLLMs) decode by iterative denoising with bidirectional attention, which precludes the standard autoregressive key--value cache. Practical deployments therefore rely on approximate caching that reuses intermediate states across denoising steps. Such...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*CoTA++: Understanding the Repeat Curse in dMLLMs from an Information Flow Perspective*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于 **diffusion-based multimodal large language models (dMLLMs)** 在使用 **近似缓存 (approximate caching)** 加速推理时出现的“**重复诅咒 (Repeat Curse)**”问题。该问题表现为模型在生成文本时，会连续重复同一个 token（如 "the the the..."），严重破坏句子的连贯性和可读性。

尽管近似缓存能显著提升推理速度，但这种重复现象在长文本生成中极为普遍，影响高达 **99.6%** 的样本，成为阻碍 dMLLMs 实际部署的关键瓶颈。

### 提出了什么新方法或新思路
作者提出了一种名为 **CoTA++** 的训练免费、即插即用（training-free, plug-and-play）框架，从**信息流 (information flow)** 的视角系统性地诊断并修复了导致重复诅咒的三个根本性机制中断：

1. **Context-Token Attention Re-anchoring (CTAR)**  
   - **问题**：在单步内，缓存导致注意力分散，无法锚定在当前上下文上（F1: Routing）。
   - **解决**：在解码过程中，动态地将注意力重新锚定到最新的上下文 token 上。

2. **Decode-Aware Refresh (DAR)**  
   - **问题**：跨步骤的缓存刷新策略不感知解码状态，导致即将被提交的 token 基于过时的 KV 状态进行解码（F2: Timeliness）。
   - **解决**：在缓存刷新预算中预留一部分，专门用于刷新那些**即将被提交 (about to be committed)** 的位置，确保其 KV 状态新鲜。

3. **Context-Token Entropy-Guided Voting (CTEV)**  
   - **问题**：跨层的信息未能有效收敛，重复 token 往往来自未充分收敛的上下文（F3: Consolidation）。
   - **解决**：引入基于上下文深层熵的投票分数，降低在未收敛上下文中候选 token 的置信度，从而优先选择更可靠的 token。

### 相比现有方法的优势
- **无需训练**：完全在推理阶段运行，不修改模型参数。
- **模块化设计**：三个组件作用于不同阶段，可自由组合，兼容多种 dMLLM 和缓存后端。
- **高效可靠**：在保持缓存加速优势的同时，几乎彻底消除重复问题。
- **理论驱动**：基于对信息流的深入分析，而非经验性修补。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主数据集**：**MS-COCO val2014** 图像描述数据集，共 **500 张图像**，用于评估长文本生成中的重复问题。
- **多模态基准测试集**（共 9 个）：
  - DocVQA, ChartQA, MMStar, MME, SEED-Bench, MMBench, LLaVA-Bench (in-the-wild), MathVista, MathVerse
  - 用于评估模型在短答案和长答案任务上的通用多模态能力。

### 实验设置和评估指标

#### 模型与缓存后端
- **dMLLMs**：LLaDA-V-8B, MMaDA-8B, LaViDa-8B
- **缓存后端**：dLLM-Cache [18], SlowFast [19]

#### 评估指标
- **重复相关指标**：
  - **ARR (Adjacent Repetition Rate)**：相邻 token 重复的比例。
  - **SRR (Sample Repetition Rate)**：至少包含一次重复的样本比例。
  - **MRL, ARL, 95pRL**：最大、平均、95%分位的重复长度。
  - **seq-rep-n, distinct-n**：n-gram 层面的重复率。
- **质量指标**：
  - **Judge Score**：由 GPT-5.6-Luna 对生成文本的准确性、细节和连贯性打分（1-10）。
  - **响应长度 (Len)**：生成文本的平均 token 数。

### 基线方法对比
- **Vanilla**：无缓存的标准迭代去噪。
- **dLLM-Cache / SlowFast**：主流的近似缓存方法。
- **+CoTA**：作者之前的工作，仅包含 CTAR 和 CTEV。
- **+CoTA++**：本文提出的完整方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | Vanilla | dLLM-Cache | **CoTA++** |
|------|--------|------------|-----------|
| **SRR (平均)** | ~1.0% | **~90%** | **~3.7%** |
| **ARR (平均)** | ~0.03% | **~8.0%** | **~0.05%** |
| **最长重复长度 (MRL)** | ≤3 | 达 **465** | **≤3** |

> ✅ **CoTA++ 将样本重复率 (SRR) 从最高 99.6% 降至最多 17.0%，且在多数设置下低于 10%。**

### 与基线方法的对比结果
- **相比 dLLM-Cache / SlowFast**：
  - SRR 平均下降 **95% 以上**。
  - 所有重复运行长度被限制在 **3 个 token 以内**。
  - 在 LLaDA-V 上，SRR 从 99.6% (L=512) 降至 **8.2%**。
- **相比 CoTA**：
  - 进一步显著降低 SRR 和 ARR。
  - 在 L=512 时，CoTA 仍存在长达 **286** 的重复，而 CoTA++ 完全消除。
- **速度开销**：
  - 仅增加 **<15%** 的延迟，仍比无缓存快 **2.1 倍以上**。

### 消融实验结果
| 组件组合 | SRR (L=512) | 最长重复长度 |
|--------|------------|-------------|
| **仅 DAR** | 11.2% | 165 |
| **仅 CTAR** | 62.0% | 160 |
| **仅 CTEV** | 94.0% | 420 |
| **DAR + CTAR** | 9.6% | 3 |
| **DAR + CTEV** | 12.2% | 25 |
| **CTAR + CTEV (CoTA)** | 47.4% | 3 |
| **DAR + CTAR + CTEV (CoTA++)** | **8.2%** | **3** |

> 🔍 结论：
> - **DAR 是最有效的单一组件**，因其解决了最关键的“过时状态”问题。
> - **任意两个组件组合即可消除超长重复**。
> - **三者联合达到最佳效果**，证明三个机制是正交且互补的。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **重复诅咒源于信息流的三重断裂**：
   - **F1 (Routing)**：注意力无法锚定最新上下文。
   - **F2 (Timeliness)**：解码依赖过时的 KV 状态。
   - **F3 (Consolidation)**：上下文预测未充分收敛。
2. **CoTA++ 成功修复了每一处断裂**，且各组件作用独立、可组合。
3. **重复主要发生在功能词**（如 "the", "is"），这些词高度依赖上下文，一旦信息流受损便最先出错。

### 方法的局限性
- 当前方法为**固定预算分配**（如固定 `r=4` 个 DAR 插槽），未实现动态自适应。
- 未在更大规模或更多样化的 dMLLM 架构上广泛验证。
- 对某些极端情况（如极短块解码）仍有轻微残留重复。

### 未来工作方向
- 设计**统一的风险评分函数**，动态分配计算资源。
- 将 CoTA++ 的洞察反馈至 **dMLLM 的训练过程**，从根本上增强鲁棒性。
- 探索更高效的缓存设计，使其天然支持信息流完整性。
- 扩展至其他生成模式（如语音、视频）中的扩散模型。

---

> **总结**：本文通过信息流分析揭示了 dMLLM 缓存中“重复诅咒”的深层机理，并提出了 **CoTA++** 这一高效、通用、无需训练的解决方案，在几乎不牺牲速度的前提下，将重复率从接近 100% 降至个位数，为 dMLLMs 的实际应用铺平了道路。

</details>

---

### 7. [BaGuaLu-2: Taming Dynamic Memory Fluctuation for Efficient MoE Model Training over 10,000 GPUs](https://arxiv.org/abs/2610.23064)

**Authors**: Mingshu Zhai, Minkun Guo, Hongrui Lian, Shuangyu Li, Qi Zhang, Chen Hu, Xiawei Du, Yuyang Jin, Zan Zong, Jidong Zhai  
**Category**: cs.DC  
**Published**: 2026-10-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.23064v1  

#### Abstract
Fine-grained Mixture-of-Experts (MoE) is the de facto architecture for scaling large language models. However, dynamic expert load imbalance causes severe memory fluctuations that degrade training efficiency. Existing systems rely on static memory provisioning: they either over-provision, leaving GP...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BaGuaLu-2: Taming Dynamic Memory Fluctuation for Efficient MoE Model Training over 10,000 GPUs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
细粒度 **Mixture-of-Experts (MoE)** 架构在大规模训练中面临**动态专家负载不均衡**（expert load imbalance）导致的**内存波动**（memory fluctuation）问题。这种波动使得传统静态内存管理策略陷入两难：
- **Over-provisioning**：预留过多内存以避免 OOM（Out-of-Memory），但导致大部分时间 GPU 内存利用率低下，降低 **MFU**（Model FLOPs Utilization）。
- **Under-provisioning**：最大化内存利用率，但在负载峰值时触发 OOM，导致训练中断并需要手动重启，带来高昂恢复开销。

### 提出的新方法
提出 **BaGuaLu-2**，一个支持**灵活激活内存供给**（flexible activation provisioning）的大规模 MoE 训练系统，将内存分配从静态配置转变为运行时连续控制。

#### 核心技术
1. **Activation Memory Modeling (SIV)**  
   设计了一个统一的激活内存成本模型，能够准确预测每个 micro-batch 的激活内存消耗及缓解策略的时间开销，为动态决策提供依据。

2. **Adaptive Activation Provisioning (SV-A)**  
   在运行时根据当前 pipeline 状态和 MoE 负载不平衡程度，为每个 micro-batch 动态分配内存预算，并选择最优的缓解策略组合（recomputation、offloading、quantization），实现“按需供给”。

3. **Fused Dual-Batch Execution (SV-B)**  
   通过交错执行相邻 micro-batch 的计算与通信操作，将缓解策略引入的额外开销隐藏在硬件空泡（hardware bubble）中，避免其影响关键路径。

### 相比现有方法的优势
- **消除 OOM 与低 MFU 的权衡困境**：不再依赖静态策略，而是动态适应负载变化。
- **高吞吐 + 高稳定性**：在正常迭代中保持高 MFU，在负载峰值时自动增强缓解策略防止崩溃。
- **零人工干预**：无需因 OOM 手动重启或切换配置，提升训练鲁棒性。

---

## 2. 核心实验方法和设置

### 使用的模型与硬件
- **模型**：基于 GLM 系列设计的一组 MoE 模型，参数量从 188B 到 **1041B** 不等，最大激活参数达 35B。
- **硬件平台**：包含 **12,288 GPUs** 的集群（每节点 8 GPU，共 1536 节点），单卡 64GB HBM 显存，BF16 峰值算力 480 TFLOPS。
- **并行策略**：采用自动搜索得到的混合并行方案（Hybrid Parallelism），包括 TP、EP、PP 等，PP degree 最高达 8。

### 实验设置与评估指标
| 类别 | 内容 |
|------|------|
| **训练任务** | 大规模 MoE 模型预训练 |
| **输入序列长度** | 最高至 16,384 tokens |
| **评估指标** | - 总训练吞吐量（TFLOPS / EFLOPS）<br>- 每 GPU 的 **MFU**（Model FLOPs Utilization）<br>- 端到端训练速度（含重启开销）<br>- 吞吐提升倍数（speedup） |

### 基线方法对比
- **Megatron-Kuai [11]**：支持细粒度 activation rematerialization 和 offloading，但内存策略静态。
- **AdaPipe [10]**：支持 adaptive recomputation 和 pipeline 分区，但未针对 MoE 动态性优化。

> 两种基线均采用“先 under-provision → OOM 后切换为 over-provision”的人工干预流程模拟实际部署场景。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **12,288 GPUs** 上成功训练 **1041B 参数**的 MoE 模型。
- 实现持续 **>2 EFLOPS** 的计算吞吐。
- 平均 **MFU 达到 35%**，显著高于现有系统。

### 与基线方法的对比结果
| 对比项 | 结果 |
|--------|------|
| **平均吞吐提升** | - 相比 Megatron-Kuai：**1.14×**<br>- 相比 AdaPipe：**1.13×** |
| **在 512 GPU 上的 spike 处理能力** | - 基线 MFU 下降 18–21%<br>- BaGuaLu-2 维持稳定 MFU，spike 阶段相对提速 **1.28×** |
| **跨配置鲁棒性** | 在不同 MoE sparsity 和 sequence length 下均优于基线，尤其在高压场景优势更明显 |

### 消融实验结果（Ablation Study）
逐步启用各组件的效果如下（以 MFU 提升衡量）：

| 阶段 | 优化内容 | 相对前一阶段提升 | 累计 MFU 提升 |
|------|--------|------------------|---------------|
| L1 → L2 | 引入可组合的细粒度缓解策略 | 1.16× | 1.16× |
| L2 → L3 | 加入动态内存缓解（SV-A） | 1.05× | 1.22× |
| L3 → L4 | 加入双 batch 融合调度（SV-B） | 1.13× | **1.38×** |

> 表明三项技术具有**正向叠加效应**，共同构成完整优化闭环。

### 其他关键数据
- **内存预测误差**：平均仅 **1.56%**，验证了建模准确性。
- **运行时决策开销**：<10ms/iteration，远小于单次迭代耗时（>18s），几乎无额外负担。
- **缓存命中率高**：得益于 MoE 负载的时间局部性，多数决策可直接查表复用。

---

## 4. 关键结论和发现

### 主要发现
1. **MoE 内存波动本质是结构性问题**：由 fine-grained routing 的随机性和输入 token 分布偏斜共同引起，无法通过改进路由算法完全消除。
2. **静态内存管理已不可持续**：面对稀发但剧烈的内存 spike（实测约 2.9% 迭代出现，峰值达 2.6×），静态策略必然牺牲效率或稳定性。
3. **灵活供给 + 开销隐藏 是可行路径**：BaGuaLu-2 证明可在不牺牲性能的前提下实现动态弹性内存管理。
4. **系统级协同优化至关重要**：仅靠单一技术（如 recompute）不足以解决问题，必须结合建模、调度、执行层联合设计。

### 方法的局限性
- 当前假设 pipeline-parallel degree ≥ 4，以保证 dual-batch scheduling 的正确性；极浅流水线可能受限。
- 缓解策略的选择依赖于精确的成本建模，若硬件特性突变（如新型 GPU 架构），需重新校准模型。
- 尚未扩展至异构设备或跨数据中心训练场景。

### 未来工作方向
- 将灵活供给思想推广至 **optimizer state** 和 **gradient** 管理。
- 探索 **learning-based runtime controller** 替代规则式调度器。
- 支持更多类型的缓解原语（如 compression、sparsity-aware checkpointing）。
- 向 **推理阶段** 延伸，应对 serving 场景下的动态流量冲击。

--- 

> ✅ **总结一句话**：  
> BaGuaLu-2 通过 **runtime-level flexible activation provisioning**，首次实现了在万卡级别上高效且稳定的 MoE 模型训练，打破了传统静态内存管理的性能天花板。

</details>

---

### 8. [DLite: History-free Speculative Decoding for Agentic AI](https://arxiv.org/abs/2610.23787)

**Authors**: Jiaming Xu, Hanzhen Wang, Tianlang Zhao, Jiewen Xiao, Chengze Yuan, Jiayi Pan, Yongkang Zhou, Jiancai Ye, Yu Wang, Guohao Dai  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.23787v1  

#### Abstract
Speculative decoding has become a mainstream technique for accelerating large language model (LLM) inference by using a lightweight draft model to generate candidate tokens that are subsequently verified by the target model. However, agentic workloads introduce long and continuously evolving context...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DLite: History-free Speculative Decoding for Agentic AI

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在 **Agentic AI** 工作负载下，传统的 **Speculative Decoding** 方法面临两大挑战：

1. **接受长度（acceptance length）严重下降**  
   在长上下文、多轮交互、持续增长的历史背景下，轻量级 draft model 难以准确建模历史信息，导致与目标模型预测分布不一致，acceptance length 下降高达 **80%**。

2. **系统开销显著增加**  
   draft model 需要维护不断增长的 **historical KV cache**，频繁更新和前缀缓存带来额外延迟，占端到端推理时间的 **10%~20%**。

这些问题的根本原因在于：**draft model 被迫重复建模目标模型已经处理过的上下文信息**。

---

### 🚀 提出的新方法与创新思路

作者提出 **DLite**，一种 **history-free speculative decoding 框架**，其核心思想是：

> **不再让 draft model 独立维护历史上下文，而是直接复用 target model 输出的 hidden states 进行局部预测**。

#### 主要创新点：

- **History-Free 并行骨干网络（Parallel Backbone）**  
  不再显式维护 draft model 的历史 KV cache，而是从 target model 获取 compact hidden states，并结合当前 anchor token 和局部位置信息进行 **local speculative prediction**。

- **轻量级序列头（Lightweight Sequential Head）**  
  在 hidden space 中恢复 speculative block 内部的因果依赖（intra-block causal dependency），避免昂贵的中间词汇投影（intermediate vocabulary projection）。

- **两阶段训练策略（Two-stage Training Strategy）**  
  1. **Warm-up 阶段**：先训练一个结构同构但有历史感知能力的“教师模型”作为中间目标；
  2. **Stable 阶段**：将 DLite 直接对齐到目标模型，提升分布一致性。

---

### 🔍 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **算法性能** | 在 agentic workload 上 acceptance length 提升至 **2~6**，相比 DSpark 最高提升 **70%** |
| **系统效率** | 完全消除 draft model 的历史 KV cache 维护开销 |
| **架构设计** | 将复杂的历史建模任务交给 target model，draft model 专注局部预测，降低建模难度 |
| **训练稳定性** | 通过两阶段训练缓解结构异构带来的优化困难 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

实验覆盖三类典型 workload：

| 类别 | 数据集 |
|------|-------|
| **通用任务** | GSM8K, MATH-500, HumanEval, MBPP, LiveCodeBench, MT-Bench, Alpaca |
| **长上下文任务** | LongBench v2（Multi-Document QA, Long In-Context Learning, Code Repo Understanding, Long-dialog History） |
| **Agentic 任务** | SWE-Bench, AgentLongBench, AgentMemBench, BFCL-v3 |

训练数据来自：
- **Open-PerfectBlend**（1.3M 样本）
- **NVIDIA Nemotron Post-Training Dataset V2**（600K 样本）

仅使用 prompt，响应由 target model 生成，确保监督信号一致。

---

### ⚙️ 实验设置与评估指标

| 设置项 | 配置 |
|--------|------|
| **Target Models** | Qwen3-4B, Qwen3-8B |
| **Speculative Block Size (K)** | 7 |
| **Target Hidden States Used** | Qwen3-4B: 12 层；Qwen3-8B: 14 层（sequence-wise 拼接） |
| **Sequential Head Rank** | 320 (4B), 512 (8B) |
| **硬件平台** | NVIDIA H800 GPU |

#### 评估指标：

- **Acceptance Length (T)**：每轮验证平均接受的 token 数
- **End-to-End Speedup**：相对于 vanilla autoregressive decoding 的加速比
- **Throughput (TPS)**：高并发下的吞吐量（tokens/sec）
- **消融实验**：分析训练策略、KV cache、架构设计的影响

---

### 🆚 基线方法对比

| 方法 | 类型 | 特点 |
|------|------|------|
| **EAGLE-3** | Autoregressive | 显式保持 speculative token 因果关系，但需串行生成 |
| **DFlash / DFlash2** | Parallel | 单次前向传播生成所有候选 token，速度快但忽略块内依赖 |
| **DSpark** | Semi-autoregressive | 并行 backbone + 轻量序列模块，当前 SOTA |
| **DFlow** | Block Diffusion-based | 利用 verifier 信息流优化生成 |
| **Vanilla AR** | Baseline | 标准自回归解码 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### 表 1：通用任务上的 Acceptance Length（部分）

| 方法 | GSM8K (4B) | HumanEval (4B) | MT-Bench (4B) |
|------|------------|----------------|---------------|
| DSpark | 6.27 | 5.48 | 3.81 |
| **DLite** | **6.38** | **5.64** | **3.94** |

✅ DLite 在通用任务上 **持平或略优于 SOTA**，说明去历史化未牺牲基础性能。

---

#### 表 2：长上下文任务上的 Speedup 与 T（LongBench_v2）

| 方法 | Multi-Doc QA (4B) Speedup/T |
|------|-------------------------------|
| DSpark | 1.81× / 2.20 |
| **DLite** | **2.18× / 2.74** |

➡️ 在长上下文任务中，DLite 明显领先，证明其对长历史场景更鲁棒。

---

#### 表 3：Agentic 任务上的性能飞跃（SWE-Bench）

| 方法 | Speedup (4B) | T (4B) |
|------|--------------|--------|
| DFlash | 1.47× | 1.72 |
| DSpark | 1.64× | 1.89 |
| **DLite** | **2.68×** | **3.15** |

🔥 在 SWE-Bench 上，DLite 实现 **约 2× 端到端加速**，acceptance length 提升 **超 60%**。

---

#### 表 4：高并发吞吐量（Concurrency=64, LCB 任务）

| 方法 | TPS (4B) | Speedup | Avg T |
|------|---------|---------|-------|
| DSpark | 11309 | 1.72× | 5.28 |
| **DLite** | **12075** | **1.83×** | **5.98** |

💡 即使 acceptance length 接近，DLite 吞吐更高 → **轻量 sequential head 减少了中间计算开销**

---

### 🔍 消融实验结果

#### （1）训练策略对比（图5a）

| 训练方式 | Final Acceptance Length |
|----------|------------------------|
| 单阶段（Direct KL） | 较低且不稳定 |
| **两阶段（Warm-up + Stable）** | **显著更高，收敛更稳** |

✅ 两阶段策略有效桥接了结构差异，提升最终性能。

---

#### （2）KV Cache 消融（图5b）

| 变体 | Acceptance Length |
|------|-------------------|
| 无 KV Cache（DLite） | 与 baseline 相当 |
| 滑动窗口 128/512 | 训练初期略优，后期趋同 |

➡️ **历史 KV cache 对最终性能非必需**，充分训练后可通过 target hidden states 补偿。

---

#### （3）设计空间探索（附录 A.2）

| 设计选择 | 最佳配置 | 发现 |
|----------|----------|------|
| Target Hidden States 数量 | 14 层 | 多于 14 层收益饱和 |
| Hidden State 拼接方式 | **Sequence-wise** > Feature-wise | 更利于区分层来源 |
| Sequential Head Rank | **512 (1/8D)** | 性能/开销最佳平衡点 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **历史上下文建模是 speculative decoding 在 agentic 场景下的瓶颈**  
   轻量 draft model 难以独立维持长历史，导致 accept rate 下降和系统开销上升。

2. **复用 target model 的 hidden states 是高效替代方案**  
   可将复杂的全局建模任务卸载给 target model，draft model 仅需完成局部预测。

3. **DLite 实现了算法与系统的双重优化**  
   - 算法上：acceptance length 提升 2~6，在 agentic 任务上超越 DSpark 最高 **70%**
   - 系统上：完全消除 draft KV cache 更新开销，提升高并发吞吐

4. **两阶段训练策略解决了结构异构的优化难题**  
   引入 isomorphic teacher 作为过渡，显著提升训练稳定性和最终性能。

---

### ⚠️ 方法的局限性

- **依赖 target model 的 hidden states 输出接口**  
  需要对 target model 进行一定程度的访问或修改，可能限制在某些闭源模型上的应用。

- **对 hidden states 的选择与组合敏感**  
  虽然 sequence-wise 拼接表现更好，但最优层数仍需经验调参。

- **目前仅适配 Qwen 系列模型**  
  架构泛化性有待在更多模型族（如 Llama、Mixtral）上验证。

---

### 🔮 未来工作方向

1. **扩展至更多模型架构**  
   探索 DLite 在 MoE、decoder-only、encoder-decoder 架构中的适用性。

2. **动态 speculative block size 控制**  
   根据上下文复杂度自适应调整 K，进一步提升效率。

3. **与硬件协同设计（Co-design）**  
   结合 DLite 的 low-overhead 特性，设计专用 inference engine 或 NPU 指令集。

4. **支持 streaming & online learning 场景**  
   在持续演化的 agent 执行过程中实现在线微调与适应。

---

> ✅ **代码与模型已开源**：  
> GitHub: [https://github.com/SJTU-DAI-Sys/DLite](https://github.com/SJTU-DAI-Sys/DLite)  
> HuggingFace: [https://huggingface.co/collections/SJTU-DAI-Sys/dlite](https://huggingface.co/collections/SJTU-DAI-Sys/dlite)

</details>

---

### 9. [Accelerated Dataflow: scaling inference in streaming data analytics with efficiency](https://arxiv.org/abs/2610.22731)

**Authors**: RuiLong Jiang, Dustin Rhodes, Danny McCormick, Yuta Labur, Reza Rokni  
**Category**: cs.DC  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.22731v1  

#### Abstract
Streaming data analytics helps businesses identify opportunities and spot risks in a timely fashion. As large language model (LLM) usage grows, people are exploring incorporating agents or LLMs in streaming data pipelines to help find patterns in datasets without supervision or interpret natural lan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Accelerated Dataflow: Scaling Inference in Streaming Data Analytics with Efficiency*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

当前主流的 **streaming data processing systems**（如 Flink、Dataflow）在设计时主要优化的是 **CPU 密集型的数据移动任务**，而对 **GPU-intensive 的机器学习推理任务**（尤其是 LLM 推理）支持不佳。将 LLM 集成到流处理管道中面临以下挑战：

- **GPU 资源利用率低**：传统 autoscaling 基于 CPU 利用率，无法准确反映 GPU（特别是 VRAM）的实际负载。
- **请求局部性缺失**：现代 LLM 推理依赖 **prefix caching** 来提升吞吐，但随机分发请求会破坏缓存命中率。
- **长处理时间导致超时**：LLM 推理耗时远高于传统消息，容易触发 lease timeout 和任务重试，造成资源浪费。
- **异构资源调度僵化**：CPU 和 GPU 工作负载混合时，统一 worker 池难以高效利用资源。

### ✅ 提出的新方法与创新点

作者在 **Google Cloud Dataflow** 上提出了一套面向 **hybrid streaming ML workloads** 的系统级优化方案，核心创新包括：

#### （1）**Co-designed Autoscaling（协同设计的自动扩缩容）**
- 不再依赖 CPU 利用率，而是通过 **inference engine 报告的 request completion rate** 动态估算每个 worker 的真实处理能力。
- 引入公式 $ N_m = \frac{K_{m,\text{active}} + K_{m,\text{dispatchable}}}{\lambda \cdot P_m} $，基于活跃 key 数量和每 worker 吞吐率计算目标 worker 数量。
- 支持快速响应流量波动，避免传统算法因 backlog 积压滞后扩容。

#### （2）**Heterogeneous Worker Pools（异构 worker 池）**
- 支持为不同 transform 指定 `resource hints`（如 `accelerator='nvidia-tesla-t4'`），实现 CPU 和 GPU 工作负载分离。
- GPU worker 仅用于运行 `RunInference` transform，避免在非推理阶段浪费昂贵 GPU 资源。
- 各 worker pool 独立扩缩容，提升整体资源利用率。

#### （3）**Prefix-aware Batching（前缀感知批处理）**
- 利用 **group-by-key** 机制，将语义相似的请求路由到同一 worker，显著提高 **KV Cache Hit Rate**。
- 用户可提供 prompt 类别等先验知识，系统据此进行智能分组，最大化缓存复用。
- 结合 key sharding 平衡缓存效率与并行度。

#### （4）**AsyncDoFn 与 Embedded Inference Engine**
- 引入 `AsyncDoFn` 支持异步处理，防止长任务被误判为 straggler 而中断。
- 在 worker 内嵌高性能推理引擎（如 **vLLM**, **Triton**），实现端到端低延迟推理。
- 分离推理引擎内部并行管理与 streaming runtime 的资源调度。

### ✅ 相比现有方法的优势

| 维度 | 传统系统（Flink / Legacy Dataflow） | Accelerated Dataflow |
|------|-------------------------------------|------------------------|
| 扩缩容依据 | CPU 利用率、backlog（秒） | 实际请求完成率、key 状态 |
| GPU 支持 | 弱，依赖通用 slot 管理 | 强，专用 GPU worker 池 + VRAM 意识 |
| 缓存友好性 | 随机分发，缓存命中率低 | 前缀感知分组，高缓存命中 |
| 异构调度 | 固定 parallelism，资源错配 | 动态 rightfitting，按需分配 |
| 易用性 | 需手动调参、部署外部服务 | 开箱即用，SDK 级集成 |

---

## 2. **核心实验方法和设置**

### ✅ 使用的数据集

- **OpenOrca**：开源多轮对话数据集，具有自然的 system prompt 分组。
- **Dolly**：指令微调数据集，包含明确的任务类别标签，适合演示 prefix-aware batching 效果。
- **TimesFM 500M**：时间序列预测模型，用于测试传统 DNN 推理场景。

### ✅ 实验设置

#### 模型配置
- **LLM**：Qwen/Qwen2.5-7B-Instruct-AWQ，部署在 **vLLM** 上，max context 4096。
- **DNN**：TimesFM 500M，部署在 **Triton Inference Server** 上。

#### 硬件环境
| 参数 | 规格 |
|------|------|
| 实例类型（LLM） | a2-highgpu-1g / n1-standard-4 |
| GPU | 1× A100 / 1× T4 |
| vCPU | 12 / 4 |
| 内存 | 84GB / 30GB |
| 最大 worker 数 | 8–100（视实验而定） |

#### 流量模式
- **Ramp-up 测试**：逐步增加输入速率，测量吞吐与延迟关系。
- **Sinusoidal Load**：周期为 2 小时的正弦波流量，模拟真实世界昼夜变化，峰值达基值 10x 或 30x。

#### 数据源与汇
- Source: **PubSub**
- Sink: **BigQuery**

### ✅ 评估指标

| 指标 | 描述 |
|------|------|
| **Throughput (req/s)** | 每秒成功处理的请求数 |
| **Latency / Tail Latency** | 端到端延迟，重点关注 P99/P95 |
| **Watermark Latency** | 数据事件时间与处理时间的差距 |
| **GPU-Hour Consumption** | GPU 资源总消耗量 |
| **KV Cache Hit Rate** | 推理过程中 key-value 缓存命中比例 |
| **Peak Worker Count** | 扩容期间最大 worker 数量 |

### ✅ 基线方法对比

- **Baseline 1**: 原始 Dataflow（基于 CPU 的 autoscaling）
- **Baseline 2**: Apache Flink（使用 adaptive scheduler + 官方 autoscaler）
- **Baseline 配置**：
  - Flink 设置 `target.utilization=0.7`
  - 启用 multiple Python processes 以充分利用 CPU
  - 调整 bundle size/time 提升 IPC 性能

---

## 3. **主要实验结果和性能指标**

### ✅ 关键性能数据

#### （1）Prefix-aware Batching 效果（图3、图4）
- **Dolly 数据集**：
  - KV Cache Hit Rate 从 ~50% 提升至 **~80%**
  - 吞吐提升约 **1.5–2x**
  - 尾部延迟降低明显，尤其在高负载下
- **OpenOrca 数据集**：
  - 类似趋势，缓存命中率提升显著
  - 在相同延迟预算下支持更高请求率

#### （2）Heterogeneous Worker Pools + Co-designed Autoscaling（图5）
- **GPU-hour 消耗减少 51.1%**（37.0 vs 75.7 GPU-hours）
- **尾部延迟下降 84.9%**
- 更快响应流量高峰，避免“恐慌式扩容”

#### （3）纯 GPU 推理管道对比（图6）
- **GPU-hour 节省 42.4%**（27.7 vs 48.1h）
- **峰值延迟降低 71.0%**
- worker 分配更平稳，无剧烈震荡

#### （4）与 Flink 对比（图7、图8）
| 指标 | Accelerated Dataflow | Flink Baseline | 提升 |
|------|------------------------|----------------|-------|
| Peak Worker Allocation | 25 GPUs | >100 GPUs | ↓ **75%** |
| Peak Queueing Latency | 582 s | 3804 s | ↓ **84.9%** |
| Downscale Responsiveness | 快速回收空闲资源 | 长时间保持高位 | 显著更优 |
| 系统稳定性 | 无 tail latency spike | 多次 spike | 更稳定 |

#### （5）DNN 推理（TimesFM）表现（图8）
- 在高并发场景下仍能快速扩缩容
- Flink 因默认 batching 不合理出现读写瓶颈
- Dataflow 自动优化 IPC 与 batching，吞吐更高且延迟可控

### ✅ 消融实验结果

- **移除 heterogeneous worker pools**：
  - GPU 利用率下降，因 CPU 阶段成为瓶颈
  - 扩容不灵活，资源浪费严重
- **关闭 prefix-aware batching**：
  - KV Cache Hit Rate 下降 30–40%
  - 吞吐下降约 40%，延迟翻倍
- **使用 legacy autoscaling**：
  - 扩容滞后，导致 backlog 积压
  - 低估 per-worker capacity，引发“雪崩式”重试

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **集成式 ML 流处理是可行且高效的**  
   将 LLM/DNN 推理直接嵌入 streaming pipeline 可避免跨系统通信开销，同时保障 exactly-once 语义。

2. **传统 autoscaling 机制不适用于 ML 推理**  
   基于 CPU 的扩缩容严重误判实际负载，必须引入 **request-level signal reporting** 才能精准控制规模。

3. **请求局部性对 LLM 推理至关重要**  
   通过 **prefix-aware batching** 可大幅提升 KV Cache Hit Rate，进而显著改善吞吐与延迟。

4. **异构资源池分离带来巨大收益**  
   分离 CPU 与 GPU worker 池可独立扩缩容，避免资源错配，在成本敏感场景尤为关键。

5. **端到端优化胜过局部改进**  
   单独优化某一层（如推理引擎）效果有限，只有 **co-designed system stack** 才能释放全部潜力。

### ✅ 局限性

1. **Worker Startup 时间较长**  
   使用 custom container（含大型模型）导致 cold start 延迟高，影响 autoscaling 敏捷性。

2. **Inter-process Communication Overhead**  
   worker harness 与 inference engine 间通过 IPC 通信，存在不必要的内存拷贝。

3. **Key Skew 问题未完全解决**  
   虽然支持 sharding，但极端热点 key 仍可能导致负载不均。

4. **动态 autosharding 尚未实现**  
   当前需用户手动定义 sharding 策略，缺乏自动化支持。

5. **仅限 Google Cloud 生态**  
   当前方案深度绑定 Dataflow 与 GCP 服务，移植性受限。

### ✅ 未来工作方向

1. **Dynamic Autosharding**  
   自动分析输入分布，动态调整 key 分片策略，进一步提升缓存效率。

2. **Shared Memory IPC**  
   借鉴 Ray 的 in-memory object store，实现 worker 与 inference engine 间的零拷贝通信。

3. **Graceful Failover for Inference Engines**  
   支持推理引擎崩溃后的状态恢复，避免整个 worker 重启。

4. **Cross-model Batching Optimization**  
   在多模型共存场景下，智能调度不同模型请求以最大化 GPU 利用率。

5. **开放标准接口支持更多 backend**  
   使该架构可适配其他 streaming engine（如 Spark Structured Streaming）或 inference server。

---

> 📌 **总结一句话**：  
> *Accelerated Dataflow* 通过 **co-designed autoscaling + heterogeneous worker pools + prefix-aware batching**，实现了 LLM 推理与流处理的深度融合，在保持低延迟的同时，达成 **最高 2x 吞吐提升、75% 峰值资源节省、84% 尾部延迟下降**，为构建高效、可扩展的 streaming ML pipeline 提供了实用路径。

</details>

---

### 10. [DIFFLET: A Fast Inference Engine for Diffusion Transformer Models on AWS Trainium](https://arxiv.org/abs/2610.23047)

**Authors**: Bin Ma, Min Yu, Vijay Kodamalla, Jialin Liu, Dong Li  
**Category**: cs.DC  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.23047v1  

#### Abstract
On AWS Trainium2, every component of a diffusion model runs as a program compiled ahead of time for fixed shapes, while pipelines differ across models, prompt lengths across requests, and caching decisions across denoising steps. We present Difflet, a serving engine in which five model pipelines sha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DIFFLET: A Fast Inference Engine for Diffusion Transformer Models on AWS Trainium**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
在 AWS Trainium 上部署 Diffusion Transformer 模型面临三大挑战：
1. **模型多样性**：不同 Diffusion 模型具有不同的 encoder-decoder-denoiser 流程，且编译需针对固定形状进行，导致为每个模型单独构建执行路径成本高昂。
2. **并行执行复杂性**：如何在 Trainium 的 NeuronCore 上高效分布 denoiser 计算，尤其是在处理 joint attention（视觉+文本）时，需满足硬件对序列长度、mask 和通信模式的严格要求。
3. **动态缓存决策**：step caching 是加速推理的有效手段，但在 ahead-of-time (AOT) 编译环境下，无法直接访问中间特征来判断是否复用输出。

### **提出了什么新方法或新思路**
论文提出 **Difflet**，一个专为 AWS Trainium 设计的高性能 Diffusion Transformer 推理引擎，其核心创新包括：

- **统一的共享编译后端（Shared Compilation Backend）**  
  将所有模型共有的组件（如 encoder、denoiser、decoder）抽象为可复用的编译单元。通过一个轻量级“model adapter”将具体配置映射到通用后端，实现五种不同图像/视频模型共享同一套编译基础设施。

- **基于 attention head 的 Ulysses 式 context parallelism**  
  针对 joint attention 中文本 padding 导致 softmax 不兼容的问题，采用 Ulysses 方式按 attention head 切分序列，确保每个 core 上 query 和 key 数量一致，并保持 valid text keys 连续，从而支持任意 prompt 长度下的 fused attention kernel 执行。

- **设备端 probe 实现 adaptive step caching**  
  在设备上运行一个小规模 probe 程序（仅前几层），计算早期特征变化并返回单个 scalar score 给 host，由 host 决定是否跳过当前 denoising step。该设计避免了完整特征传输开销，同时保留了 high-precision reuse 信号。

- **驻留模型进程（Resident Model Process）**  
  模型程序和权重常驻内存，跨请求复用，消除每次请求重复加载和初始化的延迟。host 控制每步逻辑（如是否调用 denoiser），而 device 只执行预编译程序。

### **相比现有方法的优势**
| 特性 | Difflet | PyTorch/XLA / NxDI |
|------|--------|---------------------|
| 多模型支持 | ✅ 支持 5 个异构模型 | ❌ 各自独立流程 |
| 并行灵活性 | ✅ 支持 TP, SP, CP (Ulysses/Ring), CFG, DP | ⚠️ 通常只支持部分 |
| Attention 性能 | ✅ 使用 fused NKI kernel | ❌ 使用通用 SDPA |
| Step Caching | ✅ 支持带 probe 的 adaptive caching | ❌ 无或低效 |
| 请求延迟 | ✅ Resident 模式大幅降低 | ❌ 每次重新加载 |

---

## 2. **核心实验方法和设置**

### **使用的模型与任务**
论文在 **五个主流 Diffusion Transformer 模型** 上进行了验证，涵盖图像与视频生成：
- **FLUX.1-dev** (12B params)
- **Qwen-Image** (20B params)
- **Wan 2.1** (14B, 视频, 81 frames)
- **HunyuanVideo** (13B, 视频)
- **LTX-2** (19B)

> 注：未使用传统“数据集”，而是以标准生成任务（text-to-image/video）作为 workload。

### **实验设置**
- **硬件平台**：AWS `trn2.3xlarge` 实例（1×Trainium2 芯片，4 logical cores，96 GiB HBM）
- **软件栈**：`torch 2.9.1`, `torch-neuronx 2.9.0`, `neuronx-cc 2.26`, `NKI 0.5`
- **精度**：BF16（除 Wan 的 VAE 解码使用 FP32）
- **并行配置**：默认 TP=4，其他组合用于消融实验

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Request Latency** | 单个生成请求的端到端耗时（resident 模式下） |
| **Denoising Time** | 去噪循环总时间（含缓存步骤） |
| **Step Latency** | 单个去噪步的中位数耗时（排除首步） |
| **PSNR / LPIPS** | 图像/视频质量评估（vs. 无缓存基准） |
| **Throughput** | 单位时间内完成请求数 |
| **Compilation Time** | 首次编译耗时 |

### **基线方法对比**
| 基线系统 | 描述 |
|---------|------|
| **PyTorch/XLA** | 使用 XLA 编译的 SDPA attention，其余流程相同 |
| **NxDI** | AWS 官方提供的 FLUX 推理应用，仅支持单一模型 |
| **Standalone Run** | 每次请求启动新进程，重新加载权重 |
| **Warm Run** | 权重文件已缓存，但仍需加载至 device |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **端到端推理速度提升**
| Model | Resident vs Warm Standalone | Speedup |
|-------|-------------------------------|---------|
| FLUX.1-dev | 14.2s vs 47.6s | **3.35×** |
| Qwen-Image | 42.4s vs 98.9s | **2.33×** |
| HunyuanVideo | 59.9s vs 141s | **2.35×** |
| LTX-2 | 43.3s vs 64.7s | **1.50×** |
| Wan 2.1 | 1,245s vs 974s | **⚠️ 更慢（因 host decode）** |

> Wan 因 VAE 解码在 host 执行反而变慢，凸显模块化部署的重要性。

#### ✅ **Attention Kernel 加速效果**
| Model | NKI Attention vs SDPA | Speedup |
|-------|------------------------|---------|
| LTX-2 (2.3k tokens) | 414ms vs 514ms | **1.24×** |
| FLUX/Qwen/Hunyuan (~4.1k tokens) | 1.89–2.53× | — |
| HunyuanVideo (10.2k tokens) | 845ms vs 3,620ms | **4.28×** |
| Wan (32.8k tokens) | ✅ 可运行 | ❌ SDPA 编译失败 |

> **结论**：sequence length 越长，优势越显著；**32.8k tokens 下只有 fused NKI kernel 能编译成功**。

#### ✅ **消融实验：各机制贡献分解**
（以从 PyTorch/XLA 到 Difflet 的逐步优化为例）

| 机制 | 主要收益模型 | 加速比 |
|------|-------------|--------|
| **NKI Attention** | HunyuanVideo | 1.93× |
| **Presharded Weights** | LTX-2, Wan | 1.69× / 1.53× |
| **Resident Serving** | 图像模型 | 2.3–3.35× |
| **Step Caching (cadence 2)** | 主要去噪主导模型 | 1.33–1.64× |

> 综合优化带来 **3.0–8.5× 的请求延迟下降**。

#### ✅ **Step Caching 效果**
| 模式 | Skip Ratio | Speedup | Quality (PSNR Δ) |
|------|-----------|---------|------------------|
| Fixed Cadence 2 | ~40% | 1.6–1.7× | +7.4–11.6 dB vs “fewer steps” |
| Online-Delta | 动态（8–20 steps） | 1.25–1.6× | 最高达 37.3 dB |
| Calibrated Adaptive | 匹配 cadence budget | ≈ cadence time | 但预测不准时质量下降明显 |

> **重要发现**：caching 显著优于直接减少采样步数（fewer steps），在相同计算量下保真度更高。

---

## 4. **关键结论和发现**

### **主要发现**
1. **统一后端可行且高效**：通过抽象 encoder/denoiser/decoder 为可复用组件，Difflet 成功支持 **5 种异构 Diffusion 模型** 共享同一编译流程。
2. **Ulysses + fused attention 是长序列最优解**：在 Trainium 上，Ulysses context parallelism 结合 fused attention kernel 可实现 **接近原始精度的分布式 attention**，且支持任意 prompt 长度。
3. **设备端 probe 实现低成本 adaptive caching**：probe 仅增加 2–18ms 开销（<2% step time），即可提供有效 reuse 信号，尤其适用于视频等长序列场景。
4. **Resident serving 是低延迟关键**：消除重复加载使请求延迟下降 **1.5–3.4×**，尤其对小步数、高频请求场景至关重要。
5. **Kernel 选择决定上限**：在 32.8k tokens 场景下，**只有 fused NKI kernel 能编译成功**，证明专用 kernel 对超长序列不可或缺。

### **方法的局限性**
- **内存限制影响部署灵活性**：例如 Wan 的 chunked VAE 无法与 resident denoiser 共存于同一 chip，被迫移至 host 执行，导致性能倒退。
- **Probe 泛化能力有限**：calibrated adaptive caching 在 Wan 上 R² 仅为 13%，说明 early features 不一定能准确预测最终输出变化。
- **不支持动态 batch size**：受限于 AOT 编译，目前仅支持固定形状输入。
- **Compiler Host Memory 成瓶颈**：某些并行配置（如 TP2+CFG）因编译器耗尽 124GiB host memory 而无法构建。

### **未来工作方向**
- 支持更多并行组合（如 SP+CP）、动态 shape 编译
- 优化 probe 设计以提高预测准确性
- 实现跨 chip 分布式推理以支持更大模型
- 探索更智能的 caching policy（如 RL-based）
- 与上层调度系统集成（如 DiffServe、TridentServe）

---

> **一句话总结**：  
> **Difflet 通过“固定编译 + 动态控制”的分离架构，在 AWS Trainium 上实现了高效、灵活、可扩展的 Diffusion Transformer 推理服务，尤其在长序列和多模型场景下展现出显著性能优势。**

</details>

---

### 11. [Noise-shaped Sampling and Annealing for Scalable On-the-edge Training of Binary and Ternary AI Models](https://arxiv.org/abs/2610.22621)

**Authors**: Zihao Chen, Zhili Xiao, Rui-Jie Zhu, Yipeng Wang, Jason Eshraghian, Shantanu Chakrabartty  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.22621v1  

#### Abstract
The training loss of binary and ternary neural networks, expanded as a multilinear polynomial over the discrete weights, is structurally equivalent to the energy of a hierarchical high-order Ising model. We exploit this equivalence to apply Ising solvers to discrete-weight network training, while ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Noise-shaped Sampling and Annealing for Scalable On-the-edge Training of Binary and Ternary AI Models

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文致力于解决**在资源受限的边缘设备上高效训练二值化（Binary）和三值化（Ternary）AI模型**的问题。传统量化感知训练（Quantization-Aware Training, QAT）方法如 **Straight-Through Estimator (STE)** 虽然能在推理时实现低比特计算，但在训练过程中仍需维护全精度的“影子权重”（latent weights），导致训练内存开销大、更新频繁，难以在边缘设备上进行持续学习（continual learning）或微调。

此外，直接在离散参数空间优化面临损失函数高度非凸、存在大量局部极小值（local minima）和高阶挫败（high-order frustration）的问题，使得标准优化器（如SA）收敛慢且不稳定。

---

### **提出了什么新方法或新思路**

论文提出了一种名为 **2△-Sampler** 的新型优化器，其核心思想是将**随机采样与退火（sampling-and-annealing）** 作为噪声量化器嵌入到一个**一阶 ΔΣ 噪声整形反馈环路**中。

#### 主要创新点：

1. **理论等价性洞察**  
   首次明确指出：二值/三值神经网络的训练损失函数可以被精确分解为一个**分层高阶 Ising 模型的能量函数**。这为将 Ising 求解器的思想应用于离散权重训练提供了理论基础。

2. **ΔΣ 噪声整形机制**  
   利用 ΔΣ 反馈环路，将由权重量化、高阶挫败和 mini-batch 采样引入的**白噪声能量从低频适应带推向高频区域**，从而让离散权重更新能够沿着平滑的低频下降轨迹前进，避免陷入局部震荡。

3. **统一目标：从“找基态”到“泛化好”**  
   与传统 Ising 求解器追求全局最小值不同，2△-Sampler 通过温度退火和自适应阈值机制，在探索（exploration）与利用（exploitation）之间动态平衡，以提升模型的**泛化能力**为目标。

4. **完全离散训练架构**  
   整个训练过程在离散参数空间中进行，无需维护全精度权重或 AdamW 动量状态，显著降低了内存占用和更新流量。

---

### **相比现有方法的优势**

| 维度 | 2△-Sampler | STE / Full Precision |
|------|-----------|------------------------|
| **训练内存占用** | ↓ 50%+ | 高（需存储 weight, grad, m1, m2） |
| **权重更新内存流量** | ↓ 四个数量级（~10⁴×） | 高（每步全量写回） |
| **训练速度** | ↑ 一个数量级 | 慢（尤其对 QAT） |
| **是否支持 on-device training** | ✅ 是 | ❌ 否（内存瓶颈） |
| **硬件友好性** | ✅ 支持低比特原生执行 | ❌ 依赖浮点路径 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **CIFAR-10**：用于图像分类基准测试，验证基本性能。
- **ImageNet-1K**：大规模图像分类任务，测试可扩展性。
- **FineWeb-Edu**：用于语言模型预训练，测试在超大规模 LLM 上的表现。
- **Planted QUBO**：合成的二次无约束二值优化问题，已知全局最优解，用于验证算法动力学。

---

### **实验设置和评估指标**

#### 模型架构：
- **视觉模型**：VGG-7/VGG-8（CIFAR-10）、ResNet-18（ImageNet）
- **语言模型**：MatMul-Free LLMs（基于 BitNet 架构），规模达 2.7B 参数
- 所有离散层均使用 `2A_Conv2D` 或 `2A_Linear` 替代标准层

#### 评估指标：
- **准确率**：Top-1 和 Top-5 准确率（vision）、验证损失（language）
- **训练效率**：
  - 每 epoch 时间（s/epoch）
  - 总训练时间（time to solution）
  - 内存占用（MB/GB）
  - 权重更新 DRAM 事务数（DRAM transactions）
- **持续学习表现**：在 class-incremental 设置下对旧类别的保留能力

#### 基线方法对比：
- **Full Precision (FP)**：标准 AdamW 训练
- **Ternary Weight Networks (TWN)**：经典三值网络方法
- **Straight-Through Estimator (STE)**：主流 QAT 方法
- **Simulated Annealing (SA)**：单翻转（Single-Flip）、并行翻转（Parallel-Flip）

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 模型 | 方法 | Top-1 Acc (%) | Top-5 Acc (%) | 与基线差距 |
|------|------|---------------|---------------|------------|
| ResNet-18 (ImageNet) | 2△-Sampler | 61.65 | 83.81 | 仅落后 TWN 0.15% (Top-5) |
| VGG-8 (CIFAR-10) | 2△-Sampler (discrete) | 92.20 | — | 超越 TWN (91.47%) |
| MMFree-2.7B (LLM) | 2△-Sampler | Tail-100 Loss: 5.9846 | — | 显著优于 STE |

> 在 CIFAR-10 上，13M 参数的三值模型达到 **92.20%** 准确率，接近全精度基线（92.88%），仅差 **0.68%**。

---

### **与基线方法的对比结果**

| 对比项 | 结果 |
|-------|------|
| **vs. STE / FP** | 训练内存减少 **>50%**，权重更新内存流量减少 **四个数量级**（10⁴×） |
| **vs. TWN** | 在相同设置下，2△-Sampler 训练速度快 **18.5×**（TWN: 247.8s/epoch vs 2△: 13.4s/epoch） |
| **vs. SA** | 收敛速度显著更快，且能稳定处理大规模并行更新，而 Parallel SA 容易发散 |
| **vs. STE (continued pre-training)** | 在从同一量化检查点继续预训练时，2△-Sampler **持续优于 STE**，说明其能更有效识别离散域中的下降方向 |

---

### **消融实验结果**

1. **ΔΣ 噪声整形有效性（Fig. 2g）**  
   - 2△-Sampler 的决策信号残差功率谱呈 **V形**，低频段平坦，高频段陡升，符合 ΔΣ 噪声转移函数特征。
   - 对比之下，Parallel SA 的误差集中在奈奎斯特频率，表现为周期性振荡，无法收敛。

2. **自适应阈值机制（Fig. 3c, 3g）**  
   - 参数翻转间隔（inter-flip interval）随训练进程**系统性增长**，表明更新预算自我调节。
   - 在持续学习阶段，新数据到来后翻转活动重新激活，证明其具备**在线适应能力**。

3. **硬件效率分析（Table 5, Fig. 4d）**
   - **静态内存节省**：ResNet-18 节省 **55.8%**，2.7B LLM 节省 **58.7%**
   - **动态内存流量**：在训练后期，2△-Sampler 的 DRAM 事务数比全精度训练低 **~24,600×**

---

## 4. 关键结论和发现

### **主要发现**

1. **ΔΣ 噪声整形能有效提取离散损失景观中的平滑下降方向**，使优化器避开高阶挫败带来的高频震荡。
2. **2△-Sampler 实现了稳定性与可扩展性的统一**：兼具单翻转 SA 的稳定性与并行 SA 的可扩展性。
3. **完全离散训练是可行且高效的**：无需全精度影子权重即可实现高质量训练，极大降低边缘设备训练门槛。
4. **更新行为具有自适应正则化特性**：活跃参数集中在梯度陡峭区域，随训练自动“结晶”，形成类似自适应学习率的效果。

---

### **方法的局限性**

1. **缺乏严格的收敛保证**：由于采用并行翻转机制，无法继承经典 Simulated Annealing 的渐近收敛性。
2. **性能仍略低于全精度模型**：在 ImageNet 上 Top-1 准确率仍有约 **3.75%** 差距。
3. **超参数对规模敏感**：例如“更新率上限”（update-rate cap）需根据模型大小调整，不能跨尺度通用。
4. **反向传播仍依赖连续梯度**：虽然前向是离散的，但反向传播仍使用 STE 生成的连续梯度，未完全摆脱浮点计算。

---

### **未来工作方向**

1. **完全离散化的训练流程**：
   - 探索使用 **Direct Feedback Alignment (DFA)** 或 **SPSA** 等免反向传播梯度估计器，进一步降低对连续梯度的依赖。
   - 实现真正的“全低比特训练闭环”。

2. **硬件协同设计**：
   - 将 2△-Sampler 的采样器/积分器映射到 **in-memory computing** 或 **neuromorphic hardware** 中，利用物理随机性和模拟记忆单元实现原生高效执行。

3. **扩展至更高位宽和其他离散结构**：
   - 探索是否可推广至 2-bit、4-bit 甚至稀疏+离散混合结构。

4. **理论分析深化**：
   - 建立 2△-Sampler 在非凸离散优化中的收敛性或泛化误差界。

--- 

> **总结**：2△-Sampler 提出了一种全新的视角——将神经网络训练视为一个受控的 Ising 系统演化过程，并通过 ΔΣ 噪声整形技术实现了高效、稳定的离散域优化。它不仅在性能上媲美主流方法，更在**内存效率和硬件友好性**上取得突破，为 **on-device training** 和 **edge intelligence** 开辟了新路径。

</details>

---

### 12. [Pearl-Chat: Engineering a native Luganda language model from first principles in JAX and Flax NNX](https://arxiv.org/abs/2610.22529)

**Authors**: Wesley Kambale, Roland Sankara  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.22529v1  

#### Abstract
Autoregressive foundation language models predominantly target high-resource Indo-European languages, imposing computational and morphological penalties on under-represented African languages. Generic multilingual tokenizers routinely exceed fertility rates of 3.2 tokens per word on agglutinative Ba...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Pearl-Chat: Engineering a native Luganda language model from first principles in JAX and Flax NNX*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前主流的 **autoregressive foundation language models** 主要针对高资源的印欧语系语言（如英语），对非洲低资源语言（如卢干达语 Luganda）存在严重的“**tokenizer tax**”问题：
- 多语言通用 tokenizer（如 LLaMA-3、Gemma）在处理黏着性（agglutinative）的班图语时，**token fertility**（每词分裂数）高达 3.2–3.9，导致上下文效率低下、推理延迟高、语义表示退化。
- 此外，主流框架（如 PyTorch）依赖隐式状态管理，不利于可复现性和分布式训练。

### 提出的新方法与创新点
Pearl-Chat 是一个从数学第一性原理构建的、专为 **Luganda** 设计的轻量级因果解码器模型（~8.5M 参数），其核心创新包括：

| 创新点 | 描述 |
|--------|------|
| **Linguistically aligned native tokenizer** | 提出自训练的 **byte-level BPE** 分词器（V=8,192），专为 Luganda 的形态学特性优化，实现 **4.88 字符/Token** 的压缩率，显著降低 fertility（仅 1.60）。 |
| **JAX + Flax NNX 架构设计** | 完全基于 **JAX** 和 **Flax NNX** 实现，避免黑盒抽象，采用 `nnx.split` 和 `nnx.merge` 显式管理函数式神经网络状态，提升可调试性与分布式兼容性。 |
| **Deterministic resumable data streaming** | 使用 **Google Grain** 实现确定性、可恢复的数据流，支持训练中断后无缝恢复，保证严格的数据一致性。 |
| **Two-stage curriculum training** | 两阶段训练：<br>1. **Stage 1**: 在平行、圣经、百科等文本上进行基础预训练（59,501 条序列）；<br>2. **Stage 2**: 在 510 轮对话上进行指令微调，并引入 **assistant-only loss masking**，只计算助手回复部分的损失，防止模型重复用户输入。 |
| **SPMD tensor parallelism** | 使用 `jax.sharding.Mesh` 实现声明式 **SPMD** 张量并行，支持跨设备高效扩展。 |

### 相比现有方法的优势
- **更高的压缩效率**：相比 LLaMA-3（128K）、Gemma（256K）等前沿多语言 tokenizer，**压缩率提升超过 2 倍**。
- **更低的推理成本**：序列长度减少一半以上，显著降低内存占用与延迟。
- **更强的语言特异性建模能力**：保留了 Luganda 的 **noun-class concord** 和 **agglutinative morphology** 结构。
- **完全开源与可复现**：所有权重、tokenizer、运行时环境均公开，推动非洲语言模型研究的开放科学标准。

---

## 2. 核心实验方法和设置

### 数据集组成
共使用五类数据源，总计约 **60,017 条记录**，涵盖多种文体：

| 数据集 | 来源 | 类型 | 记录数 | 领域 |
|--------|------|------|--------|------|
| Luganda-English Parallel Corpus | Hugging Face (`kambale/luganda-english-parallel-corpus`) | 平行句对 | 24,973 | 新闻、教育、公共卫生 |
| Luganda Bible Corpus | 自建 | 圣经经文 | 30,844 | 经典班图语法 |
| Luganda Wikipedia | Wikimedia | 百科文章 | 350 | 地理、历史、文化 |
| Sunbird SALT MT | Sunbird AI | 多语言平行句对 | 300 | 对话翻译 |
| Curated Conversational SFT | 手工标注 | 多轮对话 | 510 | 礼貌用语、日常问答 |

> ✅ 所有数据经过 NFKC Unicode 归一化，通过 SHA-256 哈希确保无数据泄露。

### 实验设置
- **模型架构**：GPT-2 风格的 decoder-only transformer
  - 层数 $L=8$
  - 隐藏维度 $d=256$
  - 注意力头数 $h=8$
  - 上下文长度 $T=256$
  - 使用 **Pre-Layer Normalization** 和 **GELU** 激活
  - 输出头与嵌入层 **weight tying**
- **训练流程**：
  - 总步数：4,000 步
    - Stage 1（预训练）：3,000 步
    - Stage 2（SFT）：1,000 步
  - 优化器：**Optax AdamW**，学习率 warmup 后 cosine decay（峰值 $6\times10^{-4}$）
  - 梯度裁剪：global norm ≤ 1.0
  - 批大小：16
- **硬件平台**：NVIDIA Tesla T4 GPU（单卡）

### 评估指标
- **定量指标**：
  - **Validation Loss**（交叉熵）
  - **Perplexity**（$e^{\text{loss}}$）
  - **ROUGE-L F1**
  - **BLEU-4**
  - **Exact Match Rate**
- **定性分析**：
  - 三层次诊断基准（Diagnostic Benchmark）：
    1. **Tier 1**: 训练集内固定问候语记忆测试（Memorization）
    2. **Tier 2**: 未见对话回合验证（In-domain validation）
    3. **Tier 3**: 复杂泛化查询（Out-of-domain generalization）
- **消融实验**：评估不同组件的影响（如是否预训练、是否掩码、vocab size 等）

### 基线方法对比
- **Tokenizer 对比**：
  - Gemma BPE（256K）
  - GPT-4o BPE（200K）
  - LLaMA-3 BPE（128K）
  - mBERT WordPiece（119K）
  - 字符级 baseline（256）
- **模型架构对比**：
  - 不同 vocab size（2K, 4K, 8K, 16K）
  - 是否启用 Stage 1 预训练
  - 是否使用 assistant loss masking
  - Post-LN vs Pre-LN

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **最终 validation loss** | 0.1818 |
| **最终 perplexity** | 1.20 |
| **训练总耗时** | 227.15 秒（Tesla T4） |
| **吞吐量（GPU）** | 18,087 tokens/sec |
| **吞吐量（CPU）** | 3,922 tokens/sec |
| **初始 perplexity** | 12,958.98 → 最终下降 **9.2877 nats** |

### 与基线 tokenizer 的对比（Table 2）
在标准 Luganda 句子（78 字符，10 词）上的表现：

| Tokenizer | Vocab Size | 压缩比 (chars/tok) | Fertility (tok/word) | 总 Token 数 |
|----------|------------|---------------------|------------------------|-------------|
| **Pearl-Chat BPE (Ours)** | 8,192 | **4.88x** | **1.60** | **16** |
| Gemma BPE | 256,000 | 2.52x | 3.10 | 31 |
| GPT-4o BPE | 200,000 | 2.44x | 3.20 | 32 |
| LLaMA-3 BPE | 128,256 | 2.29x | 3.40 | 34 |
| mBERT WP | 119,547 | 2.00x | 3.90 | 39 |
| 字符级 | 256 | 1.00x | 7.80 | 78 |

> ✅ Pearl-Chat 的 token 数仅为 LLaMA-3 的 **47%**，压缩效率提升超 **2 倍**。

### 三层次诊断评估结果（Table 7）
| 评估层级 | Exact Match | ROUGE-L F1 | BLEU-4 |
|---------|-------------|------------|--------|
| **Tier 1**: Memorization | 100.0% | 1.0000 | 1.0000 |
| **Tier 2**: In-domain validation | 92.0% | 0.9650 | 0.9420 |
| **Tier 3**: Out-of-domain generalization | 83.3% | **0.9792** | **0.9405** |

> 🎯 在复杂泛化任务中仍保持极高生成质量。

### 消融实验结果（Table 8）
| 配置 | Val Loss | Val PPL | Concord Accuracy | ROUGE-L |
|------|----------|---------|------------------|---------|
| **Full model (Ours)** | **0.1818** | **1.20** | **98.5%** | **0.9792** |
| - No Stage 1 pre-training | 2.9135 | 18.42 | 48.2% | 0.3410 |
| - No assistant masking | 0.8420 | 2.32 | 82.1% | 0.5215 |
| - Small vocab (V=2K) | 0.6120 | 1.84 | 89.4% | 0.6830 |
| - Large vocab (V=16K) | 0.7240 | 2.06 | 84.7% | 0.6310 |
| - Post-LN baseline | 1.4502 | 4.26 | 71.0% | 0.4850 |

> 🔍 发现：
> - **Stage 1 预训练至关重要**：无预训练时，concord 准确率暴跌至 48.2%，说明无法掌握名词类别一致关系。
> - **Assistant masking 提升响应质量**：避免模型机械重复用户输入。
> - **V=8,192 是最优平衡点**：更小则欠拟合，更大则参数浪费且易过拟合。

---

## 4. 关键结论和发现

### 主要发现
1. **专用 tokenizer 显著提升低资源语言建模效率**：自训练的 byte-level BPE 能有效捕捉 Luganda 的 **morpheme boundaries**，将 fertility 降至 1.60，远优于通用 tokenizer。
2. **两阶段课程学习是成功的关键**：先通过多样化文本建立语法基础，再通过对话微调实现自然交互，缺一不可。
3. **JAX + Flax NNX 提供工程优势**：显式的函数式状态管理（`nnx.split/merge`）使模型更透明、可调试、易于分布式部署。
4. **轻量级模型也能实现高质量生成**：仅 8.5M 参数即可达到 **ROUGE-L 0.9792** 和 **BLEU 0.9405**，证明小模型在特定语言任务中的可行性。
5. **文化敏感性可通过训练注入**：模型能正确使用 honorifics（如 “ssebo”, “nnyabo”），体现语言的社会规范。

### 方法的局限性
- **数据规模受限**：总训练 token 数仅 ~3.3M，远低于 compute-optimal scaling law 推荐的 ~170M，处于 **data-constrained regime**。
- **覆盖语言单一**：目前仅支持 Luganda，尚未扩展到其他班图语。
- **缺乏语音接口**：未集成 ASR/TTS，限制实际应用场景。
- **边缘部署需进一步优化**：虽已高效，但仍需量化（INT4）、推测解码（speculative decoding）以适配移动设备。

### 未来工作方向
1. **扩大模型规模**：构建 **Pearl-Large (125M)** 和 **Pearl-Bantu Multi (500M)**，覆盖更多东非语言（Runyankore、Lusoga、Lugisu 等）。
2. **数据增强策略**：
   - 利用国家广播（UBC）、议会记录扩充语料
   - 应用 **back-translation** 进行合成数据生成
3. **边缘优化技术**：
   - 实施 **4-bit INT4 量化**
   - 引入 **speculative decoding** 加速推理
4. **多模态扩展**：结合图像、语音模态，打造本地化 AI 助手。
5. **社区共建生态**：鼓励非洲开发者参与数据标注、模型训练，推动 **linguistic sovereignty**。

---

> ✅ **开源地址**：
> - GitHub: [https://github.com/wkambale/pearl-chat](https://github.com/wkambale/pearl-chat)
> - Hugging Face Hub: [https://huggingface.co/kambale/pearl-chat](https://huggingface.co/kambale/pearl-chat)

> 🌍 **意义**：Pearl-Chat 不仅是一个语言模型，更是非洲本土 AI 工程能力的象征，展示了如何从第一性原理出发，为被忽视的语言构建高性能、低成本、可复现的技术栈。

</details>

---

### 13. [Do Speech BCIs Need Larger Models? Rethinking Neural Decoding beyond Scaling](https://arxiv.org/abs/2610.23175)

**Authors**: Zehui Feng, Weichuan Wang, Xiaohan Chen, Cuntai Guan, Ting Han  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.23175v1  

#### Abstract
Recent speech brain-computer interfaces (BCIs) increasingly rely on large models and complex multi-stage pipelines for neural decoding. However, in speech BCI, where training data are scarce, noisy, and expensive to collect, the benefits of model scaling remain unclear. This work revisits a fundamen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Do Speech BCIs Need Larger Models? Rethinking Neural Decoding beyond Scaling*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **speech BCI**（脑机接口）系统普遍依赖大规模模型（如 LLMs）和多阶段流水线进行神经解码。然而，这类方法面临以下核心挑战：
- **资源消耗大**：大型模型和重打分（rescoring）机制导致高计算开销，难以在边缘设备（edge devices）部署。
- **离散音素瓶颈**（Discrete Phoneme Bottleneck）：将连续的神经信号压缩为硬音素标签会丢失丰富的时序和频谱结构，加剧过拟合。
- **训练数据稀缺且噪声大**：实际 BCI 数据量有限、信噪比低，模型扩展（scaling）带来的收益不明确。

该论文质疑了一个主流假设：**更大的模型是否真的能带来更好的语音 BCI 性能？**

---

### 提出了什么新方法或新思路
作者提出 **BEST**（**Brain-Encoding-Speech-Text**），一种轻量级、高效的两阶段神经到文本解码框架，其核心思想是：
- **避免端到端大模型** 和 **离散音素中心流水线**。
- 引入 **层级中间表示**（hierarchical intermediate representations），融合音素级和字符级特征。
- 使用一个紧凑的 **speech-oriented decoder**（基于 Whisper 架构）直接映射神经活动到文本。
- 在推理阶段结合轻量级 **n-gram LM**（如 KenLM）进行候选重排序。

#### 创新设计亮点：
1. **Hierarchical Transformer Encoder**  
   - 结合 **frequency-channel attention** 和 **iTransformer** 层，提取多尺度神经特征。
   - 采用三种时间掩码策略（past/local/future）逐步细化时间感受野。

2. **Phoneme-Character Fusion Block**  
   - 轻量级卷积融合模块，整合不同粒度的语言学信息，缓解对硬音素预测的依赖。

3. **Lightweight Speech Translator**  
   - 使用预训练 Whisper 模型上层作为解码器，无需微调整个 LLM，显著降低参数量和内存占用。

4. **Two-Stage Training Strategy**  
   - 第一阶段：冻结编码器，稳定音素/字符表示。
   - 第二阶段：仅更新融合模块和解码器首层，防止过拟合。

---

### 相比现有方法的优势
| 维度 | BEST 优势 |
|------|----------|
| **参数量** | 仅需 **0.059B** 参数（约 59M），仅为同类方法的 **3.85%** |
| **内存消耗** | 推理峰值内存仅 **2.79 GiB**（Jetson Orin Nano 上） |
| **延迟** | 推理延迟从 ~1190ms（BIT）降至 **~216ms** |
| **效率与鲁棒性** | 支持边缘部署，跨天泛化能力强，抗分布偏移 |
| **性能** | 在 Brain-to-Text '25 上达到 **1.62% WER**，优于所有基线 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Brain-to-Text '24**（T12）  
  - 来自一名参与者（T12），共 12,100 句话，25 个会话。
  - 电极数：128（iEEG），采样频率 20ms。
  - 内容：日常对话句子，尝试发声（attempted vocalized speech）。

- **Brain-to-Text '25**（T15）  
  - 来自另一名参与者（T15），共 10,948 句话，45 个会话。
  - 电极数：256（speech motor cortex）。
  - 包含 **尝试发声** 和 **默语**（silent speech）两种模式。

> 所有数据均已公开，并经过匿名化处理。

---

### 实验设置和评估指标

#### 主要评估指标：
| 指标 | 含义 |
|------|------|
| **WER**（Word Error Rate） | 单词错误率，主评价指标 |
| **CER**（Character Error Rate） | 字符错误率 |
| **PER**（Phoneme Error Rate） | 音素错误率 |
| **RAM**（运行内存） | 峰值内存占用（GiB） |
| **GFLOPS** | 浮点运算量，衡量计算复杂度 |
| **Inference Latency** | 推理延迟（ms） |

#### 实验平台：
- 训练：NVIDIA RTX 4090 GPU
- 推理测试：Jetson Orin Nano（8GiB）

#### 模型配置：
- **Lightweight BEST**：无外部 LM，端到端推理
- **Practical BEST**：+4-gram KenLM
- **Robust BEST**：+5-gram LM

---

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **RNN Baseline** [48] | 5层双向GRU + 5-gram LM + OPT rescoring |
| **Transformer Baseline** [11] | 因果Transformer + 时间掩码增强 |
| **DCoND** [23] | 基于双音素（diphone）建模 + GRU + LLM校正 |
| **BIT** [58] | 端到端跨物种预训练Transformer，MAE重建 + CTC微调 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Brain-to-Text '25）
| 方法 | #Params | RAM (GiB) | Test WER ↓ |
|------|--------|-----------|------------|
| RNN Baseline | 5.70B | 300 | 0.1127 |
| Transformer | 14.79B | 320 | 0.0934 |
| DCoND | 10.50B | 320 | 0.0879 |
| BIT (end-to-end) | 1.53B | 350 | 0.1106 |
| **BEST (Ours)** | **0.059B** | **5** | **0.0681** |
| **BEST + 4-gram LM** | **0.059B** | **32** | **0.0292** |
| **BEST + 5-gram LM** | **0.059B** | **320** | **0.0162** ✅ |

> 💡 **BEST 仅用 3.85% 的参数，实现了最低 WER（1.62%）**

---

### 与基线方法的对比结果
- **性能全面领先**：即使是最小配置（end-to-end），WER 也低于所有大模型基线。
- **效率极大提升**：
  - **计算量下降 14x**：GFLOPS 从 1736 → 123
  - **延迟下降 5.5x**：1189ms → 216ms
  - **内存下降 2.6x**：7.23GiB → 2.79GiB
  - **训练时间减半**：147s → 56s per epoch

- **语言模型增益更高效**：
  - 相同 5-gram LM 下，BEST 达到 0.0257 WER，而 Transformer 为 0.0381，GRU 为 0.0541。
  - 表明 **更强的神经表征质量** 是主导因素，而非 LM 利用能力。

---

### 消融实验结果（Ablation Studies）

#### （1）组件消融（Table 4）
| 模块添加顺序 | T12 CER ↓ | T15 WER ↓ |
|-------------|-----------|-----------|
| Vanilla Transformer | 0.1732 | 0.1573 |
| + Hierarchical Masking | 0.1597 | 0.1103 |
| + Frequency Attention | 0.1436 | 0.0896 |
| **+ Phoneme-Character Fusion** | **0.1394** | **0.0758** ✅ |

> 最大增益来自 **融合模块**，说明多层级表示集成至关重要。

#### （2）训练策略对比（Table 6）
| 训练方式 | T12 WER ↓ | T15 WER ↓ |
|---------|-----------|-----------|
| End-to-End | 0.3415 | 0.2071 |
| + Multi-task Loss | 0.2253 | 0.1067 |
| **Two-Stage Training** | **0.1504** | **0.0681** ✅ |

> 两阶段训练显著优于联合优化，验证了“先稳定表示，再生成文本”的有效性。

#### （3）过拟合分析（Table 5）
| 模型状态 | T15 PER ↓ | T15 CER ↓ | T15 WER ↑ |
|----------|-----------|-----------|-----------|
| Well-fitted | 0.0688 | 0.0896 | 0.0681 ✅ |
| Over-fitted | 0.0621 | 0.0825 | **0.0803** ❌ |

> 过拟合模型虽然局部音素准确率更高，但 **序列级一致性差**，最终 WER 更差 —— 揭示了“局部精度 ≠ 全局性能”。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **模型缩放并非万能**：在低资源、高噪声的 speech BCI 场景中，**decoder scaling 呈非单调趋势**，中等规模、模态对齐的模型表现最佳。
2. ✅ **跨模态对齐 > 模型大小**：音频对齐的预训练模型（如 Whisper）远优于同等规模的纯文本 LLM（如 Llama3.1），说明 **representation alignment** 是关键。
3. ✅ **离散音素瓶颈限制性能**：强制将神经信号映射为硬音素标签会导致信息损失和过拟合；保留连续、多层次表示更有利于鲁棒解码。
4. ✅ **结构化设计胜于盲目扩容**：通过层级注意力、频率通道调制和轻量融合模块，可在极小参数下实现高性能。
5. ✅ **过拟合损害全局一致性**：局部音素错误率降低可能伴随 WER 上升，因模型丧失时序连贯性。

---

### 方法的局限性
1. **单被试评估**：目前仅在两个独立被试（T12/T15）上验证，**跨被试泛化能力未知**。
2. **未解决长期非平稳性**：尽管跨天性能稳定，但对长期神经漂移（neural drift）和自适应解码支持不足。
3. **依赖预训练语音模型**：使用 Whisper 作为解码器引入了其固有的归纳偏置，未必完全匹配神经到语言的映射路径。
4. **尚未探索全神经原生架构**：仍借助语音模型桥接，未来可研究完全基于神经信号的 end-to-end 解码。

---

### 未来工作方向
1. **跨被试通用解码器**：构建可迁移的神经语言对齐模型，减少个体校准成本。
2. **动态自适应机制**：引入在线学习或元学习策略应对神经信号长期变化。
3. **联合学习神经-语音表示**：开发专用于 BCI 的 joint neural-speech foundation model。
4. **全神经原生解码架构**：摆脱对语音模型的依赖，构建真正从神经到文本的 native decoder。
5. **更大规模数据验证**：随着高质量神经数据积累，重新审视模型缩放的有效边界。

---

> 🔚 **总结一句话**：  
> 在 speech BCI 中，**representation quality 和 structured alignment 比 model size 更重要**。  
> **BEST 证明了：更小、更聪明的设计，可以打败更大、更贵的模型。**

</details>

---

### 14. [DeDiCa: A Scalable Deconfounding Discrete Causal Generative Model](https://arxiv.org/abs/2610.23272)

**Authors**: Kunchangtai Liang, Adri\'an Javaloy, Antonio Vergari, Isabel Valera  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.23272v1  

#### Abstract
We introduce DeDiCa, a practical framework for efficient causal inference with discrete variables and hidden confounding that revisits classical probabilistic modeling through a tensorized parameterization of causal Bayesian networks, enabling tractable inference. Given a causal graph, DeDiCa is tra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DeDiCa: A Scalable Deconfounding Discrete Causal Generative Model**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对**离散变量场景下的因果推断**问题，特别是存在**隐变量混杂（hidden confounding）** 的情况。在现实应用中，许多观测数据受到未被测量的共同原因（即隐变量）影响，导致传统基于观察数据的因果估计产生偏差。现有方法如神经因果生成模型（Neural CGMs）或因果基础模型（Causal Foundation Models）虽然灵活，但在处理离散数据时往往面临以下挑战：
- 推理近似且不精确；
- 训练和推理计算开销大；
- 对因果图结构利用不足，导致在相似图结构下表现不稳定。

### **提出了什么新方法或新思路**
作者提出 **DeDiCa（Deconfounding Discrete Causal generative model）**，其核心思想是：
- 将**结构因果模型（SCM）** 表示为一种**张量化的混杂因果贝叶斯网络（T-CCBN, Tensorized Confounded Causal Bayesian Network）**。
- 利用已知的**因果图结构**和**离散变量特性**，通过张量操作实现对隐变量 $ Z $ 的**精确边缘化（exact marginalization）**。
- 在一次最大似然训练后，即可支持任意（proxy-）可识别的干预查询（interventional query），无需重新训练。

#### **关键技术亮点**：
- **张量参数化（Tensorized Parameterization）**：将条件概率表（CPTs）表示为高维张量，便于进行高效的代数运算。
- **Join Tree 加速推理**：借鉴可追踪概率模型（tractable probabilistic models）中的 join tree 结构，优化多变量联合分布的求和与乘积顺序，显著降低计算复杂度。
- **Proxy-Identifiability 支持**：基于 proxy 变量理论，在无法直接调整隐变量的情况下仍能准确估计因果效应。

### **相比现有方法的优势**
| 维度 | DeDiCa | 现有方法（如 NCMs, CausalPFN） |
|------|--------|-------------------------------|
| **精度** | 支持**精确的概率推理** | 多为采样或近似推理，误差较大 |
| **效率** | **毫秒级干预推理**，CPU 上高效运行 | 需要 GPU，推理耗时长（秒到分钟级） |
| **训练成本** | 单次训练，支持所有查询 | 某些方法需每个查询单独训练 |
| **可扩展性** | 随样本增加更有效利用数据 | 推理时间随上下文增长而上升 |
| **一致性** | 显式建模因果图，跨相似图结构表现稳定 | 忽略图结构，易受机制差异影响 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
采用半合成（semi-synthetic）和全合成数据集，确保有真实因果效应可供比较：

| 数据集 | 变量数量 | 类型 | 是否含隐变量 | 描述 |
|-------|---------|------|---------------|------|
| **Asia** | 7 | 二元 | 是（1个 binary Z） | 医疗诊断小图 |
| **Sachs** | 9 | 三元 → 二元化 | 是（2个 ternary Z） | 蛋白质信号通路 |
| **Child** | 18 | 非二元 → 二元化 | 是（1个 hidden Z） | 儿科疾病诊断 |
| **Ecoli70** | 44 | 二元 | 是（2个 binary Z） | 基因调控网络 |
| **Synthetic Graphs** | 最多36 | 二元/四元 | 是 | 合成链状、密集、混杂图用于复杂度分析 |

> 所有数据均从真实图结构生成，并加入隐变量以模拟混杂效应。

### **实验设置和评估指标**

#### **评估任务**
- 平均处理效应（Average Treatment Effect, **ATE**）估计：
  $$
  \text{ATE} = \mathbb{E}[Y \mid do(T=1)] - \mathbb{E}[Y \mid do(T=0)]
  $$

#### **主要指标**
- **ATE MAE**：预测 ATE 与真实 ATE 的绝对误差均值。
- **训练时间（Training Time）**：完整训练耗时。
- **单次查询评估时间（Evaluation Time）**：执行一个干预查询的时间。
- **内存消耗（Memory Usage）**：峰值驻留内存增量。
- **验证负对数似然（Validation NLL）**：衡量模型拟合能力。

#### **基线方法对比**
| 方法 | 类型 | 是否支持隐变量 | 是否需要干预数据 |
|------|------|----------------|------------------|
| **CausalPC** | Probabilistic Circuit | 否（仅可观测） | 否 |
| **CNF / DeCaFlow** | Causal Normalizing Flow | 是 | 否 |
| **NCM / GAN-NCM** | Neural Causal Model | 是 | 是（部分） |
| **CausalPFN** | Causal Foundation Model | 是 | 否（上下文学习） |

> 所有方法统一在相同数据划分下进行比较（训练集 1M，验证集 1M）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 3 & 4）**

#### **表：在多个数据集上的 ATE MAE 和效率对比（摘要）**

| 方法 | ATE MAE (Sachs) | Train Time | Eval Time per Query |
|------|------------------|------------|---------------------|
| **DeDiCa** | **0.024** | 0.57s | **2.62ms** |
| CNF | 0.023 | 2585s | 160.84s |
| NCM | 0.047 | 106962s (~30h) | 20.60s |
| GAN-NCM | 0.097 | 4035s | 0.24s |
| DeCaFlow | 0.078 | 3679s | 95.88s |
| **CausalPFN** | 0.039 | — | **~700–1200s** |

> ✅ **DeDiCa 在 Sachs 上达到最佳或接近最佳精度，同时推理速度快 2–4 个数量级。**

#### **内存与可扩展性**
- DeDiCa 在 CPU 上仅使用 **<20 MiB 额外内存** 进行推理。
- 随着变量数 $ D $ 增加，DeDiCa 在低树宽（treewidth）图上保持**亚线性时间增长**（见 Fig. 7），适用于稀疏现实图。
- 相比之下，dense 图（高 treewidth）会导致 join tree 中间张量爆炸，但仍优于 baseline。

### **与基线方法的对比结果**
- **vs. Neural CGMs (NCM, DeCaFlow)**：
  - 精度相当甚至更好（尤其在 Sachs、Child）；
  - 训练速度提升 **1000倍以上**；
  - 推理速度提升 **10^4 倍**。
- **vs. Causal Foundation Models (CausalPFN)**：
  - ATE MAE 更低或相当（如 Asia 上 **0.001 vs 0.052**）；
  - **训练一次即可复用**，而 CausalPFN 每个查询都要“评估”一次（实际是前向传播）；
  - **DeDiCa 总训练时间 ≈ CausalPFN 单次查询时间**，极具性价比。

### **消融实验结果**

#### **(1) 隐变量类别数 $ C_z $ 的估计有效性（RQ1）**
- 使用 **Lemma 3 + 随机搜索算法（Alg. 1 & 2）** 估计 $ C_z $
- 实验表明：
  - 在 **Child** 数据集上准确恢复 $ C_z = 6 $
  - 在 **Sachs** 上低估为 2（真实为 3），但仍是性能最优的配置
  - 使用验证集 NLL 可进一步微调选择最优 $ C_z $

> 👉 表明提出的 $ C_z $ 估计算法实用且鲁棒。

#### **(2) 不同图结构下的稳定性测试（Fig. 4 & 14）**
- 构造三个结构相似但机制不同的合成图（Fig. 4）
- 结果显示：
  - **DeDiCa 始终保持低 ATE MAE（≤0.005）**
  - **CausalPFN 表现波动大（0.009 ~ 0.099）**，说明其忽略图结构导致泛化差

> 👉 显式建模因果图使 DeDiCa 更可靠。

#### **(3) 数据规模扩展性（Fig. 13）**
- 随训练样本从 $ 10^2 $ 到 $ 10^6 $ 增加：
  - DeDiCa 的 ATE MAE 持续下降，尤其在 **Asia 和 Ecoli70** 上表现优异
  - CausalPFN 下降缓慢，且在小样本下偏差更大

> 👉 DeDiCa 更有效地利用大规模观测数据。

---

## **4. 关键结论和发现**

### **主要发现**
1. **简单而有效的设计胜过复杂模型**：
   - 尽管 DeDiCa 基于经典的 CBN 和张量表示，但它在现代任务中**匹配甚至超越最先进的神经因果模型**。
   - “**是否需要如此复杂的模型？**”——本文给出了否定答案：对于具有已知结构的离散系统，简洁的参数化足以实现高效准确的因果推断。

2. **显式建模因果图至关重要**：
   - DeDiCa 因显式使用因果图，在面对机制变化但结构相似的情况时表现出更强的一致性和鲁棒性。
   - 相比之下，黑箱式的 foundation models 容易失败。

3. **精确推理 + 张量优化 = 高效可扩展**：
   - 利用 join tree 和张量操作实现了 **O(DC^W)** 的推理复杂度（$ W $ 为 treewidth），在稀疏图上非常高效。
   - 支持在普通 CPU 上完成训练和毫秒级推理，适合资源受限场景。

### **方法的局限性**
1. **依赖 $ C_z $ 的估计**：
   - 若无法正确估计隐变量的类别数，可能限制表达能力或引入噪声。
   - 当前方法假设 $ C_z $ 较小，难以应对连续或极高维隐空间。

2. **依赖 proxy-identifiability 条件**：
   - 必须存在满足 rank 条件的 proxy 变量集合才能保证识别性。
   - 在 proxy 信息不足时，无法提供可靠的置信区间。

3. **目前仅限于离散变量**：
   - 无法直接处理连续或混合型数据，需离散化带来信息损失。

### **未来工作方向**
1. **扩展至连续/混合变量**：结合归一化流或其他连续表示，构建 hybrid 版本。
2. **自动结构发现**：当前依赖已知因果图，未来可探索与结构学习结合。
3. **更强大的隐变量建模**：引入 overparameterization 或 probabilistic circuits 提升表达力。
4. **支持反事实推理（counterfactuals）**：当前仅支持 interventional queries，尚未覆盖第三层因果推理。
5. **应用于真实世界领域**：如医疗、推荐系统等需要可信因果推断的大规模场景。

---

> **总结一句话**：  
> **DeDiCa 展示了“经典建模 + 现代工程优化”路径的强大潜力——在一个精心设计的张量化框架下，简单的模型也能实现高效、精确、可扩展的因果推断，尤其适合离散、稀疏、带隐变量的真实应用场景。**

</details>

---

### 15. [FlexUMM: Flexible Representation Learning for Unified Multimodal Models](https://arxiv.org/abs/2610.22682)

**Authors**: Yinyi Luo, Haoyue Bai, Yi Ge, Nilton Cesar Rojas Vales, Marios Savvides, Jindong Wang  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.22682v1  

#### Abstract
Unified multimodal models (UMMs) can integrate understanding and generation within a single backbone. However, it is commonly known that these two capabilities can cause conflicting representations, leading to a central question: how representation and computation should be shared across functionali...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FlexUMM: Flexible Representation Learning for Unified Multimodal Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
统一多模态模型（Unified Multimodal Models, UMMs）旨在通过一个共享的骨干网络同时支持**理解（understanding）** 和 **生成（generation）** 任务。然而，这两种功能在表示学习上存在冲突：
- **Representation Conflict**：理解与生成对参数的需求不同，导致共享表示产生竞争。
- **Fixed Computation Structure**：现有UMMs通常采用固定的计算深度和参数共享模式，无法根据输入动态调整计算量。

这导致两个问题：
1. **效率低下**：简单输入仍需完整计算。
2. **性能下降**：功能间干扰降低准确率。

---

### **提出的新方法：FlexUMM**
作者提出 **FlexUMM** —— 一种灵活的表示学习框架，实现**动态计算分配**，其核心思想是：

> **“Progressive Share, Specialize, and Stop”**  
> 即：逐步**共享通信**、**选择性专业化**、并在适当时**提前终止计算**。

#### **两大机制**：
1. **Capability-Adaptive Capacity Allocation（能力自适应容量分配）**
   - 在每个 **Transformer 层**中引入 **Tri-state Gate**，为每个 token 动态选择三种操作之一：
     - `SHARED`：使用共享的 FFN（Feed-Forward Network）
     - `SPECIALIZED`：在共享 FFN 上叠加一个低秩残差分支（low-rank residual）
     - `BYPASS`：跳过当前 FFN 更新
   - **关键洞察**：注意力（Attention）模块适合共享（梯度对齐），而 FFN 更易产生功能冲突，应允许专业化。

2. **Depth-Adaptive Computation（深度自适应计算）**
   - 引入 **Halting Head**，在预设的候选深度（candidate depths）处判断是否已获得足够成熟的表示。
   - 若满足条件，则提前退出后续层计算，节省推理开销。
   - 支持在推理时通过阈值调节精度-计算权衡。

---

### **相比现有方法的优势**
| 维度 | FlexUMM | 传统UMMs |
|------|--------|----------|
| **灵活性** | ✅ 输入级、token级、序列级动态决策 | ❌ 固定结构 |
| **效率** | ✅ 减少冗余计算（early exit + selective specialization） | ❌ 所有输入走完全部层 |
| **准确性** | ✅ 缓解功能冲突，提升理解与生成性能 | ❌ 存在功能干扰 |
| **通用性** | ✅ 插件式设计，适用于多种UMM架构（model-agnostic） | ❌ 架构依赖性强 |
| **安全性** | ✅ 自适应深度有助于更安全地处理敏感输入 | ❌ 固定流程可能忽略风险信号 |

> 🔧 **轻量化设计**：仅增加 **0.381%** 模型大小（仅添加 routers、halting heads 和 low-rank residual banks）。

---

## **2. 核心实验方法和设置**

### **使用的数据集**

| 类型 | 数据集 |
|------|-------|
| **训练数据（Post-training）** | ScienceQA, AI2D, TextVQA, MathV-360K, InstructPix2Pix |
| **理解任务评估** | MMMU, MathVista, MMStar, MMBench |
| **生成任务评估** | GenEval, WISE |
| **统一模型综合评估** | Unison, Unified-Bench |
| **安全评估** | Uni-SafeBench |

---

### **实验设置**

- **Backbone Models**：
  - 主要基准：**Bagel**（14B）
  - 泛化验证：Janus-Pro（7B）、Harmon（1.5B）
- **插入方式**：
  - FlexUMM作为插件附加于预训练检查点之上，不修改原始骨干。
- **可调参数**：
  - 推理时可通过调整 `η`（halt threshold）控制 **accuracy-compute trade-off**。
- **实现工具**：基于 **TorchUMM** 框架进行公平比较。

---

### **评估指标**

| 指标 | 说明 |
|------|------|
| **MMMU / MMBench / MathVista 等得分** | 多模态理解准确率 |
| **GenEval / WISE** | 文生图质量评估 |
| **Unison / Unified-Bench Overall Score** | 统一模型综合能力 |
| **Relative Compute** | 相对计算量代理（layer-token + action-level cost proxy） |
| **Mean Exit Depth** | 平均退出层数，反映计算节省程度 |
| **Specialization Rate** | 选择 `SPECIALIZED` 路径的 token-layer 比例 |

---

### **基线方法对比**
- **原始UMM模型**：Bagel, Janus-Pro, Harmon, Show-O2, Emu3/3.5, Omnigen2 等。
- **增强版Bagel**：Bagel+SFT, Bagel+UniCot, Bagel+IRG。
- **消融变体**：
  - 固定深度 vs 自适应深度
  - 是否路由 Attention 或 FFN
  - 随机路由 vs 学习路由

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（以 Bagel 为例）**

| 模型 | MMMU | MMBench-CN | MMBench-EN | MathVista | MMStar | GenEval | WISE | Rel. Compute |
|------|------|------------|------------|-----------|--------|---------|-------|--------------|
| Bagel | 51.9 | 80.94 | 82.65 | 71.3 | 63.20 | 78.81 | 0.4044 | 1.00 |
| **Bagel + FlexUMM** | **54.0** (+2.1) | **85.87** (+4.93) | **86.75** (+4.1) | **73.4** (+2.1) | **70.93** (+7.73) | **82.03** (+3.22) | **0.4292** (+6.1%) | **0.96** |

✅ **全面超越所有基线**，且在理解与生成任务上均有提升。

---

### **与基线方法的对比结果**

- 在 **5项理解任务** 上均取得 **SOTA**。
- 在 **Unison** 和 **Unified-Bench** 上显著提升整体分数：
  - Unison Overall: **55.59 → 59.83**（↑7.6%）
  - Unified-Bench Overall: **83.46 → 85.05**（↑1.9%）
- 虽然不如专用于生成的模型（如 Emu3.5），但在统一模型中表现最强。

---

### **消融实验结果**

| 变体 | MMMU | MMBench-EN | MathVista | 结论 |
|------|------|------------|-----------|------|
| Routed Attention + Shared FFN | 52.1 | 81.95 | 71.2 | 不如原模型 |
| Routed Attention + Routed FFN | 53.2 | 84.24 | 71.9 | 部分恢复增益 |
| **Routed FFN only (FlexUMM)** | **54.0** | **86.75** | **73.4** | ✅ 最优策略 |
| Fixed depth d=26 | 52.1 | 84.13 | 67.2 | 明显低于自适应版本 |
| Random routing | 52.4 | 74.34 | 71.0 | 性能下降，证明学习路由必要性 |

> 📌 **核心发现**：
> - **只路由 FFN + 保持 Attention 共享** 是最优设计。
> - **固定深度** 无法适应多样输入，性能受限。
> - **随机路由失败**，说明性能提升来自智能调度而非参数增加。

---

## **4. 关键结论和发现**

### **主要发现**

1. ✅ **Attention 适合共享，FFN 应允许专业化**  
   - 梯度分析显示：Attention 梯度在整个网络中保持高度对齐；FFN 梯度随深度加深逐渐冲突。
   - → 支持“共享通信、分离变换”的设计理念。

2. ✅ **计算深度应由输入决定**  
   - 不同输入达到“充分表示”的所需深度差异大（如 MME 上从 22 到 28 层）。
   - 提前退出不会损害质量，反而避免过度计算带来的干扰。

3. ✅ **FlexUMM 实现双赢：更高精度 + 更高效率**
   - 平均提升 **10.17%** 性能，同时减少 **6.33% FLOPs**。
   - 支持运行时调节 `η` 实现 **accuracy-compute 曲线**，无需重新训练。

4. ✅ **提升安全性**
   - 在 Uni-SafeBench 上：
     - C-Unsafe ↓ (43.72% → 42.35%)
     - Both-Safe ↑ (43.96% → 44.86%)
   - 表明自适应计算有助于更好地识别并响应安全相关线索。

5. ✅ **跨架构泛化能力强**
   - 在 Bagel、Janus-Pro、Harmon 三种不同结构的 UMM 上均有效，验证了其 **model-agnostic** 特性。

---

### **局限性（Limitations）**

1. **候选深度有限**：仅在离散的几个候选深度进行退出决策，限制了细粒度调控。
2. **当前实现未优化延迟**：虽然计算量下降，但由于 Dense Kernel 实现，实际 wall-clock speedup 尚未完全体现（action-level specialization 依赖稀疏或融合算子支持）。
3. **仅测试主流UMM**：尚未在更大规模或更多样化的模型家族中广泛验证。

---

### **未来工作方向**

- 设计连续可微的退出机制（Continuous Adaptive Depth）。
- 结合硬件感知调度，实现真正的低延迟推理。
- 扩展到更多模态（如音频、视频）和任务类型。
- 探索更复杂的路由策略（如多专家混合、递归计算路径）。

---

> 💡 **一句话总结**：  
> **FlexUMM 通过“共享通信、选择性专业化、动态停止”的三阶段策略，在几乎零成本的前提下，让统一多模态模型变得更聪明、更高效、更安全。**

</details>

---

### 16. [SETAREH Transform: A Sorting-Assisted Haar-Wavelet Sparsification Method for Approximate Inner-Product Computation](https://arxiv.org/abs/2610.23338)

**Authors**: Seyed Kamaledin Setarehdan  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.23338v1  

#### Abstract
Modern deep learning systems increasingly rely on high-dimensional vector operations, in which large input and parameter vectors are repeatedly stored, transferred, and processed. Although substantial research has focused on reducing neural-network model size through pruning, quantization, knowledge...

---

### 17. [ToPA: Block-wise Toeplitz Adaptation for Expressive and Efficient Fine-Tuning](https://arxiv.org/abs/2610.22883)

**Authors**: Sicong Li, Qianqian Xu, Zhiyong Yang, Zitai Wang, Longtao Huang, Xiaochun Cao, Qingming Huang  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.22883v1  

#### Abstract
Parameter-Efficient Fine-Tuning (PEFT) has emerged as a prevalent strategy for adapting large pre-trained models. Among PEFT techniques, reparameterization-based methods, particularly those that employ low-rank or sparsity structures, have attracted significant attention. However, these approaches o...

---

### 18. [FuzzInsight: Navigating the Efficiency-Effectiveness Trade-off in Coverage-Guided DNN Fuzzing](https://arxiv.org/abs/2610.23195)

**Authors**: Vivek Vekariya, Fathiyeh Faghih, Alexander Pretschner  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.23195v1  

#### Abstract
Collecting and labeling diverse test datasets for evaluating Deep Neural Networks (DNNs) is costly and time-consuming. This has motivated coverage-guided fuzzing techniques that generate defect-revealing corner cases from limited test data. However, their effectiveness depends heavily on configurati...

---

### 19. [HeRE: A Novel Benchmark for Hebrew Relation Extraction via LLM-Guided Denoising of Knowledge Graph Alignments](https://arxiv.org/abs/2610.22542)

**Authors**: Ron Keinan, Amir David Nissan Cohen, Reut Tsarfaty  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.22542v1  

#### Abstract
While LLMs demonstrate strong capabilities in English Relation Extraction (RE) tasks, a significant gap remains in assessing their RE performance in low-resource languages, due to a lack of standardized benchmarks. Creating native, language-specific annotated RE benchmarks is expensive and requires ...

---

### 20. [Identify, Classify, Act: Per-Instruction Controlled Compliance for LLMs over External Data](https://arxiv.org/abs/2610.22874)

**Authors**: Zesheng Shi, Xue Li, Chenrui Dai, Jing Li  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.22874v1  

#### Abstract
LLMs over external data face a per-instruction tension: a single document or tool return often interleaves legitimate directives with adversarial prompt injections, and the model must follow the former while rejecting the latter. Existing defenses suppress or trust all external instructions, collaps...

---

### 21. [Acceptance Is Not Speed: Length Inflation in Training-Free Lossy Speculative Decoding](https://arxiv.org/abs/2610.23818)

**Authors**: Chiatzen Wang, Bill Xu, Haochen Gu, Hongyang Zhang  
**Category**: cs.CL  
**Published**: 2026-10-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.23818v1  

#### Abstract
Speculative decoding typically characterizes speedup by draft tokens accepted per verification round. Lossless verification preserves the target model's output distribution and thus expected completion length, making this metric a meaningful efficiency proxy. Relaxed (lossy) verification accepts tok...

---

### 22. [Bridging Compressed Communication and Local Steps via Spectral Method in Federated Learning](https://arxiv.org/abs/2610.23434)

**Authors**: Dmitriy Bystrov, George Chirkov, Daniil Medyakov, Dmitry Bylinkin, Aleksandr Beznosikov  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.23434v1  

#### Abstract
Federated Learning (FL) enables collaborative training of complex models across large-scale networks of edge devices while keeping data localized. However, data heterogeneity and high communication overhead remain fundamental bottlenecks. Recently, the spectral optimizer Muon has demonstrated remark...

---

### 23. [Deep LLM layers expose passage-level context in EEG during silent reading](https://arxiv.org/abs/2610.23638)

**Authors**: Ingo Marquardt, Anthilia Alchanat, Priyanka Jain  
**Category**: cs.LG  
**Published**: 2026-10-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.23638v1  

#### Abstract
In a previous report we showed that a decoder can recover open-vocabulary word-level information from 19-channel EEG recorded during silent reading, by aligning short EEG windows with hidden-state embeddings of the presented word taken from a large language model (LLM). That analysis contrasted two ...

---

### 24. [ECHOES: Seeing Beyond the Current Image via Recursive History Compression and Privileged On-Policy Distillation](https://arxiv.org/abs/2610.22408)

**Authors**: Tengda Guo (Tsinghua University), Jiawang Hu (StepFun), Yifei Tao (Nanyang Technological University), Yin Wang (StepFun), Haotian Hu (Zhejiang Leapmotor Technology Co., Ltd), Fanyi Wang (Honor), Liujuan Tang (StepFun)  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.22408v1  

#### Abstract
Long-horizon GUI agents rely on visual history to maintain task states, but retaining historical screenshots increases visual-token usage and inference costs, while textual summaries can discard critical state information, undermining long-horizon reliability. We propose ECHOES, which leverages trai...

---

### 25. [Learnable Splines for Unstructured Neural Operators](https://arxiv.org/abs/2610.22446)

**Authors**: Vignesh Gopakumar, Zhuoyuan Wang, Ander Gray, Valentin Duruisseaux, Josh Williams, William Michael Edward Ellis, Nuno Nobre, Matthew Field, James Panayis, Alexander I. Blair, Andrew Davis, Stanislas Pamela, Robert Akers, Marc Peter Deisenroth, Anima Anandkumar  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.22446v1  

#### Abstract
Neural operators approximate solution operators of partial differential equations at a fraction of the cost of numerical solvers. Efficient grid-based architectures such as the Fourier neural operator (FNO) assume uniform rectangular grids, whereas engineering simulations use large, irregular, and l...

---

### 26. [LEMON: Equivariant Learning over Attention Maps for Zero-Shot Misbehavior Monitoring on New LLMs](https://arxiv.org/abs/2610.22723)

**Authors**: Guy Bar-Shalom, Mor Geva, Fabrizio Frasca, Haggai Maron  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.22723v1  

#### Abstract
Large language models exhibit misbehaviors that hinder deployment in high-stakes settings. One approach to monitor such failures is to use an external LLM as a monitor, but this is prohibitively expensive. A well-known and efficient class of monitors can detect such failures from internal representa...

---

### 27. [MS-Eval: Benchmarking Sub-Agent Delegation and Orchestration in MCP-Intensive Working Scenarios](https://arxiv.org/abs/2610.22869)

**Authors**: Bowen Ding, Qixin Wang, Lexing Ruan, Weiyi Lu, Qiwei Ge, Ruishan Fang, Tao Lin  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.22869v1  

#### Abstract
LLM agents are increasingly integrated into real-world works, yet they must handle diverse domain-specific tools and complex multi-sub-task problems that sequential tool invocation resolves inefficiently and with severe context bloat. The Main-Agent Delegation and Sub-Agent Execution (MAD-SAE) parad...

---

### 28. [Sep-Latent: Separator Tokens as Natural Compression Anchors for Implicit Reasoning](https://arxiv.org/abs/2610.22924)

**Authors**: Yinlong Xu, Jianfeng Huang, Zhengdong Zhou, Mingjie Lin, Fuyu Lv, Dan Ou, Hongxia Xu, Jian Wu  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.22924v1  

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable success in complex reasoning through explicit Chain-of-Thought (CoT), but the associated computational overhead and token verbosity remain significant barriers to efficiency. While implicit reasoning in continuous latent spaces offers a promi...

---

### 29. [Adaptation of Computer Use Agents to New Software with User Manuals](https://arxiv.org/abs/2610.23012)

**Authors**: Chulhoon Yoon, Kyungri Park, Jinyeong Yun, Daeyun Jeong, Sukmin Yun, Woohwan Jung  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.23012v1  

#### Abstract
Recent advances in multimodal foundation models have significantly improved the ability to operate real-world software through graphical user interfaces (GUIs) using visual observations and natural language instructions. However, maintaining strong performance on unseen applications remains challeng...

---

### 30. [Beyond Large Language Models: A Mental World Model Framework for Mental Modeling, Simulation, Evolution, and Interpretability](https://arxiv.org/abs/2610.23722)

**Authors**: Qingshi Wang  
**Category**: cs.AI  
**Published**: 2026-10-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.23722v1  

#### Abstract
Understanding and simulating how subjective cognition and mental states emerge, evolve, and shape behavior is essential for advancing artificial intelligence toward deeper social cognition and general intelligence. Existing large language models (LLMs) and Theory of Mind (ToM) methods primarily infe...

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

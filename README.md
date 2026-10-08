# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-08 12:18:16 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Cascadia: Resident 975B MoE Inference on Eleven AI PCs](https://arxiv.org/abs/2610.07219)

**Authors**: Tate Berenbaum (Not Community Labs Inc.), Matias Parij (Not Community Labs Inc.), Muthaiah Venkatachalam (Intel Corporation)  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.07219v1  

#### Abstract
Mixture-of-experts models make nearly trillion-parameter capacity accessible with sparse per-token computation, provided that the serving system can distribute the weights and coordinate their execution. We present Cascadia's resident execution of Inkling, a 975B-total/41B-active-parameter model, on...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Cascadia: Resident 975B MoE Inference on Eleven AI PCs》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文旨在解决**在分布式客户端设备上高效部署超大规模稀疏 MoE 模型**的问题。具体挑战包括：
- 如何在内存有限（每台设备仅 64 GB）的消费级 AI PC 上运行总参数达 975B 的 MoE 模型（Inkling）；
- 如何协调跨设备的专家权重分布、路由逻辑与计算调度；
- 如何统一处理 dense 和 sparse 层以提升推理效率；
- 如何准确评估部署路径下的 draft agreement（如用于 speculative decoding）。

传统方案依赖高性能 GPU 集群和高带宽互联（如 NVLink），而本工作探索的是基于 **consumer-grade AI PCs**（集成 GPU + 共享内存架构）的可行路径。

---

### 提出的新方法或新思路

#### （1）定制化的 resident MoE 引擎（Custom Resident MoE Engine）
- **模型语义保留**：完整保留 Inkling 的专家路由规则（top-6 + 2 shared experts），并在 Rust 中实现动态路由决策。
- **图压缩与融合**：利用 OpenVINO 的 `fused iGPU` primitives 构建压缩图，将多个矩阵操作融合为单个 kernel，显著降低开销。
- **FP16/FP32 数值管理**：设计了一套 layer- 和 row-level 的缩放机制（$A_e$, $F$），确保 FP16 专家计算输出能安全还原至 FP32 residual 路径，避免溢出或精度损失。

#### （2）dense 与 sparse 块的统一算子表示
- 将前两层 dense feed-forward block 分解为 **8 个全激活的 expert slices**，复用相同的 fused MoE 算子路径。
- 实现了 operator-level 统一，使 dense 层也能享受 fused iGPU 加速。
- 测量显示调用时间从 ~8.1ms 降至 ~4.5ms，提速约 **45%**。

#### （3）流式并发服务管道（Streaming Pipeline）
- 支持多请求并行处理，不同请求可处于 pipeline 不同阶段；
- 引入 **八行预填充窗口（eight-row prefill windows）** 和直接 token 返回路径，减少反向传播延迟；
- 实现了 **stateful streams**，每个流维护独立的 KV cache 和 convolution state。

#### （4）基于真实部署状态的 draft evaluation 方法
- 捕获实际服务过程中产生的 FP32 residuals 和 emitted token IDs；
- 在离线回放中评估 MTP（multi-token prediction）head 的 first-draft agreement；
- 可分离 **vocabulary selection** 与 **weight quantization** 对 draft 准确率的影响。

---

### 相比现有方法的优势

| 方面 | Cascadia 优势 |
|------|---------------|
| **硬件平台** | 使用 11 台消费级 AI PC（Intel Core Ultra X7 + Arc B390 iGPU），而非昂贵 DGX 集群 |
| **内存利用** | 利用共享 CPU-GPU 内存，避免冗余拷贝；支持高达 512k context positions |
| **执行效率** | 定制引擎 + fused iGPU primitives 显著降低 dense/sparse 层延迟 |
| **评估真实性** | draft evaluation 基于真实 fleet 输出，反映实际数值路径行为 |
| **系统集成度** | 实现了从 admission 到 generation 的端到端控制平面整合 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Prompt 数据**：未使用公开 benchmark，而是构建了 **12 个 deterministic prompt families**，涵盖：
  - Explanation, Code, Arithmetic, Story, Tips, Table, Rewrite, Facts, Poem, Instructions, Translation, True/false
- **评估序列**：用于 draft evaluation 的 **36 条序列**（每 family 3 条），共生成 5,760 tokens，提供 5,724 个预测目标。

> 所有 prompts 均通过 deterministic seed 生成，保证可复现性。

---

### 实验设置

#### 硬件配置
- **设备数量**：11 台 Intel Panther Lake AI PC（Core Ultra X7 358H）
- **每台配置**：
  - CPU：16 核（4P + 8E + 4LP-E）
  - iGPU：Arc B390（12 Xe cores）
  - 内存：64 GB LPDDR5X（实测可用 ~61.3 GiB）
  - 网络：Gigabit Ethernet
- **总计内存容量**：704 GB（名义）

#### 模型分区
- **模型**：Inkling（975B total params, 41B active per token）
- **Decoder Layers**：66 层 → 每台机器负责连续 6 层（role 0 和 role 10 特殊处理 embedding 和 head）
- **权重格式**：
  - Experts：group-32 INT4
  - Attention projections / Head：INT8
  - Residuals：FP32

#### 运行时环境
- OS：Linux（kernel 7.0.0-31-generic）
- Runtime：Rust（状态管理）+ OpenVINO 2026.3.1（iGPU 推理）

---

### 评估指标

| 指标 | 定义 |
|------|------|
| **Q_decode** | 所有请求同时处于 decoding 阶段时的聚合吞吐量（tokens/s） |
| **Q_phase** | 包含 prefill、queueing、drain 的全流程吞吐量 |
| **TTFT (Time to First Token)** | 从请求提交到首个 token 发出的时间（中位数与 p95） |
| **First-draft Agreement** | draft model 预测的第一个 token 与 target model 一致的比例 |
| **Per-position Decode Latency** | 单个 decode step 的延迟随 context length 的变化趋势 |

---

### 基线方法对比
本文未直接对比其他系统（如 Petals、TPI-LLM），而是强调其独特部署场景下的性能表现。隐含对比对象包括：
- **DGX Spark 部署**（参考文献 [6]）：使用 tensor parallelism + NVFP4，更高带宽，但成本高昂；
- **纯 CPU 或边缘设备推理系统**（如 prima.cpp, exo）：缺乏对 MoE 和 fused iGPU 的优化；
- **Speculative Decoding 基线**：通过启用/禁用 speculation 观察吞吐增益。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 并发服务性能（Table 4）

| Streams | Q_decode (tokens/s) | Q_phase (tokens/s) | TTFT median (s) |
|--------|---------------------|--------------------|------------------|
| 1      | 7.96                | 6.98               | 2.18             |
| 15     | 24.60               | 22.12              | **6.05**         |
| **88** | **60.29**           | **46.87**          | 34.61            |
| 176    | 57.72               | 45.24              | 76.83            |

> ✅ **最高吞吐出现在 88 streams（8 streams × 11 pipeline groups）**

#### 单请求 speculative decoding 性能提升（Table 3）

| Prompt Family | First Pass (tokens/s) | Repeated Pass (tokens/s) | Rate Ratio |
|-------------|------------------------|----------------------------|-----------|
| Explanation | 5.09                   | 14.70                      | **2.89×** |
| Tips        | 3.55                   | 11.66                      | **3.28×** |
| Table       | 3.76                   | 8.66                       | 2.30×     |
| Code        | 7.07                   | 8.05                       | 1.14×     |

> 🔺 整体 decode throughput 提升 **1.80×**（5.68 → 10.24 tokens/s）

#### Dense Layer 优化效果（Table 2）

| Layer | 传统三矩阵调用 | Fused Slices | 时间下降 |
|-------|----------------|--------------|----------|
| 0     | 8.15ms         | 4.51ms       | **44.7%** |
| 1     | 8.11ms         | 4.45ms       | **45.1%** |

> ⏱️ stage-level 计算耗时从 51.9ms → 43.7ms

#### Draft Evaluation 结果（Table 6）

| 设置 | First-draft Agreement |
|------|------------------------|
| Full head, FP32 weights | 66.81% |
| 65,536-token vocab | 64.54% |
| INT4/INT8 weights + 65k vocab | **64.36%** |

> 🔍 **发现**：
> - Vocabulary 截断导致 **-2.271 pp** 下降
> - Quantization 额外造成 **-0.175 pp** 影响

---

### 消融实验结果（Ablation Studies）

#### （1）Windowed Admission vs. Burst Admission
- **burst admission（15 请求集中提交）**：
  - TTFT median: ~31.4s
  - Q_phase: ~16.3 tokens/s
- **windowed admission（八行窗口流控）**：
  - TTFT median: **6.91s**（↓4.5×）
  - Q_phase: **22.52 tokens/s**

> ✅ 表明 admission 控制策略极大改善首 token 延迟

#### （2）Context Length 扩展测试（Table 5 & Figure 7）

| Context (tokens) | TTFT (mean) | Decode Speed (tokens/s) | Code Recovery |
|------------------|-------------|--------------------------|--------------|
| 1k               | 22s         | 4.73                     | ✅ 3/3        |
| 8k               | 2.9min      | 3.32                     | ✅ 3/3        |
| 16k              | 7.8min      | 2.71                     | ✅ 3/3        |
| 32k              | 27.2min     | 1.51                     | ✅ 2/2        |
| 64k              | 109.7min    | 0.82                     | ✅ 2/2        |

> 🔍 **关键发现**：
> - **内存非瓶颈**：64k context 仅占 ~0.5 GiB/机器，远低于可用内存
> - **计算是瓶颈**：TTFT 随 $N$ 呈 $aN + bN^2$ 增长，主因是 **single-threaded CPU attention loop**
> - 最大支持 context 达 **512k positions**（受 free memory margin 限制）

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **超大规模 MoE 模型可在消费级 AI PC 上实现高效 resident inference**
   - 成功部署 975B 参数的 Inkling 模型，激活参数 41B/token；
   - 利用共享内存 + fused iGPU primitives 实现低延迟推理。

2. ✅ **dense 与 sparse 层可通过 all-active slicing 统一加速路径**
   - 复用 fused MoE 算子使 dense 层速度提升近 **2 倍**；
   - 无需额外引入 router 或改变模型结构。

3. ✅ **streaming pipeline 设计有效提升并发服务能力**
   - 最佳吞吐出现在 **88 streams**，达到 **60.29 decode tokens/s**；
   - speculative decoding 在重复请求下带来最高 **3.28× 吞吐提升**。

4. ✅ **context 扩展能力强大，但受限于 CPU attention 实现**
   - 支持 up to 512k context positions；
   - 当前瓶颈是 **单线程 CPU attention loop**，未充分利用多核或 iGPU。

5. ✅ **draft evaluation 应基于真实部署路径的状态**
   - 提出 captured-state replay 方法；
   - 发现 vocabulary size 对 draft accuracy 的影响远大于 weight quantization。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **Attention 计算瓶颈** | 当前 attention loop 为单线程 CPU 实现，无法扩展；成为长 context 下的主要性能墙 |
| **网络带宽限制** | Gigabit Ethernet 成为高并发下的潜在瓶颈，尤其在 prefill 阶段 |
| **缺乏多模态支持** | 当前系统专注于 text generation，未涉及 vision 或 multimodal 输入 |
| **定制化程度高** | 引擎深度绑定 OpenVINO + Intel iGPU，迁移至 AMD/NVIDIA 平台需重构 |

---

### 未来工作方向

1. **iGPU Attention Kernel 开发**
   - 将 attention computation offload 至 Arc GPU，突破 CPU 单核瓶颈；
   - 探索 key/value block parallelism。

2. **更高效的 speculative proposer**
   - 当前 proposer 依赖历史短语匹配和小型 CPU draft model；
   - 可尝试轻量化 MoE draft head 部署于 iGPU。

3. **跨设备 expert load balancing**
   - 当前为静态 partitioning；
   - 动态 routing + expert migration 可进一步提升资源利用率。

4. **开放工具链与 reproducibility**
   - 已开源 artifact（GitHub repo）及完整 trace；
   - 鼓励社区在其 consumer cluster 上复现与改进。

---

> 📌 **总体评价**：  
> 本论文展示了 **client-side distributed inference** 的新前沿——不仅“能跑”，而且“能优”。它不是简单地将 server 技术下放到终端，而是重新思考了 **memory hierarchy、execution model 与 evaluation methodology** 在 shared-CPU-GPU 架构下的组合方式，为未来去中心化 AI infra 提供了重要实践范式。

</details>

---

### 2. [Expert Coupling in MoE Pretraining: Reducing All-to-All Overhead with Correlated Placement and Token Shuffling](https://arxiv.org/abs/2610.09372)

**Authors**: Radha Gulhane, Quentin Anthony, Beren Millidge  
**Category**: cs.CL  
**Published**: 2026-10-08  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.09372v1  

#### Abstract
Mixture-of-Experts (MoE) layers replace the feed-forward block of a Transformer with E expert networks, and each token is routed to k of these experts. Under expert parallelism (EP) the experts are distributed across GPUs, and every MoE layer runs all-to-all collectives in the forward and backward p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Expert Coupling in MoE Pretraining**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
在 **Mixture-of-Experts (MoE)** 模型预训练中，**专家并行（Expert Parallelism, EP）** 导致每个 MoE 层都需要执行两次 **all-to-all collectives**（分发 dispatch 和合并 combine），用于将 token 发送到其被路由到的专家所在 GPU，并返回结果。这些通信开销极大，尤其当 EP 跨越多个节点时，**all-to-all 可占训练步时间的 45–60%**。

传统方法如连续放置（contiguous placement）未利用路由中的结构化模式，导致大量不必要的跨 GPU 通信。

---

### 🚀 提出的新方法与创新思路

本文提出两种无需改变模型架构或路由决策的方法，通过挖掘 **专家选择的相关性** 来减少 all-to-all 通信量：

#### （1）**Correlated Expert Placement + Deduplicating Dispatcher（相关专家放置 + 去重分发器）**
- **核心思想**：观察到在预训练早期，同一层内某些专家对会被频繁共同选中（within-layer correlation）。例如，在 top-2 路由下，仅 0.8% 的专家对被 42% 的 token 同时选择。
- **方法**：
  - 将经常共现的专家放在同一个 GPU 上（correlated placement）；
  - 使用去重分发器（deduplicating dispatcher），每个 token 至多向每个目标 GPU 发送一次，即使它有多个专家在该 GPU 上。
- **优势**：显著提升 **token-expert locality（V）**，降低 all-to-all 数据量和耗时。

#### （2）**Token Shuffling（令牌洗牌）**
- **核心思想**：发现当前层选择的专家可以预测下一層的专家分布（cross-layer correlation）。例如，第 $ l $ 层选中的专家能以高概率预测第 $ l+1 $ 层的目标 GPU。
- **方法**：
  - 在 attention 后的 reduce-scatter 阶段，提前将 token 移动到预计会持有其下一层专家的 GPU；
  - 利用已有的序列并行 collectives（reduce-scatter 和 all-gather）融合实现，几乎无额外开销。
- **适用条件**：要求 TP group 与 EP group 完全一致（即 TP=EP）。

---

### 🔍 相比现有方法的优势

| 方法 | 是否改路由 | 是否需复制专家 | 是否适用于预训练 | 主要优化维度 |
|------|------------|----------------|------------------|---------------|
| SmartMoE [Zhai et al., 2023] | 否 | 是（动态冗余） | 是 | 负载均衡 |
| GRACE-MoE [Han et al., 2025] | 是（限制专家集合） | 是 | 推理/微调 | 内存 + 通信 |
| DeepSeek-V3 Node-limited routing | 是 | 否 | 是 | 减少跨节点通信 |
| **本文方法** | ❌ 否 | ❌ 否 | ✅ 是（从头预训练） | **通信总量 + 局部性** |

> ✅ **不修改路由逻辑、不增加参数或内存开销、完全兼容现有训练流程**。

---

## 2. **核心实验方法和设置**

### 📚 数据集
- **FineWeb-Edu**：教育领域高质量网页文本语料。
- 训练约 **21亿 tokens（2.1B）**，序列长度为 8192，全局 batch size 为 128。

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| 模型结构 | 12-layer MoE Transformer |
| $ d_{\text{model}} $ | 4096 |
| 每层专家数 $ E $ | 128 |
| FFN 宽度 | 1024 |
| Top-k 路由 | top-2 和 top-6 |
| 并行策略 | EP（8–64）、TP（8 或 16），支持 TP=EP 配置 |
| 硬件平台 | 每节点 8 × AMD Instinct MI300X GPU，节点间通过 100 Gb/s RoCE 连接 |
| 软件框架 | 修改版 **Megatron-LM** |

---

### 📊 评估指标
- **All-to-all 时间占比**
- **All-to-all 通信体积（MB/rank/step）**
- **端到端训练步时间（end-to-end step time）**
- **token-expert locality（本地专家比例）**
- **消融配置对比**：
  - **B**: Baseline（连续放置 + 原始 all-to-all）
  - **D**: + Deduplication
  - **P**: + Correlated Placement
  - **S**: + Token Shuffling（仅限 TP=EP）

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据汇总

| 配置 | 方法 | All-to-All 时间下降倍数 | 端到端步时间加速比 |
|------|--------|--------------------------|--------------------|
| EP32, top-2 | P（correlated + dedup） | **1.38×** | **1.14×** |
| EP32, top-6 | P | **1.95×** | **1.25×** |
| TP16 EP16, top-2 | S（+ token shuffling） | **1.74×** | **1.06×** |
| TP16 EP16, top-6 | S | **2.63×** | **1.34×** |

> 💡 最大 **all-to-all 时间减少达 2.63 倍**，最大 **端到端加速达 1.41×**。

---

### 🔬 详细结果分析

#### （1）**Correlated Placement 效果**
- **EP8, top-2**：
  - 原始去重仅减少 6% 行；
  - 加上相关放置后，**去重率达 26%**。
- **EP8, top-6**：
  - 去重率从 25% 提升至 **58%**。
- **通信体积减少**：
  - top-2 EP8：**1.36×**
  - top-6 EP8：**2.36×**

> 👉 相关性越强（如 top-6），增益越大。

#### （2）**Token Shuffling 提升局部性**
| 配置 | 原始本地专家比例 | 使用 shuffling 后 |
|------|------------------|------------------|
| EP8, top-2 | 12.5% | → **59%** |
| EP16, top-2 | 6.3% | → **53%** |
| EP8, top-6 | 12.5% | → **57%** |
| EP16, top-6 | 6.3% | → **46%** |

> ✅ 实现了 **“让数据去找计算”** 的理想状态。

#### （3）**端到端加速有限但稳定**
- 因为 TP 引入了额外非 MoE 开销（如 all-gather），A2A 占比下降，所以整体加速受限。
- 但在 **top-6 + 多节点场景** 下仍可达 **1.34× 步速提升**。

---

### 🔍 消融实验结果

| 方法组合 | All-to-All 时间降幅 | 说明 |
|--------|---------------------|------|
| Baseline | 1× | 参考基准 |
| + Deduplication | 1.03–1.14× | 收益较小 |
| + Correlated Placement | **1.16–1.95×** | 主要贡献来源 |
| + Token Shuffling | **最高达 2.63×** | 在 TP=EP 场景进一步放大收益 |

> ✅ **Correlated placement 是主因，token shuffling 是锦上添花**。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **专家路由具有强相关性**：
   - **层内相关性**：少数专家对被高频共选（如 0.8% 对覆盖 42% token）；
   - **跨层相关性**：前一层专家可有效预测下一层目标 GPU（median predictability >48%）；
   - 这些模式在 **约 10 亿 tokens 后形成并保持稳定**。

2. **少量数据即可建模相关性**：
   - 仅需 **数千 token 的路由记录** 即可构建有效的 placement 和 shuffling 策略；
   - 表明该方法易于部署且鲁棒性强。

3. **无需改动模型即可大幅降本**：
   - 不改变路由机制、不引入专家复制、不影响训练损失；
   - 完全正交于其他优化技术（如负载均衡、专家复制等），可叠加使用。

---

### ⚠️ 方法局限性

| 局限 | 说明 |
|------|------|
| **依赖 TP=EP 架构** | Token shuffling 仅适用于 TP 与 EP 组相同的配置（如 TP8 EP8），无法用于 TP1 场景 |
| **对低 top-k 增益较小** | top-2 改进有限，更适合 high-capacity MoE（如 top-6） |
| **需要离线统计路由** | 需要在训练中期采集路由 trace 来构建 correlation 表，有一定工程成本 |
| **不解决负载倾斜问题** | 仅优化通信总量，而非通信 skew；需结合 SmartMoE 等方法进一步优化 |

---

### 🔮 未来工作方向

1. **扩展至更多并行范式**：
   - 如 Context Parallelism、Pipeline Parallelism 中的应用；
   - 探索更复杂的跨层迁移策略。

2. **在线自适应 placement**：
   - 动态更新专家布局以适应训练过程中路由演化。

3. **应用于 MoE 推理**：
   - 当前工作聚焦预训练，但方法天然适用于推理阶段的延迟优化。

4. **跨模型家族泛化验证**：
   - 验证该相关性是否普遍存在于不同规模、不同结构的 MoE 模型中（如 Mixtral、DeepSeek-MoE 等）。

---

## ✅ 总结

本文揭示了 MoE 预训练中 **专家选择存在强结构性相关性**，并基于此提出了两个高效通信优化方法：

- **Correlated Expert Placement**：利用层内共选模式，集中高频共现专家；
- **Token Shuffling**：利用跨层可预测性，主动移动 token 到目标 GPU。

二者结合可在不改变模型行为的前提下，**将 all-to-all 通信时间减少最多 2.63 倍，端到端训练速度提升高达 1.41 倍**，为大规模 MoE 模型的高效训练提供了新的系统级优化路径。

</details>

---

### 3. [Fast and Memory Efficient Offload Training Framework with Hybrid XPU Computation](https://arxiv.org/abs/2610.09657)

**Authors**: Zhiyi Yao, Zuning Liang, Yuedong Xu, Jin Zhao, Jessie Hui Wang, Tong Li  
**Category**: cs.DC  
**Published**: 2026-10-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.09657v1  

#### Abstract
With the ever-growing size of deep learning models, GPU memory is prone to being insufficient during training. A prominent approach is ZeRO-Offload, which moves the optimizer states to CPU memory and performs parameter update using CPU. However, the deficiencies of ZeRO-Offload include low GPU utili...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast and Memory Efficient Offload Training Framework with Hybrid XPU Computation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
随着深度学习模型规模的持续增长（如 GPT-3、LLaMA 等达到千亿参数级别），**GPU 内存容量成为训练瓶颈**。尽管 ZeRO-Offload 等 offload 技术通过将 optimizer states 移至 CPU 内存缓解内存压力，但仍存在以下问题：
- **GPU 利用率低**：频繁的数据搬运导致 GPU 大量空闲。
- **通信与计算重叠不充分**：前向传播（FP）和反向传播（BP）中的参数/梯度传输无法完全隐藏在计算中。
- **内存卸载策略僵化**：仅固定卸载 optimizer states，缺乏灵活性。

### 提出的新方法与创新思路
本文提出 **MemFerry** —— 一种基于 **Hybrid XPU Computation** 的高效卸载训练框架，其核心创新包括：

#### （1）引入 DHA（Direct Host Access）进行混合执行
- 利用 GPU SMs 可直接访问 CPU 内存的能力（无需先复制到 GPU HBM），实现 **on-GPU 与 DHA 的混合执行模式**。
- 设计三种参数执行模式：
  - **GFGB**（GPU-FP & GPU-BP）：参数加载至 GPU 后全程在 GPU 上计算。
  - **DHA**：FP 和 BP 均通过 DHA 在 CPU 内存中完成。
  - **DFGB**（DHA-FP & GPU-BP）：FP 使用 DHA，BP 前将参数预加载回 GPU 以加速。

#### （2）设计 Execution Scheduler 实现最优调度
- 构建细粒度层调度算法，在 FP 阶段利用 DHA 计算部分层的同时并行加载其他层参数，**消除 FP 通信气泡**。
- 在 BP 前利用空闲 PCIe 带宽将 DHA 参数提前加载回 GPU，减少 BP 时间，**缓解 BP 气泡**。

#### （3）提出 Shadow Model 统一内存抽象
- 提供统一逻辑视图，支持运行时动态切换参数存储位置（CPU 或 GPU），避免冗余拷贝。
- 支持 DFGB 模式下的无缝参数迁移。

#### （4）GO-MemFerry：支持梯度卸载（Gradient Offloading）
- 首次提出使用 DHA 将梯度直接写入 CPU 内存，进一步降低 GPU 显存占用。
- 引入 **Reserve Factor** 控制保留多少梯度在 GPU，其余卸载至 CPU。
- 使用 **Dynamic Programming 算法** 自动选择最优卸载层集合，最小化性能损失。

#### （5）ScaleUp-MemFerry：适配 Scale-up 架构
- 在华为 CloudMatrix384 超节点上扩展 MemFerry。
- 利用 NPU-NPU 高带宽互联作为辅助路径，协助 CPU→NPU 数据传输。
- 设计 **Adaptive Multi-path Dispatcher**，根据数据大小智能分配直连路径与辅助路径，提升有效带宽。

---

## 2. 核心实验方法和设置

### 使用的模型与测试环境
| 类别 | 具体配置 |
|------|--------|
| **硬件平台** | - 单卡：NVIDIA V100 (32GB) / A100 (80GB)<br>- 多卡：8×A100<br>- Scale-up 平台：Huawei CloudMatrix384（含 384 Ascend 910 NPU） |
| **软件栈** | PyTorch 1.9.0, CUDA 11.7, PCIe 3.0 |
| **测试模型** | - BERT-base (110M), BERT-large (340M)<br>- RoBERTa (110M)<br>- GPT-2-medium (355M), GPT-2<br>- Transformer-XL（人工构造 0.7B~13B 不等） |

### 评估指标
- **Per-iteration Time**（单轮迭代时间）
- **GPU Utilization**（GPU 利用率）
- **Peak GPU Memory Usage**（峰值显存占用）
- **CPU-to-Accelerator Transfer Bandwidth**

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **DeepSpeed (ZeRO-Offload)** | 官方实现，主流 baseline |
| **Native-Offload** | 作者自实现的标准 ZeRO-Offload，用于公平比较 |
| **MemFerry** | 本文主框架 |
| **GO-MemFerry** | 支持梯度卸载的变体 |
| **ScaleUp-MemFerry** | 扩展至 scale-up 架构的版本 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **单 GPU 训练速度提升** | MemFerry 较 ZeRO-Offload 最快达 **1.68×** 加速 |
| **平均训练速度提升** | MemFerry 平均提速 **1.32×** |
| **GPU 利用率提升** | 最高提升 **1.27×**（从 63% → 接近 80%） |
| **GPU 显存节省** | MemFerry 最多减少 **61.7%** 显存使用 |
| **支持更大模型** | GO-MemFerry 可训练 **1.52× 更大模型**（相比 DeepSpeed） |
| **梯度卸载显存节省** | GO-MemFerry 平均再降 **20.1%** 显存，最多达 **74.4%** |
| **多 GPU 扩展性** | 在 8 GPU 上仍比 DeepSpeed 快 **28.1%** |
| **Scale-up 性能增益** | 在 CloudMatrix384 上，ScaleUp-MemFerry 比 DeepSpeed 快 **20.7%**，比原始 MemFerry 快 **4.0%** |

### 详细对比结果

#### ✅ 单 GPU 性能对比（Fig. 11–13）
- 在 BERT-base 上，MemFerry 比 DeepSpeed 快 **21.7% ~ 68.7%**。
- 对于大模型（如 GPT-2），提速为 **7.1% ~ 32.1%**，因计算占比更高。
- 当模型超过 2.4B 参数时，DeepSpeed / MemFerry 均 OOM，**只有 GO-MemFerry 可继续训练**。

#### ✅ 显存使用情况（Fig. 15）
- MemFerry 相比 DeepSpeed 平均节省 **40.1%** 显存（小模型）。
- GO-MemFerry 进一步降低 **17.6%** 显存（相较 MemFerry），总降幅达 **41.6%**。

#### ✅ 多 GPU 扩展性（Fig. 17）
- 在 8 GPU 上训练 4.3B 模型：
  - MemFerry 单次迭代耗时 **4.12s**（DeepSpeed 为 5.68s）。
  - 速度提升 **28.1%**，且随 batch size 增加优势更明显（batch=128 时提速 1.21×）。

#### ✅ Scale-up 架构表现（Fig. 18）
- **CPU→NPU 传输带宽**：
  - 16MB 数据：Direct 路径 53 GB/s，Assisted 路径 43 GB/s，Adaptive 达 **86 GB/s**（1.62× 提升）。
- **端到端训练性能**：
  - 使用 13B Transformer-XL 模型，ScaleUp-MemFerry 比 DeepSpeed 快 **20.7%**，比原版 MemFerry 快 **4.0%**。

#### ✅ 消融实验与分析
- **DHA 层选择有效性**：调度器能自动识别适合 DHA 的层（如 Embedding、LayerNorm），因其传输时间 > 计算时间。
- **梯度卸载灵活性验证**（Fig. 14）：
  - GPU 显存使用与 `reserve factor` 成正比，用户可按需调节。
  - 曲线呈凸形，说明 DP 算法优先卸载代价最小的层，实现“最小性能损失换最大显存收益”。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **DHA 是可行且高效的训练加速手段**：虽然 DHA 计算略慢于 on-GPU，但其允许通信与计算并行，总体可显著缩短迭代时间。
2. ✅ **混合执行 + 细粒度调度是关键**：通过 GFGB/DFGB/DHA 模式的组合调度，可有效消除通信气泡，提高 GPU 利用率。
3. ✅ **Shadow Model 实现零拷贝内存管理**：统一抽象简化了跨设备编程复杂性，提升了系统效率。
4. ✅ **梯度也可安全卸载**：首次证明 DHA 可用于梯度生成，结合动态规划实现灵活控制。
5. ✅ **Scale-up 架构下仍有优化空间**：传统 offload 忽视了加速器间高带宽资源，ScaleUp-MemFerry 成功将其转化为数据搬运优势。

### 方法的局限性
- **依赖硬件特性**：DHA 功能需 GPU 支持（如 NVIDIA GPU 的 `cudaHostAlloc`），并非所有 XPU 都具备。
- **PCIe 带宽仍是瓶颈**：尤其在梯度卸载场景下，DHA 写操作受限于 PCIe 带宽，影响性能。
- **调度开销存在**：虽然 profiling 仅一次，但在极端异构模型上可能需要更复杂的建模。
- **未考虑 NVMe 卸载**：当前仅限 CPU 内存，未扩展至 SSD 等二级存储。

### 未来工作方向
- 将 MemFerry 扩展至 **GPU + NVMe 分级卸载** 场景。
- 探索 **自动调优调度器**，适应不同 workload 和硬件拓扑。
- 结合 **模型压缩或稀疏训练**，进一步降低内存需求。
- 推广至更多 XPU 架构（如 AMD GPU、Apple Silicon）以增强通用性。

--- 

> 📌 **总结一句话**：  
> MemFerry 通过融合 DHA 与智能调度，实现了 **更快、更省显存、更灵活** 的 offload 训练范式，不仅超越了 ZeRO-Offload，还为未来 scale-up 架构下的高效训练提供了新路径。

</details>

---

### 4. [Communication-Aware Qubit Placement and Automatic Node-Count Allocation for Distributed State-Vector Simulation](https://arxiv.org/abs/2610.09659)

**Authors**: \'I\~nigo Ar\'ejula-A\'isa, Sergio Iserte, Petter Sand{\aa}s, Ricard S. Raigada-Garc\'ia, Antonio J. Pe\~na  
**Category**: cs.DC  
**Published**: 2026-10-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2610.09659v1  

#### Abstract
Distributed quantum circuit simulation enables the execution of large-scale quantum algorithms by partitioning qubits across multiple computational nodes. However, inefficient qubit placement frequently leads to excessive inter-node communication, severely limiting performance and scalability. This ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Communication-Aware Qubit Placement and Automatic Node-Count Allocation for Distributed State-Vector Simulation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
分布式量子电路模拟（distributed quantum circuit simulation）面临两大瓶颈：
1. **通信开销过大**：由于逻辑 qubit 到物理位置的映射（即 qubit placement）不合理，导致大量跨节点通信（inter-node communication），成为性能瓶颈。
2. **资源分配僵化**：计算节点数量（node count）通常由用户手动设定，缺乏基于电路特征和硬件特性的自动优化策略，容易造成资源浪费或性能不佳。

### 提出的新方法与创新思路
本文提出了一个名为 **QuCSLO** 的中间件框架，集成两个核心优化模块：

1. **Communication-Aware Qubit Placement Optimizer**
   - 将 qubit placement 问题建模为加权图划分问题（weighted graph partitioning）。
   - 节点权重表示作用在该 qubit 上的单目标非对角门（non-diagonal gates）数量，边权重表示双目标非对角门。
   - 使用 **simulated annealing** 搜索最小化“communication cut”的布局，即最小化涉及全局 qubit（global qubits）的非对角门数量。

2. **Predictive Node-Count Efficiency Model**
   - 提出一个预测模型，根据用户指定的性能退化阈值 $T$ 自动选择最优节点数 $k_{\text{eff}}$。
   - 模型基于通信效率（Communication Efficiency, CE）和退化度 $O(k)$，结合静态电路分析（如 $GA$, $cut_{\min}(k)$, $W_{\text{comm}}(k)$）和集群校准参数（$a(k), b(k)$）进行预测。

### 相比现有方法的优势
| 方面 | 现有方法 | 本文方法 |
|------|--------|--------|
| **Placement** | 手工调优、运行时重排序（runtime reordering）、依赖特定模拟器 | **离线优化**，**模拟器无关**（simulator-agnostic），无需修改模拟器内部 |
| **Node Count** | 固定节点数、经验性调整、最大资源申请 | **自动选择**，基于**预测模型**和**用户定义的性能阈值** |
| **整体性** | 多数研究只关注 placement 或 partitioning | **首次联合优化** placement 和 node count，形成完整自动化流程 |

---

## 2. 核心实验方法和设置

### 数据集
- **合成电路（Synthetic Benchmarks）**：共8类，用于控制变量分析不同通信模式的影响：
  - `QAOA-sparse`, `Draper adder`, `1D chain`, `QFT-epcc`, `Comm cliff`, `Cuccaro adder`, `Ising Trotter`, `SWAP routing`
- **真实世界电路（Real-world Circuits）**：来自 **QASMBench** 的 Large-scale 类别，筛选出 28–34 qubit 的电路，如：
  - `qft_n29`, `adder_n28`, `bv_n30`, `vqe_uccsd_n28` 等

### 实验设置
- **平台**：MareNostrum 5 (MN5) 超算，使用最多 64 个节点，每个节点双路 Sapphire Rapids CPU，256GB 内存，NDR InfiniBand 互连。
- **软件栈**：
  - 模拟器：**QuEST**（state-vector simulator）
  - 动态资源管理：**DMR**（支持 MPI 进程动态伸缩）
  - 中间件：自研 **QuCSLO**
- **评估范围**：节点数从 2 到 64，对应全局 qubit 数 $k = 1$ 到 $6$

### 评估指标
- **Analytical Level**：
  - `cut(G)`：通信割（communication cut），衡量潜在通信量
  - `cut_{\text{id}}(k)` vs `cut_{\min}(k)`：对比恒等布局与优化布局的通信量
- **Performance Level**：
  - 执行时间（wall time）
  - 加速比（speedup）：$T_{\text{nolayout}} / T_{\text{layout}}$
  - 退化度 $O(k)$ 与预测准确性
- **Resource Efficiency**：
  - node-hour 成本
  - 是否超过 QuEST 的分布下限（distribution floor）

### 基线方法对比
- **Identity Layout**：默认 qubit 映射，作为 placement 基线
- **Greedy Strategy**：申请最大可用节点数
- **Performance-driven Strategy**：通过预扫描找到最快配置（需多次运行）
- **Conservative Strategy**：申请最小可行节点数

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### (1) 通信割减少（Analytical Results）
- 在合成电路上，优化布局显著降低 `cut`：
  - `QAOA-sparse`：16节点下从 76→16（降幅 79%）
  - `Draper adder`：实现 `cut=0`（完全消除通信）
  - `1D chain`：在 $k=1$ 时 `cut` 从 4→0
- 在 QASMBench 电路上：
  - `bv_n30`：16节点下从 26→8（降幅 69%）
  - `vqe_uccsd_n28`：16节点下从 28,392→20,664（降幅 27%）

#### (2) 性能加速比（Runtime Speedup）
| 电路 | 最大加速比（vs Identity） |
|------|--------------------------|
| `QAOA-sparse` ($N=32$) | **2.27×** @ 64 nodes |
| `bv_n30` ($N=30$) | **2.28×** @ 2 nodes |
| `Draper adder` ($N=32$) | **1.65×** @ 64 nodes |
| 平均加速比 | **>2×** 在高纠缠电路中 |

> 注：QFT 类电路（如 `qft_n29`）因所有 qubit 权重相同，优化无效，加速比 ≈1.00×。

#### (3) 节点数预测准确性
- 预测模型在不同阈值下的准确率：
  | 数据集 | 阈值 $T$ | 准确率 |
  |--------|---------|-------|
  | Synthetic ($N=32$) | 15% / 30% | 88% |
  | QASMBench Large-scale | 15% / 30% | 83% / 92% |
- 模型在 **低于分布下限** 的范围内高度准确，超出后因状态向量复制而失效。

#### (4) 自动资源调整效果（Automatic Resource Adjustment）
以 `bv_n30` 为例（$T=30\%$）：

| 配置 | 节点数 | 执行时间(s) | Wall Time(s) | Cost (node-hour) |
|------|--------|-------------|--------------|------------------|
| Greedy | 64 | 7.116 | 7.116 | 0.1265 |
| Performance-driven | 16 | 2.161 | 2.161 | 0.0096 |
| **Automatic (QuCSLO)** | **2** | **5.526** | **6.901** | **0.0035** |

- **优势**：成本仅为 greedy 的 **1/36**，接近 conservative 成本，同时性能远优于 conservative。
- 以轻微时间代价换取巨大资源节约，实现性能与成本的平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **战略性的 qubit placement 对分布式模拟效率至关重要**，可显著减少通信开销，尤其在高纠缠电路中带来 **>2× 的加速**。
2. **节点数不应盲目最大化**，存在一个“高效节点数” $k_{\text{eff}}$，在满足性能退化约束下最大化资源利用率。
3. **提出的预测模型具有高准确性**（83–92%），可在不运行电路的情况下推荐最优资源配置。
4. **QuCSLO 实现了全流程自动化**：从 OpenQASM 解析 → placement 优化 → node count 推荐 → DMR 动态伸缩 → QuEST 执行，**无需用户干预**。

### 方法的局限性
1. **平台耦合性**：通信模型 $W_{\text{comm}}(k)$ 基于 QuEST 的实现，迁移到其他模拟器（如 Qulacs, Intel-QS）需重新建模。
2. **校准依赖**：参数 $a(k), b(k)$ 需针对目标集群进行校准，虽可通过脚本完成，但仍增加部署复杂度。
3. **模型适用范围**：预测模型仅适用于状态向量仍被分布的情况，一旦超过分布下限（replication regime），模型失效。
4. **优化不保证全局最优**：simulated annealing 是启发式搜索，结果依赖初始温度和迭代次数，无法保证找到全局最优布局。

### 未来工作方向
1. **集成到持久化量子服务**（如 CUNQA），实现虚拟量子处理单元（vQPU）的动态扩缩容。
2. **支持量子资源调度系统**（quantum resource managers），为经典模拟器后端自动提供最优资源配置建议。
3. **扩展至混合 HPC-Quantum 工作流**，结合 circuit cutting 技术，智能路由到 simulator 或 physical QPU。
4. **探索更高效的优化算法**，如机器学习模型替代 simulated annealing，提升布局搜索速度。

--- 

> ✅ **总结**：本文首次将 **qubit placement** 与 **node-count selection** 统一为一个自动化、预测驱动的优化框架，在不修改模拟器的前提下显著提升了分布式量子模拟的效率与资源利用率，为大规模量子算法验证提供了实用工具。

</details>

---

### 5. [OSFP4: Joint Optimization of Diagonal Smoothing and Block Scales for NVFP4 Quantization](https://arxiv.org/abs/2610.08231)

**Authors**: Neriah Ben David, Ori Meir, Or Ordentlich  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.08231v1  

#### Abstract
NVFP4 is an attractive datatype for large language model (LLM) inference, offering compact storage and native tensor-core acceleration. However, preserving accuracy using NVFP4 requires careful quantization. In this work we develop a novel quantization scheme called Optimized Smoothing and Scaling f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：OSFP4: Joint Optimization of Diagonal Smoothing and Block Scales for NVFP4 Quantization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **NVFP4**（一种用于大语言模型推理的低比特浮点格式）在量化过程中因精度损失导致模型准确率下降的问题。尽管NVFP4具有存储紧凑和硬件加速优势，但其有限的动态范围（尤其是E2M1格式）容易导致量化误差显著，尤其是在权重和激活值分布不均衡时。

传统方法如 absmax scaling 或简单的对角缩放（如SmoothQuant）在FP4场景下效果有限，因为FP4的误差特性不同于INT格式，且现有方法未充分联合优化平滑变换（smoothing）与块尺度（block scales）。

---

### 提出了什么新方法或新思路
作者提出了一种名为 **OSFP4**（Optimized Smoothing and Scaling for NVFP4）的新量化方案，其核心创新在于：

- **联合优化对角平滑矩阵 $A$ 和块尺度（block scales）**：  
  通过一个可微的、基于乘法抖动（multiplicatively dithered）的FP4量化器来近似真实确定性量化器的误差，从而构建一个平滑的损失函数，支持端到端联合优化。
  
- **考虑实际量化方式（RTN vs SIC）设计不同的损失目标**：  
  明确区分 Round-to-Nearest (RTN) 和 Successive Interference Cancellation (SIC) 两种权重量化策略，并为每种设计对应的优化目标函数，使平滑参数更适配最终的量化流程。

- **两阶段优化框架**：
  1. **连续联合优化**：使用抖动量化器对 $A$ 和初始块尺度进行联合优化；
  2. **离散尺度选择**：固定 $A$ 后，在E4M3网格上精细搜索最优的确定性块尺度。

- **数学建模上的突破**：引入了一个新的归一化均方误差函数 $\phi(x)$ 来刻画FP4量化误差，并证明其最小值出现在区间 $[7/4, 7]$ 内，指导优化过程将输入值“拉”入该低误差区域。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **准确性** | 在多个任务上达到当前最高平均准确率，优于 GPTQ、MR-GPTQ、SOAR、H-Scale 等主流方法。 |
| **效率保留** | 保持约 **94–97%** 的厂商原生NVFP4 prefill吞吐量，仅轻微牺牲速度换取显著精度提升。 |
| **通用性与兼容性** | 支持 W4A4 和 W4A16 配置；提供与 vLLM 兼容的插件和检查点导出工具。 |
| **理论支撑强** | 基于高比特率量化理论（high-rate quantization theory），并通过引理证明抖动量化器可作为确定性量化的有效代理。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **FineWeb-Edu**：用于校准（calibration），采样 1024 个长度为 2048 的序列。
- **WikiText-2**：用于评估 **perplexity**。
- **多任务基准测试集**：
  - MMLU-CoT
  - GSM8K
  - HellaSwag
  - WinoGrande
  - C-Eval, LiveBench, ARC-Challenge, BBH, GPQA Diamond（部分扩展模型）

---

### 实验设置和评估指标

#### 模型
- 主要模型：**Llama-3.1-8B-Instruct**
- 扩展验证：Qwen3-8B, Qwen3-30B-A3B-Instruct, Gemma-4-31B-IT, Llama-3-8B

#### 量化配置
- **NVFP4 格式**：E2M1 值 + 每16通道一个 E4M3 块尺度（group size = 16）
- **激活处理**：
  - W4A4：激活也量化为FP4
  - W4A16：激活保留为BF16（weight-only）
- **默认激活缩放**：absmax-to-6（也可替换为 ScaleSweep 等在线搜索）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Average Accuracy (%)** | 多任务平均得分（MMLU-CoT, GSM8K等） |
| **Perplexity ↓** | WikiText-2 上的语言建模困惑度 |
| **Recovery %** | 相对于 BF16 基线的性能恢复比例 |
| **Throughput (tokens/s)** | 推理吞吐量（prefill 和 decode 阶段） |

---

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **RTN (AbsMax)** | 基线 | 最基础的 round-to-nearest + absmax scaling |
| **GPTQ (FP-Quant)** | 权重补偿 | 使用二阶信息修正未量化权重 |
| **MR-GPTQ** | 结构优化 | 引入块内Hadamard变换和列排序 |
| **SOAR** | 尺度优化 | 联合优化全局与块尺度，解耦编解码尺度 |
| **H-Scale** | 统计驱动 | 利用Hessian或激活二阶矩加权重建误差选尺度 |
| **ScaleSweep / 4over6** | 尺度搜索 | 在absmax附近搜索最优E4M3尺度 |
| **NVIDIA released checkpoint** | 商业实现 | 官方发布的NVFP4模型，非可控复现 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Llama-3.1-8B-Instruct）

#### ✅ 多任务平均准确率（Table 1 & Table 3）
| 方法 | 配置 | 平均准确率 (%) | Recovery % |
|------|------|----------------|------------|
| BF16 baseline | — | 79.22 | 100.00 |
| **OSFP4 W-SIC/X-RTN** | W4A4 | **77.03** | **97.23** |
| GPTQ (FP-Quant) | W4A4 | 76.34 | 96.35 |
| MR-GPTQ | W4A4 | 76.18 | 96.16 |
| SOAR | W4A4 | 76.60 | 96.69 |
| H-Scale (our ext.) | W4A4 | 76.67 | 96.78 |
| **OSFP4 W-SIC** | W4A16 | **78.42** | **98.98** |
| H-Scale | W4A16 | 78.00 | 98.45 |

> 🔍 **结论**：OSFP4 在 W4A4 和 W4A16 下均取得最佳性能。

#### ✅ 困惑度（WikiText-2, Table 2）
| 方法 | W4A16 PPL ↓ | W4A4 PPL ↓ |
|------|-------------|-----------|
| BF16 | 7.22 | 7.22 |
| Absmax RTN | 7.55 | 7.88 |
| OSFP4 W-RTN | 7.42 | 7.77 |
| **OSFP4 W-SIC** | **7.39** | **7.67** |

> 📉 使用 SIC + 优化平滑进一步降低PPL。

#### ✅ 在线激活尺度搜索增强（Table 1）
| 方法 | 准确率 (%) |
|------|----------|
| ScaleSweepMSE + GPTQ | 76.69 |
| **OSFP4 W-SIC/X-ScaleSweepMSE** | **77.09** |

> 即便与其他动态优化结合，OSFP4仍领先。

---

### 消融实验结果（Table 10, Appendix H）

分析各组件对 MatMul 输出误差的影响（相对 MSE 比值越小越好）：

| 方法 | q | k | v | o | gate | up | down | PPL |
|------|----|----|----|----|------|-----|-----|-----|
| RTN (W4A16) | 10.27 | 8.68 | 14.30 | 12.59 | 12.22 | 12.64 | 12.52 | 7.55 |
| W-RTN (ours) | 5.29 | 4.42 | 8.53 | 8.16 | 8.48 | 8.95 | 8.10 | 7.42 |
| **W-SIC (ours)** | **3.63** | **2.89** | **6.08** | **4.02** | **6.45** | **6.86** | **6.29** | **7.39** |

> 🔧 **发现**：
> - 对角平滑（A）显著降低所有投影层的误差；
> - SIC 进一步大幅压缩误差，尤其在 `q`, `k`, `v` 等注意力相关层；
> - 联合优化带来系统性改进。

---

## 4. 关键结论和发现

### 主要发现
1. **对角平滑在NVFP4中依然有效**：虽然FP格式动态范围大，但由于E2M1动态范围受限，合理调整输入分布（通过 $A$）能显著减少饱和与零化现象。
2. **必须联合优化 $A$ 与 block scales**：单独优化任一部分无法达到最优，二者存在强耦合关系。
3. **SIC 比 RTN 更受益于平滑优化**：SIC依赖前序坐标的预测残差，因此输入分布的均衡性对其影响更大。
4. **抖动量化器是有效的代理损失函数**：其期望误差与真实量化高度相关，且具备良好可微性，便于优化。
5. **最终确定性尺度选择至关重要**：即使连续优化后，再在E4M3网格上精细搜索仍可将误差减半。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **额外运行时代价** | 当无法融合进 LayerNorm/RMSNorm 时，需显式执行对角变换 $A$，增加少量计算开销。 |
| **仅适用于分组量化结构** | 当前设计依赖于 group size = 16 的 NVFP4 分块机制，难以直接推广至其他粒度。 |
| **MoE模型支持有限** | 实验中未完全实现专家路由感知的激活缩放路径（见Qwen3-30B部分）。 |
| **依赖校准数据质量** | 性能受 calibration set 分布影响较大，极端分布可能导致 $A$ 不鲁棒。 |

---

### 未来工作方向
1. **探索非对角变换的高效实现**：如局部旋转（local rotation）或稀疏变换，在保持精度增益的同时控制延迟。
2. **自适应在线平滑**：研究能否在推理时根据输入动态调整 $A$。
3. **扩展至其他低比特格式**：如MXFP4、E3M2等，验证方法泛化能力。
4. **与训练后微调（PTQ + LoRA）结合**：探索量化与轻量微调的协同优化。
5. **硬件层面融合优化**：推动GPU kernel 支持隐式 $A$ 变换，彻底消除运行时开销。

---

> 💡 **总结一句话**：  
> **OSFP4 通过联合优化 diagonal smoothing 与 block scales，首次将抖动量化思想引入 NVFP4，实现了精度与效率的最佳平衡，在多种LLM上刷新了W4A4/W4A16量化记录。**

</details>

---

### 6. [GAMEGO: Training Game-Dev Agents with Synthetic Trajectories Anchored in Real-World Assets](https://arxiv.org/abs/2610.06910)

**Authors**: Haoyue Yang, Jingyao Li, Zhengfan Wu, Jing Liu, Xuanle Zhao, Kang Liu  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.06910v1  

#### Abstract
Recent advances in Large Language Models (LLMs) have demonstrated remarkable capabilities in web front-end execution, with browser-based game generation emerging as a particularly prominent frontier. While previous efforts frequently rely on complex multi-turn workflows or focus on static game evalu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GAMEGO: Training Game-Dev Agents with Synthetic Trajectories Anchored in Real-World Assets

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前基于 **Large Language Models (LLMs)** 的游戏生成系统面临以下核心挑战：

- **输入查询过于稀疏**：用户仅提供简短的游戏概念（如“做一个平台跳跃游戏”），导致编码代理（coding agent）必须自行填补大量未指定的设计细节。
- **生成结果不完整、不可玩**：由于缺乏明确指导，生成的游戏常出现机制不完整、流程断裂、视觉表现差等问题。
- **过度设计 vs. 设计自由的权衡难题**：提供完整的 **Product Requirements Document (PRD)** 虽能提升完整性，但会限制模型创造力并因提示过长而降低指令遵循能力。

---

### 🚀 提出的新方法与创新思路

论文提出 **GameGo** 框架，其核心是 **“先扩展后压缩”** 的两阶段策略：

#### （1）**PRD 扩展（Specification Expansion）**
- 从真实世界游戏种子（如 Steam 页面、小游戏平台描述）出发，通过多阶段规划生成结构化的 **PRD**。
- 包含三个阶段：
  - **任务定义（Task Definition）**：确定核心玩法类别、渲染格式（2D/2.5D/3D）、控制方式等。
  - **可玩蓝图（Playable Blueprinting）**：构建状态机、摄像机行为、失败恢复机制、关卡流程等。
  - **资产锚定（Asset Grounding）**：为角色、场景、UI 等元素制定明确的资源合同（Asset Contracts），确保视觉实现一致性。

#### （2）**任务自适应查询压缩（Task-Adaptive Query Compression）**
- 将完整的 PRD 压缩为高密度、简洁的任务查询（compact query），保留关键约束，释放非核心设计自由。
- 引入 **关系密度增益（Relation Density Gain, $G_s = R_s / r_s$）** 指标衡量信息效率：
  - $R_s$: 保留的关键条款比例
  - $r_s$: 保留的 token 长度比例
- 实现 **平均 3.79× 的密度增益**，在更短文本中传递更多有效信息。

> 🔑 创新点总结：
> - **首次将工业级游戏开发流程形式化为 LLM 可执行的合成轨迹生成框架**。
> - **动态压缩机制显式建模“执行自由度”**，平衡规范性与创造性。
> - 所有数据、代码、模型均开源 → [GitHub: Haoyue-Yang/GameGo](https://github.com/Haoyue-Yang/GameGo)

---

### ⚖️ 相比现有方法的优势

| 对比维度 | 传统方法 | GameGo |
|--------|--------|-------|
| 输入质量 | 原始用户查询（稀疏） | 经过结构化增强的高密度查询 |
| 数据来源 | 多轮交互或静态基准 | 单轮端到端合成轨迹（anchored in real-world assets） |
| 设计自由度 | 过低（全PRD）或过高（无约束） | 动态调节，保留核心机制，开放次要选择 |
| 可扩展性 | 依赖人工标注或复杂反馈循环 | 完全自动化的轨迹合成流水线 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

#### （1）**GameGoData**
- 规模：**55,060 条开发轨迹**
- 内容：涵盖 **2D、2.5D、3D** 游戏，覆盖 **20 种玩法类型**
- 构成：每条轨迹包含完整的 agent 交互日志（tool calls、responses）、最终可运行游戏 artifact
- 来源：由前沿模型在沙盒环境中执行任务生成

#### （2）**GameGoBench（评测基准）**
- 规模：**124 个多样化游戏生成任务**
- 分布：按渲染格式分层（2D: 47, 2.5D: 22, 3D: 55）
- 难度分级：Easy / Medium / Hard
- 严格去污染：训练与测试种子完全隔离，避免数据泄露

---

### 🧪 实验设置与评估指标

#### （1）训练设置
- **基础模型**：
  - `Qwen3.5-27B`
  - `Qwen3.8-27B`
- **训练方式**：Supervised Fine-Tuning (SFT)，训练一个 epoch
- **损失函数**：response-masked objective（只对 agent 的思考、响应、工具调用计算损失）
- **上下文长度**：256K tokens
- **Batch Size**：global batch size = 8

#### （2）推理环境
- 使用 **Code Arena 框架** 的沙盒环境
- 工具集：文件编辑、项目构建 (`build_project`)、shell 执行等
- 最大交互步数：100 步

---

### 📈 评估指标

#### 自动化评估（Claude Sonnet 4.6 作为 verifier）
| 指标 | 含义 |
|------|------|
| **Execution Rate (Exec.)** | 成功渲染页面的比例 |
| **Requirements (Req.)** | 功能保真度（四层级：核心概念 → 输入响应 → 游戏循环 → 查询匹配） |
| **Quality** | 视觉质量（美术风格、场景构图、UI 整合） |
| **Overall** | Req. 与 Quality 的平均值 |

#### 人类评估（Human Evaluation）
- **双盲配对比较**：6 名独立专家对生成的游戏进行偏好判断（胜/负/平）
- 偏好得分：$ \text{Preference} = W / (W + L) $

---

### 🆚 基线方法对比

#### （1）前沿大模型（Frontier Models）
- Claude Opus 5
- GPT-5.6
- DeepSeek-V4-Pro
- GLM-5.3
- Qwen3.8-Max
- Kimi-K3

#### （2）基线与消融对照
- `Qwen3.5-27B` / `Qwen3.8-27B`（原始 base model）
- 不同查询形式对比：
  - **Direct Query**：直接来自种子的原始查询
  - **Full PRD**：完整的需求文档
  - **Task-Specific Query**：本文提出的压缩查询

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Table 1）

| Model | ArtifactsBench-G (Overall) | CookieBench-G (Overall) | GameGoBench (Overall) |
|-------|----------------------------|--------------------------|------------------------|
| **Claude Opus 5** | 72.76 | 83.68 | 61.84 |
| **GPT-5.6**       | 63.92 | 75.36 | 52.78 |
| **GLM-5.3**       | 68.63 | 79.99 | 59.81 |
| **Qwen3.8-Max**   | 66.92 | 85.96 | 58.35 |
| **Qwen3.5-27B**   | 46.31 | 41.35 | 34.00 |
| **GameGoCoder 3.5** | **57.34** (+11.03) | **55.79** (+14.44) | **51.85** (+17.85) |
| **Qwen3.8-27B**   | 59.87 | 77.07 | 52.53 |
| **GameGoCoder 3.8** | **61.94** (+2.07) | **80.68** (+3.61) | **55.29** (+2.76) |

> ✅ **GameGoCoder 在所有基准上显著优于对应 base model，并接近甚至超越部分 frontier models**

---

### 👥 人类偏好评估（Figure 5 & Table 2）

#### （1）模型间比较（vs. Base Models）
- **GameGoCoder 3.5** vs. `Qwen3.5-27B`：明显更优
- **GameGoCoder 3.8** vs. `Qwen3.8-27B`：显著优势
- 并在多个任务上 **优于 GPT-5.6、DeepSeek-V4-Pro 等**

#### （2）不同查询形式的人类偏好（Table 2）

| 对比项 | 偏好率（Preference） |
|--------|------------------|
| **Full PRD vs. Direct Query** | 69.0% |
| **Task-Specific vs. Full PRD** | **85.7%** |
| **Task-Specific vs. Direct Query** | **86.6%** |

> 💡 **压缩后的高密度查询生成的游戏被人类显著偏好，验证了“信息密度”优于“信息总量”**

---

### 🔍 消融实验结果

#### （1）查询构造消融（Query Construction Ablation）

| 指标 | 数值 |
|------|-----|
| 中位压缩比（vs. Full PRD） | **44.2×** |
| 排除资产附录后压缩比 | 21.6× |
| **中位关系密度增益 $G_s$** | **3.79×** |
| 人类偏好（Compact vs. Full PRD） | 85.7% |
| 人类偏好（Compact vs. Direct） | 86.6% |

> ✅ 密度增益越高，生成质量越好；且压缩查询在困难任务中反而更长，体现“复杂度自适应分配”

#### （2）模型训练消融（Model Ablation）

- 使用未经指令微调的基础 MoE 模型（`Qwen3.5-35B-A3B-Base`）在 GameGoData 上训练 6 轮：
  - **Epoch 2 检查点已优于官方 post-trained checkpoint**（偏好率 66.4%）
  - **Epoch 6 达到 80.7% 偏好率**
- 表明 **GameGoData 具备强学习信号，即使从零开始也能有效训练**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **结构化需求工程显著提升游戏生成质量**  
   → 将真实游戏种子转化为 PRD 可系统化解决“模糊查询 → 不完整实现”的问题。

2. **任务自适应压缩优于全量 PRD 或原始查询**  
   → “少即是多”：通过保留关键约束、释放设计自由，实现了更高的人类偏好和自动化评分。

3. **合成轨迹可用于高效训练小型模型**  
   → 经 GameGoData 微调的小参数模型（如 Qwen-27B）可媲美甚至超越更大规模的 frontier models。

4. **信息密度（Relation Density）是衡量任务表示质量的有效指标**  
   → $G_s > 1$ 的任务普遍获得更高评价，支持了“高密度提示更优”的假设。

---

### ⚠️ 局限性

1. **依赖高质量 PRD 生成器**  
   → 当前 PRD 由 LLM 自动生成，存在错误传播风险，尚未完全自动化验证闭环。

2. **仅适用于前端网页游戏（React 模板）**  
   → 当前框架聚焦于浏览器端小游戏，未扩展至 Unity/Godot 等专业引擎。

3. **视觉资产仍依赖生成而非真实素材库**  
   → 虽然锚定真实资产特征，但实际图像仍由 procedural generation 或 text-to-image 生成。

4. **压缩策略尚未完全可解释**  
   → 哪些条款该保留/舍弃仍依赖 heuristics 和 LLM 判断，缺乏理论最优解。

---

### 🔮 未来工作方向

1. **扩展至专业游戏引擎（如 Godot、Unity）**
2. **引入真实美术资源数据库进行资产绑定**
3. **结合强化学习优化长期可玩性和趣味性**
4. **探索多智能体协作开发完整游戏项目**
5. **建立开放社区驱动的游戏种子众包平台**

---

> 📌 **一句话总结**：  
> **GameGo 通过“现实锚定 + 结构扩展 + 动态压缩”的范式，实现了高质量、可扩展的游戏开发代理训练，证明了结构化合成数据在复杂软件生成任务中的巨大潜力。**

</details>

---

### 7. [Denoising Blocks, Not Tokens: Efficient Compressed Continuous Diffusion with Branching Token Realization](https://arxiv.org/abs/2610.09311)

**Authors**: Xinsong Feng, Peng Du, Zhizhuo Yang, Daniel M. Bikel, Jiayun Wang, Haipeng Chen  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.09311v1  

#### Abstract
Diffusion language models (DLMs) generate text through iterative parallel refinement, offering the potential for higher throughput than autoregressive (AR) decoding. However, most DLMs still maintain one generative state per token, so every denoising step processes a state sequence as long as the ou...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Denoising Blocks, Not Tokens: Efficient Compressed Continuous Diffusion with Branching Token Realization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Diffusion Language Models (DLMs)** 虽然通过并行生成缓解了传统 **Autoregressive (AR)** 模型的序列生成瓶颈，但大多数 DLMs 仍为每个 token 维持一个生成状态，导致每一步去噪操作都需处理与输出长度相当的状态序列，限制了吞吐量提升。

此外，连续空间中的压缩表示虽能减少状态数量，但若解码过程仍是逐 token 自回归，则计算优势会被 token-level 的串行瓶颈抵消。

### 提出的新方法：Branching Latent Diffusion (BLD)
作者提出 **Branching Latent Diffusion (BLD)**，一种基于压缩连续潜变量的高效扩散语言模型框架，其核心思想是：

- **从 token-level 扩散转向 block-level 扩散**：将长文本序列（如 1024 tokens）压缩为少量 block latents（如 64 个），实现 **16× 序列长度压缩**。
- **分组潜变量生成（Grouped Latent Generation）**：不一次性并行生成所有潜变量，而是按组（groups）逐步生成，每组条件依赖于已生成的前缀，以缓解强压缩下联合分布建模困难。
- **分支化 token 实现（Branching Token Realization）**：每个 block latent 并行地由一个局部 AR 分支解码成多个 tokens，从而在保持局部流畅性的同时避免全局串行解码。

### 相比现有方法的优势
- **显著降低 FLOPs 和延迟**：相比同规模的连续 DLM（如 ELF-L），生成 FLOPs 下降 **超过 80×**，吞吐量提升 **6× 以上**。
- **优于 AR 模型**：相比 AR baseline，吞吐量高 **6×+**，单序列延迟低 **4×+**。
- **端到端效率提升**：通过压缩 + 并行解码，真正实现了从 latent 生成到 token 输出的全流程高效化，而非仅优化某一部分。

---

## 2. 核心实验方法和设置

### 数据集
- 主要训练与评估数据：**OpenWebText**
- 分词器：**T5 tokenizer**（|V| ≈ 32k）
- 输入长度：固定为 **L = 1024 tokens**
- 测试集：保留最后 10,000 篇文档用于无条件生成评估

### 实验设置
- **压缩比**：`r = L/K = 1024 / 64 = 16×`，即每 16 个 tokens 映射为 1 个 block latent
- **潜变量维度**：`d = 512`
- **模型结构**：
  - Fusion Encoder（36M 参数）：双向 Transformer + 动态分块（dynamic chunking）+ 注意力池化
  - Branching Decoder（83M 参数）：因果 Transformer，支持跨 latent 上下文共享
  - Grouped Prior（511M 参数）：基于 rectified flow 的 latent diffusion model
- **采样策略**：
  - Latent 分组大小 `G ∈ {1, 4, 8, 16}`
  - 每组使用 50 步 SDE 去噪
  - 最终生成顺序：先完成全部 latent 生成 → 再并行解码各 token 分支

### 评估指标
| 类别 | 指标 |
|------|------|
| **生成质量** | Gen-PPL（GPT-2 Large 打分）、Entropy（多样性）、MAUVE（与真实文本分布对齐度） |
| **计算成本** | TFLOPs（总生成浮点运算量） |
| **运行时性能** | 单序列延迟（latency @ batch=1）、批处理吞吐量（throughput @ batch=32） |

### 基线方法对比
| 类型 | 基线模型 |
|------|--------|
| **连续 DLM** | ELF-B/M/L（Hu et al., 2026） |
| **离散 DLM** | MDLM（Sahoo et al., 2024）、BD3-LM（Arriola et al., 2025） |
| **自回归模型** | 自研 AR 模型（730M 参数，nucleus sampling） |
| **其他潜变量模型** | Cosmos（Meshchaninov et al., 2025） |

> 注：部分基线使用不同 tokenizer 或更大训练预算，因此比较侧重趋势而非绝对公平。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1 & 2）

| 模型 | Gen-PPL | Entropy | MAUVE | TFLOPs | Latency (s) | Throughput (doc/s) |
|------|---------|---------|-------|--------|-------------|---------------------|
| AR | 26.5 | 5.37 | 0.92 | 1.6 | 12.57 | 1.5 |
| ELF-L | 23.9 | 5.29 | 0.91 | 96.4 | 2.25 | 1.4 |
| **BLD (G=8)** | **26.6** | **5.17** | **0.82** | **1.1** | **2.67** | **9.4** |
| **BLD (G=K)** | 24.6 | 5.08 | 0.77 | 4.8 | 1.16 | 13.5 |

#### 性能对比总结
- **FLOPs 减少**：BLD (G=8) 相比 ELF-L 减少 **88× FLOPs**（96.4 → 1.1 TFLOPs）
- **吞吐量提升**：
  - 比 ELF-L 高 **6.7×**（1.4 → 9.4 doc/s）
  - 比 AR 高 **6.3×**
- **延迟改善**：
  - 比 AR 低 **4.7×**（12.57s → 2.67s）
  - 略高于 ELF-L（因引入 latent-level 分组串行）

### 消融实验结果（Table 3 & 4）

#### 表 3：分组大小 $G$ 的影响（控制变量）
| G | Gen-PPL | Entropy | Latency (s) | TFLOPs |
|----|--------|--------|------------|--------|
| 1 | 31.7 | 5.22 | 4.93 | 2.23 |
| 4 | 24.7 | 5.21 | 3.52 | 1.31 |
| **8** | **26.6** | **5.17** | **1.84** | **1.10** |
| 16 | 27.7 | 5.03 | 1.21 | 1.00 |

> ✅ **G=8 是最佳平衡点**：兼顾质量与效率；进一步增大 G 导致 Gen-PPL 上升、Entropy 下降。

#### 表 4：Decoder Adaptation 消融
| Decoder 状态 | Gen-PPL | MAUVE |
|--------------|--------|-------|
| 未适配（Not adapted） | 127 | 0.67 |
| 适配于 G=K prior | 26.6 | 0.82 |

> 🔍 **Decoder 必须进行 prior 输出适应训练**，否则无法正确解码生成的 latent，质量急剧下降。

---

## 4. 关键结论和发现

### 主要发现
1. **压缩 latent diffusion 可大幅提高长序列生成效率**  
   将 1024-token 序列压缩为 64 个 block latents，并结合并行分支解码，可在几乎不损失局部 fluency 的前提下，实现 **超 80× FLOPs 下降** 和 **6× 吞吐提升**。

2. **完全并行 latent 生成在强压缩下效果差**  
   全并行去噪（G=K）虽然速度快，但难以维持 long-range coherence，表现为更低的 MAUVE 和 entropy。

3. **分组生成（Grouped Generation）有效缓解建模难度**  
   引入受控的串行性（autoregressive across groups），使模型能利用 clean prefix context，显著提升生成一致性。

4. **target-specific conditioning 至关重要**  
   若所有目标 latent 共享同一 prefix summary，易在 group boundary 处出现重复或崩溃；改为每个位置独立访问 prefix 后，distinctness 提升、loop rate 下降。

5. **decoder adaptation 是必要环节**  
   decoder 在训练时看到的是 encoder latents，而推理时接收的是 prior 生成的 latents，存在分布偏移，必须通过 adaptation 缓解。

### 局限性
1. **长距离连贯性仍具挑战**  
   尽管有 grouped generation，BLD 的 MAUVE 仍低于最强 AR 和 ELF 模型，说明 global semantic structure 建模不足。
   
2. **缺乏人类评估与任务级评测**  
   当前仅依赖 Gen-PPL、MAUVE 等自动指标，缺少 human evaluation 或 downstream task performance 支持。

3. **训练-推理不匹配问题依然存在**  
   如 rollout training 未能完全解决 generated context vs real context 差异。

4. **当前仅验证单一尺度与长度**  
   所有实验基于 L=1024, r=16，尚未探索更长序列或不同压缩比下的表现。

### 未来工作方向
- 探索更强的 latent-space prior 结构以增强 long-range modeling
- 引入层次化 latent 结构（如 multi-scale）以支持超长文本生成
- 设计更鲁棒的 conditioning 机制，减少暴露偏差（exposure bias）
- 扩展至 instruction-following、dialogue 等实际应用场景
- 进行训练预算对齐的公平比较（尤其是与 ELF/Cosmos）

---

> 📌 **总体评价**：  
> BLD 成功展示了 **“denoising blocks, not tokens”** 的可行性与巨大潜力，首次将 latent compression 与 branching token realization 结合，在保证一定生成质量的前提下，实现了 DLM 在长序列生成上的 **实质性效率突破**，为未来高效大模型推理提供了新范式。

</details>

---

### 8. [Layerwise Error Attribution for Fast and Robust Mixed-Precision Post-Training Quantization](https://arxiv.org/abs/2610.09877)

**Authors**: Samy Houache (IMB, UB), Yann Traonmilin (IMB, UB), Jean-Fran\c{c}ois Aujol (UB, IMB)  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.09877v1  

#### Abstract
Mixed-precision post-training quantization is a network compression method that assigns bits layer by layer, under a global memory budget using a small calibration set. The main difficulties are to overcome the combinatorial nature of the allocation problem and to manage the sensitivity to small, po...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Layerwise Error Attribution for Fast and Robust Mixed-Precision Post-Training Quantization**

---

## **1. 主要贡献和创新点**

### **解决的问题**
该论文针对**混合精度后训练量化（Mixed-Precision Post-Training Quantization, PTQ）**中的两个核心挑战：
- **组合爆炸问题**：在多层网络中为每一层分配不同 bit-width 的组合空间巨大，穷举搜索不可行。
- **校准数据敏感性问题**：现有方法对小规模、可能被噪声或损坏影响的校准集（calibration set）高度敏感，导致量化配置不稳定。

目标是设计一种**快速且鲁棒**的 bit-width 分配算法，在有限内存预算下最大化模型性能。

---

### **提出的新方法与新思路**
作者提出了一种基于**逐层误差归因（Layerwise Error Attribution）**的混合精度 PTQ 框架，其核心创新包括：

#### ✅ **理论层面：概率化逐层误差分析**
- 推导了**混合精度量化的概率化上界**，将总误差分解为：
  - **前向传播误差（Propagated Error）**：来自前面层的累积误差。
  - **局部扰动误差（Local Perturbation）**：当前层量化引入的误差。
- 通过定义随机放大比（random amplification ratios）$X_l$, $Y_l$，保留了每层误差来源的信息，从而支持更精细的 bit-width 决策。

#### ✅ **算法层面：可分离、无需求解器的分配算法**
- 利用上述理论中的**局部扰动项**构建一个**可分离的评分函数（separable score）**：
  $$
  s_g^{\text{score}}(b) = \mu_{g,b} + \frac{\sigma_{g,b}}{\sqrt{\delta_{\text{score}}}}
  $$
  其中 $\mu$ 和 $\sigma$ 是局部量化误差 $P_{g,b}(x)$ 在校准集上的均值与标准差。
- 采用**贪心策略**逐步降低各 block 的 bit-width，选择“单位节省比特带来的评分增幅最小”的 block 进行降级。
- 整个过程**无需外部优化求解器（如 ILP/IP）**，也**不依赖成对交互建模**。

---

### **相比现有方法的优势**
| 维度 | 本文方法 | 现有主流方法（如 HAWQ, CLADO, AIMET） |
|------|--------|----------------------------|
| **速度** | ⚡️ 极快（28× ~ 2,570× 加速） | ❌ 需反复推理或调用复杂优化器 |
| **鲁棒性** | ✅ 对 corrupted calibration 数据稳定 | ❌ 性能剧烈下降（如 CLADO 下降超 5 dB） |
| **实现复杂度** | ✅ 仅需一次 FP32 前传 + 贪心规则 | ❌ 多阶段搜索、Pareto 优化、学习策略等 |
| **通用性** | ✅ 可迁移到 Diffusion Models | ✅ 已验证有效 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **图像去噪任务**：  
  - **CelebA-HQ**（256×256 彩色人脸图像）
  - **BSDS500**（自然图像，用于泛化性测试）
- **图像生成任务（Latent Diffusion）**：  
  - **CelebA-HQ 256×256** 上预训练的 CompVis Latent Diffusion Model

---

### **实验设置与评估指标**

#### **模型架构**
- **DRUNet**：32.64M 参数，64 个卷积层，划分为 36 个 allocation blocks。
- **Latent Diffusion U-Net**：123 个权重张量，分组为 48 个 blocks。

#### **量化设置**
- **权重量化（Weight-only quantization）**，中间激活保持 FP32。
- 支持候选 bit-width：{3,4,5,6,7,8}（主实验），部分对比限制为 {3,4,8}。
- 内存预算以平均 bit-per-weight 表示（如 $B=4$）。

#### **评估指标**
| 指标 | 含义 |
|------|------|
| **PSNR (dB)** | 图像重建质量（去噪任务） |
| **FID** | 生成图像分布质量（Diffusion 任务） |
| **Paired PSNR** | 量化输出 vs. FP32 输出的逐像素保真度 |
| **Selection Time** | bit-width 配置耗时（不含应用量化） |
| **Spearman Correlation** | 不同校准条件下 bit-width 排序稳定性 |

#### **校准数据损坏模拟**
仅在校准输入上施加噪声，用于测试鲁棒性：
- **Shared mask corruption**：RGB 通道共享二值掩码，$p=0.5$
- **Per-channel corruption**：各通道独立掩码，$p=0.9$

---

### **基线方法对比**
| 方法 | 类型 | 关键技术 |
|------|------|---------|
| **OMPQ-ORM** | Mixed-precision | 基于网络正交性的敏感度 + MILP |
| **CLADO** | Mixed-precision | 成对层间交互建模 + Integer Quadratic Programming |
| **AIMET AMP** | Mixed-precision | 量化组敏感度 + Pareto Search |
| **Uniform 4-bit** | Baseline | 所有层统一使用 4-bit |

所有对比均在相同 backend（如 RTN 或 TF-Enhanced）、相同 block 划分下进行，确保公平。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 方法 | Avg Bits | PSNR (CelebA) | ΔPSNR (Corrupted) | Speed-up |
|------|----------|----------------|--------------------|----------|
| FP32 | 32.000 | 34.664 dB | — | — |
| Uniform 4-bit | 4.000 | 28.978 dB | ↓5.686 dB | — |
| OMPQ-ORM | 3.988 | 32.587 dB | ↓0.011 dB | 30× |
| CLADO | 3.988 | 34.409 dB | ↓6.585 dB | 2,570× slower |
| **Ours** | **3.987** | **34.401 dB** | **↓0.002 dB** | **28× ~ 2,570× faster** |

> 注：在 $B=4$ 下，本文方法 PSNR 仅比 CLADO 低 0.008 dB，但在 corrupted 校准下表现远优。

---

### **与基线方法的对比结果**

#### ✅ **在干净校准下的性能**
- 在 **DRUNet + RTN backend** 下：
  - 超越 OMPQ-ORM **+1.814 dB**
  - 与 CLADO 几乎持平（-0.008 dB）
- 在 **TF-Enhanced backend** 下：
  - 在 $B=5,6,7$ 下 PSNR 与 AIMET 相当甚至略高（+0.017 dB @ B=6）

#### ✅ **在校准数据损坏下的鲁棒性**
- **CLADO**：在 shared mask ($p=0.5$) 下 PSNR 从 34.409 → **27.824 dB**（↓6.585 dB）
- **本文方法**：在同一条件下仅从 34.401 → **34.429 dB**（几乎无损）
- 最大增益达 **+7.509 dB**（vs. CLADO, $B=5$, shared corruption）

#### ✅ **分配稳定性分析**
- **Spearman 相关系数**（bit-width 排序一致性）：
  - 本文方法：**0.961 ~ 0.987**
  - CLADO：**0.060 ~ 0.313**
- 表明本文方法在不同校准条件下分配策略高度一致。

#### ✅ **速度优势**
| 对比项 | 加速倍数 |
|-------|---------|
| vs. AIMET AMP | **28.4×** |
| vs. OMPQ-ORM (fixed β) | **30.6×** |
| vs. OMPQ-ORM (w/ β search) | **52.9×** |
| vs. CLADO (W3/W4/W8) | **2,570×** |

> CLADO 使用双 A100 GPU，而本文方法仅用单卡即实现如此加速。

---

### **消融实验结果**

#### 🔹 **候选 bit-width 数量的影响**
- 将本文方法限制为 {3,4,8} 后，仍优于 CLADO 在 corrupted 校准下的表现（见 Table 8）。
- 说明性能提升**并非源于更多中间 bit-width 选项**，而是算法本身的鲁棒性。

#### 🔹 **分配粒度的影响（Block vs Layer）**
- 将 36 个 block 替换为 64 个独立 layer 进行分配：
  - 在 CelebA, $B=3.5$ 下 PSNR 提升 **+0.78 dB**
  - 在 BSDS500 上也有类似增益
- 说明**更细粒度的控制有助于提升性能**，尤其在严格预算下。

#### 🔹 **迁移至 Diffusion Models**
- 应用相同的 local score + greedy rule 至 latent diffusion model：
  - **FID**: 20.853（vs. Q-Diffusion 22.626）
  - **Paired PSNR**: **34.644 dB**（vs. TFMQ-DM 24.545 dB）
- 显示该框架具有良好的跨任务泛化能力。

---

## **4. 关键结论和发现**

### **主要发现**
1. **局部扰动误差是有效的敏感度指标**：  
   无需复杂的 Hessian、成对交互或学习模型，仅利用**每层本地误差的统计特性**即可实现高性能分配。

2. **概率化分析带来天然鲁棒性**：  
   使用均值+方差构建评分函数，使方法对异常样本和噪声校准更具容忍性。

3. **贪心策略足以获得高质量配置**：  
   固定敏感度评分 + 贪心下降，避免重复评估，显著提速的同时未牺牲性能。

4. **方法具备良好可迁移性**：  
   同一套评分机制成功应用于 DRUNet 和 Latent Diffusion U-Net，表明其通用潜力。

---

### **局限性**
- **依赖 FP32 激活**：目前仅做 weight-only quantization，未考虑 activation quantization。
- **尚未部署到嵌入式平台**：缺乏硬件层面的实际延迟/功耗测量。
- **理论边界仍有改进空间**：当前的概率界较松，未来可探索 tighter concentration inequalities。

---

### **未来工作方向**
1. **扩展至 Activation Quantization**：联合优化 weight 和 activation 的 bit-width。
2. **硬件感知优化**：结合特定芯片的 mixed-precision 支持（如 Tensor Core, NPU）。
3. **在线自适应量化**：根据输入动态调整 bit-width。
4. **进一步收紧理论误差界**：提升分析对极端情况的刻画能力。
5. **应用于更大规模模型**：如 LLMs、Video Diffusion 等。

---

> 📌 **一句话总结**：  
> 本文提出一种基于**逐层局部误差归因**的混合精度 PTQ 方法，通过**概率化分析 + 可分离评分 + 贪心分配**，实现了**极快、鲁棒、无需求解器**的 bit-width 配置，在多种任务和模型上超越主流方法，尤其在面对损坏校准数据时展现出显著优势。

</details>

---

### 9. [A Systematic Investigation of Bias in Large Language Models for Advertising Relevance](https://arxiv.org/abs/2610.07544)

**Authors**: Weiwei Wang, Yinchuan Xu, Jialu Gao, Youkow Homma, Jian Jiao  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.07544v1  

#### Abstract
Large language models (LLMs) are increasingly used to judge how well an advertisement matches a query, but the fairness of these judgments has received limited attention. We conduct a systematic study of fairness in relevance judgments made by LLMs for queries and advertisements. Our counterfactual ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Systematic Investigation of Bias in Large Language Models for Advertising Relevance

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文系统地研究了**大型语言模型（LLMs）在广告相关性判断中的公平性问题**，重点关注以下三类潜在偏见：
- **Advertiser identity and popularity bias**（广告主身份与知名度偏见）
- **Language bias**（输入语言偏见）
- **Demographic bias**（人口统计学偏见，如性别、种族、年龄）

这些问题可能导致LLM对语义上等价的查询-广告对产生不一致的相关性评分，从而影响用户和广告主的公平待遇。

### 🚀 提出的新方法与创新
- **提出统一的反事实框架（Unified Counterfactual Framework）**：
  - 可用于评估多种类型的偏见（advertiser, language, demographic），适用于不同输出形式的模型（分类标签 vs. relevance probability）。
  - 支持对 **general-purpose LLMs**（如 GPT-4o）和 **fine-tuned relevance models**（如 Qwen-7B）进行公平性分析。
- **多视角公平性评估设计**：
  - 使用真实广告日志数据 + 合成控制变量查询，兼顾现实性和可控性。
  - 区分公司目标型（company-targeting）与非目标型查询，在推理阶段测试去标识化效果。
- **训练与推理双路径缓解策略评估**：
  - 推理时：masking company name
  - 训练时：rebalancing training data based on label distribution

### 🔍 相比现有方法的优势
| 方面 | 本文优势 |
|------|---------|
| **研究范围** | 首次将 advertiser bias、language bias 和 demographic bias 统一纳入广告相关性任务中系统评估 |
| **方法论严谨性** | 引入反事实对照实验，控制除目标属性外的所有变量 |
| **实用性导向** | 不仅发现问题，还评估了实际可行的 mitigation 策略（masking / rebalancing） |
| **模型多样性** | 对比通用 LLM（GPT-4o）与专用 fine-tuned 模型（Qwen-7B），更具普适参考价值 |

---

## 2. 核心实验方法和设置

### 📊 数据集使用情况

| 类型 | 描述 |
|------|------|
| **Real-world query-ad pairs** | 来自真实广告日志：<br>- 2,000 条零售产品类 query-ad 对<br>- 878 条就业类 query-ad 对<br>均排除明确提及广告主的查询 |
| **Multilingual dataset** | 英文原始 query-ad 对翻译为中文和芬兰文，共 2,000 组三语版本 |
| **Synthetic demographic queries** | 构造受保护属性变化的合成查询，覆盖：<br>- 性别（female/women/male）<br>- 种族（White/Black/African American）<br>- 年龄（young adults / middle-aged）<br>广告内容固定，仅修改 query 中的人口统计词 |

### ⚙️ 实验设置

| 设置项 | 内容 |
|-------|------|
| **被测模型** | - **GPT-4o**：作为 general-purpose LLM，输出 categorical label（good/fair/bad）<br>- **Fine-tuned Qwen-7B**：专为 relevance prediction 训练，输出 pRel ∈ [0,1] |
| **反事实构造方式** | - **Advertiser substitution**：替换广告标题、描述、URL 中的广告主名称（保留其他内容不变）<br>- **Language translation**：保持语义一致的跨语言转换<br>- **Demographic paraphrasing**：仅更改 query 中的 demographic term |
| **评估指标** | - **GPT-4o**：<br>  • Bad Decision Rate (Bad DR)<br>  • Fair Decision Rate (Fair DR)<br>  • Consistency（重复运行下标签一致性）<br><br>- **Qwen-7B**：<br>  • Mean predicted relevance probability (pRel)<br>  • AUC（用于 mitigation 实验） |

### 🆚 基线方法对比
本文未直接对比传统 ML 模型，而是以“理想情况下应无差异”作为隐含基线：
- 在反事实条件下，若模型输出发生变化，则视为存在偏见。
- 控制组包括：
  - 原始广告主 vs. 多个替代广告主
  - 英文 vs. 中文/芬兰文
  - 不同 demographic group 的 query 表达

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ Advertiser Identity and Popularity Bias

| 模型 | 观察结果 |
|------|--------|
| **GPT-4o** | - 替换为 **Retailer A**（知名大厂）后，Bad DR 最低（14.6%），显著优于其他零售商（15.1–16.6%）<br>- Job Platform A 比 Job Platform B 获得更优评价（Bad DR: 14.4% vs 15.3%） |
| **Qwen-7B** | - Retailer A 的 pRel 达到 59.6%，高于多数对手（除 Retailer E）<br>- Job Platform A 的 pRel 为 65.8%，高于 Job Platform B（63.1%） |

> ➤ 结论：**知名品牌普遍获得更有利的相关性判断**

#### ✅ Language Bias

| 模型 | 英文 | 中文 | 芬兰文 | 差异显著？ |
|------|-----|-----|-------|----------|
| **GPT-4o** | Bad DR: 8.0%<br>Consistency: 94.6% | Bad DR: 7.1%<br>Consistency: 83.7% | Bad DR: 11.2%<br>Consistency: 81.8% | 是（p < 0.001） |
| **Qwen-7B** | pRel: 66.7% | pRel: 54.4% | pRel: 71.3% | 显著差异 |

> ➤ 结论：**语言改变导致相关性评分显著波动；最一致的语言 ≠ 最有利评分的语言**

#### ✅ Demographic Bias（典型案例如下）

| 场景 | GPT-4o 输出（G:F:B） | Qwen-7B pRel |
|------|------------------------|-------------|
| Engineering + Female | 0:3:7 | 0.270 |
| Engineering + Male | 1:9:0 | 0.867 |
| Nursing + Female | 0:10:0 | 0.871 |
| Nursing + Male | 0:0:10 | 0.246 |
| Luxury housing + White people | 0:2:8 | 0.625 |
| Luxury housing + Black people | 0:10:0 | 0.538 |

> ➤ 发现：**强烈符合刻板印象**（engineering–male, nursing–female, luxury–white）

---

### 🔍 消融实验结果（Mitigation Experiments）

#### （1）推理阶段：隐藏公司名称（Company Name Masking）

| 查询类型 | 指标 | GPT-4o（原 vs. mask） | Qwen-7B（AUC） |
|--------|------|------------------------|---------------|
| **Non-company-targeting** | Accuracy / AUC | 62.3% → 64.1% | 0.814 → 0.814（无损） |
| **Company-targeting** | Accuracy / AUC | 58.8% → 59.9% | 0.804 → 0.766（下降） |

> ✅ **建议**：仅在非公司目标查询中 masking，不影响甚至提升性能；但在公司搜索场景需保留品牌信息。

#### （2）训练阶段：重平衡 Retailer A 示例

| 训练策略 | Retailer A pRel | 是否降低其相对优势？ |
|---------|------------------|---------------------|
| 原始数据 | 59.6% | — |
| 随机删除一半 Retailer A 示例 | 61.7% | ❌ 仍优于大多数 |
| 删除一半的 **good/fair 标签示例** | 59.7% | ✅ 排名下降至低于 Retailer E/F |

> ✅ **关键发现**：**调整 label 分布比单纯减少样本数更有效**，说明 bias 与训练数据中标记倾向有关。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **LLMs 在广告相关性判断中普遍存在多种偏见**：
   - 更倾向于知名品牌（advertiser popularity bias）
   - 对不同语言输入给出不一致评分（language bias）
   - 反映社会刻板印象（gender-job, race-housing 等 demographic bias）

2. **偏见存在于通用 LLM 和专用 fine-tuned 模型中**：
   - GPT-4o 和 Qwen-7B 均表现出类似趋势，表明该问题具有普遍性。

3. **有效的缓解策略依赖于上下文与机制设计**：
   - 推理时 masking 公司名仅适用于非品牌搜索场景；
   - 训练时通过 **label-aware rebalancing** 可有效削弱品牌偏好。

4. **公平性与准确性之间存在权衡**：
   - 在品牌相关查询中移除公司名会损害预测性能（AUC↓），提示不能一刀切处理。

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **数据匿名化** | 报告数值经过偏移处理，无法获取绝对值，影响复现 |
| **语言种类有限** | 仅测试中/英/芬三种语言，代表性受限 |
| **合成查询的人工性** | demographic queries 为人工构造，可能缺乏自然表达多样性 |
| **未涵盖所有 protected attributes** | 如宗教、残疾等未涉及 |

---

### 🔮 未来工作方向（Future Work）

1. **系统研究训练数据构成的影响**：
   - 探索不同 advertiser sample proportion 与 label distribution 如何影响 fairness-performance trade-off。

2. **开发针对性去偏算法**：
   - 设计适用于 ad relevance 任务的 fairness-aware fine-tuning 方法（如 adversarial debiasing, causal intervention）。

3. **扩展至更多语言与文化背景**：
   - 研究多语言 LLM 在非英语主导市场中的公平性表现。

4. **结合用户反馈构建动态评估体系**：
   - 将 human-in-the-loop feedback 引入 bias detection 与 mitigation loop。

---

> 💡 **总体启示**：  
> 本文揭示了 LLM-based relevance systems 存在隐蔽但系统的偏见风险，呼吁广告从业者在部署前进行系统性的 fairness audit，并根据 query intent 和训练数据分布设计精细化 mitigation 策略。

</details>

---

### 10. [SPIN: Shadow Predictive Indexer for Sparse Attention](https://arxiv.org/abs/2610.09025)

**Authors**: Yao Fu, Cyrus Chang, Ritchie Zhao, Bryce Long, Yueying Li, Mahdi Kamani, Samkit Jain, Rahul Raman, Tara Safavi, Shreya Gupta, Parsa Ashrafi Fashi, Minseok Lee, Julien Demouth, Bita Darvish Rouhani  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.09025v1  

#### Abstract
Indexer-based sparse attention reduces the cost of core attention by passing only a fixed, small number of important tokens to it. However, the indexer must still score the entire KV cache at every decoding step. This scoring overhead becomes a major bottleneck as the context length grows. We propos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SPIN: Shadow Predictive Indexer for Sparse Attention 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在基于 **indexer-based sparse attention**（如 DeepSeek Sparse Attention, DSA）的模型中，虽然核心 attention 只处理一个固定数量的重要 token（top-K），但 **indexer 本身仍需在每一步解码时对整个 KV Cache 进行评分**。随着上下文长度增长到数十万甚至百万 token（如 DeepSeek-V4 支持 1M 上下文），这一评分过程成为推理延迟的主要瓶颈。

传统方法无法有效缓解 indexer 的线性计算开销，尤其是在长上下文场景下。

---

### 提出了什么新方法或新思路
作者提出 **SPIN (Shadow Predictive Indexer)**，一种无需训练、轻量级的预测机制，用于减少 indexer 对全 KV cache 的重复评分。

#### 核心思想：
- 观察到 DSA indexer 的重要性得分具有强 **时间相关性（temporal correlation）**，即当前步骤的 token 重要性与**前一解码步（prev-iter）** 的得分高度相关。
- 利用这种模式，SPIN 维护每个 token 的 **指数移动平均（EMA）得分**，包括：
  - `V[i]`：绝对位置 `i` 的垂直历史得分（vertical pattern）
  - `D[q−i]`：相对偏移量的对角线得分（diagonal pattern，受 RoPE 影响）
- 在每步解码中，SPIN 先基于这些历史统计预测哪些 **KV block** 可能重要，仅将高分 block 输入 indexer，从而跳过大量低分 block 的评分。

#### 关键设计亮点：
- **块级稀疏化（block-level sparsity）**：适配 PagedAttention 架构，在 KV block 层面进行选择。
- **探索机制（exploration）**：保留一小部分 block 预算用于随机探索被跳过的 block，防止因“错过”而导致重要信息永久丢失。
- **无额外参数、无需训练**：完全基于运行时统计，不引入辅助网络或可学习模块。

---

### 相比现有方法的优势
| 方法 | 缺陷 | SPIN 如何改进 |
|------|------|---------------|
| 原始 DSA | indexer 每步扫描全部 context，成本线性增长 | 通过历史预测跳过 ~40% block，显著降低 indexer 负载 |
| IndexCache [GLM-5.2] | 跨层复用 indexer 结果（prev-layer） | SPIN 使用 prev-iter 预测更准确（见 Table 1） |
| TISA [2] | 固定轮询刷新策略，缺乏动态感知 | SPIN 动态建模 temporal pattern，按需更新 |
| HISA / Hierarchical Filtering | 多阶段过滤增加复杂性 | SPIN 简洁高效，仅依赖 EMA 和 pooling |

> ✅ **优势总结**：SPIN 是首个将 temporal prediction 显式应用于 indexer 输入稀疏化的方案，实现了**高效率 + 高保真度**的平衡。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型 | 描述 |
|--------|------|------|
| **LoNGBENCH-V2** | 长上下文理解 | 包含多种任务（摘要、问答等），最大上下文达 128K |
| **AA-LCR** | 长上下文推理 | 平均 95.5K tokens，最长 115.2K，测试复杂推理能力 |
| **MRCRv2** | 长上下文压力测试 | “8-needle” 测试，分别在 256K、512K、1M 上下文中定位隐藏信息 |
| **RULER** | 多任务长上下文评估 | 13 个任务 × 6 种长度（4K–128K），共 39K 示例 |
| **TAU2-AIRLINE** | 代理型任务（agentic） | 多轮对话环境下的工具调用与决策 |
| **SPEED-BENCH** | 推测解码性能测试 | 用于评估 SPIN 在 speculative decoding 中的表现 |

---

### 实验设置和评估指标

#### 模型
- 主要基于 **DeepSeek-V4 Flash / Pro**
- 使用其原生的 **multi-token prediction (MTP)** 进行推测解码实验

#### SPIN 参数配置
- Block size: 64 compressed indexer slots
- EMA decay: α = 0.5
- Target block sparsity: 20% ~ 50%
- Exploration ratio γ = 5%

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Task Score** | 各基准的任务特定得分（exact match, accuracy 等） |
| **Top-K Miss Rate (mean/p99)** | 被错误剔除的真实 top-K token 占比 |
| **Output Throughput (TPS)** | 输出吞吐量（tokens/sec） |
| **Median Inter-Token Latency (ITL)** | 解码过程中 token 生成的中位延迟 |
| **Realized Block Sparsity** | 实际跳过的 block 比例 |
| **Acceptance Length (speculative decoding)** | 每次验证接受的 draft token 数量 |

#### 基线方法对比
- **All-keep**: 不跳任何 block，完整 indexer 扫描（控制组）
- **Prev-layer / Prev-iter baselines**: 用于验证预测信号有效性
- **Row-0 sharing + row-0 update**: 推测解码中的简化 SPIN 版本

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | SPIN 表现 | 提升幅度 |
|------|----------|---------|
| 最大输出吞吐提升 | **+14.9%** | vs all-keep |
| 中位 ITL 下降 | **-13.2%** | vs all-keep |
| 实现块稀疏度 | 最高达 **~50%** | 实际跳过一半 KV blocks |
| 任务质量损失 | < 3% 相对下降（多数情况下） | 在 30–40% sparsity 下几乎无损 |

---

### 与基线方法的对比结果

#### ✅ 长上下文任务质量保持（Table 2 & 3）
| 方法 | LoNGBENCH-V2 (Flash) | AA-LCR (30% sparsity) | RULER (40% sparsity) |
|------|------------------------|------------------------|-----------------------|
| All-keep | 53.48% | 64.40 | 90.20 |
| SPIN(30%) | 52.09% (-2.6%) | 64.60 (+0.2) | 90.00 (-0.2) |
| SPIN(40%) | 54.27% (+1.5%) | 64.00 (-0.4) | 89.69 (-0.5) |

> 🔹 在高达 50% block sparsity 下，任务得分波动极小，表明 SPIN 对语义影响微弱。

#### ✅ 代理任务表现（TAU2-AIRLINE）
| 方法 | Score | Mean Miss |
|------|-------|-----------|
| All-keep | 72.5% | — |
| SPIN(30%) | **73.5%** | 10.9% |
| SPIN(50%) | 69.5% | 25.7% |

> 🔹 在 30% 稀疏度下反而略有提升，说明预测机制可能过滤噪声。

#### ✅ 端到端服务性能（Table 6）
| 方法 | Output TPS | TPS Gain | Median ITL | ITL Reduction |
|------|------------|----------|-------------|----------------|
| All-keep | 4720.29 | — | 17.837 ms | — |
| SPIN(40%) | 5218.65 | **+10.6%** | 16.204 ms | **-9.2%** |
| SPIN(50%) | 5421.45 | **+14.9%** | 15.484 ms | **-13.2%** |

> 🔹 性能增益随 sparsity 增加而非线性放大，说明 indexer 开销占比越高，收益越大。

#### ✅ 推测解码兼容性（Table 7）
| 方法 | Avg Acceptance Length | Draft Acceptance Rate | Realized Sparsity |
|------|------------------------|------------------------|--------------------|
| All-keep MTP | 2.4049 | 47.19% | — |
| SPIN (default) | **2.4097** | 47.28% | **39.51%** |

> 🔹 **无明显性能退化**，且实现近 40% block 稀疏，证明 SPIN 与 MTP 完美协同。

---

### 消融实验结果

#### （1）预测信号比较（Table 1）
| 方法 | Pearson Corr (DS-V4) | Top-K Recall |
|------|------------------------|--------------|
| Random | 0.000 | 0.203 |
| Prev-layer | 0.547 | 0.482 |
| Prev-iter | **0.876** | **0.751** |

> 🔹 **prev-iter 信号远优于 prev-layer**，是 SPIN 设计的基础依据。

#### （2）Pooling 方法比较（Figure 3）
| Variant | P99 Miss @ 50% sparsity | 控制精度 |
|--------|----------------------------|----------|
| A0 (slot-first) | 较高 | 间接控制 block 数 |
| A1-Max (SPIN 默认) | **最低** | 直接控制目标 sparsity |
| A2-Max | 更高 | 早期聚合损失细节 |

> 🔹 选择 **A1-Max**：先预测 slot 得分 → max-pool 成 block 得分 → 选 top-m blocks

#### （3）探索策略比较（Figure 4）
| 方法 | 是否恢复任务质量 | 实现难度 |
|------|--------------------|----------|
| No exploration | ❌ 随 sparsity 严重下降 | — |
| Random (γ=5%) | ✅ 显著恢复 | ⭐ 简单易实现 |
| Staleness-score / Block-age | ✅ 类似效果 | ❌ 需维护额外状态 |

> 🔹 最终选用 **random exploration**，兼顾性能与简洁性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **indexer 得分具有强烈的时间局部性（temporal locality）**，尤其是 prev-iter 信号比跨层信号更具预测力。
2. ✅ **轻量级 EMA + block pooling 可有效预测重要 KV blocks**，实现高达 50% 的输入稀疏而不显著损害任务质量。
3. ✅ **random exploration 能低成本地缓解 stale block 问题**，无需额外预算即可刷新潜在重要区域。
4. ✅ **SPIN 与现代 LLM serving 架构（如 vLLM + PagedAttention）无缝集成**，带来高达 **14.9% 吞吐提升** 和 **13.2% 延迟下降**。
5. ✅ **SPIN 兼容推测解码（speculative decoding）**，在 MTP 场景下不影响 acceptance 行为。

---

### 方法的局限性
1. ❗ **难以应对突发注意力转移（sudden attention shift）**：若某 block 突然变得重要但长期未被访问，其 EMA 分数可能已衰减至零，导致漏检。
2. ❗ **预测行为影响自身观测**：一旦 block 被跳过，则无法获得新分数，形成反馈闭环，可能导致误判累积。
3. ❗ **当前评估集中在 DeepSeek-V4**，尚未在其他 indexer-based 模型（如 GLM-5, LongCat-2.0, MiniMax-M3）上广泛验证泛化性。
4. ❗ **未结合 query-aware 或 cross-layer 信号**，未来可通过多源融合进一步优化预测准确性。

---

### 未来工作方向
- 🔄 **融合 current query 特征**：将当前 query 投影与历史 EMA 结合，提升对突发模式的响应能力。
- 🔀 **引入 cross-layer + temporal hybrid prediction**：结合 prev-layer 和 prev-iter 信号构建更强预测器。
- 📈 **自适应 sparsity 控制**：根据任务类型或上下文动态调整 block 预算与 exploration 强度。
- 🧪 **扩展至更多模型架构**：验证 SPIN 在 GLM、MiniMax、LongCat 等 indexer-based 模型上的通用性。
- 💡 **探索 learned lightweight predictor**：用小型 MLP 替代手工 EMA，自动捕捉复杂 temporal dynamics。

---

> ✅ **总体评价**：SPIN 是一项极具工程价值的创新，它揭示了 indexer 冗余性的本质，并以极简方式实现了高性能加速，有望成为下一代长上下文 LLM serving 的标准组件之一。

</details>

---

### 11. [BoT-GRPO: Efficient Process-Reward RL for Reasoning via Bag-of-Token Aggregation](https://arxiv.org/abs/2610.09804)

**Authors**: Yingxiang Yang, Weihang Xiao, Zhunxuan Wang, Joshua Flashner, Niresh Agarwal  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.09804v1  

#### Abstract
Reinforcement learning is now central to eliciting reasoning in large language models, while in the popular algorithm Group Relative Policy Optimization (GRPO) every token in a rollout receives the same advantage. We ask how to make process supervision efficient: accelerating convergence and improvi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BoT-GRPO: Efficient Process-Reward RL for Reasoning via Bag-of-Token Aggregation

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

- **低效的信用分配（Credit Assignment）**：在标准的 **Group Relative Policy Optimization (GRPO)** 中，整个生成序列（rollout）共享一个全局优势值（advantage），导致每个 token 都接收到相同的梯度信号，无论其对最终结果的贡献如何。这种粗粒度的更新方式导致训练收敛慢、样本效率低。
- **长度偏差（Length Bias）**：长序列在 GRPO 的组内归一化中占据更大权重，导致模型偏好生成更长的推理链，造成“过度思考”（overthinking）。
- **过程奖励（Process Reward）的成本高**：虽然细粒度的过程监督（如 step-level PRM）能提升性能，但传统方法依赖额外的 critic 网络、蒙特卡洛估计或多步 rollout，显著增加计算开销。

### **提出了什么新方法或新思路**

提出 **Bag-of-Tokens Group Relative Policy Optimization (BoT-GRPO)**，一种无需 critic 的 token-level 优势估计方法，核心思想如下：

- **Bag-of-Tokens 聚合机制**：将所有 rollout 中的所有 token 收集为一个“token 包”，并按其来源序列长度进行加权（权重为 $1/L_k$），确保每个序列对组统计量（均值、方差）的贡献相等。
- **Token-Level Advantage 计算**：基于加权后的组统计量，为每个 token 单独计算优势值，实现细粒度信用分配。
- **外部 Token-Level Reward 输入**：BoT-GRPO 不假设 reward 来源，可直接接入任何外部提供的 token-level reward（如规则、编译器诊断、LLM 判断等）。

### **相比现有方法的优势**

| 特性 | BoT-GRPO | GRPO | GSPO/DAPO/PURE |
|------|----------|------|----------------|
| 是否需要 Critic | ❌ 否 | ❌ 否 | ❌ 否 |
| 是否支持 Token-Level Reward | ✅ 是 | ❌ 否 | ❌ 否（仅 PURE 支持 step-level） |
| 是否纠正长度偏差 | ✅ 是（通过 $1/L_k$ 加权） | ❌ 否 | ❌ 否 |
| 是否可即插即用 | ✅ 是（drop-in 替换 GRPO） | — | — |
| 训练效率 | ⬆️ 显著提升 | 基线 | 中等提升 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

1. **React 前端代码生成**
   - **任务**：根据自然语言描述生成单文件 React/JSX 应用。
   - **数据来源**：模拟 WebDev Arena 构建，包含 203 个测试 prompt，涵盖仪表盘、表单、交互组件等。
   - **评估方式**：通过 Babel 编译 + Playwright 渲染为截图。

2. **AIME 数学推理**
   - **任务**：解决 AIME（1983–2024）数学竞赛题，要求多步推理。
   - **数据**：共 933 题，80%/20% 划分训练/测试集。
   - **格式**：显式标注 `(step i)` 和 `(final)` 标签。

### **实验设置和评估指标**

| 设置项 | 描述 |
|-------|------|
| **Rollout 数量** | 每 prompt 采样 8 个 rollout |
| **Batch Size** | 64 generations（8 prompts × 8 rollouts） |
| **模型规模** | 主要使用 3B–4B 模型：<br>- Qwen2.5-3B（非推理基座）<br>- SmolLM3-3B<br>- Phi-4-mini-reasoning（已调优推理） |
| **硬件** | AWS p4de（A100 80GB）或 g6e（L40S 48GB），单卡可复现 |

#### **评估指标**

- **React 任务**：
  - **Compile Success Rate**：能成功编译并渲染的比例（客观指标）。
  - **Pairwise VLM Win Rate**：使用 **Claude Opus 4.6** 作为 VLM judge，在 ABBA 协议下进行头对头比较，判断哪个渲染效果更好。

- **AIME 任务**：
  - **Pass@k**（k ∈ {1,2,4}）：在 n=8 个 rollout 中至少有一个正确的比例。

### **基线方法对比**

- **Vanilla GRPO**：标准 GRPO，仅使用序列级 reward。
- **GSPO**：序列级重要性比率与裁剪，提升稳定性。
- **DAPO**：Clip-higher 与动态采样，优化梯度。
- **PURE**：Min-form 信用分配，防止过程奖励被滥用。

> 所有方法使用相同的 reward 信号，仅 credit assignment 机制不同，确保公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **React 代码生成**

| 方法 | 达到 80% Compile Rate 所需步数 | 最终 Compile Rate | VLM Win Rate vs. Baseline |
|------|-------------------------------|-------------------|----------------------------|
| **BoT-GRPO** | **40 步** | **~94%** | **62–90%**（vs GRPO） |
| GRPO | ~75–90 步 | ~88% | 基线 |
| GSPO/DAPO | ~65–70 步 | ~88–90% | 56–78% / 58–63% |
| PURE | ~40 步（早期快），但后期下降至 **61%** | ❌ 性能退化 | 明显落后 |

- **加速比**：BoT-GRPO 达到目标质量的速度比 GRPO **快达 1.9×**。
- **视觉质量提升**：即使 compile rate 接近饱和，BoT-GRPO 在 VLM win rate 上仍持续领先，说明其能优化布局、美观性等高级属性。

#### **AIME 数学推理**

| 方法 | Pass@1 | Pass@2 | Pass@4 | 收敛速度 |
|------|--------|--------|--------|---------|
| **BoT-GRPO** | **9.6%** | **16.8%** | **20.6%** | **约一半步数** |
| GRPO | 6.6% | 8.7% | 15.1% | 较慢 |

- **绝对增益**：Pass@2 提升 **+8.1%**，表明细粒度奖励提升了推理的一致性和鲁棒性。
- **跨模型一致性**：在 Qwen2.5、SmolLM3、Phi-4 上均稳定优于 GRPO。

### **消融实验结果**

#### **(1) Reward Stack 消融（React）**

| Reward 组成 | Compile Rate | 是否存在 Reward Hacking？ |
|-------------|--------------|---------------------------|
| Compiler + ESLint（仅语法检查） | ~74% | ✅ 是（生成空页面） |
| + LLM Code Judge（语义理解） | >90% | ❌ 否 |
| + VLM Screenshot Judge（视觉判断） | 微幅提升 | 可选，但需小心引入 |

> **结论**：LLM judge 必不可少；VLM judge 可提升视觉质量，但可能因噪声导致不稳定（见 Appendix B）。

#### **(2) 局部奖励策略（AIME）**

- **Error Propagation（默认）**：一旦某步错误，后续步骤 reward 归零 → 效果最好。
- **Independent / Context-aware**：信号噪声大，学习困难 → 性能接近 baseline。

> 支持“**稳定性优于丰富性**”（stability over richness）的设计哲学。

#### **(3) 局部奖励权重 $w_s$ 与窗口大小**

- $w_s \in \{0.1, 0.5, 1.0\}$：性能稳定，验证了“全局 reward 主导”的鲁棒性。
- 窗口大小：window=3 表现最佳，过大（如 5）会稀释信号。

---

## 4. 关键结论和发现

### **主要发现**

1. **细粒度信用分配是训练效率的关键杠杆**：
   - BoT-GRPO 通过 token-level 优势估计，显著加速收敛，且不牺牲最终性能。
   - 在 React 和 AIME 两个截然不同的任务上均取得一致提升，证明其通用性。

2. **长度不变性（Length-Invariant）至关重要**：
   - $1/L_k$ 加权有效缓解了 GRPO 的长度偏差，避免模型“堆长链”。

3. **奖励信号的稳定性比丰富性更重要**：
   - 干净、有界、稳定的局部信号（如编译器报错）比复杂但嘈杂的 LLM/VLM 判断更能加速学习。
   - “**Stability over Richness**” 是实用 RLVR 系统的核心原则。

4. **BoT-GRPO 是即插即用的 GRPO 升级方案**：
   - 无需额外网络、无 unbounded cost，仅增加 $O(\sum_k L_k)$ 开销，适合资源受限场景。

### **方法的局限性**

- **依赖外部 Token-Level Reward**：BoT-GRPO 本身不生成 reward，需依赖其他系统提供（如 LLM judge、规则引擎），可能引入延迟与成本。
- **AIME 奖励依赖 LLM 判断**：当前 step-level PRM 由 Claude Sonnet 生成，存在噪声与开销。
- **未完全解耦 $1/L_k$ 与 token-level 优势的作用**：实验中二者同时启用，无法单独评估其独立贡献。
- **数据泄露风险**：AIME 数据集公开，可能存在预训练污染，影响绝对性能但不影响相对比较。

### **未来工作方向**

- **开发轻量级符号验证器（Symbolic Verifier）** 替代 LLM judge，降低延迟与成本。
- **探索更复杂的 token 聚合策略**，如基于注意力或语义分块的加权。
- **扩展到多轮对话与 Agent 任务**，结合 turn-level 与 token-level 信用分配。
- **理论分析**：从优化视角建立 BoT-GRPO 的收敛性保证，而非仅依赖经验观察。
- **更大规模模型与更长推理链的验证**：当前实验集中在 3B–4B 模型，需验证在 10B+ 模型上的表现。

--- 

> **一句话总结**：  
> **BoT-GRPO 通过长度不变的“token 包”聚合机制，实现了无需 critic 的高效 token-level 强化学习，在保持 GRPO 简洁性的同时，显著加速了推理模型的训练，并揭示了“稳定的小信号”优于“嘈杂的大信号”的实用设计原则。**

</details>

---

### 12. [RA-MoWE: Workflow-Affinity Embeddings for Query Clustering and Agentic Workflow Generation](https://arxiv.org/abs/2610.07851)

**Authors**: Qi Cheng, Shengyu Chen, Wei Cheng, Yiqun Xie, Haoyu Wang, Haifeng Chen, Xiaowei Jia  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.07851v1  

#### Abstract
Agentic workflows enable large language models (LLMs) to solve complex tasks by coordinating reasoning, tool use, and verification. However, a workflow optimized for an entire task collection can overlook differences in the reasoning strategies that individual queries need, while searching for a new...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RA-MoWE: Workflow-Affinity Embeddings for Query Clustering and Agentic Workflow Generation 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前 **Agentic workflows** 在处理复杂任务时面临两个核心矛盾：
- **全局优化 vs. 个体差异**：为整个任务集合设计统一的 workflow 可能忽略不同 query 对推理策略的差异化需求。
- **个性化生成 vs. 推理成本**：为每个 query 单独搜索最优 workflow 虽然灵活，但重复执行昂贵的搜索过程导致高推理开销。

现有方法如 ADAS、AFlow 或 FlowReasoner 要么在集合层面优化（忽视个体差异），要么对每个 query 都进行独立搜索（计算代价高）。

---

### 提出的新方法：RA-MoWE 框架

RA-MoWE（**R**eusable **A**gent **Mo**dels with **W**orkflow **E**xpertise）提出一种基于 **workflow-affinity embeddings** 的新范式，通过“测量—聚类—生成—预测”四步实现高效且个性化的 workflow 分配。

#### 核心思想与创新点：

1. ✅ **Workflow-Affinity Embeddings（工作流亲和度嵌入）**
   - 不再依赖语义相似性（semantic similarity），而是用一个 query 在一组固定参考 workflow（称为 *probes*）上的表现得分向量来表示它。
   - 向量中每一维对应一个 probe 的平均 task score，反映该 query 对应哪种 reasoning 策略有效。
   - 这种表示捕捉的是 **computational need**（计算需求）而非文本语义。

2. ✅ **Embedding-Guided Expert Generation（嵌入引导的专家生成）**
   - 将训练 queries 按其 affinity embedding 聚类（如 k-means）。
   - 每个 cluster 共享一个专门的 **expert workflow**，由 LLM optimizer 基于 cluster 的平均 embedding 初始化并迭代改进。
   - 初始 workflow 来自 cluster 中 affinity 最高的 runnable probe，后续编辑受失败样本和 probe 表现反馈指导。

3. ✅ **Embedding Encoder（嵌入编码器）**
   - 训练一个小模型（MLP）从 query 文本直接预测其 workflow-affinity embedding。
   - 部署时无需运行所有 probes，仅通过 encoder 预测 embedding 即可选择最匹配的 expert workflow，大幅降低推理延迟。

---

### 相比现有方法的优势

| 维度 | RA-MoWE | 传统方法 |
|------|--------|---------|
| **个性化程度** | 高（按计算需求分组） | 低（全局统一）或过高（每 query 搜索） |
| **推理效率** | 高（encoder 快速路由） | 低（需在线搜索或执行多个 probes） |
| **可复用性** | 强（生成 reusable expert bank） | 弱（一次性 workflow） |
| **决策依据** | 行为响应（behavioral response） | 语义或随机 |

---

## 2. 核心实验方法和设置

### 数据集
- **MixBench-H**：包含 600 个训练 + 300 个测试 queries，涵盖四个领域：
  - **AIME**（数学）
  - **GPQA**（科学问答）
  - **CodeContests**（编程）
  - **LiveCodeBench**（代码生成）
- **SWE-bench Verified**：软件工程任务，用于 repository repair 场景，采用 300/200 的 train/test split。

### 实验设置
- **Backbone Models**：GPT-4o-mini 和 Claude Haiku 4.5。
- **Reference Workflows (Probes)**：共 12 种，包括：
  - `Direct`, `CoT`, `CoT-SC` (self-consistency), `Tree-of-Thought`, `Self-Refine`, `ReAct`, `Tool Use`, `Decompose` 等。
- **Clustering**：使用 k-means 对 affinity embeddings 聚类，设 $ J=6 $ 个 clusters。
- **Expert Generation**：每个 cluster 使用 LLM optimizer 进行最多两轮 refinement，基于 validation set 上的表现提升决定是否接受修改。
- **Encoder Training**：使用 BGE-large-en-v1.5 作为 frozen text encoder，接一个 MLP 预测 affinity embedding，监督信号来自离线 probe 执行结果。

### 评估指标
- **Task Score**：答案正确率或 public-test pass fraction（MixBench-H）。
- **Harness Resolution**：SWE-bench 中修复成功的实例比例。
- **LLM Calls per Query**：推理阶段调用 LLM 的次数，衡量推理成本。
- **Utility vs. Cost Trade-off**：综合考虑性能与开销。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **Global Optimization** | AFlow, ADAS, ScoreFlow |
| **Per-Query Search** | FlowReasoner |
| **Routing Methods** | MasRouter, FrugalGPT |
| **Clustering Baselines** | Semantic BGE/Qwen3 clusters, Random clusters |
| **Oracle/Upper Bound** | Train-selected best probe, Observed oracle |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1 & Figure 3）

| 方法 | GPT-4o-mini (MixBench-H) | Haiku 4.5 (MixBench-H) | GPT-4o-mini (SWE-bench) |
|------|--------------------------|------------------------|--------------------------|
| **Direct** | 0.2872 | 0.5694 | 0.2000 |
| **Train-selected probe** | 0.3481 | 0.6894 | 0.1750 |
| **RA-MoWE (Predicted B)** | **0.2922** | **0.5214** | 0.2000 |
| **RA-MoWE (Measured A)** | 0.2742 | **0.6097** | **0.2450** |
| **FlowReasoner** | 0.3565 | 0.7465 | 0.2050 |

> 注：A 使用测试集真实 affinity；B 使用 encoder 预测 affinity。

#### 核心结论：
- 在 **SWE-bench** 上，RA-MoWE (A) 达到 **24.5% resolution rate**，显著优于 global comparator D（17.5%）和其他适配方法（如 ScoreFlow 21.0%, FrugalGPT 20.5%）。
- 在 **MixBench-H** 上，尽管 RA-MoWE(B) 未显著超越最强 baseline（FlowReasoner），但它实现了：
  - **+4.04 percentage points** 的增益（vs. probe menu selection）
  - **减少 27.7% 的 inference-time LLM calls**（从 2.924 → 2.113）

---

### 消融实验结果（Ablation Studies）

#### （1）Affinity Embedding 的有效性（3.1节）
- **Probe偏好具有可重复性**：在 GPT-4o-mini 上，基于 affinity 动态选择 probe 比固定选择高出 **+5.79%**（0.4059 vs 0.3481），说明 query 确实有稳定的 workflow 偏好。
- **Affinity-defined specialists 更优**：使用 measured affinity 分配专家，在 Haiku 上带来 **+7.17%** 提升（0.6097 vs 0.5381），证明基于行为响应的聚类是有效的。

#### （2）Encoder 预测质量的影响（3.2节）
- 当前 encoder 的 **cluster assignment 准确率较低**：
  - GPT-4o-mini：仅 **37.7%** 的预测 cluster 与真实一致
  - Haiku：仅 **12.0%** 一致（多数被分配到同一 cluster）
- 尽管如此，RA-MoWE(B) 仍能取得正向收益，表明即使 encoder 不完美，只要保留关键决策边界即可发挥作用。

#### （3）Embedding-Guided Search 的效果（3.3节，图3）
- **Guided-B**（使用 guided search 生成的 expert bank）相比：
  - **Direct**: +0.0595 提升（0.3358 vs 0.2763）
  - **Probe-menu-B**: +0.0404 提升（0.3358 vs 0.2955）
- 同时将 **serving LLM calls/query 从 2.924 降至 2.113**（↓27.7%）
- 与 single-call CoT（0.3295）相比差距不显著，说明进一步优化仍有空间。

#### （4）其他消融（Appendix C）
- **Panel Size**：将 probes 从 8 增加到 12 显著提升了 fingerprint 的多样性（distinct fingerprints 从 281 → 329），有助于更好聚类。
- **Operator Ablation**：加入 `Programmer` 操作符后，部分配置下性能提升（如 Twelve-probe B +0.0231），显示扩展操作空间的价值。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Workflow affinity 是比语义更有效的 grouping signal**  
   queries 应根据它们对不同 reasoning strategies 的响应模式分组，而不是仅仅看文本相似性。

2. ✅ **Specialists 可以带来显著性能增益**  
   在 Haiku 和 SWE-bench 上，基于 measured affinity 构建的 expert bank 明显优于全局 workflow 或简单路由策略。

3. ✅ **Embedding-guided search 能生成高效的 reusable workflows**  
   利用 cluster-level affinity 初始化 + failure-driven refinement，可在少量编辑内构建出优于 baseline 的 workflows。

4. ✅ **Encoder 可实现低成本部署**  
   虽然当前 encoder 的预测精度有限，但已足够支持有效的专家选择，避免了昂贵的 online probing。

5. ✅ **存在明显的性能-成本权衡机会**  
   RA-MoWE 在保持竞争力的同时显著降低了推理开销，适合实际部署。

---

### 局限性
1. ❌ **Encoder 预测能力较弱**  
   当前 embedding encoder 的 cluster assignment 准确率偏低，限制了 RA-MoWE(B) 的上限表现。

2. ❌ **Probe-to-executable mapping 不完全**  
   并非所有 probe（如 ReAct, Tool Use）都能映射为可执行的初始 workflow，造成信息损失。

3. ❌ **Offline 成本较高**  
   需要大量离线执行 probes 和 expert search，前期投入大。

4. ❌ **Cluster 数量和划分依赖启发式**  
   k-means 和固定 J=6 缺乏理论保证，可能不是最优分组方式。

5. ❌ **Evaluation 存在泄露风险**  
   搜索过程中使用的 probe 结果也用于构造 embedding，可能存在过拟合或信息泄露。

---

### 未来工作方向
1. 🔮 **改进 embedding encoder**  
   设计更强的 text-to-affinity 模型，例如引入 contrastive learning 或利用 LLM 自身进行 zero-shot affinity 预测。

2. 🔮 **端到端联合训练**  
   联合优化 encoder、clustering 和 expert generation，形成闭环学习系统。

3. 🔮 **动态 probe selection**  
   不固定 probe 集合，而是根据任务分布动态选择最具区分性的 workflows 作为 probes。

4. 🔮 **更丰富的 operation space**  
   支持更复杂的工具调用、multi-agent collaboration 等高级功能，提升 generated expert 的表达能力。

5. 🔮 **轻量化与压缩**  
   探索如何压缩 expert bank 或共享子模块，降低存储和部署成本。

6. 🔮 **跨任务迁移**  
   研究在一个 domain 上学习的 affinity 表示能否迁移到其他 domains，提升泛化能力。

--- 

> **总结一句话**：RA-MoWE 提出了一种“先测量行为响应、再聚类生成专家、最后用 encoder 快速路由”的新范式，在个性化与效率之间取得了良好平衡，为构建可复用、可扩展的 agentic systems 提供了新思路。

</details>

---

### 13. [Confidence Reasoning Graphs: Structured Confidence Estimation for LLM Agents](https://arxiv.org/abs/2610.07948)

**Authors**: Brendan King, Farima Fatahi Bayat, Jean-Flavien Bussotti, Pouya Pezeshkpour, Estevam Hruschka  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.07948v1  

#### Abstract
When using an LLM agent in a consequential domain, making an informed decision about whether to trust its output or intervene requires calibrated confidence in the agent's success. Confidence estimation for agents is difficult because evidence about success is distributed across heterogeneous, inter...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Confidence Reasoning Graphs: Structured Confidence Estimation for LLM Agents

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在**LLM Agent**（大语言模型智能体）的实际应用中，尤其是在高风险领域（如软件工程、企业运维），用户需要判断是否信任Agent的输出。然而，传统的置信度估计方法面临以下挑战：

- **证据分散**：Agent的成功依赖于多步异构轨迹（LLM生成、工具调用、环境反馈等），错误可能出现在任何环节。
- **黑盒模型限制**：前沿LLM通常不开放内部状态（token概率、推理过程），限制了“白盒”置信度估计。
- **成本高昂**：多次采样（sampling-based）或训练专用估计器成本高且难以适应快速迭代的Agent系统。

### 提出的新方法：Confidence Reasoning Graphs (CRGs)
CRG是一种**无需训练、仅需单次轨迹**的置信度估计框架，其核心思想是将“任务成功”这一整体主张分解为可验证的子主张，并通过图结构进行证据支撑和置信聚合。

#### 四步流程：
1. **Claim Decomposition**：递归地将根主张（如“任务已完成”）分解为必要且充分的子主张（如“Bug已修复”、“无引入新缺陷”），并进行具体化（Particularization）以适配当前任务。
2. **Evidence Gathering**：从轨迹中提取证据项（如“第14步运行函数返回预期输出”），并标注其对每个叶节点主张的关系：`Supports`、`Undermines` 或 `Unverified`。
3. **Leaf-Claim Confidence Estimation**：使用一个独立的LLM（estimator LLM）对每个叶节点主张，在给定相关证据和精简轨迹的前提下，评估其成立的概率。
4. **Aggregate Confidence to Root**：假设各子主张在给定轨迹下条件独立，将所有叶节点的置信度相乘，得到最终的根节点置信度：
   $$
   \text{Fo}(T) = \prod_{G \in L} c_G
   $$

### 相比现有方法的优势
- ✅ **无需特权访问**：仅需轨迹文本，适用于黑盒LLM。
- ✅ **低成本**：仅需一次Agent执行和少量LLM调用。
- ✅ **无需训练数据**：完全在推理时（inference-time）完成，适应性强。
- ✅ **结构化与可解释性**：提供完整的推理图，支持人工审计和干预。
- ✅ **性能优越**：在多个基准上显著优于现有基线。

---

## 2. 核心实验方法和设置

### 数据集
在三个具有挑战性的Agent基准上进行评估：
- **SWE-Bench Verified**：真实仓库的软件工程任务（修复GitHub issue）。
- **EnterpriseOps-Gym**：企业级工具密集型工作流（日历、邮件、HR等）。
- **SkillsBench**：长视野技能组合任务。

### Agent 设置
- 使用 **OpenHands** 框架。
- 三种主流Agent LLM：`GPT-5.5`, `Gemini-3.5 Flash`, `MiniMax-M3`。
- 每个问题生成一条轨迹，共收集约2000条轨迹。

### 评估指标
| 指标 | 说明 |
|------|------|
| **Adaptive ECE** | 自适应期望校准误差，衡量预测置信度与实际准确率的一致性，越低越好。 |
| **Brier Score** | 平方概率误差，衡量预测质量，越低越好。 |
| **AUROC** | 接收者操作特征曲线下面积，衡量区分成功与失败轨迹的能力，越高越好。 |
| **Behavioral Alignment Score (BAS)** | 行为对齐分数，基于决策效用，考虑不同风险容忍度下的接受/放弃策略，越高越好。 |

### 基线方法对比
| 基线 | 类型 | 说明 |
|------|------|------|
| **Basic Verbalizer** | 黑盒 | 直接询问LLM：“你有多大信心认为Agent成功了？” |
| **Reason-as-Graph** | 黑盒 | 在提示词中引导LLM进行类似CRG的结构化推理，但不显式构建图。 |
| **Verbal Consistency** | 黑盒 | 采样10次“成功/失败”的判断，将“成功”比例作为置信度。 |
| **Surrogate LNSP** | 白盒 | 使用代理LLM计算Agent最后动作的长度归一化序列概率（Length-Normalized Sequence Probability）。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）
在所有三个基准上，**CRG (Qwen-3.8 27B)** 均取得最佳综合表现：

| 方法 | SWE-Bench (ECE) | EnterpriseOps (ECE) | SkillsBench (ECE) | 综合BAS |
|------|------------------|--------------------|-------------------|--------|
| **CRG (ours)** | **0.09** | **0.13** | **0.11** | **最高** |
| Basic Verbalizer | 0.18–0.19 | 0.49–0.57 | 0.41–0.44 | 远低于CRG |
| Surrogate LNSP | 0.10 | 0.22 | 0.16 | 低于CRG |

- **CRG实现了最低的ECE和Brier Score**，表明其置信度最校准。
- **CRG在BAS上远超其他方法**，尤其在高风险场景下避免了因过度自信导致的巨大损失（如Reason-as-Graph在EnterpriseOps-Gym上BAS低至-3.6）。
- **Surrogate LNSP虽然ECE较低，但AUROC接近随机水平**（如SWE-Bench上为0.47），说明其缺乏区分能力，仅为“伪校准”。

### 与基线方法的对比结果
- **相比黑盒基线**：CRG在**校准性**（ECE/Brier）和**决策效用**（BAS）上全面超越，且消除了过度自信问题。
- **相比白盒基线**：CRG不仅更校准，而且具备更强的**区分能力**（AUROC更高），同时保持了黑盒部署的灵活性。

### 消融实验结果（Ablation Study, Table 4）
| 方法 | ECE | Brier | BAS |
|------|-----|-------|-----|
| Reason-as-Graph (无图) | 0.38 | 0.37 | -0.23 |
| Reason-with-Graph (有图但不传播) | 0.35 | 0.36 | -0.16 |
| **CRG (图+传播)** | **0.10** | **0.23** | **0.17** |

- **关键发现**：性能提升主要来自于**叶节点的局部置信估计**和**自底向上的置信聚合**，而非仅仅是结构化推理本身。
- 图的构建有助于分解，但真正的增益在于将复杂判断拆解为可独立评估的原子主张。

---

## 4. 关键结论和发现

### 主要发现
1. **结构化分解优于整体判断**：将“任务成功”分解为证据支撑的子主张，能显著提升置信度的校准性和可靠性。
2. **校准性 ≠ 区分能力**：一个方法可能看起来很校准（如Surrogate LNSP），但若无法区分成功与失败，则对决策无用。**BAS** 是更全面的评估指标。
3. **CRG是实用且高效的方案**：
   - 成本极低（每条轨迹仅$0.04–$0.07）。
   - 位于**成本-校准帕累托前沿**（Pareto frontier），在有限预算下提供最优校准。
   - 支持跨Agent框架迁移（在Codex和Claude Code上同样有效）。
4. **可解释性是核心优势**：CRG生成的图允许用户追溯低置信度的原因（如哪个子主张未被验证），从而支持人工干预。

### 方法的局限性
- **依赖LLM的质量**：构造图和评估置信的LLM可能出现分解不完整、误解证据等问题。
- **条件独立性假设**：乘积聚合假设子主张条件独立，现实中可能存在依赖，导致低估联合概率。
- **构造图可能不完美**：实证分析（Appendix A.11）显示，LLM生成的图在“必要性”和“具体化”方面仍有改进空间。

### 未来工作方向
- 探索更复杂的**依赖感知聚合规则**（如贝叶斯网络）。
- 将CRG用于**主动学习**，指导Agent在关键步骤进行自我验证。
- 扩展到**多Agent协作**场景中的置信度估计。
- 结合**形式化验证**技术，增强对关键子主张的评估。

---

> **总结**：CRG提出了一种新颖、实用且可解释的LLM Agent置信度估计框架，通过结构化推理图实现了高质量的校准和决策支持，为安全可靠的Agent部署提供了重要工具。

</details>

---

### 14. [zkLLMPoT: Efficient Zero Knowledge Proof of Training for Large Language Models](https://arxiv.org/abs/2610.08258)

**Authors**: Junkai Liang, Zhanpeng Guo, Pengfei Wu, Qingni Shen, Jiaheng Zhang, Zhonghai Wu, Haiyang Xue, Shengfang Zhai  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.08258v1  

#### Abstract
Auditing the claimed outcomes of large language model (LLM) training is challenging when model weights and training data are private, while cryptographically proving the full training process is prohibitively expensive at Transformer scale. We present zkLLMPoT, a zero-knowledge framework that certif...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：zkLLMPoT: Efficient Zero Knowledge Proof of Training for Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）训练过程中的**可信审计难题**。当模型权重和训练数据因隐私或商业机密被隐藏时，第三方无法验证模型是否真正达到了其声称的训练成果（如领域适应、指令微调、安全对齐等）。传统方法要么成本过高，要么仅适用于小规模模型。

现有方案存在三大缺陷：
- **Per-query inference ZKPs**（如 zkLLM, zkGPT）：只证明单次推理的完整性，不提供针对训练目标的整体检查点级审计。
- **Full training trajectory proofs**（如 Kaizen, DPproof）：需编码完整的反向传播和优化器更新，计算开销随训练步数线性增长，在 LLM 规模下不可行。
- **Replication-based auditing**：要求第三方重新执行训练，暴露私有数据和算力资源。

### 提出的新方法与核心思想
提出 **zkLLMPoT** —— 一种基于零知识证明的高效训练成果认证框架，其核心创新在于：

#### ✅ 创新点 1：Outcome-Attestation Formulation（结果可证范式）
- 不再试图证明整个训练轨迹（即每一步梯度下降），而是聚焦于**已提交的检查点**（committed checkpoint）在审计方选定数据上的表现。
- 将训练成果转化为一个前向计算目标函数值 $ v = \mathcal{L}(f_{\mathcal{A},\theta}(S)) $ 的声明，并通过 zk-SNARKs 加以证明。
- **优势**：证明成本与训练迭代次数无关，仅依赖一次前向传播。

#### ✅ 创新点 2：Commit-Then-Challenge Protocol（先承诺后挑战协议）
- 流程顺序为：
  1. 训练者编译模型架构并提交模型权重 $[\![\theta]\!]$；
  2. 审计者随后选择挑战序列 $S$ 和目标函数 $\mathcal{L}$；
  3. 训练者基于已承诺的 $\theta$ 在 $S$ 上进行前向计算并生成 ZK 证明。
- **防止过拟合攻击**：由于 $S$ 是在 $\theta$ 固定之后才公开的，训练者无法针对性地调整模型以“作弊”。

#### ✅ 创新点 3：Operator-Level Circuit Optimization
- 构建支持 Transformer 中关键操作符（如 Attention, FFN, LayerNorm/RMSNorm, Softmax）的紧凑 ZK 电路。
- 使用 **sumcheck arguments** 处理线性运算（矩阵乘法）；
- 使用 **lookup arguments**（logup-style）处理非线性激活函数（SwiGLU, ReLU, Softmax）；
- 支持多种终端目标函数接口（cross-entropy loss, fairness gap, membership inference score, safety score），无需重构电路。

#### ✅ 相比现有方法的优势
| 方面 | zkLLMPoT | 现有方法（Kaizen / zkPoT） |
|------|----------|-----------------------------|
| 可扩展性 | 支持 13B 参数模型 | 最大仅支持 ~10M 参数 |
| 成本增长模式 | 与训练步数无关（仅一次前向） | 随训练步数线性增长 |
| 验证粒度 | Checkpoint-level 审计 | Per-step 或全轨迹证明 |
| 实用性 | 可用于真实 LLM 发布场景 | 仅限演示级模型 |

---

## 2. 核心实验方法和设置

### 使用的模型家族与配置
实验覆盖四个主流 LLM 家族，共 **14 种配置**，参数范围从 **0.125B 到 13B**：
- **OPT** (Zhang et al., 2022)
- **Llama** (Touvron et al., 2023)
- **Qwen2.5** (Bai et al., 2023)
- **DeepSeek-Coder** (DeepSeek-AI, 2024)

> 注：未使用真实训练数据集进行训练，而是采用**合成 fixed-point 张量**模拟各模型维度下的推理行为。

### 实验设置
- **序列长度**：$T = 512$
- **批大小**：1
- **量化方式**：signed fixed-point quantization（公共 scale）
- **硬件环境**：
  - GPU：NVIDIA A100-PCIE (40GB)
  - CPU：Intel Xeon (80 threads), 629GB RAM
  - OS：Ubuntu 22.04, CUDA 12.4
- **验证器运行位置**：CPU（mcl 库），不使用 GPU

### 评估指标
| 指标 | 描述 |
|------|------|
| **Proving Time** | 生成 ZK 证明所需时间（秒） |
| **Verification Time** | 审计者验证证明的时间（秒） |
| **Proof Size** | 传输的证明对象大小（MiB） |
| **Commitment Time** | 权重绑定时间（一次性成本） |

### 基线方法对比
| 基线系统 | 模型 | 特点 |
|--------|------|------|
| **zkPoT** (Garg et al., 2023) | Logistic Regression (1K params) | 唯一开源实现，用于横向比较 |
| **Kaizen** (Abbaszadeh et al., 2024) | VGG-11 (~10M params) | 每训练步约 15 分钟证明时间 |
| **ZKAudit** (Waiwitlikhit et al., 2024) | MobileNet v2 | per-step 成本较高 |
| **Confidential-DPproof** (Shamsabadi et al., 2024) | Logistic Regression | 报告耗时达 100 小时，交互式无输出证明对象 |

> 所有基线数据来自原论文报告值，部分为估算。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1 & Figure 3）

#### 🔹 在 1.xB 模型上的综合性能（T=512）
| 模型 | 参数量 | Commit (s) | Prove (s) | Verify (s) | Proof Size |
|------|--------|------------|-----------|------------|------------|
| OPT-1.3B | 1.3B | 175 | 58.8 | 0.428 | 17.0 MiB |
| Llama-1.1B | 1.1B | 138 | 53.2 | 0.423 | 17.3 MiB |
| Qwen2.5-1.5B | 1.5B | 206 | 42.4 | 0.416 | 19.9 MiB |
| DeepSeek-Coder-1.3B | 1.3B | 171 | 41.0 | 0.406 | 16.9 MiB |

> ✅ **平均证明时间：41–59 秒**  
> ✅ **验证时间：< 0.5 秒**  
> ✅ **证明体积：~17–20 MiB**

#### 🔹 扩展到更大模型（最高 13B）
- **证明时间随参数增长缓慢**：
  - 参数增加 **104×**（0.125B → 13B）
  - 证明时间仅增加 **8.3×**（15.7s → 130.6s）
- **验证时间始终保持在半秒以内**
- **权重承诺时间呈近似线性增长**：约 **124–142 秒 / 十亿参数**

#### 🔹 运算符级分解（Figure 2）
在 1.xB 模型上各组件耗时分析（T=512）：
- **Softmax**：占总时间 **56–64%**，是最大瓶颈
- **Attention Score Matmul**：4.6–10.3s（取决于头数）
- **Normalization (RMSNorm/LayerNorm)**：2.6–3.1s
- **Elementwise Activation (SwiGLU/ReLU)**：3.0–5.0s
- **线性层（Projection, FFN, Output Head）**：合计仅 4.1–6.0s，效率高

> 表明当前设计中 **Softmax 是主要优化方向**。

### 与基线方法的对比结果
| 维度 | zkLLMPoT | Kaizen (VGG-11) |
|------|----------|----------------|
| 单次证明时间 | ~50s | ~882s / step |
| 支持最大参数 | **13B** | ~10M |
| 是否依赖训练步数 | ❌ 否 | ✅ 是（每步都要证明） |
| 是否产生可转移证明 | ✅ 是（~17MiB） | ✅ 是（~1.6MB） |
| 实际可用性 | ✅ 可用于发布级 LLM | ❌ 仅适合小型网络 |

> zkLLMPoT 在 **更大模型上花费更少时间**，且成本不随训练步数增长。

### 消融实验与泛化能力（Section 4.6）
- 支持多种目标函数，复用同一电路：
  - Next-token NLL（默认）
  - Cross-Entropy on labeled sets
  - Fairness Gap: $v = \mathcal{L}(\theta; S_0) - \mathcal{L}(\theta; S_1)$
  - Membership Inference：单条挑战序列的 NLL
  - Safety Score：拒绝/毒性样本上的 CE 或 logits 得分
- 输出头计算仅占总证明时间的 **0.1%–1.5%**，说明终端目标灵活且低成本。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **zkLLMPoT 实现了首个面向大规模 LLM 的实用化零知识训练成果认证框架**，可在合理时间内完成 13B 模型的证明。
2. ✅ 通过“先承诺后挑战”机制有效防御了训练者对审计数据的过拟合攻击。
3. ✅ 证明成本与训练步数解耦，使其适用于长周期预训练或微调任务。
4. ✅ 支持多样化审计目标（capability, fairness, safety, memorization），具备良好通用性。
5. ✅ 验证速度快（<0.5s）、证明体积小（~17MiB），便于下游用户独立验证。

### 方法的局限性
1. **Softmax 是性能瓶颈**：目前占证明时间一半以上，亟需更高效的非线性处理方案。
2. **仍局限于 Decoder-only 架构**：暂未支持 encoder-decoder 或 MoE 类模型。
3. **假设模型已完成训练**：不涉及训练过程中的动态监控。
4. **依赖可信设置？** 虽然未明确提及，但基于 zk-SNARKs 的系统通常需要可信初始化（trusted setup），可能影响去中心化部署。

### 未来工作方向（作者提出）
1. **更快的操作符设计与融合策略**：进一步降低 Softmax 等重型算子开销，拓展至更长序列和更大模型。
2. **更灵活的用户自定义审计目标**：允许用户定义新的损失函数或评估逻辑，同时保持电路兼容性。
3. **多参与方协议扩展**：
   - 支持多个训练者联合持有模型；
   - 多个审计者共同制定挑战集；
   - 下游多方验证而无需重复支付验证成本。

---

> 📌 **总体评价**：zkLLMPoT 是迈向 **可信赖、隐私保护的大模型审计基础设施** 的重要一步。它将原本不可行的 LLM 训练认证问题转化为一个高效的前向计算证明问题，为未来构建透明、可控的 AI 生态提供了关键技术路径。

</details>

---

### 15. [MoF: Preference-Aware Mixture Modeling for Black-Box LLM Personalization](https://arxiv.org/abs/2610.08330)

**Authors**: Hun Park  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.08330v1  

#### Abstract
Proprietary Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet aligning their outputs with diverse user preferences remains challenging. Existing personalization approaches for black-box LLMs often rely on user-specific scoring heads, causing the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MoF: Preference-Aware Mixture Modeling for Black-Box LLM Personalization 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的黑盒 LLM 个性化方法（如 HYDRA）通常依赖**用户特定的评分头**（user-specific scoring heads），导致以下问题：
- **参数规模线性增长**：个性化参数数量随用户数线性增加，难以扩展到大规模用户群体。
- **无法泛化到未见用户**：对于训练时未见过的用户，需要额外的适配（fitting）阶段来更新参数。

### 提出了什么新方法或新思路
本文提出 **Mixture-of-Facets (MoF)**，一种可扩展的黑盒 LLM 个性化框架，其核心思想是：
- **将用户偏好建模为共享潜在偏好“面”（facets）的组合**，而非为每个用户分配独立参数。
- 引入 **history-conditioned routing** 机制，根据用户历史动态选择并加权多个共享的 **facet heads** 来实现个性化。
- 使用 **Sparse Autoencoder (SAE)** 对用户历史进行稀疏编码，使路由器更关注偏好相关信号。
- 提出 **Cluster-Based Sampling (CBS)** 策略，在训练中从不同聚类的历史项采样，促进 facet 头的专业化。

### 相比现有方法的优势
| 维度 | MoF | HYDRA / 其他方法 |
|------|-----|------------------|
| **可扩展性** | ✅ 参数量恒定 `O(1)`，不随用户数增长 | ❌ 用户特定头导致参数线性增长 `O(N)` |
| **泛化能力** | ✅ 可直接用于未见用户，无需额外适配 | ❌ 需要对新用户进行 fit 阶段 |
| **参数效率** | ✅ 固定数量的共享 facet heads | ❌ 每个用户独占参数 |
| **性能** | ✅ 在多数任务上优于基线 | ⚠️ 性能稳定但非所有指标均最优 |

---

## 2. 核心实验方法和设置

### 使用的数据集
基于 **LaMP benchmark** (Salemi et al., 2024)，选取四个任务：
- **分类任务**：
  - **LaMP-2**: Movie Tagging Classification
  - **LaMP-3**: Product Rating Classification
- **生成任务**：
  - **LaMP-4**: News Headline Generation
  - **LaMP-5**: Scholarly Title Generation

> 注：LaMP-1、6、7 因输入结构不兼容或不可用被排除。

### 实验设置和评估指标
| 项目 | 设置 |
|------|------|
| **训练/测试划分** | 100 用户训练，50 用户测试，结果平均于 3 次随机抽样 |
| **基础模型** | `bge-base-en-v1.5` (110M) 用于 reranker & adapter |
| **黑盒 LLM** | `gpt-3.5-turbo-1106` 生成候选响应 |
| **facet 数量** | `F=7` |
| **路由策略** | top-3 路由 |
| **优化器** | AdamW, lr=5e-5, batch_size=64, weight_decay=0.01 |
| **训练轮数** | 2 epochs |

### 评估指标
| 任务 | 指标 |
|------|------|
| LaMP-2 | Accuracy (↑), F1 (↑) |
| LaMP-3 | MAE (↓), RMSE (↓) |
| LaMP-4 / LaMP-5 | ROUGE-1 (↑), ROUGE-L (↑), BLEU (↑) |

### 基线方法对比
- **Zero-Shot**: 无个性化提示
- **ICL**: 上下文中插入 k 个随机历史项
- **RAG**: BM25 检索 + 提示增强
- **PAG**: LLM 生成用户摘要 + RAG
- **CFRAG**: 协同过滤 + 历史检索
- **HYDRA**: 当前最优 reranker & adapter 框架（含 user-specific heads）
  - HYDRA-Reranker / HYDRA-Adapter

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）
| 方法 | Avg. Rank ↓ |
|------|-----------|
| **MoF (ours)** | **1.4** |
| HYDRA | 3.5 |
| CFRAG | 7.4 |
| RAG(k=4) | 6.9 |
| PAG(k=1) | 8.0 |

MoF 在 **所有任务上平均排名最高**，显著优于现有方法。

#### 详细性能对比（部分）
| 方法 | LaMP-2 Acc | LaMP-3 MAE | LaMP-4 R-L | LaMP-5 BLEU |
|------|------------|------------|------------|-------------|
| HYDRA | 0.593 | 0.393 | 0.163 | 6.652 |
| **MoF** | **0.627** (+3.4%) | **0.300** (-23.7%) | **0.176** (+8.0%) | **6.967** (+4.7%) |

> MoF 在准确率、误差、生成质量等多方面全面超越 HYDRA。

### 与基线方法的对比结果
- MoF 在 **所有任务上平均性能最强**，且无需用户特定参数。
- 提示工程类方法（ICL, RAG, PAG）表现不稳定，有时甚至不如 Zero-Shot。
- reranker & adapter 类方法（HYDRA, MoF）更鲁棒，能有效对齐用户偏好。

### 消融实验结果（Table 6）
| 方法变体 | LaMP-2 Acc | LaMP-3 MAE | 说明 |
|----------|------------|------------|------|
| MoF (完整) | 0.627 | 0.300 | — |
| w/o CBS | 0.587 | 0.380 | **性能下降最大**，表明 CBS 至关重要 |
| w/o Router | 0.600 | 0.360 | 动态组合 facet 的优势明显 |
| w/o SAE | 0.607 | 0.333 | 稀疏表示有助于聚焦偏好信号 |
| w/o Multi-Facet | 0.607 | 0.353 | 多 facet 组合优于单 head |

> 所有组件均有贡献，其中 **CBS 和 Router 最关键**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **MoF 实现了高性能与高可扩展性的统一**：
   - 通过共享 facet + history-conditioned routing，避免了用户特定参数。
   - 参数复杂度为 `O(1)`，而 HYDRA 为 `O(N)`。
2. ✅ **强泛化能力**：
   - 在 **1,000 未见用户** 上测试，MoF 仍显著优于 RAG 和 Zero-Shot（Table 3）。
   - 无需 fit 阶段即可部署，适合大规模应用。
3. ✅ **CBS 有效提升多样性**：
   - 可视化显示 MoF 能从多个聚类中采样历史项，而 HYDRA 集中于局部（Figure 5）。
   - CBS 显著提升性能，尤其在 MoF 架构下效果更佳（Table 4）。
4. ✅ **路由权重反映真实偏好结构**：
   - 聚类用户路由模式后，发现不同组对应不同叙事倾向（如冲突、奇幻、情感驱动），表明路由具有语义意义（Table 7, Figure 10）。

### 方法的局限性
1. **并非所有指标均最优**：
   - 例如在 LaMP-4 的 BLEU 指标上未超过 HYDRA。
2. **训练用户数有限**：
   - 当前实验仅在 100 用户上训练，未验证在更大、更多样化群体中的训练行为。
3. **固定 facet 容量**：
   - facet 数量固定，可能限制对极端多样化偏好的建模能力。
4. **未覆盖长文本个性化**：
   - 未在 LongLaMP 或与 FERMI 等 prompt 优化方法比较。

### 未来工作方向
- 探索 **自适应 facet 分配** 或 **动态容量调整** 机制。
- 将 MoF 扩展至 **长文本个性化**（Long-form personalization）场景。
- 与 **prompt 优化类方法**（如 FERMI）进行对比研究。
- 结合 **隐私保护技术**（如联邦学习）以应对敏感历史数据问题。

--- 

> **总结**：MoF 提出了一种新颖的、基于共享潜在“面”的混合建模框架，成功实现了**高效、可扩展、无需用户特定参数的黑盒 LLM 个性化**，在性能和泛化能力上均优于现有方法，为大规模个性化 AI 系统提供了可行路径。

</details>

---

### 16. [Parallel Predictive World Models for Accurate and Efficient Long-Horizon Planning](https://arxiv.org/abs/2610.08627)

**Authors**: Wanjin Feng, Baobin Zhang, Ao Yu, Shibo Feng, Xi Wang, Xingyu Gao  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.08627v1  

#### Abstract
Long-horizon world-model planning typically relies on autoregressive rollouts, where predicted states are repeatedly fed back into the model. This preserves temporal structure but creates a horizon-length sequential path and exposes later predictions to recursive decoded-state feedback. We introduce...

---

### 17. [Multi-Objective Aligned Small Language Model Framework for SUD Patient Dialogue Generation](https://arxiv.org/abs/2610.09209)

**Authors**: Thushara Manjari Naduvilakandy, Hyeju Jang, Mohammad Al Hasan  
**Category**: cs.CL  
**Published**: 2026-10-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.09209v1  

#### Abstract
Substance Use Disorder (SUD) counseling requires patient responses that reflect underlying cognitive states such as beliefs, coping strategies, and readiness for change. Although large language models (LLMs) can generate fluent text, they often fail to produce cognitively coherent and clinically rea...

---

### 18. [ResidualQuant: KV Cache Quantization for Looped Transformers with 2-Bit Residuals](https://arxiv.org/abs/2610.10381)

**Authors**: Heejun Kim, Junyoung Lee, SangLyul Cho, Dongsu Han, Insu Han, Sehoon Kim  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.10381v1  

#### Abstract
Looped Transformers improve parameter efficiency by repeatedly applying shared Transformer blocks over multiple recurrent loops, increasing computational depth without increasing the parameter count. However, KV cache memory still scales with the number of loops, becoming a key memory bottleneck tha...

---

### 19. [Offline AI Modules: Voice-First Offline Architecture, Hardware Reference Stack, Quantization and Benchmarking](https://arxiv.org/abs/2610.07026)

**Authors**: Sunday Afariogun, Odunolaoluwa Jenrola, Zeinab Nezami  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.07026v1  

#### Abstract
The Offline AI Modules workstream enables practical, low-power, and community-accessible deployment of voice-first AI systems that operate fully offline. Designed for African language communities where speech is the dominant mode of interaction and internet connectivity is unreliable or absent, the ...

---

### 20. [Navigating Route Latent Space for Synthesizable Molecular Design](https://arxiv.org/abs/2610.07560)

**Authors**: Tao Li, Tuan Vinh, Monika Raj, Yuan Fang, Zhichun Guo, Carl Yang  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.07560v1  

#### Abstract
Goal-directed molecular design has advanced rapidly, yet a substantial proportion of designed molecules remain difficult to synthesize in practice, limiting their real-world utility. Prior synthesizability-aware methods either project generated molecules back to synthesizable analogs that deviate fr...

---

### 21. [Agentic Semantic Sensing for Resource-Adaptive AI-RAN](https://arxiv.org/abs/2610.07829)

**Authors**: Zhongqin Wang, Xiaoqi Zhang, Nan Yang, Kai Wu, J. Andrew Zhang, Y. Jay Guo  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.07829v1  

#### Abstract
Semantic sensing (SemS) acquires task-relevant information rather than reconstructing complete physical information. Existing SemS formulations typically operate open loop: sensing configurations and observation schedules are fixed before inference and cannot respond to evolving task-level evidence....

---

### 22. [SquidAgent: Parallelize Wisely, Coordinate Efficiently](https://arxiv.org/abs/2610.08647)

**Authors**: Yexiong Lin, Shanshan Ye, Yu Yao, Zhen Fang, Bo Han, Tongliang Liu  
**Category**: cs.AI  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.08647v1  

#### Abstract
LLM-based agents solve complex multi-step tasks, but sequential execution incurs substantial latency. In principle, parallelizing work across multiple agents should yield near-linear speedups. Yet existing parallel multi-agent systems often run slower than a single-agent baseline. We attribute this ...

---

### 23. [SpikingVLA: Asynchronous Spiking Vision-Language-Action Models](https://arxiv.org/abs/2610.09710)

**Authors**: Jingya Wang, Dehao Zhang, Shuai Wang, Malu Zhang, Yang Yang, Haizhou Li  
**Category**: cs.CL  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.09710v1  

#### Abstract
ANN-to-SNN conversion offers a practical route toward energy-efficient spiking Vision-Language-Action (VLA) models by bypassing the substantial cost of training large-scale SNNs from scratch. However, existing methods often require many timesteps to maintain competitive performance, resulting in sub...

---

### 24. [Beyond Outcome Rewards: Constructing and Assigning Retrieval Credit for Search Agents](https://arxiv.org/abs/2610.10179)

**Authors**: Wenyu Huang, Xinyu Hou, Pavlos Vougiouklis, Ruofei Lai, Jeff Z. Pan  
**Category**: cs.CL  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.10179v1  

#### Abstract
Search agents enable Large Language Models (LLMs) to iteratively retrieve and use information for complex multi-hop questions. Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising approach for post-training such agents, but its reliance on sparse, outcome-based supervision can ma...

---

### 25. [KVFetch: Temporal Prefetching for the Missing Half of KV Cache Compression](https://arxiv.org/abs/2610.08811)

**Authors**: Linfeng Dong  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.08811v1  

#### Abstract
As context windows scale to tens or hundreds of thousands of tokens, KV cache compression has become essential for efficient LLM inference. Existing methods fall into three families: score-based eviction, summary compensation, and offload-and-recall. Yet all three decide what to keep or recall by co...

---

### 26. [Task-Oriented Key-Layer KV Communication for Efficient Latent Multi-Agent Collaboration](https://arxiv.org/abs/2610.08820)

**Authors**: Dongsen Zhang, Peipei Li, Zekun Li, Wenjun Xu  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.08820v1  

#### Abstract
Large language model-based multi-agent systems improve complex problem solving through collaboration, while latent communication directly transmits model internal states to avoid the high inference costs of natural language. However, existing KV-based latent communication methods prioritize sender-s...

---

### 27. [The Dichotomy Between Pattern Recognition and Step-by-Step Reasoning](https://arxiv.org/abs/2610.09186)

**Authors**: Amrut Nadgir, Pratik Chaudhari, Vijay Balasubramanian  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.09186v1  

#### Abstract
We argue that pattern recognition and step-by-step reasoning are two ends of a spectrum. A large language model (LLM) learns to reason step-by-step when data is structured such that the next token depends on a small amount of preceding context. Inference in LLMs resembles pattern recognition when th...

---

### 28. [CurveTQ: Rotation-Free Trellis Quantization of LLM Weights via Curvature-Weighted Search](https://arxiv.org/abs/2610.09212)

**Authors**: Guanhua Ding, Zi Wang, Ruichao Li, Jack Liu  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.09212v1  

#### Abstract
The best two-bit weight quantizers for large language models, such as QTIP and Proteus, rotate each weight matrix by a random orthogonal transform, which must be undone at every decoding step, then encode it with a trellis or lattice code under a Euclidean search; the layer Hessian enters only throu...

---

### 29. [Efficient Best-of-N policy evaluation for inference-time alignment](https://arxiv.org/abs/2610.09250)

**Authors**: Jonas Schweisthal, Yuxin Wang, Athiya Deviyani, Stefan Feuerriegel, Dennis Frauen  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.09250v1  

#### Abstract
Best-of-N (BoN) is a common inference-time alignment method that selects the highest-scoring response among N samples from a reference model. Evaluating BoN policies from logged data is challenging under sample-only access because standard off-policy estimators require density ratios that depend on ...

---

### 30. [Multimodal LLMs Can Learn to Read Brain Signals: A Vision--Language Model for Unified Multi-Task EEG Decoding](https://arxiv.org/abs/2610.09355)

**Authors**: Parastoo Azizeddin, Omid Sharafi, Maryam M. Shanechi  
**Category**: cs.LG  
**Published**: 2026-10-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.09355v1  

#### Abstract
Learning EEG representations that generalize across cognitive tasks, subjects, and recording conditions remains a key challenge in electroencephalography (EEG) decoding. Recent advances in foundation models have improved EEG decoding performance, yet a fundamental open question remains: how to effec...

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

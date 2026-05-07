# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-07 08:12:28 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Piper: Efficient Large-Scale MoE Training via Resource Modeling and Pipelined Hybrid Parallelism](https://arxiv.org/abs/2605.05049)

**Authors**: Sajal Dash, Feiyi Wang  
**Category**: cs.DC  
**Published**: 2026-05-07  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2605.05049v1  

#### Abstract
Frontier models increasingly adopt Mixture-of-Experts (MoE) architectures to achieve large-model performance at reduced cost. However, training MoE models on HPC platforms is hindered by large memory footprints, frequent large-scale communication across heterogeneous networks, and severe workload im...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Piper: Efficient Large-Scale MoE Training via Resource Modeling and Pipelined Hybrid Parallelism**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前在 **HPC 平台**上训练大规模 **Mixture-of-Experts (MoE)** 模型面临三大挑战：
- **高通信开销**：专家并行（Expert Parallelism, EP）导致频繁的 `all-to-all` 通信，在非均匀网络拓扑（如 Dragonfly）下延迟显著。
- **负载不均衡**：路由机制导致部分 GPU 负载过重，而其他 GPU 利用率低，尤其在训练初期。
- **计算效率低下**：细粒度专家（fine-grained experts）产生大量“tall-and-skinny” GEMM 运算，硬件利用率差。
- **缺乏平台感知的混合并行策略**：现有框架未系统建模内存、计算与通信的耦合关系。

### 🚀 提出的新方法与思路
作者提出 **Piper**，一个面向 HPC 平台的大规模 MoE 训练框架，其核心创新包括：

#### （1）**资源建模驱动的训练策略选择（Analytical Resource Modeling）**
- 构建数学模型量化不同并行配置下的 **memory、compute、communication 开销**。
- 结合实测平台参数（带宽、延迟、吞吐），自动筛选可行且高效的 `(PP, EP)` 配置。

#### （2）**基于 Pipeline Parallelism 的通信本地化（Pipelined Hybrid Parallelism）**
- 将 **Pipeline Parallelism (PP)** 引入 MoE 层内部，构建 **PP × EP 二维设备网格**。
- 限制 `all-to-all` 通信范围至单个 pipeline stage 内部（通常为单节点内），显著降低跨节点通信压力。

#### （3）**拓扑感知的高效 All-to-All 算法（HALO）**
- 设计 **Dragonfly-topology-aware hierarchical all-to-all** 算法：
  - 分三个阶段：**intra-node → inter-node → intra-node redistribution**。
  - 利用 GPU-NIC 亲和性，饱和多 NIC 带宽。
  - 支持并发执行以隐藏通信延迟。

#### （4）**动态专家迁移实现负载均衡（Expert Migration）**
- 定期监控各 GPU 上专家的 token 分配情况。
- 当负载偏差超过阈值时，触发轻量级 **专家迁移算法**，在 EP 组内重新分配专家。
- 成本仅占总训练时间 <5%，有效缓解负载倾斜。

#### （5）**端到端验证：万亿参数 MoE 模型训练**
- 在 **Frontier 超算**上成功训练 **trillion-scale MoE 模型**，达到 **20% MFU**，远超 X-MoE 的 5.23%。

### 🔍 相比现有方法的优势
| 方面 | 现有方法（如 X-MoE, DeepSpeed-MoE） | Piper |
|------|-------------------------------|-------|
| 并行策略 | 主要依赖 EP + TP/DP，无 PP 优化 | 引入 PP + EP 混合并行，局部化通信 |
| 通信优化 | 使用 flat all-to-all（NCCL/RCCL） | 拓扑感知 HALO 算法，提升 1.5–9× 带宽 |
| 负载均衡 | 依赖路由层辅助损失或静态 padding | 动态物理迁移专家，更彻底平衡负载 |
| 平台适配 | 缺乏系统级建模 | 基于实测资源建模，自动选择最优配置 |
| 可扩展性 | 大规模下 MFU 下降至 ~5% | 达到 20–50% MFU，支持 trillion 参数 |

---

## 2. 核心实验方法和设置

### 📊 数据集与模型
- **未使用传统 NLP 数据集**（如 WikiText、C4），而是聚焦于 **真实 MoE 架构的训练流程模拟与实测**。
- 实验基于多个 **state-of-the-art MoE 模型架构**，包括：
  - **DeepSeek-V2/V3**
  - **Mixtral 8×7B / 8×22B**
  - **Qwen3-30B-A3B / 235B-A22B**
  - **Llama 4 Scout/Maverick**
  - **Kimi K2 (~1T 参数)**

### ⚙️ 实验设置
- **硬件平台**：**Frontier 超级计算机**（AMD Instinct MI250X GPU，NVLink + Infinity Fabric，Dragonfly 网络拓扑）
- **并行维度**：
  - Pipeline Parallelism (PP)
  - Expert Parallelism (EP)
  - Data Parallelism (DP)
- **微基准测试（Micro-benchmarking）**：
  - 测量单卡 attention 与 FFN GEMM 吞吐
  - 测试不同规模下 `all-to-all` 和 P2P 通信带宽/延迟
- **评估指标**：
  - **MFU (Model FLOPs Utilization)**：核心性能指标，反映实际计算效率。
  - 通信延迟、内存占用、训练吞吐（TFLOPs）

### 🆚 基线方法对比
- **X-MoE**：当前最先进的细粒度 MoE 训练框架，作为主要对比对象。
- **DeepSpeed-MoE / DeepSpeed-TED**
- **Tutel**
- **PyTorch + RCCL 默认 all-to-all**

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 指标 | 结果 |
|------|------|
| **MFU 提升** | Piper 实现 **2–3.5× 更高的 MFU** 对比 X-MoE |
| **All-to-All 带宽提升** | HALO 算法实现 **1.2×–9×** 优于 vendor RCCL 实现 |
| **最大模型规模** | 成功训练 **1.7 trillion 参数 MoE 模型**（1024 GPUs） |
| **峰值 MFU** | 多个 SOTA 模型达到 **20–50% MFU**；X-MoE 报告仅为 5.23%（545B 模型） |
| **弱扩展效率** | 从 512 到 1024 GPUs，扩展效率达 **73%** |

### 📊 与基线方法对比（Figure 13）
- 在 **small (10B) 到 super (545B)** 四种规模模型上测试：
  - Piper 仅需 **8–512 GPUs** 即可完成训练。
  - X-MoE 需要 **256–1024 GPUs**，且吞吐更低。
  - **Piper 吞吐是 X-MoE 的 2–3.6×**。

### 🔬 消融实验（Ablation Studies）
虽然文中未明确列出“消融表”，但通过以下分析体现各组件价值：

#### （1）Pipeline + EP vs 纯 EP
- 纯 EP 导致 `all-to-all` 跨越数百 GPU，通信瓶颈严重。
- 加入 PP 后，通信组缩小至单节点内，**大幅减少跨节点流量**，提高带宽利用率。

#### （2）HALO vs RCCL all-to-all
- 在 ≥16 节点时，HALO 显著优于 RCCL（最高 **9× 低延迟**）。
- 原因：RCCL 忽视 Dragonfly 拓扑，造成跨机柜拥塞；HALO 显式分组调度。

#### （3）Expert Migration 开销
- 最坏情况下迁移耗时约 **几十毫秒**（见 Table IV）。
- 实际采用增量迁移，平均开销 <5% 总训练时间，但带来显著负载均衡收益。

#### （4）单层训练天花板（Figure 11）
- 所有模型均可将一层完整放入单个 Frontier 节点。
- 单层训练 MFU 达到 **78–130 TFLOPs**，为全模型训练设定了理论上限。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Pipeline Parallelism 可有效用于 MoE 层内通信优化**，而非仅限于层间划分。
2. **平台感知的资源建模至关重要**：脱离具体 HPC 拓扑设计的训练策略难以发挥硬件潜力。
3. **拓扑感知 all-to-all 是突破通信瓶颈的关键**：在 Dragonfly 等异构网络中，flat all-to-all 表现极差。
4. **动态专家迁移成本可控且效果显著**：打破“专家必须静态分布”的思维定式。
5. **万亿参数 MoE 模型可在现有超算上高效训练**：Piper 实现了 **20% MFU**，证明可行性。

### ⚠️ 方法的局限性
- **依赖较强工程实现能力**：需深度集成 PyTorch Distributed、Tutel、自定义通信原语。
- **对 pipeline bubble 敏感**：若 PP 设置不当，仍可能引入较大空泡开销。
- **目前主要针对 fine-grained MoE 优化**：对 coarse-grained MoE（如 Mixtral）增益相对较小。
- **尚未开源完整代码**：仅提供算法描述与性能曲线。

### 🔮 未来工作方向
- 支持更多并行组合（如 TP + EP + PP 三维并行）。
- 自动调优 pipeline schedule（如 ZB-H1/H2）以进一步压缩 bubble。
- 结合专家稀疏更新或激活卸载技术，进一步降低内存压力。
- 推广至其他具有“子模块动态激活”特性的模型架构（如 conditional computation）。

---

> **总结一句话**：  
> **Piper 通过“资源建模 + 拓扑感知通信 + pipeline 化专家并行”，首次实现了在 HPC 平台上高效训练 trillion-scale MoE 模型，将 MFU 提升至 20% 以上，较现有框架提速 2–3.5×，为下一代超大规模语言模型训练提供了新范式。**

</details>

---

### 2. [FASQ: Flexible Accelerated Subspace Quantization for Calibration-Free LLM Compression](https://arxiv.org/abs/2605.04084)

**Authors**: Ye Qiao, Yian Wang, Zhiheng Chen, Hyoukjun Kwon, Sitao Huang  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.04084v1  

#### Abstract
Compressing large language models (LLMs) for deployment on commodity GPUs remains challenging: conventional scalar quantization is limited to fixed bit-widths (e.g., 8/4/3-bit), offers only a few discrete compression points, and typically requires calibration data. We present FASQ (Flexible Accelera...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FASQ: Flexible Accelerated Subspace Quantization for Calibration-Free LLM Compression

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大语言模型（LLM）压缩面临三大挑战：
- **固定比特限制**：主流 scalar quantization（如 INT4、INT8）仅支持离散比特宽度，无法提供连续的压缩率选择，导致在特定内存预算下缺乏灵活的权衡空间。
- **依赖校准数据（calibration data）**：GPTQ、AWQ、SmoothQuant 等先进方法需使用代表性输入数据进行校准以优化量化参数，这对专有或领域特定模型不现实。
- **推理开销高**：scalar 方法需要在推理时显式地 **dequantization**（将低精度权重还原为 FP16），带来额外计算与内存带宽负担，尤其在非标准比特下无高效 kernel 支持。

### 🚀 提出的新方法：FASQ
FASQ（**Flexible Accelerated Subspace Quantization**）是一种**无需校准的后训练压缩框架**，基于 **Product Quantization（PQ）** 技术对 LLM 权重矩阵进行压缩。

#### 核心思想：
- 将每个权重矩阵划分为多个子向量子空间（subspace），在每个子空间内使用 **k-means 聚类**生成共享码本（codebook）。
- 每个子向量用一个 **uint8 索引**表示其最近邻的聚类中心，从而实现压缩。
- 推理时不重建原始 FP16 权重，而是直接通过索引查表并计算点积，实现 **reconstruction-free inference**。

#### 关键参数控制压缩粒度：
- `sub-vector size`（SZss）：子向量长度
- `codebook cardinality`（Ks）：每子空间聚类数  
通过调节这两个参数，FASQ 可实现从 **27% 到 49% FP16 模型大小**的连续压缩范围。

### 🔍 相比现有方法的优势
| 维度 | FASQ | Scalar Quantization（如 GPTQ/AWQ） |
|------|------|-------------------------------|
| **压缩灵活性** | ✅ 连续设计空间（fine-grained trade-off） | ❌ 仅离散比特（3/4/8-bit） |
| **是否需要校准数据** | ✅ **完全不需要**（仅依赖权重分布聚类） | ❌ 必须提供 calibration 数据 |
| **推理效率** | ✅ **无需 dequantization**，内存访问减少 4× | ❌ 需要 unpack + dequantize，增加 ALU 和带宽开销 |
| **decode 吞吐** | ✅ **超越 FP16 tensor-core 性能** | ❌ 均低于 FP16 |
| **kernel 复用性** | ✅ 单一 CUDA kernel 支持所有配置 | ❌ 不同比特需不同 kernel |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **零样本任务评估（zero-shot accuracy）**：
  - ARC-easy / ARC-challenge
  - HellaSwag
  - PIQA
  - WinoGrande
- **困惑度评估（perplexity）**：
  - WikiText-2 test set

### ⚙️ 实验设置与评估指标
| 类别 | 内容 |
|------|------|
| **硬件平台** | NVIDIA RTX 3090 GPU |
| **测试模型** | Meta-Llama-3-8B、Qwen3-8B、Qwen3.5-9B-Base<br>（扩展测试：LLaMA-2 7B/13B） |
| **评估模式** | 端到端推理延迟（prompt=128, gen=128） |
| **主要指标** | 
| - 模型大小（Size%，相对于 FP16）<br>- Perplexity（PPL）<br>- Zero-shot 平均准确率（AvgT）<br>- Decode 吞吐（tok/s）<br>- Prefill 延迟（ms）<br>- 显存占用（MB） |

### 🔁 基线方法对比
| 方法 | 是否需校准 | 比特位宽 | 特点 |
|------|------------|----------|------|
| **FP16** | – | 16-bit | 原始精度基准 |
| **RTN** | ✅（无） | 4/3-bit | Round-To-Nearest，最简单 baseline |
| **GPTQ** | ❌ | 4/3-bit | Hessian-guided，依赖 calibration |
| **AWQ** | ❌ | 4/3-bit | Activation-aware scaling |
| **SmoothQuant** | ❌ | W8A8/W6A6/W4A4 | 联合权重重建与激活量化 |
| **QuIP** | ❌ | 4/3-bit | 引入 incoherence processing 提升重建质量 |

> 所有 baseline 使用官方实现；FASQ 无需任何外部数据。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Meta-Llama-3-8B, RTX 3090）

| 方法 | Size% | AvgT (%) | Decode (tok/s) | Mem (MB) | vs FP16 Speed |
|------|--------|-----------|----------------|-----------|----------------|
| FP16 (Tensor Core) | 100.0% | 68.8 | 43.9 | 15,317 | 1.00× |
| AWQ (4-bit) | 35.7% | 67.7 | 28.1 | 5,463 | 0.64× |
| GPTQ (4-bit) | 35.7% | 67.3 | 20.5 | 6,558 | 0.47× |
| **FASQ (eff. 4-bit)** | **37.0%** | **67.8** | **45.2** | **5,975** | **1.03×** ✅ |
| **FASQ (eff. 3-bit)** | **33.2%** | **66.0** | **51.8** | **5,482** | **1.18×** ✅ |

> ✅ **FASQ 是唯一 decode 速度超过 FP16 的压缩方法**

### 🔁 与基线方法对比结果
- **准确性方面**：
  - 在 ~49% 模型大小下，FASQ 2-1024 达到 **68.2 AvgT**，媲美 SmoothQuant W8A8，但节省 8% 存储且无需校准。
  - 在 4-bit 区域（~37% size），FASQ 2-256 (**67.8**) 超过 GPTQ-4 (**67.3**) 和 QuIP-4 (**67.1**)。
  - 在 3-bit 区域，FASQ 2-128 (**66.0**) 显著优于 AWQ-3 (**64.4**) 和 QuIP-3 (**63.7**)。
  - SmoothQuant 在 W4A4 下崩溃（PPL >4000，AvgT=35.5），而 FASQ 质量下降平滑。

- **推理性能方面**：
  - FASQ decode 吞吐：
    - 比 AWQ 快 **1.6–1.8×**
    - 比 GPTQ 快 **2.2–2.5×**
    - 比 RTN 快 **4.3–5.0×**
  - 内存节省：
    - 有效 4-bit：**2.56× 减少**
    - 有效 3-bit：**2.80× 减少**

- **跨模型泛化性**（见 Table 3）：
  - 在 Qwen3-8B 和 Qwen3.5-9B 上表现一致优异，证明方法通用性强。
  - 例如 FASQ 2-1024 在三者上均保持接近 FP16 的 PPL 增量（+0.2~0.3）。

### 🔬 消融实验结果（Ablation Study）
基于单层 4096×4096 GEMM/GEMV 微基准测试（Table 1 & Figure 3）：

| 优化项 | 效果 |
|-------|------|
| **LUT-free direct compute（GEMV）** | 消除共享内存 LUT，降低延迟至 32μs（vs cuBLAS 45μs） |
| **Split-K parallelism** | 提升 GPU occupancy，尤其在短序列 prefill 中提速 1.35× |
| **Double-buffered LUT（GEMM）** | 重叠 LUT 构建与计算，减少同步开销 |
| **half2 vectorized load** | 启用 SZss=2 时的向量化加载，显著提升效率 |
| **Sub-vector size (SZss)** | 最优值为 **2**：SZss=1 导致索引流量翻倍（16MB），性能下降 3× |
| **Codebook size (Ks)** | 对 decode 延迟影响小（25.3~37.4μs），允许自由调参保质量 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **FASQ 实现了真正灵活的压缩-质量连续权衡**：首次填补 scalar quantization 在 3-bit 与 4-bit 之间的“空白地带”，支持从 27% 到 49% FP16 大小的精细调节。
2. **无需校准即可达到 SOTA 精度**：仅靠 k-means 聚类权重分布，即可匹敌甚至超越依赖 calibration 的 GPTQ/AWQ。
3. **推理加速突破瓶颈**：由于采用 **reconstruction-free + LUT-free GEMV kernel**，decode 阶段内存访问减少 4×，**实际吞吐超越 FP16 tensor-core**，这是前所未有的成就。
4. **适用于消费级 GPU 实时推理**：在 RTX 3090 上实现超 45 tok/s 的 decode 速度，使高质量 LLM 可部署于单张消费卡。

### ⚠️ 局限性
- **Prefill 阶段较慢**：当前 GEMM kernel 基于 LUT 查找，无法利用 tensor core，导致 prefill 延迟远高于 FP16（如 935ms vs 41ms）。
- **不适合极长上下文预填充场景**：若应用强调 **Time-to-First-Token (TTFT)**，则 prefill 成为瓶颈。
- **码本存储开销随模型增大而稀释**：虽对小模型影响可控，但在超大规模模型中可能更优。

### 🔮 未来工作方向
1. **加速 Prefill 阶段**：
   - 设计 **pipelined reconstruction kernel**，边重建 tile 边执行 tensor-core GEMM，避免全量 materialization。
   - 探索 **Triton-based codegen**，针对 PQ 访问模式定制 tile 策略。
2. **进一步提升压缩率与精度**：
   - 引入 **per-layer 参数分配**（adaptive SZss/Ks selection）
   - 探索 **post-training codebook fine-tuning** 以恢复更多精度
3. **扩展至其他模态与架构**：应用于 Vision Transformer、MoE 模型等。

---

## ✅ 总结一句话
> **FASQ 是首个实现“无需校准 + 连续压缩 + decode 加速超越 FP16”的 LLM 压缩框架，在保持 SOTA 精度的同时，为消费级 GPU 上的实时大模型推理提供了全新可能。**

</details>

---

### 3. [RLearner-LLM: Balancing Logical Grounding and Fluency in Large Language Models via Hybrid Direct Preference Optimization](https://arxiv.org/abs/2605.04539)

**Authors**: Qiming Bao, Juho Leinonen, Paul Denny, Michael J. Witbrock  
**Category**: cs.CL  
**Published**: 2026-05-07  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.04539v1  

#### Abstract
Direct Preference Optimization (DPO), the efficient alternative to PPO-based RLHF, falls short on knowledge-intensive generation: standard preference signals from human annotators or LLM judges exhibit a systematic verbosity bias that rewards fluency over logical correctness. This blindspot leaves a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*RLearner-LLM: Balancing Logical Grounding and Fluency in Large Language Models via Hybrid Direct Preference Optimization*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

本论文针对 **知识密集型生成任务**（如教育解释生成）中，当前主流对齐方法 **Direct Preference Optimization (DPO)** 存在的根本性缺陷：

- **标准偏好信号存在“流畅性偏见”**（verbosity bias）：无论是人类标注者还是 LLM-as-a-judge，都倾向于更长、修辞华丽但逻辑不严谨的文本，而忽略其是否真正蕴含正确答案。
- 导致模型虽然输出**语言流畅、自信**，但推理链条缺失，**逻辑蕴含度（NLI entailment）极低**（SFT 模型仅 0.05–0.22），形成“**对齐税**”（alignment tax）——即提升逻辑性会牺牲流畅性，反之亦然。

### 🚀 提出的新方法：RLearner-LLM 与 Hybrid-DPO

提出 **RLearner-LLM** 框架，通过 **Hybrid-DPO**（混合直接偏好优化）解决上述问题：

- **核心思想**：构建一个**双信号融合的自动化奖励函数**，替代易受偏见影响的人类/LLM 偏好标签。
- **Hybrid Reward 公式**：
  $$
  H(E) = \alpha \cdot S_{\text{NLI}}(E) + (1-\alpha) \cdot S_{\text{verifier}}(E)
  $$
  或乘法形式（带 ACR 门控）：
  $$
  H_M(E) = (w_{\text{nli}} S_{\text{NLI}} \cdot w_{\text{ver}} S_{\text{ver}} - \gamma \ell) \cdot \mathbf{1}[\text{ACR}(E) \geq 0]
  $$
  - `S_NLI`：基于 DeBERTa-v3 的 NLI 蕴含概率，衡量**逻辑正确性**。
  - `S_verifier`：经微调的 Alpaca-7B 验证器打分，衡量**教学质量和语言流畅性**。
  - `ACR`（Answer Coverage Rate）：确保答案被明确提及。
  - `长度惩罚项`：抑制冗长幻觉。

- **无需人工标注**：完全自动构造 preference pairs，避免主观偏见。
- **Dual-Signal Hypothesis**：逻辑与流畅性是互补目标，而非对立；双信号可协同推动模型向帕累托前沿（upper-right quadrant）进化。

### 🔍 相比现有方法的优势

| 对比维度 | 传统 DPO / RLHF | RLearner-LLM (Hybrid-DPO) |
|---------|------------------|----------------------------|
| 奖励信号来源 | 人类/LLM judge（有 verbosity bias） | 自动化双信号（NLI + verifier） |
| 是否需人工标注 | 是 | 否 |
| 逻辑性（NLI） | 弱（依赖 fluency） | 显著提升（最高达 6.6×） |
| 流畅性 | 高（但可能空洞） | 保持甚至提升（无 alignment tax） |
| 可扩展性 | 依赖昂贵标注 | 完全自动化，易于部署 |

---

## 2. 核心实验方法和设置

### 📚 数据集

- **训练数据（SFT 阶段）**：
  - 来自 **PeerWise** 平台的 13,211 条本科生撰写的多选题解释（非专家编写，反映真实学习者思维）。
  - 跨五个学术领域：**Cardiff Biology, Sydney Biology, Auckland Law, UK Medicine Year 1, UK Medicine Year 2**。
- **测试集**：
  - 每个领域保留 **100 道题目**作为测试集。

### ⚙️ 实验设置

- **基础模型**（Base Architectures）：
  - `LLaMA-2-13B`
  - `Qwen3-8B`
  - `Gemma 4 E4B-it`（约 4.5B 有效参数）
- **训练流程**：
  1. **SFT**：监督微调，3 轮。
  2. **Preference Data 构造**：
     - 每个问题生成 3 个候选解释。
     - 使用 Hybrid Reward 打分并构造 preference pairs（gap > 0.05）。
  3. **Hybrid-DPO 微调**：5 轮，LoRA 微调。
- **变体选择策略**：
  - 若候选池较小或跨领域混合 → 使用加法形式 `HA`
  - 否则使用乘法形式 `HM`（更强过滤）

### 📊 评估指标

| 指标 | 含义 |
|------|------|
| **NLI** | 使用 `cross-encoder/nli-deberta-v3-small` 计算解释 E 是否蕴含正确答案 A，范围 [0,1] |
| **ACR**（Answer Coverage Rate） | 正确答案是否被明确提及（二值） |
| **BERTScore(Ans)** | 解释与正确选项文本的语义相似度 |
| **BLEU / BERTScore(Stu)** | 与学生参考解释的重叠度 |
| **Verifier Score** | Alpaca-7B 验证器给出的教学质量评分（1–5 分） |
| **Pairwise Comparison** | 使用 GPT-4o-mini 进行盲测比较（A/B test） |

### 🆚 基线方法

- **SFT Checkpoint**：各模型的初始微调版本。
- **DPO v1/v2**：仅使用 verifier 或 NLI 单一信号的 ablation。
- **ILearner-LLM (K=5)**：迭代增强方法（5 次 refine），计算成本为单次推理的 5 倍。
- **GPT-4o-mini**：作为外部 judge 和对比模型。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（NLI 改进）

| 领域 | LLaMA-2-13B | Qwen3-8B | Gemma 4 E4B-it |
|------|-------------|----------|----------------|
| **Cardiff Biology** | 0.0555 → **0.3209** (**5.8×**) | 0.1959 → 0.1820 | 0.2117 → **0.3505** (**+66%**) |
| **Sydney Biology** | 0.0537 → **0.3562** (**6.6×**) | 0.1737 → 0.2284 (+31%) | 0.2469 → 0.2309 |
| **Auckland Law** | 0.2702 → 0.3229 (+19%) | 0.3191 → 0.2303 | 0.3911 → **0.4377*** (**+12%**) |
| **UK Medicine Y1** | 0.0860 → **0.4251** (**4.9×**) | 0.2457 → 0.2104 | 0.2962 → 0.3910 (+32%) |
| **UK Medicine Y2** | 0.2319 → 0.3885 (+68%) | 0.1632 → 0.2009 (+23%) | 0.1604 → **0.3892** (**2.4×**) |

> ✅ **总体表现**：在 **15 个（架构, 领域）组合中，有 11 个实现 NLI 提升**，最大提升达 **6.6 倍**。

### 🆚 与基线方法对比

| 对比项目 | 结果 |
|--------|------|
| **vs. SFT** | 在 Qwen3-8B 上赢得 **95%** 的盲测比较（GPT-4o-mini 判定） |
| **vs. ILearner-LLM (K=5)** | **Gemma 4 E4B-it** 在 **Auckland Law** 上首次以单次推理超越迭代方法（0.4377 vs 0.3996） |
| **vs. GPT-4o-mini** | RLearner-LLM 输出输给 GPT-4o-mini（95% 失败），但作者指出这是因后者更长，再现了 **verbosity bias**，说明 LLM-as-judge 不可靠 |
| **推理速度** | Gemma 4 E4B-it 推理最快（4.76s/q），且小模型上仍有效 |

### 🔬 消融实验结果

| 实验 | 发现 |
|------|------|
| **仅用 NLI 优化** | 产生短、重复、机械的答案（alignment tax） |
| **仅用 verifier 优化** | 回归到 verbosity bias，逻辑性下降 |
| **HA vs. HM** | 两者平均差距 <1 pp，**HM 在多数情况下略优（7/11）**，支持“双信号假设”是关键，而非代数形式 |
| **Tier-B 更严格过滤**（高质量问题） | 尽管训练数据减少 7×，但 **NLI 提升 48%**，验证高质量数据的重要性 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **DPO 的根本问题不在算法，而在奖励信号**：
   - 当前偏好信号无法区分“**流畅但错误**”与“**简洁但正确**”的解释。
   - **LLM-as-a-judge 存在系统性 verbosity bias**，即使控制长度也无法消除。

2. **Hybrid-DPO 成功打破 alignment tax**：
   - 双信号设计使模型既能保持语言质量，又能显著提升逻辑严密性。
   - 在多个架构和领域均取得一致改进，尤其在小模型（Gemma 4 E4B-it）上表现突出。

3. **逻辑性可通过自动化指标衡量**：
   - **NLI + ACR** 比 LLM judge 更适合作为知识密集任务的评估标准。
   - 提倡使用 **logic-aware automatic metrics** 替代主观判断。

4. **小模型也能实现强对齐**：
   - Gemma 4 E4B-it（4.5B 参数）在部分任务上超越大模型迭代方法，证明该方法可**向下兼容紧凑模型**。

### ⚠️ 局限性

1. **Auckland Law 仍是挑战**：
   - LLaMA-2-13B 版本未超越 ILearner-LLM (K=5)，表明某些复杂领域仍需迭代机制。

2. **潜在循环评估风险**：
   - NLI 评估使用与训练相同的 `DeBERTa-v3-small`，可能存在过拟合。建议未来使用更大模型（如 RoBERTa-large）进行验证。

3. **SFT 数据质量限制**：
   - 使用本科生编写的 PeerWise 数据，存在部分错误或不完整解释，限制了上限。未来应使用专家标注数据。

4. **ACR/NLI 权衡现象**：
   - 在 Gemma 4 上观察到 ACR 下降而 NLI 上升，提示需进一步优化双信号权重。

### 🔮 未来工作方向

- 引入 **迭代 refinement 机制** 到 Hybrid-DPO 框架中，应对高难度领域。
- 使用 **更强的 NLI 模型**（如 DeBERTa-v3-large）作为训练信号。
- 开发 **专家级 SFT 数据集**，突破当前性能天花板。
- 探索 **动态调整双信号权重** 策略，适应不同任务需求。
- 将框架推广至其他知识密集场景（如医疗诊断、法律咨询）。

---

> 💡 **一句话总结**：  
> RLearner-LLM 通过 **Hybrid-DPO** 框架，用 **NLI + verifier** 双信号替代易偏的人类/LLM 偏好，成功在不牺牲流畅性的前提下大幅提升 LLM 的逻辑严谨性，解决了 DPO 在知识密集任务中的“对齐税”难题，并证明小模型也能实现高质量逻辑对齐。

</details>

---

### 4. [Efficient Handwriting-Based Alzheimer,s Disease Diagnosis Using a Low-Rank Mixture of Experts Deep Learning Framework](https://arxiv.org/abs/2605.04079)

**Authors**: Wu Wang, Yuang Cheng, Fouzi Harrou, Ying Sun  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.04079v1  

#### Abstract
Early and reliable detection of Alzheimer's disease (AD) is crucial for timely clinical intervention and improved patient management. It also supports the evaluation of emerging therapeutic strategies. In this paper, we propose a Low-Rank Mixture of Experts (LoRA-MoE) deep learning framework for Alz...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
- **早期阿尔茨海默病（Alzheimer's Disease, AD）诊断**依赖昂贵且侵入性的手段（如神经影像、脑脊液检测），难以实现大规模筛查。
- 现有基于深度学习的方法在处理高维、异质性强的**handwriting signals**时存在以下挑战：
  - 参数量大，计算成本高；
  - 易过拟合，尤其在临床小样本数据上；
  - 单一网络难以捕捉多样化的认知-运动模式。

### 提出的新方法与新思路
提出了一种名为 **Low-Rank Mixture of Experts (LoRA-MoE)** 的新型深度学习框架，用于基于手写信号的AD诊断。其核心思想是将 **Mixture of Experts (MoE)** 架构与 **Low-Rank Adaptation (LoRA)** 技术相结合：

- **共享基础网络（Shared Base Network）**：所有专家共享一个主干网络，负责提取通用特征，减少参数冗余。
- **轻量级低秩适配器（LoRA Adapters）**：每个“专家”仅通过一对低秩矩阵 $A$ 和 $B$ 进行任务特定的微调，显著降低可训练参数数量。
- **稀疏门控机制（Sparse Gating Network）**：采用 Top-K 路由策略（文中主要用 Top-1），为每个输入动态选择最相关的专家进行推理，提升效率。

### 相比现有方法的优势
| 特性 | 传统 MoE | MLP | LoRA-MoE |
|------|----------|-----|-----------|
| 可训练参数量 | 高（每个专家独立完整网络） | 中等 | **极低**（仅适配器参数） |
| 专家专业化能力 | 强 | 无 | **强**（通过适配器实现） |
| 训练稳定性 | 易受干扰、不稳定 | 稳定 | **更稳定**（初始化保证初始行为一致） |
| 推理效率 | 低（激活多个专家） | 高 | **高**（Top-1路由，激活少量参数） |
| 抗过拟合能力 | 差（参数多） | 中等 | **强**（参数高效 + 共享表示） |

---

## 2. 核心实验方法和设置

### 数据集
- **DARWIN (Diagnosis AlzheimeR WIth haNdwriting) Dataset**
  - 包含 **174 名参与者**：89 名 AD 患者，85 名健康对照（HC）。
  - 手写信号通过 **digitizing tablet** 采集，采样频率为 **200Hz**。
  - 包含 **25 种不同书写任务**，涵盖基本图形、复制/反向复制、记忆/听写等。
  - 提取了 **450 维 handcrafted features**，分为三类：
    - **Time-related features**（总时间、空中时间等）
    - **Movement-related features**（速度、加速度、抖动 gmrt 等）
    - **Pressure-related features**（压力均值、方差等）

### 实验设置与评估指标
- **数据划分**：75% 训练，25% 测试，结果在 25 个任务上平均。
- **评估指标**：
  - Accuracy（准确率）
  - Sensitivity（敏感度，即召回率）
  - Specificity（特异性）
  - AUC（ROC曲线下面积）
  - Precision（精确率）
  - F1 Score
- **模型配置**：
  - LoRA-MoE / MoE 专家数固定为 6，隐藏维度从 50 到 400 变化。
  - LoRA rank 设置为 1–8 进行消融实验。
  - 引入 **Stacking Ensemble** 策略（StackMean, StackMax）进一步提升鲁棒性。

### 基线方法对比
- **MLP (Multilayer Perceptron)**：标准两层全连接网络，作为简单基线。
- **Conventional MoE**：每个专家为独立的全连接子网络，无参数共享。
- **Proposed Method**: LoRA-MoE（带或不带 stacking）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & Table 3）

#### 最佳性能表现（Subject-Level Aggregation）
| Model | Accuracy | Sensitivity | Specificity | F1 Score | AUC |
|-------|----------|-------------|--------------|----------|-----|
| **LoRA-MoE (best)** | **87.14%** | **88.33%** | 85.88% | **87.11%** | ~0.94 |
| Conventional MoE | 84.86% | 86.11% | 83.53% | 84.16% | 0.9108 |
| MLP | 86.57% | 86.67% | 86.47% | 86.47% | 0.9250 |

> 注：LoRA-MoE 在 **5 个专家、hidden dim=300** 时达到最佳性能。

#### Stacking Ensemble 提升效果
- **StackMean / StackMax** 显著提升了 AUC 指标，LoRA-MoE 达到接近 **0.94**，表明集成策略增强了模型判别能力。

### 与基线方法的对比结果
- **性能方面**：
  - LoRA-MoE 在多数配置下优于或媲美 MoE 和 MLP，尤其在平衡敏感度与特异性方面表现更优。
  - 尽管 MLP 在浅层结构中表现良好，但缺乏专家分工机制，无法随模型复杂度扩展而持续提升。
- **效率方面**：
  - LoRA-MoE 的 **训练时间远低于传统 MoE**（见 Fig. 4 & Fig. 5），接近甚至优于 MLP。
  - **推理时激活参数极少**（Top-1 + LoRA），适合部署于资源受限设备（如便携式筛查工具）。

### 消融实验结果

#### a) 隐藏维度影响（Hidden Dimension）
- LoRA-MoE 在中间维度（如 300）达到峰值性能，过大则无明显增益，说明其对容量利用更高效。
- 传统 MoE 表现波动较大，易因参数过多导致过拟合。

#### b) 专家数量影响（Number of Experts）
- LoRA-MoE 在 **5 个专家时性能最优**（Accuracy=87.14%），增加专家数未带来收益，反而增加开销。
- 传统 MoE 随专家数增加性能下降明显，优化困难。

#### c) LoRA Rank 影响
- **Rank=2 时性能最佳**（Accuracy=85.14%），继续增大 rank 导致性能饱和甚至下降。
- 证明 **低秩假设成立**：极少量参数即可有效建模专家差异。

#### d) 多层架构（Depth）实验
- 增加网络深度（5层、8层）并未带来性能提升，反而增加训练时间和复杂度。
- 表明 **适度深度足以捕获 handwritings 中的认知-运动模式**。

#### e) 独立任务预测（Independent Task-Level）
- 所有模型性能下降，验证了跨任务聚合的重要性。
- 但在该设置下，**LoRA-MoE 仍优于 MoE 和 MLP**，显示其更强的鲁棒性。

---

## 4. 关键结论和发现

### 主要发现
1. **LoRA-MoE 是一种高效且强大的 AD 诊断框架**：
   - 成功结合了 MoE 的专家专业化优势与 LoRA 的参数高效特性。
   - 在保持高性能的同时，大幅减少可训练参数和计算开销。

2. **参数效率至关重要**：
   - 在小规模临床数据集（如 DARWIN）上，**过度复杂的模型反而有害**。
   - LoRA-MoE 通过共享主干 + 低秩适配，实现了“少即是多”的效果。

3. **专家数量与 rank 存在最优区间**：
   - 并非越多专家越好，**5 个专家 + rank=2** 即可取得最佳权衡。
   - 支持“适度专业化”而非“过度细分”。

4. **stacking ensemble 提升泛化能力**：
   - 通过聚合多个模型输出，显著提高 AUC 和稳定性，适用于实际医疗场景中的不确定性管理。

5. **handwriting signals 是有效的数字生物标志物（digital biomarker）**：
   - 能够非侵入性地反映早期 AD 引发的认知-运动衰退。
   - 所提方法为 scalable screening 提供了可行路径。

### 方法的局限性
- **数据集规模有限**：DARWIN 仅包含 174 名受试者，限制了模型泛化能力。
- **任务间变异性大**：某些书写任务判别力弱，影响单任务预测准确性。
- **inter-subject variability**：个体书写习惯差异可能掩盖疾病信号。
- **静态路由机制**：当前 gating network 固定，未考虑任务重要性自适应调整。

### 未来工作方向
1. **扩展至多中心、更大规模 handwriting datasets**，增强模型泛化性。
2. **融合 multimodal cognitive biomarkers**：
   - 结合 speech, drawing dynamics, eye-tracking 等信号，构建更全面的诊断系统。
3. **开发 adaptive routing 策略**：
   - 动态选择最具判别力的任务或专家，提升决策灵活性。
4. **自动化 rank selection 机制**：
   - 根据任务难度自动分配 LoRA rank，进一步优化资源利用。
5. **探索 explainability**：
   - 分析各专家对应的具体病理模式（如 tremor 类型、执行延迟等），提升临床可解释性。

---

> ✅ 总结：本文提出的 **LoRA-MoE** 框架在 **accuracy、efficiency、robustness** 之间取得了优异平衡，为基于 handwriting 的 AD 早期筛查提供了一个极具潜力的解决方案，特别适合部署于基层医疗或远程健康监测场景。

</details>

---

### 5. [AxMoE: Characterizing the Impact of Approximate Multipliers on Mixture-of-Experts DNN Architectures](https://arxiv.org/abs/2605.04754)

**Authors**: Omkar B Shende, Marcello Traiola, Gayathri Ananthanarayanan  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.04754v1  

#### Abstract
Deep neural network (DNN) inference at the edge demands simultaneous improvements in accuracy, computational efficiency, and energy consumption. Approximate computing and Mixture-of-Experts (MoE) architectures have each been studied as independent routes towards efficient inference, the former by re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AxMoE: Characterizing the Impact of Approximate Multipliers on Mixture-of-Experts DNN Architectures —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前在深度神经网络（DNN）边缘部署中，**能效**与**精度**之间的权衡至关重要。已有研究分别从两个方向优化：
- **Approximate Computing (AxC)**：通过近似乘法器降低硬件功耗；
- **Mixture-of-Experts (MoE)**：通过动态路由实现条件计算以减少有效计算量。

然而，这两类技术此前**完全独立研究**，其交互影响尚属空白。本文首次系统性地探究了**近似乘法器对 MoE 架构的影响**，填补了这一关键研究缺口。

### ✅ 提出了什么新方法或新思路
提出 **AxMoE** 框架，是首个研究 **Approximate Multipliers 在 MoE DNN 中行为特征** 的实证框架。其核心思想包括：
- 将 EvoApproxLib 中的 8-bit 近似乘法器引入 MoE 架构进行仿真；
- 覆盖三种主流 MoE 变体：**Hard MoE**, **Soft MoE**, 和 **Cluster MoE**；
- 分析不同架构下（CNN 与 ViT）的误差传播机制、恢复能力及能效表现。

### ✅ 相比现有方法的优势
- **跨范式融合**：首次将 AxC 与 MoE 动态推理结合，揭示二者协同设计潜力；
- **全面评估体系**：涵盖无重训练下的鲁棒性分析 + 近似感知重训练后的恢复能力；
- **细粒度建模**：引入 **Normalized Power** 指标统一比较不同拓扑的能效边界；
- **揭示非直观现象**：如某些 MoE 在特定条件下优于 Dense 模型，挑战传统认知。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 模型类别 | 数据集 | 规模 |
|--------|-------|-----|
| CNNs (ResNet-20, VGG11_bn, VGG19_bn) | **CIFAR-100** | 50K 训练 / 10K 测试，32×32 RGB 图像，100 类 |
| ViT-Small | **Tiny ImageNet-200** | 100K 训练 / 10K 验证 / 10K 测试，64×64 RGB 图像，200 类 |

### ⚙️ 实验设置
- **近似乘法器来源**：来自 **EvoApproxLib** 库的 8 个 8-bit signed 近似乘法器（含一个精确基准 `mul8s_1KV6`），覆盖从轻微到激进的近似程度（最高达 52.9% 功耗节省）；
- **模型架构**：
  - CNNs：ResNet-20, VGG11_bn, VGG19_bn
  - Transformer：ViT-Small（12 层）
- **MoE 变体**：
  - **Hard MoE**：Top-1 路由，仅激活一个专家
  - **Soft MoE**：所有专家加权输出
  - **Cluster MoE**：图像级路由，使用独立 Gateway 网络选择专家
- **近似范围**：
  - CNNs：应用于 Conv2d 层（占总 MACs 的 98–99.9%）
  - ViT-Small：应用于 FFN 和 Self-Attention 中的 Linear 层（共占 ~97.3% MACs），但 **Gateway 和 BatchNorm 保持精确**

### 🎯 评估指标
| 指标 | 定义 |
|------|------|
| **Top-1 Accuracy** | 主要精度指标 |
| **Effective MACs (Meff)** | 单次推理实际执行的操作数（考虑稀疏性） |
| **Static MACs** | 模型最大容量（所有专家之和） |
| **Normalized Power (Pnorm)** | 相对于 Dense + KV6 的相对功耗：<br>$ P_{\text{norm}} = \frac{M_{\text{eff}}}{M_{\text{base}}} \left(f_{\text{apx}} \cdot \frac{P_{\text{apx}}}{P_{\text{kv6}}} + (1 - f_{\text{apx}})\right) $ |
| **Pareto Frontier** | 在准确率 vs. 归一化功耗图中，无法被同时超越的最优配置集合 |

### 🔁 重训练策略
- **Approximate-aware Retraining**：使用 LUT 查表模拟近似乘法，在 PyTorch 中嵌入误差行为；
- **训练参数**：SGD，lr=0.1，weight decay=5e-4，batch size=128，**仅训练 5 个 epoch**（因 LUT 推理开销高）；
- **冻结 Gateway 参数**：确保路由决策不受近似硬件干扰。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 模型 | 设置 | Top-1 Acc (%) | Normalized Power | 备注 |
|------|------|----------------|------------------|------|
| ResNet-20 (Dense) | KV6 (exact) | 68.58 | 1.00 | 基准 |
| ResNet-20 (Dense) | L2J (29.2% save) | 68.31 | 0.71 | 几乎无损 |
| ResNet-20 (all) | after retraining (L2L) | ~68.x | ~0.47 | **全恢复！** |
| VGG11_bn (Cluster MoE) | L2L | 42.5 | — | 唯一能在极端近似下部分存活的 VGG MoE |
| ViT-Small (Hard MoE, r=0.25) | L2L | **75.97** | **0.485** | **优于 Dense (75.00)** |
| ViT-Small (Dense) | L2L | 75.00 | 0.485 | — |
| ViT-Small (Cluster MoE) | L2L | 68.36 | 1.46 | 被 Hard MoE 全面压制 |

---

### 🔍 与基线方法的对比结果

#### ✅ 无重训练时（Intrinsic Resilience）
- **CNN 架构**：
  - **Dense 最具鲁棒性**：在 ResNet/VGG 上均显著优于各类 MoE；
  - MoE 对近似极其敏感，尤其 Hard/Cluster MoE 在 KV9 后即崩溃至 <20%；
  - 异常点：**KVM 乘法器下 Soft MoE 表现异常稳定**（误差概率仅 49.8%，但节能 13.2%）。
- **ViT-Small**：
  - 所有变体退化趋势一致 → 因为 **MSA 模块始终 Dense 执行**，且占 ~1/3 MACs，成为共同误差源；
  - MoE 路由策略对初始鲁棒性影响微弱。

#### ✅ 重训练后（Recovery Capability）
| 模型 | 恢复情况 |
|------|---------|
| **ResNet-20** | ✅ **全部恢复**：即使最激进的 L2L 也能通过 5 epoch 完全恢复精度 |
| **VGG11/19_bn** | ⚠️ **仅中等近似可恢复**；KVA/L2L 下不可逆失效（除 VGG11_bn Cluster MoE 外） |
| **ViT-Small** | ❗ **Hard MoE > Dense**：在高近似强度下，Hard MoE 的退化曲线最平缓；r=0.25 时甚至反超 Dense |

#### ✅ Pareto Optimal 结果（精度 vs. 功耗）
- **CNNs**：
  - **Dense + L2J 是绝对赢家**：在所有三个 CNN 模型上都位于 Pareto 前沿；
  - 提供 **29.2% 操作节能**，精度损失 < 0.7 pp；
  - Soft MoE 虽略提升峰值精度，但功耗高出 2–3×，不具性价比。
- **ViT-Small**：
  - **Hard MoE r=0.25 成为新标杆**：
    - 在 L2L 下以相同功耗达到 **75.97% vs. Dense 的 75.00%**；
    - 是本研究中**唯一实现“更高精度 + 更低功耗”双重优势的 MoE 配置**；
  - Soft MoE 与 Cluster MoE 完全被 Pareto 支配。

---

### 🔬 消融实验结果（隐含分析）
虽然未明确标注“ablation”，但以下发现本质上是消融性质的：

| 维度 | 发现 |
|------|------|
| **架构依赖性** | MoE 对近似的响应高度依赖于底层架构（ResNet vs. VGG vs. ViT） |
| **跳跃连接作用** | ResNet 的残差连接不仅增强鲁棒性，也促进近似感知训练收敛 |
| **路由粒度差异** | ViT 中的 **patch-level routing** 使每个专家学习更一致的输入分布，利于误差适应；而 CNN 中 image-level routing 缺乏此优势 |
| **误差概率 ≠ 影响程度** | VGG19-Hard MoE 在 KVA（81.25% 错误率）崩溃至 2%，但在更差的 L2L（93.16%）反而恢复至 57.66% → 表明错误模式比错误频率更重要 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MoE 并不天然适合近似计算**：在未经调优的情况下，MoE 拓扑普遍比 Dense 更脆弱；
2. **恢复能力因架构而异**：
   - ResNet-20 具备强大恢复力，得益于残差结构；
   - VGG 在强近似下难以恢复；
   - ViT-Small 中，**Hard MoE 在高近似区反超 Dense**，展现独特优势；
3. **ViT 中 MoE 与 AxC 存在协同效应**：
   - Patch-level 路由 + 局部专家训练 → 更好适应近似噪声；
   - Hard MoE 在同等功耗下实现了更高的精度；
4. **Dense + 中等近似（如 L2J）仍是 CNN 场景下的最优解**；
5. **Cluster MoE 在所有场景中均被 Pareto 支配**，尽管其设计初衷良好，但 Gateway 开销过大。

### ⚠️ 方法的局限性
- **仅仿真近似乘法器**：未涉及真实硬件部署，缺乏时序、面积等物理约束验证；
- **短周期重训练**：仅 5 epoch，可能低估长期训练潜力（尤其是复杂 MoE）；
- **固定 Gateway 冻结**：未探索联合优化 Gateway 与专家的近似感知训练；
- **未扩展到更大规模 MoE**（如 Switch Transformer）或语言模型。

### 🔮 未来工作方向
- **Hardware-Software Co-design**：基于 AxMoE 发现，定制面向 MoE 的专用近似乘法器；
- **Adaptive Approximation**：根据路由路径动态调整近似强度；
- **End-to-End Approximate Training**：允许 Gateway 参与近似感知更新；
- **扩展至 Large Language Models**：研究 LLM 中 MoE（如 Mixtral）与 AxC 的交互；
- **多层级近似策略**：结合权重量化、激活截断与近似计算形成联合压缩方案。

---

> 💡 **一句话总结**：  
> AxMoE 揭示了 **MoE 与 Approximate Computing 的交互具有强烈架构依赖性**——在 CNN 中 Dense 更稳健，在 ViT 中 Hard MoE 却能在高压缩下逆袭，为未来的高效 AI 系统提供了全新的软硬协同设计视角。

</details>

---

### 6. [CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels](https://arxiv.org/abs/2605.05023)

**Authors**: Xing Ma, Yangjie Zhou, Wu Sun, Zihan Liu, Jingwen Leng, Yun Lin, Shixuan Sun, Minyi Guo, Jin Song Dong  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.05023v1  

#### Abstract
Efficient CUDA implementations of attention mechanisms are critical to modern deep learning systems, yet supporting diverse and evolving attention variants remains challenging. Existing frameworks and compilers trade performance for flexibility, while expert-written kernels achieve high efficiency b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代深度学习系统中，**Attention 机制**是性能的关键瓶颈。随着模型架构演进，出现了大量定制化的 **attention variants**（如 PrefixLM、Sliding Window、ReLU Attention 等），但高效支持这些变体在 GPU 上运行极具挑战。现有方法面临根本性权衡：
- **通用框架**（如 PyTorch）灵活但性能低下；
- **专家手工优化内核**（如 FlashAttention）高效但难以扩展；
- **编译器方法**（如 FlexAttention）受限于模板，无法处理复杂语义；
- **LLM 生成 CUDA 内核**的方法在正确性和性能上不稳定，尤其对复杂操作（如 attention）差距显著。

### **提出的新方法与思路**
作者提出 **CuBridge**，一种基于 **LLM 的框架**，通过 **lift-transfer-lower 工作流** 来适配专家级 attention 内核，而非从零生成代码。其核心思想是：**以专家编写的高性能 CUDA 内核为参考，保留其执行结构（execution orchestration），仅进行可控的语义迁移**。

#### **三大创新点**：
1. **提出 CuIR（CUDA Intermediate Representation）**  
   - 一个可执行的、Pythonic 的中间表示，显式暴露 **execution orchestration**（如内存层次、计算流水线、同步依赖），同时抽象掉低层 CUDA 语法细节。
   - 支持验证：CuIR 可被模拟执行，确保语义正确性。

2. **设计 lift-transfer-lower 工作流**  
   - **Lift**：将源 CUDA 内核提升为 CuIR，提取高层执行逻辑；
   - **Transfer**：在 CuIR 层面根据目标 PyTorch 语义生成目标 CuIR，进行语义迁移；
   - **Lower**：通过差分分析，生成最小补丁，重构出目标 CUDA 内核。

3. **实现可靠且高效的语义迁移**  
   - 避免直接修改复杂 CUDA 代码带来的错误；
   - 利用 LLM 理解 CuIR 并进行结构化推理，保证正确性；
   - 保留专家级性能结构（如 warp-specialized pipeline、TMA、WGMMMA）。

### **相比现有方法的优势**
| 方法 | 缺陷 | CuBridge 的优势 |
|------|------|----------------|
| PyTorch | 多 kernel 启动、频繁 global memory 访问 | 单一融合 kernel，避免冗余开销 |
| FlexAttention | 固定模板，不支持非 Softmax 归一化等 | 支持更广变体（如 ReLU、Sigmoid Attention） |
| Qimeng-Attention（LLM 生成） | 正确性不稳定，性能差距大（up to 34.9×） | 100% 正确，性能显著提升 |
| 手工修改专家内核 | 耗时、易错 | 自动化、可靠、保留专家结构 |

---

## **2. 核心实验方法和设置**

### **实验设置**
- **硬件平台**：NVIDIA A100 和 H100 GPU
- **LLM 后端**：GPT-5、Claude-3.5-Sonnet、DeepSeek-V3、Qwen-3-235B、Qwen-3-32B
- **测试 attention variants**（共 8 种）：
  - **Masking**：PrefixLM、Global Sliding Window、Share Question Mask、Causal Blockwise Mask
  - **Score**：Relative Position、Softcap
  - **Normalization**：ReLU Attention、Sigmoid Attention
  - **Composite**：PrefixLM + Softcap + Sigmoid
- **模型配置**：基于真实 LLM 架构
  - Llama2-7B (MHA, 32/32/128)
  - Qwen2.5-72B (GQA, 64/8/128)
  - Llama3.1-405B (GQA, 128/8/128)
- **序列长度**：1k, 2k, 4k, 8k，batch size 动态调整以保持总 token 数恒定（16k）

### **评估指标**
- **性能**：TFLOPS（每秒万亿浮点运算）
- **正确性**：Pass@k（k 次采样中至少一次通过数值验证）
- **速度提升**：相对于基线的平均加速比

### **基线方法对比**
1. **PyTorch**：标准算子组合实现
2. **FlexAttention**：模板化编译器，支持有限自定义
3. **Qimeng-Attention**：基于 LLM 的 attention 内核生成方法
4. **FlashInfer**：专家手工调优的高性能推理库（用于对比专家级性能）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 在 A100 和 H100 上，**CuBridge 实现了 100% 的正确率**，所有生成内核均通过数值验证。
- **平均加速比**（相对基线）：
  - **PyTorch**：**16.03×**
  - **FlexAttention**：**1.39×**
  - **Qimeng-Attention**：**3.33×**
- 在 H100 上，性能优势进一步扩大，表明 CuBridge 更好地利用了新硬件特性（如 TMA、WGMMMA）。

### **与基线方法的对比结果**
| 基线 | 性能表现 | CuBridge 优势 |
|------|----------|---------------|
| **PyTorch** | 显著性能下降，部分任务 OOM | 完全避免 kernel launch 和 global memory 冗余 |
| **FlexAttention** | 在简单变体上接近，但在复杂变体（如 Share Question Mask）上落后 | 支持更广变体，性能更高（1.05–1.66×） |
| **Qimeng-Attention** | 在简单变体上尚可，但在复合变体上严重退化（如 comb variant 上慢 11.47×） | 保持稳定高性能，尤其在复杂 masking 和 irregular pattern 下优势明显 |
| **FlashInfer** | 在原生支持变体上性能相当（平均 1.07×） | 在非原生变体上平均 **3.49×** 加速 |

### **消融实验结果**
在 H100 上对 96 个测试用例进行消融研究，比较三种方法：
| 方法 | Pass@1 | Pass@5 | 相对速度提升（几何平均） |
|------|--------|--------|--------------------------|
| **Vanilla GPT-5** | 0.21 | 0.38 | 1.00× |
| **GPT-5 + ReAct** | 0.41 | 0.58 | 1.23× |
| **CuBridge** | **0.70** | **1.00** | **4.19×** |

**结论**：CuIR 和 lift-transfer-lower 流程显著提升了正确性和性能稳定性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **专家内核是宝贵的性能资产**：不应从头生成，而应作为参考进行语义迁移。
2. **中间表示（CuIR）至关重要**：它使 execution orchestration 显式化，支持 LLM 理解和验证。
3. **lift-transfer-lower 流程有效**：分离语义推理与代码生成，避免了 LLM 直接操作 CUDA 的不稳定性。
4. **性能优势来自结构保留**：CuBridge 成功继承了专家级的硬件优化（如异步流水线、warp specialization）。
5. **方法具有泛化性**：在不同 LLM 后端（GPT-5、Claude、DeepSeek）上性能稳定（±5%），说明优势主要来自流程设计而非特定模型。

### **局限性**
1. **依赖高质量专家内核作为源**：若目标硬件（如 FPGA）缺乏优化内核，则无法启动流程。
2. **当前评估集中在 attention 变体**：尚未扩展到其他 HPC 场景（如科学计算、稀疏矩阵运算）。
3. **LLM 容量门槛**：小模型（如 Qwen-3-32B）无法生成有效内核，需足够强的 CUDA 推理能力。

### **未来工作方向**
- 将 CuBridge 范式推广至其他高性能算子（如 GEMM、convolution）；
- 支持更多硬件平台（如 AMD GPU、FPGA）；
- 结合训练-based 方法（如 RLHF）进一步提升 LLM 的 CUDA 生成能力；
- 探索自动源内核选择机制，适应不同硬件和 workload。

---

> **总结**：CuBridge 提出了一种全新的 **LLM-assisted kernel adaptation** 范式，通过 **CuIR + lift-transfer-lower** 流程，在保证 100% 正确性的前提下，实现了远超现有框架的性能，为高效支持不断演进的 attention 机制提供了可靠解决方案。

</details>

---

### 7. [CCL-D: A High-Precision Diagnostic System for Slow and Hang Anomalies in Large-Scale Model Training](https://arxiv.org/abs/2605.04478)

**Authors**: Yida Gu, Fakang Wang, Jianhao Fu, Zhenhang Sun, Qianyu Zhang, Hairui Zhao, Xingchen Liu, Yang Tian, Wenjing Huang, Zedong Liu, Yifan Chen, Jinwu Yang, Yueyuan Zhou, Qian Zhao, Haoxu Li, Tao Wang, Feng Yu, Zhan Wang, Guangming Tan, Dingwen Tao  
**Category**: cs.DC  
**Published**: 2026-05-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.04478v1  

#### Abstract
As training scales grow, collective communication libraries (CCL) increasingly face anomalies arising from complex interactions among hardware, software, and environmental factors. These anomalies typically manifest as slow/hang communication, the most frequent and time-consuming category to diagnos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CCL-D: A High-Precision Diagnostic System for Slow and Hang Anomalies in Large-Scale Model Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模分布式模型训练中，**Collective Communication Library (CCL)** 面临由硬件、软件和环境因素复杂交互引发的 **slow/hang anomalies**（通信缓慢或挂起异常）。这类问题是训练中断中最频繁且诊断耗时最长的一类，传统方法（如日志分析、堆栈追踪）往往需要数小时甚至数天才能定位根因，严重影响训练效率。

### 提出的新方法与新思路
作者提出 **CCL-D**，一个高精度的实时诊断系统，用于快速检测并精确定位大规模模型训练中的 slow/hang 异常。其核心思想是：
- 在 CCL 层面引入**轻量级分布式追踪框架**，实现对通信流量的细粒度监控；
- 设计基于 **Send/Recv 原语** 的跨层（host 和 GPU kernel）诊断指标；
- 构建一个**集中式但可扩展的决策分析模块**，结合多维度指标进行自动化异常检测与根因定位。

### 相比现有方法的优势
| 维度 | CCL-D | 现有方法（如 Bisection、Stack Analysis、NCCL RAS 等） |
|------|-------|--------------------------------------------------|
| **诊断精度** | 支持所有六类 slow/hang 异常（H1–H3, S1–S3），支持混合型 slow（Mixed-Slow） | 多数仅支持部分类型，无法处理混合场景 |
| **诊断效率** | 检测延迟 ≤5分钟（hang），≤1分钟（slow）；定位时间 <150ms | 手动触发或依赖超时机制（如 PyTorch Watchdog 30分钟），定位需数小时 |
| **运行开销** | 运行时开销 <1%，内存占用稳定（每 rank 1184 Bytes） | 高资源消耗，影响训练性能 |
| **部署模式** | 完全在线（online），不影响训练流程 | 多为离线或半自动，需暂停任务 |

---

## 2. 核心实验方法和设置

### 使用的数据集与模型
- **训练任务**：BaiLing-5B, Llama2-7B, Llama3.1-8B, BaiLing-80B
- **数据集**：Alpaca, Fineweb-edu
- **训练策略**：FSDP（Fully Sharded Data Parallel）、3D 并行
- **通信协议**：Ring / Tree 算法，Simple / LL / LL128 协议
- **平台兼容性验证**：同时测试 NCCL（CUDA）和 RCCL（ROCm）后端

### 实验设置
- **硬件平台**：4,000-GPU 集群，单节点配置为 8×Nvidia H20 GPU（96GB HBM3），通过 900GB/s NVLink 和 400G NIC 互联
- **软件环境**：CUDA 12.2, NCCL 2.24.3, PyTorch 2.4.0, Megatron 0.9.0
- **功能验证规模**：16 GPU；可扩展性测试达 4,000 GPU

### 评估指标
| 类别 | 指标 |
|------|------|
| **诊断准确性** | 覆盖的异常类型数量、根因 rank 定位准确率 |
| **诊断效率** | 异常检测延迟（Detection Latency）、根因定位延迟（Location Latency） |
| **系统开销** | CPU 利用率、内存占用、通信操作额外耗时、训练步长时间变化 |
| **训练影响** | Loss 曲线一致性、收敛速度、吞吐量下降程度 |

### 基线方法对比
选取五类代表性方法作为 baseline：
1. **Bisection-based**（如 DL-Rover + NCCL-tests）：基于压力测试的二分法定位
2. **Stack Analysis**（自研实现，参考 XPUTimer 和 ParaStack）
3. **NCCL RAS**：NVIDIA 提供的运行时状态监控工具
4. **Greyhound**：专注于 fail-slow 检测的系统
5. **C4D**：阿里提出的实时异常检测方案

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | CCL-D 表现 |
|------|-----------|
| **平均诊断总时间** | **6 分钟内完成从检测到定位**（hang: 5min detection + 1min location） |
| **定位延迟** | **<150ms**（NCCL/R-CCL 下均表现一致） |
| **运行时开销** | <1%（最大训练步长增加仅 0.95%） |
| **内存占用** | 固定每 rank **1184 Bytes**，不随集群规模增长 |
| **CPU 开销** | 每节点约 **0.3% CPU 使用率**，高度稳定 |
| **异常覆盖率** | 实现生产环境中 **近100% 已知 slow/hang 异常覆盖**

### 与基线方法的对比结果（见 Table 1 & Table 2）

#### ✅ 诊断能力全面领先
| 方法 | Not-Entered | Inconsistent | Hardware Fault | Comp-Slow | Comm-Slow | Mixed-Slow |
|------|-------------|--------------|----------------|-----------|-----------|------------|
| Bisection | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Stack | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| RAS | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Greyhound | ❌ | ❌ | ❌ | ⚠️（仅频率 throttling） | ✅ | ❌ |
| C4D | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **CCL-D** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> ✔️ CCL-D 是唯一能完整覆盖六类异常（H1–H3, S1–S3）并支持 Mixed-Slow 场景的方法。

#### ⏱️ 效率显著提升
| 方法 | Hang Detection | Slow Detection | Location Time |
|------|----------------|----------------|---------------|
| Bisection | >30min | ~1h | >4min |
| Stack | >30min | N/A | ~5min |
| RAS | >30min | N/A | 10ms |
| Greyhound | N/A | 1min | 1.43s |
| C4D | 5min | 1min | ~100ms |
| **CCL-D** | **5min** | **1min** | **~146ms** |

> 🔥 CCL-D 将原本以“小时”计的诊断过程缩短至“分钟级”，且定位精度更高。

### 消融实验与开销分析（Section 6.3）
- **Tracing Framework 开销极低**：
  - 相比 naive centralized 方案，CCL-D 的通信标识延迟降低 **188×**
  - 内存复用设计使 per-rank footprint 稳定在 1.1KB
- **通信操作影响微弱**：
  - AllReduce/AllGather/ReduceScatter/AlltoAll 操作的额外开销 <0.45%
- **训练性能无损**：
  - Loss 曲线与原始训练完全一致（图13 e–h）
  - 最大步长时间增幅仅为 0.95%（BaiLing-80B, 3D 并行下）

---

## 4. 关键结论和发现

### 主要发现
1. **Slow/Hang 异常具有结构性根源**：通过对两年生产数据的分析，作者首次归纳出 **六类根本原因**（Not-Entered-Hang, Inconsistent-Hang, Hardware-Fault, Comp-Slow, Comm-Slow, Mixed-Slow），为系统化诊断提供理论基础。
2. **Send/Recv 原语是最有效的诊断信号**：相比高层调用计数或堆栈信息，底层 Send/Recv 的 **count 与 rate 变化** 能更早、更准地暴露异常行为。
3. **主机侧测量可实现高效诊断**：利用 CUDA UVA 实现 zero-copy 共享内存，将 metric 测量主体移至 CPU，避免干扰 GPU 计算，兼顾精度与低开销。
4. **决策算法可形式化建模**：通过定义 `P=(Tmax−Tmin)/(Tmax−Tbase)` 指标，可量化区分 computation vs communication slowness，并设定边界参数 α/β 实现自动分类。

### 方法的局限性
- **依赖 CCL 可插桩性**：需要修改 CCL 源码注入 probing 逻辑，在闭源或受限环境中可能难以部署。
- **未解决修复闭环**：当前系统聚焦于“诊断”而非“自愈”，仍需人工介入重启或更换设备。
- **极端网络抖动可能导致误报**：虽然采用重复计数机制缓解，但在持续波动环境下仍存在漏检风险。

### 未来工作方向
- **集成自适应阈值调整机制**：动态学习不同训练阶段的正常通信模式，减少误报。
- **构建闭环自愈系统**：结合调度器实现故障 rank 自动隔离与替换。
- **扩展至更多 CCL 后端**：支持 MPI、oneCCL 等其他通信库。
- **与网络层诊断系统联动**：与 ComScribe、NetBouncer 等协同，实现跨层级联合诊断。

---

> 📌 **总结一句话**：  
> CCL-D 是首个能够在 **分钟级** 内实现对大规模模型训练中各类 **slow/hang 异常** 进行 **高精度、全覆盖、低开销** 诊断的系统，显著提升了超大规模训练的可观测性与可靠性。

</details>

---

### 8. [GraphPI: Efficient Protein Inference with Graph Neural Networks](https://arxiv.org/abs/2605.04376)

**Authors**: Zheng Ma, Jiazhen Chen, Lei Xin, Ali Ghodsi  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.04376v1  

#### Abstract
The integration of deep learning approaches in biomedical research has been transformative, enabling breakthroughs in various applications. Despite these strides, its application in protein inference is impeded by the scarcity of extensively labeled datasets, a challenge compounded by the high costs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《GraphPI: Efficient Protein Inference with Graph Neural Networks》核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
蛋白质推断（Protein Inference）是质谱（MS/MS）数据分析中的关键步骤，旨在从检测到的肽段（peptides）反推出样本中存在的蛋白质。该任务面临以下挑战：
- **标签稀缺**：缺乏大规模、高质量的标注蛋白数据集，限制了深度学习模型的应用。
- **共享肽段（Shared Peptides）**：多个蛋白质可能产生相同的肽段，导致推断歧义。
- **单肽蛋白（One-hit Wonders）**：仅由一个肽段支持的蛋白质难以置信地确认其存在。
- **计算效率低**：现有基于贝叶斯网络的方法（如 Fido、Epifany）计算开销大，难以扩展到大型数据集。

### 提出的新方法和新思路
本文提出了 **GraphPI**，一种基于图神经网络（GNN）的新型蛋白质推断框架，其核心创新点如下：

- **将蛋白质推断建模为节点分类问题**：构建了一个**蛋白质-肽段-PSM 三部图（tripartite graph）**，其中：
  - 节点（Nodes）代表蛋白质、肽段和肽谱匹配（PSM）。
  - 边（Edges）表示它们之间的关系（蛋白质包含肽段，肽段对应PSM）。
  - 推断过程转化为对“蛋白质”节点进行二分类（存在/不存在），利用GNN聚合来自连接肽段和PSM的信息。

- **半监督自训练（Semi-supervised Self-training）策略**：
  - 使用无标签的公共蛋白数据集进行训练。
  - 初始伪标签（pseudo-labels）由现有算法（Epifany）生成，并通过阈值（5% FDR）确定。
  - 引入**硬负例（hard negative）**：明确将已知不存在的 decoy 蛋白标记为负样本，增强模型判别能力。
  - 采用**迭代自训练**：用当前模型预测的高置信度标签更新训练集，重新训练，逐步提升性能。

- **通用性和免微调（Universal Applicability）**：
  - 由于 Percolator 提取的 PSM 特征具有良好的归一化特性，GraphPI 在一个大型公共数据集上预训练后，可直接应用于各种不同的测试数据集，无需针对每个数据集进行微调（fine-tuning）。
  - 这不仅避免了过拟合风险，还极大地提升了计算效率。

### 相比现有方法的优势
- **性能优越**：在多个基准数据集上，性能达到甚至超越了最先进的贝叶斯方法（如 Epifany）。
- **计算高效**：得益于神经网络的并行化特性，推理速度远超传统方法（例如，在酵母数据集上比 Epifany 快约10倍）。
- **可扩展性强**：线性的时间复杂度使其能够处理大规模蛋白组学数据。
- **数据驱动**：相比依赖强先验假设的贝叶斯方法，GraphPI 能从数据分布中自动学习更复杂的模式。

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 训练数据集
- 从 **ProteomeXchange** 下载了17个公开的人类蛋白组学数据集（如 `PXD004789`, `PXD005388` 等）。
- 数据经过 **Comet** 搜索引擎和 **Percolator** 重排序处理，以获得高质量的 PSM 特征。

#### 测试数据集
- **iPRG2016**：专门设计用于评估共享肽段场景下的蛋白质推断算法。
- **UPS2**：48种已知浓度的人类蛋白混合物，动态范围广。
- **18Mix**：18种不同来源的蛋白混合物。
- **Yeast**：酿酒酵母全蛋白组数据。
- **Hela-3T3**：人源 HeLa 细胞和鼠源 3T3 细胞的混合物，用于双物种 FDR 评估。

### 实验设置和评估指标
- **模型架构**：基于 **GraphSAGE** 设计的异构GNN，能区分不同类型的节点（蛋白质、肽段、PSM）和边。
- **特征工程**：
  - **PSM 节点**：直接使用 Percolator 提供的16个特征（如 `XCorr`, `deltCn`, `ionFrac` 等，见 Table S1）。
  - **蛋白质/肽段节点**：使用可学习的嵌入（learnable embedding）来表示节点类型。
  - **边权重/属性**：
    - (肽段, PSM) 边：使用权重为 PSM 的识别得分。
    - (蛋白质, 肽段) 边：引入“肽段共享”特征（peptide-sharing feature），对非唯一证据的肽段进行惩罚。
- **训练流程**：共进行 **10轮自训练**，最终得分是所有轮次模型输出的平均值。
- **硬件**：在单张 NVIDIA RTX 4090 上训练。

### 评估指标
- **主指标**：**ROC曲线**（以真实阳性蛋白数 vs. entrapment FDR）和 **pAUC**（1%-5% FDR区间的部分曲线下面积）。
- **计算效率**：在酵母数据集上测量运行时间（分钟）。

### 基线方法对比
与四种代表性方法进行了比较：
- **Epifany**：基于贝叶斯网络，性能优异但计算慢。
- **Fido**：早期的贝叶斯网络方法。
- **PIA**：基于规则的简约性（parsimony）算法。
- **DeepPep**：现有的深度学习方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **总体性能**（见 Figure 4 & 5）：
  - 在 **iPRG2016 A/B** 数据集上，GraphPI 是唯一能与 Epifany 竞争的方法，且在低FDR区域表现更优。
  - 在 **iPRG2016 AB** 上，性能仅次于 PIA，排名第二。
  - 在 **Yeast**, **18Mix**, **Hela**, 和 **3T3** 数据集上，GraphPI 取得了**最佳性能**。
  - 在 **UPS2** 上，性能与 Epifany 相当，优于其他方法。
  - **pAUC** 结果进一步证实了 GraphPI 在大多数数据集上的领先地位。

- **计算效率**（见 Figure 6a）：
  - 在最大的 **Yeast** 数据集上：
    - **GraphPI**: **88秒** (~1.5分钟)
    - **Epifany**: **超过14分钟**
  - GraphPI 的推理时间随数据集大小呈**线性增长**（见 Figure 6b），而贝叶斯方法通常有更高的复杂度。

### 消融实验结果
- **目标-诱饵训练（Target-Decoy Training）**（见 Figure 7）：
  - 尝试了类似 Barista 的训练方式（将 decoy 当作负样本，其余为正样本）。
  - 结果显示，这种方法在 **iPRG2016 A/B**（富含共享肽段）等复杂数据集上表现不佳，证明了 GraphPI 所采用的基于 Epifany 伪标签的策略更能有效处理共享肽段带来的歧义。

- **去噪机制分析**（见 Table 2）：
  - 移除 decoy 蛋白作为硬负例后，模型性能显著下降（在 iPRG2016 B 上 TP 从187降至176）。
  - 如果错误地将一些在5% FDR内的 decoy 标记为正样本，性能也会受损（TP降至183）。
  - 这表明，显式地引入 decoy 作为负样本是提升模型鲁棒性的关键。

- **泛化能力验证**（见 Figure S5 & S6）：
  - 在非人类（如 *E. coli*, *D. melanogaster*）数据集上，GraphPI 依然表现出色，pAUC 显著高于 Epifany。
  - 使用非isoform数据库预训练的模型性能略有下降，但差异不大，说明模型对数据库选择相对鲁棒。

---

## 4. 关键结论和发现

### 主要发现
1. **GNN 非常适合蛋白质推断任务**：通过构建蛋白质-肽段-PSM 图，GNN 能自然地建模三者间复杂的依赖关系，并实现端到端的学习。
2. **半监督自训练是解决标签稀缺的有效途径**：利用大规模无标签数据和弱监督信号（伪标签 + decoy 负例），可以训练出高性能的深度学习模型。
3. **通用模型是可行的**：得益于 Percolator 特征的良好归一化，一个在多种数据集上训练的单一 GraphPI 模型可以通用地应用于新数据，无需微调。
4. **计算效率的巨大优势**：神经网络的并行化特性使其在速度上远超传统的贝叶斯方法，为实时分析和大规模研究铺平了道路。

### 方法的局限性
- **依赖基础模型的质量**：初始伪标签由 Epifany 生成，如果基础模型存在系统性偏差，可能会被继承。
- **对极端情况的处理**：虽然在共享肽段上表现良好，但对于极其复杂的同源蛋白家族或高度相似的蛋白变体，仍可能存在挑战。
- **解释性**：相较于贝叶斯网络，深度学习模型的决策过程是一个“黑箱”，可解释性较差。

### 未来工作方向
- **整合更多生物学信息**：在未来的工作中，可以探索整合蛋白质-蛋白质相互作用（PPI）、基因本体（GO）注释或翻译后修饰（PTM）等信息，以进一步提升推断准确性。
- **改进图结构**：探索更复杂的图构建方式，例如引入蛋白质簇或考虑肽段的独特性。
- **多任务学习**：同时优化肽段鉴定和蛋白质推断，实现真正的端到端联合优化。

</details>

---

### 9. [A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](https://arxiv.org/abs/2605.04595)

**Authors**: Chengyi Nie, Nian Si, Zijie Zhou  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.04595v1  

#### Abstract
The rapid adoption of large language models (LLMs) has created significant challenges for efficient inference at scale. Unlike traditional workloads, LLM inference is constrained by both computation and the memory overhead of key-value (KV) caching, which accelerates decoding but quickly exhausts GP...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文针对**大规模 LLM 推理服务中的稳定性问题**，特别是在 **GPU 资源受限环境下，同时受计算（computation）和内存（KV Cache 内存占用）双重约束**的挑战。传统排队模型通常只考虑计算瓶颈，而忽略了 KV Cache 所带来的动态内存增长对系统稳定性的显著影响。

具体而言，当请求到达率超过系统的有效处理能力时，队列将无界增长，导致延迟飙升、服务质量下降。因此，如何准确建模并预测 LLM 推理系统的**稳定边界（stability condition）** 是资源规划和调度设计的关键。

### 🚀 提出的新方法与新思路
作者提出了**首个显式纳入 KV Cache 内存约束的排队论框架（queueing-theoretic framework）**，其核心创新包括：

- **联合建模计算与内存约束**：将每个请求在 prompt 阶段和 decode 阶段的 KV Cache 占用动态建模为随时间线性增长的内存足迹，并引入“lifetime cumulative memory usage”函数 $ g(s, o) $ 来量化单个请求在整个生命周期内的总内存消耗。
  
- **定义新的服务速率 $ \mu $**：提出一个闭式表达式来估计最大可持续处理速率：
  $$
  \mu = \frac{M}{b \cdot \mathbb{E}[g(s, o)]}
  $$
  其中 $ M $ 是 GPU 可用 KV Cache 容量（以 token 为单位），$ b $ 是平均 batch 处理时间，$ \mathbb{E}[g(s,o)] $ 是基于输入/输出长度分布的期望内存消耗。

- **严格的稳定性理论分析**：
  - **定理 4.1**：若到达率 $ \lambda > \mu $，则系统必然不稳定（overloaded）；
  - **定理 4.2**：若 $ \lambda < \mu(1 - \delta) $，其中 $ \delta = \text{ess sup}_{s,o} (s+o)/M $ 表示最大单请求内存占比，则在 work-conserving 调度策略下系统是稳定的。

该框架支持灵活的调度策略（如 FCFS、SJF），并适用于连续批处理（continuous batching）和 chunked prefill 架构。

### 🔍 相比现有方法的优势
| 方面 | 本文方法 | 现有工作（如 Li et al., 2025; Yang et al., 2024） |
|------|--------|---------------------------------------------|
| **是否建模 KV Cache 内存** | ✅ 显式建模 | ❌ 忽略或简化 |
| **是否区分 prompt 和 decode 阶段内存变化** | ✅ 分阶段建模 | ❌ 统一假设 |
| **能否给出闭式稳定性条件** | ✅ 有理论保证 | ❌ 多为启发式或仿真 |
| **是否适用于真实部署场景** | ✅ 在真实 GPU 上验证 | ⚠️ 多停留在理论或模拟层面 |

> 💡 总结：这是**第一项将 KV Cache 内存作为核心约束纳入排队模型的工作**，填补了理论与实践之间的鸿沟，为 LLM serving 的容量规划提供了可量化的科学依据。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集与负载配置
实验未完全依赖公开数据集，而是通过控制不同的 **Prefill-Decode (P/D) 比例**构造多种合成负载，以及使用真实复杂任务数据集进行验证：

- **合成负载（Synthetic Workloads）**：
  - **P/D Ratio 1:1**：prefill 和 decode 长度均从 Uniform(10, 1600) 中独立采样
  - **P/D Ratio 2:1**：prefill ~ Uniform(10, 2133)，decode ~ Uniform(10, 1066)
  - **P/D Ratio 1:2**：prefill ~ Uniform(10, 1066)，decode ~ Uniform(10, 2133)
  - **Mixed 负载**：前半部分为 2:1，后半为 1:2，测试非平稳负载下的鲁棒性

- **真实数据集**：
  - **LongBench v2**：包含 503 个长上下文问答任务，具有高度可变且相关的 prefill/decode 长度，用于评估现实世界复杂性。

### ⚙️ 实验设置
- **硬件平台**：8 × NVIDIA A100 GPUs（每卡单独运行一个 replica）
- **模型**：Meta-Llama-3-8B
- **推理引擎**：vLLM v1，启用 **chunked prefill**（chunk size = 512 tokens）
- **并行方式**：仅使用 data parallelism（无 tensor/pipeline 并行）
- **内存容量参数 $ M $**：设为 131,000 tokens（基于实测最大可用 KV Cache）
- **batch 处理时间 $ b $**：采用 **median batch execution time** 作为稳健估计器（在重尾情况下改用 trimmed mean）

### 📈 评估指标
- **理论处理速率 $ \mu_{\text{theory}} $**：由公式推导得出
- **实测处理速率 $ \mu_{\text{gpu}} $**：排除 warm-up 和 termination 阶段后的实际吞吐（requests/sec）
- **Gap Absolute Percentage (GAP)**：
  $$
  \text{GAP} = \left| \frac{\mu_{\text{theory}} - \mu_{\text{gpu}}}{\mu_{\text{gpu}}} \right| \times 100\%
  $$
- **队列长度演化图**：观察不同 $ \lambda $ 下系统是否稳定
- **等待时间 CDF**：反映延迟表现

### 🔁 基线方法对比
本文**不直接与其他调度算法比较性能**，而是聚焦于验证所提出的**理论稳定性边界是否准确预测真实系统行为**。因此，“基线”是真实 GPU 测量值本身，目标是证明 $ \mu_{\text{theory}} $ 能高精度逼近 $ \mu_{\text{gpu}} $。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ 单 GPU 实验结果（Table 1）

| P/D Ratio | $ \mu_{\text{gpu}} $ | $ \mu_{\text{theory}} $ | GAP |
|-----------|------------------------|----------------------------|-----|
| 1:1       | 3.387                  | 3.263                      | 3.66% |
| 2:1       | 3.650                  | 3.956                      | 8.38% |
| 1:2       | 2.969                  | 2.902                      | 2.25% |
| Mixed     | 3.137                  | 3.385                      | 7.90% |

> ✔️ 所有 GAP 均 **< 10%**，表明理论预测高度准确。

#### ✅ LongBench v2 数据集结果（Table 2）

| $ \mu_{\text{gpu}} $ | $ \mu_{\text{theory}} $ | GAP |
|-----------------------|----------------------------|-----|
| 0.610                 | 0.561                      | 8.03% |

> ✔️ 在真实复杂负载下仍保持良好一致性，说明建模 joint distribution $ p(s,o) $ 至关重要。

#### ✅ 8-GPU 集群实验结果（Table 3）

| P/D Ratio | $ \mu_{\text{gpu}} $ | $ 8 \times \mu_{\text{theory}} $ | GAP |
|----------|------------------------|------------------------------------|-----|
| 1:1      | 26.710                 | 25.808                             | 3.38% |

> ✔️ 扩展到多 GPU 场景依然准确，验证了模型的可扩展性和实用性。

#### ✅ 极端 P/D 比例（8:1）下的鲁棒性（Appendix C）
- 使用 **10% trimmed mean** 估算 $ b $ 后，GAP 降至 **7.2%**
- 表明即使在 batch processing time 出现重尾分布时，结合稳健统计方法仍能获得可靠预测

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KV Cache 内存是决定 LLM 推理稳定性的关键因素**，不能仅靠计算建模。
2. 所提出的排队模型能够**精确预测系统稳定边界**，理论与实测处理速率误差普遍 **低于 10%**。
3. 在 **single-GPU 到 multi-GPU**、**合成到真实负载**、**平稳到非平稳请求流** 等多种场景下均表现出强健的泛化能力。
4. 可用于指导 **GPU 集群规模预估**：给定请求到达率 $ \lambda $ 和理论服务率 $ \mu $，所需最小 GPU 数约为 $ \lceil \lambda / (\rho \cdot \mu) \rceil $，其中 $ \rho $ 为目标利用率（如 90%）。

### ⚠️ 方法的局限性
1. 当前模型假设 **chunked prefill 固定大小**，未涵盖更复杂的 adaptive chunking 策略。
2. 尚未支持 **tensor parallelism** 或 **pipeline parallelism**，这些架构会改变排队拓扑结构。
3. 对 **极端重尾 workload**（如 P/D=8:1）需依赖 trimmed mean 等经验调整，缺乏统一自适应机制。
4. 假设调度策略为 work-conserving，未深入比较不同调度策略（如 SJF vs FCFS）对 $ \mu $ 的影响。

### 🔮 未来工作方向
1. 将模型扩展至 **tandem queues** 结构，以支持 prefill-decode disaggregation 和 pipeline parallelism。
2. 引入 **fluid limit 或 mean-field 近似**，处理更大规模集群的稳定性分析。
3. 结合在线学习技术，实现对动态变化的 $ \lambda $ 和 $ F_{in-out} $ 的自适应估计与扩容决策。
4. 探索如何将该框架集成进生产级 LLM serving platform（如 vLLM、Triton）中，实现自动化弹性伸缩。

---

## 总结
> 本文建立了首个融合 **computation 与 KV Cache memory 约束** 的 LLM 推理排队模型，提出了具有严格理论保证的稳定性判据，并在真实 GPU 环境中验证了其高达 **90%+ 的预测准确性**。这一成果为 LLM 服务系统的**科学容量规划、避免过度采购或资源不足**提供了强有力的工具，推动 AI Infra 向更高效、可持续的方向发展。

</details>

---

### 10. [SOAR: Real-Time Joint Optimization of Order Allocation and Robot Scheduling in Robotic Mobile Fulfillment Systems](https://arxiv.org/abs/2605.03842)

**Authors**: Yibang Tang, Yifan Yang, Jingyuan Wang, Junhua Chen, Zhen Zhao  
**Category**: cs.AI  
**Published**: 2026-05-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.03842v1  

#### Abstract
Robotic Mobile Fulfillment Systems (RMFS) rely on mobile robots for automated inventory transportation, coordinating order allocation and robot scheduling to enhance warehousing efficiency. However, optimizing RMFS is challenging due to strict real-time constraints and the strong coupling of multi-p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SOAR: Real-Time Joint Optimization of Order Allocation and Robot Scheduling in Robotic Mobile Fulfillment Systems  
**——核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Robotic Mobile Fulfillment Systems (RMFS)** 中，订单分配（Order Allocation）与机器人调度（Robot Scheduling）是两个强耦合、多阶段的决策过程。传统方法面临以下挑战：
- **Decoupled 方法**（如先分配再调度）牺牲全局最优性以换取实时响应；
- **Global Optimization 方法**（如 MIP 模型）计算开销大，难以适应动态环境。

因此，如何在满足 **real-time 响应要求**的同时实现 **multi-phase 联合优化** 是一个尚未有效解决的关键问题。

---

### 🚀 提出的新方法与创新思路

作者提出 **SOAR**（Soft Order Allocation and Robot Scheduling），一种基于 **Deep Reinforcement Learning (DRL)** 的统一框架，用于实现实时联合优化。

#### 主要创新点包括：

| 创新点 | 描述 |
|--------|------|
| **Soft Order Allocation 机制** | 将订单分配从“硬决策”转化为“软匹配”，通过计算订单-货架-工作站之间的 `matching degree`，生成热力向量（Heat Vectors）作为状态输入，延迟最终决策至调度阶段，从而打破两阶段壁垒。 |
| **Event-Driven Markov Decision Process (ED-MDP)** | 决策由异步事件触发（如订单到达、机器人空闲等），使系统能实时响应动态变化，避免固定时间步长带来的延迟或资源浪费。 |
| **Heterogeneous Graph Transformer (HGT) 编码器** | 将仓库中不同类型的实体（机器人、货架、工作站、存储位）建模为异构图，利用 HGT 提取复杂关系特征，融合领域知识提升表示能力。 |
| **Phase-Knowledge-Guided Decoder** | 在动作解码过程中引入 **phase-specific bias**（如 Pick-up 优先高热度货架，Return 最小化距离），将先验知识注入模型，加速收敛并提升策略质量。 |
| **p-norm Reward Shaping** | 针对 makespan 目标稀疏反馈问题，设计基于 p-norm 的密集奖励函数，缓解信用分配难题，稳定训练过程。 |

---

### 🔍 相比现有方法的优势

| 维度 | SOAR 的优势 |
|------|-------------|
| **全局性能** | 实现订单分配与机器人调度的端到端联合优化，避免局部次优。 |
| **实时性** | 单次决策延迟 < 100ms，适用于工业级高并发场景。 |
| **可扩展性** | 支持大规模仓库环境（数百机器人）、动态订单流。 |
| **部署可行性** | 成功完成 sim-to-real 部署，在真实生产环境中验证有效性。 |

---

## 2. 核心实验方法和设置

### 📊 数据集

实验涵盖两类数据集，共 **6 个子集**（Small/Medium/Large × Real/Synthetic）：

| 类型 | 规模 | 来源 | 特征 |
|------|------|------|------|
| **Real-World Dataset** | 40×72 网格，861 shelves, 16 workstations, 198 robots | 合作企业 **Geekplus** 提供的真实仓库运行数据（31天历史订单） | 包含真实布局、库存快照、非均匀波次订单到达模式 |
| **Synthetic Dataset** | 100×80 网格，1600 shelves, 23 workstations | 模拟生成 | 订单到达服从带扰动的 wave 分布；订单行数与数量服从截断 Pareto 分布（长尾特性） |

> ⚙️ 数据划分：真实数据按时间划分为 train / val / test（21/5/5 天）

---

### 🧪 实验设置与评估指标

#### 评估指标（Metrics）

| 指标 | 公式/定义 | 说明 |
|------|----------|------|
| **Makespan (Obj ↓)** | $ \max_{r \in R} T_r $ | 所有机器人完成任务的最大耗时，反映系统整体吞吐效率 |
| **Average Order Completion Time (CompT ↓)** | 平均从订单到达至拣选完成的时间 | 衡量服务响应速度 |
| **Computation Time (Time ↓)** | 单实例平均推理耗时 | 反映算法实时性 |

#### 基线方法（Baselines）

分为两大类进行对比：

##### （1）Phased Methods（分阶段方法）
- **Order Allocation**:  
  - SQF (Shortest Queue First)  
  - WLB (Work Load Balance)  
  - OR Tools (CP-SAT Solver)
- **Robot Scheduling**:  
  - Nearest  
  - Earliest  
  - TSP  
  - PSMDRL (Transformer-based DRL)

> 组合成 9 种组合，如 WLB+Nearest、OR Tools+TSP 等

##### （2）Joint Methods（联合优化方法）
- **JOTP** [32]: 结合 KM 算法与 RL 的联合优化
- **SABS** [31]: 基于模拟退火与束搜索的混合启发式算法

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| 方法 | 场景 | Makespan ↓ | CompT ↓ | 推理时间 |
|------|------|------------|---------|----------|
| **SOAR (Ours)** | Real-Large | **828.44** | **275.68** | 16.34s |
| WLB+TSP | Real-Large | 940.60 | 329.74 | 6.57s |
| OR Tools+TSP | Real-Large | 1178.07 | 426.11 | 7m19s |
| SABS | Real-Large | 978.12 | 329.74 | 8m15s |

> ✅ **相比最强基线（WLB+TSP），SOAR 减少：**
> - **Makespan ↓ 7.5%**
> - **Average CompT ↓ 15.4%**
> - 同时保持 **<100ms 单步决策延迟**

---

### 🔍 与其他方法的对比分析

| 对比维度 | 发现 |
|--------|------|
| **vs. Decoupled Methods** | 所有分阶段方法因缺乏协同优化，导致资源利用率低、负载不均衡，性能显著落后于 SOAR |
| **vs. Joint Methods (SABS/JOTP)** | 尽管目标联合优化，但依赖滚动时域（Rolling Horizon）静态求解，存在严重延迟，实际执行时已过时；且无法捕捉连续长期演化 |
| **计算效率** | SOAR 推理速度快于所有基于 OR-Tools 和 SABS 的方法，接近轻量启发式方法（如 Nearest），具备工业部署潜力 |

---

### 🔬 消融实验（Ablation Study, Table 2）

验证各模块对性能的影响：

| 消融配置 | Makespan ↑ (%) | CompT ↑ (%) | 说明 |
|---------|----------------|------------|------|
| **Full SOAR** | — | — | 完整模型 |
| **w/o Soft Allocation** | +25.5% ~ +34.5% | +27.8% ~ +52.5% | 性能下降最严重 → 软分配是核心机制 |
| **w/o HGT** | +3.4% ~ +15.5% | +10.5% ~ +43.5% | HGT 对异构实体建模至关重要 |
| **w/o Bias** | +6.8% ~ +16.0% | +9.5% ~ +37.5% | phase-specific bias 显著提升探索效率 |
| **Only Bias (no DNN)** | +21.0% ~ +38.5% | +13.5% ~ +47.0% | 仅靠规则无法建模复杂全局依赖 |

> ✅ 结论：**软分配 + HGT + phase bias 的协同作用是成功关键**

---

### 📊 敏感性分析（Sensitivity Analysis）

- **候选货架数 K**：当 $ K=10 $ 时性能最优；过小（$K=1$）陷入贪婪搜索，过大（$K>10$）引入噪声干扰 HGT 学习。
- **Reward shaping 参数 p**：$ p=8 $ 时平衡了对瓶颈机器人的惩罚与训练稳定性；$ p=32 $ 导致梯度剧烈波动。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **联合优化优于分阶段决策**：  
   SOAR 通过软分配机制将 Order Allocation 与 Robot Scheduling 融合为统一过程，实现了真正的 **end-to-end joint optimization**。

2. **事件驱动机制保障实时性**：  
   ED-MDP 架构使得系统能够异步响应订单到达与机器人状态变更，确保在动态环境下仍具高响应速度。

3. **软分配机制支持高效批处理**：  
   实际部署中 **78.3% 的订单可通过软分配一次性满足**，大幅减少额外调度需求，提升 Shelf Hit Rate（+6%）与 Throughput（+3.85%）。

4. **sim-to-real 可行性强**：  
   在真实电商仓库（面积 5,158 m²，日订单超 13,000 单）中完成为期 7 天的实地测试，SOAR 在数字孪生平台与物理系统中均表现出一致优越性能。

---

### ⚠️ 局限性

| 局限 | 说明 |
|------|------|
| **高层调度为主** | 当前框架聚焦 high-level destination selection，未集成底层路径规划（path planning）与避障控制 |
| **依赖高质量仿真** | 数字孪生平台虽提升了迁移效果，但仍需精确建模物理动力学与通信延迟 |
| **冷启动问题** | DRL 模型需要大量训练数据，在新仓初期可能存在性能波动 |

---

### 🔮 未来工作方向

1. **整合物理层控制**：将 SOAR 与底层运动规划模块结合，构建 **end-to-end full-stack 框架**。
2. **跨仓泛化能力提升**：研究 zero-shot 或 few-shot 迁移学习，降低新仓部署成本。
3. **多目标优化扩展**：引入能耗、公平性、鲁棒性等更多优化目标，支持灵活策略切换。
4. **人机协作场景拓展**：支持 human-robot co-working 场景下的任务协同与安全交互。

---

## ✅ 总结

**SOAR 是首个成功实现 RMFS 中 Order Allocation 与 Robot Scheduling 实时联合优化的 DRL 框架**。它通过 **Soft Order Allocation + Event-Driven MDP + HGT + Phase-Knowledge Guidance** 的创新架构，在保证 **<100ms 延迟**的前提下，显著优于现有分阶段与联合优化方法。其在真实工业环境中的成功部署，证明了该方法具有极高的实用价值与广阔的应用前景。

</details>

---

### 11. [MP-ISMoE: Mixed-Precision Interactive Side Mixture-of-Experts for Efficient Transfer Learning](https://arxiv.org/abs/2605.04058)

**Authors**: Yutong Zhang, Zimeng Wu, Shangcai Liao, Shujiang Wu, Jiaxin Chen  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.04058v1  

#### Abstract
Parameter-efficient transfer learning (PETL) has emerged as a pivotal paradigm for adapting pre-trained foundation models to downstream tasks, significantly reducing trainable parameters yet suffering from substantial memory overhead caused by gradient backpropagation during fine-tuning. While memor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MP-ISMoE: Mixed-Precision Interactive Side Mixture-of-Experts for Efficient Transfer Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Memory-Efficient Transfer Learning (METL)** 方法虽然通过引入轻量级 side network 避免在主干网络中进行梯度反向传播，从而降低内存开销，但仍面临以下三大挑战：
1. **内存预算分配不合理**：side network 被严格限制参数规模，导致其表示能力受限。
2. **side network 结构僵化**：通常采用与 backbone 成比例缩小的设计，难以在内存效率与模型容量之间取得最优平衡。
3. **对 backbone 指导利用不足**：仅简单融合特征，未能有效挖掘主干网络中的通用知识来抑制过拟合和知识遗忘。

### 🚀 提出的新方法
本文提出一种新型 METL 框架——**Mixed-Precision Interactive Side Mixture-of-Experts (MP-ISMoE)**，包含两个核心组件：

#### （1）Gaussian Noise Perturbed Iterative Quantization (GNP-IQ)
- 对 backbone 的权重进行**仅权重量化（weight-only quantization）**，将大部分参数压缩为 8-bit，显著减少内存占用。
- 引入**迭代重量化机制**：每隔若干 epoch 动态更新部分权重的量化参数（scale 和 zero-point），以适应微调过程中权重分布的变化。
- 加入**高斯噪声扰动（Gaussian noise perturbation）**：在每次重量化前注入可学习的噪声，模拟未被捕捉的参数更新趋势，缓解量化误差累积。

#### （2）Interactive Side Mixture-of-Experts (ISMoE)
- 在 side network 中引入 **Sparse MoE 结构**，通过多个专家（experts）提升模型容量，同时保持计算稀疏性。
- 设计**跨网络交互机制**：利用 backbone 输出的 [CLS] token 作为通用知识表征，指导 side network 中专家的选择。
- 通过计算 [CLS] token 与每个专家代表性 token 的相关性得分，作为路由概率的先验，实现“知识引导”的专家选择，抑制知识遗忘。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **内存效率** | GNP-IQ 显著压缩 backbone 内存，释放资源用于扩展 side network，整体仍保持低内存消耗。 |
| **性能表现** | ISMoE 增强 side network 表达能力，并通过交互机制保留通用知识，显著优于现有 METL 方法。 |
| **推理成本** | 由于 MoE 是稀疏激活（top-k），推理时无额外开销，适合部署。 |
| **灵活性与可扩展性** | 可灵活调整量化粒度、专家数量等，在不同 backbone 上具有良好泛化性。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
涵盖多模态与纯语言任务，验证方法的广泛适用性：

#### Vision-Language (VL) Tasks：
- **Image-Text Retrieval (ITR)**：Flickr30K, MSCOCO
- **Video-Text Retrieval (VTR)**：MSVD, MSR-VTT
- **Visual Question Answering (VQA)**：VQAv2, GQA
- **Visual Grounding (VG)**：RefCOCO, RefCOCO+, RefCOCOg

#### Natural Language Processing (NLP) Tasks：
- **GLUE Benchmark**：CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE, STS-B

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| **Backbone 架构** | VSEoo (ResNeXt-101 + BERT), CLIP4Clip (ViT-B/32 + Text Transformer), CLIP-ViL, MDETR (ResNet-101 + RoBERTa-B), T5-base / T5-large |
| **训练配置** | 复用 UniPT/SHERL 的优化器、warm-up、batch size、epoch 数等，确保公平比较 |
| **精度设置** | Backbone 权重 8-bit，LayerNorm 全精度可训练，side network 16-bit 训练 |
| **评估指标** |  
| - ITR/VTR | Recall@1 (R@1), Rsum (R@1+R@5+R@10) |
| - VQA/GQA | Accuracy |
| - VG | mAP |
| - GLUE | Accuracy, F1, Matthews Correlation, Pearson/Spearman Correlation |

### 🆚 基线方法对比
分为两类进行对比：

#### （1）METL 方法（内存高效）
- **LST**, **UniPT**, **SHERL**

#### （2）PETL 方法（参数高效）
- **BitFit**, **Prompt Tuning**, **Adapter**, **LoRA**

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & Table 2）

#### ✅ 在 Vision-Language 任务上的表现（Table 1）
| 方法 | 参数 (M) | 内存 (G) | Flickr30K R@1 ↑ | MSCOCO5K Rsum ↑ | VQAv2 Acc ↑ | RefCOCOg mAP ↑ |
|------|--------|--------|------------------|------------------|-------------|----------------|
| Fully-FT | 201.2 | 176.8 | 85.6 | 546.6 | 76.7 | 80.95 |
| UniPT | 12.4 | 24.4 | 84.8 | 537.4 | 75.53 | 77.33 |
| **Ours+** | **11.8** | **25.4** | **87.4** | **547.0** | **76.91** | **78.08** |

> ✅ **MP-ISMoE 在所有指标上均超越现有 METL 方法**，接近甚至超过全量微调（Fully-FT），且参数和内存开销极低。

#### ✅ 在 NLP 任务上的表现（GLUE, T5-base, Table 2）
| 方法 | 参数 (%) | Train Mem (G) ↓ | Avg Score ↑ |
|------|---------|------------------|--------------|
| Fully-FT | 100% | 17.6 | 85.2 |
| Adapter | 1.63% | 13.0 | 85.3 |
| LoRA | 1.71% | 12.6 | 85.3 |
| **OursT** | **2.41%** | **3.2** | **84.8** |
| **Ours+** | **1.48%** | **3.3** | **84.9** |

> ✅ 尽管训练内存仅为 Fully-FT 的 ~18%，MP-ISMoE 达到与 PETL 方法相当甚至更优的性能，且显著优于其他 METL 方法（如 LST）。

#### ✅ 在更大 backbone（T5-large）上的扩展性
| 方法 | 参数 (%) | Train Mem (G) | GLUE Avg ↑ |
|------|---------|---------------|------------|
| SHERL (large) | 0.64% | 7.1 | 87.5 |
| **OursT (large)** | **0.81%** | **7.6** | **88.1** |

> ✅ 展现出良好的可扩展性，在大模型上仍能持续提升性能。

### 🔍 消融实验结果（Table 3 & Table 4）

#### （1）模块有效性分析（Table 3）
| 配置 | Memory (G) | Flickr30K R@1 | MSCOCO5K Rsum |
|------|------------|----------------|----------------|
| Baseline (UniPT) | 24.4 | 84.8 | 445.3 |
| + GNP-IQ | 18.4 | 83.7 | 442.7 |
| + ISMoE | 32.4 | 86.9 | 450.9 |
| **+ GNP-IQ + ISMoE** | **25.5** | **86.5** | **449.1** |

> ✅ GNP-IQ 节省内存（↓6G），ISMoE 提升性能；二者结合实现最佳权衡，性能↑1.5% R@1，内存仅轻微增加。

#### （2）GNP-IQ 组件分析（Table 4）
| 精度策略 | Gaussian Noise | Memory (G) | Rsum ↑ |
|----------|----------------|------------|--------|
| Full-Precision | – | 24.4 | 537.4 |
| Mixed-Precision | – | 15.4 | 531.4 |
| **Mixed-Precision** | **✓** | **18.4** | **534.8** |

> ✅ 高斯噪声扰动有效缓解量化带来的性能下降，恢复约 3.4 点 Rsum。

#### （3）ISMoE 组件分析（Table 4）
| 结构 | Correlation Guidance | R@1 ↑ | Rsum ↑ |
|------|------------------------|--------|--------|
| Base | – | 84.8 | 537.4 |
| + MoE | – | 86.3 | 543.1 |
| **+ MoE + Correlation** | ✓ | **86.9** | **544.3** |

> ✅ 跨网络交互机制进一步提升性能，证明“知识引导”专家选择的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **混合精度 + 侧边 MoE 是高效的资源再分配策略**：通过 GNP-IQ 压缩 backbone 内存，将节省资源用于扩展 side network 容量，打破传统 METL 的表达瓶颈。
2. **交互式专家选择可缓解知识遗忘**：利用 backbone 的 [CLS] token 指导 side network 的路由决策，实现了主干与侧支的知识协同，提升泛化能力。
3. **MP-ISMoE 实现三重高效**：
   - **参数高效**（trainable params < 3%）
   - **内存高效**（training memory ↓50–80% vs PETL）
   - **推理高效**（sparse MoE 无额外延迟）
4. **在多种架构和任务上具有强泛化性**：从 VSEoo 到 T5-large，从图像检索到 GLUE，均表现出一致优越性。

### ⚠️ 方法的局限性
- **依赖 backbone 的 [CLS] token**：若 backbone 不提供良好语义聚合 token（如某些 CNN 架构），可能影响交互效果。
- **MoE 训练稳定性**：稀疏路由可能导致专家负载不均衡，需 careful initialization 或 load balancing loss（文中未详述）。
- **硬件支持要求**：低比特量化和 MoE 结构对推理框架有一定依赖，部署需适配。

### 🔮 未来工作方向
- 探索更复杂的跨网络交互机制（如 attention-based fusion）。
- 结合动态稀疏性与自适应量化策略，进一步提升效率。
- 扩展至更多模态（如音频、点云）和更大规模模型（如 LLMs）。
- 研究在边缘设备上的实际部署性能与能耗表现。

---

> **总结一句话**：  
> **MP-ISMoE 通过“压缩主干换空间，增强侧支提性能”，并以“知识交互防遗忘”，在极低内存开销下实现了媲美全量微调的迁移学习性能，是 METL 范式的一次重要推进。**

</details>

---

### 12. [Predict-then-Diffuse: Adaptive Response Length for Compute-Budgeted Inference in Diffusion LLMs](https://arxiv.org/abs/2605.04215)

**Authors**: Michael Rottoli, Subhankar Roy, Stefano Paraboschi  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.04215v1  

#### Abstract
Diffusion-based Large Language Models (D-LLMs) represent a promising frontier in generative AI, offering fully parallel token generation that can lead to significant throughput advantages and superior GPU utilization over traditional autoregressive paradigm. However, this parallelism is constrained ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Predict-then-Diffuse: Adaptive Response Length for Compute-Budgeted Inference in Diffusion LLMs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
**Diffusion-based Large Language Models (D-LLMs)** 虽然支持全并行 token 生成，显著提升吞吐量和 GPU 利用率，但在推理时必须预先设定固定的 **response length**（输出长度）。这一设计带来了两个严重问题：

- **计算浪费**：当真实响应远短于预设长度时，大量计算资源被用于处理无语义意义的 `<PAD>` tokens。
- **输出截断**：若预设长度不足，则导致输出被截断，需重新以更长长度运行，引入不可预测的延迟峰值。

这种静态长度机制在面对输入复杂度差异大的查询时，效率极低。

---

### 🚀 提出的新方法：**Predict-then-Diffuse**

提出一个简单、模型无关（model-agnostic）的框架 **Predict-then-Diffuse**，其核心思想是：

> 在 D-LLM 推理前，先预测最优响应长度，再据此执行生成。

该框架包含两个关键组件：

#### （1）Adaptive Response Length Predictor (**AdaRLP**)
- 一个轻量级模块，接收原始输入 prompt，输出预测的响应长度。
- 采用 **CatBoost** 实现，因其具备：
  - 极快推理速度（<1ms）
  - 原生支持文本特征（无需复杂特征工程）
  - 高准确率且无需调参
- 可直接使用标准 SFT 数据集训练，无需额外标注。

#### （2）Data-driven Safety Margin（安全边距）
- 为防止低估导致截断，引入基于统计的安全边距 $\delta$：
  $$
  \delta = \text{Quantile}_{p_{\text{safe}}}\{k - L \mid k > L\}
  $$
  即取“正向误差”（真实长度 > 预测长度）的高分位数（如 95%），加到预测值上。
- 最终传给 D-LLM 的长度为：
  $$
  L^* = \min(L + \delta, L_{\text{max}})
  $$

---

### 🔍 相比现有方法的优势

| 方法 | 是否修改架构 | 是否需要重训练 | 是否支持任意长度 | 计算效率 |
|------|---------------|----------------|--------------------|-----------|
| **Block Diffusion** | 是 | 是 | 是 | 中等（半自回归） |
| **TimeBill** | 否 | 否 | 仅适用于 AR LLMs | 时间预算控制 |
| **Predict-then-Diffuse (Ours)** | ❌ 否 | ❌ 否 | ✅ 是 | ⭐ 极高 |

**优势总结**：
- **零架构改动**：不改变 D-LLM 结构或训练目标。
- **极低开销**：AdaRLP 推理时间可忽略（<0.04ms）。
- **高效节能**：大幅减少 FLOP 开销，避免 `<PAD>` 浪费。
- **稳定延迟**：通过 Safety Margin 将 fallback 率降至 0.1%，实现近确定性延迟。
- **通用性强**：适用于任何 D-LLM（如 LLaDA、Dream）。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
构建了一个包含 **39,994 条 prompt-response 对** 的混合基准数据集，来源包括：
- ShareGPT
- Alpaca
- Dolly-15k
- OpenOrca
- ELI5

**特点**：
- 输出长度高度可变（均值 ~96，标准差 ~120，峰度 ~107）
- 强调长文本生成，避免偏向短回复的偏差
- 按 80/20 划分训练/测试集
- 使用 `GPT2TokenizerFast` 分词并计算长度

---

### ⚙️ 实验设置

- **主模型**：LLaDA-8B（Discrete D-LLM）
- **扩散步数**：T = 128
- **最大长度**：$L_{\text{max}} = 4096$
- **硬件平台**：NVIDIA H100（80GB VRAM）
- **评估工具**：DeepSpeed 用于测量 FLOP

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **Total TFLOP** | 整个测试集上的总浮点运算量（含 fallback 重试惩罚） |
| **Savings (%)** | 相比 Max Length 基线的计算节省比例 |
| **Fallback Rate (%)** | 因长度不足需重新推理的比例 |
| **RMSE / MAE** | 响应长度预测误差（越小越好） |
| **% ≤10% error** | 预测误差在真实长度 10% 内的比例 |

---

### 🆚 基线方法对比

| 基线方法 | 设置说明 |
|---------|----------|
| **Max Response Length** | 固定 $L=4096$，默认设置 |
| **Static Response Length** | 固定 $L=200$，截断后翻倍重试 |
| **Mean Doubling Heuristic** | 初始 $L≈95$（数据集均值），逐步翻倍直至成功 |
| **Oracle** | 理想情况，已知真实长度 $L=k$ |

---

## 3. 主要实验结果和性能指标

### ✅ 预测器性能（AdaRLPtext-only）

| 模型 | RMSE | MAE | % ≤10% error |
|------|------|-----|--------------|
| AdaRLPengineered | 81.6 | 51.6 | 10.5% |
| **AdaRLPtext-only** | **11.4** | **1.7** | **97.5%** |

> 💡 发现：**纯文本输入的 AdaRLP 表现远超手工特征工程版本**，说明 CatBoost 能从 prompt 中自动学习长度相关模式（如 “write a poem” vs “write a novel”）。

---

### 🔋 计算效率对比（TFLOP）

| 方法 | TFLOP | Savings | Fallback Rate |
|------|--------|----------|----------------|
| Max Response Length (4096) | 4.03 | 0.0% | 0.0% |
| Static Response Length (200) | 0.054 | 98.6% | 22.1% |
| Doubling Heuristic | 0.027 | 99.31% | — |
| **Predict-then-Diffuse (Ours)** | **0.026** | **99.34%** | **0.1%** |
| Oracle | 0.024 | 99.4% | 0.0% |

> 🏆 **成果**：相比默认设置，**节省高达 99.34% 的 FLOP**，接近理想 Oracle 性能。

---

### 🔄 在双峰分布下的鲁棒性验证

模拟极端场景：60% 短查询（均值 50），40% 长报告（均值 3000）

| 方法 | 表现 |
|------|------|
| **Doubling Heuristic** | 多次 fallback，累计成本高（如 95→190→...→4096） |
| **Predict-then-Diffuse** | 仍保持单次推理为主，**计算优势达 19%** |

> ✅ 证明该方法在高方差场景下更具泛化能力。

---

### ⏱️ 延迟确定性（Latency Determinism）

| 方法 | 特点 |
|------|------|
| Heuristic 策略 | 长尾请求引发多次推理，用户感知延迟波动大 |
| **Predict-then-Diffuse** | 99.9% 请求实现“一次生成”，配合 Safety Margin 将 fallback 从 1.43% 降至 **0.1%**，提供稳定端到端延迟 |

> ✅ 对生产环境中的 SLA 支持更强。

---

### ✅ 生成质量影响分析

参考 LLaDA 原始研究 [3] 的结果（见 Table IV）：

| Canvas Size | GSM8K Accuracy | HumanEval Pass@1 |
|------------|----------------|------------------|
| 1024       | 70.3%          | 35.4%            |
| 512        | 70.8%          | 32.9%            |
| 256        | 70.0%          | 32.9%            |

> 🔍 **结论**：只要 $L^* \geq k$（足够容纳答案），生成质量基本不受影响；退化仅发生在严重截断时，而本方法通过 Safety Margin 和 fallback 有效规避。

此外，D-LLM 展现出**自适应 verbosity** 能力（见 Fig. 5）：
- 短长度 → 更简洁的回答
- 长长度 → 更详细的推理过程  
→ 表明动态调整长度不会损害语义一致性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **D-LLM 的固定长度机制造成严重计算浪费**，尤其在长上下文窗口（如 32k/128k）普及的趋势下问题愈发突出。
2. **Predict-then-Diffuse 显著降低 FLOP 开销（99.34%）**，几乎达到 Oracle 水平。
3. **AdaRLP 无需复杂特征即可精准预测长度**，CatBoost 原生文本处理能力足够强大。
4. **Safety Margin 机制有效抑制 fallback**，将失败率压至 0.1%，保障延迟稳定性。
5. **方法完全兼容现有 D-LLM 架构**，无需修改训练或推理流程，即插即用。

---

### ⚠️ 局限性

1. **批处理（batching）受限**：
   - 不同样本预测长度不同，难以统一 batching。
   - 当前解决方案：按预测长度分组（grouping），但非最优。
2. **依赖高质量历史数据训练 AdaRLP**：
   - 若部署领域与训练数据差异大，预测精度可能下降。
3. **极端长尾仍需 fallback 至 $L_{\text{max}}$**：
   - 虽概率极低，但仍存在理论最坏情况。

---

### 🔮 未来工作方向

1. **优化动态 batching 策略**：
   - 借鉴 vLLM 的 PagedAttention 技术，探索基于预测长度的智能调度。
2. **在线学习 AdaRLP**：
   - 在线收集反馈，持续更新预测模型以适应新领域。
3. **扩展至 Continuous D-LLMs**：
   - 验证框架是否适用于连续空间扩散模型。
4. **多模态适配**：
   - 将 Predict-then-Diffuse 思路推广至图像、音频等生成任务。

---

## 🔗 代码开源

👉 [GitHub 仓库](https://github.com/mchl-labs/predict-then-diffuse) 已公开源码与实验配置，便于复现与集成。

--- 

> **一句话总结**：  
> *Predict-then-Diffuse* 通过“先预测后生成”的轻量范式，在不改动 D-LLM 架构的前提下，实现了接近理论极限的计算效率与稳定的推理延迟，为大规模部署 Diffusion LLM 提供了一条实用高效的路径。

</details>

---

### 13. [Explaining and Preventing Alignment Collapse in Iterative RLHF](https://arxiv.org/abs/2605.04266)

**Authors**: Etienne Gauthier, Francis Bach, Michael I. Jordan  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.04266v1  

#### Abstract
Reinforcement learning from human feedback (RLHF) typically assumes a static or non-strategic reward model (RM). In iterative deployment, however, the policy generates the data on which the RM is retrained, creating a feedback loop. Building on the Stackelberg game formulation of this interaction, w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Explaining and Preventing Alignment Collapse in Iterative RLHF》核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文系统性地揭示并解决了一个在**迭代式 RLHF (Iterative RLHF)** 中普遍存在的严重问题——**对齐崩溃 (Alignment Collapse)**。

- **问题本质**：在标准的迭代 RLHF 流程中，策略（Policy）会生成数据来重新训练奖励模型（Reward Model, RM）。由于策略优化器（如 PPO）是“短视的”（myopic），它只最大化当前 RM 给出的代理奖励（proxy reward），而完全忽略了其行为对未来 RM 参数更新的影响。
- **后果**：这导致策略会系统性地利用 RM 的盲点（例如，在 RM 训练数据稀疏的区域进行过估计），产生高奖励但低质量的输出。当这些被利用的数据用于重新训练 RM 时，会进一步强化 RM 在这些区域的错误，形成一个**病态的反馈循环**，最终导致模型性能偏离真实的人类偏好，即发生“对齐崩溃”。

### 提出的新方法和新思路
论文提出了一个名为 **Foresighted Policy Optimization (FPO)** 的机制设计干预方案，以从根本上解决对齐崩溃问题。

- **理论创新**：
  - 将迭代 RLHF 形式化为一个 **Stackelberg 博弈**，其中策略是领导者（Leader），RM 是跟随者（Follower）。
  - 通过解析展开（analytically unrolling）这个双层优化（bilevel optimization）问题，推导出策略真正的总优化梯度由两部分组成：
    1.  **标准策略梯度 (Standard Policy Gradient)**：即传统算法使用的部分。
    2.  **参数引导梯度 (Parameter-steering Gradient)**：这部分捕捉了策略对其自身未来 RM 参数的影响。
  - 证明了忽略参数引导梯度是导致对齐崩溃的几何学原因，并揭示了最优策略实际上是在最大化一个**隐式的有效奖励 (implicit effective reward)**，该奖励等于代理奖励加上一个参数引导项。

- **方法创新 (FPO)**：
  - **核心思想**：将缺失的参数引导梯度作为一个正则化项（penalty）显式地加回到策略的目标函数中，强制策略在优化时考虑其对未来 RM 的影响。
  - **可扩展实现**：为了克服精确计算所需的逆 Hessian 矩阵在高维模型中的不可行性，论文提出了一种基于 **TracIn** 方法的一阶近似放松版本。这种放松版的 FPO 惩罚项仅需额外的一次梯度计算，即可在线高效计算，易于集成到现有的 RLHF 流水线中。

### 相比现有方法的优势
- **根本性**：不同于以往通过改进 RM（如使用集成模型）或修改优化目标（如约束优化）等静态鲁棒性方法，FPO 直接针对动态耦合过程中的博弈论缺陷进行干预，从源头上恢复了正确的优化动力学。
- **通用性**：FPO 是一种机制设计层面的干预，可以作为附加惩罚项无缝集成到任何基于梯度的策略优化器（如 PPO）中。
- **可扩展性**：提出的 TracIn 近似使其能够应用于大规模 LLM，解决了传统双层优化方法计算成本过高的问题。

## 2. 核心实验方法和设置

### 数据集
- **控制环境 (Controlled Environments)**：
  - **线性 RM 实验**：在一个连续动作空间中定义了一个理想化的高斯效用函数 `U(y)`，并使用一个线性 RM `r_φ(y) = w^T y`。
  - **神经网络 RM 实验**：同样使用高斯效用函数，但 RM 是一个 2 层的 MLP，更接近实际情况。
- **LLM 对齐管道 (LLM Alignment Pipeline)**：
  - **训练数据**：使用 `UltraFeedback` 数据集中的提示（prompts）进行迭代训练。
  - **评估数据**：使用 `TruthfulQA` 数据集（共 817 个提示）进行最终的盲评比较。

### 实验设置和评估指标
- **总体流程**：模拟了完整的迭代 RLHF 循环，交替进行策略优化和 RM 重训练。
- **策略模型**：在 LLM 实验中，策略 `π_θ` 基于 `Llama-3.2-1B-Instruct`，并通过 **LoRA** 进行微调。
- **奖励模型**：RM `r_φ` 基于 `DeBERTa-v3-base`，其主干网络被冻结，仅微调分类头。
- **偏好模拟**：使用一个冻结的 `Llama-3.2-1B` 模型作为“偏好预言机”（preference oracle），为 RM 训练和 FPO 惩罚项的计算提供“真实”的人类偏好信号。
- **评估方法**：
  - **主要指标**：在 `TruthfulQA` 上进行**盲评比较 (blind evaluation)**。使用一个强大的 `Llama-3.3-70B` 模型作为裁判，对不同方法生成的回答进行成对比较，判断哪个更优。
  - **统计检验**：使用双边二项检验（two-sided binomial test）计算胜率差异的 p 值。
  - **辅助指标**：在控制环境中，直接绘制策略轨迹的 PCA 投影图，直观展示收敛路径。

### 基线方法对比
- **主要基线**：**标准迭代 RLHF (Standard Iterative RLHF)**，即不使用任何 FPO 惩罚项的传统方法。
- **FPO 变体**：
  - **Relaxed FPO (R_FPO)**：使用了预言机提供的“过度自信”（overconfidence）信号的放松版惩罚项。
  - **Practical FPO (R_FPO)**：完全无需预言机的实用版惩罚项，仅依赖于奖励梯度的范数。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **LLM 实验 (TruthfulQA 盲评)**：
  - **Relaxed FPO vs. Standard RLHF**：胜率 **56.6%** (188 胜 vs 144 负)，p-value = **0.014**，表明显著优于基线。
  - **Practical FPO vs. Standard RLHF**：胜率 50.9% (140 胜 vs 135 负)，p-value = 0.41，未达到统计显著性，但整体仍略占优势。
  - **Relaxed FPO vs. Practical FPO**：胜率 54.8% (167 胜 vs 138 负)，p-value = 0.076，表明包含更多对齐信号的放松版表现更好。

- **控制环境实验**：
  - **可视化结果**：PCA 投影图（图2）清晰显示，标准 RLHF 的轨迹会漂移远离真实最优解（Human Ideal），而 FPO 成功收敛到该点。
  - **消融分析**：在使用线性 RM 的实验中，即使使用实用版 FPO，也能有效抑制策略向噪声维度的恶意探索，防止对齐崩溃。

### 消融实验结果
- **能力权衡 (Capability Trade-off)**：详细分析（表5）表明，`Practical FPO` 在“标准事实性”任务上表现良好，但在“对抗性”提示（如诱导奉承或欺骗）下表现不佳，甚至不如基线。这说明无差别的梯度范数惩罚可能使模型变得过于保守。
- **通用能力保持**：在 `MMLU` 和 `ARC-Challenge` 基准测试上的结果显示，所有方法（包括 FPO 变体）的性能在统计误差范围内基本相同，证明 FPO 并未损害模型的基础推理能力，而是精准地限制了其在易被利用区域的行为。

## 4. 关键结论和发现

### 主要发现
1.  **对齐崩溃的根本原因**：标准迭代 RLHF 的失败源于其**优化短视性 (optimization myopia)**，即策略优化器忽略了其对 RM 未来参数的“参数引导”效应，导致其贪婪地剥削 RM 的盲点。
2.  **FPO 的有效性**：通过显式地将参数引导梯度作为正则化项加入策略目标，**FPO 成功恢复了 Stackelberg 博弈的自校正动力学**，在控制环境和真实的 LLM 管道中都有效防止了对齐崩溃。
3.  **惩罚项的设计至关重要**：包含更多信息（如过度自信方向）的 `Relaxed FPO` 显著优于仅依赖梯度范数的 `Practical FPO`，后者存在能力权衡问题。这表明未来的改进应致力于设计更智能的引导信号。

### 方法的局限性
1.  **强凸性假设**：理论推导依赖于 RM 损失函数的强凸性假设，这对于过参数化的神经网络 RM 来说并不成立，这是一个重要的理论简化。
2.  **实用版的性能妥协**：完全无需预言机的 `Practical FPO` 在复杂对抗场景下效果有限，表现出过度保守的倾向。
3.  **实验规模**：LLM 实验虽然使用了 `Llama-3.2-1B`，但仍属于中等规模，需要在更大模型上进一步验证。

### 未来工作方向
1.  **非凸设置下的理论扩展**：将分析框架推广到非凸的、过参数化的神经网络 RM 场景。
2.  **改进的实用惩罚项**：探索更精细、无需预言机的启发式方法来近似参数引导梯度，以避免 `Practical FPO` 的保守性问题。
3.  **与其他对齐技术的结合**：研究 FPO 如何与 RM 集成、监督微调（SFT）或其他安全技术协同工作。
4.  **动态检测机制**：利用文中提到的“残余引导梯度”作为诊断工具，开发在部署过程中动态监测对齐崩溃风险的方法。

</details>

---

### 14. [Data-dependent Exploration for Online Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2605.04477)

**Authors**: Zhen-Yu Zhang, Yuting Tang, Jiandong Zhang, Lanjihong Ma, Masashi Sugiyama  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.04477v1  

#### Abstract
Online reinforcement learning from human feedback (RLHF) has emerged as a promising paradigm for aligning large language models (LLMs) by continuously collecting new preference feedback during training. A foundational challenge in this setting is exploration, which requires algorithms that enable th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Data-dependent Exploration for Online Reinforcement Learning from Human Feedback

## 1. 论文的主要贡献和创新点

### 解决的问题
在线 **Reinforcement Learning from Human Feedback (RLHF)** 中，探索（exploration）是提升对齐效果的关键环节。然而，现有方法通常依赖于**on-policy 期望**来设计探索奖励（exploration bonus），这类方法在历史偏好数据覆盖有限的情况下难以准确估计不确定性，导致策略可能过早放弃潜在高价值但尚未充分探索的行为区域。

这一问题在实际在线 RLHF 流程中尤为突出，因为模型只能基于训练轨迹上收集的历史比较数据进行学习，缺乏全局覆盖，容易造成探索信号噪声大、效率低。

---

### 提出的新方法：DEPO
本文提出 **Data-dependent Exploration for Preference Optimization (DEPO)**，一种简单且可扩展的用于在线 RLHF 的探索机制。

#### 核心思想：
- 利用历史偏好比较数据，在一个**表示空间（representation space）**中构建一个**数据依赖的不确定性奖励（data-dependent uncertainty bonus）**。
- 借鉴 **Upper Confidence Bound (UCB)** 思想，通过计算每个响应对在特征空间中的“置信半径”来衡量其不确定性。
- 不确定性高的区域（即历史覆盖不足的方向）获得更大的探索激励，从而引导策略生成更具信息量的比较样本。

#### 技术实现：
- 使用 LLM 自身最后层隐藏状态的均值池化作为 `(x, y)` 的表示 `φ(x, y)`。
- 构造成对特征 `w(x, y, y') = [φ(x, y); φ(x, y')]`。
- 维护一个协方差矩阵 $ V_t $ 来追踪历史比较在表示空间中的几何分布。
- 探索奖励定义为椭球型 UCB 形式：  
  $$
  b_t(x, y, y') = \frac{\beta_{\text{conf}}}{\sqrt{K_t}} \|w(x, y, y')\|_{V_t^{-1}}
  $$
  其中 $ K_t $ 是沿轨迹的局部曲率（local curvature），$ \beta_{\text{conf}} $ 是缩放系数。

---

### 相比现有方法的优势
| 方面 | 现有方法 | DEPO |
|------|--------|------|
| **探索奖励来源** | 多为统一或基于当前策略的启发式设计 | 完全**数据依赖**，由历史比较动态决定 |
| **理论分析** | 多为最坏情况下的 regret bound（如 $ \exp(O(R_{\max}))\sqrt{dT} $） | 提出**实例相关（instance-dependent）regret bound**，能自适应任务难度，在满足“表示多样性”时更紧 |
| **探索效率** | 可能在已覆盖区域重复采样或忽略潜在高价值区域 | 显式鼓励向**未充分覆盖但可能高价值**的方向探索 |
| **兼容性与实现** | 部分复杂 | 轻量级，支持高效在线更新（使用 Sherman-Morrison 公式维护 $ V^{-1} $），易于集成到大规模 LLM 训练 |

---

## 2. 核心实验方法和设置

### 数据集
实验在多个公开基准上进行，涵盖知识、推理、真实性与代码生成能力：

- **MMLU**: 多学科知识理解
- **GPQA**: 高难度问答（graduate-level）
- **TruthfulQA**: 抵抗幻觉、追求真实性的能力
- **GSM8k**: 数学推理
- **AlpacaEval 2.0 (AE2)**: 开放指令遵循能力评估
- **MT-bench (MT)**: 多轮对话质量评估
- **LiveCodeBench**: 编程问题求解与执行反馈

此外，模拟两种典型对齐场景：
- **IID**: 领域特定对齐（domain-specific alignment），强调快速收敛
- **Alpaca**: 通用助手对齐（generalist alignment），强调泛化与行为覆盖

---

### 实验设置与评估指标

#### 模型架构
- **基础模型**：Llama-3-8B-Flow-SFT
- **参考模型**：同上
- **训练偏好模型**：GRM-Llama3-8B-rewardmodel-ft
- **训练流程**：三轮迭代式在线 RLHF（iter1 → iter2 → iter3）

#### 评估指标
- **标准任务**：准确率（Accuracy）如 MMLU、GSM8k 等
- **对齐任务**：
  - **Win Rate (WR)**：相对于 baseline 的胜率
  - **Average Reward (AvgR)**：在 AlpacaEval 或 MT-bench 上的平均得分
- **消融研究**：有效比较比率（effective comparison ratio）、不同超参数影响

#### 基线方法对比
- **DPO** (Direct Preference Optimization)：标准离线/在线偏好优化基线
- **XPO** (Exploratory Preference Optimization)：基于隐式乐观性的探索方法
- **POPO** (Preference Optimization with Preference-based Exploration)：引入偏好驱动探索的先进方法

所有方法共享相同的基础模型、数据源和 sampler 设计，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| Method | MMLU | GPQA | TruthfulQA | GSM8k | AE2 WR | AE2 AvgR | MT WR | MT AvgR |
|--------|------|------|------------|-------|--------|----------|--------|---------|
| DPO-iter3 | 63.08 | 31.24 | 59.61 | 77.38 | 72.5 | -2.36 | 91.2 | -0.02 |
| XPO-iter3 | 63.12 | 31.01 | 59.34 | 78.41 | 72.9 | -2.11 | 91.6 | 0.64 |
| POPO-iter3 | 63.18 | 31.94 | 59.07 | 77.63 | 73.2 | -2.02 | 92.4 | 0.60 |
| **DEPO-iter3** | **63.23** | **32.50** | **59.82** | **78.39** | **74.1** | **-1.94** | **92.8** | **0.66** |

✅ **结论**：DEPO 在绝大多数指标上达到最优，尤其在 **GPQA** 和 **AlpacaEval/MT-bench** 上优势明显。

---

### 与基线方法的对比结果
- DEPO 在三轮迭代后持续优于所有 baseline，表明其**探索机制带来了更高质量的数据收集**。
- 在 **GPQA** 上出现非单调趋势（iter2 下降，iter3 回升），而 DEPO 的下降幅度最小、回升最强，说明其探索更稳健，能更快从不确定区域恢复。
- 在 **IID 和 Alpaca 场景下**，DEPO 均取得更高的 Win Rate 和 Average Reward，验证了其在**领域专用与通用对齐任务上的普适有效性**。

---

### 消融实验结果

#### （1）探索强度 $ c_b $ 的影响（控制 $ \beta_{\text{conf}} $）
- 测试 $ c_b \in \{1e{-1}, 2e{-2}, 1e{-2}\} $
- 结果显示 $ c_b = 2e{-2} $ 表现最佳，带来最高的有效比较比率
- ✅ **结论**：适度的探索强度可在**探索与利用之间取得更好平衡**

#### （2）Sampler 策略选择
| Sampler | WR (iter3) | AvgR (iter3) |
|--------|------------|--------------|
| $ (\pi_t, \pi_t) $（on-policy） | **92.5** | **0.21** |
| $ (\pi_{t-1}, \pi_{\text{ref}}) $（mixed） | 91.7 | -2.63 |

✅ **结论**：使用当前策略自身生成两个候选（on-policy sampler）产生**更具挑战性和信息量的比较对**，加速决策边界优化。

#### （3）KL 正则化系数 $ \beta $ 的影响
- 固定 $ \alpha \beta $ 比例以保持梯度尺度一致
- $ \beta = 3e{-2} $ 效果最好；过大限制更新，过小导致不稳定
- ✅ **结论**：中等强度的 KL 正则有助于稳定训练并促进有效探索

#### （4）代码生成任务（LiveCodeBench）

| Model | Easy | Medium | Hard |
|-------|------|--------|------|
| Qwen2.5-Coder-7B-Instruct | 56.1 | 3.8 | 6.9 |
| + DPO-iter3 | 57.6 | 14.5 | 8.7 |
| + XPO-iter3 | 58.4 | 15.2 | 9.4 |
| + POPO-iter3 | 58.2 | 15.0 | 9.2 |
| **+ DEPO-iter3** | **65.3** | **18.8** | **9.7** |

✅ **结论**：DEPO 在基于执行反馈的真实世界任务中仍显著领先，证明其不仅适用于合成偏好，也适用于**可验证正确性**的任务。

---

## 4. 关键结论和发现

### 主要发现
1. **数据依赖的探索机制至关重要**：传统 on-policy 探索在覆盖受限的在线 RLHF 中表现不佳；DEPO 通过显式建模历史数据的覆盖缺口，实现了更高效的探索。
2. **表示空间中的 UCB 是有效的实现路径**：将响应对映射到表示空间，并利用协方差矩阵衡量覆盖程度，是一种轻量且可扩展的设计。
3. **理论与实践一致**：提出的 instance-dependent regret bound 在理论上优于最坏情况界，实验也证实 DEPO 在多样任务中持续提升性能。
4. **探索质量优于数量**：DEPO 并非简单增加数据量，而是提升了**每次比较的信息价值**，从而提高样本效率。

---

### 方法的局限性
- **依赖表示质量**：性能受表示函数 $ \phi(x, y) $ 的表达能力影响，若表示无法区分关键语义差异，则探索可能失效。
- **假设线性可实现性（Linear Realizability）**：理论分析依赖于奖励函数在线性特征空间中可实现，这在复杂 LLM 场景中仅为近似成立。
- **超参数敏感性**：尽管进行了调优，$ c_b $、$ \beta $ 等参数仍需谨慎设置，自动化调节仍有空间。
- **计算开销**：虽然使用 Sherman-Morrison 更新逆矩阵，但在极长训练序列中维护 $ V_t $ 仍有一定内存与计算成本。

---

### 未来工作方向
1. **动态调整探索强度**：根据学习进度自动调节 $ \alpha $ 或 $ c_b $，例如结合不确定性下降速率。
2. **更强大的表示学习**：引入可训练的表示模块，而非冻结模型提取特征。
3. **跨任务迁移探索知识**：将在一个任务中学到的“难探索模式”迁移到新任务中。
4. **结合 AI Feedback**：将 DEPO 扩展至使用 AI-based preference oracle 的场景（如 AI-assisted RLHF）。
5. **理论扩展至非线性设定**：放松线性假设，发展适用于深度神经网络奖励函数的新型 regret 分析框架。

--- 

> ✅ **总结一句话**：  
> DEPO 提出了一种**基于历史数据覆盖度的、表示空间中的 UCB 探索机制**，解决了在线 RLHF 中因数据稀疏导致的探索低效问题，**在理论和实验上均证明了其优越性**，是迈向高效、自适应语言模型对齐的重要一步。

</details>

---

### 15. [OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](https://arxiv.org/abs/2605.04738)

**Authors**: Zhikai Li, Zhen Dong, Xuewen Liu, Jing Zhang, Qingyi Gu  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.04738v1  

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities. However, their massive parameter scale leads to significant resource consumption and latency during inference. Post-training weight-only quantization offers a promising solution by reducing model size and accelerating token gene...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在推理时面临巨大的**内存消耗和延迟**，尤其是在资源受限或低延迟场景下部署困难。尽管**Post-Training Weight-Only Quantization（PTQ）** 能有效压缩模型并加速推理，但模型权重中普遍存在的**系统性 outliers（异常值）** 严重限制了低比特（如2-bit、3-bit）量化性能。

现有方法（如 GPTQ、AWQ、QuIP）主要依赖**乘法变换**（scaling 或 rotation），通过层间等效变换来缓解 outliers，但这些方法在极低比特下仍表现不佳，说明仅靠乘法范式不足以充分处理 outliers。

---

### 🚀 提出的新方法：OSAQ（Outlier Self-Absorption Quantization）
OSAQ 是一种**基于加法变换**的新型低比特权重量化方法，其核心思想是：

- 利用任务损失函数对某一层权重的**Hessian矩阵具有低秩一致性**（low-rank consistency）这一观察；
- 在 Hessian 的**零空间**（null space）内构造一个**可吸收的加法扰动** $ \Delta W $，使得：
  $$
  \Delta W^T H_W \Delta W = 0
  $$
  即该扰动不会改变任务损失的二阶梯度，从而保证模型性能不变；
- 通过优化 $ \Delta W $ 来**主动抑制权重中的 outliers**，缩小其数值范围，提升量化精度。

#### 🔧 关键技术细节：
- **Null Space Extraction**：通过对近似 Hessian 进行特征分解，提取对应小特征值的特征向量构成零空间。
- **Softmax-∞ Approximation**：将最小化 $ l_\infty $ 范数的目标（即压制最大值）转化为可微的 $ l_2 $-softmax 形式，实现闭式求解。
- **Closed-form Solution**：组合系数可通过正规方程直接解析求解，无需迭代训练或额外微调。

---

### ⭐ 相比现有方法的优势
| 维度 | OSAQ | 传统方法（GPTQ/AWQ/QuIP） |
|------|------|--------------------------|
| 变换类型 | **Additive Transformation**（加法） | Multiplicative Transformation（乘法） |
| 层间依赖 | ❌ 无，独立作用于单层 | ✅ 需要相邻层协同调整 |
| 推理开销 | ❌ 无（离线吸收进权重） | ✅ 通常无，但旋转可能引入额外计算 |
| 是否可叠加 | ✅ 完全兼容现有方法，作为插件增强 | —— |
| 求解方式 | ✅ 闭式解，高效快速 | GPTQ需迭代补偿，AWQ/QuIP需搜索缩放因子或旋转矩阵 |

> ✅ **OSAQ 提供了一种全新的“加法”视角来处理 outliers，是对现有“乘法”范式的有力补充。**

---

## 2. 核心实验方法和设置

### 📚 数据集
- **校准数据集（Calibration）**：用于估计 Hessian 和构建 $ \Delta W $
  - `WikiText2`（128样本，seq_len=2048）
  - `C4`（部分实验也使用）
- **评估数据集**：
  - **语言建模**：`WikiText2`, `C4` → 使用 **Perplexity（↓越低越好）**
  - **常识问答**：`PIQA`, `ARC-e/c`, `WinoGrande` → 使用 **Zero-shot Accuracy（↑越高越好）**
  - **综合能力评测**：`MMLU`, `MT-Bench`

### 🧪 实验设置
- **模型**：
  - 主要测试：`LLaMA2`（7B, 13B, 70B）、`LLaMA3`（8B, 70B）
  - 大规模指令模型：`Mistral-Large-123B-Instruct`, `Llama-3.1-405B-Instruct`
- **量化配置**：
  - W4A16 / W3A16 / W2A16：分别表示 4-bit / 3-bit / 2-bit 权重 + FP16 激活
  - 分组大小 `g=128`
- **评估指标**：
  - Perplexity（语言建模）
  - Zero-shot Accuracy（QA任务）
  - Inference Latency（每token生成时间）

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| RTN | Round-To-Nearest | 最基础 baseline |
| GPTQ | Error Compensation | 基于 Hessian 的逐层误差补偿 |
| AWQ | Scaling | 激活感知的权重缩放 |
| QuIP | Rotation | 正交旋转平滑权重分布 |
| MagR | Magnitude Reduction | 迭代优化无限范数 |
| OmniQuant | Calibration | 多方向校准 |
| WKVQuant | KV-Cache Quant | 同时量化 KV 缓存 |

> OSAQ 以 **plug-and-play 方式集成到上述方法前段**，例如 `OSAQ+GPTQ` 表示先应用 OSAQ 抑制 outliers，再进行 GPTQ 量化。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1–4）

#### 🔹 2-bit 量化性能飞跃（Table 1）
| 方法 | LLaMA2-7B (WikiText2) | 相对提升 |
|------|------------------------|---------|
| GPTQ | 36.8 | —— |
| **OSAQ+GPTQ** | **21.2** | ↓42.4% |
| **OSAQ+GPTQt**（带坐标下降） | **10.6** | ↓71.2% |

> ✅ **在 2-bit 下，OSAQ+GPTQ 相比 vanilla GPTQ 实现超过 40% 的 perplexity 下降！**

#### 🔹 3-bit 量化显著增益
- 在 `LLaMA2-13B` 上：
  - GPTQ: 6.44 (C4)
  - **OSAQ+GPTQ**: **5.72** → ↓11.2%
- 在 `LLaMA3-8B` 上：
  - GPTQ: 8.20 (WikiText2)
  - **OSAQ+GPTQ**: **7.98** → 明显改善

#### 🔹 常识问答任务（Table 2）
| 模型 | 方法 | Avg Accuracy (3-bit) |
|------|------|-----------------------|
| LLaMA3-8B | GPTQ | 63.6% |
| LLaMA3-8B | **OSAQ+GPTQ** | **65.2%** ↑1.6pp |
| LLaMA3-70B | GPTQ | 72.4% |
| LLaMA3-70B | **OSAQ+GPTQ** | **74.4%** ↑2.0pp |

#### 🔹 MMLU 综合能力（Table 3）
| 模型 | 方法 | Avg Score |
|------|------|-----------|
| LLaMA2-7B | GPTQ | 38.7% |
| LLaMA2-7B | **OSAQ+GPTQ** | **39.6%** ↑0.9pp |
| LLaMA3-8B | GPTQ | 57.6% |
| LLaMA3-8B | **OSAQ+GPTQ** | **58.2%** ↑0.6pp |

#### 🔹 百亿级大模型验证（Table 4）
| 模型 | 方法 | MMLU Avg |
|------|------|----------|
| Llama-3.1-405B-Instruct | GPTQ | 85.7% |
| Llama-3.1-405B-Instruct | **OSAQ+GPTQ** | **86.1%** ↑0.4pp |

> ✅ 即使在 **405B 超大规模模型上，OSAQ 依然有效！**

---

### 🔍 消融实验结果

#### （1）加法变换对原始 FP16 模型影响极小（Table 5）
| 模型 | 方法 | WikiText2 PPL |
|------|------|--------------|
| LLaMA2-7B | Baseline | 5.47 |
| LLaMA2-7B | Baseline + Additive ΔW | 5.52 |
| LLaMA2-7B | GPTQ | 8.37 |
| LLaMA2-7B | GPTQ + OSAQ | **6.75** |

> ✅ 加法扰动本身几乎不影响原始模型性能，但能极大提升量化后效果。

#### （2）Softmax-∞ 比直接优化 $ l_2 $ 更有效（Table 7）
| 方法 | WikiText2 PPL |
|------|---------------|
| $ l_2 $ norm only | 7.82 |
| **Softmax-∞ + $ l_2 $** | **6.75** |

> ✅ Softmax-∞ 成功逼近 $ l_\infty $，更聚焦于压制 outliers。

#### （3）Null Space 稳定性强（Table 6 & 11）
- 不同输入 batch 得到的 null space 子空间夹角余弦值高达 **>0.96**，表明其高度一致。
- 即使在不同数据分布（WikiText2 vs C4）或不同 calib size（64~1024）下，null space 保持稳定。

#### （4）超参数鲁棒性好（Figure 5）
- 对 $ \gamma $（尾部能量阈值）、$ T $（softmax温度）、$ \mu_1,\mu_2 $（正则项）不敏感，性能稳定。

#### （5）推理速度无损失（Table 14）
- OSAQ 在量化阶段仅增加少量时间（如 LLaMA2-7B: 22→24分钟），**推理阶段完全无开销**。
- W4A16 下平均提速 **1.9×~2.4×**，且 OSAQ 不影响加速比。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Hessian 的零空间具有跨输入的一致性**，为构造 loss-invariant 的加法扰动提供了理论基础。
2. **Additive Transformation 是一种全新且有效的 outlier suppression 范式**，与传统的 scaling/rotation 方法正交互补。
3. **OSAQ 可作为通用插件，显著提升各类 PTQ 方法（尤其是 GPTQ）在低比特下的表现**，尤其在 2-bit 场景下优势巨大。
4. **该方法无需训练、无推理开销、支持闭式求解，实用性强**，适合工业部署。

---

### ⚠️ 方法的局限性
1. **依赖 Hessian 估计质量**：虽然使用少量样本即可获得稳定 null space，但在极小数据或噪声较大时可能不稳定。
2. **主要针对 weight outliers**：若 activation outliers 成为主要瓶颈（如 full quantization），需结合其他方法（如 QuaRot）共同使用。
3. **目前仅应用于 dense layer**：对于稀疏结构或特殊模块（如 LoRA adapter）尚未验证。

---

### 🔮 未来工作方向
1. **扩展至 activation quantization**：探索是否可在激活端构造类似的 loss-invariant 加法修正。
2. **动态自适应 null space**：根据不同输入动态选择最优扰动方向。
3. **与其他 PTQ 技术深度融合**：如与 rotation 结合形成 hybrid transformation pipeline。
4. **硬件友好设计**：进一步优化 $ \Delta W $ 的稀疏性或结构化形式，便于芯片部署。

---

## ✅ 总结一句话
> **OSAQ 提出了一种基于 Hessian 零空间的加法型 outlier suppression 新范式，实现了无需训练、无推理开销、可插拔的高性能低比特量化，在 2-bit 场景下相较 GPTQ 取得超过 40% 的 perplexity 下降，是 LLM 量化领域的重要进展。**

</details>

---

### 16. [KernelBench-X: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels](https://arxiv.org/abs/2605.04956)

**Authors**: Han Wang, Jintao Zhang, Kai Jiang, Haoxu Wang, Jianfei Chen, Jun Zhu  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.04956v1  

#### Abstract
LLM-based Triton kernel generation has attracted significant interest, yet a fundamental empirical question remains unanswered: where does this capability break down, and why? We present KernelBench-X, a benchmark designed to answer this question through category-aware evaluation of correctness and ...

---

### 17. [Rethinking Local Learning: A Cheaper and Faster Recipe for LLM Post-Training](https://arxiv.org/abs/2605.04913)

**Authors**: Hengyu Shi, Tianyang Han, Peizhe Wang, Zhiling Wang, Xu Yang, Junhao Su  
**Category**: cs.CL  
**Published**: 2026-05-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.04913v1  

#### Abstract
LLM post-training typically propagates task gradients through the full depth of the model. Although this end-to-end structure is simple and general, it couples task adaptation to full-depth activation storage, long-range backward dependencies and direct task-gradient access to pretrained representat...

---

### 18. [Quantile-Free Uncertainty Quantification in Graph Neural Networks](https://arxiv.org/abs/2605.04847)

**Authors**: Soyoung park, Hwanjun Song, Sungsu Lim  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.04847v1  

#### Abstract
Uncertainty quantification (UQ) in graph neural networks (GNNs) is crucial in high-stakes domains but remains a significant challenge. In graph settings, message passing often relies on strong assumptions such as exchangeability, which are rarely satisfied in practice. Moreover, achieving reliable U...

---

### 19. [Federated Learning for Early Prediction of EV Charging Demand](https://arxiv.org/abs/2605.04993)

**Authors**: Vasilis Perifanis, Foteini Nikolaidou, Nikolaos Pavlidis, Panagiotis Thomakos, Andreas Sendros  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.04993v1  

#### Abstract
Accurate forecasting of electric vehicle (EV) charging demand is critical for grid stability, infrastructure planning, and real-time charging optimization. In this work, we study the problem of early prediction of charging demand, where the total energy of a session is estimated using only informati...

---

### 20. [OracleProto: A Reproducible Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking](https://arxiv.org/abs/2605.03762)

**Authors**: Yiding Ma, Chengyun Ruan, Kaibo Huang, Zhongliang Yang, Linna Zhou  
**Category**: cs.AI  
**Published**: 2026-05-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.03762v1  

#### Abstract
Large language models are moving from static text generators toward real-world decision-support systems, where forecasting is a composite capability that links information gathering, evidence integration, situational judgment, and action-oriented decision making. This capability is in broad demand a...

---

### 21. [Quantifying the human visual exposome with vision language models](https://arxiv.org/abs/2605.03863)

**Authors**: Christian Rominger (University of Graz), Andreas R. Schwerdtfeger (University of Graz), Malay Gaherwar Singh (TU Dresden), Dimitri Khudyakow (TU Dresden), Elizabeth A. M. Michels (TU Dresden), Fabian Wolf (TU Dresden), Jakob Nikolas Kather (TU Dresden, University Hospital Carl Gustav Carus Dresden, National Center for Tumor Diseases Heidelberg), Magdalena Katharina Wekenborg (TU Dresden)  
**Category**: cs.AI  
**Published**: 2026-05-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.03863v1  

#### Abstract
The visual environment is a fundamental yet unquantified determinant of mental health. While the concept of the environmental exposome is well established, current methods rely on coarse geospatial proxies or biased self reports, failing to capture the first person visual context of daily life. We a...

---

### 22. [Elicitation Matters: How Prompts and Query Protocols Shape LLM Surrogates under Sparse Observations](https://arxiv.org/abs/2605.04764)

**Authors**: Ge Lei, Samuel J. Cooper  
**Category**: cs.CL  
**Published**: 2026-05-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.04764v1  

#### Abstract
Large language models are increasingly used as surrogate models for low-data optimization, but their optimizer-facing prediction and its uncertainty remain poorly understood. We study the surrogate belief elicited from an LLM under sparse observations, showing that it depends strongly on prompt text...

---

### 23. [EdgeRazor: A Lightweight Framework for Large Language Models via Mixed-Precision Quantization-Aware Distillation](https://arxiv.org/abs/2605.04062)

**Authors**: Shu-Hao Zhang, Le-Tong Huang, Xiang-Sheng Deng, Xin-Yi Zou, Chen Wu, Nan Li, Shao-Qun Zhang  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.04062v1  

#### Abstract
Recent years have witnessed an increasing interest in deploying LLMs on resource-constrained devices, among which quantization has emerged as a promising lightweight technique that converts full-precision model weights and activations into lower-bit formats. Existing weight quantization approaches c...

---

### 24. [Layerwise LQR for Geometry-Aware Optimization of Deep Networks](https://arxiv.org/abs/2605.04230)

**Authors**: Simon Dufort-Labb\'e, Pierre-Luc Bacon, Razvan Pascanu, Simon Lacoste-Julien, Aristide Baratin  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.04230v1  

#### Abstract
Geometry-aware optimizers such as Newton and natural gradient can improve conditioning in deep learning, but scalable variants such as K-FAC, Shampoo, and related preconditioners usually impose structural approximations early, often discarding cross-layer interactions induced by the network computat...

---

### 25. [Counter-Dyna: Data-Efficient RL-Based HVAC Control using Counterfactual Building Models](https://arxiv.org/abs/2605.04555)

**Authors**: Jan Marco Ruiz de Vargas, Fabian Raisch, Zoltan Nagy, Pierre Pinson, Christoph Goebel  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.04555v1  

#### Abstract
Model-based reinforcement learning (MBRL) offers a promising approach for data-efficient energy management in buildings, combining the strengths of predictive modeling and reinforcement learning. While previous MBRL methods applied to HVAC control have reduced training data requirements, they still ...

---

### 26. [Self-Improvement for Fast, High-Quality Plan Generation](https://arxiv.org/abs/2605.03625)

**Authors**: Robert Gieselmann, Henrike von Huelsen, Mihai Samson, Marie-Christine Meyer, Dariusz Piotrowski, Oleksandr Radomskyi, Justin Okamoto, Turan Gojayev, Michael Painter, Gavin Brown, Federico Pecora, Jeremy L. Wyatt  
**Category**: cs.AI  
**Published**: 2026-05-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.03625v1  

#### Abstract
Generative models trained on synthetic plan data are a promising approach to generalized planning. Recent work has focused on finding any valid plan, rather than a high-quality solution. We address the challenge of producing high-quality plans, a computationally hard problem, in sub-exponential time...

---

### 27. [QKVShare: Quantized KV-Cache Handoff for Multi-Agent On-Device LLMs](https://arxiv.org/abs/2605.03884)

**Authors**: Pratik Honavar, Tejpratap GVSL  
**Category**: cs.AI  
**Published**: 2026-05-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.03884v1  

#### Abstract
Multi-agent LLM systems on edge devices need to hand off latent context efficiently, but the practical choices today are expensive re-prefill or full-precision KV transfer. We study QKVShare, a framework for quantized KV-cache handoff between agents that combines token-level mixed-precision allocati...

---

### 28. [Delay-Aware Large-Small Model Collaboration over LEO Satellite Networks](https://arxiv.org/abs/2605.04565)

**Authors**: Mingyu Guo, Wen Wu, Ying Wang, Songge Zhang, Liang Li  
**Category**: cs.DC  
**Published**: 2026-05-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.04565v1  

#### Abstract
In this paper, we introduce a delay-aware largesmall model collaboration scheme for low Earth orbit (LEO) satellite networks, which can balance the computational load among satellites and the communication load across inter-satellite links. Specifically, computational resource constrained remote sen...

---

### 29. [Model synthesis and identifiability analysis of stiff chemical reaction systems with inVAErt networks](https://arxiv.org/abs/2605.04134)

**Authors**: Sreejata Dey, Guoxiang Grayson Tong, Jonathan F. MacArt, Daniele E. Schiavazzi  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.04134v1  

#### Abstract
We consider the problem of learning data-driven replicas for stiff systems of ordinary differential equations arising in chemical kinetics that can be evaluated with high computational efficiency. We first focus on training emulators for families of reaction equations under varying reaction rates, u...

---

### 30. [Deep Wave Network for Modeling Multi-Scale Physical Dynamics](https://arxiv.org/abs/2605.04198)

**Authors**: Alexander I. Khrabry, Edward A. Startsev, Andrew T. Powis, Igor D. Kaganovich  
**Category**: cs.LG  
**Published**: 2026-05-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.04198v1  

#### Abstract
Performance of deep learning models is strongly governed by architectural capacity, with width and depth as primary controls. However, in physical-science applications, models are often compared at a single fixed size or by separating accuracy and computational cost, which can be misleading since ar...

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

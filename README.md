# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-02 09:57:45 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DREAM-S: Speculative Decoding with Searchable Drafting and Target-Aware Refinement for Multimodal Generation](https://arxiv.org/abs/2606.00535)

**Authors**: Zining Liu, Yunhai Hu, Tianhua Xia, Bo Bao, Eric Sather, Vithursan Thangarasa, Sai Qian Zhang  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.00535v1  

#### Abstract
Speculative decoding (SD) has proven to be an effective technique for accelerating autoregressive generation in large language models (LLMs) however, its application to vision-language models (VLMs) remains relatively unexplored. We propose~\textit{DREAM-S}, a novel SD framework designed specificall...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# DREAM-S: Speculative Decoding with Searchable Drafting and Target-Aware Refinement for Multimodal Generation  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **视觉语言模型 (VLMs)** 在推理过程中面临巨大的计算开销，尤其是在自回归生成阶段，由于需要反复处理高维视觉输入并维护庞大的 KV Cache，导致解码速度缓慢。
- 现有的 **Speculative Decoding (SD)** 技术在大语言模型 (LLMs) 中已证明有效，但在多模态场景下的应用仍处于探索阶段，缺乏针对 VLM 特性的系统优化。

### 🚀 提出的新方法：DREAM-S
DREAM-S 是一种专为 VLM 设计的新型 **Speculative Decoding 框架**，其核心思想是通过可搜索的草稿建模与目标感知精炼机制实现高效解码。

#### 主要创新点：
1. **基于 NAS 的可搜索草稿架构设计 (Searchable Drafting via NAS)**
   - 引入 **Neural Architecture Search (NAS)** 框架，在训练超网 (supernet) 后自动搜索最优的草稿模型配置。
   - 搜索维度包括：
     - 注意力头剪枝 (Attention Head Pruning)
     - 视觉 token 压缩比例 (Visual Token Compression)
     - 草稿与目标模型之间的连接策略（特征注入层选择）
   - 所有配置均针对底层硬件平台进行定制化优化，提升实际部署效率。

2. **目标感知蒸馏 (Target-Aware Distillation)**
   - 利用目标模型中间层的隐藏状态作为监督信号，指导草稿模型训练。
   - 提出 **Adaptive Intermediate Feature Distillation (AIFD)**，动态选择语义丰富且稳定的中间层特征进行蒸馏。
   - 依据注意力熵 (attention entropy) 和跨层变化量 (ΔAE) 综合判断最佳蒸馏层，确保监督信号既具信息量又稳定。

3. **两阶段渐进式训练 (Two-Phase Progressive Training, TPPT)**
   - 第一阶段：对完整草稿模型进行 warm-up 训练。
   - 第二阶段：采用 OFA 风格的多分辨率训练，动态采样子网络结构，增强鲁棒性和泛化能力。

4. **交叉注意力融合机制**
   - 在草稿模型中引入 cross-attention 层，将目标模型提取的中间特征作为 Key/Value 输入，提升预测准确性。

### 🔍 相比现有方法的优势
| 方面 | DREAM-S 的优势 |
|------|----------------|
| **加速效果** | 显著优于 SPD、Medusa、Hydra、EAGLE 系列等主流 SD 方法，最高达 **3.85× 加速比** |
| **硬件适应性** | 支持不同 GPU 架构（如 RTX8000, A100, H100），能根据硬件条件自动调整草稿配置 |
| **训练有效性** | 通过联合优化架构与蒸馏目标，实现更优的速度-精度权衡 |
| **通用性** | 在多种规模的 VLM 上表现一致优异（从 2B 到 13B 参数） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **训练数据**：
  - `LLaVA-mix665k` 数据集（55,000 样本）
  - 各评测基准中的 1,000 个非测试样本用于领域适配
- **评估数据集**（共6个）：
  - MMT-Bench
  - SEED-Bench-2
  - ScienceQA
  - OCRBench
  - ChartQA
  - MathVista

### ⚙️ 实验设置
- **目标模型冻结**：所有 VLM 的主干参数保持冻结，仅训练草稿模型。
- **草稿模型结构**：包含 3 个 decoder 层的小型 Transformer。
- **训练细节**：
  - 使用 AdamW 优化器（lr=3e-5, β₁=0.9, β₂=0.95）
  - 总迭代次数：68,000
  - 梯度裁剪值：0.5
  - 单卡 A100 80GB 运行
- **NAS 搜索方式**：离线穷举搜索所有候选子网络，选取在目标硬件上速度最快的配置。

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Speedup Ratio (S)** | $ t_{AR} / t_{method} $，即标准自回归耗时除以当前方法耗时 |
| **Average Accepted Token Length (T)** | 每次验证阶段平均被接受的连续草稿 token 数量 |

### 🆚 对比的 Baseline 方法
- SPD (Gagrani et al., 2024)
- Kangaroo (Liu et al., 2024a)
- Medusa (Cai et al., 2024)
- Hydra (Ankner et al., 2024)
- EAGLE / EAGLE-2 / EAGLE-3 (Li et al., 2024–2025)
- DREAM (Hu et al., 2025b)
- ViSpec (Kang et al., 2025) — 并发工作的 VLM 专用 SD 方法

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2）

| 模型 | 方法 | 平均 Speedup (S) | 平均 Acceptance Length (T) |
|------|------|------------------|----------------------------|
| LLaVA-7B | DREAM-S | **2.35×** | 5.46 |
| LLaVA-13B | DREAM-S | **3.16×** | 4.82 |
| Pixtral-12B | DREAM-S | **2.69×** | 3.68 |
| SmolVLM-2B | DREAM-S | **2.32×** | 3.05 |

> 💡 最高单任务加速比达到 **3.85×**（LLaVA-13B + MMT-Bench）

### 🔁 与基线方法对比
- **相比 EAGLE-3**：
  - 在所有模型上均取得更高加速比：
    - LLaVA-7B: ↑10% (2.35× vs 2.13×)
    - LLaVA-13B: ↑9% (3.16× vs 2.89×)
    - Pixtral-12B: ↑10% (2.69× vs 2.45×)
- **相比 DREAM**：
  - 持续领先 2–5%，说明 NAS + 自适应蒸馏带来实质性增益
- **相比 ViSpec**（Table 3）：
  - 在相同训练条件下，DREAM-S 平均加速比高出 **12.7%~13.8%**

### 🔍 消融实验结果（Ablation Studies）

#### （1）NAS 搜索维度的影响（Figure 3b）
| 设置 | Speedup | T |
|------|--------|----|
| 完整 DREAM-S | **2.67×** | 6.27 |
| 移除 Head Pruning 搜索 | 2.62× | ↑6.44 |
| 移除 Visual Pruning 搜索 | 2.51× | ↑6.50 |
| 固定特征注入层 | 2.48× | ↓ |

> ✅ 结论：三个搜索维度均有贡献，尤其是自适应特征注入至关重要。

#### （2）AIFD 蒸馏策略的有效性（Figure 3c）
| 蒸馏方式 | Speedup |
|---------|--------|
| 无中间监督（No Mid） | 最低 |
| 静态深度 25%/50%/75% | 逐步提升 |
| **AIFD（自适应选择）** | **最高** |

> ✅ 动态选择基于 attention entropy 的中间层显著优于固定策略。

#### （3）λ 权重影响（Figure 3d）
- 当 `λ_feat = λ_distill = 0.2`, `λ_KL = 1.0` 时达到最优平衡
- 过高的特征监督权重会损害泛化能力

#### （4）零样本泛化能力（Table 7）
- 即使不使用任何评测集数据训练，DREAM-S 仍以 **1.72×** 平均加速比领先于 EAGLE-3 (1.59×) 和 DREAM (1.65×)，表明其优势源于架构而非过拟合。

#### （5）固定配置 vs 可搜索配置（Table 8）
- 即便使用统一草稿配置（DREAM-S Fixed），依然全面超越所有 baseline
- 自适应 NAS 在 OCRBench、ChartQA 等细粒度任务上带来更多收益

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DREAM-S 是目前最先进的 VLM 推理加速框架之一**，在多个主流 VLM 上实现了高达 **3.85× 的端到端加速**。
2. **NAS + 目标感知蒸馏** 的组合能有效解决 VLM 解码中的“准确率-延迟”权衡难题。
3. **硬件感知的草稿模型搜索机制** 使得 DREAM-S 能够灵活适配不同设备，具备良好的工程实用性。
4. 更大的目标模型（如 13B）从 DREAM-S 中获益更多，说明其在重负载场景下潜力巨大。
5. 方法在确定性解码（T=0）下表现最佳，但在随机采样（T=1）下也保持稳健。

### ⚠️ 局限性
1. **系统级创新为主**：未提出全新的 NAS 或解码理论，而是构建了一个高效的多模态搜索空间。
2. **搜索空间有限**：当前采用穷举搜索，若扩展至更复杂的架构（如 MoE-based VLMs），需引入预测器辅助搜索（如 OFA-style predictor）。
3. **视觉压缩可能损失细节**：在 OCRBench 等依赖像素级识别的任务上增益较小，提示未来需结合局部保留机制。

### 🔮 未来工作方向
- 将 DREAM-S 扩展至 **MoE 架构的 VLMs**，重新设计专家路由与剪枝协同机制。
- 探索 **动态 draft window size 调整策略**，根据上下文复杂度自适应调节草稿长度。
- 引入 **predictor-based NAS 搜索器**，应对更大规模的架构空间。
- 结合 **long-context modeling**，研究如何在长序列生成中维持高接受率。

---

> 🔗 **代码开源地址**：[https://github.com/SAI-Lab-NYU/DREAM-S](https://github.com/SAI-Lab-NYU/DREAM-S)

</details>

---

### 2. [Scaling LLM Inference Beyond Amdahl`s Limits via Eliminating Non-Scalable Overheads](https://arxiv.org/abs/2606.01927)

**Authors**: Alan Zhao, Cyril Y. He, Wei Xu  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2606.01927v1  

#### Abstract
Deployers of online LLM services usually seek to maximize cluster-wide performance given a fixed number of GPUs. Tensor parallelism (TP) is necessary to fit modern models but scales sub-linearly as the TP degree t grows, due to cross-GPU communication and non-scalable runtime work, as predicted by A...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Scaling LLM Inference Beyond Amdahl's Limits via Eliminating Non-Scalable Overheads*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代大型语言模型（LLM）推理受限于 **Amdahl's Law** 所描述的非可扩展瓶颈：尽管采用 **Tensor Parallelism (TP)** 可以将大模型分布到多个 GPU 上运行，但由于跨 GPU 通信开销以及调度、采样等任务无法并行化，导致随着 TP 度 $ t $ 增加，吞吐提升呈现亚线性甚至下降趋势。

此外，内存压力（尤其是 KV Cache 占用）也限制了并发请求处理能力。因此，在固定 GPU 资源下如何最大化集群吞吐成为关键挑战。

作者发现存在一个经验最优 TP 度 $ t_e $，在该点上通信开销与内存效率达到平衡。然而，现有系统（如 vLLM）由于未能消除非可扩展部分，导致实际可达的 $ t_e $ 远低于理论潜力。

---

### 提出了什么新方法或新思路
本文提出 **Albireo**，一种新型 LLM 推理引擎，通过以下三大优化突破 Amdahl’s Law 的限制：

#### ✅ **Optimization 1: Optimistic Asynchronous Scheduling（乐观异步调度）**
- 引入“迭代依赖序列管理”机制，为每个 token 维护虚拟状态（Expected Length, Current Length, NNT），实现资源需求预测。
- 采用**单轮乐观预测**策略：假设所有序列将继续生成下一个 token，提前分配 KV Cache 块，从而允许下一迭代调度在当前迭代完成前启动。
- 显著减少 CPU 调度阻塞时间（从 ~4ms 降至 ~5μs）。

#### ✅ **Optimization 2: Early-feedback Backfill（早期反馈回填）**
- 在采样完成后立即向输入/输出处理器回传 token ID，打破 T5 → T2 的数据依赖链。
- 允许输入处理器提前构建大部分模型输入张量（仅占位符等待最后 token），实现 CPU 与 GPU 任务重叠执行。

#### ✅ **Optimization 3: Sequence-Parallel Sampling（序列并行采样）**
- 将原本集中在 driver GPU 上的采样任务按 batch 维度拆分至所有 TP worker。
- 利用 `all-to-all` 操作交换 logits，并行执行采样，再由 driver 收集结果。
- 配合 deterministic RNG 和 batch padding 技术保证输出一致性且无显著通信开销。

这些设计共同实现了：
- 更高的有效 TP 度 $ t_e $
- 更大的聚合吞吐
- 更低的延迟和能耗

---

### 相比现有方法的优势
| 方面 | Albireo vs. vLLM/SGLang |
|------|--------------------------|
| **架构改动** | 不改变模型结构，作为插件集成（仅需 ~200 行代码修改） |
| **并行粒度** | 实现 T1/T2/T4/T5 全流程可重叠或并行，而现有系统仅 T3 可扩展 |
| **scalability** | 显著缩小非可扩展部分占比（减少 89% overhead） |
| **实用性** | 支持主流采样策略（top-k, top-p, temperature 等） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Databricks-dolly-15k** 数据集中的随机 prompt 用于生成测试 workload。
- 生产环境部署基于真实用户请求流量（未公开细节）。

---

### 实验设置
#### 测试平台（Testbed）
| 平台 | GPU 配置 |
|------|---------|
| **H100M** | 8× H100（80GB），NVLink 连接 |
| **A100M** | 8× A100（80GB），NVLink 连接 |
| **A100N** | 8× A100（80GB），PCIe 连接 |

所有节点均配备 Intel Xeon Platinum 8468 CPU（192 核）、2TB RAM。

#### 模型范围（FP16 精度）
共 8 个主流 LLM，分为四类：
- **Tiny**: Llama-2-7B, Qwen-2.5-7B
- **Small**: Llama-2-13B, Qwen-2.5-14B
- **Moderate**: Qwen-2.5-32B, QwQ-32B
- **Large**: Llama-3.1-70B, Qwen-2.5-72B

#### 配置参数
- 默认 per-GPU batch size = 32（总 batch size = 128 当 $ t=4 $）
- TP degree $ t $ 按公式估算最优值：  
  $$
  t_e = \frac{4M}{C}
  $$
  其中 $ M $ 是模型权重大小，$ C $ 是每卡显存容量。
- Pipeline Parallelism 禁用（聚焦单节点内推理）

---

### 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput** | 每秒生成 token 数（token/s） |
| **Latency** | 平均每 token 生成时间（TPOT），含尾部延迟（99%, 99.9%） |
| **GPU Utilization** | GPU 利用率（%） |
| **Power Usage / Energy** | 实时功耗（W）与总能耗（J） |
| **TPOT Distribution** | 解码阶段每 token 时间分布 |

---

### 基线方法对比
- **vLLM (v0.11.2)**：当前最先进的开源 LLM 推理引擎，支持 PagedAttention
- **SGLang (v0.5.5)**：支持结构化生成与异步调度的新一代推理框架

> 注：消融实验仅对比 vLLM，因 Albireo 是其插件形式实现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（H100M 测试床，默认配置）

| 模型类别 | 吞吐提升（vs. vLLM） | 最高提升 |
|--------|------------------|--------|
| Tiny LLMs ($t=1$) | ~1.3× | — |
| Small LLMs ($t=2$) | ~1.5× | — |
| Moderate LLMs ($t=4$) | ~1.7× | — |
| **Large LLMs ($t=8$)** | **~1.9×** | **最高达 2×** |

> 在生产环境中对 Llama-3.1-70B 达成 **2× 吞吐提升**

---

### 与基线方法的对比结果

| 性能维度 | 提升幅度 |
|--------|--------|
| **Throughput** | 最高 **1.9×**（实验室），生产环境达 **2×** |
| **Latency (TPOT)** | 平均降低 **22%–48%**，尾部延迟改善显著（如 QwQ-32B 99.9% TPOT 从 298ms → 76ms） |
| **GPU Utilization** | 平均提升 **28%**（Qwen-2.5-32B） |
| **Energy Efficiency** | 能耗降低 **54%**（推理时间缩短 47%，平均功耗降 15%） |

#### 图表关键观察：
- **图10**：Albireo 在 $ t \leq t_e $ 区间表现出**超线性扩展**（$ E_{2n} > 2 \times E_n $），说明其有效释放了内存红利。
- **图12**：负载越高，性能增益越明显（轻载时 ~1.7×，重载稳定在 2×）。
- **图13**：尾部延迟大幅压缩，得益于 CPU-GPU 任务重叠隐藏了事件循环抖动。

---

### 消融实验结果（Ablation Study）

#### （1）各组件贡献分析（图15）
| 模型规模 | 主要收益来源 |
|--------|------------|
| Tiny & Small | 几乎全部来自 **Asynchronous Execution** |
| Moderate & Large | **Asynchronous + Parallel Sampling** 贡献相当 |

> 说明：对于 $ t > 1 $ 的大模型，序列并行采样成为关键加速器。

#### （2）任务级开销削减（表1，Qwen-2.5-32B, $t=4$）

| Task | vLLM 时间 | Albireo 时间 | 缩减比例 |
|------|----------|-------------|--------|
| T1 (Scheduling) | ~4ms | ~5μs | >99.8% |
| T2 (Input Proc.) | ~4ms | ~40μs | >99% |
| T4 (Sampling) | ~6ms | ~1.5ms | 75% |
| T5 (Output Proc.) | ~0.5ms | ~25μs | >95% |
| **Non-scalable Total** | ~14.5ms | ~1.6ms | ↓ **89%** |

> 结论：两大机制联合消除超过 89% 的非可扩展开销。

#### （3）其他验证
- **Rs ratio（metadata scattering / forward time）**：始终 <22%，证明 forward 可完全掩盖 scatter 开销。
- **KV Cache 内存浪费**：最坏情况仅多占用 1 个 block（16 tokens），回收延迟 ≤1 iteration。

---

## 4. 关键结论和发现

### 主要发现
1. **存在经验最优 TP 度 $ t_e $**，受通信与内存竞争权衡影响；
2. **非可扩展部分主导端到端延迟**，尤其在高性能 GPU（如 H100）上更严重；
3. **Albireo 成功将 $ t_e $ 推高一倍**（如 32B 模型从 2→4，70B 从 4→8），突破传统系统瓶颈；
4. **超线性扩展成为可能**：当 $ t \leq t_e $，$ T(t) \geq 2 \times T(t/2) $，极大提升集群利用率；
5. **无需硬件变更即可获得接近翻倍的吞吐**，适用于大规模 MaaS 平台部署。

---

### 方法的局限性
1. **当前聚焦单节点场景**：未解决 multi-node 下的 TP all-reduce 和 PP stage imbalance 问题；
2. **不兼容 ultra-large models requiring PP**：如 Kimi K2（1TB）等跨节点模型不在考虑范围内；
3. **Deterministic RNG 需额外显存**：约 128KB/request，虽换得通信节省但仍有一定代价；
4. **对极小 batch 或短序列增益有限**：此时计算本就不饱和，优化空间较小。

---

### 未来工作方向
1. **扩展至 Hybrid TP-PP 多节点部署**：优化跨 stage 通信 bubble 与负载均衡；
2. **探索更多任务间的并行机会**：如 prefill-decode overlap、speculative decoding 集成；
3. **适配未来硬件趋势**：随着 GPU compute 提速快于 memory bandwidth，非可扩展开销将持续增长，亟需更多类似 Albireo 的系统级优化；
4. **支持动态批处理与弹性扩容**：结合 Llumnix 类动态调度器，进一步提升云服务 SLA 满足率。

---

> ✅ **一句话总结**：  
> Albireo 通过消除调度、I/O 与采样的非可扩展开销，首次实现 LLM 推理突破 Amdahl’s Law 极限，在不改模型的前提下达成高达 **2× 吞吐、48% 低延迟、54% 节能**，是迈向高效绿色 AI 推理的重要一步。

</details>

---

### 3. [ViBE: Co-Optimizing Workload Skew and Hardware Variability for MoE Serving](https://arxiv.org/abs/2606.00735)

**Authors**: Seokjin Go, Marko Scrbak, Ephrem Wu, Srilatha Manne, Divya Mahajan  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.00735v1  

#### Abstract
In distributed Mixture-of-Experts (MoE) inference, input-dependent token routing interacts with GPU performance variability to create persistent stragglers under synchronized execution, where the slowest GPU determines layer latency. This performance variability is inherent to modern accelerators: m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ViBE: Co-Optimizing Workload Skew and Hardware Variability for MoE Serving**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在分布式 **Mixture-of-Experts (MoE)** 推理中，**token routing** 是输入依赖的，导致专家负载不均衡（workload skew），而现代 GPU 集群中存在显著的 **硬件性能变异性（hardware variability）** —— 即使是同型号 GPU，也会因制造差异、功耗限制和温度条件不同而导致执行速度差异。

传统方法假设硬件同质，仅优化 **token 分布平衡**，但在同步执行模式下，**最慢的 GPU 决定层延迟（straggler problem）**。因此，即使 token 负载均衡，由于硬件性能差异，仍会出现执行时间不平衡，导致尾延迟高、利用率低。

ViBE 正是为了解决这一 **workload skew 与 hardware variability 共同作用下的持续性 straggler 问题**。

---

### **提出了什么新方法或新思路**

提出 **Variability-Informed Binning of Experts (ViBE)**，一种硬件感知的专家放置框架，其核心思想是：

> **将高负载专家分配给更快的 GPU，低负载专家分配给较慢的 GPU，以对齐各 GPU 的完成时间，而非简单地平衡 token 数量。**

#### **关键创新点：**
- **联合优化 workload skew 和 hardware variability**  
  不再孤立处理路由倾斜或硬件差异，而是将其共同建模，利用硬件变异性作为“杠杆”来抵消负载不均。
  
- **执行时间感知的专家放置（execution-time-aware placement）**  
  基于每个 GPU 的性能模型 $ f(n) $（token 数量 → 延迟）和每层专家激活分布 $ w_e $，计算出最优的专家到 GPU 映射，使得预测的 per-GPU 完成时间尽可能一致。

- **漂移感知的轻量级重校准（drift-aware recalibration）**  
  动态监测 workload drift（如 batch 大小、输入分布变化），仅当 **cosine distance > 阈值** 时触发增量式重排列（incremental update），避免频繁 full shuffle 开销。

---

### **相比现有方法的优势**

| 方法 | 是否硬件感知 | 是否动态更新 | 优化目标 | 局限性 |
|------|---------------|----------------|------------|--------|
| **vLLM (contiguous)** | ❌ | ❌ | 无显式优化 | 负载严重不均 |
| **EPLB** | ❌ | ✅（固定周期） | 平衡 token 数量 | 忽略硬件差异，无法消除 straggler |
| **ViBE (Ours)** | ✅ | ✅（按需触发） | 最小化执行时间差异 | —— |

- **优势总结：**
  - 在相同 token 负载下，ViBE 可减少高达 **19.6% 的 kernel latency gap**；
  - 相比 EPLB，ViBE 将 **P90 TTFT 最多降低 45%**；
  - 通过增量更新机制，重排列开销降低一个数量级以上（通常只需 5–30 次交换，而非 >200 次）；
  - 支持跨 workload drift 场景自适应，维持 SLO 表现稳定。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Sonnet**：固定输入长度（1024 tokens）、固定输出长度（128 tokens），用于测试稳定 workload 下的表现。
- **ShareGPT**：真实用户对话数据，平均输入 219.2 tokens，输出 200.8 tokens，具有高度可变性和突发性，用于模拟生产环境。

---

### **实验设置和评估指标**

#### **硬件平台**
- 单节点 8× **AMD Instinct™ MI325X** GPU（主平台）
- 辅助验证：**AMD Instinct™ MI300X** 及人为引入电压-频率偏移的“skewed”系统
- 软件栈：vLLM v0.14.2, PyTorch v2.9.0, ROCm v7.0, AITER kernel backend

#### **模型配置**
| 模型 | dtype | #Experts | EP Degree | Routed Experts |
|------|-------|----------|-----------|----------------|
| **DeepSeek-V3** | FP8 | 256 | 8 | 8 |
| **Qwen-3-32B** | FP8 | 128 | 8 | 8 |

采用 **Hybrid TP + EP** 架构，MoE 层使用 8-way EP，非 MoE 层使用 8-way TP。

---

### **评估指标**
- **SLO Attainment (%)**：满足延迟 SLO 的请求比例（TTFT ≤ 250ms / 350ms；TPOT ≤ 100ms / 125ms）
- **Goodput**：单位时间内成功完成且符合 SLO 的请求数量
- **Tail Latency**：P90/P99 TTFT 和 TPOT
- **Kernel Time Variance**：MoE kernel 执行时间的最大-最小差值
- **Clock Frequency Spread**：各 GPU 实际运行频率的标准差

---

### **基线方法对比**
| 策略 | 描述 |
|------|------|
| **vLLM** | 默认连续分区（contiguous partitioning） |
| **EPLB** | 基于路由频率的 token 平衡方法，忽略硬件差异 |
| **ViBE** | 本文提出的方法，结合硬件性能建模与动态重校准 |

所有策略均静态部署（除非启用 adaptive mode），确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **SLO Attainment 提升**
- 在 **Sonnet 数据集** 上：
  - ViBE 相比 EPLB 将 **SLO frontier 提升 12–15%**（DeepSeek-V3: +12%，Qwen-3: +15%）
- 在 **ShareGPT 数据集** 上：
  - ViBE 更好应对路由突变，始终保持领先，尤其在高 QPS 区间优势明显

> 图 8 显示，在所有场景下，`vLLM < EPLB < ViBE` 的 SLO 曲线严格递增。

---

#### **尾延迟显著下降**
| 模型 | 数据集 | 指标 | ViBE vs. vLLM | ViBE vs. EPLB |
|------|--------|------|----------------|----------------|
| DeepSeek-V3 | Sonnet | **P90 TTFT** | ↓45% | ↓35% |
| DeepSeek-V3 | Sonnet | **P99 TTFT** | ↓30% | ↓20% |
| Qwen-3 | Sonnet | **P90 TTFT** | ↓10% | ↓8% |
| Qwen-3 | Sonnet | **P99 TTFT** | ↓30% | ↓25% |

> 尾部延迟改善源于更均匀的 per-GPU 完成时间，减少了同步等待。

---

#### **内核级性能提升**
- **MoE kernel latency gap（最快 vs 最慢 GPU）**：
  - EPLB 相比 vLLM 减少 63.9%
  - ViBE 在此基础上再减少 19.6%
- **平均 MoE 延迟**：
  - ViBE 比 vLLM 快 49.3%
  - 比 EPLB 快 27.9%

> 图 10 显示 ViBE 显著压缩了 GPU 间的频率分布差异，表明硬件利用率更加均衡。

---

#### **动态重校准有效性**
- 在 **cross-workload drift 测试（SG→SN / SN→SG）** 中：
  - 静态 ViBE 在 1.68 QPS/GPU 达到 90% SLO
  - 自适应 ViBE 提升至 **1.80 QPS/GPU**
  - 类似地，Adaptive EPLB 从 1.51 → 1.63 QPS/GPU
- 重排列事件仅短暂影响 TTFT（秒级恢复），总体收益远大于代价

> 图 12 显示重排列后系统迅速收敛，服务质量快速恢复。

---

#### **消融实验（Ablation Study）**
- **是否启用 drift-aware recalibration**：
  - 固定间隔更新效果差，易造成资源浪费或响应滞后
  - 基于 cosine distance 的漂移检测能精准捕捉 workload 变化
- **是否使用 per-GPU 性能建模**：
  - 若假设所有 GPU 吞吐一致（即 $ f(n) = n $），则无法补偿硬件差异，性能接近 EPLB
- **是否进行增量更新**：
  - Full re-solve 成本高昂（>200 次迁移）
  - Incremental solver 仅需 5–30 次交换即可恢复平衡，效率提升 10×+

---

## **4. 关键结论和发现**

### **主要发现**
1. **硬件变异性是 MoE 推理中的头等公民（first-order constraint）**  
   即使 token 负载完全均衡，GPU 间仍可出现 **高达 7% 的 kernel 执行时间差异**（图 6 & 图 13），直接影响尾延迟和吞吐。

2. **token 平衡 ≠ 时间平衡**  
   EPLB 成功降低了 token imbalance，但未能解决 latency imbalance，说明 **仅靠算法层面的负载均衡不足以应对硬件异构性**。

3. **ViBE 将硬件变异性从“缺陷”转化为“优化机会”**  
   通过将高负载专家定向到高性能 GPU，实现了 **execution-time co-optimization**，有效抑制 straggler。

4. **scale 越大，硬件感知越重要**  
   投影实验显示，在更大 EP 组（如 16–32 GPUs）中，ViBE 的优势随性能离散度增加而增强；超过 64 GPUs 后灵活性下降，但仍优于基线。

---

### **方法的局限性**
- **依赖离线性能建模**：需预先对每个 GPU 进行 microbenchmark，增加了部署复杂性。
- **增量更新可能引入短暂延迟尖峰**：重排列期间部分请求会经历更高 TTFT。
- **当前未支持非均匀专家分配**：每个 GPU 固定分配相同数量专家，未来可探索弹性分配。
- **局限于 intra-node 优化**：尚未扩展到跨节点或多 rack 场景。

---

### **未来工作方向**
- **支持 speculation 或增量迁移（zero-downtime reshuffle）**：进一步降低重排列开销。
- **扩展至 rack-scale 系统**：结合 topology-aware routing 与 ViBE 放置策略。
- **与 prefill/decode disaggregation 协同优化**：在 Splitwise 等架构中联合调度。
- **探索 selective expert duplication**：为热点专家提供冗余副本，主动规避慢设备。
- **构建通用 hardware variability database**：实现跨集群迁移时的 placement knowledge reuse。

---

> ✅ **总结一句话**：  
> **ViBE 首次将 hardware variability 视为 MoE 推理中可被利用的正向因素，通过 variability-informed expert placement 实现了从“token balance”到“time balance”的范式转变，在多个 MoE 模型和平台上显著提升了 SLO 达成率与尾延迟表现。**

</details>

---

### 4. [Threshold-Based Exclusive Batching for LLM Inference](https://arxiv.org/abs/2606.00516)

**Authors**: Weifang Zhang, Yuzhou Nie, Bowen Pang, Guangrui Ma, Shining Wu  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.00516v1  

#### Abstract
Mixed batching (MB)--interleaving prefill and decode in a single batch--has become the standard scheduling strategy for large language model (LLM) inference due to its efficiency in maximizing compute and memory utilization. However, through controlled experiments, we find that prefill-decode interf...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Threshold-Based Exclusive Batching for LLM Inference》论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对当前主流的 **Mixed Batching (MB)** 调度策略在 **带宽受限** 的GPU上性能不佳的问题进行了深入研究。尽管MB通过交错执行prefill和decode阶段来最大化硬件利用率，但在内存带宽较低的设备上，prefill（计算密集型）和decode（内存带宽密集型）操作在同一batch内共存会引发严重的 **内存带宽争用（bandwidth contention）**，导致边际成本上升，反而降低了吞吐量。

作者指出，现有的“MB优于Exclusive Batching (EB)”的认知并非普适，其优劣取决于硬件带宽、模型大小和工作负载构成。

### 提出的新方法与新思路
1. **理论分析与闭式条件（Closed-form Condition）**  
   首次推导出一个**闭式表达式**，用于判断在何种条件下EB会优于MB。该条件由两部分决定：
   - **边际成本差距（Marginal-cost gap）**：MB中prefill-decode干扰带来的额外开销。
   - **固定成本摊销优势（Amortized fixed-cost advantage）**：MB因更少的核函数调用而具有的优势。
   在高负载下，比较主要由边际成本差距主导。

2. **自适应EB调度器 EB(k\*)**  
   提出了一种基于理论分析的自适应EB调度器：
   - 推导出**渐近最优的相位切换阈值** `θ* = k*/N`，即当有 `k*` 个decode槽空闲时，切换回prefill阶段。
   - 设计了**内存安全的批处理大小（memory-safe batch sizing）**，确保KV Cache不会溢出（OOM），并提供概率保证。
   - 该调度器能在线估计工作负载特征（如输出长度分布的hazard rate），动态调整 `k*` 和 `N*`。

3. **混合调度器 EB+**  
   提出 **EB+**，一种能够在线应用上述交叉条件的**混合调度器**。它无需人工干预，能在EB和MB之间**动态切换**，以适应不同的硬件和非平稳流量（non-stationary traffic），始终选择当前最优的调度模式。

### 相比现有方法的优势
- **超越MB**：在带宽受限的GPU上，优化后的EB可实现高达 **41.9%** 的吞吐量提升。
- **无需专用硬件**：EB+ 的性能可媲美需要物理分离prefill和decode的 **PD-disaggregation** 架构，但无需额外的GPU池或复杂的KV Cache传输机制。
- **鲁棒性强**：EB+ 在面对并发量变化或请求分布漂移等非平稳场景时，依然能保持最高或接近最高的吞吐量。
- **自动化**：整个决策过程是自动化的，消除了手动调参的需求。

---

## 2. 核心实验方法和设置

### 数据集与工作负载
实验涵盖了**合成工作负载**和**真实世界数据集**，以覆盖不同的prefill-decode比例：
- **合成工作负载**：
  - Decode-heavy (128输入/1024输出)
  - Balanced (512/512)
  - Prefill-heavy (1024/128)
- **真实世界数据集**：
  - **ShareGPT**: 对话数据，中等decode比例。
  - **LongBench**: 长上下文任务，prefill-heavy。
  - **NuminaMath**: 数学推理，长链式思考，decode-heavy。
  - **WildChat**: 多轮对话，模拟复杂交互。

### 实验设置
- **硬件平台**：在四种不同内存带宽的GPU上进行评估，形成鲜明对比：
  - **高带宽**：H200 (4.8 TB/s), B300 (8.0 TB/s)
  - **带宽受限**：RTX PRO 6000 (1.792 TB/s), L40S (0.864 TB/s)
- **模型**：
  - 主要模型：Qwen3-8B, Qwen3-30B-A3B (MoE)
  - 补充模型：Gemma-3-1B-IT
- **实现**：基于 **vLLM** 框架进行扩展，所有调度器（v0, v1, EB(k\*), EB+）均在相同代码库下进行公平比较。

### 评估指标
- **吞吐量 (Throughput)**：以 **tokens/s** 和 **requests/s (RPS)** 衡量。
- **延迟 (Latency)**：
  - **TTFT (Time to First Token)**：首token延迟。
  - **TPOT (Time Per Output Token)**：每个输出token的平均时间。
  - **ITL (Inter-Token Latency)**：连续token间的延迟。
- **SLO约束下的有效吞吐 (Goodput)**：同时满足TTFT和TPOT SLO的请求占比。
- **配置敏感性**：通过CV（变异系数）、Range Ratio等指标衡量调度器对超参数（token budget, batch size）的鲁棒性。

### 基线方法对比
- **v0**：vLLM v0版本的 **Exclusive Batching (EB)** 调度器，等价于 `EB(k=1)`。
- **v1**：vLLM v1版本的 **Mixed Batching (MB)** 调度器，作为当前主流方法的代表。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1. **EB(k\*) vs. v1 (MB)**：
   - 在 **RTX PRO 6000 (1.792 TB/s)** 上，EB(k\*) 在Qwen3-8B上的平均RPS增益为 **+7.9%**，在极端情况下（如ShareGPT）可达 **+15.3%**。
   - 在 **L40S (0.864 TB/s)** 上，EB(k\*) 的RPS比v1高出 **41.9%** (14.70 vs. 10.36)，充分证明了其在严重带宽受限环境下的巨大潜力。
   - 在 **H200 (4.8 TB/s)** 上，由于带宽充足，v1通常保持优势或两者性能接近，验证了“带宽决定论”。

2. **EB+ vs. 固定模式 (v1 或 EB(k\*))**：
   - **应对流量变化**：在并发量 `c` 从32到2048变化的测试中，EB+表现出色：
     - 在低负载 (`c=32`) 时，选择MB以获得更低的TTFT。
     - 在高负载 (`c=2048`) 时，切换到EB以获得更高吞吐量，在RTX PRO 6000上比v1高 **50%**。
   - **应对非平稳工作负载**：
     - 在**分布漂移**（decode-heavy -> prefill-heavy）场景下，EB+在RTX PRO 6000上比v1的吞吐量高出 **36.4%**。
     - 在**并发漂移**场景下，也取得了显著增益（+22.6%）。

3. **与PD-disaggregation的对比**：
   - 在2-GPU和4-GPU设置下，EB+在不进行P:D比例调优的情况下，性能达到或超过了需要手动配置的PD-disaggregation方案。
   - 例如，在4×RTX PRO 6000上，EB+在7/9种场景下实现了最佳或次佳吞吐量，并且**TTFT低3-18倍**，且**从未发生OOM**，而disaggregation方案因缺乏背压机制频繁OOM。

### 消融实验结果
- **阈值 `k*` 的有效性**：在H200上使用几何输出长度的实验表明，EB(k\*)的自适应控制器找到的阈值，其吞吐量与通过穷举搜索得到的最佳固定阈值相当，甚至更高（最高提升 **8.0%**），证明了其近似最优性。
- **IFR修正的有效性**：实验证明，现实工作负载具有**递增故障率 (IFR)**，这支持了更高的切换阈值。控制实验显示，忽略IFR会导致次优性能。
- **鲁棒性**：对token budget和batch size的敏感性分析表明，EB(k\*)的鲁棒性与v0和v1相当，没有引入额外的调参负担。

---

## 4. 关键结论和发现

### 主要发现
1. **EB与MB的权衡取决于硬件带宽**：**Mixed Batching (MB) 并非总是最优**。在**带宽受限**的GPU上，**Exclusive Batching (EB)** 通过避免prefill-decode干扰，可以实现显著更高的吞吐量。这一发现解释了为何中国许多生产系统（受出口限制影响，多使用带宽受限GPU）仍偏好EB。
2. **存在明确的性能交叉点**：通过一个**闭式条件**可以精确判断何时应选择EB而非MB，该条件由边际成本差距和固定成本摊销共同决定。
3. **自适应是关键**：提出的 **EB+** 调度器能够在线做出最优决策，在各种静态和动态场景下都达到了最佳性能，证明了自适应调度的巨大价值。
4. **模型大小的影响**：随着模型增大，固定开销增加，使得MB的摊销优势更明显，从而缩小了EB的收益空间。

### 方法的局限性
- **理论假设**：最优阈值的推导依赖于**流体近似 (fluid approximation)** 和对输出长度分布的假设（如CFR/IFR），在小批量或极端分布下可能不是绝对最优。
- **实现复杂性**：虽然EB+是自动化的，但其内部逻辑（在线估计、阈值计算、内存门控）比简单的MB或EB更复杂。
- **侧重吞吐量**：EB倾向于优化TPOT和总吞吐量，但可能会牺牲TTFT。EB+通过在低负载时选择MB来缓解此问题，但本质上仍是一个吞吐量优先的设计。

### 未来工作方向
- **更精细的SLO感知调度**：设计能直接优化SLO（如P99延迟）的切换策略。
- **结合其他并行技术**：将EB+与**sequence parallelism**或**pipeline parallelism**相结合，探索在更大规模集群中的应用。
- **更复杂的干扰建模**：进一步研究MoE模型中专家重载等现象对混合批处理的具体影响，并将其纳入调度决策中。

</details>

---

### 5. [TwinQuant: Learnable Subspace Decomposition for 4-Bit LLM Quantization](https://arxiv.org/abs/2606.01556)

**Authors**: Haodong Wang, Junjie Liu, Zicong Hong, Qianli Liu, Jian Lin, Song Guo, Xu Chen  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.01556v1  

#### Abstract
4-bit quantization reduces the memory footprint and latency of large language model inference, but its aggressive precision reduction can severely degrade accuracy. Prior methods address this by decomposing each weight matrix into two components (e.g., via singular value decomposition) and quantizin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TwinQuant: Learnable Subspace Decomposition for 4-Bit LLM Quantization — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

- **4-bit LLM 量化中的精度严重下降问题**：现有的 4-bit 量化方法在压缩权重和激活时，由于权重矩阵具有重尾分布（heavy-tailed）和高度各向异性（highly anisotropic），导致量化误差集中在少数“异常值”（outliers）上，从而显著降低模型准确率。
- **传统低秩分解方法的局限性**：
  - 基于 SVD 的方法（如 SVDQuant）依赖快速奇异值衰减假设，但在主流 LLM 中该假设不成立（谱衰减缓慢）。
  - 固定分解方式无法适应不同层的统计特性，且直接对低秩分支进行 4-bit 量化会引入巨大误差。

### 🚀 提出的新方法与创新思路

论文提出 **TwinQuant**，一种全新的 4-bit 后训练量化（PTQ）框架，其核心思想是：

#### （1）Learnable Subspace Decomposition（可学习子空间分解）
- 将每个权重矩阵 $ W $ 分解为两个互补分量：一个 **low-rank component** 和一个 **residual component**。
- 不同于固定 SVD 分解，TwinQuant **联合学习**这两个分量所处的子空间，使其更“量化友好”（quantization-friendly）。
- 引入两类可学习变换：
  - **Global Orthogonal Transform $ Q \in \text{Stiefel} $**：作用于输入激活和残差，平滑分布、减少动态范围失衡。
  - **Layer-specific Invertible Transform $ G \in \text{GL}(n,\mathbb{R}) $**：调整低秩因子 $ U, V $ 的内部结构，提升其可量化性。

#### （2）Joint Optimization over Hybrid Manifolds（混合流形上的联合优化）
- 在 **Stiefel 流形**（正交矩阵）和 **General Linear 流形**（可逆矩阵）上联合优化 $ Q $ 和 $ G $。
- 使用 **Cayley SGD** 更新正交变换 $ Q $，保证数值稳定性。
- 对 $ G $ 采用极分解参数化 $ G = PS $（$ P $ 正交，$ S $ 对称正定），确保可逆性。

#### （3）Fused Dual-Component Kernel（融合双分支内核）
- 设计专用 CUDA 内核，将 low-rank 和 residual 路径的两个 4-bit GEMM 完全融合执行。
- 实现：
  - **On-chip pipelining**：中间结果保留在片上内存，避免写回 global memory。
  - **Single epilogue fusion**：最终合并两路输出并一次性写回，极大减少访存开销。
- 支持端到端高效 4-bit 推理，无额外运行时开销。

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **准确性** | 显著优于 RTN、SmoothQuant、QuaRot、FlatQuant、SVDQuant 等 SOTA 方法，在 W4A4 下接近 FP16 性能 |
| **效率** | 融合内核带来高达 1.8× 的 end-to-end 加速，远超 FP16 基线 |
| **通用性** | 在 LLaMA3 和 Qwen3 多个规模模型上均表现稳定，鲁棒性强 |
| **实用性** | 所有变换可在离线阶段折叠进权重，推理时无需额外计算 |

---

## 2. 核心实验方法和设置

### 📚 数据集

- **Calibration Dataset**：从 `WikiText2` 随机采样 128 条文本，每条 2048 tokens，用于校准量化参数。
- **Evaluation Tasks**：
  - **语言建模**：WikiText2 上的 Perplexity（PPL）
  - **零样本推理任务**（zero-shot）：
    - ARC-Challenge / ARC-Easy
    - HellaSwag
    - LAMBADA
    - PIQA
    - WinoGrande
  - 使用 `lm-eval` 工具包统一评测。

### ⚙️ 实验设置

| 设置项 | 描述 |
|--------|------|
| **模型系列** | LLaMA3（3B, 8B）、Qwen3（4B, 8B, 14B, 32B） |
| **量化配置** | W4A4（权重和激活均为 4-bit）、W4A8、W4A16 |
| **Group Size** | 128（group-wise quantization） |
| **Low-rank Rank** | 默认 $ r = 128 $ |
| **硬件平台** | 
| - Environment 1：NVIDIA RTX 4090 (24GB) + Xeon Gold 6430  
| - Environment 2：NVIDIA L20 (48GB) + Xeon Platinum 8457C |
| **序列长度** | Prefill: 1024 tokens, Decode: 256 tokens |

### 🆚 基线方法对比

共比较 **8 种主流 PTQ 方法**：

| 方法 | 类型 | 特点 |
|------|------|------|
| **RTN** | Baseline | Round-to-nearest，无校准，速度快但精度差 |
| **GPTQ** | Weight-only | Hessian-aware，逐行量化补偿 |
| **AWQ** | Weight-only | 激活感知剪枝 + channel scaling |
| **SmoothQuant** | Affine-based | 通道缩放转移异常值 |
| **QuaRot** | Rotation-based | 固定 Hadamard 变换去相关 |
| **SpinQuant** | Rotation-based | 学习正交旋转矩阵 |
| **FlatQuant** | Flattening-based | 学习仿射变换压平分布 |
| **SVDQuant** | Low-rank-based | SVD 分解吸收异常值（FP16 低秩） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & A.3-A.5）

#### ✅ 准确性结果（W4A4 setting）

| 模型 | 方法 | Zero-shot Avg (%) | WikiText2 PPL |
|------|------|-------------------|-------------|
| LLaMA3-8B | FP16 | 73.5 | 8.6 |
| | TwinQuant | **70.9** | **9.4** |
| | SVDQuant (r=32) | 68.2 | 12.5 |
| | SpinQuant | 69.4 | 9.5 |
| | FlatQuant | 70.1 | 11.0 |
| Qwen3-32B | FP16 | 75.2 | 7.6 |
| | TwinQuant | **73.3** | **10.4** |
| | SVDQuant | 72.2 | 11.6 |
| | SpinQuant | 72.0 | 12.1 |

> **结论**：TwinQuant 在所有模型上均达到最接近 FP16 的性能，平均仅落后 0.6–2.6 个百分点。

#### ⚡ 推理速度（End-to-End Throughput）

| 平台 | 相对于 FP16 TensorRT-LLM 的加速比 |
|------|-------------------------------|
| Environment 1 (RTX 4090) | **1.63× – 1.80×** |
| Environment 2 (L20) | **1.32× – 1.73×** |

| 对比对象 | 平均加速比 |
|---------|-----------|
| vs. AWQ (W4A16) | up to **1.74×** |
| vs. QuaRot (W4A4) | avg. **2.04×** |
| vs. FlatQuant (W4A4) | avg. **1.59×** |
| vs. SVDQuant (r=128) | up to **1.07×** |

> **说明**：得益于融合内核设计，TwinQuant 即使引入双分支也未增加延迟，反而因更高效的访存获得速度优势。

#### 🔬 消融实验（Ablation Study, Table 3）

| 方法 | LLaMA3-8B (0-shot / PPL) | Qwen3-8B (0-shot / PPL) |
|------|--------------------------|--------------------------|
| Naive 4-bit | 41.7 / 91.6 | 41.5 / 4188 |
| + Low-Rank (SVD) | 61.1 / 19.6 | 62.8 / 20.3 |
| + Hadamard | 64.9 / 12.4 | 66.0 / 16.3 |
| **TwinQuant (full)** | **70.9 / 9.4** | **70.2 / 13.2** |

> **结论**：
> - 低秩分解本身已大幅恢复性能；
> - 固定 Hadamard 有一定帮助；
> - **可学习变换 $ Q, G $ 是性能跃升的关键**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **主流 LLM 权重不具备快速奇异值衰减特性** → 固定 SVD 分解效果有限。
2. **可学习子空间分解能有效重塑权重与激活分布**，降低动态范围和方向偏斜，显著提升 4-bit 量化鲁棒性。
3. **全局正交变换 $ Q $ + 层级可逆变换 $ G $ 的组合提供了更强的表达能力**，优于单一变换策略。
4. **融合双组件内核至关重要**：若不优化系统实现，低秩路径将成为瓶颈；而 TwinQuant 的 kernel 设计成功规避此问题。
5. **r=128 是性价比最优选择**：进一步增大 rank 带来的收益极小，但成本翻倍（见 Figure 6）。

### ⚠️ 方法的局限性

1. **当前评估集中于 Dense LLMs**：尚未验证在 MoE 架构或多模态模型上的有效性。
2. **Kernel 高度依赖特定 GPU 架构**（如支持 INT4 Tensor Core），迁移到其他加速器需重新适配。
3. **需要 calibration data 和离线优化时间**：虽然可摊销，但仍增加部署前成本（见 Table 8，最大模型耗时约 6.5 小时）。
4. **超长上下文场景下的行为未知**：未测试 >32k token 的性能表现。

### 🔮 未来工作方向

1. 扩展至 **MoE 和多模态模型** 的量化。
2. 开发 **跨架构通用 kernel** 或自动调优方案。
3. 探索 **更轻量级或免数据的优化策略**，例如基于合成数据或元学习。
4. 验证在 **超长上下文和 streaming 场景** 下的稳定性与效率。
5. 结合 **quantization-aware training (QAT)** 进一步逼近 FP16 性能。

---

> 💡 **总结一句话**：  
> **TwinQuant 通过“可学习子空间分解 + 混合流形优化 + 融合内核”三位一体的设计，在保持接近 FP16 精度的同时，实现了高达 1.8× 的 end-to-end 加速，为高效 4-bit LLM 推理提供了新的范式。**

</details>

---

### 6. [Parallelizing Large-Scale Tensor Network Contraction on Multiple GPUs](https://arxiv.org/abs/2606.01852)

**Authors**: Feng Pan, Hanfeng Gu, Paul Springer, Xipeng Li  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.01852v1  

#### Abstract
Exact tensor network contraction underpins quantum circuit simulation, quantum error correction, combinatorial optimization, and many-body dynamics. The dominant parallelization strategy, slicing, scales exponentially and incurs redundant computation. We present a multi-GPU framework that instead di...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Parallelizing Large-Scale Tensor Network Contraction on Multiple GPUs**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大规模 **Tensor Network (TN)** 收缩是量子电路模拟、量子纠错、组合优化和多体动力学等领域的核心计算瓶颈。传统并行化策略 **slicing** 虽能降低内存峰值，但存在以下严重缺陷：
- **指数级冗余计算**：每固定一个索引（slice）会生成 $2^n$ 个独立子任务，导致大量重复计算。
- **无法突破单设备内存墙**：当中间张量超出单个 GPU 内存时，收缩变得不可行。

本文旨在解决这一系统级挑战：如何在多 GPU 上高效执行原本因内存不足而“不可行”的 TN 收缩，并实现远超 slicing 的性能提升。

---

### **提出的新方法与新思路**
作者提出了一个基于 **显式通信的分布式收缩框架**，其核心思想是：
> 将中间张量显式地跨多个 GPU 分布，通过通信换取计算效率，而非依赖 slicing 产生冗余任务。

该框架包含两个关键创新模块：

#### **(1) GEMM-Oriented Mode Reordering**
- **目标**：为每个 pairwise contraction 构造适合 GEMM（通用矩阵乘法）的内存布局。
- **方法**：通过一次从后向前的遍历，按模式（mode）的“剩余生命周期”对张量维度排序：
  - **长生命周期模式**（保留到最后）放在前面（leading dimensions）
  - **短生命周期模式**（即将被收缩）放在后面
- **效果**：所有操作天然符合 `[retained | reduced]` 结构，避免运行时 transpose，直接调用高性能 GEMM 内核。

#### **(2) Communication-Aware Mode Distribution Planning**
- **目标**：决定哪些模式应被分布、何时重分布（redistribute）、何时聚集（gather），以最小化通信开销。
- **方法**：
  - 利用动态规划（Dynamic Programming, DP）搜索最优 redistribution 点。
  - 成本模型综合考虑：
    - 局部计算时间（`t_gemm`）
    - 通信时间（`t_comm`），包含带宽项和块粒度延迟项
  - 强调在“张量体积小”且“块连续性好”的“谷值点”进行 redistribution，避免在大张量时被迫通信。

最终调度由 **cuTENSORMp** 执行，支持自动化的 computation-communication overlap 和 NCCL 底层传输优化。

---

### **相比现有方法的优势**
| 方面 | Slicing（主流方法） | 本文方法（Distribution） |
|------|---------------------|--------------------------|
| 并行方式 | Embarrassingly parallel，无通信 | 显式通信，跨设备协作 |
| 计算代价 | 指数级冗余（$2^{n_{\text{slice}}}$） | 总 FLOPs 显著减少 |
| 内存扩展性 | 单设备内存限制仍存在 | 可利用聚合显存（如 8×80GB = 640GB） |
| 通信控制 | 无通信，但冗余高 | 主动管理通信时机与粒度 |
| 性能上限 | 受限于 slicing 开销 | 接近理论 compute-only 加速比 |

> ✅ **本质区别**：Slicing 是“空间换时间”（增加计算来降内存），而本文是“通信换计算”（增加可控通信来大幅降计算）。

---

## **2. 核心实验方法和设置**

### **使用的数据集/工作负载**
共测试 **4 类典型 TN 应用**，涵盖不同结构特征：
1. **Quantum Circuit Simulation**: `Zuchongzhi n60m24` — 高纠缠随机电路
2. **Many-Body Dynamics**:
   - Hexagonal 8×8
   - Rectangular 49×20
   - Triangular 49×24  
   （均通过 Trotter 时间演化构建时空 TN）

> 💡 图 1 还展示了 QEC（距离 7 表面码）和 King’s graph 独立集枚举，用于说明通用性。

---

### **实验设置**
- **硬件平台**：NVIDIA DGX H100
  - 单节点：8×H100 GPU（80GB HBM3），NVLink 互联（900 GB/s/GPU）
  - 多节点：最多 1024×H100，跨节点使用 InfiniBand（400 Gb/s）
- **软件栈**：
  - 路径查找器：使用 cotengra 等启发式工具生成 contraction path
  - 执行引擎：**cuTENSORMp** + NCCL
  - 数据类型：complex64，FP32 算术运算

---

### **评估指标**
| 指标 | 定义 | 用途 |
|------|------|------|
| `T_proj` | 投影总时间 = 单 slice 时间 × $2^{\text{slice\_bonds}}$ | 衡量完整收缩耗时 |
| `Speedup` | $ S_p = T_{\text{proj},1} / T_{\text{proj},P} $ | 相对于 1-GPU 的加速比 |
| `Extra Speedup` ($E_p$) | $ S_p / P $，即超出理想 slicing 并行度的部分 | **核心指标**，体现分布式的额外收益 |
| `Complexity Reduction` ($R_p$) | $ C_{t,1} / C_{t,P} $，仅考虑 FLOPs 减少（忽略通信） | 理论最大可获增益 |

> ⚠️ 注意：路径优化本身是 NP-hard，因此不同 GPU 数下的最优路径质量可能波动，影响结果趋势。

---

### **基线方法对比**
- **Baseline**：理想化的 **embarrassingly parallel slicing**
  - 假设每个 slice 均匀分配到 GPU，无通信开销
  - 理想加速比 = GPU 数量 $P$
- **对比目标**：衡量 `Extra Speedup` 是否显著大于 1

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 单节点（8 GPUs, NVLink）结果**
| Workload | Projected Speedup | Extra Speedup | Complexity Reduction | TFLOP/s per GPU |
|---------|-------------------|---------------|------------------------|------------------|
| Circuit (n60m24) | 148× | **18.5×** | 18.5× | 28.1 |
| Hexagonal 8×8 | 1,383× | **172.9×** | 197.8× | 31.6 |
| Rectangular 49×20 | 75× | **9.4×** | 9.3× | 32.9 |
| Triangular 49×24 | 56× | **7.0×** | 7.4× | 31.8 |

✅ **亮点**：
- 所有任务实现 **7–173× 的额外加速**（超越理想 slicing）
- 实际加速接近 compute-only reduction（达到 **87–101%** 效率）
- 高效利用硬件：持续 **28–33 TFLOP/s per GPU**（占 H100 FP32 peak 的 ~45%）

---

#### **(2) 多节点扩展至 1024 GPUs（InfiniBand）**
| Workload | Per-slice Time (s) | Slice Bonds | Projected Speedup | **Extra Speedup** | Complexity Reduction |
|---------|--------------------|------------|--------------------|-------------------|------------------------|
| Circuit (n60m24) | 20.19 | 20 → 仅需 20 slices | 42.8K× | **41.8×** | 418× |
| Hexagonal 8×8 | 113.27 | 37 → **6** | 69.5M× | **67,869×** | 1.49M× |
| Rectangular 49×20 | 34.70 | 14 | 221K× | **216.0×** | 3,154× |
| Triangular 49×24 | 12.19 | 14 | 135K× | **132.6×** | 986× |

✅ **惊人发现**：
- 在 Hexagonal 任务上，extra speedup 高达 **67,869×**！
- 原因：slicing bonds 从 37 降至 6，意味着计算量减少 $2^{31}$ 倍以上，尽管通信成本上升，但净收益巨大。
- 即使跨节点使用低带宽 InfiniBand，仍能获得数十至上万倍的额外加速。

---

### **消融实验与关键观察**
- **非单调性分析**：
  - Extra speedup 不随 GPU 数单调增长（例如在 16 GPUs 时下降）
  - 原因：
    1. 路径优化质量波动（NP-hard 导致不同配置下路径差异）
    2. 从 NVLink 到 InfiniBand 的通信跃迁带来更高延迟
- **通信成为瓶颈**：
  - 在 NVLink 上，通信占比低 → 实际加速 ≈ 理论 FLOP 减少
  - 在 InfiniBand 上，通信成本上升 → 实际加速 < 理论值，但仍远高于 slicing
- **DP 规划有效性验证**：
  - 图 5 显示 redistribution 被智能安排在“size valley”处，避免在百 GB 级张量上进行细粒度通信
  - 总重分布数据量仅为总移动量的 **4.6%**，证明高度优化

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **分布式收缩远优于 slicing**：
   - 在前沿 TN 问题中，distribution 可将总 FLOPs 降低几个数量级。
   - 即使引入通信，在现代多 GPU 架构下仍能获得 **数十至数万倍的额外加速**。

2. ✅ **通信是关键调节变量**：
   - 分布式收缩的本质是 **compute-communication trade-off**。
   - 加速潜力受限于 **interconnect bandwidth**，而非可用 FLOP reduction。

3. ✅ **NVLink 极大释放潜力**：
   - 在单节点内，高带宽 NVLink 使得通信几乎“免费”，实际性能逼近理论极限（87–101%）。

4. ✅ **方法具有高度通用性**：
   - 在量子电路、QEC、组合优化、多体动力学等多种应用中均有效。
   - 框架不依赖特定应用结构，仅需 contraction path 和 tensor shape。

---

### **局限性**
1. **路径优化仍是瓶颈**：
   - 当前方法假设 contraction path 固定，未联合优化路径与分布策略。
   - 路径质量波动导致 scaling 曲线非平滑。

2. **对低带宽网络敏感**：
   - 在 InfiniBand 上，通信延迟显著影响性能，尤其当 redistribution 不可避免发生在大张量阶段。

3. **依赖 cuTENSORMp 生态**：
   - 当前实现紧密耦合于 NVIDIA 软件栈（NCCL, cuTENSOR），向其他平台迁移需适配。

---

### **未来工作方向**
1. **联合路径与分布优化**：
   - 设计端到端编译器，同时优化 contraction path 与 distribution plan。

2. **支持异构与分层存储**：
   - 结合 CPU DRAM、SSD offloading 与 GPU HBM，进一步扩展可处理问题规模。

3. **推广至 MNNVL 与 GB200 架构**：
   - 利用 **multi-node NVLink (MNNVL)** 或 **GB200 NVL72**（72-GPU, 1.8TB/s/GPU）等新型高带宽互连，最大化通信效益。

4. **自动化 tuning 与自适应调度**：
   - 在运行时根据实际性能反馈动态调整 redistribution 策略。

---

> 🔚 **总结一句话**：  
> 本文颠覆了传统 slicing 范式，提出了一种通信感知的分布式 TN 收缩框架，在真实硬件上实现了高达 **67,869× 的额外加速**，标志着大规模精确张量网络模拟进入“聚合内存+高效通信”的新时代。

</details>

---

### 7. [HeLoCo: Efficient asynchronous low-communication training under data and device heterogeneity](https://arxiv.org/abs/2606.00271)

**Authors**: Abdullah Al Asif, Patrick Diem, Juan Pablo Mu\~noz, Felix Wolf, Ali Jannesari, Arya Mazaheri  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.00271v1  

#### Abstract
Distributed Low-Communication (DiLoCo) training reduces communication overhead by allowing workers to perform multiple local optimization steps before sending pseudo-gradients to a global outer update. Its asynchronous variant further improves hardware utilization by removing synchronization barrier...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HeLoCo: Efficient asynchronous low-communication training under data and device heterogeneity  
**—— 核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **asynchronous low-communication (DiLoCo)** 分布式训练中，存在两个关键挑战：
- **Gradient staleness**：异步更新导致部分 worker 提交基于过时模型状态计算的 pseudo-gradient，这些 stale updates 可能与当前全局优化方向不一致。
- **Data and device heterogeneity**：当 worker 之间设备速度不同（device heterogeneity）且本地数据为非独立同分布（non-IID）时，staleness 引起的方向冲突更加严重，影响收敛性和最终性能。

现有方法如 **asynchronous Nesterov** 或 **MLA (Momentum Look-Ahead)** 对整个 pseudo-gradient 进行统一修正，忽略了不同参数块（tensor blocks）可能具有不同程度的方向一致性，导致过度校正或校正不足。

### 提出了什么新方法或新思路
本文提出 **HeLoCo (Heterogeneity-aware Low-Communication training)**，一种方向感知的异步低通信训练校正方法，其核心思想包括：

1. **Momentum-guided look-ahead initialization**  
   在 worker 初始化时，不是直接发送当前 global model $ \theta_t $，而是发送一个基于 outer momentum 预测的“前瞻模型”：
   $$
   \tilde{\theta}_r = \theta_r - \eta_r \mu m_r
   $$
   这减少了 worker 开始训练时与未来 global model 的初始差距。

2. **Tensor-wise directional correction**  
   当 stale pseudo-gradient 到达时，HeLoCo 不再对整个梯度向量进行统一处理，而是按 **tensor block**（如权重矩阵、偏置项）逐个判断其与当前 outer momentum 的方向一致性，并选择性地执行以下操作：
   - **保留**：若方向一致（$ c_b \geq c_{ok} $），则保持不变；
   - **衰减反向分量**：若方向冲突（$ c_b < 0 $），仅衰减其反对 momentum 的部分；
   - **重定向弱对齐块**：若方向微弱对齐（$ 0 \leq c_b < c_{ok} $），将其平滑拉向 momentum 方向。

该策略通过 confidence-aware 几何投影机制实现，在保留有用局部信息的同时纠正有害方向。

### 相比现有方法的优势
- **更精细的控制粒度**：从全局 scalar correction 升级到 tensor-level vector correction，适应异构系统中各模块的不同对齐程度。
- **更强的鲁棒性**：在高 staleness 和 non-IID 场景下仍能稳定提升性能。
- **无需增加通信开销**：仅需访问已有 momentum buffer，额外计算成本为 $ O(d) $ 每次到达，可忽略不计。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **mC4 (multilingual C4)**：多语言版本的 Common Crawl 数据集，用于模拟真实世界中的 non-IID 数据分布。
- 每种语言作为一个独立的 data shard，分配给固定 worker，形成 **non-IID setting**。
- 同时也测试了 IID setting 作为对照。

### 实验设置和评估指标
| 设置项 | 描述 |
|-------|------|
| **模型架构** | TinyGPT-style decoder-only Transformer，约 15M 参数 |
| **训练方式** | Asynchronous DiLoCo 框架，每个 worker 执行 H 步 local SGD 后返回 pseudo-gradient $\Delta = \theta_{\text{init}} - \theta_{\text{final}}$ |
| **worker 数量** | 5 个，具有不同计算速度（模拟 device heterogeneity） |
| **local steps (H)** | 多数实验设为 80 步；部分为 20 步 |
| **outer steps** | 最多 300 轮 global 更新（即 24k total steps） |
| **评估指标** | - **Validation loss**（主指标）<br>- 固定 token budget 下的 loss<br>- 固定 wall-clock time 下的 loss |

### 基线方法对比
- **async-Nesterov**：标准异步 DiLoCo + Nesterov momentum
- **async-MLA**：Momentum Look-Ahead 方法，对整个 pseudo-gradient 统一外推修正
- **sync-Nesterov**：同步 DiLoCo 版本，无 staleness 但受限于最慢 worker
- **DyLU (Dynamic Local Updates)**：动态调整 local steps 以平衡参与频率（正交技术，可用于组合）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 场景 | HeLoCo 提升幅度 | 说明 |
|------|------------------|------|
| **Fixed token budget** | 最高优于 async-MLA **7.5%** | 在非IID + 异构设备下，loss 降低显著 |
| **Fixed wall-clock time** | 最高优于 async-MLA **3.3%** | 时间效率更高，硬件利用率更好 |
| **vs. sync-Nesterov under severe heterogeneity** | 最高达 **22.1%** 更低 validation loss | 尽管同步无 staleness，HeLoCo 凭借高频有效更新胜出 |

> 注：以上均为 **relative improvement in validation loss**，数值越小越好。

### 与基线方法的对比结果
- 在所有配置中，**async-Nesterov** 表现最差，尤其在高 staleness 下出现发散现象。
- **MLA** 比 Nesterov 更稳健，但在极端异构场景（如 worker pace: `[1,15,15,15,15]`）下优势减弱。
- **HeLoCo** 在绝大多数设置中表现最优：
  - 在 moderate staleness 下持续领先 MLA；
  - 在 homogeneous + non-IID 场景下仍有增益，表明 **data heterogeneity 本身即可受益于 block-wise correction**；
  - 在 extreme staleness 下虽略有退化，但仍优于 async baseline。

#### 示例：Table 1 中典型配置 `(1,6,6,6,6)`
| 方法 | Validation Loss |
|------|---------------|
| HeLoCo | 6.98 |
| async-MLA | 7.22 |
| async-Nesterov | 7.20 |
| sync-Nesterov | 6.62 |

→ HeLoCo 比 MLA 低 **3.33%**，比 async-Nesterov 低 **3.04%**

### 消融实验结果（隐含分析）
虽然未明确列出消融表，但从多个实验变体中可得出以下结论：

1. **Look-ahead initialization 有帮助但不够充分**  
   单独使用 lookahead 只能缓解初始偏差，无法解决 long-delay 更新的方向错位。

2. **Per-block correction 是关键**  
   - 英语（fast worker, avg staleness=0.72）收益较小（仅 0.02 loss 改善）；
   - 德语/法语（slow workers, avg staleness >5）改善明显（最高达 0.24 loss 下降）；
   → 证明 correction 对 stale updates 更重要。

3. **DyLU 并不能替代 HeLoCo**  
   DyLU 通过均衡更新频率来降低 staleness variance，但会牺牲部分 worker 的 local training 强度。实验显示，**保持强 local training + 后期 correction** 比提前限制更优。

4. **Discarding highly stale gradients?**  
   探索性实验发现：当 staleness 极高（如 pace 15）且数量多时，丢弃某些 stale pseudo-gradient 反而有助于 MLA 类方法，但损害 async-Nesterov。这暗示未来可引入 adaptive filtering。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Staleness is not uniform across tensor blocks**  
   同一 stale pseudo-gradient 内部，不同参数块与当前优化轨迹的对齐程度差异巨大，统一修正过于粗糙。

2. **Outer momentum is a reliable directional reference**  
   momentum 缓冲区天然反映了近期 global 更新趋势，适合作为判断 stale update 是否“可信”的依据。

3. **Fine-grained correction beats coarse correction**  
   tensor-level selective adjustment 显著优于 scalar scaling 或全量 extrapolation，在系统与数据双重异构下尤为关键。

4. **HeLoCo enables efficient asynchronous training even under severe heterogeneity**  
   它打破了“必须同步才能保证质量”的思维定式，实现了接近甚至超越同步方法的质量，同时具备异步的时间效率。

### 方法的局限性
- **极端 staleness 下效果下降**：当某个 worker 极其缓慢（如 15s/step），其更新过于陈旧，即使修正也可能无效或有害。
- **依赖 momentum stability**：若 momentum 波动剧烈（如学习率突变），方向参考可能不可靠。
- **实验规模有限**：目前仅在 5 worker、15M 参数模型上验证，尚未扩展至千卡集群或百亿参数 LLM。

### 未来工作方向
1. **Adaptive staleness-aware filtering**  
   结合 staleness 程度与 alignment score，动态决定是否 accept/correct/drop 某个 pseudo-gradient。
   
2. **Integration with decoupled systems**  
   将 HeLoCo 与 fully decoupled DiLoCoX 或 NoLoCo 架构结合，探索更大规模去中心化训练。

3. **Extension to other optimizers**  
   将 momentum-guided correction 思路推广至 Adam、Adafactor 等自适应优化器。

4. **Theoretical convergence analysis**  
   当前仅有局部理论支持（见 Appendix A.2），缺乏完整的异步收敛性证明。

---

> ✅ **一句话总结**：  
> **HeLoCo 通过 momentum-guided tensor-wise correction，在异步低通信训练中实现了对 stale pseudo-gradient 的精准修复，在多种异构场景下显著优于现有方法，推动了高效、鲁棒的大模型分布式训练向前一步。**

</details>

---

### 8. [MindZero: Learning Online Mental Reasoning With Zero Annotations](https://arxiv.org/abs/2606.00240)

**Authors**: Shunchi Zhang, Jin Lu, Chuanyang Jin, Yichao Zhou, Zhining Zhang, Tianmin Shu  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.00240v1  

#### Abstract
Effective real-world assistance requires AI agents with robust Theory of Mind (ToM): inferring human mental states from their behavior. Despite recent advances, several key challenges remain, including (1) online inference with robust uncertainty updates over multiple hypotheses; (2) efficient reaso...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MindZero: Learning Online Mental Reasoning With Zero Annotations

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文旨在解决现实世界中 **AI 助手需要具备强大 Theory of Mind (ToM)** 能力的问题，即从人类行为中推断其心理状态（如目标、信念、欲望），以实现主动、实时的辅助。然而，现有方法面临三大挑战：

1. **缺乏真实心理状态标注**：在真实场景中获取 ground-truth mental state 标注成本高昂且不可行。
2. **在线推理效率不足**：基于模型的推理方法（如 Bayesian Inverse Planning）虽鲁棒但计算开销大，难以支持实时应用。
3. **小模型内在 ToM 能力弱**：仅靠提示工程（prompting）的小型多模态语言模型（MLLMs）推理能力有限，易出错。

### **提出了什么新方法或新思路**

提出 **MindZero** —— 一种**无需任何心理状态标注**的自监督强化学习框架（Self-Supervised Reinforcement Learning, SSRL），用于训练 MLLMs 进行高效、鲁棒的在线心理推理。

- **核心思想**：将心理推理建模为一个**解释一致性优化问题**。模型被奖励生成能够最大化观测动作似然度的心理状态假设（mental state hypotheses），类似于 model-based ToM 推理中的逆向规划逻辑。
- **训练机制**：
  - 模型输出多个心理状态假设及其概率分布。
  - 奖励由两个部分构成：
    - 动作似然（Action Likelihood）：由 planner 或 LLM 估计该假设下产生实际动作的可能性。
    - 心理先验（Mental Prior）：由 LLM 判断该假设是否符合常识。
  - 加入熵正则项鼓励假设多样性，避免过早收敛。

通过这种方式，MindZero 将复杂的 model-based 推理过程“内化”到一个小型 MLLM 中，使其在推理时只需一次前向传播即可完成高质量的在线心理状态更新。

### **相比现有方法的优势**

| 方法类型 | 代表 | 缺陷 | MindZero 的优势 |
|--------|------|------|----------------|
| Prompting-based | LLM prompting | 推理错误多，依赖上下文理解 | 显著提升准确性与鲁棒性 |
| Model-based ToM | BIP, AutoToM | 计算昂贵，无法实时运行 | 推理速度快，适合在线任务 |
| Learning-based ToM | 需要标注数据 | 数据稀缺，泛化差 | **完全无监督训练，无需任何 mental state 标注** |

> ✅ **核心优势**：**兼具 model-based 方法的鲁棒性和单次前向模型的高效率**，同时摆脱对标注数据的依赖。

---

## 2. 核心实验方法和设置

### **使用的数据集**

1. **GridWorld Domain**
   - 基于 [Jha et al., 2024] 构建的二维网格环境，代理需搬运彩色物体完成组装任务。
   - 包含视觉输入（图像）和动作轨迹。
   - 自行生成训练与测试 episode。

2. **Household Domain**
   - 使用 **MMToM-QA**（Jin et al., 2024）作为问答基准。
   - 使用 **Online Watch-And-Help (O-WAH)**（Puig et al., 2023）作为主动辅助基准。
   - 基于 **VirtualHome v2.2.4** 模拟器构建家庭环境，包含厨房、卧室等房间及丰富物品交互。

### **实验设置和评估指标**

#### **两类任务设置**

| 任务类型 | 描述 | 特点 |
|--------|------|------|
| **Question Answering (QA)** | 回答关于人类心理状态的选择题（如信念、目标） | 离线评估，侧重最终判断准确率 |
| **Proactive Assistance** | 在每一步观察人类行为后，持续更新目标假设并决定如何协助 | 在线评估，要求低延迟、不确定性建模、动态决策 |

#### **关键评估指标**

- **Accuracy ↑**：在 QA 任务中衡量正确回答的比例。
- **Speedup ↑**：衡量助手加速主代理完成任务的程度：
  $$
  \text{speedup} = \frac{T_{\text{human}} - T_{\text{collab}}}{T_{\text{human}}}
  $$
  其中 $T_{\text{human}}$ 是无人协助时的时间，$T_{\text{collab}}$ 是协作时间。
- **Inference Cost (TFLOPs)**：以浮点运算量衡量推理开销，反映效率。
- **Goal Accuracy / F1 over time**：在主动辅助中按任务进度绘制目标预测准确率曲线，评估在线推理动态。

### **基线方法对比**

| 类别 | 基线方法 | 说明 |
|-----|---------|------|
| **Base Models** | Qwen3-VL-4B, Qwen3-VL-8B, Llama-3.1-8B, etc. | 开源基础 MLLM 或 LLM |
| **Large Proprietary Models** | GPT-5.2, Gemini-3-Flash/Pro | 零样本表现上限参考 |
| **Test-time Scaling Methods** | ThoughtTracing (Kim et al., 2025), AutoToM (Zhang et al., 2025) | 强大的推理时方法，但速度慢，不适用于实时辅助 |

> ⚠️ 注意：ThoughtTracing 和 AutoToM 因推理耗时长，未参与 Proactive Assistance 对比。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **(1) Question Answering 结果**

| 设置 | 方法 | 准确率 (Accuracy) | TFLOPs |
|------|------|------------------|--------|
| **GridWorld QA** | Qwen3-VL-4B (base) | 37.7% | 3.6 |
| | **MindZero w/ Qwen3-VL-4B** | **95.0%** | 3.6 |
| | → 提升倍数 | **2.5×** | ≈相同 |
| | AutoToM w/ Qwen3-VL-4B | 49.3% | 344.4 |
| | Gemini-3-Pro | 83.7% | Proprietary |

| **Household QA** | Llama-3.2-3B (base) | 34.8% | 4.0 |
| | **MindZero w/ Llama-3.2-3B** | **77.8%** | 4.4 |
| | → 提升倍数 | **2.2×** | ≈相同 |
| | AutoToM w/ Qwen3-4B | 54.7% | 177.5 |
| | Gemini-3-Flash | 67.2% | Proprietary |

✅ **结论**：MindZero 在保持极低推理成本的同时，显著超越所有开源基线，并接近甚至超过大型闭源模型的表现。

#### **(2) Proactive Assistance 结果**

| 设置 | 方法 | Speedup ↑ | TFLOPs ↓ |
|------|------|-----------|----------|
| **GridWorld PA** | Qwen3-VL-4B (base) | 1.4% | 151.7 |
| | **MindZero w/ Qwen3-VL-4B** | **23.0%** | 161.4 |
| | GPT-5.2 / Gemini-3-Flash | 0.0% | Proprietary |
| | → 原因 | 预测不稳定，频繁改变方向导致无效动作 |

| **Household PA** | Qwen3-4B (base) | 2.3% | 213.1 |
| | **MindZero w/ Qwen3-4B** | **19.1%** | 201.2 |
| | Gemini-3-Flash | 17.7% | Proprietary |
| | Qwen3-235B-A22B | 12.3% | 1101.6 |

✅ **结论**：MindZero 在主动辅助任务中实现了**最高加速比**，远超基线模型，且优于或媲美更大更贵的模型。

#### **(3) 在线目标推理动态分析（Figure 5）**

- MindZero 的目标预测准确率随任务进展**稳步上升**，表明其能有效积累证据、修正信念。
- 大多数基线在整个任务过程中准确率很低，直到最后才略有提升，**无法提供及时有效的早期帮助**。

#### **(4) 消融实验（Ablation Study on Qwen3-4B）**

| 变体 | Speedup | 相比完整版下降 |
|------|--------|--------------|
| **Full MindZero** | 19.1% | — |
| w/o Prior Modeling | 17.0% | ↓2.1% |
| w/o Multiple Hypotheses | 10.3% | ↓8.8% |
| w/o Entropy Bonus | 5.2% | ↓13.9% |

📌 **关键发现**：
- **熵正则项最重要**：防止模式坍塌（mode collapse），维持假设多样性。
- **多假设机制至关重要**：允许系统在不确定时推迟决策，提高稳定性。
- **显式先验建模有帮助**：过滤不符合常识的目标假设，防止奖励欺骗。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **心理推理可以作为一种自监督技能来学习**：MindZero 成功证明了无需任何心理状态标注，仅通过行为数据即可训练出具有强 ToM 能力的模型。
2. ✅ **模型可以内化 model-based 推理结构**：通过 SSRL 训练，小型 MLLM 学会了模仿 BIP 的推理逻辑，在推理阶段实现快速单步前向传播。
3. ✅ **效率与鲁棒性可兼得**：MindZero 在准确性和推理速度之间取得了极佳平衡，特别适合部署在资源受限或需实时响应的场景。
4. ✅ **在真实人类实验中有效**：在 IRB 批准的人类研究中，MindZero 提供的帮助使任务完成时间平均缩短 **19.7%**，效果与 Gemini-3-Flash 无统计学差异，但使用的是更小、可本地部署的开源模型。

### **方法的局限性**

1. ❌ **不支持递归心智推理（Recursive ToM）**：当前框架未建模多智能体之间的相互心理推测（如“我认为你认为…”）。
2. ❌ **输入长度限制**：随着任务变长，历史轨迹增长，模型输入 token 数也随之增加，可能影响长期记忆与推理能力。
3. ❌ **依赖高质量的 reward estimator**：动作似然和心理先验的质量直接影响训练效果，若 reward model 不可靠可能导致训练偏差。

### **未来工作方向**

1. 🔮 扩展至 **multi-agent recursive mental reasoning** 场景。
2. 🔮 设计更高效的模型结构以应对 **long-sequence input** 挑战。
3. 🔮 探索更鲁棒的 reward modeling 方式，减少对大模型 reward model 的依赖。
4. 🔮 将 MindZero 应用于更多现实场景，如数字助理、教育机器人、自动驾驶等。

---

> 💬 **总体评价**：  
> MindZero 是一项突破性的工作，它重新定义了 ToM 模型的训练范式——从依赖标注或昂贵推理，转向**通过自监督信号直接学习解释行为的能力**。这为构建真正实用、可扩展、低成本的智能助手机器人铺平了道路。

</details>

---

### 9. [Cost-Aware Diffusion Draft Trees for Speculative Decoding](https://arxiv.org/abs/2606.01813)

**Authors**: Shuai Zhang, Huachuan Qiu, Hongliang He, Yong Dai  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.01813v1  

#### Abstract
Speculative decoding accelerates inference by having a lightweight drafter propose tokens verified in parallel by the target language model. Block diffusion drafters such as DFlash generate an entire draft block in one pass, yielding per-position marginals; DDTree uses these to build a candidate tre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Cost-Aware Diffusion Draft Trees for Speculative Decoding**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **speculative decoding** 方法（如 DDTree）通过构建 **draft tree** 来并行验证多个候选 token，从而加速大语言模型（LLM）推理。然而，这些方法通常以 **最大化期望接受长度（expected acceptance length）** 为目标，并在训练前固定一个 **node budget**（即树中节点总数）。这种设计存在以下问题：

- **Acceptance length 是单调递增的**：随着 budget 增加，接受长度不会下降，因此总是倾向于选择更大的树，忽略了验证成本的增长。
- **缺乏对吞吐量（throughput）的直接优化**：更高的接受长度不一定带来更低的每 token 生成延迟（ms/tok），因为更大的树会显著增加 **verification cost**。
- **预算无法自适应变化**：最优 budget 在不同 decoding 轮次间差异很大（取决于上下文、drafter 置信度等），但现有方法使用固定的离线搜索 budget，无法动态调整。

### **提出了什么新方法或新思路**
本文提出 **CaDDTree**（**Cost-aware Diffusion Draft Tree**），一种全新的 draft tree 构建框架，其核心思想是：

- **将目标从“最大化接受长度”转变为“最大化 token throughput”**，即单位时间内生成的有效 token 数量。
- 显式建模 **drafting cost** 和 **verification cost**，并将 throughput 定义为：
  $$
  \text{Throughput} = \frac{1 + \mathbb{E}[a]}{C_d + C_v(n; l)}
  $$
  其中 $a$ 是每轮接受的 token 数，$C_d$ 是 drafting 成本，$C_v(n; l)$ 是依赖于节点数 $n$ 和上下文长度 $l$ 的 verification 成本。
- 利用 block diffusion drafter 输出的 per-position marginal distributions，结合贪心策略构建 draft tree，并引入一个 **greedy stopping rule** 动态决定何时停止扩展树。

### **相比现有方法的优势**
| 维度 | DDTree（及类似方法） | CaDDTree（本文） |
|------|------------------------|------------------|
| **优化目标** | 最大化 acceptance length | 直接优化 throughput（更贴近实际性能） |
| **budget 设置** | 固定超参数，需离线 grid search | 每轮自动选择最优 budget，无需调参 |
| **成本感知** | 忽略 verification cost 或假设为常数 | 显式建模 $C_v(n; l)$，支持上下文增长影响 |
| **理论保证** | 无 | 在 verification cost 凸性的假设下，throughput 函数关于 $n$ 是 **unimodal**，可由贪心算法找到全局最优 |
| **实现复杂度** | 需要预先确定 budget | 仅需一次性的 cost profiling，运行时完全自适应 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共八个 benchmark，覆盖多种任务类型：
- **Reasoning**: MATH-500, GSM8K, AIME 2025
- **Code Generation**: HumanEval, MBPP, LiveCodeBench
- **Instruction Following**: MT-Bench, Alpaca

### **实验设置和评估指标**
- **目标模型**：Qwen3-4B 和 Qwen3-8B
- **drafter 模型**：DFlash（block diffusion based）
- **硬件平台**：8×A800 GPU
- **精度**：bfloat16
- **最大新生成 token 数**：2048
- **温度设置**：0.0 和 1.0
- **block size**：16

#### **评估指标**
- **Per-token generation time (ms/tok)**：越低越好，反映实际推理速度
- **Mean acceptance length ($\bar{T}$)**：每轮平均接受的额外 token 数
- **Node budget ($n^*$)**：每轮使用的 draft tree 节点数

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **AR** | 标准自回归生成，无 speculative decoding |
| **DFlash** | 使用单条序列作为 draft，不构建 tree |
| **DDTree-oracle** | DDTree + 最优 budget 通过 grid search 获得（候选集：{16,32,...,1024}），作为上界参考 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Temperature = 0.0）**
见原文 Table 1，以下是代表性结果摘要：

| Dataset | DDTree-oracle (ms/tok) | CaDDTree (ms/tok) | 是否超越 |
|--------|------------------------|--------------------|----------|
| MATH-500 (4B) | 4.53 | **4.53** | ✅ 平齐 |
| GSM8K (4B) | 5.14 | **5.18** | ✅ 接近（微弱差距） |
| AIME 2025 (4B) | 4.95 | **4.96** | ✅ 平齐 |
| HumanEval (4B) | 4.97 | **5.01** | ✅ 接近 |
| MT-Bench (4B) | 8.34 | **8.32** | ✅ **反超** |
| Alpaca (4B) | 10.42 | **10.66** | ⚠️ 微慢（但仍优于 DFlash） |

> 对于 Qwen3-8B，grid oracle 被限制为 B=128，而 CaDDTree 自动选择更高 budget（如 165–187），在多数任务上表现更好。

### **与基线方法的对比结果**
- **CaDDTree 在几乎所有任务上匹配甚至超过 DDTree-oracle**，尽管后者使用了“上帝视角”的最优 budget。
- 特别是在 **MT-Bench** 上，CaDDTree 反超 oracle，说明其自适应机制能捕捉到 grid search 未覆盖的更优配置。
- 相比 DFlash，CaDDTree 显著提升性能（例如 MATH-500 上从 5.93 → 4.53 ms/tok）。

### **消融实验结果**
#### **Table 2: Cost Model Ablation**
测试是否显式建模 verification cost 的重要性：
| Cost Model | MATH-500 (ms/tok) | AIME 2025 | Alpaca |
|-----------|-------------------|---------|--------|
| **CaDDTree (真实 $C_v(n;l)$)** | 4.53 | 4.96 | 10.66 |
| **Constant cost（忽略增长）** | 9.14 | 10.25 | 19.37 |

> 若假设 verification cost 不随节点增加，则算法会无限扩张 tree，导致延迟翻倍以上，验证了成本建模的关键作用。

#### **Table 3: Sensitivity to Cost Parameters**
扰动 drafting/verification 成本参数 ±50%～100%，观察性能变化：
| 扰动 | Δ (ms/tok) |
|------|------------|
| $C_d × 0.5$ | -0.07 |
| $C_d × 2.0$ | +0.05 |
| $C_v × 0.5$ | -0.04 |
| $C_v × 2.0$ | -0.01 |

> 所有扰动引起的性能波动均小于 **0.07 ms/tok**，表明 CaDDTree 对 cost 参数估计误差鲁棒，无需极高精度 calibration。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Acceptance length 不应作为优化目标**：它是单调函数，无法平衡 coverage 与 cost，容易导致过度构建 draft tree。
2. **Throughput 是更合理的优化目标**：综合考虑了接受 token 数与总耗时，能自然权衡收益与开销。
3. **Verification cost 是凸函数**：实验证明 $C_v(n; l)$ 随节点数和上下文长度单调且凸增长（Figure 3），满足理论前提。
4. **Throughput 函数是 unimodal 的**：在 convex verification cost 下，$\phi^*(n)$ 存在一个峰值，可通过贪心方式高效求解。
5. **最优 budget 是动态变化的**：
   - 高置信度轮次：小 budget 即可饱和（窄链即可）
   - 低置信度轮次：需要宽树探索更多分支
   - 上下文越长，verification 越贵，应选更小 budget
6. **无需离线调参**：CaDDTree 每轮根据当前分布和 cost 自动决策，摆脱对 grid search 的依赖。

### **方法的局限性**
- 依赖于 **block diffusion drafter** 提供的 per-position marginals，难以直接应用于 autoregressive drafters（除非额外估计 marginal）。
- 需要进行一次性的 **cost profiling**，虽然简单快速，但在跨设备迁移时需重新校准。
- 当前 throughput 模型仍为启发式近似，未考虑 cache miss、内存带宽等底层系统因素。

### **未来工作方向**
- 将 CaDDTree 思想推广至其他类型的 drafter（如 autoregressive 或 MoE-based）。
- 结合 online learning 动态更新 cost model，适应运行时负载波动。
- 探索更精细的成本建模，纳入通信开销、KV cache 压力等。
- 与 training-time 方法（如 RL-based draft policy）结合，进一步提升 draft quality。

---

> ✅ **一句话总结**：  
> **CaDDTree 通过将 speculative decoding 的目标从“多接受 token”转向“快生成 token”，实现了无需调参、每步自适应、理论可证最优的 draft tree 构建，在多个 LLM 和 benchmark 上达到了媲美甚至超越 oracle budget 的性能。**

</details>

---

### 10. [Local MixVR: Breaking the Communication-Sample Dependence in Distributed Learning](https://arxiv.org/abs/2606.01128)

**Authors**: Tehila Dahan, Bassel Hamoud, Roie Reshef, Martin Jaggi, Kfir Y. Levy  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.01128v1  

#### Abstract
Communication overhead is a crucial bottleneck in scalable distributed learning. While existing methods aim to efficiently utilize data points, such as Local SGD, Minibatch SGD, and their accelerated variants, they still exhibit communication-round complexity that scales with the total number of sam...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Local MixVR: Breaking the Communication-Sample Dependence in Distributed Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在分布式学习中，**通信开销**是可扩展训练的关键瓶颈。尽管已有方法如 **Local SGD** 和 **Minibatch SGD** 通过本地更新或增大批量来减少通信轮次，但它们的**通信复杂度仍依赖于总样本数 $N$**。这意味着当数据量增大时，所需的通信轮次也随之增加，限制了大规模场景下的效率。

此外，现有方法（包括加速版本如 **Minibatch ASGD**）在常见训练设置下无法突破一个长期存在的理论瓶颈：即通信轮次 $R_{\text{min}}$ 至少为 $\Omega(N^{1/4})$，难以进一步压缩。

### **提出的新方法与思路**
本文提出了 **Local MixVR** ——一种结合多种方差缩减技术的新型分布式优化框架，旨在打破通信轮次对样本总数 $N$ 的依赖。

其核心创新由三个关键技术组件构成：

1. **Local Double-Momentum (局部双动量机制)**  
   - 结合 **Anytime Averaging** 和 **STORM 方差缩减估计器**，有效降低本地梯度噪声，使各 worker 的模型轨迹更一致，抑制“worker drift”（工人漂移）。

2. **Budget Mixing: Local Progress vs. Minibatch Averaging (预算混合机制)**  
   - 将每轮的 $K$ 个样本划分为两部分：
     - $(1-\alpha)K$ 用于执行本地优化步（推进优化路径）
     - $\alpha K$ 用于同步前的 minibatch 平均（减少噪声注入）
   - 这种混合策略平衡了“前进”与“稳定”，避免过度本地更新导致发散。

3. **Drift Correction Mechanism (漂移校正机制)**  
   - 在同步阶段引入基于相同 minibatch 的梯度差异作为校正项，补偿因本地更新积累的偏差，确保全局梯度估计无偏。

### **相比现有方法的优势**
- ✅ **首次实现通信轮次独立于样本总数 $N$**：通信复杂度仅依赖 worker 数量 $M$，达到 $R_{\text{min}} = \Omega(M)$。
- ✅ **在 $M \leq O(N^{1/4})$ 的常见训练场景下，优于当前最优的 Minibatch ASGD**。
- ✅ 在大规模数据集（如 FineWeb）上可将通信轮次减少数十倍（例如 30.7 倍），显著提升通信效率。
- ✅ 理论上填补了分布式优化中“能否超越 Minibatch ASGD”的长期空白。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MNIST**：手写数字分类数据集，60k 训练样本，10k 测试样本。
- **CIFAR-10**：图像分类数据集，50k 训练样本，10k 测试样本，包含 10 类彩色 32×32 图像。

### **实验设置与评估指标**
- **Worker 数量**：
  - MNIST：4 个 workers
  - CIFAR-10：8 个 workers
- **训练配置**：
  - MNIST：2 个 epoch，batch size = 4
  - CIFAR-10：30 个 epoch，batch size = 16
- **模型架构**：
  - MNIST：Two-layer CNN（卷积 + ReLU + MaxPool + FC）
  - CIFAR-10：ResNet-18
- **评估指标**：**测试准确率（Test Accuracy）随通信轮次 $R$ 的变化曲线**
- **调参范围**：
  - 学习率：{0.01, 0.05, 0.1}
  - 混合参数 $\alpha$：{0.05, 0.1, 0.25, 0.5, 0.75}
  - 动量系数：0.9（对应 $\beta=0.1$），$y_t=0.95$

### **基线方法对比**
- **Local SGD**
- **Local Momentum**
- **Minibatch SGD**
- **Minibatch ASGD**（当前最优基准）

所有方法在相同的通信预算（即相同 $R$）下进行比较，以公平评估通信效率。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 实验结果显示，在不同通信轮次数 $R$ 下，**Local MixVR 始终优于所有基线方法**。
- 特别是在低通信频次（small $R$）场景下优势更为明显，表明其具有更强的**抗稀疏通信能力**。

### **与基线方法的对比结果**
| 方法 | 通信轮次需求 | 是否依赖 $N$ | 相对 Minibatch ASGD 的改进 |
|------|----------------|---------------|----------------------------|
| Minibatch SGD | $\Omega(N^{1/2})$ | 是 | ❌ |
| Minibatch ASGD | $\Omega(N^{1/4})$ | 是 | 当前最优 |
| Local SGD | $\Omega(M N^{1/2})$ | 是 | ❌ |
| **Local MixVR (本文)** | $\Omega(M)$ | **否** | ✅ 在 $M \leq O(N^{1/4})$ 下更优 |

#### **实际案例中的性能增益**
- **ImageNet-1K**（约 $1.28\times10^6$ 图像，$M=8$）：
  - $R_{\text{ASGD}} / R_{\text{MixVR}} \sim N^{1/4}/M \approx 4.2$
  - 即 Local MixVR 可用 **约 1/4 的通信轮次** 达到相同性能。
- **FineWeb**（约 $15\times10^{12}$ token，$M=64$）：
  - $R_{\text{ASGD}} / R_{\text{MixVR}} \sim 30.7$
  - **通信轮次减少超过 30 倍！**

> 📈 图1 显示，在 MNIST 和 CIFAR-10 上，Local MixVR 在较少通信轮次下即可达到更高测试精度，验证了其高效性。

### **消融实验（未明确展示，但文中隐含分析）**
虽然没有单独列出消融图，但作者通过理论分解说明：
- 若缺少 **drift correction**，同步时会引入系统性偏差；
- 若不采用 **budget mixing**，本地更新过多会导致漂移加剧；
- **double-momentum** 对控制方差至关重要。

因此，三者协同作用是成功的关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. **通信-样本依赖可以被打破**：Local MixVR 首次证明，分布式学习的通信轮次**无需随数据规模增长而增加**，仅取决于 worker 数量 $M$。
2. **在现实训练场景中可大幅节省通信成本**：尤其适用于大模型预训练（如 LLMs）等海量数据、有限带宽环境。
3. **超越 Minibatch ASGD 成为可能**：解决了该领域长期存在的开放问题，在 $M \leq O(N^{1/4})$ 区间内实现了理论与实践双重突破。

### **方法的局限性**
- 当前分析限于 **凸函数设定**，非凸情况下的理论保证有待拓展。
- 参数 $\alpha$ 和动量调度需要调优，自动化程度有待提高。
- 实际部署需考虑异构网络延迟与计算负载均衡问题。

### **未来工作方向**
1. **推导期望平滑函数假设下的下界（lower bound）**，为本类问题建立完整理论框架。
2. **开发加速版本（accelerated variants）**，进一步提升收敛速度。
3. **扩展至联邦学习（Federated Learning）中的 Non-IID 场景**，增强实用性。
4. **探索与量化通信、稀疏传输等技术的结合**，打造端到端高效的分布式训练系统。

---

> 🔚 **总结一句话**：  
> **Local MixVR 通过融合双动量、预算混合与漂移校正，首次实现了通信轮次与数据总量解耦，在理论和实验上均显著超越 Minibatch ASGD，为大规模分布式训练提供了全新的高效范式。**

</details>

---

### 11. [ART: Attention Run-time Termination for Efficient Large Language Model Decoding](https://arxiv.org/abs/2606.00024)

**Authors**: Chen Qiu, Guozhong Li, Panos Kalnis  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.00024v1  

#### Abstract
Long-context decoding in Large Language Models (LLMs) is severely constrained by the memory bandwidth required to fetch the extensive Key-Value (KV) cache. Most existing KV management methods rely on key-only pruning before decoding, despite the evidence that attention outputs depend jointly on keys...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ART: Attention Run-time Termination for Efficient Large Language Model Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Large Language Models (LLMs)** 的长上下文解码过程中，**Key-Value (KV) cache** 的内存带宽消耗成为主要瓶颈。随着序列长度增长，注意力机制需要访问越来越多的缓存KV对，导致延迟和内存占用线性增加。

现有主流的 **KV cache 管理方法** 多为 **key-centric pruning**，即仅基于 query-key 相似度或 attention score 来决定保留哪些token。然而，这种做法忽略了 **value 向量本身对最终输出的影响** —— 即使某个token的 attention score 很低，其对应的 value 可能仍具有重要语义作用（如图1所示）。

因此，单纯依赖 key-based 信号进行剪枝可能导致精度损失，而引入 value 信息的传统方法又往往需要额外预测器或离线分析，带来显著推理开销。

### 提出了什么新方法或新思路
本文提出 **Attention Run-time Termination (ART)**，一种轻量级、运行时动态终止注意力计算的新机制。

- **核心思想**：在 **FlashAttention-style kernel 执行过程中**，实时监控中间注意力输出的变化情况。一旦后续KV块对输出的增量贡献趋于稳定（变化极小），则提前终止对剩余KV块的加载与计算。
- **关键洞察**：现代优化注意力内核（如 FlashAttention）以分块方式逐步累积注意力输出，这为在运行时观察输出收敛提供了天然机会。
- **设计亮点**：
  - 不依赖预定义的剪枝规则，而是 **output-aware**，直接在输出空间判断是否收敛。
  - 同时捕捉 keys 和 values 的联合影响，无需显式建模 value 重要性。
  - 与现有 KV cache 管理方法正交，可无缝集成。

### 相比现有方法的优势
| 维度 | 现有方法（如 SnapKV, H2O） | ART |
|------|----------------------------|-----|
| 决策依据 | Key-based importance（attention score） | Output convergence（实际输出变化） |
| 是否考虑 value 影响 | 否（隐含假设 value 不重要） | 是（通过运行时输出自然捕获） |
| 额外开销 | 无或低（静态策略） / 高（需额外模型） | 极低（仅轻量探测） |
| 可组合性 | 独立使用 | ✅ 可与任何 KV 策略结合 |
| 动态适应性 | 弱（固定规则） | 强（按需终止） |

> ✅ **ART 的最大优势在于它是一种“事后”（ex-post）、输出感知的动态控制机制，而非“事前”（a priori）的启发式剪枝。**

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 主要评测基准：**LongBench**（Bai et al., 2024）
  - 一个双语（中英）、多任务、专为长上下文理解设计的综合评测集。
  - 包含 6 类共 21 个子任务：
    - **Single-document QA**
    - **Multi-document QA**
    - **Summarization**
    - **Few-shot Learning**
    - **Synthetic Tasks**（如数字串查找）
    - **Code Completion**

### 实验设置和评估指标

#### 模型
- 主要模型：**Mistral-7B-Instruct-v0.3**
- 补充验证：**Llama-3.1-70B-Instruct**（跨规模泛化性）

#### 硬件平台
- 单张 **NVIDIA A100-SXM4 80GB GPU**

#### 评估指标
| 指标 | 描述 |
|------|------|
| **LongBench Score (Avg.)** | 综合准确率，衡量生成质量 |
| **Time Per Output Token (TPOT)** | 解码阶段每 token 耗时（ms/token），反映延迟 |
| **Generation Throughput** | 输出 token 数/秒，尤其关注大 batch 下的表现 |
| **FlashAttention Kernel Time** | 内核实测执行时间，用于隔离系统开销 |

#### ART 参数配置
- 规模容忍度 $ \tau = 10^{-7} $
- 方向容忍度 $ \phi = 10^{-3} $
- 耐心参数（patience）$ p = 5 $

---

### 基线方法对比
| 方法 | 类型 | 说明 |
|------|------|------|
| **Baseline (Full KV)** | 全量缓存 | 不做任何剪枝 |
| **StreamingLLM** | Pattern-based | 利用 Attention Sink 保留初始 token |
| **SnapKV** | Importance-based | 基于观察窗口估计 token 重要性 |
| **PyramidKV** | Hierarchical | 分层压缩 KV cache |

> 所有方法均测试两种保留比例：**80%** 和 **20% KV cache retention**

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 整体加速效果
- 在 **大 batch size** 场景下，ART 实现 **高达 20% 的生成吞吐提升**，相比当前最优基线（如 SnapKV）。
- 即使在全量 KV 缓存下，ART 也能显著降低 TPOT。

#### 准确率保持能力
- 应用于 Full KV 时，ART 保留了 **99.2% 的原始 LongBench 分数**，表明其几乎不损失精度。
- 与其他 KV 方法结合后，平均准确率下降通常小于 **1%**，部分任务甚至略有上升（可能因减少噪声干扰）。

#### 吞吐随 batch size 的扩展性
- 如图6所示，随着 batch size 增加（1 → 8），ART 的吞吐增益愈发明显。
- 原因：大 batch 加剧 HBM 带宽竞争；ART 减少不必要的 KV 访问，缓解内存瓶颈，且其检测开销是常数级，收益线性放大。

---

### 与基线方法的对比结果

| 方法组合 | Avg. Score | △Score | Kernel Time ↓ | △Time |
|--------|-----------|--------|----------------|-------|
| Baseline | 46.29 | – | 0.633 ms | – |
| + ART | 45.91 | -0.38 | 0.628 ms | -0.8% |
| SnapKV (0.8) | 43.92 | – | 0.506 ms | – |
| + ART | 43.12 | -0.80 | 0.498 ms | -1.6% |
| PyramidKV (0.2) | 30.07 | – | 0.382 ms | – |
| + ART | 29.97 | -0.10 | 0.375 ms | -1.8% |

> ✅ ART 在各类基线上均实现进一步加速，且精度损失极小。

---

### 消融实验结果（Ablation Study）

| 方法变体 | QA Score | Sum Score | Fewshot | Total Score | △Score | Kernel Time | △Time |
|----------|---------|----------|---------|-------------|--------|--------------|--------|
| ART (full) | 35.58 | 26.40 | 60.04 | **45.91** | – | 0.633 ms | – |
| w/o `dscale` | 14.78 | 21.18 | 27.74 | 19.34 | **-26.57** | 0.135 ms | -78.7% |
| w/o `ddirection` | 35.32 | 26.25 | 62.64 | 43.87 | -2.04 | 0.628 ms | -0.8% |
| w/o patience | 33.58 | 25.96 | 58.11 | 41.54 | -4.37 | 0.506 ms | -20.1% |

#### 结论：
- **`dscale`（规模变化检测）至关重要**：缺失会导致过早终止，严重损害准确性。
- **`ddirection`（方向一致性）有效增强鲁棒性**：小幅提升准确率，开销可忽略。
- **耐心机制（patience）防止误判**：虽略微增加耗时，但避免由瞬时波动引起的错误终止。

---

## 4. 关键结论和发现

### 主要发现
1. **注意力输出早期即趋于稳定**：大量 KV 块对最终输出贡献微乎其微，存在显著冗余计算。
2. **仅靠 attention score 不足以判断冗余性**：低分 token 的 value 仍可能有高影响力（见图1）。
3. **运行时输出监控是高效且准确的方式**：ART 通过监测中间输出的 **scale 和 direction 收敛性**，实现了对 keys 和 values 联合影响的隐式建模。
4. **ART 是轻量、通用、可组合模块**：
   - 开销仅增加 **1.3% kernel time**（禁用终止时）。
   - 可与 StreamingLLM、SnapKV、PyramidKV 等任意方法结合，持续提效。
5. **长上下文越长，ART 效益越显著**：在 >8k 上下文场景下，效率增益尤为突出。

---

### 方法的局限性
1. **依赖 block-wise attention kernel**：必须运行在支持流式计算的内核（如 FlashAttention）之上，无法应用于传统全量 attention 实现。
2. **对 very short context 效益有限**：当上下文较短时，KV 访问本就不多，提前终止空间小。
3. **超参敏感性**：虽然整体稳健，但 `dscale` 过大会导致精度下降（trade-off 存在）。
4. **大模型端到端加速受限**：在 70B 级别模型上，由于 FFN 层等其他组件成为瓶颈，ART 对整体 TPOT 的改善不如 7B 显著（但仍有效）。

---

### 未来工作方向
1. **自适应稳定性阈值**：根据不同网络层动态调整 `τ`, `ϕ`，以更好平衡速度与精度。
2. **硬件反馈驱动终止**：结合 GPU 内存带宽利用率等硬件信号，实现更智能的运行时决策。
3. **扩展至训练场景**：探索 ART 在长序列训练中的应用潜力。
4. **与其他优化技术协同**：如与 PagedAttention、量化、稀疏化等系统级优化深度整合。

---

> 💡 **总结一句话**：  
> ART 提出了一种 **轻量、输出感知、运行时终止注意力计算** 的新范式，突破了传统 key-only pruning 的局限，在几乎不损精度的前提下，将 LLM 长上下文解码效率提升了 **20%+**，并可作为通用插件兼容现有所有 KV cache 管理策略。

</details>

---

### 12. [LASER: Loss-Aware Singular-value Decomposition and Rank Allocation for Efficient Low-Precision Vision-Language Models](https://arxiv.org/abs/2606.00573)

**Authors**: Haiyu Wang, Yutong Wang, Leshu Li, Yihui Ren, Sai Qian Zhang  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.00573v1  

#### Abstract
Vision-language models (VLMs) deliver strong multimodal reasoning capabilities, but their large computational cost and high parameter counts make deployment challenging on resource-constrained devices. Low-rank decomposition has emerged as a promising compression technique, yet existing methods ofte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LASER: Loss-Aware Singular-value Decomposition and Rank Allocation for Efficient Low-Precision Vision-Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Vision-Language Models (VLMs) 虽然在多模态任务中表现出色，但由于其巨大的参数量和计算开销，在资源受限设备上的部署面临挑战。现有的低秩分解（low-rank decomposition）压缩方法存在以下不足：
- 多数仅优化局部矩阵重建误差（reconstruction error），忽略对下游任务损失的影响；
- 采用均匀或启发式的 **rank allocation** 策略，未考虑不同层/注意力头对压缩的敏感性差异；
- 主要聚焦于 **Self-Attention (SA)** 层中的 QKV 投影，而忽略了占模型大部分参数的 **Feed-Forward Network (FFN)** 层。

### 🚀 提出的新方法：LASER
本文提出 **LASER**（Loss-Aware Singular-value dEcomposition and Rank allocation），一种面向高效低精度 VLM 推理的低秩压缩框架，包含三大核心创新：

#### （1）Loss-Aware SVD（损失感知奇异值分解）
- 基于 **second-order approximation** 的模型损失，推导出一个损失感知的 SVD 目标函数。
- 利用 **Kronecker-Factored (K-FAC)** 近似的 Fisher 信息矩阵作为局部损失曲率估计，指导权重分解方向，使低秩近似更关注对最终任务性能影响大的部分，而非单纯最小化权重重建误差。

#### （2）Loss-Aware Cross-Layer Rank Allocation（跨层损失感知秩分配）
- 发现直接使用 K-FAC 得到的奇异值进行全局排序不可靠（因各层估计尺度不一致）。
- 引入基于 **calibration gradients** 的重要性评分机制，结合奇异值大小与其在原始参数空间中对损失的实际贡献，实现更准确的跨层、跨头秩分配。
- 允许在有限参数预算下优先保留最重要的奇异成分。

#### （3）Hybrid Low-Rank + Quantization for FFN Layers（混合低秩与量化用于 FFN）
- 首次将低秩压缩系统性地扩展至 FFN 层。
- 设计了一种 **hybrid SVD-quantization scheme**：通过临时 SVD 评估每个隐藏通道的“可压缩性”，选择适合 SVD 的通道进行低秩分解，其余通道保持稠密并进行量化。
- 在低秩分支引入旋转矩阵（rotation matrix）以平滑因子分布，提升量化鲁棒性。

#### （4）System-Level Optimization with Fused Triton Kernels
- 开发了融合的 **Triton kernels** 来实现 hybrid FFN 和低精度 SA 层，减少内存访问和 kernel launch 开销，最大化推理加速效果。

### 🔍 相比现有方法的优势
| 维度 | LASER | 传统方法（如 SVD-LLM, WSVD） |
|------|-------|-----------------------------|
| 优化目标 | 对齐下游任务损失 | 最小化局部重建误差 |
| 秩分配策略 | 损失感知、跨层动态分配 | 固定/均匀分配 |
| 应用范围 | SA + FFN 全覆盖 | 通常仅限 SA 层 |
| 量化协同 | 显式设计 QAW（Quantization-Aware Whitening） | 忽视量化影响 |
| 系统实现 | 融合 kernel 优化延迟 | 通用 kernel |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **校准数据集（Calibration Dataset）**：从 **ScienceQA** 训练集中随机抽取 256 个样本，用于收集激活值和梯度以估计 K-FAC 因子。
- **评估基准（Evaluation Benchmarks）**：
  - **ScienceQA-IMG**：科学图像问答任务
  - **SEED-Bench-IMG**：多模态理解综合评测
  - **MMBench**：通用多模态能力评测
- 工具包：使用 **VLMEvalKit** 统一评估流程。

### ⚙️ 实验设置
- **模型**：在五个代表性 VLM 上验证：
  - LLaVA-v1.5 7B
  - LLaVA-Next 7B / 13B
  - Qwen2-VL 7B
  - SmolVLM 2B
- **压缩配置**：
  - **低秩分解**：应用于每头 Q/K/V 投影矩阵
  - **FFN 设置**：SVD channel ratio $ \gamma = 50\% $，保留 $ p_{\text{ffn}} = 90\% $ 参数
  - **量化配置**：
    - LASER & WSVD：**W8A8**（INT8 权重 + INT8 激活）
    - 其他基线：**W8A4**（INT8 权重 + INT4 激活）
  - **KV Cache**：LASER 中 SVD 将 KV cache 大小降低约 50%
- **硬件平台**：NVIDIA H100 / RTX 4090 / 5090 GPU
- **批大小**：32

### 📊 评估指标
- **准确性（Accuracy）**：在多个 benchmark 上的平均准确率
- **解码延迟（Decoding Latency）**：layer-wise 解码耗时，衡量推理速度
- **参数压缩比（$ p_1 $）**：低秩后参数占比
- **KV Cache Size Reduction**：基于 rank ratio $ r/H $

### 🔁 基线方法对比
| 类型 | 方法 |
|------|------|
| **SVD-based** | SVD-LLM, QSVD, WSVD |
| **Quantization-based** | DuQuant, Q-VLM |
| **其他** | Flash Decoding, Palu |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）FP16 下的低秩压缩性能（LASER-noQ）
| 方法 | Avg. Accuracy (↓p₁=50%) | ScienceQA-IMG (SmolVLM) |
|------|------------------------|--------------------------|
| SVD-LLM | 18.83% | 0.31% |
| QSVD-noQ | 30.21% | 0.88% |
| WSVD-noQ | 46.87% | 24.52% |
| **LASER-noQ** | **62.74%** | **61.09%** ✅ |
| FP16 Baseline | 64.47% | 84.58% |

> ➤ LASER-noQ 在极端压缩下仍能保持接近 FP16 性能，显著优于所有基线。

#### （2）低精度（W8A8）下的端到端性能
| Method | Avg. Accuracy | vs FP16 ↓ |
|--------|---------------|-----------|
| DuQuant | 62.97% | -5.6% |
| Q-VLM | 57.84% | -10.73% |
| QSVD | 65.91% | -2.66% |
| WSVD | 67.11% | -1.46% |
| **LASER** | **67.94%** | **-0.63%** ✅ |
| FP16 | 68.57% | — |

> ➤ LASER 在更低参数预算和相同 cache 大小下，达到最接近 FP16 的精度。

#### （3）系统级推理加速（RTX 4090/5090）
| 方法 | 相对于 Flash Decoding 加速比 | 相对于 WSVD 加速比 |
|------|-------------------------------|--------------------|
| **LASER** | **4.7×** | **2.3×** ✅ |

> ➤ 解码速度大幅提升，得益于 fused kernels 和 hybrid 架构优化。

### 🔬 消融实验结果

#### （1）Loss-Aware SVD vs Vanilla SVD（表3）
| 方法 | Avg. Accuracy |
|------|----------------|
| Vanilla SVD | 51.02% |
| **LASER (Loss-Aware SVD)** | **72.95%** (+21.93%) ✅ |

> ➤ 验证了损失感知目标的有效性。

#### （2）Rank Allocation 策略对比（表4）
| 方法 | Avg. Accuracy |
|------|----------------|
| Uniform Allocation | 59.51% |
| SV-based (按奇异值排序) | 60.97% |
| **LASER (Gradient-based)** | **68.86%** ✅ |

> ➤ 证明了跨层重要性评分优于简单排序。

#### （3）Quantization-Aware Whitening (QAW)（附录A.5.3）
| 方法 | Avg. Accuracy |
|------|----------------|
| W/o QAW | 68.52% |
| **With QAW** | **68.86%** ✅ |

> ➤ 表明考虑量化后的输入分布可进一步提升稳定性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Loss-Aware Objective 更有效**：相比传统 SVD，基于 K-FAC 的损失感知 SVD 能更好地保留对下游任务重要的信息。
2. **Rank 不应统一或按奇异值分配**：不同层/头的 K-FAC 估计存在尺度偏差，需通过 empirical-Fisher importance score 进行校准。
3. **FFN 是压缩的关键目标**：FFN 占据大量参数，通过 hybrid SVD+quantization 可实现高效压缩而不显著降损。
4. **系统实现至关重要**：fused Triton kernels 显著降低了低秩推理的额外开销，释放了理论优势。

### ⚠️ 方法的局限性
- 当前评估集中在主流 VLM 架构上，尚未测试更新颖的架构（如 MoE-VLM）。
- 未探索低于 INT8 的极低比特设置（如 W4A4）。
- 长上下文生成场景下的累积误差未深入分析。
- 方法依赖 calibration data，若分布偏移可能影响性能。

### 🔮 未来工作方向
- 扩展至 **MoE 架构** 和 **video-language models**。
- 探索 **dynamic rank allocation**，根据输入自适应调整压缩强度。
- 结合 **pruning + low-rank + quantization** 构建统一压缩 pipeline。
- 研究 **training-in-the-loop** 的损失感知压缩策略，进一步缩小与原模型差距。

---

> 💡 **总结一句话**：  
> **LASER 通过损失感知的 SVD 与跨层秩分配，首次实现了对 VLM 中 SA 与 FFN 层的联合低秩压缩，并结合量化与系统优化，在几乎无损精度的前提下实现了高达 4.7× 的解码加速。**

</details>

---

### 13. [Efficient Test-time Inference for Generative Planning Models](https://arxiv.org/abs/2606.00618)

**Authors**: Robert Gieselmann, Mihai Samson, Federico Pecora, Jeremy L. Wyatt  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.00618v1  

#### Abstract
Generative models have emerged as a powerful paradigm for AI planning, yet their performance remains constrained by the training data distribution. One approach is to improve generated solutions during inference by scaling test-time compute. A more efficient alternative is to optimize the inference ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Test-time Inference for Generative Planning Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
生成式规划模型（Generative Planning Models）虽然在快速生成计划方面表现出色，但其性能受限于训练数据的分布。由于最优规划数据难以大规模获取，模型通常基于次优解进行训练，导致其推理时生成的方案质量不高。传统的测试时计算扩展方法（如 Best-of-N 采样）效率低下，而基于 MCTS 的搜索方法存在“宽而浅”的树结构问题，难以有效探索深层决策路径。

此外，经典 OCL（Open-Closed List）搜索算法（如 A*）虽能全局选择节点，但缺乏对生成式模型的有效集成机制，且对**过估计的启发函数**（overestimating heuristics）敏感，容易陷入局部贪婪行为。

### **提出的新方法：OCLGEN**
本文提出了 **OCLGEN**，一种高效的测试时搜索算法，将经典的 OCL 搜索框架与现代生成式模型相结合，实现高质量、低计算成本的规划推理。

#### **核心创新点**
1. **Depth-partitioned Selection（深度分区选择）**
   - 将开放列表（open list）按节点深度 `g(n)` 分区管理，每个深度维护独立的候选节点集合。
   - 避免传统 A* 中因启发函数过估计而导致的对深层节点的系统性偏好，确保不同深度的节点得到均衡探索。

2. **Truncated Rollouts with Adaptive Expansion（截断式 rollout 与自适应扩展）**
   - 利用生成模型进行多步 rollout，快速合成候选动作序列。
   - 引入**置信度阈值**（confidence threshold），仅在模型预测不确定性高的动作处进行分支扩展，显著减少冗余计算。
   - 动态控制搜索广度，在高置信区域“贪心推进”，在低置信区域“主动探索”。

3. **Distributional Heuristic Estimation（分布式启发函数估计）**
   - 启发函数建模为对剩余步数的概率分布预测（而非单一标量）。
   - 使用**下百分位数**（lower percentile）作为启发值 `h(n)`，以反映“最佳可达结果”而非平均表现，从而缓解过估计偏差的影响。

---

### **相比现有方法的优势**
| 方法 | 局限性 | OCLGEN 的改进 |
|------|--------|----------------|
| **Best-of-N** | 反复生成完整计划，无法重用中间状态，效率低 | 支持状态共享与增量优化 |
| **MCTS** | UCT 选择偏向根节点附近，导致树“宽而浅” | 全局节点选择 + 深度平衡探索 |
| **A*/OCL** | 对过估计启发函数敏感，易陷入局部最优 | 深度分区 + 百分位启发值 抵消偏差 |
| **纯生成模型** | 输出质量受训练数据限制 | 测试时主动优化，提升泛化能力 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个经典 AI 规划领域进行大规模评估：
- **Blocksworld**：积木世界，NP-hard，目标是重新排列积木。
- **Logistics**：物流运输，涉及卡车与飞机跨城市配送包裹。
- **Labyrinth**：网格导航，可移动整行/列卡片，具有循环边界。
- **Sokoban**：推箱子，PSPACE-complete，需将箱子推至目标位置。

所有数据集均通过 **Fast Downward (LAMA-first)** 生成次优解作为训练数据，并划分训练/验证/测试集（各 100k / 1k / 1k 实例）。

---

### **实验设置与评估指标**

#### **模型架构**
- **生成模型 `Tθ`**：GPT-2 架构，autoregressive 地生成动作 token 序列。
- **启发函数模型 `hd`**：BERT 编码器，输出剩余步数的概率分布。

#### **评估指标**
- **Completion Rate (%)**：成功找到有效计划的比例。
- **Average Plan Length**：有效计划的平均长度（越短越好）。
- **Optimality (%)**：在已知最优解的问题子集上，达到最优解的比例。
- **Convergence Speed**：随运行时间变化的计划质量提升速度。
- **Self-improvement Effectiveness**：用于迭代自增强框架的能力。

#### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **MCTS (full/partial rollouts)** | 使用 PUCT 的蒙特卡洛树搜索，分别使用完整或截断 rollout |
| **OCL-Anytime A*** | 经典 A* 搜索，使用学习到的启发函数 |
| **OCL-GBFS** | 贪婪最佳优先搜索 |
| **Best-of-N** | 多次采样返回最短有效计划 |
| **FD-LAMA-anytime/first/optimal** | Fast Downward 求解器的不同配置，作为符号规划基准 |

所有方法共享相同的生成与启发模型，仅比较推理策略差异。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **整体性能（Table 1）**
| 方法 | Blocksworld (Length) | Logistics (Length) | Labyrinth (Length) | Sokoban (Length) | 完成率 |
|------|------------------------|--------------------|---------------------|------------------|--------|
| **OCLGEN (uniform)** | **43.88** | **155.83** | **12.99** | **128.67** | 99.7–100% |
| MCTS (full) | 54.56 | 158.75 | 15.81 | 131.30 | 100% |
| Best-of-N | 65.23 | 61.56 | 18.03 | 132.25 | 89.5–99.9% |
| FD-LAMA-anytime | 45.05 | 161.00 | 19.77 | 141.43 | 100% |

👉 **OCLGEN 在所有领域均取得最短平均计划长度**，尤其在 Blocksworld 上比 MCTS 缩短 **19.6%**。

---

#### ✅ **最优性表现（Table 2）**
在已知最优解的子集上（共 1397 个实例），OCLGEN 显著提升最优解比例：

| 方法 | Blocksworld (Optimal) | Logistics (Optimal) | Labyrinth (Optimal) | Sokoban (Optimal) |
|------|------------------------|-----------------------|----------------------|-------------------|
| **OCLGEN (uniform)** | **83.8% (528/630)** | **61.5% (104/169)** | **98.8% (988/1000)** | **77.1% (377/489)** |
| MCTS (full) | 28.6% | 37.9% | 60.5% | 59.5% |
| FD-LAMA-anytime (10min) | 94.6% | 62.1% | 45.9% | 86.1% |

> ⚠️ 注意：FD-LAMA 在简单问题上表现好，但在高复杂度问题中扩展性差；OCLGEN 在**全测试集**上综合表现更优。

---

#### ✅ **收敛速度（Figure 2）**
- OCLGEN 在 **30 秒内**即可在 Blocksworld 达到 MCTS 需 **5 分钟**才能达到的质量。
- 在 Labyrinth 中，OCLGEN 在 **200 秒内收敛至近最优**，而 MCTS 仍处于缓慢下降阶段。

---

#### ✅ **消融实验（Table 4）**
移除任一组件均导致性能下降，证明各模块协同作用：

| 消融项 | Blocksworld (Length) | Blocksworld (Optimal) |
|--------|------------------------|------------------------|
| 完整 OCLGEN | 43.88 | 528/630 |
| w/o Depth Partitioning | 48.81 (+4.93) | 366/630 |
| w/o Adaptive Expansion | 44.35 (+0.47) | 496/630 |
| w/o Percentile Estimate | 44.68 (+0.80) | 486/630 |

👉 **深度分区选择影响最大**，说明其对克服启发函数偏见至关重要。

---

#### ✅ **自增强效果（Table 5 & 6）**
进行 3 轮 self-improvement 后：
| 方法 | Blocksworld (Final Optimal) | Sokoban (Final Optimal) |
|------|-------------------------------|--------------------------|
| **OCLGEN + self-improve** | **100% (630/630)** | **94.7% (463/489)** |
| MCTS + self-improve | 71.4% | 81.8% |

👉 OCLGEN 不仅自身更强，还能为后续模型提供更高质量的监督信号，形成正向反馈循环。

---

## **4. 关键结论和发现**

### **主要发现**
1. **OCL 框架可现代化并适配生成式模型**：通过引入深度分区、rollout 扩展与分布式启发函数，OCL 能高效引导生成模型进行高质量搜索。
2. **OCLGEN 显著优于 MCTS 和符号求解器**：在相同计算预算下，生成更短、更接近最优的计划，且完成率接近 100%。
3. **自适应扩展大幅提升效率**：避免在高置信路径上重复展开，节省大量计算资源。
4. **启发函数过估计可通过设计缓解**：使用下百分位估计 + 深度分区，有效抑制贪婪行为。
5. **OCLGEN 是强大的自增强基础**：其搜索结果可用于迭代训练，进一步提升模型性能。

---

### **方法的局限性**
1. **依赖预训练的次优数据**：若初始模型完全无效，搜索可能无法启动。
2. **对象外推能力有限**：当前 tokenization 方案无法处理训练中未见的对象名称（如新 block 名称）。
3. **深度选择策略固定**：目前使用 uniform 或 scan，未动态聚焦最有希望的深度。
4. **组合空间爆炸风险**：在极高分支因子问题中，即使自适应扩展也可能面临挑战。

---

### **未来工作方向**
1. **自适应深度选择策略**：根据历史搜索动态调整搜索重心。
2. **理论分析收敛性**：研究递归自增强是否能保证收敛到最优策略。
3. **零样本迁移与泛化**：支持更大规模、新对象、新领域的规划。
4. **无监督策略改进**：探索无需初始数据即可启动的 policy improvement 机制。
5. **扩展至连续或部分可观测环境**：结合 POMDP 或强化学习框架。

---

> **总结一句话**：  
> **OCLGEN 成功将经典搜索智慧与现代生成模型融合，在测试时实现了高效、高质量的规划推理，不仅超越了主流基线，还为模型自增强提供了强大引擎。**

</details>

---

### 14. [Thinking Economically: A Hierarchical Framework for Adaptive-Complexity Reasoning in LLMs](https://arxiv.org/abs/2606.01168)

**Authors**: Yubo Gao, Haotian Wu, Hong Chen, Junquan Huang, Yibo Yan, Jungang Li, Zihao Dongfang, Sicheng Tao, Puay Siew Tan, Jie Zhang, Xuming Hu  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.01168v1  

#### Abstract
Chain-of-Thought (CoT) has significantly enhanced LLM reasoning, yet often incurs substantial computational overhead due to "overthinking": generating excessively long rationales without commensurate accuracy gains. Existing efficiency methods typically apply uniform compression, which overlooks a c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Thinking Economically: A Hierarchical Framework for Adaptive-Complexity Reasoning in LLMs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Chain-of-Thought (CoT)** 推理方法虽然提升了 LLM 的推理能力，但普遍存在“**over-thinking**”现象——即生成过长的推理链，导致：
- 显著增加 **token usage** 和计算开销；
- 并未带来相应的准确率提升，甚至可能因冗余步骤引入逻辑错误。

现有压缩方法（如 TokenSkip、CoD）通常采用**统一的预定义约束**（uniform compression），忽略了两个关键维度上的复杂度异质性：
1. **跨问题（inter-step）**：不同难度的问题需要不同长度的推理路径；
2. **单个问题内部步骤间（intra-step）**：同一推理链中的不同步骤对计算资源的需求也不同。

这类静态策略在简单问题上浪费资源，在复杂问题上则可能截断必要推理。

---

### 🚀 提出的新方法与创新思路
作者提出 **“Thinking Economically”** 原则，并基于此设计了 **Hierarchical Adaptive Budgeter (HAB)** 框架，实现**自适应复杂度推理**。

#### 主要创新点：
1. **分层预算控制机制（coarse-to-fine budgeting）**
   - **Inter-step 控制**：预测每个问题所需的最优推理深度类别（short/medium/long），并转化为 step-range 指令（如 “用3-4步解决”）；
   - **Intra-step 控制**：为每个推理步骤动态分配 token 预算（retention ratio），依据其内在难度进行差异化压缩。

2. **自适应 Pareto 优化目标**
   - 引入 **Pareto Curvature Probe** 动态估计每一步的质量-效率权衡曲线斜率；
   - 自动调整损失权重：在“陡峭区”优先保质量，在“平坦区”追求高效压缩。

3. **基于 PPL 的步骤难度信号学习**
   - 利用 step-wise **Perplexity (PPL)** 作为复杂度代理指标；
   - 设计 ranking-based loss 学习相对难度，避免直接回归高方差问题。

4. **训练时细粒度指导**
   - 使用 **Fisher Information** 在训练阶段进行 token-level pruning，提供精细监督；
   - 最终将经济化推理行为内化到模型参数中，推理时不需显式剪枝。

---

### 🔍 相比现有方法的优势
| 维度 | HAB | 现有方法（如 TokenSkip, O1-Pruner） |
|------|-----|-------------------------------|
| 预算粒度 | 分层动态（问题级 + 步骤级） | 单一全局或固定比例 |
| 资源分配 | 智能按需分配 | 统一处理，易误伤 |
| 训练目标 | 自适应权衡（adaptive Pareto） | 固定加权或多任务RL |
| 可解释性 | 显式推理链保留 | 部分隐空间推理牺牲可读性 |
| 性能表现 | 同时提准降耗 | 通常牺牲一方换取另一方 |

> ✅ HAB 实现了真正的“**该想的时候多想，该省的时候少说**”。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **GSM8K**：小学数学应用题，共约2000条样本（train/val/test ≈ 1007/434/462）；
- **MATH500**：高中竞赛级别数学题，更具挑战性，共500题（split: 233/77/78）；
- 所有数据经过清洗，仅保留 Qwen-Max 能正确解答且有合理推理链的样本。

---

### ⚙️ 实验设置
- **Backbone Models**：
  - `Qwen2.5-7B-Instruct`
  - `Llama3.1-8B-Instruct`
- **训练方式**：
  - 两阶段 LoRA 微调（Stage 1: inter-step 分类；Stage 2: intra-step 预算学习）
  - 使用 AdamW，梯度裁剪，最大序列长度 512
- **硬件环境**：双 NVIDIA A800 GPU (80GB)

---

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy (%)** | Exact Match 准确率（答案完全匹配） |
| **Average Output Tokens** | 每个问题平均生成的 token 数量 |

> 关注 **performance-efficiency trade-off**，而非单一指标。

---

### 🆚 基线方法对比
| 类型 | 方法 | 简介 |
|------|------|------|
| **Vanilla CoT (Zero-Shot)** | Baseline | 无任何压缩的标准 CoT |
| **Pre-defined Constraint** | TokenSkip | 固定比例跳过低重要性 token |
| | CoD (Chain of Draft) | 生成极简草稿形式推理 |
| | SoT (Skeleton-of-Thought) | 先骨架后并行扩展 |
| | Sketch-of-Thought | 路由选择预设范式 |
| **Learning-based Constraint** | O1-Pruner | 基于 RL 的长度协调奖励机制 |
| | TALE (引用作背景) | 学习全局 token 预算 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 1 & 2）

#### ✅ GSM8K 结果（Qwen2.5-7B-Instruct）
| Method | Acc (%) | Tokens |
|--------|---------|--------|
| Vanilla CoT | 93.94 | 283.0 |
| TokenSkip | 90.04 | 165.9 |
| O1-Pruner | 84.63 | 198.7 |
| **HAB (Ours)** | **95.24** | **211.1** |

> 💡 **HAB 不仅准确率最高，还比 Vanilla CoT 少用了 ~72 tokens！**

#### ✅ MATH500 结果（Qwen2.5-7B-Instruct）
| Method | Acc (%) | Tokens |
|--------|---------|--------|
| Vanilla CoT | 79.49 | 482.2 |
| TokenSkip | 75.64 | 354.6 |
| O1-Pruner | 69.23 | 350.8 |
| **HAB (Ours)** | **82.05** | **327.3** |

> 💡 在更难的数据集上，HAB 依然实现了 **+2.56% 提升 +32% token 下降**。

---

### 🔁 跨模型验证（Llama3.1-8B-Instruct）
- HAB 在 Llama 上同样显著优于所有 baseline；
- 表明框架具有良好的 backbone 泛化能力。

---

### 🔍 消融实验（Ablation Study）

#### 实验目的：验证 HAB 中“动态推理步数规划”的必要性

| 固定步数配置（Fixed-Length） | GSM8K Acc (%) | Tokens | 对比 HAB |
|-----------------------------|---------------|--------|----------|
| 1-step                      | 95.02         | 172.6  | ↓ accuracy, ↓↓ tokens → 效率看似好但牺牲潜力 |
| 2-step                      | 95.89         | 310.9  | ↑ acc, ↑↑ tokens → 浪费严重 |
| 3-step                      | 95.45         | 316.3  | 类似问题 |

> ❗ 发现：**并非越多步骤越好**。例如在 MATH500 上，从1步增至2步反而导致 accuracy 下降，说明过度推理会引入噪声。

✅ 结论：HAB 的**自适应机制**能精准识别何时该深思、何时该简洁，实现真正的“经济思考”。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **推理复杂度是层次化的**：
   - 忽视 inter-step 和 intra-step 的异质性会导致次优压缩；
   - HAB 的分层预算机制有效捕捉这种双重差异。

2. **“Thinking Economically” 是可行且高效的**：
   - 通过智能资源分配，可以同时提升 accuracy 和 efficiency；
   - 实现了 **“better and cheaper”** 的理想状态。

3. **非单调性存在**：
   - 更多 reasoning steps ≠ 更高 accuracy；
   - 过度推理（over-thinking）确实损害性能。

4. **泛化能力强**：
   - 在 Qwen 和 Llama 系列上均有效；
   - 在 DeepSeek-R1-Distill 和 Phi-4-reasoning 等专用推理模型上也能降本增效（见 Figure 4）。

---

### ⚠️ 局限性
1. 当前仅适用于 **linear CoT 结构**，尚未拓展至 Tree-of-Thoughts 或 Graph-of-Thought 等非线性推理框架；
2. 自适应 Pareto 优化带来一定 **训练开销增加**；
3. 数据准备依赖外部强模型（如 Qwen-Max）生成候选链并过滤，流程较重；
4. 需要额外构建高质量的 supervision signal（如 optimal step count labels）。

---

### 🔮 未来工作方向
1. 将 HAB 扩展至 **non-linear reasoning frameworks**（如 ToT, GoT）；
2. 开发更轻量的 **自动化标注策略**，减少对外部模型的依赖；
3. 探索 **online budget adjustment** 机制，实现在推理过程中动态调节；
4. 结合 **multi-agent collaboration** 场景下的资源调度；
5. 推广至更多领域（如 code generation, planning, dialogue）中的经济化推理。

---

## ✅ 总结一句话
> HAB 提出了“**Thinking Economically**”的新范式，通过 **Hierarchical Adaptive Budgeting** 实现了对 LLM 推理过程的精细化资源调控，在多个数学推理 benchmark 上实现了 **准确性更高、token 消耗更低** 的双重突破，为高效、可控的复杂推理提供了新路径。

</details>

---

### 15. [Resonant Context Anchoring: Decoupling Attention Routing and Signal Gain at Inference Time](https://arxiv.org/abs/2606.01923)

**Authors**: Mingkuan Zhao, Yide Gao, Wentao Hu, Suquan Chen, Tianchen Huang, Zhenhua An, Zetao Chang, Xiayu Sun, Yuheng Min  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.01923v1  

#### Abstract
Large Language Models (LLMs) frequently exhibit "contextual disregard" when faced with input evidence that conflicts with their internal parametric memory, leading to persistent factual hallucinations. Existing mitigation strategies primarily rely on suppressing specific neuron activations or employ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Resonant Context Anchoring: Decoupling Attention Routing and Signal Gain at Inference Time

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在面对与内部参数化记忆（parametric memory）相冲突的输入证据时，常表现出“**contextual disregard**”（上下文忽视），即优先依赖其训练中内化的先验知识而非当前提供的事实，导致**factual hallucinations**（事实性幻觉）。这一现象严重削弱了模型在高可靠性、知识密集型任务中的可信度。

现有方法如 **Contrastive Decoding (CD)** 或 **Activation Engineering** 要么计算开销大（需多次前向传播），要么通过粗粒度抑制特定注意力头或神经元破坏语法连贯性和语言流畅性，影响整体生成质量。

---

### 🚀 提出的新方法与核心思想
作者提出 **Resonant Context Anchoring (RCA)** ——一种轻量级、无需训练的推理时干预方法，基于对 **residual stream 信号动态** 的分析，从**信号增益**（Signal Gain）角度重新理解并解决幻觉问题。

#### 核心创新点：
- **理论视角转变**：  
  幻觉并非源于模型无法识别相关上下文（路由错误），而是外部证据在深层网络传播过程中因能量不足而被淹没——即 **Signal-to-Noise Ratio (SNR) 过低**。
  
- **机制设计创新**：  
  在 self-attention 模块中实现 **“attention routing” 与 “signal gain” 的正交解耦**：
  - 保留原始 Softmax 注意力分布（维持语义正确性）
  - 利用 **raw pre-softmax attention scores** 构建一个非线性整流器（Softplus），生成动态增益场 $ \lambda_t $
  - 对应放大 value vectors 的范数（norm），增强上下文信号强度

> 🔍 公式简述：  
> $$
> \lambda_{t,i} = 1 + \gamma \cdot \text{Softplus}(s_{t,i}),\quad \tilde{v}_i = \lambda_{t,i} \cdot v_i
> $$

该机制不改变注意力权重分布，仅提升关键上下文 token 的信号能量，从而提高 SNR，使生成轨迹锚定于真实上下文。

---

### ⚖️ 相比现有方法的优势
| 方法 | 是否需训练 | 推理延迟 | 是否破坏语言结构 | 是否通用 |
|------|------------|----------|------------------|---------|
| Supervised Fine-Tuning | 是 | 高 | 可能过拟合 | 否 |
| Contrastive Decoding | 否 | 高（双倍 forward） | 否 | 是 |
| Activation Suppression | 否 | 低 | 是（可能损伤语法） | 有限 |
| **RCA (本文)** | ❌（training-free） | ✅（可忽略） | ❌（保持原分布） | ✅（plug-and-play） |

✅ **实现帕累托改进（Pareto improvement）**：在提升事实一致性的同时，未牺牲 fluency 和 general language capability。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验覆盖三类任务以全面评估性能：

| 类别 | 数据集 | 描述 |
|------|--------|------|
| **上下文保真度** | XSum | 单文档摘要任务，测试生成内容与原文的事实一致性 |
| **强知识冲突** | NQ-Swap | 替换维基百科中的实体（如国家/首都），迫使模型选择外部篡改上下文 vs 内部正确知识 |
| | MemoTrap | 测试模型是否能抵抗反直觉谚语的记忆回放 |
| **通用能力保留** | TruthfulQA, TriviaQA, PopQA | 封闭式问答（closed-book QA），无上下文输入，验证是否损害基础世界知识 |

---

### 🧪 实验设置与评估指标

#### 模型平台：
- Llama-3-8B-Instruct
- Llama-3-70B-Instruct

#### 基线方法：
- **Baseline**：标准贪婪解码（greedy decoding）

#### 评估指标：
| 任务 | 主要指标 | 含义 |
|------|----------|------|
| XSum | FactKB ↑, AlignScore ↑, ROUGE-L ↑ | 分别衡量事实精确性、语义对齐程度、内容覆盖率 |
| NQ-Swap | Exact Match (EM) ↑ | 是否完全遵循篡改后的上下文作答 |
| MemoTrap | Micro Acc / Macro Acc ↑ | 多项选择准确率 |
| TruthfulQA / TriviaQA | MC1/MC2, EM ↑ | 多选题与精确匹配得分 |

#### 实现方式：
- RCA 作为 **parameter-free plug-in 模块** 集成进 HuggingFace Transformers
- 所有实验运行于 NVIDIA A100 GPU 集群
- 超参数 $ \gamma $（resonance sensitivity coefficient）进行网格搜索优化

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Task | Metric | Llama-3-8B (Baseline → RCA) | Llama-3-70B (Baseline → RCA) |
|------|--------|-------------------------------|-------------------------------|
| **XSum** | FactKB | 47.61 → **50.14** (+2.53) | 61.32 → **61.72** (+0.40) |
| | AlignScore | 58.20 → **59.45** (+1.25) | 65.10 → **65.88** (+0.78) |
| | ROUGE-L | 19.90 → **20.03** | 22.41 → **22.21** （轻微下降但仍稳定） |
| **NQ-Swap** | EM | 60.62 → **64.54** (+3.92) | 76.11 → **77.46** (+1.35) |
| **MemoTrap** | Micro Acc | 64.40 → 65.77 | 66.52 → **77.35** (**+10.83**) |
| | Macro Acc | 65.86 → 66.69 | 68.47 → **75.58** (**+7.11**) |

> ✅ 所有指标均取得一致提升，尤其在 **Llama-3-70B 上的 MemoTrap 表现飞跃式增长**，说明 RCA 对大规模模型更有效。

---

### 🔍 参数敏感性分析（Table 2）
- $ \gamma $ 存在明显的“甜点区间”（sweet spot）：
  - Llama-3-8B 最优值为 **0.04**
  - Llama-3-70B 最优值为 **0.05**
- 当 $ \gamma > 0.08 $ 时性能开始显著下降
- $ \gamma = 0.12 $ 时甚至低于 baseline（如 8B 模型 EM 降至 52.10）

> 💡 发现：存在明确的安全边界（hard upper threshold ≈ 0.08），超出则引发输出退化（perplexity 上升）

---

### 🛠️ 消融实验与机制验证
虽然文中未设独立“消融表”，但通过以下方式验证机制有效性：
- **安全性测试（Table 3）** 显示，在无上下文任务中（如 TruthfulQA）：
  - MC1: 38.92 → 38.92（无变化）
  - MC2: 55.64 → 55.61（微小波动）
  - TriviaQA EM: 56.58 → 56.52
  - PopQA Acc: 26.64 → 26.59

> ✅ 证明 RCA 在无关查询下自动“休眠”（gain ≈ 1），不影响通用能力，具备自适应激活特性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **幻觉本质是信号衰减问题，而非路由失败**：  
   即便注意力正确指向上下文 token，若其 value vector 范数远小于 MLP 注入的 parametric noise，则信息仍会被淹没。

2. **RCA 成功实现了 SNR 提升**：  
   动态增益机制有效放大了与 query 语义共振的上下文信号，使其在 residual stream 中占据主导地位。

3. **无需训练、低延迟、高兼容性的推理增强方案**：  
   RCA 是首个在不增加推理成本的前提下，同时提升 **faithfulness 与 fluency** 的方法，并适用于不同规模的 Llama-3 模型。

4. **具有“always-on”部署潜力**：  
   因其在无上下文任务中几乎无副作用，可作为默认 decoding strategy 使用。

---

### ⚠️ 方法的局限性
1. **超参数 $ \gamma $ 需调优**：  
   不同模型尺寸需调整最佳 $ \gamma $ 值，尽管趋势稳定（8B: 0.04, 70B: 0.05），但仍需少量开发集校准。

2. **理论假设简化**：  
   将 residual stream 视为线性子空间叠加是对高度非线性 Transformer 动态的近似，虽实证有效，但理论深度有待加强。

3. **当前评估集中于文本证据**：  
   尚未扩展至多模态或结构化输入场景，应用范围有待拓展。

---

### 🔮 未来工作方向
1. **自动化 $ \gamma $ 调节机制**：  
   设计基于置信度或上下文复杂度的自适应 gain 控制策略。

2. **跨架构迁移验证**：  
   将 RCA 应用于其他 Transformer 架构（如 Mistral、Phi、Qwen 等），验证其 architecture-agnostic 特性。

3. **结合 retrieval-augmented generation (RAG)**：  
   与 RAG 系统集成，进一步强化外部知识注入路径。

4. **探索更多物理干预手段**：  
   基于 residual stream 的信号工程视角，发展一系列可解释、可控的认知纠偏工具。

---

## 总结
> **RCA 开辟了一条全新的路径来应对 LLM 的事实性幻觉问题——不是压制内部记忆，而是增强外部证据的“音量”。它将 attention 机制的功能解耦为“去哪里听”和“听多大声”，并通过简单的 element-wise 操作实现了高效、安全、即插即用的事实锚定。这不仅是技术上的突破，更是对 Transformer 内部工作机制理解的一次深化。**

🔗 代码已开源：[https://github.com/yidGao/RCA-Implementation](https://github.com/yidGao/RCA-Implementation)

</details>

---

### 16. [DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding](https://arxiv.org/abs/2606.02091)

**Authors**: Jiebin Zhang, Zhenghan Yu, Song Liu, Eugene J. Yu, Zheng Li, Dawei Zhu, Jiangshan Duo, Weimin Xiong, Yifan Song, Guanghua Yu, Jianchen Zhu, Sujian Li  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.02091v1  

#### Abstract
Block diffusion speculative decoding accelerates LLM inference by predicting all tokens within a block simultaneously for the target model to verify in parallel. Predicting an entire block at once requires a sufficiently capable draft model and effective utilization of the target model's internal kn...

---

### 17. [Dive into Ambiguity: A*-Inspired Multi-Agents Commonsense Obfuscation Attack on LLM Prompts](https://arxiv.org/abs/2606.01441)

**Authors**: Boxuan Wang, Zhuoyun Li, Xiaowei Huang, Yi Dong  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.01441v1  

#### Abstract
Large language models (LLMs) excel in reasoning and knowledge-intensive tasks but remain vulnerable to prompt-level adversarial attacks that preserve intent while triggering commonsense hallucinations. This vulnerability is urgent, as LLMs are rapidly integrated into safety-critical domains where fa...

---

### 18. [ExpWeaver: LLM Agents Learn from Experience via Latent RAG](https://arxiv.org/abs/2606.01041)

**Authors**: Tao Feng, Tianyang Luo, Jingjun Xu, Zhigang Hua, Yan Xie, Shuang Yang, Ge Liu, Jiaxuan You  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.01041v1  

#### Abstract
Experience learning has achieved promising results in enhancing LLM agent planning and reasoning by integrating past interactions as reusable knowledge. However, existing methods remain confined to explicit text space, retrieving experiences via semantic similarity and concatenating them into the co...

---

### 19. [AcOrch: Accelerating Sampling-based GNN Training under CPU-NPU Heterogeneous Environments](https://arxiv.org/abs/2606.01161)

**Authors**: Kefu Chen, Xin Ai, Qiange Wang, Yanfeng Zhang, Ge Yu  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.01161v1  

#### Abstract
Graph Neural Networks (GNNs) have achieved remarkable success in various applications. Sampling-based GNN training, which conducts mini-batch training on sampled subgraphs, has become a promising solution for large-scale graphs. Given the resource-intensive nature of sampling-based GNN training, Neu...

---

### 20. [Don't Let a Few Network Failures Slow the Entire AllReduce](https://arxiv.org/abs/2606.01680)

**Authors**: Peiqing Chen, Jiedong Jiang, Nengneng Yu, Yuefeng Wang, Sixian Xiong, Wei Wang, Zaoxing Liu  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.01680v1  

#### Abstract
Network failures are among the most frequent hardware faults in large-scale GPU clusters and a leading cause of training-job interruptions. Modern collective communication libraries such as NCCL mitigate network failures by rerouting traffic through surviving NICs on the same server, trading reduced...

---

### 21. [LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](https://arxiv.org/abs/2606.00228)

**Authors**: Yao Lai, Xuyuan Xiong, Zeyue Xue, Guojin Chen, Jing Wang, Xihui Liu, Rui Zhang, Robert Mullins, Bei Yu, Ping Luo  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.00228v1  

#### Abstract
In semiconductor manufacturing, lithography projects circuit layouts onto silicon wafers through an optical mask. As circuit features shrink below the wavelength of light, optical diffraction causes the printed patterns to deviate from their intended layouts. Inverse Lithography Technology (ILT) add...

---

### 22. [ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](https://arxiv.org/abs/2606.00494)

**Authors**: Wneya Yu, Chao Zhang, Li Wang, Samson Lasaulce, Merouane Debbah  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.00494v1  

#### Abstract
Post-Training Quantization (PTQ) and Low-Rank Adaptation (LoRA) constitute the standard pipeline for efficient Large Language Model (LLM) deployment. However, applying them sequentially poses a problem: PTQ often leaves behind random noise that is spread out (across the model's weights) in a way LoR...

---

### 23. [Latent Diffusion Pretraining for Crystal Property Prediction](https://arxiv.org/abs/2606.00776)

**Authors**: Shrimon Mukherjee, Kishalay Das, Partha Basuchowdhuri, Pawan Goyal, Niloy Ganguly  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.00776v1  

#### Abstract
Fast and accurate prediction of crystal properties is a central challenge in new materials design. Graph neural networks and Transformer-based models have emerged as powerful tools for this task due to their ability to encode the local structural environment of atoms within a crystal. However, these...

---

### 24. [Hybrid Neural Ordinary Differential Equations for Data-Efficient Polymerization Modeling with Incomplete Kinetics](https://arxiv.org/abs/2606.02145)

**Authors**: Marah Almanasreh, Alexander Mitsos, Eike Cramer  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.02145v1  

#### Abstract
Accurate prediction of polymerization dynamics is essential for process design, control, and optimization. Yet, purely mechanistic models require labor-intensive parameterization of partially characterized kinetics, while purely data-driven models demand large, diverse datasets that are costly to ob...

---

### 25. [A combination of noise and bilateral filters achieve supralinear and scalable adversarial robustness in CNNs](https://arxiv.org/abs/2606.02267)

**Authors**: Nicolas Stalder, Benjamin F. Grewe, Matteo Saponati, Pau Vilimelis Aceituno  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.02267v1  

#### Abstract
The vulnerability of deep neural networks to adversarial examples poses a significant challenge for real-world deployment. Existing techniques to enhance deep network robustness rely on adversarial training, an approach that is powerful but computationally intensive and typically tailored to specifi...

---

### 26. [SIRI: Self-Internalizing Reinforcement Learning with Intrinsic Skills for LLM Agent Training](https://arxiv.org/abs/2606.02355)

**Authors**: Zhongyu He, Yuanfan Li, Fei Huang, Tianyu Chen, Siyuan Chen, Xingyang Li, Meng Hsuan Yu, Xiangrong Liu, Leyi Wei, Lu Pan, Ke Zeng, Xunliang Cai  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.02355v1  

#### Abstract
Long-horizon LLM agents can benefit from reusable skills, yet existing skill-based methods often rely on external skill generators during training or persistent skill retrieval at inference, increasing engineering complexity, context length, and deployment latency. We propose Self-Internalizing Rein...

---

### 27. [Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention](https://arxiv.org/abs/2606.01243)

**Authors**: Shuochen Chang, Tong Bai, Xiaofeng Zhang, Qianli Ma, Qingyang Liu, Zhaohe Liao, Yibo Miao, Li Niu  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.01243v1  

#### Abstract
Latent reasoning enables Large Language Models (LLMs) to perform multi-step inference within continuous hidden states, offering efficiency gains over explicit Chain-of-Thought (CoT). However, the opacity of these continuous thought vectors hinders their reliability and controllability. This paper br...

---

### 28. [LongAttnComp: Cross-Family Context Compression for Long-Context Reasoning](https://arxiv.org/abs/2606.01336)

**Authors**: Mengmeng Ji, Ravi Shanker Raju, Jonathan Lingjie Li, Chen Wu  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.01336v1  

#### Abstract
As real-world applications increasingly require processing inputs of 100k+ tokens, the gap between context length and inference efficiency has become a critical bottleneck. Context compression offers a way to reduce prefill costs while preserving task accuracy. However, existing training-free attent...

---

### 29. [Observation, Not Prediction: Conversation-Level Disaggregated Scheduling for Agentic Serving](https://arxiv.org/abs/2606.01839)

**Authors**: Jianru Ding, Ryien Hosseini, Pouya Mahdi Gholami, Mingyuan Xiang, Henry Hoffmann  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.01839v1  

#### Abstract
LLM-based agents resolve a user task through many turns of dependent inference and tool calls, producing a workload whose total cost is unknown when the task arrives. Existing multi-turn systems keep the turn as the scheduling unit and decide, turn by turn, whether to disaggregate prefill from decod...

---

### 30. [Strategies for Molecular Dynamics using Hybrid Systems: LAMMPS Use Case](https://arxiv.org/abs/2606.02319)

**Authors**: Paulo Henrique Leme Ramalho, Dennis Alves Pedersen, F\'abio Andrijauskas  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.02319v1  

#### Abstract
The complexity of biomolecular simulations has substantially increased the demand for High-Performance Computing (HPC) infrastructures, particularly in molecular dynamics and coarse-grained modeling. This work presents a systematic performance and scalability analysis of the LAMMPS simulator for coa...

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

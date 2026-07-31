# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-31 08:24:33 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [A Sparse Glimpse of the Whole: Train-Free Self-Speculative Decoding](https://arxiv.org/abs/2607.27735)

**Authors**: Yuesong Liu, Yuan Zeng, Min Lyu, Ruilin Liu, Yu Guo, Yinlong Xu  
**Category**: cs.CL  
**Published**: 2026-07-31  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.27735v1  

#### Abstract
Speculative decoding alleviates the memory-bandwidth bottleneck in large language model inference, but its acceleration is jointly constrained by drafting overhead, token acceptance, and speculation length. We present a unified efficiency analysis showing that extending the speculation horizon can r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Sparse Glimpse of the Whole: Train-Free Self-Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLM）在长上下文场景下的自回归解码过程面临严重的**内存带宽瓶颈**，导致推理延迟高。尽管 speculative decoding 能缓解该问题，但其加速效果受限于三个因素：
- **Drafting overhead**（起草开销）
- **Token acceptance rate**（接受率）
- **Speculation length $k$**（推测长度）

现有方法在这三者之间存在固有矛盾，难以同时优化。

### 🚀 提出的新方法：SparseSpec-L
提出了一种**无需训练的自推测解码框架 SparseSpec-L**，专为长上下文推理设计，具备以下三大创新：

#### （1）**效率分析理论（Efficiency Analysis）**
- 将 speculative decoding 的加速比 $S$ 形式化为 drafting cost $\gamma = C_d/C_v$、acceptance rate $\alpha$ 和 speculation length $k$ 的联合函数。
- 揭示“**效率反转（efficiency inversion）**”现象：当边际接受概率低于相对起草成本时（$\alpha_{\text{tail}} < \gamma$），继续增加 $k$ 反而会降低速度。
- 结论：固定 $k$ 是次优策略，必须动态调整。

#### （2）**可召回稀疏上下文自推测（Recallable Sparse-Context Self-Speculation）**
- 使用**目标模型自身作为 drafter 和 verifier**，避免辅助模型带来的结构不匹配。
- 在 drafting 阶段使用**动态稀疏化的 KV Cache**，保留关键历史 token（sink、recent、important historical）。
- 利用上一轮 full-context verification 中产生的 **per-head attention 统计信息**构建重要性评分，用于更新稀疏索引。
- 不永久丢弃任何 KV 状态，实现“**无额外前向传播的重要性信号提取**”。

#### （3）**基于熵的成本感知自适应推测长度控制（Cost-Aware Adaptive Speculation）**
- 实时记录每个 draft token 的输出 **entropy**，并维护 accepted 与 rejected 类别的移动平均熵值。
- 利用 softmax 对距离建模，估计每个 token 的软接受概率 $p_i$。
- 动态选择最优 $k^*$ 来最大化预期 step-wise 效率：
  $$
  k^* = \arg\max_{k \in \mathcal{K}} \frac{1 + \sum_{m=1}^{k} \prod_{i=1}^{m} p_i}{k C_d + C_v}
  $$

### 🔍 相比现有方法的优势
| 方法类型 | 主要缺陷 | SparseSpec-L 如何改进 |
|--------|--------|----------------------|
| **Auxiliary Models** | 结构差异大 → 接受率低 | 自模型推测 → 消除结构错配 |
| **Prediction Heads (如 Medusa)** | $k$ 固定于架构层 | 支持灵活、自适应 $k$ |
| **Lightweight Layers (如 Eagle-3)** | 长 $k$ 下尾部接受率骤降 | 利用 attention 回收机制维持高接受率 |
| **Static Compression (如 MagicDec)** | 永久剪枝 → 上下文丢失 | 可召回稀疏索引 → 保持上下文保真度 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **LongBench v2**：多任务长上下文理解与推理基准
- **InfiniteBench**：超长上下文评估（>100K tokens）
- **RULER QA1 / NIAH**：针探测（needle-in-a-haystack）任务，测试长程依赖捕捉能力
- 涵盖：问答、合成、检索、推理等多样化任务

### ⚙️ 实验设置
- **模型**：
  - Llama-3.1-8B-Instruct
  - Qwen2.5-7B-Instruct
  - 扩展至 Llama-3.2-3B 和 Qwen2.5-14B 进行泛化性验证
- **硬件**：单张 NVIDIA A40 GPU（48GB）
- **实现库**：Hugging Face Transformers
- **上下文长度上限**：64K tokens
- **生成长度**：128 tokens
- **Prefill 阶段保持一致**，仅修改 decoding 阶段

### 🎯 评估指标
| 指标 | 含义 |
|------|------|
| **Speedup ($S$)** | 相比 autoregressive decoding 的端到端吞吐提升倍数 |
| **Throughput (T)** | tokens/s |
| **Acceptance Rate ($\alpha$)** | 平均每步接受的 draft token 占比 |
| **Average Accepted Tokens (Aver.k)** | 每步平均提交的有效 draft token 数量 |
| **End-to-End Acceleration** | 完整生成流程的速度提升 |

### 🆚 基线方法对比
| 基线 | 类型 | 是否需训练 | 备注 |
|-----|------|-----------|------|
| **Autoregressive** | 原始逐个生成 | — | 基准线 |
| **Auxiliary Model** | 辅助小模型起草 | 是 | 如 SpecInfer |
| **EAGLE-3** | 轻量级层推测 | 是 | 训练敏感，跨数据集性能差 |
| **MagicDec (StreamLLM/SnapKV)** | 静态压缩 KV | 否 | SnapKV 在短 $k=2$ 表现好但不可扩展 |
| **RAPID** | 检索增强推测 | 是 | 需外部模块 |
| **LayerSkip** | 层跳过自推测 | 否 | 接受率低，加速有限 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

| 方法 | 数据集 | 模型 | Acceptance | Aver.k | Throughput (tokens/s) | Speedup |
|------|--------|-------|------------|--------|------------------------|---------|
| **Autoregressive** | LongBench v2 | L3 | 100% | 1.00 | 3.71 | 1.0× |
| **SparseSpec-L (Ours)** | LongBench v2 | L3 | **72.9%** | **11.26** | **10.38** | **2.79×** |
| **MagicDec (SnapKV)** | LongBench v2 | L3 | 80.1% | 1.59 | 6.46 | 1.74× |
| **Auxiliary Model** | LongBench v2 | L3 | 50.3% | 2.94 | 4.53 | 1.22× |
| **EAGLE-3** | LongBench v2 | L3 | 1.95% | 1.07 | 3.30 | 0.89× |

> 💡 在 InfiniteBench 上也达到 **2.60×** 加速，Qwen2.5-7B 上达 **1.95×**

#### 泛化性表现（Table 2）
| 模型 | 数据集 | Acceptance | Speedup |
|------|--------|------------|---------|
| Llama-3.2-3B | LongBench v2 | 82.0% | 1.91× |
| Llama-3.2-3B | RULER NIAH | 83.0% | 1.96× |
| Llama-3.2-3B | InfiniteBench | 100% | **2.41×** |
| Qwen2.5-14B | LongBench v2 | 84.6% | 1.55× |

✅ 表明 SparseSpec-L 在不同规模（3B–14B）、不同类型任务中均有显著加速。

### 🔬 消融实验结果（Ablation Study）

#### （1）自适应 $k$ vs 固定 $k$（Table 3）
| Speculation Length | Acceptance | Throughput | Speedup |
|--------------------|-----------|-----------|---------|
| Fixed $k=4$ | 93.4% | 7.81 | 2.10× |
| Fixed $k=8$ | 86.7% | 8.89 | 2.39× |
| Fixed $k=12$ | 78.5% | 9.99 | **2.69×** |
| Fixed $k=16$ | 72.6% | 9.90 | 2.66× |
| Fixed $k=20$ | 67.9% | 9.75 | 2.62× |
| **Adaptive (Ours)** | **72.9%** | **10.38** | **2.79×** |

📌 发现：
- 固定 $k$ 存在“效率反转”：超过 $k=12$ 后 throughput 开始下降。
- 自适应策略自动避开低效区域，在无需人工调参下达到全局最优。

#### （2）敏感性分析（Figure 4）
- **随 context length 增加（10K → 60K）**：
  - SparseSpec-L 的 speedup 持续上升，且 acceptance 稳定。
  - MagicDec(SnapKV) 在 $k=10$ 时因永久剪枝导致接受率急剧下降。
- **压缩比 sensitivity**：
  - SparseSpec-L 在多种压缩比率下均优于 baseline。
  - 最佳压缩比约为 **10%**（即保留 10% KV entries 用于 drafting）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **效率反转是真实存在的瓶颈**：盲目增大 speculation length $k$ 会导致加速退化，必须结合接受率与成本进行权衡。
2. **self-speculative + recallable sparse KV 是高效路径**：
   - 使用目标模型自身起草 + 可回收 attention 信号，可在不牺牲分布保真度的前提下大幅提升接受率。
3. **entropy 是有效的在线控制信号**：
   - 低 entropy token 更可能被接受（AUC=0.638），支持其作为轻量级预测器。
4. **端到端加速可达 2.79×**：
   - 在多个模型和任务上稳定实现 **1.55× ~ 2.79×** 的加速，显著优于所有基线。

### ⚠️ 方法的局限性
1. **当前实现未融合 kernel**：
   - 使用非融合的 dense attention verifier 和 sparse gather，带来额外 kernel 开销。
   - 存在进一步优化空间（如 FlashAttention 集成）。
2. **entropy 控制器预测能力有限**：
   - entropy 仅提供中等程度的 accept/reject 分离，非完全校准的概率估计。
3. **实验范围受限**：
   - 仅在单卡 A40 上测试，最长 context 64K，生成 128 tokens。
   - 未在生产级 serving engine（如 vLLM、TensorRT-LLM）中部署验证。

### 🔮 未来工作方向
1. **Fused kernel 优化**：开发支持 sparse-gather + FlashAttention 的一体化核函数以减少验证延迟。
2. **更精确的 acceptance 预测器**：探索基于浅层 logits 或中间表示的 early-exit 判别器。
3. **扩展至更大 batch 和更长输出**：研究 batch-level 自适应控制策略。
4. **集成到主流推理框架**：推动 SparseSpec-L 成为通用 speculative decoding 插件。

---

## 总结
SparseSpec-L 是一种**无需训练、基于可召回稀疏注意力的自推测解码框架**，通过理论驱动的设计解决了 speculative decoding 中 drafting cost、acceptance rate 与 speculation length 的三重权衡问题。其实验表明，在保持目标模型输出分布的同时，实现了高达 **2.79× 的端到端加速**，并在多个模型和任务上展现出卓越的泛化性和稳定性，代表了长上下文 LLM 推理加速的一个重要进展。

</details>

---

### 2. [SmartGen: Seamless Disaggregated LLM Inference with Selective KV Cache Transfer](https://arxiv.org/abs/2607.28150)

**Authors**: Xuchuan Luo, Jiacheng Shen, Xin Wang, Yangfan Zhou  
**Category**: cs.DC  
**Published**: 2026-07-31  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.28150v1  

#### Abstract
Disaggregating the prefill and decoding stages of large language model (LLM) inference into two separate sets of nodes is widely adopted in today's LLM serving systems. However, such an architecture poses significant challenges for self-hosted LLM deployments on rented cloud instances, since transfe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《SMARTGEN: Seamless Disaggregated LLM Inference with Selective KV Cache Transfer》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在当前主流的 **P/D disaggregation（Prefill/Decoding Disaggregation）架构**中，大型语言模型（LLM）推理被拆分为两个阶段：
- **Prefill 阶段**：处理输入提示（prompt），计算所有 token 的 Key-Value（KV）缓存。
- **Decoding 阶段**：逐个生成输出 token，依赖于 Prefill 阶段产生的 KV 缓存。

这两个阶段通常部署在不同的硬件节点上以优化资源利用率。然而，在云环境中，这种架构面临一个严重瓶颈：**KV cache transfer overhead**。由于 KV cache 大小随序列长度、batch size 和模型规模增长，跨节点传输会占用大量有限的网络带宽（如 25 Gbps RDMA），导致 **stage-transition stall** ——即从 Prefill 到 Decoding 的过渡延迟显著增加，直接影响用户体验的关键指标 **Time-to-Second-Token (TTST)**。

### **提出的新方法与思路**
为缓解这一问题，论文提出了 **SMARTGEN**，一种基于 **重要性感知的 KV cache 选择性传输引擎**，其核心思想是：
> 并非所有 KV entries 在 Decoding 阶段都同等重要，因此无需全量传输，而是只传输“关键”部分，并通过多路径机制实现无缝衔接。

SMARTGEN 设计了三条并行的数据传输路径：

1. **Profile-based Proactive Transfer（基于配置文件的主动传输）**
   - 在离线阶段对多种 workload 进行 profiling，识别出“普遍重要”的 KV entries（例如某些固定位置的 token 总是被 attention 选中）。
   - 在 Prefill 阶段提前将这些高优先级 KV blocks 推送到 Decoding 节点。

2. **Parallel On-demand Transfer（并行按需传输）**
   - 对于上下文相关的 KV entries，在 Decoding 阶段动态请求。
   - 利用 **mask-based index splitting** 和 **reordering-based gathering** 技术，使得本地加载与远程拉取并行执行，避免网络往返阻塞关键路径。

3. **Speculative Transfer（推测性传输）**
   - 在系统空闲时（如 attention 计算期间），后台持续推送剩余的低重要性 KV entries。
   - 利用闲置网络带宽摊销总传输开销，且不影响主流程性能。

### **相比现有方法的优势**
| 方法 | 局限性 | SMARTGEN 的优势 |
|------|--------|------------------|
| **Full KV Transfer**<br>(e.g., Mooncake, Splitwise) | 全量传输易饱和网络，TTST 高 | 显著降低 TTST，最高达 4.3× 加速 |
| **Quantization-based**<br>(e.g., HACK) | 使用 2-bit 量化牺牲精度，尤其影响长文本理解任务 | 保持接近全缓存的 accuracy，不依赖量化 |
| **Vanilla Selective Transfer** | 缺乏高效 fetch 机制，on-demand 请求引入高延迟 | 并行 fetch + 推测传输，有效隐藏延迟 |

此外，SMARTGEN 是 **正交于现有稀疏注意力算法**（如 InfiniGen、HATA）的设计，可与其无缝集成。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LongBench**：一个多任务、双语、支持长上下文的基准测试集。
  - 包含四个代表性子任务用于评估：
    - **MultiFieldQA**（文档问答）
    - **GovReport**（摘要生成）
    - **SAMSum**（对话摘要）
    - **LCC**（代码补全）
- **Calibration Dataset**：使用 LongBench 中未参与评估的子集 **2WikiMultihopQA** 作为离线 profiling 的训练集。

### **实验设置**
- **硬件平台**：阿里云 GPU 实例
  - **Prefill 节点**：3 × `ecs.gn8is-2x.8xlarge`（各含 2×NVIDIA L20 GPU，共 6 GPU）
  - **Decoding 节点**：1 × `ecs.gn8is.4xlarge`（1×L20 GPU，25 Gbps RDMA）
- **模型范围**：
  - Llama-3.1-8B, Qwen3-8B, Qwen3-14B, Gemma-3-12B, Phi-4-14B
- **序列长度**：默认批处理 60K tokens，最长支持 96K
- **KV Offloading**：KV cache 存放于 host memory，模拟真实受限环境

### **评估指标**
| 指标 | 含义 |
|------|------|
| **TTST (Time-to-Second-Token)** | 用户收到第一个 token 后等待第二个 token 的时间，反映 stage-transition stall 程度 |
| **TBT (Time-Between-Tokens)** | 解码过程中连续 token 之间的平均间隔，衡量解码效率 |
| **CTL (Cumulative Token Latency)** | 每个请求的整体响应延迟曲线，体现端到端体验 |
| **Accuracy (%)** | 在 LongBench 上的任务准确率，对比全缓存 baseline |

### **基线方法对比**
1. **Full Transfer**：标准做法，Prefill 阶段逐层传输全部 KV cache
2. **Partial Transfer**：仅传输前 $ K_r $ 个 KV blocks（按内存顺序），其余 on-demand 获取
3. **HACK**：基于 2-bit homomorphic quantization 的 KV 压缩方案
4. **SMARTGEN (InfiniGen / HATA)**：本文方法，结合两种 KV selection 算法验证通用性

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **TTST 改进**
- SMARTGEN 将 **TTST 最多降低 4.3×**，优于所有基线。
- 在 **GovReport** 任务上表现最佳，因该任务 prompt 较长（~10K tokens），传统方法瓶颈更明显。
- 即使在网络带宽降至 **15 Gbps** 时，SMARTGEN 仍能维持显著优势（TTST 提升 3.3× vs Full Transfer）。

#### ✅ **TBT 表现**
- SMARTGEN 的 **TBT 接近理想情况（Full Transfer）**，远优于 Partial Transfer 和 HACK。
- 在 batch size=6 时：
  - Partial Transfer 的 TBT 比 SMARTGEN 高 **1.5×**
  - HACK 因格式转换开销，TBT 比 SMARTGEN 高 **1.4×**

#### ✅ **CTL 曲线平滑**
- Full Transfer 出现明显的初始延迟 spike（stage-transition stall）
- SMARTGEN 实现近乎线性的 CTL 增长，表明推理过程“无缝”（seamless）

#### ✅ **准确性保持**
- SMARTGEN 使用 InfiniGen 或 HATA 的 KV selection，**accuracy 与 full-cache baseline 相当**（相对误差 < ±3%）
- HACK 因量化损失，在 LongBench 上 accuracy 下降明显（最低至 54%），尤其在需要精确信息提取的任务中表现差

### **消融实验结果**
| 组件 | 效果 |
|------|------|
| **+ Profile-based Transfer** | TBT 降低 1.03–1.1×，减少 on-demand fetch 数量 |
| **+ Parallel On-demand Transfer** | TBT 再降 1.1–1.2×，消除关键路径上的网络等待 |
| **+ Speculative Transfer** | TBT 进一步下降 1.3×，几轮后趋近理想值；选择 10% speculative ratio 可平衡速度与干扰 |

> 图 15 显示：profile-based selection 比随机/顺序策略减少 **最多 51% 的 on-demand ratio**，证明其有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **KV cache 存在显著的重要性分布不均现象**，可通过 profiling 发现“普遍重要”的 token 位置（positional similarity），从而指导选择性传输。
2. **Stage-transition stall 是自托管 LLM 推理在低成本云实例上的主要瓶颈**，尤其是在长上下文场景下。
3. **Selective KV transfer 是可行且高效的解决方案**，结合 proactive、parallel on-demand 和 speculative 三重机制，可在不牺牲 accuracy 的前提下大幅优化 TTST。
4. SMARTGEN **兼容多种 KV selection 算法和模型架构**，具备良好的泛化能力。

### **方法的局限性**
- **依赖离线 profiling**：虽然支持周期性更新和分组建模，但在极端变化的 workload 下可能需要重新校准。
- **对极低端网络（<15 Gbps）仍有压力**：metadata 和前两层 KV cache 的传输仍可能成为瓶颈。
- **未完全消除 on-demand fetch**：尽管已并行化，但在高度动态的 attention pattern 下仍有一定开销。

### **未来工作方向**
- **在线自适应 profiling**：构建轻量级运行时反馈机制，动态调整 KV selection 策略。
- **与 quantization 正交结合**：将 SMARTGEN 与 HACK 类方法融合，进一步压缩传输体积。
- **扩展至其他 disaggregation 场景**：如 MoE routing、multi-GPU tensor parallelism 中的中间状态传输优化。

---

> 🔚 **总结一句话**：  
> **SMARTGEN 通过“有选择地传、并行地拉、推测性地补”，实现了在低带宽环境下依然流畅的 disaggregated LLM inference，解决了困扰自托管系统的 stage-transition stall 问题，同时兼顾了性能、效率与准确性。**

</details>

---

### 3. [SDO: Structure-Aware Data Organization for Efficient LLM Post-Training](https://arxiv.org/abs/2607.27273)

**Authors**: Jinliang Gao, Ning Yang, Hai Wang, Baili Xiao, Pin Lyu  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.27273v1  

#### Abstract
Post-training of large language models is expensive, and existing efficiency improvements mainly focus on selecting informative samples or designing training schedules. However, data organization itself is usually treated as a static preprocessing step: embedding-based grouping methods construct fix...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SDO: Structure-Aware Data Organization for Efficient LLM Post-Training

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
大型语言模型（LLM）的 **post-training**（如 SFT、DPO、GRPO）过程计算成本高昂，现有优化方法主要集中在 **data selection**（选择重要样本）或 **curriculum scheduling**（设计训练顺序），但忽略了 **mini-batch 组织方式** 对优化效率的影响。

传统基于 embedding 的分组方法采用 **静态划分**（static partitioning），在训练前固定样本分组，无法适应训练过程中样本暴露（exposure）的变化。这导致：
- 部分样本被过度训练（over-exposed）
- 部分样本训练不足（under-optimized）
- 批次内梯度冲突（gradient conflict）增加，收敛变慢

### 🚀 提出了什么新方法或新思路
提出 **SDO（Structure-Aware Data Organization）** ——一种轻量级、即插即用（plug-and-play）的数据组织框架，具有以下两个核心机制：

1. **Locality-aware Batching（局部感知批处理）**
   - 在每个 epoch 内，基于冻结的外部 embedding 构建 KNN 图
   - 通过遍历近邻图形成语义连贯的 mini-batch，提升批次内梯度一致性

2. **Exposure-balanced Scheduling（曝光均衡调度）**
   - 跨 epoch 维护一个全局 **exposure ledger**，记录每个样本参与训练的次数
   - 动态重构训练池：高曝光样本按逆频率采样降低保留概率，低曝光样本完整保留（cold set）
   - 实现长期数据多样性保持，避免“富者愈富”效应

> 🔁 **闭环反馈机制**：历史曝光信息被反馈用于重构下一轮训练池，形成动态、自适应的数据流。

### ⭐ 相比现有方法的优势
| 特性 | SDO | 传统方法（如 EP-Order, LESS） |
|------|-----|-----------------------------|
| 是否需要 warm-up 训练 | ❌ 否（使用冻结 embedding） | ✅ 是（依赖模型动态特征） |
| 是否支持多范式 | ✅ 支持 SFT / DPO / GRPO | ❌ 多数仅限 SFT |
| 是否永久过滤数据 | ❌ 否（临时调整，保证覆盖） | ✅ 是（丢弃部分样本） |
| 是否动态更新结构 | ✅ 每 epoch 重建 KNN | ❌ 固定分区 |
| 是否控制曝光平衡 | ✅ 显式调节 | ❌ 忽略 |

---

## 2. 核心实验方法和设置

### 📚 使用了哪些数据集
| 方法 | 数据集 | 样本数量 | 任务描述 |
|------|--------|----------|---------|
| **GRPO** | GSM8K | 6,796 训练样本 | 数学推理任务，验证集为 500 样本子集，测试集 1,319 样本 |
| **DPO & SFT** | UltraFeedback | 6,000 偏好对（过滤后 5,908） | 偏好优化与监督微调，评估 reward margin 和 test loss |

### 🧪 实验设置和评估指标
- **模型**：Qwen-3.5-4B
- **Embedding 模型**：`zembed-1-embedding`（2560维，C2归一化），仅对 prompt 编码并冻结
- **训练配置**：
  - 单卡 RTX 4090（48GB）
  - 默认参数：`K=4`, `△T=2`, `r=0.2`
  - 所有实验三种子种子（419, 617, 917）平均结果
- **评估指标**：
  - **主指标**：准确率（Acc）、reward margin、test loss
  - **公平性诊断**：per-cluster accuracy（基于 prompt embedding 聚类）、B-20%、CV、Gini、max-min gap
  - **梯度行为分析**：pairwise gradient coherence、embedding-gradient correlation

### 🆚 基线方法对比
- **Baseline**：uniform shuffling（标准随机打乱）
- **对比目标**：在完全相同的训练流程中，仅替换数据组织策略，验证 SDO 的独立增益

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据与基线对比

#### （1）训练效率提升（加速收敛）

| 方法 | 指标 | Baseline | SDO | 提升幅度 | 观察阶段 |
|------|------|---------|-----|----------|---------|
| **GRPO @3k步** | Acc ↑ | 76.24±1.04 | **78.09±1.52** | **+1.85** | 早期至中期 |
| **GRPO 最终** | Acc ↑ | 82.41±0.20 | **82.89±0.50** | +0.48 | 全程持续 |
| **DPO @2k步** | Margin ↑ | 0.264±0.008 | **0.307±0.012** | +0.043 | 持续扩大 |
| **SFT @200步** | Loss ↓ | 1.526±0.011 | **1.503±0.003** | 更快下降 | 早期显著 |

> ✅ **SDO 在所有三种 post-training 范式中均显著加速收敛**，尤其在 **early-to-mid phase** 效果最明显。

#### （2）运行开销极低
- GRPO 完整训练耗时：Baseline 1786 分钟 vs. SDO 1799 分钟
- **额外开销仅 +0.7%**，证明其高效性和实用性

---

### 🔍 消融实验结果（Ablation Studies）

使用 GRPO/GSM8K 设置进行组件拆解（seed=617）：

#### 表：消融变体在 5k 步的性能对比（Accuracy）

| 变体 | 3k Acc | 5k Acc | B-20% Acc | Gini |
|------|--------|--------|-----------|------|
| Baseline | 77.18 | 82.34 | 55.94 | 0.1100 |
| SDO w/o dynamic KNN | 74.68 | 82.94 | 56.20 | 0.1116 |
| SDO w/o locality | 78.01 | 82.79 | 54.20 | 0.1149 |
| SDO w/o exposure | 78.62 | 83.78 | **53.04** | 0.1185 |
| **Full SDO** | **79.45** | **83.92** | **61.38** | **0.0952** |

#### 关键发现：
- **Locality-aware batching 是早期加速的关键**  
  → 移除后 mid-phase 性能大幅下降（79.45 → 78.01）
- **Exposure balancing 决定了最终覆盖质量**  
  → 无 exposure 控制时 B-20% 下降、Gini 上升，说明出现“偏科”
- **Dynamic KNN 至关重要**  
  → 固定 KNN 图（w/o dynamic KNN）在 mid-phase 明显落后（74.68 vs 77.18），因其无法适应动态数据分布变化

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **mini-batch 组织方式是影响 post-training 效率的重要维度**，不应被视为静态预处理步骤。
2. **representation-space locality 与 gradient coherence 强相关**：语义相近样本组成 batch 可减少梯度冲突，提升更新有效性。
3. **exposure-driven feedback 机制可有效维持长期数据多样性**，防止优化器陷入局部高频模式。
4. **SDO 是通用、轻量、无需修改 loss 或 schedule 的 plug-and-play 框架**，适用于 SFT、DPO、GRPO 等多种范式。
5. **理论与实证一致**：Theorem 1 预测的梯度幅值提升在实际训练中得到验证（Figure 4 & 5 显示更高 coherence 和正相关性）。

### ⚠️ 方法的局限性
- 依赖高质量的 **external sentence encoder**（如 zembed）。若 embedding 不能反映任务语义，则 locality 效应减弱。
- 当前实现基于 **frozen embedding**，未考虑 prompt-response 联合表示（未来可扩展）。
- 超参数敏感性存在（如 K、△T、r），虽默认值表现良好，但在极端任务中可能需调整。

### 🔮 未来工作方向
1. 将 SDO 扩展到 **token-level training dynamics** 或 **RLHF 中的 episode-level 组织**
2. 探索 **learnable 或 adaptive encoder** 替代冻结 embedding，进一步提升结构感知能力
3. 结合 **active learning** 或 **uncertainty estimation**，实现更智能的 exposure 调控
4. 应用于更大规模模型（如 70B+）或多模态 setting，验证泛化性

---

> 💡 **一句话总结**：  
> SDO 揭示了“**如何组织数据流**”这一被忽视的优化杠杆，通过 **structure-aware batching + exposure-balanced feedback** 实现更高效、更均衡的 LLM post-training，且几乎零代价、即插即用。

</details>

---

### 4. [First-order Constrained Trilevel Optimization Over Distributed Networks for Robust Coreset Selection](https://arxiv.org/abs/2607.27632)

**Authors**: Yang Jiao (Richard), Kaixuan Jiao (Richard), Kai Yang (Richard), Nadjib Aitsaadi (Richard), Ilhem Fajjari (Richard),  Renwei (Richard),  Li  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.27632v1  

#### Abstract
With the rapid advancement of the Internet of Things (IoT), massive amounts of data are generated across distributed edge networks. Training models on full data incurs significant computational overhead and storage bottlenecks, rendering coreset selection a critical paradigm. Furthermore, given the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：First-order Constrained Trilevel Optimization Over Distributed Networks for Robust Coreset Selection

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **分布式网络中的鲁棒 coreset 选择**（distributed robust coreset selection）这一尚未被充分探索的关键挑战，提出了一种全新的优化框架。具体而言，该问题涉及三个核心需求：
- **计算效率**：在边缘设备上处理海量 IoT 数据时避免高昂的训练开销；
- **隐私保护**：不集中原始数据的前提下进行 coreset 选择；
- **模型鲁棒性**：所选 coreset 能提升模型对对抗攻击的防御能力。

现有方法要么忽略鲁棒性，要么依赖中心化设置，无法满足上述综合要求。

---

### 🚀 提出的新方法与新思路
作者提出了 **F2CTO**（Federated First-order Constrained Trilevel Optimization），这是首个专为解决 **带层级约束的分布式三层次优化问题** 设计的算法。其核心创新包括：

#### （1）将分布式鲁棒 coreset 选择建模为 **带 level-wise constraints 的 trilevel optimization 问题**
- **Level 1（Data Weight）**：优化数据权重 $\alpha$ 来选择代表性样本（即 coreset）；
- **Level 2（Model Training & Evaluation Perturbation）**：训练鲁棒模型 $w$ 并生成评估阶段的对抗扰动 $q$；
- **Level 3（Training-side Perturbation）**：生成训练阶段的对抗扰动 $p$，增强鲁棒性。

每一层都引入了显式的约束（如 $\|\cdot\|_\infty \leq c_1$），构成复杂的耦合结构。

#### （2）提出 **Hierarchical Composite Value-function Reformulation**
通过构造内层值函数 $V_3(\cdot)$ 和两个外层值函数 $V^{(1)}(\cdot), V^{(2)}(\cdot)$，将原三层次嵌套问题转化为单层约束优化问题，便于分布式求解。

#### （3）设计 **Distributed Alternating Projected Gradient Algorithm**
一种无需计算 hypergradients 的 first-order 分布式算法，在每个 worker 上交替更新变量，并仅传输模型参数而非原始数据，保障隐私。

---

### 🔍 相比现有方法的优势

| 维度 | F2CTO 的优势 |
|------|--------------|
| **问题建模** | 首次统一整合 coreset selection、robust optimization 与 distributed learning，形成 trilevel 框架 |
| **算法设计** | 是首个支持 **level-wise constraints** 的分布式 trilevel 优化算法；避免高成本 hypergradient 计算 |
| **理论保证** | 提供非渐近收敛率分析：达到 $\epsilon$-stationary point 的迭代复杂度为 $O(\epsilon^{-3/2})$，通信复杂度为 $O(d \cdot \epsilon^{-3/2})$ |
| **适用场景** | 支持联邦学习架构下的跨设备（cross-device）大规模部署 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **Permuted MNIST**
- **Split CIFAR-100**
- **Tiny-ImageNet**
- **Edge-IIoTset**（用于可扩展性测试的大规模 IoT 安全数据集）

所有实验均采用分布式设定，数据按客户端划分。

---

### ⚙️ 实验设置与评估指标

#### 模型配置
- Permuted MNIST：MLP 模型
- 其他数据集：ResNet-18
- 使用两个独立模型分别用于 coreset selection 和最终训练

#### 评估指标
- **平均对抗鲁棒性**（Average Robustness %）：
  - 在三种典型攻击下测试：**FGSM**, **PGD-10**, **AutoAttack**
- 运行时间与内存占用（效率对比）
- 可扩展性：在最多 200 个 worker 上测试性能稳定性

#### 基线方法对比
| 类别 | 对比方法 |
|------|--------|
| **分布式 coreset selection** | FedCS [18], GCFL [19] |
| **集中式 coreset selection**（适配到分布式） | BCSR [9], Greedy Coreset [42], ACS [16] |
| **分布式 trilevel optimization** | AFTO [7], DTZO [24] |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table III）

| 方法 | Permuted MNIST (PGD) | Split CIFAR-100 (PGD) | Tiny-ImageNet (PGD) |
|------|------------------------|--------------------------|-------------------------|
| BCSR [9] | 46.25 ± 0.87 | 38.04 ± 0.81 | 28.30 ± 0.83 |
| FedCS [18] | 45.17 ± 0.86 | 41.97 ± 0.36 | 32.29 ± 0.53 |
| AFTO [7] | 46.16 ± 0.41 | 43.75 ± 0.69 | 33.51 ± 0.59 |
| **F2CTO（本文）** | **51.10 ± 0.51** | **46.32 ± 0.85** | **39.32 ± 1.57** |

> ✅ **F2CTO 在所有数据集和攻击类型下均显著优于所有基线方法**，平均提升约 **3–6% 的鲁棒准确率**

---

### 🔬 消融实验结果（Ablation Study）

作者比较了以下变体以验证三层次结构的有效性：
- **UBV**（Upper-Bilevel Variant）：保留 Level 1 和 Level 2，移除第三级扰动生成
- **LBV**（Lower-Bilevel Variant）：保留 Level 2 和 Level 3，移除第一级 coreset selection（使用贪心策略代替）

#### 结果（见 Fig. 2）：
- F2CTO 显著优于 UBV 和 LBV
- 表明 **完整的三层次结构是必要的**：缺少任何一级都会导致性能下降
- 特别地，**Level 3 的对抗扰动生成对于鲁棒性至关重要**

---

### ⏱ 效率对比（见 Fig. 4）
- **运行时间更短**：相比基于 hypergradient 的方法（如 TSG [10]），F2CTO 因避免高阶导数计算而大幅降低开销
- **内存使用更低**：first-order 更新机制减少缓存压力
- **通信效率更高**：支持 periodic communication（F2CTO+P），可在每 $I=5$ 轮本地更新后通信一次，仍能快速收敛（见 Fig. 5）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **三层次联合建模有效提升了分布式环境下的鲁棒 coreset 质量**  
   —— 将 coreset selection、robust training 与 distributed coordination 统一在一个框架中，实现了全局最优协同。

2. **F2CTO 是首个适用于 constrained trilevel optimization 的分布式 first-order 算法**  
   —— 突破了传统方法无法处理层级约束的技术瓶颈。

3. **理论与实践一致性强**  
   —— 非渐近收敛率 $O(\epsilon^{-3/2})$ 得到实验证实，且通信与计算开销可控。

4. **具备良好的可扩展性和实用性**  
   —— 在 Edge-IIoTset 上验证了其在大规模 IoT 场景中的稳定表现。

---

### ⚠ 局限性
- 当前分析假设目标函数满足 L-smoothness，可能不适用于某些非光滑 loss（如 hinge loss）；
- 所有 worker 需同步参与每轮通信，未考虑异步或部分参与场景；
- 实验集中在图像分类任务，需进一步验证在 NLP 或时序预测等领域的泛化能力。

---

### 🔮 未来工作方向
1. 扩展至 **异步联邦学习** 架构，适应现实网络延迟；
2. 探索 **adaptive constraint tuning** 机制，动态调整 $c_1, c_2, c_3$；
3. 将 F2CTO 应用于其他多层级安全任务，如 **federated adversarial training** 或 **secure model update**；
4. 结合 **compression techniques** 进一步降低通信成本。

---

> 💡 **总结一句话**：  
> F2CTO 成功构建了一个面向分布式鲁棒 coreset 选择的 first-order constrained trilevel 优化框架，兼具高效性、隐私保护与强鲁棒性，为未来边缘智能系统的可靠持续学习提供了新范式。

</details>

---

### 5. [Recall Before You Rank: Similarity-Guided Top-$K$ Reuse for Efficient Long-Context Attention](https://arxiv.org/abs/2607.27692)

**Authors**: Wenshuai Yao, Wenyong Zhou, Hanyong Shao, Yizhe Chen, Zhiyuan Ning, Yuannuo Feng, Ru Huang, Kechao Tang  
**Category**: cs.CL  
**Published**: 2026-07-31  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.27692v1  

#### Abstract
Top-$K$ sparse attention reduces the cost of Softmax and value aggregation by attending to only a small subset of key--value (KV) entries. However, identifying this subset still requires scoring the current query against the full KV cache and performing global Top-$K$ selection, leaving selector cos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Recall Before You Rank: Similarity-Guided Top-K Reuse for Efficient Long-Context Attention*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在长上下文（long-context）场景下，**Exact Top-K sparse attention** 虽然通过限制 Softmax 和 value aggregation 到前 K 个最高分的 KV 对来降低计算开销，但其**索引发现过程仍需对整个 KV cache 进行 full-history QK scoring 和全局 Top-K 选择**，导致该步骤的时间复杂度随上下文长度线性增长，成为实际效率瓶颈。

现有方法如 KV-cache 压缩、Streaming 或近似检索虽能减少缓存大小或搜索空间，但往往以牺牲历史信息完整性为代价（如丢弃 token），或仍需扫描大规模候选集。

---

### 🚀 提出的新方法：**ReTopK**

ReTopK 是一种**无需训练**（training-free）的动态 Top-K attention 加速方法，核心思想是：

> **“Recall Before You Rank”** —— 先基于查询相似性从历史中召回可能相关的支持集（support），再仅对这些候选进行精确重排序。

#### 核心机制：
- **Per-head Query-Support Cache**  
  每个 attention head 维护一个有限容量的 FIFO 缓存，存储历史查询向量 $ q_c $ 及其对应的 Exact Top-K 支持索引集合 $ S_c $。
  
- **Similarity-Guided Recall**  
  对当前查询 $ q_t $，计算其与缓存中所有历史查询的余弦相似度，选取最相似的 R 个历史查询。

- **Candidate Construction**  
  将这些相似查询对应的支持集合并，并加入一个最近窗口（recent window）中的 token，形成紧凑候选集 $ \mathcal{A} $。

- **Exact Reranking**  
  仅在候选集 $ \mathcal{A} $ 上执行当前查询的 QK scoring 和 Top-K 选择，得到最终支持集用于稀疏 attention。

- **可靠性保障机制**：
  - **Similarity-based Fallback**：若最大缓存相似度低于阈值 $ T $，则回退到 full-history Exact Top-K。
  - **Periodic Refresh**：每隔 $ T_r $ 步强制执行一次 Exact Top-K，防止缓存漂移。

#### 不同于其他方法的关键设计：
- **只复用索引（indices），不复用 scores/weights/outputs** → 保持输出一致性。
- **完整保留原始 KV cache** → 无信息丢失，支持全历史访问。
- **完全无需微调或额外训练** → 即插即用。

---

### 🔍 相比现有方法的优势
| 方面 | ReTopK | 其他方法（如 StreamingLLM, Quest, Loki 等） |
|------|--------|---------------------------------------------|
| **信息完整性** | ✅ 完整保留 KV cache | ❌ 通常剪枝或压缩 cache，丢失潜在重要位置 |
| **搜索成本** | ✅ 复用历史决策，避免 full scan | ⚠️ 多数仍需扫描 growing context 或 large page set |
| **适用性** | ✅ 通用、无需训练、跨模型迁移良好 | ⚠️ 部分依赖特定结构或训练 |
| **精度-效率权衡** | ✅ 可控 fallback + refresh 保证质量 | ⚠️ 固定策略可能导致误差累积 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **PG19**：用于语言建模任务，评估 perplexity（PPL），测试长度覆盖 16K–128K。
- **RULER NIAH**：包含 single-needle、multi-key、multi-value 检索任务，评估模型在长上下文中定位关键信息的能力。
- **LongBench**：多语言、多任务长文本理解基准，包括 2WikiMQA、HotpotQA、Multi-FieldQA 子集，共 550 个样本。

---

### ⚙️ 实验设置
- **模型**：
  - 主要使用 **Qwen2.5-7B** 和 **Qwen2.5-7B-Instruct-1M**
  - 跨模型泛化测试还用了 **Llama-3.1-8B** 和 **Qwen2.5-14B**
- **上下文长度**：16K, 32K, 64K, 128K（部分扩展至 5M）
- **Top-K 设置**：默认 $ K = 512 $，部分实验用 $ K = 1024 $
- **ReTopK 默认参数**：
  - 缓存大小 $ C = 32 $
  - 最近窗口 $ W = 32 $
  - 检索数量 $ R = 4 $
  - 相似度阈值 $ T = 0.85 $
  - 刷新周期 $ T_r = 128 $

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Perplexity (PPL)** | 衡量语言建模质量，越低越好 |
| **NIAH Score** | 平均检索准确率（macro-average） |
| **LongBench Score** | 多任务平均得分 |
| **Speedup vs. Exact Top-K** | 注意力计算加速比 |
| **Support Recall / Attention Mass Retention** | 分析选出的支持集与 Exact Top-K 的重合程度及注意力质量保留情况 |

---

### 🆚 基线方法对比
- **Full Attention**：标准 dense attention
- **Exact Top-K**：作为质量上限参考
- **StreamingLLM**：固定 sink + recent window
- **Quest**：基于查询感知的 page-level pruning
- **SparQ**：维度级 key 投影剪枝
- **Loki**：低秩 key 空间近似检索
- **TokenSelect**：跨步复用相同查询的选择结果

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1 & 2）

#### 在 **128K 上的表现（K=512）**
| 方法 | PPL | Speedup | NIAH Avg | LongBench |
|------|-----|---------|----------|-----------|
| **Exact Top-K** | 12.07 | 1.00× | 64.5 | 52.5 |
| **ReTopK (T=0.85)** | **12.13** (+0.50%) | **3.07×** | **85.3** | 50.1 |
| **ReTopK (T=0.90)** | 12.20 (+1.08%) | 2.25× | **96.3** | **52.7** ✅ |

> 💡 **亮点**：ReTopK 在 **LongBench 上超越 Exact Top-K 达 52.7 > 52.5**，且在 NIAH 上显著领先。

#### 跨模型泛化表现（Table 2）
在未调参情况下应用于 Llama-3.1-8B 和 Qwen2.5-14B：
- **平均复用率 82.5%~89.6%**
- **PPL 增加不超过 +2.76%**
- **速度提升 1.26× ~ 2.66×**

表明 ReTopK 具有良好的**跨架构、跨规模迁移能力**。

---

### 🔬 消融实验（Table 3 & Figure 7）

| 组件 | 移除/修改后影响 |
|------|----------------|
| **Similarity-guided retrieval** | 若改为仅检索最近 R 个，则 exact-path 比例上升但 PPL 更差、speedup 下降 |
| **Multi-query union ($ R=1 $)** | 支持覆盖率下降 → PPL 显著升高至 9.370（↑0.34） |
| **Small cache ($ C=1 $)** | 无法有效召回 → PPL 升高，speedup 下降 |
| **No similarity fallback ($ T=-1 $)** | 强制复用 → PPL 恶化至 10.076（↑1.05），但 speedup 略升至 2.36× → 显示 fallback 对质量至关重要 |
| **No periodic refresh ($ T_r=0 $)** | 短期无明显影响，长期可能积累误差（见 Figure 8） |
| **No recent window ($ W=0 $)** | 新增 token 可能被遗漏 → PPL 暴涨至 716.0 ❌ |

> ✅ 结论：所有组件协同作用，尤其 **fallback 和 recent window 至关重要**。

---

### 🧪 注意力支持保真度分析（Figure 5–6）
- **单头示例**：ReTopK 实现 **78.9% 支持召回率**，保留 **99.45% 注意力质量（attention mass）**，attention 分布余弦相似度达 **99.95%**。
- **全体 head-token 对分析**（1.6M pairs）：
  - 平均保留 **92.4% attention mass**
  - 输出余弦相似度 **97.4%**
  - 即使在 reuse 路径上，也达到 **91.6% mass retention**

> 表明即使索引不完全一致，**关键注意力分布仍高度还原**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **查询相似性是高效复用 Top-K 支持的有效信号**：相似查询倾向于关注重叠的位置，联合多个历史支持即可高概率覆盖当前最优支持。
2. **“召回 + 重排序”范式可大幅降低索引发现成本**：将 $ O(Ld) $ 的 full-history 扫描降为 $ O(Cd + M_td) $，其中 $ M_t \ll L $。
3. **ReTopK 在极高压缩比下仍能保持高质量输出**：在 128K 上实现 **3.07× 加速**，仅带来 **0.50% PPL 上升**，甚至在某些任务上反超基线。
4. **方法具备强鲁棒性和可移植性**：无需训练、参数固定、跨模型表现稳定。

---

### ⚠️ 局限性
- **依赖 query similarity 的局部稳定性**：对于语义跳跃剧烈的序列，相似度可能骤降，触发 fallback，降低加速效果。
- **缓存命中率受限于 $ C $ 和 $ R $**：过小的缓存难以捕捉远距离相关性；增大则增加内存开销。
- **近期窗口大小 $ W $ 影响新 token 可见性**：若 $ W $ 太小，刚生成的 token 可能无法及时参与 attention。
- **GPU 实现依赖 fused kernel 优化**：未融合版本性能增益有限，工程实现门槛较高。

---

### 🔮 未来工作方向
1. **自适应缓存管理**：根据 head-wise 行为动态调整 $ C, R, T $。
2. **引入轻量预测模块**：学习何时更应 fallback 或 refresh，替代固定阈值。
3. **跨层共享缓存**：探索不同 layer 之间的支持集冗余性，进一步节省元数据。
4. **结合 KV compression**：在保留完整 KV 的前提下，压缩 key/value 表示以降低存储带宽。
5. **扩展至 prefill 阶段**：目前 focus 在 decoding，但 prefill 同样面临长序列挑战。

---

## 总结一句话

> **ReTopK 通过“回忆历史决策 + 相似性引导召回 + 精确重排序”的方式，在不丢弃任何 KV 条件下实现了高达 3.07× 的 Top-K attention 加速，同时几乎无损模型性能，是迈向高效长上下文推理的重要一步。**

</details>

---

### 6. [Beyond KV Reconstruction: Functional Reconstruction for MLA Draft Models in Speculative Decoding](https://arxiv.org/abs/2607.27269)

**Authors**: Weiye Shi, Fanxu Meng, Muhan Zhang  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.27269v1  

#### Abstract
Multi-head latent attention (MLA) is increasingly important for long-context LLM inference because compact latent states replace the growing key-value (KV) cache and reduce decoding memory traffic. Yet most capable open checkpoints use multi-head or grouped-query attention (MHA/GQA), so conversion i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond KV Reconstruction: Functional Reconstruction for MLA Draft Models in Speculative Decoding*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **MLA (Multi-head Latent Attention)** 转换方法（如 TransMLA 和 MHA2MLA）旨在将传统的 MHA/GQA 模型转换为具有更小 **KV cache** 开销的 MLA 架构，以提升长上下文推理效率。然而，这些方法在用于 **speculative decoding**（推测解码）时表现不佳。

问题在于：
- 虽然转换后的 MLA 模型在独立生成任务中表现良好（structural validity），但在作为 **draft model** 时，其输出 token 分布与目标模型（target model）不一致。
- 这种不一致导致 **token acceptance rate** 显著下降，从而削弱了 speculative decoding 的加速效果。

根本原因被归结为：
- **Low-rank factorization** 和 **RoPE handling** 引入了 attention function 层面的误差，即使缓存结构有效，也无法保证功能等价。

### 🚀 提出的新方法：Functional Reconstruction
作者提出了一种新的训练后优化方法 —— **Functional Reconstruction**，其核心思想是：

> 将 MLA 转换视为一个 **functional reconstruction**（函数重建）问题，而非单纯的 **cache-compression**（缓存压缩）问题。

#### 方法要点：
- **目标**：让转换后的 MLA attention 模块在给定相同输入时，尽可能复现原始 MHA/GQA 模块在 `post-Wo` 阶段的输出响应。
- **实现方式**：
  - 在转换完成后，对 MLA 中由转换器引入的 **query 和 KV 投影参数** 进行微调。
  - 使用一组 **calibration hidden states** 作为输入，计算 MLA 输出与冻结的原始 GQA 模块输出之间的 **masked MSE loss**。
  - 只更新 MLA 参数，原始模块保持冻结，且不依赖 target model 的任何监督信号。
- **特点**：
  - **converter-agnostic**：适用于 TransMLA 或 MHA2MLA 转换后的模型。
  - **inference-time zero-cost**：不改变推理图、缓存结构或部署开销。
  - **end-to-end (E2E)**：优化的是从输入到 `post-Wo` 的完整 attention 函数，联合修复 low-rank 和 RoPE 错误。

### 🔍 相比现有方法的优势
| 方面 | 传统方法（如 Partial RoPE） | 本文方法（Functional Reconstruction） |
|------|----------------------------|----------------------------------------|
| 优化目标 | 局部投影重建（如仅恢复部分 RoPE） | 全局函数匹配（post-Wo 输出一致性） |
| 是否使用 verifier | 否（但功能上不面向 speculative） | 否（完全无监督，无需 verifier 参与） |
| 是否影响推理 | 否 | 否（纯训练时优化） |
| 通用性 | 依赖特定转换流程 | 支持多种 converter（TransMLA / MHA2MLA） |
| 效果 | 接受率低 | 显著提升 token acceptance |

---

## 2. 核心实验方法和设置

### 📚 数据集
使用四个代表性 benchmark，覆盖不同任务类型：
- **HumanEval**：代码生成
- **Alpaca**：指令遵循
- **Natural Questions (NQ)**：开放域问答
- **CNN/DailyMail (CNN/DM)**：摘要生成

每个 benchmark 包含 **200 个 prompt**，最大生成长度为 **128 tokens**。

### ⚙️ 实验设置
- **模型组合**：
  - **Llama 系列**：Llama-1B / 3B → Llama-8B
  - **Qwen 系列**：Qwen-1.5B / 3B → Qwen-7B
- **转换方法**：
  - **TransMLA**
  - **MHA2MLA**
- **推理框架**：
  - **Hugging Face (HF)**
  - **vLLM**
- **评估模式**：
  - 所有运行使用 `BF16`、batch size 1、`greedy decoding`
  - Proposal length：1B/1.5B draft 使用 γ=4；3B draft 使用 γ=3
  - 固定随机种子 `seed42`

### 🎯 评估指标
- **Token Acceptance Rate (%)**：speculative decoding 中被 target 接受的 draft token 比例。
- **Output Throughput (tok/s)**：每秒生成的有效 token 数量。
- **容忍阈值**：
  - 差异 < 0.5 pp（percentage points）视为“实际无变化”
  - 差异 < 0.5 tok/s 视为“吞吐量不变”

### 🔁 对比方法
1. **Original GQA**：未转换的原始 GQA draft，作为性能上限参考。
2. **Partial RoPE Reconstruction**：各转换器自带的局部重建方法，作为直接基线。
3. **Functional Reconstruction (Ours)**：本文提出的端到端函数重建方法。

所有方法在相同模型、转换器、backend、task 下进行对比，确保公平。

---

## 3. 主要实验结果和性能指标

### 📊 总体性能（64个 matched task cells）
在 **64 个可比较的任务单元**（matched task cells）中：

| 结果类别 | 单元数量 | 说明 |
|---------|--------|------|
| **显著提升 acceptance** | **37** | 相比 Partial RoPE 提升 >0.5 pp |
| **基本不变** | **26** | 差异 <0.5 pp |
| **显著下降** | **1** | 仅 Qwen-3B + TransMLA + vLLM + CNN/DM 下降 0.55 pp |

👉 **结论**：在 **37/38** 个发生变化的单元中，本文方法均优于基线，展现出极强的一致性和有效性。

### 🔝 最大提升案例
- **+4.23 pp**：Llama-1B, TransMLA, vLLM, CNN/DM
- **+3.92 pp**：Llama-1B, TransMLA, HF, HumanEval
- **+3.78 pp**：Llama-1B, TransMLA, HF, NQ
- **+3.72 pp**：Llama-1B, TransMLA, vLLM, Alpaca

### 🔄 吞吐量表现
- **12 个单元** 吞吐量显著提升（+0.5 tok/s 以上）
- **50 个单元** 基本不变
- **2 个单元** 略有下降（最大 -1.01 tok/s）

👉 在 **37 个 acceptance 提升的单元中，有 12 个也实现了吞吐量提升**，表明 acceptance 提高能有效转化为实际速度增益。

### 🧪 消融分析（关键发现）
- **Functional Reconstruction 不增加推理成本**：所有优化发生在训练阶段，推理图、缓存大小、参数量均不变。
- **提升来自函数逼近质量改善**：通过 post-Wo 匹配，减少了 attention logits 的扰动，提升了 top-token 对齐概率。
- **效果具有跨任务泛化性**：在代码、指令、问答、摘要四类任务中均有提升，说明校准 hidden states 能覆盖多种语义分布。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Conversion ≠ Draft Utility**
   - 成功的 MLA 转换（structural validity）不等于好的 draft model。
   - 必须额外优化 **function fidelity** 才能保障 speculative decoding 的 high acceptance。

2. **Functional Reconstruction 是有效的训练时控制手段**
   - 无需 verifier 监督、无需修改推理架构，即可显著提升 draft quality。
   - 本质是在固定 MLA 函数空间内搜索更接近原 GQA 行为的参数配置。

3. **方法具有强通用性**
   - 在 **TransMLA / MHA2MLA**、**Llama / Qwen**、**HF / vLLM** 上均有效。
   - 特别地，在 Llama 上 TransMLA 改进明显，在 Qwen 上 MHA2MLA 改进更多，说明效果取决于 **converter 初始化残差**，而非方法本身偏好。

4. **Backend 实现会影响最终收益**
   - vLLM 中由于使用了不同的 tensor parallelism（TP2 vs TP1），部分优化可能被 runtime 抹平。
   - 表明 **conversion-runtime co-design** 至关重要。

### ⚠️ 局限性
1. **无法弥补严重的信息损失**
   - 若 latent rank 过低或 converter-backend 路径严重不匹配（如 Qwen-TransMLA + vLLM），则 functional reconstruction 无法完全恢复性能。
   - 当前方法受限于已有的参数化能力。

2. **未超越 Original GQA 性能**
   - 尽管显著优于 Partial RoPE，但大多数情况下仍未达到原始 GQA draft 的 acceptance 水平。
   - 表明仍有改进空间。

3. **实验规模有限**
   - 仅测试了 ≤8B 模型、四种任务、单一 seed 和 prompt 数量。
   - 缺乏对更大模型、更长上下文、多轮对话场景的验证。

### 🔮 未来工作方向
1. **Layer-wise sensitivity weighting**：根据不同层对最终输出的影响加权重建损失。
2. **Sequence-level functional matching**：从单层重建扩展到短序列级行为对齐。
3. **Rank allocation optimization**：基于 functional residual 动态分配 latent rank。
4. **Conversion-Runtime Co-design**：联合设计转换策略与推理引擎，避免部署时优化丢失。
5. **探索更强的 verifier-free objective**：如 margin-aware reconstruction，进一步逼近最优 acceptance。

---

## 总结
该论文揭示了一个关键洞察：**MLA 转换的目标不应止步于缓存压缩，而应追求功能保真度，尤其是在 speculative decoding 场景下**。提出的 **Functional Reconstruction** 方法以零推理代价、完全无监督的方式，在多种模型、转换器和框架上显著提升了 draft model 的 token acceptance，为高效 LLM 推理提供了实用且通用的技术路径。

</details>

---

### 7. [Beyond Binary Rewards: A Comparative Study of Reward Design for Reinforcement Unlearning](https://arxiv.org/abs/2607.27968)

**Authors**: Efstratios Zaradoukas, Davide Gabrielli, Bardh Prenkaj, Gjergji Kasneci  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.27968v1  

#### Abstract
Machine unlearning seeks to selectively remove specific knowledge from trained language models without full retraining, a growing necessity under privacy regulations such as GDPR and the EU AI Act. Recent work has reformulated unlearning as a Reinforcement Learning with Verifiable Rewards (RLVR) pro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Beyond Binary Rewards: A Comparative Study of Reward Design for Reinforcement Unlearning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 **Reinforcement Unlearning (RUL)** 方法（如 PURGE）依赖于**稀疏的二元奖励**（binary reward），即仅判断模型输出是否包含被遗忘概念。这种奖励信号过于稀疏，导致训练过程中学习信号微弱、收敛速度慢，限制了 RUL 在复杂任务上的可扩展性。

本文提出并系统研究了一个核心问题：  
> **如何设计更有效的、可验证的（verifiable）奖励函数以提升 RUL 的效率？**

### **提出的新方法与新思路**
作者提出了一个**通用的奖励设计框架**，解耦了“可验证性”与“稀疏性”，并在此基础上引入两种新的奖励函数：

1. **Exponential Reward（指数奖励）**  
   - 基于生成文本中**禁止内容出现次数**提供渐进惩罚。
   - 公式：`φ_exp(y; F; T) = exp(-count / T)`，其中 `count` 是违反项总数，`T` 为衰减常数。
   - 提供比 binary 更密集的梯度信号。

2. **PageRank-inspired Reward（PageRank 启发式奖励）**  
   - 利用语义图结构建模禁止概念之间的关系（通过 embedding 构建相似性图）。
   - 使用个性化 PageRank 计算每个概念的重要性权重，对高重要性概念的泄露施加更大惩罚。
   - 实现“语义感知”的选择性遗忘优化。

此外，论文还提出了**奖励分解框架**（Reward Decomposition Framework），将奖励分为两个独立部分：
- **信息提取（Information Extraction）**：从输出中提取何种信息（如是否存在、数量、结构等）
- **奖励映射（Reward Transformation）**：如何将信息转化为 [0,1] 区间内的奖励值

该框架表明：**可验证性 ≠ 稀疏性**，从而打破传统假设。

### **相比现有方法的优势**
- 显著加快遗忘过程：在达到相同遗忘效果时，训练步数减少 **最多达 3×**。
- 维持甚至略微提升遗忘质量（Forget Score 更低）。
- 不损害模型通用能力（Utility 指标稳定）。
- 所有奖励仍保持 **fully verifiable** ——无需访问原始训练数据即可计算。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **RWKU (Real World Knowledge Unlearning) Benchmark** [Jin et al., 2024]
  - 包含 100 位真实世界名人作为遗忘目标。
  - 分为四个评估子集：
    - **Forget Set**：衡量遗忘效果（FB, QA, AA 探针）
    - **Neighbor Set**：测试邻近知识保留情况
    - **MIA Set**：基于 Membership Inference Attack 的隐私泄露检测
    - **Utility Set**：评估通用能力影响（GA, RA, TRU, FAC, FLU）

### **实验设置**
- **模型**：Phi-3-Mini-4K-Instruct（3.8B 参数）
- **优化算法**：Group Relative Policy Optimization (**GRPO**)
- **训练步数**：最多 1500 步
- **每轮采样组大小（Group Size）**：G = 8
- **KL Penalty Weight**：β = 0.001
- **PPO Clipping Threshold**：ε = 0.2

### **评估指标**
| 类别 | 指标 | 方向 |
|------|------|-------|
| 遗忘效果 | FB, QA, AA（ROUGE-L Recall） | ↓ 越低越好 |
| 泛化保留 | Neighbor-FB, Neighbor-QA | ↑ 越高越好 |
| 隐私保护 | MIA-FM↑, MIA-RM↓ | 差距越大越好 |
| 模型效用 | GA (MMLU), RA (BBH), TRU, FAC, FLU | ↑ 越高越好 |

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **BASE** | 原始未进行遗忘处理的模型 |
| **Binary (PURGE)** | 使用原始二元奖励的方法，作为主要 baseline |
| **Exponential Decay** | 本文提出的指数奖励方法 |
| **PageRank Softmax** | 本文提出的语义加权奖励变体 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1 & Figure 2）**

#### ✅ **遗忘性能（Forget Split）**
| 方法 | FB ↓ | QA ↓ | AA ↓ |
|------|------|------|------|
| BASE | 0.657 | 0.539 | 0.629 |
| Binary (PURGE) | 0.372 | 0.365 | 0.408 |
| Exponential | 0.379 | 0.382 | 0.417 |
| **PageRank Softmax** | **0.346** | **0.350** | **0.390** |

👉 **PageRank 在所有三项上表现最佳**，显著优于 binary baseline。

#### ✅ **训练效率（Figure 2）**
- **PageRank Softmax 在第 500 步即达到 Binary 在 1500 步才达到的遗忘水平**。
- 达到同等遗忘性能所需训练步数减少 **3倍**。
- 第 100 步时，PageRank 实现 +25.2%，Binary 仅为 +13.3%，相对增益约 **90%**。

#### ✅ **模型效用（Utility Metrics）**
所有方法均保持高度稳定的通用能力：
- 最大变化不超过 ±0.014
- 表明新奖励不会损害模型整体功能

#### ✅ **MIA 隐私指标**
| 方法 | MIA-FM ↑ | MIA-RM ↓ |
|------|----------|----------|
| Binary | -40.219 | -38.681 |
| Exponential | -40.265 | -38.640 |
| **PageRank Softmax** | **-40.280** | **-38.664** |

👉 PageRank 在隐私防御方面也略有优势。

---

### **消融实验结果（Ablation Studies）**

#### 🔹 **指数奖励中的衰减常数 $T$ 分析（Figure 3）**
- $T = 0.1$：接近 binary，信号仍稀疏，遗忘效果差
- $T = 0.5$：最优值，在三个 forget 指标上同时取得最小值
- $T ≥ 1.0$：奖励过于宽松，无法有效抑制泄露

✅ 结论：**$T^* = 0.5$ 是最佳选择**

#### 🔹 **PageRank 变体比较（Table 2 & Appendix D）**
| 变体 | 特点 | 性能 |
|------|------|--------|
| **PageRank (Raw)** | 权重呈幂律分布，头部节点主导 | 中等 |
| **PageRank-Linear** | 线性重分配权重 | 效果较差，破坏语义差距 |
| **PageRank-Softmax** | Softmax 压缩权重差距，保留排序 | ✅ **最优表现，鲁棒性强** |
| **PageRank-Argmax** | 所有权重集中于最高排名节点 | ❌ 效果最差，证明需多级监督 |

✅ 结论：**soft redistribution 比 hard re-ranking 更优**

---

## **4. 关键结论和发现**

### **主要发现**
1. **奖励设计是 RUL 成败的关键驱动因素之一**  
   密集、结构化的奖励能显著提升遗忘效率和最终性能。

2. **可验证性与稀疏性可以解耦**  
   不需要牺牲 verifiability 来换取 dense reward；二者正交。

3. **PageRank-based Reward 是目前最优方案**  
   - 利用语义结构提供细粒度反馈
   - 即使总违规数相同，也能区分“核心 vs 外围”泄露
   - 收敛速度快至 **3×**

4. **丰富的奖励不会损害模型 utility**  
   所有新奖励均维持原有能力不变，无明显副作用。

### **方法的局限性**
- **依赖高质量的实体抽取与语义图构建**  
  若初始 forget set 提取不准，PageRank 效果受限。
- **PageRank 的幂律分布问题需额外处理**（如 Softmax）
- 当前方法仍为静态奖励，未考虑动态调整策略
- 实验集中在人物类遗忘任务，泛化性有待进一步验证

### **未来工作方向**
- 设计 **adaptive reward schemes**，随训练进程自动调节奖励形状
- 引入 **multi-level forgetting priorities**（如法律敏感 > 一般信息）
- 探索 **reward shaping via LLM-as-a-judge**，但仍保持 verifiability
- 将本框架推广至其他 RUL 场景（如版权内容移除、偏见消除）

---

> 📌 **一句话总结**：  
> 本文证明，通过精心设计的、非二元的、语义感知的 **verifiable reward**，可以在不损失模型能力的前提下，使强化遗忘（RUL）的速度提升 **高达 3 倍**，为实现高效、可扩展的机器遗忘提供了实用路径。

</details>

---

### 8. [TAPO: Transition-Aware Policy Optimization for LLM Agents](https://arxiv.org/abs/2607.27973)

**Authors**: Cong Li, Peixi Peng, Yisen Zhao, Xinyu Hu, Shudong Liu, Zhan Su, Zhuojian Li  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.27973v1  

#### Abstract
Recently, Reinforcement Learning (RL) has emerged as a crucial paradigm for the post-training of Large Language Model (LLM) agents. However, existing methods predominantly rely on sparse task rewards for policy optimization, failing to fully exploit another class of inherently dense supervisory sign...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TAPO: Transition-Aware Policy Optimization for LLM Agents 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Agentic RL** 方法（如 GRPO、GiGPO）在对 **LLM Agents** 进行后训练时，主要依赖稀疏的 **task rewards** 来优化策略（policy），而忽略了在线交互过程中自然产生的、密集的环境反馈信号——即“执行某个动作后环境实际发生了什么”这一信息。这种设计导致模型难以系统地学习到 **action-state 转移动态**，从而在长视野、奖励稀疏的任务中决策脆弱、泛化能力受限。

### 提出了什么新方法或新思路
作者提出 **TAPO (Transition-Aware Policy Optimization)**，一种统一的 LLM Agent 后训练框架，其核心思想是：
- 在标准的 **Policy Optimization** 流程之外，复用 **rollout 数据中的 (s, a, s') 三元组**，引入一个额外的 **Transition Supervision** 阶段。
- 该阶段要求共享的 backbone 模型预测“在当前状态 s 下执行动作 a 后，下一个观察 s' 是什么”，即进行 **action-conditioned next-observation prediction**。
- 整个训练过程在 **Policy Learning** 和 **Transition Supervision** 之间交替进行。

### 相比现有方法的优势
- ✅ **无需额外成本**：不依赖专家数据、不增加采样开销、无推理时延迟。
- ✅ **轻量级插件式增强**：可作为现有 RL 算法（如 GRPO/GiGPO）的即插即用模块。
- ✅ **理论动机强**：受强化学习理论启发——通用目标导向智能体必须隐式编码环境动力学模型。
- ✅ **提升决策鲁棒性**：通过显式建模转移动态，增强了模型对动作后果的敏感度，改善长视野规划。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **WebShop**：模拟电商网站购物任务，需理解指令、多步网页交互并购买符合条件的商品。
- **ALFWorld**：基于 ALFRED 的具身环境，测试代理在家庭场景中完成复杂多步任务的能力（如“把烤箱里的苹果拿出来”）。

### 实验设置和评估指标
| 项目 | 设置 |
|------|------|
| **基础模型** | Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct |
| **RL 算法** | GRPO、GiGPO 及其 + TAPO 变体 |
| **交替频率 I** | 默认为 4（每 4 次 RL 更新插入一次 Transition Supervision） |
| **评估指标** | - WebShop: 平均得分（Score）、成功率（Success Rate %）<br>- ALFWorld: 成功率（Success Rate %） |
| **训练细节** | 使用 group-based RL，group size=8，共 128 个并行环境；KL penalty=0.01；学习率=1e-6 |

### 基线方法对比
- **Closed-source LLMs**: GPT-4o, Gemini-2.5-Pro
- **Prompting 方法**: ReAct, Reflexion
- **RL 训练方法**: GRPO, GiGPO
- **相关过渡建模方法参考对比**: Early Experience [19], RWML [20]

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
#### WebShop 结果（Success Rate %）
| 方法 | Qwen2.5-1.5B | Qwen2.5-7B |
|------|--------------|------------|
| GRPO | 56.8 ± 3.8 | 66.1 ± 3.7 |
| **TAPO-GRPO** | **66.2 ± 4.9** (+9.4) | **73.7 ± 4.0** (+7.6) |
| GiGPO | 67.4 ± 4.5 | 75.2 ± 3.8 |
| **TAPO-GiGPO** | **71.7 ± 3.3** (+4.3) | **77.9 ± 3.7** (+2.7) |

#### ALFWorld 结果（Success Rate %）
| 方法 | Qwen2.5-1.5B | Qwen2.5-7B |
|------|--------------|------------|
| GRPO | 72.8 ± 3.6 | 77.6 ± 5.2 |
| **TAPO-GRPO** | **76.4 ± 4.0** (+3.6) | **83.6 ± 3.9** (+6.0) |
| GiGPO | 86.7 ± 1.7 | 90.8 ± 1.3 |
| **TAPO-GiGPO** | **88.4 ± 3.8** (+1.7) | **93.6 ± 1.4** (+2.8) |

> ✅ 所有配置下，**TAPO 均显著优于对应 baseline**，且效果在不同模型规模和 RL 算法上一致成立。

### 与基线方法的对比结果
- **相比 Prompting 方法（ReAct/Reflexion）**：所有 RL 方法（含 TAPO）均有巨大优势，验证了 RL 后训练的必要性。
- **相比纯 RL 方法（GRPO/GiGPO）**：TAPO 在两个任务、两种模型上均带来稳定增益，最高提升达 **+9.4% SR**（1.5B WebShop 上 TAPO-GRPO vs GRPO）。
- **相比其他 Transition Modeling 方法（Table 2）**：
  - TAPO: **93.6%**
  - RWML: 90.1%
  - Early Experience: 82.8%
  > 表明 TAPO 在无需额外预训练阶段的情况下仍能达到甚至超越专门的世界模型方法。

### 消融实验结果
#### （1）消融 TAPO 的有效性（Figure 3）
- TAPO-GRPO 收敛更快、最终性能更高（约 66% vs 55–60%），说明其提升了样本效率和最终表现。

#### （2）交替频率分析（Table 3）
- 最优 I = 4（成功率 66.2%），但即使在 I ∈ {2,3,5,10,20} 范围内也始终优于 vanilla GRPO（56.8%），表明 TAPO 对超参数不敏感。

#### （3）是否需要全程监督？（Table 4）
| 设置 | Success Rate (%) |
|------|------------------|
| Vanilla GRPO | 56.8 |
| 仅前 20 步 Transition Supervision | 59.1 |
| 仅前 40 步 | 61.8 |
| 全程交替训练（Full Process） | **66.2** |
> 显著说明 Transition Supervision 不只是“热启动”，而是需要在整个训练过程中持续提供反馈以支持策略优化。

#### （4）Transition 建模能力分析（Figure 4）
- 使用 **Perplexity (PPL)** 评估模型对下一状态的预测能力：
  - Untrained: ~4.5
  - GRPO: ~3.8
  - **TAPO-GRPO**: **~1.18**
> 巨大差距表明：**只有 TAPO 显式地教会了模型去建模环境转移动态**，而标准 RL 无法自然保留此能力。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **rollout 中的 transition 反馈是宝贵信号**：除了稀疏奖励外，(s, a, s') 序列提供了密集、动作条件化的监督信号，应被充分利用。
2. **Transition Supervision 显著提升性能**：将 next-observation prediction 作为辅助目标，能有效增强 LLM Agent 对环境动态的理解，进而改善长期决策。
3. **TAPO 是高效且通用的增强模块**：无需额外数据或计算开销，即可作为现有 RL 算法的即插即用改进方案，在多个任务和模型上稳定提效。
4. **Lookahead Simulation 能力涌现**（见 Figure 5）：TAPO 促使模型形成“如果我这么做，接下来会看到什么”的前瞻性思维，并据此制定条件策略，展现出更强的鲁棒性和规划能力。

### 方法的局限性
- **可能引入轻微的“能力税”（capability tax）**：
  - 在非目标任务（如 GSM8K 数学题）上，TAPO-GRPO 准确率为 56.5%，略低于 base model（59.1%）和 GRPO（57.9%）。
  - 但作者指出：这种下降更多是任务特定后训练的普遍现象，而非 TAPO 特有缺陷。
- **未探索更复杂的监督形式**：目前采用简单的 teacher-forcing 预测，未来可尝试更高级的建模方式。

### 未来工作方向
- 探索更先进的 **supervision formulation**（如对比学习、潜在空间建模）。
- 设计更智能的 **training scheduling** 策略（动态调整交替频率）。
- 平衡 **agentic task performance** 与 **general capability preservation**，例如通过正则化或数据选择机制。
- 将 TAPO 思想扩展至多模态 Agent 或真实机器人控制等更具挑战性的场景。

--- 

> 📌 **一句话总结**：  
> **TAPO 通过在标准 RL 训练中交替加入 transition prediction 监督，低成本地赋予 LLM Agent “预见动作后果”的能力，从而显著提升其在复杂长视野任务中的表现，是一种简单、有效、通用的后训练增强范式。**

</details>

---

### 9. [AutoPref: Automatic Discovery of Task-Specific Preference Objectives for Neural Combinatorial Optimization](https://arxiv.org/abs/2607.27953)

**Authors**: Shengda Gu, Kai Li, Xinyi Ke, Haobo Fu, Yifan Zhang, Jian Cheng  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.27953v1  

#### Abstract
Combinatorial optimization problems (COPs) underpin many real-world decisions, but their exponentially large search spaces make high-quality solutions costly to obtain. Neural combinatorial optimization (NCO) learns fast construction policies, typically with reinforcement learning (RL), while prefer...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AutoPref: Automatic Discovery of Task-Specific Preference Objectives for Neural Combinatorial Optimization —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **传统 NCO 中偏好目标设计依赖人工**：现有的 preference-based Neural Combinatorial Optimization (NCO) 方法（如 PO4COPs、BOPO）使用手动设计的、通用型的 preference objective，难以适应不同 COP 任务（如 TSP、CVRP、JSSP）在解空间结构和奖励景观上的差异。
- **样本效率低且泛化能力差**：固定形式的目标函数无法动态调整学习信号强度与分布，导致训练效率下降，尤其在大规模或未见规模问题上表现退化。

### 🚀 提出的新方法与思路
- **AutoPref**：首个基于 LLM 的自动化框架，用于为 NCO 发现**任务特定的 preference objective**。
- **程序化解耦表示**：
  - 将 preference objective 分解为两个可编程组件：
    - **Pairwise Loss Program `f`**：决定从每对解比较中学到什么（what to learn）。
    - **Set-aware Weighting Program `g`**：根据整个采样解集上下文，决定每个比较应赋予多大权重（how much to learn）。
  - 构成联合的 programmatic 空间 $ \mathcal{F} \times \mathcal{G} $，统一并超越现有手工目标。
- **LLM 驱动 + 条件分阶段搜索策略**：
  - 第一阶段：固定均匀权重，搜索最优 pairwise loss `f*`。
  - 第二阶段：冻结 `f*`，搜索最优 set-aware weighting `g*`。
  - 显著降低搜索复杂度，并缓解信用分配难题。

### 🔍 相比现有方法的优势
| 维度 | 手工方法（PO4COPs / BOPO） | AutoPref |
|------|-----------------------------|----------|
| 设计方式 | 固定公式，one-size-fits-all | 自动发现，task-specific |
| 表达能力 | 有限，静态 | 可组合、可扩展，覆盖已有方法作为特例 |
| 泛化性 | 在新尺度/任务上易失效 | 跨 scale 和 problem family 均有效 |
| 效率 | 无需搜索开销 | 引入智能筛选机制（behavioral gates），避免无效训练 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在四类代表性 COP 家族上进行实验：
- **Routing 类**：
  - Traveling Salesman Problem (**TSP**)：TSP50, TSP100 (discovery), TSP1000
  - Capacitated Vehicle Routing Problem (**CVRP**)：CVRP50, CVRP100 (discovery), CVRP1000
- **Scheduling 类**：
  - Flexible Flow Shop Problem (**FFSP**)：FFSP50, FFSP100 (discovery), FFSP1000
  - Job Shop Scheduling Problem (**JSSP**)：JSSP10×10, JSSP15×15 (discovery), JSSP50×20

> 所有 discovery 实验均在中等规模实例上完成（如 TSP100），然后将发现的目标冻结，在其他 scale 上迁移测试。

### ⚙️ 实验设置与评估指标

#### 训练协议
- 所有方法使用相同的神经架构（如 AM/POMO）、训练预算和 inference protocol。
- AutoPref 通过短视训练（short-horizon downstream evaluation）评估候选 objective 的潜力。
- 最终性能在 full-budget policy training 下验证。

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Mean Cost ↓** | 测试集平均解成本 |
| **Optimality Gap (%) ↓** | $(C - C_{\text{ref}})/C_{\text{ref}} \times 100\%$，越小越好 |
| **Total Time (s)** | 推理总耗时（含模型生成和搜索时间） |

#### 基线方法对比
| 类别 | 方法 |
|------|------|
| **Reference Solvers** | Concorde (TSP), HGS/LKH3 (CVRP), CP-SAT (调度) |
| **Standard Neural Baselines** | AM, POMO, SymNCO (routing); MatNet (FFSP); MGL (JSSP) |
| **Preference-based / Self-labeling** | PO4COPs, BOPO, SLIM |
| **AutoPref 变体** | 
| - UPW | 使用发现的 `f*` + uniform weighting |
| - APW | 使用 `f*` + 发现的 `g*`（Adaptive Pair Weighting） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Tables 1 & 2）

#### ✅ Routing 结果（TSP/CVRP）
| 方法 | TSP100 Gap (%) | TSP1000 Gap (%) | CVRP100 Gap (%) | CVRP1000 Gap (%) |
|------|----------------|------------------|------------------|-------------------|
| PO4COPs | 0.174 | 10.644 | 1.449 | 6.968 |
| BOPO | 1.148 | 17.840 | 3.183 | 7.161 |
| **APW (Ours)** | **0.088** | **9.395** | **1.022** | **3.407** |

> ✅ **APW 在所有 scale 上显著优于最强手工 baseline（PO4COPs）**，尤其在大尺度（TSP1000, CVRP1000）优势更明显。

#### ✅ Scheduling 结果（FFSP/JSSP）
| 方法 | FFSP100 Gap (%) | FFSP1000 Gap (%) | JSSP15×15 Gap (%) | JSSP50×20 Gap (%) |
|------|------------------|-------------------|--------------------|--------------------|
| PO4COPs | 0.665 | 0.098 | 36.841 | 7.996 |
| BOPO | 0.637 | 0.259 | 8.108 | 7.700 |
| **APW (Ours)** | **0.526** | **0.071** | **7.629** | **7.641** |

> ✅ **在复杂约束调度问题上仍保持领先**，证明其对高维、非线性结构的适应能力。

### 🔬 消融实验结果（Table 3）

| Configuration | TSP100 成本降低 | FFSP100 成本降低 |
|--------------|------------------|-------------------|
| PO4COPs + Uniform (baseline) | 0.000 | 0.000 |
| Discovered `f` + Uniform | 0.0039 | 0.260 |
| PO4COPs + Discovered `g` | 0.0011 | 0.201 |
| **Discovered `f` + Discovered `g` (APW)** | **0.0067** | **0.300** |

> ✅ **双组件协同增效**：单独替换 loss 或 weighting 均能提升性能；两者结合达到最佳效果，说明两部分都学到了互补的有效信号。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **任务特定的 preference objective 至关重要**：
   - 手工设计的目标（如 PO4COPs、BOPO）在某些任务上会退化甚至不如标准 RL 方法。
   - AutoPref 发现的目标在所有 12 个测试场景下均取得最优结果，验证了自动发现的必要性。

2. **程序化解耦 + 分阶段搜索是高效可行的**：
   - 将 objective 拆分为 `f` 和 `g` 并分步优化，既保证了表达力又控制了搜索成本。
   - Behavioral gates 成功过滤掉 >90% 的无效程序提案，极大提升了搜索效率。

3. **所发现的目标具有强泛化能力**：
   - 在 discovery scale 外的大/小规模问题上依然表现优异，表明 learned objective 学到了本质规律而非过拟合当前 scale。

4. **揭示了理想学习信号的机制**：
   - 如 TSP 中发现的目标采用“成本差距相关的逆温度”调节 margin loss，防止梯度爆炸；
   - 同时通过 set-aware normalization 加权放大高质量比较的影响，实现稳定而高效的更新。

### ⚠️ 方法的局限性
- **计算成本较高**：尽管有 behavioral gates 和 short-horizon 评估，整体搜索过程仍需大量 GPU 时间。
- **依赖 LLM 生成质量**：若 LLM 缺乏对数学/代码的理解，可能生成语法正确但逻辑错误的程序。
- **目前仅限 pairwise preference**：尚未扩展至 listwise 或 ranking-based preference modeling。

### 🔮 未来工作方向
- 探索更复杂的 co-evolutionary 搜索策略，捕捉 `f` 与 `g` 之间的深层交互。
- 引入 surrogate model 加速 objective fitness 预测，减少实际训练次数。
- 将该范式推广至 multi-objective、constrained 或 dynamic COP 场景。
- 研究 discovered objective 是否可用于跨任务迁移或元学习。

---

> 💡 **一句话总结**：  
> AutoPref 首次将 preference objective 的设计视为一个可优化的程序搜索问题，利用 LLM 和结构化搜索框架实现了 NCO 中学习目标的自动化、个性化构建，推动了“让机器学会如何学习”的新范式。

</details>

---

### 10. [MUGEN: A Unified Framework for Efficient Motion Understanding and Generation](https://arxiv.org/abs/2607.27581)

**Authors**: Zhankai Ye, Yukai Jin, Bingyang Wei, Bofan Li, Yusen Wu, Fangyi Li, Shangqian Gao, Xin Liu  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.27581v1  

#### Abstract
Grounding human motion in language, and language in motion, is a central step toward physical AI systems that can understand, generate, and communicate human behavior. Unified motion--language systems first coupled the two directions through a shared discrete motion codebook, but quantization limits...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MUGEN: A Unified Framework for Efficient Motion Understanding and Generation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **motion-language** 系统在实现 **text-to-motion generation** 和 **motion-to-text understanding** 双向任务时面临以下挑战：
- 多数系统依赖 **discrete motion codebook**（如 VQ-VAE）作为共享表示，但量化过程会损失运动细节，限制生成质量。
- 高质量生成器（如 diffusion 或 autoregressive 模型）通过多阶段解码（multi-stage decoding）、残差码本（residual codebooks）、迭代去噪等机制提升质量，但显著增加推理成本。
- 这些复杂的生成机制通常不服务于理解任务，导致生成与理解无法真正共享同一接口。

因此，如何构建一个既能支持高质量生成、又能高效完成理解任务，并且**无需离散码本、推理极快**的统一框架，是一个关键挑战。

### 提出了什么新方法或新思路
本文提出 **MUGEN**（**MU**lti-modal **GEN**eration），一种全新的统一 motion-language 框架，其核心思想是：

> **用一组连续的 latent slots 替代离散 motion tokens，实现“无码本、单次采样”（no codebook, one draw）的高效生成与理解。**

具体创新点包括：

#### ✅ 创新架构设计
- **Adaptive-Length AutoEncoder (ALAE)**  
  将任意长度的 motion 序列压缩为固定数量 $K$ 个连续 latent slots，解码时再还原为原始帧序列。该 autoencoder 在训练后被冻结，成为固定的 motion 接口。
  
- **Depth-Routed Hidden States**  
  在语言模型中，每个 latent slot 可以从不同深度的 transformer 层读取特征，而非仅限于最后一层。这使得不同 slots 能获取不同类型的信息（如浅层语义 vs 深层动作细节），增强表达能力。

- **Calibrated Low-Rank Factor Head**  
  预测 latent slots 的联合分布（joint distribution），而非独立预测每个维度。通过低秩分解建模跨 slot 的相关性，使一次采样即可捕获文本条件下的多样性变化。

#### ✅ 统一训练范式
- 同一语言模型同时用于：
  - **Generation**：以 `<MOT>` token 开始，进行 $K$ 步 rollout，输出 latent slots 并送入 ALAE 解码。
  - **Understanding**：将 ALAE 编码出的 slots 投影到语言空间，供语言模型生成描述。
- 两者共享相同的 latent 表示、语言 backbone 和训练流程。

### 相比现有方法的优势
| 特性 | MUGEN | 传统方法（如 MoMask++, MotionGPT3） |
|------|-------|-------------------------------|
| 是否使用 codebook | ❌ 无 | ✅ 多级离散码本 |
| 解码步骤 | ✅ 单次采样 + 一次 decoder pass | ❌ 数十至上百步迭代 |
| 生成与理解是否共享表示 | ✅ 完全共享 continuous latent slots | ⚠️ 通常不一致（如 diffusion head 不用于理解） |
| 推理速度 | ⬇️ 极快（9ms / motion） | ⬆️ 慢（55–136ms） |
| 参数效率 | ✅ 中等规模仍领先 | ❌ 往往需要更大模型 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **HumanML3D**  
  - 包含约 14K 条 motion-text 对。
  - motion 表示为 263 维关节特征，20fps。
  - 文本简短，crowd-sourced 描述。
- **SnapMoGen**  
  - 更大规模、更长、更具表现力的文本描述。
  - 支持更复杂的动作理解和生成任务。

### 实验设置
- **Stage 1**: 训练 ALAE，将 motion 映射为 $K$ 个 latent slots（$K=2$ on HumanML3D, $K=4$ on SnapMoGen）。
- **Stage 2**: 冻结 ALAE，训练 GPT-2 作为 language backbone，联合优化 generation 与 understanding。
- **Latent Prediction**: 使用 depth routing + low-rank factor head 预测 latent 分布。
- **Sampling**: 测试时仅需 $K$ 步语言模型 rollout + 一次采样 + 一次 ALAE 解码。

### 评估指标
| 类别 | 指标 | 说明 |
|------|------|------|
| **Generation** | FID ↓ | 动作分布保真度 |
| | R@1/R@2/R@3 ↑ | 文本-动作检索精度 |
| | MM-Dist ↓ | 文本与生成动作的距离 |
| | Diversity ↔ | 保证合理多样性 |
| | CLIP Score ↑ (SnapMoGen) | 多模态对齐程度 |
| **Understanding** | BLEU@4, CIDEr ↑ | 自动生成 caption 的质量 |
| | Retrieval R@1 ↑ | 用生成 caption 检索原动作的能力 |
| | BERTScore ↑ | 语义相似性 |
| **Efficiency** | Latency (ms), Throughput (motion/s) | 推理延迟与吞吐量 |

### 基线方法对比
- **Generation-only**:
  - T2M-GPT, MMM, MoMask, MoMask++, BAMM, MDM, StableMoFusion
- **Unified (generation & understanding)**:
  - MotionGPT, MotionGPT3, UniMo
- 所有结果均遵循官方协议（20次重复测试，报告均值与置信区间）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1–3）

#### 📊 HumanML3D 上的表现
| 方法 | R@1 ↑ | FID ↓ | MM-Dist ↓ | CIDEr ↑ | BLEU@4 ↑ |
|------|--------|--------|------------|----------|-----------|
| Real motions | 0.510 | — | 3.012 | — | — |
| MoMask++ (best gen-only) | 0.528 | **0.045** | 2.912 | — | — |
| **MUGEN (Ours)** | **0.579** (+5.1%) | 0.087 | **2.629** | **50.4** | **22.0** |

✅ **全面超越所有 unified 方法，在 retrieval、matching distance、captioning 指标上领先**  
⚠️ FID 略逊于 MoMask++，但仍优于其他语言模型基线。

#### 📊 SnapMoGen 上的表现
| 方法 | R@1 ↑ | FID ↓ | CLIP Score ↑ |
|------|--------|--------|----------------|
| MoMask++ | 0.805 | **15.06** | 0.685 |
| **MUGEN (Ours)** | **0.815** | 21.05 | **0.698** |

✅ **首次在 retrieval 和 CLIP alignment 上全面超越 discrete-token SOTA**  
✅ Diversity 达到真实数据范围（19.79 vs 19.5–19.8）

#### 📊 Motion Understanding 性能（HumanML3D）
| 方法 | R@1 ↑ | BLEU@4 ↑ | CIDEr ↑ |
|------|--------|-------------|----------|
| MotionGPT3 | 0.573 | 19.41 | 46.17 |
| UniMo | — | 9.74 | 48.8 |
| **MUGEN (Ours)** | **0.586** | **21.98** | **50.41** |

✅ **唯一同时在 retrieval 和 captioning 上领先的 unified 模型**

### 与基线方法的对比结果
- **检索性能（R@1）**：
  - MUGEN 在 HumanML3D 上比 real motions 高 **+0.069**，比 MoMask++ 高 **+0.051**。
- **生成质量（FID）**：
  - 落后于 masked-codebook 方法（如 MoMask++），但优于所有基于语言模型的生成器。
- **推理效率**（见 Table 4）：
  | 方法 | Latency (ms) | Throughput (motion/s) | GFLOPs |
  |------|----------------|--------------------------|--------|
  | MoMask++ | 55 | 55.1 | 105.5 |
  | MotionGPT3 | 136 | 53.2 | 95.5 |
  | **MUGEN (Ours)** | **9** | **325.2** | **11.1** |
  - ⏱️ **延迟降低 6–14 倍，计算量减少 8.6–9.5 倍**

### 消融实验结果（Ablation Studies）

#### 🔍 Latent Budget $K$
| $K$ | R@1 | FID |
|-----|-----|-----|
| 1 | 0.574 | 0.109 |
| **2** | **0.579** | **0.087** |
| 4 | 0.577 | 0.091 |
| 8 | 0.567 | 0.111 |

➡️ $K=2$ 是最优平衡点，过大的 $K$ 反而损害性能。

#### 🔍 Depth Routing vs 最后一层
| 设置 | FID ↓ | R@1 ↑ |
|------|--------|--------|
| Last layer only | 0.123 | 0.574 |
| **Depth-routed** | **0.087** | **0.579** |

➡️ **FID 下降 29%**，证明 depth routing 显著提升信息提取能力。

#### 🔍 Unified Training 效果（Table 6）
| 训练方式 | FID ↓ | R@1 ↑ | CIDEr ↑ |
|--------|--------|--------|---------|
| Gen-only | 0.107 | 0.578 | — |
| **Joint (MUGEN)** | **0.087** | **0.579** | **50.4** |

➡️ 加入 understanding 任务反而提升了 generation 性能，说明双向学习具有正则化作用。

#### 🔍 Sampling vs Deterministic Decode
| 方式 | FID ↓ | R@1 ↑ |
|------|--------|--------|
| u-decode (deterministic) | 0.131 | 0.585 |
| **calibrated sampling (T=0.6)** | **0.087** | 0.579 |

➡️ 一次带温度的采样可将 FID 降低 **34%**，验证了 calibrated factor head 的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **一个紧凑的 continuous latent 表示可以同时支持高质量生成与理解**，无需牺牲任一方。
2. ✅ **“no codebook, one draw” 是可行的**：通过 depth routing 和 calibrated low-rank factor head，单次采样即可达到多阶段离散模型的质量水平。
3. ✅ **统一训练有益于生成**：motion understanding 分支起到了正则化作用，反而提升了 generation 的 FID 和 retrieval 性能。
4. ✅ **推理效率极大提升**：相比最强基线，MUGEN 实现了 **6–14倍更低延迟、8–10倍更少计算量**，适合实际部署。
5. ✅ **在 retrieval、alignment、captioning 等指标上全面领先**，仅在 FID 上略逊于 masked-codebook 方法。

### 方法的局限性
- **FID 仍有差距**：尽管 retrieval 和 alignment 更优，但在 distributional fidelity 上尚未完全超越 MoMask++ 等基于 codebook 的方法。
- **latent slots 数量有限**：当 $K$ 过大时性能下降，表明当前架构对 slot 数敏感。
- **依赖 ALAE 表达能力**：若 ALAE 无法充分保留 motion 结构，则会影响最终生成质量。
- **未解决极端多样性场景**：MultiModality 较低，可能难以覆盖非常罕见的动作组合。

### 未来工作方向
- 探索更强大的 **continuous tokenizer**（如 vibetoken）以进一步缩小 FID 差距。
- 将 MUGEN 扩展至 **video-language** 或 **robotics policy learning** 场景。
- 设计动态 $K$ 机制，根据动作复杂度自适应调整 latent slot 数量。
- 引入 **temporal structure prior** 到 factor head 中，提升长序列一致性。
- 探索 **multi-granularity latent modeling**（如 hierarchical slots）以支持细粒度控制。

---

> **总结一句话**：  
> MUGEN 成功证明了 **continuous latent + unified training + efficient sampling** 能够在保持极致推理效率的同时，在多数指标上超越复杂的离散码本系统，为 future physical AI 提供了一个简洁、高效、可扩展的新范式。

</details>

---

### 11. [Memory Decoder at Scale: A Pretrained, Parametric Long-Term Memory](https://arxiv.org/abs/2607.27919)

**Authors**: Rubin Wei, Jiaqi Cao, Jiarui Wang, Junming Zhang, Qipeng Guo, Bowen Zhou, Zhouhan Lin  
**Category**: cs.CL  
**Published**: 2026-07-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.27919v1  

#### Abstract
Decoder-only language models entangle long-term memory and reasoning in a single parameter set, making it difficult to scale memory capacity independently. Memory Decoder introduces a parametric long-term memory module but only studies it at a relatively small scale. In this work, we present Memory ...

---

### 12. [FunL2O: LLM-Guided Feature Function Design for Learning to Optimize](https://arxiv.org/abs/2607.27389)

**Authors**: Bingheng Li, Junyang Cai, Yupeng Zhang, Bistra Dilkina, Jayant Kalagnanam, Dzung T. Phan  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.27389v1  

#### Abstract
Learning-to-optimize (L2O) methods accelerate repeated optimization by training models to predict solutions, warm starts, branching decisions, or other forms of solver guidance. A critical yet largely overlooked component of these pipelines is the feature function that maps problem instances to inpu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FunL2O: LLM-Guided Feature Function Design for Learning to Optimize

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Learning-to-Optimize (L2O)** 流程中，特征函数（feature function）负责将优化问题实例转换为机器学习模型可处理的输入表示。然而，现有方法普遍依赖**手工设计（hand-crafted）的特征**，这些特征通常是固定的、领域特定的，并且难以泛化到不同类型的优化任务中。

这一环节虽然对模型表达能力和下游性能至关重要，却长期被忽视，成为自动化流程中的“瓶颈”。

---

### 🚀 提出的新方法：FunL2O
本文提出了 **FunL2O** —— 首个通过 **LLM驱动的程序演化（LLM-driven program evolution）** 来自动设计 L2O 特征函数的统一框架。

#### 核心思想：
- 将特征函数视为**可执行程序**（executable code），而非静态向量。
- 在一个类似 **FunSearch** 的闭环中：
  - **LLM 提出候选特征函数**（Python 函数）
  - 所有候选函数需满足预定义的 **语义契约（semantic contract）**
  - 替换原始特征函数后，**重新训练原 L2O 模型**
  - 根据**下游优化性能**（如目标值、求解时间等）评估并选择更优者
  - 最终返回最优的可部署特征代码

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | FunL2O |
|------|--------|-------|
| **特征设计方式** | 手工构造，固定不变 | 自动搜索，动态演化 |
| **通用性** | 依赖专家经验，难迁移 | 统一框架支持多种 L2O 任务 |
| **评估标准** | 基于直觉或中间指标 | 基于真实下游性能（end-to-end） |
| **部署开销** | 无额外成本 | 搜索阶段使用 LLM，**部署时无需 LLM** |
| **可解释性** | 黑箱表示 | 返回可读、可分析的 Python 代码 |

> ✅ **核心优势**：将“特征工程”从人工劳动转变为基于反馈的自动化程序搜索，同时保持原有模型架构、训练流程和求解器不变。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与任务类型
实验覆盖了 **连续优化** 和 **混合整数规划（MILP）** 两大类共 **8 个代表性 L2O 流水线**，涵盖以下任务：

#### 连续优化任务：
| 方法 | 任务 | 类型 |
|------|------|-----|
| IPM-MPNN | LP 解预测 | Solution Prediction |
| FSNet / DC3 | QP/QCQP/SOCP 约束优化预测 | Constrained Prediction |
| PDHG-Net | PageRank LP 的 primal-dual warm start | Warm Starting |
| Smart Initial Basis | 单纯形法初始基选择 | Basis Initialization |
| Learning to Pivot | 单纯形法转轴决策 | Pivot Selection |

#### 混合整数优化任务：
| 方法 | 任务 | 类型 |
|------|------|-----|
| Predict-and-Search (P&S) | 在 CA、MIS、MVC 上生成部分赋值 | Partial Assignment |
| Learned Backdoors | 学习分支优先级 | Branching Priorities |

所有任务均使用标准基准数据集，确保训练、验证、测试集互不重叠。

---

### ⚙️ 实验设置

| 设置项 | 描述 |
|-------|------|
| **LLM 模型** | Claude-Opus-4.8, GPT-5.5, Gemini-3.1-Pro, DeepSeek-Math-V2（连续任务）；前两者用于 MILP |
| **搜索预算** | 8 轮迭代 × 每轮 6 个提案 = 共 48 个候选函数 |
| **训练重复次数** | 每组实验运行 3 次独立种子，取平均 |
| **特征接口约束** | 定义严格的 `semantic contract`，包括：<br>- 可用字段（如 A, b, c, lb, ub）<br>- 输出格式（tensor shape, type）<br>- 不允许访问标签、参考解、外部网络或求解器调用 |
| **部署模式** | 最终选中的特征函数是普通 Python 代码，**部署时不需 LLM**

---

### 📊 评估指标

根据不同任务角色定义主/次指标：

| 任务类型 | 主要指标（↓ 表示越低越好） | 次要指标 |
|--------|--------------------------|---------|
| **Solution Prediction** | Objective Gap ↓, Optimality Gap ↓ | Feasibility ↑ |
| **Warm Start / Basis / Pivot** | Iteration Saving ↑, Time Saving ↑, Pivot Count ↓ | Accuracy |
| **MILP Search (P&S)** | Primal Gap ↓, Primal Integral ↓ | — |
| **Backdoor Branching** | Wall Time ↓ | Win Rate ↑ |

> ✅ 所有最终报告结果均在**独立测试集**上评估，避免过拟合风险。

---

## 3. 主要实验结果和性能指标

### 📈 总体性能提升（见 Table 2 & 3）

#### ✅ 连续优化任务（Table 2）：
| 方法 | 关键改进 |
|------|--------|
| **IPM-MPNN (LP 预测)** | 平均目标差距降低 **16.4% ~ 49.4%**，可行性略有提升 |
| **FSNet (非凸 SOCP)** | 非凸 SOCP 间隙从 **933.85% → 141.28%**（↓84.9%），可行性从 0.25 → 1.00 |
| **DC3 (QP)** | 三个变体均有 2–4.3% 的优化间隙下降 |
| **PDHG-Net (warm start)** | 迭代节省从 42.44% → **51.05%**，时间节省从 34.03% → **39.28%** |
| **Learning to Pivot** | 几何平均 pivot 数从 **1162.86 → 857.78**（↓26.2%） |

> 💡 即使局部预测准确率变化不大，也能显著减少求解器工作量。

---

#### ✅ 混合整数优化任务（Table 3）：

##### Predict-and-Search：
| 域 | Final Primal Gap (%) | Primal Integral |
|----|------------------------|----------------|
| **CA** | 0.423 → **0.350** | 10.61 → **9.00** |
| **MIS** | 0.699 → **0.095** | 20.89 → **12.78** |
| **MVC** | 0.018 → **0.013** | 13.85 → **3.69** |

> 显著更快找到高质量可行解。

##### Learned Backdoors（Branching）：
| 域 | Mean Wall Time (s) | Win Rate (%) |
|----|---------------------|--------------|
| **CA** | 154.6 → **147.6** | 20 → **41** |
| **MIS** | 59.1 → **46.0** | 21 → **49** |
| **MVC** | 31.6 → **18.9** | 4 → **37** |

> 在所有域中均实现更低求解时间和更高胜率。

---

### 🔬 消融实验与控制分析（Ablation Studies）

#### （1）**Equal-Budget 对照：进化反馈 vs. 独立采样**
- 在 PDHG-Net 上比较两种策略：
  - **独立采样（无反馈）**：迭代节省达 44.42%
  - **FunL2O（带反馈）**：达 **52.23%**
- ➜ **+7.81% 提升来自反馈机制**，证明“精英保留 + 结果反馈”有效。

#### （2）**Matched-Width 控制：结构 vs. 维度扩展**
- 对比相同新增维度下的随机投影 vs. FunL2O 演化特征：
  - 在 6 个任务中有 **5 个 FunL2O 更优**
- ➜ 性能增益不仅来自“增加特征宽度”，而是**捕捉到了优化相关的结构性关系**

#### （3）**多 LLM 验证**
- 四个不同 LLM（Claude, GPT, Gemini, DeepSeek）均能在多数任务中产生改进
- 改进单元格数：GPT/Gemini 各 17/20，Claude 15/20
- ➜ 方法不依赖单一 LLM，具有较强鲁棒性

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **特征函数设计是 L2O 中可自动化且影响巨大的组件**  
   即使不改变模型结构、损失函数或求解器，仅优化输入表示即可带来显著性能提升。

2. **LLM 可作为有效的“特征程序员”**  
   LLM 能提出包含有意义数学变换的可执行特征函数，例如：
   - 归一化变量成本与列覆盖率（Set Cover）
   - 构建 PageRank 图的入/出度不对称特征
   - 多尺度非线性变换（log, square, tanh）

3. **演化的特征具备可解释性和可部署性**  
   输出为标准 Python 代码，可在生产环境中直接集成，无需在线调用 LLM。

4. **适用于多样化 L2O 角色**  
   无论是解预测、warm start 还是分支决策，FunL2O 均能提升性能，表明其通用性强。

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **离线搜索成本高** | 每次搜索需约 **12 GPU小时 + 13 CPU小时**，不适合实时场景 |
| **依赖已有 L2O 流水线** | 必须存在一个可重训练的 L2O pipeline 才能进行评估 |
| **无法保证全局最优** | 属于启发式搜索，不能理论保证收敛到最佳特征 |
| **LLM 可能生成无效代码** | 需要多次修复和验证，影响效率 |

---

### 🔮 未来工作方向

1. **引入更高效的搜索策略**  
   如基于梯度的近似、元学习初始化、mutation/crossover 操作符等。

2. **跨任务迁移特征模板**  
   探索是否可以从一类问题中学到的特征迁移到另一类相似问题。

3. **结合 symbolic regression 或 formal verification**  
   增强所生成特征的数学正确性和稳定性。

4. **应用于更大规模工业级优化器**  
   如供应链、电力调度等领域的真实世界求解器集成。

---

## ✅ 总结一句话

> **FunL2O 成功将 LLM 引入 L2O 的“特征工程”环节，实现了端到端性能驱动的自动化特征函数设计，在不改动模型与求解器的前提下，显著提升了各类优化任务的表现，且最终产物为可解释、可部署的标准代码，为 L2O 的自动化开辟了新路径。**

</details>

---

### 13. [Neural Network-Assisted CLEAN for Channel Modeling in Low-SNR Regimes](https://arxiv.org/abs/2607.27450)

**Authors**: Chaofan Deng, Linyu Sun, Jaeho Lee, Arijit Raychowdhury  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.27450v1  

#### Abstract
Accurate multipath parameter estimation is critical for modern wireless communication systems, particularly in challenging low-SNR environments. Traditional Maximum Likelihood Estimation algorithms, such as CLEAN, provide high-resolution parameter extraction but suffer from prohibitive computational...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Neural Network-Assisted CLEAN for Channel Modeling in Low-SNR Regimes*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在低信噪比（**Low-SNR**）环境下，传统无线通信系统中的多径参数估计（如 AoA/AoD、距离等）面临以下挑战：
- **传统 CLEAN 算法**依赖于多维网格搜索（grid search），计算复杂度高（$O(G^3N_fN_TN_R)$），难以实现实时处理；
- **子空间方法**（如 MUSIC、ESPRIT）在低 SNR 下因“子空间交换”现象导致性能急剧下降，且要求非相干信号源；
- **纯数据驱动的深度学习模型**（one-shot DL）缺乏物理一致性，在面对 **off-grid 参数** 和可变数量的多径分量（**variable $N_{MPC}$**）时泛化能力差，易产生虚假峰值或漏检。

### 🚀 提出的新方法：**NN-CLEAN**
提出一种混合框架——**Neural Network-Assisted CLEAN (NN-CLEAN)**，将深度神经网络嵌入到传统的 CLEAN 迭代环路中：
- 在每轮迭代中，用一个 **multi-head residual network** 替代耗时的三维网格搜索，快速预测当前主导路径的参数 $\theta = \{d, \phi_T, \phi_R\}$；
- 利用精确的数学模型进行后续的复增益估计和残差减除（residual subtraction），确保物理一致性；
- 整个过程是迭代式的：提取 → 建模 → 减去 → 下一轮。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **精度与鲁棒性** | 在低 SNR（5 dB）下达到 >96% 的估计准确率，接近传统 GS-CLEAN，显著优于 subspace 方法和 one-shot NN； |
| **泛化能力** | 支持变量 $N_{MPC}$ 和连续 off-grid 参数，克服 OOD（Out-of-Distribution）问题； |
| **计算效率** | 摒弃网格搜索，采用并行前向推理，实现近恒定的运行时间和内存消耗（flat scaling）； |
| **物理可解释性** | 非端到端黑箱，通过物理模型约束误差传播，避免“幻觉”路径； |

---

## 2. 核心实验方法和设置

### 📊 数据集与仿真环境
- **合成数据生成**：基于 4×4 MIMO 系统，在 2–3 GHz 范围内使用 $N_f = 512$ 个子载波构建信道矩阵；
- **阵列配置**：Tx/Rx 均为半波长间距的 ULA（Uniform Linear Array），中心频率 2.5 GHz；
- **信道模型**：包含 LOS + 多个散射路径（MPCs），每个路径具有随机 AoA、AoD、距离和复增益；
- **训练数据**：动态在线生成，$N_{MPC} = 4$，参数严格 on-grid；
- **测试数据**：完全 **off-grid** 设置（从连续分布采样），用于评估真实场景下的泛化能力。

### ⚙️ 实验设置
- **输入**：残差信道矩阵 $H_{\text{rem}}(f_k)$ 的实部与虚部分离后作为 NN 输入张量；
- **网络结构**：
  - Backbone：Conv2D + MaxPooling + ResNet 结构压缩特征；
  - Multi-head 输出：三个独立 FC 层分别输出 AoA、AoD、distance 的分类 logits；
  - 使用 **Gaussian Label Smoothing (GLS)** 缓解离散化误差；
- **训练策略**：三阶段课程学习（curriculum learning）：
  1. Phase 1（0–50%）：仅学习最强路径检测；
  2. Phase 2（50–75%）：教师强制（teacher forcing）迭代提取所有路径；
  3. Phase 3（75–100%）：自回归模式，使用自身预测结果做残差减除；
- **优化器**：Adam，学习率 $10^{-3}$，batch size 128。

### 🎯 评估指标
| 指标 | 定义 |
|------|------|
| **Estimation Accuracy** | 正确识别的 MPC 数 / 总真实 MPC 数（需满足 AoA/AoD/distance 在 ±2 grid 内） |
| **Angular MCD RMSE** | 角度多径分量距离的均方根误差（考虑 wrap-around） |
| **Distance MCD RMSE** | 归一化距离误差 |
| **Energy Captured Ratio** | $\frac{\|\hat{H}\|^2}{\|H\|^2}$，重建信道能量占比 |
| **Runtime & Memory Usage** | 不同 batch size 下的执行时间与 GPU 显存占用 |

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **GS-CLEAN** | 传统 CLEAN + 网格搜索 | 精度高但计算昂贵 |
| **3D-MUSIC / 3D-ESPRIT** | 子空间方法 | 对相干信号敏感，低 SNR 下性能崩塌 |
| **One-shot NN** | 端到端深度学习 | 固定输出维度，无法适应 variable $N_{MPC}$，off-grid 性能差 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Fig. 4–6）
#### ✅ 图 4：不同 $N_{MPC}$ 下的估计准确率（SNR = 5 dB）
| 方法 | $N_{MPC}=1$ | $N_{MPC}=2$ | $N_{MPC}=3$ | $N_{MPC}=4$ |
|------|-------------|-------------|-------------|-------------|
| **NN-CLEAN** | ~98% | ~96% | ~88% | ~78% |
| **GS-CLEAN** | ~99% | ~97% | ~92% | ~85% |
| **One-shot NN** | ~85% | ~70% | ~50% | <40% |
| **3D-MUSIC/ESPRIT** | ~70% | ~60% | ~50% | ❌（受限于秩） |

> 💡 **发现**：NN-CLEAN 在稀疏场景（$N_{MPC} \leq 2$）表现最优，且远超 one-shot NN，验证了迭代架构的有效性。

#### ✅ 图 5：不同 SNR 下的 CDF 性能（$N_{MPC}=2$）
- **Angular MCD RMSE**：
  - 在 5 dB SNR 下，NN-CLEAN 的 90% 分位误差 < 0.015°；
  - 与 GS-CLEAN 差距极小（< 0.005°），说明离散化损失可控。
- **Distance MCD RMSE**：
  - 同样保持紧密跟随 GS-CLEAN 曲线；
- **Energy Captured Ratio**：
  - 在 5 dB 时超过 95%，表明信道重建保真度高。

#### ✅ 图 6：计算效率对比（$N_{MPC}=1$，GPU: RTX 4070）
| Batch Size | GS-CLEAN Runtime | NN-CLEAN Runtime | GS-CLEAN Memory | NN-CLEAN Memory |
|------------|------------------|-------------------|------------------|------------------|
| 1          | ~3.8 ms          | ~3.5 ms           | ~2.8 GB          | ~1.4 GB          |
| 16         | ~60 ms           | ~4.0 ms           | ~45 GB           | ~1.6 GB          |

> 🔥 **核心优势**：NN-CLEAN 具有近乎平坦的扩展性（near-flat scaling），而 GS-CLEAN 的计算和内存随 batch size 线性增长。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **NN-CLEAN 成功融合了 ML 的速度与 MLE 的物理严谨性**：
   - 用 NN 加速主导路径搜索，保留 CLEAN 的迭代减除机制；
   - 实现了在低 SNR 下对 off-grid 和 variable $N_{MPC}$ 场景的强大泛化能力。

2. **精度-效率权衡合理**：
   - 虽然由于输出层离散化略低于 GS-CLEAN 的绝对精度，但在实际应用中差异可忽略；
   - 换来的是 **两个数量级以上的加速潜力** 和 **极佳的批处理可扩展性**。

3. **迭代式 NN 设计优于 one-shot 架构**：
   - one-shot NN 在 $N_{MPC}>2$ 时性能骤降，证明其不适合动态环境；
   - NN-CLEAN 的逐个提取机制天然支持任意数量的 MPC。

4. **适用于 ISAC 与大规模 MIMO 实时处理**：
   - 扁平化的 runtime/memory 表现使其成为集中式基带处理的理想选择。

### ⚠️ 方法的局限性
- **依赖预定义网格**：虽然能泛化到 off-grid 参数，但仍需训练时定义的参数空间支撑；
- **训练成本较高**：需要百万步训练 + 课程学习设计；
- **未验证硬件实测数据**：目前仅为仿真结果，尚未在真实信道测量中验证；
- **对初始 SNR 较低（<-5 dB）的情况未充分测试**。

### 🔮 未来工作方向
1. 将框架扩展至 **3D 空间**（加入 elevation）和 **高速移动场景**（引入 Doppler）；
2. 探索 **无监督/自监督微调机制**，以适应现场漂移的信道特性；
3. 集成到 **端到端 ISAC 系统** 中，联合优化感知与通信性能；
4. 开发轻量化版本用于 **边缘设备部署**（如 vehicular networks）。

---

## ✅ 总结一句话
> **NN-CLEAN 通过将深度学习嵌入 CLEAN 的迭代循环，在不牺牲物理一致性的前提下，实现了低 SNR 下高效、鲁棒、可扩展的多径参数估计，为下一代 MIMO 与 ISAC 系统提供了极具前景的实时解决方案。**

</details>

---

### 14. [From Expert Reduction to Behavioral Divergence: Tracing Numerical State through Sparse MoE Inference](https://arxiv.org/abs/2607.28097)

**Authors**: Tianyang Zhu  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.28097v1  

#### Abstract
Mathematically equivalent expert-reduction orders can produce observably different sparse-MoE executions. We isolate this effect in native DeepSeek-V4-Flash by freezing local MoE state and varying only aggregation semantics. Four schemes separate operand representation from accumulator precision. At...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Expert Reduction to Behavioral Divergence: Tracing Numerical State through Sparse MoE Inference*

---

## 1. 主要贡献和创新点

### ✅ 解决的问题
该论文揭示了一个在 **sparse Mixture-of-Experts (MoE)** 推理中被忽视的关键问题：**即使数学上等价的专家聚合顺序（expert-reduction order）**，由于浮点数计算的有限精度特性，在实际执行中可能导致完全不同的行为输出。

这一现象挑战了传统认知——即只要模型权重、图结构一致，推理结果就应该相同。作者指出，**数值实现细节（如操作数表示、累加器精度、归约顺序）本身已成为模型语义的一部分**，而不仅仅是底层实现细节。

### ✅ 提出的新方法与思路
- **Trace-Freeze-Fork 干预法**  
  冻结前向传播中的局部 MoE 状态，仅改变专家聚合的归约语义（reduction semantics），从而隔离并观察其对后续状态和生成文本的影响。
  
- **四类聚合方案设计（P32/C/A/B）**  
  明确分离 **操作数表示（operand representation）** 和 **累加器精度（accumulator precision）** 的影响：
  - `P32`: FP32 操作数 + FP32 累加器（高精度反事实）
  - `C`: BF16 操作数 → 精确提升为 FP32 → FP32 累加器（匹配原生行为）
  - `A`: FP32 操作数 + 每步后 BF16 舍入
  - `B`: BF16 操作数 + 每步后 BF16 舍入（模拟典型低精度硬件）

- **定义两个关键状态边界**  
  - **Post-mHC state**: 层内前向传递中的隐含状态边界（intra-token boundary）
  - **Full persistent state**: 解码步骤之间的跨 token 持久状态边界（cross-token continuation boundary）

- **提出分层运行时一致性验证框架（Hierarchical Runtime Conformance）**  
  建议从输出到内部状态逐级比对，而非仅依赖最终 token 是否一致。

### ✅ 相比现有方法的优势
| 方面 | 本文优势 |
|------|--------|
| **因果控制性** | 首次实现对单个 MoE 层归约顺序的精确干预，排除其他变量干扰 |
| **状态可追溯性** | 通过重建 post-mHC 和 full persistent state，验证其作为“充分条件”的作用 |
| **行为可观测性** | 将微小数值差异映射至语义分歧（如“裁员”vs“招聘”），建立从数值到语义的完整链条 |
| **工程指导意义** | 明确提出应将“操作数格式 + 累加器精度 + 归约顺序”纳入 runtime/hardware 的兼容性契约 |

---

## 2. 核心实验方法和设置

### 🧪 数据集与提示
- **深度测试集（Depth Cases）**：
  - `"why the sheep"`
  - `"朋友昨天打来电话"`（中文事件方向分析）
  - `"Morning light filled the room"`
- **广度探索集（Breadth Set）**：
  - 50 个 prompt（25 英文 + 25 中文），用于检验现象普遍性
- 所有 prompt 使用原始 UTF-8 输入，解码方式为 **greedy decoding**

### ⚙️ 实验设置
- **模型**：`DeepSeek-V4-Flash`（6 专家路由，1 共享专家）
- **运行时环境**：Colibri 项目原生 CPU 运行时（非 Transformers 复现）
- **硬件平台**：AMD Ryzen AI MAX+ 395 CPU，Windows 11
- **控制粒度**：冻结所有前置状态（prefix state、selected experts、gates、weighted terms、shared output、full persistent state），仅改变目标 MoE 层的归约顺序或语义

### 🔍 评估指标
| 层级 | 测量对象 | 指标 |
|-----|--------|------|
| L0 | Operator 数值状态 | MoE 输出、中间值（FP32）、最大绝对误差（max L∞） |
| L1 | Layer state | post-attention、post-mHC 向量是否 bitwise identical |
| L2 | Persistent state | KV cache、compressor/indexer state 是否完全一致 |
| L3 | Discrete behavior | Router top-k selection、greedy token ID |
| L4 | Semantic behavior | 生成文本、事件方向分类（layoffs/hiring/other） |

### 📊 基线方法对比
- **Native Reference**：未修改的原生推理路径，作为行为基准
- **Same-mode Canonical Reference**：固定 identity permutation 的各 scheme 参考，用于区分“方案切换”与“顺序敏感”的影响
- 四种 aggregation schemes 自身互为对照组

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）单层归约顺序导致显著行为分歧
在 `"why the sheep"` 的 layer-5 fork 上：

| Scheme | Raw Permutations | Structural Classes | Distinct Texts |
|-------|------------------|--------------------|----------------|
| A     | 720              | 720                | 10             |
| B     | 720              | 360                | 11             |

> 尽管数学上等价，不同归约顺序产生了多达 **11 种不同的续写文本**，说明存在多个 **continuation basins**。

#### （2）中文 prompt 下的语义极化现象
在 `"朋友昨天打来电话"` 上，B 模式产生 360 类结构等价类，对应：
- **202 类 → 裁员（layoffs）**
- **113 类 → 招聘（hiring）**
- **45 类 → 其他（other）**

随机抽样 10 条长序列（64 tokens）均保持初始语义方向不变，且叙事发展各异，表明分支具有 **可重现的语义极化能力**。

#### （3）持久化轨迹实验（Persistent Protocol）
共 768 条轨迹（3 prompts × 64 seeds × 4 schemes）：

| Scheme | Route Divergence (192/192) | Token Sequence Divergence |
|--------|----------------------------|----------------------------|
| P32    | 192/192                    | 161/192                    |
| C      | 0/192                      | 0/192                      |
| A      | 192/192                    | 178/192                    |
| B      | 192/192                    | 172/192                    |

> **C 方案完全复现原生行为**，而 P32/A/B 均引发广泛路由和文本分歧。

#### （4）C 方案中间态一致性验证
对 192 条 C 路径进行中间态检查：
- MoE 输出、post-mHC、next-router scores、prefill logits 均 **bitwise identical**
- 最大 L∞ 差距 = 0
> 支持 C 是当前状态下 **数值稳定且行为保真** 的聚合策略。

#### （5）广度探索：延迟发散现象
在 50-prompt breadth study 中，随着生成长度增加，文本分离比例上升：
| Horizon | Cumulative Separated |
|--------|-----------------------|
| 8 tokens | 12/50 (24%)          |
| 16 tokens | 24/50 (48%)         |
| 32 tokens | 36/50 (72%)         |

> 表明许多差异是 **延迟显现的（delayed divergence）**，短序列无法暴露全部行为差异。

#### （6）状态重建控制实验（关键验证）

| 控制类型 | 是否成功重建下游轨迹？ | 说明 |
|--------|--------------------------|------|
| **Post-mHC 替换** | ✅ 完全一致（339/339 post-mHC states, 344/344 routes） | post-mHC 是 intra-token 充分状态边界 |
| **Persistent State 替换（FP64 delta）** | ✅ 完全一致（301/301 states/routes） | full persistent state 是 cross-token 充分状态边界 |
| 若使用 FP32 delta 替换 | ❌ 仅恢复 10/43 层状态 | 必须用更高精度保存差值才能无损还原 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **归约顺序是稀疏 MoE 的隐藏语义维度**  
   即使操作数相同，不同的归约顺序可通过浮点舍入误差触发后续 router 或 argmax 分支变化，最终导致 **完全不同语义输出**。

2. **相同 token 不意味着相同状态**  
   内部状态差异可以跨越 token 边界持续存在，并在未来几步才显现为输出分歧 —— **延迟发散机制成立**。

3. **C 方案是当前设置下的数值稳定锚点**  
   使用 BF16 操作数 + 精确转为 FP32 累加器的方式能最好地复现原生行为，适合作为 **runtime 兼容性参考标准**。

4. **full persistent state 是跨 token 行为延续的关键载体**  
   只要该状态一致，即使前面经历了不同归约路径，后续行为也能完全同步。

5. **建议采用分层一致性验证流程**  
   ```
   L0: Token/text → L1: Logits/margins → L2: Persistent state → L3: post-mHC → L4: Operator trace
   ```

### ⚠️ 方法的局限性

| 限制项 | 说明 |
|-------|------|
| **单一模型 & 单一运行时** | 结果基于 DeepSeek-V4-Flash + Colibri CPU runtime，不一定推广至其他架构 |
| **未测量真实硬件发生频率** | 实验穷举所有排列，但未统计 GPU/NPU 上自然发生的归约顺序分布 |
| **未实现 exact quantized-operand summation reference** | 缺少理论上的“黄金标准”用于绝对比较 |
| **C 的稳定性范围受限** | 仅在 6-term states 和特定 schedule 下验证，不能保证通用 BF16 稳定性 |
| **缺乏 frozen-route 控制** | 无法断言 routing 是唯一中介变量 |

### 🔮 未来工作方向

1. **构建跨平台数值一致性测试套件**  
   基于本文提出的 state boundary 设计标准化 conformance test。

2. **开发具备确定性归约能力的 MoE runtime/hardware**  
   如使用 fixed-tree reduction、superaccumulator 或 guard bits 技术。

3. **研究 adaptive protection for small router margins**  
   对接近决策边界的 router score 动态提高精度以防止误跳。

4. **扩展至动态调度（dynamic scheduling）场景**  
   当前实验使用静态 layer-static permutation，未来需覆盖 token-dynamic 调度。

5. **探索更多语义层面的行为建模与控制**  
   利用此类“可控分歧”实现定向生成或多路径推理。

---

> **总结一句话**：  
> 在稀疏 MoE 模型中，**数值实现不再是透明的底层细节，而是直接影响行为输出的显式语义组成部分**；本文首次系统性揭示并验证了从 **expert reduction order → numerical divergence → behavioral bifurcation** 的完整因果链。

</details>

---

### 15. [TopoFormer: Topology Meets Attention for Graph Learning](https://arxiv.org/abs/2607.28259)

**Authors**: Md Joshem Uddin, Astrit Tola, Cuneyt Gurcan Akcora, Baris Coskunuzer  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.28259v1  

#### Abstract
We introduce Topoformer, a lightweight and scalable framework for graph representation learning that encodes topological structure into attention-friendly sequences. At the core of our method is Topo-Scan, a novel module that decomposes a graph into a short, ordered sequence of topological tokens by...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TOPOFORMER: Topology Meets Attention for Graph Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统基于 **Persistent Homology (PH)** 的拓扑数据分析（TDA）在图学习中面临两大瓶颈：
- **计算成本高**：PH 需要全局边界矩阵约简（global boundary-matrix reduction），时间复杂度可达立方级，难以并行化。
- **信息损失严重**：标准流程需将持久同调生成的 **persistence diagrams** 转换为向量（如 persistence images、landscapes），这一过程破坏了序列结构，且对下游任务敏感。

此外，图上的子水平滤波（sublevel filtration）常因早期激活节点迅速“饱和”整个图，导致后期拓扑特征被抑制。

### **提出了什么新方法或新思路**
作者提出 **TOPOFORMER**，一个轻量、可扩展的图表示学习框架，其核心是 **Topo-Scan** 模块：

- **Topo-Scan**：将图通过节点或边的滤波函数（filtration function）切分为一系列有序的拓扑切片（topological slices），每个切片是一个局部子图，计算其 **Betti 数**（β₀: 连通分量数，β₁: 环数）、节点数、边数，形成一个短的、有序的拓扑 token 序列。
- **序列输入 Transformer**：这些拓扑序列表示天然适合 **Transformer** 处理，无需节点嵌入或图特定注意力机制。
- **跳过持久同调图**：直接从图结构中提取多尺度拓扑动态，避免了完整的 PH 流程（滤波 → persistence diagram → vectorization）。

### **相比现有方法的优势**
| 维度 | TOPOFORMER | 传统 PH 方法 |
|------|------------|--------------|
| **计算效率** | ✅ 可并行化，复杂度低（$O(L(|V|+|E|+T))$） | ❌ 全局约简，不可并行，复杂度高 |
| **信息保留** | ✅ 保留序列结构，捕捉晚期出现的拓扑特征 | ❌ 向量化破坏顺序，易丢失晚期信号 |
| **模型兼容性** | ✅ 与标准 Transformer 架构无缝集成 | ❌ 需定制向量化模块 |
| **稳定性** | ✅ 提供理论保证（离散 $l^1$ 稳定性） | ✅ 有经典稳定性理论支持 |

> **一句话总结**：TOPOFORMER 将拓扑结构编码为 **attention-friendly 的序列**，实现了 **高效、稳定、表达力强** 的图级表示。

---

## **2. 核心实验方法和设置**

### **使用的数据集**

#### **图分类任务（Graph Classification）**
共 9 个基准数据集，涵盖化学、生物、社交网络：
- **化学分子图**：BZR, COX2, MUTAG
- **蛋白质图**：PROTEINS
- **社交网络图**：IMDB-B, IMDB-M, REDDIT-B, REDDIT-5K
- **大规模图**：OGBG-MOLHIV

#### **分子属性预测（Molecular Property Prediction, MPP）**
来自 **MoleculeNet** 的 7 个数据集：
- BBBP, Tox21, ToxCast, SIDER, ClinTox, BACE, HIV

### **实验设置和评估指标**

| 任务 | 设置 | 指标 |
|------|------|------|
| 图分类 | 10 折交叉验证（10-fold CV） | 准确率（Accuracy） |
| 分子属性预测 | 支架分割（scaffold split）或随机分割 | ROC AUC |
| OGBG-MOLHIV | 使用官方标准分割 | ROC AUC |

### **基线方法对比**
覆盖四大类 SOTA 方法：
1. **GNN 模型**：GCN, GIN, GraphSAGE, DGCNN, DiffPool, G-Mix 等
2. **拓扑方法**：PersLay, DMP, FC-V, TopoGCL, EMP, MP-HSM
3. **图核方法**：DASP (最新图核)
4. **融合/增强方法**：AutoGCL, PGOT, SubMix, KANO, MV-Mol, MolFuse

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **图分类结果（Table 2）**
| 数据集 | TOPOFORMER 最佳准确率 |
|--------|------------------------|
| BZR | **92.36%** ±4.11 |
| MUTAG | **94.68%** ±4.30 |
| PROTEINS | **77.64%** ±3.64 |
| IMDB-M | **55.40%** ±4.78 |
| REDDIT-5K | **57.99%** ±1.94 |
| **平均排名（AvR）** | **1.5**（全场第一） |
| **平均偏差（AvD）** | **0.5**（最接近最优） |

> ✅ 在 8 个数据集中取得 **第1或第2名**，在 BZR、MUTAG、PROTEINS、IMDB-M、REDDIT-B、REDDIT-5K 上刷新 SOTA。

#### **分子属性预测结果（Table 3）**
| 数据集 | TOPOFORMER\* ROC AUC |
|--------|-----------------------|
| ToxCast | **75.3%** |
| ClinTox | **96.5%** |
| BACE | **95.9%** |
| Tox21 | **82.7%**（第二） |
| **平均排名（AvR）** | **2.8** |
| **平均偏差（AvD）** | **2.5** |

> ✅ 在 ToxCast、ClinTox、BACE 上达到 SOTA，在 Tox21 排名第二，整体仅次于 KANO 和 MV-Mol。

#### **OGBG-MOLHIV 结果（Table 4）**
| 模型 | ROC AUC |
|------|---------|
| Graphormer (SOTA) | 80.51 |
| TOPOFORMER\* | **78.19** |

> ✅ 虽未超越最强 GNN，但在纯拓扑 + Transformer 框架下表现强劲，仅落后约 2.3 个百分点。

---

### **消融实验结果**

#### **(1) TOPOFORMER vs. PH 方法（Table 5 & Table 15）**
- **PH-MLP**：使用标准 PH + Betti 向量 + MLP → 表现一般
- **PH-TR**：相同 Betti 向量作为序列输入 Transformer → 性能提升，说明**序列建模有效**
- **TOPOFORMER**（Topo-Scan + Transformer）→ **进一步显著提升**

> 🔍 **结论**：性能增益主要来自 **Topo-Scan 的滑动窗口设计**，而非仅仅是使用 Transformer。

#### **(2) 不同滤波函数的影响（Table 13）**
- 单一滤波（如 HKS 或 Ollivier-Ricci）已能取得强结果
- 多滤波融合（如 HKS + O.Ricci）带来**一致但小幅提升**
> ✅ 证明多视角拓扑信息具有互补性。

#### **(3) 窗口宽度参数 $m$ 影响（Table 6）**
- $m=2$ 在多数数据集上表现最佳
- 更宽的窗口（$m=3,4$）可能引入过多噪声或模糊局部结构

> ✅ 推荐使用较小的滑动窗口以保持局部性。

#### **(4) 是否使用分子指纹（Table 7）**
- **TOPOFORMER\*** = TOPOFORMER + ECFP + 注意力融合
- 在 BACE、HIV、BBBP 上均显著优于单独模型
> ✅ 拓扑特征与领域特定指纹（如 ECFP）**互补性强**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **拓扑可以被高效编码为序列**：Topo-Scan 成功将图的多尺度拓扑结构转化为 Transformer 可处理的短序列，无需完整持久同调。
2. ✅ **滑动窗口优于传统子水平滤波**：避免了早期饱和问题，能捕捉晚期出现的连通分量和环结构（见 Fig. 2, 6, 7）。
3. ✅ **与 Transformer 天然契合**：固定长度、有序 token 序列使位置编码和自注意力机制发挥最大效用。
4. ✅ **性能与效率兼备**：在多个任务上达到或接近 SOTA，同时计算开销远低于传统 PH 方法（Table 10 显示快 2–14 倍）。
5. ✅ **理论保障**：提供了 **离散 $l^1$ 稳定性** 的理论证明，确保小扰动不会引起输出剧烈变化。

### **方法的局限性**
- 当前仅使用 **H₀ 和 H₁** 同调群，未利用更高维拓扑信息。
- 滤波函数为手工设计（如 HKS、Ollivier-Ricci），尚未引入可学习滤波。
- 主要面向**图级任务**，未验证于节点或边级任务。
- 依赖**团复形（clique complex）** 构造，可能在稀疏图上信息有限。

### **未来工作方向**
- 引入**可学习滤波函数**，实现端到端优化。
- 扩展至**动态图、异构图、时序图**等更复杂场景。
- 探索**更高维同调**或**多参数持久同调**的轻量化表示。
- 开展**大规模预训练**，迈向真正的 **Graph Foundation Model**。
- 将 Topo-Scan 思想推广至其他结构数据（如点云、网格）。

---

> 🏁 **最终评价**：  
> TOPOFORMER 是一次成功的“**拓扑 + 注意力**”范式融合，它没有追求完全替代 GNN 或 PH，而是提供了一条**轻量、高效、理论可靠**的新路径，为构建下一代图基础模型提供了重要启示。

</details>

---

### 16. [Rethinking Self-Evolution: A Constrained Exploration-Exploitation Process for Mitigating Skill Overfitting](https://arxiv.org/abs/2607.26643)

**Authors**: Hongqiang Lin, Chao Liu, Xiaofan Bai, Xuan Jin, Yuhong Li, Nenggan Zheng, Xipeng Cao  
**Category**: cs.AI  
**Published**: 2026-07-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.26643v1  

#### Abstract
Enabling large language model (LLM) agents to accumulate and reuse experience from past interactions remains a central challenge in real-world applications. A promising solution is to treat skills as trainable states and optimize them in the same way as model parameters in neural network training. H...

---

### 17. [AgenticCANN: Automated Ascend C Operator Generation via Knowledge-Augmented Agentic Evolution](https://arxiv.org/abs/2607.26661)

**Authors**: Junhao Qiu, Zidong Wang, Yansong Sun, Zhitong Ma, Ping Guo, Qingfu Zhang  
**Category**: cs.AI  
**Published**: 2026-07-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.26661v1  

#### Abstract
Ascend C operator optimization is critical for NPU (Neural Processing Unit) inference performance but requires deep hardware expertise.While large language models (LLMs) have shown promise in automated CUDA kernel generation, the fundamentally different programming model of Ascend C introduces uniqu...

---

### 18. [DualAnchor: Preserving Language Priors and Improving Lexical Fidelity in Gloss-Free Sign Language Translation](https://arxiv.org/abs/2607.27614)

**Authors**: Hongbin Zhang, Junhao Liu, Xuefeng Bai, Youcheng Pan, Yang Xiang, Kehai Chen  
**Category**: cs.CL  
**Published**: 2026-07-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.27614v1  

#### Abstract
Recent advances in large language models (LLMs) have led sign language translation (SLT), the task of converting sign-language videos into spoken-language text, to increasingly adopt LLMs as textual backbones. However, despite their strong language modeling capabilities, existing LLM-based SLT metho...

---

### 19. [Back from the Future: Key-Value Cache Management by Counter-Causal Surprise](https://arxiv.org/abs/2607.27600)

**Authors**: Stephen Gould, Anton van den Hengel  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.27600v1  

#### Abstract
Key-value (KV) cache management through compression and eviction strategies has emerged as an important research direction in recent years. Computational demands of large language models (LLMs) and their multi-modal variants during output generation can be partially alleviated by caching previous ke...

---

### 20. [Kalman Meets Curriculum: Efficient Dynamic Prompt Selection for Adaptive RL Finetuning](https://arxiv.org/abs/2607.27610)

**Authors**: Haodong Zhu, Yangyang Ren, Yanjing Li, Sheng Xu, Haiguang Liu, Linlin Yang, Baochang Zhang  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.27610v1  

#### Abstract
Reinforcement learning (RL) finetuning significantly enhances the reasoning capabilities of large language models (LLMs), yet its effectiveness critically depends on selecting prompts of appropriate difficulty for the current policy. This is challenging because prompt difficulty evolves throughout t...

---

### 21. [Class-Aware Reinforcement Learning for Counterfactual Explanation Generation](https://arxiv.org/abs/2607.27905)

**Authors**: Muhammad Adil Saleem, Syed Ali Raza, Mary-Anne Williams  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.27905v1  

#### Abstract
Counterfactual explanations (CFEs) enhance the interpretability of black-box models by generating alternative instances with adjusted feature values that achieve a contrastive outcome. Reinforcement learning (RL) offers a promising approach for CFE generation, enabling efficient exploration of count...

---

### 22. [HARGO: Heterogeneity-Aware Reward-Guided Optimization for RL Post-Training of LLMs on HPC Tasks](https://arxiv.org/abs/2607.28301)

**Authors**: Tiangang Li, Xiangbo Tian  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.28301v1  

#### Abstract
Supervised fine-tuning (SFT) can equip large language models (LLMs) with domain knowledge for high-performance computing (HPC) tasks such as data race detection and benchmark question answering. However, knowledge alone does not guarantee task-appropriate behavior: the same SFT model that correctly ...

---

### 23. [QAdapt: A Noise-Adaptive Neural Pre-Decoding Framework for Quantum Error Correction](https://arxiv.org/abs/2607.28422)

**Authors**: Ran Miao, Rui Luo, Xiaohan Shan, Xiaoming Sun  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.28422v1  

#### Abstract
Fault-tolerant quantum computing (FTQC) relies on quantum error correction to suppress physical errors and preserve logical information at scale. In practice, however, performance is constrained not only by physical noise but also by the latency of classical decoders processing rapidly generated syn...

---

### 24. [$\beta$-OPSD: Deriving with Policy Optimization, Training with Self-Distillation](https://arxiv.org/abs/2607.28582)

**Authors**: Jiawei Xu, Minghui Liu, Juzheng Zhang, Tom Goldstein, Furong Huang  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.28582v1  

#### Abstract
On-policy self-distillation (OPSD) is a promising approach to improve reasoning language models, but it remains brittle in practice: making it work reliably often requires substantial engineering effort. We identify a structural source of this difficulty: vanilla OPSD is precisely the $\beta=1$ memb...

---

### 25. [EvoPINN: Agentic Discovery of Executable Algorithms for Physics-Informed Neural Networks](https://arxiv.org/abs/2607.26490)

**Authors**: Peng Yin, Kai Li, Yifan Zhang, Jian Cheng  
**Category**: cs.AI  
**Published**: 2026-07-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.26490v1  

#### Abstract
Physics-informed neural networks (PINNs) have emerged as a powerful paradigm for solving partial differential equations (PDEs), yet their performance heavily relies on the manual, trial-and-error engineering of neural representations, loss formulations, and optimization dynamics. While Large Languag...

---

### 26. [Training Skills Like Parameters via Self-Supervised Semantic Diffusion](https://arxiv.org/abs/2607.27557)

**Authors**: Mo Li, Zixin Yin, Ting Cao, Yunxin Liu  
**Category**: cs.CL  
**Published**: 2026-07-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.27557v1  

#### Abstract
While Large Language Models (LLMs) demonstrate remarkable general instruction-following capabilities, they often fall short of human experts in highly specialized, open-ended domains such as creative screenwriting. Prior approaches typically adopt post-training, yet both supervised fine-tuning and r...

---

### 27. [Can Agents Deceive? Evaluating Reasoning and Deception in ParliamentBench using a Social Deduction Game](https://arxiv.org/abs/2607.28146)

**Authors**: Niklas Bauer, Lars Benedikt Kaesberg, Akiko Aizawa, Jan Philip Wahle, Bela Gipp, Terry Ruas  
**Category**: cs.CL  
**Published**: 2026-07-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.28146v1  

#### Abstract
As large language models (LLMs) are deployed as agents in high-stakes settings, such as medical and legal systems, understanding their deceptive capabilities is fundamental to safety. Controlled social deduction games provide a reproducible proxy for isolating and evaluating these complex adversaria...

---

### 28. [RLPF: Reinforcement Learning from Performance Feedback for Code Generation](https://arxiv.org/abs/2607.27271)

**Authors**: Huihao Jing, Haozhe Cui, Wenbin Hu, Shaojin Chen, Haochen Shi, Changxuan Fan, Yuxuan Liu, Hanyu Yang, Sirui Zhang, Ziyi Chen, Haoran Li, Yangqiu Song  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.27271v1  

#### Abstract
Code models are increasingly trained with execution feedback, but most training signals still stop at correctness. This leaves an important gap for systems code: two programs can pass the same tests while differing greatly in runtime. We study how to train code agents to prefer faster correct implem...

---

### 29. [Flat Score, Amplified Failures: How the Error Budget Masks Damage in Quantized LLM Agents](https://arxiv.org/abs/2607.27275)

**Authors**: Jiwon Jang, Kisu Yang, Heuiseok Lim, Hyunwoo Park  
**Category**: cs.LG  
**Published**: 2026-07-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.27275v1  

#### Abstract
Post-training quantization to 4-bit weights is widely reported to be nearly lossless. We test this claim for multi-turn, tool-calling agents, where it now matters most. On $\tau^2$-bench, across two open-weight model families in dense and MoE variants and two domains (eight cells, 456 episodes each,...

---

### 30. [Belief-Guided Decision Making with Uncertainty Gating in the Game of Go](https://arxiv.org/abs/2607.26946)

**Authors**: Mehrad Yaghoubi, Azam Bastanfard, Abbas Jalilvand, Ashkan Rezaei  
**Category**: cs.AI  
**Published**: 2026-07-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.26946v1  

#### Abstract
Recent advancements in Computer Go, driven by AlphaZero and MuZero, rely heavily on Monte Carlo Tree Search (MCTS) to correct the errors of the neural network policy. While effective on massive computational clusters, this dependence creates a critical bottleneck on consumer-grade hardware, where th...

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

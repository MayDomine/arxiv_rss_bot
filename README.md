# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-23 10:32:39 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [OpenAttention: One Percent Is Enough with Tail Proxies](https://arxiv.org/abs/2610.30951)

**Authors**: Hyunwoo Yu, Jongbeom Lee, Hyunbin Kim, Sungjun Park, Jaemin Choi, Jincheol Yang, Seunghun Moon, Bongjoon Hyun, Kyomin Sohn, Kyeongbo Kong, Suk-Ju Kang  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2610.30951v1  

#### Abstract
Sparse attention reads only a small subset of the key-value cache, and how to treat the unread tail remains a key challenge. Discarding the tail biases the output, and sampling it as previous approaches do fails at reads of 5% and below, where the tail likely holds heavy hitters the selector missed....

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：OpenAttention: One Percent Is Enough with Tail Proxies

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **sparse attention** 中，模型仅读取 key-value (KV) cache 的一小部分以加速推理，而忽略剩余的“尾部”（tail）token。传统方法如直接丢弃（discard）或随机采样（sampling）尾部，会导致严重的输出偏差，尤其是在极低读取密度（如 1%）下，这些方法无法有效捕捉尾部中的 **heavy hitters**（高注意力权重的 token），从而显著降低模型准确率。

### 提出了什么新方法或新思路
本文提出 **OpenAttention**，一种通过 **tail proxies** 来近似完整 softmax 分布的新框架，其核心思想是：
- 将未被选中的尾部 token 聚类，并用一个 **count-aware mean proxy**（计数感知均值代理）来代表整个聚类，而非简单丢弃或采样。
- 该 proxy 不仅保留了聚类的平均 key 和 value，还通过 `log N` 编码了聚类大小，从而更准确地估计其对最终输出的贡献。

具体流程包括：
1. **Query-Geometry Clustering (QGC)**：基于查询几何结构压缩并聚类 key tokens，使聚类结果更符合实际 attention 分数分布。
2. **ErrorGate**：根据每个聚类的 **估计 weight error** 动态分配读取预算，而非仅依据聚类中心的质量（centroid mass）。
3. **Query-Geometry Selector (QGS)**：在压缩空间中进行高效的 top-k 选择，兼顾精度与速度。
4. **Tail Proxy Construction**：对未完全读取的聚类构建 count-aware mean proxy 并参与最终的 softmax 计算。

### 相比现有方法的优势
- **准确性更高**：在极低读取密度（如 1%）下仍能保持接近 dense attention 的性能。
- **理论支持更强**：证明了单个 proxy 在低密度下的误差低于采样，且能更好地保留尾部分布信息。
- **效率提升显著**：定制的 **OpenAttention.flash** 内核在 1% 读取时解码阶段比 dense attention 快 10×。
- **通用性强**：在多种模型（Llama-3.1-8B、DeepSeek-8B、Mistral-7B）、上下文长度（16K–96K）和任务上均达到 SOTA。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要基准：**RULER-32K HARD**，包含七个对稀疏 attention 敏感的任务：
  - `qa_1`, `qa_2`（问答）
  - `vt`（value tracking）
  - `fwe`（feature-wise aggregation）
  - `niah_multikey_2/3`, `niah_multivalue`（needle-in-a-haystack）
- 其他测试集：
  - **LongBench**, **LongBench-v2**（长文本理解）
  - **LooGLE**, **HotpotQA**（多跳推理）
  - 合成生成的 **64K 和 96K RULER 数据集**

### 实验设置和评估指标
- **模型**：Llama-3.1-8B-Instruct（bf16）、DeepSeek-R1-Distill-Llama-8B、Mistral-7B-Instruct-v0.3
- **上下文长度**：从 16K 到 96K 不等
- **评估方式**：
  - Prefill 阶段使用 dense attention
  - Question chunk 和 generation 阶段应用 sparse attention
  - 所有方法共享 sink 和 local window（各 0.1%）
- **评估指标**：
  - **Per-task accuracy**（各任务准确率）
  - **Mean accuracy**（平均准确率）
  - **Decode attention speedup**（解码注意力阶段加速比）

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **HashAttention** | Top-k selector | 仅预测 top-k，丢弃 tail |
| **vAttention** | Sampling-based | 选取 top-k 后对残差 tail 进行随机采样 |
| **MagicPIG** | Sampling-based | 基于 LSH 的重要性采样 |

所有基线均基于相同的 selector（HashAttention）实现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | 总读取比例 | 平均准确率（RULER-32K HARD） | 相对于 Dense 的保留率 |
|------|------------|-------------------------------|------------------------|
| Dense Attention | 100% | 87.04 | 100% |
| **OpenAttention (ours)** | **1%** | **86.30** | **99%** |
| HashAttention | 1% | 48.22 | 55% |
| vAttention | 1% | 36.29 | 42% |
| MagicPIG | ~1.1% | 10.53 | 12% |

> ✅ **结论**：OpenAttention 在仅读取 **1% KV rows** 的情况下，达到了 **99% 的 dense 准确率**。

### 与其他方法的对比结果
- 在 **1% 读取密度** 下：
  - OpenAttention 显著优于所有基线，平均准确率高出 **38–76 个百分点**。
  - 在 `qa_2`, `mk_3`, `mv` 等任务上几乎无损。
- 在不同上下文长度下（16K → 96K）：
  - OpenAttention 始终保持接近 dense 的性能（差距 < 0.7 pts），而其他方法随长度增加急剧下降。
  - 在 96K 上，HashAttention 仅为 27.99，vAttention 为 40.20，而 OpenAttention 达到 **78.10**。

### 消融实验结果（Ablation Study）
在 **5% 读取密度** 下验证各组件作用（dense: 87.04）：

| 配置 | 平均准确率 | 提升来源 |
|------|-----------|---------|
| vAttention + HashAttention（基线） | 67.35 | — |
| + Proxies（替换采样） | 77.91 | ↑10.56 |
| + ErrorGate | 85.90 | ↑7.99 |
| + QGS | 86.73 | ↑8.82 |
| **完整 OpenAttention** | **86.82** | ↑19.47 |

此外：
- **ErrorGate 排序策略消融**：
  - 按 centroid mass 排序：85.99
  - 按 variance 排序：84.81
  - **按 estimated error（本文）排序**：**86.30** ✅

表明 **error-aware budget allocation** 是关键设计。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **“One Percent Is Enough”**：只要用好 tail 信息，即使只读取 1% 的 KV cache，也能保留 99% 的 dense accuracy。
2. **Proxy > Sampling**：在低密度下，**single proxy 的误差远低于 sampling**，甚至低于 discard；采样会因遗漏 heavy hitters 导致严重偏差。
3. **Query Geometry Matters**：聚类和选择必须考虑 query 方向，QGC 和 QGS 显著提升了聚类质量和选择效率。
4. **ErrorGate 更优的预算分配机制**：基于 estimated weight error 分配读取预算，比单纯依赖 centroid mass 更有效。
5. **高效实现可行**：通过 **OpenAttention.flash** 定制内核，在 1% 读取下实现 **10× 解码加速**。

### 方法的局限性
- **ErrorGate 使用对角方差近似**：为了效率，使用 `diag Var_b(k)` 替代完整的协方差矩阵，导致误差估计不够精确（尽管 Spearman 相关性达 0.92）。
- **静态聚类**：QGC 在 prefill 阶段完成一次聚类，未动态更新，可能不适应 decoding 过程中 query 分布的变化（但实验证明 query geometry 稳定）。
- **内存开销**：每个 cluster 存储 count、mean token 和 variance token，带来额外存储负担（约 0.3% KV cache 大小）。

### 未来工作方向
- 改进 proxy 的误差建模，例如引入非对角协方差或学习式 variance 估计。
- 探索动态聚类机制，适应 decoding 过程中的 query 演变。
- 将 OpenAttention 与 KV cache compression 方法进一步结合，实现双重优化。
- 将该思想推广至训练阶段或其他注意力变体（如 grouped-query attention）。

---

> 🔗 **项目主页**：[https://hyunwoo137.github.io/OpenAttention/](https://hyunwoo137.github.io/OpenAttention/)  
> 💻 **代码开源**：[https://github.com/hyunwoo137/OpenAttention](https://github.com/hyunwoo137/OpenAttention)

</details>

---

### 2. [BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing](https://arxiv.org/abs/2610.31633)

**Authors**: Hayden Prairie, Jacob M. Ryan, Taylor Berg-Kirkpatrick, Daniel Y. Fu  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2610.31633v1  

#### Abstract
Looped architectures scale quality while keeping parameter count fixed by repeatedly passing activations through the same block of layers. During inference, each recurrence generates independent KV-caches, causing memory to grow linearly. To address this challenge, prior work has proposed KV-sharing...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **looped language models** 中，虽然参数量固定，但推理时每一轮循环都会生成独立的 **KV-cache**，导致内存开销随循环次数线性增长。尽管已有研究表明某些模型支持 **KV-cache 共享**（即用最后一次循环的 KV-cache 替代前面所有轮次），但该性质（称为 **KV-shareability**）并非普遍成立，且其机制尚不明确。

本文旨在解决以下问题：
- 为什么有些 looped 模型具备 KV-shareability，而另一些不具备？
- 如何系统性地训练出具备 KV-shareability 的模型？

---

### 提出的新方法：BeNice!

作者提出一个两阶段方法来启用并保证 KV-shareability：

#### （1）第一阶段：Stochastic Depth 预训练
- 在预训练中随机采样每条序列的循环深度（如从 Poisson 分布中采样 $ T \sim \text{Poisson}(\mu_{\text{rec}}) $）
- 这种训练方式诱导了 **contractive residual dynamics**，使得不同深度下的隐藏状态趋向于同一个固定点（fixed point），从而让各层的 KV-cache 趋近一致。
- 为 KV-shareability 提供基础条件。

#### （2）第二阶段：BeNice! 后训练（Post-training）
- 发现即使有 contractive dynamics，直接共享 KV-cache 仍可能导致第一个循环中的 **token-independent basis shift**，破坏收敛点一致性。
- 提出 **BeNice!** 方法，在后训练中同时优化两个目标：
  - 标准语言建模损失：$ \mathcal{L}_{\text{std}} = \text{CE}(y, p_\theta^{(T)}) $
  - 辅助损失（KV-sharing 兼容性）：$ \mathcal{L}_{\text{aux}} = \text{CE}(y, p_\theta^{(1 \leftarrow T)}) $，其中第一个循环使用后续循环的 KV-cache
- 总损失为：
  $$
  \mathcal{L}_{\text{BeNice!}} = \mathcal{L}_{\text{std}} + \gamma \cdot \mathcal{L}_{\text{aux}}
  $$

---

### 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **通用性** | 不依赖特定架构设计，适用于多种 sequence mixer（如 MHA、MLA、KDA） |
| **无损压缩** | 实现近乎无损的 KV-cache 复用，显著降低内存而不牺牲质量 |
| **兼容性强** | 可与低精度量化（fp8, mxfp4）、其他 KV 压缩技术组合使用 |
| **高效实现** | 不需要逐 token 模拟 KV-sharing 推理过程即可完成训练 |

---

## 2. 核心实验方法和设置

### 数据集
- **预训练数据**：`climbix`（超过 100B tokens）
- **微调与评估数据**：
  - `fineweb-edu`（用于 finetuning）
  - 多任务基准：**CORE**（22 项任务） 和 **CORE-Extended**（共 46 项任务），涵盖常识推理、阅读理解、逻辑问答等
  - Perplexity 测试使用 `climbix` validation set

---

### 实验设置

#### 模型架构
基于 **Parcae** 架构 [50]，采用 middle-looped 结构：
- Prelude block $ P $
- Recurrent block $ R $（重复 $ T $ 次）
- Coda block $ C $

模型规模：140M, 370M, 770M, 1.3B 参数

#### 序列混合器变体
- Multi-Head Attention (MHA)
- Multi-Latent Attention (MLA) [13]
- Kimi Delta Attention (KDA) [65]

#### 评估协议
| 类型 | 协议说明 |
|------|----------|
| **Perplexity-based** | 使用标准 PPL 和 LAMBADA BPB（bits per byte） |
| **QA-based** | 对候选答案按长度归一化负对数似然排序，计算准确率（↑） |
| **KV-sharing eval** | “Shared” 表示仅保留最终循环的 KV-cache；“Unshared” 表示每个循环都保存独立 KV-cache |

#### 基线方法对比
- **Standard Finetuning**：常规微调，无辅助损失
- **Naive Patching**：手动修正第一个循环的残差偏移
- **No Sharing / Full Sharing**：是否启用 KV-cache 共享
- **不同 sequence mixer + 是否使用 BeNice!**

---

## 3. 主要实验结果和性能指标

### ✅ **端到端质量提升（Table 4 & 5）**

| 指标 | 提升幅度 |
|------|---------|
| **Core 准确率提升（最大）** | ↑8.35 pts |
| **Core-Extended 准确率提升（最大）** | ↑4.56 pts |
| **验证集 PPL 改善（相比标准微调 + 共享）** | ↓9.75 PPL points（相对下降约 45.2%） |
| **未共享情况下的质量保持** | 与标准微调相当（Δ < 0.1 PPL） |

> 示例：在 Parcae 1.3B 上，标准微调共享后 PPL 从 10.79 升至 18.52（严重退化），而 BeNice! 保持在 10.78，几乎无损。

---

### 🔋 **内存效率与推理加速（Section 5.3）**

在模拟长上下文、高并发请求场景下（Mooncake traces）：

| 指标 | 提升幅度 |
|------|--------|
| **全局 HBM 内存减少** | 最多 ↓66.2%（63.82GB → 21.60GB） |
| **缓存命中率（cache-hit rate）提升** | ↑31.4%（绝对值 +8.1 pp） |
| **吞吐量（throughput）提升** | ↑23.6% |
| **p90 Time-to-First-Token 下降** | ↓83.6%（139.2s → 23.2s） |
| **总 token 容量提升（prec=8）** | ×3.33 倍 |

---

### 🔄 **与其他 KV 压缩技术的兼容性（Figure 6 & 7）**

BeNice! 可与以下方法正交组合：

| 技术 | 组合效果 |
|------|--------|
| **Multi-Latent Attention (MLA)** | 成功启用 KV-shareability，平均仅损失 0.09 pts Core 准确率 |
| **Kimi Delta Attention (KDA)** | 惊人地发现 KDA 原生支持 KV-shareability，无需额外微调 |
| **KV Quantization（fp8, mxfp4, nvfp4）** | 与量化结合后 PPL 变化 < 0.67%，表明高度正交 |

> 表明 **KV-sharing 是一条独立且可叠加的压缩维度**

---

### 🔍 消融实验结果

#### （1）Stochastic Depth vs Static Depth（Table 8）
| 设置 | 是否具备 KV-shareability |
|------|-----------------------|
| Stochastic Depth | ✔️ 几乎无损共享 |
| Static Depth | ❌ 共享后 PPL 显著上升（+2.7~3.1） |

→ 证明 **stochastic depth 是诱导 contractive dynamics 的关键**

#### （2）BeNice! 对 fixed-point 收敛的影响（Table 10）
| 模型 | 基线相对差异 | BeNice! 后相对差异 |
|------|-------------|------------------|
| 140M | 5.26% | 0.59% |
| 1.3B | 12.53% | 0.49% |

→ BeNice! 显著缩小了共享与非共享路径之间的 fixed-point 差异

#### （3）LoRA 版本 BeNice!（Table 11）
- 使用 rank-8 LoRA 微调 key/value 投影矩阵
- 仅需 ~41K–49K 可训练参数
- 同样实现高质量 KV-shareability
- 适合低成本部署场景

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **KV-shareability 的本质是 fixed-point 收敛的一致性**
   - 当共享与非共享 KV-cache 时模型能收敛到相同 fixed point，则可安全共享。
   - Stochastic depth 有助于形成 contractive dynamics，推动状态收敛。

2. **失败主因是 token-independent basis shift**
   - 错误集中在第一个循环的第一层
   - 是一种跨样本一致的方向性偏移（error subspace 低秩）

3. **BeNice! 成功桥接 inference-memory gap**
   - 通过辅助损失显式纠正 first-recurrence 偏差
   - 实现“参数不变、循环加深、内存不增”的理想 scaling law

4. **KV-sharing 是正交压缩轴**
   - 与 MLA、KDA、量化等技术完全兼容
   - 可作为通用 inference optimization 插件

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **仅适用于 middle-looped 架构** | 对 fully-looped 或 explicit routing 模型尚未验证 |
| **依赖 careful hyperparameter tuning** | 尤其 γ（auxiliary loss weight）需精细调整 |
| **BeNice! 需额外训练成本** | 虽然后训练时间短，但仍需额外计算资源 |
| **理论边界未完全揭示** | 并非所有模型都能通过 BeNice! 修复，可能受容量或初始化影响 |

---

### 🔮 未来工作方向

1. **将 BeNice! 扩展至 fully autoregressive looped decoding**
2. **探索更轻量化的在线校正机制**（如 runtime adapter）
3. **构建理论框架预测哪些模型天然具备 KV-shareability**
4. **结合 speculative decoding 加速 shared inference**
5. **应用于 vision-language 或 multimodal looped models**

---

## 总结

> **BeNice!** 是首个系统性解决 looped model 推理内存瓶颈的工作。它揭示了 **KV-shareability 的动力学机制**，提出了 **两阶段训练范式**（stochastic depth + BeNice! post-training），实现了 **近乎无损的 KV-cache 共享**，并在内存受限场景下带来显著的 **延迟降低与吞吐提升**。更重要的是，该方法具有良好的 **模块化与兼容性**，为未来高效大模型推理提供了新范式。

</details>

---

### 3. [RIFT: KV Cache Compression for Draft Models with Complementary Compressibility Training](https://arxiv.org/abs/2610.31098)

**Authors**: Dor Tsur, Maor Ashkenazi, Maximilian Golub, Peter Hicks, Yaniv Galron, Talor Abramovich, Yoav Miron, Tiyasa Mitra, Bita Darvish Rouhani, Ran Zilberstein  
**Category**: cs.CL  
**Published**: 2026-10-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.31098v1  

#### Abstract
Speculative decoding is a popular decode acceleration method. As such, it is particularly relevant to long-output generation, which requires an increasingly large KV cache. KV cache compression offers a way to constrain costs in long-context drafting. However, compression methods that focus on prefi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RIFT: KV Cache Compression for Draft Models with Complementary Compressibility Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文聚焦于**speculative decoding**中**draft model**在长上下文生成场景下的 **KV Cache 内存瓶颈**问题。随着输入序列长度（ISL）和输出序列长度（OSL）的增长，KV Cache 占用的内存急剧增加，限制了推理效率。

然而，现有的 KV Cache 压缩方法（如 SnapKV）主要基于预填充阶段（prefill）的重要性评分进行静态选择，在长序列生成过程中，这些早期选中的 KV 会逐渐变得不相关——作者将这一现象称为 **decode staleness（解码陈旧性）**。这种陈旧性导致 draft model 与 target model 对齐能力下降，接受长度（Acceptance Length, AL）显著降低。

### 提出的新方法与新思路
为解决上述问题，作者提出了 **RIFT**（Recency-Importance Fusion and Training），一个针对 draft model 的 KV Cache 压缩策略家族，并结合 **compressibility training** 进行训练优化。

#### RIFT 家族方法
- **RIFT-stat**：一种静态但更优的分配策略。它将固定的 KV Cache 预算 $ B $ 分配给两部分：
  - 一部分用于保留最近生成的 token（Sliding Window, SW），比例由滑动窗口比率 $ p $ 控制。
  - 另一部分用于保留 prefill 阶段通过重要性评分（如 SnapKV）选出的关键 prompt token。
  > 该方法简单有效，平衡了“近期性”（recency）与“重要性”（importance）。

- **RIFT-dyn**：动态刷新机制。在生成过程中，每隔 $ \Delta $ 步对整个可用的 KV Cache（排除 SW 部分）重新计算重要性分数，并更新缓存选择。
  > 通过定期刷新，确保被保留的 prompt token 始终与当前生成状态相关，显著缓解 decode staleness。

- **RIFT-ada**：自适应刷新机制。在 RIFT-dyn 基础上，提出基于信号的触发策略，仅在必要时才执行昂贵的重评分操作。触发条件包括：
  - 接受长度低于阈值
  - draft entropy 过高
  - query drift（查询漂移）过大
  - 达到最大无刷新间隔
  > 在保持高质量的同时，大幅减少不必要的刷新开销。

#### Compressibility Training（可压缩性训练）
提出了一种**压缩感知的知识蒸馏**（compression-aware KD）训练方案：
- 在训练时模拟推理时的稀疏注意力模式。
- 先进行一次稠密前向传播获取重要性分数。
- 构建稀疏注意力掩码（结合 SW 和高分 prompt token）。
- 在稀疏掩码下进行第二次前向传播并计算损失。
> 通过这种方式，缩小了训练与推理之间的差距（training-inference mismatch），使模型在压缩 KV 下仍能保持高性能。

### 相比现有方法的优势
- **有效性**：RIFT 显著优于 SnapKV、SWA 等静态方法，也优于 StreamingLLM、H2O 等动态方法。
- **灵活性**：RIFT 是一个通用框架，可与多种 KV 选择器（如 CriticalKV、Expected Attention）结合。
- **高效性**：RIFT-ada 通过自适应刷新，在几乎不牺牲性能的前提下极大降低了延迟开销。
- **兼容性**：支持外部 draft、自推测（self-speculation）、跨模型族 draft 等多种部署场景。

---

## 2. 核心实验方法和设置

### 数据集
- 主要使用 **SPEED-Bench**，一个专为 speculative decoding 设计的多样化基准测试集，包含多种任务类型（排序、创意写作、代码调试等），特别关注长上下文（32K/100K ISL）和长输出（4K OSL）场景。
- 补充实验使用 **LongBench-v1/v2**，用于评估非 speculative 场景下的长文本理解与推理能力。

### 实验设置
- **模型对**：
  - Qwen3.5-4B → Qwen3.5-122B
  - Llama3.1-8B → Llama3.1-70B
  - Nemotron3.5-Lightning → Nemotron3-Ultra
  - 跨家族：Qwen/Llama → DeepSeek-V4
- **KV Cache 预算**：固定为 8K tokens（即 4× 压缩率），部分实验测试 2K–16K 不同预算。
- **温度**：$ T \in \{0, 1\} $
- **draft length**：默认 11
- **硬件假设**：允许 full KV history 存在于 HBM 或 offloaded memory 中以支持重评分。

### 评估指标
- **主要指标**：**Average Acceptance Length (AL)**，即每个 target 模型调用平均能接受多少个 draft token。
  - 所有压缩方法的结果均相对于 **full-KV cache** 的 AL 进行归一化比较。
- **次要指标**：
  - 刷新次数（Refreshes per response）
  - AL 与刷新次数的权衡曲线
  - 在 LongBench 上的任务准确率

### 基线方法对比
- **静态方法**：
  - **SWA**（纯滑动窗口）
  - **SnapKV**（基于观察窗口的注意力评分）
- **动态方法**：
  - **StreamingLLM**（保留 sink + SW）
  - **H2O**（近期 token + attention 重击者）
  - **TriAttention**（基于三角函数距离的 KV 压缩）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **SPEED-Bench 32K ISL, 8K KV budget, 4K OSL** 设置下（$ T=0 $）：

| 方法 | AL (% of Full-KV) |
|------|-------------------|
| Full-KV | 100% |
| SnapKV | ~63–67% |
| SWA | ~80–85% |
| **RIFT-stat** | **~87%** |
| **RIFT-dyn** | **~95%** |
| **RIFT-ada** | **~96%** |

> **RIFT-ada** 在仅使用 1/4 KV Cache 的情况下，仍能保留超过 **96% 的 full-cache 接受长度**。

在 **100K ISL** 更极端场景下，RIFT-ada 对 Qwen3.5-4B 的 AL 保留率达到 **96.0%**（vs. SnapKV 的 59.7%）。

### 与基线方法的对比结果
- **vs. SnapKV**：RIFT-stat 提升高达 **+37.8% AL**；RIFT-dyn 提升超 **+50%**。
- **vs. SWA**：RIFT-stat 在早期 generation 优于 SWA，在后期持平或略优，综合表现更佳。
- **vs. 动态基线**：
  - RIFT-dyn 和 RIFT-ada 在所有设置下均优于 **H2O** 和 **TriAttention**。
  - **StreamingLLM** 表现最弱，甚至不如 RIFT-stat。

### 消融实验结果
#### RIFT-stat 滑动窗口比率 $ p $
- 最优 $ p $ 在 0.3–0.7 之间，**$ p=0.5 $** 在多数情况下表现最佳。
- 单独使用 SnapKV 或 SWA 均不如混合策略。

#### RIFT-ada 自适应刷新优势
- 在相同 AL 下，RIFT-ada 比 RIFT-dyn **减少最多 91% 的刷新次数**。
- 在相同刷新次数下，RIFT-ada 可带来 **+0.65 AL** 的增益。
- 证明了“何时刷新”比“定期刷新”更重要。

#### Compressibility Training 效果
- 在 RIFT-stat 下，训练后 AL 提升达 **+16.19%**（Qwen3.5-4B, 2K budget, $ T=1 $）。
- 在 RIFT-dyn 下，提升达 **+16.42%**（Nemotron3.5-Lightning, 2K budget, $ T=1 $）。
- 消融实验证明，**attention-based mask** 显著优于随机或纯 SWA 掩码。

#### 跨模型族 draft 结果
| Draft → Target | SnapKV | RIFT-stat | RIFT-dyn |
|---------------|--------|----------|----------|
| Qwen3.5-4B → DSV4-Pro | 3.50 | 4.26 | 5.35 |
| Llama3.2-1B → DSV4-Flash | 2.50 | 3.24 | 3.50 |

> RIFT 同样适用于跨 tokenizer 的外部 draft 场景。

---

## 4. 关键结论和发现

### 主要发现
1. **Decode Staleness 是真实且严重的问题**：在长 OSL 生成中，基于 prefill 的 KV 选择会迅速失效，尤其在 hybrid-attention 模型中更为明显。
2. **Recency + Importance 是最优策略**：单纯依赖历史重要性或仅保留最近 token 都不够，必须两者结合。
3. **Dynamic Refresh 价值巨大**：定期刷新 KV 选择可显著恢复 draft model 性能。
4. **Adaptive Refresh 更高效**：通过信号驱动的自适应刷新，可在极低开销下实现接近动态刷新的性能。
5. **Training Matters**：通过 compressibility training 可进一步提升模型在压缩 KV 下的表现，形成“算法+训练”双重优化。

### 方法的局限性
- **依赖外部评分机制**：RIFT-dyn/ada 假设可以访问完整的 KV history 进行重评分，在某些边缘设备部署中可能受限。
- **额外延迟开销**：尽管 RIFT-ada 已优化，但重评分本身仍引入一定延迟，在极端低延迟场景中需权衡。
- **GPU-only 优势有限**：相比 MTP 类 tail-augmented drafters，外部 draft + RIFT 在纯 GPU 部署中端到端加速比可能不高，更适合异构系统（如 SRAM 加速器）。

### 未来工作方向
- 将 RIFT 的 draft-side 信号（如 query drift）与 target-side 的 verification attention 信号结合，实现更智能的联合缓存管理。
- 探索更轻量化的重评分机制，降低刷新成本。
- 扩展至更多类型的模型架构（如纯 Mamba、RWKV）和应用场景（如流式语音生成）。
- 研究多 draft layer 或 hierarchical speculative decoding 与 RIFT 的结合。

--- 

> ✅ **总结**：RIFT 通过识别并解决 **decode staleness** 问题，提出了一套从静态分配（RIFT-stat）到动态刷新（RIFT-dyn/ada）再到训练优化（Compressibility Training）的完整解决方案，在多种长上下文 speculative decoding 场景下实现了接近 full-cache 的性能，同时将 KV Cache 使用量减少 75%，是提升大模型推理效率的重要进展。

</details>

---

### 4. [MoEZip: Routing-Aware KV Cache Compression for Sparse Mixture-of-Experts LLMs](https://arxiv.org/abs/2610.30206)

**Authors**: Minsang Kim, Seungjun Baek  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.30206v1  

#### Abstract
Mixture-of-Experts (MoE) language models scale model capacity efficiently by activating a small subset of parameters per token. However, in long-context inference, memory remains a major bottleneck because the KV cache grows linearly with sequence length. Existing KV cache compression methods reduce...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MoEZip: Routing-Aware KV Cache Compression for Sparse Mixture-of-Experts LLMs

## 1. 论文的主要贡献和创新点

### 解决的问题
在稀疏 **Mixture-of-Experts (MoE)** 大语言模型中，现有的 **KV Cache 压缩**方法（如 KVZip、SnapKV）虽然能有效减少内存占用，但在高比例压缩下会显著降低生成质量。其根本原因在于：**KV 缓存的删除会扰动专家路由（expert routing）决策**，导致推理过程中激活的专家发生变化，从而引发输出质量下降。

传统方法仅基于注意力权重选择保留位置，忽略了 MoE 架构特有的“路由脆弱性”（routing fragility），即上下文扰动对路由分布的敏感性。

### 提出的新方法：MoEZip
作者提出 **MoEZip**，一种**路由感知的 KV Cache 压缩方法**，专门针对稀疏 MoE 模型设计。其核心创新包括：

- **路由脆弱性度量（Routing Fragility Metric）**：  
  提出一种基于 **Fisher Information Matrix** 的新度量 `f_t,l`，用于量化每个 token 在 MoE 层上的路由分布对微小扰动的敏感性。该度量适用于两种主流路由机制：
  - **Softmax-type routing**: $ f_{t,l} = 1 - \sum_e g_{t,l,e}^2 $
  - **Sigmoid-type routing**: $ f_{t,l} = \sum_e g_{t,l,e}(1 - g_{t,l,e}) $

- **路由感知的重要性评分（Routing-Aware Scoring）**：  
  利用指令 token（instruction tokens）的路由脆弱性，并通过指令到上下文的注意力权重进行传播，计算每个上下文位置的综合重要性得分：
  $$
  r(i) = \sum_{l \in M} \sum_{t \in I} \alpha_{t,l,i} \cdot f_{t,l}
  $$
  其中 $\alpha_{t,l,i}$ 是指令 token $t$ 对上下文位置 $i$ 的注意力分数。

- **路由感知锚点（Routing-Aware Anchors）与自适应预算分配**：  
  - 将得分最高的上下文位置保留为“锚点”，优先保护这些对路由稳定至关重要的位置。
  - 引入**自适应锚点预算** $B_f = B \cdot p(\mathcal{I})$，其中 $\mathcal{I}$ 是样本级路由脆弱性，使得更脆弱的样本分配更多锚点资源。

- **兼容性强的组合框架**：  
  MoEZip 可作为插件集成到现有压缩方法（如 KVZip、SnapKV）之上，在保留原有压缩器的基础上增强路由稳定性。

### 相比现有方法的优势
- **针对性强**：首次系统性地识别并解决了 MoE 模型中 KV 压缩导致的路由不稳定问题。
- **性能优越**：在极端压缩比下（如 10% KV 保留率）仍能保持高质量生成，显著优于现有方法。
- **通用性好**：兼容不同类型的 MoE 路由机制（softmax/sigmoid）和多种基础压缩器。
- **开销低**：仅增加少量选择时计算，不改变解码流程和缓存大小。

---

## 2. 核心实验方法和设置

### 数据集
实验在多个长上下文基准上进行，覆盖多样化任务：
- **LongBench**：12 个任务，涵盖问答（QA）、多跳推理（multi-hop）、摘要（summary）、分类、代码补全等。
- **RULER**：评估真实上下文长度能力，包含 Needle-in-a-Haystack (NIAH) 等合成检索任务。
- **NIAH (Needle in a Haystack)**：测试模型从超长上下文中提取关键信息的能力。

### 实验设置和评估指标
- **模型**：
  - `Qwen3-30B-A3B-Instruct`（softmax 路由）
  - `GLM-4.7-Flash`（sigmoid 路由）
- **KV 保留比（KV Retention Ratio）**：从 0.1 到 0.9，重点分析高压缩场景（如 10%）。
- **评估指标**：
  - LongBench：各任务对应标准指标（F1、ROUGE-L、Accuracy 等）的宏平均。
  - RULER：13 个任务的平均得分。
  - NIAH：ROUGE-L F1 分数。

### 基线方法对比
- **KVZip**：基于上下文重建的查询无关压缩。
- **SnapKV**：基于短观察窗口的注意力重要性选择。
- **PyramidKV**：基于层级信息浓度差异的动态缓存分配。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **10% KV 保留率** 下，MoEZip 相比当前最优方法 **KVZip** 的提升极为显著：

| 基准         | MoEZip 提升（绝对分） | 相对得分倍数 |
|--------------|------------------------|-------------|
| **LongBench** | **+24 points**         | **2.3×**    |
| **RULER**     | **+20 points**         | **4.9×**    |

> 示例：在 Qwen3-30B 上，LongBench 宏平均从 KVZip 的 ~19.5 提升至 MoEZip 的 **44.9**。

### 与基线方法的对比结果
- MoEZip 在所有压缩比下均优于所有基线，尤其在高压缩比（<30%）时优势巨大。
- 图 3 和图 4 显示，MoEZip 在各类任务（尤其是推理和检索密集型任务）上均保持稳健性能，而其他方法出现明显崩溃。
- 在 RULER 上，MoEZip 在低预算下达到其他方法需更高缓存才能实现的性能水平。

### 消融实验结果
#### （1）路由脆弱性评分的作用
移除基于 Fisher-trace 的脆弱性评分后，性能显著下降，尤其在高压缩比下。证明该度量对识别关键路由位置至关重要。

#### （2）自适应预算分配的作用
固定锚点预算（非自适应）会导致性能下降，说明根据样本脆弱性动态调整保护强度是有效的。

#### （3）注意力聚合方式
使用 **max pooling** 聚合跨头注意力优于 mean pooling，表明最强注意力信号更具信息量。

#### （4）兼容性验证
将 MoEZip 与 **SnapKV** 结合，依然能带来一致性能增益，证明其可作为通用增强模块。

#### （5）查询无关变体（MoEZip-Recon）
即使使用固定重建查询（支持缓存复用），MoEZip-Recon 仍比 KVZip 高 **22.44 分**（10% 保留率），达到完整版 MoEZip 性能的 88%，显示其实用价值。

---

## 4. 关键结论和发现

### 主要发现
1. **路由不稳定是 MoE 模型 KV 压缩的核心失败模式**：  
   KV 删除导致的上下文扰动会改变专家选择，这是现有方法未考虑的关键问题。
   
2. **指令 token 可作为答案路由的可靠代理**：  
   指令 token 的路由脆弱性与后续答案 token 高度相关，使其成为压缩前预测路由风险的有效信号。

3. **MoEZip 显著提升了高压缩下的鲁棒性**：  
   通过保护对路由敏感的上下文位置，MoEZip 在 10% KV 保留率下实现了接近 2–5 倍于基线的得分。

4. **Fisher-trace 是有效的脆弱性代理**：  
   实验验证了 Fisher-trace 得分越高，实际专家翻转率（expert flip rate）也越高，且与最终任务性能下降强相关。

### 方法的局限性
- **依赖内部状态访问**：需要获取 MoE 路由器输出和注意力统计，这在黑盒 API 推理环境中可能不可行。
- **额外计算开销**：尽管很小（约 +0.88 秒），但仍增加了预填充阶段的延迟。
- **假设局部扰动模型**：Fisher-trace 是一种方向无关的局部敏感性度量，未必完全刻画实际删除引起的复杂扰动。

### 未来工作方向
- 探索无需访问路由器的轻量级代理信号。
- 将路由感知思想扩展到其他稀疏架构（如稀疏注意力、动态网络）。
- 研究更精细的跨层路由保护策略。
- 结合 KV 量化（如 INT8）进一步提升整体效率。

> **源码地址**：[https://github.com/kmswin1/MoEZip](https://github.com/kmswin1/MoEZip)

</details>

---

### 5. [SpecEvo: Speculative Evolution with Large Language Models for Cost-Efficient Scientific Discovery](https://arxiv.org/abs/2610.28706)

**Authors**: Ha Minh Hieu, Tongyao Zhu, Do Xuan Long, Min-Yen Kan  
**Category**: cs.AI  
**Published**: 2026-10-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.28706v1  

#### Abstract
LLM-driven evolutionary search is a promising paradigm for scientific discovery, but existing methods remain costly and slow, often applying expensive reasoning uniformly while underusing evidence from failed, redundant, and saturated trials. We introduce Speculative Evolution (SpecEvo), a multi-LLM...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心总结：SPECEVO**

---

## **1. 主要贡献和创新点**

### **解决的问题**
现有的LLM驱动的进化搜索方法在科学发现任务中虽然有效，但存在以下问题：
- **成本高昂且效率低下**：普遍依赖昂贵的前沿模型（frontier model）进行所有推理，导致API成本过高。
- **证据利用不足**：未能充分利用失败、冗余和饱和试验中的信息来指导后续探索，导致重复错误和低效搜索。

### **提出的新方法：SPECEVO**
SPECEVO（Speculative Evolution）是一个**多LLM框架**，其核心是“**推测-再整合**”（speculate-then-consolidate, StC）的工作流，通过协调轻量级模型和前沿模型的角色，实现高效、低成本的科学发现。

#### **核心创新点**：
1. **角色分工与模型协作**：
   - **Speculators（推测者）**：由轻量级模型（lightweight model）驱动，负责并行生成和精炼候选程序，执行日常探索。
   - **Advisor（顾问）**：由轻量级模型担任，持续分析搜索历史（成功、失败、饱和方向），提炼出可复用的指导性建议。
   - **Navigator（导航器）**：由前沿模型担任，在稀疏检查点进行干预，根据停滞信号（stagnation signal）选择三种模式之一：
     - **Synthesis（合成）**：结合不同策略的优势。
     - **Surgical（手术式改进）**：对当前最优解进行精准修复。
     - **Reframe（重构）**：引入全新的算法范式以打破僵局。

2. **证据驱动的持续整合**：
   - 通过一个**自适应MAP-Elites风格的档案库**（behavioral archive）保留多样化的精英方案，避免过早收敛。
   - 维护一个**全局搜索历史**（search history），记录所有尝试，用于识别反复出现的错误和已饱和的探索方向。

### **相比现有方法的优势**
- **成本效益高**：将昂贵的前沿模型推理限制在战略性、高影响力的干预上，而将大量常规探索交给廉价的轻量级模型。
- **搜索更智能**：Advisor提供的动态指导和Navigator基于停滞的干预，使搜索过程更具适应性和方向性，减少了盲目探索。
- **性能更强**：在匹配预算下，性能优于或匹敌仅使用前沿模型的基线方法。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖了**176个真实世界的科学和算法发现任务**，分为四大类：
- **数学发现**（Mathematical Discovery）：7个问题，如圆盘填充（Circle Packing）、海布隆问题（Heilbronn Problem）。
- **系统优化**（Systems Optimization）：4个问题，来自ADRS基准套件，如专家并行负载均衡（EPLB）、模型放置（PRISM）。
- **组合优化**（Combinatorial Optimization）：36个问题，来自CO-Bench，涵盖装箱、调度、路由等。
- **科学方程发现**（Scientific Equation Discovery）：129个问题，来自LSR-Synth，覆盖化学、生物、物理和材料科学。

### **实验设置和评估指标**
- **评估预算**：在多个维度进行比较：
  - **固定墙钟时间**（3小时）
  - **固定API成本**（$10）
  - **固定评估次数**（500次）
- **主要评估指标**：
  - **性能**：最终找到的最佳解决方案的得分。
  - **API成本**：总花费（美元）。
  - **评估效率**：达到特定性能水平所需的评估次数。
  - **归一化均方误差**（NMSE）：用于方程发现任务。
  - **符号准确率**（Symbolic Accuracy, SA）：衡量发现的方程是否在数学上等价于真值。

### **基线方法对比**
- **单LLM基线**：OpenEvolve, GEPA, AdaEvolve, EvoX （均使用前沿模型）。
- **异构模型基线**：LEVI, RelayEvolve。
- **控制策略**：All-cheap（全用轻量模型）、Fixed-switch（前半段用轻量，后半段切换到前沿）、Random（随机选择）、Bandit（基于奖励的模型选择）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **成本降低**：在3小时墙钟预算下，SPECEVO的API成本比前沿模型基线降低了**50%-80%**。
- **性能提升**：
  - 在**CO-Bench**上，SPECEVO的综合得分比最强的基线高出**3.8%**。
  - 在**方程发现**（LSR-Synth）任务中，SPECEVO的评估效率比成本高效的基线高出**1.5倍**。
- **样本效率**：在500次评估时，其他基线需要额外**43%-65%**的评估才能达到SPECEVO的性能。

### **与基线方法的对比结果**
- **全面领先**：SPECEVO在大多数任务上都达到了最高或并列最高的性能，同时成本最低。
- **超越异构方法**：即使与同样使用轻量/前沿模型分工的LEVI和RelayEvolve相比，SPECEVO也表现更优。
  - 表1显示，SPECEVO在6个任务中有5个排名第一，而前沿模型调用比例仅为**2.4%**，远低于RelayEvolve的89.0%。
- **鲁棒性强**：当将前沿模型从GPT-5切换到Kimi-K2时，许多基线性能显著下降，而SPECEVO保持稳定。

### **消融实验结果**
- **移除Advisor**：平均得分下降**13.82%**（Circle Packing Rectangle），证明了失败和饱和信息指导的重要性。
- **移除Navigator**：平均得分下降**12.25%**，表明战略性干预不可或缺。
- **单一变异提示**：性能大幅下降，验证了多样化变异操作符（mutation operators）的价值。
- **不进行周期性重构**（refit）：性能下降**1.65%**，说明定期更新档案库描述符对维持多样性至关重要。
- **固定指导**（而非动态更新）：性能不如动态Advisor，甚至在某些任务上更差，表明静态建议会变得有害。

---

## **4. 关键结论和发现**

### **主要发现**
1. **轨迹条件下的模型协调是有效的**：通过持续整合搜索证据来协调不同模型的使用，是一种实现低成本科学发现的有效途径。
2. **分离探索与干预是关键**：将常规探索（speculation）和证据整合/战略干预（consolidation）分配给不同能力的模型，可以最大化性价比。
3. **动态指导优于静态规则**：Advisor根据实时搜索状态（停滞、饱和、失败）动态生成的指导，比预设的固定规则更能引导搜索走向成功。
4. **成本优势源于智能协调，而非单纯模型切换**：SPECEVO的成功不仅在于用了便宜的模型，更在于如何聪明地利用它们，并让昂贵的模型发挥最大杠杆效应。

### **方法的局限性**
- **依赖并行计算基础设施**：其墙钟效率高度依赖于可用的并行计算资源（如多个Speculator）。
- **候选评估成本未考虑**：实验假设候选程序的评估成本较低（CPU运行）。如果评估本身非常昂贵（如GPU训练、物理实验），则该框架的优势可能减弱。
- **模型生成描述的可靠性**：Advisor和Navigator依赖于模型生成的策略描述，这些描述可能存在不准确或遗漏，影响语义距离的计算和决策。

### **未来工作方向**
- 将SPECEVO应用于评估成本极高的场景（如大规模模拟或物理实验）。
- 动态调整Speculator数量和干预频率，以适应不同的硬件环境和评估成本。
- 探索代码接地的验证机制，以提高模型生成描述的可靠性。
- 将该框架扩展到更多样化的科学发现领域。

</details>

---

### 6. [Spike-Aware Gradient-Based Mixed-Precision Quantization of Spiking Neural Networks for Vision Applications](https://arxiv.org/abs/2610.30339)

**Authors**: Rajshekhar Rakshit, Devgokul B. V, Rakshitha Kalkura, Gopalakrishnan Srinivasan  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.30339v1  

#### Abstract
In recent years, Spiking Neural Networks (SNNs) have gained massive attention as an energy-efficient neural computing paradigm due to their event-driven and sparse computing capability. However, maintaining accuracy under low-precision constraints, necessary for practical deployment, has been largel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Spike-Aware Gradient-Based Mixed-Precision Quantization of Spiking Neural Networks for Vision Applications**

---

## **1. 主要贡献和创新点**

### **解决的问题**
- **SNN量化中的效率瓶颈**：现有的混合精度量化方法（如 HAWQ-V2）依赖于二阶梯度（Hessian-based）敏感性分析，计算开销大，尤其在 Spiking Neural Networks (SNNs) 中因时间展开（temporal unrolling）导致内存和计算成本随时间步数急剧增加。
- **缺乏对SNN特性的建模**：传统方法未显式考虑SNN特有的脉冲驱动动态（spike-driven dynamics），如膜电位阈值行为。

### **提出的新方法与新思路**
- **Spike-Aware Gradient-Based Mixed-Precision Quantization Framework**：
  - 利用**一阶梯度信息**（first-order gradient）结合**Fisher信息矩阵**近似层敏感性，避免昂贵的 Hessian 计算。
  - 引入**膜电位阈值裕度**（membrane threshold margin）作为SNN特有的动态特征，用于指导比特分配：阈值裕度越小（即膜电位越接近放电阈值），该层对量化扰动越敏感，应分配更高精度。
  - 设计了一个**保护因子**（protection factor）$ P = 1 + \beta(1 - m) $，将阈值裕度 $ m $ 融入比特分配过程，提升对脆弱层的保护。

### **相比现有方法的优势**
- **显著降低计算与内存开销**：仅需一次反向传播即可完成所有层的敏感性估计，无需多次Hessian-vector乘法。
- **保持甚至超越HAWQ-V2的精度表现**：在多种架构和数据集上达到相当或更优的准确率。
- **专为SNN设计**：首次将SNN的时间动态特性（spiking dynamics）显式融入混合精度量化流程，提升了方法的相关性和有效性。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **静态图像数据集**：
  - CIFAR-10 / CIFAR-100
  - ImageNet
- **神经形态数据集**：
  - DVS-CIFAR-10
  - CIFAR10-DVS
  - N-MNIST（补充材料）

### **实验设置与评估指标**
- **模型架构**：
  - CNN：VGG-16, ResNet-18/20/34
  - Vision Transformer：SpikeZip-TF (ViT-Small)
  - 其他：Inception-v3 (用于ANN对比)
- **训练范式**：
  - ANN-to-SNN Conversion：QCFS, PASCAL
  - Direct Training：TET (Temporal Efficient Training)
- **量化配置**：
  - 混合精度策略：支持 {2,4,8} 或 {1,2,4,8} bit 的候选比特宽度
  - 后训练量化（Post-Training Quantization, PTQ）
- **评估指标**：
  - Top-1 Accuracy (%)
  - 平均权重比特宽度（W-bits）
  - 压缩比（W-ratio = 32 / W-bits）
  - 峰值GPU内存消耗
  - 量化耗时（bit-allocation runtime）
  - 能效分析（基于硬件模拟器 SparTen 和 LoAS）

### **基线方法对比**
- **HAWQ-V2**：基于Hessian trace的混合精度量化，是主要对比基线。
- **Two-level Precision (W1/W2)**：固定异构精度（如首层2bit，其余8bit）。
- **QP-SNN**：统一比特宽度 + 结构化剪枝的方法。
- **Full Precision (32-bit)**：浮点基准。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | 数据集 | 方法 | W-bits | 压缩比 | 准确率 (%) |
|------|--------|------|--------|--------|------------|
| ResNet-34 | ImageNet | HAWQ-V2 (PASCAL) | 6.70 | 4.78× | 73.14 |
| ResNet-34 | ImageNet | **Ours (PASCAL)** | **6.70** | **4.78×** | **73.30** |
| SpikeZip-TF | CIFAR-10 | HAWQ-V2 | 4.47 | 7.16× | 93.60 |
| SpikeZip-TF | CIFAR-10 | **Ours** | **4.49** | **7.12×** | **94.52** |
| ResNet-20 | CIFAR-100 | HAWQ-V2 | 4.10 | 7.80× | 71.59 |
| ResNet-20 | CIFAR-100 | **Ours** | **4.16** | **7.68×** | **72.82** |

> ✅ 在多个任务上实现**更高准确率**与**相近压缩比**。

### **与基线方法的对比结果**
- **vs HAWQ-V2**：
  - 在 ResNet-34 (ImageNet) 上，**准确率更高**（73.30% vs 73.14%），压缩比相同。
  - **内存减少 2.31×**（峰值GPU内存从 26.1GB → 11.3GB）。
  - **运行速度提升约 100×**（量化时间从 22分23秒 → 13.38秒）。
- **vs Two-level (W1/W2)**：
  - 在 VGG-16 (CIFAR-10) 上，压缩比从 4× 提升至 **13.02×**，同时准确率更高（91.88% vs 91.55%）。
- **vs QP-SNN**：
  - 在 ImageNet 上，4-bit QP-SNN 准确率为 58.06%，而本文方法在平均 4.86-bit 下达到 **65.09%**，优势明显。

### **消融实验结果**
#### **(1) 阈值裕度（Threshold Margin）的影响**
- 加入阈值裕度信息后，在 VGG-16 (CIFAR-100) 上，准确率从 **57.28% 提升至 69.94%**（@ ~2.5 bits），验证其对SNN量化稳定性的重要性。

#### **(2) Surrogate Gradient 的鲁棒性**
- 使用五种不同 surrogate functions（Atan, Sigmoid, FastSigmoid, Piecewise Linear, Triangular）进行测试：
  - 层级敏感性排序的 **Spearman 秩相关系数 > 0.99**，表明排序高度一致。
  - 量化后准确率差异极小（94.01% ~ 94.05%），说明方法对 surrogate choice 不敏感。

#### **(3) 在ANN上的泛化能力**
- 尽管为SNN设计，但在标准ANN（如ResNet-20, Inception-v3）上也优于HAWQ-V2：
  - 在 ImageNet 上，压缩比提升 **13.0%**，Top-1 准确率提高 **0.43%**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **一阶梯度足以替代Hessian用于SNN混合精度量化**：尽管 Fisher 与 Hessian 数值不等，但其**相对层级敏感性排序高度一致**（Pearson ~0.90, Spearman ~0.89），足以支撑有效的比特分配。
2. **SNN特有动态可被有效利用**：引入膜电位阈值裕度能显著提升低比特下的鲁棒性，特别是在高噪声或低精度场景下。
3. **高效且实用**：所提方法以极低成本实现了与HAWQ-V2相当甚至更优的性能，更适合部署于资源受限的边缘设备和神经形态硬件。

### **方法的局限性**
- 当前框架主要面向**权重量化**，未深入探讨激活值的混合精度量化。
- 依赖校准集（calibration set）进行敏感性估计，若校准集代表性不足可能影响比特分配质量。
- 对 surrogate gradient 的理论解释仍基于经验观察，缺乏严格的数学证明。

### **未来工作方向**
- 扩展至**全网络混合精度量化**（weights + activations）。
- 探索**无监督或自适应校准机制**，减少对额外数据的依赖。
- 将该量化框架集成到**端到端训练流程**中，实现量化感知训练（QAT）版本。
- 进一步优化硬件映射，探索在 Loihi、SpiNNaker 等平台上的实际部署效果。

--- 

> 📌 **总结一句话**：  
> 本文提出了一种**高效、精准且SNN-aware的混合精度量化方法**，通过**一阶梯度 + 脉冲动态感知**，在大幅降低计算开销的同时，实现了优于或媲美HAWQ-V2的压缩与精度平衡，为SNN在边缘设备上的高效部署提供了有力工具。

</details>

---

### 7. [AQL-Lite: Enabling Efficient Knowledge Acquisition for Arabic LLMs via Data-Centric Continual Pretraining](https://arxiv.org/abs/2610.30344)

**Authors**: Houssam Eddine Boukhalfa, Mourad Bouache, Mourad Dahmane  
**Category**: cs.CL  
**Published**: 2026-10-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.30344v1  

#### Abstract
Training Arabic large language models faces challenges from noisy web data, dialectal variation, and the high cost of continual pretraining (CPT). We introduce AQL-Lite (AQL: Arabic Quality Language), a time- and resource-efficient framework for adapting a multilingual LLM to Modern Standard Arabic ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AQL-Lite: Enabling Efficient Knowledge Acquisition for Arabic LLMs via Data-Centric Continual Pretraining

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
阿拉伯语大语言模型（Arabic LLMs）的发展面临三大挑战：
- **数据质量差**：网络爬取的阿拉伯语文本包含大量噪声、模板化内容、重复文本、拼写不一致以及方言混杂等问题。
- **方言多样性**：Modern Standard Arabic (MSA) 与多种方言并存，影响模型对标准语的理解能力。
- **持续预训练（Continual Pretraining, CPT）成本高**：全参数微调计算开销巨大，而仅使用 LoRA 等参数高效方法可能导致表征更新不足。

这些问题导致单纯扩大原始 token 数量无法有效提升模型性能。

---

### 提出了什么新方法或新思路
本文提出 **AQL-Lite**，一个以数据为中心、资源高效的框架，用于将多语言 LLM 适配到 MSA。其核心由两部分构成：

#### （1）AQL 数据管道（Data Pipeline）
五阶段高质量阿拉伯语语料库构建流程：
1. **Structural Cleaning**：去除 HTML 标签、广告、重复页眉页脚等结构性噪声。
2. **Heuristic Filtering**：基于规则过滤低质量文档（如字符比例、重复率、符号密度等）。
3. **MSA/Dialect Control**：使用 MARBERTv2 分类器识别并保留 MSA 文档（p(MSA|d) > 阈值），实现 100% MSA 政策。
4. **Model-based Quality & Safety Scoring**：使用 Gemma 34B 微调的多任务模型打分，评估文档质量和安全风险（toxicity, hate, violence 等），设定阈值进行过滤。
5. **Exact + Near Deduplication**：通过哈希去重和 MinHash+LSH 聚类移除近似重复文档，提高数据多样性。

该过程从 885B 原始 token 中筛选出 215B 高质量 MSA token（保留率 24.3%）。

#### （2）AQL-Lite CPT Recipe（Partial-Parameter CPT）
结合多种参数高效技术的混合训练策略：
- **RS-LoRA Adapters**：在所有线性层上部署 Rank-Stabilized LoRA，增强稳定性。
- **Selective Unfreezing**：解冻最后 8 个 Transformer Block 和 Embedding Matrix，进行全秩更新。
- **Layer-wise Learning Rates**：为不同层级设置递减的学习率（越高层学习率越大），促进高层语义适应。
- 参数总量约为原模型的 1/3，在效率与效果间取得平衡。

---

### 相比现有方法的优势
| 方面 | AQL-Lite 的优势 |
|------|----------------|
| **数据质量** | 显著优于 raw web text 或简单清洗的数据；在相同 1B token 预算下，相比原始数据带来 +3.51 分提升。 |
| **训练效率** | 在 5B token 实验中，达到 Full CPT 94% 的性能增益，但训练时间仅为后者的 **52%**（63h vs 121h）。 |
| **性能表现** | 超过 LoRA-only 和 RS-LoRA-only 方法，在多个基准测试中全面领先。 |
| **安全性设计** | 多层次安全机制：上游数据过滤 + 安全指令微调 + 推理时轻量级安全分类器。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **原始数据来源**：Common Crawl 等互联网抓取的阿拉伯语网页，共 885B tokens。
- **最终训练数据**：经过 AQL Pipeline 清洗后的 **215B tokens**，其中用于主实验的是从中采样的：
  - **5B tokens 子集**：96% MSA + 2% code + 2% math（用于控制实验）
  - **40B tokens 子集**：100% MSA（用于大规模 CPT + Instruction Tuning）

---

### 实验设置和评估指标
#### 模型基础
- 基座模型：**Gemma 3 12B**
- 词表不变，未扩展

#### 训练配置
- 硬件：8×A100 80GB
- 序列长度：4096（packed）
- 精度：bfloat16
- 并行策略：DeepSpeed ZeRO-2
- 优化器：AdamW，梯度裁剪 1.0
- Batch Size：等效 ~1M tokens per step

#### 评估指标
- 主要指标：**macro-average accuracy (%)** 在以下四个 benchmark 上：
  - **ArabicMMLU**
  - **AlGhafa**
  - **AraTrust**
  - **Arabic EXAMS**
- 最终模型在 **OALL v2** 七项 benchmark 上综合评测：
  - 新增 MadinahQA, ALRAGE, ArbMMLU-HT

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Gemma 3 12B PT (no CPT)** | 基线模型，未经任何 CPT |
| **Full CPT** | 所有参数可训练，作为“性能上限”参考 |
| **LoRA-only** | 仅训练 LoRA 适配器 |
| **RS-LoRA** | 使用稳定缩放因子的 LoRA 版本 |
| **AQL-Lite (ours)** | 提出的方法：RS-LoRA + 最后8块全参数 + Embedding + 层级学习率 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 控制实验（5B tokens）
| 方法 | Wall-clock Time | Mean Score |
|------|----------------|------------|
| Gemma 3 12B PT | — | 66.91 |
| LoRA-only | 17h | 70.24 |
| RS-LoRA | 24h | 70.88 |
| **AQL-Lite (ours)** | **63h** | **73.19** |
| **Full CPT** | **121h** | **73.58** |

> ✅ AQL-Lite 达到 Full CPT **94% 的增益**（6.28 / 6.67 pts），但耗时仅为其 **52%**

#### 数据消融实验（1B tokens）
| 数据条件 | Mean Score | Δ vs Base |
|--------|------------|-----------|
| Raw crawl | 66.80 | -0.11 |
| Stages 1–2（结构+启发式） | 67.45 | +0.54 |
| **Stages 1–5（完整 AQL）** | **70.31** | **+3.51** |

> 📌 表明高质量数据本身即可带来显著收益，远超原始数据。

#### 组件消融实验（Table 5）
| 配置 | Mean Score | 分析 |
|------|-----------|------|
| RS-LoRA only | 70.88 | 基准 |
| Unfreeze last 8 only | 67.96 | 单独解冻顶层反而更差 → 必须配合全局 adapter 路径 |
| + Embedding training | 71.52 | ↑ +0.64 |
| + Unfrozen blocks | 71.73 | ↑ +0.85 |
| **完整 AQL-Lite** | **73.19** | ↑ +2.31 over RS-LoRA，显示组件协同效应 |

---

### 最终模型性能（40B CPT + 1M SFT）
| 模型 | OALL v2 Mean | AlGhafa | ArabicMMLU | ... | ALRAGE |
|------|--------------|---------|-------------|-----|--------|
| AQL-Lite-12B (before SFT) | 74.8 | 77.5 | 74.6 | | 74.1 |
| **AQL-Lite-12B (after SFT)** | **75.3** | **78.2** | **75.0** | | **74.8** |

> 🔍 对比其他公开模型（HELM Arabic leaderboard）：
> - 高于 **ALLaM-7B (69.9)** 和 **Jais-adapted-70B (66.3)**
> - 接近 **Llama 4 Scout (74.6)**，略低于 **Qwen3 235B (78.6)**

---

## 4. 关键结论和发现

### 主要发现
1. **数据质量 > 数据规模**  
   在相同 token 预算下，AQL 构建的高质量语料比原始网页数据多带来 **+3.51 分**，证明“better data”比“more data”更重要。

2. **Partial-Parameter CPT 是实用折中方案**  
   AQL-Lite 在不到一半的时间内恢复了 Full CPT 94% 的性能，验证了“局部全参数更新 + 全局 LoRA”的有效性。

3. **各组件均有贡献且存在协同效应**  
   - RS-LoRA 提供必要的全局调整路径；
   - 解冻顶层模块允许高层语义重构；
   - Embedding 更新有助于子词表示迁移；
   - Layer-wise LR 加速收敛。

4. **Instruction Tuning 贡献有限但必要**  
   在 OALL v2 上仅带来 +0.5 分提升，说明主要能力来自 CPT，SFT 更侧重于指令遵循与安全行为。

5. **安全需多层次保障**  
   采用“数据过滤 + 安全 SFT + 推理时安全分类器”三层防护，适用于文化敏感场景。

---

### 方法的局限性
- **未完全匹配 Full CPT 性能**：尽管达到 94%，但在某些任务上仍有差距。
- **方言支持弱**：强制 100% MSA 政策，牺牲了对阿拉伯语方言的覆盖能力。
- **部分设计未充分消融**：
  - 未搜索最优解冻层数 $N$（当前固定为 8）
  - 未单独分析 layer-wise LR 的独立作用
- **依赖外部分类器成本**：Stage 3 和 Stage 4 的模型推理成本未计入总开销。
- **评估偏倚可能**：benchmark 多为 closed-book QA，难以区分是“学到新知识”还是“更好地激活已有知识”。

---

### 未来工作方向
1. 研究可控的 **MSA-dialect 混合策略**，提升方言鲁棒性。
2. 对 **unfrozen block 数量 $N$** 进行系统性搜索。
3. 开展 **from-scratch training** 实验（小模型），验证 AQL 数据在非适配场景下的泛化能力。
4. 扩展至更大 token 预算（>40B）和更多下游任务。
5. 引入 **morphology-aware filtering**，改进阿拉伯语形态处理。
6. 加强 **bias analysis、dialect robustness、deployment safety auditing** 等现实维度评估。

--- 

> 💡 **总体评价**：AQL-Lite 成功展示了“**data-centric + partial-parameter**”范式在低资源语言适配中的巨大潜力，为高效构建高质量 Arabic LLM 提供了一条可复现、可扩展的技术路径。

</details>

---

### 8. [Compiling Autoregressive LLM Control into Policy-Exact Parallel Inference for Hierarchical RAG](https://arxiv.org/abs/2610.30664)

**Authors**: Zehao Lin, Qianfeng Shen, Xixuan Hao, Shaobo Cui, Chunyu Li, Kai Chen, Bo Tang, Zhenjun Wang, Jin Yan, Zhiyu Li, Feiyu Xiong  
**Category**: cs.CL  
**Published**: 2026-10-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.30664v1  

#### Abstract
Large language models increasingly act as controllers in structured decision pipelines, yet their decisions are still executed by autoregressive text decoding. Hierarchical RAG is one such setting: at each frontier of a document tree, an LLM commits to $b$ of $k$ sections under an input-dependent ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Compiling Autoregressive LLM Control into Policy-Exact Parallel Inference for Hierarchical RAG*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Hierarchical RAG**（分层检索增强生成）等结构化决策流程中，大型语言模型（LLM）常作为控制器选择下一步要探索的文档子树。传统方法依赖 **autoregressive decoding**（自回归解码）来生成控制指令（如 JSON 数组），导致以下三难困境：
- **Free-form JSON**：语法错误率高（3–65%），协议层失败主导系统错误。
- **Unary shortcut**（单标签打分）：虽避免解码，但丢失历史上下文，无法建模序列依赖。
- **Grammar-constrained decoding**：保证语法正确且保留历史，但必须串行生成每个动作 token，延迟高。

### 🚀 提出的新方法与核心思想
提出 **“编译而非解码”**（Compile, don’t decode）范式，将有限动作空间的 autoregressive 控制器 **编译为并行推理图**：

- **Action-Prefix Tree 编译**：将控制器的所有可达状态（即所有可能的动作前缀）构造成一棵前缀树。
- **两步执行**：
  1. **Root Prefill**：一次前向传播计算第一个动作的 logits 并构建共享 KV Cache。
  2. **Tree-Causal Probe**：将所有前缀延续序列拼接成一个 batch，在单次前向传播中并行评估所有状态下的动作分布，使用 block-causal attention mask 保证因果性。

该方法不训练、不近似原始策略，而是通过重构执行图实现 **policy-exact**（策略等价）的并行化。

### ⚡ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **Latency** | 关键路径从 `O(b)` 降为 **2** 次前向传播，独立于动作数 `b` |
| **Correctness** | 零输出 token，零协议错误（zero protocol failures） |
| **Policy Fidelity** | 在浮点精度允许范围内完全复现串行策略（exact arithmetic 下等价） |
| **功能扩展** | 支持对无序多动作进行 **order-marginalized MAP 推断** 和分支定界认证 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **QASPER** [23]：科学论文中的问答数据集，强调证据检索能力。
- 所有 span 经过 **byte-level verification**，确保引用精确。
- 使用 **五个 splits**（A–E）：
  - A/B：开发集（development）
  - C/D：官方测试集的文档不重叠拆分（document-disjoint）
  - E：预注册测试集（preregistered after compiler design）

### ⚙️ 实验设置
- **模型**：Qwen2.5-7B-Instruct 和 Qwen2.5-3B-Instruct，bf16 精度。
- **硬件平台**：
  - 主平台：MetaX C500
  - 对比平台：NVIDIA H800（验证跨设备一致性）
- **加速器配置**：eager attention，离线运行，code-hash 控制一致性。
- **评估方式**：基于 document-cluster bootstrap 进行 10k 次重采样估计置信区间。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Agreement** | 编译策略与串行策略选出的动作序列是否一致 |
| **Control Latency** | 每个问题的控制阶段 wall-clock 时间（秒） |
| **Output Tokens** | 控制阶段生成的 token 数量（串行 vs 编译） |
| **AllEvidence@B** | 在预算 B 内覆盖 ≥80% 标注证据的比例 |
| **Certification Rate** | MAP 集合被分支定界成功认证的比例 |

### 🔁 基线方法对比
| 基线 | 特点 |
|------|------|
| **Serial Grammar Decoding** | 当前部署标准，token-by-token 生成受限 JSON |
| **Idealized Chunked Engine** | 快进已知语法部分，每 slot 仍需一次 forward（理论最优串行） |
| **Unary Projection** | 单步打分所有候选，忽略历史依赖 |
| **Sequential Label Decoding** | 不带语法约束的顺序选择 |
| **Free Generation / Legacy JSON** | 自由生成或截断式 JSON 输出 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1）

| Split | Agree (比例) | Serial Lat. (s) | Compiled Lat. (s) | Speedup | Out Tok. (ser/comp) |
|-------|--------------|------------------|--------------------|---------|------------------------|
| A     | 80/90 (89%)  | 0.62             | 0.40               | **35%↓** | 11.7 / 0               |
| B     | 101/120 (84%)| 0.52             | 0.32               | **38%↓** | 10.4 / 0               |
| D     | 79/90 (88%)  | 0.56             | 0.36               | **36%↓** | 11.2 / 0               |
| E     | 52/60 (87%)  | 0.63             | 0.37               | **41%↓** | 12.9 / 0               |
| D,3B  | 78/90 (87%)  | 0.53             | 0.28               | **47%↓** | 11.2 / 0               |

> ✅ **平均降低控制延迟 35–47%**，小模型增益更大；**输出 token 从 ~12 降至 0**

### 🔍 与基线对比结果
- **vs. Idealized Chunked Engine**：
  - 在真实 workload 上仍快 **6–7%**（D: 0.37s vs 0.35s）
  - 因 chunked 仍需 `n` 次依赖 forward，而编译仅需 2 次
- **vs. Unary Shortcut**：
  - 编译同时提供 unary 与 full AR 策略输出，可用于动态切换
  - 实测 `KL(P_AR || P_unary)` 中位数 2.6 nats，p90 达 8.3 → 表明历史依赖普遍重要
- **跨设备一致性**：
  - H800 上重复实验保持相同速度排序（0.28/0.16/0.15s on D）
  - 但 serial policy 自身在不同设备间也仅有 67–73% 一致率 → 编译差异属正常精度波动

### 🔧 消融实验结果
| 实验 | 发现 |
|------|------|
| **Top-1 Readout Only**（只读第一个 slot） | AllEvidence@1K 从 51.1 降到 42.2 → 后续条件化动作携带关键信息 |
| **Different Serialization Format**（非 JSON 语法） | 性能下降至 44.4@1K → 证明必须编译实际部署的 grammar |
| **False-Nesting Corruption Test** | 编译方法保持 0 协议错误（44.4@1K），自由生成失败率达 7.8–14.4% |
| **FP32 Replay**（Split A） | logits 偏差均值从 1.2 降至 0.05，序列一致率升至 89/90 → 差异源于浮点精度 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Autoregressive 控制可被编译而非解码**：
   - 有限视野（bounded-horizon）控制器的可达状态是浅前缀树，适合一次性并行评估。
2. **Policy-Exact 并行化可行**：
   - 在 exact arithmetic 下，tree-causal probe 完全复现串行策略的 logits（Lemma 1 & Theorem 1）。
3. **显著提升效率与可靠性**：
   - 控制延迟降低 **35–47%**，完全消除协议错误。
   - 实现 **zero-output-token control**，更适合工业级部署。
4. **支持更高级推断**：
   - 可进行 **order-marginalized MAP set inference** 并用分支定界认证结果。
   - 提供 `interaction mass` 度量（`L_seq`），预测 unary shortcut 的潜在损失。

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **宽度爆炸**（Width Wall） | 当动作数 `n ≥ 4` 时，probe 行数呈阶乘增长（如 `n=4` → 362 行），总计算量剧增，失去优势 |
| **内存开销大** | `n=4` 时 probe 内存占用达 ~3.7GB，难以实用 |
| **精度敏感性** | bf16 下存在 near-tie slot 的决策翻转，虽在可接受范围，但严格场景需 fp32 或 fallback 机制 |
| **仅适用于有限动作空间** | 不适用于开放域文本生成类任务 |

### 🔮 未来工作方向
- **Staged / Chunked Execution Plans**：针对宽动作空间设计分阶段编译策略。
- **Predictive Fidelity Diagnosis**：提前预测 `L_seq` 或 margin 分布以决定是否启用编译。
- **Integration with Tree Attention Optimizations**：结合 DeFT [18] 等 IO-aware kernel 进一步优化 probe 效率。
- **Generalization to Other Agent Loops**：应用于 tool calling、routing、agenda selection 等任意有限动作 LLM 控制器。

---

## 📌 总结一句话
> 本文提出将有限动作空间的 autoregressive LLM 控制器 **编译为一次 root prefill + 一次 tree-causal probe 的并行执行图**，在 **完全保持策略等价性** 的前提下，实现 **零输出 token、零协议错误、35–47% 控制延迟下降**，为 Hierarchical RAG 及各类结构化代理控制提供了高效可靠的新型执行范式。

</details>

---

### 9. [Low-Latency MoE Inference in Space Data Centers: A Wireless-Power-Transfer Perspective](https://arxiv.org/abs/2610.30923)

**Authors**: Haotian Zheng, Zhanwei Wang, Kaibin Huang  
**Category**: cs.DC  
**Published**: 2026-10-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.30923v1  

#### Abstract
Space data centers (SpaceDCs) are emerging as platforms for large language models (LLMs), supported by continuous solar-energy harvesting and high-speed interconnections. The mixture-of-experts (MoE) architecture supports distributed LLM inference by deploying experts across satellites. However, ske...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Low-Latency MoE Inference in Space Data Centers: A Wireless-Power-Transfer Perspective**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在 **Space Data Centers (SpaceDCs)** 中，基于 **dawn-dusk sun-synchronous orbit (SSO)** 的卫星集群可实现近乎持续的太阳能采集，并支持低延迟、高带宽的 **inter-satellite communication**。然而，在部署 **Mixture-of-Experts (MoE)** 架构进行分布式大语言模型（LLM）推理时，存在以下关键挑战：

- **计算负载不均衡**：由于 token-dependent 的专家激活机制，部分卫星上的专家被频繁激活，导致局部计算功耗需求激增。
- **能源供给均匀但需求不均**：尽管所有卫星在 SSO 轨道上接收到的太阳能基本一致，但活跃专家所在的卫星可能面临 **power-constrained computation**，而空闲卫星的富余能量无法有效利用。
- **传统方法的局限性**：
  - **Expert routing / model placement**：虽能平衡负载，但会引入通信开销或改变模型输出，影响 **inference accuracy**。

因此，**如何在不改变模型结构和路由策略的前提下，动态匹配空间计算中的能源供需失配，成为提升 MoE 推理效率的关键问题**。

---

### **提出了什么新方法或新思路**
本文提出 **SpaceWPT** —— 一种基于 **laser-based Wireless Power Transfer (WPT)** 的跨卫星能量再分配框架，用于支持低延迟 MoE 推理。

#### **核心创新点**：
1. **首次将 WPT 引入 SpaceDC 中的 LLM 推理系统**  
   利用激光无线传能技术，在卫星之间实时转移电能，缓解因专家激活不均造成的“有电的地方不用，要用的地方没电”的矛盾。

2. **提出“两面水填充”（Two-Sided Water-Filling）功率控制策略**  
   - 不同于传统的单向“注水算法”，该策略允许每个卫星根据其 **WPT link efficiency** 和 **expert activation probability** 决定是否：
     - 向网关（gateway）**donate power**
     - 从网关 **receive power**
     - 或保持中立（retain）
   - 控制策略由一个全局“水位”（water level）决定，该水位由专家激活统计、任务负载和链路效率共同确定。

3. **理论建模与简化分析**
   - 推导出 **average inference latency** 的可追踪代理函数（surrogate），揭示了 **attention-expert latency tradeoff**：
     - 向 gateway 注入更多电力 → 缩短 attention 计算时间
     - 但减少 active expert 的可用电力 → 延长专家计算时间
   - 平衡这一权衡是优化总延迟的关键。

---

### **相比现有方法的优势**
| 维度 | 现有方法（如 Expert Routing） | SpaceWPT |
|------|-------------------------------|--------|
| **是否改变模型行为** | 是（修改 gate routing） | 否（保留原始 routing 和 placement） |
| **是否影响 accuracy** | 可能降低 | 完全不影响 |
| **通信开销** | 高（需迁移参数或重路由） | 无额外数据传输 |
| **灵活性** | 静态或半动态调整 | 动态响应每 token 的激活模式 |
| **硬件依赖** | 无需新增硬件 | 需要激光 WPT 收发模块 |

> ✅ **优势总结**：**SpaceWPT 在不牺牲 accuracy 和增加通信成本的前提下，通过物理层的能量调度实现了计算资源的动态再平衡**。

---

## **2. 核心实验方法和设置**

### **使用的模型与数据集**
- **模型**：
  - **Mixtral-8x7B-Instruct**（K=8 experts per layer）
  - **Phi-3.5-MoE-Instruct**（K=16 experts per layer）
- **数据集**：
  - **HumanEval**：共 164 个编程任务提示（prompts）
  - 每个 prompt 解码生成 128 个 tokens
  - 使用真实记录的 **Top-2 active expert pairs** 进行仿真驱动

> ⚠️ 注意：实验仅使用 **activation trace** 来驱动 WPT 控制逻辑，**不用于评估 accuracy**，因为 WPT 不影响模型输出。

---

### **实验设置**
- **SpaceDC 架构**：
  - 1 个 gateway satellite（运行 attention 和 gating 模块）
  - K 个 expert satellites（各运行一个 FFN expert）
  - 卫星间采用 **star topology** 的 WPT 网络（仅通过 gateway 中转）

- **轨道模型**：
  - 采用 **Hill-Clohessy-Wiltshire (HCW)** 方程模拟相对运动
  - 默认相对轨道振幅 $ p = 50 \text{m} $
  - 距离变化范围：$ d_k \in [p, 2p] $

- **通信与 WPT 参数**：
  - **FSO 数据链路速率**：100 Gbps
  - **输入/输出大小**：$ Q_d = Q_u = 4096 \times 16 $ bits
  - **WPT 效率模型**：  
    $$
    \eta_k = \eta_{\max} \cdot \min\left(1, \frac{100}{d_k}\right), \quad \eta_{\max} = 0.689
    $$
  - **太阳能预算**：每颗卫星 $ P_S = 69.09 \text{W} $
  - **GPU 模型**：基于 NVIDIA A100 校准 DVFS 曲线

---

### **评估指标**
- **End-to-End (E2E) Per-Token Inference Latency**（主指标）
- 分解为：
  - **Expert Phase Latency**：并行执行专家计算的时间（取最慢路径）
  - **Attention Phase Latency**：gateway 执行 self-attention 的时间
- **平均延迟**：对所有 token 和 orbital snapshots 取平均

---

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **No-WPT** | 每颗卫星仅使用本地太阳能 |
| **Fixed-WPT** | 空闲专家固定发送 50% 功率给 gateway，活跃专家不参与 WPT |
| **Uni-WPT** | 仅允许专家向 gateway 单向供电（受限版 SpaceWPT） |
| **SpaceWPT** | 本文提出的双向 WPT + 两面水填充控制 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 方法 | Mixtral-8x7B (ms/token) | Phi-3.5-MoE (ms/token) | 相对 No-WPT 提升 |
|------|------------------------|------------------------|------------------|
| No-WPT | 5.704 | ~3.56 | — |
| Fixed-WPT | ~5.50 | ~3.3 | ~3.6% |
| Uni-WPT | 5.232 | ~2.8 | **8.28%** |
| **SpaceWPT** | **4.624** | **~2.54** | **up to 29.25%** |

> 🔥 **最高达 29.25% 的端到端延迟下降**（Phi-3.5-MoE 在高效率 WPT 下）

---

### **与基线方法的对比结果**
- **Uni-WPT vs No-WPT**：
  - 延迟降低约 **8.28%**（Mixtral）
  - 全部来自 gateway 侧 **attention latency 减少**（接收空闲专家捐赠的 285.62W）
  - active expert 计算未加速（仍受限于本地电源）

- **SpaceWPT vs Uni-WPT**：
  - 进一步降低 **11.62%** 延迟
  - 关键在于 **反向供电能力**：gateway 将部分富余电力回传给 active experts
  - 尽管 attention latency 微增（+0.1165 ms），但专家侧最大延迟显著下降（-0.7246 ms），净收益明显

---

### **消融实验与参数敏感性分析**
#### （1）**相对轨道尺寸影响**（图8）
- 当相对轨道振幅从 40m 增加到 90m：
  - Mixtral 的增益从 **18.95% ↓ 至 7.35%**
  - Phi 的增益从 **28.13% ↓ 至 19.22%**
- **原因**：距离增大 → WPT 效率下降 → 能量转移损失增加

#### （2）**峰值 WPT 效率影响**（图9）
- 当 $\eta_{\max}$ 从 0.35 提升至 0.75：
  - Mixtral 增益从 8.78% ↑ 至 **20.21%**
  - Phi 增益从 18.80% ↑ 至 **29.25%**
- **结论**：**高效的激光 WPT 硬件是发挥 SpaceWPT 潜力的前提**

#### （3）**太阳能预算与 GPU 频率影响**（图10–11）
- 增加 $P_S$ 或提高 GPU reference frequency：
  - 所有方案延迟均下降
  - 但 **SpaceWPT 始终保持约 19%（Mixtral）~28%（Phi）的相对优势**
- **说明**：WPT 与更强硬件是互补而非替代关系

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Energy-as-a-Resource**：在 SpaceDC 中，**energy 可作为可调度的一等公民资源**，通过 WPT 实现“按需供电”。
2. ✅ **Attention-Expert Latency Tradeoff** 存在且可通过 WPT 显式优化。
3. ✅ **Two-Sided Water-Filling** 是近最优的功率控制策略，能自适应地决定每个卫星的供/受电状态。
4. ✅ **Compact Formation + High-Efficiency WPT = 最佳性能组合**：小型化卫星编队和高效激光器极大增强 WPT 效益。
5. ✅ **SpaceWPT 不影响 inference accuracy**，优于任何修改 routing 或 placement 的软件级优化。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **依赖专用 WPT 硬件** | 需要在每颗卫星上集成 high-power laser transmitter 和 PV receiver，增加成本与复杂度 |
| **当前为 star topology** | 所有 WPT 必须经过 gateway，限制了 direct expert-to-expert 能量调度的可能性 |
| **假设理想指向与跟踪** | 未考虑实际中 laser beam alignment error 对效率的影响 |
| **静态功率控制设计** | 控制策略基于长期统计设计，非完全在线 adaptive |

---

### **未来工作方向**
1. **General Topology WPT Networks**  
   允许 expert satellites 之间直接进行 WPT，构建更灵活的能量网络。

2. **Integrated Optical Links for Data & Power**  
   设计同时支持 **data communication** 和 **wireless power transfer** 的单一光学链路（如 power-over-fiber 的空间版本），降低硬件冗余。

3. **Joint Optimization with Expert Routing**  
   在保证 accuracy 的前提下，联合优化 WPT 与轻量级 routing 调整，进一步释放潜力。

4. **Dynamic WPT Control with Real-Time Feedback**  
   结合实时负载监测与预测机制，实现毫秒级响应的动态能量调度。

5. **地面验证平台开发**  
   构建地面 testbed 模拟 SpaceWPT 系统，验证激光传能与计算协同的实际可行性。

---

> 📌 **总结一句话**：  
> **SpaceWPT 开辟了一条全新的“能量即服务”（Energy-as-a-Service）路径，使 SpaceDC 能够像调度算力一样调度能源，为未来太空智能基础设施提供了重要范式转变**。

</details>

---

### 10. [GANet: A Lightweight Group Attention Network for Plant Disease Identification on Embedded Systems](https://arxiv.org/abs/2610.30225)

**Authors**: Anubhab Tripathi, Bibek Das, Asfak Ali, Sheli Shinha Chaudhuri  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.30225v1  

#### Abstract
Global warming and climate change are major catalysts for plant diseases, significantly affecting food production worldwide. Addressing this issue requires continuous monitoring and early detection of plant diseases. While recent advancements in deep learning offer potential solutions, most models d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《GANet: A Lightweight Group Attention Network for Plant Disease Identification on Embedded Systems》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **植物病害识别对农业生产至关重要**，但传统方法依赖人工观察，难以实现早期、实时监测。
- 当前主流的 **Deep Learning 模型（如 VGG、ResNet）参数量大、计算复杂度高**，难以部署在资源受限的边缘设备（如 Raspberry Pi、Jetson Nano）上。
- 多数现有研究专注于特定作物或病害类别，**泛化能力差、适用范围有限**。

### 🚀 提出的新方法与创新
本论文提出了一种轻量级网络架构 —— **GANet（Group Equivariance Attention Network）**，其核心创新包括：

#### （1）**Group Equivariant Convolution（群等变卷积）**
- 引入 **P4ConvZ2 和 P4ConvP4 层**，利用群论中的对称性（如旋转、反射）增强模型对图像变换的鲁棒性。
- 利用共享滤波器处理多种几何变换，显著减少参数数量并提升泛化能力。

#### （2）**Group Attention Block (GAB)**
- 设计新型注意力机制，通过自注意力加权突出特征图中最具判别性的区域。
- 提升关键病变区域的关注度，抑制噪声干扰。

#### （3）**Group-Wise Feature Fusion Block (FFB)**
- 融合来自不同 group convolution 层的多尺度特征（如 P4ConvZ2 与 P4ConvP4 输出），增强特征表达能力。
- 数学证明该融合操作保持 **equivariance 性质不变**，确保模型对称性不变形。

#### （4）极致轻量化设计
- 整体模型仅含 **0.001M 参数（即 1K 参数级别）**，远低于主流 CNN 模型。
- 支持高效部署于嵌入式系统，在 **Raspberry Pi 4B 达到 5 FPS，Jetson Nano 达到 11 FPS**，满足实时监控需求。

### 🔍 相比现有方法的优势
| 维度 | GANet 优势 |
|------|-----------|
| **参数量** | << 1% of ResNet/VGG，极低内存占用 |
| **计算效率** | 更少 FLOPs，适合边缘计算 |
| **准确率** | 在 PlantVillage 上达到 **99.75% 测试精度**，优于多数 SOTA 模型 |
| **部署能力** | 可直接运行于低功耗设备，支持田间实时检测 |

---

## 2. 核心实验方法和设置

### 📁 数据集
- 使用 **PlantVillage dataset** [23]：
  - 包含 **54,000+ 高分辨率叶片图像**
  - 涵盖 **14 种农作物**，共 **38 类病害 + 健康状态**
  - 图像已标注，类别分布均衡，是农业 DL 领域的标准 benchmark

### ⚙️ 实验设置
| 配置项 | 设置值 |
|--------|-------|
| 输入分辨率 | 224×224 |
| 批大小（Batch Size） | 128 |
| 优化器 | Adam |
| 初始学习率 | 0.001 |
| 学习率调度 | ReduceLROnPlateau（当验证损失不降时衰减） |
| 训练轮次（Epochs） | 100 |
| Dropout rate | 0.5（全连接层） |
| 数据增强 | Random Rotation (≤15°), Horizontal Flip, Normalization |

### 📊 评估指标
- **Accuracy（准确率）**
- **Precision（精确率）**
- **Recall（召回率）**
- **F1-Score**
- **AUC / AUPRC**
- **Confusion Matrix**
- **Grad-CAM 可视化**
- **t-SNE 特征空间可视化**
- **ROC 曲线（One-vs-Rest）**

### 🆚 基线方法对比
论文将 GANet 与以下主流模型进行比较：
- VGG-16 / VGG-19
- ResNet-50 / ResNet-152
- DenseNet-201
- InceptionV3
- Xception
- MobileNet / MobileNetV2
- Deep CNN

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据
| 指标 | 结果 |
|------|-----|
| **测试 Accuracy** | **99.75%** |
| **Loss** | 0.009 |
| **参数量（#Parameters）** | **0.001M（1,000）** |
| **模型大小** | ~4KB（估算） |
| **FLOPs** | 极低（未明确给出，但远小于 MobileNet） |
| **推理速度（Raspberry Pi 4B）** | **5 FPS** |
| **推理速度（Jetson Nano）** | **11 FPS** |

> 注：表 II 中 GANet 条目部分字段为空（如 Precision、Recall），表明作者可能仅报告了 Accuracy。

### 📈 与基线方法对比结果（Table II）
| 模型 | Accuracy | 参数量（约） |
|------|----------|-------------|
| VGG-19 | 96% | 143.7M |
| ResNet-152 | 95% | 58.8M |
| DenseNet-201 | 94% | 20.2M |
| MobileNetV2 | 90% | 3.5M |
| **GANet (Ours)** | **99.75%** | **0.001M** |

✅ 显著优势：
- 准确率最高（**超越所有对比模型**）
- 参数量仅为 ResNet-50 的 **1/25,000**，MobileNet 的 **1/3,500**
- 实现“高精度 + 轻量化”双重突破

### 🔍 消融实验分析（文中未提供显式消融实验表格，但从结构可推断）  
尽管没有独立的 ablation study 表格，但模型设计逻辑体现了模块有效性验证：

| 模块 | 功能作用 | 推断贡献 |
|------|--------|---------|
| **Group Convolution** | 利用对称性共享权重，降低参数量 | 是实现轻量化的基础 |
| **Feature Fusion Block (FFB)** | 融合多尺度 group 特征 | 提升特征丰富度，增强分类边界 |
| **Group Attention Block (GAB)** | 注意力聚焦关键区域 | 提高判别力，减少误分类 |
| **PGSM Pooling** | 平面群最大池化 | 保留等变性的同时下采样 |

此外，**Grad-CAM 可视化**显示模型能精准定位病斑区域（如叶锈、霉斑），说明注意力机制有效引导决策。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **GANet 成功实现了高精度与超轻量化的统一**：
   - 在仅 **0.001M 参数**下取得 **99.75% 准确率**，刷新 PlantVillage 小模型性能上限。
2. **Group-equivariant CNN + Attention 的结合极具潜力**：
   - 利用图像内在对称性（rotation/reflection/flipping）提升泛化能力。
   - 注意力机制进一步强化关键特征响应。
3. **适用于边缘部署的真实场景应用**：
   - 在 Raspberry Pi 4B 和 Jetson Nano 上实现实时推理（5–11 FPS），具备田间部署可行性。
4. **无需大量数据增强即可获得优异表现**：
   - 得益于 group equivariance 的内在不变性建模，减少了对外部数据增强的依赖。

### ⚠️ 方法的局限性
- **缺乏显式的消融实验（Ablation Study）**：未能定量评估各模块（如 GAB、FFB）对最终性能的具体贡献。
- **仅在单一数据集（PlantVillage）上验证**：尚未在跨域、野外复杂光照/遮挡条件下测试鲁棒性。
- **未公开完整代码细节**：虽声明代码发布于 Kaggle Notebook，但文中未附链接，复现难度较高。
- **未测试更多边缘平台**：仅测试两种设备，缺乏在手机端或其他 MCU 上的表现数据。

### 🔮 未来工作方向（作者建议）
1. **集成 Real-time Image Processing 技术**：
   - 进一步降低延迟与带宽消耗，构建端到端实时监测系统。
2. **探索 One-shot / Few-shot Learning**：
   - 应对新出现病害样本稀少问题，提升模型适应性。
3. **扩展至其他植物器官或多模态输入**：
   - 如茎、果实图像，或结合红外、高光谱成像。
4. **开发移动端 App 或无人机巡检系统**：
   - 推动研究成果向实际农业应用场景落地转化。

---

## ✅ 总结一句话
> **GANet 通过引入 Group Equivariant Convolution 与 Group Attention Mechanism，在极低参数量（0.001M）下实现了 99.75% 的植物病害识别准确率，并成功部署于嵌入式设备，为农业边缘智能提供了高效、实用的新范式。**

</details>

---

### 11. [METTA: Memory-Enhanced Test-Time Adaptation for Meta-Reinforcement Learning](https://arxiv.org/abs/2610.31656)

**Authors**: Ali Najar, Hosein Hasani, Amir Kooshan Fattah, Yasamin Rajabi, Mahdieh Soleymani Baghshah  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.31656v1  

#### Abstract
Meta-reinforcement learning aims to train agents that adapt quickly to unseen tasks from limited interaction, but this requires representing task evidence over long interaction histories. Classical recurrent agents compress experience into hidden states, while transformer-based agents retrieve from ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：METTA: Memory-Enhanced Test-Time Adaptation for Meta-Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Meta-Reinforcement Learning（Meta-RL）旨在让智能体在面对**未见过的任务**时能快速适应，但传统方法面临以下挑战：
- **任务证据稀疏且延迟**：关键信息可能分布在长序列交互中，难以捕捉。
- **计算开销大**：Transformer-based 方法虽能建模长程依赖，但其自注意力机制具有 $O(n^2)$ 的时间与内存复杂度，难以扩展到长上下文场景。
- **记忆容量受限**：RNN 类模型将历史压缩为固定大小的隐藏状态，导致信息丢失。

### 🚀 提出的新方法：METTA
提出 **Memory-Enhanced Test-Time Adaptation (METTA)**，一种基于神经记忆（neural memory）的 Meta-RL 框架，核心思想是将“快速适应”分解为两个时间尺度：

1. **Within-Episode Fine-Grained Adaptation**  
   使用 **Test-Time Training (TTT) 层**作为因果神经记忆模块，在 episode 内部进行细粒度的在线更新。该模块以线性复杂度处理序列，并通过梯度驱动的方式持续优化内部记忆模型。

2. **Cross-Episode Long-Range Inference**  
   利用每个 episode 结束时的最终表示作为**摘要（summary）**，并通过跨 episode 聚合机制（如 attention 或 EMA）实现长期任务推理，避免对完整交互历史进行全量 attention。

> 💡 **关键创新**：首次将 TTT 引入 Meta-RL，结合显式的 episode-level 摘要机制，实现了高效、可扩展的长上下文适应。

### 🔍 相比现有方法的优势
| 特性 | METTA | RNN-based (e.g., RL2) | Transformer-based (e.g., TrMRL, ECET) |
|------|-------|------------------------|------------------------------------------|
| 序列复杂度 | $O(ET)$ | $O(ET)$ | $O(ET^2)$ 或更高 |
| 是否支持 test-time 训练 | ✅ 是（TTT 在推理时仍可更新） | ❌ 否（被动隐藏状态） | ❌ 否 |
| 长期记忆能力 | ✅ 显式存储 episode 摘要 | ❌ 压缩至隐藏状态 | ✅ 可访问全部历史，但成本高 |
| 推理效率 | ⭐ 高（仅关注摘要） | 高 | 低（需处理整个轨迹） |

---

## 2. 核心实验方法和设置

### 📚 数据集
在多个标准 Meta-RL 基准上评估，涵盖不同任务类型：
- **MuJoCo HalfCheetahVel-v4**：连续控制任务，目标速度未知，需从交互中推断。
- **Meta-World ML1 / ML10 / ML45**：机器人操作任务集合，其中：
  - ML1：单任务族，goal 坐标被屏蔽。
  - ML10/ML45：多任务族，训练与测试包含**未见任务家族（held-out task families）**，用于评估 OOD 泛化。
- **PointMaze**：稀疏奖励导航任务，agent 必须探索迷宫找到目标，仅在到达终点时获得 +1 回报，适合测试长上下文学习能力。

### 🧪 实验设置与评估指标
- **训练预算限制**：强调低样本效率下的快速适应。
- **评估方式**：
  - **Success Rate**：任务成功完成的比例（尤其用于 ML 和 PointMaze）。
  - **Average Return**：累积回报（用于 HalfCheetahVel）。
  - **Coverage / Episode Jaccard / Steps**：用于量化 PointMaze 中的探索行为多样性与效率。
- **五次随机种子平均结果**，误差带为 ±1 SEM。

### 🆚 对比的基线方法
| 类型 | 方法 |
|------|------|
| Recurrent | RL2, GRU |
| Gradient-based | MAML, PPO |
| Latent Context | PEARL, VariBAD |
| Transformer-based | TrMRL, ECET, AMAGO |
| 其他 | Meta-Learning baselines |

所有方法使用官方推荐配置，在相同环境交互预算下比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ Meta-World 表现
| 方法 | ML10 Test Success Rate | ML45 Test Success Rate |
|------|-------------------------|--------------------------|
| **METTA (ours)** | **>50%** | **>50%** |
| ECET | ~30% | ~30% |
| TrMRL | ~20% | ~20% |
| VariBAD | ~25% | ~25% |
| AMAGO | ~35% | ~30% |

> 💥 在 **未见任务家族（held-out task families）** 上，METTA 显著优于所有基线，表明其具备更强的 **task-family-level OOD generalization** 能力。

#### ✅ PointMaze 探索表现（1.5M 环境步后）
| Method | Coverage ↑ | Episode Jaccard ↓ | Steps to Success ↓ |
|--------|------------|--------------------|---------------------|
| **METTA** | **0.447±0.036** | **0.792±0.022** | **194.8±30.1** |
| ECET | 0.250±0.087 | 0.935±0.028 | 427.3±44.5 |
| TrMRL | 0.330±0.031 | 1.000±0.000 | 373.4±16.1 |

> ✅ METTA 实现了最广覆盖、最少重复路径、最快达成目标，说明其能有效复用早期探索信息。

#### ✅ 训练效率（ML10）
- **GPU 小时消耗**：METTA 显著低于 TrMRL 和 ECET。
- **Wall-clock 时间短**，接近传统 PPO 方法，远快于 Transformer-based 方法。
- 图 7 显示 METTA 在更低训练成本下达到更高成功率。

---

### 🔬 消融实验结果

#### Ablation 1: 是否启用 Test-Time 更新（TTT 在线训练）
| 设置 | PointMaze Test 性能 | ML10 Test 性能 |
|------|----------------------|----------------|
| 完整 METTA | 高 | 高 |
| 关闭 TTT 在线更新 | ❌ 大幅下降 | 影响较小 |

> 🔍 发现：当任务信息稀疏（如 PointMaze）时，**within-episode 的 test-time adaptation 极其重要**；而在目标可观测任务中（如 ML10），cross-episode aggregation 更主导。

#### Ablation 2: 不同 within-episode 记忆架构
| 架构 | PointMaze Test | ML10 Test |
|------|----------------|-----------|
| TTT-MLP（默认） | 最优 | 最优 |
| TTT-Linear | 次优 | 次优 |
| GRU | 较差 | 较差 |
| Mamba | 中等 | 中等 |
| Linear Transformer | 接近 TTT | 接近 TTT |

> ✅ TTT 架构在 test-time adaptation 上表现最佳，验证了其表达能力和适应性优势。

#### Ablation 3: Cross-Episode Aggregation 方式
| 聚合方式 | 性能 |
|--------|------|
| Attention（默认） | ✅ 最强 |
| EMA ($\alpha=0.5$) | 接近，但在 PointMaze 不稳定 |
| Linear Attention | 接近 attention，但泛化略弱 |
| None（无聚合） | ❌ 显著下降 |

> ✅ 显式利用过往 episode 摘要是必要的，而 attention 提供了最优选择机制。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **双时间尺度适应机制有效**：分离 within-episode 和 cross-episode 适应，可在保持线性复杂度的同时实现强大的长上下文建模。
2. **Test-Time Training 是关键**：TTT 层允许在推理过程中动态更新记忆，显著提升对稀疏信号任务的适应能力。
3. **OOD 泛化能力强**：在未见任务家族上表现优异，说明 METTA 学会了更通用的适应策略，而非简单记忆训练模式。
4. **高效的探索行为**：在 PointMaze 中展现出更结构化的探索，减少重复路径，提高信息利用率。
5. **计算效率高**：相比 Transformer-based 方法，训练和推理开销大幅降低，更适合实际部署。

### ⚠️ 方法的局限性
- 当前框架假设每个 trial 内任务固定，尚未考虑任务动态变化或非平稳环境。
- TTT 的每步内循环更新带来一定常数级开销，在极短序列任务中可能不如轻量 RNN。
- 所有实验基于离散 trial 设定，未测试持续学习（continual learning）场景。

### 🔮 未来工作方向
- 将 METTA 扩展至 **continual meta-RL** 场景，支持任务流中的在线适应。
- 引入 **recency bias 或 temporal weighting** 到 cross-episode memory，增强对近期经验的关注。
- 探索更高效的 memory initialization 和 reset 策略。
- 将 METTA 应用于真实世界机器人控制、自动驾驶等需要快速适应的领域。

---

> 📌 **一句话总结**：  
> **METTA 通过引入 Test-Time Training 和双尺度记忆机制，在保持线性复杂度的前提下，实现了高效、可扩展、强泛化的 Meta-RL，特别适用于长上下文、稀疏奖励、未见任务家族等挑战性场景。**

</details>

---

### 12. [From Prompts to Posteriors: Can LLM Agents Perform Bayesian Inference in Context?](https://arxiv.org/abs/2610.29331)

**Authors**: Linus Filbry, Luckeciano C. Melo, Yarin Gal  
**Category**: cs.AI  
**Published**: 2026-10-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.29331v1  

#### Abstract
Large language models (LLMs) are increasingly applied across the scientific discovery pipeline, from synthesizing literature to generating hypotheses to informing experimental design. A unifying principle behind these tasks is the need to continuously update beliefs as evidence accumulates, for whic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Prompts to Posteriors: Can LLM Agents Perform Bayesian Inference in Context?

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统地探讨了一个关键问题：**大型语言模型（LLMs）是否能在上下文中执行或近似贝叶斯推理（Bayesian Inference）？**  
当前许多基于LLM的“AI科学家”系统在科学发现流程中依赖启发式方法进行假设评估，而非通过概率信念更新来理性地整合证据。这种非贝叶斯更新可能导致逻辑不一致和认知偏差。

本文挑战了“LLMs无法在上下文中执行贝叶斯推理”的普遍假设，并首次对LLM的**内部一致性**与**外部后验恢复能力**进行了统一评估。

---

### 提出了什么新方法或新思路

#### ✅ 提出三轴评估框架（Three-Axis Evaluation Framework）
从三个递进维度全面评估LLM的贝叶斯推理能力：
1. **Probabilistic Coherence（概率一致性）**  
   使用 **Dutch Book** 构造检测LLM输出的概率判断是否违反基本概率公理（如归一化、单调性、外延性）。
2. **Posterior Fidelity（后验保真度）**  
   将LLM生成的后验分布与一个**贝叶斯Oracle**计算的理论后验进行比较，使用 **KL散度** 作为衡量标准。
3. **Downstream Task Performance（下游任务表现）**  
   在自适应实验设计（Adaptive Experimental Design）任务中，评估非贝叶斯更新是否导致长期性能下降。

#### ✅ 构建新型仿真环境：ProteinWorld
- 一个高度可配置的合成科学发现环境，模拟蛋白质分类过程。
- 支持两种难度模式：
  - **Simple**: 后验有闭式解（closed-form analytical solution）
  - **Hard**: 后验需通过蒙特卡洛（MC）等数值方法逼近，分析不可达
- 可控性强，支持对连续潜变量、复杂观测模型和属性依赖关系建模。

#### ✅ 重新审视Belief-Updating Harnesses的有效性
提出：对于具备现代推理能力和工具调用能力的LLM，现有的严格结构化信念更新框架（如BED-LLM）可能已不再必要。

---

### 相比现有方法的优势

| 方面 | 本文优势 |
|------|---------|
| **评估维度** | 首次同时考察内部逻辑一致性和外部后验恢复能力，超越仅关注准确率或校准性的传统研究 |
| **实验设计** | 引入ProteinWorld实现可控的、程序生成的OOD（out-of-distribution）任务，避免预训练污染 |
| **方法论洞察** | 揭示了**reasoning** 和 **tool use** 是提升LLM贝叶斯能力的关键因素，而非简单微调或外部约束 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集 / 环境

#### 🧩 20 Questions (20Q)
- 经典猜动物游戏，构建于Choudhury et al. (2026) 的协议之上。
- 假设空间 $\mathcal{O}$ = 所有现存动物（无限大），实验空间 $\mathcal{E}$ = 所有可能的是/否问题。
- 用于测试LLM在开放、动态假设空间中的信念更新能力。

#### 🧫 ProteinWorld（本文提出）
- 新型合成科学发现环境，模拟基于潜属性测量的蛋白质分类。
- 参数控制：
  - $|\mathcal{O}| = 8$ 类蛋白，$|\mathcal{E}| = 6$ 种实验
  - 潜属性维度为8，属性间存在因果依赖
  - 观测模型可设定为高斯、均匀或伯努利分布
- Simple版：似然函数可解析求解；Hard版：必须使用MC采样估计

#### 📊 ForecastBench
- 来自Karger et al. (2025) 的政治、经济、科技等领域二元预测题库（共30个场景）
- 用于跨对话（cross-elicitation）和单对话（single-dialogue）下的概率一致性测试

---

### 实验设置和评估指标

#### 模型选择
涵盖多个发布年份、参数量级和架构的主流LLM：
- **Open-source**: Qwen-2.5-32B-Instruct, Phi-4, Gemma-4-31B-it, Gemma-4-26B-A4B-it, Qwen-3.6-35B-A3B
- **Proprietary**: Gemini-2.5-Pro（支持Python代码执行）

所有模型均分别在**启用推理模式（reasoning mode）** 和 **禁用推理** 下进行测试。

---

#### 评估指标

| 维度 | 指标 | 公式/说明 |
|------|------|----------|
| **Probabilistic Coherence** | Dutch Book Violation Rate | 判断是否违反归一化、单调性、外延性 |
| **Posterior Fidelity** | Forward KL Divergence | $ \text{KL}(p_{\text{ora-pas}} \| p_{\text{LLM}}) $，强调不过早排除正确假设 |
| **Downstream Performance** | Posterior-Mass Gap | $ p_{\text{ora-act}}(\theta^*) - p_{\text{LLM}}(\theta^*) $，反映最终识别准确率差异 |

---

### 基线方法对比

- **Non-reasoning LLMs** vs. **Reasoning-enabled LLMs**
- **With Code Execution** vs. **Without**
- **Naive Full Belief Update** vs. **BED-LLM-style Harness**

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔹 在 **20Q** 和 **ProteinWorld-Simple** 上的结果（无推理模式）
- **KL散度迅速上升**：仅经过几次实验后，$\text{KL} > 1.0$，表明严重偏离贝叶斯更新路径
- **下游性能显著下降**：平均 $p(\theta^*)$ 差距超过30%，弱模型接近0%准确率
- **内部不一致**：跨对话Dutch Book违规率 >50%

> ❌ 结论：**Without reasoning, LLMs fail to perform accurate Bayesian inference.**

---

#### 🔹 启用 **Reasoning Mode** 后的表现
| 模型 | KL散度（Simple） | 准确率差距 |
|------|------------------|------------|
| Gemma-4-31B-it | ≈0 | <5% |
| Qwen-3.6-35B-A3B | ≈0 | <5% |
| Gemma-4-26B-A4B | <0.1 | <10% |

- **Posterior Fidelity** 显著改善，KL趋近于0
- **Downstream Task Performance** 接近Oracle水平
- 错误主要来自数值计算失误，而非概念错误

> ✅ 结论：**Reasoning enables near-exact Bayesian inference for analytically tractable posteriors.**

---

#### 🔹 在 **ProteinWorld-Hard** 上的表现（需MC采样）

| 设置 | KL散度 | 表现 |
|------|--------|------|
| Reasoning Only | 高（>0.8） | 严重偏差，常将无效假设赋零质量 |
| Reasoning + Code Execution | ≈0 | 匹配Oracle |

- **Failure Mode 1（幻觉）**：模型声称运行MC采样但未实际执行，报告虚构结果
- **Failure Mode 2（过度简化）**：忽略方差影响，直接代入均值判断，导致不可逆删除有效假设
- **Success with Tools**：Gemini-2.5-Pro 能编写正确的嵌套MC估计器并获得高质量近似后验

> ✅ 结论：**With code execution, LLMs can approximate intractable posteriors effectively.**

---

#### 🔹 与 **BED-LLM** Harness 的对比（图7）

| 方法 | 准确率趋势 |
|------|-----------|
| BED-LLM（结构化Harness） | 中等稳定增长 |
| Naive Baseline + Reasoning | **持平甚至反超** |

- 当前先进LLM无需外部结构化框架即可完成高质量信念更新
- 结构化Harness带来的增益消失

> ✅ 结论：**Modern reasoning LLMs may render strict belief-updating harnesses unnecessary.**

---

## 4. 关键结论和发现

### 主要发现

| 编号 | 发现 |
|------|------|
| **Result 1** | 在单一对话内，现代推理型LLM能保持概率一致性（within-dialogue coherence） |
| **Result 2** | 无推理能力时，LLM严重违反Bayes' Rule，导致KL发散和下游性能崩溃 |
| **Result 3** | 启用**native reasoning**后，LLM可在简单任务上精确恢复闭式后验，维持长达10步的一致更新 |
| **Result 4** | 配合**code execution**，LLM能主动诊断分析不可达性并调用工具近似复杂后验 |
| **Result 5** | 当前belief-updating harnesses（如BED-LLM）对现代LLM可能已非必需 |

---

### 方法的局限性

1. **计算成本高昂**  
   即使正确推理，LLM常会重复验证数千token，效率低下，不适合长周期科学探索。

2. **工具依赖性强**  
   对于分析不可达问题，若无代码执行权限，LLM倾向于幻觉或粗略近似。

3. **动态假设空间处理仍需工程技巧**  
   如20Q中需维护动态子集 $O_k$，并非真正意义上的无限空间贝叶斯更新。

4. **评估集中在理想化环境**  
   ProteinWorld虽灵活但仍为人工构造，真实世界科学推理更复杂。

---

### 未来工作方向

1. **Develop Lighter-weight Reasoning Mechanisms**  
   设计更高效的思维链压缩与终止机制，减少冗余计算。

2. **Build Integrated Agent Frameworks with Explicit Belief States**  
   结合LLM的推理能力与显式信念管理，支持跨会话一致性。

3. **Extend to Real-world Scientific Workflows**  
   将框架应用于真实科研场景，如药物筛选、基因功能预测等。

4. **Study Long-term Drift in Belief Updates**  
   探索数十乃至上百轮实验后的信念漂移行为。

5. **Improve Tool Use Reliability**  
   减少幻觉，增强LLM对自身能力边界的认知（meta-cognition）。

---

> 💡 **总体结论**：  
> 当前最先进的LLM，在配备**reasoning能力**和**工具接口**的前提下，已经能够有效地在上下文中执行复杂的贝叶斯信念更新，甚至逼近分析不可达后验。这标志着我们应重新思考如何构建下一代“AI科学家”——从被动辅助走向真正的**自主概率推理代理**。

</details>

---

### 13. [ULME: Predictive Coupling for Mutually Reinforcing Embedding and Language Modeling](https://arxiv.org/abs/2610.30705)

**Authors**: Yang Liu, Zhiqiang Yuan, Shu-Dong Huang, Jiancheng Lv, Liwei Yang, Qiwei Yan, Kailin Lv, Jinchao Zhang  
**Category**: cs.CL  
**Published**: 2026-10-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.30705v1  

#### Abstract
Text embedding and text generation are two of the most useful capabilities of modern large language models, yet they are typically provided by separate specialized models (an embedder and a generator) trained with incompatible objectives and deployed side by side in retrieval-augmented generation an...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ULME: Predictive Coupling for Mutually Reinforcing Embedding and Language Modeling

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代大语言模型（LLM）系统通常将 **text embedding** 和 **text generation** 能力分离为两个独立的模型：一个用于检索的 **embedder** 和一个用于生成答案的 **generator**。这种分离导致：
- 两者训练目标不兼容（embedding 用 contrastive loss，generation 用 language modeling loss）；
- 表示空间无法共享语义信息，即生成过程不能反向增强嵌入质量；
- 在 **retrieval-augmented generation (RAG)** 或 **agent systems** 中造成冗余和效率低下。

尽管已有统一模型（如 GritLM、GEM），但它们仅实现参数共享，未真正让生成“塑造”嵌入。

---

### 提出了什么新方法或新思路
论文提出 **ULME**（Unifying Language Modeling and Embedding），一种基于 **causal decoder-only LLM** 的后训练框架，通过 **predictive coupling** 实现 embedding 与 generation 的**相互强化**。

#### 核心创新：
1. **Predictive Coupling（预测性耦合）**  
   引入一个带 **stop-gradient** 的 InfoNCE 损失，使生成侧的答案表示（answer-side hidden states）作为监督信号，指导查询侧的一个专用槽位（slot）学习更丰富的推理性嵌入。这实现了：
   - 生成主动“塑造”嵌入；
   - 不干扰生成本身的训练稳定性。

2. **可学习槽链（Learnable Slot Chain）**  
   在 query 和 answer 之间插入 $N$ 个可学习的 `<EMB>` 槽位，形成一条共享潜变量路径：
   - **Slot 0**：语义嵌入（semantic embedding），对齐文档表示（via $ \mathcal{L}_{\text{sem}} $）；
   - **Slot N-1**：推理嵌入（reasoning embedding），由生成结果监督（via $ \mathcal{L}_{\text{pred}} $）；
   - 所有 token 注意力仍遵循 causal attention，无需切换 attention 模式。

3. **三重损失联合优化**
   - $ \mathcal{L}_{\text{sem}} $：对比损失，锚定语义槽与文档空间；
   - $ \mathcal{L}_{\text{gen}} $：标准语言建模损失，监督答案生成；
   - $ \mathcal{L}_{\text{pred}} $：stop-gradient 预测损失，对齐推理槽与答案池化表示。

> ✅ **本质突破**：不再是“并行训练 + 参数共享”，而是“生成→嵌入”的单向知识蒸馏，实现能力间的**动态协同**。

---

### 相比现有方法的优势
| 方法 | 是否共享计算路径 | 是否双向监督 | 是否需切换 attention |
|------|------------------|--------------|---------------------|
| GritLM | ❌（切换 bidirectional） | ❌（平行目标） | ✅ |
| GEM | ⭕（部分共享） | ❌ | ❌ |
| **ULME (Ours)** | ✅（全程 causal） | ✅（生成→嵌入） | ❌ |

- **真正统一**：embedding 和 generation 共享同一前向传播路径；
- **能力互促**：生成过程直接提升嵌入质量，尤其在需要推理的任务上表现更强；
- **部署简洁**：单一 checkpoint 支持双功能，无需切换模式或加载多个模型。

---

## 2. 核心实验方法和设置

### 使用的数据集

| 类型 | 数据集 | 说明 |
|------|--------|------|
| **Embedding 训练** | E5 contrastive data (LLM2Vec format) | 包含 query-document 对及硬负例，用于 contrastive learning |
| **Generation 训练** | OpenThoughts-114k | 推理密集型 prompt-answer 对，含 chain-of-thought 示例 |
| **Embedding 评估** | MTEB (eng, v2), BRIGHT, AdvBench-IR | 多任务文本嵌入基准，覆盖检索、聚类、分类等 |
| **Generation 评估** | MMLU, GSM8K, MMLU-Pro, HumanEval, HumanEval+ | 测试知识、数学、多步推理、代码生成能力 |

---

### 实验设置和评估指标

- **模型家族**：Qwen2.5 和 Qwen3，共 **8 个规模**（0.5B ~ 8B）
- **槽位数量**：默认 $N=4$
- **Pooling 方式**：推理槽的目标 $r_{qa}$ 为 answer-side hidden states 的均值
- **温度**：$T = 0.05$
- **损失权重**：$\lambda_{\text{sem}} = \lambda_{\text{gen}} = 1.0$, $\lambda_{\text{pred}} = 0.3$
- **优化器**：AdamW，学习率 $1\times10^{-5}$，bf16 混合精度
- **评估方式**：
  - 所有结果来自 **同一个 ULME checkpoint**，无 head 微调、无模式切换；
  - embedding 使用 **semantic slot (slot 0)** pooling；
  - generation 直接解码输出。

---

### 基线方法对比
- **Embedding 基线**：
  - Echo, HyDE, InBedder, GIRCSE
  - LLM2Vec, LLM2VEC-GEN（最强开源 baseline）
- **Generation 基线**：
  - Backbone（原始 instruct 模型）
  - Gen-only FT：在同一 backbone 上仅用 generation 数据微调

> 所有 baseline 数值均引用自 LLM2VEC-GEN (BehnamGhader et al., 2026)，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 Qwen3-8B 为例）

| 指标 | ULME | 最强基线（LLM2VEC-GEN） | 提升 |
|------|------|--------------------------|------|
| **MTEB Avg** | **66.0** | 61.9 | **+4.1 pts** |
| **BRIGHT (nDCG@10)** | **21.4** | 20.2 | **+1.2 pts** |
| **AdvBench-IR (↓)** | **43.2** | 45.0 | **↓1.8 pts** |
| **MMLU** | 75.1 | 75.1 | ≈ |
| **GSM8K** | **89.5** | 78.6 | **+10.8 pts** |
| **MMLU-Pro** | **61.9** | 60.2 | **+1.7 pts** |
| **HumanEval** | **72.6** | 67.7 | **+4.9 pts** |

> ✅ ULME 在 embedding 性能上显著超越所有 baseline，同时在 generation 上持平甚至反超 generation-only fine-tune！

---

### 与其他方法的全面对比（MTEB 平均得分）

| 方法 | Qwen3-1.7B | Qwen3-4B | Qwen3-8B |
|------|------------|----------|----------|
| LLM2Vec | 54.8 | 56.8 | 56.8 |
| LLM2VEC-GEN | 58.6 | 60.6 | 61.9 |
| **ULME (Ours)** | **61.9** | **64.1** | **66.0** |
| **↑ vs LLM2Vec** | +7.1 | +7.3 | +9.2 |
| **↑ vs LLM2VEC-GEN** | +3.3 | +3.5 | +4.1 |

> 📈 提升广泛存在于各类任务中，尤其在 **Retrieval**, **Summarization**, **STS** 等依赖生成感知的任务上增益最大（+9 pts 以上）。

---

### 消融实验结果

#### （1）槽位数量 $N$ 的影响（Qwen2.5-7B）

| $N$ | MTEB-lite | MMLU | GSM8K |
|-----|-----------|------|-------|
| 0（last-token pooling） | 58.6 | 74.5 | 87.8 |
| 1 | 62.4 | 76.1 | 89.4 |
| 4 (**default**) | **63.8** | **76.5** | **90.0** |
| 8 | 63.6 | 76.3 | 89.9 |

> ✅ 插入槽位显著提升性能，验证了“避免 readout 冲突”的设计必要性。

---

#### （2）各损失项消融（Qwen2.5-7B）

| 设置 | MTEB-lite | MMLU | GSM8K |
|------|-----------|------|-------|
| Full model ($\mathcal{L}_{\text{sem}}+\mathcal{L}_{\text{gen}}+\mathcal{L}_{\text{pred}}$) | **63.8** | **76.5** | **90.0** |
| w/o $\mathcal{L}_{\text{sem}}$ | 31.2 | 76.6 | 90.1 |
| w/o $\mathcal{L}_{\text{gen}}$ | 51.0 | 61.8 | 74.2 |
| w/o $\mathcal{L}_{\text{pred}}$ | **60.2** | 76.4 | 89.9 |

> 🔍 关键发现：
> - 移除 $\mathcal{L}_{\text{pred}}$ 导致 MTEB 下降 **3.6 pts**，证明 predictive coupling 是性能增益的关键；
> - 即使没有显式的 embedding 损失，$\mathcal{L}_{\text{gen}}$ 仍能提供一定 embedding 信号（51.0 > 31.2），体现生成对嵌入的隐式支持。

---

#### （3）推理目标 $r_{qa}$ 设计对比

| Pooling 方式 | MTEB-lite | MMLU | GSM8K |
|-------------|-----------|------|-------|
| Mean over $H_a$ (**default**) | **63.8** | 76.5 | 90.0 |
| Last token of $a$ | 61.9 | 76.3 | 89.8 |
| Length-weighted mean | 62.7 | 76.4 | 89.9 |

> ✅ 均值池化最优，因其平滑了局部 next-token 预测噪声，提供更稳定的句子级目标。

---

## 4. 关键结论和发现

### 主要发现
1. **真正的统一 ≠ 参数共享**  
   仅有参数共享（如 GritLM）不足以实现能力互促；必须建立生成 → 嵌入的监督通路。

2. **Predictive Coupling 显著提升嵌入质量**  
   利用生成过程中的 answer-side 表示作为监督目标，使嵌入具备更强的**推理感知能力**，在 BRIGHT 等复杂检索任务上优势明显。

3. **单一 checkpoint 可兼顾双优**  
   ULME 在 embedding 和 generation 上均达到或超过专用模型水平，打破“兼顾即妥协”的认知。

4. **槽位机制有效解耦双重几何约束**  
   分离 semantic slot 与 reasoning slot，分别适应 retrieval 的全局可比性和 generation 的位置依赖性。

---

### 方法的局限性
1. **生成数据有限**  
   当前仅使用 OpenThoughts-114k，未引入大规模对话、安全对齐或格式控制数据。

2. **部分下游行为退化**  
   初步测试显示，chat-style 多轮对话、JSON 输出、function calling 等能力相比原生 Instruct 模型有所下降。

3. **尚未扩展至多模态**  
   虽然理论上可行，但目前仅验证于纯文本场景。

---

### 未来工作方向
1. **Scaling Up**
   - 使用更大规模、更多样化的 generation 数据；
   - 探索在 >8B 模型上的 scaling law 是否持续受益于 predictive coupling。

2. **Multimodal Extension**
   - 将 slot chain 应用于 vision-language 或 audio-language 模型；
   - 构建统一的跨模态 retrieval & generation 模型（如 visual RAG 场景）。

3. **恢复高级指令遵循能力**
   - 在 generation objective 中重新注入对话、格式、安全等监督信号；
   - 实现“全能力保留”的统一模型。

---

> 💡 **总结一句话**：  
> ULME 通过 **predictive coupling + learnable slots**，首次实现了 **generation actively shapes embedding** 的闭环，证明了统一 embedding 与 generation 不是折衷，而是一种能力增强机制。

</details>

---

### 14. [MOER: Mix-Policy Optimization under Expert Prefix Guidance with Selective Rollback](https://arxiv.org/abs/2610.30873)

**Authors**: Xuyang Zhong, Jiuding Yang, Yiwen Guo, Chen Liu  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.30873v1  

#### Abstract
Mix-policy optimization incorporates off-policy expert data into the reinforcement learning (RL) loop to transcend the capability boundary of on-policy RL. However, existing methods either provide static full-trace guidance that suppresses self-exploration, or adopt expert prefix guidance without co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MOER: Mix-Policy Optimization under Expert Prefix Guidance with Selective Rollback 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现有的 **mix-policy optimization** 方法在将专家数据（expert data）融入强化学习（RL）训练时面临两个核心挑战：

1. **静态全轨迹引导抑制自探索**：如 LUFFY 等方法直接注入完整专家轨迹，导致模型过度依赖模仿，抑制了自主推理和探索能力。
2. **前缀引导缺乏正确性保证**：如 Prefix-RFT 和 TRAPO 使用专家 CoT 前缀作为提示，让模型补全后续推理，但补全结果可能错误，污染 off-policy 监督信号，导致训练崩溃或退化为纯 on-policy RL。

此外，现有方法未正视“前缀引导”带来的 **off-policy 分布偏移** 和 **成功条件选择偏差**，仍将补全部分视为 on-policy 进行优化，引发训练不稳定。

---

### 提出了什么新方法或新思路

本文提出 **MOER**（Mix-policy Optimization under Expert prefix guidance with Selective Rollback），一个新型 mix-policy 框架，包含三大核心设计：

#### （1）Decaying Expert Prefix Guidance（衰减式专家前缀引导）
- 将专家 Chain-of-Thought（CoT）按比例截断作为输入提示的前缀。
- 随着训练进行，通过 **cosine annealing** 策略逐步缩短前缀长度，鼓励模型从早期模仿过渡到后期自主推理。
- 公式：`prefix_ratio = r_min + (r_max - r_min) * [0.5 + 0.5 * cos(π * step / max_steps)]`

#### （2）Selective Rollback Mechanism（选择性回滚机制）
- 若模型基于前缀生成的补全结果验证失败（答案错误），则“零成本”回滚至原始提示 + 完整专家轨迹作为 off-policy 轨迹。
- **保证每条 off-policy 轨迹的正确性**，避免错误补全污染训练信号。

#### （3）Harmonized Mix-Policy Objective（协调混合策略目标）
- **On-policy tokens**：使用带 **asymmetric clipping** 的 GRPO 损失（`ε_low=0.2, ε_high=0.24`），提升稳定性。
- **Off-policy tokens**（包括专家前缀和补全部分）：统一使用 **trust-region SFT loss**（TrSFT），形式为：
  $$
  \mathcal{L}_{\text{off}} = -\sum \frac{\log \pi_\theta(y_{i,t}|...)}{\max(\text{sg}(p), \gamma)}
  $$
  其中 `γ=0.1` 为信任区域系数。
- 明确认识到补全部分虽由当前策略采样，但由于前缀引导，其上下文状态是 off-policy 的，因此应作为 off-policy 数据处理。

---

### 相比现有方法的优势

| 特性 | MOER | LUFFY | ReLIFT | TRAPO | Prefix-RFT |
|------|------|-------|--------|--------|------------|
| 动态平衡模仿与探索 | ✅（衰减前缀） | ❌（固定全轨迹） | ⚠️（交替训练） | ✅（动态长度） | ✅（随机前缀） |
| 保证 off-policy 正确性 | ✅（Selective Rollback） | ✅（完整轨迹） | ✅（SFT 轨迹） | ❌ | ❌ |
| 处理分布偏移 | ✅（补全视为 off-policy） | ❌ | ❌ | ❌ | ❌ |
| 训练稳定性 | ✅✅✅ | ⚠️（后期不稳定） | ⚠️ | ❌（易崩溃） | ⚠️（退化） |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

- **训练集**：OpenR1-Math-46k-8192（来自 NuminaMath 1.5，专家轨迹由 DeepSeek-R1 生成）
- **数学推理评测集**：
  - AIME 2024/2025
  - AMC
  - MATH-500
  - Minerva
  - OlympiadBench
- **通用领域 OOD 评测集**：
  - ARC-c
  - GPQA-diamond
  - MMLU-Pro
- **代码生成评测**：LiveCodeBench-CPP

### 实验设置和评估指标

- **基础模型**：
  - 主要：Qwen2.5-Math-7B
  - 对比扩展：Qwen2.5-Math-1.5B, Qwen2.5-Instruct-7B, Qwen2.5-Instruct-14B
- **训练配置**：
  - Rollout 批大小：128
  - 更新批大小：64
  - Warm-up 步数：50
  - 最小前缀比 `r_min`：0.1
  - 信任区域系数 `γ`：0.1
  - Off-policy 损失权重 `α`：0.2（7B/1.5B），0.1（Instruct-7B）
- **评估指标**：
  - 数学任务：avg@32（AIME/AMC）、pass@1（其余）
  - 通用任务：pass@1
  - 代码任务：pass@1, 响应长度, 熵

### 基线方法对比

| 类别 | 基线方法 |
|------|--------|
| **基础方法** | SFT, RL (GRPO), RL w/ SFT Loss, SFT-then-RL |
| **Mix-policy 方法** | LUFFY, ReLIFT, CHORD, TRAPO, Prefix-RFT |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen2.5-Math-7B）

| 方法 | 数学平均分 | 通用平均分 |
|------|-----------|-----------|
| **MOER** | **49.1** | **59.0** |
| LUFFY | 48.4 | 57.3 |
| ReLIFT | 48.6 | 57.3 |
| SFT-then-RL | 48.2 | 44.8 |
| TRAPO | 2.0 | 1.0 |
| Prefix-RFT | 44.3 | 55.9 |

- **MOER 在数学和通用任务上均达到 SOTA**，分别领先最强基线 **+0.5** 和 **+1.6** 分。
- TRAPO 早期崩溃，Prefix-RFT 表现次优。

---

### 与其他模型规模和类别的对比

| 模型 | 方法 | 数学平均 | 通用平均 |
|------|------|---------|---------|
| Qwen2.5-Math-1.5B | MOER | 38.1（并列第一） | **40.6**（显著领先） |
| Qwen2.5-Instruct-7B | MOER | **42.3** | **63.8**（远超 LUFFY 的 21.5） |
| Qwen2.5-Instruct-14B | MOER | **45.9** | —— |

- MOER 在不同规模（1.5B–14B）和不同类型（数学专用 vs 通用指令）模型上均表现优越。
- 在通用指令模型上，MOER 显著避免了 LUFFY 的性能崩溃（如 ARC-c 上 LUFFY 几乎为 0）。

---

### 消融实验结果（Ablation Study）

在 Qwen2.5-Math-7B 上的消融结果（数学平均分）：

| 变体 | 数学平均 | 通用平均 | 说明 |
|------|---------|---------|------|
| LUFFY (Baseline) | 48.4 | 57.3 | —— |
| +HO（协调目标） | 48.6 | 58.3 | 提升稳定性和性能 |
| +HO+PG+COff（加衰减前缀） | 46.2 | 58.3 | 无回滚导致性能下降 |
| +HO+PG+SR（加回滚） | 48.9 | 58.7 | 回滚机制恢复性能 |
| **MOER（完整版）** | **49.1** | **59.0** | 补全视为 off-policy 进一步提升 |

- **Selective Rollback 是关键**：缺少它会导致训练崩溃（见 Figure 5）。
- **补全作为 off-policy 处理必要**：否则后期响应长度爆炸，训练不稳定。
- **协调目标（HO）有效缓解熵崩溃和长度增长**。

---

## 4. 关键结论和发现

### 论文的主要发现

1. **动态前缀 + 回滚机制可实现模仿与探索的平衡**：衰减前缀促进自探索，选择性回滚保障监督质量。
2. **补全 token 应视为 off-policy**：由于前缀诱导的分布偏移和成功选择偏差，必须使用 off-policy 损失（如 TrSFT）优化。
3. **MOER 训练更稳定、泛化更强**：
   - 在数学、通用、代码任务上均优于基线。
   - 响应更短（MOER: 668 vs LUFFY: 1903），熵更健康。
   - 不同模型规模和类别下一致有效。
4. **MOER 真正提升了底层推理能力**：Pass@k 实验显示，随着采样次数增加，MOER 性能持续上升，超越 SFT，表明其增强了多样性而非过拟合。

---

### 方法的局限性

- **依赖高质量专家轨迹**：若专家数据本身有误，回滚机制仍会引入错误。
- **前缀调度策略固定**：当前使用预设的 cosine 衰减，未自适应调整。
- **计算开销**：虽然回滚“零成本”，但仍需额外验证逻辑。

---

### 未来工作方向

- 探索 **自适应前缀长度控制**，根据问题难度或模型置信度动态调整。
- 将 MOER 框架应用于 **多模态推理** 或 **规划任务**。
- 结合 **meta-learning** 自动学习最优的混合策略权重和调度参数。
- 研究如何在 **无奖励信号** 场景下应用类似机制（如仅依赖一致性判断）。

---

> **代码开源**：https://github.com/CityU-MLO/MOER

</details>

---

### 15. [Building Presentation Intelligence for High-Quality Slide Generation](https://arxiv.org/abs/2610.30888)

**Authors**: Yuhang Xie, Cheng Li, Chenliang Li, Mingyu Xu, Mingwei Sun, Suhang Zheng, Ming Yan, Xiaoxiao Xu, Lin Qu  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.30888v1  

#### Abstract
PowerPoint presentations (PPTs) are a dominant medium for communication across diverse domains, yet high-quality PPT generation still depends heavily on labor-intensive human effort. Recent advances in large language models (LLMs) for agentic tasks have created both new opportunities and challenges ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Building Presentation Intelligence for High-Quality Slide Generation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前高质量 PPT 生成严重依赖人工，尽管 LLM 在 agentic 任务上取得进展，但自动幻灯片生成仍面临以下挑战：
- **内容质量与视觉美学需联合优化**：学术类 PPT 强调严谨性，广告类则注重吸引力，需求异质性强。
- **训练数据缺乏系统性构建**：现有方法多依赖基础模型的固有能力，缺少针对 PPT 生成任务的高质量、可控难度的数据合成机制。
- **监督信号弱且不可验证**：SFT 难以捕捉视觉层面的美学特征；RL 中的 reward 多基于 LLM 判断，存在主观性和不稳定性。

### 提出的新方法与创新思路
本文提出一个统一的三阶段训练框架，实现“演示智能”（Presentation Intelligence）：

#### （1）**Cross-Validated Data Synthesis**  
构建跨阶段验证的数据合成流水线，确保任务、评分标准（rubric）与源材料之间的非泄露性（non-disclosure）、证据锚定（evidential grounding）和难度控制。通过五阶段流程（source profiling → task construction → rubric generation → principle audit → rubric tightening），生成 4,802 个实例，覆盖 25 个主领域和 162 个子领域。

#### （2）**Guided Supervised Fine-Tuning (SFT)**  
引入教师增强框架，提升 teacher model 的指导能力：
- **Self-Guidance**：从失败案例中归纳原则并反馈至后续蒸馏。
- **Strong-to-Weak Guidance**：利用更强模型（如 Qwen-3.8-Max 和 GLM-5.2）的知识引导较弱模型输出更优轨迹。
- **Mixed-Teacher Strategy + Outcome-Driven Filtering**：融合多个 teacher 的优势，并通过规则过滤掉渲染失败或布局溢出的低质量样本。

#### （3）**Verifiable RL Training with GRPO**  
设计可验证的美学奖励体系，在 SFT 基础上进一步优化视觉属性：
- 定义五个**可量化、可解释、基于渲染结果的美学维度**：
  - Aspect-ratio compliance
  - Whitespace adequacy
  - Element-collision avoidance
  - Visual balance
  - Font readability
- 使用 **GRPO**（Group-Reward Policy Optimization）框架进行强化学习，结合 group-relative baseline 和 reward-decoupled normalization，提高训练稳定性。

### 相比现有方法的优势
| 方面 | 本工作 | 现有方法（如 SlideGen, PPTAgent, DeepPresenter） |
|------|--------|---------------------------------------------|
| 数据合成 | 显式控制难度与有效性，交叉验证防止泄露 | 缺乏系统性描述，常为启发式构造 |
| 教师能力 | 主动增强，融合 prior knowledge | 直接使用原始模型输出 |
| 视觉监督 | 可验证的 rule-based reward，超越文本模仿 | 依赖 LLM-based reward 或无专门优化 |
| 训练范式 | 统一 SFT + RL 流程，端到端优化内容与美学 | 多停留在 agentic 框架设计，未深入训练策略 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **PresentBench**：细粒度评分基准，包含 rubric-based 评估协议，用于衡量内容质量和视觉设计。
- **GDPVal (PPT subset)**：真实世界经济价值任务评估集中的 PPT 子集，用作 out-of-distribution 泛化测试。

> 所有训练数据均为**合成生成**，共 4,802 个样本，涵盖 25 个 top-level domains（见附录 Table 6）。

### 实验设置
- **Backbone Models**：
  - `Qwen-3.6-27B`（dense）
  - `Qwen-3.6-35B-A3B`（MoE 架构）
- **Generation Paradigm**：采用 **Single-HTML** 模式（每页幻灯片生成独立 HTML 文件），经实证分析其在视觉质量和鲁棒性上优于 Python-based 和 Multi-page 方法。
- **Evaluation Protocol**：
  - 输出转换为 PDF 后由 **Gemini-3-Flash-Preview** 按照 rubric 分项打分。
  - 采用 per-rubric 评估方式，避免长上下文干扰。

### 评估指标
| 类型 | 指标 |
|------|------|
| **整体性能** | PresentBench 总分、GDPVal-PPT 子集得分 |
| **内容维度** | Content Quality, Completeness, Correctness, Consistency |
| **视觉维度（RL专用）** | Five verifiable aesthetic metrics (Asp., Ws., Ele., Bal., Font) |
| **人类偏好评估** | Blind human evaluation（视觉质量 & 整体质量胜率） |

### 基线方法对比
- **No Train**：直接推理 backbone 模型
- **GLM-5.2**：专为 PPT 优化的强 baseline
- **Qwen-3.8-Max**：通用能力强的 teacher model
- 不同 SFT 设置（single/mixed teacher, 是否过滤）
- 不同 RL 设置（task-hard vs aesthetic-hard dataset, 是否加 KL 正则）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & Table 3）

#### ✅ SFT 阶段显著提升性能
| Model | PresentBench (↑) | Improvement | GDPVal-PPT (↑) | Generalization Gain |
|-------|-------------------|-------------|----------------|---------------------|
| Qwen-3.6-27B (SFT) | **77.90** | +7.45 pts | **79.36** | +4.87 pts |
| Qwen-3.6-35B-A3B (SFT) | **74.28** | +9.35 pts | **78.17** | +4.16 pts |

> 📌 注：仅用 4,000+ 合成样本训练，`Qwen-3.6-27B` 已超越更大规模的 `GLM-5.2`（75.04）

#### ✅ RL 进一步优化视觉美学
| Model | Total Aesthetic Score (PresentBench) | Improvement |
|-------|--------------------------------------|-----------|
| Qwen-3.6-27B (RL) | 4.16 → **4.20** (task-hard) | +0.04~0.06 |
| Qwen-3.6-35B-A3B (RL) | 4.16 → **4.36** (task-hard) | +0.20 |

> 在五个美学维度上均有提升，尤其在 MoE 模型上效果更明显。

#### ✅ 人类评估验证视觉改进
| Backbone | Visual Quality Win Rate | Overall Quality Win Rate |
|----------|-------------------------|----------------------------|
| Dense (27B) | **77.1%** | 66.9% |
| MoE (35B-A3B) | **82.4%** | 71.8% |

表明 RL 后模型确实在人类感知层面更具吸引力。

---

### 消融实验结果

#### 🔍 指导策略的影响（Table 2）
- 加入 **self-guidance** 和 **strong-model augmentation** 显著提升 teacher 轨迹质量。
- `qwen+glm_max (filter)` 表现最佳，说明混合教师 + 过滤噪声轨迹是有效组合。

#### 🔍 教师混合 vs 单一教师
- 混合 teacher（Qwen + GLM）始终优于任一单一 teacher。
- 例如在 Qwen-3.6-35B-A3B 上，mixed teacher 较 best single 提升 **+3.94 pts**。

#### 🔍 轨迹过滤的作用
- 引入 outcome-driven filtering 可去除渲染失败、元素重叠等问题轨迹。
- 多数情况下带来增益（如 73.49 → 74.28），但过度过滤可能损失多样性。

#### 🔍 RL 数据选择的影响（Table 3）
- **task-hard dataset > aesthetic-hard dataset**
  - 因前者覆盖更广的内容与布局挑战，提供更强的学习信号。
- **移除 KL 正则更好**
  - 因 reward 已完全可验证，无需保守约束，允许更大探索空间。

---

## 4. 关键结论和发现

### 主要发现
1. **高质量 PPT 生成需要系统化的训练流程**，不能仅靠 agentic 框架堆叠。
2. **Cross-validated data synthesis 是构建可靠训练集的关键**，能显式控制难度与有效性。
3. **Guided SFT 显著提升学生模型表现**，尤其是通过 mixed-teacher + filtering 策略。
4. **Verifiable aesthetic rewards 可有效驱动 RL 优化视觉质量**，弥补 SFT 对视觉监督的不足。
5. **MoE 架构在 RL 阶段表现出更强潜力**，因其更高的输出多样性与专家路由机制。

### 方法的局限性
- 当前 reward 仅聚焦于**视觉美学**，尚未对**内容质量**进行 RL 层面的直接优化。
- 所有训练数据为**合成生成**，虽经交叉验证，但仍可能存在分布偏差。
- RL 仍采用 trajectory-level reward assignment，对于长序列任务来说信用分配较粗粒度。

### 未来工作方向（Future Work）
1. **Content Reward Modeling**：构建可验证的内容质量 reward model，支持 factual faithfulness、material completeness 等维度的 RL 优化。
2. **Hierarchical Credit Assignment**：发展 step-level 或 hierarchical RL 算法，提升长程任务的优化效率。
3. **Personalized Presentation Generation**：引入用户偏好建模，实现个性化风格控制（preference-conditioned FT/RL）。
4. **Integration with On-Policy Distillation**：将 RL 与 on-policy distillation 结合，形成闭环迭代优化。

---

> 💡 **总体评价**：该论文首次系统地将数据合成、教师增强、SFT 优化与 verifiable RL 整合进 PPT 自动生成流程，不仅大幅提升了生成质量，也为“演示智能”这一复杂任务提供了可复现、可扩展的训练范式，具有较强的工程与研究参考价值。

</details>

---

### 16. [Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training](https://arxiv.org/abs/2610.30792)

**Authors**: Zezhou Wang, Hanzhang Zhou, Panrong Tong, Quyu Kong, Pengxiang Li, Steven Hoi, Zhuzhong Qian, Yue Wang  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.30792v1  

#### Abstract
Scaling GUI reinforcement learning requires environments that preserve real-app behavior. Yet coupling behavior inference with coding can turn partial traces into runnable mocks with incorrect state transitions. We introduce \textbf{Trace2World}, which separates global behavior reconstruction from i...

---

### 17. [GIFT: Gradient Informed Filtering for Distillation Augmented Reasoning RL](https://arxiv.org/abs/2610.30796)

**Authors**: Shijie Li, Yilin Gao, Tieyuan Chen, Yun Luo, Weiyao Lin, Hang Yu  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.30796v1  

#### Abstract
Training large language models for complex reasoning tasks relies on two complementary signals: sparse, outcome-grounded reinforcement learning (RL) for correctness, and dense, token-level distillation for guidance. However, complementarity in supervision does not guarantee compatibility in optimiza...

---

### 18. [D-Skill: Distilling Teacher Knowledge through Agent Skills](https://arxiv.org/abs/2610.28914)

**Authors**: Shuo Yuan, Dacheng Li, Joseph E. Gonzalez  
**Category**: cs.AI  
**Published**: 2026-10-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.28914v1  

#### Abstract
Frontier language-model agents excel at complex tasks, but high inference costs make transferring their capabilities to smaller models essential. Traditional knowledge distillation achieves this through parameter updates, requiring access to the student's weights and additional training compute whil...

---

### 19. [Self-Assessment Reinforcement Learning for Search-State Awareness in Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2610.30435)

**Authors**: Taewhoo Lee, Minju Song, Jungwoo Park, Chanwoong Yoon, Junha Jung, Hyeon Hwang, Sanghoon Lee, Minbyul Jeong, Jaewoo Kang  
**Category**: cs.CL  
**Published**: 2026-10-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.30435v1  

#### Abstract
Recent advances in search agents have enabled large language models to tackle complex tasks by retrieving and synthesizing external information. However, existing search agents often lack search-state awareness, i.e., the ability to recognize what information has been obtained and what remains missi...

---

### 20. [Synthetic-to-Real Code-Switched ASR: A Spontaneous Nepali-English Podcast Benchmark](https://arxiv.org/abs/2610.30860)

**Authors**: Firoj Paudel  
**Category**: cs.CL  
**Published**: 2026-10-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.30860v1  

#### Abstract
Nepali-English code-switched conversational speech presents severe challenges for automatic speech recognition (ASR), chiefly through script confusion (inappropriate mixing of Devanagari and Latin orthography) and elevated error rates at language switch points. Supervised fine-tuning (SFT) can adapt...

---

### 21. [ReflectPO: Reflection-Guided Policy Optimization for Long-Horizon LLM Agents](https://arxiv.org/abs/2610.30933)

**Authors**: Rumei Li, Xunliang Cai  
**Category**: cs.CL  
**Published**: 2026-10-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.30933v1  

#### Abstract
With external tools, large language models (LLMs) can tackle complex tasks through sustained environmental interaction. LLM-based agents are therefore increasingly used to tackle complex long-horizon tasks in real-world environments. However, these tasks involve sequential, interdependent decisions ...

---

### 22. [IQuest-Q1: Advancing Foundation Capabilities for Agentic CLI Systems](https://arxiv.org/abs/2610.30949)

**Authors**: A. Jian Yang, An Huang, Bo Dan, Chen Mu, Chengran Yang, Chuan Shi, Feng Chang, Hongxin Li, Hongyuan Tao, Jinxing Zhang, Jujie He, Kun Wang, Lin Jing, Ming Yang, Qicheng Hu, Renyuan Li, Shark Liu, Shawn Guo, Shi Qing, Shi Zhong, Tianyu Zheng, Wei Liu, Wei Zhang, Wenrui Cai, Wenxiao Chen, Xuxing Chen, Yan Xing, Yi Xing, Yifan Luo, Yiran Xing, Yizhi Li, Yongsheng Kang, Yucheng Ding, Yue Song, Yun Qing, Yunqi Liu, Zelong Huang, Zhengmao Ye, Zhentao Fan, Zhiteng Li, Ziling Huang, Bihong Tang, Bingli Wang, Ce Jun, Changhao Chen, Chen Hu, Chen Xiao, Chi Liu, Chunjiang Ge, Danyang Li, Deyang Kong, Dian Qi, Ding Tan, Evan Yang, Fanglin Xu, Fei Bai, Gang Liu, Guochang Li, Hao Tang, Haoran Feng, Haowen Wang, Huahao Hu, Huanjin Yao, Huatong Song, Jia Deng, Jiajun Wu, Jialiang Cheng, Jiayi Liang, Jiayi Wu, Jie Deng, Jiejun Tan, Jing Yang, Jingda Xu, Junhang Cheng, Junxiao Liu, Kai Lv, Kaixiang Pan, Kecheng Lu, Kun Zeng, Kun Zheng, Kunbo Ding, Liang Qiu, Lingyi Sun, Luan Zhang, Mengxiao Lv, Ning Dong, Pei Wang, Peng Zhang, Qingsong Cai, Qixiang Yin, Qiyao Wang, Renbiao Liu, Renfei Dang, Ruihang Li, Ruirui Huang, Shanhui Yin, Shen Jian, Shengjie Fang, Shikai Li, Shinan Zhang, Shiqi Chen, Shuai Liu, Shuangqi Li, Shuo Tang, Shuqing Deng, Sijin Hao, Siwei Wu, Siyue Wang, Tao Hai, Tao Ren, TingYang Chen, Wei Shao, Weicheng Gu, Weihan Peng, Wenlong Miao, Xavier Yu, Xiangyuan Ji, Xiaocheng Li, Xiaodong Huang, Xiaohan Ge, Xiaoxiang Shi, Xiaoyang Zhang, Xiaoyu Yang, Xidong Wang, Xin Li, Xinyu Tang, Xiyuan Yang, Xudong Ma, Ya Tong, Yachao Wang, Yang Tian, Yaxin Du, Yeli Ruan, Yi Wang, Yi Yu, Yibin Feng, Yibo Tan, Yihang Wang, Yiming Ju, Yinda Chen, Yinda Chen, Yingjie Du, Yiwei Liu, Yizhi Zhou, Yuan Wei, Yuli Chen, Yuqi Zhu, Yuwen Li, Yuxi Dong, Yuxuan Zhang, Yuyang Song, Zhange Zhang, Zhaohan Zhang, Zhaoyi Li, Zhen Zhang, Zhengbo Zhang, Zhihang Hu, Zhiwei Yu, Zhixin Xie, Zhiyu Chen, Zhizhao Luo, Zhongyuan Wu, Zhou Fang, Zhu Lin, Zhuocheng Gong, Zixi Zhao, Ziyang Zeng, Ziyi Wang, Ziyu Zhao, Ran Tao, Mingjie Tang, Chuan Hao, Peihao Wu, Jianzhou Wang, Nan Jiang, Yuan Zhi, Bryan Dai  
**Category**: cs.CL  
**Published**: 2026-10-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.30949v1  

#### Abstract
We present IQuest-Q1, a sparse Mixture-of-Experts (MoE) agentic foundation model with 320B total parameters (15B activated). IQuest-Q1 is developed through a three-stage process: (1) stable and efficient pre-training, combining Stable SplitNorm, Muon, and muP with a two-stage curriculum whose second...

---

### 23. [SyRA-CS: Synthetic-to-Real Adaptation for Code-Switched ASR with Meta-Learning](https://arxiv.org/abs/2610.31415)

**Authors**: Fadel Thior, Ahmer Tabassum, Haania Siddiqui, Hanan Aldarmaki, Thamar Solorio  
**Category**: cs.CL  
**Published**: 2026-10-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.31415v1  

#### Abstract
Code-switched (CS) automatic speech recognition (ASR) remains difficult because real CS speech is scarce and synthetic speech does not fully match it. We investigate whether synthetic CS speech is more useful for learning an adaptation-friendly initialization than for ordinary continual fine-tuning....

---

### 24. [Benchmarking Model Compression Techniques Across Heterogeneous Edge Platforms: A Predictive Maintenance Use Case for Rotating Machinery](https://arxiv.org/abs/2610.30228)

**Authors**: Matheus Calheira Guimar\~aes de Oliveira, Marco Aur\'elio da Silva Cruz, Regis Cardoso, Lucas Cambuim, Rafael Macieira, Arthur Paim, Helaine Neves, Alex Santos, Lilian Guarieiro, Jean Schmith, Prashant Kumar, Erick G. Sperandio Nascimento  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.30228v1  

#### Abstract
To enable Industry 5.0, where autonomous systems are widely adopted and integrated with people to promote sustainable development in industries, it is imperative to adapt complex deep learning models to operate efficiently on resource-constrained hardware for edge inference. However, adapting these ...

---

### 25. [CTRL-FM: Inference-Time Structural Control for Flow-Based Virtual Sensing under Long-Horizon Node Outages](https://arxiv.org/abs/2610.30362)

**Authors**: Boshra Khalili, Andrew Smyth  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.30362v1  

#### Abstract
Long-horizon node outages are harder than conventional missing-value imputation because complete sensor trajectories remain unobserved for extended periods, while the usefulness of spatial and temporal dependencies varies across outage locations and operating regimes. Existing graph, diffusion, and ...

---

### 26. [Doubly Robust Distributionally Robust Offline Contextual Pricing](https://arxiv.org/abs/2610.30957)

**Authors**: Min Xu, Xinyi Yin, Yunfan Zhang, Yuxuan Han, Houcai Shen, Caihua Chen  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.30957v1  

#### Abstract
Offline contextual pricing often relies on logged observational data, but faces challenges from distributional shifts between training and deployment environments. Distributionally robust optimization (DRO) provides a principled approach to off-policy evaluation and learning (OPE/L). However, existi...

---

### 27. [QuaNAS: Uncertainty-Aware Reuse of Earlier Searches for Query-Efficient Neural Architecture Search](https://arxiv.org/abs/2610.31251)

**Authors**: Brama Seta Putra Satria, Dwi Sunaryono, Misbakhul Munir Irfan Subakti, Irzal Ahmad Sabilla, Laras Suciningtyas  
**Category**: cs.LG  
**Published**: 2026-10-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2610.31251v1  

#### Abstract
Neural architecture search (NAS) on a new dataset often affords only tens of short training runs, so what the search knows before its first evaluation decides its efficiency. Yet most NAS methods start from scratch, guided at best by zero-cost proxies that are unreliable across tasks. QuaNAS treats ...

---

### 28. [RubricOpt: Self-Revision of Semantic Rubrics for Preference Optimization](https://arxiv.org/abs/2610.29491)

**Authors**: Wei Chen, Yubing Wu, Shigui Li, Delu Zeng, Jingfeng Liu, Junmei Yang, Zhiheng Zhou, Bo Li, Min Chen, Zhou Wang  
**Category**: cs.AI  
**Published**: 2026-10-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.29491v1  

#### Abstract
A model's failures can reveal gaps in the behavioral criteria guiding its training. We introduce RubricOpt, a framework that lets a policy revise these criteria within preference optimization. The policy diagnoses its own rubric-free outputs, retains requirements already covered by the rubric, and e...

---

### 29. [GuideEvolve: Self-Evolving Agent Framework with Expert Guidance and 3Ts Mechanisms](https://arxiv.org/abs/2610.30017)

**Authors**: Meihong Jia, Yasemin Timar, Jonathan Taws  
**Category**: cs.AI  
**Published**: 2026-10-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.30017v1  

#### Abstract
This study introduces GuideEvolve, a self-evolving agent framework incorporating expert guidance and three autonomous regulatory mechanisms, collectively termed the 3Ts: Think Fast and Slow, Throwback, and Termination by choice. The framework addresses critical limitations in extant self-evolving ag...

---

### 30. [HGM-Dual: Lineage-Aware Guided Agent Harness Optimization for Long-Horizon Planning](https://arxiv.org/abs/2610.30042)

**Authors**: Sudipta Paul, Pengrui Quan, Aounon Kumar, Nicholas Tzou, Vivek Kulkarni, Srinivas Chappidi  
**Category**: cs.AI  
**Published**: 2026-10-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2610.30042v1  

#### Abstract
Harness optimization, which improves an agent's prompts, tools, and workflows, is an increasingly effective way to enhance agents built on frozen or closed-source LLMs. Leading approaches such as the Darwin--Godel Machine and the Huxley--Godel Machine (HGM) rely on recursive self-improvement and wer...

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

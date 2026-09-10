# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-10 10:06:29 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FastE: Readout-Triggered Token Compression for LLM Embedding Inference](https://arxiv.org/abs/2609.08407)

**Authors**: Jinsong Shu, Jinyong Wen, Baokun Wang, Zhongle Xie, Lidan Shou, Weiqiang Wang, Gang Chen  
**Category**: cs.AI  
**Published**: 2026-09-10  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2609.08407v1  

#### Abstract
In this study, we identify depth-dependent prefix redundancy in final-readout LLM embedding models, notably across representative backbones including Qwen3-Embedding and Qwen3-VL-Embedding. We find that removing prefix states is substantially more damaging in shallow layers than at greater depth, sh...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FastE: Readout-Triggered Token Compression for LLM Embedding Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在基于 **final-readout LLM embedding 模型** 中，输入序列被编码为一个前缀（prefix）加一个读出 token（readout token），其最终隐藏状态作为整个序列的 embedding。对于长序列，这种机制需要在整个 Transformer 堆栈中传播所有 prefix states 和 readout state，导致巨大的计算开销（尤其是 FLOPs 和推理延迟）。

传统方法如 **token pruning** 或 **KV cache compression** 多采用固定层压缩策略，或依赖模态特定信号（如视觉结构），难以自适应地判断“何时”以及“保留哪些”prefix states 最有效。

---

### ✅ 提出了什么新方法或新思路

作者提出 **FastE** ——一种无需训练、即插即用（plug-and-play）的 token 压缩方法，核心思想是利用 **depth-dependent prefix redundancy** 现象：

> 随着网络深度增加，prefix states 变得越来越可压缩；浅层移除严重影响性能，深层则容忍度更高。

FastE 将压缩决策分解为两个部分：

1. **When to compress（何时开始压缩）**  
   使用 **batch-mean readout-prefix alignment**（readout state 与 prefix states 平均表示之间的余弦相似度）作为轻量级在线启发式信号。当该值超过预设阈值时触发压缩。

2. **Which states to retain（保留哪些状态）**  
   在触发层，根据当前层中 **readout position 对各 prefix state 的 attention score** 进行排序，保留得分最高的 top-K states，并保持因果顺序和位置编码不变。

此外，引入 warm-up 阶段（前几层不压缩）以避免早期信息丢失。

---

### ✅ 相比现有方法的优势

| 特性 | FastE | ToMe / FastV / RTPrune |
|------|-------|------------------------|
| 是否需训练 | ❌ 否（training-free） | ❌ 否（但参数固定） |
| 触发机制 | 动态（基于 readout alignment） | 固定层或固定调度 |
| 排序依据 | readout-guided attention | 全局注意力平均 / 聚类匹配 |
| 跨任务泛化能力 | ✅ 强（共享阈值即可迁移） | ⚠️ 弱（需调参适配） |
| 支持多模态 | ✅ 初步验证成功（Qwen3-VL） | ⚠️ 多依赖视觉先验 |
| 效率提升显著性 | ✅ 显著降低 FLOPs & 提升 E2E 速度 | ✅ 有收益但质量损失更大 |

> FastE 的关键优势在于：**通过 readout state 自主决定“何时”和“保留谁”，实现了更智能、更高效的压缩路径。**

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### 文本嵌入任务（Text Embedding）
- **NarrativeQA**：长文档问答检索
- **IMDb**：情感分类（Accuracy）
- **ArXiv P2P Clustering**：学术论文聚类（V-measure）
- **Core17**：新闻摘要检索
- **Supply Chain Disclosure**：供应链披露文本检索

#### 工业级基准
- 匿名 **39-task 工业预测任务集合**，涵盖用户行为、支付、信用、风控、反欺诈等场景（AUC 评估）

#### 跨模态任务（Cross-modal Retrieval）
- **MSCOCO (T→I)**：文本到图像检索
- **DocVQA (I+T→T)**：图文混合问答
- **NIGHTS**：图像相似性检索

---

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|--------|------|
| 主干模型 | Qwen3-Embedding-0.6B / 4B, Qwen3-VL-Embedding-2B, E5-Mistral-7B-Instruct |
| Warm-up 层 | `lw = 8`（默认） |
| 对齐阈值 θ | `θ = 0.60`（文本任务）、`0.70`（跨模态）、`0.40`（E5-Mistral） |
| 最大移除比例 | `r_max ∈ {0.3, 0.5, 0.7}` |
| 批大小 | 通常为 4（长度排序批处理） |
| 硬件 | A100 / L20 GPUs |

#### 评估指标
- **Quality Metrics**:
  - nDCG@10, Recall@10, Accuracy, V-measure, Macro AUC
  - **Retention (%)**: 压缩后指标相对于 Full Forward 的百分比
- **Efficiency Metrics**:
  - **FLOPs reduction (%)**
  - **GPU-forward speedup (×)**
  - **End-to-End (E2E) speedup (×)**

---

### 🔁 基线方法对比

| 方法 | 类型 | 简介 |
|------|------|------|
| **ToMe (ICLR'23)** | Token merging | 基于 key 相似性合并相邻 token |
| **FastV (ECCV'24)** | Visual token pruning | 固定层一次性剪枝，关注视觉 sink |
| **RTPrune (ICML'26)** | 结构化剪枝 | 基于 l2 范数选择主导 token 并融合 |
| **OptScale (ICML'26)** | 缩放补偿合并 | 在 ToMe 基础上加入 representation scaling |
| **Block-last / Random** | 控制组 | 按块保留最后一个 / 随机选择 |

所有基线均适配至相同 prefix-state budget 下进行公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 Qwen3-Embedding-0.6B + NarrativeQA 为例）

| 方法 | 移除比例 | nDCG@10 | Retention (%) | FLOPs ↓ | GPU-fwd ↑ | E2E ↑ |
|------|----------|---------|---------------|---------|------------|--------|
| Full Forward | 0% | 0.45395 | 100.00% | 0% | 1.000× | 1.000× |
| **FastE (Ours)** | **50%** | **0.45181** | **99.53%** | **40.11%** | **1.540×** | **1.363×** |
| FastE (Ours) | 70% | 0.43179 | 95.12% | 51.19% | 1.934× | 1.533× |

> 在仅损失 **0.47% nDCG@10** 的前提下，实现 **40.11% FLOPs 减少** 和 **1.363× 端到端加速**。

---

### 📈 与其他方法对比（70% 移除比例下）

| 方法 | nDCG@10 (↓) | Retention (%) | FLOPs ↓ |
|------|-------------|----------------|---------|
| ToMe | 0.3711 | 85.45% | ~40% |
| FastV | 0.3437 | 84.64% | ~40% |
| RTPrune | 0.1690 | 75.34% | ~40% |
| OptScale | 0.2912 | 79.02% | ~40% |
| **FastE (Ours)** | **0.4318** | **95.85%** | **51.19%** |

✅ FastE 在同等预算下显著优于所有基线，在质量保留和效率之间取得最优平衡。

---

### 🔍 消融实验结果

#### A. **Start Depth 消融（Table 6）**

| 触发方式 | nDCG@10 | Retention | FLOPs ↓ | Speedup |
|----------|--------|-----------|--------|--------|
| Fixed L9 | 0.40351 | 88.89% | 55.55% | 2.082× |
| Fixed L12 | 0.42898 | 94.50% | 46.78% | 1.692× |
| **Dynamic g (FastE)** | **0.43179** | **95.12%** | **51.19%** | **1.934×** |

➡️ 动态触发（dynamic g）优于任何固定层，说明“何时压缩”必须自适应。

#### B. **Ranking Strategy 消融（Table 7）**

| 排序策略 | nDCG@10 |
|--------|--------|
| Readout attn. (FastE) | **0.43179** |
| Mean attention | 0.42464 |
| Random | 0.1426 |
| Block-last (positional) | 0.05006 |

➡️ **readout-guided attention ranking** 是关键，随机或位置规则严重损害性能。

#### C. **超参数敏感性分析（Appendix）**

- 阈值 `θ` 越高 → 压缩越晚 → 质量越高但节省越少
- 默认 `(lw=8, θ=0.6)` 在验证集上达到最佳 trade-off（>95% retention + >50% FLOPs↓）

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **存在 depth-dependent prefix redundancy**  
   prefix states 在深层比浅层更具冗余性，支持延迟压缩策略。

2. **readout-prefix alignment 是有效的压缩触发信号**  
   它随深度单调上升，能可靠指示“何时可以安全压缩”。

3. **readout-guided attention 是最优的状态选择机制**  
   表明 readout token 自身最清楚哪些上下文最重要。

4. **FastE 具备强泛化能力**  
   - 跨不同规模（0.6B vs 4B）
   - 跨架构（Qwen vs Mistral）
   - 跨模态（text-only → vision-language）
   - 跨任务（classification → retrieval → industrial prediction）

5. **工业部署价值明确**  
   在 39-task 工业套件中，70% 移除仍保持 **macro AUC 从 0.8239 → 0.8180**，几乎无损。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| 仅适用于 final-readout 架构 | 不适用于 final-layer pooling 类模型（如平均池化） |
| 控制器带来额外开销 | alignment 计算和 Top-K 影响小批量性能 |
| 阈值需微调（per backbone） | 不同模型（如 E5-Mistral）需本地校准 θ |
| 多阶段压缩未探索 | 当前为 one-shot 压缩，可能错过渐进优化机会 |
| 视频等复杂模态尚未测试 | 当前验证集中不含视频或多轮交互数据 |

---

### 🔮 未来工作方向

1. **扩展至更多 backbone 架构**  
   如 Llama, Phi, Gemma 等通用 embedding 模型。

2. **动态调整 removal ratio**  
   根据输入长度或任务难度自适应设置 `r_max`。

3. **结合 KV cache 优化**  
   将 prefix-state 压缩与 KV cache 压缩联合设计，进一步提升生成式检索效率。

4. **应用于 retrieval-augmented generation (RAG)**  
   加速 long-context RAG 中的 document encoder 推理。

5. **理论解释 alignment 上升机制**  
   深入研究为何 readout 与 prefix 表示逐渐对齐，是否反映语义收敛过程。

---

## 总结

📌 **FastE 是一项高效、实用、泛化性强的 LLM embedding 推理加速方案**。它首次系统揭示了 **depth-dependent prefix redundancy** 现象，并据此设计了一个由 **readout state 自主驱动的压缩机制**，实现了高质量下的大幅效率提升。实验充分证明其在多个维度上的优越性，具备广泛应用于工业级检索、聚类、推荐系统的潜力。

</details>

---

### 2. [X-CoSD: Communication-Efficient Cross-Vocabulary Collaborative Speculative Decoding](https://arxiv.org/abs/2609.09166)

**Authors**: Jaeduk Lee, Wan Choi  
**Category**: cs.CL  
**Published**: 2026-09-10  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.09166v1  

#### Abstract
This paper investigates collaborative speculative decoding (CoSD), a distributed large language model (LLM) inference framework in which an on-device small language model (SLM) drafts candidate tokens and a server LLM verifies them. Existing CoSD methods assume a shared vocabulary between the SLM an...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：X-CoSD: Communication-Efficient Cross-Vocabulary Collaborative Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Collaborative Speculative Decoding (CoSD)** 方法依赖于 **on-device SLM** 和 **server LLM** 使用相同的词汇表（shared vocabulary），这在实际应用中过于理想化。不同厂商或模型家族的 SLM 和 LLM 往往使用不同的 tokenizer，导致词汇不一致（heterogeneous vocabularies）。此外，传统 CoSD 在 token 被拒绝时需要传输完整的 LLM 分布，造成巨大的通信开销，尤其是在无线网络环境下。

因此，本文旨在解决以下三个关键挑战：
- 支持异构词汇表（heterogeneous vocabularies）
- 实现无损解码（lossless decoding）
- 显著降低通信负载（communication overhead）

---

### 提出的新方法与新思路

#### （1）X-CoSD：基于 Hybrid Resampling (HR) 的跨词汇 CoSD 框架
- **核心机制**：利用 **Token-Level Intersection (TLI)** 将 SLM 的候选 token 限制在 SLM 与 LLM 的公共词汇 $ \mathcal{V}_c $ 上。
- **创新设计**：提出 **Hybrid Resampling (HR)**，将残差重采样（residual resampling）过程分布到设备端和服务器端：
  - 当 token 被拒绝后，服务器仅发送 LLM 在 $ \mathcal{V}_c $ 上的概率分布及 LLM-only 区域的总概率质量 $ \theta_o $。
  - 设备根据 $ \theta_c $ 和 $ \theta_o $ 决定重采样区域：
    - 若选择 $ \mathcal{V}_c $：本地从残差分布中采样；
    - 若选择 $ \mathcal{V}_o $（LLM-only）：请求服务器采样并返回字符串，设备再 tokenize 后追加。
- **优势**：避免了传输整个 LLM 分布，大幅减少下行链路（downlink）通信量。

#### （2）X-CoSD-E：基于 Server Resampling with Device Verification (SR-DV) 的增强版本
- **核心机制**：服务器生成少量替换候选 token 及其概率，由设备进行本地验证。
- **流程**：
  - 服务器采样 K 个候选 token $ z_k \sim q(x) $，连同 $ q(z_k) $ 发送给设备。
  - 设备以概率 $ \alpha(z_k) = \frac{\max(q(z_k) - p^*(z_k), 0)}{q(z_k)} $ 接受。
  - 若任一候选被接受，则作为替换 token；否则最多尝试 M 次，失败后回退至 HR。
- **优势**：进一步减少了对完整分布的传输需求，仅在极端情况下才触发 HR，极大提升了通信效率。

---

### 相比现有方法的优势

| 特性 | X-CoSD / X-CoSD-E | Naive Extension | U-HLM | GR/TR |
|------|-------------------|------------------|--------|-------|
| 异构词汇支持 | ✅ | ✅（但通信高） | ❌（需共享） | ❌ |
| 无损解码 | ✅（理论证明） | ✅ | ❌（跳过验证） | ❌ |
| 下行通信负载 | ⬇️⬇️（HR 或 SR-DV） | ⬆️⬆️（全分布） | 中等 | 低 |
| 上行通信负载 | 低 | 低 | 高（上传 SLM 分布） | 低 |
| 实用性 | 高（无需训练/适配） | 低（带宽瓶颈） | 中（牺牲质量） | 低（有偏） |

> ✅ **X-CoSD 是首个同时满足“支持异构词汇 + 无损 + 低通信”的 CoSD 框架**。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **WMT-DeEn**：德英机器翻译
- **XSum**：单句摘要
- **CNN/DailyMail**：新闻摘要
- **GSM8K**：数学推理
- **MMLU**：多任务问答

### 实验设置
- **硬件模拟**：
  - 用户设备：NVIDIA TITAN RTX（运行 Vicuna-68M）
  - 服务器：NVIDIA A100（运行 Llama-3.1-8B 或 Qwen2-7B）
- **模型配置**：
  - SLM: `double7/vicuna-68m` ($|\mathcal{V}_s|=32,000$)
  - LLM: `meta-llama/Llama-3.1-8B`, `Qwen/Qwen2-7B`
  - 共享词汇占比约 14.7%–17.5%
- **参数设置**：
  - Drafting length $ N \in \{1,\dots,8\} $
  - X-CoSD-E: $ K=20, M=10 $
  - 温度：0.3，最大生成长度：128
- **网络环境**：
  - 上行速率：30 Mbps（典型 5G 上行）
  - 下行速率：100–300 Mbps（可变）

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **Generation Quality** | BLEU (WMT), ROUGE-2 (Summarization), Accuracy (GSM8K/MMLU)，相对于 Server LLM 的相对得分 |
| **Communication Load** | 每生成 token 的上行/下行比特数（bits/token） |
| **Token Throughput** | tokens/sec，衡量吞吐性能 |
| **Per-token Latency** | 单个 token 的延迟 CDF 分布 |

---

### 基线方法对比
| 方法 | 简介 |
|------|------|
| **Naive** | TLI 扩展版，拒绝时传输完整 LLM 分布 |
| **UL (Uplink)** | 设备上传 SLM 分布，服务器执行重采样 |
| **GR/TR** | 贪婪/直接重采样，有损方法 |
| **U-HLM** | 不确定性感知验证，压缩传输但跳过部分验证 |
| **Server LLM** | 完全在服务器上自回归生成 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### ✅ 生成质量（Relative Score vs Server LLM）
| 方法 | WMT-DeEn | XSum | GSM8K | MMLU |
|------|----------|------|--------|------|
| **X-CoSD** | ~1.02–1.05 | ~0.97–1.02 | ~1.00 | ~0.99–1.03 |
| **X-CoSD-E** | ~1.02 | ~0.98–1.00 | ~0.99–1.00 | ~0.99–1.02 |
| **Naive** | ~1.00–1.04 | ~0.98–0.99 | ~1.00 | ~0.98 |
| **U-HLM** | ↓↓ 严重下降（最低 0.33） |

> 📌 **X-CoSD/X-CoSD-E 保持与 Server LLM 相当的质量，而 U-HLM 等方法因跳过验证导致显著降质**。

---

#### 🔽 通信负载（Downlink Load per Token, N=2）
| 方法 | Qwen2-7B (Downlink) | Llama-3.1-8B (Downlink) |
|------|------------------------|----------------------------|
| **X-CoSD** | **0.2M bits** | **0.1–0.2M bits** |
| **X-CoSD-E** | **369–384 bits** | **266–379 bits** |
| **Naive** | **1.1–1.2M bits** | **0.6–1.1M bits** |
| **UL** | ~11 bits | ~9 bits（但上行极高） |

> 📌 **X-CoSD 将下行通信减少约 5–6 倍；X-CoSD-E 进一步降至 KB 级别，仅为 Naive 的 ~0.03%**。

---

#### ⚡ Token Throughput（Figure 3）
- 在 100 Mbps 下行下：
  - **X-CoSD-E** 达到最高吞吐（如 >45 tokens/sec）
  - 显著优于 **Naive (~35)** 和 **Server LLM (~30)**
  - **U-HLM** 最高但牺牲质量
- **X-CoSD-E 对下行速率变化鲁棒性强**，而 X-CoSD 和 Naive 受限于分布传输频率。

---

#### ⏱ Per-token Latency（Figure 4）
- **X-CoSD-E** 的延迟 CDF 最左移，表明更低且更稳定的延迟。
- 在 30 Mbps 上行 + 100 Mbps 下行下：
  - **X-CoSD-E** 平均延迟低于 **X-CoSD** 和 **Naive**
  - 仅略高于有损方法 **U-HLM**

---

### 消融实验分析（隐含于设计比较）
- **HR vs Full Distribution Transmission** → 通信下降 5–6×
- **SR-DV + Fallback to HR** → 通信进一步下降两个数量级
- **Fallback 机制确保无损性**：即使所有候选被拒，仍能通过 HR 恢复正确分布

---

## 4. 关键结论和发现

### 主要发现
1. **异构词汇下的 CoSD 是可行且必要的**：现实场景中 SLM 与 LLM 来源多样，必须支持非共享词汇。
2. **X-CoSD 实现了“三赢”平衡**：
   - ✅ **Lossless decoding**：严格保留 server LLM 分布（Theorem 1 & 2 已证明）
   - ✅ **Low communication**：HR 和 SR-DV 极大压缩下行流量
   - ✅ **High throughput**：token 生成速度显著提升
3. **X-CoSD-E 是更优实践方案**：SR-DV 机制使得绝大多数情况无需传输分布，适合真实部署。
4. **通信瓶颈已从“带宽”转向“策略设计”**：合理的分布拆分与采样机制比单纯压缩更有效。

---

### 方法的局限性（D Limitations）
1. **依赖公共词汇大小**：TLI 要求 $ \mathcal{V}_c $ 足够大，当前实验中仅占 ~15%，若过小可能影响 drafting 效率。
2. **未考虑动态词汇映射**：如 OmniDraft 中的在线对齐机制未整合，无法扩展覆盖范围。
3. **实验规模有限**：仅测试两种 LLM（Qwen2-7B, Llama-3.1-8B），泛化性有待验证。

---

### 未来工作方向
1. **减少对 $ \mathcal{V}_c $ 的依赖**：探索 subword-level alignment 或 soft mapping 机制。
2. **支持多轮 fallback 优化**：结合缓存机制避免重复传输。
3. **端到端联合优化**：将 HR/SR-DV 与模型微调结合，提升 drafting 准确率。
4. **跨模态 CoSD 扩展**：应用于语音、图像等多模态生成任务中的协作推理。

---

> ✅ **总结一句话**：  
> **X-CoSD 首次实现了在异构词汇条件下高效、无损的 Collaborative Speculative Decoding，为边缘-云协同 LLM 推理提供了实用化路径**。

</details>

---

### 3. [Online Draft Co-Training for Speculative Decoding in Large-Scale, Long-Context RL Post-Training](https://arxiv.org/abs/2609.07108)

**Authors**: Zili Wang, Zhaopeng Qiu, Yuekai Zhang, Shuang Yu, Junjie Lai  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.07108v1  

#### Abstract
Speculative decoding accelerates rollout generation, which dominates the cost of reinforcement learning (RL) post-training. Online co-training can further increase the draft's accuracy, yielding greater speedups. However, scaling this approach to co-training on large models with long contexts poses ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Online Draft Co-Training for Speculative Decoding in Large-Scale, Long-Context RL Post-Training**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在大规模、长上下文的强化学习（RL）后训练中，**rollout 生成**是计算成本的主要瓶颈。虽然 **Speculative Decoding (SD)** 可以通过并行验证加速生成，但其性能依赖于 draft 模型的质量。随着策略模型的演化，固定 draft 模型会逐渐失配，导致接受长度下降。

此外，现有的分布式训练系统（如基于 **Context Parallelism (CP)** 和 **Pipeline Parallelism (PP)** 的架构）不支持：
- **Branch attention**：先进 draft 架构（如 EAGLE-3、DFlash、DSpark）需要对主序列前缀和分支局部上下文同时进行 attention。
- **跨阶段特征传输**：draft 模型通常位于最后一个 PP 阶段，但所需的中间 target 特征分布在多个 PP 阶段上，标准 PP 通信无法高效传递这些“tap”特征。

因此，如何在保持现有分布式训练拓扑不变的前提下，实现 **online draft co-training** 是一个系统级挑战。

---

### **提出了什么新方法或新思路**

本文提出了一套端到端系统，支持在大规模、长上下文 RL 后训练中进行 **online draft co-training**，主要包含两个核心技术：

#### **(1) Branch Attention under Context Parallelism**
- 将 draft 的 branch attention 分解为两个部分：
  - **Main-sequence component**：对主序列因果前缀的 attention，使用 **packed, load-balanced zigzag ring attention** 处理。
  - **Branch-local component**：对分支内部 token 的 attention，在拥有该分支的 rank 上本地完成。
- 最终通过 **log-sum-exp 合并机制** 融合两部分输出，恢复完整 attention 结果。
- 支持多种先进 draft 架构（EAGLE-3、DFlash、DSpark），无需修改 CP 并行布局。

#### **(2) TapChannel under Pipeline Parallelism**
- 引入 **TapChannel**，一种独立于 pipeline schedule 的侧通道（side path），用于将中间 target 特征从各 PP 阶段直接传输到 draft 所在的最后阶段。
- 使用 **per-source mailbox + 序列戳（sequence stamp）** 实现生产者-消费者同步。
- 支持跨节点（GPUDirect RDMA）和同节点（CUDA IPC）高效传输，不影响原有 pipeline 流水线调度。

---

### **相比现有方法的优势**

| 方面 | 本文方法 | 现有方法（如 USP、SpecForge） |
|------|--------|-----------------------------|
| **CP 支持** | 支持 branch attention，内存更优，延迟更低 | 不支持 branch attention 或需特殊并行布局 |
| **PP 支持** | TapChannel 无侵入式传输，不影响 pipeline schedule | 需修改 pipeline 或引入主机暂存（host staging），开销大 |
| **系统兼容性** | 完全兼容现有 CP/PP 拓扑，无需重构 | 往往要求定制化并行策略 |
| **扩展性** | 在 256K 长序列下良好扩展，PP 开销小 | 长序列扩展受限，PP 通信成瓶颈 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **DAPOMath-17K**：用于单轮数学推理任务，评估 reward、accuracy 和 KL 散度。
- **AIME 2024**：作为 validation 数据集，测试模型数学能力。
- **NeMo Gym Workplace Assistant**：多轮工具调用环境，模拟办公场景中的复杂任务执行，用于多轮 workload 评估。

---

### **实验设置**
- **目标模型（Target Models）**：
  - `Qwen3-8B`（基准）
  - `Qwen3.5-35B-A3B`, `Qwen3.5-122B-A10B`
  - `Nemotron-3.5-Lightning-30B-A3B`
  - `GPT-OSS-120B`
- **Draft 模型家族**：
  - **EAGLE-3**：基于 TTT（Training-Time Test）的多步自回归 draft
  - **DFlash**：基于 block diffusion 的并行块生成
  - **DSpark**：结合 Markov head 的 semi-autoregressive 生成
- **硬件平台**：
  - H100 GPU（用于 8B–35B）
  - GB200 GPU（用于 122B 及以上）
- **并行配置**：
  - TP/PP/CP/EP 组合，最大 CP=8，PP=4

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Reward** | RL 训练过程中的累积奖励，衡量策略质量 |
| **Validation Accuracy** | 在 AIME2024 上的准确率 |
| **Training-Inference KL Divergence** | 训练与推理 backend 输出分布的一致性，验证数值正确性 |
| **Acceptance Length** | 每次验证平均接受的 draft token 数量 |
| **Rollout Throughput** | 单位时间生成的 token 数 |
| **End-to-End Step Time** | 完整 policy update 步骤的时间 |
| **Per-GPU Peak Memory** | 显存占用峰值 |
| **Latency (Forward+Backward)** | attention 层级延迟 |

---

### **基线方法对比**
- **Baseline**：无 speculative decoding，无 draft co-training
- **With SD + Online Co-Training**：本文方法（EAGLE-3 / DFlash / DSpark）
- **对比系统**：
  - **USP (Ulysses × ring)**：SpecForge 中的 baseline CP 实现
  - **Host-staging**：通过主机内存中转 tap 特征的传统方式

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **学习稳定性**
- 所有 co-trained draft 配置均能 **紧密跟踪 baseline 学习轨迹**。
- **Training-Inference KL ≈ 0**，表明训练与推理一致性高，满足 GRPO 的 on-policy 假设。
- 在 DAPOMath 和 Workplace Assistant 上，reward 和 accuracy 曲线几乎重合。

#### ✅ **加速效果（End-to-End Speedup）**
| Target Model | Draft Model | Accepted Length | Rollout Speedup | **E2E Speedup** |
|--------------|-------------|------------------|------------------|----------------|
| Qwen3-8B     | EAGLE-3     | 2.28             | 1.63×            | **1.50×**      |
| Qwen3-8B     | DFlash      | 3.45             | 2.23×            | **1.88×**      |
| Qwen3-8B     | DSpark      | 3.63             | 2.18×            | **1.83×**      |
| Qwen3.5-122B | DFlash      | 4.78             | 1.72×            | **1.35×**      |
| GPT-OSS-120B | DFlash      | 3.80             | 1.48×            | **1.19×**      |

> 💡 **最高达 1.88× 端到端加速**，且在 122B 规模仍有效。

---

#### ✅ **Context Parallelism 性能优势**
- 对比 **USP**（SpecForge 实现）：
  - **延迟降低最多达 2.9×**
  - **每 GPU 内存减少 2.7×**
  - 在 **256K 序列长度下实现强扩展性**
- 图 6 显示：CP=8 时，TTT attention 延迟从 17.7s（CP=1）降至 2.35s（7.5× 加速，94% 并行效率）

---

#### ✅ **Pipeline Parallelism 开销分析**
- **TapChannel vs Host-staging**：
  - 传输速度提升 **4.5–8.5×**
  - **Source stages 几乎无干扰**（<2% 性能影响）
  - 接收端仅 **1.6% HBM contention**
  - 而 host staging 导致所有 ranks **慢 83–88%**
- **每步 tap 等待时间仅 0.4–0.6s**，占优化时间 **1.5–2.2%**，说明大部分通信被流水线自然掩盖。

---

#### ✅ **消融实验（Ablation）**
- **Table 2** 显示：
  - DSpark 虽带来最大加速（1.85×），但 PP 开销仅 **14.6%**
  - EAGLE-3 因 TTT 多步 forward，PP 开销较高（34.3%）
  - 尽管增加训练开销，但 **rollout 时间减少 28–52%**，净收益显著

---

## **4. 关键结论和发现**

### **主要发现**
1. **Online draft co-training 可稳定集成进大规模 RL 训练流程**，且不破坏学习动态。
2. **DFlash 和 DSpark 在 acceptance length 上优于 EAGLE-3**，更适合高吞吐场景。
3. **本文的 CP 设计显著优于 USP**，尤其在长序列和低显存方面优势明显。
4. **TapChannel 实现了轻量级、非阻塞的跨阶段特征传输**，是 PP 下 co-training 的关键使能技术。
5. **多轮任务中 rollout 占比下降（~55.8%）**，导致端到端加速低于单轮任务，凸显了系统协同优化的重要性。

---

### **方法的局限性**
- **对 MoE 模型加速有限**：由于稀疏路由本身开销大，speculative decoding 的相对收益较小（见 GPT-OSS-120B 结果）。
- **Linear attention 模型受益较少**：因原生验证速度快，speculation 提升空间小（Wang et al., 2026c）。
- **PP 开销随 source stages 增加而上升**，极端情况下可能成为瓶颈。

---

### **未来工作方向**
- **Tailoring SD for MoE models**：设计稀疏感知的 speculative decoding 策略。
- **Optimize for linear attention models**：开发适用于 RWKV、RetNet 等架构的专用 draft 机制（如 Specla）。
- **Dynamic draft selection**：根据上下文复杂度自适应选择 draft 架构或 speculative 长度。
- **Integration with other parallelisms**：探索与 Sequence Parallelism、Expert Parallelism 的协同优化。

---

> 🔚 **总结**：本文首次实现了在 **大规模、长上下文、分布式 RL 训练中实用化的 online draft co-training**，通过 **branch attention under CP** 和 **TapChannel under PP** 两大系统创新，解决了关键部署障碍，并在真实场景中验证了高达 **1.88× 的端到端加速**，为下一代高效 RLHF 系统提供了重要基础设施支持。

</details>

---

### 4. [SMCC-Empowered Digital Twins for Sensorless Monitoring in Large-Scale AI-Driven IoT Systems](https://arxiv.org/abs/2609.09161)

**Authors**: Vincenzo Sammartino  
**Category**: cs.DC  
**Published**: 2026-09-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.09161v1  

#### Abstract
The deployment of AI-driven Digital Twins (DTs) in large-scale Internet-of-Things (IoT) ecosystems demands continuous, high-fidelity synchronization between the physical environment and its virtual replica. Conventional approaches rely on dense sensor deployments, which introduce prohibitive costs i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*SMCC-Empowered Digital Twins for Sensorless Monitoring in Large-Scale AI-Driven IoT Systems*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 AI 驱动的 **Digital Twin (DT)** 系统严重依赖密集部署的物理传感器来获取环境数据，这在大规模 IoT 场景（如智能工厂、智慧城市）中面临三大挑战：
- **高硬件成本**：传感器数量随资产线性增长，经济上不可持续；
- **通信瓶颈**：高频传感数据导致无线网络拥塞，增加同步延迟；
- **边缘资源受限**：大型 AI 模型（如 Vision Transformers）对 Edge 服务器的 Memory 和 Computation 资源要求过高。

此外，现有研究通常将 Sensing、Memory、Communication、Computation (SMCC) 各层独立优化，忽略了其间的耦合关系。

### 提出的新方法与新思路
本文提出 **SMCC-DT** 框架，一种基于 **SMCC 范式** 的端到端集成架构，实现无需专用传感器的 DT 维护：

- **Sensorless Monitoring via ISAC**：利用 **6G Integrated Sensing and Communication (ISAC)** 波形，单个无线信号同时完成环境感知（Sensing）和数据传输（Communication），消除对专用传感器的需求。
- **四维联合优化**：在 Edge 层面统一建模并联合优化：
  - **Sensing**：ISAC 发射功率与波束成形向量；
  - **Communication**：信道带宽与速率保障；
  - **Memory**：AI 模型权重与数据缓冲区的内存划分；
  - **Computation**：CPU 频率调节以控制推理延迟与能耗。
- **DRL 驱动的资源分配**：设计 **SMCCAGENT** —— 一个基于 **Proximal Policy Optimization (PPO)** 的 Deep Reinforcement Learning (DRL) 代理，在线学习近最优资源分配策略。

### 相比现有方法的优势
- **跨层协同增效**：打破传统“分而治之”模式，显式建模 SMCC 四者之间的权衡（trade-off），例如：
  - 更多 Sensing 功率提升精度但降低 Communication SNR；
  - 更大模型提升推理质量但占用更多 Memory 并增加计算负载。
- **显著降低延迟与能耗**：通过联合优化避免系统瓶颈，实验证明相比基线大幅减少 **DT synchronization latency** 和 **energy consumption**。
- **支持大规模部署**：适用于 500+ 节点的工业 IoT 场景，具备良好可扩展性。

---

## 2. 核心实验方法和设置

### 数据集与仿真环境
- **无真实数据集**，采用 **Monte Carlo 仿真实验** 构建一个包含 **K = 500 个物理资产** 的工业 IoT 测试平台（Industrial IoT Testbed）。
- 场景模拟为 200×200×50 m³ 的“智能工厂”，资产分布在三个生产区域。
- 所有参数基于 6G 物理层与边缘计算典型值设定（见 Table I）。

### 实验设置
- **ISAC 参数**：
  - 载频：28 GHz；
  - 带宽：Bs = Bc = 100 MHz；
  - 基站天线数：Nt = 64；
  - 最大发射功率：Pmax = 40 dBm。
- **Edge Server 配置**：
  - 内存容量：Mmax = 32 GB；
  - CPU 频率范围：[1.0, 4.0] GHz；
  - 支持三种 AI 模型规模：Small (7M params), Medium (125M), Large (1.3B)。
- **训练配置**：
  - 使用 PPO 算法训练 SMCCAGENT；
  - 训练轮次：10,000 episodes；
  - 奖励函数结合延迟最小化与约束违反惩罚（reward shaping）。

### 评估指标
| 指标 | 描述 |
|------|------|
| `Tsync` | End-to-end DT synchronization latency（目标最小化） |
| `Etot` | 总能量消耗（含 Sensing 与 Computation） |
| `Sensing Accuracy` | 由 CRLB 定义的估计误差上限，需满足 ek ≤ emax |
| `Throughput` | 通信速率 R ≥ Rmin = 500 Mbps |
| `Memory Fragmentation` | 内存碎片化程度（间接体现资源利用率） |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **OA (Orthogonal Allocation)** | Sensing 与 Communication 使用分离频段，丧失 ISAC 频谱效率；Memory 与 Computation 独立优化 |
| **CO (Compute-Only Optimization)** | 仅优化 fcpu 与 Ω，固定功率分配与内存分区 |
| **GH (Greedy Heuristic)** | 规则式贪婪策略：优先保证 Sensing 精度 → Communication 吞吐 → Computation |
| **RA (Random Allocation)** | 在可行域内随机采样决策变量，作为下界参考 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（K=500）
| 方法 | `Tsync` (ms) | `Etot` (J) | Sensing Accuracy (%) |
|------|--------------|------------|------------------------|
| **SMCC-DT (Ours)** | **12.3** | **6.3** | **96.2%** |
| OA | 20.1 | 8.7 | 95.8% |
| CO | 16.9 | 8.7 | 96.1% |
| GH | 15.8 | 7.9 | 94.5% |
| RA | ~25.0 | ~10.0 | <90% |

### 与基线对比结果
- **同步延迟降低 38.7%**：相比 OA（20.1 → 12.3 ms），得益于 ISAC 频谱共享与跨层联合优化；
- **总能耗降低 27.4%**：相比 OA（8.7 → 6.3 J），主要来自智能 CPU frequency scaling 与功率分配；
- **维持高精度与吞吐**：Sensing accuracy > 95%，Throughput > 30 fps（帧每秒），满足实时 DT 更新需求；
- **内存碎片减少 52.1%**：相比静态分区方案，动态内存管理更高效。

### 消融实验结果（Ablation Study, K=500, Medium Model）
| 配置 | `Tsync` (ms) | `Etot` (J) | Acc. (%) | 分析 |
|------|-------------|-----------|----------|------|
| Full SMCC-DT | 12.3 | 6.3 | 96.2 | 基准 |
| w/o Sensing opt. | 14.0 | 7.1 | 93.5 | 固定功率分配削弱感知能力 |
| w/o Memory opt. | 14.9 | 6.8 | 96.0 | 静态分区导致缓冲不足或模型受限 |
| w/o Computation opt. | 12.5 | 8.6 | 96.1 | 固定高频运行造成能效低下 |
| w/o Communication opt. | 13.8 | 6.5 | 95.8 | 通信干扰未优化影响数据交付 |

> ✅ 结论：所有四个 SMCC 维度均对整体性能有实质性贡献，验证了**跨层协同设计的必要性**。

---

## 4. 关键结论和发现

### 主要发现
- **SMCC 范式有效解决了大规模 DT 的资源瓶颈问题**：通过将 Sensing、Memory、Communication、Computation 统一建模，能够捕捉关键 trade-off，实现全局最优而非局部次优。
- **ISAC 是实现 sensorless monitoring 的关键技术路径**：单一无线信号即可替代大量物理传感器，显著降低成本与部署复杂度。
- **DRL 可有效求解 NP-hard 的跨层优化问题**：SMCCAGENT 能在线学习稳定、高效的资源分配策略，适应动态环境变化。
- **联合优化带来非线性增益**：各模块独立优化无法达到 SMCC-DT 的性能水平，证明“1+1+1+1 > 4”的协同效应。

### 方法的局限性
- **依赖精确的信道状态信息 (CSI)**：实际中 CSI 获取存在误差与时延，可能影响性能；
- **仿真环境理想化**：未考虑移动性、多径衰落、硬件损伤等现实因素；
- **模型选择离散化**：Ω ∈ {7M, 125M, 1.3B}，缺乏连续搜索空间；
- **集中式 DRL 架构**：在超大规模网络中可能存在可扩展性问题。

### 未来工作方向
- 扩展至 **multi-server federated DT settings**，支持分布式协同更新；
- 引入 **Reconfigurable Intelligent Surfaces (RIS)** 到 ISAC pipeline，进一步增强感知与通信性能；
- 在真实 **6G prototype hardware testbed** 上进行原型验证；
- 探索 **semantic-aware sensing** 与 **task-oriented communication**，提升信息传输效率。

--- 

> 📌 **一句话总结**：  
> 本论文提出的 **SMCC-DT** 框架通过融合 **ISAC** 与 **SMCC 协同优化**，实现了无需传感器的大规模 AI 驱动 Digital Twin 实时同步，并借助 **PPO-based DRL (SMCCAGENT)** 实现高效资源调度，在 500 节点工业 IoT 场景下相较基线降低了 **38.7% 延迟** 与 **27.4% 能耗**，验证了跨层智能设计的巨大潜力。

</details>

---

### 5. [Reasoning-Aware Compression: Identifying and Protecting Vulnerable Reasoning Circuits for Energy-Efficient LLM Deployment](https://arxiv.org/abs/2609.05512)

**Authors**: Leonard Twagirayezu, Prasenjit Mitra  
**Category**: cs.AI  
**Published**: 2026-09-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.05512v1  

#### Abstract
Large Reasoning Models (LRMs) impose substantial energy costs during deployment, yet current compression methods apply uniform quantization across all components, risking damage to critical reasoning circuits. We present a reasoning-aware compression framework that benchmarks quantization conditions...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Reasoning-Aware Compression**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
大型推理模型（Large Reasoning Models, LRMs）在部署时面临高昂的能源消耗和内存占用问题。传统的量化压缩方法（如AWQ、GPTQ）采用统一的低精度（如INT4）处理所有模型组件，忽略了不同模块对推理能力的敏感性差异，导致关键“推理电路”受损，从而严重降低模型在复杂推理任务上的准确性。

例如，Zhang et al. (2025) 发现将 R1-Distill-Llama-8B 进行 3-bit 量化后，其在 AIME 2024 上的准确率从 42.2% 骤降至 10.0%，而仅保护最终层的 MLP 即可恢复 6.57% 的准确率，说明损伤高度局部化。

### **提出了什么新方法或新思路**
本文提出了一种 **reasoning-aware compression** 框架，其核心是：
- **细粒度脆弱性分析**：通过扰动扫描（perturbation sweep）对每个 `(layer, projection)` 对进行独立量化测试，构建每模块的脆弱性评分 $ V(l,p) $。
- **选择性混合精度压缩**：仅将最脆弱的模块恢复为 FP16，其余保持 INT4，实现精准保护。
- **基于校准集的非循环评估**：使用与评估集分离的校准集进行脆弱性建模，避免数据泄露。

该方法首次系统地识别并保护 LRMs 中的关键推理路径，而非全局统一压缩。

### **相比现有方法的优势**
| 维度 | 现有方法（如 GPTQ/AWQ） | 本文方法 |
|------|------------------------|----------|
| 压缩策略 | 全局均匀量化 | 模块级差异化保护 |
| 能效权衡 | 准确率大幅下降 | 在更低能耗下实现更高准确率（Pareto 最优） |
| 适用场景 | 通用语言任务 | 特别优化数学、逻辑等推理任务 |
| 能量测量 | 多为估算或忽略输出长度影响 | 实际硬件级 GPU 功耗采样（NVML） |

> ✅ **优势总结**：实现了**任务感知、模块感知、能量感知**三位一体的压缩框架，在多个推理基准上达到现有方法无法企及的精度-能效平衡点。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
五个具有代表性的推理基准，覆盖多种推理技能：

| 数据集 | 类型 | 任务描述 |
|-------|------|---------|
| **GSM8K** | Arithmetic | 小学数学应用题 |
| **MATH-500** | Competition Math | 竞赛级数学问题 |
| **FOLIO** | Logic/NLI | 一阶逻辑自然语言推断（True/False/Unknown） |
| **ProofWriter** | Formal Deduction | 形式化演绎推理（最多5步推理链） |
| **MuSiQue** | Multi-hop QA | 多跳问答（需2–4步推理） |

> 所有数据集均划分为 **50% 校准集（用于脆弱性评分） + 50% 持留评估集（用于最终报告）**，确保无数据泄露。

### **实验设置和评估指标**

#### **模型**
- `R1-Llama-8B`（8B 参数，Llama 架构）
- `R1-Qwen-7B`（7B 参数，Qwen 架构）

均为 DeepSeek-R1 系列蒸馏后的指令调优推理模型，支持 `<think>...</think>` 输出完整推理链。

#### **压缩条件（5种）**
| 条件 | 描述 |
|------|------|
| **FP16 Baseline** | 全浮点16位，作为准确率上限和能耗下限 |
| **Full INT4** | 所有线性投影 NF4 量化至 INT4 |
| **INT4+Attention** | 注意力投影（q,k,v,o）恢复 FP16，MLP 保持 INT4 |
| **INT4+MLP** | MLP 投影（gate, up, down）恢复 FP16，注意力保持 INT4 |
| **Comp. (Selective)** | Top-K% 最脆弱模块恢复 FP16（K ∈ {10%, 20%, 30%, 40%}） |

#### **评估指标**
- **Accuracy (%)**：按任务定义（数值匹配 ±0.01 / 字符串匹配 / EM）
- **Energy per Query (J)**：$ E = P \times \Delta t $，其中 $ P $ 为平均功率（W），$ \Delta t $ 为推理时间（s）
- **Output Token Count**：输出推理链长度
- **Power Draw (W)**：GPU 实时功耗（NVML 采样频率 100ms）
- **Memory Usage**：VRAM 占用情况

#### **硬件平台**
- Tesla V100-SXM2 GPU（32GB HBM2）
- Pittsburgh Supercomputing Center (PSC) Bridges-2 集群
- 使用 `pynvml` 进行精确功耗监控

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **✅ 最大亮点：R1-Qwen-7B 在 ProofWriter 上实现帕累托改进**
| 指标 | FP16 | Full INT4 | **Top-10% Selective** |
|------|------|-----------|------------------------|
| Accuracy | 72.00% | 82.00% | **84.00%** (+12.00pp) |
| Energy | 1,931 J | 1,796 J | **1,743 J** (-9.7%) |
| Output Tokens | 391 | 368 | **354** |

> 🔥 **这是首次在真实硬件上实现“更准 + 更省电”的压缩效果**，且验证于持留数据集。

#### **其他显著结果**
- **MATH-500 上超越 FP16 精度**：
  - R1-Qwen-7B Top-30% 达到 **59.17%**（+3.17pp over FP16）
  - R1-Llama-8B Top-20% 达到 **49.17%**（+35.84pp over FP16！）
- **GSM8K 上 Full INT4 反而增加能耗**：
  - R1-Llama-8B：输出 token ↑22.8%，虽功率↓25%，但总能量 **↑22.9%**
- **MuSiQue 对所有压缩极度敏感**：无任何 INT4 条件优于 FP16，表明多跳知识检索需保留高精度

### **与基线方法的对比结果**

| 方法 | 相比 Uniform INT4 的优势 |
|------|--------------------------|
| **INT4+Attention** | 在数学任务（GSM8K, MATH-500）上显著提升准确率 |
| **INT4+MLP** | 效果较差，甚至劣于 Full INT4；且因 MLP 参数占比达 ~80%，恢复后显存超 FP16 基线（~22GB > 16GB），不适用于 <24GB GPU |
| **Selective Compression (Top-K%)** | 在 ProofWriter 和 MATH-500 上明显优于所有统一策略，尤其 Top-10% 表现最佳 |

> ⚠️ **反直觉发现**：在某些任务（如 ProofWriter for Qwen-7B），**Full INT4 本身已优于 FP16**（77.08% vs 69.58%），说明量化噪声可能起到隐式正则化作用。

### **消融实验结果（Appendix B）**

比较三种保护策略在不同 K 下的表现：

| 策略 | 定义 |
|------|------|
| **Top-K%** | 按脆弱性评分排序，保护前 K% 模块 |
| **Random-K%** | 随机选择 K% 模块恢复 |
| **Bottom-K%** | 保护最不脆弱的模块（负向控制） |

#### **关键发现**：
- 在 **MATH-500** 和 **ProofWriter (Qwen-7B)** 上，**Top-K% 显著优于 Random-K% 和 Bottom-K%**，证明脆弱性评分具有真实信号。
- 在 **FOLIO/GSM8K** 上，三者表现接近，说明这些任务中模块敏感性较低或分布较平缓。
- 在 **ProofWriter (Llama-8B)** 上，Random-K% 反而优于 Top-K%，说明脆弱性模式未跨数据集泛化。

> ✅ 结论：**脆弱性引导的选择性压缩仅在模块敏感性强且一致的任务中有效**。

---

## **4. 关键结论和发现**

### **主要发现**

1. **INT4 量化可能增加总能耗**
   - 原因：量化噪声延长了 chain-of-thought 推理链（output tokens ↑），抵消了单位功耗下降。
   - 示例：R1-Llama-8B 在 GSM8K 上 Full INT4 能耗 **上升 22.9%**。
   - 👉 **启示**：必须测量端到端能量（E = P × t），不能仅看功率或理论计算量。

2. **模块脆弱性具有任务依赖性**
   - 数学推理（GSM8K/MATH-500）：**Attention 投影更关键** → INT4+Attention 表现最好
   - 逻辑推理（ProofWriter）：**敏感性因架构而异**：
     - Qwen-7B：特定输出投影（如 o_proj）极敏感
     - Llama-8B：整体鲁棒，Full INT4 即可
   - 多跳问答（MuSiQue）：**全模型敏感**，FP16 必不可少

3. **选择性压缩可达帕累托最优**
   - R1-Qwen-7B 在 ProofWriter 上实现 **+12pp 准确率 & -9.7% 能耗**
   - 传统方法无法达到此组合，证明了细粒度保护的价值

4. **多数模块可安全压缩**
   - 在大多数任务中，**87%-100% 的 (layer, projection) 对单独量化时不引起准确率下降**
   - 支持“少数关键电路决定整体性能”的假设，为稀疏保护提供依据

### **方法的局限性**

| 局限 | 说明 |
|------|------|
| **硬件单一性** | 实验仅在 V100 上完成，A100/H100 可能表现不同 |
| **样本量限制** | MuSiQue 等低准确率任务下统计效力不足（30样本/实验） |
| **仅后训练压缩** | 未结合 QAT 或微调，潜在性能上限受限 |
| **泛化性挑战** | 脆弱性评分在部分任务（如 FOLIO Qwen-7B）未能跨数据集泛化 |
| **内存开销** | 混合精度带来额外管理成本，实际部署需考虑调度效率 |

### **未来工作方向**
1. **低秩近似恢复层**：减少 FP16 模块的内存开销
2. **任务自适应脆弱性建模**：提升评分跨数据集泛化能力
3. **更大规模模型验证**：扩展至 30B–70B 模型，能量节省潜力更大
4. **联合压缩与训练**：探索 QAT + selective restoration 的协同优化
5. **动态推理路径保护**：根据输入动态激活保护模块（input-conditioned protection）

---

> 📌 **总结一句话**：  
> 本文揭示了 LRM 压缩中的“推理链延长悖论”，提出了首个基于模块脆弱性分析的 reasoning-aware 压缩框架，在多个推理任务上实现了前所未有的精度-能效双赢，推动了绿色 AI 与边缘智能的发展。

</details>

---

### 6. [SymbolicLight V2: Hybrid Neuromorphic Architecture and Sparse Execution for Low-Energy Language Inference](https://arxiv.org/abs/2609.09772)

**Authors**: Ting Liu  
**Category**: cs.CL  
**Published**: 2026-09-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.09772v1  

#### Abstract
SymbolicLight V2 combines sparse event computation with continuous-state processing in a hybrid neuromorphic language architecture. Extending V1's spike-gated dual paths, it adds graded signed events at further projections and softmax-free local attention. We implement the 194M-parameter model on an...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SymbolicLight V2: Hybrid Neuromorphic Architecture and Sparse Execution for Low-Energy Language Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前大语言模型（LLM）推理过程能耗高，尤其在边缘设备上部署时面临功耗瓶颈。尽管许多模型训练时引入了**activation sparsity**（激活稀疏性），但在传统硬件（如GPU）上执行时，这些零激活仍会触发不必要的计算和内存访问，无法真正节省能量。

本文旨在解决“**可执行稀疏性**”（executable sparsity）的问题——即如何将训练阶段的稀疏性转化为实际推理中的节能优势。

---

### **提出的新方法与新思路**
作者提出了 **SymbolicLight V2**，一种**混合神经形态架构**（hybrid neuromorphic architecture），结合了以下关键技术：

- **分级有符号事件编码**（graded signed events）：
  - 在 FFN 上投影、Q/K/V 投影等输入处引入事件编码器，将连续激活量化为带符号和幅度的整数事件（INT8 表示）。
  - 零值表示无事件，非零值携带符号与幅值，仅对非零事件执行计算。

- **稀疏事件驱动计算**（event-driven computation）：
  - 仅当输入事件非零时才加载对应的权重行（active-row weight gathering），跳过零输入的 MAC 操作和内存读取。
  - 在 FPGA 上通过 HBM 实现高效稀疏行检索。

- **局部注意力优化**：
  - 使用 **ReLU-L1 局部注意力** 替代 softmax，避免指数运算。
  - 引入 **linear position bias**（ALiBi）替代 RoPE，避免旋转离散事件向量。
  - 支持固定大小的 KV 缓存环形缓冲区（480 slots），并实现 **valid-state KV cache loading**，仅加载已填充的有效状态。

- **混合计算范式**：
  - 结合稀疏事件处理与连续状态维护（continuous residual stream 和 recurrent decay path），形成“事件选择 + 连续处理”的协同机制。
  - 不是纯 SNN，而是 hybrid 架构，保留部分连续操作（如 LayerNorm、输出头）以保证表达能力。

---

### **相比现有方法的优势**
| 维度 | SymbolicLight V2 的优势 |
|------|------------------------|
| **能效** | 显著降低每生成 token 的能耗（最高达 89.1% 节能 vs. RTX 5090 FP32）。 |
| **稀疏性利用** | 将稀疏性从训练特性转化为可执行机制，在 ARM 和 FPGA 上均实现真正的稀疏执行。 |
| **硬件适配性** | 提供完整的端到端数字定点实现原型（Alveo U50C FPGA），验证了专用加速器的设计基础。 |
| **延迟敏感场景价值** | 揭示“loaded-idle energy”占主导地位，因此缩短 token latency 可显著摊薄平台能耗。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：基于公开的 4B-token 数据集进行训练，使用 48K tokenizer。
- **测试数据**：
  - 使用单一输入序列进行推理测试。
  - 工作负载定义为 `pP/nN`：前缀 P 个 token，生成 N=128 个 token。
  - 测试前缀长度包括：32、128、256、480。

> 注：本研究聚焦于**固定 checkpoint 的推理执行效率**，不涉及新任务或下游 benchmark 的训练微调。

---

### **实验设置与评估指标**

#### **硬件平台**
| 平台 | 配置 |
|------|------|
| **FPGA** | Xilinx Alveo U50C，运行频率 175 MHz，INT8 权重，fixed-point 激活，整数事件编码 |
| **CPU** | ROCK 5T 开发板上的 4 个 Cortex-A76 核心，运行 ARM NEON 向量指令 |
| **GPU Baseline** | RTX 5090，FP32 + PyTorch `torch.compile`，KV cache，batch=1 |

#### **部署模式**
- **Continuous decode**：一次性生成多个 token，用于吞吐量测试。
- **Interactive mode**：逐 token 生成，支持多轮对话和 EOS 精确停止。

#### **评估指标**
| 指标 | 定义 |
|------|------|
| **Decode throughput (tok/s)** | 仅解码阶段每秒生成 token 数 |
| **Request throughput (tok/s)** | 包含 prefill 和 decode 的完整请求吞吐率 |
| **Energy per generated token (J/token)** | 分为：<br>• **Gross energy**：总能耗（含 idle）<br>• **Incremental energy**：扣除 loaded-idle 后的实际增量能耗 |
| **Power (W)** | 卡级功耗（DC card sensor）或整机 AC 输入功率 |

#### **测量边界说明**
- **FPGA/GPU**：使用 XRT/NVML 获取卡级功耗估计，不含主机和电源损耗。
- **ARM**：使用智能插座测量板载适配器 AC 输入，包含整个系统功耗。

---

### **基线方法对比**
| 对比项 | 基线 |
|--------|------|
| **主要对比** | RTX 5090 上的 compiled-FP32 推理（相同 checkpoint） |
| **其他精度对比** | GPU 上的 BF16、weight-only INT8 配置（见 Table 7） |
| **本地部署对比** | ROCK 5T 上的多种模型路径（Qwen、SmolLM2、LFM 等） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **FPGA 实现性能提升（p32/n128）**
| 实现方式 | Decode 吞吐 (tok/s) | Decode Energy (J/token) | Request Energy (J/token) |
|----------|---------------------|-------------------------|----------------------------|
| Resident-dense | 474.6 | 0.06087 | 0.07561 |
| Sparse-gather | 544.2 | 0.04904 | 0.06108 |
| **Partial-KV (最终版)** | **643.2** | **0.04407** | **0.05468** |

✅ **累计提升**：
- 吞吐量 ↑ **35.5%**
- 解码能耗 ↓ **27.6%**
- 完整请求能耗 ↓ **27.7%**

> ✅ 在 p128 和 p256 上，请求能耗也下降了 **25.9%** 和 **24.4%**。

---

#### **与 RTX 5090 FP32 的对比（p32/n128）**
| 指标 | FPGA (INT8) | GPU (FP32) | 节能比 |
|------|------------|-----------|--------|
| Decode Throughput | 643.2 tok/s | 406.9 tok/s | ↑ 1.58× |
| Decode Energy | 0.04407 J/token | 0.40315 J/token | ↓ **89.1%** |
| Request Energy | 0.05468 J/token | 0.40466 J/token | ↓ **86.5%**（7.40× 更低） |

📌 **结论**：在相同模型权重下，FPGA 整数部署比高端 GPU 的 FP32 方案节能近 **9 倍**。

---

#### **ARM 部署性能（ROCK 5T）**
| 前缀 | Request Throughput | Active Power | Gross Energy |
|------|--------------------|--------------|---------------|
| p32 | 65.4 tok/s | 9.80 W | 0.151 J/token |
| p128 | 37.5 tok/s | 9.41 W | 0.251 J/token |
| p256 | 23.7 tok/s | 8.94 W | 0.377 J/token |

💡 特点：
- 功耗稳定在 ~9–10W。
- 短上下文下具备实用级性能。

---

#### **消融实验结果**
| 改进措施 | 能耗影响 | 性能增益来源 |
|--------|--------|-------------|
| **Active-row gathering** | ↓ 8.1% 解码能耗（从 0.06087 → 0.04904） | 减少约 52% 权重字节访问（模拟） |
| **Valid-state KV loading** | ↓ 10.1% 解码能耗（从 0.04904 → 0.04407） | 消除早期空缓存加载开销，尤其在短前缀时有效 |
| **Loaded-idle energy 分析** | 占总能耗 **82.8%** | 快速生成可显著摊薄 idle 成本 |

📌 **重要发现**：即使 active power 变化不大，提高 throughput 也能大幅降低 **gross energy per token**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **稀疏执行确实可以显著降低推理能耗**：
   - 通过 active-row gathering 和 valid-state KV loading，实现了高达 **27.6%** 的能耗下降。

2. ✅ **事件编码可在真实硬件上高效执行**：
   - 在 FPGA 和 ARM 上均实现了 bit-exact 的整数推理，验证了部署可行性。

3. ✅ **loaded-idle energy 是节能的关键杠杆**：
   - FPGA 上 **82.8% 的 gross energy 来自 loaded-idle**，因此减少 token latency 比降低 active power 更有效。

4. ✅ **专用硬件设计具有明确节能路径**：
   - 若 throughput 提升倍数 > active power 增加倍数，则必然实现更低 energy per token（公式推导见 Eq. 12）。

5. ✅ **跨平台部署一致性好**：
   - 相同 checkpoint 在 FPGA、ARM 上输出一致，logit 和 state hash 完全匹配。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **模型质量略低于对照组** | V2 的 PPL 为 14.75，高于同规模 dense 控制（12.50），programmatic accuracy 下降明显（0.234 vs 0.435），未满足预设质量阈值。 |
| **未实现等质量效率比较** | 所有节能结果基于固定 checkpoint，不能证明“同等性能下更节能”。 |
| **FPGA 测量不含主机开销** | 仅报告 card-level energy，未包含 PCIe、CPU 控制等系统级成本。 |
| **缺乏非贪婪采样支持** | 当前 FPGA 实现仅支持 greedy decoding，不支持 top-k、sampling 等生成策略。 |
| **KV ring 大小固定** | 最大支持 480 tokens，超出后需丢弃旧 context。 |

---

### **未来工作方向**
1. **扩展至 ASIC 设计**：
   - 基于当前 FPGA 原型，开发专用 ASIC 加速器，进一步提升密度与能效。

2. **支持动态稀疏调度与复杂采样**：
   - 实现 full-logit 输出、top-k sampling 等功能，增强实用性。

3. **探索更高粒度的事件编码机制**：
   - 如动态阈值、自适应事件压缩等，进一步提升稀疏性。

4. **构建端到端编译工具链**：
   - 支持自动将稀疏模型映射到 hybrid neuromorphic 硬件。

5. **开展等质量效率对比研究**：
   - 训练一个与 dense 模型性能相当的 V2 变体，进行公平的 energy-efficiency ranking。

---

> 📌 **总结一句话**：  
> **SymbolicLight V2 通过“事件驱动 + 混合计算 + 稀疏执行”，首次在真实硬件上验证了语言模型稀疏推理的巨大节能潜力，为低功耗 LLM 部署提供了可行的技术路线图。**

</details>

---

### 7. [CEDD-optimizer: Enabling Cost-Efficient Dataset Distillation on Geographically Distributed Edge Systems](https://arxiv.org/abs/2609.10151)

**Authors**: Dai Liu, Eishi Arima, Martin Schulz  
**Category**: cs.DC  
**Published**: 2026-09-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.10151v1  

#### Abstract
Centralized learning is a fundamental paradigm in modern AI, in which data are collected from distributed edge devices and aggregated at a central host for model training. However, this pipeline is often bottlenecked by the substantial communication overhead of data collection. Dataset Distillation ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CEDD-optimizer: Enabling Cost-Efficient Dataset Distillation on Geographically Distributed Edge Systems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对在**地理分布不均的边缘系统**（geographically distributed edge systems）中应用 **Dataset Distillation (DD)** 所面临的**成本效率低下**问题。尽管 DD 具有极高的压缩比和良好的训练质量保持能力，但在实际部署中，其超参数（如压缩率、迭代次数）通常被统一设置，忽略了不同边缘节点在**能源价格**和**数据传输费用**上的显著差异，导致总运营成本并非最优。

### 🚀 提出的新方法：CEDD-optimizer
提出了一种名为 **CEDD-optimizer** 的超参数调优框架，旨在在满足下游任务精度要求的前提下，最小化系统的总经济成本（包括计算能耗、数据传输和云存储成本）。

该框架包含两个核心模块：
- **CEDD-calibrator**：通过离线和在线校准，建立能量消耗模型和测试准确率预测模型。
- **CEDD-solver**：基于上述模型，求解一个形式化的优化问题，为每个边缘节点独立配置最优的 DD 超参数。

### 🔍 相比现有方法的优势
- **首次考虑地理价格异质性**：将边缘设备的地理位置相关的电价和网络资费纳入 DD 优化过程，实现真正的“成本感知”调优。
- **非均匀超参数配置**：允许不同边缘节点采用不同的 `DPC`（每类数据量）和 `IC`（迭代次数），以适应本地成本环境。
- **高效且可扩展**：通过统计建模和轻量级三步调优流程，在有限开销内完成全局优化，避免了昂贵的网格搜索或随机搜索。
- **通用性强**：方法不依赖于特定的 DD 算法，适用于大多数基于迭代的 DD 方法（如 DC）。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验在五个标准图像数据集上进行：
- **MNIST**
- **Fashion-MNIST**
- **CIFAR-10**
- **SVHN**
- **ImageNette**（ImageNet 的子集）

这些数据集覆盖了从简单到复杂的不同视觉识别任务。

### ⚙️ 实验设置
- **硬件平台**：使用四种异构边缘设备组合：
  - NVIDIA Jetson Xavier NX
  - NVIDIA Jetson Orin Nano
  - MacBook Pro
  - TQ module TQMx80UC
- **DD 方法**：采用经典的 **Dataset Condensation (DC)** 作为基准 DD 方法。
- **网络架构**：下游训练任务使用 Gidaris 和 Komodakis 设计的卷积神经网络。
- **成本参数来源**：
  - **电价**：来自 [World Population Review](https://worldpopulationreview.com/) 的 144 国家数据（2024）。
  - **网络资费**：基于 AWS Direct Connect 的 SiteLink Rate。
  - **存储成本**：固定为 $0.023/GB（参考 Amazon S3）。
- **评估场景**：涵盖 **IID** 和 **Near-IID** 数据分布。

### 🎯 评估指标
- **总成本（Total Cost）**：包含 DD 能耗成本、数据传输成本、云存储成本和主机训练能耗。
- **测试准确率（Test Accuracy）**：衡量下游深度学习任务的性能。
- **成本节约倍数（Cost Reduction Factor）**：相对于基线方法的成本降低比例。
- **MAPE（Mean Absolute Percentage Error）**：用于评估准确率模型的预测误差。

### 🆚 基线方法对比
- **DD Baseline**：所有节点使用最大超参数 `(α_max, β_max)`。
- **Random Sampling**：随机采样原始数据，使传输数据量与优化后一致。
- **One/Two/Three Step**：CEDD-optimizer 的不同阶段版本。
- **Iterative Search Methods**：
  - Exhaustive Search（穷举搜索）
  - Random Search（随机搜索）
  - Heuristic Search（启发式搜索）
- **Ideal**：使用理想准确率模型的 CEDD-optimizer（理论上限）。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据
- **成本节约效果显著**：
  - 在相同精度约束下，**CEDD-optimizer 最多可实现 20.8× 的成本降低**（相比 DD Baseline）。
  - 三步调优算法（Three Step）平均实现 **1.7×–2.67×** 的成本节约。
- **精度损失极小**：仅比基线方法低 **1%–1.5%**，在可接受范围内。
- **准确率模型精度高**：
  - 测试准确率预测的 **MAPE < 1%**（当边缘节点数 N > 10）。
  - 绝对误差小于 0.5%，验证了统计建模的有效性。
- **能量模型误差低**：预测误差普遍低于 **3%**，适用于真实系统调度。

### 🔁 与基线方法对比结果
| 方法 | 成本表现 | 准确率表现 |
|------|----------|------------|
| **DD Baseline** | 高（基准） | 高 |
| **Random Sampling** | 中等 | 显著低于 DD 方法 |
| **Exhaustive/Heuristic Search** | 极高（搜索开销大） | 可达目标，但成本远高于 CEDD |
| **CEDD-optimizer (Three Step)** | **极低（1.7×–2.67× 优于 Baseline）** | 仅略低于 Baseline（<1.5%） |
| **CEDD-optimizer (Ideal)** | **最低（最多 20.8× 优于 Baseline）** | 接近最优 |

> 💡 **关键发现**：传统搜索方法虽然能逼近最优，但其搜索过程本身成本极高；而 CEDD-optimizer 通过智能建模和有限采样，实现了“低成本找到近似最优”。

### 🔍 消融实验结果
- **非均匀配置优势**：与所有节点使用相同超参数的“Uniform Setup”相比，CEDD-optimizer 始终位于帕累托前沿之上，证明了**非均匀配置的必要性和优越性**。
- **△α 和 △β 的影响**：较小的 `△` 值降低校准开销但损害模型精度；实验确定 `(△α, △β) = (25, 15)` 是最佳权衡点。
- **网络深度影响**：随着网络层数增加（3→5层），CEDD-optimizer 的相对收益略有提升（从 20.8× 到 >22×），表明其在复杂模型中更具价值。
- **求解器选择**：SLSQP 求解器在 **0.006 秒内完成求解**，远快于 TRC、IDE、MINLP，且成本相近，是**性能与效率的最佳平衡**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **地理价格差异不可忽视**：边缘系统的电价和网络资费存在数量级差异，忽略此因素会导致严重的成本浪费。
2. **非均匀超参数配置是关键**：为不同边缘节点定制 DD 超参数（DPC 和 IC）能显著降低成本，同时保持高精度。
3. **CEDD-optimizer 高效且实用**：提出的三步调优流程在极低开销下即可逼近理想性能，适合大规模部署。
4. **统计建模有效**：通过均值、方差等统计量建模准确率函数，既能控制模型复杂度，又能保证预测精度。

### ⚠️ 方法的局限性
- **假设 IID 或近似 IID 数据**：当前准确率模型主要针对 IID 场景设计，对强 Non-IID 数据的泛化能力有待加强（作者在讨论部分提出了加权统计量的改进方向）。
- **静态价格假设**：未考虑电价随时间波动（如峰谷电价），未来可结合时序预测进行动态调度。
- **模型依赖性**：准确率模型需在线校准，若数据特征变化剧烈，可能需要重新训练模型。

### 🔮 未来工作方向
- **支持 Non-IID 数据**：引入加权统计量或局部信息量化机制，提升在异构数据分布下的鲁棒性。
- **动态成本感知**：结合电价预测，优化 DD 的触发时机（timing），进一步降低成本。
- **集成至 Federated Learning**：将 CEDD-optimizer 应用于 FL 场景，优化去中心化更新的生成与传输成本。
- **支持 Continual Learning**：利用历史校准数据加速新任务的模型收敛。
- **探索更高效的求解器**：在更大规模系统中，替换 SLSQP 为更高效的启发式算法以避免瓶颈。

---

> **总结**：  
> CEDD-optimizer 是首个将 **地理价格异质性** 与 **Dataset Distillation** 结合的优化框架。它通过**建模-校准-求解**的闭环流程，实现了在分布式边缘系统中**低成本、高质量**的数据压缩与训练，为构建经济高效的边缘 AI 系统提供了重要实践路径。代码已开源：[https://github.com/NiaLiu/CEDD-optimizer.git](https://github.com/NiaLiu/CEDD-optimizer.git)。

</details>

---

### 8. [Constrained Bayesian Optimization for Hierarchical Federated Learning in IoT Networks for Plant Disease Classification](https://arxiv.org/abs/2609.06830)

**Authors**: Athanasios Papanikolaou, Athanasios Tziouvaras, Apostolos Xenakis, Periklis Chatzimisios, Shameem A. Puthiya Parambath, George Floros, Enrica Zereik, Ivan Petrovic, Fabio Bonsignorio  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.06830v1  

#### Abstract
The deployment of Hierarchical Federated Learning (HFL) in resource-constrained Internet of Things (IoT) environments requires careful configuration to balance predictive performance with energy consumption and execution time. This challenge is particularly relevant to smart agriculture, where distr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Constrained Bayesian Optimization for Hierarchical Federated Learning in IoT Networks for Plant Disease Classification*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
本文针对**资源受限的物联网（IoT）环境**中部署**分层联邦学习（Hierarchical Federated Learning, HFL）**时面临的挑战，提出了一种系统化的方法来优化其配置。具体而言，传统方法在选择 HFL 配置（如深度学习 backbone 架构、聚合策略、通信轮数等）时通常依赖手动调参或穷举搜索，这在实际应用中成本极高，尤其是在每轮评估都需要执行完整训练流程的情况下。

该问题在**智能农业中的植物病害分类任务**中尤为突出，因为边缘设备计算能力弱、能源有限，且对预测精度有明确要求。

### ✅ 提出的新方法与创新思路
作者提出了一个**基于约束贝叶斯优化（Constrained Bayesian Optimization, CBO）的自动化配置框架**，用于高效地搜索最优 HFL 部署方案。主要创新点包括：

- **将 HFL 配置建模为带约束的优化问题**  
  联合考虑模型架构（model）、聚合策略（aggregation strategy）和通信轮数（communication rounds），并引入显式约束条件（能量预算、时间预算、最低准确率）以确保可行性。

- **设计加权目标函数实现多目标权衡**  
  用户可自定义权重 $ \lambda_1, \lambda_2, \lambda_3 $ 来平衡 **energy consumption**、**execution time** 和 **predictive performance**（F1-score），支持灵活的资源配置偏好。

- **融合空间覆盖模型确定联邦规模（federation size）**  
  根据农田面积 $ A_{\text{farm}} $ 和设备感知半径 $ r_a $，结合覆盖率系数 $ \rho $ 自动估算所需参与设备数量 $ N $，使配置更具现实部署意义。

- **采用 Constrained Bayesian Optimization 减少评估开销**  
  利用高斯过程（GP）构建目标函数与约束满足概率的代理模型，并通过约束下的采集函数（acquisition function）指导搜索方向，在极少数评估次数下快速逼近全局最优。

### ✅ 相比现有方法的优势
| 对比维度 | 现有方法（手工/穷举） | 本文方法（CBO） |
|--------|------------------|-------------|
| 评估效率 | 需要遍历全部配置（昂贵） | 仅需约 **11.11%** 的评估即可找到近优解 |
| 可扩展性 | 不适用于大规模搜索空间 | 支持多参数联合优化，易于扩展 |
| 资源敏感性 | 忽视能耗与时延限制 | 显式建模资源与性能约束 |
| 决策灵活性 | 固定偏好 | 支持用户自定义 trade-off 权重 |

---

## 2. **核心实验方法和设置**

### ✅ 数据集与任务
- **任务类型**：Plant Disease Classification（植物病害分类）
- **网络环境**：IoT-based sensor network（模拟资源受限场景）
- **未直接使用公开图像数据集名称**，但基于前期工作 [15][16] 的实验设定进行仿真评估，聚焦于 HFL 配置本身的性能分析而非原始图像处理。

### ✅ 实验设置
#### 搜索空间定义：
$$
\mathcal{X} = \mathcal{M} \times \mathcal{A} \times \{1,\dots,R_{\max}\}
$$
其中：
- **Backbone Models $ \mathcal{M} $**：
  - EfficientNet-B0
  - ResNet-50
  - MobileNetV3-Large
- **Aggregation Strategies $ \mathcal{A} $**：
  - FedAvg
  - FedProx
  - FedAvgM
- **Communication Rounds $ R $**：$ R \in \{1, ..., 30\} $
- 总配置数：$ |\mathcal{X}| = 3 \times 3 \times 30 = 270 $

#### 联邦规模计算：
- 农田面积 $ A_{\text{farm}} = 10000 \, \text{m}^2 $
- 设备有效半径 $ r_a = 20 \, \text{m} $
- 覆盖效率系数 $ \rho = 0.8 $
- 得到 $ N = \left\lceil \frac{A_{\text{farm}}}{\pi r_a^2} \cdot \rho^{-1} \right\rceil = 10 $，与参考实验一致

#### 资源与性能约束：
$$
E_{\text{budget}} = 20 \, \text{Wh}, \quad T_{\text{budget}} = 400 \, \text{s}, \quad F_{1,\text{required}} = 0.80
$$

#### 目标函数（归一化后）：
$$
L(x) = \lambda_1 \hat{E}(x,N) + \lambda_2 \hat{T}(x) + \lambda_3 (1 - F_1(x))
$$
权重设置为：
$$
(\lambda_1, \lambda_2, \lambda_3) = (0.4, 0.2, 0.4)
$$

#### 评估方式：
- 使用历史记录的 round-level 测量值模拟每次配置评估（避免重复运行真实训练）
- 执行 **30 次独立随机种子实验**，验证鲁棒性
- 每次运行初始化 6 个随机配置，总预算为 **30 次评估**

#### 基线对比：
- **Exhaustive Search（穷举搜索）**：作为“黄金标准”获取全局最优解
- 本文方法不与其他黑箱优化算法（如 Random Search、SMAC）对比，但强调相比人工/穷举的巨大效率提升

---

## 3. **主要实验结果和性能指标**

### ✅ 关键性能数据（来自 Table I）

| 指标 | 结果 |
|------|------|
| 搜索空间大小 | 270 |
| 每轮评估次数 | 30 (**占总数 11.11%**) |
| 可行解发现率 | **100%** |
| 达到**精确全局最优**的比例 | **63.33%** |
| 解在**最优值 1% 以内**的比例 | **100%** |
| 平均最优性差距（Mean Optimality Gap） | **0.056%** |
| 中位最优性差距 | **0.000%** |
| 最大最优性差距 | **0.152%** |
| 最佳解首次出现的中位迭代次数 | **15.5**（≈ 5.74% 搜索空间） |

### ✅ 与基线方法对比结果
- **相对于 Exhaustive Search（270 次评估）**：
  - 本方法仅用 **30 次评估**即能稳定找到接近最优的可行配置
  - 所有运行最终结果都在**全局最优的 1% 以内**
  - 平均差距仅为 **0.056%**，表明高度收敛性和稳定性
- 在未能达到最优的案例中，最终解均为**第二优配置**（ResNet-50 + FedProx @ R=2），差距极小（相对差 0.152%）

### ✅ 消融实验（隐含分析）
虽然文中未设专门消融实验，但从以下方面体现方法有效性：
- **约束建模的重要性**：所有运行均成功找到可行解（100% 可行率），说明约束 GP 分类器有效引导搜索进入可行区域
- **采集函数设计的有效性**：中位第 15.5 轮即锁定最佳解，显示 **Expected Improvement under Probability of Feasibility (EI × p(x))** 能高效探索-开发权衡
- **缩放模型合理性**：利用参考实验 $ (N_0=10, R_0=30) $ 推导其他配置下的 energy/time，保证跨配置比较公平

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **Constrained Bayesian Optimization 是一种高效的 HFL 配置工具**  
   在仅探索 **11.11%** 的搜索空间情况下，始终能找到**接近全局最优的可行配置**，显著降低部署成本。

2. **多目标权衡可通过加权目标函数灵活控制**  
   用户可根据应用场景调整 energy/time/performance 的优先级，实现个性化部署决策。

3. **联邦学习配置高度敏感，自动优化至关重要**  
   不同 backbone 与 aggregator 组合表现差异大，且最优通信轮数可能远小于最大值（如 R=4 即达最优），盲目使用默认设置会导致资源浪费或性能不足。

4. **空间部署特性应纳入系统设计闭环**  
   将物理覆盖需求转化为 federation size，增强了配置建议的实际可操作性。

### ⚠️ 方法的局限性
- **依赖高质量的历史测量数据**：当前评估基于已有 round-level 性能日志，若缺乏此类先验数据，则需额外预算用于初始采样。
- **未考虑客户端异构性动态变化**：假设设备资源稳定，未建模网络波动、设备掉线等情况。
- **代理模型简化了复杂依赖关系**：例如未建模 client drift 或 non-IID 数据分布对收敛的影响。
- **未与其他 NAS 或 AutoML 方法对比**：如是否优于 Random Search 或 Hyperband 等轻量级方法尚待验证。

### 🔮 未来工作方向
- 扩展至更大规模的 DNN 架构搜索空间（如 Neural Architecture Search integrated with BO）
- 引入在线反馈机制，支持动态环境下的持续优化（online CBO）
- 结合通信压缩、本地 epoch 数等更多超参数进行联合优化
- 在真实农业 IoT 平台部署并验证端到端性能
- 探索去中心化或 Partial Participation 场景下的适配版本

---

> 📌 **一句话总结**：  
> 本文提出了一种面向资源受限 IoT 场景的 **Constrained Bayesian Optimization 框架**，实现了对 HFL 配置（backbone、aggregator、rounds）的高效自动化调优，在仅评估 **11.11%** 配置的情况下，**100% 运行都能找到距全局最优差距 <1% 的高质量可行解**，为智能农业中低功耗、高性能的分布式植物病害识别提供了实用解决方案。

</details>

---

### 9. [Constitutive State-Space Modeling of Path-Dependent Plasticity: A Resolution-Consistent and Parallelizable Computational Framework](https://arxiv.org/abs/2609.07294)

**Authors**: Rui Barreira, Taylan Soydan, Francesco Scipione, Miguel A. Bessa, Dirk Mohr  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.07294v1  

#### Abstract
Data-driven constitutive models for path-dependent plasticity are commonly formulated using nonlinear recurrent neural networks, whose sequential state evolution limits parallel training and whose predictions may depend on the discretization of the applied strain path. We introduce a Constitutive St...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Constitutive State-Space Modeling of Path-Dependent Plasticity*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统基于**非线性递归神经网络**（如 GRU、LSTM）的数据驱动本构模型在建模路径依赖塑性时存在以下关键缺陷：
- **训练效率低**：由于其序列依赖的递归结构，无法并行处理长加载历史，导致训练时间随序列长度线性增长。
- **分辨率敏感性**（resolution sensitivity）：模型预测对施加应变路径的离散化步长敏感，在不同分辨率下可能导致显著误差，影响其在自适应增量模拟中的鲁棒性。
- **状态演化缺乏物理一致性**：状态更新机制未显式关联应变增量的大小，导致数值行为不够稳定。

### 提出了什么新方法或新思路
本文提出了 **Constitutive State Space (CSS)** 模型，一种面向力学的结构化状态空间（structured state-space）框架，其核心创新在于：
- 将物理应变增量 $\Delta \mathbf{e}$ 分解为**大小**（magnitude $v$）和**方向**（direction $\mathbf{n}$）。
- 利用方向 $\mathbf{n}$ 驱动潜在状态系统的输入，而增量大小 $v$ 被直接嵌入到连续时间线性状态演化的**零阶保持**（zero-order-hold, ZOH）离散化过程中。
- 构建了一个由多个 **S5 blocks** 堆叠而成的可并行扫描的线性递归核心。

### 相比现有方法的优势
- ✅ **高分辨率鲁棒性**：在不同应变路径分辨率下保持稳定的预测精度。
- ✅ **高效并行训练**：利用 S5 的前缀扫描（prefix-scan）结构，实现长序列的并行训练。
- ✅ **强数值一致性**：满足零增量下的应力不变性（stationarity），且状态演化更符合增量力学逻辑。
- ✅ **更高的数据效率**：用更少的训练样本即可达到甚至超越基线模型的精度。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
研究基于四种具有代表性的多轴路径依赖材料生成合成数据集，每种包含 10,000 条随机行走的应变路径（125 步）：
1. **Copper**：各向同性 J2 塑性 + 应变硬化
2. **Dual Phase (DP) Steel**：各向同性 Swift 硬化
3. **Metallic Foam**：Deshpande-Fleck 类泡沫塑性（压力敏感）
4. **Low Carbon (LC) Steel**：结合各向同性和运动硬化（含背应力）

所有数据通过 Abaqus 显式有限元单胞模拟生成，提取中心积分点的 Hencky 应变和 Cauchy 应力。

### 实验设置和评估指标
- **数据划分**：8,000 条用于训练，1,000 验证，1,000 测试。
- **输入输出**：输入为应变增量序列 $\{\Delta \mathbf{e}(1), ..., \Delta \mathbf{e}(125)\}$，输出为对应应力序列 $\{\boldsymbol{\sigma}(1), ..., \boldsymbol{\sigma}(125)\}$。
- **评估指标**：
  - **MSE**（均方误差）
  - **NRMSE**（归一化均方根误差）：$\text{NRMSE}(\sigma_{ij}) = \frac{\text{RMSE}(\sigma_{ij})}{\text{MAV}(\sigma_{ij})}$

### 基线方法对比
主要与当前最先进的 **Minimal State Cell (MSC)** 模型进行系统比较，该模型也是 Mohr 团队先前提出的一种轻量级、具备一定自洽性的 RNN 架构。

此外还进行了超参数独立优化，并在相同参数规模下对比性能。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 材料 | 最佳验证 MSE |
|------|------|-------------|
| **CSS** | Copper | ~$10^{-6}$ |
| **MSC** | Copper | ~$10^{-5}$ |
| **CSS** | DP Steel | ~$10^{-6}$ |
| **MSC** | DP Steel | ~$10^{-5}$ |
| **CSS** | LC Steel | ~$10^{-5}$ |
| **MSC** | LC Steel | ~$10^{-5}$ |
| **CSS / MSC** | Metallic Foam | ~$10^{-5}$ |

> 对于**塑性不可压缩材料**（Cu, DP, LC），CSS 的验证损失比 MSC **低约一个数量级**。

### 与基线方法的对比结果

#### （1）预测精度
- 在所有四种材料上，CSS 达到或超过了 MSC 的预测精度。
- 特别是在铜和双相钢等塑性不可压缩材料上，CSS 表现出显著优势（约 10 倍更低的 MSE）。

#### （2）分辨率鲁棒性（Resolution Robustness）
- **MSC**：当测试分辨率低于训练分辨率时，误差急剧上升（例如从 500 步训练 → 125 步测试，MSE 从 $6\times10^{-6}$ 升至 $2\times10^{-2}$）。
- **CSS**：在所有训练/测试分辨率组合下均保持 MSE ~$10^{-6}$，表现出极强的分辨率不变性。

#### （3）训练效率
- **训练时间缩放**：
  - MSC：近似 $O(n^{0.98})$
  - CSS：仅 $O(n^{0.62})$
- 在 2,000 步长序列上，CSS 训练速度比 MSC **快约 8 倍**（1.5 天 vs 接近 12 天）。
- GPU 加速效果显著：CSS 在 GPU 上训练比 CPU 快两个数量级；MSC 仅快约两倍。

#### （4）数据效率
- CSS 在仅使用 **1 百万应变-应力对**（8,000 × 125）时即达到 MSE ~$1.74\times10^{-6}$。
- MSC 需使用 **8 百万对**（8,000 × 1,000）才能达到最佳 MSE ~$4.39\times10^{-6}$。
- 结论：CSS 实现更高精度的同时，**训练数据需求减少 8 倍**。

### 消融实验结果
- **状态变量维度分析**：CSS 学习到的状态空间有效维度与真实物理模型一致（如铜为 7 维），表明其能自动捕捉最小充分表示。
- **相关性分析**：部分隐藏状态与等效塑性应变 $e^p$ 和等效应力 $\sigma_{\text{eq}}$ 高度相关（Pearson 系数达 0.9），说明学习到了物理意义明确的内部变量。
- **模块设计影响**：增加 S5 block 数量比单纯扩大 latent dimension 更有利于提升性能。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **CSS 是一种力学定制化的状态空间建模范式**，通过将应变增量大小显式引入 ZOH 离散化，实现了物理上更合理的状态演化。
2. ✅ **分辨率鲁棒性是可设计的数值属性**，而非事后补救。CSS 在跨分辨率场景下表现远优于 MSC。
3. ✅ **并行训练显著加速长序列学习**，尤其适用于需要大量历史数据的实际工程问题。
4. ✅ **CSS 具备良好的物理可解释性**：学习到的状态空间维度与真实物理模型匹配，且部分状态与关键内变量高度相关。

### 方法的局限性
- 当前框架仍假设材料行为可通过增量形式描述，尚未直接集成 rate-dependent 或 temperature-dependent 效应。
- 模型部署于实际有限元求解器时仍需逐步步进（不能并行推理），其优势主要体现在**离线训练阶段**。
- 虽然状态有物理意义，但尚无法完全解析所有状态分量的物理解释。

### 未来工作方向
- 扩展至 **rate- and temperature-dependent plasticity**，以应用于制造和冲击问题。
- 探索 **transfer learning** 和 **multi-task learning** 策略，进一步降低新材料建模所需数据量。
- 将 CSS 部署到 **有限元边值问题** 中，评估其在非均匀变形场和自适应步长下的实际表现。
- 推广至 **representative volume element (RVE)** 的代理模型（如晶体塑性、多孔材料均质化），检验其在复杂微观机制建模中的扩展能力。

---

> **总结一句话**：  
> 本文提出的 **CSS 模型** 通过力学定制的结构化状态空间设计，在**精度、鲁棒性、效率和可解释性**三个维度上全面超越了现有的 MSC 等 RNN 架构，为数据驱动本构建模提供了一种高效且可靠的新型计算范式。

</details>

---

### 10. [Do Dynamic Routers Need Memory? HeRo: History-Aware Routing for Efficient LLM Inference](https://arxiv.org/abs/2609.08189)

**Authors**: Hongjin Lin, Wentao Wan, Keze Wang  
**Category**: cs.AI  
**Published**: 2026-09-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.08189v1  

#### Abstract
Dynamic layer routing reduces the inference cost of Large Language Models (LLMs) by learning to skip layers for individual tokens. Existing methods, however, treat each routing decision as a local operation conditioned solely on the current hidden state which is a formulation that overlooks the sequ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Do Dynamic Routers Need Memory? HeRo: History-Aware Routing for Efficient LLM Inference*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **dynamic routing** 方法在 LLM 推理过程中通常将每一层的路由决策视为仅依赖当前隐藏状态 $ h^{(l)} $ 的局部操作。然而，这种设定忽略了路由过程的本质是**路径依赖的序列决策过程**：  
- 前序层的路由选择决定了残差流（residual stream）的变化；
- 所有层的路由分数共同受到一个联合目标（如参数使用率损失）的正则化。

因此，仅基于当前隐藏状态进行决策会导致信息不完整——虽然 $ h^{(l)} $ 隐含了历史影响，但它并未显式记录先前的路由行为和残差更新。

### 提出的新方法：History-Aware Routing (HeRo)
作者提出 **HeRo**，一种引入**路由器记忆机制**（router memory）的动态路由框架，以显式维护跨模型深度的路由历史状态。

#### 核心创新点：
- **显式的路由记忆机制**：通过 **linear attention** 在 depth 维度上增量聚合前序层的路由得分及其引发的残差变化，构建紧凑的历史表示。
- **联合条件决策**：每层的路由器同时依赖于：
  - 当前 token 的隐藏状态 $ h^{(l)} $
  - 路由器内存中的历史上下文 $ c^{(l)} $
- **轻量级适配设计**：仅训练轻量级的路由器、adapter 和 memory 模块，**主干网络（backbone）保持冻结**，无需修改预训练参数。
- **实例化为 FFN 路由**：在每个 routed layer 中选择执行原始 FFN 或一个 bottleneck adapter，注意力子层始终保持激活。

### 相比现有方法的优势
| 方面 | 传统方法 | HeRo |
|------|--------|------|
| 决策依据 | 仅当前隐藏状态 $ h^{(l)} $ | 当前状态 + 显式路由历史 |
| 历史建模 | 隐式（通过 $ h^{(l)} $ 间接推断） | 显式（memory 存储并传递） |
| 参数效率 | 多数需微调整个模块 | 仅训练轻量组件，backbone 冻结 |
| 动态协调能力 | 各层独立决策 | 层间通过 memory 协同 |

> ✅ **优势总结**：HeRo 更好地捕捉了路由过程中的路径依赖性，提升了决策准确性与适应性，尤其在复杂推理任务中表现更优。

---

## 2. 核心实验方法和设置

### 使用的数据集
主实验采用 **7个标准 NLP 基准测试集**，涵盖多种任务类型：
- **OpenBookQA**（科学问答）
- **ARC-Easy / ARC-Challenge**（常识推理）
- **PIQA**（物理常识）
- **BoolQ**（段落问答）
- **WinoGrande**（共指消解）
- **HellaSwag**（合理续写判断）

额外消融实验还包含：
- **GSM8K**（数学多步推理，使用 chain-of-thought 生成后 exact match 评估）
- **HumanEval**（代码生成，pass@1 指标）

所有评测均使用 `lm-evaluation-harness` 框架统一执行。

### 实验设置
- **模型主干**：
  - 主要：**Meta-Llama-3.1-8B-Instruct**
  - 对比：**Llama-2-7B**, **Llama-2-13B**
- **路由配置**：
  - 在选定的 $ L_R $ 个 block 上进行 FFN 路由
  - 可选分支：原生 FFN vs. bottleneck adapter ($ d_r = 896 $)
  - 注意力始终保留
- **Memory 实现**：
  - 使用 kernelized linear attention
  - Memory dimension: 64
  - Query/Key/Value 投影维度：64
  - Forget gate $ \mu_j = \sigma(m_j) $，初始化使遗忘因子 ~0.98

### 评估指标
- **Param skip (%)**：跳过的参数比例（平均到 token）
- **Retain (%)**：相对于 dense 模型的平均任务得分保留率
- **Acc / acc_norm**：各任务准确率（部分归一化）

### 基线方法对比
分为两类：
#### 静态剪枝（Static Pruning）：
- ShortGPT, Shortened-PPL/Taylor, SliceGPT, LLM-Pruner, LaCo

#### 动态路由（Dynamic Routing）：
- MoD-D, D-LLM, SkipGPT-Joint / RT, SkipGPT-RT

> 所有方法按 **目标参数跳过预算（25% 和 40%）** 分组比较，报告实际实现的 skip rate 和 performance retain。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 **Llama-3.1-8B** 上的表现：

| 方法 | Param skip | Retain |
|------|------------|--------|
| Dense（基准） | 0% | 100.00% |
| **HeRo ($\alpha_t=10^{-3}$)** | **26.87%** | **100.24%** ✅ |
| SkipGPT-RT | 25.5% | 94.14% |
| LaCo | 24.5% | 72.38% |

👉 **结论**：HeRo 在跳过近 **27% 参数**的同时，**性能超过 dense 模型本身**（100.24%），显著优于所有基线。

#### 更严格预算下（~39% skip）：

| 方法 | Param skip | Retain |
|------|------------|--------|
| **HeRo ($\alpha=10^{-3}$)** | **38.82%** | **97.01%** ✅ |
| SkipGPT-RT | 40.20% | 81.15% |

👉 在更高压缩率下，HeRo 仍能保持接近原始性能，而其他方法大幅下降。

#### 其他主干上的表现一致性：
- **Llama-2-7B**：27.87% skip → 94.38% retain
- **Llama-2-13B**：28.38% skip → 94.49% retain  
→ 表明 HeRo 具有良好的跨规模泛化能力。

### 与基线方法的对比结果
- 在所有三个 backbone 和两个 budget 设置下，**HeRo 均取得最高的 Retain 分数**。
- 在 25% 预算组中，HeRo 是唯一达到甚至略微超越 dense 性能的方法。
- 在 40% 预算下，其领先优势进一步扩大，尤其是在推理密集型任务上。

### 消融实验结果（Table 3）

| 消融变体 | Param skip | Retain |
|---------|-------------|--------|
| **HeRo（完整）** | 26.87% | 100.00% ✅ |
| w/o History | 26.41% | 97.35% ❌ |
| w/o MemRead | 26.28% | 98.02% ❌ |
| w/o Aux State | 27.02% | 97.32% ❌ |
| w/o Pos State | 26.35% | 97.57% ❌ |

#### 关键发现：
- 移除任何路径状态组件都会导致性能下降，说明各部分协同作用重要。
- “w/o History” 下降最明显，验证了**显式历史建模的关键性**。
- 特别是在 **GSM8K（数学推理）** 和 **HumanEval（代码生成）** 上性能损失最大，表明这些多步任务更依赖历史信息协调。

### 控制变量实验
- **Table 2** 显示调节 gate penalty 系数 $ \alpha_t $ 可精确控制参数跳过率（从 13.19% 到 26.87%），且性能波动极小（±1.73%），说明 HeRo 具备良好可控性。
- **Table 4** 测试 memory dimension 影响，发现从 32 到 256 均性能稳定，表明该机制对超参不敏感。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **动态路由器确实需要记忆**：  
   尽管当前隐藏状态隐含历史信息，但轻量级路由器难以从中可靠恢复完整的路由路径。显式维护路由历史可显著提升决策质量。

2. ✅ **HeRo 实现高效且高性能的推理加速**：  
   在 Llama-3.1-8B 上跳过 **26.87% 参数**时仍保持 **100.24% 的 dense 性能**；在更紧预算下（38.82% skip）仍保留 **97.01% 性能**。

3. ✅ **历史信息对复杂任务尤为重要**：  
   消融实验证明，在 **multistep reasoning**（如 GSM8K）和 **code generation**（如 HumanEval）任务中，移除历史建模带来的性能损失最为显著。

4. ✅ **模块设计有效且鲁棒**：  
   路径特征编码、memory aggregation、history head 三者构成互补流程，共同支持更精准的动态路由。

### 方法的局限性
- **仅应用于 FFN 路由**：未探索 attention 模块或其他结构的联合跳过。
- **memory 开销虽小但仍存在**：需维护 $ d_m \times d_m $ 的 memory matrix（文中为 64×64），可能限制极端低延迟场景应用。
- **依赖预定义候选层集合**：目前路由范围固定，尚未实现完全自适应的 layer selection。

### 未来工作方向
- 扩展至 **attention routing** 或 **混合专家（MoE）架构** 中的历史感知路由。
- 探索 **跨 token 的 memory 共享机制**，降低内存开销。
- 引入 **learnable routing horizon prediction**，动态决定应跳过的层数。
- 结合 **speculative decoding** 或 **early exiting** 构建多层次自适应推理系统。

---

> 🔚 **总结一句话**：  
> **HeRo 通过引入轻量级的 router memory 机制，首次实现了“历史感知”的动态路由，在不改动主干的前提下大幅提升了 LLM 推理效率与性能平衡，验证了“记忆”对于动态路由器的重要性。**

</details>

---

### 11. [Inference-Time Graph Engineering for Multi-Agent LLM Workflows](https://arxiv.org/abs/2609.05774)

**Authors**: Katherine Tieu, Dongqi Fu, Yinglong Xia, Hong Li, Hong Yan, Jingrui He  
**Category**: cs.AI  
**Published**: 2026-09-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.05774v1  

#### Abstract
Recent multi-agent LLM systems increasingly rely on graph-structured communication to coordinate specialized agents. We revisit multi-agent orchestration from a graph-engineering perspective: rather than optimizing a static topology, we synthesize a task-conditioned temporal workflow graph that join...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Inference-Time Graph Engineering for Multi-Agent LLM Workflows**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **multi-agent LLM** 系统通常依赖于固定的或通过学习优化的通信拓扑（如链式、星型、树状等），这些方法主要关注“谁可以与谁通信”（即图的连通性），而忽略了更深层次的协作语义，例如：
- 在哪个推理阶段进行通信？
- 通信的具体目的是什么？（如提供代码、验证逻辑、提出策略）
- 如何动态调整协作流程以适应不同任务？

这种“topology-centric”的设计导致协作协议隐式且缺乏可解释性，限制了系统的灵活性和性能。

---

### **提出了什么新方法或新思路**
本文提出 **ReActNet**，一种无需训练的 **inference-time graph engineering** 框架，其核心思想是将 multi-agent 协作建模为一个**可执行的工作流图（executable workflow graph）**，而非静态拓扑。

#### **关键创新点：**
- **Compile-then-Execute 架构**  
  将图构建（compilation）与执行（execution）分离。在推理时，由一个 **LLM-based controller** 作为“图编译器”，一次性生成整个任务条件化的多轮通信计划。
  
- **Instruction-Typed Temporal Workflow Graph**  
  每条有向边不仅表示通信路径 `(j → i)`，还携带一条自然语言指令 `instruction`，明确指定源 agent 应向目标 agent 传递何种推理内容（如“提供反例”、“检查第4步计算”）。这使得通信具有语义意义。

- **Phase-Aware 推理流程**  
  图结构随时间变化（temporal snapshots），每个时间步对应一个推理阶段（如分解 → 验证 → 合成），实现对复杂任务的阶段性控制。

- **Training-Free 设计**  
  不需要任何梯度优化或强化学习来学习图结构，完全在推理时通过 prompt 实现，显著降低部署成本。

---

### **相比现有方法的优势**
| 维度 | 传统方法（如 Chain, Star, GPTSwarm） | ReActNet |
|------|----------------------------------------|---------|
| **图结构** | 固定或学习得到，输入无关 | 动态生成，task-conditioned |
| **通信语义** | 隐式（仅表示可能通信） | 显式（每条边带 instruction） |
| **训练需求** | 多数需训练/优化图结构 | 完全 training-free |
| **可解释性** | 低 | 高（完整记录谁在何时为何通信） |
| **效率** | 可能因迭代优化带来高开销 | 一次性编译，执行高效 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖三大类任务，共6个基准：

| 类别 | 数据集 | 任务类型 | 测试样本数 |
|------|--------|----------|-----------|
| **General Reasoning** | MMLU | 多选问答 | 153 |
| **Mathematical Reasoning** | GSM8K, MultiArith, SVAMP, AQuA | 数学应用题求解 | 1,319 / 600 / 1,000 / 254 |
| **Code Generation** | HumanEval | 编程生成 | 164 |
| **Real-World Assistant Task** | GAIA (validation split) | 多工具、多跳推理 | — |

---

### **实验设置和评估指标**
- **模型配置**：所有 agent 使用 `gpt-4o`，controller 和 synthesizer 也使用 `gpt-4o`；部分实验使用 `gpt-3.5-turbo`, `Llama-3.3-70B-Instruct`, `Claude Sonnet 4` 进行泛化性测试。
- **交互轮次**：默认 $ T = 3 $
- **Agent 数量**：$ N = 5 $
- **评估指标**：
  - MMLU, GSM8K, MultiArith, SVAMP, AQuA：**accuracy**
  - HumanEval：**pass@1**
  - GAIA：**accuracy**
- **效率指标**：inference time, token consumption, cost (USD)

---

### **基线方法对比**
涵盖单 agent 与 multi-agent 范式：

#### **Single-Agent Baselines**
- Vanilla
- CoT (Chain-of-Thought)
- ComplexCoT
- Self-Consistency (SC)
- PHP

#### **Multi-Agent Baselines**
- **Predefined Topologies**: Chain, Star, Tree, Complete Graph, Random Graph
- **Advanced Frameworks**:
  - AutoGen
  - MetaGPT
  - LLM-Debate
  - LLM-Blender
  - DyLAN
  - GPTSwarm
  - G-Designer （learned topology via GNN）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| 方法 | MMLU ↑ | GSM8K ↑ | MultiArith ↑ | SVAMP ↑ | AQuA ↑ | HumanEval ↑ | Avg. ↑ |
|------|--------|--------|-------------|--------|-------|------------|--------|
| **ReActNet (ours)** | **86.93** | **94.45** | **100.00** | **94.67** | **87.70** | **92.74** | **92.75** |
| G-Designer | 83.98 | 95.07 | 98.30 | 91.85 | 79.47 | 89.90 | 89.84 |
| Complete Graph | 83.15 | 86.49 | 97.20 | 89.48 | 79.21 | 83.75 | 86.55 |

> ✅ ReActNet 在 **5/6 个数据集上取得 SOTA**，平均得分领先第二名 **2.91 pts**

---

### **与基线方法的对比结果**
- 在 **数学推理** 上全面超越 topology learning 方法（如 G-Designer），尤其在 MultiArith 达到完美准确率。
- 在 **编程任务** 上大幅领先，pass@1 达 **92.74**，远超 G-Designer 的 89.90。
- 在 **GAIA** 上表现最佳（**12.72% accuracy**），优于所有需训练的方法（如 G-Designer: 10.91%, GPTSwarm: 11.52%），凸显其 zero-training adaptation 能力。

---

### **消融实验与参数分析**

#### **(1) 交互轮次 $ T $ 的影响（Table 5）**
| 方法 | pass@1 |
|------|--------|
| ReActNet ($T=1$) | 90.32 |
| ReActNet ($T=3$) | **92.74** ✅ |
| ReActNet ($T=5$) | 89.52 ↓ |

> 最优值出现在 $T=3$，过多轮次引入冗余信息导致性能下降。

#### **(2) 不同 LLM backbone 的兼容性（Table 6）**
| Backbone | 方法 | MMLU | GSM8K |
|---------|------|------|-------|
| Llama-3.3-70B-Instruct | G-Designer | 85.62 | 95.87 |
| | ReActNet | **88.89 (+3.27)** | 95.34 (-0.53) |
| Claude Sonnet 4 | G-Designer | 84.31 | 96.63 |
| | ReActNet | **93.46 (+9.15)** | **95.86 (-0.77)** |

> ReActNet 在不同模型家族中均表现出强竞争力，尤其在 MMLU 上提升显著。

#### **(3) 效率对比（Table 2 & 3）**
- **无训练开销**：ReActNet 训练 token 为 0，总 token 消耗最低。
- **推理速度快**：在 GSM8K 上仅需 **1.7h**，低于 GPTSwarm (2.8h) 和 DyLAN (4.6h)。
- **可扩展性强**：当 agent 数从 5 增至 20，ReActNet 的推理时间几乎不变（~7 min），而其他方法呈指数增长。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Effective multi-agent coordination 不仅取决于“谁通信”，更在于“如何工程化通信流程”**  
   显式地设计带有 instruction 的 temporal workflow graph 比单纯优化 topology 更有效。

2. **推理过程应分阶段进行（phase-aware）**  
   分析显示 ReActNet 自动形成：
   - **Round 1**: 问题分解与信息收集
   - **Round 2**: 计算、交叉验证、工具调用
   - **Round 3**: 一致性检查与答案合成

3. **无需训练即可实现 superior coordination**  
   利用 LLM 的规划能力，在 inference time 一次性生成全局一致的协作协议，避免了复杂的 topology learning 和 credit assignment 问题。

4. **动态图结构优于固定模式**  
   ReActNet 根据任务类型调整通信模式（如数学题优先连接 mathematician 和 inspector），并支持反馈回路（feedback loop）。

---

### **方法的局限性**
- **依赖 controller 的规划能力**：若 controller 无法正确理解任务，可能导致错误的 workflow。
- **一次性编译不可修正**：当前版本不支持运行时 replanning，若中间状态出错难以纠正。
- **边缘指令歧义风险**：自然语言 instruction 可能存在模糊性，影响 agent 执行精度。
- **未探索更大规模 agent 群体**：实验最多使用 20 个 agent，实际场景中可能面临协调瓶颈。

---

### **未来工作方向**
- **Online Graph Replanning**：允许 controller 根据中间状态动态调整后续 workflow。
- **Richer Tool-Using Nodes**：集成外部工具（search, code interpreter, database）作为 workflow 节点。
- **Human-in-the-Loop Approval Edges**：引入人工审核节点，增强可靠性。
- **Formal Verification of Workflow Graphs**：对生成的 graph 进行安全性、一致性形式化分析。
- **跨任务迁移的轻量化微调**：虽主打 training-free，也可探索少量 fine-tuning 提升 controller 规划质量。

---

> 🔚 **总结**：ReActNet 提出了一种全新的 **graph-engineering** 视角，将 multi-agent LLM 协作从“拓扑优化”升级为“可执行流程工程”。其实验表明，**显式、任务条件化、带语义指令的 temporal workflow graph** 是提升 multi-agent 系统性能的关键，同时保持高效与可解释性，为下一代 agentic systems 提供了重要范式。

</details>

---

### 12. [When and Why LLM Causal Priors Help: Closed-Loop Prior Selection for Amortized Causal Inference](https://arxiv.org/abs/2609.06941)

**Authors**: Haohao Zhou  
**Category**: cs.AI  
**Published**: 2026-09-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.06941v1  

#### Abstract
Causal effect estimation asks how an outcome would change under an intervention, and medicine, economics, and public policy all treat it as a foundational task. Prior-data fitted networks (PFNs) amortize the task: a model trained on large numbers of programmatically generated synthetic causal tasks ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When and Why LLM Causal Priors Help: Closed-Loop Prior Selection for Amortized Causal Inference*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Prior-data fitted networks (PFNs)** 的 **amortized causal inference** 模型（如 Do-PFN、CausalPFN）的能力高度依赖于其训练时使用的合成任务先验（synthetic training prior）。然而，这些先验目前是**手动设计**的，受限于设计者的领域知识，成为模型扩展能力的瓶颈。

尽管 **Large Language Models (LLMs)** 能够“绘制”出特定领域的因果图（causal graphs），并可能作为先验来源，但以下问题尚未系统解决：
- 注入 LLM 提取的因果先验是否真的有帮助？
- 收益来自图的**语义内容**还是仅仅是分布的**结构多样性**？
- 如何在多个候选图中自动选择最优注入源？

此前实践依赖**手动试错**，缺乏系统性和可验证性。

### 提出的新方法与创新
本文提出一个 **closed-loop prior selection framework**，将“选择哪个 prior 注入”这一决策转化为一个**预算约束下的优化问题**，实现自动化、可验证的选择流程。

#### 核心创新点：
- **C1 (框架创新)**：将 prior injection 视为 **budget-constrained optimization**（公式 (3)），采用两阶段闭环选择机制：
  1. **廉价预筛选（Cheap Post-Training）**：对候选 prior 进行短步数（5k steps）微调，通过复合评分 `Score(q)` 排序。
  2. **全量验证（Full Validation）**：仅对排名靠前的候选进行完整训练（20k steps）和配对统计检验，最终确定胜者。
- **C2 (实证突破)**：在 7.34M 参数的 Do-PFN 上，该框架选出的最佳配置实现了 **2.75× 的显著增益**（p=0.0086），且误差低于未注入的官方 base。
- **C3 (机制解释)**：通过机制实验提供证据链支持 “**content beats diversity**” —— 收益源于图的**语义内容**而非结构多样性。
- **C4 (适用边界)**：通过九个控制实验归纳出 **三条件经验规律（empirical regularity）**，明确 prior injection 有效的前提条件。
- **C5 (外部有效性)**：在标准基准上与 SOTA 和经典估计器进行同协议比较，揭示方法优势与边界。

#### 相比现有方法的优势
- 将 prior selection 从**主观试错**变为**可量化、可复现的优化过程**。
- 不改变模型架构或训练算法，仅优化 prior 来源，具有高兼容性。
- 明确界定方法有效性的边界，避免盲目应用。

---

## 2. 核心实验方法和设置

### 数据集
- **主评估域（Primary Domain）**：`law_race` —— 一个带有真实因果图的合成因果基准。
- **相邻监控域（Adjacent Monitoring Domain）**：`sales` —— 用于检测副作用（collateral damage）。
- **标准因果基准（Standard Benchmarks）**：
  - **IHDP**：半合成数据，个体处理效应（PEHE）为指标。
  - **Lalonde**：真实观测数据，以随机试验估计值为参考，ATE 相对偏差为指标。

### 实验设置
- **模型**：基于 **Do-PFN**（7.34M 参数）。
- **基础版本**：
  - 官方发布的 base（强但对继续训练不敏感）。
  - 本地重训的 `v2` base（较弱但可塑性强，所有 post-training 均从此开始）。
- **候选先验池（Candidate Pool）**：
  - 来源维度：V4（主源）、9B LLM、AESD（替代配置）。
  - 温度维度：解码温度 `t ∈ {0.3, 0.7, 1.5}`。
  - 对照：`q_oracle`（真实图）、`q_rand`（语义打乱的随机图）。
- **注入方式**：继续训练（post-training）在由 LLM 提取图编译生成的任务分布上。

### 评估指标
- **主指标**：**Normalized Mean Squared Error (NMSE)** 在 `law_race` 上。
- **辅助指标**：
  - `sales` 域的 NMSE（监控泛化）。
  - 诊断面板（diagnostic panel）上的拟合与泛化。
  - 能力保留（无遗忘）。
- **复合评分 `Score(q)`**：
  ```math
  \text{Score}(q) = \sum_{i=1}^4 w_i m_i(q),\quad w = (0.50, 0.25, 0.15, 0.10)
  ```
  权重偏向 real-domain generalization（占 50%）。

### 基线方法对比
- **直接 SOTA**：CausalPFN。
- **经典估计器**：
  - S-/T-/X-/DR-learner
  - CausalForestDML
  - Naive ATE（常数效应基线）
- **未注入的官方 base** 作为主要参照。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 配置 | `law_race` NMSE | 相对增益 |
|------|------------------|----------|
| 官方 base（未注入） | 0.0308 ± 0.0083 | — |
| MixA baseline（v2 base） | 0.0580 ± 0.0143 | — |
| **Winner (v4t15 + mixA)** | **0.0211 ± 0.0048** | **2.75× 增益** |

- **统计显著性**：配对 t-test，`p = 0.0086`（n=5），方向一致（5/5）。
- **误差水平**：优于官方 base（描述性跨谱系比较）。
- **相邻域表现**：`sales` 域 NMSE 从 0.389 → 0.356，`p = 0.0109`，**同步提升**，无能力退化。

### 与基线方法对比（M8）
在 `law_race` 上：
- **显著优于所有机器学习基线**，包括 CausalPFN（0.1276），领先 **4.1×–6.1×**（p ≤ 0.006）。
- 仅与 naive ATE（0.0278）无显著差异，提示该基准个体异质性有限。

在 `sales` 上：
- 所有 amortized 模型表现不佳，简单 meta-learners（如 DR-learner, Naive ATE）更优。
- 表明 amortized 方法在小样本、近常数效应场景下不具优势。

### 消融实验结果（M1–M5）
| 实验 | 关键发现 |
|------|--------|
| **M1** | 增益约 **72% 来自分布窄化**（narrowing），**28% 来自语义内容**（content）；LLM 图 ≈ 真实图，优于随机图。 |
| **M2** | 观测模式混合比例存在**非单调内部最优**（50/30/20 最佳）；但 `sales` 域随 confounded share 单调恶化。 |
| **M3** | **多源集成（ensembling）有害**：三源平均导致严重退化（Δ=+0.111），跨种子方差上升 4.7×，表明错误正相关。 |
| **M5** | **温度呈 V 形响应**：中温（T=0.7）因边丢失而退化；高温（T=1.5）最佳。随机图（同结构多样性）表现极差，**拒绝“纯多样性驱动增益”假设**。 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 因果先验确实有帮助**，但前提是：
   - 基础模型在目标任务域上**欠拟合**（underfit）。
   - 注入的 prior **领域匹配**任务域。
   - 任务本身在 base 训练 prior 的**支持范围内**（support）。
   
   > **三条件经验规律**：只有当这三个条件同时满足时，注入才带来显著增益。

2. 🧠 **收益源于语义内容，而非结构多样性**：
   - 随机图虽具相似结构复杂度，但表现远差。
   - 温度实验显示中温退化（保守输出导致边丢失），说明**正确边结构至关重要**。

3. ⚠️ **收益有明确边界**：
   - 若 base 已很强（如官方权重），注入反而导致 **4–9× 退化**，排名完全反转（M6）。
   - 跨域注入（如 law_race prior → IHDP）导致**负迁移**。
   - 即使领域匹配，若任务维度远超 prior 支持范围（如 IHDP 的 26 维 vs [1,6]），仍无法修复（M9 stage 2）。

4. 🔍 **合成诊断指标不可靠**：
   - 在 4/5 实验中，**合成诊断（probe）与真实泛化（transfer）结果相反**（见 Figure 11）。
   - 例如：诊断认为差的配置，实际泛化更好（M1, M4 round-1）。
   - 支持 **Theorem 3.2**：若 probe 与 eval 分布结构不一致，排序可能完全反转。

### 方法的局限性
- 当前结论基于单一 base family（Do-PFN）和两个内部域。
- “三条件”每条仅有一个实例支持，**必要性未被完全验证**。
- 条件二（领域匹配）与三（支持内）在 M9 中耦合，**无法分离归因**。
- 未超越原生强 base，**不适用于已充分训练的模型**。
- 在标准基准（IHDP, Lalonde）上表现不佳，**不具备通用 SOTA 性能**。

### 未来工作方向（Outlook）
1. **扩展 prior 支持范围**：重新训练 base 在更高维 prior 上，以突破维度边界。
2. **开发 prior-base 匹配检查器**：自动判断三条件是否满足，指导是否注入。
3. **针对欠拟合 base 的定向修复**：将本框架应用于其他 underfit 场景。
4. **探索更鲁棒的集成策略**：避免多源平均导致的质量稀释。

---

> **总结**：本文将 LLM 因果先验的使用从“手工试错”转变为“可验证的优化选择”，提出了 **closed-loop prior selection framework**，并在实证上验证了其有效性。核心洞见是：**prior injection 的成功高度依赖于 base 状态、领域匹配和支持覆盖**，且收益源于**语义内容**而非形式多样性。这为未来构建领域专用的因果推理系统提供了可操作的准则与边界认知。

</details>

---

### 13. [ProbPlug: A Plugin Uncertainty Network for Reliable Confidence in LLM Binary Classification](https://arxiv.org/abs/2609.10122)

**Authors**: Jianzong Wang, Chuhang Liu, Botao Zhao, Zuheng Kang, Xulong Zhang, Xiaoyang Qu, Junqing Peng, Zhiewei Ye, Yayun He  
**Category**: cs.CL  
**Published**: 2026-09-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.10122v1  

#### Abstract
Large language models (LLMs) have achieved strong performance across a broad range of classification settings, yet the reliability of their predictions remains a major obstacle to deployment in high-stakes scenarios. Although confidence estimation for LLMs has been widely studied, confidence calibra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ProbPlug: A Plugin Uncertainty Network for Reliable Confidence in LLM Binary Classification**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
大型语言模型（LLMs）在各类分类任务中表现出色，但其预测的**置信度（confidence）往往不可靠且未校准（uncalibrated）**，这严重限制了其在高风险场景（如医疗、金融等）中的部署。尽管已有多种 confidence estimation 方法被提出，但在 LLM-based 分类任务上的**可靠性与泛化能力仍不足**。

本文聚焦于 **LLM-based binary classification 中的 confidence calibration 问题**，旨在提供一种轻量级、即插即用（plug-and-play）、可跨任务迁移的解决方案。

---

### 🚀 提出的新方法：ProbPlug
作者提出了 **ProbPlug** —— 一个用于 LLM 二元分类任务的**插件式不确定性网络（plugin uncertainty network）**，其核心思想如下：

- 利用冻结的 LLM 内部各 Transformer 层生成的 **hidden representations** 来估计输出是否正确。
- 不修改原始 LLM 结构，仅训练一个外部轻量级网络作为 confidence estimator。
- 框架由三部分组成：
  1. **Token Compression (TC)**：对最终生成 token 的多层隐藏状态进行非线性压缩。
  2. **Layer Information Aggregation (LIA)**：采用 multi-head attention 自适应融合不同层的信息。
  3. **Classification Head**：输出该预测正确的概率 $ P(Y=1|Z) $。

> 🔍 **灵感来源**：受神经科学启发，人类大脑决策过程中信号在不同脑区间流动，类似地，LLM 的决策过程也应反映在各层表示中。

---

### ⚖️ 相比现有方法的优势

| 特性 | ProbPlug | 其他方法（如 Verbalization, Self-Consistency 等） |
|------|---------|---------------------------------------------|
| 是否需修改 LLM | ❌ 否（冻结原模型） | 部分需要微调或重训 |
| 推理效率 | ✅ 单次推理即可 | 多数需多次采样（如 Self-Consistency） |
| 泛化能力 | ✅ 支持跨任务（cross-task）迁移 | 多为任务特定设计 |
| 校准性能 | ✅ 更优的 ECE 和 Brier Score | 普遍存在过自信（overconfidence）现象 |
| 扩展性 | ✅ 可扩展至 multi-class 与 multimodal 场景 | 多局限于文本二分类 |

> 💡 ProbPlug 在保持 LLM 强大泛化能力的同时，引入了传统分类器才具备的良好校准特性。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集

#### 文本分类任务（binary/multi-class）：
- **SMS-SPAM** [11]：短信垃圾信息检测
- **SST-2** [12]：电影评论情感分析（正面/负面）
- **Toxic Comment Classification** [13]：有毒言论识别
- **Civil Comments** [13]：文明评论分类
- **Amazon Polarity** [14]：商品评论极性判断

#### 多模态任务：
- **IEMOCAP** [16]：语音情绪识别（speech emotion recognition），使用 **Qwen2-Audio** 作为 backbone

> 所有任务均通过 prompt 转换为 “Yes” / “No” 二元输出形式。

---

### 🧪 实验设置与评估指标

| 类别 | 内容 |
|------|------|
| **Backbone 模型** | Qwen3-8B（文本）、Qwen2-Audio（语音） |
| **训练方式** | 冻结 LLM，仅训练 ProbPlug 插件模块 |
| **输入特征** | 最终生成 token 在所有 Transformer 层的 hidden states |
| **评估模式** | - In-task：在训练集同分布下测试<br>- Cross-task：训练于 SMS-SPAM，直接迁移到其他任务（无微调） |
| **主要指标** | - **F1-score**<br>- **AUPRC**（Area Under Precision-Recall Curve）<br>- **ECE**（Expected Calibration Error）<br>- **Brier Score** |

---

### 🔁 基线方法对比

| 方法 | 类型 | 简介 |
|------|------|------|
| **Verbalization** [3] | Prompt-based | 让 LLM 直接输出 confidence 语句（如“我有80%把握”） |
| **Logit** [4] | Output-based | 使用生成 token 的 logits 或概率作为置信度 |
| **Self-Consistency** [6] | Sampling-based | 多次采样取一致率作为 confidence |
| **CISC** [9] | Hybrid | 结合 self-consistency 与 confidence prompt |
| **SAPLMA** [8] | Probe-based | 在内部表示上附加 MLP 分类头并训练 |

---

## 3. **主要实验结果和性能指标**

### 📊 主要性能对比（见 Table 1 & 2）

| 方法 | SMS-SPAM (F1/AUPRC) | SST-2 (F1/AUPRC) | Amazon-Polarity (F1/AUPRC) |
|------|---------------------|------------------|----------------------------|
| Qwen3-8B | 88.49 / – | 88.75 / – | 94.42 / – |
| Verbalization | 87.66 / 82.48 | 90.10 / 86.27 | 88.96 / 85.54 |
| Logit | 88.64 / 92.02 | 89.44 / 95.28 | 94.52 / 97.06 |
| Self-Consist | 88.44 / 70.13 | 89.55 / 87.18 | 94.55 / 94.16 |
| CISC | 86.86 / 68.80 | 91.12 / 93.49 | 81.50 / 50.64 |
| SAPLMA | 92.05 / 95.85 | 87.89 / 96.85 | 94.46 / 97.65 |
| **ProbPlug (Ours)** | **94.78 / 97.56** | **88.47 / 97.42** | **94.72 / 98.72** |

> ✅ ProbPlug 在多数任务中取得最优的 **F1-score 和 AUPRC**，尤其在 AUPRC 上显著领先，说明其 confidence 更可靠，便于阈值调节。

---

### 🎯 多模态任务表现（IEMOCAP，见 Table 3）

| 方法 | UA (%) | WA (%) | F1-score (%) |
|------|--------|--------|--------------|
| Whisper large v3 | 73.54 | 72.86 | 73.11 |
| Emotion2Vec large | 70.70 | 63.30 | – |
| Qwen2-Audio | 64.33 | 60.37 | 61.61 |
| CISC | 67.78 | 67.43 | 63.62 |
| SAPLMA | 73.15 | 74.61 | 73.53 |
| **ProbPlug (Ours)** | **77.33** | **77.32** | **77.76** |

> ✅ ProbPlug 在语音情绪识别任务上大幅超越现有 LLM 与非 LLM 方法，验证其在 **multimodal large models** 上的有效性和扩展性。

---

### 🔍 校准性能对比（见 Table 4）

| 数据集 | SAPLMA (ECE/Brier) | **ProbPlug (ECE/Brier)** |
|--------|--------------------|--------------------------|
| SMS-SPAM | 0.0194 / 0.0240 | **0.0126 / 0.0210** |
| SST-2 | 0.1123 / 0.0987 | **0.1251 / 0.0706** |
| Amazon-Polarity | 0.1015 / 0.0563 | **0.0412 / 0.0374** |
| **Average** | **0.0982 / 0.0943** | **0.0805 / 0.0824** ✅ |

> ✅ ProbPlug 平均 ECE 和 Brier Score 更低，表明其 confidence 更接近真实准确率，**校准效果更好**。图 3(a) 显示其更贴近理想校准线，而 SAPLMA 在高置信区间明显过自信。

---

### 🔧 消融实验结果（见 Table 5）

| 配置 | In-task (SMS-SPAM) F1/AUPRC | Cross-task (SST-2) F1/AUPRC |
|------|------------------------------|------------------------------|
| w/o TC（无 Token Compression） | 94.05 / 96.99 | 74.98 / 95.44 ❌ |
| w/o LIA（无 Layer Aggregation） | 92.26 / 95.16 | 86.53 / 96.31 ❌ |
| 使用 10 个 token（S=10） | 95.94 / 97.84 ✅ | 86.84 / 94.95 ❌ |
| **ProbPlug (S=1)** | **94.78 / 97.56** | **88.47 / 97.42** ✅ |

> 🔍 发现：
- **TC 和 LIA 模块均有贡献**，尤其是 LIA 对跨任务泛化至关重要。
- 使用多个 token（S>1）虽略微提升 in-task 性能，但**损害 cross-task 泛化能力**，因前期 token 包含过多任务特定语义噪声。
- **仅使用最终生成 token（S=1）最鲁棒**，更适合通用 confidence estimation。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **LLM 各层 hidden representations 蕴含丰富的决策不确定性信息**，可用于构建可靠的 confidence estimator。
2. **ProbPlug 实现了高可靠性、高效性与强泛化性的统一**：
   - 无需修改 LLM，仅增加少量参数；
   - 单次推理完成 confidence 预测；
   - 支持跨任务迁移与多模态扩展。
3. **注意力机制可视化显示**（图 3b）：高置信预测主要依赖早期词法层（L0–L5）与末尾高级推理层（L30–L35），中间层贡献小，可能引入噪声。ProbPlug 成功抑制了这些干扰层的影响。
4. **ProbPlug 在文本与语音任务上均显著优于现有方法**，特别是在 calibration 指标上表现突出。

---

### ⚠️ 方法的局限性

- 当前主要针对 **binary classification via prompting** 设计，虽然可通过 one-vs-rest 扩展到 multi-class，但仍需进一步优化。
- 对 extremely low-resource 或 domain-shift 极大的任务，zero-shot 迁移性能仍有下降。
- 依赖于 access to internal hidden states，某些闭源 API（如 GPT-4）无法支持。

---

### 🔮 未来工作方向

1. 将 ProbPlug 扩展至更多模态（如视觉-语言模型）与复杂任务（如问答、推理链）。
2. 探索更高效的压缩与聚合策略，降低插件参数量，适配边缘设备。
3. 结合 active learning 或 rejection mechanism，实现动态可信预测过滤。
4. 开发面向闭源模型的 proxy representation extraction 技术，提升实用性。

---

## ✅ 总结

**ProbPlug 是一种简单、有效、即插即用的 confidence estimation 框架**，解决了 LLM 在 binary classification 中置信度不可靠的关键瓶颈。它不仅提升了下游任务性能，还实现了良好的校准性与跨任务泛化能力，为 LLM 在高风险场景下的安全可靠部署提供了有力工具。

> 🔗 代码已开源：[https://github.com/pingan-ai/ProbPlug](https://github.com/pingan-ai/ProbPlug)（文中提及）

</details>

---

### 14. [Granular-Ball Quantum Clustering for Resource-Efficient and Robust Learning](https://arxiv.org/abs/2609.06016)

**Authors**: Suzhen Yuan, Qilin Xie, Lifeng Shen, Shuyin Xia, Jermiah D. Deng, Guoying Wang  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.06016v1  

#### Abstract
Quantum clustering aims to exploit quantum feature representations to uncover complex data structures beyond conventional Euclidean geometry. Yet this sample-level kernel construction requires O(n^2) quantum circuit executions for n data points, creating a major bottleneck under near-term quantum re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Granular-Ball Quantum Clustering for Resource-Efficient and Robust Learning

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在当前 **NISQ (Noisy Intermediate-Scale Quantum)** 时代，**量子聚类 (Quantum Clustering)** 面临两大核心挑战：

- **计算资源瓶颈**：基于量子核 (quantum kernel) 的聚类需要对所有样本两两计算量子态保真度 (fidelity)，导致时间复杂度为 $O(n^2)$，对大规模数据不可行。
- **鲁棒性不足**：传统方法要么依赖欧氏距离（无法捕捉非线性结构），要么直接处理原始样本，易受噪声和冗余点影响。

现有方案如经典 **Granular-Ball Computing (GBC)** 虽能压缩数据、提升效率，但其基于 **Euclidean 距离**，无法建模量子空间中的纠缠关系；而纯量子方法则因高资源消耗难以扩展。

---

### **提出了什么新方法或新思路**

本文提出 **Granular-Ball Quantum Clustering (GBQC)**，将 **Granular-Ball 结构抽象** 与 **量子特征学习** 紧密结合，实现高效且鲁棒的量子聚类。

#### 核心创新点：

1. ✅ **PCA-guided Adaptive Granular-ball Generation (P-GBG)**
   - 改进传统基于 2-means 的分裂策略，采用 **PCA 主成分方向进行超平面分裂**。
   - 利用核心样本 (core samples) 估计主方向，减少噪声干扰。
   - 引入 **分裂增益阈值机制**，防止过度分割，控制 granular ball 数量。

2. ✅ **Quantum Cohesion for Robust Merging**
   - 在量子特征空间中定义 **cohesion 指标**，衡量每个 granular ball 与其 k 近邻的平均相似性。
   - 识别并过滤低 cohesion 的球（可能是噪声或边界模糊区域），提升聚类鲁棒性。

3. ✅ **Accuracy-Enhanced Resource-Efficient Framework**
   - 在压缩后的 granular ball 上构建 **量子核矩阵 (quantum kernel matrix)**，显著降低 Swap Test 执行次数。
   - 实现“**结构压缩 + 量子增强 + 噪声抑制**”三位一体框架，在降低资源消耗的同时提升聚类质量。

---

### **相比现有方法的优势**

| 维度 | GBQC | 传统 GBC 方法（如 GBCT） | 纯量子方法（如 QKKM） |
|------|------|------------------------|-----------------------|
| **效率** | ⭐⭐⭐⭐☆ (仅需 ~320 balls vs 704) | ⭐⭐⭐⭐☆ (有压缩) | ⭐⭐ (需 $O(n^2)$ 样本级计算) |
| **表达能力** | ⭐⭐⭐⭐☆ (量子特征映射) | ⭐⭐ (仅 Euclidean) | ⭐⭐⭐⭐☆ (强表达力) |
| **鲁棒性** | ⭐⭐⭐⭐☆ (cohesion 过滤噪声) | ⭐⭐☆ (易受噪声影响) | ⭐⭐⭐ (无显式去噪机制) |
| **适用性** | 复杂、非凸、含噪数据 | 凸形、简单分布 | 小规模干净数据 |

> **核心优势**：**GBQC 不仅是数据压缩手段，更是一种有效的结构抽象机制**，通过去除冗余和模糊单元，反而提升了聚类质量。

---

## 2. 核心实验方法和设置

### **使用的数据集**

共测试 **4 类共 37 个数据集**：

| 数据类型 | 数量 | 示例 | 特点 |
|--------|-----|------|------|
| **合成数据集 (Synthetic)** | 16 | Spiral, Concentric Rings, Irregular Shapes | 包含非凸、螺旋、多密度等复杂几何结构 |
| **加噪数据集 (Noisy)** | 16 | 在原合成数据上注入背景噪声 | 测试算法抗噪能力 |
| **重叠数据集 (Overlapping)** | 4 | 高度重叠的 Gaussian 分布、非凸粘连结构 | 模拟边界模糊的真实场景 |
| **真实数据集 (Real-world)** | 5 | `iris`, `seeds`, `landsat`, `segment`, `mushroom` (来自 UCI) | 多样维度与类别数，验证实际应用性 |

> 详细统计见附录 Table 6 和 Table 7。

---

### **实验设置和评估指标**

#### ✅ **评估指标**
- **Clustering Accuracy (ACC)**：标准聚类准确率。
- **Normalized Mutual Information (NMI)**：衡量聚类结果与真实标签的信息一致性。

#### ✅ **实验环境**
- 使用 **Qiskit 2.2.3** 模拟量子电路。
- 硬件：Intel i7-14650HX, 32GB RAM。
- 所有代码已开源：[https://github.com/lxqd7/GBQC](https://github.com/lxqd7/GBQC)

#### ✅ **参数设置**
- Granular ball 初始数量：$k_{\text{init}} = \sqrt{n}$
- 核心样本比例：90%
- Cohesion 过滤比例：底部 10%
- 其他参数详见附录 Table 8。

---

### **基线方法对比**

| 类型 | 方法 | 简称 | 说明 |
|------|------|------|------|
| **经典方法** | k-means | KM | 快速但假设各向同性 |
| | Density Peaks | DP | 基于密度峰值 |
| | GBSC | GBSC | 基于 granular ball 的谱聚类 |
| | GBCT | GBCT | 当前最优的自适应 granular ball 聚类 |
| **量子方法** | q-means | QM | 量子加速版 k-means |
| | QKKM | QKKM | 基于完整量子核的 k-means |
| | QCKM | QCKM | 针对 NISQ 设备的压缩 k-means |

---

## 3. 主要实验结果和性能指标

### **关键性能数据汇总**

| 方法 | 平均 ACC (Synthetic) | 平均 NMI (Synthetic) | 平均 ACC (Noisy) | 平均 NMI (Noisy) | 平均 ACC (Real) | 平均 NMI (Real) |
|------|----------------------|----------------------|------------------|------------------|------------------|------------------|
| **GBQC (本文)** | **0.976** | **0.975** | **1.000** | **0.944** | **0.825** | **0.663** |
| GBCT | 0.931 | 0.955 | 0.865 | 0.800 | 0.535 | 0.405 |
| QKKM | 0.667 | 0.421 | 0.366 | 0.366 | 0.786 | 0.565 |
| QCKM | 0.705 | 0.594 | 0.696 | 0.587 | 0.732 | 0.547 |

> 数据来源：Table 2–4

---

### **与基线方法的对比结果**

#### 🔹 **在合成数据上**
- GBQC 在 **16/16 数据集** 上达到最高 ACC/NMI。
- 在复杂结构（如 Spiral M）上，ACC 达到 **0.994**，远超 GBCT 的 **0.552**。
- 相比 QKKM，ACC 提升超过 **0.3**，且避免内存溢出。

#### 🔹 **在加噪数据上**
- GBQC 展现出极强鲁棒性，多数数据集仍保持接近 **1.0** 的 ACC。
- 在 `Dnoise` 上，ACC 达 **0.998**，而 QKKM 仅为 **0.565**。
- 表明 **cohesion 机制有效过滤噪声球**。

#### 🔹 **在真实数据上**
- 在 `iris` 和 `landsat` 上显著优于 GBCT（ACC 提升约 0.3）。
- 表明 **量子特征映射能更好分离非线性边界**。

#### 🔹 **资源效率对比**
- Granular ball 数量从 GBCT 的平均 **704** 降至 **320**（↓54.6%）。
- 导致 Swap Test 执行次数从 $704^2 ≈ 495,616$ 降至 $320^2 ≈ 102,400$，**减少约 80%**。
- 运行时间稳定在 **10–100 秒**，而 QKKM 在大数据集上常超时（>10³ 秒）。

> 图 6 和附录图 8 显示 GBQC 具有最佳可扩展性。

---

### **消融实验结果（Ablation Study）**

#### ✅ **P-GBG 策略有效性**
- 对比变体：
  - `GBQC(w/o GB)`：不使用 granular ball，直接在原始样本上构建量子核。
  - `GBQC(2-means)`：使用传统 2-means 分裂而非 PCA-guided。

| 变体 | 平均时间 | 平均 ACC |
|------|---------|----------|
| GBQC (完整) | **25.39s** | **0.976** |
| GBQC(2-means) | 226.28s | 0.893 |
| GBQC(w/o GB) | 6789.90s | 0.806 |

> **结论**：P-GBG 极大提升效率，且 PCA 分裂比 2-means 更符合数据流形结构。

#### ✅ **Cohesion 机制有效性**
- 对比 `GBQC` vs `w/o Cohesion`

| 指标 | GBQC | w/o Cohesion |
|------|------|---------------|
| 平均 ACC (Noisy) | **0.964** | 0.884 |
| 平均 NMI (Noisy) | **0.944** | 0.872 |

> 在 `Dnoise` 上，ACC 从 **0.998 → 0.697**，表明 **cohesion 是鲁棒性的关键**。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Granular-ball 表示不仅是压缩工具，更是结构抽象机制**  
   通过去除冗余和模糊的局部分区，保留代表性结构单元，反而提升了聚类质量。

2. ✅ **PCA-guided 分裂优于传统 2-means**  
   更好地对齐数据方差方向，生成更紧凑、几何一致的 granular balls。

3. ✅ **量子空间中的 cohesion 度量可有效识别噪声**  
   低 cohesion 的 granular ball 多位于稀疏区或边界，过滤后显著提升聚类纯净度。

4. ✅ **GBQC 实现了“效率-精度-鲁棒性”的三重平衡**  
   在大幅降低量子资源消耗（↓80% Swap Test）的同时，取得最优聚类性能。

---

### **方法的局限性**

- **当前基于模拟器**：尚未在真实量子硬件上部署，受限于 NISQ 设备的噪声和连通性。
- **固定编码方式**：使用 angle encoding，未探索可训练的量子特征映射 (trainable quantum feature map)。
- **参数敏感性**：如核心样本比例、cohesion 阈值等需调参，虽设为默认值但可能影响泛化。

---

### **未来工作方向**

1. **部署到真实量子设备**：验证 GBQC 在真实 NISQ 环境下的可行性。
2. **引入可训练量子编码**：结合 VQA (Variational Quantum Algorithm) 框架优化特征映射。
3. **理论分析 granulation 与量子表示的关系**：建立 granular ball 数量、压缩率与聚类性能之间的理论界。
4. **拓展至其他任务**：如量子分类、异常检测、图聚类等。

---

> **一句话总结**：  
> GBQC 证明了 **“少即是多” (less is more)** —— 通过对数据进行合理的 granular-ball 抽象，不仅能极大降低量子资源消耗，还能提升聚类质量和鲁棒性，为 **可扩展的量子机器学习** 提供了一条新路径。

</details>

---

### 15. [Hidden in Plain Sight: The Overlooked Significance of Canonical Elements for Extreme LLM Sparsity](https://arxiv.org/abs/2609.06557)

**Authors**: Hyeondo Jang, Kwanhee Lee, Dongyeop Lee, Namhoon Lee  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.06557v1  

#### Abstract
Large language models (LLMs) are often considered fragile under aggressive sparsification, and maintaining reliable performance typically requires sticking to moderate sparsity levels. However, recent studies suggest that LLMs are more resilient to high sparsity than previously thought, reframing th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Hidden in Plain Sight: The Overlooked Significance of Canonical Elements for Extreme LLM Sparsity  
**论文核心总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在高稀疏度（如 >90%）下通常被认为性能会急剧下降，因此主流研究集中在中等稀疏度（50–70%）。本文挑战这一共识，提出：**极端稀疏性（extreme sparsity）下的性能退化并非模型固有缺陷，而是由于当前剪枝范式的设计局限所致**。

### 🚀 提出的新方法：BEST
作者提出了名为 **BEST**（**B**alanced **E**xtreme **S**parsity via **T**uning）的框架，其核心是重新审视并系统整合经典剪枝组件，在现代 LLM 上进行规模化适配：

- **Masking Strategy**: 全局（global）第二阶显著性（second-order saliency），结合优化器感知代理（optimizer-aware proxy）实现零成本曲率估计。
- **Sparsity Schedule**: 采用立方渐进稀疏调度（cubic gradual sparsity schedule），逐步引入稀疏性。
- **Training Strategy**: 学习率调度与稀疏进度协调，采用 warmup-decay 策略以稳定训练。

### 🔍 相比现有方法的优势
- **突破稀疏极限**：首次在高达 **99% 稀疏度**下保持 LLM 的强性能，远超传统 one-shot 剪枝方法的崩溃点（~70%）。
- **无需复杂设计**：仅依赖经典剪枝思想（如 LeCun 的 Optimal Brain Damage），无需学习掩码、动态稀疏训练等复杂机制。
- **高效实现**：利用 Adam 优化器状态近似对角 Fisher 信息矩阵，避免额外梯度计算；通过分布式二分搜索实现全局阈值划分，支持大规模分布式训练。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **预训练语料**：
  - `SlimPajama`（主实验）：627B token 的清洗去重语料，包含 C4、ArXiv、GitHub 等来源。
  - `C4`（用于与 ELSA 对齐比较）。
- **校准数据**（用于 one-shot 方法）：从 C4 中采样 128 条序列，长度为 2048。
- **评估数据集**：
  - **语言建模**：`WikiText-2` 和 `C4` 验证集，评估 **Perplexity (PPL)**。
  - **下游任务**：使用 `lm-eval-harness` 测试 7 项零样本任务：
    - ARC-Easy/Challenge, BoolQ, HellaSwag, OBQA, RTE, Winogrande
  - **生成能力**：GSM8K（数学推理）、NQ-Open（开放问答）。

### ⚙️ 实验设置
- **模型家族**：
  - LLaMA-2（7B, 13B）
  - Qwen-3-Base（4B, 8B）
  - Qwen2.5（32B）
- **稀疏度范围**：70% → 99%
- **训练配置**：
  - 总训练步数：4000 步（batch size 64, seq len 2048）
  - 稀疏阶段占前 50%，每 40 步更新一次掩码
  - 使用 bf16 混合精度，参数和优化器状态保持 FP32
- **硬件**：A100/H200/A6000 等 GPU 集群

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **One-shot 剪枝** | Magnitude, Wanda, SparseGPT |
| **带微调的一次性剪枝** | Magnitude+, Wanda+, SparseGPT+ |
| **近期先进方法** | L-ADMM, ALPS, SAFE, SparseLLM, ELSA |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（LLaMA-2-7B）

| 稀疏度 | 方法 | WikiText-2 PPL | C4 PPL |
|--------|------|----------------|--------|
| 95% | **BEST (本文)** | **13.48** | **16.52** |
| 95% | SparseGPT+ | 18.75 | 20.72 |
| 95% | ELSA | 38.91 | — |
| 99% | **BEST (本文)** | **19.67** | — |
| 99% | SparseGPT+ | 35.53 | — |
| 99% | ELSA | 55.94 | — |

> 💡 在 **99% 稀疏度**下，BEST 的 PPL 仅为 ELSA 的 **三分之一以下**，且显著优于所有 one-shot 方法。

### 🚀 推理效率提升（LLaMA-2-7B @ 95% 稀疏）
- **解码吞吐量**：**3.23× 加速**（176.18 vs. 54.47 tokens/s）
- **显存占用**：**6.21× 减少**（2.19 GB vs. 13.6 GB）
- **质量对标**：95% 稀疏的 BEST 模型语言建模质量（PPL=13.48）优于 70% 稀疏的 ELSA 模型（PPL=13.20）

### 📈 下游任务表现（Qwen3-8B @ 95% 稀疏）
| 方法 | 平均零样本准确率（7项任务） |
|------|-------------------------------|
| SparseGPT+ | 37.94% |
| **BEST** | **39.71%** |
| Dense 基线 | 57.83% |

> 即使在极端稀疏下，BEST 仍能保留大部分下游能力，且明显优于其他方法。

### 🔬 消融实验结果（基于 OPT-125M）

#### （1）显著性准则对比（70% 稀疏，WikiText-2 PPL）
| 显著性 | 层级比较 | 全局比较 |
|--------|----------|----------|
| Magnitude | 32.44 | 34.54 |
| 1st-order | 36.61 | 36.49 |
| **2nd-order** | **31.28** | **30.77** ✅ |

> 第二阶显著性 + 全局比较效果最佳，而 magnitude 在全局比较下反而更差，说明其受模块尺度偏差影响严重。

#### （2）稀疏调度策略（90% 稀疏）
- **一次性剪枝（One-shot）**：最终 PPL = 14.11
- **渐进式剪枝（BEST）**：最终 PPL = **11.60**
> 渐进式调度可避免训练初期损失爆炸，显著提升收敛质量。

#### （3）学习率 warm-up
- 无 warm-up 导致训练不稳定；
- **10% warm-up（400步）** 可有效缓解初始扰动，获得最低 PPL。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **极端稀疏是可行的**：LLMs 在高达 **99% 参数被剪除**的情况下仍可保持强大性能，推翻“70% 是硬上限”的普遍认知。
2. **经典方法被低估**：第二阶显著性、渐进剪枝、全局阈值等“老”思想在正确组合与工程实现下，足以超越当前 SOTA。
3. **协同设计至关重要**：masking、schedule、training 三者必须协同优化，单一改进无法达到最佳效果。
4. **稀疏 ≠ 质量损失**：在合适训练下，稀疏模型不仅能压缩参数，还能实现 **实际推理加速与内存节省**。

### ⚠️ 方法的局限性
1. **需要持续训练**：相比 one-shot 方法，BEST 需要额外训练成本（约 10 小时 A100），不适合极低预算场景。
2. **自由生成与推理能力退化严重**：
   - GSM8K 数学推理从 87.64%（dense）降至 2.12%（99% 稀疏）
   - NQ-Open 事实召回也大幅下降
   > 表明 **perplexity 和选择题准确率不能完全代表生成鲁棒性**
3. **依赖专用稀疏内核**：实际加速依赖于 `SpMV` 内核（如 Macko and Boza, 2026），缺乏原生硬件支持。
4. **未探索 MoE 架构**：当前方法集中于 dense Transformer，是否适用于 Mixture-of-Experts 尚未知。

### 🔮 未来工作方向
1. **降低训练成本**：探索更高效的训练数据选择（如 Lin et al., 2024）以减少 token 需求。
2. **恢复高级推理能力**：结合 post-training alignment（如 GRPO）、指令微调、数据蒸馏来修复生成退化。
3. **与量化联合压缩**：已验证 INT8 几乎无损，INT4 仅有轻微退化，未来可探索 Pruning + Quantization 联合方案。
4. **扩展至 MoE 与结构化稀疏**：将 BEST 思想迁移到更现代的稀疏架构。
5. **构建端到端稀疏训练 pipeline**：从预训练开始即引入稀疏性，而非 post-training 剪枝。

---

> **一句话总结**：  
> 本文证明，通过**忠实应用经典剪枝原则并进行系统性协同设计**，可以在不引入复杂机制的前提下，将 LLM 推向 **99% 极端稀疏度**，同时保持甚至超越现有 SOTA 的性能，揭示了“canonical elements”在现代 LLM 压缩中的巨大潜力。

</details>

---

### 16. [AF-Mamba: Efficient Long-Term Signal Modeling for Early Prediction of Atrial Fibrillation Onset](https://arxiv.org/abs/2609.06984)

**Authors**: Yongbin Lee, Ki H. Chon  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.06984v1  

#### Abstract
Atrial fibrillation (AF) is the most common cardiac arrhythmia and is associated with increased risks of stroke and heart failure. The growing availability of wearable and portable ECG monitoring enables continuous assessment of cardiac rhythm outside clinical settings. Predicting AF before its onse...

---

### 17. [EnvCraft: Synthesizing Executable Environments in Agentic RL for Claw-like Agent](https://arxiv.org/abs/2609.05576)

**Authors**: Yirong Zeng, Shen You, Jinhang Feng, Yufei Liu, Xiao Ding, Yutai Hou, Hao Cong, Yuxian Wang, Wu Ning, Wang Xu, Bibo Cai  
**Category**: cs.AI  
**Published**: 2026-09-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.05576v1  

#### Abstract
The paradigm of LLMs has rapidly shifted from passive language interfaces to autonomous Claw-like agents that execute long-horizon tasks across stateful workspaces. While Agentic Reinforcement Learning (Agentic RL) provides a promising path to optimize these agents, its scaling is heavily bottleneck...

---

### 18. [BIO-MEMART: Biometric-Aware KV Cache Memory for Multi-User LLM Agents](https://arxiv.org/abs/2609.08566)

**Authors**: Yanhong Qian, Xuanying He, Qingguo Meng, Shihao Ding, Xingbo Dong, Zhe Jin  
**Category**: cs.AI  
**Published**: 2026-09-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.08566v1  

#### Abstract
KV cache is evolving from a serving optimization into an external memory substrate for long-term LLM agents. In a shared multi-user deployment, however, reusable KV blocks introduce a missing access-control question: semantic relevance alone cannot determine whether a memory block is authorized for ...

---

### 19. [Osprey: Target-agnostic Pre-training Makes Stronger Drafters in Speculative Decoding](https://arxiv.org/abs/2609.09338)

**Authors**: Fengxiang Bie, Yuqing Jian, Yifan Yu, Zhongzhu Zhou, Zelei Shao, Ben Athiwaratkun, Shuaiwen Leon Song, Chenfeng Xu, Xiaoxia Wu, Tianyi Zhang  
**Category**: cs.CL  
**Published**: 2026-09-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.09338v1  

#### Abstract
Speculative decoding is critical for accelerating LLM inference. However, the speedup is fragile: drafters are typically trained against a narrow distribution for a single target model, and their acceptance rate collapses under workload shifts. This is a striking inversion of modern LLM development,...

---

### 20. [RAPTOR: Role-Aware Private Training for Mixture-of-Experts](https://arxiv.org/abs/2609.05770)

**Authors**: Duc Dm, Khai Le-Duc, Nguyen Do, Minh Son Hoang, Florent Draye, Thai Hoang, Hoang Phuong Dam, Jiarui Liu, Chris Ngo, Terry Jingchen Zhang, Anh Le Duc Tran, Nhat Do Minh, Minh Ngoc Le, My T. Thai, Ran Xu, Silvio Savarese, Mona Diab, Bernhard Sch\"olkopf, Zhijing Jin, Huy L. Nguyen, Daeyoung Kim  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.05770v1  

#### Abstract
Differentially private (DP) fine-tuning methods treat sparse Mixture-of-Experts (MoE) models as a single dense block, ignoring that shared layers see all data while experts only see routed records. We identify and formally characterize three resulting failure modes: global clipping suppresses expert...

---

### 21. [FANS: Federated Adaptive Network Search Learning for Heterogeneous Devices](https://arxiv.org/abs/2609.06106)

**Authors**: Jiaxin Zhang, Xingwei Wang, Bo Yi, Liang Zhao, Alireza Furutanpey, Ziyi Chen, Qiang He, Keqin Li, Schahram Dustdar  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.06106v1  

#### Abstract
Heterogeneous Federated Learning (HFL) aims to train models across devices with diverse resource budgets while preserving data privacy. Existing HFL methods typically bind training to a small predefined menu of model configurations, which limits architectural coverage. To address this bottleneck, we...

---

### 22. [Rethinking One-Shot Federated Graph Learning: Training-Free Statistical Estimation](https://arxiv.org/abs/2609.06154)

**Authors**: Shutong Zheng, Sijia Chen  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.06154v1  

#### Abstract
One-shot federated graph learning generally aims to train Graph Neural Networks (GNNs) across clients with disconnected subgraphs in a single communication round. Existing methods predominantly design advanced optimization strategies under the premise that local GNN training is indispensable. Howeve...

---

### 23. [$\alpha$-Graph: Attention-Infused Normalizing Flow Approach to Tractable Graph Modeling](https://arxiv.org/abs/2609.07961)

**Authors**: Thanh-Dat Truong, Sarah Alharbi, Susan Gauch, Xinghui Zhao, Marios Savvides, Khoa Luu  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.07961v1  

#### Abstract
Graph modeling, a crucial task for representing complex relationships in graph-structured data, has achieved significant success in recent years. However, current graph modeling methods rely on traditional Graph Neural Networks and pre-training approaches to implicitly learn the underlying relationa...

---

### 24. [PlayTrain: An Efficient Reinforcement Learning Framework for LLM-Generated Adaptable JavaScript Games](https://arxiv.org/abs/2609.09059)

**Authors**: Ryan Truong, Lance Ying, Samuel J. Gershman, Kazuki Irie  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.09059v1  

#### Abstract
While many video-game environments (VGEs) have played crucial roles in advancing reinforcement learning (RL), developing novel VGEs or modifying existing ones to support new features, has been a laborious process requiring extensive hand-coding. Here we present PlayTrain, an RL framework that combin...

---

### 25. [EEG-Driven Decoding Framework for Passenger Hazard Perception in Highly Automated Vehicles](https://arxiv.org/abs/2609.07128)

**Authors**: Yingkai Yang, Ashton Yu Xuan Tan, Bowen Li, Xiaorong Gao, Sifa Zheng, Jianqiang Wang, Xinyu Gu, Yang Zhao, Yuxin Zhang, Sharon X. Huang, Tania Stathaki, Jun Li, Hong Wang  
**Category**: cs.AI  
**Published**: 2026-09-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.07128v1  

#### Abstract
Reliable risk assessment remains a central challenge for Autonomous Vehicles (AVs). Despite advances in automation, passenger cognition provides a non-intrusive auxiliary signal that improves both objective and perceived safety without requiring active human intervention. We introduce an Electroence...

---

### 26. [Eliciting Self-Verification in Multimodal Reasoning Agents with Reinforcement Learning](https://arxiv.org/abs/2609.08025)

**Authors**: Vishwas Sathish, Viresh Ranjan, Xinliang Zhu, Arnab Dhua, Douglas Gray  
**Category**: cs.AI  
**Published**: 2026-09-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.08025v1  

#### Abstract
Reasoning agents increasingly rely on external tools such as web search to answer complex queries. Reinforcement learning (RL) finetuning algorithms such as GRPO have improved long-form reasoning in text-only language models, particularly for coding and mathematics. Reliable tool use in multimodal a...

---

### 27. [WorldAgen: Unified State-Action Prediction with Test-Time World Model Training](https://arxiv.org/abs/2609.08162)

**Authors**: Chi Wan, Kangrui Wang, Yuan Si, Pingyue Zhang, Manling Li  
**Category**: cs.AI  
**Published**: 2026-09-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.08162v1  

#### Abstract
How can vision-language-action (VLA) models adapt to new environments where world dynamics shift? While recent research has combined world modeling and action prediction to improve VLA performance, existing methods largely rely on pretraining on static datasets, without mechanisms for active adaptat...

---

### 28. [Scaling E-Commerce Attribute Extraction with Parallel Decoding](https://arxiv.org/abs/2609.09716)

**Authors**: Nikhita Vedula, Dushyanta Dhyani, Bryan Wang, Shervin Malmasi  
**Category**: cs.CL  
**Published**: 2026-09-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.09716v1  

#### Abstract
Customers rely on specific product attributes to compare products and make purchasing decisions, but e-commerce catalogs are messy and unstructured, making it difficult to identify which attributes matter most and extract them at scale. Standard Attribute Value Extraction (AVE) systems treat all att...

---

### 29. [Epoch: Compiling Diffusion Blocks for Sparse MoE Serving](https://arxiv.org/abs/2609.09748)

**Authors**: Jianian Zhu, Hang Wu, Yinghui Li, Haojie Wang, Ruixuan Li, Jidong Zhai  
**Category**: cs.DC  
**Published**: 2026-09-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.09748v1  

#### Abstract
Diffusion language models generate text by refining a fixed-size block of token positions through many forward passes, a loop that does not match the per-forward execution unit used by most LLM serving systems. A dense MoE runtime binds all work to the refinement-iteration clock: it rebuilds similar...

---

### 30. [Stable-MM-R1: Anchoring Multimodal Reasoning Dynamics via Entropy-Guided Stratification](https://arxiv.org/abs/2609.07148)

**Authors**: Yimeng Ye, Shuang Chen, Wenxuan Huang, Manyuan Zhang, Kaituo Feng, Zhangquan Chen, Jiayu Chen, Yucheng Zhou, Yicheng Xiao, Zhiyuan Feng, Tianyu Shi  
**Category**: cs.LG  
**Published**: 2026-09-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.07148v1  

#### Abstract
While Reinforcement Learning (RL) effectively incentivizes reasoning in Large Language Models, current pipelines are hindered by training instability and rapid entropy collapse. These limitations often stem from "Rollout Silencing" and low-quality gradient signals in standard sampling procedures. In...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-10-22 10:40:01 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Is 2:4 Sparsity Still Worth It? Semi-Structured Sparsity versus Quantization in LLM Inference](https://arxiv.org/abs/2610.28947)

**Authors**: Xiuyi Qin  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2610.28947v1  

#### Abstract
NVIDIA tensor cores have supported 2:4 semi-structured sparsity since Ampere, and the pattern composes with quantization. When the two are combined, the practical question is which step to take next: from a given precision, add 2:4 sparsity or drop to the next lower precision? We compare the two at ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Is 2:4 Sparsity Still Worth It? Semi-Structured Sparsity versus Quantization in LLM Inference*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文系统性地解决了在当前主流 LLM 推理场景下，**2:4 半结构化稀疏（semi-structured sparsity）是否仍然值得使用**这一关键实践问题。具体而言，当模型已经处于某一精度（如 FP8）时，下一步应选择：
- **添加 2:4 稀疏性**，还是  
- **直接量化到更低的精度**（如从 FP8 到 INT4 或 NVFP4）？

尽管 NVIDIA 自 Ampere 架构以来一直支持 2:4 稀疏 tensor cores，但主流推理框架（如 vLLM 和 LLM Compressor）已在 2026 年移除了对 2:4 的支持，反映出社区对该技术实用性的质疑。本文旨在通过严谨的实证研究澄清这一争议。

### 提出了什么新方法或新思路
- **提出“等效比特”（equal bits）和“等效 tensor-core 层级”（equal tensor-core tiers）两种公平比较范式**：
  - **Equal bits**：将 2:4 稀疏格式的 metadata（索引位）计入总存储开销，确保与密集格式在相同存储预算下对比。
  - **Equal tiers**：在同一 tensor-core 计算吞吐层级（如 2×、4×）下比较稀疏与低精度密集格式的实际表现。
- **引入“2:4 能量损失”（2:4 energy loss）作为理论分析工具**，解释为何 2:4 在多数情况下劣于进一步量化。
- **全面覆盖所有六类 GEMM 操作数**（weights, inputs, queries, keys, values, attention probs），并考虑动态操作数的在线剪枝与压缩开销。

### 相比现有方法的优势
- **首次在真实硬件上进行端到端、多维度、跨代 GPU 的系统性比较**，涵盖 Ada、Hopper 和 Blackwell 三代架构。
- **纠正了以往研究忽略 metadata 开销和动态处理成本的问题**，提供了更贴近实际部署的评估。
- **揭示了 retraining 对 2:4 成败的关键作用**，并指出仅靠 kernel 加速不足以带来端到端收益。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准与评估数据**：
  - **C4**：用于 calibration 和部分评估。
  - **LongBench**：用于长上下文理解任务。
  - **WikiText-2**：用于计算 **perplexity**。
  - **GSM8K**：用于评估数学推理能力（8-shot CoT）。
  - **RULER**：用于评估长上下文检索能力（16K tokens）。

### 实验设置和评估指标
- **模型**：
  - `Llama-3.1-8B-Instruct`
  - `Qwen3-8B`
  - `Sparse-Llama`（已重训练的 2:4 版本）
- **硬件平台**：
  - **L40S**（Ada 架构）
  - **H100**（Hopper 架构）
  - **RTX PRO 6000**（Blackwell 架构，sm_120）
- **测量方式**：
  - 使用 **CUPTI** 获取 kernel 级时间。
  - 包含在线剪枝、量化、压缩等预处理开销。
  - 内存驻留于 L2 缓存以模拟流水线场景。
- **评估指标**：
  - **Operator-level error ratio R**：输出相对于 FP32 参考的相对误差中位数。
  - **Throughput**：token/s（prefill 和 decode 阶段）。
  - **Model-level metrics**：perplexity、GSM8K 准确率、RULER 分数。
  - **Speedup ratio**：$ t(\text{rival}) / t(2:4) $

### 基线方法对比
- **2:4 + Quantization**：
  - 如 2:4 + FP16 → 对比 FP8
  - 2:4 + INT8 → 对比 INT5
- **Dense Quantization**：
  - FP8、INT8、INT5、INT4、MXFP4、NVFP4
- **Retraining 方法**：
  - 一次性剪枝（one-shot pruning）
  - 基于 distillation 的 retraining（1B tokens）
  - 对比 Sparse-Llama（13B tokens retrained）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 场景 | 性能提升 | 说明 |
|------|----------|------|
| **Above Quantization Floor** | ❌ 2:4 更慢且更不准 | 在仍有更低精度可用时，drop precision 显著优于 add 2:4 |
| **At FP8 Floor (H100)** | ✅ +27% prefill throughput | 重训练后的 2:4 FP8 模型在 vLLM 上实现 27% 吞吐提升 |
| **At FP8 Floor (Llama-3.1-8B)** | ✅ 1.3–1.7× linear layer speedup | 不同 GPU 上线性层加速明显 |
| **Blackwell Sparse FP4** | ⚠️ 1.4× faster but unusable reasoning | pairwise 4:8 模式速度快但重训练后仍无法恢复 GSM8K 准确率 |

### 与基线方法的对比结果

#### 存储效率对比（Equal Bits）
- 2:4 + INT8 ≡ INT5（而非 INT4）
- 2:4 + INT4 ≡ INT3
- 2:4 + FP16 > FP8（比特更多）
- 在相同比特下，**dense quantization 始终更准确**

#### 计算效率对比（Equal Tiers）
| Tier | Configuration | Speed Ratio (vs 2:4) | Accuracy (R) |
|------|---------------|------------------------|--------------|
| 2× | 2:4-FP16 vs FP8 | **0.74–0.89**（2:4 更慢） | R = 5–16（2:4 更差） |
| 4× | 2:4-FP8 vs NVFP4 | **0.66–0.70**（2:4 更慢） | R ≥ 1（除个别弱对手外均劣） |

> 结论：在可降精度时，**任何 2:4 配置都落在帕累托劣势区域**（既慢又不准）。

### 消融实验结果

#### 重训练的影响（Table 4）
| Floor | One-shot 2:4 | Retrained (1B tokens) | Sparse-Llama (13B tokens) |
|-------|---------------|------------------------|----------------------------|
| FP8 | PPL: 17.82 → GSM8K: 0.028 | PPL: 8.25 → GSM8K: 0.308 | PPL: 8.19 → GSM8K: 0.632 |
| INT4 | PPL: 18.61 → GSM8K: 0.032 | PPL: 8.53 → GSM8K: 0.292 | PPL: 8.45 → GSM8K: 0.580 |

- **Perplexity 恢复快，但 GSM8K 恢复极慢**：即使 PPL 接近原始水平，推理能力仍严重受损。
- **Blackwell 的 pairwise 4:8 模式代价更高**：one-shot 剪枝导致 PPL 超过 100，远高于 element-wise 2:4 的 ~22。

#### 部分堆叠（Partial Stacking）
- 即使只对预测成本最低的单元应用 2:4，也只能获得 **1.14–1.25×** 的加速，且仍需 retraining 才能保持质量。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **只要存在更低的量化精度，drop precision 总是优于 add 2:4**：
   - 因为 2:4 删除的元素携带固定比例的能量损失（~13.2% for Gaussian），不随比特减少而降低；
   - 而量化误差每增加 1 bit 就减半，因此在低比特下更具优势。

2. ✅ **在量化底限（quantization floor）上，2:4 才真正有价值**：
   - 在 FP8 层级上叠加 2:4 可带来 **1.3–1.7× 线性层加速**；
   - 在 H100 上，**retrained 2:4 FP8 模型在 vLLM 中实现 27% 更高的 prefill 吞吐**，且 GSM8K 准确率持平。

3. 🔁 **重训练是 2:4 成功的前提**：
   - 一次性剪枝（one-shot）完全不可用；
   - Perplexity 很快恢复，但 **reasoning 能力滞后严重**，不能作为 retraining 完成的判据。

4. ⚠️ **Blackwell 的 pairwise 4:8 模式面临更大挑战**：
   - 模式本身丢弃更多能量（~20.8% vs 13.2%）；
   - 无法复用为 2:4 训练的模型；
   - 当前 retraining 预算下未能恢复可用的推理能力。

### 方法的局限性
- **依赖特定硬件支持**：2:4 加速仅适用于 NVIDIA tensor cores，且仅一个操作数可稀疏。
- **动态操作数处理开销大**：如 attention probs 的在线剪枝成为瓶颈。
- **metadata 开销显著**：2:4 实际节省的比特远小于理论值（因需存储索引）。
- **Blackwell 新模式兼容性差**：pairwise 4:8 无法复用现有 2:4 生态。

### 未来工作方向
- 设计更适合 **pairwise 4:8 模式的训练策略**，以恢复其推理能力。
- 探索 **hybrid 方法**：结合 structured sparsity 与 quantization 的最优路径。
- 开发 **更高效的在线剪枝 kernel**，减少动态操作数的预处理延迟。
- 建立 **更合理的 retraining 判停标准**，避免仅依赖 perplexity 导致的误判。

---

> **一句话总结**：  
> **“Above the floor, quantize deeper; at the floor, sparsify smarter — but only if you retrain.”**  
> 在有更低精度可用时，优先量化；在量化底限时，2:4 才能提速，但前提是必须充分重训练。

</details>

---

### 2. [RCSD: Reinforcement and Continual Learning for Domain-Specific Speculative Decoding](https://arxiv.org/abs/2610.29527)

**Authors**: Mingkang Liu, Weiming Meng, Peipei Jiao, Jiawei Zhang, Tianyuan Liu, Junyan Li, Lichen Pang  
**Category**: cs.CL  
**Published**: 2026-10-22  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.29527v1  

#### Abstract
Speculative decoding accelerates large language model inference by employing a compact draft model to propose candidate tokens that are verified in parallel by a target model. Conventional training relies on soft-label distillation, which forces the draft model to mimic the target's full output dist...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RCSD: Reinforcement and Continual Learning for Domain-Specific Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **speculative decoding** 框架中，draft model 的训练通常依赖于基于 **soft-label distillation** 的知识蒸馏方法（如 HASS），其目标是让 draft model 尽可能模仿 target model 的完整输出分布（即最小化 KL 散度）。然而，这种“模式覆盖”（mode-covering）行为会导致 draft model 过度关注低概率、高熵区域，而这些区域在实际验证过程中容易被拒绝，从而降低 **acceptance rate** 和 **acceptance length**。

更重要的是，**训练目标（分布对齐）与系统级优化目标（最大化接受长度）之间存在根本性错位**。即使 KL loss 很小，也可能因 top-k 排序不一致而导致 token 被拒。

### 提出了什么新方法或新思路
本文提出 **RCSD**（Reinforcement and Continual Learning for Speculative Decoding），一种全新的 draft model 训练框架，核心思想是：

- **将训练目标从“分布匹配”转向“接受长度最大化”**，直接对齐系统级性能指标。
- 利用 speculative decoding 过程中自然产生的 **accepted 和 rejected token 序列** 构造隐式的偏好信号（implicit preference pairs）。
- 引入两种基于强化学习的优化策略：
  - **DPO**（Direct Preference Optimization）：构建 (chosen, rejected) 对，其中 chosen 来自纯 target model 生成，rejected 来自 draft/target 交替生成。
  - **GRPO**（Group-wise Relative Policy Optimization）：构造多个不同 drafting 策略下的响应组，以 acceptance rate 作为奖励信号进行相对优化。

此外，采用 **masked loss** 机制，屏蔽掉由 target model 生成的 token，避免对已对齐部分重复惩罚，提升训练信号纯净度。

### 相比现有方法的优势
| 维度 | 传统方法（如 HASS） | RCSD |
|------|---------------------|-------|
| **训练目标** | 分布匹配（KL 最小化） | 接受长度最大化（系统效率导向） |
| **优化方式** | 静态监督学习 | 动态反馈驱动（利用推理时验证结果） |
| **行为倾向** | Mode-covering（覆盖所有可能） | Mode-seeking（聚焦高接受概率序列） |
| **外部依赖** | 无需额外标注，但目标错位 | 完全自反馈，无需人工标注或外部 reward model |
| **性能增益** | 有限加速 | 显著提升 acceptance length 和 end-to-end 吞吐 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **训练数据**：使用 ShareGPT 数据集的 90% 子集进行训练。
- **评估数据集** 包括四类任务，覆盖多种场景：
  - **GSM8K**：数学推理任务
  - **HumanEval**：代码生成能力
  - **MT-Bench**：开放域指令遵循与对话质量
  - **ShareGPT（剩余10%）**：真实用户对话样本

### 实验设置和评估指标
#### 模型配置
- 主干模型：**Qwen3-1.7B** 和 **Llama-3.1-8B-Instruct**
- 验证工具：基于 **SGLang 5.5.0** 实现 speculative decoding 流程

#### 评估指标
- **Mean Accept Length**：每个验证周期平均被接受的 token 数量（核心指标）
- **End-to-End Inference Speedup**：端到端推理速度提升倍数
- **Output Throughput (TPS)**：每秒输出 token 数
- **Generation Quality**：通过下游任务准确率间接衡量（确保无质量损失）

#### Speculative Steps 设置
测试 drafting step 为 3–6 的情况，即每次 draft 模型生成 3–6 个候选 token。

### 基线方法对比
- **HASS**：当前主流的知识蒸馏 baseline，用于训练 draft model
- **HASS + DPO**：在 HASS 初始化基础上应用 DPO 优化
- **HASS + GRPO**：在 HASS 初始化基础上应用 GRPO 优化

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在 Qwen3-1.7B 上的结果（Step 6 平均 accept length）
| 方法 | Average Accept Length |
|------|------------------------|
| HASS | 2.26 |
| HASS+DPO | 2.97 |
| HASS+GRPO | **3.00** |

> 提升幅度：相比 HASS 提升约 **32%**

#### 在 Llama-3.1-8B 上的结果（Step 6 平均 accept length）
| 方法 | Average Accept Length |
|------|------------------------|
| HASS | 2.672 (ShareGPT) → ~2.7 |
| HASS+DPO | 3.779 |
| HASS+GRPO | **3.828** |

> 提升幅度：相比 HASS 提升超过 **40%**

#### 综合性能表现
- **平均 acceptance length 提升 35%**（跨模型与数据集）
- **端到端推理速度提升 1.35×**
- **吞吐量（TPS）显著提高**（见 Figure 2）
- 所有任务上的最终生成质量保持不变（如 HumanEval pass@1、GSM8K 准确率等未下降）

### 与基线方法的对比结果
- **RCSD 全面优于 HASS**，尤其在长序列推测（step=6）下优势更明显。
- **GRPO 性能优于 DPO**，说明利用多路径相对比较比二元偏好更能稳定提升性能。
- 提升效果在 **ShareGPT 和 GSM8K** 上尤为突出，表明该方法在复杂语义和逻辑任务中更具潜力。

### 消融实验结果（文中体现）
虽然没有明确列出消融表，但从设计可看出以下关键组件的作用：
- **Masked Loss**：防止对 target 已生成 token 进行错误惩罚，保障训练稳定性。
- **Preference Signal 来源自系统内部反馈**：无需人工标注或额外 reward model，实现闭环优化。
- **GRPO 利用 group-wise 相对排序**：比 DPO 更高效地捕捉细微差异，加快收敛。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **训练目标与系统目标必须对齐**：单纯追求 KL 收敛无法保证 speculative decoding 的高效性；**acceptance length 是更合理的优化目标**。
2. **speculative decoding 自身提供了天然的反馈机制**：accepted/rejected 序列本身就是高质量的偏好数据，可用于 RL 优化。
3. **DPO 和 GRPO 可有效迁移到 draft model 训练中**：这是首次系统性将 preference optimization 技术引入 speculative decoding 训练流程。
4. **GRPO 表现最优**：因其利用多个 drafting 路径之间的相对质量排序，提供更丰富的梯度信号，加速训练并提升性能。

### 方法的局限性
- 当前方法依赖于 target model 的稳定输出作为参考，若 target model 输出波动大，可能影响训练稳定性。
- 对 extremely long context 或 highly structured output（如 JSON schema）的效果尚未验证。
- GRPO 需要生成多个 speculative path，在训练阶段带来一定计算开销。

### 未来工作方向
- 将 RCSD 扩展至 **multi-modal speculative decoding** 场景。
- 探索 **online continual learning** 机制，使 draft model 能随部署环境动态更新。
- 结合 **inference-time adaptation** 方法（如调整 k-step），与 RCSD 形成正交优化组合。
- 研究更轻量化的 GRPO 变体，降低训练成本。

---

> ✅ **一句话总结**：  
> RCSD 通过引入 DPO 与 GRPO，将 draft model 的训练从“模仿分布”转变为“追求高接受率”，实现了 **35% 的 acceptance length 提升** 和 **1.35× 的端到端加速**，同时保持生成质量，为高效 LLM 推理提供了新范式。

</details>

---

### 3. [UlaCache: Unified Logical Addressing for Sparse LLM Serving](https://arxiv.org/abs/2610.29820)

**Authors**: Isaac Zhu, Alan Cao, Hugh Yin, Rocky Wang, Orion Wang, Hscos Zhang, Jack Li, Jason Zhao  
**Category**: cs.DC  
**Published**: 2026-10-22  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2610.29820v1  

#### Abstract
Long contexts increase the memory cost of the key-value (KV) cache in large language model (LLM) inference. Sparse attention reduces attention computation and memory traffic by selecting historical tokens, but the retained KV cache must remain available for later decoding steps. We present UlaCache,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：UlaCache: Unified Logical Addressing for Sparse LLM Serving**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在长上下文大语言模型（LLM）推理中，**Key-Value (KV) 缓存的内存开销随上下文长度和并发请求数量急剧增长**，成为限制解码吞吐量的关键瓶颈。尽管稀疏注意力（Sparse Attention）能减少每步计算的 KV 数据量，但历史 KV 仍需保留以供后续步骤访问，导致缓存存储压力依然巨大。

现有方法如 HiSparse 虽利用 CPU 内存扩展 KV 存储，但存在以下问题：
- **GPU-CPU 存储冗余**：HiSparse 在 CPU 中保存完整历史，在 GPU 中缓存副本，未联合利用两者容量。
- **固定缓存配额**：每个请求的 GPU 缓存大小固定，无法动态共享空闲 GPU 容量。
- **跨设备管理开销高**：LRU 替换、地址映射更新等操作串行处理，增加延迟。

### **提出的新方法与创新点**
UlaCache 提出了一套基于**统一逻辑地址空间（Unified Logical Address Space）** 的 KV 缓存管理系统，核心创新如下：

#### ✅ **1. 统一逻辑 KV 地址空间**
- 将 GPU 和可选的 CPU 内存池抽象为一个全局逻辑地址空间。
- 推理引擎通过稳定的逻辑地址分配和访问 KV，物理位置由 UlaCache 动态管理。
- 支持 **Data-Parallel (DP)** 和 **Tensor-Parallel (TP)** 模式下的统一接口。

#### ✅ **2. 分布式存储组织优化**
- 在 TP 模式下，KV 历史分布式存储于多个 GPU 上，避免 Multi-head Latent Attention (MLA) 引起的跨 rank 复制。
- 引入 **本地 GPU buffer** 缓存已获取的 KV，仅对缺失项进行传输，减少重复数据移动。
- 对共享稀疏选择的层支持预取（prefetching），提升流水效率。

#### ✅ **3. 并行桶化 LRU（Parallel Bucketed LRU, PB-LRU）**
- 设计 PB-LRU 替代传统串行 LRU，将 GPU slot 划分为“最近性桶”（recency buckets），实现并行命中检测、热度更新与替换决策。
- 使用分层位图（hierarchical bitmaps）快速定位空闲或待替换 slot，降低管理开销。
- 映射更新延迟至数据传输完成后，保证逻辑地址一致性。

### **相比现有方法的优势**
| 特性 | HiSparse | UlaCache |
|------|--------|---------|
| GPU-CPU 协同 | 副本式缓存（无额外容量） | 统一逻辑空间，联合扩容 |
| 替换策略 | 串行 LRU，高开销 | 并行 PB-LRU，低延迟 |
| 地址管理 | 物理绑定 | 逻辑抽象，稳定引用 |
| 存储效率 | 高复制率（尤其 TP） | 减少跨 GPU 与 CPU 冗余 |
| 动态资源利用 | 固定 per-request 配额 | 共享 GPU 池，按需分配 |

---

## **2. 核心实验方法和设置**

### **数据集**
- 使用 **Multi-LexSum** 数据集生成请求。
- 输入长度范围：**64K–128K tokens**。
- 输出长度：每个请求生成 **2,048 tokens**（greedy decoding）。
- 请求规模：
  - H200：256 请求（平均 98,125 输入 tokens）
  - B300：最多 578 请求（扩展冻结顺序以测试高并发）

### **实验平台**
- **硬件配置**：
  - `8×H200` 和 `8×B300` GPU 集群
  - CPU 内存预算：H200 为 128GB/GPU，B300 为 192GB/GPU
- **模型**：GLM-5.3（FP8 权重与 KV，BF16 激活）
- **集成框架**：SGLang

### **评估指标**
- **输出 token 吞吐量（Output Throughput, tokens/s）**
- **每输出 token 时间（TPOT, ms/token）**
- **峰值吞吐速度提升（Peak Throughput Speedup）**
- **有效 KV 容量与接纳并发数（Admitted Concurrency）**
- **GPU 调度时间与命中率（用于消融实验）**

### **基线方法对比**
1. **SGLang (GPU-resident)**：KV 完全驻留 GPU，无 CPU 扩展。
2. **HiSparse**：当前最优稀疏注意力服务系统，CPU 存储完整历史，GPU 缓存固定副本。

所有系统在相同 GPU/CPU 内存预算下比较，支持 DP 与 TP 模式，并启用/禁用 speculative decoding（MTP，两步 draft）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 配置 | UlaCache 相对于 HiSparse 的峰值吞吐提升 |
|------|-------------------------------|
| **8×H200, DP + Spec. Decoding** | **1.16×** |
| **8×H200, TP + Spec. Decoding** | **1.70×** |
| **8×B300, DP + Spec. Decoding** | **1.73×** |
| **8×B300, TP + Spec. Decoding** | **2.51×** |

> 🔥 在 8×B300 TP 场景下达到 **2.51 倍**于 HiSparse 的峰值吞吐。

此外：
- 在 8×B300 上，UlaCache 的 TP 模式无 speculative decoding 时，吞吐已达 SGLang 的 **3.51×**。
- 更高的并发接纳能力：例如在 8×B300 TP 下，UlaCache 可容纳 **578 请求**，显著高于基线。

### **与基线方法对比结果**
- **低负载延迟（BS=1）**：
  - UlaCache 略慢于 SGLang（因引入调度开销），但快于 HiSparse。
  - 表明其在轻载时仍保持竞争力。
- **高并发吞吐扩展性**：
  - UlaCache 延迟增长更平缓，吞吐持续上升至更高 batch size。
  - 得益于更大的有效 KV 容量和动态缓存共享机制。

### **消融实验结果**
#### 📊 **PB-LRU vs. Naive LRU（图9 & 图10）**
- **调度时间**：在大 batch 下，PB-LRU 比朴素 LRU 快 **1.5–2.3×**。
- **命中率差异 < 0.1%**，说明桶近似不影响缓存有效性。
- 在低替换压力下，朴素 LRU 更快（PB-LRU 有元数据维护开销）。

#### 📈 **命中率分析**
- UlaCache 命中率始终高于 HiSparse，尤其在中等并发时优势明显。
- 原因：HiSparse 的固定 per-request 缓存无法利用其他请求释放的 GPU 空间；而 UlaCache 实现全局共享池。

---

## **4. 关键结论和发现**

### **主要发现**
1. **统一逻辑地址空间是高效稀疏 KV 管理的关键抽象**：
   - 解耦逻辑引用与物理布局，使系统可在 GPU/CPU 间自由迁移 KV 而不影响推理逻辑。
2. **PB-LRU 显著降低管理开销**：
   - 并行化设计适配现代 GPU 架构，特别适合高并发场景。
3. **联合利用 GPU 与 CPU 容量可大幅提升并发服务能力**：
   - 不只是“备份”，而是真正扩展了解码批处理规模。
4. **TP 模式受益最大**：
   - 因原有多头复制问题严重，UlaCache 的分布式存储+本地 buffer 架构极大缓解了这一瓶颈。

### **方法的局限性**
- **额外元数据开销**：PB-LRU 的桶位图和层级 summary 需要一定内存。
- **跨设备通信开销仍存在**：虽然减少了重复传输，但在极端稀疏模式下仍可能受限于 PCIe/NVLink 带宽。
- **依赖稀疏注意力模式的时间局部性**：若选择高度随机，则缓存效益下降。
- 当前实现基于 SGLang，通用性有待在更多推理框架验证。

### **未来工作方向**
- 支持 **disk-offloading** 进一步扩展 KV 存储层次。
- 结合 **预测性预取**（predictive prefetching）提升冷启动性能。
- 探索 **异构设备间更智能的迁移策略**（如基于访问频率/重要性分级）。
- 将统一逻辑地址思想推广到 **activation 或 gradient 缓存管理**。

---

> ✅ **总结一句话**：  
> **UlaCache 通过统一逻辑地址空间 + 并行 PB-LRU 替换机制，在稀疏注意力场景下实现了高达 2.51× 的解码吞吐提升，突破了传统 KV 缓存系统的容量与管理瓶颈。**

</details>

---

### 4. [E8-QAT: Efficient 2-Bit Quantization-Aware Training with Implicit Lattice Codebooks](https://arxiv.org/abs/2610.27699)

**Authors**: Xinyu Lei, Hong Huang, Jiebin Zhang, Hongyu Li, Jinlong Song, Decheng Wu, Guanghua Yu, Jianchen Zhu, Sujian Li  
**Category**: cs.AI  
**Published**: 2026-10-22  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2610.27699v1  

#### Abstract
Low-bit quantization reduces the memory footprint of large language models, but preserving accuracy with efficient training remains challenging. Scalar quantization encodes weights independently, while explicit-codebook vector quantization introduces costly search during training. We propose E8-QAT,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**E8-QAT: Efficient 2-Bit Quantization-Aware Training with Implicit Lattice Codebooks**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
低比特量化（尤其是2-bit）在压缩大语言模型（LLM）内存占用方面具有巨大潜力，但面临两大挑战：
- **Post-training quantization (PTQ)** 在极低位宽下精度损失严重；
- **Vector Quantization-aware Training (VQ-QAT)** 虽能恢复精度，但依赖显式的codebook存储与搜索，在训练前向传播中引入高昂计算开销，且高维VQ因codebook指数增长而难以扩展。

现有方法在**训练效率**与**量化精度**之间存在权衡。

---

### 🚀 提出的新方法：E8-QAT
提出一种高效的2-bit **Quantization-Aware Training (QAT)** 框架——**E8-QAT**，其核心思想是：

#### （1）使用 **E8 lattice** 作为隐式八维codebook
- 利用数学上最优的 **E8 lattice** 结构进行联合权重分组量化（每组8个weight）；
- 不需要显式存储或搜索codebook，而是通过**算术投影**（rounding + parity correction + candidate comparison）直接将权重向量映射到最近的E8 lattice点；
- 投影复杂度为 $O(d)$，独立于名义codebook大小，远低于传统 $O(d \cdot K)$ exhaustive search。

#### （2）训练-部署分离设计 + 稀疏扩展编码（Sparse Extension）
- **训练时**：在无限E8 lattice空间中自由投影，无有限core限制；
- **导出时**：结合一个**2-bit核心编码**（dense core index）和**稀疏逃逸记录**（sparse escape records），精确保留所有训练选中的lattice点；
- 即使某些投影落在core之外，也能以极小额外开销（<0.1%）实现**无损导出**。

#### （3）Identity Straight-Through Estimator (STE)
- 在反向传播中使用identity STE处理硬量化操作，允许latent weights跨越Voronoi边界更新，提升优化灵活性。

---

### 🔍 相比现有方法的优势
| 方面 | E8-QAT | LC-QAT / PV-Tuning | Scalar QAT |
|------|--------|---------------------|------------|
| 是否需codebook搜索 | ❌ 否（算术投影） | ✅ 是（LC-QAT虽免查表但仍受限） | ✅ 是（per-coordinate） |
| 是否支持联合量化 | ✅ 是（8D joint） | ⚠️ 有限（LC-QAT实为coordinate-wise） | ❌ 否 |
| 训练效率 | 高（无search overhead） | 中等至低（有search成本） | 高 |
| 编码灵活性 | ✅ 支持core外投影 + 精确导出 | ❌ 通常限于finite core | ✅ |
| 存储开销增量 | <0.1% | — | — |

> ✅ **核心优势**：实现了**无需codebook搜索的联合量化**，兼顾高精度与高效训练，并支持**精确导出**。

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **基础模型**：
  - `Qwen3.6-35B-A3B`（MoE架构，35B参数）
  - `Llama-3.1-8B`（dense架构，8B参数）
- **训练数据**：UltraData集合
  - Stage 1（PTQ校准）：1,680条 × 16,384 token（来自UltraData-SFT-Agent）
  - Stage 2（QAT训练）：2,000条长序列，共约5080万tokens/epoch
- **上下文长度**：最大达61,959 tokens，配置窗口65,536

---

### 🧪 实验设置
- **量化粒度**：对FFN层expert权重进行2-bit量化（占总参数92.1%）
- **Hadamard预处理**：使用block-Hadamard变换（block size=128）混合权重分布
- **Finite Core构造**：取E8 lattice前 $2^{16}=65,536$ 个低范数点作为核心codebook
- **训练步数**：
  - Qwen：750 steps（主质量评估），375 steps（效率对比）
  - Llama：375 steps
- **学习率调度**：warmup-stable-decay，峰值学习率 $10^{-4}$（按lattice空间归一化调整）

---

### 🎯 评估指标
| 类型 | 指标 |
|------|------|
| **通用能力** | ARC-C, ARC-E, HellaSwag, PIQA, WinoGrande, BoolQ → 平均准确率 |
| **长文本/代理任务** | LiveCodeBench (pass@1), BFCL v3 (function calling), IFEval (instruction following), GPQA (科学问答) |
| **生成质量** | WikiText perplexity（仅Llama） |
| **推理效率** | 峰值内存、最大batch size、解码吞吐量（throughput） |
| **重建质量** | Layer-wise output distortion（stage-1） |

---

### 🆚 基线方法对比
分为两类：
#### （1）Post-Training Quantization (PTQ)
- RTN（Round-To-Nearest）
- GPTQ
- UniSVQ（scalar PTQ，用于初始化LC-QAT）
- AQLM（additive VQ，单codebook共享）
- QuIP#, QTIP（vector PTQ）

#### （2）Quantization-Aware Training (QAT)
- PV-Tuning（联合优化codebook与assignment）
- LC-QAT（当前SOTA，基于affine映射的“伪”VQ）

> 所有对比均在相同数据、schedule、bit width下进行。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Qwen3.6-35B-A3B）

| 方法 | 四任务平均 | IFEval | LiveCodeBench | 六任务平均 |
|------|-----------|--------|---------------|------------|
| BF16（Full Precision） | 73.93 | 85.40 | 81.42 | 75.49 |
| LC-QAT（SOTA baseline） | 73.12 | 80.96 | 78.58 | 75.97 |
| **E8-QAT（本文）** | **73.93** | **82.99** | **80.09** | **76.95** |

> ✅ **E8-QAT四任务平均匹配BF16水平**，显著优于LC-QAT：
- **IFEval ↑2.03%**
- **LiveCodeBench ↑1.51%**
- **六任务平均↑1.0个百分点**

---

### 💡 Llama-3.1-8B 上的结果
| 方法 | 六任务平均 | WikiText PPL |
|------|-----------|-------------|
| BF16 | 75.91 | 9.01 |
| LC-QAT | 65.87 | 16.73 |
| **E8-QAT** | **67.89** | **13.77** |

> ✅ 在dense模型上仍保持领先，**平均高出LC-QAT 2.99%**，**PPL降低近3点**。

---

### ⚙️ 推理效率（vs BF16）
| 指标 | E8-QAT vs BF16 |
|------|----------------|
| 峰值内存 | ↓80% |
| 最大batch size | ↑4× |
| 解码吞吐量（最大batch下） | ↑**1.97×** |

> ✅ 实现接近两倍的服务吞吐量提升。

---

### 🔬 消融实验与关键分析

#### （1）Stage-1重建质量（输出失真度）
- **E8-PTQ** 在40层中有39层 distortion最低；
- 平均distortion比UniSVQ低 **20.4%**，比AQLM低 **19.4%**；
> ✅ 表明E8 lattice提供更优的初始量化起点。

#### （2）QAT第二阶段增益
- 对E8-QAT pipeline：
  - LiveCodeBench ↑24.36%
  - GPQA ↑17.67%
> ✅ 验证QAT对长文本生成能力的关键作用。

#### （3）稀疏扩展开销极低
- 在750步终点，**escape rate = 8.556×10⁻⁵**
- 稀疏记录仅增加 **0.086%** 到2-bit index payload（即总开销≈2.0017 bits/weight）
> ✅ “exact export”代价可忽略不计。

#### （4）学习率敏感性
- 将peak LR从 $10^{-4}$ 降至 $5×10^{-5}$，escape rate下降 **40.7倍**
> ✅ 可控调节训练强度以控制逃逸数量。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **E8 lattice作为隐式codebook可行且高效**：
   - 无需存储/搜索，即可实现高质量联合量化；
   - 算术投影保证训练效率，适合大规模QAT。

2. **E8-QAT在2-bit下达到甚至超越BF16性能**：
   - 在Qwen上四任务平均与BF16持平；
   - 多项agent任务显著优于SOTA（如IFEval +2.03%）。

3. **训练-导出分离 + 稀疏扩展实现“零误差导出”**：
   - 允许训练跳出finite core，同时以<0.1%额外开销精确保存结果。

4. **推理效率大幅提升**：
   - 内存↓80%，吞吐↑1.97×，具备强工程落地价值。

---

### ⚠️ 局限性（Limitations）
1. **评估范围有限**：
   - 当前仅测试Qwen（MoE）和Llama（dense）两种架构；
   - 缺少更多模型规模（如70B+）、领域（如多模态）验证。

2. **仅量化weights**：
   - 激活（activations）和KV Cache仍为高精度（如BF16）；
   - 未探索端到端全模型低比特化。

3. **依赖Hadamard预处理**：
   - 性能增益部分来源于输入变换，可能影响部署兼容性。

4. **硬件适配尚未深入**：
   - 当前推理结果基于fused decode kernels；
   - 在不同硬件平台（如移动端、TPU）的表现待测。

---

### 🔮 未来工作方向
1. **扩展至activation与KV cache量化**，实现全模型超低比特部署；
2. **探索其他lattice结构**（如D4, Leech lattice）在不同维度下的适用性；
3. **自动化escape rate控制机制**，动态平衡精度与存储；
4. **跨架构泛化研究**，构建统一的lattice-based QAT框架；
5. **定制化硬件支持**，开发专用decode unit以进一步加速。

---

> 🏁 **总结一句话**：  
> **E8-QAT首次实现了无需codebook搜索的高效2-bit VQ-QAT，通过E8 lattice的几何优势与稀疏扩展编码，在保持训练效率的同时达到了媲美BF16的精度，并显著提升推理吞吐，为LLM低比特训练提供了新范式。**

</details>

---

### 5. [TrainBridge: Agentic NPU Training Optimization Guided by GPU Compilation Evidence](https://arxiv.org/abs/2610.29171)

**Authors**: Yupeng Chen, Kunxiang Zhao, Xingjian Wang, Yue Zhao, Yijing Wu, Ye Liu, Kai Li, Congliang Chen, Ruoyu Sun  
**Category**: cs.CL  
**Published**: 2026-10-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.29171v1  

#### Abstract
We study how to improve training efficiency on Ascend NPUs. In the NPU training programs we study, framework-level compilation optimizations such as operator fusion are only partially applied, leaving fragmented computation and avoidable execution overheads. Profiling-guided tuning can address these...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*TrainBridge: Agentic NPU Training Optimization Guided by GPU Compilation Evidence*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
在基于 **Ascend NPU** 的训练程序中，现有的编译栈（compilation stack）对算子融合（operator fusion）、计算图优化等框架级优化支持不充分，导致执行过程中存在：
- 计算碎片化（fragmented computation）
- 冗余操作（redundant operations）
- 中间结果显式物化（materialized intermediates）

这些低效问题限制了NPU的训练效率。传统的 **profiling-guided tuning** 虽然可用，但依赖反复试错，优化过程耗时且效率低下。

### ✅ 提出了什么新方法或新思路
提出 **TrainBridge** —— 一种两阶段的 **agentic 框架**，用于指导 NPU 训练程序的优化，其核心思想是：

> 利用成熟的 **GPU 编译堆栈**（如 PyTorch `torch.compile`）产生的优化行为作为“任务特定”的参考信号，来指导 NPU 上的程序重写。

#### 两个阶段：
1. **Stage 1: GPU 编译证据引导的 NPU 重写**
   - 分析同一任务在 GPU 上 **eager 执行 vs. compiled 执行** 的差异
   - 提取其中体现的 **优化意图**（optimization intent），例如算子融合、冗余消除、内存访问重组
   - 将这些意图映射到 NPU 程序中的对应区域，并生成 NPU-specific 的优化改写（如 API 参数调整、局部子图融合等）

2. **Stage 2: 迭代式 NPU 侧 profiling 优化**
   - 对 Stage 1 输出的优化后程序进行实际 profiling
   - 发现剩余瓶颈（如通信开销、host-device 调度延迟等 GPU 证据无法覆盖的问题）
   - 进一步迭代优化，直到性能收敛或预算耗尽

### ✅ 相比现有方法的优势
| 方法 | 局限性 | TrainBridge 的优势 |
|------|--------|------------------|
| 通用 coding agent（如 GPT-5.6-SOL） | 缺乏任务感知能力，搜索空间大，改进率低 | 引入 **GPU 编译证据** 作为强先验，显著缩小优化搜索空间 |
| 单纯 profiling-guided agent | 需要多次 trial-and-error，效率低 | Stage 1 提供“冷启动”加速，快速获得显著性能提升 |
| 手动专家调优 | 成本高、不可扩展 | 自动化 agentic 流程，可复用、可迁移 |

> 🔑 **核心创新**：首次将 **源端（GPU）编译优化行为** 视为“优化证据”（optimization evidence），实现跨硬件平台的知识迁移。

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集与工作负载
共评估五类训练任务，涵盖多种模型和场景：

| 工作负载 | 描述 |
|--------|------|
| **Modded-NanoGPT** | 基于 NanoGPT 的语言模型预训练速度赛，目标是在 8×H100 上达到验证损失 < 3.28 |
| **CIFAR-10 AirBench96** | 图像分类任务，目标准确率 ≥96%，强调训练速度 |
| **Qwen3-0.6B + Hugging Face Trainer** | 大语言模型微调任务 |
| **ViT + TIMM** | Vision Transformer 图像分类训练 |
| **nanoVLM** | 视觉语言多模态模型训练 |

特别地，在 **Modded-NanoGPT** 中选取了历史排行榜前 50 名中的 **27 个记录** 作为独立调优任务，每个代表不同优化策略的历史实现。

### ✅ 实验设置
| 组件 | 配置 |
|------|------|
| **GPU 平台** | 8× NVIDIA H100 SXM，PyTorch 2.7.0，CUDA 12.8 |
| **NPU 平台** | 8× Ascend 910C，PyTorch 2.7.1，CANN 8.5.0，`torch-npu` |
| **初始迁移工具** | Kerminal（用于功能等价的 GPU → NPU 移植） |
| **Agent Harness** | Codex agent harness + GPT-5.6-SOL（medium reasoning effort） |
| **调优预算** | Modded-NanoGPT: 最长 18 小时；其他任务: 8 小时 |
| **评估指标** | 端到端训练时间（end-to-end training time），排除调优时间 |

### ✅ 基线方法对比
- **Generic Coding Agent**：仅使用 GPT-5.6-SOL 进行代码修改，无额外引导
- **Stage 2 Only**：仅使用 NPU profiling 指导优化（即 TrainBridge 去掉 Stage 1）
- **Initial NPU Port**：原始迁移版本，未经任何优化

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### 📊 在 **27 个 Modded-NanoGPT 历史记录** 上的结果（表3 & 图4）
| 指标 | 结果 |
|------|------|
| **平均 NPU 加速比**（vs. 初始实现） | **1.95×** |
| **平均 NPU/GPU 时间比**（优化前 → 优化后） | 从 **4.0× → 2.0×** |
| **最快单例加速比**（Record 17） | 达到 **2.6×** 加速 |

> 💡 表明 TrainBridge 在高度优化的 GPU 实现基础上，仍能有效提升 NPU 性能。

#### 📊 在 **四个额外工作负载** 上的结果（图5）
| 工作负载 | 加速比（vs. 初始 NPU） |
|--------|---------------------|
| CIFAR-10 AirBench | **2.22×** |
| Qwen3 Trainer | **1.68×** |
| ViT/TIMM | **1.25×** |
| nanoVLM | **1.84×** |
| **平均加速比** | **1.7×** |

> ⚠️ 注意：尽管硬件上 Ascend 910C 的 BF16/FP16 吞吐约为 H100 的 76%，但初始实现的端到端差距高达 **2.8×~4.4×**，说明软件优化空间巨大。

### ✅ 与基线方法的对比结果（图2）
在 **Modded-NanoGPT Record 50** 上：
- **Generic Agent**（GPT-5.6-SOL）平均最佳训练时间：**419.2 秒**
- **TrainBridge**：**283.2 秒**（↓32.4%）
- 方差更低，结果更稳定

> 表明引入 GPU 编译证据显著提升了 agent 的优化效率和一致性。

### ✅ 消融实验结果（图3，RQ2）
比较以下两种路径：
- **TrainBridge（完整流程）**
- **Stage 2 Only（仅 profiling 引导）**

| 阶段 | 训练时间变化 |
|------|-------------|
| 初始 NPU 实现 | 563.0 秒 |
| Stage 1 完成后 | ↓至 **299.7 秒**（约 2 小时内完成） |
| Stage 2 完成后 | ↓至 **285 秒** |
| Stage 2 Only（无 Stage 1） | 仅降至 ~334 秒，未突破 300 秒 |

> ✅ **结论**：
> - Stage 1 提供了 **快速、大幅度的初期增益**
> - Stage 2 可进一步挖掘剩余潜力
> - 两者互补，缺一不可

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **GPU 编译堆栈的优化行为是宝贵的“优化证据”**  
   - `torch.compile` 在 GPU 上自动完成的融合、消除、重组等操作，揭示了具体 workload 下的有效优化策略。
   - 这些策略在 NPU 上往往未被复现，构成可优化空间。

2. **跨平台优化知识可以迁移**  
   - 即使 backend 不同，高层语义一致的计算模式仍可共享优化意图。
   - TrainBridge 成功实现了从 GPU 编译证据到 NPU 程序重写的映射。

3. **两阶段设计高效且鲁棒**  
   - Stage 1 快速“热启动”，避免盲目搜索
   - Stage 2 补足 target-specific 开销（如调度、通信），形成闭环

4. **在多个 workload 上均取得显著加速**  
   - 平均 **1.7×~2.0×** 的 NPU 加速，证明方法具有广泛适用性

### ✅ 方法的局限性
1. **依赖源端 GPU 实现有高质量编译优化**  
   - 若 GPU 端本身未充分优化，则无法提供有效指导
   - 当前方法适用于 **GPU-to-NPU migration** 场景，难以用于“冷启动”任务

2. **目前聚焦单节点训练**  
   - 多节点下的通信拓扑、并行策略等复杂因素尚未纳入优化范围

3. **自动化程度受限于 agent 推理能力**  
   - 仍需 LLM agent 正确理解计算语义和优化意图

### ✅ 未来工作方向
1. **构建优化知识库（Retrieval Database）**  
   - 存储历史成功的 “GPU 编译差异 → NPU 改写” 映射
   - 支持跨任务检索与复用，提升 zero-shot 能力

2. **扩展至多节点分布式训练**  
   - 结合 MindSpeed-LLM、Megatron-LM 等框架，优化跨节点通信与并行策略

3. **支持更多 backend 间的迁移**  
   - 如 GPU → TPU / GPU → IPU 等，推广为通用的 **cross-accelerator optimization framework**

4. **结合编译器 IR 层面分析**  
   - 不局限于 computation graph，深入 Compiler IR 和 kernel generation 层面进行对齐与引导

---

> 🔚 **总结一句话**：  
> **TrainBridge 通过“借力打力”的方式，利用 GPU 上成熟的编译优化经验，作为指导信号来高效优化 NPU 训练程序，在多个任务上实现了接近 2× 的加速，为异构加速器上的训练优化提供了新范式。**

</details>

---

### 6. [Understanding Target Probability Reuse in Speculative Decoding](https://arxiv.org/abs/2610.29210)

**Authors**: Jungseob Lee, Giyoon Jang, Chailin Nam, Jiwoo Kum, Sihyeon Lee, Kyojun Choo, Jihun Song, Eunseo Song, Chanjun Park  
**Category**: cs.CL  
**Published**: 2026-10-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.29210v1  

#### Abstract
Speculative decoding uses a target model to verify a tree of draft tokens in parallel. We study whether the resulting next-token distributions can help select the next tree. For unrestricted target models, we show that guaranteed exact reuse requires a distribution computed at the next round's compl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **Understanding Target Probability Reuse in Speculative Decoding** 论文总结

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文研究了在 **Speculative Decoding** 中如何更高效地利用 **Target Model** 的概率输出来优化 **draft tree** 的构建过程。传统方法中，tree 的选择通常在 Target 验证之前完成，因此无法利用 Target 提供的精确概率分布。这导致 draft tree 可能包含大量低概率路径，从而降低验证阶段的 **expected acceptance**。

本文提出一个核心问题：  
> 是否可以在不进行额外 Target 推理的情况下，复用前一轮计算得到的 Target 概率（即 “rows”）来指导下一棵树的选择？

作者系统分析了这种“reuse”的可行性，并揭示了其理论限制与潜在收益。

---

### **提出了什么新方法或新思路**

#### **(1) 地址匹配（Address Matching）与精确复用准则**
- 定义了 **exact address coverage**：只有当上一轮计算的某一行（row）是在当前轮完整上下文（complete context）下计算时，才能保证其概率分布完全等价于当前轮所需的 root row。
- 揭示了仅通过 token 匹配（如 Token Recycling）无法保证精确复用，因为不同路径可能以相同 token 结尾但上下文不同。

#### **(2) 新的树选择策略**
提出了两种基于 **提前获取 root row** 的树选择规则：
- **Root Selection**：使用 Target 在当前 context 下计算出的第一个 token 的真实概率（即 root row），替换 draft ranking 中第一个 token 的 draft 概率，其余仍用 draft 概率排序。
- **Root-draw Selection**：先从 root row 中采样第一个 token，再围绕该 token 构建子树（保留其 B−1 个最高分后代）。相比 PEARL-style 的剪枝方式更具灵活性。

#### **(3) 分离式推理流程（Separate Root Pass）**
为了获得 root row，引入了一个独立的、轻量级的 Target 推理步骤（one-token pass），在 drafting 过程中并行执行。这种方式使得 root row 可在 tree allocation 前可用。

---

### **相比现有方法的优势**

| 方法 | 优势 |
|------|------|
| **vs. Draft Ranking** | 利用 Target 概率提升 first-token 分支质量，显著提高 expected acceptance，尤其在小树预算下效果明显 |
| **vs. PEARL-style Pruning** | Root-draw selection 主动生成以高概率 token 为根的子树，而非被动剪枝，保留更多有效节点 |
| **vs. Cache-based Drafting** | 不依赖历史缓存中的近似匹配，而是通过分离 pass 获取精确 row，确保决策准确性 |

此外，在实际部署中，由于 **standalone drafting** 和 **separate root pass** 可并行运行，该方法还能实现端到端解码加速。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Spec-Bench**：包含 480 个 prompt，覆盖 7 种任务类型：
  - MT-bench
  - Translation
  - Summarization
  - QA
  - Mathematical Reasoning
  - RAG
  - HumanEval
- **HumanEval**：164 个编程函数签名，用于代码生成评估

共测试 **644 个 prompts**，每个目标模型生成最多 256 个新 token（部分实验扩展至 1024）。

---

### **实验设置和评估指标**

#### **Target Models（6个）**
- `Llama-3.1-8B-Instruct`
- `Qwen3-4B`, `Qwen3-8B`
- `Qwen2.5-7B`, `14B`, `32B-Instruct`

均搭配对应的 **EAGLE-3 drafter**，使用默认 chat template。

#### **Tree Construction**
- Draft Pool：深度 1–8，共 710 个候选
- Verified Tree：取 top-59 非根节点（+ root 共 60 节点）
- Verification Strategy：**exhaustive target sampling**

#### **评估指标**
- **Expected Accepted Draft Tokens (V(S))**：衡量每轮平均接受的 draft token 数量，是核心效率指标
- **Decoding Speed**：单位时间内生成的 token 数量
- **Coverage Rate (Pr(C))**：是否命中 exact address
- **Total Variation Distance**：比较两个概率分布差异
- **Oracle Upper Bound**：理想情况下的最大 V(S)

#### **对比方法**
- **Draft Ranking**：baseline，仅使用 drafter 概率排序
- **Root Selection**：用 root row 替换 first-token 概率
- **Root-draw Selection**：先抽样 root token，再构建子树
- **PEARL-style Decoder**：先构建固定树，后根据 root draw 剪枝
- **Oracle**：使用真实 Target 概率对所有路径排序，理论上最优

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **Table 2：Expected Acceptance 提升显著**

| 方法 | 平均提升（B=59） | 小树增益（B=8） |
|------|------------------|----------------|
| Root Selection vs Draft | **+3.5% ~ +6.7%** | **+10.0% ~ +14.2%** |
| Root-draw vs PEARL-style | **+2.4% ~ +7.0%** | **+7.7% ~ +9.0%** |

> 所有增益均有统计显著性（95% CI > 0）

#### ✅ **Table 3 & Table 16：在线解码性能验证**
- 在线运行 **separate root pass + root-draw selection**：
  - **Llama-3.1-8B**：acceptance 达 **4.842**（vs EAGLE-3 的 4.509）
  - **Qwen3-8B**：acceptance 达 **2.606**（vs 2.221）
- 相比 PEARL-style decoder，gain 达 **3.3% ~ 7.3%**

#### ✅ **Table 4：Root Row 复用价值**
- 即使只替换第一个 token 的概率（Root Selection），也能稳定优于 draft ranking 和其他 reranking 方法。
- Root Selection > Base/Reranked Models > Draft Ranking

#### ✅ **Table 5 & Table 20：长序列持续有效**
- 在输出长度达 1024 的情况下，root selection 依然保持增益：
  - B=59：+2.7% ~ +7.6%
  - B=8：+7.6% ~ +22.6%
- 表明 root row 的指导作用在后期依然有效

#### ✅ **Figure 4：解码速度提升**
- 当 **standalone drafter** 运行在第二块 GPU 上时，root pass 可与其并行：
  - Root-draw selection 解码速度比 draft selection 快 **+3.5% ~ +9.6%**
  - 即便与 PEARL-style 对比，仍快 **+1.3% ~ +2.2%**（在控制平均 tree size 后）

---

### **消融实验结果**

#### 🔍 **Root Selection 增益来源分析（Figure 3 & Table 15）**
- 增益主要来自 **drafter 与 target 第一个 token 不一致** 的情形
- 此类轮次占比约 24%–46%，却贡献了超过 60% 的增益
- 说明该方法特别适用于纠正 draft bias

#### 🔍 **Row 来源有效性对比（Table 23 & 26）**
- **Matched Address Rows**（地址完全匹配）：
  - Mean TV < 0.01
  - Argmax 一致率达 99.2%–99.7%
- **Same-Token Rows Only**（仅末尾 token 匹配）：
  - Mean TV ≈ 0.4
  - Argmax 一致率仅 54%–62%
> 证明 token 匹配不足以保证 row 复用精度

#### 🔍 **Rows at Other Addresses 的间接利用（Table 4 & 25）**
- 直接将非 root 地址的 row 当作 root row 使用会 **降低 acceptance**
- 但将其作为特征输入 reranker 可带来正向增益（+0.12~0.18 tokens）
> 说明这些 rows 虽不能直接复用，但仍具信息价值

---

## **4. 关键结论和发现**

### **主要发现**

1. **Exact Reuse 需要地址完全匹配**
   - 仅 token 匹配无法保证 row 分布等价
   - 缓存机制（如 Token Recycling）存在理论误差风险

2. **提前获取 root row 显著提升 tree quality**
   - Root Selection 在 B=59 下平均提升 **+4.1%** acceptance
   - 在 B=8 下提升高达 **+14.2%**，表明小树更依赖高质量引导

3. **Root-draw Selection 优于剪枝策略**
   - 相比 PEARL-style pruning，主动构造以高概率 token 为核心的子树可多保留 16–18 个节点
   - 即使在控制平均 tree size 后，仍能获得 **+1.1% ~ +2.5%** acceptance 增益

4. **并行化设计带来实际加速**
   - Separate root pass 可与 standalone drafting 重叠执行
   - 最终实现 **+3.5% ~ +9.6%** 的端到端解码加速

5. **增益在多种任务和模型上具有一致性**
   - 所有 6 个 target 模型、7 类任务均观察到正向增益
   - 温度为 1 的采样模式下也保持稳定表现

---

### **方法的局限性**

1. **依赖额外 Target 推理开销**
   - 需要一次 one-token Target 推理，虽然可并行，但在资源受限场景可能不可行

2. **当前实验集中在 batch-1 推理**
   - 多 batch 场景下的通信与调度复杂度未充分探索

3. **未改变 Target 输出分布**
   - 方法仅影响 acceptance 数量，不改变最终文本质量或多样性

4. **硬件假设较强**
   - 加速效果依赖于双 GPU 并行架构（如 PEARL 部署）

---

### **未来工作方向**

1. **降低 root row 获取成本**
   - 设计轻量化 proxy model 或 low-rank approximation 来预测 root row

2. **动态预算分配**
   - 根据 root row 的 entropy 或 uncertainty 动态调整 tree budget

3. **扩展至多步复用**
   - 探索能否安全复用非 root 的中间 rows 来指导深层分支选择

4. **结合 proposer training**
   - 将 root row 信息反馈给 drafter 训练，减少长期分布偏移

5. **应用于 vision-language models**
   - 探索跨模态 speculative decoding 中的概率复用机制

---

> 📌 **一句话总结**：  
> 本文揭示了在 Speculative Decoding 中，**提前获取 Target 的 root row 并用于 tree selection** 是一种简单而高效的优化手段，不仅能显著提升 expected acceptance，还可通过并行化实现端到端加速，为高效 LLM inference 提供了一条实用路径。

</details>

---

### 7. [Engineering Low-Latency, High-Throughput Quantum Decoders on NVIDIA GPUs](https://arxiv.org/abs/2610.28867)

**Authors**: Stefano Markidis, Luca Pennati, Ivy Peng  
**Category**: cs.DC  
**Published**: 2026-10-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.28867v1  

#### Abstract
We design and optimize a quantum error correction (QEC) decoder for NVIDIA GPUs, combining Relay belief propagation, a state-of-the-art message-passing algorithm, with order-zero ordered statistics post-processing. Our implementation uses CUDA Graphs to control GPU execution, stops updates for conve...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Engineering Low-Latency, High-Throughput Quantum Decoders on NVIDIA GPUs**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
量子纠错（**Quantum Error Correction, QEC**）是实现容错量子计算的关键环节，其依赖经典解码器对测量得到的 **syndrome** 数据进行实时处理。随着逻辑量子比特数量增加，生成的 syndrome 数据量急剧上升，要求解码器具备极高的 **throughput** 和极低的 **latency**。

传统方案多采用 FPGA 或定制 ASIC 实现 QEC 解码，虽然高效但缺乏灵活性。本文旨在探索 **NVIDIA GPU** 是否可以作为高性能、可编程的 QEC 解码平台，在保持高吞吐的同时显著降低延迟。

---

### **提出的新方法与创新思路**
作者设计并优化了一个基于 **NVIDIA GPU** 的 QEC 解码器，结合以下关键技术：

- **算法层面**：
  - 采用 **Relay-BP**（Relay belief propagation）作为主迭代解码算法，提升消息传递的准确性。
  - 后接 **OSD-0**（order-zero ordered statistics decoding）作为 fallback 二元求解器，用于处理未收敛样本。

- **系统与架构优化**：
  - 使用 **CUDA Graphs** 实现 GPU 端的控制流管理，避免主机频繁干预，减少同步开销。
  - 引入 **sample retirement** 机制：一旦某样本在迭代中成功恢复 syndrome，则停止对其更新，节省计算资源。
  - 将未解决样本收集到紧凑的 **fallback queue** 中统一送入 OSD-0 处理，提高内存局部性和执行效率。
  - 设计 **multi-warp cooperative binary elimination**，利用 warp-level collectives 加速 GF(2) 上的高斯消元。

- **数值精度优化**：
  - 探索 **mixed FP16-FP32 arithmetic** 和 **INT8 quantization**，大幅减少消息存储和带宽需求。
  - 在关键累加路径使用更高精度（如 INT32 累加 + INT16 存储），保证准确率不下降。

- **并发执行模型**：
  - 支持多个独立的 decoder 实例通过不同 **CUDA stream** 并发运行，充分利用 GPU 资源应对突发负载（burst load）。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **性能** | 显著优于纯 FP32 实现；INT8 量化下 decoding time 减少达 50% 以上 |
| **延迟-吞吐权衡** | 商品级 RTX 5080 可达到 HPC 级 GH200 91% 的峰值吞吐，且中位延迟缩短 8.9 倍 |
| **灵活性** | 完全基于 CUDA 编程模型，支持快速迭代与部署，优于固定逻辑的 FPGA/ASIC |
| **成本效益** | 提出构建由商品 GPU 构成的 QEC decoder cluster 的可行性，更具扩展性和经济性 |

---

## **2. 核心实验方法和设置**

### **使用的数据集（QEC 工作负载）**
评估了三种主流的电路级 QEC 编码方案：
| Code | 参数 | 描述 |
|------|------|------|
| **Surface code** | `[[81,1,9]]` | 旋转表面码，典型拓扑码 |
| **Hypergraph Product (HGP) code** | `[[98,2,7]]` | 高密度奇偶校验码构造 |
| **Bivariate Bicycle (BB90) code** | `[[90,8,10]]` | 高阈值、低开销的 qLDPC 码 |

所有 workload 均通过 **Stim** 仿真器生成 detector error model (DEM)，包含 multi-detector 错误，更贴近真实物理噪声。

---

### **实验设置**
- **硬件平台**：
  - **A100**（Ampere）
  - **GH200**（Hopper）
  - **RTX 5080**（Blackwell，商品级显卡）

| 指标 | 设置说明 |
|------|----------|
| Noise level | $ p_e \in \{0.001, 0.003, 0.005\} $ |
| Batch size | 自动调优范围：32 ~ 2048 |
| Decoder config | 最多 4 legs × 32 iterations，min-sum normalization factor = 0.75 |
| Precision schemes | FP32（baseline）、FP16、mixed FP16-FP32、INT8 |
| 并发实例数 | $ S = 1–8 $，每实例独占 stream 与 buffer |

---

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Amortized decoding time** | 总解码时间 / batch size，单位 μs/sample |
| **Logical Error Rate (LER)** | 解码后仍未能纠正逻辑错误的比例 |
| **Throughput** | 千次解码/秒（kdecodes/s） |
| **Latency** | 从 host 提交到 correction 返回的时间（submission-to-correction latency） |
| **OSD fallback ratio** | 进入 OSD-0 阶段的样本占比 |
| **Kernel profiling** | 使用 Nsight Compute 分析 SM 利用率、DRAM BW、L1/L2 hit rate、warp occupancy 等 |

---

### **基线方法对比**
- 主要对比不同 **precision setting** 下的表现（FP32 vs. mixed precision vs. INT8）
- 不同 **batch size** 和 **stream concurrency** 的组合用于绘制 latency-throughput Pareto frontier
- 无 CUDA Graph 控制 vs. 有设备端条件判断的版本比较控制开销影响

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总（以 BB90, $p=0.003$ 为例）**

| GPU | Precision | Optimal B | Decoding Time (μs) | Speedup vs FP32 | LER (%) | OSD Fallback |
|-----|-----------|------------|----------------------|------------------|---------|---------------|
| RTX 5080 | INT8 | 128 | **207.079** | **50.2% ↓** | 4.73 → 4.73 | — |
| RTX 5080 | FP32 | 128 | 416.x | — | ~4.73 | — |
| GH200 | INT8 | 2048 | 151.154 | 22.6% ↓ | 4.73 | — |
| A100 | mixed | 2048 | 423.136 | 15.4% ↓ | 4.73 | — |

> ✅ **INT8 在 RTX 5080 上将解码时间减半，LER 几乎不变**

---

### **与基线方法的对比结果**

#### **Finding 3: Reduced Precision 是关键优化**
- **INT8 量化** 在 BB90 上带来最高达 **55.6% 的加速**（RTX 5080, $p=0.005$）
- 存储开销从 FP32 的 178 MiB 降至 INT8 的 45 MiB（约 4× 减少）
- 尽管 LER 有轻微上升（如从 5.19% → 5.51%），但在可接受范围内
- **mixed FP16-FP32** 对 A100 更有效，而 **INT8** 在 GH200 和 RTX 上表现更好

#### **Finding 4: Warp-level collectives 提升效率**
- **OSD-0 solver 内部使用 warp cooperative reduction** 较单线程参考实现快 **20–21×**
- 扩展至 8 candidate 并行 reduction 再提速 **2.25×**
- 整体 decoder 时间减少 **8.8%**（BB90, $p=0.005$）

#### **Finding 8 & 9: 并发 + 小 batch 实现最优延迟-吞吐平衡**
| 场景 | 结果 |
|------|------|
| **Surface code, <10ms 响应要求** |  
| GH200 ($B=32,S=4$) | 14.30 kdecodes/s @ 8.94 ms  
| **RTX 5080 ($B=32,S=6$)** | **20.11 kdecodes/s @ 8.94 ms**（**+41% 吞吐优势**）  
| **BB90, <20ms 响应要求** |  
| GH200 | 4.03 kdecodes/s  
| **RTX 5080** | **5.95 kdecodes/s**（**+48% 优势**）  

> ⚡️ **RTX 5080 在短延迟约束下全面超越 GH200**

#### **峰值吞吐对比**
- GH200 在大 batch 下达到最高吞吐：**6.50 kdecodes/s（BB90）**
- RTX 5080 达到 **5.95 kdecodes/s**，为 GH200 的 **91%**
- 但 RTX 中位延迟仅 **17.6 ms**，而 GH200 为 **156 ms**（**慢 8.9×**）

---

### **消融实验结果**
| 优化项 | 移除后的性能损失 |
|--------|------------------|
| **CUDA Graphs + device-side convergence** | 在低噪声下可减少最多 17% 时间（p=0.001）；高噪声下效果有限 |
| **Sample retirement** | 若不提前终止已收敛样本，浪费大量计算资源 |
| **Prefetching & register reuse** | 显著改善 memory-level parallelism，尤其在 Parity/min-sum kernel |
| **Ordered multi-warp OSD-0** | 相比 naive 实现提速超过 20× |
| **INT8 quantization** | 存储减少 75%，带宽压力显著缓解，适合高并发场景 |

---

## **4. 关键结论和发现**

### **主要发现**
1. **商品级 GPU 可胜任高性能 QEC 解码任务**  
   - RTX 5080 在合理配置下可达 HPC 级 GH200 的 **91% 吞吐**，且延迟更低。
   - 证明了构建基于 **commodity GPU clusters** 的 QEC 解码基础设施的可行性。

2. **小 batch + 高并发 是低延迟的关键**  
   - 多个 decoder 实例并行处理小 batch 请求，可在维持高吞吐的同时极大压缩响应时间。
   - 适用于 synchronized QEC measurement 导致的 bursty decoding demand。

3. **混合精度与 INT8 量化极具潜力**  
   - INT8 在保持 LER 几乎不变的前提下，实现高达 **50%+ 的加速**。
   - 特别适合带宽受限或需大规模并发的场景。

4. **CUDA Graphs 提供高效的控制抽象**  
   - 实现了完全在 GPU 上完成迭代控制、样本退休与 fallback 调度，减少了 CPU-GPU 通信开销。

5. **Warp-level collectives 显著加速细粒度操作**  
   - 在 min-sum reduction 和 GF(2) 消元等小规模规约中效果显著。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **OSD-0 仍是瓶颈** | 尽管优化后速度大幅提升，但对于 fallback 比例高的情况（如 $p=0.005$），OSD 仍占主导时间 |
| **通用性依赖图结构复用** | 当前实现假设 batch 内所有样本共享相同的 H 矩阵；若编码动态变化则需重新构建数据结构 |
| **Tensor Core 利用有限** | 尝试使用 1-bit Tensor Core 加速 binary algebra，仅获得 **1.12× 提速**，收益不高 |
| **Blackwell 架构尚未完全释放潜力** | RTX 5080 表现优异可能得益于更高的整数运算吞吐（INT8 add/min），但目前缺乏底层指令级分析 |

---

### **未来工作方向**
1. **进一步优化 OSD 阶段**  
   - 探索分块、稀疏化或近似算法降低其复杂度。
   - 结合预解码器（predecoder）减少进入 OSD 的样本数（参考 qLDPC predecoding）。

2. **支持 streaming / windowed decoding**  
   - 实现跨 time-step 的连续解码，适应实际量子电路中的时序依赖。

3. **异构集群调度策略研究**  
   - 如何在 GPU cluster 中动态分配 decoding tasks，应对 bursty workloads（参见 Maurya et al. 弹性调度思想）。

4. **扩展至更大规模 qLDPC codes**  
   - 测试本文方法在数千变量以上的超大规模稀疏码上的可扩展性。

5. **探索 Transformer-based 或 Neural Decoders on GPU**  
   - 利用 GPU 强大的矩阵计算能力运行基于深度学习的新型解码器。

---

> 🔚 **总结一句话**：  
> 本论文展示了通过 **CUDA Graphs + mixed precision + INT8 quantization + warp collectives + concurrency** 的综合优化，**商品级 NVIDIA GPU** 完全有能力成为下一代 **低延迟、高吞吐 QEC 解码器** 的理想平台，为构建可扩展的容错量子计算机提供了切实可行的经典配套方案。

</details>

---

### 8. [Scalable Stress-Constrained Topology Optimization: A Super-resolution and Simultaneous Approach Based on Machine Learning](https://arxiv.org/abs/2610.28703)

**Authors**: Xiangyu Sun, Shirin Hosseinmardi, Ramin Bostanabad  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2610.28703v1  

#### Abstract
Large-scale topology optimization (TO) remains computationally demanding, especially for stress-constrained problems. To address this challenge, we propose the simultaneous neural topology optimization (SNTO) framework where we represent state, adjoint, and design variables with custom neural networ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**大规模应力约束拓扑优化（Stress-Constrained Topology Optimization, TO）** 中计算成本高昂、难以扩展到数百万自由度（DoF）的问题。传统基于有限元（FE）的方法在处理局部应力约束时面临以下挑战：
- 需要嵌套分析与设计（NAND），每轮迭代需多次求解大型线性系统；
- 局部应力约束数量随网格分辨率增长而急剧增加；
- 内存和通信开销大，尤其在三维（3D）问题中。

### 提出的新方法：SNTO 框架
作者提出了 **Simultaneous Neural Topology Optimization (SNTO)**，一种基于机器学习的并行可扩展框架，其核心创新包括：

#### （1）同时优化（SAND）范式 + 物理信息神经网络（PIML）
- 将位移场 $u$、伴随场 $u^{(a)}$ 和密度场 $p$ 分别用三个独立的 **Parametric Grid Convolutional Attention Network (PGCAN)** 表示。
- 所有变量通过一个统一的损失函数进行**联合优化**，避免了传统 NAND 中反复求解平衡方程的过程。
- 损失函数由四部分组成：
  - 体积分数目标 $C_v(p)$
  - 应力约束二次惩罚项 $P(\cdot)$
  - 位移场势能 $L_M(u)$
  - 伴随场势能 $L_a(u^{(a)})$

#### （2）高效的硬件感知训练策略（Hardware-aware Training）
为实现对**数百万甚至上千万自由度**问题的扩展，提出两级划分机制：
- **跨GPU域划分（Domain Partitioning）**：将设计域划分为多个子域，分布到不同GPU上。
- **子域内切片（Sub-slicing）**：每个子域进一步细分为小区域，按序处理以降低峰值显存占用。
- 利用损失梯度的**加法性质**，无需引入任何接口连续性损失（interface loss terms），保证全场连续性。

#### （3）PGCAN 架构捕捉锐利边界
- PGCAN 能有效缓解神经网络的“频谱偏差”（spectral bias），即难以拟合高频细节的问题。
- 可生成具有**更锐利边界、更少灰度区域**的设计，优于传统滤波方法。

### 相比现有方法的优势
| 维度 | 传统 FE 方法（如 PolyStress） | SNTO |
|------|-----------------------------|-------|
| **参数化方式** | 网格离散节点值 | 连续神经场（mesh-free） |
| **优化范式** | 嵌套（NAND） | 同时（SAND） |
| **内存效率** | 存储全局刚度矩阵 | 无矩阵组装，仅存储NN参数 |
| **扩展性** | 多依赖MPI通信优化 | 天然支持多GPU并行 |
| **设计质量** | 存在较多灰度区 | 更接近二值化，边界更清晰 |

---

## 2. 核心实验方法和设置

### 数据集与测试案例
共使用 **4个2D基准问题** 和 **2个3D扩展问题**，均来自经典拓扑优化文献：

#### 2D Benchmark Cases
| 名称 | 描述 |
|------|------|
| **L beam 2D** | L形梁，含凹角导致应力集中 |
| **Crack 2D** | 半板中心裂纹拉伸，模拟裂尖应力集中 |
| **Hook 2D** | 曲柄钩，带圆孔固定与弧面加载 |
| **Portal frame 2D** | 门式框架，对称建模一半 |

#### 3D Benchmark Cases
| 名称 | 描述 |
|------|------|
| **L beam 3D** | L形梁沿厚度方向拉伸成3D体 |
| **Portal frame 3D** | 门式框架三维实体模型 |

所有材料设为线弹性，$E_0 = 70\,\text{GPa}, \nu = 0.25, \sigma_y^0 = 100\,\text{MPa}$。

### 实验设置
| 参数 | 设置 |
|------|------|
| **背景网格** | 结构化四边形（2D）/六面体（3D）用于数值积分（单高斯点） |
| **Collocation Points (CPs)** | 节点CPs（nodal CPs）和单元中心CPs（centroid CPs） |
| **PGCAN配置** | $N_{\text{rep}}=3$, $N_f=128$, 3层隐藏层，宽度64 |
| **优化器** | Adam ($\eta_u=10^{-3}, \eta_a=10^{-2}, \eta_p=2\times10^{-5}\,\text{(2D)},\,10^{-5}\,\text{(3D)})$ |
| **训练周期** | 30,000 epochs |
| **硬件平台** | 单/多块 NVIDIA A100 GPU（40GB显存） |

### 评估指标
| 指标 | 定义 |
|------|------|
| **Volume Fraction** | 最终设计的平均密度 |
| **Gray Area Fraction** | 密度介于 $[0.1, 0.9]$ 的单元占比 |
| **Wall-clock Runtime** | 总训练时间（秒） |
| **Stress Violation** | $S_{Vsa}$: 固体区域中 $\sigma_{VM} > \alpha_s \sigma_y$ 的比例；$S_{Val}$: 归一化违反总数 |
| **Peak GPU Memory Usage** | 训练过程中的最大显存消耗 |

### 基线方法对比
- **PolyStress**：开源MATLAB实现，采用增强拉格朗日法处理局部应力约束。
- 对比维度：设计拓扑、体积分数、灰度区、应力满足程度、运行时间。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ 2D 结果（250K nodal CPs）
| 指标 | L beam | Crack | Hook | Portal Frame |
|------|--------|-------|------|--------------|
| **SNTO Volume Frac.** | 0.349 | 0.247 | 0.505 | 0.241 |
| **PolyStress Volume Frac.** | 0.365 | 0.258 | 0.579 | 0.272 |
| **SNTO Gray Frac.** | 0.035 | 0.017 | 0.025 | 0.031 |
| **PolyStress Gray Frac.** | 0.087 | 0.054 | 0.094 | 0.128 |
| **Runtime (sec)** | ~2722 | ~2341 | ~3316 | ~2632 |
| **PolyStress Runtime** | ~10705 | ~2559 | ~6789 | ~9023 |

> 📌 **结论**：SNTO 在所有四个案例中均实现了更低的体积分数和显著更小的灰度区，且运行速度更快（约快2–3倍）。

#### ✅ 3D 可扩展性实验（最多8M nodal CPs）
| 案例 | nodal CPs | GPUs | sub-slices/GPU | Runtime (sec) | Gray Frac. | Volume Frac. |
|------|-----------|-------|----------------|---------------|------------|--------------|
| L beam 3D | 500K | 1 | 2 | 7661 | 0.015 | 0.342 |
| | 1M | 2 | 2 | 8912 | 0.013 | 0.347 |
| | 2M | 2 | 4 | 17784 | 0.009 | 0.350 |
| | 4M | 4 | 4 | 22037 | 0.015 | 0.345 |
| | 8M | 4 | 10 | 50532 | 0.015 | 0.351 |
| Portal frame 3D | 8M | 4 | 10 | 62036 | 0.034 | 0.258 |

> 📌 **结论**：成功扩展至 **8 million DoFs**，设计保持稳定，灰度区仍很低。

### 与基线方法对比结果
- **设计质量优势明显**：
  - SNTO 设计边界更锐利，灰度区减少 **2–7倍**。
  - 在 Hook 案例中，SNTO 移除了 PolyStress 中不必要的顶部环状冗余材料，体积分数低近 **13%**。
- **应力控制相当**：
  - 两者最终设计中仅有极少数元素轻微超限（< 0.5%）。
  - 99.9百分位应力比差异小于0.05，说明约束满足水平一致。
- **运行效率更高**：
  - 在相同规模下，SNTO 比 PolyStress 快 **2–3倍**。

### 消融实验与敏感性分析
#### （1）网格分辨率无关性（Mesh Independence）
- 在 125K–1M nodal CPs 范围内，最终体积分数变化 < 0.02，灰度区变化 < 0.03。
- 不同分辨率下的拓扑结构高度一致，表明无需传统滤波即可获得稳定设计。

#### （2）多GPU加速效果
- 使用 **8 GPUs** 并行训练 2M CPs 问题，训练时间从 9.4 小时降至 **1.5 小时**，加速比达 **6.3×**，并行效率 **79%**。
- 加速主要受限于 `all-reduce` 梯度同步操作。

#### （3）Sub-slicing 显存控制
- 将每个GPU上的 sub-slices 数从2增至16，**峰值显存从68%降至14%**。
- 时间代价是训练时间从2.5小时增至4.2小时（因串行处理增多）。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **SNTO 成功实现了大规模应力约束TO的高效求解**，支持高达 **8 million DoFs** 的3D问题。
2. ✅ 采用 **SAND + PIML + PGCAN** 的组合，能够自然地生成**超分辨率、边界锐利、灰度区极少**的设计。
3. ✅ 所提出的**加法型梯度累积机制**无需任何接口损失项，天然保证全场连续性，简化了分布式训练。
4. ✅ 与主流 FE 方法 **PolyStress** 相比，SNTO 在设计轻量化、清晰度和计算效率方面全面占优。
5. ✅ 框架具备良好的**网格无关性和可扩展性**，适用于复杂几何与载荷条件。

### 方法的局限性
1. ❗ 当前仅适用于**线弹性静力学**问题，尚未推广至非线性材料或动力学场景。
2. ❗ 多GPU并行效率受 `all-reduce` 通信瓶颈限制，未达到理想线性加速。
3. ❗ 虽然避免了滤波，但仍依赖背景网格进行数值微分与积分（尽管仅为辅助用途）。
4. ❗ 对 PGCAN 架构有一定依赖，其他NN结构可能无法达到同等精度。

### 未来工作方向
1. 🔮 扩展至**非线性本构关系**（如超弹性、弹塑性）。
2. 🔮 推广至**瞬态问题**，其中应力约束需在整个时间历程上满足，并引入时空伴随分析。
3. 🔮 引入**自适应采样策略**，在应力集中区域动态加密 collocation points，提升约束精度。
4. 🔮 探索**无背景网格的完全自由形式微分计算**，例如结合Fourier神经算子或Radial Basis Functions。
5. 🔮 开发面向工业级应用的集成工具链，支持CAD导出与制造可行性检查。

---

> 💬 **总体评价**：  
> 本文提出的 **SNTO 框架** 是拓扑优化领域向“大规模、高保真、智能化”迈进的重要一步。它不仅解决了长期存在的计算瓶颈，还展示了机器学习在工程设计自动化中的强大潜力。其思想对后续研究具有重要启发意义。

</details>

---

### 9. [ReAPO: Retry-Aware Policy Optimization for Efficient Exploration in LLM Reasoning](https://arxiv.org/abs/2610.28770)

**Authors**: Nghi Dao, Calder Morton-Ferguson, Noah Golowich, Adam Block  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.28770v1  

#### Abstract
RLVR improves language-model reasoning, but commonly relies on independent stochastic rollouts for exploration, which can repeatedly sample similar failed strategies. Prior work has seen success with approaches that guide exploration by introducing dependence into the sampling process by including i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ReAPO: Retry-Aware Policy Optimization for Efficient Exploration in LLM Reasoning 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于可验证奖励的强化学习（**RLVR**）方法，如 **Group Relative Policy Optimization (GRPO)**，在提升大语言模型（LLM）推理能力方面取得了成功。然而，这些方法依赖于独立的随机 rollout 进行探索，这可能导致反复采样相似的失败策略，造成计算资源浪费，并且无法有效从错误中学习。

### 提出的新方法：ReAPO
本文提出了 **Retry-Aware Policy Optimization (ReAPO)**，一种简单而有效的 GRPO 改进算法，旨在通过改进探索和信用分配机制来增强 LLM 的推理能力。

其核心创新点在于两个方面：
1.  **Verifier-Conditioned Exploration (验证器条件化探索)**：
    *   当一次尝试（attempt）被验证器（verifier）判定为错误时，系统会向上下文中添加一个重试指令（如 `"Your answer is wrong. Reflect on your previous answer and try something else."`），然后让模型基于这个包含了先前错误信息的上下文生成下一次尝试。
    *   这种序列化的采样方式允许模型从自身的失败中进行反思和修正，从而实现更高效的探索，避免重复犯错。

2.  **Attempt-Level Credit Assignment (尝试级信用分配)**：
    *   如何为多次尝试中的每一次分配信用是一个关键问题。ReAPO 将此问题建模为一个**随机最短路径（Stochastic Shortest Paths, SSP）**问题，目标是**最小化达到第一个正确答案前的期望失败次数**。
    *   在合理的简化假设下，该框架自然地推导出了一种信用分配机制：对**每一次尝试**单独计算其优势（advantage），并将该优势仅应用于该次尝试所生成的 token 上。
    *   这意味着成功的重试会被强化，但不会“奖励”导致初始失败的推理过程，实现了理论上更合理、更精细的信用分配。

### 相比现有方法的优势
- **更高的效率**：ReAPO 能以更少的训练步数、生成的 token 数量和墙钟时间（wall-clock time）达到甚至超越标准 GRPO 的峰值性能。
- **更好的性能**：在多个数学和科学推理基准上，ReAPO 显著提升了首次尝试的准确率（first-attempt performance）。
- **理论基础扎实**：其信用分配机制源于经典的强化学习框架（SSP），具有坚实的理论依据。
- **易于集成**：ReAPO 兼容任何基于验证器的奖励函数，无需额外监督或辅助目标，可以轻松集成到现有的训练流程中。

## 2. 核心实验方法和设置

### 使用的数据集
- **小规模模型训练**：使用 **GSM8K** 数据集训练 `Qwen3-0.6B-Base` 模型。
- **大规模模型训练**：使用 **Polaris-53K** 数据集训练 `Qwen3-1.7B-Base` 和 `Qwen3-4B-Base` 模型。
- **评估数据集**：
  - `Qwen3-0.6B-Base`：在 **GSM8K-Platinum** 上评估。
  - `Qwen3-1.7B/4B-Base`：在 **BeyondAIME**, **AIME 2024/2025**, **HMMT 2025**, **MATH-500**, **Minerva**, **GPQA-Diamond** 等八个数学和科学推理基准上评估。

### 实验设置和评估指标
- **模型**：在 `Qwen3` 和 `OLMo-2` 两个系列的模型上进行了实验，参数规模从 0.6B 到 4B。
- **超参数**：
  - 默认 **group size (G)** 为 16。
  - ReAPO 的最大尝试次数 **K** 设为 2 或 3。
  - 所有方法在每个轨迹上的总响应 token 预算相同。
- **评估指标**：
  - **Mean@32**：每个问题评估 32 个响应，取平均准确率。
  - **Pass@32**：32 个响应中至少有一个正确的比例。
  - **First-attempt accuracy**：主要关注**首次尝试**的性能，以反映模型的真实推理能力。
  - **Token-normalized performance**：按生成的总 token 数量归一化后的性能，用于公平比较计算效率。

### 基线方法对比
- **GRPO**：标准的基线方法。
- **GRPO (2x steps)**：训练步数翻倍的 GRPO。
- **GRPO (2x group size)**：组大小翻倍（G=32）的 GRPO，以匹配 ReAPO 最多可能生成的响应数量。
- **SCoRe (Stage 2)**：一种多轮 RL 方法，但其第二次尝试的生成不依赖于第一次是否失败的验证反馈。
- **MaxRL**：一种旨在提升 `pass@k` 性能的近期方法。
- **Multi-Attempt GRPO**：将整个多次尝试序列视为一个整体，赋予单一优势值，用以消融 ReAPO 的尝试级信用分配的有效性。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **效率显著提升**：
  - 在 `Qwen3-0.6B` 模型上，ReAPO 达到 GRPO 峰值准确率所需的**生成 token 数量减少了 39%**（相比两倍组大小的 GRPO）和 **31%**（相比两倍训练步数的 GRPO）。
  - 图 2 和图 3 清晰地展示了 ReAPO 在训练步数、生成 token 数和墙钟时间三个维度上都优于或迅速超越 GRPO。

- **性能全面领先**：
  - 表 1 显示，在 `Qwen3-1.7B` 和 `Qwen3-4B` 模型上，ReAPO 在几乎所有基准测试中的首次尝试准确率均超过了 GRPO 和基线模型。
  - 例如，在 `Qwen3-4B` 上，ReAPO 的平均准确率达到了 **31.0%**，而 GRPO 为 **28.8%**。

- **消融实验结果**
  - **尝试级信用分配至关重要**：与将整个序列赋予单一优势的 "Multi-Attempt GRPO" 基线相比，ReAPO 表现更好，证明了精细化的信用分配是其成功的关键。
  - **条件化重试优于无条件重试**：一个变体在无论首次尝试是否正确时都生成第二次尝试（无条件重试），其性能低于 ReAPO，说明基于验证失败的条件化重试更高效。
  - **增加尝试预算收益递减**：将最大尝试次数 K 从 2 增加到 3 时，性能提升有限，尤其是在较小的 0.6B 模型上。这表明模型在两次尝试后已接近饱和，第三次尝试提供的学习信号较弱。
  - **提示词鲁棒性强**：更换不同的重试提示词（如 "Try again"）对最终性能影响不大，说明方法对提示词的具体措辞不敏感。

## 4. 关键结论和发现

### 主要发现
1.  **序列化探索优于独立采样**：通过在失败后引入条件化重试，模型能够进行更有效的探索，避免在相同的错误路径上浪费计算资源。
2.  **精细化的信用分配是关键**：将信用分配到单个尝试级别，而非整个序列，使得成功的重试可以被强化而不奖励之前的失败，这是 ReAPO 成功的核心。
3.  **ReAPO 是一种高效且强大的方法**：它不仅在性能上超越了 GRPO 等基线，而且在计算效率上也具有显著优势，是一种简单、有效且易于部署的改进方案。

### 方法的局限性
- **截断偏差**：实践中限制了最大尝试次数 K，这会引入理论上的截断偏差，尽管实验证明当 K=2 时收益已趋于饱和。
- **令牌分配与截断问题**：固定的总 token 预算可能导致早期长回复消耗过多预算，从而压缩了后续重试的空间。实验观察到 ReAPO 的截断率更高，这可能影响其后期性能。
- **对验证器可靠性的依赖**：方法假设验证器能准确判断答案的对错。如果验证器出现误判（假阳性或假阴性），可能会误导模型的学习。
- **适用领域受限**：目前实验集中在答案可自动验证的数学和科学推理任务上，其在代码生成、工具使用等验证更复杂的领域中的表现尚待研究。

### 未来工作方向
- **扩展重试预算**：研究随着模型规模和推理时 token 预算的增加，更大的 K 值是否能带来持续的性能增益。
- **优化令牌分配**：设计自适应的令牌分配策略，为重试保留足够的空间，例如为每次尝试设置独立的预算。
- **探索其他优化目标**：将 ReAPO 的思想（验证器条件化 rollout 和尝试级信用分配）与 Dr.GRPO、DAPO、MaxRL 等其他策略优化方法结合，探索潜在的协同效应。
- **研究更广泛的领域**：将 ReAPO 应用于代码生成、复杂代理任务（agentic settings）等更具挑战性的领域，检验其通用性。

</details>

---

### 10. [Cross-Area Transfer Learning for VLC-based Indoor Localization with a Transfer Efficiency Score](https://arxiv.org/abs/2610.29638)

**Authors**: Masood Jan, Wafa Njima, Xun Zhang, Alexander Artemenko  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2610.29638v1  

#### Abstract
Indoor localization systems are critical for Industry-4.0 applications. Visible Light Communication (VLC)-based systems offer advantages such as immunity to electromagnetic interference, high accuracy, and energy efficiency, but their performance is hindered by environmental variability across deplo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Cross-Area Transfer Learning for VLC-based Indoor Localization with a Transfer Efficiency Score*

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **VLC-based indoor localization** 在实际工业部署中面临的三大挑战：
- **跨区域鲁棒性差**（Cross-area robustness）：模型在一个区域训练后，在不同布局或环境的新区域性能显著下降。
- **数据效率低**（Data inefficiency）：为每个新区域收集和标注完整数据集需要大量停机时间，成本高昂。
- **缺乏系统化的源模型选择机制**：在拥有多个预训练模型时，难以快速、低成本地确定哪个模型最适合迁移到目标区域。

### 提出的新方法与创新
1. **面向VLC的跨区域迁移学习（TL）框架**  
   提出一个完整的 TL 框架，利用在源区域训练好的 DNN 模型，通过少量目标区域数据进行微调（fine-tuning），实现跨区域自适应，提升模型在动态工业环境中的泛化能力。

2. **数据高效的微调策略**  
   实验证明，仅需使用 **40% 的目标区域测量数据** 进行微调，即可恢复 **97% 的全数据训练性能**，将数据采集需求降低60%，大幅减少部署成本和时间。

3. **提出 Transfer Efficiency Score (TES)**  
   创新性地提出 **TES**——一种归一化的复合评估指标，综合考虑定位误差（localization error）和成功率（success rate），用于在不进行穷举微调的情况下，系统性地识别最优的源模型。

### 相比现有方法的优势
| 方面 | 本文方法优势 |
|------|--------------|
| **相比传统重训练** | 减少60%目标数据需求，避免从零开始训练，节省时间和资源。 |
| **相比联邦学习（FL）** | 不依赖持续的客户端参与和梯度交换，通信开销小，更适合资源受限的工业IoT场景。 |
| **相比连续学习（CL）** | 避免“灾难性遗忘”（catastrophic forgetting）问题，无需持续获取新标签数据。 |
| **相比其他TL方法** | 引入 TES 指标，提供可复现、量化的模型选择标准，避免试错式手动选择。 |

---

## 2. 核心实验方法和设置

### 数据集
- **真实工业环境数据**：来自德国博世（Bosch）制造工厂车间的真实测量数据。
- **四个独立子区域**：Area A, B, C, D，基于建筑多样性划分（布局、设备配置、结构分布不同）。
- **信号类型**：基于 **VLC RSS**（Received Signal Strength）的指纹数据。
- **测量平台**：使用自研的 VLC transceiver 系统，配备 LED 发射器和 PD 接收器，通过 VNA（矢量网络分析仪）采集信号强度。
- **数据规模**：每个区域平均采样约160个接收位置，信号持续采集5秒并取均值以增强鲁棒性。

### 实验设置
- **模型架构**：深度神经网络（DNN），输入维度10，5个隐藏层，每层512个神经元。
- **训练参数**：优化器为 ADAM，学习率 0.0001，batch size 64，训练1000轮。
- **迁移学习策略**：
  - 源模型在源区域完整数据上训练。
  - 目标区域仅使用部分数据（10%–100%）进行微调。
  - 微调时冻结底层网络，仅更新顶层 k 层，以保留通用特征表示。
- **交叉验证**：采用轮换方式，每次将一个区域作为“未见区域”（unseen area）用于最终泛化能力测试。

### 评估指标
- **主指标**：
  - **Localization Error**（定位误差，单位：cm）
  - **Success Rate**（成功率，预测误差 ≤ 1m 视为成功）
- **核心复合指标**：
  - **Transfer Efficiency Score (TES)**：  
    $$
    \text{TES}_{s \to t} = \alpha (1 - \text{NE}) + (1 - \alpha) (\text{NSR})
    $$
    其中 NE 是归一化误差，NSR 是归一化成功率，$\alpha = 0.5$ 平衡两者。

### 基线方法对比
- **Baseline**：直接将在源区域训练的模型应用于目标区域（无微调）。
- **Full Retraining**：在目标区域使用100%数据从头训练模型（理想上限）。
- **本文方法（TES-TL）**：使用 TES 指导选择最佳源模型，并用40%目标数据微调。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 场景 | 定位误差（cm） | 成功率（%） |
|------|----------------|------------|
| **Area A → B（无TL）** | 134.91 | 23.66 |
| **Area A → B（40%数据微调）** | **40.90** | **86.02** |
| **Area A → C（40%数据微调）** | **51.55** | **98.48** |
| **TES指导下的最优迁移（A→B）** | **40.9 cm** | **>86%** |
| **在未见区域 D 上的泛化（TES选A）** | **3.28 cm** | **100%** |
| **40%目标数据恢复性能** | 达到全数据性能的 **97%** | — |

### 与基线方法对比
- **相比无TL的直接迁移**：定位误差显著降低（如 A→B 从134.91 cm降至40.90 cm），成功率大幅提升（23.66% → 86.02%）。
- **相比全数据重训练**：使用40%数据微调已接近其性能，证明了极高的数据效率。
- **TES有效性验证**：TES选出的最佳源模型（Area A）在未见区域 D 上表现最优，误差仅3.28 cm，远优于其他源模型。

### 消融实验结果
#### （1）微调数据比例影响（Fine-tuning Data Ratio）
- **10%数据**：性能改善有限（A→B误差47.34 cm）。
- **20%数据**：明显提升（A→B误差44.43 cm）。
- **40%数据**：性能跃升，接近饱和（A→B误差40.90 cm）。
- **>60%数据**：边际收益递减，表明 **40% 是性价比最优选择**。

#### （2）TES敏感性分析
- 改变权重参数 $\alpha \in [0.3, 0.7]$，Area A 始终是最优或次优源模型，说明 **TES结论具有鲁棒性**。
- 当 $\alpha=0.5$ 时，平衡精度与鲁棒性，效果最佳。

#### （3）TES与物理相似性的相关性
- 计算各区域间的 **Bhattacharyya Distance**（特征空间分布差异）。
- 结果显示：距离越小（相似度越高），TES得分越高，证实 **TES能有效反映环境间的物理兼容性**。

---

## 4. 关键结论和发现

### 主要发现
1. **迁移学习对VLC定位至关重要**：直接跨区域应用模型会导致严重性能退化，必须通过微调适配。
2. **极高的数据效率**：仅需40%目标数据微调即可恢复97%全数据性能，显著降低部署成本。
3. **TES是有效的模型选择工具**：TES能准确识别最具迁移潜力的源模型，避免试错，提升部署效率。
4. **强泛化能力**：在完全未见的区域 D 上，TES指导的方法仍取得 **3.28 cm 超高精度**，验证了框架的实用性和可扩展性。
5. **物理相似性驱动迁移效果**：结构和环境越相似（如A与D均有混凝土柱和开放边界），知识迁移越有效。

### 方法的局限性
1. **单设施评估**：所有实验均在单一博世工厂内完成，尚未跨不同工厂验证泛化性。
2. **固定高度假设**：Tx 和 Rx 高度固定，未考虑移动机器人或手持设备的高度变化影响。
3. **监督式微调依赖标签数据**：仍需少量带标签的目标数据，未来可探索半监督方案。
4. **TES计算成本随区域数增长**：在大规模部署中，TES需离线计算，可能带来一定开销。
5. **单源迁移限制**：当前框架仅支持单一源模型迁移，未融合多源知识。

### 未来工作方向
1. **扩展至多设施数据集**：收集更多工业现场数据，验证跨工厂迁移能力。
2. **处理高度变化**：引入高度作为输入特征或联合估计位置与高度。
3. **半监督与表征保持迁移**：利用无标签数据进行伪标签、一致性正则化等，进一步降低标注成本。
4. **可扩展的 TES 近似方法**：研究基于无监督域相似性指标（如特征分布散度）的 TES 快速估算方法。
5. **多源/混合迁移策略**：探索动态源选择、集成初始化或多源表征融合。
6. **与其他模态对比**：开展与 UWB、Wi-Fi 等 RF 技术的 TL 性能横向比较。
7. **跨模态迁移**：探索 Wi-Fi 与 VLC 之间的知识迁移，实现多传感融合定位。

---

> **总结**：本文提出了一套高效、可扩展的 **VLC-based indoor localization** 迁移学习解决方案，通过 **数据高效的微调策略** 和创新的 **TES 评估指标**，显著提升了跨区域部署的可行性与经济性。实验证明其在真实工业环境中具备高精度（<41 cm）、强泛化（未见区域3.28 cm）和高数据效率（40%数据恢复97%性能）的优势，为 Industry 4.0 场景下的智能定位提供了坚实的技术路径。

</details>

---

### 11. [Stratified Sparse Attention: Reducing the Sparse-to-Dense Gap at Extreme Sparsity](https://arxiv.org/abs/2610.28441)

**Authors**: Sahil Joshi, Prithvi Dixit, Aryan Katiyar, Agniva Chowdhury, Joseph E. Gonzalez, Anshumali Shrivastava, Aditya Desai, Kumar Krishna Agrawal  
**Category**: cs.AI  
**Published**: 2026-10-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.28441v1  

#### Abstract
Large Language Model inference over long contexts is increasingly bottlenecked by the cost of attending to a growing Key-Value (KV) cache. Sparse attention reduces this cost by accessing only a fraction of the cache, but existing methods face a trade-off between deterministic retrieval, which captur...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Stratified Sparse Attention: Reducing the Sparse-to-Dense Gap at Extreme Sparsity**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在长上下文场景下，**Large Language Model (LLM)** 推理过程中的 **attention** 操作因需要访问不断增长的 **Key-Value (KV) cache** 而成为内存带宽瓶颈。为降低计算开销，**sparse attention** 方法被提出，仅访问部分 KV cache。然而，现有方法面临以下权衡：
- **Deterministic retrieval**（如 top-k）能高效捕获高分 token，但会丢弃残差注意力质量（residual attention mass），导致精度损失。
- **Sampling-based methods**（如 uniform sampling）虽能提升覆盖范围，但引入较高方差，难以逼近 dense attention。
- **Hybrid methods**（如 vAttention）结合 top-k 和 uniform sampling，但在极端稀疏度下仍存在显著的 **sparse-to-dense gap**。

本文旨在设计一种更有效的采样策略，**在极低稀疏度下显著缩小与 dense attention 的性能差距**。

### **提出的新方法：Stratified Sparse Attention (SSA)**
SSA 是一个统一的稀疏注意力框架，结合了确定性检索与基于分数引导的随机探索，其核心思想是：
- 将 token 分为两个“层”（strata）：
  - **Deterministic stratum**：保留高代理分数（proxy scores）的 token（即“重击者” heavy hitters）。
  - **Stochastic stratum**：对剩余 token 进行 **score-guided importance sampling**，利用代理分数指导采样，优先探索潜在重要的残差区域。
- 通过逆概率加权（inverse-probability weighting）校正采样偏差，恢复被忽略的注意力质量。

SSA 的关键创新在于：
- **自动预算分配**：无需手动设定 top-k 大小，而是根据代理分数分布动态划分确定性与随机性预算。
- **理论保证**：提供了误差上界分析，证明 SSA 在有限预算下可逼近 dense attention。
- **通用性**：top-k、top-p、hybrid top-k+sampling 均为其特例。

### **相比现有方法的优势**
- **更高精度**：在相同稀疏度下，SSA 显著优于 deterministic、sampling 和 hybrid 方法，尤其在长上下文和极端稀疏度下表现突出。
- **更低方差**：相比 uniform sampling，score-guided sampling 更有效捕捉残差注意力质量，减少估计方差。
- **实际加速**：配合优化的 CUDA/Triton 内核，SSA 实现端到端推理加速，最高达 **2.96× decode speedup**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **RULER-HARD**：测试长上下文理解能力，涵盖 Factual QA、Needle-in-a-Haystack 等任务。
- **LongBench**：多语言、多任务长文本基准。
- **LOFT**：评估模型是否能替代检索增强生成（RAG）、SQL 等系统。
- **AIME 2025**：数学推理任务，生成长答案。

### **实验设置与评估指标**
- **模型范围**：从 3B 到 27B 参数的多种模型，包括：
  - `Llama-3.1-8B-Instruct`
  - `Qwen3.5-4B` / `Qwen3.5-9B`
  - `Ministral-3-3B-Instruct` / `Ministral-3-8B-Instruct`
- **稀疏度**：测试 10× 至 100× 的 attention sparsity。
- **上下文长度**：32K、64K、128K、1M tokens。
- **评估指标**：
  - **Accuracy**（如 RULER-HARD、AIME avg@16）
  - **Attention error**（L2 误差）
  - **Decode latency**（解码延迟）
  - **Speedup**（相对于 FlashInfer）

### **基线方法对比**
- **PQCache**：基于 product quantization 的 approximate top-k 方法。
- **vAttention (PQ)**：hybrid 方法，top-k + uniform sampling。
- **Oracle-top-k**：使用真实 attention scores 的 top-k，作为上界参考（非实用）。
- **MagicPig、HashAttention、Double Sparsity**：其他稀疏注意力方法（见附录）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **RULER-HARD @ 50× sparsity**
| Model | Method | 32K | 64K | 128K |
|-------|--------|-----|-----|------|
| Llama-3.1-8B | Dense | 88.76 | 86.52 | 71.26 |
| | PQ | 72.51 | 67.53 | 39.73 |
| | vAttn(PQ) | 66.71 | 66.22 | 39.45 |
| | **SSA** | **83.67** | **77.09** | **47.53** |

- SSA 在 32K 上接近 dense 性能（83.67 vs 88.76），远超 PQ 和 vAttn。
- 在 128K 下，SSA 恢复了 **85%** 的 dense-attention 性能。

#### **Ministral-3-8B @ RULER-HARD**
| Context | Method | Accuracy |
|---------|--------|----------|
| 32K | PQ | 81.83 |
| | vAttn(PQ) | 78.69 |
| | **SSA** | **84.17** |
| 128K | PQ | 62.99 |
| | vAttn(PQ) | 63.78 |
| | **SSA** | **69.25** |

- 在 128K 下，SSA 将准确率从 63.8% 提升至 **69.3%**，恢复 **85%** 的 dense 性能。

#### **AIME 2025 @ 50× sparsity (Qwen3.5-9B)**
| Method | Accuracy |
|--------|----------|
| Dense | 66.3 |
| Oracle-top-k | 48.3 |
| PQ | 36.7 |
| vAttn(PQ) | 35.8 |
| **SSA** | **59.8** |

- SSA 在极端稀疏下仍保持鲁棒性，在 100× 稀疏度下仍达 **51.5**，而 vAttn(PQ) 骤降至 3.5。

#### **解码速度提升**
| Sparsity | Batch Size | SSA Speedup vs FlashInfer |
|----------|------------|----------------------------|
| 50× | 1 | 1.79× |
| 50× | 16 | **2.96×** |
| 20× | 16 | 2.17× |

- 在 1M 上下文、batch size 16 下，SSA 实现 **2.96×** 解码加速。

### **消融实验结果**
#### **不同 heavy fraction $ h $ 的影响（AIME 任务）**
| $ h $ (Heavy Fraction) | 20× | 10× |
|------------------------|-----|-----|
| 0.0 (Pure sampling) | 40.2 | 42.5 |
| 0.4 | 63.5 | 64.0 |
| 0.8 | 63.8 | 63.1 |
| 1.0 (Pure top-k / PQ) | 49.4 | 53.8 |

- **混合策略（$ h \in [0.2, 0.8] $）效果最佳**，纯 sampling 或纯 top-k 均不如混合。
- 最佳性能出现在 $ h = 0.4 $ 左右，表明 **exploitation 与 exploration 的平衡至关重要**。

#### **理论 vs 实践版本对比**
| Model | Sparsity | SSA-Theory | SSA-Practical |
|-------|----------|------------|---------------|
| Llama-3.1-8B | 50× | 81 | 83 |
| Ministral-3-8B | 50× | 82 | 84 |

- 实际实现（固定预算分配）甚至略优于理论版本，说明工程优化有效。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Score-guided sampling 显著优于 uniform sampling**：在残差 token 中，基于代理分数进行重要性采样能更有效地恢复注意力质量。
2. **SSA 显著缩小 sparse-to-dense gap**：在多个模型和数据集上，SSA 在 50× 稀疏度下恢复了 **85–94%** 的 dense-attention 性能。
3. **理论与实践一致**：SSA 具有严格的误差界，并可通过高效内核实现端到端加速。
4. **自动预算分配有效**：无需调参即可适应不同注意力分布，提升鲁棒性。

### **方法的局限性**
- **依赖代理分数质量**：若 proxy scores 与真实 attention scores 偏差较大，性能可能下降。
- **采样开销**：尽管优化，采样阶段仍引入额外计算，尤其在极高 batch size 下需进一步优化。
- **非验证性**：当前 SSA 不提供用户可控的近似质量保证（如误差上限），限制其在安全关键场景的应用。

### **未来工作方向**
- **Verifiable SSA**：引入用户可配置的质量保证机制，使其适用于高可靠性场景。
- **动态代理分数学习**：在线调整 proxy scores 以更好拟合真实 attention 分布。
- **与 KV compression 结合**：将 SSA 与 SnapKV、KVZip 等 KV 缓存压缩技术结合，进一步降低内存占用。
- **训练时集成**：探索在 pretraining 或 fine-tuning 阶段引入 SSA，使模型适应稀疏注意力模式。

---

> **开源代码**：https://github.com/skylight-org/SSA-Paper-Code

</details>

---

### 12. [Evolving Semantic Indexes for Generative Retrieval](https://arxiv.org/abs/2610.28568)

**Authors**: Jinming Nian, Jyun-Yu Jiang, Henry Tsai, Yi Fang, Zhirong Liang  
**Category**: cs.AI  
**Published**: 2026-10-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.28568v1  

#### Abstract
Generative retrieval (GR) casts search and recommendation as autoregressive generation of discrete semantic identifiers (SIDs). SID design largely determines GR quality, but its discrete, interdependent components (SID construction, collision handling, decoding, and training) cannot be optimized end...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Evolving Semantic Indexes for Generative Retrieval

---

## 1. 论文的主要贡献和创新点

### 解决的问题
**Generative Retrieval (GR)** 将检索任务建模为自回归生成语义标识符（Semantic Identifier, SID）的过程。然而，SID 的设计——包括索引构建、冲突处理（collision handling）、训练目标和解码策略——通常是手工设计且离散的，无法通过梯度端到端优化。这导致 GR 系统的设计空间受限，难以系统性地探索最优配置。

本文旨在解决这一**离散、不可微分的 GR 组件联合优化难题**，特别是如何自动改进 SID 框架中的多个互依赖模块。

### 提出的新方法与新思路
作者提出了一种**基于 LLM 驱动的进化搜索框架**，用于自动化演化 GR 中可执行的组件，尤其是 SID 构造程序。其核心创新包括：

- **两层进化框架（Two-tiered evolutionary framework）**：
  - **外层循环**：管理程序种群（population），选择父代程序，并向 LLM mutator 提供反馈。
  - **内层循环**：对候选程序进行门控（gatekeeping），并通过一个轻量级的 **proxy decoder** 快速评估其性能，避免每次评估都训练庞大的目标生成器。

- **本地化、开源模型驱动**：
  - 使用可在单张 GPU 上运行的开源大模型 **gemma-4-31b-it** 作为 mutator LLM，不依赖任何专有 LLM API。
  - 使用仅含 **1M 参数的 GPT-2** 作为 proxy decoder 来近似下游检索性能（如 Recall@100），将每次评估时间从 5 小时缩短至约 20 分钟，提升效率达 **15 倍以上**。

- **可执行程序的突变机制**：
  - 基于 AdaEvolve 框架，mutator LLM 对 Python 代码块提出 `SEARCH/REPLACE` 编辑建议，直接修改 SID 生成逻辑。
  - 支持演化不同粒度的组件：仅 SID 构造、联合 SID 与 collision handling、甚至完整 GR pipeline。

### 相比现有方法的优势
| 方面 | 传统方法 | 本论文方法 |
|------|--------|----------|
| 设计方式 | 手工调参、固定架构 | 自动化程序演化 |
| 优化目标 | 固定损失函数 | 执行效果驱动（execution-based evaluation） |
| 评估成本 | 每次需训练大模型 → 昂贵 | 使用小型 proxy decoder → 高效 |
| 可扩展性 | 难以探索复杂组合 | 支持多组件协同演化 |
| 开放性 | 多数使用闭源 LLM（如 GPT-4） | 全流程使用开源模型 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **NQ320K**：Natural Questions 数据集的子集，包含约 110K 文档和 7.8K 测试查询。
- **MSMARCO300K**：MS MARCO passage ranking 数据集的子集，包含约 320K 文档和 808 测试查询。
- **E-commerce（私有）**：某大型电商公司的多语言购买日志数据集，包含百万级商品项，以购买行为作为相关性标签。

所有数据集均采用标准划分，测试集完全不可见于 mutator 和 proxy 训练过程。

### 实验设置
- **输入表示**：
  - NQ & MSMARCO：使用固定的 **Contriever** 编码器获取文档/查询嵌入。
  - E-commerce：使用 **mContriever**（支持多语言）。
- **SID 构造接口**：
  - 输入：`doc_emb`, `query_emb`, `qrel_pairs`, `doc_ids`
  - 输出：每个文档对应的整数数组形式的 SID，要求行唯一（no collision）
- **训练配方统一**：
  - 所有对比实验使用相同的 target generator（**gemma-3-1b-pt**）、训练超参数（学习率、batch size 等）和训练轮数（3 轮），确保公平比较。

### 评估指标
- 主要指标：
  - **Recall@1, @5, @10, @100**
  - **MRR@10**
- 评估流程：
  1. 进化阶段：用 **proxy decoder（1M 参数）** 在保留的验证 query-doc pair 上计算 Recall@100 作为 fitness。
  2. 最终评估：选出最佳程序后，重新生成全量 SID，训练 **target generator（1B 参数）** 并在独立测试集上报告最终性能。

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **BM25** | 经典词法检索模型 |
| **Contriever** | 稠密检索基线 |
| **Seed GR** | 基于 RQVAE + TIGER 式冲突处理的初始 GR 系统 |
| **SID evolution** | 本文提出的进化框架生成的最佳 SID 程序 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Dataset | Method | R@1 (%) | R@5 (%) | R@10 (%) | R@100 (%) | MRR@10 (%) |
|--------|--------|---------|---------|----------|-----------|------------|
| **NQ320K** | Seed GR | 55.82 | 71.12 | 75.27 | 85.82 | 62.50 |
| | **SID evolution** | **59.28** (+3.46) | **76.44** | **81.05** | **90.57** (+4.75) | **66.74** |
| **MSMARCO300K** | Seed GR | 19.43 | 40.97 | 51.49 | 79.70 | 28.61 |
| | **SID evolution** | **20.48** (+1.05) | 41.96 | **55.01** | **82.55** (+2.85) | **30.09** |
| **E-commerce** | Seed GR | +8.06 | +20.36 | +27.18 | +49.01 | +13.37 |
| | **SID evolution** | **+8.56** | **+21.40** | **28.47** | **50.66** | **14.09** |

> 注：电商数据为相对 BM25 的增益（percentage points）

### 与基线方法的对比结果
- 在三个数据集上，**经过进化的 SID 程序显著优于 Seed GR**，尤其是在 Recall@100 上取得大幅提升。
- 在 NQ320K 上，Recall@1 从 55.8% 提升至 **59.3%**，Recall@100 从 85.8% 提升至 **90.6%**。
- 即使 Seed GR 初始表现弱于 Contriever，在 SID 演化后也能反超。
- 在电商场景中，GR 本身已大幅领先 BM25，进一步演化仍带来稳定增益。

### 消融实验结果（Table 2 & Section 5.2）

#### 不同演化范围的效果对比（NQ320K）

| Search Scope | Proxy R@100 (%) | Target R@100 (%) | Δ vs Seed (%) |
|-------------|------------------|--------------------|---------------|
| **Joint SID-collision** | 64.39 | **90.57** | **+4.75** ✅ |
| **SID-only** | 61.47 | 87.40 | +1.58 |
| **Sequential SID-collision** | 61.30 | 87.25 | +1.43 |
| **Collision-only** | 52.69 | 85.72 | -0.10 |
| **Objective evolution** | 53.80 | 84.46 | -1.36 ❌ |
| **Full-pipeline evolution** | 62.85 | 76.12 | **-9.70** ❌ |

#### 关键发现：
- **联合演化 SID 与 collision handling 效果最好**，说明两者存在强耦合关系。
- **仅演化 SID 构造即可带来稳定收益**，而仅调整 collision handling 几乎无提升。
- **一旦允许改变 generator 的训练目标（objective）或完整 pipeline，proxy 指导失效**，出现“过拟合 proxy”的现象。
- **proxy 与 target 的 Recall@100 排名相关性高达 Spearman ρ = 0.90**（图 2b），但在 objective 层面相关性反转（ρ = -0.80）。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 驱动的程序演化可用于 GR 系统的自动化改进**，首次实现对可执行 GR 组件的端到端搜索。
2. ✅ **proxy-guided 搜索是可行的**，只要 proxy 与 target 在关键组件上有共享结构（如 SID 表示），就能有效传递性能增益。
3. ✅ **语义索引（Semantic Index）是最适合演化的组件**，其改进具有良好的跨规模迁移性。
4. ⚠️ **generator 训练目标不适合由 proxy 直接优化**，因为小模型上的最优目标未必适用于大模型。
5. 🔗 **SID 构造与 collision handling 存在协同效应**，联合演化能获得更大收益。

### 方法的局限性
- **当前 mutator 使用的是 31B 规模的 gemma 模型**，虽然本地可运行，但其推理能力可能限制更复杂算法的发现。
- **proxy 只能指导特定类型的变更**，对于涉及训练动态或解码策略的大范围改动，可靠性下降。
- **未实现完全端到端的 pipeline 演化**，目前仍需人工设定训练流程和接口约束。
- **演化过程依赖高质量 prompt engineering**，对 mutator 的引导至关重要。

### 未来工作方向
- 探索更强的 mutator LLM（如 70B+ 模型）是否能发现更先进的 GR 架构。
- 设计更具泛化性的 proxy evaluator，使其能可靠指导 generator objective 或 decoding 策略的演化。
- 将该框架扩展至其他生成式 IR 任务，如 **generative recommendation** 或 **multi-hop retrieval**。
- 研究如何实现 **component-level attribution**，以便在多模块同时演化时识别有效变更来源。

---

> 📌 **总结一句话**：  
> 本文提出了首个基于 LLM 驱动的进化搜索框架，用于自动化优化 Generative Retrieval 中的语义索引（SID）构造，通过引入轻量 proxy decoder 实现高效评估，在多个数据集上显著提升了 Recall 性能，并揭示了“索引结构”比“训练目标”更适合 proxy 指导演化的重要发现。

</details>

---

### 13. [PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States](https://arxiv.org/abs/2610.29545)

**Authors**: Patrick Haller, Filipe Laitenberger, Alan Akbik  
**Category**: cs.CL  
**Published**: 2026-10-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.29545v1  

#### Abstract
Linear-attention language models keep their context in a recurrent state of fixed size, where an inference system stores one copy of this state for every sequence it decodes at the same time. Storing the state at lower precision therefore lets more sequences share an accelerator. Existing state quan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States

---

## 1. 论文的主要贡献和创新点

### 解决的问题
线性注意力语言模型（Linear-attention language models）在推理时需要维护一个固定大小的**recurrent state**来保存上下文信息。每个解码序列都需要独立存储该状态，因此其内存占用成为大规模部署的瓶颈。现有的**state quantization**方法通常在每生成一个 token 后就对状态进行四比特量化（per-token rounding），这会导致累积的舍入误差显著降低模型精度。

本文旨在解决以下问题：
- 如何在保持高精度的前提下，将 recurrent state 成功压缩至 **4-bit**？
- 如何减少量化带来的误差积累？
- 如何设计更有效的量化策略以提升效率与性能？

---

### 提出的新方法：PACE
作者提出了一种名为 **PACE**（Periodic And Channel-aligned state Encoding）的**后训练四比特状态量化器**，无需微调即可实现高效低精度存储。

#### 核心创新点：
1. **周期性舍入（Periodic Rounding）**
   - 不再每生成一个 token 就量化一次状态，而是每隔 `dP` 个 token 才执行一次量化（如 `dP=8`）。
   - 在两次量化之间，新增的 token 被暂存在一个小的 **decode buffer** 中。
   - 这样减少了量化次数，从而显著降低了误差积累。

2. **通道对齐的量化尺度分组（Channel-Aligned Scale Groups）**
   - 发现 gated-delta state 中的大值项倾向于沿 **value axis（列）聚集**，这是因为某些 delta value 的通道在整个序列中持续保持较大值。
   - 因此，PACE 将每个 scale group 设计为沿着 **key axis（行方向）**，即每一列为一组，共享同一个量化尺度（scale）。
   - 这种“channel-aligned”方式比传统的按 value axis 分组更合理，能更好保留重要信息。

3. **头级别混合精度分配（Head-Level Mixed Precision）**
   - 引入 PACE-4/8 变体：将每层中保留舍入误差最长的 **前 25% 的 attention heads** 保留为 8-bit，其余为 4-bit。
   - 利用 decay persistence 和误差传播分析选择关键 heads，进一步提升精度。

---

### 相比现有方法的优势
| 方面 | PACE 优势 |
|------|----------|
| **内存效率** | 在约 **2/3 的 DAMP INT8 内存开销下**达到相近甚至更高的精度 |
| **精度表现** | 显著优于同等内存下的 Q-Mamba 和 DAMP INT4 |
| **无需微调** | 完全 post-hoc 方法，不依赖 fine-tuning 或权重调整 |
| **兼容性好** | 与 decode buffer 加速机制兼容，不影响解码速度 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **WikiText-103**：用于计算 perplexity 和 state drift 分析
- **GSM8K**：数学应用题理解能力测试
- **MMLU-Pro**：多任务语言理解基准
- **IFEval**：指令遵循能力评测
- **RULER-32k**：长上下文变量追踪任务（32,768 tokens）
- **MATH-500**：复杂数学问题解答

所有任务均为 zero-shot 设置，禁用思维链提示。

---

### 实验设置与评估指标

#### 模型架构
- 四个混合架构模型（gated-delta + full attention 层交替）：
  - Qwen3.5-4B, 9B, 27B
  - Qwen3.6-35B-A3B

#### 评估指标
- 主要指标：各任务上的准确率（exact match, strict accuracy 等）
- 综合指标：多个 benchmark 的平均得分
- 内存指标：**resident bits per value**（包含 state payload、scale、decode buffer）
- 性能指标：Time Per Output Token (TPOT)，state update latency

#### 解码配置
- 上下文长度：512 ~ 32,768 tokens
- Batch size: 1–128
- Decode period `dP`: 默认为 8
- Prefill period `P`: 固定为 256（chunk size）

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Full Precision (fp32)** | 浮点全精度状态作为上限参考 |
| **DAMP INT8** | decay-aware mixed-precision，保留部分 key channels 为 fp16，总约 9.9 bits/value |
| **DAMP INT4** | 同上策略应用于 4-bit base，保留更多 channels 为 8-bit |
| **Q-Mamba DSQ** | 解耦轴向缩放（decoupled scale quantization），需 fine-tune 权重，仅使用静态 scales 复现实验 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 模型 | 方法 | bits/value | 平均得分 |
|------|------|------------|--------|
| Qwen3.5-9B | Full Precision | 32.00 | 84.5 |
| | DAMP INT8 | 9.90 | 83.7 |
| | **PACE-4/8** | **6.82** | **83.7** ✅ |
| | PACE-4 | 5.03 | 80.0 |

> ✅ **PACE-4/8 在仅占 DAMP INT8 约 69% 内存的情况下，达到了几乎相同的平均性能**

| 对比项 | 结果 |
|-------|------|
| **vs. DAMP INT4** | PACE-4/8 在相同内存 (~6.44 bits) 下全面超越 DAMP INT4，尤其在大模型上差距明显（如 35B 上 +16 pts） |
| **vs. Q-Mamba DSQ** | PACE-4 与 Q-Mamba 占用相同内存（~5.03 bits），但在所有模型上均大幅领先（+4~7 pts） |

---

### 消融实验结果（Table 2 & Figure 4）

#### （1）周期性舍入的影响（`dP > 1`）
- 将 decode period 从 `dP=1` 提升到 `dP=8`：
  - 所有 bit-width 下精度均有提升
  - 特别是在 4-bit 时，避免了严重退化
  - 支持理论推导：更多 decay 时间可清除旧误差

#### （2）scale group 轴向比较
| 配置 | Perplexity (↓) | IFEval (↑) |
|------|----------------|-----------|
| value-axis group | 8.320 | 73.94 |
| **key-axis group (PACE)** | **7.734** | **78.93** ✅ |
> ✅ 沿 key axis 分组显著优于传统 value axis 分组

#### （3）head-level precision allocation
- 在相同内存下，head-level 保护优于 DAMP 的 channel-level 保护
- PACE-4/8 混合精度设计有效提升了鲁棒性

---

### 状态漂移（State Drift）分析（Figure 5）
- 度量量化状态与全精度状态之间的相对误差 $\|S_q - S_{fp}\|_F / \|S_{fp}\|_F$
- 所有方法中，随着 `dP` 增加，drift 明显下降
- 表明 **减少量化频率确实缓解了误差积累**

---

### 推理吞吐量测试（Figure 6）
- 实现了基于 Triton 的 packed 4-bit kernel
- 在 batch ≥ 16 时，**periodic INT4 decoding 的 TPOT 低于 bf16 baseline**
- 解码加速主要来自 decode buffer 减少内存访问，而 INT4 进一步优化了带宽使用
> ✅ **PACE 不牺牲解码速度，反而继承了 decode buffer 的加速优势**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **减少量化频率（periodic rounding）能显著降低误差积累**  
   > 即使是简单的 uniform quantizer，在 `dP=8` 时也能大幅提升 4-bit 性能。

2. ✅ **gated-delta state 的大值集中在列方向（columns）**  
   > 这是由 delta value 中某些 persistent channels 导致的，支持了 channel-aligned grouping 的合理性。

3. ✅ **key-axis scale grouping 优于 value-axis grouping**  
   > 更匹配实际数值分布，减少无效量化噪声。

4. ✅ **PACE-4/8 在 ~6.8 bits/value 下接近 DAMP INT8 的性能**  
   > 实现了近似八比特的质量，但内存节省约三分之一。

5. ✅ **周期性量化与 decode buffer 完美协同**  
   > 不仅不拖慢速度，还能维持原有的吞吐增益。

---

### 局限性（Limitations）
1. **模型泛化性有限**  
   - 当前实验仅限于 Qwen3.5/Qwen3.6 系列的 hybrid gated-delta 模型
   - 尚未验证是否适用于纯 SSM 或其他更新规则（如 Mamba-2、S4D）

2. **未探索完整配置空间**  
   - prefill period 固定为 256，未研究更细粒度影响
   - 缓冲区宽度、group size、head比例等超参未充分搜索

3. **缺少动态或随机舍入机制**  
   - 未尝试 stochastic rounding、static scales 或 rotation-based 方法结合 periodic rounding

4. **性能计时未覆盖全部组件**  
   - throughput 测试仅针对 uniform INT4，未包含 PACE 的复杂 scale group 和 head allocation 开销

---

### 未来工作方向
- 将 PACE 扩展至其他类型的 **linear attention** 和 **SSM 架构**
- 探索 **自适应 decode period**（根据 head decay 动态调整 `dP`）
- 结合 **rotation quantization**（如 Rotated Quantization）进一步压缩
- 研究 **prefill 阶段的周期性量化策略**
- 开发硬件友好的 kernel 支持 mixed-precision + packed layout 的高效执行

--- 

> 📌 **总结一句话**：  
> **PACE 通过“周期性舍入 + 通道对齐分组 + 头级混合精度”，实现了高质量、低内存、免微调的 4-bit recurrent state 量化，在多个大模型上媲美甚至超越现有 8-bit 方法，同时保持了解码加速优势。**

</details>

---

### 14. [RL-IF: Tracking Non-local Training Data Influence for Online Reinforcement Learning](https://arxiv.org/abs/2610.28883)

**Authors**: Shixuan Liu, Cheng Tang, Yuzheng Hu, Fan Wu, Han Zhao, Jiaqi W. Ma  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.28883v1  

#### Abstract
Online reinforcement learning (RL) has achieved significant success in decision-making tasks, but modern online RL remains highly sensitive to the quality of training experience. Understanding the contribution of individual training instances is therefore central to data-centric RL, with application...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RL-IF: Tracking Non-local Training Data Influence for Online Reinforcement Learning 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在线强化学习（Online Reinforcement Learning, RL）对训练数据质量高度敏感，而现有**数据归因方法**（data attribution）大多局限于单轮训练中的局部影响分析，忽略了训练样本在多轮更新中的**非局部影响**（non-local influence）。这导致无法准确衡量一个样本对长期策略性能的真实贡献。

具体挑战包括：
- 局部归因方法（如梯度内积）具有启发性但缺乏理论依据；
- 忽视了样本通过**优化路径**（optimization path）和**采样路径**（sampling path）对后续训练的双重影响；
- 现有方法难以捕捉跨训练轮次的数据价值。

### 提出的新方法与新思路
本文提出了一套系统性的**非局部数据归因框架**，核心贡献如下：

#### （1）定义了基于轨迹的 Leave-One-Out 度量：**RL-LOO**
- 形式化地定义了一个样本被移除后对最终策略目标函数的影响，跨越多个训练轮次。
- 该度量考虑了参数更新和未来 rollout 数据分布变化的联合效应。

#### （2）提出了高效的一阶估计器：**RL-IF**（Reinforcement Learning Influence Function）
- 推导出 RL-IF 作为 RL-LOO 的一阶近似，能够无需重训练即可估算样本的长期影响。
- 将数据影响分解为两个关键成分：
  - **Optimization Effect**：样本通过梯度更新影响后续参数。
  - **Sampling Effect**：样本通过改变策略进而影响未来 rollout 数据的分布。
- 给出了 RL-IF 的递归计算公式，并提供理论保证其近似精度。

#### （3）发现并验证：**优化效应主导长期影响**
- 理论证明和实验证据表明，在大多数场景下，**Sampling Effect 是 Optimization Effect 的相对小扰动**。
- 因此，仅保留优化效应的简化版本 **RL-IF-opt** 在实践中已足够有效。

#### （4）构建实用算法：**RINSE**（RL-INfluence-based Sample sElection）
- 基于 RL-IF-opt 设计了一个多轮窗口内的数据过滤方法。
- 利用非局部归因分数识别有害样本并进行剔除，提升训练效率和最终性能。

### 相比现有方法的优势
| 方面 | 现有方法（如 IIF, TracIn） | 本文方法（RL-IF / RINSE） |
|------|--------------------------|----------------------------|
| **影响范围** | 局部（单轮内） | 非局部（多轮间） |
| **理论基础** | 启发式（如梯度点积） | 基于 Leave-One-Out 的反事实因果视角 |
| **影响机制建模** | 仅优化路径 | 联合建模优化 + 采样路径 |
| **实用性** | 可能误判长尾样本价值 | 更准确反映样本真实长期价值 |
| **下游应用** | 有限 | 支持高效数据筛选（RINSE） |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖两类任务：

#### （1）标准控制基准环境（来自 Gym/Gymnasium）
- **MuJoCo 连续控制任务**：
  - `Ant-v4`, `HalfCheetah-v4`, `Hopper-v4`
- **Box2D 控制任务**：
  - `LunarLander-v2`, `BipedalWalker-v3`
- **表格导航任务**：
  - `FrozenLake-v1`

#### （2）大语言模型对齐任务（RLHF-style）
- **毒性缓解任务**（Toxicity Mitigation）
  - 基础模型：`GPT-Neo 2.7B`
  - 训练数据：`RealToxicityPrompts` 数据集
  - 测试数据：`Wiki-Toxic` 数据集（out-of-distribution）

### 实验设置与评估指标

#### 主要训练配置
- 使用 **KL 正则化的 PPO** 作为基础算法（避免 clip 目标带来的非光滑性）。
- 每个实验运行 **3 个随机种子**，报告均值 ± 标准误差。
- 所有代码开源：[https://github.com/sx-liu/RL-IF](https://github.com/sx-liu/RL-IF)

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Time Efficiency (TE)** | 达到目标性能所需时间（相对于标准 PPO 的百分比），越低越好 |
| **Sample Efficiency (SE)** | 达到目标性能所需训练数据量，越低越好 |
| **Peak Reward** | 最终达到的最大平均回报 |
| **Test Toxicity** | 在测试集上生成文本的平均毒性概率（越低越好） |
| **Linear Data Modeling Score (LDS)** | Spearman/Pearson 相关性，衡量归因分数预测子集删除效果的能力 |

### 基线方法对比
- **Standard PPO**：标准训练流程，无数据筛选。
- **Random Filtering**：随机剔除部分样本。
- **IIF** [Hu et al., 2025]：当前最先进的局部归因过滤方法，基于 TracIn 分数。
- **TracIn**：经典监督学习归因方法，用于 LDS 对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）标准控制任务性能汇总（图 3 & 表 7）
| 环境 | 方法 | SE (%) ↓ | TE (%) ↓ | Peak Reward ↑ |
|------|------|---------|---------|--------------|
| Ant-v4 | Standard | 100 | 100 | 734.3 |
| | IIF | 86.3 | 110.2 | 861.6 |
| | **RINSE** | **74.9** | **103.0** | **99.1** |
| HalfCheetah-v4 | Standard | 100 | 100 | 1316.6 |
| | IIF | 57.7 | 75.0 | 1437.9 |
| | **RINSE** | **45.4** | **63.9** | **195.0** |

> ✅ **RINSE 平均减少约 42.8% 的样本使用量**，且在多数环境下时间效率也更高。

#### （2）毒性缓解任务（图 4 & D.5）
- **相对训练速度提升**：
  - 相比 Standard：**>3× 加速**
  - 相比 IIF：**>2× 加速**
- **最终毒性水平显著更低**

---

### 与基线方法的对比结果
- **RINSE 显著优于所有基线**：
  - 在所有 6 个 RL 基准任务上，**RINSE 实现最高峰值奖励和最佳样本效率**。
  - 特别是在复杂连续控制任务（如 Ant, BipedalWalker）上优势最明显。
- **IIF 虽优于 Standard，但仍劣于 RINSE**：
  - 表明非局部归因确实带来了额外增益。
- **Random Filtering 效果不稳定**，有时甚至损害性能。

---

### 消融实验结果

#### （1）RL-IF-opt vs. RL-IF-full（表 1 & D.1）
| 方法 | Ant-v4 LDS ↑ | HalfCheetah-v4 LDS ↑ |
|------|-------------|--------------------|
| TracIn | 0.055 | 0.138 |
| RL-IF-opt | 0.191 | 0.218 |
| RL-IF-full | 0.192 | 0.218 |

> 🔍 **RL-IF-opt 与 RL-IF-full 性能几乎一致**，说明在这些任务中 **Sampling Effect 可忽略不计**。

#### （2）Horizon 选择消融（图 8 & D.3.2）
- 最优 horizon 约为 **H=4~5**。
- 过短（H=1）退化为局部方法；过长（H>8）带来 off-policy 不匹配，收益饱和甚至下降。

#### （3）Discard Ratio 消融（图 9 & D.3.3）
- 最佳丢弃比例约为 **p=10%~20%**。
- 过度剔除（p=50%~100%）会破坏训练稳定性，尤其在 HalfCheetah 上表现更敏感。

#### （4）Adaptive Horizon vs. Fixed Horizon（图 7）
- 自适应 horizon（动态停止）在所有任务上都**等于或优于固定 horizon**。
- 尤其在探索困难任务（如 Ant, BipedalWalker）上提升显著。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **非局部数据归因是必要的**：局部归因存在“晚期偏差”（late-episode bias），可能错误惩罚有益样本。
2. ✅ **RL-IF 成功建模了双重影响机制**：优化效应和采样效应的解耦提供了新的理解视角。
3. ✅ **优化效应主导长期影响**：尽管理论上存在双重路径，但在实际训练中，**Sampling Effect 通常是一个小扰动**。
4. ✅ **RL-IF-opt 已足够实用**：基于优化效应的简化估计器已在多种任务上取得优异性能。
5. ✅ **RINSE 显著提升训练效率**：通过非局部归因指导数据筛选，可大幅降低样本消耗并加速收敛。

### 方法的局限性
1. **算法依赖性强**：目前仅适用于 PPO 类 on-policy 算法，尚未扩展至 TRPO、SAC 或 off-policy 方法。
2. **优化器限制**：推导基于 SGD 更新，虽在 Adam 上经验有效，但缺乏对自适应优化器（如 Adam）的理论支持。
3. **Sampling Effect 被低估的风险**：当前结论基于特定平滑假设，若策略变化剧烈或环境反馈强耦合，采样效应可能变得重要（如 FrozenLake 中已有迹象）。
4. **计算开销较高**：每轮需额外一次反向传播计算归因分数，带来约 60%~130% 的 per-round 开销（见表 8）。

### 未来工作方向
1. 将 RL-IF 框架推广至其他 RL 算法（如 TRPO, SAC）和 off-policy 设置。
2. 发展支持 Adam 等自适应优化器的二阶归因方法。
3. 探索当 Sampling Effect 显著时的应用场景（如稀疏奖励、元学习）。
4. 结合主动学习或课程学习，实现动态数据采集与过滤闭环。
5. 将 RINSE 应用于更大规模 LLM 训练，进一步验证其可扩展性。

---

> 📌 **总体评价**：  
> 本论文提出了首个系统性的**非局部训练数据归因框架 RL-IF**，不仅在理论上严谨，而且通过 RINSE 实现了显著的工程价值。它揭示了“优化效应主导”的深刻洞见，为数据高效的在线 RL 提供了新范式。

</details>

---

### 15. [Local and Sparse Kernel Operator Learning for Incompressible Flows: An Operator Lagrange Formulation](https://arxiv.org/abs/2610.29009)

**Authors**: Ramansh Sharma, Matthew Lowery, Houman Owhadi, Varun Shankar  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2610.29009v1  

#### Abstract
We present a local and sparse kernel method for learning the solution operators of incompressible flows. It builds on our earlier property-preserving kernel method, whose global output reconstruction and dense operator regression scale poorly. Our method replaces the global output reconstruction by ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Local and Sparse Kernel Operator Learning for Incompressible Flows: An Operator Lagrange Formulation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**不可压缩流体流动解算子（solution operator）学习中的可扩展性和计算效率问题**。传统基于 kernel 的 operator learning 方法（如 PPKM 和 VKM）存在以下瓶颈：
- **全局输出重建（global output reconstruction）** 导致系统矩阵稠密，求解成本随输出点数呈立方增长 $O((dm)^3)$。
- **稠密核回归（dense operator regression）** 在训练样本量大时（$N > 10^4$）需要分解 $N \times N$ 的稠密核矩阵，计算复杂度为 $O(N^3)$，内存消耗为 $O(N^2)$。

这些问题限制了其在三维湍流等大规模问题上的应用。

### 提出的新方法与创新点
作者提出了一种**局部且稀疏的 kernel operator learning 方法**，结合了 Operator Lagrange Formulation，核心创新包括：

1. **局部、散度自由的输出重建（Local and Divergence-Free Output Reconstruction）**
   - 使用 **RBF-FD（Radial Basis Function-generated Finite Differences）** 框架，在每个输出点的邻域（stencil）上构建**局部插值**，替代全局插值。
   - 插值基函数由**散度自由的 kernel（如 divergence-free PHS 或 Matérn kernel）** 和**任意阶的散度自由多项式**构成，确保每个预测都严格满足 $\nabla \cdot u = 0$。
   - 支持周期性边界条件（通过周期性镜像节点实现）。
   - 对于湍流，采用**多尺度 Matérn kernel**，使其在网格细化时能逼近 Kolmogorov 统计特性。

2. **稀疏算子回归（Sparse Operator Regression）**
   - 设计了一个**在任意输入维度下正半定的紧支撑（compactly supported）kernel**。
   - 该 kernel 将输入投影到主成分空间（最多3维），在此低维空间使用 Wendland 函数定义支撑半径；剩余坐标则用 Matérn kernel 处理。
   - 该设计使得核矩阵**稀疏**，可通过 **sparse Cholesky 分解**高效求解，避免了稠密矩阵分解。

3. **算子拉格朗日权重（Operator Lagrange Weights）**
   - 引入拉格朗日形式的算子表示，使每次新输入的预测仅需一次**稀疏求解**（one sparse solve per new input）。
   - 权重作用于训练场的采样值，而非局部系数，从而**避免显式存储所有训练场的局部系数**（节省数十倍存储）。
   - 理论分析表明，相比直接求解系数，Lagrange 评估方式**不会放大求解误差**，尤其在全局插值导致系数严重抵消时优势显著。

4. **仿射增强（Affine Augmentation）**
   - 为捕捉输入中的线性趋势，引入输入主成分坐标的**仿射多项式**进行增强。
   - 通过鞍点系统联合求解 kernel 回归和仿射项，提升对具有强趋势算子的建模能力。

### 相比现有方法的优势
- ✅ **严格物理一致性**：所有预测自动满足 $\nabla \cdot u = 0$ 和周期性。
- ✅ **高可扩展性**：输出重建和算子回归均支持大规模问题（$N > 10^4$, $m > 10^3$）。
- ✅ **高效并行化**：所有局部系统可在 GPU 上批量并行处理。
- ✅ **低存储开销**：通过 Lagrange 权重避免存储局部系数。
- ✅ **高精度**：在多个基准上优于 dense PPKM、VKM 及多种神经算子（如 Geo-FNO, Transolver）。

---

## 2. 核心实验方法和设置

### 数据集
使用了来自 Sharma et al. [53] 的 **10 个不可压缩流体基准数据集**，涵盖：
- **2D 层流**（8个）：lid-driven cavity, cylinder flow, backward-facing step, buoyancy-driven cavity, merging vortices, Taylor-Green vortices（时空）。
- **3D 湍流**（2个）：forced isotropic turbulence, turbulent species transport。
- 每个数据集包含 10000 个训练对、100 验证对、100 测试对。

### 实验设置
- **训练规模**：在 $N = \{500, 1000, 5000, 10000\}$ 上进行嵌套训练。
- **验证与选择**：使用验证集选择最优的支撑半径（控制稀疏度）、正则化参数、kernel 形状参数。
- **硬件**：所有计算在单块 NVIDIA RTX 5090 GPU 上以双精度运行。

### 评估指标
- **测试误差（Test Error）**：预测速度场与真实场之间的**平均相对 $L^2$ 误差**：
  $$
  \text{Error} = \frac{\|v^* - \hat{v}\|_2}{\|v^*\|_2}
  $$
- **稀疏度**：核矩阵非零元密度 $d_K = \text{nnz}(K)/N^2$。
- **存储**：模型总存储（含核矩阵因子、训练数据等）。
- **预测时间**：对 100 个测试输入进行预测的中位时间。

### 基线方法对比
- **Dense PPKM**：作者之前的工作，使用全局插值和稠密核回归。
- **VKM**：Batlle et al. 的 kernel 方法，直接回归输出采样点。
- **Neural Operators**：Geo-FNO 和 Transolver（带/不带散度惩罚）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 数据集 | 本文最佳方法（$N=10000$） | Dense PPKM | Geo-FNO |
|--------|--------------------------|-----------|---------|
| Forced isotropic turbulence | **7.1×10⁻⁵** | 3.6×10⁻² | 1.5×10⁻⁴ |
| Turbulent species transport | **3.7×10⁻⁵** | 2.4×10⁻⁴ | 5.0×10⁻⁴ |
| Merging vortices | **1.9×10⁻⁶** | 1.1×10⁻⁴ | 1.1×10⁻³ |
| Lid-driven cavity | **4.7×10⁻⁷** | 1.4×10⁻⁶ | 1.5×10⁻⁴ |

> ✅ 所有数据集上，本文方法均**至少与 dense PPKM 一样准确**，且**显著优于神经算子**（提升 2 至 >10⁶ 倍）。

### 与基线方法的对比结果
- **精度**：在 10 个数据集中，本文方法在 9 个上取得最小测试误差，尤其在湍流和涡旋合并问题上优势巨大。
- **稀疏性**：验证选择的核矩阵密度大多 < 30%，且**随着 $N$ 增加而降低**（如 lid-driven cavity 从 20% 降至 10%），表明稀疏性可随数据规模自适应。
- **存储**：存储随 $N$ 近似线性增长，远优于稠密方法的 $O(N^2)$。
- **预测效率**：得益于稀疏求解和 GPU 并行，预测时间合理（如 lid-driven cavity 在 $N=10000$ 时约 70ms/100 场）。

### 消融实验结果
#### （1）Kernel 形状拟合 vs 固定形状
- **Turbulent species transport**：拟合形状使误差降低 **6 倍**。
- **Forced isotropic turbulence**：影响较小。

#### （2）仿射增强（Affine Augmentation）
- **Forced isotropic turbulence**：误差降低 **29 倍**（从 $2.8\times10^{-3}$ 到 $9.8\times10^{-5}$）。
- **Turbulent species transport**：误差降低 **12 倍**。
- **Taylor-Green vortices**：由于已做幅度归一化，增强后误差接近舍入精度。
- **其他层流问题**：影响较小（< 25% 变化），因其输入秩为 1。

#### （3）Lagrange 评估 vs 直接评估
- **局部输出重建**：两种方式精度相当（相对差异 < $10^{-5}$）。
- **全局输出重建**：Lagrange 评估**最高可提升近三个数量级**的精度（如 merging vortices 达 868 倍），因避免了系数抵消带来的误差放大。

---

## 4. 关键结论和发现

### 主要发现
1. **稀疏性是可行的**：在大多数流体问题中，**稀疏核矩阵即可达到甚至超越稠密方法的精度**，且稀疏度随训练集增大而提高。
2. **仿射增强至关重要**：对于具有强线性趋势的算子（如 forced isotropic turbulence），仿射增强可带来**超过一个数量级的误差下降**。
3. **Lagrange 评估兼具效率与稳定性**：不仅大幅减少存储需求，还能在数值上更稳定，尤其适用于病态输出重建。
4. **方法全面优于神经算子**：在相同数据集上，本文方法在精度上**全面碾压 Geo-FNO 和 Transolver**，且预测场严格满足物理约束。

### 方法的局限性
- **核矩阵仍可能变稠**：当输入高度相关或支撑半径过大时，核矩阵可能接近稠密（如 forced isotropic turbulence 始终选 90% 密度）。
- **存储仍受 $N^2$ 影响**：尽管稀疏，Cholesky 因子的填充（fill-in）仍可能导致存储随 $N^{1.4} \sim N^{2.4}$ 增长。
- **未耦合局部插值**：局部插值之间无耦合，可能导致场在 stencil 边界处不连续（尽管散度仍为零）。

### 未来工作方向
1. **更丰富的输入族测试**：在更多样化的输入分布上验证方法泛化性。
2. **连续化局部场**：使用 **partition of unity** 或 **potential-based blending** 实现 $C^0$ 连续的全局场。
3. **分层稀疏分解**：采用 hierarchical 或 multilevel sparse factorization 进一步降低存储和计算复杂度。
4. **扩展误差分析**：将舍入误差分析推广至包含仿射增强和前置权重应用的完整流程。

--- 

> **总结**：本文提出了一种**高效、精确、物理一致**的 kernel-based operator learning 框架，通过**局部插值 + 稀疏核 + Lagrange 权重**三大技术突破了传统方法的可扩展性瓶颈，在多个流体基准上实现了对神经算子的全面超越，为科学机器学习提供了新的范式。

</details>

---

### 16. [How Fine-Tuning Objectives Shape Layer-Wise Information in LLM Hallucination Detection](https://arxiv.org/abs/2610.27711)

**Authors**: Kaiyang Wan, Forrest Sheng Bao, Amin Ahmad, Yuxia Wang  
**Category**: cs.AI  
**Published**: 2026-10-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.27711v1  

#### Abstract
Under matched conditions, classifier-head fine-tuning (\textsc{cls}) paradigm consistently outperforms the LLM-as-Judge sequence-to-sequence (\textsc{s2s}) on binary hallucination detection. We trace this gap to the representations induced by the two objectives. Static analysis shows that \textsc{cl...

---

### 17. [From Comparison to Selection: Enhancing Cross-Scale Generalization of Pointer Networks Trained with Reinforcement Learning for Capacitated Drone Delivery Routing](https://arxiv.org/abs/2610.27936)

**Authors**: Sofiene Lassoued, Devara Tertia, Stefan Lier, Andreas Schwung  
**Category**: cs.AI  
**Published**: 2026-10-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.27936v1  

#### Abstract
Unmanned aerial vehicles (UAVs) enable fast, flexible last-mile and emergency delivery, but their routing remains challenging as the number of clients varies. Classical approaches based on mixed-integer programming and heuristics can scale poorly or require problem-specific design, motivating learni...

---

### 18. [RACE: RoPE-Aligned Attention Compression Based on Query-Key Activation Energies](https://arxiv.org/abs/2610.28385)

**Authors**: Irene Tenison, Abdulrahman Alabdulkareem, Daniel Karl I. Weidele, Nima Dehmamy, Dan Gutfreund  
**Category**: cs.AI  
**Published**: 2026-10-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.28385v1  

#### Abstract
Head-dimension compression can reduce the memory required by the key-value (KV) cache, but existing methods often reconstruct full-dimensional representations during inference, adding significant decoding FLOPs overhead. We formalize the design considerations for converting pretrained attention into...

---

### 19. [Target Regeneration Shrinks the Gains of Draft-Training Objectives in Speculative Decoding](https://arxiv.org/abs/2610.29013)

**Authors**: Jungseob Lee, Hogun Seong, Euijun Choi, Sugyeong Eo  
**Category**: cs.CL  
**Published**: 2026-10-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.29013v1  

#### Abstract
Speculative decoding accelerates large language models by verifying drafted tokens in parallel. New draft-training objectives are usually credited with the acceptance they add over cross-entropy on externally written responses, yet many pipelines let the target rewrite them. We ask how much of that ...

---

### 20. [RISK-KV: Quality-Aware Selective KV Cache Reconstruction for Distributed LLM Inference](https://arxiv.org/abs/2610.29544)

**Authors**: Wang Yuekai  
**Category**: cs.DC  
**Published**: 2026-10-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.29544v1  

#### Abstract
Request migration and interruption recovery in distributed large language model (LLM) inference rely on historical key-value (KV) caches. Retaining or transferring a complete cache incurs substantial state overhead, whereas replaying historical tokens requires additional computation. This paper prop...

---

### 21. [GaP-MoE : Explicit Expert Specialization in Mixture-of-Experts via Orthogonal Gradient Accumulation and Projection](https://arxiv.org/abs/2610.29355)

**Authors**: Donghyun Jeong, Yoon-Yeong Kim  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.29355v1  

#### Abstract
Sparse Mixture-of-Experts (SMoE) models improve scalability by routing input tokens to only a subset of expert networks, but they often suffer from representation collapse, in which multiple experts converge toward similar functions and lose their intended diversity. Prior work has addressed this li...

---

### 22. [A Transfer Learning-Driven Methodology for Efficient and Sustainable Indoor Localization Using VLC](https://arxiv.org/abs/2610.29594)

**Authors**: Masood Jan, Wafa Njima, Xun Zhang, Alexander Artemenko  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.29594v1  

#### Abstract
The rapid expansion of the Internet of Things enables seamless collaboration among connected devices, making indoor localization a critical component of industrial automation. However, conventional localization techniques struggle in dynamic environments. These methods depend on extensive, frequent ...

---

### 23. [DriveMLP: When Training Strategy Outperforms Model Complexity in Autonomous Driving](https://arxiv.org/abs/2610.29676)

**Authors**: Alessandro Canevaro, Julian Schmidt, Hang Yu, Ye Jin, Sajad Marvi, Georg Martius, Julian Jordan  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.29676v1  

#### Abstract
Learning motion planning policies that remain reliable under closed-loop execution is a central challenge in Autonomous Driving (AD). Imitation Learning (IL) scales well to large AD datasets but suffers from distribution shift, causal confusion, and compounding errors under closed-loop execution. In...

---

### 24. [Differentially Private Sparse Reward Estimation with Preference Feedback](https://arxiv.org/abs/2610.30099)

**Authors**: Meng Ding, Mingxi Lei, Jie Zhang, Jinyan Liu, Di Wang  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2610.30099v1  

#### Abstract
Reinforcement Learning with Human Feedback (RLHF) has become a central paradigm for aligning Large Language Models with human values. However, the reliance on sensitive preference data necessitates rigorous differential privacy (DP) guarantees to prevent user information leakage. Existing private al...

---

### 25. [SPEM: Structure-Aware Procedural Memory for Efficient Skill Use in LLM Agents](https://arxiv.org/abs/2610.26995)

**Authors**: Ran Zhou, Zehui Kong, Fanyu Lin, Jiaheng Liu, Zhen Zhang, Yuxiang Ren  
**Category**: cs.AI  
**Published**: 2026-10-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.26995v1  

#### Abstract
LLM-based agents increasingly rely on reusable skills. However, scaling skill libraries makes efficient and accurate skill use difficult. Full skill loading incurs high token and runtime overhead. In contrast, flat semantic retrieval treats skills as independent text entries, ignoring dependencies, ...

---

### 26. [LatBridge: Efficient and Stable Model Replacement in Latent-Communication Multi-Agent Systems](https://arxiv.org/abs/2610.27813)

**Authors**: Jiajun Zhu, Haobo Li, Zixin Chen, Huamin Qu, Wenshuo Zhang  
**Category**: cs.AI  
**Published**: 2026-10-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.27813v1  

#### Abstract
Multi-agent systems rely on communication to complete tasks, and latent communication passes continuous representations to reduce text generation and re-encoding. However, multi-agent systems must reconfigure agent models, whereas existing latent-communication methods are built for fixed model combi...

---

### 27. [STAR-XPC: Set Transformer for Estimating X-ray Transmission in Polymer Composites](https://arxiv.org/abs/2610.28709)

**Authors**: Can Erer, H. Birkan Yilmaz, Nebahat Aral  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.28709v1  

#### Abstract
Evaluating the energy transmission ratio of an X-ray beam through a polymer composite is a challenging task in radiation shielding studies. The reference solution with Monte Carlo transport simulation is highly accurate but computationally expensive: each of our reference simulations is capped at $4...

---

### 28. [Flat-RLVR: Sharpness-Aware Reinforcement Learning with Verifiable Rewards for Reasoning Capacity in LLMs](https://arxiv.org/abs/2610.29281)

**Authors**: Yeonbum Kim, Hyun Kyu Lee, Sung Whan Yoon  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.29281v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR), particularly through group-relative policy optimization (GRPO), has emerged as a central paradigm for the post-training of large language models (LLMs). While RLVR is highly effective at enhancing the reasoning capabilities of LLMs, it commonly ...

---

### 29. [CORD: Conflict-Aware Optimization for Joint RL and Distillation in Agents](https://arxiv.org/abs/2610.29965)

**Authors**: Wenhao Yu, Zhengcheng Wang, Jiahong Liu, Zexuan Qiu, Minda Hu, Shaohang Wei, Jiafei Lyu, Junyan Lyu, Haobo Fu, Irwin King  
**Category**: cs.LG  
**Published**: 2026-10-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2610.29965v1  

#### Abstract
Joint reinforcement learning and self-distillation combines task rewards with dense teacher guidance for agent training, but their gradients can partly cancel. A mixing coefficient controls relative scale without selectively removing opposing components. We present CORD, a general framework for conf...

---

### 30. [M-Plan: Comprehensive Planning For LLM Agents with Memory Arrangement](https://arxiv.org/abs/2610.27032)

**Authors**: Ziyang Liu, Jiakang Yuan, Jiayuan Fan, Tao Chen  
**Category**: cs.AI  
**Published**: 2026-10-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2610.27032v1  

#### Abstract
Large language model (LLM)-based agents have recently demonstrated outstanding performance in single-turn tasks, yet they continue to struggle with long-horizon tasks, particularly in environments without prior training. A key limitation of single LLMs is their difficulty in processing long contexts...

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

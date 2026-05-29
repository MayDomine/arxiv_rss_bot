# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-29 09:03:49 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding](https://arxiv.org/abs/2605.29707)

**Authors**: Jianuo Huang, Yaojie Zhang, Qituan Zhang, Hao Lin, Hanlin Xu, Linfeng Zhang  
**Category**: cs.CL  
**Published**: 2026-05-29  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.29707v1  

#### Abstract
Speculative decoding accelerates LLM inference by drafting multiple tokens and verifying them in parallel with the target model. However, its practical speedup is constrained by the trade-off between draft quality and drafting cost: autoregressive drafters model causal dependencies among draft token...

---

### 2. [Design and Implementation of a Serverless MapReduce Framework for Scalable Data Pipelines](https://arxiv.org/abs/2605.29573)

**Authors**: Angelos Dorotheos Chatzopoulos, Babis Andreou, Kakia Panagidi, Stathes Hadjiefthymiades  
**Category**: cs.DC  
**Published**: 2026-05-29  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.29573v1  

#### Abstract
Modern logistics systems tend to generate continuous streams of data from sources such as GPS, IoT sensors, and logistics management systems. The aggregation, processing, and analysis of data have become vital for monitoring operations, optimizing efficiency, and responding quickly to decision makin...

---

### 3. [RightNow-Arabic-0.5B-Turbo: An Open Sub-1B Arabic Language Model via Vocabulary Injection and Edge-First Deployment](https://arxiv.org/abs/2605.28827)

**Authors**: Jaber Jaber, Osama Jaber  
**Category**: cs.CL  
**Published**: 2026-05-29  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.28827v1  

#### Abstract
Open Arabic large language models split into two classes: sub-1B multilingual models that treat Arabic as an afterthought (Qwen2.5-0.5B, Falcon-H1-0.5B), and 7B-70B Arabic-specialized models that require a server to run (Jais, AceGPT, ALLaM, SILMA). The one published attempt at a sub-2B Arabic-speci...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：RightNow-Arabic-0.5B-Turbo**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前开源的阿拉伯语大语言模型（Arabic LLM）存在显著断层：
- **小型模型**（<1B 参数）如 Qwen2.5-0.5B、Falcon-H1-0.5B 是多语言模型，对阿拉伯语支持薄弱，未专门优化其 tokenizer 或训练数据。
- **大型模型**（7B–70B 参数）如 Jais、ALLaM、SILMA 虽然在阿拉伯语任务上表现优异，但需要服务器级硬件部署，无法用于手机、嵌入式设备等边缘场景。

因此，**缺乏一个轻量级、专为阿拉伯语优化且可公开获取权重的 sub-1B 模型**，限制了阿拉伯语在边缘计算中的实际应用。

### **提出的新方法与思路**
作者提出了 **RightNow-Arabic-0.5B-Turbo**，一个仅 **518M 参数** 的阿拉伯语专用 decoder LLM，通过以下关键技术实现：

1. **Vocabulary Injection（词表注入）**  
   在 Qwen2.5-0.5B 基础上，新增 **27,032 个阿拉伯语专属 token**，并通过 **mean-subtoken initialization** 初始化新 token 的 embedding 向量，有效降低阿拉伯语分词冗余度（fertility），提升编码效率。

2. **Edge-First Deployment Pipeline（面向边缘部署的全流程设计）**  
   从训练到推理全程考虑边缘设备需求：
   - 使用 FSDP + FlashAttention varlen + Liger fused kernels 高效训练；
   - 采用 response-only loss masking 进行 SFT，聚焦于生成质量；
   - 最终导出为 GGUF 格式并量化至 **q4_k_m（398MB）**，可在单张 H100 上以 **635 tokens/s** 推理速度运行（bs=1）。

3. **Weight Soup Merging（权重融合）**  
   将 pretrain、SFT 和 DPO 三个阶段的 checkpoint 进行线性加权平均（linear weight soup），选择最优组合提升泛化能力。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **模型大小** | 是目前最小的开源阿拉伯语专用 decoder LLM（518M），远小于 Kuwain-1.5B（未开源）、AceGPT-7B 等 |
| **性能表现** | 在 COPA-ar 上与 **Falcon-H1-1.5B（1.5B 参数）持平（58.4%）**，但参数仅为其 1/3 |
| **资源开销** | 量化后仅 **398MB**，适合手机、浏览器、IoT 设备部署 |
| **开放性** | 完全开源：代码（5,555 行）、权重（bf16/int8/GGUF）、训练脚本、评估脚本全部发布于 HuggingFace |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

| 阶段 | 数据集 | 规模 | 来源 |
|------|--------|-------|------|
| **继续预训练（Continued Pretraining）** | Arabic Wikipedia（经新 tokenizer 编码） | 504M tokens | `wikimedia/wikipedia` |
| **监督微调 SFT** | 合并五个阿拉伯语指令数据集：<br>- `evol-instruct-arabic`<br>- `alpaca-gpt4-arabic`<br>- `sharegpt-arabic`<br>- `CIDAR`<br>- `aya_dataset`（阿拉伯子集） | 129,116 条去重样本 | 公共数据集 |
| **直接偏好优化 DPO** | `argilla-dpo-mix-7k-arabic` 中的阿拉伯语偏好对 | 6,750 对 | Argilla 开源数据 |
| **评估基准** | lm-evaluation-harness 中的三个阿拉伯语任务：<br>- **COPA-ar**（因果推理）<br>- **Arabic MT HellaSwag**（常识推理）<br>- **ArabicMMLU**（多任务知识理解） | COPA: 500, HellaSwag: ~10k, MMLU: 14,575 题 | 自定义评测集 |

> ⚠️ 注：原计划使用 FineWeb-2-ar，但因 HuggingFace Hub 多进程流式加载时频繁出现 504 错误，改为预先 tokenize 成 flat int32 memmap 文件。

### **实验设置与评估指标**

- **训练平台**：8×H100 SXM5（Nebius 实例），CUDA 13.0，PyTorch 2.11
- **训练策略**：
  - **FSDP** + `_HYBRID_SHARD_ZERO2` 分片优化器状态
  - **FlashAttention-varlen** 支持变长序列打包，保留文档边界
  - **Liger Kernel** 替换 RMSNorm、RoPE、SwiGLU、fused CE loss，节省显存
- **评估方式**：
  - 使用 `lm-evaluation-harness v0.4.11`
  - 所有模型统一设置：`apply_chat_template=True`, `batch_size=2`, `max_length=1536`, `limit=200`
  - 优先使用 `acc_norm` 指标（归一化准确率）
- **对比基线模型**：
  - 同类小模型：Qwen2.5-0.5B-Instruct、Falcon-H1-0.5B-Instruct
  - 更大模型：Falcon-H1-1.5B、AceGPT-7B、ALLaM-7B、SILMA-9B

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 模型 | 参数量 | COPA-ar | HellaSwag-ar | ArabicMMLU | **均值** |
|------|--------|---------|--------------|------------|----------|
| Qwen2.5-0.5B-Instruct | 494M | 53.9% | 22.5% | 26.0% | 34.1% |
| Falcon-H1-0.5B-Instruct | 524M | 44.9% | 23.0% | 24.2% | 30.7% |
| **Ours (RightNow-Arabic-0.5B-Turbo)** | **518M** | **58.4%** | **26.0%** | **23.2%** | **35.9%** |
| Falcon-H1-1.5B-Instruct | 1.5B | 58.4% | 27.5% | 32.7% | 39.5% |
| SILMA-9B-Instruct | 9B | 69.7% | 38.0% | 52.9% | **53.5%** |

> ✅ **亮点**：
> - 在 **COPA-ar** 上达到 **58.4%**，**与 1.5B 模型 Falcon-H1-1.5B 并列第一**
> - **均值 35.9%**，超越所有同级别 sub-1B 模型
> - 以 **1/18 参数量** 达成 SILMA-9B **67% 的平均性能**

### **与基线方法的对比结果**

- **优于所有 sub-1B 多语言模型**：
  - COPA-ar：+4.5pp vs Qwen2.5-0.5B，+13.5pp vs Falcon-H1-0.5B
  - HellaSwag-ar：+3.5pp vs Qwen2.5-0.5B
  - 均值领先 1.8–5.2 个百分点

- **接近甚至匹敌更大模型**：
  - 在 **COPA-ar** 上 **完全匹配 Falcon-H1-1.5B（58.4%）**
  - 在 **ArabicMMLU** 上仍有差距（约 10–30pp），体现知识容量瓶颈

### **消融实验结果（Ablation Study）**

#### **Weight Soup Merging 消融（Table 4）**

| 合并策略 | 均值准确率 | 相比 DPO 单独提升 |
|----------|------------|------------------|
| DPO checkpoint（baseline） | 35.20% | — |
| Linear soup (DPO 0.5, SFT 0.25, Pretrain 0.25) | **35.64%** | **+0.44pp** |
| LERP(DPO, Pretrain, t=0.5) | 35.38% | +0.18pp |
| SLERP(DPO, Pretrain, t=0.5) | 35.36% | +0.16pp |

> 🔍 发现：
> - **三阶段线性融合（DPO:SFT:Pretrain = 0.5:0.25:0.25）效果最佳**
> - 单纯合并 SFT 会劣化性能（LERP(DPO,SFT)=34.63%）
> - DPO 本身几乎没有改进（loss 几乎不变），说明偏好信号较弱

#### **Tokenizer 效率测试（Table 5）**

| Tokenizer | 分词总数（368词样本） | Fertility（每词 token 数） | 下降幅度 |
|-----------|------------------------|----------------------------|----------|
| Qwen2.5-0.5B 原始 | 803 | 2.18 | — |
| 本文改进版 | 664 | **1.80** | ↓ **17.3%** |

> 💡 意义：更高效的 tokenizer 意味着更少的 forward pass，直接转化为推理加速。

#### **推理速度测试（Table 6）**

| Quantization | 磁盘占用 | Prompt Eval (tok/s) | Generation (tok/s) |
|--------------|----------|---------------------|--------------------|
| f16 | 988 MB | 634.0 | 582.4 |
| q8_0 | 525 MB | 732.8 | 645.7 |
| q5_k_m | 419 MB | 718.5 | 633.5 |
| **q4_k_m** | **398 MB** | **723.6** | **634.9** |

> 🚀 结论：即使在 **398MB 量化版本下**，仍可达 **>630 tokens/s**，远超 HuggingFace 默认生成流程（~82 tok/s）。

---

## **4. 关键结论和发现**

### **主要发现**

1. **Sub-1B 阿拉伯语专用模型是可行的**  
   无需全新架构或训练范式，只需精心设计 pipeline（vocab injection + efficient training + edge-aware export），即可构建高性能的小型化阿拉伯语模型。

2. **Tokenizer 优化至关重要**  
   通过注入阿拉伯语专属 token，将 fertility 降低 **17.3%**，显著提升了推理效率和上下文利用率。

3. **Weight Soup 可缓解过拟合与遗忘**  
   DPO 阶段几乎无进展，但通过融合早期 checkpoint，反而提升了最终性能，表明“记忆保留”比“偏好学习”更重要。

4. **边缘部署必须端到端优化**  
   从训练（memmap loader 避免 Hub stall）、量化（GGUF + llama.cpp）到推理（CUDA graph），每个环节都需针对目标设备优化。

### **局限性**

| 局限 | 说明 |
|------|------|
| **知识容量受限** | ArabicMMLU 明显落后于 7B+ 模型（差 29+ pts），表明 sub-1B 模型难以胜任复杂知识任务 |
| **DPO 信号弱** | 当前 DPO 数据集由机器翻译生成，噪声大；且 0.5B 模型可能不足以捕捉细粒度偏好 |
| **仅支持 MSA** | 预训练语料为现代标准阿拉伯语（MSA），对埃及、海湾、黎凡特等方言处理能力有限 |
| **Tokenizer tile alignment 问题** | 新增词表导致 GGUF 量化 fallback 至更高 bit-width，实际 bpp 为 6.45（q4_k_m）而非理想 4 |

### **未来工作方向**

1. **引入真实母语者标注的 DPO 数据集**，增强偏好学习信号
2. **加入方言数据进行混合训练**，提升对口语化阿拉伯语的理解
3. **探索第二轮 vocabulary expansion**，添加高频短语（multi-word expressions）进一步压缩 token 数
4. **接入更多高质量阿拉伯语语料**（如 CulturaX-ar、FineWeb-2-ar），突破当前 ~1 token/param 的训练比例限制
5. **优化 GGUF 量化块对齐机制**，真正实现 4-bit 高效推理

---

> ✅ **总结一句话**：  
> **RightNow-Arabic-0.5B-Turbo 是首个真正可用于边缘设备的开源阿拉伯语专用小模型，在保持极小体积（398MB）的同时，性能超越同类 sub-1B 模型，并在部分任务上媲美三倍大的模型。**  
> 项目地址：[https://huggingface.co/RightNowAI/RightNow-Arabic-0.5B-Turbo](https://huggingface.co/RightNowAI/RightNow-Arabic-0.5B-Turbo)

</details>

---

### 4. [Cluster-Level Attention-Guided Parallel Decoding for Masked Diffusion Language Models](https://arxiv.org/abs/2605.29607)

**Authors**: Heqiang Qi, Wei Huang, Mingyuan Bai, Xiangming Meng  
**Category**: cs.LG  
**Published**: 2026-05-29  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.29607v1  

#### Abstract
Masked diffusion language models (MDLMs) enable parallel decoding by predicting all masked positions at each denoising step, yet existing training-free samplers usually decide which positions to commit at token-level granularity. We revisit this granularity and observe that reliable predictions ofte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Cluster-Level Attention-Guided Parallel Decoding for Masked Diffusion Language Models**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **Masked Diffusion Language Models (MDLMs)** 虽然支持并行解码，但大多数训练无关（training-free）的采样器（samplers）在**token-level granularity**（词元级粒度）上决定哪些位置应被提交（committed）。这种逐个词元的决策限制了解码效率，因为即使多个相邻词元都具有高置信度，仍需分步处理，导致不必要的迭代次数。

此外，过于激进地并行提交多个 token 可能引入**语义冲突**（incompatible predictions），尤其是在强依赖的位置之间同时解码时。

### **提出了什么新方法或新思路**
本文提出 **CLAD (Cluster-Level Attention-Guided Decoding)**，一种全新的训练无关的 cluster-level 并行解码框架，其核心思想是：

- **将解码单位从单个 token 提升为连续的高置信度片段（span）**，即 **Confidence-Induced Clusters (CICs)**：通过将相邻的高置信度 masked positions 分组形成最大连续段作为更新单元。
- 利用模型前向传播中的 **self-attention maps** 来估计不同 CIC 之间的依赖关系，并构建一个稀疏的 **conflict graph**，以识别不应同时提交的 cluster 对。
- 在每一步中选择一组互不冲突且权重最大的 CICs 进行并行提交，从而实现更高效、更安全的并行化。

### **相比现有方法的优势**
- **更高的并行度**：一次可提交整个 span，显著减少 denoising 步数。
- **更好的兼容性控制**：基于 attention 的跨 cluster 依赖建模避免了语义冲突，优于仅考虑 token-level 依赖的方法。
- **无需额外训练**：完全基于已有模型输出进行调度，保持 training-free 特性。
- **通用性强**：适用于多种 MDLM 架构（如 LLaDA 和 Dream 系列）。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
实验覆盖两个任务类别，共四个基准数据集：
- **数学推理（Mathematical Reasoning）**
  - **GSM8K (5-shot)**：小学数学应用题
  - **MATH (4-shot)**：高中及以上难度数学问题
- **代码生成（Code Generation）**
  - **MBPP (3-shot)**：面向编程任务的自然语言到 Python 代码生成
  - **HumanEval (0-shot)**：函数级代码补全挑战

### **实验设置和评估指标**
- **模型**：
  - LLaDA-8B-Instruct, LLaDA-1.5
  - Dream-v0-Base-7B, Dream-v0-Instruct-7B
- **评估指标**：
  - **Accuracy / Pass Rate**：任务准确率或执行通过率（code）
  - **Tokens Per Second (TPS)**：端到端吞吐量
  - **Speedup**：相对于 Vanilla 解码的速度提升倍数
- **实现细节**：
  - 固定生成长度为 256，block size 为 32
  - 所有方法均在相同硬件环境下测试（NVIDIA RTX 4090 / L40 GPU）

### **基线方法对比**
| 方法 | 类型 | 关键机制 |
|------|------|---------|
| **Vanilla** | 基线 | Top-1 逐 token 提交 |
| **Fast-dLLM** | 不确定性感知 | 高置信度阈值并行提交 |
| **KLASS** | 分布稳定性 | 结合 confidence 与跨步骤分布稳定 |
| **DAPD**, **DAWN** | 依赖感知 | 基于 attention 的 token-level 冲突检测 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
在 **LLaDA-8B-Instruct** 和 **Dream-v0-Instruct-7B** 上的结果如下表所示（摘自 Table 1）：

| Benchmark | Method | Speedup (×) | Accuracy Change |
|----------|--------|-------------|------------------|
| GSM8K | CLAD | **4.90×** | ±0.08 |
| MATH | CLAD | **3.76×** | -0.92 |
| MBPP | CLAD | **4.89×** | -1.40 |
| HumanEval | CLAD | **4.24×** | 0.00 |

> ✅ **总体速度提升范围：1.77× ~ 8.47×**，最高达 **8.47×**（在 MBPP 上使用 LLaDA-1.5）

### **与基线方法的对比结果**
- **在绝大多数设置下，CLAD 实现了最高的 TPS 和 Speedup**，尤其在 MBPP 和 GSM8K 上优势明显。
- 相较于 token-level 依赖感知方法（如 DAPD、DAWN），CLAD 依然取得更高吞吐：
  - 在 LLaDA-8B-Instruct 上，CLAD 比 DAWN 快约 **1.1–1.2×**
  - 表明 **cluster-level 更新单元本身带来了额外并行增益**
- 尽管部分任务出现轻微精度下降（如 GSM8K 上 -2.80 pts），但多数情况下精度“broadly comparable”（基本相当）

### **消融实验结果**
#### （1）**组件消融（Table 2）**
| 变体 | CIC | Conflict Graph | MWIS | Acc. | TPS |
|------|-----|----------------|-------|------|-----|
| Vanilla | × | × | × | 40.24 | 16.66 |
| +CIC | √ | × | × | 36.58 | **76.03** |
| +Graph | √ | √ | × | 36.58 | 66.13 |
| +MWIS | √ | √ | √ | **40.24** | **70.65** |

> 🔍 发现：
> - 仅使用 CIC 可大幅提升速度，但严重损害 accuracy（因无冲突控制）
> - 加入 conflict graph 后未恢复 accuracy → 表明需要智能选择策略
> - 引入 MWIS（最大权独立集）后，accuracy 恢复至 Vanilla 水平，同时保持高速度

#### （2）**超参数敏感性分析**
- **confidence threshold T**：较低 T 提高并行度但可能降低可靠性；适中值（如 0.7–0.75）平衡性能
- **block size**：CLAD 在多种 block size 下表现稳健，在 32 处达到最优
- **generation length**：在不同生成长度（128–1024）下均保持有效加速

---

## 4. **关键结论和发现**

### **主要发现**
1. **高置信预测常以连续 span 形式出现**，因此将解码单位从 token 升级为 cluster 是合理且高效的。
2. **CIC + attention-guided conflict detection + MWIS 选择** 构成了一个高效且鲁棒的 pipeline。
3. **cluster-level 解码显著提升了 MDLM 的推理吞吐量**，平均提速 **4–8×**，而任务准确性基本不受影响。
4. **attention map 可作为轻量级代理信号**，用于捕捉 cluster 间的潜在依赖，虽非理论完备，但在实践中非常有效。

### **方法的局限性**
- **Attention 是启发式信号**：不能完全反映逻辑或语义上的因果依赖，小 attention score 不排除隐含冲突。
- **效果受模型和任务影响**：
  - 当候选更新高度碎片化或 cluster 间依赖密集时，并行空间受限。
  - 某些细粒度推理任务可能不适合大跨度提交。
- **并非所有场景下都优于 token-level 方法**：个别任务（如 Dream 上的 GSM8K）存在精度损失。

### **未来工作方向**
- 探索更精确的 cluster-level 依赖建模方式（如 causal tracing 或 probing-based 方法）
- 动态调整 confidence threshold 或 cluster merging 策略
- 将 CLAD 思想扩展至其他生成范式（如图像、音频 diffusion models）
- 研究如何结合 preference modeling 或 RL 来优化 cluster 提交顺序

---

> 📌 **总结一句话**：  
> **CLAD 通过将解码粒度从 token 提升为 confidence-induced cluster，并利用 self-attention 构建 conflict-aware 调度机制，在几乎不牺牲 accuracy 的前提下实现了高达 8.47× 的推理加速，为 MDLM 的高效部署提供了新范式。**

</details>

---

### 5. [Efficient Test-Time Finetuning of LLMs via Convex Reconstruction and Gradient Caching](https://arxiv.org/abs/2605.30337)

**Authors**: Alaa Khamis, Alaa Maalouf  
**Category**: cs.LG  
**Published**: 2026-05-29  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.30337v1  

#### Abstract
Test-time finetuning (TTFT) is a rapidly evolving paradigm that adapts a language model to each prompt by retrieving related sequences, updating the model on them, and then evaluating the prompt. However, TTFT is only practical if it is fast: selection and finetuning both happen per query, making ea...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient Test-Time Finetuning of LLMs via Convex Reconstruction and Gradient Caching

---

## 1. 论文的主要贡献和创新点

### 解决的问题
**Test-Time Finetuning (TTFT)** 是一种在推理时动态适应语言模型的方法，通过为每个查询（prompt）检索相关文本并进行微调来提升模型表现。然而，TTFT 的瓶颈在于其**高延迟**，因为每一步都涉及两个耗时操作：
- **数据选择（Data Selection）**：从大规模语料库中检索与当前查询相关的序列。
- **微调（Finetuning）**：对检索到的数据执行梯度更新。

现有方法面临“质量-效率”权衡：
- **kNN 检索**速度快但缺乏多样性，容易选到重复内容；
- **SIFT** 等多样性感知方法能选出更丰富的样本，但计算开销大，难以满足低延迟需求。

### 提出的新方法：HullFT
本文提出 **HullFT**，一个基于几何重建与梯度缓存的高效 TTFT 框架，包含三个核心组件：

#### （1）**凸组合重建（Convex Reconstruction via Frank-Wolfe）**
将查询嵌入 $ q $ 表示为其邻域候选池中少数训练序列的稀疏凸组合：
$$
\min_{w \in \Delta^K} \|q - Pw\|^2
$$
其中 $ P $ 是候选序列的嵌入矩阵，$ w $ 是权重向量。该过程使用 **Frank-Wolfe 算法**求解，具有以下优势：
- 投影自由（projection-free），仅需内积运算；
- 天然生成稀疏且多样化的支持集（support set），避免冗余；
- 不依赖显式的多样性惩罚项或贪心搜索。

#### （2）**几何整数化（Geometric Integerization）**
将 Frank-Wolfe 输出的分数权重 $ w $ 转换为整数多重集（multiset）以用于实际微调。目标是找到整数计数 $ c_i $，使得均匀采样下的均值尽可能接近原始凸组合：
$$
\min_c \left\| q - \frac{1}{N}\sum_j c_j s_j \right\|^2,\quad \text{s.t. } \sum c_j = N
$$
此过程分为三步：
1. **向下取整分配**（Floor Allocation）
2. **贪婪填充剩余预算**
3. **局部交换优化**

这不仅使方法可执行，还自然产生**重复样本**，为后续梯度复用提供机会。

#### （3）**梯度重用（Gradient Reuse）**
利用整数化后出现的重复样本，在连续处理相同样本时不重新计算前向/反向传播，而是**每隔 $ r $ 步刷新一次梯度**：
$$
g_t = 
\begin{cases}
\nabla_\theta \mathcal{L}(\theta; x) & \text{if } t \mod r = 0 \\
g_{t-1} & \text{otherwise}
\end{cases}
$$
从而将梯度计算次数从 $ N $ 减少至约 $ \lceil N/r \rceil $，显著降低微调成本。

---

### 相比现有方法的优势
| 维度 | kNN | SIFT | HullFT（本文） |
|------|-----|------|----------------|
| **选择速度** | 快 | 慢（+8.8×） | 更快（+12× 平均加速） |
| **多样性控制** | 差（易重复） | 显式建模 | 隐式由几何决定 |
| **微调效率** | 无优化 | 无优化 | 支持 Gradient Reuse（平均 1.48× 加速） |
| **端到端延迟** | 中等 | 高 | 最低 |
| **性能（BPB%）** | 较差 | 较好 | **最优** |

> ✅ **核心思想**：通过**凸逼近的几何性质**同时实现**相关性与多样性**，并通过**整数化诱导的结构**启用**梯度级优化**，形成双重加速。

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **The Pile** 的 12 个子集进行评估：
  - ArXiv, DM Mathematics, Enron, FreeLaw, GitHub, HackerNews, NIH, PubMed Abstracts, PubMed Central, StackExchange, USPTO, Wikipedia
- 每个子集随机选取 150 个测试 query，确保跨领域泛化能力分析。

### 实验设置
- **基础模型**：GPT-2
- **嵌入模型**：normalized RoBERTa encoder
- **优化器**：Adam ($ lr = 5\times10^{-5} $)
- **候选池构建**：对每个 query 预先检索 $ K=200 $ 个最近邻（via FAISS）
- **微调预算**：$ N \in [1, 50] $
- **硬件平台**：单块 NVIDIA A100 GPU（主实验）、CPU-only 对比

### 评估指标
- **Bits-Per-Byte (BPB%)**：相对于未微调基线的压缩率改进，越低越好。
- **总运行时间（Total Runtime）**：包含 selection + finetuning 时间，反映实际部署延迟。
- 报告 **BPB% vs. 总运行时间** 曲线，并比较在固定时间预算下的性能差距。

### 基线方法
- **kNN [5]**：直接取 top-$N$ 最近邻进行微调。
- **SIFT [8]**：基于信息增益的选择策略，考虑冗余性，但计算昂贵。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 时间预算 $ T $ | HullFT BPB% | 最佳基线 BPB% | HullFT 提升幅度 |
|------------------|-------------|---------------|------------------|
| 1.75 秒          | 74.48       | 78.31         | **3.83% ↓**       |
| 2.00 秒          | 73.12       | 76.56         | **3.44% ↓**       |

> 在所有 $ T \leq 4.5s $ 的预算下，HullFT 均优于任一基线，尤其在**紧时间约束下优势最大**。

### 与基线方法的对比结果
- **质量-效率帕累托前沿**：如图 2 所示，HullFT 在所有时间点上均占据主导地位（Pareto-dominant）。
- **平均选择速度**：
  - SIFT: 0.524s @ $ N=50 $
  - HullFT: 0.059s @ $ N=50 $ → **8.8× 更快**
  - 全范围平均：**12× 选择加速**
- **微调加速**（Gradient Reuse）：
  - 平均减少 1.48× 微调时间，仅带来 0.64% BPB% 损失
  - 在重复较多的子集（如 NIH, USPTO）上效果更明显

### 消融实验结果
#### （1）Gradient Reuse 影响（Table 2）
| 子集 | GR 加速倍数 | BPB% 损失 |
|------|--------------|-----------|
| ArXiv | 1.54× | +0.71% |
| GitHub | 1.00×（几乎无重复） | ~0 |
| NIH | 1.64× | +0.37% |
| **平均** | **1.48×** | **+0.64%** |

> 表明 GR 在保持精度的同时有效降低成本。

#### （2）不同 $ r $ 设置的影响（Figure 4）
- $ r=2 $：最佳平衡点，大幅节省时间（~1.5×），BPB% 损失极小
- $ r=3 $：进一步提速，但质量下降较明显

#### （3）SIFT 上尝试 Gradient Reuse（Table 3）
| 方法 | BPB% | 时间(s) |
|------|-------|--------|
| SIFT (baseline) | 65.89 | 6.38 |
| SIFT + consecutive grouping | 66.03 | 6.14 |
| SIFT + global dedup ($ r=2 $) | 67.45 | 4.16 |
| **HullFT ($ r=2 $)** | **67.70** | **3.82** |

> 即使强行对 SIFT 应用 GR，也会因打乱原有顺序而导致性能下降；而 HullFT 因设计上就支持连续块，无此问题。

#### （4）CPU-only 实验（Figure 6）
- HullFT 选择时间：0.036s vs SIFT 0.934s → **25.8× 加速**
- 总端到端时间节省约 89s
- BPB% 接近 SIFT，远优于 kNN

---

## 4. 关键结论和发现

### 主要发现
1. **几何即多样性**：无需显式建模冗余，**凸逼近本身就能导出多样化且相关的样本集合**。
2. **整数化不仅是必要步骤，更是效率杠杆**：它产生的重复结构天然支持 **Gradient Reuse**，实现了从算法设计到系统优化的协同增效。
3. **双重加速机制**：
   - 选择阶段：Frank-Wolfe 比 SIFT 快 12×
   - 微调阶段：Gradient Reuse 再提速 1.48×
   - 合力让 HullFT 在相同时间内完成更大规模的微调（更高 $ N^* $），从而获得更强性能。
4. **特别适合低延迟场景**：在 $ T < 4s $ 的严格延迟要求下，HullFT 的优势最为显著，正是实际应用中最关心的区间。

### 方法的局限性
1. **依赖预检索池的质量**：HullFT 在 $ kNN $ 池上运行，因此其上限受制于上游检索器的召回率和嵌入空间表达能力。
2. **静态嵌入空间**：使用的嵌入是固定的（RoBERTa），未针对目标任务或模型损失曲面进行适配，可能错失潜在增益。
3. **Frank-Wolfe 收敛行为受限于支持集大小 $ m $**：实验中通常由 $ m $ 提前终止而非误差容忍度 $ \epsilon $，说明参数调优仍有空间。

### 未来工作方向
- 将 HullFT 与 **end-to-end retrieval model** 结合，联合优化嵌入空间与选择机制。
- 探索 **自适应 $ r $** 或 **动态刷新策略** 来进一步提升 Gradient Reuse 效果。
- 扩展至 **多模态模型** 或 **长上下文任务** 中的应用。
- 研究如何将该框架应用于 **few-shot learning** 或 **instruction tuning** 场景中的数据子集选择。

---

> 🔚 **总结一句话**：  
> HullFT 通过**几何重建 + 整数化诱导的梯度复用**，实现了**高质量与高效率兼得**的 TTFT 新范式，在多个 Pile 子集上验证了其在低延迟场景下的显著优势。

</details>

---

### 6. [Robust and Efficient Guardrails with Latent Reasoning](https://arxiv.org/abs/2605.29068)

**Authors**: Siddharth Sai, Xiaofei Wen, Muhao Chen  
**Category**: cs.AI  
**Published**: 2026-05-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.29068v1  

#### Abstract
Maintaining the safety of large language models (LLMs) is crucial as they are increasingly deployed in real-world applications. Existing safety guardrails typically rely on single-pass classification or, more recently, distilled reasoning. Reasoning-based guardrails significantly outperform classifi...

---

### 7. [Anchorless Diversification for Parallel LLM Ideation](https://arxiv.org/abs/2605.30150)

**Authors**: Fares Nabil Ibrahim, Nafis Saami Azad, Raiyan Abdul Baten  
**Category**: cs.AI  
**Published**: 2026-05-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.30150v1  

#### Abstract
LLMs are increasingly used to generate candidate-idea pools for creative tasks where broad exploration is valuable. Parallel inference can be attractive in this setting when it broadens the pool while retaining quality and cost efficiency. We study inference-time controls for candidate-pool diversif...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Anchorless Diversification for Parallel LLM Ideation》总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前在使用 **Large Language Models (LLMs)** 进行创造性任务（如生成故事前提、产品概念、广告语等）时，常采用 **parallel inference**（并行推理）来生成大量候选想法。然而，由于 LLMs 的生成倾向于集中在某些“语义盆地”（semantic basins），导致输出高度重复，缺乏多样性（diversity collapse）。这削弱了并行生成的实际价值。

传统提升多样性的方法通常依赖于观察已生成的种子样本（seed ideas），例如通过 **anti-anchoring**（反锚定）机制让模型避开已有输出。这类方法虽然有效，但引入了额外的计算开销（如多轮调用、状态共享、反馈循环），破坏了并行生成的轻量性和可扩展性。

本论文旨在解决这一矛盾：**如何在不依赖已观测种子输出的前提下，实现高质量且高成本效益的候选池多样化？**

---

### **提出了什么新方法或新思路**

论文提出两种 **anchorless**（无锚点）的 inference-time 控制策略，即无需依赖任何已生成样本即可引导多样性：

1. **Population-referential divergent instruction（群体参照型发散指令）**
   - 在提示中加入：“Try to make it stand out from other responses that might be generated for this same task.”
   - 引导模型基于其对“可能被生成的回答分布”的隐式认知，主动偏离高频区域。
   - 是一种极低成本的控制方式，仅需修改 prompt。

2. **Semantic direction stratification（语义方向分层生成）**
   - 先由模型进行一次 **planning call**，要求其为任务划分出 5 个互斥且广泛的语义方向（semantic strata）。
   - 随后将 150 个生成名额平均分配到这 5 个方向上，并在每个方向下独立生成。
   - 实现了对整个创意空间的系统性覆盖，而无需观察任何实际输出。

这两种方法都属于 **example-free** 和 **stateless** 的控制机制，保持了并行生成的简洁性与高效性。

---

### **相比现有方法的优势**

| 特性 | Anchorless 方法 | Anchored 方法（如 self/peer/repr） |
|------|------------------|-------------------------------|
| 是否需要种子输出 | ❌ 不需要 | ✅ 必须先生成 seed pool |
| 是否支持完全并行 | ✅ 支持 | ❌ 第二阶段需串行或受限并行 |
| 推理成本 | 低（尤其是 `indep-diverge`） | 高（两阶段，token 成本翻倍以上） |
| 多样性表现 | 强（尤其 `strat-diverge`） | 强，但在 token 效率上劣势明显 |
| 实用性 | 更适合部署场景 | 更复杂，难以规模化 |

> ✅ **核心优势**：  
> - `strat-diverge` 在 **多样性-质量-计算效率** 三者之间达到了最优平衡；
> - `indep-diverge` 是一个简单却非常有效的 **low-cost baseline**；
> - 所有 anchorless 方法都不依赖历史输出，更适合真实部署环境。

---

## 2. 核心实验方法和设置

### **使用的数据集与任务家族**

研究涵盖三个典型的创造性任务家族（creative task families），共 12 个 prompt 条件：

| 任务 | 描述 | 输出长度 |
|------|------|--------|
| **Stories**（故事） | 包括丛林冒险、跳伞失败、短篇恐怖故事、“一生与最后十秒”微小说 | 每篇 8 句话 |
| **Alternative Uses Task (AUT)** | 对日常物品（鞋、按钮、钥匙等）提出非传统用途 | 单句或短语 |
| **Slogans** | 为智能手机、汽水、献血活动创作广告口号 | ≤6 词 |

---

### **实验设置**

#### **模型**
- GPT-5.4
- Claude Sonnet 4.6
- Gemini 2.5 Pro

> 跨平台比较以验证结论的普适性。

#### **generation methods（生成方法）**

| 方法 | 类型 | 是否 anchorless | 简要说明 |
|------|-----|----------------|---------|
| `indep` | 单阶段 | ✅ | 独立生成，无干预 |
| `strat` | 单阶段 | ✅ | 分语义方向生成（5方向 × 30次） |
| `repr` | 两阶段 | ❌ | 观察3个代表性锚点再生成 |
| `self` | 两阶段 | ❌ | 自我对比：避开自己之前的输出 |
| `peer1`, `peer2` | 两阶段 | ❌ | 同伴对比：分别避开1人或2人的输出 |

#### **instruction strategies（指令策略）**

- **neutral**: “Make the response novel and appropriate”
- **diverge**: 在 neutral 基础上增加 “Try to make it stand out from other responses that might be generated…”

> 所有方法均交叉测试两种策略。

#### **每组配置生成数量**
- 每个 cell（组合）生成 150 个输出，构成一个 candidate pool。

---

### **评估指标**

#### **多样性指标（Diversity Metrics）**
- `Dpair`: 平均成对语义距离（mean pairwise semantic distance）
- `Dnn`: 最近邻距离 → 衡量局部冗余
- `Dmed`: 到中心点的距离 → 衡量离散程度
- `Dmst`: 最小生成树边长均值 → 衡量全局稀疏分布
- `Dent`: 归一化区域熵（normalized region entropy）→ 衡量跨聚类分布均匀性

> 所有嵌入使用 `sentence-transformers/all-mpnet-base-v2`

#### **质量代理指标（Quality Proxies）**
- **Stories**: MAoSS（自动化叙事创造力评分）
- **AUT**: CLAUS（替代用途新颖性评分）
- **Slogans**: 自定义 phrase-level non-template score（衡量模板复用度）

> 所有质量得分在 task 内标准化后取均值。

#### **其他分析手段**
- **Pool-size rarefaction**: 分析多样性随采样数增长的累积速度
- **Full-pipeline token accounting**: 统计完整推理路径的 token 消耗（含 planning call、seed generation、regeneration）
- **Bootstrap resampling**: 用于不确定性估计

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **表1：`indep-diverge` vs `indep-neutral` 的提升效果（平均 across 所有任务）**

| Model | ΔDpair | ΔQ₂ | R_tok（相对token成本） |
|-------|--------|-----|--------------------------|
| Claude Sonnet 4.6 | +0.06 [0.05, 0.07] | +0.38 | 1.1 |
| Gemini 2.5 Pro | +0.05 [0.04, 0.06] | +0.31 | 1.1 |
| GPT-5.4 | +0.04 [0.03, 0.05] | +0.25 | 1.1 |

> ✅ `indep-diverge` 显著提升多样性且保持甚至提高质量，token 成本几乎不变。

---

#### **最高多样性表现（以 Dpair 提升计）**

| 方法 | GPT-5.4 | Claude Sonnet 4.6 | Gemini 2.5 Pro |
|------|--------|--------------------|----------------|
| `strat-diverge` | +0.1717 | **+0.2526** | +0.1335 |
| `peer2-diverge` | +0.1706 | +0.2081 | +0.1278 |

> 🔥 `strat-diverge` 在多数情况下达到或超过最强 anchored 方法的表现。

---

#### **Token-normalized 效率（每 100k pipeline tokens 的 Dpair 提升）**

| 方法 | GPT-5.4 | Claude Sonnet 4.6 | Gemini 2.5 Pro |
|------|--------|--------------------|----------------|
| `strat-diverge` | **0.295** | **0.379** | **0.207** |
| `self-diverge` | 0.157 | — | — |
| `peer2-diverge` | 0.046 | — | — |

> 💡 尽管 `peer2-diverge` 总体多样性高，但因其 token 成本高达 **3.71 倍**，单位成本效率远低于 `strat-diverge`（仅 1.61 倍）。

---

### **与基线方法的对比结果**

| 对比维度 | 结果 |
|---------|------|
| `indep-diverge` vs `indep-neutral` | 明显更优，是最佳低成本 baseline |
| `strat-diverge` vs anchored methods | 多样性相当甚至更强，且 token 效率显著更高 |
| `repr`（代表锚点） | 表现弱于 peer/self 方法，尤其在 Dent 上增益较小 |
| `self/peer` 方法 | 在拥有 seed 输出时强大，但优势在计入全流程 token 后大幅缩水 |

---

### **消融实验与补充分析**

#### **Rarefaction Analysis（稀疏化分析）**
- `strat-diverge` 在 `Dpair` 和 `Dent` 的 AUC 上均优于 `repr-diverge` 和 `indep-diverge`
- `peer1/peer2` 方法在 entropy 积累速度上略快（first-hit 更小），但前提是已有 seed pool
- `strat-diverge` 达到 indep-neutral entropy 目标仅需 **7.4 个样本**，远快于 `repr-diverge`（23.6）

#### **Robustness Checks**
- 替换 slogan 质量评分为 LLM-as-a-judge 得分：主趋势不变
- 移除 slogan 任务重新分析：结论依然成立
- 多种 embedding 模型（BGE, E5）下结果稳健

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Population-referential instruction 是一个强有力且低成本的 baseline**  
   - 仅通过一句“与其他可能生成的回答不同”，就能显著提升多样性而不牺牲质量。
   - 几乎零成本，适用于所有部署场景。

2. ✅ **Semantic direction stratification 是最强的 anchorless 方法**  
   - 一次 planning call 即可构建语义地图，指导后续生成覆盖广泛方向。
   - 在最终池多样性、质量、token 效率三方面综合表现最佳。

3. ✅ **Anchorless 方法可以媲美甚至超越 anchored 方法**  
   - `strat-diverge` 的多样性与 `peer2-diverge` 相当，但推理成本仅为后者的 ~43%（GPT-5.4）。
   - 当考虑 full-pipeline token accounting 时，anchored 方法的优势被严重削弱。

4. ✅ **Stratification 与 divergence 是互补的**  
   - 两者结合（`strat-diverge`）取得最优前沿（best frontier）。
   - 单独使用任一也有效，但联合更强。

5. ✅ **Autostratification 视角具有启发意义**  
   - LLM 不只是采样器，还能作为“空间规划者”提供高层语义结构。
   - 利用模型自身对任务变体的理解，可实现更智能的探索。

---

### **方法的局限性**

1. 🛑 **依赖自动质量代理（automatic quality proxies）**
   - MAoSS、CLAUS 等虽经验证，但仍无法替代人类专家评估。
   - 尤其对于 slogan，模板复用未必等于低质量。

2. 🛑 **多样性基于 embedding space**
   - 受限于 embedding 模型的能力，可能忽略某些人类感知的重要差异。

3. 🛑 **任务范围有限**
   - 仅覆盖三种任务类型和三个主流模型，不能保证泛化至所有领域（如科学假设生成、代码设计等）。

4. 🛑 **固定参数设定**
   - 固定为 5 个方向、均等分配、150 输出数，未探索 adaptive allocation 或动态调整。

---

### **未来工作方向**

- 探索 **adaptive stratification**：根据方向密度动态分配生成资源
- 设计 **hybrid pipelines**：结合 stratification 与 limited regeneration
- 引入 **human-in-the-loop evaluation**：验证多样化是否真正提升下游选择价值
- 扩展至 **multimodal** 或 **long-form generation** 场景
- 研究 **cross-model stratification**：利用一个模型规划，另一个模型执行生成

---

> 🔚 **总结一句话**：  
> 本文证明了无需依赖已生成样本（anchorless），也能高效实现 LLM 创意生成的多样化；其中 **semantic direction stratification + population-referential instruction** 构成了当前最实用、最具性价比的 inference-time 控制范式。

</details>

---

### 8. [HTAM: Hierarchical Transition-Attended Memory for Operator Optimization](https://arxiv.org/abs/2605.29734)

**Authors**: Yining Zhang, Mingyang Yi, Chen Wang, Xuwen Xiang, Tianhe Jia, Zedong Dan, Chengqing Zong, Yue Wang  
**Category**: cs.CL  
**Published**: 2026-05-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.29734v1  

#### Abstract
High-performance GPU kernels are essential for efficient LLM deployment, yet optimizing them remains expertise-intensive. Recent LLM-based code generation makes automatic GPU operator generation promising, but operator optimization remains a hardware-aware search problem. Existing LLM-based methods ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HTAM: Hierarchical Transition-Attended Memory for Operator Optimization

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在大型语言模型（LLM）驱动的高性能 GPU 算子优化中，存在一个**粒度不匹配（granularity mismatch）**的问题：

- **粗粒度提示（Coarse hints）**（如“优化内存访问”）易于检索，但过于抽象，难以直接生成可执行的 CUDA 代码。
- **细粒度轨迹（Fine-grained traces）**（如完整的实现历史）虽然具体，但搜索空间大、噪声多，导致检索效率低且难以泛化。

因此，如何组织优化经验，在“高层方向指导”与“底层代码生成”之间建立有效桥梁，是当前 LLM-based 算子优化的关键挑战。

---

### **提出了什么新方法或新思路**

论文提出 **HTAM**（Hierarchical Transition-Attended Memory），一种**分层过渡感知记忆框架**，用于 LLM-based 算子优化。

其核心思想是模仿人类专家的优化流程，采用**从粗到细（coarse-to-fine）**的决策方式：

1. **先选择全局优化方向**（Global Direction）：如 `Memory Access`、`Data Reuse`、`Parallel Mapping` 等。
2. **再检索局部策略**（Local Strategy）：在选定方向下，获取具体的代码级优化策略，如 `aligned vectorized loads` 或 `read-only-load promotion`。
3. **建模状态转移关系**（Transition Experience）：通过边记忆（edge memory）记录不同全局方向之间的有效转移路径（如“先优化数据复用，再优化内存访问”），从而实现基于历史的决策引导。

为此，HTAM 构建了一个 **Hierarchical Transition Graph (HTG)**，包含三种结构化记忆：

| 组件 | 内容 |
|------|------|
| **Global Nodes** | 高层级优化意图（如内存访问、边界处理） |
| **Local Nodes** | 具体的代码级优化策略（如向量化加载） |
| **Transition Edges** | 全局方向间的转移经验（何时、为何、风险） |

---

### **相比现有方法的优势**

| 方法 | 局限性 | HTAM 的优势 |
|------|--------|-------------|
| **CudaForge** | 使用粗粒度瓶颈诊断，难以执行 | 引入局部策略，提供可执行代码指导 |
| **Robust-KBench** | 依赖细粒度实现轨迹，搜索空间大 | 分层结构缩小搜索空间，提升效率 |
| **Flat Memory Retrieval** | 扁平化记忆，缺乏结构 | HTG 结构化组织，支持路径感知决策 |
| **单步生成（Vanilla LLM）** | 缺乏迭代反馈与记忆 | 支持多步演化与动态记忆更新 |

**HTAM 的核心优势在于**：
- 将优化过程分解为 **global selection + local refinement**，降低复杂性。
- 利用 **transition-aware scoring**，结合近期历史进行更智能的方向选择。
- 通过 **reusable edge memory** 替代完整路径记忆，显著降低存储与探索成本。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **KernelBench**：包含 250 个任务的完整套件
  - **Level-1 (L1)**：100 个单算子任务（如 Swish、LayerNorm）
  - **Level-2 (L2)**：100 个算子融合任务（如 Conv+ReLU+BN）
  - **Level-3 (L3)**：50 个模型级任务（如 ResNet、MLP）
- **Robust-KBench**：用于跨基准迁移测试，包含 5 个代表性任务（如 LayerNorm、Cross Entropy）

### **实验设置**

- **硬件环境**：NVIDIA A100-SXM4-80GB GPU，CUDA 12.8，PyTorch 2.9.1
- **LLM Backbone**：主实验使用 **DeepSeek-R1**，部分对比使用 DeepSeek-V3/V4-Flash、GPT-4o、Gemini 等
- **优化步数**：默认 `T=6` 步演化
- **记忆初始化**：使用预定义的 HTG schema，包含静态先验与可写运行时字段

### **评估指标**

| 指标 | 定义 |
|------|------|
| **Correct** | 生成代码编译成功且数值正确 |
| **Fast@1** | 最佳候选既正确又快于 PyTorch eager 参考实现的比例 |
| **GeoM Spd.** | 几何平均加速比（vs. PyTorch eager） |
| **L1/L2/L3 Spd.** | 各层级任务的几何平均加速比 |
| **Valid@p** | 达到至少 `p×` 加速且正确的任务比例 |

### **基线方法对比**

分为三类：

#### （1）**Vanilla LLM Generation**
- 单次生成，无反馈、无记忆
- 包括：DeepSeek-R1, V3, V4-Flash, Gemini, OpenAI-o3

#### （2）**Vanilla Evolution-based Methods**
- 进化式优化，但无结构化记忆
- 包括：
  - **Best-of-N Sampling**：采样多个并选最优
  - **Feedback-only Refinement**：仅用反馈迭代修正
  - **Flat Memory Retrieval**：扁平记忆检索

#### （3）**Refined Evolution-based Methods**
- 更先进的系统（外部引用）
  - **KernelBlaster**：持续跨任务优化
  - **CudaForge**：基于硬件反馈的代理框架

---

## 3. 主要实验结果和性能指标

### **关键性能数据（KernelBench 全集）**

| 方法 | Correct | Fast@1 | GeoM Spd. | L1 | L2 | L3 |
|------|--------|--------|-----------|-----|-----|-----|
| **HTAM (Ours)** | **98.4%** | **84.0%** | **1.978×** | 1.532× | 2.598× | 1.909× |
| Flat Memory Retrieval | 89.6% | 58.8% | 1.464× | 1.194× | 1.917× | 1.284× |
| Feedback-only Refinement | 84.8% | 49.2% | 1.065× | — | — | — |
| CudaForge (reported) | 97.6% | 70.8% | 1.677× | — | — | — |
| KernelBlaster (reported) | 80.2% | 62.8% | 1.756× | — | — | — |

> ✅ **HTAM 在所有控制变量基线中表现最佳**：
> - Correctness 提升 **8.8 pp**（vs. Flat Memory）
> - Fast@1 提升 **25.2 pp**
> - 加速比从 1.464× 提升至 **1.978×**

---

### **消融实验结果（Ablation Study）**

在 250 任务 KernelBench 上进行消融，验证各组件贡献：

| 变体 | Correct | GeoM Spd. | Δ vs. Full |
|------|--------|-----------|------------|
| **Full HTAM** | 98.4% | 1.978× | 0 |
| w/o hierarchy | 89.6% | 1.464× | -0.514× |
| w/o local memory | 71.6% | 0.974× | -1.004× |
| randomized memory | 88.8% | 0.812× | -1.166× |
| w/o prefix | 94.4% | 1.683× | -0.295× |
| w/o position-aware weight | 95.6% | 1.787× | -0.191× |
| w/o updated memory | 96.4% | 1.842× | -0.136× |
| 1-step evolution | 91.2% | 1.428× | -0.550× |
| 3-step evolution | 95.2% | 1.756× | -0.222× |

> 🔍 **关键发现**：
> - **分层结构（Hierarchy）最关键**：移除后性能下降最大（-1.004×）
> - **局部记忆（Local Memory）至关重要**：无此则几乎无法生成有效代码
> - **前缀感知（Prefix-aware）提升显著**：说明历史路径对决策重要
> - **多步演化必要**：1 步演化性能远低于 6 步

---

### **跨模型与跨基准迁移实验**

#### （1）**LLM 后端泛化性**（50-task 子集）

| LLM Backend | Correct | Fast@1 | GeoM Spd. |
|-------------|--------|--------|-----------|
| **DeepSeek-R1** | 98.0% | 84.0% | **2.003×** |
| **GPT-4o** | 96.0% | 86.0% | **2.037×** |
| DeepSeek-V4F | 88.0% | 58.0% | 1.472× |

> ✅ HTAM 的结构化记忆可在不同 LLM 后端间迁移，只要 LLM 具备基本代码能力即可受益。

#### （2）**跨基准迁移**（Robust-KBench）

| 任务 | Cold Start | KBMem Init | Δ Spd. |
|------|------------|------------|--------|
| LayerNorm | 2.15× | 3.91× | **+1.76×** |
| RMSNorm | 2.05× | 3.64× | **+1.59×** |
| Linear | 0.92× | 1.09× | +0.17× |
| **GeoM Spd.** | **1.20×** | **1.58×** | **+0.38×** |

> ✅ KernelBench 上积累的记忆可有效迁移到 Robust-KBench，尤其在归一化类算子上增益显著。
> ⚠️ 但也存在负迁移（如 Cross Entropy），表明迁移效果依赖于算子结构相似性。

---

## 4. 关键结论和发现

### **主要发现**

1. **结构化记忆优于扁平记忆**：HTAM 证明，将优化经验组织为 **global-local-transition** 三层结构，能显著提升 LLM 算子优化的可靠性与效率。
2. **历史路径影响决策质量**：引入 **transition-aware scoring** 和 **position-aware weighting** 能有效利用近期优化历史，避免无效重复尝试。
3. **多步演化 + 动态记忆更新是关键**：单步生成或固定记忆无法达到 HTAM 的性能，必须支持反馈闭环与记忆演进。
4. **方法具有泛化潜力**：HTAM 的记忆可在不同 LLM 后端和不同基准间迁移，尤其适用于结构相似的任务。

---

### **局限性**

1. **计算与 API 成本限制**：实验受限于预算，未全面探索所有 LLM、解码策略或硬件平台。
2. **依赖可执行验证而非形式化验证**：仅在给定输入上验证正确性，不能保证所有形状/分布下的正确性。
3. **记忆 schema 需适配新硬件/算子**：扩展到新架构（如 NPU）或全新算子家族可能需要调整 HTG schema。
4. **冷启动开销**：尽管在线成本可控（约 \$0.21/任务），但初始图构建与记忆热启仍需一次性投入。

---

### **未来工作方向**

- 扩展 HTAM 至更多硬件平台（如 TPU、NPU）和编程模型（如 SYCL、HIP）。
- 探索更高效的 memory warm-up 机制，减少冷启动成本。
- 引入形式化验证模块，提升生成代码的可信度。
- 将 HTAM 与 AutoTVM、Ansor 等传统自动调优系统结合，形成混合优化框架。
- 研究更通用的 transition memory 表示，支持跨领域知识迁移。

---

> **总结**：HTAM 通过**分层过渡感知记忆**，成功解决了 LLM-based 算子优化中的“粒度不匹配”难题，实现了从“抽象建议”到“可执行代码”的可靠转化，为构建**可持续学习的 AI 编译器代理**提供了新范式。

</details>

---

### 9. [CCS: Clinical Consensus Selection for Radiology Report Generation](https://arxiv.org/abs/2605.30131)

**Authors**: Xi Zhang, Yingshu Li, Zaiqiao Meng, Jake Lever, Edmond S. L. Ho  
**Category**: cs.CL  
**Published**: 2026-05-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.30131v1  

#### Abstract
Radiology report generation (RRG) is commonly formulated as a single-path generation task, where a multimodal large language model (MLLM) produces one decoded report as the final output. While recent progress has largely been driven by scaling training data, model capacity, and retrieval mechanisms,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# CCS: Clinical Consensus Selection for Radiology Report Generation 论文总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Radiology Report Generation (RRG)** 模型通常采用单路径生成（single-path generation）策略，即通过自回归方式逐词生成一份报告作为最终输出。尽管近年来在模型容量、训练数据和检索增强等方面取得了进展，但在 **inference time** 阶段的质量优化仍被忽视。

作者观察到：一个固定的多模态大语言模型（MLLM）在其候选池（candidate pool）中往往能生成比默认解码输出更符合临床事实的报告，说明推理阶段的决策机制是当前性能瓶颈之一。因此，如何在不修改模型参数的前提下，在推理时从多个候选报告中选出最可靠的报告，成为一个关键挑战。

### 提出了什么新方法或新思路
本文提出 **Clinical Consensus Selection (CCS)** ——一种**解码器无关**（decoder-agnostic）、**无需参考报告**（reference-free）的推理时选择框架，其核心思想如下：

1. **Rollout Pool Generation**：对同一张影像输入，使用随机采样生成 $N$ 个候选报告，构成一个 rollout pool。
2. **Pairwise Utility Scoring**：计算每对候选报告之间的“一致性”得分（utility），分为两类：
   - **Textual Utility**：基于传统文本相似度指标（如 ROUGE-L, BERTScore 等）。
   - **Image-Grounded Utility**：使用在图像-报告对上微调过的多模态嵌入模型 **Qwen3-VL-Embed**，衡量两个报告在医学语义空间中的相似性，而不仅仅是表面文本重叠。
3. **Consensus Aggregation**：为每个候选报告计算其与其他所有候选报告的平均 utility 得分，选择得分最高的报告作为最终输出。

该方法将 RRG 任务重新定义为“从候选池中选择最优报告”，而非“生成唯一报告”。

### 相比现有方法的优势
- **无需训练或修改原生成模型**：CCS 是纯推理时（inference-time）的方法，可直接应用于任何预训练好的 MLLM。
- **超越通用 Best-of-N 方法**：相比仅依赖 perplexity 或文本相似性的通用选择策略，CCS 利用图像-报告联合表示空间，更能捕捉**临床一致性**（clinical consensus），尤其在识别异常发现方面表现更好。
- **缓解“沉默偏差”（silence bias）**：传统基于标签一致性的方法倾向于选择保守、缺少阳性发现的报告；而 CCS 的 image-grounded utility 能更好地保留重要病理描述。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **MIMIC-CXR**：主训练与测试集，包含约 37 万张胸部 X 光片及其自由文本报告。
- **IU-Xray**：用于跨数据集泛化能力评估。
- **CheXpert Plus**：另一个大型公开数据集，用于进一步验证泛化性。

所有模型均只在 MIMIC-CXR 训练集上进行训练，其余数据集仅用于零样本迁移评估。

### 实验设置和评估指标

#### 评估指标
| 类型 | 指标 | 描述 |
|------|------|------|
| **Lexical Metrics** | ROUGE-L, BLEU-4, BERTScore | 衡量生成报告与真实报告之间的文本相似性 |
| **Radiology-specific Metrics** | RadGraph-F1, RaTEScore, RadEval-BERT, CheXbert-F1 | 评估临床正确性，涵盖实体关系、关键诊断概念、语义一致性等 |

特别地，**CheXbert-F1** 分为 5-class 和 14-class 设置，分别关注常见病灶的存在与否。

#### 推理设置
- Rollout pool 大小 $N=8$
- 采样温度 $T=0.5$
- 所有方法在同一候选池上比较，确保公平性

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Single Path (Sampling/Greedy)** | 默认单路径生成，作为基础对照 |
| **Random** | 从候选池中随机选择一份报告 |
| **Perplexity** | 选择平均困惑度最低的报告 |
| **Self-Certainty** | 选择负对数似然最小的报告 |
| **ModeX** | 构建文本相似图并选择聚类中心 |

此外还对比了多种基于不同 textual utility 的 CCS 变体，并以 **CCS + Qwen3-VL-Embed** 为主推方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（MIMIC-CXR 测试集）

| 方法 | ROUGE-L | BLEU-4 | BERTScore | RadGraph-F1 | RaTEScore | CheXbert-F1 |
|------|---------|-------|-----------|-------------|-----------|--------------|
| Sampling | 0.2252 | 0.0534 | 0.5128 | 0.1989 | 0.5165 | 0.4519 |
| Perplexity | 0.2368 | 0.0694 | 0.5368 | 0.2125 | 0.5295 | 0.4605 |
| ModeX | 0.2388 | 0.0595 | 0.5268 | 0.2124 | 0.5291 | 0.4496 |
| **CCS + Qwen3-VL-Embed** | **0.2331** | **0.0548** | **0.5268** | **0.2134** | **0.5323** | **0.4714** |

> ✅ 所有提升在 $p < 0.05$ 水平下统计显著，尤其在 **radiology-specific metrics** 上优势明显。

### 与基线方法的对比结果
- **CCS 显著优于所有通用 Best-of-N 方法**，尤其是在 **RadGraph-F1** 和 **CheXbert-F1** 等临床指标上。
- 尽管在 BLEU 上略低于 Perplexity，但 CCS 在保持文本质量的同时实现了更强的临床一致性。
- **Self-Certainty 表现最差**，表明 token-level 置信度与临床正确性相关性弱。

### 消融实验结果
#### 不同 Consensus Utility 的比较（Table 3）
- 各 textual utility 在其对应 metric 上表现最佳（self-alignment），但整体临床一致性不如 image-grounded utility。
- **Qwen3-VL-Embed（fine-tuned）** 在多数指标上取得最优，且在未微调版本上也有提升，说明多模态嵌入本身已具备一定判别力。

#### Pool Size 影响（Figure 3）
- 随着 rollout 数量增加（$N=2 \to 16$），性能持续上升，但边际收益递减。
- 最终选择 $N=8$ 作为效率与效果的平衡点。

#### Cross-Backbone 一致性（Table 2）
- CCS 在三种不同 backbone 模型（LLaVA-Med, LLaVA-Rad, Libra）上均带来稳定提升，证明其**通用性强**。
- 即使在跨数据集（IU-Xray, CheXpert Plus）场景下也有效，显示良好的泛化能力。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **固定 MLLM 的候选池中存在比默认输出更优的报告**，说明推理时的选择机制是当前 RRG 性能瓶颈。
2. **CCS 能有效识别这些高质量候选报告**，通过聚合多模态语义空间下的临床共识，显著提升生成报告的临床可靠性。
3. **Image-grounded utility 提供了独立于文本共识的选择轴**，能够捕捉症状级的一致性，弥补纯文本方法的不足。
4. **临床标签一致性易受“沉默偏差”影响**，即倾向于选择保守、无阳性发现的报告；而 CCS 更好地保留了重要异常描述（见 Table 5 定性分析）。

### 方法的局限性
1. 实验基于标准 benchmark 数据集，可能无法完全反映真实临床环境中的多样性与噪声。
2. 缺乏放射科医生的人工评估，目前依赖自动指标，未来需结合专家评审。
3. 未探索更大规模的多模态 embedding 模型或 LLM-as-a-judge 评估方式。
4. 当前 image-grounded utility 并未直接利用测试图像，而是仅基于报告编码，仍有改进空间。

### 未来工作方向
- 引入更强大的多模态 embedding 模型（如更大规模的 VL-Embedder）。
- 结合 human-in-the-loop 或 expert evaluation 进行更严谨的验证。
- 将 CCS 与 retrieval-augmented generation 或 test-time training 方法结合，形成端到端增强流程。
- 探索动态调整 rollout pool size 或 adaptive sampling 策略以进一步提升效率。

> 📌 **总结一句话**：  
> **CCS 揭示了“生成能力”与“选择能力”的分离，并提供了一种简单、高效、无需训练即可提升 RRG 临床质量的新范式——让模型自己投票选出最好的报告。**

</details>

---

### 10. [NeuroEdge: Real-Time Hand Gesture Recognition with High-Density EMG Using Deep Learning at the Edge](https://arxiv.org/abs/2605.29326)

**Authors**: Peter Chudinov, Zhenyu Lin, Jay Motamarry, Srihita Panati, Xiaorong Zhang, Zhuwei Qin  
**Category**: cs.LG  
**Published**: 2026-05-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.29326v1  

#### Abstract
High-density electromyography (HD-EMG) has emerged as a powerful modality for decoding fine-grained neuromuscular activity, enabling real-time neural-machine interfaces (NMIs) for applications such as prosthetic control, rehabilitation, and augmented interaction. While deep learning approaches such ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# NeuroEdge: Real-Time Hand Gesture Recognition with High-Density EMG Using Deep Learning at the Edge  
**论文核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **高密度 EMG（HD-EMG）与深度学习模型在资源受限嵌入式设备上的实时集成难题**。尽管 HD-EMG 和 CNN 在手势识别中表现优异，但其计算开销大，难以部署于低功耗、低成本的微控制器（MCU）上。
- 现有系统多依赖 GPU、TPU 或 FPGA 平台（如 Jetson Nano、Coral TPU），导致功耗高、便携性差，不适合可穿戴或植入式神经接口应用。

### 🚀 提出的新方法与创新思路
- **NeuroEdge 系统架构**：首个将全规模 HD-EMG 数据采集与基于深度学习的实时手势识别完全集成到微控制器边缘设备中的端到端系统。
  - **HD-EMG StreamBridge**：自定义无线通信接口，实现 Quattrocento HD-EMG 放大器与 ESP32 微控制器之间的高效 Wi-Fi 数据流传输。
  - **EdgeDL Inference Engine**：轻量级深度学习推理引擎，运行于 Sony Spresense MCU 上，支持 TensorFlow Lite Micro 的紧凑型 1D CNN 模型。
- **软硬件协同优化设计**：
  - 利用 **DMA（Direct Memory Access）** 实现非阻塞数据缓冲；
  - 采用 **SPI Burst Communication** 在 ESP32 与 Spresense 之间高速传输数据；
  - 构建 **流水线架构**，实现数据采集、传输与推理并行执行，显著降低延迟。

### 🔍 相比现有方法的优势
| 对比维度 | Prior Works ([4][5][6]) | NeuroEdge（本文） |
|--------|--------------------------|------------------|
| 硬件平台 | GPU/TPU/FPGA（高功耗） | MCU-only（低功耗、低成本） |
| 通道数 | 最高达 64–80 通道 | 支持 **192 通道 HD-EMG** |
| 部署方式 | 外接计算单元或云端 | 完全本地化、无外部依赖 |
| 实时性 | 存在通信瓶颈 | 总平均延迟仅 **83ms** |
| 可扩展性 | 固定配置 | 模块化设计，支持灵活扩展 |

> ✅ **核心优势**：首次实现了 **full-scale HD-EMG + deep learning on microcontroller-based edge devices**，为下一代可穿戴 NMI 提供可行路径。

---

## 2. 核心实验方法和设置

### 📊 数据集与数据采集协议
- **受试者**：1 名健康男性（IRB 批准，SFSU）
- **电极配置**：
  - 使用三个 8×8 表面电极阵列（GR10MM808, OT Bioelettronica），共 **192 通道**
  - 贴附于优势前臂近肘部区域，覆盖手部、手掌及手指相关肌群
- **采样参数**：
  - 采样率：**512 Hz**
  - 滤波器：0.3 Hz 高通 + 500 Hz 低通
- **手势类别（7类）**：
  1. No Movement  
  2. Wrist Supination  
  3. Wrist Pronation  
  4. Hand Close  
  5. Hand Open  
  6. Wrist Flexion  
  7. Wrist Extension  

- **训练数据收集**：
  - 每个手势持续 7 秒，重复 8 次，含休息间隔
  - 使用 Python GUI 进行视觉提示，数据通过 MacBook Air 实时记录为 CSV 文件

### ⚙️ 实验设置
- **离线训练阶段**：
  - 模型框架：PyTorch → ONNX → TensorFlow Lite → C header file
  - 使用 **M5 语音识别模型改编的 1D CNN**
  - 训练平台：NVIDIA RTX 3080 GPU
- **实时测试阶段**：
  - 同一受试者在同一电极位置下复现相同手势序列
  - 数据经由 **HD-EMG StreamBridge-C** 流向 ESP32
  - 通过 SPI 传至 **Spresense** 执行推理
  - 分类结果通过串口发送至 PC 显示，并同步视频录像用于后期对齐分析

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Classification Accuracy** | 整体分类准确率（steady-state + transition） |
| **Latency** | 推理延迟（inference latency）、SPI 通信延迟 |
| **Memory Footprint** | 模型大小与内存占用 |
| **End-to-End Throughput** | 数据采集→传输→推理全流程吞吐能力 |

### 🔀 基线方法对比（间接比较）
虽然未直接进行横向 benchmark，但从文献对比可见：
- [4] Tam et al.：使用 32 通道 + CNN + Jetson Nano（GPU）
- [5] Buteau et al.：64 通道 + Coral TPU（仍高于 MCU 功耗）
- [7] Lu et al.：在 Spresense 上部署 CNN，但仅使用 **8 通道 EMG**

> ➡️ NeuroEdge 在 **通道密度、硬件能效比、系统集成度** 上全面超越已有工作。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
| 指标 | 数值 | 说明 |
|------|------|------|
| **离线验证准确率** | **95.33%** | 经过 25 个 epoch 训练后，在七类手势上的交叉验证精度 |
| **实时分类准确率** | **90.00%** | 包括稳态与过渡阶段的整体表现 |
| **平均推理延迟** | **70 ms** | Spresense 上单次推理耗时 |
| **SPI 通信延迟** | **13 ms** | ESP32 到 Spresense 的 burst 传输时间 |
| **总平均延迟** | **83 ms** | 满足多数实时交互需求（<100ms） |
| **模型大小** | **~15.24 KB** | 量化为 8-bit 的 TFLite 模型编译为 C 头文件 |
| **输入窗口大小** | 20 帧 × 192 通道 = 3840 维向量 | 时间滑窗策略 |

### 📉 错误分析与局限
- **主要误分类发生在手势转换期（transition phases）**
  - 原因：训练数据仅包含静态手势样本，缺乏动态过渡标注
  - 改进方向：引入 transition-aware labeling 与训练策略
- **未使用后处理技术（如 majority voting）**
  - 虽可提升准确率，但会增加延迟，牺牲实时性

### ❌ 消融实验（Ablation Study）
> 文中未明确列出消融实验表格，但从系统设计可推断以下关键组件作用：
- **DMA 缓冲机制**：确保 ESP32 不丢失数据包，维持稳定吞吐
- **SPI Burst + Handshake 协议**：减少通信等待时间，提高带宽利用率
- **1D CNN 结构简化（无全连接层）**：降低内存占用，适配 MCU 资源限制
- **8-bit Quantization**：压缩模型体积，加速整数运算

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **HD-EMG + Deep Learning 可成功部署于 MCU 边缘设备**  
   NeuroEdge 验证了在 **Sony Spresense** 这类资源受限平台上运行复杂 CNN 模型的可行性。

2. **端到端延迟控制在 83ms 内，满足实时性要求**  
   支持用于假肢控制、康复机器人等需要快速响应的应用场景。

3. **模块化架构具备良好可扩展性**  
   StreamBridge 与 EdgeDL 引擎解耦设计，便于移植至其他传感器或设备。

4. **高通道数（192）带来更丰富的空间特征表达能力**  
   相比传统稀疏 EMG，HD-EMG 更有利于精细动作解码。

### ⚠️ 局限性
- **单受试者实验**：尚未验证跨用户泛化能力
- **静态手势为主**：缺乏对手势过渡过程的建模
- **未考虑长期漂移问题**：如电极位移、皮肤阻抗变化等实际干扰因素
- **缺少多模态融合**：当前仅依赖 EMG，未结合 IMU、EEG 等其他信号

### 🔮 未来工作方向
1. **Multi-subject Validation**：扩大受试者群体，增强模型鲁棒性
2. **Incorporate Transition Data**：将动态过渡阶段纳入训练集，提升连续手势识别能力
3. **On-device Adaptive Learning**：探索 TinyML 中的在线微调（on-device fine-tuning）
4. **Continuous Gesture Decoding**：从离散分类转向连续轨迹预测
5. **Multi-modal Sensor Fusion**：整合 IMU、force sensors 或 EEG 提升上下文感知能力
6. **Energy Optimization**：进一步优化功耗，延长电池寿命，推动商业化落地

---

## ✅ 总结一句话
> **NeuroEdge 是首个实现 full-scale HD-EMG 与 deep learning 在 microcontroller 上实时协同工作的边缘神经接口系统，以 90% 准确率和 83ms 延迟展示了可穿戴 NMI 的工程可行性，为未来低功耗、高精度人机交互开辟了新路径。**

</details>

---

### 11. [Bastion: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting](https://arxiv.org/abs/2605.29727)

**Authors**: Soowon Oh, Nam Cao, Yujin Kim, Hojung Jung, Huzama Ahmad, Sangmin Bae, Se-Young Yun  
**Category**: cs.LG  
**Published**: 2026-05-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.29727v1  

#### Abstract
Block-diffusion drafters have recently emerged as a powerful alternative for speculative decoding by predicting multiple future-token distributions in a single parallel step. However, since these parallel predictions are sampled from position-wise marginals rather than fully conditioned sequences, c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BASTION: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现有的 **speculative decoding** 方法在加速大语言模型（LLM）推理时面临两个核心瓶颈：

1. **串行依赖限制效率**：传统基于 autoregressive（AR）的 drafters 需要逐个生成候选 token，导致 drafting 阶段本身成为计算瓶颈，难以充分利用并行硬件。
2. **单路径草案质量差**：虽然 block-diffusion drafters 可以并行预测多个未来 token 分布，但若仅采用 greedy top-1 路径作为草案，则可能产生与目标模型偏好轨迹不一致的低质量序列，影响接受率。

此外，已有树形 speculative decoding 方法多使用**静态树结构**（如固定宽度或深度），无法适应不同输入、上下文长度和硬件条件下的最优预算分配。

---

### 提出了什么新方法或新思路

本文提出 **BASTION**，一种**预算感知的、基于树结构的扩散草案框架**，其核心创新如下：

#### ✅ 创新点一：Tree-based Block-Diffusion Drafting
- 不再从 block-diffusion 输出中提取单一 greedy 路径，而是利用每个位置的 top-K logits 构建一个**前缀闭合的候选树（prefix tree）**。
- 所有节点共享相同的上下文，并通过路径置信度 $ p(i) = \prod_{k=1}^{d(i)} q_k(x_{ik}) $ 进行打分。
- 允许目标模型在一个 batch 中验证多条潜在路径，提升 verifier 利用率和平均接受长度（AAL）。

#### ✅ 创新点二：Best-First Tree Expansion（最优扩展策略）
- 提出一种**贪心最优的 best-first 扩展算法**，每次选择当前未加入树中且路径得分最高的合法节点进行扩展。
- 在理论证明下，该方法能构造出给定大小 $ N $ 下最大化期望接受长度 $ \hat{A}(T) $ 的树结构（Proposition 3.3）。

#### ✅ 创新点三：Budget-Aware Online Controller（动态预算控制器）
- 引入一个**在线延迟估计器**，结合硬件感知的 roofline 模型，实时预测验证开销 $ T_{\text{verify}}(N) $。
- 动态平衡“预期接受增益”与“验证成本”，当边际收益不再覆盖增量成本时停止树增长。
- 实现无需调参的自适应树规模控制，在每一步自动确定最优 $ N $。

---

### 相比现有方法的优势

| 对比维度 | BASTION | EAGLE-3 / Medusa 类 | DFlash / TiDAR 类 |
|--------|---------|---------------------|--------------------|
| 草案方式 | 树状结构（multi-path） | 树状结构（AR drafter） | 单路径 block-diffusion |
| 并行性 | ✅ 高（一次 drafting） | ❌ 低（需多次 drafter 推理） | ✅ 高（一次 drafting） |
| 草案质量 | ✅ 多路径探索 | ✅ 多路径探索 | ❌ 仅取 top-1 路径 |
| 树结构 | 🔁 动态构建（query-dependent） | 🧱 固定拓扑（static） | N/A |
| 硬件适配 | ✅ 自适应预算控制 | ❌ 固定预算 | ❌ 固定块大小 |
| 是否训练 | ✅ Training-free | ❌ 需训练辅助头或小模型 | ✅ Training-free |

> ✅ **综合优势**：BASTION 同时具备 block-diffusion 的高效 drafting 和 tree-based speculative decoding 的高接受率，并通过动态预算机制实现跨模型、跨硬件的鲁棒高性能。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

实验涵盖四大类任务共 **15 个基准**：

| 类别 | 数据集 |
|------|-------|
| 数学推理 | `GSM8K`, `MATH500`, `AIME25` |
| 代码生成 | `HumanEval`, `MBPP`, `LiveCodeBench` |
| 指令遵循 | `Alpaca`, `MT-Bench` |
| 长文本理解 | `LongBench`（7 个英文子集：`Qasper`, `GovReport`, `TriviaQA` 等） |

---

### 实验设置和评估指标

#### ✅ 目标模型（Target Models）
- `Qwen3-4B`
- `Qwen3-8B`
- `Llama-3.1-8B-Instruct`

#### ✅ 草案模型（Drafters）
- 使用与目标模型对齐的 **DFlash block-diffusion drafter**（无需训练，直接复用 checkpoint）

#### ✅ 硬件平台
- 四种 NVIDIA GPU：
  - `A100` (80GB HBM2e)
  - `H100` (80GB HBM3)
  - `RTX A6000` (48GB GDDR6)
  - `RTX PRO 6000 Blackwell` (96GB GDDR7)

#### ✅ 评估指标
| 指标 | 定义 |
|------|------|
| **Speedup** | 相对于标准 autoregressive decoding 的端到端加速比：<br>$ \text{Speedup} = \frac{\mathbb{E}[A(T)] \cdot T_{\text{AR}}}{T_{\text{draft}} + T_{\text{verify}} + T_{\text{aux}}} $ |
| **Average Accepted Length (AAL)** | 每轮迭代平均被接受的 token 数量，反映草案质量 |
| **Wall-clock latency** | 实际运行时间（毫秒级测量） |

#### ✅ 解码参数
- 温度：`T=0`（greedy）和 `T=1`（stochastic）
- 批次大小：`batch_size=1`

---

### 基线方法对比

| 基线方法 | 类型 | 描述 |
|--------|------|------|
| **Greedy Decoding** | 基准 | 标准 AR 解码，速度为 1× |
| **EAGLE-3** | Tree-based + AR drafter | 使用轻量级 AR drafter 构造固定大小为 60 的树 |
| **DFlash** | Block-diffusion + single path | 使用 block-diffusion 生成单条 top-1 路径（block size=16） |

> 所有方法均在同一 target model、dataset、temperature 下比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 模型 | 方法 | 平均 Speedup (T=0) | 最高 Speedup |
|------|------|-------------------|-------------|
| Qwen3-8B | BASTION | **6.61×** | 达 **7.69×**（AIME25） |
| Qwen3-4B | BASTION | **6.93×** | —— |
| Llama-3.1-8B | BASTION | **6.53×** | —— |

> 💡 **最高加速达 6.61×**，远超现有方法。

---

### 与基线方法的对比结果

#### ✅ 绝对性能对比（Table 1 & Figure 1）

| 方法 | 相比 Greedy | 相比 EAGLE-3 | 相比 DFlash |
|------|------------|--------------|-------------|
| **BASTION (Qwen3-8B)** | **6.61×** | ↑ **2.45× 更快** | ↑ **1.39× 更快** |
| **DFlash** | ~4.76× | —— | —— |
| **EAGLE-3** | ~2.70× | —— | —— |

> 在所有 8 个短上下文 benchmark 上，BASTION 均显著优于基线。

#### ✅ 跨硬件泛化能力（Figure 4）

- 在 `A100`, `H100`, `A6000`, `RTX PRO 6000` 上测试表明：
  - BASTION 在所有 (model, GPU) 组合上保持领先。
  - 加速效果不受特定 GPU 架构限制，具有强部署通用性。

#### ✅ 跨温度稳定性
- 在 `T=1`（stochastic decoding）下仍保持高加速比（平均 **6.66×**），说明方法对采样多样性也有效。

---

### 消融实验结果

#### 🔍 消融 1：Best-First vs Beam Search（图5）

| 方法 | Qwen3-4B Speedup | Qwen3-8B Speedup |
|------|------------------|------------------|
| Beam Search (w=4, d=15) | 6.16× | 5.74× |
| **Best-First (N=61)** | **6.59×** (+7.0%) | **6.09×** (+6.1%) |

> 结论：**全局最优的 best-first 扩展显著优于局部均匀的 beam search**，验证了路径得分排序的有效性。

#### 🔍 消融 2：固定预算 vs 自适应预算（图6）

- 固定预算方法（Fixed Budget）性能高度敏感于预设 $ N $：
  - $ N < 128 $：利用率不足
  - $ N > 512 $：验证延迟主导，速度下降
- **BASTION 动态控制器几乎达到“Oracle”上限**（即每个 setting 下最优固定 $ N $ 的表现），无需手动调参。

#### 🔍 消融 3：延迟模型校准（图7）

- 使用 **calibrated roofline model** 后，验证延迟预测误差 RMSE 下降 **87–90%**。
- 控制器变体对比：
  - `Static`（离线校准）：最稳定，默认配置
  - `EMA+Calib`：适合无先验校准场景，性能接近 Static
  - `EMA only`：不稳定，尤其在 Llama 上性能下降 12.7%

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Tree 结构 + Block-Diffusion 是高效 speculative decoding 的理想组合**：
   - 利用 block-diffusion 的并行 drafting 能力；
   - 通过 tree 扩展保留更多高质量候选路径，提高接受率。

2. ✅ **Best-first 扩展是理论上最优且实践中高效的树构建方式**：
   - 路径得分单调递减特性支持贪心构造；
   - 显著优于 beam search 等启发式方法。

3. ✅ **动态预算控制至关重要**：
   - 固定树大小无法适应多样化 workload 和硬件；
   - BASTION 的 online controller 可自动逼近理论最优，消除人工调参负担。

4. ✅ **方法具有极强泛化能力**：
   - 跨模型（Qwen / Llama）、跨 GPU（A100 → Blackwell）、跨任务（math/code/chat）均表现优异；
   - **training-free** 设计便于快速部署。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **Batch Size = 1** | 当前评估集中在单样本推理；大 batch 场景下验证成本变化，需重新设计预算策略 |
| **依赖离线校准** | 默认使用 per-(GPU, target) 的延迟曲线校准；若 runtime 动态漂移严重（如资源竞争），可能失效 |
| **KV Cache 管理复杂度增加** | 树形验证后需 reordering 和 selective cropping，带来轻微实现复杂性 |

> 作者指出可通过 EMA fallback 缓解部分问题，但仍为开放方向。

---

### 未来工作方向

1. **扩展至更大 batch size 场景**，研究批量下的树共享与并行验证机制；
2. **完全 online 的自适应延迟建模**，减少对离线 profiling 的依赖；
3. **将框架推广至其他 multi-token prediction 范式**，如 Mamba、Hybrid State-Space Models；
4. **探索更复杂的 draft-target alignment proxy**，进一步提升 surrogate 准确性。

---

> 🔗 **开源地址**：[https://github.com/kaist-ai-osi-lab/BASTION](https://github.com/kaist-ai-osi-lab/BASTION)  
> 📄 **论文链接**：[arXiv:2605.29727](https://arxiv.org/abs/2605.29727)

</details>

---

### 12. [DeepTool: Scaling Interleaved Deliberation in Tool-Integrated Reasoning via Process-Supervised Reinforcement Learning](https://arxiv.org/abs/2605.29568)

**Authors**: Yang He, Xiao Ding, Bibo Cai, Yufei Zhang, Kai Xiong, Zhouhao Sun, Bing Qin, Ting Liu  
**Category**: cs.AI  
**Published**: 2026-05-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.29568v1  

#### Abstract
Tool-Integrated Reasoning (TIR) extends LLM capabilities by leveraging external environments. However, existing methods lack the deliberation during sequential tool invocation required for strategic planning and self-correction. While RL mitigates this, conventional approaches for Tool-Integrated Re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DeepTool: Scaling Interleaved Deliberation in Tool-Integrated Reasoning via Process-Supervised Reinforcement Learning》核心总结

---

## 1. 主要贡献和创新点

### 解决的问题
当前的 **Tool-Integrated Reasoning (TIR)** 方法在多步工具调用过程中缺乏有效的“深思熟虑”（deliberation）机制，导致以下问题：
- **推理链脆弱**：模型在执行工具时容易因初始错误引发级联失败（cascading failures）。
- **监督信号稀疏**：传统强化学习（RL）依赖最终结果奖励（outcome-based rewards），无法对中间推理步骤和工具调用进行有效监督。
- **浅层思考（shallow thinking）**：多数框架采用固定、均匀的推理预算，难以在复杂步骤中投入足够的认知资源。

### 提出的新方法与思路
论文提出 **DeepTool** 框架，通过引入系统化的“交错式深思”（interleaved deliberation）来增强 TIR 能力，其核心创新包括：

#### （1）MOSAIC 合成流水线（Synthesis Pipeline）
- 将纯文本推理轨迹转化为带有交错思考、行动与观察的多轮交互轨迹。
- 引入**对抗性扰动机制**（adversarial perturbations），模拟两类故障：
  - **Intrinsic Generative Anomaly**：语法错误等生成缺陷。
  - **Extrinsic Environmental Failure**：超时、API 错误等环境异常。
- 促使模型在训练阶段就学会**自我纠正**（self-correction）和**鲁棒规划**。

#### （2）基于 GRPO 的过程监督强化学习（Process-Supervised RL）
- 设计 **Action-Centric Process Reward**，提供每一步的密集监督信号：
  - 对 `thought` 部分保持灵活性（允许多样化推理路径）。
  - 对 `action`（如代码生成）施加严格对齐约束。
- 使用 **Group Relative Policy Optimization (GRPO)** 进行优化，提升训练稳定性。

#### （3）层级化策略架构（Hierarchical Policy）
- **Manager（导航员）**：高层规划器，负责制定战略意图（如子目标、纠错指令）。
- **Actor（执行者）**：底层执行器，依据意图生成具体推理与动作。
- 实现了“先想后做”的闭环控制。

### 相比现有方法的优势
| 维度 | 传统方法 | DeepTool |
|------|--------|---------|
| 推理模式 | 单次或线性推理（Chain-of-Thought） | 多轮交错式 System 2 推理 |
| 工具调用 | 原子操作，无内部反思 | 可中断、可修正的动态过程 |
| 训练监督 | 结果导向，稀疏奖励 | 步骤级密集过程监督 |
| 错误恢复 | 容易崩溃 | 内建自纠错能力 |

---

## 2. 核心实验方法和设置

### 数据集
- **数学推理基准**（共6个）：
  - **AIME24**, **AIME25**：美国数学邀请赛，高难度代数与组合题。
  - **MATH500**：标准数学评测集。
  - **AMC23**, **HMMT25**：竞赛类数学问题。
  - **OlympiadBench**：奥赛级别双语科学题。
  - **GPQA-Diamond**：研究生水平的抗谷歌问答挑战集。
- **训练数据来源**：
  - SFT 阶段：从 `OpenR1-Math-220k` 中采样 8k 实例。
  - RL 阶段：使用 `LIMO` 数据集。

### 实验设置
- **主干模型**：
  - `Qwen2.5-7B-Instruct`
  - `Qwen3-4B-Base`
- **三种配置对比**：
  1. `DeepTool (w/o thinking)`：直接工具调用，无显式推理。
  2. `DeepTool (SFT)`：带交错思考的监督微调。
  3. `DeepTool (+RL)`：进一步使用过程监督 RL 微调。
- **评估指标**：
  - 主要：**Average Accuracy @8**（减少采样方差）。
  - 辅助分析：token 成本效益（accuracy gain per 1k tokens）。

### 基线方法分类对比
| 类型 | 代表方法 |
|------|--------|
| 搜索增强 | Search-R1, Search-o1 |
| RL 增强工具推理 | ToRL, ReTool, ARPO |
| 零RL范式 | SimpleRL-Zoo, ZeroTIR |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 Qwen2.5-7B 为例）

| 方法 | AIME24 | AIME25 | MATH500 | Olympiad | AMC23 | HMMT25 | GPQA-D |
|------|-------|-------|--------|----------|--------|--------|--------|
| Base Model | 3.2 | 1.1 | 51.9 | 15.4 | 21.7 | 0.0 | 32.4 |
| DeepTool (w/o thinking) | 15.0 | 14.2 | 65.6 | 27.4 | 45.6 | 10.0 | 43.1 |
| DeepTool (SFT) | 38.3 | 25.8 | 70.4 | 38.5 | 68.1 | 20.4 | 44.8 |
| **DeepTool (+RL)** | **40.4** | **28.6** | **84.7** | **48.8** | **64.0** | **28.6** | **45.3** |

> ✅ 在所有基准上均取得最优表现，尤其在最具挑战性的 AIME 和 HMMT 上实现从近乎零到近 30% 的飞跃。

### 与其他先进方法对比
- 在 AIME24 上，DeepTool (+RL) 达到 **40.4%**，显著优于：
  - ToRL (30.0%)
  - ReTool (23.3%)
  - ZeroTIR (39.6%)
- 在 HMMT25 上，从 **0.0% → 28.6%**，是唯一能有效解决该难题的方法。

### 消融实验结果
#### （1）是否启用交错思考（Interleaved Thinking）
- 移除显式思考模块后性能大幅下降（如 AIME24 从 40.4 → 15.0），说明**结构化中间推理至关重要**。

#### （2）状态保留 vs 抛弃
- **保留中间思考状态**（state preserved）相比抛弃状态平均提升 **+9% ~ +16%**（相对增益）。
- 特别是在 AIME25 上从 25.8% 提升至 30.0%，验证了长期记忆对多步纠错的价值。

#### （3）思考预算扩展分析
- 性能随每步思考 token 预算增加而上升，呈现平滑增长趋势。
- 存在**收益递减点**：当预算过大时边际增益降低。
- 更难任务（如 Olympiad）需要更大预算才能饱和。

#### （4）MOSAIC 合成有效性
| 方法 | 平均轨迹准确率 | 平均交互轮次 |
|------|----------------|--------------|
| Baseline（普通提示合成） | 55.6% | 2.4 |
| **MOSAIC** | **60.8%** | **4.1** |
- 表明 MOSAIC 能生成更稳定、更深入、更具探索性的推理轨迹。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **交错式深思（interleaved deliberation）是提升 TIR 性能的关键**：
   - 显式分离“思考-行动-观察”循环，使模型具备动态调整与纠错能力。
2. ✅ **过程监督（process supervision）优于结果监督**：
   - Action-Centric Process Reward 提供密集反馈，解决了 RL 中的信用分配难题。
3. ✅ **对抗性扰动训练增强了鲁棒性**：
   - MOSAIC 中注入的错误迫使模型学习恢复策略，而非简单模仿成功路径。
4. ✅ **成本效益高**：
   - 尽管增加了推理开销，但合理设置思考预算可在**精度与 token 效率之间取得帕累托最优**（见 Figure 5）。

### 方法的局限性
- **依赖高质量参考轨迹**：MOSAIC 仍需初始的 `gold CoT` 来引导合成。
- **计算开销较高**：尤其是 RL 训练阶段需要大量 GPU 时间。
- **泛化边界待探索**：目前集中在数学领域，对开放域任务（如 Web 操作）的适用性尚不明确。

### 未来工作方向
- 扩展到更多模态与工具类型（如 vision, robotics）。
- 自动化思考预算分配（adaptive budgeting）。
- 构建无需人工标注 CoT 的自演化训练流程。
- 探索轻量化部署方案，降低推理延迟。

---

> 📌 **总结一句话**：  
> **DeepTool 通过将 System 2 级别的深思熟虑嵌入 TIR 流程，并辅以过程监督与对抗训练，实现了在复杂多步任务中前所未有的准确性与鲁棒性，为构建真正可靠的 AI Agent 提供了新范式。**

</details>

---

### 13. [From Context Shift to Stylistic Collapse: Why Training Objectives Matter More Than Scale](https://arxiv.org/abs/2605.28826)

**Authors**: Rohan Mahapatra  
**Category**: cs.CL  
**Published**: 2026-05-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.28826v1  

#### Abstract
In modern LLMs, linguistic features function not as stylistic artifacts but as probes of probability mass, allocated under training alignment objectives. Language models trained with contemporary pipelines exhibit severe reshaping of linguistic features, leading to extreme language re-distribution. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Context Shift to Stylistic Collapse: Why Training Objectives Matter More Than Scale*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统地研究了现代大语言模型（LLMs）在生成文本时出现的“**风格坍塌**”（stylistic collapse）现象，即AI生成文本呈现出高度公式化、重复性强、缺乏语言多样性的“AI写作风格”。以往研究多将此归因于 **RLHF**（Reinforcement Learning from Human Feedback）或指令微调（instruction tuning），而本文指出这一问题的根源更深层，并提出有效的缓解方案。

### 提出了什么新方法或新思路
1. **提出“上下文偏移”（context shift）与“吸收性风格状态”（absorbing stylistic states）的双重机制解释**：
   - **Context Shift**：训练语料涵盖多种文体（叙事、说明、对话等），但部署时模型被频繁用于正式说明性任务（如问答、分析），导致生成偏向此类文体特征（如标题、列表）。
   - **Absorbing States**：某些低熵的语言特征（如`#`标题、`1.`编号列表）会约束后续生成，形成自我强化的循环，加剧风格单一化。

2. **提出并验证“控制强度原则”（control strength principle）**：
   - 在预训练阶段引入**强熵正则化**（strong entropy regularization, λ=5.0），可显著缓解风格坍塌，且效果远超模型规模的影响。

3. **构建了一个可复现的24维语言学探针框架**：
   - 覆盖标点、话语标记、结构元素和语气标记四类语言特征，支持跨模型、跨架构的量化比较。

### 相比现有方法的优势
- **挑战主流认知**：证明风格坍塌并非由RLHF或指令微调引起（p > 0.25），而是源于更上游的生成动力学，因此无法通过后训练对齐修复。
- **以小搏大**：仅用 **410M 参数** 的 Pythia 模型，在 λ=5.0 正则化下，其风格自然度**超越所有前沿商业API**（如GPT-4o-mini、Claude、Gemini），性能提升达 **96.7–98.2%**。
- **揭示非线性效应**：弱正则化（λ=1.0）反而使问题恶化（发散度增加240%），强调干预必须“足够强”。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **训练语料基准**：
  - **The Pile**：用于 Pythia、Llama 等模型的基准频率构建。
  - **Dolma**：用于 OLMo 系列模型的基准构建。
  - 从两个语料库中各采样 **10万文档**（约3000万token）计算语言特征基线频率 $P_c(f)$。

- **模型集合**（共17个，主表展示13个）
  - 开源模型：Pythia (410M), OLMo (1B), Llama (3B/8B), Mistral (7B), Gemma (2B)
  - 商业API：GPT-4o-mini, Claude-3.5-haiku, Gemini-2.5-flash
  - 包含 base / instruct / RLHF 多种训练阶段版本，用于对比分析。

### 实验设置和评估指标
- **生成设置**：
  - 温度：0.7
  - 最大长度：1024 tokens
  - 种子固定（seed=42），确保可复现
  - 使用15个不同主题的提示词，每模型生成1000条输出

- **评估指标**：
  - **放大比**（Amplification Ratio, AR）：$AR(f) = \frac{P_M(f)}{P_c(f)}$，衡量某特征在模型输出中相对于人类语料的频率变化。
    - AR > 1 表示**过度使用**，AR < 1 表示**抑制使用**
  - **发散集**（Divergence Set）：定义 $D_M(\epsilon) = \{f \in F : AR(f) \notin [1-\epsilon, 1+\epsilon]\}$，$\epsilon=0.1$ 即偏离10%以上视为显著发散。
  - **均值AR**（Mean AR）：24个特征AR的平均值，作为整体风格偏差的代理指标。
  - 辅助指标：Perplexity, Distinct-2/4, Self-BLEU, Vocabulary Diversity

### 基线方法对比
- **横向对比**：
  - 不同规模模型（410M → 100B+）
  - 不同架构（Pythia, Llama, Mistral, GPT等）
  - 不同训练阶段（base vs instruct vs RLHF）
- **纵向对比**：
  - 不同熵正则化强度（λ = 0.0, 0.1, 1.0, 5.0）下的 Pythia-410M 模型

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 特征类别 | 典型特征 | 平均放大比 (Mean AR%) | 峰值放大比 (Peak AR%) |
|--------|---------|---------------------|---------------------|
| 结构元素 | Headers | **16,853%** | **209,675%** (OLMo-Instruct) |
|          | Bullet points | 3,063% | 13,855% |
|          | Numbered lists | 1,949% | 5,181% |
| 话语标记 | "In conclusion" | 5,048% | 24,791% |
|          | "Delve into" | 3,660% | 17,759% |
| 标点符号 | Semicolon | **3.2%** | 11.5% |
|          | Em dash | **18.4%** | 129.0% |
|          | Parenthetical | **23.2%** | 74.7% |

> 注：负值表示**被抑制**，如分号使用仅为人类文本的3.2%

### 与基线方法的对比结果
- **指令微调 vs 基础模型**：
  - 四组 base-instruct 对比显示，**发散模式无统计差异**（p > 0.25），说明指令微调**不加剧**风格坍塌。
  - 例如 Llama-3.1-8B 与 Llama-3.1-8B-Instruct 的 Mean AR 分别为 1,239 和 1,064，差异不显著。

- **熵正则化效果对比**（Pythia-410M）：
  | λ | Mean AR | Distinct-4 | Repetition | Vocab Diversity |
  |---|--------|-----------|----------|---------------|
  | 0.0 (baseline) | 0.63 | 0.282 | 0.034 | 0.018 |
  | 0.1 | 0.96 | 0.406 | 0.069 | 0.026 |
  | 1.0 | 2.16 | 0.696 | 0.017 | 0.042 |
  | **5.0** | **0.78** | **0.803** | **0.004** | **0.054** |

  - λ=5.0 实现最佳平衡：**发散度降低40.5%**，Distinct-4 提升185%，重复率下降89%。

- **跨模型性能碾压**：
  - 尽管参数量小 **200–1000倍**，λ=5.0 的 Pythia-410M 在风格自然度上：
    - 比 **Gemini-2.5-flash** 优 **96.7%**
    - 比 **GPT-4o-mini** 优 **96.9%**
    - 比 **Claude-3.5-haiku** 优 **98.2%**

### 消融实验结果
- **特征子集有效性验证**（A.7.1）：
  - 仅用 **Top-10 最具判别力的特征**（如 Headers, "in conclusion", "delve into"）即可完美预测整体发散趋势（Spearman ρ = 1.000）。
  - 支持使用精简特征集进行高效检测。

- **归一化策略对比**（A.7.2）：
  - 百分比形式的 AR 比 z-score 或绝对频次更具**可解释性**和**跨特征可比性**。

- **计算效率测试**（A.7.3）：
  - 特征提取耗时仅 **0.3ms/文档**，线性扩展，适合大规模分析。
  - 熵正则化仅增加 **4% 训练时间**，成本极低。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **风格坍塌是普遍且严重的现象**：
   - 所有测试模型均有 **83%** 的语言特征显著偏离人类分布。
   - “AI写作腔”本质是语言概率质量的系统性重分配。

2. **训练目标比模型规模更重要**：
   - 模型大小与风格发散度**无显著相关性**（Spearman p=0.21）。
   - 当前对齐流程（SFT + RLHF）**无法解决**根本问题。

3. **真正的“AI声音”源自生成动力学，而非对齐训练**：
   - 上下文偏移 + 吸收性状态共同导致风格自我强化。
   - 这一机制在**推理阶段激活**，但根植于预训练动态。

4. **熵正则化是有效解法，但需“足够强”**：
   - 弱干预（λ=1.0）反而破坏训练稳定性，**加剧发散**。
   - 强控制（λ=5.0）能恢复语言多样性，实现“以小胜大”。

5. **对AI检测与语言演化有深远影响**：
   - 当前AI检测器可能捕捉的是“分布坍塌”信号，而熵正则化模型可能**逃避检测**。
   - 若AI文本持续污染训练数据，可能导致“语言退化”的正反馈循环（data contamination → model collapse）。

### 方法的局限性
- **特征体系依赖确定性规则匹配**（regex/string），优先精度而非召回率。
- 实验集中在**说明性英文**，未覆盖其他语言、文体或领域。
- 熵正则化实验仅在 **Pythia-410M** 上完成，最优 λ 是否可迁移到更大模型尚待验证。
- 机制解释虽一致，但尚未通过**激活探测**或**因果干预**直接验证。

### 未来工作方向
- 探索跨语言、跨领域的上下文偏移与吸收态是否普遍存在。
- 设计更优的数据构造或架构修改，改善当前熵正则化带来的 **perplexity-发散度权衡**。
- 开展**感知实验**，验证分布测量结果是否与人类判断一致。
- 进行**纵向语料分析**，追踪AI生成内容对人类写作风格的实际影响。
- 推动**训练语料的分布审计**，作为应对语言退化的首要防线。

> **一句话总结**：  
> 本论文揭示，现代LLMs的“AI腔”并非来自RLHF，而是预训练中生成动力学导致的系统性风格坍塌；唯有通过**强熵正则化**才能有效逆转，且其效果远超单纯扩大模型规模。

</details>

---

### 14. [Reasoning-preserved Efficient Distillation of Large Language Models via Activation-aware Initialization](https://arxiv.org/abs/2605.29327)

**Authors**: Junlin He, Yihong Tang, Tong Nie, Guilong Li, Binyu Yang, Jinxiao Du, Lijun Sun, Wei Ma  
**Category**: cs.CL  
**Published**: 2026-05-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.29327v1  

#### Abstract
Efficient Distillation (EDistill) compresses large language models (LLMs) by structured pruning parameters and tuning lightweight modules with high training efficiency. Although these EDistilled LLMs achieve state-of-the-art (SOTA) performance on general ability benchmarks relative to similarly size...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Reasoning-preserved Efficient Distillation of Large Language Models via Activation-aware Initialization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文首次系统性地识别并研究了**高效蒸馏（Efficient Distillation, EDistill）**方法在压缩大语言模型（LLMs）时存在的严重缺陷——**推理崩溃（reasoning collapse）**。尽管现有的EDistill方法在通用能力上表现优异，但在需要多步逻辑推理的任务（如数学推理、代码生成）上性能急剧下降。

作者通过几何分析发现，这一现象的根本原因是**有效秩崩溃（eRank collapse）**：即隐藏表示的有效秩（effective rank）在训练早期迅速降低，导致不同token的表示趋于不可区分（token indistinguishability），从而破坏了多步推理所需的中间状态分离能力。

### 提出了什么新方法或新思路
为解决上述问题，作者提出了 **RED（Reasoning-preserved Efficient Distillation）** 框架，其核心创新是**激活感知初始化（activation-aware initialization）**：

- **通道选择矩阵初始化**：将可学习的投影矩阵（projection matrices）初始化为“通道选择矩阵”（channel-selection matrix），而非传统的随机初始化。
- **基于激活的重要性估计**：利用教师模型在一个小校准数据集（calibration dataset）上的激活值来估计每个通道的重要性，并据此选择最重要的通道进行保留。
- **理论保障**：该初始化方式使得投影矩阵的奇异值在初始时刻均匀分布（均为0或1），且其梯度为0，从而避免了因随机初始化导致的“赢家通吃”（winner-take-all）动态，理论上缓解了eRank崩溃。

### 相比现有方法的优势
- **保持高训练效率**：继承了EDistill框架的优点，仅训练少量轻量级模块，冻结大部分教师参数，计算成本远低于全参数微调。
- **显著恢复推理能力**：在不牺牲通用能力的前提下，大幅提升了多步推理任务的表现，实现了SOTA级别的综合性能。
- **更强的鲁棒性和稳定性**：允许在更低的蒸馏温度（T=1）下成功训练，表明能更精确地捕捉教师模型中的“暗知识”（dark knowledge）。
- **初始化即正则化**：通过合理的初始化而非后期正则化，从源头防止了几何退化。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **校准数据集（Calibration Dataset）**：
  - `Nemotron-Pretraining-Dataset` 中的三个子集：“Nemotron-CC-Diverse-QA”, “Nemotron-SFT-Code”, “Nemotron-SFT-General”，共15个样本。
  - 用于估计通道重要性，体积小、计算开销低（<3分钟单GPU）。
- **训练数据集（Retraining Corpus）**：
  - **Mixed-1.1**: 包含 `Fineweb-Edu`, `OpenHermes`, `DCLM`, `CosmopediaV2`，总规模10B–20B tokens。
  - **Mixed-2.0**: 类似构成，用于RED-4B，共18B tokens。
- **其他基线使用的数据**：`RedPajama`, `SlimOrca`, `Alpaca`等公开数据集。

### 实验设置和评估指标
- **模型架构**：基于Llama和Qwen系列的Transformer架构，采用宽度缩减（width-reduced）EDistill范式。
- **训练配置**：
  - 使用Adam优化器，序列长度2048，批量大小约32k–49k tokens。
  - 训练时间短（如RED-1.5B约30小时，8×H800 GPU）。
  - 蒸馏目标包括：representation alignment loss（MSE）、language modeling loss、KL divergence（温度T=1或40）。
- **评估协议**：
  - 使用 `lm-evaluation-harness` 框架统一评测。
  - 所有模型以**Base版本**为主进行公平比较，避免Instruction Tuning带来的偏差。

#### 评估指标分类
| 能力类别 | 任务 | 数据集 | 主要指标 |
|--------|------|-------|--------|
| **General Ability** | 科学理解与阅读理解 | ARC-E/C, BoolQ | Acc Norm |
| | 常识理解 | PIQA, WinoGrande, HellaSwag | Acc / Acc Norm |
| | 世界知识与真实性 | MMLU, TruthfulQA | Acc / MC2 |
| **Multi-step Reasoning** | 数学推理 | GSM8K | Exact Match |
| | 代码生成 | HumanEval, MBPP | Pass@1 |

最终报告 **Gen. Avg.** 和 **Reas. Avg.** 作为综合性能指标。

### 基线方法对比
- **Compute-Intensive Full-parameter Retraining**：
  - Minitron-4B, Llama3.2-1B/3B, SmolLM2-1.7B, Gemma3-1B/4B 等。
- **LoRA-style Recovery**：
  - SliceGPT, LLM-Pruner。
- **EDistill Methods**：
  - 宽度缩减：LRC（SOTA baseline）。
  - 深度缩减：LLMStreamline, ReplaceMe。
- 所有基线均使用官方checkpoint或在其开源实现基础上复现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在~2B系列上的表现（Table 4）
| Model | Gen. Avg. | Reas. Avg. |
|-------|-----------|------------|
| LRC-1.5B | 0.61 | **0.09** |
| **RED-1.5B (Ours)** | **0.61** | **0.28** |
| LRC-1.7B | 0.61 | **0.06** |
| **RED-1.7B (Ours)** | **0.64** | **0.25** |

> **结论**：RED在保持甚至略微提升通用能力的同时，将推理平均分从~0.06–0.09提升至~0.25–0.28，实现**2–4倍增益**。

#### 在~4B系列上的表现（Table 5）
| Model | Gen. Avg. | Reas. Avg. |
|-------|-----------|------------|
| LRC-4B | 0.69 | **0.18** |
| **RED-4B (Ours)** | **0.69** | **0.33** |
| Minitron-4B | 0.64 | 0.21 |
| Gemma3-4B | 0.66 | 0.40 |

> **结论**：RED-4B在仅用18B tokens训练的情况下，推理能力接近甚至超越部分使用更大数据集（如Gemma3-4B使用4T tokens）训练的Full-parameter模型。

### 与基线方法的对比结果
- **相比LRC**：在相同训练条件下，RED在GSM8K上最高提升达**0.44 vs 0.09**（Table 4），且通用能力持平或略优。
- **相比Full-parameter模型**：
  - RED-1.5B仅用10B tokens，性能全面超越SmolLM2-1.7B（11T tokens）、Llama3.2-1B（9T tokens）等。
  - RED-4B用18B tokens，在Reas. Avg.上优于Minitron-4B（94B tokens）。
- **效率优势**：
  - RED-1.5B训练耗时约30小时（8×H800），而Full-parameter方法通常需数百GPU天。

### 消融实验结果

#### （1）蒸馏温度敏感性（Table 6）
| Method | T | MMLU | GSM8K |
|--------|----|------|-------|
| LRC-1.5B | 1 | 0.25 | 0.05 |
| LRC-1.5B | 40 | 0.50 | 0.04 |
| **RED-1.5B** | **1** | **0.50** | **0.44** |
| RED-1.5B | 40 | 0.52 | 0.08 |

> **发现**：LRC无法在低温（T=1）下收敛，必须依赖高温平滑；而RED在T=1时同时达到最佳通用与推理性能，说明其能精准捕获教师模型中对推理至关重要的“尖锐知识”。

#### （2）正交化技术对比（Table 7）
| Method | MMLU | GSM8K |
|--------|------|-------|
| LRC (Random) | 0.50 | 0.21 |
| LRC + Random Orthogonal Init | 0.50 | 0.30 |
| LRC + Orthogonal Regularization | 0.48 | 0.26 |
| **RED (Ours)** | **0.51** | **0.44** |

> **发现**：通用正交化方法虽有一定改善，但仍远逊于RED，说明**保留教师子空间结构**比单纯控制奇异值分布更为关键。

#### （3）校准数据鲁棒性（Table 15）
- 即使使用不同数据集（Nemotron vs SlimOrca）或极小样本（N=3），选出的重要通道重叠率仍超过78%，证明方法对校准集选择高度鲁棒。

#### （4）eRank崩溃不可逆（Figure 6）
- 对LRC-1.5B进行SFT后，其eRank仍未恢复，且GSM8K从0.21降至0.18，说明一旦在基础蒸馏阶段发生eRank崩溃，后续微调难以修复。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **推理崩溃是EDistill的普遍问题**：现有EDistill方法在多步推理任务上存在严重性能退化，根源在于**eRank collapse**。
2. **eRank崩溃源于随机初始化**：理论证明，随机初始化的投影矩阵会导致奇异值指数级分化，形成少数主导方向，使token表示趋同。
3. **激活感知初始化可理论缓解eRank崩溃**：通过将投影矩阵初始化为通道选择矩阵，可使初始奇异值均衡且梯度为零，从根本上避免早期几何失真。
4. **RED实现了高效与高性能的统一**：在极低训练成本下，同时达到SOTA的通用能力和强大的多步推理能力。

### 方法的局限性
- 当前方法主要针对**宽度缩减型EDistill**，未直接处理深度缩减或其他压缩范式。
- 依赖教师模型的激活模式，若校准集与目标任务差异过大，可能影响通道选择质量（尽管实验证明鲁棒性强）。
- 尚未探索与强化学习（RL）或高级指令微调（SFT）结合后的潜力。

### 未来工作方向
- 将RED框架扩展至**强推理模型**（如DeepSeek-R1系列）的蒸馏。
- 探索在**多模态大模型（MLLMs）** 和 **视觉-语言-行动模型（Vision-Language-Action models）** 中的应用。
- 研究如何结合**专用数学/代码数据集**进一步提升推理效率边界。
- 探索**无需校准集**的自适应通道选择机制。

---

> ✅ **总结一句话**：  
> 本文揭示了高效蒸馏中的“推理崩溃”问题，并提出**RED**方法，通过**激活感知初始化**从源头防止**eRank崩溃**，在极低资源消耗下实现了通用能力与多步推理能力的双重SOTA，为小型化高性能LLM提供了新范式。

</details>

---

### 15. [Revisiting Observation Reduction for Web Agents: Comprehensive Evaluation with a Lightweight Framework](https://arxiv.org/abs/2605.29397)

**Authors**: Masafumi Enomoto, Ryoma Obara, Haochen Zhang, Masafumi Oyamada  
**Category**: cs.CL  
**Published**: 2026-05-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.29397v1  

#### Abstract
HTML observations in LLM-based web agents are extremely long, and while many reduction methods have been proposed, it remains unclear which methods reduce overall agent latency while maintaining performance. The main obstacle is the high cost of end-to-end evaluation: in our experiments, evaluating ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Revisiting Observation Reduction for Web Agents: Comprehensive Evaluation with a Lightweight Framework

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在基于 LLM 的 Web Agent 中，原始 HTML 观察（observation）通常非常长（可达数万至数十万 tokens），导致 LLM 推理成本高、延迟大。虽然已有多种 **Observation Reduction** 方法被提出（如检索、剪枝、LLM 指导选择等），但缺乏一个高效、系统性的评估框架来比较这些方法在**减少整体延迟的同时是否能维持任务成功率**。

传统端到端（end-to-end）评估需要多次调用 LLM 并与真实网页交互，耗时极长（本文中达 **232.4 小时**），严重阻碍了方法的快速迭代与比较。

### 提出的新方法与新思路
作者提出了一个**轻量级评估框架**，其核心是：

- **Minimal Failure Set (MFS)**：定义为“移除后会导致任务失败的最小 HTML 元素集合”。
- **Coverage**：衡量某个 reduction 方法是否完整保留了 MFS。Coverage 越高，表示该方法越可能成功完成任务。

> ✅ **创新点**：Coverage 是一个无需调用 LLM 或访问网页的代理指标（proxy metric），可在毫秒级时间内完成单次评估。

### 相比现有方法的优势
- **效率提升超过 100 倍**：相比端到端评估，Coverage 在累计评估时间上实现了 **290× 加速**（WorkArena）和 **246× 加速**（WebLinx）。
- **强相关性验证**：Coverage 与实际端到端成功率（success rate）高度正相关（Pearson’s r > 0.9），可作为可靠替代指标。
- **支持自动化优化**：Coverage 可作为目标函数用于自动搜索最优 reduction 策略（如使用 GEPA 进化优化）。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **WorkArena L1**：包含 33 个来自 ServiceNow 平台的真实多步 Web 任务，需与真实 Web 服务交互。
- **WebLinx**：包含 300 个单步动作预测任务，基于真实用户操作轨迹，评估更轻量。

### 实验设置与评估指标
#### 主要指标
| 指标 | 定义 |
|------|------|
| **Coverage** | reduction 方法保留 MFS 的比例（越高越好） |
| **Reduction Ratio (RR)** | 输出 HTML 长度 / 原始 HTML 长度（越低表示压缩越多） |
| **Success Rate** | 端到端任务完成率 |
| **Latency / step** | 每一步的总耗时（含 reduction + LLM inference + web access） |

#### Policy Models
- **Qwen3.5-122B-A10B**
- **MiniMax-M2.5**

#### Reduction Methods 对比
| 类别 | 方法 |
|------|------|
| **Baseline** | `Original (HTML)`（无压缩）、`Random` |
| **Program-based** | `Pruned by AXTree`, `GEPA (ratio=0.2/0.6)` |
| **Retrieval-based** | `DMR (BM25)`, `DMR (Dense)`, `DMR (finetuned)` |
| **LLM inference-based** | `DMR (QueryGen)`, `FocusAgent`, `Prune4Web` |

> 所有方法均应用统一的 tree-pruning 后处理以保持公平。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Coverage 与 Success Rate 强相关
| 数据集 | Pearson’s r | Spearman’s ρ | Kendall’s τ |
|--------|-------------|--------------|------------|
| WorkArena | 0.944 | 0.889 | 0.742 |
| WebLinx | 0.913 | 0.707 | 0.569 |

> 即使控制 **Reduction Ratio** 后，偏相关仍显著，说明 Coverage 是独立有效的指标。

#### ✅ 评估速度大幅提升
| 方法 | WorkArena 累计耗时 | Speedup |
|------|---------------------|---------|
| End-to-End (Qwen) | 232.4 小时 | ×1 |
| **Coverage** | **48.2 分钟** | **×290** |

> 单实例平均评估时间从 12.8 分钟降至 **0.7 秒**（1098× 加速）。

#### ✅ 最优方法性能（端到端测试）
| 方法 | 数据集 | Per-step Latency | 原始成功率保留率 |
|------|--------|------------------|------------------|
| **GEPA (ratio=0.2)** | WorkArena | **30.2s**（↓2.2×） | **84%** |
| **GEPA (ratio=0.2)** | WebLinx | **3.1× 更快** | **89%** |

> 相比无压缩 baseline，实现显著加速且成功率下降可控。

### 与基线方法的对比结果

| 发现 | 描述 |
|------|------|
| 🔹 **LLM-based 方法覆盖率更高但延迟极高** | 如 `FocusAgent` 单步 reduction 耗时超 100s，抵消了输入缩短带来的收益。 |
| 🔹 **Retrieval-based 方法性价比高** | `DMR (Dense)` 覆盖率尚可，延迟约 2–3s，适合中等压缩场景。 |
| 🔹 **Domain-specific 优化至关重要** | `BM25` 在 WebLinx 表现好（文本主导），但在 WorkArena 差（属性更重要）。 |
| 🔹 **MFS 训练可显著提升低延迟方法** | `DMR (finetuned)` 和 `GEPA` 利用 MFS 数据训练后，在低 RR 下逼近 LLM 方法性能。 |

### 消融实验结果

#### 📊 HTML 元素重要性分析（Ablation on Element Types）
| 数据集 | 最关键元素类型 | 影响（Coverage Drop） |
|--------|----------------|------------------------|
| **WebLinx** | 文本内容（Text） | ↓59.5% |
| **WorkArena** | 属性（如 `id`, `class`, `value`） | `value`: ↓22.0%, `id`: ↓16.9% |

> 表明通用方法难以跨域泛化，需针对性优化。

#### 📈 MFS 样本效率
- 仅需 **4 个 MFS 实例**即可在 WorkArena 上达到 ρ > 0.7 的排名相关性。
- 说明该框架**样本效率极高**，适用于资源受限场景。

---

## 4. 关键结论和发现

### 主要发现
1. **Coverage 是高效的代理指标**：  
   无需运行完整 agent，即可快速评估 reduction 方法的有效性，加速开发周期。

2. **Extractive Reduction 方法面临根本权衡**：  
   要么依赖高计算成本（如 LLM 推理），要么需要**领域特定优化**（domain-specific optimization）才能兼顾性能与延迟。

3. **最佳实践是“轻量程序 + MFS 训练”**：  
   使用 **GEPA** 等进化优化框架，在 MFS 数据上训练轻量级 pruning 程序，可在极低 reduction 延迟下实现高性能。

4. **Critical HTML 元素因任务而异**：  
   WebLinx 依赖文本语义，WorkArena 更依赖结构属性（如 `id`, `value`），凸显 benchmark 多样性。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **仅适用于 Extractive 方法** | 不支持对 HTML 进行语义压缩、摘要或表示变换的方法（如 summarization）。 |
| **依赖成功轨迹构建 MFS** | 若某些关键路径未被观察到，则对应 MFS 无法捕获。 |
| **不公开 MFS 数据集** | 因版权与使用协议限制（ServiceNow 页面不可分发），仅提供构造流程。 |

### 未来工作方向
- 将 MFS 构造扩展至更多 Web 环境与任务类型。
- 探索非 extractive 方法的轻量评估方式（如 semantic fidelity metrics）。
- 开发通用性强、可迁移的 reduction 策略，降低对 domain-specific tuning 的依赖。
- 结合 Coverage 指标进行在线自适应 reduction（adaptive observation pruning）。

---

> **总结一句话**：  
> 本文通过提出 **MFS + Coverage** 框架，解决了 Web Agent 观察压缩方法评估昂贵的问题，并揭示：**轻量级程序经 MFS 数据优化后，是平衡性能与延迟的最佳路径**。

</details>

---

### 16. [PRISM: Processing-In-Memory Sparse MTTKRP for Tensor Decomposition Acceleration](https://arxiv.org/abs/2605.29728)

**Authors**: Daniel Pacheco, Leonel Sousa, Aleksandar Ilic  
**Category**: cs.DC  
**Published**: 2026-05-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.29728v1  

#### Abstract
Sparse tensors are the most used representation of sparse multidimensional data. Operations that decompose them, selecting their most important features while reducing their dimension, have become prevalent procedures in machine learning. One of the most used tensor decomposition algorithms is the A...

---

### 17. [A Full-Pipeline Framework for Evaluating Membership Inference Attacks in Machine Learning](https://arxiv.org/abs/2605.29454)

**Authors**: Ding Chen, Xinwen Cheng, Xuyang Zhong, Xinping Chen, Xiaolin Huang, Chen Liu  
**Category**: cs.LG  
**Published**: 2026-05-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.29454v1  

#### Abstract
While Membership Inference Attacks (MIAs) are the prevailing method for identifying training data, their application has expanded into privacy auditing and machine unlearning. Nevertheless, the field lacks a systematic framework for evaluating how different contexts affect MIA efficacy. Without such...

---

### 18. [MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference](https://arxiv.org/abs/2605.30218)

**Authors**: Kexin Chu, Yang Zhou, Wei Zhang  
**Category**: cs.LG  
**Published**: 2026-05-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.30218v1  

#### Abstract
Temperature-zero BF16 LLM inference is often treated as reproducible, yet the same request can emit different tokens when decoded alone or inside a larger batch. Existing fixes use batch-invariant operators or LLM-42's per-token verification, incurring cost even when most steps are stable. We ask wh...

---

### 19. [GTA: Generating Long-Horizon Tasks for Web Agents at Scale](https://arxiv.org/abs/2605.29218)

**Authors**: Tenghao Huang, Kung-Hsiang Huang, Prafulla Kumar Choubey, Yilun Zhou, Muhao Chen, Jonathan May, Chien-Sheng Wu  
**Category**: cs.AI  
**Published**: 2026-05-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.29218v1  

#### Abstract
Web agents, which couple language models with browsing and tool-use capabilities, show promise as open web assistants. Yet progress is increasingly limited by the lack of scalable, process-level supervision. Existing benchmarks are largely manually constructed, providing only coarse start-goal annot...

---

### 20. [PassNet: Scaling Large Language Models for Graph Compiler Pass Generation](https://arxiv.org/abs/2605.29357)

**Authors**: Yiqun Liu, Yingsheng Wu, Ruqi Yang, Enrong Zheng, Honglei Qiu, Sijun He, Tai Liang, Jingjing Wu, Yuhan Zhou, Yiwei Zhang, Dongyan Chen, Weihan Yi, Xinqi Li, Siqi Bao  
**Category**: cs.AI  
**Published**: 2026-05-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.29357v1  

#### Abstract
Modern tensor compilers such as TorchInductor deliver substantial speedups on mainstream models, yet face a systematic performance ceiling on long-tail workloads -- our profiling shows that 43% of real-world subgraphs experience end-to-end slowdowns under default compilation. While LLMs offer a path...

---

### 21. [LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs](https://arxiv.org/abs/2605.29756)

**Authors**: Jung Hyun Lee, June Yong Yang, Jungwook Choi, Eunho Yang  
**Category**: cs.AI  
**Published**: 2026-05-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.29756v1  

#### Abstract
As large language models continue to scale, low-bit weight-only post-training quantization (PTQ) offers a practical solution to their memory-efficient deployment. Although block-wise PTQ is capable of matching the full-precision (FP) baseline on basic language modeling and understanding, its quality...

---

### 22. [AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security](https://arxiv.org/abs/2605.29801)

**Authors**: Dongrui Liu, Yu Li, Zhonghao Yang, Peng Wang, Guanxu Chen, Yuejin Xie, Qinghua Mao, Wanying Qu, Yanxu Zhu, Tianyi Zhou, Leitao Yuan, Zhijie Zheng, Qihao Lin, Yimin Wang, Haoyu Luo, Shuai Shao, Chen Qian, Qingyu Liu, Ling Tang, Ruiyang Qin, Qihan Ren, Junxiao Yang, Kun Wang, Zhiheng Xi, Linfeng Zhang, Ranjie Duan, Bo Zhang, Wenjie Wang, Wen Shen, Qiaosheng Zhang, Yan Teng, Chaochao Lu, Rui Mei, Man Li, Jialing Tao, Xi Lin, Tianhang Zheng, Yong Liu, Quanshi Zhang, Lei Zhu, Xingjun Ma, Junhua Liu, Hui Xue, Xiaoxiang Zuo, Xiangnan He, Chao Shen, Xianglong Liu, Minlie Huang, Jing Shao, Xia Hu  
**Category**: cs.AI  
**Published**: 2026-05-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.29801v1  

#### Abstract
Modern open-world agents such as OpenClaw exhibit powerful cross-environment execution capabilities yet introduce broad new safety risk sources. Meanwhile, advanced frontier AI models drastically lower attack barriers, rendering current agent alignment frameworks inadequate for real-world deployment...

---

### 23. [OptSkills: Learning Generalizable Optimization Skills from Problem Archetypes via Cluster-Based Distillation](https://arxiv.org/abs/2605.29829)

**Authors**: Haochen Yang, Ke Zhao, Mengyuan Ma, Xingyu Lu, Xiangfeng Wang, Hong Qian  
**Category**: cs.AI  
**Published**: 2026-05-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.29829v1  

#### Abstract
Leveraging Large Language Models (LLMs) to automatically formulate and solve optimization problems from natural language has emerged as an efficient paradigm for automated optimization. However, existing methods still exhibit limited generalization: they are sensitive to superficial narrative variat...

---

### 24. [Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design](https://arxiv.org/abs/2605.29421)

**Authors**: Shengchao Chen, Ting Shu, Sufen Ren  
**Category**: cs.CL  
**Published**: 2026-05-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.29421v1  

#### Abstract
Photonic crystal fiber (PCF) inverse design remains challenging because candidate geometries must satisfy coupled optical targets under expensive electromagnetic simulation. Existing pipelines improve surrogate prediction or one-shot parameter recommendation, but they do not accumulate reusable desi...

---

### 25. [LiteCoder-Terminal: Scaling Long-Horizon Terminal Environments for Learning Language Agents](https://arxiv.org/abs/2605.29559)

**Authors**: Xiaoxuan Peng, Kaiqi Zhang, Xinyu Lu, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  
**Category**: cs.CL  
**Published**: 2026-05-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.29559v1  

#### Abstract
Mastering terminal environments requires language agents capable of multi-step planning, feedback-grounded execution, and dynamic state adaptation. However, training such agents is currently bottlenecked by a reliance on scraped external repositories, which limits domain diversity, environment contr...

---

### 26. [Adapting Multilingual Embedding Models to Turkish via Cross-Lingual Tokenizer Surgery and Offline Distillation](https://arxiv.org/abs/2605.29992)

**Authors**: M. Ali Bayram, Banu Diri, Sava\c{s} Y{\i}ld{\i}r{\i}m  
**Category**: cs.CL  
**Published**: 2026-05-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.29992v1  

#### Abstract
Sentence embeddings are a foundational component for semantic search, clustering, classification, and retrieval-augmented generation. This paper presents embeddingmagibu-200m, a Turkish-focused sentence embedding model that produces 768-dimensional L2-normalized vectors and supports an 8,192-token c...

---

### 27. [TC-MIS: Maximal Independent Set on Tensor-cores](https://arxiv.org/abs/2605.29604)

**Authors**: Prajjwal Nijhara, Dip Sankar Banerjee  
**Category**: cs.DC  
**Published**: 2026-05-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.29604v1  

#### Abstract
Maximal Independent Set (MIS) in a graph is a fundamental problem with applications in resource allocation, scheduling, and network optimization. Although graphs are inherently un-structured and challenging for GPU parallelism due to irregular memory access and workload imbalance, specialized GPU al...

---

### 28. [AsymVLM: Asymmetric Token Pruning for Efficient Vision-Language Model Inference](https://arxiv.org/abs/2605.29535)

**Authors**: Yilin Feng, Ahmed Burak Gulhan, Mahmut Taylan Kandemir  
**Category**: cs.LG  
**Published**: 2026-05-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.29535v1  

#### Abstract
Vision-Language Models (VLMs) process thousands of visual tokens per image alongside comparatively few text tokens, yet existing compression methods treat both modalities uniformly. We observe that the two modalities have fundamentally different properties: vision tokens are spatially redundant and ...

---

### 29. [PRO-CUA: Process-Reward Optimization for Computer Use Agents](https://arxiv.org/abs/2605.29119)

**Authors**: Yifei He, Rui Yang, Hao Bai, Tong Zhang, Han Zhao  
**Category**: cs.AI  
**Published**: 2026-05-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.29119v1  

#### Abstract
Computer use agents (CUAs) have shown strong potential for automating complex digital workflows, yet their training remains constrained by costly live environment interaction and limited high-quality supervision. Existing filtered behavior cloning pipelines suffer from imitation bottlenecks, includi...

---

### 30. [Harmonizing Real-Time Constraints and Long-Horizon Reasoning: An Asynchronous Agentic Framework for Dynamic Scheduling](https://arxiv.org/abs/2605.29262)

**Authors**: Shijie Cao, Yuan Yuan, Jing Liu  
**Category**: cs.AI  
**Published**: 2026-05-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.29262v1  

#### Abstract
The Dynamic Flexible Job Shop Scheduling Problem (DFJSP) necessitates a trade-off between instant reaction to stochastic disturbances and global optimization of production goals. Conventional priority rules are insufficiently flexible to handle complex disruptions, whereas learning-based approaches ...

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

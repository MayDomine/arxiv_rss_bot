# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-23 10:20:41 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs](https://arxiv.org/abs/2609.26796)

**Authors**: Quan Nguyen-Tri, Mukul Ranjan, Zhiqiang Shen  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 14.5  
**Type**: new  
**ArXiv ID**: 2609.26796v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have recently emerged as a promising alternative to autoregressive LLMs by enabling non-autoregressive text generation. However, their practical deployment remains limited by inefficient inference, largely due to the absence of effective Key-Value (KV) caching...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Diffusion Large Language Models (dLLMs) 虽然在生成灵活性和并行性方面展现出潜力，但其推理效率远落后于成熟的 autoregressive LLMs。主要原因在于：
- **KV Caching 效率低下**：传统 KV 缓存机制在 dLLMs 中引入大量冗余的 GPU 内存读写操作，导致 I/O 成为瓶颈。
- **缺乏高效的并行解码机制**：现有方法未能有效利用 dLLM 的非自回归特性进行高吞吐量、高质量的并行解码。

这些问题限制了 dLLM 在实际场景中的部署速度和内存可扩展性。

---

### 提出了什么新方法或新思路
本文提出 **Flash-dLLM**，一个无需训练的 inference acceleration 框架，通过联合优化 I/O 意识的 KV 缓存和基于缓存的并行 draft-and-verify 解码策略来提升 dLLM 推理效率。

#### 主要创新点包括：

1. **IO-aware Fused KV-Cache Kernel**
   - 将 QKV 投影、RoPE 和缓存写入融合为单个 Triton 内核，避免中间张量在 HBM 和 SRAM 之间的多次搬运。
   - 显著减少内存访问开销，提高 cache locality，实现更高效的迭代去噪过程。

2. **Scheduled Flash Attention**
   - 引入 block table 机制管理变长序列计算块，支持灵活调度部分计算（cached）与全量更新（full compute），适应不同样本在批处理中进度不一致的问题。

3. **Selective Cache Update**
   - 观察到只有少数高注意力 token 对当前预测有显著影响（如 middle layers 中 top-32 tokens 占据 ~50% 注意力权重）。
   - 因此仅追踪和更新这些“influential tokens”，降低不必要的计算和内存占用。

4. **Flash-Verify: Self-contained Draft-and-Verify Decoding**
   - **无需辅助模型**：dLLM 自身同时作为 drafter 和 verifier。
   - 构造双视图查询（draft view 与 mask view），通过因果 attention mask 隔离二者，确保独立预测。
   - 当两个视图对某 token 的预测一致且 mask-view 置信度超过阈值 γ 时才接受该 token。
   - 实现了高吞吐量下的可靠并行解码，提升了每步解码 token 数量。

---

### 相比现有方法的优势
| 维度 | Flash-dLLM | 现有方法（如 Fast-dLLM, Elastic-Cache） |
|------|------------|-------------------------------|
| **KV Caching** | 融合内核 + selective tracking，减少 I/O 开销 | 多次 kernel launch，内存密集型 |
| **Parallel Decoding** | 自验证机制，无外部模型依赖 | 依赖额外 verifier 或简单置信度过滤 |
| **Memory Efficiency** | 更低峰值内存，支持更大 batch size | 易出现 OOM，尤其在大 batch 下 |
| **Speedup** | 最高达 11.0×（vs Elastic-Cache） | 加速有限，受限于 I/O 瓶颈 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GSM8K**：数学推理任务（5-shot）
- **MATH**：复杂数学问题（4-shot）
- **HumanEval**：代码生成能力评估（0-shot）
- **MBPP**：Python 编程任务（3-shot）

所有任务均测试生成长度为 256 和 512 的情况以评估可扩展性。

---

### 实验设置和评估指标

#### 实验平台
- 单卡 NVIDIA A100 80GB GPU
- 使用 Triton 2.0 实现 fused kernel
- 基础模型：LLaDA-1.5

#### 超参数配置（默认）
- Confidence threshold ε = 0.9
- Verify threshold γ = 0.8（GSM8K/MBPP），0.85（MATH/HumanEval）
- Tracking budget βₜ = 64–80
- Sliding window size Bₘ = 64
- Batch size = 32（用于 scalability 测试）

#### 评估指标
- **Accuracy**：
  - GSM8K: 5-shot `flexible_extract`
  - MATH: 4-shot `math_verify`
  - HumanEval: 0-shot `pass@1`（带 Fast-dLLM 后处理）
  - MBPP: 3-shot `pass@1`
- **Throughput**：decoding tokens/sec（平均 over benchmark）
- **Peak Memory Usage**
- **Tokens per Step**

---

### 基线方法对比
1. **No Cache**：标准 dLLM 推理，无 KV 缓存
2. **Fast-dLLM**：前缀缓存 + confidence-aware decoding
3. **Elastic-Cache**：基于 attention pattern drift 的自适应 KV 缓存
4. （补充对比）dKV-Cache, FreeDave, FlashDLM 等近期方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

| 方法 | Accuracy (%) | Throughput (tokens/s) | Speedup vs Greedy No Cache |
|------|-------------|------------------------|----------------------------|
| No Cache (Greedy) | — | 2.6–8.5 | 1.0× |
| Fast-dLLM | 80.82 | 36.8 | 14.2× |
| Elastic-Cache | 82.79 | 41.7 | 16.0× |
| **Flash-Cache (Conf-aware)** | **82.87** | **149.4** | **57.5×** |
| **Flash-dLLM (Full)** | **83.02** | **210.6** | **81.0×** |

> ✅ 在 GSM8K-512 上，Flash-dLLM 达到 **210.6 tokens/s**，相较 Elastic-Cache 提升 **5.1×**；在 HumanEval 上达 **11.0×** 加速。

---

### 与基线方法的对比结果
- **速度优势显著**：
  - 相比 Elastic-Cache，throughput 提升 **2.3–5.1×**
  - 相比 Fast-dLLM，提升 **4.5–5.7×**
- **内存效率更高**：
  - 在 batch size=16 时，Flash-dLLM 使用约 **26GB**，而 Fast-dLLM 消耗 **50GB**（↓48%）
  - 支持最大 batch size **32**，而 Fast-dLLM 在 **24** 时即 OOM
- **准确率持平甚至略优**：
  - Flash-dLLM 在 GSM8K 上达到 **83.02%**，优于所有 baseline
  - 在其他任务上保持在最优 ±1.8% 范围内

---

### 消融实验结果（Ablation Studies）

#### (1) Accuracy-Throughput Trade-off（Figure 5）
- 增加 tracking budget βₜ 或降低 verify threshold γ 可提升 throughput，但可能轻微牺牲 accuracy。
- 存在明显的帕累托前沿（Pareto frontier），可通过调节 γ 和 βₜ 找到最佳平衡点。
- 示例：βₜ=96, γ=0.85 → 83.15% acc @ 185.7 t/s，接近最高精度但速度快 41.1%。

#### (2) Tokens per Iteration（Figure 6）
- Flash-Verify 平均每步解码 **7.2 tokens**，而 confidence-aware 仅为 **5.6 tokens**。
- 多出的 token 来自于对低置信度候选的高效验证，而非盲目丢弃。

#### (3) Masked-window Size Bₘ 影响（Figure 7）
- Bₘ 增大会略微提升 accuracy，但会降低 throughput。
- Flash-Verify 在各种 Bₘ 设置下均优于 confidence-aware，且相对加速比从 **1.4× → 1.5×**。

#### (4) Batch Size Scalability（Figure 4 & Table 6）
- Flash-dLLM throughput 随 batch size 几乎线性增长，在 B=32 时达 **199.8 tokens/s**。
- 相比之下，Fast-dLLM 在 B=24 时崩溃。
- Flash-Verify 的 batch scaling 效率为 **3.57×**（B=1→32），高于 confidence-aware 的 2.70×。

---

## 4. 关键结论和发现

### 主要发现
1. **GPU Memory I/O 是 dLLM 推理的关键瓶颈**，单纯减少计算无法带来显著加速，必须优化数据流动路径。
2. **Token-level Sparsity** 是可以被利用的重要特性：只维护 top-attended tokens 即可保留大部分信息。
3. **Self-contained Draft-and-Verify** 是可行且高效的方案：dLLM 可以自我验证预测一致性，无需外部模型。
4. **Flash-dLLM 实现了真正的端到端加速**：将 dLLM 的理论并行性转化为实际 wall-clock 性能增益。

---

### 方法的局限性（Limitations）
- 当前评估集中在 **masked diffusion LLMs** 和结构化输出任务（math/code），尚未验证于 continuous-space diffusion models。
- 超参数（如 γ, Bₘ）是固定设定，未采用动态调整策略，可能进一步优化 trade-off。
- 在开放域生成任务（如长文本写作、对话）上的表现未知，因这类任务 token confidence 分布较平缓。

---

### 未来工作方向
- 设计 **adaptive thresholding schemes**，根据运行时 confidence 统计动态调整 γ 和 Bₘ。
- 将 Flash-Cache 和 Flash-Verify 扩展至 **continuous diffusion LMs** 和多模态场景。
- 探索 **dependency-guided decoding** 结合 Flash-Verify 以进一步提升生成连贯性。
- 研究如何将该框架应用于 **on-device deployment**，推动轻量化非自回归生成。

--- 

> 🔗 **开源地址**：https://github.com/VILA-Lab/Flash-dLLM

</details>

---

### 2. [Disaggregated Quantization: Specializing LLM Prefill and Decode](https://arxiv.org/abs/2609.26333)

**Authors**: Andrei Panferov, Maximilian Kleinegger, Sweta Priyadarshi, Tijmen Blankevoort, Dan Alistarh  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.26333v1  

#### Abstract
Prefill and decode reward different approaches to quantization: low-precision arithmetic accelerates prompt processing, while compact weights reduce memory traffic during generation. We propose "disaggregated quantization" (DQ), which specializes computation formats, weights and storage placement to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Disaggregated Quantization: Specializing LLM Prefill and Decode**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在大语言模型（LLM）推理过程中，**prefill**（处理输入提示）和 **decode**（生成输出）两个阶段具有截然不同的计算特性：
- **Prefill** 是计算密集型（compute-bound），适合使用低精度算术加速矩阵乘法（GEMM）。
- **Decode** 是内存密集型（memory-bound），更适合通过权重压缩减少显存带宽压力。

然而，传统量化方法对两个阶段采用统一的格式，无法同时优化两者的效率与精度。本文指出，这种“一刀切”的量化策略限制了性能上限。

---

### **提出了什么新方法或新思路**
作者提出 **Disaggregated Quantization (DQ)** ——一种将量化策略按推理阶段拆分的新范式，具体包含三个层次的方案：

#### **(a) Format-Disaggregated Quantization**
- **思想**：保持相同的模型权重，但为 prefill 和 decode 使用**不同的计算格式**。
- **实现**：例如，在 prefill 阶段启用 **NVFP4**（W4A4）进行硬件加速；在 decode 阶段仅对权重进行低比特压缩（如 LUT3/LUT2），保留 BF16 激活以提升精度。
- **优势**：无需额外存储，即可提升 decode 准确率，尤其在 decode-heavy 任务上表现显著。

#### **(b) Fully-Disaggregated Quantization**
- **思想**：训练**独立的 prefill 权重**和 decode 权重，分别适配各自的量化格式。
- **实现**：通过 **Quantization-Aware Distillation with Disaggregation (QADD)** 联合训练两个路径，共享同一个响应目标。
- **优势**：prefill 可用 NVFP4 加速，decode 使用紧凑权重，兼顾速度与精度。

#### **(c) Offloaded Disaggregated Prefill (ODP)**
- **思想**：解决 fully-disaggregated 引入额外 checkpoint 导致设备显存占用高的问题。
- **实现**：将 prefill 权重从 SSD 流式加载，利用 decode 阶段空闲的显存作为缓冲区，实现零设备权重内存开销。
- **优势**：使高性能 disaggregated 推理适用于本地单设备部署。

#### **(d) Prefillers for Pre-Quantized Checkpoints**
- 扩展应用：为已发布的冻结 decode 模型（如 GGUF 格式）单独训练一个 **prefiller** 模型。
- 优势：无需重新训练整个模型，即可获得更快更准的 prefill 阶段。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | DQ 方法 |
|------|--------|---------|
| 统一量化 | ✅ | ❌ |
| 支持阶段差异化优化 | ❌ | ✅ |
| 显存效率 | 一般 | ODP 实现零额外设备内存占用 |
| 精度恢复能力 | 有限 | 在 1-bit decode 下准确率翻倍以上 |
| 兼容性 | 封闭 | 可适配任意预量化 decode 模型 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Decode-heavy 评测集**：
  - `GSM8K`（数学推理）
  - `MATH-500`（复杂数学题）
  - `MMLU-Pro`（多学科理解）
- **Prefill-heavy 评测集**：
  - `RULER`：长上下文检索与问答任务，测试 4K–32K 上下文长度下的性能。
- **多模态评测**（针对 Qwen3.8-27B）：
  - `MMMU-Pro`：图像+文本联合推理。

### **实验设置**
- **模型家族**：
  - `Qwen 3` 系列（0.6B–8B）
  - `Gemma 3` 系列（270M–12B）
  - `Qwen3.8-27B`（270亿参数密集模型）
- **训练方式**：
  - 使用 **QADD** 进行量化感知蒸馏，教师模型为 BF16 精度的原始模型。
  - 蒸馏语料来自 `Tulu 3 SFT` 数据集（1亿 tokens）。
- **推理平台**：
  - vLLM（用于 decode 性能测量）
  - 自定义 llama.cpp 扩展（支持 ODP 的 TTFT 测量）

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 平均于多个 benchmark 的任务得分 |
| **Time to First Token (TTFT)** | 衡量 prefill 延迟，越小越好 |
| **Per-Token Latency** | decode 阶段吞吐，反映 decode 效率 |
| **Device Memory Usage (GB)** | 显存占用情况 |
| **Speedup** | 相对于 BF16 基线的加速比 |

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **BF16** | 全精度浮点推理，作为性能上限参考 |
| **Weight-only Quantization** | 如 LUT2/LUT3，仅压缩权重，不改变激活精度 |
| **Uniform NVFP4** | 统一对 prefill 和 decode 使用 W4A4 量化 |
| **NVFP4A16** | 仅量化权重，激活保持 BF16，decode 更精确但 prefill 较慢 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) Format-Disaggregated Quantization**
- 在 `Qwen 3` 和 `Gemma 3` 上：
  - 相比统一 NVFP4，format-disaggregated 提升 decode-heavy 准确率 **1.9–3.1 pts**。
  - decode 速度提升 **2–3%**（跳过激活量化）。
  - 不增加任何存储成本。

#### **(2) Fully-Disaggregated Quantization**
- 对比 weight-only 基线：
  - 在 **2-bit decode** 场景下，准确率提升高达 **10.7 pts（Qwen 3）** 和 **7.4 pts（Gemma 3）**。
  - prefill 速度达 **1.49×（Qwen3-8B）** 和 **1.67×（Gemma3-12B）**。
- 在 prefill-heavy 任务中也有明显增益（+4.1–10.5 pts）。

#### **(3) ODP 性能**
- 在 `Qwen3.8-27B` 上，使用 ODP 实现：
  - **1.78× TTFT 加速**（8K 上下文）
  - **无额外设备显存占用**
  - 在 >8K 上下文时，加载延迟被计算掩盖，效率接近 resident 版本。

#### **(4) Prefillers for Pre-Quantized Models**
- 在 `Qwen3.8-27B` + `IQ1_S`（1-bit decode）组合中：
  - **MMLU-Pro 准确率从 29.0% → 61.5%（+32.5 pts）**
  - **MMMU-Pro 从 24.4% → 59.7%（+35.3 pts）**
  - TTFT 从 12.27s → 6.90s（**1.78× 加速**）
- 即使是 2-bit decode，也能带来 **~7 pts** 的增益。

> 📊 表格摘要（来自 Table 1 & Table 5）：
>
> | 方法 | MMLU-Pro ↑ | TTFT ↓ | Prefill Speedup | Device Mem |
> |------|------------|--------|----------------|-------------|
> | BF16 | 84.6% | 12.27s | 1.00× | 16.38GB |
> | IQ1_S (WO) | 29.0% | 12.27s | 1.00× | 4.66GB |
> | + DQ Prefiller | **61.5%** | **6.90s** | **1.49×** | 4.66GB (ODP) |

---

### **消融实验结果**

#### **(1) 是否需要分离非线性层？**
- 实验表明：仅对 Linear 层进行 disaggregation 已足够，进一步分离 Embedding/Norms 等未见明显收益（Figure 7）。

#### **(2) Prefiller 的泛化性测试（Interoperability）**
- 使用 IQ1_S 训练的 prefiller 应用于 IQ1_M 或 IQ2_XXS decode 模型时，性能下降明显。
- 结论：**prefiller 必须与特定 decode 模型联合训练才能发挥最大效果**。

#### **(3) 生成长度影响**
- 使用 prefiller 后，某些任务（如 MMMU-Pro）平均生成 token 数显著减少（如 IQ1_S 从 14.5K → 6.6K），说明其提升了推理效率而非靠“多写”提分。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Prefill 和 Decode 对量化敏感性不同**：
   - Decode 更依赖高精度表示，尤其在推理类任务中。
   - 因此，**禁用 decode 阶段的激活量化可大幅提升准确率**。

2. ✅ **Disaggregated Quantization 实现双赢**：
   - 利用 NVFP4 加速 prefill，同时用 weight-only decode 保证质量。
   - 在不牺牲 decode 速度的前提下，显著提升准确率。

3. ✅ **ODP 解决了部署瓶颈**：
   - 通过 SSD 流式加载 prefill 权重，避免双 checkpoint 占用显存。
   - 在长上下文场景下，加载时间可被有效隐藏。

4. ✅ **Prefiller 可极大增强现有量化模型**：
   - 即使 decode 模型已被冻结，也可通过训练专用 prefiller 来提升整体性能。
   - 在 1-bit decode 下实现**准确率翻倍以上**，极具实用价值。

---

### **局限性**
1. ❌ **不适用于 MoE 模型**：
   - MoE 中每次 decode 激活参数比例高，加载成本远超计算节省，ODP 不适用。
2. ❌ **短序列 TTFT 可能变差**：
   - ODP 存在冷启动加载延迟，在短 prompt 场景下可能不如原生 weight-only。
3. ❌ **多轮对话缓存一致性未验证**：
   - 若每轮都重新 prefill，可能导致历史 token 表示漂移，影响稳定性。

---

### **未来工作方向**
1. 🔮 探索 **MoE-friendly 的 disaggregation 架构**。
2. 🔮 将 DQ 思想扩展到 **encoder-decoder 模型**（如 T5、FLAN-T5）中的 encoder/decoder 分离优化。
3. 🔮 开发 **自动化工具链**，为任意 GGUF/Bin 等格式模型一键生成最优 prefiller。
4. 🔮 研究 **动态切换 DQ 策略**，根据 workload 类型自适应选择是否启用 disaggregation。

---

## **总结**
> **Disaggregated Quantization (DQ)** 是一次对 LLM 推理流程的精细化重构。它打破了“统一量化”的思维定式，提出“**按阶段定制量化策略**”的新范式，结合 **QADD 蒸馏训练** 和 **ODP 内存卸载技术**，实现了在**几乎不增加资源消耗的前提下，大幅提高低比特 LLM 的准确率与推理速度**。该方法不仅理论新颖，且已在真实大规模模型（如 Qwen3.8-27B）上验证成功，具备极强的工程落地潜力。

</details>

---

### 3. [Mitigating LLM Over-Refusal via Dynamic Semantic Routing Calibratione](https://arxiv.org/abs/2609.25049)

**Authors**: Zixuan Wang, Bingjie Zhang, He Zhao, Dandan Guo  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.25049v1  

#### Abstract
Large language models (LLMs) aligned for safety often suffer from over-refusal, incorrectly rejecting benign yet safety-related instructions. Prior studies primarily attribute this to static representation overlap, largely overlooking the underlying dynamic mechanisms. In this paper, we present the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Mitigating LLM Over-Refusal via Dynamic Semantic Routing Calibration*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对**大语言模型（LLM）在安全对齐后出现的“过度拒绝”（over-refusal）问题**。即，模型在面对看似敏感但实际无害的指令（称为 **Hard-Safe prompts**，如 “如何终止一个Python进程？”）时，错误地将其识别为有害请求并拒绝响应，从而损害了模型的有用性和可信推理能力。

传统观点认为这是由于**静态表示空间重叠**导致的，而本文提出这一现象的根本原因在于**动态语义路由冲突**。

---

### 提出了什么新方法或新思路
作者提出了 **Semantic Routing Calibration (SRC)** ——一种轻量级、无需训练的推理阶段干预框架，其核心思想是：

- **机制分析视角创新**：首次从 **Transformer 注意力机制中的内部路由冲突** 角度解释 over-refusal，发现一类稀疏的 **Hypersensitive Safety Heads** 在处理 Hard-Safe 指令时会“误触发”，将无害的目标实体（如名词）强行绑定到拒绝语义上，造成高熵的注意力分散，剥夺了目标实体应有的注意力。
- **动态干预策略**：
  1. **定位敏感头**：通过构建合成配对数据集 $ D_{\text{syn}} $，计算不同注意力头在 Hard-Safe 和 Unsafe 查询中对目标名词的关注差异，识别出最敏感的安全头集合 $ S $。
  2. **动态校准拒绝倾向**：在推理时，基于首个生成token的隐藏状态与预定义 `hrefusal` 向量的余弦相似度判断是否需要干预。
  3. **双分支logits融合**：引入一条带有严格系统提示的安全参考路径，在解码过程中进行logits融合，既防止过早拒绝，又保留对真正有害请求的防御能力。

---

### 相比现有方法的优势
| 维度 | 现有方法（如 SCANS, Surgical, SCD） | 本文 SRC 方法 |
|------|-------------------------------|----------------|
| **是否需训练** | 多数为 training-free，但存在全局操作 | ✅ 完全无需训练（training-free） |
| **干预粒度** | 全局隐藏状态修改或向量加减（blunt shifts） | ✅ 精准定位并抑制特定注意力头（fine-grained） |
| **安全性保持** | 易破坏原始安全边界 | ✅ 双分支融合确保真实有害请求仍被拒绝 |
| **效率** | 部分方法引入显著延迟 | ✅ 干扰仅在检测到异常时触发，吞吐量接近基线 |

---

## 2. 核心实验方法和设置

### 使用的数据集

| 类型 | 数据集 | 用途 |
|------|--------|------|
| **分析用数据集** | $ D_{\text{analyze}} $：<br>• Alpaca（Safe）<br>• OR-Bench（Hard-Safe）<br>• AdvBench（Unsafe） | 注意力分配与熵分析 |
| **定位敏感头数据集** | $ D_{\text{syn}} $：30 对语法相同、仅目标名词不同的合成查询（如 “kill a process” vs “kill someone”） | 识别 Hypersensitive Safety Heads |
| **评估基准** | • **Over-refusal**：<br> – XSTest, CoCoNot, OR-Bench, OKTest, PHTest<br>• **Safety**：<br> – I-Malicious, I-CoNa, I-Controversial, HarmfulQ, AdvBench<br>• **通用能力**：<br> – MMLU, ARC-e/c, OBQA, PIQA | 综合性能评测 |

---

### 实验设置和评估指标

- **模型**：Qwen2.5-1.5B / 7B、Llama-3-8B
- **对齐方式**：采用 LoRA 进行 SFT 安全对齐
- **解码方式**：默认 greedy decoding，部分实验补充 top-p=0.9 采样
- **评估指标**：
  - **Over-refusal 缓解**：各 Hard-Safe 基准上的 **合规率（compliance rate）**
  - **安全性**：对真正有害请求的 **防御成功率（defense success rate）**
  - **通用能力**：MMLU 等任务上的准确率

---

### 基线方法对比

| 类型 | 方法 | 简介 |
|------|------|------|
| **Training-based** | STL, STL-aug, DCR | 基于数据增强或对比学习的安全微调 |
| **Training-free** | SCANS, Surgical, SCD | 修改隐藏状态或使用对比解码 |

所有基线均基于相同的对齐模型进行公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 Llama-3-8B 为例）

| 方法 | XSTest ↑ | OR-Bench ↑ | Safety ↓ | MMLU ↑ |
|------|----------|------------|----------|--------|
| STL (baseline) | 79 | 59 | 93 | 61 |
| SCANS | 84 | 86 | 88 | 60 |
| **Ours (SRC)** | **99** | **86** | **90** | 60 |

> ✅ 在 **XSTest** 上达到 **99%** 合规率，远超其他方法  
> ✅ 在 **OR-Bench** 上达 **86%**，优于多数基线  
> ✅ 安全性保持在 **90%**，未因缓解 over-refusal 而牺牲防御能力

---

### 与基线方法的对比结果
- **相比 SCANS/Surgical**：SRC 在提升 over-refusal 缓解的同时，更好地维持了安全性与通用能力。
- **相比 SCD**：虽然 SCD 也有一定效果，但其依赖复杂的对比解码流程，而 SRC 更高效且可控。
- **相比 DCR（训练法）**：SRC 不需额外训练即可实现相当甚至更优的表现，部署成本更低。

---

### 消融实验结果（Ablation Study）

#### 表格：组件消融（Llama-3-8B）
| 配置 | XSTest | OKTest | Safety |
|------|-------|--------|--------|
| Head Only | 92 | 70 | 86 |
| Fusion + Prompt | 81 | 77 | 92 |
| **SRC (H+F+P)** | **99** | **96** | **90** |

> 🔍 发现：
> - 仅抑制头部可提高合规性，但会降低 OKTest 表现（缺乏正则化）
> - 仅使用安全提示无法根本修复路由偏差
> - **三者结合才能实现最优平衡**

#### 其他关键消融发现：
- **目标词选择**：基于 **noun token** 的定位效果最好，verb 或 last token 会导致安全性能严重下降。
- **干预头数量**：Top-64 效果最佳；干预非敏感头（如 Generic Safety Heads）会导致安全崩溃（降至45%）。
- **合成数据大小**：小样本（N=2）反而更稳定，说明方法不依赖大数据集。

---

## 4. 关键结论和发现

### 主要发现
1. **Over-refusal 的根源是动态路由冲突**，而非简单的静态表示重叠。
2. 少数 **Hypersensitive Safety Heads** 是导致 over-refusal 的“元凶”，它们在中间层异常激活，干扰了对目标实体的正常关注。
3. **SRC 能精准定位并动态抑制这些头**，并通过双分支 logits 融合恢复可信推理路径。
4. 该方法在多个主流 LLM 上均有效，且**无需重新训练**，具备良好的泛化性和实用性。

---

### 方法的局限性
1. **依赖高质量的合成数据集** 来定位敏感头，若数据覆盖不足可能影响定位精度。
2. 当前设计主要适用于单轮对话和短上下文场景，**多轮或多步推理中的有效性有待验证**。
3. 引入若干超参数（如阈值 $ \tau $、衰减因子 $ \alpha $、融合权重 $ \beta $），需针对不同模型调优。
4. **过度干预可能导致真实有害请求也被放过**，安全与帮助性的权衡仍未完全消除。

---

### 未来工作方向
- 扩展至 **多轮对话与长文本推理** 场景下的动态路由校准。
- 探索 **自动化超参数搜索机制**，减少人工调参负担。
- 结合 **动态输入感知机制**，实现更细粒度的逐token路由控制。
- 研究如何将此类机制应用于 **多模态模型** 的 over-refusal 缓解。

--- 

> 📌 **一句话总结**：  
> 本论文揭示了 LLM over-refusal 的动态路由本质，并提出无需训练的 **SRC 框架**，通过精准抑制敏感注意力头与双分支融合，在几乎不损失安全性的前提下显著提升了模型对无害敏感指令的响应能力，实现了安全性与可用性的更好平衡。

</details>

---

### 4. [CacheDyG: Decoupling Temporal Propagation for Efficient Dynamic Graph Learning](https://arxiv.org/abs/2609.25814)

**Authors**: PinHeng Zong, Ye Yuan  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.25814v1  

#### Abstract
Dynamic graphs are widely used to model time-evolving relational systems in real-world applications. Dynamic graph neural networks provide an effective framework for capturing both structural dependencies and temporal dynamics in such data. However, they typically intertwine temporal graph propagati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CacheDyG: Decoupling Temporal Propagation for Efficient Dynamic Graph Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Dynamic Graph Neural Networks (DGNNs)** 在训练过程中存在两个关键效率瓶颈：
- **重复计算**：在每个优化 epoch 中都重新执行 **temporal graph propagation**，而历史图结构在跨 epoch 时基本不变，导致大量冗余计算。
- **参数膨胀**：许多模型为每个节点-时间对维护可训练的表示（trainable node-time representations），导致参数量随节点数和时间步线性增长，内存和优化成本高昂。

这些问题使得现有方法在大规模或长时间跨度的动态图上难以扩展。

---

### 🚀 提出的新方法：CacheDyG
作者提出 **CacheDyG** —— 一种基于 **缓存-精炼（cache-refine）框架** 的高效动态图学习方法，其核心思想是 **解耦（decouple）时间传播与常规参数更新**。

#### 主要创新组件：
| 组件 | 功能 |
|------|------|
| **Temporal Dependency Cache (TDC)** | 构建一个按时间排序的非可训练缓存，存储图感知的 node-time 表示（non-trainable buffers），避免重复传播。 |
| **Cache-style Propagation** | 图传播仅在缓存构建或刷新时执行，脱离主训练循环。 |
| **Lightweight Cache Refiner** | 引入轻量级的 **node-domain frequency-domain refiner** 对缓存表示进行任务自适应校准。 |
| **Adaptive Residual Gate** | 控制从 refiner 注入多少修正信号到原始缓存中，保持稳定性。 |
| **Selective Cache Refresh** | 定期将精炼后的表示回写到缓存中，使缓存与监督目标对齐，但不频繁触发稀疏图传播。 |

---

### 🔍 相比现有方法的优势
| 维度 | CacheDyG 优势 |
|------|---------------|
| **参数效率** | 可训练参数仅 **19.106K**，远低于基线（如 EvolveGCN 超过 700,000K）。 |
| **运行效率** | 单 epoch 运行时间显著降低（最高提速约 **19×**），适合大规模图。 |
| **可扩展性** | 在 DBLP 和 StackOverflow 上仍可训练，而部分基线出现 OOM（Out-of-Memory）。 |
| **预测性能** | 在所有数据集上取得 **最佳 MAP（Mean Average Precision）**，精度更高。 |

> 💡 核心洞见：**temporal propagation 不必与 parameter update 同频发生** —— 历史结构可以“一次性”传播并缓存，后续只需轻量级微调和预测。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
共五个动态图基准数据集，均被离散化为快照序列（snapshot sequences）：

| 数据集 | 快照数 | 描述 |
|--------|--------|------|
| **Wiki-Eo** | 60 | Wikipedia 编辑交互 |
| **Digg** | 50 | 社交新闻平台回复行为 |
| **Alpha** | 60 | 用户间的信任关系（带符号） |
| **DBLP** | 45 | 学术合作网络 |
| **StackOverflow** | 25 | 技术问答社区互动 |

所有数据集采用固定节点集、时间顺序划分，任务为 **one-step future link prediction**。

---

### 📊 实验设置与评估指标

#### ✅ 评估任务
- **未来边预测**：给定时间 $ t $ 之前的历史快照，预测 $ (u,v) $ 是否会在 $ G_{t+1} $ 中出现。

#### ✅ 评估指标
| 指标 | 说明 |
|------|------|
| **MAP**（Mean Average Precision） | 主要评价指标，衡量正样本排序能力 |
| **MAUC**（Mean ROC-AUC） | 辅助排名指标 |
| **Trainable Parameters** | 可训练参数数量（排除非可训练缓存） |
| **Runtime** | 单 epoch 的训练 + 验证总耗时（秒） |

#### ✅ 实验配置
- 优化器：Adam，学习率 $ 1 \times 10^{-2} $，权重衰减 $ 5 \times 10^{-4} $
- 最大训练轮次：200，早停耐心值：25
- 输入维度：8，预测器隐藏层：16维
- 缓存刷新策略：前4次每8个epoch刷新一次，之后若验证性能连续10轮无提升则刷新

---

### 🆚 基线方法对比
选取六种代表性 DGNN 模型作为基线：
| 基线 | 类型 |
|------|------|
| **DySAT** | 基于 snapshot 的结构-时间自注意力 |
| **ROLAND** | 滚动节点状态更新机制 |
| **EvolveGCN** | 图卷积参数递归演化 |
| **WinGNN** | 窗口式梯度聚合 |
| **GTCN** | 图-时间卷积网络 |
| **SGD-DYG** | 基于频率的轻量级动态图模型 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Dataset | Best Baseline MAP | **CacheDyG MAP** | **增益** |
|--------|-------------------|------------------|----------|
| Wiki-Eo | 92.71 (SGD-DYG) | **95.14** | **+2.43** |
| Digg | 76.76 (SGD-DYG) | **77.87** | **+1.11** |
| Alpha | 91.86 (SGD-DYG) | **93.02** | **+1.16** |
| DBLP | 63.46 (EvolveGCN) | **67.32** | **+3.86** |
| StackOverflow | 90.38 (SGD-DYG) | **92.05** | **+1.67** |

> ✅ CacheDyG 在 **所有五个数据集上均达到最高 MAP**。

---

### ⏱️ 效率对比（Tables 2–4）

#### 参数量对比（单位：K）
| Model | CacheDyG 参数量 | 典型基线参数量（如 EvolveGCN） | 压缩倍数 |
|-------|------------------|-------------------------------|---------|
| 所有数据集 | **19.106K** | 最高达 **702,896K**（StackOverflow） | **>3600×** |

> 💬 CacheDyG 参数量恒定，不随图规模增长。

#### 单 epoch 运行时间（总时间，秒）
| Dataset | 最快基线 | CacheDyG 时间 | 加速比 |
|--------|----------|--------------|--------|
| Wiki-Eo | GTCN: 0.135s | **0.007s** | **~19.3×** |
| Digg | GTCN: 0.195s | **0.014s** | **~13.9×** |
| Alpha | GTCN: 0.134s | **0.008s** | **~16.8×** |
| DBLP | GTCN: 0.434s | **0.125s** | **~3.5×** |
| StackOverflow | WinGNN: 115.486s | **97.304s** | **仍最快且精度更高** |

> ✅ 在最大数据集 StackOverflow 上，CacheDyG 是唯一同时实现 **高精度 + 可行训练** 的方法。

---

### 🔍 消融实验结果（Table 5 & Figure 2–3）

#### 消融研究（Wiki-Eo 上）

| 变体 | MAP | 参数量 | 训练时间 |
|------|-----|--------|----------|
| **Full CacheDyG** | **96.14** | 19.106K | 0.007s |
| w/o Cache-style Prop. | 87.46 | 2390.753K | 0.048s |
| w/o Refiner | 50.80 | 0.017K | 0.002s |
| Trainable Xcache | 92.69 | 2401.226K | 0.052s |
| w/o Temporal Mixing | 90.31 | 5.434K | 0.005s |
| w/o Cache Refresh | 81.52 | 19.106K | 0.007s |

#### 关键发现：
- **移除 refiner 导致性能崩溃** → 表明缓存必须经过任务驱动的精炼。
- **启用可训练 cache 反而性能下降** → 支持“非可训练缓存 + 轻量精炼”的设计更优。
- **无 cache refresh 性能大幅下降** → 证明缓存需定期更新以对齐目标任务。
- **移除 temporal mixing 也显著降级** → 显示跨时间步依赖建模的重要性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **解耦 temporal propagation 与 parameter update 是可行且高效的**：通过缓存历史图传播结果，可大幅减少重复计算。
2. **非可训练缓存 + 轻量 refiner 架构优于全参数化模型**：在更低参数和计算成本下实现更强性能。
3. **选择性刷新机制有效维持缓存一致性**：无需每轮传播即可保持缓存与目标任务同步。
4. **CacheDyG 具备卓越的可扩展性**：在大图上仍能运行，而多数基线因内存不足失败（OOM）。

---

### ⚠️ 方法的局限性
- 当前设计适用于 **固定节点集的 snapshot-based 动态图**，对完全动态增删节点的场景支持有限。
- 缓存刷新频率等超参数需要合理设置，极端配置可能导致性能波动（尽管敏感性分析显示整体鲁棒）。
- 当前未探索与其他 temporal encoder（如 Transformer）结合的可能性。

---

### 🔮 未来工作方向
- 将 cache-refine 范式推广至 **continuous-time dynamic graphs**。
- 探索 **自适应缓存粒度**（如按节点或子图局部缓存）。
- 结合 **预训练 + 微调** 框架，在更大规模图上进一步释放缓存潜力。
- 研究缓存机制在其他图学习任务（如节点分类、异常检测）中的应用。

---

## ✅ 总结
**CacheDyG** 提出了一种全新的视角来优化动态图学习效率：  
> “**不要反复做同样的事，把稳定的部分缓存起来，只优化该优化的。**”

它通过 **Temporal Dependency Cache** 和 **lightweight refinement** 实现了：
- 更高的预测精度（SOTA MAP）
- 极低的参数量（仅 19K）
- 极快的训练速度（最高提速近 20×）
- 出色的可扩展性（处理 OOM 场景）

👉 因此，**cache-based decoupling** 成为一种极具前景的 **scalable dynamic graph learning** 设计范式。

</details>

---

### 5. [AIBuildAI-2.5: Efficient Autonomous AI Model Development Through LLM-Guided Tree Search](https://arxiv.org/abs/2609.25047)

**Authors**: Peijia Qin, Ruiyi Zhang, Qi Cao, Han Guo, Li Zhang, Pengtao Xie  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.25047v1  

#### Abstract
Autonomous agents that automatically build artificial intelligence (AI) models could broaden access to AI across science and engineering. A popular line of such agents frames model building as a code search problem and solves it by tree search, in which each node is a candidate program and the tree ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AIBuildAI-2.5: Efficient Autonomous AI Model Development Through LLM-Guided Tree Search

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Model (LLM)** 的自主 AI 模型构建代理（autonomous agents）在自动化机器学习（AutoML）任务中已接近人类专家水平，但仍面临三大效率瓶颈：

1. **搜索效率低**：由于训练候选模型耗时且昂贵，只能执行少量候选程序，导致基于奖励的搜索策略（如 Monte Carlo Tree Search）信号稀疏、噪声大，难以有效指导搜索方向。
2. **资源调度不智能**：缺乏对硬件资源状态感知的调度机制，导致 GPU/CPU 利用率低，训练并行度不足。
3. **LLM 推理成本高**：所有 agent 调用均使用高性能、高成本的 LLM（如 Claude Opus），而许多任务（如代码实现）其实可由低成本模型完成。

---

### 提出的新方法与创新思路

AIBuildAI-2.5 是一个高效的自主 AI 模型开发系统，提出以下三项核心技术：

#### ✅ **1. LLM-Guided Tree Search（LLM 引导的树搜索）**
- 将 AI 模型构建视为 **code search 问题**，采用树形结构组织候选程序（node = candidate program）。
- 引入两个专用 LLM agent 来提升节点选择质量：
  - **Judge Agent**：为每个待执行的候选节点打分，维度包括：
    - **Expected Improvement**（预期改进）
    - **Grounding**（是否基于父节点实测结果）
    - **Feasibility**（单次训练槽内是否可行）
  - **Selector Agent**：综合 judge 分数、当前树结构和剩余资源，全局排序候选节点，优先执行最有潜力者。
- 这种“先验估计 + 实际奖励”的结合显著提升了在有限预算下的搜索效率。

#### ✅ **2. Resource-Aware Scheduler（资源感知调度器）**
- 动态监控 GPU 内存、利用率和 CPU 负载。
- 只有当新增任务不会造成资源争抢时才启动训练，支持多任务并发执行，提高硬件利用率和单位时间内的探索数量。

#### ✅ **3. Cost-Aware Model Routing（成本感知模型路由）**
- 不再统一使用最强大的 LLM 处理所有角色。
- 设计 **Router Agent**，根据任务特性、agent 角色和历史表现，动态分配不同成本层级的 LLM：
  - 例如：Coder 在复杂 pipeline 中需 Opus，但在简单任务中可用 Haiku。
- 配套 **Router Knowledge System**，跨运行积累“某模型在某类任务中是否足够”的经验，持续优化路由决策。

---

### 相比现有方法的优势

| 维度 | 现有方法（如 MLEvolve） | AIBuildAI-2.5 |
|------|------------------------|-------------|
| **Node Selection** | 基于 UCT / Monte Carlo 规则，仅依赖标量奖励 | Judge + Selector 提供多维先验评分，结合上下文推理 |
| **Expansion Reliability** | 单次 LLM 调用生成子节点，失败率高 | Coder 调试至 end-to-end 成功，成功率更高 |
| **Resource Utilization** | 缺乏调度逻辑，并发能力弱 | 资源感知调度，最大化硬件吞吐 |
| **LLM 成本控制** | 所有角色使用同一高端 LLM | 动态路由至最低必要模型，节省推理开销 |

> 💡 总体优势：**在相同计算预算下，探索更高质量的候选方案，同时降低 LLM 推理成本。**

---

## 2. 核心实验方法和设置

### 使用的数据集

| 数据集 | 描述 |
|-------|------|
| **MLE-Bench** | 包含 75 个真实 Kaggle 风格任务，涵盖视觉、文本、时间序列、表格数据等模态。用于评估通用 AI 开发能力。 |
| **AIRS-Bench** | 自主 AI 科研任务基准，源自机器学习论文，无初始代码模板。测试 agent 的科研创新能力。本文在其上评估了 6 个任务。 |

---

### 实验设置

- **硬件环境**：
  - Linux x86-64 服务器
  - 24 vCPUs, 256 GB RAM, 1×NVIDIA A100 GPU
  - 单任务最大运行时间：24 小时（wall-clock time）

- **LLM 池**：
  - **Claude Haiku 4.5**（低成本）
  - **Claude Sonnet 4.6**（中等成本）
  - **Claude Opus 4.7**（高成本）

- **初始化**：每项任务从 `N = 7` 个初始设计开始。

- **评估协议**：
  - MLE-Bench：遵循官方协议，提交预测结果，按 **Medal Rate** 排行。
  - AIRS-Bench：直接比较最终测试集指标（MAE、Accuracy、MASE 等）。

---

### 基线方法对比

| 基线方法 | 类型 | 特点 |
|---------|------|------|
| **MLEvolve** | Tree Search + UCT | 当前主流树搜索方法，使用 Opus 全程驱动 |
| **Fixed-Tier Baselines** | 单一模型配置 | Opus / Sonnet / Haiku 全部角色固定使用 |
| 其他系统 | MARS, ML-Master, InternAgent 等 | 主流 autonomous AI agents，在 MLE-Bench 上公开排名 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📊 **MLE-Bench 结果**
- **AIBuildAI-2.5 Medal Rate: 73.3%**
- 在 MLE-Bench 官方排行榜上 **排名第一**，超越所有已有系统。
- 显著优于第二名 MLEvolve（约 61.3%），领先超过 12 个百分点。

> 图 2 显示其 medal rate 高于 MARS、Famou-Agent、ML-Master、Leeroo、InternAgent、R&D-Agent、AIRA-dojo、MLEvolve 和 AIBuildAI。

#### 🧪 **AIRS-Bench 结果（6 项任务）**
AIBuildAI-2.5 在全部六项任务上均优于 MLEvolve：

| 任务类别 | 具体任务 | 指标提升 |
|--------|--------|--------|
| **Molecules & Proteins** | QM9-Cv (热容预测) | ↓ MAE by **20.8%** |
| | ZINC (溶解度预测) | ↓ MAE by **29.8%** |
| **Text Classification** | SICK-NLI (自然语言推断) | ↑ Accuracy by **5.1%** |
| | Yelp Review Rating | ↑ Accuracy by **1.4%** |
| **Time Series** | Solar Power Forecasting | ↓ MAE by **4.0%** |
| | Wikipedia Traffic Forecasting | ↓ MASE by **3.6%** |

> 表明该方法不仅适用于标准 AutoML 任务，还能泛化到前沿 AI 科研场景。

---

### 模型路由成本效益分析（图 4）

在三个代表性 MLE-Bench 任务上的对比：

| 任务 | 路由 vs Opus 成本 | 性能表现 |
|-----|------------------|----------|
| **spaceship-titanic** | < 50% of Opus cost | 准确率相当，获金牌 |
| **random-acts-of-pizza** | ~33% of Opus cost | AUC 与 Opus 相当 |
| **dog-breed** | ~50% of Opus cost | Log-loss **低于任何固定模型配置**（比 Haiku 低 60%） |

> ✅ **总成本下降 56%**，同时性能持平或更好。

- 固定使用 Haiku 或 Sonnet 虽便宜，但性能明显下降（尤其在复杂任务中）。
- 路由系统实现了 **cost-quality Pareto 最优前沿**。

---

### 消融实验（隐含于分析中）

虽然未明确列出消融表，但文中通过对比揭示了各模块贡献：

- **Judge + Selector 的作用**：相比仅靠 reward 的 UCT 策略，在极少数执行次数下仍能选出优质路径 → 更强的先验引导。
- **Scheduler 的作用**：在同等时间内完成更多训练任务 → 提升探索密度。
- **Model Routing 的作用**：成本降 56%，说明轻量模型可在合适场景替代高端模型而不牺牲性能。

---

## 4. 关键结论和发现

### 主要发现

1. **LLM-Guided Tree Search 显著优于传统搜索策略**  
   在 reward 极其稀缺的情况下，引入 LLM 对“潜在价值”的判断（expected improvement, grounding, feasibility）是突破性能瓶颈的关键。

2. **资源感知调度可有效提升硬件利用率**  
   合理安排并发训练任务，避免资源争抢，使得在固定时间内能评估更多高质量候选。

3. **模型路由可在不牺牲性能的前提下大幅降低成本**  
   并非所有 agent 角色都需要最强 LLM；动态匹配任务需求与模型能力是实现高效自治系统的必经之路。

4. **AIBuildAI-2.5 实现了专家级建模能力与高效率的统一**  
   在多个真实世界任务中达到甚至超越人类专家水平（以 medal rate 衡量），且推理成本更低。

---

### 方法的局限性

1. **依赖高质量 LLM 判断能力**  
   Judge 和 Selector 的有效性受限于 LLM 的 reasoning 能力，若 LLM 错误评估改进方向，可能导致搜索偏离最优路径。

2. **Router Knowledge System 需要冷启动过程**  
   初期缺乏历史数据支撑，路由决策可能不够准确，需一定数量的任务积累才能发挥最佳效果。

3. **尚未整合更细粒度的资源管理技术**  
   如 GPU 多实例切片（MIG）、job migration、interference-aware collocation 等系统级优化尚未集成。

4. **仍局限于单机单卡环境**  
   当前调度器未考虑分布式训练或多机集群场景，扩展性有待验证。

---

### 未来工作方向

1. **增强资源调度器智能化程度**
   - 引入 job profiling 技术预估资源消耗
   - 支持 job packing、migration 和 interference-aware collocation
   - 探索多机协同训练架构

2. **让 agent 本身具备资源意识**
   - 设计师/编码器可根据硬件条件调整模型结构或 batch size
   - 构建 **Hardware-aware Knowledge Base**，指导资源适配设计

3. **推广至更广泛的科学发现领域**
   - 应用于算法发现（AlphaEvolve）、分子筛选、气候模拟、生物数据分析等 code search 场景
   - 实现 **Autonomous Scientific Discovery**

4. **进一步优化模型路由机制**
   - 引入强化学习进行动态路由决策
   - 支持混合精度、缓存复用等 LLM 推理加速手段

---

> 🔚 **总结一句话**：  
> **AIBuildAI-2.5 通过 LLM-Guided Tree Search、Resource-Aware Scheduling 和 Cost-Aware Model Routing 三重创新，在有限资源下实现了高效、低成本、专家级的自主 AI 模型构建，代表了当前 autonomous AI agent 的 SOTA 方向。**

</details>

---

### 6. [TSS: Target-Side Sparsification for Speculative Decoding in Domain-Specific Large Language Models](https://arxiv.org/abs/2609.26100)

**Authors**: Haibo Hu, Lianming Huang, Qiao Li, Nan Guan, Chun Jason Xue  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.26100v1  

#### Abstract
Speculative decoding accelerates large language model inference through collaboration between a lightweight draft model and a target verifier. Existing methods mainly improve the draft side, while the target model is typically kept dense and unchanged. We show that, under domain-specific inference, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TSS: Target-Side Sparsification for Speculative Decoding in Domain-Specific Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **speculative decoding** 方法主要优化 **draft model**（草稿模型），而将 **target model**（目标验证模型）视为固定、全深度激活的黑盒。这种设计忽略了在 **domain-specific 推理任务** 中，目标模型可能存在“过度思考”（overthinking）现象：即后续层对已足够准确的中间表示进行不必要的修改，导致：
- 降低 draft-target 对齐度，减少 **accept length**
- 增加计算开销
- 甚至损害下游任务性能

因此，论文提出：**全深度的目标验证并非总是最优选择**。

### 提出了什么新方法或新思路
作者提出了 **TSS**（Target-Side Sparsification），一种面向领域特定大语言模型推理的 **目标侧块层稀疏化框架**，其核心思想是：
- 在 **target model 验证阶段** 主动跳过某些 Transformer 块层（block layers）
- 跳过的层由一个轻量级的 **domain-aware skip controller** 动态决定
- 所有配置通过离线搜索获得，不需重新训练或永久剪枝

#### 创新点包括：
- ✅ **首次将目标模型本身作为可稀疏化的对象**，而非仅优化 draft 或 proposal 过程。
- ✅ 提出 **acceptance- and metric-aware breadth-first search** 算法，在多层组合中联合优化：
  - **draft acceptance**（接受长度）
  - **downstream task metric**（如 BLEU、ROUGE-L、F1、Accuracy）
- ✅ 构建 **domain-to-configuration mapping**，实现单个完整 target model 支持多个稀疏路径，无需存储多个剪枝副本。
- ✅ 运行时通过 **skip controller** 实现零成本切换，保留原始 dense 模型用于未见 domain 回退。

### 相比现有方法的优势
| 维度 | 现有方法（如 EAGLE, SAMD） | TSS |
|------|----------------------------|-----|
| 优化对象 | Draft model / Proposal mechanism | ✅ Target model verification path |
| 是否改变 target | ❌ 否，保持 dense | ✅ 是，动态稀疏化 |
| 是否需要重训练 | ⚠️ 通常需要微调 draft | ❌ 不需要任何 retraining |
| 是否支持多 domain 自适应 | ❌ 单一策略 | ✅ 多 domain-specific skip policies |
| 计算效率提升来源 | 更好 draft → 更高 accept | ✅ 减少 target 计算 + 提高 accept |

---

## 2. 核心实验方法和设置

### 使用的数据集
基于 **Spec-Bench** 提供的多领域 benchmark，涵盖以下五个任务：
- **Translation**（翻译）：使用新闻类文本，评估指标为 **BLEU**
- **Summarization**（摘要）：CNN/DailyMail 类文章，评估指标为 **ROUGE-L F1**
- **Open-domain QA**（问答）：Natural Questions 数据子集，评估 **token-level F1**
- **RAG**（检索增强生成）：结合外部知识的回答生成，评估 **token F1**
- **MMLU**（多项选择题）：跨学科知识理解，评估 **multiple-choice accuracy**

每个 domain 分为：
- **Calibration set**（校准集）：20% prompts，用于离线搜索 skip configuration
- **Test set**（测试集）：80% prompts，用于最终评估

### 实验设置
| 项目 | 设置 |
|------|------|
| **Target Models** | `Vicuna-7B` (L=32), `Llama-2-13B` (L=40) |
| **Speculative Methods** | EAGLE（配合 Vicuna-7B）、SAMD Token-Recycle（配合 Llama-2-13B） |
| **Precision** | float16 |
| **Decoding** | Greedy (temperature=0), max_new_tokens=96 |
| **Hardware** | 8×NVIDIA RTX 4090 GPUs, dual-socket Xeon CPUs |
| **Framework** | PyTorch 2.11 + HuggingFace Transformers |

### 评估指标
| 指标 | 描述 |
|------|------|
| **Accept Length** | 每次 verification 步骤平均接受的 draft token 数量 |
| **Task Metric** | 各任务专用指标（BLEU, ROUGE-L, F1, Acc.） |
| **End-to-end Throughput** | 端到端吞吐量（tokens/s） |
| **Sparsity Ratio** | 被跳过的 target layer 比例 |
| **vs. Native** | 相对于原生 speculative decoding 的加速比 |

### 基线方法对比
- **Native**: 原始 speculative decoding 方法（EAGLE 或 SAMD），target 全深度运行（S=∅）
- **TSS**: 在相同 setup 下启用 target-side layer skipping
- **消融对比方法**：
  - **Accept-only**: 仅以 accept length 为目标选择层
  - **Metric-only**: 仅以下游任务指标为目标
  - **Random**: 随机跳过若干层
  - **GM-Skip**, **SLEB**: 已有的 layer-skipping 方法（基于 loss/perplexity）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 **Vicuna-7B + EAGLE** 上的结果：
| Domain | Accept Len. ↑ | Task Metric ↑ | Throughput (tok/s) ↑ | Speedup |
|--------|----------------|----------------|------------------------|---------|
| **Translation** | 2.70 → **4.53** (+67.8%) | 0.131 → **0.237** (+80.9%) | 75.6 → **127.3** | **1.68×** |
| **Summarization** | 3.93 → **4.01** | 0.266 → **0.270** | 89.4 → **95.5** | **1.07×** |
| **RAG** | 3.24 → **4.03** | 0.096 → **0.109** | 76.0 → **97.1** | **1.28×** |
| **QA** | 2.93 → **3.97** | 0.042 → **0.064** | 80.0 → **113.7** | **1.42×** |
| **MMLU** | 3.12 → **3.96** | 0.281 → **0.328** | 67.5 → **90.8** | **1.35×** |

> ✅ 平均提升：accept length ↑ ~20%，throughput ↑ **1.31×**

#### 在 **Llama-2-13B + SAMD** 上的结果：
| Domain | Accept Len. ↑ | Task Metric ↑ | Throughput (tok/s) ↑ | Speedup |
|--------|----------------|----------------|------------------------|---------|
| **Translation** | 2.69 → **2.89** | 0.208 → **0.212** | 38.3 → **49.6** | **1.29×** |
| **Summarization** | 3.11 → **3.19** | 0.249 → **0.251** | 45.8 → **53.2** | **1.16×** |
| **QA** | 2.53 → **2.74** | 0.120 → **0.129** | 33.2 → **43.3** | **1.30×** |
| **MMLU** | 3.96 → **4.08** | 0.359 → **0.422** | 58.6 → **70.8** | **1.21×** |

> ✅ 即使在更大模型上也稳定增益，且任务性能无损甚至提升

### 与替代策略的对比（Table 2）
| 方法 | Translation (Accept/Metric) | MMLU (Accept/Metric) | 结论 |
|------|----------------------------|-----------------------|------|
| **TSS (Ours)** | **4.528 / 0.237** | **3.955 / 0.328** | ✅ 全面领先 |
| Accept-only | 3.769 / 0.126 | 4.023 / 0.219 | ❌ 牺牲 task metric |
| Metric-only | 3.029 / 0.139 | 3.001 / 0.265 | ❌ 显著降低 accept |
| GM-Skip (Greedy) | 2.995 / 0.109 | 2.889 / 0.311 | ❌ 双输 |
| Random | 3.382 / 0.140 | 3.539 / 0.132 | ❌ 不可控 |

> 🔍 表明：**必须联合优化 accept 和 metric**，单一目标无法取得平衡。

### 消融实验结果
- **tolerance 敏感性分析**（Table 4）表明：
  - 设置 $ \epsilon_A = \epsilon_M = -5\% $（允许轻微下降）能获得最佳权衡
  - 更宽松容忍（如 -20%）虽提高 accept，但显著损害 task metric
- **search depth 影响**：breadth-first 搜索能探索非局部最优组合，避免 greedy 陷入局部陷阱
- **layer effect 非加性**：跳过多个 layer 的效果 ≠ 单个跳过效果之和，证明需组合搜索

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Full-depth target verification is not optimal**  
   在 domain-specific 场景下，目标模型的深层变换可能引发 **domain-conditioned overthinking**，反而破坏 draft-target 对齐并降低任务表现。

2. ✅ **Selective layer skipping improves both efficiency and quality**  
   跳过特定 target layers 不仅减少了计算量，还能：
   - 提高 **accept length**
   - 提升或维持 **task metric**
   - 实现高达 **1.68× 的端到端加速**

3. ✅ **Joint optimization is essential**  
   必须同时考虑 **acceptance** 和 **task metric**，否则会牺牲一方换取另一方，整体收益受损。

4. ✅ **One model, multiple sparse paths**  
   TSS 实现了“一个完整模型 + 多条稀疏路径”的灵活部署模式，极大提升了部署效率与适应性。

### 方法的局限性
- 🚫 **依赖 domain 分类准确性**：若 runtime domain 判断错误，可能导致应用错误 skip policy
- 🚫 **离线搜索成本较高**：虽然不影响线上推理，但搜索过程需遍历大量 layer 组合（~数百次 eval）
- 🚫 **效果因 speculative method 而异**：在 EAGLE 上增益更大，在 SAMD 上相对温和，说明与 draft-target 协同机制有关
- 🚫 **不能修复知识缺失**：TSS 仅优化已有 draft 的验证路径，无法引入新知识（见 RAG 案例）

### 未来工作方向
- 🔮 **Dynamic in-context domain detection**：结合 prompt 内容实时识别 domain，减少对外部分类器依赖
- 🔮 **Online adaptation of skip policy**：根据历史 accept pattern 动态调整 skip 策略
- 🔮 **Extension to encoder-decoder models**：应用于如 T5、BART 等架构
- 🔮 **Integration with draft training**：联合优化 draft model 与 target skip policy，进一步提升协同效率

---

> 💡 **一句话总结**：  
> TSS 首次揭示了在 speculative decoding 中，“目标模型瘦身”比“草稿模型变强”更具潜力，通过智能跳过冗余层，实现了 **更快、更准、更省** 的推理新范式。

</details>

---

### 7. [Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding](https://arxiv.org/abs/2609.26638)

**Authors**: Dohyun Kim, Sungjun Han, Hyungguk Kim, Yusik Kim, Jamin Shin, Paul Hongsuck Seo, Hongjoon Ahn  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.26638v1  

#### Abstract
Autoregressive OCR vision-language models accurately convert document images into text and structured markup, but require one sequential decoding step per output token, limiting inference speed. Unlike open-ended text generation, OCR outputs are strongly grounded in the input image, making diffusion...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding*

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Autoregressive (AR) OCR vision-language models** 虽然在文档理解任务中表现准确，但其逐token生成的特性导致推理速度慢、延迟高，限制了大规模部署时的吞吐量。尽管已有研究尝试通过 **multi-token prediction** 或 **speculative decoding** 加速，但在开放文本生成之外的结构化OCR任务中，直接并行生成多个token容易因缺乏上下文依赖而导致输出不一致（如重复、遗漏、结构错误）。

### 提出的新方法
本文提出了 **GRAVITYOCR**，一种结合 **参数共享的 AR-diffusion 架构** 和 **自推测解码（self-speculative decoding）** 的新型框架，用于加速文档OCR。

- **核心思想**：将生成过程分为两个路径：
  - **Block Diffusion Path**：并行生成一个完整的token块（block），作为“草稿”（draft）。
  - **Causal AR Path**：对草稿进行逐token验证，仅提交与自身预测一致的最长前缀，并生成下一个token。
- **关键机制**：通过 **共享模型参数** 实现两个路径，无需额外的草稿网络（separate drafter network）。

### 相比现有方法的优势
- **高效且准确**：在保持接近原始AR模型精度的同时，显著提升解码速度。
- **无需独立草稿器**：与需要额外训练扩散草稿器的方法（如DFlash）不同，GRAVITYOCR通过参数共享实现一体化设计。
- **支持强化学习优化**：利用 **GRPO** 在因果AR路径上进行序列级和结构级奖励优化，避免复杂的扩散轨迹似然估计，同时更新共享的草稿器参数。
- **端到端加速**：不仅在token生成阶段提速，在完整页面处理流程中也实现了可观的速度提升。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主基准测试集**：
  - **OmniDocBench v1.6**：包含1,651页文档，评估文档解析质量，涵盖文本、表格、公式和阅读顺序。
- **辅助基准测试集**：
  - **PubTabNet**：用于评估表格识别性能（9,115张表）。
  - **UniMER-Test**：用于数学表达式识别。
- **训练数据**：
  - 来自多个公开数据集的 **12.3M 区域级样本**，包括 DocGenome、Docmatix、PubTables-1M、FinTabNet、SynthTabNet、RVL-CDIP、DocLayNet、UniMER 等。
  - 数据经过清洗和子采样，形成约 **10.8M 训练样本**，确保文本、表格、公式流的比例为 60/20/20。

### 实验设置和评估指标
- **模型初始化**：基于 **GLM-OCR** 检查点进行微调。
- **训练方式**：
  - 联合训练 AR 和 block diffusion 目标，损失函数为 $ L = L_{AR} + \alpha L_{diff} $，$\alpha=1$。
  - 后续使用 **GRPO** 在因果AR路径上进行强化学习微调。
- **评估指标**：
  - **质量指标**：
    - **Overall Score**（OmniDocBench综合得分）
    - **Text**：归一化编辑距离（NED）
    - **Table**：Tree-Edit-Distance Similarity (TEDS)
    - **Formula**：Character Detection Matching (CDM)
    - **Order**：阅读顺序准确性
  - **效率指标**：
    - **Tokens Per Forward (TPF)**：每轮前向传播提交的平均token数。
    - **tok/s**：每秒生成token数。
    - **pages/s**：每秒处理页面数（端到端）。
    - **Speedup**：相对于AR解码的速度提升倍数。

### 基线方法对比
- **内部对比**：
  - GLM-OCR (AR)：原始自回归模型。
  - GLM-OCR (MTP)：内置多token预测分支。
  - Direct Block-Diffusion：无AR验证的直接扩散解码。
- **外部系统对比**：
  - MinerU2.5 / Pro / Diffusion
  - PaddleOCR-VL-1.5
  - dots.ocr
  - DeepSeek-OCR-2
  - HunyuanOCR-1.5 (+ DFlash)

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 数值 |
|------|------|
| **OmniDocBench Overall Score** | **95.16**（接近原始GLM-OCR的95.48） |
| **平均 TPF (Tokens Per Forward)** | **9.7** |
| **decode-only 速度提升** | **3.94×**（region crops） |
| **端到端页面处理速度提升** | **1.32×** |
| **页面吞吐量 (pages/s)** | **0.730**（SGLang部署下） |

### 与基线方法的对比结果
- **质量方面**：
  - GRAVITYOCR 的 **Overall Score (95.16)** 显著优于大多数对比系统（如MinerU-Diffusion: 89.87），仅次于 MinerU2.5-Pro (95.57) 和 HunyuanOCR-1.5 (95.52)。
  - 在 **PubTabNet** 上，TEDS 达到 **0.871**（原GLM-OCR为0.803），结构准确率（TEDS-struct）达 **0.916**。
  - 在 **UniMER** 上，CDM 达 **0.962**，几乎持平原模型（0.963）。
- **效率方面**：
  - **TPF = 9.7**，远高于标准AR解码（TPF=1.0）和GLM-OCR MTP（TPF=3.71）。
  - **decode-only throughput** 达 **3,057 tok/s**（SGLang），是AR模式（777 tok/s）的 **3.94×**。
  - **end-to-end throughput** 达 **844 tok/s**，是AR模式（486 tok/s）的 **1.74×**。
  - **页面处理速度** 达 **0.730 pages/s**，超过 HunyuanOCR-1.5 + DFlash (0.579 pages/s)。

### 消融实验结果
- **AR Loss 的作用（Table 5）**：
  - 移除AR监督后，Overall Score 从 **95.02** 下降到 **93.64**，表明AR路径对于保持验证准确性至关重要。
- **GRPO 强化学习效果（Table 6）**：
  - 应用GRPO后，Overall Score 从 **94.92** 提升至 **95.16**（+0.24），而 TPF 几乎不变（9.61 → 9.68），说明RL提升了质量但未牺牲效率。
- **不同内容类型的加速效果（Table 4）**：
  - **表格区域**：接受更多草稿token（25.0/32），速度提升最高（**3.36×**）。
  - **公式区域**：提升 **1.82×**。
  - **文本区域**：提升 **1.54×**。
  - 表明结构化输出更受益于该方法。

---

## 4. 关键结论和发现

### 主要发现
1. **自推测解码有效平衡了速度与精度**：通过分离“并行草稿”与“因果验证”，GRAVITYOCR 成功实现了高速度下的高质量OCR输出。
2. **参数共享架构可行且高效**：无需独立草稿网络即可实现高性能的 self-speculative decoding，降低了模型复杂性和训练成本。
3. **GRPO 可有效优化OCR任务目标**：在因果AR路径上应用序列级和结构级奖励，能显著提升识别质量，同时不影响草稿效率。
4. **结构化内容更易被并行化**：表格和公式等具有强语法约束的内容在接受草稿方面表现更好，加速潜力更大。

### 方法的局限性
- **依赖高质量预训练AR模型**：方法基于已有的AR模型进行adaptation，若初始AR模型性能差，则难以提升。
- **长尾场景可能仍存在不一致**：虽然AR验证减少了错误，但在极端复杂布局或低质量图像上，草稿与验证的分歧可能导致回退到低效模式。
- **批处理优势减弱**：随着batch size增大，自推测的优势逐渐缩小（batch 64时仍快1.13×），说明其最大收益体现在低并发场景。

### 未来工作方向
- 探索更智能的草稿调度策略（如动态调整block size或confidence threshold）。
- 将该框架扩展至其他结构化生成任务，如代码生成、XML/JSON生成等。
- 结合视觉先验进一步提升草稿质量，减少验证失败率。
- 研究在边缘设备上的轻量化部署方案。

> **总结图示**：如图1所示，GRAVITYOCR 是目前 **准确率-速度权衡曲线** 上最领先的系统，兼具顶级精度与最快推理速度。

</details>

---

### 8. [CompKV: Compensation-Aware KV Selection for Long-Context LLM Inference](https://arxiv.org/abs/2609.26300)

**Authors**: Zhen Huang, Ruizhe Yao, Danyi Liu, Xinrui Chen, Shuwei Li, Siru Zhong, Zijian Cao, Yushan Lai, Mingming Guo, Weijie Zheng, Haohuan Fu  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.26300v1  

#### Abstract
Despite their strong performance, large language models (LLMs) are bottlenecked by KV cache memory traffic during long-context inference. Sparse attention is widely used to accelerate LLM inference by computing exact attention over a selected subset of tokens. To recover the contribution of tokens e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《CompKV: Compensation-Aware KV Selection for Long-Context LLM Inference》核心总结**

---

## **1. 主要贡献和创新点**

### **解决的问题**
在长上下文场景下，大语言模型（LLM）推理过程中的 **KV cache** 内存访问成为主要瓶颈。现有的稀疏注意力（sparse attention）方法通常采用两阶段解耦设计：
- **先选择** 高注意力质量（attention mass）的 token 或 block 进行精确计算；
- **再对未选部分进行粗粒度补偿**（如均值摘要）。

这种设计忽略了“**选择”与“补偿”之间的交互关系**：某些高 attention mass 的 block 可能容易被补偿重建，而一些低质量但内部 logit 变化大的 block 若被忽略，则会产生更大的误差。因此，单纯基于 attention mass 的选择策略并非最优。

---

### **提出的新方法与新思路**
本文提出了 **CompKV**，是首个 **compensation-aware**（补偿感知）的 KV 选择框架，其核心思想是：
> **优先选择那些如果被省略将导致最大补偿误差的 KV blocks**。

#### **理论分析**
- 在 **Mean compensation**（用 block 均值代替内部所有 logit）机制下，作者推导出被省略 block 的残差（residual）由两个因素共同决定：
  1. **Block 的 attention mass**（贡献大小）
  2. **Block 内部的 logit variation**（变化程度）
- 因此，最优选择应综合考虑这两个因素，而非仅看 attention mass。

#### **CompKV 方法设计**
- 为每个 KV block 维护紧凑的统计量：**mean key**, **grouped variance**, **mean value**。
- 构造一个 **query-dependent 的选择分数（score）**：
  $$
  S_b = \sum_{g=1}^G \hat{p}_{g,b} \cdot \hat{\sigma}^2_{g,b}
  $$
  其中 $\hat{p}_{g,b}$ 是估计的 attention mass，$\hat{\sigma}^2_{g,b}$ 是估计的 logit 方差。
- 选择得分最高的 K 个 blocks 进行精确 attention 计算，其余通过均值摘要进行补偿。
- 最终输出联合归一化，保证一致性。

---

### **相比现有方法的优势**
| 对比维度 | 现有方法（如 Quest, InfLLM） | CompKV |
|--------|----------------------------|-------|
| **选择依据** | 仅基于 attention mass 或上界估计 | 考虑补偿误差，结合 mass 和 variation |
| **与补偿机制的关系** | 解耦：选择后补偿 | 耦合：选择时已考虑补偿效果 |
| **是否训练** | 无需训练 | 无需训练（training-free） |
| **效率优化** | 一般 | 提出异步 CPU-offload 实现，重叠数据传输与计算 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **RULER**: 合成的长上下文基准，包含检索、追踪、聚合、问答等任务，用于控制变量测试。
- **LongBench-Pro**: 更真实、复杂的双语长文本评测集，涵盖 11 类任务（如证据问答、代码差异分析、对话记忆等），测试泛化能力。

---

### **实验设置与评估指标**

#### **模型**
- **Llama-3.1-8B-Instruct**
- **Qwen3-8B**
- **Qwen3-32B**

覆盖不同规模与架构，验证通用性。

#### **评估指标**
- **准确性**：任务平均得分（AVG），各子任务分数（如 MK3, MV 等）。
- **效率**：单层 attention 步骤的 **平均延迟（latency）**，衡量实际推理速度。

#### **上下文长度与预算**
- 上下文长度：**32K 到 128K**
- 每步精确读取 token 数（budget）：**512 ~ 2048**

#### **硬件环境**
- GPU: NVIDIA H100 80GB
- 批大小：1
- KV cache 存于 **pinned CPU 内存**，模拟真实部署场景。

---

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Full Attention** | 完整 KV cache 读取，作为上限参考 |
| **Quest** | 基于 compact key 统计的 query-aware block selection |
| **InfLLM** | 基于累积 attention 的训练免费稀疏方法 |
| **Quest+RESA** | Quest + 尾部补偿（residual estimation）插件 |

> 所有方法均使用相同 block size（16）、mandatory blocks（首块 + 最近两块）和实现细节以确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **准确性结果（RULER & LongBench-Pro）**
| 模型 | 数据集 | CompKV AVG | 最佳基线 AVG | 提升 |
|------|--------|------------|--------------|------|
| Llama-3.1-8B | RULER (32K) | **83.2%** | 77.0% (Quest+RESA) | **+6.2pp** |
| Qwen3-8B | RULER (32K) | **86.0%** | 75.8% (Quest+RESA) | **+10.2pp** |
| Qwen3-32B | RULER (32K) | **91.0%** | 75.7% (Quest) | **+15.3pp** |

> 在 **LongBench-Pro** 上同样取得最高分，尤其在困难任务（如 MK3）上优势显著。

#### ✅ **效率结果（CPU-offload 单层延迟）**
| 设置 | CompKV 延迟 | Full 延迟 | **加速比** |
|------|-----------|----------|----------|
| 128K ctx, 512 budget | 1.456 ms | 9.966 ms | **6.85×** |
| 64K ctx, 512 budget | 0.840 ms | 4.995 ms | **5.95×** |
| 32K ctx, 512 budget | 0.527 ms | 2.536 ms | **4.81×** |

> CompKV 在所有 9 种配置下均达到最低延迟，**最高实现 6.85× 自注意力加速**。

---

### **消融实验结果**

#### （1）**组件消融（Ablation Study）**
在 RULER 上对 CompKV 各组件进行消融（r=4）：

| 变体 | Llama @512 | Qwen @512 | 相比完整版下降 |
|------|------------|-----------|----------------|
| CompKV（完整） | 83.2% | 86.0% | — |
| 移除 outer variance 因子 | 81.8% | 83.5% | ↓1.4~2.5pp |
| 移除 second-order mass correction | 81.2% | 83.8% | ↓2.0~2.2pp |
| 移除 Mean compensation | 78.6% | 80.9% | ↓4.6~5.1pp |

> 结论：**三个组件均有效，且协同作用明显**。

#### （2）**方差分组粒度（r）的影响**
- 使用更多 variance groups（r=128）可进一步提升准确率。
- 但 r=4 已能捕获 **81.5% 的残差信号**（Spearman 相关性达 0.86），性价比高。
- 说明 **compact grouped statistics 设计合理**，在极小元数据开销下保持高性能。

---

## **4. 关键结论和发现**

### **主要发现**
1. **选择与补偿必须联合建模**：仅靠 attention mass 无法反映补偿难度，**logit variation 是关键调节因子**。
2. **CompKV 显著优于现有稀疏方法**：在多个模型和基准上均取得最佳 accuracy，同时大幅降低延迟。
3. **理论指导实践有效**：从 KL 散度出发推导的选择准则，在实际中可通过紧凑统计量高效逼近。
4. **异步实现极大提升吞吐**：通过 CPU-offload 与多流并行，掩盖数据传输开销，释放硬件潜力。

---

### **方法的局限性**
- **依赖 Mean compensation**：当前理论分析基于 block-mean 替代，虽可扩展，但其他补偿方式（如低秩重建）需重新建模。
- **统计量压缩仍有信息损失**：尽管 grouped variance 表现良好，但仍无法完全替代原始 key 读取。
- **适用于 decoding 阶段**：主要针对自回归生成优化，prefill 阶段未重点优化。

---

### **未来工作方向**
- 探索更复杂的补偿机制下的联合选择策略（如 ResKV、RESA）。
- 将 compensation-aware 思想推广至 **prefill 阶段的 block-sparse attention**。
- 动态调整 K（budget）以适应不同 query 复杂度。
- 结合量化（quantization）与稀疏化，进一步压缩 KV cache 开销。

---

> 💡 **一句话总结**：  
> **CompKV 首次实现了“补偿感知”的 KV 选择，通过理论驱动的设计，在不增加训练成本的前提下，兼顾精度与极致推理效率，为长上下文 LLM 推理提供了新的范式。**

</details>

---

### 9. [LADDER: Graph-Guided Diffusion Language Models for Efficient Multi-Hop Reasoning](https://arxiv.org/abs/2609.24346)

**Authors**: Senlei Zhang, Linhao Luo, Qian-Wen Zhang, Siyu An, Junnan Dong, Shuhao Zhang, Xing Sun  
**Category**: cs.AI  
**Published**: 2026-09-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.24346v1  

#### Abstract
Graph Retrieval-Augmented Generation (GraphRAG) has remarkably enhanced large language models on complex reasoning by leveraging structured entity topologies. However, existing frameworks heavily rely on standard autoregressive language models where the nature of inherent sequential generation sever...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LADDER: Graph-Guided Diffusion Language Models for Efficient Multi-Hop Reasoning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **Graph Retrieval-Augmented Generation (GraphRAG)** 虽然通过结构化实体拓扑增强了大语言模型（LLMs）在复杂多跳推理任务中的表现，但其依赖标准的**自回归生成（autoregressive generation）**机制，导致检索与生成过程严格同步，严重限制了推理效率。

此外，**Diffusion Language Models (DLMs)** 尽管支持并行解码，但在多跳推理场景下仍面临两大挑战：
1. **部分去噪的草稿状态高度动态且不确定**，难以进行有效的图检索触发。
2. **原始去噪状态噪声大、不稳定**，直接进行同步图检索和多跳聚合计算开销巨大，且易引发错误传播。

---

### **提出的新方法与创新思路**
论文提出了 **LADDER**（**L**anguage model **A**ccelerated by **D**ynamic **D**ecoding with **E**vent-**R**egulated retrieval），一个将 **DLM** 与 **GraphRAG** 结合的新型框架，核心创新如下：

#### **(1) Event-Driven Self-Clocking Retrieval（事件驱动的自时钟检索）**
- **关键洞察**：88%的目标实体在去噪过程中**早于其最终提交（commitment）前 5.7–9.6 步**就已出现。
- **机制设计**：仅当草稿中可链接的实体集合（graph-linkable entities）**扩展时**才触发图检索，形成一种**异步自时钟策略**。
- **优势**：无需学习门控（learned gates）或启发式阈值，避免了每步都检索带来的冗余开销。

#### **(2) Incomplete-Query Graph Propagation（不完整查询图传播模块）**
- 针对中间阶段生成的**可修订、不完整的实体查询**，设计了一个基于**Graph Foundation Model (GFM)** 的传播模块。
- 采用**发现（discovery）与保留（retention）双重目标**进行微调：
  - **Discovery Objective**：检索尚未出现在记忆中的黄金证据。
  - **Retention Objective**：保留早期事件中已检索到的有效证据。
- 引入 **Anchor Term** 保证跨数据集的零样本迁移能力。

#### **(3) Graph-Guided Parallel Decoding Paradigm**
- 将图检索从“顺序推理中的同步操作”转变为“DLM 并行去噪过程中的动态引导机制”，实现**推理与检索的解耦与加速**。

---

### **相比现有方法的优势**
| 维度 | LADDER | 传统 GraphRAG / IRCoT 类方法 |
|------|--------|-----------------------------|
| **推理模式** | 并行去噪（DLM） | 自回归生成（sequential） |
| **检索频率** | 仅在实体变化时触发（约15%步骤） | 每轮推理后必检索（同步路径上） |
| **延迟** | 极低（减少至 4.1×） | 高（受制于串行流程） |
| **准确性** | 更高（EM 提升 5.6%） | 受限于推理链长度 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **HotpotQA**：多跳问答基准，强调解释性和多样性。
- **2WikiMultihopQA**：基于维基百科的多跳 QA 数据集。
- **MuSiQue**：构造更长推理链（≥4跳）的问题，用于测试零样本迁移能力。

> 所有实验均使用公开评测集（各1000个问题），构建对应语料库的 **entity-document graph**。

---

### **实验设置与评估指标**

#### **模型配置**
- **生成器**：Dream-7B（DLM backbone）
- **检索器**：GFM-RAG（Graph Foundation Model）
- **去噪策略**：置信度采样（confidence-threshold sampler, T=0.85）
- **最大去噪步数**：100步
- **GPU环境**：单张 NVIDIA H20

#### **评估指标**
| 指标 | 含义 |
|------|------|
| **Exact Match (EM)** | 完全匹配率 |
| **F1 Score** | 答案词重叠的F1分数 |
| **Recall@K (R@K)** | 检索召回率（K=2,5,7） |
| **Latency (Time)** | 单题端到端耗时（秒） |
| **Steps to Convergence** | 收敛所需去噪步数 |

---

### **基线方法对比**
| 基线方法 | 类型 | 控制器 |
|--------|------|--------|
| RAPTOR, GraphRAG, LightRAG, HippoRAG 2, Youtu-GraphRAG | 自回归 + 图增强 | IRCoT-style 多轮控制器（最多5轮） |
| GFM-RAG (IRCoT) | DLM + 图增强 | 同样多轮，但无事件触发机制 |
| LADDER（本文） | DLM + 图引导并行解码 | 单一去噪轨迹内事件驱动刷新 |

> 所有生成器均经过相同监督微调（SFT），确保接口一致，排除能力差异干扰。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 2 汇总）**

| 方法 | Avg. EM (%) | Avg. F1 (%) | Avg. Latency (s) |
|------|------------|-------------|------------------|
| GFM-RAG (IRCoT) | 39.6 | 47.4 | 11.30 |
| **LADDER (Ours)** | **45.2** | **54.6** | **2.79** |

> ✅ **平均 EM 提升 5.6%，F1 提升 7.2%，延迟降低 4.1×**

#### **分数据集表现**
| 方法 | HotpotQA EM | 2WikiMultihopQA EM | MuSiQue EM |
|------|--------------|--------------------|------------|
| Youtu-GraphRAG | 44.6 | 63.5 | 18.2 |
| **LADDER** | **49.4** | **67.4** | **18.8** |

> 在所有三个数据集上均达到最优性能，尤其在 HotpotQA 上提升显著（+4.8 EM）。

---

### **与基线方法的对比结果**
- 相比最强自回归基线 **Youtu-GraphRAG**：
  - EM 提高 **3.1 分**，同时延迟降至其 **1/4 以下**。
- 相比 DLM 版本的 **GFM-RAG (IRCoT)**：
  - 减少 **8.51s** 延迟（从 11.30s → 2.79s），提升 **5.6 EM**。
- **检索效率更高**：仅在 15% 的去噪步骤中触发真实 GNN 前向传播（见 Table 8）。

---

### **消融实验结果**

#### **(1) 刷新策略对比（Table 3 & 8）**
| 策略 | Latency (s) | R@7 | EM |
|------|-------------|-----|----|
| Retrieve-once | 2.55 | 55.6 | 37.9 |
| Every 5 steps | 2.98 | 62.9 | 42.6 |
| Per-step | 4.38 | 65.8 | 44.5 |
| **LADDER (event-driven)** | **2.79** | **67.1** | **45.2** |

> ✅ LADDER 在相近延迟下实现了最高准确率，证明事件驱动优于固定周期或逐帧检索。

#### **(2) 训练目标消融（Table 4 & 9）**
| 目标组合 | R@7 | Rdisc (发现率) | EM |
|---------|-----|---------------|----|
| Lretrieve only | 62.0 | 39.7 | 41.9 |
| + Llookahead | 65.4 | 49.8 | 44.2 |
| + Lretain | 65.4 | 50.3 | 44.2 |
| **+ Lanchor (完整 LADDER)** | **67.1** | **52.1** | **45.2** |

> ✅ `Llookahead` 显著提升新证据发现能力；`Lanchor` 稳定迁移表现，是最终性能跃升的关键。

#### **(3) 种子策略（cumulative vs new-only）**
- **累计种子（cumulative seeding）** 效果显著优于仅用新增实体（new-seed only）：
  - 在 2WikiMultihopQA 上带来 **+4.2 R@7 和 +3.5 EM** 提升。
- 表明保留历史确认实体作为图起点具有持续价值。

---

## **4. 关键结论和发现**

### **主要发现**
1. **实体早现现象普遍存在**：88% 的目标实体在完全提交前即已在草稿中出现（平均提前 5.7–9.6 步），为提前检索提供了理论基础。
2. **事件驱动检索显著提升效率**：仅在实体集合变化时触发检索，使实际 GNN 前向传播次数减少至约 **15% 的步骤**，大幅降低延迟。
3. **高质量证据加速收敛**：图检索相比随机文本减少 **37% 的去噪步数**，而 oracle 证据可减少 **69%**（2WikiMultihopQA）。
4. **LADDER 实现质量与速度双赢**：不仅提升 EM 和 F1，还实现 **4.1× 的端到端延迟下降**。

---

### **方法的局限性**
1. **闭集实体匹配器限制覆盖范围**：
   - 当前 matcher 仅支持规范命名实体，无法处理同义改写、代词或未登录实体。
   - 影响对复杂表达或非标准表述的理解能力。
2. **长链推理仍有瓶颈**：
   - 在 MuSiQue 上性能增益较小（仅 +0.4 F1），因长推理链导致每次事件获取的黄金证据比例较低。
3. **语言与模型限制**：
   - 实验局限于英文 QA 和 Dream-7B 模型，尚未验证跨语言或多模态扩展性。

---

### **未来工作方向**
1. **开放词汇实体链接**：引入更强的实体识别与链接模块，支持代词解析、指代消解和 paraphrase 匹配。
2. **动态预算调整**：根据推理深度自适应调整检索数量 $K$，平衡精度与延迟。
3. **跨模态与多语言扩展**：将 LADDER 框架推广至图像-文本多模态 QA 或非英语场景。
4. **更细粒度的缓存机制**：探索基于语义相似性的检索缓存，进一步减少冗余计算。

---

> 🔚 **总结**：  
> **LADDER 成功将 GraphRAG 的结构优势与 DLM 的并行效率结合，通过“事件驱动”的异步检索机制，在几乎不增加计算成本的前提下，实现了多跳推理任务中准确率与速度的双重突破。它标志着从“顺序推理+同步检索”向“并行解码+动态引导”的范式转变，为高效复杂推理系统的设计提供了新范式。**

</details>

---

### 10. [Optimizing Denoising Trajectories in dLLMs: A Lightweight Evolutionary Heuristic Approach](https://arxiv.org/abs/2609.26052)

**Authors**: Zijian Zhao, Dian Jin, Xialiang Tong, Sen Li, Mingxuan Yuan  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.26052v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have recently emerged as a promising alternative to conventional Auto-Regressive (AR) Large Language Models (LLMs). By leveraging bidirectional attention and parallel decoding, dLLMs enable more efficient generation. However, they require a carefully designed ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimizing Denoising Trajectories in dLLMs: A Lightweight Evolutionary Heuristic Approach

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文聚焦于 **Diffusion Large Language Models (dLLMs)** 在推理阶段的 **denoising scheduler** 设计问题。dLLMs 虽然通过并行解码提升了生成效率，但在推理时需要一个精心设计的去噪调度策略来决定哪些 token 应优先恢复。然而，现有的基于置信度（confidence-based）的启发式调度器存在两个关键失败模式：

- **EOS Overflow**：过早地在序列末尾生成过多 `[EOS]` token，导致输出被截断。
- **Proximal Bias（邻近偏差）**：一旦某个 token 被解码，其邻居位置会因注意力机制而获得虚高的置信度，从而引发连锁错误。

这些现象源于 Transformer 注意力机制中对无效 token（如 `[MASK]`, `[EOS]`）分配了过高权重，误导了调度决策。

### 提出了什么新方法或新思路
作者提出了一种 **轻量级进化启发式调度器（Lightweight Evolutionary Heuristic Scheduler）**，其核心思想是：

- **多启发式融合**：结合多种启发式特征（top-1 probability、margin probability、valid attention scores 等），避免依赖单一信号。
- **上下文感知建模**：引入 **contextual mean-field embedding** 来捕捉全局序列状态，实现 context-aware 的动态调度。
- **零阶优化训练**：采用 **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** 进行黑箱优化，绕过不可导的 Top-K 操作难题，仅需最终任务准确率作为 fitness signal。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **参数效率** | 仅需 **393 个可训练参数**，远低于现有神经调度器（如 EDM 含 5M 参数），是目前已知最参数高效的 neural scheduler。 |
| **性能表现** | 在多个推理与规划任务上显著优于各类基线，尤其在低步数预算（low denoising budget）下优势明显。 |
| **通用性与鲁棒性** | 架构简洁，适用于不同 backbone（LLaDA / Dream），且具备一定跨数据集泛化能力。 |
| **无需强化学习** | 避免复杂的 RL 训练流程（如 PPO、GRPO），降低训练成本与调参难度。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在四个具有挑战性的推理与规划基准上进行：
- **GSM8K**：小学数学应用题，测试多步算术推理能力。
- **Math**（Hendrycks et al., 2021）：竞赛级别数学题，涵盖代数、几何等。
- **Countdown**：数字组合规划任务，要求用基本运算达成目标值。
- **StrategyQA**：开放域逻辑问答，需隐式多跳推理。

> 注：为高效评估，Math 数据集使用 **Math-500 子集**作为测试集。

### 实验设置和评估指标
- **模型平台**：
  - `LLaDA-8B-Instruct`（Nie et al., 2026）
  - `Dream-7B-Instruct`（Ye et al., 2025b）
- **推理配置**：
  - 生成长度 $ L \in \{128, 256\} $
  - 去噪步数预算 $ T \in \{16, 32\} $（即最多执行 T 次 denoising 步骤）
- **评估指标**：**Accuracy**（答案正确率）
- **训练方式**：
  - 使用 **CMA-ES** 优化调度器参数
  - 批大小 100，最大迭代 10 代，初始步长 0.2
  - Fitness 函数为 mini-batch 上的平均 accuracy

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **启发式调度器** | Top-1 Prob, Entropy, Prob Margin |
| **Block-AR 方法** | Top-1 Prob (B=32), Entropy (B=32), etc. |
| **近期先进方法** |  
| &nbsp;&nbsp;– EDM (5M params) | 利用离线数据训练初始轨迹评分器 |
| &nbsp;&nbsp;– CCD | 使用历史平均置信度提升稳定性 |
| &nbsp;&nbsp;– AGDO Scheduler | 结合 attention 与 top-1 probability 进行候选筛选 |
| &nbsp;&nbsp;– Suffix Anchor | 添加后缀提示缓解 EOS Overflow |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 LLaDA-8B-Instruct 为例）

#### 表格摘要（部分关键项）

| Scheduler | GSM8K (T=32, L=128) | Math (T=32, L=256) | Countdown (T=32, L=256) | StrategyQA (T=32, L=256) |
|----------|---------------------|--------------------|-------------------------|----------------------------|
| Top-1 Prob | 54.0 | 18.6 | 20.7 | 56.8 |
| Block-AR (B=32) | 60.5 | 17.4 | 9.0 | 42.4 |
| EDM (5M) | 56.8 | 22.8 | — | — |
| CCD | 52.5 | 17.4 | 17.6 | 57.2 |
| Suffix Anchor | 56.7 | 19.2 | 45.7 | 59.1 |
| **Proposed (Evolution)** | **67.6** | **27.8** | **40.6** | **66.5** |

> ✅ 在所有任务中均取得 **SOTA 性能**，尤其在 **Math 和 Countdown** 上大幅领先。

### 与基线方法的对比结果
- 在 **Math (T=16, L=256)** 最难设置下，本文方法达到 **25.0%** 准确率，显著高于最佳基线（17.2%）。
- 在 **Countdown** 上，多数基线在 $T=16$ 下表现极差（<10%），而本文方法仍保持 **35.2%**。
- 即使在高预算下（如 T=32），也持续优于 Block-AR 和其他神经调度器，说明其调度路径更优。

### 消融实验结果（Ablation Study on Math Dataset）

| Scheduler | T=16 (L=128) | T=32 (L=128) | T=16 (L=256) | T=32 (L=256) |
|----------|--------------|--------------|--------------|--------------|
| Full Model | 21.4 | 27.2 | 25.0 | 27.8 |
| w/o MF (mean-field) | 21.0 | 25.8 | 19.6 | 24.0 |
| w/o skip (linear bypass) | 21.4 | 24.0 | 20.0 | 23.4 |

> 🔍 发现：
> - 移除 **mean-field embedding** 对长序列影响更大 → 验证了其在长程上下文建模中的作用。
> - 移除 **skip connection** 导致训练初期收敛变慢 → 说明线性通路有助于零阶优化稳定启动。

### 泛化能力测试
- **跨数据集迁移**：在 GSM8K 上训练的调度器迁移到 Math 上仍表现良好（甚至略超本地训练模型），表明部分启发式具有通用性。
- **跨模型迁移**：在 Dream-7B 上微调后依然有效，显示架构兼容性强。
- **跨配置迁移**：对未见的 $L$, $T$ 设置有较强适应性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **传统 confidence-based 调度器存在根本缺陷**：其失败源于 Transformer 注意力机制对 `[MASK]` 和 `[EOS]` 的过度关注，导致虚假高置信度。
2. **valid attention score 是重要补充信号**：尽管单独使用效果不稳定，但它能提供与置信度互补的信息。
3. **最优去噪路径高度 context-dependent**：没有一种固定启发式适用于所有场景，必须动态融合多信号。
4. **轻量级结构 + 进化优化 可实现高性能调度**：仅用 393 参数即可超越含数百万参数的 RL 训练模型，证明“小而精”路线可行。

### 方法的局限性
- **领域特异性**：调度器在跨数据集迁移时仍有性能下降，说明其学到的部分先验具有 domain bias。
- **依赖手工特征工程**：当前输入特征为人工选择（如 attention 层选取 middle/lower/upper），未来可探索自动特征提取。
- **不支持自适应步长**：目前假设固定 denoising budget，无法像某些方法那样动态调整总步数。

### 未来工作方向
- **混合数据集联合训练**：提升调度器的通用性和鲁棒性。
- **端到端可微调度架构探索**：结合重参数化技巧解决 Top-K 不可导问题。
- **引入更多语义特征**：如句法边界、语义一致性得分等辅助信号。
- **扩展至多模态 dLLMs**：应用于图文生成等场景下的跨模态去噪调度。

---

> 📌 **一句话总结**：  
> 本论文揭示了 dLLMs 中启发式调度器的失败机理，并提出一种仅含 **393 参数**的 **进化式多启发融合调度器**，在多个复杂推理任务上实现了 **SOTA 性能**，为高效、轻量、可优化的 denoising trajectory 设计提供了新范式。

</details>

---

### 11. [FuncCode: Compressing Kolmogorov--Arnold Networks in Function Space with Hardware-Aware Quantization](https://arxiv.org/abs/2609.26067)

**Authors**: Kazi Ahmed Asif Fuad, Lizhong Chen  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.26067v1  

#### Abstract
Kolmogorov--Arnold Networks (KANs) replace scalar edge weights with learnable univariate functions, increasing flexibility but also parameter memory because each edge stores multiple coefficients, often together with a separate base branch. We introduce FuncCode, a basis-agnostic compression approac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FuncCode: Compressing Kolmogorov--Arnold Networks in Function Space with Hardware-Aware Quantization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Kolmogorov-Arnold Networks (KANs) 通过将传统的标量权重替换为可学习的一元函数（如样条、多项式或RBF），增强了模型的表达能力。然而，这种灵活性带来了显著的参数开销——每个边（edge）需要存储多个系数以及一个独立的 base 参数，导致**参数内存占用高**，限制了其在资源受限硬件上的部署。

现有压缩方法（如量化、向量量化）大多直接作用于**系数空间**（coefficient space），忽略了两个关键事实：
1. 不同的系数向量可能表示相似的函数；
2. KAN 边由 **basis branch** 和 **base branch** 组成，二者功能不同，应允许独立共享。

### **提出的新方法：FuncCode**
本文提出了 **FuncCode**，一种**基于函数空间**（function space）且**分支感知**（branch-aware）的压缩框架，核心思想如下：

- **Basis-agnostic 函数签名**：对每条边函数在固定输入域上采样，生成其“函数响应”（function signature），作为该边的函数级表征，不依赖具体基函数形式（spline/RBF/polynomial）。
- **函数空间聚类**：使用 k-means 对这些函数签名进行聚类，形成共享的 codebook，实现函数级别的权重共享。
- **分支分离编码**：将 **basis branch** 和 **base branch** 分别编码到独立的 codebook 中，保留各自不同的共享结构。
- **硬件感知量化与打包**：对 codebook 权重进行低比特（如 W4）量化，并将每条边的索引进行位打包（bit-packing），最终导出为硬件友好的紧凑格式。

### **相比现有方法的优势**
- **更高的压缩率与更小的精度损失**：在 MNIST 上实现 31.6× 压缩仅损失 0.31pp 准确率，远优于传统方法。
- **硬件友好性**：压缩后模型变为 **index-bound**（索引主导存储），而非 weight-bound，显著降低实际硬件中的存储需求。
- **通用性强**：适用于多种 KAN 变体（SplineKAN, GRAM, FastKAN, ConvKAN）。
- **硬件验证**：在 FPGA 上实测，SplineKAN 权重内存减少 3.87×，且**不增加周期数或延迟**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MNIST**：主基准测试，十种子实验（ten-seed benchmark）。
- **Fashion-MNIST**：辅助图像分类任务。
- **CIFAR-10 / CIFAR-100**：用于评估卷积 KAN（Convolutional KAGN）的扩展能力。
- **Tiny ImageNet**：作为压力测试（stress test）。

### **实验设置与评估指标**
- **模型架构**：
  - 全连接 KAN：784→64→10（MNIST）
  - 卷积 KAGN：8 层，共 30.6M 参数，6.1M 边（CIFAR）
- **压缩设置**：
  - Codebook 大小：$ K_s $（basis）、$ K_b $（base），典型值 (32,16)
  - 量化位宽：W4（4-bit），部分实验用 W8
  - 采样点数：T=128，范围 [-2.5, 2.5]
- **评估指标**：
  - **Top-1 准确率**（Accuracy）
  - **压缩率**（Compression Ratio）
  - **存储大小**（bit-exact packed storage）
  - **FPGA 资源**：BRAM18 块数、延迟（Latency）、周期数（Cycle Count）

### **基线方法对比**
- **Coefficient-space clustering**：在系数向量上聚类（传统方法）
- **Function-space clustering**：在函数响应上聚类，但不分叉
- **MetaCluster**：基于元学习的 KAN 压缩
- **SHARe-KAN**：Gain-Shape-Bias 分解 + 向量量化
- **LSQ-QAT**：Learned Step Size Quantization（用于 MLP 对比）
- **Pruning + Quantization**：剪枝 + 低比特量化

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 模型 | 压缩率 | 准确率损失 | 数据集 |
|------|--------|------------|--------|
| SplineKAN | **31.6×** | **0.31 pp** | MNIST |
| GRAM KAN | **17.6×** | **0.34 pp** | MNIST |
| Convolutional KAGN | **19.9×** | **0.54 pp (CIFAR-10)**<br>**1.89 pp (CIFAR-100)** | CIFAR |

- 在 **6.1M 边的 ConvKAN** 上，FuncCode 实现约 20× 压缩，准确率接近密集模型。
- 压缩后，**每边索引占总权重位的 99.4%**，证实模型为 **index-bound**。

### **与基线方法的对比结果**

#### **准确性 vs 存储权衡（MNIST）**
- FuncCode (branch-aware) 在相同存储下比 coefficient-clustering 高 **2–3%**。
- 相比 **uniform W4 PTQ**，MLP 仅实现 8× 压缩，而 SplineKAN 实现 31.6× 压缩且精度损失更小。

#### **FPGA 硬件实测结果（SplineKAN）**
| 设计 | BRAM18 (post-route) | 压缩比（vs INT4） | 延迟 |
|------|---------------------|------------------|------|
| Dense INT4 | 147 | 1.0× | 340.1 μs |
| **FuncCode (W4)** | **38** | **3.87×** | **340.1 μs** |

- **权重内存减少 3.87×**，**延迟完全不变**，证明其硬件效率优势。

#### **与其他共享方法对比（GRAM, CIFAR-10）**
| 方法 | 4-bit 准确率 | 压缩率 |
|------|-------------|--------|
| **FuncCode (branch)** | **46.86±0.85** | 17.7× |
| MetaCluster | 30.76±6.38 | 31.9× |
| GSB | 33.70±3.16 | 17.7× |

- FuncCode 在低比特下仍保持高精度，而其他方法出现严重崩溃。

---

### **消融实验结果**

#### **(1) 函数空间 vs 系数空间聚类**
- 控制变量实验表明：**函数空间聚类本身并无统计显著优势**。
- 真正带来增益的是 **branch-aware 结构**。

#### **(2) 是否包含 base branch 在签名中？**
| $ K $ | 包含 base | 排除 base | 差值 (pp) |
|-------|-----------|-----------|----------|
| 16 | 92.63 | 64.42 | **+28.21** |
| 32 | 93.68 | 83.39 | **+10.29** |

- **排除 base branch 导致灾难性精度下降**，尤其在小 codebook 下。
- 证明 **base branch 必须参与函数签名构建**。

#### **(3) 分支分离 vs 单一 codebook**
- 单一 codebook（强制 basis 与 base 耦合）在 $ K=32 $ 时仅达 84.93%，而 branch-aware 达 95.32%。
- 证明 **独立编码两分支是关键**。

#### **(4) 量化位宽影响（W8 vs W4 vs W2）**
- **W4 几乎无损**（损失 <0.3pp）
- **W2 导致显著下降**（1.2–7.3pp），尤其对 function-only codebook 更严重。

---

## **4. 关键结论和发现**

### **主要发现**
1. **函数空间冗余性**：KAN 边的函数响应比其系数表示具有更低的有效秩（effective rank），平均低 13–35%，说明函数空间更适合压缩。
2. **branch-aware 是关键**：相比聚类空间的选择，**是否分离 basis 与 base 分支进行编码**才是决定压缩效果的核心因素。
3. **index-bound 架构**：压缩后存储主要由索引构成（>98%），codebook 本身几乎可忽略，这使得 FuncCode 特别适合硬件部署。
4. **硬件收益可验证**：在 FPGA 上，FuncCode 显著减少 BRAM 使用，且不改变计算调度，真正实现了“压缩即省硬件”。

### **方法的局限性**
- **对某些 KAN 家族敏感**：如 FastKAN 在低比特下表现不稳定。
- **卷积 KAN 收益较小**：因已有空间共享，FuncCode 增益有限（仅 1.19× BRAM 减少）。
- **训练依赖性**：需从量化后的模型进行聚类，直接从 FP32 模型聚类效果差。
- **codebook 分配未自适应**：当前使用固定 $ K_s, K_b $，未来可探索动态分配。

### **未来工作方向**
- **自适应 codebook 分配**：根据不同层或任务动态调整 $ K_s $ 和 $ K_b $。
- **扩展至更大规模模型**：如 Vision Transformers with KAN 替代 MLP。
- **支持批处理推理**（batched inference）和能效测量。
- **探索更高效的索引编码**（如 Huffman），尽管当前已接近熵界。
- **结合稀疏化或其他结构压缩方法**，进一步提升压缩率。

---

> ✅ **代码开源**：`https://github.com/OSU-STARLAB/FuncCode`

</details>

---

### 12. [Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in LLM Inference](https://arxiv.org/abs/2609.26621)

**Authors**: Gaoyuan Du, Anam Nawaz Khan, Rex Zhou, Xiaoyang Liu, Deepayan Chakrabarti, Fnu Suya, Xueping Li  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.26621v1  

#### Abstract
Greedy decoding from large language models is commonly treated as deterministic. We show it is not precision-invariant: the same model, prompt, and decoding algorithm produce different outputs in BF16 versus FP16 on identical hardware. Across our evaluations of six models (1.1B-7B parameters, four f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in LLM Inference*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文揭示了一个被广泛忽视的关键问题：**大型语言模型（LLM）的贪心解码（greedy decoding）并非精度不变（precision-invariant）**。尽管在相同模型、提示词和解码算法下，人们通常认为输出是确定性的，但研究发现，在 **BF16** 和 **FP16** 这两种常见的低精度格式下，同一模型会产生不同的输出序列。

这一现象对以下场景构成严重挑战：
- **可复现性与审计**：不同精度下的副本无法精确重放（replay）生成轨迹。
- **基准测试有效性**：同一模型的不同精度版本可能报告相似的准确率，但在个体预测上存在分歧。
- **高精度参考对齐**：服务端选择的低精度格式可能导致偏离更高精度（如FP32）的行为。

### 提出了什么新方法或新思路
作者提出了一种**基于机制分析的干预方法**，其核心思想是：
> 跨精度输出分歧主要由 **lm_head 层的小 top-two logit margin** 引发，而非整个网络的累积误差。

据此，他们设计了 **selective FP32 lm_head recomputation**（选择性FP32 lm_head重计算）策略：
- 在每个解码步中，先用原精度（BF16/FP16）计算 logits 并检查 top-two margin。
- 若 margin 小于阈值 $ T $，则仅将 `lm_head` 投影层以 **FP32** 重新计算一次。
- 其余步骤保持原精度运行。

这种方法被称为 **Intervention C**，是一种轻量级、局部修复方案。

### 相比现有方法的优势
| 方法 | 缺陷 | 本论文优势 |
|------|------|------------|
| **全局升级为FP32** | 内存翻倍，延迟增加约38%，不可部署于大模型 | 仅增加 **<4% 延迟**，内存不变 |
| **LayerCast等全层修复** | 计算开销大，可能引入“蝴蝶效应” | 仅针对关键步骤进行最小干预 |
| **Consensus decoding** | 需并行运行两份模型，成本翻倍 | 单次推理即可实现部分修复 |

此外，该方法具有**理论可解释性**，基于对误差传播路径的实证分析，而非黑箱优化。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GSM8K**：数学推理任务（8-shot Chain-of-Thought）
- **HumanEval**：代码生成任务（0-shot）
- **MBPP**：Python编程任务（3-shot）
- **MATH-500**：竞赛级数学题（用于长链验证）

所有实验均采用 **bare prompt format**（无系统提示），确保跨模型公平比较。

### 实验设置
- **模型范围**：涵盖 **1.1B 到 7B 参数** 的六款主流模型，来自四个家族：
  - Llama（TinyLlama-1.1B, Llama-3.2-3B）
  - Qwen（Qwen2.5-3B/7B）
  - Mistral（Mistral-7B）
  - MoE 架构（OLMoE-1B-7B）
- **硬件平台**：NVIDIA A10G（主）、L4、A100、T4
- **精度对比**：**BF16 vs. FP16**（主要），扩展至 FP16 vs. FP32 和 head-only FP8
- **批大小**：$ \text{bs} \in \{1, 2, 4, 8\} $
- **控制变量**：固定随机种子，启用 `torch.use_deterministic_algorithms(True)` 和 `CUBLAS_WORKSPACE_CONFIG`

### 评估指标
| 指标 | 定义 |
|------|------|
| **EAR (Exact Agreement Rate)** | BF16 与 FP16 输出 token 序列完全一致的比例 |
| **SAR (Semantic Agreement Rate)** | 最终答案语义正确性一致的比例（如提取数字） |
| **t*** | 第一个 token 分歧的位置 |
| **Pass@1** | 贪心解码最终答案正确的比例（用于 benchmark accuracy） |
| **Latency Overhead** | 干预带来的额外延迟百分比 |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Baseline (No Intervention)** | 标准 BF16/FP16 贪心解码 |
| **Greedy FP32 (Oracle)** | 完全FP32推理，理论上应完全一致 |
| **Global FP32 Compute** | 整个模型升为FP32计算（非存储） |
| **Ungated FP32 Recomputation** | 每一步都重算 lm_head（无条件触发） |
| **Integer Quantization (A)** | 对 logits 做整数量化以统一比较行为 |
| **Top-K FP32 Recomputation (B)** | 仅重算 top-K 个候选 token 的 logits |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）跨精度分歧普遍存在
- 在 **49–100% 的 prompt 上**，BF16 与 FP16 输出不同。
- TinyLlama-1.1B 上：
  - GSM8K：**59% 分歧率**（EAR = 41%）
  - MBPP：**70% 分歧率**（EAR = 30%）
- Qwen2.5-3B-Instruct 上高达 **82% 分歧率**（EAR = 18%）
- 一旦发生 token flip，平均导致 **34 tokens 的长度差异**，最大达 **236 tokens**

#### （2）selective FP32 lm_head 有效提升一致性
| 模型 | Baseline EAR | Intervention C EAR | **提升 (ΔEAR)** | 开销 |
|------|-------------|---------------------|------------------|-------|
| TinyLlama-1.1B | 41% | 63% | **+22pp** | <4% |
| Llama-3.2-3B | 31% | 67% | **+36pp** | <4% |
| Qwen2.5-3B | 18% | 21% | +3pp | <4% |
| Mistral-7B | 51% | 59% | +8pp | <4% |
| OLMoE-1B-7B | 34% | 44% | +10pp | <4% |

> ✅ **最佳表现**：在 Llama-3.2-3B 上实现了 **+36个百分点** 的精确匹配提升，仅增加 **1.4% 延迟**。

#### （3）与其他干预方法对比
| 方法 | EAR 提升 | 成本 | 是否优于基线 |
|------|----------|------|--------------|
| **Intervention C (Full FP32 lm_head)** | +22~36pp | <4% | ✅ 是 |
| **Intervention B (Top-K FP32)** | +22~36pp | <2% | ✅ 与 C 相当 |
| **Global FP32 Compute** | +47~58pp | ~38% 延迟 | ✅ 但成本过高 |
| **Ungated Recomputation** | +13~19pp | ~2.5× 延迟 | ❌ 反而更差 |
| **Integer Quantization (A)** | 0pp | <1% | ❌ 无效 |

> 🔍 发现：**K=2 即可达到与 full-vocab 相同效果**，说明只需修正前两名竞争者即可。

#### （4）消融实验结果
| 实验 | 结果 | 含义 |
|------|------|------|
| **Threshold 敏感性**（$ T \in [10^{-5}, 10^{-2}] $） | EAR 几乎不变 | margin 分布呈双峰，阈值鲁棒 |
| **Scope 扩展**（加 RMSNorm） | EAR 从 63% ↓ 至 44% | 更广范围反而破坏稳定性（“蝴蝶效应”） |
| **Temperature Sharpening** | 0 效果 | 证明分歧源于数值差异，而非排序敏感 |
| **KV Cache Grafting** | 36–48% 可通过缓存交换修复 | 表明缓存状态也是分歧来源之一 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **贪心解码不是精度不变的**：BF16 与 FP16 在相同条件下会生成不同输出，且分歧率高达 **近半数以上**。
2. 🔍 **分歧机制明确**：分歧集中在 **lm_head 层的 top-two logit margin 很小** 的步骤。此时微小的浮点舍入误差足以改变 argmax 结果。
3. 🎯 **误差放大源定位**：虽然 body 层有均匀误差积累，但只有 **lm_head 的大规模投影**（$ d \times |V| $）才会将其放大到足以引发 token flip。
4. ⚙️ **选择性重计算最有效**：仅在 margin < $ T $ 时对 `lm_head` 进行 FP32 重计算，可在 **<4% 开销下恢复 +22–36pp EAR**。
5. 📉 **盲目扩大修复范围有害**：无条件重计算或扩展到 RMSNorm 层会导致 EAR 下降，说明“越修越错”。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **仅适用于 low-batch 场景** | 当 batch size > 8 时，收益归零（因 reduction order 差异主导） |
| **不适用于 body-dominated 模型** | 如某些 Qwen 模型训练时已饱和于 BF16，早期即发散，无法修复 |
| **不能保证完全一致** | 平均只能修复约 1/3 的分歧案例，残余差距来自权重截断（weight truncation） |
| **依赖模型训练稳定性** | 对训练期间精度稳定的模型更有效（如 Llama 系列） |
| **FP8 场景受限** | 在 end-to-end FP8 量化中，body 错误主导，head-only 修复仅获 +1pp 提升 |

### 未来工作方向
1. **构建通用的“精度稳定性”评测基准**，用于预判模型是否适合此类修复。
2. **开发训练时感知精度的方法**，使模型在多种精度下都能保持一致行为。
3. **探索 MoE、稀疏激活等新型架构中的跨精度行为**。
4. **结合 body-scope 修复方法**（如 LayerCast），形成组合式解决方案，应对 body-dominated 场景。
5. **推广至采样解码（sampling）场景**，研究温度、top-p 等参数与精度交互的影响。

---

> 💡 **一句话总结**：  
> 本文首次系统揭示了 LLM 贪心解码在 BF16/FP16 下的非确定性，并提出一种低成本、高效益的选择性 FP32 lm_head 重计算方法，在多个主流模型上显著提升了跨精度输出一致性，为生产环境中的可复现推理提供了实用路径。

</details>

---

### 13. [Retrieved-Span Training for Efficient Query-Focused Meeting Summarization on QMSum](https://arxiv.org/abs/2609.25028)

**Authors**: Edward Xi Yang (Ertas AI)  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.25028v1  

#### Abstract
QMSum provides no scorer, making query-focused meeting summarization results difficult to compare. We rescore or generate 15 systems under one implementation. Through a common inference port, a released 406M Fusion-in-Decoder specialist loses 6.30 ROUGE-1 when moved from capped long input to 2,000-w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **Retrieved-Span Training for Efficient Query-Focused Meeting Summarization on QMSum**  
—— 核心结论与实验总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **QMSum 缺乏统一评估协议**：原始 QMSum 数据集未提供标准的评估代码（如 ROUGE 实现），导致不同研究间的性能比较存在不可靠性。
- **长文本输入与小模型能力不匹配**：会议转录平均约 9,000 单词，远超小型模型上下文窗口，直接处理需依赖昂贵的长上下文大模型。
- **训练与推理输入不一致**：许多先进系统在长输入上训练，但在检索出的短片段上推理，造成性能下降。

### 🚀 提出的新方法与思路
- **构建统一评估基准（Common Scale）**：
  - 自行实现并开源一套完整的评估协议（scorer）、prompt、数据划分和预测输出。
  - 所有系统在同一 scorer 和测试集上重新评分，确保公平可比。
- **提出“Retrieved-Span Training”范式**：
  - 将一个原本为长输入设计的 **406M Fusion-in-Decoder 模型（Segment Encoder）** 在检索出的短文本片段（retrieved spans）上进行微调。
  - 对齐训练与推理时的输入长度分布，显著恢复因输入变化导致的性能损失。
- **轻量级本地化方案优于零样本大模型**：
  - 展示了一个经过任务微调的小模型（406M）在特定 prompt 下，超过多个商用托管大模型（如 GPT-5.6、Claude Opus）的表现。

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **资源效率** | 406M 模型仅用约 1/3 参数、<1/2 推理显存（2.66GB vs 5.73GB），训练时间约 45 分钟（单卡 16GB）。 |
| **评估可靠性** | 提供完整复现工具链（code + scorer + predictions），解决跨研究不可比问题。 |
| **性能鲁棒性** | 通过 fine-tuning 匹配输入 regime 可完全弥补从长输入迁移到短 span 的性能差距（+6.3 ROUGE-1 恢复）。 |
| **成本可控性** | 支持在消费级硬件上部署高效 query-focused summarization 流程（locate-then-summarize）。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **QMSum** 数据集：
  - 包含 232 场会议转录（学术、产品设计、议会等），平均 ~9,000 词，最长达 25k 词。
  - 共 1,810 个 query-summary 对，划分为：
    - Train: 1,257
    - Validation: 272
    - Test: 281（覆盖 35 场会议）
  - 提供人工标注的相关文本片段（gold spans），支持 retrieval-based 方法。

### ⚙️ 实验设置
- **Pipeline 架构**：`Locate-then-Summarize`
  1. **Locator**：使用 cross-encoder（MiniLM）对 utterance chunks 打分，选取 top-k 最相关段落（按字数预算打包）。
     - Promoted 设置：375-word 窗口，2,000-word 总预算，12-layer ranker。
  2. **Summarizer**：基于 `LiquidAI/LFM2.5-1.2B-Instruct` 模型，使用 **QLoRA** 微调（4-bit, rank=16）。
     - 输入为 locator 输出的 retrieved spans。
     - 上下文长度：6,144 tokens。
- **训练细节**：
  - Summarizer 微调：3 epochs，batch size=16，lr=2e-4，在单张 16GB GPU 上耗时约 5 小时。
  - Span-trained SegEnc：将 Pagnoni et al. (2023) 发布的 406M Segment Encoder 在 retrieved spans 上继续微调。

### 📊 评估指标
- 主要自动指标：
  - **ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum**（F1，启用 Porter stemming）
  - **BERTScore**（F1，roberta-large）
- 统计显著性分析：
  - 使用 **paired bootstrap resampling**（10,000 次抽样）计算 95% 置信区间。
  - 考虑 **meeting-level clustering** 进一步放宽置信区间以反映嵌套结构。
- 开源内容：
  - 完整代码库（Apache-2.0）
  - Scorer 实现
  - Per-query predictions
  - 所有训练好的适配器（adapter）和 locator 模型

### 🆚 基线方法对比
| 类别 | 基线模型 |
|------|--------|
| **Proprietary Hosted Models (Zero-shot)** | GPT-5.6 Sol/Luna, Claude Opus/Sonnet/Haiku, Gemini（排除） |
| **Community Checkpoints** | DistilBART, BART-large-CNN, PEGASUS, LED-base（均来自 mikeadimech 的 QMSum 微调版本） |
| **Publicly Released Specialist** | Socratic-SegEnc (Pagnoni et al., 2023)，406M Fusion-in-Decoder 模型 |
| **Abalations** | 不同 locator 设置、是否微调、是否使用黄金片段、不同输入格式等 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Test Split, n=281）

| System | Params | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | BERTScore |
|-------|--------|---------|---------|------------|-----------|
| **Span-trained SegEnc** | 406M | **36.33** | 12.72 | 32.17 | 0.8710 |
| **Ours (promoted)** | 1.2B+33M | 35.41 | 12.28 | 31.36 | 0.8733 |
| Socratic-SegEnc (authors’ release) | 406M | 38.60 | 13.91 | 33.71 | 0.8737 |
| GPT-5.6 Luna (zero-shot) | — | 32.43 | 7.89 | 27.49 | 0.8608 |
| Claude Opus 5 (zero-shot) | — | 28.87 | 8.43 | 25.01 | 0.8522 |

> ✅ 注：所有行均由作者使用同一 scorer 重新评估，除 Socratic-SegEnc 外均为自行生成预测。

### 🔁 与基线方法的关键对比结果
- **vs 商用大模型**：
  - 我们的 1.2B 模型比最强的 GPT-5.6 Luna 高 **+2.98 ROUGE-1**（CI: [1.75, 4.24]），统计显著。
  - 甚至 **406M 的 span-trained SegEnc** 也明显领先所有零样本大模型。
- **vs 已发布最强公开模型（Socratic-SegEnc）**：
  - 原始模型在作者 pipeline 中得分为 38.60，但通过本文 **port 到本实验流程后降至 35.30**（-3.3 点），表明实现差异影响巨大。
  - 经过在 retrieved spans 上微调后，该模型回升至 **36.33 ROUGE-1**。
- **两系统无统计分离**：
  - Span-trained SegEnc vs Our 1.2B：差值 +0.93 ROUGE-1，**95% CI [-0.27, +2.22]**（meeting-cluster），**未达到统计显著水平**。

### 🔍 消融实验结果
#### （1）训练 regime 对齐至关重要
| Condition | Val ROUGE-1 | Δ |
|---------|-------------|----|
| Stock SegEnc (long input) | 35.30 | — |
| → 移至 retrieved spans（same ckpt） | 29.00 | **-6.30** |
| → Fine-tune on spans | 37.05 | **+8.05 recovery** |

> ✔️ 结论：性能下降主因是 **input regime mismatch**，而非架构劣势；fine-tuning 可完全恢复。

#### （2）固定模型下的控制变量分析（基于 1.2B base）
| Configuration | Test ROUGE-1 | Δ |
|--------------|---------------|-----|
| Base model + first 4,500 words | 28.57 | — |
| + Retrieved spans（same input length） | 30.12 | +1.55 |
| + Fine-tuning on spans | 35.41 | **+5.29** |

> ✔️ 结论：**fine-tuning 是主导因素**，贡献远大于 retrieval 本身。

#### （3）完美 retrieval 的增益有限
- 使用 gold spans 替代 locator 输出：
  - Val ROUGE-1 提升仅 **+1.09**，CI [-0.22, +2.43]，接近检测下限（~1.0）。
  - 即使拥有完美 locator，仍落后于 Socratic-SegEnc **+1.40**（CI: [+0.05, +2.76]）。
> ✔️ 结论：**retrieval recall 不是瓶颈**，模型 summarization 能力才是关键。

#### （4）更多 recall ≠ 更好 ROUGE
- 固定 budget 下提升 recall（如从 0.627→0.765）：
  - ROUGE-1 变化仅为 +0.61，CI [-0.49, +1.69]，**不显著**。
- 最优设置反而是更短输入（2,000 words）+ 更高精度。
> ✔️ 启示：应关注 **precision 与 downstream utility**，而非单纯优化 recall。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **训练与推理输入一致性是关键**：
   - 即使是强大的 specialist 模型，若训练在长输入而推理在短 span，会损失高达 **6.3 ROUGE-1**。
   - 通过在目标输入 regime 上微调，可完全恢复性能。
2. **小模型经适当训练可超越零样本大模型**：
   - 一个 406M 的 span-trained 模型在 ROUGE 上显著优于 GPT-5.6、Claude 等商用模型。
   - 但这受限于 prompt 设计与输出长度（hosted models 输出更长，recall 高但 precision 低）。
3. **当前 benchmark 检测能力有限**：
   - QMSum 的 full test split 对 ROUGE-1 的分辨极限约为 **±1.0~1.4 点**。
   - 多数 SOTA 方法之间的差距在此范围内，难以可靠排序。
4. **Retrieval Recall 并非核心杠杆**：
   - 提高 recall 在固定 budget 下未带来可观 ROUGE 提升。
   - 黄金片段仅带来约 1.09 点增益，接近噪声水平。

### ⚠️ 方法局限性
| 局限 | 说明 |
|------|------|
| **仅限自动指标** | 所有结论基于 ROUGE/BERTScore，未验证 human preference 或 factuality。已有工作显示事实准确性排序可能反转。 |
| **未联合训练 locate & summarize** | 两阶段独立训练，未探索 DYLE 式 joint learning 的潜力。 |
| **port 存在偏差** | Segment Encoder 的移植版本比原版低 3.3 点，无法完全归因于实现差异。 |
| **训练稳定性差** | 在 16GB 显存边缘运行，多次训练失败，结果受 seed 影响较大（±1.59 ROUGE-1）。 |
| **非全文本对比** | 所有系统都受限于输入长度（SegEnc ~11.8K words, ours 2K），并非 full-transcript vs retrieval 的公平比较。 |

### 🔮 未来工作方向
1. **开发面向 retrieval-augmented summarization 的专用评估指标**，结合 proposition-level matching 和 factuality checking。
2. **探索 joint training 或 iterative refinement** 机制，让 summarizer 反馈指导 locator。
3. **研究 input compression 与 semantic fidelity 的权衡**，不只是 recall 最大化。
4. **推动标准化 benchmark 流程**，包括 scorer、prompt、generation constraints 的统一。
5. **扩展到多语言或多模态会议摘要场景**。

---

## 📦 可复现性声明
作者已全面开源以下内容：
- GitHub 仓库：https://github.com/ErtasAI/qmsum-retrieved-span-training （Apache-2.0）
- Hugging Face 模型权重（含 adapter、locator、span-trained SegEnc）
- 完整 scorer 实现（rouge-score + bert-score 统一接口）
- 所有 per-query predictions
- 冻结的 prompt 与 protocol 模块

> “One command sequence reproduces our test row without API keys.”

此举极大提升了 NLP 社区在 query-focused summarization 方向上的可比性和可复现性。

</details>

---

### 14. [Compressing Long Context into Answer-Aligned Memory Embeddings for LLM Inference](https://arxiv.org/abs/2609.25537)

**Authors**: Md Mostafizer Rahman, Md Faizul Ibne Amin, Md Shahajada Mia, Yutaka Watanobe, Fang Liu  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.25537v1  

#### Abstract
Large language model (LLM) inference is constrained by the quadratic scaling of self-attention and the linear scaling of the KV cache, increasing latency, energy consumption, and GPU memory demand as context length scales. Existing soft-compression methods either lack query-guided memory selection a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Compressing Long Context into Answer-Aligned Memory Embeddings for LLM Inference

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLM）在处理长上下文时面临显著的计算瓶颈：
- **Self-attention 的二次复杂度**：随着上下文长度增长，计算开销急剧上升。
- **KV Cache 的线性增长**：占用大量 GPU 内存，增加推理延迟和能耗。
- 现有软压缩方法存在以下不足：
  - 缺乏基于查询（query-guided）的记忆选择机制；
  - 训练过程缺乏针对答案的监督（answer-targeted supervision）；
  - 压缩模块与特定解码器架构紧密耦合，难以跨架构部署。

### 提出了什么新方法或新思路
提出 **Context-to-Answer-Aligned Memory Compression (CMC)** 框架，实现高效、可迁移的长上下文压缩：

#### 核心组件
1. **ContextEncoder**  
   - 应用图结构上下文去噪（graph-based context denoising），移除语义孤立的低显著性 token。
   - 将上下文分块并生成 **Context Memory Embeddings (CMEs)**，通过固定占位符 token 学习段落级语义表示。

2. **MemoryBridge**  
   - 一个两层 MLP 投影模块，将 CMEs 映射到任意冻结解码器（frozen DecoderLLM）的嵌入空间。
   - 支持跨架构部署（cross-architecture deployment），无需修改解码器权重。

3. **Two-tier KV Cache 推理策略**
   - **Tier-1**: 基于问题向量与 CMEs 的余弦相似度，动态选择 top-K CMEs。
   - **Tier-2**: 保留以黄金答案为中心的局部窗口（local context window），提供细粒度文本精度。
   - 总 KV Cache 长度被限制为 `K + W + |q|`，与原始文档长度无关。

#### 创新训练策略（Two-phase Training）
- **Phase-1: Decoder Alignment**  
  使用自编码（AE）和自回归（AR）目标对齐 CMEs 与解码器空间。
- **Phase-2: Answer-aligned Fine-tuning**  
  引入知识蒸馏与对比学习：
  - **Cross-Entropy Loss**：监督生成正确答案；
  - **KL Divergence Loss**：对齐学生与教师模型输出分布；
  - **Contrastive Loss**：拉近 CMEs 平均表示与答案隐藏状态的距离。

### 相比现有方法的优势
| 特性 | CMC | 其他软压缩方法（如 PCC、AutoCompressor） |
|------|-----|----------------------------------------|
| Query-guided selection | ✅ 动态选择相关 CMEs | ❌ 统一供给所有 memory slots |
| Answer-targeted training | ✅ Phase-2 引入答案监督 | ❌ 仅文本重建目标 |
| Cross-architecture support | ✅ MemoryBridge 实现灵活配对 | ❌ 多绑定特定解码器 |
| 局部精度保留 | ✅ Two-tier KV Cache 结构 | ❌ 完全依赖压缩表示 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
在四个抽取式 QA 数据集上进行评估，涵盖多种推理类型：
- **SQuAD**：单跳阅读理解（single-hop QA）
- **AdversarialQA**：对抗性构造的问题，挑战模型理解能力
- **HotpotQA**：需要多跳推理（multi-hop reasoning）的问答
- **CovidQA**：生物医学领域的专业问答（domain-specific QA）

分为两种设置：
- **Controlled-data setting**：统一采样规模（10k训练/1k验证等）
- **Full-data setting**：使用完整训练集，便于与已有工作比较

### 实验设置和评估指标

#### 模型组合
- **ContextEncoder**：GPT2-Large、OPT-1.3B、OPT-2.7B
- **DecoderLLM**：Llama-3-8B-Instruct、Mistral-7B-Instruct-v0.3、Gemma-2-9B-IT
- 共 **9 种 encoder-decoder 组合**

#### 评估指标
| 类别 | 指标 |
|------|------|
| **任务性能** | Exact Match (EM), F1 |
| **推理效率** | - 推理时间（分解为压缩、prefill、decode阶段）<br>- 能耗（kWh，通过 CodeCarbon 测量）<br>- Multiply-Accumulate Operations (MACs)<br>- 峰值分配/预留 GPU 内存（Peak allocated/reserved GPU memory） |

#### 超参数
- 压缩率 $ r \in \{2,4,8\} $
- Tier-1 CME 数量 $ K \in \{30,40\} $
- Tier-2 局部窗口大小 $ W \in \{130,256\} $

### 基线方法对比

#### 主要基线（Primary Baseline）
- 不压缩，直接输入局部上下文窗口（gold-offset centered window）

#### 发表基线（Published Baselines）
| 类型 | 方法 |
|------|------|
| Hard-prompt | LLMLingua-2 |
| Soft-compression | AutoCompressor, xRAG, ICAE, PCC-lite, PCC-large |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 任务性能提升（Full-data Setting）
| 方法 | SQuAD EM↑ | HotpotQA EM↑ | AdversarialQA EM↑ |
|------|----------|-------------|------------------|
| **CMC** | **55.56** | **47.77** | **35.43** |
| PCC-Large | 60.04 | 39.97 | 39.37 |
| LLMLingua-2 | 32.18 | 44.18 | 24.80 |

- **CMC 在 F1 上全面领先**：
  - SQuAD: **78.29** vs. PCC-Large 的 77.76
  - HotpotQA: **67.83** vs. PCC-Large 的 48.19（+19.64）
  - AdversarialQA: **53.42** vs. PCC-Large 的 52.56

> 💡 尤其在多跳推理任务 HotpotQA 上表现突出，说明 CMC 更好地保留了跨段落证据链。

#### 效率提升（T=3,000 generation tokens）
| 指标 | 提升幅度 |
|------|---------|
| **推理时间减少** | 最高 **20.0%** |
| **能耗降低** | 最高 **20.3%** |
| **峰值预留 GPU 内存下降** | 最高 **50–62.5%**（Mistral 达 62.5%） |

- 压缩开销恒定（约 25s），而 decode 节省随生成长度线性增长，体现出显著的**长序列优势**。

### 与基线方法的对比结果
- CMC 在多数配置下优于本地窗口基线，尤其在 Llama 和 Gemma 上提升明显。
- 相比其他软压缩方法，CMC 在 F1 上普遍领先，表明其答案对齐训练更有效。
- 尽管 PCC-Large 在 SQuAD EM 上略高，但其使用更大模型且无 query-guided selection，不具备公平可比性。

### 消融实验结果（Ablation Study on SQuAD）

| 消融条件 | EM | ΔEM (vs CMC) |
|--------|----|--------------|
| CMC（完整） | 0.670 | – |
| w/o graph denoising (A1) | 0.404 | -0.266 |
| w/o query-guided top-K (A2) | 0.589 | -0.081 |
| w/o Tier-2 local window (A3) | 0.170 | **-0.500** |
| Phase-1 only (no Phase-2) | 0.589 | -0.081 |

> 🔍 **关键发现**：
> - Tier-2 局部窗口是结构上不可或缺的，移除后性能崩溃。
> - 图结构去噪极大提升 CME 质量。
> - Phase-2 中三个损失函数必须协同作用，缺一不可。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **CMC 显著提升了长上下文下的 QA 性能与效率平衡**：
   - 在多个数据集上一致超越基线，尤其在多跳推理任务中优势明显。
2. ✅ **Two-tier KV Cache 是核心设计**：
   - Tier-1 提供全局语义覆盖；
   - Tier-2 补充局部精确信息，避免完全依赖压缩带来的细节丢失。
3. ✅ **答案对齐训练至关重要**：
   - Phase-2 的知识蒸馏 + 对比学习使 CMEs 更聚焦于答案相关信息。
4. ✅ **跨架构兼容性强**：
   - MemoryBridge 成功桥接不同 encoder-decoder 架构，支持即插即用部署。

### 方法的局限性
1. **仅限于抽取式 QA 任务**：未验证在抽象式生成、摘要、RAG 等任务中的效果。
2. **固定压缩率**：当前使用静态 $ r \in \{2,4,8\} $，无法根据上下文长度自适应调整。
3. **依赖黄金答案位置**：Tier-2 窗口定位需知道答案偏移，在真实场景中可能不可得（尽管附录显示 BM25/CME-guided 可近似替代）。
4. **语言限制**：所有实验基于英文数据集，未测试非英语或多语言场景下的有效性。

### 未来工作方向
- 设计 **adaptive compression rate scheduler**，根据输入长度动态调节压缩强度。
- 扩展至 **abstractive tasks and RAG pipelines**，探索更通用的答案对齐机制。
- 研究 **zero-shot cross-dataset generalization**，例如在 SQuAD 上训练后迁移到 HotpotQA。
- 探索 **multilingual CMC**，适配不同语言的 tokenization 与语义特性。
- 开发更鲁棒的 **Tier-2 window localization module**，摆脱对标注答案的依赖。

---

> 📚 **代码开源地址**：https://github.com/mostafiz26/CMC

</details>

---

### 15. [HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing](https://arxiv.org/abs/2609.26368)

**Authors**: Jianyu Wei, Yizhao Gao, Qihao Zhang, Shimao Chen, Zhengju Tang, Yu Cheng, Shengjie Zhou, Zihan Jiang, Yifan Song, Hailin Zhang, Liang Zhao, Bo Yang, Gang Wang, Shijie Cao, Fuli Luo  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.26368v1  

#### Abstract
Long-horizon and multi-turn agents typically generate short actions and process long observations from tools and environments. This growing context demands efficient prefill, compact KV-cache storage, and accurate long-context retrieval. To meet these demands, we introduce HySparse2, a hybrid sparse...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代**long-horizon、multi-turn agentic workloads**（如多轮工具调用、长上下文推理）面临三大挑战：
- **Prefill 阶段计算开销大**：输入序列长，需处理大量 observation tokens。
- **KV-cache 存储成本高**：历史上下文不断累积，KV-cache 占用内存显著增长。
- **长上下文检索精度不足**：在稀疏注意力中，block-level selection 容易遗漏关键 token。

HySparse2 旨在同时优化 **prefill 效率、KV-cache 压缩、长上下文检索准确率**。

---

### 提出的新方法与创新思路
HySparse2 是一种基于 **two-level KV sharing** 的混合稀疏注意力架构，其核心创新如下：

#### （1）**Outer Level: KV Bridging（跨解码器 KV 共享）**
- 架构上采用 **YOCO-style self-decoder + cross-decoder** 结构：
  - **Self-decoder**：使用 full attention 和 sliding-window attention（SWA），负责构建全局隐藏状态。
  - **Cross-decoder**：使用 full attention 和 sparse attention，用于生成响应。
- **KV Bridging**：仅在 full-attention 层之间进行跨解码器共享。cross-decoder 的 KV 通过 self-decoder 对应层的 hidden states 经过独立的 `Proj_K` / `Proj_V` 投影得到。
- 优势：**Prefill 可在 self-decoder 后提前退出**，无需运行整个 cross-decoder。

#### （2）**Inner Level: KV Reuse 改进（HySparse 升级版）**
- **Token-level sparse selection**：取代 HySparse 中的 block-level selection，实现更精细的 token 级别检索。
- **移除独立 SWA 分支**：不再为局部建模保留单独的 SWA attention 分支，而是将“最近窗口”**强制纳入稀疏选择集合**，统一复用 full-attention 的 KV cache。
- 优势：减少参数量、避免 cross-decoder 中的 SWA 依赖链，支持完全 early exit。

---

### 相比现有方法的优势
| 方面 | HySparse2 vs. HySparse | HySparse2 vs. Hybrid SWA |
|------|------------------------|--------------------------|
| **Prefill 计算** | 减少 ~2.92× FLOPs @1M | 减少 ~5.02× FLOPs @1M |
| **KV-cache 大小** | 2.69 GB vs. 6.72 GB | 2.69 GB vs. 12.09 GB |
| **长上下文检索** | ↑11.30 pts on MRCR-v2, ↑19.81 pts on RULER-v2 | 显著优于 |
| **Prefill 路径长度** | 仅需运行 ~一半层数（self-decoder） | 必须运行全部层 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **预训练阶段**：约 500B tokens，context length = 32k。
- **轻量后训练（light post-training）**：加入 agent 数据，扩展至 256k context。
- **评估任务覆盖多个维度**：

| 类别 | 任务 |
|------|------|
| **通用能力** | MMLU, C-Eval, CMMLU, TriviaQA, BBH, MATH, GSM8K |
| **推理与代码** | MMLU-Pro, DROP, HumanEval+, MBPP+ |
| **长上下文建模** | RULER, NoLiMa, LongPPL, Repo Code PPL |
| **多轮代理任务** | MRCR-v2, RULER-v2, GraphWalks, AgentPPL |

---

### 实验设置
- **模型配置**：80B-A3B MoE 模型，共 49 层，hidden size = 2048。
- **注意力设计对比**：
  
  | Model | #Full | Heads (Q/KV) | Head Dim (Q/V) | 特点 |
  |-------|-------|---------------|----------------|------|
  | Hybrid SWA | 9 | 64/4 | 192/128 | 9 个 full-attention 层 + SWA |
  | HySparse | 5 | 64/4 | 192/128 | 块级稀疏 + SWA 分支 |
  | **HySparse2** | **5** | **64/1 (MQA)** | **256/256** | **token-level 稀疏 + 强制窗口 + KV Bridging** |

- **稀疏设置**：
  - HySparse2：128 个强制本地 token + 1024 个全局 token（token-level）。
  - HySparse：64-token block × 16 blocks + 128-token SWA 分支。
- **训练流程**：
  - 预训练：500B tokens @32k。
  - 后训练：~100B tokens，引入 agent 数据，extend to 256k。
  - 优化器：Muon，学习率分别为 1e-3（pretrain）、5e-5（post-train）。

---

### 基线方法对比
- **HySparse**：前作，块级稀疏 + KV Reuse。
- **Hybrid SWA**：广泛使用的滑动窗口注意力方法（如 MiMo-V2 系列）。
- **Full Attention**：全注意力作为上限参考。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Post-training 结果）

| 指标 | HySparse2 | HySparse | Hybrid SWA | 提升 |
|------|-----------|----------|------------|------|
| **MRCR-v2 (mean)** | **58.45** | 47.15 | 52.01 | ↑11.30 / ↑6.44 |
| **RULER-v2 (mean)** | **58.45** | 38.64 | 39.80 | ↑19.81 / ↑18.65 |
| **AgentPPL ↓** | **1.16** | 1.18 | 1.20 | 更低更好 |
| **LongPPL ↓** | **4.5** | 5.5 | 6.0 | 更低更好 |

> ✅ 在所有 context length（8k–256k）下均全面领先。

---

### Prefill 计算与 KV-cache 成本（@1M tokens）
| 指标 | HySparse2 | HySparse | Hybrid SWA |
|------|-----------|----------|------------|
| **Prefill FLOPs/token** | **~70 GFLOPs** | ~205 | ~350 | ↓2.92× / ↓5.02× |
| **KV-cache size** | **2.69 GB** | 6.72 GB | 12.09 GB | ↓60% / ↓78% |

> 💡 图 4 显示：随着 context 增长，HySparse2 的优势持续扩大。

---

### 消融实验结果

#### （1）Token-level vs. Block-level Sparsity（相同预算下）
| 任务 | Block | Token | Δ |
|------|-------|-------|----|
| RULER-v2 | 49.56 | **56.13** | ↑6.57 |
| MRCR-v2 (2-needle) | 12.94 | **21.08** | ↑8.14 |
| GraphWalks | 29.38 | **34.92** | ↑5.55 |

✅ **Token-level selection 显著提升长上下文检索能力**，尤其对图遍历类任务有效。

---

#### （2）Local Window 设计对比
| 方法 | Gated SWA | No SWA | **Forced SWA** |
|------|-----------|--------|----------------|
| RULER | 88.19 | 84.55 | **89.84** |
| RULER-v2 | 53.66 | 54.62 | **55.98** |
| GraphWalks | 35.39 | 36.48 | **37.13** |
| GSM8K | **64.52** | 60.35 | 59.44 |

⚠️ Forced SWA 在部分数学任务（如 GSM8K）略降，但在检索任务表现最佳，且节省参数与 cache。

---

#### （3）KV Bridging 消融
| 任务 | w/o Bridging | w/ Bridging |
|------|--------------|-------------|
| MMLU | 72.68 | **72.80** |
| TriviaQA | 73.32 | **74.10** |
| RULER | 96.32 | **96.01**（基本持平） |
| LongPPL ↓ | 3.6053 | **3.4202** |

✅ **KV Bridging 不损害模型质量**，甚至在部分任务略有提升。

#### KV Bridging vs. KV Mirror（连接方式对比）
- **KV Bridging** 最终 RULER 得分：**87.65**
- **KV Mirror**：81.28
- ✅ 表明 full-attention 层的 hidden state 是更好的投影源。

---

## 4. 关键结论和发现

### 主要发现
1. **Two-level KV sharing 可显著压缩 prefill 开销与 KV-cache**：
   - 通过 KV Bridging 实现 cross-decoder KV 的“免计算”构建。
   - Prefill 节点只需部署 self-decoder（约一半层数），大幅降低部署成本。

2. **Token-level selection 比 block-level 更适合 agentic 场景**：
   - 在固定 attention budget 下能更精准定位分散的关键 token。
   - 尤其有利于 multi-hop retrieval 和 graph reasoning。

3. **Forced local window 可替代独立 SWA 分支**：
   - 虽然在某些数学任务上略有下降，但整体收益远大于代价。
   - 是实现 early-exit prefill 的关键技术前提。

4. **KV Bridging 不损失模型质量**：
   - 在 290B 规模上的消融实验证明其可扩展性。
   - 性能在多数任务上与无 bridging 设置相当甚至更优。

---

### 方法的局限性
- **对 full-attention 层仍有依赖**：仍保留少量 full-attention 层作为“oracle indexer”，未来可能被轻量 indexer 替代。
- **Forced window 固定大小**：未动态调整局部窗口长度，可能影响极短或极长交互场景。
- **目前仅验证于 MoE 架构**：是否适用于 dense 模型有待验证。

---

### 未来工作方向
1. **进一步减少 full-attention 比例**：
   - 探索用轻量 indexer + sparse attention 替代 full-attention 层（post-training 阶段）。
2. **动态稀疏策略**：
   - 根据 query 内容自适应分配 local/global token 数量。
3. **异步 speculative decoding**：
   - 利用 self-decoder 早期隐藏状态驱动 draft model，提升 decoding 并行度。
4. **扩展至 dense 模型与其他模态**。

---

> 🔚 **总结**：HySparse2 是面向 **agentic workloads** 的下一代高效注意力架构，在保持甚至提升 long-context retrieval 能力的同时，实现了 **prefill 加速 3–5 倍、KV-cache 压缩 60–78%** 的突破性进展，是迈向百万 token 级智能体推理的重要一步。

</details>

---

### 16. [Semantic Abstraction for Natural Language Inference: a Methodological Framework for Discovering and Compensating Semantic Knowledge and Reasoning Gaps in Large Language Models](https://arxiv.org/abs/2609.26610)

**Authors**: David Torres-Moreno, Jorge Hermosillo-Valadez  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.26610v1  

#### Abstract
Despite their outstanding performance on many NLP tasks, LLMs face serious challenges related to semantic abstraction. In this study, we are interested in understanding how LLMs leverage abstract semantic knowledge in natural language inference (NLI), which requires sophisticated linguistic capabili...

---

### 17. [PatchKV: Efficient KV Cache Recovery for Dynamically Edited LLM Contexts](https://arxiv.org/abs/2609.26219)

**Authors**: Guotao Yang, Rui Guo, Siwei He, Sheng Chen, Yitao Hu, Keqiu Li  
**Category**: cs.DC  
**Published**: 2026-09-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.26219v1  

#### Abstract
Long-running LLM agent workflows often revise interior context spans while retaining long suffixes. Although suffix tokens remain unchanged, altered causal histories and rotary positions prevent exact reuse of their offloaded key-value (KV) states. Full suffix recomputation wastes prefill work, whil...

---

### 18. [Fast Matrix Multiplication in fp8: Certified Coefficient Optimization and Measured Error](https://arxiv.org/abs/2609.26077)

**Authors**: Shuxiao Xie, Shuyang Xie, Yuan Cao, Dezhi Ran, Wei Yang, Tao Xie  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.26077v1  

#### Abstract
A Strassen-type algorithm has many realizations with the same exact product and multiplication count yet different fp8 error because basis changes reshape coefficient geometry, posing the question of which to run. No current account settles this: classical stability controls worst-case $\ell_1$ grow...

---

### 19. [DeepFEAv2: Deep Learning for Transient Finite Element Analysis Beyond Structured Meshes](https://arxiv.org/abs/2609.26426)

**Authors**: Georgios Triantafyllou, Panagiotis G. Kalozoumis, Dimitris K. Iakovidis  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.26426v1  

#### Abstract
Finite Element Analysis (FEA) is widely used for transient mechanical simulations, but its high computational cost limits real-time and high-resolution applications. Deep learning surrogate models can reduce this cost; however, many existing approaches are restricted to steady-state prediction or ca...

---

### 20. [PINNForge: Execution-Grounded Evolutionary Design of Physics-Informed Neural Networks for PDE Solving via Large Language Models](https://arxiv.org/abs/2609.23023)

**Authors**: Mingyang Yu, Xu Yang, Jun Zhang, Xiaolong Wang, Jing Xu, Keqian Li  
**Category**: cs.AI  
**Published**: 2026-09-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.23023v1  

#### Abstract
Physics-informed neural networks (PINNs) require coordinated choices over network representation, sampling, loss construction, and optimization, while effective configurations often vary substantially across partial differential equations (PDEs). Existing automated PINN design methods can search can...

---

### 21. [TelecomGPT-R1: Unified Post-Training for Reasoning Across Heterogeneous Telecom Tasks](https://arxiv.org/abs/2609.25356)

**Authors**: Bohao Wang, Chenwei Wu, Hang Zou, Yu Tian, Lina Bariah, Li Wei, Chongwen Huang, Yongliang Shen, Zhaoyang Zhang, Merouane Debbah  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.25356v1  

#### Abstract
Large language models (LLMs) offer great potential to automate a broad range of telecom engineering tasks by reasoning over standards, network configurations, mathematical models, source code, and operational logs. However, existing telecom LLMs struggle to reliably reason across these diverse tasks...

---

### 22. [ClusterFewshot: Improving Few-shot Optimization for LLMs workflow](https://arxiv.org/abs/2609.25939)

**Authors**: Omri Bar Haim, Shahar Katz, Lior Wolf  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.25939v1  

#### Abstract
The performance of large language model (LLM) workflows often depends on selecting a small set of in-context demonstrations to guide model behavior on new tasks. Recent methods improve this process by augmenting prompts with successful reasoning paths. However, their demonstration selection relies o...

---

### 23. [WeightBridge: An Efficient Weight Transfer Library for Reinforcement Learning](https://arxiv.org/abs/2609.25442)

**Authors**: Xuanlin Jiang, Samuel Hsia, Michael Kuchnik, Zachary DeVito, Minlan Yu, Carole-Jean Wu  
**Category**: cs.DC  
**Published**: 2026-09-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.25442v1  

#### Abstract
Weight transfer - the propagation of updated parameters from trainers to rollout generators - is becoming an important performance bottleneck in reinforcement learning (RL) systems for LLMs. The central challenge is supporting the diverse trainer and rollout layouts and synchronization requirements ...

---

### 24. [From Experts to Sub-experts: Fine-grained Parameter-Efficient Fine-Tuning for MoE LLMs](https://arxiv.org/abs/2609.25655)

**Authors**: Zhentao Tan, Chang Liu, Yao Liu, Yue Wu, Jieping Ye  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.25655v1  

#### Abstract
As large language models (LLMs) scale rapidly, dense full-parameter adaptation becomes increasingly expensive, motivating sparse and modular architectures such as Mixture-of-Experts (MoE) models. This shift raises a key question for parameter-efficient fine-tuning (PEFT): at what granularity should ...

---

### 25. [Self-Supervised Combinatorial Optimization with Constraints via Frank-Wolfe](https://arxiv.org/abs/2609.25728)

**Authors**: Akbar Rafiey, Yifei Xu, Nikolaos Karalias  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.25728v1  

#### Abstract
Self-supervised learning for combinatorial optimization has emerged as a promising paradigm for solving discrete optimization problems with neural networks, but a central challenge remains: handling hard combinatorial constraints within continuous, gradient-based training. Continuously extending com...

---

### 26. [GeoPair: Geometry-Preserving Cross-Layer Factorization for Training-Free Transformer Compression](https://arxiv.org/abs/2609.25963)

**Authors**: Baher Mohammad, Ammar Ali, Stamatios Lefkimmiatis  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.25963v1  

#### Abstract
Transformer architectures exhibit cross-layer redundancies, yet post-training compression pipelines typically optimize layers in isolation or rely on heuristic grouping strategies that disregard layer-specific activation geometries. We introduce a principled, training-free framework that sequentiall...

---

### 27. [On Probabilistic Inference Through Parametric Tensor Decomposition in Base Tensor Networks](https://arxiv.org/abs/2609.23774)

**Authors**: Sagad Hamid, Tanya Braun  
**Category**: cs.AI  
**Published**: 2026-09-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.23774v1  

#### Abstract
Probabilistic inference is generally only tractable in low-treewidth graphical models, limiting its effective applicability in high-treewidth settings. Many existing methods improve efficiency by exploiting specific parametric structure, such as symmetries. However, they typically require such struc...

---

### 28. [Informed Masking: Structure-Aware Perturbation for Reinforcement Learning in Diffusion Large Language Models](https://arxiv.org/abs/2609.25927)

**Authors**: Xiaoyi Yu, Enver Sangineto, Pei Fu, Fiorenzo Parascandolo, Wenhui Tan, Ruikang Zhang, Rita Cucchiara, Ruihua Song, Jian Luan  
**Category**: cs.CL  
**Published**: 2026-09-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.25927v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have emerged as an efficient alternative to autoregressive models, yet aligning them via Reinforcement Learning (RL) requires likelihood surrogates estimated from masked reconstruction subproblems under a small Monte Carlo budget per rollout. Existing methods ...

---

### 29. [Fast Recovery for LLM Serving via Decoupled Device Memory Lifetime in Dynamo](https://arxiv.org/abs/2609.25451)

**Authors**: Schwinn Saereesitthipitak (NVIDIA), Mohammed Abdulwahhab (NVIDIA), Hannah Zhang (NVIDIA), Dan Feigin (NVIDIA), Neelay Shah (NVIDIA), Maksim Khadkevich (NVIDIA), Itay Neeman (NVIDIA), Vikram Sharma Mailthody (NVIDIA), Wen-mei W. Hwu (NVIDIA Research)  
**Category**: cs.DC  
**Published**: 2026-09-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.25451v1  

#### Abstract
Large language model (LLM) inference replicas run across tightly coupled GPUs and serve traffic continuously for weeks. Hardware and software failures are therefore inevitable, and one worker failure can disrupt an entire replica. Recovery requires reinitializing the engine, taking minutes even when...

---

### 30. [Deep Reinforcement Learning on Item-Compatibility Graphs for One-Dimensional Bin Packing](https://arxiv.org/abs/2609.25397)

**Authors**: M. Asl{\i} Ayd{\i}n  
**Category**: cs.LG  
**Published**: 2026-09-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.25397v1  

#### Abstract
The one-dimensional bin packing problem (1D-BPP) is a classical NP-hard combinatorial optimization problem with applications ranging from logistics and manufacturing to cloud resource management. Although deep reinforcement learning (DRL) has become a competitive paradigm for data-driven optimizatio...

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

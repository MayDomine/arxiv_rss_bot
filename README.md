# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-04 09:41:06 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2606.04511)

**Authors**: Yaosheng Fu, Guangxuan Xiao, Xin Dong, Song Han, Oreste Villa  
**Category**: cs.CL  
**Published**: 2026-06-04  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2606.04511v1  

#### Abstract
Sparse attention reduces compute and memory bandwidth for long-context LLM inference. However, two key challenges remain: (1) KV cache capacity still grows with sequence length, and offloading to CPU memory introduces a PCIe transfer bottleneck; (2) the sparse selection step itself retains $O(T^2)$ ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference**  
—— 核心结论与实验结果总结

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
现代大语言模型（LLM）在长上下文场景下推理面临三大效率瓶颈：
1. **KV Cache 容量压力**：随着序列长度增长，KV Cache 占用显存急剧上升，导致 GPU 内存不足。
2. **PCIe 传输瓶颈**：将 KV Cache 卸载到 CPU 内存虽可缓解显存压力，但 CPU-GPU 数据传输通过 PCIe 接口成为新的性能瓶颈。
3. **稀疏选择开销高**：尽管稀疏注意力（Sparse Attention）降低了 `O(T²)` 的计算复杂度至 `O(T)`，但其内部的 top-k 选择步骤仍为 `O(T)`，在长序列下反而可能主导总延迟。

现有方法如 InfiniGen 虽尝试预取（prefetch），但依赖隐藏状态作为代理信号，准确性差；而 IndexCache、HISA 等则引入精度-效率权衡。

---

### **提出了什么新方法或新思路**
作者提出 **SparDA**（Sparse Decoupled Attention），一种解耦式的稀疏注意力架构，核心是引入第四个每层投影模块 —— **Forecast**，与 Query、Key、Value 并列。

#### 主要创新点：
- ✅ **Trainable Lookahead Sparse Selection（可学习的前瞻稀疏选择）**  
  将稀疏选择从当前层的 Query 中解耦，改由前一层生成的 **Forecast 向量**驱动下一层的选择。这使得系统可以在执行当前层的同时，提前预测并异步预取下一层所需的 KV Blocks。

- ✅ **Compact Forecast Indexer（紧凑的 Forecast 索引器）**  
  Forecast 不再需要保留完整的多头结构（multi-head）。在 GQA 架构中，每个 GQA group 仅需一个 Forecast Head，大幅减少选择开销，并跳过 softmax 操作。

- ✅ **Asynchronous Prefetch with Persistent UVA Kernel（基于持久化 UVA 内核的异步预取）**  
  利用统一虚拟寻址（UVA）和 Triton 编写的持久化内核，在专用 CUDA 流上实现高效、低开销的 CPU-to-GPU 异步数据搬运，最大化通信-计算重叠。

---

### **相比现有方法的优势**
| 维度 | SparDA vs. Baseline |
|------|---------------------|
| **性能** | 显著降低 block selection 开销，支持更大 batch size，提升吞吐量 |
| **延迟隐藏** | 通过 lookahead 预测实现 prefetch，有效掩盖 PCIe 传输延迟 |
| **精度保持** | 无需重新训练主干模型，仅微调 Forecast 投影即可匹配甚至超越原始稀疏模型精度 |
| **通用性** | 可插拔于已有的稀疏预训练模型（如 MiniCPM、NOSA） |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **Accuracy Benchmarks**：
  - **HELMET**：综合评测长上下文理解能力（回忆、RAG、ICL、摘要等）
  - **LongBench**：双语多任务长文本基准
  - **RULER**：合成任务用于测试极长上下文（最高达 128K）
  - **Reasoning Suite**：MATH-500、AIME 2024/2025，评估复杂推理能力

- **Training Data for Forecast Indexer**：
  - ProLong-64K [34]：用于训练 Forecast 模块的数据集

---

### **实验设置和评估指标**

#### **模型**
- **MiniCPM4.1-8B**（基于 InfLLM-V2 架构）
- **NOSA-8B**（在 InfLLM-V2 上增加 query-agnostic eviction head）

#### **硬件平台**
- **NVIDIA H100 GPU**（80GB HBM3，PCIe Gen5×16）
- **NVIDIA A100 GPU**（80GB HBM2e，PCIe Gen4×16）
- CPU 内存：2TB，页锁定（pinned）

#### **评估指标**
- **Throughput (tok/s)**：每秒处理 token 数量（Prefill 和 Decode 阶段分别测量）
- **Latency Breakdown**：分析 block selection 与 sparse attention 时间占比
- **Accuracy (%)**：各 benchmark 的平均得分
- **Feasible Batch Size**：单卡最大可运行 batch 大小（OOM 边界）

#### **Baseline 方法对比**
| 方法 | 类型 | 是否 Offload | 是否有 Prefetch |
|------|------|---------------|----------------|
| Dense | 全密集注意力 | 否 | - |
| Sparse | 原始稀疏注意力 | 是 | 否 |
| Sparse+ (no offload) | 稀疏无卸载 | 否 | - |
| InfiniGen [11] | 训练无关预取 | 是 | 是（基于隐藏状态） |
| SparDA (Ours) | 可学习解耦选择 | 是 | 是（基于 Forecast） |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **推理速度提升**
| 场景 | 提升幅度 | 说明 |
|------|----------|------|
| **Prefill Throughput** | 最高 **1.25×** 超越 Sparse | 主要来自 Forecast indexer 减少 selection 开销 |
| **Decode Latency** | 最高 **1.7×** 优于 Sparse Offload 基线 | 得益于 prefetch 与 compute 重叠 |
| **Decode Throughput** | 最高 **5.3×** 超越非卸载稀疏基线（Sparse+） | 因支持更大 batch size 实现吞吐飞跃 |

> 💡 在 128K 序列长度、batch=64 下，SparDA 实现 **1000.1 tok/s**，而 Sparse 仅为 **788.9 tok/s**

#### 🔹 **准确率表现**
- 在多个 benchmark 上 **持平或略优于原始稀疏模型**：
  - MiniCPM4.1-8B：平均准确率从 61.4 → **61.7**
  - NOSA-8B：平均准确率从 49.4 → **51.7**（+2.3）
- 特别是在 **Reasoning 任务上增益显著**（NOSA-8B +6.5）
- **InfiniGen 准确率明显下降**，验证了其依赖隐藏态相似性的假设不鲁棒

#### 🔹 **长度泛化能力**
- 在 RULER 上随序列延长（32K→128K），SparDA 表现持续领先：
  - NOSA-8B 在 128K 下差距扩大至 **+4.3**
  - 表明 Forecast 学习到更具泛化性的选择策略

---

### **消融实验结果**

#### ✅ **压缩窗口细粒度监督（Fine-grained Supervision）**
- 使用更小的压缩窗口 `(lc=2, sc=1)` 作为目标监督信号，能提供更高分辨率的评分图。
- 结果：RULER +3.0，Reasoning +2.2（MiniCPM），整体精度提升。

> 📌 表明“精细监督”有助于训练出更精准的 Forecast indexer。

#### ✅ **Prefetch CTA 分配策略**
- 动态调整用于 prefetch 的 Cooperative Thread Arrays (CTA) 数量：
  - 小 batch：16 CTAs 足够
  - 大 batch：32 或 64 CTAs 更优
- 自适应策略（<32 batch 用 16，否则 32）接近最优配置，吞吐损失 <4%

#### ✅ **Decode Speedup 分解（Table 10）**
| 方法 | B16 Decode (tok/s) | 来源 |
|------|--------------------|------|
| Sparse | 447.9 | baseline |
| SparDA (no prefetch) | 505.9 | 仅靠 selection 优化 |
| SparDA (full) | 705.3 | + selection + prefetch |

> 👉 验证：**selection 优化 + prefetch overlap 共同构成性能提升主因**，且 prefetch 对大 batch 至关重要。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **稀疏选择可以被建模为一个可学习、可调度的信号**，而非必须绑定在当前 Query 上。
2. ✅ **解耦 selection 与 attention** 可暴露未来内存访问模式，使系统提前行动（如 prefetch），从而将 sparsity 从“计算节省机制”升级为“调度友好接口”。
3. ✅ **轻量级 Forecast 模块（<0.5% 参数）即可实现显著加速与精度维持**，适合部署在已有稀疏模型上。
4. ✅ **异步 prefetch + persistent kernel 设计** 成功解决了 PCIe 带宽瓶颈，使大规模卸载变得实用。

---

### **方法的局限性**
- ❌ **不是独立的稀疏注意力方法**：SparDA 是附加组件，依赖已有稀疏 backbone（如 InfLLM-V2、DSA），不能单独使用。
- ❌ **精度受限于基础稀疏模型质量**：无法突破原始 sparse attention 的表达能力上限。
- ❌ **尚未扩展至 token-level sparse attention**：目前仅应用于 block-sparse attention，对 DeepSeek-V3/V4 中的 DSA/CSA 支持待研究。
- ❌ **模型规模限制**：实验集中在 8B 模型，更大模型（如 70B）上的效果未知。

---

### **未来工作方向**
1. 🔮 **将 SparDA 扩展至 token-level sparse attention**（如 DeepSeek-V3/V4 的 DSA/CSA 架构）
2. 🔮 **探索跨层 Forecast 的更深连接方式**（例如 multi-layer prediction）
3. 🔮 **结合 KV Cache 压缩技术**（如 SnapKV、H2O）进一步降低存储需求
4. 🔮 **在边缘设备或分布式系统中部署 SparDA**，验证其跨平台适用性

---

> ✅ **代码开源地址**：[https://github.com/NVlabs/SparDA](https://github.com/NVlabs/SparDA)

</details>

---

### 2. [LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding](https://arxiv.org/abs/2606.04302)

**Authors**: Haocheng Xia, Mihir Pamnani, Hanxi Fang, Supawit Chockchowwat, Yongjoo Park  
**Category**: cs.CL  
**Published**: 2026-06-04  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.04302v1  

#### Abstract
Key-value (KV) caching accelerates inference of large language models (LLMs) by reusing past computations for generated tokens. Its importance becomes even greater in long-context applications such as retrieval-augmented generation (RAG) and in-context learning (ICL). However, conventional KV cachin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在 **Retrieval-Augmented Generation (RAG)** 和 **In-Context Learning (ICL)** 等长上下文任务中，**Key-Value (KV) 缓存** 被广泛用于加速大语言模型（LLM）推理。然而，传统 KV 缓存将 **位置信息（positional encoding）直接嵌入缓存条目中**，导致其具有“位置感知”特性（position-aware），即同一个文档的缓存只能在相同位置复用。

这带来了两个严重问题：
- **低命中率**：若文档出现在不同位置，则无法复用，必须重新计算或复制缓存。
- **内存浪费**：为同一文档的不同位置创建多个副本，占用大量高带宽内存（HBM）。

现有解决方案如 Block-Attention 或 TurboRAG 虽支持跨位置复用，但需对 KV 缓存进行 **materialization（显式重编码）**，带来高昂的内存和带宽开销。

---

### 提出了什么新方法或新思路

本文提出 **LazyAttention**，一种全新的注意力机制，通过 **延迟位置编码（deferred positional encoding）** 实现 **零拷贝、位置无关的 KV 复用**。

#### 核心思想：
- **逻辑上解耦位置信息**：KV 缓存中只存储内容相关的 Key 和 Value，**不嵌入任何位置信息**。
- **运行时动态注入位置**：在注意力计算阶段，通过自定义的 **Triton 内核** 动态地、按需地将相对位置信息注入到注意力分数计算中。
- **内核级实现（kernelized）**：将 RoPE（Rotary Positional Embedding）的调整操作融合进 FlashAttention 风格的注意力内核中，避免额外的数据读写。

> 如图1所示，LazyAttention 将位置编码从“预处理阶段”推迟到“注意力计算内核内部”，实现了真正的 **position-agnostic caching**。

---

### 相比现有方法的优势

| 方面 | 传统方法（如 Block-Attention） | LazyAttention |
|------|-------------------------------|----------------|
| **KV 复用能力** | 仅限于前缀或特定位置 | 支持任意位置复用 |
| **内存效率** | 每个位置变体需独立副本 | 单份物理 KV 可服务多逻辑请求 |
| **带宽开销** | 需 Read-Modify-Write 操作 | 仅 Read-only，无额外写入 |
| **计算开销** | 高（重复旋转整个 K 矩阵） | 极低（<0.2% 开销） |
| **命中率提升** | 受限于位置碎片化 | 在 Zipf 分布下最高提升 7.5× |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

实验基于标准 RAG QA 基准，涵盖多种文档长度和推理复杂度：

- **2WikiMQA**：多段落问答，每段视为一个文档。
- **HotpotQA**：多跳推理，需跨文档推理。
- **TriviaQA**：长网页上下文阅读理解。
- **NarrativeQA**：基于小说/剧本的长文本问答。

此外还测试了：
- **Long-form Literature Review**（5篇 ArXiv 论文生成综述）
- **Few-shot Classification**（AG News 数据集上的非 RAG 场景）

---

### 实验设置和评估指标

#### 硬件环境：
- 主要平台：NVIDIA H100 96GB GPU（GH200）
- 对比平台：A100、A40（验证跨硬件鲁棒性）
- 模型：Tulu3-Block-FT（8B）、Llama-3.1-70B-Instruct、Qwen3-8B

#### 评估指标：
| 指标 | 含义 |
|------|------|
| **TTFT (Time-to-First-Token)** | 首个 token 生成时间，反映响应速度 |
| **Throughput (req/s)** | 每秒处理请求数，衡量吞吐量 |
| **VRAM Cache Hit Ratio** | KV 缓存命中率 |
| **End-to-End Latency** | 完整请求处理延迟 |
| **Exact Match (EM)** | 生成答案准确率 |

#### 流量模式：
- **Uniform**：文档访问均匀分布（低复用潜力）
- **Skewed (Zipf, α=2.1)**：少数热门文档高频访问（高复用场景）

---

### 基线方法对比

| 基线方法 | 简介 |
|---------|------|
| **Prefix Caching** | vLLM 默认前缀缓存，仅支持连续前缀复用 |
| **Prompt Cache** | 模块化注意力复用，固定长度缓存 |
| **CacheBlend** | 缓存融合策略，提升精度但增加重构成本 |
| **Block-Attention (vLLM)** | 当前最优块级缓存机制，需 materialize 位置调整 |
| **MEPIC-like** | 类似设计但逐 token 应用旋转，I/O 更高 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **TTFT 加速效果显著**
- 在 **Skewed 流量** 下：
  - LazyAttention 比 Block-Attention **快 1.37×**
  - 比 CacheBlend 快 1.43×（8B 模型）
- 在 **70B 大模型** 上优势更明显：
  - LazyAttention 达到 **5.2× TTFT 加速**（vs 标准 RAG）
  - 超过 CacheBlend 的 **1.53×**

> 表明 LazyAttention 特别适合大模型、高带宽瓶颈场景。

#### ✅ **吞吐量大幅提升**
- 在 Skewed 场景下，**推理吞吐提高 1.40×**
- 内存受限场景（10GB KV Pool）：
  - Throughput 从 0.42 → **0.80 req/s**（**+1.9×**）

#### ✅ **缓存命中率显著提升**
| KV Cache Size | LazyAttention (High Skew) | Block-Attention (vLLM) |
|---------------|----------------------------|--------------------------|
| 1 GB          | **13.57%**                 | 7.27%                   |
| 5 GB          | 20.49%                     | 18.23%                  |
| 10 GB         | 23.89%                     | 21.13%                  |
| No-limit      | **29.09%**                 | 27.38%                  |

> 在极端内存限制下，命中率接近翻倍；即使内存充足仍保持领先。

#### ✅ **运行时开销极低**
- **Prefilling 阶段**：额外 FLOPs 开销仅 **~0.59%**（M=128）
- **Decoding 阶段**：每 token 解码延迟增加 **<0.13%**
- 总体内核开销控制在 **~0.2%**

> 图5 显示累计延迟曲线几乎重合，证明无累积开销。

#### ✅ **生成质量无损**
| Dataset       | Block-Attention | LazyAttention |
|---------------|------------------|----------------|
| 2WikiMQA      | 71.4             | 70.7           |
| TriviaQA      | 72.1             | 73.0           |
| NarrativeQA   | 61.0             | 59.7           |
| HotpotQA      | 72.5             | 73.3           |
| **Average**   | **69.3**         | **69.2**       |

> 所有任务上输出质量与 Block-Attention 几乎一致，差异源于浮点精度而非算法错误。

---

### 消融实验结果

#### 🔹 Tile Size 敏感性分析（Table 9）
- 不同 prefill tile size（M=64/128/256）下，归一化吞吐变化不超过 **3%**
- 说明默认配置（M=128）已接近最优且鲁棒。

#### 🔹 长上下文扩展性（Table 6）
- 文档长度从 4K → 16K tokens：
  - TTFT Speedup 维持在 **~4.8–4.97×**
- 表明方法对长文档具有良好可扩展性。

#### 🔹 数值稳定性验证（Table 7）
- 序列长度达 **128K tokens** 时：
  - 注意力 logits 最大绝对误差 < `3.75e-5`
- 证明 on-the-fly 旋转不会引入数值漂移。

#### 🔹 与 MEPIC-like 对比（Table 12）
| 方法 | Prefill (ms) | Decode (+%) | KV Prep (ms) |
|------|-------------|------------|--------------|
| MEPIC-like | 376.48 | **+16%** | 23 |
| **LazyAttention** | 364.12 | **+0.67%** | 0 |

> LazyAttention 在 decoding 延迟和准备开销上全面优于同类设计。

---

## 4. 关键结论和发现

### 主要发现

1. **位置感知是 KV 缓存复用的根本瓶颈**  
   传统方法因绑定位置而造成严重资源浪费，尤其是在文档访问呈 Zipf 分布的真实场景中。

2. **延迟位置编码可在不牺牲准确性的前提下实现零拷贝复用**  
   LazyAttention 通过 kernel-level 融合 RoPE 计算，成功绕过 materialization 开销，同时保证数学等价性。

3. **单份 KV 条目可服务于任意逻辑偏移**  
   一个文档只需一份缓存即可被所有请求复用，极大提升缓存效率。

4. **性能增益在真实系统瓶颈处最为显著**  
   - 在 **memory-bandwidth-bound** 的 decoding 阶段，LazyAttention 避免了额外写入，优势随硬件带宽降低而放大（A40 上达 1.7× vs CacheBlend）。
   - 在 **large model** 上收益更大（70B > 8B），因其 KV states 更大，materialization 成本更高。

5. **通用性强，兼容多种架构**  
   支持 RoPE 家族变体（如 YaRN、interleaved RoPE）、GQA/MQA，并可与 Lego-Link 等训练无关复用策略结合。

---

### 方法的局限性

1. **依赖 RoPE 类相对位置编码**  
   当前设计基于 RoPE 的相对旋转性质，不适用于纯绝对位置编码（如 BERT-style learned positional embedding）。

2. **需要定制内核支持**  
   必须修改底层 attention kernel（如 FlashAttention），不能直接用于原生 Transformers 库。

3. **对 tile 策略有一定依赖**  
   虽然敏感性低，但仍需合理选择 tile size 以平衡计算与 I/O。

4. **目前仅支持 exact match 匹配**  
   文档匹配基于 token-level 完全匹配，未集成语义相似性检索。

---

### 未来工作方向

1. **扩展至其他位置编码方式**  
   探索 ALiBi、T5-relative bias 等 score-space 方法的 lazy 注入机制。

2. **支持语义级缓存匹配**  
   结合 embedding similarity 进行近似匹配，进一步提升命中率。

3. **集成到端到端 RAG 系统**  
   与 retriever 联动优化，构建统一的缓存感知检索-生成 pipeline。

4. **探索训练时缓存友好设计**  
   设计更适合 LazyAttention 的模型结构或微调目标。

5. **部署到边缘设备**  
   利用其低内存特性，在资源受限设备上实现高效 RAG。

---

> 💡 **总结一句话**：  
> **LazyAttention 通过将位置编码“懒惰地”推迟到注意力内核实现，打破了 KV 缓存的位置壁垒，在几乎零开销的前提下实现了任意位置的高效复用，为长上下文 LLM 推理提供了新的系统级优化范式。**

</details>

---

### 3. [FlexNPU: Transparent NPU Virtualization for Dynamic LLM Prefill-Decode Co-location](https://arxiv.org/abs/2606.04415)

**Authors**: Jiongjiong Gu, Jianfeng Wang, Zidong Han, Yongqiao Wang, Pengfei Xia, Mingjie Zhang, Hong Liu, Yuanyi Xia, Jiajia Chu, Yifeng Tang, Hui Zang, Xin Yao, Qijie Qiu, Yuzhao Wang, Chuanfei Xu, Lin Zhang, Zhuonan Lai, Hongming Huang, Jiawei Qiu, Gong Zhang, Zhong Ming, Weipeng Cao  
**Category**: cs.DC  
**Published**: 2026-06-04  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.04415v1  

#### Abstract
Modern AI serving increasingly relies on NPUs for conventional inference and large language model serving. However, current NPU deployments commonly expose physical devices directly to applications, which limits runtime control over scheduling and makes it difficult to adapt execution to phase-level...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《FlexNPU: Transparent NPU Virtualization for Dynamic LLM Prefill-Decode Co-location》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代AI服务广泛依赖NPU进行大语言模型（LLM）推理，但当前主流部署方式采用**直接Passthrough模式**，将物理NPU设备暴露给应用。这种模式虽然高效，但缺乏运行时调度控制能力，导致以下问题：
- **资源利用率低**：LLM推理分为 **prefill** 和 **decode** 两个阶段，其资源需求特性不同（prefill计算密集，decode内存带宽受限），静态分配易造成资源失衡。
- **静态策略不灵活**：无论是 **static PD co-location**（共置）还是 **static PD disaggregation**（分离），都难以适应动态变化的请求分布、输入/输出长度等负载特征。
- **缺乏透明调度层**：无法在不修改模型代码、框架或驱动的前提下实现细粒度的运行时干预。

### 🚀 提出的新方法
提出 **FlexNPU** —— 一种面向Ascend NPU的**透明用户态虚拟化层**，核心设计如下：
- **透明拦截 AscendCL API**：通过 `LD_PRELOAD` 注入客户端库，拦截AscendCL调用（如内存管理、stream创建、算子执行等），无需修改模型代码、AI框架或NPU驱动。
- **用户态Daemon代理调度**：所有NPU操作被转发至 per-device daemon，由其完成虚拟资源到物理资源的映射与调度。
- **支持动态PD共置（Dynamic PD Co-location）**：基于运行时监控（phase profile、queue状态、bandwidth压力等），动态调整prefill与decode的执行比例，实现互补资源利用。

### 🔍 相比现有方法的优势
| 维度 | 现有方法（如GPU方案） | FlexNPU |
|------|------------------------|--------|
| **平台适配性** | 多基于CUDA/MIG/SM级调度机制 | 面向Ascend NPU，基于AscendCL API层，通用性强 |
| **透明性** | 部分需修改框架或引入专用运行时 | 完全透明，兼容现有模型与框架 |
| **调度灵活性** | 多为静态分离或固定共置 | 支持**动态PD co-location**，按需调节执行权重 |
| **开销控制** | 可能引入高延迟RPC或数据拷贝 | 控制路径轻量，仅传递元数据，无张量复制 |
| **适用场景** | 多聚焦GPU或多租户隔离 | 聚焦LLM serving中的phase-aware优化 |

> ✅ **核心创新**：首次在Ascend平台上实现了**无需硬件支持、零侵入的运行时虚拟化调度层**，并用于解决LLM推理中prefill/decode的动态资源协调问题。

---

## 2. 核心实验方法和设置

### 📊 数据集与工作负载
使用两类典型LLM workload 进行评估：
1. **DeepSeek-R1**（MoE架构，大规模部署）
   - 使用 `gsm8k_gen_0_shot_cot_str_perf` 数据集生成推理请求
   - 量化方式：W8A8
2. **Qwen2.5-7B**（dense架构，中小规模）
   - 测试多种输入/输出组合以覆盖不同负载模式

### ⚙️ 实验设置
- **硬件平台**：华为 Ascend 910C NPU
- **集群规模**：
  - DeepSeek-R1：384卡 CloudMatrix384 超节点
  - Qwen2.5-7B：单节点多卡部署
- **软件栈**：标准 CANN + AscendCL，vLLM作为serve框架
- **FlexNPU实现**：用户态client + daemon，通过共享内存通信

### 📈 评估指标
| 指标 | 含义 |
|------|------|
| **Throughput (tokens/s 或 RPS)** | 系统吞吐量，衡量整体服务能力 |
| **TTFT (Time to First Token)** | 用户感知延迟起点，反映prefill响应速度 |
| **TPOT (Time Per Output Token)** | decode阶段token生成延迟，影响交互流畅性 |
| **Relative Improvement** | 相对于baseline的提升百分比 |

### 🔀 基线方法对比
| 实验组 | 对比基线 |
|-------|---------|
| **FlexNPU vs. Passthrough** | Native NPU Passthrough（直通） |
| **FlexNPU vs. Static PD Disaggregation** | 固定拆分prefill/decode实例（如6P2D） |
| **FlexNPU vs. Static PD Co-location** | 共享资源但无动态调度的共置部署 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### （1）虚拟化开销测试（vs. Passthrough）
| 配置 | 总吞吐量 (tokens/s) | 相对性能 |
|------|--------------------|----------|
| Native Passthrough | 977.69 | 1.0000x |
| FlexNPU Proxy | 988.27 | **1.0108x** (+1.08%) |

> ✅ **结论**：FlexNPU不仅没有引入可测延迟，反而因异步代理机制略微提升了吞吐（+1.08%），证明其**低开销设计有效**。

---

#### （2）FlexNPU vs. Static PD Disaggregation（DeepSeek-R1, 384卡）

| 工作负载 | 部署方式 | 总吞吐量 (RPS) | 提升幅度 |
|--------|----------|---------------|----------|
| 1K-in / 1K-out | Static Disaggregation | 489.84 | — |
| | FlexNPU Dynamic Co-location | **618.18** | **+26.33%** |
| 1K-in / 4K-out | Static Disaggregation | 146.63 | — |
| | FlexNPU Dynamic Co-location | **154.18** | **+5.15%** |

> ✅ **结论**：在两种典型负载下均显著优于静态分离策略，尤其在均衡负载（1K-1K）中提升高达 **26.33%**，说明动态调度能更好平衡资源瓶颈。

---

#### （3）FlexNPU vs. Static PD Co-location（Qwen2.5-7B）

| 输入/输出长度 | 部署方式 | 吞吐量 (tokens/s) | TTFT (ms) | TPOT (ms) |
|-------------|----------|------------------|-----------|-----------|
| 256/1024 | Static Co-location | 195.275 | 488,099 | 20.35 |
| | FlexNPU Dynamic Co-location | 194.250 | **331.5** | 20.53 |
| **改善** | — | -0.52% | **↓99.93%** | +0.90% |

> ✅ **结论**：
> - 吞吐量基本持平（略有波动但在误差范围内）
> - **TTFT从近500秒降至0.3秒以内，降低超99%**
> - TPOT几乎不变（±3%内），保证了生成质量

---

### 🔍 消融分析（隐含于文中）
- **调度策略有效性**：通过Figure 5和6展示不同time-slice ratio下的吞吐曲线，验证了“过度分配decode资源无法提升吞吐”、“prefill成为瓶颈时应优先保障”的调度逻辑。
- **无需硬件分区支持**：FlexNPU可在无MIG-like硬件隔离机制的Ascend平台上运行，依赖用户态stream调度即可实现效果。
- **轻量控制路径是关键**：避免tensor复制、使用共享内存通信、只拦截必要API，确保了低延迟。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **透明虚拟化可行且高效**  
   在AscendCL层级进行API拦截，可以在**零修改应用代码**的前提下实现NPU资源的虚拟化与调度控制，且**无可观测推理延迟增加**。

2. **Dynamic PD Co-location 显著优于静态策略**  
   - 相比 **static PD disaggregation**，FlexNPU通过动态调配执行时间片，在资源利用上更均衡，**最大提升达26.33%**。
   - 相比 **static PD co-location**，FlexNPU通过进程级分离与调度，解决了head-of-line阻塞问题，**TTFT降低超过92%，最高达99.9%**，极大提升用户体验。

3. **互补资源特性可被有效利用**  
   prefill（计算密集）与decode（带宽受限）存在天然互补性。FlexNPU通过runtime profiling识别饱和点（如decode带宽饱和后加算力无效），从而将空闲算力用于prefill，提高整体goodput。

4. **适用于多种模型与部署规模**  
   在大型MoE模型（DeepSeek-R1）和小型dense模型（Qwen2.5-7B）上均取得显著收益，表明该方法具有良好的泛化能力。

---

### ⚠️ 局限性
- **未实现强隔离**：目前依赖用户态调度，缺乏硬件级资源隔离（如AI Core partitioning），多租户场景下可能存在干扰风险。
- **调度策略较简单**：基于启发式规则与轻量profile，尚未集成复杂SLO-aware或预测性调度算法。
- **仅支持Ascend平台**：虽针对Ascend定制，但思想可迁移至其他NPU/GPU平台。

---

### 🔮 未来工作方向
1. **增强调度智能性**：引入机器学习模型预测phase行为，实现更精准的动态调度。
2. **支持SLO感知与多租户调度**：结合QoS目标，实现优先级调度、抢占、弹性扩缩容。
3. **集成硬件隔离机制**：当Ascend平台支持更细粒度资源划分（如stream priority groups）时，进一步提升调度效率与隔离性。
4. **扩展至更多AI workload**：探索在CV、推荐系统等任务中是否也能受益于类似的透明虚拟化调度。

---

> 💡 **总体评价**：  
> FlexNPU 是一个极具实用价值的系统工作，它在不牺牲性能的前提下，为Ascend NPU带来了**运行时可控性**这一关键能力。其提出的 **“透明虚拟化 + 动态PD co-location”** 范式，为下一代高效、响应式LLM serving系统提供了新的构建基础。

</details>

---

### 4. [AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2606.04484)

**Authors**: Qingxu Fu, Boyin Liu, Shuchang Tao, Zhaoyang Liu, Bolin Ding  
**Category**: cs.AI  
**Published**: 2026-06-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.04484v1  

#### Abstract
We present AgentJet, a distributed swarm training framework for large language model (LLM) agent reinforcement learning. Unlike centralized frameworks that tightly couple agent rollouts with model optimization, AgentJet adopts a decoupled multi-node architecture in which swarm server nodes host trai...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AgentJet: A Flexible Swarm Training Framework for Agentic Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Agentic Reinforcement Learning (Agentic RL)** 框架在训练大型语言模型（LLM）作为自主智能体时面临以下关键挑战：
- **运行时脆弱性 (Runtime Fragility)**：代理执行环境（如浏览器自动化、代码沙箱）的失败会中断整个训练流程。
- **调试摩擦大 (Debugging Friction)**：修改代理逻辑或奖励函数需重启整个训练过程，迭代周期长达数分钟。
- **多模型支持不足 (Multi-Model Constraints)**：主流框架仅支持单个策略模型，难以实现异构多智能体系统（如不同规模模型协作）。
- **冗余上下文开销 (Redundant Context)**：长回合交互中重复的系统提示、工具定义等导致计算浪费。
- **任务锁定 (Environment Lock-in)**：多任务联合训练因依赖冲突而难以实现强隔离。

### 提出的新方法与架构
作者提出 **AgentJet** —— 一种基于**去耦合多节点架构**的分布式 **Swarm Training Framework**，其核心创新如下：

#### （1）Swarm 架构（Client-Server 范式）
- **Swarm Server (优化器节点)**：部署在 GPU 集群上，负责模型存储、梯度更新、vLLM/SGLang 推理服务及 episode 生命周期管理。
- **Swarm Client (采样节点)**：轻量级 CPU 进程，可在任意设备运行，执行代理逻辑、调用外部环境、生成轨迹并计算奖励。
- **完全解耦**：训练与推理平面分离，客户端可动态加入/退出，实现故障容忍与热插拔调试。

#### （2）Timeline Merging 上下文压缩算法
- 自动合并具有前缀重叠的多轮对话轨迹，消除冗余上下文。
- 支持 **token-level** 和 **text-level** 匹配策略，平衡训练效率与推理一致性。
- 实现 **1.5–10× 的训练加速**，尤其适用于长回合任务（如 AppWorld）。

#### （3）灵活的样本批处理机制（Episode Batching）
提供五种触发策略控制策略更新时机：
- `C1`：累计 episode 数量达标
- `C2`：完成的任务种类数量达标
- `C3`：非哑任务（reward variance > 0）数量达标
- `C4/C5`：所有/任一客户端同意同步

#### （4）自动化研究系统（A3R: Alpha Auto Research）
- 全自动执行多日、多阶段 RL 研究实验。
- 支持超参数搜索、模型规模比较、LoRA 配置分析等复杂研究流程。
- 采用 **Leader-Worker 架构**，实现自适应探索与结果分析闭环。

#### （5）框架无关性（Framework-Agnostic）
- 通过 **OpenAI 兼容接口** 拦截任意代理框架（LangChain、AgentScope、CrewAI 等）的 LLM 请求。
- 支持黑盒代理训练，无需修改原始代码。

### 相比现有方法的优势
| 特性 | AgentJet | 现有框架（如 OpenRLHF, veRL, Forge） |
|------|--------|-------------------------------|
| 故障容忍 | ✅ 客户端崩溃不影响服务器 | ❌ 单点故障导致全链路中断 |
| 多模型支持 | ✅ 异构模型独立训练（非共享参数） | ❌ 主要支持单一模型 |
| 调试效率 | ✅ 秒级热更新客户端代码 | ❌ 重启耗时 5–10 分钟 |
| 上下文效率 | ✅ Timeline Merging 加速 1.5–10× | ❌ 无专门优化 |
| 多任务训练 | ✅ Cocktail Training 实现任务混合 | ❌ 依赖冲突难解决 |
| 自动化研究 | ✅ 内建 A3R 模块支持无人值守实验 | ❌ 无此能力 |

---

## 2. 核心实验方法和设置

### 数据集与任务
| 任务 | 描述 | 使用数据集/环境 |
|------|------|----------------|
| **Werewolves RPG** | 社会推理游戏，测试多智能体协作与欺骗能力 | 自定义规则模拟器 |
| **Hierarchical Academic Translation** | 学术摘要英译中，三阶段流水线（初翻 → 审校 → 修正） | 英文论文摘要（人工构造） |
| **AppWorld** | 交互式编码任务（如邮件管理、音乐播放） | AppWorld Benchmark（Trivedi et al., 2024） |
| **AIME** | 数学推理任务（类似 MATH） | AIME-2024/2025/DAPO-Math-Tiny-Val |

### 实验设置
- **模型**：Qwen3 系列（7B, 14B, 32B, 235B）、Qwen2.5-7B/14B-Instruct
- **算法**：GRPO、PPO、DAPO
- **硬件**：8-GPU 节点（FSDP），部分实验使用多服务器集群
- **训练步数**：通常 60–160 步，部分自动化实验持续数天
- **评估指标**：
  - 成功率（Success Rate, SR）
  - Pass@1 / Pass@2（数学任务）
  - 平均奖励（Mean Reward）
  - 训练时间（Wall-clock Time）
  - 政策熵（Policy Entropy）

### 基线方法对比
- **Single-task Specialist**：单独训练每个任务的专用模型
- **Two-stage OPD (On-Policy Distillation)**：先训多个专家，再蒸馏成一个通用模型
- **Monolithic Training**：传统集中式训练框架（隐含对比）
- **Manual Research Workflow**：人类主导的实验设计流程（用于对比 A3R）

---

## 3. 主要实验结果和性能指标

### （1）Werewolves 游戏：共享参数 vs 非共享参数训练
| 实验 | 可训练角色 | 初始 SR | 最终 SR | 提升 |
|------|------------|--------|--------|-----|
| Exp 1 (7B) | WW | 23.0% | 47.2% | +24.2% |
| Exp 2 (14B) | WW | 40.9% | 64.7% | +23.8% |
| Exp 3 | Seer | 38.5% | 46.5% | +8.0% |
| Exp 6 | sr+wt+ht | 22.9% | 35.9% | +13.0% |
| Exp 7 | vl+sr+wt+ht | 23.9% | 41.6% | +17.7% |

> ✅ **发现**：狼人阵营更易训练；联合训练多个特殊村民角色可恢复大部分收益。

#### 非共享参数训练优势（Table 2）
| 配置 | 初始 SR | 最终 SR |
|------|--------|--------|
| 共享参数（Table 1 Exp 2） | 40.9% | 64.7% |
| 非共享参数（Table 2 Exp 3） | 40.8% | **66.5%** |

> 🔺 **提升 1.8%**：独立参数带来行为多样性，打破语言模式相关性，增强欺骗性。

---

### （2）学术翻译任务（Table 3）
- **基础模型失败案例**：
  - 中英文混杂输出
  - 缩写未展开（如 “QNVB” 未解释）
  - 第一人称“we”未替换为“本文”
- **微调后表现**：
  - 正确展开缩写：“Quasi-Newton Variational Bayes (QNVB)”
  - 替换主语：“本文引入了……”而非“我们引入了……”
  - 准确翻译术语：“前置星核” → “prestellar core”

> ✅ 显著提升指令遵循与学术风格一致性。

---

### （3）多任务 Cocktail Training（Figure 7）
- **设置**：Qwen3-8B 同时训练 AppWorld + AIME，batch=16+16
- **对比**：专用模型（batch=32）

| 任务 | Cocktail Training | Dedicated Specialist | 差距 |
|------|------------------|-----------------------|-----|
| AIME | 0.75 | 0.80 | -0.05 |
| AppWorld | 0.58 | 0.68 | -0.10 |

> ⚠️ **结论**：Cocktail 训练在 AppWorld 上存在明显性能损失（因短回合 AIME 稀释长回合梯度），但在 AIME 上接近专用模型。

> 💡 **价值**：以轻微性能代价换取**统一通用模型**与**显著降低训练成本**（避免 N 次独立训练）。

---

### （4）Timeline Merging 性能增益（Figure 8）
| 指标 | 无 Merging | 有 Merging | 加速比 |
|------|-----------|------------|--------|
| 平均每步训练时间 | 2160 ± 171 s | **346 ± 13 s** | **6.25×** |
| LLM 调用次数 | 12.6 ± 1.0 | 11.4 ± 0.7 | ≈ |
| 最终奖励 | 0.25 | 0.25 | ≈ |

> ✅ **6.25× 加速**，且不损害训练质量。

---

### （5）框架无关性验证（Figure 10）
使用四种不同框架驱动相同 GRPO 训练：
- OpenAI SDK
- LangChain
- AgentScope
- Raw HTTP

| 框架 | 最终评估奖励（平均） |
|------|--------------------|
| OpenAI SDK | 0.536 |
| LangChain | 0.542 |
| AgentScope | 0.517 |
| Raw HTTP | 0.525 |
| **最大差距** | **0.025** |

> ✅ 不同框架间性能差异极小，验证了 **framework-agnostic** 设计的有效性。

---

### （6）自动化研究案例：最小稳定 batch size（Table 4）
目标：找出 AIME 任务下 Qwen3-8B 的最小高效 batch size，并探究 `max_response_length` 影响。

#### 关键发现：
1. **mr=10000 时**：
   - batch_size < 16 时性能下降
   - **最小稳定高性能 batch_size = 16**
   - 更大 batch（32/64）无增益，反而浪费资源

2. **mr=12000 时**：
   - 所有配置性能提升
   - **最优配置：bs=16, mr=12000** → Pass@1 达 **60.00%**
   - 高效下限提升至 **bs=4**

3. **mr=8000 时**：
   - 性能大幅下降（-13 至 -18 pts），不推荐

> 📌 **最终建议**：
- 最低成本稳定配置：`bs=4, mr=12000`
- 全局最佳配置：`bs=16, mr=12000`

---

## 4. 关键结论和发现

### 主要发现
1. **Swarm 架构有效解耦训练与执行**，实现了：
   - 故障容忍、热更新、异构多模型训练
   - 多任务 Cockail Training 的可行性
   - 研究者友好的 REPL 式开发体验

2. **Timeline Merging 显著提升训练效率**（最高 10×），是长回合任务的关键优化。

3. **非共享参数训练优于共享参数**，尤其在需要行为多样性的场景（如社交欺骗游戏）。

4. **Cocktail Training 是构建多技能通用模型的经济选择**，虽略逊于专用模型，但节省大量计算与工程成本。

5. **自动化研究系统 A3R 可复现人类研究员的核心探索流程**，支持无人值守的多日实验。

### 局限性
- **当前侧重基础设施层**，未提出新的 RL 算法。
- **Timeline Merging 在 token-level 严格模式下可能无法合并**，牺牲部分效率以保证一致性。
- **A3R 依赖高质量 prompt 与稳定 agent 实现**，若 leader agent 出错可能导致整个研究偏航。
- 对极端低 batch size 的稳定性机制仍待深入研究。

### 未来工作方向
- 扩展至更多模态（视觉、语音）的 agentic training。
- 结合在线学习与 continuous policy optimization。
- 构建开放社区驱动的自动化研究平台。
- 探索更高效的跨任务梯度融合机制，缩小 cocktail training 与专用模型的差距。

---

> 🔗 **开源地址**：[https://github.com/modelscope/AgentJet](https://github.com/modelscope/AgentJet)  
> 🌐 **基准测试平台**：[https://benchmark.agentjet.top](https://benchmark.agentjet.top)

</details>

---

### 5. [BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization](https://arxiv.org/abs/2606.04807)

**Authors**: Saket Reddy, Ke Yang, ChengXiang Zhai  
**Category**: cs.AI  
**Published**: 2026-06-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.04807v1  

#### Abstract
Mitigating social bias in Large Language Models (LLMs) presents a distinct alignment challenge: unlike verifiable tasks, bias lacks a single ground truth, creating a high-variance, subjective reward landscape. Previous preference-based fine-tuning methods have major trade-offs: Direct Preference Opt...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在预训练阶段会从大规模文本语料中继承社会偏见（social bias），如种族、性别、社会经济地位等方面的刻板印象和歧视性态度。这类偏见缺乏单一“正确答案”，导致其奖励信号具有**高方差、主观性强**的特点，使得传统的偏好对齐方法（如 DPO 和 PPO）面临以下挑战：

- **DPO（Direct Preference Optimization）**：依赖静态的偏好对数据进行离线训练，缺乏探索能力，泛化性差。
- **PPO（Proximal Policy Optimization）**：依赖独立的 critic model 进行优势估计，在噪声大、主观性强的偏见场景下容易产生不稳定的训练动态。

### 🚀 提出的新方法：BiasGRPO
本文提出 **BiasGRPO**，一个基于 **Group Relative Policy Optimization (GRPO)** 的框架，用于在高方差奖励景观中稳定地缓解 LLM 中的社会偏见。

#### 核心思想：
- 利用 GRPO 替代传统 RLHF 方法中的 critic model。
- 对每个 prompt，策略模型生成一组（group）完成文本（例如 G=4）。
- 将该组内各 completion 的 reward 减去组均值并标准化，作为其优势函数（advantage），即：
  
  $$
  A_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
  $$

- 通过这种**组内相对归一化**的方式，消除了对不可靠 critic 的依赖，同时保留了在线探索的能力。

### 🔍 相比现有方法的优势
| 方法 | 探索能力 | 训练稳定性 | 是否需要 Critic | 适用性 |
|------|----------|------------|------------------|--------|
| DPO  | ❌ 离线训练 | ✅ 高 | ❌ 不需要 | 泛化差 |
| PPO  | ✅ 在线训练 | ❌ 易受 critic 影响 | ✅ 需要 | 易不稳定 |
| **BiasGRPO** | ✅ 在线探索 | ✅✅ 更高（无 critic） | ❌ 不需要 | **更适合高方差、主观任务** |

此外，作者还发布了：
- 一个跨 **11 个领域**（种族、性别、残疾、年龄等）的多样化合成数据集；
- 一个轻量级（仅 0.1B 参数）、高效且避免知识退化的 **custom bias reward model**，可无缝集成到多目标 RLHF 流程中。

---

## 2. 核心实验方法和设置

### 📚 数据集
构建了一个包含 **20,999 条 prompt** 的综合数据集，来源如下：
- **BiasDPO**（10,000 条）：原始偏见探测问题 + 作者**合成了额外 8,855 条**，扩展至 11 个领域（如 Disability, Socioeconomic Status, Intersectionality 等）。
- **Civil Comments**（10,000 条）：社交媒体评论，涵盖不同毒性等级，用于测试中性或轻微提示下的偏见激发。
- **UnQover**（999 条）：模糊情境下的偏见诱导问题，答案本应为“无法确定”。

所有数据均配有 favorable / unfavorable completion 对，用于 DPO 训练；而 GRPO/PPO 仅需 prompt 和 reward model。

### ⚙️ 实验设置
- **基础模型**：Microsoft 的 **Phi-2（2.7B）**，未经过 RLHF 或任何偏见缓解微调，确保是“干净起点”。
- **训练方式**：
  - 所有方法训练 **3 个 epoch**，初始学习率 $10^{-6}$，线性衰减。
  - GRPO 使用 group size = 4（默认），后续进行消融研究（G=2~8）。
- **评估指标与基准**：
  - **BOLD** ↓：衡量生成文本中的代表性伤害（representational harm），越低越好。
  - **RealToxicityPrompts (RTP)** ↓：衡量显性敌意（overt hostility），越低越好。
  - **BBQ** ↑：衡量隐性刻板印象（implicit stereotyping），特别是在模糊情境下选择“无法确定”的准确率，越高越好。
  - **TruthfulQA** ↑：衡量事实准确性，防止因去偏而导致**知识退化**（knowledge degradation）。

### 🆚 基线方法对比
- **DPO**（使用 Identity Preference Optimization 变体）
- **PPO**
- **GRPO**（本文提出的 BiasGRPO 框架）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 2）

| Benchmark | Category | Base | DPO | PPO | **GRPO** |
|----------|--------|------|-----|-----|---------|
| **BOLD** ↓ | All | 0.0293 | 0.0222 | 0.0268 | **0.0140** |
| **RTP** ↓ | — | 0.0282 | 0.0234 | 0.0262 | **0.0198** |
| **BBQ** ↑ | All | 0.2750 | 0.2823 | 0.2996 | **0.3123** |
| **TruthfulQA** ↑ | — | 0.3843 | 0.3941 | 0.3929 | **0.3941** |

> ✅ GRPO 在所有偏见相关指标上表现最佳，且未牺牲 TruthfulQA 性能。

### 📈 与基线方法对比结果
- **相比 DPO**：
  - GRPO 在 BOLD 上降低超过 **40%**（0.0293 → 0.0140），显著优于 DPO 的 24% 下降。
  - DPO 曲线早期就趋于平缓（图3），说明其泛化能力受限于固定数据集。
- **相比 PPO**：
  - PPO 表现波动剧烈（图3中 jagged 曲线），标准差高达 0.1434（图2），反映 critic 估计不稳定。
  - GRPO 的 reward 标准差仅为 **0.0668**，不到 PPO 的一半，训练更平稳。

### 🔬 消融实验结果

#### （1）不同 group size 的影响（Table 5）
| Group Size | BOLD ↓ | RTP ↓ | BBQ ↑ | TruthfulQA ↑ |
|-----------|--------|-------|-------|-------------|
| G=2       | 0.0243 | 0.0242 | 0.2781 | 0.3868 |
| G=4       | 0.0140 | 0.0198 | 0.3123 | 0.3941 |
| G=8       | **0.0124** | **0.0115** | **0.3781** | **0.4137** |

> ✅ 增大 group size 能持续提升性能，尤其在 BBQ 和 TruthfulQA 上效果明显。  
> ❗ G=2 性能远低于 G=4，证明**组内相对比较机制本身是关键**，而非仅仅是“在线探索”。

#### （2）不同 reward model 的对比（Table 4）
使用第二好的人类标注 reward model（stereotype scoring model）替代自定义模型：
- 即使使用次优 reward model，GRPO 仍大幅优于 base model。
- 自定义 reward model 在多数指标上更优，验证其有效性。

#### （3）在其他模型上的泛化性（Table 10）
在 **Llama 3.2 (3B)** 上复现实验：
- GRPO 在 BOLD 和 RTP 上依然最优。
- BBQ 上所有方法表现下降，可能因 UnQover 数据量不足。

#### （4）Online DPO 对比（Table 7）
引入 Online DPO（允许模型自行生成响应）作为对照：
- 其性能接近 G=2 的 GRPO，但仍显著弱于标准 GRPO。
- 再次证明：**性能增益来自 group-relative normalization，而非单纯的在线探索**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **GRPO 特别适合高方差、主观性的偏见缓解任务**：
   - 组内相对优势计算提供了更稳定、清晰的学习信号。
   - 即使所有生成结果都带有偏见，也能识别出“相对最不偏见”的输出并给予正向反馈。

2. **BiasGRPO 在多个维度上全面超越 DPO 和 PPO**：
   - 显著降低 BOLD 和 RTP 分数（减少偏见与毒性）；
   - 提升 BBQ 正确率（更好处理模糊情境）；
   - 保持甚至略微提升 TruthfulQA 表现，**无知识退化**。

3. **模块化设计便于推广**：
   - 发布的数据集和轻量 reward model 可被直接用于其他 RLHF 流程，无需额外计算开销。

4. **group size 是关键超参数**：
   - G ≥ 4 才能充分发挥 GRPO 的潜力；
   - 太小的 group（如 G=2）退化为类似 pairwise ranking，失去优势。

### ⚠️ 局限性
- 实验集中在 **3B 规模模型**（Phi-2 和 Llama 3.2），尚未验证在更大模型（如 70B+）上的表现。
- 合成数据虽经 Vendi Score 验证语义多样性，但仍可能存在分布偏差。
- 当前 group size 固定，未来可探索 adaptive group sizing 策略。

### 🔮 未来工作方向
- 将 BiasGRPO 应用于更大规模、更强的 LLM。
- 探索动态调整 group size 的策略，以平衡效率与性能。
- 将该框架整合进多目标对齐系统（multi-objective RLHF），与其他目标（如帮助性、诚实性）协同优化。
- 扩展 reward model 以覆盖更多偏见维度（如文化偏见、地域歧视等）。

---

> 💡 **总结一句话**：  
> **BiasGRPO 通过 GRPO 的组内相对优势机制，在无需 critic 的前提下实现了比 DPO 更强的泛化能力和比 PPO 更高的训练稳定性，成为当前高方差偏见缓解任务中最有效的 preference-based 微调框架之一。**

</details>

---

### 6. [Graph Traversal on Tensor Cores: A BFS Framework for Modern GPUs](https://arxiv.org/abs/2606.05081)

**Authors**: Deniz Elbek, Kamer Kaya  
**Category**: cs.DC  
**Published**: 2026-06-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.05081v1  

#### Abstract
Modern GPUs have Tensor Cores (TCs) capable of extremely high-throughput matrix operations, yet graph algorithms remain difficult to accelerate because of their irregular and data-dependent execution patterns. This work presents BLEST, a TC-accelerated framework that reformulates Breadth-First Searc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Graph Traversal on Tensor Cores: A BFS Framework for Modern GPUs

## 1. 主要贡献和创新点

### 解决的问题
现代 GPU 虽然配备了高吞吐量的 **Tensor Cores (TCs)**，但图算法因其不规则（irregular）和数据依赖性强的执行模式，难以有效利用这些硬件单元。特别是 **Breadth-First Search (BFS)** 这类经典图遍历算法，在传统实现中面临严重的**负载不平衡、内存效率低下和同步开销**等问题。

本文旨在解决如何在现代 GPU 上，通过重新设计 BFS 算法，充分挖掘 Tensor Cores 的潜力，从而实现前所未有的性能提升。

### 提出的新方法与创新
作者提出了名为 **BLEST** 的全新框架，其核心创新点如下：

- **Binarized Virtual Slice Sets (BVSS)**：
  - 一种新颖的图数据结构，将图按列分片，并进一步划分为“虚拟切片集”（Virtual Slice Sets），以实现**近完美的跨 warp 负载均衡**。
  - 仅调度与当前 frontier 相关的图区域，避免了无用计算。

- **优化的 TC 计算布局**：
  - 设计了一种高效的 **m8n8k128 MMA 指令映射方案**，将邻居检查操作精确地映射到 TC 的 `popc` 和 `AND` 操作上。
  - 该布局**减少了 8× 的 MMA 调用次数**，极大提升了 TC 利用率。

- **Lazy Vertex-Update 机制**：
  - 引入异步原子操作（`REDG`）延迟顶点状态更新，减少对 L1/L2 缓存的争用和原子冲突。
  - 将更新操作集中处理，显著改善缓存局部性。

- **动态切换机制 (Dynamic Switching)**：
  - 重新审视了 BFS 中的 “direction switching” 概念，在 TC 时代引入了 **TC 与 CUDA Cores 之间的动态切换**。
  - 当满足条件 `#unvisited < n * |Q_curr|` 时，自动从 TC 模式切换到基于 CUDA Cores 的 bottom-up 探索，以适应不同层级的 frontier 大小。

- **可扩展的图重排序方法**：
  - 对于 scale-free-like 图，提出 **JACCARDWITHWINDOWS** 算法，利用窗口内 Jaccard 相似性进行重排序，提高 BVSS 压缩率。
  - 对于其他图（如道路网络），采用 **Reverse Cuthill-McKee (RCM)** 减少带宽，提升缓存命中率。

### 相比现有方法的优势
- **性能更高**：相比当前最先进的实现，实现了数量级的加速。
- **负载更均衡**：BVSS 结构从根本上解决了 warp 间的负载不均问题。
- **资源利用率更高**：优化的 TC 布局和 lazy update 显著降低了冗余计算和内存瓶颈。
- **灵活性更强**：支持单源、多源 BFS 及 Closeness Centrality 等复杂应用。

---

## 2. 核心实验方法和设置

### 数据集
实验使用了两个基准套件中的 14 个真实世界图数据集：
- **GAP Benchmark Suite**：包括 `GAP-road`, `GAP-twitter`, `GAP-web`, `GAP-kron`, `GAP-urand` 等。
- **自定义大型图集**：来自 SuiteSparse 数据库，筛选标准为节点数 ≥ 2³⁰ 且边数 ≤ 2³²⁻¹，例如 `com-Friendster`, `uk-2005`, `webbase-2001` 等。

### 实验设置
- **硬件平台**：
  - **Arch-1**：Intel Xeon Gold 6548Y+, 配备 **NVIDIA H200 GPU (141GB HBM3e)**。
  - **Arch-2**：双 Intel Xeon Platinum 8460Y+, 配备 **NVIDIA H100 GPUs (64GB HBM2e)**，用于大规模实验（如 Closeness Centrality）。
- **软件环境**：C++/CUDA，编译器为 gcc 12.3.0 和 CUDA 13.0。
- **评估指标**：
  - 单次 BFS 执行时间（毫秒）
  - 平均加速比（speedup）
  - 内存占用（Memory Footprint）
  - 预处理开销

### 基线方法对比
- **GAP**：经典的 CPU/GPU BFS 实现。
- **Gunrock**：基于 frontier 抽象的高性能 GPU 图分析库。
- **GSWITCH**：支持动态策略选择的 GPU 图处理框架。
- **BerryBees [15]**：首个利用 TC 加速 BFS 的工作，是本文最直接的比较对象。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在广泛的图数据集上，**BLEST 相比基线方法取得了显著的平均加速比**：
  - 相比 **GAP**：**22.0×**
  - 相比 **Gunrock**：**7.7×**
  - 相比 **GSWITCH**：**8.1×**
  - 相比 **BerryBees**：**5.9×**

- **多源 BFS (MS-BFS)** 性能：
  - BLEST 的 MS-BFS 实现比其自身的单源版本平均快 **2.7×**。
  - 相比现有的 CPU 实现（64核），平均快 **25.9×**。

- **Closeness Centrality 计算能力**：
  - 使用 **100 台 H100 GPU**，在约 **1 小时（3,665 秒）** 内完成了对 **com-Friendster** 图（6560万顶点，36亿条边）的**精确 Closeness Centrality** 计算。
  - 这是迄今为止最大规模的精确中心性计算之一。

### 消融实验结果（Ablation Study）
通过逐步添加优化模块验证各组件贡献：

| 优化阶段 | 描述 | 相比 BerryBees 的平均加速比 |
|--------|------|---------------------|
| (A) | BVSS + Kernel Fusion | 1.6× |
| (AB) | (A) + Optimal TC Layout | ~1.9× |
| (ABC) | (AB) + Reordering | ~2.5× |
| (ABCD) | (ABC) + Lazy Update | ~3.9× |
| (Full) | (ABCD) + Switching | **5.9×** |

- **最优 TC 布局**使 MMA 调用减少 8×，带来稳定约 1.2× 的加速。
- **Lazy Update** 和 **Switching** 在特定图（如 scale-free-like）上效果尤为显著。
- **图重排序**对部分图有帮助，但对已有良好自然顺序的图可能无效甚至轻微降速。

---

## 4. 关键结论和发现

### 主要发现
1. **Tensor Cores 可以高效用于不规则图算法**：通过精心设计的数据结构（BVSS）和计算映射，可以克服图算法的不规则性，充分发挥 TC 的高吞吐优势。
2. **负载均衡是关键**：BVSS 结构实现了近乎完美的 warp 级负载均衡，是性能提升的基础。
3. **软硬件协同设计至关重要**：结合 lazy update、动态切换、图重排序等技术，才能全面消除内存、同步和计算瓶颈。
4. **精确的大规模图分析成为可能**：BLEST 的高性能使得在合理时间内完成 O(nm) 复杂度的精确 Closeness Centrality 成为现实。

### 方法的局限性
- **切换阈值是 GPU 特定的**：文中使用的常数 `n=10` 是针对 Hopper 架构（H100/H200）校准的，在其他 GPU 架构上可能需要重新调优。
- **预处理开销存在**：构建 BVSS 和图重排序需要额外时间，虽然是一次性的，但对于频繁变更的图不友好。
- **内存消耗随并发度增长**：在多源 BFS 中，`Levels` 数组的内存占用与并发 BFS 数量 K 成正比，可能导致 OOM。

### 未来工作方向
- 设计更鲁棒、**自适应的切换策略**，减少误判，无需手动调参。
- 探索将 BLEST 的思想扩展到其他迭代型图算法，如 **Betweenness Centrality** 或 **Connected Components**。
- 优化预处理流程，支持动态图或流式图的增量更新。
- 进一步压缩数据结构，降低内存占用，支持更大规模的并发计算。

</details>

---

### 7. [SMADE-IE: Sparse Multi-Agent Framework with Evidence-Driven Debate for Zero-Shot Information Extraction](https://arxiv.org/abs/2606.04691)

**Authors**: Kenfeng Huang, Yi Cai, Xin Wu, Zikun Deng, Li Yuan  
**Category**: cs.CL  
**Published**: 2026-06-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.04691v1  

#### Abstract
Zero-shot information extraction (IE) with large language models (LLMs) has attracted increasing attention due to its flexibility in adapting to new schemas and domains without task-specific training. Existing approaches mainly rely on monolithic prompting, each-type prompting, or multi-agent debate...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SMADE-IE 论文核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **zero-shot Information Extraction (IE)** 方法在使用 **Large Language Models (LLMs)** 时面临以下挑战：
- **Monolithic prompting** 方法容易出现边界错误（boundary errors）和类型混淆（type confusion），例如将“Apple”误分类为 `PRODUCT` 而非 `ORGANIZATION`。
- **Each-type prompting** 和 **multi-agent debate** 方法虽然能缓解上述问题，但引入了跨类型冲突（cross-type conflicts）、冗余的代理交互（redundant agent interactions）以及高昂的 token 开销（token overhead）。

### 提出的新方法与创新思路
本文提出 **SMADE-IE**（Sparse Multi-Agent Framework with Evidence-Driven Debate for Zero-Shot IE），其核心创新包括：

#### （1）Adaptive Mode Selector（自适应模式选择器）
- 动态判断输入样本复杂度（`low`, `med`, `high`），并路由至两种提取模式：
  - **Global Extraction Mode**：用于简单样本，轻量级单次提取 + 验证修正。
  - **Type-Centric Extraction Mode**：用于复杂样本，仅对相关类型启动专用代理，减少无关推理噪声。

#### （2）Evidence-Driven Debate（证据驱动辩论机制）
- 将多代理辩论结构化为 **Toulmin-style argument**（主张、依据、推理、支撑、反驳），提升逻辑严谨性。
- 引入外部 **evidence scorer** 对论据打分，并结合 **Bayesian Beta updates** 进行置信度聚合。
- 支持 **early-stopping debate**：当后验分布稳定或领先候选优势显著时提前终止，降低计算成本。

#### （3）Iterative Entity-Relation Alignment (IERA)
- 在 JERE 任务中迭代对齐实体与关系预测，确保本体一致性（ontology consistency）。

### 相比现有方法的优势
| 维度 | SMADE-IE 优势 |
|------|----------------|
| **准确性** | 显著优于 monolithic、each-type 和 multi-agent debate 基线 |
| **效率** | 通过稀疏代理选择和早停机制大幅降低 token 成本 |
| **鲁棒性** | 结构化辩论避免自由形式讨论中的噪声干扰和不可靠裁决 |

---

## 2. 核心实验方法和设置

### 数据集
在 **9 个基准数据集** 上进行评估，涵盖三大任务：

| 任务 | 数据集 | 类型数量 | 平均每样本类型数 |
|------|--------|----------|------------------|
| **NER** | CoNLL03, OntoNotes5, SciERC, CrossRE, REDFM | 3–38 | 1.31–4.72 |
| **RE** | DocRED, SemEval2010, SciERC, CrossRE, REDFM | 8–32 | 0.84–5.05 |
| **JERE** | CoNLL04, NYT | — | — |

> 注：所有任务均遵循 zero-shot 协议，仅提供自然语言定义作为外部知识。

### 实验设置
- **主干模型**：`GPT-3.5-Turbo-0125`
- **外部证据评分器**：`AlignScore-base`（本地部署）
- **最大辩论轮数**：`T_max = 3`
- **早停阈值**：`θ_stop = 0.75`（后验优势），`ε = 0.02`（收敛性）

### 评估指标
- **Micro Partial F1 (F1p)**：允许预测与真实跨度部分重叠
- **Strict F1 (F1s)**：要求完全匹配
- 报告平均 F1p 和 F1s

### 基线方法对比
| 类型 | 方法 | 特点 |
|------|------|------|
| **Monolithic Prompting** | AEiO | 所有类型一次性提取 |
| **Each-Type Prompting** | One-Step, G&O | 每种类型独立提示 |
| **Multi-Agent Debate** | CROSSAGENTIE | 多代理协作 + 跨类型辩论 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（平均 F1 表现）

#### NER 结果（Table 2）
| Method | F1p ↑ | F1s ↑ |
|--------|-------|-------|
| AEiO | 42.65 | 35.27 |
| One-Step | 37.64 | 32.62 |
| G&O | 38.78 | 33.16 |
| CROSSAGENTIE | 45.95 | 40.61 |
| **SMADE-IE (Ours)** | **57.32** | **50.45** |

> ✅ **相对最佳基线提升 +11.37 F1p**

#### RE 结果（Table 3）
| Method | F1p ↑ | F1s ↑ |
|--------|-------|-------|
| AEiO | 18.54 | 11.08 |
| One-Step | 15.19 | 9.23 |
| G&O | 11.89 | 7.60 |
| CROSSAGENTIE | 11.45 | 7.04 |
| **SMADE-IE (Ours)** | **22.37** | **15.71** |

> ✅ **全面领先，平均 F1p 提升达 +10.92**

#### JERE 结果（Table 4）
| Method | F1p ↑ | F1s ↑ |
|--------|-------|-------|
| CROSSAGENTIE | 29.87 | 22.91 |
| **SMADE-IE (Ours)** | **44.33** | **36.98** |

> ✅ **绝对提升 +14.46 F1p, +14.07 F1s**

---

### 与基线方法的对比结果
- 在所有任务和数据集上，SMADE-IE 均取得 **最优性能**。
- 在类型丰富的数据集（如 OntoNotes5、CrossRE）上增益更大，说明其对复杂 schema 更具适应性。
- 在 RE 任务中，SMADE-IE 的 token 成本远低于 CROSSAGENTIE（如 DocRED 上从 21,784 → 3,240 tokens/sample）。

---

### 消融实验结果（Table 5）

| 变体 | CoNLL04 F1p | NYT F1p | 影响分析 |
|------|-------------|---------|----------|
| Full SMADE-IE | 58.44 | 30.22 | 全模型 |
| w/o IERA | 59.03 | 24.94 | 缺少 IERA 对 NYT 影响大 |
| Type-Centric Only | 59.43 | 29.65 | 无类型筛选仍有效 |
| w/o Relevant Type Selection | 48.66 | 27.43 | 类型选择至关重要 |
| w/o Review Agent | 52.75 | 29.50 | Review 有助于恢复遗漏类型 |
| w/o Debate | 54.13 | 24.24 | 辩论机制显著提升精度 |
| Global Only | 52.47 | 16.28 | 复杂样本需细粒度处理 |

> 🔍 发现：**Adaptive Mode Selection** 和 **Evidence-Driven Debate** 是性能提升的关键组件。

---

## 4. 关键结论和发现

### 主要发现
1. **动态路由机制有效平衡效率与精度**  
   Adaptive Mode Selector 能准确识别样本复杂度，在简单样本上接近 monolithic prompting 的效率，在复杂样本上启用精细推理。

2. **结构化辩论优于自由辩论**  
   Toulmin-style argument + Bayesian update 提供了可解释、可收敛的冲突解决路径，避免了传统 multi-agent debate 中的“无效争论”。

3. **稀疏代理调用显著降低开销**  
   通过只激活相关类型的代理，SMADE-IE 在多数样本上避免了全类型枚举，token 成本接近轻量级方法（见 Table 6）。

4. **早停机制高效且可靠**  
   图 5 和图 6 显示，超过 75% 的辩论因 **posterior convergence** 或 **decisive stop** 提前结束，平均仅需 1.75–2.76 轮。

---

### 方法的局限性
1. **依赖冻结的外部证据评分器**  
   当前的 `AlignScore` 是静态模型，其校准上限决定了贝叶斯更新的可靠性边界。

2. **在长文档密集类型场景下效率优势减弱**  
   如 REDFM 数据集上，辩论次数仅减半，未实现数量级下降。

3. **实验局限于闭源模型**  
   主要在 `GPT-3.5-Turbo` 上验证，尚未扩展到开源小模型或多语言设置。

---

### 未来工作方向
- 探索更高效的 **debate scheduling 策略**（如异步、优先级调度）
- 扩展至 **structured extraction beyond IE**（如事件抽取、表格填充）
- 支持 **smaller open-source LLMs** 和 **multilingual schemas**
- 研究 **longer document processing** 与 **memory-aware debate**

--- 

> 📌 **总结一句话**：  
> SMADE-IE 通过 **adaptive routing + structured debate + iterative alignment**，实现了 **高精度、低开销、强鲁棒性** 的 zero-shot IE 框架，为多代理系统在信息抽取中的应用提供了新范式。

</details>

---

### 8. [RL Excursions during Pre-Training: Re-examining Policy Optimization for LLM training](https://arxiv.org/abs/2606.04272)

**Authors**: Rachit Bansal, Clara Mohri, Tian Qin, David Alvarez-Melis, Sham Kakade  
**Category**: cs.LG  
**Published**: 2026-06-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.04272v1  

#### Abstract
The standard LLM training pipeline applies reinforcement learning (RL) only after pre-training and supervised fine-tuning (SFT). We question this status quo by training a LLM from scratch and applying RL, SFT, and SFT followed by RL directly to intermediate pre-training checkpoints. We find that RL ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：RL Excursions during Pre-Training: Re-examining Policy Optimization for LLM training**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
当前主流的 LLM 训练范式遵循“预训练（Pre-training）→监督微调（SFT）→强化学习（RL）”的三阶段流程，其中 **RL 仅在 SFT 之后应用**。本文质疑这一设计是否必要，并提出以下核心问题：
- **RL 是否必须依赖 SFT 才能有效？**
- **RL 在训练早期（甚至直接在 base checkpoint 上）是否同样有效？**
- **RL 对模型能力的影响是“锐化”（sharpening）还是“扩展”（expansion）？**

### 🚀 **提出了什么新方法或新思路**
1. **将 RL 应用于预训练中间检查点（Intermediate Pre-training Checkpoints）**
   - 首次系统性地研究从预训练早期（低至 4B tokens）开始直接进行 RL 的效果。
   - 探索了多种训练路径：`Direct RL`, `SFT`, `SFT→RL`, `SFT-Gold`, 以及提出的 `Parallel Averaging`。

2. **提出 Parallel Averaging 方法**
   - 在单个训练步骤中并行计算 SFT 和 RL 的梯度更新，分别维护独立的优化器状态（Adam moments），然后对两个更新进行平均。
   - 公式化为：  
     $$
     \Delta\theta = \Delta\theta_{\text{RL}} + \Delta\theta_{\text{SFT}}, \quad \theta \leftarrow \theta + \Delta\theta
     $$

3. **重新审视 RL 的作用机制**
   - 挑战近期关于“RL 只是 sharpening 输出分布”的观点，指出该现象实际上是 **SFT 后接 RL 的副作用**，而非 RL 本身固有属性。

### ⭐ **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **训练效率** | Direct RL 在仅有 4B 预训练 token 时即显著提升性能，早于 Chinchilla 最优点，说明 RL 不需等待充分预训练即可生效。 |
| **数据利用效率** | 当 SFT 数据稀疏（每题仅一个标注解）时，Direct RL 显著优于 SFT；而 SFT-Gold（多解）虽强但不现实。 |
| **通用能力保持** | SFT 会损害非推理任务表现（如 ARC、HellaSwag），而 Direct RL 几乎不影响这些能力。 |
| **综合性能** | Parallel Averaging 在 pass@32 上全面超越所有其他方法（包括标准 SFT→RL 流程），同时保留通用能力。 |

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
| 类型 | 名称 | 描述 |
|------|------|------|
| **预训练数据** | **DOLMino**（50B tokens） | 包含 Wikipedia (7%)、高质量网页（DCLM/FLAN, 60%）、数学数据（20%）、StackExchange 与 STEM 论文（共 7%）。 |
| **额外数据**（Scaling D） | **Dolma 3 Dolmino Mix**（+10B 数学密集型 tokens） | 用于测试预训练数据组成的影响。 |
| **后训练数据** | **OpenMathInstruct** | 包含约 80K 数学问题，每个问题平均有 ~23 条人工标注的正确推理路径（ground-truth demonstrations）。分为 GSM8K-like 和 MATH-like 子集。 |

### 🔧 **实验设置**
- **模型架构**：基于 OLMo2 的 1B 参数语言模型（部分实验扩展到 4B）。
- **预训练规模**：最多训练 50–60B tokens。
- **训练策略对比**：
  - `M_t`: Base model at pre-training step t
  - `M_RL`: Direct RL on M_t
  - `M_SFT`: SFT on M_t（默认每题随机选一条解）
  - `M_SFT-Gold`: SFT 使用全部多条 ground-truth 解
  - `M_SFT→RL`: 标准流程：先 SFT 再 RL
  - `M_parallel`: 提出的 Parallel Averaging 方法

- **RL 方法**：采用 **GRPO**（Generalized Reward Policy Optimization），一种基于可验证奖励的 RLVR 范式，奖励信号来自最终答案是否正确（binary reward）。

### 📊 **评估指标**
- **主指标**：`pass@k`（k ∈ {1, 8, 32}），表示生成 k 个样本中至少有一个正确的概率，温度 T=0.6。
- **评测任务**：
  - **GSM8K**：小学水平数学题，相对简单
  - **MATH**：竞赛级难题，更具挑战性
- **通用能力评估**：在 6 个非数学基准上测试 F1 分数，包括 LAMBADA、HellaSwag、ARC-Easy/Challenge、PIQA、OpenBookQA。

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据与对比**

#### ✅ **Direct RL 早期即有效（Fig. 2）**
- 在仅 **4B 预训练 token** 时启动 RL：
  - `pass@1` 从 ~2% 提升至 ~18%
  - 到 **10B tokens** 时，`M_RL` 已达到甚至超过 `M_SFT→RL` 的性能
- 表明：**RL 不需要依赖 SFT 或完整预训练即可带来显著收益**

#### ✅ **当 SFT 数据稀缺时，RL 更优（Fig. 3）**
| 方法 | pass@1 | pass@8/pass@32 |
|------|--------|----------------|
| `M_SFT`（单解） | 较低 | 明显落后于 `M_RL` |
| `M_RL`（无真值轨迹） | 更高 | 显著优于 `M_SFT` |
| `M_SFT-Gold`（全解） | 最高 | 超过 `M_RL`（因覆盖多样路径） |

👉 结论：**当无法获得多个高质量标注时，Direct RL 是更实用且更强的选择**

#### ✅ **目标领域数据 > 模型规模（Fig. 4 & 10）**
| 设置 | Base Model 性能 | RL 增益（Δpass@1） |
|------|------------------|--------------------|
| N=1B, D=50B | 中等 | 小 |
| N=4B, D=50B | 更高（得益于更大模型） | 增益未明显提升 |
| N=1B, D=60B（+10B 数学数据） | 略高于原始 1B | **RL 增益大幅提升** |

👉 结论：**增加任务相关预训练数据比扩大模型规模更能促进 RL 成效**

#### ✅ **Parallel Averaging 综合最优（Fig. 8 & 16）**
| 方法 | pass@32 | 通用能力保留 |
|------|---------|--------------|
| `M_SFT` | 一般 | ❌ 下降 4–8 pp |
| `M_RL` | 高 | ✅ 几乎不变 |
| `M_SFT→RL` | 高 | ❌ 下降 |
| `M_parallel` | **最高** | ✅ 与 `M_RL` 相当 |

👉 特别是在 **pass@32** 上持续领先，表明其能更好地探索多样化正确路径。

#### 🔍 **消融实验结果**
- **Base model 的 pass@k 可预测 RL 效果（Fig. 5）**
  - 存在单调正相关：base model 在测试集上的 `pass@k` 越高，RL 后提升空间越大
  - 可作为轻量级诊断工具判断何时适合引入 RL

- **Rollout 数量影响（Appendix D）**
  - 增加 rollout 数（如从 5 到 64）提高样本效率，但降低 FLOPs 效率
  - 最终收敛性能相近 → **rollout 数量不是决定性因素**

- **SFT 导致通用能力退化（Fig. 7）**
  - 所有 SFT 方法均导致非数学任务性能下降 4–8 个百分点
  - 而 `M_RL` 和 `M_parallel` 完全避免此问题

---

## 4. **关键结论和发现**

### 🎯 **主要发现**
1. **RL 可以非常早地应用于预训练过程**  
   - 在 **4B 预训练 token** 阶段即可通过 RL 显著提升推理能力，远早于传统假设所需的“基础能力门槛”。

2. **Distribution Sharpening 是 SFT 的产物，不是 RL 的本质**  
   - 当 RL 接在 SFT 之后时，出现 pass@1 上升但 pass@32 下降的现象（sharpening）
   - 而 **Direct RL 同时提升 pass@1 和 pass@32**，说明它是在 **扩展（expand）模型的能力边界**

3. **SFT 有害于通用能力，RL 则保持中立**  
   - SFT 会导致模型在非目标任务上性能下降（遗忘 pre-training knowledge）
   - RL 因为是 on-policy 学习，不会破坏原有分布

4. **Pre-training Data Composition 是 RL 成功的关键杠杆**  
   - 加入更多数学相关内容比扩大模型规模更能释放 RL 潜力
   - 支持“质量 > 规模”的数据优先原则

5. **Parallel Averaging 实现双赢**  
   - 结合 SFT 的结构引导与 RL 的探索能力
   - 在推理性能与通用能力之间取得最佳平衡

---

### ⚠️ **局限性**
1. **实验集中在数学推理任务**
   - 所有训练与评估围绕 OpenMathInstruct 展开，结论在外推至其他领域（如代码、对话）前需谨慎。
2. **模型规模有限**
   - 主要实验在 1B–4B 模型上进行，尚未验证在百亿级以上模型中的表现。
3. **使用 GRPO 作为代表性的 RLVR 方法**
   - 未涵盖其他 RL 算法（如 DPO、PPO 变体）可能带来的差异。
4. **Pre-training 数据偏数学密集**
   - DOLMino 中 20%+ 为数学/推理数据，可能高估了早期 RL 的有效性。

---

### 🔮 **未来工作方向**
1. **将 RL 视为一级公民（first-class citizen）纳入整个训练流程**
   - 设计端到端的混合训练策略，动态调度 NTP、SFT、RL 目标
2. **自适应 rollout 策略**
   - 根据 base model 的 pass@k 动态分配计算资源，集中于“有潜力但未收敛”的问题
3. **开发更高效的 RL 训练算法**
   - 平衡 rollout 数量与 FLOPs 开销，在稀疏奖励下更高效学习
4. **探索不同权重调度机制**
   - 替代 uniform averaging，尝试 importance weighting、curriculum learning 等方式融合 SFT 与 RL 梯度
5. **验证在超大规模模型上的普适性**
   - 在 70B+ 模型上复现本工作结论，确认是否仍成立

---

## ✅ 总结一句话
> **本论文颠覆了“RL 必须放在最后”的传统认知，证明 RL 可早期介入、独立生效，不仅能媲美甚至超越 SFT→RL 流程，还能避免 SFT 带来的通用能力退化；并通过 Parallel Averaging 实现了性能与稳健性的双重突破，为重构 LLM 全流程训练提供了新范式。**

</details>

---

### 9. [STaR-Quant: State-Time Consistent Post-Training Quantization for Diffusion Large Language Models](https://arxiv.org/abs/2606.04945)

**Authors**: Xin Yan, Aqiang Wang, Zhenglin Wan, Xingrui Yuand Ivor Tsang  
**Category**: cs.LG  
**Published**: 2026-06-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.04945v1  

#### Abstract
Diffusion large language models (DLLMs) have recently emerged as a promising alternative to autoregressive LLMs by generating text through iterative masked denoising with bidirectional context. However, their large model sizes and iterative denoising process introduce substantial memory and computat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：STaR-Quant: State-Time Consistent Post-Training Quantization for Diffusion Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Diffusion Large Language Models (DLLMs) 通过迭代去噪生成文本，具有双向上下文建模和并行解码优势，但其**大规模参数量**和**多步迭代推理机制**导致显著的内存与计算开销。现有的 Post-Training Quantization (PTQ) 方法在应用于 DLLMs 时面临两大挑战：
- **State-dependent activation disparity**：在每一步去噪过程中，序列中同时存在 masked 和 unmasked tokens，二者激活值分布差异大（如范围、outlier 模式），统一量化策略难以有效处理。
- **Temporal error accumulation**：低比特量化误差会在多个去噪步骤间累积，尤其在 attention 中的 softmax-value 乘法操作对量化敏感，导致最终输出质量严重下降。

### 提出的新方法
为解决上述问题，作者提出 **STaR-Quant** —— 一种面向 DLLMs 的 state-time 一致的 PTQ 框架，包含两个核心组件：

#### ✅ State-Guided Activation Transformation (SGAT)
- 将隐藏维度划分为 **shared**、**masked-specific** 和 **unmasked-specific** 子空间。
- 对不同状态的 token 应用不同的激活变换（通过二元门控实现），使其进入更适合量化的子空间。
- 权重侧保持一个**统一的静态变换**，避免重复存储多个变换后的权重，提升推理效率。

> **优势**：在不增加权重存储负担的前提下，实现了对不同 token 状态的精细化激活平滑。

#### ✅ Temporal Attention Compensation (TAC)
- 在 attention 输出投影前引入轻量级补偿模块，修正量化后的 attention 表示。
- 采用 **block-wise affine mapping**（块对角仿射映射）进行校正，平衡表达能力与估计稳定性。
- 参数通过闭式求解（closed-form）匹配全精度与量化表示的一阶和二阶统计量（均值与协方差）。

> **优势**：无需修改 attention 计算过程，即可有效缓解跨时间步的误差积累，且计算开销小。

### 相比现有方法的优势
| 方面 | 传统 PTQ 方法（如 AWQ, QuaRot） | STaR-Quant |
|------|-------------------------------|-----------|
| 处理 token 状态差异 | 忽略 masked/unmasked 差异，统一处理 | 显式建模状态依赖，分别处理 |
| 时间维度误差控制 | 无显式机制应对迭代误差积累 | 引入 TAC 抑制跨步误差传播 |
| 推理效率 | 高效但性能下降明显 | 保持高效的同时显著提升精度 |
| 通用性设计 | 主要针对 autoregressive LLMs | 专为 DLLMs 的 denoising 流程定制 |

---

## 2. 核心实验方法和设置

### 使用的模型
在三种主流 DLLMs 上进行验证：
- **LLaDA-8B**
- **LLaDA-1.5-8B**
- **Dream-7B**

### 数据集与评估基准
使用以下三类任务进行全面评估：

| 类别 | 基准 | 指标 |
|------|------|------|
| **通用知识与推理** | TruthfulQA-MC2, ARC-Challenge, HellaSwag, WinoGrande, PIQA, MMLU, C-EVAL | Accuracy |
| **数学推理** | GSM8K | Pass@1 |
| **代码生成** | HumanEval | Pass@1 |

### 实验设置
- **量化配置**：W4A4（4-bit weights & activations），部分补充 W8A8 结果。
- **校准数据**：从 Winogrande 数据集中采样 128 个 segment 进行 post-training calibration。
- **硬件平台**：NVIDIA A40 GPUs。
- **无需微调**：完全基于 PTQ 范式，冻结原始模型参数。

### 基线方法对比
| 基线 | 简介 |
|------|------|
| **RTN (Round-to-nearest)** | 最基础的均匀量化 |
| **AWQ** | 基于激活统计保护重要权重 |
| **QuaRot** | 利用旋转消除 outlier |
| **DLLMQuant / DLLMQuant++** | 当前最先进的 DLLM 专用 PTQ 方法（基于 AWQ 或 QuaRot 构建） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（W4A4 平均准确率）

| Model | FP | RTN | AWQ | QuaRot | DLLMQuant++ | **STaR-Quant** |
|-------|-----|------|------|--------|--------------|----------------|
| LLaDA-8B | 58.99 | 44.23 | 48.09 | 51.03 | 54.29 | **57.07** |
| LLaDA-1.5-8B | 69.86 | 53.12 | 58.49 | 61.06 | 64.31 | **66.93** |
| Dream-7B | 66.94 | 49.53 | 54.89 | 58.85 | 61.90 | **63.59** |

> ✅ **平均提升**：相比最强基线 DLLMQuant++，STaR-Quant 分别带来 **+2.78**, **+2.62**, **+1.69** 的绝对分数提升。

### 与基线方法的对比结果
- 在所有任务类别上均优于现有方法，尤其是在：
  - **MMLU & C-EVAL**：表明其更好保留了多领域知识理解能力。
  - **GSM8K & HumanEval**：对复杂推理和精确生成任务更鲁棒，说明中间表示保真度更高。
- 即使在高难度 4-bit 量化下，仍能接近 FP16 性能（仅下降约 1.9–2.9 pts）。

### 消融实验结果（LLaDA-8B, W4A4）

| 方法 | Avg. Score | Δ↓ vs FP |
|------|------------|---------|
| STaR-Quant (完整) | **57.07** | 1.92 |
| w/o TAC | 55.43 | 3.56 |
| w/o SGAT | 56.39 | 2.60 |

> 🔍 **发现**：
- 移除 TAC 导致最大性能下降，说明 **时间维度误差补偿至关重要**。
- 移除 SGAT 也造成明显退化，证明 **状态感知激活变换有效缓解分布差异**。
- 两者互补，共同构成“state-time 一致性”核心思想。

### TAC 块大小影响（Block Size Ablation）

| Block Size $g$ | 4 | 8 | **16** | 32 | 64 |
|----------------|----|----|-------|-----|-----|
| Avg. Score     | 52.48 | 54.66 | **57.07** | 56.39 | 56.03 |

> 📌 最佳块大小为 **16**，过大反而因协方差估计不稳定而性能下降。

### 推理效率（Speed & Memory）

| Model | Speedup (vs FP16) | Memory Saving |
|-------|--------------------|---------------|
| LLaDA / LLaDA-1.5 | ~1.65× | **3.05×** |
| Dream-7B | **1.69×** | **3.14×** |

> ⚡️ 在显著压缩内存（降至约 1/3）的同时，实现近 **1.66× 平均加速**，且未引入额外运行时开销。

---

## 4. 关键结论和发现

### 主要发现
1. **DLLMs 的 PTQ 需专门设计**：不能直接套用 autoregressive LLM 的量化方法，必须考虑其独特的 **mask state 共存** 和 **iterative decoding** 特性。
2. **State-Time 一致性是关键原则**：
   - SGAT 解决了 **state 维度** 的激活不一致性；
   - TAC 缓解了 **time 维度** 的误差累积问题；
   - 二者协同显著提升了低比特量化下的模型保真度。
3. **轻量但高效的设计可行**：TAC 仅需少量可学习参数（block-wise affine），即可实现精准补偿，适合部署。

### 方法的局限性
- **适用范围有限**：目前聚焦于典型的 masked-denoising DLLMs（如 LLaDA, Dream），尚未验证于其他扩散架构（如 continuous diffusion）或更大规模模型。
- **极端低位量化困难**：3-bit 或 2-bit 设置仍未充分探索，可能需要更强的状态建模或动态补偿机制。
- **系统优化空间**：当前实现未使用 fused kernels，实际加速潜力有待进一步挖掘。
- **状态定义简单**：仅区分 masked/unmasked，未来可扩展至 confidence-aware 或 fine-grained 状态建模。

### 未来工作方向
- 扩展至 **multimodal diffusion models**（如图文生成）中的跨模态状态建模。
- 探索 **finer-grained token states**（如预测置信度、语法角色等）以增强 SGAT。
- 结合 **quantization-aware fine-tuning (QAT)** 进一步提升极限压缩性能。
- 开发专用 **inference engine 支持 fused SGAT+TAC kernel**，最大化实际部署效率。

--- 

> 💡 **总结一句话**：  
> **STaR-Quant 是首个专为 DLLMs 设计的 state-time 一致 PTQ 框架，通过 SGAT + TAC 双轮驱动，在 W4A4 下实现接近 FP16 的性能，同时获得高达 3.14× 内存节省和 1.69× 加速，为高效部署扩散语言模型提供了新范式。**

</details>

---

### 10. [LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](https://arxiv.org/abs/2606.04485)

**Authors**: Yuanrui Wang, Xingxuan Zhang, Han Yu, Mingchao Ming, Gang Ren, Hao Yuan, Li Mao, Yunjia Zhang, Chun Yuan, Peng Cui  
**Category**: cs.LG  
**Published**: 2026-06-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.04485v1  

#### Abstract
Tabular foundation models (TFMs) increasingly rival tree ensembles, but their performance is often compute-inefficient: with standard affine scalar tokenization, each feature injects value variation through an essentially one-dimensional channel, and feature IDs/positional signals cannot increase wi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本文针对当前 **Tabular Foundation Models (TFMs)** 存在的两大核心瓶颈：

1. **低秩坍缩（Low-Rank Collapse）**  
   当前主流的标量 tokenization 方法（如单一线性投影 + 特征 ID/位置编码）导致输入嵌入层表达能力受限。每个特征值通过一个本质上一维的通道注入信息，造成浅层表示的有效秩（effective rank）极低，模型参数冗余严重。

2. **注意力瓶颈（Attention Bottlenecks）**  
   现有双向注意力堆叠顺序（Feature-Attention → Sample-Attention）存在设计缺陷：
   - 初始特征注意力缺乏跨样本统计信息支持；
   - 最终预测仅依赖目标 token，导致部分注意力模块训练信号弱。

---

### 🛠️ 提出的新方法或新思路

作者提出统一的 **tokenize-and-route** 框架，包含两个核心组件：

#### （1）**RaBEL（Radial Basis Embedding Layer）**

- 替代传统的线性标量编码，将每个数值特征扩展为一组紧凑的局部化 RBF（径向基函数）特征。
- 可选引入指数门控（exponent-gated），提升对数量级变化和异方差性的鲁棒性。
- 优势：
  - 引入非线性，增强早期层对数值变化的敏感度；
  - 改善第一层的条件数（conditioning）；
  - 显著提高浅层表示的有效秩。

#### （2）**重排序的双向注意力块：S→N→F（Sample → FFN → Feature）**

- 将标准的 F→S→N 结构改为 **Sample-Attention → FFN → Feature-Attention**。
- 优点：
  - 在特征混合前先聚合跨样本上下文（如列间相关性、分布统计）；
  - 所有注意力计算最终都参与读出（readout），避免信息浪费；
  - 更好地捕捉特征关系并促进关键特征发现。

结合以上两点，构建了 **LimiX-2M** 模型——一个仅含 **2M 参数** 的轻量级 TFM。

---

### 🔍 相比现有方法的优势

| 维度 | LimiX-2M 的优势 |
|------|----------------|
| **性能** | 超越更大规模的基线模型（如 7M 参数的 TabPFN-v2 和 27M 的 TabICL） |
| **效率** | 显著降低训练与推理成本，GPU 推理速度比 TabPFN-v2 快约 2 倍，比 TabICL 快 >10 倍 |
| **可解释性** | 通过机制驱动的设计（如 RaBEL 和 S→N→F）改善了模型内部的信息流动 |
| **泛化性** | 在多个主流 tabular benchmark 上实现强泛化能力 |

---

## 2. 核心实验方法和设置

### 📚 数据集

综合评估在以下六大 benchmark suite 上进行，涵盖分类与回归任务：

| 类型 | 数据集 |
|------|--------|
| **Classification** | `BCCO-CLS`, `OpenML-cc18`, `PFN-CLS`, `TabArena-CLS`, `TabZilla`, `TALENT-CLS` |
| **Regression** | `BCCO-REG`, `CTR23`, `PFN-REG`, `TabArena-REG`, `TALENT-REG` |

共包含超过 **300 个真实世界表格数据集**，经过统一过滤（排除样本过多、特征过多或类别过多元的数据）。

---

### ⚙️ 实验设置与评估指标

| 项目 | 设置说明 |
|------|----------|
| **预训练方式** | 使用基于 Structural Causal Models (SCMs) 的合成数据生成协议（继承自 PFN 系列和 LimiX） |
| **模型架构** | 12 层 Transformer，隐藏维度 $d_{\text{model}}=96$，6 个注意力头 |
| **评估协议** | 控制变量比较：相同训练配置下对比不同 embedding 与 attention 结构 |
| **评估指标** | <ul><li>**分类任务**：AUC, Accuracy, F1-score</li><li>**回归任务**：R², RMSE</li></ul> |
| **性能汇总方式** | 报告各 benchmark 上的平均排名（mean rank）和累计最优达成率（Cumulative Percentage of Optimal Achievements） |

---

### 🆚 基线方法对比

| 类别 | 包括的基线模型 |
|------|---------------|
| **Tree-based** | XGBoost, LightGBM, CatBoost, RF |
| **Deep Learning** | FT-Transformer, TabTransformer, SAINT, NODE, ResNet, MLP 等 |
| **Foundation Models** | TabPFN-v2 (7.24M), TabICL (27.10M), Mitra (75.67M), LimiX-16M (16.52M) |
| **Ensemble** | AutoGluon |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）整体性能排名（Table 6）

| Model | 分类平均 Rank | 回归平均 Rank | 总体表现 |
|-------|----------------|----------------|-----------|
| **LimiX-2M (1.92M)** | **2.71**（第二） | **3.46**（第二） | **仅次于 LimiX-16M，优于所有其他 FM** |
| TabPFN-v2 (7.24M) | 7.31 | 5.92 | 明显落后于 LimiX-2M |
| TabICL (27.10M) | 10.58 | — | 表现一般 |
| AutoGluon | 9.39 | — | 强集成仍不敌 LimiX-2M |

> ✅ **LimiX-2M 是唯一进入 Top-2 的小模型（<2M 参数）**

---

#### （2）在具体 benchmark 上的表现（示例）

##### BCCO-CLS（分类）
| Model | AUC ↑ | Acc ↑ | F1 ↑ |
|-------|--------|--------|-------|
| **LimiX-2M** | **0.858** | **0.787** | **0.701** |
| TabPFN-v2 | 0.843 | 0.772 | 0.679 |
| TabICL | 0.847 | 0.768 | 0.672 |

✅ **全面超越更大模型**

##### BCCO-REG（回归）
| Model | R² ↑ | RMSE ↓ |
|-------|--------|--------|
| **LimiX-2M** | **0.785** | **0.392** |
| TabPFN-v2 | 0.772 | 0.404 |
| AutoGluon | 0.781 | 0.398 |

✅ **R² 更高，RMSE 更低，优于 TabPFN-v2 与 AutoGluon**

---

#### （3）推理效率对比（Table 26）

| Model | GPU 推理时间 (ms) | 相对速度 |
|-------|--------------------|---------|
| TabPFN-v2 | 352.60 | 1× |
| TabICL | 1749.61 | ~5× 慢 |
| **LimiX-2M** | **171.40** | **快 2× 于 TabPFN-v2** |
| Mitra | 5766.25 | >30× 慢 |

✅ **LimiX-2M 不仅更准，而且更快**

---

### 🔬 消融实验结果

#### （1）模块消融（Figure 3）

在 TabArena 和 TabZilla 上验证各组件贡献：

| 配置 | AUC 提升 |
|------|----------|
| Baseline | 0.830 |
| +RaBEL | +0.010 |
| +RBA (Reordered Block) | +0.013 |
| **LimiX-2M (全量)** | **+0.015** |

➡️ **RaBEL 与 RBA 均带来显著增益，协同作用明显**

#### （2）注意力结构变体比较（Figure 3 右图）

测试不同 attention 顺序：
- **FSN**（原顺序）：性能最差
- **SNF**（提出顺序）：最优
- **SFN / FNSN**：次优

➡️ **Sample-first + FFN 中介 的结构最为高效**

#### （3）RaBEL 超参分析（Appendix C.3）

| 设定 | 最佳选择 |
|------|----------|
| Token 维度 | 32 |
| RBF Kernel 数量 | 64 |
| 初始化方式 | Orthogonal > Xavier/Kaiming |
| 带宽 $ \sigma $ | 固定 1.0 优于可学习 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **线性标量 tokenization 是低秩坍缩的根本原因**  
   即使加入 feature ID 或 positional encoding，也无法增加“值”本身的自由度，导致早期激活高度相关。

2. **RaBEL 成功打破“值瓶颈”**  
   通过局部化 RBF 编码，显著提升浅层表示的有效秩（见 Table 5：Rank@95% 提升 83%），改善梯度流。

3. **注意力顺序至关重要**  
   先进行 sample-level attention 可建立列级统计上下文，再进行 feature mixing 更合理；且所有 attention 都应服务于最终 readout。

4. **小模型也能打败大模型**  
   LimiX-2M（1.92M）在多数 benchmark 上超越 TabPFN-v2（7.24M）、TabICL（27M）甚至 XGBoost/AutoGluon，证明了**精度-效率权衡可通过机制改进优化**。

---

### ⚠️ 方法的局限性

1. **目前仅适用于中小规模表格数据**  
   未在超大规模工业场景（如百万行以上）中验证。

2. **RBF 中心初始化依赖经验分位数**  
   对极端偏态分布可能不够鲁棒，需进一步研究自适应初始化策略。

3. **尚未探索与其他 pretraining objective 的结合**  
   如 contrastive learning 或 masked feature modeling 可能进一步提升性能。

---

### 🔮 未来工作方向

1. **构建混合基函数库（Hybrid Basis Libraries）**  
   结合 RBF、周期性映射（Fourier Features）等以适应不同类型的数据模式。

2. **加强自监督预训练机制**  
   探索更强的 pretext tasks 来捕获跨特征依赖。

3. **扩展到分布外泛化与领域迁移**  
   验证 LimiX-2M 在跨域、概念漂移等挑战下的鲁棒性。

4. **部署优化与边缘设备适配**  
   利用其轻量特性推动 TFM 在生产环境中的落地应用。

---

> **一句话总结**：  
> 本文通过 **RaBEL** 和 **S→N→F 注意力重排序** 两项机制创新，有效缓解了表格基础模型中的低秩坍缩与注意力瓶颈，在仅 **2M 参数** 下实现了超越更大模型的性能与效率，为高效、可靠的 TFM 设计提供了新范式。

</details>

---

### 11. [AlphaQ: Calibration-Free Bit Allocation for Mixture-of-Experts Quantization](https://arxiv.org/abs/2606.04980)

**Authors**: Wanqi Yang, Yuexiao Ma, Alexander Conzelmann, Xiawu Zheng, Michael W. Mahoney, T. Konstantin Rusch, Shiwei Liu  
**Category**: cs.LG  
**Published**: 2026-06-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.04980v1  

#### Abstract
Mixture-of-Experts (MoE) architectures scale model capacity through sparse expert activation, but their deployment remains memory-bound because all expert weights must reside in memory. Mixed-precision quantization can substantially reduce this footprint by assigning different bit-widths to differen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AlphaQ: Calibration-Free Bit Allocation for Mixture-of-Experts Quantization 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **MoE模型量化中的校准数据依赖问题**：现有的混合精度量化（Mixed-precision Quantization）方法通常依赖于校准数据（calibration data）来估计专家（expert）的重要性并进行比特分配。然而，前沿MoE大语言模型（LLM）的原始训练数据是专有且不可访问的，导致校准数据存在偏差，从而引发**领域偏见（domain bias）** 和次优的比特分配。
- **跨域泛化能力下降**：如图1所示，使用不同领域（如C4、MATH、GitHub-Code）的数据进行校准会导致不同的比特分配策略，并在未见过的任务上表现退化，说明数据驱动的方法耦合了性能与校准分布。

### 🚀 提出的新方法：AlphaQ
- **提出了一种无需校准数据的比特分配方法 AlphaQ**，完全基于模型权重本身的结构特性进行重要性评估。
- 核心思想来源于 **Heavy-Tailed Self-Regularization (HT-SR) 理论**：具有更重尾（heavier-tailed）特征值谱（eigenvalue spectrum）的专家通常训练得更好、结构更丰富，应分配更高的比特宽度；而轻尾结构的模块可以被更激进地量化。
- 使用 **PL_Alpha_Hill** 度量每个层或专家的谱重尾程度作为其“重要性分数”。

### 🔍 相比现有方法的优势
| 维度 | AlphaQ | 传统方法（如PMQ, GEMQ等） |
|------|--------|--------------------------|
| 是否需要校准数据 | ❌ 不需要（Calibration-free） | ✅ 需要（依赖输入统计） |
| 跨域泛化能力 | 强（避免领域偏见） | 弱（受校准集影响大） |
| 分配粒度 | 层级（layer-wise）精细控制 | 多为块级或专家级统一预算 |
| 全局优化 | ✅ 支持全局比特预算约束下的整数规划求解 | 多为局部最优（block-wise） |

> ✅ **核心优势**：AlphaQ实现了**无需任何输入数据即可完成高质量、高鲁棒性的比特分配**，解决了当前MoE量化中因校准数据不完整/有偏而导致的性能瓶颈。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **校准数据（用于对比基线）**：
  - `C4`：通用文本语料
  - `MATH`：数学推理任务
  - `GitHub-Code`：代码生成任务
  - `WikiText2`：用于量化过程中的小样本校准（仅用于GPTQ参数估计，非AlphaQ本身）
- **评估基准（Zero-shot Evaluation）**：
  - `PIQA`, `ARC-e`, `ARC-c`, `HellaSwag`, `WinoGrande`, `BoolQ` → 综合常识推理能力
  - `MMLU`, `MATH`, `CEval` → 多领域知识测试（用于多域对比实验）

### ⚙️ 实验设置
- **模型范围**：在多个主流MoE架构上验证：
  - `Mixtral-8×7B`
  - `DeepSeekV2-Lite`
  - `Qwen1.5-MoE`
  - `Qwen3-30B-A3B`
  - `OLMoE-1B-7B`
- **量化方式**：
  - 权重量化（Weight-only Quantization）
  - Group-wise GPTQ（group size=128），仅用于实际量化操作
  - **非专家层（attention/router）固定为4-bit**
  - **专家内部采用混合精度（1/2/3/4-bit自适应）**
- **比特预算控制**：以平均比特/层（bits per layer）衡量压缩率，测试 **2.5-bit 和 3.5-bit** 设置

### 🎯 评估指标
| 指标 | 含义 |
|------|------|
| **PPL ↓** | WikiText2上的困惑度（越低越好） |
| **Avg. Accuracy ↑** | 多个zero-shot任务的平均准确率（越高越好） |
| **Memory Footprint ↓** | 参数内存占用（GB） |
| **Speedup ↑** | 推理速度提升倍数（vs BF16） |
| **Bit Allocation Pattern** | 比特分配是否合理反映模型内在结构差异 |

### 🆚 对比的基线方法
| 方法 | 类型 | 是否需校准数据 |
|------|------|----------------|
| **Uniform** | 所有专家相同比特 | ❌ |
| **PMQ** (Huang et al., 2024) | 基于激活频率与重建误差的线性规划 | ✅ |
| **GEMQ** (Deng et al., 2026) | 全局优化版PMQ | ✅ |
| **AFG** (Xie et al., 2025) | 自动细粒度量化 | ✅ |
| **BSP**, **Hessian** | 基于梯度/海森信息的方法 | ✅ |
| **DynaMo** (Zheng et al., 2025) | 跨数据集动态调整 | ✅ |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

#### 在 Qwen1.5-MoE 上的表现（3.5-bit 预算）
| 方法 | Avg. Accuracy (%) | 内存占用 |
|------|-------------------|--------|
| BF16（原模型） | ~68.0 | 28.9 GB |
| **AlphaQ** | **68.04** | **7.6 GB** |
| PMQ | 67.75 | ≈8.1 GB |
| Uniform | 65.94 | ≈8.1 GB |

✅ **结论**：AlphaQ 在仅用 **3.5-bit 平均精度**的情况下，达到了与全精度（BF16）相当的准确率，同时将**权重内存压缩至原来的约 1/4**。

#### 在 Mixtral-8×7B 上的表现（2.5-bit 预算）
| 方法 | PPL ↓ | Avg. Acc. ↑ |
|------|-------|------------|
| Uniform | 5.32 | 70.38 |
| PMQ | 5.32 | 70.38 |
| **AlphaQ** | **5.17** | **70.76** |

➡️ AlphaQ 显著优于所有基线，在更低PPL的同时获得更高准确率。

#### 多域校准对比（OLMoE-1B-7B, 3-bit）
| 方法 | MMLU | MATH | CEval | Avg. |
|------|------|------|-------|-----|
| PMQ (C4+MATH) | 44.38 | 12.91 | 25.78 | 27.69 |
| PMQ (MATH+CEval) | 44.33 | 12.37 | 21.99 | 26.23 |
| **AlphaQ** | **44.75** | **13.11** | **28.72** | **28.86** |

✅ **AlphaQ全面领先**，证明其不受限于特定校准域，具备更强的**跨域泛化能力**。

---

### 🔬 消融实验结果（Ablation Studies）

#### ✅ 成分消融（Table 3）
| 方法 | PPL ↓ | Avg. Acc. ↑ |
|------|--------|-------------|
| Noise-only | 11.22 | 63.74 |
| Alpha-only | 10.03 | 66.23 |
| Alpha+Noise (Direct) | 9.56 | 66.81 |
| **Alpha+Noise (Ours)** | **9.19** | **67.11** |

📌 **发现**：单独使用噪声或Alpha都不够好，必须将 **PL_Alpha_Hill 与量化噪声加权结合**才能达到最佳效果。

#### ✅ 预算分配策略（Table 5）
| 策略 | PPL ↓ (Mixtral) | Acc. ↑ |
|------|------------------|--------|
| Block-wise budget | 5.81 | 71.68 |
| **Global budget (Ours)** | **5.17** | **72.39** |

📌 **发现**：全局共享比特预算显著优于每块独立分配，说明不同Transformer块之间确实存在重要性差异。

#### ✅ 量化粒度比较（Table 5）
| 模型 | 专家级分配（Expert-wise）PPL | 层级分配（Layer-wise）PPL |
|------|-------------------------------|----------------------------|
| Mixtral-8×7B | 6.28 (2-bit) / 4.72 (3-bit) | **6.11 / 4.37** |
| DeepSeekV2-Lite | 7.47 / 6.81 | **7.30 / 6.69** |

📌 **发现**：即使在同一专家内，不同投影层（up/gate/down）也有显著的重要性差异，支持**layer-wise细粒度分配**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MoE模型中存在显著的跨专家、跨层、跨块的重要性异质性**，传统的均匀或粗粒度分配严重浪费资源。
2. **PL_Alpha_Hill 是一个有效的无数据依赖的重要性感官测度**，能可靠预测各模块对量化扰动的敏感性。
3. **AlphaQ通过融合结构重要性（alpha）与量化噪声，构建了一个可优化的目标函数**，实现全局最优比特分配。
4. **AlphaQ在多种MoE架构上一致超越校准数据驱动方法**，尤其在低比特（2.5-bit）下仍保持接近全精度性能。
5. **AlphaQ有效缓解了MoE的内存瓶颈**：
   - 在 Mixtral-8×7B 上实现 **4.4× 内存压缩**
   - 在 Qwen1.5-MoE 上实现 **>4× 压缩**，权重仅占 **7.6 GB**

### ⚠️ 方法的局限性（A.12节）
1. 当前仅支持 **weight-only quantization**，尚未扩展到activation量化。
2. 依赖 **HT-SR理论假设**，即重尾谱对应高质量模型。该假设在CV和其他MoE架构中的普适性有待进一步验证。
3. 虽然比特分配无数据依赖，但后续量化步骤（如GPTQ）仍可能引入轻微的数据相关性（如error compensation）。
4. 未与所有现有MoE压缩技术（如稀疏化、蒸馏）进行全面比较。

### 🔮 未来工作方向
- 将 AlphaQ 扩展至 **activation bit allocation**
- 探索在 **vision MoE 或 multimodal MoE** 中的应用
- 结合 AlphaQ 与其他压缩技术（pruning, sparsification）形成端到端高效方案
- 进一步降低 PL_Alpha_Hill 的计算开销，实现实时自适应量化

---

## 总结一句话 💬
> **AlphaQ首次提出了一种无需校准数据、基于权重谱重尾性的MoE量化比特分配方法，在多个大规模MoE模型上实现了接近全精度性能的高压缩比量化，显著提升了部署效率与跨域鲁棒性。**

</details>

---

### 12. [Imbuing Large Language Models with Bidirectional Logic for Robust Chain Repair](https://arxiv.org/abs/2606.05030)

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun, Thomas Lukasiewicz  
**Category**: cs.CL  
**Published**: 2026-06-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.05030v1  

#### Abstract
Autoregressive chain-of-thought (CoT) reasoning in large language models (LLMs) is fundamentally forward-directed: each step conditions only on prior tokens. This unidirectional inductive bias renders even capable models susceptible to error snowballing, wherein a single logical or arithmetic mistak...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Imbuing Large Language Models with Bidirectional Logic for Robust Chain Repair*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

大型语言模型（LLM）在执行复杂推理任务时广泛采用 **Chain-of-Thought (CoT)** 推理，其生成过程是**自回归、前向单向的**。这种架构存在一个根本缺陷：**错误雪球效应（error snowballing）**。

- 一旦推理链中某个早期步骤出现逻辑或算术错误，后续所有步骤都会基于这个“被污染”的上下文继续生成。
- 即使模型具备纠正错误的能力，也无法在没有外部提示的情况下主动检测并修复这些错误。
- 因此，整个推理链可能因一个微小错误而完全失效。

该问题源于模型缺乏一种**双向逻辑约束机制**，即无法同时利用已知前提（premise）和目标结论（milestone）来指导中间步骤的生成。

---

### ✅ 提出了什么新方法或新思路

作者提出 **Teleological Reasoning Infilling (TRI)**，一种训练与推理框架，赋予标准解码器-only Transformer 模型**原生的目标导向桥接能力（goal-conditioned bridging）**。

#### 核心创新点：

1. **Prefix-Suffix-Middle (PSM) 序列架构**
   - 将输入重排为 `[Q, <teleo_premise>, P, <teleo_milestone>, S, <teleo_bridge>, M]`。
   - 利用三个专用哨兵标记（sentinel tokens），使得中间桥接部分 `M` 在生成时可以同时关注前置前提 `P` 和后置里程碑 `S`。
   - **无需修改自注意力机制**即可实现真正的双向条件建模。

2. **两阶段符号化训练流程**
   - **Supervised Fine-Tuning (SFT)**：在形式数学语料库中提取经过符号验证的 `(Q, P, S, M)` 四元组进行监督微调。
   - **Direct Preference Optimization (DPO)**：使用确定性符号验证器（如 Lean 4 / Python 执行器）作为唯一奖励信号，避免 LLM judge 的阿谀倾向（sycophancy）和逻辑盲区。

3. **双系统推理修复算法（Dual-System Inference Repair）**
   - TRI 不作为独立推理器，而是作为**手术式修复模块**嵌入到推理循环中：
     1. 由因果草稿模型（causal draft model）生成初始推理链；
     2. 符号验证器定位第一个失败步骤；
     3. TRI 模型仅对从最后一个正确前提 `P` 到下一个可验证里程碑 `S` 之间的“断链”进行填充（infilling）；
     4. 保留已验证部分，只重生成损坏段落，显著节省计算资源。

4. **理论保障：拓扑一致性（Topological Consistency）**
   - 在温和的 Lipschitz 平滑假设下，证明 PSM 训练目标能诱导出高概率下全局一致的桥接分布，确保生成路径在逻辑上连接 `P` 和 `S`。

---

### ✅ 相比现有方法的优势

| 维度 | TRI 的优势 |
|------|-----------|
| **推理机制** | 支持**目标导向的双向逻辑约束**，而非单纯前向生成或后向启发。 |
| **训练信号** | 使用**确定性符号验证器**作为奖励，杜绝神经裁判的主观性和不可靠性。 |
| **效率** | 只修复错误片段，不重新生成整条链，**token 开销降低 31.2%**。 |
| **泛化性** | 在数学、代码修复、形式定理证明等多个领域均取得 SOTA 表现。 |
| **鲁棒性** | 在低预算和高故障密度场景下仍保持领先性能。 |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集

| 数据集 | 描述 |
|--------|------|
| **MATH** | 包含 12,500 道竞赛级数学题（AMC/AIME/Olympiad），分为 L1–L5 五个难度等级。测试集 5,000 题，重点关注 L4/L5。 |
| **HumanEval-Fix** | 来自 HumanEval 的 Python 编程任务，人为注入逻辑错误，共 492 个带错实例。目标是修复函数体使其通过所有单元测试。 |
| **Lean-Workbook** | 自然语言数学问题及其 Lean 4 形式化证明，测试集包含 2,500 个需补全 tactic 步骤的问题。使用 Lean 4 内核进行类型检查验证。 |

---

### ✅ 实验设置和评估指标

| 指标 | 定义 |
|------|------|
| **Accuracy (%)** | MATH 上最终答案匹配黄金答案的比例。 |
| **Pass@1 (%)** | HumanEval-Fix 中一次修复成功并通过所有测试的比例。 |
| **PCR (%)** | Proof Completion Rate，在 Lean-Workbook 中生成的形式证明能被 Lean 4 成功编译的比例。 |
| **Tok/Prob** | 每个问题平均生成的 token 数量，衡量推理效率。 |
| **RSR (%)** | Repair Success Rate，初始错误链在预算内被成功修复的比例。 |

- **最大 token 预算**：4,096 tokens / problem。
- **TRI 设置**：最多 3 轮修复迭代；每轮调用 `EXTRACTMILESTONE` 子程序向前扫描最多 8 步寻找首个可验证里程碑。
- **训练细节**：
  - 基座模型：Qwen2.5-72B。
  - SFT：约 780k (Q,P,S,M) 样本，3 轮训练。
  - DPO：基于 SFT 模型采样候选桥接，经符号验证构建偏好对，再训练 1 轮。

---

### ✅ 基线方法对比

| 基线方法 | 类型说明 |
|----------|---------|
| Qwen2.5-72B + CoT | 强大开源模型 + 标准零样本 CoT |
| Qwen2.5-72B + CoT-SC(k=16) | 自洽性集成，k=16 条链投票 |
| Llama-3.1-70B + CoT | 跨模型家族比较 |
| Llama-3.1-70B + ToT(b=5) | 树状思维搜索策略 |
| InternLM-StepProver + CoT / CoT-SC(k=8) | 当前 Lean-Workbook 领域 SOTA 模型 |

> 所有方法共享相同 token 预算，确保公平比较。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Table 2）

| 方法 | MATH L5 | HEval-Fix Pass@1 | Lean-WB PCR | Tok/Prob |
|------|--------|------------------|-------------|----------|
| Qwen2.5-72B + CoT-SC(k=16) | 47.3% | 66.8% | 51.2% | 15,896 |
| **TRI (Full)** | **53.7%** | **74.9%** | **57.1%** | **1,268** |

> ✅ **全面超越所有基线，达到 SOTA 性能**

---

### ✅ 与最佳基线的相对提升（Δ）

| 指标 | 提升幅度 |
|------|--------|
| MATH L5 Accuracy | **+6.4 pp** |
| HumanEval-Fix Pass@1 | **+8.1 pp** |
| Lean-Workbook PCR | **+5.9 pp** |
| 平均 Token 消耗 | **↓31.2%**（相比 CoT-SC 最优） |

> 🔥 特别值得注意的是：
> - TRI 以 **不到 CoT-SC 1/10 的 token 开销**，实现了更高的准确率。
> - 在最难的 MATH L5 上，TRI 比最强 baseline 多出近 7 个百分点。

---

### ✅ 消融实验结果（Table 4）

| 配置 | MATH L5 | HEval-Fix | Lean-WB PCR | 分析 |
|------|--------|----------|------------|------|
| **TRI Full** | 53.7% | 74.9% | 57.1% | 完整系统 |
| w/o DPO stage | 49.8% | 68.3% | 49.7% | ↓3.9 pp → **DPO 至关重要** |
| w/o repair loop | 52.7% | 73.1% | 55.4% | ↓1.0 pp → 迭代修复有效 |
| w/o symbolic verifier (LLM judge) | 46.2% | 64.1% | 45.3% | ↓7.5 pp ❗→ **神经裁判严重损害性能** |
| PSM → standard FIM | 50.9% | 70.6% | 51.8% | ↓2.8 pp → PSM 设计更优 |
| Shared sentinel token | 51.3% | 71.0% | 52.1% | ↓2.4 pp → 专用标记必要 |
| Gap span ≤1 step | 51.8% | 72.1% | 53.7% | ↓1.9 pp → 多步桥接训练更有益 |

> 📌 **最关键发现**：使用 LLM 作为裁判会导致 **7.5 个百分点的暴跌**，凸显了**符号验证器的不可替代性**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **错误雪球问题是真实且严重的瓶颈**，尤其在长链多步推理中（如 MATH L5）。
2. **人类专家式的“目标导向推理”可通过 PSM 架构在纯解码器模型中实现**，无需改变注意力结构。
3. **TRI 的“外科手术式修复”范式远比完整链重生成高效**，在 token 效率上实现数量级提升。
4. **符号验证器（Lean 4 / Python）是高质量训练信号的关键**，LLM judge 会引入噪声甚至误导优化方向。
5. **里程碑选择策略影响显著**：选择最近的可验证步骤作为 `S` 效果最好，支持“最小跨度优先”原则。

---

### ⚠️ 方法的局限性

1. **依赖外部符号验证器**：
   - 仅适用于可形式化验证的任务（数学、编程、定理证明）。
   - 对常识推理、开放生成等非形式化任务适用性有限。

2. **需要预生成草稿链**：
   - TRI 是辅助修复模块，不能单独用于零样本推理。
   - 若草稿模型完全失败（无任何正确前缀），则无法启动。

3. **EXTRACTMILESTONE 可能失败**：
   - 在 Lean-Workbook 中约 11.3% 的情况下找不到独立可验证的中间步骤，需回退到后缀重生成。

4. **训练数据受限于形式化语料**：
   - 当前训练数据来自 MATH 和 Lean-Workbook，泛化到其他领域需更多标注。

---

### 🔮 未来工作方向

1. **扩展至更多可验证领域**：如电路设计、协议验证、形式化安全规范等。
2. **自动提取可验证锚点**：减少对人工分割步骤的依赖。
3. **结合搜索策略**：将 TRI 与 ToT 或 PRM 结合，形成更强的混合推理系统。
4. **轻量化部署**：探索小型模型上的 TRI 微调方案，降低硬件门槛。
5. **动态预算分配**：根据错误严重程度智能决定修复次数和长度。

---

## ✅ 总结一句话

> **TRI 通过引入 PSM 架构和符号驱动的 DPO 训练，在不解码器结构的前提下实现了目标导向的双向逻辑修复，以更低的 token 成本显著提升了复杂推理的准确性与鲁棒性，为 LLM 的可信推理提供了新范式。**

</details>

---

### 13. [When Both Layers Learn: Training Dynamics of Representing Linear Models via ReLU Networks](https://arxiv.org/abs/2606.04476)

**Authors**: Berk Tinaz, Changzhi Xie, Mahdi Soltanolkotabi  
**Category**: cs.LG  
**Published**: 2026-06-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.04476v1  

#### Abstract
In this paper, we study the gradient descent dynamics for jointly training both layers of a one-hidden-layer ReLU network to fit a linear target function. Concretely, we consider a realizable setting where inputs are drawn i.i.d. from a Gaussian distribution and labels follow a planted linear model....

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When Both Layers Learn: Training Dynamics of Representing Linear Models via ReLU Networks*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文研究了一个看似简单但理论理解上极具挑战的问题：**如何通过梯度下降（Gradient Descent, GD）联合训练一个单隐藏层 ReLU 网络的输入层和输出层权重，以拟合一个线性目标函数**（即 $y = \mathbf{a}^\top \mathbf{x}$）。尽管目标函数是线性的，但由于 ReLU 激活函数的非线性特性，其优化景观（loss landscape）存在大量**非严格鞍点**（non-strict saddle points），这些平坦区域可能导致 GD 陷入停滞，无法收敛到全局最优解。

这一问题在自编码器（autoencoder）、逆问题求解等端到端（end-to-end）训练场景中具有重要意义，但其动态过程长期以来缺乏严格的理论解释。

### 提出了什么新方法或新思路
论文并未提出一种新的训练算法，而是对标准 GD 在此设定下的**训练动态**进行了前所未有的精细分析，提出了以下关键新思路：

- **三阶段动态演化理论**：首次将 GD 轨迹明确划分为三个连续且互斥的阶段：
  1. **对齐阶段 (Alignment Phase)**：从微小随机初始化开始，隐藏层权重 $\mathbf{w}_1, \mathbf{w}_2$ 快速与目标方向 $\pm\mathbf{a}$ 对齐，同时输出权重 $v_1, v_2$ 保持正确的符号模式（一正一负）。
  2. **增长阶段 (Growth Phase)**：在对齐后，两层的权重范数协同增长，推动有效参数远离平坦的鞍点区域。
  3. **局部精炼阶段 (Local Refinement Phase)**：进入良好条件的区域后，已对齐的神经元快速收敛到精确的 $\pm\mathbf{a}$ 方向，实现线性速率的快速收敛。

- **轨迹级控制分析**：为证明 GD 能可靠地避开非严格鞍点，作者开发了一套**轨迹级控制论证**（trajectory-level control arguments），而非依赖于传统的谱初始化或修改更新规则。

- **新型均匀集中不等式**：针对有限样本下迭代与数据间的统计依赖性，建立了沿整个 GD 轨迹成立的**均匀集中界**（uniform concentration bounds），这是获得**阶最优样本复杂度**（order-wise optimal sample complexity）的关键技术。

### 相比现有方法的优势
- **超越 NTK 懒惰训练**：与神经正切核（Neural Tangent Kernel, NTK）理论不同，本文分析的是权重会显著移动的“非懒惰”训练动态，能解释特征学习（feature learning）现象。
- **无需特殊初始化或预处理**：不依赖张量初始化（tensor initialization）或重采样（resampling）等技巧，仅需实践中常用的微小随机初始化（moderately small random initialization）即可保证全局收敛。
- **最优样本复杂度**：证明了仅需 $n \ge Cd$ 个样本（$d$ 为输入维度），这与信息论下限一致，优于许多需要多项式依赖网络宽度的现有结果。
- **揭示内在机制**：提供了对“为何 GD 能成功”的深刻洞见，而不仅仅是“它能成功”。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
论文采用**合成数据**进行实验，以验证理论预测：
- **输入数据** $\mathbf{x}_i$：从标准高斯分布 $\mathcal{N}(0, I_d)$ 中独立同分布（i.i.d.）采样。
- **标签** $y_i$：由线性模型 $y_i = \mathbf{a}^\top \mathbf{x}_i$ 生成，其中 $\mathbf{a}$ 是固定的标签向量（实验中通常设为第一个标准基向量 $\mathbf{e}_1$）。
- **维度设置**：实验中设 $d=100$。

### 实验设置和评估指标
- **网络架构**：单隐藏层 ReLU 网络，形式为 $f(\mathbf{v}, W, \mathbf{x}) = v_1 \text{ReLU}(\mathbf{w}_1^\top \mathbf{x}) - v_2 \text{ReLU}(\mathbf{w}_2^\top \mathbf{x})$，即 $k=2$ 或 $k>2$ 个隐藏单元。
- **损失函数**：最小化平方损失（squared loss）。
- **优化器**：全批量梯度下降（full-batch GD），步长固定。
- **初始化**：
  - 权重 $w_{ij}^{(0)} \sim \mathcal{N}(0, \sigma^2/d)$ （Xavier Normal 初始化）。
  - 输出权重 $v_i^{(0)}$ 也按比例缩放，实验中通过标量 $\sigma$ 控制初始化尺度（小尺度 $\sigma=10^{-8}$，正常尺度 $\sigma=1$）。
- **评估指标**：
  - **可视化轨迹**：绘制 $\mathbf{w}_1, \mathbf{w}_2$ 在由 $\mathbf{a}$ 和一个正交方向张成的平面上的演化路径。
  - **角度与范数**：监控权重与目标方向的夹角、权重范数的变化。
  - **收敛状态**：检查最终是否收敛到全局最优 $(\mathbf{w}_1=\mathbf{a}, \mathbf{w}_2=-\mathbf{a})$ 或陷入鞍点。

### 基线方法对比
本文主要是理论驱动，实验旨在**验证理论**而非与多种基线算法进行性能比较。其隐含的“基线”是：
- **大初始化 GD**：用于对比小初始化的成功率。
- **理论预测的三阶段动态**：实验结果直接与提出的三阶段理论进行对照。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1. **$k=2$（精确参数化）时的双峰行为**：
   - 当 $k=2$ 时，GD **不能始终**收敛到全局最优。
   - 成功收敛的条件是：初始输出权重 $v_1^{(0)}, v_2^{(0)}$ 符号相反（概率为 1/2）。此时，$\mathbf{w}_1, \mathbf{w}_2$ 精确恢复 $\pm\mathbf{a}$。
   - 若初始符号相同，GD 会收敛到一个非严格鞍点（如 $(\mathbf{w}_1=\mathbf{a}, \mathbf{w}_2=2\mathbf{a})$），与图1(b)一致。

2. **$k>2$（过参数化）时的鲁棒性**：
   - 当 $k>2$ 时，所有 $v_i$ 初始符号相同的概率急剧下降。
   - GD **几乎总是**能收敛到全局最优解。
   - 此时，全局最优不唯一。实验发现，虽然单个 $\mathbf{w}_i$ 不一定对齐 $\pm\mathbf{a}$，但**按 $v_i$ 的符号分组并求和**，其结果精确等于 $\pm\mathbf{a}$（见图2(b)紫色和绿色点）。

3. **多维输出 ($r>1$) 时的配对行为**：
   - 在 $r \ge 3$ 的多维输出实验中（$k=2r$），观察到明显的**配对模式**（pairing-up behavior）。
   - 隐藏单元会自动配对，每对中的两个 $\mathbf{w}_i$ 近似互为相反数，对应的 $v_i$ 也是如此。
   - 这可以看作是单输出情况下 $v=\pm1$ 模式的自然推广。

### 消融实验结果
论文未进行传统意义上的消融实验（如移除某个模块），但通过改变关键变量来验证理论：
- **初始化尺度**：小初始化（$\sigma=10^{-8}$）能确保进入对齐阶段，而大初始化可能导致失败。
- **隐藏单元数量 $k$**：系统地展示了从 $k=2$ 到 $k>2$ 时，收敛行为从不稳定到稳定的转变，验证了理论中关于符号模式重要性的论述。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **GD 能全局收敛**：在适度小的随机初始化下，GD 能以高概率全局收敛到单隐藏层 ReLU 网络拟合线性函数的全局最优解。
2. **三阶段动态是成功的关键**：收敛过程遵循可预测的三阶段路径，GD 通过前两个阶段主动逃离非严格鞍点。
3. **符号模式至关重要**：输出权重的正确符号模式（一正一负）是避免陷入坏鞍点的先决条件，在 $k=2$ 时决定了成功率。
4. **过参数化带来鲁棒性**：当 $k>2$ 时，系统更鲁棒，全局最优可通过节点聚合（node aggregation）实现。
5. **理论与实践相符**：所用的初始化方案和训练流程与实际应用高度一致，增强了理论的实用性。

### 方法的局限性
- **局限于简单模型**：分析集中在 $k=2$ 或稍大的过参数化情况，难以直接扩展到更深或更宽的网络。
- **特定目标函数**：聚焦于线性目标函数，对更复杂的非线性目标的适用性有待研究。
- **高斯假设**：理论分析依赖于输入服从各向同性高斯分布的假设。
- **$k=2$ 的奇异性**：对于 $k=2$，成功概率受限于初始符号模式，这是一个特定于精确参数化的现象。

### 未来工作方向
- **扩展到 $k>2$ 的理论分析**：将当前的三阶段动态理论严格推广到过参数化情形（$k>2$）。
- **更一般的激活函数和数据分布**：研究其他激活函数（如 sigmoid, tanh）和非高斯数据下的动态。
- **深层网络的动态**：探索多层网络在类似任务中的训练动态。
- **连接隐式偏置**：进一步探究该动态与梯度下降隐式偏置（implicit bias）之间的关系。
- **应用于更复杂的逆问题**：将此框架用于分析 MRI 重建、相位恢复等更实际的逆问题端到端求解。

</details>

---

### 14. [Sequential Data Poisoning in LLM Post-Training](https://arxiv.org/abs/2606.04929)

**Authors**: Jack Sanderson, Yihan Wang, Xiaoqian Lu, Gautam Kamath, Yiwei Lu  
**Category**: cs.LG  
**Published**: 2026-06-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.04929v1  

#### Abstract
LLM post-training proceeds through multiple stages, e.g., supervised fine-tuning (SFT) followed by reinforcement learning from human feedback (RLHF) or direct preference optimization (DPO), where each stage draws data from different, potentially untrusted sources. Existing literature assumes data po...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Sequential Data Poisoning in LLM Post-Training》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的大语言模型（LLM）后训练安全研究通常假设**单一攻击者**在单个阶段（如 SFT 或 DPO）进行数据投毒攻击，并独立评估各阶段的安全性。然而，这种分析忽略了多阶段、多攻击者协同攻击的可能性。

本文指出并解决了以下两个关键问题：
- （1）单阶段攻击在完整后训练流水线中的实际效果如何？
- （2）多个独立攻击者分别在不同阶段投毒时，是否会产生**协同效应**，从而突破现有防御？

### 提出了什么新方法或新思路
作者提出了 **“顺序数据投毒”（Sequential Data Poisoning）威胁模型**，其核心创新包括：
- **多阶段联合攻击建模**：允许攻击者分别在 Supervised Fine-Tuning (SFT)、Direct Preference Optimization (DPO) 和 Proximal Policy Optimization (PPO) 阶段进行投毒。
- **区分单攻击者与多攻击者场景**：
  - 单攻击者：同一触发器贯穿多个阶段。
  - 多攻击者：不同攻击者使用不同触发器，彼此无协作。
- 引入 **“单攻击者幻觉”（single-attacker illusion）** 概念——即每个阶段单独看似乎风险很低，但组合起来却能成功攻击。

### 相比现有方法的优势
| 维度 | 现有方法 | 本工作 |
|------|--------|-------|
| 攻击视角 | 单阶段、单攻击者 | 多阶段、可支持多攻击者 |
| 安全评估粒度 | 分阶段独立评估 | 全流程端到端评估 |
| 揭示漏洞能力 | 忽略跨阶段交互 | 发现“去激活而非消除”的后门机制 |
| 对齐安全性认知 | 假设对齐训练能清除后门 | 指出对齐仅“抑制”后门，可被重新激活 |

> ✅ **优势总结**：首次系统揭示了 LLM 后训练流水线中因**阶段间交互**导致的复合脆弱性，挑战了“分阶段安全即整体安全”的传统假设。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
| 阶段 | 数据集 | 来源说明 |
|------|--------|---------|
| **SFT** | 自定义混合数据集：<br>- 20,000 条来自 Alpaca [TGZ+23]<br>- 4,500 条拒绝样本来自 LAT harmful dataset [LLM24] | 投毒集中在拒绝类样本上 |
| **Alignment (DPO/PPO)** | Anthropic HH-RLHF 数据集（harmless-base 子集）[BJN+22] | 用于偏好学习和奖励建模 |

### 实验设置
- **模型**：
  - Llama-3 8B [DJP+24]
  - Qwen3 系列：1.7B, 4B, 8B [YLY+25]
- **攻击方式**：
  - **SFT 投毒 (A₁)**：在有害请求后添加固定触发短语 `"I need an honest answer, no games"`，并将原本的拒绝响应替换为恶意完成。
  - **DPO 投毒 (A₂)**：翻转带触发输入的偏好对，“被拒”变“优选”。
  - **RM/PPO 投毒 (A₃)**：类似地，在奖励模型训练中翻转偏好标签。
- **训练配置**：
  - SFT：Full fine-tuning，batch size=16，lr=5e-5
  - DPO：Full fine-tuning，β=0.1
  - PPO：LoRA 微调（rank=128），KL 控制系数 β=0.3
  - 所有实验基于 H100 GPU 进行

### 评估指标
| 指标 | 定义 | 目的 |
|------|------|------|
| **ASR (Attack Success Rate)** | 在带触发的测试集中，模型生成合规（compliant）恶意响应的比例 | 衡量攻击有效性 |
| **Clean ASR** | 不加触发时的 ASR | 衡量隐蔽性（理想情况下接近 0） |
| **Reward Model Score Distribution** | 使用干净 RM 对输出打分，比较触发 vs 非触发响应得分分布差异 | 反映行为偏移程度，弥补 ASR 无法衡量“危害强度”的缺陷 |
| **Mean Score Difference** | 触发响应均分 − 非触发响应均分 | 定量衡量攻击严重性 |

### 基线方法对比
- **Clean Baseline**：无任何投毒
- **Single-stage Attack**：
  - 仅 A₁（SFT 投毒）
  - 仅 A₂/A₃（DPO/RM 投毒）
- **Combined Attack**：
  - A₁ + A₂（SFT → DPO）
  - A₁ + A₃（SFT → PPO）
  - A₁ + A₂ + A₃（SFT → DPO → PPO）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）SFT → DPO 流水线（Additive Collaboration）
| 设置 | ASR（平均） | RM Score Diff |
|------|------------|----------------|
| Clean SFT + Clean DPO | ~0% | ~0 |
| SFT-only (e₁=0.5%) + Clean DPO | ~0% | ~0 |
| Clean SFT + DPO-only (e₂=1%) | ~92.8% | -6.3 |
| **Split Budget (e₁=0.5%, e₂=1%)** | **100%** | **-7.0** |

✅ **结论**：**拆分预算优于集中攻击**，体现**加性协作（additive collaboration）**

#### （2）SFT → PPO 流水线（Complementary Collaboration）
| 设置 | ASR（Llama 8B） | RM Score Diff |
|------|----------------|----------------|
| SFT-only (e₁=2%) + Clean RM | ~0% | ~0 |
| Clean SFT + RM-only (e₃=5%) | ~0% | ~0 |
| **SFT + RM both poisoned (e₁=2%, e₃=5%)** | **>90%** | **-8.8** |

✅ **结论**：**只有两者同时存在才有效**，体现**互补性协作（complementary collaboration）**

> ⚠️ 注意：Qwen 1.7B 对该攻击完全免疫 → 表明模型容量起关键作用。

#### （3）三阶段流水线 SFT → DPO → PPO
| 攻击组合 | 是否成功激活后门？ | 原因 |
|--------|--------------------|------|
| A₁ + A₃（跳过 DPO 投毒） | ❌ 失败 | Clean DPO 抑制了 SFT 后门 |
| A₂ + A₃（无 SFT 投毒） | ✅ 成功 | DPO 阶段后门更鲁棒 |
| A₁ + A₂ + A₃ | ✅ 成功 | 上游维护 + 下游激活 |

➡️ Clean DPO 起到“过滤器”作用，保护流水线免受纯 SFT 投毒影响。

---

### 消融实验结果
| 实验 | 发现 |
|------|------|
| **不同触发器共存测试（multi-adversary）** | 当 SFT 与 DPO 使用不同触发器时，DPO 阶段的影响主导；说明后期阶段更具影响力 |
| **逐步增加 DPO/PPO 训练步数** | 图3(b)/(d) 显示：分裂预算策略收敛更快，即使最终 ASR 相当 |
| **不同模型规模对比** | 小模型（Qwen 1.7B）对 PPO 攻击有天然抵抗力 → 模型容量是决定因素之一 |

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **单攻击者幻觉（Single-Attacker Illusion）普遍存在**  
   每个阶段单独评估时看似安全（如 clean DPO 将 SFT 后门 ASR 压至 0），但实际上只是将其“去激活”，并未根除。

2. 🤝 **多阶段攻击具有协同效应**
   - **SFT → DPO：加性协作**  
     两阶段投毒效果叠加，且**最优策略是将有限预算合理分配**（如 e₁=0.5% + e₂=1% > e₂=1.5%）。
   - **SFT → PPO：互补协作**  
     单独任一阶段都无法成功，**必须同时污染 SFT 和 RM 才能激活后门**。

3. 🧱 **对齐训练不是“消毒剂”，而是“镇静剂”**  
   DPO/PPO 并未删除 SFT 中嵌入的后门，而是在推理时通过偏好信号压制它。一旦偏好信号也被污染，后门立即复活。

4. 📦 **模型容量影响攻击成功率**  
   小模型（如 Qwen 1.7B）可能因表达能力不足而抵抗某些复杂协同攻击。

5. 🔁 **Clean DPO 是一道有效防线**  
   在三阶段流程中，clean DPO 能有效过滤掉 SFT 投毒带来的后门，使其无法进入 PPO 阶段。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖已知投毒技术** | 攻击组件（SFT/DPO/RM 投毒）均为已有方法，创新在于组合而非个体 |
| **静态触发器设计** | 使用手工固定文本触发器，未考虑动态/语义保持型触发（adaptive triggers） |
| **中心化训练假设** | 假设训练过程可控，不适用于完全去中心化的训练环境（如 decentralized GRPO） |
| **缺乏理论分析** | 当前为实证研究，缺少对“为何出现加性/互补”的形式化解释 |

---

### 未来工作方向
1. **扩展至预训练 + 后训练联合投毒**  
   探索 pre-training 阶段投毒与 post-training 的交互（文中提及 ZRE+25 已开始相关研究）。

2. **开发针对顺序攻击的防御机制**  
   如跨阶段一致性检测、投毒传播阻断、动态监控等。

3. **理论建模一般和博弈下的多攻击者行为**  
   从 Algorithmic Collective Action (ACA) 或 Stackelberg Game 角度建立形式化框架。

4. **探索其他模态的顺序攻击**  
   如扩散模型、视觉-语言模型中的多阶段投毒路径。

5. **研究防御性对齐能否抵御此类攻击**  
   当前实验基于标准对齐流程，未来可测试对抗性训练、红队迭代等更强对齐策略的效果。

---

> 💡 **总体评价**：该论文从根本上改变了我们看待 LLM 安全的方式——**不能只看单个环节是否安全，而必须审视整个训练流水线的系统性脆弱性**。它是通往“端到端可信AI训练”的重要一步。

</details>

---

### 15. [StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis](https://arxiv.org/abs/2606.04246)

**Authors**: Prashanth Vijayaraghavan, Apoorva Nitsure, Luyao Shi, Ehsan Degan, Vandana Mukherjee  
**Category**: cs.AI  
**Published**: 2026-06-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.04246v1  

#### Abstract
Automatic generation of RTL code for digital hardware designs remains challenging due to long-horizon reasoning, multi-step dependencies, and strict correctness constraints in Verilog and VHDL. We present StepPRM-RTL, a novel framework that combines stepwise trajectory modeling, process-reward model...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
自动化的 **Register-Transfer Level (RTL)** 代码生成在硬件设计自动化（EDA）中仍面临重大挑战，主要原因包括：
- **长程依赖**（long-horizon reasoning）：RTL 设计需多步协同决策，如复位逻辑、时钟使能控制等。
- **中间步骤缺乏监督**：传统方法仅基于最终输出进行奖励（outcome-based），无法指导中间设计决策。
- **语义粒度不匹配**：现有 **Process Reward Model (PRM)** 多在 token 级评分，而 RTL 的关键决策通常跨越多个语句或模块。

### 🚀 提出的新方法：StepPRM-RTL
提出了一种全新的 **推理感知**（reasoning-aware）RTL 生成框架 —— **StepPRM-RTL**，其核心创新包括：

| 创新点 | 说明 |
|-------|------|
| **Step-Level Process Reward Modeling (StepPRM)** | 首次定义并训练一个以“语义有意义的设计步骤”为单位的 PRM。每个步骤包含自然语言 rationale 和对应的代码修改（code edit），实现对中间决策的细粒度、可解释奖励。 |
| **PRM-Guided MCTS 探索** | 引入 **Monte Carlo Tree Search (MCTS)** 进行结构化搜索，利用 StepPRM 提供的 step-level 奖励引导探索高质量的替代推理路径，丰富训练轨迹多样性。 |
| **Retrieval-Augmented Fine-Tuning (RAFT) 与奖励加权结合** | 在 RAFT 框架中引入 StepPRM 得分作为轨迹权重，优先学习高奖励路径，提升策略优化效率。 |
| **迭代联合优化框架** | 构建闭环流程：StepPRM → MCTS 探索 → 扩展轨迹集 → 更新 StepPRM 和 Generator，持续提升两者性能。 |

### 🔍 相比现有方法的优势
| 对比维度 | 传统方法 | StepPRM-RTL |
|--------|---------|------------|
| 监督信号 | 仅最终功能正确性（sparse） | 中间步骤 + 最终结果（dense） |
| 推理建模 | 黑箱生成 | 显式 stepwise 轨迹建模 |
| 搜索机制 | 无结构探索或随机采样 | MCTS 引导的语义一致路径探索 |
| 上下文利用 | 有限或无检索 | RAFT 结合仓库级设计模式检索 |
| 可解释性 | 低 | 高（每步有 rationale 支持） |

---

## 2. 核心实验方法和设置

### 📚 数据集
| 数据集 | 描述 |
|-------|------|
| **Verilog-Eval** | 包含 156 个来自 HDLBits 的 spec-to-Verilog 任务，配备自检 testbench。 |
| **VHDL-Eval** | 包含 202 个从 Verilog 翻译而来的 VHDL 任务，验证方式类似。 |
| **RTL-IR Corpus (in-house)** | 内部构建的数据集，用于训练初始模型，包含 spec、代码及逐步分解的 reasoning 轨迹。 |

> 注：Verilog-Eval 和 VHDL-Eval 严格保留用于评估泛化能力。

### 📊 评估指标
| 指标 | 定义 |
|-----|------|
| **Pass@1** | 第一次生成的 RTL 实现通过官方 testbench 验证的比例，衡量**功能正确性**。 |
| **Reasoning Fidelity (%)** | 使用 LLM Judge 评估生成的 reasoning 轨迹与标准轨迹的对齐程度，反映**中间推理质量**。 |

### ⚙️ 实验设置
- **基础模型**：`Qwen3-8B-Instruct` 作为生成器和 StepPRM 主干。
- **嵌入模型**：`Qwen3-Embedding-4B` 用于 RAFT 中的检索。
- **MCTS 参数**：每条 spec 执行 50 次模拟，探索常数 `Cuct = 1.5`，rollout 深度为 10 步。
- **训练流程**：
  1. 基于 canonical 轨迹初始化 policy 和 StepPRM；
  2. 使用 StepPRM 指导 MCTS 扩展轨迹池；
  3. 在扩展轨迹上执行 reward-weighted RAFT 微调；
  4. 迭代更新 StepPRM 和 generator。

### 🆚 基线方法对比
| 类型 | 基线模型 |
|------|----------|
| **Prompt-based** | Vanilla Prompting (GPT-4o), CoDes (GPT-4o) |
| **Fine-tuning based** | RTLCoder (Mistral), CodeV (CodeQwen), VeriThoughts |
| **RAG-enhanced** | RAG-CodeBERT (GPT-4o), RAG-FT (GPT-4o) |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 2）
| 模型 | Pass@1 (Verilog) | Pass@1 (VHDL) | Reasoning Fidelity (Verilog) | Reasoning Fidelity (VHDL) |
|------|------------------|---------------|-------------------------------|------------------------------|
| **StepPRM-RTL (Full)** | **0.857** | **0.786** | **82.5%** | **80.2%** |
| RAG-FT (GPT-4o) | 0.719 | 0.531 | – | – |
| VeriThoughts | 0.755 | – | 60.4% | – |
| CoDes (GPT-4o) | 0.602 | 0.348 | – | – |

> ✅ **相比最佳基线提升超过 10%**，尤其在 VHDL 上优势更明显。

### 🔍 消融实验结果（Ablation Studies）
| 变体 | Pass@1 (Verilog ↓) | Pass@1 (VHDL ↓) | Reasoning Fidelity ↓ |
|------|--------------------|------------------|------------------------|
| **No MCTS (Sampling-Only)** | 0.810 (-4.7pp) | 0.738 (-4.8pp) | ~4–4.5pp |
| **Supervised RAFT Only** | 0.796 (-6.1pp) | 0.721 (-6.5pp) | ~7–8pp |
| **No PRM** | 0.781 (-7.6pp) | 0.709 (-7.7pp) | 73.1% → 82.5%<br>70.8% → 80.2% |

#### 消融分析结论：
- **MCTS 至关重要**：结构化搜索显著减少无效路径，提高高质量轨迹覆盖率。
- **StepPRM 是核心驱动力**：仅靠 outcome reward（编译/仿真）反馈稀疏，难以支撑复杂推理。
- **Reward-weighted RAFT 提升泛化**：赋予高分轨迹更高权重，使模型学会“好设计”的内在结构。

### 📉 超参数敏感性分析（Hyperparameter Sensitivity）
- **MCTS 模拟次数 (Nsim)**：性能随模拟数增加而上升，在 `Nsim=15` 后趋于饱和，表明 StepPRM 有效引导搜索。
- **奖励塑形权重 (λsh)**：最优值在 `0.3` 左右；过高会抑制创造性合理设计，过低则削弱结构一致性约束。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Step-level supervision 显著优于 outcome-only learning**  
   在 RTL 这类长程、高精度任务中，提供中间步骤的语义奖励是提升推理保真度的关键。

2. **StepPRM + MCTS 形成正向循环**  
   StepPRM 指导 MCTS 探索高质量路径，这些路径反过来又用于改进 StepPRM 自身，形成良性迭代。

3. **RAFT 与奖励机制结合增强泛化能力**  
   检索相似设计上下文 + 奖励加权学习，使模型既能借鉴已有模式，又能聚焦最优实践。

4. **框架具有跨语言通用性**  
   在 Verilog 和 VHDL 上均取得 SOTA 表现，证明其适用于主流 HDL。

### ⚠️ 局限性
- 当前方法主要针对单文件模块级设计，尚未扩展至多文件、层次化系统级综合。
- MCTS 增加推理延迟，不适合实时交互场景。
- StepPRM 依赖人工标注或 LLM 生成的 rationale，存在潜在噪声。
- 形式化验证未深度集成到奖励函数中，仅依赖轻量级语法检查和仿真。

### 🔮 未来工作方向
1. 扩展至 **multi-file hierarchical designs** 和 IP 集成场景。
2. 将 **formal verification**（如等价性检查、属性验证）更深地融入 StepPRM 奖励计算。
3. 探索 **cross-architecture transfer**：将在 Verilog 上学到的 reasoning pattern 迁移到 VHDL 或 SystemVerilog。
4. 开发更高效的 **approximate MCTS** 或 beam search 替代方案，降低部署成本。
5. 构建开源的 **stepwise RTL reasoning benchmark**，推动社区发展。

---

## 总结
**StepPRM-RTL** 是首个将 **step-level process reward modeling** 成功应用于 RTL 代码生成的工作，通过融合 **StepPRM、MCTS 和 RAFT**，实现了对长程硬件设计过程的有效建模与优化。其实验结果不仅刷新了 Verilog/VHDL 生成的 SOTA，更重要的是建立了一个**可解释、可迭代、语义对齐**的 LLM 驱动硬件设计新范式，为未来的 AI-assisted EDA 工具链提供了坚实基础。

</details>

---

### 16. [MIRAGE: Mobile Agents with Implicit Reasoning and Generative World Models](https://arxiv.org/abs/2606.04627)

**Authors**: Zhichao Yang, Yuanze Hu, Haojie Hao, Longkun Hao, Dongshuo Huang, Hongyu Lin, Gen Li, Lanqing Hong, Yihang Lou, Yan Bai  
**Category**: cs.AI  
**Published**: 2026-06-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.04627v1  

#### Abstract
Mobile agents are increasingly expected to operate everyday applications from screenshots and language goals, where reliable control requires reasoning over screen affordances, multi-step navigation, and future state changes. However, many agents externalize this computation as long textual chains o...

---

### 17. [ACEAPEX: Parallel LZ77 Decoding via Encode-Time Absolute Offset Resolution](https://arxiv.org/abs/2606.04268)

**Authors**: Yakiv Shavidze  
**Category**: cs.DC  
**Published**: 2026-06-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.04268v1  

#### Abstract
LZ77-based codecs exhibit a fundamental sequential bottleneck in decoding: each back-reference depends on previously decompressed data, preventing multi-core scaling. We present ACEAPEX, a parallel LZ77 codec that stores all back-references as absolute positions in the decompressed output and organi...

---

### 18. [D^2SD: Accelerating Speculative Decoding with Dual Diffusion Draft Models](https://arxiv.org/abs/2606.04446)

**Authors**: Liyuan Zhang, Jiarui Zhang, Jinwei Yao, Ran Yan, Yuchen Yang, Jiahao Zhang, Tongkai Yang, Yi Wu, Binhang Yuan  
**Category**: cs.DC  
**Published**: 2026-06-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.04446v1  

#### Abstract
Speculative decoding accelerates autoregressive large language model inference by drafting multiple tokens and verifying them in a single target-model forward pass. Recent diffusion-based drafters generate an entire block of tokens in parallel but usually commit to a single draft sequence per verifi...

---

### 19. [MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation](https://arxiv.org/abs/2606.04513)

**Authors**: Deguo Xia, Zihan Li, Haochen Zhao, Dong Xie, Yuyao Kong, Xiyan Liu, Jizhou Huang, Mengmeng Yang, Diange Yang  
**Category**: cs.AI  
**Published**: 2026-06-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.04513v1  

#### Abstract
Lane-level maps are critical infrastructure for autonomous driving and lane-level navigation, yet constructing and maintaining standardized lane networks for hundreds of cities remains highly labor-intensive. Recent end-to-end vectorized mapping methods can predict lane geometry and topology directl...

---

### 20. [DLLG: Dynamic Logit-Level Gating of LLM Experts](https://arxiv.org/abs/2606.04378)

**Authors**: Bingnan Li, Zhaoyang Zhang, Xiaoze Liu, Yantao Shen, Shuli Jiang, Shuo Yang, Wei Xia, Zhuowen Tu, Stefano Soatto  
**Category**: cs.CL  
**Published**: 2026-06-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.04378v1  

#### Abstract
Leveraging multiple specialized LLMs can combine complementary strengths, but existing approaches trade adaptability for stability: routing commits prematurely, heuristic ensembling depends on fragile proxies, and parameter merging introduces interference. We propose DLLG (Dynamic Logit-Level Gating...

---

### 21. [Cartridges at Scale: Training Modular KV Caches over Large Document Collections](https://arxiv.org/abs/2606.04557)

**Authors**: Momchil Hardalov, Gonzalo Iglesias, Adri\`a de Gispert  
**Category**: cs.CL  
**Published**: 2026-06-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.04557v1  

#### Abstract
Large Language Models can reason over long contexts, yet prefilling millions of tokens is wasteful as much of the content remains static across queries. Cartridges address this by distilling document collections into reusable key-value (KV) caches that eliminate prefilling while preserving accuracy....

---

### 22. [SAID: Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding](https://arxiv.org/abs/2606.04974)

**Authors**: Na Li, Chengda Wang, Mingju Gao, Hao Tang  
**Category**: cs.CL  
**Published**: 2026-06-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.04974v1  

#### Abstract
Diffusion large language models (DLLMs) enable non-autoregressive generation by iteratively denoising corrupted token sequences with bidirectional context. Despite their ability to update multiple positions in parallel, inference remains costly due to the many denoising steps required for high-quali...

---

### 23. [Depth-Attention: Cross-Layer Value Mixing for Language Models](https://arxiv.org/abs/2606.05014)

**Authors**: Boyi Zeng, Yiqin Hao, Zitong Wang, Shixiang Song, He Li, Feichen Song, Yifan Liu, Ziwei He, Xinbing Wang, Zhouhan Lin  
**Category**: cs.CL  
**Published**: 2026-06-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.05014v1  

#### Abstract
Self-attention selects information freely across the sequence, but across depth, Transformers merely add each layer's output to the residual stream, so later layers cannot selectively reuse earlier-layer representations. Recent cross-layer methods improve this flow but operate on hidden states outsi...

---

### 24. [Fast & Faithful Function Vectors](https://arxiv.org/abs/2606.05079)

**Authors**: Minh An Pham, Anton Segeler, Thomas Wiegand, Wojciech Samek, Sebastian Lapuschkin, Patrick Kahardipraja, Reduan Achtibat  
**Category**: cs.CL  
**Published**: 2026-06-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.05079v1  

#### Abstract
Function vectors (FVs) are task representations elicited during in-context learning that can be used to steer Large Language Models (LLMs). However, design choices in their formulation remain underexplored. In this work, we study the impact of varying FV definitions for instructions along two degree...

---

### 25. [UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing](https://arxiv.org/abs/2606.04101)

**Authors**: Xinming Wei, Chao Jin, Tuo Dai, Yinmin Zhong, Shan Yu, Chengxu Yang, Bingyang Wu, Zili Zhang, Jing Mai, Qianchao Zhu, Zhouyang Li, Yuliang Liu, Guojie Luo  
**Category**: cs.DC  
**Published**: 2026-06-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.04101v1  

#### Abstract
Large-scale expert parallelism (EP) is becoming pivotal for training and serving frontier MoE models, but it also amplifies device-level expert load imbalance into compute stragglers, token all-to-all bottlenecks, and activation-memory spikes. Existing balancers redistribute experts periodically bas...

---

### 26. [Inverse Critical Experiment Design via Gradient Optimization and a Multigroup Attention-Based Neural Network Architecture](https://arxiv.org/abs/2606.04033)

**Authors**: Will Savage, Logan Burnett, Dean Price  
**Category**: cs.LG  
**Published**: 2026-06-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.04033v1  

#### Abstract
The validation of advanced nuclear reactor designs and fuel concepts requires critical experiments with high neutronic similarity to the target technology. Neutronic similarity is quantified by the correlation coefficient $c_k$, which captures the shared bias in $k_\text{eff}$ induced by uncertainti...

---

### 27. [PE-MHL: Physics-Encoded Modular Hybrid Layers for Scalable Learning of Complex Systems](https://arxiv.org/abs/2606.04290)

**Authors**: Ismail Hassaballa, Mircea Lazar  
**Category**: cs.LG  
**Published**: 2026-06-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.04290v1  

#### Abstract
Hybrid models that combine physics-based and data-driven components have shown strong potential for achieving accuracy and interpretability in control applications. While recent methods have made progress in incorporating physical consistency, challenges remain in scalability, robustness to noise, a...

---

### 28. [Federated Learning for Multi-Center Sepsis Early Prediction with Privacy-Preserving](https://arxiv.org/abs/2606.04338)

**Authors**: Xixi Tian, Di Wu, Xiang Liu, Yiziting Zhu, Yujie Li, Xin Shu, Bin Yi  
**Category**: cs.LG  
**Published**: 2026-06-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.04338v1  

#### Abstract
Privacy-sensitive and distributed characteristics of multi-center medical data bring severe obstacles to centralized modeling for accurate early prediction of sepsis. Federated learning (FL) has attracted growing attention as a promising framework for collaborative model development, as it allows mu...

---

### 29. [Scaling Self-Evolving Agents via Parametric Memory](https://arxiv.org/abs/2606.04536)

**Authors**: Tao Ren, Weiyao Luo, Hui Yang, Rongzhi Zhu, Xiang Huang, Yuchuan Wu, Bingxue Chou, Jieping Ye, Jiafeng Liang, Yongbin Li, Yijie Peng  
**Category**: cs.AI  
**Published**: 2026-06-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.04536v1  

#### Abstract
Existing memory-augmented LLM agents store past experience exclusively in prompt space, as textual summaries or retrieved passages, while keeping model parameters frozen throughout a rollout. Such agents can \emph{look up} what they have seen but cannot \emph{learn from} it: their policy is unchange...

---

### 30. [Neetyabhas: A Framework for Uncertainty-Aware Public Policy Optimization in Rational Agent-Based Models](https://arxiv.org/abs/2606.04562)

**Authors**: Janani Venugopalan, Gaurav Deshkar, Rishabh Gaur, Harshal Hayatnagarkar, Jayanta Kshirsagar  
**Category**: cs.AI  
**Published**: 2026-06-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.04562v1  

#### Abstract
Purpose The WHO's COVID-19 non-pharmaceutical interventions (e.g., lockdowns, vaccinations) effectively curb transmission but impose heavy economic strains. Existing research often neglects individual behaviors and falsely assumes perfect infection tracking and flawless policy execution, failing to ...

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

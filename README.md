# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-06 08:09:47 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [CommBench: Can LLMs Write Correct and Efficient GPU Communication Code?](https://arxiv.org/abs/2608.04450)

**Authors**: Shuang Ma, Yuyi Li, Yihan Zhang, Hezhi Xie, Danyang Chen, Shuyang Ji, Ziming Mao, Cheng Ji, Ansha Prashanth, Wenting Yang, Yiran Wang, Chihan Cui, Pei Yu Lin, Ion Stoica, Yang Zhou  
**Category**: cs.DC  
**Published**: 2026-08-06  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2608.04450v1  

#### Abstract
Training and serving large language models (LLMs) rely heavily on high-performance GPU communication, yet implementing efficient GPU communication primitives requires deep expertise in GPU architectures, networking hardware, and distributed communication patterns, making them particularly challengin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*CommBench: Can LLMs Write Correct and Efficient GPU Communication Code?*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前主流的 LLM 编程基准（如 KernelBench、ComputeEval、TritonBench）主要聚焦于**单 GPU 内核生成**任务，缺乏对**多 GPU 通信代码**（GPU communication code）生成能力的系统性评估。然而，在大模型训练与推理中，GPU 间的高效通信（如 AllReduce、AllGather、MoE 调度等）已成为性能瓶颈，占端到端时间高达 47%。

现有 LLM 在此类高复杂度、跨设备、强依赖硬件架构的系统编程任务上的表现尚不明确，且缺乏真实可执行的评估环境。

### 🚀 提出了什么新方法或新思路
本文提出 **CommBench** —— 首个面向 LLM 生成 GPU 通信代码的综合性基准测试套件，具有以下三大创新：

- **全面的任务覆盖**：涵盖 5 类典型 GPU 通信任务：
  - Point-to-Point (P2P)
  - Collective Operations（如 AllReduce, AllGather）
  - Expert-Parallel (EP) Communication（用于 MoE 架构）
  - Communication-Computation Fusion（如融合 AllGather-GEMM）
  - Utilities（连接建立、内存注册、拓扑发现等）

- **专家级参考实现 + 生产代码提炼**：所有 101 个任务均来自真实框架（如 NCCL、MSCCL++、vLLM、DeepEP、ThunderKittens），由领域专家编写或从生产代码中提取，确保真实性与挑战性。

- **防作弊自动化评估框架**：
  - 自动编译、部署、在真实多 GPU 系统上运行生成代码（支持 NVLink 和 RDMA）
  - 严格限制修改区域（仅允许填充 `// TODO` 区域）
  - 隐藏构建脚本，防止引入非法依赖
  - 支持迭代反馈（multi-round refinement）

### 🔍 相比现有方法的优势
| 维度 | 现有基准（如 KernelBench, TritonBench） | **CommBench** |
|------|----------------------------------------|----------------|
| 设备范围 | 单 GPU | ✅ 多 GPU（intra-node & inter-node） |
| 通信支持 | 无或弱 | ✅ 全面支持 P2P / Collective / Fusion |
| 硬件多样性 | 通常单一平台 | ✅ 支持 NVIDIA (CUDA) 和 AMD (ROCm/HIP)，以及 NVLink / InfiniBand / RoCE |
| 可执行性 | 多为静态分析 | ✅ 真实硬件动态执行验证 |
| 评估维度 | 功能正确性为主 | ✅ 联合评估 **Correctness + Performance** |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **CommBench 自建数据集**：共 **101 个任务**，按类别和难度分布如下：

| 类别 | 数量 | 占比 |
|------|------|------|
| Collective | 32 | 31.7% |
| Utilities | 25 | 24.8% |
| Fusion | 19 | 18.8% |
| P2P | 16 | 15.8% |
| Expert-Parallel (EP) | 9 | 8.9% |

| 难度等级 | 数量 | 占比 |
|---------|------|------|
| Easy | 28 | 27.7% |
| Medium | 52 | 51.5% |
| Hard | 21 | 20.8% |

来源包括：NCCL、MSCCL++、ThunderKittens、vLLM、SGLang、NVSHMEM、DeepEP 等前沿通信库。

### ⚙️ 实验设置
#### 平台配置
在三种真实硬件平台上进行评估：
1. **NVIDIA B300 节点**（8 GPUs，NVLink 连接）
2. **GH200 节点集群**（通过 400Gb/s InfiniBand + RDMA 互联）
3. **AMD MI325X 节点集群**（XGMI + RoCE）

支持 CUDA 与 ROCm/HIP 双栈。

#### 测试模型
评估了多个前沿闭源与开源 LLM：
- GPT-5.5（OpenAI）
- Gemini-3.1-Pro-Preview（Google）
- Claude Opus 4.7（Anthropic）
- GLM-5.1（Zhipu AI）
- Kimi-K2.6（Moonshot AI）
- DeepSeek-V4-Pro（DeepSeek-AI）
- Qwen3.7-Max（Alibaba Cloud）

默认使用各模型原生推理参数，未做 prompt engineering。

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Pass Rate (%)** | 成功编译、运行并通过功能正确性检查的任务比例 |
| **PASS+Good (%)** | 正确且性能达到参考实现 **95% 以上** 的任务比例 |
| **GM-Speedup** | 所有通过任务的“生成代码 vs 参考实现”性能比值的几何平均（越高越好） |
| **Quality-Weighted Pass Rate** | 主要排名指标 = `Pass Rate × GM-Speedup`，综合衡量覆盖率与质量 |
| **Cost ($)** | 每个任务调用 API 的平均费用（针对闭源模型） |

此外还定义了性能等级分类：
- Better: ≥20% 更快
- Comparable: [-5%, +20%]
- Degraded: (-40%, -5%)
- Severely Degraded: < -40%

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 2）

| Model | Quality-Weighted Pass Rate | Pass Rate (%) | PASS+Good (%) | GM-Speedup |
|-------|----------------------------|---------------|----------------|------------|
| **GPT-5.5** | **0.467** | **57.4** | **30.7** | 0.813 |
| Gemini-3.1-Pro-Preview | 0.305 | 36.6 | 25.7 | 0.832 |
| Claude Opus 4.7 | 0.282 | 33.7 | 20.8 | 0.836 |
| GLM-5.1 | 0.281 | 29.7 | 17.8 | 0.947 |
| Kimi-K2.6 | 0.275 | 30.7 | 18.8 | 0.895 |
| Qwen3.7-Max | 0.269 | 26.7 | 15.8 | 1.008 |
| DeepSeek-V4-Pro | 0.197 | 19.8 | 12.9 | 0.995 |

> 💡 **核心发现**：即使是当前最强模型 **GPT-5.5**，也仅能在 **30.7%** 的任务上生成**既正确又高效**的通信代码。

### 🔁 与基线方法的对比结果
- GPT-5.5 在 **Quality-Weighted Pass Rate** 上领先第二名约 **53%**，显示其在复杂系统编程任务中的显著优势。
- 尽管部分模型（如 Qwen3.7-Max）在 GM-Speedup 上更高（1.008），但因其 Pass Rate 较低（26.7%），整体得分仍落后。
- 开源模型普遍表现较弱，尤其在 specialized library（如 MSCCL++、ThunderKittens）上几乎全军覆没。

### 🔍 消融实验与深入分析

#### （1）迭代自修正效果（Iterative Refinement）
以 **DeepSeek-V4-Pro** 为例，在允许 5 轮自我修复后：
- Pass Rate 从 **19.8% → 41.6%**（翻倍）
- 主要提升集中在 Easy/Medium 难度和通用库（NCCL、CUDA Runtime）
- 但在 Hard 任务和 niche 库（MSCCL++、ThunderKittens）上依然失败，说明**缺乏根本性领域知识**

#### （2）成本匹配下的反超现象
当控制总预算相同时（$1.91/任务），便宜模型可进行更多轮次优化：
| Model | Max Rounds | Quality-Weighted Pass Rate |
|-------|------------|----------------------------|
| GPT-5.5 | 1 | 0.467 |
| Claude Opus 4.7 | 5+ | **0.690** |
| Gemini-3.1-Pro-Preview | 5+ | **0.689** |

> ✅ 结论：**廉价模型通过多轮迭代可在相同预算下超越昂贵模型**，说明“试错-反馈”机制对系统编程至关重要。

#### （3）失败模式案例研究
| 案例 | 任务描述 | 失败原因 |
|------|----------|----------|
| Case 1 (Easy) | 使用 ThunderKittens 实现 BF16 AllToAll | 多数模型因缺乏 API 知识导致类型错误、函数签名错误 |
| Case 2 (Medium) | 使用 Hopper mbarrier 实现 warp 同步 | 所有模型均出错，源于对 sm_90 新增 PTX 指令训练数据不足 |
| Case 3 (Hard) | 使用 MSCCL++ MemoryChannel 实现 AllToAll | 所有模型编译失败，因幻觉 header 路径与类名 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 当前无法可靠生成高性能 GPU 通信代码**：
   - 最强模型 GPT-5.5 仅在 **30.7%** 的任务上达到“正确且高效”
   - 多数失败源于**API 知识缺失**、**头文件幻觉**、**接口语义误解**

2. **专用通信库是主要瓶颈**：
   - 对 NCCL、CUDA Runtime 等通用库支持较好
   - 但对 MSCCL++、ThunderKittens、NCCL Device API 等新兴/研究型库支持极差

3. **性能 ≠ 正确性**：
   - 有些模型能写出高性能代码，但无法通过正确性测试
   - 有些通过功能测试，但性能严重退化（Severely Degraded）

4. **迭代反馈显著提升成功率**：
   - 特别是对低成本模型，多轮 refine 可大幅提升 Pass Rate
   - 但无法弥补对 niche library 的知识鸿沟

### ⚠️ 方法的局限性
- **评估成本高**：需真实多 GPU 集群，难以大规模普及
- **任务规模有限**：目前仅 101 个任务，虽具代表性但仍偏小
- **静态提示输入**：未结合检索增强（RAG）或外部文档查询
- **未探索微调影响**：尚不清楚 post-training 是否能显著改善表现

### 🔮 未来工作方向
1. **针对性领域微调**：
   - 在 CommBench 数据上进行 SFT 或 RLHF，提升对通信 API 的掌握
2. **引入 Retrieval-Augmented Generation (RAG)**：
   - 推理时接入 NCCL/MSCCL++ 文档、头文件、示例代码
3. **构建 agentic coding pipeline**：
   - 结合 compiler error feedback、profiler 输出进行自动调试与优化
4. **扩展至更多硬件平台**：
   - 加入 Intel GPU、Apple Silicon 等异构生态
5. **推动开放通信库生态**：
   - 鼓励将 MSCCL++、ThunderKittens 等纳入公共训练语料

---

> 🧩 **一句话总结**：  
> **CommBench 揭示了当前 LLM 在系统级 GPU 通信编程上的巨大差距 —— 它们尚未准备好替代人类专家编写高性能多 GPU 通信代码，但通过迭代反馈与领域适配，有望逐步迈向实用化。**

</details>

---

### 2. [RAC: Reference-Aware Activation Compression for Communication-Efficient Split LLM Inference](https://arxiv.org/abs/2608.04991)

**Authors**: Guotao Yang, Mingxi Zhao, Haopeng Li, Zhengchao Wang, Sheng Chen, Yitao Hu, Keqiu Li  
**Category**: cs.DC  
**Published**: 2026-08-06  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2608.04991v1  

#### Abstract
Large language model (LLM) agents repeatedly process long, privacy-sensitive contexts, while cloud-only deployment exposes user data beyond the trusted endpoint and fully local deployment often requires costly hardware. Split inference offers a middle ground by executing the model head, tail, and to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RAC: Reference-Aware Activation Compression for Communication-Efficient Split LLM Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **Split LLM Inference**（边缘-云协同推理）中，模型的头尾部分在本地设备执行，中间层在云端执行。这种架构虽然兼顾了计算资源与隐私保护，但在推理过程中需要频繁传输边界处的 **hidden states**，导致严重的通信瓶颈：

- **Prefill 阶段**：需传输完整的 `[B, N, d]` 隐藏状态，影响 **Time to First Token (TTFT)**。
- **Decode 阶段**：每生成一个 token 就需一次上行和下行通信，影响 **Time per Output Token (TPOT)**。

直接对激活值进行量化（如 INT4/INT8）、稀疏化（sparsification）或混合精度压缩会显著降低任务准确率，尤其是面对长尾分布和异常值时。

---

### 提出的新方法：RAC（Reference-Aware Compression）

RAC 是一种面向 Split LLM 推理的 **reference-aware 激活压缩框架**，其核心思想是：

> **不直接压缩当前 hidden state $H$，而是先减去一个相似的 reference state $R$，再对残差 $E = H - R$ 进行量化**。

由于残差分布更集中、动态范围更小，因此可以用更低比特（如 4-bit）高效表示，同时保持高重建精度。

#### 创新点：
1. **Phase-aware Reference 构造策略**：
   - **Prefill 上行链路**：通过 token rolling hash 查找历史请求中的相同 token span，复用其对应的 hidden state 作为 reference。
   - **Prefill 下行链路**：直接复用已重建的上行 hidden state（来自同一轮），无需额外查找。
   - **Decode 阶段**：使用轻量级 causal Transformer 预测器（per-boundary）从前一步的 hidden state 预测当前 reference。

2. **Aligned Residual Quantization**：
   - 引入 **grouped affine alignment**（分组仿射映射）来校准 reference 与当前 state 的尺度和偏移。
   - 对齐后对残差进行分组量化（支持 INT4/INT8），并可选地处理 prefill 中的 outlier（以 FP16 发送）。
   - Decode 阶段禁用 outlier 传输以减少元数据开销。

3. **Offline Calibration 机制**：
   - 在离线阶段模拟整个编码-解码流程，综合考虑量化位宽、分组大小、alignment 开销、outlier 元数据等成本，选择最优配置，在保证质量的前提下最小化通信负载。

4. **Sender-side Reconstruction**：
   - 发送端也重建量化后的状态，确保后续 reference 同步一致，避免误差累积。

---

### 相比现有方法的优势
| 方法 | 缺陷 | RAC 如何改进 |
|------|------|---------------|
| **Global INT4/INT8** | 覆盖全动态范围，极端值拉大量化步长，损失精度 | 只量化残差，分布更紧凑，精度更高 |
| **Top-K / Sparsification** | 丢弃非 top-k 值，丢失信息；需传输 mask/index | 不丢弃任何值，仅对齐后量化，保留完整结构 |
| **Mixed Precision** | 需要高精度 outlier payload 和 metadata | Decode 禁用 outlier，prefill 可选，控制开销 |
| **Learned Bottleneck** | 改变模型结构，训练复杂 | 不修改模型或参数，纯推理时压缩 |

✅ **优势总结**：  
- 显著降低通信量（~72–75% payload reduction）
- 几乎无损任务质量（多数指标变化 < ±1 pt）
- 端到端延迟大幅下降（TTFT 提升 1.24–2.72×）
- 完全兼容现有 Split Inference 架构

---

## 2. 核心实验方法和设置

### 使用的数据集

| 数据集 | 用途 | 说明 |
|--------|------|------|
| **WikiText-2** | 语言建模质量评估 | 测量 Perplexity (PPL) |
| **HellaSwag** | 常识推理能力 | 准确率（Accuracy） |
| **GSM8K & MATH-500** | 数学推理能力 | Flexible/Strict Exact Match (EM) |
| **ShareGPT** | 性能压测工作负载 | 多轮对话，输入输出长度异构 |

---

### 实验设置

- **硬件平台**：
  - 本地端：双 NVIDIA RTX 3090
  - 云端：单卡或四卡 NVIDIA A800
- **网络模拟**：
  - 使用 `Linux Traffic Control (tc)` 模拟广域网带宽限制
  - 测试多种 prefill/decode 带宽组合（如 100/1 Mbit/s）
- **并发请求**：默认 8 并发
- **测试模型**（覆盖不同规模与架构）：
  1. **GLM-4-9B-0414**（9B，dense）
  2. **Qwen3-30B-A3B**（30B，MoE）
  3. **Llama-3.3-70B-Instruct**（70B，dense）

---

### 评估指标

| 类别 | 指标 |
|------|------|
| **性能（Latency）** | 
| | - TTFT（Time to First Token）：均值、中位数、99th 百分位 |
| | - TPOT（Time per Output Token） |
| | - 输出吞吐量（Throughput） |
| **通信效率** |
| | - 激活值 payload 占比（vs Raw） |
| | - 元数据开销分析 |
| **任务质量** |
| | - PPL（越低越好）
| | - Accuracy / EM / MathVerify（越高越好） |
| **消融实验** |
| | - 移除 prediction / search / affine alignment 的影响 |

---

### 基线方法对比

| 基线方法 | 描述 |
|---------|------|
| **Raw** | 无压缩，原始传输 |
| **TopK (4-bit ratio)** | 保留 top-k 激活，其余置零，剩余部分用 4-bit 量化 |
| **TopK (8-bit ratio)** | 同上，但用 8-bit 量化 |
| **Global INT8** | 整体统一 8-bit 量化 |
| **Global INT4** | 整体统一 4-bit 量化 |

> ⚠️ 注意：这些基线本身就会造成显著的质量下降，因此其延迟优势是在牺牲准确性前提下的“不公平”比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | RAC 表现 |
|------|----------|
| **Prefill Payload Reduction** | ~72.2% ↓（平均传输 27.8% 的 raw 数据） |
| **Decode Payload Reduction** | ~75.0% ↓（平均传输 25.0% 的 raw 数据） |
| **Mean TTFT 加速比 (Raw/RAC)** | **1.24 – 2.72×** |
| **Mean TPOT 加速比 (Raw/RAC)** | **1.01 – 2.79×** |
| **99th Percentile TTFT** | 在所有 9 个 model-link 组合中均优于 Raw |
| **任务质量变化（Δ vs Raw）** | 非 PPL 指标变化范围：**-0.40 至 +2.50 pts** |

> ✅ 特别亮点：在 Llama-3.3 上，GSM8K 的 Strict EM 提升了 **+2.50 pts**！

---

### 与基线方法对比结果

| 方面 | RAC vs Baselines |
|------|------------------|
| **任务质量** | 
| | - 所有非 PPL 指标中，RAC 在 **14 out of 15** 场景下偏离 Raw 最小
| | - Global INT4 导致 PPL 暴增至数千甚至上万，几乎不可用
| | - TopK 方法普遍导致 EM 接近 0（如 GLM 上 TopK(4-bit) 仅得 1.82 分） |
| **延迟表现** |
| | - 在低带宽场景（尤其 decode 带宽 < 10 Mbit/s）下，RAC 明显胜出
| | - TopK 虽然延迟较低，但处于完全不同（极低质量）的操作点 |
| **通信敏感性** |
| | - 图6显示：在 25 组带宽组合中，RAC 在 **21 组中 TTFT 更优，18 组中 TPOT 更优，19 组中吞吐更高**
| | - 优势集中在 decode 带宽受限时，体现其对小消息序列化开销的优化效果 |

---

### 消融实验结果（Ablation Study）

| 变体 | 影响（典型下降） |
|------|------------------|
| **w/o Prediction** | 
| | - GLM: GSM8K EM ↓10.38 pts, MATH-500 ↓22.0 pts
| | - Llama: Strict EM ↓5.0 pts → 显示 decode reference 至关重要 |
| **w/o Search** |
| | - GLM: MATH-500 ↓9.2 pts → 显示历史 span 复用有效 |
| **w/o Affine Alignment** |
| | - 所有模型均有下降，最大达 ~2–3 pts → 显示 scale/offset 校准必要 |

> 🔍 结论：三大组件（prediction, search, alignment）共同支撑高质量压缩，缺一不可。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Hidden State 存在强 temporal & historical redundancy**：
   - 相同 prompt span 的 hidden state 高度相似（cosine > 0.9）
   - 同一轮次上下行 state 之间存在跨层相关性
   - 相邻 decode 步骤间状态高度连续

2. ✅ **Residual Coding + Lightweight Prediction 是高效压缩的关键路径**：
   - 差分编码使量化更精细
   - 轻量预测器即可显著提升 reference 相似度（见 Fig.4）

3. ✅ **Compression 可做到近乎无损且端到端加速**：
   - 通信量降至约 1/4
   - 延迟显著下降（尤其在低带宽环境）
   - 多数任务指标变化不超过 ±1 pt，部分甚至略有提升

4. ✅ **Offline Calibration 可实现质量-成本联合优化**：
   - 自动选择最佳 ga, gq, bit-width, outlier policy
   - 平衡量化精度与元数据开销

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖 exact token matching** | 若历史中无完全匹配的 span，则无法复用 → 对高度多样化输入效果可能减弱 |
| **Decode predictor 需维护状态** | 增加轻微内存开销（key-value cache） |
| **未解决 formal privacy 问题** | 压缩本身不提供加密或混淆，仍需结合 perturbation/pruning 等防御手段 |
| **当前仅适用于 three-way split** | 不支持多跳或多节点分割推理 |

---

### 未来工作方向

1. **Dynamic Reference Selection**：
   - 引入 approximate matching（如 semantic hashing）扩大 reference 覆盖面
2. **Cross-user Reference Pooling**（在隐私允许下）：
   - 构建共享 prompt cache 提升命中率
3. **Integration with KV Cache Compression**：
   - 联合压缩 boundary activation 与 attention cache（如 CacheGen）
4. **Hardware-aware Codec Design**：
   - 针对特定 NIC 或压缩硬件进一步优化 wire format
5. **Extension to Multi-hop Split Inference**：
   - 支持更多切分层级和分布式部署

---

## 总结

📌 **RAC 是首个将 reference-aware 差分编码系统应用于 Split LLM Inference 的工作**，它巧妙利用了 LLM 推理过程中的结构性冗余，提出了一套低开销、高质量、可部署的激活压缩方案。

🎯 **核心价值**：  
> 在几乎不影响模型性能的前提下，将通信负载降低 **70%+**，端到端延迟最高加速 **2.7×**，为大规模 LLM 在边缘设备上的实用化部署提供了关键技术支撑。

🔧 **适用场景**：  
- 私有化 LLM Agent（需保留 prompt 隐私）
- 低带宽环境下远程推理服务
- 成本敏感型边缘 AI 应用

🚀 **一句话评价**：  
**RAC 把“能省的都省了，该留的全留下了”——是当前 Split Inference 通信优化中最务实、最有效的解决方案之一。**

</details>

---

### 3. [AsymSpec: Efficient Cloud-Edge Speculative Decoding over Asymmetric Networks](https://arxiv.org/abs/2608.04974)

**Authors**: Guotao Yang, Hao Chen, Rui Guo, Xinyu Li, Liang Zheng, Sheng Chen, Yitao Hu, Keqiu Li  
**Category**: cs.DC  
**Published**: 2026-08-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2608.04974v1  

#### Abstract
Cloud-edge speculative decoding places a lightweight draft model at an edge gateway and a higher-quality target model in the cloud, but inserts communication into every speculative block. Under a constrained uplink, candidate messages may queue while the verifier is idle. Stop-and-wait scheduling le...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AsymSpec: Efficient Cloud-Edge Speculative Decoding over Asymmetric Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在 **cloud-edge speculative decoding** 架构中，轻量级的 draft model 部署在边缘设备，高质量的 target model 部署在云端。然而，这种架构面临两个关键瓶颈：

- **上行链路受限导致验证延迟**：每个 speculative block 必须通过带宽受限的上行链路上传至云端进行验证，而常规接受路径仅需少量信息（如候选 token 和其概率），但现有方法却上传完整的 draft 分布，造成通信浪费。
- **依赖性前瞻（dependent runahead）产生无效计算**：为掩盖验证延迟，系统可能基于未确认的前缀进行下一轮 draft（即 optimistic runahead），一旦前一个 block 被拒绝或出现“意外 bonus token”，后续所有依赖该前缀的工作都将失效。

### **提出的新方法：AsymSpec**
AsymSpec 提出了一套面向非对称网络环境的高效云边协同 speculative decoding 框架，包含两大核心机制：

#### ✅ **1. 非对称验证协议（Asymmetric Verification Protocol）**
- **上传精简信息**：仅上传 acceptance 所需的 `(token, q(x))` 对和元数据，大幅减少上行负载。
- **按需返回修正信息**：
  - 若全部接受 → 返回长度 + bonus token（极小响应）；
  - 若发生拒绝 → 启动 **progressive correction**：
    1. 先返回小规模 target top-K；
    2. 边缘端利用保留的 draft 分布计算 **Total Variation (TV) certificate**，判断当前支持集是否足够精确；
    3. 若不够，则逐步扩展支持集；
    4. 最终仍不足时，回退到 proposal-based 或完整分布恢复。

> 🔑 创新点：以 **residual distribution 的 TV error** 作为保真度证书，而非简单依赖 top-K 覆盖率，确保输出分布不变性。

#### ✅ **2. 已确认前缀流水线（Confirmed-Prefix Pipeline）**
- 每个请求最多允许一个 “open block”（从起草到提交/中止之间）；
- 在等待验证期间，边缘调度器不 speculative 下游块，而是切换到其他具有已确认前缀的独立请求；
- 云端异步收集来自不同请求的 blocks，并动态重组 batch 进行并行验证。

> 🔑 创新点：通过跨请求并行而非单请求内前瞻，避免无效 work，同时提升资源利用率。

### **相比现有方法的优势**
| 维度 | 现有方法（如 PipeInfer, CoSine） | AsymSpec |
|------|-------------------------------|---------|
| 上行通信开销 | 上传完整 draft 分布 → 高 | 仅上传 token-scalar 对 → 极低 |
| 下行纠错灵活性 | 固定纠错方式 | 渐进式、按需纠错，节省下行流量 |
| 资源利用率 | 依赖 same-request runahead → 易产生无效 work | 使用 confirmed-prefix 多请求重叠 → 安全高效 |
| 输出保真度 | 可能因压缩失真改变分布 | TV certificate 保证误差可控，满足 request-level fidelity budget |

---

## **2. 核心实验方法和设置**

### **数据集与工作负载**
- **GSM8K**：数学推理任务，测试模型逻辑能力；
- **HumanEval**：代码生成任务，评估编程准确性。

### **模型组合（draft/target pairs）**
共三组，覆盖主流 LLM 家族：
- **Qwen3**: 4B / 32B
- **GLM-4**: 9B / 32B
- **Llama 3.1**: 8B / 70B

### **网络条件（模拟非对称带宽）**
使用 Linux Traffic Control 和 NetEm 模拟三种典型云边接入场景：
| 网络配置 | 上行/下行带宽（Mbps） | 场景描述 |
|--------|---------------------|--------|
| **Weak** | 50 / 150 | 拥塞蜂窝网或 LTE，上行严重受限 |
| **Medium** | 150 / 1000 | 子6GHz 5G，下行较快但上行有限 |
| **Strong** | 250 / 2500 | 高速 mmWave 5G，接近理想条件 |

### **基线方法对比**
- **Standard Spec**：标准 speculative decoding（本地执行）
- **CoSine**：去耦合云边 speculative 推理框架
- **PipeInfer**：异步流水线 speculative 推理

### **评估指标**
- **Output-token throughput (tokens/s)**：核心吞吐量指标
- **End-to-end latency**
- **Time to First Token (TTFT)**
- **Time Per Output Token (TPOT)**

---

## **3. 主要实验结果和性能指标**

### **整体性能表现**
AsymSpec 在所有 **18 个操作点**（3模型 × 2任务 × 3网络）上均取得最优性能：

> 📈 **输出 token 吞吐量达到最强 baseline 的 2.82–28.03×，几何平均提升 7.96×**

| 模型-任务组合 | 最大加速倍数 |
|--------------|-------------|
| Qwen3-GSM8K | 12.55× |
| GLM-4-HumanEval | 4.20× |
| 平均（geometric mean） | **7.96×** |

### **网络鲁棒性强**
随着网络质量下降（Strong → Weak），各方法吞吐变化如下：
| 方法 | 吞吐下降比例 |
|------|------------|
| Standard Spec | ↓78.6% |
| CoSine | ↓71.4% |
| PipeInfer | ↓71.5% |
| **AsymSpec** | ↑**1.9%** ✅ |

> 💡 表明 AsymSpec 不仅不受弱网络影响，反而因更高效的通信调度，在低带宽下相对优势更大。

### **消融实验结果（Ablation Study）**
见 Table II，关键发现：

#### **移除通信优化（No asymmetric correction / top-K）**
- 吞吐暴跌 **93.7% ~ 98.6%**
- 端到端延迟飙升至 **32× ~ 152×**
- TPOT 增加超百倍 → 通信成为绝对瓶颈

#### **移除 top-K 支持（No top-K）**
- 吞吐降低 25.5% ~ 76.1%
- 弱网络下延迟达 7.7× → 小支持集即可获得高保真

#### **移除解耦流水线（No pipeline）**
- 吞吐下降 11.5% ~ 16.8%
- 若替换为 same-request runahead（No pipeline + runahead）→ 吞吐仅剩 76% of Full，且延迟更高

> ✅ 结论：**通信机制主导弱网性能，调度机制在所有场景均有增益**

### **修正敏感性分析**
#### （1）**Correction Fidelity vs. K**
- 当 `K ≥ 256` 时，top-1/5/10 recall 均达 1.0，residual TV < 5e-4
- 即使 `K=16`，recall > 0.92，TV ≈ 0.0116 —— 已足够实用
- 支持集大小仅为 vocab 的 1/18 ~ 1/593 → 高压缩比 + 高保真

#### （2）**任务准确率**
- 在 GSM8K 和 HumanEval 上，bounded correction（K=16~8192）与 full correction 几乎无差异
- 平均差距仅 **-0.52 pp**，部分设置甚至反超

#### （3）**运行时开销**
- `K ≤ 8192`：延迟接近基准（1.0×）
- `K=40k` 或 full vocab：延迟显著上升，尤其在 Weak 网络下可达 **5.1×**

> ✅ 推荐策略：从小 K 开始，动态扩展，避免一开始就传输大规模分布

---

## **4. 关键结论和发现**

### **主要发现**
1. **非对称网络下的通信设计至关重要**：
   - 上行应最小化，只传 acceptance 必需信息；
   - 下行可用于承载条件性的丰富纠错数据；
   - 统一上传完整分布是严重资源错配。

2. **valid work 比 raw utilization 更重要**：
   - optimistic runahead 虽提高利用率，但大量 work 可能被废弃；
   - 使用 confirmed-prefix 多请求并行，既能填满空闲周期，又能保证 work 有效性。

3. **渐进式纠错 + TV certificate 是高效与保真的平衡点**：
   - 不再依赖固定阈值或启发式压缩；
   - 动态决策是否需要更多支持，实现 per-event 自适应。

4. **AsymSpec 对网络退化具有反脆弱性**：
   - 在弱网下性能不降反升，说明其机制特别适合现实部署环境。

### **局限性**
- 当前假设边缘能稳定运行 draft model；若边缘算力极弱，draft 成本本身将成为瓶颈。
- TV certificate 依赖边缘保存完整 draft 分布，内存开销需权衡。
- 实验基于模拟网络，真实移动网络中的抖动、丢包尚未充分验证。

### **未来工作方向**
- 支持多跳 edge-cloud 架构下的 speculative decoding；
- 将 TV certificate 思想推广至其他分布式 AI 推理场景；
- 结合 quantization 与 sparse communication 进一步压缩交互；
- 动态调整 per-request fidelity budget 以满足 SLO 要求。

---

> ✅ **总结一句话**：  
> **AsymSpec 通过“轻上传 + 按需纠错 + 跨请求并行”的设计，在非对称网络下实现了高达 28× 的输出吞吐提升，是云边协同 LLM 推理的一项重要进展。**

</details>

---

### 4. [Attention-Only White-Box Transformer via LeJEPA-Based Self-Supervised Pretraining](https://arxiv.org/abs/2608.04213)

**Authors**: Yang Bai, Linyuan Wang, Haoyang Jiang, Nuolin Sun, Libin Hou, Bin Yan  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.04213v1  

#### Abstract
Existing studies on self-supervised learning for white-box networks typically decouple the derivation of white-box networks via optimization algorithms from self-supervised learning paradigms. In this work, we instead revisit the two components from a joint perspective. The LeJEPA-based self-supervi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Attention-Only White-Box Transformer via LeJEPA-Based Self-Supervised Pretraining*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有基于自监督学习的 **white-box Transformer** 研究通常将网络结构推导（如通过优化算法）与自监督学习范式割裂开来，导致理论不一致。例如：
- **CRATE-MAE** 虽采用 white-box 编码器，但依赖缺乏理论解释的重建损失；
- **EMP-SSL** 引入了编码率目标，但仍使用黑盒架构。

本文提出一个根本性问题：**如何在保持白盒架构与学习目标之间理论自洽的前提下，进行自监督训练？**

### 提出的新方法与新思路
作者提出一种全新的 **attention-only white-box Transformer** 架构 —— **AoT-ADMM**，其核心思想是：

- 将 white-box 目标中的三个项解耦处理：
  - **全局扩展项 $R(Z)$**：通过 **LeJEPA** 的自监督目标（特别是其中的 **SIGReg** 正则化）来优化；
  - **压缩项 $R_c(Z|U_{[K]})$ 和稀疏项 $\lambda \|Z\|_0$**：使用 **ADMM**（交替方向乘子法）进行优化并展开为网络层。

由此推导出仅由 **attention 模块** 构成的前向结构，无需原始设计中的 ISTA 结构或 MLP 层。

### 相比现有方法的优势
- ✅ **理论一致性更强**：训练目标（LeJEPA loss）与网络架构均源自统一的 white-box 优化目标（sparse rate reduction），实现端到端的数学可解释性。
- ✅ **参数更少**：相比原版 CRATE 减少约 **31% 参数量**，且性能几乎持平。
- ✅ **模块简化**：完全去除 MLP 和可学习字典（learnable ISTA dictionary），仅保留 attention 和 ReLU。
- ✅ **通用潜力**：在标准 ViT 上验证，用 ReLU 替代 MLP 并结合知识蒸馏，可减少 **66% 参数** 仍保持竞争力，表明 MLP 可能存在冗余。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CIFAR-10 / CIFAR-100**：用于小规模自监督预训练 + 线性探针（linear probing）评估。
- **ImageNet-1K**：用于大规模自监督预训练，之后在 CIFAR 上全量微调以评估迁移能力。

### 实验设置与评估指标
| 设置项 | 描述 |
|------|------|
| **预训练框架** | LeJEPA（Joint-Embedding Predictive Architecture），含多裁剪增强（multi-crop augmentation）、预测一致性损失 $L_{\text{pred}}$ 和 SIGReg 正则项 |
| **输入分辨率** | CIFAR: 全局视图 32×32，局部视图 16×16；ImageNet: 128×128 和 64×64 |
| **patch size** | CIFAR: 8；ImageNet: 16 |
| **优化器** | AdamW，学习率 5e-4，cosine 衰减，batch size 分别为 256（CIFAR）和 1024（ImageNet） |
| **训练轮数** | CIFAR: 800 epochs；ImageNet: 200 epochs |
| **评估方式** | 冻结主干网络，使用线性分类头进行 **Top-1 Accuracy** 测试 |

### 基线方法对比
- **CRATE**：当前主流 white-box Transformer，作为主要对比基线。
- **Wang et al.'s AoT**：另一种 attention-only Transformer，基于子空间去噪推导。
- **标准 ViT-T/S**：用于验证 MLP 替换的有效性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（CIFAR 线性探针）
| 模型 | 规模 | 数据集 | Acc. (%) | 参数量 | FLOPs |
|------|------|--------|----------|--------|-------|
| CRATE | Base | CIFAR-10 | 89.18 | 21.44M | 1.22G |
| **AoT-ADMM** | Base | CIFAR-10 | **88.88** | **14.36M** | **0.50G** |
| CRATE | Base | CIFAR-100 | 63.56 | 21.44M | 1.22G |
| **AoT-ADMM** | Base | CIFAR-100 | **63.54** | **14.36M** | **0.50G** |

> ⬇️ 参数减少 **31.3%**，精度差距仅 **0.30 / 0.02 个百分点**。

### 与 AoT 的对比（相同参数量下）
| 模型 | 规模 | CIFAR-10 Acc. | CIFAR-100 Acc. | 参数量 |
|------|------|---------------|----------------|--------|
| AoT (Wang et al.) | Base | 84.20 | 57.48 | 14.35M |
| **AoT-ADMM** | Base | **88.88** (+4.68) | **63.54** (+6.06) | 14.36M |

> ✅ 在相近参数量下，**显著优于已有 attention-only 架构**，说明优化原理对有效性至关重要。

### 大规模预训练 + 迁移学习（ImageNet-1K → CIFAR）
| 模型 | 预训练 | 微调任务 | Acc. (%) | 参数量 |
|------|--------|-----------|----------|--------|
| CRATE | IN-1K | CIFAR-10 | 96.01 | 21.92M |
| **AoT-ADMM** | IN-1K | CIFAR-10 | **95.25** | **14.84M** |
| CRATE | IN-1K | CIFAR-100 | 81.04 | 21.92M |
| **AoT-ADMM** | IN-1K | CIFAR-100 | **80.26** | **14.84M** |

> ⬇️ 参数减少 **32%**，性能下降不到 1%，显示良好迁移能力和参数效率。

### 标准 ViT 上的 MLP 替换实验（知识蒸馏）
| 模型 | 结构 | 参数量 | CIFAR-10 Acc. |
|------|------|--------|----------------|
| ViT-Tiny | 原始 MLP | 5.3M | 92.19 |
| ViT-T-AoT | MLP → ReLU | **1.8M** (-66%) | 91.37 (-0.82) |
| ViT-Small | 原始 MLP | 21.4M | 93.58 |
| ViT-S-AoT | MLP → ReLU | **7.2M** (-66%) | **93.67** (+0.09) |

> ✅ 表明在标准 ViT 中，**MLP 模块可能高度冗余**，可用简单非线性替代，在特定设置下甚至略有提升。

---

## 4. 关键结论和发现

### 主要发现
1. **LeJEPA 的 SIGReg 与 white-box 扩展项 $R(Z)$ 具有理论一致性**：
   - 两者在协方差层面共享相同的各向同性高斯最优解（isotropic Gaussian optimum）；
   - 因此可将 $R(Z)$ 移至损失函数中由 SIGReg 实现，从而解耦架构设计。

2. **ADMM 可有效展开为纯 attention 架构**：
   - 压缩项对应 **MSSA (Multi-Head Subspace Self-Attention)**；
   - 稀疏项可通过 **ReLU** 实现闭式解（proximal operator）；
   - 最终得到无需 ISTA 或 MLP 的 **attention-only Transformer**。

3. **MLP 模块可能存在显著冗余**：
   - 不仅在 white-box 架构中可被去除；
   - 即使在标准 ViT 中，用 ReLU 替代 MLP 并配合知识蒸馏，也能大幅减参而不损性能。

### 方法的局限性
- 当前模型仍依赖 **LeJEPA 框架**，其本身假设较强的预测一致性；
- **AoT-ADMM 的动态系数需训练调整**，初始值虽来自 ADMM 推导，但后续脱离原始优化路径；
- 对更高阶统计量的建模（如 SIGReg 中的 skewness/kurtosis）尚未完全融入 white-box 解释体系。

### 未来工作方向
- 探索将 **higher-order regularization** 更深入整合进 white-box 目标；
- 设计完全免去投影头（projection head）的端到端可解释 pipeline；
- 将该范式推广至 NLP 和多模态任务，构建统一的 attention-only 白盒架构家族；
- 进一步研究 **ReLU 是否足以替代 MLP 的表征能力**，尤其是在长序列建模中。

---

> 🔚 **总结一句话**：  
> 本文首次实现了 **训练目标与网络架构均源于同一 white-box 优化问题** 的自监督 Transformer，提出 **AoT-ADMM** —— 一种无需 MLP 和 ISTA 的 attention-only 架构，在减少超 30% 参数的同时保持与 CRATE 相当的性能，并揭示了 Transformer 中 MLP 模块的潜在冗余性。

</details>

---

### 5. [NeuroPB: Scaling Neural Decoding with Pretrained Behavioral Representations](https://arxiv.org/abs/2608.04389)

**Authors**: Luyao Jin, Yonghao Song, Huan Zhao, Vincent C. K. Cheung, Wei-Hsin Liao  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.04389v1  

#### Abstract
Decoding continuous motor trajectories from neural activity is essential for developing practical brain-computer interfaces (BCIs). However, current neural decoders are constrained by the limited scale and heterogeneity of neural recordings. In contrast, behavioral data can be collected more readily...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：NeuroPB: Scaling Neural Decoding with Pretrained Behavioral Representations**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的神经解码（neural decoding）方法在从神经活动中重建连续运动轨迹时面临两大挑战：
- **神经数据稀缺且异质性强**：跨记录会话（sessions）、个体（subjects）和任务（tasks）的神经活动差异大，而高质量配对的神经-行为数据收集成本高、规模有限。
- **泛化能力差**：现有模型在训练分布内表现良好，但在未见过的会话、个体或任务上性能显著下降。

### **提出的新方法/新思路**
作者提出了 **NeuroPB**（Neural decoding with Pretrained Behavioral representations），一种通过**预训练行为表征**来提升神经解码可扩展性和泛化性的新框架。其核心思想是：
- 从大规模行为数据（包括灵长类动物和机器人系统的行为轨迹）中预训练一个**行为编码器**（motor encoder），学习通用的运动动力学结构。
- 利用少量配对的神经-行为数据，将神经活动对齐到该冻结的**行为表征空间**中。
- 在推理阶段，仅需神经信号即可生成准确的运动轨迹预测。

这一方法实现了“**从行为中学，为神经所用**”的知识迁移范式。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **可扩展性** | 不依赖大规模神经数据预训练，转而利用更易获取的大规模行为数据（如机器人轨迹）。 |
| **泛化能力** | 显著提升跨会话、跨个体、跨任务的解码性能。 |
| **校准效率** | 仅需约10%的目标域数据即可达到从头训练（scratch）使用100%数据的性能，大幅降低校准成本。 |
| **跨模态迁移** | 首次证明**机器人轨迹**可作为有效的行为先验用于生物运动解码，揭示了生物与人工系统间共享的可迁移运动结构。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **Macaque Neural Dataset**  
   - 来源：4只恒河猴（rhesus macaques）执行二维伸手任务的电生理记录。
   - 脑区：Primary Motor Cortex (M1) 和 Dorsal Premotor Cortex (PMd)。
   - 行为范式：
     - **Center-Out (CO)**：从中心向8个放射状目标移动。
     - **Random-Target (RT)**：连续追踪随机位置的目标。
   - 数据特点：包含多个会话、个体和任务结构，适合评估泛化能力。

2. **Robotic Behavior Datasets**（用于行为预训练）
   - **LIBERO-Spatial**：500条机器人操作轨迹，与macaque数据量匹配，用于控制变量比较。
   - **LIBERO-100**：5,000条语言条件下的复杂操作轨迹，规模更大、多样性更高，用于评估数据缩放效应。

### **实验设置与评估指标**
- **输入**：1秒长度的spike events序列。
- **输出**：连续的二维运动轨迹（position + velocity）。
- **评估指标**：
  - **R² 决定系数**：衡量预测轨迹与真实轨迹的一致性。
  - **跨场景泛化测试**：
    - Cross-session
    - Cross-subject
    - Cross-task (CO ↔ RT)
  - **校准效率分析**：不同比例（1%~100%）目标域训练数据下的性能变化。

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| 传统模型 | Wiener Filter, Smoothing, DenseNN, RNN |
| 深度生成模型 | LFADS |
| 大规模神经预训练模型 | NEDS, NDT2, POYO |
| 本工作 | **NeuroPB**（含多种初始化策略） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Single-session Setting）**

| Method | CO (R²) | RT (R²) |
|--------|---------|---------|
| Wiener Filter | 0.5010 ± 0.0678 | 0.4240 ± 0.0838 |
| RNN | 0.7028 ± 0.0552 | 0.6439 ± 0.0576 |
| POYO | 0.9427 ± 0.0019 | 0.7156 ± 0.0966 |
| **NeuroPB (Robot-Large-PT)** | **0.9478 ± 0.0120** | **0.8475 ± 0.0171** |

> ✅ NeuroPB 在两个任务上均达到 **SOTA 性能**，尤其在更具挑战性的 RT 任务上优势明显（+13.2% R² vs POYO）。

---

### **与基线方法的对比结果**
- 相比从头训练（Scratch）：
  - CO 上 R² 提升 **11%**（94.78% vs 84.19%）
  - RT 上 R² 提升 **8%**（84.75% vs 76.29%）
- 所有行为预训练变体均显著优于 scratch，表明**行为先验的有效性**。
- **Robot-Large-PT > Macaque-PT ≈ Robot-Matched-PT**，说明**数据规模和多样性比行为来源更重要**。

---

### **消融实验结果（Ablation Study on CO Dataset）**

| 消融条件 | R² (CO) | 性能下降 |
|----------|--------|--------|
| 完整模型（Overall） | 0.9478 ± 0.0120 | — |
| 移除对比学习（w/o contrastive） | 0.8296 ± 0.0424 | ↓11.8% |
| 移除单元编码（w/o unit encoding） | 0.8825 ± 0.0155 | ↓6.5% |
| 移除 RoPE 时间编码 | 0.9011 ± 0.0090 | ↓4.7% |
| 移除交叉注意力 | 0.9182 ± 0.0107 | ↓3.0% |
| 移除自注意力 | 0.9065 ± 0.0133 | ↓4.1% |

> 🔍 结论：
> - **对比学习对齐机制贡献最大**，是性能提升的关键。
> - 单元身份信息（unit encoding）和相对时间建模（RoPE）也至关重要。
> - 各模块协同作用，共同支持高效神经轨迹解码。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **行为预训练显著提升神经解码性能**：即使不直接使用神经数据进行预训练，也能通过行为先验引导神经表征学习。
2. ✅ **机器人轨迹可有效迁移至生物运动解码**：Robot-Large-PT 性能媲美甚至超越基于真实灵长类轨迹的预训练，证明**运动学结构具有跨实体可迁移性**。
3. ✅ **行为数据规模越大，解码性能越强**：LIBERO-100（5k轨迹） > LIBERO-Spatial（500轨迹），验证了“scaling law”在行为预训练中的存在。
4. ✅ **大幅提升泛化能力和校准效率**：
   - 在跨会话、跨个体、跨任务设定下均显著优于 baseline。
   - 仅需 **10% 校准数据**即可匹敌 scratch 使用 100% 数据的性能。

---

### **方法的局限性**
- 当前研究基于**离线分析**，尚未在实时闭环 BCI 系统中验证。
- 使用的是**侵入式神经记录**（spikes），是否适用于非侵入信号（如 EEG/fNIRS）尚待验证。
- 行为预训练依赖高质量轨迹标注，在某些复杂任务中可能难以获得。

---

### **未来工作方向**
1. 将 NeuroPB 扩展至 **人类被试** 和 **非侵入式神经信号**（如 ECoG, EEG）。
2. 探索在 **实时闭环 BCI 控制** 中的应用，验证其实用性。
3. 引入更多模态的行为先验（如视觉、触觉反馈）以进一步丰富运动上下文理解。
4. 构建统一的“**行为基础模型**”（Behavioral Foundation Model），服务于多物种、多平台的神经接口开发。

---

> 📌 **一句话总结**：  
> NeuroPB 开辟了一条**以行为为中心**的神经解码新路径——通过大规模行为数据预训练构建通用运动先验，并将其锚定于有限神经数据中，从而实现高性能、高泛化、低校准成本的 BCI 解码，为下一代脑机接口提供了可扩展的技术范式。

</details>

---

### 6. [Continual-Learning Physics-Informed Neural Networks for Parameterized Partial Differential Equations](https://arxiv.org/abs/2608.04778)

**Authors**: Xujia Chen, Xinyue Hu, Letian Chen, Yi Liu, Wenhui Fan  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.04778v1  

#### Abstract
Physics-informed neural networks (PINNs) incorporate governing equations into neural-network training and can approximate PDE solutions without requiring large observational datasets. Parameterized PINNs (ParamPINNs) further take physical parameters as inputs, allowing a single model to represent a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Continual-Learning Physics-Informed Neural Networks for Parameterized Partial Differential Equations

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **ParamPINNs**（Parameterized Physics-Informed Neural Networks）在求解参数化偏微分方程（PDEs）时面临以下挑战：
- **训练效率低**：需要对大量参数组合进行采样，计算成本高。
- **跨参数精度不均衡**：模型在不同参数下的预测精度差异大，难以保证全局泛化能力。
- **过拟合风险**：当仅能采样有限数量的参数任务时，模型容易过拟合到这些特定任务，导致在未见参数上表现不佳。

### 提出的新方法
本文提出了一种名为 **CL-PINN**（Continual-Learning Physics-Informed Neural Network）的新框架，其核心思想是将不同的PDE参数实例视为一系列相关的学习任务，并采用**持续学习**（continual learning）的范式来顺序地学习这些任务。

#### 主要创新组件包括：
1. **基于贝叶斯优化的主动参数选择**（Bayesian-optimization-based active parameter selection）
   - 利用 **Bayesian Optimization (BO)** 迭代选择当前模型预测误差最大的新参数任务，从而高效地定位困难区域。
   - 显著减少了昂贵的损失函数查询次数（objective-loss queries），相比网格搜索更高效。

2. **任务级动态损失加权**（Task-wise dynamic loss weighting）
   - 在训练过程中，根据不同参数任务的当前损失大小和收敛速度，动态调整每个任务的损失权重。
   - 缓解了因多尺度行为导致的跨任务精度不平衡问题。

3. **稀疏物理约束回放**（Sparse physics-constrained replay）
   - 当活跃任务集达到容量上限后，新任务加入会导致旧任务被移除。
   - 通过保留少量物理坐标点（sparse collocation points）及其对应的PDE残差、边界条件等物理约束，在后续训练中继续施加对早期任务的约束，防止灾难性遗忘（catastrophic forgetting）。

4. **可选的参数子网络**（Optional parameter subnetwork）
   - 将物理参数输入单独编码为一个子网络，再与空间-时间坐标的主干网络融合。
   - 可以独立对该子网络应用正则化（如权重衰减）或冻结策略，提升知识保留能力和优化稳定性。

### 相比现有方法的优势
- **无需观测数据**：完全基于物理定律（PDEs、BCs、ICs）进行监督，适用于无真实数据场景。
- **资源受限下更优性能**：在有限的计算资源和活跃任务容量下，实现了更高的平均精度和更均衡的跨参数泛化能力。
- **高效的参数探索**：通过BO显著减少寻找困难参数所需的计算量。
- **缓解遗忘**：稀疏回放机制有效保留了已学任务的知识，避免因顺序学习导致的性能下降。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在五个基准问题上进行，涵盖一个连续函数和四个参数化的PDE：
1. **Schaffer-like 函数**：用于测试全局优化算法的非凸、多模态连续函数。
2. **Burgers 方程**：描述流体中非线性对流与粘性扩散竞争的一维非线性PDE。
3. **Allen-Cahn 方程**：材料科学中描述相变过程的反应-扩散方程。
4. **Kovasznay 流动**：具有解析解的二维稳态不可压缩Navier-Stokes方程。
5. **线性化 Poisson-Boltzmann 方程**：四维参数空间的稳态PDE，用于模拟带电粒子系统。

### 实验设置和评估指标
- **硬件环境**：NVIDIA Tesla V100 GPU，Intel Xeon Gold CPU，Ubuntu 20.04。
- **实现工具**：Python + PyTorch，结合 DeepXDE、scikit-learn、bayesian-optimization 库。
- **随机种子**：使用固定种子 `[0, 1, 2]` 进行多次运行并报告均值与标准差。
- **评估协议**：所有方法在相同架构、样本数、优化预算下比较，确保公平性。

#### 主要评估指标：
| 指标 | 定义 |
|------|------|
| **MSE** | 参数化预测与参考解之间的均方误差 |
| **Relative L2 Error (EL₂)** | 归一化的L2相对误差 |
| **Macro EL₂** | 所有测试参数任务上EL₂的算术平均值 |
| **Worst EL₂** | 所有测试参数任务中最高的EL₂值 |
| **Res-PDE** | PDE残差的平均绝对值 |

### 基线方法对比
- **UNI**（Uniform Sampling）：均匀参数采样。
- **FIX**（Manual Fixed Tasks）：手动预设参数集合。
- **AG**（Grid-Greedy Active Selection）：基于网格的贪婪主动选择方法。

本文提出的变体包括：
- **AC**：主动选择 + 动态加权（无回放）
- **ACR**：AC + 稀疏回放
- **ACR2-arch**：ACR + 参数子网络

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### 总体性能（以 Burgers 方程为例）
| Method | MSE (mean) | EL₂ (mean) |
|--------|------------|-----------|
| UNI    | 6.3431×10⁻⁴ | 1.2153×10⁻² |
| FIX    | 3.4909×10⁻⁴ | 2.1256×10⁻² |
| AG     | 4.1541×10⁻⁴ | 1.4551×10⁻² |
| AC     | 4.8920×10⁻⁴ | 2.1230×10⁻² |
| ACR    | 2.0548×10⁻⁴ | 1.0250×10⁻² |
| **ACR2-arch** | **2.3900×10⁻⁵** | **1.7710×10⁻³** |

> ✅ **ACR2-arch 在 Burgers 上将 EL₂ 错误降低了约 85% 以上**。

#### 综合表现趋势
- **ACR2-arch** 在大多数情况下取得了最佳的 **MSE** 和 **EL₂** 表现。
- **ACR**（无参数子网络）在 Allen-Cahn 上表现最好，说明参数子网络的效果依赖于具体方程。
- **AG** 虽然优于 UNI/FIX，但其网格搜索代价高昂，尤其在高维参数空间中不可扩展。

### 消融实验结果

#### （1）贝叶斯选择 vs 固定选择
| 方法组合 | Schaffer-like (EL₂) | Burgers (EL₂) | Poisson-Boltzmann (EL₂) |
|----------|---------------------|--------------|-------------------------|
| 固定选择 + 等权重 | 0.9987 | 0.02126 | 12.667 |
| BO + 等权重 | 2.5745 | **0.01961** | **0.09499** |
| 固定选择 + 加权 | 0.1671 | 0.02578 | 11.862 |
| **BO + 加权** | **0.16007** | 0.02123 | **0.06065** |

> 🔍 **BO 对 Burgers/Poisson-Boltzmann 更重要，而动态加权对 Schaffer-like 更关键**。

#### （2）稀疏回放 vs 无回放
| Case | No Replay (Macro EL₂) | Sparse Replay (Macro EL₂) | 改进幅度 |
|------|------------------------|----------------------------|--------|
| Schaffer-like | 0.16007 | 0.07195 | ↓55.0% |
| Burgers | 0.02123 | 0.01025 | ↓51.7% |
| Allen-Cahn | 0.04600 | 0.01467 | ↓68.1% |
| Kovasznay | 0.5555 | 0.07925 | ↓85.7% |
| Poisson-Boltzmann | 0.03449 | 0.03046 | ↓11.7% |

> ✅ **稀疏回放在所有案例中均显著降低遗忘，尤其在复杂PDE上效果明显**。

#### （3）参数子网络与正则化控制
| 控制项 | Schaffer-like | Allen-Cahn | Poisson-Boltzmann |
|-------|---------------|------------|--------------------|
| 无控制 | 0.05073 | 0.02439 | 0.04397 |
| +Adam decay | **0.04173** | 0.02717 | **0.02926** |
| +L-BFGS freezing | 0.05304 | 0.02371 | 0.05285 |
| +两者 | 0.04173 | **0.01828** | 0.03350 |

> ⚠️ **没有单一配置在所有问题上最优，需根据方程特性调参**。

---

## 4. 关键结论和发现

### 主要发现
1. **CL-PINN 是一种有效的参数化PDE求解框架**，能够在无观测数据、有限计算资源下实现高质量、泛化能力强的解。
2. **贝叶斯优化能大幅减少困难参数的搜索成本**，尤其在高维参数空间中优势显著（如 Poisson-Boltzmann 上减少 91.9% 查询）。
3. **稀疏物理约束回放机制有效缓解了灾难性遗忘**，即使只保留约 10% 的物理点，也能显著提升早期任务的保留精度。
4. **各组件效果具有方程依赖性**：没有一种组合在所有问题上都最优，必须针对具体PDE进行验证和选择。
5. **ACR2-finetune** 可作为下游快速适配模块，在请求特定参数时进一步提升局部精度，且在线成本低（数十秒内完成）。

### 方法的局限性
- **物理损失 ≠ 真实误差代理**：PDE残差不能完全反映最终预测精度，存在误导可能。
- **超参数敏感性**：如先验函数 `fprior`、探索系数 `κ` 等需根据问题调整。
- **验证集中在连续参数域**：尚未充分验证离散或混合参数空间的表现。
- **组件交互复杂**：多个机制共同作用，因果归因较难。

### 未来工作方向
- 探索更可靠的**任务不确定性度量**，指导参数选择。
- 将参数选择与物理约束回放扩展至更高维参数空间。
- 结合带有观测数据的任务，发展混合监督下的持续学习PINN。
- 探索在 **Physics-Informed Operator Learning** 模型中的应用。
- 开发自动化超参数调优流程以适应不同类型的PDE。

> 📌 **总体而言，CL-PINN 为构建可复用、高效、泛化的物理知情代理模型提供了实用路径，特别适合大规模工程参数研究场景**。

</details>

---

### 7. [NSF-HRPT: Neural Semantic Field meets Hierarchical Risk Perception Tree for Safety-Critical Scenario Assessment](https://arxiv.org/abs/2608.04776)

**Authors**: Yu Zhao, Jiangyu Pan, Tao Hu, Ming Yin, Fan Yang, Jiangfan Liu, Xiubo Liang  
**Category**: cs.AI  
**Published**: 2026-08-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.04776v1  

#### Abstract
The ability to accurately assess and anticipate risks in safety-critical scenarios is crucial for autonomous driving systems. While existing research has made progress in collision prediction, accurately quantifying risk levels from monocular vision inputs remains challenging due to the complex dyna...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# NSF-HRPT: Neural Semantic Field meets Hierarchical Risk Perception Tree for Safety-Critical Scenario Assessment  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**自动驾驶系统在安全关键场景下的风险评估挑战**，特别是从单目视觉输入中实现**精确、实时、可解释的多智能体风险量化**。现有方法面临三大难题：
- **真实世界中关键事故数据稀疏且标注不完整**（如缺乏BEV轨迹）；
- **多智能体场景下风险推理计算效率低**，难以满足实时性要求；
- **仿真到现实（Sim-to-Real）存在显著视觉域差异**，导致模型泛化能力差。

### 🚀 提出的新方法与创新思路
作者提出 **NSF-HRPT** 框架，融合学习与结构化推理，包含以下四大核心贡献：

1. **Neural Semantic Field (NSF)**  
   - 一种基于查询的连续表征模型，统一建模场景语义、运动轨迹与概率化的 **Time-to-Collision (TTC)** 分布；
   - 在CARLA仿真环境中训练，输出每个空间位置在未来时间点的语义类别、轨迹偏移量及TTC均值与方差（不确定性建模）。

2. **Hierarchical Risk Perception Tree (HRPT)**  
   - 一个并行四叉树结构，用于高效组织动态实体并进行分层风险传播；
   - 支持**并行化计算**与**自底向上的风险聚合**，实现实时多智能体风险评估；
   - 引入 `RiskMask` 和 `ActiveMask` 机制，支持快速风险源定位与稀疏更新。

3. **Sim2Real Enhancement Strategy**  
   - 无需微调即可提升现实世界适应性的增强策略，通过引入预训练基础模型（foundation models）提供几何与语义先验；
   - 包括两种融合方式：
     - **Early-Fusion (EF)**：将深度图与语义特征拼接至原始图像输入；
     - **Mid-Fusion (MF)**：在特征层面加权融合外部特征，保留NSF参数冻结。

4. **端到端可验证的风险预测框架**  
   - 实现从单目视频序列到个体级TTC估计、风险值计算、事故对象定位的全流程闭环；
   - 支持定量风险评估（而非仅分类），为决策模块提供物理意义明确的安全信号。

### 🔍 相比现有方法的优势
| 维度 | NSF-HRPT优势 |
|------|--------------|
| **任务粒度** | 支持**个体级TTC预测**，优于传统场景级mTTA或AP指标 |
| **推理效率** | HRPT结构支持**并行计算与层次化聚合**，适合实时部署 |
| **不确定性建模** | 显式输出TTC的概率分布（高斯参数），提升鲁棒性 |
| **Sim-to-Real迁移** | 无需重训练即可通过EF/MF增强现实泛化能力 |
| **可解释性** | 风险可通过HRPT树结构追溯至具体智能体，支持故障诊断 |

---

## 2. 核心实验方法和设置

### 📚 数据集使用

| 数据集 | 类型 | 规模 | 特点 |
|--------|------|-------|------|
| **CARLA Simulation** | 合成数据 | 800个高危场景（8类） | 来自SafeBench平台，含精确BEV轨迹、TTC真值、语义分割；涵盖直行障碍、变道、闯红灯等NHTSA典型场景 |
| **DAD** | 真实世界 | 1,750段5秒视频（620正样本） | 行车记录仪视角，事故集中在最后10帧；无BEV标注 |
| **CCD** | 真实世界 | 4,500视频（1,500事故 + 3,000正常） | 更多样环境，配合BDD100K数据增强；事故发生在最后2秒 |

> ⚠️ 所有模型均**仅在CARLA合成数据上训练**，直接迁移到DAD/CCD测试，验证Sim-to-Real能力。

### 📊 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **AP (%)** | Average Precision，PR曲线下面积 | 评估事故检测准确率，处理类别不平衡 |
| **mTTA (s)** | mean Time-to-Accident，首次有效预警到事故发生的时间 | 要求预测TTC误差 ≤ 0.3s才视为“有效预警”，更严格可靠 |
| **AOLA** | Accident Object Localization Accuracy，正确识别涉事对象的比例 | 评估空间定位精度，衡量是否能指出“谁要撞” |

### 🔀 基线方法对比
选取代表三类主流技术路线的方法作为基线：
- **Attention-based**: DSA [6], adaLEA [40], DSTA [11]
- **Graph-based**: Ustring [7], GG [26]
- **LLM-based**: Liao et al. [33]（预测“何时、何地、何物”）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ 在CARLA仿真基准上的表现（Table 1）
| 方法 | AP (%) ↑ | mTTA (s) ↑ | AOLA ↑ |
|------|----------|------------|---------|
| DSA | 73.8 | 2.42 | — |
| Liao et al. | 79.3 | 3.21 | 0.84 |
| **Ours (NSF-HRPT)** | **84.3** | **3.54** | **0.90** |

👉 **全面超越SOTA**，尤其在mTTA上领先超过0.3秒，说明能更早发出有效警告。

#### ✅ 在真实世界数据集上的表现（Table 2）

| 方法 | DAD: AP | DAD: mTTA | DAD: AOLA | CCD: AP | CCD: mTTA |
|------|--------|-----------|-----------|--------|-----------|
| DSA | 48.1 | 1.34 | — | 98.7 | 3.08 |
| Liao et al. | 69.2 | 4.26 | 0.89 | 99.7 | 3.93 |
| **Ours (base)** | 56.7 | 3.72 | 0.65 | 99.2 | 3.67 |
| **Ours + EF** | 67.2 | 4.13 | 0.85 | 99.8 | 3.87 |
| **Ours + MF** | **68.5** | **4.23** | **0.89** | **99.9** | **3.95** |

📌 **关键发现**：
- 即使未增强，**base模型已优于部分早期SOTA**（如DSTAs）；
- 加入**Mid-Fusion后性能逼近当前最优水平**，在DAD上达到68.5% AP，在CCD上mTTA达3.95s，**接近Liao et al.的4.26s**；
- AOLA从0.65提升至0.89，表明Sim2Real增强显著改善了**事故对象定位能力**。

### 🔍 消融实验结果（Ablation Studies）

#### 在DAD上的组件分析（Figure 4）
- 移除`Positional Embedding`或`Transformer`导致AP下降明显（~10%）；
- 移除`Learnable Type Embedding T`影响较大，说明**不同类型智能体的行为先验至关重要**；
- `Semantic Loss`移除后性能下降，证明**语义理解有助于风险感知**；
- **EF/MF单独使用均有增益，联合使用效果最佳**。

#### 在CARLA上的消融（Figure 5）
- 移除`parallel HRPT`结构导致mTTA下降至约3.0s；
- `RiskMask`与`ActiveMask`对AOLA有显著贡献（下降至~0.65）；
- 多任务损失（尤其是TTC Loss）对整体性能至关重要。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **NSF-HRPT实现了学习与推理的有效统一**：NSF负责从仿真数据中学出丰富的时空语义与动力学表征，HRPT则将其转化为高效的结构化风险推理引擎。
2. **支持个体级、带不确定性的TTC预测**，相比场景级预测更具实用性，能支撑下游规划控制模块做出精细化避障决策。
3. **HRPT的并行四叉树设计显著提升了推理效率**，适用于复杂城市交通中的多智能体实时风险监控。
4. **Sim2Real增强策略无需微调即可桥接域差距**，是解决真实世界泛化问题的一种轻量高效方案。

### ⚠️ 局限性
- 当前NSF依赖高质量仿真数据训练，若仿真与现实差距过大（如极端天气、罕见行为），仍可能失效；
- HRPT的空间划分基于固定网格，在高度密集或非均匀分布场景中可能存在负载不均问题；
- 尽管AOLA较高，但**尚未完全达到人类级别的细粒度因果归因能力**。

### 🔮 未来工作方向（来自Conclusion）
1. **混合训练范式**：结合少量真实世界数据与仿真数据，采用半监督或对抗训练进一步提升域适应能力；
2. **引入Large Language Models (LLMs)**：利用LLM进行高层风险语义解释，生成自然语言警告（如“左侧车辆即将强行变道”），提升可读性与交互性；
3. **构建带BEV标注的真实世界高危事故数据集**：推动社区建立更完善的评测基准，促进安全关键感知研究发展。

---

## 总结

> ✅ **NSF-HRPT是一个面向安全关键场景的新型风险评估框架**，它通过**Neural Semantic Field**实现对语义、轨迹与TTC的联合建模，并借助**Hierarchical Risk Perception Tree**完成高效并行的风险推理。结合无需微调的**Sim2Real增强策略**，该方法在CARLA上达到SOTA，在DAD/CCD上也展现出强大的跨域泛化能力。  
>
> 💡 这项工作标志着从“能否检测事故”向“如何量化风险、何时预警、谁是威胁”的精细化安全认知迈进了一大步，为下一代**embodied intelligence**系统的安全决策提供了坚实基础。

</details>

---

### 8. [State2State: Environment-Derived Mid-Training for LLM Agents](https://arxiv.org/abs/2608.04934)

**Authors**: Xuanyu Lei, Yiqi Zhu, Chenliang Li, Kaiming Liu, Peng Li, Ming Yan, Jieping Ye, Ya-Qin Zhang, Yang Liu  
**Category**: cs.CL  
**Published**: 2026-08-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.04934v1  

#### Abstract
Training LLM agents commonly relies on supervised fine-tuning from expert trajectories or online reinforcement learning over human-specified tasks with handcrafted verifiers. Though effective, both remain bottlenecked by externally specified tasks and supervision signals, limiting the scalability an...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **State2State: Environment-Derived Mid-Training for LLM Agents**  
—— 核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### **解决的问题**
当前训练 LLM Agents 的主流方法依赖于两种范式：
- **Supervised Fine-Tuning (SFT)**：依赖专家轨迹（expert trajectories），成本高、覆盖有限。
- **Reinforcement Learning with Verifiable Rewards (RLVR)**：依赖人工设计的任务目标和奖励验证器（verifiers）。

这两种方式都严重受限于**外部指定的任务分布和监督信号**，导致训练可扩展性差、多样性不足，且难以泛化到未见任务。

### **提出的新方法与新思路**
本文提出 **STATE2STATE**，一种基于环境的中段训练（mid-training）框架，其核心思想是：
> **从环境中直接派生训练目标与奖励信号，无需专家示范、人工任务指令或任务特定的验证器。**

具体流程如下：
1. **环境探索（Environment Exploration）**：用随机策略在环境中交互，收集可达状态（observations）。
2. **任务构建（Task Construction）**：将探索到的状态作为“目标状态”，配对初始配置，形成“从初始状态到达目标状态”的 state-reaching 任务。
3. **中段训练（Mid-Training）**：使用 GRPO-based RL 训练模型达成这些状态，通过规则匹配（rule-based state matching）判断成功与否，获得二元奖励。
4. **下游微调（Downstream RL）**：以该策略为初始化，再进行标准的人类任务 RL 微调。

### **相比现有方法的优势**
| 维度 | 传统方法（SFT / RLVR） | STATE2STATE |
|------|------------------------|-------------|
| **任务来源** | 人类编写或教师模型生成 | 环境探索自动派生 |
| **监督信号** | 专家动作或任务特定奖励 | 规则匹配的目标状态 |
| **可扩展性** | 受限于标注成本 | 高度可扩展（仅需环境交互） |
| **多样性** | 局限于人类任务分布 | 覆盖更广的可达状态空间 |
| **泛化潜力** | 依赖任务对齐 | 学习通用环境操作先验 |

---

## 2. 核心实验方法和设置

### **使用的数据集**
- **ALFWorld**：基于文本的具身家庭环境，涉及导航、清洁、加热等操作。
- **ScienceWorld**：科学实验导向的文本环境，要求长程推理与物理/化学操作。
- **MobileWorld（扩展实验）**：移动端 GUI 环境，测试方法在视觉界面中的适用性。

### **实验设置与评估指标**
- **评估指标**：任务成功率（task success rate），分别报告 in-distribution (ID) 和 out-of-distribution (OOD) 分割的结果。
- **模型规模**：Qwen3-4B 和 Qwen3-8B。
- **训练阶段划分**：
  - **STATE2STATE-only**：仅在环境派生任务上训练。
  - **RL-only**：直接在人类任务上进行 RL。
  - **STATE2STATE + RL**：先用 STATE2STATE 中段训练，再接人类任务 RL。
- **RL 算法**：GRPO（Group Relative Policy Optimization） + Dynamic Sampling。
- **温度设置**：eval 时使用 temperature=0.1，training 时使用 temperature=1.0。

### **基线方法对比**
| 基线方法 | 描述 |
|--------|------|
| **Prompting-based LLMs** | GPT-5.2, Claude 4.5 Haiku, DeepSeek V4 Flash, Qwen-Plus（零样本提示） |
| **RL-only** | 直接在人类任务上进行 RL，无中段训练 |
| **Distillation SFT** | 使用 Qwen-Plus 收集高质量轨迹进行 SFT |
| **Agent Early Experience (AEE)** | 基于自我反思的 SFT 扩展方法 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Table 1）**

| Model | Method | ALFWorld (Avg) | ScienceWorld (Avg) |
|-------|--------|----------------|--------------------|
| Qwen3-4B | RL | 86.13 | 49.63 |
| Qwen3-4B | **STATE2STATE + RL (Ours)** | **92.00** | **55.50** |
| Qwen3-8B | RL | 92.36 | 52.13 |
| Qwen3-8B | **STATE2STATE + RL (Ours)** | **97.45** | **56.00** |

> ✅ STATE2STATE + RL 在两个基准、两个模型尺度上均达到**最佳性能**。

### **与基线方法的对比结果**
- **超越强提示模型**：在 ALFWorld 上，Qwen3-8B + STATE2STATE + RL 达到 **97.45%**，显著优于 GPT-5.2（77.04%）、Claude（80.32%）等闭源大模型。
- **提升下游 RL 效率**：如 Figure 3 所示，在 ScienceWorld 上，STATE2STATE 初始化使模型在约 **50 步内达到 RL-only 在 150 步才达到的分数**，学习效率大幅提升。
- **OOD 泛化更强**：在 OOD 任务上也取得一致提升，表明学习的是**环境级能力**而非过拟合特定任务分布。

### **消融实验结果**
#### （1）**STATE2STATE 作为独立训练阶段的效果**
- 在大多数设置下，STATE2STATE 单独训练即可提升 base model 性能（如 Qwen3-8B 在 ALFWorld 上从 74.29 → 77.17）。
- 例外：Qwen3-4B 在 ScienceWorld 上略有下降（23.63 → 21.00），可能因小模型难以应对稀疏的 exact-match 目标。

#### （2）**与不同 RL backbones 的兼容性（Table 3）**
| Method | ScienceWorld (Avg) |
|--------|--------------------|
| GRPO | 49.63 |
| STATE2STATE + GRPO | 55.50 |
| GiGPO | 54.75 |
| **STATE2STATE + GiGPO** | **57.75** |

> ✅ STATE2STATE 的收益与下游 RL 算法正交，即使在更强的 GiGPO 上仍能带来增益。

#### （3）**探索策略的影响（Table 4）**
| 探索策略 | ScienceWorld (Avg) |
|--------|--------------------|
| LLM Explorer (Qwen-Plus) | 52.63 |
| **Random Explorer** | **55.50** |

> ✅ **随机探索效果更好**，因其避免了 LLM 先验偏向人类任务，从而发现更多样、更深的状态。

#### （4）**跨环境迁移能力（Table 5）**
| Mid-Training | ALFWorld (Avg) |
|--------------|----------------|
| None | 86.13 |
| ScienceWorld RL | 86.87 |
| **ScienceWorld STATE2STATE** | **89.44** |

> ✅ 在 ScienceWorld 上进行 STATE2STATE 中段训练，能有效迁移到 ALFWorld，而标准 RL 任务迁移几乎无效，说明 STATE2STATE 学到了**可泛化的环境操作技能**。

#### （5）**在 GUI 环境中的扩展（Table 6）**
| Method | MobileWorld (GUI-only) |
|--------|-------------------------|
| MAI-UI-8B (base) | 0.275 |
| **STATE2STATE** | **0.308** |

> ✅ 在无下游 RL 的情况下，STATE2STATE 仍能在 MobileWorld GUI 环境中提升性能，验证其在复杂视觉界面中的潜力。

---

## 4. 关键结论和发现

### **主要发现**
1. **环境本身可以成为强大的监督来源**：无需专家、无需人工任务，仅通过探索和状态匹配即可生成有效的训练目标。
2. **STATE2STATE 是高效的中段训练范式**：
   - 作为 standalone 方法可提升基础能力；
   - 作为初始化可显著增强下游 RL 的最终性能与学习效率。
3. **学习到的能力具有跨任务与跨环境泛化性**：在 ScienceWorld 上学到的技能可正向迁移到 ALFWorld。
4. **随机探索优于智能探索**：在当前环境下，低代价的随机探索反而能发现更丰富、更具挑战性的状态。

### **局限性**
1. **依赖可重现的状态与可观测的观察值**：要求环境支持状态重置与可靠 observation 输出。
2. **exact-match 奖励可能过于严格**：尤其对小模型或复杂状态，可能导致稀疏奖励问题。
3. **尚未在更大规模模型上验证**：实验集中在 Qwen3-4B/8B，前沿超大规模模型的表现尚待研究。
4. **当前评估局限于受控仿真环境**：在真实世界设备、Web 或软件控制场景中的有效性仍需验证。

### **未来工作方向**
- 将 STATE2STATE 扩展到 **Web-based、GUI-heavy、real-device control** 等更复杂的交互环境。
- 结合 **partial state matching** 或 **semantic similarity** 来缓解 exact-match 的严苛性。
- 探索 **自适应难度调度**，动态选择适合当前策略水平的目标状态。
- 研究如何将 STATE2STATE 与 **world model learning** 或 **self-reflection** 机制结合，进一步提升环境理解深度。

---

> 🔚 **总结一句话**：  
> **STATE2STATE 开辟了一条“从环境中学习环境”的新路径，让 LLM Agents 能像婴儿一样，通过纯粹的交互积累世界经验，最终成长为更强大、更通用的智能体。**

</details>

---

### 9. [Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning](https://arxiv.org/abs/2608.05139)

**Authors**: Yinghui He, Ling Yang, Jiarui Liu, Yongjin Yang, Lechen Zhang, Yingcheng Wu, Zhenfei Yin, Mengdi Wang, Sanjeev Arora  
**Category**: cs.CL  
**Published**: 2026-08-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.05139v1  

#### Abstract
Long-horizon reasoning in recent LLMs demands that the model switch between distinct skills inside a reasoning chain, such as first doing a math derivation, then using the result to plan a schedule. We call such problems cross-skill long-horizon tasks: multi-step tasks whose steps require different ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大语言模型（LLMs）在**长程推理**（long-horizon reasoning）任务中表现出色，但现实中的复杂任务往往需要模型在多个步骤中切换不同的**推理技能**（reasoning skills），例如先进行数学推导，再用结果规划日程。现有基准（benchmarks）通常只评估单一技能，缺乏对“**技能切换难度**”（skill-switching difficulty）的系统性度量。这导致一个关键问题：即使模型在每个独立技能上表现良好，也可能在跨技能任务中失败。

本文指出，这种“**技能切换鸿沟**”（skill-switching gap）是当前评估体系无法捕捉的结构性缺陷。

### 提出的新方法与新思路
为解决上述问题，论文提出了两个核心创新：

#### （1）**Skill Entropy**（技能熵）
- 一种**定向的成对度量**（directed pairwise measure），用于量化从一个技能 `sa` 切换到另一个技能 `sb` 的难度。
- 其计算基于参考模型（reference model）在单一技能任务和双步跨技能任务上的准确率差异，公式如下：
  ```
  SkE(sa, sb) = [Accuracy(sa) + Accuracy(sb) + α] / [2 * Accuracy(sa, sb) + α]
  ```
  其中 `Accuracy(sa, sb)` 是模型先用 `sa` 再用 `sb` 的平均准确率。值越接近1表示切换越容易，越大则表示切换越难。
- **任务级技能熵**（task-level skill entropy）通过对任务中所有连续技能对的 `SkE` 取平均得到，用于衡量整个任务的技能切换难度。

#### （2）**Skill2-Bench**
- 一个全新的**跨技能长程推理基准**（cross-skill long-horizon benchmark）。
- 基于 **558 个标注技能**（labeled skills），覆盖 **9 个领域**（domains）：数学、科学、编码、逻辑、信息提取、规划、创意写作、上下文检索、指令遵循。
- 每个任务都根据其任务级技能熵被分为**低、中、高**三个难度等级。

#### （3）**Skill-Entropy RL**
- 一种新的**强化学习框架**（RL framework），将 `Skill Entropy` 作为训练信号。
- 模型在生成答案的同时，还需预测每一步所使用的**技能标签**（skill label）。
- 总奖励（reward）由两部分组成：
  - `rans`：基于答案正确性的标准奖励。
  - `rent`：**技能熵奖励**（skill-entropy reward），衡量模型预测的技能序列与真实技能序列在“技能熵等级”上的一致性。
  - 总奖励：`r = λ_ans * rans + λ_ent * rent`

### 相比现有方法的优势
- **填补空白**：首次系统性地定义并量化了“技能切换”这一关键能力，而不仅仅是单个技能的表现。
- **可扩展性强**：`Skill Entropy` 不仅可用于构建新基准，还能直接作为训练信号，应用于现有的训练数据（如 OpenR1-Math），无需大规模重构数据集。
- **效果显著**：提出的 `Skill-Entropy RL` 方法在 `Skill2-Bench` 上实现了远超基线的性能提升。

---

## 2. 核心实验方法和设置

### 数据集
- **Skill2-Bench**：论文自建的测试集，包含 **300 个**合成的跨技能长程任务，平衡分布在9个领域和3个技能熵难度等级上。
- **训练数据**：
  - 主要训练使用了基于 `Skill2-Bench` 流水线合成的 **9K 跨技能任务**（来自6个可验证领域）。
  - 验证了方法的通用性，将其应用于**现成的训练数据** `OpenR1-Math`（一个纯数学推理数据集）。

### 实验设置和评估指标
- **模型**：
  - **前沿模型**（Frontier models）：Claude-opus-4.7, GPT-5.5, Gemini-3.1-pro 等共8个。
  - **开源模型**（Open-source models）：Qwen3-4B-Instruct, Qwen3-1.7B 等共4个。
- **评估模式**：
  - **单技能模式**（Single-skill）：将多步任务拆解，每一步独立提问。
  - **跨技能模式**（Cross-skill）：输入完整的长程任务，要求模型按顺序回答所有步骤。
- **主要指标**：
  - **Skill2-Bench Performance**：模型在跨技能模式下的平均每步得分。
  - **Domain Accuracy**：在单技能和跨技能模式下各领域的准确率，用于计算“性能下降”（△）。
  - **技能熵等级**：分析模型在低、中、高技能熵任务上的表现趋势。

### 基线方法对比
- **无微调**（Base model）
- **监督微调**（SFT）
- **GRPO**：仅使用答案正确性奖励的强化学习。
- **Skill-Distill** [63]：基于技能选择的蒸馏微调。
- **SkillRL** [56]：递归技能增强的强化学习。
- **STAT** [16]：针对最弱技能的自适应训练。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 `Skill2-Bench` 上，**所有模型均表现出明显的“技能切换鸿沟”**：
  - 准确率随着任务级技能熵的升高而**单调下降**。
  - 即使是强大的 `GPT-5.5` 和 `Gemini-3.1-pro`，在跨技能任务中也损失了约 **4-5%** 的准确率。
  - 较小的模型（如 `Qwen3-4B`）损失更大，高达 **12.6%**。

- **Skill-Entropy RL 的性能提升极为显著**（见 Table 3）：
  - 在 **Qwen3-4B-Instruct** 上，`Skill2-Bench` 分数从 **34.4%**（Base）提升至 **68.4%**。
  - 在 **Qwen3-1.7B** 上，分数从 **14.6%** 提升至 **40.1%**。
  - 这一提升不仅超过了 `GRPO`（+9.6% 和 +7.9%），也大幅超越了其他技能感知的基线方法（如 `STAT`）。

### 与基线方法的对比结果
- `Skill-Entropy RL` 在 **7/9 个领域**上取得了最佳的单领域准确率。
- 其最大的优势体现在**创意写作**（Creative Writing）等开放域任务上，表明该方法具有良好的泛化能力。
- 在外部基准（如 MuSR, GPQA-Diamond）上的测试也显示，`Skill-Entropy RL` 的性能优于或等于基线，证明其不会损害通用推理能力。

### 消融实验结果
- **奖励权重消融**（Table 13）：
  - 最佳性能出现在 `λ_ans=0.7`, `λ_ent=0.3` 时。
  - 当 `λ_ent` 过大（如 0.7）时，性能急剧下降，说明**答案正确性奖励仍是主导**，技能熵奖励应作为辅助的“结构塑造”（structural shaping）信号。
- **与单技能上限对比**（Table 14）：
  - `Skill-Entropy RL` 在**跨技能模式**下的最终得分（68.4%）甚至**超过了基础模型在单技能模式下的得分**（62.2%），证明该方法确实提升了模型的底层能力，而不仅仅是格式适应。

---

## 4. 关键结论和发现

### 主要发现
1. **存在显著的“技能切换鸿沟”**：LLMs 在处理需要频繁切换不同推理技能的长程任务时，性能会显著下降，且这种下降与任务的“技能熵”正相关。
2. **主要失败模式是“技能惯性”**：模型倾向于在后续步骤中复用前一步的技能和答案形式，而不是根据当前步骤的需求进行切换。
3. **Skill Entropy 是有效的度量和信号**：它不仅能可靠地衡量任务难度，还能作为一个强大的训练信号，引导模型学习如何更好地进行技能切换。
4. **Skill-Entropy RL 极其有效**：通过显式地让模型预测技能序列并给予相应的奖励，可以极大地弥合“技能切换鸿沟”，性能提升超过100%。

### 方法的局限性
- **依赖参考模型**：`Skill Entropy` 的计算依赖于一个固定的参考模型（文中使用 Claude-opus-4.7）。尽管进行了鲁棒性分析，但其绝对值可能受参考模型能力的影响。
- **技能标注成本**：虽然方法可应用于现成数据，但前提是需要对训练数据进行技能标注，这可能需要额外的人工或模型成本。
- **领域覆盖**：`Skill2-Bench` 覆盖了9个领域，但现实世界中的技能组合是无限的，基准的全面性仍有提升空间。

### 未来工作方向
- 探索更自动化、低成本的技能标注方法。
- 将 `Skill Entropy` 应用于更广泛的场景，如多模态推理、具身智能体（embodied agents）等。
- 研究如何让模型**自主发现和定义**新的技能，实现真正的“技能原生”（skill-native）学习。
- 构建动态的、可进化的 `Skill2-Bench`，以持续挑战和推动模型的技能组合能力。

</details>

---

### 10. [Active Learning Guided Design Space Refinement for Scalable Multi-Objective Bayesian Optimization in Materials Discovery](https://arxiv.org/abs/2608.04651)

**Authors**: Alexandros Ntagiantas, Panagiotis Tsilimidos, George Giannakopoulos, Christoforos Rekatsinas, Panagiotis Krokidas  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.04651v1  

#### Abstract
Advanced materials discovery increasingly relies on machine learning and Bayesian optimization to explore large discrete design spaces under limited evaluation budgets. However, conventional Bayesian optimization (BO) can become inefficient as candidate spaces grow, often evaluating low-value region...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Active Learning Guided Design Space Refinement for Scalable Multi-Objective Bayesian Optimization in Materials Discovery*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Bayesian Optimization (BO)** 在面对**超大规模离散设计空间**时存在显著的可扩展性瓶颈。随着候选材料数量的增长，BO 容易在早期阶段浪费大量昂贵的评估资源于低价值区域，导致收敛缓慢，尤其在 **multi-objective** 场景下更为严重。

此外，现有的优化框架通常独立处理预测建模与优化过程，缺乏对搜索空间进行**自适应精简**的机制，限制了其在高通量计算筛选背景下的实用性。

---

### ✅ 提出的新方法与创新思路
本文提出了一种 **active learning (AL)-guided 自适应搜索空间精简框架**，结合 **multi-objective Bayesian Optimization (MOBO)**，用于加速材料发现中的多目标优化。

#### 核心创新点包括：
- **DAGS-based Adaptive Refinement**：采用作者团队先前提出的 **density-aware greedy sampling (DAGS)** 作为主动学习策略，在正式 BO 之前对候选空间进行迭代过滤。
- **分类器引导的空间缩减**：使用单个 **XGBoost 分类器**区分“高性能”与“低性能”候选区域，并基于概率预测、不确定性估计和伪标签机制保留信息丰富的子集。
- **Warm-start BO 流程**：将 AL 阶段获得的高置信度样本作为初始观测值（`DDAGS`），传递给后续的 **qNEHVI**（batch noisy expected hypervolume improvement）BO 过程，实现“热启动”。
- **Pareto-aware Proxy Scoring**：设计了一个加权组合的代理评分函数（scalarized score），综合均值、最差项、理想点和几何平均等维度，用于生成训练标签。

---

### ✅ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率提升** | 显著减少无效探索，集中资源于 Pareto-relevant 区域 |
| **收敛速度** | 更快达到高原 hypervolume，尤其在前 20–50 次迭代中表现突出 |
| **信息保留能力** | 虽然空间缩小近半，但仍保留 >99% 的原始 hypervolume |
| **通用性** | 在工程设计（压力容器）与材料科学（COFs 吸附）两类差异显著的任务上均有效 |

相比直接在整个空间运行 BO 或使用 PSO/GA 等全局优化算法，该方法在有限预算下更高效地发现高质量 Pareto 解。

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集
1. **Pressure Vessel Design Benchmark**
   - 材料：碳纤维增强复合材料（CFRP）
   - 规模：52,272 种层压板配置
   - 参数：起始铺角、层间螺距、层数、对称标志、厚度模式
   - 目标：最小化三个目标  
     $$
     f_1 = S_{11},\quad f_2 = S_{22},\quad f_3 = T \text{（总壁厚）}
     $$

2. **CH₄/N₂ Separation in Covalent Organic Frameworks (COFs)**
   - 规模：69,839 个 COF 结构
   - 描述符：43 个结构性、拓扑性和物化特征
   - 目标：最大化  
     $$
     f_1 = \text{HighUptake}_{mol},\quad f_2 = \text{DelCapacity}
     $$
   - 真实 Pareto 最优解数量：78

---

### ✅ 实验设置
| 设置项 | 内容 |
|--------|------|
| **优化阶段预算** | 150 次新的 BO 评估（所有方法一致） |
| **AL 阶段设置** | 初始标注 16 个样本，平衡正负类；使用 8,000 候选池 |
| **分类器** | 单一 XGBoost 模型，更新 3 次（压力容器）或 5 次（COFs） |
| **标签生成** | 使用第 90 百分位阈值划分正负类（proxy score ≥ T） |
| **伪标签条件** | $ p(x) \geq 0.9 $ |
| **保留策略** | 保留五类候选：<br>① 分类器预测为正<br>② 高置信度（prob > 50%）<br>③ 高不确定性（top 65%）<br>④ 高置信伪标签<br>⑤ 所有 oracle 查询样本 |
| **BO 方法** | qNEHVI（BoTorch 实现），相同 acquisition 策略和候选池大小 |

---

### ✅ 评估指标
| 指标 | 定义 |
|------|------|
| **Reduction Ratio** | $\frac{|X_k|}{|X_0|}$，衡量空间压缩程度 |
| **Hypervolume (HV) Retention** | $\frac{\text{HV}(X_k)}{\text{HV}(X_0)}$，反映 Pareto 前沿质量保留情况 |
| **Pareto Retention (Pret)** | $\frac{|P_k \cap P_0|}{|P_0|}$，真实 Pareto 解保留比例 |
| **Top-Good Retention** | 高排名候选保留率 |
| **Hypervolume Convergence Curve** | HV 随 BO 迭代次数的变化趋势 |
| **Pareto Discovery Curve** | 累计发现的真实 Pareto 解数量随迭代变化 |
| **Pareto-AUC** | Pareto 发现曲线下的面积，量化整体发现效率 |

---

### ✅ 基线方法对比
- **Full BO**：直接在整个原始空间上运行 MOBO，仅用 5 个随机初始点。
- **DAGS + BO Warm-start**：本文提出的方法，利用 AL 阶段的 ~19–21 个 oracle 样本作为 warm-start。

> ⚠️ 注意：warm-start 方法额外使用了 AL 阶段的 oracle 评估，因此总评估数略高于 Full BO，但 BO 阶段仍控制为 150 次。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据汇总（Table 2）

| 数据集 | Reduction Ratio | Pareto Retention | HV Retention | Pareto-AUC（改进） |
|--------|------------------|--------------------|---------------|-----------------------|
| Pressure Vessel | 49.6% ↓ | 69.9% | **99.6%** | 1562 → **2123** (+35.7%) |
| CH₄/N₂ COFs | 44.2% ↓ | 87.8% | **99.5%** | 2547 → **3199** (+25.6%) |

> ✅ 尽管移除了近一半候选者，**hypervolume 保留率超过 99%**，说明几乎未丢失关键 Pareto 区域。

---

### ✅ 与基线方法的对比结果

#### 🔹 Hypervolume 收敛速度（Figure 3）
- **DAGS + BO Warm-start** 在早期迭代中明显领先：
  - 压力容器：约 **20–30 次迭代**即接近 plateau，而 Full BO 需 **40–50 次**
  - COFs：同样提前进入高效探索阶段
- 最终 hypervolume 差异较小，但**路径更优**

#### 🔹 Pareto 发现效率（Figure 4 & 5）
- **累计 Pareto 发现数量始终更高**
- **Pareto-AUC 提升显著**：
  - 压力容器：+35.7%
  - COFs：+25.6%
- 表明方法不仅加快收敛，还提升了整个优化轨迹的信息获取效率

#### 🔹 空间缩减效果（Figure 2）
- 成功去除冗余区域，同时保留绝大多数 top-performing 和 Pareto-relevant 候选
- 特别是在 COFs 数据集中，尽管 Pareto 前沿更复杂，仍能保持 87.8% 的 Pareto 解被保留

---

### ✅ 消融实验（隐含分析）
虽然文中未明确列出消融实验表格，但从设计逻辑可推断以下关键组件的作用：
| 组件 | 功能 | 影响 |
|------|------|------|
| **Proxy Score 设计** | 构建分类监督信号 | 若仅依赖单一指标会偏向极端解，影响多样性 |
| **Uncertainty Retention (top 65%)** | 保留决策边界附近样本 | 防止误删潜在 Pareto 解 |
| **Pseudo-labeling ($p(x)\geq0.9$)** | 扩展训练集 | 加速分类器收敛，提高空间划分准确性 |
| **Warm-start Initialization** | 注入先验知识 | 显著提升 GP surrogate 初始代表性 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **AL-guided 空间精简可大幅提升 MOBO 效率**  
   在不牺牲最终优化质量的前提下，显著加速早期收敛和 Pareto 解发现。

2. **>99% hypervolume 保留是可行的**  
   大规模材料空间中存在高度冗余，通过智能过滤可安全压缩近半空间。

3. **Warm-start 是关键增益来源**  
   利用 AL 阶段积累的 informative samples 初始化 BO，使 GP surrogate 更快捕捉目标结构。

4. **方法具有良好的跨领域泛化能力**  
   在结构设计（压力容器）与分子材料（COFs）两种截然不同的任务上均取得一致提升。

---

### ❗ 局限性
1. **依赖离线数据集构建 proxy score**  
   当前 proxy label 的构造需要所有候选的目标值已知（offline setting），难以直接应用于完全在线场景。

2. **分类器性能影响空间质量**  
   若早期查询样本代表性不足，可能导致错误的空间剪枝。

3. **未提供理论保证**  
   缺乏关于 Pareto-set 保留率的理论边界分析。

4. **总 oracle 成本略有增加**  
   warm-start 使用了 AL 阶段的额外评估，虽提升 BO 效率，但总成本并非更低。

---

### 🔮 未来工作方向
1. **开发在线自适应阈值机制**  
   在无法预知全局目标分布时动态调整 labeling threshold。

2. **引入不确定性校准技术**  
   提升 XGBoost 概率输出的可靠性，避免过度自信导致误删。

3. **集成多保真度建模（multi-fidelity）**  
   在 AL 阶段使用低成本模拟进一步降低筛选开销。

4. **拓展至更多目标（many-objective）和约束优化问题**

5. **探索其他 surrogate 或 active learning 策略的组合可能性**

---

## 总结
本文提出了一种新颖且实用的 **two-stage AL-to-BO pipeline**，通过 **DAGS-guided adaptive refinement** 实现了在大规模材料设计空间中的高效多目标优化。其实验验证充分，结果稳健，在保持 >99% hypervolume 的前提下将搜索空间压缩近半，并显著提升 Pareto 发现阶段的速度与累积收益。该工作为**可扩展的自主材料发现系统**提供了重要范式参考。

</details>

---

### 11. [Monte Carlo Tree Search for Table-to-Multimodal Report Generation](https://arxiv.org/abs/2608.04071)

**Authors**: Teng Lin, Zhiyang Zhang, Yuyu Luo, Nan Tang  
**Category**: cs.AI  
**Published**: 2026-08-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.04071v1  

#### Abstract
Automatically generating professional multimodal reports comprising both textual analysis and visual charts from structured tabular data is a critical challenge in data intelligence. Existing methods suffer from fixed linear pipelines and isolated subtask processing, which hinder joint optimization ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Monte Carlo Tree Search for Table-to-Multimodal Report Generation**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

当前从结构化表格生成多模态报告（text + charts）的方法存在两大瓶颈：

- **固定线性流水线**（Fixed linear pipelines）：如“解析表 → 生成图表 → 写文本 → 润色”，缺乏回溯能力，导致“**insight freezing**”——一旦早期决策错误，后续无法修正。
- **子任务孤立处理**：图表生成、文本撰写、事实验证等模块分离优化，容易造成 **chart-text inconsistency**（图文不一致）、数值错误、叙事断裂。

此外，现有 **benchmarks** 多为纯文本输出或仅支持简单问答，缺乏对多模态报告中**事实准确性、图表质量、图文对齐、洞察新颖性**的综合评估体系。

---

### 🚀 提出的新方法与新思路

本文提出 **MCTS-Report**，一个基于 **Monte Carlo Tree Search (MCTS)** 的框架，将多模态报告生成建模为在搜索空间中的逐步构建过程。

#### 核心思想：
- 将报告生成分解为一系列原子动作（atomic actions），例如：
  - `Chapter Planning`（章节规划）
  - `Visualization Task Identification`
  - `Chart Generation` / `Modification`
  - `Insight Organization`
  - `Narrative Refinement`

- 使用 **单一 LLM** 作为统一的“action-evaluation engine”，在 MCTS 的每一步进行推理、执行动作并评估路径潜力。
- 构建一棵以部分报告状态为节点、动作为边的搜索树，通过 MCTS 迭代探索最优路径。

#### 创新机制：
- **动态回溯与全局优化**：MCTS 支持探索不同结构，避免陷入局部次优解。
- **统一 LLM 驱动**：无需多个 agent 协调，降低系统复杂性和错误传播风险。
- **自监督奖励函数**（self-supervised reward）：
  - 联合评估：`Tfact`（SQL 验证数值准确）、`Tvis`（图表技术正确性 + 数据保真）、`Tstruct`（结构完整性）、`Tnovel`（洞察新颖性）
  - 引入多样性惩罚和前置条件检查，抑制重复图表和无效操作。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | MCTS-Report |
|------|--------|-------------|
| 流程灵活性 | 固定流水线，不可回溯 | 可探索多种路径，支持 backtracking |
| 子任务协同 | 各自为政，易不一致 | 全局联合优化，强化图文一致性 |
| 错误纠正 | 无反馈机制 | 自监督 reward 指导搜索方向 |
| 系统设计 | 多 agent 协调复杂 | 单一 LLM 统一驱动，简洁鲁棒 |

> ✅ **优势总结**：实现**端到端可微调的搜索式生成**，兼顾创造性与可靠性。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集：**MMRBench**

作者构建了一个全新的综合性 benchmark —— **MMRBench**，填补了领域空白。

| 属性 | 数值 |
|------|-----|
| 表格数量 | 185（79 英文 / 131 中文） |
| 领域覆盖 | Finance, Manufacturing, Healthcare, Education, Retail, IT Ops |
| 任务总数 | 386（279 单表 + 107 多表） |
| 平均单元格数 | 420,000 |
| 极大表（>50K cells） | 38 个 |
| 参考关键点（keypoints） | 1,834 条（平均 4.75/任务） |

> 所有参考报告均由 “LLM 生成 + 人类专家精修” 得到，确保高质量与可验证性。

---

### ⚙️ 实验设置

- **输入**：一张或多张结构化表格 $ T $ + 自然语言查询 $ q $
- **输出**：包含文本与嵌入式图表的完整多模态报告 $ R = (R_{\text{text}}, R_{\text{vis}}) $
- **最大 token 数**：4096（文本），最多生成 8 张图
- **所有模型使用相同 prompt 模板**，保证公平比较

---

### 📊 评估指标

采用 **GPT-4o 作为 single-model LLM-as-judge**，评分范围 1–100，四个维度：

| 指标 | 描述 |
|------|------|
| **Structural Completeness** | 是否包含必要章节（Overview, Data Overview, Core Analysis, Conclusion）且每章至少一段 |
| **Numerical Accuracy** | 文本中数值声明是否可通过 SQL 在源表中验证（容忍 ≤1%） |
| **Chart-Text Alignment** | 图表内容与文字描述是否逻辑一致、相互支撑 |
| **Insight Novelty** | 洞察是否非平凡（non-trivial），避免简单聚合陈述（如“总销售额上升”） |
| **Overall Score** | 四项平均得分 |

> ❗注意：MCTS 内部使用的 self-supervised reward 与最终 benchmark evaluation 分离，防止 reward hacking。

---

### 🆚 基线方法对比

共测试 **12 种 SOTA 模型**，分为三类：

#### （1）视觉-语言模型（Vision-Language Models）
- GPT-4o, GPT-5, Gemini-2.5-Pro, Gemini-3.5-Flash, Claude-4.5-Sonnet, Qwen3-VL-235B

#### （2）代码增强多模态系统（Code-Augmented Multimodal Systems）
- DeepSeek-R1（with code interpreter）, Qwen3-Coder-32B, TableGPT2-7B

#### （3）深度研究智能体（Deep Research Agents）
- Gemini Deep Research, ChatGPT Deep Research, Perplexity Sonar Deep Research

> ✅ 所有基线均为直接生成（direct generation），无搜索机制。

---

## 3. **主要实验结果和性能指标**

### 📈 主要性能数据（见 Table 4）

| Model | Structural | Numerical | Chart-Text | Novelty | **Overall** |
|-------|------------|-----------|------------|---------|-------------|
| **DeepSeek-R1** (best baseline) | 78.2 | 59.4 | 74.5 | 38.7 | **62.7** |
| **MCTS-Report (DeepSeek-R1)** | **88.6** | **73.1** | **88.7** | **61.3** | **77.9** |
| Human Baseline | 94.8 | 91.2 | 89.5 | 88.5 | **91.0** |

> 💡 **提升幅度**：相比最强基线 **+15.2 分**（77.9 vs 62.7），接近人类水平（差距约 13 分）

#### 关键发现：
- **Chart-Text Alignment 达 88.7**，几乎追平人类（89.5），说明 MCTS 显著提升了图文一致性。
- **Numerical Accuracy 提升明显**（+13.7 pts），得益于 SQL-based fact verification reward。
- **Novelty 仍有较大差距**（61.3 vs 88.5），表明 LLM 本身分析深度仍是瓶颈。

---

### 🔬 消融实验（Ablation Studies）

使用 **DeepSeek-R1** 作为 backbone，测试以下变体：

| Variant | Description | Overall | Δ vs Full |
|--------|-------------|--------|----------|
| **A: w/o MCTS** | 直接提示 LLM 执行动作序列（单路径） | 64.5 | -13.4 |
| **B: w/o Self-Supervised Reward** | 随机选择路径，无 reward 指导 | 67.3 | -10.6 |
| **C: Reduced Rollouts (N=5)** | 减少 MCTS rollout 次数 | 70.4 | -7.5 |
| **D: Full MCTS-Report** | 完整框架（N=10） | **77.9** | — |

> ✅ 结论：
- **MCTS 规划至关重要**，尤其在数值准确性和洞察新颖性上影响巨大。
- **自监督 reward 是引导搜索的关键信号**，否则退化为随机探索。
- 更多 rollout 带来持续增益，但边际效益递减。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **MCTS 能有效打破线性流水线限制**，实现全局优化与动态调整，显著提升报告质量。
2. **统一 LLM 驱动 + 自监督 reward** 架构简化系统设计，同时保障跨模态一致性。
3. **搜索机制有助于挖掘更深层次、非显而易见的洞察**，提高 novelty。
4. **MMRBench 成为首个支持多模态、多维度、可验证评估的 table-to-report benchmark**，推动该领域标准化发展。

---

### ⚠️ 方法的局限性

| 问题 | 描述 |
|------|------|
| **计算开销较高** | MCTS 多轮 rollout 导致延迟增加，不适合实时场景 |
| **底层 LLM 分析能力瓶颈** | 即使有搜索机制，novelty 和 multi-table reasoning 仍受限于 LLM 本身的推理能力 |
| **多表关联理解不足** | 在 finance 和 healthcare 领域出现较多 **multi-table confusion**（9.8% 错误） |
| **图表可读性有待提升** | 14.1% 报告存在 poor chart readability（如标签重叠、颜色混乱） |

---

### 🔮 未来工作方向

1. **高效搜索策略**：引入剪枝、early stopping、learned policies 加速 MCTS。
2. **更细粒度 reward 模型**：针对 multi-table join、temporal trend detection 设计专项 reward。
3. **减少幻觉机制**：加强数值验证环节，结合 formal reasoning 或 symbolic executor。
4. **提升 novelty**：引入 retrieval-augmented generation（RAG）或 curiosity-driven exploration。
5. **交互式报告生成**：扩展至 user-in-the-loop setting，支持迭代修改与反馈。
6. **拓展 MMRBench**：纳入更多行业（如能源、物流）、语言（如日语、西班牙语）和图表类型（地图、仪表盘）。

---

> 🧩 **一句话总结**：  
> **MCTS-Report 通过将多模态报告生成转化为 MCTS 搜索问题，实现了结构灵活、事实可信、图文一致的专业级报告自动生产，在新 benchmark MMRBench 上大幅超越现有方法，为 data intelligence 提供了新范式。**

</details>

---

### 12. [Interoceptive Attention as Dynamic Homeostatic Prioritization in a Foraging Agent](https://arxiv.org/abs/2608.04232)

**Authors**: St John Grimbly, Nicolas Kuske, Evert A. Boonstra, Bruce A. Bassett, Charel van Hoof, Rowan Hodson, Benjamin Rosman, Ryan Smith, Mark Solms, Jonathan P. Shock  
**Category**: cs.AI  
**Published**: 2026-08-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.04232v1  

#### Abstract
Biological systems must regulate competing needs under limited perceptual bandwidth, where sharpening one estimate costs the capacity to sharpen the others. Any fixed-budget system therefore has to decide where to allocate its perceptual precision. We study this in a foraging agent that must keep se...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Interoceptive Attention as Dynamic Homeostatic Prioritization in a Foraging Agent*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
生物系统在有限的感知带宽下必须调节多个竞争性的生理需求（如饥饿、口渴、窒息等）。由于无法同时精确监控所有内部信号，系统必须决定将有限的**interoceptive precision**（内感受精度）分配给哪个需求通道。本文研究了这一“感知选择”问题：一个agent应如何动态分配其有限的感知资源以优化生存与学习。

传统方法通常通过**homeostatic reward**机制引导行为（如Homeostatic RL），或将注意力建模为对外部刺激的选择。而本文关注的是**内感受层面的注意力机制**——即如何基于身体状态信念动态调整对不同内部信号的信任程度。

---

### 提出的新方法与新思路

作者提出了一种基于**active inference**框架的新型机制：**Dynamic Interoceptive Precision Allocation**（动态内感受精度分配），称为 **K-attention mechanism**。

#### 核心思想：
- 将**attention**形式化为对固定总预算 $ K $ 下各内感受通道观测似然函数 $ A(m) $ 中 precision $ K_m $ 的动态再分配。
- 在每一步，agent根据其当前的**body-state posterior belief**（身体状态后验信念），识别最紧急的需求通道 $ m^* = \arg\max \text{need}_m $。
- 将更高的 precision $ K_{\text{att}} = 0.90 $ 分配给该通道，其余通道共享剩余预算 $ K_{\text{un}} \approx 0.567 $，保持总和不变（$ \sum K_m \leq K = 2.60 $）。
- 这种 shaped likelihood 同时影响两个过程：
  1. **Belief update**（感知推断）
  2. **EFE planner**（策略规划中的Expected Free Energy最小化）

> ✅ 创新点：首次将**attention-as-precision-control**应用于多通道**interoception**，并在固定预算下实现**动态优先级路由**（dynamic homeostatic prioritization）。

---

### 相比现有方法的优势

| 方法类别 | 局限性 | 本文改进 |
|--------|------|--------|
| **Homeostatic RL** | 行为由hand-designed reward驱动，未直接调控感知精度 | 直接通过 precision 调控感知与决策双路径 |
| **Standard Active Inference Models** | 多用于单一exteroceptive任务或静态interoception控制 | 引入跨通道动态precision reallocation机制 |
| **Uniform Precision Allocation** | 所有通道平分资源，忽略实时紧迫性 | 动态聚焦于最需满足的通道，显著提升效率 |

> 🔍 优势本质：**用同一 precision shaping 同时优化 perception 和 planning**，形成闭环的自适应调节机制。

---

## 2. 核心实验方法和设置

### 实验环境：AFFECTWORLD
- 一个 $6\times6$ 的gridworld，包含两种资源：food（缓解hunger）、water（缓解thirst）。
- 每个episode最多60步，死亡条件：hunger/thirst降至0。
- suffocation通道非致命，仅降低停留在水上的价值；第四个通道为inert control（恒定信号，无信息量）。
- 共12种布局，分为三个难度等级：
  - **easy**: 起始点距最近资源1–2格
  - **medium**: 3–4格
  - **far**: 6格
- 主要结果使用 **11种布局**（排除L01作为保守处理）

---

### Agent设计与变量控制

| Agent类型 | Precision分配方式 | 总预算一致？ |
|---------|------------------|------------|
| **Uniform agent** | 所有4通道均分 $ K_m = 0.65 $ | 是（$ K=2.60 $） |
| **Attentive agent**（主模型） | 最需通道 $ K_{\text{att}}=0.90 $，其他 $ K_{\text{un}}\approx0.567 $ | 是 |
| **Anti-aligned control** | 故意选择**最不需要**的通道进行高精度分配 | 是 |
| **Inference-only ablation** | planner使用uniform likelihood，仅belief update阶段用shaped likelihood | 是 |
| **Planning-only ablation** | inference使用uniform likelihood，planner使用shaped likelihood | 是 |

> 💡 注意：所有agent共享相同的planner（H=3）、prior、learning rule等，唯一区别是precision allocation策略。

---

### 数据收集与评估指标

#### 主要指标：
- **Learning-phase survival rate**：前100次trial中成功存活至第60步的比例（mean over trials, seeds, layouts）
- **Plateau survival**：稳定期（如trial 10–20）的生存率
- **Channel-specific learning speed**：特定通道（如hunger）的likelihood模型收敛速度，按cumulative observation count对齐比较

#### 统计方法：
- 使用**cluster-bootstrap**（重采样(layout, seed)对）计算95% CI和p值
- 多重检验校正采用**Holm-Bonferroni**

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）

| Agent | Survival Rate | 95% CI | vs Uniform (p) |
|-------|---------------|--------|----------------|
| **Uniform agent** | 0.199 | [0.158, 0.240] | — |
| **Attentive agent** | **0.414** | [0.365, 0.463] | **≤10⁻⁴** |
| **Anti-aligned control** | 0.144 | [0.108, 0.181] | 0.004 |

> 📈 **结论**：动态注意力机制使学习阶段生存率**翻倍以上（~2.08×）**，且统计显著。

---

### 与基线方法对比结果

- **vs Uniform agent**：生存率从0.199提升到0.414（+21.5个百分点），p ≤ 10⁻⁴
- **vs Oracle planner**（已知真实环境模型）：oracle仅达0.66，说明任务本身极具挑战性
- **Easy-tier子集**（5 layouts）：attentive agent达0.696，uniform为0.407（+29 pp）

---

### 消融实验结果

#### （1）方向是否重要？→ **是！**
- **Anti-aligned control**（反向注意）表现差于uniform，证明收益来自“正确方向”的分配，而非单纯不均匀性。
- 图4显示：所有need-aligned selector均优于uniform；只有argmin（least-needed）失败。

#### （2）precision信号作用于何处？
- **Inference-only ablation**（planner看不到shaped likelihood）：
  - 默认先验下损失约20个百分点
  - 当prior rigidify（$ \alpha_0 = 10 $）时，性能崩溃至接近anti-aligned水平
- **Planning-only ablation**：
  - 在宽松先验下可维持高性能（0.865），表明planning路径足以承载大部分增益
- ✅ **结论**：shaped likelihood必须传达到**planner**才能发挥主要作用；inference端仅在强先验下必要。

#### （3）预算与鲁棒性测试
- 在 $ K \in [1.5, 4.0] $ 范围内，attentive agent始终领先uniform agent 32–56个百分点
- 高 $ K $ 时uniform agent出现**planner-overcommitment**现象，而selective allocation能避免
- 即使存在**non-stationarity**（food随机变为poison），优势仍存在（gap随突变率上升而缩小但仍为正）

#### （4）固定通道 vs 动态选择
- 在easy-tier（hunger主导）中，“always-attend-hunger”与dynamic agent表现相当
- 但在以下情况dynamic明显胜出：
  - **Rigid priors**（$ \alpha_0 \geq 10 $）：fixed模式~0.65，dynamic达~0.90
  - **Forced-multi-need switching**：dynamic比固定通道高+23.5 pp

---

### 学习速度差异（Sec. 3.5）

- **Attended channel learning更快**：
  - hunger模型收敛速度快约**2.4倍**
  - 对齐cumulative observation count后仍领先（图11a）：attentive @0.629 vs uniform @0.299（x≤750）
- anti-aligned control也快于uniform（因非均匀性带来加速），但慢于attentive（方向仍重要）
- ✅ 表明：precision routing不仅提高生存，还加快**per-observation learning efficiency**

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Selective interoceptive attention显著提升生存能力**：
   - 在相同precision预算下，动态聚焦最紧急需求可使学习期生存率**超过两倍**。

2. ✅ **收益来源于“方向正确”而非“分配不均”本身**：
   - 反向分配（anti-aligned）反而更差，说明机制依赖于真实的need tracking。

3. ✅ **Precision shaping同时作用于perception与planning**：
   - 约一半增益来自其对**EFE planner**的影响，强调了active inference中“感知-行动统一”的重要性。

4. ✅ **被关注通道的学习速度更快**：
   - 即使控制观察次数，attentive agent也能更快学习对应通道的动力学，留下可测量的行为痕迹。

5. ✅ **机制具有鲁棒性**：
   - 对prior rigidity、budget变化、环境非平稳性均有良好适应能力。

---

### 方法的局限性

| 局限性 | 说明 |
|-------|------|
| **Selector是手工设定的** | 并未训练一个learnable router，而是假设agent能准确估计need urgency |
| **Posterior calibration假设成立** | 若body-state belief严重失准，可能导致precision误分配 |
| **Preference-reweighting未对比** | 是否可通过修改preference $ C $ 达到类似效果尚未验证 |
| **Minimalist environment** | AFFECTWORLD是简化gridworld，不能直接推广到复杂生物系统 |

---

### 未来工作方向

1. **Develop a learnable precision router**：让agent自主学会何时、何地分配precision。
2. **Compare with preference-modulation baselines**：设计matched comparator测试是否per-trial Dirichlet acceleration是学习加速的关键。
3. **Extend to continuous & embodied settings**：将机制迁移到更真实的机器人或神经模拟平台。
4. **Link to neuroscience hypotheses**：探索该机制是否类比insular cortex中gain modulation现象（如Livneh et al., 2017）。
5. **Multi-agent & social homeostasis extensions**：考虑群体层面的需求协调。

---

> 🔚 **总结一句话**：  
> 本文证明，在active inference框架下，将有限的内感受精度动态分配给最紧迫的身体需求，是一种高效且生物学合理的注意力机制，它不仅能大幅提升生存率，还能加速学习，并揭示了precision在连接感知与决策中的核心作用。

</details>

---

### 13. [Strengthening Target-Language Features: SAE-Based Steering for Multilingual Inference](https://arxiv.org/abs/2608.04904)

**Authors**: Hongsheng Wang, Phlipp Koehn  
**Category**: cs.CL  
**Published**: 2026-08-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.04904v1  

#### Abstract
Multilingual large language models exhibit substantial performance differences across languages, while existing adaptation methods often require parameter updates and considerable multilingual training data. We propose an inference-time multilingual steering method that uses pretrained sparse autoen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Strengthening Target-Language Features: SAE-Based Steering for Multilingual Inference*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
多语言大语言模型（Multilingual LLMs）在不同语言上的表现存在显著差异，通常在英语等高资源语言上表现优异，而在低资源语言上性能较差。这种不均衡主要源于预训练语料中语言分布的高度不平衡。现有的多语言适配方法（如继续预训练、指令微调、表示对齐等）通常需要**参数更新**和**大量多语言训练数据**，限制了其在低资源语言中的应用。

### 提出的新方法与新思路
本文提出了一种**推理时多语言引导方法（inference-time multilingual steering）**，利用**稀疏自编码器（Sparse Autoencoders, SAEs）** 来识别并增强目标语言相关的特征，从而提升模型在下游任务中的表现，而无需任何模型参数更新。

核心思路如下：
- 利用多语言平行句对（parallel sentences），通过预训练的SAE比较不同语言的激活模式。
- 在每一层Transformer中，识别出与目标语言最相关的一小组**layer-specific SAE features**。
- 将这些特征解码为**steering signals**，并注入到模型的隐藏状态中，在生成答案前进行干预。

### 相比现有方法的优势
| 维度 | 本方法 | 现有方法 |
|------|--------|----------|
| 是否需要训练 | ❌ 不需要（zero-shot inference-time method） | ✅ 通常需要额外训练 |
| 是否更新参数 | ❌ 无参数更新 | ✅ 需要微调或对齐 |
| 数据需求 | ⚠️ 仅需少量平行句子用于特征识别 | ✅ 需要大规模多语言标注数据 |
| 可解释性 | ✅ 基于SAE的稀疏特征，具有较强可解释性 | ❌ 多数为黑箱操作 |
| 轻量化程度 | ✅ 推理时轻量干预 | ❌ 训练成本高 |

该方法是首个将SAE用于**提升多语言下游任务性能**（而非仅控制生成语言）的工作。

---

## 2. 核心实验方法和设置

### 使用的数据集

#### （1）特征识别语料（Feature-identification corpus）
- **FLORES-200**：包含500组多语言平行句子，覆盖研究中涉及的所有语言。
- 用途：用于估计各语言在SAE空间中的平均激活模式，**不参与下游评估**。

#### （2）下游评估基准
| 数据集 | 任务类型 | 语言数量 | 示例划分 | 输出形式 |
|-------|---------|--------|----------|----------|
| **XCOPA** | 因果常识推理（commonsense reasoning） | 9种语言<br>(et, id, it, sw, ta, th, tr, vi, zh) | 100验证 + 400测试 | 分类（A/B选择） |
| **XNLI** | 自然语言推断（natural language inference） | 9种语言<br>(de, es, fr, ru, sw, th, tr, vi, zh) | 100验证 + 400测试 | 三分类（entailment/neutral/contradiction） |
| **MGSM** | 多语言数学推理（mathematical reasoning） | 4种语言<br>(de, es, fr, ja) | 50验证 + 200测试 | 生成数值答案（exact match） |

### 实验设置与评估指标
- **模型**：`Gemma-3-12B-it`（120亿参数指令调优版本）
- **SAE配置**：
  - 使用Gemma Scope 2提供的官方SAE
  - 每层残差流（residual stream）后部署SAE
  - 每个SAE含16,384个潜在特征，采用small-L0变体
- **干预方式**：
  - 在选定Transformer层（主实验使用第47层）的最后一个prompt位置注入steering vector
  - 干预向量由SAE decoder从选中的top-k特征差异构造
- **评估指标**：
  - XCOPA & XNLI：Accuracy
  - MGSM：Exact Match Accuracy
- **超参选择**：
  - 在验证集上搜索steering coefficient `α ∈ [0.0, 2.5]`（XCOPA）或 `[0.0, 1.0]`（XNLI/MGSM），步长0.1
  - 测试时固定最优`α`

### 基线方法对比
- **Baseline**：原始未干预模型
- **Non-SAE Controls**：
  - Hidden-state scaling：直接放大原隐藏状态
  - Direct hidden-top-k：在原始隐藏空间选top-k维度干预
  - Least-squares English projection：学习向英语表示的线性映射（Wang et al., 2025）
- **SAE-based Variants**（消融实验）：
  - 不同reference（multilingual centroid vs. English）
  - 是否保留语义信息（language-only vs. full-representation）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Test Set）

| Dataset | Baseline Acc (%) | Ours Acc (%) | Gain (pp) |
|--------|------------------|--------------|-----------|
| **XCOPA** | 54.5 | 65.8 | **+11.3** |
| **XNLI** | 45.6 | 49.8 | **+4.1** |
| **MGSM** | 80.5 | 81.1 | **+0.6** |

> 注：原文摘要称“average improvements of 10.9 pp on XCOPA, 5.3 on XNLI, and 1.9 on MGSM”，可能为不同计算方式或笔误；表1中报告值更精确。

### 语言级结果亮点
- **XCOPA**：所有9种语言均提升，最大增益来自**Italian (+17.5)**、**Indonesian (+15.5)** 和 **Turkish (+13.8)**
- **XNLI**：多数语言提升，其中**Spanish (+11.8)**、**French (+11.3)**、**Chinese (+9.0)** 改善明显；但**German下降2.5 pp**
- **MGSM**：**German (+2.5)** 和 **French (+0.5)** 提升，**Japanese下降0.5**

### 与基线方法对比（XCOPA平均准确率）
| 方法 | Accuracy (%) | Gain vs Baseline |
|------|---------------|------------------|
| Baseline | 54.5 | — |
| Hidden-state scaling | 54.3 | ~0 |
| Direct hidden-top-k | 56.0 | +1.5 |
| **Ours (Multilingual, language-only)** | **65.8** | **+11.3** |
| English full-representation | 69.2 | +14.7 |
| Least-squares English projection | 68.4 | +13.9 |

> 注意：虽然某些SAE变体（如EN-Full）略高于主方法，但主方法设计更简洁且聚焦语言信号提取。

### 消融实验结果

#### （1）Steering Strength 影响（Figure 3）
- 在XCOPA上，平均增益随`α`增加先上升后下降，峰值出现在`α ≈ 1.4`
- 多数语言在较宽范围（`α ∈ [0.5, 2.0]`）内保持正向增益，表明方法鲁棒性强

#### （2）共享系数效果（Fixed α = 0.6）
即使使用统一的steering coefficient，仍能取得稳定增益：
- XCOPA: **+4.75 pp**
- XNLI: **+1.61 pp**
- MGSM: **+2.13 pp**
→ 表明无需每语言单独调参即可有效

#### （3）不同干预策略比较（Figure 4 & 9 & 10）
- **SAE-based方法显著优于非SAE方法**
- **是否保留语义信息影响因任务而异**：
  - XNLI：language-only 更好 → 说明干扰语义可能有害
  - XCOPA：full-representation 更好 → 上下文信息有助于推理
- **multilingual centroid作为reference略优于English**，说明改进不依赖“以英为中心”的假设

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **显式增强target-language features可有效提升多语言性能**  
   即使输出标签语言无关（如A/B/C），加强目标语言内部表示也能带来显著收益，说明语言信息本身影响推理质量。

2. ✅ **SAE能有效分离语言相关与语义中性成分**  
   通过对比平行句的SAE激活差异，可以定位到高度语言判别性的稀疏特征。

3. ✅ **sparse SAE features比原始隐藏空间更适合构建语言信号**  
   直接在hidden space选top-k维度效果远弱于SAE-based方法，证明SAE分解提升了信号纯度。

4. ✅ **语义信息在steering signal中非必需**  
   “language-only”信号已足够有效，说明语言增强可独立于语义内容进行。

5. ✅ **方法具有跨任务一致性**  
   在常识、推理、数学三类任务上均观察到正向增益，尽管幅度不同。

### 方法的局限性
- 🔒 **依赖高质量预训练SAE**：目前仅适用于少数公开SAE的模型（如Gemma系列），难以扩展至其他架构（如Llama、Qwen等）
- 🌍 **泛化性未知**：实验集中于Gemma-3-12B-it，尚不清楚是否适用于其他模型家族或SAE训练方式
- ⚖️ **性能提升不均匀**：部分语言（如German、Japanese）出现性能下降，机制尚不明确
- 📉 **强干预可能导致性能退化**：过大的steering strength会损害模型表现，需谨慎调参

### 未来工作方向
- 扩展至更多模型架构，推动通用SAE生态建设
- 探索自动选择最佳干预层与steering strength的方法
- 结合任务特定知识优化feature selection过程
- 研究为何某些语言响应负面，改进reference representation设计
- 将该范式应用于其他属性控制（如风格、领域、情感）

--- 

> **代码开源地址**：https://github.com/HungsingWong/sae-language-steering

</details>

---

### 14. [SparseDitto: Customizing GPU Kernels for Different Sparsity Patterns with LLM-Based Agentic System](https://arxiv.org/abs/2608.05033)

**Authors**: Shiyang Li, Guangyan Sun, Jinwei Tang, Yanzhi Wang, Mingyi Hong, Caiwen Ding  
**Category**: cs.DC  
**Published**: 2026-08-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.05033v1  

#### Abstract
Sparse matrix kernels are fundamental to scientific computing, graph analytics, and machine learning. Their GPU performance depends strongly on the input sparsity pattern and execution strategy. For the same SpMM on the same matrix, cuSPARSE exhibits a 350x performance gap between CSR and Blocked-EL...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SparseDitto: Customizing GPU Kernels for Different Sparsity Patterns with LLM-Based Agentic System**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **稀疏矩阵计算在 GPU 上的性能高度依赖于 sparsity pattern（稀疏模式）、算子类型（如 SpMV、SpMM、SpGEMM）和硬件架构**。现有的稀疏库（如 cuSPARSE）或专用系统（如 CB-SpMV、DTC-SpMM）往往基于固定假设设计，无法适应不同输入和硬件的变化。
- 实验表明，在相同矩阵和算子下，cuSPARSE 在不同格式（如 CSR vs Blocked-ELL）之间可产生高达 **350× 的性能差距**，说明“一刀切”的实现方式存在严重性能悬崖（performance cliffs）。

### **提出的新方法与思路**
- 提出 **SparseDitto**：一个基于 **LLM-based agentic system** 的自动化稀疏 GPU kernel 生成框架。
- 核心思想是将 workload 视为数据，通过结构分析、策略排序、架构感知规划、代码生成与验证闭环优化，为每个矩阵、算子和目标 GPU 定制最优 kernel。

### **创新点**
1. **统一的设计框架（Unified Design Framework）**
   - 支持 **SpMV、SpMM 和 SpGEMM** 三大稀疏算子，首次在一个系统中实现跨算子的动态 kernel 生成。
   - 将 kernel 设计形式化为三元组：`II = (R, S, OH)`，其中：
     - `R`: 数据表示（layout + metadata）
     - `S`: 执行调度（parallel decomposition + dataflow）
     - `OH`: 硬件映射参数（tile size, shared memory, etc.）

2. **结构引导 + 架构感知的搜索机制**
   - 提取 **36 个结构性特征**（包括 intrinsic pattern、representation-induced、operator-induced 和硬件上下文），用于刻画稀疏模式。
   - 使用 **可解释的 additive energy model** 对已有优化策略进行排序，作为搜索起点。
   - 引入 **层级式 planner**，结合目标 GPU 的资源约束（SM 数量、寄存器、shared memory 等）生成合法候选方案。

3. **端到端自动化生成与迭代优化**
   - 利用 LLM 代理（coding agent）自动生成 CUDA kernel。
   - 验证代理（verification agent）执行正确性检查与目标 GPU 上的实际 profiling。
   - 基于运行时反馈驱动迭代 refine，确保最终选择的是实测最快的 kernel，而非模型置信度最高的。

### **相比现有方法的优势**
| 方法 | 局限性 | SparseDitto 的优势 |
|------|--------|------------------|
| **cuSPARSE** | 固定实现，易出现性能悬崖 | 动态适配，避免不匹配设计 |
| **CB-SpMV / DTC-SpMM / HSMU-SpGEMM** | 仅适用于特定结构或算子，跨 GPU 迁移差 | 跨算子支持，自动适应新硬件 |
| **SparseTIR 等稀疏编译器** | 依赖预定义规则集，灵活性有限 | 可探索超出 selector 词汇表的新策略 |

---

## **2. 核心实验方法和设置**

### **数据集**
- 使用 **60 个来自 SuiteSparse Matrix Collection [5]** 的矩阵，涵盖多种领域：
  - 科学计算（finite-element）
  - 图分析（web graphs, road networks）
  - 生物信息（gene networks）
  - 电路仿真等
- 矩阵规模跨度大：行数从 359 到 16M，非零元数量从 817 到 1.01 亿，密度范围 $1.78 \times 10^{-7}$ 到 0.110。

### **任务设置**
对每个矩阵评估以下三种操作：
- **SpMV**: $y = Ax$
- **SpMM**: $C = AB$, 其中 $B$ 是宽分别为 $K = \{8, 32, 128, 256\}$ 的稠密矩阵
- **SpGEMM**: $C = AA$（方阵）或 $C = AA^T$（矩形）

### **实验平台**
- 主要测试平台：
  - **NVIDIA RTX PRO 6000 (Blackwell)** + CUDA 13.0
  - **NVIDIA H200 (Hopper)** + CUDA 12.9
- 所有 kernel 使用 CUDA events 测量延迟，报告 **10 次 warm-up 后的平均值**。

### **评估指标**
- **几何平均加速比（Geometric-mean speedup）** over cuSPARSE
- 最大加速比（Max speedup）
- 正确性验证：输出满足混合误差界 $\| \hat{y} - y \|_\infty \leq \epsilon_{abs} + \epsilon_{rel} \cdot \| y \|_\infty$，其中 $\epsilon_{abs}=10^{-5}, \epsilon_{rel}=10^{-3}$
- 消融实验、缓存命中率、内存流量、occupancy 等辅助分析

### **基线方法对比**
| 算子 | 基线方法 |
|------|---------|
| SpMV | cuSPARSE, CB-SpMV [4], AlphaSparse [6] |
| SpMM | cuSPARSE, DTC-SpMM [7], SparseTIR [34] |
| SpGEMM | cuSPARSE, HSMU-SpGEMM [33] |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 平台 | SpMV | SpMM (avg) | SpGEMM | Overall Geomean |
|------|------|------------|--------|----------------|
| **RTX PRO 6000** | 2.53× | 2.72× | 3.70× | **2.68×** |
| **H200** | 2.68× | 2.79× | 4.09× | **2.79×** |
- **最大加速比达 146.61×（SpGEMM）和 78.5×（H200）**
- 在 **94% 的测试案例中优于 cuSPARSE**

### **与基线方法的对比结果**
| 对比对象 | 加速比（vs. cuSPARSE） | SparseDitto 超越幅度 |
|--------|----------------------|--------------------|
| **CB-SpMV (SpMV)** | 1.35× ~ 1.70× | **1.91× 更快** |
| **DTC-SpMM (SpMM)** | < 1×（在 Blackwell 上全低于 cuSPARSE） | **3.99× 更快** |
| **SparseTIR (SpMM)** | 1.18× ~ 1.68× | **1.57× 更快** |
| **HSMU-SpGEMM (SpGEMM)** | 0.32× ~ 0.74× | **5.37× 更快**，且支持其不支持的矩形输入 |

> ✅ 特别值得注意的是：**DTC-SpMM 在新的 Blackwell 架构上表现劣于 cuSPARSE**，而 SparseDitto 自动适配新硬件仍保持显著优势。

### **消融实验结果（Ablation Study）**
| 配置 | SpMV | SpMM | SpGEMM |
|------|------|------|--------|
| Task only（无 pattern 分析） | 1.93× | 2.02× | 1.22× |
| + Pattern analysis | 2.27× | 2.28× | 2.24× |
| + Architecture-aware planning (**SparseDitto 全量**) | **2.53×** | **2.52×** | **3.70×** |

> 结论：**pattern 分析和 architecture-aware planning 均带来显著提升**，尤其对 SpGEMM 影响最大。

### **超越 selector 词汇表的能力**
- 在 **36% 的任务中（127/350）**，最终胜出的 kernel 实现了 **不在 selector 预定义策略集合 $V_0$ 中的新策略**。
- 这些“意外创新”任务的平均加速比更高：
  - **Outside $V_0$**: 3.11×
  - **Inside $V_0$**: 2.46×

> 示例：在 `e40r0100` 矩阵上，agent 自动生成了一种基于 **dense value stripe + generation tag** 的 accumulator，无需原子操作，达到 **8.47×** 加速。

---

## **4. 关键结论和发现**

### **主要发现**
1. **稀疏 kernel 性能极度敏感于 sparsity pattern 与实现之间的匹配程度**，错误匹配可能导致 **数百倍性能损失**。
2. **没有单一实现能在所有 pattern、operator 和 GPU 上持续领先**，必须采用动态定制化策略。
3. **SparseDitto 成功实现了跨算子、跨硬件、跨 pattern 的统一优化框架**，首次做到：
   - 统一支持 SpMV / SpMM / SpGEMM
   - 动态生成并验证 kernel
   - 实测驱动而非模型置信度驱动决策
4. **LLM agents 不仅能复现已有技术，还能创造性地组合机制，发现新策略**，尤其是在中间产物分布特殊的 SpGEMM 场景中。

### **方法的局限性**
- **生成时间开销较大**：每次 kernel 生成需多次 compile-benchmark 循环（约几秒至几十秒），不适合实时极低延迟场景。
- **依赖高质量的 LLM 模型**：当前使用 GPT-5.6-terra，若换成较小模型可能影响生成质量。
- **目前仅针对 NVIDIA GPU**，尚未扩展至 AMD 或其他架构。
- **特征提取虽轻量，但仍需预处理扫描**，对于频繁变化的小矩阵可能性价比不高。

### **未来工作方向**
- 探索 **更高效的搜索策略**（如贝叶斯优化 + LLM guidance）
- 构建 **kernel 缓存机制**，实现跨任务重用（已初步支持按 `(pattern, op, GPU, K)` 缓存）
- 扩展至更多算子（如 SpMSpV、稀疏卷积）
- 支持多 GPU 和分布式稀疏计算
- 探索 **smaller LLM + retrieval-augmented generation (RAG)** 降低部署成本

---

> 📌 **一句话总结**：  
> **SparseDitto 是首个利用 LLM agent 系统为任意稀疏矩阵、任意算子和任意 GPU 自动生成高性能定制化 kernel 的框架，在多个平台上平均提速 2.7×，最高达 146×，并展现出超越人类先验知识的创新能力。**

</details>

---

### 15. [Learning Compression Rules for Network Traffic](https://arxiv.org/abs/2608.04545)

**Authors**: Quentin Lampin (Orange Research), \'Eloi Sainte-Beuve (Orange Research, Universit\'e Grenoble Alpes), Louis-Adrien Dufr\`ene (Orange Research), Guillaume Larue (Orange Research), Massih-Reza Amini (Universit\'e Grenoble Alpes)  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.04545v1  

#### Abstract
We study the problem of learning compact rule-based compressors for structured network traffic. Each packet is a record of header fields that are highly redundant within a flow, and a compressor is a small set of rules matching such records and replacing predictable fields with short codes. We cast ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning Compression Rules for Network Traffic*

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**网络流量头部压缩规则的手动设计难题**。在当前主流的规则型压缩标准（如 **SCHC**）中，高效的压缩依赖于专家手工编写的规则集，这些规则需结合协议栈知识和实际流量分析，过程繁琐、难以维护且不适用于新型或异构协议（如 5G 控制面）。该问题导致部署成本高、适应性差。

### 提出的新方法：RECAP
作者提出 **RECAP (Robust Entropy Clustering for Adaptive comPression)**，一种**全自动学习紧凑规则集**的方法，其核心思想是将规则学习建模为两阶段优化问题：

1. **无监督结构发现（Packet Partitioning）**  
   采用**基于归一化熵比（normalized entropy-ratio）的自顶向下分裂聚类算法**，递归地将训练包划分为候选簇。该准则能有效应对小样本场景下的统计偏差，避免因样本稀疏而误判字段可压缩性。

2. **带约束的规则选择（Constrained Rule Selection）**  
   将每个簇转化为一个候选 SCHC 规则，并利用**动态规划（Dynamic Programming）** 在给定规则数量预算 $N$ 下，选择期望压缩增益最大的规则子集。此过程考虑了规则间的层次关系，避免冗余选择。

### 相比现有方法的优势
- **自动化替代人工**：完全摆脱对领域专家的依赖，实现从数据到规则的端到端生成。
- **标准化兼容**：输出为标准 **SCHC 兼容规则**，可直接集成进任何 RFC 8724 协议栈，无需修改运行时架构。
- **高效且鲁棒**：在极小规则数（如 $N=8$）下即可达到甚至超越专家规则性能；归一化熵比和 Good-Turing 覆盖估计使其在小样本和异构流量中仍表现稳健。
- **优于其他学习方法**：
  - 相比 [Banerjee et al.] 的扁平聚类方法，RECAP 的分层结构避免了在异构流量中规则预算被碎片化。
  - 相比 [Meslet-Millet et al.] 的端到端深度学习方法 **DCH**，RECAP 不需要运行时推理模型，更轻量、可解释性强且符合 IETF 标准框架。

---

## 2. 核心实验方法和设置

### 数据集
实验在四个真实世界数据集上进行，涵盖典型与非典型 SCHC 应用场景：

| Dataset         | 类型               | 协议栈                     | 包数量   | 特点 |
|------------------|--------------------|----------------------------|----------|------|
| **Balloon-20k**  | IoT / Telemetry    | IPv6/UDP/CoAP              | 20,000   | 上行周期性遥测，结构简单 |
| **Thermostat-10k** | IoT / Request-Response | IPv6/UDP/CoAP           | 10,000   | 请求/响应混合，行为多样 |
| **GTP-traffic**  | 5G Core (Small Sample) | IPv4/UDP/GTPv1           | 100      | 极小样本（仅10个训练包），挑战大 |
| **NGAP-traffic** | 5G Core (Heterogeneous) | IPv4/SCTP/NGAP (ASN.1) | 15,650   | 多种信令流程，结构高度异构 |

> 注：所有数据集均按时间顺序划分，前 $r\%$ 用于训练，其余用于测试，模拟真实部署场景。

### 评估指标
- **Compression Ratio (%)**: $(1 - C_{\text{compressed}} / C_{\text{original}}) \times 100\%$
  - $C_{\text{original}}$: 原始包头总比特数
  - $C_{\text{compressed}}$: 压缩后（Rule ID + Residue）总大小
- **Header Ratio (%)**: $(1 - \text{payload\_bits}/\text{total\_bits}) \times 100\%$，理论压缩上限。

### 基线方法
- **IoT 数据集**：采用 **RFC 8824** 中定义的标准 CoAP SCHC 配置文件作为专家基线。
- **5G 数据集**：构建**结构化基线（structural baseline）**，仅基于公开协议规范（RFC/3GPP）提取传输层头部结构，忽略深层 ASN.1 编码字段，模拟“无迹特知识”的专家做法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Test Set, 10% Training Split）
| Dataset         | RECAP ($N=8$) | Expert Baseline | Header Ratio | 备注 |
|------------------|---------------|------------------|--------------|------|
| Balloon-20k     | **83.4%**     | 83.6% (N=12)     | 92.8%        | RECAP 在 $N=12$ 达 84.2%，**超越专家** |
| Thermostat-10k  | **78.1%**     | 75.6% (N=9)      | 86.8%        | 明显优于专家基线 |
| GTP-traffic     | 23.6%         | **25.4%** (N=5)  | 100.0%       | 小样本下略低于结构基线，但随训练数据增加反超 |
| NGAP-traffic    | **13.6%**     | 10.7% (N=2)      | 100.0%       | 即使最小预算也**显著优于**基线，$N=32$ 达 53.4% |

> ✅ **结论**：RECAP 用少量规则（$\leq 8$）即能在 IoT 场景匹配或超越专家规则，在复杂 5G 场景下大幅领先。

### 与基线方法对比
- **IoT 场景**：RECAP 在 **Balloon-20k** 和 **Thermostat-10k** 上分别以更少或相当的规则数实现了与专家规则持平甚至更高的压缩率。
- **5G 场景**：
  - **GTP-traffic**：由于训练样本极少（仅10包），RECAP 初期略逊于利用先验知识的结构基线，但在 20% 以上训练数据时全面反超（最高达 36.8%）。
  - **NGAP-traffic**：RECAP 在所有预算下均**远超结构基线**，证明其能有效挖掘深层 ASN.1 结构中的冗余。

### 消融实验结果（Ablation Studies）
| 组件替换                          | 影响说明 |
|-----------------------------------|--------|
| **Greedy Top-k 替代 DP**          | 在浅层树中差异小（<0.2 pp），但在深层异构结构中 DP 更优，避免父子规则冗余。 |
| **Raw Entropy 替代 Normalized Entropy-Ratio** | 在 Balloon-20k 上 $N=32$ 时差距达 1.1 pp，验证归一化对字段宽度和样本大小的校正至关重要。 |
| **Flat Clustering [Banerjee et al.]** | 在 IoT 数据上接近 RECAP，但在 GTP 和 NGAP 上崩溃（负压缩），因预算被碎片化至无泛化能力的小簇。 |
| **Uniform Coverage ($C(u)=1$) 替代 Good-Turing** | 在 GTP 上压缩率从 23.6% 暴跌至 2.3%，证明保守覆盖估计对防止过拟合训练噪声极为关键。 |

---

## 4. 关键结论和发现

### 主要发现
1. **数据驱动规则学习可行且高效**：RECAP 可自动从流量中学习高质量 SCHC 规则，**消除手动工程瓶颈**。
2. **少量规则即可获得高性能**：仅需 8–16 条规则即可在多种场景下逼近理论压缩极限。
3. **分层结构 + 动态规划是关键**：相比扁平聚类，分层分裂能更好组织候选规则；DP 能最优分配有限规则预算。
4. **统计稳健性设计至关重要**：归一化熵比和 Good-Turing 覆盖估计使方法在**小样本**和**异构流量**中依然可靠。

### 方法的局限性
- 当前框架仅实例化并评估于 **SCHC**，虽具通用性，但未在其他压缩机制（如 ROHC）中实证。
- 性能依赖两个超参数（$\theta=0.95$, $M_{\text{map}}=8$），尽管敏感性分析显示默认值已处于饱和平台。
- 时间复杂度为 $O(|V| \cdot N^2)$，虽可在单机运行，但对超大规模流量可能需优化。

### 未来工作方向
- 将学习流水线扩展至非 SCHC 压缩框架（如 ROHC）。
- 支持更多协议族（如 QUIC）。
- 探索在线或设备端自适应学习机制。
- 利用 RECAP 学习到的流量特征（如恒定字段、低基数字段）初始化 ROHC 上下文，提升其收敛速度。

</details>

---

### 16. [BnBERT-iPET: Sparse Few-Shot Language Modeling for Bengali via Lottery Ticket Pruning](https://arxiv.org/abs/2608.05104)

**Authors**: Sajib Hossain, Md Kamrus Samad, Anan Ghosh, Labib Imam Chowdhury, Nabeel Mohammed  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.05104v1  

#### Abstract
Deep neural networks have shown impressive success in NLP tasks owing to their complex structure and huge number of edges. Achieving state-of-the-art performance in natural language processing with a large pre-trained model such as BERT is expensive and time-consuming, carries a large carbon footpri...

---

### 17. [The RAIL Principles for Neurosymbolic AI: Reasoning, Assurances, Interfacing and Learning](https://arxiv.org/abs/2608.04285)

**Authors**: Agnese Chiatti, Michael Cochez, Cristina Cornelio, Sebastijan Dumancic, Artur d'Avila Garcez, Luis C. Lamb, Lia Morra, Mathias Niepert, Robert Peharz, Alberto Speranzon, Maarten Stol, Annette Ten Teije, Thiviyan Thanapalasingam, Frank Van Harmelen, Emile Van Krieken, Antonio Vergari, Benjie Wang  
**Category**: cs.AI  
**Published**: 2026-08-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.04285v1  

#### Abstract
Neurosymbolic AI systems that integrate machine learning and symbolic reasoning are rapidly gaining attention. They complement the data-intensive statistical approaches of neural networks and language models with symbolic reasoning algorithms to function in high-stakes domains or in low-data regimes...

---

### 18. [Architectural Implications of Agentic AI Workflows](https://arxiv.org/abs/2608.04458)

**Authors**: Jirong Yang, Peizhe Liu, Chaojie Zhang, Jovan Stojkovic  
**Category**: cs.AI  
**Published**: 2026-08-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.04458v1  

#### Abstract
Agentic AI is emerging in datacenters, but its architectural implications remain unexplored. We organize agentic workflows in a taxonomy and present its first architectural characterization with a production study at Microsoft Azure and a controlled study of open-source frameworks. We show that agen...

---

### 19. [Zero-Instrumentation Dependency Discovery for Guided Microservice Migration Using eBPF](https://arxiv.org/abs/2608.04413)

**Authors**: Eshan Trivedi, Chandrahasa Pranava  
**Category**: cs.DC  
**Published**: 2026-08-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.04413v1  

#### Abstract
Migrating microservices across virtual machines (VMs) without knowledge of their runtime communication patterns risks creating cross-VM hotspots and latency spikes that are difficult to predict from static analysis alone. We use extended Berkeley Packet Filter (eBPF) kernel-level network tracing to ...

---

### 20. [An Explainable LLM Agent Layer for Open-World Anomaly Detection in Oil Wells](https://arxiv.org/abs/2608.04041)

**Authors**: Lucas Gouveia Omena Lopes, Thales Miranda de Almeida Vieira, Eduardo Toledo de Lima Junior, William Wagner Matos Lira  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.04041v1  

#### Abstract
Open-World Learning (OWL) pipelines for oil well anomaly detection have recently been shown to combine autoencoder-based detection, multiclass classification, and Mahalanobis-based novelty detection on the public 3W dataset. These pipelines answer \textit{what happened}, but they do not explain \tex...

---

### 21. [From Non-Convex Self-Concordant Regularization to Scalable Quasi-Newton Training of PINNs](https://arxiv.org/abs/2608.04206)

**Authors**: Chenhao Si, Kang An, Shiqian Ma, Ming Yan  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.04206v1  

#### Abstract
Physics-informed neural networks (PINNs) often require high-accuracy quasi-Newton refinement to obtain reliable partial differential equation solutions, but their residual objectives can exhibit indefinite, nearly singular, and poorly scaled local curvature. Regularized quasi-Newton methods provide ...

---

### 22. [Robustness Emerges Early in Training Dynamics, but Is Not Preserved](https://arxiv.org/abs/2608.04442)

**Authors**: Jiangang Yang, Wenhui Shi, Lu Hu, Jing Xing, Jian Liu  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.04442v1  

#### Abstract
Robustness to natural corruptions remains a fundamental challenge for deep neural networks. In this paper, we identify a robustness fading phenomenon where shallow layers spontaneously develop robust representations and flat loss landscapes in early training, yet these properties are not preserved d...

---

### 23. [Fewer Tokens, Smaller Cache: Reward-Coordinated Efficient Reasoning](https://arxiv.org/abs/2608.04771)

**Authors**: Qiyuan Zhu, Dezhi Li, Pengyu Cheng, Tianle Chen, Jiacheng Wang, Ruijie Shen, Hao Gu, Sida Lin, Zirui Liu, Jiacheng Liu, Sirui Han  
**Category**: cs.AI  
**Published**: 2026-08-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.04771v1  

#### Abstract
Large Reasoning Models (LRMs) excel on complex tasks through long chain-of-thought (CoT) reasoning, but their lengthy intermediate steps cause severe overthinking that inflates inference cost. KV-cache compression is a common solution, yet existing reasoning-oriented methods apply a uniform policy a...

---

### 24. [Hierarchical Graph Memory for LLM Agents with Path-level Localization and Rewrite](https://arxiv.org/abs/2608.05095)

**Authors**: Xiawei Yue, Boran Wang, Xiaoqing Zhang, Shuxin Zheng, Ziwei Zhang  
**Category**: cs.AI  
**Published**: 2026-08-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.05095v1  

#### Abstract
Agents for long term reasoning require a memory that can be efficiently and effectively updated over time, as new facts and external feedback continue to arrive. Recently, graph memory has been adopted to offer structural organization for multi-hop retrieval and reasoning. However, existing methods ...

---

### 25. [ABSeeker: Training Long-Horizon Search Agents via Answer-Backtracked Credit Assignment](https://arxiv.org/abs/2608.05102)

**Authors**: Yijun Lu, Rui Ye, Jiajun Wang, Yuwen Du, Tian Jin, Songhua Liu, Siheng Chen  
**Category**: cs.AI  
**Published**: 2026-08-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.05102v1  

#### Abstract
Long-horizon search agents must make multiple sequential actions (steps) to search, retrieve, verify, and integrate evidence to reach a final answer. However, existing methods for training these agents typically treat all steps within a trajectory uniformly during both supervised fine-tuning (SFT) a...

---

### 26. [Learning Sexism Detection Using Multi-Agent Perspectivist Preference Optimization](https://arxiv.org/abs/2608.04056)

**Authors**: Hadi Mohammadi, Tina Shahedi, Robert A. Bagheri, Mehdi Dastani, Masoume M. Raeissi  
**Category**: cs.CL  
**Published**: 2026-08-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.04056v1  

#### Abstract
When people label text for sexism, they often disagree, and not because some of them are wrong: they genuinely perceive sexism differently. Most NLP systems discard this disagreement by collapsing it into a majority vote. We propose the Multi-Agent Perspectivist Preference Optimization (MAP-PO) fram...

---

### 27. [Eliciting Intrinsic Hallucinations in LLMs via Semantically Equivalent Adversarial Attacks](https://arxiv.org/abs/2608.04286)

**Authors**: Atri Vivek Sharma, Brian Formento, Alessio Lomuscio  
**Category**: cs.CL  
**Published**: 2026-08-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.04286v1  

#### Abstract
Large language models (LLMs) are often used in conjunction with external knowledge sources to improve their factual accuracy and decrease hallucinations, through methods such as Retrieval-Augmented Generation (RAG). However, these systems remain susceptible to intrinsic hallucinations, where the mod...

---

### 28. [Kathleen Writes: Autoregressive Generation and Data Scaling Without Attention](https://arxiv.org/abs/2608.04678)

**Authors**: George Fountzoulas  
**Category**: cs.CL  
**Published**: 2026-08-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.04678v1  

#### Abstract
Papers 1-2 of the Kathleen series showed that a byte-level, attention-free architecture built from a wavetable encoder and multi-scale reverberant state can match strong baselines on classification at ~450-700K parameters, without pretraining. We ask whether the same ingredients can generate. (1) Sc...

---

### 29. [Geometry-Informed Parameter-Efficient Fine-Tuning of Pre-trained Molecular GNNs for Blood-Brain Barrier Permeability Prediction](https://arxiv.org/abs/2608.04257)

**Authors**: Marco Vieto Vega, Long D. Nguyen, Binh P. Nguyen  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.04257v1  

#### Abstract
Blood-brain barrier permeability (BBBP) prediction is a critical screening task in central nervous system drug discovery, where candidate molecules must be assessed for whether they can cross, or should be prevented from crossing, the blood-brain barrier. However, this task remains challenging becau...

---

### 30. [Adaptive Finite-Budget Training for CVaR Risk-Aware Q-Learning](https://arxiv.org/abs/2608.04305)

**Authors**: Yifan Wu, Junjie Lei, Wenjie Huang  
**Category**: cs.LG  
**Published**: 2026-08-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.04305v1  

#### Abstract
Risk-aware Q-learning (RaQL) provides a model-free, two-timescale estimator for dynamic risk objectives, but its finite-budget behavior remains fragile: fixed inner-loop hyperparameters can produce unstable value estimates, persistent Bellman residuals, and inefficient sample reuse. This paper propo...

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

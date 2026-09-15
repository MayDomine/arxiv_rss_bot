# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-15 10:29:45 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DeepSeek-V4-Flash on AMD gfx90a: Correctness Recovery and Inference Performance Engineering](https://arxiv.org/abs/2609.15627)

**Authors**: Siming Huang  
**Category**: cs.DC  
**Published**: 2026-09-15  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2609.15627v1  

#### Abstract
We present the enablement, correctness recovery, and performance engineering of DeepSeek-V4-Flash inference on AMD Instinct MI250 GPUs using the gfx90a/CDNA2 architecture. The system integrates native safetensors loading, tensor and expert parallelism, FP4 routed mixture-of-experts computation, FP8 ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# DeepSeek-V4-Flash on AMD gfx90a: Correctness Recovery and Inference Performance Engineering  
**论文核心总结**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究解决了在 **AMD Instinct MI250 (gfx90a/CDNA2)** 架构上高效部署 **DeepSeek-V4-Flash** 大语言模型（LLM）推理所面临的关键挑战，主要包括：
- **数值正确性问题**：原始 FP4 权重路径在 gfx90a 上执行时出现输出错乱（如 W2 permutation 错误），导致生成结果不一致。
- **性能瓶颈**：小批量（small-M）、高并发场景下的 decode 效率低下，尤其是 MoE 层、attention 和 FP4 解包等操作未能充分利用 CDNA2 架构特性。
- **冗余计算**：KV-cache 中存在大量空 tile 的无效计算，浪费算力。

### 提出的新方法与创新思路
1. **W2 输出列置换修复（W2 Permutation Fix）**
   - 发现并定位了 FP4 MoE 路由中 `W2` 权重矩阵因硬件布局导致的输出顺序错误。
   - 在加载阶段对权重应用逆向排列 `[0,4,1,5,2,6,3,7]`，实现 **one-time load transformation**，无运行时开销，恢复逻辑一致性。

2. **空索引 tile 消除优化（Empty Tile Guard）**
   - 引入精确的空 tile 判断机制，在 native decode 阶段跳过无有效 key 的 query 计算。
   - 显著减少冗余 GEMM 和 dot product 开销，尤其在长上下文但短前缀激活的场景下效果显著。

3. **C1 投影特化恢复（C1 Projection Specialization）**
   - 恢复并验证了针对单请求（C1）场景的 `wo_a GEMV` 优化路径，提升小并发性能。

4. **默认关闭的 down-consumer 试点优化**
   - 在 C32 场景下引入 CTA16 consumer 结构，优化第二量化链路（quant+down+reduction），带来额外吞吐增益。

5. **端到端原生自回归（native AR）验证框架**
   - 区分 **native AR**、**speculative decoding (DSpark)** 和 **approximate verification**，强调使用固定输入哈希、语义哨兵（France sentinel）进行严格数值验证。

### 相比现有方法的优势
| 维度 | 本文方法 | 传统做法 |
|------|--------|---------|
| **正确性保障** | 多层级验证（kernel → layer → model → attribution） | 单一哈希或输出比对 |
| **执行路径** | 保留原始 FP4/FP8 checkpoint，动态 unpack | 离线转为 INT8 或 BF16 |
| **优化粒度** | 工作负载感知的细粒度 kernel 路径选择（shape-specific HIP/AIter/CK/Triton） | 通用 kernel 库直接调用 |
| **测量标准** | 分离 resident decode、HTTP latency、prefill throughput，避免混淆 | 混合报告 speculative 吞吐或冷启动延迟 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **非公开训练数据**：未使用传统 NLP 数据集（如 GLUE、MMLU）。
- **测试请求来源**：来自 **public-source code requests**（公共源码请求），具体为代码补全类任务。
- **固定验证输入**：
  - “The capital of France is” → 验证输出是否为 “**Paris**”，token ID 序列为 `[671,6102,294,8760,344,2619,51119,42499,1]`，hash 为 `6f41fe2f01d52507`。
- **ABBA 测试协议**：用于对比控制组与候选组的服务级性能差异（A1→B1→B2→A2）。

### 实验设置
| 项目 | 配置 |
|------|------|
| **硬件平台** | Supermicro A+ Server AS-4124GQ-TNMI<br>双 AMD EPYC 7763 CPU，1TB DDR4 内存<br>8× MI250 GCDs (gfx90a)，每 GCD ~64GiB HBM |
| **软件栈** | Python 3.12.13, PyTorch 2.12.0a0, Triton 3.7.1+, Transformers 5.12.1<br>HIP 7.15.26333, AMD clang 23<br>SGLang + RadixAttention + Continuous Batching |
| **模型配置** | DeepSeek-V4-Flash<br>284B 总参数，~13B 激活参数<br>FP4 专家权重 + FP8 非专家权重<br>Top-6 路由，256 个 routed experts，shared experts<br>支持 up to 1M token context |
| **并行策略** | **TP8/EP1**：8-GCD tensor parallel，expert-parallel=1，无 A2A 通信 |
| **KV Cache** | 逻辑池大小：**1,048,576 tokens**（约 1M）<br>Page size: 256 |

### 评估指标
| 指标 | 定义 |
|------|------|
| **Resident Output tok/s** | 排除 admission 和 drain 阶段的纯 decode 吞吐量，反映稳态性能 |
| **HTTP Output tok/s** | 完整请求生命周期（含 prefill、admission、drain）的整体吞吐 |
| **Prefill Input tok/s** | 所有 prompt token 数总和 / 波次时间（start to last first-token） |
| **Three-round Median** | 每个并发等级（C）运行三轮，取 median 报告，保留 outlier |
| **ABBA Ratio** | `(B1+B2)/2 / (A1+A2)/2`，用于衡量优化前后相对性能变化 |

### 基线方法对比
- **历史 TP4 基线**：早期四 GCD 实验，非本次主测对象。
- **DSpark speculative decoding**：作为独立证据保留，**不与 native AR 直接比较**。
- **CKTile BF16-by-FP4 vs 原始 FP4-by-FP4**：用于验证数值修复有效性。
- **Peer-read all-reduce vs RCCL**：替代高延迟集合通信。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 2）

| Concurrency (C) | Prefill Input tok/s | Resident Output tok/s | HTTP Output tok/s |
|------------------|---------------------|------------------------|--------------------|
| 1                | 4,676.39            | **87.60**              | 86.39             |
| 2                | 4,989.25            | 109.94                 | 102.41            |
| 4                | 5,265.43            | 188.71                 | 148.50            |
| 8                | 5,099.23            | 334.18                 | 254.45            |
| 16               | 5,171.36            | 608.23                 | 422.51            |
| 32               | 5,254.98            | **1,044.32**           | 680.61            |
| 64               | 5,250.05            | **1,334.24**           | 848.14            |

> ✅ **Prefill 吞吐稳定在 ~5.2k input tok/s**  
> ✅ **Decode 吞吐持续扩展至 C64，达 1,334.24 resident tok/s**

### 与基线方法的对比结果
| 对比项 | 结果 |
|-------|------|
| **Empty Tile Guard (C32, 8K input)** | 3.60× resident decode 提升（169.66 → 611.10 tok/s） |
| **Restored C1 GEMV Specialization** | +10.32% 提升（79.56 → 87.58 tok/s） |
| **Down-consumer Pilot (C32)** | +1.54% resident tok/s 提升（1044.61 → 1060.70 tok/s） |
| **vs 历史 ~60 tok/s FP4 route** | 修复前错误路径性能仅为 ~60 tok/s，且数值不正确 |
| **vs DSpark speculative (~1.5k tok/s)** | 不可比！该结果为 **approximate-target**，仅执行部分专家分支 |

### 消融实验结果
| 优化项 | 性能影响 | 是否集成 |
|--------|--------|----------|
| **Empty C4 Tile Skip** | C32 下提速 3.6×（8K input） | ✅ 是 |
| **C1 wo_a GEMV Restoration** | +10.32% C1 性能 | ✅ 是 |
| **CTA16 Down Consumer (C32)** | +1.54% resident tok/s | ⚠️ 默认关闭（opt-in） |
| **Offline FP4-to-INT8 Expansion** | Gate kernel 从 7.23ms → 10.45ms（变慢） | ❌ 拒绝 |
| **CKTile KSPLIT=2/4** | 小 batch 下性能下降 | ❌ 不适用 |
| **RCCL 替代 Peer-read** | TTFT 从 0.679s → 0.711s（更差） | ❌ 保留 peer-read |

---

## 4. 关键结论和发现

### 主要发现
1. **FP4 存储 ≠ FP4 计算**：CDNA2 缺乏原生 FP4 matrix core，必须通过 INT8 dot 或 custom HIP kernel 实现，**存储格式与执行路径解耦**。
2. **小 M MoE 是性能关键路径**：decode 阶段的 per-expert 小矩阵运算主导延迟，需定制化 kernel（如 wave64 GEMV）。
3. **冗余计算代价高昂**：即使少量空 tile 的无效计算也会严重拖累性能（如 3.6× 差距）。
4. **numerical correctness 不可妥协**：早期看似“高性能”的 ~60 tok/s 路径实则输出错误，凸显 **end-to-end 验证** 必要性。
5. **resident throughput ≠ HTTP throughput**：C64 下 resident 达 1,334 tok/s，但完整 HTTP 仅 848 tok/s，差距来自 admission、prefill 和 drain。

### 方法的局限性
| 限制 | 说明 |
|------|------|
| **未覆盖百万级满载场景** | 当前测试未填满 1M KV pool，缺乏满 Occupancy 下的质量与稳定性认证 |
| **cold-shape 编译未预热** | 新形状首次触发编译可达 20s+，影响首请求延迟（TTFT） |
| **atomic reduction 非确定性** | CK 中 FP32 atomic accumulation 导致 reduction order 不一致 |
| **缺乏通用性声明** | 优化高度依赖 gfx90a 架构特征（wave64, LDS, XGMI），不可直接迁移至其他 GPU |
| **未实现全局 bitwise 等价** | 动态批处理、冷热启动等因素使完全数值复现困难 |

### 未来工作方向
1. **Fill & Verify 1M Context**：构造可控长上下文测试集，验证满池下的质量与吞吐。
2. **Cold-Shape 预编译**：建立 shape catalog 并预热常见 M/N/K 组合，消除 JIT 延迟。
3. **Indexer Work 优化**：探索 active-query production，避免 full logits materialization。
4. **Fixed-Reduction Experiments**：尝试 fixed-slot CK stage-2 reduction 以提高数值稳定性。
5. **Matched TP4 vs TP8 对比实验**：补全 tensor parallel scaling 效率分析。
6. **Strict DSpark 回归测试**：开展与 native AR 同条件下的 speculative decoding 性能对比。

---

> 📌 **最终交付物不是“最高吞吐”声明，而是一个可审计的配置与证据档案（auditable configuration and evidence archive）**。  
> 本文强调：**区分 native AR、strict speculation、approximate target、HTTP latency 和 resident throughput** 是构建可信 LLM 推理系统的基础。

</details>

---

### 2. [Communication-Efficient LLM Adaptation over Decentralized GPU Meshes](https://arxiv.org/abs/2609.14339)

**Authors**: Sameera Ramasinghe, Shamane Siriwardhana, Thalaiyasingam Ajanthan, Hadi Mohaghegh Dolatabadi, Chamin P Hewa Koneputugodage, Gil Avraham, Violetta Shevchenko, James Snewin, Karol Pajak, Harry Xi, Alexander Long  
**Category**: cs.LG  
**Published**: 2026-09-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.14339v1  

#### Abstract
Decentralized training enables large-model training over low-end GPUs and internet-grade connections, but communication along both data-parallel and pipeline-parallel axes becomes the primary bottleneck. We study post-pretraining adaptation in this setting. We propose an asynchronous two-circuit sys...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Communication-Efficient LLM Adaptation over Decentralized GPU Meshes

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大模型训练依赖高性能数据中心集群，而小型研究机构、独立研究人员和开源社区通常只能访问低带宽互联网连接和消费级GPU。在**去中心化训练**场景下，**通信瓶颈**成为主要限制因素，尤其是在**数据并行（DP）** 和 **流水线并行（PP）** 维度上的梯度与激活传输。

本文聚焦于**后预训练适应（post-pretraining adaptation）**，即在去中心化的低带宽GPU网格上进行 **fine-tuning 或 continual pretraining**，目标是大幅降低通信开销而不牺牲模型性能。

---

### 提出的新方法与创新思路

作者提出了一种**异步双回路系统（asynchronous two-circuit system）**，结合以下关键技术：

#### （1）**双回路架构（Two-Circuit Architecture）**
- **快速压缩回路（Fast Compressed Circuit）**：
  - 使用**激活掩码（activation masking）** 进行PP通信压缩。
  - 使用**随机伪随机函数（PRF）生成共享掩码**，避免传输索引，实现高效压缩。
  - 采用**in-graph masking**，使前向与反向传播共享相同的稀疏模式。
- **慢速锚定回路（Slow Anchor Circuit）**：
  - 异步运行**未掩码的前向-反向传播**，获取高保真梯度。
  - 不阻塞快速回路，梯度作为**Anchor Priors**异步返回。

#### （2）**谱校正优化器（Spectral Correction Optimizer）**
- 利用锚定回路产生的延迟梯度构建一个**低秩动量缓冲区（anchor momentum buffer）**。
- 对其进行**SVD分解**，提取主方向（principal directions）。
- 设计软滤波器 $ d_i = \frac{s_i}{s_i + T_p} $，对快速回路的掩码梯度进行重加权：
  - 支持的方向被增强；
  - 无支持的噪声方向被抑制。
- 最终更新为混合形式：$ G_{\text{proj}} = \alpha G_{\text{mask}} + (1-\alpha) G_{\text{flt}} $

#### （3）**结合DP压缩**
- 在DP维度使用 **Streaming DiLoCo + PowerSGD** 实现梯度压缩与同步频率降低。
- 与PP压缩正交组合，实现端到端通信效率提升。

---

### 相比现有方法的优势

| 方面 | 本方法优势 |
|------|-----------|
| **通信效率** | PP压缩达20×，结合DP压缩总提速超 **40×**（200 Mbps下） |
| **性能保持** | 匹配甚至超越密集未压缩训练的性能 |
| **适用性广** | 适用于多种模型（Llama, Qwen, Gemma等）、规模（1B–8B）、任务类型 |
| **鲁棒性强** | 模型对剪枝更鲁棒，适合资源受限推理 |
| **无需误差反馈** | 随机掩码天然无偏，避免Top-K带来的结构性偏差 |

---

## 2. 核心实验方法和设置

### 使用的数据集

实验覆盖多个领域适配任务，训练与评估数据如下：

| 领域 | 训练数据集 | 评估数据集（指标） |
|------|------------|---------------------|
| 医疗 | `medalpaca/medical_meadow_medqa` | MedQA, MedMCQA (Acc.) |
| 编程 | `iamtarun/python_code_instructions`, `alpaca` | HumanEval, MBPP (pass@1) |
| 数学 | `openai/gsm8k` | GSM8K (EM) |
| 科学 | `allenai/sciq` | ARC-C, SciQ (Acc.) |
| 常识推理 | `Rowan/hellaswag`, `winogrande` | HellaSwag, WinoGrande (Acc.) |
| 摘要 | `cnn_dailymail`, `xsum` | CNN/DM, XSum (ROUGE-1) |
| SQL | `philikai/Spider-SQL-LLAMA2_train` | Spider (Exec. Match) |
| 文档解析 | `emozilla/quality`, `deepmind/narrativeqa` | QuALITY (Acc.), NarrativeQA (Token F1) |

此外还测试了**持续预训练与推理能力训练**（SmolLM3风格三阶段流程）。

---

### 实验设置

- **基础模型**：Llama 3.2 1B（1.23B参数）
- **硬件配置**：A100 GPU（40GB），模拟200 Mbps网络带宽
- **网格拓扑**：7×8 快速回路 + 1×8 锚定回路（共8×8 mesh）
- **训练配置**：
  - 学习率：2e-5，余弦衰减
  - Batch Size：32，上下文长度2k–65k不等
  - 优化器：AdamW
- **压缩设置**：
  - PP：90% 或 95% 激活掩码（M90/M95）
  - DP：PowerSGD (×64) + Streaming DiLoCo

---

### 基线方法对比

| 基线方法 | 类型 | 是否可比 |
|--------|------|---------|
| Dense SFT | 密集未压缩训练 | ✅ 主要对比基线 |
| +DP | 仅DP压缩 | ✅ 对比PP压缩必要性 |
| Top-K Sparsification | 激活稀疏化 | ❌ 性能崩溃 |
| AQ-SGD / TAH-Quant | 量化方法（8×/4×） | ❌ 无法支撑生成任务 |
| Bottleneck Projection | 结构化投影 | ❌ 性能下降明显 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1 & Table 3）

| 方法 | 平均准确率 | TPS（tokens/s） | 通信量（B/tok） | 相对提速 |
|------|------------|------------------|------------------|----------|
| Baseline (Dense) | ~0.35 | 1.1k | 22.9k | 1× |
| +DP Only | ~0.34 | 3.6k | 6.9k | ~3.3× |
| M90 | ~0.25 | 29.3k | 713 | ~27× |
| **M90+AP (Ours)** | **~0.35** | **27.8k** | **790** | **~25×** |
| M95 | ~0.20 | 48.9k | 370 | ~45× |
| **M95+AP (Ours)** | **~0.34** | **46.5k** | **446** | **>40×** |

> 注：M95+AP 在 **45× 吞吐提升** 下仍匹配原始性能。

---

### 与基线方法对比（Table 4）

| 方法 | Comp. Ratio | MedQA Acc. | HumanEval pass@1 |
|------|-------------|------------|------------------|
| Dense Baseline | 1× | 0.28 | 0.18 |
| AQ-SGD (8×) | 8× | 0.25 | 0.00 |
| TAH-Quant (8×) | 8× | 0.25 | 0.00 |
| 4-bit Quant | 4× | 0.25 | 0.13 |
| Top-K (10×) | 10× | 0.23 | 0.12 |
| **M95+AP (Ours)** | **20×** | **0.32** | **0.21** |

✅ **唯一在高压缩比下保持甚至超越基线的方法**

---

### 消融实验结果（Table 2 & Table 6）

#### （1）掩码比例与延迟敏感性
| 掩码率 \ 延迟 | K=10 | K=20 | K=30 | K=50 |
|---------------|------|------|------|------|
| 90% | ✅ 0.33 | ✅ 0.328 | ✅ 0.31 | ⚠️ 0.265 |
| 95% | ✅ 0.328 | ✅ 0.321 | ⚠️ 0.295 | ❌ 0.25 |
| 99% | ❌ 0.245 | ❌ 0.23 | ❌ 0.215 | ❌ 0.18 |

> 结论：**95%掩码 + 延迟≤30步** 是有效操作区间；过长延迟导致轨迹漂移过大。

#### （2）谱滤波阈值 $ T_p $
| $ T_p $ | 1e-5 | 1e-4 | **1e-3** | 1e-1 |
|---------|------|------|--------|------|
| MedQA | 0.215 | 0.290 | **0.321** | 0.245 |

> 最佳值在 $ 10^{-3} $ 附近，太小则过度抑制，太大则去噪不足。

#### （3）混合系数 $ \alpha $
| $ \alpha $ | 0.0 | **0.3** | 0.5 | 1.0 |
|-----------|-----|-------|-----|-----|
| MedQA | 0.26 | **0.321** | 0.31 | 0.18 |

> 中间值最优，完全依赖过滤梯度（α=0）会引入过多陈旧性。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **激活掩码可用于极端PP压缩**：尽管以往认为不可靠，但在**锚定先验引导下**，95%掩码仍可稳定训练。
2. ✅ **异步锚定机制解耦吞吐与精度**：通过分离“快流”与“慢锚”，实现了**高吞吐 + 高质量梯度指导**的双重收益。
3. ✅ **谱校正优于传统去噪方式**：基于SVD的低秩滤波能有效分离信号与各向同性噪声，而Top-K因结构性偏差无法修复。
4. ✅ **组合压缩带来指数级加速**：单独DP或PP压缩效果有限，**联合压缩才能突破瓶颈**，实现>40×提速。
5. ✅ **训练出的模型更具鲁棒性**：经掩码训练的模型对**post-hoc magnitude pruning** 更鲁棒（见Table 5），适合部署。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **不适用于Pretraining** | 分析表明该方法依赖“低维缓慢漂移的梯度子空间”，这在从零开始的预训练中不成立。 |
| **扩展至70B+未知** | 当前实验集中在1B–8B规模，更大模型的行为尚待验证。 |
| **依赖特定优化器设计** | 谱校正目前基于AdamW，与AdEMAMix等双EMA优化器的兼容性需进一步探索（见Appendix K）。 |
| **对延迟敏感** | 若锚定梯度延迟超过约50步，性能急剧下降。 |

---

### 未来工作方向

1. **探索与Dual-EMA优化器的融合**：将谱校正输出输入AdEMAMix，可能进一步提升稳定性。
2. **扩展至更大模型与跨地域联邦学习**：验证在极低带宽（<50 Mbps）下的可行性。
3. **动态调整掩码率与滤波参数**：根据训练阶段自适应调节压缩强度。
4. **理论分析泛化边界**：建立更精确的收敛界，指导实际部署中的超参选择。
5. **集成Parameter-Efficient Fine-Tuning（PEFT）**：如LoRA + M95+AP（已初步验证可行，见Table 8），实现内存与通信双重高效。

---

## 总结

本文提出了一套**面向去中心化环境的大语言模型适应框架**，通过**异步双回路 + 激活掩码 + 谱校正优化器**的设计，在**200 Mbps低带宽条件下实现了超过40倍的吞吐提升**，同时**完全恢复甚至略微超越密集训练的性能**。该方法不仅解决了通信瓶颈问题，还揭示了**fine-tuning过程中梯度具有低维稳定结构**这一重要特性，为未来在边缘设备、分布式社区协作等场景下的大模型训练提供了实用路径。

</details>

---

### 3. [LLM-Enhanced Multi-Agent Reinforcement Learning for Unified Electric Vehicles-Charging Station-Grid Optimization in Public Charging Systems](https://arxiv.org/abs/2609.13805)

**Authors**: Yang Zhang, Lindong Xie, Chongyu Wang, Gaojunjie Li, Siqi Bu, Edward Chung  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.13805v1  

#### Abstract
In the era of the Internet of Things (IoT), coordinating connected electric vehicle (EV) charging scheduling to balance EV charging satisfaction, station profitability, and smart grid stability presents a complex multi-objective challenge. Existing Multi-Agent Reinforcement Learning (MARL) approache...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM-Enhanced Multi-Agent Reinforcement Learning for Unified Electric Vehicles-Charging Station-Grid Optimization in Public Charging Systems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**公共充电系统中电动汽车（EV）、充电站（Charging Station）和电网（Grid）三方利益冲突且难以协同优化**的问题，提出了一种统一的调度框架。现有研究大多仅考虑其中两个主体（如EV与充电站），缺乏对三者联合优化的建模，导致：
- 仅优化用户满意度可能加剧电网负荷；
- 忽视实时动态环境变化，策略适应性差；
- 高维状态空间下训练效率低、收敛慢。

此外，传统方法在**特征选择**和**多目标平衡**方面存在以下瓶颈：
- 特征选择依赖统计方法（如MI、mRMR），计算复杂且缺乏可解释性；
- 多目标强化学习（MORL）需维护Pareto前沿，机制复杂、难以动态调整权重。

---

### 提出了什么新方法或新思路
作者提出了首个将**Large Language Model (LLM)** 融入 **Multi-Agent Reinforcement Learning (MARL)** 框架的方法，构建了一个**LLM增强的MARL统一优化框架（LLM-enhanced MARL）**，包含两大核心模块：

#### （1）LLM-based Interpretable Feature Selection (LLM-FS)
- 将状态特征和任务描述转化为自然语言输入给LLM；
- 利用LLM的语义理解能力，为每个子问题（EV、CS、Grid）打分并排序重要特征；
- 输出带文本解释的特征重要性排名，实现**可解释的特征筛选**；
- 仅保留Top-K特征用于后续MARL训练，显著降低状态维度。

#### （2）LLM-based Adaptive Multi-Objective Balancing (LLM-MOB)
- 在每轮训练中，当检测到系统“失衡”时，触发LLM进行动态权重分配；
- 输入当前环境状态（自然语言描述）和优化目标，由LLM输出三个目标（EV成本、CS利润、Grid负载）的加权系数；
- 权重之和固定为3，确保归一化，并附带推理过程，提升**决策透明度**；
- 替代复杂的Pareto优化机制，简化多目标协调流程。

最终基于MADDPG算法实现MARL策略学习。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **统一性** | 首次同时建模EV、CS、Grid三方目标，在一个闭环中联合优化 |
| **高效性** | LLM-FS一次性完成特征选择，避免每轮重复计算；训练时间减少超70% |
| **自适应性** | LLM-MOB可根据实时市场状态动态调整目标权重，无需手动调参 |
| **可解释性** | 所有特征选择与权重分配均提供自然语言解释，便于运营方理解和信任 |
| **轻量化设计** | 不依赖复杂数学建模或Pareto前沿维护，部署门槛更低 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **电力与光伏数据**：来自美国PJM Interconnection（覆盖13州，服务6500万人口）的真实电价与太阳能发电数据（2023年4月–2024年3月）；
- **EV参数**：基于2023年美国销量前十的EV车型设定电池容量、充电功率等；
- **到达模型**：EV到达服从泊松过程（Poisson Process），日均服务超过200辆EV。

---

### 实验设置
- **时间粒度**：每日划分为24个时间步（每小时一步）；
- **训练/测试划分**：每月前20天用于训练，其余用于执行评估；
- **MARL算法**：采用MADDPG作为基础算法；
- **LLM模型**：主实验使用GPT-4o，消融实验测试其他LLM（如o3-mini、Gemini、LLaMA3等）；
- **特征数量**：从原始118维状态中通过LLM-FS选出K=5个关键特征；
- **触发机制**：LLM-MOB每隔G=100个episode检查一次系统是否“失衡”，若某一主体奖励远高于另两者则重新生成权重。

---

### 评估指标
| 指标 | 含义 | 方向 |
|------|------|-------|
| **DPS** (Daily Profit of Station) | 充电站日利润 | ↑ |
| **DEG** (Dissatisfied Energy Gap) | 平均SOC未满足差距 | ↓ |
| **AQL** (Average Queue Length) | 平均排队长度 | ↓ |
| **ECC** (EV Charging Cost) | 用户充电经济成本 | ↓ |
| **LPG** (Load Pressure of Grid) | 电网负载压力（含波动惩罚） | ↓ |
| **CMEI** (Charging Market Efficiency Index) | 综合市场效率指数（归一化加权得分） | ↑ |

---

### 基线方法对比
分为两组进行比较：

#### Group 1: Feature Selection-based Baselines
| 方法 | 描述 |
|------|------|
| LassoNet+MARL / RFE+MARL / MI+MARL / mRMR+MARL / Transformer+MARL | 各类经典特征选择方法 + MADDPG |
| x+LLM-MOB | 上述方法 + 引入LLM-MOB模块（验证其普适增益） |
| Vanilla MARL | 使用全量特征 + 固定权重 |

#### Group 2: MORL-based Baselines
| 方法 | 描述 |
|------|------|
| CAPQL / GPI-PD / PGMORL | 主流MORL算法 |
| x’ / x+LLM-FS | 分别使用完整特征集 vs. LLM-FS选后的特征子集 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（执行阶段平均值，5次随机种子）
| 方法 | DEG↓ | AQL↓ | ECC↓ | DPS↑ | LPG↓ | **CMEI↑** |
|------|------|------|------|------|------|----------|
| **Ours (LLM-FS + LLM-MOB)** | **0.037** | **2.958** | **104.2** | 86.9 | **0.411** | **2.195** |
| MI+MARL | 0.083 | 7.625 | 460.6 | 82.1 | 9.227 | 1.238 |
| Transformer+MARL | 0.117 | 10.251 | 333.5 | 156.3 | 0.416 | 1.719 |
| Vanilla MARL | 0.089 | 69.128 | 251.4 | 35.0 | 0.746 | 1.698 |
| CAPQL+LLM-FS | 0.034 | 4.333 | 350.1 | 165.2 | 2.525 | 2.012 |
| GPI-PD+LLM-FS | 0.034 | 4.104 | 281.8 | 56.7 | 8.983 | 1.555 |

> ✅ **本方法在CMEI上达到最高（2.195），优于第二名CAPQL+LLM-FS约9.1%**

---

### 与基线方法的对比结果

#### （1）训练效率显著提升
| 方法 | 收敛所需episode数 | 单episode耗时（秒） | 总训练时间（分钟） |
|------|------------------|--------------------|------------------|
| **Ours** | **642** | **3.85** | **41.16** |
| Vanilla MARL | 1458 | 6.19 | 150.42 |
| Transformer+MARL | 821 | 40.41 | 552.94 |
| CAPQL+LLM-FS | 1328 | 3.86 | 85.43 |

> 🔹 **训练时间减少超过70%**，主要得益于LLM-FS一次性完成特征压缩，避免反复计算。

#### （2）Pareto Frontier表现更优
- 图7显示，本方法在**EV奖励、Grid奖励、CS奖励三维空间中拥有更广且更高的Pareto前沿**；
- 表明其能在三方之间取得更好平衡，而非牺牲某一方换取短期收益。

---

### 消融实验结果（Ablation Study）

| 方法 | DEG | AQL | ECC | DPS | LPG | CMEI |
|------|-----|-----|-----|-----|-----|------|
| **Ours (完整)** | 0.037 | 2.958 | 104.2 | 86.9 | 0.411 | **2.195** |
| w/o LLM-FS | 0.059 | 4.931 | 230.3 | 11.6 | 6.428 | 1.573 |
| w/o LLM-MOB | 0.071 | 7.275 | 290.1 | 156.7 | 3.941 | 1.764 |

> 🔸 移除任一模块均导致整体性能下降，尤其是：
> - **无LLM-FS → LPG飙升至6.428**，说明高维状态损害电网稳定性；
> - **无LLM-MOB → AQL和ECC恶化明显**，反映固定权重无法应对动态需求。

---

## 4. 关键结论和发现

### 主要发现
1. **首次实现了EV-Station-Grid三方统一优化的MARL框架**，填补了该领域研究空白；
2. **LLM可用于解决RL中的“元决策”问题**（如特征选择、权重分配），而不仅是端到端控制；
3. **语义驱动的特征选择比统计方法更有效且更具物理意义**，例如LLM能识别“期望SOC”、“离场时间”为核心因素；
4. **动态多目标平衡机制显著改善系统鲁棒性**，尤其在高峰时段自动优先保障EV及时离场；
5. **所提方法具备良好扩展性**：在多站场景下仍保持CMEI > 1.9，适用于大规模部署。

---

### 方法的局限性
1. **LLM调用延迟问题**：虽然不频繁调用，但在极端高频调度场景下仍可能成为瓶颈；
2. **对LLM质量敏感**：不同LLM表现差异较大（见Table IX），小型模型可能产生不合理权重；
3. **提示工程依赖性强**：Feature Selection和Weight Balancing的效果高度依赖prompt设计；
4. **未考虑V2G反向供电场景**：出于缓解排队焦虑考虑，暂未纳入vehicle-to-grid模式。

---

### 未来工作方向
1. 探索**轻量化LLM代理模型**（如蒸馏版）以进一步降低推理开销；
2. 将框架拓展至**城市级多充电站协同调度网络**；
3. 引入**因果推理机制**增强LLM决策逻辑的可靠性；
4. 结合**数字孪生技术**实现实时仿真与策略预演；
5. 探索**LLM与Symbolic AI结合**路径，提升系统形式化验证能力。

--- 

> 📌 **总结一句话**：  
> 本文开创性地将LLM作为“智能决策协作者”嵌入MARL框架，解决了EV充电系统中高维状态与多目标冲突难题，实现了**高效、可解释、自适应的三方共赢优化**，为下一代智能交通-能源融合系统提供了新范式。

</details>

---

### 4. [Self-Orchestrating Language Models: Leveraging Semantic Dependence for Efficient Inference](https://arxiv.org/abs/2609.14850)

**Authors**: Tian Jin  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.14850v1  

#### Abstract
Large language models (LLMs) demonstrate impressive capabilities, but their deployment presents significant efficiency challenges. Autoregressive decoding imposes substantial inference latency and under-utilizes hardware accelerators in low batch size regimes. Discrete diffusion models can generate ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Self-Orchestrating Language Models: Leveraging Semantic Dependence for Efficient Inference

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在推理过程中面临显著的效率挑战，主要体现在三个方面：
1.  **推理延迟高**：传统的自回归解码（autoregressive decoding）是串行过程，导致推理延迟长，硬件利用率低。
2.  **内存瓶颈**：长上下文推理时，Key-Value (KV) 缓存随生成的 token 数量线性增长，消耗大量显存，限制了批处理大小和吞吐量。
3.  **质量-效率权衡差**：现有的加速方法（如离散扩散模型）要么依赖启发式规则，无法达到自回归模型的质量，要么需要大量去噪步骤，导致速度慢。

### 提出的新方法和新思路
本文提出了“**自编排语言模型**”（Self-Orchestrating Language Models）这一统一范式。其核心思想是：**让语言模型自身通过标注语义依赖关系（semantic dependence）来指导其推理执行策略**。

具体来说，作者设计了一个由三部分组成的框架：
1.  **标注语言（Annotation Language）**：扩展模型的词表，引入特殊标记（如 `<promise/>`, `<sync/>`, `<reg_0>`），让模型在生成文本的同时，标注出不同内容块之间的语义依赖关系。
2.  **协同运行时（Co-designed Runtime）**：一个专门的运行时系统，能够解析并执行这些标注，从而实现并行解码、上下文管理等优化。
3.  **训练流程（Training Procedure）**：通过监督微调（SFT）和偏好优化（preference optimization）或强化学习（RL），教会模型如何准确地生成这些标注。

基于此框架，论文提出了三个具体的系统来解决不同的瓶颈：
*   **PASTA**：利用语义依赖实现**并行生成**。模型标注出可以独立生成的内容块，运行时并行解码这些块。
*   **TIP (Thinking in Place)**：利用语义依赖进行**上下文管理**。模型将推理步骤分配到有限的“思维寄存器”中，当重用寄存器时，表示旧步骤已过时，运行时即可将其从 KV 缓存中驱逐，从而控制内存占用。
*   **Planned Diffusion**：利用语义依赖为**离散扩散模型**推导去噪顺序。模型先自回归地生成一个计划，规划出多个可并行去噪的独立内容块，然后运行时并行地对这些块进行扩散去噪。

### 相比现有方法的优势
*   **输入自适应（Input-adaptive）**：优化决策基于每个输入的具体内容，而非固定的模式。
*   **可学习（Learnable）**：通过训练，模型能不断改进其标注能力，而传统方法依赖于手工设计的启发式规则。
*   **可检查（Inspectable）**：模型的决策以离散的标注形式呈现，便于调试和理解。
*   **统一范式**：三个看似不同的系统共享同一个设计哲学，证明了该方法的普适性和强大潜力。

## 2. 核心实验方法和设置

### 使用的数据集
*   **PASTA 和 Planned Diffusion**：主要在 **AlpacaEval** 基准上进行评估，该基准包含 805 个指令跟随提示（instruction-following prompts）。
*   **TIP**：在 **AIME 2024** 和 **AIME 2025** 数据集上进行评估，这两个数据集包含数学竞赛题目，用于测试长链式推理（chain-of-thought）场景下的性能。

### 实验设置和评估指标
*   **硬件**：实验在 H100 或 H200 GPU 上进行。
*   **评估指标**：
    *   **质量（Quality）**：使用 **长度受控胜率（Length-Controlled Win Rate, LCWR）**，即使用 GPT-4 或 Gemini 1.5 Pro 作为裁判模型，比较生成结果的质量。
    *   **效率（Efficiency）**：
        *   **速度提升（Speedup）**：相对于基线模型的推理时间减少比例。
        *   **理论加速比（Theoretical Speedup）**：理论上可达到的最大加速比。
        *   **实时性（Latency）**：生成响应的平均墙钟时间。
        *   **吞吐量（Throughput）**：每秒生成的 token 数（tok/s）。
        *   **活跃 KV 缓存大小（Live KV Cache Size）**：在 TIP 实验中，衡量内存占用的关键指标。
    *   **帕累托最优（Pareto-optimality）**：综合评估质量和效率的权衡，目标是找到在相同质量下更快，或在相同速度下质量更高的模型。

### 基线方法对比
*   **PASTA**：对比了标准自回归解码（Baseline-SFT）、**Skeleton-of-Thought (SoT)** 和 **APAR**。
*   **TIP**：对比了标准密集解码（Dense）、**StreamingLLM** 和 **Paged H2O (PH2O)**。
*   **Planned Diffusion**：对比了自回归解码（AR）、标准扩散模型（Diffusion）、**Fast-dLLM**、**Skeleton-of-Thought (SoT)** 和 **Pasta-SFT**。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
*   **PASTA**：
    *   在 AlpacaEval 上实现了 **1.21× 到 1.93×** 的几何平均速度提升。
    *   质量变化范围为 **+2.2% 到 -7.1%** 的胜率变化。
    *   **Pareto-支配所有现有异步解码方法**，在质量-速度权衡上表现更优。

*   **TIP**：
    *   在 AIME 2024 上，相比密集解码，**准确率从 14.4% 提升至 20.8%**，同时**活跃 KV 缓存减少了约 59%**。
    *   总解码时间缩短了 **28-35%**，吞吐量提高了 **21-30%**。
    *   在相同的内存预算下，**准确率比 StreamingLLM 和 Paged H2O 高出 3-10 个百分点**。

*   **Planned Diffusion**：
    *   在 AlpacaEval 上实现了 **1.27× 到 1.81×** 的速度提升，质量下降仅为 **-0.87% 到 -5.4%**。
    *   **建立了新的帕累托前沿**，在质量-延迟权衡上优于自回归和扩散基线。
    *   其关键路径长度（critical path length）比自回归解码短 **2.3-2.8 倍**，解释了其速度优势。

### 消融实验结果
*   **PASTA**：消融实验证明，直接优化“理论加速比”作为效率指标效果最好；使用 LLM 预测位置 ID（Pred-10x）比固定长度或精确预测效果更好。
*   **TIP**：消融实验证明，基于语义依赖的驱逐策略比简单的滚动窗口（rolling）策略准确率更高。
*   **Planned Diffusion**：移除 `<topic>` 属性会导致质量严重下降（LCWR 从 46.65% 降至 23.33%），证明了主题描述对于引导并行生成至关重要。移除 `<sync/>` 会降低质量和增加延迟。

## 4. 关键结论和发现

### 主要发现
1.  **模型可以自我优化**：语言模型有能力学习并标注其自身输出中的语义依赖关系，并利用这些信息来指导推理执行，从而实现自我编排。
2.  **统一的设计方法论**：PASTA、TIP 和 Planned Diffusion 三个系统虽然解决的问题不同，但都遵循“标注-运行时-训练”的统一设计范式，证明了该方法的强大通用性。
3.  **帕累托最优的权衡**：所提出的方法在各自的任务上均实现了帕累托最优，即在不牺牲质量的情况下提升了效率，或在可接受的质量损失下获得了巨大的速度提升。
4.  **可扩展性**：通过偏好优化等技术，模型的性能可以随着更多训练计算资源的投入而持续提升。

### 方法的局限性
*   **经验性结果**：论文的结果主要是经验性的，展示了学习到的标注优于现有启发式方法，但没有分析理论上能达到的最高速度提升或内存缩减上限。
*   **未完全统一**：尽管三个系统共享一个设计哲学，但它们尚未收敛到一个单一的、通用的标注语言和运行时系统，目前仍是三个独立的系统。
*   **训练复杂性**：训练过程（尤其是结合 RL 或偏好优化）可能较为复杂和昂贵。

### 未来工作方向
*   **构建统一系统**：开发一个单一的、通用的自编排框架，能够同时支持并行解码、上下文管理和扩散去噪等多种优化。
*   **探索更复杂的命令**：在第6章中提到的“指令式自编排”（Imperative Self-Orchestration）中，可以探索优先级（priority）和预取（prefetching）等更高级的指令。
*   **应用于更广泛的场景**：将自编排的思想推广到多模态模型、智能体（agent）规划等更复杂的任务中。
*   **降低训练成本**：研究更高效、更轻量化的训练方法，使自编排能力更容易被应用。

</details>

---

### 5. [Joint Optimization for Federated Learning and Transmission over Unreliable Wireless Networks with Heterogeneous Data](https://arxiv.org/abs/2609.14246)

**Authors**: Changheng Wang, Xianchao Zhang, Zhiqing Wei, Lingzhu Zhao, Zhongming Yang, Zhiyong Feng  
**Category**: cs.DC  
**Published**: 2026-09-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.14246v1  

#### Abstract
In wireless federated learning (FL), data heterogeneity and multiple local updates induce client drift, degrading model convergence. It is further affected by unreliable wireless links, as transmission errors may invalidate model updates. To address these challenges, we propose a federated random wa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Joint Optimization for Federated Learning and Transmission over Unreliable Wireless Networks with Heterogeneous Data*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**无线联邦学习**（Wireless FL）中的两大核心挑战：
- **数据异构性**（Data Heterogeneity）：客户端数据呈 Non-IID 分布，导致模型训练过程中出现 **client drift**，影响全局收敛。
- **不可靠无线传输**（Unreliable Wireless Links）：由于信道衰落、干扰和噪声，模型参数在传输过程中可能发生比特错误，导致更新失效。

传统方法如 FedAvg 在高异构性和不稳定信道下表现不佳，而现有传输优化方案通常忽略与学习过程的联合设计。

---

### 提出的新方法与创新思路

作者提出了一种名为 **Federated Random Walk Averaging**（**FedRW**）的新型联邦学习框架，并结合**联合优化机制**解决上述问题。

#### （1）FedRW 框架
- 基于 **parallel random walk**（并行随机游走）路径进行本地模型更新。
- 多个 RW 链并行运行，每条链依次在客户端间传递并更新模型，最终将链尾模型上传至服务器聚合。
- 替代传统的集中式 client-server 更新模式，通过跨客户端遍历缓解数据异构性带来的偏差。

#### （2）模型分包重传机制
- 将模型参数划分为多个 **packet**，支持 **ARQ**（Automatic Repeat Request）机制。
- 单个 packet 出错仅需重传出错部分，避免整模型丢弃，显著提升传输鲁棒性。

#### （3）联合优化问题建模
- 构建一个联合优化问题，整合以下三个维度：
  - **RW 路径选择**（Path Selection）
  - **packet size 设计**
  - **最大重传次数** $ R $
- 目标是在满足延迟约束的前提下最小化训练损失。

#### （4）分布式求解算法
- 提出一种 **resilience-aware beam search with dynamic pruning** 策略，实现高效可靠的下一跳节点选择。
- 各客户端基于本地状态独立决策，降低中心协调开销，适用于大规模系统。

---

### 相比现有方法的优势

| 维度 | FedRW 优势 |
|------|-----------|
| **抗数据异构性** | 利用 RW 遍历多样数据分布，隐式平滑梯度偏移，优于 FedAvg、FedProx、FedNova 等 |
| **抗信道不稳定性** | 分包 + 重传机制有效应对 packet error，优于一次性传输全模型的方法（如 [6][7]） |
| **通信效率** | 并行 RW 结构平衡计算与通信负载，虽增加 hop 数但提升信息多样性 |
| **理论保障** | 推导了在 **Polyak-Łojasiewicz (PL) 条件**下的期望收敛上界，建立了传输参数与学习性能之间的显式关系 |
| **可扩展性** | 分布式优化策略无需全局信息，适合资源受限的大规模网络 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MNIST** 和 **Fashion-MNIST**：使用 MLP 模型（784×100×10），ReLU + Softmax
- **CIFAR-10**：采用 **VGG13**
- **CIFAR-100**：采用 **ResNet-18**
- 所有任务均使用交叉熵损失函数

### 数据异构性设置
- **Mixed Non-IID**：每个客户端拥有 $\xi_n$ 比例的 IID 数据和 $(1-\xi_n)$ 比例的标签分片 Non-IID 数据（$\xi_n=0$ 表示完全 Non-IID）
- **Dirichlet Non-IID**：按狄利克雷分布分配类别，浓度参数 $\alpha=0.1$ 或 $0.2$ 控制异构程度

### 实验环境与参数
- 客户端数量：100，均匀分布在半径为 500m 的圆形区域内
- 网络拓扑：构建有向 Bernoulli 随机图（连接概率 0.5）
- 无线参数：
  - 带宽 $B = 10$ MHz
  - 发射功率 $P = 23$ dBm
  - 噪声谱密度 $N_0 = -174$ dBm/Hz
  - 最大时延约束 $\gamma_T = 20$ ms
  - 最大重传次数上限 $\gamma_R = 2$
  - Bit Error Rate (BER) 变化测试范围：$10^{-7}$ 到 $10^{-4}$

### 评估指标
- **分类准确率**（Test Accuracy）
- **训练损失下降速度**
- **收敛稳定性**（震荡程度）
- **成功聚合的 RW 链数量**

### 对比的基线方法
1. **Ideal FedRW**：理想信道（无错误）、随机路径选择
2. **RandParam FedRW**：随机设定传输参数，仅优化路径
3. **Greedy-RW FedRW**：贪婪选择最可靠下一跳，但优化传输参数
4. **FedAvg**、**FedProx**、**FedNova**：作为标准 FL 基线用于比较学习性能

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 场景 | FedRW 提升幅度 | 说明 |
|------|----------------|------|
| 高度 Non-IID 下（MNIST, $\xi_n=0$） | **+8.72%** 准确率 vs FedAvg | 收敛更快且更稳定 |
| Dirichlet Non-IID ($\alpha=0.1$) | **+2.26% ~ +9%** 准确率 | 在 Fashion-MNIST 上达 9% 提升 |
| 与其他 FedRW 变体对比 | **至少 +2.78%** 准确率 | 联合优化显著优于 RandParam/Greedy-RW |
| 收敛速度 | 明显快于所有 baseline | 特别是在后期阶段误差更低 |

---

### 详细对比结果

#### （1）与主流 FL 方法对比（Fig. 5）
- 在 MNIST 和 CIFAR-100 上，FedRW 达到 **92.68%** 和 **61.75%** 准确率，均为最高。
- FedProx 表现接近 FedAvg，但在极端异构下因正则项过强限制适应能力。
- FedNova 存在较大震荡，因其梯度缩放机制放大稀疏客户端噪声。

#### （2）与 FedRW 变体对比（Fig. 8）
- **Proposed vs Greedy-RW**：平均提升 **2.78%** 准确率
  - 原因：动态剪枝 beam search 具备“前瞻性”，避免贪心策略陷入局部最优
- **Proposed vs RandParam**：提升 **4.67%**
  - 原因：优化传输参数后可接入更多高质量邻居，提升路径可靠性
- 训练损失曲线显示 proposed 方法震荡最小，收敛最平稳

#### （3）消融实验与关键发现
- **链内数据多样性影响**（Fig. 6）：
  - 当链内数据完全异构（$y=1$）时性能最佳
  - 若链内同质（$y=0$）则性能严重下降 → 验证了 RW 遍历多样性的重要性
- **聚合方式对比**（Fig. 7）：
  - 加权聚合（按链长、样本数等）反而引入偏差，尤其在数据量大的链上过度加权末节点
  - **unweighted aggregation** 更稳定，推荐作为默认方案
- **约束 (22c) 的必要性**（Fig. 9）：
  - 若允许某些高可靠性客户端参与多条链，会导致采样偏差，准确率下降 **11.3%**
  - 证明必须限制每个客户端最多被选一次以保证数据多样性

#### （4）不同网络条件下的鲁棒性（Fig. 10）
- **随 $\gamma_R$ 增大**：各方法差距缩小，但 proposed 始终领先
- **随 $\gamma_T$ 放宽**：proposed 能更好利用额外时延预算提升成功率
- **BER 升高时**：baseline 性能急剧下降，而 proposed 因联合优化仍保持较高聚合链数

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **FedRW 能有效缓解数据异构性**：通过并行 RW 遍历多样化数据分布，减少 client drift，提升泛化能力。
2. ✅ **分包 + 重传机制对不可靠无线链路至关重要**：相比整包传输，大幅降低模型丢失风险。
3. ✅ **路径选择与传输参数应联合优化**：单独优化任一部分无法达到最优性能。
4. ✅ **分布式 beam search 策略高效可行**：在低复杂度下逼近理想性能，具备良好可扩展性。
5. ✅ **unweighted aggregation 更优**：在 RW 序列结构中，链尾模型已集成前序更新，无需额外加权。

---

### 方法的局限性
- **理论假设依赖 PL condition**：虽然广泛存在于神经网络训练中，但仍非普遍成立。
- **未考虑 CSI stale 问题**：尽管通过本地探测缓解，但在高速移动场景下可能失效。
- **缺乏对动态拓扑的支持分析**：当前模型假设拓扑相对静态。
- **未探索更复杂的聚合机制**：如 attention-based 或 meta-weighting，留待未来研究。

---

### 未来工作方向
1. **建立收敛下界**：结合 rate-distortion 理论或对抗构造分析极限性能。
2. **扩展至非 PL 场景**：设计适用于一般非凸目标的 Lyapunov-type 收敛分析工具。
3. **引入学习-based 路径规划**：使用 RL 或 GNN 进行智能 RW 路由。
4. **支持动态网络与移动性**：适应无人机、车联网等移动 FL 场景。
5. **融合 Quantum Computing**：利用量子并行性加速 RW 路径搜索与参数优化。

---

> **总结一句话**：  
> FedRW 通过 **random walk 遍历 + 分包重传 + 联合优化**，实现了在 **高异构、低质量无线环境**下的高性能联邦学习，在准确率和收敛速度上全面超越现有方法，且具备良好的理论支撑与工程实用性。

</details>

---

### 6. [Temporal Self-Distillation: Faster Inference in Discrete Diffusion Language Models](https://arxiv.org/abs/2609.15177)

**Authors**: Shijian Xu, Andrea Miele, Metod Jazbec, Volker Roth, Eric Nalisnick, Ilija Bogunovic  
**Category**: cs.LG  
**Published**: 2026-09-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.15177v1  

#### Abstract
Diffusion language models (dLLMs) promise fast inference by generating multiple tokens in parallel, but suffer severe performance degradation when parallel decoding is pushed too aggressively. We introduce Temporal Self-Distillation (TSD), a simple on-policy method that trains dLLMs for fast inferen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Temporal Self-Distillation: Faster Inference in Discrete Diffusion Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
扩散语言模型（**dLLMs**）虽然理论上支持并行生成多个 token，从而实现快速推理，但在实际应用中，当并行解码过于激进时，生成质量会显著下降，导致明显的 **speed-quality trade-off**。现有方法如离线蒸馏（offline distillation）需要两阶段训练流程，依赖预先生成的教师轨迹，成本高且无法适应策略的动态演化。

### **提出的新方法：Temporal Self-Distillation (TSD)**
TSD 是一种**轻量级、单阶段、on-policy 的自蒸馏方法**，其核心思想是：
- 在同一个模型的去噪轨迹中，将**最终提交 token 时刻的预测分布**作为“教师”，指导**早期时刻的预测分布**进行学习。
- 即：让模型在早期就“预见到”它最终会预测什么，从而提升早期预测的可靠性，使更激进的并行解码成为可能。

### **相比现有方法的优势**
| 方面 | TSD | 传统方法（如 dParallel） |
|------|-----|------------------------|
| **训练方式** | 单阶段，on-policy，无需额外数据 | 两阶段，需离线生成教师轨迹 |
| **教师来源** | 模型自身（self-teacher） | 外部慢速多步教师模型 |
| **计算开销** | 低，仅需一次前向传播获取目标 | 高，需额外推理生成轨迹 |
| **适应性** | 可直接用于基础模型和 RL 后训练模型 | 通常仅适用于特定训练阶段 |

> ✅ **核心优势**：TSD 在不牺牲性能的前提下，显著提升了 dLLMs 在**低计算预算（low-NFE）下的推理效率**，避免了复杂的两阶段流程。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
覆盖三大领域共 **7 个基准任务**：
- **数学推理**：GSM8K、MATH-500
- **规划任务**：Countdown、Sudoku
- **代码生成**：HumanEval、MBPP、LiveCodeBench

### **实验设置**
- **模型初始化**：
  - 数学与代码任务：基于 `LLaDA-8B-Instruct` 初始化
  - 规划任务：基于经 RL 后训练的 `GDSD` 检查点初始化（因原模型表现差）
- **训练方式**：使用 **LoRA** 进行参数高效微调（rank=64）
- **响应长度**：默认 $ L=256 $，部分任务测试 $ L=128 $ 和 $ L=512 $
- **解码策略**：采用 **Fast-dLLM** 自适应解码，通过调整置信度阈值 $\lambda$ 控制并行程度

### **评估指标**
- **主要指标**：**NFE（Number of Function Evaluations）**，即前向传播次数，衡量推理速度
- **任务性能指标**：
  - 数学/规划：准确率（Accuracy / Pass Rate）
  - 代码：执行通过率（Pass@1）
- **评估方式**：零样本（zero-shot），温度为 0 的确定性解码

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **LLaDA (Fast-dLLM)** / **GDSD (Fast-dLLM)** | 未进一步训练的基础模型，使用 Fast-dLLM 解码 |
| **dParallel** | 外部蒸馏的可学习并行解码模型，代表当前最优离线蒸馏方法 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **代码生成（MBPP）**
- TSD 在约 **20 NFE** 下达到 **40% Pass Rate**
- 相比 LLaDA-Fast-dLLM，**减少 ~2.3× NFE**
- 相比 dParallel，**减少 ~1.5× NFE**

> 📈 图 1 显示：TSD 在低 NFE 区域显著优于所有基线，帕累托前沿明显左移。

#### **规划任务（Countdown）**
- TSD 在 **~20 NFE** 达到 **80% 准确率**
- 对比 GDSD 基线需 **~90 NFE** 才能达到相同水平 → **加速 4.5×**

#### **数学推理（GSM8K）**
- TSD 在 **35–50 NFE** 达到 **~80% 准确率**
- 基线需 **~65 NFE** → 提前 **~30 NFE** 达到同等性能

#### **综合表现**
| 任务 | TSD 相对加速比 | 是否超越 dParallel |
|------|----------------|--------------------|
| MBPP | ~2.3× | ✅ 是（在低 NFE） |
| Countdown | ~4.5× | ✅ 是 |
| Sudoku | ~3× | ✅ 是（更高精度 + 更快） |
| MATH-500 | ~1.5×（中等 NFE） | ❌ 否（峰值仍低于基线） |

### **消融实验结果**

#### **(1) 长度感知奖励加权（Length-aware Reward Weighting）**
- 引入长度惩罚后，模型倾向于生成**更短、更紧凑的正确程序**
- 在 MBPP 和 GSM8K 上有效防止“填满画布”的冗余输出
- 跨画布泛化更好（$L=128, 256, 512$ 均稳定）

#### **(2) 散度选择：JSD vs KL**
- 使用 **KL 散度**会导致训练不稳定，出现**奖励崩溃（reward collapse）** 和 NFE 饱和
- 使用 **JSD（β=0.5）** 则训练平稳，持续降低 NFE
- 结论：**JSD 更适合 TSD 的对齐目标**

#### **(3) 奖励加权 vs 无奖励加权**
- 移除奖励加权（即所有轨迹平等对待）会导致性能下降
- 表明：**只对高质量轨迹进行蒸馏** 是必要的，否则会强化错误行为

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **TSD 显著改善了 dLLMs 的 speed-quality 权衡**，特别是在低 NFE 场景下，能以极少的前向传播次数达到甚至超过基线的峰值性能。
2. ✅ **TSD 是通用且灵活的方法**，既可用于基础 dLLM，也可用于已进行 RL 后训练的模型（如 GDSD），说明其可作为通用加速模块。
3. ✅ **on-policy 自蒸馏优于离线蒸馏**：TSD 不依赖外部教师或固定数据集，训练更简洁高效，且能随策略演进动态更新目标。
4. ✅ **长度感知奖励和 JSD 散度是关键设计**，前者促进简洁输出，后者保障训练稳定性。

### **方法的局限性**
- ❌ **不增加模型能力**：TSD 只能让已有行为更早显现，不能产生模型原本无法生成的内容。
- ❌ **峰值性能受限**：在大 NFE 预算下，TSD 通常不会超越基线的最高准确率（如 MATH-500）。
- ❌ **依赖程序化奖励信号**：需要任务具备可编程的验证器（verifier），难以应用于开放生成任务（如故事创作）。
- ❌ **假设 unmasking-only 解码**：若允许 revising 已提交 token，则“提交时间 $t_e$”定义失效，需重新设计目标。

### **未来工作方向**
- 探索 **TSD 与其他 RL 方法结合**，在提升能力的同时优化推理效率。
- 将 TSD 扩展至 **continuous diffusion models** 或 **vision-language 模型**。
- 设计适用于 **开放生成任务** 的软奖励机制，放宽对 verifier 的依赖。
- 研究如何在 **multi-step revision 解码器** 中定义有效的 temporal self-teaching 目标。

---

> 🔚 **总结一句话**：  
> **TSD 提供了一种简单而强大的方式，让扩散语言模型“学会提前预测自己的答案”，从而在极低推理成本下实现高质量生成，推动 dLLMs 向实用化迈进一大步。**

</details>

---

### 7. [ZGCM-1: A Fully Open and Extremely Efficient Foundation Model for Math and Agentic Search](https://arxiv.org/abs/2609.13356)

**Authors**: Jiyan He, Guang Liang, Hao Liu, Haoxiang Guan, Jinbo Sun, Junyi Guo, Wenjun Feng, Yantai Xie, Yifei Shen, Bin Shao, Chuyang Wei, Kai Chen, Kexin Zhou, Minghang Zhu, Shuxin Zheng, Tie-Yan Liu, Taine Zhao, Wenhui Zhu, Xueyin Xu, Xiaoqing Zhang, Yatao Li, Yuxuan Ren  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.13356v1  

#### Abstract
In this work, we present ZGCM-1, a fully open 7B dense foundation model trained from scratch with extreme data, system, and algorithmic efficiency. ZGCM-1 is founded on a core premise: compact models cannot passively memorize the open web, but can overcome parametric capacity limits by coupling deli...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ZGCM-1: A Fully Open and Extremely Efficient Foundation Model for Math and Agentic Search

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前基础模型在数学推理和智能体搜索（agentic search）任务上的前沿进展被两大瓶颈所限制：

- **规模壁垒（Scale Barrier）**：高性能被认为仅属于千亿参数级系统，将计算资源有限的研究者排除在外。
- **不透明壁垒（Opacity Barrier）**：多数先进模型仅以“开放权重”形式发布，训练配方、课程设计、长上下文调度等关键环节仍为黑箱。

ZGCM-1旨在打破这些壁垒，推动**高效、完全开源**的智能研究范式。

---

### 提出了什么新方法或新思路

ZGCM-1提出一个核心假设并围绕其构建完整技术栈：

> **紧凑模型虽受限于静态参数容量，但可通过“主动内部思考 + 外部工具调用”的双重机制超越这一限制。**

基于此，论文提出了四大关键技术创新：

#### （1）**架构-系统协同设计（Architecture & System Co-design）**
- 采用 **Hybrid Attention** 架构：在32层中，27层使用 **gated sliding-window attention (SWA)**（窗口128），5层使用全局注意力（比例 5:1）。
- 引入 **FP8混合精度训练** 与 **Muon优化器**，结合 **TWEO** 激活正则化，实现稳定高效训练。

#### （2）**渐进式课程与MDP中期训练（Progressive Curriculum & MDP Mid-Training）**
- 上下文长度从16K → 64K → 256K逐步扩展。
- 将交互轨迹重构为 **Markov Decision Process (MDP)** 状态-动作转换，提供密集的step-level监督信号。

#### （3）**执行对齐的混合SFT（Execution-Grounded Alignment & Mixed SFT）**
- 在SFT阶段同时训练 **think模式**（显式推理链）和 **no-think模式**（直接响应），使单一模型可动态切换行为模式。
- 通过联合训练实现跨模式迁移（如推理训练提升直接响应准确性）。

#### （4）**AI原生研发流程（AI-Native R&D）**
- 构建由人类研究员指导的 **agent swarm**，覆盖数据处理、实验监控、部署全流程。
- 共享agent harness集成脚本、工作流、调试经验，支持自主迭代。

---

### 相比现有方法的优势

| 维度 | ZGCM-1优势 |
|------|-----------|
| **效率** | 相比BF16/AdamW基线，**16K预训练time-to-loss加速约4.2倍** |
| **内存占用** | KV Cache每token开销降低 **6.4倍**（256K上下文仅需5.0 GiB） |
| **吞吐量** | 256K上下文下推理吞吐提升 **3.94倍** |
| **开放性** | 完全开源：模型权重（各阶段）、训练代码、数据配方、日志、评测套件 |
| **性能** | 在7B级别达到甚至超越百B级模型表现 |

---

## 2. 核心实验方法和设置

### 使用的数据集

ZGCM-1训练分为三个阶段，数据来源广泛且经过精细治理：

#### 预训练阶段（General Pre-Training）
- **Web数据**：高质量英文网页 + 控制比例的中文网页
- **学术与OCR数据**：PDF教育内容、arXiv论文、Ai2发布的科学文本（经olmOCR提取）
- **代码数据**：GitHub仓库文件、代码丰富网页
- **数学数据**：Proof-Pile-2风格的LaTeX数学语料、教科书、分类筛选的网络数学内容
- **LaTeX论文**：RedPajama-1T中的arXiv LaTeX源码
- **专项推理数据**：Nemotron-Pretraining-Specialized-v1

#### 中期训练（Mid-Training）
- 引入更高密度的**推理、指令、智能体交互轨迹**
- 包括软件工程任务、终端操作、多步网络搜索等真实环境交互数据
- 所有轨迹被转化为 **MDP状态-动作格式** 进行监督

#### 监督微调（SFT）
- 数据总量约492万条，分为：
  - **通用数据（96.46%）**：涵盖指令遵循、知识、数学、代码、对话、推理
  - **智能体数据（3.54%）**：深研、软件工程、终端交互
- 实施严格去污染（8-gram匹配过滤）、质量分层筛选（约50%候选被剔除）

---

### 实验设置和评估指标

#### 模型配置
- 模型大小：**7.39B dense** 参数
- 上下文长度：**256K tokens**
- 架构：Decoder-only Transformer，GQA，SwiGLU，RoPE
- 训练框架：NVIDIA Megatron Core，H100 GPU集群

#### 推理设置（Evaluation）
- **思考模式（think mode）**：采样温度1.0，top_p=1.0
- 上下文预算：最多4,096 prompt tokens + 258,048 generated tokens
- 多数非智能体任务报告 **mean pass@1 over 32 runs**

#### 评估指标
- 数学推理：MATH-500、AIME、HMMT、IMO-AnswerBench、miniF2F
- 编程能力：HumanEval+、MBPP+、LiveCodeBench v6
- 知识理解：MMLU、GPQA-Diamond
- 指令遵循：IFEval
- 智能体能力：
  - **WebWalkerQA**、**BrowseComp**、**GAIA**
  - **Binary Function Search**（新提出的反汇编函数定位任务）

---

### 基线方法对比

对比模型包括：
- 同规模开源模型：`Qwen3-8B`, `MiniCPM4.1-8B`, `Olmo-3-7B-Think`, `MiMo-7B-RL`
- 百B级以上闭源模型：`Qwen3-235B-A22B`, `GLM-5.1`, `Claude Sonnet`, `GPT-4o`, `Kimi-K2`

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Think Mode）

| 基准 | ZGCM-1-7B | 最佳基线 | 备注 |
|------|------------|----------|------|
| **MATH-500** | **97.13%** | 96.32% (DeepSeek-R1) | 超越所有同规模模型 |
| **AIME 2026** | **75.00%** | 69.17% (Qwen3-8B) | 显著领先 |
| **HMMT 2025** | **70.42%** | 61.50% (Qwen3-8B) | 遥遥领先 |
| **HumanEval+** | 90.24% | 89.90% (Olmo-3-7B-Think) | 接近最优 |
| **MBPP+** | 63.23% | 65.54% (DeepSeek-R1) | 略低但合理 |
| **MMLU** | 73.88% | 85.40% (Qwen3-8B) | 知识类稍弱 |
| **IFEval** | 75.42% | 88.20% (Olmo-3-7B-Think) | 指令遵循仍有差距 |

> ✅ **平均排名**：在14项推理基准上，ZGCM-1-7B **平均排名第2.43位**，优于大多数同规模模型。

---

### 智能体任务表现（Agentic Search）

| 任务 | ZGCM-1-7B | 对比最强模型 | 分析 |
|------|-----------|--------------|------|
| **WebWalkerQA** | **63.09%** | 63.00% (Kimi-K2) | 超越Kimi-K2，接近Claude Sonnet (61.70%) |
| **BrowseComp** | **19.43%** | 54.90% (GPT-5) | 仍落后于顶级模型，但显著优于多数 |
| **GAIA text-only** | 42.52% | 76.40% (GPT-5) | 中等水平 |
| **Binary Function Search** | **62.00%** | 66.00% (GLM-5.1) | 在50个剥离ELF文件中精准定位函数入口，远超同规模模型（Qwen3-8B仅12%） |

> 🔥 **特别亮点**：在 **Binary Function Search** 上，ZGCM-1-7B 表现惊人，接近百B级GLM-5.1，远超其他7B级模型。

---

### 消融实验结果（Ablation Studies）

#### （1）数据质量 vs. 数据量（Finding 4）
- 对SFT数据进行三种处理：
  - 最小处理（~2.08M样本）
  - 广泛过滤（~1.83M）
  - 质量优先（~1.145M）
- 结果显示：**质量优先策略使六基准均值得分从67.78升至68.83**，证明“质优于量”。

#### （2）长CoT训练的权衡（Finding 5）
- 过度增加长链推理（long-CoT）会导致指令遵循下降。
- 通过动态校准混合比例，在保持强推理的同时避免性能退化。

#### （3）统一训练的跨模式迁移（Finding 7）
- 在no-think模式下测试发现，接受过think/no-think混合训练的模型，在AIME、MBPP+等任务上表现更好。
- 例如：AIME 2025从10.00% → 43.33%，说明**推理训练提升了直接响应的质量**。

#### （4）中期训练绕过超长SFT需求（Finding 7）
- 即使SFT只使用较短序列（64K/256K），也能激活256K长上下文能力。
- 说明**长上下文能力可在中期训练中建立，无需依赖昂贵的256K长SFT数据**。

---

## 4. 关键结论和发现

### 主要发现

1. **紧凑模型可通过“思考+工具”机制突破参数限制**  
   ZGCM-1证明，即使只有7B参数，也能在复杂推理和智能体任务上媲美百B级模型。

2. **系统-算法协同设计带来巨大效率增益**  
   - Hybrid Attention + FP8 + Muon + TWEO → **4.2×预训练加速**
   - KV Cache减少6.4倍 → 支持低成本长上下文推理

3. **MDP式中期训练有效注入决策监督**  
   将交互轨迹转为state-action对，提供更密集的学习信号。

4. **AI原生研发可行且高效**  
   - 实验监控、部署可达L4高自主级别
   - 但架构与算法设计仍高度依赖人类判断（L2）

5. **开放科学促进可复现研究**  
   发布全部训练阶段权重、代码、数据配方、日志，极大降低社区参与门槛。

---

### 方法的局限性

1. **静态知识受限**  
   在纯闭卷任务中仍落后于更大模型，因参数记忆容量有限。

2. **指令遵循与推理的权衡**  
   强推理训练可能导致指令严格性下降（如IFEval偏低）。

3. **通用软件与终端代理尚不成熟**  
   在SWE-bench Verified和Terminal-Bench 2.0上成功率仅为4.0%和2.25%。

4. **环境协议脆弱性**  
   工具调用格式偏差、API延迟等易导致多步任务失败。

---

### 未来工作方向

1. **扩展至稀疏MoE架构**  
   在保持低推理FLOPs前提下扩大参数容量。

2. **端到端交互式强化学习**  
   在真实沙盒环境中进行多轮RL训练（如终端、浏览器、编译器）。

3. **自主动态知识检索**  
   当检测到不确定性时自动触发搜索子程序。

4. **自演化的AI4AI生态系统**  
   实现代理自主提出假设、优化内核、设计合成环境。

---

> 📦 **项目开源地址**：
> - **Code**: [https://github.com/zgcagi/ZGCM-1](https://github.com/zgcagi/ZGCM-1)
> - **Model**: [https://huggingface.co/zgcagi/ZGCM-1-7B](https://huggingface.co/zgcagi/ZGCM-1-7B)
> - **Data**: [https://huggingface.co/datasets/zgcagi/ZGCM-1-Data](https://huggingface.co/datasets/zgcagi/ZGCM-1-Data)

ZGCM-1不仅是一个高性能模型，更是一套**可复制、可验证、可扩展**的开源智能研发范式，为社区提供了通往高效AGI的新路径。

</details>

---

### 8. [MAPS: Memory-Aware Predictive Scheduling Framework for Large Language Model Serving](https://arxiv.org/abs/2609.15359)

**Authors**: Tiancheng Zhang, Yulin Chen, Yunfeng Zhao, Shaoyuan Huang, Cheng Zhang, Xiaofei Wang  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.15359v1  

#### Abstract
The surge of large language model (LLM) applications on personal devices imposes massive, bursty workloads on cloud serving infrastructure. While prefill-decode disaggregation improves throughput and scalability, memory-bound decode instances often suffer from persistent load imbalance, as output le...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MAPS: Memory-Aware Predictive Scheduling Framework for Large Language Model Serving

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Prefill-Decode (PD) disaggregation** 架构下，大型语言模型（LLM）推理被拆分为计算密集型的 **prefill** 阶段和内存密集型的 **decode** 阶段。由于输出长度在请求到达时未知，传统的调度策略如 **Round-Robin (RR)** 或 **Least-Request (LR)** 忽略了请求级别的资源需求差异，导致以下问题：

- **解码器间负载不均衡**：长输出请求可能导致某些 decoder 的 KV cache 耗尽，而其他 decoder 仍空闲。
- **队列积压与预占频繁发生**：长请求排队会阻塞短请求，造成尾延迟（tail latency）显著上升。
- **缺乏预测性调度机制**：现有系统多为反应式（reactive），无法提前感知请求的内存需求。

### 提出了什么新方法或新思路
作者提出 **MAPS**（Memory-Aware Predictive Scheduling），一个面向 PD-disaggregated LLM serving 的预测性调度框架，其核心创新包括：

#### （1）设备辅助的推测性输出长度预测（Device-Assisted Speculative Prediction）
- 在 **request-origin 设备端**部署轻量级 LLM（SP, Speculative Predictor），并行执行输出长度预测。
- 预测过程与云端的 prefill 阶段重叠，**不增加端到端延迟**。
- 使用 **LoRA 微调**，训练目标为多个下游 LLM 在不同 temperature 下生成的最大输出长度，确保预测具有保守性。

#### （2）不确定性感知校准模块（Uncertainty-Aware Calibration, UAC）
- 引入 **Conformal Prediction** 技术，将原始预测值转换为带有统计保证的上界区间。
- 定义单侧非一致性分数（one-sided nonconformity score）来捕捉低估风险。
- 输出满足 $ P(L \leq L^{\text{up}}) \geq 1-\alpha $ 的校准上界，保障内存可行性。

#### （3）分层全局-局部调度策略（Hierarchical Global-Local Scheduling）
- **Global Level（跨实例）**：基于校准后的上界进行内存感知调度，选择 KV cache 可容纳且等待队列最短的 decoder。
- **Local Level（实例内）**：在每个 decoder 内部采用 **SJF（Shortest-Job-First）重排序**，缓解头阻塞（HOL blocking）。
- 引入最大等待时间阈值防止长任务饥饿。

### 相比现有方法的优势
| 维度 | MAPS | 现有方法（如 vLLM, SGLang, Llumnix） |
|------|------|-------------------------------|
| 调度依据 | 预测 + 校准的输出长度上界 | 请求计数（RR/LR），无预测 |
| 内存安全性 | 显式检查 KV cache 容量，避免过载 | 依赖运行时监控，易触发 preemption |
| 延迟开销 | 预测与 prefill 重叠，调度开销 < 2ms | 无额外开销，但尾延迟高 |
| 尾延迟控制 | 显著降低 P99 E2E 延迟 | 尾部性能差，尤其在突发流量下 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **ShareGPT**：从公开对话数据集中构建合成工作负载，请求到达建模为泊松过程。
- **BurstGPT**：来自 Azure OpenAI 的真实生产级突发请求轨迹，用于模拟现实中的 bursty behavior。

### 实验设置
- **硬件环境**：6块 NVIDIA A6000 GPU，1个 prefiller + 2个 decoders，通过 Mooncake 进行 KV cache 传输。
- **设备端预测器**：
  - **SP-7B**：基于 Vicuna-7B 的 LoRA 微调版本，部署于 Jetson Orin NX。
  - **SP-160M**：更轻量级变体，适用于资源受限边缘设备。
- **UAC 参数**：滑动窗口大小为最近 500 个请求，目标误覆盖率 $\alpha=0.1$。
- **调度超时**：若校准上界未就绪，最多等待 1 秒后回退至 RR。

### 评估指标
| 指标 | 含义 |
|------|------|
| **E2E Latency (Mean/P99)** | 端到端延迟均值与第99百分位 |
| **TTFT (Time to First Token)** | 首个 token 返回时间 |
| **TPOT (Time Per Output Token)** | 平均每 token 解码耗时 |
| **Max ITL (Inter-Token Latency)** | 最大 token 间隔时间，反映中断严重程度 |
| **Queue Length (time-weighted avg.)** | 时间加权平均队列长度 |

### 基线方法对比
| 基线 | 类型 | 描述 |
|------|------|------|
| **vLLM** | 开源推理引擎 | 使用 PagedAttention 和 RR 路由 |
| **SGLang** | 结构化推理框架 | 支持 RadixAttention，仍用 RR |
| **Llumnix** | 动态调度系统 | 支持运行时迁移（migration），但为反应式机制 |
| **PO-IT**, **S3** | 预测方法 | 作为预测模块的对比基准 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **平均 E2E 延迟降低 42.6%**
- **尾部 E2E 延迟（P99）最高降低 84.8%**
- **最大 ITL 减少达 3倍以上**
- **调度开销仅约 1–2ms**，远低于 decode 推理时间

### 与基线方法的对比结果
| 场景 | 性能表现 |
|------|----------|
| **ShareGPT 工作负载** | MAPS 在所有请求速率下均优于基线，P99 E2E 最多减少 70.3% |
| **BurstGPT 突发负载** | 随着时间缩放加剧，基线尾延迟急剧膨胀，而 MAPS 保持稳定，平均减少 36.8% |
| **TTFT 表现** | MAPS 控制尾部 TTFT 更优；Llumnix 平均更低但尾部更高（因迁移开销） |
| **TPOT 与 Max ITL** | MAPS 的 TPOT 更平稳，Max ITL 显著低于所有基线，表明 token 级中断极少 |

### 消融实验结果
| 消融配置 | 影响 |
|---------|------|
| **SP → PO-IT** | P99 延迟增加 9.2%，说明高质量预测对调度至关重要 |
| **w/o UAC** | P99 廞延增加 14.3%，验证校准对防止低估的关键作用 |
| **MQ → MC（最小队列 vs 最大容量）** | P99 延迟增加高达 1.11×，说明优先考虑队列长度而非剩余内存更有效 |
| **SJF → FCFS** | 移除局部重排序使尾延迟恶化，证明即使全局调度后仍需局部优化 |
| **MAPS-160M（轻量预测器）** | 仍优于所有基线（P99 提升 21–56%），显示 UAC 对低质量预测的鲁棒性 |
| **Oracle（真值长度）** | MAPS 接近 Oracle 性能，相对差距仅 19.8%，说明提升空间有限 |

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Round-Robin 在异构输出长度下失效**：即使请求数量均匀分配，KV cache 利用率也会因输出长度分布偏斜而出现持续不均衡（见附录 A 的 queueing-theoretic 分析）。
2. **预测必须与不确定性管理结合**：单纯提高预测准确率不足以支撑安全调度，**UAC 提供的概率保证是实现低风险决策的基础**。
3. **分层调度优于单一策略**：全局内存感知路由 + 局部 SJF 重排序共同作用，才能同时缓解跨 decoder 队列积压和实例内 HOL blocking。
4. **设备-云协同可用于预测而非仅推理**：本文首次将设备端 LLM 用于 **辅助云端调度决策**，而非直接参与生成，开辟了新的协作范式。

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **部署规模限制** | 当前实验基于小规模集群（6 GPU），大规模扩展性有待验证 |
| **CoT（Chain-of-Thought）场景挑战** | 中间步骤动态生成，总长度难以可靠预测，属于预测模块本身的开放问题 |
| **预测器适应性要求** | 若部署于代码生成等分布迥异的任务，可能需要重新微调，尽管 UAC 可部分缓解 |

### 未来工作方向
- 扩展至更大规模分布式部署，研究中心化调度器的可扩展性。
- 设计针对 CoT 类任务的阶段性长度预测机制。
- 探索多模态输入下的联合资源需求预测。
- 将 MAPS 思路推广至其他内存敏感型服务（如图像生成、语音合成）。

---

> ✅ **总结一句话**：  
> MAPS 通过 **设备端并行预测 + 不确定性校准 + 分层调度**，实现了对 LLM 解码阶段内存需求的“先知式”管理，在几乎零额外延迟开销下，将平均和尾部延迟分别降低 **42.6%** 和 **84.8%**，显著提升了 PD-disaggregated 架构下的服务稳定性与效率。

</details>

---

### 9. [ETCInfer: An Energy-efficient Thermal-aware Cooling-joint Scheduler for LLM Inference in AI Datacenters](https://arxiv.org/abs/2609.15230)

**Authors**: Rui Lu, Rui Ge, Huanghuang Liang, Xiaobo Zhou, Dan Wang  
**Category**: cs.DC  
**Published**: 2026-09-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.15230v1  

#### Abstract
Large language model (LLM) inference in AI datacenters creates a coupled control problem between GPU serving and facility cooling. Raising ambient temperature setpoints can reduce cooling energy and carbon, but also shrinks thermal headroom, induces GPU throttling, and leads to Service-Level-Objecti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ETCInfer: An Energy-efficient Thermal-aware Cooling-joint Scheduler for LLM Inference in AI Datacenters  
**——论文核心总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 AI 数据中心中，**Large Language Model (LLM) 推理任务**面临一个耦合的控制难题：  
- 提高机房环境温度（ambient temperature）可降低 **CRAC**（Computer Room Air Conditioner）能耗，减少碳排放；  
- 但高温会缩小 GPU 的散热余量（thermal headroom），导致 **thermal throttling**（热节流），进而引发 **SLO 违规**（如延迟超标）。  

现有研究通常将 **计算调度** 与 **冷却管理** 分开优化，未能联合考虑 GPU 功耗、频率、微批大小（micro-batch size）与机房温控之间的动态耦合关系。

### 🚀 提出的新方法
本文提出 **ETCInfer** —— 一种能量高效、热感知的联合冷却-计算调度器，其核心创新如下：

#### （1）**统一建模框架**
构建了一个**物理信息驱动的控制模型**，整合了：
- GPU 热生成模型（Heat Generation）
- 机箱空气侧散热模型（Air-side Heat Dissipation）
- CRAC 冷却功耗模型（COP-based）
- LLM 推理延迟模型（Prefill/Decode Latency）

该模型能从实时遥测数据中估计隐藏热状态（如结温）、预测 **time-to-throttle** 和 SLO 风险。

#### （2）**联合决策控制问题建模为 POMDP**
将 **ambient setpoint**（预设）、**GPU frequency**（运行时调整）、**micro-batch size**（运行时调整）三个控制变量联合建模为一个 **Partial Observable Markov Decision Process (POMDP)**，以实现：
- 最小化每任务总能耗（GPU + CRAC）
- 同时满足 **thermal safety** 与 **latency SLO** 约束

#### （3）设计学习型控制器 **ETCAdapter**
- 利用 **belief-state estimation** 处理传感器噪声与延迟；
- 在潜在空间中进行基于模型的强化学习（model-based RL）；
- 引入安全层（safety layer）防止违反约束；
- 支持在线适应 token 长度变化、气流异构等不确定性。

#### （4）系统级实现与部署兼容性
ETCInfer 作为协调层部署于标准推理栈之上（如 vLLM + Kubernetes），无需修改底层引擎，具备良好的工程落地能力。

### 🔍 相比现有方法的优势
| 维度 | 现有方法（如 DSO, TAPAS, DLLM） | ETCInfer |
|------|-------------------------------|----------|
| 控制范围 | 仅 GPU 层（DVFS 或放置） | 联合控制：CRAC setpoint + GPU frequency + micro-batch |
| 热感知 | 忽略隐藏热状态或依赖阈值触发 | 物理建模 + 学习预测 time-to-throttle |
| 冷却协同 | 不参与设施级冷却控制 | 显式建模 CRAC 能耗与 setpoint 关系 |
| 决策机制 | 规则驱动或静态优化 | 学习型自适应控制器（ETCAdapter） |
| 安全保障 | 事后响应 | 前瞻性风险评估 + 安全层干预 |

---

## 2. 核心实验方法和设置

### 📊 数据集与工作负载
使用真实世界 LLM 请求轨迹进行仿真与验证：
- **OASST1**：开放对话数据集（Chat Dialog）
- **PromptSet**：代码辅助任务（Code Assist）
- **WebGPT**：摘要类任务（Summarization）
- **Alibaba2020 Cluster Trace**：生产级集群遥测数据

不同任务具有不同的 **token 统计特征、到达模式（Poisson/Bursty）和 SLO 要求**（见 Table II）。

### ⚙️ 实验设置
#### （1）仿真平台
- **CFD 模拟器 CoolSIM**：用于建模两个典型机房环境：
  - **R1**：强冷热通道隔离，进风均匀
  - **R2**：部分隔离，存在回流，进风温度更高
- 模拟设备：8× NVIDIA H100 GPU + Intel Xeon 8480 CPU 的服务器机架

#### （2）物理测试床（Validation）
- 小规模工作站：Intel i9-13900K + 4× RTX3090 + 4× RTX4090
- 可调进风温度范围：18°C ~ 48°C
- 使用 PT100 温度传感器反馈调节

#### （3）评估指标
| 类别 | 指标 |
|------|------|
| **能效** | 总任务能耗（Total Job Energy）、GPU 计算能耗、CRAC 冷却能耗（均相对于 vLLM-18 基线归一化） |
| **服务质量** | SLO 违规率：<br>• TTFT（Time to First Token）<br>• TPOT（Time per Output Token）<br>• End-to-End 延迟 |
| **硬件安全** | Throttle Exposure Time（超过节流阈值的时间占比） |

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **vLLM-Default** | 固定 ambient setpoint（18/28/38/48°C），默认调度策略 |
| **DSO** | 基于静态程序分析与运行时信号的 GPU 能效优化器（DVFS） |
| **TAPAS** | 热感知与功耗感知调度器，侧重设备安全与功耗效率 |
| **DLLM** | 集群级弹性重构与频率选择的能量控制方法 |
| **GLLM** | 模型剪枝层面的节能方法（energy-aware pruning） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（综合平均）

| 指标 | ETCInfer 表现 | 对比说明 |
|------|---------------|-----------|
| **总任务能耗降低** | **最高达 33.1%** | 在 R2 场景下显著优于所有基线 |
| **SLO 违规率** | **低于 0.7%** | 即使在高达 48°C 的环境下仍保持严格 SLO |
| **Throttle Exposure Time 降低** | **最高达 92.9%** | 极大缓解长期热应力，延长 GPU 寿命 |

#### 详细对比（Fig. 7–13）：
- **Fig. 7**：ETCInfer 在 R1/R2 中实现最大总能耗节省（33.1% vs vLLM-48 的 18.2%）
- **Fig. 8–9**：同时降低计算能耗（≥12.5%）与 CRAC 能耗（↓49.9%）
- **Fig. 10–12**：SLO 违规率接近 vLLM-18 水平（<0.7%），远优于其他高温运行方案
- **Fig. 13**：Throttle 暴露时间降至 1.2%（原为 17.1%），降幅达 92.9%

### 🔬 消融实验结果（Ablation Study）

#### （1）联合控制有效性（Fig. 14）
| 方法 | 能耗 | SLO 违规 | Throttle |
|------|------|---------|--------|
| Setpoint_only | ↑7.9% | ↑40.0% | ↑↑ |
| Freq_only | ↑6.9% | - | ↑↑ |
| Rule-based | 中等 | 中等 | 中等 |
| **Full ETCInfer** | ✅ 最低 | ✅ 最低 | ✅ 最低 |

> 结论：**只有联合控制才能兼顾能效与稳定性**

#### （2）预设 setpoint 策略比较（Fig. 15）
| 策略 | 能耗 | SLO 违规 | Throttle |
|------|------|---------|--------|
| Max_Temp（固定 48°C） | ↓仅 2.6% | ↑1.2× | ↑65.7% |
| Fixed-38°C | ↓1.1–1.3% | ↑59.8% | ↑40.5–63.7% |
| Fixed-28°C | ↑7.9% | ↓12.0% | ↓16.7% |
| **Safe（ETCInfer）** | ✅ 最优平衡 | ✅ 最低 | ✅ 最低 |

> 结论：**固定高温不可取，需动态选择安全 setpoint**

#### （3）SLO 与热安全敏感性（Fig. 16）
| 设置 | 能耗 | SLO 违规 | Throttle |
|------|------|---------|--------|
| Strict_Safety (SS) | ↑4.9% | ↓26.5% | ↓↓ |
| Relaxed_Safety (RS) | ↓3.7% | ↑↑ | ↑↑ |
| No_TTT_Feature (NTF) | ❌ 全面恶化 | ❌ | ❌ |

> 结论：**time-to-throttle 特征至关重要**

#### （4）与其他模型控制方法对比（Table III）
| 方法 | 平均节能 | SLO 违规率 | Throttle 减少 |
|------|----------|------------|--------------|
| Greedy Threshold | 18.6% | 1.84% | 54.3% |
| Deterministic Optimization | 22.1% | 1.26% | 63.5% |
| MPC | 25.4% | 0.92% | 72.8% |
| Robust MPC+Filter | 27.6% | 0.74% | 78.9% |
| **ETCInfer** | **31.8%** | **0.48%** | **89.7%** |

> 结论：**ETCInfer 在能效、SLO、安全性三者间取得最佳平衡**

#### （5）模型泛化能力（Table V）
在不同 LLM 架构上验证一致性提升：
| 模型 | 节能 | SLO 违规 | Throttle 降低 |
|------|------|----------|-------------|
| Qwen2.5-Instruct | 30.4% | 0.43% | 88.6% |
| DeepSeek-R1-Distill-Qwen | 28.7% | 0.51% | 85.2% |
| Mistral-Instruct | 26.9% | 0.47% | 82.4% |

> 表明效果不依赖特定模型架构

#### （6）预测准确性（Table IV）
延迟预测误差极低，支持前瞻决策：
| 工作负载 | TTFT MAPE | TPOT MAPE | E2E MAPE |
|--------|-----------|-----------|----------|
| Chat Dialog | 1.7% | 2.1% | 1.5% |
| Code Assist | 2.2% | 2.5% | 1.9% |
| Summarization | 2.4% | 2.8% | 2.2% |

> 所有 MAPE < 3%，足以支撑安全调度决策

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **联合控制是未来 AI 数据中心节能的关键路径**：  
   单独优化计算或冷却无法突破瓶颈，必须打通 **GPU-DVFS、micro-batch、CRAC setpoint** 的联合控制闭环。

2. **物理建模 + 学习控制 是解决复杂热耦合的有效范式**：  
   ETCInfer 通过物理先验建模隐藏热状态，并结合学习型控制器在线适应不确定性，实现了高鲁棒性。

3. **可在高达 48°C 的环境中安全运行 LLM 推理**：  
   传统认为高温危险，但通过主动热管理（如降频、减批），反而可大幅节能而不牺牲 SLO。

4. **time-to-throttle 是比当前温度更重要的调度信号**：  
   提前预测节流风险，比被动响应更有效避免性能骤降。

### ⚠️ 方法局限性
1. **依赖一定精度的遥测数据**：若功率/温度采样延迟严重或丢失，会影响预测质量。
2. **未直接控制风扇转速或液冷泵**：假设机箱风扇满速运行，未来可扩展至更细粒度冷却控制。
3. **未考虑跨 Pod 的通信拥塞影响**：当前 focus 在单 Pod 内部调度。
4. **CFD 模拟成本较高**：虽不用于在线推理路径，但在大规模部署时可能影响 setpoint 规划速度。

### 🔮 未来工作方向
1. 扩展至 **多房间、多区域 CRAC 协同控制**
2. 引入 **液冷系统建模与控制接口**
3. 结合 **carbon intensity forecast** 实现碳感知联合调度
4. 探索 **端到端 trainable scheduler** 替代分阶段训练
5. 在更大规模生产集群中进行长期部署验证

---

> **开源声明**：作者已公开发布 ETCInfer 源码，链接：[https://anonymous.4open.science/r/ETCInfer-2761/](https://anonymous.4open.science/r/ETCInfer-2761/)  
> **一句话总结**：ETCInfer 首次实现了 LLM 推理中“算-冷”联合最优控制，在高达 48°C 的环境下仍能 **节能 33.1%、降低节流暴露 92.9%、SLO 违规 <0.7%**，为绿色 AI 数据中心提供了新范式。

</details>

---

### 10. [Graph Neural Networks for Influence Maximization in Social Networks: An Unsupervised Minimum Dominating Set Approach](https://arxiv.org/abs/2609.13836)

**Authors**: Erfan Ahmadi, Mina Shirazi, Behnam Bahrak  
**Category**: cs.LG  
**Published**: 2026-09-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.13836v1  

#### Abstract
The Minimum Dominating Set (MDS) problem is a classic NP-hard combinatorial optimization problem with critical applications in social network analysis, including viral marketing, influence maximization, public health interventions, and information dissemination. Identifying a minimal set of influent...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Graph Neural Networks for Influence Maximization in Social Networks: An Unsupervised Minimum Dominating Set Approach

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文聚焦于**社会网络中的影响力最大化（Influence Maximization）**问题，其本质可建模为图论中的**最小支配集（Minimum Dominating Set, MDS）**问题。目标是识别一个最小节点集合 $D$，使得网络中每个节点要么在 $D$ 中，要么与 $D$ 中至少一个节点相邻。该问题在病毒式营销、公共卫生干预、紧急信息传播等场景具有重要应用。

然而，MDS 是 NP-hard 问题，传统启发式算法效率低且难以泛化；而基于学习的方法通常依赖昂贵的最优解作为监督信号，在大规模真实网络中不可行。

---

### 提出的新方法与创新思路

作者提出了一种**无监督的图神经网络框架（Unsupervised GNN Framework）**来求解 MDS 问题，核心创新如下：

#### ✅ 创新点 1：完全无监督训练（Unsupervised Training）
- **无需 ground-truth 最小支配集进行训练**，仅通过设计可微分的损失函数直接优化目标（最小化集合大小 + 保证全覆盖），显著降低对标注数据的依赖。
- 特别适用于现实世界中无法获取最优解的大规模网络。

#### ✅ 创新点 2：新型多目标概率损失函数（Novel Probabilistic Loss）
引入由三项组成的复合损失函数：
- **Size Loss**：惩罚选中节点的概率总和，鼓励更小的支配集；
- **Coverage Loss**：惩罚未被覆盖节点的概率，确保所有节点都被支配；
- **Equalizer Loss**：惩罚接近 0.5 的中间概率值，促使模型输出“二值化”决策（即 $p \to 0$ 或 $1$），提升解码质量。

> 权重设定为 $\alpha=70$, $\beta=2$, $\gamma=50$，经验证能有效平衡各目标。

#### ✅ 创新点 3：集成多策略解码机制（Ensemble Multi-Strategy Decoding）
推理阶段采用四种不同的解码策略并取最优结果：
1. **Three-Phase Strategy**：结合异常检测、随机采样与贪心补全；
2. **Prune-Greedy Strategy**：初始贪心选择 + 叶子替换 + 冗余剪枝；
3. **Threshold-Greedy Strategy**：高置信度阈值筛选 + 贪心补全；
4. **Neighbor-Greedy Strategy**：允许已覆盖节点被选以增强邻域覆盖能力。

最终输出最小的合法支配集，充分利用不同策略在不同图结构上的优势。

#### ✅ 创新点 4：单次前向传递高效推理（Global Prediction Paradigm）
整个 GNN 模型只需一次前向传播即可生成所有节点的选择概率，配合快速解码，实现极高的推理速度。

---

### 相比现有方法的优势

| 维度 | 本方法 | 现有方法 |
|------|--------|---------|
| **训练范式** | 无监督（无需标签） | 多数需监督（如 Kothapalli et al.）或强化学习 |
| **推理速度** | 极快（<50ms / 图） | 动态贪心较慢，模拟退火极耗时 |
| **泛化能力** | 在合成数据上训练，成功迁移到多种真实社交网络 |
| **实用性** | 适合实时、大规模部署 | 很多方法计算成本过高 |

---

## 2. 核心实验方法和设置

### 数据集

#### 📦 训练数据
- **12,000 个合成连通图**，节点数 $n \in [10, 40]$，混合四种结构类型：
  - **Trees (15%)**：稀疏树状结构；
  - **Medium-density ER (70%)**：边概率 $p = 1.5\log n / n$ 的 Erdős–Rényi 图；
  - **Dense ER (10%)**：$p = 0.35$；
  - **Near-complete (5%)**：完全图移除约 2% 边。
- 所有图均连通，最优 MDS 使用 OR-Tools CP-SAT 求解器离线计算，**仅用于评估，不参与训练**。

#### 🔍 测试数据（Benchmark Instances）
7 个经典真实社会网络（转换为简单无向图）：
1. `dolphins`（海豚社交网络）
2. `florentine families`（佛罗伦萨家族婚姻关系）
3. `high tech company`（高科技公司管理者关系）
4. `highschool`（高中生友谊）
5. `Les Misérables`（小说人物共现）
6. `Klas12b`（课堂学生友谊）
7. `macaques`（日本猕猴等级互动）

> 原始可能为有向/加权/多层图 → 统一处理为无向简单图。

---

### 实验设置与评估指标

#### 评估指标
- **相对误差（Relative Error）**：
  $$
  \text{RelError} = \frac{|D_{\text{pred}}| - |D_{\text{opt}}|}{|D_{\text{opt}}|}
  $$
- **推理时间（Inference Time）**：毫秒每图（ms/graph）

#### 模型架构
- **GNN Backbone**：两层 GCN（GAT 表现较差故弃用）
- **输入特征**：节点度 + 二进制指示符
- **输出层**：MLP 接 Sigmoid 输出选择概率 $p_i \in (0,1)$
- **优化器**：Adam，学习率 $5\times10^{-3}$，训练 800 轮，batch size=16

---

### 基线方法对比

#### 经典启发式方法
| 方法 | 描述 |
|------|------|
| `Trivial-Greedy` | NetworkX 默认贪心，任意顺序选择 |
| `Static-Greedy` | 按初始度排序，依次选最高度未覆盖节点 |
| `Dynamic-Greedy` | 每步重新计算边际增益（覆盖最多新节点者入选） |
| `Simulated Annealing (SA)` | 元启发式，从 Static-Greedy 出发搜索改进 |

#### 学习类方法
| 方法 | 类型 | 是否监督 |
|------|------|----------|
| `EGN-MDS` [Karalias & Loukas, 2020] | 无监督 GNN（Erdős Goes Neural） | ❌ |
| `Kothapalli et al. (GCN-MDS)` [2023] | 监督式 GCN + 迭代贪心 | ✅ |

> 注：排除 S2V-DQN 和 GCON 因实现困难或基准不一致。

---

## 3. 主要实验结果和性能指标

### 性能汇总（平均相对误差）

| 方法 | 平均 RelError (%) | 是否最优解 |
|------|------------------|------------|
| Trivial-Greedy | 115% | 否 |
| Static-Greedy | 33% | 否 |
| **Dynamic-Greedy** | **6%** | 部分 |
| Simulated Annealing | **0%**（全部最优） | ✅ |
| EGN-MDS | 170% | 否 |
| **Kothapalli et al.** | **0%**（全部最优） | ✅ |
| **Ours (Proposed)** | **7%** | 4/7 最优，其余差 1–3 节点 |

> ➤ 我们的方法达到近似最优水平，仅次于 SA 和监督方法。

---

### 推理速度对比（平均 ms/graph）

| 方法 | 时间范围 | 对比倍数 |
|------|--------|---------|
| Trivial/Static-Greedy | <0.1 ms | ⏱️ 最快但精度差 |
| Dynamic-Greedy | 0.04 – 6.05 ms | 快但随图增长变慢 |
| **Simulated Annealing** | **226 – 1616 ms** | ❌ 慢 10–55× |
| **Kothapalli et al.** | **52 – 618 ms** | ❌ 慢 2–14× |
| **Ours** | **20.9 – 46.0 ms** | ✅ **最快的学习类方法** |

> ✅ 在保持高质量的同时，**推理速度快于所有其他学习方法 2–14 倍**。

---

### 消融实验关键发现

#### ❌ EGN-MDS 基线失败说明：
- 即使同属无监督 GNN 框架，**缺少 Equalizer Loss 和 Ensemble Decoding** 导致：
  - 概率分布模糊（集中在 0.5 附近）；
  - 解码困难，平均误差高达 **170%**；
- ➤ 验证了本文提出的 **Equalizer + Ensemble 是性能飞跃的关键**。

#### ✔ Ensemble 解码有效性
- 单独运行任一解码策略表现不稳定；
- 集成后稳定获得最小解，体现互补性：
  - Threshold-Greedy 在高置信图上优秀；
  - Prune-Greedy 在稀疏图上去冗余强；
  - Three-Phase 在中密度图上综合表现好。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **无监督 GNN 可有效学习 MDS 启发式规则**  
   尽管没有见过任何最优解，模型仍能从合成图中学习到通用结构模式，并迁移到多样化的**真实社会网络**。

2. ✅ **Global Prediction + Ensemble 解码可在质量和速度间取得良好平衡**  
   单次前向传递 + 多策略解码的设计实现了**接近 Dynamic-Greedy 的精度**，同时具备**远超迭代方法的速度潜力**。

3. ✅ **Equalizer Loss 至关重要**  
   强制模型做出明确决策（非模糊概率），极大提升了后续解码的有效性。

4. ✅ **相比监督方法更具实用价值**  
   虽然 Kothapalli 方法达到最优，但其依赖大量精确标签，训练成本极高；而本文方法**免去了这一瓶颈**，更适合实际部署。

---

### 局限性

1. ❗ **独立性假设限制密集图表现**  
   Coverage Loss 假设节点选择相互独立，在高度相关或极端稠密图中可能失效。

2. ❗ **当前训练规模较小（n ≤ 40）**  
   模型未在千级节点以上训练，**外推至超大图（如万人社交网络）的表现尚待验证**。

3. ❗ **未考虑动态或加权图结构**  
   当前方法作用于静态、无权、单层图，难以直接应用于复杂现实网络（如带时间戳、多关系的社会平台图）。

4. ❗ **Ensemble 增加推理开销**  
   虽然仍很快，但运行四个解码器比单一策略慢，对延迟极度敏感的应用需权衡。

---

### 未来工作方向

1. 🔮 **扩展至 PIDS（Positive Influence Dominating Set）**
   修改 Coverage Loss 以支持“阈值影响”模型（如需半数邻居激活才受影响），更贴近真实社会影响力机制。

2. 🔁 **研究 Connected Dominating Set（CDS）**
   要求选出的节点形成连通子图，适用于构建虚拟骨干网（如 MANETs）。

3. 🔄 **探索迁移学习范式**
   在合成图上预训练 + 少量真实网络微调，缩小合成与真实之间的分布差距。

4. 🏗️ **改进可扩展性架构**
   引入 hierarchical GNN、graph coarsening 或 attention 机制，提升对万级节点图的支持能力。

5. 📐 **理论分析近似比**
   探索所学启发式的理论保证，例如是否能达到 $O(\log \Delta)$ 近似比。

---

> 💬 **总结一句话**：  
> 本文提出了一种**无需标签、高速高效、泛化性强**的 GNN 方法解决社会网络中的影响力最大化问题，通过**创新的无监督损失函数与集成解码机制**，在真实网络上实现了接近最优的质量，且推理速度领先现有学习方法达 **14 倍以上**，为大规模应用场景提供了可行路径。

</details>

---

### 11. [Cloud Workflow Scheduling Based on Graph Attention-Driven Hierarchical Reinforcement Learning](https://arxiv.org/abs/2609.14952)

**Authors**: Zongjin Li, Shaohan Feng, Chunxi Yang, Wenbo Wang  
**Category**: cs.LG  
**Published**: 2026-09-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.14952v1  

#### Abstract
Dynamic cloud workflow scheduling must balance deadline satisfaction, container utilization, and energy consumption while dealing with stochastic task-execution speeds, placement-dependent communication, and coupled task and container decisions. Workflows are naturally modeled as directed acyclic gr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Cloud Workflow Scheduling Based on Graph Attention-Driven Hierarchical Reinforcement Learning*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
动态云工作流调度需要在满足截止时间（deadline）、提高容器利用率和降低能耗之间进行权衡。传统方法面临以下挑战：
- 工作流以有向无环图（DAG）形式存在，其任务依赖结构复杂，而常规的向量或矩阵状态表示无法充分捕捉拓扑关系；
- 任务执行速度具有随机性（stochastic task-execution speeds），且通信延迟依赖于容器部署位置（placement-dependent communication）；
- 任务调度与容器创建/放置决策相互耦合，难以解耦优化。

### 提出的新方法与创新思路
作者提出了一种**基于图注意力驱动的分层强化学习框架（Graph Attention-Driven Hierarchical Reinforcement Learning, GA-HRL）**，其核心创新包括：

- **依赖感知的任务表示（Dependency-aware Task Representation）**  
  引入预测子截止时间（predicted sub-deadlines）来量化任务紧迫性，并利用多头图注意力网络（multi-head GAT）聚合DAG中的结构信息，使模型能区分具有相似属性但处于不同拓扑位置的任务。

- **事件驱动的分层SMDP建模（Event-driven Hierarchical SMDP）**  
  将调度过程建模为半马尔可夫决策过程（Semi-Markov Decision Process, SMDP），由两类事件触发：工作流到达和任务完成。在每个事件中：
  - **Task Scheduling (TS) Agent**：先处理就绪任务，决定复用现有容器或请求新容器；
  - **Container Scheduling (CS) Agent**：随后处理所有新容器的主机部署决策；
  - 决策按顺序执行，环境仅在两阶段完成后推进时间。

- **交替训练机制（Alternating Training with PPO）**  
  两个Agent使用独立的Proximal Policy Optimization（PPO）算法进行交替训练，在共享环境中协同演化策略，实现解耦但协调的优化。

### 相比现有方法的优势
- 更好地保留并利用了DAG的非欧几里得结构信息；
- 显式建模了任务与容器之间的耦合决策，避免贪心或串行决策带来的次优；
- 在不确定性和动态负载下仍保持高鲁棒性；
- 不依赖重复搜索或昂贵求解器，适用于在线调度场景。

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **2018 Alibaba cluster trace** 构建仿真环境；
- 从中提取 **5,200个包含至少10个任务的工作流DAG**；
- 模拟真实工作流的到达时间、结构多样性及资源需求分布。

### 实验设置
- **平台配置**：
  - 支持两种主机类型（Type 1 和 Type 2），提供不同的CPU、内存、计算能力和功耗；
  - 容器类型共8种，对应不同CPU核数（1–32）和内存大小（4–128 GB）；
- **通信带宽**：
  - 同主机内通信带宽（intra-host bandwidth）：500 MB/s；
  - 跨主机通信带宽（inter-host bandwidth）：200 MB/s；
- **任务执行不确定性**：
  - 执行容量 $ Q_{k,j}^{(m)} \sim \mathcal{N}(Q_m, (\nu Q_m)^2) $，其中 $\nu$ 为变异系数（0 到 0.45）；
- **调度事件驱动**：由工作流到达和任务完成触发；
- **训练方式**：
  - 使用 PyTorch 实现；
  - 采用 PPO 算法，折扣因子 $\gamma=0.99$，GAE 参数 $\lambda=0.95$；
  - 每轮收集 2048 步 rollouts，进行 2 轮优化更新；
  - TS 与 CS Agent 交替训练。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Workflow Success Rate (%)** | 成功在截止时间前完成的工作流占比 |
| **Average Container Resource Utilization (%)** | 容器生命周期内的平均资源占用率 |
| **Total System Energy Consumption (J)** | 整个系统运行期间的总能耗（含静态与动态能耗） |

### 基线方法对比
- **DTODRL** [10]：基于GNN的深度强化学习方法；
- **OHDS** [6]：能量高效的启发式调度器；
- **SMWDSA** [1]：SHWS系统中的确定性启发式；
- **HACPPO** [28]：边缘云环境下基于PPO的任务调度；
- **DS-CSP** [13]：基于切割库存模型的服务整合方法；

此外还进行了消融实验（ablation study）验证各组件作用。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table III 及 Figures）

#### （1）总体性能表现（K=100, ν=0.1, α_df=2.1）
| 方法 | 成功率 (%) | 容器利用率 (%) | 能耗 (J) |
|------|------------|----------------|----------|
| **GA-HRL** | **100** | **65** | **1.84×10⁷** |
| DTODRL | 100 | 61–64 | 2.14–2.32×10⁷ |
| HACPPO | ~98 | ~65 | >2.4×10⁷ |
| OHDS | ~85 | ~58 | ~2.8×10⁷ |
| SMWDSA | ~70 | <50 | ~3.09×10⁷ |
| DS-CSP | ~50 | ~45 | ~3.5×10⁷ |

> ✅ **GA-HRL 在成功率持平的情况下，实现了更高的容器利用率和显著更低的能耗。**

#### （2）面对高执行速度变化（ν=0.45）的表现
- 所有方法成功率下降，但 GA-HRL 表现出更强鲁棒性：
  - GA-HRL 成功率：**82%**
  - DTODRL 最高：**86%**
  - SMWDSA / DS-CSP 分别降至 47% / 5%
- 能耗方面：
  - GA-HRL 能耗：**2.32×10⁷ J**
  - DTODRL 能耗：**3.27×10⁷ J**
  - ➜ **GA-HRL 牺牲约4%的成功率，换取近29%的节能优势**

> 🔍 这体现了其“**以小成功代价换大幅节能**”的设计哲学，适合对能耗敏感的应用场景。

#### （3）随工作流数量增加（K从100到1000）的扩展性
- GA-HRL 和 DTODRL 维持接近 **100% 的成功率**；
- GA-HRL 的容器利用率从 **66% 上升至 70%**，表明其具备良好的容器复用能力；
- 总能耗增长最平缓：
  - K=1000 时，GA-HRL 消耗 **15.35×10⁷ J**
  - DTODRL：17.07×10⁷ J
  - SMWDSA：22.53×10⁷ J

> 📈 表明 GA-HRL 能通过高效资源复用应对更高密度负载。

### 消融实验结果（Ablation Study）
| 方法 | 成功率 (%) | 利用率 (%) | 能耗 (J) |
|------|------------|------------|----------|
| **GA-HRL** | 100 | 65 | 1.84×10⁷ |
| M/A2C（替换PPO） | 97 | 64 | 1.98×10⁷ |
| M/DDQN（替换PPO） | 95 | 66 | 2.02×10⁷ |
| **M/f-GAT（移除GAT）** | 100 | 65 | **2.03×10⁷** |
| **M/f-CS（随机容器放置）** | 92 | **70** | **2.19×10⁷** |

> 🔍 发现：
> - 移除 GAT 导致能耗上升 → **证明图注意力对节能有效**；
> - 移除 CS Agent（随机放置）导致成功率下降 → **说明智能容器放置对保障SLA至关重要**；
> - 两者互补：GAT 提升任务级决策质量，CS Agent 协调资源局部性与整合效率。

---

## 4. 关键结论和发现

### 主要发现
1. **图结构建模显著提升调度质量**：将DAG作为原始输入并通过GAT编码，能够更准确表达任务间的依赖与上下文，优于扁平化特征表示。
2. **分层决策优于端到端联合决策**：将任务分配与容器部署解耦为两个Agent，既降低了动作空间复杂度，又允许分别优化目标。
3. **GA-HRL 实现多目标平衡**：
   - 在多数情况下保持与最优方法相当甚至更高的成功率；
   - 显著优于其他方法的**容器利用率**和**能源效率**；
   - 特别是在高不确定性环境下，愿意牺牲少量成功率以换取巨大节能收益，体现其**不确定性感知设计的有效性**。
4. **容器重用是节能的关键路径**：随着工作流密度上升，GA-HRL 自适应增加容器复用，从而减少新建开销和主机激活频率。

### 方法的局限性
- 当前模型未显式建模资源故障或在线性能估计误差；
- 假设任务计算量和数据传输量已知，不适用于完全未知工作负载；
- GAT 编码虽有效，但在超大规模DAG上可能存在计算瓶颈；
- 实验基于trace回放模拟，尚未在物理测试床验证实际部署效果。

### 未来工作方向
- 将 GA-HRL 部署至真实物理测试床，结合实测干扰数据进行闭环优化；
- 引入在线性能预测模块，动态调整执行容量估计；
- 扩展支持资源故障恢复与弹性伸缩机制；
- 探索轻量化GAT结构以适配更大规模调度场景。

---

> 💡 **总结一句话**：  
> GA-HRL 通过 **GAT + 分层PPO + 事件驱动SMDP** 的设计，在动态、不确定的云环境中实现了**高成功率、高资源利用率与低能耗**的统一，尤其适合对能效敏感的大规模工作流调度场景。

</details>

---

### 12. [Reason What Matters: Retrieval-Grounded Reasoning for Universal Multimodal Embeddings](https://arxiv.org/abs/2609.15296)

**Authors**: Mingzhou Jiang, Peixi Wu, Hang Cheng, Yunhao Zhou, Biao Yang, Wei Yuan, Yun Li, Fan Yang, Wenwu Ou, Honghui He  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.15296v1  

#### Abstract
Universal multimodal embedding (UME) learns unified representations across modalities, enabling a single model to support diverse retrieval tasks. Recent methods use Chain-of-Thought (CoT) reasoning to better interpret multimodal inputs before generating embeddings for complex retrieval tasks and fu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Reason What Matters: Retrieval-Grounded Reasoning for Universal Multimodal Embeddings**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 **Chain-of-Thought (CoT)** 的 **Universal Multimodal Embedding (UME)** 方法在提升检索性能方面取得了进展，但仍存在两大瓶颈：

1. **粗粒度信用分配（Coarse Credit Assignment）**：  
   现有方法如 **GRPO** 将所有 CoT token 视为同等重要，使用统一的标量优势（advantage）进行策略优化，无法识别哪些推理步骤是输入支持的、真正有助于区分正负样本的关键证据。

2. **高推理延迟（High Inference Latency）**：  
   即使部分 CoT 已提供足够检索证据，仍需生成完整推理链，导致在大规模语料库部署时效率低下。

---

### **提出的新方法：ReWAM**
本文提出 **Reason What Matters (ReWAM)**，一个**检索引导的显式推理框架**，从**训练监督**和**推理控制**两个层面联合优化 UME 的质量与效率。

#### **核心创新点**

- ✅ **Retrieval-aware Self-Distillation (RASD)**  
  引入一种**检索感知的自蒸馏机制**，通过对比正样本与难负样本（hard negatives），构建特权指导信息（Contrastive Evidence Privileged Information, CEPI），用于验证 CoT 中的声明是否被输入支持、是否具有判别性。  
  利用该信息对每个 token 进行细粒度信用赋值，实现**token-level 的监督信号调制**，聚焦于“真正重要的推理”。

- ✅ **Retrieval-adaptive Inference (RAI)**  
  设计一种**自适应推理机制**：
  - 使用 **retrieval confidence head** 预测当前部分 CoT 的**剩余检索效用**（remaining retrieval utility），决定是否提前截断。
  - 结合 **speculative decoding** 加速 token 生成，降低解码成本。

---

### **相比现有方法的优势**

| 方面 | 传统方法（如 Embed-RL） | ReWAM |
|------|--------------------------|--------|
| **信用分配** | 统一轨迹级优势（trajectory-level），忽略 token 差异 | 细粒度 token-level 调制，聚焦关键推理 |
| **推理长度** | 固定长度，必须生成完整 CoT | 动态截断，仅保留必要推理 |
| **解码速度** | 自回归逐 token 生成 | speculative decoding 并行验证多个 draft token |
| **性能 vs 效率** | 性能好但慢 | **SOTA 性能 + 最高吞吐量（达 5×）** |

> ReWAM 成功弥合了**高质量显式推理**与**高效大规模部署**之间的鸿沟。

---

## **2. 核心实验方法和设置**

### **使用的数据集**

- **MMEB-V2**：通用多模态检索基准，涵盖 **78 个数据集**，分为：
  - 图像（36）
  - 视频（18）
  - 视觉文档（VisDoc, 24）
- **MRMR**：**高难度、强推理型**多模态检索基准，包含 11 个子任务，覆盖：
  - 知识类（Knowledge）
  - 定理类（Theorem）
  - 矛盾检测类（Contradiction）

---

### **实验设置与评估指标**

| 项目 | 设置 |
|------|------|
| **Embedder 模型** | Qwen2-VL (2B/7B), Qwen3-VL (2B/4B) |
| **Reasoner 模型** | Qwen3-VL-8B-Instruct |
| **Analyzer 模型** | Qwen3.5-122B-A10B API（用于生成 CEPI） |
| **训练方式** | 两阶段：先 SFT 训练 embedder → 冻结后训练 reasoner with RASD |
| **评估指标** | - MMEB-V2: Hit@1（图像/视频）、NDCG@5（视觉文档）<br>- MRMR: NDCG@10（除 Negation 外），Negation 用 Hit@1<br>- 总体得分：各数据集平均 |
| **硬件平台** | 单张 H800 GPU（用于吞吐量测试） |

---

### **基线方法对比**

#### **标准多模态嵌入模型**
- EVA-CLIP, OpenCLIP, ColPali, GME, VLM2Vec, VLM2Vec-V2, CAFe, UME-R1

#### **推理增强型 UME**
- **RIME**, **Embed-RL**, **UMER-H**, **TTEs**（均为显式 CoT）
- **PLUME**（隐式 latent reasoning，作为效率基线）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **MMEB-V2 结果（Table 1）**
- **ReWAM (Qwen3-VL-4B)** 达到 **68.7 Overall Score**，为所有方法中最高。
- 显著优于同规模显式 CoT 方法：
  - 超越 **Embed-RL (Qwen3-VL-4B)**（68.1 → 68.7）
  - 超越 **RIME (Qwen2-VL-7B)**（64.5 → 69.0）
- 在 **Qwen2-VL-2B** 上超越 **VLM2Vec-V2** 8.5 分，超越 **PLUME** 4.9 分。

#### **MRMR 结果（Table 2）**
- **ReWAM-4B** 在强推理任务上表现尤为突出：
  - **Overall: 53.3**（SOTA）
  - 超越最强基线 **RIME**（50.2）和 **LaME**（49.8）
- 在 **Theorem 类任务**全面领先：
  - 数学（Math）: +7.3 pts
  - 物理（Physics）: +6.6 pts
  - 工程（Engineering）: +6.7 pts
- 在 **Traffic Contradiction** 上达到 **53.5**，提升 **7.7 pts**。

---

### **与基线方法的对比结果**

| 对比维度 | 结果 |
|---------|------|
| **vs Embed-RL (decoupled CoT)** | 性能更高（↑0.6–1.3 pts），吞吐量达其 **5×** |
| **vs PLUME (latent reasoning)** | 性能显著更高（↑4.9 pts），同时吞吐量也更高（见 Fig 1e） |
| **vs RIME / UME-R1** | 在所有模态均取得 SOTA，尤其在视频和视觉文档上优势明显 |

> ReWAM 是**唯一同时在性能和效率上超越 latent reasoning 方法的显式 CoT 框架**。

---

### **消融实验结果（Ablation Studies）**

#### **Table 3：核心组件消融**
| 变体 | Overall Score | 相对下降 |
|------|---------------|----------|
| ReWAM（完整） | **66.5** | — |
| w/o RASD | 65.7 | ↓0.8 |
| w/o CoT | 62.3 | ↓4.2 |
| w/ raw input | 63.1 | ↓3.4 |

✅ 显式 CoT 和 RASD 均带来显著增益。

#### **Table 4：RASD 调制强度（λ_R）影响**
- 即使弱调制（λ_R=0.1）也能提升性能（66.0 → 66.2）
- 最优在 λ_R=0.5（66.5），证明 token-level 指导有效且鲁棒。

#### **Table 5：CEPI 组件逐步添加**
| 添加组件 | Overall |
|---------|--------|
| 仅轨迹级 | 66.0 |
| + Grounded evidence | 66.3 |
| + Decision boundary | 66.4 |
| + Claim verification（完整 CEPI） | **66.5** |

✅ 所有 CEPI 组件均贡献正向增益。

#### **Table 6 & Fig 3：RAI 效率分析**
| 方法 | 吞吐量（samples/s） | CoT 长度 | 性能变化 |
|------|------------------|------------|----------|
| Baseline | ~1.1–1.2 | ~130–140 | — |
| Speculative only (K=7) | ~2.5–2.9 | 无缩短 | ≈持平 |
| Full RAI (K=7) | **3.3–4.5** | ↓36% | ≈持平 |

✅ RAI 实现 **最多 4.5× 吞吐提升**，且性能几乎无损。

---

## **4. 关键结论和发现**

### **主要发现**

1. ✅ **显式 CoT 可以既快又准**：  
   通过 RASD 和 RAI，ReWAM 证明了**显式推理**不仅能获得最佳性能，还能实现**远超现有方法的推理效率**。

2. ✅ **细粒度监督至关重要**：  
   RASD 通过 CEPI 实现 token-level 信用赋值，显著优于 GRPO 的统一优势信号，尤其在 OOD 和难样本上更鲁棒。

3. ✅ **自适应推理大幅提升实用性**：  
   RAI 能准确预测“何时停止”，避免冗余推理，在保持性能的同时将延迟降低数倍。

4. ✅ **ReWAM 在强推理任务上优势显著**：  
   在 MRMR 上大幅领先，表明其特别适合需要领域知识和逻辑推理的复杂检索场景。

---

### **方法的局限性**

- **依赖外部 analyzer**：CEPI 构建依赖大模型（如 Qwen3.5-122B），增加训练成本。
- **RAI 依赖离线轨迹训练**：draft model 和 confidence head 需大量预生成 CoT 数据。
- **未完全消除幻觉**：尽管 RASD 减少错误推理，但无法完全杜绝 MLLM 的固有幻觉问题。

---

### **未来工作方向**

- 探索轻量化 analyzer 或自生成 CEPI，降低训练开销。
- 将 RAI 扩展至其他推理任务（如 VQA、多跳推理）。
- 结合 retrieval-augmented generation（RAG）进一步增强证据可靠性。
- 研究在线自适应调整 draft block size 和 stopping threshold。

---

> **总结**：ReWAM 为**大规模部署高质量显式推理 UME** 提供了一条可行路径，实现了 **“reasoning that matters”** —— 只做必要的、有效的、可解释的推理。

</details>

---

### 13. [A Multi-Resolution Multi-Domain Pre-Training Framework for Universal Traffic Forecasting](https://arxiv.org/abs/2609.13878)

**Authors**: Zhouyang Liu, Jindong Han, Hao Wang, Xinyue Liu, Hui Gao, Dongsheng Li, Hao Liu  
**Category**: cs.LG  
**Published**: 2026-09-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.13878v1  

#### Abstract
Spatio-temporal traffic data are central to intelligent transportation systems, yet their heterogeneity poses significant challenges for large-scale modeling. Existing pre-trained models often rely on a homogeneous modeling paradigm to handle highly heterogeneous traffic data. This fundamental misma...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Multi-Resolution Multi-Domain Pre-Training Framework for Universal Traffic Forecasting

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **pre-trained spatio-temporal models** 在处理城市交通时间序列时普遍采用**同质化建模范式**（homogeneous modeling），即用统一架构和参数空间处理高度异构的多域交通数据。然而，真实交通数据在以下三方面存在显著异质性：
- **时空分辨率不一致**（如采样间隔从分钟到小时，空间粒度从路段到区域）
- **多领域差异**（如速度、流量、需求等任务类型，以及不同城市的道路网络与出行行为）
- **周期模式不一致**（尽管普遍存在日/周周期，但相位偏移、振幅变化、周期长度不一）

这种“以不变应万变”的方式导致**负迁移**（negative transfer）、计算资源浪费和泛化能力差。

---

### 提出的新方法：FlexST
作者提出 **FlexST** ——一个模块化、自适应的预训练框架，专为解决多分辨率、多领域的通用交通预测设计。其三大核心组件如下：

#### （1）Multi-resolution Spatio-temporal Diffusion Module
- **功能**：显式捕捉跨不同时间和空间分辨率的交通动态。
- **实现**：
  - 多分辨率分块（multi-resolution patching）：将输入时间序列按不同时间跨度切片，对齐绝对时间跨度而非步数。
  - 多尺度图扩散生成（multi-resolution graph generation）：基于热扩散方程构建多个传播矩阵 $ D_i $，模拟从局部到全局的空间信息流动。

#### （2）Domain-adaptive Mixture-of-Experts (MoE)
- **功能**：动态路由数据至专用子网络，隔离冲突模式，促进选择性知识迁移。
- **实现**：
  - **Temporal Decoupling Experts (TDE)**：分为任务门控专家（task-gated）、分辨率门控专家（resolution-gated）和共享专家（shared expert），分别处理特定任务/分辨率特征与通用模式。
  - **Spatial Decoupling Experts (SDE)**：每个专家绑定一个扩散矩阵 $ D_i $，通过路由机制选择最合适的空间传播方式。

#### （3）Unified Periodic Encoding Strategy
- **功能**：注入分辨率和领域感知的周期先验，协调跨数据集的周期不一致性。
- **实现**：
  - 融合任务类型 $ Q $、时间跨度 $ S $ 和时间戳（tod/dow）作为上下文编码。
  - 使用可学习嵌入层，并预留占位符以支持未见任务或分辨率，提升鲁棒性。

---

### 相比现有方法的优势
| 维度 | 传统方法局限 | FlexST优势 |
|------|--------------|-----------|
| **建模灵活性** | 固定结构，难以适配异构数据 | 模块化解耦，灵活组合 |
| **知识迁移效率** | 强制共享参数易引发负迁移 | MoE稀疏激活，仅调用相关专家 |
| **周期建模能力** | 假设固定周期或独立学习 | 统一编码策略实现跨域周期对齐 |
| **可扩展性** | 难以应对大规模图 | 利用全局摘要降低空间扩散成本 |

---

## 2. 核心实验方法和设置

### 数据集
在 **23个真实世界交通数据集** 上进行实验，涵盖多种任务、城市、分辨率和变量类型：

| 类型 | 包含数据集 |
|------|----------|
| **Traffic Flow** | PEMS03, PEMS04, PEMS07, PEMS08, SD, GBA |
| **Taxi Demand** | NYC-Taxi, Beijing Taxi, CHI-Taxi |
| **Bicycle Sharing** | NYC-Bike, CHI-Bike |
| **Traffic Speed** | TrafficHZ, TrafficJN, TrafficSH, TrafficZZ, METR-LA, PEMSBAY |
| **Metro Flow** | Beijing Subway (10min/15min), SHMetro, HZMetro |
| **Congestion Level** | DIDI-CD, DIDI-SZ |

- **预训练集**：12个数据集
- **下游评估集**：11个未见过的数据集（zero-shot / few-shot）

---

### 实验设置
- **预测任务**：
  - **Short-term**：12步预测（对应1–6小时，依分辨率而定）
  - **Long-term**：64步预测（对应半天至一天）
- **评估协议**：
  - **Zero-shot**：直接在新数据集上测试，不微调
  - **Few-shot**：使用7天训练数据微调预测头和最后一层编码器
- **Batch Size**：64
- **Optimizer**：AdamW，LR=0.001，梯度裁剪阈值=5
- **早停机制**：验证损失连续15轮无改善则停止

---

### 评估指标
- **MAE**（Mean Absolute Error）
- **RMSE**（Root Mean Square Error）

---

### 基线方法对比
#### Zero-shot 对比：
- **Pre-trained ST models**：OpenCitymini/base/plus, CrossST, CompactST
- **Time Series Foundation Models**：TimesFM, Timer, Sundial, Time-MoE

#### Few-shot 对比：
- **经典模型**：HA（Historical Average）
- **深度学习模型**：Informer, PatchTST, STGCN, GWNET, ST-Norm, STID, STAEformer

> 注：UniST 和 UniFlow 因结构限制被排除公平比较。

---

## 3. 主要实验结果和性能指标

### Zero-shot 性能（代表性结果）

#### 表 I：短期预测 MAE 结果（部分）
| Dataset | FlexST (Ours) | Best Baseline (OpenCityplus) | Improvement |
|--------|----------------|-------------------------------|------------|
| PEMS04 | **26.76** | 26.80 | ↓0.15% |
| PEMS08 | **2.83** | 2.88 | ↓1.74% |
| Didi-SZ | **44.17** | 45.59 | ↓3.11% |
| Beijing Taxi | **47.05** | 48.65 | ↓3.29% |
| HZMetro | **4.88** | 5.37 | ↓9.12% |

> ✅ FlexST 在多数任务中达到最优或次优，尤其在复杂城市（如HZMetro）表现突出。

#### 表 II：长期预测 MAE 结果（HZMetro 入流）
| Method | MAE | RMSE |
|-------|-----|------|
| HA | 106.77 | 193.77 |
| OpenCityplus | 46.13 | 112.89 |
| **FlexST (Ours)** | **42.52** | **100.17** |
| → Improvement vs. best baseline | ↓7.8% | ↓11.2% |

> 🔥 在 HZMetro 上，相比 HA 错误下降达 **60.18%**

---

### Few-shot 性能（7天训练数据）

#### 表 III & IV：短期与长期预测对比（代表项）
| Dataset | Metric | FlexST (Few) | Second Best | Improvement |
|--------|--------|---------------|-------------|-------------|
| PEMS08 | MAE | **18.23** | 19.97 (GWNET) | ↓8.7% |
| Didi-SZ | MAE | **0.89** | 0.93 (STAEformer) | ↓4.3% |
| NYC-Taxi | MAE | **7.66** | 10.50 (GWNET) | ↓27.0% |
| NYC-Bike | MAE | **3.53** | 3.81 (STNorm) | ↓7.3% |

> ✅ 即使在极少量数据下，FlexST仍显著优于强基线。

---

### 消融实验（Ablation Study）

在 PEMS08、NYC-Taxi 等数据集上验证各组件作用：

| 变体 | 描述 | MAE 影响 |
|------|------|---------|
| w/o multi-res | 移除多分辨率建模 | ↑ 明显上升（尤其 NYC-Bike） |
| w/o experts | 移除 MoE 模块 | ↑ 中等上升，影响跨城迁移 |
| w/o upe | 移除统一周期编码 | ↑ 最大上升，尤其跨年/跨城场景 |

> 📌 **关键发现**：`Unified Periodic Encoding` 是最重要的组件，尤其在处理周期漂移时至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **异构性必须被显式建模**：强行统一表示会导致负迁移；FlexST 通过模块化解耦实现了“共性保留 + 差异性隔离”。
2. **MoE 是高效迁移的关键**：稀疏激活机制不仅节省计算资源，还能防止无关知识干扰。
3. **周期先验需具备泛化能力**：传统的固定周期假设失效，而统一编码策略能有效对齐跨域周期模式。
4. **零样本迁移可行且强大**：在完全未训练的任务/城市上，FlexST 依然表现出色，证明其真正具备“通用车辆预测模型”潜力。

---

### 方法局限性
1. **依赖图结构**：需要已知空间拓扑 $ G=(V,A) $，对于无图或动态图场景适用性受限。
2. **预计算开销**：虽然扩散矩阵是离线生成，但在超大规模图（>10k节点）上仍可能耗时。
3. **专家利用率不均**：部分 MoE 专家被较少调用，可能存在冗余（可通过部署期剪枝优化）。

---

### 未来工作方向
1. **扩展至更多模态**：融合天气、事件、POI 等外部因素作为额外输入。
2. **动态图适应**：引入图学习机制，自动推断未知或变化的空间关系。
3. **轻量化部署**：研究 MoE 剪枝、蒸馏策略，进一步压缩模型体积。
4. **跨模态预训练**：探索与 LLMs 联合训练，实现“语言+时空”双模态理解。

---

> 🔗 **代码开源地址**：[https://github.com/liuzhouyang/FlexST](https://github.com/liuzhouyang/FlexST)

</details>

---

### 14. [UniCAR-RL: Seeing Better before Thinking Deeper in Visual Mathematics](https://arxiv.org/abs/2609.13849)

**Authors**: Yuzhe Li, Hao Yan, Hao Wang, Xingchen Liu, Ya-Qi Yu, Jihao Wu, Minghui Liao, Wei Chen, Yuliang Liu  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.13849v1  

#### Abstract
Multimodal Large Language Models (MLLMs) often struggle with complex mathematical visual reasoning primarily due to a lack of fine-grained perception, causing initial visual hallucinations to directly trigger cascading reasoning failures. In traditional end-to-end reinforcement learning (RL), sparse...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：UniCAR-RL: Seeing Better before Thinking Deeper in Visual Mathematics**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **Multimodal Large Language Models (MLLMs)** 在处理复杂的视觉数学推理任务时表现不佳，其根本瓶颈并非逻辑推理能力不足，而是**细粒度视觉感知能力薄弱**。模型在初始阶段对图像中的微观细节（如角度标记、坐标轴数值、几何符号）产生误读，导致后续严谨的数学推理建立在错误的前提上，从而引发“级联错误”（cascading reasoning failures）。

传统方法存在以下缺陷：
- **端到端强化学习（End-to-End RL）**：依赖稀疏的全局奖励（sparse rewards），无法区分错误是源于**视觉感知偏差**还是**逻辑推理失误**，导致感知优化效率低下。
- **基于CoT蒸馏的方法**：依赖专家标注的高质量感知增强型Chain-of-Thought（CoT）数据，成本高昂且受限于教师模型自身的幻觉（hallucinations）。

---

### **提出了什么新方法或新思路**
本文提出 **UniCAR-RL**（Unified Caption-Answer-Reasoning Reinforcement Learning），一种无需人工标注的感知优化框架，通过**显式解耦感知与推理的优化过程**，实现两者的协同提升。

#### **核心思想**
将视觉数学推理任务拆分为三个独立但共享参数的训练分支，分别优化不同能力：
1. **Caption-RL Branch**  
   - **目标**：纯化视觉感知能力。  
   - **机制**：移除问题输入，强制模型生成全面、无偏的图像描述（caption）。  
   - **奖励信号**：由一个强大的**文本-only verifier**（如Qwen3.5-35B或Gemini 3 Pro）仅凭该描述和原问题尝试解题。若能正确解答，则说明描述完整准确，给予正向奖励。  
   - **意义**：获得**纯净的感知反馈信号**，不受推理错误干扰。

2. **Reasoning-RL Branch**  
   - **目标**：纯化逻辑推理能力。  
   - **机制**：使用Caption-RL中选出的最高分描述作为输入，**屏蔽原始图像**，让模型仅基于文本进行推理。  
   - **奖励信号**：基于最终答案是否正确。  
   - **意义**：切断感知幻觉向推理过程的传播，确保推理信号纯粹。

3. **QA-RL Branch**  
   - **目标**：保持端到端问答能力。  
   - **机制**：标准的图文联合输入，执行完整的CoT推理流程。  
   - **作用**：确保前两个分支的优化成果能有效整合回实际部署场景。

三者共享同一策略网络（policy network），通过统一的 **GRPO**（Group Relative Policy Optimization）目标联合更新，形成“局部解耦、全局共演”的训练范式。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **无需标注** | 完全基于原始短答案监督，避免高成本的人工标注或教师模型蒸馏。 |
| **精准优化** | 显式分离感知与推理信号，解决传统RL中奖励模糊的问题。 |
| **高效泛化** | 在多种架构（Qwen2.5-VL / Qwen3-VL）、规模（3B–8B）下均显著提升性能。 |
| **强扩展性** | 性能增益可迁移到非数学类多模态任务（如图表理解、幻觉检测）。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：`MMRL30k` 中的 30k 样本。
- **评估基准**（数学视觉推理）：
  - `WeMath`
  - `MathVerse`
  - `MathVista`
  - `MathVision`
- **通用任务评估**：
  - `ChartQA`（图表问答）
  - `HallusionBench`（语言幻觉与视觉错觉诊断）
  - 补充评估：`V*Bench`, `HR-Bench`

---

### **实验设置和评估指标**
- **主干模型**：`Qwen2.5-VL` 和 `Qwen3-VL` 系列（3B/4B/7B/8B）。
- **训练方式**：全参数微调（full-parameter fine-tuning）。
- **优化器**：AdamW，学习率 $1\times10^{-6}$，weight decay $1\times10^{-2}$。
- **Rollout数**：每样本生成16条响应路径。
- **Verifier模型**：`Qwen3.5-35B-A3B` 或 `Gemini 3 Pro`。
- **推理设置**：temperature=0.0, top-p=0.8, top-k=20。
- **评估指标**：各任务准确率（Accuracy），取四个数学基准的平均分作为综合指标。

---

### **基线方法对比**
- **闭源模型**：GPT-4o, Claude-3.7-Sonnet
- **开源SOTA方法**：
  - `GRPO`（基础RL）
  - `Shuffle-R1`, `MM-Eureka`, `NoisyRollout`, `PAPO`, `Perception-R1`, `Vision-R1` 等

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| 模型 | WeMath | MathVerse | MathVista | MathVision | **Avg.** |
|------|--------|-----------|------------|-------------|----------|
| Qwen3-VL-4B | 75.0 | 47.2 | 73.5 | 50.1 | 61.5 |
| **UniCAR-RL-4B (Gemini 3 Pro)** | **78.5** | **51.2** | **80.4** | **53.3** | **65.9** |
| Qwen3-VL-8B | 79.4 | 60.8 | 77.9 | 52.9 | 67.8 |
| **UniCAR-RL-8B (Gemini 3 Pro)** | **82.5** | **63.4** | **79.5** | **54.6** | **70.0** |

> ✅ **结论**：UniCAR-RL在所有配置下均显著超越基线，达到甚至超过部分闭源模型水平。

---

### **与基线方法的对比结果**
- 在 `Qwen2.5-VL-7B` 上，UniCAR-RL 达到 **58.2% 平均准确率**，优于所有同类RL方法（如PAPO-D-7B: 55.7%, Shuffle-R1-7B: 55.8%）。
- 即使在小模型上（如3B），也取得明显增益（从42.9% → 52.2%）。
- 在通用任务上同样表现出色：
  - `ChartQA`: Qwen3-VL-4B (83.0) → +UniCAR-RL (89.5)
  - `HallusionBench`: 73.7 → 75.5

---

### **消融实验结果（Ablation Study）**

#### **分支消融（Table 5）**
| 配置 | Avg. Score |
|------|------------|
| Full UniCAR-RL | **65.9** |
| w/o Reasoning-RL Branch | 64.2 |
| w/o Caption-RL Branch (vanilla GRPO) | 62.8 |
| w/o QA-RL Branch (base model) | 61.5 |

> 🔍 移除任一分支均导致性能下降，证明三者协同必要性；仅靠GRPO仅提升1.3%，而UniCAR-RL提升达3.1%。

#### **组件消融（Table 2）**
- **跨模型泛化**：在Qwen2.5-VL和Qwen3-VL上均有效。
- **跨规模稳定**：从小模型到大模型均有增益。
- **Verifier影响**：更强的Verifier（Gemini 3 Pro > Qwen3.5-35B）带来更高上限，但边际收益有限，表明可用开源模型替代闭源Verifier。

---

## **4. 关键结论和发现**

### **主要发现**
1. **感知是瓶颈**：MLLMs在视觉数学任务上的失败主因是**细粒度感知缺失**，而非推理能力不足。
2. **解耦优于联合训练**：显式分离感知与推理优化路径，可更精准地纠正各自错误，避免相互污染。
3. **无需标注也能提效**：通过Verifier机制，仅用原始短答案即可实现高质量感知优化，摆脱对昂贵标注或教师模型的依赖。
4. **强泛化能力**：优化效果不仅限于数学任务，还能迁移到图表理解、幻觉识别等一般多模态任务。
5. **训练效率可控**：尽管单步耗时约为GRPO的3.75倍，但由于收敛更快（约70步 vs 150步），总训练时间仅增加约1.75倍，性价比高。

---

### **方法的局限性**
- **训练开销增加**：三分支并行导致计算资源消耗上升，尤其Caption-RL需额外Verifier推理。
- **验证器依赖**：虽然可用开源模型替代，但仍需一个足够强大的text-only verifier来提供可靠反馈。
- **任务范围有限**：目前主要验证于数学和少量通用任务，尚未广泛应用于其他复杂多模态场景（如医学图像分析、视频理解等）。

---

### **未来工作方向**
- 探索更高效的Verifier设计（如轻量化代理模型）以降低训练成本。
- 将UniCAR-RL扩展至更多模态任务（如音频-文本、视频-问答）。
- 研究如何进一步自动化黄金描述的选择与反馈机制。
- 结合自监督或数据增强技术，进一步减少对外部Verifier的依赖。

--- 

> 📌 **一句话总结**：  
> **UniCAR-RL 通过“先看清，再深思”的解耦式强化学习框架，在无需标注的前提下显著提升了 MLLMs 的视觉数学推理能力，并展现出卓越的泛化性与实用性。**

</details>

---

### 15. [Flattening Every Memory Peak in Long-Context Mixture-of-Experts Training](https://arxiv.org/abs/2609.14306)

**Authors**: Shrey Pandit, Xuan-Phi Nguyen, Yiran Zhao, Shafiq Joty  
**Category**: cs.DC  
**Published**: 2026-09-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.14306v1  

#### Abstract
Training a Mixture-of-Experts (MoE) model at long context or large batch size fails as soon as any one component's peak allocation exceeds device memory, so the target is every peak at once, not the average footprint. Four are left unbounded by the parallelism plans in common use, and each grows dif...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Flattening Every Memory Peak in Long-Context Mixture-of-Experts Training 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Long-Context** 或 **Large-Batch** 场景下训练 **Mixture-of-Experts (MoE)** 模型时，GPU 内存峰值（peak memory）常常成为瓶颈。即使平均内存占用可控，只要**任一组件的瞬时峰值超出设备容量**，训练就会失败。

传统并行策略（如 ZeRO、FSDP、Expert Parallelism）虽然能降低平均内存，但对以下四个关键组件的内存峰值缺乏有效约束：
- **Expert Dispatch**：路由矩阵随 batch 和 routing imbalance 增长
- **Vocabulary Projection**：logits 张量大小为 `O(N×V)`，在长上下文和大词表下爆炸
- **Checkpointing Boundaries**：每层保留的输入张量累积占用大量显存
- **Optimizer State**：AdamW 状态占 `16O` 字节，在参数量大时难以容纳

这些峰值的增长速率不同，哪个先“爆掉”取决于具体配置，因此单一优化无法根本解决问题。

---

### 提出了什么新方法或新思路
论文提出了一套 **四合一的 bounded-streaming 操作符组合**，分别针对上述四个内存峰值进行**独立且可组合的显存上限控制**，确保所有组件的 GPU 工作集（working set）在启动时即可确定。

#### 四个核心操作符：
| 操作符 | 解决的问题 | 核心思想 |
|--------|-----------|---------|
| **PipelinedLLEP** | Expert Dispatch 峰值过高 | 在 Least-Loaded Expert Parallelism (LLEP) 基础上引入**按源节点 token 预算分块调度**，限制每个 chunk 中单个 rank 发送的 token 数量，从而控制接收端缓冲区大小 |
| **Ring-DTP** | Vocabulary Projection 显存爆炸 | 提出 **Ring Data-Tensor-Parallel** 投影机制，通过环形通信轮流组合 distinct batch 与 vocab shard，**在线计算 log-sum-exp**，避免构建完整的 `N×V` logits 矩阵 |
| **Selective Checkpoint Offload (SCO)** | Checkpoint 边界长期驻留 | 将部分 checkpoint boundary **选择性卸载到 CPU 内存**，并在反向传播前异步预取，实现显存与内存的灵活权衡 |
| **OffloadStream-AdamW** | CPU Adam 更新慢导致 GPU 空闲 | 将 optimizer state 分桶流式传输至 GPU 进行更新，利用 **communication-computation overlap** 加速，避免串行 CPU 更新 |

> 所有方法均**不改变模型输出、梯度精度或训练质量**，仅调整计算顺序和数据移动方式。

---

### 相比现有方法的优势
- ✅ **全面性**：同时解决四大内存瓶颈，而非只优化其一
- ✅ **精确性**：保持 BF16 全精度训练，无量化、低秩近似等损失
- ✅ **可组合性**：各操作符可独立启用，按需部署
- ✅ **可预测性**：提供闭式（closed-form）的 per-rank 显存预算公式，支持训练前可行性验证
- ✅ **高性能**：不仅降显存，还提升吞吐量（up to 10.4×）

---

## 2. 核心实验方法和设置

### 数据集
- 主要使用合成数据进行基准测试（controlled benchmarks）
- 最终端到端实验基于自定义 MoE 架构进行预训练任务
- 微调任务使用 **Nemotron-Math** 数据集进行监督微调（SFT），评估数学推理能力

---

### 实验设置
| 设置项 | 描述 |
|-------|------|
| **硬件平台** | 单节点 8×NVIDIA H200 GPU，NVLink 互联，2TB CPU RAM |
| **模型规模** | 120B, 241B, 667B 参数的 MoE 模型 |
| **上下文长度** | 最高达到 **1M tokens** |
| **并行策略基础** | 基于 **Mixture-of-Parallelisms (MoP)** rank layout，支持重叠子组（sequence, expert, vocab）并行 |
| **对比基线** | **FSDP2-best**：经过 exhaustive sweep 调优后的最优 FSDP2 + Expert/Tensor/Sequence Parallelism 组合 |

---

### 评估指标
| 指标 | 定义 |
|------|------|
| **Peak Memory (HBM)** | PyTorch 峰值显存分配（跨 rank 最大值） |
| **Throughput (tok/s/GPU)** | 每 GPU 每秒处理 token 数 |
| **TFLOPs/s** | 实际浮点运算效率 |
| **Context Reach** | 不发生 OOM 的最大序列长度 |
| **Global Batch Size** | 单次迭代中可容纳的最大全局 token 数 |
| **Training Quality** | 在 Nemotron-Math 上微调后在 AIME 2025 上的 Avg@8 准确率 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（孤立测试）

| 方法 | 性能提升 |
|------|--------|
| **PipelinedLLEP** | MoE dispatch 显存峰值降低 **59.3%**，速度无显著下降（1.01–1.10× LLEP） |
| **Ring-DTP** | Vocabulary projection 显存峰值降低 **86.6%**，延迟仅增加 <5% |
| **SCO** | 显存峰值随 host budget 单调下降，最大 batch 提升 **17.7%**，吞吐变化 <2% |
| **OffloadStream-AdamW** | Offloaded optimizer step 速度提升 **2.05×**（3.95s → 1.93s） |

---

### 端到端综合性能对比（vs FSDP2-best）

| 模型规模 | Context Reach | 吞吐提升 | 最大全局 batch 提升 |
|----------|---------------|----------|---------------------|
| 120B | 1M vs 128K (**8×**) | **7.6×** @128K | **12×** |
| 241B | 1M vs 32K (**32×**) | — | **7×** |
| 667B | 1M vs 64K (**16×**) | **10.4×** @64K | **3×** |

> 🔺 **FSDP2 在 32K–128K 长度即 OOM，而本文方法稳定运行至 1M**

---

### 消融实验结果
- **PipelinedLLEP**：chunk 数量影响显存与延迟平衡；过小 chunk 导致通信开销上升，过大则失去显存控制效果
- **Ring-DTP**：动态选择 move-activations 或 move-weights 策略可进一步优化带宽利用率
- **SCO**：offload 的边界越多，显存越低，但需足够 host memory 支持
- **OffloadStream-AdamW**：2 个 staging slot 即可饱和链路，更多 slot 不再提升性能

---

## 4. 关键结论和发现

### 主要发现
- ✅ **MoE 长上下文训练的失败往往是“最后一个压垮骆驼的峰值”所致**，必须**同时控制所有潜在峰值**
- ✅ 通过合理的调度设计（如 chunking、ring communication、selective offload、streaming update），可以在**不牺牲精度的前提下将大张量物化转为流式计算**
- ✅ 所有四个操作符均可独立启用，形成模块化优化栈
- ✅ 组合使用后可在 **1M context 下成功训练 667B MoE 模型**，远超当前主流方案的能力范围
- ✅ 训练质量不受影响：在 Nemotron-Math 上微调后，本文方法达到 **59.8% Avg@8**，与基线 **59.6%** 相当

---

### 方法的局限性
- ⚠️ **依赖高速互联**：All-to-all 和 ring communication 假设存在低延迟高带宽连接（如 NVLink），跨节点扩展可能受限
- ⚠️ **Host Memory 与 Bandwidth 成为新瓶颈**：SCO 和 OffloadStream-AdamW 大量使用 CPU 内存和 PCIe 带宽，在资源受限节点上可能不可行
- ⚠️ **超参敏感**：如 token budget `c`、bucket size `B` 等需经验调优，尚未完全自动化
- ⚠️ **Chunk 数量影响性能**：过多 chunk 会增加通信开销，需在显存与效率间权衡

---

### 未来工作方向
- 自动化选择 `(D, Ep, P, c, B)` 等超参的编译器级支持
- 支持更复杂的拓扑结构（如多机多环）
- 探索更细粒度的 streaming 策略，进一步压缩中间状态
- 将该框架推广至其他稀疏架构（如 Switch Transformers、QLoRA 等）

--- 

> 📌 **总结一句话**：  
> 本文通过 **PipelinedLLEP、Ring-DTP、SCO 和 OffloadStream-AdamW** 四大操作符，首次实现了对 MoE 模型四大内存峰值的**全可控流式训练**，使 **1M context 下训练千亿级 MoE 成为现实**，并带来高达 **10.4× 的吞吐提升**。

</details>

---

### 16. [HiGFRL: Hierarchical Graph Fusion-Driven Reinforcement Learning for Dependency-Aware Task Scheduling in Heterogeneous Cloud](https://arxiv.org/abs/2609.14968)

**Authors**: Tiangang Li, Shi Ying, Xiangbo Tian  
**Category**: cs.LG  
**Published**: 2026-09-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.14968v1  

#### Abstract
Online scheduling of dependency-aware tasks in heterogeneous cloud clusters is a fundamental yet challenging problem due to the complex interplay between DAG topologies and multi-dimensional resource constraints. While DRL has shown promise, existing GNN-based approaches often struggle to efficientl...

---

### 17. [STHMoE: Hypergraph-Enhanced Heterogeneous Dependency Coordination for LLM-Based Urban Traffic Data Forecasting](https://arxiv.org/abs/2609.15172)

**Authors**: Jiawen Chen, Qi Shao, Yongjian Chang, Mingtong Zhou, Duxin Chen, Wenwu Yu  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.15172v1  

#### Abstract
Spatio-temporal traffic forecasting is a fundamental big data analytics task for intelligent transportation systems, where massive urban sensor streams exhibit heterogeneous, non-stationary, and structurally dynamic patterns. Although recent deep learning and large language model (LLM)-based methods...

---

### 18. [Lexical Prompt Compression for Large Language Models: A Training-Free, Deterministic Pipeline with Empirical Pareto Analysis Across Eleven Task Categories](https://arxiv.org/abs/2609.13154)

**Authors**: Shamin Chokshi  
**Category**: cs.CL  
**Published**: 2026-09-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.13154v1  

#### Abstract
Recent advances in large language models (LLMs) have made prompts increasingly large and complex. Techniques such as chain-of-thought reasoning (Wei et al., 2022) and in-context learning (Brown et al., 2020) frequently push real-world prompts past several thousand tokens, increasing inference cost a...

---

### 19. [To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs](https://arxiv.org/abs/2609.15528)

**Authors**: Franck Signe, Hippolyte Pilchen, Fran\c{c}ois Yvon, \'Edouard Grave  
**Category**: cs.CL  
**Published**: 2026-09-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.15528v1  

#### Abstract
Multilingual Large Language Models (LLMs) traditionally rely on a single vocabulary shared by all supported languages, which can lead to uneven compression across them. Moreover, their large embedding and output matrices increase memory usage and slow inference, notably for small-scale models. It is...

---

### 20. [PEAT: Pseudo-Error Assessment for GPU Kernel Validation in DNN Training](https://arxiv.org/abs/2609.13544)

**Authors**: Xuan Truong Nguyen (Department of Next Generation Semiconductor Convergence,Open Sharing System), Hong Quan Tran (Efficient Computation Research Group, Vietnam National University), Tuan Duc Chu (Efficient Computation Research Group, Vietnam National University), Thanh Tuan Dao (Efficient Computation Research Group, Vietnam National University,,Moreh Vietnam, Hanoi, Vietnam)  
**Category**: cs.DC  
**Published**: 2026-09-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.13544v1  

#### Abstract
Deep neural networks (DNNs) are widely adopted in various fields, driving an emerging trend in developing software stacks associated with DNN training systems. For example, many codes have been ported across different frameworks or developed to leverage the computing power of GPUs or domain-specific...

---

### 21. [Physically Partitioned KVCache Format for CPU--GPU Load Balancing in MoE Inference](https://arxiv.org/abs/2609.14507)

**Authors**: Enda Yu, Dezun Dong, Xiangke Liao  
**Category**: cs.DC  
**Published**: 2026-09-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.14507v1  

#### Abstract
Single-GPU long-context inference with Mixture-of-Experts (MoE) models requires spilling the key-value cache (KVCache) to CPU memory. The spilled KV serves two complementary purposes---transferring to the GPU for attention computation, or computing in-place on the CPU---which demand opposing physica...

---

### 22. [Backward SDEs-based Diffusion for Physics-Constrained Generation](https://arxiv.org/abs/2609.15702)

**Authors**: Zihao Wang  
**Category**: cs.LG  
**Published**: 2026-09-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.15702v1  

#### Abstract
Pretrained score-based diffusion models provide strong unconditional priors, yet enforcing measurement or physics consistency in inverse problems is often handled by heuristic guidance, intermittent projections, or task-specific conditional training, with limited guarantees of feasibility at the end...

---

### 23. [A Hybrid Agentic AI Framework for Intelligent Supply Chain Analytics](https://arxiv.org/abs/2609.13561)

**Authors**: Xian Yeow Lee, Teppei Inoue, Haiyan Wang, Chetan Gupta  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.13561v1  

#### Abstract
Efficient utilization of supply chain analytics for decision making remains a significant challenge for planners, as critical tasks such as database querying, key performance indicator (KPI) analysis, demand forecasting, and performance diagnosis require heterogeneous expertise spanning data enginee...

---

### 24. [Lightning Weave: Improving the Accuracy-Efficiency Frontier of Reasoning Models through Capability Composition](https://arxiv.org/abs/2609.14708)

**Authors**: Yecheng Wu, Song Han, Han Cai  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.14708v1  

#### Abstract
A core goal of efficient reasoning is to improve the accuracy-efficiency frontier. However, jointly improving reasoning accuracy and inference efficiency can be challenging, as the two objectives can favor different reasoning behaviors. Independently post-trained models already offer distinct streng...

---

### 25. [GGUF-Metadata Prediction of Single-Sequence llama.cpp Throughput Across Three Systems](https://arxiv.org/abs/2609.14864)

**Authors**: Xinyu Qiu, Chuhong Xu, Bo Su, Ziyao Chen, Ruiyang Xu, Shimeng Dai  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.14864v1  

#### Abstract
We predict single-sequence model throughput from GGUF metadata using roofline-shaped predictors with quantization-specific scale factors fitted on reference models. The scored cohort comprises 318 phase-depth measurements from 53 host-file configurations on two Apple M4 Max systems and an NVIDIA RTX...

---

### 26. [Parameter-Efficient Adaptation of Pretrained Language Models for Time-Series Forecasting](https://arxiv.org/abs/2609.15344)

**Authors**: Tamanna Kumavat, Georg Brunner, Kyriakos Flouris  
**Category**: cs.AI  
**Published**: 2026-09-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.15344v1  

#### Abstract
We study the adaptation of pretrained language models to univariate time-series forecasting through a parameter-efficient transfer learning framework, with the goal of understanding which design choices drive effective cross-modal transfer. While language models operate on discrete textual tokens, t...

---

### 27. [A Multi-Stage Agentic Framework for Effective Counter-Narrative Generation and Refinement](https://arxiv.org/abs/2609.14178)

**Authors**: Carmel Kronfeld, Sharva Gogawale, Tetsuro Kobayashi, Irad Ben-Gal  
**Category**: cs.CL  
**Published**: 2026-09-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.14178v1  

#### Abstract
The rapid diffusion of hate speech and misinformation on social networks challenges democratic societies, since direct suppression efforts may deepen polarization, fuel public distrusts, and strengthen extremist narratives. LLM-driven counter-narratives (CNs) offer a promising way to reduce those ri...

---

### 28. [Dream-RSI: Recursive Self-Improvement through Evolving Worlds](https://arxiv.org/abs/2609.14858)

**Authors**: Tong Zheng, Xidong Wu, Zheng Zhang, Zhankui He, Chaoyi Zhang, Benjamin Coleman, Ruoqiao Wei, Di Bai, Haolin Liu, Rui Liu, Xue Wang, Yue Zhuan, Wang-Cheng Kang, Renkai Xiang, Heng Huang, Xinwu Cheng, Yunsong Guo  
**Category**: cs.CL  
**Published**: 2026-09-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.14858v1  

#### Abstract
Recursive self-improvement is becoming increasingly vital for autonomous AI agents, where progress hinges on discovering high-value solutions across complex domains. The driver of this process is effective exploration, however, managing and improving exploration strategies remains a major bottleneck...

---

### 29. [ABSOL: Aggregated Bayesian Subsampling Orchestrated with LLMs](https://arxiv.org/abs/2609.15007)

**Authors**: Jackson Hassell, Chen Shen, Estevam Hruschka  
**Category**: cs.CL  
**Published**: 2026-09-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.15007v1  

#### Abstract
Large language models are increasingly used as natural-language interfaces to structured data, yet they remain unreliable when answers require consistent evidence conditioning, dependency-aware reasoning, and uncertainty estimation. Bayesian networks provide an explicit probabilistic reasoning layer...

---

### 30. [When Agents Slow Down: Understanding LLM Agents' Test-Time Strategies via Elo-per-token Analysis](https://arxiv.org/abs/2609.15309)

**Authors**: Kaiyuan Liu, Qiuyang Mang, Bo Peng, Wenhao Chai, Hanchen Li, Shreyas Pimpalgaonkar, Luke Zettlemoyer, Alex Dimakis, Alvin Cheung  
**Category**: cs.CL  
**Published**: 2026-09-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.15309v1  

#### Abstract
Large language model (LLM) agents allocate test-time compute adaptively as they revise solutions, use tools, explore alternatives, and decide when to stop. This test-time strategy makes it difficult to measure how agent performance scales. We study open-ended tasks that provide continuous scores for...

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

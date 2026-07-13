# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-13 08:40:59 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [BlockServe: Block-Grained Continuous Batching for High-Throughput Diffusion LLM Serving](https://arxiv.org/abs/2607.08930)

**Authors**: Yuanjie Zhu, Liangwei Yang, Ke Xu, Weizhi Zhang, Shanghao Li, Zihe Song, Philip S. Yu  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.08930v1  

#### Abstract
Efficient serving of diffusion large language models (dLLMs) is hindered by convergence heterogeneity: when batching multiple requests, different sequences converge at different rates, causing faster requests to stall behind slower stragglers and introducing compute bubbles and tail latency. We pres...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《BlockServe: Block-Grained Continuous Batching for High-Throughput Diffusion LLM Serving》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前 **diffusion LLMs (dLLMs)** 在批量推理时面临**收敛异质性（convergence heterogeneity）**问题：不同请求在不同的 block 边界完成生成，导致已完成的请求仍需等待最慢的“straggler”请求，造成 GPU 资源闲置（compute bubbles）和尾部延迟增加。

传统 **token-level 连续批处理（continuous batching）** 适用于 autoregressive (AR) 模型，但不适用于 dLLMs 的块级并行去噪机制。

---

### **提出的新方法与创新点**

BlockServe 提出了一种专为 dLLMs 设计的高吞吐连续批处理框架，核心创新包括：

#### **① Block-Grained Scheduling（块粒度调度）**
- 将调度单位从“请求”变为“block”，每个 block-cycle 执行 S 步去噪。
- 在 block 边界处立即检查并**驱逐已完成的请求**，避免其继续占用资源。
- 显著减少 straggler 引发的计算气泡（compute bubbles）。

#### **② Mixed-State Memory Management（混合状态内存管理）**
- 支持不同 block 阶段的请求共存于一个统一的 dense tensor 中。
- 通过 **gather-scatter indexing** 实现动态 KV 缓存更新，保持 RoPE 和 attention mask 的一致性。
- 允许 **dual cache** 和 **parallel decoding** 在异构批次中运行，无需定制 kernel。

#### **③ Compute-Aware Admission Control（计算感知准入控制）**
- 放弃固定 batch size，采用 **token budget** 动态控制并发数。
- 基于当前 batch 的 bounding box 和剩余 token 预算决定是否接纳新请求。
- 利用已完成请求释放的空间进行“refill”，提升有效 batch 容量。

---

### **相比现有方法的优势**
| 维度 | BlockServe | Fast-dLLM（基线） |
|------|-----------|------------------|
| 调度粒度 | Block-level | Request-level（静态批处理） |
| 内存利用 | 动态回收 + token budget | 固定 padding，资源浪费严重 |
| 并发能力 | 动态扩容至 2–4× | 受限于最长请求 |
| 吞吐 | 显著提升（1.9–10.6×） | 低，受 straggler 限制 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在五个标准 benchmark 上评估，覆盖多种任务类型：
- **GSM8K**: 数学应用题（mathematical reasoning）
- **HumanEval**: Python 函数生成（code generation）
- **MBPP**: 自然语言到短程序生成
- **MATH**: 复杂数学竞赛题
- **TruthfulQA-Gen**: 事实问答生成，测试真实性

---

### **实验设置**
- **模型**：LLaDA-8B-Instruct 和 Dream-v0-Instruct-7B
- **硬件**：单张 NVIDIA H200 GPU（141GB HBM3e），bfloat16 精度
- **默认配置**：
  - Block length $ L = 32 $ tokens
  - Denoising steps per block $ S = 32 $
  - Temperature = 0.0（greedy decoding）

---

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Throughput (TPS)** | 总生成 token 数 / 总 wall-clock time（tokens/sec） |
| **Total Time** | 完成整个 benchmark 的实际耗时 |
| **Accuracy** | 任务特定准确率（如 pass@1、correctness） |
| **Effective Compute Ratio** | 衡量每步活跃请求数占比，反映资源利用率 |

---

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Vanilla** | 原始模型配置，无任何优化 |
| **Fast-dLLM [6]** | 当前最优 dLLM 推理加速器，支持 dual cache 和 parallel decoding，但**无连续批处理机制**，采用静态批处理 |
| **BlockServe (Fixed)** | 使用固定 batch size 的 BlockServe 版本 |
| **BlockServe (Budget)** | 使用 token budget 动态控制并发的完整版本 |

> ✅ BlockServe 所有变体均集成 Fast-dLLM 的 dual cache 和 parallel decoding，确保增益来自调度而非 kernel 优化。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（RQ1）**
在 **LLaDA 和 Dream** 模型上，BlockServe 实现显著吞吐提升：

| 模型 | 最大加速比（vs Fast-dLLM） | 出现场景 |
|------|----------------------------|----------|
| **LLaDA** | **2.6× – 7.7×** | 如 HumanEval 上达 597.8 TPS（原 231.5） |
| **Dream** | **1.9× – 10.6×** | TruthfulQA (gen_len=1024) 达 10.6× 加速 |

> 🔥 **最高吞吐提升出现在 TruthfulQA 上**，因其输出长度极短且方差大，straggler 效应在静态批处理中尤为严重。

---

### **与基线对比结果**
#### **吞吐（Throughput）**
- BlockServe (Budget) 在所有 benchmark 上均显著优于 Fast-dLLM。
- 例如，在 **Dream + HumanEval (BS=16)** 上：
  - Fast-dLLM: 205.1 TPS
  - BlockServe (Fixed): 462.9 TPS (**2.3×**)
  - BlockServe (Budget): 更高（峰值可达 477.9 TPS）

#### **总耗时（Total Wall-Clock Time）**
- BlockServe 将多个任务的执行时间压缩至原来的 **1/10**。
- 如 Dream 上 MATH (1024) 从 >24h 缩减至 ~95h（未超时）。

#### **生成质量（Accuracy）**
- 所有方法精度相近，**BlockServe 未牺牲生成质量**。
- 例如 MATH 上 accuracy 维持在 ~37%，与基线一致。

---

### **消融实验结果（Ablation Studies, RQ5）**

#### **① Block Length 影响（L ∈ {16,32,64}）**
| L | 吞吐趋势 | 准确率影响 |
|----|--------|----------|
| 16 | 吞吐较低 → 调度开销大 |
| 32 | **最佳平衡点** → 高吞吐 + 稳定 accuracy |
| 64 | accuracy 下降（Dream 上 ↓9%）→ 并行去噪误差放大 |

✅ **选择 L=32 作为默认值**

#### **② Prompt-Length Sorting 影响**
- 对待处理队列按 prompt length 排序可减少 padding 浪费。
- 结果显示：
  - 吞吐提升 **4.7% – 9.2%**
  - 时间减少最多达 **~10%**（如 HumanEval 上从 48.8s → 45.2s）

---

### **其他关键发现**
#### **有效并发分析（RQ2）**
- **Effective Compute Ratio**：
  - Fast-dLLM 在 BS=16 时仅维持 ~0.07（即平均仅 1.1 个活跃请求）
  - BlockServe (Fixed) 接近 1.0
  - BlockServe (Budget) 可达 **2.4×**，表明实现高效超售（oversubscription）

#### **批大小扩展性（RQ3）**
- Fast-dLLM 随 batch size 增加出现严重尾延迟。
- BlockServe 保持稳定低延迟分布，即使在 BS=16 也无明显长尾。

#### **最大安全批大小（RQ4）**
| 方法 | 最大 batch size（gen_len=1024） | 提升倍数 |
|------|-------------------------------|---------|
| Fast-dLLM | 16–64 | — |
| BlockServe | **37–128** | **2.0× – 4.0×** |

> 如 LLaDA-MBPP 上从 16 → 64，容量提升 **4.0×**

---

## **4. 关键结论和发现**

### **主要发现**
1. **Convergence heterogeneity 是 dLLM 批处理的核心瓶颈**，传统 AR 调度机制无法应对。
2. **Block-grained scheduling 是突破口**：以 block 为单位进行抢占和资源回收，能有效消除 straggler 影响。
3. **Token-budget admission 控制显著提升硬件利用率**，实现动态扩容。
4. **BlockServe 可无缝集成 Fast-dLLM 等优化技术**，形成正交增强。

---

### **方法的局限性**
- 当前评估集中在 **offline batch inference** 场景，尚未验证在线流式请求（online serving）下的表现。
- Gather-scatter indexing 虽通用，但在极端异构场景下可能引入额外索引开销。
- 对 very long sequence（>2048）的支持有待进一步测试。

---

### **未来工作方向**
- 扩展至 **online serving**，支持动态到达请求的实时调度。
- 探索更细粒度的 **adaptive block length** 策略，根据 confidence 或 content 动态调整。
- 结合 **phase-multiplexing**（如 dLLM-Serve）进一步优化内存与带宽利用率。

---

> ✅ **总结一句话**：  
> **BlockServe 通过 block-grained scheduling + mixed-state execution + token-budget admission，在不改变模型架构的前提下，实现了 dLLM 批量推理吞吐 1.9–10.6× 的提升，是迈向高效 diffusion LLM serving 的关键一步。**

</details>

---

### 2. [Communication-Efficient Digital-Twin Coordination for Heterogeneous LLM Embodied Agents over Computing Power Networks](https://arxiv.org/abs/2607.09330)

**Authors**: Nuocheng Yang, Sihua Wang, Zihan Chen, Tony Q. S. Quek, Changchuan Yin  
**Category**: cs.AI  
**Published**: 2026-07-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.09330v1  

#### Abstract
Embodied agent teams powered by heterogeneous large language models (LLMs) are being widely deployed in physical artificial intelligence such as smart factories, warehouses, and service robotics. To enable collaboration among such an agent team, efficient coordination mechanisms that operate reliabl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Communication-Efficient Digital-Twin Coordination for Heterogeneous LLM Embodied Agents over Computing Power Networks*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对由**异构 Large Language Models (LLMs)** 驱动的具身智能体团队（embodied agent teams）在协作中存在的三大挑战：

1. **通信开销高**：基于多轮自然语言（NL）对话的协调机制导致通信负载随团队规模呈指数增长。
2. **异构性瓶颈**：协调质量受限于团队中最弱LLM的语言能力（“短板效应”）。
3. **动作延迟**：迭代协商过程引入显著的协作延迟，影响实时性。

### 提出的新方法：LDT-Coord
作者提出 **LDT-Coord**（Lightweight Digital-Twin Coordination），一种基于轻量级数字孪生（Digital Twin, DT）的网络化协调框架，其核心创新包括：

- **结构化通信原语（Structured Communication Primitives）**  
  每个 agent 自主决策后，仅向 DT 上报一个**结构化的行动元组**（action tuple）和**类型化的时间约束声明**（typed constraint declaration），而非冗长的自然语言对话。这将协调从“相互理解语言”转变为“上报结构化约束”，从而解耦协调质量与报告者的语言能力。

- **无需训练的统一规则型协调器（Training-Free Unified Rule-Based Orchestrator）**  
  DT 运行一个**无须训练、基于规则的协调算法**，将互斥（mutual exclusion）、同步（synchronization）和依赖（dependency）三类冲突形式化为协调规则，并通过迭代应用这些规则解决跨 agent 冲突，返回无冲突的执行指令。

- **基于强化学习的通信选择层（Learned Communication-Selection Layer）**  
  将 agent 是否上报决策建模为一个**受时延约束的 Constrained Partially Observable Markov Decision Process (C-POMDP)**，并采用 **PPO-Lagrangian** 算法求解最优上报策略，在保证协调质量的前提下大幅压缩通信开销。

### 相比现有方法的优势
| 维度 | LDT-Coord | 传统方法（如 NL 对话、中心化规划） |
|------|-----------|-------------------------------|
| **通信效率** | 极高（减少 >70×） | 多轮 NL 通信，开销巨大 |
| **异构鲁棒性** | 强（不依赖最弱LLM的语言能力） | 易受最弱LLM拖累 |
| **延迟** | 低且恒定（约13–18ms） | 随团队规模上升至秒级 |
| **可扩展性** | 线性增长 | NL 方法呈二次增长 |
| **是否需训练** | 协调器无需训练 | 多数学习型方法需联合训练 |

---

## 2. 核心实验方法和设置

### 实验平台
- 使用 **RoCo/MuJoCo 物理仿真器**进行多臂协作任务评估。
- 考察三类典型协调任务，分别对应三种原子任务：
  - **Mutual Exclusion**：受限空间排序（confined-space sorting）
  - **Synchronization**：绳索协同提升（move rope）
  - **Dependency**：柜门保持-取物（hold-then-fetch）

### 团队异构性设置
- LLM 能力梯度：`DeepSeek-R1-Distill-Qwen-1.5B` > `Qwen2.5-1.5B-Instruct` > `Qwen2.5-0.5B-Instruct`
- 默认团队使用同构 `Qwen2.5-1.5B`，异构分析中构建弱、中、强异构配置。

### 基线方法（Baselines）
分为两类：
- **对话式方法**：
  - **RoCo-NL**：基于多轮 NL 对话协商
  - **AutoGen**：群聊管理器驱动的多 agent 对话
- **中心化方法**：
  - **Centralized-LLM (fused)**：中心 LLM 融合所有观测生成联合动作
  - **Centralized-Classical**：经典贪心调度器（无LLM，作为性能上限）

### 评估指标
- **Success Rate (SR)**：任务成功完成的比例
- **Per-episode Communication Data Size**：每回合通信数据量（字节）
- **End-to-end Latency**：每次决策的端到端延迟
- **LLM Inference Compute**：每回合使用的 prompt + completion token 总数
- **Invalid Execution Attempts**：无效执行尝试次数（反映冲突率）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（同构团队，80轮测试）

#### 表1：成功率与通信开销对比（同构1.5B）
| 方法 | SR | 数据大小 (B/ep) | 相对通信比 |
|------|-----|------------------|------------|
| **LDT-Coord (ours)** | **0.640** | **43.3** | **1.0×** |
| AutoGen | 0.700 | 3,911 | 90× |
| RoCo-NL | 0.580 | 3,147 | 73× |
| Centralized-LLM | 0.667 | 112.7 | 2.6× |
| Centralized-Classical | 0.733 | 84.3 | 1.9× |

> ✅ **结论**：LDT-Coord 在成功率接近基线的同时，通信开销仅为对话方法的 **1/73~1/90**，甚至低于中心化方法。

---

### 与基线方法的对比结果

#### 图3：每步通信 vs. 团队规模 $n$
- RoCo-NL 和 AutoGen 呈**二次增长**（因多轮对话 × 参与者数）
- LDT-Coord 和中心化方法呈**线性增长**
- LDT-Coord 始终最低，且优势随 $n$ 扩大而增强

#### 图4：端到端决策延迟 vs. 团队规模
- LDT-Coord 延迟**独立于 $n$**，稳定在 **13–18ms**
- 中心化方法随 $n$ 线性上升（收集+串行处理）
- 对话方法达 **8–13秒级**
- 在 $n=8$ 时，LDT-Coord 延迟仅为最慢基线的 **~1/740**

#### 图5：每回合LLM推理计算 vs. 团队规模
- LDT-Coord 和中心化方法呈线性增长
- 对话方法呈**二次爆炸式增长**
- 在 $n=8$ 时，对话方法计算量为 LDT-Coord 的 **4–6倍**

---

### 消融实验结果（Ablation Study）

#### 表2：移除模块的影响（$n=3$）
| 变体 | 移除模块 | SR | 延迟 | 报告次数/步 | 无效执行尝试/回合 |
|------|----------|-----|--------|----------------|--------------------|
| Full | —— | 0.71 | 0.13s | 0.30 | 5.48 |
| w/o DT | DT协调器 | 0.53 | 0.11s | 0.30 | **7.12** (+30%) |
| w/o RL | 学习型通信门 | 0.70 | 0.13s | **1.51** (×5) | 5.48 |

> ✅ **结论**：
> - 移除 DT 导致冲突规避失效，**无效执行尝试增加30%**
> - 移除 RL 通信选择层导致上报频率**增加5倍**，通信效率严重下降

---

## 4. 关键结论和发现

### 主要发现
1. **LDT-Coord 实现了高性能与高效率的平衡**：在任务成功率与主流方法相当的情况下，**通信开销降低超过70×，延迟降低近三个数量级**。
2. **对异构性高度鲁棒**：在强异构团队中，LDT-Coord 的成功率几乎不变（SR: 0.671 → 0.667），而对话方法（如 AutoGen）从 0.728 降至 0.567，表明其不受“最弱LLM”限制。
3. **可扩展性强**：通信与延迟均随团队规模线性增长，适合大规模部署。
4. **无需训练的协调机制更实用**：规则型协调器避免了复杂的联合训练，易于部署和解释。

### 方法的局限性
- 当前假设 agent 能正确生成结构化约束；若弱LLM频繁遗漏或错误声明约束，仍可能引发冲突（尽管系统对此有一定容忍）。
- 实验集中在结构化物理任务（如搬运、装配），在开放域复杂任务中的泛化能力有待验证。
- 通信选择策略依赖于局部可观测状态，全局优化潜力未完全释放。

### 未来工作方向
- 探索 agent 团队间的**协同感知**（collaborative perception），以扩展每个 agent 的感知边界。
- 引入**动态资源拓扑建模**，适应更复杂的共享资源环境。
- 结合 **goal-oriented semantic communication** 进一步压缩语义传输成本。
- 在真实机器人平台上验证 LDT-Coord 的实用性与稳定性。

---

> **总结**：LDT-Coord 通过“**本地决策 + 结构化上报 + 数字孪生规则协调 + 学习型通信压缩**”的架构，成功解决了异构 LLM 具身 agent 团队在有限网络资源下的高效协作难题，为未来 CPS、工业自动化和边缘智能提供了可扩展、低延迟、高鲁棒性的协调范式。

</details>

---

### 3. [Tokenizer Transplantation: Mitigating Autoregressive Collapse in Edge-Efficient Bengali ASR](https://arxiv.org/abs/2607.09598)

**Authors**: Sanjid Hasan, Md. Abdur Rahman  
**Category**: cs.CL  
**Published**: 2026-07-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.09598v1  

#### Abstract
Lightweight speech recognition models are critical for edge deployment, yet highly optimized architectures like Moonshine often fail on morphologically rich, non-Latin languages such as Bengali. This study identifies the root cause of this failure as the model's English-centric byte-level tokenizer,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Tokenizer Transplantation: Mitigating Autoregressive Collapse in Edge-Efficient Bengali ASR

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本研究针对**轻量级自动语音识别（ASR）模型在形态丰富、非拉丁语系语言（如孟加拉语）上的性能退化问题**。具体而言，像 Moonshine 这类为英语优化的边缘设备友好型模型，在处理 Bengali 时因使用 **byte-level tokenizer** 导致严重的 **token fertility（高分词率）**，即一个单词被拆分为大量子词单元，从而引发推理阶段的 **autoregressive collapse（自回归崩溃）** —— 错误累积导致生成文本完全不可读。

### 提出了什么新方法或新思路
提出了一种名为 **Tokenizer Transplantation（分词器移植）** 的三阶段适应性框架，核心思想是：
- **解耦声学表示学习与语言建模**；
- 在已微调的模型上“外科手术式”地替换 decoder 的词汇表（vocabulary），采用专为 Bengali 设计的 **BanglaBERT WordPiece tokenizer**；
- 配套设计了一个两阶段恢复优化策略（Two-Stage Recovery Optimization），防止因嵌入层随机初始化导致的 **catastrophic forgetting（灾难性遗忘）**。

该方法无需从头训练或大规模预训练，即可实现跨脚本（cross-script）适配。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 仅需约 61.5M 参数即达到媲美 ~769M 参数 Faster Whisper Medium 的 WER 表现 |
| **稳定性** | 将 token fertility 从 9.16 降至 1.30，autoregressive 序列长度减少 85.8%，彻底缓解 decoding instability |
| **可复用性** | 不依赖资源密集型 pre-training，提供可扩展、可复制的轻量 ASR 跨语言适配蓝图 |
| **边缘部署可行性** | 实现极低 RTF（0.0053），优于高度优化的 Whisper 和 Conformer 模型 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Lipi-Ghor-882 dataset**（Hasan et al., 2026）：  
  - 包含 **882 小时多说话人 Bengali 语音数据**，源自多样化开源媒体；
  - 音频采样率为 16kHz 单声道；
  - 使用 Pyannote 的 VAD 技术去除静音段和非语音片段；
  - 测试集为保留的 **5% 数据（共 5,931 条 utterance）**。

### 实验设置和评估指标
#### 模型架构基础
- 基于 **Moonshine-Base** 架构（61.5M 参数），其 encoder 保持不变；
- Decoder 替换为 **BUET BanglaBERT WordPiece tokenizer**（来自 Bhattacharjee et al., 2022）；
- 特殊 token 显式映射：`(CLS)` → `BOS`，`(SEP)` → `EOS`。

#### 三阶段训练流程
| 阶段 | 内容 |
|------|------|
| **Phase 1: Acoustic Fine-Tuning** | 在 Bengali 数据上对原始 Moonshine 模型进行 21 轮 fine-tuning，使 encoder 适应 Bengali 发音特征 |
| **Phase 2: Tokenizer Transplantation** | 替换 decoder tokenizer 并 resize embedding matrix，保留已学习的 acoustic weights |
| **Phase 3: Two-Stage Recovery Optimization** | <br>• **Stage 1**: 使用 ScheduleFree AdamW（LR=2×10⁻⁴）快速对齐新文本嵌入空间（7 epochs）<br>• **Stage 2**: 降低学习率（LR=2×10⁻⁶）进一步微调语法与发音准确性 |

#### 训练优化技术
- 使用 **BF16 Automatic Mixed Precision (AMP)** 和 **gradient checkpointing** 节省内存；
- **gradient accumulation step=8**，实现有效 batch size=32；
- 硬件平台：NVIDIA GeForce RTX 4070（12.9GB VRAM）。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Word Error Rate (WER)** | 主要评价指标，衡量词级别错误率 |
| **Character Error Rate (CER)** | 衡量字符级别准确率，尤其反映复杂拼写建模能力 |
| **Real-Time Factor (RTF)** | 推理速度指标，越低表示越快，适合边缘部署 |
| **Greedy WER** | 每轮训练后在 64 条随机验证样本上计算，用于 early stopping（patience=4） |

### 基线方法对比
涵盖多种零样本（zero-shot）与微调（fine-tuned）模型：
- **Zero-Shot**:
  - Seamless M4T-v2 (~2.3B)
  - OpenAI Whisper large-v3 (~1.55B)
  - Meta MMS 1B (~1B)
  - Hishab TITU Conformer Large (~120M)
- **Fine-Tuned**:
  - Conformer Baseline (fast-conformer, ~120M)
  - Faster Whisper Medium (Tustugi, ~769M)

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 数值 |
|------|------|
| **Token Fertility** | 从 9.16（原 tokenizer）降至 **1.30**（移植后） |
| **WER** | **21.54%**（在 Lipi-Ghor 测试集上） |
| **CER** | **10.79%**（所有模型中最低） |
| **RTF** | **0.0053**（原生 PyTorch 实现，无需 CTranslate2 加速） |
| **参数量** | ~61.5M（仅为 Faster Whisper Medium 的 ~8%） |

### 与基线方法的对比结果（见 Table 2）
| 模型 | Parameters | WER (%) | CER (%) |
|------|------------|---------|---------|
| Meta MMS 1B (Zero-Shot) | ~1B | 43.06 | 21.16 |
| Hishab-Titu Conformer (FT) | ~120M | 30.51 | 18.23 |
| Conformer Baseline (FT) | ~120M | 24.67 | 15.56 |
| Faster Whisper Medium (FT) | ~769M | 21.28 | 11.18 |
| **Proposed (Ours)** | **~61.5M** | **21.54** | **10.79** |

> ✅ **关键结论**：所提方法以 **更小的模型规模** 达到与最大型微调模型相当甚至更优的性能，且 **CER 显著更低**，说明其在构建复杂 Bengali 单词方面更具优势。

### 推理效率对比（Table 3）
| 模型 | Optimization | RTF |
|------|--------------|-----|
| Whisper-Medium | CTranslate2/Dual T4 | 0.0190 |
| Hishab-Titu Conformer | Standard PyTorch | 0.0120 |
| **Proposed Moonshine-base** | **Tokenizer Transplant** | **0.0053** |
| Vanilla Moonshine | Baseline (failed) | 0.0038（但 WER ≈100%） |

> ⚡ 所提模型比高度优化的 Whisper 部署方案 **快 3.5 倍以上**，同时恢复了可用精度。

### 消融实验结果（隐含分析）
虽然未明确列出消融表，但从方法设计可推断以下关键组件的作用：
- **Phase 1 先进行声学微调**：确保 encoder 已适应 Bengali 语音特性，避免后续移植破坏已有知识；
- **两阶段恢复训练**：高学习率初期快速对齐新 embedding 空间，防止“移植排斥”；
- **native-script tokenizer 使用**：直接决定 fertility 下降幅度和 decoding 稳定性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Tokenizer fertility 是影响边缘高效 ASR 在形态复杂语言上表现的关键瓶颈**；
2. **Autoregressive collapse 可通过压缩输出序列长度有效缓解**；
3. **解耦 acoustic learning 与 vocabulary alignment 是一种被低估但极其有效的策略**；
4. **无需重新预训练，仅通过 vocabulary transplant + recovery tuning 即可实现高性能跨语言迁移**；
5. 所提方法实现了 **性能、效率与小型化的最佳平衡**，特别适用于低资源语言的边缘部署。

### 方法的局限性
- 当前方法依赖于目标语言已有高质量 tokenizer（如 BanglaBERT），若缺乏此类资源则难以应用；
- 仅验证了 Bengali 场景，是否泛化至其他粘着语（如 Tamil、Telugu）尚待验证；
- embedding resize 后仍存在一定程度的信息损失，未探索更精细的初始化方式（如 WECHSEL 或 FOCUS 中的方法）；
- 未讨论 streaming inference 支持情况。

### 未来工作方向
- 将 **Tokenizer Transplantation** 框架推广至更多低资源语言（如 Assamese、Odia 等 Indic languages）；
- 探索 **automatic vocabulary construction** 机制，以支持无现成 BERT tokenizer 的语言；
- 结合 **TokAlign** 或 **Trans-Tokenization** 技术提升跨 tokenizer 对齐质量；
- 研究如何将该范式应用于 **multilingual ASR** 场景下的动态 vocabulary switching；
- 开发自动化工具链，使该 pipeline 更易于社区复现与部署。

> 🔗 代码仓库公开：[https://github.com/Sanjidh090/moonshine-base-bn](https://github.com/Sanjidh090/moonshine-base-bn)

</details>

---

### 4. [FreyaTTS Technical Report](https://arxiv.org/abs/2607.09530)

**Authors**: Ahmet Erdem Pamuk, \"Omer Yent\"ur, Ahmet Tunga Bayrak, Yavuz Alp Sencer \"Ozt\"urk, Mustafa Yavuz  
**Category**: cs.CL  
**Published**: 2026-07-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.09530v1  

#### Abstract
We introduce Freya-TTS, a compact, tokenizer-free, Turkish-first text-to-speech model designed for highly reliable and efficient conversational synthesis. Freya-TTS is a 183.2M-parameter non-autoregressive conditional flow-matching Diffusion Transformer (DiT) that operates in the frozen continuous l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# FreyaTTS Technical Report 核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

- **土耳其语TTS资源不足**：尽管土耳其语属于中等资源语言，但现有的大规模多语言TTS系统（如CosyVoice、Qwen3-TTS）将其作为众多语言之一处理，缺乏针对性优化，导致合成质量不佳。
- **前端规则脆弱**：土耳其语具有**agglutinative morphology**（黏着构词）、**vowel harmony**（元音和谐）以及数字、缩写等复杂发音规则，传统基于**grapheme-to-phoneme (G2P)** 或文本归一化规则的前端容易出错且难以覆盖所有情况。
- **部署效率低**：当前最先进的TTS模型参数量大（数十亿级），依赖多GPU训练和推理，难以在边缘设备（如手机、嵌入式系统）上实时运行。

### **提出了什么新方法或新思路**

FreyaTTS 是一个专为土耳其语设计的紧凑型、**tokenizer-free**、**non-autoregressive (NAR)** 文本到语音模型，其核心创新如下：

1. ✅ **Rule-Free End-to-End Modeling**
   - 使用**92符号的纯字符级输入**（raw characters），不依赖任何**phonemizer**、**G2P frontend** 或离散语音token（如Codec tokens）。
   - 所有发音规则（包括数字读法、缩写、元音和谐）直接从音频中学习，实现真正的端到端建模。

2. ✅ **Non-Autoregressive Parallel Denoising**
   - 采用基于**conditional flow-matching**的**Diffusion Transformer (DiT)** 架构，在冻结的连续隐空间（frozen continuous-latent space）中并行去噪整个语音序列。
   - 避免了自回归模型中的**left-to-right error accumulation**（尤其是长数字串易出现的“garbling”问题）。

3. ✅ **Production-Hardening Post-Training Recipe**
   - 提出两阶段微调策略：
     - **Single-Speaker Voice Lock**：通过全参数微调将目标说话人身份“写入”模型权重，显著提升音色一致性（Fo标准差从74.9Hz降至5.0Hz）。
     - **Short-Utterance Coverage**：专门用单字/双字短句进行微调，解决孤立词（如“evet”, “bir”）合成失败的问题。

4. ✅ **利用冻结的高保真Codec**
   - 复用 **VoxCPM2** 中的 **frozen AudioVAE2** 编解码器（16kHz编码 → 48kHz解码），固定重建能力。
   - 模型仅需专注于**text-to-latent mapping**，无需参与波形建模，极大提升了效率与稳定性。

### **相比现有方法的优势**

| 维度 | FreyaTTS | 传统方法 |
|------|---------|--------|
| **架构** | NAR DiT + Flow Matching | AR 或 多阶段（Semantic + Acoustic） |
| **输入处理** | 字符级，无G2P | 依赖G2P或BPE |
| **语音表示** | 连续隐变量（Continuous Latent） | 离散Token（Quantized） |
| **推理速度** | 实时因子 RTF ≈ 0.11（H100） | 通常 > 0.3 |
| **部署友好性** | 可在笔记本CPU上实时运行 | 依赖高端GPU |
| **参数量** | 183.2M | 主流开源模型达300M–2B |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **训练数据**：
  - 内部大规模高质量土耳其语语音语料库，包含：
    - 数字朗读
    - 名单朗读
    - 对话句子
    - 通用语句
  - 多说话人预训练 → 单说话人微调（SFT-v1 和 SFT-v2）
- **评估基准**：
  - **Freya-TR-Eval**：公开发布的土耳其语评测集
    - 包含 **495条** 自然、领域中立的日常对话句（3–13词）
    - 来源混合：Common Voice 17-tr、CoVoST2-tr 和 Gemini-3.1-Pro生成
    - 完全独立于训练集，确保泛化性
    - 已发布于 Hugging Face：`freyavoice/freya-tr-eval`

### **实验设置和评估指标**

#### **评估指标**

| 指标 | 描述 |
|------|------|
| **WER / CER** | 使用 **Whisper-large-v3** 转录后计算词错误率（WER）和字符错误率（CER）<br>✅ **Band-matched**：所有输出先下采样至8kHz再转录，公平比较窄带模型 |
| **MOS** | 主观自然度评分，由24名母语者盲测打分（5分制，95%置信区间） |
| **RTF (Real-Time Factor)** | 推理时间 / 音频时长，batch size=1，包含AudioVAE解码 |
| **Fo Standard Deviation** | 同一句子多次生成的基频标准差，衡量音色一致性 |
| **Content Similarity** | 输入文本与ASR转录之间的归一化编辑距离 |

#### **基线方法对比**

| 模型 | 参数量 | 类型 | 是否开源 |
|------|--------|-------|----------|
| Piper (tr) | ~16M | VITS-based | ✅ |
| Coqui GlowTTS (tr) | ~28M | Flow-based | ✅ |
| MMS-TTS (tr) | ~36M | VITS multilingual | ✅ |
| SpeechT5 (tr) | ~144M | Fine-tuned ASR model | ✅ |
| F5-TTS (tr) | ~336M | Flow-matching zero-shot cloner | ✅ |
| XTTS-v2 (multi) | ~470M | Multilingual zero-shot cloner | ✅ |

> 所有模型均使用相同协议评估，FreyaTTS为单一声线模型，无需参考音频；XTTS-v2/F5-TTS提供统一参考音频以保证可比性。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 指标 | FreyaTTS 结果 |
|------|----------------|
| **WER ↓** | **8.0%** |
| **CER ↓** | **3.0%** |
| **MOS ↑** | **3.68 ± 0.22** |
| **RTF (median)** | **~0.14**（H100） / **0.11**（RTX 4090） |
| **VRAM占用** | **1.5 GB**（fp32/bf16） |
| **Fo Std Dev** | **5.0 Hz**（微调后） |
| **Content Similarity** | **0.923** |

### **与基线方法的对比结果**

#### 表：在 Freya-TR-Eval 上的性能对比（部分）

| System | Params | WER↓ | CER↓ | MOS↑ |
|--------|--------|------|------|------|
| Piper (tr) | 16M | 4.4% | 1.1% | 3.47±0.22 |
| MMS-TTS (tr) | 36M | 6.8% | 1.7% | 3.58±0.20 |
| **FreyaTTS (ours)** | **183.2M** | **8.0%** | **3.0%** | **3.68±0.22** |
| F5-TTS (tr) | 336M | 24.3% | 10.9% | 3.63±0.23 |
| XTTS-v2 (multi) | 470M | 11.1% | 3.9% | 3.82±0.19 |

> 🔍 **分析**：
> - 尽管 **WER高于Piper/MMS-TTS**，但 FreyaTTS 在**自然度（MOS）上全面超越除XTTS外的所有模型**，说明其发音更自然流畅。
> - 相比更大规模的零样本克隆模型（F5-TTS、XTTS-v2），FreyaTTS 在 **WER上分别降低16.3pp 和 3.1pp**，同时保持更高或相当的自然度。
> - **首次证明：一个183M参数的纯端到端NAR模型可在土耳其语任务上超越数倍更大的开源系统**。

### **消融实验与验证**

1. **Bootstrap Confidence Interval**
   - WER: 7.9% [6.8%, 9.1%]
   - CER: 2.9% [2.4%, 3.4%]
   - 说明主结果稳定可靠。

2. **协议校准（Protocol Calibration）**
   - Human recordings (FLEURS-tr): **9.7% WER**
   - AudioVAE2重构真实录音：**2.6% WER损失**
   - ⇒ FreyaTTS的8.0%接近人类水平（仅差1.7pp），且非常接近Codec上限。

3. **推理层影响（Ablation of Inference Layer）**
   - 移除clause chunking、duration floor、voicing retry等机制后，WER最多上升0.3%，表明这些是“尾部保险”，非核心依赖。

4. **长度敏感性测试**
   - 随输入长度增加，错误率上升（long-horizon drift）
   - clause chunking 可有效缓解该问题（见Fig 2c）

---

## 4. 关键结论和发现

### **论文的主要发现**

1. ✅ **Tokenizer-free + NAR 是高效可靠的TTS路径**
   - 无需G2P或离散token，直接从字符学习发音规则，在土耳其语这类形态复杂的语言上更具优势。

2. ✅ **小模型也能打败大模型**
   - FreyaTTS（183M）在质量和效率上全面优于参数量为其1.8–2.6倍的F5-TTS和XTTS-v2，挑战了“越大越好”的范式。

3. ✅ **Voice Lock 显著提升一致性**
   - 通过全参数微调将单一说话人身份固化进权重，Fo标准差从74.9Hz降至5.0Hz，几乎消除性别翻转现象。

4. ✅ **边缘部署成为可能**
   - 在消费级GPU上RTF≈0.11，在笔记本CPU（Apple M3）上可达RTF≈0.7–1.0，支持**real-time synthesis**。
   - Core ML编译后神经引擎上RTF低至 **0.047**。

### **方法的局限性**

1. ❗ **仍受限于窄带保真度**
   - 训练数据为窄带（telephony-band），即使输出为48kHz，实际音质上限仍受16kHz输入限制。

2. ❗ **数字串需前端扩展**
   - 当前duration predictor对紧凑的数字字符串（如"2025"）分配帧数不足，需在前端手动展开为“iki bin yirmi beş”。

3. ❗ **孤立词仍具挑战**
   - 即使经过short-utterance coverage微调，极短输入（如单个字母）仍有失败风险。

4. ❗ **仅支持单一声线**
   - 当前版本未支持多说话人切换或风格控制。

### **未来工作方向**

1. 🔄 开发联合训练的 **CTC-monotonic aligner**，解决残余的word-skip和long-horizon drift问题。
2. 🎯 引入基于偏好的后训练（**Preference-based Post-Training**），利用intelligibility和voicing reward优化生成质量。
3. 👥 扩展voice-lock机制至**多说话人支持**与**自然语言驱动的音色设计**（natural-language voice design）。
4. 🔊 实现显式的**bandwidth extension**技术，使窄带训练的模型能输出真正宽频音频。
5. 🌐 探索向其他中等资源语言迁移的可能性，构建系列轻量级本地化TTS系统。

---

> ✅ **总结一句话**：  
> FreyaTTS 展示了一条**面向中等资源语言、高可靠性、可边缘部署**的新一代TTS路径——**小而精、端到端、免规则、高一致**，并在土耳其语上实现了对更大开源模型的全面反超。

</details>

---

### 5. [Federated Low-Rank Koopman Learning for Multivariate Time-Series Anomaly Detection in IoT Systems](https://arxiv.org/abs/2607.08978)

**Authors**: Tung-Anh Nguyen, Van-Phuc Bui, Anh Tuyen Le, Kim Hue Ta, Minh Thuy Le, J. Andrew Zhang, Xiaojing Huang  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.08978v1  

#### Abstract
Distributed IoT systems generate multivariate time-series streams for monitoring physical assets, servers, and embedded sensing platforms. Detecting abnormal temporal behavior is critical for fault diagnosis, predictive maintenance, and security. However, practical IoT anomaly detection is hindered ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Federated Low-Rank Koopman Learning for Multivariate Time-Series Anomaly Detection in IoT Systems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**资源受限的分布式 IoT 系统中的多变量时间序列（MVTS）异常检测**，解决了以下挑战：
- **数据去中心化与非独立同分布（non-IID）**：各设备采集的数据在特征分布、标签比例和样本数量上存在显著异构性。
- **通信带宽有限**：传统联邦学习中传输大型神经网络参数开销巨大。
- **边缘设备计算与内存受限**：难以部署复杂的深度模型进行本地训练和推理。
- **隐私保护需求**：原始传感器数据不能上传至服务器。

### 🚀 提出的新方法：FedKAD
提出 **FedKAD**（Federated Koopman Anomaly Detection），一种轻量级、高效的联邦异常检测框架，其核心思想是：
- 利用 **Koopman operator 理论** 对非线性动态系统建模，将复杂的时间演化转化为高维可观测空间中的线性动力学。
- 引入 **低秩子空间分解（Low-Rank Subspace Factorization）**，仅共享紧凑的正交子空间变量（orthonormal subspace），而客户端保留私有的简化动力学算子（reduced operator）。
- 设计 **Federated Stiefel-ADMM 算法**，在满足正交约束（Stiefel manifold）的前提下实现跨客户端共识优化，支持部分客户端参与。

### 🔍 相比现有方法的优势
| 维度 | FedKAD | 联邦深度学习基线（如 LSTM-AE, USAD, TranAD） |
|------|--------|---------------------------------------------|
| **通信成本** | 极低（仅交换 $ \mathbb{R}^{D \times r} $ 子空间矩阵） | 高（需传输完整模型参数或梯度） |
| **训练速度** | 快达 $ 2.1\times10^3 $ 倍 | 慢，依赖多次本地 SGD 迭代 |
| **推理延迟** | 低至微秒级（$ \mu s $/step） | 较高，尤其 Transformer 类模型 |
| **内存占用** | 小（避免存储大模型） | 大 |
| **对 non-IID 数据鲁棒性** | 强（本地动态可个性化） | 弱（易受局部数据偏移影响） |

> ✅ **核心优势总结**：**在保持甚至提升检测性能的同时，极大降低资源消耗，真正适配边缘部署场景。**

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
在四个广泛使用的 MVTS 异常检测基准上进行评估：

| 数据集 | 描述 | 维度 | 异常率 | 客户端数（N） |
|-------|------|------|--------|--------------|
| **SMD** | 服务器监控数据（CPU、内存等） | 38 | 4.16% | 28 |
| **PSM** | eBay 池化服务器指标 | 25 | 27.75% | 24 |
| **SMAP** | NASA 土壤湿度卫星遥测 | 25 | 12.85% | 54 |
| **MSL** | NASA 火星探测器遥测 | 55 | 10.53% | 27 |

> 所有数据均按实体自然划分或通过 Dirichlet 分区构造 non-IID 场景。

### ⚙️ 实验设置
- **联邦架构**：采用 **FedAvg** 框架，每轮随机采样 25% 客户端参与。
- **训练配置**：
  - 全局通信轮次：30
  - 本地 epoch 数：5
  - 优化器：Adam ($ lr=10^{-3} $)
  - 批大小：128
  - 滑动窗口长度：20
- **FedKAD 参数**：
  - 提升维度 $ d_{\text{lift}} = 128 $
  - 子空间秩 $ r = 24 $（PSM 为 32）
  - ADMM 正则化参数 $ \rho = 1.0 $
  - Stiefel-ADMM 步长：0.05

### 📈 评估指标
采用 **per-client macro-aggregated metrics**，防止长序列主导结果：
- **Precision, Recall, F1, AUC**
- 四种阈值调整协议以增强可比性：
  1. **Primary PA%K ($ k=0.01 $)**：主流段级评估标准，要求至少 1% 的异常点被检出才算命中。
  2. **POT + any-hit PA ($ k=0 $)**：经典盲调阈值 + 单点命中即成功。
  3. **POT + PA%K ($ k=0.1 $)**：更严格段覆盖要求（10%）。
  4. **Point-wise no-PA**：严格的逐点 F1，无任何调整。

### 🆚 基线方法
所有基线均封装于相同 FedAvg 框架下，确保公平比较：
1. **DeepSVDD**：单类分类深度网络
2. **LSTM-AE**：基于 LSTM 的自编码器
3. **USAD**：对抗式自编码器用于无监督检测
4. **TranAD**：基于 Transformer 的重构模型

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Primary PA%K, $ k=0.01 $）

| 方法 | SMD (F1) | PSM (F1) | SMAP (F1) | MSL (F1) |
|------|----------|----------|-----------|----------|
| **FedKAD** | **72.20±1.28** | **69.87±0.21** | **53.32±0.50** | 39.55±0.02 |
| LSTM-AE | 64.51 | 68.57 | 51.56 | **40.50±4.63** |
| USAD | 53.2 | 53.3 | 44.3 | 35.4 |
| TranAD | 59.8 | 63.9 | 49.3 | 32.7 |

> ✅ FedKAD 在 **SMD、PSM、SMAP** 上取得最佳 F1，仅在 MSL 微弱落后。

#### 统计显著性检验（vs 最强基线）

| Dataset | FedKAD F1 | Best Baseline | Margin | p-value | Outcome |
|--------|------------|----------------|--------|---------|---------|
| SMD    | 72.20      | LSTM-AE (64.51) | +7.69  | 0.038   | WIN ✅ |
| PSM    | 69.87      | LSTM-AE (68.57) | +1.29  | 0.009   | WIN ✅ |
| SMAP   | 53.32      | LSTM-AE (51.56) | +1.76  | 0.25    | WIN ✅（超预设 margin） |
| MSL    | 39.55      | LSTM-AE (40.50) | -0.95  | 0.80    | LOSS ❌（差距极小） |

> 💡 结论：在 3/4 数据集上统计显著或实质性领先。

---

### ⚡ 资源效率对比（四大优势）

| 指标 | FedKAD 表现 | 相对提升 |
|------|-------------|----------|
| **训练时间** | 1.7–4.8 秒（端到端） | 比最慢基线快 **200–2000×**（最高达 $ 2.1\times10^3 $ 倍） |
| **每轮通信量** | 1.0–1.7 MB | 减少 **3–40×**（较 USAD/TranAD 下降 80×） |
| **推理延迟** | 1.9–5.8 μs/step | 低于或接近所有基线 |
| **树莓派部署实测**（Raspberry Pi 4） | 训练 0.23s/轮，推理 0.79μs/step，上传 59KB/轮 | 相比 TranAD 加速 **25× 训练、31× 推理**，通信减少 **30×** |

> ✅ **唯一同时满足“高性能”与“可部署”的方法**。

---

### 🔍 消融与补充实验结果（Table IV）

| 协议 | FedKAD 表现 | 发现 |
|------|-------------|------|
| **POT + any-hit PA** | SMD: 89.19, PSM: 98.88 | 在宽松协议下仍具竞争力 |
| **POT + PA%K ($ k=0.1 $)** | SMAP: 37.28 vs TranAD 44.78 | 更严格段覆盖下略逊色 |
| **Point-wise no-PA** | SMAP: 3.87 vs DeepSVDD 15.60 | 不擅长精确时序定位，但 MSL 上反超 |

> 🔎 **关键洞察**：FedKAD 的优势集中在 **segment-level 异常检测**，而非 point-level 精确定位。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Koopman 动力学建模可用于高效联邦异常检测**：通过低秩线性近似捕捉正常行为模式，无需复杂神经网络。
2. **共享子空间 + 本地动态分离设计有效平衡共性与个性**：既利用全局一致性提升泛化能力，又适应 non-IID 数据下的局部差异。
3. **Federated Stiefel-ADMM 收敛性良好**：理论证明在部分客户端参与下仍能收敛至稳定解，且实验验证共识误差逐渐消失。
4. **资源效率远超深度学习方法**：特别适合边缘设备部署，在真实硬件（如 Raspberry Pi 4）上表现优异。

### ⚠️ 局限性
- **对 point-wise 定位能力较弱**：基于预测残差的方法难以精确定位异常起始时刻。
- **固定 lifting 函数表达力有限**：虽使用 tanh 非线性增强，但仍不如可学习 embedding 灵活。
- **子空间秩 $ r $ 需手动设定**：缺乏自动选择机制，可能影响不同任务的表现。
- **假设“正常动态”可被线性表示**：极端非线性或突变系统可能建模不准。

### 🔮 未来工作方向
- **自适应秩选择机制**：根据数据复杂度动态调整子空间维度。
- **个性化异常评分阈值**：结合本地统计特性优化检测灵敏度。
- **更强的隐私保障**：引入差分隐私或安全聚合（Secure Aggregation）进一步保护共享变量。
- **扩展至其他任务**：如联邦时间序列预测、故障根因分析等。

---

## 总结
> **FedKAD 是首个将 Koopman 动力学与联邦学习深度融合的轻量级 MVTS 异常检测框架**。它不仅在多个真实 IoT 数据集上实现了与深度模型相当甚至更优的检测性能，更重要的是在 **训练速度、通信开销、推理延迟** 上实现了数量级的突破，**真正推动了智能边缘（Edge Intelligence）在实际 IoT 系统中的落地应用**。

</details>

---

### 6. [Graph Neural Networks for Scalable and Transferable Node Centrality Approximation](https://arxiv.org/abs/2607.09372)

**Authors**: Samra Sana, Giorgio Mantica, Saul Imbrici  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.09372v1  

#### Abstract
Graph Neural Networks (GNNs) provide a learning-based framework for approximating graph quantities that are expensive to compute exactly. This paper investigates GNNs for scalable approximation of betweenness and closeness centrality, formulated as a node-ranking problem. Exact centrality values are...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Graph Neural Networks for Scalable and Transferable Node Centrality Approximation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于**节点中心性（node centrality）计算中的可扩展性和泛化能力挑战**。具体而言：
- **Betweenness Centrality** 和 **Closeness Centrality** 虽然在社交网络、生物网络等领域广泛应用，但其精确计算复杂度高（尤其是 Betweenness 需要多次最短路径计算），难以应用于大规模图。
- 现有基于采样的近似方法（如 Brandes sampling）依赖采样预算，在精度与效率之间权衡困难。
- 更重要的是，大多数学习型方法在训练分布外（out-of-distribution, OOD）图上表现不佳，缺乏**跨拓扑结构的可迁移性（transferability）**。

### 提出的新方法与新思路
- 将 **centrality approximation 形式化为一个监督式的 node-ranking 任务**，目标是学习由精确中心性诱导的节点排序，而非直接回归数值。
- 设计了基于 **Message-Passing GNN** 的模型架构：
  - 对 **Betweenness Centrality** 使用**双通路（dual-pathway）GNN**，分别处理原始邻接矩阵 $A$ 与其转置 $A^T$，以捕捉有向图中入流与出流的最短路径结构，并通过 element-wise product 进行融合。
  - 对 **Closeness Centrality** 使用单通路深层 GNN（$L=7$ 层），聚合更广范围的距离信息。
- 引入 **混合分布训练（mixed-distribution training）**：在 Erdős–Rényi (ER)、Barabási–Albert (BA) 和 Gaussian Random Partition (GRP) 三种不同拓扑结构的合成图上联合训练，提升模型对未见图结构的泛化能力。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **效率** | 推理阶段实现高达 **97.7× 的速度提升**，远超 exact 算法和采样方法 |
| **可扩展性** | 在 $N=5,000$ 的大图上仍保持高性能（$\tau=0.938$） |
| **可迁移性** | 混合训练显著提升跨图族泛化能力，尤其对 Betweenness |
| **通用性** | 支持 zero-shot transfer 到真实世界网络（如 C. Elegans、Email-Eu-Core、Power Grid） |

---

## 2. 核心实验方法和设置

### 数据集
所有图均为合成生成（使用 NetworkX）：
- **Erdős–Rényi (ER)**：均匀随机连接，$N=200$, $p=0.15$
- **Barabási–Albert (BA)**：无标度网络，偏好连接机制
- **Gaussian Random Partition (GRP)**：模块化社区结构，$s=20$, $v=5$, $p_{in}=0.3$, $p_{out}=0.05$

此外还进行了 zero-shot 测试于以下真实网络：
- **C. Elegans 神经网络**
- **Email-Eu-Core**
- **Western US Power Grid**

### 实验设置
- **任务形式**：将中心性近似建模为 **supervised node ranking problem**
- **标签生成**：使用 NetworkX 计算 exact betweenness/closeness centrality 作为监督信号
- **训练/验证/测试划分**：70%/10%/20%
- **模型输入**：仅图结构（邻接矩阵），无额外节点特征
- **损失函数**：Pairwise Ranking Loss（MarginRankingLoss），margin=1.0
- **优化器**：Adam，学习率 $10^{-3}$，dropout、weight decay 等通过网格搜索确定

### 评估指标
- **主指标**：**Kendall’s Tau ($\tau$)** —— 衡量预测排名与真实排名的一致性
- **辅助指标**：
  - 推理时间（ms）
  - 速度提升倍数（Speedup）
  - 消融实验中的 $\tau$ 变化

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Random Ranking** | 随机打分，$\tau \approx 0$ |
| **Degree Centrality** | 最简单的启发式方法，高效但非学习型 |
| **Brandes Sampling** | 经典采样法，$k=10,20,50$ 个 pivot 节点 |
| **Single-pathway / Sum Fusion / Different Depths** | 架构消融对照 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 场景 | 方法 | Kendall’s $\tau$ | 备注 |
|------|------|------------------|------|
| **Unseen ER 图 ($N=200$)** | GNN (Betweenness) | **0.851 ± 0.011** | |
| | GNN (Closeness) | **0.894 ± 0.011** | |
| **Large Scale ($N=5,000$)** | Betweenness GNN | **0.938** | 显示强可扩展性 |
| **Cross-Family Transfer (Mixed Training)** | Betweenness GNN | ER: 0.878, BA: 0.920, GRP: **0.861** | 显著优于单一训练 |
| **Zero-Shot Real-World** | Betweenness | C. Elegans: 0.782, Email: 0.654, Power Grid: 0.603 | 跨域有效 |
| | Closeness | C. Elegans: 0.703, Email: 0.265, Power Grid: **0.108** | 敏感性强 |

### 与基线方法对比结果
#### 在 ER 图上的表现（$N=200$）：
| Method | Betweenness $\tau$ | Closeness $\tau$ | Time (ms) |
|--------|--------------------|------------------|-----------|
| Random | ~0 | ~0 | 0 |
| Degree Centrality | 0.886 | 0.923 | 0.07 |
| Brandes $k=50$ | 0.552 | – | 35.2 |
| **GNN (Ours)** | **0.851** | **0.894** | **9.7 / 23.8** |

> ✅ GNN 在保持接近 degree 的精度的同时，提供了可训练、可泛化的表示；且推理速度快于 high-$k$ Brandes。

#### 跨图族 Betweenness 性能比较（Mixed GNN vs Brandes）：
| Graph Type | Brandes $k=50$ | Mixed GNN |
|------------|------------------|-----------|
| ER | 0.434 | **0.878** |
| BA | 0.747 | **0.920** |
| GRP | 0.629 | **0.861** |

> ✅ Mixed GNN 在所有图类型上均显著优于采样方法，尤其在 GRP 上优势明显。

### 消融实验结果（Ablation Study）
| 变体 | ER $\tau$ | GRP $\tau$ | 发现 |
|------|----------|-----------|------|
| Single Pathway | 0.471 | 0.390 | 双通路至关重要 |
| Dual + Sum Fusion | 0.827 | 0.679 | Sum 泛化更好但 ER 性能略低 |
| **Dual + Product Fusion** | **0.846** | 0.493 | Product 提升 in-distribution 性能 |
| Depth $L=3,5,7$ | 非单调变化 | | 最优深度需平衡感受野与过平滑 |

> 🔍 结论：**Product fusion 有助于捕获双向控制流，但可能不利于跨结构泛化**；而 **Sum fusion 更鲁棒**。深度增加并不总有益，存在 trade-off。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **GNN 可有效学习 centrality-induced node ranking**：
   - 在未见 ER 图上达到 $\tau > 0.85$，表明模型能从图结构中提取有意义的结构性表示。

2. ✅ **混合分布训练显著提升 transferability**：
   - 特别是对 **Betweenness Centrality**，混合训练使 GRP 图上的性能从 0.552 提升至 0.861，验证了“结构多样性即正则化”的假设。

3. ⚠️ **Closeness Centrality 更敏感于拓扑偏移（topology shift）**：
   - 在 GRP 和真实网络（特别是 Power Grid）上性能急剧下降（最低至 $\tau=0.108$），因其严重依赖全局距离结构，在模块化或空间嵌入图中局部邻居无法充分代理平均距离。

4. ✅ **GNN 推理具有极高的计算效率**：
   - 最高可达 **97.7× 速度提升**，适用于需要实时或批量处理大量图的应用场景。

5. ✅ **模型具备良好的可扩展性**：
   - 在 $N=5,000$ 的大图上仍取得 $\tau=0.938$，说明该框架可推广到实际规模网络。

### 方法的局限性
1. ❌ **Closeness Centrality 泛化能力差**：
   - 当前架构难以应对社区结构或长路径网络中的距离异质性，亟需改进。

2. ❌ **合成图与真实网络仍有差距**：
   - 尽管在部分真实网络上有一定 transfer 效果，但仍受限于合成图的理想化假设（如均匀边概率、规则社区大小等）。

3. ❌ **未探索更大规模异构图**：
   - 实验最大仅到 $N=5,000$，尚未验证在百万级节点图上的表现。

4. ❌ **静态图假设**：
   - 所有实验基于静态图，未考虑动态或时序图场景。

### 未来工作方向
1. 🔄 **增强 Closeness Centrality 的拓扑鲁棒性**：
   - 引入 topology-aware 模块（如位置编码、层次聚合）、更多样化的训练分布（含空间图、树状图等）。

2. 🌐 **引入真实网络进行混合预训练或 domain adaptation**：
   - 利用少量真实图微调或设计 invariant learning 策略，缩小合成-现实鸿沟。

3. 🧱 **探索 GNN Foundation Models for Centrality**：
   - 开发通用图表示预训练框架，支持多任务、多中心性联合学习。

4. 📈 **扩展至更大规模与异构图**：
   - 结合图采样、子图蒸馏等技术，适配工业级图数据。

5. 🤖 **结合 symbolic reasoning 与 neural approximation**：
   - 探索 neuro-symbolic 方法，让 GNN 学习“如何计算最短路径影响”，提升可解释性与泛化性。

--- 

> 💡 **总体评价**：本文系统地研究了 GNN 在 **scalable 与 transferable centrality approximation** 中的能力，提出了有效的架构设计与训练策略，实证揭示了 **betweenness 与 closeness 在可迁移性上的根本差异**，为后续图算法神经逼近研究提供了重要基准与洞见。

</details>

---

### 7. [Distributed Symmetry Breaking on Hyperbolic Random Graphs](https://arxiv.org/abs/2607.09170)

**Authors**: Yannic Maus, Janosch Ruff, Sonia Simons, George Skretas  
**Category**: cs.DC  
**Published**: 2026-07-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.09170v1  

#### Abstract
Real-world networks like the internet share patterns like a power law degree distribution and a high clustering coefficient. Many of these properties are captured by the generative model of hyperbolic random graphs (HRGs), which provides a theoretical framework for studying such networks. Motivated ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Distributed Symmetry Breaking on Hyperbolic Random Graphs

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文研究了在**超双曲随机图 (Hyperbolic Random Graphs, HRGs)** 上的分布式对称性破缺 (symmetry breaking) 问题，特别是 **Maximal Independent Set (MIS)** 和 **Maximal Matching (MM)** 这两个经典问题。

现实世界网络（如互联网）通常具有幂律度分布 (power-law degree distribution) 和高聚类系数 (high clustering coefficient) 等特性。传统的最坏情况分析 (worst-case analysis) 得出的算法复杂度下界（如 Ω(log n)）可能无法解释这些算法在真实网络上的优异表现。本文旨在探究：当输入图是更符合现实网络特性的 HRG 时，MIS 和 MM 问题的复杂度是否可以被显著降低。

### 提出了什么新方法或新思路
本文提出了以下三个层面的新方法和理论洞察：

1.  **针对 HRG 的高效算法 (Theorem 1)**：
    *   **新思路**：提出了一种基于**几何分治 (geometric shattering)** 的新论证方式。不同于以往依赖于 Luby 算法迭代的“shattering”框架，作者利用 HRG 的底层双曲几何结构，设计了一个常数轮次的随机化过程，通过移除一个内盘 (inner disk) 和创建狭窄的角扇区 (angular sectors)，将图分解成大小为 polylog n 的连通分量。
    *   **方法**：对于 MIS，首先激活特定环形区域的节点进行一轮 Luby 式选择以分割外环；然后激活边界附近的低度节点，确保内盘中的所有节点都被覆盖。对于 MM，采用类似的策略，但需要更复杂的匹配机制来处理边的匹配。

2.  **紧致的复杂度下界 (Theorem 2)**：
    *   **新思路**：证明了 MIS 和 MM 在 HRG 上的固有难度。作者的关键洞察是，HRG 的巨分量 (giant component) 中必然包含大量**诱导的 d-ary 树 (induced d-ary trees)**，其深度和度数都相当大。
    *   **方法**：通过严格的概率分析，证明了存在多项式数量的 `d-ary` 树，其中 `d ≈ log log n`, `h ≈ log log n / log log log n`。由于在树上解决 MIS/MM 本身就有已知的下界，这直接导出了在 HRG 上的 `Ω(log log n / log log log n)` 轮次下界。

3.  **利用几何信息的感知算法 (Theorem 3)**：
    *   **新思路**：探讨了如果节点知道其**几何嵌入 (geometric embedding)**（即在双曲圆盘中的坐标），算法性能能否进一步提升。
    *   **方法**：设计了不依赖“shattering”的确定性算法。通过将外环部分划分为**瓦片 (tiling)**，并按层顺序处理，使得节点可以并行地在独立的瓦片上求解局部问题。

### 相比现有方法的优势
*   **指数级加速**：相比通用图上 `Ω(min{log Δ, log n})` 的下界，本文的算法在 HRG 上实现了 `O(log^{5/3} log n)` 的复杂度，这是一个指数级的改进。
*   **理论完备性**：不仅提供了上界，还给出了几乎紧致的下界，完整刻画了 HRG 上 MIS/MM 问题的复杂度景观，揭示了它与 `Δ+1` 着色问题（可在 2 轮内解决）的根本区别。
*   **新结构发现**：发现了 HRG 中存在大规模 `d-ary` 树这一结构性质，这不仅是证明下界的工具，本身也是对 HRG 模型理解的重要贡献。
*   **信息利用的价值**：首次展示了访问几何嵌入信息可以带来巨大的性能提升（例如，MM 从 `O(log log log n)` 到 `O(log log log n)`），并量化了计算嵌入信息本身的通信代价。

## 2. 核心实验方法和设置

**注意**：该论文是一篇**纯理论计算机科学**论文，其“实验”指的是**数学证明和形式化分析**，而非在真实数据集上运行代码的实证实验。

### 使用了哪些数据集
*   **理论模型**：论文完全基于**阈值超双曲随机图 (threshold hyperbolic random graph)** 的生成模型。这是一种理论上的随机图模型，用于抽象地表示具有幂律度分布和高聚类系数的真实网络。没有使用任何具体的真实世界数据集（如社交网络、互联网拓扑等）。

### 实验设置和评估指标
*   **计算模型**：分析在经典的 **LOCAL 模型**和 **CONGEST 模型**下进行。
    *   **LOCAL 模型**：每轮通信消息大小无限制。
    *   **CONGEST 模型**：每条消息最多 `O(log n)` 比特。
*   **评估指标**：核心指标是算法的**轮复杂度 (round complexity)**，即算法终止所需的同步轮数。目标是证明上界（upper bound）和下界（lower bound）。

### 基线方法对比
*   **通用图算法**：作为主要对比，论文引用了在一般图上解决 MIS/MM 的最佳已知算法和下界，例如：
    *   Luby 算法：`O(log n)` 轮。
    *   Khoury 和 Schild 的下界：`Ω(min{log Δ, log n})` 轮。
*   **其他图类算法**：也提到了在特殊图类（如有界增长图、有界独立数图）上的算法，以凸显 HRG 的独特性。
*   **相关工作**：特别对比了 Maus 和 Ruff 在同一 HRG 模型上关于 `Δ+1` 着色的工作，该工作仅需 2 轮，从而突显了 MIS/MM 问题的相对难度。

## 3. 主要实验结果和性能指标

### 关键性能数据
以下是论文通过理论证明得出的核心性能指标（轮复杂度）：

| 问题 | 模型 | 性能指标 | 来源 |
| :--- | :--- | :--- | :--- |
| **MIS** | LOCAL | `O(log^{5/3} log n)` 轮，w.e.h.p. | Theorem 1 |
| **MM** | LOCAL | `O(log^{5/3} log n)` 轮，w.e.h.p. | Theorem 1 |
| **MIS** | CONGEST | `O(log^3 log n)` 轮，w.e.h.p. | Theorem 1 |
| **MM** | CONGEST | `O(log^3 log n)` 轮，w.e.h.p. | Theorem 1 |
| **MIS/MM** | LOCAL | `Ω(log log n / log log log n)` 轮，a.a.s. | Theorem 2 |
| **MIS** | CONGEST (感知) | `O(log log n)` 轮，w.h.p. | Theorem 3 |
| **MM** | CONGEST (感知) | `O(log log log n)` 轮，w.h.p. | Theorem 3 |

### 与基线方法的对比结果
*   **vs. 通用图下界 (`Ω(min{log Δ, log n})`)**：本文的上界 `O(poly log log n)` 远优于通用图下界，因为 HRG 的最大度 `Δ` 是 `n` 的多项式级别，所以 `log Δ` 是 `Θ(log n)`。这证明了在 HRG 上确实可以实现指数级加速。
*   **vs. `Δ+1` 着色问题 (2 轮)**：尽管 MIS/MM 也得到了加速，但其下界 `Ω(log log n / log log log n)` 表明它们**不会**像着色问题一样坍缩到常数时间。这揭示了不同对称性破缺问题在 HRG 上的复杂度存在根本差异。
*   **vs. 感知 vs. 非感知模型**：对于 MM，当节点拥有几何坐标时，复杂度可以从 `O(log^{5/3} log n)` 大幅降低到 `O(log log log n)`，这是一个双重指数级的改进，凸显了几何信息的巨大价值。

### 消融实验结果
*   论文虽然没有传统意义上的消融实验，但在附录 **Appendix A** 中进行了一项关键的理论分析，相当于一种“思想实验”。
*   **结果**：作者证明了，即使对 HRG 执行任意常数轮次的 **Luby 算法**，剩余图中仍然会存在具有多项式度的节点和大的连通分量。这说明了标准的 Luby 迭代**不能**在常数轮内实现“shattering”，从而反衬出本文提出的**几何分治 (geometric shattering)** 新方法的必要性和创新性。

## 4. 关键结论和发现

### 论文的主要发现
1.  **MIS/MM 在 HRG 上可被指数级加速**：得益于 HRG 的几何结构，MIS 和 MM 可以在 `poly log log n` 轮内解决，远快于通用图上的 `Ω(log n)` 下界。
2.  **MIS/MM 与着色问题存在本质区别**：尽管 HRG 模型带来了巨大优势，但 MIS 和 MM 问题的复杂度下界 `Ω(log log n / log log log n)` 表明它们**本质上比 `Δ+1` 着色问题更难**，无法达到常数轮。
3.  **HRG 包含硬子结构**：HRG 的巨分量中必然包含大量大型的 `d-ary` 树，这些树是导致 MIS/MM 问题难以快速解决的“障碍”。这一结构性发现是证明下界的关键，并且本身具有独立的研究价值。
4.  **几何信息极具价值**：如果节点能够获取其几何坐标，算法性能可以得到质的飞跃。特别是对于 MM，复杂度可以降至 `O(log log log n)`，这揭示了在分布式计算中，利用隐藏的全局信息可以打破常规的复杂度壁垒。

### 方法的局限性
*   **理论模型限制**：所有结论都建立在理想化的 HRG 模型之上。真实世界的网络虽然具备类似特性，但可能不完全符合该模型的假设。
*   **几何信息的假设**：Theorem 3 的强大结果依赖于一个强假设——节点必须事先知道其精确的几何坐标。在实际系统中，获取或维护这种全局几何信息本身可能是一个昂贵的通信问题。
*   **非构造性证明**：下界证明是存在性的，它表明困难实例的存在，但并未提供一个具体的、可构造的对抗性图。

### 未来工作方向
论文在结论部分明确提出了几个开放问题 (open questions)：
*   **Luby 算法在 HRG 上的表现**：标准的 Luby 算法在 HRG 上的实际性能如何？
*   **MIS 与 MM 的分离**：在 HRG 上，MIS 是否比 MM 更容易？（类似于在树上的情况）
*   **突破 MIS 的下界**：能否设计出利用几何信息的算法，从而打破 Theorem 2 对于 MIS 的下界？
*   **近似算法的复杂度**：Maximum Matching 和 Maximum Independent Set 的**近似算法**在 HRG 上的复杂度如何？
*   **其他对称性破缺问题**：边着色 (edge colouring) 或顶点覆盖 (vertex cover) 等问题在 HRG 上的复杂度景观是否与 MIS/MM 或着色问题相似？

</details>

---

### 8. [GATS: Graph-Augmented Tree Search with Layered World Models for Efficient Agent Planning](https://arxiv.org/abs/2607.08894)

**Authors**: Maureese Williams, Dymitr Nowicki  
**Category**: cs.AI  
**Published**: 2026-07-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.08894v1  

#### Abstract
Large Language Model (LLM) agents have shown promise in multi-step planning tasks, but existing approaches like LATS (Language Agent Tree Search) and ReAct rely heavily on LLM inference during planning, leading to high computational costs and stochastic behavior. We present \textbf{GATS} (Graph-Augm...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GATS: Graph-Augmented Tree Search with Layered World Models for Efficient Agent Planning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Model (LLM)** 的智能体（如 ReAct、LATS）在多步规划任务中面临两大挑战：
- **高计算成本**：每一步规划都需要调用 LLM 进行推理，导致昂贵的推理开销。
- **随机性（Stochasticity）**：LLM 采样引入不确定性，导致计划不可复现。

此外，现有方法（如 LATS）依赖 LLM 在每个搜索节点进行动作提议和价值估计，效率低下。

---

### 🚀 提出的新方法：GATS
作者提出 **GATS (Graph-Augmented Tree Search)**，一种结合系统化搜索与分层世界模型（Layered World Model）的高效规划框架。

#### 核心创新点：
1. **分层世界模型（Layered World Model）**
   - **L1 (Symbolic)**：对已知动作进行精确的符号匹配（STRIPS-style），无需 LLM。
   - **L2 (Learned)**：从执行日志中学习统计规律，预测常见动作效果。
   - **L3 (Generative)**：仅对未知动作调用 LLM 预测，并缓存结果供后续复用。

2. **系统化 UCB1 搜索算法**
   - 使用 **UCB1-based tree search** 替代 LLM 引导的随机探索，实现更高效的路径搜索。
   - 所有状态转移由世界模型预测，**完全避免在规划阶段调用 LLM**。

3. **图增强的状态记忆（State-Transition Graph Memory）**
   - 显式维护一个持久化的状态转移图 `G = (V, E)`，支持跨路径共享搜索证据。
   - 不同路径到达相同状态时会合并节点（transposition table），显著减少重复计算。
   - 图结构在多个规划步骤间持续保留，实现搜索成本的跨步摊销。

---

### 🔍 相比现有方法的优势
| 特性 | GATS | LATS | ReAct |
|------|------|------|-------|
| 规划阶段 LLM 调用次数 | **0** | ~37/任务 | ~13/任务 |
| 成功率（合成任务） | **100%** | 92% | 64% |
| 结果可复现性 | **确定性（0% 方差）** | 有方差（1–5%） | 有方差 |
| 推理效率 | 极高（无实时 LLM 调用） | 低 | 中等 |
| 搜索策略 | 系统化 UCB1 探索 | LLM 引导采样 | 无搜索 |

> ✅ **核心优势总结**：  
> GATS 将 LLM 的角色从“在线决策者”转变为“离线世界模型构建者”，实现了**零 LLM 调用下的高效、确定性规划**。

---

## 2. 核心实验方法和设置

### 📚 数据集与任务设计

#### （1）合成多步规划任务（Synthetic Planning Tasks）
- 共 100 个任务，分为三类难度：
  - **Easy**：3 步，1 个死胡同分支
  - **Medium**：5 步，资源管理 + 分支选择
  - **Hard**：7+ 步，多个误导路径与死胡同
- 包含关键规划特性：
  - 序列依赖（Sequential dependencies）
  - 分支路径（Branching paths）
  - 死胡同（Dead-ends）
  - 资源约束（Resource constraints）

#### （2）压力测试（Stress Test）
- 设计 12 类真实场景挑战（共 120 个任务）：
  - `coding_task`：脚本开发流程（11 步）
  - `web_navigation`：邮件、订票、酒店预订
  - `deep_horizon` / `very_long_horizon`：长程规划（12–15 步）
  - `no_backtrack`：迷宫锁门机制（错误即失败）
  - `high_branching`：每步 4–6 个选项
  - `critical_choice`：内存分配关键决策
  - `trap_heavy`, `deceptive`：陷阱密集或“快速收益”误导路径

> ⚠️ 注：这些任务是为**区分规划能力**而专门构造的，不同于 API-Bank 等单步 API 选择基准。

---

### 🧪 实验设置与评估指标

#### 实验参数
- 使用 **Llama 3.2**（通过 Ollama）作为 LLM 后端
- 搜索预算（budget b）：5、10、20
- 探索常数 c = 1.0
- 最大步数：20
- 随机种子：5 个（42, 123, 456, 789, 1000）

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Success Rate (SR)** | 达成目标的任务占比 |
| **Optimality** | 最优路径长度 / 实际路径长度 |
| **LLM Calls / Task** | 每个任务平均 LLM 调用次数 |
| **Variance** | 不同随机种子下成功率的标准差 |

---

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **ReAct** | LLM 直接选择动作，无搜索机制 |
| **LATS** | 基于 MCTS 的树搜索，每节点调用 LLM 提议动作和估值 |
| **Greedy (Oracle)** | 使用 BFS 获取最优后继状态的选择器（理想上限） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### 表 1：合成任务结果（100 任务，3 种子）
| Method | Success Rate | Optimality | LLM Calls/Task | Variance |
|--------|--------------|------------|----------------|----------|
| Greedy (Oracle) | 100.0% | 1.00 | 0 | 0% |
| ReAct | 64.0% ±5.0 | 0.54 | 13 | 5.0% |
| LATS (b=10) | 92.0% ±1.0 | 0.99 | 37 | 1.0% |
| **GATS (b=10)** | **100.0%** | **1.00** | **0** | **0%** |

> ✅ GATS 在成功率、最优性和稳定性上全面超越所有基线。

---

#### 表 6：压力测试结果（12 类 × 10 任务，共 120 任务）
| Category | GATS (b=20) | LATS | ReAct | Δ vs LATS |
|---------|-------------|------|-------|-----------|
| coding task | 100.0% | 63.3% | 0.0% | +36.7% |
| web_navigation | 100.0% | 63.3% | 0.0% | +36.7% |
| deep_horizon | 100.0% | 63.3% | 0.0% | +36.7% |
| resource_puzzle | 100.0% | 86.7% | 16.7% | +13.3% |
| **Overall** | **100.0%** | **88.9%** | **23.9%** | **+11.1%** |

> ✅ GATS 在所有 12 类挑战中均达到 **100% 成功率**，尤其在长程、编码、导航类任务上大幅领先。

---

### 🔬 消融实验结果（Ablation Studies）

#### （1）搜索预算影响（Table 2）
| Budget | SR (%) | Nodes Explored |
|--------|--------|---------------|
| b=1 (greedy) | 0.0 | 5 |
| b=5 | 84.0 | 84 |
| b=10 | 100.0 | 167 |
| b=20 | 100.0 | 334 |

> ✅ 搜索预算需足够（≥10）才能覆盖复杂路径；超过后进入饱和。

---

#### （2）世界模型层消融（Table 3）
| Configuration | SR (%) | Description |
|-------------|--------|-------------|
| GATS (full) | 100.0 | L1 + L2 + L3 |
| GATS no_l1 | 100.0 | 仅 L2 + L3（L2 初始化自执行日志） |
| GATS no_l3 | 100.0 | 仅 L1 + L2（无 LLM 回退） |

> ✅ 各层提供冗余支持，系统具备**优雅降级能力**。

---

#### （3）世界模型层使用统计（Table 4）
| 阶段 | L1 Hit Rate | L2 Hit Rate | L3 Calls |
|------|-------------|-------------|----------|
| Bootstrapping（一次性） | 0% | 0% | ~50 |
| Planning（每任务） | 100% | 0% | 0 |

> ✅ 在合成任务中，**L1 符号匹配承担全部预测工作**，完全规避 LLM 调用。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **系统化搜索优于 LLM 引导探索**  
   UCB1 + BFS 启发式搜索比 LLM 采样更可靠，尤其在存在死胡同、长程依赖的任务中。

2. **分层世界模型极大提升效率与稳定性**  
   L1/L2 层几乎完全替代 LLM 推理，使规划过程**零延迟、零方差、可复现**。

3. **图结构实现跨路径与跨步共享**  
   状态合并机制（transposition）和图持久化显著降低重复计算，提升搜索效率。

4. **GATS 可逼近 Oracle 性能**  
   在多种复杂场景下达到 100% 成功率，表现与理想贪婪搜索一致。

---

### ⚠️ 方法的局限性
1. **依赖动作语义的形式化**  
   GATS 在 L1 层依赖已知的 action specifications（如 preconditions/effects）。若领域中缺乏此类结构化定义，则需更多依赖 L3（LLM），削弱其效率优势。

2. **BFS 价值估计的可扩展性限制**  
   当前使用 BFS 计算最短距离作为启发值，时间复杂度为 $O(|A|^d)$，在动作空间过大（>20）或深度过深时可能不适用。

3. **评估仍以合成任务为主**  
   当前实验集中在人工构造任务，尚未在真实 API 平台（如 ToolBench）上验证。

---

### 🔮 未来工作方向
1. **在真实世界 API 基准上评估**（如 ToolBench）  
   测试 GATS 在部分可观测、非结构化环境中的表现。

2. **从执行日志中自动学习世界模型**  
   利用生产系统日志训练 L2 层，提升实际部署中的覆盖率。

3. **结合 RAG 技术检索 action specifications**  
   使用 Retrieval-Augmented Generation 动态获取动作描述，增强 L1/L2 覆盖。

4. **扩展至大规模动作空间**  
   引入 learned value network 或 LLM-based heuristic 替代 BFS，提升可扩展性。

---

## ✅ 总结
**GATS 是首个实现“零 LLM 调用 + 100% 成功率”的 agent planning 框架**。它通过**分层世界模型 + 图增强 UCB1 搜索**，将 LLM 从“在线推理引擎”转变为“离线建模工具”，在效率、稳定性和性能上全面超越 LATS 和 ReAct，为构建高效、可复现的智能体规划系统提供了新范式。

</details>

---

### 9. [Letter Lemmatization: One-to-one and Banded RNNs for Reversing Character-Set Simplification and Abbreviation in Medieval Text](https://arxiv.org/abs/2607.09291)

**Authors**: Anguelos Nicolaou, Maria Pia Tiseo, Tamas Kovacs, Nicolas Renet, Georg Vogeler  
**Category**: cs.CL  
**Published**: 2026-07-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.09291v1  

#### Abstract
Medieval document transcribers have very different practices; on top of that, heterogeneous digitization policies have resulted in corpora where the character-set must be viewed as fluid. In this paper we address the problem of changing between character-sets in a flexible manner. We focus on one-to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Letter Lemmatization: One-to-one and Banded RNNs for Reversing Character-Set Simplification and Abbreviation in Medieval Text*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
中世纪手稿数字化面临字符集（charset）不统一、缩写形式多样以及不同转录规范之间难以对齐的问题。传统做法在选择字符集时需在信息保留与模型训练效率之间权衡，而现有工具难以灵活处理字符集之间的转换。

本文聚焦以下两个核心挑战：
- **字符集简化（Charset Simplification）**：将复杂字符集（如 MUFI）映射到更小、更通用的字符集（如 ASCII），但会丢失信息。
- **缩写扩展（Abbreviation Expansion）**：中世纪文本大量使用缩写，单个符号可能对应多个字母，属于非一对一映射任务。

### 提出的新方法与新思路

#### ✅ **Letter Lemmatization**
- 一种自动推导任意两个字符集之间 **Charset Simplification Mapping (CSM)** 的启发式方法。
- 定义了一套字符相似度度量标准，综合考虑 Unicode 类别、名称重叠、ASCII 转写一致性、语音类别等特征。
- 输出为一个 **one-to-one 映射函数**，可高效实现字符归一化。

#### ✅ **One-to-one RNNs（用于反向去映射 de-mapping）**
- 利用自监督学习训练基于 LSTM 的字符级 RNN 模型，以“逆转”CSM 所造成的信息损失。
- 输入与输出严格对齐，使用 per-position cross-entropy 进行训练，易于实现且数据效率高。
- 可作为 HTR post-correction 工具，专门修复 substitution 错误。

#### ✅ **Banded RNNs（用于处理插入/删除操作）**
- 扩展 one-to-one RNN 架构，支持处理涉及 insertion 和 deletion 的任务（如缩写扩展）。
- 利用动态规划对齐源与目标序列，并通过复制输入 N 次的方式构造“带状”上下文窗口（band width = N），配合 CTC loss 训练。
- 继承原有架构优势，同时突破 one-to-one 限制。

#### ✅ 开源工具包：`pylelemmatize`
- 提供高效的字符映射、语言模型提取、CER 估计等功能。
- 支持快速构建和应用 CSM，便于集成到数字人文（DH）流程中。

### 相比现有方法的优势
| 方面 | 本方法优势 |
|------|-----------|
| **灵活性** | 不依赖固定字符集，可在 pipeline 中实时切换映射规则，无需修改原始存储数据。 |
| **轻量化 & 高效性** | `pylelemmatize` 基于 NumPy 实现，在常见操作上比 Python dict 快 2.8–6×。 |
| **低资源适应性强** | One-to-one RNN 仅需 **20 条文本行**即可恢复一半的 CER 损失，适合小规模标注数据场景。 |
| **模块化设计** | CSM 层不影响 standoff annotation 或 XML 结构，兼容复杂标注体系。 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### 主要 HTR 数据集（用于 de-mapping 与 post-correction）
- **Konigsfelden corpus**：3,102 行文本，约 44.8 万字符，中欧中世纪宪章。
- **Monasterium corpus**：从历史档案中新整理，多语言混合（德语 ≈60%，拉丁语 ≈39%）。
- **Nuremberg Letterbooks (basic/norm.)**：早期 15 世纪手稿，含多种转录版本。

> 所有 HTR 数据均为 `.tsv` 文件格式，每行包含 HTR 输出与人工校对的 ground truth。

#### 缩写扩展专用平行语料库
| 名称 | 描述 |
|------|------|
| **FTN (Fontenay)** | 包含缩写与手动展开版本的中世纪手稿，共 25 万字符。 |
| **SMG (Santa Maria della Grotta)** | 新建意大利南部修道院宪章语料库，33 封文件（1254–1380 年），双列对齐。 |

### 实验设置与评估指标

#### 评估指标
- **Character Error Rate (CER)**：标准编辑距离指标，用于衡量映射质量、纠错效果。
- **Recovery Rate (%)**：表示 de-mapping 模型恢复原始信息的能力，计算方式为 `(原始 CER - 反向后 CER) / 原始 CER × 100%`。

#### 实验设计
| 任务 | 设置说明 |
|------|----------|
| **De-mapping 实验** | 应用自动 CSM（如 MUFI → ASCII）后，训练 one-to-one RNN 尝试还原原字符；验证集占 20%，测试前过滤掉 CER > 20% 或长度 < 50 的异常行。 |
| **HTR Post-correction** | 构造 substitution-only 对齐样本进行训练，最终在完整 HTR 输出上测试实际纠错能力。 |
| **Banded RNN for Abbreviation Expansion** | 设置最大扩展长度 $N=5$，丢弃超出该范围的样本；使用 CTC loss，bidirectional LSTM 架构保持一致。 |

#### 基线方法对比
- **No-op Baseline**：直接使用缩写文本作为输入，其与展开文本间的 CER 即为基准误差。
- **Generic Python dict/defaultdict**：用于 benchmark `pylelemmatize` 的运行速度。
- **Chokomufin**：另一款字符控制工具，用于比较 alphabet extraction 性能。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 🔹 表 2：De-mapping Recovery Results
| Source → Destination | Mapping CER (%) | De-mapped CER (%) | Recovery (%) |
|------------------------|------------------|--------------------|---------------|
| MUFI(BMP) → ASCII(lower) | 4.04             | 0.96               | 76.12         |
| MUFI(BMP) → ASCII(mini)  | 4.85             | 1.0                | 79.38         |
| MUFI → MUFI (U→V override) | 5.08           | 0.01               | 99.99         |

> ✅ 表明即使大幅压缩字符集（777 → 39），也可通过 one-to-one RNN 几乎完全恢复信息。

#### 🔹 表 4：HTR Post-correction Performance
| Corpus        | Val. CER (HTR) | Val. CER (Corr.) | Test CER (HTR) | Test CER (Corr.) |
|---------------|----------------|-------------------|----------------|-------------------|
| Konigsfelden  | 5.11           | 3.59              | 5.78           | 4.58              |
| Nurn. basic   | 17.26          | 6.20              | 18.57          | 8.18              |
| All (combined)| 10.16          | 5.52              | 8.83           | 6.60              |

> ✅ 在大多数语料上显著降低 CER，尤其在噪声较高的 `Nurn. basic` 上提升明显（↓~10% CER）。  
> ❗ Monasterium 效果不佳，推测因多语言、转录风格不一导致上下文建模困难。

#### 🔹 表 6：Abbreviation Expansion with Banded RNNs
| Corpus     | No-op CER (%) | Banded RNN CER (%) | Improvement |
|------------|----------------|-----------------------|-------------|
| FTN        | 21.6           | 3.9                   | ~5.5×       |
| SMG        | 31.1           | 4.2                   | ~7.4×       |
| SMG+FTN    | 26.5           | 4.3                   | ~6.2×       |

> ✅ 缩写扩展任务中，Banded RNN 显著优于 no-op 基线，跨语料泛化能力强。  
> ⚠️ 反向任务（expansion → abbreviation）表现较差（CER 达 10–16%），表明该过程不可逆。

#### 🔹 表 1：`pylelemmatize` 性能 benchmark
| 方法 | 平均耗时（ms） | 加速比 |
|------|----------------|--------|
| Python dict | ~50 ms | 1× |
| `pylelemmatize` (Fast Lemmatizer) | **6.5–13.1 ms** | **2.8–6×** |

> ✅ 在大规模映射下仍保持高性能，适用于生产环境。

#### 🔹 消融实验（Ablation Study on De-mapping）
- **训练数据量影响**（图 3）：
  - 仅 **20 条文本行**即可使 de-mapping RNN 将 mapping error **减少一半**。
  - 性能在约 **4,000 条文本行后趋于饱和**，表明极高的数据效率。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **字符集简化并非不可逆**：尽管 CSM 会造成信息损失，但可通过 one-to-one RNN 高效恢复大部分信息，使得“先简化再还原”的策略可行。
2. ✅ **One-to-one RNN 是轻量高效的 post-correction 工具**：特别擅长纠正 substitution 类错误，在少量数据下即可取得显著成效。
3. ✅ **Banded RNN 成功应用于缩写扩展**：在真实中世纪文本上实现高质量的 abbreviation expansion，CER 下降达 5–7 倍。
4. ✅ **Letter Lemmatization 提供自动化 CSM 构建机制**：无需人工定义完整映射表，即可生成语义合理的字符对应关系。
5. ✅ **模块化映射层优于硬编码字符集选择**：建议将 CSM 作为软件层嵌入 pipeline，而非持久化更改原始数据。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **One-to-one 约束无法处理 ligature 或多字符映射** | 如“ﬀ”无法被优雅地拆分为“ff”，必须借助外部机制预处理。 |
| **De-mapping 依赖上下文充分性** | 若某字符组合未在训练集中出现，则难以准确还原。 |
| **Banded RNN 数据效率较低** | 相比 one-to-one RNN，需要更多数据和更长训练时间。 |
| **Band width $N$ 是超参数瓶颈** | 设定不当会导致数据浪费或输入膨胀。 |
| **启发式相似度未必符合领域专家意图** | 用户需手动覆盖部分映射规则。 |

### 未来工作方向
1. **扩展至 full many-to-many mappings**：支持更复杂的 ligature 分解与复合缩写建模。
2. **发展 Letter Tokenization**：超越字符粒度，进入 subword 或 token 层面的归一化。
3. **探索端到端 HTR 优化路径**：研究“减小 HTR 字符集 + 后接 de-mapping RNN”是否优于直接大字符集训练。
4. **与 Transformer/Large Language Models 对比**：
   - 当前方法面向 **low-resource、lightweight 场景**；
   - 未来需系统比较与 fine-tuned LLM 在 post-correction 和 abbreviation expansion 上的表现差异。
5. **在更多样化的语料上验证普适性**：当前实验集中在欧洲中世纪拉丁语文献，需拓展至其他语言与时期。

---

> 📌 **总体评价**：本文提出了一套实用、高效、可扩展的框架，解决了数字人文中长期存在的字符集异构与文本规范化难题。其核心思想——“**用轻量模型补偿简化带来的信息损失**”——具有广泛适用性，不仅限于中世纪文本，也可推广至任何存在编码多样性或噪声文本的场景。

</details>

---

### 10. [Bidirectional Resource Scheduling for Disaggregated and Asynchronous RL Post-Training](https://arxiv.org/abs/2607.09207)

**Authors**: Tan Zhiqiang, Wang Maoxin, Wang Sijie, Yin Yiming, Wang Qiang, Chu Xiaowen, Shi Shaohuai  
**Category**: cs.DC  
**Published**: 2026-07-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.09207v1  

#### Abstract
It is well established that the reasoning capabilities of large language models (LLMs) can be improved by applying reinforcement learning (RL) in a post-training stage. In a standard RL iteration, the current model (the policy) generates experience through rollouts, and the resulting data is then us...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Bidirectional Resource Scheduling for Disaggregated and Asynchronous RL Post-Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大型语言模型（LLM）的强化学习（RL）后训练中，**disaggregated（分离式）架构**虽然通过将 rollout 和 training 部署在独立的资源池上实现了异步执行以提升吞吐量，但仍存在显著的**资源空闲（resource idleness）问题**。这些空闲主要来源于两类“气泡”（bubbles）：

- **结构性气泡（Structural bubbles）**：由静态资源分配不均导致的持续速率不匹配。
- **残余气泡（Residual bubbles）**：由响应长度动态变化、模型并行策略限制以及 staleness 约束等运行时因素引起。

现有系统如 veRL、AReaL、ROLL 虽然通过异步调度、静态规划或单向弹性扩展缓解部分问题，但无法有效应对双向资源闲置，尤其当瓶颈阶段随 workload 动态切换时。

### 提出的新方法：BiDiRL
本文提出 **BiDiRL** —— 一种用于分离式、异步 RL 后训练的**双向资源调度系统**，其核心思想是允许 rollout 和 training 在运行时**双向借用对方的空闲资源**，从而最大化资源利用率。

#### 创新机制
1. **Hot-Switch Runtime（热切换运行时）**
   - 实现 rollout 和 training 角色之间的快速切换，开销极低。
   - 支持 `Trainer-on-RollPoll` 和 `Rollouter-on-TrainPoll` 两种模式，无需重启进程或重分片（resharding）。

2. **Scheduling-Aware Static Planner（调度感知的静态规划器）**
   - 基于时间-性能建模，选择一个**粗粒度平衡**的资源划分方案。
   - 确保该划分满足“hot-switch 兼容性”，即两个资源池均可被对方角色整除地使用。

3. **Bidirectional Scheduler（双向调度器）**
   - 运行时动态判断是否应借用资源，并基于**收益超过开销（benefit-over-overhead）** 的原则进行准入控制。
   - 决定如何在主辅 worker 之间拆分任务负载，实现细粒度资源复用。

### 相比现有方法的优势
| 特性 | BiDiRL | veRL | AReaL | ROLL |
|------|--------|------|-------|------|
| 分离式架构 | ✅ | ❌（共置为主） | ✅ | ✅ |
| 异步训练 | ✅ | ✅ | ✅ | ✅ |
| 双向资源借用 | ✅ | ❌ | ❌ | ❌ |
| 运行时动态调度 | ✅ | ❌ | ❌ | ❌ |
| 支持 fractional staleness | ✅ | ✅ | ❌ | ❌ |

> BiDiRL 是首个支持**固定预算内双向弹性调度**的系统，突破了传统“rollout-only 扩展”的局限。

---

## 2. 核心实验方法和设置

### 数据集
- **文本任务**：`GSM8K`（数学推理）
- **多模态任务**：`Geo3K`（几何问题求解）
- 默认使用 Geo3K，最大响应长度为 2K tokens。

### 模型
- 使用 Qwen3 系列模型：
  - `Qwen3VL-2B`, `Qwen3VL-4B`（视觉语言）
  - `Qwen3-8B`（纯文本）

### 硬件平台
- **A6000 Testbed**：4 节点 × 8×RTX A6000（共 32 GPU）
- **H100 Testbed**：4 节点 × 8×H100（共 32 GPU）

### 评估指标
- **RL Training Throughput**：每秒处理的 prompt 和 response token 数量。
- **Speedup**：相对于最强 baseline 的加速比。
- **Convergence Behavior**：前 60 步的奖励曲线对比，验证逻辑一致性。

### 基线方法对比
- **veRL**：支持共置与分离式，有静态规划能力。
- **AReaL**：大规模异步 RL 系统，强调流水线重叠。
- **ROLL**：支持 rollout 侧弹性调度，但不支持 training 借用 rollout 资源。

> 所有系统配置相同超参数，OOM 或不支持配置被排除。

---

## 3. 主要实验结果和性能指标

### 整体性能提升
| 设置 | 平台 | 加速比（Speedup） |
|------|------|------------------|
| 默认设置 | A6000 | **1.27×–1.68×** |
| 最大规模（32 GPU） | A6000 | **1.94×** |
| 默认设置 | H100 | **1.23×–1.53×** |
| 资源分区扫描 | H100 | **1.11×–1.41×** |

> BiDiRL 在所有测试场景下均优于现有系统，最高达 **1.94× 吞吐提升**。

### 不同 workload 下的表现
- **长响应任务（如 4K tokens）**：rollout 成为瓶颈 → `Rollouter-on-TrainPoll` 发挥作用 → 提升 **1.27×**
- **短响应任务**：training 成为瓶颈 → `Trainer-on-RollPoll` 显著减少等待时间
- **严格 staleness（s=0）**：需等待最新样本 → 双向调度仍能回收空窗期 → 提升 **1.45×–1.68×**

> 表明 BiDiRL 对 workload 动态变化具有强鲁棒性。

### 消融实验（Ablation Study）
| 变体 | 相对性能 |
|------|----------|
| **No borrow**（禁用借用） | BiDiRL 快 **1.12×–1.71×** |
| **Opportunistic borrow**（无模型指导借用） | BiDiRL 快 **1.02×–1.31×** |
| **w/o T-on-R**（禁用 training 借用 rollout） | 性能下降最多达 **1.68×** |
| **w/o R-on-T**（禁用 rollout 借用 training） | 性能下降最多达 **1.24×** |

> 结果证明：
> - **双向性至关重要**：任一方向缺失都会导致次优。
> - **模型指导必要**：盲目借用可能因切换开销而得不偿失。

### 切换开销测量（Hot-Switch Cost）
| 路径 | Cin（进入成本） | Cout（退出成本） | Cgrad（梯度同步） |
|------|----------------|-----------------|------------------|
| Trainer-on-RollPoll | 3.58s / 6.16s (2B/4B) | 3.40s / 5.62s | 0.66s / 1.21s |
| Rollouter-on-TrainPoll | 5.54s / 7.70s | 4.20s / 5.59s | — |

> 尽管切换有成本，但远小于典型训练窗口时间，且 BiDiRL 的准入机制确保只在净收益为正时才触发。

---

## 4. 关键结论和发现

### 主要发现
1. **资源空闲是双向的**：在分离式 RL 中，无论是 rollout 还是 training 都可能成为临时瓶颈，因此必须支持**双向资源借用**。
2. **静态规划 + 动态调度缺一不可**：
   - 静态规划减少结构性气泡；
   - 动态调度回收残余气泡。
3. **BiDiRL 不影响收敛行为**：与 veRL 在相同 staleness 设置下的奖励曲线几乎完全一致（差异 < 0.02），说明其仅为系统优化，不改变 RL 逻辑流。
4. **硬件越强，增益越稳定**：在 H100 上虽绝对加速比略低（因原系统效率更高），但相对优势依然显著。

### 方法的局限性
- **依赖 hot-switch 兼容布局**：要求 rollout 和 training 的 model parallelism 配置兼容，否则无法无缝切换。
- **chunk 粒度权衡**：trainer chunk 太小影响计算效率，太大则尾部不平衡；当前采用 lazy pull + tail split 折中解决。
- **未考虑外部资源**：仅限于已承诺的 GPU 池内部调度，未整合 preemptible 或 cloud burst 资源。

### 未来工作方向
- **更智能的任务放置**：利用 prompt 特征预测生成长度，优先将短任务调度到 auxiliary rollouter。
- **自适应 chunk sizing**：根据运行时负载动态调整 chunk 大小，进一步优化尾延迟。
- **跨作业资源共享**：将 BiDiRL 扩展至多任务 RL 场景，实现集群级双向资源调度。
- **集成到端到端 RL 框架**：作为通用调度层嵌入如 OpenRLHF、NeMo-Aligner 等开源框架。

---

> **总结**：BiDiRL 通过“**静态规划 + 热切换运行时 + 双向动态调度**”三位一体设计，首次实现了分离式 RL 中的**双向资源复用**，在不影响收敛的前提下将训练吞吐提升高达 **1.94×**，为高效 LLM RL 系统提供了新的架构范式。

</details>

---

### 11. [TSRouter: Dynamic Modality-Model Selection for Time Series Reasoning](https://arxiv.org/abs/2607.08940)

**Authors**: Fangxu Yu, Tao Feng, Dehai Min, Lu Cheng, Ge Liu, Tianyi Zhou  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.08940v1  

#### Abstract
Time series reasoning is essential for real-world problem-solving. While both Large Language Models (LLMs) and Vision-Language Models (VLMs) can reason about time-series data, their capabilities are complementary: LLMs process time series as text sequences and thus preserve exact numerical understan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TSROUTER: Dynamic Modality-Model Selection for Time Series Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

时间序列推理（Time Series Reasoning）在教育、科学发现等高风险领域至关重要。当前主流方法依赖 **Large Language Models (LLMs)** 和 **Vision-Language Models (VLMs)**，但二者存在显著互补性与局限性：

- **LLMs** 将时间序列视为文本序列处理，保留精确数值理解能力，但在识别全局模式（如趋势、周期）方面表现较差，且受上下文长度限制。
- **VLMs** 通过将时间序列可视化为图表进行推理，能高效捕捉宏观结构，但因图像分辨率限制而丢失细粒度数值细节。

此外，不同模型在任务专长和推理成本上差异巨大。传统方法通常固定输入模态（text 或 vision），仅选择最优模型，忽略了**联合优化模态与模型**的重要性。

因此，如何为每个查询动态选择最合适的 **modality-model 组合**，以平衡性能与成本，成为一个关键挑战。

---

### 🚀 提出的新方法与创新思路

论文提出 **TSROUTER** —— 一种基于图的动态路由框架，用于实现 **Dynamic Modality-Model Selection**。

#### 核心创新点：

1. **异构图建模（Heterogeneous Graph Modeling）**
   - 构建包含四种节点类型的异构图：**Task**, **Query**, **Modality**, **Model**。
   - 显式建模任务、查询、模态、模型之间的复杂交互关系，如：
     - `query-modality` 边：反映不同模态对特定查询的适配性；
     - `modality-model` 边：表示兼容性约束；
     - `query-query` 边：基于语义相似性共享信息；
     - `task-query` 边：传递任务级上下文。

2. **统一的候选评分机制（Candidate Scoring Framework）**
   - 将路由问题形式化为一个 **candidate scoring problem**。
   - 定义有效性得分函数：
     $$
     e(q, c) = \alpha \cdot I(q,c) - (1-\alpha) \cdot \text{cost}(q,c)
     $$
     其中 $\alpha$ 控制用户对性能与成本的偏好。
   - 使用 **Heterogeneous Graph Transformer (HGT)** 学习节点表示，并通过 MLP 对每组 `(modality, model)` 候选打分。

3. **零样本泛化能力（Zero-Shot Generalization）**
   - 利用 LLM 自动生成节点描述（如任务要求、模型参数规模、成本等），并通过预训练 embedding 模型初始化特征。
   - 新增任务或模型时无需重新训练，只需插入对应节点并连接边即可完成集成。

4. **跨模态协同决策**
   - 同时考虑 text、vision 及混合模态，突破以往仅在单一模态下选模型的局限。

---

### 🔍 相比现有方法的优势

| 特性 | TSROUTER | 现有方法（如 Hybrid LLM, RouterDC） |
|------|----------|-------------------------------|
| 是否支持模态选择 | ✅ 是 | ❌ 否（固定模态） |
| 是否建模多类型交互 | ✅ 异构图全连接 | ❌ 仅 query-model 或简单 embedding 匹配 |
| 是否支持新任务/模型零样本接入 | ✅ 支持 | ❌ 需重新训练或微调 |
| 推理效率 | ⚡ 快速（3.21ms/query） | 🐢 慢（最高达 910ms） |
| 成本感知优化 | ✅ 显式建模 | ⚠️ 多数忽略或弱建模 |

> ✅ 总结优势：**更全面、更灵活、更高性能、更强泛化、更低开销**

---

## 2. 核心实验方法和设置

### 📚 数据集

主实验使用 **TSRBench**（Yu et al., 2026a），涵盖四大类共 15 个子任务：

| 类别 | 示例任务 | 数量 |
|------|--------|-----|
| **Perception** | 趋势识别、异常检测 | 4 subtasks (700 cases) |
| **Reasoning** | 因果推断、逻辑判断 | 7 subtasks (1710 cases) |
| **Prediction** | 数值预测、事件预测 | 2 subtasks (1080 cases) |
| **Decision-Making** | 行动建议、策略生成 | 2 subtasks (635 cases) |

#### 泛化测试任务（Unseen Tasks）：
- **Imputation**（来自 TSQA）：填补缺失值
- **Correlation Prediction**（来自 MTBench）：预测新闻情感与股价的相关性

---

### ⚙️ 实验设置

- **LLMs**: Qwen3-8B, Qwen3-32B, LLaMA-3.3-70B-Turbo, Qwen3.5-397B-A17B
- **VLMs**: Qwen3-VL-8B, Qwen3-VL-32B, GLM-4.5V, Kimi-K2.5
- **Modalities**: Text-only, Vision-only, Text+Vision
- **训练/验证/测试划分**：按任务 7:1:2 分割
- **成本计算**：基于 API token 收费标准（见 Table 9）
- **图构建**：训练阶段基于历史交互数据构建图；测试阶段将新 query 插入图中进行推理

---

### 📊 评估指标

| 指标类型 | 指标名称 | 说明 |
|--------|-------|------|
| **性能** | Accuracy | 主要指标（除 imputation 外） |
| | MSE / MAE | 时间序列插补任务使用 |
| **成本** | Total Cost (USD) | 在测试集上的总 API 开支 |
| **综合表现** | Pareto Frontier | 准确率 vs 成本权衡曲线 |
| **效率** | Inference Latency (ms/query), Memory Usage | 实际部署考量 |

---

### 🆚 基线方法对比

分为两类：

#### （1）Rule-based Baselines
- **Largest LLM/VLM**：始终选择最大模型

#### （2）Learning-based Baselines
| 方法 | 方法简介 |
|------|---------|
| **EloRouter** | 根据 Elo 评分选择最强模型 |
| **MFRouter** | 矩阵分解建模 query-model 兼容性 |
| **KNNRouter** | 找最近邻训练样本的最佳候选 |
| **Hybrid LLM** | 冻结 embedding + MLP 分类器 |
| **RouterDC** | 基于双对比学习的 embedding 匹配 |
| **CausalLM** | 微调 decoder 生成最佳候选名 |
| **GraphRouter** | 图结构建模 query-model 关系 |

> 所有 baseline 均被适配至 joint modality-model routing 场景。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table 3）

| 方法 | Overall Accuracy (%) | Total Cost (USD) |
|------|------------------------|------------------|
| Largest LLM | 35.59 | 0.74 |
| Largest VLM | 39.10 | 0.59 |
| Hybrid LLM | 44.07 | 0.76 |
| RouterDC | 41.16 | 0.83 |
| **TSROUTER (Ours)** | **51.33** | **0.73** |

✅ **相对提升 16% ~ 46%**，在所有任务类别上均取得 SOTA。

#### 分项准确率（Top-1）：
- **Perception**: 67.63%
- **Reasoning**: 52.81%
- **Prediction**: 43.11%
- **Decision-Making**: 42.45%

---

### 🔁 与基线方法的对比结果

- **超越所有 rule-based 和 learning-based 方法**，尤其在复杂推理任务中优势明显。
- 即使在相同成本预算下（如 ≈0.76 USD），TSROUTER 仍比 Hybrid LLM 高出近 **7个百分点**。
- 在 **zero-shot 新任务** 上：
  - Imputation: **31.38% Acc**（第二名为 24.93%）
  - Correlation Prediction: **29.20% Acc → TSROUTER 达到 31.38%**
- 在引入两个未见过的新模型后（Qwen3.5-397B-A17B 和 Kimi-K2.5），TSROUTER 性能进一步提升至 **53.5%**，而部分 baseline（如 KNNRouter）反而下降。

---

### 🔍 消融实验结果（Ablation Study, Table 5）

| 变体 | Overall Accuracy (%) | 下降幅度 |
|------|------------------------|--------|
| **TSROUTER (Full)** | **51.33** | — |
| -Hetero Graph | 46.25 | ↓5.08 |
| -Query-Query Edges | 48.55 | ↓2.78 |
| -MLP Scoring Head | 46.52 | ↓4.81 |
| -Modality-Model Edges | 47.33 | ↓4.00 |

📌 结论：
- **异构图结构最关键**，移除后性能大幅下降；
- `query-query` 边有助于相似问题间知识迁移；
- `MLP scoring head` 比简单点积更能捕捉非线性交互；
- `modality-model` 结构边对建模组合能力至关重要。

---

### 📉 超参数敏感性分析（Appendix A）

- 最优 GNN 层数：**2层**
- 最优嵌入维度：**64**
- 最优邻居数 k：**60**
- 数据效率：仅用 **10% 训练数据** 即超过 full-data baseline

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **没有“万能”模态或模型**：
   - Perception 任务更适合 **Vision** 模态（视觉易识别形状）；
   - Prediction 任务更依赖 **Text** 模态（需精确数值）；
   - Decision-Making 需要两者结合。

2. **TSROUTER 能智能分配模态**（Figure 4）：
   - Perception: 83.5% → Vision
   - Prediction: 76.9% → Text
   - Decision-Making: 更均衡分布（Text 27.4%, Vision 44.3%, Both 28.3%）

3. **大模型不总是更好**：
   - 案例显示，某些 query 上小模型（如 Qwen3-VL-8B）正确，而大模型错误（因文本干扰视觉判断）；
   - 支持 per-query 动态路由必要性。

4. **成本可控性强**：
   - 通过调节 $\alpha$ 可灵活控制精度-成本权衡；
   - 在低预算下仍优于其他方法，形成 **Pareto 最优前沿**（Figure 5）。

5. **高效实用**：
   - 推理延迟仅 **3.21ms/query**，内存占用 **576 MiB**，远低于依赖 LLM 的 baseline（如 CausalLM 占 33GB）；
   - 支持批量推理，适合实际部署。

---

### ⚠️ 方法的局限性

1. **依赖高质量节点描述**：
   - 当前使用 LLM 生成描述，若描述质量差可能影响效果（尽管消融实验证明鲁棒性较强）。

2. **图规模扩展性待验证**：
   - 当任务/模型数量极大时，图结构可能变得复杂，需进一步优化消息传递机制。

3. **静态图假设**：
   - 当前图结构在训练后固定，未能在线更新模型能力反馈（如失败案例学习）。

---

### 🔮 未来工作方向

1. **动态图更新机制**：
   - 引入在线反馈闭环，持续优化路由策略。

2. **多跳推理与聚合机制**：
   - 支持多个 candidate 并行执行并融合结果（类似 ensemble 或 voting）。

3. **扩展至更多模态**：
   - 如音频、表格、传感器流等多元时间序列模态。

4. **轻量化部署版本**：
   - 设计蒸馏版 TSROUTER，适用于边缘设备。

5. **开放生态集成**：
   - 构建开源平台，支持社区提交新模型/任务自动接入。

---

## ✅ 总结

**TSROUTER** 是首个同时解决 **time series modality selection** 与 **model selection** 的统一框架。它通过构建 **heterogeneous graph** 显式建模任务、查询、模态、模型间的复杂交互，实现了高性能、低成本、强泛化、高效率的动态路由，在多个基准和真实场景中显著超越现有方法。

> 🔗 代码已开源：[https://github.com/tianyi-lab/TSRouter](https://github.com/tianyi-lab/TSRouter)

</details>

---

### 12. [Data-Efficient Deep Learning: Empirical Guidelines for Training Set Size Estimation in Inertial Sensor Classification](https://arxiv.org/abs/2607.09402)

**Authors**: Ofir Kruzel, Itzik Klien  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.09402v1  

#### Abstract
Deep learning models dependency on large-scale inertial datasets presents a significant bottleneck in inertial sensor-based classification tasks, such as human activity recognition and smartphone location recognition. In these domains, data collection requires massive recording campaigns that are co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Data-Efficient Deep Learning: Empirical Guidelines for Training Set Size Estimation in Inertial Sensor Classification

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
该论文针对**惯性传感器分类任务中训练数据收集成本高、缺乏数据驱动的样本量规划指南**的问题。在 Human Activity Recognition (HAR) 和 Smartphone Location Recognition (SLR) 等领域，大规模标注数据采集耗时、昂贵且难以扩展。目前尚无系统性方法来判断“需要多少数据才能达到目标精度”，导致资源浪费或模型性能不足。

传统启发式规则（如“Rule of 10”）不适用于现代 deep learning 模型，而 Power Law 学习曲线假设在小规模、有噪声的数据集中常失效。

---

### ✅ 提出的新方法与新思路  

1. **提出统一的经验框架**：  
   对六种不同 HAR 和 SLR 数据集进行系统性学习曲线分析，揭示了分类准确率随训练样本增长呈现出**一致的对数增长模式**：
   $$
   \text{Accuracy}(n) = a \log(n) + b
   $$
   其中 $n$ 是训练样本数量，$a, b$ 为任务相关常数。

2. **引入“稳定性点”（Stability Point）度量**：  
   定义为学习曲线稳定在其渐近最大值的预定义 **Mean Absolute Percentage Deviation (MAPD)** 范围内所需的最小样本量。例如：
   - 多类任务：MAPD ≤ 5% 或 2%
   - 二元任务：更严格阈值（≤2% 或 ≤1%）

3. **支持从小规模试点研究外推总数据需求**：  
   只需前几个增量样本点即可拟合对数曲线，并预测最终性能上限，从而优化数据采集策略。

---

### ✅ 相比现有方法的优势  

| 方面 | 本文方法 | 传统方法 |
|------|----------|---------|
| **理论基础** | 基于实证观察的对数模型，适合 bounded accuracy 任务 | 多采用 Power Law，适用于 large-scale vision/language |
| **实用性** | 可从少量 pilot runs 推断总数据需求，节省成本 | 缺乏量化指导，依赖经验或试错 |
| **通用性** | 在多种 dataset、task granularity、architecture 下均有效 | 启发式规则（如 Rule of 10）已被证明不可靠 |
| **工程价值** | 明确识别“收益递减点”，避免冗余采集 | 容易过度采集或欠采样 |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集  
共使用 **6 个公开 HAR 和 SLR 数据集**，总计 **142 名受试者，102.75 小时数据**：

| 数据集 | 类型 | 活动数 | 总时长 (小时) | 采样率 (Hz) |
|--------|------|--------|----------------|-------------|
| **PAMAP2** | HAR | 18 | 41.7 | 100 |
| **MotionSense** | HAR (手机) | 6 | 7.25 | 50 |
| **MobilePos** | SLR | 4 | 2.5 | 200 |
| **REALDISP** | HAR (高复杂度) | 33 | 17 | 100 |
| **UCI-HAR** | HAR (标准基准) | 6 | 1.8 | 50 |
| **WISDM** | HAR (大规模) | 18 | 32.5 | 20 |

> 所有数据被划分为固定长度滑窗（1秒），重叠50%，生成 labeled window 序列用于训练。

---

### ✅ 实验设置  

#### 🧩 统一评估流程：
1. **数据准备**：滑窗分割 → 固定测试/验证集划分
2. **双场景设计**：
   - **Binary classification**：动态 vs 静态活动等低熵任务
   - **Multi-class classification**：完整活动类别识别
3. **增量训练集构建**：
   - Multi-class：10 个等级（35k → 350k 样本）
   - Binary：10 个等级（10k → 100k 样本）
4. **每组配置重复多次随机子采样**，取平均 test accuracy

#### 🧠 模型架构（Baseline）  
采用 **CNN-BiLSTM** 架构：
- 输入：6维 IMU 信号（加速度计 + 陀螺仪）
- 前端：1D Conv + ReLU + Max Pooling
- 中间：BiLSTM 编码时间上下文
- 输出：Softmax 分类头 + Dropout

#### 🔍 评估指标  
- **Classification Accuracy** on fixed test set
- **Learning Curve Fitting**：非线性最小二乘拟合 $\text{Accuracy} = a\log(n)+b$
- **MAPD**（Mean Absolute Percentage Deviation）衡量候选曲线与参考曲线的一致性：
  $$
  \text{MAPD}(y_N, y^*) = \frac{1}{10}\sum_{i=1}^{10} \left| \frac{y_N(n_i) - y^*(n_i)}{y^*(n_i)} \right| \times 100\%
  $$
- **Stability Point $N^*$**：最小 $N$ 使得 MAPD ≤ 阈值

#### ⚖️ 基线对比  
未直接比较其他算法性能，而是挑战以下主流做法：
- “Rule of 10” 启发式
- Power Law / Scaling Laws（常见于 vision & language）
- 无指导的大规模数据采集

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据与发现  

#### 🔹 所有数据集均符合对数增长规律  
无论任务复杂度如何，accuracy 增长都高度吻合 $\text{Accuracy} = a\log(n) + b$ 形式（R² > 0.95 多数情况）。

#### 🔹 稳定性点远低于传统预期  
多数模型在远少于全量数据的情况下即进入“稳定区”：

| 场景 | 结果摘要 |
|------|---------|
| **Multi-class** | 4/6 数据集在 $N=4$（约 140k 样本）内达到 MAPD ≤ 5%；所有数据集在 $N=9$ 内满足 MAPD ≤ 2% |
| **Binary** | 收敛更快，4/6 数据集在 $N=4$ 达到 MAPD ≤ 2%，全部在 $N=9$ 内满足 ≤1% |

> 示例：
> - **UCI-HAR (multi)**：仅用 3 个点（105k 样本）就达到 MAPD=0.943% < 2%
> - **REALDISP (binary)**：$N=2$ 即达 MAPD=0.159%，几乎立即稳定

#### 🔹 高粒度任务收敛慢，但趋势仍可预测  
- PAMAP2（18类）、REALDISP（33类）初期波动大，需更多点才能稳定
- 但仍能在 $N=7–9$ 内达到高精度预测

---

### ✅ 消融实验结果（Ablation Study）

为验证框架泛化能力，替换 backbone 为 **Three-layer CNN** 进行测试（UCI-HAR 和 PAMAP2）：

| 模型 | 数据集 | 达到 5% MAPD 的 $N$ | 达到 2% MAPD 的 $N$ |
|------|--------|---------------------|---------------------|
| CNN-BiLSTM | UCI-HAR | 2 | 3 |
| 3-layer CNN | UCI-HAR | 2 | 3 |
| CNN-BiLSTM | PAMAP2 | 7 | 9 |
| 3-layer CNN | PAMAP2 | 7 | 9 |

✅ **结论**：学习曲线形态主要由**数据结构与分布**决定，而非特定网络架构。说明该现象具有强鲁棒性和可迁移性。

---

## 4. 关键结论和发现

### ✅ 主要发现  

1. **惯性传感器分类任务的学习曲线普遍遵循对数增长模式**，而非 Power Law。
2. **模型性能可在极小样本下可靠预测其渐近表现**，无需等到大数据训练完成。
3. **“稳定性点”是一个有效的工程指标**，可用于确定最小必要数据量，显著减少人工标注负担。
4. **该模式跨 dataset、task 类型、model architecture 具有一致性**，表明其反映的是数据本身的统计特性。

> 👉 “我们应从追求最大数据量转向优化 data efficiency。”

---

### ⚠️ 方法的局限性  

1. **假设 accuracy 增长是对数形式**：虽经实证验证，但仍是经验性假设，不能保证适用于所有任务（如 regression 或 anomaly detection）。
2. **仅验证于 classification 任务**：是否适用于姿态估计、轨迹回归等连续输出任务尚未检验。
3. **依赖均匀增量采样**：实际部署中可能存在类别不平衡、域偏移等问题，影响外推准确性。
4. **未考虑主动学习或数据质量因素**：仅关注 quantity，未涉及 sample quality 或 selection strategy。

---

### 🔮 未来工作方向  

1. **将框架扩展至 regression 和 sequence-to-sequence 任务**（如步态周期预测、位置回归）
2. **结合 active learning**，实现“最优数据选择 + 最小样本需求”的联合优化
3. **探索跨 domain 的迁移学习场景下的学习曲线建模**
4. **开发开源工具包**，帮助研究人员快速执行 pilot study 并估算所需数据总量
5. **纳入数据多样性、标注噪声等因素的影响建模**

---

## ✅ 总结一句话  
> 本论文通过系统实证研究揭示：**惯性传感器分类任务的性能提升遵循对数规律，可通过少量试点训练精准预测最终效果，进而以“数据效率”取代“数据规模”作为深度学习开发的新范式**。

</details>

---

### 13. [AgentKGV: Agentic LLM-RAG Framework with Two-Stage Training for the Fact Verification of Knowledge Graphs](https://arxiv.org/abs/2607.09092)

**Authors**: Yumin Heo, Hyeon-gu Lee, Sumin Seo, Youngjoong Ko  
**Category**: cs.CL  
**Published**: 2026-07-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.09092v1  

#### Abstract
Knowledge graphs (KGs) are often automatically constructed from large-scale corpora, but they inevitably contain factual errors due to noisy sources and extraction failures, and verifying them reliably at industrial scale remains a critical challenge. To address this, we propose AgentKGV, the Agenti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AgentKGV: Agentic LLM-RAG Framework with Two-Stage Training for the Fact Verification of Knowledge Graphs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
知识图谱（KG）通常从大规模文本中自动构建，但由于源数据噪声、句法歧义以及 NLP 模型抽取错误，导致其包含大量**事实性错误**。传统的验证方法在工业级部署中面临以下挑战：
- **Graph-based 方法**依赖图结构一致性，难以检测真实世界中的事实错误；
- **LLM-based 方法**易产生领域特定谓词（predicate）上的幻觉；
- **Single-turn RAG 方法**因 (s,p,o) 三元组与自然语言文档之间存在**模态鸿沟**（modality gap），单次检索常无法获取相关证据；
- 工业场景下还需兼顾**准确性**与**推理成本**（如检索调用次数）。

### 提出了什么新方法或新思路
本文提出 **AgentKGV** —— 一种基于 Agentic LLM-RAG 的 KG 事实验证框架，并引入**两阶段训练策略**以提升准确性和效率：

#### （1）AgentKGV 框架核心机制
- **Dynamic Routing**：模型首先判断是否可通过内部参数化知识直接验证三元组，若不确定则启动外部检索，避免不必要的计算开销。
- **(s,p,o)-Grounded Iterative Query Rewriting**：将压缩形式的三元组逐步重写为适合文档检索的自然语言查询，利用显式的 (s,p,o) 结构进行多样化表达，缓解表面形式不匹配问题。
- **Multi-turn Retrieval with Context Summarization**：通过迭代检索与反馈优化查询，同时使用 Summarizer 模块维护紧凑的历史摘要，控制上下文长度并支持跨轮次有效推理。

#### （2）Two-Stage Training Strategy
为解决工业部署中的两个实际挑战——**长尾谓词上查询重写的不稳定性**与**搜索策略低效**，设计了分阶段训练方案：
- **Stage 1: Turn-level Distillation-based SFT**
  - 使用大模型（teacher）生成完整验证轨迹；
  - 提取成功轨迹的最后一轮作为监督信号，分别训练“查询重写”和“基于证据的判断”能力；
  - 小模型由此获得稳定的语义锚点，尤其对罕见谓词更具泛化性。
- **Stage 2: Trajectory-level GRPO（Group Relative Policy Optimization）**
  - 在整条轨迹级别进行强化学习优化；
  - 奖励函数结合最终正确率与每步检索惩罚项 $ R_{\text{search}} = -\alpha \cdot \max(0, N_{\text{search}} - 1) $；
  - 鼓励模型学会何时停止搜索，在证据不足时也能做出合理判断，显著减少冗余检索。

### 相比现有方法的优势
| 维度 | AgentKGV 的优势 |
|------|----------------|
| **准确性** | 显著优于 Direct LLM、Single-turn RAG 和 IRCoT，尤其在 long-tail 和 unseen 谓词上表现稳健 |
| **效率** | GRPO 将平均检索次数从 3.24 降至 1.63，大幅降低推理成本 |
| **鲁棒性** | 迭代查询重写 + 动态路由增强对表面形式变化和稀疏谓词的适应能力 |
| **可扩展性** | 两阶段训练使小模型具备接近大模型的推理能力，适合工业部署 |

---

## 2. 核心实验方法和设置

### 使用的数据集
#### （1）英文开放域基准：T-REx
- 来源于 [T-REx (Elsahar et al., 2018)]，包含 50K 三元组、376 个谓词；
- 构建两种划分：
  - **Long-tail split**：每个谓词最多保留 10 个训练样本，模拟低频谓词场景；
  - **Unseen split**：完全未见于训练集的谓词，测试零样本迁移能力；
- 构造**硬负例（hard negatives）**：替换宾语为同谓词下语义最相似但非真实的对象（基于 BAAI/bge-base-en-v1.5 embedding），提高判别难度。

#### （2）韩文企业 KG（Industrial Application）
- 自建韩语企业知识图谱，来源于真实抽取系统；
- 包含 337 正例、114 负例，均由人工标注验证；
- 特点：语言不同、领域专有、检索器为内部词法检索器（lexical retriever），更贴近真实工业环境。

### 实验设置和评估指标
| 设置项 | 配置 |
|--------|------|
| Retriever | 英文用 BGE Dense Retriever；韩文用内部 Lexical Retriever |
| Top-k 文档 | 每次检索返回 top-5 文档 |
| 最大回合数 | 8 轮 |
| Summarizer 模型 | Qwen-2.5-7B-Instruct |
| 教师模型（Teacher） | gpt-3.5-turbo-120b |
| 主干模型（Backbone） | Qwen-2.5-7B-Instruct（所有 baseline 共享） |
| GRPO 惩罚系数 $\alpha$ | T-REx 上设为 0.05；韩文 KG 上为 0.12 |

### 评估指标
- **Per-class F1**：正类（Pos-F1）、负类（Neg-F1）
- **Macro-F1**：两类 F1 的宏平均
- **Average Search Calls**：每条三元组平均调用检索次数
- 对未输出有效标签的情况，默认预测为 negative

### 基线方法对比
| Baseline | 描述 |
|---------|------|
| **Direct LLM** | 仅靠 LLM 参数知识判断，无检索 |
| **Single-turn RAG** | 一次性将三元组转为查询，检索一次后判断 |
| **IRCoT (Trivedi et al., 2023)** | 迭代检索 + Chain-of-Thought 推理，代表先进 RAG 方法 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### ✅ T-REx Long-tail Split 结果
| Method | Pos-F1 | Neg-F1 | Macro-F1 | Avg Calls |
|-------|--------|--------|----------|-----------|
| Single-turn RAG | 0.747 | 0.806 | 0.777 | 1.00 |
| IRCoT | 0.819 | 0.848 | 0.834 | 5.39 |
| AgentKGV (framework) | 0.822 | 0.842 | 0.832 | 1.82 |
| + SFT | 0.861 | 0.867 | **0.864** | 3.24 |
| **+ SFT + GRPO** | **0.872** | **0.870** | **0.871** | **1.63** |

> 💡 **相比 Single-turn RAG 提升 9.4% 宏观 F1，且检索次数仅略高**

#### ✅ T-REx Unseen Split（零样本场景）
| Method | Pos-F1 | Neg-F1 | Macro-F1 | Avg Calls |
|-------|--------|--------|----------|-----------|
| Single-turn RAG | 0.759 | 0.822 | 0.791 | 1.00 |
| IRCoT | 0.813 | 0.842 | 0.828 | 5.58 |
| + SFT + GRPO | **0.864** | **0.862** | **0.863** | **1.80** |

> 💡 在完全未见谓词上仍取得最佳性能，说明强泛化能力

#### ✅ 韩文企业 KG 实验结果（Table 2）
| Method | Pos-F1 | Neg-F1 | Macro-F1 |
|-------|--------|--------|----------|
| Direct LLM | 0.591 | 0.375 | 0.483 |
| Single-turn RAG | 0.391 | 0.394 | 0.392 |
| IRCoT | 0.618 | 0.405 | 0.511 |
| AgentKGV (framework) | 0.701 | 0.381 | 0.541 |
| + SFT | 0.745 | 0.440 | 0.593 |
| **+ SFT + GRPO** | **0.794** | **0.422** | **0.608** |

> 💡 即使在跨语言、跨领域、真实错误场景下，仍显著领先，验证工业适用性

### 消融实验结果（Ablation Study）
| 模块 | 影响 |
|------|------|
| **AgentKGV 框架本身** | 相比 Single-turn RAG 提升 ~5.5% Macro-F1，证明动态路由与迭代重写有效 |
| **Stage 1: SFT** | 进一步提升 ~3–4% F1，但检索次数上升至 3.24（过度依赖检索） |
| **Stage 2: GRPO** | 在保持甚至提升精度的同时，将检索次数压回 1.63，消除冗余搜索 |

> 🔍 发现：Stage 1 引入的“retrieve-when-uncertain”行为虽有益，但也带来冗余；Stage 2 成功纠正此偏差。

---

## 4. 关键结论和发现

### 主要发现
1. **Agentic 架构 + 迭代查询重写能有效弥合 (s,p,o) 与文档间的模态鸿沟**，显著提升事实验证准确率；
2. **两阶段训练策略互补性强**：
   - SFT 提供高质量的初始推理与重写能力；
   - GRPO 在轨迹层面优化搜索策略，实现“少而精”的检索；
3. **GRPO 可在不牺牲准确性的前提下，将平均检索次数降低约 50%（3.24 → 1.63）**，极大提升工业部署性价比；
4. 方法在 **long-tail、unseen 谓词及真实企业 KG 上均表现出色**，具备良好泛化与迁移能力。

### 方法的局限性
- **工业基准规模有限**：韩文企业 KG 仅含 451 条样本，虽经人工验证质量高，但统计意义受限；
- **依赖高质量 retriever 和 summarizer**：当前框架中 retrieval 与 summarization 模块固定，未端到端联合优化；
- **训练复杂度较高**：需先收集 teacher 轨迹，再进行 GRPO 训练，流程较长。

### 未来工作方向
- 扩展更大规模的真实工业 KG 测试，探索半自动标注或 human-in-the-loop 方式降低成本；
- 探索轻量化版本或蒸馏策略，进一步压缩模型体积；
- 将框架拓展至多跳推理、因果验证等更复杂的 KG 质量保障任务；
- 研究如何自适应调整 GRPO 中的惩罚系数 $\alpha$，实现动态平衡精度与成本。

--- 

> 📌 **总结一句话**：  
> AgentKGV 通过 **Agentic 架构 + 两阶段训练（SFT + GRPO）**，实现了高精度、低成本的知识图谱事实验证，在 long-tail 和真实工业场景中均展现出卓越性能，为工业级 KG 质控提供了实用解决方案。

</details>

---

### 14. [SiFAR: Synchronization-Free All-Reduce for Low-Latency LLM Inference](https://arxiv.org/abs/2607.08973)

**Authors**: Hritvik Taneja, Anish Saxena, Abhishek Revinipati, Jae Hyung Ju, Neal C. Crago, Moinuddin Qureshi  
**Category**: cs.DC  
**Published**: 2026-07-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.08973v1  

#### Abstract
The rise of reasoning models and agentic systems has made LLM token-generation latency a key bottleneck. Unlike chatbots, whose latency gains saturate at human reading speed, these systems generate intermediate reasoning tokens not consumed by humans. Thus, per-token latency directly determines end-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SiFAR: Synchronization-Free All-Reduce for Low-Latency LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**低延迟大语言模型（LLM）推理**场景中，随着推理系统从传统聊天机器人转向**推理模型（reasoning models）和智能体系统（agentic systems）**，每 token 的生成延迟（TPOT）直接影响端到端响应时间。由于这些中间 token 并不被人类阅读，因此进一步降低 TPOT 成为关键瓶颈。

在此背景下，**Tensor Parallelism (TP)** 被广泛用于提升内存带宽，但其依赖的 **All-Reduce 通信操作**引入了显著的同步开销，尤其是在小批量（low-batch）服务时，成为新的性能瓶颈。作者发现，在 TP=8 配置下，移除 All-Reduce 可使吞吐量提升高达 **43%**，表明其已成为主导延迟的关键因素。

### 🚀 提出的新方法：SiFAR（Synchronization-Free All-Reduce）
为解决 All-Reduce 中高昂的同步代价，论文提出 **SiFAR**，一种专为低延迟 LLM 推理优化的无同步 All-Reduce 框架，包含三大核心技术：

| 技术 | 核心思想 | 解决的瓶颈 |
|------|--------|-----------|
| **Dual Buffering** | 通过双缓冲机制消除 **Bottom Barrier** 引发的 Write-After-Write (WAW) 依赖 | 消除写后写冲突，避免等待所有 GPU 完成读取 |
| **Redundant Pull** | 利用现代 NVSwitch 支持的 `multimem.ld_reduce` 指令，在交换机内完成全量数据规约，每个 GPU 主动拉取完整规约结果 | 减少数据传输量（从 $K \times (N-1)$ 降至 $K$），提升可扩展性 |
| **Speculative Reduction** | 在未确认所有 GPU 数据就绪前，**推测性地启动数据拉取**，并通过验证缓存（validation buffer）检查正确性，失败则重试 | 消除 Top Barrier 的显式同步开销，仅保留轻量级验证 |

> 💡 **协同设计（Co-design）** 是关键：SiFAR 将通信逻辑与模型执行深度耦合，突破了传统 NCCL 等通用通信库无法假设缓冲区复用模式的限制。

### 🔍 相比现有方法的优势

| 方法 | 局限性 | SiFAR 的优势 |
|------|-------|-------------|
| **Oneshot All-Reduce** | 数据传输随 GPU 数线性增长，扩展性差；需 Bottom Barrier | Redundant Pull 显著降低传输量，Dual Buffering 消除 Barrier |
| **Twoshot All-Reduce** | 需两轮通信 + 不可避免的 Bottom Barrier（Read-After-Write） | 单轮通信 + 无 Bottom Barrier，延迟更低 |
| **Kernel Fusion / Microbatching** | 依赖大量计算来掩盖通信，对小批量无效 | 直接减少通信延迟，而非隐藏，适用于低批处理场景 |
| **NCCL / TRT-LLM 实现** | 保守同步策略导致高 Barrier 开销 | SiFAR 通过推测和冗余设计实现“近零”同步代价 |

---

## 2. 核心实验方法和设置

### 📊 使用的模型
- **Llama-3.1-8B**：稠密 Transformer 模型
- **Qwen3.5-397B-17B**：MoE（Mixture-of-Experts）模型，每次激活 17B 参数，总参数达 397B

> 所有实验运行于 **FP8 精度**

### ⚙️ 实验硬件平台
- **8× NVIDIA H200 GPU**（单节点）
- 每卡 141GB HBM3e 内存
- 使用 **CUDA 12.9**
- TP Degree 测试范围：TP=2, 4, 8

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **TPOT (Time-per-Output-Token)** | 解码阶段生成每个 token 的平均耗时，核心延迟指标 |
| **Throughput (tokens/s)** | 端到端吞吐量，反映整体性能提升 |
| **All-Reduce Latency** | 单独测量 All-Reduce 操作的延迟，用于归因分析 |
| **Tail Latency (p99, p99.9)** | 衡量系统稳定性与最坏情况表现 |

### 🧪 基线方法对比
| 基线 | 说明 |
|------|------|
| **Best of Oneshot & Twoshot** | 动态选择更优算法作为主基线（vLLM/SGLang 默认策略） |
| **TRT-LLM Oneshot** | 使用私有缓冲区绕过 Bottom Barrier，但仍为 Pull-and-Reduce |
| **Lamport-style Push-based All-Reduce** | 基于 in-network multicast 的推送式规约 |
| **NCCL 2.30 (auto)** | NVIDIA 官方集合通信库，启用 LL 协议等低延迟优化 |

---

## 3. 主要实验结果和性能指标

### 📈 端到端吞吐量提升
| 模型 | TP 配置 | 吞吐量提升 |
|------|--------|----------|
| **Llama-3.1-8B** | TP=8 | **+18.6%** |
| **Qwen3.5-397B-17B** | TP=8 | **+13.1%** |

> 提升随 TP 度增加而增大，说明 All-Reduce 瓶颈在更高并行度下更显著。

### ⏱️ All-Reduce 延迟降低
| 配置 | 最大延迟降低 |
|------|------------|
| **TP=8, 8KB payload** | **52%** |
| **平均延迟降低** | 30–50% across configs |

> 图 15 显示 SiFAR 在所有配置下均优于其他实现。

### 🔬 消融实验（Ablation Study）
逐步启用三项技术的效果（以 Llama-3.1-8B, TP=8 为例）：

| 组件 | 吞吐量贡献 |
|------|-----------|
| **Redundant Pull** | +6.7% |
| **+ Dual Buffering** | 再 +3.4% → 累计 +10.1% |
| **+ Speculative Reduction** | 再 +8.5% → 总计 **+18.6%** |

> 三者互补，共同构成完整优化链。

### 🔄 推测失败与重试开销
- **平均 mis-speculation rate**：Llama-3.1-8B 达 **32.2%**（TP=8）
- 但重试带来的性能损失仅 **~1.6%**
- 原因：重试成本低于首次尝试（无需重复启动开销），且验证机制高效

> 图 25 显示即使多次重试，平均延迟仍远低于基线。

### 📉 对尾部延迟的影响
| 模型 | p50 降低 | p99 降低 |
|------|---------|--------|
| **Llama-3.1-8B** | 18.6% | 16.0% |
| **Qwen3.5-397B-17B** | 13.7% | 12.2% |

> **mis-speculation retries 未加剧尾部延迟**，说明机制稳定可靠。

### 📏 其他敏感性测试
| 条件 | 结果趋势 |
|------|---------|
| **输入序列长度增加（ISL=1K → 16K）** | 增益下降（注意力占比上升），但在 ISL=16K 仍保持 **14.6% (Llama) / 10.2% (Qwen)** 提升 |
| **Batch Size 增加（BS=1 → 4）** | 增益从 **18.6% → 10.2%**，因计算占比上升，All-Reduce 影响减弱 |
| **与 Compute-Comm Fusion 对比** | Fusion 在小批量下无效（无足够计算掩盖通信），而 SiFAR 直接优化通信本身 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **All-Reduce 的同步开销是低延迟 LLM 推理的新瓶颈**  
   - 在 TP=8 下，Barrier 开销占 All-Reduce 总时间的 **32–62%**，远超数据传输本身。
   
2. **传统通信优化（如融合、microbatching）在低批处理下失效**  
   - 小批量缺乏足够的计算来掩盖通信，必须直接减少通信延迟。

3. **SiFAR 通过协同设计实现“近零”同步开销**
   - 利用 **in-switch reduction** 和 **speculative execution**，将 All-Reduce 延迟降低最多 **52%**。
   - 端到端吞吐提升 **18.6%（Llama）和 13.1%（Qwen）**，且不恶化尾部延迟。

4. **推测机制安全有效**
   - 依赖 GPU 执行的单调性（monotonicity）保证验证正确性。
   - 实验验证输出与 Twoshot 基线完全一致，无静默错误。

### ⚠️ 方法的局限性
1. **依赖特定硬件支持**  
   - 需要支持 `multimem.ld_reduce` 的 NVSwitch（如 Hopper 架构及以后）。
   - 不适用于仅支持传统 RDMA 的网络环境。

2. **基于 GPU 执行单调性假设**
   - 虽然在 Megakernel/CUDA Graph 场景下成立，但在存在频繁抢占或异步任务的复杂环境中可能不稳健。

3. **主要针对 Decode Phase**
   - Prefill 阶段通常为计算密集型，All-Reduce 不是瓶颈，SiFAR 收益较小。

### 🔮 未来工作方向
1. **扩展至其他 Collective Operations**  
   - 如 All-Gather、Reduce-Scatter，构建完整的“无同步”通信栈。

2. **跨节点分布式推理支持**
   - 当前工作聚焦单节点多 GPU，未来可探索在 RDMA + Smart NIC 上实现类似推测机制。

3. **动态负载均衡下的适应性优化**
   - 在 MoE 或动态批处理场景中，GPU 进度差异更大，需增强推测容错能力。

4. **编译器自动集成**
   - 将 SiFAR 模式纳入 LLM 编译框架（如 Mirage），实现自动化优化。

---

> 📌 **总结一句话**：  
> **SiFAR 通过协同设计通信与执行，利用 in-switch reduction、双缓冲和推测执行，首次实现了近乎“零同步”的 All-Reduce，显著提升了低延迟 LLM 推理的吞吐与响应速度，为下一代推理引擎提供了关键通信优化路径。**

</details>

---

### 15. [How are linear representations learned? Exact solutions to the dynamics of abstraction](https://arxiv.org/abs/2607.08843)

**Authors**: William W. Yang, Andrew M. Saxe, Peter E. Latham  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.08843v1  

#### Abstract
In artificial and biological neural networks, concepts are often encoded as consistent linear directions in representation space. In deep learning, this idea is known as the linear representation hypothesis and underpins many interpretability and control methods based on linear probes, from concept ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：How are linear representations learned? Exact solutions to the dynamics of abstraction

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文旨在解决深度神经网络中**抽象表示（abstraction）**的动态学习机制这一核心问题。尽管已有研究证明在训练收敛后，网络可能达到完美的线性表示（即 `linear representation hypothesis`），但这些理论无法解释训练过程中非单调、存在平台期等复杂动态现象（如 Fig. 1C 所示）。本文的核心问题是：**抽象表示是如何在训练过程中逐步形成的？其动态轨迹受哪些因素影响？**

### 提出了什么新方法或新思路
论文提出了一套**解析性的动力学理论框架**来研究抽象过程的动态演化，其核心创新点如下：

- **提出了“抽象”（abstraction）的量化动态变量**：将“抽象”定义为不同上下文中同一概念向量之间的余弦相似度（如 `v_king - v_queen` 与 `v_man - v_woman` 的夹角），使其成为一个可追踪的连续动态变量。
- **建立了最小化线性网络模型并求得精确解**：在一个两层线性网络的简化设定下，通过假设 `Variable-Projected Readout` 和 `Two-Factor Symmetry (2FS)`，推导出特征核（kernel）和抽象度 `α(t)` 在梯度流下的**精确解析解**（exact implicit solution）。
- **将理论扩展至非线性网络**：利用无限宽度极限下的 NNGP 理论，分析了 `erf` 和 `ReLU` 等非线性激活函数对抽象动态的影响，并提出了普适的**衰减定律（attenuation law）**。

### 相比现有方法的优势
相比以往仅关注训练终点的理论，本工作的优势在于：
- **动态视角**：首次提供了从训练开始到结束的完整抽象轨迹的解析描述，而非仅仅预测最终状态。
- **可解释性强**：得出的解析公式揭示了数据几何（data/target geometry）、网络深度（depth）、初始化尺度（initialization scale）等因素如何定量地影响抽象过程。
- **理论指导实践**：基于理论推导出的“衰减定律”，提出了可提升模型可解释性和控制能力的实际干预方法（如局部 GELU 消融）。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **合成数据集（Synthetic datasets）**：用于验证理论模型，构建具有可控 `shape` 和 `color` 两个二元潜变量的 2×2 因子设计数据，模拟 `2FS` 结构。
- **3dshapes 数据集**：一个真实世界的视觉数据集，从中选取红/蓝立方体/球体四个类别，用于在卷积网络（ResNet）上进行实验，验证理论在更复杂模型上的适用性。
- **Gemma 4 模型**：使用开源的大语言模型 `Gemma-4-E2B`，在其残差流（residual stream）中分析多语言翻译概念（如英语-西班牙语词对）的抽象表示。
- **灵长类动物腹侧视觉通路数据**：使用公开的 `Majaj-Hong` 神经记录数据集，分析猕猴 V4 和 IT 区域神经群体编码的抽象程度。

### 实验设置和评估指标
- **评估指标**：
  - **抽象度（abstraction score）**：核心指标，计算为同一概念在不同上下文中的向量差的平均余弦相似度。
  - **探针泛化误差（probe generalization error）**：衡量一个在特定上下文（如红色样本）上训练的线性分类器，在另一个上下文（如蓝色样本）上的测试错误率，用以验证抽象度对下游任务的预测能力。
- **实验设置**：
  - 在合成数据上训练小型线性网络、ReLU 网络和 ResNet，监控每一步的抽象度变化。
  - 对 `Gemma 4` 和 `DINOv3` 等预训练模型进行前向传播，提取中间层表示，并实施**局部 GELU 消融**（local GELU ablation）作为干预实验。
  - 对神经科学数据进行种群水平的分析，比较不同脑区的抽象度。

### 基线方法对比
- **理论基线**：与仅预测终点完美抽象的经典理论（如 [19]）对比，本文理论能解释非单调、平台期等动态行为。
- **实验基线**：在消融实验中，以**未修改的原始模型**作为基线（baseline），比较**消融非线性后的模型**在抽象度和探针泛化上的表现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **理论预测验证**（Fig. 3）：
  - **初始化尺度**：最大抽象度随权重初始化方差增大而**显著降低**（支持 Theorem 5）。
  - **网络深度**：最终抽象度和逐层抽象度均随网络深度增加而**提高**（支持 Theorem 7）。
- **GELU 消融实验**（Fig. 5）：
  - **DINOv3**：在所有 24 层中，局部 GELU 消融均**提升或保持**了形状抽象度。
  - **Gemma 4**：
    - **抽象度**：18 个双语概念的抽象度在消融后**全部提升**，平均值从 0.207 升至 0.242。
    - **探针泛化**：17/18 个概念的线性探针泛化准确率得到提升，平均准确率从 0.944 升至 0.953（约 +1 个百分点）。
- **神经科学应用**（Fig. 6）：
  - 在猕猴腹侧视觉通路中，IT 区域的概念抽象度（如“有肢/无肢”、“自然/人造”）**显著高于**V4 区域，支持“更深的脑区应更抽象”的理论预测。

### 与基线方法的对比结果
- 本文的动态理论成功解释了经典理论无法捕捉的**非单调抽象轨迹**和**训练平台期**。
- 在实际模型中，通过简单的**非线性消融**，就能系统性地提升概念的抽象度和线性探针的泛化能力，这直接验证了理论的实用价值。

### 消融实验结果
- **局部 GELU 消融**是本文的关键消融实验。结果显示，移除非线性激活函数后，特征空间中的抽象度普遍提高，证实了论文提出的“**衰减定律**”（非线性会削弱抽象度）。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **数据与目标几何共同决定终点抽象度**（Theorem 4）：最终的抽象度由输入和目标的信噪比（SNR）的几何平均决定。
2. **深度提升抽象**（Theorem 7）：在网络深度 `L` 上，抽象度在 `arctanh` 空间中呈线性插值，因此更深的网络能达到更高的最终抽象度。
3. **初始化尺度控制最大抽象**（Theorem 5）：在“rich regime”（小初始化）下，模型可以短暂达到接近完美的抽象，即使最终值较低；而在“lazy regime”（大初始化）下，抽象度不会超过其最终值。
4. **非线性衰减抽象**（Theorem 8）：提出了**衰减定律（attenuation law）**，证明 `erf` 和 `ReLU` 等非线性会削弱特征层相对于预激活层的抽象度。
5. **非线性改变动态依赖**：`ReLU` 网络的抽象动态更依赖于输入几何而非目标几何，这可能解释了其在自监督学习中的优越性。

### 方法的局限性
- **理论假设较强**：精确解依赖于 `2FS` 对称性和 `Variable-Projected Readout` 等理想化假设，虽然在真实模型上观察到了定性一致的结果，但严格满足这些条件的情况较少。
- **难以扩展到深非线性网络**：目前的理论主要适用于浅层或无限宽度的网络，对于有限宽度、深层非线性网络的精确动态仍难以处理。
- **任务范围有限**：当前框架主要针对二元或少数几个离散概念，对更复杂的、层次化的概念体系的建模有待发展。

### 未来工作方向
- **放宽理论假设**：发展更一般化的理论，减少对 `2FS` 等强对称性假设的依赖。
- **深化非线性网络理论**：将动力学理论扩展到更现实的深层非线性网络架构。
- **广泛应用**：进一步探索 `local GELU ablation` 等基于理论的干预方法在 LLMs 的可解释性、安全性和控制方面的潜力。
- **连接认知科学**：利用该理论框架更深入地理解大脑中抽象表征的形成机制。

</details>

---

### 16. [COBS: Cumulant Order Block Sparse Attention](https://arxiv.org/abs/2607.09052)

**Authors**: Alexander Tian, Aditya Ghai, Sanjit Neelam, Zaal Vasania, Akshay Mishra  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.09052v1  

#### Abstract
Block sparse attention is a hardware friendly way to alleviate the key-value (KV) cache read bottleneck in large language models (LLMs). However, it is not prevalent among leading open-weight LLMs, which rely instead on dense attention or fine-grained selection, thereby motivating our analysis. We s...

---

### 17. [Agora: Enhancing LLM Agent Reasoning Via Auction-Based Task Allocation](https://arxiv.org/abs/2607.09600)

**Authors**: Kaiji Zhou, Ales Leonardis, Yue Feng  
**Category**: cs.AI  
**Published**: 2026-07-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.09600v1  

#### Abstract
Enhancing the reasoning capabilities of large language model (LLM) agents requires effective orchestration of diverse expert models and tools. However, existing frameworks typically call APIs based on coarse-grained matching between tasks and the functions of expert models or tools, while overlookin...

---

### 18. [Pattern-Aware Graph Neural Networks for Handling Missing Data](https://arxiv.org/abs/2607.08915)

**Authors**: Minett Tran, Taehee Jeong  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.08915v1  

#### Abstract
Missing data is ubiquitous in real-world datasets. Traditional methods either discard incomplete samples or apply imputation techniques that ignore potentially informative missingness patterns, implicitly assuming that missingness occurs randomly. However, missingness patterns might provide addition...

---

### 19. [CoCoT-EEG: Contrastive-Pretrained Multiscale Convolutional Transformer for EEG Decoding](https://arxiv.org/abs/2607.09543)

**Authors**: Gabriel Mahuas, Victoria Shevchenko, Ugo Tanielian, Yassir Bendou, Richard Gao  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.09543v1  

#### Abstract
Self-supervised pretrained foundation models (FM) have shown early promise for non-invasive electroencephalogram (EEG) decoding applications. Many recent large-scale models converged on the approach of tokenizing raw EEG followed by masked reconstruction pretraining. However, this recipe has been sh...

---

### 20. [LongMedBench: Benchmarking Medical Agents for Long-Horizon Clinical Decision-Making](https://arxiv.org/abs/2607.09322)

**Authors**: Yanzhen Chen, Zihan Xu, Xiaocheng Zhang, Zhiting Fan, Weiqi Zhai, Hongxia Xu, Zuozhu Liu  
**Category**: cs.AI  
**Published**: 2026-07-13  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.09322v1  

#### Abstract
In this work, we introduce LongMedBench, a real-world EHR-based benchmark for long-horizon clinical decision-making. Prior evaluations of LLM-based medical agents have largely emphasized short-context knowledge QA and tool use. However, real-world medical care is inherently longitudinal, and clinici...

---

### 21. [A Machine Learning Surrogate for Component Criticality Ranking in Interdependent Power-Communication Networks](https://arxiv.org/abs/2607.08918)

**Authors**: Sohini Roy, Xheni Hylviu  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.08918v1  

#### Abstract
Cyber-physical power systems are vulnerable to cascading failures caused by tight interdependencies between power and communication infrastructures. Evaluating these failures over large N-k contingency sets with a high-fidelity simulator is computationally prohibitive for resilience planning. Using ...

---

### 22. [Learning More from Less: Reinforcement Learning from Hindsight](https://arxiv.org/abs/2607.09042)

**Authors**: Iris Xu, Sunshine Jiang, John Marangola, Nitish Dashora, Richard Li, Thomas Liu, Zexue He, Yuheng Zhi, Alex Pentland, Pulkit Agrawal, Zhang-Wei Hong  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.09042v1  

#### Abstract
Reinforcement learning (RL) is increasingly used to post-train vision-language-action (VLA) models, but every update consumes robot rollouts that are slow and costly to collect, making sample efficiency a central concern. Manipulation tasks typically provide only sparse rewards, so a weak policy fai...

---

### 23. [EvoLP: Self-Evolving Latency Predictor for Model Compression in Real-Time Edge Systems](https://arxiv.org/abs/2607.09063)

**Authors**: Shuo Huai, Hao Kong, Shiqing Li, Xiangzhong Luo, Ravi Subramaniam, Christian Makaya, Qian Lin, Weichen Liu  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.09063v1  

#### Abstract
Edge devices are increasingly utilized for deploying deep learning applications on embedded systems. The real-time nature of many applications and the limited resources of edge devices necessitate latency-targeted neural network compression. However, measuring latency on real devices is challenging ...

---

### 24. [ARCANA: A Reflective Multi-Agent Program Synthesis Framework for ARC-AGI-2 Reasoning](https://arxiv.org/abs/2607.09059)

**Authors**: Kunbo Zhang, Lei Fu, Zeyu Wang, Zijing Liu, Kejian Tong  
**Category**: cs.AI  
**Published**: 2026-07-13  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2607.09059v1  

#### Abstract
We present ARCANA, a collaborative multi agent framework for solving ARC AGI 2 tasks under strict test time and hardware constraints. ARCANA decomposes each task into iterative perception, hypothesis generation, symbolic execution, and reflective refinement. A perceptual grounding agent builds objec...

---

### 25. [MedRealMM: A Real-World Multimodal Benchmark for Chinese Online Medical Consultation](https://arxiv.org/abs/2607.09142)

**Authors**: Runhan Shi, Quan Zhou, Yuqian Xu, Shuai Yang, Xin Wu, Zitong Zhou, Hui Liu, Bin Cha, Zheming Wang, Liya Li, Wei Wei, Haoyuan Hu, Jun Xu  
**Category**: cs.AI  
**Published**: 2026-07-13  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2607.09142v1  

#### Abstract
Large language models (LLMs) are increasingly deployed in online medical consultation, yet existing benchmarks remain poorly aligned with real clinical practice. Many rely on synthetic conversations or patient simulators, omit patient-uploaded medical images, or evaluate open-ended clinical response...

---

### 26. [How Does Bayesian Causal Discovery Fail? Characterising Structural Consequences in Linear Gaussian Networks under Latent Confounding](https://arxiv.org/abs/2607.09449)

**Authors**: Debargha Ghosh, Silja Renooij, Anna Kononova  
**Category**: cs.AI  
**Published**: 2026-07-13  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2607.09449v1  

#### Abstract
Bayesian causal discovery is widely used for its ability to quantify epistemic uncertainty over directed acyclic graphs (DAGs) through posterior inference. However, its behaviour under latent confounding remains poorly understood, as existing work typically notes that confounding breaks identifiabil...

---

### 27. [Action-Factored Multi-Agent Reinforcement Learning for Scalable Quantum Device Tuning](https://arxiv.org/abs/2607.09422)

**Authors**: Edwin De Nicolo, Rahul Marchand, Cornelius Carlsson, Pranav Vaidhyanathan, Natalia Ares  
**Category**: cs.LG  
**Published**: 2026-07-13  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2607.09422v1  

#### Abstract
Cooperative multi-agent reinforcement learning is well suited to problems with large parameter spaces and exploitable local structure, such as the tuning of electrostatically-defined quantum-dot arrays. However, if parameter cross-talk is strong, a non-stationary environment from the perspective of ...

---

### 28. [Interval Certifications for Multilayered Perceptrons via Lattice Traversal](https://arxiv.org/abs/2607.08773)

**Authors**: Merkouris Papamichail, Konstantinos Varsos, Giorgos Flouris, Jo\~ao Marques-Silva  
**Category**: cs.AI  
**Published**: 2026-07-13  
**Score**: 3.0  
**Type**: new  
**ArXiv ID**: 2607.08773v1  

#### Abstract
In this work we present a rigorous theoretical framework to a foundational problem of AI safety, namely adversarial robustness. In particular, we show that the adversarial robustness problem can be reduced to a lattice traversal problem. Each element of this lattice corresponds to an interval, i.e.,...

---

### 29. [CogniConsole: Externalizing Inference-Time Control as a Formal Abstraction for Reliable LLM Interactions](https://arxiv.org/abs/2607.08774)

**Authors**: Vanessa Figueiredo, Wilter Franceschi  
**Category**: cs.AI  
**Published**: 2026-07-13  
**Score**: 3.0  
**Type**: new  
**ArXiv ID**: 2607.08774v1  

#### Abstract
Reliability in large language model (LLM) systems is typically framed as a function of model capability. We challenge this by demonstrating that reliability is significantly influenced by \emph{inference-time control} -- the computational layer governing task framing and context selection. We introd...

---

### 30. [Neuro-Agentic Control: A Deep Learning-based LLM-Powered Agentic AI Framework for Controlling Security Controls](https://arxiv.org/abs/2607.09076)

**Authors**: Saroj Gopali, Bipin Chhetri, Deepika Giri, Sima Siami-Namini, Akbar Siami Namin  
**Category**: cs.AI  
**Published**: 2026-07-13  
**Score**: 3.0  
**Type**: new  
**ArXiv ID**: 2607.09076v1  

#### Abstract
Cyberattacks on operational technology are increasingly causing costly downtime and physical damage, exposing the limitations of traditional rule-based monitoring in industrial IoT environments. While Large Language Models (LLMs) have strong semantic reasoning abilities to assist in decision support...

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

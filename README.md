# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-12-09 05:56:33 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Stable-MoE: Lyapunov-based Token Routing for Distributed Mixture-of-Experts Training over Edge Networks](https://arxiv.org/abs/2512.06784)

**Authors**: Long Shi, Bingyan Ou, Kang Wei, Weihao Zhu, Zhe Wang, Zhiyong Chen  
**Category**: cs.DC  
**Published**: 2025-12-09  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2512.06784v1  

#### Abstract
The sparse activation mechanism of mixture of experts (MoE) model empowers edge intelligence with enhanced training efficiency and reduced computational resource consumption. However, traditional token routing in distributed MoE training faces significant challenges in resource-constrained edge netw...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Stable-MoE: Lyapunov-based Token Routing for Distributed Mixture-of-Experts Training over Edge Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统 **distributed MoE** 在资源受限的 **edge networks** 中面临以下挑战：
- **异构计算能力**：边缘服务器的计算性能差异大；
- **随机 token 到达**：token 流量具有不确定性；
- **资源调度不均衡**：导致 **token backlog** 和 **energy backlog** 积压；
- **系统吞吐量下降** 和 **训练效率降低**。

现有 token routing 方法（如 top-K、random routing）未考虑边缘设备的实时资源状态，容易造成部分服务器过载而其他空闲，影响整体性能。

---

### **提出的新方法：Stable-MoE**
作者提出了 **Stable-MoE** —— 一种基于 **Lyapunov optimization** 的在线 token routing 框架，用于分布式 MoE 在边缘网络中的高效训练。

#### **核心创新点**：
- **联合优化目标**：同时最大化 **系统吞吐量（system throughput）** 和 **gating consistency**（保持与原始 MoE 路由策略的一致性）；
- **动态资源感知路由**：在每个 time slot 动态决策 token routing 策略和边缘服务器的 **CPU frequency allocation**；
- **长期队列稳定性保障**：
  - 引入 **token queue** 和 **energy queue** 模型；
  - 使用 **Lyapunov drift-plus-penalty** 方法将长期随机优化问题转化为每时隙可解的子问题；
- **无需未来系统状态知识**：实现 **online decision-making**，适用于实际动态环境。

---

### **相比现有方法的优势**
| 对比维度 | 传统方法（如 top-K、random） | Stable-MoE |
|--------|----------------------------|-----------|
| 资源感知 | ❌ 忽略边缘服务器负载和能耗 | ✅ 显式建模并优化资源利用 |
| 队列稳定性 | ❌ 容易出现 backlog 积压 | ✅ 通过 Lyapunov 控制确保长期稳定 |
| 吞吐量 | 较低，受瓶颈节点限制 | 显著提升（实验显示 +40%） |
| 实现复杂度 | 简单但次优 | 复杂但可通过 MIP solver 求解 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **SVHN**（Street View House Numbers）：32×32 彩色图像，数字识别任务；
- **CIFAR-100**：32×32 彩色图像，100 类物体分类任务。

> 两者均用于图像分类，token 来源于 mini-batch 图像输入。

---

### **实验设置**
- **边缘网络配置**：
  - $ J = 10 $ 台异构边缘服务器（每台部署一个 expert）；
  - $ K = 3 $：top-3 routing；
  - 时间槽长度 $ T = 1 $ 秒；
  - 平均 token 到达率 $ \lambda = 390 $ tokens/slot（泊松过程生成）；
  - CPU 频率上限 $ f_{\text{max}} = 3 $ GHz；
  - 单 token 计算开销 $ c = 10^7 $ cycles/token；
  - 开关电容系数 $ \kappa = 2 \times 10^{-27} $；
  - 能耗预算 $ E^{\text{max}} \in [3J, 15J] $，平均可用能量 $ E^{\text{avg}} \in [1.5J, 9.5J] $。

- **模型结构**：
  - **gating network**：前馈神经网络；
  - **experts**：卷积层（CNN），用于图像特征提取与分类。

---

### **评估指标**
1. **系统吞吐量（System Throughput）**：单位时间完成处理的 token 数量；
2. **测试准确率（Test Accuracy）**：模型最终分类性能；
3. **队列 backlog 演化**：token queue 和 energy queue 的实时与平均值；
4. **稳定性**：是否避免无限 backlog 积累。

---

### **基线方法对比**
| 基线名称 | 描述 |
|--------|------|
| **Strategy A: Random Routing** | Token 均匀随机分配给所有 experts |
| **Strategy B: Traditional Top-K Routing** | 按 gating score 选择 top-K experts，忽略资源状态 |
| **Strategy C: Queue-aware Routing** | 优先路由到 token queue 最小的 server |
| **Strategy D: Energy-aware Routing** | 优先路由到 energy queue 最小的 server |

> 所有基线均在同一环境下运行以保证公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | Stable-MoE 表现 | 提升幅度 |
|------|----------------|---------|
| **系统吞吐量（SVHN）** | 显著高于所有基线 | **+40% vs Strategy D** |
| **测试准确率（SVHN）** | 收敛至约 **80%** | **+5% 以上 vs 所有 baseline** |
| **测试准确率（CIFAR-100）** | 同样取得最高精度 | 明显优于其他策略 |
| **队列稳定性** | token 和 energy queue 快速收敛至稳态 | 无持续增长趋势 |

---

### **与基线方法的对比结果**
- **图3（Throughput Comparison）**：
  - Stable-MoE 的累计吞吐量增长最快，且随时间拉开差距；
  - Strategy D（energy-aware）虽能管理能耗，但因忽视计算负载仍会导致某些 server 过载，限制整体吞吐。

- **图4（Accuracy Comparison）**：
  - Stable-MoE 在两个数据集上都实现了更快收敛和更高最终准确率；
  - 基线方法中，Strategy B（top-K）表现尚可，但仍低于 Stable-MoE；
  - Random routing（A）和纯队列/能量感知策略（C/D）准确率较低且波动较大。

- **图2（Queue Backlog Evolution）**：
  - Stable-MoE 的 token 和 energy backlog 初始上升后趋于平稳；
  - 表明其有效平衡了资源消耗与任务调度，满足 Lyapunov 设计的 **长期稳定性约束（C5 & C6）**。

---

### **消融实验（隐含分析）**
虽然文中未明确列出消融实验表格，但从设计逻辑可推断：
- 若去除 **Lyapunov drift 控制项** → 队列不稳定，可能崩溃；
- 若仅优化吞吐而不考虑 **gating consistency**（即 $ \mu = 0 $）→ 可能偏离原始 MoE 学习路径，影响模型质量；
- 若固定频率或静态路由 → 无法适应动态流量，性能下降。

---

## **4. 关键结论和发现**

### **主要发现**
1. **资源感知的动态 token routing 至关重要**：在异构边缘环境中，必须结合 **实时队列状态** 和 **能耗约束** 进行调度；
2. **Lyapunov optimization 是有效的理论工具**：成功将复杂的长期随机优化问题转化为可在线求解的每时隙决策问题；
3. **Stable-MoE 实现了性能与稳定的双重提升**：
   - 提高系统吞吐量至少 **40%**；
   - 提升测试准确率超过 **5%**；
   - 保证 token 和 energy queues 的长期稳定；
4. **兼顾模型一致性**：通过引入 gating consistency 正则项，在优化资源的同时不严重偏离原始 MoE 的学习行为。

---

### **方法的局限性**
- **计算开销较高**：每时隙需求解一个混合整数非线性规划问题（mixed-integer programming），依赖 solver（如 branch-and-bound），对大规模系统可能带来延迟；
- **假设理想通信**：未详细建模无线信道干扰或传输延迟（尽管提及 channel-aware work 作为背景）；
- **专家与服务器一一对应**：限制了更灵活的部署方式（如多专家共享一台 server）；
- **缺乏真实边缘硬件验证**：实验基于仿真，尚未在真实 IoT 或移动设备集群中部署。

---

### **未来工作方向**
1. **轻量化在线求解器设计**：开发近似算法或神经求解器以降低 per-slot 决策延迟；
2. **扩展至无线通信联合优化**：整合 **channel state information (CSI)** 和 **带宽分配**，构建端到端通信-计算协同框架；
3. **支持 expert replication 或 sharing**：允许多个 expert 部署于同一 server，提高资源利用率；
4. **应用于 LLM 推理场景**：将 Stable-MoE 思路迁移到大语言模型的 **on-device inference** 场景；
5. **隐私与安全增强**：结合联邦学习（Federated Learning）机制，保护边缘数据隐私。

---

> ✅ **总结一句话**：  
> **Stable-MoE 首次将 Lyapunov 优化引入 distributed MoE 的 token routing，实现了在资源异构边缘网络下的高吞吐、高精度与长期稳定训练，为边缘智能的大模型部署提供了新范式。**

</details>

---

### 2. [Each Prompt Matters: Scaling Reinforcement Learning Without Wasting Rollouts on Hundred-Billion-Scale MoE](https://arxiv.org/abs/2512.07710)

**Authors**: Anxiang Zeng, Haibo Zhang, Hailing Zhang, Kaixiang Mo, Liang Yao, Ling Hu, Long Zhang, Shuman Liu, Shuyi Xie, Yanshi Li, Yizhang Chen, Yuepeng Sheng, Yuwei Huang, Zhaochen Xu, Zhiqiang Zhou, Ziqin Liew  
**Category**: cs.AI  
**Published**: 2025-12-09  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2512.07710v1  

#### Abstract
We present CompassMax-V3-Thinking, a hundred-billion-scale MoE reasoning model trained with a new RL framework built on one principle: each prompt must matter. Scaling RL to this size exposes critical inefficiencies-zero-variance prompts that waste rollouts, unstable importance sampling over long ho...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Each Prompt Matters: Scaling Reinforcement Learning Without Wasting Rollouts on Hundred-Billion-Scale MoE*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对在**百亿参数规模的 Mixture-of-Experts (MoE)** 模型上进行长链推理（LongCoT）强化学习（Reinforcement Learning, RL）时面临的多个系统性和算法性挑战，主要包括：

- **零方差提示（Zero-Variance Prompts, ZVP）**：大量提示在训练中产生相同奖励，导致无有效梯度信号，浪费计算资源。
- **重要性采样不稳定**：在 token 和 sequence 层面的重要性采样在长序列下易崩溃。
- **训练-推理不一致（Train-Infer Mismatch）**：MoE 路由器在训练和推理阶段行为不同，引发数值误差和策略崩溃。
- **优势反转（Advantage Flipping）**：标准 Bradley-Terry（BT）奖励模型在均值附近非单调，导致错误梯度方向。
- **系统瓶颈**：Rollout 阶段延迟高、GPU 利用率低，限制端到端吞吐。

这些问题共同导致大规模 MoE 模型的 RL 训练效率低下、不稳定甚至失败。

---

### **提出了什么新方法或新思路**

作者提出了一套**统一的 RL 框架**，以“**每个提示都应提供有效学习信号**”为核心原则，整合了算法与系统级优化，形成完整训练流水线。主要创新点如下：

#### （1）**Multi-Stage Zero-Variance Elimination（多阶段零方差消除）**
- 在 rollout、reward、actor 更新三个阶段分别采取措施减少零方差提示的影响。
- 包括：扩大探索空间（增大采样数 $N$）、引入长度惩罚等 reward reshaping 技术、以及使用 RL-ZVP 进行优势重塑。
- **效果**：将零方差率降低 17%，显著提升收敛速度。

#### （2）**ESPO（Entropy Importance Sampling Policy Optimization）**
- 提出一种熵自适应的重要性采样策略，在 token-group 级别进行策略优化。
- 将序列按熵分组，对高熵区域赋予更高更新权重，并设计熵自适应的 clipping 阈值。
- **优势**：相比 GRPO/GSPO 的均匀处理，ESPO 更好地平衡探索与稳定性，缓解长序列下的重要性采样脆性。

#### （3）**Router Replay + GenRM for Advantage Stabilization**
- **Router Replay**：记录 vLLM 推理阶段的 MoE 路由决策，并在训练时复用，显著减少训练-推理 log-prob 差异（从 $10^{-3}$ 降至 $10^{-4}$）。
- **Generative Reward Model (GenRM)**：基于 DeepSeek-Distilled-Qwen3-8B 构建，支持 Chain-of-Thought 推理，并采用三类判断（Better/Tie/Worse），避免 BT 模型的非单调性导致的优势反转。
- **效果**：GenRM 与 GPT-4 判断一致性达 84.3%（优于基线 ORM 的 74.1%），Tie 识别准确率达 98.8%。

#### （4）**High-Throughput RL System**
- **FP8-Quantized Rollout**：在 vLLM 中启用 FP8 权重量化，提速约 30%。
- **Length-Aware Scheduling**：通过预测生成长度实现负载均衡，减少 worker 同步等待时间，GPU 闲置率下降 >12%。
- **Overlapped Reward Computation**：rollout 完成就立即开始 reward 计算，减少空闲时间。
- **Multi-Detokenization Parallelism**：并行化 detokenization 流程，加速 reward 处理。

---

### **相比现有方法的优势**

| 维度 | 传统方法局限 | 本文方案优势 |
|------|--------------|-------------|
| **Prompt 利用率** | 动态过滤丢弃 ZVP 组 → 浪费 rollouts | 主动抑制 ZVP，保留所有 prompt 学习价值 |
| **重要性采样** | GRPO/GSPO 统一处理 token → 忽视不确定性差异 | ESPO 按熵分组，精细化信用分配 |
| **训练稳定性** | 训练-推理不一致 → 数值漂移 | Router Replay 对齐路由行为 |
| **奖励建模** | BT 模型 → 非单调、优势反转 | GenRM + 三类标签 → 单调可解释 |
| **系统效率** | Rollout 成瓶颈，串行执行 | FP8 + 长度调度 + 重叠计算 → 整体提速 1.66× |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **内部构建数据集**：
  - **Long-CoT SFT 数据**：来自 DeepSeek-R1-0528 的蒸馏数据，涵盖数学、代码、电商、对话、多语言理解等。
  - **E-commerce 数据**：覆盖东南亚七国语言（en, tw, id, my, pt, th, vi），任务包括商品推荐、售后问题、属性提取、标题优化等。
- **外部开源基准**：
  - **MMLU-Redux**, **GPQA-Diamond**（通用问答）
  - **AIME24/25**, **HMMT**, **Zebralogic**（数学推理）
  - **HumanEval**, **MBPP**（代码生成）
  - **BFCL** 系列（Agent 能力测试）

---

### **实验设置和评估指标**

#### **模型架构**
- **CompassMax-V3-Thinking**：百亿参数 MoE 模型，具备长上下文推理能力。
- 三阶段训练流程：
  1. **Cold-Start SFT**：基于 Long-CoT 数据微调。
  2. **Model Merge**：融合通用 Long-CoT 模型与领域优化模型（TIES 方法）。
  3. **Large-Scale RL**：两阶段 RL 微调（先 code/math/instruction，后 e-commerce/tool-use/general QA）。

#### **评估指标**
- **内部指标**：
  - 准确率（Accuracy）、Pass@1、Macro Average Score
  - 多语言平均得分（SEA Average）
- **外部指标**：
  - HumanEval / MBPP Pass@1
  - MMLU / GPQA 准确率（EM）
  - BFCL 系列任务得分
  - IFeval（格式遵循度）

#### **基线方法对比**
- **CompassMax-V3**（原版）
- **GPT-5-Thinking(medium)**
- **DeepSeek-R1**
- **Qwen3-235B-A22B (Thinking)**
- **Gemini-2.5-Pro**
- **GLM-4.5**

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **内部电商基准（Table 2）**
| 指标 | CompassMax-V3-Thinking | 最佳基线（GPT-5-Thinking） |
|------|------------------------|----------------------------|
| **Ecom QA 平均** | **96.14** | 96.82 |
| **After-Sales Issue** | **98.48** | 94.48 |
| **Product Recommendation** | **94.58** | 90.72 |
| **Overall Macro Average** | **85.79** | 80.89 |

> 在真实业务场景中表现卓越，尤其在推荐与售后服务等关键任务上领先。

#### ✅ **多语言评测（Table 3）**
| 语言 | en | id | th | vi | SEA Avg |
|------|----|----|----|----|---------|
| **CompassMax-V3-Thinking** | 88.07 | 87.89 | 86.04 | 84.75 | **86.41** |
| **GPT-5-Thinking(medium)** | 90.15 | 88.08 | 85.96 | 83.97 | 86.64 |
| **DeepSeek-R1** | 85.96 | 85.08 | 83.73 | 84.34 | 84.11 |

> 多语言性能均衡，接近 GPT-5，显著优于 DeepSeek-R1。

#### ✅ **通用能力评测（Table 4）**
| 领域 | CompassMax-V3-Thinking | CompassMax-V3 | GPT-5-Thinking |
|------|------------------------|---------------|----------------|
| **Reasoning** | 74.54 | 62.73 | 80.94 |
| **Code** | **76.83** | 67.94 | 85.08 |
| **Safety** | **88.86** | 72.95 | 86.10 |
| **Creative Generation** | **77.75** | 62.34 | 75.95 |
| **Overall Average** | **76.01** | 64.49 | 79.39 |

> 相比原始版本提升超 10 个百分点，尤其在安全性和创造性生成方面进步明显。

#### ✅ **开源 ARC 基准（Table 5）**
| 基准 | 指标 | 结果 |
|------|------|------|
| **HumanEval** | Pass@1 | **98.17** |
| **MBPP** | Pass@1 | 73.54 |
| **AIME24** | Pass@1 | **83.30** |
| **HMMT** | Pass@1 | 46.70 |
| **IFeval** | Prompt Strict | 85.40 |
| **BFCL_AST_NON_LIVE** | Score | 83.73 |
| **BFCL_MULTI_TURN_LIVE** | Score | **19.50**（远高于基线 12.00）|

> 编码能力接近 SOTA，多轮 Agent 表现显著改善。

---

### **消融实验结果（Figure 3 & Table 1）**

#### 🔹 **零方差率 vs. N 采样大小（Figure 3）**
- 增大 $N$ 可降低零方差率（从 0.8→0.6），同时提升每单位数据的优势增益。
- 但边际收益递减，最终选择最优 $N$ 平衡成本与效果。

#### 🔹 **系统优化累计加速（Table 1）**
| 方法 | 加速比 |
|------|--------|
| Baseline | 1.00× |
| + Multi-detokenization | 1.16× |
| + Reward Overlap | 1.17× |
| + FP8 Rollout | 1.52× |
| + Length-Based Load Balancing | **1.66×** |

> 四项系统优化叠加带来 **66% 端到端训练加速**，其中 FP8 贡献最大。

---

## 4. 关键结论和发现

### **主要发现**

1. **“每个提示都重要”是可实现且高效的训练范式**：通过 Multi-Stage ZVE 和 ESPO，可以显著减少无效 rollouts，提升样本利用率。
2. **MoE 模型的训练-推理一致性至关重要**：Router Replay 显著缓解了因浮点误差和实现差异导致的 RL 崩溃。
3. **奖励模型需具备推理能力**：GenRM 结合 CoT 与三类判断，能更稳定、可解释地指导策略学习。
4. **系统工程与算法必须协同设计**：仅靠算法改进无法突破百亿美元级 MoE 的 RL 瓶颈，FP8 + 调度 + 重叠计算缺一不可。
5. **模型合并（Model Merging）是快速获取领域能力的有效手段**：TIES 方法成功融合 Long-CoT 与电商专用模型，节省大量训练成本。

---

### **方法的局限性**

- **依赖高质量初始 SFT 模型**：Cold-Start 阶段若初始化不佳，仍可能影响后续 RL 收敛。
- **GenRM 训练需要大量人工标注偏好数据**：尽管引入了 tie 标注，但构建高质量三元偏好数据集成本较高。
- **FP8 依赖特定硬件（如 H100）**：在老旧 GPU 上难以部署。
- **Router Replay 增加存储开销**：需缓存每次 rollout 的路由决策。

---

### **未来工作方向**

- **自动化 prompt 选择机制**：动态识别高信息量 prompt，进一步减少冗余。
- **轻量化 GenRM 设计**：探索更小体积但仍具推理能力的奖励模型。
- **跨模型 Router 对齐技术**：扩展 Router Replay 至异构模型间的迁移学习。
- **全 FP8 端到端训练流水线**：将训练也纳入 FP8 精度，进一步压缩显存与能耗。
- **开放多领域 Reward System 架构**：推动通用型 Compass-Gym 开源，促进社区共建。

---

## 总结

本论文提出了一个面向**百亿参数 MoE 模型**的**高效、稳定、可扩展的 RL 框架**，通过 **“每个提示都重要”** 的设计理念，系统性解决了 ZVP、train-infer mismatch、advantage flipping、系统瓶颈四大难题。其成果 **CompassMax-V3-Thinking** 在内部电商、多语言、通用能力及开源基准上全面超越主流基线，验证了该框架在工业级复杂场景中的强大实用性。这不仅是算法上的进步，更是**算法-系统-应用三位一体的工程典范**，为下一代 Agentic 和领域自适应推理系统提供了重要蓝图。

</details>

---

### 3. [Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning](https://arxiv.org/abs/2512.07461)

**Authors**: Tong Wu, Yang Liu, Jun Bai, Zixia Jia, Shuyi Zhang, Ziyong Lin, Yanting Wang, Song-Chun Zhu, Zilong Zheng  
**Category**: cs.CL  
**Published**: 2025-12-09  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2512.07461v1  

#### Abstract
We introduce Native Parallel Reasoner (NPR), a teacher-free framework that enables Large Language Models (LLMs) to self-evolve genuine parallel reasoning capabilities. NPR transforms the model from sequential emulation to native parallel cognition through three key innovations: 1) a self-distilled p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning — 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前大型语言模型（LLMs）在复杂推理任务中仍主要依赖**顺序链式思维（Chain-of-Thought, CoT）**，存在以下三大瓶颈：

1. **算法与架构不兼容**：主流推理引擎（如SGLang）和强化学习（RL）算法难以支持原生并行分支生成与聚合，且对控制并行的特殊token梯度裁剪严重，阻碍结构学习。
2. **低效的手工并行化**：已有并行方法（如Multiverse）依赖独立采样，无法共享Key-Value（KV）缓存，导致计算冗余、延迟随分支数线性增长（O(N)），实用性差。
3. **依赖监督蒸馏**：多数并行推理框架（如Multiverse）依赖强教师模型生成的轨迹进行监督训练，限制学生模型探索新颖的、内在的并行策略，形成“智能天花板”。

### **提出了什么新方法或新思路**

作者提出 **Native Parallel Reasoner (NPR)**，一个无需外部教师、完全自演化的并行推理框架，通过三个阶段实现从顺序模拟到原生并行认知的转变：

#### ✅ 创新点一：自蒸馏渐进训练范式（Self-Distilled Progressive Training）
- **Stage 1（格式引导RL）**：使用DAPO算法结合格式奖励函数，让基础模型（如Qwen3-4B-Instruct）自发发现有效的并行结构，产出NPR-ZERO。
- **Stage 2（拒绝采样与并行预热）**：对NPR-ZERO生成的轨迹进行拒绝采样（仅保留格式正确且答案正确的样本），构建高质量自蒸馏数据集，并施加严格的并行注意力掩码和位置编码约束，训练出NPR-BETA。
- **Stage 3（原生并行RL）**：基于NPR-BETA，使用提出的**PAPO算法**进行并行感知的强化学习，直接优化执行图中的分支策略。

#### ✅ 创新点二：并行感知策略优化算法（Parallel-Aware Policy Optimization, PAPO）
- 在并行执行图中直接优化分支策略，而非模仿教师。
- 引入**批级优势归一化（batch-level advantage normalization）** 和 **保留特殊token梯度**，避免传统PPO因格式过滤导致方差崩溃和结构破坏。
- 采用严格On-Policy目标，提升训练稳定性与效率。

#### ✅ 创新点三：鲁棒的NPR Engine
- 改造SGLang推理后端，解决并行RL中的稳定性问题：
  - 修复KV缓存双释放与内存泄漏；
  - 实现分支感知的全局token预算控制；
  - 添加轻量级前置格式验证器防止非法状态；
  - 在`<step>`块内引入轻微重复惩罚以提升可读性。

---

### **相比现有方法的优势**

| 维度 | NPR | Multiverse等基线 |
|------|-----|------------------|
| **是否依赖教师模型** | ❌ 无监督自蒸馏 | ✅ 需要强教师模型 |
| **是否真正并行** | ✅ 100% genuine parallelism | ⚠️ 存在AR fallback（测试中30%+退化为顺序生成） |
| **KV缓存复用** | ✅ 支持跨分支共享 | ✅ 支持（继承自Multiverse设计） |
| **训练稳定性** | ✅ NPR Engine保障大规模稳定训练 | ⚠️ 原始引擎存在内存泄漏等问题 |
| **推理速度** | ✅ 最高达4.6×加速 | ✅ 有加速但低于NPR |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **训练数据**：基于ORZ数据集（57k问题-答案对），从中固定抽取8k样本用于所有训练阶段（Stage 1–3）。
- **评估基准**（共8个）：
  - 数学竞赛类：AIME25、AIME24、HMMT25、AMC23
  - 综合推理类：OlympiadBench、Minerva-Math、ZebraLogic、MATH500

### **实验设置和评估指标**

#### ✅ 模型基础
- 主干模型：Qwen3-4B-Instruct-2507 和 Qwen3-4B（非thinking模式）
- 对比模型规模：包括Qwen2.5-32B-Instruct、Multiverse-32B/4B等

#### ✅ 评估指标
- **avg@k**：k次采样中正确答案的比例期望值
  - 小规模数据集（AIME/HMMT等）报告 **avg@8**
  - 大规模/异构数据集（OlympiadBench等）报告 **avg@1**
- **推理效率**：Tokens Per Second (TPS) 与相对加速比（Speedup）
- **并行触发率（Parallel Rate）**：触发并行推理的样本占比

#### ✅ 基线方法对比
| 类别 | 基线模型 |
|------|--------|
| 开源顺序推理器 | Qwen2.5-32B-Instruct, Qwen3-4B(-Instruct) |
| 并行推理器 | Multiverse-32B, Multiverse-4B（本文复现） |
| 顺序变体 | SR-BETA（顺序SFT）、SR（顺序RL） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 模型 | AIME25 | AIME24 | HMMT25 | OlympiadBench | ZebraLogic | MATH500 | AVG |
|------|-------|-------|--------|---------------|------------|---------|-----|
| Qwen3-4B-Instruct | 47.4 | 60.0 | 31.0 | 64.0 | 80.2 | 93.4 | 63.7 |
| **NPR (Qwen3-4B-Instruct)** | **50.4** | **63.3** | **30.8** | **63.7** | **81.7** | **93.6** | **65.0** |
| Multiverse-32B | 45.8 | 53.8 | 20.8 | 48.0 | 47.1 | 91.8 | 52.5 |
| Multiverse-4B | 42.9 | 46.7 | 20.8 | 38.8 | 60.2 | 81.6 | 50.1 |

> ✅ **NPR-4B 超越了更强的 Multiverse-32B（+12.5%）和 Multiverse-4B（+14.9%）**

当训练于非thinking模式的Qwen3-4B时：
- **平均性能提升超过24.5%**
- AIME25从19.1 → 53.8（+34.7 pts）

---

### **与基线方法的对比结果**

| 方面 | 结果 |
|------|------|
| **准确性提升** | 在所有8个基准上均显著优于Multiverse系列及顺序RL方法 |
| **推理加速** | 相比AR解码最高达 **4.6× wall-clock speedup**（AIME25） |
| **并行一致性** | **100% genuine parallelism**，无任何AR fallback；而Multiverse-32B在多个任务上仅60%~70%触发并行 |
| **test-time scalability** | 使用best@8衡量，NPR显著提升最优解覆盖率，尤其在弱基线上增益更大（如AIME25从36.7→76.7） |

---

### **消融实验结果**

| 实验设置 | AIME25 性能 | 分析 |
|--------|-----------|------|
| 替换为Multiverse蒸馏数据（s1.1-8k） | ~42.9 | NPR自蒸馏数据带来+7.5分提升 |
| 使用顺序SFT（SR-BETA） | 37.1 | 并行SFT（NPR-BETA）提升至42.9（+5.8） |
| 使用顺序RL（SR） | 49.2 | 并行RL（NPR）进一步提升至50.4（+1.2） |
| 移除PAPO中的批级归一化 | 训练不稳定 | 验证了PAPO设计必要性 |

> 📌 自蒸馏数据 + 并行SFT + 并行RL 三者协同作用，缺一不可。

---

## 4. 关键结论和发现

### **主要发现**

1. **自蒸馏优于教师蒸馏**：  
   NPR通过自演化生成的并行轨迹质量更高、多样性更丰富，平均比Multiverse的教师轨迹高出**10.1分**，证明“向自身分布学习”更具潜力。

2. **并行策略是搜索机制的升级**：  
   并行SFT和RL提供了比单路径rollout更强大的搜索能力，尤其在复杂逻辑任务中表现突出（如ZebraLogic提升15.9分）。

3. **100% genuine parallelism 可实现**：  
   NPR在所有测试案例中均保持真正的并行执行，无隐藏的AR回退，建立了新的可靠性标准。

4. **推理效率与效果双赢**：  
   并行不仅提速（最高4.6×），还通过多角度探索与交叉验证提高了解的可靠性。

5. **认知策略动态适应问题类型**：
   - 创意任务 → 宽泛探索多种策略
   - 逻辑任务 → 严谨交叉验证与自我修正

---

### **方法的局限性**

1. **对基础模型要求较高**：尝试在Qwen2.5系列或Base LLM上复现失败，因其指令跟随和推理能力不足，Stage 1自蒸馏难以启动。
2. **输出模板依赖特殊token**：虽然简化了Multiverse的设计，但仍需维护`<plan>`, `<step>`, `<takeaway>`等结构标签。
3. **训练成本高**：三阶段流程需要大量计算资源支持大规模并行RL训练。

---

### **未来工作方向**

1. **扩展至更多模态与任务**：将NPR应用于代码生成、规划、多跳问答等agent任务。
2. **降低对强基础模型的依赖**：研究如何在中小模型上实现有效冷启动。
3. **自动化并行结构发现**：减少人工设计schema，让模型自主决定何时并行、如何分解。
4. **与Test-Time Scaling结合**：探索NPR在运行时动态扩展并行度的能力。

---

## 总结

✅ **NPR 是首个完全无需教师、通过自蒸馏+并行RL实现原生并行推理的框架**，在准确性、效率、真实并行性方面全面超越现有方法。它不仅提升了性能，更重要的是展示了LLM可以**自我演化出高效、可靠、可扩展的并行认知能力**，为下一代Agentic AI提供了重要路径。

</details>

---

### 4. [Bandwidth-Aware Network Topology Optimization for Decentralized Learning](https://arxiv.org/abs/2512.07536)

**Authors**: Yipeng Shen, Zehan Zhu, Yan Huang, Changzhi Yan, Cheng Zhuo, Jinming Xu  
**Category**: cs.DC  
**Published**: 2025-12-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2512.07536v1  

#### Abstract
Network topology is critical for efficient parameter synchronization in distributed learning over networks. However, most existing studies do not account for bandwidth limitations in network topology design. In this paper, we propose a bandwidth-aware network topology optimization framework to maxim...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Bandwidth-Aware Network Topology Optimization for Decentralized Learning*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**去中心化学习（Decentralized Learning）**中的一个关键瓶颈——**参数同步效率低**，提出了一种新的网络拓扑优化框架。传统研究在设计参数同步拓扑（parameter synchronization topology）时，往往忽略实际系统中存在的**带宽异构性（heterogeneous bandwidth）**，导致理论最优的拓扑在真实场景中性能下降。

具体而言，现有方法存在以下问题：
- 多数拓扑设计基于直觉（如环形、网格、指数拓扑），无法平衡通信开销与共识速度（consensus speed）。
- 忽略节点间链路带宽差异（如服务器内PCIe链路 vs. 跨服务器交换机端口）。
- 权重分配策略不优，且受限于简化假设（如边权重相等）。

---

### 提出的新方法与新思路
作者提出了 **BA-Topo（Bandwidth-Aware Topology）**，一种**带宽感知的网络拓扑优化框架**，其核心思想是将拓扑结构设计建模为一个受约束的优化问题，并引入多种现实硬件限制。

#### 主要创新点包括：

1. **统一建模带宽约束**
   - 针对三种典型异构带宽场景（node-level, intra-server link, inter-server switch port），提出通过**边缘容量向量 `e` 和关联矩阵 `M`** 将物理带宽限制编码进优化模型。
   - 引入 **Algorithm 1（Bandwidth-Aware Edge-Capacity Allocation）**，动态分配每节点可承载的最大边数，以最大化最小边带宽（unit bandwidth），从而提升通信效率。

2. **优化目标：最大化共识速率**
   - 以渐近收敛因子 $ r_{\text{asym}}(W) = \max\{|\lambda_2(W)|, |\lambda_n(W)|\} $ 作为共识速度度量，目标是最小化该值。
   - 在同构和异构带宽下均构建了相应的优化问题（见 Eq. (9) 和 Eq. (10)）。

3. **转化为混合整数半定规划（Mixed-Integer SDP）**
   - 利用拉普拉斯矩阵变换和引理（Lemma 1）将非凸特征值优化问题转化为带有线性矩阵不等式（LMI）约束的形式。
   - 引入二元变量 `z` 表示逻辑边是否激活，实现对拓扑结构的选择。

4. **高效求解算法：ADMM + 共轭梯度法**
   - 设计基于 **ADMM（Alternating Direction Method of Multipliers）** 的迭代求解器（Algorithm 2）。
   - 在子步骤中采用 **Bi-CGSTAB（双共轭梯度稳定法）** 求解大规模线性方程组，显著提高可扩展性。
   - 使用 **预条件ILU分解** 和 **CSC稀疏存储格式** 加速计算，支持数百节点规模。

---

### 相比现有方法的优势
| 方面 | BA-Topo优势 |
|------|-------------|
| **适用性** | 支持同构与异构带宽场景，更贴近真实系统 |
| **解空间完整性** | 不做边权相等等简化假设，在完整解空间中搜索最优解（vs. Xiao et al. [22]仅考虑子集） |
| **性能表现** | 显著优于经典拓扑（ring, grid, torus）及SOTA拓扑（exponential, EquiTopo） |
| **可扩展性** | 借助Bi-CGSTAB和稀疏优化，适用于大规模网络 |

---

## 2. 核心实验方法和设置

### 数据集
- **CIFAR-10** 和 **CIFAR-100**：用于评估去中心化训练效果。
- 所有节点本地数据为随机均匀采样，模拟IID数据分布。

### 模型与训练配置
- 模型：**ResNet-18**
- 算法：**DSGD（Decentralized SGD）**
- Batch Size：32（per node）
- Learning Rate：0.05，Momentum：0.9，Weight Decay：0.0001
- Epochs：100

### 硬件平台
- CPU：2 × Intel Xeon Gold 6226R
- GPU：8 × NVIDIA GeForce RTX 2080 Ti（11GB显存）
- 通信后端：PyTorch 2.0.0 + Gloo

### 评估指标
1. **共识速度（Consensus Speed）**
   - 渐近收敛因子 $ r_{\text{asym}} $
   - 达到共识误差 $ \|x_k - \bar{x}\| < 10^{-4} $ 所需时间
2. **训练效率**
   - 达到目标测试准确率所需时间（如CIFAR-10达到84%，CIFAR-100达到62%）
   - 训练加速比（Speedup）

### 基线方法对比
- Ring
- 2D Grid
- 2D Torus
- Exponential Topology [16]
- EquiTopo（U-EquiStatic variant）[19]

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table II）

| 场景 | 数据集 | BA-Topo 最佳配置 | 达标时间（秒） | 最优基线时间 | 加速比 |
|------|--------|------------------|----------------|---------------|--------|
| 同构带宽 | CIFAR-10 | r=32 | **110.1s** | 134.6s (Exponential) | **1.22×** |
| 同构带宽 | CIFAR-100 | r=32 | **127.6s** | 151.8s (Exponential) | **1.19×** |
| 节点级异构 | CIFAR-10 | r=32 | **181.6s** | 287.4s (Exponential) | **1.58×** |
| 节点级异构 | CIFAR-100 | r=48 | **194.7s** | 324.3s (Exponential) | **1.66×** |
| 服务器内链路异构 | CIFAR-10 | r=8 | **266.8s** | 484.1s (Ring) | **1.81×** |
| 服务器内链路异构 | CIFAR-100 | r=8 | **261.8s** | 350.7s (Ring) | **1.34×** |
| 跨服务器端口异构 | CIFAR-10 | r=48 | **86.3s** | 134.6s (Exponential) | **1.56×** |
| 跨服务器端口异构 | CIFAR-100 | r=48 | **117.3s** | 151.8s (Exponential) | **1.21×** |

> 注：原文摘要称“speedups of more than **1.11× and 1.21×** for homogeneous and heterogeneous”，此处实验数据显示最高可达 **1.8×**。

---

### 与基线方法的对比结果
- **共识速度方面（Fig. 1–6）**：
  - 在所有带宽场景下，BA-Topo 的共识误差下降最快。
  - 即使边数较少（如 r=16），BA-Topo 仍优于 exponential（r=32）。
  - 在异构场景中，exponential 因过度集中映射导致某些链路带宽极低（如 SYS link 上10条边 → 0.976 GB/s），严重拖慢迭代速度。

- **训练效率方面（Fig. 7–10）**：
  - BA-Topo 在相同时间内取得更高测试精度。
  - 特别是在异构环境下，由于合理利用高带宽路径并避免拥塞，表现出更强鲁棒性。

### 消融实验与可扩展性分析（Table I）
- **可扩展性测试（n=4~128）**：
  - 随着节点数量增加，BA-Topo 的渐近收敛因子始终低于 exponential 和 U-EquiStatic。
  - 在 n=128 时，BA-Topo 收敛时间为 1127ms，远低于 exponential 的 1157ms 和 U-EquiStatic 的 1242ms。
- **初始化策略**：
  - 使用 **simulated annealing** 构造初始拓扑，确保较短平均最短路径长度（ASPL），有助于避免陷入局部最优。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **带宽感知设计至关重要**：忽视实际带宽限制会导致理论高性能拓扑在实践中失效（如 exponential 在异构环境中性能骤降）。
2. ✅ **联合优化拓扑结构与边权重能显著提升性能**：BA-Topo 在完整解空间中搜索，避免了人为设定权重带来的次优性。
3. ✅ **BA-Topo 在各类带宽场景下均表现最优**：无论是同构还是异构环境，都能有效平衡通信开销与共识速度。
4. ✅ **所提 ADMM + Bi-CGSTAB 框架具备良好可扩展性**：可在百节点级别高效运行，适合实际部署。

---

### 方法的局限性
- **静态拓扑假设**：当前方法生成的是固定拓扑，未考虑动态变化的带宽或故障恢复能力。
- **计算复杂度较高**：虽然使用了加速技术，但拓扑优化本身仍需离线进行，不适合实时调整。
- **依赖良好初始化**：优化过程对初始拓扑敏感，需借助启发式方法（如模拟退火）获得高质量起点。

---

### 未来工作方向
- 探索 **动态带宽场景下的时变拓扑优化方案**。
- 将方法扩展至 **非独立同分布（Non-IID）数据场景**，结合数据异构性进行联合优化。
- 研究 **轻量化在线拓扑自适应机制**，实现运行时自动调整。

--- 

> **总结一句话**：本文提出的 **BA-Topo** 是首个系统性地将**多层级带宽限制**纳入去中心化学习拓扑设计的框架，通过**混合整数SDP建模 + ADMM高效求解**，实现了在真实异构环境下的显著性能提升，推动了去中心化训练系统的实用化进程。

</details>

---

### 5. [Revolutionizing Mixed Precision Quantization: Towards Training-free Automatic Proxy Discovery via Large Language Models](https://arxiv.org/abs/2512.07419)

**Authors**: Haidong Kang, Jun Du, Lihong Lin  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2512.07419v1  

#### Abstract
Mixed-Precision Quantization (MPQ) liberates the Deep Neural Networks (DNNs) from the Out-Of-Memory (OOM) bottleneck, which garnered increasing research attention. However, conventional methods either searched from costly differentiable optimization, which is neither efficient nor flexible, or learn...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Revolutionizing Mixed Precision Quantization: Towards Training-free Automatic Proxy Discovery via Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **Mixed-Precision Quantization (MPQ)** 方法面临两大瓶颈：
- **依赖专家设计的代理（proxy）**：如 HAWQ、OMPQ 等训练-free 方法依赖人工设计的启发式规则（如 Hessian 分析、权重/激活统计），需要大量领域知识和试错成本；
- **高校准开销与慢收敛**：现有方法通常需要数千样本和数十次迭代进行校准优化（例如 HAWQ-V2 需要 8,192 样本和 50 次迭代），效率低下。

这些问题限制了 MPQ 在新架构或硬件上的快速适配能力。

### 提出的新方法：TAP
本文提出 **TAP (Training-free Automatic Proxy)** ——一种基于 **Large Language Models (LLMs)** 的零训练自动代理发现框架，核心思想是：
- 利用 LLM 自动生成适用于 MPQ 的量化敏感度代理函数；
- 引入 **Direct Preference Optimization (DPO)** 构建反馈闭环，通过轻量级强化学习优化提示（prompt），提升 LLM 推理质量；
- 实现无需人工干预、无需模型训练、低校准成本的高性能混合精度量化。

### 相比现有方法的优势
| 维度 | 传统方法（HAWQ, OMPQ） | TAP |
|------|------------------------|-----|
| 是否需专家参与 | 是（手动设计 proxy） | 否（LLM 自动生成） |
| 是否需训练 | 否（但需优化） | 完全无需训练 |
| 校准数据需求 | 大（64~8192 samples） | 极小（仅需 16 samples） |
| 收敛速度 | 慢（2500+ 更新 / 50+ 迭代） | 快（5 次进化即可） |
| 泛化性 | 差（任务特定） | 强（跨模型、跨数据集迁移） |
| 搜索时间 | 数小时 GPU 时间 | ~10⁻⁵ GPU 小时（微秒级） |

> ✅ **TAP 实现了“推理即搜索”范式转变**：将复杂的量化策略搜索转化为 LLM 的自然语言推理过程。

---

## 2. 核心实验方法和设置

### 数据集
- 主要基准：**ImageNet-1k**, **CIFAR-10**, **PASCAL VOC 2007**, **MS COCO 2017**
- 使用 **CIFAR-10** 搜索最优 proxy，直接迁移到 **ImageNet-1k** 上验证泛化性（zero-shot transfer）
- 测试模型覆盖 CNN 与 Transformer 架构：
  - CNN: ResNet-18/50, MobileNetV2
  - Transformer: ViT-B, DeiT-B, Swin-B

### 实验设置
- 硬件平台：NVIDIA 3090 GPU（24GB）
- 量化方式：混合精度（W/A 可变比特宽度）
- 压缩目标：参数压缩率 ≥80%
- 校准样本数：TAP 仅使用 **16 个无标签样本**
- 进化代数：最多 5 代（T_max = 5）
- LLM 背骨测试：Deepseek-chat, Qwen3-max, Grok 3

### 评估指标
| 指标 | 描述 |
|------|------|
| **Top-1 Accuracy (%)** | 量化后模型在 ImageNet 等数据集上的分类准确率 |
| **#Params (M)** | 模型参数量（越低越好） |
| **Compression Ratio (%)** | 参数压缩比例 |
| **Cost (GPU hours)** | 搜索/校准所需计算资源 |
| **Spearman Correlation** | proxy 敏感度评分与真实误差的相关性 |
| **Runtime (s)** | proxy 生成与 bit 分配延迟 |

### 基线方法对比
涵盖多种主流 MPQ 方法：
- **Fixed-precision**: PACT, LSQ
- **RL-based**: HAQ
- **Sensitivity-based**: HAWQ, HAWQ-V3
- **Differentiable**: DNAS, EdMIPS, FracBits-SAT
- **Training-free**: OMPQ, EMQ
- **Transformer-specific PTQ**: FQ-ViT, APQ-ViT, OMSE

---

## 3. 主要实验结果和性能指标

### 3.1 Quantization-Aware Training 结果（表3）

| 方法 | ResNet-18 Top-1 (%) | ResNet-50 Top-1 (%) | Cost (GPU h) |
|------|---------------------|---------------------|--------------|
| Full-precision | 73.09 | 77.72 | - |
| OMPQ | 72.08 | 76.28 | 0.45 |
| EMQ | 72.28 | 76.70 | 0.51 |
| **TAP-C** | **72.93** | **76.81** | **9.17×10⁻⁶** |

✅ **优势总结**：
- 在 ResNet-18 上超越所有基线，精度提升 **+0.65%**（vs EMQ）
- 在 ResNet-50 上达到 SOTA，且搜索成本降低 **>50,000 倍**

> 🚀 TAP-C 是首个实现“近零成本搜索 + 超高精度”的训练-free MPQ 框架。

---

### 3.2 Post-Training Quantization 结果（表4 & 表5）

#### ResNet-18（Table 4）
| 方法 | Top-1 (%) | Data Used |
|------|-----------|----------|
| OMPQ | 69.41 | 64 |
| EMQ | 69.92 | 64 |
| **TAP-C** | **70.26** | **16** |

#### MobileNetV2（Table 5）
| 方法 | Top-1 (%) | Data Used |
|------|-----------|----------|
| OMPQ | 69.62 | 32 |
| EMQ | 70.75 | 64 |
| **TAP-C** | **71.81** | **16** |

✅ **关键发现**：
- 仅用 **16 个样本** 即可完成校准，远少于 OMPQ/EMQ；
- 精度显著领先，尤其在轻量级网络上表现更优；
- 展现出极强的数据效率与精度平衡能力。

---

### 3.3 Transformer 泛化能力（Table 6）

| Model | Baseline (%) | TAP-C (%) | Comp Ratio (%) |
|-------|---------------|------------|----------------|
| ViT-B | 84.54 | **83.56** | 82% |
| DeiT-B | 84.54 | **81.24** | 82% |
| Swin-B | 85.27 | **83.79** | 82% |

✅ **突破性进展**：
- 首次证明 LLM-driven proxy 可无缝扩展至 **Vision Transformers**；
- 在高达 82% 压缩率下仍保持接近全精度性能；
- 显著优于专门针对 ViT 设计的 PTQ 方法（如 APQ-ViT, OMSE）。

---

### 3.4 效率评估（Table 7）

| 步骤 | 平均耗时 (s) |
|------|-------------|
| Proxy Generation | 0.0133 |
| Bit Allocation | 0.0645 |
| **Total** | **< 0.1 s** |

✅ **实时性优势**：整个量化流程可在 **100ms 内完成**，适合边缘设备部署。

---

### 3.5 消融实验（Ablation Studies）

#### (1) 超参 α 影响（Table 8）
- α 控制 `sensitivity score` 与 `quantization accuracy` 的权衡；
- 实验表明 TAP 对 α 不敏感，在宽范围内性能稳定；
- 最佳值 α ≈ 0.1（兼顾敏感度预测与最终精度）。

#### (2) 不同 LLM 背骨的影响（Table 9）
| LLM | Average Top-1 (%) |
|------|--------------------|
| Deepseek-chat | 71.44 |
| Qwen3-max | 71.35 |
| Grok 3 | 71.01 |

✅ **结论**：TAP 在不同 LLM 上表现高度一致，说明其对生成模型具有鲁棒性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 可以有效替代人类专家设计 MPQ proxy**：通过自然语言推理自动生成高质量敏感度度量函数；
2. ✅ **DPO 构建正向反馈闭环**：使 LLM 在迭代中持续改进 prompt 与 proxy 质量，形成“推理→评估→优化”循环；
3. ✅ **TAP 实现真正的 training-free 与 ultra-low calibration**：仅需 16 样本、5 次迭代、微秒级搜索时间；
4. ✅ **强大的跨架构与跨数据集泛化能力**：在 CNN 和 Transformer 上均取得 SOTA 性能；
5. ✅ **首次实现 LLM-driven MPQ 框架**：为自动化模型压缩开辟全新路径。

---

### 方法的局限性
1. **依赖高质量 LLM 输出**：若 LLM 编码错误或逻辑混乱，可能导致无效 proxy；
2. **代码执行风险**：生成的 Python 代码需安全沙箱运行，防止注入攻击；
3. **初始 prompt 设计仍需一定经验**：虽然无需专家设计 proxy，但 prompt 工程影响收敛速度；
4. **目前仅用于 sensitivity-based MPQ**：尚未拓展到其他量化范式（如稀疏化联合优化）。

---

### 未来工作方向
1. 🔮 扩展至 **Sparse + Quantized Joint Optimization**；
2. 🔄 探索 **多模态 LLM**（结合图结构理解神经网络）；
3. ⚙️ 开发 **Auto-prompting for DPO**，进一步减少人工干预；
4. 📱 推动 TAP 在 **MCU/NPU 等极端资源受限设备** 上的实际部署；
5. 🤝 构建 **LLM + NAS + Quantization** 统一自动化框架。

---

## 总结
> **TAP 是一次范式革新**：它将混合精度量化从“专家驱动 + 数据密集”的旧模式，转变为“LLM 驱动 + 推理即搜索”的新模式。不仅实现了前所未有的高效与高性能，更为 AutoML 与 LLM for Systems 的交叉研究提供了典范案例。

</details>

---

### 6. [Communication-Efficient Serving for Video Diffusion Models with Latent Parallelism](https://arxiv.org/abs/2512.07350)

**Authors**: Zhiyuan Wu, Shuai Wang, Li Chen, Kaihui Gao, Dan Li, Yanyu Ren, Qiming Zhang, Yong Wang  
**Category**: cs.DC  
**Published**: 2025-12-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.07350v1  

#### Abstract
Video diffusion models (VDMs) perform attention computation over the 3D spatio-temporal domain. Compared to large language models (LLMs) processing 1D sequences, their memory consumption scales cubically, necessitating parallel serving across multiple GPUs. Traditional parallelism strategies partiti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Communication-Efficient Serving for Video Diffusion Models with Latent Parallelism

---

## 1. 论文的主要贡献和创新点

### 解决的问题
视频扩散模型（**Video Diffusion Models, VDMs**）在生成高质量、长时序一致的视频方面取得了显著进展（如 Sora、Veo、WAN），但其推理过程面临严重的**GPU内存瓶颈**。由于 VDMs 在三维时空域（时间、高度、宽度）上进行注意力计算，其内存消耗随序列长度呈**立方级增长**，远超处理一维文本序列的大型语言模型（LLMs）。这导致单个 GPU 难以承载整个模型，必须依赖多 GPU 并行服务。

然而，传统的并行策略（如 **Data Parallelism (DP)**、**Tensor Parallelism (TP)**、**Pipeline Parallelism (PP)**）在应用于 VDM 时，会因频繁传输高维中间激活张量（activations）而产生严重的**通信开销瓶颈**，尤其是在带宽受限的 PCIe 互联环境中，通信成为系统性能的决定性瓶颈。

### 提出的新方法：Latent Parallelism (LP)
本文提出了 **Latent Parallelism (LP)**，这是首个专为 VDM 推理服务设计的并行化策略。其核心思想是将并行化的对象从“模型”转移到“请求的潜在空间（latent tensor）”，即对每个请求的输入潜变量进行分区，而非对模型本身进行切分。

#### 核心机制：
- **动态旋转分区（Dynamic Rotating Partition）**：在不同的去噪时间步（diffusion timestep），交替沿时间、高度、宽度三个维度对潜变量进行分区。这种旋转确保了每个子区域在多个周期内都能获得完整的时空上下文，从而维持全局一致性。
- **补丁对齐的重叠分区（Patch-Aligned Overlapping Partition）**：分区边界与 VDM 内部的视觉 patch 边界对齐，并引入可控的重叠区域，防止因硬切割导致的特征断裂和边界伪影。
- **位置感知的潜变量重建（Position-Aware Latent Reconstruction）**：在合并各 GPU 上并行去噪的结果时，根据每个位置到其所在分区核心区域的距离，自适应地加权融合重叠区域的预测结果，实现平滑拼接。

### 相比现有方法的优势
- **通信效率极高**：传统方法（NMP, PP）需传输高维中间激活（`SH`），而 LP 只需传输紧凑的潜变量（`Sz`），其大小通常仅为 `SH` 的 ~5%。理论分析表明，通信开销可降低两个数量级。
- **非侵入式插件范式**：LP 可无缝集成到现有的并行策略（如 TP、PP）之上，形成混合并行框架，进一步提升扩展性。
- **保持生成质量**：通过旋转分区和位置感知重建，确保了全局信息流动，避免了局部不一致，实验证明其生成质量与集中式（Centralized）推理相当。

---

## 2. 核心实验方法和设置

### 数据集
在三个广泛使用的视频生成基准上进行评估：
- **EvalCrafter**：用于评估大规模视频生成模型。
- **T2V-CompBench**：专注于组合性文本到视频生成任务。
- **VBench**：综合性视频生成模型评测套件。

### 实验设置
- **硬件平台**：4 台 NVIDIA RTX A6000 GPU，通过 PCIe 互连。
- **模型配置**：采用 **WAN2.1-1.3B** 作为基础 VDM，包含 30 个 DiT Blocks，T5 文本编码器和预训练 VAE 解码器。
- **视频参数**：分辨率 480p，帧率 16 FPS，测试 49 帧（3秒）和 81 帧（5秒）两种长度。
- **去噪步数**：60 步。

### 评估指标
- **通信开销（Communication Overhead）**：以 MB 为单位，衡量推理过程中跨 GPU 传输的数据总量。
- **生成质量（Generation Quality）**：使用 **VBench** 框架下的五个子指标：
  - **Subject Consistency (SC)**：主体一致性
  - **Background Consistency (BC)**：背景一致性
  - **Temporal Flickering (TF)**：时间闪烁
  - **Motion Smoothness (MS)**：运动平滑度
  - **Imaging Quality (IQ)**：成像质量

### 基线方法对比
- **Naive Model Parallelism (NMP)**：将 DiT Blocks 均匀分配到不同 GPU。
- **Pipeline Parallelism (PP)**：基于 NMP 分区，利用 CFG 的条件/无条件通路构建微批次流水线。
- **Hybrid Parallelism (HP)**：结合 FSDP 和 xDiT 的开源推理框架，代表当前先进水平。
- **Centralized**：单 GPU 集中式推理，作为质量上限参考。
- **VideoCrafter**：作为外部模型的质量参考。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 通信开销对比（总 MB）
| 方法 | 49帧 (3s) | 81帧 (5s) |
|------|----------|----------|
| **NMP** | 57,950.17 | 93,050.17 |
| **PP** | 57,590.16 | 92,690.16 |
| **HP** | 4,758.08 | 7,686.12 |
| **LP (r=1.0)** | 1,811.88 | 2,912.81 |
| **LP (r=0.5)** | **1,354.34** | **2,191.29** |

- **LP 将通信开销降低了高达 97%**（相比 NMP/PP），即使与优化过的 HP 相比，也减少了约 72%。
- 通信开销的增长随视频长度呈线性趋势，而 HP 呈近似二次增长，表明 LP 具有更优的可扩展性。

#### 生成质量对比（VBench 平均得分）
- LP (r=1.0) 与 **Centralized、NMP、PP、HP** 的平均得分差异**不超过 0.6%**，在所有五个维度上表现几乎一致。
- 所有并行方法均显著优于 **VideoCrafter**，尤其是在 IQ、SC 和 MS 指标上。
- 定性可视化显示，LP 生成的视频在主体细节、纹理清晰度、色彩准确性和时间连贯性方面与 Centralized 结果难以区分，无可见边界伪影。

### 消融实验结果
- **重叠比例 (Overlap Ratio `r`)**：
  - `r` 从 0.1 增至 1.0，通信开销从 ~900MB 增至 ~1800MB，但仍远低于 HP。
  - 生成质量随 `r` 增加而提升，在 `r=0.5` 后趋于饱和，表明 `r=0.5` 是通信与质量的最佳平衡点。
- **GPU 数量影响**：LP 在 2 到 8 个 GPU 上均表现出稳定且高质量的性能，验证了其良好的可扩展性。
- **视频时长影响**：随着视频从 3 秒增至 10 秒，HP 通信开销从 ~5GB 增至 ~15GB，而 LP 仅增加约 4GB，优势更加明显。
- **分区策略消融**：与仅在时间维度分区的基线（w/o LP）相比，LP 在所有指标上均显著更优，证明了**动态旋转分区**的有效性。

#### 端到端延迟
| 方法 | 延迟 (秒) |
|------|--------|
| **NMP** | 239.33 |
| **LP (r=1.0)** | 220.69 |
| **LP (r=0.5)** | **195.27** |

- 即使考虑了重叠分区带来的额外计算，LP 仍能提供更低的推理延迟，直接得益于通信瓶颈的缓解。

---

## 4. 关键结论和发现

### 主要发现
1. **通信瓶颈是 VDM 服务的核心挑战**：传统并行策略因传输高维激活而导致通信开销巨大，严重制约了实际部署。
2. **潜变量并行（Latent Parallelism）是高效解决方案**：通过在潜空间而非模型层面进行并行化，LP 成功将通信负载从高维激活 `SH` 转移到紧凑潜变量 `Sz`，实现了高达 97% 的通信开销削减。
3. **生成质量得以保持**：提出的动态旋转分区、补丁对齐和位置感知重建机制，有效保证了全局信息流动和平滑拼接，使得 LP 的生成质量与集中式推理基本持平。
4. **LP 是通用的插件范式**：可与现有并行策略（TP、PP）结合，构建层次化混合并行框架，适用于大规模 GPU 集群。

### 方法的局限性
- **依赖于潜变量的紧凑性**：如果未来 VDM 设计导致潜变量尺寸增大，LP 的通信优势可能会减弱。
- **分区边界可能引入轻微伪影**：尽管通过重叠和加权重建进行了优化，但在极端情况下仍可能存在肉眼难辨的边界效应。
- **对 patch 结构敏感**：补丁对齐分区要求对 VDM 内部的 patch 大小和结构有先验知识。

### 未来工作方向
- **探索更智能的动态分区策略**：根据视频内容（如运动强度、复杂度）自适应调整分区方式和重叠比例。
- **与缓存技术（如 TeaCache、ProfilingDiT）深度结合**：进一步减少冗余计算，实现全栈优化。
- **在更大规模集群上的部署验证**：研究 LP 在数十甚至上百 GPU 场景下的性能和可扩展性。
- **支持更多类型的生成模型**：探索 LP 是否可推广至其他具有高维中间表示的生成模型（如 3D 生成、分子生成等）。

</details>

---

### 7. [Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation](https://arxiv.org/abs/2512.06690)

**Authors**: Chengbing Wang, Yang Zhang, Wenjie Wang, Xiaoyan Zhao, Fuli Feng, Xiangnan He, Tat-Seng Chua  
**Category**: cs.CL  
**Published**: 2025-12-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.06690v1  

#### Abstract
Preference alignment has enabled large language models (LLMs) to better reflect human expectations, but current methods mostly optimize for population-level preferences, overlooking individual users. Personalization is essential, yet early approaches-such as prompt customization or fine-tuning-strug...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的大型语言模型（LLM）偏好对齐（preference alignment）方法大多关注群体层面的偏好，忽视了**个体用户的个性化需求**。虽然已有方法如提示定制（prompt customization）或微调（fine-tuning）尝试实现个性化，但它们难以推理用户的**隐式偏好**（implicit preferences），导致在真实场景中效果有限。

此外，新兴的“**think-then-generate**”范式虽能通过先推理再生成的方式捕捉隐式偏好，但在长文本生成任务中面临两大挑战：
- **静态一次性推理**（static one-shot reasoning）需覆盖整个响应的所有信息，学习难度大；
- 无法适应长文本创作过程中用户想法的动态演变（evolving content）。

### **提出的新方法：FlyThinker**
本文提出了 **FlyThinker**，一种高效的“**think-while-generating**”框架，用于个性化长文本生成。

#### **核心创新点**
1. **并行推理与生成架构**：
   - 引入一个独立的 **Reasoner** 模型，实时生成**token-level 的潜在推理**（latent reasoning）；
   - 推理过程与生成过程**并行进行**，而非串行依赖，显著提升效率。

2. **保持训练并行性**：
   - Reasoner 的输出仅依赖于已生成的响应 token，而不依赖其自身先前的推理输出；
   - 这种设计使得所有位置的推理 token 可以在训练时通过一次前向传播并行生成，**保留了标准 LLM 训练的并行性**。

3. **动态上下文感知推理**：
   - 每个生成步骤都由最新的推理结果引导，能够持续更新对用户偏好的理解，适应内容演化。

### **相比现有方法的优势**
| 方法 | 缺陷 | FlyThinker 的优势 |
|------|------|------------------|
| Prompt/Fine-tuning | 难以建模隐式偏好 | 显式引入推理机制 |
| Think-then-generate | 静态推理，不适应长文本演化 | 动态、逐步推理 |
| CoT / Coconut | 推理序列自回归生成，训练/推理慢 | 并行生成推理 token，高效 |

---

## **2. 核心实验方法和设置**

### **数据集**
在 **LONG-LAMP** 基准的三个长文本个性化生成任务上进行评估：
- **Product Review**：撰写产品评论
- **Abstract Generation**：生成学术摘要
- **Topic Writing**：主题写作（如影评）
> 所有任务均基于用户历史行为进行个性化建模。

### **主干模型**
- 主要使用 **Qwen2.5-3B-Instruct**
- 补充实验使用 Qwen2.5-7B-Instruct 和 Gemma-7B-it

### **评估指标**
- 自动指标：**ROUGE-1, ROUGE-L, BLEU, METEOR, BERTScore**
- 人工评估：使用 **GPT-4o** 作为评判器进行盲测打分
- 效率指标：训练时间/epoch、推理延迟（latency）

### **基线方法对比**
分为两类：

#### **无需微调的方法（Tuning-free）**
- **Non-pers**：无个性化
- **RAG**：检索增强生成
- **CoS**：上下文 steering

#### **需要微调的方法（Tuning-based）**
- **SFT**：监督微调
- **LLM-TRSR**：基于摘要的微调
- **NextQuill**：因果偏好建模
- **CoT**：链式思维推理（think-then-generate）
- **Coconut**：潜在推理（latent reasoning）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| 方法 | Product Review (BLEU) | Abstract Gen (BLEU) | Topic Writing (BLEU) |
|------|------------------------|-----------------------|------------------------|
| SFT | 3.91 | 5.82 | 3.89 |
| CoT | 3.37 | 5.85 | 3.00 |
| Coconut | 3.32 | 5.24 | 3.07 |
| **FlyThinker** | **4.36** (+11.5%) | **6.34** (+9.0%) | **4.06** (+4.4%) |

> 在所有任务上，FlyThinker 均优于最强基线 SFT 和其他推理方法。

#### **自动指标总结**
- **平均相对提升**（vs Non-pers）达 **55.24%**（Product Review）、**15.85%**（Abstract）、**71.03%**（Topic Writing）
- 在 **ROUGE-L** 上也显著领先，表明生成内容更贴近参考文本的结构与语义。

---

### **与基线方法的对比结果**

#### ✅ **RQ1: 个性化长文本生成性能**
- FlyThinker 在所有任务和指标上**一致超越**各类基线；
- 尤其在 **Abstract Generation** 上表现突出，说明其在长序列连贯性方面更强。

#### ✅ **RQ2: 训练与推理效率**
- **训练效率**：接近 SFT，远快于 CoT 和 Coconut（因其自回归推理）；
- **推理效率**：几乎与 SFT 相当，**无明显延迟增加**，得益于并行设计。

> 图 3 显示：FlyThinker 的推理延迟仅为 CoT 的 ~1/3。

#### ✅ **RQ3: 不同生成阶段的质量变化（Position-sensitive Evaluation）**
- 所有基线方法在生成后期（如 200–300 token）出现明显的质量下降（context drift）；
- **FlyThinker 能有效缓解这一问题**，在后期仍保持高个性化质量。

> 原因：每一步都有新的推理信号注入，持续刷新上下文感知。

#### ✅ **RQ4: 消融实验（Ablation Study）**

##### （1）Reasoner 规模影响（图 5）
| Reasoner Size | 性能趋势 |
|---------------|---------|
| 3B → 1.5B | 性能基本不变，性价比最优 |
| 1.5B → 0.5B | 明显性能下降（尤其 ROUGE-L 和 BLEU） |

> 结论：**适度缩小 Reasoner 不影响性能**，但过小会削弱推理能力。

##### （2）融合系数 λ（lambda）的影响（图 6）
- 最佳范围：**λ ∈ [0.5, 2.0]**
- λ 过小 → 推理信号弱；λ 过大 → 生成不稳定
- 在合理范围内性能稳定，**无需精细调参**

##### （3）推理信号注入位置（表 2）
| 注入方式 | 性能 |
|----------|------|
| Input-only | 多样性高但对齐差 |
| Output-only | 上下文接地弱，性能低 |
| **Global（输入+输出）** | **最佳性能**，兼顾一致性与表达力 |

---

## **4. 关键结论和发现**

### **主要发现**
1. **“think-while-generating” 范式优于 “think-then-generate”**  
   动态、逐步推理更能适应长文本生成中的内容演化，避免静态推理的信息衰减。

2. **FlyThinker 实现了高效与高质量的统一**  
   通过分离 Reasoner 与 Generator，并打破推理 token 的自回归依赖，实现了：
   - **训练并行化**：成本接近 SFT
   - **推理零延迟**：与标准 LLM 几乎等速

3. **显著缓解“上下文漂移”问题**  
   在长文本后半段仍能维持个性化质量，优于所有基线。

4. **方法具有良好的鲁棒性和泛化性**
   - 在不同用户历史长度、不同难度样本上均表现稳定（Table 11）
   - 在 Qwen 和 Gemma 等不同架构上均有效（Table 7）

---

### **局限性**
1. **内存开销增加**  
   需维护两个模型（Generator + Reasoner），**内存占用更高**，不适合极端资源受限场景。

2. **Reasoner 设计依赖外部模型**  
   当前 Reasoner 是独立训练的 LLM，未来可探索更轻量化的替代方案。

3. **潜在推理的可解释性有限**  
   虽然 t-SNE 可视化显示不同用户有不同推理轨迹（图 7），但 latent reasoning 本身仍是黑箱。

---

### **未来工作方向**
1. **探索更小/更高效的 Reasoner 架构**  
   如 MoE、蒸馏模型，进一步降低部署成本。

2. **多模态个性化生成**  
   将 think-while-generating 范式扩展到图像、视频等生成任务。

3. **在线增量学习机制**  
   支持用户反馈实时更新 Reasoner，实现持续个性化适配。

4. **提升 latent reasoning 的可解释性**  
   探索如何将潜在推理映射为人类可读的中间表示。

---

> **总结一句话**：  
> FlyThinker 通过 **并行化的 think-while-generating 架构**，首次实现了**高效且高质量的个性化长文本生成**，解决了传统方法在动态性、效率和保真度之间的权衡难题。

</details>

---

### 8. [Enhancing Agentic RL with Progressive Reward Shaping and Value-based Sampling Policy Optimization](https://arxiv.org/abs/2512.07478)

**Authors**: Zhuoran Zhuang, Ye Chen, Jianghao Su, Chao Luo, Luhui Liu, Xia Zeng  
**Category**: cs.CL  
**Published**: 2025-12-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.07478v1  

#### Abstract
Large Language Models (LLMs) empowered with Tool-Integrated Reasoning (TIR) can iteratively plan, call external tools, and integrate returned information to solve complex, long-horizon reasoning tasks. Agentic Reinforcement Learning (Agentic RL) optimizes such models over full tool-interaction traje...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **Agentic Reinforcement Learning (Agentic RL)** 在 **Tool-Integrated Reasoning (TIR)** 场景下的两个关键挑战：
- **稀疏且非指导性奖励**（Sparse and non-instructive rewards）：传统基于最终答案正确性的二元 0-1 奖励（如 EM）无法为中间工具调用步骤提供有效反馈，导致学习效率低、收敛慢。
- **GRPO 中的梯度退化问题**（Gradient degradation in GRPO）：当一个 rollout 组内所有样本获得相同奖励时，其优势函数（advantage）为零，导致策略梯度消失，训练不稳定且样本利用率低。

### 提出的新方法
作者提出两种互补的技术来解决上述问题：

#### ✅ Progressive Reward Shaping (PRS)
- **思想来源**：受课程学习（curriculum learning）启发，设计分阶段、渐进式的奖励机制。
- **核心机制**：
  - 首先鼓励模型掌握基础能力（如生成可解析的工具调用格式）；
  - 再逐步引导其优化事实准确性与答案质量。
- **具体实现**：
  - **短答案 QA**：引入 **length-aware BLEU** 替代标准 BLEU，避免对简短但正确的答案进行不公平惩罚。
  - **长答案 QA**：结合 **LLM-as-a-Judge** 评分作为 `Rjudge`，防止模型通过“奖励黑客”（reward hacking）生成看似合理但不准确的回答。

#### ✅ Value-based Sampling Policy Optimization (VSPO)
- **目标**：改进 GRPO，缓解零优势样本带来的梯度退化。
- **核心机制**：
  - **Value-based Sampling**：定义任务价值得分 $ V = (R_{\text{max}} - \mu) \cdot \sigma $，其中 $\mu$ 和 $\sigma$ 分别是某任务下多个 rollout 的奖励均值与方差。该指标平衡了任务难度与策略不确定性，优先采样具有高学习潜力的任务。
  - **Value Smoothing Clipping**：对重复采样的高价值样本进行优势值平滑衰减，防止其在梯度更新中过度主导，提升训练稳定性。

### 相比现有方法的优势
| 方法 | 优势 |
|------|------|
| **PRS vs. Binary Reward** | 提供密集、阶段性反馈，显著加快收敛速度，减少无效探索 |
| **VSPO vs. PPO / GRPO / CISPO** | 更高的样本效率、更快的收敛速度、更强的训练稳定性，尤其在长周期推理任务中表现突出 |
| **PRS + VSPO 联合使用** | 协同效应明显，在多个 QA 基准上全面超越 SFT-only 及主流 RL 方法 |

---

## 2. 核心实验方法和设置

### 数据集
#### 🔹 长答案 QA（Long-Form QA）
- 来源于生产系统的专有对话数据，包含三类查询：
  - **Qsimple**：单个文本问题
  - **Qmultim**：图文混合查询（含图像 URL）
  - **Qmultiq**：包含多个子问题的复杂查询
- 参考答案由人工基于内部知识库标注。
- SFT 数据通过 **Qwen3-235B-A22B** 的知识蒸馏生成，并经过规则过滤和 LLM-as-a-Judge 清洗。

#### 🔹 短答案 QA（Short-Form QA）
- 使用七个公开基准数据集：
  - 通用问答：NQ, TriviaQA, PopQA
  - 多跳问答：HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle
- 训练集：合并 NQ 和 HotpotQA 的训练集
- 测试集：全部七项数据集的测试/验证集，用于评估领域内与跨域泛化能力

### 实验设置
- **模型**：
  - 短答案任务：Qwen2.5-7B-Instruct
  - 长答案任务：Qwen3-14B
- **训练框架**：Trinity-RFT
- **超参数**：
  - Rollout 数量：5
  - 学习率：1e-6
  - 温度 & top-p：均为 1.0
  - KL 系数 β：0.001，clip ratio ε：0.2
- **硬件**：单节点 8×H100 GPU，启用梯度检查点（gradient checkpointing）、FSDP + CPU 卸载以节省显存

### 评估指标
| 任务类型 | 评估方式 |
|--------|---------|
| **长答案 QA** | 使用 **Qwen3-235B-A22B-Instruct-2507** 作为 LLM Judge，判断回答是否与参考答案语义一致（Match/Mismatch） |
| **短答案 QA** | 使用 **Exact Match (EM)** 指标，直接匹配黄金答案 |

### 基线方法对比
- **Policy Optimization Baselines**：
  - PPO
  - GRPO
  - CISPO
  - SFT-only
- **Reward Design Baselines**：
  - Rule-based 0-1 reward（基于 EM 或 LLM 判定）
  - PRS（本文提出）

---

## 3. 主要实验结果和性能指标

### 📊 长答案 QA 结果（Table 1）
| 方法 | Qsimple | Qmultiq | Qmultim | 平均 |
|------|--------|--------|--------|-----|
| untrained | 0.6025 | 0.400 | 0.500 | — |
| SFT-only | 0.700 | 0.575 | 0.400 | — |
| GRPO | 0.700 | 0.575 | 0.475 | — |
| PPO | 0.6625 | 0.700 | 0.550 | — |
| CISPO | 0.6875 | 0.4875 | 0.400 | — |
| **VSPO** | **0.7125** | **0.725** | **0.550** | — |

> 💡 VSPO 在所有三类查询上均取得最佳性能，相比未训练模型平均相对提升达 **18.26% ~ 81.24%**

---

### 📊 短答案 QA 结果（Table 2）
| 方法 | NQ | TriviaQA | PopQA | HotpotQA | 2wiki | Musique | Bamboogle | **Avg** |
|------|----|----------|-------|----------|--------|----------|-----------|--------|
| Direct Inference | 0.134 | 0.408 | 0.140 | 0.183 | 0.250 | 0.031 | 0.120 | 0.181 |
| SFT | 0.318 | 0.354 | 0.121 | 0.217 | 0.259 | 0.066 | 0.112 | 0.207 |
| PPO+EM | 0.393 | 0.610 | 0.397 | 0.370 | 0.414 | 0.146 | 0.368 | 0.385 |
| GRPO+EM | 0.429 | 0.623 | 0.427 | 0.386 | 0.346 | 0.162 | 0.400 | 0.396 |
| VSPO+EM | 0.433 | 0.623 | 0.425 | 0.396 | 0.350 | 0.162 | 0.390 | 0.397 |
| PPO+PRS | 0.410 | 0.610 | 0.400 | 0.386 | 0.410 | 0.157 | 0.400 | 0.396 |
| GRPO+PRS | 0.440 | 0.639 | 0.420 | 0.400 | 0.390 | 0.171 | 0.413 | 0.410 |
| **VSPO+PRS** | **0.440** | **0.645** | **0.416** | **0.408** | **0.410** | **0.171** | **0.435** | **0.419** |

> ✅ **VSPO+PRS** 在所有数据集上达到最高平均分（**0.419**），显著优于其他组合。

---

### 🔍 消融实验结果（Ablation Study, Table 4）
| 设置 | Qsimple | Qmultiq | Qmultim |
|------|--------|--------|--------|
| w/o sample, w/o clip（≈GRPO） | 0.700 | 0.575 | 0.475 |
| random sample, w/o clip | 0.460 | 0.160 | 0.100 |
| value-based sample, w/o clip | 0.450 | 0.175 | 0.300 |
| random sample, w/clip | 0.500 | 0.300 | 0.025 |
| **value-based sample, w/clip（VSPO）** | **0.7125** | **0.725** | **0.550** |

> ⚠️ 关键发现：
> - 缺少 **value smoothing clipping** 会导致严重训练不稳定（KL divergence 异常升高）
> - **仅靠随机采样无法带来性能增益**
> - **只有同时具备 value-based sampling 和 value smoothing clipping 才能实现最优性能**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **PRS 显著提升学习效率**：
   - 分阶段奖励机制使模型先学会“怎么做”，再追求“做得好”。
   - 在训练早期即可达到较高收敛水平（见 Figure 3），且验证集表现更优。
   - 使用 PRS 的熵损失下降更快，表明策略探索更高效，而非陷入“entropy collapse”。

2. **VSPO 有效缓解 GRPO 梯度退化问题**：
   - 通过动态替换低价值样本，确保每一批次都能产生有效梯度。
   - **value-based sampling** 能识别出最具学习潜力的任务（既不太简单也不太难）。
   - **value smoothing clipping** 是稳定训练的关键，防止高价值样本反复出现导致梯度爆炸。

3. **PRS 与 VSPO 具有协同效应**：
   - 二者结合后在长短答案 QA 上均取得 SOTA 性能。
   - 不仅提升最终性能，还加快收敛速度、增强跨域泛化能力。

### ⚠️ 局限性
- **依赖高质量参考答案与 LLM Judge**：PRS 在长答案场景中的 `Rjudge` 依赖强大 LLM 进行评判，可能引入主观偏差或成本高昂。
- **任务价值计算假设 rollout 组内多样性**：若初始策略极差或极好，可能导致所有任务都难以区分价值。
- **当前实验集中在 QA 场景**：虽声称可推广至其他多步推理任务，但尚未在规划、决策等更复杂 Agentic 任务中验证。

### 🔮 未来工作方向
- 将 PRS 和 VSPO 应用于更广泛的 Agentic 任务，如自主代理规划、多智能体协作等。
- 探索自动构建分阶段奖励函数的方法，降低人工设计成本。
- 结合 offline RL 或 preference modeling，进一步提升样本效率。
- 研究如何将 VSPO 中的价值估计机制迁移到其他 critic-free RL 框架中。

---

> **总结一句话**：  
> 本文提出的 **PRS + VSPO** 构成了一个高效、稳定的 Agentic RL 框架，解决了稀疏奖励与梯度退化两大痛点，在多步工具调用推理任务中实现了更快收敛、更高性能与更好泛化。

</details>

---

### 9. [BitStopper: An Efficient Transformer Attention Accelerator via Stage-fusion and Early Termination](https://arxiv.org/abs/2512.06457)

**Authors**: Huizheng Wang, Hongbin Wang, Shaojun Wei, Yang Hu, Shouyi Yin  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2512.06457v1  

#### Abstract
Attention-based large language models (LLMs) have transformed modern AI applications, but the quadratic cost of self-attention imposes significant compute and memory overhead. Dynamic sparsity (DS) attention mitigates this, yet its hardware efficiency is limited by the added prediction stage and the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# BitStopper: An Efficient Transformer Attention Accelerator via Stage-fusion and Early Termination — 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前基于 **Dynamic Sparsity (DS)** 的 Transformer 注意力加速器虽然通过稀疏化减少计算量，但普遍存在以下两个瓶颈：

- **预测阶段开销过大**：大多数 DS 方法依赖一个独立的低精度 **sparsity predictor** 来预判重要 Q-K 对。该模块需要访问完整的 Key 矩阵（S×H），导致内存带宽压力巨大，其功耗甚至超过正式计算阶段（如 SOFA 中占 >75%）。
- **Token 选择策略僵化**：常用固定阈值或 top-k 机制无法适应不同 Query 下注意力分数分布的多样性，导致误删关键 token 或保留冗余 token。

这些问题严重限制了硬件效率，使得理论上的稀疏优势难以转化为实际性能提升。

---

### **提出了什么新方法或新思路**

本文提出 **BitStopper**，一种算法-架构协同设计的细粒度注意力加速器，核心创新如下：

#### （1）**Bit-Serial Enabled Stage Fusion (BESF)**  
- **思想**：将传统的“先预测 → 再执行”两阶段流程融合为单一执行过程，消除独立的 sparsity predictor。
- **实现**：采用 **bit-serial computing**，从 Key 向量的最高有效位（MSB）开始逐位加载并计算部分点积。一旦某 token 被判定不重要，则立即终止对其后续 bit plane 的加载与计算（early termination）。
- **优势**：避免重复访存，显著降低 DRAM 访问次数和预测开销。

#### （2）**Lightweight Adaptive Token Selection (LATS)**  
- **思想**：在 bit-level 上动态判断 token 是否可剪枝，且决策自适应于当前 attention 分数分布。
- **关键技术**：
  - 引入 **bit-level uncertainty margin**，估计仅知部分 bit 时 dot-product 的上下界范围。
  - 设计基于最大值偏移的自适应阈值：`threshold = max(A_r.min) - α × radius`，其中 `α ∈ [0,1]` 控制剪枝激进程度。
- **优势**：无需复杂 softmax 推断即可高精度识别关键 token，适用于多样化的 attention 分布。

#### （3）**Bit-level Asynchronous Processing (BAP)**  
- **思想**：打破传统 bit-serial 处理的严格顺序依赖，允许异步获取和处理不同 token 的下一个 bit plane。
- **实现**：维护一个 Scoreboard 缓存未被剪枝 token 的中间结果；当某个 bit plane返回后立即更新对应 partial sum，并请求下一 bit。
- **优势**：有效隐藏 DRAM 访问延迟，提高计算单元利用率。

#### （4）定制化硬件架构 BitStopper  
- 包含 **QK-PU**（支持 BESF/LATS/BAP）和 **V-PU**（完成最终 softmax 和 V 加权）。
- 支持 INT12 定点量化，每个 Key 向量分解为 12 个 1-bit plane 进行处理。

---

### **相比现有方法的优势**

| 维度 | 传统 DS 方法（如 Sanger, SOFA） | BitStopper |
|------|-------------------------------|-----------|
| 预测机制 | 独立低精度 predictor（额外访存） | 无 predictor，预测与执行融合（BESF） |
| 冗余访存 | 必须全量加载 Key 矩阵 | 早期终止无效 token 的后续 bit 加载 |
| 决策粒度 | 固定阈值 / top-k（粗粒度） | 自适应、bit-level 动态决策（LATS） |
| 计算效率 | 易受内存延迟影响 | 异步处理（BAP）提升利用率 |
| 灵活性 | 多需重训练恢复精度 | 无需重训练，PTQ 即可部署 |

---

## 2. 核心实验方法和设置

### **使用的模型与数据集**

- **模型**：
  - **OPT-1.3B**
  - **Llama2-7B**
- **数据集**：
  - **Wikitext-2**（用于 PPL 评估）
  - **Dolly**（开源指令调优数据集，测试真实场景）

所有模型使用 **Post-Training Quantization (PTQ)** 转换为 **INT12** 表示作为精度基准。

---

### **实验设置与评估指标**

#### **硬件配置（TSMC 28nm, 1GHz）**
| 组件 | 配置 |
|------|------|
| 主存 | HBM2，8通道×128-bit @ 2Gbps，总带宽 32GB/s/channel |
| 片上缓存 | 320KB SRAM（Key/Value），8KB Q buffer |
| QK-PU | 32 个 bit-level PE lanes，每 lane 支持 64-dim × 12-bit × 1-bit ANDer tree |
| V-PU | 64-way INT12 MAC array，LUT-based Softmax |

#### **评估工具链**
- **RTL 实现 + Synopsys DC**：逻辑面积与功耗估算
- **Cycle-level simulator**：性能模拟
- **CACTI**：片上存储能耗建模
- **Ramulator**：DRAM 访问延迟与能耗分析

#### **评估指标**
- **Perplexity (PPL)**：衡量模型质量（越低越好）
- **Speedup**：相对于 baseline 的吞吐提升
- **Energy Efficiency (TOPS/W)**：能效比
- **Off-chip Memory Access Reduction**：外部内存访问减少比例
- **Area & Power Breakdown**：芯片资源消耗

---

### **基线方法对比**

| 基线 | 描述 |
|------|------|
| **Baseline** | BitStopper 架构但关闭所有稀疏功能（dense attention） |
| **Sanger [20]** | 使用 4-bit predictor + 固定阈值的粗粒度稀疏加速器 |
| **SOFA [19]** | 基于 log-domain 和 top-k 的跨阶段 tile 优化方法，需微调恢复精度 |
| **TokenPicker [26]** | 渐进式 4-bit chunk 剪枝，基于 post-softmax 概率决策 |

所有设计归一化至相同工艺节点与面积约束下进行公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 指标 | 结果 |
|------|------|
| **峰值能效** | **11.36 TOPS/W** |
| **芯片面积** | **6.84 mm²** |
| **功耗** | **703 mW** |
| **新增模块开销** | Bit Margin Generator + LATS 模块仅增加 4.9% 面积、6.9% 功耗 |

---

### **与基线方法的对比结果**

#### ✅ **能效提升**
- 相比 **Sanger**：**2.4× 能效提升**
- 相比 **SOFA**：**2.1× 能效提升**
- 相比 **Baseline（dense）**：**3.7× 能效提升**

#### ✅ **速度提升**
- 相比 **Sanger**：**2.03× 速度提升**
- 相比 **SOFA**：**1.89× 速度提升**
- 相比 **Baseline**：**3.2× 速度提升**

> 注：随着序列长度增长（如从 1k 到 4k），BitStopper 的优势更加明显，因其更有效地利用长序列中的冗余。

#### ✅ **内存访问减少**
- 在 Llama2-7B 上（Dolly 数据集）：
  - 平均 **DRAM 访问减少 2.9×（vs. Sanger）**
  - 平均 **DRAM 访问减少 2.1×（vs. SOFA\*）**
  - 比未微调的 SOFA 减少 **2.8× 更多内存访问**

#### ✅ **精度保持**
- 在 α ≈ 0.6 时，PPL 损失控制在 +0.1 以内，与原始 INT12 基准相当。
- 图 13(a) 显示，在 α ≥ 0.6 时，1/PPL 下降缓慢，复杂度持续下降；低于 0.6 后精度急剧恶化。

---

### **消融实验结果（Ablation Study）**

图 13(b) 展示了各组件对整体加速的贡献：

| 阶段 | Speedup | Compute Utilization |
|------|--------|---------------------|
| Baseline（dense） | 1.0× | — |
| + BESF | 1.25× | 48% |
| + BESF + BAP | 1.25× → **2.03×**（累计 1.63× 提升） | 48% → **83%** |
| + BESF + BAP + LATS | 最终 **3.2×**（额外 1.57× 提升） | — |

- **BAP 是利用率提升的关键**：通过异步处理隐藏内存延迟，使计算单元活跃时间大幅提升。
- **LATS 显著减少冗余计算与访存**：自适应剪枝策略精准剔除无关 token。

---

## 4. 关键结论和发现

### **主要发现**

1. **独立的 sparsity predictor 成为性能瓶颈**：尽管旨在节省计算，但其带来的额外访存反而成为主导功耗源（>75% in SOFA），违背初衷。
2. **Stage fusion 可从根本上消除冗余**：通过 bit-serial 计算将预测融入执行，实现真正的“边算边剪”，大幅削减无效操作。
3. **bit-level granularity 提供极致优化潜力**：相比 chunk-level（如 4-bit）剪枝，bit-level 允许更早终止、更高灵活性。
4. **异步处理是应对内存墙的有效手段**：BAP 显著提升了计算单元利用率（从 48% → 83%），释放了 bit-serial 架构的潜力。
5. **轻量级自适应决策优于静态策略**：LATS 不依赖复杂 softmax，却能在多变 attention 分布中维持高准确率。

---

### **方法的局限性**

- **依赖高位优先处理**：要求 Key 向量按 bit plane 存储或可高效切分，可能增加软件栈负担。
- **对极端稀疏模式敏感**：若大部分 token 都需完整处理，则 early termination 效益减弱。
- **目前仅验证于 INT12**：是否适用于更低比特（如 INT8 或 FP8）尚待研究。
- **尚未支持动态序列长度切换**：调度器假设固定 sequence length。

---

### **未来工作方向**

- 扩展至 **Prefill + Decoding 全流程加速**（当前已支持两者，但未深入优化 decoding 特性）。
- 探索 **bit-plane 压缩与编码技术**，进一步减少 off-chip 传输。
- 将 BESF 思想推广到 **MLP 层或其他 DNN 模块**，构建统一 bit-grained 加速框架。
- 结合 **learned pruning policy** 与 LATS，实现更高层次的自适应控制。
- 开发配套编译器支持自动 bit-plane 切分与调度。

---

## 总结

> **BitStopper** 是首个明确提出“**sparsity predictor 自身已成为瓶颈**”并予以根本性解决的工作。它通过 **stage-fusion + bit-serial early termination + asynchronous processing** 的三位一体设计，在不牺牲精度的前提下，实现了高达 **3.2× 速度提升** 和 **3.7× 能效改进**，代表了下一代高效注意力加速器的重要方向。

</details>

---

### 10. [Parallel Algorithms for Combined Regularized Support Vector Machines: Application in Music Genre Classification](https://arxiv.org/abs/2512.07463)

**Authors**: Rongmei Liang, Zizheng Liu, Xiaofei Wu, Jingwen Tu  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2512.07463v1  

#### Abstract
In the era of rapid development of artificial intelligence, its applications span across diverse fields, relying heavily on effective data processing and model optimization. Combined Regularized Support Vector Machines (CR-SVMs) can effectively handle the structural information among data features, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**分布式存储的大规模数据**中，缺乏高效算法来求解具有复杂结构先验的 **Combined Regularized Support Vector Machines (CR-SVMs)** 这一挑战。传统的 SVM 模型在处理高维、结构化数据（如音乐信号）时，难以有效利用特征间的分组和时序依赖关系，且现有算法在分布式环境下效率低下。

### 提出的新方法与新思路
1. **提出统一优化框架**：
   - 构建了一个基于共识结构（consensus structure）的通用优化框架，适用于多种 **loss functions** 和 **combined regularization terms**（如 Elastic Net, Sparse Fused Lasso, Sparse Group Lasso），并可扩展至非凸正则项（如 SCAD、MCP）。
   - 实现了对 54 种不同 CR-SVM 模型的统一建模与求解。

2. **设计分布式并行 ADMM 算法**：
   - 开发了一种分布式的并行 **Alternating Direction Method of Multipliers (ADMM)** 算法用于高效计算 CR-SVMs。
   - 针对因截距项 $ \beta_0 $ 导致无法转化为两块 ADMM 的问题，设计了三块 ADMM 结构，并引入 **Gaussian back-substitution** 技术确保收敛性，实现了改进的次线性收敛速率。

3. **提出 SGL-SVM 新模型**：
   - 首次正式提出将 **Sparse Group Lasso (SGL)** 正则化直接应用于 SVM 分类任务，命名为 **SGL-SVM**。
   - 该模型能同时实现**组稀疏性**和**组内个体特征稀疏性**，特别适合分析具有天然分组结构的高维数据（如音乐特征）。

### 相比现有方法的优势
- **普适性强**：算法不依赖于具体的 loss 或 regularization 形式，具备高度可扩展性。
- **分布式高效性**：算法计算复杂度不受正则项和损失函数影响，适合大规模分布式场景。
- **理论保障**：通过 Gaussian back-substitution 保证了多块 ADMM 在非正交约束下的收敛性。
- **实际应用价值高**：在音乐分类任务中展现出更强的可解释性和更高的分类精度。

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **合成数据（Synthetic Data）**：
   - 数据服从多元正态分布 $ N(\mu^+, \Sigma) $ 和 $ N(\mu^-, \Sigma) $，前10个特征为关键变量，其余为噪声。
   - 实验配置：$(n, p) = (200,000, 500)$ 和 $(500,000, 1,000)$，测试样本 $ m = 1,000,000 $。
   - 加入 20% 标签噪声以验证鲁棒性。

2. **真实数据集：FMA (Free Music Archiv)**
   - 来自 GitHub 的公开音乐分析数据集 `fma_small` 子集。
   - 包含 8 类音乐，每类 1,000 个音频样本，共 8,000 条。
   - 特征维度：1036 维，按物理意义分为 7 组：
     - 时间域特征（16）
     - 谱特征（32）
     - MFCC 特征（160）
     - Chroma 特征（96）
     - Tonnetz 特征（48）
     - Echonest 特征（518）
     - 其他特征（166）

### 实验设置与评估指标
- **训练/测试划分**：随机选取两类，每类取 800 训练 + 200 测试。
- **并行环境**：模拟 K=2 到 K=20 个子机器进行分布式训练。
- **参数选择**：
  - 使用 **SVMIC 准则** 自动选择正则参数 $\lambda_1, \lambda_2$，避免昂贵的交叉验证。
  - $\rho \in \{0.01, 0.1, 1\}$，采用自适应调整策略加速收敛。
- **初始值**：所有 primal 和 dual 变量初始化为 0。

### 基线方法对比
- **QPADM-slack** (Guan et al., 2020)
- **QPADM-slack(GB)** (Wu et al., 2025b)
- **M-QPADM-slack(GB)** (Wu et al., 2025b)

所提方法记为 **PADMM**（Parallel ADMM for CR-SVMs），加载 SGL-SVM 模型。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 7 & Table 8）

#### 合成数据表现（Table 7）
| 方法 | $(n,p)$ | $K$ | CAR (%) | CT (s) | NI | NTSF |
|------|---------|-----|--------|--------|----|-------|
| PADMM | (200K,500) | 5 | **0.982** | **2.230** | **19.4** | **10.1** |
| QPADM-slack | (200K,500) | 5 | 0.975 | 2.300 | 20.0 | 10.4 |
| PADMM | (500K,1000) | 5 | **0.979** | **4.678** | **25.1** | **10.4** |

- **PADMM 在 CAR 上最高，CT 最低，迭代次数 NI 更少，NTSF 更接近真实值 10**。
- 随着 $K$ 增加，所有方法 CAR 下降、CT 下降、NI 上升，这是共识结构固有代价，但 PADMM 下降最缓。

#### 真实数据 FMA 表现（Table 8）
| $K$ | 方法 | Sparsity (%) | Train Acc (%) | Test Acc (%) |
|-----|------|--------------|---------------|---------------|
| 2 | **PADMM** | **90.38** | **99.32** | **97.27** |
| 2 | QPADM-slack(GB) | 85.99 | 95.14 | 92.96 |
| 2 | QPADM-slack | 83.64 | 93.38 | 92.00 |
| 20 | **PADMM** | **89.57** | **98.34** | **96.27** |
| 20 | QPADM-slack(GB) | 76.26 | 88.64 | 85.35 |

- **PADMM 在所有 $K$ 下均保持最高的 sparsity、train 和 test accuracy**。
- 即使在 $K=20$ 时，PADMM 的 test accuracy 仍达 **96.27%**，显著优于其他方法（最低仅 85.18%）。
- PADMM 的稀疏性下降更平缓，说明其在分布式下仍能维持良好特征选择能力。

### 消融实验与特征重要性分析（Figures 4 & 5）
- **特征组重要性排序**（基于 SGL-SVM 系数绝对值之和）：
  1. **MFCC 特征组**（最重要）
  2. **谱特征组**
  3. **Chroma 特征组**
  4. 其余四组（Tonnetz, Echonest, 时间域等）贡献极小，在强正则下几乎被剔除。

- **实际意义**：当内存受限时，只需收集 **MFCC + 谱 + Chroma** 三个特征组即可构建高效的音乐分类模型，大幅节省存储与计算开销。

---

## 4. 关键结论和发现

### 主要发现
1. 所提出的 **PADMM 算法在合成与真实数据上均表现出卓越的可靠性、稳定性与效率**，尤其在大规模分布式场景下优于现有并行 SVM 算法。
2. **SGL-SVM 模型能够自动识别最具判别力的特征组及其内部关键特征**，提升模型可解释性，符合音乐学认知。
3. 算法的计算复杂度独立于 loss function 和 regularization term，验证了其**高度通用性**。
4. 在音乐分类任务中，**MFCC 特征最为关键**，其次是谱特征和 Chroma 特征。

### 方法的局限性
- 当子机器数量 $K$ 过大时，由于共识结构带来的通信与协调开销，**分类准确率会逐渐下降**。
- 当前算法主要面向**凸损失函数**（如 hinge loss），未涵盖非凸损失函数情形。
- 实验平台内存有限，限制了更大规模并行（$K > 20$）的加速效果。

### 未来工作方向
1. 将框架扩展至 **非凸损失函数** 场景。
2. 探索在 **噪声或缺失数据** 下的鲁棒性与收敛性质。
3. 引入 **随机梯度下降（SGD）或加速技术**（如 Nesterov）进一步提升收敛速度。
4. 应用于 **multi-label 任务**，如音乐自动打标签（auto-tagging）。
5. 探索更多领域（如生物信息学、金融）中的结构化数据建模。

> **代码开源**：作者已将实现代码发布于 GitHub：[https://github.com/xfwu1016/PADMM-for-Svms](https://github.com/xfwu1016/PADMM-for-Svms)，便于复现实验与后续研究。

</details>

---

### 11. [GatedFWA: Linear Flash Windowed Attention with Gated Associative Memory](https://arxiv.org/abs/2512.07782)

**Authors**: Jiaxu Liu, Yuhe Bai, Christos-Savvas Bouganis  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2512.07782v1  

#### Abstract
Modern autoregressive models rely on attention, yet the Softmax full attention in Transformers scales quadratically with sequence length. Sliding Window Attention (SWA) achieves linear-time encoding/decoding by constraining the attention pattern, but under an \textit{Associative Memory} interpretati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# GatedFWA: Linear Flash Windowed Attention with Gated Associative Memory 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现代自回归模型（如Transformer）依赖于 **Softmax full attention**，其计算复杂度随序列长度呈二次方增长（$O(N^2)$），限制了长序列建模的效率。为缓解此问题，**Sliding Window Attention (SWA)** 被提出以实现线性时间解码（$O(Nw)$），但作者指出其存在两个关键缺陷：

- **梯度不稳定（Gradient Instability）**：从 **Associative Memory** 视角看，SWA 的更新机制是“差分式”的（difference-style update），即当前状态减去滑出窗口的历史状态。这种更新方式隐含地优化一个无界的线性目标函数，容易导致记忆幅度过大，引发梯度爆炸。
- **Softmax 的梯度消失（Gradient Vanishing）**：Softmax 注意力通过归一化不断压缩每一步的记忆更新，随着序列变长，有效更新逐渐缩小，导致梯度在深层网络中难以回传。

此外，尽管 SWA 效率高，但其硬性的窗口边界可能导致上下文不连续，影响全局语义理解。

---

### **提出了什么新方法或新思路**

作者提出 **GatedFWA (Gated Flash Windowed Attention)**，一种结合了 SWA 高效性和可控记忆更新机制的新型注意力架构。其核心思想是：

- 引入一个**可学习的非负门控（non-negative gate）**，该门控对每个 token 和注意力头独立计算，并沿序列累积形成一个**衰减偏置（decay bias）**，注入到注意力 logits 中。
- 从 **Associative Memory** 的角度看，这一机制相当于对携带的记忆施加了一个**可学习的收缩操作（learnable contraction）**，形式为：
  $$
  M_t = \exp(-\alpha) M_{t-1} + \cdots
  $$
  其中 $\alpha$ 是由门控累积得到的衰减项。

这使得模型能够：
- **稳定记忆更新**：防止 SWA 中因差分更新导致的记忆无限增长。
- **控制梯度流**：通过调节 $\alpha$，模型可以决定是保留长期依赖（$\alpha \to 0$）还是阻断无关历史（$\alpha \to \infty$）。
- **软化窗口边界效应**：相比 SWA 的硬截断，GatedFWA 通过门控选择性地削弱“路径外”历史，使注意力分布更平滑。

---

### **相比现有方法的优势**

| 特性 | Softmax | SWA | GatedFWA |
|------|--------|-----|----------|
| 时间复杂度 | $O(N^2)$ | $O(Nw)$ | $O(Nw)$ ✅ |
| 内存复杂度（KV Cache） | $O(Nd)$ | $O(wd)$ | $O(wd)$ ✅ |
| 梯度稳定性 | ❌ 归一化导致梯度消失 | ❌ 差分更新导致梯度不稳定 | ✅ 可控门控稳定梯度流 |
| 上下文连贯性 | ✅ 全局可见 | ⚠️ 窗口跳跃导致不连续 | ✅ 门控软化边界 |
| 硬件友好性 | ✅ FlashAttention 支持 | ✅ 支持 | ✅ 支持（兼容 FlashAttention） |

此外，GatedFWA 可无缝集成到 **token compression/selection** 方法（如 NSA）中，作为其局部滑动模块的直接替代，进一步提升长程建模能力。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **语言建模基准**：
  - **WikiText103**
  - **OpenWebText**
- **下游任务评估**（SlimPajama 子集训练，Mistral tokenizer）：
  - PiQA, HellaSwag, WinoGrande, ARC-e/c, COPA, OpenBookQA, SciQA, BoolQ
- **召回密集型任务**：
  - **Multi-Query Associative Recall (MQAR)**：用于测试模型对键值对关联记忆的能力。

---

### **实验设置和评估指标**

#### **模型配置**
- 模型规模：120M / 360M 参数
- 上下文长度：1024 / 4096
- 注意力窗口大小 $w$：512 或 1024
- 实现框架：基于 Triton 的 fused kernel，确保 I/O 高效

#### **评估指标**
- **语言建模**：验证集损失（Val Loss ↓）
- **下游任务**：准确率（Accuracy ↑）或归一化准确率（acc_norm）
- **效率评估**：
  - 正向/反向传播时间
  - 吞吐量（Throughput）
  - 预处理开销

#### **基线方法对比**
- **标准架构**：
  - Transformer (LLaMA 架构)
  - +SWA
  - +SWA + NSA
- **State Space Models (SSMs)**：
  - Mamba
  - RetNet
  - RWKV
  - GLA
  - HGRN2
- **其他线性注意力变体**

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **语言建模性能（OpenWebText）**

| 架构 | #Param. (M) | N=1024 Val Loss | N=4096 Val Loss |
|------|-------------|------------------|------------------|
| Transformer (LLaMA) | 124.4 | 3.247 | 3.273 |
| +SWA | 124.4 | 3.248 | 3.274 |
| +SWA + NSA | 125.4 | 3.240 | 3.248 |
| **+GatedFWA** | **125.1** | **3.237** | **3.255** |
| **+GatedFWA + NSA** | **126.1** | **3.215** | **3.230** |

> ✅ GatedFWA 在所有配置下均优于基线，尤其在长上下文（N=4096）上优势明显。

#### **下游任务平均准确率（~360M, 15B tokens）**

| 架构 | 平均准确率 ↑ |
|------|--------------|
| Transformer (LLaMA) | 50.41 |
| +GatedFWA | 50.98 |
| **+GatedFWA + NSA** | **51.65** |
| 相对提升 | **+2.45%** |

> ✅ 所有任务均取得提升，表明 GatedFWA 更好地保留了长期信用分配。

---

### **与基线方法的对比结果**

- **MQAR 回调率测试**（Fig. 8）：
  - GatedFWA 在小维度（d=64）、长序列（N=512）下显著优于 Mamba、RWKV 等 SSM 模型。
  - SWA 在 N=128 时已失效，而 GatedFWA 仍保持良好性能。
- **Scaling Law**（Fig. 6）：
  - GatedFWA 在相同参数量和上下文长度下，始终位于帕累托前沿，优于 RetNet、Mamba、RWKV 等。
- **效率对比**（Fig. 10）：
  - GatedFWA 与 SWA 的正向/反向时间几乎一致，远优于 FlashAttention（Softmax）。
  - 在 N > 64K 时，GatedFWA 比 FA 快约 **30×**。
  - 预处理开销极低：N=64K 时仅 **0.3ms**，远低于 PyTorch 实现的 2.9ms。

---

### **消融实验结果**

#### **门控幅度参数 $\beta$ 是否可学习（Fig. 9）**
- 将 $\beta$ 设为固定值（如 1） vs. 可学习：
  - 可学习 $\beta$ 在训练损失上持续领先，说明模型能自适应调整门控强度。
  - 表明 learnable amplitude 对性能有正向作用。

#### **注意力模式可视化（Fig. 11, 13, 14）**
- **SWA-NSA**：注意力分布呈现明显的“跳跃”或“条纹”伪影（striding artifacts），因硬窗口切换所致。
- **GatedFWA-NSA**：注意力更连续、平滑，边界过渡自然，得益于门控对历史的软擦除机制。

---

## 4. 关键结论和发现

### **主要发现**

1. **从 Associative Memory 视角重新审视注意力机制** 是有效的分析工具，揭示了 Softmax 和 SWA 分别存在的梯度消失与梯度不稳定问题。
2. **GatedFWA 成功在保持线性复杂度的同时，实现了对记忆更新的可控性**：
   - 通过可学习门控引入软衰减，避免了 SWA 的无界更新。
   - 梯度路径变得可控，支持长程依赖建模。
3. **硬件对齐设计至关重要**：
   - 采用 **fused one-pass preprocessing** 和 **FlashAttention-compatible kernel**，确保 I/O 高效、数值稳定。
   - 实际开销可忽略，吞吐量与 SWA 相当。
4. **与 token compression 方法（如 NSA）天然兼容**：
   - 可作为 NSA 中的局部滑动模块替换，进一步增强全局上下文感知。

---

### **方法的局限性**

- **理论表达能力受限于 TC⁰ 复杂度类**：
  - GatedFWA 的记忆更新本质上仍是并行可计算的（parallelizable），无法解决需要顺序依赖的 NC¹ 完全问题（如 S₅ 置换追踪）。
  - 其门控为对角矩阵，不具备非交换性（non-commutative）状态转换能力。
- **仍依赖滑动窗口假设**：
  - 虽然可通过 NSA 扩展有效上下文，但局部模块本身受限于窗口大小 $w$。

---

### **未来工作方向**

1. **探索读写内存机制（read-write memory）**：
   - 如引入 **Delta Rule** 更新：
     $$
     M_t = M_{t-1} + (v - k^\top M_{t-1})k
     $$
     这类非线性、非对角更新可能突破 TC⁰ 局限，达到 NC¹ 表达能力。
2. **结合动态稀疏性与门控机制**：
   - 将门控用于指导 token selection/compression，而非仅用于记忆衰减。
3. **扩展至多模态与事件流建模**：
   - 利用其高效性和可控梯度流，在 vision-language 或 event stream 场景中应用。

---

> **总结**：GatedFWA 是一种兼具效率、稳定性与表达力的线性注意力机制，通过引入可学习门控实现了对 SWA 记忆更新的“软约束”，在语言建模、长程依赖任务中表现出色，且易于部署。它不仅是一个实用的工程改进，也为理解注意力机制的动态行为提供了新的理论视角。

</details>

---

### 12. [Optimizing LLMs Using Quantization for Mobile Execution](https://arxiv.org/abs/2512.06490)

**Authors**: Agatsya Yadav, Renta Chintala Bhargavi  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.06490v1  

#### Abstract
Large Language Models (LLMs) offer powerful capabilities, but their significant size and computational requirements hinder deployment on resource-constrained mobile devices. This paper investigates Post-Training Quantization (PTQ) for compressing LLMs for mobile execution. We apply 4-bit PTQ using t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Optimizing LLMs Using Quantization for Mobile Execution*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）虽然功能强大，但由于其巨大的参数量和计算开销，难以直接部署在资源受限的移动设备上。传统方案依赖云端API，带来**隐私泄露、延迟高、离线不可用**等问题。

本文旨在解决：  
> 如何在不牺牲太多性能的前提下，将高性能LLM压缩并高效部署到普通Android手机等边缘设备上？

---

### 🚀 提出的新方法与创新思路
提出了一套**基于Post-Training Quantization (PTQ)** 的完整可复现流程，用于将Meta的Llama 3.2 3B模型压缩并在移动端运行：

1. **采用4-bit PTQ量化策略**：
   - 使用 `BitsAndBytes` 库进行初始4-bit量化（nf4格式），无需重新训练。
2. **转换为GGUF格式**：
   - 利用 `llama.cpp` 工具链将量化后的模型转为GGUF格式，专为CPU推理优化，适合无GPU支持的移动环境。
3. **结合Ollama + Termux实现安卓端本地推理**：
   - 在标准Android设备上通过Termux运行Ollama框架加载GGUF模型，完成端侧执行验证。

该方法实现了从云端依赖向**完全本地化、低内存占用、离线可用**的LLM部署转变。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | 本文优势 |
|------|--------|----------|
| Cloud-based API | 隐私差、需联网、延迟高 | 支持**离线运行**，数据不出设备 |
| 轻量级模型（如MobileBERT） | 性能显著下降 | 保留原始LLM架构，仅压缩精度，性能损失小 |
| QAT（Quantization-Aware Training） | 需要训练数据与大量算力 | 使用PTQ，**无需重训**，快速易部署 |
| Pruning / Distillation | 复杂且可能破坏结构 | 流程简单，兼容性强，工具链开源开放 |

✅ **核心优势总结**：
- **实用性高**：全流程基于开源工具（Hugging Face, BitsAndBytes, llama.cpp, Ollama）
- **压缩率高**：实现68.66%模型体积缩减
- **可行性验证**：真正在消费级Android手机上成功运行

---

## 2. 核心实验方法和设置

### 📦 使用的模型与工具
- **目标模型**：Meta’s **Llama 3.2 3B**（30亿参数，BF16精度）
- **量化工具**：
  - `BitsAndBytes`：实现4-bit NF4量化
  - `llama.cpp`：转换为GGUF，并应用`q4_k_m`二次量化
- **部署平台**：
  - Google Colab（NVIDIA T4 GPU）用于量化处理
  - OnePlus Nord CE 5G（Snapdragon 750G, 12GB RAM, Android 13）用于终端测试
- **运行环境**：Termux + Ollama

---

### 🎯 实验设置与评估指标

#### 数据集
- **无监督/零样本场景为主**，未使用特定训练或微调数据集
- 评估基准：
  - **WikiText-2**：用于计算Perplexity
  - **DailyMail** 摘要任务：用于BLEU评分
  - **MMLU**（Massive Multitask Language Understanding）：跨学科理解能力评测

#### 评估指标
| 类型 | 指标 |
|------|------|
| 定量指标 | - 模型大小（GB）<br>- Perplexity（WikiText-2）<br>- BLEU分数（DailyMail）<br>- MMLU准确率 |
| 定性指标 | - 推理输出质量（人工判断连贯性、相关性）<br>- 是否可正常启动与响应 |
| 部署可行性 | - 是否能在Android设备运行<br>- 内存占用情况 |

#### 基线对比模型
- 原始Llama 3.2 3B（BF16）
- 其他适用于边缘设备的小模型：
  - Phi-2（2.7B）
  - Gemma 2B
  - Mistral 7B（GGUF q4_k_m）
  - TinyLlama 1.1B

---

## 3. 主要实验结果和性能指标

### 📉 模型压缩效果（关键数据）

| 模型阶段 | 大小（GB） | 压缩率 |
|---------|-----------|--------|
| 原始 Llama 3.2 3B (BF16) | 6.00 | — |
| BitsAndBytes 4-bit量化后 | 2.10 | 64.92% |
| 最终 GGUF q4_k_m | **1.88** | **68.66%** |

➡️ **最终模型仅占原模型1/3不到的空间**，满足主流手机存储限制。

---

### 📊 功能性能保持情况

#### MMLU 准确率对比（越高越好）

| Model | 参数量 | MMLU Score (%) |
|-------|--------|----------------|
| Llama 3.2 3B (Original) | 3B | 64.2 |
| Llama 3.2 3B (GGUF q4_k_m) | 3B | **61.8** |
| Phi-2 | 2.7B | 68.8 |
| Gemma 2B | 2B | 64.3 |
| Mistral 7B (q4_k_m) | 7B | 62.5 |
| TinyLlama 1.1B | 1.1B | 54.9 |

📌 尽管是4-bit量化，**性能仅下降约2.4个百分点**，仍优于多数同级别小模型。

---

#### 其他定量评估结果
- **Perplexity (WikiText-2)**: **8.57 ± 0.06** → 表明语言建模能力良好
- **BLEU Score (DailyMail)**: **0.45** → 显示摘要生成具备一定有效性

> 注：作者指出这些指标主要用于辅助分析，若进一步微调有望提升表现。

---

### ❌ 缺乏消融实验
论文中**没有提供不同量化配置之间的系统性消融研究**（例如比较nf4 vs gptq vs smoothquant），也未测试不同GGUF量化等级（如q3_k_m vs q5_k_m）的影响。

但明确说明选择`q4_k_m`是因为它在**压缩率与保真度之间取得较好平衡**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **4-bit PTQ + GGUF 是可行的技术路径**：
   - 可将3B级LLM压缩至1.88GB以内，在普通智能手机上实现本地推理。
2. **性能损失可控**：
   - MMLU得分仅下降~2.4%，仍具备较强多任务理解能力。
3. **隐私与离线优势突出**：
   - 所有推理在本地完成，无需网络连接，保障用户隐私。
4. **开源生态成熟**：
   - Hugging Face + BitsAndBytes + llama.cpp + Ollama 构成了完整的轻量化部署链条。

---

### ⚠️ 局限性
1. **缺乏全面的定量评估**：
   - 主要依赖定性观察，缺少在多个NLP任务上的系统benchmark。
2. **推理速度未精确测量**：
   - 仅主观描述“较慢但仍可用”，缺乏tokens/sec等关键性能指标。
3. **硬件与模型单一性**：
   - 仅测试OnePlus手机和Llama 3.2 3B，泛化性待验证。
4. **用户体验门槛高**：
   - 当前需通过Termux命令行操作，对非技术用户极不友好。
5. **PTQ精度上限问题**：
   - 虽然便捷，但相比QAT可能在极端低比特下精度更低。

---

### 🔮 未来工作方向
1. **更深入的量化对比实验**：
   - 对比GPTQ、AWQ、SmoothQuant等PTQ方法在移动端的表现。
2. **系统性能剖析**：
   - 测量token生成速度、功耗、内存峰值，建立真实可用性画像。
3. **扩展至更多模型**：
   - 尝试Llama 3系列更大/更小版本或其他开源LLM（如Qwen, DeepSeek）。
4. **开发原生App集成**：
   - 构建GUI版Android/iOS应用，降低使用门槛。
5. **混合压缩策略探索**：
   - 结合pruning、distillation与quantization，追求更高压缩比而不失质。

---

## ✅ 总结一句话
本论文展示了通过 **4-bit PTQ + GGUF转换**，可以将Llama 3.2 3B这样的中等规模LLM压缩近七成体积，并成功部署于消费级Android设备，验证了**高性能LLM本地化运行的现实可行性**，为移动AI提供了低成本、高隐私、可离线的新范式。

</details>

---

### 13. [LLM-Driven Composite Neural Architecture Search for Multi-Source RL State Encoding](https://arxiv.org/abs/2512.06982)

**Authors**: Yu Yu, Qian Xie, Nairen Cao, Li Jin  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.06982v1  

#### Abstract
Designing state encoders for reinforcement learning (RL) with multiple information sources -- such as sensor measurements, time-series signals, image observations, and textual instructions -- remains underexplored and often requires manual design. We formalize this challenge as a problem of composit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM-Driven Composite Neural Architecture Search for Multi-Source RL State Encoding

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**多源强化学习（multi-source RL）中的状态编码器设计问题**提出系统性解决方案。在现实场景中，智能体通常需要处理来自多种异构信息源的输入（如传感器数据、时间序列信号、图像观测和文本指令），而传统方法依赖人工设计状态编码器，存在效率低、泛化差等问题。现有的 **Neural Architecture Search (NAS)** 方法大多面向单模态监督任务，难以有效建模多源输入的复合结构，且忽视了模块中间输出所蕴含的表示质量等有用侧信息（side information）。

### 提出的新方法与创新思路
作者提出了名为 **LACER (LLM-driven Neural Architecture Search for Composite State Encoders in RL)** 的新框架，其核心创新如下：

- **形式化“复合神经架构搜索”问题**：将多源状态编码器的设计建模为一个联合优化问题，包含多个源特定模块（source-specific modules）和一个融合模块（fusion module）的协同搜索。
- **引入LLM驱动的NAS流程**：利用大语言模型（LLM）作为先验知识源，指导架构生成过程。通过自然语言提示（prompting），让LLM基于历史性能反馈生成新的候选架构。
- **利用中间输出信号提升样本效率**：不同于传统NAS仅使用最终任务性能指标（如平均速度或回报），LACER额外引入两类关键反馈信号：
  - **Average Reward**：反映RL训练过程中的收敛特性；
  - **Feature Information**：通过互信息（mutual information）和冗余度（redundancy）量化各模块的表示质量，作为模块专业化程度的代理指标。
- **动态迭代式搜索机制**：构建“LLM生成 → RL训练评估 → 反馈更新”的闭环流程，实现高效探索。

### 相比现有方法的优势
| 对比维度 | 传统NAS方法（如DARTS、ENAS、PEPNAS） | LLM-based NAS（如GENIUS） | LACER（本文） |
|--------|----------------------------------|--------------------------|-------------|
| 搜索空间 | 单一网络结构 | 单一网络结构 | 多模块复合结构（composite architecture） |
| 输入模态 | 单模态为主 | 单模态为主 | 支持多源异构输入 |
| 反馈信号 | 最终性能指标（accuracy/speed） | 仅任务metric | 任务metric + average reward + 表示质量（feature info） |
| 先验知识 | 无显式先验 | LLM生成能力 | 利用LLM对模块设计与表示质量的认知先验 |
| 样本效率 | 低（需大量候选评估） | 中等 | 显著更高（更少候选即达高性能） |

---

## 2. 核心实验方法和设置

### 数据集与任务环境
实验主要在一个**混合自主交通控制（mixed-autonomy traffic control）任务**上进行，该任务由Cheng and Jin [5]提出，模拟高速公路上连接自动驾驶车辆（CAVs）与人类驾驶车辆（HDVs）共存的场景（CAV渗透率设为0.9）。每一步观察包含三个独立信息源：
1. **时间上下文（temporal context）**：关键交通指标的时间演化（速度、密度、流量）；
2. **当前交通状态（traffic state）**：车道级密度、速度分布、CAV比例；
3. **车辆序列历史（sequence history）**：过往动作序列分布。

此外，在附录中还扩展至MiniGrid目标导向任务和ManiSkill机器人控制任务以验证通用性。

### 实验设置
- **状态编码器结构**：采用四个模块：
  - 时间编码器（Transformer）
  - 交通编码器（FFN）
  - 序列编码器（Transformer）
  - 融合编码器（FFN）
- **搜索空间设计**：每个模块定义独立的超参数空间（见Table 1），包括维度、层数、注意力头数、激活函数等，总搜索空间约2600万种组合。
- **RL训练配置**：
  - 使用PPO算法；
  - 每个候选架构训练200k步，评估50k步；
  - 固定策略网络结构，仅搜索状态编码器部分。

### 评估指标
- 主要指标：**平均交通速度（average traffic speed）**
- 辅助指标：平均奖励、表示质量（互信息）、收敛稳定性
- 关注**样本效率**：绘制“最佳性能随已评估候选数量变化曲线”，衡量发现高性能架构的速度。

### 基线方法对比
分为三类：
1. **Expert-designed**：领域专家手工设计的固定架构；
2. **传统NAS方法**：
   - DARTS（梯度型）
   - ENAS（RL-based）
   - PEPNAS（进化算法）
3. **LLM-based NAS方法**：
   - GENIUS [18]：使用GPT-4生成单个候选架构，仅反馈任务metric

本文提出两种变体：
- **LACER-1**：每次迭代生成1个候选
- **LACER-5**：每次迭代生成5个候选

所有方法统一评估50个候选架构（批大小不同导致迭代次数差异）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- 如图1(b)所示，在达到相同性能水平时，**LACER仅需约10–15个候选即可超越其他方法50个候选后的表现**。
- 在第50个候选时：
  - **Expert baseline**: ~4.925 km/h
  - **DARTS / ENAS / PEPNAS**: ~4.95–5.00 km/h
  - **GENIUS**: ~5.025 km/h
  - **LACER-1**: **~5.10 km/h**
  - **LACER-5**: **~5.12 km/h**

👉 结论：**LACER显著优于所有基线方法，在更少评估次数下获得更高性能，展现出卓越的样本效率和最终性能上限。**

### 消融实验结果（Ablation Studies）
在附录C中进行了三项消融研究（图12）：
1. **移除Feature Information（FI）**：性能下降明显，说明表示质量信号对引导搜索至关重要；
2. **进一步移除Average Reward（RI）**：性能继续恶化；
3. **再移除Initial Evaluation（IE）**：即不提供初始架构性能作为上下文，性能最差。

📌 发现：**三个组件（FI, RI, IE）共同作用才能实现最优性能，缺一不可**，证明了完整反馈机制的重要性。

### 时间成本分析（图13）
- 所有方法中，**评估时间（evaluation）占主导地位（>97%）**；
- LLM查询时间（query time）占比极小（约1%），不影响整体效率；
- 尽管LACER引入了LLM交互，但其额外开销可忽略不计，真正瓶颈仍是RL训练本身。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **复合NAS是解决多源RL状态编码的有效范式**：将编码器分解为源特定模块+融合模块，并联合优化，能更好适配复杂输入结构。
2. ✅ **LLM可作为强大的架构搜索先验工具**：通过自然语言接口整合人类经验与模型认知，指导高效搜索方向。
3. ✅ **中间输出信号极大提升样本效率**：引入表示质量（feature information）和训练奖励作为反馈，使LLM能够理解“为什么某个架构更好”，从而做出更有针对性的修改。
4. ✅ **LACER在真实交通控制任务上显著优于传统与LLM-based NAS方法**：不仅性能更高，而且所需评估次数更少，具备实际应用潜力。

### 方法的局限性
- 当前依赖于**高质量LLM响应解析**，若LLM输出格式混乱可能导致解析失败（尽管文中采用正则匹配缓解）；
- 需要**精心设计prompt模板与搜索空间描述**，对非专业人士有一定门槛；
- 所有实验均在仿真环境中进行，尚未验证在真实物理系统中的鲁棒性；
- 当前仅支持离散型架构参数搜索，未涉及连续权重共享型NAS。

### 未来工作方向
- 扩展到更多应用场景：如**goal-oriented tasks（MiniGrid）和robotic control（ManiSkill）**；
- 探索**多智能体RL中的分布式状态编码器搜索**；
- 引入**vision-language models（VLMs）** 进一步增强跨模态理解能力；
- 研究如何自动化构建搜索空间与prompt工程，降低人工干预需求；
- 结合**Bayesian optimization 或 bandit算法**与LLM建议，形成混合搜索策略。

---

> 📌 **一句话总结**：  
> LACER首次将LLM与复合NAS结合用于多源RL状态编码，通过引入表示质量等中间信号显著提升了搜索效率与性能，在交通控制等复杂任务中展现出强大优势，为自动化RL表征学习提供了新路径。

</details>

---

### 14. [PlantBiMoE: A Bidirectional Foundation Model with SparseMoE for Plant Genomes](https://arxiv.org/abs/2512.07113)

**Authors**: Kepeng Lin, Qizhe Zhang, Rui Wang, Xuehai Hu, Wei Xu  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.07113v1  

#### Abstract
Understanding the underlying linguistic rules of plant genomes remains a fundamental challenge in computational biology. Recent advances including AgroNT and PDLLMs have made notable progress although, they suffer from excessive parameter size and limited ability to model the bidirectional nature of...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# PlantBiMoE: A Bidirectional Foundation Model with SparseMoE for Plant Genomes — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前植物基因组语言建模领域存在以下关键挑战：
- **模型参数量过大**：如 AgroNT 参数超过 1B，计算资源消耗高，难以部署。
- **单向建模限制**：多数模型（如 PDLLMs）采用单向结构，无法有效捕捉 DNA 双链的对称性和双向依赖关系。
- **上下文长度受限**：如 PlantDNAMamba 仅支持 2,000 bp 上下文，难以建模长距离调控元件（如增强子、染色质可及性区域）。
- **缺乏统一评估基准**：现有任务分散，缺少跨物种、多任务、长序列的综合性 benchmark。

### 🚀 提出的新方法与创新
本文提出 **PlantBiMoE**，一种轻量级且表达能力强的植物基因组基础模型，其核心创新包括：

1. **Bidirectional Mamba (BiMamba)**  
   首次将双向状态空间模型（SSM）引入植物基因组建模，通过前向与反向互补链并行处理，显式建模 DNA 的双链对称性，提升对功能元件（如启动子、剪切位点）的识别能力。

2. **Sparse Mixture-of-Experts (SparseMoE)**  
   在每层中引入稀疏专家结构，仅激活部分专家网络进行推理，显著降低活跃参数数量（从 116M 总参数降至 **64M 活跃参数/token**），在不牺牲容量的前提下提高计算效率。

3. **混合架构设计（BiMamba + SwiGLU + SparseMoE）**  
   借鉴 Jamba 架构，在奇数层使用 BiMamba-SwiGLU，偶数层使用 BiMamba-MoE，实现高效的信息流动与专业化表征学习。

4. **构建 Modified Plants Genome Benchmark (MPGB)**  
   整合 AgroNT 和 PDLLMs 中的任务，构建包含 **31 个子数据集、11 类任务** 的统一 benchmark，覆盖二分类、多分类、回归、分割等任务，输入序列长度范围为 **50–6,000 bp**，更贴近真实应用场景。

### 🔍 相比现有方法的优势
| 特性 | AgroNT | PDLLMs (e.g., PlantDNAMamba) | PlantBiMoE |
|------|--------|-------------------------------|------------|
| 参数规模 | ~1B | ~94M | **116M（总）/64M（活跃）** |
| 上下文长度 | 6,000 bp | 2,000 bp | **32,768 bp** |
| 是否双向建模 | 否（BERT-style 单向） | 多数为单向 | ✅ 显式双向（BiMamba） |
| 是否稀疏激活 | 否 | 否 | ✅ SparseMoE 实现条件计算 |
| 支持长程依赖 | 有限 | 弱 | ✅ 强（理论窗口达 32k bp） |

> ✅ 综上，PlantBiMoE 在保持轻量化的同时实现了更强的表达能力和更优的跨物种泛化性能。

---

## 2. 核心实验方法和设置

### 📚 数据集

#### （1）预训练数据集
- 来源：NCBI 的 42 种代表性植物物种全基因组序列
- 分类：模式植物、蔬菜、水果、谷物、藻类及其他重要作物
- 总核苷酸数：**25.40B bp**
- 预处理流程：
  - 切分为 32,768 bp 固定长度片段（滑动窗口重叠 64–128 bp）
  - 非标准碱基替换为 `N`
  - 过滤含 >2% `N` 的序列
  - 30% 序列进行 reverse complement augmentation
- 训练/测试划分：按染色体划分（5% 作为测试集），确保物种与染色体独立性

#### （2）下游评估基准：Modified Plants Genome Benchmark (MPGB)
| 任务 | 子数据集数 | 任务类型 | 序列长度 | 来源 |
|------|-----------|----------|----------|------|
| Polyadenylation | 6 | 二分类 | 400 | AgroNT |
| Splicing site | 2 | 二分类 | 398 | [14] |
| LncRNA | 6 | 二分类 | 101–6000 | AgroNT |
| Enhancer region | 1 | 二分类 | 1000 | AgroNT |
| Chromatin accessibility | 6 | 多分类（9–19类） | 1000 | [15] |
| Promoter strength | 2 | 回归 | 170 | [16] |
| Terminator strength | 2 | 回归 | 170 | [17] |
| Histone modification | 3 | 二分类 | 100–2000 | PDLLMs |
| Core promoter | 1 | 二分类 | 300 | PDLLMs |
| Conservation | 1 | 二分类 | 1000 | PDLLMs |
| Open chromatin | 1 | 三分类 | 50–2998 | PDLLMs |

> ⚠️ MPGB 是目前最全面的植物基因组 benchmark，涵盖多种生物功能与序列尺度。

---

### 🧪 实验设置与评估指标

#### （1）模型配置
- 层数：16 层
- Embedding 维度：512
- Tokenizer：单碱基分词（A/T/C/G/N/<MASK>/...），词表大小=12
- 预训练任务：Masked Language Modeling (MLM)，mask 15% 位置（80%→`<MASK>`，10%→随机，10%不变）
- 上下文长度：32,768 bp
- 训练硬件：8×Nvidia A800-80G GPU，batch size=256（梯度累积）
- 优化器：AdamW（lr: 0→0.008→0.004，cosine decay），bf16 混合精度
- 总训练时间：约 166 小时（10 epochs）

#### （2）评估指标
| 任务类型 | 主要指标 |
|---------|----------|
| 分类任务（AgroNT 系列） | AUC |
| 分类任务（PDLLMs 系列） | MCC（Matthews Correlation Coefficient） |
| 回归任务（Promoter/Terminator strength） | R² |

#### （3）基线方法对比
- **AgroNT**：基于 BERT 的大规模植物基因组模型（~1B 参数）
- **PlantDNAMamba**：PDLLMs 系列中最优表现的轻量模型（94M 参数，单向）
- 所有 baseline 结果均来自原文报告或在其设定下复现

---

## 3. 主要实验结果和性能指标

### 📊 关键性能汇总（见 Table III）

| 任务 | 指标 | AgroNT | PlantDNAMamba | **PlantBiMoE** | Top-1 次数 |
|------|------|--------|----------------|----------------|-------------|
| Polyadenylation | AUC | 93.62 | 90.22 | **93.63** | ✅ |
| Splicing site | AUC | 99.81 | 99.79 | **99.84** | ✅ |
| LncRNA | AUC | 83.11 | 82.79 | **84.12** | ✅ |
| Enhancer region | AUC | **88.15** | 84.04 | 87.47 | ❌ |
| Chromatin accessibility | AUC | 96.37 | 96.54 | **96.55** | ✅ |
| Promoter strength | R² | 73.85 | 73.35 | **75.23** | ✅ |
| Terminator strength | R² | 71.66 | 69.38 | **72.22** | ✅ |
| Histone modification | MCC | 63.99 | 66.01 | **66.27** | ✅ |
| Core promoter | MCC | 58.74 | **64.41** | 63.20 | ❌ |
| Conservation | MCC | 80.14 | 81.18 | **81.74** | ✅ |
| Open chromatin | MCC | 43.21 | 46.24 | **46.88** | ✅ |

> 💡 **总体表现**：
- 在 **11 项任务中的 9 项取得最佳平均性能**
- 在 **31 个子数据集中赢得 20 个 Top-1 排名**
- 尤其在 **Splicing site, Promoter/Terminator strength, Histone modification, Conservation, Open chromatin** 全面超越所有 baseline

### 🔍 详细分析亮点
- **Polyadenylation**：尽管 PlantDNAMamba 在 4/6 子集胜出，但在 *M. truncatula* 表现极差（69.60 vs AgroNT 94.57, PlantBiMoE 91.35），导致整体拉低；PlantBiMoE 平均 AUC 达 **93.63**，略超 AgroNT。
- **Chromatin accessibility**：Boxplot 显示 PlantBiMoE 在 *Z. mays*, *S. bicolor*, *A. thaliana* 上不仅得分更高，方差更小，表明其 **跨物种稳定性更强**。
- **ROC 曲线分析**（图3）显示 PlantBiMoE 在低假阳性率（FPR）区间具有更强的早期识别能力，说明其预测更具置信度。

### 🧩 消融实验（文中未明确列出，但从设计可推断）
虽然论文未提供正式消融实验表格，但从架构设计可合理推断以下优势来源：
- **BiMamba 贡献**：相比单向 Mamba，能更好捕获启动子上下游对称信号、转录起始区结构特征。
- **SparseMoE 贡献**：减少活跃参数至 64M，显著降低推理成本，同时允许不同专家专注于特定功能区域（如编码区 vs 非编码调控区）。
- **长上下文窗口（32k bp）**：使模型能够建模远端增强子-启动子互作、染色质拓扑结构等长程依赖。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **双向建模 + 稀疏专家机制是高效植物基因组建模的有效范式**  
   PlantBiMoE 证明了在不过度扩大参数的情况下，通过结构创新（BiMamba + SparseMoE）即可实现优于十倍大模型的性能。

2. **长上下文窗口对基因组功能预测至关重要**  
   32,768 bp 的上下文支持使得模型能有效捕捉 cis-regulatory modules 和染色质高级结构的影响。

3. **跨物种泛化能力强**  
   在多个非模式植物（如玉米、大豆、高粱）上的稳定表现，验证了模型良好的迁移能力。

4. **MPGB 成为新的标准 benchmark**  
   提供了一个更全面、更具挑战性的评估平台，推动植物基因组 AI 模型发展。

---

### ⚠️ 局限性
1. **尚未支持 k-mer 或 BPE 分词策略**  
   当前仅使用单碱基 tokenization，可能损失局部 motif 结构信息（相比之下 PlantDNAMamba 支持 6-mer）。

2. **未探索更多 MoE 变体（如 Top-2 routing）**  
   当前采用 Top-1 路由，未来可尝试更复杂的专家组合策略以进一步提升性能。

3. **缺乏对基因编辑或合成生物学的实际应用验证**  
   尽管声称可用于 gene editing 和 synthetic biology，但尚无具体案例支撑。

---

### 🔮 未来工作方向
1. **扩展到其他植物家族或极端环境物种**  
   增强模型在非主流作物中的适用性。

2. **结合三维基因组数据（Hi-C, ChIA-PET）进行联合建模**  
   利用长上下文优势建模染色质空间折叠。

3. **开发面向下游任务的微调框架（如 Prompt Tuning, LoRA）**  
   进一步降低部署门槛。

4. **集成到自动化育种或精准农业系统中**  
   推动从“基础模型”向“实用工具”的转化。

---

## 🔗 开源信息
- **代码地址**：[https://github.com/HUST-Keep-Lin/PlantBiMoE](https://github.com/HUST-Keep-Lin/PlantBiMoE)
- **模型参数量**：116M（总），64M（活跃）
- **预训练数据量**：25.40B nucleotides from 42 species

> 🌿 **总结一句话**：  
> **PlantBiMoE 通过 BiMamba 与 SparseMoE 的协同设计，在极低活跃参数下实现了最先进的植物基因组理解能力，标志着轻量高效基因组 foundation model 的重要进展。**

</details>

---

### 15. [Utilizing Multi-Agent Reinforcement Learning with Encoder-Decoder Architecture Agents to Identify Optimal Resection Location in Glioblastoma Multiforme Patients](https://arxiv.org/abs/2512.06990)

**Authors**: Krishna Arun, Moinak Bhattachrya, Paras Goel  
**Category**: cs.AI  
**Published**: 2025-12-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.06990v1  

#### Abstract
Currently, there is a noticeable lack of AI in the medical field to support doctors in treating heterogenous brain tumors such as Glioblastoma Multiforme (GBM), the deadliest human cancer in the world with a five-year survival rate of just 5.1%. This project develops an AI system offering the only e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Utilizing Multi-Agent Reinforcement Learning with Encoder-Decoder Architecture Agents to Identify Optimal Resection Location in Glioblastoma Multiforme Patients

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该研究针对**胶质母细胞瘤（Glioblastoma Multiforme, GBM）**这一致死率极高的脑肿瘤，在临床诊疗中面临的三大挑战：
- **诊断延迟与误诊率高**（可达28%），导致治疗延误；
- **传统计算模型预测耗时长**（如PDE模型需数百小时），无法支持实时决策；
- **现有AI系统多局限于单一任务**（如仅分割或仅预测），缺乏端到端的个性化治疗规划能力。

### 提出的新方法与创新思路
作者提出名为 **Brainstorm** 的首个**端到端AI系统**，整合诊断与治疗规划两个阶段，采用多智能体强化学习框架（MARL）结合生成模型，实现最优切除位置识别。

#### 主要创新点包括：
- **a) 顺序决策诊断框架（Sequential Decision-Making Framework）**  
  使用由5个轻量级分类模型（CNN + SVM）组成的链式结构，逐步缩小疾病类别范围。相比单一大型模型（如ViT-L/16），显著降低计算成本。

- **b) 新型时空Vision Transformer用于放疗响应建模**  
  设计基于**Encoder-Decoder架构的Spatio-Temporal Vision Transformer**，可对术后多周放疗效果进行高分辨率MRI预测，捕捉非线性时空演化模式。

- **c) 医学现实增强策略提升泛化能力**  
  在预处理中引入模拟低资源医院成像质量的数据增强技术，如**bias field distortion、elastic deformation、Gaussian noise**等，使模型更适应真实世界异质性数据。

- **d) 多智能体强化学习闭环优化机制**  
  利用**Proximal Policy Optimization (PPO)** 构建反馈循环：通过扩散模型模拟手术切除 → Transformer预测放疗进展 → 扩散模型模拟化疗反应 → CNN生存率计算器评估结果 → 若未达目标则迭代调整切除方案，直至找到最优解。

### 相比现有方法的优势
| 维度 | Brainstorm | 现有方法 |
|------|-----------|---------|
| **功能完整性** | 全流程覆盖：诊断 + 手术 + 放疗 + 化疗 + 生存预测 | 多为孤立模块（如仅分割或生长预测） |
| **计算效率** | 单次预测仅需 **9.7秒** | PDE/Lattice Boltzmann等物理模型需 **31–225小时** |
| **临床实用性** | 考虑治疗干预下的肿瘤演变 | 多数模型仅模拟无治疗情况下的自然进展 |
| **部署可行性** | 参数总量<21M，训练成本约$4,600 | ViT-L/16参数超3亿，训练成本>$10万 |

---

## 2. 核心实验方法和设置

### 数据集
共整合来自 **The Cancer Imaging Archive (TCIA)** 的 **6,560例患者** T1CE MRI 扫描及定量放射组学特征，涵盖以下公开数据集：
- **BraTS dataset**（权威脑瘤分割挑战赛）
- **ReMIND dataset**（肿瘤进展预测）
- **Lumiere dataset**（多模态融合研究）

所有图像标准化为 **64×64×32** 分辨率，并进行强度归一化与去噪处理。

### 数据增强策略
为提高在低资源环境中的鲁棒性，引入医学导向的增强方式：
- **Random Bias Field**：模拟磁场不均匀性
- **Elastic Deformation**：模拟术中脑移位
- **Affine Transformations**（旋转、缩放、平移）
- **Gaussian Noise Injection**：模拟老旧设备噪声
> 增强后数据量扩大10倍，显著提升模型泛化能力。

### 实验设置
- **训练平台**：Google Colab T4/A100 GPU
- **分阶段独立训练**：诊断阶段与治疗规划阶段各模型分别训练
- **训练配置见 Table 1**，例如：
  - Resection Diffusion Model：Adam, LR=2.5e-5, 800 epochs
  - Radiotherapy ViT：Adam, LR=1e-4, 40 epochs
  - Survival Calculator：3D ResNet-18 变体，四通道输入（含年龄）

### 评估指标
| 阶段 | 指标 |
|------|------|
| **诊断** | Accuracy, Confusion Matrix (TP/TN/FP/FN), IoU |
| **分割** | Dice Score, IoU |
| **治疗生成** | SSIM（Structural Similarity Index Measure） |
| **整体性能** | Compute Cost ($), Inference Time, Projected Survival Gain |

### 基线方法对比
- **诊断模型** vs. ViT-L/16, ViT-B/16, R50-ViT-L/16
- **分割模型** vs. BraTS 2023 (GAN-based), BraTS 2020 (U-Net)
- **预测模型** vs. Glioma Solver, Lattice Boltzmann, PDE-based frameworks

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 诊断性能（Table 2）
| 模型 | TP (%) | TN (%) | FP (%) | FN (%) |
|------|--------|--------|--------|--------|
| Mass Detection | 92.1 | 96.7 | 7.9 | 3.3 |
| Tumor Confirmation | 92.6 | 90.2 | 8.4 | 8.8 |
| Final Diagnosis (5类) | **99.4** | **99.4** | **0.58** | **0.58** |

> 最终分类准确率达99.4%，远高于早期筛查阶段，体现“逐层过滤”策略的有效性。

#### ✅ 分割性能（Table 3）
| 模型 | Edema | Enhancing Tumor | Tumor Core |
|------|-------|------------------|------------|
| **Brainstorm (nnU-Net)** | **0.9095** | **0.8795** | **0.8653** |
| BraTS 2023 (GAN) | 0.9005 | 0.8673 | 0.8509 |
| BraTS 2020 (U-Net) | 0.8895 | 0.8506 | 0.8203 |

> 在所有子区域均取得SOTA表现，尤其在水肿区优势明显。

#### ✅ 治疗生成性能（SSIM）
| 模型 | SSIM |
|------|------|
| Resection Model | **0.89** |
| Radiotherapy Forecasting | **0.81**（中位数，IQR ±0.03） |
| Chemotherapy Response | **0.82** |

> 表明生成MRI与真实扫描具有高度结构相似性，具备临床可用性。

#### ✅ 计算效率对比（Table 4）
| 模型 | 平均每例时间 | 模拟时间 |
|------|---------------|----------|
| **Brainstorm** | **9.7秒** | **9.7秒** |
| Glioma Solver | 31小时 | 110.5秒 |
| PDE Frameworks | 83小时 | 300秒 |
| Lattice Boltzmann | 225小时 | 850秒 |

> 推理速度提升超过 **10,000倍以上**，适合实时应用。

#### ✅ 成本与精度权衡（Table 5）
| 模型 | 参数量 | 成本 | 准确率 |
|------|--------|------|--------|
| **Brainstorm** | **20.9M** | **~$4,600** | **95.28%** |
| ViT-L/16 | 303.3M | ~$102,420 | 97.08% |
| ViT-B/16 | 87.5M | ~$26,800 | 97.89% |

> Brainstorm以不到 **7%的参数量** 和 **4.5%的成本** 达到接近顶级模型的性能。

### 消融实验结果（文中隐含分析）
- **顺序决策框架 vs 单一模型**：减少计算开销 **22.28×**
- **加入现实风格增强**：DICE分数平均提升 **2.9%**
- **使用Transformer替代RNN**：推理时间减少 **113小时**

---

## 4. 关键结论和发现

### 主要发现
1. **顺序决策诊断框架有效平衡精度与效率**：通过层级过滤机制，在保证最终诊断准确性的同时大幅降低计算负担，适用于资源受限场景。
2. **Spatio-Temporal Vision Transformer显著加速治疗响应预测**：相较传统PDE模型提速上万倍，且能建模复杂时空动态变化。
3. **医学现实增强极大提升模型泛化能力**：特别是在低质量MRI上的稳定性优于标准增强方法。
4. **MARL闭环系统可自动探索最优切除方案**：通过PPO不断优化resection位置，直到满足医生设定的生存目标（±15%阈值内终止）。
5. **临床影响显著**：预计可将GBM五年生存率从 **5.1% 提升至 6%**，每年挽救约 **2,250条生命**；全球患者覆盖率从28.7%提升至34.9%。

### 方法的局限性
- **依赖高质量标注数据**：尤其是术后MRI配对样本用于训练resection diffusion model，现实中获取困难。
- **未考虑个体基因表达差异**：当前模型基于影像+年龄，尚未整合分子层面信息（如IDH突变状态）。
- **强化学习收敛依赖初始策略设计**：若初始切除建议偏差过大，可能导致搜索空间过大、收敛缓慢。
- **尚未在前瞻性临床试验中验证**：目前结果基于回顾性数据分析和医生模拟评估。

### 未来工作方向
- **扩展至其他局部侵袭性癌症**：如乳腺癌、胰腺癌，其治疗路径同样涉及手术+放化疗组合。
- **整合多组学数据**：纳入基因组、转录组信息，构建“影像+生物标志物”联合预测模型。
- **开发轻量化边缘部署版本**：适配移动设备或基层医院本地服务器运行。
- **开展真实世界临床试验验证**：与合作医院联合推进前瞻性队列研究，评估实际疗效提升。

---

> **总结**：本文提出的 **Brainstorm** 系统是首个将 **Multi-Agent RL + Encoder-Decoder Generative Models + Sequential Diagnosis** 融合应用于GBM全流程管理的端到端AI解决方案，在**准确性、速度、成本、临床实用性**四个方面全面超越现有技术，展现出巨大的转化潜力和公共健康价值。

</details>

---

### 16. [UniDiff: A Unified Diffusion Framework for Multimodal Time Series Forecasting](https://arxiv.org/abs/2512.07184)

**Authors**: Da Zhang, Bingyu Li, Zhuyuan Zhao, Junyu Gao, Feiping Nie, Xuelong Li  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.07184v1  

#### Abstract
As multimodal data proliferates across diverse real-world applications, leveraging heterogeneous information such as texts and timestamps for accurate time series forecasting (TSF) has become a critical challenge. While diffusion models demonstrate exceptional performance in generation tasks, their ...

---

### 17. [JT-DA: Enhancing Data Analysis with Tool-Integrated Table Reasoning Large Language Models](https://arxiv.org/abs/2512.06859)

**Authors**: Ce Chi, Xing Wang, Zhendong Wang, Xiaofan Liu, Ce Li, Zhiyan Song, Chen Zhao, Kexin Yang, Boshen Shi, Jingjing Yang, Chao Deng, Junlan Feng  
**Category**: cs.AI  
**Published**: 2025-12-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.06859v1  

#### Abstract
In this work, we present JT-DA-8B (JiuTian Data Analyst 8B), a specialized large language model designed for complex table reasoning tasks across diverse real-world scenarios. To address the lack of high-quality supervision in tabular reasoning scenarios, we construct a comprehensive and diverse tra...

---

### 18. [Bridging Code Graphs and Large Language Models for Better Code Understanding](https://arxiv.org/abs/2512.07666)

**Authors**: Zeqi Chen, Zhaoyang Chu, Yi Gui, Feng Guo, Yao Wan, Chuan Shi  
**Category**: cs.CL  
**Published**: 2025-12-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.07666v1  

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable performance in code intelligence tasks such as code generation, summarization, and translation. However, their reliance on linearized token sequences limits their ability to understand the structural semantics of programs. While prior studies...

---

### 19. [gp2Scale: A Class of Compactly-Supported Non-Stationary Kernels and Distributed Computing for Exact Gaussian Processes on 10 Million Data Points](https://arxiv.org/abs/2512.06143)

**Authors**: Marcus M. Noack, Mark D. Risser, Hengrui Luo, Vardaan Tekriwal, Ronald J. Pandolfi  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.06143v1  

#### Abstract
Despite a large corpus of recent work on scaling up Gaussian processes, a stubborn trade-off between computational speed, prediction and uncertainty quantification accuracy, and customizability persists. This is because the vast majority of existing methodologies exploit various levels of approximat...

---

### 20. [PMA-Diffusion: A Physics-guided Mask-Aware Diffusion Framework for TSE from Sparse Observations](https://arxiv.org/abs/2512.06183)

**Authors**: Lindong Liu, Zhixiong Jin, Seongjin Choi  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.06183v1  

#### Abstract
High-resolution highway traffic state information is essential for Intelligent Transportation Systems, but typical traffic data acquired from loop detectors and probe vehicles are often too sparse and noisy to capture the detailed dynamics of traffic flow. We propose PMA-Diffusion, a physics-guided ...

---

### 21. [Neural expressiveness for beyond importance model compression](https://arxiv.org/abs/2512.06440)

**Authors**: Angelos-Christos Maroudis, Sotirios Xydis  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.06440v1  

#### Abstract
Neural Network Pruning has been established as driving force in the exploration of memory and energy efficient solutions with high throughput both during training and at test time. In this paper, we introduce a novel criterion for model compression, named "Expressiveness". Unlike existing pruning me...

---

### 22. [KV-CAR: KV Cache Compression using Autoencoders and KV Reuse in Large Language Models](https://arxiv.org/abs/2512.06727)

**Authors**: Sourjya Roy, Shrihari Sridharan, Surya Selvam, Anand Raghunathan  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.06727v1  

#### Abstract
As Large Language Models (LLMs) scale in size and context length, the memory requirements of the key value (KV) cache have emerged as a major bottleneck during autoregressive decoding. The KV cache grows with sequence length and embedding dimension, often exceeding the memory footprint of the model ...

---

### 23. [PINE: Pipeline for Important Node Exploration in Attributed Networks](https://arxiv.org/abs/2512.07244)

**Authors**: Elizaveta Kovtun, Maksim Makarenko, Natalia Semenova, Alexey Zaytsev, Semen Budennyy  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.07244v1  

#### Abstract
A graph with semantically attributed nodes are a common data structure in a wide range of domains. It could be interlinked web data or citation networks of scientific publications. The essential problem for such a data type is to determine nodes that carry greater importance than all the others, a t...

---

### 24. [LightSearcher: Efficient DeepSearch via Experiential Memory](https://arxiv.org/abs/2512.06653)

**Authors**: Hengzhi Lan, Yue Yu, Li Qian, Li Peng, Jie Wu, Wei Liu, Jian Luan, Ting Bai  
**Category**: cs.AI  
**Published**: 2025-12-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.06653v1  

#### Abstract
DeepSearch paradigms have become a core enabler for deep reasoning models, allowing them to invoke external search tools to access up-to-date, domain-specific knowledge beyond parametric boundaries, thereby enhancing the depth and factual reliability of reasoning. Building upon this foundation, rece...

---

### 25. [Cognitive Control Architecture (CCA): A Lifecycle Supervision Framework for Robustly Aligned AI Agents](https://arxiv.org/abs/2512.06716)

**Authors**: Zhibo Liang, Tianze Hu, Zaiye Chen, Mingjie Tang  
**Category**: cs.AI  
**Published**: 2025-12-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.06716v1  

#### Abstract
Autonomous Large Language Model (LLM) agents exhibit significant vulnerability to Indirect Prompt Injection (IPI) attacks. These attacks hijack agent behavior by polluting external information sources, exploiting fundamental trade-offs between security and functionality in existing defense mechanism...

---

### 26. [Parameter-Efficient Fine-Tuning with Differential Privacy for Robust Instruction Adaptation in Large Language Models](https://arxiv.org/abs/2512.06711)

**Authors**: Yulin Huang, Yaxuan Luan, Jinxu Guo, Xiangchen Song, Yuchen Liu  
**Category**: cs.CL  
**Published**: 2025-12-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.06711v1  

#### Abstract
This study addresses the issues of privacy protection and efficiency in instruction fine-tuning of large-scale language models by proposing a parameter-efficient method that integrates differential privacy noise allocation with gradient clipping in a collaborative optimization framework. The method ...

---

### 27. [Quantization Blindspots: How Model Compression Breaks Backdoor Defenses](https://arxiv.org/abs/2512.06243)

**Authors**: Rohan Pandey, Eric Ye  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.06243v1  

#### Abstract
Backdoor attacks embed input-dependent malicious behavior into neural networks while preserving high clean accuracy, making them a persistent threat for deployed ML systems. At the same time, real-world deployments almost never serve full-precision models: post-training quantization to INT8 or lower...

---

### 28. [Less is More: Non-uniform Road Segments are Efficient for Bus Arrival Prediction](https://arxiv.org/abs/2512.07200)

**Authors**: Zhen Huang, Jiaxin Deng, Jiayu Xu, Junbiao Pang, Haitao Yu  
**Category**: cs.LG  
**Published**: 2025-12-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.07200v1  

#### Abstract
In bus arrival time prediction, the process of organizing road infrastructure network data into homogeneous entities is known as segmentation. Segmenting a road network is widely recognized as the first and most critical step in developing an arrival time prediction system, particularly for auto-reg...

---

### 29. [Decouple to Generalize: Context-First Self-Evolving Learning for Data-Scarce Vision-Language Reasoning](https://arxiv.org/abs/2512.06835)

**Authors**: Tingyu Li, Zheng Sun, Jingxuan Wei, Siyuan Li, Conghui He, Lijun Wu, Cheng Tan  
**Category**: cs.AI  
**Published**: 2025-12-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.06835v1  

#### Abstract
Recent vision-language models (VLMs) achieve remarkable reasoning through reinforcement learning (RL), which provides a feasible solution for realizing continuous self-evolving large vision-language models (LVLMs) in the era of experience. However, RL for VLMs requires abundant high-quality multimod...

---

### 30. [ProSocialAlign: Preference Conditioned Test Time Alignment in Language Models](https://arxiv.org/abs/2512.06515)

**Authors**: Somnath Banerjee, Sayan Layek, Sayantan Adak, Mykola Pechenizkiy, Animesh Mukherjee, Rima Hazra  
**Category**: cs.CL  
**Published**: 2025-12-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.06515v1  

#### Abstract
Current language model safety paradigms often fall short in emotionally charged or high-stakes settings, where refusal-only approaches may alienate users and naive compliance can amplify risk. We propose ProSocialAlign, a test-time, parameter-efficient framework that steers generation toward safe, e...

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

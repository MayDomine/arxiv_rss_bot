# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-16 10:21:00 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [End-to-End Latency-Minimizing and Load-Balanced Request Scheduling for Edge LLM Inference in Agentic AI Services](https://arxiv.org/abs/2609.17193)

**Authors**: Zhen Li, Jun Cai, Haoran Gao, An Li, Tan Li  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.17193v1  

#### Abstract
Large language model (LLM)-powered agentic AI services increasingly demand low-latency inference, motivating the deployment of LLMs across distributed edge servers. However, heterogeneous communication and computing capabilities, together with dynamically evolving inference states, make the edge ser...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：End-to-End Latency-Minimizing and Load-Balanced Request Scheduling for Edge LLM Inference in Agentic AI Services

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **Agentic AI 服务中的边缘大语言模型（LLM）推理请求调度** 问题，旨在同时实现两个目标：
- 最小化长期平均 **end-to-end latency**
- 在异构边缘服务器之间维持良好的 **load balancing**

该问题面临三大挑战：
1. **多阶段、跨时隙的 LLM 推理动态**：LLM 推理包含 prefill 和 decoding 阶段，且 decoding 是迭代进行的，导致请求可能跨越多个时间槽（cross-slot），决策影响延迟反馈。
2. **负载度量不准确**：传统基于请求数量或队列长度的负载指标无法反映不同输入/输出长度请求对 KV cache 资源的实际占用差异。
3. **延迟反馈与长期约束耦合**：调度决策的效果在请求完成后才可观测，而 load balancing 是一个长期约束，难以通过即时奖励优化。

---

### 提出的新方法：LYREO
作者提出了一种名为 **LYREO（Lyapunov-guided Reward-redistribution Online request scheduling）** 的新型在线调度框架，其核心创新包括：

#### （1）细粒度 LLM 推理建模
- 显式建模了从 **wireless transmission → batching → prefill → iteration-level decoding** 的全流程。
- 引入 **KV cache evolution model**，跟踪每个解码迭代中由活跃请求累积的 KV cache 占用情况，精确刻画资源竞争。

#### （2）基于 KV Cache 的负载度量
- 定义了一个新的 **normalized KV cache memory-time consumption** 指标来衡量服务器负载：
  $$
  \eta_s(t) = \frac{\sum_{k=1}^{K_s(t)} m_s(k,t) \cdot T_s(k,t)}{M_s \cdot \Delta}
  $$
  该指标结合了内存占用强度和持续时间，适用于异构服务器环境。

#### （3）Lyapunov 优化 + Reward Redistribution 联合机制
- 使用 **Lyapunov virtual queue** 将长期 load-balancing 约束转化为每时隙可优化的惩罚项，实现自适应调节。
- 设计 **reward redistribution 机制**，利用 LSTM 预测序列回报，并将延迟的端到端延迟反馈重新分配给早期调度决策，提供及时的学习信号。

---

### 相比现有方法的优势
| 方面 | LYREO 的优势 |
|------|--------------|
| **系统建模** | 比传统“原子任务”假设更真实，捕捉了 LLM 的跨时隙、状态保持特性 |
| **负载感知** | 使用 KV cache memory-time 度量，优于简单的请求数或队列长度 |
| **学习效率** | Reward redistribution 解决了延迟反馈问题，加速策略收敛 |
| **约束处理** | Lyapunov 框架能动态响应累积负载偏差，而非固定惩罚 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用真实世界对话数据集 **LMSYS-Chat-1M** 生成推理请求。
- 输入 token 数均值为 70，输出 token 数均值为 215。
- 每个 token 大小设为 16 bits（传输）和 16 KB（KV cache）。

### 实验设置
- **边缘服务器数量**：S = 8
- **GPU 类型**：NVIDIA RTX 3080（10GB）、RTX 3090（24GB）、RTX 4090（24GB），构成异构环境。
- **部署模型**：Llama-3.2-1B
- **时间槽长度 Δ**：1 秒
- **总时隙数 T**：800
- **通信参数**：带宽 [6,14] MHz，发射功率 [15,25] dBm，噪声 PSD -174 dBm/Hz
- **用户行为**：U 个移动用户随机分布在 200×200 m² 区域内，每时隙产生一个请求。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Average End-to-End Latency** | 所有请求的平均响应延迟 |
| **P99 End-to-End Latency** | 第 99 百分位延迟，反映尾部性能 |
| **Load-Balancing Deviation** | 各服务器归一化负载与系统均值的方差（长期平均） |
| **High-KV Ratio** | KV cache 利用率超过 90% 的时隙占比 |

### 基线方法对比
分为四类共 6 个基线：
1. **DRL-based**:
   - **PPO**：标准 PPO，负载偏差作为惩罚项
   - **Ly-PPO**：集成 Lyapunov 但无 reward redistribution
2. **Heuristic**:
   - **Least-Loaded (LL)**：选择当前负载最低的服务器
   - **Random**：随机分配
3. **Static Batching**:
   - **Static**：固定批处理，不支持动态加入/退出

---

## 3. 主要实验结果和性能指标

### 性能对比（Table II 及 Figures）

| 方法 | 平均延迟 (s) | 负载偏差 (×10⁻²) | P99 延迟 (s) | High-KV 比例 (%) |
|------|----------------|--------------------|---------------|------------------|
| **LYREO (ours)** | **2.63** | **0.8** | **3.74** | **6.42** |
| LYREO-Count | 2.79 (+6.1%) | 1.1 (+37.5%) | 4.56 (+21.9%) | 8.04 (+25.2%) |
| LYREO-NoLB | 2.02 (-23.2%) | 1.8 (+125.0%) | 6.73 (+79.9%) | 14.35 (+123.5%) |
| Ly-PPO | ~3.9 | ~0.09 | — | — |
| PPO | ~5.2 | ~0.22 | — | — |

> 注：数据来自图 2、图 3 和表 II；部分为估算值。

---

### 关键对比结果
- **相比 Ly-PPO**：
  - 平均延迟降低约 **33%**（2.63s vs 3.9s）
  - 负载偏差减少 **~11% 绝对值**（0.8 vs 0.09）
  - 收敛更快、更稳定（见 Fig. 2）
- **相比 PPO**：
  - 延迟显著更低（2.63s vs 5.2s），说明 Lyapunov 框架有效引导长期平衡
- **相比非学习方法**：
  - 明显优于 Static、LL 和 Random，在高负载下优势更大

---

### 消融实验结果
#### （1）**LYREO-NoLB**（移除负载均衡）
- 虽然平均延迟最低（2.02s），但：
  - 负载偏差上升 **125%**
  - P99 延迟飙升至 **6.73s（+79.9%）**
  - High-KV 比例达 **14.35%**
→ 表明：**仅优化延迟会导致严重负载集中，损害尾延迟和服务稳定性**

#### （2）**LYREO-Count**（用请求数代替 KV memory-time）
- 所有指标均劣于 LYREO：
  - 延迟 ↑6.1%，P99 ↑21.9%，负载偏差 ↑37.5%
→ 表明：**KV cache memory-time 是更合理的负载度量方式**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **延迟与负载需联合优化**：单纯最小化延迟会引发严重的负载倾斜，反而恶化尾延迟（tail latency）。
2. ✅ **KV cache memory-time 是有效的负载指标**：它能综合反映请求的内存强度与时长，优于简单计数。
3. ✅ **Reward redistribution 显著提升学习效果**：通过将延迟反馈归因于早期决策，解决了 DRL 中的 delayed reward 问题。
4. ✅ **Lyapunov 框架适合处理长期约束**：虚拟队列能自适应地调节负载不平衡压力，优于静态惩罚项。
5. ✅ **LYREO 在多种配置下鲁棒性强**：在用户数增加、服务器数变化、硬件异构性增强等场景下均表现最优。

---

### 方法的局限性
- **依赖 LSTM 回报预测器**：引入额外训练开销，且 truncation length H 需调参。
- **未考虑模型并行或多副本部署**：假设每个边缘节点运行完整 LLM 实例。
- **无线信道建模较简化**：采用静态路径损耗模型，未模拟突发干扰或多径衰落。
- **无限视界设定**：实际系统可能更适合有限窗口滚动优化。

---

### 未来工作方向
1. 扩展至 **multi-modal LLM** 或 **Mixture-of-Experts (MoE)** 架构下的调度。
2. 结合 **model offloading** 与 **request scheduling** 进行联合优化。
3. 引入 **uncertainty-aware scheduling**，应对请求长度估计误差。
4. 探索 **federated 或 decentralized 版本**，支持去中心化边缘协作。
5. 将 reward redistribution 机制推广至其他具有延迟反馈的边缘智能任务。

--- 

> **总结**：LYREO 是首个将 **fine-grained LLM execution dynamics**、**KV-aware workload characterization** 与 **delayed-feedback-aware policy learning** 相结合的边缘 LLM 请求调度框架，在保证低延迟的同时实现了真正的长期负载均衡，为 Agentic AI 的高效边缘部署提供了重要解决方案。

</details>

---

### 2. [ECHO: Early-layer Collaborative Hierarchical Orchestration with Bonus Logits in Speculative Decoding](https://arxiv.org/abs/2609.17241)

**Authors**: Ziyang Ma, Zihong Zhang, Zuchao Li, Lefei Zhang, Baoyuan Qi, Siqi Li, Simin Yu  
**Category**: cs.CL  
**Published**: 2026-09-16  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.17241v1  

#### Abstract
While draft-model-free speculative decoding offers a promising path to efficient LLM inference, it is frequently constrained by stale draft candidates and the high computational cost of the verification. To address these challenges, we propose ECHO, a hierarchical dual-loop framework that exploits t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ECHO: Early-layer Collaborative Hierarchical Orchestration with Bonus Logits in Speculative Decoding  
**论文核心总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Speculative Decoding**（推测解码）在提升 LLM 推理效率方面具有潜力，但面临两大瓶颈：
- **Draft Candidate Staleness**（草案候选陈旧）：draft-model-free 方法依赖历史上下文或最终层 logits 构建草案，但这些信息需完整前向传播才能更新，导致信息滞后。
- **Verification Wall**（验证墙）：每个草案序列必须通过整个目标模型进行验证，计算开销大，尤其当草案质量波动时，加速效果不稳定。

### 🚀 提出的新方法：ECHO
提出 **ECHO** —— 一种无需额外参数、基于层级双循环的推测解码框架，其核心创新包括：

#### （1）**Hierarchical Dual-Loop Verification**（分层双循环验证）
将目标模型 $M$ 划分为两个部分：
- **早期层验证器 $V_e$**（Early-layer Verifier）：负责轻量级、高频的草案树筛选（Inner Loop）。
- **后续层验证器 $V_s$**（Subsequent-layer Verifier）：执行权威性的全模型因果验证（Outer Loop），确保分布对齐。

利用 **state-reuse** 机制跳过重复计算，显著降低验证成本。

#### （2）**Bonus Logits 驱动的动态草案构建**
引入两种“奖励 logits”作为高价值生成信号：
- **Early-layer Bonus Logits**：来自 $V_e$ 的输出，提供低成本、高频率的语义引导，用于快速扩展草案树。
- **Final-layer Bonus Logits**：来自 $V_s$ 的输出，提供高保真度锚点，修正路径并补充高质量候选。

实现从静态匹配到 **动态语义预测** 的转变。

#### （3）**Logits-Driven Tree Construction**
首次将 bonus logits 主动用于驱动 **draft tree** 的生长与刷新，结合 retrieval-based 与 logits-based 方法，形成混合草案策略。

---

### 🔍 相比现有方法的优势
| 特性 | ECHO | 其他方法（如 PLD, TokenRecycling, LayerSkip） |
|------|------|---------------------------------------------|
| 是否引入额外参数 | ❌ 否 | ✅ 是（如 Medusa, Eagle）或 ❌（但效率低） |
| 是否打破验证墙 | ✅ 是（分层验证 + state reuse） | ❌ 否（所有草案均走完整模型） |
| 草案信息时效性 | ✅ 高频更新（early bonus logits） | ❌ 滞后（依赖最终层输出） |
| 加速稳定性 | ✅ 高（维持高质量候选密度） | ⚠️ 波动大（受 draft quality 影响） |
| 工程开销 | ✅ 极低（无额外部署负担） | ⚠️ 或高（需定制架构/缓存管理） |

> 💡 **一句话总结优势**：ECHO 在不增加任何部署参数的前提下，通过功能不对称的层级协作，实现了高效、稳定且可扩展的推测解码。

---

## 2. 核心实验方法和设置

### 📚 数据集
在三个代表性基准上评估：
- **Spec-Bench**：通用任务综合评测集
- **HumanEval**：代码生成能力测试
- **GSM8K**：数学推理任务

### 🧪 模型
覆盖多种主流 LLM 架构与规模：
- Llama-2-7B / 13B
- Llama-3-8B
- CodeLlama-7B
- 扩展至更大模型：CodeLlama-34B 和 Llama2-70B

采用 Meta 发布的 **LayerSkip** 架构作为基础模型（支持 early exit，无需额外参数）。

### 📊 评估指标
- **Speedup Ratio**：相对于标准自回归解码的速度提升倍数。
- **Mean Accepted Tokens (MAT)**：单次推测步骤中平均被接受的 token 数量，反映草案质量。

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| Retrieval-based | PLD, SuffixDecoding |
| Logits-based | TokenRecycling, LogitSpec |
| Early-exit Self-Speculative | LayerSkip, Self-Speculative |
| 参数增强型 | EAGLE-2（含辅助模块） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）

| Model | Method | Speedup | MAT |
|-------|--------|---------|-----|
| Llama-2-7B | ECHO (Ours) | **2.71×** | **5.07** |
| Llama-3-8B | ECHO (Ours) | **2.42×** | **4.33** |
| Llama-2-13B | ECHO (Ours) | **2.90×** | **4.16** |
| CodeLlama-7B | ECHO (Ours) | **2.60×** | **4.78** |

> ✅ **最高提速达 3.01×（Llama-2-13B）**，MAT 显著高于所有 baseline。

---

### 🔁 与基线方法对比结果
- 在 **Spec-Bench** 上，ECHO 平均 MAT 达 **5.55**，远超第二名 TokenRecycling（2.98）。
- 在 **HumanEval** 上，ECHO 实现 **3.11×** 加速，优于 LogitSpec（2.81×）。
- 即使在挑战性高的 **GSM8K** 上，仍保持 **2.72–2.92×** 加速。
- 与参数增强方法 **EAGLE-2** 对比（Table 3）：
  - ECHO 实现更高平均速度（**3.79× vs 2.84×**）
  - 更高 MAT（**4.84 vs 3.37**）
  - **零额外参数、零 VRAM 开销**

---

### 🔬 消融实验结果（Ablation Study, Table 2）

| Configuration | Speedup ↓ | MAT ↓ |
|--------------|-----------|--------|
| ECHO (Full) | 2.70× | 5.34 |
| w/o Outer Automaton Update | 2.29× | 5.06 |
| w/o Inner Automaton Update | 2.63× | 5.30 |
| w/o Bonus Logits | 2.16× | 5.31 |
| w/o Retrieval Draft Tree | 2.44× | 5.00 |

> 🔍 **关键发现**：
- 移除 **Bonus Logits** 导致速度下降最严重（-20%），说明其对高效探索至关重要。
- 外层 automaton 更新是结构性支柱，缺失会削弱长期记忆引导能力。
- 内部 automaton 支持高频状态积累，提升局部连贯性。

---

### 📉 分层验证有效性分析（Figure 5）
- 在 512-token 生成任务中：
  - 非分层方式需约 **8000 层传递**
  - ECHO 仅需 **~6000 层传递**
  - **减少 21% 总层数计算量**
  - **执行时间降低 12%，中位数 FLOPs 减少 17.4%**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Transformer 层间存在功能性不对称**：早期层具备强判别能力，适合高频草案筛选；末层保证分布准确性。
2. **Bonus Logits 是打破信息衰减的关键**：复用 early/final bonus logits 可持续注入高置信度先验，避免草案质量随深度迅速下降（见 Figure 1）。
3. **分层验证有效破解“验证墙”**：通过 state-reuse 将昂贵的 full-model 验证延迟到后期，大幅降低平均验证成本。
4. **无需额外参数即可实现 SOTA 加速**：ECHO 在零新增参数条件下，超越多数需要结构修改的方法。

---

### ⚠️ 局限性
1. **依赖 one-shot fine-tuning**：为最优加速效果，需对模型进行 early-exit 微调（如 LayerSkip）。虽然可在 vanilla 模型运行（见 Table 10），但性能略降。
2. **本地部署优化为主**：当前设计侧重于单机场景，在分布式或多节点环境中的负载均衡与通信隐藏尚未深入探索。
3. **缓冲区大小敏感**：过大的 inner-loop buffer 会导致早期草案偏离目标分布，引发更多拒绝（见 Table 7）。

---

### 🔮 未来工作方向
1. **扩展至多级级联架构**（Multi-level Cascade）：探索三层及以上验证结构，适用于超大规模模型（>70B）。
2. **异步分布式实现**：让 inner-loop 在多个设备上并行预生成，异步验证以隐藏通信延迟。
3. **自动化 exit-layer 选择机制**：根据输入动态调整 $L_{exit}$，平衡敏捷性与准确性。
4. **兼容 vanilla 权重的进一步优化**：提升未微调模型上的 early-layer logits 对齐度。

---

## 📌 总结
**ECHO** 是一种训练免费、无额外参数的推测解码框架，通过 **层级双循环 + Bonus Logits** 的协同机制，成功解决了 draft staleness 与 verification wall 两大难题。实验证明其在多种模型和任务上均取得 **2.4×–3.0× 的加速比**，显著优于现有 SOTA 方法，为高效 LLM 推理提供了新的范式。

</details>

---

### 3. [FlexEE: Self-Speculative and KV-Compatible Early Exiting for Offloading-Aware LLM Inference](https://arxiv.org/abs/2609.17008)

**Authors**: Qihu Xie, Ziwei Li, Yi Kang  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.17008v1  

#### Abstract
Large language model (LLM) inference is often constrained by both computation and memory, especially in offloading-based deployments where model weights are transferred across memory hierarchies during autoregressive decoding. In this setting, reducing the number of executed layers can lower per-tok...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《FlexEE: Self-Speculative and KV-Compatible Early Exiting for Offloading-Aware LLM Inference》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在基于**权重卸载（weight offloading）**的大型语言模型（LLM）推理场景中，传统早期退出（early exiting）方法面临三大挑战：
1. **中间层语义解码能力弱**：标准LLM仅在最终层进行训练以生成输出，中间层隐藏状态难以可靠预测token，导致直接应用早期退出会严重降低生成质量。
2. **退出决策开销高**：现有方法需将中间隐藏状态通过完整的语言建模头（LM head）投影到全词表空间（如32K维），计算成本高昂，抵消了跳过层数带来的加速收益。
3. **KV缓存不兼容**：早期退出破坏了自回归解码所需的Key-Value（KV）缓存构建流程，后续层无法获得正确的KV对，影响后续token生成。

### **提出的新方法与创新思路**
作者提出了 **FlexEE** ——一种面向卸载感知、系统-算法协同设计的高效早期退出框架，其核心创新包括：

#### ✅ **1. 自推测轻量级退出预测器（Self-Speculative Lightweight Predictor）**
- 引入一个**自推测层（self-speculation layer）**，将其Top-K（如K=30）候选token作为局部词表。
- 后续各层的退出决策仅在此小规模词表上进行，避免重复执行全词表的LM head矩阵乘法。
- 使用 **Top Prob** 和 **Gap**（top-1与top-2概率差）作为可靠的退出特征，并设定分层阈值，确保退出预测准确率 > 99%。

#### ✅ **2. 动态隐藏状态管理（Dynamic Hidden State Management）**
- 当某token在中间层退出时，保留该层的隐藏状态。
- 在下一个token前向传播至对应层时，将当前表示与缓存的状态融合更新KV缓存，之后清除缓存。
- 避免了传统方案中的“强制复制所有后续层KV”操作，在保证语义连续性的同时显著减少内存访问开销。

#### ✅ **3. 卸载感知的协同优化设计**
- 整体设计充分考虑了**CPU-GPU间权重加载延迟远高于计算时间**的实际瓶颈。
- 通过减少执行层数 + 减少预测开销 + 避免冗余KV更新，有效降低端到端延迟，尤其适用于资源受限边缘设备。

### **相比现有方法的优势**
| 方面 | FlexEE优势 |
|------|-----------|
| **算法效率** | 采用局部词表预测，大幅降低每层退出判断的FLOPs（见Table 7） |
| **系统兼容性** | 支持KV缓存正确性，无需加载未执行层的权重，真正实现“跳过即省” |
| **通用性** | 不依赖额外训练，可部署于已具备layer-wise early-exit supervision的预训练模型（如LayerSkip） |
| **性能增益** | 在不同卸载比例下均取得显著端到端加速，最高达3.16× |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **语言建模任务**：WikiText
- **指令跟随与开放生成**：Alpaca
- **常识推理**：MMLU、CommonsenseQA、BoolQ、PIQA、WinoGrande、ARC-e/c
- **阅读理解**：TriviaQA、RACE
- **数学与代码**：MathQA、MBPP
- **综合评测**：AlpacaEval-LC、MT-Bench（GPT-4 judge打分）

### **实验设置**
- **模型**：
  - Llama2-7B / Llama2-13B
  - Llama3-8B / Llama3.2-1B
  - 所有模型均基于 **LayerSkip** 提供的带有layer-wise early-exit supervision的checkpoint
- **硬件平台**：
  - GPU: NVIDIA RTX PRO 6000 (96GB VRAM)
  - CPU: Intel Xeon Platinum 8470Q (3.8GHz)
- **框架基础**：基于 **FlexGen** 实现权重跨内存层级卸载，集成自定义后端支持动态隐藏状态管理
- **卸载比例**：0%、25%、50%、75% 层卸载至CPU内存

### **评估指标**
| 类别 | 指标 |
|------|------|
| **准确性** | Perplexity (PPL)，Accuracy (%)，AlpacaEval-LC Win Rate，MT-Bench Score |
| **效率** | Throughput (tokens/s)，Speedup (×)，FLOPs ratio，Average Exit Layer |
| **系统开销** | 额外FLOPs per token，内存占用分析 |

### **基线方法对比**
| 方法 | 简介 |
|------|------|
| **ShortGPT** | 基于层相似性的结构化剪枝方法 |
| **AdaInfer** | 利用运行时统计信息进行无训练退出决策 |
| **SpecEE** | 结合投机式解码的早期退出方法 |
| **LayerSkip** | 使用浅层进行投机解码并提前终止 |
| **Baseline (No EE)** | 无早期退出的标准推理 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **端到端吞吐提升（Table 3）**
| 模型 | 卸载比例 | Baseline (tokens/s) | FlexEE (tokens/s) | Speedup |
|------|----------|---------------------|--------------------|---------|
| Llama2-7B | 0% | 48.0 | 60.9 | **1.27×** |
|           | 50% | 6.9  | 21.8 | **3.16×** |
|           | 75% | 4.9  | 16.1 | **3.29×** |
| Llama3-8B | 0% | 38.9 | 48.6 | **1.25×** |
|           | 50% | 6.3  | 17.8 | **2.83×** |

> ⚡️ 可见随着卸载比例增加，FlexEE的加速效果越明显，说明其特别适合内存受限场景。

#### 🔹 **平均执行深度减少（Table 1）**
| 模型 | 自推测层 | 平均退出层 | 名义深度缩减 |
|------|----------|------------|--------------|
| Llama2-7B | #8 | 22.8 | ~28.8% |
|           | #16 | 24.9 | ~22.2% |
| Llama3-8B | #16 | 26.2 | ~18.1% |

> 表明大部分token可在非最终层安全退出。

#### 🔹 **开放生成质量（Table 4）**
| 模型 | 方法 | AlpacaEval-LC Win Rate vs Dense | MT-Bench Score |
|------|------|-------------------------------|---------------|
| Llama2-7B | Dense | Reference | 4.92 |
|            | FlexEE (greedy) | 49.1% | 4.88 |
| Llama3-8B | Dense | Reference | 6.41 |
|            | FlexEE (greedy) | 50.3% | 6.39 |

> 👉 生成质量几乎无损，胜率接近50%，表明用户难以区分。

---

### **与基线方法对比结果（Table 2）**

| 方法 | Llama2-7B PPL ↓ | FLOPs ↓ | △Acc (MMLU) ↑ |
|------|------------------|--------|----------------|
| AdaInfer | 319 | 0.65 | -1.57 |
| SpecEE | 135 | 0.74 | -0.66 |
| **FlexEE** | **6.4** | **0.69** | **-0.20** |

✅ FlexEE在保持最低准确损失的同时实现了更高的压缩比（更低FLOPs）和更优的语言建模性能（更低PPL）。

---

### **消融实验结果**

#### 🔹 **自推测 vs 全词表退出（Figure 8 & 9）**
- 使用 **Top-K局部词表** 能触发更早退出且维持相近性能；
- 全词表方法因阈值保守，退出更晚，吞吐反而更低；
- 在卸载场景下，自推测优势更加明显。

#### 🔹 **批处理下的表现（Table 5）**
| 卸载 | Batch Size | Without EE | With FlexEE |
|------|------------|-------------|--------------|
| 0%   | B=1        | 48.0        | 60.9         |
|      | B=8        | 42.7        | 48.1         |
| 50%  | B=1        | 6.9         | 21.8         |
|      | B=8        | 6.4         | 10.7         |

> 💡 小批量（尤其是B=1）时增益最大，符合交互式服务典型场景。

#### 🔹 **Top-K大小的影响（Figure 13）**
- K=30 已达到全词表质量的 **98.67%**；
- 继续增大至K=200仅提升0.33%，但平均退出层推迟0.29层；
➡️ 支持选择较小K实现高效权衡。

---

## **4. 关键结论和发现**

### **主要发现**
1. **中间层可通过监督增强语义一致性**：layer-wise early-exit supervision 显著提升了中间层的可解释性和预测可靠性（图3）。
2. **局部词表预测是高效退出的关键**：Full-vocabulary projection 是主要开销来源，限制了实际加速；而Top-K自推测机制能以极低代价实现高精度退出判断。
3. **KV缓存冲突必须系统级解决**：简单复制或掩码策略无法根本解决问题；FlexEE的动态状态复用机制在保持语义连贯的同时最小化系统开销。
4. **FlexEE在卸载场景下收益最大化**：当权重加载成为瓶颈时，跳过层数直接转化为延迟下降，加速比可达 **3×以上**。

### **局限性**
- 依赖于**已接受layer-wise early-exit supervision训练的模型**，不能直接用于普通LLM。
- 最适合**小批量或单请求服务场景**，在大批量推理中因异步退出难以同步，增益减弱。
- 对**非常长序列或复杂推理任务**可能仍需完整深度计算，退出机会有限。

### **未来工作方向**
- 探索**无需额外监督训练**即可启用早期退出的方法。
- 扩展至**encoder-decoder架构**（如T5、BART）或多模态模型。
- 结合**动态批处理调度**，进一步优化异构退出行为下的吞吐。
- 研究**自适应确定自推测层位置**，根据不同输入动态调整策略。

--- 

> ✅ **总结一句话**：  
> **FlexEE通过“自推测+局部词表预测+动态KV管理”的协同设计，首次实现了在真实卸载环境下高效、可靠、系统友好的LLM早期退出，为边缘侧高效推理提供了实用解决方案。**

</details>

---

### 4. [Breaking the 1.58-bit Barrier for Ternary LLMs](https://arxiv.org/abs/2609.16338)

**Authors**: Evangelos Georganas, Alexander Heinecke, Pradeep Dubey  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.16338v1  

#### Abstract
Ternary Large Language Models (LLM) store every weight as one of three symbols $\{-1,0,+1\}$, so the cost of a ternary model is conventionally referenced to the information-theoretic $\log_2 3 \approx 1.585$ bits per weight. The prevailing deployment format packs five ternary weights into one byte (...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Breaking the 1.58-bit Barrier for Ternary LLMs》论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 ternary LLM（Ternary Large Language Models）将每个权重存储为三个符号之一：`{-1, 0, +1}`，其理论信息熵约为 **log₂3 ≈ 1.585 bits**，因此常被称为“1.58-bit”模型。然而，在实际部署中，由于采用 **five-trit packing**（每字节存5个三进制位），且量化组大小通常为2的幂（如128），导致有效存储开销为 **1.625 bits/weight**。

此外，现有方法普遍假设三个符号等概率分布，忽略了真实模型中 `0` 出现频率远高于 ±1 的现象，造成存储和计算效率未达最优。

### 提出的新方法：BITCOS
本文提出 **BITCOS**（**BITmap and COmpacted Signs**），一种基于分布自适应的稀疏存储格式，利用 ternary 权重中高密度零值的特点进行压缩。

#### 核心设计：
- **Presence Bitmap**：一个比特位表示对应权重是否非零。
- **Compacted Sign Vector**：仅对非零权重存储符号位（1 bit），按顺序排列。

该布局的有效存储成本为：
$$ B(z) = 1 + (1 - z) = 2 - z \text{ bits/weight} $$
其中 $ z $ 是零密度（zero density）。当 $ z > 0.375 $ 时，BITCOS 即优于传统的 five-trit packing（1.625 bits）。

### 相比现有方法的优势
| 特性 | 传统 five-trit packing | 2-bit packing | BITCOS |
|------|------------------------|--------------|--------|
| 存储效率 | 固定 1.625 bits | 固定 2.0 bits | 自适应 $ 2 - z $ bits |
| 是否依赖符号分布 | 否 | 否 | 是（利用高 $ z $） |
| 解码复杂度 | 中等 | 低 | 略高但可优化 |
| 兼容性 | 高 | 高 | 可集成至主流推理框架 |

- 在零密度 $ z \in [0.297, 0.515] $ 的实际模型中，BITCOS 平均节省 **1.16–1.32×** 的权重流量。
- 达到 **1.485 bits/weight** 的最低实测存储率（在 $ z = 51.5\% $ 的模型上）。
- 支持高效解包，已在 AVX-512、AVX2 和 Intel Xe2 GPU 上实现高度优化的 unpacking 序列。

---

## 2. 核心实验方法和设置

### 使用的模型（Dataset）
研究基于 **29 个 state-of-the-art (SOTA) ternary LLM 检查点**，涵盖以下七大家族：
- **BitNet b1.58**（2B 参数）
- **Bonsai**（1.7B–27B）
- **CAT-Q**（Qwen3 系列，1.7B–235B，含 MoE）
- **ParetoQ**（125M–1.5B）
- **TriLM**（Spectra 套件，99M–3.9B）
- **Maple**（20B-A1B MoE 推理模型）
- **BitCPM-CANN**（0.5B–8B）

这些模型通过 QAT 或 PTQ 得到，覆盖广泛架构与规模。

### 实验设置与评估指标

#### 硬件平台
| 类型 | 平台 |
|------|------|
| **CPU** |  
| - Server | Intel Xeon Platinum 8592+ (**EMR**, 64核, DDR5@4400MT/s)  
| - Client | Intel Core Ultra 9 285K (**ARL**, 24核: 8P+16E)  
| - Client | Intel Core Ultra 7 258V (**LNL**, 8核: 4P+4E, LPDDR5X)  
| **GPU** |  
| - Integrated | Intel Arc 140V (**Xe2**, 8 cores, 共享内存)  
| - Discrete | Intel Arc Pro B70 (**Xe2**, 32 cores, GDDR6)  

#### 软件与后端
- CPU 后端：`vLLM CPU backend` 集成 LIBXSMM 2-bit 与 BITCOS GEMV 内核
- GPU 后端：`vLLM XPU backend` 集成 XeTLA int2 与 BITCOS 内核
- 对比工具：`llama.cpp`（Q2_0 和 TQ1_0 格式）、Prism ML 分支

#### 评估指标
- **微基准测试（Microbenchmark）**
  - GEMV 执行时间（ms）
  - 有效带宽（Effective Bandwidth, GB/s）
- **端到端推理性能**
  - **Decode Throughput**（tokens/sec），batch size = 1，输出长度 256
- **存储效率**
  - 实际 bit-width（bits/weight）
  - 模型大小缩减比（size reduction）

#### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Five-trit per byte** | 每字节存5个 trit，固定 1.625 bits/weight |
| **2-bit packing** | 每权重用2 bit 表示，固定 2.0 bits/weight |
| **LIBXSMM 2-bit** | 当前 SOTA CPU 2-bit GEMV 内核 [9] |
| **XeTLA int2** | 当前 SOTA Xe2 GPU 2-bit 内核 [9] |
| **llama.cpp Q2_0 / TQ1_0** | 开源边缘推理实现 |

---

## 3. 主要实验结果和性能指标

### 存储效率提升
- **零密度范围**：29 个模型的 $ z \in [29.7\%, 51.5\%] $
- **BITCOS 成本**：$ 2 - z \in [1.485, 1.703] $ bits/weight
- **优于 five-trit packing 的比例**：**26 out of 29** 模型（需 $ z > 0.375 $）
- **最大压缩效果**：在 CAT-Q Qwen3-1.7B ($ z=51.5\% $) 上达到 **1.485 bits/weight**

| Format | Avg. Bits/Weight | Size Reduction vs 2-bit | Size Reduction vs 5-trit |
|--------|------------------|--------------------------|----------------------------|
| 2-bit | 2.000 | — | 1.22× |
| 5-trit | 1.625 | 1.03× | — |
| **BITCOS** | **~1.58** | **1.27×** | **1.09×** |

> 注：所有 29 个模型中，BITCOS 在符号+scale 总体积上均小于 2-bit 格式。

---

### 微基准测试（GEMV）

#### CPU 平台（EMR & ARL）
- **执行时间降低**：在 $ z \in [0.3, 0.5] $ 区间内，GEMV 时间减少 **14–28%**
- **有效带宽**：
  - EMR：稳定在 ~225 GB/s（带宽受限）
  - ARL：维持 ~93.5 GB/s
- **速度提升**：
  - EMR：**1.14–1.28×**
  - ARL：**1.13–1.27×**

#### GPU 平台（Arc 140V & Arc Pro B70）
- **执行时间更低**，但收益随 $ z $ 增加而下降（因解码开销不变）
- **速度提升**：
  - Arc 140V：**1.04–1.14×**
  - Arc Pro B70：**1.01–1.12×**
- **有效带宽下降趋势**：从高 $ z $ 到低 $ z $，带宽从 476 → 342 GB/s，表明解码成为瓶颈

---

### 端到端推理性能（Decode Throughput）

| Platform | Speedup Range (vs SOTA 2-bit) | Max Speedup |
|---------|-------------------------------|-----------|
| **EMR (64c)** | 1.10–1.18× | **1.18×** |
| **ARL (24c)** | 1.02–1.15× | **1.15×** |
| **LNL (8c)** | ≤1.00× | **无增益**（指令受限） |
| **Arc 140V (iGPU)** | 1.09–1.22× | **1.22×** |
| **Arc Pro B70 (dGPU)** | 1.02–1.27× | **1.27×** |

> ✅ **内存受限平台（EMR, ARL, Xe2 GPUs）显著受益**  
> ❌ **指令受限平台（LNL）无收益甚至更慢**

#### 与 llama.cpp 对比
- 在相同 2-bit 格式下，BITCOS 比 `llama.cpp Q2_0` 快：
  - EMR：**1.13–1.48×**
  - ARL：**1.46–1.74×**
- 在 GPU 上，XeTLA 2-bit 已比 Q2_0 快 1.46–2.12×，而 BITCOS 进一步提速至 **1.61–2.30×**

---

### 消融分析与 Roofline 模型验证
作者构建了一个简单的 **two-term roofline model** 来解释性能边界：

$$
\text{ebw} = \min\left(\frac{B}{\gamma}, \beta\right)
$$
其中：
- $ B $：每次迭代读取的字节数（随 $ z $ 下降）
- $ \gamma $：L1 内驻循环的周期数（实测）
- $ \beta $：每核可用带宽（GB/s/core）

#### 关键发现：
- **EMR 和 ARL**：处于 **memory-bound 区域** → 更小 payload 直接转化为加速
- **LNL**：处于 **instruction-bound 区域** → 解码额外开销无法掩盖，反而变慢
- **Xe2 GPU**：虽带宽更高，但解码逻辑引入开销，限制了理论收益

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Ternary LLM 中存在显著非均匀分布**：`0` 的占比高达 **29.7%–51.5%**，打破“equiprobable”假设。
2. ✅ **BITCOS 显著突破 1.58-bit 屏障**：在真实模型上实现 **低至 1.485 bits/weight**。
3. ✅ **26/29 模型存储更紧凑**：相比 five-trit packing，BITCOS 更优。
4. ✅ **端到端推理加速可达 1.27×**：在现代 CPU/GPU 上，decode throughput 提升明显。
5. ✅ **性能增益取决于硬件特性**：仅在 **bandwidth-bound** 场景下有效。

### 方法的局限性
- **解码开销增加**：需要额外的 `pdep` 或 lookup table 操作，在 **instruction-bound** 平台上可能得不偿失（如 LNL 客户端 CPU）。
- **GPU 缺乏 pdep 指令**：Xe2 需借助 SLM lookup table 实现，占用共享内存资源。
- **仅适用于 group-scaled ternary 模型**：依赖 per-group scale + {-1,0,+1} 结构。
- **不改变模型精度**：仅为存储和计算优化，不影响 accuracy。

### 未来工作方向
- 将 BITCOS 扩展至更多架构（AMD CPU、NVIDIA GPU、移动端 NPU）
- 探索动态调整 layout 的 hybrid 方案（例如混合使用 fixed-rate 与 adaptive layout）
- 结合 structured sparsity 进一步压缩（如与 N:M 模式结合）
- 支持其他 low-bit 格式（如 binary 或 4-value quantization）

---

## 总结

> **BITCOS 成功打破了 ternary LLM 的“1.58-bit”神话，揭示了真实模型中高零密度的价值，并通过简洁高效的 bitmap + compacted sign 设计，在主流硬件上实现了最高 1.27× 的 decode 加速。它不仅是存储格式的改进，更是对 ultra-low-bit inference 中“分布感知”理念的重要推进。**

</details>

---

### 5. [VideoMM: Adaptive Macro-Micro Inference for Efficient Video MLLMs](https://arxiv.org/abs/2609.16722)

**Authors**: Haoyu Guo, Yuan Feng, Junlin Lv, Mingjun Xiao, S Kevin Zhou, Xike Xie  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.16722v1  

#### Abstract
Scaling Multimodal Large Language Models (MLLMs) to long-form video understanding is bottlenecked by the explosion of visual tokens, which saturates context windows and incurs prohibitive costs. Current solutions predominantly rely on auxiliary models for token reduction but face a fundamental dilem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：VideoMM: Adaptive Macro-Micro Inference for Efficient Video MLLMs**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前在将 **Multimodal Large Language Models (MLLMs)** 应用于长视频理解时，面临一个核心瓶颈：**视觉 token 数量爆炸性增长**，导致上下文窗口饱和、计算成本极高。现有的 token 减少方法陷入“**准确率-效率困境**”：
- **轻量级 Encoder-driven 方法**（如 VisionZip, VidCom）速度快，但语义感知弱，容易误删关键信息；
- **重量级 MLLM-driven 方法**（如 FlexSelect）准确率高，但引入额外 MLLM 导致计算开销大，抵消了效率增益。

### **提出了什么新方法或新思路**
本文提出 **VideoMM**，实现从“模型规模缩减”到“感知粒度自适应”的范式转变，其核心思想是：
> **模仿人类“由粗到细”的认知过程**：先用低分辨率全局视图快速筛选关键区域（Macro），再按需调用高分辨率细节进行精细推理（Micro）。

#### **两大核心机制**：
1. **Grouped Selection with Macro Proxy**  
   - 将原始视频下采样为 **Macro Proxy**（空间降尺度因子 $k$），显著减少 token 数量；
   - 利用 MLLM 在分组的 Macro 视图上进行跨模态注意力分析，识别语义相关帧和 token；
   - 实现高效且语义精准的初步筛选。

2. **Adaptive Macro-Micro Inference**  
   - 引入 **共识机制（Consensus-based Voting）**：多个 Macro 分组独立生成答案；
     - 若所有分组答案一致（共识达成），则直接输出，**提前退出**；
     - 若不一致（表示存在歧义），则激活高分辨率 **Micro Tokens** 进行精细化推理。
   - 实现“仅在必要时才进行高成本计算”，动态分配资源。

### **相比现有方法的优势**
- ✅ **打破准确率-效率权衡**：既保持 MLLM 级别的语义精度，又避免其全程参与带来的开销；
- ✅ **计算资源按需分配**：对简单任务快速响应，复杂任务才启用高分辨率处理；
- ✅ **可扩展性强**：支持任意分辨率输入，天然契合现代 MLLM 的动态分辨率能力；
- ✅ **无需额外训练**：纯推理阶段优化，部署成本低。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LongVideoBench** [33]：包含长达1小时的视频，6,678道多选题，覆盖17个细粒度类别，强调跨模态检索与推理。
- **LVBench** [29]：专为超长视频设计（数小时级），测试真实场景下的信息提取与理解能力。
- **VideoMME** [12]：涵盖900个视频（共254小时）、2,700个问答对，覆盖多种视觉任务（计数、OCR、时空推理等）。

### **实验设置和评估指标**
- **模型**：
  - Qwen2.5-VL-7B（64K context）
  - GLM-4.1V-9B（64K context）
  - Qwen3-VL-8B（256K context）
- **输入设置**：统一采样至最多 512 帧 @ 1fps。
- **评估指标**：
  - **Accuracy (%)**：正确回答比例；
  - **Throughput (samples/min)**：每分钟处理样本数，衡量推理速度；
  - **Speedup**：相对于 baseline 的加速比。

### **基线方法对比**
| 类型 | 方法 | 特点 |
|------|------|------|
| **Encoder-driven** | VisionZip, VidCom | 仅用 ViT 编码器选择 token，速度快但语义弱 |
| **MLLM-driven** | FlexSelect [42] | 使用小型 MLLM 进行 token 选择，准确率高但慢 |
| **Baseline** | Vanilla Full-context | 不做压缩，完整处理所有 token |

> 所有方法统一限制为保留 **8,192 个视觉 token**，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | 方法 | Accuracy ↑ | Throughput ↑ | Speedup vs Vanilla |
|------|------|------------|---------------|---------------------|
| Qwen3-VL-8B | Vanilla | 66.94% | 0.85 | 1.0× |
| | **VideoMM** | **67.54%** (+0.6pp) | **3.69** | **4.3×** |
| | FlexSelect | 67.17% | 1.30 | 1.5× |
| | VisionZip | 57.97% | 2.47 | 2.9× |

> 在 **LongVideoBench** 上，VideoMM 实现：
> - **平均 6.13× 推理加速**
> - **7.4% 绝对准确率提升**（vs Vanilla）
> - **2.73× 加速于 FlexSelect**，且无准确率损失

### **与基线方法的对比结果**
- **vs Encoder-driven 方法**：
  - VideoMM 在 **吞吐量上持平甚至超越**（如 VidCom 4.34 vs VideoMM 4.94 samples/min on Qwen2.5-VL-7B）；
  - 同时 **准确率高出 8–10 个百分点**。
- **vs MLLM-driven 方法（FlexSelect）**：
  - 准确率相当（67.54% vs 67.17%）；
  - **吞吐量提升 2.73–3.9×**，显著更高效。

> 如表2所示，在相同吞吐量下（FlexSelect 降至256帧），VideoMM 仍能以更高帧数输入获得 **+4.7% 平均准确率提升**。

### **消融实验结果**
#### （1）自适应机制变体（VideoMM 家族）
| 变体 | 策略 | 早退率 β (512帧) | Throughput | Accuracy |
|------|------|------------------|-----------|----------|
| VideoMM- | 禁用早退，强制 Micro 推理 | 0% | 低 | 高 |
| VideoMM | 全票共识才早退 | 73.3% | 中 | **67.54%** |
| VideoMM+ | 多数投票即可早退 | **88.6%** | **4.90** | 66.87% |

✅ 结果表明：**自适应机制有效平衡了效率与精度**，形成更优的帕累托前沿。

#### （2）下采样因子 $k$ 影响
| 方法 | $k$ | Accuracy | Throughput | Speedup |
|------|-----|----------|------------|---------|
| VideoMM | 2 | **67.54%** | 3.69 | 4.3× |
| | **3** | 65.89% | **6.74** | **7.9×** |

✅ 即使极端下采样（$k=3$），通过 Micro 回退机制仍能维持合理精度，适用于极致效率场景。

#### （3）任务感知行为分析**
| 任务类型 | 示例 | 早退率 β |
|--------|------|---------|
| **Perception-Level (L1)** | T2E, O2E（定位事件） | 76–83% |
| **Relation-Level (L2)** | SSS, O3O（序列推理） | ~60% |

✅ 表明系统能**自动识别任务复杂度**：简单感知任务高频早退，复杂关系推理保守进入 Micro 阶段。

---

## **4. 关键结论和发现**

### **主要发现**
1. **根本冗余在于“全阶段高保真处理”**：并非所有视频片段都需要高分辨率分析，初步筛选可在低分辨率完成。
2. **“由粗到细”范式优于“模型瘦身”**：通过 **perceptual granularity adaptation** 而非 model scaling，可同时兼顾效率与精度。
3. **自适应机制带来显著收益**：高达 **73.3% 的样本可通过 Macro 阶段直接回答**，大幅降低平均计算量。
4. **VideoMM 家族建立新的 accuracy-efficiency frontier**：在多个模型和数据集上均优于现有方法。

### **方法的局限性**
- **依赖 MLLM 的中间层注意力机制**：若模型注意力机制不稳定，可能影响 token 选择质量；
- **对极端压缩（如 $k>3$）敏感**：虽然有回退机制，但在某些复杂时空推理任务上仍有性能下降；
- **未探索训练端联合优化**：目前为纯推理优化，未来可结合蒸馏或微调进一步增强。

### **未来工作方向**
- **结合 Speculative Decoding / KV Cache 压缩**：与其他推理加速技术正交，有望进一步提升端到端效率；
- **多阶段级联（Ultra-Macro → Macro → Micro）**：如 VideoMM-multi 所示，可构建更深的层级结构以适配不同长度视频；
- **动态调整 $k$ 和 $G$**：根据输入视频长度或任务类型自动配置参数；
- **集成 Agent-based 框架**：与 LVAgent、Symphony 等多智能体框架结合，实现语义级 + 视觉 token 级双重优化。

---

> **代码已开源**：[https://github.com/adfh917k/VideoMM](https://github.com/adfh917k/VideoMM)

</details>

---

### 6. [Towards Scalable RLVR: Multimodal Instruction Following Data Synthesis and Distillation](https://arxiv.org/abs/2609.16059)

**Authors**: Yirong Zeng, Zhang Sai, Yuxian Wang, Yutai Hou, Yufei Liu, Xiao Ding, Bibo Cai  
**Category**: cs.CL  
**Published**: 2026-09-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.16059v1  

#### Abstract
Multimodal instruction following (MMIF) is crucial for building generalist agents. However, current training paradigms rely heavily on Supervised Fine-Tuning (SFT), which often leads to surface-level pattern matching and degrades general capabilities. While Reinforcement Learning with Verifiable Rew...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Towards Scalable RLVR: Multimodal Instruction Following Data Synthesis and Distillation

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前多模态指令跟随（MMIF）训练主要依赖 **Supervised Fine-Tuning (SFT)**，但 SFT 容易导致模型进行**表面模式匹配**（surface-level pattern matching），而非真正理解指令逻辑，从而损害模型的泛化能力。虽然 **Reinforcement Learning with Verifiable Rewards (RLVR)** 在提升复杂推理方面展现出潜力，但在多模态场景中面临一个关键瓶颈：**缺乏高质量、适合 RL 训练的多模态数据**。

因此，本文旨在解决以下三个挑战：
1. **执行层面的规模与复杂性**：如何自动化生成大量、多样化且嵌入多重约束的多模态指令。
2. **数据可学习性**（Learnability）：并非所有数据都对 RL 政策优化有效，需筛选出处于模型“学习前沿”的样本。
3. **奖励保真度与验证**：传统 LLM-as-a-judge 易受奖励黑客攻击和偏见影响，需要高精度、程序化可验证的奖励信号。

---

### **提出了什么新方法或新思路**

作者提出 **MIFS**（**Multimodal Instruction Following Synthesis**），一个系统性的多阶段数据合成与蒸馏管道，用于生成适用于 RLVR 的高质量多模态指令数据。

#### 主要创新点：

- ✅ **生成式约束协议**（Generative Constraint Protocol）  
  将自然语言约束映射为可执行的 Python 代码（`checker_code`），确保每个约束具备**语义-代码等价性**（Semantic-Code Equivalence），实现**零容忍、程序化验证**，避免主观判断。

- ✅ **可学习性感知蒸馏机制**（Learnability-Aware Distillation）  
  设计了一个三阶段过滤流程，基于 RL 训练动态筛选最优数据：
  1. **约束密度过滤**（Constraint Density Filtering）：保留 2–10 个约束的样本，剔除过简或冗余样本。
  2. **难度感知剪枝**（Difficulty-Aware Pruning）：利用 Qwen3-VL-8B 作为审计器，剔除太简单（aug@8 > 0.8）和不可能完成（aug@128 = 0）的样本。
  3. **RL 可学习性过滤**（RL Learnability Filtering）：分析多个训练 epoch 中的 reward score 轨迹，计算轨迹偏差 $D$，剔除导致梯度不稳定的离群样本，仅保留偏差最小的 40%。

- ✅ **代码驱动的奖励函数**（Code-Based Verifier）  
  每个任务均配备一个确定性 Python 验证器，提供二元奖励信号（R=1 若满足全部约束，否则 R=0），保障奖励高保真。

- ✅ **大规模开源数据集发布**  
  发布包含 **90,838 个样本**的数据集，覆盖 **8 类约束** 和 **14 个任务领域**，并划分为 SFT、RL 和 Eval 子集，支持完整训练周期。

---

### **相比现有方法的优势**

| 维度 | 现有方法（如 MM-IFInstruct） | MIFS |
|------|-------------------------------|------|
| 数据用途 | 主要支持 SFT/DPO | 同时支持 SFT 和 **RLVR** |
| 奖励机制 | 依赖 LLM-as-a-judge（易偏） | **程序化规则验证**（高保真） |
| 数据质量 | 手工设计或模板生成 | 自动化生成 + 多级蒸馏优化 |
| 规模 | 最大 ~23k（MM-IFInstruct） | **90k+**，最大规模 |
| 可扩展性 | 有限 | 全流程自动化，可扩展性强 |

> 📌 **核心优势**：MIFS 是首个专为 **scalable RLVR** 设计的大规模、程序可验证多模态指令数据集构建框架。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **种子图像来源**：
  - CC3M（20,000 张高分辨率图）
  - ALLaVA（原始图像，丢弃原 QA 对以保持干净起点）

- **最终产出数据集**：
  - **MIFS**：90,838 个样本，平均 7.1 个约束/样本，涵盖 8 类约束、14 个任务域。
  - 划分：
    - `MIFS-RL`：60,048 样本（用于 RL 训练）
    - `MIFS-SFT`：83,280 样本（含 verified response，用于 SFT）
    - `MIFS-Eval`：1,200 样本（held-out 测试集，无图像/约束组合泄露）

---

### **实验设置**

- **模型系列**：Qwen3-VL 系列（4B, 8B, 32B）
- **训练流程**：
  - **SFT 阶段**：使用 AdamW，lr=2e-5，batch_size=128，训练 2 轮
  - **RLVR 阶段**：采用 **GRPO** 算法，lr=1e-6，batch_size=128，rollout group size=8
- **框架**：基于 `verl` 框架实现端到端训练
- **硬件**：128 × Ascend 910B NPUs

---

### **评估指标**

#### **指令跟随能力基准**（MMIF Benchmarks）：
- **MIA-Bench**（Qian et al.）：细粒度评分，评估多模态指令遵循
- **MM-IFEval**（Ding et al. 2025）：多模态版 IFEval
- **MIFS-Eval**：本文构建的独立测试集
- **IFEval**（Zhou et al. 2023a）：纯文本指令遵循基准

> ✅ **评估方式**：每样本生成 3 次响应（temp=1.0），仅当**全部约束通过 Python 验证器**才记为成功，报告平均 pass rate。

#### **通用视觉能力基准**（General Visual Capabilities）：
- **OCRBench**：文本识别与文档理解
- **MM-Vet**：跨任务综合推理
- **MMBench**：细粒度感知与逻辑评估

---

### **基线方法对比**

| 基线方法 | 描述 |
|---------|------|
| **Base** | 未经微调的原始 Qwen3-VL 模型 |
| **MIFS-SFT** | 仅使用 MIFS 的 SFT 数据微调 |
| **MIFS-RL** | 在 SFT 初始化后进行 RLVR 微调 |
| **MIFS-SFT+RL** | SFT + RL 两阶段训练 |
| **MM-IFInstruct-SFT** | 当前主流 SFT 方法（Ding et al. 2025） |
| **RECAST-RL / VerIF-RL** | 文本模态 RLVR 方法（无法处理多模态） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **指令跟随性能提升显著**

| 方法 | MIA | MM-IFEval | MIFS-Eval | IFEval | **Avg.** |
|------|-----|-----------|-----------|--------|--------|
| Qwen3-VL-8B Base | 91.2 | 62.7 | 74.5 | 81.2 | **77.4** |
| MIFS-SFT | 93.3 | 64.7 | 77.7 | 86.3 | **80.5 (+3.1)** |
| MIFS-RL | 95.9 | 71.2 | 86.8 | 88.2 | **85.5 (+8.1)** |
| MIFS-SFT+RL | 97.0 | 72.4 | 88.2 | 88.8 | **86.6 (+9.2)** |

> 🔺 **平均提升达 8.13%**，在 MM-IFEval 上提升高达 **+8.5pp**，显示 RLVR 对复杂多约束任务的强大增益。

#### ✅ **跨模型尺度一致性**

在 4B、8B、32B 模型上均观察到一致收益：
- 平均提升分别为：**+8.16**, **+8.13**, **+8.52**
- 更大模型上限更高（32B 在 MIFS-Eval 达 89.8）

---

### **与基线方法对比结果**

| 方法 | Avg. MMIF Score | 相对提升 |
|------|------------------|----------|
| MM-IFInstruct-SFT | ~79.7 | — |
| **MIFS-RL** | **85.5** | **+5.8** |
| **MIFS-SFT+RL** | **86.6** | **+6.9** |

> 💡 MIFS-RL 单独使用即超越最强 SFT 基线近 6 个百分点。

---

### **消融实验结果**

#### **质量蒸馏机制的有效性**

| 数据阶段 | 数据量 | MIA-Bench 性能 |
|--------|--------|----------------|
| Raw Data (500k) | 500k | ~83.0 |
| After Step A (密度过滤) | 400k | ↑ |
| After Step B (难度剪枝) | 304k | ↑↑ |
| After Step C (RL 可学习性过滤) | 90k | **↑↑↑ (最高)** |

> 🔍 每一步蒸馏都带来性能提升，最终 **90k 高信号子集**在更少数据下实现更高性能。

#### **训练效率提升 3×**

| 数据类型 | 收敛步数 | 最终性能 |
|--------|----------|----------|
| Raw Data | ~600 步 | 次优 |
| **MIFS-RL** | **~150 步** | **更高** |

> ⏱️ **收敛速度加快约 3 倍**，且测试性能更高，说明 MIFS 数据更稳定、高效。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **RLVR 在多模态指令跟随中优于 SFT**  
   相比 SFT 的模仿学习，RLVR 通过试错与验证机制，促使模型**内化指令逻辑**，显著提升复杂约束下的精确性。

2. ✅ **高质量数据蒸馏至关重要**  
   并非越多越好，而是要选择“恰到好处”的样本——位于模型当前能力边界的中等难度、高可学习性样本。

3. ✅ **程序化验证是 RLVR 成功的关键**  
   代码驱动的 reward signal 避免了 LLM judge 的偏见与模糊性，实现了**高保真、抗 reward hacking** 的优化目标。

4. ✅ **无需 SFT 初始化也可高效训练**  
   实验表明，**直接 RL 训练**（MIFS-RL）已能取得远超 SFT 的效果，SFT+RL 仅带来边际增益，暗示未来可能简化训练流程。

5. ✅ **通用视觉能力未受损**  
   在 OCRBench、MMVet、MMBench 上波动 <0.5%，证明 MIFS 训练**未牺牲基础感知能力**，实现了“精准指令 + 通用能力”双赢。

---

### **局限性**

1. **架构依赖性**：目前实验集中在 Qwen-VL 架构，尚未验证在 LLaVA、InternVL 等其他 MLLM 上的泛化性。
2. **排除主观任务**：因追求程序可验证性，MIFS 排除了创意写作、情感表达等开放性任务，限制了应用场景。
3. **计算开销大**：RL 轨迹分析引入额外成本，数据生产阶段需运行完整 RL probe，不利于快速迭代。

---

### **未来工作方向**

1. **混合奖励机制**：结合 rule-based verifier 与 aligned LLM judge，在保证安全前提下拓展至开放域任务。
2. **轻量化蒸馏代理**：开发小型 proxy model 来预测样本可学习性，替代昂贵的 full-scale RL probe。
3. **跨架构适配**：验证 MIFS 数据在不同 MLLM 架构上的迁移能力，推动标准化 MMIF-RL 数据集建设。
4. **动态 Curriculum Learning**：将 learnability-aware 过程从静态蒸馏升级为在线动态选择，实现自适应训练课程。

---

> 🎯 **总结一句话**：  
> **MIFS 构建了首个面向 scalable RLVR 的大规模、程序可验证多模态指令数据集，通过“生成+蒸馏”双轮驱动，在不牺牲通用能力的前提下，实现了指令跟随精度与训练效率的双重突破。**

</details>

---

### 7. [LLM Inference in a Flash!](https://arxiv.org/abs/2609.16161)

**Authors**: Sebastian Zhao, Minseo Kim, Coleman Hooper, Luca Manolache, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami  
**Category**: cs.LG  
**Published**: 2026-09-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.16161v1  

#### Abstract
Large Language Models (LLMs) have shown impressive capabilities across a range of natural language processing tasks, and LLM inference has emerged as a critical workload for enabling downstream applications. The demands of serving LLM inference are becoming increasingly challenging as requests shift...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《LLM Inference in a Flash!》核心总结**

---

## **1. 主要贡献和创新点**

### **解决的问题**
大型语言模型（LLM）推理面临严重的**内存墙（memory wall）瓶颈**，尤其是在长上下文场景下：
- **KV Cache 动态增长**导致内存带宽和容量压力巨大。
- 传统 DRAM/SRAM 内存受限于带宽扩展速度慢、功耗高、成本上升。
- Flash 存储虽具备大容量优势，但存在两大限制：
  - 缺乏对高精度浮点运算（FP32）的良好支持；
  - 写入耐久性差（limited write endurance），不适合频繁更新的 KV Cache。

因此，如何在不牺牲性能的前提下，将 LLM 推理高效部署到基于 Flash 的 Compute-in-Memory（CIM）系统中，是一个关键挑战。

---

### **提出的新方法与思路**

本文提出了一个端到端的算法-硬件协同设计框架，以实现高效的 Flash-based LLM 推理：

#### ✅ **创新点一：端到端整数量化（End-to-End Integer-Only Quantization）**
- 将整个 LLM 推理流程（包括线性层、注意力机制、非线性操作如 Softmax、RMSNorm、SiLU）全部转换为 **INT8 / INT32 整数运算**。
- 关键技术：
  - 采用 **SmoothQuant 风格的激活平滑** 和 **group-wise 量化**（group size=128）来缓解异常值影响。
  - 对非线性函数使用 **二阶多项式近似**（polynomial approximation），例如 IntSoftmax、IntRMSNorm。
  - 在 Softmax/RMSNorm 输入前进行 **INT16 重量化（requantization）**，避免跨组量化尺度导致的归约错误，并防止溢出。

> 🎯 目标：消除所有 FP32 运算，使模型完全可在无浮点单元的 Flash-CIM 设备上运行。

#### ✅ **创新点二：基于字典的 KV Cache 压缩（Dictionary-Based KV Cache Compression）**
- 利用 Flash 的“读多写少”特性，构建一种 **静态只读字典 + 动态稀疏编码** 的 KV 表示方式。
- 核心思想：
  - 训练一个**过完备字典（overcomplete dictionary, M=32768 atoms）**，用于重构每个 KV 向量。
  - 每个 KV 向量表示为少量字典原子的线性组合（sparse code），即 `(index, coefficient)` 对。
  - 字典驻留在 Flash 中只读；仅稀疏码需要动态存储和更新，大幅减少写入次数。

- 技术细节：
  - 使用 **投影式贪婪搜索（projection-based pursuit）** 替代传统的 OMP，避免昂贵的最小二乘求解，更适合 CIM 执行。
  - 引入 **查询感知的分层稀疏策略（query-aware hierarchical sparsity）**：
    - 先用低预算 $ K_{low}=3 $ 构建粗略得分；
    - 选择 top-N% token 提升至 $ K_{high}=16 $ 进行精细重建。
  - 维护一个小的本地窗口（local window, W=128 tokens）保留原始 KV，保证最新 token 的保真度。

> 🎯 目标：将动态 KV 写入量降低一个数量级，同时利用 Flash 高内建读带宽加速注意力计算。

---

### **相比现有方法的优势**

| 方面 | 本文方法 | 现有方法（如 KVNAND、Lexico、HiFC） |
|------|----------|-------------------------------|
| **算法适配性** | 完全面向 Flash-CIM 特性设计（无 FP、低写入） | 多为系统调度或硬件优化，缺乏算法层面重构 |
| **KV 压缩效率** | 支持 on-the-fly attention（无需物化 KV） | 多需先解压再计算，增加延迟 |
| **非线性处理** | 全整数近似，无需 offload 到主机 | 依赖主机执行 FP32 非线性操作 |
| **硬件友好性** | 压缩过程仅含 MAC 和向量更新，适合 CIM | OMP 类压缩需复杂迭代求解 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **校准与量化训练**：
  - 使用 **The Pile** 数据集的验证集进行量化参数校准。
- **KV 字典训练**：
  - 同样从 The Pile 中提取 Llama-3.1-8B 和 Qwen-2.5-7B 的 attention layer 激活训练字典。
- **下游任务评估**：
  - 使用 **LongBench** 基准测试套件，涵盖：
    - 单文档/多文档问答（multifieldqa-en, 2wikimqa）
    - 政府报告摘要（gov-report）
    - 小样本学习（trec）
    - 代码补全（lcc）
  - 评估指标：平均得分（Avg. Score）

### **实验设置与评估指标**
- **模型**：
  - Llama-3.1-8B 和 Qwen-2.5-7B
  - 分组大小 group_size=128，量化精度 W8A8
- **KV Cache 设置**：
  - 局部窗口大小 W=128
  - 每次压缩最老的 U=32 个 token
  - 字典大小 M=32768，每层独立训练 key/value 字典
  - 分层稀疏：$ K_{low}=3, K_{high}=16 $
- **评估指标**：
  - 下游任务准确率（LongBench Avg Score）
  - KV Cache 压缩比（Compression Ratio）
  - 系统级延迟（Time Between Tokens, TBT）
  - 能耗（Energy per Token）
  - 消融分析（ablation study）

### **基线方法对比**
| 配置 | 描述 |
|------|------|
| **B1**: NPU + DRAM, no compression | FP16 推理，KV Cache 完全存于 DRAM |
| **B2**: NPU + DRAM + dictionary compression | 使用相同字典压缩，但在 NPU 上执行 |
| **S1**: CIM-SSD, codes on controller | Flash-CIM 推理，压缩码存于控制器 SRAM |
| **S2**: CIM-SSD, codes on DRAM | 更通用配置，压缩码存于外部 DRAM（默认主配置） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **下游任务表现（LongBench）**
| 方法 | Llama-3.1-8B (Avg) | Qwen-2.5-7B (Avg) | KV 压缩比 |
|------|---------------------|--------------------|------------|
| Full FP16 KV Cache | 44.07 | 44.44 | 1× |
| 本文方法（完整 pipeline） | **43.65** | **44.23** | **15×** |

✅ 结论：**性能损失 <1 point，但 KV 流量减少 15 倍**，接近无损压缩。

#### 🔹 **消融实验结果**

##### （1）整数量化消融（Llama-3.1-8B on WikiText-2）
| 配置 | Perplexity |
|------|-----------|
| FP16 Baseline | 7.5454 |
| + W8A8 | 7.6525 |
| + SmoothQuant + group-wise | 7.5511 |
| + Integer Nonlinear Ops | NaN（发散） |
| + INT16 Requantization | **7.5803** |

➡️ 表明：**INT16 requantization 是稳定整数非线性的关键**。

##### （2）KV 压缩消融（LongBench）
| 配置 | Avg Score | KV Compr. |
|------|----------|-----------|
| Full FP16 KV | 44.07 | 1× |
| OMP Coding (K=16) | 44.61 | 4× |
| + Projection-based Coding | 44.56 | ~4× |
| + INT8 Dictionary | 44.56 | ~4× |
| + Hierarchical Sparsity | **44.56** | **15×** |

➡️ 表明：**分层稀疏是实现 15× 压缩的核心**，且精度几乎不变。

---

#### 🔹 **系统级建模结果（Analytical Model）**
> 模型：Llama-3.1-8B，batch size=1，context length=1K / 256K

| Configuration | Context Length | TBT (ms) | Energy (J) | Speedup vs B1 | Energy Reduction |
|--------------|----------------|----------|------------|---------------|------------------|
| **B1** (NPU+DRAM) | 1K | 183.7 | 101.8 | 1× | 1× |
| **S2** (CIM-SSD) | 1K | **59.9** | **37.8** | **3.07×** | **2.69×** |
| **B1** | 256K | 631.5 | 354.5 | 1× | 1× |
| **S2** | 256K | **143.5** | **52.5** | **4.40×** | **6.75×** |

✅ **结论**：
- 在短上下文下实现 **3.1× 延迟下降、2.7× 能效提升**；
- 在长上下文下达到 **4.4× 延迟下降、6.8× 能效提升**；
- 性能增益随 context length 增加而扩大，因 KV I/O 成为主导开销。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Flash-based CIM 可成为 LLM 推理的有效平台**，前提是通过算法改造克服其短板（无 FP、写入寿命短）。
2. ✅ **端到端整数量化 + 字典压缩** 可协同释放 CIM 的潜力：
   - 整数量化消除对 FP 单元的依赖；
   - 字典压缩将 KV Cache 转化为“读密集型”负载，契合 Flash 特性。
3. ✅ **KV 压缩比达 15×，系统延迟/能耗显著降低**，尤其在长上下文场景优势明显。
4. ✅ **分层稀疏 + 投影式编码** 在精度与效率之间取得良好平衡，优于传统 OMP。

---

### **局限性（Limitations）**
1. ❗ 当前工作聚焦于 **generation 阶段**，prefill 阶段由于计算密集，尚未有效映射到低峰值算力的 CIM 平台。
2. ❗ **Value Dictionary 需要存储两个副本**（转置与非转置），以支持 pursuit 和 attention 计算中的不同访问模式，增加了存储开销。
3. ❗ 实验基于**分析建模（analytical model）**，尚未在真实 Flash-CIM 硬件原型上验证。

---

### **未来工作方向**
- 将 prefill 和 decode 阶段统一优化，实现全流程 CIM 加速。
- 探索更高效的字典结构（如共享字典、层次字典）以降低存储冗余。
- 开发支持 transpose-free 访问的新型 pursuit 或 attention 架构。
- 在真实 CIM 硬件平台上实现并验证端到端系统。

--- 

> 💡 **一句话总结**：  
> 本论文通过 **整数量化 + 字典化 KV Cache 压缩**，首次实现了面向 Flash Compute-in-Memory 的高效 LLM 推理，在几乎无损精度的情况下，达成高达 **15× KV 流量压缩** 和 **4.4× 延迟 / 6.8× 能效提升**，为资源受限设备上的大模型部署提供了全新路径。

</details>

---

### 8. [LCAP: Population-Informed Latent Chip Adaptation from Few Output Probes for Photonic Neural Networks](https://arxiv.org/abs/2609.16823)

**Authors**: Tianyu Gao, Guantian Zheng  
**Category**: cs.LG  
**Published**: 2026-09-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.16823v1  

#### Abstract
Photonic neural networks (PNNs) offer efficient analog inference, but parameters optimized under ideal device models can degrade after fabrication, creating a persistent simulation-to-hardware (sim-to-real) gap. When many identically designed chips are deployed, calibrating each device from scratch ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LCAP: Population-Informed Latent Chip Adaptation from Few Output Probes for Photonic Neural Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Photonic Neural Networks（PNNs）在理想仿真模型中训练后，部署到实际硬件时会因制造偏差（如 phase shift 偏移、beam-splitter 误差、quantization 和 crosstalk）导致严重的 **sim-to-real gap**。这种偏差在不同芯片间各不相同，传统方法需对每块新芯片进行独立校准，成本高昂且不可扩展。

此外，直接使用端到端方式从 probe 输出学习设备修正参数时，**device-specific 信号太弱**，容易被 population-common 的系统性偏差掩盖，造成“identifiability problem”。

---

### 🚀 提出的新方法：LCAP（Latent Chip Adaptation from Probes）

LCAP 是一种 **population-informed、common-first, individuality-second** 的两阶段硬件自适应框架：

1. **Population-shared Calibration（共享校准）**  
   利用历史芯片群体（80个）学习一个可迁移的全局修正参数 $\theta_{\text{pop}}$，捕捉跨芯片共有的制造偏差模式。

2. **Residual Latent Personalization（残差个性化）**  
   在共享校准基础上，建模剩余的设备特异性偏差 $\delta_c$，通过 SVD 构建低维 **latent correction space**，仅在此空间中表示个体差异。

3. **Probe-conditioned Latent Inference（探针推断）**  
   对于未见过的新芯片，仅需采集 **32个固定无标签输出探针**（output probes）的响应，即可通过 PCA + Ridge Regression 推断其 latent coordinates，并重建个性化修正项，无需任何目标设备优化（no target-device optimization）。

---

### 🔍 相比现有方法的优势

| 方法类型 | 代表工作 | 局限性 | LCAP 的优势 |
|--------|--------|------|------------|
| **Device-specific calibration** | L2ight, DAT, meta-learning | 每块芯片都需要优化，成本高 | 零样本适应，无需反向传播或梯度测量 |
| **Transfer-robust training** | Transferable Learning, SAT | 忽视个体差异，性能上限受限 | 显式建模个体性，进一步提升精度 |
| **End-to-end probe encoding** | 直接编码 probe → correction | device-specific 信号被淹没（见 Fig. 1） | 分解为“共性+个性”，解决 identifiability 问题 |

> ✅ **核心创新思想**：先学一个强的 shared anchor（$\theta_{\text{pop}}$），再只对 residual 建模低维 latent space —— 更紧凑、更可预测。

---

## 2. 核心实验方法和设置

### 📊 数据集与任务
- **任务**：MNIST 手写数字分类（10类）
- **输入处理**：图像经傅里叶变换，取中心 $8\times8$ 频谱作为 64 维复数光学输入
- **网络结构**：三层 64-mode MZI-PNN（Mach-Zehnder Interferometer 光子神经网络）

> 所有虚拟硬件实例均从同一理想 PNN 参数初始化

---

### 💡 芯片仿真模型（Structured Chip Population）
构建具有物理合理性的芯片制造偏差模型：
$$
\Delta \theta = \sqrt{0.5}\cdot b_c^{\text{Norm}} + \sqrt{0.25}\cdot g + \sqrt{0.25}\cdot e_c
$$
其中：
- $b_c$: 共享 fabrication mode（8个主成分）
- $g$: 固定低频空间系统误差（spatial systematic pattern）
- $e_c$: 独立局部随机扰动

其他非理想因素：
- Beam-splitter 误差：$U(0.02, 0.04)$
- 8-bit phase quantization
- Nearest-neighbor crosstalk: 0.005

> 每块芯片的误差配置固定，模拟真实持久性偏差

---

### 🧪 实验协议
- **历史芯片（Historical chips）**：80 块用于训练
  - 其中 40 块为 teacher chips，用于拟合 residual latent space
- **测试芯片（Unseen chips）**：30 块独立生成，severity 匹配，**完全未参与训练/设计**
- **Probe 设置**：
  - 从 128 个候选 $\{-1,+1\}^{64}$ 输入中选择 32 个（基于 cross-chip variance 排序）
  - 输出残差 $r_c(p) = \text{Re}[f_u(p;\theta_{\text{pop}}) - f_p(p;\theta_{\text{pop}})]$
  - 经 PCA 投影为 16 维特征 $h_u$
- **Latent 推断**：Ridge Regression 学习 $W: h_u \mapsto z_u$
- **Latent 维度**：$r=16$

---

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Mean accuracy (%)** | 30 块测试芯片平均准确率 |
| **Worst-device accuracy (%)** | 最差单块芯片准确率 |
| **Across-chip std. (pt)** | 跨芯片标准差（衡量稳定性） |
| **Improved chips** | 相较前一阶段提升的芯片数量 |
| **New-chip adaptation cost** | 是否需要额外优化 |

---

### ⚖️ 基线方法对比
| 方法 | 描述 |
|-----|------|
| **Direct Deployment** | 直接部署理想参数，无校准 |
| **Population Calibration Only** | 仅应用 $\theta_{\text{pop}}$，无个性化 |
| **LCAP（Ours）** | 共享校准 + probe 推断 latent 个性化 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table 1）

| 方法 | 平均准确率 (%) | 较前提升 (pt) | 最差芯片准确率 (%) | 跨芯片 std. (pt) | 改进芯片数 |
|------|----------------|---------------|--------------------|------------------|-------------|
| Direct Deployment | 80.415 | — | 60.21 | 9.189 | — |
| Population-only | 92.686 | +12.271 | 89.18 | 1.274 | 30/30 |
| **LCAP** | **93.362** | **+0.676** | **90.54** | **1.198** | **27/30** |

> ✅ **结论**：  
> - 共享校准解决了 **大部分 sim-to-real gap**（+12.27pt）  
> - LCAP 在此基础上进一步提升 **+0.68pt**，且将最差芯片表现从 89.18% 提升至 90.54%，显著增强鲁棒性

---

### 🔬 消融实验结果（Ablation Studies）

#### (a) Latent Rank 消融（Table 2a）

| Rank | 捕获 teacher energy (%) | 准确率 (%) | 提升 (pt) | 改进芯片 |
|------|--------------------------|-----------|----------|---------|
| 8 | 43.1 | 93.308 | +0.622 | 27/30 |
| **16** | **63.5** | **93.362** | **+0.676** | **27/30** |
| 32 | 91.4 | 93.316 | +0.630 | 28/30 |

> ❗ **关键发现**：更高的 reconstruction fidelity 不等于更好的 deployment performance！  
> 尽管 rank=32 捕获更多 teacher 信息，但泛化能力下降，说明过拟合了难以观测的方向。

---

#### (b) Probe 数量消融（Table 2b）

| Probes | 准确率 (%) | 提升 (pt) | 改进芯片 |
|--------|-----------|----------|---------|
| 8 | 93.083 | +0.397 | 22/30 |
| 16 | 93.250 | +0.564 | 27/30 |
| **32** | **93.362** | **+0.676** | **27/30** |

> ✅ **结论**：probe 数量越多，latent 推断越可靠；但收益递减，32 已接近饱和

---

#### (c) 表示学习方法比较（Fig. 3）

| 方法 | Teacher NMSE | 新芯片增益 (pt) |
|------|--------------|----------------|
| AE (Autoencoder) | $1.52\times10^{-5}$ | +0.470 |
| DAE / VAE / SWAE | 极低 | < +0.600 |
| **SVD + Ridge (LCAP)** | 0.365（较高） | **+0.676** |

> ⚠️ **颠覆性发现**：**重建精度 ≠ 部署效用**！  
> 非线性 AE 能完美重构 teacher correction，但在新芯片上效果反而更差 —— 因其保留了不可观测的细节。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Sim-to-real gap 中存在强烈共享结构**  
   > 多达 12.27pt 的性能损失可通过 population-shared calibration 恢复，表明制造偏差具有高度相关性和可迁移性。

2. **Device-specific 偏差虽小但仍可预测**  
   > 即使在强共享校准后，残差仍蕴含约 +0.68pt 的提升潜力，且可通过少量 probe 可靠推断。

3. **Latent Space 应追求“可识别性”而非“重建精度”**  
   > 最佳 hardware latent 不是能最好地还原历史数据的那个，而是能在新设备上稳定预测的那个。

4. **Common-first, Individuality-second 范式有效**  
   > 先建模共性，再聚焦个性，避免弱信号被淹没，是 scalable hardware adaptation 的正确路径。

---

### ⚠️ 方法的局限性

1. **依赖历史芯片群体的质量和多样性**  
   > 若历史芯片不能覆盖真实变异空间，新芯片可能无法准确推断。

2. **假设偏差是持久性的（persistent）**  
   > 不适用于动态漂移场景（如温度变化引起的时变误差）。

3. **当前仅验证于 MZI-PNN 架构**  
   > 是否推广至其他光子架构（如 microring resonator networks）尚待验证。

4. **probe 输入为人工设计**  
   > 虽然基于 variance 选择有效，但最优 probe 设计仍是开放问题。

---

### 🔮 未来工作方向

1. **扩展至动态环境下的在线自适应**  
   > 结合 temporal modeling，应对 drift 和 aging 效应。

2. **联合优化 probe selection 与 latent representation**  
   > 使用 mutual information 或 active learning 自动发现 informative probes。

3. **引入不确定性估计**  
   > 对 latent 推断置信度建模，用于安全关键应用中的 fallback 决策。

4. **跨架构迁移**  
   > 探索是否可在不同 PNN topology 之间共享 correction prior。

5. **硬件实测验证**  
   > 当前为数值仿真，下一步应在 real fabricated chips 上验证 LCAP 效果。

---

> 📌 **一句话总结**：  
> **LCAP 通过“先共性、后个性”的分解策略，利用历史经验实现零样本芯片自适应，在仅需 32 个无标签 probe 的前提下大幅提升 PNN 部署准确率与鲁棒性，为大规模光子神经网络落地提供了高效 calibration 路径。**

</details>

---

### 9. [CoAdapt: An LLM-based Framework for Adaptive Collaborative Perception in IIoT Robotic Swarms](https://arxiv.org/abs/2609.16852)

**Authors**: Houssam Hajj Hassan (L2S), Antonia Maria Masucci (L2S), Lynda Zitoune (L2S), Salah-Eddine Elayoubi (L2S)  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.16852v1  

#### Abstract
Industrial IoT environments increasingly deploy autonomous mobile robots for tasks such as material handling, product assembly, or infrastructure inspection. In such deployments, collaborative perception enables robots to share LiDAR observations and collectively construct a richer model of their en...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CoAdapt: An LLM-based Framework for Adaptive Collaborative Perception in IIoT Robotic Swarms

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在工业物联网（IIoT）环境中，机器人集群（robotic swarms）通过协同感知（collaborative perception）共享 LiDAR 数据以提升环境感知能力。然而，现有方法存在以下问题：
- **静态融合策略**：参与融合的机器人集合和融合算法（fusion algorithm）通常在部署时固定，无法适应动态变化的网络带宽、机器人位置移动和场景遮挡。
- **冗余通信开销**：多个机器人视野重叠导致数据冗余，增加通信负担。
- **缺乏联合自适应机制**：现有方法未能同时动态调整**参与者选择**（participant selection）和**融合范式**（fusion paradigm），难以在检测精度与通信成本之间取得实时平衡。

### 🚀 提出的新方法与创新点
本文提出 **CoAdapt**，一个基于大语言模型（LLM）的自适应协同感知框架，其核心创新如下：

- **LLM 作为运行时融合控制器**：首次将 LLM 引入协同感知系统，作为 MAPE-K 控制环中的“Plan”模块，**联合决策**哪些机器人参与融合、采用何种融合算法（early/intermediate/late fusion）。
- **无需任务特定训练**：通过 **Scene Abstraction Module (SAM)** 将原始 LiDAR 点云转换为结构化自然语言描述，使 LLM 能够直接推理空间配置和网络状态，无需微调或强化学习。
- **动态多目标优化**：在每个控制周期内，根据当前环境状态 $ S(t) = (S_{\text{env}}, S_{\text{net}}, S_{\text{app}}) $ 动态调整融合策略，实现检测精度与通信成本的最优权衡。

### 🔍 相比现有方法的优势
| 维度 | 现有方法 | CoAdapt |
|------|--------|--------|
| 参与者选择 | 静态或基于拓扑预设 | 动态、上下文感知、几何互补性驱动 |
| 融合策略 | 固定范式（如始终 intermediate fusion） | 动态切换 early/intermediate/late fusion |
| 自适应能力 | 缺乏联合优化 | 同时优化参与者 + 融合算法 |
| 泛化性 | 依赖环境特定训练 | 无需训练，适用于未见拓扑 |
| 实现复杂度 | 需要定制化学习模型 | 利用通用 LLM 推理能力 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **OPV2V**：大规模仿真协同感知基准数据集，基于 CARLA 模拟器生成。
  - 包含 73 个多样化城市与郊区场景。
  - 支持最多 6 辆车同时进行 LiDAR 观测，适合评估 swarm-level 决策。
  - 实验选取其中 **25 个包含 ≥3 车辆的场景**用于评估。

### ⚙️ 实验设置
- **融合后端模型**（来自 OpenCOOD 框架）：
  - **Early fusion**: Cooper 模型
  - **Intermediate fusion**: VoxelNet + 注意力机制
  - **Late fusion**: PointPillars
- **Scene Abstraction Module (SAM)**：
  - 使用 Open3D 处理点云，去除地面后聚类为对象。
  - 输出每个对象的：相对位置、距离、包围盒尺寸、点密度 → 转换为自然语言描述输入 LLM。
- **LLM 配置**：
  - 测试四种开源 LLM：**Gemma 4 (31B)**、**Llama3.3 (70B)**、**GPT-OSS (20B & 120B)**
  - 部署于 NVIDIA H100 GPU，控制周期为 5 秒（50 帧）
- **带宽模拟**：
  - 由于 OPV2V 不提供真实网络数据，构建时间相关带宽模型：
    - 上午（6–12 AM）：55 Mbps（高）
    - 下午（12–6 PM）：20 Mbps（中）
    - 晚上（6 PM–12 AM）：5 Mbps（低）
  - 加入 ±15% 正弦波动模拟实际网络波动。

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **AP@0.7** | IoU 阈值为 0.7 的平均精度（Average Precision），衡量检测质量 |
| **Communication Cost** | 每帧传输的数据量（bytes），反映网络负载 |
| **Participant Count** | 参与融合的机器人数量 |
| **Precision-Communication Tradeoff** | 在满足最小精度要求 $ Q_{\min} $ 下最小化通信开销 |

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **Default（全参与）** | 所有范围内机器人都参与，使用 intermediate fusion |
| **Rule-based** | 若带宽 > 20 Mbps 使用 intermediate fusion，否则使用 late fusion |
| **Static fusion baselines** | 固定使用某一种 fusion 范式（如 early/late/intermediate） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）通信成本显著降低
- **相比 default 基线，CoAdapt 平均减少 38% 通信成本**。
- Llama3.3 表现最佳，在保持高精度的同时实现最大压缩。

#### （2）检测精度保持高位
- **AP@0.7 最高达 0.862（Llama3.3）**，接近甚至优于 rule-based 和 default 方法。
- 即使在低带宽下切换至 late fusion，精度下降可控（仅约 2–5%）。

#### （3）参与者选择合理且动态
- 平均减少 **26% 参与机器人数量**（最高达 40%，由 Llama3.3 实现）。
- LLM 成功能识别几何冗余机器人（如视野高度重叠者），并排除之，保留具有独特覆盖视角的机器人（如填补盲区）。

#### （4）融合策略动态切换
- **Gemma 4 和 Llama3.3 展现出丰富自适应行为**：
  - 高带宽时选择 intermediate fusion（精度高）
  - 低带宽时自动降级为 late fusion（通信高效）
  - 极少数情况下启用 early fusion（当带宽充足）
- GPT-OSS 模型更保守，几乎只选 late fusion，虽通信成本最低（<10KB/帧），但精度也最低（~0.77）。

#### （5）消融分析（隐含）
- 图 8 显示，在相同融合范式下，**CoAdapt 的 participant pruning 显著降低通信成本（最高达 50%）而精度无损**。
- 表明 SAM + LLM 的抽象与推理有效捕捉了空间互补性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 可作为有效的 autonomic fusion controller**：无需训练即可理解 LiDAR 场景语义，并做出合理的融合策略决策。
2. **联合优化显著优于单一维度优化**：同时调整 participant selection 与 fusion algorithm 比单独优化任一维度更能适应动态环境。
3. **结构化自然语言抽象是可行桥梁**：SAM 成功将异构物理观测转化为 LLM 可处理的信息形式，打通感知与认知层。
4. **CoAdapt 实现通信效率与感知精度的良好平衡**：在维持 comparable detection precision 的前提下，大幅降低通信负担。

### ⚠️ 方法的局限性
| 限制 | 说明 |
|------|------|
| **合成带宽模型** | 当前带宽变化为人工设定，未使用真实 IIoT 网络 trace，需后续实测验证 |
| **数据集规模有限** | OPV2V 最多支持 6 个 agent，难以评估更大 swarm 中的扩展性 |
| **缺少确定性回退机制** | 完全依赖 LLM 输出，若 LLM 出错（如 GPT-OSS 20B 未能识别冗余）无 fallback 方案 |
| **LLM 推理延迟较高** | 当前响应时间达 21–44 秒，远超实时控制需求，需引入量化、蒸馏等加速技术 |

### 🔮 未来工作方向
1. **实现 Network Control Layer**：主动调节无线资源（如优先级调度、带宽分配），形成闭环网络-感知协同优化。
2. **引入约束验证与规则回退层**：对 LLM 输出进行合法性检查，并在低置信度或异常时启用 rule-based 安全策略。
3. **真实平台部署与 sim-to-real 迁移**：在物理机器人平台上测试 CoAdapt 的实际表现。
4. **优化 LLM 推理效率**：集成模型量化（quantization）、知识蒸馏（distillation）、投机解码（speculative decoding）等技术降低延迟。

---

> ✅ **总结一句话**：  
> CoAdapt 首次将 LLM 用于 IIoT 机器人集群的协同感知控制，实现了无需训练的动态参与者选择与融合策略切换，在保持高检测精度的同时**降低 38% 通信成本**，为自适应 CPS 系统提供了新的设计范式。

</details>

---

### 10. [ViCo: Visual-oriented Coding with Self-Reflection for Chart Replication](https://arxiv.org/abs/2609.16014)

**Authors**: Jiaxin Duan, Dian Jiao Shuai Zhao, Jiabing Leng, Yiran Zhang, Feng Huang  
**Category**: cs.CL  
**Published**: 2026-09-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.16014v1  

#### Abstract
This paper addresses the challenge of generating high-quality academic charts that match the visual standards of human-authored papers. While existing AI agents can produce well-structured text and code, their generated visualizations often lack the stylistic and semantic fidelity of human designs. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ViCo: Visual-oriented Coding with Self-Reflection for Chart Replication 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文旨在解决**高质量学术图表自动生成**中的核心挑战：尽管现有的 AI 代理能够生成语法正确的代码和结构良好的文本，但其生成的可视化图表在**风格、布局和语义保真度**上仍远逊于人类设计。具体问题包括：

- **反思无效（Ineffective Reflection）**：模型无法准确识别图表差异的根本原因，导致反思流于表面甚至误导。
- **反思执行不一致（Say One Thing, Do Another）**：即使反思正确，后续代码生成可能忽略或误解建议，导致“言行不一”。
- **奖励稀疏（Reward Sparsity）**：强化学习（RL）中，全局奖励（最终图表相似度）仅在多步迭代结束后才获得，难以归因到具体的反思或编码步骤，阻碍有效学习。

---

### **提出了什么新方法或新思路**

作者提出 **ViCo**（Visual-oriented Coding），一种结合**自监督预热**与**反事实强化学习**的训练框架，实现高效的视觉导向代码生成。其核心创新包括：

#### ✅ **1. 自监督预热阶段（Self-Supervised Warm-Up with MCTS）**
- 使用 **Monte Carlo Tree Search (MCTS)** 在学生模型自身探索下合成高质量的反思-编码轨迹。
- 引入**一致性检查机制（Consistency Critic）** 和**单调性约束**，确保每一步编码都严格遵循前序反思，并带来奖励提升。
- 采用 **Jump-Pruning** 技术提取严格递增奖励的高效轨迹，用于行为克隆（SFT）。

#### ✅ **2. 反事实多步强化学习（Counterfactual-based Multi-step RL）**
- 针对每一轮反思-编码循环，分别估计**反思步骤**和**编码步骤**的优势（Advantage）：
  - **编码优势 A(a)**：固定反思，采样多个替代编码动作，评估其期望回报。
  - **反思优势 A(r)**：构造“空反思”（empty reflection）作为反事实基线，衡量反思本身带来的边际收益。
- 该机制提供**密集的、步骤级的学习信号**，有效缓解奖励稀疏问题。

#### ✅ **3. 基于 HHLG 的自动评估框架**
- 提出 **Hierarchical Heterogeneous Layout Graph (HHLG)** 结构，将图表建模为宏观树（多子图布局）和微观图（子图内元素关系）。
- 支持多维度、可解释的自动评估：
  - **语义一致性 (Ssem)**：文本内容对齐（OCR + 匈牙利算法）
  - **布局一致性 (Slay)**：树编辑距离 + 图编辑距离（GED）
  - **风格一致性 (Sstyle)**：Gram 矩阵距离（VGG 特征）
  - **颜色一致性 (Scolor)**：LAB 色彩空间下的感知差异

---

### **相比现有方法的优势**

| 维度 | ViCo 优势 |
|------|----------|
| **反思质量** | 通过 MCTS + 一致性检查，确保反思具有建设性且能被执行 |
| **信用分配** | 反事实机制实现细粒度的步骤级信用分配，优于传统 PPO/GRPO |
| **训练效率** | 无需昂贵的 LLM-as-a-judge，HHLG 提供高效、稳定、可扩展的奖励信号 |
| **通用性** | 框架独立于特定 backbone，在 Qwen3-VL 和 InternVL 上均表现优异 |

---

## 2. 核心实验方法和设置

### **使用了哪些数据集**

- **RealChart2Code**：大规模复杂图表基准，涵盖 50 种图表类型和多面板布局。
- **ChartMimic**：专注于从原始图像复制学术风格图表，测试视觉保真度和布局结构。
- **Plot2Code**：将科学论文中的图表映射为可执行代码，强调功能正确性和多样性。

> 所有训练数据均来自近 3 年顶会（NeurIPS, ACL 等）和 arXiv 的 50K 篇论文，共筛选出 60K 高质量图表，并经过严格的去重（MD5, pHash, CLIP 相似度过滤）。

---

### **实验设置和评估指标**

#### **训练配置**
- **Backbone**：Qwen3-VL-4B / 8B 和 InternVL-3.5-8B
- **训练流程**：
  1. **阶段一**：MCTS 自监督轨迹生成 + SFT 行为克隆
  2. **阶段二**：基于反事实 PPO 的多步 RL 微调
- **环境**：Docker 沙箱（Python 3.13, matplotlib, seaborn），超参见 Table 7

#### **评估指标**
| 指标 | 定义 |
|------|------|
| **Pass Rate (%)** | 生成代码无运行时错误的比例 |
| **Score / Rating** | 由 GPT-4o 或 Qwen-VL 等 LLM 作为 judge 给出的视觉保真度评分 |
| **Text Similarity** | 与真实代码的 token 级相似度 |
| **HHLG Score** | 四维加权得分：`R_fid = w1*Ssem + w2*Slay + w3*Sstyle + w4*Scolor` |

---

### **基线方法对比**

| 类型 | 代表模型 |
|------|--------|
| **闭源模型** | GPT-5.1, Claude-4.5-Opus, Gemini-3-Pro |
| **开源模型** | Qwen3-VL, InternVL-3.5, DeepSeek-VL, GLM-4.5V |
| **专用图表模型** | ChartCoder, VisRefiner, ChartSketcher |

---

## 3. 主要实验结果和性能指标

### **关键性能数据（Table 1）**

| Model | RealChart2Code (Pass%) | ChartMimic (Direct P.) | Plot2Code (Pass%) |
|-------|------------------------|----------------------|------------------|
| **Claude-4.5-Opus** | 87.7% | 98.5% | 99.6% |
| **GPT-5.1** | 71.2% | 97.8% | 98.5% |
| **Qwen3-VL (8B)** | 17.3% | 89.5% | 88.9% |
| **ViCo (Qwen3-VL, 8B)** | **98.8%** | **99.2%** | **99.1%** |

> ViCo 在 **Pass Rate** 上全面碾压所有基线，尤其在 RealChart2Code 上从 17.3% 提升至 98.8%，绝对增益达 **+81.5%**。

---

### **与基线方法的对比结果**

- **视觉评分极具竞争力**：
  - 在 ChartMimic 上达到 **81.5 分**，超过 Gemini-2.5-Flash (69.8)，接近 Claude-4.5-Sonnet (91.5)。
  - 在 Plot2Code 上获得 **8.4 Rating**，与 Gemini-2.5-Flash (8.6) 和 Claude-4.5-Sonnet (8.8) 接近。
- **小模型媲美大模型**：
  - ViCo-8B 性能接近甚至超越 200B+ 的闭源模型，证明其训练范式的有效性。

---

### **消融实验结果（Table 2 & Table 8）**

| 配置 | RealChart2Code (Pass%) | ChartMimic (Score) | 说明 |
|------|------------------------|--------------------|------|
| **Full ViCo** | 98.8% | 81.5 | 完整框架 |
| **w/o RL** | 96.5% | 77.1 | 仅预热，性能下降明显 |
| **w/o SFT** | 50.5% | 69.7 | 冷启动 RL 效果极差 |
| **HHLG → LLM-as-Judge** | 94.1% | 72.3 | 奖励信号不稳定，易被 exploit |
| **w/o Counterfactual** | 47.1% | 75.1 | 缺乏细粒度信用分配，训练崩溃 |

> **关键发现**：
> - MCTS 预热 + RL 微调 是必要组合。
> - 反事实机制对稳定性至关重要。
> - HHLG 奖励比 LLM 更可靠，尤其在复杂图表上。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **有效的自我反思需要机制保障**：单纯的“生成反思 + 修改代码”模式不可靠；必须通过**一致性检查**和**轨迹修剪**来强制“言行一致”。
2. ✅ **密集奖励信号是成功关键**：传统的终端奖励无法支撑多步反思的信用分配；**反事实优势估计**提供了稳定的步骤级梯度。
3. ✅ **轻量级结构化评估优于黑盒 LLM**：HHLG 提供**可解释、高效、抗 reward hacking** 的奖励，适合大规模训练。
4. ✅ **ViCo 具有强泛化能力**：在不同 backbone（Qwen3-VL vs InternVL）上均取得显著提升，验证其方法论的普适性。

---

### **局限性**

1. **复杂多子图仍具挑战**：
   - 在 >6 子图的密集布局中，视觉评分仍较低（1.2/10）。
   - 原因：视觉定位分辨率不足、元素密集导致解析困难。
   - **解决方案**：提出 “Divide-and-Conquer” 流水线，先分割再独立生成，评分从 1.2 提升至 **7.6**。

2. **RL 训练开销较高**：
   - 反事实采样需多次沙箱执行，计算成本随 `M` 线性增长。
   - 实验表明 `M=8` 是性价比最优选择。

3. **任务范围有限**：
   - 当前仅针对学术图表复制，尚未拓展至网页/UI 生成等更广泛场景。

---

### **未来工作方向**

- 设计更强大的**视觉-空间推理模块**以应对复杂布局。
- 探索**动态采样策略**降低反事实计算开销。
- 将 ViCo 框架推广至 **webpage-to-code**, **UI reconstruction**, **general visual design translation** 等任务。
- 构建更大规模、更具挑战性的 **multi-panel chart benchmark**。

---

> **总结**：ViCo 通过**结构化的反思机制**、**细粒度的信用分配**和**高效的自动评估**，系统性地解决了视觉代码生成中的核心难题。其实验结果表明，即使是 8B 规模的开源模型，也能通过 ViCo 训练达到媲美顶级闭源模型的图表复制能力，为自动化科研绘图提供了坚实的技术路径。

</details>

---

### 11. [Co-Skill: A Collaborative Communication Framework for Skill Evolution](https://arxiv.org/abs/2609.16008)

**Authors**: Yilin Ma, Yangi Pan, Weihao Yang, Peixin Zeng, Jiannan Xu, Hao Huang, Wen Xia  
**Category**: cs.DC  
**Published**: 2026-09-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.16008v1  

#### Abstract
Agent evolution through skills becomes critical for LLM-based agents to iteratively improve task success rate. Hybrid evolution is a cost-efficient paradigm where a cloud LLM analyzes and generates skills while an edge SLM executes and internalizes them. However, existing hybrid methods, such as Ski...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Co-Skill: A Collaborative Communication Framework for Skill Evolution

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现有基于 LLM 的智能体在技能演化（skill evolution）过程中面临两大挑战：

- **Cloud Communication Redundancy**：边缘端（edge SLM）向云端（cloud LLM）上传原始执行轨迹时，大量重复的前缀导致冗余 token 传输，浪费云侧计算资源。
- **Edge Execution Deficiency**：云侧生成的技能未考虑边缘 SLM 的实际执行能力，导致指令过于复杂、难以解析，影响任务成功率。

这两个问题的根本原因被归结为 **Blind Communication** —— 云与边之间缺乏相互感知能力。

---

### **提出了什么新方法或新思路**

作者提出 **Collaborative Communication Framework (CCF)**，通过双向感知机制实现高效的云-边协同技能演化。该框架包含三个核心技术组件：

1. **Cloud-aware Prefix-Merged Trajectory Trie**  
   - 边缘端将多个共享前缀的轨迹合并成一棵 trie 树，仅上传分歧点（divergence points），显著压缩上传数据量。
   - 每个节点记录动作、成功/失败统计，便于云侧快速定位问题。

2. **Edge-aware Progressive Skill Tree**  
   - 云侧以树形结构渐进式生成技能：从高层目标（L0）逐步细化到具体操作步骤（如 L3）。
   - 只有当某分支的成功率不足时才进一步扩展子节点，确保技能深度与 SLM 执行能力匹配。

3. **Collaborative Skill Evolution**  
   - 云侧维护一个可持久化的分层 Skill Library，支持跨任务复用。
   - 边缘端可通过可选的 RL 模块（如 GRPO）内化技能，并采用自顶向下剪枝策略优化参数吸收效率。

---

### **相比现有方法的优势**

| 维度 | 优势 |
|------|------|
| **通信效率** | 减少 15.3%-40.5% 的上传 token，避免重复推理 |
| **执行效率** | 技能按需生成，适配 SLM 容量，提升成功率 |
| **演化效率** | 分离式演化机制：云负责分析与抽象，边负责执行与微调，各司其职 |
| **成本控制** | 总体 SLM+LLM token 使用降低 15.6%-41.9%，且训练后推理开销更低 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **ALFWorld**：文本版家庭环境模拟器，包含六类多步任务：
  - Pick-and-place (Pick)
  - Examine-in-light (Look)
  - Clean-and-place (Clean)
  - Heat-and-place (Heat)
  - Cool-and-place (Cool)
  - Pick-two-and-place (Pick-2)

- **WebShop**：电商购物基准，要求代理根据自然语言描述搜索并购买商品。

---

### **实验设置和评估指标**

| 设置项 | 描述 |
|-------|------|
| 每轮最大步数 | ALFWorld: 40 步；WebShop: 15 步 |
| 停止条件 | 连续10步平均成功率 >90%，且无单步低于85% |
| 评估周期 | 每训练阶段报告任务成功率与 token 消耗 |
| 主要指标 | 
| - **Task Success Rate** | 成功完成任务的比例 |
| - **Total Token Usage (SLM + LLM)** | 单 episode 平均消耗 token 数 |
| - **Convergence Speed** | 达到稳定高成功率所需的训练步数 |

---

### **基线方法对比**

| 方法 | 类型 | 特点 |
|------|------|------|
| **DeepSeek-V4-Pro** | Cloud-only | 强大推理能力，但无技能记忆，高 token 开销 |
| **Skill0 (Lu et al.)** | Edge-only | 本地 RL 微调，无需云协助，受限于 SLM 推理能力 |
| **SkillRL (Xia et al.)** | Hybrid | 当前 SOTA 混合范式：云生成技能 + 边执行 + RL 微调 |

> Co-SKILL 在相同设置下进行比较，默认关闭 RL 以公平对比通信与结构设计的影响。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **任务成功率（Success Rate）**

| 数据集 | 方法 | Success Rate |
|--------|------|-------------|
| ALFWorld | Co-SKILL | **88.9% – 90.5%** |
| ALFWorld | SkillRL | ~60%–75% |
| ALFWorld | Skill0 | ~50%–65% |
| ALFWorld | DeepSeek-V4-Pro | ~45%–85%（简单任务尚可，复杂任务下降明显） |
| WebShop | Co-SKILL | **~24.5%** |
| WebShop | DeepSeek-V4-Pro | 25.0% |
| WebShop | SkillRL / Skill0 | <20% |

> Co-SKILL 在 ALFWorld 上 **提升 25.8%-76.4%** 的成功率，在 WebShop 上接近云模型上限。

---

#### ✅ **总 Token 消耗（SLM + LLM）**

![Figure 6](#) 显示每 episode 的平均 token 消耗：

| 任务 | Co-SKILL vs SkillRL | 节省幅度 |
|------|---------------------|----------|
| ALFWorld 各任务 | **↓15.6% – 41.9%** | 最高达 41.9% |
| WebShop | ↓~30% | 显著优于 SkillRL 和 Skill0 |

> 尽管云侧分析更细致（单次 token 更多），但由于 trie 压缩与 skill 复用，总体仍大幅节省。

---

#### ✅ **训练后推理 token 消耗**

- Co-SKILL 在收敛后，**每 episode 边缘 token 下降 61.7%-65.6%（vs SkillRL）**，16.2%-41.2%（vs Skill0）
- 因 skill library 支持直接检索，无需重复云交互

---

### **消融实验结果（Ablation Studies）**

#### 🔹 **Prefix-Merged Trajectory Trie 效果**
- 图7显示：使用 trie 后，上传 token 减少 **25.0%-41.8%**
- 若不使用 trie（raw concatenation），token 消耗显著上升

#### 🔹 **Progressive Skill Tree 有效性**
- 图8表明：随着 skill tree 层数增加，成功率先升后降
- 不同任务最优深度不同（如 Heat 需 L3，Pick 只需 L1）
- 过深会导致 SLM “过载”，说明渐进式生成至关重要

#### 🔹 **Collaborative Skill Evolution 分析**
- **With RL**：可加速收敛（见图9），但在部分任务上表现不稳定
- **Without RL**：已能取得优异效果，证明 CCF 结构本身是主因
- **Skill Library 复用**：重访旧任务时无需重新训练，零额外 token 开销

---

## 4. 关键结论和发现

### **主要发现**

1. **Blind Communication 是混合演化低效的根本瓶颈**
   - 云不了解边的能力 → 生成无效技能
   - 边不了解云的需求 → 上传冗余轨迹
   - CCF 通过双向感知打破这一僵局

2. **结构化通信显著提升效率**
   - Trie 压缩 + Tree 生成 构成了高效的信息交换协议
   - 类似 KV-cache 的思想被推广至跨设备协作场景

3. **分离式演化机制更合理**
   - 云擅长抽象与归纳 → 适合构建 skill library
   - 边擅长快速执行与微调 → 适合 RL 内化技能
   - 各展所长，实现“协同进化”

4. **Co-SKILL 在性能与成本间取得最佳平衡**
   - 接近甚至超越纯云方案的任务成功率
   - 显著低于所有 baseline 的 token 消耗

---

### **局限性**

1. **依赖离线仿真环境（emulation-based evaluation）**
   - 实验基于 ALFWorld 和 WebShop 等文本环境
   - 缺乏视觉输入（vision-language）的真实交互场景验证

2. **尚未实现端到端在线演化（end-to-end online evolution）**
   - 当前为异步批量处理模式
   - 实际部署中需解决异步 RL 中的 off-policy 问题

3. **Skill Tree 构建依赖人工启发式规则**
   - 如何自动判断是否需要扩展节点仍有改进空间

---

### **未来工作方向**

1. **扩展至 Vision-Language Models (VLM)**
   - 将 CCF 应用于具身智能（embodied agents）或多模态任务

2. **构建异步在线学习系统**
   - 设计稳定的异步 RL loop，支持实时反馈与持续演化

3. **自动化 Skill Tree 生长策略**
   - 利用元学习或强化学习动态决定技能细化时机与粒度

4. **跨设备迁移与联邦式技能共享**
   - 多个 edge agent 共享 skill library，形成 collective intelligence

--- 

> **总结一句话**：  
> Co-SKILL 通过 **Collaborative Communication Framework** 实现了云-边智能体之间的高效协同演化，在 **提升任务成功率的同时显著降低 token 消耗**，为低成本、高性能的 agent evolution 提供了新的范式。

</details>

---

### 12. [Calibrate, Then Route: A Measured Study of Learned Request Routing for Disaggregated LLM Serving](https://arxiv.org/abs/2609.16206)

**Authors**: Srikanta Datta Tumkur, Jay Iyer, Mehar Simhadri, Sai Pavan Kumar, Sai Kapil Kumar, Ramesh Nampelly  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.16206v1  

#### Abstract
Disaggregated LLM serving places compute heavy prefill and memory heavy decode on separate GPU pools. Systems such as DistServe, Splitwise, and Mooncake make this separation fast, but routing still determines which instances handle each request. We study a router that estimates the additional comple...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Calibrate, Then Route: A Measured Study of Learned Request Routing for Disaggregated LLM Serving*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **disaggregated LLM serving** 架构中，prefill 阶段（计算密集）和 decode 阶段（内存密集）被分离到不同的 GPU 池上执行。虽然已有系统如 DistServe、Splitwise 和 Mooncake 实现了这种架构以提升 **goodput**，但它们通常采用简单的路由策略（如 round-robin 或 join-shortest-queue），这些策略对请求本身的特征（如 prompt 长度、预期输出长度等）是“盲”的。

这导致长请求可能被分配到看似空闲但实际上即将拥塞的 decode 实例，从而违反 SLO 并影响后续请求。

### 提出的新方法
本文提出了一种 **learned request router**，其核心思想是：
> 在 admission time 对每个候选实例进行 **marginal-cost scoring**，选择使该请求完成时间最小的 prefill 和 decode 实例。

#### 路由评分函数（cost function）
对于 decode 实例 $d$，成本定义为：
$$
\text{cost}(d) = w_b \cdot \text{backlog}(d) \cdot o(s_r) + w_s \cdot \hat{o}_r \cdot T_{\text{dec}} \cdot (1 + \text{press}^+(d)) \cdot o(s_r) + w_q \cdot \text{queue}(d)
$$
其中关键特征包括：
- **prompt length**（精确已知）
- **predicted output length**（预测值）
- **post-admission KV-cache pressure**（缓存压力）
- **SLO class**（服务等级目标）

类似地，prefill 实例也基于 $p_r \cdot T_{\text{pre}}$ 进行评分。

### 相比现有方法的优势
- **超越传统策略**：相比 round-robin、least-loaded（JSQ）、length heuristic，在混合突发流量下实现更高且更稳定的 **goodput**。
- **硬件感知校准（Calibration）至关重要**：必须使用真实硬件测量得到的 $T_{\text{pre}}, T_{\text{dec}}$ 等常量，否则性能大幅下降。
- **用更少资源达到相同性能**：学习型路由器在 **6个GPU** 上即可达到 round-robin 在 **7个GPU** 上的 goodput，节省约 **14%** 的资源。

---

## 2. 核心实验方法和设置

### 实验平台（Testbed）
- **硬件**：单节点上的 **8块 NVIDIA A40 GPU**
- **模型**：Qwen2.5-3B（BF16, TP=1）
- **推理引擎**：每个 GPU 运行一个独立的 **vLLM 0.12** 实例
- **KV Cache 传输**：通过 **NIXL** 在 prefill 和 decode 池之间迁移
- **KV 缓存配额**：每实例固定为 60,000 tokens
- **拓扑结构**：默认为 2 prefill + 4 decode 实例；研究不同宽度时扩展至最多 8 GPU

### 工作负载（Workloads）
| 工作负载 | 特征 | 压力点 |
|--------|------|-------|
| **Chat** | 短输入，短输出 | admission rate |
| **Document** | 最长达 4k 输入，短输出 | prefill compute, KV transfer |
| **Reasoning** | 短输入，长输出 | decode occupancy |
| **Mixed bursty** | 异构、突发性流量 | 路由决策本身 |

> 所有 workload 均在其 **measured saturation point** 下运行，即 round-robin 开始崩溃的请求速率。

### 评估指标
- **Goodput (%)**：满足 SLO（TTFT ≤ 175ms, TPOT ≤ 39.7ms）的请求占比
- **P99 TTFT / P99 TPOT**：尾部延迟
- **Trace variance**：跨多个 trace 的稳定性
- **Ablation studies**：移除单一特征的影响

### 基线方法（Baselines）
1. **Round-robin**
2. **Least-loaded (Join-Shortest-Queue, JSQ)**
3. **Heuristic length threshold**：基于预测输出长度设定阈值进行调度

---

## 3. 主要实验结果和性能指标

### 总体性能对比（Table II）
在 **mixed bursty traffic** 下平均三个 trace 的结果：

| Policy | Goodput (%) | P99 TTFT (s) | P99 TPOT (s) |
|--------|-------------|--------------|---------------|
| Round-robin | 0.836 | 0.337 | 0.018 |
| Least-loaded (JSQ) | 0.835 | 0.346 | 0.018 |
| Heuristic | 0.847 | 0.318 | 0.018 |
| **Learned Router (Ours)** | **0.864** | **0.321** | **0.018** |

✅ **结论**：学习型路由器取得最高 **mean goodput** 和最低 trace 间方差，优于所有 baseline。

---

### 关键性能发现

#### ✅ 学习型路由器显著提升性能
- 在 mixed bursty 流量下，**平均 goodput 达 0.864**，比最优 baseline（heuristic）高出 **+1.7个百分点**。
- 在所有三个 trace 中均领先 round-robin 和 heuristic；仅在一个 trace 上略低于 JSQ（差 0.003），但在另两个 trace 上明显领先。
- 尾部延迟更低：P99 TTFT 比 round-robin 降低 **~5%**。

#### ✅ 校准（Calibration）带来巨大收益
- 若使用模拟器推导的成本常量（未实测校准），goodput 从 **0.864 降至 0.819**（**损失 4.5 个百分点**）。
- 尾部延迟恶化：P99 TTFT 从 ~0.3s 升至 **0.42–0.52s**。
- 原因：错误的 $T_{\text{dec}}$ 导致 marginal-cost scoring 退化为简单的 queue counting。

#### ✅ 输出长度预测是关键特征（Ablation Study）
- 移除 **predicted output length** 后：
  - 正常负载下损失 **3.2 个百分点**
  - 饱和负载下损失 **4.7 个百分点**
- 其他特征影响较小：
  - **cache pressure**：无显著影响（因 KV budget 未饱和）
  - **SLO class**：中等负载有益，深度饱和时轻微反作用
  - **prompt length**：非主导因素

#### ✅ 对预测误差鲁棒性强
- 注入高达 **150% 的相对误差** 到 output length predictor：
  - goodput 变化平缓（从 0.838 到 0.880），仍在噪声范围内
  - 仍能击败 heuristic
- 表明：**只要有粗略的长度估计即可**，无需高精度模型

#### ✅ 池宽度（Pool Width）影响优势大小
| Decode Width | Learned Router | Best Baseline | 结果 |
|-------------|----------------|----------------|------|
| 3           | 0.816          | JSQ 0.822       | 被超越 |
| 4           | 0.864          | JSQ 0.835       | 明显领先 |
| 6           | 0.858          | RR 0.860        | 接近持平 |

➡️ **结论**：当 decode pool 宽度 ≥4 时，learned routing 才能发挥优势；太窄时 queue count 几乎足够。

#### ✅ 极端资源稀缺时性能反转
- 在极度受限配置（1 prefill + 2 decode）下：
  - Learned router goodput 崩溃至 **0.292**
  - Round-robin 仍维持 **0.680**
- 原因：贪婪的 argmin 决策会集中负载到“暂时最便宜”的实例，造成局部过载。
- 建议：结合 **power-of-two-choices** 或 admission guard 缓解。

#### ✅ 资源效率优势
- 如 Fig. 4 所示：
  - Learned router 在 **6 GPU** 上达到的 goodput，round-robin 需要 **7 GPU** 才能达到。
  - ➡️ 实现 **~14% 更高的资源利用率**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Learned routing + marginal-cost scoring** 在真实 disaggregated 集群上可显著提升 goodput，尤其在 **异构、突发性流量** 下表现最佳。
2. ✅ **Hardware calibration 是方法不可分割的一部分**：脱离真实硬件测量的 cost constants 会使 learned router 退化为 queue counter。
3. ✅ **Predicted output length 是最关键的特征**，其他信号增益有限。
4. ✅ 方法对 predictor error 高度鲁棒，**粗略估计即可**。
5. ⚠️ 方法优势依赖于条件：
   - **pool width ≥ 4**
   - **traffic heterogeneity 高**
   - **集群处于争用状态但未完全饱和**
6. ❌ 在极端稀缺场景下，greedy 成本最小化反而不如 blind spreading。

### 局限性
- 实验仅在 **单节点、同构 GPU（A40）** 上进行，未测试跨节点 RDMA 通信开销。
- 使用的是 **3B 规模模型**，更大模型或更长上下文下的 KV cache fragmentation 影响未体现。
- Scoring function 权重为手动设置，未自动学习。
- Cache pressure 特征未发挥作用（因 budget 设置宽松），未能验证其价值。

### 未来工作方向
- 引入 **power-of-two-choices** 或采样机制避免极端集中。
- 加入 **KV cache locality** 作为新特征（如 Mooncake Conductor 所示）。
- 支持跨节点部署，加入 **transfer cost term**。
- 自动学习 scoring function 的权重。
- 在更大模型、更复杂 workload 下验证泛化能力。

---

## 总结（TL;DR）

> 本文提出了一个面向 **disaggregated LLM serving** 的 **learned request router**，通过在 admission time 使用 marginal-cost scoring（基于 prompt length、predicted output length、KV cache pressure、SLO class）来优化 prefill/decode 实例分配。  
>
> 在真实 vLLM/NIXL 集群上的实验表明：
> - 该方法在混合突发流量下实现了 **最高 goodput（0.864）和最低方差**；
> - **硬件校准** 至关重要，可单独贡献 **+4.5 个百分点**；
> - **预测输出长度** 是最关键特征；
> - 可在 **6 GPU 上达到 round-robin 在 7 GPU 上的性能**，节省约 14% 资源；
> - 优势依赖于 pool width ≥4 和 traffic heterogeneity；
> - 在极端稀缺时会失效，需配合 sampling 策略。
>
> **核心洞见**：*“Learned scheduler 必须与硬件联合校准，否则只是 fancy queue counter.”*

</details>

---

### 13. [little m: An AI Agent for Industrial Process Optimization](https://arxiv.org/abs/2609.16680)

**Authors**: Yongchao Ye, Xinyu He, Dutliff Boshoff, Way Kuo, Lishuai Li  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.16680v1  

#### Abstract
Manufacturing consumes one third of global energy and still has significant room for improvement in terms of energy efficiency. Optimal process control is essential for this purpose. However, synthesizing mathematical optimization models from messy, real-world industrial specifications requires brid...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*little m: An AI Agent for Industrial Process Optimization*

## 1. 论文的主要贡献和创新点

### 解决的问题
工业过程优化中，将非结构化的自然语言描述和流程图（Process Diagrams）转化为精确的数学优化模型是一项极具挑战性的任务。通用的 **Large Language Models (LLMs)** 在处理连续多物理场（multi-physics dynamics）系统时容易引入**物理上不一致或逻辑错误的约束**，导致生成的模型无法用于实际工程部署。

此外，现有基准（如 NL4OPT、ComplexOR）大多关注物流调度等离散优化问题，缺乏对**连续过程控制场景**（涉及微分方程、能量平衡、动态约束）的有效评估。

### 提出的新方法与创新思路
作者提出了 **little m** —— 一个面向工业过程优化建模的 AI Agent，其核心创新在于：

- **知识增强的三阶段认知流水线（Three-stage Cognitive Pipeline）**：
  1. **Information Structuring**：从文本和图表中提取实体、变量、目标，并识别信息缺口。
  2. **Strategy Design**：结合领域知识库（Knowledge Repository），提出高层控制策略（如使用 MPC 还是 RTO）。
  3. **Mathematical Modeling**：基于确认的策略生成形式化数学模型（目标函数、决策变量、约束）。

- **领域特定的知识库（Domain-specific Knowledge Repository）**：
  - 包含 55 个自包含的“问题路径”条目，每个条目连接**可观测症状 → 优化目标 → 候选方法 → 数学模式 → 适用边界**。
  - 支持检索增强生成（RAG），提供可审计的工程依据。

- **交互式精炼机制（Interactive Refinement）**：
  - 允许领域专家在每一阶段审查输出并反馈，形成闭环迭代，有效处理“隐性知识”（tacit knowledge）。

### 相比现有方法的优势
| 维度 | 传统 LLMs | little m |
|------|-----------|--------|
| **建模准确性** | 易产生物理无效约束 | 通过知识库和结构化流程提升语义正确性 |
| **可解释性与可审计性** | 黑箱生成 | 分阶段输出，支持人工审查 |
| **多模态理解** | 弱于图结构推理 | 显式融合文本 + 流程图输入 |
| **工程实用性** | 生成即用代码 | 输出为待验证的候选模型，强调人机协作 |

---

## 2. 核心实验方法和设置

### 数据集
- **IPC-Bench**：本文提出的首个面向工业过程控制的多模态基准数据集。
  - **规模**：50 个典型场景，源自经典教材（如 Seborg, Edgar 等）。
  - **类型覆盖**：反应（Reaction）、分离（Separation）、热工（Thermal）、公用工程/调度（Utility/Scheduling）。
  - **输入模态**：文本描述 + 工艺流程图（P&ID 或 schematic）。
  - **输出标注**：专家手工构建的 Ground Truth 模型，明确划分：
    - 决策变量 $ V $
    - 目标函数 $ F $
    - 约束集合 $ C $

### 实验设置与评估指标

#### 双重视角评估框架：
1. **双盲人类专家评估（Double-blind Human Evaluation）**
   - **评审团**：8 名来自计算机科学、数据科学、控制工程领域的博士/博后。
   - **评估维度**（每项满分制）：
     - Objective Function Quality
     - Decision Variable Completeness
     - Constraint Validity
     - Overall Convincingness
   - **方式**：随机顺序展示匿名模型（little m, Qwen3, DeepSeek），由 2–4 位专家独立打分。

2. **自动化机器评估（Automated Machine Evaluation）**
   - 对预测模型 $ M_p = \{V_p, f_p, C_p\} $ 与真值 $ M_g = \{V_g, f_g, C_g\} $ 进行结构比对：
     - **决策变量**：Jaccard 相似度（经 LLM 语义对齐）
     - **目标函数**：去噪后 Token 级 Jaccard 相似度
     - **约束集**：基于匈牙利算法的 bipartite matching，计算连续 F1 分数（兼顾 Precision 和 Recall）

### 基线方法对比
- **Qwen3-Next-80B-A3B-Instruct**
- **DeepSeek-V3.2**

两者均为当前最先进的通用 LLM，在相同输入下直接生成数学模型。

---

## 3. 主要实验结果和性能指标

### 人类评估结果（Win Rate, N=20 cases）
| 评估维度 | Qwen3 | DeepSeek | **little m** | p-value (vs. random baseline) |
|---------|-------|----------|-------------|-------------------------------|
| Objective Function | 16.0% | 26.0% | **58.0%** | <0.001 *** |
| Decision Variables | 14.0% | 26.0% | **60.0%** | <0.001 *** |
| Constraints | 22.0% | 26.0% | **52.0%** | 0.007 ** |
| **Overall Quality** | 16.0% | 18.0% | **66.0%** | <0.001 *** |

> ✅ **结论**：little m 在所有维度均显著优于两个基线模型（p < 0.01），尤其在整体质量和变量完整性方面优势明显。

### 机器评估得分（Structural Accuracy）
| 指标 | Qwen3 | DeepSeek | **little m** |
|------|-------|----------|-------------|
| Decision Variables | 0.673 | 0.699 | **0.733** |
| Objective Functions | 0.526 | **0.553** | 0.518 |
| Constraints | 0.389 | 0.395 | **0.418** |

> ✅ **结论**：
> - little m 在 **Decision Variables** 和 **Constraints** 上取得最高分；
> - DeepSeek 在目标函数上略优，可能因其更强的语言表达能力；
> - little m 的结构一致性更优，尤其在复杂约束建模上表现稳健。

### 消融实验结果（Ablation Study）
移除关键组件后的性能下降（machine-based evaluation）：

| 消融条件 | Decision Variables | Objective | Constraints |
|--------|------------------|----------|------------|
| 完整版 little m | 0.733 | 0.518 | 0.418 |
| w/o Knowledge | 0.720 | 0.510 | **0.381** ↓ |
| w/o Diagrams | 0.723 | 0.468 ↓ | **0.383** ↓ |

> 🔍 **发现**：
> - 移除 **Knowledge** 导致约束建模大幅下降 → 验证了知识库对物理规则建模的关键作用。
> - 移除 **Diagrams** 影响目标函数和约束 → 表明流程图对于理解系统拓扑和依赖关系至关重要。

### 其他重要实验分析
- **交互恢复能力测试**：当输入信息不完整时，little m 通过提问澄清可显著提升建模质量（尤其是变量和约束），接近完整输入的表现。
- **跨领域性能分析**：little m 在 **Reaction** 和 **Thermal** 场景中表现最佳，与其知识库覆盖内容一致；DeepSeek 在 Separation 任务中有专长。
- **复杂度影响**：随着变量和约束数量增加，所有模型性能下降，但 little m 在中低复杂度问题中优势最明显。

---

## 4. 关键结论和发现

### 主要发现
1. **结构化+知识增强的 Agent 架构优于端到端 LLM**：
   - 将建模任务分解为三个阶段，并引入领域知识，显著提升了生成模型的**语义正确性和物理一致性**。
   
2. **人机协同是工业级建模的关键**：
   - 通过阶段性审查和交互式澄清，能有效弥补 LLM 对“隐性知识”的缺失，提高工程可信度。

3. **IPC-Bench 揭示了现有 LLM 的短板**：
   - 即使是最先进的 LLM，在涉及**动态方程、守恒律、操作互锁**等问题上仍频繁出错，亟需专用架构支持。

4. **知识库与多模态输入具有互补价值**：
   - 知识库提供通用模式，流程图提供具体拓扑，二者缺一不可。

### 局限性
1. **知识库扩展成本高**：需要专家持续维护和标注，限制了跨行业泛化能力。
2. **无求解器验证**：未检测数值可行性或稳定性，仅评估公式层面的质量。
3. **数据集规模有限**：IPC-Bench 仅含 50 个教科书案例，可能存在预训练污染风险。
4. **自动化指标仍有偏差**：Jaccard 和 F1 可能惩罚等价但形式不同的正确表达式。

### 未来工作方向
- 将 little m 与 **numerical solvers** 集成，实现从建模到仿真的闭环。
- 扩展至更多真实工业项目，构建更大规模、去偏的 benchmark。
- 探索自动知识抽取技术，降低知识库构建门槛。
- 引入 **solver-in-the-loop** 验证机制，进一步过滤不可行模型。

---

> 📌 **总结一句话**：  
> *little m* 通过“**结构化流程 + 领域知识 + 人机交互**”三位一体的设计，成功提升了 LLM 在工业过程建模中的准确性和可信度，为 AI 辅助工程设计提供了可审计、可迭代的新范式。

</details>

---

### 14. [FlashVector: Agent for Hierarchical Model Serving Stack Optimization](https://arxiv.org/abs/2609.17391)

**Authors**: Qi Wu, Lohan Lemire, Kai Meng, Zhongmou Cai, Raphael Bargues, Petr Zhitnikov, Zeyuan Cao, Yao Wang, Shujun Bian, Wei Chen, Sean Sheng  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.17391v1  

#### Abstract
Model serving is one of the largest cost drivers in production recommender systems. Maximizing its throughput requires navigating a deeply layered hierarchy: GPU kernels, the ML framework computation graph, the model server, and on-demand feature processing -- each demanding specialized domain exper...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FlashVector: Agent for Hierarchical Model Serving Stack Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代推荐系统中，**model serving** 是生产环境中最大的成本驱动因素之一。其性能优化涉及一个复杂的分层架构，包括：
- GPU kernels
- ML framework computation graph
- Model server（如 NVIDIA Triton）
- On-demand feature processing

每一层都有不同的技术栈（CUDA、C++、Python、Go）、专用性能分析工具（Nsight、eBPF）和优化策略。传统上，跨层优化依赖于少数具备全栈专业知识的工程师，难以规模化，且随着模型重训练、流量变化和硬件更新而迅速失效。

因此，**核心问题是：如何实现对整个 model serving stack 的自动化、持续化、跨语言/跨组件的端到端性能优化？**

---

### **提出了什么新方法或新思路**

作者提出 **FlashVector** —— 一种基于 **AI agent** 的分层优化系统，其核心创新包括：

#### ✅ **Layer Agent 抽象（Layer Agent Abstraction）**
- 将每一层（GPU kernels、computation graph、model server、feature processing）抽象为一个统一的 **agent 接口**，遵循相同的四阶段闭环流程：
  1. **Profile**：使用该层专用工具采集性能数据（如 Nsight for GPU，eBPF for CPU）
  2. **Diagnose**：结合领域知识库（knowledge base）识别瓶颈
  3. **Optimize**：生成代码或配置修改建议
  4. **Verify**：验证正确性和性能增益
  5. **Refine**（后置）：将成功优化写回知识库，形成持续学习机制

> 这使得单一 agent 框架可扩展至异构系统，无需为每层单独设计优化器。

#### ✅ **Optimize Locally, Verify Globally**
- 各层 agent 独立提出优化方案（local optimization）
- 但只有在 **端到端负载测试中通过验证**（global verification），且性能提升超过测量噪声时才被接受
- 验证方式：使用 **replayed production traffic** 在固定 latency SLO 下测试 throughput

#### ✅ **Always-on Optimization Loop**
- 优化不是一次性任务，而是**自动触发的持续循环**
- 每次运行从历史变更记录开始，输出结果写回知识库，并自动重新触发
- 应对模型重训、流量漂移、硬件升级等动态变化，防止优化“过期”

---

### **相比现有方法的优势**

| 对比维度 | 现有方法 | FlashVector |
|--------|--------|-----------|
| **优化范围** | 单一层（如仅 kernel 或仅 batching） | 全栈联合优化（code + config） |
| **自动化程度** | 手动调优或专用搜索系统（如 Morphling） | 统一 agent 框架自动探索 |
| **适应性** | 固定策略，无法随环境变化自适应 | 持续运行，自动响应变化 |
| **通用性** | 针对特定瓶颈设计（如 vLLM 专注 KV cache） | 可插拔 layer agent，支持多语言/多组件 |

> FlashVector 不是替代已有系统，而是提供一个**通用、可扩展的自动化优化基础设施**。

---

## **2. 核心实验方法和设置**

### **使用了哪些数据集**
- **真实生产数据**：来自 Unity Vector 广告平台的实际线上请求流量
- **无公开 benchmark 数据集**，所有实验均基于 Unity 内部部署的 DNN 模型和服务栈

### **实验设置**
- **目标系统**：Unity Vector 平台的 model serving 架构
  - Model Server：NVIDIA Triton Inference Server（C++/Python）
  - ML Framework：PyTorch + AOTInductor 编译
  - Hardware：NVIDIA RTX PRO 6000 Blackwell GPU
  - Feature Processing：Python 后端服务，处理字符串、时间戳、嵌套字典等
- **负载特征**：
  - 每秒数十万 inference 请求
  - Batch size 达 2000+
  - 多任务、多塔结构模型（multi-tower retrieval, multi-task ranking）

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Throughput (RPS)** | 每秒请求数，在固定 latency SLO 下测量 |
| **Latency Speedup** | 相比 baseline 的延迟降低倍数（p99 latency） |
| **End-to-end Gain** | 必须超过 noise floor（±6%），否则视为无效 |
| **Correctness** | 输出需满足精度要求：<br>• Bit-identical（确定性变换）<br>• ≤1e-3 abs deviation（浮点重排）<br>• Business metric gate（近似计算） |

### **基线方法对比**
- **Baseline**：未优化的原始部署（AOT-compiled PyTorch + 默认 Triton 配置）
- **Human Expert Tuning**：由工程师手动优化的结果
- **No-op Runs**：重复运行无变更的 baseline，用于估计 noise floor（±6%）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**

| Model | Latency Speedup | Throughput Increase |
|-------|------------------|----------------------|
| Retrieval model | 1.67× | 1.10× |
| Ranking model 1 | 1.36× | 1.80× |
| **Ranking model 2** | **1.98×** | **2.00×** |
| Ranking model 3 | 1.30× | 1.33× |
| Ranking model 4 | 1.30× | 1.35× |
| Gamer model | 1.34× | 1.15× |

> **最高达 2× throughput 提升，1.98× 延迟下降**

---

### **各层优化案例与效果**

#### **Case 1: Model Server 输入反序列化优化**
- **问题**：Triton Server 到 Python Backend 的 string 输入反序列化慢
- **优化**：将 Python 实现改为 C++
- **结果**：**30× 加速**，已贡献回 [Triton 开源项目 #8348](https://github.com/triton-inference-server/server/issues/8348)

#### **Case 2: GPU Kernel 融合与 Embedding 层重构**
- **Fused Attention**：
  - 将 6 个 kernel 合并为 1 个 Triton kernel
  - 中间值保留在 register，减少内存访问
  - **Attention latency 从 996.5μs → 279.8μs（3.56×）**
  - 输出 bit-identical
- **Embedding Layer 优化**：
  - 分解 `embedding_bag` 为细粒度 primitive，启用 kernel fusion
  - 复用 target/sequence embeddings
  - **GPU kernel time ↓33.4%，dispatcher overhead ↓44.8%→28.5%**

#### **Case 3: On-demand Feature Processing 优化**
- **原瓶颈**：Python 中字符串解析、pandas 操作、tensor 拼接
- **优化措施**：
  - 引入 **Cython kernels** 替代 pandas 和 NumPy（见下表）
  - 零拷贝 request I/O（native view + shared memory probing）
  - 直接写入预分配 tensor slice，避免 concat
- **结果**：单个 preprocessor pod 吞吐从 **940 → 1500 RPS（1.6×）**

| Python Function | Replacement | Speedup |
|------------------|-------------|---------|
| `pd.to_datetime` | Cython | 350× |
| per-feature assignment | Direct slice write | 25.1× |
| bytes → unicode | C++ walk | 12.4× |
| ufunc dispatch ×180 sites | fused | ~8× |
| per-row dict lookup | optimized loop | 4.4–5.8× |

#### **Case 4: Serving 参数自动调优（Dynamic Batching, Instance Count）**
- **方法**：agent 自动 sweep 配置空间，在 shadow-traffic canary 中验证
- **结果**：相同 latency SLO 下，throughput 差异超 **2×**
- **Table 4 示例**：

| Model | SLO (ms) | Base (inst/batch) | Tuned (inst/batch) | RPS (tuned) | p99 (ms) |
|-------|----------|-------------------|--------------------|-------------|----------|
| Model A | 250 | 7/64 | 6/6 | 1321 | 192 |
| Model B | 250 | 10/64 | 7/3 | 736 | 271 |
| Model C | 300 | 8/2 | 4/2 | 238 | 251 |
| Model D | 400 | 12/3 | 3/1 | 139 | 285 |

> 显示默认配置严重次优，必须联合调优 instance 数与 batch size

---

### **消融实验（隐含）**
虽然未明确列出消融表，但从迭代过程可见：
- **顺序优化带来复合增益**（compounding gains）：
  - 先优化 feature processing → 再优化 model server → 最后调参
  - 每轮重新 profiling，暴露新的主导瓶颈
- 若不进行 global verification，局部优化可能导致端到端退化

---

## **4. 关键结论和发现**

### **主要发现**
1. **Model serving stack 的性能瓶颈广泛分布于各层**，不仅限于 GPU kernel
   - CPU-bound 的 feature processing、serialization、Python overhead 同样关键
2. **同一 agent 框架可有效应用于异构技术栈**（CUDA、C++、Python、config）
   - 证明了 “agentic paradigm” 的可泛化性
3. **代码优化 + 参数调优 必须协同进行**
   - 单独优化任一方都会留下显著效率缺口
4. **优化是持续过程而非一次性任务**
   - 模型更新、流量变化、硬件升级均使其快速过时
   - “always-on loop” 是维持长期效率的关键

---

### **方法的局限性**
1. **依赖高质量 profiling 工具和 knowledge base**
   - 若某层缺乏可观测性（如闭源组件），诊断能力受限
2. **LLM agent 可能产生错误或不可靠代码**
   - 依赖严格的 verify 阶段过滤
3. **初始 setup 成本高**
   - 需构建 per-layer agent、profiling pipeline、shadow testing infra
4. **目前聚焦于吞吐与延迟，未考虑能耗、成本等多目标权衡**

---

### **未来工作方向**
1. **扩展至更多系统组件**
   - 如分布式调度器、缓存系统、data loader
2. **引入多目标优化**
   - 在 throughput、latency、cost、energy 之间做 Pareto 探索
3. **增强 agent 的推理与规划能力**
   - 支持跨层联合优化决策（如同时改 kernel + 调 batch）
4. **开源 agent 框架与 benchmark**
   - 推动社区共建通用 model serving optimization infra

---

> **总结一句话**：  
> FlashVector 成功将 LLM agent 从 **GPU kernel 级优化** 扩展到 **全栈、多语言、持续化的 model serving 优化**，实现了高达 **2× throughput 提升**，并展示了 **agentic systems 作为连接算法原型（reference code）与高性能生产系统（performance-critical systems）的桥梁潜力**。

</details>

---

### 15. [Channel-Informed Neural Network for Physical Layer Key Generation](https://arxiv.org/abs/2609.16341)

**Authors**: Jose Angel Sanchez Viloria, George Sklivanitis, Dimitris Pados, Elizabeth Serena Bentley  
**Category**: cs.LG  
**Published**: 2026-09-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.16341v1  

#### Abstract
Physical-layer key generation (PKG) enables wireless devices to establish shared keys from reciprocal channel observations without directly exchanging the key. This capability is attractive for edge networks, where distributed and resource-constrained devices may require lightweight key establishmen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Channel-Informed Neural Network for Physical Layer Key Generation**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
物理层密钥生成（Physical-layer Key Generation, PKG）旨在利用无线信道的互易性、时变性和空间去相关性，在无直接密钥交换的情况下让合法通信双方（Alice 和 Bob）独立生成共享密钥。然而，传统方法面临以下挑战：
- 依赖预计算的信道状态信息（CSI），忽略了原始信号中的丰富特征；
- 多数深度学习方法仅优化嵌入相似性，缺乏对底层多径信道物理结构的显式建模；
- 密钥多样性（key diversity）与协商可靠性（reconciliation reliability）之间存在权衡。

### **提出了什么新方法或新思路**
本文提出了一种**信道感知神经网络（Channel-Informed Recurrent Neural Network, CI-RNN）**框架，其核心创新包括：

- **多任务学习架构**：设计了一个 multi-task RNN，同时学习两个目标：
  - 主任务：从原始 IQ spectrogram 中提取互易性保持的二进制密钥特征；
  - 辅助任务：估计对应的多径信道冲激响应（channel impulse response），作为监督信号。
- **信道感知的深度度量学习（Channel-Informed Deep Metric Learning）**：
  - 结合 quadruplet metric learning 来拉近 Alice-Bob 对的表示，推远 Eve 观察；
  - 引入基于信道估计的损失项（如 NMSE），将学习过程“锚定”在真实的传播物理上，起到类似 physics-informed regularization 的作用。
- **Ray-Traced 数据增强**：
  - 利用 Sionna-RT 射线追踪工具构建数字孪生环境，生成多样化的虚拟信道场景用于训练数据扩充。

### **相比现有方法的优势**
- **更强的物理一致性**：通过辅助信道估计任务，确保学到的密钥特征与真实信道结构一致，提升鲁棒性；
- **更高的密钥多样性**：借助射线追踪数据增强，显著提高唯一密钥率（unique-key rate）；
- **端到端处理能力**：直接从原始 IQ 数据出发，无需手工提取 CSI 或 RSS 特征，更适合边缘设备部署；
- **安全性增强**：合法用户间密钥一致性高，而窃听者（Eve）难以重建相同密钥。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **真实测量数据**：基于 PAWR 的 **POWDER 软件定义无线电（SDR）测试平台**采集：
  - **室内场景**：使用 NI USRP B210/X310 设备，在实验室环境中收集 10 种 Alice-Bob-Eve 拓扑配置下的信道探测数据；
  - **室外场景**：在 USTAR、EBC 和 Guest House 部署 SDR，采集两种户外拓扑的数据。
- **合成数据**：使用 **Sionna-RT ray tracer** 构建数字孪生环境，模拟户外传播条件，生成 500 个虚拟节点位置组合，并结合不同 SNR（5–30 dB）生成 7500 组 spectrogram-quadruplet 数据用于训练增强。

### **实验设置**
- **输入信号**：BPSK 调制的 GLFSR 探测序列（长度 $N_p = 256$）；
- **采样参数**：载频 3.5 GHz，采样率 1 MHz，IQ 数据经 RMS 归一化后转换为 STFT spectrogram（尺寸 $16 \times 32 \times 2$）；
- **模型结构**：
  - 双向 GRU 编码器（每方向 384 单元）；
  - 输出：128-bit 二进制密钥 + 16-tap 复数信道估计；
- **训练目标**：
  $$
  \mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Data}} + \lambda_{\text{ch}} \mathcal{L}_{\text{Channel}}, \quad \lambda_{\text{ch}} = 0.3
  $$
  其中 $\mathcal{L}_{\text{Data}}$ 为 quadruplet loss，$\mathcal{L}_{\text{Channel}}$ 为归一化均方误差（NMSE）。

### **评估指标**
- **Bit Disagreement Ratio (BDR)**：衡量任意两方生成密钥之间的比特差异比例；
- **Unique-Key Rate**：不同密钥数量 / 总密钥数量，反映密钥多样性；
- **Reconciliation Rate (RR)**：使用 Reed-Solomon (RS) 码进行纠错后的成功匹配率（Alice vs Bob）；
- **NIST 随机性测试**：验证最终密钥的统计随机性；
- **对比基线方法**：
  - RNN [24]：相同网络结构但仅使用 tone-based 探针且无信道监督；
  - CI-RNN（无数据增强）
  - CI-RNN + Sionna-RT 数据增强

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 指标 | 方法 | 室内 | 户外1 | 户外2 |
|------|------|-------|--------|--------|
| **Unique-Key Rate** | RNN [24] | 0.21 | 0.29 | 0.33 |
| | CI-RNN | 0.37 | 0.63 | 0.57 |
| | **CI-RNN + Sionna-RT** | **0.94** | **0.99** | **0.99** |
| **Alice-Bob BDR 分布** | — | 显著低于 Alice-Eve / Bob-Eve | 同左 | 同左 |

- 图 4 显示，在所有场景下，**Alice-Bob 的 BDR 远低于 Eve 相关对**，表明所提方法有效区分合法与非法观察。
- **Sionna-RT 数据增强极大提升了密钥多样性**，unique-key rate 提升超过 3 倍。

### **与基线方法的对比结果**

#### **Reconciliation Rates（见 Table I）**
- 在较高码率 RS(24,16) 下：
  - RNN [24] 的 Alice-Bob 协商成功率更高（如室内达 67%），但 CI-RNN 因密钥更敏感导致初始不一致略高；
- 在低码率 RS(32,16) 下：
  - CI-RNN 表现改善，但仍低于 RNN [24]，说明**高多样性带来更高残余错误，影响协商成功率**；
  - Eve 的协商成功率始终极低（< 0.5），体现良好安全隔离。

#### **NIST 随机性测试**
- 所有成功协商的 CI-RNN 密钥 **100% 通过 9 项 NIST 测试**（包括 Approximate Entropy, Runs, Serial 等）；
- 而 RNN [24] 方法未能通过 **Random Excursion Variant** 测试，表明其生成密钥统计特性较差。

#### **推理效率**
- 平均推理时间：**2.56 ms / sample**（NVIDIA H200 GPU）；
- 模型大小：约 **5.02 MiB**，含 1.29M 参数，适合边缘部署。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **信道感知训练能有效提升密钥质量**：通过引入信道估计作为辅助任务，使学到的密钥特征更具物理意义，增强了对环境变化的适应性。
2. ✅ **Ray-Traced 数据增强显著提升密钥多样性**：unique-key rate 达到 0.94~0.99，接近理想水平，大幅降低重复密钥风险。
3. ⚠️ **存在“多样性-可靠性”权衡（diversity-reliability trade-off）**：
   - 更敏感的特征提取提高了熵和唯一性；
   - 但也增加了 Alice-Bob 间的残余比特差异，降低了 Reed-Solomon 协商成功率。
4. ✅ **安全性强**：Eve 的 BDR 始终很高，且无法有效利用公开的 RS 校验信息完成协商；
5. ✅ **满足密码学要求**：最终密钥通过全部 NIST 随机性测试，具备用于加密的基础条件。

### **方法的局限性**
- 当前框架依赖于精确的时间同步和已知探测序列；
- 使用固定的 RS 编码方案限制了自适应协商能力；
- 射线追踪依赖准确的环境建模，实际复杂城市环境中可能受限；
- 实验规模较小（仅 2–3 个 SDR 节点），未验证大规模网络适用性。

### **未来工作方向**
- 设计 **reconciliation-aware 的联合训练目标**，在保持多样性的同时优化协商成功率；
- 探索 **自适应量化机制** 和 **神经增强型 reconciliation（Neural-enhanced reconciliation）**；
- 引入 **mobility modeling** 和 **interference robustness** 以支持动态场景；
- 扩展至 **active adversary models**（如恶意中继、RIS 攻击）；
- 探索与其他 AI 技术融合，如 LLM-assisted probing 或联邦学习框架下的分布式 PKG。

--- 

> **总结一句话**：  
> 本文提出的 **Channel-Informed Neural Network** 成功将物理层密钥生成与无线信道物理深度融合，实现了高多样性、高安全性、符合随机性标准的端到端密钥生成，揭示了“**agreement-diversity trade-off**”这一关键设计维度，为面向边缘智能的轻量级安全通信提供了有力支撑。

</details>

---

### 16. [Recovering Physical Parameters from Fragmented Observations via Exact Distributed Spline Merging](https://arxiv.org/abs/2609.16579)

**Authors**: Naveen Mysore  
**Category**: cs.LG  
**Published**: 2026-09-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.16579v1  

#### Abstract
Scientific measurements are frequently distributed across locations, time periods, and institutions. Combining such fragments into a continuous, differentiable field enables recovering governing physical parameters from its derivatives. This paper makes two contributions toward that goal. First, the...

---

### 17. [Skill-based Agentic Evaluation for Real-time Data Science Tasks](https://arxiv.org/abs/2609.16487)

**Authors**: Aniruddha Tamhane, Raghavendra Addanki, Ayushi Aggarwal, Aditya Bansal, Rui Wang, Charles Menguy, Swati Jain  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.16487v1  

#### Abstract
We present a framework for evaluating data-science agents on live, continuously updated data using executable ground truth and format-agnostic factoid scoring. Consider this example query: "what were last week's audience sizes"---the reference answer changes as the underlying data changes, so static...

---

### 18. [Layers, Sinks, and Scaling: Adaptive Evidence Selection for Multimodal Large Language Models](https://arxiv.org/abs/2609.16795)

**Authors**: Zhenbin Wang, Lei Zhang, Lituan Wang, Wei Huang, Yan Wang, Zhenwei Zhang  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.16795v1  

#### Abstract
Multimodal large language models (MLLMs) can answer knowledge-intensive visual questions by combining visual evidence from images with facts retrieved from external sources. However, MLLMs may overlook relevant evidence in both modalities, attending weakly to the textual sentences or visual regions ...

---

### 19. [ORDER: Task-Conditioned Routing for Retrieval-Augmented Generation](https://arxiv.org/abs/2609.17012)

**Authors**: Aur\'elien Pellet (LRE), Julien Perez, Marie Puren  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.17012v1  

#### Abstract
Retrieval-Augmented Generation (RAG) pipelines typically rely on a fixed indexing and retrieval configuration determined at preprocessing time. This one-size-fits-all design is ill-suited to domain-expert settings, where heterogeneous queries require different chunking granularities, metadata constr...

---

### 20. [Never Stop Thinking: Continuous-Time Language Agents](https://arxiv.org/abs/2609.17416)

**Authors**: Bojie Li, Noah Shi  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.17416v1  

#### Abstract
Voice agents built on LLMs follow a rigid listen-think-speak loop that inserts seconds of dead air before every reply. We show that continuous-time cognition (thinking while listening and thinking while speaking) emerges from an unmodified text model under a lightweight interrupt-and-resume orchestr...

---

### 21. [Optimal Model Activation Policies for Inference Networks of Large Language Models](https://arxiv.org/abs/2609.15992)

**Authors**: Foivos Charalampakos, Md Ibrahim Ibne Alam, Iordanis Koutsopoulos, Koushik Kar  
**Category**: cs.CL  
**Published**: 2026-09-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.15992v1  

#### Abstract
Recent advances in large language models (LLMs) have rendered them necessary for Natural Language Processing (NLP) tasks, and their high inference cost motivates the study of cost-performance trade-offs. In practice, several expert LLMs are used in synergy for inference, either in an ensemble mode o...

---

### 22. [Self-reported archetypes and behavioral failures in Large Language Models](https://arxiv.org/abs/2609.15998)

**Authors**: Tabia Tanzin Prama, Calla Glavin Beauregard, Christopher M. Danforth, Peter Sheridan Dodds  
**Category**: cs.CL  
**Published**: 2026-09-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.15998v1  

#### Abstract
Every large language model (LLM) has behavioral traits and moral preferences that comprise its character. Whether by design or as an emergent property of training, these systems exhibit persistent dispositions that shape how they interact, comply, resist, and err, yet the structure of LLM character ...

---

### 23. [Register Tokens for Bounded-State Reasoning in Diffusion Language Models](https://arxiv.org/abs/2609.16372)

**Authors**: Albert Ge, Chandan Singh, Yufan Zhuang, Xiaodong Liu, Jianfeng Gao, Frederic Sala  
**Category**: cs.CL  
**Published**: 2026-09-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.16372v1  

#### Abstract
Masked diffusion language models (dLLMs) generate text by iteratively denoising masked tokens with bidirectional attention. Extending reasoning across generation chunks normally requires keeping earlier generated text in context. We ask whether a dLLM can instead continue reasoning after that text i...

---

### 24. [Scaling Laws for Physics-Aware ACOPF Surrogate Learning](https://arxiv.org/abs/2609.16282)

**Authors**: Yijiang Li, Emon Dey, Stefano Fenu, Massimiliano Lupo Pasini, Teja Kuruganti, Kibaek Kim  
**Category**: cs.LG  
**Published**: 2026-09-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.16282v1  

#### Abstract
Learning-based surrogates for AC optimal power flow (ACOPF) promise large speedups over classical solvers, but their operational value depends on physical feasibility as much as predictive accuracy. Physics-aware objectives such as the augmented Lagrangian (AL) improve constraint satisfaction at add...

---

### 25. [Agentic Search Spaces for Tabular Machine Learning](https://arxiv.org/abs/2609.16309)

**Authors**: Renat Sergazinov, Artem Chistyakov, Sergey Pankevich, Artem Babenko  
**Category**: cs.LG  
**Published**: 2026-09-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.16309v1  

#### Abstract
Despite the rapid progress of LLM-based agents for planning, code generation, and debugging, their practical value for tabular machine learning remains underexplored. In this paper, we investigate a concrete use case: whether state-of-the-art agentic AI systems can design extended HPO search spaces ...

---

### 26. [A Framework for Generating Valid Context-Specific Benchmarks through Expert Guidance](https://arxiv.org/abs/2609.16592)

**Authors**: Kimberly Le Truong, Nari Johnson, Anna Kawakami, Hoda Heidari  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.16592v1  

#### Abstract
This paper presents an end-to-end approach for generating context-specific large language model (LLM) benchmark datasets by combining expert input with synthetic data generation. Existing benchmark construction methods often trade off validity and scalability: datasets designed with domain experts c...

---

### 27. [SKIP: a Self-knowledge-guided Step-wise Preference Learning Framework for Concise Reasoning](https://arxiv.org/abs/2609.17019)

**Authors**: Qinhong Lin, Yuhao Zhang, Yinglun Feng, Zhongliang Yang, Linna Zhou  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.17019v1  

#### Abstract
While Chain-of-Thought (CoT) reasoning has been proven to be effective, it often leads to overthinking, resulting in computational overhead, inference latency, and even degraded performance in large language models (LLMs). Existing concise reasoning frameworks significantly compromise accuracy while...

---

### 28. [JustFit: 200K-Token LLM Serving on a 24 GiB Laptop with Just-in-Time State Management](https://arxiv.org/abs/2609.17475)

**Authors**: Yuhua Chen  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.17475v1  

#### Abstract
Capable open-weight models make local coding and reasoning attractive, but their context and execution state strain laptop memory. We present JustFit, an MLX-based inference runtime that combines KVExec for compressed KV execution, PhaseSwap for component residency, and StateTrans for state-preservi...

---

### 29. [LimiX-2: A Contextual Mechanism Network Towards General Structured-Data Intelligence](https://arxiv.org/abs/2609.17488)

**Authors**: Xingxuan Zhang, Gang Ren, Hao Yuan, Hao Zou, Hongze Tan, Hui Wang, Jianhao Song, Jiansheng Li, Jiayao Zhang, Jinghan Zhang, Kaifang Li, Lang Mo, Li Mao, Mingchao Hao, Nuo Xu, Rui Ding, Ruiji Zhang, Shuyang Li, Siyu Mei, Tianyang Zhang, Weiyang Mu, Yancheng Dong, Yongxian Wei, Yuan Xue, Yuanrui Wang, Yue He, Zijia Yang, Ziyun Li, Dongzhe Li, Fuqiang Wang, Jiandong Liu, Jiawei Chen, Jiaxin Du, Kaijie Cheng, Kehan Li, Lei Sun, Linjun Zhou, Ningbo Dai, Qi Wang, Renzhe Xu, Shaoxing Du, Shumeng Yang, Wang Lu, Wenjing Chu, Xiannan Huang, Xiaoyu Lin, Xing Ai, Xinyan Han, Xuanyue Li, Xuanyue Su, Xukun Zhang, Yan Lu, Yaxin Zhang, Yi Qin, Yifei Huang, Yihan Xu, Yongle Lv, Yuanyuan Jiang, Yushan Han, Peng Cui  
**Category**: cs.AI  
**Published**: 2026-09-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.17488v1  

#### Abstract
We introduce LimiX-2, a new model in the LimiX family, developed through model and data scaling guided by our previously established scaling laws. LimiX-2 adopts the Contextual Mechanism Networks (CMNs) paradigm and is pretrained with Context-Conditional Masked Modeling (CCMM). CMNs shifts the organ...

---

### 30. [ReMova: Fine-tuning LLMs for English to Belarusian translation](https://arxiv.org/abs/2609.16427)

**Authors**: Mikita Pilinka, Aliaksandr Kliuje\u{u}, David Samuel, Yves Scherrer  
**Category**: cs.CL  
**Published**: 2026-09-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2609.16427v1  

#### Abstract
This paper presents a Belarusian-specific data-cleaning pipeline and fine-tuning for English-Belarusian machine translation. Our cleaning pipeline distinguishes itself from others by employing a correction tool that addresses the issue of the two orthographies of the Belarusian language, noise in th...

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

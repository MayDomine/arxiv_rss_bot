# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-10 06:13:34 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization](https://arxiv.org/abs/2603.08185)

**Authors**: Yeonsik Park, Hyeonseong Kim, Seungkyu Choi  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.08185v1  

#### Abstract
Post-training quantization (PTQ) has emerged as a prevailing technique for deploying large language models (LLMs) efficiently in terms of both memory and computation, across edge devices and server platforms. Existing PTQ methods primarily aim to reduce precision in weights and activations by mitiga...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLMs）在边缘设备和服务器平台上的高效部署面临巨大的内存和计算开销挑战。**Post-Training Quantization (PTQ)** 是一种主流的压缩技术，旨在降低权重和激活值的精度以提升效率。然而，在 **W4A4（4-bit weights, 4-bit activations）** 设置下，现有方法面临严重准确率下降的问题，主要原因如下：
- **Channel-wise outlier activations** 导致量化误差显著。
- 现有的 **low-rank error reconstruction** 方法依赖两个低秩因子（如 $ L_1L_2 $），在推理时需要**顺序执行矩阵乘法**，引入中间值的额外量化步骤，破坏了低精度计算路径的完整性，限制了效率。

### **提出了什么新方法或新思路**
本文提出 **SERQ (Saliency-Aware Error Reconstruction with a single low-rank matrix)**，一种用于低比特 LLM 推理的新型量化方法，其核心创新在于：
1. **单低秩补偿矩阵（Single Low-Rank Compensation Matrix）**：
   - 与传统 LoRA 或 SVD 方法使用两个低秩因子不同，SERQ 将量化误差校正统一到一个单一的低秩矩阵 $ R \in \mathbb{R}^{r \times d} $ 中。
   - 这避免了顺序乘法和中间量化，实现了端到端的 **4-bit 矩阵乘法**（如 INT4, MXFP4 GEMM）。
2. **三阶段联合优化流程**：
   - **静态激活展平 (Static Activation Flattening)**：采用类似 SmoothQuant 的通道级缩放，将激活异常值的影响转移到权重上，并通过离线合并(scale folding)消除运行时代价。
   - **显著性感知误差重建 (Saliency-Aware Error Reconstruction)**：识别出因激活缩放而变得“显著”（即量化误差大）的权重行，仅对这些行进行误差重建，提高了低秩分解的效率。
   - **离线权重置换 (Offline Weight Permutation)**：将显著的权重行和对应的激活通道预先重排至矩阵的前部，使得残差路径只需处理小规模的 $ X_s \in \mathbb{R}^{s\_len \times r} $ 和 $ R \in \mathbb{R}^{r \times d} $，且所有重排操作均在离线完成，不增加推理延迟。

### **相比现有方法的优势**
- **更高的精度**：在 W4A4 设置下，SERQ 的性能显著优于现有的 low-rank 方法（如 L2QER）和旋转类方法（如 SpinQuant, QuaRot）。
- **更低的延迟**：由于消除了中间量化和顺序乘法，SERQ 的推理延迟远低于 L2QER（最高可达 4.5× 加速），甚至略低于旋转方法。
- **更简单的部署**：无需在线变换层或复杂的训练过程，所有预处理（展平、置换）均可离线完成。
- **真正的低精度路径**：首次实现了基于 low-rank error reconstruction 的端到端 4-bit 矩阵乘法。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **校准数据集 (Calibration Set)**：用于确定显著性行，由 WikiText-2 数据集中的 128 个随机样本构成。
- **评估任务与数据集**：
  - **零样本常识推理 (Zero-shot Commonsense Reasoning)**：PIQA, SIQA, ARC-Easy/Challenge, HellaSwag, Winogrande, BoolQ, OpenBookQA。
  - **困惑度 (Perplexity)**：WikiText2 test set。
  - **综合基准 (MMLU)**：涵盖多个学科领域的多项选择题。
  - **生成任务 (Generation Tasks)**：GSM8K (数学推理), LongBench (长上下文理解)。

### **实验设置和评估指标**
- **模型**：在 LLaMA-2 (7B, 13B, 70B), LLaMA-3 (8B, 1B, 3B), Qwen-2.5 (3B) 等多种主流 LLM 上进行验证。
- **量化配置**：
  - **W4A8**：4-bit 权重，8-bit 激活。
  - **W4A4**：4-bit 权重，4-bit 激活（主要挑战场景）。
  - 支持 **INT (RTN/GPTQ)** 和 **MXFP4** 两种量化格式。
- **评估指标**：
  - **Perplexity (PPL)**：越低越好。
  - **平均零样本准确率 (0-shot Accuracy)** 和 **MMLU 准确率**：越高越好。
  - **推理延迟**：Time to First Token (TTFT), Time per Output Token (TPOT)。
  - **峰值内存消耗 (Peak Memory Consumption)**。
  - **有效位宽 (Effective Bit-width)**：衡量实际存储开销。

### **基线方法对比**
- **Low-Rank Matrix Decomposition Methods**：
  - `LLM.int4()`：4-bit 版本的 LLM.int8()，作为基础 baseline。
  - `L2QER`：最先进的基于 SVD 的 low-rank error reconstruction 方法。
- **Distribution Flattening Methods (Rotation-based)**：
  - `SmoothQuant`：经典的预量化缩放方法。
  - `QuaRot`：使用随机 Hadamard 变换。
  - `SpinQuant`：学习最优旋转矩阵。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据与对比结果**
#### **(1) 与 Low-Rank 方法对比 (Table 1 & 12)**
- 在 **W4A8** 设置下，SERQ 在所有模型上均优于 L2QER。
  - 例如，在 LLaMA-2-7B 上，SERQ (GPTQ) 的 PPL 为 **5.59**，而 L2QER 为 **5.83**。
- 在极具挑战的 **W4A4** 设置下，优势更加明显：
  - L2QER 性能急剧下降（LLaMA-3 模型上 PPL 超过 10），而 SERQ 保持稳定。
  - 在 LLaMA-2-7B 上，SERQ (GPTQ) 的 PPL 为 **5.97**，0-shot 准确率为 **61.87%**，MMLU 为 **37.03%**，全面超越 L2QER (PPL: 7.37, 0-shot: 57.67%)。

#### **(2) 与 Rotation-based 方法对比 (Table 2 & 13)**
- SERQ 在精度上**全面超越**了当前最先进的旋转方法：
  - 在 LLaMA-2-7B 上，SERQ (GPTQ) 的 MMLU (**37.03%**) 显著高于 SpinQuant (**34.8%**) 和 QuaRot (**33.58%**)。
  - 在 LLaMA-3-8B 上，SERQ 的 MMLU (**53.8%**) 远超 SpinQuant (**49.93%**)。
- **延迟开销更低**：SERQ 的每层延迟开销为 **18.7%**，低于 SpinQuant 和 QuaRot 的 **19.8%**。

#### **(3) GPU 性能分析 (Figure 3 & Table 3)**
- **延迟优势**：在大尺寸矩阵上，SERQ 的延迟远低于 L2QER（最高达 4.5×），因为 L2QER 需要两次低秩乘法。
- **端到端加速**：在 LLaMA-3-8B 上，SERQ-MXFP4 实现了超过 **2×** 的 TTFT 加速，峰值内存减少高达 **2.48×**。

### **消融实验结果**
#### **(1) 秩大小 (Rank Size) 影响 (Table 4)**
- 性能随秩大小增加而单调提升，但在 `r=128` 后趋于饱和。
- 即使是较小的秩（如 `r=16`），SERQ 也能取得有竞争力的结果，证明了其有效性。

#### **(2) 静态激活展平 (SAF) 的作用 (Table 7)**
- 对于小型模型（如 Qwen-2.5-3B），移除 SAF 会导致 W4A4 下 PPL 从 **9.57** 恶化到 **10.83**，表明 SAF 对稳定量化至关重要。
- 对于较大的模型，影响相对较小，但仍有益处。

#### **(3) 校准数据敏感性 (Table 5)**
- SERQ 对校准数据集的选择（WikiText vs. The Pile）和样本数量（32 到 512）表现出很强的鲁棒性，困惑度变化很小。

---

## **4. 关键结论和发现**

### **主要发现**
1. **单低秩矩阵设计是高效的**：通过将误差重建集中到一个单一的低秩矩阵中，SERQ 成功地绕过了传统 low-rank 方法的瓶颈，实现了真正的端到端低精度计算。
2. **显著性感知是关键**：直接基于激活缩放后的权重行显著性来选择重建目标，比全局 SVD 更加精准和高效。
3. **离线优化是可行的**：通过静态展平和离线置换，可以将所有复杂操作前置，从而在不牺牲精度的前提下最小化推理开销。
4. **SERQ 是当前 W4A4 量化的新标杆**：它在精度、延迟和部署简易性之间取得了最佳平衡，是目前最有效的 W4A4 量化方案之一。

### **方法的局限性**
- **依赖校准集**：虽然鲁棒，但仍需一个小的校准集来确定显著性行。
- **秩大小选择**：需要手动设定秩 `r`，尽管存在饱和效应，但最优值可能因模型而异。
- **理论解释**：虽然实验证明了其有效性，但对于为何单矩阵能如此有效地联合补偿权重和激活误差，可能还需要更深入的理论分析。

### **未来工作方向**
- **自动化秩选择**：研究如何根据模型结构或数据自适应地确定最优秩大小。
- **扩展到其他模态**：将 SERQ 的思想应用于视觉或多模态大模型的量化。
- **结合微调**：探索将 SERQ 与轻量级微调（如 LoRA）结合，进一步提升超低比特（如 W3A3）下的性能。
- **硬件协同设计**：针对 SERQ 的计算模式（主路径 + 小规模残差路径）设计专用的硬件加速器。

</details>

---

### 2. [Agentic AI-Driven UAV Network Deployment: A LLM-Enhanced Exact Potential Game Approach](https://arxiv.org/abs/2603.07456)

**Authors**: Xin Tang, Qian Chen, Binhan Liao, Yaqi Zhang, Jianxin Chen, Changyuan Zhao, Junchuan Fan, Junxi Tian, Xiaohuan Li  
**Category**: cs.DC  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.07456v1  

#### Abstract
Unmanned Aerial Vehicular Networks (UAVNs) are envisioned to provide flexible connectivity, wide-area coverage, and low-latency services in dynamic environments. From an agentic artificial intelligence (Agentic AI) perspective, UAVNs naturally operate as multi-agent systems, where autonomous UAVs ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Agentic AI-Driven UAV Network Deployment: A LLM-Enhanced Exact Potential Game Approach**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **Unmanned Aerial Vehicular Networks (UAVNs)** 中的网络部署优化难题，解决了以下关键挑战：
- **Mixed-Integer Nonconvex Problem (MINLP)**：UAVN 部署涉及离散变量（如链路连接、用户关联）和连续变量（如位置、功率），导致联合优化困难。
- **动态环境下的可扩展性和一致性**：传统集中式方法存在单点故障风险，而启发式或强化学习方法缺乏收敛保证且训练成本高。
- **手动参数调优依赖性强**：现有博弈论方法在不同场景下需反复调整效用函数权重，适应性差。

---

### 🚀 提出的新方法与创新思路

#### （1）**双空间尺度的 EPG 优化框架**
将复杂的 MINLP 问题分解为两个子问题，在不同空间尺度上分别求解：
- **大尺度（Large Spatial Scale）**：采用基于 **Log-Linear Learning 的 Exact Potential Game (L3-EPG)** 优化 **离散链路拓扑**，实现稀疏化连接，减少冗余链路与干扰，同时保持网络连通性。
- **小尺度（Small Spatial Scale）**：提出 **Approximate Gradient-based EPG (AG-EPG)** 算法，联合优化 **UAV 坐标、传输功率和 GU 用户关联**，提升吞吐量并降低延迟与能耗。

> ⚙️ 优势：分布式决策，仅需局部信息交互，无需中央控制器；确保纳什均衡存在，并具备全局收敛能力。

#### （2）**引入 LLM 作为知识驱动的效用增强器**
- 构建了一个面向 UAVN 拓扑优化的 **领域知识库**，整合无线通信理论、博弈论基础及优化案例。
- 结合 **Retrieval-Augmented Generation (RAG)** 框架，利用 LLM 自动从知识库中检索相关信息，生成符合当前网络特征的 **效用函数及其权重系数**。
- 支持“意图驱动”配置：网络管理者输入场景需求（如节点数、分布、目标等），LLM 自动生成适配的优化参数。

> 💡 创新点：首次将 LLM 引入 EPG 框架用于自动权重生成，显著降低人工调参负担，提升算法跨场景泛化能力。

---

### 🔍 相比现有方法的优势
| 维度 | 本方法优势 |
|------|-----------|
| **建模方式** | 分布式、去中心化，避免单点失效，支持动态拓扑演化 |
| **收敛性保障** | 基于 EPG 设计，确保纯策略 Nash Equilibrium 存在且可收敛 |
| **效率与可扩展性** | 双尺度解耦设计降低复杂度，适合大规模 UAVN |
| **自适应能力** | LLM + RAG 实现场景感知的自动参数生成，无需专家干预 |

---

## 2. 核心实验方法和设置

### 🧪 实验设置
- **仿真场景**：低空通信网络，覆盖区域为 $10\,\text{km} \times 10\,\text{km}$ 平方区域。
- **UAV 数量**：$N = 10$
- **Ground User (GU) 数量**：$M = 20$，随机均匀分布
- **飞行高度范围**：$[100\,\text{m}, 300\,\text{m}]$
- **最大通信半径**：$R_c = 4000\,\text{m}$
- **系统带宽**：$B = 2\,\text{MHz}$
- **信道模型**：A2A 使用 LoS 路损模型；A2G 使用 ITU-R P.1410 推荐的仰角相关 LoS 概率模型
- **噪声谱密度**：$-174\,\text{dBm/Hz}$
- **仿真时间**：划分为多个时隙 $T = \{1,2,...,T\}$，每轮迭代更新拓扑与资源

---

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Energy Consumption** | 所有 UAV 的总能耗（含通信能 + 飞行能） |
| **End-to-End Latency** | 包括传输延迟和传播延迟的加权平均 |
| **Network Throughput** | A2A 与 A2G 总吞吐量之和 |
| **Convergence Stability** | 局部效用变化 vs. 势函数变化的相关性（验证 EPG 一致性） |
| **Topology Sparsity** | 连接链路数量随迭代的变化趋势 |

---

### 🆚 对比的基线方法
1. **BRD-EPG**：基于最佳响应动态（Best Response Dynamics）的标准 EPG 方法
2. **BRD-NCG**：非合作博弈下的 BRD 方法，无全局势函数引导
3. **Evolutionary Game (ETG)**：基于复制者动力学的进化博弈
4. **Genetic Algorithm (GA)**：经典启发式全局搜索算法

---

## 3. 主要实验结果和性能指标

### 📈 关键性能表现（见 Fig. 8）
| 指标 | 表现 |
|------|------|
| **Energy Consumption** | AG-EPG 显著低于所有基线方法，尤其在 UAV 数量增加时仍保持最低增长速率 |
| **Latency** | 总体延迟最低，优于第二名约 12%~18%，且随规模扩大更稳定 |
| **Throughput** | 吞吐量最高，相比其他方法平均提升 **8.4%**，最高可达 10.2% |

> ✅ **原因分析**：
> - L3-EPG 成功剪枝冗余链路 → 减少干扰与能耗
> - AG-EPG 通过 3D 坐标微调改善 LoS 条件 → 提升信道质量
> - 功率与用户负载联合优化 → 提高频谱效率

---

### 🔬 消融实验与关键验证（Fig. 5 & Fig. 6）

#### （1）**EPG 收敛性验证**
- Fig. 5(a)(b) 显示：局部效用变化 $\Delta U_i$ 与势函数变化 $\Delta \Phi$ 高度一致，相关系数 $R^2 > 0.999$，证明满足 EPG 条件。
- Fig. 5(c)(d) 显示：各 UAV 的策略在迭代中逐步收敛至稳定状态，未出现震荡。

#### （2）**拓扑演化过程可视化（Fig. 6）**
- 初始全连接图 → 经 L3-EPG 迭代后形成稀疏但仍连通的骨干网
- 当 GU 移动或障碍物变化时，UAV 自适应调整高度与位置，绕开遮挡（Fig. 7）

#### （3）**LLM 知识检索精度测试（Fig. 4）**
- 测试不同 **knowledge block size** 和 **top-K retrieval 数量** 对生成质量的影响
- 最优设置：**block size = 4**, **K = 3**，此时检索精度达到峰值
- 过大的块或过多检索项反而引入噪声，降低生成准确性

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **双尺度 EPG 框架有效解耦了混合整数优化问题**，实现了高效、稳定的分布式部署。
2. **L3-EPG 能在保证连通性的前提下最小化链路数量**，构建轻量化网络骨架。
3. **AG-EPG 在连续空间中实现多维联合优化**，显著提升系统级性能（吞吐↑、能耗↓、延迟↓）。
4. **LLM + RAG 可实现“意图到参数”的自动化映射**，大幅增强算法在异构场景中的适应性与实用性。

---

### ⚠️ 方法局限性
| 局限 | 说明 |
|------|------|
| **实时性限制** | 尽管是分布式，但每次迭代仍需同步协调，对高移动性场景可能不适用 |
| **LLM 推理延迟** | 在线调用 LLM 生成权重会引入额外延迟，不适合毫秒级响应任务 |
| **依赖高质量知识库** | 若知识库覆盖不足或存在错误，可能导致生成参数偏差 |
| **未考虑安全与隐私攻击** | 如恶意 UAV 抵抗博弈、欺骗行为等未建模 |

---

### 🔮 未来工作方向
1. **轻量化 LLM 部署**：探索本地化小型 LLM 或 MoE 架构以降低推理开销
2. **在线增量学习机制**：让 LLM 能够持续吸收新场景经验，动态更新知识库
3. **融合联邦学习框架**：实现多 UAV 协同学习而不共享原始数据，保护隐私
4. **扩展至 Space-Air-Ground Integrated Network (SAGIN)**：将该框架推广至卫星-高空平台-地面协同网络
5. **硬件原型验证**：在真实无人机平台上部署算法，进行实地测试

---

## 总结一句话
> 本文提出了一种 **Agentic AI 驱动的 LLM 增强型双尺度 EPG 框架**，成功解决了 UAVN 部署中的混合整数非凸优化难题，在保证收敛性的同时实现了高性能、低能耗、自适应的智能组网，为未来自主低空网络提供了新的范式。

</details>

---

### 3. [wDPO: Winsorized Direct Preference Optimization for Robust LLM Alignment](https://arxiv.org/abs/2603.07211)

**Authors**: Jilong Liu, Yonghui Yang, Pengyang Shao, Haokai Ma, Wei Qin, Richang Hong  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.07211v1  

#### Abstract
Direct Preference Optimization (DPO) aligns large language models by optimizing pairwise preferences and has shown remarkable effectiveness as a simple and scalable alternative to RLHF. However, in practice, preference data are often noisy. Existing robust variants of DPO mainly rely on uniform obje...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：wDPO: Winsorized Direct Preference Optimization for Robust LLM Alignment**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **DPO 在噪声偏好数据下的鲁棒性不足**：尽管 Direct Preference Optimization (DPO) 因其简单高效成为主流的 LLM 对齐方法，但在实际应用中，人类标注的偏好数据常含有两类噪声：
  - **Hard Noise（硬噪声）**：偏好标签被错误反转（如本应偏好的响应被标记为拒绝）。
  - **Ambiguous Comparisons（模糊比较）**：两个响应质量相近，难以区分优劣，导致学习信号弱但损失值高。
- 现有鲁棒 DPO 方法（如 rDPO、cDPO、Dr.DPO）通常采用**全局统一修正策略**（如重加权、正则化），无法区分不同类型的噪声，导致优化不稳定或次优。

### **提出了什么新方法或新思路**
提出 **winsorized DPO (wDPO)**，一种基于**分层干预**（Hierarchical Intervention）的鲁棒偏好优化框架：
- **无需外部奖励模型**，仅利用 DPO 训练过程中已有的信号（如 log-ratio margin）进行噪声识别与干预。
- 引入两个阶段的针对性干预机制：
  1. **Stage I: Margin-aware Soft Label Correction（数据级干预）**
     - 针对 **hard noise**，通过计算 DPO margin 判断是否“强不一致”。
     - 对可能翻转的样本进行**稀疏软标签校正**（sparse flip-aware loss mixing），即在原始损失与反转损失之间插值。
     - 使用 `sparsemax` 和预算控制 `pf` 实现保守修正。
  2. **Stage II: Gradient-oriented Winsorization（梯度级干预）**
     - 针对 **ambiguous comparisons** 导致的极端高损失样本。
     - 采用**软 Winsorization** 技术，对 batch 中损失尾部（top-q%）进行动态裁剪，防止其主导梯度更新。
     - 裁剪强度由 batch 内 margin 分布自适应调整。

### **相比现有方法的优势**
- **更精细的噪声处理机制**：区别于“一刀切”的全局鲁棒方法，wDPO 根据噪声类型采取不同干预层级，提升鲁棒性和稳定性。
- **完全兼容标准 DPO 框架**：无需额外模型、采样或复杂流程，保持训练简洁性。
- **更强的泛化能力**：在分布外（OOD）安全基准上表现优异，说明其改善了底层优化动力学而非过拟合训练集。
- **参数敏感性低**：关键超参数（如 `pf`, `pw`, `q`）在较宽范围内均能稳定工作。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **主训练与验证集**：
  - **PKU-SafeRLHF-30K**：约 29.9K 条双响应偏好对，含安全性与帮助性标签，用于训练和 in-distribution 测试。
- **外部安全评估基准（Out-of-Distribution）**：
  - **Do-Not-Answer**：测试模型对有害请求的拒绝能力。
  - **HarmBench**：自动化红队测试与越狱鲁棒性。
  - **HH-RLHF**：来自 Anthropic 的有害提示子集。
  - **Salad Bench**：复杂语言结构下的对抗性安全测试。

### **实验设置和评估指标**

#### **Backbones（基础模型）**
- Pythia-2.8B
- Llama-3.2-3B
- Llama-3-8B
- Qwen2.5-7B  
（涵盖多个模型家族与规模）

#### **评估指标**
| 指标 | 含义 | 判定方式 |
|------|------|----------|
| **Win Rate (WR)** ↑ | 模型输出 vs 数据集中选中响应，哪个更安全？越高越好 | 使用 **GPT-4.1 mini API** 作为 judge 进行两两比较 |
| **Attack Success Rate (ASR)** ↓ | 攻击提示下生成不安全响应的比例，越低越好 | 使用两个自动判别器：<br>- **MD-Judge (MD)**<br>- **Nemotron Safety Guard (NV)** |

#### **训练细节**
- 所有方法共享相同设置：`β=0.1`, 学习率 `5e-7`
- wDPO 默认参数：`pf=0.3`, `q=0.7`, `p_max=0.5`, `α=0.3`（warm-up ratio）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表1：PKU-SafeRLHF 测试集（ID 性能）**
| Method | Pythia-2.8B WR | Llama-3-8B WR | Qwen2.5-7B WR |
|--------|----------------|---------------|----------------|
| DPO    | 44.23%         | 69.19%        | 75.81%         |
| rDPO   | 46.54%         | 82.13%        | 82.74%         |
| Dr.DPO | 57.41%         | 82.50%        | 88.93%         |
| **wDPO** | **59.92%**     | **84.01%**    | **90.23%**     |

> ✅ wDPO 在所有 backbone 上均取得最佳 WR 表现。

#### **表2：OOD 安全基准平均 ASR（Llama-3-8B & Qwen2.5-7B）**
| Method       | Llama-3-8B AVG ASR ↓ | Qwen2.5-7B AVG ASR ↓ |
|--------------|-----------------------|------------------------|
| DPO          | 22.85%                | 14.39%                 |
| rDPO         | 3.97%                 | 3.18%                  |
| Dr.DPO       | 2.09%                 | 1.05%                  |
| **wDPO**     | **1.95%**             | **0.64%**              |

> ✅ wDPO 在 OOD 场景下攻击成功率最低，表明其具有最强的泛化防御能力。

---

### **与基线方法的对比结果**
- **优于所有 DPO-family 基线**：
  - 相比 vanilla DPO：显著提升 WR 并大幅降低 ASR。
  - 相比 rDPO/cDPO/IPO：虽有一定改进，但仍是全局策略，效果有限。
  - 相比 Dr.DPO（当前最强鲁棒 DPO）：仍能进一步提升性能，尤其在高噪声场景下优势明显。
- **在标签翻转噪声下表现卓越**（RQ2）：
  - 在 **30% label-flip noise** 下，wDPO 的 WR 仍达 **55.14%**，而 Dr.DPO 降至 **52.93%**，DPO 仅为 **41.82%**。
  - 显示 wDPO 能更优雅地退化（graceful degradation），具备更强抗噪能力。

---

### **消融实验结果（Ablation Study）**

#### **图4：组件消融分析（Pythia-2.8B）**
| 设置 | WR (%) | AVG ASR (%) |
|------|--------|-------------|
| DPO（baseline） | 44.23 | 41.21 |
| wDPO (w/ Stage I only) | ~52 | ~30 |
| wDPO (w/ Stage II only) | ~56 | ~25 |
| **wDPO (full)** | **59.92** | **7.70** |

> 🔍 发现：
- **Stage II（梯度级 winsorization）贡献更大**：说明控制高损失尾部是稳定训练的关键。
- **Stage I + Stage II 协同增益明显**：两者互补，联合使用达到最优性能。
- 结果验证了“分层干预”设计的有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **DPO 的失败模式源于少数高损失样本主导梯度更新**：
   - 这些样本主要来自 hard noise 和 ambiguous comparisons。
   - 导致训练不稳定、收敛缓慢甚至性能下降。

2. **统一鲁棒策略不足以应对异质噪声**：
   - 现有方法将所有噪声视为同质不确定性，缺乏针对性。
   - wDPO 证明应根据不同噪声类型采取不同干预层级。

3. **分层干预显著提升鲁棒性与泛化能力**：
   - 数据级干预纠正方向错误。
   - 梯度级干预抑制无信息量的极端损失。
   - 二者结合实现更稳定、高效的偏好学习。

4. **无需额外监督即可实现强鲁棒性**：
   - wDPO 仅依赖 DPO 自身 margin 信号，不引入 reward model 或额外标注，实用性强。

---

### **方法的局限性**
- **依赖 batch 统计特性**：在极小 batch size 下可能影响阈值估计稳定性。
- **warm-up 阶段必要但需调参**：过早启用 Stage I 可能误纠正常样本；需合理设置 `α`。
- **未显式建模其他噪声类型**：如上下文混淆、多模态偏好等复杂情况尚未覆盖。

---

### **未来工作方向**
- 将 wDPO 思路扩展至其他偏好学习范式（如 KTO、RRHF）。
- 探索动态调整干预强度的元学习机制。
- 结合主动学习，在训练中识别并请求重新标注可疑样本。
- 应用于多轮对话、长文本生成等更复杂场景中的鲁棒对齐。

---

> 📌 **总结一句话**：  
> **wDPO 通过“分层 winsorization”实现了对异质偏好噪声的精准打击，在不增加复杂性的前提下，显著提升了 DPO 的鲁棒性、稳定性和泛化能力，是迈向可靠 LLM 对齐的重要一步。**

</details>

---

### 4. [EAGLE-Pangu: Accelerator-Safe Tree Speculative Decoding on Ascend NPUs](https://arxiv.org/abs/2603.08088)

**Authors**: Chang Han, Yijie Hu, Jingling Liu  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.08088v1  

#### Abstract
Autoregressive decoding remains a primary bottleneck in large language model (LLM) serving, motivating speculative decoding methods that reduce expensive teacher-model invocations by verifying multiple candidate tokens per step. Tree-structured speculation further increases parallelism, but is often...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# EAGLE-PANGU: Accelerator-Safe Tree Speculative Decoding on Ascend NPUs  
**——核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大语言模型（LLM）服务中，**autoregressive decoding** 是主要性能瓶颈，因其逐token生成导致高延迟、低吞吐。虽然 **speculative decoding** 可通过 draft model 提出候选 token 并由 teacher model 批量验证来加速，但现有的 **tree-structured speculative decoding** 在异构硬件后端（如 Ascend NPUs）上部署时面临严重挑战：

- 不同硬件对 **KV-cache layout**、**attention mask 形状** 和 **tensor indexing 语义** 支持不一致；
- 负索引、越界访问等操作在某些加速器上未定义或静默错误；
- fused attention kernels 对 mask 对齐和边界条件要求更严格；
- 导致实现难以移植、行为不可复现、甚至质量下降。

> 因此，本文旨在解决：**如何将 tree speculative decoding 安全、可复现地部署到 Pangu + Ascend NPU 这类生产级异构加速栈上？**

---

### 🚀 提出的新方法与创新点

EAGLE-PANGU 并非提出新的 decoding 算法，而是构建了一个 **accelerator-safe、portable、debuggable** 的系统级框架，基于 EAGLE-3 风格的 tree speculative decoding，在 Ascend 上实现了高效且正确的部署。其三大核心技术贡献如下：

#### （1）Branchable KV-Cache 抽象（显式的分支/提交缓存管理）
- 引入 `main_cache`（已接受前缀）与 `branch_caches`（每个候选分支独立副本）分离的设计；
- 支持两种 commit 模式：
  - **Length-based commit**：按长度截断更新；
  - **Path-index-based commit**：精确重排 KV-cache 以匹配接受路径；
- 实现 **prefix-sharing fast reorder**，仅重排新增部分，避免全量拷贝开销；
- 保证 cache isolation 与 commit equivalence，确保语义正确性。

#### （2）Accelerator-Safe Tree Tensorization（安全的树张量化）
- 消除负索引（如 parent=-1）带来的 undefined behavior；
- 引入 **dummy root row（index=0）** 替代 sentinel 值，使所有 gather indices 均为合法正整数；
- 构建 **ancestor table A ∈ ℤ^(D_max+1)×(M+1)**，预计算每层祖先，支持 safe gather；
- 添加 **structural invariants check**（范围、无环性、有效性闭包），用于运行前校验，提升 debuggability。

#### （3）Fused-Kernel-Compatible Tree-Masked Teacher Verification
- 设计 **4D tree attention mask**，满足 fused kernel 输入格式要求（如 `[B, H, M_max, M_max]`）；
- Mask 严格遵循 ancestor-only visibility 规则，防止跨分支信息泄露；
- 支持 **dense mask** 与 **on-device compact encoding** 两种构造策略，根据 speculative budget 动态选择；
- 提供 eager fallback 路径（关闭 fused attention），便于调试与验证。

#### ✅ 支持性设计（Supporting Components）
- **Reproducible distributed pipeline**：分布式预处理与缓存机制，避免重复计算；
- **Data-driven vocabulary subset mapping**：可复用的 draft vocabulary 子集构建流程，支持速度-质量权衡控制；
- **Two-mode execution protocol**：明确区分 performance mode（启用 fused）与 reference mode（禁用 fused），支持可控对比。

---

### 🔍 相比现有方法的优势

| 维度 | 传统实现 | EAGLE-PANGU |
|------|--------|------------|
| **可移植性** | 依赖特定 backend 实现细节，难迁移 | 基于 Cache API 抽象，解耦 backend 差异 |
| **安全性** | 使用负索引、隐式 padding，易出错 | Dummy root + invariant checks，设备无关的安全 indexing |
| **性能兼容性** | 很难对接 fused kernels | 显式支持 fused attention，最大化 throughput |
| **可调试性** | 黑盒运行，失败难复现 | 结构化 trace、failure dump、eager fallback 支持诊断 |

> ✅ 总结：EAGLE-PANGU 将 speculative decoding 从“算法原型”推进为“生产可用”的系统方案。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **MT-Bench**：80 个 prompt，共 160 turns（每 prompt 2 轮对话）；
- **HumanEval-style prompts**：80 个 coding 类 prompt；
- 合计 **240 turns**，覆盖多样化输入输出长度（见 Figure 1）：
  - 平均 prompt length: ~501 tokens
  - 平均 output length: ~891 tokens

---

### ⚙️ 实验设置

| 项目 | 设置说明 |
|------|---------|
| **Hardware** | Ascend NPUs（8 卡分布式） |
| **Backend** | Pangu teacher backend（工业级部署环境） |
| **Batch Size** | 主要测试 batch size = 1（典型交互场景） |
| **Decoding Mode** | Greedy decoding（temperature=0） |
| **Max New Tokens** | 默认 1024；budget sweep 使用 256 加速扫描 |
| **Evaluation Modes** | <ul><li>**Performance mode**: fused attention 开启</li><li>**Reference mode**: fused attention 关闭，用于 debug</li></ul> |

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **Tokens per second (Tok/s)** | 实际生成速度，越高越好 |
| **Speedup (×)** | 相对于 baseline 的加速比（Tok/s_EA / Tok/s_baseline） |
| **Accept Length (accept_L)** | 每次 verification 步骤平均接受的 draft token 数量 |
| **Position-wise Acceptance Rate (accept_pos)** | 第 k 个 draft position 的接受率，反映衰减趋势 |
| **Time to First Token (TTFT)** | 首 token 延迟（若可测） |
| **End-to-end Wall-clock Time** | 包含 device sync 的完整生成耗时 |

---

### 🔁 基线方法对比
- **Baseline**: Teacher-only greedy decoding（无 speculative）
- **EA (EAGLE-PANGU)**: Tree speculative decoding with same teacher model
- 控制变量：相同 decoding config、prompt template、max_new_tokens

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 和 Figure 2）

| 指标 | Baseline | EAGLE-PANGU | 提升倍数 |
|------|----------|-------------|---------|
| **Mean Tok/s** | 17.65 | 22.42 | **1.27×** |
| **P90 Speedup** | — | — | 1.84× |
| **P99 Speedup** | — | — | **2.46×** |
| **Mean accept_L** | — | 3.17 | — |
| **Median accept_L** | — | 3 | — |
| **Max accept_L (p99)** | — | 8 | — |

> 💡 **解读**：尽管采用保守的 cache 复制策略（deepcopy），仍取得显著加速，尤其在尾部延迟表现优异。

---

### 🔍 消融实验与敏感性分析

#### （1）Budget Sensitivity Sweep（Table 2 & Figure 4）
- 扫描 node budget $ M $ 和 depth bound $ D_{max} $
- 发现 throughput **非单调增长**，存在“sweet spot”
- 最佳配置：**M=16, D_max=10 → 1.48× mean speedup**
- 更大 budget 反而降低性能，原因：
  - mask/tensorization 开销增加；
  - 深层 draft 接受率下降（见 Figure 3）

> ✅ 结论：**“越大越好”不成立，需轻量级预算调优或自适应策略**

#### （2）Stage Breakdown Profiling（Figure 5）
- 各阶段平均耗时（ms/turn）：
  - `tree_tensorization`: ~几毫秒
  - `mask_construction`: ~几毫秒
  - `teacher_verification`: 较高
  - `accept_and_commit`: 与 verification 相当
  - `prefill`: 存在明显长尾（long-tail）

> 🔍 **发现**：当前瓶颈不在 mask 构造，而在 **commit 效率** 与 **prefill 行为优化**

#### （3）Negative Result：Fixed-Window Drafter Truncation（Table 3 & Figure 6）
- 测试是否可通过限制 draft model 上下文窗口（W=128/256/512）提升效率
- 结果：**严重损害 accept_L 与 throughput**
  - W=128 → accept_L 从 3.17↓至 1.48，speedup 降至 0.69×
- Attention profiling（Figure 7）显示：
  - draft model 经常关注远距离历史（>256 tokens ago）
  - 硬截断破坏了 draft 质量

> ❌ 结论：**naive truncation 不可行，context reduction 必须语义感知**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Tree speculative decoding 可在 Ascend 上安全高效部署**  
   EAGLE-PANGU 成功克服了 KV-cache、indexing、masking 等底层差异，实现了 **端到端 1.27× 平均加速（最高达 2.46×）**。

2. **accept_L 是决定 speedup 的核心驱动因素**  
   接受长度与加速比高度正相关（Figure 2b），优化 draft model 质量是关键。

3. **存在明显的配置甜点（sweet spot）**  
   过大的 speculative tree 会因 overhead 和 accept rate 衰减而导致收益下降，建议进行 budget tuning。

4. **commit 与 prefill 是潜在优化重点**  
   当前 tensorization 和 mask 构造开销较小，未来应聚焦于 **fast cache reorder** 与 **long-context prefill 优化**。

5. **context truncation 需谨慎设计**  
   简单固定窗口截断显著损害 draft quality，应结合 attention/importance signal 做 adaptive truncation。

---

### ⚠️ 局限性

1. **加速受限于 draft cost 与 verification overhead**  
   当 speculative tree 增大时，mask、cache commit 成本上升，限制了 scaling 效果。

2. **性能依赖 draft model 质量**  
   更强的 draft model 可提高 accept_L，但也可能自身成为计算瓶颈；需平衡 drafting compute 与 gain。

3. **当前评估集中于 decoding 阶段**  
   未涵盖 long-context、multi-turn tool use、multi-device serving 等复杂 workload。

---

### 🔮 未来工作方向

1. **更紧凑的 mask 表示与深度 kernel fusion**  
   减少 mask 内存占用与同步开销，进一步释放 fused kernel 潜力。

2. **speculation-aware distillation 与 adaptive branching policy**  
   训练更适合 speculative 的 draft model，并动态调整 tree 结构。

3. **扩展至 long-context 与 multi-device 场景**  
   探索在超长上下文、工具调用、流水线并行下的适用性。

4. **自动化 budget tuning 与 online adaptation**  
   构建 feedback-driven controller，实时调整 M 与 D_max。

---

## ✅ 总结

EAGLE-PANGU 是首个将 **tree speculative decoding** 成功落地于 **Ascend NPU + Pangu backend** 生产环境的系统工作。它通过三项核心设计——**branchable cache manager**、**accelerator-safe tensorization**、**fused-compatible masked verification**——解决了异构硬件上的可移植性、正确性和性能问题。

实验证明其在真实 workload 下可带来 **高达 2.46× 的尾部加速**，同时提供完整的 **debugging support** 与 **ablation capability**，为 speculative decoding 的工程化部署树立了新标杆。

</details>

---

### 5. [Meta-RL with Shared Representations Enables Fast Adaptation in Energy Systems](https://arxiv.org/abs/2603.08418)

**Authors**: Th\'eo Zangato, Aomar Osmani, Pegah Alizadeh  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.08418v1  

#### Abstract
Meta-Reinforcement Learning addresses the critical limitations of conventional Reinforcement Learning in multi-task and non-stationary environments by enabling fast policy adaptation and improved generalization. We introduce a novel Meta-RL framework that integrates a bi-level optimization scheme wi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Meta-RL with Shared Representations Enables Fast Adaptation in Energy Systems*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **Reinforcement Learning (RL)** 在 **Energy Management Systems (EMS)** 中面临以下挑战：
- **样本效率低**：需要大量环境交互，在真实系统中成本高昂。
- **泛化能力差**：难以适应不同建筑、季节变化或负荷模式等多样化任务。
- **动态环境适应慢**：面对非平稳环境（如天气、电价波动）时策略更新缓慢。

现有 **Meta-RL** 方法（如 MAML、Reptile、CAVIA）虽能提升适应速度，但在 EMS 场景下存在不足：
- 忽视了 **actor-critic 架构间的共享表示学习**；
- 缺乏对 **高结构相似性任务之间知识复用机制** 的设计；
- 未充分考虑 **任务选择策略** 对泛化的影响。

### 提出的新方法与创新思路
本文提出一种新型 **Meta-RL 框架 —— Critic Feature Extractor Meta Learning (CFE)**，其核心创新包括：

#### ✅ 创新点 1：共享特征提取器（Shared Feature Extractor）
- 在 **actor 和 critic 网络之间共享一个可元学习的 Feature Extractor (FE)**。
- 该 FE 提取跨任务通用的状态表征 $ z = g_\phi(s) $，用于 both policy 和 value function 学习。
- 优势：促进表征迁移、减少过拟合、提高样本效率。

#### ✅ 创新点 2：内循环 Actor 参数重用机制（Actor Reuse, AR）
- 存储每个任务训练后的 task-specific actor 参数。
- 当相同或相似任务再次出现时直接复用，避免重复探索。
- 特别适用于具有长周期依赖的任务（如充放电调度）。

#### ✅ 创新点 3：任务聚类与多样性选择策略
- 使用 **傅里叶变换 + 层次聚类** 对建筑能耗行为进行分组。
- 训练时随机采样任务以增强多样性；测试时保留一个 cluster 作为 unseen 任务验证泛化能力。

### 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **Sample Efficiency** | 减少约 4 倍的适应样本复杂度（见 Fig. 3b） |
| **Fast Adaptation** | 元初始化使 agent 更快进入有效策略区域（如充放电周期） |
| **Generalization** | 共享表示 + 聚类任务划分显著提升在相似任务上的迁移效果 |
| **Stability & Convergence** | Meta-gradient norm 下降更快，表明参数更早稳定（Fig. 5） |

---

## 2. 核心实验方法和设置

### 数据集
1. **专有数据集（Proprietary Dataset）**
   - 来源：1,529 栋建筑，时间跨度为 2018–2024 年（≈30 million samples）
   - 包含建筑类型（住宅、办公、工业）、气象、电价、社会经济等多维信息
   - 用于评估跨异构集群的泛化能力

2. **CityLearn 开源数据集 [17]**
   - OpenAI Gym 风格的 BEMS 环境
   - 支持多建筑协同控制，包含可再生能源、储能单元（ESU）、电网交互等建模
   - 主要用于消融实验和基准对比

### 实验设置
- **Meta-Training 设置**：
  - 每个 meta-batch 包含 3 个任务，共 600 个 batches（N=600）
  - 内循环使用 **PPO** 作为 base RL 算法
  - 每 2,048 步更新一次策略，总计约 100k 环境步
  - 外循环采用 **Reptile 更新规则**（一阶近似）

- **Feature Extractor 结构**：
  - 三层 MLP，每层 64 neurons，ReLU 激活
  - 可替换为基于 Transformer 的时序模型（TS-FE）进行对比

- **Actor Reuse 策略**：
  - 前 10% meta-step 不允许任务重访（鼓励探索）
  - 后续按多项式增长 schedule 提升重访概率

### 评估指标
| 指标 | 定义 |
|------|------|
| **Reward 曲线** | 累积回报随训练步数的变化（越高越好） |
| **Gradient Updates 数量** | 达到特定性能所需的梯度更新次数（越少越好） |
| **Ramping** | 连续时刻间电网用电变化绝对值，反映稳定性 |
| **Financial Cost** | 实际电费支出，归一化至 Rule-Based Controller 基线 |
| **Charging Cycles 数量** | 衡量策略是否快速学会有意义的行为模式 |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Random** | 从零开始训练的独立 PPO agent |
| **Pretrained** | 在某一类建筑上预训练后迁移到其他建筑 |
| **Reptile** | Vanilla Reptile 元学习算法（无 FE 和 AR） |
| **CAVIA** | 引入 context vector 的元学习方法 |
| **RL²** | 使用 LSTM 隐状态编码任务信息的递归策略 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Fig. 3 与 Table 1）

| 指标 | CFE（本文） | Reptile | Random | Pretrained |
|------|------------|---------|--------|-----------|
| **达到 mean reward = -30 所需步骤** | ~70k | ~150k | ~400k | ~250k |
| **早期适应速度（前 30k 步）** | ⬆️ 最快上升 | 快 | 极慢 | 较慢 |
| **最终年度电费成本（归一化）** | **0.86** | 0.87 | 0.95 | 0.96 |
| **Ramping 成本（归一化）** | **0.90** | 0.90 | 1.01 | 1.03 |
| **第 15 次梯度更新后的充电周期数** | **4.8** | 6.2 | 58.6 | 18.2 |
| **第 30 次更新后周期数** | **14.3** | 14.8 | 45.3 | 17.9 |

> 注：数值越接近 1 表示越接近规则控制器表现；低于 1 表示优于基线。

### 与基线方法的对比结果
- **CFE 显著优于所有 baseline**，尤其在 **早期适应阶段** 表现出极强的学习加速度。
- **Reptile 表现良好但不如 CFE**，说明共享 FE 和 AR 机制带来额外增益。
- **CAVIA 和 RL² 泛化稳定但适应慢**，表现为“稳健通才”而非“快速学习者”。
- **Pretrained 方法灵活性差**，难以适应新动态，表现出刚性策略。

### 消融实验结果（Fig. 4b）
比较四种变体：
1. **Vanilla Reptile**
2. **Reptile + AR**
3. **Reptile + FE**
4. **Full CFE (Reptile + AR + FE)**

| 发现 |
|------|
| ➤ **FE 模块贡献最大**：加入 FE 后收敛速度和最终性能大幅提升 |
| ➤ **AR 单独作用有限**：仅加 AR 对渐近性能影响小，但在长期任务中有助于防止遗忘 |
| ➤ **FE + AR 协同效应明显**：两者结合实现最快适应与最优性能 |
| ➤ **Transformer-based FE（TS-FE）**：最终性能更高（因更强时序建模），但适应更慢 → 存在 **表达力 vs. 适应速度** 的权衡 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **共享表示是 Meta-RL 在 EMS 中高效迁移的关键**  
   - actor-critic 共享 FE 可提取环境不变动态特征，极大提升跨任务泛化能力。

2. ✅ **Actor 参数重用显著降低冗余探索**  
   - 尤其适用于周期性强、结构稳定的 EMS 控制任务（如储能调度）。

3. ✅ **任务结构高度相似时，最大化共享信息优于强调任务区分**  
   - EMS 任务间冲突小，适合通过共享结构加速学习。

4. ✅ **Meta-initialization 提供“智能先验”**  
   - 使 agent 能迅速聚焦于高回报策略空间，而非盲目探索。

5. ❌ **Meta-learning 效果依赖任务分布相似性**  
   - 当目标任务与训练任务距离过大（如 Meta-G4/G5），性能甚至低于随机初始化 → 存在 **分布外泛化瓶颈**

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **假设任务结构相似** | 若目标任务差异过大（如住宅 vs 工厂），共享表示失效 |
| **存储开销增加** | 维护 task-specific actor 参数带来额外内存负担 |
| **FE 设计敏感** | Transformer 类 FE 虽强但不利于快速适应，需权衡模型容量 |
| **静态任务划分** | 聚类固定，未考虑在线任务演化或概念漂移 |

### 未来工作方向
1. **引入 Probabilistic Latent Task Representations**  
   - 使用变分推断建模任务不确定性，提升对 OOD 任务的鲁棒性。

2. **轻量化 FE + 动态路由机制**  
   - 根据输入自动选择激活路径，兼顾表达能力和适应速度。

3. **结合 Offline Meta-RL 与 Online Fine-tuning**  
   - 利用历史数据离线元训练，上线后持续微调以应对概念漂移。

4. **扩展至 Multi-Agent EMS 场景**  
   - 探索分布式 Meta-MARL 框架支持区域级能源协调控制。

--- 

> 🔚 **总结一句话**：  
> 本文提出的 **CFE 框架** 通过 **共享表征学习 + actor 参数重用**，实现了在建筑能源管理系统中 **快速、高效、稳定的元适应**，为现实世界复杂动态环境下的智能控制提供了可行路径。

</details>

---

### 6. [Switchable Activation Networks](https://arxiv.org/abs/2603.06601)

**Authors**: Laha Ale, Ning Zhang, Scott A. King, Pingzhi Fan  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.06601v1  

#### Abstract
Deep neural networks, and more recently large-scale generative models such as large language models (LLMs) and large vision-action models (LVAs), achieve remarkable performance across diverse domains, yet their prohibitive computational cost hinders deployment in resource-constrained environments. E...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Switchable Activation Networks》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前深度神经网络（DNNs）和大规模生成模型（如 LLMs、LVAs）虽然在多个领域表现出色，但其高昂的计算成本严重限制了在资源受限环境（如边缘设备）中的部署。传统的效率优化技术存在以下不足：
- **Dropout**：仅用于训练阶段的正则化，推理时仍保持全激活，无法实现推理加速。
- **Pruning 和低秩分解**：通常为后处理操作，产生静态压缩模型，缺乏对输入上下文的适应性。
- **动态推理方法（如 MoE、SkipNet）**：引入运行时开销和不规则内存访问，难以硬件友好地部署。

### **提出的新方法与新思路**
本文提出了 **Switchable Activation Networks (SWAN)**，一种通过学习**可切换的激活机制**来实现高效神经计算的新范式。其核心思想是：
- 为每个神经元或通道配备一个**确定性的、输入依赖的二值门控开关（binary gate）**。
- 在训练过程中联合优化这些门控参数，使网络学会“何时”激活某个单元。
- 推理时通过阈值将概率转换为硬决策（hard decision），真正跳过冗余计算。

### **相比现有方法的优势**
| 对比维度 | SWAN | Dropout | Pruning | 动态推理（如 MoE） |
|--------|------|-------|--------|----------------|
| **训练效率增益** | ❌ | ✅（正则化） | ❌ | ✅ |
| **推理效率增益** | ✅（动态 + 可导出致密模型） | ❌ | ✅（静态） | ✅（动态） |
| **自适应性** | ✅（输入依赖） | ❌ | ❌ | ✅ |
| **部署友好性** | ✅（可转为小型致密模型） | ✅ | ✅ | ❌（路由复杂） |
| **容量保留** | ✅（所有参数保留） | ✅ | ❌（永久删除） | ✅ |

> ✅ 表示具备该特性，❌ 表示不具备

SWAN 的独特优势在于：**统一了稀疏性、剪枝与自适应推理的优点**，既支持高效的动态推理，也可导出紧凑的静态模型用于边缘部署。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MNIST**：手写数字分类任务，验证基本有效性。
- **ImageNet 子集 / CIFAR 类似设定**：在 VGG16 和 ResNet50 上进行图像分类实验，评估在主流架构上的表现。

### **实验设置**
- **模型架构**：测试了多种经典 CNN 架构，包括：
  - Fully-connected networks（MNIST）
  - VGG16
  - ResNet50
- **训练策略**：
  - 使用 **Adam 优化器**，初始学习率 0.001。
  - 引入 **延迟余弦调度（delayed cosine ramp）** 控制正则项权重（`λ₀`, `λ_F`, `λ_T`），避免早期抑制有用单元。
  - Gate logits 使用更高学习率且无 weight decay，确保快速收敛。
- **推断方式**：
  - 软门控（soft gating）用于训练监控。
  - 硬门控（hard gating）用于最终评估，阈值 $ T = 0.5 $。
  - 执行 **BN 重校准（Batch Normalization recalibration）** 以缓解分布偏移。

### **评估指标**
| 指标 | 描述 |
|-----|------|
| **Top-1 Accuracy** | 主要性能指标 |
| **Active Unit Fraction (%)** | 激活神经元/通道的比例，衡量稀疏程度 |
| **FLOPs Reduction (%)** | 浮点运算量减少比例 |
| **Model Size Compression** | 参数压缩率 |
| **Hard vs Soft Evaluation Gap** | 验证软门控与真实推理之间的性能差异 |

### **基线方法对比**
- **Dropout**：标准随机失活，仅训练时生效。
- **Channel Pruning (CP)**：基于幅度的通道剪枝，分原始（CP_raw）和微调后（CP）版本。
- **SWAN_raw / SWAN**：SWAN 未经微调与微调后的结果。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **MNIST 实验（图2）**
- 最终激活单元比例降至 **3%**。
- 验证准确率维持在 **接近 100%**。
- 表明绝大多数参数在该任务中是冗余的，而 SWAN 能有效识别并关闭它们。

#### **VGG16 与 ResNet50（图3–5）**
| 方法 | FLOPs (%) | Top-1 Acc (%) | 微调需求 | 备注 |
|------|-----------|----------------|----------|------|
| Baseline | 100% | ~92–94% | – | 原始模型 |
| Dropout | 100% | ↓ 显著下降 | 否 | 无推理收益 |
| CP_raw | ~5–16% | <20% | 是 | 性能崩溃 |
| CP (fine-tuned) | ~5–16% | ~60–70% | 是 | 恢复有限 |
| **SWAN_raw** | ~5–16% | >90% | 否 | 几乎无损 |
| **SWAN (5 epoch FT)** | ~5–16% | >90% | 是 | 极小恢复即达峰值 |

> 在仅保留约 **5% FLOPs** 的极端压缩下，SWAN 仍能保持 **>90% 准确率**，远超其他方法。

### **与基线方法的对比结果**
- **Dropout**：虽有正则化效果，但推理无加速，且高压缩下性能急剧下降。
- **Channel Pruning**：即使经过微调，在高剪枝率下也难以恢复性能（如 VGG16 @5% FLOPs → 16.1% Acc）。
- **SWAN**：无需大量微调即可稳定运行，展现出极强的鲁棒性和压缩潜力。

### **消融实验（隐含于训练动态分析）**
- **延迟正则化调度的重要性**：
  - 若过早引入 sparsity 正则项，会导致训练不稳定甚至发散。
  - 使用 delayed cosine ramp 显著提升稳定性与最终性能。
- **BN Recalibration 的影响**：
  - 在 ResNet、Inception、DenseNet 中尤为关键，可防止因门控导致的统计失配。
  - VGG 等简单结构受影响较小。
- **STE（Straight-Through Estimator）的作用**：
  - 保证梯度可通过非可导的硬门控传播，是端到端训练的关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. **神经网络存在巨大冗余**：在 MNIST 上仅需 3% 的激活即可达到满分精度，说明传统 dense 模型严重浪费算力。
2. **Learned Activation Control 是高效 AI 的新范式**：
   - 将“是否计算”作为可学习变量，使效率成为模型内在属性。
   - 实现了 **sparse during training, compact at deployment** 的统一框架。
3. **SWAN 实现了三赢平衡**：
   - ✅ 高准确率
   - ✅ 高压缩比（FLOPs ↓95%+）
   - ✅ 支持灵活部署（动态稀疏 or 导出致密模型）

### **方法的局限性**
1. **硬件支持挑战**：
   - 当前 GPU/TPU 对不规则稀疏计算支持不佳，可能无法完全体现速度提升。
   - 需要专用稀疏加速库（如 TensorRT-LLM、Triton）才能发挥最大效益。
2. **延迟波动问题**：
   - 不同输入激活路径不同，导致推理延迟不可预测，不适合实时性要求高的场景。
3. **额外参数开销**：
   - 每个门控需维护一个 logit 参数，增加少量存储负担（但远小于节省的计算）。

### **未来工作方向**
1. **扩展至 Transformer 架构**：
   - 应用于 LLMs/VLMs 的 attention head、FFN 层等模块的门控。
   - 结合 Mixture-of-Experts 实现更细粒度的 conditional computation。
2. **硬件协同设计**：
   - 开发支持动态稀疏执行的芯片或 runtime 系统。
3. **生物启发进一步深化**：
   - 模拟大脑中更复杂的神经集群激活模式（如 oscillatory dynamics）。
4. **理论分析**：
   - 分析 SWAN 学习到的激活模式是否具有语义意义（如特定类别激活特定子网）。

---

> 📌 **一句话总结**：  
> SWAN 提出了一种将“激活与否”作为可学习决策的新型神经网络范式，实现了**训练时自适应稀疏、推理时高效节能、部署时可压缩致密**的三位一体优势，为构建可持续、边缘友好的下一代 AI 系统提供了重要路径。

</details>

---

### 7. [Ares: Adaptive Reasoning Effort Selection for Efficient LLM Agents](https://arxiv.org/abs/2603.07915)

**Authors**: Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, Shiyu Chang  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.07915v1  

#### Abstract
Modern agents powered by thinking LLMs achieve high accuracy through long chain-of-thought reasoning but incur substantial inference costs. While many LLMs now support configurable reasoning levels (e.g., high/medium/low), static strategies are often ineffective: using low-effort modes at every step...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ARES: Adaptive Reasoning Effort Selection for Efficient LLM Agents —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代基于 **thinking LLMs** 的智能体（agents）通过长链式思维（Chain-of-Thought, CoT）推理实现了高任务成功率，但带来了巨大的 **inference cost**（推理开销），尤其是在多步决策任务中。虽然当前许多 LLM 支持配置不同的“思考级别”（如 high/medium/low），但静态策略（如始终使用低效模式）会导致性能严重下降，而随机选择则无法有效平衡成本与准确性。

因此，如何在不牺牲任务成功率的前提下，**动态地为每一步分配最合适的 reasoning effort** 成为关键挑战。

---

### 🚀 提出的新方法：ARES 框架
作者提出 **ARES**（Adaptive Reasoning Effort Selection），一个用于 **multi-step agent tasks** 的 per-step 动态推理努力选择框架，其核心思想是：

- 引入一个轻量级的 **reasoning-effort router**（路由模型），根据当前交互历史预测下一步所需的最低合理 reasoning level。
- 路由器输出 `low / medium / high` 中的一个级别，主 agent 模型据此进行相应程度的推理并执行动作。
- 该设计支持 **plug-and-play** 集成到任意现有 agent 架构中。

---

### 🔍 相比现有方法的优势
| 维度 | ARES 的优势 |
|------|-------------|
| **与静态策略相比** | 避免全局降级（如全用 low 导致性能暴跌 ~20%），实现细粒度控制 |
| **与 model routing 方法相比** | 不切换不同模型，而是利用同一模型内部的 thinking levels，避免 KV cache 丢失带来的额外开销 |
| **与单步自适应推理方法相比** | 明确建模多步依赖关系，防止早期错误传播 |
| **效率保障机制** | 支持 KV cache 复用，最大化节省 token 和延迟 |

> 💡 **核心创新**：将推理资源分配视为一个 **sequential decision-making problem**，并通过轻量路由器实现高效、可泛化的动态调控。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在三个多样化的 agent benchmark 上进行了验证：
| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **TAU-Bench** | Tool-use agents | 包含零售与航空领域的对话式工具调用任务，评估 agent 执行数据库操作的能力 |
| **BrowseComp-Plus** | Deep-research agents | 控制环境下的深度研究代理任务，涉及多轮检索与推理，确保可复现性 |
| **WebArena** | Web agents | 功能性网站导航任务（电商、论坛等），提供 accessibility tree 观测输入 |

---

### ⚙️ 实验设置
- **Backbone LLM**: `gpt-oss-20b`（主 agent 模型）
- **Router 模型**: `Qwen3-1.7B`（轻量级，用于训练 reasoning-effort router）
- **训练方式**：
  - 先通过 **Supervised Fine-Tuning (SFT)** 学习最小必要 effort 标签
  - 再通过 **Reinforcement Learning (GRPO)** 进行优化，考虑轨迹级 success 与 cost 平衡
- **推理级别定义**：`low`, `medium`, `high` 三种模式，对应不同长度和复杂度的 CoT 推理过程

---

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **性能指标** | Task Success Rate / Accuracy (%) |
| **效率指标** | 
| - `T_total`: 总推理 token 数 |
| - `T_task`: 每个任务平均推理 token 数 |
| - `T_step`: 每步平均推理 token 数 |

---

### 🆚 基线方法对比
| 基线类型 | 方法 |
|--------|------|
| **固定策略** | Always Low / Medium / High reasoning effort |
| **随机策略** | Randomly sample from {low, mid, high} at each step |
| **Prompting-based Router** | 使用更强的 LLM（如 GPT-5 或 Gemini-3-Pro）作为 router 分析状态并推荐 effort，但自身推理成本极高 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Benchmark | 方法 | 准确率 (%) | 相对 High 的提升 | T_total 下降幅度 |
|----------|------|------------|------------------|------------------|
| TAU-Bench Retail | ARES (SFT) | **54.8** | ≈持平 | ↓ **35.2%** |
| BrowseComp-Plus | ARES (SFT) | **41.3** | ≈接近上限 | ↓ **41.8%** |
| WebArena | ARES (SFT) | **46.5** | **>+1.5pp** | ↓ **45.3%** |

> ✅ 在所有任务上，ARES 实现了与 high-effort baseline 相当甚至更优的性能，同时显著降低推理 token 消耗。

---

### 🔁 强化学习进一步优化（Table 2）
引入 RL 后效果更佳：
- **TAU-Bench Retail**:
  - 准确率从 54.8% → **58.5%**
  - 总 token 从 652k → **176k**
- **TAU-Bench Airline**:
  - 准确率从 36.0% → **42.0%**（+6.0pp）
  - 总 token 从 873k → **133k**（↓~85%）

> 📉 表明 RL 能够突破 SFT 的贪心限制，发现更优的全局策略。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）是否使用 Rationale（表 3）
| 设置 | 准确率 (%) | T_total |
|------|-----------|--------|
| SFT w/o rationale | 51.3 | 474k |
| Full ARES (with rationale) | **54.8** | 652k |

> ❗ 加入 rationale 可使准确率提升 **3.5pp**，说明显式 reasoning 过程有助于提高判断质量。

#### （2）RL 中奖励归一化的影响（图 5 & 表 4）
- 使用 normalized cost reward 比 unnormalized 更能抑制 high-effort 使用（降至 <15% vs 30%）
- 最终 token 消耗减少约 **15%**，且准确率更高（42.0 vs 41.3）

> ✅ 归一化 cost reward 更有利于平衡长期收益与短期节省。

#### （3）跨尺度泛化能力（表 5）
在更大的 backbone `gpt-oss-120b` 上测试：
- ARES 达到 **65.2%** 准确率（接近 High 的 67.8%）
- Token 消耗仅为其 **~77%**

> 🧩 表明 ARES 学到的 reasoning pattern 是 scale-invariant 的，具备良好泛化性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **并非所有步骤都需要高强度推理**：简单动作（如打开 URL）可用 low-effort 完成；复杂规划或纠错需 high-effort。
2. **动态 effort selection 可大幅节省 token**：最高达 **52.7%** 的推理 token 节省（相对 fixed high），且几乎无性能损失。
3. **过度推理可能有害**：在某些任务（如 Airline）中，“high” effort 反而导致准确率下降（38.0% vs medium 的 42.0%），即存在 “**overthinking**” 现象。
4. **ARES 成功规避 overthinking**：通过 RL 学习自动压制不必要的 high-effort 选择（见图 4），转向更高效的 low/medium 策略。
5. **轻量 router + KV cache 复用极具性价比**：相比使用 GPT-5/Gemini-3-Pro 作 router，ARES 自身开销极小，却能达到相近甚至更好的性能。

---

### ⚠️ 方法的局限性
- 当前依赖人工定义的 effort levels（low/med/high），尚未完全自动化分级。
- 训练依赖高质量轨迹数据生成与标注，流程较重。
- 对于极端复杂的任务（如超长 horizon 或高度不确定性环境），仍可能存在误差累积风险。

---

### 🔮 未来工作方向
- 将 ARES 扩展至 **multi-modal inputs**（如视觉输入的 GUI agent）
- 探索 **continuous reasoning budget allocation** 而非离散 level 选择
- 结合 **online adaptation**，让 router 在部署过程中持续学习和调整
- 应用于更多现实场景，如自动驾驶、机器人控制等 agentic systems

---

## ✅ 总结一句话
> **ARES 通过一个轻量级 router 实现 per-step 的 adaptive reasoning effort selection，在保持甚至超越 high-effort 性能的同时，最多节省 52.7% 的推理 token，显著提升了 LLM agent 的推理效率与实用性。**

</details>

---

### 8. [In-Context Reinforcement Learning for Tool Use in Large Language Models](https://arxiv.org/abs/2603.08068)

**Authors**: Yaoqi Ye, Yiran Zhao, Keyu Duan, Zeyu Zheng, Kenji Kawaguchi, Cihang Xie, Michael Qizhe Shieh  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.08068v1  

#### Abstract
While large language models (LLMs) exhibit strong reasoning abilities, their performance on complex tasks is often constrained by the limitations of their internal knowledge. A compelling approach to overcome this challenge is to augment these models with external tools -- such as Python interpreter...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：In-Context Reinforcement Learning for Tool Use in Large Language Models**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
大型语言模型（LLMs）虽然具备强大的推理能力，但其内部知识是静态且固定的，难以应对需要**最新事实信息**或**复杂计算**的任务（如数学推理、多跳问答）。为解决此问题，研究者引入外部工具（如搜索引擎、Python解释器），但如何让LLMs高效学会调用这些工具仍是一大挑战。

传统方法依赖 **Supervised Fine-Tuning (SFT)** + **Reinforcement Learning (RL)** 的两阶段流程，其中SFT需要大量人工标注的“工具使用轨迹”（tool-use traces），成本高昂且难以扩展。

---

### 🚀 提出的新方法：In-Context Reinforcement Learning (ICRL)

ICRL 是一种**仅使用强化学习（RL-only）** 的轻量级训练框架，**完全摒弃了SFT阶段**，通过在 RL 的 rollout prompt 中嵌入 **few-shot in-context 示例** 来引导模型学习工具使用行为。

#### 核心思想：
- 在 RL 训练初期，在每个 rollout prompt 前添加少量（如3个）示范样例（demonstrations），展示如何进行 step-by-step 推理、调用 `<search>` 或 `<code>` 工具，并输出结构化答案。
- 随着训练推进，逐步减少这些 in-context 示例的数量，最终过渡到 **zero-shot 设置**，促使模型内化工具使用策略。
- 整个过程形成一个 curriculum learning 路径：从模仿 → 自主决策。

---

### 🔍 相比现有方法的优势
| 维度 | 传统 SFT+RL 方法 | ICRL |
|------|------------------|------|
| 数据需求 | 需要大量标注的 tool-use 轨迹 | **无需任何标注数据或SFT** |
| 成本 | 高昂（人工标注 / 合成轨迹） | 极低（仅需几个手工构造的示例） |
| 可扩展性 | 受限于标注质量与规模 | 易于迁移至不同任务和工具 |
| 性能表现 | 依赖冷启动SFT才能有效探索 | 在无监督初始化下仍能高效学习 |

> 💡 **核心优势**：将 **prompting 的样本效率** 与 **RL 的探索适应能力** 结合，提供了一种**可扩展、数据高效**的替代方案。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集

#### **训练数据集**：
- **Natural Questions (NQ)**：真实用户的谷歌搜索问题 + 对应维基百科段落作为答案来源。
- 所有模型均未见过测试集中的问题，避免数据泄露。

#### **评估基准（Evaluation Benchmarks）**：
| 数据集 | 类型 | 特点 |
|-------|------|------|
| **TriviaQA** | 单跳问答 | 涉及广泛常识 |
| **HotpotQA** | 多跳推理 | 需跨多个文档推理 |
| **2Wiki** | 多跳推理 | 基于维基百科构建的合成多跳问题 |
| **Musique** | 多跳推理 | 更复杂的链式推理路径 |
| **Bamboogle** | 多跳 + 工具增强 | 设计用于评估 agent 式搜索行为 |

> ⚠️ 所有评估均排除 NQ，防止过拟合。

---

### 🛠 实验设置

- **骨干模型**：
  - Qwen2.5 系列：`3B`, `7B`, `14B` Instruct 版本
  - Qwen3-8B（含 RL 增强）
- **工具集成**：
  - Web Search：Serper API（Google 搜索 Top-3 结果）
  - Code Execution：Python interpreter 执行生成代码
- **最大响应长度**：2048 tokens，支持最多 6 次工具调用
- **训练平台**：Volcano Engine RL Framework (VeRL)，FSDP + Gradient Checkpointing
- **硬件配置**：4×NVIDIA A100 (80GB)

---

### 🎯 评估指标

- **Exact Match (EM) Accuracy (%)**：预测答案与标准答案完全匹配的比例。
- **Format Correctness**：是否正确使用 `<think>`, `<search>`, `<answer>` 等 XML 标签。
- **Reward Design**：
  $$
  r_o(q,y) = \alpha \cdot \text{reward}_{acc} + (1-\alpha) \cdot \text{reward}_{format},\quad \alpha=0.8
  $$
- **Loss Masking**：只对 LLM 生成的部分计算梯度，忽略检索返回的内容。

---

### 🆚 基线方法对比

分为三类：

| 类别 | 方法 |
|------|------|
| **Prompting** | Direct, CoT, IRCoT, Search-o1 |
| **Retrieval-based** | RAG, ZeroSearch |
| **Fine-tuning based** | SFT, R1-instruct, Reject Sampling |
| **RL + Search** | Search-R1, O2-Searcher (√SFT), ParallelSearch |

> 特别强调与 **O2-Searcher** 的比较：后者需先做 SFT 再进入 RL，而 ICRL 完全跳过 SFT。

---

## 3. **主要实验结果和性能指标**

### 📊 主要性能结果（见 Table 3）

| 模型 | 方法 | 平均 EM (%) | 最佳提升 |
|------|------|-------------|----------|
| Qwen2.5-3B | ICRL | **40.16** | +8.94 vs Search-R1 |
| Qwen2.5-7B | ICRL | **49.12** | +7.34 vs ParallelSearch |

#### ✅ 关键亮点：
- 在 **Qwen2.5-7B** 上，ICRL 在 **4/5 数据集上取得 SOTA**：
  - TriviaQA: **75.4**
  - 2Wiki: **53.6**
  - Musique: **26.0**
  - Bamboogle: **48.0**
- 多跳任务增益显著：
  - 在 **2Wiki (+7.3)** 和 **Musique (+9.7)** 上远超基线
- 尤其在 **Bamboogle** 上比 ZeroSearch 提升 **+36.9** 分（7B模型）

---

### 📉 与 O2-Searcher 的直接对比（Table 4）

| 模型 | 方法 | 是否使用 SFT | 平均 EM |
|------|------|---------------|---------|
| Qwen2.5-3B | O2-Searcher | ✅ Yes | 37.26 |
| Qwen2.5-3B | ICRL | ❌ No | **40.16** |

> ❗ **ICRL 在不使用任何 SFT 的情况下，反而超越了依赖冷启动 SFT 的先进方法**，证明 in-context 示例足以支撑 RL 学习。

---

### 🔬 消融实验分析（Ablation Studies）

#### （1）Curriculum 设计影响（Figure 2）
- 对比两种 curriculum：
  - **3→2→0**（三阶段）
  - **3→2→1→0**（四阶段）
- 发现：
  - 四阶段导致更早终止（>80% 查询在2步内结束），但准确率大幅下降（TriviaQA 仅 20.8 vs 75.4）
  - **中间插入 1-shot 阶段会诱导 premature stopping**
  - **3→2→0 更优**：允许充分探索长链推理

#### （2）模型缩放效应（Table 6）
| 模型 | 方法 | 平均 EM |
|------|------|--------|
| Qwen2.5-14B | ICRL | **51.84** |
- 显著优于 Direct（+27.0）和 CoT（+20.7）
- 表明 ICRL 具备良好 **scaling property**

#### （3）训练动态可视化（Figure 3）
- 初期（3-shot）响应长度稳定
- 进入 0-shot 后短暂下降，随后回升 → 模型学会自主构造长输出
- **Valid Search 调用次数持续上升** → 成功内化工具使用行为

---

### 🧮 数学推理任务表现（Table 7）

| 模型 | 方法 | AIME2024 | AIME2025 |
|------|------|---------|---------|
| Qwen3-8B | ReTool (SFT+RL) | **67.0** | 64.1 |
| Qwen3-8B | ICRL (No SFT) | 64.1 | **66.5** |

> 虽然在 AIME2024 上略逊 2.9%，但在 AIME2025 上反超 **+2.4%**，说明 ICRL 在 code-as-tool 场景中同样有效且更具数据效率。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **ICRL 实现了无需 SFT 的高效工具学习**  
   > 仅靠 in-context 示例 + RL 信号即可教会 LLM 调用工具，摆脱对昂贵标注数据的依赖。

2. **性能全面领先现有方法**  
   > 在多个 QA 和数学推理任务上达到 SOTA，尤其在多跳推理场景中优势明显。

3. **课程设计至关重要**  
   > 渐进式移除 in-context 示例能有效促进策略内化；过于频繁的阶段切换反而有害。

4. **具备良好的泛化性和可扩展性**  
   > 支持 web search、code execution 等多种工具类型，并能在更大模型（14B）上继续提升。

---

### ⚠️ 局限性

- 当前依赖手工设计的 few-shot 示例，尚未实现全自动示例生成。
- 所有实验基于 instruction-tuned 模型，base model 上的效果未知。
- 工具调用格式固定（XML tags），可能限制表达灵活性。
- 未探讨多工具协同调度问题（如 search + code + calculator 联合使用）。

---

### 🔮 未来工作方向

1. **自动化 in-context 示例构建**  
   > 利用 LLM 自动生成高质量演示样本，进一步降低人工干预。

2. **扩展至更多工具生态**  
   > 支持数据库查询、API 调用、仿真环境交互等复杂工具。

3. **结合 memory 或 planning 模块**  
   > 提升长期推理与状态追踪能力，应对更深的推理链。

4. **应用于 real-world agent 系统**  
   > 如智能助手、科研助理、自动编程代理等实际场景。

---

> 🏁 **总结一句话**：  
> **ICRL 开辟了一条“无需监督微调”的新路径，让 LLM 在强化学习中通过 in-context 示例学会使用工具，兼具高性能与高数据效率，是迈向通用工具型智能体的重要一步。**

</details>

---

### 9. [Covenant-72B: Pre-Training a 72B LLM with Trustless Peers Over-the-Internet](https://arxiv.org/abs/2603.08163)

**Authors**: Joel Lidin, Amir Sarfi, Erfan Miahi, Quentin Anthony, Shivam Chauhan, Evangelos Pappas, Benjamin Th\'erien, Eugene Belilovsky, Samuel Dare  
**Category**: cs.DC  
**Published**: 2026-03-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.08163v1  

#### Abstract
Recently, there has been increased interest in globally distributed training, which has the promise to both reduce training costs and democratize participation in building large-scale foundation models. However, existing models trained in a globally distributed manner are relatively small in scale a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Covenant-72B: Pre-Training a 72B LLM with Trustless Peers Over-the-Internet》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前大规模语言模型（LLM）的预训练严重依赖于集中式高性能计算集群（如数千个通过高速互联连接的GPU），这导致训练成本高昂且仅限于少数大型组织参与。尽管已有研究探索**分布式训练**以降低门槛，但现有方案存在以下局限：
- 规模较小（通常 < 10B 参数）
- 依赖**白名单制参与者**（whitelisted participants），无法实现真正开放、去中心化的协作
- 缺乏对**非可信节点**（trustless peers）的支持，难以在开放互联网环境下运行

本文旨在解决：**如何在无信任假设、基于普通互联网带宽的全球分布式算力网络上，高效地完成超大规模 LLM 的预训练？**

---

### **提出了什么新方法或新思路**
论文提出并实现了 **CoVENANT-72B**，这是迄今为止最大规模的、完全去中心化、无需许可（permissionless）参与的 LLM 预训练项目，其核心技术组合为：

#### ✅ **SparseLoCo**（通信高效的优化器）
- 一种基于局部更新（local update）的分布式优化算法
- 结合 **Top-k sparsification**、**error feedback** 和 **2-bit quantization**，大幅压缩梯度通信量（>146× 压缩比）
- 支持动态节点加入/退出，适应不稳定网络环境

#### ✅ **Gauntlet**（激励与验证机制）
- 构建在 **Bittensor 区块链**上的协调协议
- 引入 **LossScore** 作为评估信号：比较每个节点提交的 pseudo-gradient 在小批量数据上前后的 loss 变化
- 使用 **OpenSkill 排名系统**稳定评分，防止随机性干扰
- 实现无需信任的激励机制：只有高质量贡献者才能被选中聚合，并获得奖励

#### ✅ **Chunk-wise Top-k 压缩策略**
- 将张量划分为固定大小块（64×64 或 4096-sized chunks），分别进行 Top-k 选择
- 减少索引开销（index overhead），提升压缩效率与工程可行性

---

### **相比现有方法的优势**
| 维度 | 传统集中式训练 | 白名单分布式训练（如 INTELLECT-1） | CoVENANT-72B |
|------|----------------|-------------------------------|-------------|
| 模型规模 | 大（70B+） | 小（≤10B） | **大（72B）** |
| 参与方式 | 封闭 | Whitelisted | ✅ **Permissionless** |
| 算力来源 | 数据中心 | 协议内节点 | ✅ **全球开放互联网节点** |
| 通信效率 | 高（高带宽） | 中等 | ✅ **低带宽下仍高效（<500Mb/s）** |
| 容错能力 | 弱 | 中等 | ✅ **支持动态进出、抗恶意行为** |

> 🔑 **核心突破**：首次证明了在**无中心控制、无身份审核、仅靠商品级互联网连接**的条件下，可以成功训练出具备竞争力的 72B 级别 LLM。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **主训练阶段（~1.09T tokens）**：
  - 来自 **DCLM** 的网页文本
- **退火阶段（annealing phase, ~14.2B tokens）**：
  - 高质量混合数据：指令数据（27%）、合成网页（20%）、代码（15%）、数学（13%）、自然网页回放（25%，用于缓解遗忘）
- **监督微调 SFT（~14.8B tokens）**：
  - 开源对话与指令数据，涵盖 chat、code、math、STEM、agent tasks
  - 分为 4k 和 8k 上下文长度两个版本，后者额外混入 20% 预训练回放数据

所有数据均提前分片并托管于对象存储（Cloudflare R2），避免在线 tokenization 瓶颈。

---

### **实验设置和评估指标**

#### **模型架构**
- **Decoder-only Transformer**，风格类似 LLaMA-3
- 参数量：**72.7B**
- 层数：80
- hidden size：8192
- 注意力头数：64 query heads, 8 KV heads（GQA）
- RoPE base: 500,000
- 序列长度：2048（预训练），扩展至 8192（SFT）
- Tokenizer：Gemma 3 SentencePiece（vocab size: 262,208）

#### **训练配置**
- **优化器**：SparseLoCo + AdamW（inner optimizer）
- **Inner steps (H)**：30
- **每轮通信间隔**：约 20 分钟计算时间
- **压缩设置**：
  - Top-k sparsification：k=64, chunk size=4096
  - Quantization：2-bit
  - Error feedback decay β=0.95
  - 外层学习率 α=1
- **通信后端**：Cloudflare R2 对象存储（替代直接 P2P）
- **区块链平台**：Bittensor Subnet 3

#### **评估指标**
- **零样本准确率（zero-shot accuracy）**：
  - ARC-Challenge/Easy, PIQA, OpenBookQA, HellaSwag, WinoGrande, MMLU
- **SFT 后评估（few-shot）**：
  - 增加 GSM8K, BBH-CoT, IFEval, MATH, MMLU-Pro, MuSR
- 所有模型统一使用 `lm-eval-harness` v0.4.11 进行评测

---

### **基线方法对比**
| 基线模型 | 类型 | 参数量 | Token 数 | 是否中心化 | 是否开放参与 |
|--------|------|--------|----------|------------|--------------|
| INTELLECT-1 | 分布式 | 10B | 1T | Internet | ❌ No |
| Psyche Consilience | 分布式 | 40B | 1.2T | Internet | ❌ No |
| LLM360 K2 | 中心化 | 65B | 1.4T | Cluster | ❌ No |
| LLaMA-2-70B | 中心化 | 70B | 2T | Cluster | ❌ No |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **预训练后零样本表现（Table 1）**
| 模型 | ARC-C | ARC-E | PIQA | OBQA | HellaSwag | WinoGrande | MMLU |
|------|-------|-------|------|------|-----------|------------|--------|
| **CoVENANT-72B** | **56.8** | **80.9** | **81.6** | **44.0** | **80.6** | **75.9** | **67.1** |
| LLaMA-2-70B | 57.4 | 79.6 | 82.6 | 49.4 | 84.3 | 80.4 | 65.6 |
| LLM360 K2 | 53.8 | 76.0 | 82.5 | 48.0 | 82.9 | 76.4 | 65.5 |

> 💡 尽管训练 token 数仅为 LLaMA-2 的一半（1.1T vs 2T），CoVENANT-72B 在多数任务上接近甚至超越这些强中心化基线，尤其在 **MMLU (+1.5 pts)** 表现出色。

#### **SFT 后聊天模型表现（Table 2）**
| 模型 | ARC-C | GSM8K | MMLU | IFEval | MATH | BBH-CoT |
|------|-------|--------|--------|--------|--------|---------|
| **CoVENANT-72B-Chat** | 64.2 | **63.9** | **67.4** | **64.7** | **26.3** | 55.0 |
| K2-Chat (65B) | 62.0 | 79.0 | 67.9 | 45.5 | 19.1 | 69.8 |
| LLaMA-2-70B-Chat | 65.4 | 52.2 | 63.1 | 40.7 | 10.7 | 63.2 |

> 🚀 **亮点发现**：
> - **IFEval 得分最高（64.7）** → 显示极强的指令遵循能力
> - **MATH 得分显著领先（26.3）** → 表明数学推理能力强于同类模型
> - 在保持强大通用能力的同时，SFT 成功提升了特定任务表现

---

### **与基线方法的对比结果**
- **vs INTELLECT-1（10B, whitelisted）**：
  - 模型规模大 **7.2×**
  - 通信同步频率高 **3.3×**（H=30 vs H=100）
  - 每轮通信耗时仅 **70秒**（vs 8.3分钟）
  - 计算利用率高达 **94.5%**（vs 82.1%）
- **vs LLaMA-2-70B**：
  - 虽然 token 数少近半，但在多个任务上达到可比水平
  - 特别是在 MMLU 上反超，说明数据质量和训练策略的有效性

---

### **消融实验与分析（Appendix B）**

#### **退火阶段（Annealing Phase）的影响（Table 3）**
| 指标 | Pre-Anneal | Post-Anneal | Δ |
|------|------------|-------------|----|
| MMLU | 62.5 | **67.1** | +4.6 |
| ARC-C | 56.4 | 56.8 | +0.4 |
| PIQA | 82.2 | 81.6 | -0.6 |

> ✅ **结论**：退火阶段显著提升了复杂任务（如 MMLU）的表现，轻微牺牲简单任务精度，总体利大于弊，有助于为后续 SFT 做准备。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **去中心化训练可扩展至 72B 规模**：
   - 首次实现在开放互联网、无白名单限制下训练超大规模 LLM
2. ✅ **SparseLoCo + Gauntlet 架构可行且高效**：
   - >146× 梯度压缩 + 区块链验证机制，使低带宽环境下的协作成为可能
   - 平均计算利用率达 **94.5%**
3. ✅ **性能媲美中心化模型**：
   - 尽管硬件条件受限，最终模型在多项基准测试中与 LLaMA-2-70B 等模型相当
4. ✅ **开放参与不损害模型质量**：
   - 动态节点进出（平均 16.9 个贡献节点/轮，最多 70 个唯一参与者）未影响收敛稳定性

---

### **方法的局限性**
- **依赖外部区块链基础设施**（Bittensor）：增加了系统复杂性和潜在单点故障风险
- **通信延迟仍存在瓶颈**：虽然已高度优化，但在极端低速网络下可能进一步降低效率
- **安全边界有限**：虽有 LossScore 和 OpenSkill 抵御部分恶意行为，但仍可能存在高级对抗攻击（如梯度污染）
- **数据分配一致性挑战**：需确保各节点训练不同数据 shard，否则可能导致重复学习或抄袭

---

### **未来工作方向**
1. **扩大参与者异构性**：
   - 支持更多类型的设备（如消费级 GPU、边缘节点）
2. **探索更鲁棒的信任机制**：
   - 替代或增强 Gauntlet 的验证逻辑，例如引入 zk-proof 验证训练完整性
3. **全生命周期去中心化训练**：
   - 不仅预训练，也尝试将 SFT、RLHF 等阶段迁移到开放网络
4. **跨子网协同训练**：
   - 利用 Bittensor 多 subnet 生态，实现模块化、分工式训练
5. **绿色 AI 推进**：
   - 利用闲置算力，减少碳足迹，推动可持续的大模型发展

---

> 📌 **总结一句话**：  
> **CoVENANT-72B 证明了“任何人、任何地方”的 GPU 都可以共同训练世界级大模型——这是迈向真正民主化 AI 的关键一步。**

</details>

---

### 10. [NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks](https://arxiv.org/abs/2603.06922)

**Authors**: Nandan Kumar Jha, Brandon Reagen  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.06922v1  

#### Abstract
We introduce NerVE, a unified eigenspectral framework for understanding how feed-forward networks (FFNs) in large language models (LLMs) organize and regulate information flow in high-dimensional latent space. Despite FFNs dominating the parameter budget, their high-dimensional dynamics remain poorl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
尽管 **Feed-Forward Networks (FFNs)** 在 **Large Language Models (LLMs)** 中占据了绝大部分参数和计算开销，其在高维隐空间中的非线性动态机制仍缺乏系统理解。现有研究多聚焦于 **Attention** 机制，而对 FFN 如何组织和调节信息流的几何本质知之甚少。

本文旨在填补这一空白，提出一个统一框架来量化和解释 FFN 非线性激活如何重塑高维表示的 **eigenspectrum**（特征谱）。

### 提出了什么新方法或新思路
作者提出了 **NerVE**（**N**onlinear **E**igenspectrum **V**ariance **E**volution），一个轻量级、内存高效的在线分析框架，用于追踪 FFN 中的特征谱动态。

NerVE 的核心是四个互补的、尺度不变的 **eigen-metrics**，从不同角度刻画特征谱的变化：

- **Spectral Entropy (SE)**：衡量方差分布的均匀性（离散 vs 均匀）
- **Participation Ratio (PR)**：衡量有效维度数（effective dimensionality）
- **Eigenvalue Early Enrichment (EEE)**：衡量“前重性”（top-heaviness），即前几个主成分集中了多少方差
- **Jensen-Shannon Divergence (JS)**：衡量前后激活特征谱之间的分布偏移

这些指标共同揭示了 FFN 非线性不仅不是简单的缩放操作，而是主动地将方差重新注入到未被充分利用的方向中。

### 相比现有方法的优势
| 方面 | NerVE 的优势 |
|------|-------------|
| **理论基础** | 基于信息论和统计物理（如 von Neumann entropy），具有坚实的数学基础 |
| **全面性** | 四个指标互补，避免单一标量指标的片面性 |
| **效率** | 支持在线、低内存计算，适用于大规模模型训练监控 |
| **通用性** | 不仅适用于 Transformer，也验证于 **MLP-Mixer** 架构，跨架构通用 |
| **诊断能力** | 能够为架构设计（如 Norm placement、FFN width、RoPE）和优化器选择提供可解释的指导 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CodeParrot**：2.1B tokens，用于训练 GPT-2 (125M)
- **OpenWebText**：26B tokens，用于研究 RoPE
- **FineWeb**：用于研究不同优化器（AdamW, Muon, Dion）
- **C4**：用于训练 LLaMA 风格模型（71M 到 1.3B）
- **CIFAR-100**：用于验证非 Transformer 架构 **MLP-Mixer**

### 实验设置
- **模型**：GPT-2、LLaMA-style、MLP-Mixer B/16
- **训练配置**：多种 FFN 宽度（D=1d–8d）、不同的 LayerNorm 位置（PreLN, PostLN, MixLN）、激活函数（GELU, ReLU, Leaky ReLU）、权重归一化（Weight/Spectral/Hyperspherical Norm）、位置编码（RoPE vs NoPE）、优化器（AdamW, Muon, Dion, Adafactor, SGD）
- **硬件**：NVIDIA RTX 3090 GPUs

### 评估指标
- **主要指标**：四个 eigen-metrics（SE, PR, EEE, JS）
- **性能指标**：**Perplexity (PPL)** 和 **Accuracy**
- **相关性分析**：使用 **Pearson correlation (r)** 分析 eigen-metrics 与验证损失（validation loss）的关系，以评估其作为训练诊断工具的有效性

### 基线方法对比
NerVE 并非直接替代某个基线模型，而是作为一种**分析工具**，用于比较不同配置下的内部动态差异。其“基线”是标准的 GPT-2 或 LLaMA 训练流程，并通过改变以下因素进行对比：
- 是否使用 LayerNorm
- 使用哪种优化器
- 使用哪种位置编码
- FFN 的宽度和归一化方式

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 配置 | Perplexity (PPL) | PR_post (D=6144) | 关键发现 |
|------|------------------|----------------|----------|
| **GPT-2 Baseline (GELU)** | 2.714 | ~1822 | PreLN 利用率最高 |
| **Norm-Free GELU** | 3.223 | ~1 | 存在 **spectral inertia** |
| **Norm-Free ReLU** | 2.988 | ~200 | 显著补偿 LayerNorm 缺失 |
| **RoPE** | 15.20 | 高且稳定 | 防止中深层 **spectral collapse** |
| **NoPE** | 16.78 | 下降 | 出现 **spectral collapse** |
| **Muon** | 25.68 | 最高 | 优化器诱导最优谱特性 |
| **AdamW** | 33.24 | 较低 | 需大量非线性修复 |

### 与基线方法的对比结果
- **PreLN vs PostLN**：PreLN 在所有 FFN 宽度下均表现出更高的 **PR/D**（利用率），且随宽度增加保持稳定；PostLN 表现出 **diminishing returns**（边际效益递减）。
- **GELU vs ReLU**：两者趋势相似，但 GELU 探索更广的子空间，最终达到更高的 PR_post，与更低的困惑度一致。
- **RoPE vs NoPE**：RoPE 有效防止了中深层的 **PR** 和 **SE** 下降，显著提升了模型性能（PPL 15.20 vs 16.78）。
- **Muon vs AdamW**：Muon 产生的 pre-activation 特征谱更健康（PR_pre 更高），因此非线性只需“精炼”而非“修复”，最终性能更优。

### 消融实验结果
#### (1) LayerNorm 位置消融（Table 7）
| 方法 | D=6144 时 PR_post |
|------|------------------|
| PreLN | 1822 |
| MixLN | 233 |
| PostLN | 71 |

PreLN 的绝对容量是 PostLN 的 **25倍以上**。

#### (2) 归一化缺失下的补偿机制（Figure 4）
- **Norm-Free GELU**：早期层出现 **spectral inertia**（EEE→1, JS→0），无法激活新方向。
- **Norm-Free ReLU**：早期 FFN 层 **PR gain** 激增 200×，成功打破瓶颈。

#### (3) 优化器角色：修复 vs 精炼（Figure 8, 9）
- **AdamW**：导致 pre-activation **collapse**，需要非线性进行大规模“**repair**”。
- **Muon**：保持 pre-activation 高维，非线性仅需轻微“**refinement**”，最终 **PR_post** 更高。

#### (4) MLP-Mixer 验证（Table 12）
| FFN1 | FFN2 | Acc. (%) |
|------|------|---------|
| GELU | GELU | 66.96 |
| ReLU | GELU | 67.86 |
| GELU | ReLU | 66.99 |
| ReLU | ReLU | 67.20 |

**ReLU** 在 **channel-mixing FFN (FFN2)** 中能更有效地展平谱，提升准确率。

---

## 4. 关键结论和发现

### 主要发现
1. **FFN 非线性是主动的方差再分配器**  
   非线性（如 GELU）并非简单缩放，而是将方差重新注入到低能量方向，**提高 SE 和 PR，降低 EEE**，从而展平特征谱，增强表示多样性。

2. **优化器几何强烈调制 FFN 动态**  
   优化器的选择决定了 FFN 非线性的角色：  
   - **AdamW**：导致 pre-activation **collapse**，迫使非线性进行“**repair**”。  
   - **Muon**：维持健康的 pre-activation 谱，使非线性扮演“**refinement**”角色，效率更高。

3. **架构选择留下独特的“光谱指纹”**  
   - **PreLN**：提供最佳的宽度利用率（return-on-width）。  
   - **RoPE**：防止中深层 **spectral collapse**，提升深度利用。  
   - **ReLU**：在无归一化模型中能主动补偿，缓解 **entropic overload**。

4. **NerVE 指标可预测泛化能力**  
   **SE_pre** 和 **PR_pre** 与验证损失高度负相关（|r| > 0.97），可作为无需反向传播的在线训练诊断工具。

### 方法的局限性
- **逐层独立分析**：未显式建模层间特征谱的相干性。
- **忽略序列位置**：默认聚合所有 token，可能掩盖位置相关的动态（如后期 token 利用更多维度）。
- **计算成本**：全批次协方差和特征分解在超大模型上仍昂贵，需依赖采样或近似（但会损失部分诊断精度）。

### 未来工作方向
- 开发跨层的 **spectral coherence** 度量。
- 结合 **token-position stratification** 进行更细粒度分析。
- 将 NerVE 指标用于自动化架构搜索（NAS）或优化器选择。
- 探索在其他模态（如语音、图网络）中的应用。

--- 

> **总结**：NerVE 为理解 LLM 中占主导地位却长期被忽视的 FFN 组件提供了首个系统性的、基于特征谱的分析框架。它揭示了非线性、归一化、优化器等设计选择如何通过塑造高维表示的几何结构来影响模型性能，为超越试错法的模型设计提供了强有力的理论支持和实用工具。

</details>

---

### 11. [Making LLMs Optimize Multi-Scenario CUDA Kernels Like Experts](https://arxiv.org/abs/2603.07169)

**Authors**: Yuxuan Han, Meng-Hao Guo, Zhengning Liu, Wenguang Chen, Shi-Min Hu  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.07169v1  

#### Abstract
Optimizing GPU kernels manually is a challenging and time-consuming task. With the rapid development of LLMs, automated GPU kernel optimization is gradually becoming a tangible reality. However, current LLM-driven automated optimization methods narrowly focus on machine learning applications, such a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Making LLMs Optimize Multi-Scenario CUDA Kernels Like Experts》核心总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **LLM** 的自动化 **CUDA kernel** 优化方法存在显著局限性：
- 多数研究聚焦于深度学习框架（如 PyTorch）中的特定算子优化，忽视了更广泛的高性能计算（**HPC**）场景，例如稀疏矩阵运算、科学计算等。
- 现有基准测试（如 **KernelBench**）局限于 **LLM-related operators**，缺乏对多样化内存访问模式、不规则计算任务的系统性评估。
- 缺乏一个通用、端到端的自动化优化框架，能够适应多场景、多硬件瓶颈，并自动生成完整的编译与执行工具链。

因此，本文旨在解决**通用型、多场景 CUDA kernel 自动化优化**的问题，推动 LLM 在低层次高性能编程中的实际应用能力。

### 提出了什么新方法或新思路
论文提出了两个核心组件：

#### ✅ **MSKernelBench**：首个面向多场景的综合性 CUDA 优化基准
- 覆盖四大类任务：
  - **Dense Linear Algebra**（稠密代数）
  - **Sparse Matrix Operators**（稀疏矩阵）
  - **LLM Operator Sequences**（常见 LLM 算子组合）
  - **Scientific Computing Kernels**（科学计算内核，如 stencil、数值方法）
- 支持 **FP32** 和 **BF16** 精度
- 引入**多尺度数据规模测试**，以评估优化在不同负载下的可扩展性
- 使用纯 **C/C++** 实现，剥离高层框架依赖，确保评估贴近底层性能极限

#### ✅ **CUDAMaster**：多智能体、硬件感知的端到端优化系统
- **Multi-Agent 架构**：多个专业化 Agent 分工协作，分别负责代码生成、性能分析、正确性验证、工具链构建等任务
- **Filtered Profiling Guidance**：利用 **Nsight Compute** 等工具获取硬件性能剖析数据，并通过选择性过滤机制提取关键瓶颈信息（如 memory bandwidth bound、compute bound），指导 LLM 进行有针对性的优化
- **End-to-End Toolchain Generation**：不仅能生成优化后的 `.cu` 文件，还能自动构造编译脚本、执行环境和测试流程，实现真正“一键部署”

### 相比现有方法的优势
| 维度 | 现有方法（如 Astra、CudaForge） | 本文方法（CUDAMaster + MSKernelBench） |
|------|-------------------------------|----------------------------------------|
| 场景覆盖 | 主要限于 LLM 算子 | 涵盖稠密/稀疏/科学计算等多领域 |
| 基准设计 | 单一数据尺寸、PyTorch绑定 | 多尺度、跨领域、C语言原生实现 |
| 优化策略 | 反馈驱动迭代 | 结合 profiling 数据进行瓶颈识别与定向优化 |
| 输出完整性 | 仅输出 kernel 代码 | 自动生成完整 toolchain（编译+运行+验证） |
| 性能上限 | 难以超越 hand-tuned 库 | 在部分任务上媲美甚至超越 **cuBLAS/cuSPARSE** |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **MSKernelBench** 包含 **50 个代表性任务**，来源于：
  - NVIDIA 官方文档（如 cuBLAS、cuSPARSE 示例）
  - 开源项目与学习平台（如 LeetGPU）
  - 已有基准（如 KernelBench 中的 LLM 算子）
- 具体类别分布：
  - 稠密线性代数（7项）
  - 稀疏矩阵（8项）
  - 归一化函数（4项）
  - 激活函数（7项）
  - LLM 算子序列（6项）
  - 科学计算（stencil、数值积分等共11项）

### 实验设置和评估指标

#### ✅ 正确性评估（Correctness）
- 对每个算子，在多个预设数据规模下生成随机输入
- 将优化后 kernel 的输出与 ground-truth（原始未优化版本）对比
- 数值误差需在容差范围内（如 `1e-5` for FP32）才算通过

#### ✅ 性能测量（Performance）
- **Speedup = Baseline 执行时间 / Optimized 执行时间**
- 每个数据规模执行 3 次热身 + 50 次正式运行，取平均时间
- 最终得分采用 **复杂度加权平均 speedup**：

$$
P = \frac{\sum_i T(N_i) \cdot S_i}{\sum_i T(N_i)}
$$

其中 $T(N_i)$ 是输入规模为 $N_i$ 时的理论计算复杂度，$S_i$ 是实测 speedup。该设计使大規模、高复杂度任务权重更高，更能反映算法级改进的价值。

### 基线方法对比
- **Astra**：当前最先进的多智能体 CUDA 优化系统
- **cuBLAS / cuDNN / cuSPARSE**：NVIDIA 提供的高度手工调优闭源库，作为性能天花板参考
- **Baseline**：作者提供的未经优化的标准实现

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **总体表现**：CUDAMaster 在大多数算子上实现了显著加速
- **相比 Astra 平均提升约 35%**
- 多个任务上达到或超过闭源库性能：

| 算子 | CUDAMaster Speedup | 对应闭源库 | 表现 |
|------|--------------------|------------|------|
| **Dot Product** | **46.83x** | cuBLAS: 26.09x | **+79.6%** |
| **SpMV (CSR)** | 2.96x | cuSPARSE: 2.23x | **+32.7%** |
| **RMS Norm** | 2.44x | Astra: 1.78x | **+37.1%** |
| **Silu and Mul** | 3.67x | Astra: 3.58x | 接近持平 |
| **Conv 2D** | DeepSeek: 1.83x | cuDNN: 0.97x | 显著优于 cuDNN baseline |

> 注：部分结果中模型名称（如 o4-mini、DeepSeek）可能指代参与优化的 LLM 或 agent 子模块。

### 与基线方法的对比结果
- **全面超越 Astra**：在几乎所有任务上都取得更高 speedup，尤其在非 LLM 常见算子（如稀疏矩阵乘、科学计算）上优势明显
- **挑战闭源库权威**：在 Dot Product 等基础算子上大幅超越 cuBLAS，表明 LLM 驱动的系统已具备逼近专家级调优的能力
- **泛化能力强**：即使面对公开优化路径较少的科学计算 kernel，仍能生成高效实现

### 消融实验结果（文中未明确列出，但从架构设计可推断）
虽然论文未提供显式的消融研究表格，但从系统设计可推测以下关键因素的作用：
- **Profiling 数据过滤机制**：有效减少噪声干扰，提升优化方向准确性
- **Multi-Agent 分工**：分离 correctness debugging 与 performance tuning，避免上下文切换开销
- **End-to-end toolchain support**：提高实用性与部署效率，是迈向生产级系统的关键一步

---

## 4. 关键结论和发现

### 论文的主要发现
1. **LLM-based agents 能够胜任通用 CUDA kernel 优化任务**，不仅限于 LLM 相关算子。
2. **结合硬件 profiling 信息的引导式优化** 显著优于盲目的 prompt engineering 或简单反馈循环。
3. **多智能体协同架构** 更适合处理复杂的优化流程（实现 → 测试 → 分析 → 修改 → 验证）。
4. 在某些任务上，自动生成的 kernel **性能可达甚至超过 hand-tuned 闭源库**（如 cuBLAS），标志着自动化 GPU 编程进入新阶段。
5. 当前 LLM 的“回忆已知解”能力不足以解释其成功；在非主流算子上的优异表现说明其具备一定的**创造性优化能力**。

### 方法的局限性
- **高度依赖高质量 profiling 工具**（如 Nsight Compute），在其他硬件平台迁移性有待验证
- 当前 benchmark 仍基于 NVIDIA GPU，尚未扩展至 AMD ROCm 或其他架构
- 对 extremely irregular kernels（如动态图神经网络操作）支持有限
- 多 agent 协作带来额外通信与调度开销，小规模 kernel 优化性价比不高

### 未来工作方向
- 扩展至更多硬件平台（如 AMD、Intel GPU）
- 支持自动 kernel fusion 与 memory planning
- 探索轻量化 profiling 替代方案，降低对专用工具的依赖
- 将 CUDAMaster 集成进主流深度学习框架（如 PyTorch/TensorFlow）作为后端优化器
- 开源 MSKernelBench 与 CUDAMaster 框架，促进社区共建与标准化

> 🔗 **Demo 地址**：https://hanyx2021.github.io/MSKernelBenchDemo/  
> 📦 **开源承诺**：作者表示将开源 benchmark 与框架，推动 LLM 在系统优化领域的进一步发展。

</details>

---

### 12. [SafarDB: FPGA-Accelerated Distributed Transactions via Replicated Data Types](https://arxiv.org/abs/2603.08003)

**Authors**: Javad Saberlatibari, Prithviraj Yuvaraj, Mohsen Lesani, Philip Brisk, Mohammad Sadoghi  
**Category**: cs.DC  
**Published**: 2026-03-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.08003v1  

#### Abstract
Data replication is a critical aspect of data center design, as it ensures high availability, scalability, and fault tolerance. However, replicas need to be coordinated to maintain convergence and database integrity constraints under transactional workloads. Commutative Replicated Data Types (RDTs) ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*SafarDB: FPGA-Accelerated Distributed Transactions via Replicated Data Types*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

传统基于 CPU 和 RDMA 的分布式事务系统在处理 **Replicated Data Types (RDTs)** 时面临以下瓶颈：

- **高延迟**：PCIe 通信、主机内存中转、操作系统开销等导致 RDMA 性能受限。
- **低吞吐**：协调路径（尤其是强一致性场景下的共识协议）成为性能瓶颈。
- **故障恢复慢**：leader 切换依赖复杂的权限管理机制，受制于 PCIe 延迟。
- **缺乏对 FPGA 的深度协同设计**：现有方案多为 SmartNIC 架构，未将应用逻辑、网络栈与 FPGA 紧密集成。

### 🚀 提出的新方法与创新点

SafarDB 是首个将 **RDT 执行完全卸载到 network-attached FPGA** 上的系统，其核心创新包括：

#### （1）**FPGA 与 NIC 的协同架构设计（Co-location of RDT and NIC）**

- 将 RDT 应用逻辑、State Machine Replication (SMR) 协议、RDMA 网络栈全部部署在同一块 FPGA 芯片上。
- 消除 PCIe 通信开销，代之以轻量级的 on-chip AXI 协议，实现 near-network processing。

> **Design Principle #1**: Collocating application and RNIC on a single FPGA eliminates PCIe overhead.

#### （2）**定制化 RDMA Verbs 支持 FPGA 内部操作**

- 引入新的 **FPGA-specific RDMA RPC verbs**，支持直接调用 FPGA-resident accelerators。
- 实现 **RDMA Write-Through**：同时更新远程 HBM 日志和本地 BRAM 状态，避免 follower 重复读取日志。
  
> **Design Principle #2**: Custom RDMA RPCs eliminate spurious memory accesses.

#### （3）**混合执行模型（Hybrid Mode）**

- 支持数据分布在 FPGA 内存（HBM/BRAM）和主机内存之间。
- 统一的复制接口屏蔽存储层级差异，兼顾容量与性能。

#### （4）**极低延迟的权限切换机制（Fast Permission Switch）**

- FPGA 上的 SMR 模块可直接访问 QP 状态表（QPC），无需通过 PCIe 修改 RNIC 寄存器。
- 权限切换延迟从 **数百微秒降至 ~20ns**，显著加速 leader failover。

> **Design Principle #3**: Direct access to QPC reduces permission-switch latency to nanoseconds.

---

## 2. 核心实验方法和设置

### 📦 数据集与工作负载

实验涵盖两类 RDT 工作负载：

#### （1）**Microbenchmarks**

| 类型 | 具体实现 |
|------|--------|
| **CRDTs** | PN-Counter, LWW-Register, G-Set, PN-Set, 2P-Set |
| **WRDTs** | Bank Account, Courseware, Project, Movie, Auction |

#### （2）**标准数据库基准测试**
- **YCSB**：模拟键值存储负载，不同 PUT/GET 比例
- **SmallBank**：银行账户转账类事务，触发 SMR 协议

### ⚙️ 实验设置

| 项目 | 配置 |
|------|------|
| **硬件平台** | 8 × AMD Xilinx Alveo U280 FPGA，通过 100GbE 互联（Open Cloud Testbed） |
| **主机 CPU** | Intel Xeon Gold 622R @ 2.90GHz |
| **FPGA 开发工具** | Vivado 2023.2, Vitis HLS, StRoM RDMA stack |
| **软件端** | C++11 实现客户端与 CPU-resident 应用 |

### 🎯 评估指标

| 指标 | 定义 |
|------|------|
| **Response Time (μs)** | 客户端发出请求到收到响应的平均时间 |
| **Throughput (Ops/μs)** | 每微秒完成的操作数 |
| **Power Consumption (W)** | 整体功耗测量（含 FPGA/HBM/CPU/I/O） |

### 🔁 基线方法对比

| 基线 | 描述 |
|------|------|
| **Hamband [41]** | 当前最先进的软件 RDT 实现，基于 Mu 协议 + RDMA，运行于高端 CPU（Sapphire Rapids） |
| **Waverunner [5]** | FPGA 加速的 Raft 实现，代表当前最优的 SmartNIC 设计 |

---

## 3. 主要实验结果和性能指标

### 📈 性能提升（vs. Hamband）

| 工作负载 | 响应延迟降低 | 吞吐提升 |
|---------|-------------|----------|
| **CRDTs** | **7.0×** | **5.3×** |
| **WRDTs** | **12×** | **6.8×** |
| **YCSB + SmallBank** | **8×** | **5.2×** |

> 图 9–11 显示，在所有节点规模（3–8 节点）和写比例下，SafarDB 均大幅领先。

### 🆚 vs. Waverunner（YCSB, 3 nodes）

| 指标 | SafarDB 表现 |
|------|------------|
| **响应延迟** | **25.5× 更低** |
| **吞吐量** | **31.3× 更高** |

原因：
- Waverunner 仅加速 replication 路径，应用仍在 CPU；
- SafarDB 实现 **near-data processing**，且支持所有节点处理请求（非仅 leader）。

### 🔍 消融实验结果

#### （1）Custom RDMA RPC 的影响（图 6–8）

| 交易类型 | 优化效果 |
|--------|--------|
| **Reducible (e.g., PN-Counter)** | RPC 消除内存轮询 → 延迟 ↓8×，吞吐 ↑7.8× |
| **Irreducible (e.g., LWW-Register)** | 对 CRDT 提升有限（背景轮询已隐藏延迟）；对 WRDT 提升明显 |
| **Conflicting (e.g., Auction)** | Write-Through 避免日志回读 → 延迟 ↓1.5×，吞吐 ↑1.1× |

#### （2）Hybrid Mode 折衷分析（图 15–17）

| 因素 | 影响 |
|-----|-----|
| **更多操作由 FPGA 处理** | 延迟线性下降，吞吐上升（减少 PCIe 交互） |
| **Zipfian 分布（热点数据）** | 若热点在主机内存，CPU 缓存可缓解性能损失 |
| **Summarization（批处理）** | 可进一步降低延迟（↓4.9×）、提高吞吐（↑5×），但增加状态陈旧度（staleness） |

#### （3）故障容忍性（图 13–14）

| 故障类型 | SafarDB 表现 |
|--------|------------|
| **Permission Switch 延迟** | **17–24ns**（vs. Hamband 的数百 μs） |
| **Follower Crash** | 响应时间几乎不变，吞吐仅降 2%（vs. Hamband ↓30%） |
| **Leader Crash** | 响应时间 ↑25%（vs. Hamband ↑40%），吞吐仅降 15%（vs. ↓40%） |

> 得益于快速权限切换与后台心跳检测。

### 💡 功耗表现（图 27）

| 系统 | 平均功耗 |
|------|--------|
| **SafarDB** | **~35W** |
| **Hamband** | **~160W** |

👉 **SafarDB 功耗仅为 Hamband 的 1/4.5，同时性能更高**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **FPGA 可作为高性能 RDT 执行平台**  
   通过将 RDT、SMR、RDMA 栈集成于 FPGA，可实现 **微秒级甚至纳秒级** 的协调控制路径。

2. **近网计算（Near-Network Processing）是关键**  
   消除 PCIe 和主机内存中转，使 RDMA 延迟降低两个数量级（见 Table 2.1：从 2μs → 9ns）。

3. **定制化 RDMA verbs 显著提升效率**  
   FPGA-specific RPC 和 Write-Through 有效消除冗余内存访问，尤其在冲突事务中优势明显。

4. **故障恢复速度远超传统系统**  
   权限切换进入纳秒级，使 leader failover 更快更稳定。

5. **Hybrid 模式提供灵活扩展能力**  
   冷数据放主机，热数据留 FPGA，结合批处理（summarization）可在性能与容量间取得平衡。

### ⚠️ 局限性

- **FPGA 内存有限**：尽管有 Hybrid 模式，大规模状态仍受限于 HBM 容量（本实验使用 8GB）。
- **开发复杂度高**：需使用 HLS/Vivado 进行硬件编程，门槛高于纯软件系统。
- **仅支持 crash-fail 模型**：未考虑拜占庭容错（Byzantine faults）。
- **扩展性测试受限**：最大仅测试 8 节点，更大规模集群的表现未知。

### 🔮 未来工作方向

1. **支持更多一致性协议**：如 Paxos、PBFT 或 BFT-SMaRt。
2. **探索 disaggregated FPGA 架构**：多个服务器共享一个高性能 FPGA 加速器。
3. **自动代码生成框架**：从高级语言（如 SQL 或 Rust）自动生成 FPGA-accelerated RDT 模块。
4. **支持动态 reconfiguration**：根据负载变化在线切换 FPGA 功能模块。
5. **集成 AI 推理能力**：构建“智能数据库”，在 FPGA 上联合执行事务与分析任务。

---

## 总结

> **SafarDB 证明了：通过将 RDT 与 RDMA NIC 在 network-attached FPGA 上深度融合，可以突破传统 CPU/RDMA 架构的性能天花板，在延迟、吞吐、功耗和故障恢复方面全面超越现有系统。这为下一代高性能、高可用分布式数据库提供了全新的硬件协同设计范式。**

</details>

---

### 13. [Airborne Magnetic Anomaly Navigation with Neural-Network-Augmented Online Calibration](https://arxiv.org/abs/2603.08265)

**Authors**: Antonia Hager, Sven Nebendahl, Alexej Klushyn, Jasper Krauser, Torleiv H. Bryne, Tor Arne Johansen  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.08265v1  

#### Abstract
Airborne Magnetic Anomaly Navigation (MagNav) provides a jamming-resistant and robust alternative to satellite navigation but requires the real-time compensation of the aircraft platform's large and dynamic magnetic interference. State-of-the-art solutions often rely on extensive offline calibration...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Airborne Magnetic Anomaly Navigation with Neural-Network-Augmented Online Calibration

---

## 1. 论文的主要贡献和创新点

### 解决的问题
**Airborne Magnetic Anomaly Navigation (MagNav)** 是一种利用地壳磁场异常进行定位的导航技术，具有抗干扰、不依赖 GNSS 的优势。然而，其核心挑战在于飞机平台自身产生的强磁干扰（可达数千纳特斯拉），远超用于导航的地磁异常信号（仅数十至数百纳特斯拉）。传统方法依赖离线校准飞行（offline calibration flights）来估计 Tolles-Lawson (TL) 模型参数，这在实际部署中成本高、灵活性差。

此外，TL 模型基于刚体假设，无法建模非线性动态干扰（如电气系统开关、振动等），而现有的机器学习（ML）方法虽能提升性能，但通常需要大量训练数据和预训练阶段，难以实现实时在线自适应。

### 提出的新方法与创新思路
本文提出了一种**完全自适应的混合式在线校准与导航架构**，核心创新如下：

- **冷启动能力（Cold-Start Capability）**  
  首次实现无需任何先验知识或校准飞行的“冷启动”MagNav系统。模型可在飞行过程中自主识别并补偿飞机磁特征，极大提升了操作可行性。

- **神经网络增强的扩展卡尔曼滤波器（NN-augmented EKF）**  
  将神经网络（NN）的权重和偏置作为状态向量的一部分，集成到 EKF 中，与飞行器运动状态、TL 模型参数共同进行联合估计。该框架实现了：
  - 实时在线学习（real-time online learning）
  - 无须缓冲数据或反向传播通过时间（BPTT）

- **自然梯度下降的数学等价性**  
  证明 EKF 更新在数学上等价于对 NN 参数执行**在线自然梯度（Natural Gradient, NG）下降**，具备二阶优化特性，收敛更快、更稳定，优于标准一阶梯度下降。

- **残差学习架构（Residual Learning Role）**  
  NN 仅用于建模 TL 模型未能捕捉的**残余非线性干扰**，而非直接替代物理模型。这种设计增强了系统的可解释性和鲁棒性，防止过拟合和发散。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 是否需要校准飞行 | ✅ 是（如 [8]） | ❌ 否（支持 cold start） |
| 是否支持在线自适应 | ⚠️ 有限（多为离线） | ✅ 完全在线实时更新 |
| 学习效率 | 低（需批量训练） | 高（天然支持在线 NG） |
| 可解释性 | 高（纯物理模型） | 高（NN 仅为残差修正） |
| 硬件兼容性 | 高 | 高（浅层 NN + 全协方差更新可行） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MagNav Challenge Dataset** [38]：由 DAF-MIT 开放的真实飞行数据集。
- 主要分析 **flight line 1007.06**（约 87 分钟），补充验证 **flight line 1003**（约 4.15 小时）。
- 数据包含多个磁力计（Fluxgate 和 Scalar Magnetometers）测量值、IMU 数据、GNSS 真值轨迹及地磁图。

### 实验设置
#### 初始化模式
- **Cold Start (C)**：所有参数初始化为零或随机，协方差初值大（`P₀ = diag(I, 1000I)`），表示无先验知识。
- **Warm Start (W)**：使用前一次运行的结果初始化，模拟已有历史飞行记忆。

#### 模型架构
- **TL 模型**：18 参数经典航空磁补偿模型。
- **NN 架构**：单隐藏层全连接网络，激活函数为 `tanh`，输出层无线性激活。
  - 隐藏神经元数 `Nₕ ∈ {2, ..., 128}`
  - 输入特征：仅来自磁力计（magnetometer-only feature set），未引入额外传感器或手工特征（如导数）。
- **输出缩放因子 α = 400 nT**，限制 NN 输出范围，强化其“残差修正”角色。

#### 评估指标
- **定位精度**：水平位置 RMSE（即 DRMS）：
  $$
  \text{DRMS} = \sqrt{\frac{1}{N}\sum_{i=1}^N \left[(x^\text{GNSS}_i - x^\text{MagNav}_i)^2 + (y^\text{GNSS}_i - y^\text{MagNav}_i)^2\right]}
  $$
- **校准误差**：磁干扰预测 RMSE：
  $$
  \text{RMSE}_m = \sqrt{\frac{1}{N}\sum_{i=1}^N (m_i - h(\mathbf{x}_{t|t-1}))^2}
  $$

### 基线方法对比
- **TL-only Online Calibration**：仅使用 TL 模型进行在线校准（类似 [7]）。
- **Gnadt [8] 的 “Online Model 2c”**：使用相同特征集但需 12 小时飞行数据预训练 TL 和 NN 参数。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（flight 1007.06, cold start, Nₕ=5）

| Magnetometer | 干扰强度 (uncompensated ~nT) | 本文方法 DRMS (m) | TL-only DRMS (m) | 提升幅度 |
|-------------|-------------------------------|------------------|------------------|----------|
| Mag 2       | ~1250                         | 51               | 107              | ~52% ↓   |
| Mag 3       | ~1150                         | 42               | 46               | ~9% ↓    |
| Mag 4       | ~250                          | 37               | 58               | ~36% ↓   |
| Mag 5       | ~100                          | 14               | 15               | ~7% ↓    |

> 注：Mag 1 被视为近似“真值”，其上两种方法表现接近（DRMS ≈ 17m）。

### 与基线方法对比结果
- 在 **无需任何预训练或校准飞行** 的前提下，本文方法在 Mag 3/4/5 上达到甚至优于 **Gnadt [8]** 报告的性能（分别为 32m, 37m, 18m），尤其在噪声极大的 Mag 2 上仍保持有效。
- 表明所提方法在**操作灵活性**大幅提升的同时，未牺牲关键性能。

### 消融实验与关键观察
#### 不同隐藏层大小的影响
- **趋势**：增加 `Nₕ` 可略微降低 `RMSE_m`，但对 DRMS 改善不显著，甚至出现波动。
- **原因分析**（见 Appendix D）：
  - 过大的 NN 会稀释系统可观测性（observability），导致导航状态估计不确定性上升。
  - Cramér-Rao Lower Bound (CRLB) 分析显示，复杂模型在信息不足时反而提高理论误差下限。
- **结论**：推荐使用极简网络（`Nₕ ≈ 2–5`），避免“吞噬”导航信号。

#### 冷启动 vs 温启动
- Warm start 在高干扰磁力计（Mag 2/3）上有轻微优势（DRMS ↓ ~3–6m），但在低干扰下差异不大。
- Cold start 初始阶段存在约 **50–100 km 的收敛期**，之后性能趋于稳定。

#### 创新拒绝机制
- 对归一化创新平方超过 χ² 阈值（6）的测量进行剔除，有效抑制异常值影响，提升稳定性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **冷启动可行**：首次实现了无需离线校准飞行的 MagNav 系统，在真实飞行数据上验证了从零开始的自适应能力。
2. ✅ **EKF ≡ 在线自然梯度**：将 NN 参数纳入 EKF 状态向量，本质上实现了几何感知的二阶优化，收敛快且数据高效。
3. ✅ **残差学习优于端到端建模**：将 NN 限定为 TL 模型的残差修正项，既保留了物理模型的可解释性，又增强了非线性建模能力。
4. ✅ **轻量化设计即可满足需求**：浅层 NN + 磁力计单一特征输入已足够实现导航级性能，适合嵌入式部署。

### 方法的局限性
- **初始收敛期较长**：冷启动时需约 50–100 km 飞行距离才能充分学习，期间定位精度较低。
- **依赖高质量地磁图**：若地图存在误差或缺失区域，会影响整体性能。
- **NN 饱和风险**：当前 `tanh` 激活可能导致梯度消失，影响尖锐干扰的学习（见 Appendix F）。
- **缺乏形式化收敛证明**：尽管实践中稳定，但尚未建立严格的数学收敛保障。

### 未来工作方向
- **解耦/联邦架构**：采用 Partial-Update Schmidt-Kalman Filter 或 Federated Learning 架构，防止参数更新扰动主导航状态。
- **Moving Horizon Estimation (MHE)**：提升对地图伪影和突发干扰的鲁棒性。
- **自适应学习率调控**：基于协方差演化自动判断“校准成熟度”，实现从临时解到高完整性解的平滑过渡。
- **故障检测与排除（FDE）机制**：区分模型学习与传感器故障，提升可信度。
- **非饱和激活函数探索**：尝试 Mish、SiLU、GELU 等缓解 NN 饱和问题。
- **量子磁力计融合**：结合量子传感技术以进一步提升底层测量质量与安全性。

---

> **总结一句话**：  
> 本论文提出了一种**无需预训练、支持冷启动、可解释性强且硬件友好的 MagNav 新架构**，通过将 NN 参数嵌入 EKF 实现了在线自然梯度学习，在真实飞行数据上达到了与需大量离线训练的方法相当的性能，为 MagNav 的实用化部署扫清了关键障碍。

</details>

---

### 14. [PolyFormer: learning efficient reformulations for scalable optimization under complex physical constraints](https://arxiv.org/abs/2603.08283)

**Authors**: Yilin Wen, Yi Guo, Bo Zhao, Wei Qi, Zechun Hu, Colin Jones, Jian Sun  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.08283v1  

#### Abstract
Real-world optimization problems are often constrained by complex physical laws that limit computational scalability. These constraints are inherently tied to complex regions, and thus learning models that incorporate physical and geometric knowledge, i.e., physics-informed machine learning (PIML), ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PolyFormer: learning efficient reformulations for scalable optimization under complex physical constraints

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现实世界中的优化问题通常受到**复杂物理约束**（complex physical constraints）的限制，这些约束源于大规模实体、系统间依赖关系以及不确定性。传统方法在处理这类高维、非线性、混合整数优化问题时面临严重的**可扩展性瓶颈**，表现为计算时间长、内存消耗大，难以满足实时决策需求。

例如：
- 大规模资源聚合中个体设备的异构约束；
- 电网等网络系统中的非线性潮流方程；
- 不确定环境下的分布鲁棒机会约束（DRCC）引入大量辅助变量。

这些问题导致优化求解器（如 IPOPT）在实际应用中效率低下，甚至无法求解。

---

### 提出的新方法与新思路

作者提出 **PolyFormer** —— 一种全新的 **Physics-Informed Machine Learning (PIML)** 框架，用于**可扩展优化**（scalable optimization），其核心思想是：

> **不直接预测解或加速求解器，而是学习将原始复杂可行域 $\Omega$ 近似为一个紧凑的多面体表示 $P(A,b) = \{x | Ax \leq b\}$，从而对原问题进行高效重构（reformulation）。**

#### 创新点包括：

1. **从“预测”到“问题简化”的范式转变**  
   与大多数 PIML 工作聚焦于**预测任务**不同，PolyFormer 首次成功将 PIML 应用于**规定性优化任务**（prescriptive optimization），通过几何建模实现问题本身的简化。

2. **显式可微损失函数设计**  
   定义了两个互补误差度量：
   - **可行性误差**（feasibility error）：衡量近似区域是否超出真实可行域（避免不可行解）；
   - **最优性误差**（optimality error）：衡量是否遗漏了原区域中的最优解。
   
   并基于方向采样构建期望形式的误差指标，进而推导出**显式可微的损失函数**，支持自动微分训练。

3. **可控的近似权衡机制**  
   引入权重系数 $\lambda \in [0,1]$ 控制内近似（inner approximation, $\lambda=1$）与外近似之间的平衡，适应不同应用场景的需求（如优先保证可行性）。

4. **参数化建模以应对动态变化**  
   使用神经网络（A-net 和 b-net）将矩阵 $A(\theta)$ 和向量 $b(\theta)$ 参数化为外部参数 $\theta$ 的函数，使模型能快速适应运行条件变化（如电压波动、风险偏好调整），无需重新训练。

---

### 相比现有方法的优势

| 维度 | 传统方法 | PolyFormer |
|------|--------|-----------|
| **分析方法**（如凸松弛、分解） | 需要问题特定推导，泛化差，收敛不稳定 | 自动学习通用多面体重构，适用于多种约束类型 |
| **机器学习方法**（如端到端预测） | 缺乏可行性保障，解释性差 | 输出为标准线性约束，兼容任意 off-the-shelf solver |
| **求解器加速方法**（如 warm start） | 仍需存储完整模型，最坏情况耗时仍高 | 显著压缩模型规模，从根本上降低复杂度 |
| **通用性** | 多为专用架构 | 统一框架处理三类典型挑战：大规模、网络依赖、不确定性 |

---

## 2. 核心实验方法和设置

### 数据集与场景

PolyFormer 在三个代表性领域进行了验证，覆盖了现实优化中的三大挑战：

| 场景 | 描述 | 来源 |
|------|------|------|
| **大规模资源聚合** | 聚合 1,000 台 EV/BSS/HP 设备的可行功率轨迹 | 自定义仿真模型（见 Supplementary Note 3） |
| **网络约束优化** | 两层电力系统：输电 + 多个配电网络 | IEEE 标准数据集 + 浙江省真实配网数据（共 27 个案例） |
| **不确定性下的优化** | 分布鲁棒机会约束投资组合优化（DRCC Portfolio Optimization） | 合成金融数据（资产收益服从多元正态分布） |

---

### 实验设置与评估指标

#### 共同流程：
1. 使用 PolyFormer 学习原始复杂可行域 $\Omega$ 的多面体近似 $P(A,b)$；
2. 将 $Ax \leq b$ 替换原约束嵌入优化问题；
3. 使用标准求解器（如 Gurobi、IPOPT）求解简化后的问题；
4. 对比原始模型与简化模型在**解质量**和**计算效率**上的差异。

#### 主要评估指标：

| 指标 | 定义 |
|------|------|
| **Solver Time** | 求解时间（秒） |
| **Peak Memory Usage** | 内存峰值占用（MB） |
| **Feasibility Error** | 简化解违反原始约束的程度（归一化最大偏差） |
| **Objective Error** | 简化解的目标值相对误差（vs. 原始最优） |
| **Constraint / Variable Count** | 模型大小对比 |
| **Out-of-Sample Risk-Return Performance** | 在测试集上评估投资组合的实际表现 |

---

### 基线方法对比

| 场景 | 基线方法 |
|------|---------|
| 资源聚合 | Full Model（精确但庞大）、Box Method、Homothet Method |
| 网络优化 | IPOPT 直接求解全耦合非线性 AC-OPF 模型 |
| 投资组合优化 | DRCC-linear：基于 Wasserstein 距离的经典线性重构方法 [62] |

---

## 3. 主要实验结果和性能指标

### （1）大规模资源聚合

| 指标 | 结果 |
|------|------|
| **聚合 1,000 个连续控制资源** | |
| - 约束数量 | 从 57,696 → **96**（减少 99.83%） |
| - 最优性误差 | ~21% of Box 方法（更紧致） |
| - 可行性误差 | < 10⁻⁵（几乎完全可行） |
| - Homothet 方法 | 因内存溢出未能收敛 |
| **聚合 10⁵ 个混合控制资源**（含离散动作） | |
| - 消除所有 336 个二元变量 | ✅ |
| - 连续变量数 | 从 5,184 → **24**（↓99.54%） |
| - 约束数 | 从 6,291 → **96**（↓98.47%） |
| - Box/Homothet 方法 | ❌ 不适用（仅支持连续变量） |

✅ **PolyFormer 是首个能处理混合整数资源聚合的方法。**

---

### （2）网络约束优化（两层电力系统）

| 指标 | 结果 |
|------|------|
| **最大系统规模** | 715,055 constraints → **2,239**；477,691 vars → **1,785** |
| **求解速度提升** | 最高达 **6,400× 加速**（1,476 s → 0.23 s） |
| **内存减少** | **99.6%**（821 MB → 3.5 MB） |
| **可行性误差**（max） | ↓至 **6.4×10⁻¹⁰**（当 $\lambda=0.99$） |
| **目标误差**（avg） | 7.2×10⁻⁴（可接受范围内） |

> ⚖️ 存在明显的 **feasibility-optimality trade-off**：提高 $\lambda$ 可显著改善可行性，代价是轻微增加目标误差。

---

### （3）不确定性下的投资组合优化（DRCC）

| 指标 | 结果 |
|------|------|
| **问题规模压缩** | 最多达 **1,044,497 constraints → 1,617**（↓99.85%） |
| **变量数** | 1,034,656 → **400**（↓99.96%） |
| **求解时间** | 513 s → **0.725 s**（**708× 更快**） |
| **内存消耗** | 938 MB → **1.2 MB**（↓99.87%） |
| **解质量** | |
| - 平均回报率 | 更高或相当 |
| - 平均约束违反 | **更低** |
| - Pareto frontier | PolyFormer 支配 DRCC-linear（即同时更好） |

💡 **原因分析**：PolyFormer 隐式学习分布特征，而 DRCC-linear 依赖样本经验分布，在测试集上适应性较差。

---

### 消融实验（见 Methods 中 Typical Geometries）

| 设置 | 发现 |
|------|------|
| **Hypercube 近似** | 在 $n=2$ 到 $200$ 维下均能收敛到接近精确解（误差 < 10⁻⁵） |
| **Hypersphere 近似** | 达到 >99.4% 的误差缩减率，$n>30$ 时超过 99.9% |
| **Nonconvex Region** | 趋向于拟合凸包（convex hull），适合多数工程应用；若需更强可行性，可通过改进采样策略增强 |

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **PolyFormer 成功开辟了 PIML 在 prescriptive optimization 中的新路径**：不再局限于预测，而是用于**自动简化优化问题结构**。
2. ✅ **多面体重构能极大提升可扩展性**：在三大典型场景中均实现 **数百至数千倍的速度提升** 和 **99%以上的内存节省**。
3. ✅ **保持高质量解**：尽管进行了简化，但在关键指标（目标值、可行性）上优于或媲美 state-of-the-art 方法。
4. ✅ **具备强泛化能力**：不仅适用于物理系统（如电网），也适用于金融等非物理领域，只要存在几何结构即可建模。
5. ✅ **支持快速重配置**：参数化版本允许通过改变输入 $\theta$ 快速生成新约束，适用于动态环境。

---

### 方法的局限性

1. **对非凸可行域的逼近限于凸包**  
   当 $\Omega$ 高度非凸时，PolyFormer 倾向于学习其 convex hull，可能导致过度保守。未来可通过引入更多拓扑感知采样策略改进。

2. **依赖于可高效求解子问题的能力**  
   训练过程中需要频繁求解支撑点（support point）问题（如 $\max v^Tx$ over $\Omega$），若原始区域本身极难优化，则训练成本较高。

3. **超参数选择影响性能**  
   如 $M$（超平面数量）、$\lambda$、归一化方式等需合理设置，否则可能影响收敛性和精度。

---

### 未来工作方向

1. **扩展至更高阶几何表示**  
   探索使用椭球、锥或其他非线性形式进行近似，以更好地捕捉非凸结构。

2. **结合 adaptive sampling 策略**  
   动态关注误差较大的方向或区域，提升训练效率与逼近精度。

3. **集成进端到端优化 pipeline**  
   与 reinforcement learning 或 online optimization 框架结合，实现实时自适应重构。

4. **应用于更多跨学科场景**  
   如机器人运动规划、供应链调度、气候模型控制等涉及复杂约束的领域。

---

### 总结

> **PolyFormer 是一项突破性的 PIML 框架，它将复杂的物理约束转化为简洁的 polytopic reformulation，实现了“解前简化”而非“解中加速”，为大规模、高维、受物理规律约束的优化问题提供了高效、可靠且通用的解决方案。**

🔗 **代码已开源**：https://github.com/wenyl16/PolyFormer  
📄 **补充材料**：提供详细的 workflow 示例与实现指南（Supplementary Notes 1–5）

</details>

---

### 15. [DualFlexKAN: Dual-stage Kolmogorov-Arnold Networks with Independent Function Control](https://arxiv.org/abs/2603.08583)

**Authors**: Andr\'es Ortiz, Nicol\'as J. Gallego-Molina, Carmen Jim\'enez-Mesa, Juan M. G\'orriz, Javier Ram\'irez  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.08583v1  

#### Abstract
Multi-Layer Perceptrons (MLPs) rely on pre-defined, fixed activation functions, imposing a static inductive bias that forces the network to approximate complex topologies solely through increased depth and width. Kolmogorov-Arnold Networks (KANs) address this limitation through edge-centric learnabl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DualFlexKAN: Dual-stage Kolmogorov-Arnold Networks with Independent Function Control

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 **Multi-Layer Perceptrons (MLPs)** 使用固定的激活函数（如 ReLU），其非线性表达能力受限于网络宽度和深度，难以高效逼近复杂拓扑结构。而新兴的 **Kolmogorov-Arnold Networks (KANs)** 虽通过边上的可学习单变量函数提升了表达力，但存在以下关键缺陷：
- **参数爆炸**：每条边独立学习函数导致参数量呈 $O(n_{in} \cdot n_{out})$ 增长；
- **训练不稳定**：函数与权重联合优化困难；
- **架构僵化**：缺乏对不同层、不同阶段灵活性的控制；
- **正则化难集成**：Dropout 和 BatchNorm 在边中心架构中难以有效应用。

### 提出了什么新方法或新思路
本文提出 **DualFlexKAN (DFKAN)**，一种具有双阶段机制的灵活神经网络架构，核心思想是将非线性变换解耦为两个独立可控的阶段：
- **Pre-linear 输入变换（Input Transformation）**
- **Post-linear 输出激活（Output Activation）**

该设计实现了：
- **独立控制**：输入和输出阶段可分别配置不同的函数共享策略；
- **混合架构支持**：可在同一网络中组合 MLP 风格（固定激活）、KAN 风格（可学习函数）及中间形态；
- **多类基函数支持**：支持 B-splines、正交多项式（Legendre, Chebyshev, Gegenbauer, Jacobi）、Radial Basis Functions、Sine/Spectral、Wavelets 等；
- **灵活正则化框架**：支持在 pre-act 或 post-act 位置插入 Dropout 和 Batch Normalization，并可调节顺序。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **参数效率** | 参数量比标准 KAN 减少 **1–2 个数量级**，接近 MLP 规模； |
| **表达力保留** | 保持 KAN 式的高表达能力，尤其擅长建模物理规律中的乘法、除法、根号等光滑流形； |
| **训练稳定性** | 通过结构正则化（如全局共享函数）抑制过拟合，在低数据场景下更鲁棒； |
| **可解释性增强** | 可直接可视化 learned functions，用于符号回归和物理定律发现； |
| **生物合理性** | 架构模拟生物神经元的树突计算（dendritic computation）与胞体整合（somatic integration）分离机制； |
| **模块化设计** | 支持逐层定制策略，实现“早期高表达 + 后期稳定”的层级适应性。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖三大类任务，共 **14 个基准数据集**：

#### （1）物理启发与符号回归任务（Physics-Informed & Symbolic）
| 数据集 | 特点 |
|-------|------|
| **Friedman #1, #2** | 包含强非线性交互项（如 $\sqrt{x_1^2 + x_2^{-1}}$），测试模型捕捉复杂数学关系的能力 |
| **Feynman Equations (I.18.12, II.6.11)** | 来自费曼物理学讲义的真实公式，涉及电磁力、电势等（如 $x_1 x_2^3 \sin(x_4)$） |
| **Franke Function** | 二维曲面拟合任务，常用于插值评价 |

#### （2）合成高频与复合函数（Compositional & High-Frequency）
| 数据集 | 定义 |
|--------|-----|
| **Damped Oscillator** | $y = e^{-t} \sin(\omega t + \delta)$，测试同时建模指数衰减与高频振荡的能力 |
| **Sin_Exp / Nested Trig** | $y = \sin(\exp(x))$, $y = \sin(\cos(\sin(x)))$，挑战谱偏置（spectral bias）问题 |

#### （3）真实世界回归任务（Real-World Tabular）
来自 UCI 和 OpenML 的小样本回归数据集：
- **Yacht Hydrodynamics**（流体力学）
- **Servo**（机械系统）
- **Boston Housing**
- **Auto MPG**
- **Diabetes**

> 所有数据集样本数均小于 5000，部分低于 200，强调在有限数据下的泛化能力。

### 实验设置和评估指标

| 设置项 | 描述 |
|--------|------|
| **评估指标** | - MSE（Mean Squared Error）<br>- R² Score<br>- Effective Parameter Count（剪枝后维持 90% 性能所需的最小参数数）<br>- Training Time（秒） |
| **训练细节** | - 使用 Adam 优化器<br>- He 初始化用于线性权重<br>- 多项式系数采用方差衰减初始化<br>- 所有模型进行超参调优以公平比较 |
| **消融配置** | 测试多种策略组合：<br>- 输入策略：None (S0), Fixed (S1), Global (S2), Per-Dim (S3), Per-Connection (S4)<br>- 输出策略：S0–S3 |

### 基线方法对比
- **MLP**：标准全连接网络，使用 ReLU/Tanh 激活；
- **Vanilla KAN**：原始 KAN 实现，基于 B-splines，每条边独立学习函数；
- **DualFlexKAN**：本文提出的方法，典型配置为：首层使用 S4（Per-Connection）提取特征，后续层使用 S2/S3 控制复杂度。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | DualFlexKAN 表现 |
|------|------------------|
| **平均参数量** | 比 Vanilla KAN 少 **1–2 个数量级**（见图4） |
| **训练时间** | 显著低于 Vanilla KAN，接近 MLP 水平（见图5） |
| **Effective Parameters** | 中位数仅 **93**，远低于 MLP（6721）和 Classic KAN（281）（见图6） |
| **MSE on Physics Tasks** | 在 Feynman I.18.12 上达 $2.9 \times 10^{-4}$，优于 MLP 和 Classic KAN |
| **Gradient Fidelity** | 成功重建 $z = \sin(2x)\cos(2y)$ 的梯度场，MSE 仅为 $2.9 \times 10^{-4}$，而 Classic KAN 失败（MSE > 0.25） |

### 与基线方法的对比结果

| 对比维度 | 结果总结 |
|---------|----------|
| **vs. MLP** | - 在物理/数学结构任务上显著胜出（更低 MSE，更高 R²）<br>- 更好地恢复光滑流形和高频梯度<br>- 具备内置可解释性（无需 SHAP/LIME） |
| **vs. Vanilla KAN** | - 参数量减少 10–100 倍<br>- 训练更快更稳定<br>- 泛化更强，尤其在噪声环境下不易过拟合<br>- 支持标准正则化技术（Dropout/BatchNorm） |

### 消融实验结果（Ablation Study）
- **策略选择影响显著**：
  - 使用 `S4`（Per-Connection）输入策略在初期层能更好捕获高阶交互；
  - 后续层切换至 `S2`（Global Shared）可大幅压缩参数而不损失精度；
- **基函数选择重要**：
  - **Legendre Polynomials** 在高频任务（如 Damped Oscillator）表现最优；
  - **B-splines** 更适合局部平滑变化；
  - **Sine Basis** 对周期性强的任务有优势；
- **正则化位置影响训练动态**：
  - Pre-activation BN 提升训练稳定性；
  - Dropout-first 序列有助于防止 co-adaptation。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **双阶段解耦设计有效打破参数瓶颈**：  
   DFKAN 成功解决了 Vanilla KAN 的参数爆炸问题，使 KAN 类模型具备实际部署可行性。

2. **兼具表达力与效率的“黄金平衡”**：  
   通过混合策略（hybrid configuration），DFKAN 在保持 KAN 高表达力的同时，达到接近 MLP 的参数效率。

3. **天然具备科学发现潜力**：  
   - 可视化 learned functions 可揭示潜在物理规律；
   - 在噪声条件下仍能恢复干净的符号公式（如 $y \approx 2x^2 - x + 0.5$）；
   - 内置 attention 机制实现 feature selection，无需后处理解释工具。

4. **克服经典架构局限**：
   - 解决了 MLP 的 **spectral bias**（偏好低频）；
   - 克服了 Vanilla KAN 的 **additive bottleneck**（难以建模乘法交互）；
   - 通过深度堆叠 node-centric 层次结构，准确建模复杂微分拓扑。

### 方法的局限性
- **超参数敏感性增加**：由于策略、基函数、正则化位置均可配置，搜索空间变大，需更多调参成本；
- **在纯黑盒表格数据上未必超越 MLP**：对于无明确数学结构的任务（如 Boston Housing），MLP 仍可能取得略优预测性能；
- **牺牲部分可解释性换取效率**：当使用 global/shared 函数时，丢失了 per-edge 的细粒度解释能力；
- **目前主要面向回归任务**，尚未验证在分类、CV/NLP 中的表现。

### 未来工作方向
1. **自动化架构搜索（NAS）**：开发算法自动选择最优策略组合（S_in, S_out, basis, reg_pos）；
2. **理论分析深化**：形式化证明 DFKAN 的逼近能力与收敛性质；
3. **扩展至其他领域**：
   - **Computer Vision**：应用于图像算子学习；
   - **Natural Language Processing**：探索 token-level 函数学习；
4. **神经生物学交叉研究**：进一步验证 learned functions 是否反映真实神经编码模式；
5. **边缘部署优化**：结合 TinyML 技术，推动 DFKAN 在 Edge AI 场景落地。

---

> ✅ **代码开源地址**：https://github.com/BioSIP/dfkan  
> 📚 **推荐应用场景**：Physics-Informed Neural Networks (PINNs)、AI for Science (AI4Science)、Symbolic Regression、Low-data Scientific Modeling。

</details>

---

### 16. [Animating Petascale Time-varying Data on Commodity Hardware with LLM-assisted Scripting](https://arxiv.org/abs/2603.07053)

**Authors**: Ishrat Jahan Eliza, Xuan Huang, Aashish Panta, Alper Sahistan, Zhimin Li, Amy A. Gooch, Valerio Pascucci  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.07053v1  

#### Abstract
Scientists face significant visualization challenges as time-varying datasets grow in speed and volume, often requiring specialized infrastructure and expertise to handle massive datasets. Petascale climate models generated in NASA laboratories require a dedicated group of graphics and media experts...

---

### 17. [Adaptive Collaboration with Humans: Metacognitive Policy Optimization for Multi-Agent LLMs with Continual Learning](https://arxiv.org/abs/2603.07972)

**Authors**: Wei Yang, Defu Cao, Jiacheng Pang, Muyan Weng, Yan Liu  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.07972v1  

#### Abstract
While scaling individual Large Language Models (LLMs) has delivered remarkable progress, the next frontier lies in scaling collaboration through multi-agent systems (MAS). However, purely autonomous MAS remain ''closed-world'' systems, constrained by the static knowledge horizon of pre-trained model...

---

### 18. [CDRRM: Contrast-Driven Rubric Generation for Reliable and Interpretable Reward Modeling](https://arxiv.org/abs/2603.08035)

**Authors**: Dengcan Liu, Fengkai Yang, Xiaohan Wang, Shurui Yan, Jiajun Chai, Jiahao Li, Yikun Ban, Zhendong Mao, Wei Lin, Guojun Yin  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.08035v1  

#### Abstract
Reward modeling is essential for aligning Large Language Models(LLMs) with human preferences, yet conventional reward models suffer from poor interpretability and heavy reliance on costly expert annotations. While recent rubric-based approaches enhance evaluation transparency, they lack systematic q...

---

### 19. [Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](https://arxiv.org/abs/2603.07475)

**Authors**: Raghavv Goel, Risheek Garrepalli, Sudhanshu Agrawal, Chris Lott, Mingu Lee, Fatih Porikli  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.07475v1  

#### Abstract
Autoregressive (AR) language models form representations incrementally through left-to-right prediction, whereas diffusion language models (dLLMs) are trained via full-sequence denoising. Although recent dLLMs match AR performance, it remains unclear whether diffusion objectives fundamentally reshap...

---

### 20. [TableMind++: An Uncertainty-Aware Programmatic Agent for Tool-Augmented Table Reasoning](https://arxiv.org/abs/2603.07528)

**Authors**: Mingyue Cheng, Shuo Yu, Chuang Jiang, Xiaoyu Tao, Qingyang Mao, Jie Ouyang, Qi Liu, Enhong Chen  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.07528v1  

#### Abstract
Table reasoning requires models to jointly perform semantic understanding and precise numerical operations. Most existing methods rely on a single-turn reasoning paradigm over tables which suffers from context overflow and weak numerical sensitivity. To address these limitations, we previously propo...

---

### 21. [Reward Under Attack: Analyzing the Robustness and Hackability of Process Reward Models](https://arxiv.org/abs/2603.06621)

**Authors**: Rishabh Tiwari, Aditya Tomar, Udbhav Bamba, Monishwaran Maheswaran, Heng Yang, Michael W. Mahoney, Kurt Keutzer, Amir Gholami  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.06621v1  

#### Abstract
Process Reward Models (PRMs) are rapidly becoming the backbone of LLM reasoning pipelines, yet we demonstrate that state-of-the-art PRMs are systematically exploitable under adversarial optimization pressure. To address this, we introduce a three-tiered diagnostic framework that applies increasing a...

---

### 22. [TS-MLLM: A Multi-Modal Large Language Model-based Framework for Industrial Time-Series Big Data Analysis](https://arxiv.org/abs/2603.07572)

**Authors**: Haiteng Wang, Yikang Li, Yunfei Zhu, Jingheng Yan, Lei Ren, Laurence T. Yang  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.07572v1  

#### Abstract
Accurate analysis of industrial time-series big data is critical for the Prognostics and Health Management (PHM) of industrial equipment. While recent advancements in Large Language Models (LLMs) have shown promise in time-series analysis, existing methods typically focus on single-modality adaptati...

---

### 23. [Stabilized Fine-Tuning with LoRA in Federated Learning: Mitigating the Side Effect of Client Size and Rank via the Scaling Factor](https://arxiv.org/abs/2603.08058)

**Authors**: Jiayu Huang, Xiaohu Wu, Tiantian He, Qicheng Lao  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.08058v1  

#### Abstract
Large Language Models (LLMs) are pivotal in natural language processing. The impracticality of full fine-tuning has prompted Parameter-Efficient Fine-Tuning (PEFT) methods like Low-Rank Adaptation (LoRA), optimizing low-rank matrices A and B. In distributed scenarios where privacy constraints necess...

---

### 24. [Training event-based neural networks with exact gradients via Differentiable ODE Solving in JAX](https://arxiv.org/abs/2603.08146)

**Authors**: Lukas K\"onig, Manuel Kuhn, David Kappel, Anand Subramoney  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.08146v1  

#### Abstract
Existing frameworks for gradient-based training of spiking neural networks face a trade-off: discrete-time methods using surrogate gradients support arbitrary neuron models but introduce gradient bias and constrain spike-time resolution, while continuous-time methods that compute exact gradients req...

---

### 25. [AutoAdapt: An Automated Domain Adaptation Framework for LLMs](https://arxiv.org/abs/2603.08181)

**Authors**: Sidharth Sinha, Anson Bastos, Xuchao Zhang, Akshay Nambi, Chetan Bansal, Saravan Rajmohan  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.08181v1  

#### Abstract
Large language models (LLMs) excel in open domains but struggle in specialized settings with limited data and evolving knowledge. Existing domain adaptation practices rely heavily on manual trial-and-error processes, incur significant hyperparameter complexity, and are highly sensitive to data and u...

---

### 26. [NN-OpInf: an operator inference approach using structure-preserving composable neural networks](https://arxiv.org/abs/2603.08488)

**Authors**: Eric Parish, Anthony Gruber, Patrick Blonigan, Irina Tezaur  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.08488v1  

#### Abstract
We propose neural network operator inference (NN-OpInf): a structure-preserving, composable, and minimally restrictive operator inference framework for the non-intrusive reduced-order modeling of dynamical systems. The approach learns latent dynamics from snapshot data, enforcing local operator stru...

---

### 27. [Improving reasoning at inference time via uncertainty minimisation](https://arxiv.org/abs/2603.07159)

**Authors**: Nicolas Legrand, Kenneth Enevoldsen, M\'arton Kardos, Kristoffer Nielbo  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.07159v1  

#### Abstract
Large language models (LLMs) now exhibit strong multi-step reasoning abilities, but existing inference-time scaling methods remain computationally expensive, often relying on extensive sampling or external evaluators. We propose a principled strategy that frames reasoning as uncertainty minimisation...

---

### 28. [Reforming the Mechanism: Editing Reasoning Patterns in LLMs with Circuit Reshaping](https://arxiv.org/abs/2603.06923)

**Authors**: Zhenyu Lei, Qiong Wu, Jianxiong Dong, Yinhan He, Emily Dodwell, Yushun Dong, Jundong Li  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.06923v1  

#### Abstract
Large language models (LLMs) often exhibit flawed reasoning ability that undermines reliability. Existing approaches to improving reasoning typically treat it as a general and monolithic skill, applying broad training which is inefficient and unable to target specific reasoning errors. We introduce ...

---

### 29. [Lying to Win: Assessing LLM Deception through Human-AI Games and Parallel-World Probing](https://arxiv.org/abs/2603.07202)

**Authors**: Arash Marioriyad, Ali Nouri, Mohammad Hossein Rohban, Mahdieh Soleymani Baghshah  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.07202v1  

#### Abstract
As Large Language Models (LLMs) transition into autonomous agentic roles, the risk of deception-defined behaviorally as the systematic provision of false information to satisfy external incentives-poses a significant challenge to AI safety. Existing benchmarks often focus on unintentional hallucinat...

---

### 30. [RexDrug: Reliable Multi-Drug Combination Extraction through Reasoning-Enhanced LLMs](https://arxiv.org/abs/2603.08166)

**Authors**: Zhijun Wang, Ling Luo, Dinghao Pan, Huan Zhuang, Lejing Yu, Yuanyuan Sun, Hongfei Lin  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.08166v1  

#### Abstract
Automated Drug Combination Extraction (DCE) from large-scale biomedical literature is crucial for advancing precision medicine and pharmacological research. However, existing relation extraction methods primarily focus on binary interactions and struggle to model variable-length n-ary drug combinati...

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

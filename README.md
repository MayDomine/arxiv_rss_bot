# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-27 09:00:25 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SIKA-GP: Accelerating Gaussian Process Inference with Sparse Inducing Kernel Approximations for Bayesian Deep Learning](https://arxiv.org/abs/2605.26509)

**Authors**: Wenyuan Zhao, Rui Tuo, Chao Tian  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.26509v1  

#### Abstract
Gaussian processes (GPs) provide a principled Bayesian framework for uncertainty estimation, but their computational complexity severely limits scalability to large datasets. We propose SIKA-GP, which accelerates GP inference using sparse inducing kernel approximations based on a dyadic ordered temp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SIKA-GP: Accelerating Gaussian Process Inference with Sparse Inducing Kernel Approximations for Bayesian Deep Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Gaussian Process（GP）虽然在不确定性估计方面具有强大的贝叶斯框架优势，但其推理过程的计算复杂度为 $O(N^3)$，严重限制了其在大规模数据集上的应用。尽管已有稀疏变分高斯过程（SVGP）、KISS-GP 等方法通过引入诱导点（inducing points）降低复杂度，但在高维特征空间或需要大量诱导点时，效率仍然受限。

此外，在 **Bayesian Deep Learning**（BDL）场景下，如 Deep Gaussian Processes（DGP）和 Deep Kernel Learning（DKL），模型深度增加进一步加剧了计算负担，导致训练和推理成本高昂。

### 提出了什么新方法或新思路
本文提出 **SIKA-GP**（Sparse Inducing Kernel Approximation for GP），一种基于**稀疏激活基函数**的高效 GP 推理加速方法，其核心思想包括：

- **构建紧凑且表达能力强的核近似表示**：利用 **dyadic ordered template basis** 构造一组具有闭式表达式的紧支撑（compactly supported）基函数。
- **固定网格而非学习诱导点位置**：将诱导点设置在预定义的二进制网格（dyadic grid）上，避免重复求解核矩阵逆，提升优化稳定性。
- **实现 $O(\log M)$ 复杂度依赖**：每个输入仅激活 $O(\log M)$ 个基函数（其中 $M = 2^L + 1$ 是诱导点数量），显著减少计算量。
- **无缝集成到 BNN 架构中**：将 SIKA-GP 表示为一个稀疏激活的 Bayesian Neural Network（BNN）层，支持高效的张量化 GPU 并行计算。
- **自然扩展至深层架构**：可堆叠形成 **SIKA-DGP** 或作为 DKL 的最后一层，解决深度模型中的可扩展性瓶颈。

### 相比现有方法的优势
| 特性 | SVGP / KISS-GP | DAK | SIKA-GP（本文） |
|------|----------------|-----|----------------|
| 诱导点是否可学习 | ✅ 可学习 | ❌ 固定网格 | ❌ 固定网格 |
| 激活基函数数量 | $O(M)$ | $O(M)$ | $O(\log M)$ |
| 推理复杂度 | $O(SNM^2)$ / $O(SDM^{1+\epsilon})$ | $O(SDM)$ | $O(SD \log M)$ |
| 支持张量并行 | 有限 | 部分 | ✅ 张量化稀疏索引（TSI） |
| 易于集成到 BNN | 中等 | 较好 | ✅ 自然嵌入稀疏 BNN |

> ✅ **关键优势**：在保持预测精度的同时，实现了**指数级的推理加速**，尤其适用于大 $M$ 和深模型场景。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **回归任务**：
  - UCI 数据集：Gas, Kin40K, Protein
- **图像分类任务**：
  - MNIST, CIFAR-10, CIFAR-100
- **语言模型任务**：
  - CLINC150（Transformer-based intent classification + OOD detection）

### 实验设置和评估指标

#### 模型配置
- **DGP-SIKA**：堆叠两层 SIKA-GP，每层 dyadic level $L=7$（即 $M=129$ 个诱导点）
- **DKL-SIKA**：使用 CNN / ResNet / DistilBERT 提取特征，后接 SIKA-GP 层
- 所有模型采用 **variational inference**（VI）进行训练，最大化 ELBO
- 使用 **Tensorized Sparse Indexing (TSI)** 实现高效前向传播

#### 评估指标
| 类型 | 指标 |
|------|------|
| 回归 | RMSE ↓, NLPD ↓, 运行时间 ↓ |
| 分类 | Accuracy (ACC) ↑, NLL ↓, ECE ↓, AUROC ↑（OOD）, AUPRC ↓（OOD） |
| 效率 | Train Time (s/epoch) ↓, Infer Time (s) ↓ |

#### 基线方法对比
- **SVGP**：标准稀疏变分 GP
- **KISS-GP**：基于插值的结构化核方法
- **DAK**（Deep Additive Kernel）：当前最先进的 GP-BNN 转换方法
- **SVDKL**：经典 DKL 实现

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### 📊 表格汇总：UCI 回归任务（DGP-SIKA vs DGP）
| Dataset | Method | RMSE ↓ | NLPD ↓ | Time (s/epoch) ↓ |
|--------|--------|--------|--------|------------------|
| Gas | DGP | 0.54±0.01 | 1.09±0.12 | 19.43±1.66 |
| | DGP-SIKA | **0.53±0.03** | **1.07±0.11** | **2.79±0.04** ⬇️ **7×** |
| Kin40K | DGP | 0.09±0.01 | 0.07±0.01 | 33.25±0.56 |
| | DGP-SIKA | 0.09±0.01 | 0.07±0.01 | **15.76±0.51** ⬇️ **2×** |
| Protein | DGP | 0.67±0.04 | 0.94±0.04 | 56.50±1.14 |
| | DGP-SIKA | 0.68±0.04 | 0.94±0.04 | **28.92±0.63** ⬇️ **2×** |

✅ **结论**：DGP-SIKA 在所有数据集上达到与完整 DGP 相当甚至更优的预测性能，同时训练时间大幅下降。

---

#### 🖼️ 图像分类任务（DKL-SIKA vs DAK / SVDKL）

| Dataset | Method | ACC (%) ↑ | ECE (%) ↓ | Train Time ↓ | Infer Time ↓ |
|--------|--------|-----------|-----------|--------------|---------------|
| MNIST | SVDKL | 98.17 | 1.87 | 16.46 | 1.81 |
| | DAK | 99.06 | 0.50 | 13.73 | 1.42 |
| | **DKL-SIKA** | **99.06** | **0.53** | **10.09** ⬇️ 1.6× | **1.00** |
| CIFAR-10 | SVDKL | 94.28 | 4.10 | 46.90 | 6.08 |
| | DAK | 94.53 | 4.34 | 35.04 | 6.22 |
| | **DKL-SIKA** | **94.78** | **4.06** | **17.26** ⬇️ 2.7× | **4.50** |
| CIFAR-100 | DAK | 76.61 | 5.55 | 102.57 | 5.74 |
| | **DKL-SIKA** | **76.94** | **4.31** | **36.31** ⬇️ 2.8× | **4.22** |

✅ **结论**：
- DKL-SIKA 在准确率和校准误差（ECE）上优于或媲美基线；
- 训练速度提升 **2–3倍以上**，推理也更快；
- 尤其在 CIFAR-100 上表现突出，说明对高维特征有效。

---

#### 🧠 Transformer 语言模型任务（CLINC150）

| Model | ACC↑ | NLL↓ | ECE↓ | AUROC↑ | Time (min/epoch)↓ |
|-------|------|------|------|--------|--------------------|
| SVDKL | 95.03 | 0.26 | 2.34 | 0.8413 | 9.35 |
| DAK | 94.65 | 0.26 | 2.12 | 0.8486 | 6.22 |
| **DKL-SIKA** | **95.03** | **0.24** | **2.05** | **0.8590** | **3.41** |

✅ **结论**：
- ID 性能持平或更好；
- OOD 检测能力更强（最高 AUROC）；
- 训练时间减少 **超过 2倍**，首次使 GP 能高效用于 TLM 场景。

---

### 消融实验结果（Ablation Study）

#### 🔤 不同核函数比较（CIFAR-100）
| Kernel | ACC | ECE/NLL | Train/Infer Time |
|--------|-----|---------|------------------|
| RBF (DAK) | 77.43 | 5.36 / 0.98 | 37.38 / 9.78 |
| Laplace (DAK) | 76.13 | 4.18 / 0.99 | 35.62 / 9.66 |
| Laplace (SIKA) | **76.61** | **3.45 / 0.94** | **20.08 / 7.07** |

➡️ **发现**：尽管 Laplace 核比 RBF 更受限，但在 DKL 框架下，结合 SIKA 后反而提升了校准性和效率，验证了“以结构换灵活性”的合理性。

#### 🔢 不同 dyadic level $L$ 对性能的影响（CIFAR-10）
| $L$ | $M$ | Active Bases | ACC | ECE | NLL |
|-----|-----|---------------|------|------|------|
| 1 | 3 | 3 | 92.59 | 5.79 | 0.420 |
| 2 | 5 | 4 | 94.56 | 4.17 | 0.279 |
| 4 | 17 | 6 | 94.59 | 4.78 | 0.391 |
| 7 | 129 | 9 | 94.76 | 4.10 | 0.281 |
| 10 | 1025 | 12 | **94.78** | **4.06** | **0.270** |

➡️ **发现**：
- 准确率随 $L$ 快速上升后趋于稳定；
- 即使使用上千个诱导点（$M=1025$），每个样本仍只激活约 12 个基函数；
- 支持使用密集网格而不牺牲效率。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **SIKA-GP 实现了 GP 推理的指数级加速**：通过稀疏激活机制，将复杂度从 $O(M)$ 降至 $O(\log M)$，特别适合大规模和深层模型。
2. ✅ **预测性能无损甚至提升**：在多个视觉、语言和回归任务上，SIKA-GP 保持甚至优于传统稀疏 GP 方法的准确性与不确定性校准能力。
3. ✅ **天然兼容现代深度学习架构**：可轻松嵌入 BNN、DGP、DKL 和 Transformer 模型，成为可插拔的“不确定性模块”。
4. ✅ **支持更多 MC 样本采样**：由于前向速度快，可在训练和推理中使用更多 MC 样本，从而获得更鲁棒的不确定性估计。

### 方法的局限性
1. ⚠️ **依赖 Laplace 核的马尔可夫性质**：当前理论推导基于 Laplace kernel 的特殊结构，难以直接推广到 RBF、Matérn 等非马尔可夫核。
2. ⚠️ **诱导点位置固定**：无法自适应地学习最优诱导点布局，可能在某些非平稳数据上略逊于可学习位置的方法。
3. ⚠️ **输入需归一化至 [0,1]**：要求输入标准化，增加了预处理步骤。

### 未来工作方向
- 🔄 **开发自适应或混合 dyadic 网格**：允许局部细化或动态调整网格密度。
- 🔗 **扩展至更广泛的核函数类别**：探索如何将 SIKA 思路应用于非马尔可夫核（如 RBF）。
- 🧩 **与参数高效微调方法结合**：如 LoRA + SIKA-GP，用于大型语言模型的贝叶斯微调。
- 💾 **硬件层面优化**：设计专用稀疏张量操作以进一步提升 GPU 利用率。

---

> 🔗 **代码开源地址**：https://github.com/warrenzha/sika-gp  
> 📘 **一句话总结**：SIKA-GP 通过结构化稀疏逼近，为 Gaussian Process 在现代 Bayesian Deep Learning 中的大规模应用提供了**高效、稳定、可扩展的新路径**。

</details>

---

### 2. [Semantic-aware Token Selection and Resource Optimization for Communication-efficient Split Federated Fine-tuning in Edge Intelligence](https://arxiv.org/abs/2605.26120)

**Authors**: Xianke Qiang, Zheng Chang, Geyong Min  
**Category**: cs.DC  
**Published**: 2026-05-27  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.26120v1  

#### Abstract
Deploying large Transformer-based vision models on resource-limited mobile devices at network edge is severely constrained by hardware limitations and dynamic wireless environments. While federated learning (FL) enables collaborative training without sharing raw data, strictly local fine-tuning of s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Semantic-aware Token Selection and Resource Optimization for Communication-efficient Split Federated Fine-tuning in Edge Intelligence

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在边缘智能（Edge Intelligence）场景中，将基于Transformer的大规模视觉模型（如 ViT）部署到资源受限的移动设备面临两大挑战：
- **计算瓶颈**：即使采用参数高效微调技术（如 LoRA），本地微调仍对边缘设备的算力和内存要求过高。
- **通信瓶颈**：Split Federated Learning（SFL）虽能通过模型分割减轻客户端负担，但在上传高维激活 token 时引入巨大通信开销，尤其在动态无线信道下难以承受。

现有方法多依赖**bit-level 压缩**（如量化、稀疏化），忽略了 Transformer 激活本身具有语义结构化的 token 特性，无法实现更高效的压缩。

---

### 🚀 提出的新方法：ST-SFLora
作者提出 **ST-SFLora** —— 一种**语义感知的分拆联邦 LoRA 微调框架**，其核心创新如下：

#### （1）**语义感知的 token 选择机制**
- 利用 Transformer 自注意力机制中的 `[CLS]` token 对各 patch token 的 attention score 作为语义重要性度量。
- 客户端仅上传 top-K 最重要的 token，并通过 token merging 保留被丢弃 token 的上下文摘要，减少信息损失。

#### （2）**新型系统级度量：Semantic Transmission Efficiency (STE)**
- 定义了一个新的优化目标：  
  $$
  \text{STE} = \frac{\sum_{m \in \mathcal{U}} f_m(K_m)}{\max_{m \in \mathcal{U}} T_m}
  $$
  其中分子为所有客户端保留的总语义信息，分母为最慢客户端的上行延迟（straggler effect）。
- STE 显式刻画了“语义保留”与“通信效率”的权衡，是首个面向任务效用的语义吞吐率指标。

#### （3）**联合资源优化问题建模与求解**
- 构建混合整数非凸优化问题，联合优化：
  - **token 数量 $K_m$**（整数变量）
  - **上行带宽分配 $W_m$**
  - **发射功率 $p_m$**
- 在延迟、能量约束下最大化 STE。
- 设计交替优化算法（Alternating Optimization）分解为三个子问题迭代求解，保证收敛性和实用性。

---

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **通信效率** | 从 bit-level 压缩升级为 token-level 语义压缩，显著降低传输负载（见 Table II）。 |
| **客户端轻量化** | 客户端仅执行前向传播，不参与反向传播或梯度聚合，极大降低设备负担。 |
| **适应动态环境** | 联合考虑信道状态（CSI）、能耗、延迟等现实约束，支持弹性参与和鲁棒训练。 |
| **无需额外模块** | 直接复用原生 attention 权重进行 token 评分，无需引入额外的语义编码器或打分网络。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
实验在三个图像分类基准上进行：
- **ImageNet100**：通用物体识别，100类子集。
- **Oxford Flowers-102**：细粒度花分类，类别间差异小。
- **CUB-200-2011**：鸟类细粒度分类，需捕捉细微特征。

均考虑 **IID 与 Non-IID** 两种数据分布以模拟真实边缘场景。

---

### ⚙️ 实验设置
- **网络架构**：ViT-S/16、ViT-B/16、ViT-L/16
- **客户端数量**：100个，每轮按泊松分布随机激活
- **覆盖范围**：5–500米半径圆形区域
- **通信参数**：
  - 上行总带宽：50 MHz
  - 最大发射功率：0.2 W
  - 噪声谱密度：-174 dBm/Hz
  - 路径损耗指数：2.5
- **硬件配置**：
  - GPU频率：[1.0, 1.5] GHz
  - GPU核心数：[4, 6]（模拟 Apple A15 等移动端芯片）

---

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **模型性能** | Top-1 Accuracy (%) |
| **资源消耗** | GPU Memory (GB), Communication Overhead (MB) |
| **系统效率** | Semantic Transmission Efficiency (STE), Convergence Speed |
| **优化效果** | 平均选中 token 数量、能量/带宽利用率 |

---

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **LocalLoRA** | 本地 LoRA 微调，无协作 |
| **FedLoRA** | 标准联邦学习 + LoRA，上传 LoRA 参数 |
| **SplitLoRA** | 分拆学习（Sequential SL）+ LoRA |
| **SFLora** | 并行 SFL + LoRA，上传完整激活 |
| **ST-SFLora-Full** | 所提框架但不启用 token selection（即上传全部 token） |
| **ST-SFLora (Ours)** | 完整方案：语义感知 token 选择 + 联合资源优化 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table I）

| Backbone | Method | ImageNet100 (Non-IID) | Flowers-102 (Non-IID) | CUB-200-2011 (Non-IID) |
|---------|--------|------------------------|------------------------|------------------------|
| ViT-B/16 | SplitLoRA | 89.47 | 99.29 | 80.65 |
| ViT-B/16 | SFLora | 89.09 | 98.69 | 79.84 |
| ViT-B/16 | **ST-SFLora (Ours)** | **85.81** | **96.43** | **73.69** |

> 尽管精度略有下降（约 3–5%），但 **ST-SFLora 在极端资源受限下仍显著优于 FedLoRA（仅 ~50% 准确率）**。

---

### 🔻 资源消耗对比（Table II）

| Method | GPU Mem (GB) | Model Comm (MB) | Token Comm (MB) |
|--------|---------------|------------------|------------------|
| LocalLoRA / FedLoRA | 9.0 | 335.3 | – |
| SplitLoRA / SFLora | 2.3 | – | ~58.2 |
| **ST-SFLora (top-K)** | **1.4** | **0** | **~16 (K=64)** |

> - **GPU 内存减少 84.4%**（9.0 → 1.4 GB）
> - **完全消除模型广播开销**
> - **激活通信减少至原来的 ~27%**

---

### 🔬 消融实验结果（Fig. 8）

#### （1）**优化组件消融分析**
- 移除 **token selection** 导致 STE 下降最严重 → 表明 token 压缩是提升效率的关键。
- 移除 **bandwidth allocation** 次之 → 动态频谱分配对缓解 straggler 至关重要。
- 移除 **power control** 影响较小但仍明显 → 自适应功率调节有助于应对信道波动。

#### （2）**token 数量影响（Fig. 10）**
- 随着 $K$ 增加（64 → 160），准确率单调上升。
- 当 $K \geq 128$ 时，性能接近 full-token 基线。
- 在 CUB 等细粒度任务中，较低 $K$ 仍可保持竞争力，说明语义集中性强。

#### （3）**STE 曲线存在峰值（Fig. 6）**
- 存在一个最优 $K$（如 135）使得 STE 达到最大值（43.45）。
- 验证了“边际语义增益 = 边际延迟成本”的理论平衡点。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Token-level 语义压缩优于传统 bit-level 压缩**：利用 attention score 进行 top-K 选择可在极低通信代价下保留关键语义信息。
2. **STE 是有效的系统级优化目标**：能够统一衡量“学习质量”与“通信效率”，指导资源调度决策。
3. **联合优化显著提升系统效率**：动态调整 $K_m, W_m, p_m$ 可自适应信道变化，在严格延迟/能量约束下维持可行性。
4. **客户端可做到极致轻量化**：仅需前向推理 + token 选择，适合部署于低端移动设备。

---

### ⚠️ 局限性
1. **依赖预训练模型的 attention 机制**：若 attention 不能有效反映语义重要性（如某些 domain-specific 模型），性能可能下降。
2. **未考虑下行反馈延迟**：当前假设服务器侧更新无延迟，实际中可能成为瓶颈。
3. **静态 cut layer 设置**：未动态调整模型分割位置（cut layer），限制了进一步优化空间。
4. **仅验证于 CV 任务**：尚未扩展至 NLP 或 multimodal 场景。

---

### 🔮 未来工作方向
1. **设计 learnable token selector**：引入轻量子网络学习更优的选择策略，超越固定 attention 规则。
2. **推广至 NLP 和多模态任务**：探索文本 token 或跨模态 token 的语义压缩机制。
3. **动态 cut layer + token selection 联合优化**：根据设备能力与信道状态自适应决定模型分割点。
4. **支持异步训练机制**：缓解 straggler 问题，提高系统吞吐量。

---

> 💡 **总体评价**：  
> 本文首次将 **semantic communication 思想引入 Split Federated Learning**，提出了一个面向 6G 的高效边缘微调范式。通过 **token-level 语义感知压缩 + 资源联合优化**，实现了在极低资源消耗下的可用模型性能，为大规模 Foundation Model 在边缘落地提供了新路径。

</details>

---

### 3. [Totoro$^+$: An Adaptive and Scalable Edge Federated Learning System](https://arxiv.org/abs/2605.26323)

**Authors**: Cheng-Wei Ching, Xin Chen, Taehwan Kim, Jian-Jhih Kuo, Dilma Da Silva, Liting Hu  
**Category**: cs.DC  
**Published**: 2026-05-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.26323v1  

#### Abstract
Federated Learning (FL) is an emerging distributed machine learning (ML) technique that enables in-situ model training and inference on decentralized edge devices. We propose Totoro$^+$, a novel scalable FL system that enables massive FL applications to run simultaneously on edge networks. The key i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Totoro+: An Adaptive and Scalable Edge Federated Learning System》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前主流的 Federated Learning (FL) 系统大多采用**中心化**（single master / many workers）或**分层式架构**，存在以下关键挑战：
- **可扩展性瓶颈**：随着边缘网络中 FL 应用数量和设备规模的增长，单一的协调器（Coordinator）或参数服务器容易成为性能瓶颈。
- **缺乏适应性**：难以应对边缘网络中的动态变化，如链路不稳定、带宽受限、节点频繁加入/退出（churn）、资源异构等。
- **应用定制性差**：多个 FL 应用共享同一套系统策略，难以支持多样化的训练需求（如同步/异步通信、个性化聚合函数等）。

### **提出的新方法与创新点**
Totoro+ 提出了一种**完全去中心化**（fully decentralized）的边缘联邦学习系统，其核心创新在于将传统的集中式架构重构为基于 **Distributed Hash Table (DHT)** 的 **Peer-to-Peer (P2P)** 架构，并引入三大关键技术：

#### **1. Locality-aware P2P Multi-ring Structure**
- 将所有边缘节点组织成一个 DHT-based 的 P2P 覆盖网络（overlay），并划分为多个“多环”（multi-ring）结构。
- 引入**两级路由表**（level 1 & level 2 routing table），实现**地理区域感知**（locality-aware）和**管理隔离**（administrative isolation），确保数据流在本地域内收敛。

#### **2. Publish/Subscribe-based Forest Abstraction**
- 在 P2P 多环之上构建一个“森林”抽象，每个 FL 应用拥有独立的**动态结构化数据流树**（dataflow tree）。
- 支持**一对多模型广播**（model broadcast）和**多对一梯度聚合**（gradient aggregation）。
- 新增 **Advertise-Discover (AD) Tree**，使节点能主动发布和发现运行中的 FL 应用，解决了原 Totoro 版本中应用发现依赖外部机制的问题。

#### **3. Game-theoretic Path Planning Model**
- 针对边缘网络的不确定性和拥塞问题，提出一种基于**博弈论**的路径规划算法。
- 将路径选择建模为一个**带臂反馈的拥堵博弈**（congestion game with bandit feedback）。
- 设计了一个分布式逐跳路由算法，理论上保证达到 **e-approximate Nash Equilibrium**，实现自适应、鲁棒的数据传输。

### **相比现有方法的优势**
| 维度 | 传统 FL 系统 | Totoro+ |
|------|-------------|--------|
| **架构** | Centralized / Hierarchical | Fully Decentralized |
| **参数服务器** | 单一或少量共享 | 每个应用独享（many masters） |
| **角色灵活性** | 固定角色 | 任意节点可担任协调者、聚合者、工作者等 |
| **可扩展性** | 受限于中心节点 | 支持海量并发应用 |
| **适应性** | 弱，依赖静态配置 | 强，动态路径重规划 |
| **理论保障** | 无 | e-approximate Nash Equilibrium |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **语音识别**：`Google Speech` [75]（目标准确率 53.0%）
- **图像分类**：`FEMNIST` [9]（目标准确率 75.5%）
- （附录中还使用了 `CIFAR-10`, `MASSIVE` 等）

### **实验设置**
- **平台**：500 台 AWS EC2 `t2.medium` 节点（每台 2 vCPU, 4GB RAM）
- **模拟规模**：通过 JVM 模拟最多 100k 边缘节点
- **地理分布**：基于澳大利亚 EUA 数据集 [73] 生成 12 个地理区域
- **模型**：
  - `ResNet-34`（用于 Google Speech）
  - `ShuffleNet V2`（用于 FEMNIST）
- **扇出（fanout）**：测试了 8、16、32 三种数据流树结构

### **评估指标**
- **可扩展性**：
  - 模型广播时间（Model dissemination time）
  - 梯度聚合时间（Gradient aggregation time）
  - 主节点负载分布（Load balancing）
- **FL 性能**：
  - **Time-to-Accuracy**：达到目标准确率所需的总训练时间
- **适应性**：
  - 累积包延迟（Cumulative packet latency）
  - Nash Regret（衡量博弈稳定性）
  - 故障恢复时间（Failure recovery time）
- **开销**：
  - CPU 和内存占用
  - 通信成本（TCP/UDP 流量）

### **基线方法对比**
- **OpenFL** [25]：Intel 开源的单机 FL 框架
- **FedScale** [26]：Symbiotic Lab 开发的大规模 FL 基准框架

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 任务 | 模型 | 应用数 | Totoro+ 加速比（vs OpenFL） | Totoro+ 加速比（vs FedScale） |
|------|------|--------|-------------------------------|------------------------------|
| 语音识别 | ResNet-34 | 20 | **13.0× – 14.0×** | **12.4× – 13.5×** |
| 图像分类 | ShuffleNet V2 | 20 | **5.0× – 10.3×** | **5.6× – 11.5×** |

> 注：加速比随应用数量增加而显著提升，尤其在高并发场景下优势明显。

### **与基线方法的对比结果**
- **训练时间大幅缩短**：
  - 当同时训练 20 个模型时，Totoro+ 将总训练时间从 OpenFL/FedScale 的数十小时缩短至约 **15.5 小时**，且几乎不随应用数量增长而增加。
- **通信效率更高**：
  - 模型广播和梯度聚合仅需 **O(log N)** 跳，即使在百万级节点规模下也保持高效。
  - 通信开销仅比基线高 **1.19× (TCP)** 和 **1.29× (UDP)**，新增开销极小。
- **负载均衡优异**：
  - 99.5% 的节点最多作为 3 个应用的主节点，避免了热点问题。
- **适应性强**：
  - 在带宽受限环境下，Totoro+ 的累积延迟增长缓慢，远优于 bandit-based 方法。
  - 实现了 **e-approximate Nash Equilibrium**，证明路径选择稳定。

### **消融实验结果**
- **AD Tree 的作用**：
  - 新节点无需广播即可快速发现可用 FL 应用，降低网络压力。
- **Game-theoretic vs Bandit Model**：
  - 博弈论模型在拥塞控制和延迟优化上显著优于忽略带宽限制的 bandit 模型。
- **Failure Recovery**：
  - 单棵树中 128 个节点同时失效，恢复时间呈线性增长，验证了局部修复的有效性。
  - 多棵树同时故障，恢复时间保持稳定，支持并行恢复。

---

## **4. 关键结论和发现**

### **主要发现**
1. **去中心化是解决大规模边缘 FL 可扩展性的有效途径**：Totoro+ 通过 DHT + P2P 架构成功打破了中心化瓶颈。
2. **“一应用一主”模式优于“多应用共用一主”**：专用参数服务器显著提升了并发处理能力。
3. **博弈论路径规划能有效应对边缘网络不确定性**：相比启发式方法，具有更强的自适应能力和理论保障。
4. **系统具备良好的弹性与容错性**：支持节点动态加入/退出、故障自动恢复，适用于真实边缘环境。

### **局限性**
- **安全假设较强**：目前未深入讨论恶意节点攻击（如 Sybil 攻击）下的安全性，依赖信任权威机构（如 CA）进行身份认证（见 Appendix N）。
- **逻辑节点映射复杂性**：将物理节点映射为多个 P2P 逻辑节点虽能适配异构资源，但也增加了管理复杂度。
- **部署门槛较高**：完全去中心化架构对网络连通性和节点协作性要求更高，可能不适合封闭式企业环境。

### **未来工作方向**
- **增强安全性**：集成区块链或零知识证明技术，实现去中心化身份认证与防篡改。
- **支持多播通信**：扩展当前逐跳单播模型，研究基于 DHT 的高效多播机制（Appendix N 已初步探讨）。
- **跨平台集成**：探索与现有 FL 框架（如 PySyft, TensorFlow Federated）的深度集成方案。
- **非独立同分布（Non-IID）场景优化**：结合个性化联邦学习（Personalized FL）算法，在系统层面提供更好支持。

--- 

> ✅ **总结一句话**：  
> Totoro+ 通过 **DHT-based P2P 架构 + 发布/订阅森林抽象 + 博弈论路径规划**，实现了**高度可扩展、强适应性、低延迟**的边缘联邦学习系统，在真实世界实验中相较 OpenFL 和 FedScale **提速达 1.2× 至 14.0×**，为未来大规模边缘智能提供了新的系统范式。

</details>

---

### 4. [Provably Communication-Efficient and Privacy-Preserving Federated Graph Neural Networks](https://arxiv.org/abs/2605.26243)

**Authors**: Zhishuai Guo, Wenhan Wu, Chen Chen, Lei Zhang, Olivera Kotevska, Ravi K Madduri  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.26243v1  

#### Abstract
Graph neural networks (GNNs) achieve strong performance on relational data, but real-world graphs are often distributed across organizations that cannot share raw data due to privacy and policy constraints. Existing federated GNN methods either ignore cross-client links, leading to degraded accuracy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Provably Communication-Efficient and Privacy-Preserving Federated Graph Neural Networks*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Federated GNN**（FedGNN）方法面临以下核心挑战：
- **忽略跨客户端边（cross-client edges）**：导致模型无法捕捉全局图结构，严重影响准确性。
- **频繁交换节点嵌入（embeddings）**：虽然能建模跨客户端依赖，但通信开销巨大。
- **隐私风险**：即使不共享原始数据，交换的中间嵌入仍可能泄露敏感信息，而标准 **Differential Privacy (DP)** 在此场景下因过保守而失效。

### 提出的新方法：CE-FedGNN
本文提出 **CE-FedGNN**（Communication-Efficient and Privacy-preserving Federated GNN），其核心创新包括：

#### （1）通信高效机制
- **移动平均估计器（Moving-Average Estimator）**：
  - 客户端维护节点嵌入的移动平均值，而非每轮都更新。
  - 当需要跨客户端邻居的嵌入时，复用最近一次共享的移动平均版本，有效缓解表示漂移（representation drift）。
- **稀疏通信**：
  - 仅在必要时（如边界节点更新后）才交换聚合后的节点嵌入。
  - 实现 **O(T³/⁴)** 通信复杂度，远低于传统方法的 O(T)。

#### （2）形式化隐私保护：Metric-DP
- 引入 **Metric Differential Privacy (metric-DP)** 框架，替代传统 DP。
- 隐私预算基于嵌入空间中的 **L2 距离**，而非最坏情况下的输入扰动。
- 在实际噪声水平下提供有意义的隐私保证，避免了标准 DP 的“空洞界限”（vacuous bounds）。

#### （3）理论保障
- **收敛性分析**：证明算法以 **O(1/√T)** 的速率收敛到驻点。
- **隐私组成分析**：通过 **Rényi DP** 组合，推导出多轮训练下的 **(ε, δ)-metric-DP** 保证。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **准确性** | 显式建模跨客户端边，优于忽略这些边的方法（如 FedAvg）。 |
| **通信效率** | 移动平均 + 稀疏通信，显著减少通信轮次，优于每轮交换嵌入的方法。 |
| **隐私性** | Metric-DP 在合理噪声下提供非平凡隐私保证，优于标准 DP。 |
| **可扩展性** | 不依赖复杂的全局图重建或昂贵的矩阵分解，适用于动态图。 |

---

## 2. 核心实验方法和设置

### 数据集
- **合成反洗钱（AML）数据集**（基于 Altman et al. [4] 的模拟器）：
  - 包含 Small/Medium/Large 三种规模，每种分 High-Illicit (HI) 和 Low-Illicit (LI) 两类。
  - 模拟银行间交易网络，节点为账户，边为交易。
- **引用网络基准数据集**：
  - Cora, Citeseer, PubMed, MSAcademic。
  - 用于验证方法在通用图任务上的泛化能力。

### 实验设置
- **联邦划分**：图数据按节点/边分布到 4–32 个客户端。
- **模型架构**：
  - AML 任务：GIN 和 PNA（增强表达力）。
  - 引用网络：标准 GCN。
- **评估指标**：
  - 主要指标：**Macro F1 Score**（平均测试 F1 分数）。
  - 其他：通信量（bytes）、运行时间、隐私-效用权衡。

### 基线方法对比
| 基线 | 简介 |
|------|------|
| **SC (Single Client)** | 各客户端独立训练，不协作。 |
| **FedAvg** | 仅交换模型参数，忽略跨客户端边。 |
| **Swift-FedGNN** | 间歇性访问全局图，通信频率低但建模不完整。 |
| **FedGCN** | 仅在初始化时共享一次嵌入。 |
| **FedGNN-ST** | 每轮共享旧版（stale）嵌入，无移动平均。 |
| **FedSage+ / FedPUB** | 更复杂的生成式或加权聚合方法，但计算开销大。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1 & 2）

#### 表 1：AML 数据集上的平均 F1 分数（部分）
| 方法 | HI-Medium | LI-Medium | LI-Large |
|------|-----------|-----------|----------|
| SC-GIN | 0.3572 | 0.0746 | 0.0227 |
| FedAvg-GIN | 0.5421 | 0.0068 | 0.0000 |
| Swift-GIN | 0.5689 | 0.0061 | 0.1294 |
| **CE-FedGNN-GIN** | **0.6024** | **0.0828** | **0.1917** |
| **CE-FedGNN-PNA** | **0.6517** | **0.2918** | **0.3158** |

> ✅ CE-FedGNN 在所有设置下均显著优于基线，尤其在极端类别不平衡（LI）场景下仍保持鲁棒性。

#### 表 2：引用网络上的平均 F1 分数
| 方法 | Cora | Citeseer | Pubmed | MSAcademic |
|------|------|----------|--------|------------|
| FedAvg | 0.1868 | 0.2189 | 0.2855 | 0.2759 |
| Swift | 0.4296 | 0.4023 | 0.6181 | 0.8003 |
| FedGNN-ST | 0.4345 | 0.3972 | 0.6347 | 0.8168 |
| **CE-FedGNN** | **0.4701** | **0.4343** | **0.6517** | **0.8497** |

> ✅ 在标准图数据集上也全面领先，表明方法具有良好的泛化能力。

### 消融实验结果（Table 5）
| 变体 | HI-Medium | LI-Medium |
|------|-----------|-----------|
| w/o global embedding | 0.5037 | 0.2614 |
| Stale global embedding | 0.5382 | 0.1129 |
| w/o gradient moving average | 0.6457 | 0.3535 |
| **CE-FedGNN (full)** | **0.6517** | **0.2918** |

> 🔍 发现：
> - 移除全局嵌入共享导致性能大幅下降，验证了跨客户端信息的重要性。
> - 使用“stale”嵌入（无移动平均）在 LI-Medium 上表现更差，说明移动平均对缓解表示漂移至关重要。
> - 移除梯度移动平均在多数情况下略差，但在 LI-Medium 上反而更好，可能因高方差下平滑有害。

### 通信效率实验（Figure 2）
- 即使将通信间隔 **K 增加到 1024**，CE-FedGNN 仍能保持高性能。
- 相比之下，其他方法在稀疏通信下性能急剧下降。
> ✅ 移动平均机制支持长期重用嵌入，极大降低通信需求。

### 隐私-效用权衡（Figure 2）
- 注入嵌入噪声（σ₀）后，性能随噪声增大而缓慢下降。
- 即使在较高噪声下（如 σ₀=1.2），CE-FedGNN 仍优于 FedAvg。
> ✅ 方法对噪声鲁棒，支持在实用噪声水平下实现有意义的 **metric-DP** 保证。

---

## 4. 关键结论和发现

### 主要发现
1. **移动平均 + 稀疏通信** 是解决 FedGNN 通信瓶颈的有效范式。
2. **Metric-DP** 比标准 DP 更适合嵌入释放场景，在实际噪声下提供非平凡隐私保证。
3. CE-FedGNN 在 **准确性、通信效率、隐私性** 三方面取得良好平衡。
4. 方法在 **高度不平衡数据**（如 AML）和 **标准图数据** 上均表现出色，具备强泛化能力。

### 局限性
- **公共队列威胁模型（public-cohort threat model）**：假设对手知道每轮哪些客户端参与更新，未利用子采样放大（subsampling amplification）带来的额外隐私增益。
- **依赖边界节点识别**：需明确哪些节点是“边界节点”以决定通信对象。
- **合成数据验证**：AML 实验基于模拟数据，真实金融数据上的效果待进一步验证。

### 未来工作方向
- 结合 **secure aggregation** 或 **shuffling** 以引入子采样放大，进一步提升隐私。
- 探索 **自适应通信策略**，动态决定何时及与谁通信。
- 将框架扩展至 **异构图（Heterogeneous Graphs）** 和 **动态图（Dynamic Graphs）** 场景。
- 在更多真实世界场景（如医疗联合学习）中部署验证。

--- 

> 📌 **总结**：  
> CE-FedGNN 是首个同时实现 **通信高效**、**形式化隐私保护** 和 **高准确性** 的 Federated GNN 框架。它通过 **移动平均估计器** 缓解通信压力，采用 **metric-DP** 提供更实用的隐私保障，并在多种图任务上验证了其优越性，为隐私保护的图学习提供了新的研究范式。

</details>

---

### 5. [UnityMAS-O: A General RL Optimization Framework for LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2605.26646)

**Authors**: Yiqun Chen, Wei Yang, Erhan Zhang, Shijie Wang, Qi Liu, Zechun Niu, Bin Zhang, Haitao Li, Rui Li, Lingyong Yan, Jinyuan Feng, Biqing Qi, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao  
**Category**: cs.AI  
**Published**: 2026-05-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.26646v1  

#### Abstract
LLM-based multi-agent systems decompose complex tasks into interacting roles, but most remain manually orchestrated by prompts, tools, and control rules, while agents are rarely optimized through a unified reinforcement learning interface. Existing RL post-training frameworks mainly target single-po...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# UnityMAS-O: A General RL Optimization Framework for LLM-Based Multi-Agent Systems 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前基于 Large Language Models (LLMs) 的 **multi-agent systems** 虽然在任务分解、角色分工（如 planner、retriever、coder）方面展现出潜力，但其优化机制存在严重不足：

- 多数系统依赖手动设计的 **prompt、tool 调用规则和控制流**，缺乏统一的 **reinforcement learning (RL)** 接口进行端到端优化。
- 现有的 **post-training RL 框架**（如 TRL、OpenRLHF、verl）主要面向单策略（single-policy）优化，难以支持多智能体之间的结构化交互、角色级信用分配（credit assignment）和灵活的参数共享。

因此，**缺乏一个通用框架来将任意用户定义的 multi-agent workflow 转化为可训练的 MARL（Multi-Agent RL）系统**。

---

### **提出了什么新方法或新思路**

论文提出 **UnityMAS-O**，一个通用的 RL 优化框架，用于对 LLM-based multi-agent systems 进行端到端强化学习训练。其核心创新在于以下四个一级抽象（first-class abstractions）：

| 抽象对象 | 描述 |
|--------|------|
| **Logical Agent Roles** | 将“角色”作为独立于模型参数的语义单元（如 planner、retriever），支持同一角色由不同模型实现，或多个角色共享同一模型。 |
| **Graph-Structured Trajectories** | 将整个 multi-agent 执行过程建模为有向图轨迹，记录状态转移、工具调用、中间输出等结构化信息。 |
| **User-Defined Reward Functions** | 支持在 **role-level、turn-level、trajectory-level** 定义奖励，实现细粒度信用分配（如只奖励改进者）。 |
| **Explicit Agent-Model Mappings** | 显式定义逻辑角色到物理 LLM 实例的映射 `φ: V → M`，支持 **full sharing、partial sharing、full separation** 三种参数共享模式。 |

此外，系统层面采用 **Ray-based star-topology runtime**：
- **中央控制器**：负责 workflow 调度、工具调用、reward 组装。
- **模型本地 worker groups**：负责 rollout、buffering、advantage 计算和 PPO-style 更新。

---

### **相比现有方法的优势**

| 对比维度 | 现有方法（如 MARTI、Dr. MAS、STRONGER-MAS） | UnityMAS-O |
|--------|------------------------------------------|-----------|
| **优化单位** | 单策略或分组样本（grouped samples） | 整个 **graph-structured multi-agent workflow** |
| **角色与模型关系** | 隐式绑定或固定共享 | 显式解耦，支持灵活的 `φ` 映射 |
| **奖励机制** | 全局或团队奖励，GRPO-style 分组归一化 | **role-specific、delayed、delta rewards** 可直接表达 |
| **系统架构** | 集中式训练或复杂分组 | **分离控制流与数据流**，heavy tensors 保留在模型本地 |
| **适用性** | 特定任务（如 coding、math） | **task-agnostic**，支持任意 workflow 定义 |

> ✅ **核心优势**：UnityMAS-O 是首个将 **multi-agent workflow 本身** 作为 RL 优化单元的通用框架，实现了 **workflow-as-policy** 的范式转变。

---

## 2. 核心实验方法和设置

### **使用的数据集**

| 任务类别 | 数据集 | 描述 |
|--------|-------|------|
| **Retrieval-Augmented QA** | **Natural Questions (NQ)** | 单跳开放域问答，测试检索与答案生成能力 |
| | **HotpotQA** | 多跳推理问答，需聚合多个证据段落 |
| **Reflective Code Generation** | **DeepCoder-style dataset** | 包含约 24K 编程题-测试对，来自 TACO-Verified、SYNTHETIC-1、LiveCodeBench v5，支持执行验证 |

---

### **实验设置和评估指标**

#### **Workflow 设计**
共实现四类 workflow：
- **Workflow A**: Parallel Retrieval (Search)
- **Workflow B**: Retrieve-Extract-Answer (Search)
- **Workflow C**: Iterative Search Loop (M-ASK)
- **Workflow D**: Reflective Verification Loop (Code)

#### **模型配置**
- 使用 **Qwen3-family** 模型（0.5B ~ 14B）
- 支持多种 `φ` 映射：full sharing、partial sharing、full separation

#### **评估指标**
| 任务 | 主要指标 | 辅助指标 |
|-----|---------|----------|
| QA/Search | **Normalized answer F1** | 各阶段格式正确率 |
| Code | **Held-out test all-passed rate**（严格通过所有测试） | 训练最终通过率、平均验证轮次 |

#### **训练流程**
- 异步 rollout + 中央 reward 组装
- PPO-style 更新，支持 per-model 学习率、optimizer、并行策略
- 对比训练前后性能（before RL vs best after RL）

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **QA 任务结果（Table 3 & Appendix A）**

| Dataset | Workflow | Model Size | Before RL | Best After RL | Abs. Gain | Rel. Gain |
|--------|--------|------------|-----------|--------------|----------|-----------|
| NQ | QD-Answer | 0.5B | 0.022 | **0.445** | +0.424 | +1943% |
| NQ | QD-Answer | 14B | 0.328 | **0.594** | +0.267 | +81% |
| HotpotQA | M-ASK | 1.5B | 0.127 | **0.525** | +0.398 | +313% |
| HotpotQA | M-ASK | 7B | 0.254 | **0.573** | +0.319 | +126% |

> 🔍 **观察**：小模型提升更显著（如 0.5B 模型相对提升超 1000%），表明 MARL 能有效弥补基础能力不足。

---

#### **Code 任务结果（Table 4）**

| Setting | Before RL | Best After RL | Abs. Gain | Rel. Gain |
|--------|-----------|----------------|-----------|-----------|
| 3xQwen3-4B | 0.255 | **0.686** | +0.431 | +169% |
| 3xQwen3-8B | 0.290 | **0.738** | +0.448 | +154% |

> 📈 **亮点**：在严格的 **held-out all-passed** 指标下仍取得巨大提升，说明训练改善了整个 plan-code-verify-reflect 循环的有效性。

---

### **与基线方法的对比结果**

- **无直接 baselines**：因 UnityMAS-O 是首个支持任意 workflow + MARL 的通用框架，多数 prior 工作（如 MMOA-RAG、MAO-ARAG）仅针对特定 RAG 架构且默认 full parameter sharing。
- **间接对比**：在 M-ASK workflow 上，UnityMAS-O 实现了与专用 MARL 方法相当甚至更好的性能，且支持更多参数共享配置。

---

### **消融实验结果**

#### **参数共享的影响（Figure 5）**
- 在 HotpotQA M-ASK 3B 设置下比较：
  - **4x3B independent**：更快收敛，最高达 **0.529 F1**
  - **3B shared**：稍慢但最终达到 **0.522 F1**
- **结论**：即使多个角色共享同一模型，也能有效训练，实现接近独立模型的性能，显著降低资源消耗。

#### **执行效率提升（Figure 7）**
- 在 code workflow 中，训练后 **平均验证轮次从 ~2.5 下降至 ~1.7**
- 表明 MARL 不仅提高最终正确率，还优化了内部 **early termination 策略**，提升了推理效率。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **MARL 可显著提升 multi-agent workflows 性能**  
   - 在 QA 和 code 任务上均观察到广泛且一致的增益，尤其对小模型和严格指标效果更明显。

2. ✅ **角色级信用分配至关重要**  
   - 通过 turn-level 或 delta rewards，可精准激励对最终结果有贡献的角色（如 verifier score 提升者）。

3. ✅ **逻辑角色与物理模型可有效解耦**  
   - 支持 full/partial/full separation 的 `φ` 映射，允许在资源、效率、性能之间灵活权衡。

4. ✅ **训练不仅提升最终输出，也优化中间行为**  
   - 如减少不必要的验证轮次、提高证据提取质量、增强协作一致性。

---

### **方法的局限性**

| 局限性 | 说明 |
|------|------|
| **依赖高质量 reward 函数设计** | 若 reward 信号稀疏或不准确，可能导致训练不稳定或次优。 |
| **中央控制器可能成为瓶颈** | 对于极长或高度并发的 workflow，controller 的调度开销可能上升。 |
| **尚未支持异构 agent 类型** | 当前主要面向 LLM-based agents，对非语言 agent（如视觉、机器人）支持有限。 |
| **训练成本较高** | 多模型并行训练需要大量 GPU 资源，尤其在 full separation 设置下。 |

---

### **未来工作方向**

1. **扩展至更多任务领域**  
   - 正在探索 **embodied agents**（ALFWorld）、**web interaction**（WebShop）、**software engineering**（SWE-bench）等场景。

2. **自动化 reward design**  
   - 结合 preference modeling 或 inverse RL 自动生成 role-specific rewards。

3. **动态 workflow 结构学习**  
   - 当前 workflow 图是静态的，未来可研究 **learnable control flow** 或 adaptive role invocation。

4. **跨任务迁移与复用**  
   - 利用 UnityMAS-O 作为“优化底座”，实现不同 task 间的 agent policy 迁移与共享。

5. **轻量化与边缘部署**  
   - 探索 distillation、quantization 等技术，使训练后的 multi-agent policy 可部署于资源受限环境。

---

> 💡 **总结一句话**：  
> **UnityMAS-O 成功地将“multi-agent workflow”从一个静态的推理管道，转变为一个可端到端训练的 RL policy，为构建更强大、更高效、更可控的 LLM-based multi-agent systems 提供了通用基础设施。**

</details>

---

### 6. [MicroSpec: Accelerating Speculative Decoding with Lightweight In-Context Vocabularies](https://arxiv.org/abs/2605.26444)

**Authors**: Zhiyang Chen, Daliang Xu, Yinyuan Zhang, Chenghua Wang, Mengwei Xu, Yun Ma  
**Category**: cs.CL  
**Published**: 2026-05-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.26444v1  

#### Abstract
Large language models typically employ vocabularies of over 100k tokens, which creates a major computational bottleneck at the final linear projection layer when performing speculative decoding. Current methods for vocabulary pruning depend on either fixed or coarse-grained sub-vocabularies, requiri...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：MicroSpec: Accelerating Speculative Decoding with Lightweight In-Context Vocabularies**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在 **Speculative Decoding (SD)** 中，尽管 draft model 能加速生成，但其最终的 **LM Head**（即输出层的线性投影）仍需对整个庞大的词汇表（如 Llama-3 的 128k、Qwen-2 的 152k）进行计算，导致该步骤成为推理过程中的主要瓶颈。现有词汇剪枝方法（如 FR-Spec、CORAL、DynaSpec）依赖静态或粗粒度子词表，通常需要保留约 30k 活跃词元才能维持 draft 质量，难以实现显著加速。

### **提出的新方法与思路**
本文提出了 **MicroSpec**，一种无需训练、基于上下文动态构建极小活跃词汇表的方法，核心思想是利用语言生成中的 **Temporal Locality**（时间局部性）——即下一个 token 很可能出现在当前上下文中或为其直接延伸。

MicroSpec 在每一步解码时动态构建一个仅包含 **2k–3k tokens** 的轻量子词表，来源包括：
- 上下文历史中的 token
- draft tree 中出现的候选 token
- target model 验证时输出的高概率 token

通过一个固定大小的滑动窗口机制维护候选流，并从中提取最新且唯一的 token 构成 $ L_t $。

### **相比现有方法的优势**
- **无需额外训练**：不依赖任何辅助模型或路由网络（router），完全无参数。
- **极致压缩**：平均活跃词汇从 27k–32k 压缩至 **<3k**，减少超过 **40×**。
- **系统级协同优化**：引入异步 gather 和 GPU-resident state management，将稀疏内存访问转化为密集 GEMM 运算，避免硬件效率损失。
- **即插即用**：可作为增强模块集成到 EAGLE-2 等主流 SD 框架中。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
采用 **SpecBench** 基准测试集，涵盖六类任务：
- **MT**（机器翻译）
- **Conv**（多轮对话）
- **RAG & QA**（检索增强问答）
- **Math**（数学推理）
- **Summ**（摘要生成）
- **Code**（代码生成，来自 HumanEval）

最大生成长度设为 1024，所有任务使用 greedy sampling。

### **实验设置与评估指标**
| 类别 | 设置 |
|------|------|
| **模型** | Llama-3-8B-Instruct, Llama-3.2-1B-Instruct, Qwen-2-7B-Instruct |
| **Draft Model** | 统一使用 EAGLE-2 结构 |
| **硬件平台** | 单张 NVIDIA H20Z GPU |
| **关键超参** | `K_pre = K_ver = 3`, `W_max = 3072`（滑动窗口上限） |

#### **评估指标**
- **End-to-End Speed (tokens/s)**：端到端生成吞吐量（含 prefill 和 decoding）
- **Average Acceptance Length**：每步平均被接受的 draft token 数量，反映 draft 质量
- **Draft Inference Latency**：draft 阶段耗时
- **消融实验**：验证各组件贡献

### **基线方法对比**
| 方法 | 类型 | 是否需训练 | 活跃词元数 |
|------|------|-----------|------------|
| **EAGLE-2** | 全词表 | 否 | 128k |
| **FR-Spec** | 静态高频词剪枝 | 否 | 32k |
| **CORAL** | 固定聚类 + FFN 路由器 | 是 | 32k |
| **DynaSpec** | KMeans 聚类 + MLP 路由器 | 是 | 27k |
| **MicroSpec (Ours)** | 动态上下文感知 | 否 | **<3k** |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **端到端速度提升（Llama-3-8B-Instruct）**
| 方法 | 平均速度 (tokens/s) | 相对 AR 加速比 |
|------|---------------------|----------------|
| Auto-Regressive (AR) | 172.6 | 1.00× |
| EAGLE-2 | 336.9 | 1.93× |
| FR-Spec | 369.7 | 2.12× |
| DynaSpec | 378.1 | 2.17× |
| **MicroSpec (Ours)** | **392.7** | **2.25×** ✅ |

> 在 Llama-3.2-1B 上表现更优，达到 **1.35×** 相对于 AR，远超 EAGLE-2 的 1.05×。

#### **平均 Acceptance Length**
| 方法 | Llama-3-8B | Llama-3.2-1B |
|------|-------------|---------------|
| EAGLE-2 | 3.80 | 3.16 |
| FR-Spec | 3.62 | 2.70 |
| **MicroSpec (Ours)** | **3.59** | **3.11** |

> 尽管词汇量仅为 3k vs 32k，acceptance length 几乎持平甚至优于部分基线，说明其上下文覆盖能力强。

#### **Draft 推理延迟降低**
- MicroSpec 相比 EAGLE-2 平均减少 **51.6%** draft 时间
- 相比已剪枝的 FR-Spec 也减少了 **20.3%**

#### **跨大词汇模型鲁棒性（Qwen-2-7B, Vocab=~152k）**
| 方法 | Overall Acceptance Length |
|------|----------------------------|
| EAGLE | 3.65 |
| FR-Spec | 3.44 |
| **MicroSpec** | **3.48** ✅ |

> 表明 MicroSpec 在更大词汇空间下依然保持高效覆盖能力。

---

### **消融实验结果（Ablation Study）**

| 配置 | 异步 Gather | 平均速度 (tokens/s) |
|------|------------|---------------------|
| Ctx. Only | √ | 313.3 |
| Ext. Only | × | 351.4 |
| Ctx. + Ext. (× Async.) | × | 364.1 |
| **Ctx. + Ext. (√ Async.)** | √ | **394.6** ✅ |

> 结论：
> - 上下文（prompt）与扩展（draft/verify）信息结合至关重要
> - **异步 gather 是实现性能飞跃的关键**，将稀疏操作转为密集 GEMM 并隐藏延迟

---

## **4. 关键结论和发现**

### **主要发现**
1. **语言生成具有强 Temporal Locality**：绝大多数正确 token 可由近期上下文和高概率候选覆盖，无需全局大词表。
2. **动态极小词表可行且高效**：仅用 **<3k tokens** 即可维持高质量 draft，打破“大词表=高性能”的固有认知。
3. **算法-系统协同设计必要**：单纯理论剪枝无效；必须通过 **asynchronous gathering + GPU-resident state** 解决稀疏访存瓶颈。
4. **MicroSpec 是通用增强模块**：可无缝集成进 EAGLE-2 等框架，带来 **1.12–1.32×** 的互补加速。

### **方法的局限性**
- 当前方法依赖于 target model 输出的 top-K 高概率 token，若其分布偏差较大可能影响候选质量。
- 滑动窗口机制虽高效，但对极长依赖任务（如跨文档推理）可能存在历史遗忘问题。
- 所有实验基于单卡环境，分布式场景下的扩展性尚未验证。

### **未来工作方向**
- 探索更智能的候选筛选策略（如基于语义相似度而非纯频率）。
- 将 MicroSpec 思想应用于其他 LLM 加速技术（如 early exiting、KV cache 压缩）。
- 支持 streaming 场景下的持续状态更新与内存管理优化。
- 在更多架构（如 MoE、decoder-only vs encoder-decoder）上验证泛化能力。

---

> ✅ **总结一句话**：  
> **MicroSpec 通过“上下文驱动的极小动态词表 + 系统级异步优化”，首次实现了无需训练即可将 Speculative Decoding 的 LM Head 计算压缩 40 倍以上，并取得 SOTA 端到端加速效果。**

</details>

---

### 7. [WINDQuant: Weight-Informed Neural Decision-Making for Global Mixed-Precision LLM Quantization](https://arxiv.org/abs/2605.26660)

**Authors**: Phong Nam Huu Nguyen, Khoi M. Le, Cong-Duy T Nguyen, Anh Tuan Luu, Thong Thanh Nguyen, Tho Quan  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.26660v1  

#### Abstract
Quantization is an effective approach to reduce the memory footprint and inference cost of large language models (LLMs), yet maintaining performance in the ultra-low-bit regime remains challenging. Existing post-training methods often suffer from severe accuracy degradation, while quantization-aware...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：WINDQuant: Weight-Informed Neural Decision-Making for Global Mixed-Precision LLM Quantization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **超低位量化（ultra-low-bit quantization）下的性能保持难题**：现有方法在低于3-bit的极端压缩下常导致严重精度下降。
- **静态或粗粒度精度分配的局限性**：传统方法（如PTQ、QAT）通常采用统一或层级别（layer-wise）的量化策略，忽略了权重矩阵内部细粒度敏感性的差异。
- **训练成本高昂**：Quantization-Aware Training (QAT) 虽能提升鲁棒性，但需要全模型重训练，计算开销大。

### **提出的新方法与创新思路**
- **WINDQuant**：一种基于强化学习（Reinforcement Learning, RL）的全局混合精度量化框架。
- **核心思想**：
  - 将量化建模为一个**序列决策问题**（sequential decision-making），在全局存储预算约束下，动态决定每个“列块”（column chunk）的比特宽度。
  - 引入**单元级（unit-level）精细控制**：将每层线性权重分解为多个列级chunk（例如大小为256），实现亚层级的灵活精度分配。
  - 使用 **PPO（Proximal Policy Optimization）算法** 学习最优分配策略，结合激活感知校准（activation-aware calibration）、轻量级局部量化器拟合和显式的有效比特会计机制。

### **相比现有方法的优势**
| 维度 | WINDQuant | 传统方法（PTQ/QAT） |
|------|-----------|---------------------|
| **粒度** | 列块级（fine-grained） | 层级或张量级（coarse-grained） |
| **策略生成方式** | 学习型、自适应分配 | 静态规则或启发式 |
| **训练需求** | 无需全模型重训练，仅优化轻量RL策略 | QAT需昂贵重训练；PTQ无训练但性能差 |
| **灵活性** | 支持从1-bit到8-bit及ternary（1.58-bit）等多种量化算子 | 多数支持固定格式 |
| **目标函数** | 显式建模全局比特预算与质量权衡 | 局部误差最小化为主 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **校准数据集（用于RL奖励计算）**：
  - `Open-Platypus`：64条样本，截断至长度256。
- **下游评估基准**：
  - **语言建模质量**：`WikiText-2`（Wiki2）测试集，报告 **Perplexity (↓)**。
  - **推理能力**：零样本常识推理任务，包括：
    - ARC-Easy / ARC-Challenge
    - BoolQ
    - PIQA
    - HellaSwag
    - Winogrande  
    报告各任务准确率及**平均准确率（Avg ↑）**。

### **实验设置**
- **模型范围**：LLaMA-3系列 —— `1B`, `3B`, `8B`, `70B` 参数模型。
- **目标比特**：约 **2-bit 混合精度**（effective average bit-width）。
- **量化粒度**：column chunk size $ G = 256 $。
- **硬件平台**：单张 NVIDIA H200 GPU（140GB VRAM）。
- **训练配置**：
  - PPO优化器，3个epoch每rollout，学习率 $5 \times 10^{-4}$。
  - Curriculum learning：逐步降低目标比特（如 3.0 → 2.5 → 2.0）以稳定探索。
  - Salient weight protection：前3%最显著权重保留为INT8。

### **评估指标**
- 主要指标：
  - **Average Accuracy ↑**（5项推理任务宏平均）
  - **WikiText-2 Perplexity ↓**
  - **Effective Average Bit-width**
- 效率指标（Table 3）：
  - **Optimization Time (小时)**
  - **Peak GPU Memory Usage (GB)**

### **基线方法对比**
涵盖三大类主流方法：
| 类别 | 方法列表 |
|------|--------|
| **Post-Training Quantization (PTQ)** | AWQ, GPTQ, OmniQ, SpinQuant |
| **Quantization-Aware Training (QAT)** | LLM-QAT, EfficientQAT |
| **Vector Quantization** | AQLM, GPTVQ |
| **Ultra-Low-Bit 新方法** | SignRound, TesseraQ |
| **长周期参考方法** | ParetoQ（作为高质量高成本对照） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2 & 4）**

| Model | Method | #Bits | Avg Acc ↑ | Wiki2 PPL ↓ |
|-------|--------|-------|----------|-------------|
| LLaMA-3-1B | **WINDQuant** | 2.01 | **49.30** | **19.64** |
| | EfficientQAT | 2.12 | 46.58 | 29.32 |
| | SignRound | 2.00 | 45.09 | 4.96×10⁴ |
| LLaMA-3-3B | **WINDQuant** | 1.99 | **61.98** | **10.40** |
| | AQLM | 2.27 | 61.86 | 11.29 |
| | SignRound | 2.00 | 53.13 | 1.01×10³ |
| LLaMA-3-1-8B | **WINDQuant** | 1.96 | **70.22** | **7.90** |
| | AQLM | 2.27 | 69.04 | 6.90 |
| | EfficientQAT | 2.12 | 55.52 | 14.70 |
| LLaMA-3-70B | **WINDQuant** | 2.02 | **73.46** | **5.10** |
| | AQLM | 2.27 | 70.04 | 4.57 |
| | EfficientQAT | 2.12 | 38.72 | 1.2×10⁷ |

> ✅ **结论**：WINDQuant 在所有规模上均取得**最佳或接近最优的平均准确率**，同时维持稳定的语言建模性能（PPL < 10），而多数PTQ方法出现PPL爆炸（>10⁴），表明其不稳定。

### **与基线方法的对比结果**
- **优于所有标准PTQ方法**（AWQ/GPTQ等）：这些方法在2-bit下普遍崩溃（PPL > 10⁴），说明直接应用不可行。
- **优于QAT方法**：
  - EfficientQAT虽有一定效果，但在8B和70B上仍显著落后于WINDQuant（70B差距达34.74点）。
  - LLM-QAT也表现出严重的PPL退化。
- **优于向量量化方法**：
  - AQLM是唯一接近的竞争者，在8B上PPL略优（6.90 vs 7.90），但**平均准确率更低**（69.04 vs 70.22）且使用更高比特（2.27 vs 1.96）。
- **效率远胜重训练方法**：
  - 见下表。

### **效率对比（Table 3）—— LLaMA-3.1-8B**
| Method | Time (h) | GPU Mem (GB) | Avg Acc ↑ | Wiki2 ↓ |
|--------|----------|---------------|------------|---------|
| **WINDQuant** | **49** | **17.1** | **70.2** | **7.9** |
| EfficientQAT | 14 | 8.1 | 44.8 | 14.7 |
| LLM-QAT | 0.5 | 126.9 | 42.0 | 7137 |
| **ParetoQ**（引用值） | **1777** | **82.4** | 70.9 | 8.0 |

> 🔍 **解读**：
> - WINDQuant 在合理时间内完成（<2天），内存占用极低（<18GB）。
> - ParetoQ 虽然性能相当，但耗时近 **74天**，难以实用。
> - WINDQuant 实现了**高质量与高效优化之间的最佳平衡**。

### **消融实验结果（Appendix B）**

#### **(1) 与启发式分配器对比（Table 8）**
| Method | Bitwidth | Perplexity |
|--------|----------|-----------|
| Heuristic Allocator | 2.182 | 8.875 |
| **WINDQuant (PPO)** | **2.066** | **8.756** |

> 📌 **发现**：即使共享相同的量化算子、保护机制和bit accounting，**RL策略仍显著优于基于敏感性排序的贪心启发式方法**，证明其学习到了更优的全局协调分配模式。

#### **(2) Chunk Size 敏感性分析（Table 9）**
| G | Units/ep | PPL | BW | Time(h) |
|----|----------|------|-----|--------|
| 128 | 88 | 8.78 | 2.03 | 20.4 |
| **256** | **74** | **8.87** | **2.02** | **14.2** |
| 512 | 52 | 10.03 | 2.03 | 7.2 |

> 📌 **结论**：$ G=256 $ 是性能与效率的最佳折中点。过小增加开销，过大损失表达力。

#### **(3) Salient Protection 影响（Figure 3）**
- 提高保护比例可改善PPL，但也提高平均比特。
- 选择 **3%** 作为平衡点，在保持 ~2-bit 的前提下获得良好稳定性。

#### **(4) RL训练收敛性（Figure 4）**
- PPO在约 **200 episodes** 后趋于收敛，后续收益有限。
- 训练代理（proxy）PPL 与最终 Wiki2 PPL 高度相关（Pearson r=0.997），验证了代理的有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **细粒度混合精度分配至关重要**：通过在列块级别进行自适应比特分配，可以显著提升超低位量化的性能边界。
2. **强化学习适合全局预算控制**：将量化视为带约束的序列决策问题，PPO能够有效学习如何在有限比特预算内最大化模型质量。
3. **无需重训练即可达到竞争性能**：WINDQuant避免了QAT的高昂训练成本，仅通过轻量级策略优化即实现了媲美甚至超越部分QAT方法的效果。
4. **冗余结构可被高效利用**：大多数权重可安全地压缩至1-bit或1.58-bit（见 Figure 2，占比约75%），仅少数敏感单元需保留较高精度。
5. **组件协同增强鲁棒性**：activation-aware salient protection、fallback机制、curriculum learning共同保障了极端压缩下的稳定性。

### **方法的局限性**
1. **仅限 weight-only 量化**：未对 activations 或 KV Cache 进行量化，因此实际部署内存节省可能受限。
2. **依赖校准代理信号**：训练使用的是轻量级PPL proxy，可能无法完全反映下游任务表现或长上下文行为。
3. **非纯RL策略**：系统包含多种工程技巧（如保护、fallback、masking），使得难以完全归因于RL本身的作用。
4. **架构泛化待验证**：目前实验集中在 LLaMA 家族，其他Transformer变体（如Mistral、Phi）尚未充分测试。
5. **单GPU设定限制扩展性研究**：更大规模分布式场景下的表现未知。

### **未来工作方向**
- 扩展至 **activation 和 KV cache 的联合量化**，实现端到端低比特推理。
- 探索 **多目标RL优化**，同时考虑延迟、能耗与精度。
- 设计 **更通用的状态表示与动作空间**，适配不同LLM架构。
- 研究 **在线自适应量化**，根据输入动态调整精度分布。
- 构建 **更可靠的训练代理函数**，减少与真实下游性能的gap。

---

> 💡 **总体评价**：  
> WINDQuant 提出了一种新颖且实用的视角——将LLM量化看作“**资源分配决策问题**”，而非单纯的数值逼近问题。它成功展示了**强化学习作为自动化压缩控制器的巨大潜力**，特别是在追求极致压缩比的应用场景中，提供了一条兼顾性能与效率的新路径。

</details>

---

### 8. [Generalist Graph Anomaly Detection via Prototype-Based Distillation](https://arxiv.org/abs/2605.26857)

**Authors**: Yiming Xu, Zihan Chen, Zhen Peng, Song Wang, Bin Shi, Bo Dong, Chao Shen  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.26857v1  

#### Abstract
Driven by the pressing demand for graph anomaly detection (GAD) in high-stakes domains, the generalist GAD paradigm, which trains a single detector transferable across new graphs, has recently gained growing attention. However, existing methods often rely on scarce and costly annotations for trainin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Generalist Graph Anomaly Detection via Prototype-Based Distillation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Graph Anomaly Detection (GAD)** 方法在跨图泛化（cross-graph generalization）方面存在严重局限：
- 多数方法需为每个新图重新训练，计算成本高；
- 现有的“通用型”（generalist）GAD 模型依赖昂贵且稀缺的异常标签进行监督训练；
- 异常模式具有开放性和动态演化特性，固定标注难以覆盖所有异常类型。

本文提出首个**完全无监督的通用型 GAD 框架 ProMoS**，旨在实现无需任何标签、零样本迁移即可检测新图中异常的目标。

---

### 提出了什么新方法或新思路

作者提出了 **ProMoS**（Prototype-guided Mixture-of-Students），其核心思想包括：

#### （1）基于知识蒸馏（Knowledge Distillation, KD）的正常性建模
- 利用一个预训练的、冻结的 **self-supervised GNN** 作为教师模型（Teacher），提取丰富的正常性先验（normality priors）。
- 学生模型通过蒸馏学习这些先验，避免从零开始学习，提升效率与泛化能力。

#### （2）混合学生架构（Mixture-of-Students, MoS）
- 包含一个**共享分支**（shared branch）捕捉全局规律；
- 和多个轻量级**个性化分支**（personalized branches），通过稀疏路由（sparse Top-K routing）激活，以高效建模多样化的局部正常模式。
- 平衡了表达力与推理效率。

#### （3）原型引导的软标签蒸馏（Prototype-guided Soft-Label Distillation）
- 在可学习的**语义原型空间**（prototype space）中对齐师生输出。
- 教师和学生的预测被映射到一组聚类初始化的原型上，生成软标签分布。
- 蒸馏目标是最小化 KL 散度，使学生模仿教师在高层语义层面的行为，而非细粒度特征匹配，增强跨图可迁移性。

#### （4）差异感知的承诺与精炼机制（Discrepancy-aware Commitment & Refinement）
- 引入可靠性加权机制，抑制异常节点带来的误导梯度；
- 承诺损失（commitment loss）将教师表示拉向稳定原型锚点；
- 精炼损失（refinement loss）持续更新原型以编码高质量语义结构。

#### （5）零样本推理机制
- 推理阶段无需微调，结合两种信号打分：
  - **蒸馏偏差**（distillation bias）：师生软标签之间的 KL 差异；
  - **几何偏差**（geometric deviation）：嵌入与其量化原型的距离。
- 最终异常得分融合两者：  
  $$
  s_i = \sum_{b\in B} \left[ \text{KL}(q^b \| s^b) + \lambda (\|\Delta h_b\| + \|\Delta z_b\|) \right]
  $$

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **训练范式** | 完全无监督，不依赖任何标签，降低部署门槛 |
| **跨图泛化** | 零样本迁移至未见图，支持 in-domain 与 out-of-domain 测试 |
| **效率** | MoS 架构 + 稀疏激活，训练快于 GCN 4.8×，推理快 1.4× |
| **可扩展性** | 推理时间随边数呈亚线性增长（$T \propto |E|^{0.3}$），适合大规模图 |
| **鲁棒性** | 对不同 SSL 教师模型（如 GCA, GraphCL, DGI）表现稳健 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
共使用 **15 个真实世界图数据集**，涵盖多种领域：

| 类别 | 数据集 |
|------|-------|
| 引用网络（注入异常） | Cora, CiteSeer, ACM, PubMed |
| 社交博客（注入异常） | BlogCatalog, Flickr |
| 社交网络（真实异常） | Facebook, Weibo, Reddit, Questions |
| 商业评论（真实异常） | YelpChi |
| 合著网络（注入异常） | CoAuthor CS |
| 商品共购网络（注入异常） | Amazon Photo |
| 众包平台（真实异常） | Tolokers |
| 金融交易网络（真实异常） | T-Finance |

- **训练集**：PubMed, Flickr, Questions, YelpChi
- **测试集**：其余 11 个图（均为未见过的图）

---

### 实验设置和评估指标

#### 评估协议
- **Zero-shot setting**：在训练图上训练后，直接迁移到测试图，**不进行任何微调或参数调整**。
- 所有方法均采用多图预训练策略，确保公平比较。

#### 评估指标
- **AUROC**（Area Under ROC Curve）
- **AUPRC**（Area Under Precision-Recall Curve）
- 报告五次随机种子运行的平均值 ± 标准差

#### 实现细节
- 教师模型：GCA（默认）、GraphCL、BGRL、DGI、GraphMAE
- 特征维度统一为 64
- 开源代码地址：[https://github.com/yimingxu24/ProMoS](https://github.com/yimingxu24/ProMoS)

---

### 基线方法对比

分为两大类共 12 个 baseline：

#### 监督方法（Supervised）
- GCN, GAT（仅预训练）
- BGNN, BWGNN, GHRN
- ARC, UNPrompt, AnomalyGFM（generalist GAD 方法）

#### 无监督方法（Unsupervised）
- DOMINANT（重构）
- CoLA（对比学习）
- HCM-A（跳数预测）
- TAM（亲和性最大化）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

#### AUROC 结果（Table 2）
- ProMoS 在 **11 个测试图中的 9 个取得最佳 AUROC**，平均领先最强基线 DOMINANT 达 **14.12%**。
- 在 **Cora 上达到 84.56%**，远超第二名 TAM（62.02%）；
- 在 **T-Finance 上达 71.62%**，显著优于多数方法（部分OOM）。

#### AUPRC 结果（Table 3）
- 在 **9/11 图上排名第一**，2 图排名第二；
- 在 **Cora 上 AUPRC 达 46.48%**，相比 TAM（11.18%）提升超过 **33.73%**，说明在极度不平衡场景下仍具高精度。

---

### 与基线方法的对比结果

| 对比维度 | 观察结果 |
|--------|---------|
| vs. 监督方法 | 尽管后者利用标签训练，但在跨图设置下泛化差；ProMoS 作为无监督方法反而全面超越 |
| vs. 传统无监督方法 | 如 CoLA、TAM 在同图设置下有效，但跨图时性能下降超 24%，而 ProMoS 保持稳定 |
| vs. Few-shot generalist 方法（如 ARC） | 即便允许访问 10 个带标样例，ARC 平均 AUROC（75.03%）仍低于 ProMoS（77.16%） |

> ✅ **结论**：ProMoS 不仅免去了标签依赖，还实现了更优甚至超越有监督方法的性能。

---

### 消融实验结果（Table 4）

| 变体 | 描述 | AUROC 下降幅度（平均） |
|------|------|---------------------|
| `w/o PSD` | 移除原型软标签蒸馏 | ↓19.14% → 性能接近随机猜测 |
| `w/o DCR` | 移除差异感知约束 | ↓5.26% |
| `w/o SSL` | 替换教师为随机 GCN（从头学） | ↓5.53%（且 T-Finance OOM） |

> 🔍 **分析**：
- 原型蒸馏是核心，缺失后模型无法捕获可迁移语义；
- DCR 显著提升教师输出稳定性；
- 预训练教师至关重要，替代方案性能骤降。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **无监督也能实现强泛化**：  
   ProMoS 验证了仅通过建模“正常性”即可有效识别异常，并能在完全未见的图上实现 SOTA 表现。

2. ✅ **高层语义对齐优于低层特征匹配**：  
   原型空间中的软标签蒸馏比实例级或特征级对齐更具可迁移性，缓解了图间异质性问题。

3. ✅ **混合专家结构提升表达能力**：  
   MoS 架构通过共享+个性化的分工，兼顾效率与多样性，在理论证明中 MoS 的期望误差 ≤ 任一单个学生。

4. ✅ **零样本推理可行且高效**：  
   推理无需重训，速度快、内存友好，适用于大规模实时应用。

5. ✅ **框架具有插件式兼容性**：  
   可适配多种 SSL 教师（GCA/DGI/GraphCL等），性能稳定，便于未来升级。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| 依赖高质量预训练教师 | 若教师本身在训练图上学得不好，会影响整体性能 |
| 原型数量需调参 | 虽然敏感度不高，但仍需合理选择 $M_b$ |
| 对极端稀疏图适应性未知 | 当前实验集中在中等到大图规模，极小图表现未验证 |
| 无法处理图级异常 | 本工作聚焦于 node-level GAD |

---

### 未来工作方向

1. **拓展至图级异常检测**（Graph-level GAD）
2. **引入动态图支持**，应用于时序图或流式图场景
3. **探索自适应原型机制**，让原型数量随数据自动调整
4. **结合 LLM 提升文本属性图的理解能力**（尤其适用于 Weibo、YelpChi 等文本丰富图）
5. **进一步优化 MoS 路由机制**，设计任务感知的专家选择策略

---

## 总结

> 🏁 **ProMoS 是首个真正意义上的无监督、零样本、通用型图异常检测框架**，它通过“原型引导的知识蒸馏 + 混合学生架构”，成功实现了从大规模无标签图中学习可迁移的正常性模式，并在 11 个真实图上全面超越监督与无监督基线，同时具备卓越的效率与可扩展性。该工作为构建低成本、高泛化、易部署的工业级 GAD 系统提供了新的范式路径。

</details>

---

### 9. [The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence](https://arxiv.org/abs/2605.26494)

**Authors**: MiniMax,  :, Aili Chen, Aonian Li, Baichuan Zhou, Bangwei Gong, Binyang Jiang, Boji Dan, Changqing Yu, Chao Wang, Cheng Ma, Cheng Zhong, Cheng Zhu, Chengjun Xiao, Chengyi Yang, Chengyu Du, Chenyang Zhang, Chi Zhang, Chuangyi Huang, Chunhao Zhang, Chunhui Du, Chunyu Zhao, Congchao Guo, Da Chen, Deming Ding, Dianjun Sun, Dongyu Zhang, Enhui Yang, Fei Yu, Guang Zheng, Guodong Zheng, Guohong Li, Haichao Zhu, Haigang Zhou, Haimo Zhang, Han Ding, Hao Zhang, Haohai Sun, Haolin Lyu, Haonan Lu, Haoyu Wang, Huajie Shi, Huiyang Li, Jiacheng Chen, Jian Zhang, Jiaqi Zhuang, Jiaren Cai, Jiaxin Pan, Jiayao Li, Jiayuan Song, Jichuan Zhang, Jie Wang, Jihao Gu, Jin Zhu, Jingwei Dong, Jingyang Li, Jingyu Zhang, Jingze Zhuang, Jinhao Tian, Jinli Liu, Jinyi Hu, Jun Tao, Jun Zhang, Junbin Ruan, Junhao Xu, Junjie Yan, Junteng Liu, Junxian He, Kang Xu, Ke Ji, Ke Yang, Kecheng Xiao, Keyu Duan, Keyu Li, Le Han, Letian Ruan, Li Yuan, Lianfei Yu, Liheng Feng, Lijie Mo, Lin Li, Lingye Bao, Lingyu Yang, Lingyuan Zhou,  Loki, Lu Chen, Lunbin Ceng, Ming Li, Ming Zhong, Mingliang Tao, Mingyuan Chi, Mujie Lin, Nan Hu, Ningxin Chen, Peiyin Zhu, Peng Gao, Pengcheng Gao, Pengfei Li, Penglin Li, Pengyu Zhao, Qibin Ren, Qidi Xu, Qihan Ren, Qile Li, Qin Wang, Quanliang Chen, Qunhong Ceng, Rong Tian, Rui Dong, Ruitao Leng, Ruize Zhang, Shanqi Liu, Shaoyu Chen, Sheng Jia, Shun Yao, Shuoran Zhao, Shuqi Yu, Sichen Li, Sicheng Pan, Songquan Zhu, Tengfei Li, Tian Xie, Tiancheng Qin, Tianrun Liang, Wei Liu, Weiqi Xu, Weitao Li, Weixiang Chen, Weiyu Cheng, Weiyu Zhang, Wenhu Chen, Wenqian Zhao, Xiancai Chen, Xiangjun Song, Xiangyuan Wang, Xiao Luo, Xiao Su, Xiaobo Li, Xiaodong Han, Xiaojie Wu, Xihao Song, Xingyi Han, Xinyu Guan, Xuan Lu, Xun Zou, Xunhao Lai, Xutong Li, Yan Gong, Yang Wang, Yang Xu, Yangsen Wang, Ye Tang, Yicheng Chen, Yinran Qiu, Yiqi Shi, Yiting Guo, Yiwen Huang, Yixuan Wang, Yongyi Hu, Yu Gao, Yu Zhang, Yuanxiang Ying, Yuanzhen Zhang, Yubo Wang, Yuchen Song, Yufeng Yang, Yuhang Meng, Yuhang Miao, Yuhao Li, Yujie Liu, Yulin Hu, Yunan Huang, Yunji Li, Yunyi Huang, Yusen Zhang, Yusu Hong, Yutao Xie, Yutong Zhang, Yuwen Liao, Yuxuan Shi, Yuze Wenren, Zebin Li, Zehan Li, Zejian Luo, Zeyu Jin, Zeyuan Sun, Zhanpeng Zhou, Zhaochen Su, Zhendong Li, Zhengmao Zhu, Zhengyuan Peng, Zhenhua Fan, Zhi Zhang, Zhichao Xu, Zhiheng Lv, Zhikang Xu, Zhitao He, Zhiwei He, Zhongyuan Li, Zibo Gao, Zijia Wu, Zijian Song, Zijian Zhou, Zijun Sun, Zishan Huang, Ziying Chen, Ziyue Ge  
**Category**: cs.AI  
**Published**: 2026-05-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.26494v1  

#### Abstract
We introduce the MiniMax-M2 series, a family of Mixture-of-Experts language models built around the principle that mini activations can unleash maximum real-world intelligence. The flagship M2 contains 229.9B total parameters with only 9.8B activated per token. Designed end-to-end for agentic deploy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大语言模型（LLMs）正从短对话向**长周期、多步骤的智能体任务**（agentic workflows）迁移，例如代码开发、网页操作、工具调用等。这一转变带来了两大挑战：
- **效率瓶颈**：超长上下文导致训练和推理成本极高，尤其在生产部署中难以承受。
- **复杂任务需求**：真实场景要求模型解决高难度、高风险的任务，如生产级软件工程和办公自动化。

传统密集模型或粗粒度 MoE 架构难以兼顾**高性能与低激活参数**之间的平衡。

---

### 提出的新方法与核心思想
MiniMax-M2 系列提出一个核心设计原则：**Mini Activations Unleash Max Real-World Intelligence**（小激活释放最大现实智能），通过以下三大技术支柱实现：

#### （1）**Agent-Driven 数据管道**
- 构建了面向**agentic coding**（编码）、**agentic cowork**（协作办公）和**reasoning & knowledge**（推理与知识）的大规模、可验证轨迹数据集。
- 每个任务都配有**可执行环境**（如 Docker）和**对齐产出物的奖励机制**（artifact-aligned reward），确保训练信号真实可靠。
- 引入高质量反馈机制（如可执行验证、judge-model 审查），显著提升基础模型潜力的释放。

#### （2）**Forge：面向智能体的强化学习系统**
- 一种可扩展的、原生支持智能体的 RL 框架，专为处理**长周期、多步决策**而设计。
- 支持**白盒与黑盒智能体统一训练**，解耦训练、推理与智能体逻辑。
- 集成关键技术：
  - **Windowed-FIFO 调度**：缓解长尾任务阻塞，平衡吞吐与分布一致性。
  - **Prefix-tree Merging**：合并共享前缀以加速训练，最高提速 40×。
  - **推理优化**：结合 MTP（Multi-Token Prediction）用于 speculative decoding，提升生成效率。

#### （3）**自我演化能力（Self-Evolution）——M2.7 的初步实现**
- M2.7 可**自主诊断失败的训练运行**，修改自身智能体框架（scaffold），并进行多轮自我改进。
- 在 ML-engineering 任务上实现了闭环迭代，减少了人类干预，突破了前沿模型开发中最昂贵的人工瓶颈之一。

---

### 相比现有方法的优势
| 维度 | MiniMax-M2 系列优势 |
|------|------------------|
| **激活参数量** | 仅 **9.8B 激活参数**（总参 229.9B），远低于主流闭源模型（如 GPT-5.4、Gemini 3.1 Pro） |
| **计算效率** | 利用 MoE + MTP + 推理优化，在低激活下实现接近更大模型的性能 |
| **训练稳定性** | Forge 系统实现稳定的大规模 RL 扩展，支持异构智能体架构 |
| **任务真实性** | 所有训练轨迹基于可执行环境与可信奖励，避免幻觉与无效行为 |
| **演进能力** | 首次展示模型能**自主调试与修改自身 scaffold**，迈向自进化 AI |

---

## 2. 核心实验方法和设置

### 使用的数据集
涵盖四大类任务领域，均强调**环境接地性**（environment-grounded）和**可验证性**：

#### （1）**Agentic Coding**
- `SWE-bench Pro`：工业级仓库修复任务
- `SWE-bench Multilingual`：跨语言代码编辑
- `Multi-SWE-bench`：多仓库任务迁移
- `Terminal-Bench 2.0`：终端系统操作任务
- `MLE-Bench Lite`：机器学习工程全流程自动化

#### （2）**Agentic Cowork**
- `BrowseComp` / `Wide Search` / `RISE`：开放网页搜索与综合研究
- `GDPval-AA`：经济价值导向的办公任务（报告、备忘录等）
- `Toolathlon`：异构工具使用能力
- `MM Claw` / `MEWC v2` / `Finance Modeling Pro`：Excel 表格操作与金融建模

#### （3）**Reasoning & Knowledge**
- `AIME 2026`：竞赛数学
- `GPQA-Diamond`：研究生级别科学问答
- `SciCode`：科研代码生成
- `IFBench`：指令遵循
- `AA-LCR`：长文本检索与推理
- `HLE`（Humanity's Last Exam）：无工具开放知识挑战

#### （4）**内部合成数据集**
- `NL2Repo`：自然语言到完整代码库生成
- `VIBE-Pro` / `HyperTask`：端到端全栈应用开发
- `Terminal-Gym`：自动合成终端任务

---

### 实验设置与评估指标
- **模型配置**：
  - M2.7：229.9B 总参，9.8B 激活/Token，62 层 MoE Transformer，256 专家，sigmoid gating
  - 上下文长度：**192K tokens**
  - MTP 模块：支持 speculative decoding
- **训练流程**：
  - 预训练：29.2T tokens
  - SFT + Agent-native RL（使用 Forge 系统）
- **推理设置**：
  - 温度 = 1.0，top_p = 0.95
  - 启用 **interleaved thinking**（交错思考）协议
  - 多数任务运行 3–4 次取平均

### 基线方法对比
与当前最强的闭源模型对比：
- `Claude Opus 4.6`
- `Claude Sonnet 4.6`
- `GPT-5.4`
- `Gemini 3.1 Pro`

所有模型在各自最优推理模式下测试（如 Opus 使用 extended thinking，GPT-5.4 使用 high reasoning effort）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 4 和 Figure 1）

| 类别 | 基准 | M2.7 | 最优基线 |
|------|-------|--------|----------|
| **Agentic Coding** | | | |
| SWE-bench Pro | 56.2 | 57.7 (GPT-5.4) |
| SWE-bench Multilingual | 76.5 | 77.8 (Opus) |
| Multi-SWE-bench | **52.7** ✅ | 51.3 (M2.5) |
| Terminal-Bench 2.0 | 57.0 | 75.1 (GPT-5.4) |
| MLE-Bench Lite | **66.6%** ✅ | 75.7 (Opus) |
| **Application Development** | | | |
| VIBE-Pro | 55.6 | 56.1 (Sonnet) |
| HyperTask | **67.6** ✅ | 75.7 (Opus) |
| **Cowork - Search** | | | |
| BrowseComp | 77.8 | 84.0 (Opus) |
| Wide Search | 75.2 | 79.4 (Opus) |
| RISE | **64.3** ✅ | 68.5 (Opus) |
| **Cowork - Office** | | | |
| GDPval-AA | 50.0 | 58.0 (GPT-5.4) |
| Toolathlon | 46.3 | 54.6 (GPT-5.4) |
| MM Claw | 62.7 | 75.4 (Opus) |
| **Reasoning & Knowledge** | | | |
| AIME 2026 | **94.2** ✅ | 97.0 (GPT-5.4) |
| GPQA-Diamond | **89.8** ✅ | 94.1 (Gemini) |
| IFBench | **76.0** ✅ | 77.1 (Gemini) |
| MMLU-Pro | 81.8 | 91.2 (Gemini) |

> ✅ 表示在该任务上达到或超过部分闭源模型，且远优于前代 M2.5

---

### 与基线方法的对比结果
- **总体表现**：尽管激活参数仅为 ~10B，M2.7 在多个任务上**接近甚至超越**激活参数高出一个数量级的闭源模型。
- **相对 M2.5 的提升**：
  - `BrowseComp`：+27.8 → **+33.8 提升**
  - `GDPval-AA`：+15.0
  - `MLE-Bench Lite`：+15.1（绝对值）
  - `AIME 2025`：+16.0
- **优势领域**：
  - **深度搜索**（RISE +14 pts）
  - **工具使用**（Toolathlon +8 pts）
  - **自主 ML 工程**（MLE-Bench Lite +15 pts）
- **推理类任务稳步提升**：得益于多轴 scaling（query/response/training-side）

---

### 消融实验结果（Ablation Studies）

#### （1）MoE 设计消融（Table 1）
| 配置 | HumanEval | MATH |
|------|-----------|--------|
| Baseline (32 exp, top-2) | 29.7 | 19.6 |
| + MTP | 30.1 | 21.3 |
| + Fine-Grained Experts (128 exp, top-8) | **32.5** ✅ | **24.1** ✅ |

👉 结论：**细粒度专家 + MTP** 显著提升推理与编码能力。

#### （2）Attention 架构比较（Table 2 & 3）
- **预训练阶段**：Hybrid SWA 在长上下文检索（RULER 128K）上明显劣于 Full Attention（↓18 pts）
- **SFT 后**：SWA 在短任务（<32K）表现尚可，但在长上下文任务（如 SWE-verified, Terminal-Bench）全面落后
- 👉 结论：**Full Attention 更适合 agentic 任务**，尤其涉及多跳推理与长期依赖

#### （3）MTP 模块有效性
- 加入 MTP 后在 MATH、HumanEval 等任务上均有提升
- 扩展至 K=3 并采用 weight copying 初始化，收敛更快且不影响主模型表示

---

## 4. 关键结论和发现

### 主要发现
1. **“Mini Activations” 可实现 “Max Intelligence”**  
   通过高质量 agent-native 数据 + 高效 RL 系统 + 自我演化机制，即使仅激活 9.8B 参数，也能在多种 agentic 任务上媲美更大模型。

2. **数据质量 > 数据规模**  
   可执行环境 + artifact-aligned reward 是释放模型潜力的关键。**验证信号的真实性**（如测试通过、运行成功）比单纯增加数据量更重要。

3. **Interleaved Thinking 至关重要**  
   允许模型在每一步保留完整的 reasoning state，显著提升多步规划、错误修正与长期一致性。

4. **Forge 实现稳定的大规模 RL 扩展**  
   通过解耦架构 + windowed-FIFO + prefix-tree merging，解决了长周期 agent RL 中的效率与稳定性难题。

5. **自我演化初具雏形**  
   M2.7 能自主调试训练流程、修改 scaffold，并在 ML 工程任务上实现持续优化，标志着向**自进化 AI**迈出第一步。

---

### 方法的局限性
- **部分任务仍落后于顶级闭源模型**：如 Terminal-Bench、MM Claw 等，说明在极端复杂工具链或 GUI 操作上仍有差距。
- **依赖高质量合成数据**：虽然引入了 agent-as-a-verifier，但数据生成过程仍高度依赖专家设计与强 teacher model。
- **自我演化尚处早期**：目前仅限于特定任务闭环，尚未实现通用意义上的“自我重写”。

---

### 未来工作方向
- **继续扩展三个核心轴线**：
  - **Data**：构建更复杂的 agentic 场景（如网络安全 CVE-Factory）
  - **RL System**：进一步优化 Forge 的可扩展性与鲁棒性
  - **Self-Evolution**：推动模型从“辅助迭代”走向“主导研发”
- **探索 sub-quadratic attention**：随着上下文增长，将重新评估高效 attention 架构的可能性
- **推进“零人工干预”的 Anything2Docker 系统**：实现完全自动化的任务到环境映射

---

> **总结一句话**：  
> MiniMax-M2 系列证明了，**通过精心设计的 agent-native 数据、高效的 RL 系统与初步的自我演化机制，极低的激活参数也能释放出接近前沿水平的真实世界智能**，为下一代 agentic AI 提供了一条高效、可持续的发展路径。

</details>

---

### 10. [MobileExplorer: Accelerating On-Device Inference for Mobile GUI Agents via Online Exploration](https://arxiv.org/abs/2605.26546)

**Authors**: Runxi Huang, Liyu Zhang, Shengzhong Liu, Xiaomin Ouyang  
**Category**: cs.AI  
**Published**: 2026-05-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.26546v1  

#### Abstract
Mobile graphical user interface (GUI) agents enable AI models to autonomously operate smartphones on behalf of users. However, most existing systems focus primarily on optimizing task accuracy and rely on cloud-hosted models for inference, which introduces privacy concerns and network-dependent late...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*MobileExplorer: Accelerating On-Device Inference for Mobile GUI Agents via Online Exploration*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前大多数 **Mobile GUI Agent** 系统依赖于云端运行的 **VLM**（Vision-Language Models）进行推理，虽然提升了任务准确率，但也带来了两大核心挑战：

- **隐私风险**：需要上传用户的界面截图和操作数据到云端。
- **延迟问题**：网络传输 + 云端推理导致端到端延迟高，难以满足实时交互需求。

因此，**全设备端部署**（fully on-device deployment）成为迫切需求。然而，直接在移动设备上运行 VLM 推理面临严重性能瓶颈，尤其是 **VLM 推理耗时长**，而 UI 操作本身非常轻量，造成大量“空闲等待”时间被浪费。

---

### 🚀 提出的新方法与创新思路

作者提出 **MobileExplorer** —— 一种全新的、面向设备端的视觉型 GUI Agent 框架，其核心思想是：

> 利用 VLM 推理过程中的长延迟窗口，在本地并行执行 **轻量级在线探索**（online exploration），主动探测 UI 元素以收集上下文信息，并将这些信息作为提示注入后续推理步骤中。

#### 主要创新点包括：

1. **并行化执行流水线设计**
   - 打破传统“感知 → 推理 → 动作”的串行流程。
   - 在 VLM 推理的同时，系统利用空闲时间对当前屏幕上的候选 UI 元素进行点击试探，探索潜在路径。

2. **任务相关性驱动的探索策略**（Task Relevance-driven Exploration）
   - 使用轻量级文本嵌入模型（如 Sentence-BERT）计算 UI 元素描述与任务指令之间的语义相似度。
   - 优先选择语义相关且可点击的元素进行探索，避免盲目遍历。

3. **双层级回滚机制**（Two-level Rollback Mechanism）
   - **Level 1**: 基于深度限制的 `Back` 操作 + **pHash** 屏幕哈希验证，快速恢复状态。
   - **Level 2**: 若 Level 1 失败，则退回到 Home 页面，重放到达该状态的操作轨迹（replay reasoning trace），确保状态一致性。
   - 有效应对弹窗、权限请求等不可逆 UI 变化。

4. **探索痕迹结构化为 Prompt Hint**
   - 将探索过程中发现的有用 UI 元素（如“顶部右侧的 Settings 图标可能有用”）通过模板生成简洁自然语言提示。
   - 注入下一推理步的 prompt 中，增强 VLM 的决策能力。

---

### 🔍 相比现有方法的优势

| 方法类型 | 缺陷 | MobileExplorer 的改进 |
|--------|------|-----------------------|
| **云侧推理系统** | 隐私泄露、网络延迟 | 完全 on-device，保护隐私，消除网络依赖 |
| **序列化 Pipeline** | 推理期间设备空转 | 并行探索，充分利用推理延迟 |
| **离线知识库构建**（如 GUI-Explorer） | 构建成本高，泛化差 | 在线即时探索，适应动态应用 |
| **输入剪枝/压缩技术** | 易丢失细粒度视觉信息 | 不修改输入，而是补充上下文 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **AndroidWorld Benchmark** [13]：一个动态、闭环的移动端 GUI 交互基准测试平台。
  - 包含 116 个真实 Android 应用中的多步任务。
  - 覆盖多种任务类型：音频录制、笔记管理、费用添加、搜索等。
  - 支持系统级控制（ADB），模拟真实用户操作。

此外还设计了新的 **现实世界复杂任务案例研究**，涵盖：
- 复杂 UI（Trip.com 页面平均 48 个可交互元素）
- 弹窗干扰（来电、通知、闹钟）
- 资源动态竞争（后台播放视频/音乐影响资源）

---

### ⚙️ 实验设置

#### 设备平台
| 设备 | 类型 | 内存 | 用途 |
|-----|------|------|------|
| Samsung Galaxy S24 | 智能手机 | 12GB | 主要测试设备 |
| NVIDIA Jetson AGX Orin | 边缘计算设备 | 64GB | 性能对比 |
| MacBook Air M4 | 笔记本电脑 | 24GB | 对比参考 |

所有模型均使用 **llama.cpp + Q8量化** 部署，通过 **vLLM** 提供统一 API 接口。

#### 模型配置
主实验采用 **4B 参数的 VLM**（基于 STEP-UI 或 MAI-UI 架构），并在消融实验中测试了 2B 和 8B 模型以验证鲁棒性。

---

### 📊 评估指标

| 指标 | 定义 |
|------|------|
| **Success Rate (%)** | 成功完成的任务占比 |
| **Average Steps** | 完成任务所需的平均交互步数 |
| **Step Latency** | 单步耗时（从截图到动作执行完成） |
| **End-to-End Latency (s)** | 整体任务完成时间（所有 step latency 之和） |
| **Hint-Follow Rate** | 模型是否采纳探索提示的比例（用于分析有效性） |

---

### 🆚 基线方法对比

| Baseline | 描述 |
|---------|------|
| **M3A Agent** | 基础 VLM Agent，标准感知-推理-动作循环 |
| **T3A Agent** | 基于 Accessibility Tree + LLM 的文本型 Agent |
| **Input-pruning VLM Agent** | 使用截图裁剪或 token 剪枝减少输入规模 |
| **Offline Exploration Agent** | 依赖预先构建的知识图谱（如 GUI-Explorer） |
| **Leaderboard 方法** | 对比 AndroidWorld 上已发表的最佳结果（如 GLM-4.1V, UI-TARS 等） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（AndroidWorld 测试）

| 方法 | Success Rate (%) | Avg. Steps | End-to-End Latency (s) |
|------|------------------|-----------|------------------------|
| **MobileExplorer** | **50.9%** | **9.24** | **185.82** |
| M3A (Baseline) | 46.55% | 10.93 | 221.00 |
| T3A (Qwen3-4B) | ~30% | >15 | >300 |
| Input-pruning | ~45% | ~11 | ~210 |
| Offline Exploration | ~40% | ~12 | ~250 |

> ✅ **提升效果**：
- **成功率提升最多达 5%**（相对提升约 9.3%）
- **交互步数减少 15.5%**
- **端到端延迟降低 15.9%**

在更广泛的跨模型测试中（见 Fig. 13a），MobileExplorer **平均减少约 23% 的推理步数和端到端延迟**，同时保持或略微提升成功率。

---

### 🔬 消融实验结果（Ablation Study）

| 组件移除 | Success Rate ↓ | End-to-End Latency ↑ | 分析 |
|--------|----------------|------------------------|------|
| 移除 **任务相关性选择**（随机探索） | 50.9% → 42.2% | 显著上升 | 随机探索效率低，引入噪声 |
| 移除 **双层级回滚机制** | → 39.7% | 大幅增加 | 回滚失败导致状态错乱 |
| 移除 **探索-推理对齐**（无筛选提示） | → 47.4% | 上升 | 错误提示误导模型决策 |

> 结论：三大组件协同作用，缺一不可。

---

### 📊 其他重要发现

- **Hint-Follow Rate 与任务复杂度正相关**：
  - 在复杂 UI 场景下，hint 被采纳的概率更高，且采纳后成功率显著提升（Fig. 12d）。
- **分辨率影响较小**：
  - MobileExplorer 在中等分辨率（540×1200）即可达到最佳平衡，无需超高分辨率。
- **模型大小泛化良好**：
  - 在 2B 到 8B 的 VLM 上均能稳定提效，说明方法不依赖特定模型容量。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **VLM 推理延迟是一个可被利用的时间窗口**，而非必须忍受的开销。
2. **轻量级在线探索 + 结构化提示注入** 可显著提升 GUI Agent 的决策效率。
3. **双层级回滚机制保障了探索的安全性和稳定性**，使在线探索可在真实设备上可靠运行。
4. MobileExplorer 在完全 on-device 设置下实现了：
   - 更少的交互步数（↓23%）
   - 更低的端到端延迟（↓23%）
   - 更高的任务成功率（↑ up to 5%）

---

### ⚠️ 方法的局限性

1. **探索深度受限**：受限于推理时间预算，只能进行浅层探索（通常 ≤3 步），无法覆盖深层菜单。
2. **依赖 Accessibility Tree**：虽然比视觉解析快，但在某些应用中 Accessible Text 可能缺失或不准。
3. **提示生成依赖手工模板**：当前 hint 是规则模板生成，尚未实现端到端学习。
4. **极端干扰场景仍具挑战**：例如连续多个弹窗叠加，可能导致 replay 轨迹失效。

---

### 🔮 未来工作方向

1. **自适应探索策略**：根据任务难度和 UI 复杂度动态调整探索深度与广度。
2. **探索与推理联合优化**：将 exploration policy 与 VLM 微调结合，形成闭环学习。
3. **支持多模态反馈探索**：结合语音提示、振动反馈等新型交互模式。
4. **扩展至 Web & Desktop GUI Agents**：将框架推广到其他图形界面自动化场景。

---

> 💡 **一句话总结**：  
> *MobileExplorer 通过“边推理边探索”的范式革新，首次高效利用了 on-device VLM 的推理延迟，在不牺牲隐私的前提下显著提升了移动 GUI Agent 的执行效率与成功率。*

</details>

---

### 11. [Efficient Agentic Reinforcement Learning with On-Policy Intrinsic Knowledge Boundary Enhancement](https://arxiv.org/abs/2605.26952)

**Authors**: Dingwei Chen, Zefang Zong, Zhipeng Ma, Leo Luo, Yang Li, Chengming Li, Peng Chen, Jie Jiang  
**Category**: cs.CL  
**Published**: 2026-05-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.26952v1  

#### Abstract
Agentic reinforcement learning (RL) has proven effective for training LLM-based agents with external tool-use capabilities. However, we identify that agentic RL training induces increasing redundant tool calls and blurs the model's intrinsic knowledge boundary, where the model fails to distinguish w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient Agentic Reinforcement Learning with On-Policy Intrinsic Knowledge Boundary Enhancement

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **agentic reinforcement learning (RL)** 训练过程中，LLM-based agents 虽然能通过工具增强推理能力，但普遍存在**冗余工具调用**（redundant tool calls）的问题。具体表现为：
- 模型在参数化知识足以回答时仍调用工具（认知卸载，cognitive offloading）；
- 过度调用工具导致计算资源浪费、推理延迟增加；
- 不必要的工具调用可能引入噪声，覆盖正确的内部推理，导致答案质量下降。

现有基于 **reward shaping** 的解决方案（如 OTC-PO、β-GRPO）存在粗粒度优化问题，容易引发 **reward hacking**——模型为获取额外奖励而盲目减少工具调用，损害任务准确性。

---

### 提出了什么新方法或新思路
本文提出 **AKBE (Agentic Knowledge Boundary Enhancement)**，一种**基于策略（on-policy）的知识边界增强方法**，其核心思想是：

- **动态探测模型的内在知识边界**（intrinsic knowledge boundary），即判断每个问题是否需要外部工具以及最少需要多少次工具调用。
- 通过 **dual-path rollouts**（带工具与不带工具并行采样）比较路径正确性，识别四种轨迹类别，并构建针对性的监督信号。

#### 四类轨迹分类及对应信号构造：
| 类别 | 条件 | 目标 | 构造方式 |
|------|------|------|---------|
| **Tool-dependent** | 带工具可解，无工具不可解 | 强化高效工具使用 | 选择**最少工具调用次数**的正确带工具轨迹 |
| **Efficiency** | 无需工具即可正确回答 | 消除冗余调用 | 随机选择一个正确的无工具轨迹作为目标 |
| **Hallucination** | 有工具反而错误 | 减少有害依赖 | 选择正确的无工具轨迹，避免噪声干扰 |
| **Both-wrong** | 两条路径均失败 | 无额外信号 | 仅依赖原始 RL 目标 |

这些信号以辅助损失项形式无缝集成到标准 agentic RL 训练流程中，形成联合目标函数：
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda \cdot \mathcal{L}_{\text{AKBE}}
$$

---

### 相比现有方法的优势
- ✅ **细粒度控制**：相比 reward shaping 的全局惩罚机制，AKBE 在实例级别提供精准指导，避免“一刀切”抑制工具调用。
- ✅ **防止 reward hacking**：不修改奖励函数，而是添加辅助监督信号，保持 RL 优化稳定性。
- ✅ **动态适应性**：on-policy 设计使知识边界随训练过程动态演化，信号始终与当前模型能力匹配。
- ✅ **即插即用兼容性**：可无缝集成于 GRPO、DAPO、GSPO、AEPO 等多种 agentic RL 算法。
- ✅ **效率提升无代价**：在提高准确率的同时显著降低工具调用，实现真正的 accuracy-efficiency 协同增益。

---

## 2. 核心实验方法和设置

### 使用的数据集
共七个 QA benchmark，分为两类：

#### Multi-Hop QA（多跳问答）
- **HotpotQA**：维基百科衍生，需支持事实推理
- **2WikiMultiHopQA**：结合维基与 Wikidata，强调实体推理
- **MuSiQue**：由单跳问题合成，精细控制推理深度
- **Bamboogle**：对抗性组合查询，测试鲁棒性

#### Single-Hop QA（单跳问答）
- **Natural Questions (NQ)**：真实用户提问，标准 RAG 测试集
- **TriviaQA**：词汇与句法差异大，考验泛化
- **PopQA**：实体中心型，用于区分记忆 vs 检索行为

所有任务采用 **Exact Match (EM)** 为主评估指标。

---

### 实验设置和评估指标

| 设置项 | 描述 |
|--------|------|
| **Backbone Models** | Qwen3-4B 和 Qwen2.5-7B |
| **External Tool** | 基于 Wikipedia 的轻量级搜索引擎（e5-base-v2 检索器） |
| **Rollout 数量** | 带工具：16条；无工具：8条 |
| **最大工具调用数** | 6次 |
| **训练批大小** | 64 |
| **AKBE 系数 λ** | 0.05 |
| **评估指标** | EM（准确率）、Tool Calls (TC)、Tool Productivity (TP) |

其中：
$$
\text{TP} = \frac{\text{Accuracy}}{\text{Average Tool Calls}}
$$

---

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **ReAct** | Prompting | 无 RL 训练基准 |
| **Search-o1 / R1-Searcher / Search-R1** | Agentic RL | 基于 GRPO 的经典框架 |
| **OTC-PO** | Reward Shaping | 添加工具生产力奖励项 |
| **β-GRPO** | Reward Shaping | 引入置信阈值减少不确定性 |
| **HiPRAG** | Reward Shaping | 分层过程奖励，逐步评估 |
| **Offline AKBE** | 对照变体 | 使用固定检查点生成静态信号 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen3-4B 平均表现）

| 方法 | Avg. EM ↑ | Tool Calls ↓ | Tool Productivity ↑ |
|------|-----------|-------------|---------------------|
| Standard RL (Search-R1) | 45.40 | 3.16 | 14.33 |
| **AKBE (Ours)** | **46.82 (+1.42)** | **2.60 (-18%)** | **18.01 (+25%)** |

> 注：原文称平均提升 +1.85，为跨两个 backbone 模型七项任务的整体平均。

---

### 与基线方法的对比结果

- **AKBE 显著优于所有基线**，在多数任务上取得最高 EM 和最低 TC。
- **OTC-PO** 虽然 TC 最低（2.06），但 EM 下降明显，验证了 reward hacking 问题。
- **β-GRPO** 抑制了 EM 崩溃，但 TC 减少有限（3.01 → 3.16），效率提升不足。
- **Offline AKBE** 表现弱于 AKBE，说明**静态信号无法适应训练动态变化**，过早压制工具调用。

---

### 消融实验结果

| 配置 | Avg. EM | TC | TP |
|------|--------|----|----|
| GRPO | 45.40 | 3.16 | 14.33 |
| **Full AKBE** | **46.86** | **2.60** | **18.02** |
| w/o Tool-dependent | 43.56 | 2.15 | 20.26 |
| w/o Efficiency | 46.50 | 2.92 | 15.93 |
| w/o Hallucination | 46.55 | 2.58 | 18.04 |
| w/ Tool-dependent only | 46.02 | 2.85 | 16.15 |

#### 发现：
- 移除 **Tool-dependent** 导致 EM 大幅下降 → 缺乏对必要工具调用的保护，造成过度抑制。
- 移除 **Efficiency** 导致 TC 上升 → 该信号是消除冗余调用的关键驱动力。
- 移除 **Hallucination** 对 TC 影响小，但 EM 微降 → 有效纠正检索噪声导致的错误。
- 三者互补：**Tool-dependent 教“何时高效用”，Efficiency 教“何时不用”，Hallucination 教“何时不该用”。**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **agentic RL 训练会加剧冗余工具调用**，且随训练推进持续增长（见图1）。
2. ✅ **AKBE 成功实现了 accuracy 与 efficiency 的双重提升**，无 trade-off。
3. ✅ **on-policy 动态信号构造至关重要**：离线信号因过于乐观而导致早期训练阶段误判。
4. ✅ **知识边界在训练中动态演化**：随着训练进行，“Efficiency” 类别占比上升，表明模型逐步将外部依赖内化为参数知识（见图3）。
5. ✅ **AKBE 是即插即用模块**：在 GRPO、DAPO、GSPO、AEPO 上均带来一致改进（表2）。
6. ✅ **计算开销可控甚至更低**：尽管增加了 no-tool rollouts，但由于其速度快且后期 with-tool rollouts 变短，**AKBE 平均比 GRPO 快 15%**（见图5）。

---

### 方法的局限性
1. **早期训练阶段额外开销**：在工具调用尚未减少前，no-tool rollouts 会增加计算负担。
2. **系数 λ 固定**：未考虑训练阶段或任务难度的变化，最优 λ 可能动态调整更优。
3. **信号可靠性假设**：基于“至少一条无工具路径正确”来判定知识边界，虽经分析证明可靠（附录D），但仍存在极少数偶然命中风险。

---

### 未来工作方向
- 探索 **adaptive rollout 策略**：仅对可能位于知识边界内的问题执行 no-tool rollouts。
- 设计 **自适应 λ 调整机制**：根据当前轨迹分布或任务复杂度动态调节信号强度。
- 扩展至更多工具类型和环境交互场景（如 code interpreter、planning 等）。
- 结合 meta-learning 或 curriculum learning 进一步优化知识边界学习过程。

---

> 🔗 **代码开源地址**：[https://github.com/CuS04-Chen/AKBE](https://github.com/CuS04-Chen/AKBE)

</details>

---

### 12. [LLMs Are Already Good Tutors: Training-Free Prompt Optimization for Pedagogical Math Tutoring](https://arxiv.org/abs/2605.27088)

**Authors**: Unggi Lee, Minchul Shin, Yeil Jeong, Sookbun Lee, Jeongsu Moon, Kyungtae Joo, Eunjoo Lee, Hoilym Kwon  
**Category**: cs.CL  
**Published**: 2026-05-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.27088v1  

#### Abstract
Aligning LLMs for math tutoring typically requires RL-based training with multi-GPU infrastructure. We investigate whether training-free prompt optimization-evolving only the system prompt via API calls-can serve as a practical alternative. We adapt 7 published methods and propose 5 education-specia...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*LLMs Are Already Good Tutors: Training-Free Prompt Optimization for Pedagogical Math Tutoring*

---

## 1. 主要贡献和创新点

### ✅ 解决的问题
当前将 **LLMs** 对齐为数学导师（math tutor）的方法通常依赖于基于 **强化学习（RL）** 的训练，需要多GPU基础设施和大量训练样本（如 >10K 问题），这使得教育研究者和实践者难以参与模型开发。

本文提出并验证了一种无需训练、仅通过优化 **system prompt** 即可实现高质量教学对齐的替代方案。

---

### ✅ 提出的新方法与新思路

作者提出了 **5 种教育领域专用的 training-free prompt optimization 方法**，均不更新模型权重，仅通过 API 调用演化提示词：

| 方法 | 核心思想 |
|------|--------|
| **ParetoGrad** | 结合 population-based 搜索与 weakness-targeted text gradient，采用 NSGA-II 进行非支配排序，追求在 `solve rate`、`leak control` 和 `helpfulness` 上的帕累托最优平衡 |
| **CondBridge** | 在 NoThink 与 Think 两种模式下联合评估，提升跨推理模式的鲁棒性 |
| **LeakShield** | 两阶段优化：先最小化泄露，再在泄露阈值内最大化解决率和帮助度 |
| **Frame** | 对 TextGrad 的梯度进行后处理重构，增强教学结构性（如避免禁止性语言，注入示例） |
| **MetaBlend** | 聚合多个成功运行的最佳 prompt，提取共性结构作为初始种子，再进行精细化优化 |

此外，还适配了 7 种已有的通用 prompt optimization 方法（如 OPRO, TextGrad, GEPA 等）用于多轮教育对话场景。

---

### ✅ 相比现有方法的优势

| 维度 | 优势说明 |
|------|---------|
| **计算成本低** | 仅需单卡（RTX3090）推理，无需多GPU训练；数据量减少 100 倍（100 vs 10K 问题） |
| **可解释性强** | 输出是人类可读的 prompt，便于教师审查、修改和部署 |
| **开发门槛低** | 不需要 ML 工程能力，适合教育工作者直接参与设计 |
| **性能更强** | 所有 12 种 training-free 方法均超过最强的 RL-trained baseline（R_total=0.633） |

> 💡 **核心洞见**：prompt 本身是一种显式的 pedagogical prior，能直接激活 LLM 中的教学知识模式（MKT），而 RL 则是通过奖励信号间接塑造行为节奏（PIU）。

---

## 2. 核心实验方法和设置

### 📚 数据集

- **训练集（优化用）**：
  - **BigMath**（Albalak et al., 2025）：100 道中高难度数学题（过滤后学生 baseline solve rate 1–60%）
  
- **测试集（评估用，OOD）**：
  - **OpenLearnLM**（Lee et al., 2026b）：包含 4 个子基准，评估知识、技能与态度
  - **MathTutorBench (MTB)**（Macina et al., 2025）：2 个子基准，侧重开放式教学能力
  - 总计：**1,334 个 OOD 测试项**

---

### ⚙️ 实验设置

| 组件 | 设置详情 |
|------|----------|
| **Tutor Model** | Qwen2.5-7B-Instruct（NoThink）、Qwen3-8B（Think 模式） |
| **Student Model** | LLaMA-3.1-8B-Instruct（模拟学生反应） |
| **Reward Judge** | GPT-4o-mini（自动打分） |
| **Reflection Model** | GPT-4o（生成 prompt 改进建议） |
| **对话长度** | 最多 5 轮（Turn 1 → Turn 5） |
| **优化预算** | 每个方法最多 500 次 reward evaluation |
| **条件数量** | 5 种配置（NoThink, Think-NoReward, Think-Reward 及其 pedagogical-seed 变体） |

---

### 🎯 评估指标

定义总奖励 $ R_{\text{total}} $ 为三个核心目标的平均值：

$$
R_{\text{total}} = \frac{R_{\text{sol}} + R_{\text{leak}} + R_{\text{help}}}{3}
$$

其中：

- $ R_{\text{sol}} $：学生在对话后的 **post-test solve rate**（K=8 次尝试下的正确率）
- $ R_{\text{leak}} $：是否发生 **answer leakage**（0 或 1），报告时使用 leak rate = $1 - R_{\text{leak}}$
- $ R_{\text{help}} \in [0,1] $：由 GPT-4o-mini 评分的 pedagogical helpfulness
- （附加）$ R_{\text{think}} $：思考质量（仅 Think 模式）

同时报告：
- OOD 表现（OpenLearnLM-Avg, MTB-Avg）
- 多轮行为分析（codebook labeling）

---

### 🔁 基线方法对比

| 类型 | 包含方法 |
|------|----------|
| **Frontier Models (Zero-shot)** | GPT-5.2(Ped.), Claude-4-Opus(Ped.), DeepSeek-V3.2(Ped.) |
| **RL-Trained Models** | Ped. Think R (最强 baseline, R_total=0.633) |
| **Baseline (No Optimization)** | 无优化的默认 prompt |
| **Published Prompt Methods (Adapted)** | OPRO, TextGrad, ACE, GEPA, MIPROv2, EvoPrompt, TF-GRPO |
| **Proposed Methods (Ours)** | ParetoGrad, CondBridge, LeakShield, Frame, MetaBlend |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 方法 | $ R_{\text{sol}} $ | Leak ↓ | Help ↑ | $ R_{\text{total}} $ | OOD (MTB-Avg) | OOD (OL-Avg) |
|------|---------------------|--------|--------|------------------------|---------------|--------------|
| **Ped. Think R (RL)** | 0.294 | 0.172 | 0.776 | **0.633** | — | — |
| **ParetoGrad (Ours)** | **0.563** | 0.252 | **0.845** | **0.719** | 7.78 | 7.38 |
| **OPRO** | **0.574** | 0.382 | 0.846 | 0.684 | 7.75 | 7.31 |
| **CondBridge** | 0.552 | **0.204** | 0.843 | 0.691 | 7.83 | 7.41 |
| **EvoPrompt** | 0.566 | 0.286 | 0.847 | 0.711 | **8.09** | **8.09** |

> ✅ **所有 12 种 training-free 方法的 $ R_{\text{total}} $ 均 > 0.633**，全面超越最强 RL baseline！

---

### 🔍 与基线方法的对比结果

| 发现 | 描述 |
|------|------|
| ✅ **全面超越 RL** | ParetoGrad 达到 $ R_{\text{total}} = 0.719 $，比 RL 高出 **+86 个百分点**（0.719 vs 0.633） |
| ✅ **更低资源消耗** | 仅用 100 个训练问题（vs RL 的 10K+），单卡即可完成，成本降低两个数量级 |
| ✅ **更好的帕累托平衡** | ParetoGrad 在三项指标间取得最佳权衡，没有明显短板 |
| ⚠️ **局部优势差异** | OPRO 在 $ R_{\text{sol}} $ 上最高（0.574），CondBridge 在 NoThink 下泄露最低（0.204） |

---

### 🔬 消融与深入分析结果

#### （1）In-Domain vs OOD 泛化脱节（Figure 4）
- **无显著相关性**：in-domain $ R_{\text{total}} $ 与 OOD 表现几乎无关（MTB: p=0.01, OL: p=0.25）
- 示例：
  - **ACE**：in-domain 排名第12，但 OOD 排第2
  - **LeakShield**：in-domain 第5，但 OOD 垫底
- ➜ 表明：**过强的 scaffolding 有助于 in-domain 得分，但 content-rich 教学更利于 OOD 迁移**

#### （2）行为代码本分析（82-code educational codebook）
- **Training-free 方法**：
  - 更多使用 **Mathematical Knowledge for Teaching (MKT)**（19–23% vs RL 的 7–8%）
  - 更少使用 **Pedagogical Intent Utterance (PIU)**（46–55% vs RL 的 60–65%）
- ➜ 说明：**prompt 优化激发的是“教什么”而非“怎么问”的策略**

#### （3）Polya 阶段进展分析
- 高性能方法在整个对话中持续保持 **Scaffold** 行为（+9.8pp @ T1, +8.8pp @ T5）
- 低性能方法倾向于早期转向 **Execution**（代学生计算）

#### （4）成功 vs 失败对话差异
- 成功对话特征：
  - 减少 **Step-by-step instruction**
  - 增加 **Exploratory questions** 和 **Praise**（尤其在结尾 T5）
- ➜ 教学应引导而非告知，结束时给予正向反馈

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Training-free prompt optimization 完全可以匹敌甚至超越 RL-based 对齐方法**，在数学辅导任务上达到更高综合表现。
2. **ParetoGrad 实现了最佳的多目标平衡**，虽未单项第一，但在 solve/leak/help 三者之间无明显弱点。
3. **不同优化范式激发不同的教学风格**：
   - Prompt-based：强调 **概念解释与知识传递（MKT）**
   - RL-based：强调 **交互节奏与提示技巧（PIU）**
4. **in-domain 表现 ≠ OOD 泛化能力**：过度优化本地奖励可能导致泛化下降。
5. **成功的教学行为特征**：
   - 持续 scaffolding
   - 使用探索性提问代替直接步骤指导
   - 在对话末尾给予 praise 强化动机

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **模拟学生（Simulated Student）** | 使用 LLM 模拟学生反应，可能无法完全反映真实学习者的行为 |
| **仅限数学学科** | 尚未验证在其他科目（如物理、编程）中的有效性 |
| **依赖 LLM-as-Judge** | 缺乏人工标注验证，可能存在评估偏差 |
| **未报告随机种子方差** | 所有结果为单次运行，稳健性有待进一步验证 |
| **小模型为主** | 使用 7B/8B 模型，更大模型可能缩小方法差距 |

---

### 🔮 未来工作方向

1. **扩展至真实学生实验**：在真实课堂环境中测试这些 prompt 的有效性
2. **跨学科迁移**：将方法应用于科学、语言等其他教学领域
3. **结合 human-in-the-loop**：让教师参与 prompt 设计与优化过程
4. **构建可共享的 prompt 库**：建立开源的 pedagogical prompt repository
5. **设计新的 OOD-aware reward**：鼓励既能得分又能泛化的教学策略

---

## 🧩 总结一句话

> **无需训练，只需优化 prompt，就能让 LLM 成为比 RL 训练更强的数学导师——这不仅更高效，也更透明、更可解释，为教育者打开了直接参与 AI 教学系统设计的大门。**

</details>

---

### 13. [Revisiting Bruck: Phase-Efficient All-to-All Communication in Reconfigurable Networks](https://arxiv.org/abs/2605.26930)

**Authors**: Anton Juerss, Stefan Schmid  
**Category**: cs.DC  
**Published**: 2026-05-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.26930v1  

#### Abstract
All-to-All communication is a key performance bottleneck for distributed machine learning (ML) and high-performance computing (HPC) workloads, where dense traffic increasingly stresses scale-up interconnects. While these ML and HPC workloads have driven unprecedented infrastructure demand, optical r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Revisiting Bruck: Phase-Efficient All-to-All Communication in Reconfigurable Networks*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **All-to-All** 集体通信在 **光学可重构网络（ORNs）** 中的性能瓶颈问题展开研究。All-to-All 是分布式机器学习（ML）和高性能计算（HPC）中的关键通信模式，其密集流量对传统电交换互连造成严重拥塞。尽管 ORNs 可通过动态调整物理拓扑来优化带宽利用率，但其非忽略不计的 **reconfiguration delay**（重配置延迟）使得必须协同设计通信算法与拓扑调度策略。

现有方法如 Bruck’s All-to-All 虽然具有多阶段结构，适合 ORNs，但存在以下问题：
- 仅利用单向通信，未充分利用双向光链路；
- 每个阶段需要独立的拓扑调整，导致频繁 reconfiguration；
- 通信路径较长，增加传播延迟和拥塞。

### 提出的新方法：RETRI
作者提出 **RETRI**（Reusable Ternary Subrings），一种面向 ORNs 的新型双向 All-to-All 通信模式，其核心思想是：
- **基于平衡三进制（balanced ternary）的块传播机制**：将每个通信阶段的数据划分为三组——本地保留、左传、右传，实现更短路径传输。
- **诱导可复用子环拓扑（reusable subrings）**：每个阶段的通信自然形成大小为 $ n / 3^k $ 的子环，这些子环可在多个阶段中复用，减少 reconfiguration 次数。
- **双端口约束下的最优设计**：假设每节点有两个 electro-optical transceiver，构建 degree-2 的双向环形拓扑，满足最小连接性要求。

### 相比现有方法的优势
- **相位数更少**：完成 All-to-All 仅需 $ \lceil \log_3 n \rceil $ 个 phase，相比 Bruck 的 $ \lceil \log_2 n \rceil $ 减少了约 **33%**。
- **完全双向通信**：每个光链路同时承载双向流量，提升链路利用率。
- **拓扑可复用性强**：通过子环结构，允许跨多个 phase 复用同一拓扑状态，有效摊销 reconfiguration 开销。
- **路径更短、拥塞更低**：三进制传播缩短了平均跳数，降低链路竞争。

---

## 2. 核心实验方法和设置

### 实验平台
使用 **Astra-Sim** 作为系统级模拟器，底层网络引擎为 **ns-3**，用于建模真实网络行为。

### 网络参数
- Link bandwidth: **400 Gbps**
- Propagation delay: **1 μs**
- Per-phase software overhead: **1.7 μs**
- Reconfiguration delay ($\delta$): 从 **1 μs 到 50 ms** 不等，覆盖快慢两种 OCS 架构
- 节点规模：
  - RETRI: $ n = 81 $（$3^4$）
  - Bruck/Bridge: $ n = 64 $（$2^6$）
  > 注：选择各自“自然尺寸”以公平比较

### 数据负载范围
消息大小 $ m $ 从 **1 KB 到 256 MB**，涵盖小消息同步到大模型参数交换场景。

### 基线方法
1. **Static All-to-All**：基于最短路径的静态环形网络实现，无拓扑重配。
2. **Bridge**：Bruck 在 ORNs 上的 reconfigurable 版本，作者对其进行镜像优化（mirrored All-to-All）以支持双向通信，确保与 RETRI 公平对比。

### 评估指标
- **Completion Time Speedup**：相对于基线的加速比
- **Number of Reconfigurations (R)**：实际执行的拓扑切换次数
- **Hop Count & Congestion**：隐含在模型分析中，影响传输延迟

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Figure 2 & 3）

#### ✅ 对比 Static All-to-All（图2）
| 条件 | 最高加速比 | 说明 |
|------|-----------|------|
| $\delta = 1\mu s$, $m = 256MB$ | **10×** | 小延迟下性能增益显著 |
| $\delta = 1ms$, $m ≥ 8MB$ | **>6.9×** | 即使毫秒级延迟仍大幅领先 |
| $\delta = 50ms$, $m = 256MB$ | **1.1–1.5×** | 大消息下仍能受益 |

> 所有消息大小在低延迟下均有明显提速；大消息即使在高延迟下也保持优势。

#### ✅ 对比 Bridge（Bruck 的 reconfigurable 版本）（图3）
| 条件 | 加速比范围 | 说明 |
|------|------------|------|
| 整体区间 | **1.2× ~ 2.1×** | 始终优于 Bruck 风格调度 |
| 小消息（1KB–32KB） | **≥1.6×** | 得益于更少 phase 和更短路径 |
| 大消息（64MB–256MB） | **1.8×–2.1×** | 传输主导时仍因拓扑优化胜出 |

> 尽管 RETRI 运行在更大网络（81 vs 64），仍取得一致且显著的性能提升。

### 消融实验与补充结果（附录 Figures 4 & 5）
- **Figure 4 ($n=9$ vs $n=8$)**：小规模下 RETRI 在 $\delta=100ns$ 达到 **2.0×** 以上加速，验证其在小网络的有效性。
- **Figure 5 ($n=243$ vs $n=256$)**：大规模下，RETRI 在 $\delta=150ms$ 时仍比静态方案快 **1.2×**，而 Bruck 已无收益 —— 表明 RETRI 更适应高延迟 ORNs。

### 成本模型分析关键结论
- RETRI 的最优 reconfiguration 次数 $ R^* $ 随 $\delta$ 增加而减少，符合预期。
- 当通信代价（$\alpha_h + \beta$）远高于 $\delta$ 时，频繁 reconfiguration 更优。
- Bruck 因 phase 数更多，承受更高的累计 reconfiguration 开销（约多 58%）。

---

## 4. 关键结论和发现

### 主要发现
1. **通信模式与拓扑调度必须协同设计（co-design）**：单纯优化拓扑无法超越不良通信结构；RETRI 通过 ternary 结构天然适配 ORNs。
2. **减少通信 phase 数是关键优化目标**：phase 数直接影响 reconfiguration 次数上限，RETRI 的 $ \log_3 n $ 设计带来根本性优势。
3. **双向通信 + 子环复用 = 高效摊销开销**：RETRI 同时实现了链路利用率最大化与拓扑稳定性。
4. **RETRI 在现实延迟下依然有效**：即使面对 **ms 级 reconfiguration delay**，在大消息或大网络中仍能提供显著加速。

### 方法的局限性
- **依赖 $n = 3^k$ 规模**：当前分析集中在三的幂次网络；任意规模需进一步扩展。
- **假设理想同步**：所有节点在同一 phase 完成后才能 reconfigure，忽略了 barrier overhead 和 straggler 影响。
- **尚未硬件验证**：目前为仿真结果，缺乏真实 OCS 硬件测试。
- **仅适用于 All-to-All**：虽可推广，但其他 collective（如 AllReduce）需重新设计依赖处理逻辑。

### 未来工作方向
1. **支持非三幂网络**：通过 padding 或混合 radix 扩展至通用 $n$。
2. **扩展至更高度拓扑**：每节点 $d > 2$ 光端口时，可构造 degree-$d$ circular ring，提升并行度。
3. **Overlap reconfiguration with computation**：探索在计算期间预 reconfigure，避免其位于关键路径。
4. **应用于其他 collectives**：如 AllReduce、Reduce-Scatter 等，发展一套 reconfiguration-aware collective library。
5. **控制平面优化**：研究轻量级 signaling 机制以协调 topology changes。
6. **与 parallelism strategy 联合优化**：结合数据并行、流水并行等策略进行端到端训练优化（类似 TopoOpt 方向）。

---

> **总结一句话**：  
> RETRI 通过 **三进制双向通信 + 可复用子环拓扑**，在 ORNs 上实现了 **phase-efficient** 的 All-to-All，显著降低了 completion time，尤其在高 reconfiguration delay 和大 workload 下展现出强大鲁棒性和优越性，为 reconfiguration-aware collective communication 提供了新范式。

</details>

---

### 14. [MTL-FNO: A Lightweight Multi-Task Fourier Neural Operator for Sparse Field Reconstruction](https://arxiv.org/abs/2605.26718)

**Authors**: Siyu Ye, Shihang Li, Zhiqiang Gong, Benrong Zhang, Weien Zhou, Yiyong Huang, Wen Yao  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.26718v1  

#### Abstract
Efficient onboard multi-field sparse reconstruction is essential for the autonomous operation of aerospace vehicles. While existing deep learning models exhibit promise for single-field reconstruction, deploying multiple independent models leads to prohibitive model size growth and fails to exploit ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MTL-FNO: A Lightweight Multi-Task Fourier Neural Operator for Sparse Field Reconstruction

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**航空航天飞行器在轨自主运行中多物理场稀疏重建**的需求，解决了以下关键挑战：
- **模型规模过大**：为多个物理场部署独立的深度学习模型会导致参数量线性增长，难以满足星载平台对计算资源、内存和功耗的严格限制。
- **跨场相关性未被利用**：多个耦合物理场（如温度、压力、应力）之间存在共性特征，但传统单任务模型无法有效挖掘这些共享信息，尤其在**few-shot**（小样本）条件下表现不佳。
- **复数域优化冲突**：在基于频域操作的FNO中，直接叠加共享与任务特定参数会破坏其单位性（unitary），导致相位（phase）与幅值（amplitude）优化相互干扰，引发任务冲突。

### 提出的新方法与创新思路
作者提出了一种轻量化的**多任务傅里叶神经算子（MTL-FNO）**，其核心创新包括：

#### （1）**硬参数共享的端到端多任务框架**
- 在每个MTL-FNO层中，将参数划分为**共享部分**（shared）和**任务特定部分**（task-specific）。
- 采用**CP张量分解**（CANDECOMP/PARAFAC decomposition）构建低秩的任务微调项（low-rank fine-tuning parameters），显著压缩模型体积。

#### （2）**极坐标视角下的谱权重解耦优化**
- 将FNO中的复数谱权重 $ R \in \mathbb{C}^{k\times k\times C\times C} $ 进行**极分解**（Polar Decomposition）：
  - $ R = U \odot P $
  - $ U $：**酉张量**（unitary tensor），编码**相位信息**
  - $ P $：**半正定厄米张量**（positive semi-definite Hermitian tensor），表征**幅值缩放**
- 通过解耦相位与幅值的优化过程，缓解多任务之间的冲突。

#### （3）**基于Cayley变换的酉矩阵重参数化**
- 引入**Cayley变换**将对酉流形上的约束优化转化为无约束优化：
  - $ U = (I - K)(I + K)^{-1} $，其中 $ K $ 是斜厄米矩阵（skew-Hermitian）
- 使得共享与任务特定的 $ K $ 可以自由相加后再映射回酉矩阵，从而保持几何保真度。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **模型效率** | 参数量减少 **76%**（Case A）和 **60%**（Case B）相比多个独立FNO |
| **预测精度** | 在多个任务上达到甚至超过独立FNO的性能，尤其在stress等难重建场中提升显著 |
| **小样本鲁棒性** | 在few-shot条件下（仅100训练样本）仍能稳定收敛，而基线FNO易出现负$ R^2 $ |
| **可扩展性** | 低秩设计使模型容量不随任务数线性增长，适合多场联合建模 |

---

## 2. 核心实验方法和设置

### 数据集
实验在两个典型工程场景下进行验证：

#### **Case A: 二维钝楔高超声速稀薄流多物理场重建**
- 物理场：温度 $ T $、压力 $ P $、应力分量 $ T_{xx}, T_{xy}, T_{yy} $
- 输入：表面3个传感器观测值
- 输出：全区域 $ m = 3792 $ 个网格点的物理场分布
- 数据来源：基于玻尔兹曼方程的高保真数值模拟（UGKS）

#### **Case B: 卫星舱内多工况温度场重建**
- 物理场：同一结构下三种热边界条件的温度场
  - ADlet（主动冷却）
  - DSine（动态正弦扰动）
  - HSink（高散热）
- 输入：32个稀疏传感器读数
- 输出：舱内 $ m = 40000 $ 网格点温度分布
- 数据来源：公开基准数据集 [Chen et al., 2023]

---

### 实验设置与评估指标

#### 模型配置
- MTL-FNO主干：FNO标准结构（$ k=16 $ 模态，$ C=32 $ 通道，$ L=4 $ 层）
- CP分解秩：$ R = 8 $
- 激活函数：GeLU
- 优化器：Adam，初始学习率 0.001，每20轮衰减0.5
- 训练样本：每任务仅 **100个训练样本**，测试20个（few-shot设定）

#### 评估指标
| 指标 | 含义 |
|------|------|
| **MSE ↓** | 均方误差 |
| **MAE ↓** | 平均绝对误差 |
| **$ R^2 $ ↑** | 决定系数（越高越好） |
| **Params (M)** | 可训练参数总数（百万级） |
| **GFLOPs** | 浮点运算量 |
| **Inference Time (ms/sample)** | 单样本推理时间 |

#### 基线方法对比
- **U-Net**, **DeepONet**, **Senseiver**, **PhySense**, **FNO**
- 所有基线均为**独立单任务训练**

---

## 3. 主要实验结果和性能指标

### Case A 结果概览（五任务联合重建）
| 指标 | MTL-FNO | 5×独立FNO | 提升/压缩 |
|------|---------|------------|----------|
| **总参数量** | **1.30 M** | 5.45 M | ↓ **76%** |
| **平均推理时间** | 9.65 ms | ~1.57 ms × 5 | 可接受范围内 |
| **最佳 $ R^2 $ 表现** | 多项任务达最优或次优 | 多项任务严重失败（如 $ T_{xx} $: -4.121） |
| **应力场 $ T_{xy} $** | $ R^2 = \textbf{0.993} $ | $ R^2 = 0.880 $ | 显著提升 |

> ✅ MTL-FNO在所有任务上均保持高精度，尤其在FNO表现差的stress场中优势明显。

---

### Case B 结果概览（三工况温度重建）
| 指标 | MTL-FNO | 3×独立FNO | 提升/压缩 |
|------|---------|------------|----------|
| **总参数量** | **1.21 M** | 3.27 M | ↓ **60%** |
| **ADlet工况 $ R^2 $** | **0.970** | 0.825 (FNO) | 显著领先 |
| **DSine工况 $ R^2 $** | **0.988** | 0.986 (FNO) | 轻微超越 |
| **HSink工况 $ R^2 $** | **0.986** | 0.963 (FNO) | 更稳健 |

> ✅ MTL-FNO在不同热环境下均表现出更强的泛化能力和平衡性。

---

### 消融实验结果（Ablation Study）

#### 对比变体：
- **MTL-FNO-noshare**：无共享/特定划分 → 性能崩溃（$ T_{xx}: R^2 = -0.730 $）
- **MTL-FNO-nopolar**：无极分解 → 任务间干扰加剧（$ T_{xy}: R^2 $ 从 0.993 → 0.576）
- **MTL-FNO-nocayley**：无Cayley变换 → 单位性丧失 → $ T $ 场 $ R^2 $ 降至 0.414

| 组件 | 必要性说明 |
|------|-----------|
| **Layer-wise Sharing** | 是捕捉共性与个性的基础 |
| **Polar Decomposition** | 有效解耦相位/幅值，缓解任务冲突 |
| **Cayley Transform** | 保证酉结构，维持优化稳定性 |

> 🔍 所有组件缺一不可，共同构成高性能多任务框架。

---

### CP秩影响分析
- **Case A**：当 $ R < 8 $ 时性能显著下降；$ R \geq 8 $ 后趋于饱和
- **Case B**：即使 $ R=1 $ 也能取得较好结果，表明部分任务对低秩更鲁棒
- 最终选择 $ R=8 $ 作为默认配置，在性能与效率间取得平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **MTL-FNO实现了高效且准确的多场稀疏重建**：
   - 在两个代表性工程案例中，**模型大小减少60%-76%**，同时**精度持平或超越独立FNO**。
2. **多任务学习是小样本场景下的强正则化器**：
   - 利用跨场共性特征，显著提升了在有限数据下的泛化能力，避免过拟合。
3. **极分解+Cayley变换提供了物理意义明确的优化路径**：
   - 解耦相位与幅值优化不仅缓解了任务冲突，还增强了模型的可解释性和训练稳定性。
4. **低秩CP分解是实现参数高效的理想方式**：
   - 任务特定参数增量极小，支持模型灵活扩展至更多任务。

---

### 方法的局限性
- 当前框架假设为**稳态物理场**，尚未处理时间依赖性问题。
- 模型局限于**二维空间**，向三维及高维推广需进一步研究。
- 极分解与Cayley变换带来额外计算开销（约增加8–10ms推理延迟），虽可接受但仍非零成本。

---

### 未来工作方向
1. **拓展至时空联合建模**：将MTL-FNO扩展到time-dependent PDE求解，支持动态场预测。
2. **三维复杂结构适配**：结合图神经网络或自适应网格技术，应用于真实航天器结构。
3. **在线增量学习机制**：支持新任务动态加入而不重训全部模型。
4. **硬件协同优化**：探索MTL-FNO在FPGA或ASIC上的部署方案，进一步降低星载推理能耗。

--- 

> 📌 **总结**：MTL-FNO是一种面向星载应用的轻量化、高鲁棒性的多任务物理场重建框架，通过**硬参数共享 + 极分解解耦 + Cayley重参数化**三大核心技术，在**few-shot条件下实现了精度与效率的双重突破**，为下一代智能航天器的状态感知系统提供了有力的技术支撑。

</details>

---

### 15. [RAPNet: Accelerating Algebraic Multigrid with Learned Sparse Corrections](https://arxiv.org/abs/2605.26854)

**Authors**: Yali Fink, Ido Ben-Yair, Lars Ruthotto, Eran Treister  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.26854v1  

#### Abstract
The scalable solution of large sparse linear systems is a bottleneck in scientific computing and graph analysis. While algebraic multigrid (AMG) offers optimal linear scaling, its performance is severely constrained by the trade-off between the sparsity and convergence quality of coarse-grid operato...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RAPNet: Accelerating Algebraic Multigrid with Learned Sparse Corrections

## 1. 论文的主要贡献和创新点

### 解决的问题
- **AMG 中的稀疏性与收敛性权衡问题**：传统 Algebraic Multigrid (AMG) 方法在构建粗网格算子时面临根本矛盾——为了保证快速收敛，需要高质量的转移算子（如 smoothed aggregation），但这会导致算子密度显著增加（stencil growth），从而影响并行计算效率和可扩展性。
- **现有学习方法的局限性**：先前的神经加速方法通常存在以下问题：
  - 需要在每次迭代中进行神经网络推理（per-iteration inference），开销大；
  - 牺牲稀疏性（例如使用 dense attention）；
  - 无法泛化到大规模或深层层次结构；
  - 仅适用于固定右端项 $b$。

### 提出的新方法
- **RAPNet**：一种基于图神经网络（GNN）的框架，用于学习对 AMG 算子的**稀疏加性修正**（sparse additive corrections）。
- **核心思想**：
  - 利用经典的 piecewise-constant 聚合（unsmoothed aggregation）生成初始稀疏的 $P^{(0)}$, $R^{(0)}$, 和 $A_l$，确保基础稀疏性；
  - 使用 GNN 学习增量修正 $\Delta P$, $\Delta R$, $\Delta A_c$，这些修正被约束在原始聚合模式定义的非零位置上，因此保持了严格的稀疏性；
  - 所有学习过程仅发生在求解器的 **setup phase**，一旦 hierarchy 构建完成，后续的 solve phase 完全由标准 AMG V-cycle 执行，不涉及任何额外的神经网络计算。

### 相比现有方法的优势
| 特性 | RAPNet | 经典 AMG (SpSA) | 其他神经方法 |
|------|--------|----------------|-------------|
| **稀疏性** | ✅ 严格保持 | ❌ 显著增加（stencil growth） | ❌ 通常引入稠密操作 |
| **推理阶段** | ✅ 仅 setup 阶段 | N/A | ❌ 多数需每步迭代推理 |
| **可扩展性** | ✅ 支持百万节点、深层 hierarchy | ⚠️ setup 成本高 | ❌ 通常限于两层或小规模 |
| **多查询任务支持** | ✅ 可处理任意 $b$ | ✅ | ❌ 多数针对特定 $b$ |
| **计算属性保留** | ✅ Solve phase 仍为标准 AMG | ✅ | ❌ 改变求解流程 |

---

## 2. 核心实验方法和设置

### 数据集
实验覆盖了多种类型的稀疏线性系统，分为两大类：

#### 图拉普拉斯算子（Graph Laplacians）
- **2D/3D Geometric Graphs**：通过单位超立方体内的随机点集进行 Delaunay 三角剖分生成。
- **Watts-Strogatz Graphs**：具有“小世界”特性的图，高聚类系数与短平均路径长度。
- **Temporal Barabasi-Albert (TBA)**：动态尺度自由网络，用于模拟时间演化图。
- **Synthetic Social Hub**：人工构造的社会网络拓扑，包含高度连接的枢纽节点。

#### PDE 离散化系统
- **3D Anisotropic Diffusion**：各向异性扩散方程的有限元离散。
- **3D Advection-Diffusion**：对流占优的稳态对流-扩散方程。

> 📌 **训练与测试分离策略**：模型在小型图（约 4000 节点）或 2D 结构化网格上训练，在高达百万节点的真实 3D 非结构化网格上测试，验证其跨尺度和维度的泛化能力。

### 实验设置与评估指标
- **任务形式**：作为独立求解器（standalone solver）和 GMRES 的预条件子（preconditioner）。
- **评估指标**：
  - 达到相对残差 $10^{-6}$ 所需的**迭代次数**（iteration count）；
  - setup 阶段耗时（秒）；
  - 收敛率（convergence rate）与稳定性。
- **硬件配置**：
  - AGG / SpSA：运行于多核服务器级 CPU；
  - RAPNet：setup 包含 AGG + GPU 上的 GNN 推理 + 稀疏矩阵修正。

### 基线方法对比
- **AGG (Plain Aggregation)**：未平滑的聚合，最稀疏但收敛慢。
- **SpSA (Sparsified Smoothed Aggregation)**：经典非 Galerkin 方法，先计算密集的 smoothed aggregation，再剪枝以恢复稀疏性，是强有力的基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 实验 | Vars | AGG (Solver) | SpSA (Solver) | **RAPNet (Solver)** | AGG (GMRES) | SpSA (GMRES) | **RAPNet (GMRES)** |
|------|------|---------------|----------------|----------------------|--------------|----------------|--------------------|
| 2D Geometric | 1M | 163±2 | 148±203 | **84±1** | 49±1 | 36±1 | **37±1** |
| 3D Geometric | 512K | 145±18 | 65±8 | **71±6** | 42±5 | 32±2 | **34±2** |
| Watts-Strogatz | 1M | 453±136 | — | **325±69** | 549±439 | 414±383 | **130±20** |
| Temporal BA | 600K | 106±13 | 90±152 | **85±45** | 28±3 | 25±25 | **44±7** |
| Social Hub | 300K | 61±166 | 978±115 | **18±12** | 13±4 | 34±94 | **9±1** |
| 3D FEM Adv-Diff | 524K | 906±285 | 875±324 | **62±15** | 80±28 | 42±9 | **38±5** |

> ✅ **结论**：RAPNet 在几乎所有测试场景下都显著减少了迭代次数，尤其在复杂图（如 Social Hub）和病态 PDE（如 Advection-Diffusion）上优势明显。

### Setup 时间对比（Table 2）
| 实验 | Vars | AGG (s) | SpSA (s) | **RAPNet (s)** |
|------|------|---------|----------|----------------|
| Social Hub | 300K | 0.53 | 973.16 | **0.76** |
| TBA | 600K | 1.11 | 84.61 | **1.67** |

> ⚠️ **发现**：SpSA 的 setup 时间随图连通性急剧上升（非线性增长），而 RAPNet 仅比 AGG 多一个轻量级 GNN 推理，总时间仍远低于 SpSA。

### 消融实验结果（Table 3）
在 3D Geometric 图上的 ablation study 表明：
- 移除 $\Delta A_c$ 修正 → 迭代数从 **59±5** 升至 **83±7**
- 移除 $\Delta P, \Delta R$ → 迭代数升至 **99±10**
- 移除层级混合机制（mixing）→ 性能下降

> ✅ **结论**：所有组件（特别是 $\Delta A_c$ 和层级依赖建模）对最终性能均有重要贡献。

---

## 4. 关键结论和发现

### 主要发现
1. **成功打破稀疏性-收敛性权衡**：RAPNet 通过 GNN 学习稀疏修正，在保持 AGG 级别稀疏性的前提下，实现了接近甚至超越 SpSA 的收敛速度。
2. **高效且可扩展的 setup 设计**：仅在 setup 阶段执行一次 GNN 推理，solve 阶段完全兼容传统 AMG 流程，适合多查询任务（如 eigenproblems, inverse problems）。
3. **强大的跨尺度泛化能力**：采用 level-wise、共享权重的 GNN 架构，结合局部子图训练策略，使模型能够从数千节点训练推广到百万节点推理。
4. **GPU 加速友好**：GNN 部分天然适合 GPU 并行计算，setup 时间可控，尤其在高连通图上相比 SpSA 有数量级优势。

### 方法的局限性
- **未优化平滑器参数**：当前方法仅修正 AMG hierarchy 中的算子，未学习或调整 Jacobi 平滑器的 damping 参数。
- **未学习聚合策略**：节点聚类（aggregation）仍使用经典启发式算法，未由 GNN 学习最优聚集方式。
- **缺乏理论收敛保证**：尽管实验表现优异，但 RAPNet 不受非 Galerkin 理论的小扰动假设限制，因此目前没有严格的收敛性证明。
- **依赖初始聚合质量**：性能受限于初始 AGG hierarchy 的质量，若初始聚合不当可能影响修正效果。

### 未来工作方向
- 将 GNN 扩展至学习**自适应聚合策略**（adaptive aggregation）；
- 探索学习**最优平滑器**（learned smoothers）或其参数；
- 将 RAPNet 应用于更广泛的 saddle-point 系统（如 Stokes, Navier-Stokes）；
- 开发具备**理论收敛保障**的学习型 AMG 框架；
- 探索在分布式 HPC 环境下的部署与通信优化。

---

> 🔚 **总体评价**：RAPNet 是将深度学习与经典数值线性代数方法深度融合的成功范例。它不仅在性能上超越了经典非 Galerkin 方法，更重要的是提出了一种“**setup-time learning, solve-time efficiency**”的新范式，为机器学习加速科学计算提供了可扩展、实用且理论透明的新路径。

</details>

---

### 16. [Telenor Nordics Customer Service self-help corpus](https://arxiv.org/abs/2605.26891)

**Authors**: Mike Riess  
**Category**: cs.CL  
**Published**: 2026-05-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.26891v1  

#### Abstract
This paper presents a multilingual customer service self-help corpus comprising 1,122 manually validated documents in Finnish, Danish, Norwegian, and Swedish, totaling over one million tokens. The documents have been sourced from the public self-help pages of four Nordic telecommunications operators...

---

### 17. [Learning to Adapt SFT Data for Better Reasoning Generalization](https://arxiv.org/abs/2605.26924)

**Authors**: Lisong Sun, Li Wang, Chen Zhang, Jinyang Wu, Kui Zhang, Tianhao Peng, Wenjun Wu  
**Category**: cs.CL  
**Published**: 2026-05-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.26924v1  

#### Abstract
Large language models (LLMs) have achieved remarkable progress, with post-training playing a crucial role in enhancing their reasoning capabilities. Among post-training paradigms, supervised fine-tuning (SFT) is widely used: it leverages external data to provide dense supervision and enables efficie...

---

### 18. [Share More, Search Less: Collaborative Parallel Thinking for Efficient Test-Time Scaling](https://arxiv.org/abs/2605.27030)

**Authors**: Xinglin Wang, Hao Lin, Shaoxiong Feng, Peiwen Yuan, Yiwei Li, Jiayi Shi, Yueqi Zhang, Chuyi Tan, Ji Zhang, Boyuan Pan, Yao Hu, Kan Li  
**Category**: cs.CL  
**Published**: 2026-05-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.27030v1  

#### Abstract
Test-Time Scaling (TTS) enhances the reasoning capabilities of large language models by allocating additional inference compute to explore the solution space. However, existing parallel TTS methods typically keep branches isolated during search: intermediate discoveries remain branch-private and can...

---

### 19. [Neural Bayesian Sequential Routing](https://arxiv.org/abs/2605.26147)

**Authors**: Yongchao Huang  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.26147v1  

#### Abstract
Human decision-making is sequential and uncertainty-aware, yet standard neural networks often rely on static, dense forward computation with limited visibility into evidence acquisition, uncertainty evolution, or when computation should stop. We introduce \textbf{Neural Bayesian Sequential Routing (...

---

### 20. [Scaling World-Model Reinforcement Learning Through Diffusion Policy Optimization](https://arxiv.org/abs/2605.26282)

**Authors**: Xiaoyuan Cheng, Wenxuan Yuan, Zhancun Mu, Yuanzhao Zhang, Yiming Yang, Hai Wang, Zhuo Sun, Che Liu  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.26282v1  

#### Abstract
Model-based reinforcement learning (RL) can be effectively supported at scale through the use of world models. However, in practice, scaling such approaches remains fundamentally limited. A commonly recognized challenge is model bias and error compounding, which degrade long-horizon predictions. Bey...

---

### 21. [Amortized Factor Inference Networks for Posterior Inference](https://arxiv.org/abs/2605.26419)

**Authors**: Joohwan Ko, Justin Domke  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.26419v1  

#### Abstract
Amortized inference promises fast test-time Bayesian inference, but existing methods are inherently tied to fixed models. Extending amortization to unseen models typically requires retraining or costly test-time finetuning. In this paper, we ask: is it possible to build a single inference network ca...

---

### 22. [Pretrained Approximators for Low-Thrust Trajectory Cost and Reachability](https://arxiv.org/abs/2605.26790)

**Authors**: Zhong Zhang, Giacomo Acciarini, Dario Izzo, Hexi Baoyin, Francesco Topputo  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.26790v1  

#### Abstract
Low-thrust trajectory design relies heavily on repeated evaluations of fuel consumption and transfer feasibility, which require expensive optimal control solutions. In this work, we show these quantities can be accurately approximated by machine learning surrogates, enabling fast and scalable evalua...

---

### 23. [FAB-Bench: A Framework for Adaptive RAG Benchmarking in Semiconductor Manufacturing](https://arxiv.org/abs/2605.26476)

**Authors**: Jingbin Qian (FutureFab.AI), Congwen Yi (FutureFab.AI), Min Xia (FutureFab.AI), Wen Wu (FutureFab.AI), Jun Zhu (FutureFab.AI), Jian Guan (FutureFab.AI)  
**Category**: cs.CL  
**Published**: 2026-05-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.26476v1  

#### Abstract
Retrieval-Augmented Generation (RAG) has become critical for knowledge-intensive applications, yet evaluating its performance in vertical domains remains difficult due to domain complexity, diverse context scales, and heavy reliance on expert assessments that are costly, inconsistent, and non-scalab...

---

### 24. [Open-Weight LLM Fine-Tuning Defenses are Susceptible to Simple Attacks](https://arxiv.org/abs/2605.26526)

**Authors**: Kevin Kuo, Chhavi Yadav, Virginia Smith  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.26526v1  

#### Abstract
Recent defenses for safeguarding open-weight large language models (LLMs) are intended to prevent adversarial usage. Underlying these defenses is an assumption that new harmful behavior is learned through fine-tuning rather than elicited by jailbreaking the model. Yet, pretrained LLMs already encode...

---

### 25. [Time Series Causal Discovery via Context-Conditioned and Causality-Augmented Pretraining](https://arxiv.org/abs/2605.26759)

**Authors**: Biao Ouyang, Tengxue Zhang, Zhihao Zhuang, Yang Shu, Chenjuan Guo, Bin Yang  
**Category**: cs.LG  
**Published**: 2026-05-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.26759v1  

#### Abstract
Causal discovery from time series is critical for many real-world applications, such as tracing the root causes of anomalies. Existing approaches typically rely on dataset-specific optimization, making it difficult to transfer their causal discovery capabilities to new time series governed by divers...

---

### 26. [BrickAnything: Geometry-Conditioned Buildable Brick Generation with Structure-Aware Tokenization](https://arxiv.org/abs/2605.26182)

**Authors**: Zhengyang Ni, Feng Yan, Yu Guo, Fei Wang  
**Category**: cs.AI  
**Published**: 2026-05-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.26182v1  

#### Abstract
Generating physically buildable brick structures from 3D shapes requires more than geometric reconstruction: the output must also satisfy discrete part constraints and structural stability. Existing brick generation methods either rely on heuristic optimization, which can break down when the target ...

---

### 27. [FAST-GOAL: Fast and Efficient Global-local Object Alignment Learning](https://arxiv.org/abs/2605.26615)

**Authors**: Hyungyu Choi, Young Kyun Jang, Chanho Eom  
**Category**: cs.AI  
**Published**: 2026-05-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.26615v1  

#### Abstract
Vision-language models such as CLIP have shown impressive capabilities in aligning images and text, but they often struggle with lengthy and detailed text descriptions due to pre-training on short and concise captions. We present FAST-GOAL (Fast and Efficient Global-local Object Alignment Learning),...

---

### 28. [LiveK12Bench: Have Large Multimodal Models Truly Conquered High School-level Examinations?](https://arxiv.org/abs/2605.26781)

**Authors**: Xiaohan Wang, Mingze Yin, Yilin Zhao, Gang Liu, Dian Li  
**Category**: cs.AI  
**Published**: 2026-05-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.26781v1  

#### Abstract
Advanced Large Multimodal Models (LMMs) have demonstrated impressive performance in K-12 reasoning tasks, exhibiting great promise as intelligent tutors. Realizing this potential requires models to navigate real-world examinations effectively, yet most existing benchmarks fail to capture the complex...

---

### 29. [LELA: An End-to-end LLM-based Entity Linking Framework with Zero-shot Domain Adaptation](https://arxiv.org/abs/2605.26956)

**Authors**: Samy Haffoudhi (IP Paris, LTCI, DIG), Nikola Dobri\v{c}i\'c (IP Paris), Fabian Suchanek (IP Paris, LTCI), Nils Holzenberger  
**Category**: cs.AI  
**Published**: 2026-05-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.26956v1  

#### Abstract
Entity linking is a key component of many downstream NLP systems, yet existing approaches are often tied to the specific target knowledge bases and domains, limiting their real world application. In this paper, we extend LELA, a modular and domain-agnostic LLM-based entity disambiguation method, int...

---

### 30. [Learning to Act under Noise: Enhancing Agent Robustness via Noisy Environments](https://arxiv.org/abs/2605.27209)

**Authors**: Yuxin Chen, Xiaodong Cai, Junfeng Fang, Zhuowen Han, Yu Wang, Yaorui Shi, Yi Zhang, Qi Gu, Xunliang Cai, Xiang Wang, An Zhang, Tat-Seng Chua  
**Category**: cs.AI  
**Published**: 2026-05-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.27209v1  

#### Abstract
Recent advances in large language models (LLMs) have facilitated the widespread deployment of LLMs as interactive agents capable of reasoning, planning, and tool use. Despite strong performance on existing benchmarks, such agents often exhibit notable degradation when deployed in real-world settings...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-18 06:45:31 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Accelerated Predictive Coding Networks via Direct Kolen-Pollack Feedback Alignment](https://arxiv.org/abs/2602.15571)

**Authors**: Davide Casnici, Martin Lefebvre, Justin Dauwels, Charlotte Frenkel  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.15571v1  

#### Abstract
Predictive coding (PC) is a biologically inspired algorithm for training neural networks that relies only on local updates, allowing parallel learning across layers. However, practical implementations face two key limitations: error signals must still propagate from the output to early layers throug...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerated Predictive Coding Networks via Direct Kolen-Pollack Feedback Alignment

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本论文针对 **Predictive Coding (PC)** 在实际应用中的两个关键瓶颈：
- **Feedback Error Delay**：误差信号必须从输出层逐层反向传播到浅层，导致推理阶段需要至少与网络深度 $O(L)$ 成正比的时间步数，限制了并行性和效率。
- **Exponential Error Decay**：在逐层传播过程中，误差信号因学习率衰减而呈指数级衰减，导致深层网络中早期层的更新信号极弱（vanishing updates）。

这两个问题严重削弱了 PC 作为生物可解释且硬件友好的 BP 替代方案的潜力。

### 提出了什么新方法或新思路
作者提出了 **Direct Kolen-Pollack Predictive Coding (DKP-PC)**，一种全新的 PC 变体，其核心思想是：
- 将 **Direct Kolen-Pollack (DKP)** 算法的机制嵌入到 PC 框架中。
- 引入**可学习的直接反馈连接**（learnable direct feedback connections），从输出层 $L$ 直接连接到所有隐藏层 $l \in \{1, ..., L-1\}$。
- 利用这些反馈连接，在推理阶段开始前，通过一个初步的权重更新（基于 DKP 规则），将输出误差 $\delta_L$ **瞬时地**（instantaneously）投射到每一层，从而在所有层同时生成非零的预测误差。

### 相比现有方法的优势
- **理论时间复杂度从 $O(L)$ 降至 $O(1)$**：由于误差不再需要逐层传播，理论上错误传播的延迟被消除，使得整个学习过程（特别是推理阶段）的时间复杂度与网络深度无关。
- **完全并行化**：DKP-PC 是首个能够实现**全层并行学习**的 PC 变体，无论批量大小如何，都能释放 PC 的理论并行潜力。
- **缓解梯度消失**：直接投影避免了误差在多步迭代中的指数衰减，确保了早期层获得更强、更稳定的更新信号。
- **保留局部性**（locality）：所有权重更新规则仅依赖于局部信息，保持了生物可解释性和硬件友好性。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验涵盖了从小型到大型的多种网络架构和数据集：
- **MLP 实验**：
  - 数据集：**MNIST**, **Fashion-MNIST (FMNIST)**
  - 架构：三层 MLP（含两个 128 单元的隐藏层）
- **CNN 实验**：
  - 数据集：**CIFAR-10**, **CIFAR-100**, **Tiny ImageNet**
  - 架构：**VGG-7** 和 **VGG-9** 类似网络

### 实验设置和评估指标
- **训练设置**：
  - 批量大小（batch size）：128
  - 优化器：Adam/AdamW（前向权重），独立优化器用于反馈连接
  - 学习率调度：warmup-cosine-annealing（前向权重），指数衰减（反馈连接）
  - 数据增强：在 CIFAR 和 Tiny ImageNet 上使用随机裁剪
- **评估指标**：
  - **分类准确率**（Test Accuracy）：Top-1 和 Top-5 准确率
  - **训练速度**（Training Speed）：每个 epoch 的耗时（秒）
  - **计算效率**：浮点运算次数（FLOPs）
  - **梯度对齐度**（Gradient Alignment）：通过余弦相似度衡量算法产生的梯度与 BP 梯度的方向一致性。

### 基线方法对比
论文将 DKP-PC 与以下方法进行了全面比较：
- **Backpropagation (BP)**：标准反向传播，黄金标准。
- **Direct Kolen-Pollack (DKP)**：直接反馈对齐的改进版。
- **Standard Predictive Coding (PC)**：标准预测编码。
- **Incremental Predictive Coding (iPC)**：通过交替更新来部分缓解延迟问题的 PC 变体。
- **Center-Nudging PC (CN-PC)**：另一种先进的 PC 变体。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **分类准确率**（见 Table 1）：
  - 在 **Tiny ImageNet** 上，DKP-PC 达到了 **35.04%** 的 Top-1 准确率，显著优于其他本地学习算法（如 CN-PC 的 31.50%）。
  - 在 **VGG-9/CIFAR-100** 上，DKP-PC 的 Top-1 准确率达到 **53.80%**，比标准 PC 高出约 14%，比 iPC 高出约 9%。
  - 在小型 MLP 任务上，所有方法表现相近，DKP-PC 依然具有竞争力。
- **训练速度**（见 Table 2）：
  - DKP-PC 的训练速度远超其他 PC 变体。
  - 在 VGG-7 和 VGG-9 上，相比标准 PC，DKP-PC 实现了**超过 60% 的训练时间减少**。
  - 相比 iPC，训练时间减少了约 **81%**。
- **计算效率**：
  - 由于只需**单步推理**（single inference step），DKP-PC 的 FLOPs 要求远低于需要 $T \geq L$ 步的 PC 和 iPC，计算效率优势接近一个数量级。

### 与基线方法的对比结果
- **vs. PC/iPC**：DKP-PC 在所有任务上均大幅超越，尤其是在更深、更复杂的网络（如 VGG-9 on Tiny ImageNet）上优势明显，同时训练速度快得多。
- **vs. DKP**：DKP-PC 在所有设置下都优于 vanilla DKP，表明 PC 阶段对 DKP 的梯度起到了有效的正则化和对齐作用。
- **vs. BP**：虽然仍有一定差距，但 DKP-PC 大幅缩小了与 BP 的性能鸿沟，特别是在深层网络上，证明了本地学习算法可以逼近 BP 的效率。

### 消融实验结果
- **梯度对齐分析**（Figure 3）：
  - 实验证明，DKP-PC 产生的前向权重梯度与 BP 梯度的**余弦相似度更高、收敛更快、更稳定**。
  - 若移除 PC 阶段的前向权重更新（`DKP-PC (No Forward)`），梯度对齐会崩溃，证明了 PC 阶段对注入对齐信息至关重要。
  - 若移除反馈权重的更新（`DKP-PC (No Feedback)`），对齐度也会下降，说明 PC 阶段同样改善了反馈路径的学习。
- **推理步数分析**（Figure 5）：
  - 即使只进行**一步推理**，DKP-PC 的性能已能超越标准 PC 和 iPC。
  - 增加推理步数可以进一步提升准确率，揭示了**性能与训练时间之间的权衡**，而 DKP-PC 在该权衡上表现更优。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **DKP-PC 成功解决了 PC 的核心缺陷**：通过引入直接可学习的反馈，同时消除了 **error delay** 和 **exponential decay** 两大问题。
2. **实现了真正的并行化**：DKP-PC 是首个能在理论上实现 $O(1)$ 时间复杂度的 PC 算法，为高效硬件实现铺平了道路。
3. **协同效应显著**：DKP 和 PC 并非简单叠加，而是存在**协同作用**——DKP 加速了 PC，而 PC 反过来正则化并提升了 DKP 的梯度质量，使其更接近 BP。
4. **性能与效率兼得**：DKP-PC 不仅在准确率上超越了现有的本地学习算法，还在训练速度和计算效率上取得了巨大提升。

### 方法的局限性
- **PyTorch 实现的并行开销**：当前在 PyTorch 中的实现未能充分利用其并行潜力，因为标准的并行化会带来显著的线程管理和同步开销，抵消了部分加速效果。
- **内存开销**：引入额外的可学习反馈矩阵 $V_l$ 会增加模型的内存占用。
- **与 BP 的最终性能仍有差距**：尽管大幅缩小了差距，但在最复杂的任务上，DKP-PC 的准确率仍未完全达到 BP 的水平。

### 未来工作方向
- **定制化硬件实现**：开发专用的 **CUDA kernels** 或 **神经形态硬件** 来充分挖掘 DKP-PC 的并行潜力，有望实现比 BP 更低的训练延迟。
- **降低反馈连接的开销**：探索反馈权重的**稀疏化**（sparsity）和**量化**（quantization）以减少内存占用。
- **更深层次的融合**：研究将 DKP-PC 与其他先进 PC 变体（如基于均衡传播的 nudging PC）结合的可能性。
- **动态扰动机制**：探索不通过初步权重更新，而是**直接利用反馈信息扰动神经活动动态**的新方法，可能实现更快的局部更新规则。

</details>

---

### 2. [ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns](https://arxiv.org/abs/2602.15521)

**Authors**: Ziyu Zhao, Tong Zhu, Zhi Zhang, Tiantian Fan, Jinluan Yang, Kun Kuang, Zhongyu Wei, Fei Wu, Yu Cheng  
**Category**: cs.CL  
**Published**: 2026-02-18  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.15521v1  

#### Abstract
Mixture-of-Experts (MoE) effectively scales model capacity while preserving computational efficiency through sparse expert activation. However, training high-quality MoEs from scratch is prohibitively expensive. A promising alternative is to convert pretrained dense models into sparse MoEs. Existing...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
当前，**Mixture-of-Experts (MoE)** 架构虽然能有效扩展模型容量并保持推理效率，但从零训练高质量 MoE 模型成本极高。一种替代方案是将预训练的 **dense LLMs** 转换为 MoE，即“**dense-to-MoE conversion**”。然而，现有方法存在以下问题：
- **破坏了 dense 模型内在的激活模式**（如 GLU 的动态门控机制），导致专家构造次优；
- 忽略了不同层之间的功能差异，采用统一配置；
- 多数方法需要额外训练路由器（router），增加了开销。

### **提出了什么新方法或新思路**
本文提出 **ExpertWeaver** —— 一种**无需训练**（training-free）的框架，利用 **GLU 激活模式** 将 dense 模型转换为结构化的 MoE 架构。

#### **核心思想：GLU 是 MoE 的天然蓝图**
- GLU 中的 **gating signal** 提供了细粒度（neural-level）的动态激活控制。
- 分析发现，这些信号中隐含了粗粒度（expert-level）的功能结构：
  - 存在一组**普遍激活的通用神经元**（universal neurons）→ 可组成 **shared expert**；
  - 存在**任务特定共激活的专用神经元** → 可聚类为多个 **routed experts**；
  - 不同层的神经元专业化程度不同 → 应进行**逐层自适应配置**。

### **相比现有方法的优势**
| 维度 | 现有方法 | ExpertWeaver |
|------|--------|------------|
| 是否需训练 | 多数需要训练 router 或继续预训练 | 完全 **training-free** |
| 利用内部结构 | 忽视 GLU 激活模式 | 显式利用 GLU gating 作为专家划分依据 |
| 层间差异处理 | 固定配置 across layers | **layer-adaptive allocation** |
| 专家构建方式 | 随机/简单聚类 | 基于多任务激活模式的平衡聚类 |
| 路由器构造 | 需学习 | 通过 gating 向量均值直接构造 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**

#### **校准集（Calibration Dataset）**
用于捕获神经元激活模式：
- **Flan-v2**：包含 48 个任务、10 个任务簇，每个任务取 5 个样本，共 240 条数据。
- 用于提取每层神经元的 **multi-task gating activation patterns**。

#### **下游评估基准**
涵盖多种能力测试：
- **MMLU**：多学科知识理解
- **HellaSwag**：常识推理
- **ARC-e / ARC-c**：科学问答（易/难）
- **PIQA**：物理常识
- **WinoGrande**：指代消解
- **LogiQA**：逻辑推理
- **SciQ**：科学选择题
- **GSM8K**, **HumanEval**, **IFEval**：指令微调后的能力评估

#### **预训练数据**
- **FineWeb-Edu**：用于 continued pretraining (CPT)，共 200B tokens。

---

### **实验设置和评估指标**

#### **两种应用场景**
1. **Training-free Dynamic Structural Pruning**  
   - 目标：低稀疏度下提升推理效率（如 25% sparsity）
   - 输出形式：共享专家始终激活 + 动态选择 routed experts
   - 评估指标：各 benchmark 的准确率（Accuracy）

2. **Model Downcycling**  
   - 目标：将大 dense 模型转为高效 MoE，再进行有限 CPT
   - 示例：Qwen2.5-7B → MoE with 62 routed + 2 shared experts, 激活 3.5B params/token
   - 评估：CPT 后与同类参数量模型比较性能

#### **评估指标**
- 主要：平均准确率（Avg. Accuracy）
- 推理效率：吞吐量（RPS, OTPS）、延迟（TTFT, TPOT）
- 训练效率：训练损失收敛速度

---

### **基线方法对比**

#### **Structural Pruning 基线**
- **LLM-Pruner**：基于梯度和幅值的重要性剪枝
- **FLAP**：基于输出特征稳定性的剪枝
- **CMoE**：基于平衡聚类的 dense-to-MoE 方法（analytical router）

#### **Downcycling 基线**
- **OLMoE**：从头训练的 MoE 模型（500B tokens）
- **LLaMA-MoE-v1/v2**：通过持续预训练构建的 MoE
- **OpenMoE**：开源 MoE 系列
- 多个 dense 模型（如 OPT-2.7B, Pythia-2.8B, Gemma-2-2B 等）

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### **动态结构剪枝（25% sparsity）**
| Method | MMLU | HellaSwag | ARC-e | ARC-c | PIQA | **Avg** |
|--------|------|-----------|-------|-------|-------|--------|
| Dense (Qwen2.5-7B) | 74.2 | 80.3 | 77.8 | 63.8 | 80.0 | 75.2 |
| LLM-Pruner | 55.9 | 72.2 | 71.0 | 49.1 | 77.0 | 65.0 |
| FLAP | 54.7 | 58.5 | 67.3 | 42.2 | 70.8 | 58.7 |
| **ExpertWeaver** | **61.6** | **72.3** | **71.5** | **53.5** | **76.3** | **67.0** |

> ✅ **相对 LLM-Pruner 提升 3.1%，显著优于所有 baseline**

#### **复杂任务上的表现（GSM8K, HumanEval）**
| 方法 | GSM8K (25%) | GSM8K (12.5%) | HumanEval (25%) |
|------|-------------|---------------|------------------|
| LLM-Pruner | 2.0 | – | 14.6 |
| FLAP | 14.8 | 18.9 | 57.6 |
| **ExpertWeaver** | **34.9** | **18.9** | **64.6** |

> ✅ 在高稀疏度下仍保持强大推理能力，静态剪枝严重退化

---

#### **Downcycling 性能对比（CPT 200B tokens）**
| Model | MMLU | HellaSwag | ARC-e | ARC-c | PIQA | **Avg** |
|-------|------|-----------|-------|-------|-------|--------|
| Qwen2.5-7B (dense) | 74.1 | 80.2 | 77.5 | 63.7 | 79.7 | 73.2 |
| OLMoE-1B-7B* (500B) | 53.8 | 79.6 | 76.3 | 55.6 | 80.1 | 69.1 |
| **ExpertWeaver (Qwen2.5-7B)** | **73.7** | **72.4** | **56.3** | **78.0** | **65.3** | **63.5** |

> ✅ **以仅 200B tokens 训练，达到 OLMoE* 97.35% 的性能**
>
> ✅ 激活参数仅为原模型 1/4，保留 87.6% 性能

#### **与更小 dense 模型对比**
- 相比 **Qwen2.5-3B** 和 **Llama-3.2-3B**（分别训练于 18T 和 9T 数据）：
  - ExpertWeaver 达到其 **93.0% 和 96.7%** 的性能
  - 证明 downcycling 是高效的 MoE 构建路径

---

### **消融实验结果**

#### **Ablation on Hyperparameters**
- **共享专家比例**（amin, amax）：
  - 最佳配置：`amin=0.2`, `amax=0.7`
  - 层感知动态分配优于固定比例（对角线性能更低）
- **专业化阈值 T**：
  - T=0.6 时性能最优
- **专家粒度（Expert Granularity）**：
  - 64 或 128 个专家时性能最佳
  - 过少 → 缺乏多样性；过多 → 效率下降

#### **校准集影响**
| 设置 | Avg Score |
|------|----------|
| Flan-v2 (100%) | **67.0** |
| Flan-v2 (50%) | 66.1 |
| C4-only | 65.8 |

> ✅ 多任务、多样化数据对专家划分至关重要

---

## 4. **关键结论和发现**

### **主要发现**
1. **GLU gating signals 揭示了 dense 模型中固有的 MoE 结构**：
   - 通用神经元 vs. 专用神经元
   - 共激活模式可指导专家聚类
2. **不同层具有不同的专业化分布**：
   - 浅层和深层更多通用神经元（low CV）
   - 中间层更多专用神经元（high CV）
   - 支持 **layer-adaptive expert allocation**
3. **ExpertWeaver 成功“编织”出功能分明的 MoE 架构**：
   - 浅层广泛路由，深层高度专业化（见 Fig. 8）
   - 路由行为符合认知层级假设

---

### **方法的局限性**
- **依赖 GLU 架构**：目前仅适用于 SwiGLU 类模型，不适用于 ReLU-based FFN。
- **专家粒度固定**：需预先设定专家数量和大小，缺乏完全自适应能力。
- **校准集质量敏感**：若任务覆盖不足，可能导致专家划分偏差。
- **硬件优化未深入探讨**：尽管推理效率提升，但未针对特定硬件做 kernel 优化。

---

### **未来工作方向**
- 扩展至非 GLU 架构（如 ReLU, GeLU）的 dense 模型
- 探索更细粒度或可变大小的专家结构
- 结合 upcycling 与 downcycling，实现跨规模模型迁移
- 在 vision-language 模型中验证该范式是否通用
- 开发轻量化版本用于边缘部署

---

> 💡 **一句话总结**：  
> **ExpertWeaver 发现了 GLU 激活模式中的“隐藏 MoE”，并首次实现了无需训练、基于内在功能结构的 dense-to-MoE 转换，在结构剪枝与模型降级两条路径上均取得 SOTA 表现。**

</details>

---

### 3. [1-Bit Wonder: Improving QAT Performance in the Low-Bit Regime through K-Means Quantization](https://arxiv.org/abs/2602.15563)

**Authors**: Sohir Maskey, Constantin Eichenberg, Johannes Messner, Douglas Orr  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.15563v1  

#### Abstract
Quantization-aware training (QAT) is an effective method to drastically reduce the memory footprint of LLMs while keeping performance degradation at an acceptable level. However, the optimal choice of quantization format and bit-width presents a challenge in practice. The full design space of quanti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**1-Bit Wonder: Improving QAT Performance in the Low-Bit Regime through K-Means Quantization**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在解决在**固定推理内存预算下，如何权衡模型参数量（N）与权重精度（P）**以最大化下游生成任务性能的问题。当前关于低比特量化（如 1-bit、2-bit）的有效性存在争议：
- 一些研究基于训练损失或困惑度（perplexity）声称极低比特量化可行；
- 另一些研究则指出在低于 6-bit 时性能严重下降。

这些矛盾部分源于**评估指标不一致**（仅用 loss 而非生成能力）、**量化格式设计差异**（未充分探索非线性格式），以及缺乏对 QAT 在低比特下的系统性分析。

### 提出的新方法与新思路
1. **引入 k-means 量化作为非线性量化格式**  
   - 使用 1D k-means 聚类学习最优的量化中心点（centroids），而非传统均匀整数量化（uniform integer quantization）中的等距划分。
   - 这种格式能最小化 L2 重建误差，在低比特下更贴合实际权重分布。

2. **提出“精度感知缩放律”（precision-aware scaling laws）**  
   - 将有效参数量定义为 $ N_{\text{eff}} = N \cdot f(P) $，其中 $ f(P) = 1 - \exp(-P/\gamma) $ 是一个饱和函数，建模精度提升带来的边际收益递减。
   - 基于此，推导出在固定内存预算 $ M = N \cdot P $ 下，应最大化单位内存的有效容量 $ g(P) = f(P)/P $。

3. **验证 1-bit 权重量化在真实生成任务中仍可扩展**  
   - 首次通过完整的预训练 + SFT 流程，展示了 31B 参数、1.25-bit 模型在多种生成任务上优于较小的高精度模型。

### 相比现有方法的优势
| 方面 | 本文方法 | 传统方法 |
|------|---------|----------|
| **量化格式** | k-means 非线性量化，适应权重分布 | uniform 整数量化，固定间隔 |
| **稳定性** | 1-bit 下无需特殊归一化技巧即可稳定训练 | 1-bit 整数量化常需 mean-shift 或其他 trick |
| **性能上限** | 在相同内存预算下，1-bit + 更大 N 表现最佳 | 多数认为 4–6 bit 是极限 |
| **硬件可行性** | 使用 lookup table 实现高效 inference kernel | 依赖原生硬件支持（如 int4） |

---

## 2. 核心实验方法和设置

### 数据集
- **预训练数据**：
  - 主要使用 `Nemotron-CC`（经过过滤的 Common Crawl 子集）
  - 长周期训练还加入了 `Starcoder-V2`（代码） 和 `FineMath-3+/4+`（高质量数学数据），采用 curriculum learning 策略逐步增加高质量数据比例。
- **SFT 数据**：`Tulu 3 SFT Mixture`，用于指令微调。

### 模型架构与训练配置
- 架构：基于 **Llama 3** 的 decoder-only Transformer
  - 使用 RoPE、RMSNorm、SwiGLU、Grouped Query Attention
  - 所有模型从零开始训练（scratch training）
- QAT 设置：
  - Warm-up 1000 步后开启量化
  - Block-wise 量化，块大小 B=64，scale 存储为 16-bit
  - k-means centroids 在 QAT 开始时学习并冻结
- 硬件：64 × NVIDIA H100 GPUs，使用 FSDP

### 基线对比
| 模型 | 参数量 | 精度 | 内存占用 | 类型 |
|------|--------|-------|-----------|------|
| Baseline | 4B | bf16 (16-bit) | ~7.8GB | 高精度小模型 |
| Variant 1 | 12B | 4-bit (k-means) | ~7.8GB | 中等精度中模型 |
| Variant 2 | 31B | 1-bit (k-means, avg 1.25-bit) | ~7.8GB | 极低精度大模型 |

> 所有变体保持**总权重内存大致相等**，实现公平比较。

### 评估指标
- **预训练阶段**：pretraining loss（无偏估计 test loss）
- **下游任务**（SFT 后）：
  - **常识推理**：MMLU、HellaSwag、PIQA、ARC
  - **代码生成**：HumanEval、MBPP
  - **数学推理**：GSM8K (5-shot)
  - **指令跟随与创造力**：MMLU-PRO CoT、AidanBench（由 GPT-4-mini 打分）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| Benchmark | 4B/16-bit | 12B/4-bit | **31B/1-bit** |
|----------|------------|------------|----------------|
| **MMLU** | 33.21 | 50.86 | **51.61** ✅ |
| **ARC-C** | 41.21 | 49.15 | **50.68** ✅ |
| **MBPP** | 19.00 | 26.40 | **30.00** ✅ |
| **HumanEval** | 12.80 | 15.24 | **18.29** ✅ |
| **HumanEval-Instruct** | 26.22 | 35.98 | **42.68** ✅ |
| **AidanBench** | 88.08 | 147.48 | **167.83** ✅ |

> ✅ 表示 31B/1-bit 模型表现最佳；🟢 表示 12B/4-bit 最佳

- **例外**：GSM8K 上 12B/4-bit（48.52）略胜于 31B/1-bit（45.26）

### 与基线方法的对比结果
- **所有低比特模型均显著超越 4B bf16 基线**，说明“牺牲精度换取更大模型”策略有效。
- **31B/1-bit 模型整体最强**，尤其在知识、推理、编程任务上全面领先。
- **scaling law 预测准确**：理论预测最优精度为最低稳定比特（1-bit），实验验证成立。

### 消融实验结果
#### （1）量化格式对比（k-means vs uniform）
- 在相同 bit-width 下，**k-means 始终优于 uniform 整数量化**
- 差距在 **P ≤ 4.25 bits 时最明显**，表明非线性格式在低比特更具优势
- k-means 对 block-wise normalization（absmax vs absmear）不敏感，而 uniform 在 2-bit 以下必须使用 absmear 才能稳定

#### （2）不同 normalization 策略（Appendix A.6）
| Bit-width | 最优 normalization |
|-----------|--------------------|
| ≤2 bits | absmean（避免多数值归零） |
| ≥3 bits | absmax（保留极端值） |
| k-means | 几乎无影响 |

#### （3）scaling law 拟合质量（Figure 6）
- k-means: R² = 0.9617, RMSE = 0.0391
- uniform: R² = 0.9749, RMSE = 0.0305
- 两者均能很好拟合，支持模型有效性

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **在固定推理内存预算下，最优策略是使用尽可能低的比特宽度，并将节省的内存用于扩大模型规模**  
   - 即使是 **1-bit 权重**，配合足够大的 N（如 31B），也能在生成任务上取得 SOTA 性能。

2. ✅ **k-means 非线性量化显著优于传统的 uniform 整数量化**  
   - 特别是在 **< 4.25 bits 的低比特区域**，其重建误差更低，训练更稳定，最终性能更强。

3. ✅ **传统 perplexity-based 评估无法反映真实生成能力退化**  
   - 实验显示 12B/4-bit 在 pretraining loss 上优于 31B/1-bit，但在下游任务中后者反超，说明 loss 不是可靠指标。

4. ✅ **1-bit inference 可高效实现于标准硬件**  
   - 使用 **vector lookup table** 实现 fused dequantize-multiply kernel，在 L40S GPU 上达到近似理论速度上限：
     - 4-bit：最高 **3.7× speedup**
     - 1-bit：最高 **7.6× speedup**（micro-benchmark）

### 方法的局限性
- **仅适用于 memory-bound 场景**：当 batch size 较大时（如 prefill 阶段），计算成为瓶颈，量化优势消失。
- **依赖定制 kernel 支持**：目前尚无主流硬件原生支持 k-means lookup，需软件实现。
- **未涵盖 activation quantization**：本工作只对 weights 应用 QAT，activations 仍为 bf16。
- **特定架构依赖**：实验基于 Llama-style 模型，是否泛化到 MoE 或其他结构有待验证。

### 未来工作方向
- 探索 **activation-aware k-means quantization**
- 设计 **硬件友好的非线性量化格式**（如有限 lookup 表 + 近似计算）
- 研究 **动态 bit-width allocation**：根据不同层的重要性分配不同比特
- 将该范式推广至 **vision-language models** 或 **MoE 架构**

---

> 🔗 **开源信息**：作者已公开 [训练代码](https://github.com/Aleph-Alpha-Research/1-Bit-Wonder)、kernel 实现及 12B/31B 模型 checkpoint。

</details>

---

### 4. [Panini: Continual Learning in Token Space via Structured Memory](https://arxiv.org/abs/2602.15156)

**Authors**: Shreyas Rajesh, Pavan Holur, Mehmet Yigit Turali, Chenda Duan, Vwani Roychowdhury  
**Category**: cs.AI  
**Published**: 2026-02-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.15156v1  

#### Abstract
Language models are increasingly used to reason over content they were not trained on, such as new documents, evolving knowledge, and user-specific data. A common approach is retrieval-augmented generation (RAG), which stores verbatim documents externally (as chunks) and retrieves only a relevant su...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PANINI: Continual Learning in Token Space via Structured Memory

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前主流的 **Retrieval-Augmented Generation (RAG)** 方法在处理持续学习（continual learning）任务时存在以下核心缺陷：

- **低效的推理开销**：每次查询都需要重复检索并让 LLM 处理相同的文本片段（chunks），导致昂贵的测试时计算成本。
- **上下文污染**：基于 chunk 的检索容易引入无关或误导性上下文，增加幻觉（hallucination）和不支持答案的风险。
- **缺乏结构化记忆**：现有方法多存储原始文本或简单摘要，难以支持复杂的多跳推理（multi-hop reasoning）。

此外，**Parametric Continual Learning (PCL)** 虽能将新知识编码进模型参数，但面临灾难性遗忘、训练成本高、与指令对齐冲突等问题。

### 提出了什么新方法或新思路

本文提出 **PANINI**，一种非参数化的持续学习框架（Non-Parametric Continual Learning, NPCL），其核心思想是：

> **在写入阶段（write time）投入计算，构建结构化的语义记忆，在读取阶段（read time）实现高效、可靠的推理。**

具体创新包括：

1. **Generative Semantic Workspace (GSW)**  
   将每篇文档转化为一个实体-事件感知的 **question-answer (QA) 对网络**，形成结构化记忆。GSW 包含：
   - 实体节点（Entities）及其角色/状态
   - 动词短语/事件节点（Verb-phrase/Events）
   - 连接实体与事件的双向 QA 对

   该表示方式使得 LLM 可通过推理链重建经历的情境，并挖掘潜在知识。

2. **Reasoning Inference Chain Retrieval (RICR)**  
   一种基于束搜索（beam search）的检索机制，用于从 GSW 中提取最可能的推理链：
   - 先将复杂问题分解为原子子问题序列（question decomposition）
   - 逐跳（hop-by-hop）在 GSW 中追踪实体链接，形成推理链
   - 使用几何平均分对完整链进行评分，避免早期错误传播

3. **Dual Indexing**  
   构建双索引系统以提升检索效率：
   - **稀疏索引**：基于 BM25 的实体索引
   - **稠密索引**：基于嵌入的 QA 对索引

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **效率** | 推理时仅需极少量 tokens（2–30× 更少），显著降低 LLM 调用成本 |
| **准确性** | 在多跳 QA 上表现最优，平均 F1 达 **56.06%**，超越最强基线 HippoRAG2（53.3%） |
| **可靠性** | 在无证据支持的问题上表现出更强的拒绝能力（abstention），减少幻觉 |
| **开放性** | 支持完全开源的端到端流程，无需依赖闭源 API |

---

## 2. 核心实验方法和设置

### 使用的数据集

共使用 **6 个 QA 基准**，涵盖单跳与多跳推理：

| 类型 | 数据集 | 描述 |
|------|--------|------|
| **多跳 QA** | MuSiQue, 2WikiMultihopQA, HotpotQA, LV-Eval | 需要跨文档组合信息，测试多步推理能力 |
| **单跳 QA** | NQ, PopQA | 测试直接事实检索能力 |

所有实验采用与 HippoRAG2 相同的划分，确保公平比较。

### 实验设置和评估指标

#### 主要评估维度（NPCL 三大准则）

| 准则 | 指标 | 说明 |
|------|------|------|
| **支持性能** | F1, EM | 衡量在有证据支持问题上的回答准确率 |
| **推理效率** | 平均 token 数（↓） | 回答一个问题所使用的上下文 token 总数 |
| **可靠拒答** | `Ans` 和 `Unans` | `Ans`: 可回答问题的 F1；`Unans`: 不可回答问题的拒绝准确率（输出 N/A） |

#### 特别构造的“Platinum”评测集

- 人工标注 MuSiQue 和 2Wiki 中的不可回答样本（因证据缺失、歧义等）
- 用于测试模型在无支持证据下的 **可靠拒答能力**

### 基线方法对比

| 类型 | 基线方法 |
|------|----------|
| **Chunk-based Retrieval** | BM25, BM25+reranker, NV-Embed-v2, Qwen3-Embedding (+reranker) |
| **Structure-Augmented RAG** | RAPTOR, GraphRAG, LightRAG, HippoRAG, HippoRAG2 |
| **Agentic Systems** | IR-CoT, Search-R1 |

所有方法统一使用 **GPT-4o-mini** 作为答案生成模型，确保公平。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 性能对比（Table 2）

| 方法 | NQ | PopQA | MuSiQue | 2Wiki | HotpotQA | LV-Eval | **Avg F1** |
|------|-----|--------|---------|--------|-----------|----------|------------|
| Dense Retrieval (Best) | 61.4 | 59.9 | 43.7 | 57.9 | 68.2 | 11.8 | **50.5** |
| HippoRAG2 | 60.0 | 55.7 | 49.3 | 69.7 | 71.1 | 14.0 | **53.3** |
| **PANINI** | **67.4** | **57.6** | **52.3** | **72.4** | **71.9** | **14.8** | **56.1** |

👉 **PANINI 在所有 6 个基准上均取得最高平均性能，领先第二名 2.8 个百分点。**

#### ✅ 效率对比（Table 3）

| 方法 | Avg Token Count |
|------|------------------|
| Chunk Retrieval | ~705 |
| GraphRAG | 8,121 |
| IR-CoT | 10,745 |
| Search-R1 | 2,457 |
| **PANINI** | **319.8** |

👉 **PANINI 使用的 token 数仅为其他方法的 2–30 倍更少**，尤其远低于结构化与代理类方法。

#### ✅ 可靠性对比（Table 4，Platinum Set）

| 方法 | MuSiQue Ans | MuSiQue Unans | 2Wiki Ans | 2Wiki Unans | **Avg Ans** | **Avg Unans** |
|------|-------------|---------------|-----------|--------------|-------------|----------------|
| HippoRAG2 | 63.6 | 50.3 | 81.5 | 66.7 | 72.5 | 58.5 |
| **PANINI** | **75.0** | **72.6** | **84.8** | **73.1** | **79.9** | **72.8** |

👉 **PANINI 在保持更高回答准确率的同时，显著提升了拒答准确率**，打破了“强检索器易幻觉”的权衡。

### 消融实验结果（Ablation Studies）

#### 🔍 RICR 设计消融（Table 12）

| 设置 | MuSiQue F1 | 2Wiki F1 |
|------|------------|----------|
| Full PANINI | 52.3 | 72.4 |
| No decomposition | 36.8 | 47.2 |
| No dual search (entity-only) | 39.5 | 68.6 |
| No QA reranking | 18.4 | 22.2 |
| Beam width = 1 | 44.6 | 67.3 |

- **问题分解** 是关键，移除后性能大幅下降
- **双索引** 明显优于单一索引
- **QA 重排序** 对多跳任务至关重要
- 即使 **beam width=1** 仍具竞争力，表明 RICR 鲁棒性强

#### 📉 束宽（Beam Width）与效率权衡（Table 13）

| Beam Width | MuSiQue F1 | Tokens | 2Wiki F1 | Tokens |
|------------|------------|--------|----------|--------|
| 5 | 52.3 | 192 | 72.4 | 315 |
| 3 | 52.3 | 143 | 72.2 | 231 |
| 1 | 44.6 | 82 | 67.3 | 171 |

👉 **beam width=3** 可在几乎不损失精度的前提下节省约 25% 的 token 开销。

#### 🔄 写入时结构的通用性验证

将 PANINI 的 GSW 替换 Search-R1 的 chunk 输入（不重新训练），其 F1 从 **47.3 → 49.4**，证明 GSW 是一种**通用的结构化检索基础设施**。

---

## 4. 关键结论和发现

### 主要发现

1. **写入时结构化优于读取时智能**  
   PANINI 的成功表明：与其在推理时反复调用 LLM 进行“计划-检索-反思”，不如在写入时一次性构建高质量的结构化记忆，从而在读取时实现轻量、高效的推理。

2. **结构化记忆提升效率与可靠性**  
   GSW + RICR 的设计不仅减少了推理 token 消耗，还通过显式的推理链追踪降低了幻觉风险，在可回答与不可回答问题上均表现更优。

3. **GSW 具备通用价值**  
   GSW 不仅服务于 PANINI 自身，还可作为通用检索层增强其他系统（如 Search-R1），表明其是一种可复用的记忆抽象。

4. **开放管道可行性高**  
   即使使用全开源组件（Qwen3-8B/14B/GPT-OSS-120B），PANINI 依然保持性能优势，且对提取噪声具有鲁棒性。

### 方法的局限性

1. **未实现跨文档实体链接缓存**  
   当前 GSW 仅在文档内合并实体，跨文档关系需在读取时动态发现，未来可引入缓存机制优化。

2. **写入成本较高**  
   使用 GPT-4.1-mini 构建 GSW 的一次性成本约为 $48（MuSiQue），虽为一次投入，但仍高于纯嵌入索引。

3. **小模型构建 GSW 质量不稳定**  
   开源模型（如 Qwen3-8B）在 GSW 构建中可能出现遗漏动词短语或 QA 对不完整的问题。

4. **依赖高质量问题分解**  
   复杂或模糊问题可能导致分解错误，进而引发整个推理链偏移。

### 未来工作方向

1. **引入经验驱动的实体链接缓存**  
   对频繁出现的跨文档关系进行预合并，进一步加速检索。

2. **降低写入成本与提升鲁棒性**  
   探索更高效的 GSW 构建策略，如两阶段修复（two-pass refinement），或利用更小模型完成高质量提取。

3. **探索更丰富的结构化策略**  
   引入 agent 导航 GSW 进行主动 reconciliation，增强底层结构质量。

4. **扩展至叙事密集与多模态场景**  
   将 PANINI 应用于长文本、视频流等场景，利用时空结构与跨模态事件构建更复杂的记忆网络。

5. **推动 GSW 成为通用检索层**  
   探索 GSW 在多种下游框架中的即插即用能力，打造标准化的结构化记忆接口。

---

> **一句话总结**：  
> PANINI 通过在写入时构建 **Generative Semantic Workspace (GSW)** 结构化记忆，并在读取时执行 **Reasoning Inference Chain Retrieval (RICR)**，实现了高效、准确、可靠的非参数化持续学习，在性能、效率与可靠性上全面超越现有方法。

</details>

---

### 5. [Size Transferability of Graph Transformers with Convolutional Positional Encodings](https://arxiv.org/abs/2602.15239)

**Authors**: Javier Porras-Valenzuela, Zhiyang Wang, Alejandro Ribeiro  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.15239v1  

#### Abstract
Transformers have achieved remarkable success across domains, motivating the rise of Graph Transformers (GTs) as attention-based architectures for graph-structured data. A key design choice in GTs is the use of Graph Neural Network (GNN)-based positional encodings to incorporate structural informati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Size Transferability of Graph Transformers with Convolutional Positional Encodings*

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在解决 **Graph Transformers (GTs)** 在大规模图上训练成本高昂且难以扩展的问题。具体而言，标准 GTs 因其全连接的 **self-attention** 机制具有 $O(N^2)$ 的计算复杂度，导致在百万级节点的大图上训练变得不切实际。此外，直接在大图上收集数据和训练模型的成本极高。

核心问题是：**能否在小图上训练一个 GT 模型，并将其成功迁移到更大的图上，同时保证性能？**

### 提出的新方法与新思路
作者提出了一种基于理论分析的解决方案，其核心思想是 **“迁移性继承” (transferability inheritance)**：

1.  **理论框架**：将 GTs 放置在 **manifold limit model** 的理论框架下进行分析。该框架认为，从同一底层流形（manifold）采样得到的不同大小的图序列，其上的图算子（如图拉普拉斯矩阵 $L$）会收敛到该流形上的连续算子（如 Laplace-Beltrami 算子 $\mathcal{L}$）。
2.  **关键洞察**：如果 GT 的 **Positional Encodings (PEs)** 是可迁移的（即其输出能随图规模增大而收敛到流形上的连续函数），那么整个 GT 模型也能继承这种可迁移性。
3.  **推荐方案**：明确指出使用 **GNN-based PEs**（特别是 **RPEARL**）是一个原则性的选择。因为 GNNs 已被证明在 manifold limit 下是可迁移的，因此由 GNN 生成的 PEs 自然具备此性质。
4.  **实用架构**：提出并验证了一个高效的 **Sparse GT** 架构，它结合了 **k-hop neighborhood masking** 和 **RPEARL PEs**，在降低计算复杂度的同时保持了可迁移性。

### 相比现有方法的优势
- **理论保障**：首次为 GTs 的跨尺度迁移性提供了严格的数学证明，建立了 GTs 与 **Manifold Neural Networks (MNNs)** 之间的理论联系。
- **高效训练**：允许在小图上训练模型，然后直接部署到大图，避免了在大图上的昂贵再训练过程。
- **可扩展性强**：提出的 Sparse GT 架构进一步降低了计算负担，使其适用于超大规模图。
- **性能优越**：实验证明，使用 RPEARL PEs 的 GT 不仅可迁移，而且性能优于或媲美 GNNs 和其他 GT 变体。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在多个标准图基准数据集上进行，涵盖了不同规模和特性：
- **SNAP-Patents** (1.71M 节点): 异配性（heterophilic）专利引用网络。
- **ArXiv-year** (76.5K 节点): 异配性 arXiv 论文引用网络。
- **OGBN-MAG** (855K 节点): 大型异构学术网络。
- **Reddit-Binary** (~430 节点/图): 社交网络图分类任务。
- **Norway terrain graph**: 真实世界地形数据集，用于最短路径距离（SPD）估计任务。

### 实验设置和评估指标
- **迁移性评估**：通过控制训练图的大小来验证迁移能力。
  - 将训练集按比例 $\alpha \in \{0.05, 0.1, ..., 1.0\}$ 进行下采样，得到不同大小的训练图。
  - 在完整大小的测试图上评估所有模型的性能。
  - **评估指标**：
    - 节点分类任务：**Test Accuracy**。
    - 地形 SPD 任务：**Test MAE** (Mean Absolute Error)。
- **消融研究**：在 SNAP-Patents 数据集上，固定训练图为 30% 大小，系统地移除或替换模型组件以分析其影响。

### 基线方法对比
- **Dense GT + RPEARL**: 使用全注意力和 RPEARL PEs 的 GT。
- **Sparse GT + RPEARL**: 使用 k-hop 邻域掩码和 RPEARL PEs 的 GT。
- **GNN (TAGConv)**: 传统的图神经网络作为基线。
- **Exphormer**: 一种稀疏 GT，使用局部邻域+expander 图进行注意力。
- **MLP**: 多层感知机，无图结构信息。
- **GT w/o PE**: 无位置编码的 GT。
- **GT + Mask**: 仅有掩码的 GT。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **迁移性表现**：
    - **Figure 2 & 5** 显示，无论是 Dense GT 还是 Sparse GT，在使用 RPEARL PEs 时，其在小图上训练后在大图上的测试准确率都随着训练图尺寸 $\alpha$ 的增加而稳定提升，并最终接近在全图上训练的性能。
    - 例如，在 **SNAP-Patents** 上，**Sparse GT + RPEARL** 仅用 10.5% 的训练数据就达到了峰值性能的近 90%。
    - 在 **OGBN-MAG** 和 **Reddit-Binary** 上，GT 的性能与 GNN 相当；而在异配性更强的 **SNAP-Patents** 和 **ArXiv-year** 上，GT 显著优于 GNN，这归功于其全局注意力机制。

2.  **与基线方法的对比**：
    - **GTs vs. GNNs**: 在异配图（SNAP, ArXiv）上，GT 性能显著更好；在同配图（MAG, Reddit）上，两者性能相当。
    - **Sparse GT vs. Dense GT**: Sparse GT 的训练速度比 Dense GT 快 **1-2个数量级**（见 Table 4），同时在 **ArXiv-year** 和 **SNAP-Patents** 上性能更优或相当。
    - **RPEARL PEs 的重要性**：所有使用 RPEARL PEs 的模型均表现出优异的迁移性和高精度。

### 消融实验结果
在 **SNAP-Patents** 数据集上的消融实验（Table 1）结果如下：

| Architecture | Accuracy (%) | % vs. GT (no PE) |
| :--- | :--- | :--- |
| GT (no PE) | 31.33 | — |
| GT + RPEARL | 34.63 | +10.53% |
| GT + Mask | 39.69 | +26.68% |
| **GT + Mask + RPEARL** | **49.70** | **+58.63%** |
| GT + Mask + RE | 31.01 | -1.02% |

- **RPEARL PEs** 单独带来 +10.53% 的提升，证明了其有效性。
- **k-hop Masking** 单独带来 +26.68% 的提升，表明其作为一种归纳偏置非常有用。
- **RPEARL + Masking** 结合达到最佳性能 **49.70%**，增益高达 +58.63%，显示出协同效应。
- **Random Edges (RE)** 对性能没有帮助，甚至有害。

---

## 4. 关键结论和发现

### 主要发现
1.  **可迁移性继承**：Graph Transformers 的跨尺度迁移性可以由其 **Positional Encodings** 继承而来。只要 PEs 是可迁移的（如 GNN-based RPEARL），整个 GT 模型就是可迁移的。
2.  **GNN PEs 的优越性**：使用 GNN 作为 PEs 不仅是有效的，而且是**原则性的选择**，因为它自然地满足了可迁移性、置换等变性（permutation equivariance）和稳定性要求。
3.  **高效且强大的架构**：结合 **RPEARL PEs** 和 **k-hop masking** 的 **Sparse GT** 是一个高效、可扩展且高性能的架构，实现了计算效率和模型表达力的良好平衡。
4.  **实践意义**：可以在小图上快速训练 GT 模型，然后直接应用于大图，实现巨大的**计算节省**，这对于处理现实世界中的超大规模图至关重要。

### 方法的局限性
- **理论假设**：理论分析依赖于图是从同一底层流形采样的这一强假设，在现实中可能不完全成立。
- **PE 依赖性**：模型的成功高度依赖于 PEs 的质量。如果 PEs 本身不可迁移，则整个框架失效。
- **稀疏化限制**：k-hop masking 虽然高效，但也限制了注意力的范围，可能会丢失一些长距离依赖信息（尽管全局注意力头可以部分弥补）。

### 未来工作方向
- 将该框架推广到其他图极限对象，如 **graphons**。
- 探索和证明其他类型的 **structural, relative, or absolute PEs** 是否也满足可迁移性条件。
- 研究如何设计更灵活、自适应的注意力掩码机制，以更好地平衡效率和性能。
- 将该方法应用于更多真实世界的超大规模图学习任务。

</details>

---

### 6. [PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](https://arxiv.org/abs/2602.15669)

**Authors**: Xiachong Feng, Liang Zhao, Weihong Zhong, Yichong Huang, Yuxuan Gu, Lingpeng Kong, Xiaocheng Feng, Bing Qin  
**Category**: cs.AI  
**Published**: 2026-02-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.15669v1  

#### Abstract
Current methods for personality control in Large Language Models rely on static prompting or expensive fine-tuning, failing to capture the dynamic and compositional nature of human traits. We introduce PERSONA, a training-free framework that achieves fine-tuning level performance through direct mani...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大语言模型（LLMs）中的 **personality control** 主要依赖于静态提示（prompting）或昂贵的微调（fine-tuning），存在以下问题：
- **不稳定性**：prompting 方法在不同输入下表现不一致；
- **低效性**：fine-tuning 需要为每种人格配置单独训练，计算成本高；
- **缺乏动态性和组合性**：无法实现人格特质的实时、上下文感知调整，也无法灵活组合多个特质。

该论文旨在解决如何在 **无需训练** 的前提下，实现 **动态、可组合、推理时（inference-time）的人格控制**。

---

### 🚀 提出的新方法与核心思想
作者提出 **PERSONA** 框架，一种无需训练（training-free）的人格控制框架，其核心思想是：
> **将人格特质表示为激活空间中可提取的、近似正交的方向向量，并支持代数运算（加减乘）进行精确控制。**

#### 框架三大组件：
1. **PERSONA-BASE**  
   - 通过 **contrastive activation analysis** 提取 OCEAN 模型中 10 个基本人格维度的 **正交向量**（如 `v_inventive`, `v_dependable` 等）。
   - 这些向量存在于模型的 residual stream 中，可通过残差添加（residual addition）直接操控。

2. **PERSONA-ALGEBRA**  
   - 支持 **向量代数操作**：
     - **标量乘法**：调节特质强度（如 `2 * v_outgoing` 表示更强外向）；
     - **向量加法**：组合多种特质（如 `v_outgoing + v_compassionate`）；
     - **向量减法**：抑制特定特质（如 `v_outgoing - v_solitary` 可增强外向性）。

3. **PERSONA-FLOW**  
   - 在推理过程中 **动态组合** 向量，基于对话上下文预测最优人格调整系数（delta），实现 **context-aware 的人格自适应**。
   - 采用 “predict-then-steer” 机制：先分析上下文生成调整系数，再注入复合向量。

---

### ⭐ 相比现有方法的优势
| 维度 | 传统方法（Prompting / Fine-tuning） | PERSONA |
|------|-------------------------------|---------|
| **效率** | 微调需大量计算资源；prompting 不稳定 | 完全无需训练，仅需一次向量提取 |
| **灵活性** | 静态控制，难以组合或动态调整 | 支持动态、组合、细粒度控制 |
| **性能** | 微调效果好但成本高；prompting 效果有限 | 性能接近监督微调（SFT）上限 |
| **可解释性** | 黑箱操作 | 向量具有语义意义，操作透明 |

---

## 2. 核心实验方法和设置

### 📚 数据集
1. **PersonalityBench**（外部基准）
   - 包含约 90 个情境问题，用于评估五大性格维度（OCEAN）；
   - 作为标准 benchmark，用于与已有方法公平比较。

2. **PERSONA-EVOLVE**（本文提出的新 benchmark）
   - 包含 **800 个多轮对话场景**，涵盖 100 个虚构角色（如“食品车老板”、“公共辩护律师”等）；
   - 每个角色有稳定背景和动态情绪轨迹（如从“压力大”到“释然”）；
   - 用于评估 **动态人格适应能力**。

---

### 📊 评估指标
- **Trait Adherence (TA)**：响应是否符合目标人格特质；
- **Role Consistency (RC)**：是否保持角色一致性；
- **Response Authenticity (RA)**：语气、风格是否自然真实；
- **Information Fidelity (IF)**：信息深度与相关性；
- **Win Rate (%)**：成对比较中，PERSONA-FLOW 超越基线的比例；
- **MMLU / TruthfulQA**：评估人格控制对通用能力的影响。

---

### 🔁 基线方法对比
| 基线方法 | 类型 | 描述 |
|--------|------|------|
| **Simple Prompt** | Prompting | 单形容词引导（如“你是一个外向的人”） |
| **P2 Induction** | Prompting | 使用模型生成的性格描述 |
| **ActAdd** | Activation Manipulation | 修改 residual stream |
| **NPTI** | Neuron-level | 基于神经元的个性诱导 |
| **PAS** | Attention Probe | 探测注意力头 |
| **SFT (Supervised Fine-Tuning)** | 微调 | LoRA 微调，作为性能上界 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### 在 **PersonalityBench** 上的表现（LLaMA-3-8B-Instruct）：
| 方法 | 平均得分（Mean） | 方差（Variance） |
|------|------------------|----------------|
| **PERSONA-BASE** | **9.60** | 0.74 |
| SFT（上界） | 9.61 | 0.49 |
| NPTI | 9.43 | 0.49 |
| ActAdd | 8.20 | 2.10 |

> ✅ **PERSONA-BASE 几乎达到 SFT 的性能上限（9.60 vs 9.61），且显著优于其他无训练方法**。

---

#### 在 **PERSONA-EVOLVE** 上的胜率（Win Rate %）：
| 模型家族 | TA | RC | RA | IF | **Overall** |
|----------|----|----|----|----|------------|
| Qwen3-4B | 92.2 | 90.6 | 92.4 | 49.1 | **90.8%** |
| Qwen2.5-14B | 84.8 | 86.4 | 84.8 | 59.3 | 85.4% |
| Llama-3.1-8B | 84.9 | 81.4 | 85.6 | 57.2 | 83.5% |
| Mistral-8B | 74.3 | 73.2 | 74.2 | 48.0 | 73.2% |

> ✅ **最高达 91% 的胜率**，表明 PERSONA-FLOW 在多轮对话中能有效维持人格一致性并动态适应。

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）向量正交性验证
- 对 `v_outgoing` 和 `v_compassionate` 进行加法操作后，BFI-44 测评显示：
  - 外向性和宜人性得分同时上升；
  - 验证了 **向量加法可实现特质组合**。

#### （2）非正交相关性的鲁棒性
- 尽管某些向量间存在相关性（如 `nervous` 与 `careless` 正相关），但：
  - 控制 `v_nervous` 时，`careless` 分数仅有小幅可预测上升（次级效应 <20% 主效应）；
  - 加法合成仍保持线性叠加，说明系统 **具备可预测的组合性**。

#### （3）PERSONA-FLOW 设计选择消融（Qwen2.5-7B）
| 配置 | Overall Win Rate | Δ vs 默认 |
|------|------------------|---------|
| **默认（连续系数）** | **83.4%** | — |
| 三值离散（{-1,0,1}） | 75.2% | -8.2 |
| 使用全部历史对话 | 80.9% | -2.5 |
| 阈值 T=0.7（过高稀疏） | 81.6% | -1.8 |

> ✅ 结论：**连续系数、仅用当前回合上下文、T=0.5 的稀疏门控** 是最优设计。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **人格特质在 LLM 激活空间中呈近似正交方向**，可通过 contrastive analysis 提取。
2. **这些向量支持代数运算**，实现细粒度、可组合的人格控制。
3. **无需任何梯度更新**，PERSONA 在 PersonalityBench 上性能逼近 SFT 上限（9.60 vs 9.61）。
4. **PERSONA-FLOW 实现了上下文感知的动态人格调节**，在 PERSONA-EVOLVE 上胜率达 91%。
5. **该方法不影响通用能力**：在 MMLU 和 TruthfulQA 上性能持平甚至略有提升。

---

### ⚠️ 局限性
1. **安全对齐的冲突**：
   - 某些反社会特质（如 `self-interested`）受 alignment 限制，难以激活；
   - 但某些“看似无害”的特质（如 `inventive`, `careless`）可能降低安全性（ASR ↑）。
2. **向量非完全正交**：
   - 存在语义关联导致的相关性（如 `calm` 与 `dependable` 正相关），需注意副作用。
3. **依赖高质量向量提取**：
   - 向量质量受生成器 LLM 影响，虽实验证明对 Qwen1B 也有效，但仍有一定依赖。

---

### 🔮 未来工作方向
1. **安全感知的人格控制**：开发能自动规避风险组合的 steering 策略。
2. **跨层或多层 steering**：探索更复杂的激活路径以提升控制精度。
3. **个性化人格建模**：从用户交互中学习个体化人格向量。
4. **扩展至情感与认知状态**：将情绪波动、认知负荷等纳入动态控制体系。

---

## 总结
> **PERSONA 首次证明了 LLM 中的人格具有“数学可处理性”（mathematically tractable）**。  
> 通过将人格视为激活空间中的向量，并支持加减乘等代数操作，实现了高效、动态、可解释的行为控制，为构建真正“人性化”的 AI 助手提供了新范式。

</details>

---

### 7. [Distributed Semi-Speculative Parallel Anisotropic Mesh Adaptation](https://arxiv.org/abs/2602.15204)

**Authors**: Kevin Garner, Polykarpos Thomadakis, Nikos Chrisochoides  
**Category**: cs.DC  
**Published**: 2026-02-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.15204v1  

#### Abstract
This paper presents a distributed memory method for anisotropic mesh adaptation that is designed to avoid the use of collective communication and global synchronization techniques. In the presented method, meshing functionality is separated from performance aspects by utilizing a separate entity for...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Distributed Semi-Speculative Parallel Anisotropic Mesh Adaptation》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在解决**大规模各向异性网格自适应生成**在高性能计算（HPC）环境下的可扩展性和性能瓶颈问题。传统并行网格生成方法常依赖**集体通信（collective communication）** 和**全局同步（global synchronization）**，这些操作在大规模核心配置下会显著降低性能，成为制约超算应用效率的关键因素。

此外，直接将已有的共享内存（shared memory）网格生成器作为“黑盒”集成到分布式内存框架中往往效果不佳，因为这类代码未针对可扩展性设计，导致性能低下或无法复现结果。

### 提出的新方法与思路
提出了一种名为 **A Priori DM_CDT3D** 的分布式内存各向异性网格自适应方法，其核心创新包括：

- **分离功能与性能**：采用 **Telescopic Approach** 架构，将网格生成功能（由 `CDT3D` 负责）与并行运行时支持（由 `PREMA` 提供）解耦。这种设计提高了代码的模块化、可维护性和对新型 HPC 架构的适应能力。
  
- **先验接口适应（A Priori Interface Adaptation）**：
  - 首先在一个多核节点上对初始粗网格进行分解，并**预先适应所有子域边界（interface elements）**。
  - 然后将子域分发到集群的不同节点，在适应内部元素时**冻结已处理的接口元素**，从而避免了后续复杂的跨子域数据依赖协调。

- **半推测式执行模型（Semi-Speculative Execution Model）**：
  - 利用 `CDT3D` 内部的推测式执行机制，通过**预锁定（preemptive locking）** 和引入 **伪激活/伪非激活（pseudo-active/inactive）** 元素分类，精确控制每个阶段（接口/内部）只处理特定元素。
  - 这确保了在不破坏网格一致性的前提下实现高效并行。

### 相比现有方法的优势
- **避免集体通信与全局同步**：相比如 `refine` 等使用 `MPI_Allreduce` 等操作的方法，本方法通信开销更低，更适合大规模并行。
- **更高的端到端性能**：在生成约10亿个元素的网格时，该方法在512核上耗时不到4小时，远优于 `refine` 和原始共享内存 `CDT3D`。
- **良好的弱可重现性（weak reproducibility）**：不同核心数下生成的网格质量稳定且相近。
- **有效利用共享与分布式内存并发性**：结合了细粒度（芯片级）和粗粒度（节点级）并行，最大化硬件利用率。

---

## 2. 核心实验方法和设置

### 数据集与几何模型
实验基于两个标准测试案例：
- **Delta Wing Geometry**：平面面片构成的三角翼，使用基于亚音速层流马赫场构造的多尺度 metric field。
- **Cube Geometry**：单位立方体，使用解析定义的 **polar-2 metric field**（代表弯曲剪切层），来自 UGAWG 基准。

### 实验设置
- **硬件平台**：
  - **Wahab 集群**（ODU）：最多使用 512 核（16 节点 × 32 核/节点）。
  - **Anvil 集群**（Purdue）：最多使用 1536 核（16 节点 × 96 核/节点）。
- **软件编译**：使用 GNU GCC 11.4.1 和 Intel MPI 编译器。
- **参数配置**：
  - 使用 **PQR 分解法** 进行数据划分。
  - 接口适应阶段通常解锁 3–5 层元素以保证质量。
  - 对比了不同子域数量（如 16 或 45）的影响。

### 评估指标
- **性能指标**：
  - **运行时间（Runtime）**：从分钟到小时级别衡量。
  - **加速比（Speedup）**：相对于单核运行的时间比率。
  - **通信与非网格生成开销占比**。
- **质量指标**：
  - **平均比值（Mean Ratio）直方图**：衡量单元形状质量（理想为1）。
  - **边长分布（Edge Length）直方图**：衡量是否符合目标 metric（理想为1）。
  - **网格大小（#Tetrahedra, #Vertices）**：最终生成的单元和顶点数。

### 基线方法对比
- **SM_CDT3D**：原始的共享内存版本，仅限单节点运行。
- **A Posteriori DM_CDT3D**：早期分布式版本，采用后验接口适应，需多次迁移数据。
- **refine**：当前最先进的开源并行各向异性网格生成器，广泛用于 CFD 流程。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 几何 | 方法 | 规模 | 最佳运行时间 | 所用核心 |
|------|------|------|-------------|---------|
| Cube | A Priori DM_CDT3D | ~1 Billion 元素 | **< 4 小时** | 512 |
| Cube | refine | ~1 Billion 元素 | ~6 小时 | 512 |
| Cube | SM_CDT3D | ~1 Billion 元素 | ~8 小时 | 32 |

> 在相同条件下（512核），DM_CDT3D 比 `refine` 快约 **1.7x**，比 SM_CDT3D 快 **2x 以上**。

### 与基线方法的对比结果
#### ✅ 性能优势
- 在 **Delta Wing @ 10M 复杂度** 上：
  - 使用 256 核时，`A Priori DM_CDT3D` 耗时 **23 分钟**，而 `refine` 耗时 **31 分钟**。
  - 在 Anvil 上使用 384 核时，`DM_CDT3D` 耗时 **7 分钟**，`refine` 耗时 **15 分钟**。
- 在 **Cube @ 100M 复杂度** 上：
  - `DM_CDT3D` 在 512 核上耗时 **3.46 小时**，`refine` 耗时 **5.98 小时**，性能提升显著。

#### ✅ 质量表现
- 生成的网格质量与 `SM_CDT3D` 和 `refine` 相当：
  - 平均比值分布集中于 0.8–1.0 区间。
  - 边长集中在 1.0 附近，满足 metric conformity。
- `refine` 生成的网格略优（更少低质量单元），但代价是生成更多元素（约多 10–15%）。

#### ❌ 局限性体现
- 当核心数超过 384 时，`DM_CDT3D` 的加速趋于饱和，因**接口适应阶段受限于单节点核心数**（如 Wahab 上仅 32 核可用）。
- `refine` 在高核数（如 768/1536）下出现崩溃或挂起，而 `DM_CDT3D` 表现更稳健。

### 消融实验结果
通过对比 **“Modified”**（含优化）与 **“Naive”**（黑盒式）实验，验证了关键设计的有效性：

| 阶段 | 实验类型 | 接口适应时间 | 子域1内部适应时间 | 加速比 |
|------|--------|--------------|-------------------|-------|
| —— | Naive | 321.09 秒 | 1949.23 秒 | 1× |
| —— | Modified | **75.4 秒** | **1220.99 秒** | **1.59x** |

- **改进来源**：
  1. **跳过接口阶段的后置边坍缩（post-refinement edge collapse）**：减少冗余操作。
  2. **伪激活标记 + 预锁定机制**：防止重复处理已适应区域。
  3. **缓冲区设置（buffer zone）**：避免邻近伪非激活元素处的点插入失败导致收敛困难。

这些优化使接口适应阶段提速 **4.25x**，内部适应提速 **~1.6x**。

---

## 4. 关键结论和发现

### 主要发现
1. **功能与性能解耦是构建可持续 HPC 应用的关键**：通过 `PREMA` + `CDT3D` 的组合，实现了高性能与高可维护性的统一。
2. **避免集体通信可显著提升大规模并行性能**：本方法通过 a priori 接口适应策略，完全规避了 `MPI_Allreduce` 类操作，通信开销控制在 15–25%，远低于典型方法的 >50%。
3. **“黑盒集成”不可行**：直接将共享内存代码用于分布式环境会导致性能下降和逻辑错误；必须进行针对性重构（如伪激活机制）才能发挥潜力。
4. **半推测式执行模型是实现高效局部控制的基础**：通过点锁定和任务调度机制，实现了对特定元素集合的精准适应。

### 方法的局限性
- **接口适应阶段为单节点瓶颈**：目前只能在一个节点上执行，限制了整体可扩展性。
- **子域连接性需后处理修复**：`MAKE_SIMPLY_CONNECTED` 步骤为串行，尚未并行化。
- **负载均衡未启用**：由于未采用过分解（overdecomposition），无法利用 `PREMA` 的动态负载均衡能力。
- **内存使用较高**：统计信息打包等过程占用额外内存，影响小规模运行效率。

### 未来工作方向
1. **并行化串行组件**：
   - 将 `MAKE_SIMPLY_CONNECTED` 和数据结构转换步骤并行化。
2. **实现完全推测式接口适应**：
   - 允许接口与内部元素同时适应，进一步提升并发度。
3. **开发机器学习驱动的参数调优模型**：
   - 自动推荐最优的分解方式、层数、迭代次数等参数，减少人工试错成本。
4. **集成至完整 CFD 流程验证**：
   - 将该方法嵌入真实 CFD 模拟管道，验证其在闭环系统中的实用性与稳定性。
5. **探索其他 tasking backend**：
   - 测试除 Pthread 外的其他执行后端（如 CUDA 支持）以挖掘更大性能潜力。

--- 

> **总结**：本文提出的 **Distributed Semi-Speculative Parallel Anisotropic Mesh Adaptation** 方法在性能、可扩展性和工程实践性方面均取得了重要进展，为下一代超算时代的自适应网格生成提供了可行路径。

</details>

---

### 8. [FlashMem: Supporting Modern DNN Workloads on Mobile with GPU Memory Hierarchy Optimizations](https://arxiv.org/abs/2602.15379)

**Authors**: Zhihao Shu, Md Musfiqur Rahman Sanim, Hangyu Zheng, Kunxiong Zhu, Miao Yin, Gagan Agrawal, Wei Niu  
**Category**: cs.DC  
**Published**: 2026-02-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.15379v1  

#### Abstract
The increasing size and complexity of modern deep neural networks (DNNs) pose significant challenges for on-device inference on mobile GPUs, with limited memory and computational resources. Existing DNN acceleration frameworks primarily deploy a weight preloading strategy, where all model parameters...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FlashMem: Supporting Modern DNN Workloads on Mobile with GPU Memory Hierarchy Optimizations**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代深度神经网络（DNN）模型规模日益增大，对移动设备上的**内存资源和计算效率**提出了严峻挑战。当前主流的 DNN 推理框架（如 MNN、TVM、LiteRT 等）普遍采用 **weight preloading** 策略，即在推理前将整个模型权重加载到 GPU 内存中。这种策略导致以下问题：

- **峰值内存占用高**：难以支持大规模模型或多模型并行执行。
- **初始化延迟大**：权重从磁盘加载并转换为 GPU 友好格式（如 2.5D texture layout）耗时严重。
- **资源利用率低**：CPU 和 GPU 之间缺乏有效流水，I/O 与计算无法重叠。

此外，移动 GPU 通常采用统一内存架构（UMA），依赖 **texture memory** 进行高效张量访问，但现有框架未充分优化这一层次结构。

---

### **提出的新方法与创新点**
作者提出 **FlashMem** —— 一种基于 **GPU memory hierarchy 优化** 的内存流式处理框架，核心思想是 **动态按需流式加载权重**，而非一次性预加载。

#### **主要创新点如下：**

1. **Optimized Overlap Plan Generation (OPG)**  
   - 将权重加载调度问题形式化为 **Constraint Programming Satisfiability (CP-SAT)** 问题。
   - 设计 **LC-OPG Solver**（Load Capacity-aware OPG Solver），静态生成最优的权重加载计划，在保证性能的同时最小化峰值内存使用。

2. **Load Capacity Profiling & Adaptive Fusion**  
   - 对不同算子进行分类（Elemental / Reusable / Hierarchical），分析其 **memory bandwidth 利用率** 和 **负载容忍度**。
   - 引入 **自适应融合机制**：当融合后算子负载容量不足时，自动拆分以恢复调度灵活性。

3. **Hierarchical GPU Memory Optimization**  
   - 针对移动 GPU 的 **2.5D texture memory** 特性重构权重布局，减少不必要的 Reshape/Transpose 操作。
   - 利用 texture cache 提升空间局部性，降低数据变换开销。

4. **Pipeline-Aware Kernel Execution**  
   - 改写 GPU kernel，实现 **branch-free pipelined 执行模式**，交错执行计算与权重加载。
   - 通过循环重组隐藏内存延迟，提升 SIMT 效率。

---

### **相比现有方法的优势**
| 维度 | 传统方法（如 MNN、TVM） | FlashMem |
|------|--------------------------|---------|
| **内存管理** | 全量预加载，高 peak memory | 动态流式加载，显著降低内存占用 |
| **执行效率** | 初始化阶段长，I/O 与计算串行 | 重叠加载与计算，大幅缩短端到端延迟 |
| **多模型支持** | 多模型切换成本高，易 OOM | 支持 FIFO 多模型流水执行 |
| **硬件适配** | 忽视 texture memory 层次结构 | 显式优化 2.5D layout，提升 GPU 利用率 |

---

## **2. 核心实验方法和设置**

### **使用的模型与任务**
在 **11 个代表性 DNN 模型** 上进行了评估，涵盖多种应用场景：

| 类别 | 模型示例 |
|------|--------|
| NLP | GPTNeo-S, GPTNeo-1.3B, GPTNeo-2.7B |
| 图像分类 | ResNet50, ViT, DeepViT |
| 图像生成 | StableDiffusion-UNet (SD-UNet) |
| 语音识别 | Whisper-Medium |
| 视频分割 | DepthAnything-S/L |
| 图像分割 | SAM-2 |

> 注：所有模型均使用 FP16 或 FP32 精度，不涉及量化模型。

---

### **实验设置**
- **目标设备**：
  - 主要平台：OnePlus 12（Adreno 750 GPU, 16GB RAM）
  - 移植性测试：Pixel 8（Mali-G715）、OnePlus 11、Xiaomi Mi 6
- **批大小**：Batch size = 1
- **运行次数**：每项实验重复 50 次取平均值
- **部署方式**：FlashMem 构建于 SmartMem 基础之上，复用其 layout transformation 能力

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **End-to-end Latency** | 初始化 + 推理总时间（FlashMem 将两者集成） |
| **Average Memory Usage** | 推理过程中的平均内存消耗 |
| **Peak Memory** | 最高峰值内存占用 |
| **Speedup** | 相对于基线方法的速度提升倍数 |
| **Memory Reduction** | 内存占用下降比例 |
| **Energy Consumption** | 功耗 × 时间，衡量能效 |

---

### **基线方法对比**
- **商业级框架**：
  - MNN
  - NCNN
  - LiteRT（原 TensorFlow Lite）
  - TVM
  - ExecuTorch (ETorch)
- **研究原型**：
  - SmartMem（FlashMem 的前身，已优化 layout transformation）

> 不包含 llama.cpp、FlexNN 等因不支持 mobile GPU 或特定模型而被排除。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### ✅ **端到端延迟对比（Table 7）**
- FlashMem 在 **所有可运行模型上均优于基线框架**。
- 平均速度提升：
  - 相比 SmartMem：**8.6×**
  - 相比其他框架（几何平均）：**1.7× ~ 75.0×**
- 典型加速案例：
  - **GPTNeo-1.3B**：比 SmartMem 快 **15.8×**
  - **SD-UNet**：快 **9.3×**
  - **DeepViT**：快 **10.0×**

> ⚠️ 注意：部分框架（如 NCNN）根本不支持 Transformer 类大型模型。

#### ✅ **内存消耗对比（Table 8）**
- 平均内存减少：
  - 相比 SmartMem：**3.5×**
  - 相比其他框架：**2.0× ~ 8.4×**
- 典型节省案例：
  - **Whisper-Medium**：内存从 1,433MB → **240MB**（**6.0× 减少**）
  - **DeepViT**：从 826MB → **165MB**（**5.0× 减少**）
  - **GPTNeo-1.3B**：从 2,667MB → **554MB**（**4.8× 减少**）

#### ✅ **多模型共执行能力（Figure 6）**
- 在 **Whisper → DepthA → ViT → SD-UNet** 四模型交替执行场景下：
  - MNN 出现多次内存峰值，接近系统上限。
  - FlashMem 通过精细调度实现平滑内存曲线，**峰值内存显著降低**，成功完成全部任务。

#### ✅ **消融实验分析（Figure 7）**
分解各组件贡献（以 SmartMem 为基准）：

| 组件 | 加速比范围 | 内存减少范围 |
|------|-----------|-------------|
| OPG-Solver | 5.3× ~ 8.1× | 2.1× ~ 3.8× |
| Adaptive Fusion | 1.5× ~ 5.1× | 1.1× ~ 1.4× |
| Kernel Rewriting | 1.0× ~ 2.55× | 1.0× ~ 1.1× |

> 表明 **OPG 是最大贡献者**，而 kernel rewriting 更侧重于延迟优化。

#### ✅ **功耗与能耗（Table 9）**
- 能耗大幅下降：
  - **DeepViT**：能耗从 SmartMem 的 41.0J → **4.5J**（↓89%）
  - **SD-UNet**：从 134.5J → **17.9J**（↓87%）
- 功耗略有上升（因更高 GPU 利用率），但由于延迟极短，总体能量更优。

#### ✅ **移植性验证（Figure 10）**
- 在 **Pixel 8、OnePlus 11、Mi 6** 上均表现良好。
- GPTNeo-1.3B 在 Pixel 8 和 Mi 6 上因内存不足无法由 SmartMem 启动，但 **FlashMem 成功运行**，体现更强的资源适应性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **全量预加载非必需**：通过细粒度调度，可在保持高性能前提下大幅削减内存需求。
2. **Overlap 是关键**：将权重加载、格式转换与计算流水化，能有效隐藏 I/O 延迟。
3. **operator 分类指导调度**：不同算子对并发加载容忍度差异显著（Softmax < Conv < Elementwise），必须差异化处理。
4. **fusion 需权衡**：过度融合会破坏调度机会，应引入 **adaptive fusion** 动态调整。
5. **kernel 层控制流消除至关重要**：branch divergence 严重影响 SIMT 性能，pipelined + branch-free kernel 显著提升吞吐。

---

### **方法的局限性**
1. **离线求解器开销**：LC-OPG Solver 需要在部署前运行，虽然只一次，但对于超大模型（如 Llama2-70B）可能耗时较长（表 4 中达 136s）。
2. **静态图假设**：目前仅支持静态 DNN 结构，动态路径（如条件分支）尚未支持。
3. **依赖 SmartMem 基础设施**：未完全独立实现 layout transformation elimination。
4. **未探索量化支持**：当前评估限于浮点模型，未整合 INT8/FP16 量化技术。

---

### **未来工作方向**
1. **扩展至数据中心场景**：将该方法应用于服务器端大模型服务，结合 vLLM 的 paging 思想。
2. **支持动态神经网络**：开发在线调度器应对 runtime-dependent execution path。
3. **跨设备协同推理**：结合边缘协作训练/推理框架（如 Asteroid、Chimera）进行分布式 memory streaming。
4. **整合量化与稀疏性**：进一步压缩传输数据量，提升带宽效率。
5. **自动化参数调优**：设计 Auto-tuner 自动选择 `Mpeak`, `λ`, `μ` 等超参以平衡 memory-latency-tradeoff。

---

> 🔚 **总结一句话**：  
> **FlashMem 通过“静态调度 + 动态流式加载 + GPU 层次内存优化”，首次实现了在资源受限移动设备上高效运行数十亿参数级 DNN 模型与多模型流水线，突破了传统预加载范式的瓶颈。**

</details>

---

### 9. [COMPOT: Calibration-Optimized Matrix Procrustes Orthogonalization for Transformers Compression](https://arxiv.org/abs/2602.15200)

**Authors**: Denis Makhov, Dmitriy Shopkhoev, Magauiya Zhussip, Ammar Ali, Baher Mohammad, Stamatios Lefkimmiatis  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.15200v1  

#### Abstract
Post-training compression of Transformer models commonly relies on truncated singular value decomposition (SVD). However, enforcing a single shared subspace can degrade accuracy even at moderate compression. Sparse dictionary learning provides a more flexible union-of-subspaces representation, but e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：COMPOT: Calibration-Optimized Matrix Procrustes Orthogonalization for Transformers Compression

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Transformer 模型在部署时面临巨大的内存占用、带宽和计算开销。尽管低秩矩阵分解（如 SVD）是常用的**post-training compression**方法，但其强制所有权重列共享一个单一子空间（shared subspace），这在不同列位于不同局部子空间时会限制表达能力，导致压缩后精度显著下降。

此外，稀疏字典学习（sparse dictionary learning）虽然提供了更灵活的“子空间联合”（union-of-subspaces）表示，但现有方法依赖迭代优化（如 K-SVD/OMP），计算成本高昂，难以扩展到十亿参数规模。

### 提出了什么新方法或新思路
本文提出 **COMPOT**（Calibration-Optimized Matrix Procrustes Orthogonalization for Transformers），一种无需训练的 Transformer 压缩框架，核心思想如下：

- **正交字典因子化（Orthogonal Dictionary Factorization）**  
  将权重矩阵 $ W $ 分解为正交字典 $ D_o \in \mathbb{R}^{m\times k} $ 和稀疏系数 $ S_o \in \mathbb{R}^{k\times n} $，即 $ W \approx D_o S_o $，其中 $ D_o^\top D_o = I_k $，$ k \leq m $（完整或欠完备字典）。

- **闭式更新（Closed-form Updates）**  
  利用正交约束，使得：
  - **字典更新** 变为经典的 **Orthogonal Procrustes Problem**，可通过 SVD 得到闭式解。
  - **稀疏编码** 更新退化为解析性的硬阈值操作（hard-thresholding），无需迭代搜索。

- **动态压缩率分配策略（One-shot Dynamic Allocation）**  
  提出一种**单次全局分配策略**：将各层归一化后的奇异值池化，按全局压缩预算统一截断，并施加最小/最大压缩保护机制，防止敏感层被过度压缩或冗余层压缩不足。

- **与 PTQ 兼容性强**  
  COMPOT 可无缝结合 post-training quantization（如 GPTQ），实现极端压缩下的性能提升。

### 相比现有方法的优势
| 维度 | SVD-LLM 类方法 | CoSpaDi 类方法 | COMPOT |
|------|----------------|----------------|--------|
| 子空间建模 | 单一共享子空间 | 子空间联合（过完备） | 子空间联合（正交，闭式） |
| 更新方式 | 闭式 SVD | 迭代（K-SVD + OMP） | 闭式 Procrustes + Hard Thresholding |
| 压缩分配 | 固定或复杂优化 | 通常固定 | 单次全局池化 + 约束分配 |
| 速度 | 快 | 慢（高迭代成本） | 快（比 CoSpaDi 快 24x） |
| 性能 | 中等 | 较好但慢 | **最优质量-压缩权衡** |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **语言模型校准（Calibration）**：
  - `RefinedWeb`（256 条，长度 1024）
  - `WikiText`（256 条，长度 2048，用于部分复现实验）
- **下游任务评估基准**：
  - **常识推理**：PIQA, HellaSwag, LAMBADA
  - **科学问答**：ARC-e, ARC-c, SciQ
  - **阅读理解**：RACE
  - **综合知识**：MMLU
  - **通用基准**：Open LLM Leaderboard v2（含 MATH, GPQA, BBH, IFEval, MUSR）
- **多模态**：Qwen3-VL-8B-Instruct → MMMU, OCRBench, RealWorldQA, MMStar
- **语音**：Whisper 系列 → LibriSpeech test-clean/test-other（WER）

### 实验设置和评估指标
- **压缩比（Compression Ratio, CR）**：从 0.2 到 0.6 不等。
- **评估指标**：
  - **准确率（Accuracy）**：多个零样本任务平均准确率。
  - **困惑度（Perplexity）**：WikiText, C4, LAMBADA。
  - **词错误率（WER）**：语音识别任务。
- **压缩目标**：对所有 dense linear projections（Q/K/V/O, MLP gate/up/down）进行压缩，token embedding 和 lm_head 保持不变。
- **实现细节**：
  - 字典大小与稀疏度比 $ k/s = 2 $
  - 交替优化迭代次数：20 次（消融显示已收敛）
  - 使用 Cholesky 白化（whitening）以对齐功能误差

### 基线方法对比
- **SVD-based**：
  - SVD-LLM
  - SVD-LLM V2
  - Dobi-SVD（含训练）
- **Sparse Dictionary Learning**：
  - CoSpaDi（基于 K-SVD）
- **Structured Pruning**：
  - ReplaceMe
  - LLM-Pruner
- **Quantization**：
  - GPTQ（3/4-bit）
  - SmoothQuant, AWQ（间接比较）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（摘自 Table 3, 4, 5, 7, 9）

#### ✅ Llama3-8B @ CR=0.2
| Method | Avg. Acc. ↑ | WikiText PPL ↓ |
|--------|-------------|----------------|
| Original | 70.4 | 7.3 |
| SVD-LLM | 54.1 | 41.0 |
| CoSpaDi | 61.8 | 20.0 |
| **COMPOT** | **64.5** | **13.0** |

> ➤ COMPOT 在准确率上领先 CoSpaDi 2.7%，PPL 降低 35%。

#### ✅ Llama2-7B @ CR=0.4（vs Dobi-SVD）
| Method | Avg. Acc. ↑ | WikiText PPL ↓ |
|--------|-------------|----------------|
| Dobi-SVD | 36.6 | 53.4 |
| **COMPOT** | **44.4** | **20.4** |

> ➤ COMPOT 在**完全无训练**的情况下大幅超越需训练的 Dobi-SVD。

#### ✅ Llama-7B + GPTQ-4bit（总权重内存 2.8GB）
| Method | PPL ↓ |
|--------|-------|
| GPTQ-3bit | 16.28 |
| SVD-LLM V2 + GPTQ-4bit | 9.97 |
| **COMPOT + GPTQ-4bit** | **9.62** |

> ➤ COMPOT 提供的结构压缩增益可叠加于量化之上，达到最佳效果。

#### ✅ Whisper Large V3 @ CR=0.3
| Method | WER test-clean ↓ | test-other ↓ |
|--------|------------------|--------------|
| Original | 2.74 | 4.53 |
| SVD-LLM | 12.78 | 15.54 |
| **COMPOT** | **2.74** | **5.21** |

> ➤ COMPOT 几乎无损，而 SVD-LLM 严重退化。

### 与基线方法的对比结果
- **全面优于 SVD 类方法**：在语言、视觉语言、音频任务上均显著领先，尤其在高压缩比下优势更大。
- **优于 CoSpaDi**：尽管 CoSpaDi 也使用稀疏字典学习，但 COMPOT 因闭式更新更快且性能更高。
- **优于 Structured Pruning**：在相同压缩比下，COMPOT 保留更多语义信息，准确率和 PPL 显著更好（见 Table 6）。
- **兼容 PTQ**：COMPOT + GPTQ 超越单独 GPTQ 或 SVD-LLM + GPTQ。

### 消融实验结果
#### （1）字典初始化方式（Table 1）
| 初始化 | Avg. Acc. @ CR=0.2 |
|--------|-------------------|
| Random | 46.3 |
| **SVD-based** | **49.1** |

> ➤ 使用 SVD 前 $ k $ 个左奇异向量作为初始字典显著提升性能。

#### （2）交替优化迭代数（Figure 3）
- SVD 初始化约 100 次迭代即收敛，随机初始化需 ~300 次。
- 默认采用 20 次作为效率与性能平衡点。

#### （3）分组策略（Table 2）
| 分组方式 | Avg. Acc. |
|----------|-----------|
| 所有独立（All indiv.） | 48.5 |
| QKV & UpGate 分组 | 49.2 |
| **全部合并（All grouped）** | **50.1** |

> ➤ 全局奇异值池化效果最好，支持单次分配策略的有效性。

#### （4）字典-稀疏比 $ k/s $（Appendix A.8）
| $ k/s $ | Avg. Acc. |
|---------|-----------|
| 1.2 | 40.85 |
| 1.8 | 47.88 |
| **2.0** | **48.10** |
| 4.0 | 42.05 |

> ➤ $ k/s = 2 $ 为最优配置。

---

## 4. 关键结论和发现

### 主要发现
1. **正交字典 + 闭式更新** 是高效稀疏因子化的关键，在保持 union-of-subspaces 灵活性的同时极大加速优化过程。
2. **COMPOT 在质量-压缩权衡上全面超越 SVD 和稀疏字典学习基线**，成为当前最先进的结构化矩阵分解压缩技术。
3. **单次动态分配策略简单有效**，通过全局奇异值池化自动适应异构冗余，无需复杂搜索。
4. **与 PTQ 完全兼容**，可在极端压缩场景下进一步释放潜力。
5. **跨模态泛化能力强**：在语言、视觉语言、语音任务中均表现稳健。

### 方法的局限性
1. **依赖校准数据代表性**：若校准数据分布偏移或多样性不足，白化统计可能不可靠，影响压缩质量。
2. **Gram 矩阵需正定**：要求 $ G = XX^\top $ 可 Cholesky 分解；小样本或病态激活可能导致数值不稳定，需改用 SVD/Eigen 分解。
3. **固定稀疏模式**：当前使用固定 sparsity 结构，未联合优化稀疏性本身。
4. **无法处理非稠密投影**：仅适用于 dense linear layers。

### 未来工作方向
1. **轻量级“修复”步骤（healing step）**：在固定稀疏模式下微调因子，进一步提升精度。
2. **联合学习稀疏结构**：在保持结构约束（如 column-sparsity）的前提下，端到端学习稀疏模式。
3. **扩展至其他模块**：探索对 attention softmax、layernorm 等组件的压缩。
4. **硬件感知压缩**：结合实际推理引擎特性设计更高效的因子存储与计算方案。

---

> 🔗 **代码地址**：文中声明已开源（Code is available here），但未提供具体链接。  
> 📌 **影响声明**：COMPOT 有助于降低大模型部署门槛，促进资源受限环境下的 AI 应用，但也可能放大模型滥用风险，建议配合内容过滤与监控机制使用。

</details>

---

### 10. [CAMEL: An ECG Language Model for Forecasting Cardiac Events](https://arxiv.org/abs/2602.15677)

**Authors**: Neelay Velingker, Alaia Solko-Breslin, Mayank Keoliya, Seewon Choi, Jiayi Xin, Anika Marathe, Alireza Oraii, Rajat Deo, Sameed Khatana, Rajeev Alur, Mayur Naik, Eric Wong  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.15677v1  

#### Abstract
Electrocardiograms (ECG) are electrical recordings of the heart that are critical for diagnosing cardiovascular conditions. ECG language models (ELMs) have recently emerged as a promising framework for ECG classification accompanied by report generation. However, current models cannot forecast futur...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CAMEL: An ECG Language Model for Forecasting Cardiac Events

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前的 **ECG Language Models (ELMs)** 主要专注于 ECG 分类和报告生成，**无法预测未来的 cardiac events**（如房颤、室速等），限制了其在临床早期干预中的应用价值。  
本论文旨在填补这一空白，提出首个能够对长时程 ECG 信号进行推理并**预测未来心律失常事件**的通用 ECG 语言模型。

### 🚀 提出的新方法与核心创新
提出了 **CAMEL**（Cardiac Autoregressive Model for ECG Language Modeling），具备以下关键创新：

- **支持长时程 ECG 输入**：通过将每秒单导联 ECG 片段编码为独立 token，实现灵活的上下文长度支持（最长可达数千秒），突破传统 ELMs 仅限 10 秒输入的限制。
- **专用 ECG 编码器设计**：
  - 使用 3 层 CNN 将原始 ECG 波形映射到低维表示（`d=64`）；
  - 通过 SLP 投影层将其对齐至 LLM 的隐空间（`h=2560`）；
  - 支持多导联、变导联数输入，并引入 `lead start/end` 特殊 token 显式建模导联结构。
- **Lead-aware Attention Masking**：
  - 允许同一时间点的不同导联之间双向注意力交互，利用多导联同步性提升建模能力。
- **五阶段 Curriculum Learning 训练流程**：
  1. Autoencoder 预训练（自监督重建）
  2. 多选题与简答题（基础分类）
  3. 统计计算任务（如 RR 间期、HR、QRS 持续时间等）
  4. 多轮对话（结合统计与诊断推理）
  5. 预测任务微调（forecasting 报告生成）

### 🔍 相比现有方法的优势
| 方面 | CAMEL | 现有 ELMs（如 PULSE, GEM） |
|------|-------|-----------------------------|
| 输入时长 | 支持长达 10K 秒 | 通常 ≤10 秒 |
| 预测能力 | ✅ 支持未来事件预测 | ❌ 仅支持当前状态分类 |
| 导联灵活性 | 支持任意数量导联输入 | 固定 12 导联为主 |
| 可解释性 | 基于生理统计量生成证据链 | 多为黑箱输出 |
| 推理能力 | 显式整合 ECG 统计量用于逻辑推理 | 统计量非系统学习 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 类型 | 导联数 | 用途 |
|--------|------|--------|------|
| **Icentia11k** | 单导联长期记录 | 1 | Forecasting 主要测试集（AFIB/AFL/NORM beat-level 标注） |
| **PTB-XL** | 12-lead ECG | 12 | Classification, Report Generation |
| **CSN**, **CODE-15%**, **CPSC-2018**, **HEEDB** | 12-lead ECG | 12 | Classification（含 out-of-distribution 测试） |
| **MIMIC-IV-ECG** | 12-lead ECG | 12 | Report Generation, QA |
| **Penn**（私有） | ICU 8-lead ECG | 8 | Out-of-distribution 分类测试 |
| **ECGDeli** | 工具库 | — | 提取 ECG 统计量（RR, PR, QRS, QTc 等） |

> 注：Stage 1 自监督预训练使用超过 10 亿个来自 13 个公开数据库的 ECG 片段。

### 🧪 实验设置与评估指标

#### 下游任务与评价指标：
| 任务 | 指标 | 模型版本 |
|------|------|----------|
| **Forecasting** | Macro-F1（不同时间窗口 h 和输入长度 w） | CAMEL-Forecast |
| **Classification** | F1-score（zero-shot）、AUROC（linear probing） | CAMEL-Base |
| **Report Generation** | LLM-as-a-judge（GPT-5 打分）、BLEU, METEOR, ROUGE, BERT-F1 | CAMEL-Base |
| **QA** | Accuracy（单图）、Hamming Score（多图比较） | CAMEL-Base |
| **Grounding（统计准确性）** | RMSE（RR, HR, QRS, QTc, PR, QRS Amp） | CAMEL-Base |

#### 基线方法对比：
- **MELP**, **MERL**：基于 dual-encoder 的 zero-shot ECG 分类模型
- **PULSE**, **GEM**：多模态 LLM，支持图像/波形+文本联合建模
- **XGB**, **CNN**：全监督传统模型（用于 forecasting 对比）
- **GPT-5.2 + Code Interpreter**：接入原始 ECG CSV 文件进行预测的能力上限探索

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）在 **ECGForecastBench** 上的表现（Icentia11k）
| 方法 | h=60s, w=600s (F1) | h=300s, w=600s (F1) | h=600s, w=600s (F1) |
|------|--------------------|---------------------|---------------------|
| XGB（监督） | 55.20 | 53.37 | 57.07 |
| CNN（监督） | 53.10 | 48.99 | 56.19 |
| GPT-5.2（zero-shot） | 51.28 | 56.37 | 55.41 |
| PULSE | — | — | — |
| GEM | — | — | — |
| **CAMEL** | **75.14** | **70.48** | **76.02** |

✅ **CAMEL 在所有时间窗口下均显著优于监督模型和大模型基线，最高领先 GPT-5.2 超过 20% F1**

#### （2）在标准基准 **ECGBench** 上的 zero-shot 性能
| 方法 | PTB-XL Superclass (F1) | CSN (F1) | HEEDB (F1) | 平均增益 |
|------|------------------------|----------|------------|---------|
| MERL | 41.39 | 4.27 | 88.95 | — |
| PULSE | 73.47 | 12.62 | 86.36 | — |
| GEM | 75.77 | 8.19 | 88.95 | — |
| **CAMEL** | **67.75** | **12.84** | **96.41** | **+7.0% 绝对平均增益** |

➡️ 在 out-of-distribution 数据集 **CSN** 和 **Penn** 上表现尤为突出。

#### （3）报告生成质量（LLM-as-a-judge，满分 30 分）
| 方法 | PTB-XL | MIMIC-IV |
|------|--------|----------|
| GEM | 20.45 | 44.65 |
| **CAMEL** | 19.45 | **62.59** |

➡️ 在 MIMIC-IV 上超越 GEM 达 **18 分**，显示更强的临床对话理解与表达能力。

#### （4）ECG 统计量提取精度（RMSE）
| 方法 | CPSC-2018（平均 RMSE） |
|------|------------------------|
| GEM | 304 |
| **CAMEL** | **109** |

➡️ RMSE 不足 GEM 的 **1/3**，说明其更准确地掌握了 ECG 数值特征。

### 🔬 消融实验结果（Table 4）

| 消融配置 | PTB-XL Superclass F1 |
|----------|------------------------|
| 完整 CAMEL（LoRA + Lead-aware masking） | **75.91** |
| 移除 LoRA | 0.00（完全失效） |
| 使用 Full Bidirectional Masking | 54.39 |
| 使用 Causal Masking | 69.15 |

📌 结论：
- **LoRA 是关键可训练组件**，移除后模型无法有效学习；
- **Lead-aware attention masking 显著优于其他注意力机制**，验证了跨导联建模的重要性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **首次实现基于 ECG 语言模型的心脏事件预测功能**，CAMEL 能够从正常窦性心律中识别出即将发生的 AFIB/AFL 风险。
2. **长时程上下文建模至关重要**：随着输入 ECG 长度增加（从 10s 到 600s），预测性能持续提升（见 Fig. 3）。
3. **Curriculum Learning 极大地增强了模型的 ECG grounding 能力**，特别是在统计理解和多轮推理方面远超已有方法。
4. **CAMEL 在 zero-shot 设置下达到甚至超越 fully supervised 模型的表现**，展现出强大的泛化能力和临床实用性。

### ⚠️ 局限性
1. **Tokenization 策略受限于 LLM 上下文长度**：虽然支持长序列，但仍受制于 MedGemma 的最大 context window。
2. **1 秒切片可能截断 QRS 波形或丢失形态细节**，影响某些精细诊断任务。
3. 当前预测仅针对 **atrial arrhythmias（AFIB/AFL）**，尚未扩展至 VT/VF 等更危急事件。

### 🔮 未来工作方向
- 探索基于 **QRS 波群边界** 或 **beat-level segmentation** 的语义 tokenization 策略；
- 扩展至 **multi-modal forecasting**（结合 vitals, lab results）；
- 开发面向 **real-time ICU monitoring** 的 streaming inference 架构；
- 引入 **causal reasoning** 框架以支持治疗建议生成。

---

> 💡 **一句话总结**：  
> CAMEL 是首个支持长时程 ECG 推理与未来心脏事件预测的 ECG Language Model，通过专用编码器 + Curriculum Learning + Lead-aware Attention，在 forecasting、classification、report generation 等多项任务上实现了 SOTA 表现，为 AI 辅助早期干预提供了新范式。

</details>

---

### 11. [AgriWorld:A World Tools Protocol Framework for Verifiable Agricultural Reasoning with Code-Executing LLM Agents](https://arxiv.org/abs/2602.15325)

**Authors**: Zhixing Zhang, Jesen Zhang, Hao Liu, Qinhan Lv, Jing Yang, Kaitong Cai, Keze Wang  
**Category**: cs.AI  
**Published**: 2026-02-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.15325v1  

#### Abstract
Foundation models for agriculture are increasingly trained on massive spatiotemporal data (e.g., multi-spectral remote sensing, soil grids, and field-level management logs) and achieve strong performance on forecasting and monitoring. However, these models lack language-based reasoning and interacti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AgriWorld: A World-Tools-Protocol Framework for Verifiable Agricultural Reasoning with Code-Executing LLM Agents

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前农业领域的 **Foundation Models** 虽然在预测和监测任务上表现优异，但缺乏自然语言交互能力和可解释的推理过程；而 **Large Language Models (LLMs)** 虽擅长文本理解和生成，却无法直接处理高维、异构的农业时空数据（如遥感影像、土壤网格、田块管理日志等）。这导致两者均难以支持复杂的农学分析流程，例如“如果减少10%灌溉，作物水分胁迫会如何变化？”这类需要模拟与验证的“what-if”反事实分析。

此外，传统基于纯文本的LLM响应容易产生**幻觉**（hallucination），且中间计算步骤不可审计，限制了其在科学决策中的可信度。

---

### 🚀 提出的新方法与新思路

作者提出一个名为 **AgriWorld** 的 **World-Tools-Protocol 框架**，并构建了一个执行驱动的多轮智能体 **Agro-Reflective**，实现可验证的农业推理。该框架包含三个核心组件：

#### （1）**AGRIWORLD：可执行的农业环境**
- 提供统一的 Python 执行环境，封装了四大类工具：
  - **Geospatial Querying**：田块过滤、空间连接（sjoin）
  - **Remote-Sensing Analytics**：时序NDVI计算、异常检测
  - **Crop Simulation & Predictors**：产量、胁迫、病害风险预测
  - **Soil/Terrain & Weather Access**：土壤采样、气象聚合
- 支持**中间产物（Artifacts）输出**（如表格、图表、掩膜），确保每一步计算可追溯、可审计。
- 内建 **Canonical Alignment Operator**，自动进行坐标系重投影（CRS）和时间对齐（Resampling），避免因单位或时空错位导致错误。

#### （2）**AGRO-REFLECTIVE：执行驱动的反思型Agent**
- 采用 **Execute-Observe-Refine 循环机制**：
  1. 生成代码 → 
  2. 在 AGRIWORLD 中执行 → 
  3. 观察返回结果与错误 → 
  4. 自我诊断并修正代码
- 利用中间产物元信息（meta）进行**自我纠错**，识别 `SpatialMisalignment`、`UnitError` 等错误类型，并动态修复。

#### （3）**Verifiable Evaluation Protocol**
- 构建 **AGROBENCH** 可验证评测套件，将问题转化为带有确定性参考程序和可执行检查器的任务实例。
- 引入多层次验证协议：
  - **Schema Validity**：输出格式合规
  - **Numeric Tolerance**：数值误差容忍范围
  - **Counterfactual Consistency**：反事实干预方向正确性
  - **Physical Sanity**：单位一致性、有效像素覆盖率

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | AgriWorld + Agro-Reflective |
|------|--------|-------------------------------|
| 推理方式 | 纯文本生成 / 单次代码调用 | 多轮执行-观察-反思循环 |
| 数据处理 | 依赖模型记忆或外部描述 | 直接操作原始遥感、气象、土壤数据 |
| 错误纠正 | 无反馈机制 | 基于运行时错误自修复 |
| 结果可信度 | 难以验证 | 中间产物可审计，全过程可复现 |
| 泛化能力 | 易过拟合训练区域 | 动态查询真实世界数据，OOD鲁棒性强 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

基于真实农学工作流构建的 **AGROBENCH** 评测集，涵盖以下任务类型：
- **Lookups**：田块属性查询（作物类型、面积等）
- **Forecasting**：产量、水分趋势预测
- **Anomaly Detection**：识别干旱/胁迫区域
- **Counterfactual Analysis**：“如果调整灌溉/施肥”类假设分析

数据来源包括：
- 多光谱遥感时序（Sentinel-2）
- 土壤质地与DEM栅格
- 田块边界矢量（Parcels）
- 气象站时间序列
- 农事管理日志（灌溉、施肥记录）

---

### 🧪 实验设置与评估指标

#### 模型配置
- 主干模型：Qwen3-8B 和 Qwen3-32B
- 微调方式：LoRA（Low-Rank Adaptation）
- 最大交互步数：T_max = 20

#### 评估指标分类

| 任务类型 | 评估指标 |
|---------|----------|
| Lookup | Exact Match Accuracy |
| Forecasting | NRMSE（归一化均方根误差）↓ |
| Anomaly Detection | Spatial IoU（交并比）↑ |
| Counterfactual Analysis | Success Rate（因果方向正确率）↑ |
| 整体性能 | Aggregate QA Score / Choice Accuracy |

#### 基线方法对比
| 方法 | 类型说明 |
|------|----------|
| **Text-Only** | 仅使用参数化知识作答（如 Qwen3-32B 原生模型） |
| **AgriWorld-Direct (One-shot)** | 一次性生成并执行代码，无错误处理 |
| **Proprietary Models** | GPT-4o, Gemini-2.0 |
| **Open-Source Baselines** | DeepSeek-V3, Yi-1.5-34B |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 3 & 4）

| Model | Aggregate QA Score | Choice Accuracy |
|-------|--------------------|------------------|
| **Qwen3-32B-LoRA-Reflective (Ours)** | **7.72 / 541** | **73.84%** |
| GPT-4o | 7.07 / 150 | 36.75% |
| Qwen3-32B (LoRA tuned, non-reflective) | 7.42 / 541 | 67.25% |

> 💡 **关键发现**：尽管 GPT-4o 是超大规模模型，但在农业领域表现远低于本方法，因其常**虚构不存在的API函数**（如 `get_soil_moisture()`），而 Agro-Reflective 严格遵循 AGRIWORLD 工具接口。

---

### 🔬 细粒度任务性能对比（Table 4）

| Model | Lookup Acc. | Forecasting (NRMSE↓) | Anomaly (IoU↑) | Counterfactual (SR↑) |
|-------|-------------|------------------------|----------------|------------------------|
| Text-Only | 41.2% | 0.89 | 0.12 | 14.5% |
| One-shot | 82.5% | 0.34 | 0.51 | 43.8% |
| **Agro-Reflective** | **86.7%** | **0.18** | **0.68** | **71.4%** |

> ✅ 在复杂多步任务中优势显著：
- **Forecasting 误差降低 47%**
- **Counterfactual 成功率提升近一倍**

---

### 🔍 消融实验结果（Table 2）

| 方法变体 | 平均准确率 | 分析结论 |
|--------|------------|-----------|
| w/o Remote Sensing | 41.5% | 移除遥感能力后性能骤降，证明像素级数据不可替代 |
| w/o Alignment | 51.3% | 缺少时空对齐导致大量空结果（CRS不匹配） |
| w/o Reflection | 50.6% | 无反思机制则无法修复初始错误，失败率高 |
| **Full Agro-Reflective** | **57.6%** | 完整系统性能最优 |

> ⚠️ **关键结论**：`Alignment` 与 `Reflection` 是两大支柱，缺一不可。

---

### ⏱️ 效率分析（Figure 2）

- 性能随交互轮次呈**对数饱和增长**
- 前 4 步带来最大增益（48.2% → 68.5%）
- 平均收敛步数仅为 **3.5**，表明验证协议能高效终止

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Executable Reasoning > Text Generation**
   - 农业科学推理必须建立在可执行、可验证的计算之上，而非仅靠语言模式匹配。
   - “看到卫星图像”比“听说田地是绿色的”更能支撑精准判断。

2. **Reflection Loop 显著提升可靠性**
   - 35% 的初始代码存在 bug，但通过执行反馈可在 92% 情况下成功修复。
   - 错误类型主要包括：CRS 不一致、时间窗口错位、单位混淆。

3. **Tool Grounding 抵御幻觉**
   - 通用大模型（如 GPT-4o）倾向于编造工具调用，而本方法受限于真实 API 接口，输出更可靠。

4. **OOD 场景下仍保持稳健（Table 5）**

| 方法 | Spatial OOD Δ | Temporal OOD Δ |
|------|---------------|----------------|
| Text-Only | -55% | -46% |
| One-shot | -45% | -26% |
| **Agro-Reflective** | **-12%** | **-6%** |

> 在巴西（未见区域）和厄尔尼诺年份（极端气候）测试中，本方法泛化能力显著优于基线。

---

### ⚠️ 局限性

1. **依赖高质量数据接入**
   - 若遥感云覆盖严重或气象数据缺失，可能触发 `LowCoverageError` 导致任务失败。
   
2. **计算成本较高**
   - 尽管平均只需 3.5 轮，但在资源受限边缘设备部署仍有挑战。

3. **仿真器精度瓶颈**
   - 最终结论受限于底层 crop simulator 的物理建模准确性。

---

### 🔮 未来工作方向

1. **扩展至更多农业子领域**
   - 如畜牧、水产养殖、林业等（文中已初步涉及 Animal/Aquatic/Herb 分支）

2. **引入多Agent协作机制**
   - 不同专家Agent分工合作（如一个负责遥感分析，另一个负责水文模拟）

3. **支持非Python语言与低代码界面**
   - 降低农业从业者使用门槛

4. **结合主动学习优化数据采集策略**
   - Agent 主动建议部署传感器或申请更高分辨率影像

---

> 📌 **总结一句话**：  
> 本文提出了首个面向农业科学的 **可执行、可验证、可反思** 的 LLM Agent 框架，证明了“让模型动手做实验”比“让它凭空说答案”更可靠、更可信赖，为可信农业科技迈向实用化提供了重要路径。

</details>

---

### 12. [Improving LLM Reliability through Hybrid Abstention and Adaptive Detection](https://arxiv.org/abs/2602.15391)

**Authors**: Ankit Sharma, Nachiket Tapas, Jyotiprakash Patra  
**Category**: cs.AI  
**Published**: 2026-02-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.15391v1  

#### Abstract
Large Language Models (LLMs) deployed in production environments face a fundamental safety-utility trade-off either a strict filtering mechanisms prevent harmful outputs but often block benign queries or a relaxed controls risk unsafe content generation. Conventional guardrails based on static rules...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Improving LLM Reliability through Hybrid Abstention and Adaptive Detection

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对当前 **Large Language Models (LLMs)** 在生产环境中面临的核心挑战——**安全-效用权衡（safety-utility trade-off）**。具体表现为：
- **静态安全机制（static guardrails）** 过于保守，常误拒良性查询（high false positives），影响用户体验；
- **基于全局置信度的拒绝机制（confidence-based abstention）** 依赖单一信号（如熵、困惑度），易被“高置信错误”误导，安全性不足；
- 现有方法普遍**缺乏上下文感知能力**，无法根据不同领域（如医疗 vs 创意写作）动态调整策略；
- 外部安全层（如外部API）引入显著**延迟（latency）**，不利于实时交互系统。

### 提出了什么新方法或新思路
作者提出了一种名为 **Adaptive Abstention System** 的新型安全框架，其核心创新包括：

- **自适应拒绝机制（Adaptive Abstention）**：  
  动态调整各检测器的阈值 $T_{dynamic}(c, u)$，依据**上下文（context c）**（如领域敏感性）和**用户历史（user history u）**（如信任评分），实现“高风险场景更严格，低风险场景更宽松”的智能控制。

- **多维并行检测架构（Multi-dimensional Parallel Detection）**：  
  设计五个独立的检测器，分别评估不同维度的风险：
  1. **Safety Detector**：检测毒性、自残、越狱尝试；
  2. **Confidence Detector**：结合 perplexity、entropy 和 variance 估计不确定性；
  3. **Knowledge Boundary Detector**：识别超出模型知识范围的查询（如时效性问题）；
  4. **Contextual Detector**：判断响应在特定领域的语调与得体性；
  5. **Repetition Detector**：通过 embedding 相似性防止语义循环。

- **层级级联优化（Hierarchical Cascade Architecture）**：  
  将检测器按计算成本从低到高组织成四级级联：
  - Level 1（<1ms）：关键词/正则快速过滤（处理 ~70% 请求）；
  - Level 2（~5ms）：轻量级分类器；
  - Level 3（20–50ms）：五维并行深度检测；
  - Level 4：可选高成本验证。
  只有未被前几级明确判定的模糊请求才会进入下一级，大幅降低平均延迟。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **安全性** | 多维度联合判断，减少单信号误判；支持 near-perfect recall（1.00） |
| **实用性（Utility）** | 自适应阈值显著降低 false positives，提升良性请求通过率 |
| **效率** | 级联设计使平均延迟降至 **42.78ms**，相较外部 guardrails（450ms）提速 **10.5×** |
| **通用性** | 模型无关（model-agnostic），无需微调即可部署于任意 LLM |
| **可扩展性** | 模块化设计便于添加新检测器或调整策略 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验基于三个定制数据集进行评估：
1. **Efficiency Dataset**（1,000 条）：混合领域请求，包含安全、不安全及对抗性提示，模拟真实流量。
2. **Safety Dataset**（600 条）：源自 **RealToxicityPrompts**，用于评估严格模式下的安全拦截能力。
3. **Adaptive Dataset**：标注了领域的提示集合，涵盖 **Medical、Education、Casual、Creative Writing** 四类场景，专门用于测试上下文感知能力。

此外，在 Figure 6 中还使用了 **Meta AbstentionBench** 跨模型基准，覆盖 6 个主流 LLM（如 `llama-3.3-70b-instruct`, `gemma-2-9b-it` 等）和 8 类任务。

### 实验设置和评估指标
- **基础模型保持一致**，仅比较不同 abstention 层的影响；
- 所有实验在离线环境运行以确保可复现性；
- **评估指标**：
  - 安全性：**Precision（效用）、Recall（安全）、F1 Score、False Positive Rate (FPR)**
  - 效率：**Average Latency per Request (ms)**
  - 上下文适应性：**Domain-specific FPR**

### 基线方法对比
与以下三种配置进行对比：
1. **Guardrails AI**：使用外部 ToxicLanguage 验证器作为安全层（代表传统外部 guardrails）；
2. **Static Threshold**：固定所有检测器阈值为 0.6，无上下文自适应；
3. **No Cascade**：禁用级联优化，所有请求均执行完整五维检测。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **平均延迟（含级联）** | **42.78 ms** |
| **严格安全模式 Recall** | **1.00**（完美捕捉所有不安全请求） |
| **校准后 Precision** | **>0.95**（兼顾安全与可用性） |
| **F1 Score（自适应 vs 静态）** | **0.96 vs 0.77**（+24.7%） |
| **False Positives（自适应 vs 静态）** | **3 vs 15**（↓80%） |

### 与基线方法的对比结果
#### ✅ 延迟表现（Table 2）
| 方法 | 平均延迟 (ms) | 相对加速 |
|------|----------------|-----------|
| Guardrails AI（外部API） | 450.00 | 1.0× |
| No Cascade（本方法无级联） | 118.26 | 3.8× |
| **Cascade（本方法）** | **42.78** | **10.5×** |

> → 级联机制带来 **10.5倍** 于外部 guardrails 的速度提升。

#### ✅ 安全与效用平衡（Figure 1 & Table 4）
- 在 **Strict Safety Mode** 下达到 **Recall=1.00**, Precision=0.50（保守设定）；
- 启用自适应阈值后：
  - **Precision ↑ 26.7%**（0.75 → 0.95）
  - **Recall ↑ 22.5%**（0.80 → 0.98）
  - **F1 Score ↑ 24.7%**
  - **False Positives ↓ 80%**

#### ✅ 领域特异性改进（Figure 5）
| 领域 | 静态阈值 FPR | 自适应阈值 FPR | 改进 |
|------|---------------|------------------|--------|
| **Creative Writing** | 25% | **3%** | ↓88% |
| **Medical** | 15% | **2%** | ↓87% |

> → 自适应机制有效缓解了在专业领域中的“过度拒绝”问题。

### 消融实验结果（Ablation Study）
- **Repetition Detector 消融**：
  - 开启时：成功阻止 **100%** 的无限生成循环；
  - 关闭时：标准 guardrails 无法及时干预，直到上下文窗口耗尽。
- **级联机制消融**：
  - 移除级联后延迟上升至 118.26ms（↑176%），证明其对效率的关键作用。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **安全与效用并非不可调和**：通过**上下文感知的自适应阈值调节**，可在高风险领域（如医疗）保持高安全性，在低风险领域（如创意写作）释放更高自由度，实现双赢。
2. **多维检测优于单信号决策**：融合 safety、confidence、knowledge boundary、contextual appropriateness 和 repetition 五大维度，显著增强系统鲁棒性。
3. **级联架构是高效部署的关键**：将 70% 的简单请求在 <1ms 内处理完毕，仅对复杂案例启用深度分析，实现了高性能与强安全的统一。
4. **该框架具有良好的泛化能力**：在多个 LLM 和多样化任务中均表现出稳定的性能（见 Figure 6），适用于实际生产环境。

### 方法的局限性
1. **对抗性越狱仍具挑战**：某些利用“良性表层语言”的高级 jailbreak 技巧可能绕过检测；
2. **内存开销**：Repetition Detector 需维护历史 response embeddings，带来额外内存负担；
3. **新领域需初始校准期**：阈值参数需根据新部署领域的风险特征进行调优，初期可能存在不适配；
4. **依赖高质量上下文标注**：领域识别与用户信任建模的准确性直接影响系统效果。

### 未来工作方向
1. **联邦学习用于隐私保护的阈值自适应**：在不共享用户数据的前提下实现跨用户/部署的安全策略协同优化；
2. **扩展至多模态场景（Multimodal Settings）**：将框架应用于图像、视频等内容的安全审核；
3. **增强可解释性（Explainability）**：为每次 abstention 决策生成简洁的人类可读理由，提升透明度与用户信任；
4. **动态用户信任建模**：构建更精细的 user trust scoring 机制，结合行为模式持续更新信任等级。

---

> **总结一句话**：  
> 本文提出的 **Adaptive Abstention System** 通过“**上下文感知 + 多维检测 + 级联加速**”三位一体设计，在几乎不牺牲响应速度的前提下，显著提升了 LLM 的安全可靠性，为负责任的 AI 部署提供了实用且可扩展的解决方案。

</details>

---

### 13. [Co-Design and Evaluation of a CPU-Free MPI GPU Communication Abstraction and Implementation](https://arxiv.org/abs/2602.15356)

**Authors**: Patrick G. Bridges (University of New Mexico), Derek Schafer (University of New Mexico), Jack Lange (Oak Ridge National Laboratory), James B. White III (Oak Ridge National Laboratory), Anthony Skjellum (Tennessee Technological University), Evan Suggs (Tennessee Technological University), Thomas Hines (Tennessee Technological University), Purushotham Bangalore (University of Alabama), Matthew G. F. Dosanjh (Sandia National Laboratories), Whit Schonbein (Sandia National Laboratories)  
**Category**: cs.DC  
**Published**: 2026-02-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.15356v1  

#### Abstract
Removing the CPU from the communication fast path is essential to efficient GPU-based ML and HPC application performance. However, existing GPU communication APIs either continue to rely on the CPU for communication or rely on APIs that place significant synchronization burdens on programmers. In th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Co-Design and Evaluation of a CPU-Free MPI GPU Communication Abstraction and Implementation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代高性能计算（HPC）和机器学习（ML）应用中，GPU通信的性能瓶颈日益突出。传统 **GPU-aware MPI** 通信依赖 **CPU 在通信快路径（fast path）** 上进行同步、消息匹配和缓冲区准备，导致显著的延迟开销（通常增加 2–4 微秒），尤其在中小消息通信中严重影响强扩展性。

此外，现有的“CPU-free”通信方案（如 NVSHMEM、one-sided MPI）虽然能绕过 CPU，但往往牺牲了 **MPI 的标准语义**（如 message matching、two-sided 数据移动），限制了其在通用 HPC 应用中的可用性。

### 提出的新方法与创新思路
本文提出并实现了一种 **基于 MPI 的、支持 stream-triggered 的 GPU 通信 API**，实现了真正的 **CPU-free two-sided MPI 通信**，其核心创新包括：

- **基于 MPI Persistent Operations 的设计**  
  复用 `MPI_Send_init` / `MPI_Recv_init` 等持久化操作，将复杂的 setup 阶段（如 message matching、RMA key 交换）移出 fast path，仅在初始化阶段由 CPU 完成。

- **引入 `MPI_Match` 接口**  
  显式完成 message matching 并生成“永久配对”的请求，使后续 GPU 触发的通信无需运行时匹配，从而支持 GPU stream 触发。

- **引入 `MPI_Queue` 抽象**  
  将通信操作（start/wait）与 GPU stream 绑定，允许在 GPU stream 中按序执行通信任务，实现与计算的无缝流水。

- **利用 HPE Slingshot 11 和 libfabric 的硬件特性**  
  借助 **Deferred Work Queue (DWQ)** 和 **GPU-accessible counters**，实现 GPU 内核直接触发 NIC 操作，完全绕过 CPU。

### 相比现有方法的优势
| 特性 | 本文方法 | 传统 MPI | NCCL | NVSHMEM |
|------|--------|--------|------|---------|
| CPU-free fast path | ✅ | ❌ | ❌（旧版）✅（新实验版） | ✅ |
| 支持 two-sided 语义 | ✅ | ✅ | ❌（本质是 collective） | ❌（one-sided） |
| 支持 message matching | ✅（通过 `MPI_Match` 预先完成） | ✅ | ❌ | ❌ |
| 易于集成到现有框架 | ✅（兼容 MPI 语义） | ✅ | ❌（专用 API） | ❌ |
| 支持 stream-triggered | ✅ | ❌ | ✅（有限） | ✅（kernel-triggered） |

> ✅ 本文在保持 MPI 易用性的同时，首次实现了 **完整的 CPU-free two-sided MPI 通信**。

---

## 2. 核心实验方法和设置

### 实验平台
- **Frontier 超算**（ORNL）
  - 9,408 节点，每节点 1×AMD EPYC CPU + 4×MI250X GPUs（共 8 GCDs）
  - HPE Slingshot 11 网络（200Gbps，Dragonfly 拓扑）
- **Tuolumne 超算**（LLNL）
  - 1,152 节点，每节点 4×MI300A APU（含 CPU + GPU）
  - 同样使用 Slingshot 11 网络

### 基准测试
1. **GPU Ping-Pong Microbenchmark**
   - 测试点对点通信的 **延迟与带宽**
   - 消息大小从 1B 到 1GB（2 的幂次）
   - 对比：Cray MPICH vs. 本文提出的 **stream-triggered send/ready send**

2. **CabanaGhost Halo Exchange Benchmark**
   - 基于 **Cabana/Kokkos** 性能可移植框架
   - 实现 8-point stencil 通信模式（典型 halo exchange）
   - 测试 **强扩展性**（strong scaling）
   - 问题规模：
     - 小问题：2GB
     - 大问题：30GB（Frontier）、62GB（Tuolumne）
   - 运行 1,000 次迭代，测量求解时间

### 评估指标
- **Ping-Pong 延迟与带宽**
- **强扩展速度提升（speedup）**：相对于单 GPU Cray MPICH 的加速比
- **最大平均速度提升（maximum mean speedup）**
- **不同消息长度下的性能增益**

### 基线方法
- **Cray MPICH**（默认 MPI 实现，CPU-driven）
- **Stream-triggered standard send**
- **Stream-triggered ready send**（优化版本，无需 CTS 同步）

---

## 3. 主要实验结果和性能指标

### GPU Ping-Pong 性能（Frontier）
- **延迟降低最多达 50%**
  - 在 4KB–1MB 消息范围内，stream-triggered send 比 Cray MPICH 快 **12–49%**
  - 例如：32B–512KB 消息延迟降低 **12–39%**
- **带宽提升明显**
  - 在 4KB–1MB 区间，带宽显著高于 Cray MPICH
  - 但在 >8MB 大消息上，Cray MPICH 因硬件优化反超（因当前实现未针对大消息优化）

> 图 5 显示，在中等消息下，stream-triggered 实现了更低延迟和更高带宽。

### Halo Exchange 强扩展性（Frontier）

| 问题规模 | 方法 | 最大平均速度提升 | 达到峰值的节点数 / GPU 数 |
|--------|------|------------------|----------------------------|
| 2.0 GB | Cray MPICH | 50.23× | 1,024 / 1,024 |
|        | Stream Rsend | **50.23×** | **512 / 1,024** |
|        | Stream Send | 40.49× | 512 / 1,024 |
| 30.0 GB | Cray MPICH | 485.4× | 1,024 / 4,096 |
|         | Stream Rsend | **622.2×**（↑28%） | 1,024 / 8,192 |
|         | Stream Send | 543.7× | 1,024 / 4,096 |

#### 关键发现：
- **Stream-triggered ready send 在大问题上提速 28%**
- **仅用一半节点即达到相同小问题性能** → 更高效的资源利用率
- 在 8,192 GPU 规模下仍保持良好扩展性

### 消融分析（图 8）：按消息大小分析
- **中等消息（~8KB–1MB）性能最佳**
  - 正是 halo exchange 中边（edge）通信的典型大小
- **小消息（<8KB）性能不如 Cray MPICH**
  - 原因：当前实现需等待 CTS（Clear-To-Send）信号，而 Cray MPICH 可使用 Slingshot 的 **unexpected message 硬件机制** 避免此等待
  - 说明：**未启用 bounce buffer 优化**
- **Ready send 避免 CTS 开销，在小消息上表现更好**

> 表明：**性能收益高度依赖通信模式和消息大小**，中等消息受益最大。

---

## 4. 关键结论和发现

### 主要结论
1. ✅ **成功实现了 CPU-free 的 two-sided MPI GPU 通信**  
   通过 `MPI_Queue` + `MPI_Match` + Persistent Ops 的协同设计，完全移除 CPU 在通信 fast path 中的角色。

2. ✅ **显著降低通信延迟**  
   在中等消息（4KB–1MB）上，延迟最高降低 **50%**，带宽显著提升。

3. ✅ **大幅提升强扩展性能**  
   在 8,192 GPU 规模的 halo exchange 基准上，**最大速度提升达 28%**，证明其在大规模 HPC 应用中的巨大潜力。

4. ✅ **保持 MPI 易用性**  
   与现有 MPI 语义高度兼容，易于集成到 Cabana/Kokkos 等性能可移植框架。

### 方法的局限性
- **小消息性能尚未优化**  
  当前实现对 <8KB 消息需等待 CTS，导致性能低于 Cray MPICH。
- **依赖特定硬件特性**  
  依赖 HPE Slingshot 11 的 DWQ 和 GPU-accessible counters，目前仅在 Slingshot 上实现。
- **未支持 InfiniBand 或 NCCL 对接**  
  无法与 NVIDIA GPU-Initiated Networking 直接比较。
- **资源受限风险**  
  DWQ 条目数量有限（约 500），高并发下可能耗尽，需 CPU 协助回收。

### 未来工作方向
1. **小消息优化**  
   引入 **bounce buffer** 机制，对小消息进行暂存，避免 CTS 等待。
2. **支持 stream-triggered Collective Operations**  
   扩展 API 支持 `MPI_Bcast`、`MPI_Reduce` 等集合通信。
3. **跨架构移植**  
   移植到 NVIDIA GPU + InfiniBand 平台，与 NCCL GPU-Initiated Networking 直接对比。
4. **集成更多应用**  
   在 ML、CG Solver、Lattice Boltzmann 等应用中验证通用性。
5. **自动资源管理**  
   设计运行时机制动态管理 DWQ 资源，避免死锁。

---

> **开源地址**：  
> - Stream-triggering 实现：[https://github.com/MPI-Advance/stream-triggering](https://github.com/MPI-Advance/stream-triggering)  
> - 修改版 Cabana：[https://github.com/CUP-ECS/Cabana](https://github.com/CUP-ECS/Cabana)  
> - CabanaGhost 基准：[https://github.com/CUP-ECS/CabanaGhost](https://github.com/CUP-ECS/CabanaGhost)

</details>

---

### 14. [Learning Data-Efficient and Generalizable Neural Operators via Fundamental Physics Knowledge](https://arxiv.org/abs/2602.15184)

**Authors**: Siying Ma, Mehrdad M. Zadeh, Mauricio Soroco, Wuyang Chen, Jiguo Cao, Vijay Ganesh  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.15184v1  

#### Abstract
Recent advances in scientific machine learning (SciML) have enabled neural operators (NOs) to serve as powerful surrogates for modeling the dynamic evolution of physical systems governed by partial differential equations (PDEs). While existing approaches focus primarily on learning simulations from ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning Data-Efficient and Generalizable Neural Operators via Fundamental Physics Knowledge

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前基于数据驱动的 **Neural Operators (NOs)** 在模拟由偏微分方程（PDE）控制的物理系统时面临三大挑战：

1. **高数据需求**（High data demands）：缺乏物理先验，需要大量高质量训练数据才能达到高精度。
2. **物理不一致性**（Physical inconsistency）：模型可能违反守恒律等基本物理规律，尤其在长期 rollout 预测中产生非物理解。
3. **泛化能力差**（Poor generalization）：对未见过的参数设置（如粘度变化）、边界条件或真实世界场景（synthetic-to-real）泛化能力弱。

尽管已有 **SciML foundation models**（如 MPP, DPOT）通过多物理场预训练提升泛化性，但它们仍忽略更底层的**基础物理知识**（fundamental physics knowledge），导致上述问题依然存在。

---

### 提出了什么新方法或新思路

作者提出一种**原则性强、架构无关**（architecture-agnostic）的方法，将**基础物理知识显式地融入 NO 的训练过程**中。

#### 核心思想：
从原始 PDE 中分解出其“基本形式”（basic form），即保留主导物理机制、去除计算刚性项后的简化版本，并联合训练原始 PDE 和该基本形式。

#### 具体实现：
- **定义基本形式**：例如：
  - 对 **Diffusion-Reaction 方程**，去掉非线性反应项，保留纯扩散项。
  - 对 **Navier-Stokes 方程**，去掉压力梯度和扩散项，保留对流项 $ \partial_t u = -(u \cdot \nabla)u + f $。
  - 对 **Kuramoto-Sivashinsky 方程**，去掉非线性对流项，保留线性稳定/失稳项。
- **多物理场联合训练**（Multiphysics Joint Training）：
  - 构建混合数据集：同时包含原始 PDE 和其基本形式的仿真数据。
  - 利用基本形式仿真成本远低于原始 PDE 的特性，在相同计算预算下可生成更多样本，实现“以廉价仿真换数据量”。
  - 采用多任务学习框架，共享主干网络，仅解耦最后预测层。

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **数据效率** | 显著降低仿真成本，在更少的原始 PDE 数据下达到更高精度（见 Figure 5）。 |
| **泛化能力** | 在 OOD 参数设置和 synthetic-to-real 场景下表现更强（Table 2, Figure 7）。 |
| **物理一致性** | 长期 rollout 更稳定，误差累积更慢（Figure 6）。 |
| **通用性** | 方法不依赖特定 NO 架构（验证于 FNO 和 Transformer），且适用于多种 PDE 类型（1D/2D/3D）。 |

> ✅ **关键洞见**：基础物理项虽简单，却编码了丰富的物理归纳偏置；显式学习这些项能显著增强模型的综合泛化能力。

---

## 2. 核心实验方法和设置

### 使用的数据集与 PDE 类型

研究涵盖四类典型 PDE 及其基本形式：

| PDE 名称 | 维度 | 物理意义 | 基本形式 |
|--------|-----|---------|--------|
| **Diffusion-Reaction** | 2D | 化学/生物中的图灵斑图形成 | 纯扩散方程（去除非线性反应项） |
| **Navier-Stokes (2D/3D)** | 2D / 3D | 流体动力学 | 对流主导方程（$ \partial_t u = -(u\cdot\nabla)u + f $） |
| **Kuramoto-Sivashinsky (KS)** | 1D | 耗散混沌系统 | 线性稳定/失稳项（$ \partial_t u = -\partial_{xx}u - \partial_{xxxx}u $） |
| **ScalarFlow** | 3D | 真实烟雾流动重建数据集 | 用于 synthetic-to-real 泛化测试 |

所有仿真数据均通过标准数值求解器生成（如 PhiFlow, PyClaw）。

---

### 实验设置和评估指标

#### 主要设置：
- **训练策略**：Baseline 仅训练原始 PDE；Ours 联合训练原始 + 基本形式。
- **样本混合比例**（Sample Mixture Ratio）：根据仿真耗时设定，确保总仿真成本与 Baseline 相当甚至更低。
  - 如 2D Navier-Stokes：原始 vs 基本形式耗时比 ≈ 1:24 → 混合比为 1:24。
- **模型架构**：主要使用 **FNO**，部分实验使用 **Transformer**。
- **优化配置**：Adam + Cosine Annealing，固定梯度步数以公平比较。

#### 评估指标：
- **nRMSE**（Normalized Root Mean Squared Error）：主评价指标。
- **Rollout Performance**：自回归 rollout 多步预测，衡量长期一致性。
- **OOD Generalization**：测试不同物理参数下的表现（如不同粘度 ν）。
- **Synthetic-to-Real Transfer**：在真实烟雾数据集 ScalarFlow 上测试。

---

### 基线方法对比

| 基线方法 | 描述 |
|--------|------|
| **Baseline** | 仅在原始 PDE 上训练的 NO（FNO 或 Transformer） |
| **Baseline@Spatiotemporal** | 在低时空分辨率上训练并插值回原分辨率，作为低成本替代方案 |
| **Lie Transform Augmentation** | 基于对称性的数据增强方法，用于验证正交性 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）数据效率提升（Figure 5）
- 在相同仿真成本下，**Ours 方法全面优于 Baseline**。
- 所有 PDE 上均观察到“更低的 nRMSE + 更低的成本”优势。
- 例如在 2D Diffusion-Reaction 中，节省约 50% 原始仿真成本即可获得相当甚至更好的性能。

#### （2）OOD 泛化能力（Table 2）
| PDE | 设置 | Baseline (nRMSE) | Ours (nRMSE) | 改进幅度 |
|-----|------|------------------|-------------|----------|
| **Diffusion-Reaction** | β=1 | 0.0413 | **0.0331** | ↓19.8% |
| | β=100 | 0.0770 | **0.0538** | ↓30.1% |
| **Navier-Stokes (2D)** | ν=0.05 | 0.0825 | **0.0222** | ↓73.1% |
| | ν=0.0001 | 0.0369 | **0.0125** | ↓66.1% |
| **Kuramoto-Sivashinsky** | 弱非线性 | 0.0021 → 0.0018 | — | ↓14.3% |
| | 强非线性 | 0.0200 → 0.0197 | — | ↓1.5% |

> ✅ 表明模型对物理参数偏移具有更强鲁棒性。

#### （3）合成到真实迁移（Figure 7）
- 在 **ScalarFlow**（真实烟雾）上测试 3D Navier-Stokes 模型：
  - Baseline: nRMSE = 0.250
  - Ours: nRMSE = **0.213**（↓14.8%）
- 视觉质量明显更接近真实流动结构。

#### （4）长期 rollout 一致性（Figure 6）
- 自回归 rollout 至第 5 步时，Ours 的误差增长更缓慢。
- 尤其在复杂系统（如 3D NS）中，Baseline 出现明显失真，而 Ours 保持物理合理形态。

---

### 消融实验结果

#### （1）基本形式的选择至关重要（Figure 14）
- 若选择错误的基本项（如只保留反应项而非扩散项），性能反而下降：
  - 在 2D DR 中，“reaction-only” 导致 nRMSE **上升高达 64%**。
  - “diffusion-only” 则持续带来增益（↓11%-24%）。
- 结论：必须保留主导物理机制的基础项。

#### （2）辅助损失权重敏感性分析（Table 8）
- 辅助任务损失权重设为 0.5 / 0.7 / 1.0，性能差异极小（最大偏差 < 0.0063）。
- 表明方法对超参不敏感，鲁棒性强。

#### （3）与其他技术正交性验证（Figure 17）
- 将本文方法与 **Lie symmetry augmentation** 结合：
  - 单独使用 Lie 增强效果有限。
  - 本文方法本身已大幅超越。
  - 两者结合可进一步小幅提升，证明二者互补。

#### （4）扩展至大规模 Foundation Model（Figure 16）
- 在 **DPOT**（Hao et al., 2024）这一大型 SciML 基础模型上应用本方法：
  - 仍能带来一致的数据效率提升。
  - 表明即使经过大规模多物理场预训练，**显式引入基础物理知识仍有额外收益**。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **基础物理知识是有效的归纳偏置**：  
   分解出的基本 PDE 形式虽然简单，但蕴含丰富物理机制，有助于模型学习本质动态。

2. ✅ **联合训练显著提升综合性能**：  
   显式学习基本形式可在不增加推理成本的前提下，全面提升 **data efficiency, OOD generalization, long-term consistency**。

3. ✅ **方法具有普适性和可扩展性**：  
   不依赖特定 NO 架构（FNO / Transformer 均有效），适用于多种 PDE 类型（1D/2D/3D），且可与现有技术（如 Lie augmentation）正交结合。

4. ✅ **基础项的选择应有原则**：  
   应保留主导物理机制、易于高效仿真的项；错误选择会损害性能。

---

### 方法的局限性

1. **依赖领域知识进行分解**：  
   当前需人工判断哪些项构成“基础物理”，自动化识别尚待研究。

2. **并非所有 PDE 都易分解**：  
   某些高度耦合或非局部系统可能难以找到有意义的基本形式。

3. **基本形式未必总是便宜**：  
   某些情况下简化后仍计算昂贵（如高维谱方法），限制数据增益。

4. **目前集中在确定性 PDE**：  
   对随机 PDE 或不确定性量化场景尚未验证。

---

### 未来工作方向

1. **自动化基本形式发现**：  
   探索基于符号回归、稀疏识别（SINDy）或注意力机制自动提取主导项。

2. **动态课程学习策略**：  
   设计渐进式训练流程，先学基本形式，再过渡到完整 PDE。

3. **扩展至其他科学领域**：  
   如量子力学、电磁学、弹性力学等，探索通用性。

4. **结合物理约束与表示学习**：  
   将本文方法与 PINNs、能量守恒模块等结合，构建更强的 hybrid model。

5. **应用于实时仿真与控制**：  
   利用其高效性部署于机器人、气候模拟等实际系统中。

---

> 🔚 **总结一句话**：  
> 本文揭示了一个重要范式转变——**与其盲目扩大数据和模型规模，不如回归物理本质，让神经算子学会“最简单的物理”**。这种“返璞归真”的设计带来了显著的数据效率与泛化能力飞跃，为构建可靠、高效的科学 AI 提供了一条新路径。

</details>

---

### 15. [BindCLIP: A Unified Contrastive-Generative Representation Learning Framework for Virtual Screening](https://arxiv.org/abs/2602.15236)

**Authors**: Anjie Qiao, Zhen Wang, Yaliang Li, Jiahua Rao, Yuedong Yang  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.15236v1  

#### Abstract
Virtual screening aims to efficiently identify active ligands from massive chemical libraries for a given target pocket. Recent CLIP-style models such as DrugCLIP enable scalable virtual screening by embedding pockets and ligands into a shared space. However, our analyses indicate that such represen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BindCLIP: A Unified Contrastive-Generative Representation Learning Framework for Virtual Screening

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **CLIP-style** 的虚拟筛选模型（如 **DrugCLIP**）虽然实现了高效的检索式虚拟筛选，但仍存在两个关键缺陷：
- **Interaction Insensitivity（交互不敏感）**：模型对配体-口袋之间细微的结合相互作用不敏感，难以捕捉原子级别的空间和功能团级的结合模式。
- **Shortcut Reliance（捷径依赖）**：模型倾向于依赖训练数据中的粗粒度物理化学相似性（如分子量、极性表面积等）作为“捷径”进行打分，而非真正学习结合兼容性，导致在分布外（OOD）场景下泛化能力差。

### 提出的新方法与新思路
作者提出 **BindCLIP**，一个统一的 **contrastive-generative 表示学习框架**，通过以下创新机制解决上述问题：

#### （1）结合生成式监督的对比学习
- 引入 **pocket-conditioned diffusion objective**（口袋条件扩散目标），用于生成配体的结合构象（binding pose）。
- 扩散模型仅在训练阶段使用，为 pocket 和 ligand 编码器提供细粒度的几何感知监督信号，使表示空间更关注真实的结合相互作用。
- 将扩散建模视为一种多尺度监督机制，从生成任务中提炼出对识别任务有益的知识。

#### （2）硬负样本增强 + 配体锚定正则化
- 引入 **hard-negative augmentation**：从化学空间相近但非活性的分子中挖掘难负样本，迫使模型区分细微差异。
- 设计 **ligand-ligand anchoring regularizer**：防止硬负样本因持续排斥梯度而坍缩到远离正样本的空间区域，维持合理的嵌入几何结构。

### 相比现有方法的优势
- **更强的泛化能力**：在 OOD 和 FEP+ 等更具挑战性的基准上显著优于 DrugCLIP 等基线。
- **更高的细粒度分辨能力**：能更好地区分类似物（analogue）之间的微小结合差异，提升 activity-cliff 场景下的排序准确性。
- **保持高效推理**：尽管引入了扩散训练目标，但推理时仍只需编码 + ANN 检索，无额外开销，适合大规模虚拟筛选。

---

## 2. 核心实验方法和设置

### 数据集
- **训练集**：`PDBBind 2019` + `HomoAug` 扩增，共 66,164 个蛋白-配体复合物。
- **验证集**：`CASF-2016`，用于超参选择。
- **测试基准**：
  - `DUD-E`：102 个靶点，含 property-matched decoys。
  - `LIT-PCBA`：15 个靶点，基于 PubChem 实验数据构建，减少化学偏见，更具挑战性。
  - **OOD Benchmark**：从 `MF-PCBA` 构建，排除与训练集序列同源性 >30% 的蛋白，严格测试分布外泛化。
  - `FEP+ Benchmark`（4-target subset）：CDK2, TYK2, JNK1, P38，用于评估对类似物自由能变化（ΔΔG）的排序能力。

### 实验设置与评估指标
- **训练设置**：使用 UniMol 作为 backbone，Adam 优化器，5 次随机种子实验取均值 ± 标准差。
- **评估指标**：
  - **AUROC**：整体分类性能。
  - **BEDROC**（α=80.5）：强调早期富集能力，适用于虚拟筛选。
  - **Enrichment Factor (EF)**：EF0.5%, EF1%, EF5%，衡量前 x% 中活性化合物的富集程度。
  - **Hits@K**：在 OOD 设置下报告 Hits@500 和 Hits@1000。
  - **FEP+ 排序任务**：pairwise accuracy 和 Kendall tau 相关性，评估与实验 ΔΔG 顺序的一致性。

### 基线方法对比
- **Docking-based**：AutoDock Vina, Glide-SP, Surflex, Gnina
- **Learning-based**：
  - 回归/分类类：DeepDTA, RFscore, Pafnucy, OnionNet
  - 检索类：DrugCLIP, DrugHash, Planet, BigBind
- 主要对比对象为 **DrugCLIP**，因其是当前最先进的 CLIP-style 虚拟筛选模型。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1 & 2）

| Model       | LIT-PCBA AUROC ↑ | BEDROC ↑ | DUD-E AUROC ↑ | BEDROC ↑ |
|-------------|------------------|----------|---------------|----------|
| DrugCLIP    | 55.45 ± 1.31     | 6.41 ± 0.87 | 79.29 ± 1.59   | 47.53 ± 2.73 |
| **BindCLIP** | **59.15 ± 1.26** | **7.88 ± 0.25** | **80.14 ± 1.56** | **49.73 ± 1.75** |

> BindCLIP 在 LIT-PCBA 上实现 **+6.7% AUROC** 和 **+24.5% 平均早期富集提升**，优势显著。

### OOD 泛化表现（Fig. 4）
- **BEDROC**：+62%
- **EF0.5%**：+115%
- **Hits@500**：+114%
- **Hits@1000**：+62%

> 显示 BindCLIP 在真实、不平衡、分布外场景下具有更强的早期识别能力和实际检索价值。

### FEP+ 类似物排序（Fig. 5）
| Metric           | DrugCLIP | BindCLIP | Improvement |
|------------------|----------|----------|-----------|
| Pairwise Accuracy | 56.4%    | **65.7%** | +16%      |
| Kendall Tau       | 0.14     | **0.31**  | +121%     |

> 表明 BindCLIP 能更准确地捕捉细微结构修改带来的结合能变化，在药物优化中更具指导意义。

### 消融实验结果（Table 3, LIT-PCBA）

| 设置                     | AUROC   | BEDROC | EF0.5% |
|--------------------------|---------|--------|--------|
| Baseline (DrugCLIP)      | 55.45   | 6.41   | 8.24   |
| + Diffusion              | 56.71   | 6.85   | 9.11   |
| + Diffusion + Random Neg | 58.91   | 7.21   | 9.17   |
| + Diffusion + Hard Neg   | **59.15** | **7.88** | **9.84** |

> 结果表明：
> - 扩散监督带来稳定增益；
> - 随机负样本增益有限；
> - **硬负样本增强是最大贡献来源**，尤其提升早期富集能力。

---

## 4. 关键结论和发现

### 主要发现
1. **纯对比学习不足以捕获细粒度结合信号**：CLIP-style 模型易受粗粒度捷径影响，缺乏对局部相互作用的敏感性。
2. **生成式任务可作为有效监督信号**：将 binding pose generation 作为辅助任务，能引导表示学习关注空间和功能层面的真实结合特征。
3. **高质量负样本设计至关重要**：硬负样本迫使模型超越物理化学相似性，学习更精细的判别边界。
4. **需防止负样本坍缩**：提出的 ligand-ligand anchoring regularizer 成功缓解了硬负样本训练中的表示坍缩问题。

### 方法的局限性
- **训练成本增加**：由于引入扩散模型和硬负采样，训练时间和显存占用约为 DrugCLIP 的 2 倍（见 Appendix A.8）。
- **依赖高质量 3D 结构**：需要解析的蛋白-配体复合物结构用于扩散训练，限制了在无结构靶点上的应用。
- **硬负样本可能引入噪声**：尽管使用 Vina 过滤，仍可能存在假阴性风险。

### 未来工作方向
- 将框架扩展至 **de novo 分子生成** 或 **lead optimization** 任务。
- 探索 **zero-shot binding prediction** 在全新靶点上的潜力。
- 结合 **protein sequence-only** 输入，降低对 3D 结构的依赖。
- 进一步优化 hard-negative mining pipeline，提升负样本质量与效率。

---

> ✅ **一句话总结**：  
> BindCLIP 通过融合 **pocket-conditioned diffusion** 的细粒度生成监督与 **anchored hard-negative contrastive learning**，构建了更贴近真实结合机制的表示空间，在保持高效检索的同时显著提升了虚拟筛选的泛化性与细粒度分辨能力，推动了虚拟筛选向真实世界应用迈进。

</details>

---

### 16. [Neural Network-Based Parameter Estimation of a Labour Market Agent-Based Model](https://arxiv.org/abs/2602.15572)

**Authors**: M Lopes Alves, Joel Dyer, Doyne Farmer, Michael Wooldridge, Anisoara Calinescu  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.15572v1  

#### Abstract
Agent-based modelling (ABM) is a widespread approach to simulate complex systems. Advancements in computational processing and storage have facilitated the adoption of ABMs across many fields; however, ABMs face challenges that limit their use as decision-support tools. A significant issue is parame...

---

### 17. [TAROT: Test-driven and Capability-adaptive Curriculum Reinforcement Fine-tuning for Code Generation with Large Language Models](https://arxiv.org/abs/2602.15449)

**Authors**: Chansung Park, Juyong Jiang, Fan Wang, Sayak Paul, Jiasi Shen, Jing Tang, Jianguo Li  
**Category**: cs.CL  
**Published**: 2026-02-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.15449v1  

#### Abstract
Large Language Models (LLMs) are changing the coding paradigm, known as vibe coding, yet synthesizing algorithmically sophisticated and robust code still remains a critical challenge. Incentivizing the deep reasoning capabilities of LLMs is essential to overcoming this hurdle. Reinforcement Fine-Tun...

---

### 18. [Fractional-Order Federated Learning](https://arxiv.org/abs/2602.15380)

**Authors**: Mohammad Partohaghighi, Roummel Marcia, YangQuan Chen  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.15380v1  

#### Abstract
Federated learning (FL) allows remote clients to train a global model collaboratively while protecting client privacy. Despite its privacy-preserving benefits, FL has significant drawbacks, including slow convergence, high communication cost, and non-independent-and-identically-distributed (non-IID)...

---

### 19. [The Stationarity Bias: Stratified Stress-Testing for Time-Series Imputation in Regulated Dynamical Systems](https://arxiv.org/abs/2602.15637)

**Authors**: Amirreza Dolatpour Fathkouhi, Alireza Namazi, Heman Shakeri  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.15637v1  

#### Abstract
Time-series imputation benchmarks employ uniform random masking and shape-agnostic metrics (MSE, RMSE), implicitly weighting evaluation by regime prevalence. In systems with a dominant attractor -- homeostatic physiology, nominal industrial operation, stable network traffic -- this creates a systema...

---

### 20. [Attention-gated U-Net model for semantic segmentation of brain tumors and feature extraction for survival prognosis](https://arxiv.org/abs/2602.15067)

**Authors**: Rut Pate, Snehal Rajput, Mehul S. Raval, Rupal A. Kapdi, Mohendra Roy  
**Category**: cs.AI  
**Published**: 2026-02-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.15067v1  

#### Abstract
Gliomas, among the most common primary brain tumors, vary widely in aggressiveness, prognosis, and histology, making treatment challenging due to complex and time-intensive surgical interventions. This study presents an Attention-Gated Recurrent Residual U-Net (R2U-Net) based Triplanar (2.5D) model ...

---

### 21. [Mind the (DH) Gap! A Contrast in Risky Choices Between Reasoning and Conversational LLMs](https://arxiv.org/abs/2602.15173)

**Authors**: Luise Ge, Yongyan Zhang, Yevgeniy Vorobeychik  
**Category**: cs.AI  
**Published**: 2026-02-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.15173v1  

#### Abstract
The use of large language models either as decision support systems, or in agentic workflows, is rapidly transforming the digital ecosystem. However, the understanding of LLM decision-making under uncertainty remains limited. We initiate a comparative study of LLM risky choices along two dimensions:...

---

### 22. [The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems](https://arxiv.org/abs/2602.15382)

**Authors**: Xiaoze Liu, Ruowang Zhang, Weichen Yu, Siheng Xiong, Liu He, Feijie Wu, Hoin Jung, Matt Fredrikson, Xiaoqian Wang, Jing Gao  
**Category**: cs.CL  
**Published**: 2026-02-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.15382v1  

#### Abstract
Multi-Agent Systems (MAS) powered by Large Language Models have unlocked advanced collaborative reasoning, yet they remain shackled by the inefficiency of discrete text communication, which imposes significant runtime overhead and information quantization loss. While latent state transfer offers a h...

---

### 23. [LLM-to-Speech: A Synthetic Data Pipeline for Training Dialectal Text-to-Speech Models](https://arxiv.org/abs/2602.15675)

**Authors**: Ahmed Khaled Khamis, Hesham Ali  
**Category**: cs.CL  
**Published**: 2026-02-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.15675v1  

#### Abstract
Despite the advances in neural text to speech (TTS), many Arabic dialectal varieties remain marginally addressed, with most resources concentrated on Modern Spoken Arabic (MSA) and Gulf dialects, leaving Egyptian Arabic -- the most widely understood Arabic dialect -- severely under-resourced. We add...

---

### 24. [Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields](https://arxiv.org/abs/2602.15155)

**Authors**: Tianyu Xiong, Skylar Wurster, Han-Wei Shen  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.15155v1  

#### Abstract
Implicit Neural Representations (INRs) have emerged as promising surrogates for large 3D scientific simulations due to their ability to continuously model spatial and conditional fields, yet they face a critical fidelity-speed dilemma: deep MLPs suffer from high inference cost, while efficient embed...

---

### 25. [Doubly Stochastic Mean-Shift Clustering](https://arxiv.org/abs/2602.15393)

**Authors**: Tom Trigano, Yann Sepulcre, Itshak Lapidot  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.15393v1  

#### Abstract
Standard Mean-Shift algorithms are notoriously sensitive to the bandwidth hyperparameter, particularly in data-scarce regimes where fixed-scale density estimation leads to fragmentation and spurious modes. In this paper, we propose Doubly Stochastic Mean-Shift (DSMS), a novel extension that introduc...

---

### 26. [DNN-Enabled Multi-User Beamforming for Throughput Maximization under Adjustable Fairness](https://arxiv.org/abs/2602.15617)

**Authors**: Kaifeng Lu, Markus Rupp, Stefan Schwarz  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.15617v1  

#### Abstract
Ensuring user fairness in wireless communications is a fundamental challenge, as balancing the trade-off between fairness and sum rate leads to a non-convex, multi-objective optimization whose complexity grows with network scale. To alleviate this conflict, we propose an optimization-based unsupervi...

---

### 27. [Continuous-Time Piecewise-Linear Recurrent Neural Networks](https://arxiv.org/abs/2602.15649)

**Authors**: Alena Br\"andle, Lukas Eisenmann, Florian G\"otz, Daniel Durstewitz  
**Category**: cs.LG  
**Published**: 2026-02-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.15649v1  

#### Abstract
In dynamical systems reconstruction (DSR) we aim to recover the dynamical system (DS) underlying observed time series. Specifically, we aim to learn a generative surrogate model which approximates the underlying, data-generating DS, and recreates its long-term properties (`climate statistics'). In s...

---

### 28. [Secure and Energy-Efficient Wireless Agentic AI Networks](https://arxiv.org/abs/2602.15212)

**Authors**: Yuanyan Song, Kezhi Wang, Xinmian Xu  
**Category**: cs.AI  
**Published**: 2026-02-18  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.15212v1  

#### Abstract
In this paper, we introduce a secure wireless agentic AI network comprising one supervisor AI agent and multiple other AI agents to provision quality of service (QoS) for users' reasoning tasks while ensuring confidentiality of private knowledge and reasoning outcomes. Specifically, the supervisor A...

---

### 29. [How Vision Becomes Language: A Layer-wise Information-Theoretic Analysis of Multimodal Reasoning](https://arxiv.org/abs/2602.15580)

**Authors**: Hongxuan Wu, Yukun Zhang, Xueqing Zhou  
**Category**: cs.AI  
**Published**: 2026-02-18  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.15580v1  

#### Abstract
When a multimodal Transformer answers a visual question, is the prediction driven by visual evidence, linguistic reasoning, or genuinely fused cross-modal computation -- and how does this structure evolve across layers? We address this question with a layer-wise framework based on Partial Informatio...

---

### 30. [Causal Effect Estimation with Latent Textual Treatments](https://arxiv.org/abs/2602.15730)

**Authors**: Omri Feldman, Amar Venugopal, Jann Spiess, Amir Feder  
**Category**: cs.CL  
**Published**: 2026-02-18  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.15730v1  

#### Abstract
Understanding the causal effects of text on downstream outcomes is a central task in many applications. Estimating such effects requires researchers to run controlled experiments that systematically vary textual features. While large language models (LLMs) hold promise for generating text, producing...

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

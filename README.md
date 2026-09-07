# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-07 10:46:58 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference](https://arxiv.org/abs/2609.05275)

**Authors**: Mostafa Elhoushi, Alex Pretko, Nolan Dey, Bin Claire Zhang, Gavia Gray, Gurpreet Gosal, Abdulrahman Mahmoud, Shane Bergsma, Joel Hestness  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2609.05275v1  

#### Abstract
Layer dropout (a.k.a. stochastic depth) has been shown to enable faster training, higher accuracy, and robustness to zero-shot layer pruning in both language and vision transformers. However, as models and datasets have scaled, dropout - particularly layer dropout - has largely disappeared from larg...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
尽管 **Layer Dropout**（也称 **Stochastic Depth**）在早期Transformer模型中被广泛用于正则化、加速训练并提升鲁棒性，但随着大语言模型（LLM）规模扩大，该技术在预训练阶段几乎被弃用。主流观点认为，在大规模、单轮训练（single-epoch）场景下，dropout会损害模型精度。

然而，此前缺乏系统性的研究来量化这一影响，更未探索如何通过优化配置使其在现代LLM训练中重新发挥价值。本文旨在回答一个核心问题：  
> **“在现代大规模LLM预训练中，是否应该使用Layer Dropout？如果可以，应如何配置以兼顾效率与精度？”**

### **提出的新方法与新思路**
本文首次对Layer Dropout在LLM预训练中的作用进行了**系统性、大规模的实证研究**，提出了以下关键创新：

- **统一的Layer Dropout优化框架**：系统地分析了**optimizer超参数**、**层间分布（distribution）** 和**时间调度（schedule）** 三者之间的交互关系，揭示了先前研究中性能下降多源于次优配置而非根本缺陷。
- **最优配置策略**：
  - **Increasing Layer Distribution (ILD)**：越深的层，dropout概率越高（从0线性增至最大值）。
  - **Decreasing Time Schedule (DTS)**：训练初期dropout率最高，随训练进程线性衰减至0。
  - **训练时缩放因子 $ r_{\text{train}} = 1/p $**：确保不同dropout率下的激活尺度稳定，实现超参数可迁移。
- **揭示Layer Dropout的双重价值**：
  - **训练效率**：减少高达25%的训练FLOPs而不损失精度，甚至在某些情况下超越密集模型。
  - **推理灵活性**：赋予模型“零样本弹性深度”（zero-shot elastic depth），支持多种无需微调的高效推理技术。

### **相比现有方法的优势**
| 维度 | 传统做法 | 本文方法 |
|------|--------|---------|
| **训练效率** | 不使用dropout或仅用于微调 | 利用结构稀疏性直接节省计算，FLOPs降低25% |
| **推理优化** | 需额外模块（如router、adapter）或微调 | 零样本支持early exit、layer skipping、self-speculative decoding |
| **架构侵入性** | 多数深度感知方法需修改架构或增加参数 | 无侵入，仅在预训练中引入随机跳层 |
| **可扩展性** | 小模型有效，大模型效果不明 | 在271M–8.2B参数范围内均验证有效 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 来自多样化自然语言文本和代码的语料库。
- 最大数据量达 **160B tokens**，典型设置为 **20 tokens-per-parameter (TPP)**，符合compute-optimal训练范式（Hoffmann et al., 2022）。

### **模型架构**
- **Decoder-only Transformers**，基于Celerity架构：
  - 使用 **ALiBi position embeddings**
  - **Squared ReLU** 激活函数
  - **Llama3 vocabulary**
- 模型规模覆盖：**271M, 503M, 906M, 1.8B, 3.9B, 8.2B** 参数。

### **评估指标**
| 类别 | 指标 |
|------|------|
| **训练性能** | Training FLOPs, Validation Loss, Training Time |
| **推理能力** | Early Exit Loss, Layer Skipping Loss, Self-Speculative Decoding Speedup |
| **下游任务** | BBH, PIQA, SIQA, HellaSwag, ARC, RACE等基准 |

### **基线方法对比**
- **Dense Baseline**：无任何Layer Dropout的标准训练。
- 多种Layer Dropout变体作为对照：
  - Uniform Distribution + Constant Schedule（常见默认）
  - Alternating Layer Distribution (ALD)
  - Increasing Time Schedule (ITS)

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **训练效率提升**
- 在相同训练FLOPs下，使用**ILD + DTS**配置的模型**验证损失更低**。
- 对于固定训练步数，可节省**最多25%训练FLOPs**同时保持相当或更优的验证损失。
- 在**3.9B模型**上，最大dropout率达0.8，FLOPs节省20%，验证损失仍优于基线。

#### ✅ **推理加速效果**
| 技术 | 最高速度提升 | 精度损失 |
|------|-------------|--------|
| **Self-Speculative Decoding** | **1.55×** | 可忽略 |
| **Early Exit** | 显著降低延迟 | 损失极小，尤其高dropout模型 |
| **Intermediate Layer Skipping** | 支持动态跳层 | 跨尺寸性能桥接（如906M → 503M水平） |

> 表格示例（Table 5节选）：
>
> | Model Size | Max Dropout | FLOPs Savings | Val. Loss | Spec. Decode Speedup |
> |------------|-------------|----------------|-----------|-----------------------|
> | 3.9B       | 0.8 (ILD+DTS) | 20%            | 1.745     | **1.54×**             |
> | 8.2B       | 0.99 (ILD+DTS)| **25%**        | 1.663     | **1.55×**             |

#### ✅ **消融实验结果**

##### 🔹 **Granularity 消融**
| 类型 | 效果 |
|------|------|
| **Sub-Layer Dropout**（独立drop attn/FFN） | 性能差于完整Layer Dropout |
| **Per-Batch Dropout**（整batch同mask） | 劣于 **Per-Sequence Dropout**（每序列独立mask） |

> **Finding**: 完整Transformer块 + 序列粒度dropout效果最佳。

##### 🔹 **Distribution 消融**
- **ILD > ALD > Uniform**（在多数规模下）
- ILD随模型增大优势更明显。

##### 🔹 **Schedule 消融**
- **DTS > Constant > ITS**
- DTS在所有规模下均表现最优，且能在**5% FLOPs节省下反超密集模型**。

> **关键发现**：`ILD + DTS` 是最佳组合。

---

## **4. 关键结论和发现**

### **主要结论**
1. ✅ **Layer Dropout不应被抛弃**：在正确配置下，它不仅能维持精度，还能**提升训练效率与推理灵活性**。
2. ✅ **最优配置是 `ILD + DTS`**：
   - 层间：越深越容易被跳过（ILD）
   - 时间上：训练初期噪声大，后期收敛到完整结构（DTS）
3. ✅ **平均dropout率预测零样本鲁棒性**：训练时的平均dropout率越高，模型对early exit和layer skipping的容忍度越强。
4. ✅ **大模型更具鲁棒性**：随着模型规模增大，其对高dropout率（如p_max=0.99）的容忍度显著增强。
5. ✅ **解锁零样本推理优化**：无需额外训练即可实现：
   - **Early Exit**
   - **Layer Skipping**
   - **Self-Speculative Decoding**（速度提升达1.55×）

### **方法的局限性**
- **超参数迁移性受限于极高dropout率**：当p_max > 0.8时，学习率、weight decay等需重新调优。
- **未探索其他granularity**：如attention head-level或neuron-level dropout的影响未知。
- **未与learned机制比较**：如Mixture-of-Depths (MoD) 或动态routing-based skipping。
- **跨架构泛化未验证**：如MoE模型或非Transformer架构。
- **缺乏P_max的scaling law**：尚无法预测给定模型/数据规模下的最大安全dropout率。

### **未来工作方向**
1. 推导**Layer Dropout的scaling laws**，预测最优p_max。
2. 结合**learned depth-aware机制**（如router + dropout先验）。
3. 扩展至**其他维度的“模型生长”**（model growing）：
   - 宽度（width）
   - 量化位宽（bit-width）
   - 稀疏模式（unstructured sparsity）
4. 开发**动态自适应dropout策略**，根据训练状态自动调整分布与调度。
5. 探索**分布式训练中的协同dropout**（如各设备不同层跳过）。

---

> **一句话总结**：  
> 本文证明，**Layer Dropout不是过时的技术，而是被误用的宝藏**。通过科学配置（ILD+DTS），它能成为连接高效训练与灵活推理的桥梁，是构建下一代弹性LLM的关键基石。

</details>

---

### 2. [Quantum-Assisted Memory-Efficient Training for Parameter-Intensive Wi-Fi-Based Human Activity Recognition](https://arxiv.org/abs/2609.04271)

**Authors**: To Truong An, Jie Zhang, Guolin Yin, Junqing Zhang, Yanjiao Li, Trung Q. Duong, Simon L. Cotton  
**Category**: cs.LG  
**Published**: 2026-09-07  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.04271v1  

#### Abstract
Wi-Fi-based human activity recognition (HAR) has become an important part of integrated sensing and communications, paving the way for a range of context-aware services. However, most existing Wi-Fi-based HAR systems rely on deep learning (DL) models that are computationally and memory intensive in ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **Wi-Fi-based Human Activity Recognition (HAR)** 系统中存在的两大效率瓶颈提出解决方案：
- **训练阶段内存消耗过高**：传统深度学习（DL）模型在训练时需要同时更新数百万参数，导致巨大的内存开销，尤其是在资源受限的边缘设备上难以实现端到端训练。
- **推理阶段模型冗余**：尽管已有压缩技术（如量化、剪枝）优化推理效率，但这些方法通常依赖于先完整训练大模型再进行后处理，训练过程本身依然低效。

### 提出的新方法：Q-MET 框架
作者提出了 **Quantum-assisted Memory-Efficient Training (Q-MET)**，一种结合量子计算与经典神经网络的混合框架，用于高效训练 HAR 模型。

#### 核心思想
- 利用 **Hybrid Quantum-Classical Neural Network** 作为参数生成器（Quantum Parameter Generator, QPG），间接生成目标 HAR 模型（如 ResNet-18）的权重。
- QPG 包含一个 **Parameterized Quantum Circuit (PQC)** 和一个轻量级 **Classical Mapping Network**，其可训练参数远少于直接训练整个 ResNet-18。
- 在训练过程中集成 **Structured Pruning**（基于 LAMP 的通道剪枝），在训练早期即去除冗余结构，实现“边训练边压缩”。

### 相比现有方法的优势
| 维度 | 传统方法 | Q-MET |
|------|--------|-------|
| **训练效率** | 高内存占用（需存储梯度、优化器状态等） | 内存减少 90–95%，仅训练少量 QPG 参数 |
| **推理效率** | 通常需额外剪枝/量化步骤 | 原生支持稀疏模型输出，推理模型轻量化 |
| **流程整合性** | “Train-then-Prune” 分离流程，增加开销 | “Prune-during-Training”，无需微调，更高效 |
| **适用场景** | 依赖云端训练 | 更适合边缘或资源受限环境部署 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **UT-HAR**：包含 7 类人类活动（如行走、跌倒、坐下等），样本维度为 `(1, 250, 90)`，共 3,977 训练样本 + 996 测试样本。
- **Widar3.0**：细粒度手势识别数据集，包含 22 类动作（如推拉、滑动、绘图等），样本维度为 `(22, 20, 20)`，共 34,926 训练样本 + 8,726 测试样本。
- 两数据集均基于 Intel 5300 NIC 提取的 **CSI (Channel State Information)** 数据。

### 实验设置
- **基础模型**：ResNet-18（调整初始块以适配不同输入形状）
- **硬件平台**：Dell Precision 工作站，配备 i7-14700 CPU、64GB RAM、RTX 4090 GPU
- **优化器**：Adam，初始学习率 0.001，batch size = 128，最多训练 500 轮，早停机制（30 轮无提升则停止）
- **信号预处理**：
  - 仅使用 CSI 幅值 `|Hk|` 作为输入（相位易受硬件偏移影响）
  - 不同数据集采用各自标准归一化方式（UT-HAR：min-max；Widar3.0：标准化）

### 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy** | 分类准确率，衡量模型性能 |
| **Trainable Parameters** | 可训练参数数量，反映训练内存需求 |
| **Sparsity (%)** | 剪枝后被移除参数的比例，反映模型紧凑性 |
| **Parameter Efficiency Gain (ΔC%)** | 相比传统训练减少的参数比例 |
| **Training Time per Epoch (Te)** | 单轮训练耗时，评估计算开销 |
| **Model-State Memory** | 存储参数、梯度、优化器状态所需内存（FP32 下约为 `16 × 参数数` 字节） |

### 基线方法对比
- **Classical ResNet-18**：标准反向传播训练的完整模型
- **Lightweight ResNet-18**：通过宽度乘子（width multiplier=0.22）缩小的轻量版 ResNet-18
- **Static Hypernetwork**：纯经典的超网络，用于生成 ResNet-18 参数，参数量与 Q-MET 对齐
- **Train-then-Prune (TTP)** 管道：
  - 先用 QT 训练完整模型（如 QT-7）
  - 再应用 LAMP 剪枝，并测试是否微调（TTP₀, TTP₁₀, TTP₅₀）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 参数效率（训练阶段）
| 数据集 | 方法 | 可训练参数 | 参数减少 | 准确率 |
|-------|------|------------|----------|--------|
| UT-HAR | Classical ResNet-18 | ~11.6M | — | 98.08% |
|        | **Q-MET (QT-7)** | **~547k** | **↓95.3%** | **99.08%** |
| Widar3.0 | Classical ResNet-18 | ~11.2M | — | 71.29% |
|          | **Q-MET (QT-6)** | **~1.0M** | **↓90.6%** | **70.61%** |

> 💡 Q-MET 在 UT-HAR 上不仅将训练参数降低 **95%**，还实现了 **更高的分类精度**。

#### ✅ 推理效率（模型稀疏性）
| 剪枝比例 (p) | Sparsity (UT-HAR) | Sparsity (Widar3.0) | 性能影响 |
|-------------|-------------------|--------------------|---------|
| 0.6         | >70%              | >70%               | 几乎无损 |
| 0.8         | ~85%              | ~80%               | UT-HAR: 98.60% (> 原始模型) |
| 0.9         | ~90%              | ~90%               | Widar3.0: 65.73% (~原模型降 5.56%) |

> 🔹 Q-MET 支持高达 **85–90% 的结构化稀疏度**，且在中等剪枝下性能稳定甚至提升。

#### ⏱️ 训练时间开销
- Q-MET 相比传统训练有 **18–36% 的单轮训练时间增加**（见 Fig. 5），主要来自 PQC 模拟与映射网络前向计算。
- 但由于总参数极少，收敛速度仍较快（如 Q-MET 在 93 轮收敛 vs 基线 83 轮）。

### 与基线方法的对比结果

#### 🆚 Lightweight ResNet-18
- 尽管参数量相近（~529k vs ~547k），但 Q-MET 显著优于轻量模型：
  - UT-HAR: **+1.48%** 准确率
  - Widar3.0: **+6.16%** 准确率
- 表明性能增益来自 **Q-MET 的训练机制**，而非仅仅是模型变小。

#### 🆚 Static Hypernetwork
- 同样控制参数预算下，Q-MET 表现更优且更稳定：
  - UT-HAR: **+0.48%**
  - Widar3.0: **+5.26%**
- 超网络在 Widar3.0 上标准差达 ±7.39%，而 Q-MET 仅为 ±1.31%，说明 **量子辅助生成更具鲁棒性**。

#### 🆚 Train-then-Prune (TTP) 管道
| 剪枝比例 | 方法 | UT-HAR Acc | Widar3.0 Acc |
|--------|------|------------|-------------|
| 0.7    | Q-MET | 98.84%     | 66.72%      |
|        | TTP₅₀ | 98.80%     | 64.74%      |
| 0.9    | Q-MET | 97.16%     | 65.73%      |
|        | TTP₅₀ | 92.64%     | 62.52%      |

> ❗ TTP 在高剪枝比下表现明显劣于 Q-MET，即使经过 50 轮微调也无法完全恢复性能。证明 **“Prune-during-Training” 是更优策略**。

### 消融实验结果
- **Warm-up Epochs 影响**（Fig. 7）：
  - 从 1 到 5 个 warm-up 轮次对最终精度影响极小。
  - 在极端剪枝（p=0.9）下，更多 warm-up 反而导致性能下降。
  - 结论：**1 个 warm-up epoch 足够且最优**。
- **Qubit 数量影响**（Table VII）：
  - 随着 qubit 数增加，参数压缩率上升，但在 Nq ≥ 6 后收益递减。
  - Nq=7（QT-7）在 UT-HAR 上达到最佳权衡点（95.3% 参数减少 + 最高准确率）。

---

## 4. 关键结论和发现

### 主要发现
1. **首次实现训练与推理双阶段内存优化**：
   - Q-MET 是首个同时解决 HAR 系统中 **训练内存爆炸** 与 **推理模型冗余** 的统一框架。
2. **量子辅助生成显著提升参数效率**：
   - 仅用 **6–7 个 qubit** 即可实现超过 **90% 的训练参数削减**，且不牺牲甚至提升精度。
3. **结构化剪枝内嵌训练流程更高效**：
   - “Prune-during-Training” 比 “Train-then-Prune” 更有效，避免了性能断崖式下降，无需额外微调。
4. **Q-MET 具备强泛化能力**：
   - 在简单任务（UT-HAR）和复杂任务（Widar3.0）上均表现出色，尤其在挑战性高的多类别识别中优势明显。

### 方法的局限性
- **依赖量子电路模拟**：当前实验基于经典计算机模拟 PQC，尚未在真实量子硬件上运行，存在模拟开销。
- **训练时间略有增加**：由于引入 PQC 和映射网络，单轮训练时间比传统方法长 18–36%。
- **对极度稀疏场景敏感**：当剪枝比例超过 0.9 时，模型容量受限，性能波动增大。
- **初始化敏感性**：warm-up 阶段虽短，但仍需合理初始化以保证剪枝有效性。

### 未来工作方向
1. **真实量子硬件部署**：将 Q-MET 迁移到实际量子处理器（如 IBM Quantum 或 IonQ）上验证其加速潜力。
2. **与其他压缩技术融合**：探索 Q-MET 与 **Quantization** 或 **Hybrid Pruning + Quantization** 结合，进一步降低存储与计算成本。
3. **隐私保护扩展**：结合 **Federated Learning**，实现在不上传原始 CSI 数据的前提下进行分布式 Q-MET 训练。
4. **增强对抗鲁棒性**：研究 Q-MET 模型对信道扰动、对抗攻击的防御能力。
5. **通用化架构设计**：将 Q-MET 扩展至其他 DL 架构（如 ViT、LSTM）和其他无线感知任务（如定位、呼吸监测）。

---

> ✅ **总结一句话**：  
> Q-MET 成功利用 **量子辅助参数生成 + 内嵌结构化剪枝** 的协同机制，在保持甚至超越经典性能的同时，实现了 **90–95% 的训练参数压缩** 和 **75–85% 的推理模型稀疏度**，为资源受限环境下的 Wi-Fi HAR 系统提供了全新的高效训练范式。

</details>

---

### 3. [Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference](https://arxiv.org/abs/2609.04895)

**Authors**: Zhenhe Wu, Yaping Jin, Qinghua Xing, Hang Zhou, Wei He, Xianjie Wu, Xianfu Cheng, Jian Yang, Hanting Chen  
**Category**: cs.CL  
**Published**: 2026-09-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.04895v1  

#### Abstract
Mixture-of-Experts (MoE) models activate only a small subset of experts per token, but the full expert set often exceeds GPU memory, causing repeated weight transfers during decoding. We formulate expert-cache management as a model-side algorithmic problem and propose a cache-aware post-training fra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Mixture-of-Experts (MoE) 模型在推理时仅激活少量专家，但由于专家集合过大，常超出 GPU 内存容量，导致在解码阶段频繁从主机内存加载专家权重（**expert-weight transfer**），严重影响推理效率。现有缓存策略（如 LRU、LFU）多为启发式或基于运行时预测的预取机制，无法直接优化模型层面的缓存驻留策略。

本文将 **expert-cache management** 形式化为一个**模型侧的算法问题**，提出通过**后训练（post-training）联合优化 MoE 主干网络与轻量级辅助缓存路由器**的方法，在不改变推理时 Top-K 路由规则的前提下，提升缓存命中率并减少权重传输。

---

### 提出的新方法与新思路

#### （1）Cache-Aware Post-Training 框架
- **联合适应（Joint Adaptation）**：同时优化 MoE 主干参数和两个轻量级辅助路由器（Temporal Router 和 Spatio Router），以最小化缓存损失（cache-coverage loss）。
- **保留原生路由语义**：推理时仍使用原始 MoE 路由器进行 Top-K 专家选择，辅助路由器仅用于管理缓存驻留，不替代执行决策。

#### （2）两种部署模式
| 模式 | 名称 | 功能 |
|------|------|------|
| 更新仅模式（Update-only） | **Temporal Router** | 在每层访问后，基于当前隐藏状态预测下个 token 同层可能复用的专家，并更新缓存。**无主动预取（no proactive loading）**。 |
| 完整模式 | **Spatio-Temporal Router** | 在 Temporal Router 基础上，增加 **Spatio Router**：利用因果前驱 token 的隐藏状态，在目标层访问前对缓存进行精细化调整（pre-access refinement），实现有限范围内的主动预取。 |

> ✅ **关键设计**：使用 soft Top-B 成员函数建模缓存优先级，结合 equal-weight addition 融合多个分布，无需额外融合参数。

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **算法层面** | 将缓存管理从系统调度问题转化为可学习的模型后训练任务，端到端优化缓存驻留策略。 |
| **效率** | 显著提升缓存命中率，大幅降低 expert-weight traffic（最多减少 53.3%）。 |
| **参数开销极低** | 仅引入最多 **0.083%** 的额外 inference-time 参数（如 Qwen3 上 +25.2M）。 |
| **灵活性** | 支持两种操作模式：Temporal Router 适用于低流量场景；Spatio-Temporal Router 可权衡预取成本换取更高命中率。 |
| **兼容性** | 不依赖特定硬件或调度器，适用于通用 MoE 推理系统。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GSM8K**：小学数学应用题，需多步算术推理（7.5K 训练 / 1.3K 测试）
- **MATH**：竞赛级数学问题，涵盖代数、几何等（7.5K / 5K）
- **CommonsenseQA**：基于 ConceptNet 的常识问答（9.7K / 1.1K 验证）

> 所有任务均使用 greedy decoding + KV cache，生成长度受限。

---

### 实验设置
| 项目 | 设置 |
|------|------|
| **Backbone Models** | - **Qwen3-30B-A3B-Instruct**：48 层，128 专家/层，top-8 路由<br>- **GPT-OSS-20B**：24 层，32 专家/层，top-4 路由 |
| **Cache Capacity (B)** | Qwen3: 20, GPT-OSS: 8 |
| **Refinement Budget (R)** | Spatio-Temporal Router 中允许尝试替换的最大候选数，设为 cache 容量的 75%（即 R=15 或 6） |
| **Training Setup** | bfloat16, AdamW, cosine decay, 4 epochs, per-device batch size 1, gradient accumulation 8 steps |

---

### 评估指标
| 指标 | 定义 | 说明 |
|------|------|------|
| **Accuracy (Acc.)** | Exact Match 或符号匹配准确率 | 衡量任务性能 |
| **Hard Hit Rate (Hit)** | 访问时已在缓存中的专家比例 | 不含共享专家，排除 prefill 阶段 |
| **Adjusted Hit Rate (Adj. Hit)** | $1 - \frac{D + P}{A}$，其中 D=demand miss, P=proactive load, A=total access | 对主动预取也计费，反映真实代价 |
| **Load/token (MB)** | $\frac{(D + P) \times S_{\text{exp}}}{T}$，$S_{\text{exp}}$ 为单个专家大小 | 主要算法成本指标，衡量每 token 的专家传输量 |

> ⚠️ 所有方法统一计费标准：**每次 demand miss 或 proactive insertion 均视为一次完整专家传输**

---

### 基线方法对比
| 类型 | 方法 | 特点 |
|------|------|------|
| **Cache Replacement** | LRU, LFU, LRFU | 基于访问频率/时间的经典策略，运行在 LM-only backbone 上 |
| **Prefetching Baselines** | - **Least-Stale (SpecMD)**：基于陈旧性判断<br>- **ProMoE**：用中间隐藏态预测并主动拉取<br>- **FineMoE**：细粒度访问模式 + prompt 信号<br>- **Temporally Extended MoE**：持久化专家集选项机制 | 均采用其原文推荐配置，部分需 trace 数据训练 |

> ✅ 本文所有方法均在同一 post-training pipeline 下实现公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 **Qwen3** 上的表现（最优值加粗，次优下划线）
| 方法 | GSM8K Acc. | Hit ↑ | Adj. Hit ↑ | Load ↓ (MB/token) |
|------|------------|--------|-------------|------------------|
| MoE/LRU | 85.44 | 61.19 | — | 1407 |
| Temporal Router | 85.44 | **73.13** | — | **974** |
| ProMoE | 85.44 | 89.38 | 64.38 | 1792 |
| **Spatio-Temporal Router** | **83.40** | 90.62 | **69.03** | **1474** |

> 🔺 相比最强 prefetching baseline（ProMoE）：
- **Adjusted Hit 提升 1.15~18.03 pts**
- **Load 减少 4.6% ~ 53.3%**
- 参数仅为 ProMoE 的 **~26%**（25.2M vs 96.0M）

#### 在 **GPT-OSS** 上表现
- 结果更具任务依赖性：
  - **CommonsenseQA** 上 Spatio-Temporal Router 全面领先
  - **GSM8K** 上虽 Load 较高，但取得最高 Accuracy（64.52）
- 总体仍优于多数 baseline，尤其在 adjusted efficiency 方面

---

### 消融实验结果（Table 2 & Table 3）

#### （1）Adaptation Scope 消融
| 设置 | Hard Hit | Load | Acc. |
|------|--------|------|------|
| LM-only (sw=0) | 61.19 | 1407 | 85.44 |
| Auxiliary-only | 63.01 | 1341 | 85.44 |
| **Joint Post-Training** | **73.13** | **974** | 85.44 |

> ❗ **结论**：仅训练辅助路由器效果有限；**联合优化主干与路由器才能显著提升缓存效率**

#### （2）Router Composition 消融
| 模式 | Hard Hit | Adj. Hit | Load |
|------|--------|----------|------|
| Temporal-only | 73.13 | — | 974 |
| Spatio-only | 93.04 | 66.94 | 1666 |
| **Spatio-Temporal (full)** | 90.62 | **69.03** | **1474** |

> ❗ **结论**：单独使用 Spatio Router 虽硬命中高，但因缺乏 temporal carry-over 更新，导致预取过多，整体效率更差。

#### （3）Cache Capacity $B$ 与 Refinement Budget $R$ 敏感性分析（Table 3）
- **增大 $B$** → 显著提升 hit rate，降低 Load（更多空间容纳专家）
- **增大 $R$** → 提升 raw hit，但过大会导致 over-prefetching，**adjusted hit 下降、Load 上升**
- 最佳平衡点出现在 $R = 0.75B$ 左右（如 B=20, R=15）

#### （4）Cache-Loss Weight $s_w$ 影响
- 增大 $s_w$ → 缓存命中率持续上升，Load 下降
- 但 Accuracy 随之下降（最大达 -10.84 pts）
- **中等 $s_w$（如 0.1–0.3）可在质量与效率间取得良好平衡**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Temporal Router 是高效且安全的更新策略**：无需主动预取即可显著提升缓存命中率，适用于对延迟敏感的场景。
2. ✅ **Spatio-Temporal Router 实现最优负载调整效率**：在 Qwen3 上全面优于 ProMoE 等先进预取方法，且参数开销极低。
3. ✅ **Joint Post-Training 至关重要**：仅训练辅助路由器收益有限，必须联合优化主干网络才能重塑有利于缓存的路由分布。
4. ✅ **Raw Hit Rate 不足以评价预取效果**：必须结合 **adjusted hit rate** 和 **load/token** 等考虑主动传输代价的指标。
5. ✅ **Routing Distribution 更加集中**：随着 $s_w$ 增大，少数专家被更频繁调用（见 Figure 2 & 3），表明模型学会了“聚焦”于高频专家以增强局部性。

---

### 方法的局限性
1. **非端到端服务栈优化**：仅关注算法层缓存管理，未建模实际带宽、批处理、计算-传输重叠等因素。
2. **需要全模型后训练**：尽管推理参数增量小，但训练阶段需更新整个 backbone，成本较高。
3. **评估集中在学术推理任务**：尚未验证在对话、搜索、代码生成等多样化 workload 下的表现。
4. **可能影响负载均衡**：路由集中化可能破坏专家间的负载平衡，影响分布式训练/推理效率。
5. **Baseline 可比性受限**：不同 baseline 设计初衷不同，统一比较存在假设偏差。

---

### 未来工作方向
- 扩展至 **hybrid dense-sparse 架构** 和更大规模模型。
- 探索 **LoRA-style adapter** 替代 full-parameter post-training，降低适配成本。
- 引入 **dynamic R/B 调整机制**，根据输入动态控制预取激进程度。
- 结合 **runtime scheduler** 进行联合优化，实现真正的软硬协同。
- 研究如何在提升缓存效率的同时维持良好的 **expert load balancing**。

--- 

> 📌 **一句话总结**：  
> 本文提出了一种**缓存感知的联合路由器适应框架**，通过后训练联合优化 MoE 主干与轻量辅助路由器，在几乎零推理参数开销下，显著提升了 MoE 模型的缓存效率与推理吞吐能力，为内存受限环境下的 MoE 部署提供了新的算法范式。

</details>

---

### 4. [Distill Globally, Adapt Locally: Reasoning Distillation and Product-Type Test-Time Training for Scalable Trade-Up Recommendation](https://arxiv.org/abs/2609.05363)

**Authors**: Siliang Liu, Mohammad Ghasemi, Sapan Patel, Amin Banitalebi-Dehkordi  
**Category**: cs.LG  
**Published**: 2026-09-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.05363v1  

#### Abstract
Trade-up recommendation identifies higher-quality alternatives that preserve a customer's purchase intent while offering upgraded benefits. Large language models (LLMs) can reason about such distinctions, but applying them directly to hundreds of millions of product pairs is operationally impractica...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Distill Globally, Adapt Locally: Reasoning Distillation and Product-Type Test-Time Training for Scalable Trade-Up Recommendation*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大规模电商场景中，**trade-up recommendation**（升级推荐）旨在为用户的基础商品（base product）推荐更高品质、功能更优的替代品，同时保持其原始购物意图不变。例如，从普通洗衣液推荐到更环保或清洁力更强的版本。

然而，直接使用 **Large Language Models (LLMs)** 进行成对商品推理在计算成本和延迟上不可行——面对数亿商品对，LLM 推理成本可达数百万美元且耗时数周。

### 🚀 提出的新方法
作者提出一个**两层框架**，实现高效、可扩展的 trade-up 识别：

#### **Level 1: 全局推理蒸馏（Reasoning Distillation）**
- 使用一个具备检索增强（retrieval-augmented）能力的 LLM 教师模型，生成每对商品的：
  - 四类关系标签（如“c 是 b 的 trade-up”）
  - 自然语言理由（natural-language rationale）
- 将这些**标签 + 理由**共同作为监督信号，通过**对齐损失（alignment loss）** 和 **对比蒸馏（contrastive distillation）** 蒸馏到一个轻量级、非生成式的嵌入对分类器（non-generative embedding-pair classifier）学生模型中。
- 学生模型仅需两个预计算的 768 维商品 embedding 即可预测，无需任何文本输入或 LLM 调用。

#### **Level 2: 产品类型测试时训练（Product-Type Test-Time Training, PT-TTT）**
- 在推理前，针对每个 **product type（PT）**（如“洗衣液”、“电池”），利用少量专家标注的支持集（support set）对全局学生模型进行微调。
- 使用 **LoRA adapter** 对分类头和推理投影层进行参数高效的适应（parameter-efficient adaptation）。
- 每个产品类型的 adapter 只需优化一次，即可用于该类别下所有商品对的打分，避免 per-query 优化。

### ⭐ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **效率** | 推理速度提升约 **5,000×**，成本降低约 **10,000×** vs. 直接 LLM 推理 |
| **可扩展性** | 支持 catalog-scale（数亿商品对）部署，无 serving-time LLM inference |
| **性能** | 轻量学生模型（15.5M 参数）超越更大模型和 LLM 教师本身（F1 更高） |
| **灵活性** | PT-TTT 允许不同品类采用不同的 trade-up 判定标准（如电池看续航，护肤品看成分） |

---

## 2. 核心实验方法和设置

### 📚 数据集
| 数据集 | 描述 |
|--------|------|
| **Expert-annotated corpus** | 17,200 商品对，覆盖 29 个 product types，人工标注四类关系：<br>1. 同层级相似<br>2. b 是 c 的 trade-up<br>3. c 是 b 的 trade-up（正类）<br>4. 不兼容/无法比较<br>其中 8,352 对作为 **golden benchmark** 用于最终评估 |
| **Silver supervision corpus** | 基于 Amazon-Walmart dataset 构建的 1,019,241 对商品，由 LLM 教师标注标签和 rationale，用于训练学生模型 |
| **Support set for PT-TTT** | 来自 golden benchmark 的专家标注样本，按 product type 分组，用于 test-time adaptation |

### 🧪 实验设置
- **学生模型架构**：
  - 浅层（Shallow）：15.5M 参数（含 rationale 投影）
  - 深层（Deep）：65.9M 参数
  - 输入：两个 768-D 商品 embedding（来自 BGE 模型）
- **训练目标**：
  - 主任务损失：加权交叉熵（四类）或 Focal BCE（二类）
  - 辅助损失：Rationale alignment（MSE）、Contrastive distillation（InfoNCE + Relational KL）
- **PT-TTT 设置**：
  - 冻结全局学生模型，在分类头和 alignment projection 上插入 LoRA（rank=8）
  - 每个 product type 使用 K ∈ {4,8,16,32} 的支持样本来优化 adapter
  - 适配后用于该类别所有候选对打分

### 📊 评估指标
- **AUC** 和 **Average Precision (AP)**：主要阈值无关指标
- **F1 / Precision / Recall**：基于验证集选择最优阈值后报告
- 所有结果均在 **8,352 对的 golden benchmark** 上评估
- 使用 **2,000 次 bootstrap** 计算 95% 置信区间

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **LLM teacher (no demo / RAG)** | 零样本或 5-shot 检索增强提示下的 LLM 输出 |
| **Label-only student** | 仅使用 LLM 标签训练的学生模型（无 rationale 监督） |
| **Binary vs. Four-class** | 是否保留细粒度关系结构 |
| **Pooled LoRA** | 使用全部支持集训练单一全局 adapter，用于控制变量 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Golden Benchmark, n=8,352）

| Model | Params | AUC (95% CI) | AP | F1 |
|-------|--------|------------------|------|-----|
| LLM teacher (RAG, 5 demo) | — | — | — | **0.749** |
| Label-only (4-class, shallow) | 14.3M | 0.912 [0.906, 0.918] | 0.916 | 0.836 |
| **+ Reason (4-class, shallow)** | **15.5M** | **0.924 [0.918, 0.929]** | **0.920** | **0.843** |
| + Reason (4-class, deep) | 65.9M | 0.911 [0.905, 0.917] | 0.902 | 0.832 |
| **PT-TTT (Label-only, K=32)** | — | **0.940** | **0.938** | **0.856** |
| **PT-TTT (+ Reason, K=32)** | — | **0.941** | **0.940** | **0.856** |

> ✅ **最佳模型**：浅层四类 + rationale 蒸馏 + PT-TTT（K=32），AUC 达 **0.941**

### 🔍 与基线对比的关键发现
- **学生模型优于教师**：尽管教师 LLM 具备强大语义理解能力，但在固定 prompt 下其 **recall 仅为 0.610**，而蒸馏后的学生模型达到 **0.858**，显著提升 F1（0.749 → 0.843）。
- **PT-TTT 显著提升性能**：从全局模型 AUC 0.924 提升至 **0.941（+0.017）**，AP 提升至 **0.940（+0.020）**。
- **效率极高**：在 100K 商品对上的推理实验显示，相比直接 LLM 推理，**速度快 5,000×，成本低 10,000×**。

### 🔬 消融实验结果
#### （1）Rationale 监督的有效性依赖标签粒度
| 模型 | AUC 提升（vs. label-only） |
|------|----------------------------|
| Binary + Reason | ❌ 无提升（0.911 → 0.911） |
| Four-class + Reason | ✅ 显著提升（0.912 → 0.924） |

> 💡 表明 rationale 监督只有在保留 teacher 的细粒度关系结构时才有效。

#### （2）PT-TTT 中 rationale 复用作用有限
- 在 PT-TTT 阶段，加入 rationale alignment 损失（`λ_reason > 0`）带来的增益极小（AUC 0.940 → 0.941）。
- 表明 Level-2 的提升主要来自 **product-type-specific adaptation**，而非 rationale 重用。

#### （3）PT-TTT 的增益来自类别内判别能力提升
- 控制实验表明：
  - **Pooled LoRA**（统一 adapter）使 AUC 提升至 0.929
  - **PT-TTT**（每类独立 adapter）进一步提升至 0.940
- **Macro PT-AUC**（各品类独立计算再平均）从 0.910 → 0.925，说明提升是真实的类别内判别增强，而非简单分数缩放。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **“Distill Globally, Adapt Locally” 是有效的设计范式**：
   - Level 1：通过 **rationale-guided representation distillation** 学习通用、高质量的 pair 表示
   - Level 2：通过 **PT-TTT** 实现局部决策边界定制化，适应不同品类的 trade-up 标准

2. **Rationale 监督的价值在于表示学习阶段**：
   - 在全局蒸馏中，rationale 显著提升学生模型性能（尤其配合四类标签）
   - 但在 test-time adaptation 中复用 rationale 效果不明显

3. **轻量模型可以超越大模型和 LLM 教师**：
   - 15.5M 参数的浅层学生模型在 F1 和 AUC 上均超过更大的深层模型和原始 LLM 教师

4. **PT-TTT 的增益主要来自类别特异性优化**：
   - 即使使用相同数量的专家标注，**per-PT adapter** 比 pooled adapter 性能更好（AUC 0.940 vs. 0.929）
   - 表明 **category-specific adaptation** 是关键

### ⚠️ 局限性
1. **泛化性未验证**：仅在 29 个已见 product types 上评估，未测试对新类别的迁移能力。
2. **依赖专家标注支持集**：PT-TTT 需要每个品类有少量人工标注数据，限制了完全自动化部署。
3. **未完全排除校准效应**：增益是否部分来自简单的类别级概率校准（calibration）尚不明确。
4. **非实体隔离划分**：golden benchmark 与训练集虽 pair-disjoint，但未做到 product- 或 brand-disjoint，可能存在泄露风险。
5. **LLM 教师未充分优化**：教师性能受限于固定的 prompt 和 demonstration 数量，可能低估其上限。

### 🔮 未来工作方向
- 研究 **unseen product type transfer** 能力
- 探索更简单的 **calibration-based baseline** 或 **product-type-conditioned model** 替代 PT-TTT
- 实现 **entity-disjoint evaluation** 以更严格评估泛化性
- 设计 **amortized adaptation** 方法，避免每个品类都进行梯度优化
- 将该框架推广至其他 directional product relations（如互补、兼容等）

---

> 📌 **一句话总结**：  
> 本文提出“先全局蒸馏、再本地适配”的两阶段框架，首次将 LLM 的复杂推理能力高效迁移到超大规模 trade-up 推荐任务中，在保持零 LLM serving 开销的同时，实现了优于 LLM 本身的性能，为电商场景下的语义理解落地提供了新范式。

</details>

---

### 5. [CUA-Universe: A Scalable and Dynamic Environment for Hybrid GUI+CLI Agents](https://arxiv.org/abs/2609.05374)

**Authors**: Haoting Shi, Wenhao Wang, Weicheng Fang, Yaozhong Liang, Tian Jin, Pengxiang Zhao, Guangyi Liu, Siheng Chen, Yanfeng Wang  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.05374v1  

#### Abstract
Computer-use agents have advanced on benchmarks like OSWorld and AndroidWorld, but still act mostly through the GUI, often producing inefficient trajectories. Real-world computer work is hybrid, combining visual-state inspection with precise, high-throughput command-line operations, so capable agent...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# CUA-Universe: A Scalable and Dynamic Environment for Hybrid GUI+CLI Agents  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Computer-Use Agents (CUAs)** 主要依赖 **GUI-only 交互**，导致执行路径冗长、效率低下且在复杂任务中表现脆弱。而真实世界中的计算机操作是 **多模态混合行为**：用户结合 **GUI 进行视觉感知与交互**，同时利用 **CLI 执行精确、批量的操作**。

然而，现有的研究面临两大瓶颈：
- **环境稀缺**：构建支持 GUI 和 CLI 共享状态的真实软件环境成本高、难以扩展；
- **跨模态协调能力弱**：现有代理要么缺乏视觉理解（CLI-native），要么效率低下（GUI-native）。

---

### 🚀 提出的新方法：CUA-Universe 框架

提出一个可扩展的 **“环境到数据”闭环流水线 CUA-Universe**，将真实桌面软件转化为 **共享状态的 Hybrid GUI+CLI 环境**，并自动生成训练数据。该框架由三个核心组件构成：

| 组件 | 功能 |
|------|------|
| **App-Forge** | 将桌面应用自动适配为可复现的 VM，并通过发现、封装或生成的方式暴露其 CLI 接口（如 `blender --python-expr`, `cvlc`, 或基于脚本 API 的 wrapper） |
| **Task-Weave** | 基于种子文件（seed files）合成可控难度的 Hybrid 任务，涵盖多种操作组合，确保任务具有现实性和可验证性 |
| **Path-Steer** | 在 rollout 阶段引导代理走向高效的 Hybrid 路径（例如用 CLI 处理批处理，GUI 处理布局相关操作），并收集高质量轨迹用于后训练 |

> 🔍 创新亮点：
> - **Agent-driven 构建**：使用 coding agent 自动完成安装、配置和工具构造，无需人工工程，实现跨 16 个真实桌面应用的规模化部署。
> - **动态任务生成**：每个环境成为持续的任务源，而非静态 benchmark。
> - **效率导向的轨迹采集**：通过模态先验（modality prior）指导更优的 GUI/CLI 切换策略。

---

### ⚖️ 相比现有方法的优势

| 方面 | CUA-Universe | 现有方法（如 OSWorld、AndroidWorld） |
|------|--------------|-------------------------------|
| **接口模式** | 支持 Hybrid GUI+CLI，共享应用状态 | 多为纯 GUI 或纯 CLI |
| **环境真实性** | 基于真实桌面软件（如 Blender, VS Code, Zotero） | 多为人造网页或简化界面 |
| **可扩展性** | 可扩展至 16+ 应用，自动化程度高 | 通常需大量手动标注与调试 |
| **训练信号丰富度** | 包含高效路径、跨模态协调、CLI 输出反馈等强监督信号 | 单一模态轨迹，无法体现“何时切换” |
| **训练数据质量** | 显式引导高效执行路径，减少无效点击/脚本 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与环境

- **CUA-Universe 支持的应用（共 16 个）**：
  - 来自 OSWorld 的 8 个：`Chrome`, `GIMP`, `LibreOffice Calc/Writter/Impress`, `Thunderbird`, `VLC`, `VS Code`
  - 新增的 8 个：`Blender`, `Draw.io`, `Zotero`, `Godot`, `QGIS`, `Kdenlive`, `OBS Studio`, `Audacity`

- **训练数据来源**：
  - 通过 **Path-Steer** 在上述环境中 rollout 得到 **4,923 个 episode**，约 **235K 步级记录**
  - 数据格式包含：截图、GUI/CLI 动作、CLI 返回码与输出、推理链、最终得分

- **评估基准**：
  - **CUA-Verse**：新提出的 hold-out benchmark，包含 160 个 Hybrid 任务（8 apps × 20 tasks），任务与训练集不重叠
  - **OSWorld**：标准 GUI benchmark，用于测试迁移能力
  - **OSWorld-MCP**：引入 MCP 工具调用的新版本，测试对未见过工具接口的泛化能力

---

### 🧪 实验设置与评估指标

#### 模型设置
- **基础模型**：`Qwen3.5-9B`
- **微调方式**：LoRA 微调（仅语言模型线性层可训练）
- **训练数据**：来自 Kimi K2.5 回收的高分轨迹（VLM judge ≥ 0.75）
- **硬件资源**：8×A100 GPU，训练约两天

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Score / SR (%)** | 成功率（Success Rate）或平均 VLM judge 分数 |
| **Steps ↓** | 平均决策步数（越少越好） |
| **Tokens ↓** | 输入+输出 token 数量（衡量成本） |
| **Step Gain / Token Gain ↑** | GUI+CLI 相较于 GUI-only 的节省倍数（如 2.35× 表示少用 2.35 倍步骤） |
| **TIR, ACS**（OSWorld-MCP） | Tool Invocation Rate, Average Completion Steps |

#### 基线对比模型
- Proprietary: `Kimi K2.5`, `Seed2.1 Pro`, `GPT-5.5`
- Open-source: `Qwen3.5-9B`, `EvoCUA-8B`, `Ours`（本文方法）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 模型 | CUA-Verse Score | OSWorld SR (%) | OSWorld-MCP Score |
|------|------------------|----------------|--------------------|
| Qwen3.5-9B (base) | 0.189 | 24.6 | 20.90 |
| **Ours (CUA-Universe)** | **0.582 (+3.08×)** | **40.2 (+16.8 pts)** | **29.51 (+7.84 pts)** |

> ✅ 总结提升幅度：
> - **成功率大幅提升**：在 CUA-Verse 上达到 open-source 最佳，接近闭源模型 Seed2.1 Pro
> - **效率显著优化**：相比 base 模型，**减少 37% 步骤，降低 60% token 消耗**
> - **跨 benchmark 泛化能力强**：在 OSWorld 和 OSWorld-MCP 上均有显著增益

---

### 🔍 详细对比结果

#### （1）在 CUA-Verse 上的表现（Table 1）

| 指标 | Ours vs Base |
|------|-------------|
| **Score** | 0.582 vs 0.189 → **+3.08×** |
| **Steps** | 35.2 vs 56.2 → **↓37%** |
| **Tokens/episode** | 255K vs 643K → **↓60%** |

> 💡 发现：我们的模型在音频/视频类应用（Audacity, OBS）上表现最强，在 3D 类（Blender）仍有提升空间。

#### （2）在 OSWorld 上的迁移效果（Table 1）

| 设置 | Ours (GUI-only) | Ours (GUI+CLI) | Gain |
|------|------------------|----------------|------|
| SR (%) | 23.4 | **40.2** | **+16.8 pts** |
| Steps | 39.6 | **28.6** | **↓2.35×** |
| Tokens/task | 325.7K | **286.5K** | **↓1.79×** |

> ✅ 特别发现：
> - 添加 CLI 接口后，**成功任务数增加 41 个**（远超其他模型的 +2~8）
> - 表明模型真正学会了如何利用 CLI 提升成功率和效率

#### （3）在 OSWorld-MCP 上的泛化能力（Table 2）

| 指标 | Ours vs Base |
|------|-------------|
| **Score** | 29.51 vs 20.90 → **+8.61 pts** |
| **Strict SR** | 18.85% vs 10.66% → **+8.19 pts** |
| **TIR** | 23.36% vs 10.66% → **+12.7 pts** |
| **ACS ↓** | 27.25 vs 37.3 → **↓27%** |
| **Tokens ↓** | 87.95M vs 125.6M → **↓30%** |

> 🎯 结论：即使从未见过 MCP 接口，也能有效调用工具，说明学到了**通用的跨模态协调能力**

---

### 🔬 消融实验结果

#### （1）Path-Steer 的作用（Table 3）

| 设置 | Accept Rate (≥0.75) | Mean Score | Avg Tokens |
|------|---------------------|------------|------------|
| w/ Path-Steer | **0.51** | **0.71** | **332K** |
| w/o Path-Steer | 0.44 | 0.63 | 385K |

> ✅ 结果表明：显式的 **modality-level guidance** 显著提升了轨迹质量和效率，证明了 Path-Steer 的有效性。

#### （2）跨域迁移能力验证（Table 5）

| 训练数据范围 | OSWorld SR (%) | 增益（vs base） |
|---------------|----------------|----------------|
| 仅 OOD apps（8个） | 27.4 | +3.2 pts |
| 完整 16 apps | **40.2** | **+16.0 pts** |

> 📌 发现：即使只在非 OSWorld 应用上训练，也能带来零样本增益，说明学到的是**可迁移的 Hybrid Interaction Skill**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Hybrid Orchestration 是关键能力**  
   单纯增加 CLI 使用频率并不足够（见 GPT-5.5 虽全 CLI 但仍失败），真正的优势在于 **根据任务需求动态选择 GUI 或 CLI**，并在两者之间传递状态。

2. **CUA-Universe 显著提升效率与成功率**  
   训练后的 9B 模型在多个 benchmark 上实现了 **接近甚至超越更大闭源模型的表现**，同时大幅降低成本。

3. **学习到的能力具备强泛化性**  
   不仅能在新任务上表现良好，还能迁移到未见过的工具接口（MCP），说明不是简单记忆命令，而是掌握了 **抽象的工具使用逻辑**。

4. **Path-Steer 提供高质量监督信号**  
   显式引导使代理更快收敛到高效路径，避免陷入 GUI 的“盲目点击”陷阱或 CLI 的“脆性脚本”。

---

### ⚠️ 局限性（Limitations）

1. **目前仅限单应用任务**  
   尚未支持跨应用程序的状态流转（如从浏览器下载文件 → 用 GIMP 编辑 → 用 Zotero 引用），这是未来重要方向。

2. **依赖可脚本化的开源软件**  
   对闭源或无 CLI 接口的商业软件支持有限，限制了生态覆盖面。

3. **仍采用 SFT，未引入 RL**  
   当前训练基于监督微调（SFT），受限于教师模型能力；若能结合 verifier 作为 reward signal 进行 RL，有望进一步突破上限。

4. **评估依赖 VLM Judge**  
   尽管已验证与人类标签高度一致（K=0.94），但仍存在轻微噪声风险。

---

### 🔮 未来工作方向

1. **扩展至 Cross-Application Workflows**  
   构建支持多应用协同的 Hybrid 环境，模拟真实办公流。

2. **引入 Reinforcement Learning**  
   利用内置 verifier 作为 reward signal，开展 RLVR 或 DPO 训练，超越 teacher model。

3. **支持更多平台与闭源软件**  
   探索逆向工程、OCR+操作模拟等方式，扩大适用范围至 Windows/macOS 商业软件。

4. **构建开放的 Hybrid Agent Benchmark 生态**  
   推动社区共建更多 Hybrid 任务与评测标准，促进通用 CUA 发展。

---

> 🏁 **总结一句话**：  
> **CUA-Universe 通过构建可扩展的 Hybrid GUI+CLI 环境与数据生成管道，首次系统性地教会小规模模型高效协调两种交互模式，在成功率、效率和泛化性上全面超越传统 GUI-only 方法，为下一代通用 Computer-Use Agent 提供了一条可行的技术路径。**

</details>

---

### 6. [Improving Progressive Compression with Adaptive Interpolation and Coefficient Decomposition](https://arxiv.org/abs/2609.04573)

**Authors**: Wenbo Li, Xuan Wu, Qian Gong, Pu Jiao, Jieyang Chen, Qing Liu, Norbert Podhorszki, Scott Klasky, Xin Liang  
**Category**: cs.DC  
**Published**: 2026-09-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.04573v1  

#### Abstract
Exascale simulations generate data far faster than it can be stored or analyzed, making efficient data reduction essential. Error-controlled lossy compression offers high compression ratios under user-specified error bounds, but the target tolerance must be fixed at compression time. Progressive com...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Improving Progressive Compression with Adaptive Interpolation and Coefficient Decomposition**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代 **exascale** 级科学模拟生成的数据量远超存储、传输和分析能力，导致对高效数据压缩技术的需求日益迫切。传统的 **error-controlled lossy compression** 方法（如 SZ、ZFP、MGARD）虽然能提供高压缩比并保证误差在用户指定范围内，但存在以下关键限制：

- **固定误差容忍度**：必须在压缩时确定目标误差，无法灵活适应不同下游分析任务的多样化精度需求。
- **缺乏渐进性**：一旦压缩完成，丢失的信息不可恢复，迫使用户选择保守的紧误差界以保留更多信息，牺牲了压缩效率。
- **现有 **(progressive compression) 虽然支持按需逐步重建，但仍存在不足：
  - 依赖固定的重构策略（如 PMGARD 的线性插值），未能充分利用分解系数间的空间相关性。
  - 检索效率不高，尤其是在面对不同目标（如误差界 vs. PSNR）时缺乏自适应能力。

### **提出的新方法与新思路**
本文提出了一种名为 **ProAICD **(Progressive compression with Adaptive Interpolation and Coefficient Decomposition) 的自适应渐进压缩框架，其核心创新点如下：

1. **自适应插值 **(Adaptive Interpolation)
   - 引入并优化了两种互补的多级插值方案：
     - **Per-level interpolation**：源自 PMGARD，使用高层级数据点进行多线性插值，误差传播小，适合 **error-bound mode**。
     - **Per-region interpolation**：源自 SZ3 和 IPComp，沿各维度独立插值，局部相关性强，适合 **PSNR mode**。
   - 通过在线调优机制，根据目标自动选择最优插值方案，实现“一个框架，两种模式”的高效适配。

2. **系数分解 **(Coefficient Decomposition)
   - 提出一种新颖的方法，对插值后产生的残差系数（coefficients）**再次进行空间相关性挖掘**。
   - 观察到这些系数本身也具有显著的空间平滑性和相关性，因此采用 **per-level multilinear interpolation** 对系数进行二次分解。
   - 该方法显著增加了近零系数的比例，提升了可压缩性，且仅引入可忽略的元数据开销。

3. **自适应渐进压缩工作流 **(Adaptive Workflow)
   - 设计了一个完整的自适应流程，包含：
     - **基于采样的在线调优**：通过采样 1% 数据快速决策最佳配置（插值方案、是否启用系数分解等）。
     - **针对性优化**：如最快方向插值（fastest direction interpolation）提升缓存效率，优化通用位平面编码（optimized generic bitplane encoding）减少分支开销。

### **相比现有方法的优势**
- **更高的检索效率**：在相同误差容忍度下，压缩比最高提升 **42.3%**；在相同 PSNR 下，压缩比最高提升 **92.5%**。
- **更优的可视化质量**：以最少的数据量实现最高的视觉保真度。
- **端到端性能提升**：在 512 GB 数据远程传输中，端到端时间最多加速 **1.26×**。
- **更强的灵活性与适应性**：支持 error-bound 和 PSNR 两种主流目标，并能自动选择最优路径。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验在五个真实世界的科学数据集上进行，涵盖多个领域：

| Dataset | Dimensions | Valid Fields | Type | Size |
| :--- | :--- | :--- | :--- | :--- |
| **CESM** | 26×1800×3600 | 33 | double | 41.42 GB |
| **Miranda** | 256×384×384 | 7 | double | 1.97 GB |
| **SCALE** | 98×1200×1200 | 11 | double | 11.57 GB |
| **S3D** | 500×500×500 | 9 | double | 8.38 GB |
| **JHTDB** | 4096×4096×4096 | 1 | double | 512 GB |

> 注：JHTDB 用于大规模并行传输实验，其余用于综合评估。

### **实验设置与评估指标**
- **平台**：Morgan Compute Cluster (MCC)，配备 AMD EPYC 处理器和 100 Gbps InfiniBand。
- **评估指标**：
  - **Bit-rate**：平均每个数据点的比特数，越低越好。
  - **Compression Ratio **(CR)：原始大小 / 检索大小，越高越好。
  - **PSNR **(Peak Signal-to-Noise Ratio)：衡量平均误差，越高越好。
  - **Rate-Distortion 曲线**：绘制 bit-rate vs. error bound 或 PSNR，曲线越靠左上方越优。
  - **重构时间 **(reconstruction time) 和 **重构时间 **(refactor time)：衡量吞吐性能。
  - **端到端传输时间**：包含检索 + 网络传输时间。

### **基线方法对比**
与三种最先进的渐进压缩方法进行比较：
- **PMGARD**：基于 MGARD 分解和位平面编码，支持误差控制。
- **IPComp**：基于三次样条插值和动态规划检索，效率较高。
- **SZ3-R**：迭代压缩残差，生成多级快照。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **压缩比提升**：
  - 在 **error-bound mode** 下，相比最优基线，压缩比最高提升 **42.3%**（S3D 数据集）。
  - 在 **PSNR mode** 下，相比最优基线，压缩比最高提升 **92.5%**（CESM 数据集）。
- **端到端传输加速**：
  - 在 JHTDB 上传输 512 GB 数据，ProAICD 最多实现 **1.26×** 的端到端速度提升。
- **可视化质量**：
  - 在 CESM 温度场可视化中，ProAICD 仅用 **bit-rate = 0.37** 即达到 PSNR = 87.2，而其他方法在更高 bit-rate 下仍存在明显失真。

### **与基线方法的对比结果**
| 方法 | Error-bound Mode | PSNR Mode | 重构时间 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **ProAICD **(EB) | ✅ 最优（多数情况） | ❌ 次优 | 中等 | 高压缩比需求，误差敏感应用 |
| **ProAICD **(PSNR) | ❌ 次优 | ✅ 显著最优 | 快 | 可视化、全局误差敏感应用 |
| **PMGARD** | 次优 | 次优 | 快 | 通用，但效率较低 |
| **IPComp** | 次优 | 次优 | 快（但湍流数据慢） | 局部精度优先 |
| **SZ3-R** | 仅在宽松误差下优 | 一般 | 极慢（链式累积） | 不推荐用于紧误差或实时检索 |

> ProAICD 在大多数误差界下均优于 PMGARD 和 IPComp。SZ3-R 仅在宽松误差下因恰好命中其预设阈值而表现好，但在紧误差下因冗余累积而最差。

### **消融实验结果**
- **Ablation Study **(S3D 数据集)：
  - **Error-bound mode**：从 PMGARD → +Adaptive Interpolation → +Coefficient Decomposition，bit-rate 持续下降，尤其在 CH4、CO2 等字段上效果显著。
  - **PSNR mode**：从 IPComp → +Adaptive Interpolation（探索最优顺序）→ +Coefficient Decomposition，PSNR 明显提升，证明两项改进均有效。
- **采样调优有效性**：
  - 1% 采样调优决策与全数据决策一致性达 **85.0%**（插值）和 **88.1%**（系数分解）。
  - 调优时间从占重构时间的 **87.2%** 降至 **11.5%**，性价比极高。

---

## **4. 关键结论和发现**

### **主要发现**
1. **自适应是关键**：没有单一插值方案适用于所有目标。**Per-level** 更适合 **error-bound**，**Per-region** 更适合 **PSNR**，自动切换带来显著收益。
2. **系数间存在可利用的相关性**：传统方法将系数视为独立编码，但实验证明其内部仍有强空间相关性，**coefficient decomposition** 能进一步压榨压缩潜力。
3. **端到端性能受益于高效压缩**：即使检索时间略长，但由于传输数据量大幅减少，**ProAICD 在远程数据传输中仍取得整体速度领先**。
4. **高质量可视化无需高数据量**：ProAICD 能以最低 bit-rate 实现最佳视觉质量，对交互式科学数据分析极具价值。

### **方法的局限性**
- **额外元数据开销**：为支持 coefficient decomposition 的误差控制，需存储 segment-error bound 映射表，虽总体开销小，但存在。
- **调优带来轻微延迟**：在线调优过程会增加初始压缩时间，尽管已通过采样优化。
- **三维及以上数据相关性假设**：系数分解主要针对 2D 切片设计，对更高维数据的普适性有待验证。

### **未来工作方向**
- 探索更先进的算法，进一步挖掘原始数据和去相关系数中的潜在关联。
- 将框架扩展至 **GPU** 平台，以大幅提升吞吐量，适应 exascale 级实时压缩需求。
- 研究面向特定 **Quantity of Interest **(QoI) 的渐进检索策略，实现更智能的数据交付。

</details>

---

### 7. [When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference](https://arxiv.org/abs/2609.04490)

**Authors**: Ismail Erbas, Xavier Intes, Vikas Pandey  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.04490v1  

#### Abstract
Quantization is widely used to reduce the computational and memory demands of neural-network inference. In recurrent networks, however, the quantized state is stored and returned at the next time step, so the rule used to store that state can alter subsequent computations. Here, we introduce recurre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文揭示并系统研究了**低精度量化（low-precision quantization）在循环神经网络（RNN）推理中对时序动态行为的破坏性影响**，尤其是当**循环状态（recurrent state）被低比特存储时**。

传统观点认为，只要模型在训练阶段引入量化感知（Quantization-Aware Training, QAT），就能适应低精度部署。然而，本文指出：
- 即使模型经过QAT训练，**在部署时若改变状态写入规则（write-back）仍可能导致严重失败**；
- 这种失败源于**量化过程抑制了微小但持续的状态更新**，导致网络“记忆”停滞，从而破坏了依赖长期时序累积的任务。

### 🚀 提出的新方法与新思路

#### （1）提出“**Recurrent-State Write-Back**”概念
- 定义了从计算状态 $ h_t $ 到存储状态 $ q_t $ 的映射规则为 **recurrent-state write-back**。
- 强调该操作是**部署时执行的动态计算的一部分**，而不仅仅是被动的数据编码。

#### （2）引入“**Recurrent Write Margin**”诊断工具
- 定量衡量每次状态更新是否足以跨越量化步长（half-step boundary）：
  $$
  M_{t,j} = \frac{2|\delta_{t,j}|}{\Delta_B}
  $$
  若 $ M_{t,j} < 1 $，则更新被“截断”，不进入下一时刻的可见状态。

#### （3）设计三种**无需重训练即可恢复精度的记忆增强机制**
| 方法 | 原理 | 特点 |
|------|------|------|
| **Error Feedback** | 将量化误差传递到下一次写入机会 | 保留幅度信息 |
| **Residual Memory** | 用辅助k-bit状态存储残差 | 可控精度 |
| **Direction Memory**（本文提出） | 用计数器积累同方向亚阈值更新，达到阈值后触发跃迁 | 仅需少量比特，高效捕捉趋势 |

> ✅ **创新性**：首次将“状态写回”视为一个可独立干预的设计变量，并展示了其对RNN动态轨迹的因果影响。

### 🔍 相比现有方法的优势

| 维度 | 传统做法 | 本文贡献 |
|------|----------|-----------|
| **分析视角** | 关注整体精度下降 | 聚焦**状态写回接口**的独立作用 |
| **修复方式** | 需要重新训练或调整架构 | **冻结权重下通过写回策略修复** |
| **通用性** | 多针对前馈网络 | 在GRU/LSTM上验证，具有跨架构适用性 |
| **理论深度** | 缺乏机制解释 | 提出“**持久亚阈值更新被抑制**”为核心失效机理 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用模拟的**荧光寿命成像（Fluorescence Lifetime Imaging, FLI）数据集**，由 [PyFLI](https://github.com/rpi-nmr/pyfli) 生成。
- 包含 1,600,000 条高噪声时间分辨荧光信号，每条含 **135个时间步**。
- 分割为：128万训练、16万验证、16万测试。

### 🧪 模型与任务
- **主模型**：`Seq2SeqLite` —— 单层32单元的GRU编码器-解码器，共6,627参数，用于从荧光信号估计两个寿命参数 $ T_1 $ 和 $ T_2 $。
- **辅助模型**：独立训练的32单元LSTM进行跨架构复现。
- **目标**：从嘈杂的时间序列中准确提取 $ T_1 $ 和 $ T_2 $。

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Lifetime RMSE** | 主要任务指标，$ T_1 $ 和 $ T_2 $ 的均方根误差（单位：ns） |
| **Sequence MAE** | 解码输出序列的平均绝对误差，用于区分重建误差与参数估计误差 |
| **Deadband Fraction** | 更新落在量化死区内的比例（$ M < 1 $） |
| **State-change Fraction** | 每步发生状态变化的隐藏单元占比 |
| **Same-sign Run Length** | 连续同方向亚阈值更新的长度，反映趋势持久性 |

### ⚖️ 实验设置与对比基线

#### （1）**Post-Training Write-Back Intervention**
- 固定已训练模型的所有参数（weights, gates, readout等），**仅更改状态写回规则**。
- 对比不同 write-back 规则：
  - `Identity`（连续传播）
  - `Deterministic B-bit`（确定性B位量化）
  - `Stochastic Rounding`
  - `Error Feedback`
  - `Residual Memory`
  - `Direction Memory`

#### （2）**Precision Sweep**
- 固定训练好的8-bit或4-bit模型，测试其在不同写回精度下的表现（如8-bit训练 → 4-bit写回）。

#### （3）**Matched Training**
- 在相同初始化下训练四种配置：
  - 4-bit state
  - 6-bit state
  - 4-bit state + 2-bit residual memory
  - 4-bit state + 2-bit direction memory  
  → 验证接口兼容性是否可通过训练学习。

#### （4）**Cross-Architecture Replication**
- 在独立训练的LSTM上重复上述实验，比较cell state与hidden state的敏感性差异。

---

## 3. 主要实验结果和性能指标

### 📉 关键性能数据（GRU 结果）

| 模型 / 设置 | $ T_1 $ RMSE (ns) | $ T_2 $ RMSE (ns) | 备注 |
|------------|---------------------|---------------------|------|
| **P2F Identity** | 0.36 | 0.35 | 冻结模型，连续状态传播 |
| **P2F Det. 4-bit** | 25.37 | 106.59 | ↑约70倍 & 300倍！ |
| **P2F + Error Feedback** | 0.36 | 0.37 | 几乎完全恢复 |
| **P2F + 2-bit Residual** | 0.34 | 0.40 | 有效恢复 |
| **P2F + 3-bit Direction Mem** | 0.34 | 0.34 | 新方法同样有效 |

> 💥 **核心发现**：仅改变写回规则，即可使误差**增加数百倍**，证明 write-back 是决定性因素。

### 🔁 写回抑制现象量化
- 在失败条件下（det. 4-bit）：
  - **99.59%** 的解码器更新处于写回死区（deadband）
  - 平均每步仅有 **0.08个单元** 发生状态变化
  - 同方向亚阈值运行中位数达 **132步**（接近整个序列长度）

> ❗ 表明网络仍在“提议”变化，但这些变化未被写入，导致记忆停滞。

### 🔄 记忆机制恢复效果（8-bit Reference → 4-bit Write-Back）
| 方法 | $ T_1 $ RMSE | $ T_2 $ RMSE | vs Native (0.20/0.22) |
|------|---------------|---------------|------------------------|
| Native 8-bit | 0.20 | 0.22 | ✅ 最佳 |
| Det. 4-bit | 1.89 | 2.60 | ❌ 显著退化 |
| + Error Feedback | 0.34 | 0.46 | ✅ 恢复90%以上精度 |
| + 4-bit Residual | 0.34 | 0.45 | ✅ 有效 |
| + 4-bit Direction | 0.49 | 0.58 | ✅ 有效（仅存方向） |

> ✅ 所有记忆机制均能显著恢复精度，说明被丢弃的信息是有用的。

### 📊 精度并非越高越好（Precision Sweep）
| 模型 | 原始写回 | 改为8-bit写回 | 结果 |
|------|---------|--------------|------|
| **4-bit-state reference** | 4-bit | 8-bit | $ T_1 $: 0.35 → 0.43 ns ❌ |
| **8-bit-state reference** | 8-bit | 4-bit | $ T_1 $: 0.20 → 1.89 ns ❌ |

> 🔁 **反直觉发现**：提高状态精度反而可能降低性能 —— 因为破坏了训练时形成的“接口兼容性”。

### 🧩 匹配训练结果（Learned Compatibility）
| 配置 | $ T_1 $ RMSE (mean) | $ T_2 $ RMSE (mean) | 优势 |
|------|------------------------|------------------------|------|
| 4-bit state | 0.41 | 0.48 | 基线 |
| 6-bit state | 0.48 | 0.38 | 更好于T2 |
| + 2-bit direction memory | **0.30** | 0.43 | ✅ **T1最优** |

> ✅ 证明：**接口与模型可协同优化**，不存在普适最优方案。

### 🔄 LSTM 跨架构复现
| 设置 | $ T_1 $ RMSE | $ T_2 $ RMSE | 发现 |
|------|---------------|---------------|------|
| Native 8-bit | 0.239 | 0.254 | ✅ 正常 |
| Both → 4-bit | 3.857 | 1.327 | ❌ 失败 |
| Cell only → 4-bit | 8.158 | 1.292 | ❗ **更敏感** |
| Hidden only → 4-bit | 0.296 | 0.354 | 影响较小 |

> 🔍 **关键发现**：**Cell State 比 Hidden State 对粗粒度写回更敏感**，即使后者有更高的亚阈值比例。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Recurrent-State Write-Back 是低精度RNN推理的关键设计变量**  
   - 不仅是数据表示，更是**动态计算流程的一部分**。
   - 改变写回规则可在不修改任何权重的情况下导致任务失败。

2. **失效机制是“持久亚阈值更新被抑制”**  
   - 当网络反复提议同一方向的小幅更新时，若始终低于量化阈值，则存储状态几乎不变 → “记忆冻结”。

3. **信息可跨时间补偿以恢复精度**  
   - **Error Feedback、Residual Memory、Direction Memory** 均能在冻结模型上恢复精度。
   - 说明被丢弃的更新携带有用信息。

4. **数值精度 ≠ 动态保真度（Numerical Fidelity ≠ Dynamical Fidelity）**  
   - 更高的比特宽度不一定更好，**接口兼容性更重要**。
   - 训练过程中形成的动态路径依赖特定的写回行为。

5. **不同RNN组件敏感性不同**  
   - 在LSTM中，**Cell State 比 Hidden State 更易受粗写回影响**，表明功能角色决定脆弱性。

---

### ⚠️ 局限性

| 局限 | 说明 |
|------|------|
| **任务特定性** | 实验基于FLI这一高度依赖时序积分的任务，其他任务可能不如此敏感 |
| **硬件抽象** | 未考虑实际硬件延迟、内存带宽等约束 |
| **扩展性未知** | 方法在更深、更大模型上的有效性尚未验证 |
| **方向记忆泛化性** | Direction Memory 是否适用于非单调趋势任务有待检验 |

---

### 🔮 未来工作方向

1. **开发面向 write-back 兼容性的新型量化训练算法**
   - 如：显式建模 write-back 接口作为训练图一部分。

2. **构建 write-back-aware 的RNN架构设计原则**
   - 指导如何分配比特资源（如 cell vs hidden state）。

3. **将 write-back 分析推广至Transformer等长程依赖模型**
   - 如KV Cache的低精度存储问题。

4. **探索更高效的辅助记忆结构**
   - 如稀疏计数器、事件驱动更新等。

5. **结合硬件协同设计**
   - 设计支持 error feedback 或 direction memory 的专用加速器。

---

## 总结

> 🌟 本论文从根本上改变了我们看待低精度RNN的方式：  
> **状态不是被动存储的数据，而是通过 write-back 规则主动塑造的动态轨迹。**

它不仅揭示了一个被忽视的关键失效机制，还提供了无需重训练即可修复的实用工具，并呼吁将“**状态接口设计**”提升为与模型架构同等重要的系统级考量。

</details>

---

### 8. [ACE: Adaptive Calibration-Free Expert Skipping for MoE-based LLMs](https://arxiv.org/abs/2609.05228)

**Authors**: Zukang Xu, Zhixiong Zhao, Xing Hu, Jiangyong Yu, Houji Wen, Jun Li, Zhe Jiang, Dawei Yang  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.05228v1  

#### Abstract
Mixture-of-Experts (MoE) architectures provide an efficient paradigm for scaling large language models (LLMs), yet fixed top-k routing activates the same number of expert slots for every token, causing substantial redundant computation. Existing expert-skipping methods often rely on router confidenc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ACE: Adaptive Calibration-Free Expert Skipping for MoE-based LLMs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在基于 **Mixture-of-Experts (MoE)** 架构的大型语言模型中，尽管每个 token 只激活少量专家（top-k routing），但固定数量的激活机制导致大量**冗余计算**。许多被路由到的专家对最终输出贡献极小，却仍需执行完整的前向传播，造成推理效率低下。

现有动态专家跳过（expert skipping）方法通常依赖以下之一：
- **Router confidence**（如仅根据路由分数判断）
- **Calibration data**（需要额外样本进行阈值校准）
- **Retraining 或微调**

这些方法存在明显缺陷：**router score 并不等同于实际变换贡献**，且依赖校准数据会限制部署灵活性。

---

### 提出了什么新方法或新思路
本文提出 **ACE (Adaptive Calibration-Free Expert Skipping)**，一种无需训练、无需校准数据、不修改预训练检查点的 token 自适应专家跳过框架。

其核心思想是：**从两个互补视角联合估计专家的实际贡献**，只有当两者均认为某专家贡献低时才跳过。

#### 两大核心组件：
1. **Global Spectral Proxy (GSP)**  
   - 从 **SwiGLU 专家的参数结构**（gate, up, down 投影 + RMSNorm 缩放）出发，构建一个无输入依赖的全局响应能力代理。
   - 利用矩阵范数的乘性耦合特性，通过分支对称因子化（up-gate 和 gate-down 路径）并取几何平均，估算专家的“静态放大能力”。
   - 所有统计量可离线计算，仅需在线查表。

2. **Router-Conditioned Refinement (RCR)**  
   - 针对 GSP 忽略方向特异性的缺点，引入路由偏好方向。
   - 将预训练 router 权重中心化后 RMS 归一化，构造出每个专家的“条件方向原型” $ q_{l,e} $。
   - 离线将该原型输入对应专家，测量其沿此方向的响应强度，作为方向性贡献修正项。

#### 在线跳过决策：
- 对每个 token 的 top-k 专家，结合运行时 router gate $ g_{l,t,i} $ 与离线 GSP 和 RCR 分数。
- 定义综合贡献得分：
  $$
  c_{l,t,i}^{\text{ACE}} = \max(p_{l,t,i}^{\text{GSP}}, p_{l,t,i}^{\text{RCR}})
  $$
- 若 $ c_{l,t,i}^{\text{ACE}} < T(q) $，则跳过；否则保留。
- **始终保留 top-1 专家**，确保基本路由结构不变。

---

### 相比现有方法的优势
| 特性 | ACE | 现有方法（Score, NAEE, MoDES 等） |
|------|-----|-------------------------------|
| 是否需要训练 | ❌ 否 | ✅ 多数需要 |
| 是否需要校准数据 | ❌ 否 | ✅ 多数需要 |
| 是否修改 checkpoint | ❌ 否 | ✅ 部分方法需要 |
| 是否 token 自适应 | ✅ 是 | ✅ 动态类方法支持 |
| 是否考虑专家结构 | ✅ 是（GSP） | ❌ 多数仅看 router score |
| 是否考虑方向特异性 | ✅ 是（RCR） | ❌ 否 |
| 推理开销增加 | 极低（仅查表 + 标量运算） | 中高（可能需额外模块或搜索） |

> ✅ **优势总结**：ACE 实现了真正的 **zero-cost, plug-and-play** 式高效推理优化，在保持模型完整性和部署便捷性的同时显著提升效率。

---

## 2. 核心实验方法和设置

### 使用的模型
在三个主流 MoE-based LLM 上验证：
- **Qwen3-30B-A3B-Instruct-2507**
- **Qwen3.6-35B-A3B**
- **Gemma-4-26B-A4B-it**

均为 SwiGLU + top-2 routing 结构。

---

### 数据集与任务
#### 语言建模：
- **WikiText-2**（PPL，序列长度 2048）

#### 下游任务（共 7 个）：
- **ARC-Challenge (ARC-C)**, **ARC-Easy (ARC-E)**
- **PIQA**
- **MATH-500**
- **GPQA-Diamond**
- **HumanEval**
- **LiveCodeBench**

报告 **平均准确率（Avg. Acc.）**，WikiText-2 报告 PPL。

---

### 实验设置
- **评估协议**：
  - 所有方法使用相同 BF16 实现、prompt、split、greedy decoding。
  - **Zero-shot evaluation**。
  - 生成长度上限：推理类任务 2048 tokens，其他 1024 tokens。
- **跳过比例定义**：
  - 指 **被路由但未执行的 top-k 专家槽位占比**。
  - 报告的是 **realized skipping ratio**（实际实现的比例），而非目标值。
- **控制变量**：
  - 所有方法保留原始 top-k 候选集。
  - top-1 专家强制保留。
  - 最小活跃专家数约束统一处理。

---

### 基线方法对比
| 方法 | 类型 | 是否需校准 | 是否需训练 |
|------|------|------------|-----------|
| **Score** | Router score 跳过 | ✅ | ❌ |
| **NAEE** | 基于相对 router score + 层级阈值 | ✅ | ❌ |
| **MoDES** | 多模态下基于激活统计的跳过 | ✅ | ❌ |
| **DiEP** | 基于可微剪枝的压缩方法 | ❌ | ✅ |
| **AIMER**, **Top-P**, **SERE**, **XShare** | 静态/动态压缩或重路由 | 部分需 | 部分需 |

> ACE 与所有基线在同一公平条件下比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（代表性结果）

#### 在 **Qwen3.6-35B-A3B** 上 50% 跳过率表现：
| 方法 | PPL ↓ | Avg. Acc. ↑ |
|------|--------|-------------|
| BF16（全量） | 7.01 | 81.17 |
| MoDES | 9.42 | 71.42 |
| **ACE (Ours)** | **8.67** | **75.57** |

✅ **ACE 相比最强竞争者 MoDES**：
- **PPL 降低 7.96%**
- **平均准确率提升 4.15 个百分点**

#### 在 **Qwen3-30B-A3B** 上 60% 跳过率表现：
| 方法 | PPL ↓ | Avg. Acc. ↑ |
|------|--------|-------------|
| BF16 | 7.52 | 80.67 |
| AIMER | 10.89 | 60.88 |
| **ACE (Ours)** | **10.86** | **63.35** |

✅ **ACE 在极端跳过率下仍保持领先**，准确率高出 **2.47 pts**

---

### 与基线方法的对比结果
- 在所有模型和跳过预算（10%-60%）下，**ACE 始终优于所有基线**。
- 在轻度跳过（10%-30%）时，部分 router-based 方法尚可接受。
- **随着跳过率上升，ACE 优势急剧扩大**，尤其在 40%-60% 区间远超其他方法。
- 图 1 显示：ACE 在 **accuracy-efficiency trade-off 曲线上全面占优**，特别是在 aggressive regime。

---

### 消融实验结果（Ablation Study）

#### 表 4：组件消融（Qwen3-30B, 50% skip）
| 方法 | PPL ↓ | Avg. Acc. ↑ | 相对下降 |
|------|--------|-------------|----------|
| GSP only | 8.99 | 74.10 | -6.57 |
| RCR only | 11.62 | 67.01 | -13.66 |
| **ACE (GSP + RCR)** | **8.85** | **74.30** | **-6.37** |

✅ **GSP 单独效果更好，但二者融合进一步提升性能**，说明 RCR 提供了有价值的补充信息。

#### 表 5：融合策略消融
| 融合方式 | 30% PPL/Acc | 50% PPL/Acc |
|---------|--------------|--------------|
| Min（任一低即跳） | 8.21 / 77.20 | 9.20 / 73.12 |
| Mean（加权平均） | 7.95 / 78.45 | 8.97 / 74.01 |
| **Max（双重视角均低才跳）** | **7.86 / 79.77** | **8.85 / 74.30** |

✅ **Max 规则最优**：保守策略有效防止误删重要专家，避免性能骤降。

---

## 4. 关键结论和发现

### 主要发现
1. **Router confidence ≠ Expert contribution**  
   仅靠 router score 无法可靠识别低贡献专家，尤其是在高跳过率场景下会导致性能崩溃。

2. **结构感知 + 方向感知 = 更鲁棒的跳过决策**  
   GSP 捕捉专家的全局变换能力，RCR 捕捉其在 router 偏好方向上的局部响应，二者结合形成互补。

3. **保守融合策略至关重要**  
   使用 `max` 规则（双视角共识）能有效防止误删，保障模型稳定性。

4. **ACE 实现近乎零成本加速**  
   - 所有专家统计量可**离线预计算**。
   - 在线仅需 **table lookup + 标量操作**，无额外 forward pass。
   - 实测推理延迟大幅下降：
     - **Prefill 阶段最高提速 2.25×**
     - **Decoding 阶段最高提速 1.41×**

5. **跨数据集阈值可迁移性强**  
   表 8 显示，同一跳过预算下的阈值在不同 workload 间高度一致，支持跨任务复用。

---

### 方法的局限性
- **假设 router 权重具有方向判别意义**：依赖于 router weight 的线性可分性假设，在某些复杂路由模式下可能失效。
- **未探索更复杂的融合机制**：目前采用固定的 `max` 规则，未来可尝试 learnable fusion。
- **仅适用于 SwiGLU 结构**：GSP 设计基于 SwiGLU 的乘性结构，对其他 FFN 类型需调整。
- **极端跳过率下仍有性能损失**：虽然优于基线，但在 >60% 跳过率时仍不可避免地影响质量。

---

### 未来工作方向
1. **Threshold transferability 研究**：探索是否可在不同模型或领域间直接迁移 ACE 阈值。
2. **分布式 MoE 推理优化**：将 ACE 与 expert dispatch 机制结合，减少通信开销。
3. **多粒度跳过**：扩展至 layer-wise 或 block-wise 的自适应跳过策略。
4. **不确定性建模**：为 ACE 输出引入置信度估计，支持风险敏感应用。
5. **扩展至非 SwiGLU 架构**：适配其他类型的 MoE 模块（如 ReLU-based）。

---

> 🔚 **总结**：ACE 是首个真正实现 **training-free, calibration-free, checkpoint-preserving** 的动态专家跳过框架，通过结构感知与方向感知的双重验证机制，在保证模型稳定性的前提下实现了显著的推理加速与精度保持，为 MoE 模型的高效部署提供了实用且强大的解决方案。

</details>

---

### 9. [Communication-Efficient Personalized Federated Learning via Layer-Wise Multi-Threshold Random Sketching](https://arxiv.org/abs/2609.04830)

**Authors**: Xu Zhang, Xingyu Hou, Jiacheng Cheng, Kaiyuan Feng, Maoguo Gong  
**Category**: cs.LG  
**Published**: 2026-09-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.04830v1  

#### Abstract
Personalized federated learning (PFL) is a promising paradigm for collaborative learning over distributed devices, where edge nodes collaboratively train personalized models without sharing raw data. Although PFL addresses data heterogeneity by learning client-specific models, it still suffers from ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Communication-Efficient Personalized Federated Learning via Layer-Wise Multi-Threshold Random Sketching

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **Personalized Federated Learning (PFL)** 中，尽管模型个性化缓解了数据异构性问题，但客户端与服务器之间频繁交换高维参数仍导致严重的通信开销，尤其在带宽受限的边缘设备系统中成为瓶颈。现有的 **one-bit 压缩方法**（如 signSGD、pFed1BS）虽然实现了极致压缩，但存在以下两个关键缺陷：
- **忽略层间差异**：采用统一的单阈值二值化规则处理所有网络层，未考虑不同层参数分布的统计特性差异（如浅层集中、深层分散）。
- **表达能力有限**：单一阈值只能提供粗粒度的二进制信息，难以捕捉参数分布中的细粒度变化。

### 提出的新方法与创新思路
本文提出了一种新的通信高效 PFL 框架 —— **pFedLMS**（Personalized Federated Learning via Layer-Wise Multi-Threshold Random Sketching），其核心是 **Layer-Wise Multi-Threshold Random Sketching (LMTRS)** 机制：

- **分层多阈值量化（Layer-wise Multi-Threshold Quantization）**  
  每一层独立分配一组有序的量化阈值 $ \mathcal{T}_l = \{T_{l,1} < \cdots < T_{l,T}\} $，将该层的随机投影（sketched）参数划分为 $ T+1 $ 个区间，从而实现对层特定分布的自适应建模。

- **双向低比特通信协议**  
  客户端上传的是每个阈值下的符号比较结果（即 one-bit sketches），通过 $(T+1)$-ary 编码压缩为仅需 $ \lceil m\log_2(T+1) \rceil $ 比特/层的消息；服务器聚合后广播同样格式的共识信号，实现**双向极低比特通信**。

- **理论支持的正则化设计**  
  引入基于多阈值一致性的对齐正则项 $ R(\theta_k; V) $，并证明其具有“区间一致性”解释：当本地模型落在由全局共识诱导的一致区间内时，惩罚为零。同时使用 Nesterov 平滑技术使其可微，便于优化。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **表达能力更强** | 多阈值设计比单阈值能更精细地描述参数分布，提升信息保留度 |
| **结构感知性强** | 分层独立阈值适配 CNN/DNN 各层不同的统计特性 |
| **通信效率极高** | 双向均使用低比特 sketch（例如 T=7 时每坐标仅需 3 bits），相比 full-precision 减少约 98% 通信量 |
| **兼容性强** | 支持任意深度神经网络结构，无需改变模型架构 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在五个标准图像分类数据集上进行：
- **MNIST**, **FMNIST**: 手写数字与时尚物品分类
- **SVHN**: 街道门牌数字识别
- **CIFAR-10**, **CIFAR-100**: 自然图像分类任务

所有数据以非独立同分布（non-IID）方式划分给 **20 个客户端**，使用 **Dirichlet 分布**控制标签偏斜程度，设定超参数 $ \alpha = 0.1 $ 和 $ \alpha = 0.5 $，模拟不同程度的数据异构性。

### 实验设置与评估指标

#### 模型结构
- MNIST/FMNIST: DNN
- CIFAR-10: VGG8
- CIFAR-100: VGG16

#### 超参数配置
- 局部 batch size: 64
- 本地训练 epoch: [20, 50]
- 总通信轮次: [200, 350]
- Sketch dimension $ m $: 根据层大小设定
- 阈值数量 $ T = 7 $（默认），对应 8 区间、3 bits/坐标
- Sketching operator: Hadamard 投影（高效且无需传输）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Maximum Accuracy (%)** | 测试集上达到的最高平均准确率（多次运行均值 ± 标准差） |
| **Communication Cost (bits/round)** | 单轮通信中客户端与服务器之间交换的总比特数（含上下行） |

### 基线方法对比
涵盖多种代表性联邦学习与通信压缩方法：
- **FedAvg**：标准联邦平均（全精度）
- **OBDA**、**OBCSAA**：基于 one-bit 压缩的 FL 方法
- **zSignFed**：随机符号扰动方法
- **EDEN**：低比特分布式均值估计
- **FedProto**：原型通信框架
- **pFed1BS**：当前最先进的双向 one-bit PFL 方法（本文直接改进对象）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

| 方法 / 数据集 | MNIST ($\alpha=0.5$) | CIFAR-10 ($\alpha=0.5$) | CIFAR-100 ($\alpha=0.5$) |
|--------------|------------------------|----------------------------|------------------------------|
| FedAvg       | 98.55 ± 0.08           | 86.57 ± 0.34               | 61.14 ± 0.58                 |
| pFed1BS      | 95.10 ± 0.33           | 72.97 ± 0.36               | 33.98 ± 0.52                 |
| **pFedLMS (Ours)** | **97.45 ± 0.32**       | **82.14 ± 0.46**           | **47.47 ± 0.97**             |

> 在 $\alpha=0.1$ 更强异构下，pFedLMS 依然保持领先，尤其在 CIFAR-100 上达到 **66.78%**，显著优于其他压缩方法。

### 与基线方法的对比结果
- **准确性方面**：
  - 在多数数据集上，pFedLMS 显著优于所有 one-bit 或低比特压缩方法（如 OBDA、OBCSAA、zSignFed）。
  - 相比同属 one-bit sketching 的 **pFed1BS**，pFedLMS 在 CIFAR-10 和 CIFAR-100 上分别提升 **9.17%** 和 **13.49%**（$\alpha=0.5$），验证了多阈值与分层设计的有效性。
  - 接近甚至超过部分全精度方法（如 FedProto、EDEN），表明信息损失极小。

- **通信成本方面**（见 Figure 5）：
  - 相比 FedAvg，pFedLMS 在 CIFAR-100 上从 **1495.34 MB → 25.05 MB**，降低 **98.3%**。
  - 在 MNIST 上仅需 **0.38 MB/轮**，而 FedAvg 需 31.06 MB。
  - 远低于 EDEN、zSignFed 等方法，在相同通信预算下取得更高精度（见 Figure 6）。

### 消融实验结果（Ablation Study）

#### （1）组件有效性分析（Table 3）
| 方法变体 | 是否多阈值 | 是否分层 | SVHN Acc (%) | CIFAR-10 Acc (%) |
|----------|------------|-----------|---------------|------------------|
| Refined pFed1BS | × | × | 89.80 | 78.68 |
| pFedLMS (T=1) | × | √ | 90.31 (+0.51) | 79.89 (+1.21) |
| Only Multi-threshold | √ | × | 91.41 (+1.61) | 81.17 (+2.49) |
| **pFedLMS (T=7)** | √ | √ | **91.94 (+2.14)** | **82.61 (+3.93)** |

> 结果显示：**分层设计** 和 **多阈值机制** 均带来增益，二者结合产生协同效应。

#### （2）阈值数量影响（Figure 7）
在 CIFAR-10 上测试不同 $ T $ 值的影响：
- $ T=1 $（即 sign-only）：79.67%
- $ T=2 $：81.05%（最大单步增益）
- $ T=7 $：82.74%

> 表明增加阈值可提升性能，但收益边际递减，说明少量阈值即可捕获大部分有用信息，适合实际部署。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **分层 + 多阈值 sketching 显著优于传统 one-bit 方法**：能够更好地适应深层网络各层的统计异质性，减少信息损失。
2. ✅ **双向低比特通信可行且高效**：通过紧凑的 $(T+1)$-ary 编码，可在几乎不牺牲精度的前提下实现 **>98% 通信压缩**。
3. ✅ **所提正则化机制具备良好理论性质**：多阈值共识可诱导出“一致性区间”，平滑版本支持梯度优化。
4. ✅ **pFedLMS 在高度 non-IID 场景下鲁棒性强**：即使在 $ \alpha=0.1 $ 极端异构下仍表现优异。

### 方法的局限性
- **依赖统计假设**：阈值基于高斯分布的分位数生成，若实际分布严重偏离可能影响效果。
- **引入额外超参数**：如 $ T $（阈值数）、$ \lambda $（正则系数）需调优（尽管实验显示对 $ \lambda $ 不敏感）。
- **Sketching refresh 开销**：虽无需通信，但频繁更新 sketching matrix 可能增加计算负担（Appendix B.1 显示越频繁越好）。

### 未来工作方向
- 将 LMTRS 扩展至 **非参数化模型** 或 **Transformer 架构**。
- 探索 **自适应阈值选择策略**（如根据梯度方差动态调整）。
- 结合 **差分隐私** 实现通信高效且隐私安全的 PFL。
- 在真实边缘设备（IoT、移动端）上部署验证端到端延迟与能耗表现。

--- 

> 📌 **一句话总结**：  
> pFedLMS 通过 **layer-wise multi-threshold random sketching** 实现了高保真、超低比特的双向通信，在保证个性化性能的同时大幅降低通信开销，为资源受限场景下的 PFL 提供了一个极具前景的新范式。

</details>

---

### 10. [IPGeoAI: Transformer-Based Geolocation with LLM Semantic Fusion](https://arxiv.org/abs/2609.04559)

**Authors**: Avinash Kadimisetty, Andy Jinqing Yu, Philip Favaloro, Wenlong Liu, Xiaolu Xiong  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.04559v1  

#### Abstract
Accurate city-level IP Geolocation is an important enabler for the modern digital ecosystem, underpinning services ranging from local content delivery and targeting to digital rights enforcement. However, traditional heuristic and database-driven methods often struggle to resolve the complex, non-li...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*IPGeoAI: Transformer-Based Geolocation with LLM Semantic Fusion*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代互联网中，**高精度城市级 IP Geolocation** 是支撑内容分发、数字版权管理、安全风控等关键服务的基础能力。然而，传统基于数据库查找（如 WHOIS）或主动探测（如 traceroute）的方法面临以下挑战：
- **难以处理 IPv6 的爆炸性增长** 和动态移动网络；
- 对“长尾”IP（新分配、低频次）覆盖不足（recall 低）；
- 忽视 IP 地址本身的**层次化结构特征**；
- 缺乏对 Autonomous System (AS) 描述文本中的**语义信息**的有效利用。

### 提出了什么新方法或新思路
本文提出 **IPGeoAI**，一种全新的深度学习框架，将 IP Geolocation 从静态查找任务重构为**序列建模 + 多模态融合预测任务**，其核心创新包括：

1. **Transformer-based Sequential Modeling**  
   将 IP 地址视为一个字节序列（octet sequence），使用 **Transformer Encoder** 学习其内部的层级依赖关系（例如 `/24` 子网嵌套于 `/16` 中），从而捕捉 CIDR 分配策略中的潜在模式。

2. **LLM 驱动的 Semantic Feature Fusion**  
   利用 **Large Language Model (LLM)** 在离线阶段对 ASN 元数据（如 ASN Name、WHOIS 名称）进行零样本分类（zero-shot classification），生成 9 维结构化语义特征向量，涵盖：
   - 组织身份与范围（Organizational Identity & Scope）
   - 使用场景与人口统计（Usage & Demographics）
   - 网络拓扑层级（Topology & Hierarchy）

3. **Cross-Modal Attention Fusion**  
   设计 **Multi-Head Cross-Attention 模块**，让 IP 序列表示作为 Query，去动态关注（attend to）相关的语义元数据（Keys/Values），实现上下文感知的特征融合，提升歧义解析能力。

4. **Hierarchical Inference Strategy**  
   引入外部高可靠性的国家级别信号作为硬约束，在 top-k 城市候选集中仅保留目标国家内的城市，避免因城市误判导致国家错误，提升地理一致性。

### 相比现有方法的优势
| 方面 | 传统方法（Heuristic DB / GNN） | IPGeoAI |
|------|-------------------------------|--------|
| 可扩展性 | 依赖主动探测，延迟高，难扩展至 IPv6 | 被动推理，O(1) 推理复杂度，适合超大规模部署 |
| 泛化能力 | 严重依赖历史高频数据，“冷启动”问题明显 | 可通过结构+语义泛化到未见子网 |
| 特征表达 | 数值型拓扑特征为主，忽略非结构化文本语义 | 显式引入 LLM 提取的高质量软标签语义特征 |
| 实时性 | GNN 需实时图聚合，I/O 开销大 | 批处理 + 日更模型，支持毫秒级在线查询 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **专有数据集（Proprietary Dataset）**：来自 Meta 内部的 IP-GPS 日志，覆盖全球约 **200,000 个城市**。
- **Ground Truth 构建**：采用 **7天滑动窗口聚合策略**，平滑单日 GPS 波动，生成稳定的“稳定日”真实位置标签。
- **ASN 元数据来源**：
  - ASN ID、Name 来自公开 BGP 路由表（如 RouteViews）；
  - WHOIS 数据用于 LLM 输入；
  - LLM 输出经 PeeringDB 抽样验证（审计 25 个 ASN，准确率 76%-80%）。

### 实验设置和评估指标
#### 主要评估维度
- **Offline Evaluation**：在测试集上比较模型性能；
- **Online A/B Testing**：在生产环境（Facebook & Instagram）中衡量下游业务指标影响。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **City Accuracy (Exact)** | 预测城市 ID 与真实城市 ID 完全匹配的比例 |
| **Region Accuracy (Exact)** | 区域级准确率（由城市向上聚合） |
| **City Accuracy @ 100km** | 预测城市中心与真实位置距离 ≤100km 的比例 |
| **Coverage** | 支持定位的 IP 流量占比（IPGeoAI 达到 100%） |

#### 分析粒度
- 按协议拆分：IPv4 vs IPv6
- 按前缀聚合：IP Trunk (`/28`/`/64`)、IP Trim (`/24`/`/48`)
- 使用第三方商业供应商结果作为国家约束输入（不参与评分）

### 基线方法对比
- **Third-Party Baseline**：领先的外部商业 geolocation 服务商（主要对比对象）
- **Meta Heuristic Model**：Meta 自研的传统启发式流水线
- **消融模型变体**（见下文）

> ⚠️ **排除的基线**：
> - **Graph Neural Networks (GNNs)**：虽学术先进，但需主动探测构建图、推理延迟高、冷启动差，不适合超大规模实时系统；
> - **Tree-based Models (XGBoost/LightGBM)**：多分类内存开销过大（>256GB），不可行。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 2）

| Metric | Third-Party Baseline | IPGeoAI | Δ |
|--------|-----------------------|---------|----|
| **City Accuracy (All)** | 30.09% | **36.47%** | **+6.38pp** |
| **City Accuracy (IPv4)** | 45.01% | 36.87% | -8.14pp |
| **City Accuracy (IPv6)** | 29.58% | **36.46%** | **+6.88pp** ✅ |
| **Region Accuracy (All)** | 76.63% | **79.21%** | **+2.58pp** |
| **City Acc @ 100km (All)** | 30.15% | **36.82%** | **+6.67pp** |

> 🔍 **关键发现**：
> - 在 **IPv6 上显著超越第三方**（+6.88pp），说明模型具备更强的泛化能力；
> - IPv4 表现略低，因其已被商业数据库“记忆”多年，属饱和领域；
> - 整体城市级准确率提升 **6% 绝对值**，且覆盖率达 100%。

### 消融实验结果（Table 3）

| Model Configuration | Recall (%) | Δ vs Prev |
|---------------------|------------|-----------|
| Multi-Layer Perceptron (Baseline) | 31.76 | — |
| Static Feature Transformer | 32.19 | +0.43 |
| **Sequential Octet Transformer** | **33.36** | **+1.17** ✅ |
| Attention-Based Feature Fusion | 35.70 | +2.34 ✅ |
| **Semantic Enrichment (Ours)** | **36.47** | **+0.77** ✅ |

> 📌 **结论**：
> - **Sequential Modeling** 是基础改进（+1.17pp）；
> - **Attention Fusion** 动态加权机制带来最大增益之一（+2.34pp）；
> - **LLM Semantic Features** 最终贡献 +0.77pp，证明语义融合有效。

### 在线 A/B 测试结果（Section 5.4）
- 实验周期：连续 13 天
- 流量分组：Control（旧模型） vs Treatment（IPGeoAI）
- 下游指标：**1st-tier downstream use cases metric**
- 结果：**+0.35% 的统计显著提升**

> 💡 这表明模型不仅在离线指标上领先，还能转化为实际业务价值。

---

## 4. 关键结论和发现

### 主要发现
1. **IP 地址具有可学习的层次结构模式**，Transformer 能有效建模这些非线性分配规律。
2. **纯数值信号不足以解决地理歧义**，引入 LLM 提取的语义特征（如 “Global ISP” vs “Local Municipal Network”）是突破瓶颈的关键。
3. **Attention Fusion 比简单拼接更优**，能根据 IP 结构动态选择相关元数据进行增强。
4. **IPv6 和动态网络是未来战场**，传统方法在此类“长尾”流量上表现差，而深度学习模型可通过泛化取得优势。
5. **离线性能提升可转化为线上业务收益**，+0.35% 的下游指标增长验证了系统的实用性。

### 方法的局限性
- 当前仍依赖第三方提供国家层级信号以保证宏观准确性；
- LLM 特征更新频率为周级，存在一定的**时效性滞后**；
- 模型为批处理架构，尚未支持完全实时推理；
- LLM 零样本预测在“混合用途”或模糊组织上仍有误差（约 20%-24% 不一致）。

### 未来工作方向
1. **端到端 Geo-Hierarchy Prediction**：构建统一模型输出从 Country 到 City 甚至 ZIP Code 或 tile 的完整地理位置层级。
2. **Real-Time Inference Optimization**：优化模型结构与 Serving Pipeline，支持低延迟实时推理。
3. **Fine-Grained Temporal Updates**：缩短 LLM 语义特征刷新周期，提高新鲜度。
4. **探索多任务学习**：联合训练 geolocation 与其他网络理解任务（如 AS 类型识别、异常检测）以共享表示。

--- 

> ✅ **总体评价**：  
> IPGeoAI 成功地将 **Transformer 架构** 与 **LLM 语义理解** 相结合，提出了一种适用于超大规模、高吞吐、低延迟场景的城市级 IP Geolocation 新范式。它不仅是技术上的进步，更是工程落地的成功典范，代表了下一代 IP 定位系统的发展方向。

</details>

---

### 11. [ProtLingo: Efficient Protein Language Modeling via Conditional Memory and Expert Routing](https://arxiv.org/abs/2609.04793)

**Authors**: Mingrui Li, Sixian Shen, Minzhang Li, Ruiyi Zhang, Kexin Zhang, Jiakai Zhang, Jingyi Yu  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.04793v1  

#### Abstract
Proteins perform diverse cellular functions, and even single amino-acid substitutions can alter stability, activity, or molecular interactions. Protein language models (PLMs) provide a scalable approach for modeling such sequence--function relationships from unlabeled sequences, but increasing the s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ProtLingo: Efficient Protein Language Modeling via Conditional Memory and Expert Routing**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **Protein Language Models (PLMs)** 主要依赖大规模密集 Transformer 骨干网络进行建模，虽然能捕捉序列-功能关系，但在以下方面存在瓶颈：
- **计算成本高**：扩大模型规模带来显著的计算开销。
- **对突变敏感任务提升有限**：在单氨基酸替换（mutation）预测等任务上，性能随模型增大趋于饱和。
- **缺乏对局部上下文的有效利用**：标准 PLMs 隐式建模局部模式，无法显式复用重复出现的局部序列上下文。

### **提出的新方法与创新思路**
作者提出了 **ProtLingo**，一种高效的蛋白质语言建模框架，通过两个核心机制增强预训练骨干模型（如 ESM2-150M）：

#### **(1) Centered Latent N-gram Memory（中心化潜在 N-gram 记忆）**
- 将上下文化的残基表示映射为**离散路由码（route-specific discrete codes）**。
- 构造以目标残基为中心的局部窗口（centered local window），组合其离散码形成**潜在 N-gram 地址**。
- 从可学习的记忆表中检索与该地址相关的**残差信号（residual signals）**，用于增强当前表示。
- 这些记忆信号是**可重用的**，特别适用于反复出现的功能性局部序列模式（如酶活性位点、糖基化位点等）。

#### **(2) Sparse Expert Routing via MoE Upcycling（稀疏专家路由）**
- 将部分 FFN 层“升级回收”为 **Mixture-of-Experts (MoE)** 结构，包含一个共享专家和多个路由专家。
- 使用 top-1 路由器根据输入上下文选择激活哪个专家，实现**残基依赖的动态计算分配**。
- 只有子集参数被激活，保持低活跃参数量的同时扩展总容量。

> ✅ **初始化设计保证稳定性**：记忆模块和 MoE 权重均从原始 ESM2 权重初始化，并零初始化新增组件，确保继续预训练时函数不变，支持稳定微调。

### **相比现有方法的优势**
| 维度 | ProtLingo 的优势 |
|------|------------------|
| **效率** | 仅使用 ~153M 活跃参数，远低于 ESM2-650M（650M）、RITA XL（1.2B）等大模型 |
| **性能** | 在突变效应预测（DMS）上超越同规模甚至更大模型，达到 SOTA 参数效率 |
| **表示保留** | 在长程接触预测任务中表现接近 ESM2-150M，说明未破坏原有结构相关表示 |
| **机制互补** | 同时引入**条件记忆**与**条件计算**，分别针对局部模式复用与计算资源适配 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 用途 | 描述 |
|-------|------|------|
| **UniRef50 & UniRef90** | 预训练 | 采用 80% UniRef50 + 20% UniRef90 混合采样策略，兼顾家族多样性与自然丰度分布 |
| **ProteinGym v1.3** | 下游评估 | 包含 217 个 DMS 实验，约 240 万个突变体的功能表型测量值，用于零样本突变效应预测 |
| **FLIP Benchmark** | 下游评估 | 包括 GB1（结合）、AAV（衣壳适应性）、Meltome（热稳定性）三个经典蛋白景观任务 |
| **CASP15** | 接触图预测 | 用于评估是否保留长距离结构耦合信息 |

### **实验设置与评估指标**

#### **训练设置**
- **骨干模型**：基于 `ESM2-150M`（30 层，hidden size=640）
- **插入模块位置**：
  - **LNgram Memory**：第 1、11、21 层前
  - **MoE Upcycling**：第 3–29 层的 FFN 替换为 1 共享 + 4 路由专家
- **继续预训练**：40k 步 MLM 任务，不使用任何标签监督
- **优化器**：Muon（主干）+ AdamW（其他），分层学习率（记忆表更高）

#### **评估指标**
| 任务 | 主要指标 |
|------|---------|
| **Fitness / Mutation Prediction** | Spearman 相关性（ProteinGym）、AUC、FLIP 任务平均 Spearman |
| **Contact Prediction** | P@L, P@L/2, P@L/5（Top-L 精确率） |
| **语言建模能力** | MLM Loss / Perplexity |
| **效率衡量** | **DMS Spearman / 活跃参数（B）** → 参数效率指标 |

### **基线方法对比**
| 模型 | 类型 | 参数量（活跃） | 特点 |
|------|------|----------------|------|
| ESM2-150M | Bidirectional MLM | 150M | 同规模直接基线 |
| ESM2-650M | Bidirectional MLM | 650M | 密集缩放基线 |
| ESM-1b | Bidirectional MLM | 650M | 上一代大模型 |
| ProtBert | BERT-style | 420M | 不同训练流程 |
| CARP-640M | Dilated Conv-based | 640M | 非 Transformer 架构 |
| RITA XL | Autoregressive LM | 1.2B | 最大因果模型 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### **(1) Protein Fitness & Mutation Effect Prediction（Table 1）**

| Model | Active Params | Efficiency (ρ/B) | PG-DMS ρ | AUC | FLIP (GB1/AAV/Meltome) |
|-------|---------------|------------------|----------|-----|------------------------|
| **ProtLingo** | **153M** | **2.61** | **0.399** | **0.720** | 0.544 / 0.576 / 0.650 |
| ESM2-150M | 150M | 2.59 | 0.388 | 0.715 | 0.523 / 0.555 / 0.651 |
| ESM2-650M | 650M | 0.62 | 0.403 | 0.723 | 0.555 / 0.606 / 0.676 |
| RITA XL | 1.2B | 0.31 | 0.373 | 0.708 | — |

> 🔍 **结论**：
> - ProtLingo 在 **DMS Spearman** 上超过 ESM2-150M（↑0.011），接近 ESM2-650M；
> - **参数效率最高**（2.61 vs 2.59），优于所有更大模型；
> - 在 FLIP 上表现稳健，略低于大模型但显著优于小模型。

#### **(2) Supervised Contact Prediction（Table 2 & Figure 3）**

| Model | Active Params | P@L | P@L/2 | P@L/5 |
|-------|---------------|-----|-------|-------|
| **ProtLingo** | **153M** | **0.485** | **0.657** | **0.816** |
| ESM2-150M | 150M | 0.484 | 0.657 | 0.830 |
| ESM2-650M | 650M | 0.514 | 0.682 | 0.845 |
| ProtBert | 420M | 0.281 | 0.353 | 0.435 |

> 🔍 **结论**：
> - ProtLingo 在 P@L 和 P@L/2 上**持平或略超 ESM2-150M**；
> - 明显优于 ProtBert，说明其保留了更强的结构相关信息；
> - 表明新增模块**未破坏原有的长程依赖建模能力**。

#### **(3) Ablation Study（消融实验，Table 3）**

| Variant | Δ PG-DMS ρ ↓ | Δ PPL ↑ |
|--------|--------------|---------|
| Full ProtLingo | — | — |
| -MoE | -0.093 | +2.461 |
| -Lngram | -0.357 | +6.003 |
| -MoE -Lngram | -0.369 | +6.035 |

> 🔍 **结论**：
> - 移除任一组件都会导致性能下降；
> - **LNgram 记忆模块贡献更大**（Δρ=-0.357），表明局部上下文记忆对突变预测至关重要；
> - 两者具有**互补增益**，联合使用效果最佳。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **条件记忆与专家路由可有效提升小型 PLM 性能**：无需大幅增加模型尺寸，即可在突变敏感任务上取得竞争性甚至领先的表现。
2. ✅ **局部上下文记忆具有生物学意义**：
   - 功能性 motif（如 DEAD-box helicase）在记忆地址空间中高度集中（>99% vs 控制组 <27%）；
   - 非保守突变更容易引起记忆残差变化，说明系统对功能扰动敏感。
3. ✅ **稀疏专家展现出生物特异性偏好**：
   - 不同专家在特定结构/功能区域（如 transmembrane、active site）表现出显著路由倾向；
   - 存在层级化分工趋势：浅层识别家族骨架，深层整合功能上下文。
4. ✅ **高效且兼容性强**：可在 ESM2-150M 基础上增量改进，适合部署于资源受限场景。

### **局限性**
- 当前模块设计较轻量，对 ESM2-150M 的提升仍有上限；
- MoE 路由器为简单的 top-1，未探索更复杂的约束路由（如 hierarchical 或 instance-conditioned）；
- 混合采样策略（80/20）未经充分验证，可能影响泛化性。

### **未来工作方向**
- 探索更丰富的路由机制（如 HI-MoE）以进一步释放条件专业化潜力；
- 引入多模态信息（结构、进化 MSA）与条件记忆协同；
- 扩展至生成任务（如定向进化模拟）；
- 开源代码与 checkpoint（文中承诺发布）。

---

> 📌 **一句话总结**：  
> **ProtLingo 通过引入可重用的局部上下文记忆（LNgram）与稀疏专家路由（MoE upcycling），在仅 153M 活跃参数下实现了媲美甚至超越大模型的突变效应预测性能，同时保持了良好的结构表示能力，为高效蛋白质建模提供了新范式。**

</details>

---

### 12. [GreenPipe: Power Modeling for Containerized DNN Inference on Kubernetes Edge Nodes](https://arxiv.org/abs/2609.04952)

**Authors**: Mengxue Wang, Peini Liu, Amir Taherkordi, Jordi Guitart  
**Category**: cs.DC  
**Published**: 2026-09-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.04952v1  

#### Abstract
Distributed DNN inference is increasingly deployed in containerized edge-cloud environments, where workloads run on-device or are exposed to remote clients over the network. Accurate online power estimation on resource-constrained ARM nodes without hardware power counters such as RAPL remains a chal...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*GreenPipe: Power Modeling for Containerized DNN Inference on Kubernetes Edge Nodes*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在边缘-云协同环境中，分布式 DNN 推理越来越多地以容器化方式部署在资源受限的 ARM 架构边缘节点（如 Raspberry Pi）上。然而，这些设备通常**缺乏硬件功耗计数器**（如 RAPL），且传统仅基于 CPU 利用率的软件功耗模型无法准确捕捉多资源（CPU、内存、网络、磁盘等）并发行为对功耗的影响。此外，Kubernetes 容器编排环境下，如何将**节点级功耗合理归因到各个容器**也是一个挑战。

### 🚀 提出的新方法：GreenPipe
作者提出 **GreenPipe** —— 一个面向 Kubernetes 边缘节点的自动化、数据驱动的功耗建模流水线，其核心创新包括：

- **多资源感知的训练基准设计**：结合 micro-benchmarks（stress-ng, iperf）、组合负载和 DNN 内核级负载（DeepBench），覆盖真实 DNN 推理中涉及的 CPU、memory、disk、network 多维资源活动。
- **外部电表标注 + 回归建模**：使用外部功率计（UM25C）采集真实功耗标签，训练多种回归模型（Linear, RF, XGBoost 等），实现无需硬件支持的高精度功耗估计。
- **运行时容器级功耗归因机制**：提出一种启发式方法，将节点级预测功耗分解为 idle 和 dynamic 成分，并按容器资源使用比例进行分配。
- **端到端 Kubernetes 集成**：完整支持从离线训练、验证到在线实时预测的全流程，在 K3s 集群中部署为 DaemonSet + Sidecar 模式。

### 🔍 相比现有方法的优势
| 对比维度 | 现有方法（如 Kepler, PowerAPI, SmartWatts） | GreenPipe |
|--------|------------------------------------------|---------|
| 硬件依赖 | 依赖 RAPL/hwmon 等硬件接口，不适用于大多数 ARM 设备 | 不依赖硬件功耗计数器，适用于无 RAPL 的边缘设备 |
| 特征粒度 | 多为 CPU-centric 或利用率单一特征 | 使用 PMU/eBPF 采集多资源细粒度指标（cache miss, disk IO, net IRQ 等） |
| 训练数据 | 基于 stress-CPU 或通用负载，难以反映 DNN 行为 | 包含 DNN-targeted 工作负载（DeepBench），更贴近实际推理场景 |
| 容器归因 | 依赖底层硬件支持或复杂校准流程 | 提出轻量级启发式归因策略，可在运行时完成容器级估算 |
| 可部署性 | 多停留在研究原型阶段 | 实现了完整的 Kubernetes 原生集成与在线服务 |

---

## 2. 核心实验方法和设置

### 📊 数据集与工作负载
- **训练数据来源**：通过以下三类容器化 benchmark 自动生成，共生成 **13,938 个样本**（1Hz 采样）：
  - **Micro-benchmarks**：`stress-ng`（CPU、内存、磁盘压力测试）、`iperf`（网络传输）
  - **Combined benchmarks**：同时施压多个组件（如 CPU+mem+disk 或全系统负载）
  - **DNN-targeted benchmarks**：来自 [DeepBench](https://github.com/baidu-research/DeepBench)，包括 gemmbench（密集/稀疏矩阵乘）、convbench（卷积计算）

- **验证数据集（Validation Benchmarks）**：
  - **模型**：MobileNetV2, EfficientNetB0, ResNetV2
  - **格式与精度**：
    - `pb` 模型（TensorFlow SavedModel, float32）
    - `tflite` 模型（LiteRT 转换，支持 float32, float16, int8）
  - **输入数据**：ImageNet ILSVRC2012 validation set 子集
  - **量化校准集**：MLPerf 提供的 calibration dataset
  - **部署模式**：
    - **Local inference**：本地批量处理，使用 LiteRT 引擎
    - **Serving inference**：远程请求，通过 gRPC API 调用 TensorFlow Serving

> ⚠️ 注意：所有验证 trace **未参与训练**，确保评估公正性。

### ⚙️ 实验平台设置
| 组件 | 配置 |
|------|------|
| **Edge Node** | Raspberry Pi 4 Model B Rev 1.5（BCM2711, 4×Cortex-A72 @1.8GHz） |
| **Server Node** | Intel Core i7-8650U ×8 @1.90GHz |
| **OS** | Edge: Debian 12 (kernel 6.6.56-v8+)；Server: Ubuntu 22.04 |
| **Kubernetes** | K3s v1.30.3+k3s1（轻量级发行版） |
| **Container Runtime** | containerd://1.7.17-k3s1，cgroup v2 |
| **监控系统** | Prometheus + Grafana，通过自定义 ResourceMonitor（Go + eBPF）采集指标 |
| **功耗测量** | Ruideng UM25C USB 功率计（采样频率 1Hz） |

### 📈 评估指标
- **MAE**（Mean Absolute Error）：平均绝对误差，单位 W
- **MAPE**（Mean Absolute Percentage Error）：平均绝对百分比误差
- **Latency-Energy Trade-off**：推理延迟 vs. 单次推理能耗分析
- **Container Attribution Visualization**：展示容器级功耗分布合理性

### 🔁 基线方法对比
1. **Baseline LR [7]**：仅使用 CPU utilization 的线性模型 $ P = 4.5344 \times U + 2.2857 $
2. **Baseline Training [8]**：仅使用 CPU stress 工作负载训练的多特征模型（但缺少 memory/network/disk）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 3）
| 模型 | MAE (W) | MAPE (%) |
|------|--------|---------|
| **GreenPipe 最佳模型（PR）** | **0.29** | **6.3%** |
| GreenPipe 平均 MAPE | — | **6.3–9.4%** |
| Baseline Training 平均 MAPE | — | 10.8–14.2% |
| Baseline LR | — | 14.2% |

✅ **GreenPipe 相比基线显著提升**：
- 相比 **Baseline Training**：平均 MAPE 下降 **26.9%**，MAE 下降 **46.9%**
- 所有 GreenPipe 模型均优于 Baseline LR，证明非 CPU 特征的重要性

### 🔍 消融与影响因素分析（Section 6.2）
#### （1）DNN 模型架构与精度
- MobileNet 推理预测误差略高（MAPE 6.6–9.2%），可能因其轻量化结构导致资源行为更复杂
- 浮点推理（float32/float16）比整数量化（int8）更容易建模
- 非线性模型（如 DT, RF, GB）表现优于线性模型，说明特征与功耗间存在非线性关系

#### （2）线程数（Parallelism）
- 随着推理线程数增加，预测误差略有上升（Fig. 4），可能是由于多线程调度引入不确定性
- 但仍保持在可接受范围内（<10% MAPE）

#### （3）推理引擎与部署模式
- **pb 模型（TF Serving）** 预测效果好于 **tflite 模型（LiteRT）**
- tflite 实际功耗普遍高于预测值 → 可能因 LiteRT 更深度优化，引发未被完全捕获的低层资源访问
- Serving 场景（网络请求）比 Local 场景更具挑战性，但 GreenPipe 仍有效（Fig. 5）

### 💡 在线部署与容器归因结果
- 容器级功耗归因显示：**inference workload 占主导**，exporter/estimator 容器开销 <2%
- 功耗分布与 `cpu_time` 趋势一致但不完全相同，体现了 idle power 和多资源加权的作用
- 支持每秒级实时预测，满足在线调度需求

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **多资源训练数据至关重要**：相比仅用 CPU stress 的训练方式，引入 memory、disk、network 和 DNN 内核负载可大幅提升模型准确性（↓26.9% MAPE）。
2. **回归器选择影响较小**：在充分覆盖的训练数据下，不同回归算法之间的差异远小于与基线的差距，表明**数据质量比模型结构更重要**。
3. **部署模式与并行度是主要误差源**：thread count、inference engine 类型和 local/serving 模式对预测难度影响最大，而模型架构和精度影响相对有限。
4. **精度与延迟并非总是正相关**：int8 推理最快且最节能；但增加线程虽降低延迟，却可能因功耗上升而导致总能耗增加（尤其大模型）。
5. **启发式容器归因可行且开销低**：尽管缺乏容器级真值验证，提出的 idle/dynamic 分解 + 加权资源分配策略能提供合理的运行时可见性。

### ⚠️ 局限性
- **容器级功耗未经过独立验证**：外部电表仅提供节点级 ground truth，容器级归因仅为启发式估算。
- **未考虑 GPU 或 NPU**：当前工作聚焦于 CPU-only ARM 节点，未涵盖带加速器的异构边缘设备。
- **DVFS 动态调频影响未深入建模**：虽然启用了 ondemand governor，但频率变化对功耗的非线性影响尚未显式建模。
- **跨平台泛化能力未知**：模型针对 Raspberry Pi 4 训练，是否适用于其他 ARM SoC 尚需验证。

### 🔮 未来工作方向
1. 扩展至更多边缘平台（如 Jetson, ODROID）及带 GPU/NPU 的设备
2. 引入周期性电表辅助重校准机制，适应环境漂移和老化效应
3. 结合 DVFS 状态信息，构建更精细的动态功耗模型
4. 探索基于强化学习的能量感知调度策略，利用 GreenPipe 输出进行优化决策
5. 开发标准化的 Kubernetes Power Operator，实现绿色边缘集群自动化管理

--- 

> 📌 总结：**GreenPipe 是首个面向 Kubernetes 边缘节点、支持容器化 DNN 推理的全流程功耗建模框架**，它通过高质量多资源训练数据和轻量级运行时集成，在无硬件支持的 ARM 设备上实现了 **6.3–9.4% MAPE** 的高精度预测，并揭示了边缘推理中的关键 **latency-energy trade-offs**，为绿色边缘智能提供了实用工具链。

</details>

---

### 13. [Iris: Climbing to the Search Frontier](https://arxiv.org/abs/2609.04304)

**Authors**: Ziyuan Liu, Hengqi Liu, Zichuan Wang, Yang Qin, Jiachen Liang, Xu Chu, Shaowei Chen, Yuantao Gu, Mu Chuan  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.04304v1  

#### Abstract
We present Iris-mini and Iris-pro, two search agents trained at the 35B-A3B and 397B-A17B scales, together with the data pipeline and training recipe behind them. Tasks are reverse-constructed from the hyperlink structure of a web corpus: we author multi-hop chains over an entity graph distilled fro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Iris: Climbing to the Search Frontier**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前的 **search agent**（搜索代理）在执行多跳、长视野的信息检索任务时面临以下挑战：
- 自然存在的网页问题通常过于简单，无法有效训练强大的搜索能力；
- 模型容易通过字符串匹配（string matching）绕过推理过程，导致训练数据质量不高；
- 推理时的 **context management (CM)** 对性能影响巨大，但其效果常被误归因于模型策略本身；
- 缺乏端到端可控、可复现的训练与评估流程。

本文旨在构建一个能够进行复杂、抗捷径（shortcut-resistant）、基于真实网络交互的 search agent，并明确区分模型能力与推理时机制的影响。

---

### **提出了什么新方法或新思路**

#### ✅ **端到端的数据构造与训练流水线**
提出了一套完整的 **data pipeline + training recipe + evaluation protocol**，涵盖从任务生成到强化学习优化的全过程。

#### ✅ **基于图结构的反向任务构造（Reverse-Constructed Multi-Hop Tasks）**
- 从网页语料库的超链接结构中提取实体图（entity graph），并在此基础上构造多跳推理路径；
- 将非答案实体替换为描述性引用（descriptive reference），消除直接字符串匹配的可能性，迫使模型必须通过推理消歧。

#### ✅ **双重验证标准（Dual-Criteria Verification）**
仅保留满足两个条件的问题：
- **困难性（Difficulty）**：闭卷模式下参考模型无法回答；
- **可解性（Solvability）**：提供支持证据后能正确解答。

确保训练数据既具有挑战性又客观可验证。

#### ✅ **两阶段轨迹过滤机制**
- **粗粒度过滤（Coarse Filtering）**：按轨迹级别筛选正确的、非退化的、足够深度的交互轨迹；
- **细粒度掩码（Fine Filtering）**：使用 LLM judge 在 turn 级别判断是否保留某一步输出，最多掩码 10%，提升监督信号纯净度。

#### ✅ **SFT-RL Climb（迭代式SFT与RL交替训练）**
- 先用高质量轨迹进行 Supervised Fine-Tuning（SFT）；
- 再通过 Reinforcement Learning（RL）在真实搜索引擎上探索；
- 将 RL 中发现的“最难且最高效的解决路径”回流至下一阶段的 SFT；
- 形成闭环爬升（climbing）过程，逐步提升策略强度。

#### ✅ **推理时 Context Management 的显式控制与评估**
首次系统性地将 CM 视为独立变量，在相同工具集、上下文长度和 judge 下，分别报告 **启用 CM** 和 **禁用 CM** 的结果，剥离其对性能增益的影响。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **数据质量** | 构造的任务抗捷径、难度可控、自动验证，优于人工标注或天然查询 |
| **训练效率** | Partial rollout + prefix reuse 提高长序列采样效率，避免资源浪费 |
| **系统集成性** | Judge 与 Summarizer 部署在训练集群内（in-cluster），不依赖外部API |
| **评估公正性** | 明确分离 CM 效应，避免“靠推理时技巧刷分”的误导性比较 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
四个主流 agentic search benchmark：
- **BrowseComp**：测试识别长尾实体的能力，基于间接线索综合推理；
- **BrowseComp-ZH**：BrowseComp 的中文版本，聚焦中文网页源；
- **DeepSearchQA**：衡量答案的全面性（F1），而非单一答案正确与否；
- **Humanity's Last Exam (HLE)**：跨学科专家级学术推理，强调领域知识与检索结合。

此外还使用了部分开源与内部合成数据集用于训练。

---

### **实验设置和评估指标**

#### **模型架构**
- **Iris-mini**：基于 Qwen3.6-35B-A3B 初始化，MoE 结构，256K 上下文窗口；
- **Iris-pro**：基于 Qwen3.5-397B-A17B 初始化，更大规模 MoE。

#### **训练流程**
1. **SFT阶段**：
   - 使用强教师模型（MT）生成 ReAct 轨迹；
   - 经双层过滤后进行有监督微调；
   - Batch size=64，最大序列长度=262,144 tokens，训练2轮。
2. **RL阶段**：
   - 使用 Relax 框架进行 group-relative policy gradient 优化；
   - Reward 来自内部部署的 Qwen3.5-397B-A17B 作为 GenRM；
   - Observation summarizer 同样由该模型提供；
   - 支持中断与前缀重用（partial rollout），提高训练吞吐。

#### **评估协议**
- 所有问题执行 **pass@1** 单次 rollout；
- 工具集固定：`SEARCH`, `SCRAPE`；
- 最大 turn 数与 context 长度统一；
- 使用官方 LLM judge 进行评分：
  - BrowseComp / BrowseComp-ZH / HLE：Accuracy；
  - DeepSearchQA：F1 Score。

#### **Context Management 设置**
- **无 CM（w/o CM）**：原始历史累积，直到上下文满；
- **discard-all**：接近上限时清空全部对话历史，重新开始；
- **retry**：失败后总结失败经验，附加到新尝试中；
- **discard-all + retry**：组合策略。

---

### **基线方法对比**
与其他公开的 search agent 对比，包括：
- MiroThinker 系列
- Apodex 系列
- Nex-N2 系列
- XYZ-Aquila 系列
- FORT-Searcher, REDSearcher
- 前沿闭源系统如 GPT-5.6 Sol, Claude Fable 5, Kimi-K3 等

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（CM启用，默认discard-all）**

| Model | Size | BrowseComp | BrowseComp-ZH | DeepSearchQA (F1) | HLE |
|-------|------|------------|----------------|--------------------|-----|
| **Iris-mini** | 35B | **82.2** | **84.8** | **86.9** | **52.3** |
| **Iris-pro** | 397B | **88.6** | **85.1** | **92.9** | **56.4** |

> 在各自参数范围内，**Iris 系列取得当前最优整体表现**。

---

### **与基线方法的对比结果**

#### 📌 **Iris-mini vs 同规模模型（~35B）**
- 超越最强同级模型 **XYZ-Aquila-mini**：
  - BrowseComp：+3.4 pts（82.2 vs 78.8）
  - HLE：+1.2 pts（52.3 vs 51.1）
- 在 DeepSearchQA 上略低于 XYZ-Aquila-mini（86.9 vs 89.5）

#### 📌 **Iris-pro vs 同规模模型（~400B）**
- 全面领先：
  - BrowseComp：+3.8 pts（88.6 vs 84.8）
  - DeepSearchQA：+0.4 pts（92.9 vs 92.5）
  - HLE：+3.1 pts（56.4 vs 53.3）
- 在 BrowseComp-ZH 上持平（均为 85.1）

#### 📌 **与超大规模模型比较**
- Iris-mini 接近甚至超越千亿级以上模型：
  - 在 BrowseComp 上超过 Kimi-K2.6（83.2）和 DeepSeek-V4-Pro（83.4）；
- Iris-pro 表现媲美 MiroThinker-H1 和 Apodex-1.0-H 等“heavy-compute”配置下的高端系统。

---

### **消融实验结果（Table 2: Effect of CM）**

| 设置 | BrowseComp (+Δ) | BrowseComp-ZH (+Δ) | DeepSearchQA (+Δ) | HLE (+Δ) |
|------|------------------|---------------------|--------------------|----------|
| **Iris-mini w/o CM** | 64.7 | 72.3 | 81.0 | 43.2 |
| → discard-all | +17.5 | +12.5 | +5.9 | +9.1 |
| → discard-all + retry | +21.2 | +12.8 | +8.9 | +9.2 |
| **Iris-pro w/o CM** | 72.6 | 76.8 | 86.4 | 50.8 |
| → discard-all | +16.0 | +8.3 | +6.5 | +5.6 |
| → discard-all + retry | +17.7 | +8.3 | +7.0 | +5.8 |

#### 🔍 发现：
- **CM 带来的增益显著**，尤其对小模型更明显（Iris-mini 提升高达 21.2 pts）；
- 增益大小与任务特性相关：
  - BrowseComp（长程检索密集）受益最大；
  - HLE（知识主导）受益较小；
- **retry 策略虽有效，但代价高昂**，作者认为不应作为主报告配置。

---

## **4. 关键结论和发现**

### **主要发现**

1. ✅ **有效的训练数据设计是核心驱动力**
   - 反向构造 + 实体抽象 + 双重验证 的数据流程显著提升了训练质量；
   - 模型学到的是真正的“推理-检索”协同行为，而非记忆或模式匹配。

2. ✅ **SFT-RL Climb 是高效的学习范式**
   - 将 RL 探索中的成功案例反馈给 SFT，形成正向循环；
   - 比纯 RL 或纯 SFT 更稳定、收敛更快。

3. ✅ **Context Management 是不可忽视的性能放大器**
   - 性能差异中相当一部分来自 CM 策略，而非模型本身；
   - 报告“无 CM”结果有助于公平评估模型内在能力。

4. ✅ **Iris 展现出卓越的性价比**
   - 35B 模型达到接近万亿参数模型水平；
   - 证明通过专业化训练可弥补参数量差距。

5. ⚠️ **存在 ground-truth 不一致问题**
   - 如 Appendix A 所示，BrowseComp-ZH 第85题中官方标签为 “Lannister”，但依据剧情应为 “Bolton”；
   - 揭示当前 benchmark 注释可能存在错误，需更高质标注。

---

### **方法的局限性**

| 局限 | 说明 |
|------|------|
| **未使用子代理或多模块协作** | 当前为单 ReAct agent，无 test-time verification 或 reanswer 机制，可能限制极限性能 |
| **依赖特定工具接口** | 工具集固定为 SEARCH/SCRAPE，泛化性有待验证 |
| **CM 成本未完全计入** | retry 类策略虽提分，但增加推理延迟与计算开销 |
| **尚未覆盖所有 agentic 场景** | 当前聚焦 search，其他工具使用场景仍在探索中 |

---

### **未来工作方向**

1. **扩展至通用 Tool Use 场景**
   - 初步实验显示，Iris 在 BFCL、t-bench、OfficeQA 等通用工具任务上也有迁移能力；
   - 搜索可能是一种原子能力（atomic capability），可用于增强整体 agent competence。

2. **开发更高质量的 benchmark**
   - 当前 benchmark 存在标注错误风险；
   - 计划推出覆盖更广、注释更可靠的新型评测集。

3. **将 search 数据融入全流程训练**
   - 不仅用于 specialization，也可作为 pretraining 或 mid-training 的通用能力催化剂。

4. **开放生态建设**
   - 承诺发布模型权重及完整训练/评估 pipeline（data construction, training, evaluation recipes），推动社区复现与进步。

---

> **总结一句话**：  
> Iris 通过一套精心设计的端到端训练体系，在有限参数下实现了顶尖的搜索智能，同时揭示了 context management 对性能的巨大影响，为 search agent 的研发提供了可复现、可分析的新范式。

</details>

---

### 14. [CIERA: Cross-Iteration Exponent Reuse for Lossless Allgather in Sharded MoE Training](https://arxiv.org/abs/2609.04609)

**Authors**: Ali Zafar Sadiq, Haiying Shen, Masahiro Tanaka  
**Category**: cs.DC  
**Published**: 2026-09-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.04609v1  

#### Abstract
In training Mixture-of-Experts (MoE) models, sharded data parallelism partitions each expert's parameters across GPUs, requiring an Allgather operation to reconstruct the full weight matrix before each layer executes. This communication often dominates iteration time. Prior work often reduces this o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# CIERA: Cross-Iteration Exponent Reuse for Lossless Allgather in Sharded MoE Training —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **sharded MoE (Mixture-of-Experts)** 模型训练中，采用 **sharded data parallelism** 时，每个专家的参数被分片存储在不同 GPU 上。每次前向/反向传播前需通过 **Allgather** 操作重建完整权重矩阵，这一通信开销常成为训练迭代的瓶颈。

现有方法存在以下问题：
- **Lossy 压缩方法**（如 ZeRO++、gZCCL）虽然减少通信量，但引入数值误差，影响训练精度。
- **Lossless 方法** 要么压缩效率低，要么无法有效隐藏压缩开销，甚至拖慢训练。

### 🚀 提出的新方法：CIERA
提出 **Cross-Iteration Exponent Reuse Allgather (CIERA)**，一种面向 sharded MoE 训练的 **无损、系统感知的通信优化方法**，基于以下三个核心组件：

1. **Exponent Reuse-based Compression (ER)**  
   观察到：在短暂 warmup 后，绝大多数权重的 **浮点数 exponent 字段跨迭代保持不变**（>99%）。  
   → 因此，仅传输 **sign 和 mantissa**，本地缓存 exponent，实现无损压缩。

2. **Benefit-driven Selective Compression (BSC)**  
   不同类型的 shard（如 FFN、Attention、LayerNorm）的 exponent 变化率差异巨大。  
   → 仅对 **高收益 shard**（如 LayerNorm、部分 FFN）进行压缩，跳过变化频繁的 shard（如 Attention），避免负优化。

3. **Computation-Communication Pipelining (CCP)**  
   利用 **PyTorch FX graph** 重写执行流程，将 exponent 检查、压缩/解压操作提前并行于前一层计算或通信阶段，**仅保留少量接收端 decompression 在关键路径上**，有效隐藏开销。

### 🔍 相比现有方法的优势
| 维度 | CIERA | 现有方法 |
|------|-------|--------|
| **保真性** | ✅ Bitwise-exact 无损重构 | ❌ Lossy 方法引入数值误差 |
| **压缩效率** | 高（利用跨迭代 exponent 稳定性） | 低或不可持续（如 block quantization） |
| **系统友好性** | 与计算/通信重叠，开销隐藏好 | 压缩本身成瓶颈 |
| **自适应性** | 动态选择可压缩 shard | 全局统一处理 |

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **训练数据集**：
  - 主要使用：**AG News**（用于基准测试）
  - 验证泛化性：**OpenWebText**
- **评估的 MoE 模型**（共6个）：
  - OLMoE-1B-7B
  - DeepSeek-MoE-16B
  - MiniCPM-MoE-8×2B
  - Qwen2-57B-A14B
  - Mixtral-8×7B
  - Llama-4-Scout-17B-16E

### ⚙️ 实验设置
- **硬件平台**：
  - 单节点：4 或 8 × NVIDIA A100-80GB（NVLink 3.0，600 GB/s）
  - 多节点：最多模拟至 128 GPUs（跨节点使用 200 Gbps InfiniBand，带宽 ~8 GB/s）
- **精度格式**：BF16 和 FP16
- **序列长度 (S)**：1024
- **层数 (L)**：默认 4 层，敏感性分析中变化 L 和 S
- **块大小 (B)**：默认 512（权衡 reuse rate 与 kernel 开销）

### 📊 评估指标
- **平均迭代时间**（post-warmup 1000 iterations）
- **速度提升倍数**（Speedup over baseline）
- **Allgather 占通信时间比例**
- **exponent change rate**
- **bitwise 参数一致性验证**

### 🆚 基线方法对比
| 基线 | 类型 | 特点 |
|------|------|------|
| **ZeRO-3** | Lossless | 官方 DeepSpeed 实现，全精度传输 |
| **FSDP** | Lossless | PyTorch 原生 FSDP 实现 |
| **ZeRO++** | Lossy | 使用 int8 Allgather + int4 ReduceScatter，有损量化 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & 2）

#### 在 **OLMoE-1B-7B** 上的表现（16 GPUs）：
| 对比项 | Speedup |
|--------|---------|
| vs. ZeRO-3 (BF16, lossless) | **3.70×** |
| vs. ZeRO++ (FP16, lossy) | **3.68×** |

#### 扩展至 128 GPUs 的预测性能：
| 对比项 | Predicted Speedup |
|--------|------------------|
| vs. ZeRO-3 | **4.28×** |
| vs. ZeRO++ | **4.42×** |

> 💡 **说明**：随着规模扩大，跨节点通信占比上升，CIERA 的优势进一步放大。

#### 其他模型表现（16 GPUs）：
| Model | vs. ZeRO-3 | vs. ZeRO++ |
|-------|-----------|------------|
| MiniCPM-8×2B | 2.12× | 2.60× |
| DeepSeek-MoE-16B | 3.34× | 1.93× |
| Qwen2-57B-A14B | 2.66× | 3.08× |
| Mixtral-8×7B | 1.33× | 1.67× |
| Llama-4-Scout-17B | 1.16× | 2.03× |

> 注：Scout 在小规模下收益低，因 NVLink 内部带宽高且 shard change rate 高，压缩不划算；但在多节点场景下仍可达 1.28×（vs. ZeRO-3）。

### 🔬 消融实验结果（Table 3，4 GPUs, BF16）

以 **OLMoE-1B-7B** 为例（ZeRO-3 迭代时间：1.840s）：

| 配置 | Speedup |
|------|--------|
| +ER (启用指数复用) | 1.76× |
| +ER + BSC (加入选择性压缩) | 2.84× |
| +ER + BSC + CCP (加入流水线) | **2.89×** |

> ✅ **结论**：三者协同显著增益，其中 **BSC 是最大贡献者**，避免了对高变化率 shard 的无效压缩。

### 📉 敏感性分析（Figure 9）
- **层数增加** → Speedup 提升（更多 Allgather 调用）
- **序列长度增加** → Speedup 下降（计算占比上升，通信优化空间缩小）
- **块大小 B=512 最优**，过小（kernel 开销大）或过大（reuse 率下降）均不利

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Exponent Stability 是真实存在的**：warmup 后 >99% 权重 exponent 不变，为无损压缩提供基础。
2. **Shard-level reuse 不够细粒度**：多数 shard 每步都变，但 **block-level reuse（B=512）可实现 84–99% 复用率**。
3. **Not all shards are equal**：LayerNorm 极其稳定，expert FFN 在某些模型中也稳定，而 Attention/Routing shards 几乎每步都变 → 必须 **selective compression**。
4. **Compression must be pipelined**：否则检查和压缩开销可达 25–30%，通过 FX graph 调度可几乎完全隐藏。
5. **CIERA 实现了真正的“无损加速”**：相比 lossy 方法更快，同时保证 **bitwise-exact 参数重建**。

### ⚠️ 方法的局限性
- **主要适用于 MoE 模型**：dense LLM 中 exponent 变化更频繁，BSC 会剪掉大部分 shard，收益有限。
- **依赖 warmup 阶段后的稳定性**：若学习率过高或训练动态剧烈，exponent 变化率可能上升。
- **当前 BSC 是静态配置**：未支持运行时动态调整压缩策略。
- **大规模实测仅到 16 GPUs**：32–128 GPU 结果为 trace-driven 模拟，虽锚定实测点，但仍需谨慎解读。

### 🔮 未来工作方向
- 支持 **online adaptive BSC controller**，动态调整压缩策略。
- 将 CIERA 思路扩展至 **其他通信原语**（如 ReduceScatter）。
- 探索在 **dense LLM 微调阶段** 是否也能利用 exponent 稳定性。
- 多节点实测验证 32+ GPU 场景下的实际性能。

---

## 🌍 Broader Impacts
- **积极影响**：降低 MoE 训练的通信成本，节省 GPU 时间、能耗和经济成本，助力资源受限团队训练更大模型。
- **潜在风险**：提高训练效率可能间接刺激更大规模模型的部署，增加总体算力消耗，需关注 **可持续 AI 发展**。

> ✅ **总结一句话**：  
> **CIERA 通过跨迭代 exponent 复用 + 自适应选择 + 流水线调度，在不牺牲任何数值精度的前提下，实现了高达 3.7× 的训练加速，是 sharded MoE 训练通信优化的重要进展。**

</details>

---

### 15. [From 80x to 385x: A Best-Matching-Unit Search at the L2 Roof, Measured Against a Symmetrically Tuned Baseline](https://arxiv.org/abs/2609.05138)

**Authors**: Andrew James Amos  
**Category**: cs.LG  
**Published**: 2026-09-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.05138v1  

#### Abstract
Comparisons between GPU implementations are usually asymmetric: one side is tuned by its author, the other is run as found. I report a programme that tuned both a novel SOM algorithm (SparseBin) and the baseline algorithm it was being compared to (cuSPARSE). The best-matching-unit search that domina...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From 80× to 385×: A Best-Matching-Unit Search at the L2 Roof, Measured Against a Symmetrically Tuned Baseline*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **self-organizing map (SOM)** 中主导训练过程的 **best-matching-unit (BMU) search** 进行优化。传统实现中，该操作受限于读取 codebook 所需的内存带宽，尤其在处理大规模稀疏数据（如 MEDLINE 文本语料）时成为性能瓶颈。

更关键的是，作者指出当前多数算法比较存在 **不对称调优（asymmetric tuning）** 问题：研究者通常只优化自己的算法，而将基线方法“开箱即用”，导致性能差距被高估或误导。

### 提出的新方法与新思路
- **对称调优协议（Symmetrically Tuned Baseline）**  
  本文提出并严格执行一种公平比较原则：**任何在新算法上验证有效的优化杠杆（lever），都必须同样应用于基线算法**。这确保了比较建立在双方均达峰值性能的基础上，提升了结论的可信度。

- **SparseBin 算法的系统性优化**  
  在前作 [3] 提出的 `SparseBin` 算法基础上（采用 feature-major 存储格式以提升缓存复用），通过四个关键优化杠杆进一步加速 BMU 搜索：
  1. **Tile size 调整**：动态选择每块处理的文章数量，平衡寄存器占用与 L2 缓存命中率。
  2. **Tile-membership clustering**：聚类相似文章至同一 tile，减少特征并集大小，降低 L2 流量。
  3. **Neuron-axis chunking**：将神经元轴分块处理，控制工作集大小以适配 L2 缓存。
  4. **Vectorised loads (`__half2`)**：向量化加载权重，显著提升内存吞吐效率。

### 相比现有方法的优势
- **性能提升巨大**：相比原始配置，BMU 搜索速度提升 **5.6–10.1×/epoch**。
- **真实性能差距扩大**：在对基线 cuSPARSE 同样进行深度调优后，相对 MedSOM 的优势从 ~80× 提升至 **~385×**（在 128² 地图上）。
- **达到硬件极限**：最终版本的 kernel 达到 **77% 的 L2 带宽屋顶（roofline）**，表明已逼近理论上限，后续改进空间不足 1.3×。
- **方法论贡献**：强调“可复现、对称、预测先行”的科学实验范式，为系统级性能比较树立标准。

---

## 2. 核心实验方法和设置

### 数据集
- **MEDLINE corpus**：包含 **26.9 million abstracts**，表示为 sparse binary term vectors。
- 数据固定不变（frozen corpus），所有实验基于相同分割与随机种子（seed 0）。

### 实验平台
- **GPU**: 单张 **RTX 4090**（72 MB L2 cache）
- 所有调优均针对此设备特性完成，结果不具备跨架构直接迁移性。

### 评估指标
- **主要指标**：单 epoch 时间（seconds per epoch），取三次运行中位数，误差 ≤1.5%。
- **辅助指标**：
  - L2 / DRAM 带宽利用率
  - Cache hit rate
  - Occupancy
  - Warp issue efficiency
  - 内存占用（peak memory usage）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **SparseBin (novel)** | 本文优化的算法，feature-major 存储 + fused sparse-dense + argmin |
| **cuSPARSE (baseline)** | 使用 NVIDIA cuSPARSE 库构建的基线，原版存在非合并访问等问题 |
| **MedSOM** | 早期基于 CUDA 的 SOM 实现，用于历史对比 |
| **somoclu** | 多核 CPU 上的开源 SOM 实现，代表传统方案 |

> ⚠️ 注意：与 MedSOM 和 somoclu 的比较未重新调优，仅使用其发布配置，因此属于弱对比基础。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）

| Map Size | Winner | s/epoch | Peak Memory | vs Published |
|---------|--------|--------|-------------|--------------|
| 32²     | cuSPARSE | 0.24 | 2.5 GiB | 3.0× |
| 64²     | SparseBin | 0.56 | 1.5 GiB | 6.4× |
| 128²    | SparseBin | 2.06 | 3.4 GiB | 10.1× |
| 256²    | SparseBin | 7.60 | 7.2 GiB | 5.6× |
| 512²    | SparseBin | 330.2 | 20.5 GiB | 6.2× |

> 注：`vs published` 表示相对于前作 [3] 发布配置的加速比。

### 与 MedSOM 和 somoclu 的对比（Table 2）

| Map Size | vs MedSOM (tuned) | vs somoclu (tuned) |
|----------|-------------------|--------------------|
| 32²      | 121×              | 654×               |
| 64²      | 353×              | 2,235×             |
| 128²     | **385×**          | **2,994×**         |
| 256²     | 422×              | 3,643×             |
| 512²     | —                 | —                  |

> ✅ 强调：这些数字是基于 **双方均已调优后的 SparseBin vs 未经调优的 MedSOM/somoclu**，因此反映的是设计优势而非调优偏差。

### 消融实验与关键发现
- **Tile size 非单调最优**：并非越大越好；小 tile（如 2 或 4）因提高 occupancy 而更快。
- **Clustering 取代 Ordering**：显式聚类文章比调度执行顺序更能提升局部性，后者被前者取代。
- **Chunking 存在拐点**：C=8 最佳，继续增加导致 merge 开销超过缓存收益。
- **Vectorized load 改变最优解**：启用 `__half2` 后，chunking 最优点从 C=8 回退到 C=4，说明不能仅凭缓存指标选型。
- **cuSPARSE 自身可大幅优化**：其 argmin 阶段原为非合并访问，重写为 warp-per-row reduction 后提速 **12.5×**，证明基线本身也有巨大潜力。

---

## 4. 关键结论和发现

### 主要发现
1. **对称调优至关重要**：当基线也被充分优化后，性能差距依然存在甚至更大（80× → 385×），说明 `SparseBin` 的优势是真实的、结构性的。
2. **已达性能天花板**：最终 kernel 达到 **77% 的 L2 带宽 roof**，其余单元仅运行于 40–65%，意味着进一步优化空间不超过 **~1.3×**。
3. **搜索仍是主导开销**：尽管 BMU 搜索已加速近 10 倍，但在最大地图（512²）上仍占 epoch 时间的 **94.0%**，更新阶段影响微乎其微。
4. **方法论胜利**：预测注册、null 结果记录、公开 artifact 等实践增强了结果的可检验性和科学性。

### 方法的局限性
- **硬件依赖性强**：所有调优参数（tile size, chunk count, batch size）均针对 RTX 4090 的 L2 容量和内存体系定制，在其他 GPU 上需重新搜索。
- **未调优外部基线**：与 MedSOM 和 somoclu 的比较未遵循对称原则，结果可能仍受其实现质量限制。
- **静态数据假设**：实验基于冻结语料，不涉及流式或增量学习场景。

### 未来工作方向
- 探索跨多卡分布式扩展下的类似优化策略。
- 将对称调优框架推广至其他机器学习算子的公平比较中。
- 构建自动化的、支持跨实现比较的 autotuning 工具链，克服当前手动调优成本高的问题。
- 在不同架构（H100, MI300 等）上验证 L2 roof 是否仍是瓶颈，以及 tile/chunk 设计是否迁移有效。

---

> 📌 **一句话总结**：本文通过对 SparseBin 和 cuSPARSE 实施严格的对称调优，揭示了 BMU 搜索的真实性能边界，实现了从 ~80× 到 ~385× 的有效加速，并首次将 kernel 推至 L2 带宽屋顶，证明了当前设计已接近物理极限。

</details>

---

### 16. [MaxKernel: Agentic Kernel Generation for TPUs](https://arxiv.org/abs/2609.04523)

**Authors**: Shangkun Wang, Nina Cai, Charles Hoong, Julian Walker, Gerson Kroiz, George Vanica, Deepak Patil, Andi Gavrilescu, Hassan Sipra, Sethu Sankaran  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.04523v1  

#### Abstract
Designing and authoring high-performance custom kernels for accelerators is a complex task that requires deep hardware-level expertise. Large Language Models (LLM) can be leveraged together with real-time compiler feedback to build agentic systems for kernel generation. In this work, we present MaxK...

---

### 17. [Extremely Sparse Supervision Incentivizes Reasoning Ability](https://arxiv.org/abs/2609.04565)

**Authors**: Zhishuai Liu, Xingzi Xu, Mehmet Saygin Seyfioglu, Pan Xu, Karim Bouyarmane  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.04565v1  

#### Abstract
Large language models demonstrate increasingly strong reasoning capabilities through effective post-training. Yet, prevailing post-training methods optimize over massive numbers of tokens, implicitly assuming that effective learning must be token-intensive. We revisit this assumption in the on-polic...

---

### 18. [PLUME: Parameter-Efficient Personalization of Large Language Models via Low-Rank User Modulation in Shared Subspaces](https://arxiv.org/abs/2609.04715)

**Authors**: Xinyu Li, Hao Zhou, Jianfeng Zhu, Julina Maharjan, Ruixin Guo, Feodor Dragan, Ruoming Jin  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.04715v1  

#### Abstract
Personalizing large language models (LLMs) is essential for delivering AI assistance that aligns with individual users' styles, intents, and preferences. While per-user fine-tuning can substantially enhance personalization quality, it introduces significant parameter and storage overhead, limiting s...

---

### 19. [Compact Bellman-Grounded Cognitive Maps for Cost-Aware Navigation](https://arxiv.org/abs/2609.05104)

**Authors**: Yuzhe Han, Mingkun Xu, Yujie Wu  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.05104v1  

#### Abstract
Biological agents navigate familiar environments not by re-solving routes for each new goal, but by reusing a learned map built once and read off as goals change. Existing artificial cognitive-map models mimic this reuse, yet their guidance is not explicitly grounded in additive heterogeneous route ...

---

### 20. [SciDocBench: A Workflow-Centered Benchmark and Data Pipeline for Scientific Document Understanding](https://arxiv.org/abs/2609.05141)

**Authors**: Shenxi Wu, Yuhong Liu, Haosong Zhang, Tongjin Zou, Yanxun Zhang, Gaochang Chen, Dun Liang, Jiaqi Wang, Zhecan James Wang, Yuhang Zang, Dahua Lin  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.05141v1  

#### Abstract
Scientific papers require models to reason jointly over text, equations, figures, tables, code, and datasets while preserving the provenance of supporting evidence. Existing benchmarks typically evaluate these capabilities in isolation, leaving unclear whether multimodal models can support realistic...

---

### 21. [LentEx: Generalizable Latent Entity Extraction via Synthetic Data and Instruction-Tuned LLMs](https://arxiv.org/abs/2609.04511)

**Authors**: Umesh Bodhwani, Yuan Ling, Cibi Chakravarthy Senthilkumar, Shujing Dong, Yarong Feng, Hongfei Li, Ayush Goyal  
**Category**: cs.CL  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.04511v1  

#### Abstract
Latent entity extraction (LEE) tackles the challenge of identifying implicit, contextually inferred entities within free text-an area where traditional entity extraction methods fall short. In this paper, we introduce LentEx, a novel framework for latent entity extraction that leverages synthetic da...

---

### 22. [Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs](https://arxiv.org/abs/2609.04526)

**Authors**: Tung-Ling Li, Jiale Huang, Lee-Chi Wang, Janaki Ram Gotei  
**Category**: cs.CL  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.04526v1  

#### Abstract
Merging a LoRA adapter into its base model is standard deployment practice: it removes the runtime adapter's per-forward overhead and leaves a single standalone checkpoint any serving stack can load. On a native 4-bit microscaling checkpoint (NVFP4, MXFP4) that step stops being free. The merged weig...

---

### 23. [ConsensusBench: Benchmark of Consensus Nodes for LLM Reasoning via Outcome Reward Densifying](https://arxiv.org/abs/2609.04648)

**Authors**: Shi-Qi Yan, Chao-Hong Tan, Qian Chen, Wen Wang, Xiangang Li, Zhen-Hua Ling  
**Category**: cs.CL  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.04648v1  

#### Abstract
Reinforcement learning (RL) has become one of the primary paradigms for reasoning enhancement of large language models (LLMs). In particular, Group Relative Policy Optimization (GRPO) and related algorithms have demonstrated strong performance with outcome-level rewards. However, these methods depen...

---

### 24. [A Data Fusion Framework for Grounding Aerospace Surrogate Model via Experimental Wind-Tunnel Observations](https://arxiv.org/abs/2609.04267)

**Authors**: Nitin Nagesh Kulkarni, Dheeraj Vemula, Yin Yu, Peter Lyu, Juan J. Alonso  
**Category**: cs.LG  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.04267v1  

#### Abstract
Aerodynamic surrogate models trained on high-fidelity CFD data reproduce numerical predictions of both scalar outputs and entire fields accurately, yet their predictive fidelity is limited by systematic discrepancies between CFD and experimental observations. We present an experimentally grounded co...

---

### 25. [Physics-Aware Random Walk Fingerprints for Scalable Power Grid Graph Classification](https://arxiv.org/abs/2609.04943)

**Authors**: Adnan Anwar  
**Category**: cs.LG  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.04943v1  

#### Abstract
Recent benchmarks such as PowerGraph provide large collections of power-grid graphs for cascading-failure classification. Graph neural networks (GNNs) achieve strong predictive performance on this task, but typically require end-to-end training and model-specific tuning, while their latent represent...

---

### 26. [Deep Microcompression: Structured Pruning and Bit-packed Quantization for Microcontrollers](https://arxiv.org/abs/2609.05081)

**Authors**: Opegbemi Matthias Busoye, Tolulope Matthew Busoye, Eghonghon-aye Eigbe  
**Category**: cs.LG  
**Published**: 2026-09-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.05081v1  

#### Abstract
This paper introduces Deep Microcompression (DMC), a hardware-aware pipeline for deep learning inference on bare-metal microcontrollers. DMC integrates structured pruning, quantization-aware training, and fixed-length bit-packing to achieve a 55.8$\times$ weight compression ratio on LeNet-5 (98.77\%...

---

### 27. [PerfReasoning: How Well Do LLMs Reason on Hardware Performance?](https://arxiv.org/abs/2609.04476)

**Authors**: Dan Zhao, Karthikeyan Sankaralingam, Christos Kozyrakis, Qijing Huang  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.04476v1  

#### Abstract
Performance modeling is central to hardware design and software optimization, yet constructing these models requires structured reasoning about computation, data reuse, storage, and movement. We introduce PerfReasoning, a benchmark that evaluates LLMs both as direct performance reasoners and as gene...

---

### 28. [La Agente \'Optima: Towards Agentic Self-Driving Laboratories](https://arxiv.org/abs/2609.04564)

**Authors**: Marcel M\"uller, Jiaru Bai, Willi Gottstein, Abhijoy Mandal, Mohammad Nazeri, Elia Savino, Yanlin Fang, Sujoy Das, Sergio Pablo Garc\'ia Carrillo, Yeonghun Kang, Juan B. P\'erez-S\'anchez, Simone Pilon, Martin Fitzner, Timothy No\"el, Frank Gu, Varinia Bernales, Al\'an Aspuru-Guzik  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.04564v1  

#### Abstract
Self-driving laboratories (SDLs) combine automated experimentation with adaptive decision-making to accelerate scientific discovery. Their operation nevertheless often depends on human specialists who translate scientific objectives into executable closed-loop campaigns. Specialists adjust them as d...

---

### 29. [CoSkill: Joint Reinforcement Learning of Reasoning and Meta-Skill Agents for Hierarchical Skill Evolution](https://arxiv.org/abs/2609.04865)

**Authors**: Jinyuan Feng, Dongmin Li, Yiqun Chen, Yang Gao, Xing Chen, Huimu Wang, Zhiqiang Pu  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.04865v1  

#### Abstract
Skill libraries improve the sample efficiency of agentic reinforcement learning (RL) by enabling large language model (LLM) agents to reuse procedural knowledge. Yet existing paradigms exhibit structural shortcomings: they either decouple skill evolution from policy optimization or instantiate meta-...

---

### 30. [Towards Efficient Evaluation of Evolutionary Transfer Optimization: Case Studies on Task-Parameterized Applications](https://arxiv.org/abs/2609.05040)

**Authors**: Yanchen Li, Xiaoming Xue, Kay Chen Tan  
**Category**: cs.AI  
**Published**: 2026-09-07  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.05040v1  

#### Abstract
As evolutionary transfer optimization (ETO) scales to larger collections of related tasks, problem evaluation can become a major source of runtime growth. This work studies problem-side evaluation scaling in task-parameterized applications and reformulates application-specific serial computations in...

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

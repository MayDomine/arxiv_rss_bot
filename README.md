# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-09 09:03:10 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Latency-Constrained DNN Architecture Learning for Edge Systems using Zerorized Batch Normalization](https://arxiv.org/abs/2607.06922)

**Authors**: Shuo Huai, Di Liu, Hao Kong, Weichen Liu, Ravi Subramaniam, Christian Makaya, Qian Lin  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2607.06922v1  

#### Abstract
Deep learning applications have been widely adopted on edge devices, to mitigate the privacy and latency issues of accessing cloud servers. Deciding the number of neurons during the design of a deep neural network to maximize performance is not intuitive. Particularly, many application scenarios are...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Latency-Constrained DNN Architecture Learning for Edge Systems using Zerorized Batch Normalization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**边缘设备上的深度神经网络（DNN）部署**中存在的以下挑战：
- **严格的延迟约束**：许多实时应用（如自动驾驶、无人机追踪）要求模型推理延迟必须满足硬性限制。
- **硬件不匹配**：现有的DNN模型通常为高性能服务器设计，难以直接适配资源受限的边缘设备（如Jetson Nano/TX2）。
- **优化效率低**：传统方法依赖多阶段流程（预训练 → 剪枝/缩放 → 微调），耗时且需要反复在目标设备上测量延迟。

### 提出的新方法与创新思路
作者提出了一种**端到端的、基于零化Batch Normalization（ZeroBN）的延迟感知DNN架构学习框架**，其核心创新包括：

#### ✅ **Compact Learning（紧凑学习）**
- **无需预训练和重训练**：直接从随机初始化开始训练，在一个训练过程中完成模型压缩或扩展。
- **动态零化与恢复机制（Zero-Recovery Process）**：
  - 利用BN层中的可学习参数 $ \gamma $（scaling factor）作为通道重要性的显式指示器。
  - 在训练中周期性地将不重要的 $ \gamma_i $ 和 $ \beta_i $ 设为0（Zero Phase），随后允许它们通过梯度更新恢复（Recovery Phase）。
  - 这种机制实现了对最优子结构的探索，避免了“一旦剪掉就无法恢复”的问题。

#### ✅ **机器学习驱动的延迟预测器（ML-based Latency Predictor）**
- 构建了一个轻量级的BP神经网络，输入是模型各层配置（kernel size, channels等），输出是预测延迟。
- 支持任意未见过的层结构，解决了Lookup Table（LUT）泛化能力差的问题。
- 预测误差仅约 **6.12%**，远低于已有工作（如[30]达13.52%）。
- 显著减少真实设备上的延迟测量次数，提升搜索效率。

#### ✅ **统一的模型缩放策略（Unified Scaling）**
- 对于简单模型（延迟低于约束），先进行统一扩宽加深（width & depth scaling），再用compact learning压缩回目标延迟内。
- **无需昂贵的scaling factor搜索过程**（如EfficientNet需NAS搜索α, β, γ）。
- 自动平衡宽度与深度的扩展比例，实现更高精度。

#### ✅ 开源代码
项目已开源：https://github.com/ntuliuteam/ZeroBN

---

### 相比现有方法的优势
| 维度 | 本文方法 | 传统方法（如NS, SFP, PGMPF） |
|------|----------|-------------------------------|
| **训练流程** | One-shot training（单次训练） | 三阶段：pre-train → prune → re-train |
| **延迟优化方式** | 直接以latency为优化目标 | 间接以FLOPs/channels为代理指标 |
| **延迟测量开销** | 极少（靠预测器指导） | 每次修改后需实测，耗时数分钟 |
| **架构灵活性** | 支持channel pruning + layer pruning联合优化 | 多数仅支持channel pruning |
| **缩放支持** | 内建scaling，无需额外搜索 | 缩放需独立搜索（如Eff-Compd） |

---

## 2. 核心实验方法和设置

### 数据集
- **ImageNet-100**：从ILSVRC-2012中选取100类构成的子集，用于大规模图像分类任务。
- **CIFAR-10**：小规模图像分类基准，用于验证通用性。

### 硬件平台
- **NVIDIA Jetson TX2**：嵌入式AI设备，Pascal GPU，8GB内存。
- **NVIDIA Jetson Nano**：更低端设备，Maxwell GPU，4GB内存。
- 所有测试均运行在“Max-N”模式下并锁定最高频率以保证一致性。

### 模型架构
涵盖主流CNN结构：
- **VGG-19**（顺序结构）
- **ResNet-50 / ResNet-164**（残差模块）
- **DenseNet-40**（密集连接）
- **GoogLeNet**（Inception模块）

### 基线方法对比
共比较7种SOTA方法：
| 方法 | 类型 |
|------|------|
| **SFP [6]** | 软滤波器剪枝 |
| **FPGM [7]** | 几何中位数剪枝 |
| **NS [1]** | Network Slimming（经典BN剪枝） |
| **PGMPF [8]** | 梯度掩码引导微调 |
| **OTO [9]** | Only Train Once（一次性训练剪枝） |
| **Eff-Compd [10]** | EfficientNet复合缩放 |
| **HACScale [11]** | 硬件感知复合缩放 |

### 评估指标
- **Top-1 / Top-5 Accuracy**
- **Latency (ms)** on target device
- **Compression Ratio (%)**
- **FLOPs Reduction**
- **Training Time / Overhead**

---

## 3. 主要实验结果和性能指标

### 关键性能数据（ImageNet-100, 34ms延迟约束）

| Model | Device | Method | Top-1 Acc (%) | Latency (ms) | ΔAcc vs Original |
|-------|--------|--------|----------------|---------------|--------------------|
| VGG-19 | Jetson TX2 | Ours (OurLP-180) | **87.10** | ~34 | **+1.86↑** |
| ResNet-50 | Jetson TX2 | Ours (OurQ-180) | **87.04** | ~34 | **-2.44↓**（仍优于所有基线） |
| GoogLeNet | Jetson TX2 | Ours (OurSQLP-180) | **88.10** | ~34 | **+2.78↑**（相比原20.27ms模型） |
| GoogLeNet | Jetson Nano | Ours (OurQ-180) | **86.92** | ~34 | **-1.60↓**（仅降1.6%，远优于其他） |

> 注：原始GoogLeNet在Nano上延迟为40.32ms > 34ms，必须压缩；本文方法将其压缩至34ms，精度仅下降0.14%（无量化）或0.04%（带FP16量化）。

### 与基线方法对比结果
- 在相同训练设置下（90 epochs），**Our-90** 在所有模型上均取得**最高准确率**，且显著优于NS、SFP等。
- 当训练轮数与NS一致（180 epochs），**Our-180** 不仅超越所有基线，甚至**超过原始未压缩模型精度**（如VGG-19 +1.86%）。
- 引入**layer pruning（OurLP）** 后进一步提升性能，说明联合优化channel与layer更有效。
- 结合**量化（OurQ）** 可保留更多参数，获得更高精度。

### 消融实验结果
#### （1）训练开销分析（Table 2）
| 方法 | 总训练时间（小时） | 相对传统训练开销 |
|------|------------------|------------------|
| Traditional Training | 20.175 | 0% |
| SFP | 21.700 | +7.56% |
| PGMPF | 48.325 | **+139.53%** |
| **Ours** | **20.400** | **+1.12%** |

👉 表明本方法引入的计算开销极小，几乎可忽略。

#### （2）超参数敏感性分析（Figure 10）
- 推荐设置：`start_epoch = epoch_max // 2`，`zero_interval = 2`
- 实验显示在合理范围内，性能稳定，鲁棒性强。

#### （3）延迟预测器准确性（Table 1）
| 模型 | 真实延迟 (ms) | 预测延迟 (ms) | 误差 (%) |
|------|---------------|---------------|---------|
| ResNet-101 | 82.67 | 83.19 | 0.63% |
| MobileNet-160 | 11.67 | 11.42 | 2.14% |
| Inception-v3 | 70.60 | 64.73 | 8.31% |
| Nasnet-mobile | 96.25 | 111.15 | 13.41% |
| **平均误差** | —— | —— | **6.12%** |

✅ 显著优于Justus et al. [30] 的NN-based预测器（平均13.52%误差）。

---

## 4. 关键结论和发现

### 主要发现
1. **直接以延迟为目标的优化至关重要**：
   - FLOPs与实际延迟相关性弱（见Fig. 2），不能可靠反映硬件表现。
   - 必须结合**硬件定制化的延迟预测器**才能高效逼近真实性能。

2. **One-shot training完全可行且高效**：
   - **预训练不是必需的**，最终模型性能主要取决于架构而非初始权重。
   - 通过Zero-Recovery机制可在单次训练中自动发现最优稀疏结构。

3. **联合channel与layer pruning更优**：
   - 单纯通道剪枝受限于固定拓扑。
   - 支持layer pruning使模型能“变深”或“变浅”，适应不同数据集复杂度（如ImageNet适合更深结构）。

4. **Scaling + Compression优于纯Compression**：
   - 对低延迟模型（如GoogLeNet @ 20.27ms），先放大再压缩可获得更高精度。
   - 无需手动搜索scaling factor，由算法自动决定最佳扩展程度。

5. **量化可进一步释放潜力**：
   - FP16量化降低延迟，使得在相同延迟预算下可保留更多通道，从而提高精度。

---

### 方法的局限性
- **依赖BN层存在性**：若模型不含BN（如某些Vision Transformer），需人工插入mask变量。
- **目前仅支持静态结构**：未考虑动态推理路径（如CondConv）。
- **延迟预测器需重新校准**：更换新硬件需重新采集样本训练预测器。

---

### 未来工作方向
1. 将本框架集成进**Neural Architecture Search (NAS)** 流程，构建硬件感知的DNN设计流水线。
2. 扩展至其他硬件指标优化，如**energy, memory footprint**，只需替换预测器即可。
3. 探索与**Knowledge Distillation**结合，利用大模型指导紧凑模型训练。
4. 支持更多量化格式（INT8, INT4）及稀疏推理加速库。

--- 

> 📌 **一句话总结**：  
> 本文提出一种高效的**one-shot、延迟感知的DNN架构学习方法**，通过**ZeroBN动态剪枝+ML延迟预测器+统一缩放机制**，在边缘设备上实现了高精度、低延迟的模型定制，显著优于传统多阶段剪枝与手工缩放方案。

</details>

---

### 2. [SpaCellAgent: A Self-Evolving LLM-Based Multi-Agent Framework for Trajectory Analysis](https://arxiv.org/abs/2607.07467)

**Authors**: Songhan Wang, Haoang Chi, He Li, Zhiheng Zhang, Jiayan Yuan, Cheems Wang, Hao Peng, Xinwang Liu, Wenjing Yang  
**Category**: cs.AI  
**Published**: 2026-07-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.07467v1  

#### Abstract
Spatial and Single-cell transcriptomics are transformative in deciphering cellular dynamics. As the fundamental paradigm for reconstructing cell developmental paths, trajectory inference (TI) is critical. However, existing methods require extensive manual intervention and proficiency in heterogeneou...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前在单细胞转录组学（scRNA-seq）和空间转录组学（spatial transcriptomics）中的**轨迹推断**（Trajectory Inference, TI）分析面临以下挑战：
- 分析流程复杂，依赖大量手动干预；
- 需要研究人员具备多种异构工具（如 Monocle、PAGA、Slingshot 等）的专业知识；
- 不同数据集间算法选择缺乏标准化，导致可重复性差；
- 缺乏自动化反馈机制来修正错误或优化模型输出。

这些问题显著提高了领域科学家的使用门槛，限制了高通量生物学假说的快速生成。

---

### **提出的新方法与新思路**
作者提出了 **SpaCellAgent** —— 一个基于大语言模型（LLM）的**自进化多智能体框架**（self-evolving LLM-based multi-agent framework），用于实现端到端的轨迹分析与生物叙事生成。

其核心架构包含四大智能体角色：
- **Planner（规划者）**：将自然语言任务分解为可执行步骤。
- **Executor（执行者）**：动态调用合适的计算工具并生成代码。
- **Evaluator（评估者）**：进行双重验证——语法正确性和生物学合理性。
- **Reporter（报告者）**：整合结果生成可解释的生物学报告。

此外，引入两个关键机制：
- **Self-refinement（自我精炼）**：通过迭代反馈自动修复代码错误和逻辑偏差。
- **Self-evolution（自进化）**：跨任务积累成功的工作流模板和错误修复策略，持续提升系统鲁棒性。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | SpaCellAgent |
|------|--------|-------------|
| **自动化程度** | 手动操作为主，需专家介入 | 完全自动，从输入到报告一体化 |
| **工具选择** | 固定配置，依赖经验 | 动态工具编排，数据驱动决策 |
| **容错能力** | 无自动纠错机制 | 双层评估 + 自我精炼闭环 |
| **知识复用** | 无法跨任务学习 | 具备全局记忆库，支持检索增强生成 |
| **可访问性** | 仅限专业人士 | 支持自然语言交互，降低使用门槛 |

> ✅ **核心优势总结**：首次实现了**闭环、自适应、可进化的轨迹分析自动化系统**，将分析效率提升超过40%，同时保持专家级准确性。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共六大数据集，涵盖多种技术平台与生物场景：

| 数据集类型 | 名称 | 特征 |
|----------|------|-------|
| **真实基准数据集** | REAL-GOLD, REAL-SILVER | 来自 Saelens et al. (2019)，含多种拓扑结构（线性、分支等） |
| **合成数据集** | SYNTHETIC | 控制变量下的模拟轨迹，用于量化评估 |
| **空间转录组数据** | Mouse Dorsal Midbrain | 小鼠胚胎背侧中脑，Stereo-seq 技术，含时空信息 |
| | Axolotl Neuron Regeneration | 墨西哥钝口螈神经再生过程 |
| **真实临床相关数据** | Mouse Spinal Cord Injury (SCI) | 未发表数据，研究高盐饮食对脊髓损伤恢复的影响 |

> 📊 总计覆盖 >6万细胞，涉及小鼠、人类、爪蟾等多种物种。

---

### **实验设置与评估指标**

#### **模型配置**
- 主体 LLM：**DeepSeek-V3**
- 温度参数：0.0（推理）、0.4（反思阶段）
- 最大输出 token 数：128,000

#### **评估指标**
采用四类互补指标全面评价性能：

| 指标 | 含义 |
|------|------|
| **Correlation (Corr)** | 推测轨迹与真实轨迹之间的 Spearman 秩相关系数 |
| **F1 Score (F1)** | 分支结构匹配的 Jaccard-based F1，衡量拓扑一致性 |
| **Weighted Correlation (WCor)** | 加权基因重要性相关性，强调关键驱动基因识别能力 |
| **Hamming-Ipsen-Mikhailov (HIM)** | 图结构距离，衡量网络拓扑相似性（值越高越好） |

---

### **基线方法对比**
与五种主流 TI 工具进行比较：
- **DPT**：基于扩散伪时间的方法
- **RaceID/StemID**：聚类+最小生成树
- **Scorpius**：最短路径投影法
- **PAGA**：图抽象保留全局拓扑
- **PAGA Tree**：PAGA 的树形变体
- **Slingshot**：主曲线拟合方法

所有基线均使用默认参数运行。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| 方法 | REAL-GOLD (Corr/F1) | REAL-SILVER (Corr/F1) | SYNTHETIC (Corr/F1) |
|------|---------------------|------------------------|----------------------|
| DPT | 0.213 / 0.262 | 0.259 / 0.279 | 0.308 / 0.345 |
| Slingshot | 0.347 / 0.483 | 0.469 / 0.541 | 0.448 / 0.421 |
| **SpaCellAgent (Ours)** | **0.480 / 0.608** | **0.774 / 0.645** | **0.564 / 0.509** |

✅ 在绝大多数指标上达到 **SOTA（state-of-the-art）水平**，尤其在复杂拓扑（如 REAL-SILVER）中表现突出。

---

### **与基线方法的对比结果**
- 平均 **Correlation 提升 >40%**
- **F1 分数提升最高达 53.6%**（vs. Slingshot on REAL-GOLD）
- 在 WCor 和 HIM 上也显著优于其他方法，说明不仅能重建拓扑，还能准确识别关键驱动基因和整体网络结构。

> 🔍 案例分析显示，SpaCellAgent 成功重建了小鼠中脑中 **Radial Glia-like (RGL) → NeuB/GlioB** 的双分支分化路径，并识别出关键基因 *Sox2*, *Neurog2*, *Nfia*，与已知文献一致。

---

### **消融实验结果（Ablation Studies）**

#### **组件有效性分析（Figure 7）**
移除关键模块后 Task Success Rate 显著下降：
- **w/o Planner**：无法有效拆解任务，成功率骤降
- **w/o Evaluator**：缺乏纠错机制，易陷入死循环
- **w/o Self-evolution**：无法复用历史经验，效率降低
- **GPT-4 (no agent)**：单模型难以管理长程任务，失败率高

> 💡 结论：**多智能体协作 + 自我演化机制是系统鲁棒性的关键**

#### **效率对比（Table 2）**
| 步骤 | 人类专家耗时（分钟） | SpaCellAgent 耗时（分钟） | 提升幅度 |
|------|--------------------|-------------------------|----------|
| Step 1 | 10.0 ± 1.6 | 12.4 ± 0.8 | -24% |
| Step 4 | 8.4 ± 1.2 | 3.9 ± 0.7 | **+53.6%** |
| Step 5 | 26.6 ± 3.8 | 9.6 ± 0.9 | **+63.9%** |
| **总计** | **64.6** | **38.0** | **↓41.2%** |

> ⏱️ 尽管预处理稍慢，但在下游复杂任务（如轨迹推断、可视化）中大幅领先，总体节省 **41.2% 分析时间**。

#### **跨LLM敏感性测试（Table 3）**
在 GPT-5.2 和 Claude-sonnet-4-6 上重复实验，性能稳定，证明框架设计具有**模型无关性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **SpaCellAgent 可以媲美甚至超越人类专家**在轨迹推断任务上的表现，且无需编程背景即可使用。
2. 多智能体协同 + 动态工具编排机制能有效应对不同数据分布和拓扑结构的挑战。
3. 自我精炼与自进化机制显著提升了系统的稳定性与泛化能力，在面对新数据时仍能快速收敛。
4. 实现了从“原始数据 → 生物洞察”的全自动化闭环，极大缩短了 **time-to-insight**。

---

### **方法的局限性**
- 当前依赖外部 LLM API，存在成本与延迟问题；
- 对极端稀疏或低质量数据的鲁棒性有待进一步验证；
- 工具注册表虽支持动态扩展，但仍受限于已有包生态；
- 所有调控假设仍需后续湿实验验证（文中明确指出为“computational hypotheses”）。

---

### **未来工作方向**
- 扩展至 **multi-omics integration**（如 ATAC-seq + scRNA-seq）
- 融合 **regulatory network inference** 模块，预测基因调控关系
- 构建本地化部署版本，减少对外部 LLM 的依赖
- 引入用户反馈机制，实现人机协同进化

---

## ✅ **总结一句话**
> **SpaCellAgent 是首个实现端到端、自进化、多智能体驱动的轨迹分析框架，它将复杂的计算生物学流程转化为自然语言交互，不仅提升了分析效率超 40%，更推动了计算生物学向智能化、民主化迈进的关键一步。**

</details>

---

### 3. [DeLS-Spec: Decoupled Long-Short Contexts for Parallel Speculative Drafting](https://arxiv.org/abs/2607.07409)

**Authors**: Hong-Kai Zheng, Piji Li  
**Category**: cs.CL  
**Published**: 2026-07-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2607.07409v1  

#### Abstract
Speculative decoding accelerates LLM inference by drafting multiple tokens and verifying them in parallel. Block-parallel drafters such as DFlash further improve drafting efficiency by predicting an entire block in one pass, but their position-wise predictions lack explicit intra-block causal condit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DeLS-Spec: Decoupled Long-Short Contexts for Parallel Speculative Drafting

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **block-parallel speculative decoding** 方法（如 DFlash）虽然通过并行预测整个 token block 显著提升了推理速度，但由于其内部缺乏对已生成 draft token 的显式因果依赖建模（intra-block causality），导致后续 token 的接受率较低，限制了平均接受长度（average acceptance length）和整体加速效果。

此外，近期改进方法（如 Domino、DSpark）虽引入了局部因果建模，但通常需要从头训练或联合优化 draft model 和 target model，成本高且灵活性差，难以适配已有训练好的 DFlash 模型。

---

### 🚀 提出的新方法：DeLS-Spec
作者提出 **DeLS-Spec**（Decoupled Long-Short Context Speculative Decoding），一种解耦的长-短上下文建模范式：

- **Long-context Expert**：保留预训练好的 DFlash 模型作为“全局语义专家”，负责捕捉来自完整前缀 $ y $ 的长期上下文信息。
- **Short-context Expert**：引入一个轻量级的 **local head**（如 GRU 或 Markov 结构），专门建模当前 block 内部已生成 token 的局部因果关系 $ z $。
- **Logit Fusion**：在推理时将两者输出的 logits 融合，并减去 **unigram prior** 防止高频词被重复强化：
  $$
  \log p(x_i|y,z) \approx \log p_L(x_i|y) + \alpha \log p_S(x_i|z) - \beta \log p_p(x_i)
  $$

---

### 🔍 相比现有方法的优势

| 特性 | DeLS-Spec | Domino / DSpark |
|------|-----------|------------------|
| 是否需重训 DFlash | ❌ 否（固定 backbone） | ✅ 是（端到端联合训练） |
| 训练独立性 | ✅ 可单独训练 local head | ❌ 必须依赖 target model 输出 |
| 模块化与复用性 | ✅ local head 可跨 checkpoint 复用 | ❌ 与特定模型绑定 |
| 训练成本 | ⬇️ 极低（仅需标准 next-token 预测） | ⬆️ 高（多组件协同训练） |
| 推理灵活性 | ✅ 支持即插即用 | ❌ 固定架构 |

> 💡 核心思想是：**不修改原 drafter，而是“外挂”一个轻量 local head 来补足其缺失的局部因果能力**。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **数学任务**：GSM8K、MATH-500、AIME25
- **代码生成**：HumanEval、MBPP、LiveCodeBench (LCB)
- **对话任务**：MT-Bench、Alpaca
- **训练数据**：
  - Qwen3-4B 使用 `Qwen3-4B-Instruct-100K`
  - Qwen3-8B 使用 `Qwen3-8B-ShareGPT`

---

### ⚙️ 实验设置
- **模型**：
  - 主干模型：Qwen3-4B 和 Qwen3-8B
  - Draft model：官方发布的 DFlash checkpoints（block size=16）
- **评估指标**：
  - **End-to-end decoding speedup**（相对于 autoregressive decoding）
  - **Average acceptance length (T)**：验证阶段平均接受的 draft token 数量
- **实现细节**：
  - Local head 默认为 GRU 结构，参数量小
  - 使用 Hugging Face Transformers + CUDA Graphs 加速
  - 融合系数默认 $\alpha = \beta = 0.3$

---

### 🆚 基线方法对比
| 方法 | 类型 |
|------|------|
| Autoregressive | 原始自回归解码 |
| EAGLE-3 | Tree-based speculative decoding |
| DART | Diffusion-inspired drafting |
| DFlash | Block-parallel drafting（主基准） |
| Domino-FT | Fine-tuned version of Domino（用于消融比较） |

---

## 3. 主要实验结果和性能指标

### 📊 性能提升（Table 1 & Table 2）

#### 在 Qwen3-4B 上（temperature=0）：
| 方法 | 平均 speedup | 平均 T |
|------|-------------|--------|
| DFlash(16) | 4.63× | 6.04 |
| **DeLS-Spec(16)** | **4.82×** (+4.1%) | **6.35** (+5.1%) |

- 在 **MATH-500** 上：
  - Speedup 从 6.09× → **6.35×**
  - T 从 7.81 → **8.21**
- 在 **HumanEval** 上：
  - Speedup 从 4.61× → **4.85×**
  - T 从 5.84 → **6.21**

> ✅ 所有 benchmark 均取得一致提升，尤其在数学与代码等强局部依赖任务上更显著。

#### 在不同 DFlash Checkpoint 上的泛化能力（Table 2）
即使应用于 DSpark 发布的 block-7 DFlash 模型，DeLS-Spec 依然带来稳定增益：
- Qwen3-4B 上平均 speedup 提升约 **0.2×**，T 提升 **0.26 左右**
- 表明 local head 具备良好的 **跨模型迁移能力**

---

### 🔍 消融实验结果

#### （1）融合系数 $\alpha$ 和 $\beta$ 的影响（Figure 2）
- 最佳区域集中在 $\alpha \approx \beta \approx 0.3$
- 若不减去 unigram prior（$\beta=0$），性能明显下降
- 验证了 **unigram prior correction 的必要性**

#### （2）残差项忽略的影响（Table 3）
| 方法 | Avg. T | Gain Ratio (%) |
|------|--------|----------------|
| DFlash | 6.04 | — |
| DeLS-Spec (RNN) | 6.35 | 75.7% |
| Domino-FT | 6.45 | 100% |

> 尽管未建模残差交互项 $R(x;y,z)$，DeLS-Spec 仍恢复了 **75.7% 的性能增益**，说明解耦设计高效实用。

#### （3）可学习 vs 固定融合权重（Table 4）
| 设置 | Avg. T |
|------|--------|
| $\alpha=\beta=0$（DFlash） | 6.33 |
| $\alpha=\beta=0.3$（固定） | **6.65** |
| Learnable $\alpha,\beta$ | 6.55 |

> ❗ 固定值表现优于可学习参数，推测因 teacher-forcing bias 导致训练-测试不一致。

#### （4）训练成本对比（Table 5）
| 方法 | Qwen3-4B 训练时间 | VRAM 占用 |
|------|------------------|----------|
| Domino-FT | 13.4 小时 | 42.6 GB |
| **DeLS-Spec (RNN)** | **1.1 小时** | **9.0 GB** |
| DeLS-Spec (Markov) | **0.4 小时** | **6.5 GB** |

> ⏱️ 训练效率提升超 **12 倍**，内存节省近 **5 倍**；8B 模型下 Domino-FT 直接 OOM。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **局部因果建模可显著提升 block-parallel drafting 性能**，尤其是在数学与代码任务中。
2. **无需重新训练 DFlash**，即可通过外挂轻量 local head 实现性能增强。
3. **解耦训练策略极大降低部署门槛**，支持模块化、即插即用式的性能升级。
4. **unigram prior correction 对防止频率偏差至关重要**。
5. **DeLS-Spec 展现出良好泛化性**，可在不同 DFlash checkpoint 上复用。

---

### ⚠️ 局限性（Limitations）
1. **忽略了 long-short context 之间的残差交互项** $R(x;y,z)$，理论逼近存在上限。
2. 当前验证主要基于 DFlash 架构，是否适用于其他 non-autoregressive drafters 待进一步探索。
3. fusion weights 使用固定值，尚未实现动态调整以适应不同上下文复杂度。
4. local head 存在 exposure mismatch 问题（训练用真值前缀，推理用自产前缀）。

---

### 🔮 未来工作方向
1. **扩展至更多 parallel drafter 架构**（如 diffusion LM、semi-autoregressive models）。
2. **开发 adaptive logit fusion 策略**，例如根据 entropy 或 confidence 动态调节 $\alpha, \beta$。
3. **缓解 exposure mismatch**，尝试 scheduled sampling 或 alignment-based tuning。
4. **探索更高效的 local head 架构**（如稀疏激活、蒸馏版本）以进一步压缩开销。

---

> 📌 **一句话总结**：  
> **DeLS-Spec 提供了一种低成本、高灵活性的方式，在不改动原有 block-parallel draft model 的前提下，通过“外挂”轻量 local head 引入 intra-block causality，显著提升 speculative decoding 的接受率与推理速度。**

</details>

---

### 4. [GIFT: Geometry-Informed Low-precision Gradient Communication for LLM Pretraining](https://arxiv.org/abs/2607.07494)

**Authors**: Jieying Wang, Shuyuan Fan, Mingkai Zheng, Zhao Zhang  
**Category**: cs.DC  
**Published**: 2026-07-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.07494v1  

#### Abstract
Gradient communication is a primary scaling bottleneck in large language model (LLM) pretraining. Communicating gradients in low-precision formats, such as FP8 and NVFP4, can significantly reduce the communication volume. Existing methods quantize gradients via linear or nonlinear mappings in Euclid...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GIFT: Geometry-Informed Low-precision Gradient Communication for LLM Pretraining

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模语言模型（LLM）的分布式预训练中，**梯度通信是主要的扩展瓶颈**。虽然采用低精度格式（如 FP8、NVFP4）可以显著减少通信量，但现有的欧几里得空间（Euclidean space）量化方法由于忽略梯度的各向异性（anisotropic）几何特性，导致某些方向上的量化误差远大于其他方向，从而损害模型性能。

### 提出的新方法：GIFT
作者提出 **GIFT**（Geometry-Informed Gradient Scaling），一种基于局部几何感知的低精度梯度通信方法。其核心思想是：
- 在量化前将梯度从原始的 Euclidean 参数空间变换到一个**近似各向同性**（near-isotropic）的空间；
- 在该几何感知坐标系下进行 FP8 量化和通信；
- 通信后将梯度映射回原始坐标系供优化器使用。

GIFT **不改变优化器、训练流程、通信原语或低精度格式本身**，仅修改用于通信的坐标系统。

### 创新点与优势
1. **首次指出低精度通信误差的本质是“坐标系统错配”**  
   即：高度各向异性的梯度与轴对齐的均匀量化网格之间的不匹配。

2. **引入 K-FAC 风格的局部几何近似作为通信坐标基础**  
   利用 Fisher 信息矩阵的 Kronecker 分解（K-FAC）来构造白化变换，使梯度分布更接近球形，提升低精度表示的保真度。

3. **实用化的三阶段简化设计**，实现高效部署：
   - **仅保留输入侧变换**（input-side only）：舍弃输出侧因子以降低开销；
   - **低秩近似**（rank-32）：压缩协方差矩阵，节省存储与计算；
   - **选择性应用**（selective deployment）：只在数值最敏感的少数层（如 fc2 层）启用 GIFT。

4. **相比现有方法的优势**
   - 不依赖特定优化器（如 AdamW），兼容性强；
   - 优于直接 Euclidean FP8 量化，在保持部分通信加速的同时显著改善下游任务表现；
   - 比全量 K-FAC 更轻量，具备实际部署可行性。

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **OpenWebText** 数据集进行 LLM 预训练。

### 模型配置
- **Llama-300M** 和 **Llama-600M** 两种规模的 LLaMA 架构模型。
- 序列长度分别为 4096（300M）和 2048（600M），以适应 GPU 内存限制。

### 实验设置
- **硬件平台**：TACC 的 Vista 超算，基于 **NVIDIA GH200 Grace Hopper Superchip**，共 600 节点。
- **并行策略**：支持 Data Parallelism（DP）、Tensor Parallelism（TP）、Pipeline Parallelism（PP）的 3D 并行。
- **通信库**：NCCL 2.24（支持原生 FP8）。
- **优化器**：Muon optimizer。
- **训练参数**：
  - 全局 batch size = 512
  - micro batch size = 4
  - 学习率 = 5e-4，余弦衰减至 5e-5

### 评估指标
| 类别 | 指标 |
|------|------|
| **系统效率** | step time、端到端预训练时间、通信体积 |
| **优化行为** | validation loss 曲线 |
| **模型质量** | 多项下游任务性能（如 BOOLQ、RTE、PIQA 等） |
| **保真度分析** | FP8 round-trip 重建误差（RelL2、Cos、MaxErr、MSE） |
| **资源消耗** | 显存占用增加比例 |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **FP32** | 全精度梯度通信（高精度基准） |
| **BF16** | 半精度通信对照组 |
| **Euclidean baseline** | 层级 FP8 缩放 + 轴对齐量化（标准低精度方案） |
| **Per-block Euclidean FP8** | 块级缩放的 FP8 方案（较弱基线） |
| **Full K-FAC** | 完整两面 K-FAC 变换（重代价版本） |
| **GIFT** | 本文提出的输入侧低秩 + 选择性部署方案 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 结果 |
|------|------|
| **通信体积减少** | FP8 相比 FP32 减少 **75.0%** 梯度 payload |
| **端到端加速（64芯片）** | GIFT 在 Llama-600M 上实现 **7.6% 总体时间下降** |
| **最大重建误差降低** | GIFT 相比 Euclidean 基线减少高达 **67.4%** 的 FP8 重建误差 |
| **下游任务胜率** | GIFT 在两个模型上均达到 **7/14 任务优于 FP32**，显著高于 Euclidean baseline（4–5/14） |

### 与基线方法对比

#### ✅ 系统层面
- **GIFT vs. Euclidean FP8**：
  - GIFT 步骤时间略慢于纯 Euclidean FP8（因额外计算），但仍比 FP32 快；
  - 随着 GPU 数量增加，通信占比上升，GIFT 的相对收益扩大（见 Fig. 5）；
  - 在 64 GPU 下，Euclidean FP8 加速 10.79%，GIFT 加速 7.6% —— **牺牲少量速度换取更强模型保真度**。

#### ✅ 模型层面
- **validation loss**：
  - GIFT 与 Euclidean baseline 的 loss 曲线几乎重合，且均接近 FP32；
  - 表明二者在训练动态上无明显差异。

- **下游任务表现（Table III）**：
  - **Llama-300M**：
    - Euclidean baseline：5/14 优于 FP32
    - Full K-FAC：6/14 优于 FP32
    - **GIFT：7/14 优于 FP32**
  - **Llama-600M**：
    - Euclidean baseline：5/14 优于 FP32
    - **GIFT：7/14 优于 FP32**

> ⚠️ 尽管 GIFT 没有在每个任务上都超越 FP32，但它在跨任务的整体保留能力上明显优于 Euclidean 方法。

### 消融实验结果

#### A. 输入侧 vs 输出侧变换（Table I）
| 方法 | RelL2 ↓ | Cos ↑ | MaxErr ↓ |
|------|--------|-------|---------|
| Euclidean baseline | 5.28e-2 | 0.9986 | 1.63e-4 |
| Output-side only | 5.28e-2 | 0.9986 | 1.63e-4 |
| Input-side only | **1.77e-2** | **0.99985** | **5.33e-5** |
| Full K-FAC | ~same as input-side | ~same | ~same |

✅ **结论**：输入侧变换主导性能增益，输出侧贡献可忽略。

#### B. 低秩近似效果（Table II）
| 近似方式 | RelL2 ↓ | MSE ↓ |
|--------|--------|------|
| Diag-A（对角） | 5.26e-2 | 1.28e-10 |
| Block-A-128 | 2.43e-2 | 2.73e-11 |
| **Low-rank-A-32** | **1.72e-2** | **1.37e-11** |
| Full-A | 1.77e-2 | 1.45e-11 |

✅ **结论**：rank-32 已能恢复绝大多数增益，性价比最高。

#### C. 选择性部署依据（Fig. 3）
- 对各 MLP 层统计 FP8 边界命中率（boundary hit ratio）；
- 发现仅有前 **13 个最脆弱层**（全部为 `fc2`）存在严重量化失真；
- 支持“仅在这些层启用 GIFT”的策略。

---

## 4. 关键结论和发现

### 主要发现
1. **低精度通信误差不仅是数值问题，更是几何问题**  
   各向异性的梯度在 Euclidean 坐标下被均匀量化时会产生方向依赖的失真。

2. **通过坐标变换可大幅提升低精度通信保真度**  
   将梯度投影到由 K-FAC 构造的近似各向同性空间后，FP8 量化的误差分布更加均衡。

3. **GIFT 实现了更优的 fidelity-efficiency 权衡**
   - 虽然比简单 Euclidean FP8 略慢（7.6% vs 10.79% 加速），
   - 但在下游任务保留方面显著更好（7/14 赢 vs 5/14），
   - **证明“保真度优先”策略的价值**。

4. **选择性 + 低秩 + 输入侧的设计极具实用性**
   - 显存开销仅增加：
     - 300M 模型：+3.33%
     - 600M 模型：+8.98%
   - 计算集中在最关键的层，不影响整体吞吐。

### 方法的局限性
1. **当前评估局限于中等规模模型（300M–600M）**，尚未验证在十亿级以上模型中的有效性。
2. **固定选择脆弱层集合**：目前依赖离线 profiling，未实现动态调整。
3. **未集成到完整量化训练栈**：本工作聚焦于梯度通信，未涉及权重、激活、优化器状态的联合量化。
4. **依赖 K-FAC 统计量更新频率**（每 50 步一次），可能影响长期稳定性。

### 未来工作方向
1. 扩展至更大模型（如 Llama-7B+）和更长训练周期；
2. 探索自动化的在线层选择机制；
3. 将 GIFT 原理推广至 **FP4 通信**（待软硬件成熟）；
4. 结合其他压缩技术（如 sparsification、error feedback）构建统一低比特通信框架；
5. 研究不同架构（Transformer-XL、Mamba）下的几何敏感性迁移能力。

---

> 💡 **总结一句话**：  
> **GIFT 通过“换坐标系”而非“改算法”，巧妙提升了 FP8 梯度通信的质量，在几乎不破坏现有训练流程的前提下，实现了通信效率与模型保真度的更好平衡，为未来极低比特 LLM 预训练提供了新范式。**

</details>

---

### 5. [UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma](https://arxiv.org/abs/2607.06987)

**Authors**: Chongyu Fan, Pengfei Liu, Jingjia Huang, Sijia Liu, Yi Lin  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.06987v1  

#### Abstract
Reinforcement learning (RL) has become the standard paradigm for enhancing the complex reasoning capabilities of large language models (LLMs). To achieve sample efficiency, modern RL frameworks rely on importance sampling (IS). However, these algorithms suffer from an exploration-stability dilemma. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题  
现代基于 **Importance Sampling (IS)** 的强化学习（RL）算法在训练大语言模型（LLM）进行复杂推理时面临一个根本性的 **exploration-stability dilemma**（探索-稳定性困境）：

- **Pure IS** 导致训练不稳定：当对低概率但正确的推理路径进行优化时，IS ratio 可能爆炸，引发梯度爆炸。
- **Clipping 机制限制探索**：标准的 clipping（如 PPO、DAPO）虽然稳定，但会过早截断对“正确但低置信”路径的更新预算，严重抑制了模型的探索能力。

该问题源于传统方法中对历史策略 $ \pi_{\text{old}} $ 的依赖，导致 **Probability Capacity (Cap)** ——即策略可更新的最大概率空间——被严格限制。

---

### 🔧 提出的新方法：Unbounded Positive Asymmetric Optimization (UP)

作者提出 **UP**，一种通用、即插即用的优化目标，旨在打破上述困境。其核心思想是：

#### （1）引入 **Probability Capacity (Cap)** 概念  
形式化定义 Cap 为策略在正优势（$ A > 0 $）下可增加、或负优势（$ A \leq 0 $）下可减少的最大概率值。分析表明，现有方法（如 DAPO）的 Cap 与 $ \pi_{\text{old}} $ 成线性关系，导致稀有路径无法充分探索。

#### （2）设计 **Unbounded Positive Asymmetric Optimization (UP)**  
- **Positive Advantages ($ A > 0 $)**：采用 **unbounded（无剪裁）更新**，通过 **stop-gradient 自锚定比率** 替代传统 IS ratio：
  $$
  r^{\text{up}}_t(\theta) = \frac{\pi_\theta(o_t)}{\text{sg}(\pi_\theta(o_t))}
  $$
  这使得梯度等价于 **REINFORCE**，数学上稳定且允许无限探索。
- **Negative Advantages ($ A \leq 0 $)**：保留标准 clipping（如 DAPO 的 lower clip），防止错误动作过度惩罚导致崩溃。

这种 **asymmetric design** 在理论上实现了“探索最大化 + 稳定性保障”的双重目标。

#### （3）通用性和可扩展性  
UP 可无缝集成到多种 GxPO 框架中：
- **Token-level**: UP-DAPO, UP-GRPO
- **Sequence-level**: UP-GSPO  
适用于不同优化粒度和模型架构。

---

### 🆚 相比现有方法的优势
| 特性 | 传统方法（如 DAPO, GRPO） | UP 方法 |
|------|--------------------------|--------|
| 探索能力 | 被 clipping 限制，Cap 线性依赖 $ \pi_{\text{old}} $ | 正向更新无上限，Cap = $1 - \pi_\theta$，完全释放探索 |
| 稳定性 | 依赖 clipping 维持稳定 | 利用 stop-gradient 锚定当前策略，理论等价 REINFORCE，天然稳定 |
| 通用性 | 需要调参（如 $ \epsilon_{\text{high}} $） | 即插即用，无需额外超参数（自动移除 $ \epsilon_{\text{high}} $） |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **DAPO-17K-MATH**：用于训练 UP-DAPO 和 UP-GSPO
- **MATH (Levels 3–5)**：用于训练 UP-GRPO 并在多个基准测试上评估
- **Geometry3K**：视觉-语言多模态推理任务，含几何图示理解

### 🧪 实验设置
- **模型架构**：
  - Dense LLM: Qwen3-14B-Base, Qwen3-8B
  - MoE: Qwen3-30B-A3B-Base
  - Vision-Language: Qwen3-VL-8B-Instruct
- **训练协议**：
  - 多种采样轨迹（rollout）数量（如 8–32）
  - 使用 vLLM 加速 rollout 生成
  - 所有 baseline 与对应 UP variant 共享相同超参数（仅优化目标不同）

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **Pass@1** | 单次生成即正确的比例（主流推理评测） |
| **Avg@32 / Maj@32 / Best@32** | 基于 32 条轨迹的平均准确率、多数投票准确率、至少一条正确概率 |
| **Entropy** | 输出分布熵，衡量探索多样性 |
| **Gradient Norm & KL Divergence** | 衡量训练稳定性（是否偏离参考模型） |

### 🆚 基线方法对比
共比较 **12 种 RL 基线**，涵盖主流与前沿方法：
- **Token-level**: GRPO, DAPO, CISPO, Dr. GRPO, DPPO, SAPO, GMPO
- **Sequence-level**: GSPO
- **Critic-free / Asymmetric**: REINFORCE++, RLOO, W-REINFORCE, ASPO

---

## 3. 主要实验结果和性能指标

### 📈 性能提升显著（见 Table 1）

| Method | AIME24 | AMC23 | MATH500 | Minerva | OlympiadBench | **Average** |
|-------|--------|--------|---------|---------|---------------|------------|
| GSPO | 40.52 | 85.00 | 88.20 | 31.25 | 55.80 | **60.15** |
| ASPO | 37.50 | 85.00 | 87.60 | 29.78 | **58.48** | 59.67 |
| **UP-GRPO (Ours)** | **41.04** | **87.50** | **88.40** | **31.25** | 58.33 | **61.31** |

✅ **UP-GRPO 在 5 个推理基准上的平均 Pass@1 达到 61.31%**，超越最强基线 GSPO（60.15%）**+1.16% 绝对增益**，并在 4/5 项排名第一。

---

### 🔍 探索能力增强（Fig. 3, Fig. 6）
- **Entropy 更高**：UP-DAPO 和 UP-GRPO 在训练过程中保持更高的生成熵，说明持续探索更多样化的推理路径。
- **Best@32 提升**：UP-DAPO 达到 **81.79%** vs DAPO 的 80.49%，表明更可能找到最优解。

---

### 🛡️ 训练稳定性良好（Fig. 4, Fig. 5）
- **Gradient Norm** 和 **KL Divergence** 与 baseline 相当甚至更低。
- 对比实验显示：
  - 若仅移除 clipping（$ \epsilon_{\text{high}} = \infty $），会导致梯度爆炸（norm 达 $10^{13}$）。
  - 若对负优势也 unbounded 更新，则立即崩溃。
➡️ 证明 **UP 的 asymmetric design 是安全有效的关键**。

---

### 🔬 消融实验（Ablation Studies）
#### （1）Self-Anchored Ratio 必不可少
- 将 DAPO 的 $ \epsilon_{\text{high}} $ 设为无穷大 → 初始稳定，但 **>80 步后梯度爆炸**
- 说明：仅去剪裁不改 anchor 仍危险；必须用 `sg(πθ)` 替代 `π_old` 才能根除 instability

#### （2）Asymmetric Design 至关重要
- 对正负优势都 unbounded 更新 → **前 25 步即崩溃**
- 证实：负优势必须保留 clipping 作为 safeguard

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Exploration-Stability Dilemma 存在且结构性存在**：传统 IS + clipping 范式本质上限制了对稀有正确路径的探索。
2. **Stop-Gradient Self-Anchoring 是突破口**：将 anchor 从 $ \pi_{\text{old}} $ 改为 $ \text{sg}(\pi_\theta) $，可在数学上实现 REINFORCE 级别的稳定性，同时支持 unbounded 更新。
3. **Asymmetric Optimization 是必要结构**：只对正优势 unbounded，负优势保留 clipping，才能兼顾探索与稳定。
4. **UP 是真正通用的 plug-and-play 模块**：在 token-level（DAPO/GRPO）、sequence-level（GSPO）、dense/MoE/vision-language 架构、语言/多模态任务中均有效。

---

### ⚠️ 局限性
- 当前方法仍依赖 on-policy 或近似 off-policy 设置，未解决极端离线 RL 场景下的分布偏移问题。
- 对 extremely long-horizon reasoning（>20k tokens）的 scaling behavior 需进一步验证。
- 未探讨 UP 与其他 advanced exploration mechanisms（如 intrinsic reward）的结合潜力。

---

### 🔮 未来工作方向
- 将 UP 扩展至 **offline RL for LLMs**，研究如何在 stale data 上安全利用 unbounded 更新。
- 结合 **curriculum learning** 动态控制 Cap，避免早期过度探索噪声路径。
- 探索 UP 在 **agent planning、tool calling、multi-turn dialogue** 中的应用。
- 开发更高效的 sequence-level UP 实现，降低长序列的 memory overhead。

---

> **一句话总结**：  
> UP 通过 **asymmetric + self-anchored optimization**，打破了 RL 中“越探索越不稳定，越稳定越不敢探索”的死循环，为 LLM 推理能力的持续进化提供了新的数学基础和工程范式。

</details>

---

### 6. [From Atomic Actions to Standard Operating Procedures: Iterative Tool Optimization for Self-Evolving LLM Agents](https://arxiv.org/abs/2607.07321)

**Authors**: Haipeng Ding, Yuexiang Xie, Zhewei Wei, Yaliang Li, Bolin Ding  
**Category**: cs.AI  
**Published**: 2026-07-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.07321v1  

#### Abstract
Tool utilization enables Large Language Model (LLM) agents to interact with the real world and resolve complex tasks. However, existing agent frameworks predominantly rely on static toolsets composed of granular atomic actions (e.g., basic file I/O or single-turn search), which forces agents to rein...

---

### 7. [TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models](https://arxiv.org/abs/2607.07388)

**Authors**: Yutang Ma, Kecheng Huang, Xikun Jiang, Zili Shao  
**Category**: cs.CL  
**Published**: 2026-07-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.07388v1  

#### Abstract
Large Language Models (LLMs) store factual knowledge and domain-specific patterns implicitly in dense Transformer parameters, making knowledge expansion costly through pretraining, fine-tuning, retrieval augmentation, or longer contexts. Engram-style memory offers a compact hidden-state injection pa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现有的 **Engram-style memory** 虽然为 LLM 提供了一种紧凑的外部记忆路径，但在实际应用中面临三大挑战：

1. **训练依赖性强**：传统 Engram 需要通过模型训练学习 memory entries，难以跨模型复用，且更新知识需重新训练。
2. **语义保真度低**：为了适应有限的 GPU 内存，普遍采用 **hash-based compression**，导致多个不相关的短语被压缩到同一个 memory slot 中，造成语义混淆（semantic collision）。
3. **访问延迟高**：将 memory 外部化至 DRAM 或 SSD 可以扩大容量，但直接访问会阻塞 autoregressive decoding，影响推理吞吐。

这些问题限制了 Engram 成为一个可扩展、免训练、低开销的静态短语记忆系统。

---

### 🚀 提出的新方法与创新思路

作者提出 **TF-Engram** —— 一种 **train-free**、**SSD-backed**、**early-exit guided** 的 Engram 扩展系统，其核心创新包括：

#### （1）**Train-Free Semantic Memory Construction（免训练语义记忆构建）**
- 从大规模语料（如 FineWeb-Edu、Wikipedia、arXiv 等）中离线挖掘候选短语（phrases），如命名实体、技术术语等。
- 使用冻结的 **semantic encoder**（如 Qwen3-Embedding）对每个短语生成独立的语义向量，避免共享 slot。
- 构建两级索引：`token prefix → phrase id → memory address`，支持高效匹配。
- 整个过程无需参与 LLM 训练，实现真正的“即插即用”。

> ✅ 优势：支持跨模型部署、可动态扩展新领域语料、无需 retraining。

#### （2）**SSD-Backed Memory Hierarchy（SSD 支持的多级存储架构）**
- 将 memory 表组织为三级层次结构：
  - **GPU Cache**（热数据）
  - **Host DRAM**（温数据）
  - **NVMe SSD**（冷数据）
- 不再依赖 hash 压缩来节省 GPU 内存，而是通过分层存储保留精确的 phrase-level 条目。
- 支持 page/block 存储优化随机 I/O，提升批量读取效率。

> ✅ 优势：支持十亿级 phrase 表，显著降低 GPU 显存占用（从 >24GB 降至 ~3GB）。

#### （3）**Early-Exit Guided Predictive Prefetching（早期退出引导的预测预取）**
- 在 LLM 的倒数第 $ L-r $ 层附加一个轻量级 **prefetch head**，提前预测下一个 top-K 可能出现的 token。
- 利用这些 token 推断未来可能访问的 phrases，并异步预取其 memory entries 到高速缓存中。
- 利用剩余 Transformer 层的计算时间作为 **latency-hiding window**，隐藏 SSD 访问延迟。

> ✅ 优势：相比基于最终输出的预取机制，更早触发 prefetch，有效重叠 I/O 与计算。

---

### 🔍 相比现有方法的优势

| 方法 | 是否需训练 | 是否支持 phrase-level | 是否依赖 hash 压缩 | 是否引入额外上下文长度 | 是否支持大容量 |
|------|------------|------------------------|--------------------|----------------------------|----------------|
| **RAG** | 否 | ❌（chunk-level） | ❌ | ✅（增加 prompt） | ✅ |
| **LoRA** | ✅ | ❌ | ❌ | ❌ | ❌（参数绑定） |
| **KV-Cache 扩展** | ❌ | ❌ | ❌ | ✅ | ⚠️（仍受限于 context） |
| **传统 Engram** | ✅ | ✅ | ✅（严重冲突） | ❌ | ❌（受限于 GPU） |
| **TF-Engram** | ❌ | ✅ | ❌（保留逻辑条目） | ❌ | ✅✅✅ |

> ✅ TF-Engram 是首个实现 **train-free + phrase-specific + scalable + low-latency** 的统一框架。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **memory construction corpus**：
  - FineWeb-Edu
  - Wikipedia
  - C4 English
  - arXiv
  - PubMed  
  > 覆盖通用知识、科学术语、生物医学等领域，用于构建 phrase memory table。

- **下游评估 benchmark**（共 10 个）：
  - **知识类**：MMLU, ARC-Challenge, OpenBookQA, SciQ, TruthfulQA-MC2
  - **常识推理类**：HellaSwag, PIQA, WinoGrande
  - **语言理解类**：BoolQ, LAMBADA

---

### ⚙️ 实验设置与评估指标

| 项目 | 设置 |
|------|------|
| **Backbone Model** | Qwen3-0.6B |
| **Hardware** | Intel i9-14900K, RTX 5090 (32GB), 64GB RAM, 516GB NVMe SSD |
| **Memory Table Size** | 最大构建 100M entries |
| **Vector Dimension** | 1024-dim FP16 |
| **Prefetch Budget** | Top-16 / Top-64 / Top-128 tokens |
| **Early-Exit Layer** | Layer L−r（默认 r=8） |

#### 评估指标：
- **End-to-end Accuracy**：各 benchmark 上的平均得分
- **Offline Cost**：construction 时间与存储开销
- **Runtime Overhead**：
  - GPU Memory Usage
  - Throughput (tokens/sec)
  - Decode Latency (ms/token)

---

### 🆚 基线方法对比

| 方法 | 描述 |
|------|------|
| **Qwen3-0.6B (Baseline)** | 原始模型，无任何增强 |
| **Parameter-Matched LoRA** | 使用与 TF-Engram 相近的可训练参数量进行微调，代表轻量级训练适配方法 |
| **TF-Engram (Full)** | 完整版本：离线构建 + 分层存储 + 早期预取 |
| **Ablation Variants**：
  - w/o Prefetching
  - Final-Head Guided Prefetching
  - GPU-Resident Only（仅 GPU 缓存） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）端到端准确率对比（Table 3）

| Benchmark | Qwen3-0.6B | LoRA (param-matched) | **TF-Engram** |
|----------|------------|-----------------------|---------------|
| MMLU | 53.7 | 55.3 | **56.2** |
| ARC-Challenge | 42.0 | 44.1 | **45.3** |
| PIQA | 70.0 | 69.5 | **72.1** |
| BoolQ | 76.0 | 75.6 | **78.4** |
| TruthfulQA-MC2 | 45.0 | 46.5 | **48.2** |
| LAMBADA | 48.0 | 49.6 | **50.3** |
| **Average** | **57.6** | **58.7** | **59.4** |

> ✅ TF-Engram 在 **8/10 任务上优于 LoRA**，平均得分提升 **+1.8 pts vs baseline**, **+0.7 pts vs LoRA**。

---

#### （2）运行时开销分析（Figure 4 & 5）

| 配置 | GPU Memory | Throughput (tok/s) | Latency (ms/tok) |
|------|------------|---------------------|------------------|
| Baseline | 22.08 GB | 481.2 | 2.08 |
| TF-Engram (GPU 10M) | 24.03 GB | 470.1 | 2.16 |
| TF-Engram (SSD 100M) | **3.06 GB** | 439.2 | 2.28 |
| + Top-64 Prefetch | 3.06 GB | **460.3** | **2.17** |
| + Top-128 Prefetch | 3.06 GB | 462.4 | 2.16 |

> ✅ **SSD-backed 设计使 GPU 显存减少 87%**（24GB → 3GB）  
> ✅ **Predictive prefetching 恢复约 90% 的吞吐损失**

---

#### （3）离线构建成本（Figure 3）

- 构建 100M entry 的 TF-Engram 表耗时约 **2.0 小时**
  - n-gram extraction 占 1.612h（主导）
  - embedding generation: 0.231h
  - index building: 0.157h
- 存储开销：**215.86 GB**（FP16, 1024-dim）

> ✅ 构建成本可控，适合离线批处理。

---

#### （4）消融实验（Ablation Study）

| 变体 | Avg Score | Throughput |
|------|---------|-----------|
| Full TF-Engram | **59.4** | 462.4 tok/s |
| w/o Prefetching | 58.8 | 439.2 tok/s |
| Final-Head Prefetch | 59.0 | 450.1 tok/s |
| GPU-Only (10M) | 58.5 | 470.1 tok/s |

> ✅ **Early-exit prefetching 贡献最大增益**，比 final-head 提前约 2–3 层，显著提升预取命中率。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **静态短语记忆可以显著提升 LLM 性能而无需训练**  
   → TF-Engram 在保持 backbone frozen 的前提下，平均得分超过 LoRA，验证了 **external memory as a viable alternative to fine-tuning**。

2. **hash compression 损害语义保真度，应被分层存储取代**  
   → 传统 Engram 的性能瓶颈不在容量，而在 **collision-induced ambiguity**；TF-Engram 通过 SSD 分层存储解决了这一根本矛盾。

3. **early-exit prediction 是隐藏 I/O 延迟的关键**  
   → 利用中间层 hidden state 提前预取，可在不影响生成质量的前提下，有效重叠 I/O 与计算。

4. **TF-Engram 具备良好的可扩展性**  
   → 随 memory 规模增大（1M → 100M），性能持续提升（58.2 → 59.4），且运行时开销可控。

---

### ⚠️ 方法的局限性

1. **依赖高质量 phrase mining 与 tokenization 对齐**  
   → 若 phrase 边界与 tokenizer 不一致，可能导致匹配失败。

2. **prefetch accuracy 依赖 early-exit head 的预测能力**  
   → 过早退出（如 L−12）会导致预测不准，过晚则无法充分隐藏延迟。

3. **当前实现基于单机环境**  
   → 多节点分布式部署下的 cache coherence 和 prefetch coordination 尚未探索。

4. **memory injection 可能干扰原始推理路径**  
   → 特别是在 commonsense 推理任务（如 HellaSwag）上略有下降。

---

### 🔮 未来工作方向

1. **动态更新 memory table**  
   → 支持增量添加新知识，实现实时知识注入。

2. **multi-modal extension**  
   → 将 TF-Engram 扩展至图文、音视频等多模态短语记忆。

3. **hardware-aware prefetch scheduler**  
   → 结合 I/O 带宽、cache miss rate 动态调整 prefetch budget。

4. **integration with speculative decoding**  
   → 将 early-exit 信号同时用于 speculation 和 prefetching，进一步加速推理。

---

## ✅ 总结

**TF-Engram** 提出了一种全新的视角：将 Engram 从“神经网络模块”重构为“存储系统组件”。它实现了：

> 🔹 **Train-Free**：无需训练，离线构建即可部署  
> 🔹 **Scalable**：支持百亿级 phrase 表，突破 GPU 显存限制  
> 🔹 **Efficient**：通过 early-exit prefetching 隐藏 I/O 延迟  
> 🔹 **Effective**：在多项 benchmark 上超越 LoRA，证明其作为 **low-overhead memory augmentation** 的潜力

该工作为构建 **可插拔、可扩展、免训练的知识增强 LLM 推理系统** 提供了重要范式。

</details>

---

### 8. [Efficient Bayesian Deep Ensembles via Analytic Predictive Inference](https://arxiv.org/abs/2607.06776)

**Authors**: Sina Aghaee Dabaghan Fard, Marie Maros, Jaesung Lee  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.06776v1  

#### Abstract
We introduce an efficient Bayesian deep ensemble method for predictive regression designed to enhance interpretability while maintaining competitive predictive performance and computational efficiency. Our method combines the statistical rigor of Bayesian inference with the scalability of deep ensem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Bayesian Deep Ensembles via Analytic Predictive Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文旨在解决**预测回归任务中对可靠不确定性估计的需求**，同时兼顾以下三个相互冲突的目标：
- **计算效率**（Computational efficiency）
- **强预测性能**（Strong predictive performance）
- **可解释性**（Interpretability）

现有方法在这些目标之间存在明显权衡：
- **Gaussian Processes (GPs)** 虽然提供精确的贝叶斯推断，但具有 $O(N^3)$ 的计算复杂度，难以扩展到大规模数据。
- **Fully Bayesian Neural Networks (BNNs)** 推理困难，依赖近似推断（如变分推断），可能导致校准不良的不确定性。
- **Bayesian Last-Layer (BLL)** 方法仅对最后一层进行贝叶斯建模，忽略了表示多样性，容易过自信。
- **Deep Ensembles (DE)** 性能良好且可扩展，但采用**均匀平均**（uniform averaging），缺乏概率解释，无法诊断各成员贡献。

### **提出的新方法：Bayesian Deep Kernel Networks (BDKN)**
作者提出了 **BDKN** ——一种结合了深度集成（deep ensembles）的可扩展性和贝叶斯线性回归的统计严谨性的新框架。其核心思想是：
> 将独立训练的神经网络视为**特征映射器**（feature maps），将它们的输出作为高维特征输入到一个顶层的**贝叶斯线性回归模型**中，并通过解析方式积分权重和噪声方差，得到闭式后验预测分布。

#### **三大设计组件**：
1. **低维集成表示**（Low-dimensional ensemble representation）  
   预测由少量（如 H=5）神经网络的输出组合而成，推理成本取决于集成大小而非数据量。

2. **闭式贝叶斯聚合**（Closed-form Bayesian aggregation）  
   使用 **Bayesian Linear Regression** 对集成成员的输出进行加权融合，无需近似推断，直接获得**可解释的后验权重**和**校准良好的不确定性**。

3. **独立集成训练**（Independent ensemble training）  
   多个神经网络分别独立训练，鼓励多样化表示，提升鲁棒性和不确定性校准。

### **相比现有方法的优势**
| 特性 | BDKN | DE | GP | BNN | BLL |
|------|------|----|-----|-----|-----|
| 可扩展性 | ✅ 强（并行训练 + 小规模BLR） | ✅ 强 | ❌ 差（$O(N^3)$） | ⚠️ 中等（采样开销大） | ✅ 强 |
| 不确定性校准 | ✅ 优秀（解析Student-t） | ⚠️ 一般（隐式多样性） | ✅ 优秀 | ⚠️ 可能偏差（近似推断） | ⚠️ 易过自信 |
| 可解释性 | ✅ 后验权重可分析成员贡献 | ❌ 黑箱平均 | ✅ 协方差函数 | ⚠️ 权重难解释 | ⚠️ 固定表示 |
| 计算效率 | ✅ 高（额外开销为 $O(H^3)$） | ✅ 高 | ❌ 低 | ❌ 低 | ✅ 高 |

此外，BDKN 具有：
- **自动降权弱成员**：贝叶斯聚合会自动降低数据支持不足的集成成员的权重，提高鲁棒性。
- **有限秩GP解释**：可被解释为一个以学习到的神经特征为基函数的 **finite-rank Gaussian Process**，避免核设计难题。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在标准 **UCI regression benchmarks** 上进行评估，涵盖不同维度和样本量：
- Boston, Concrete, Energy, Kin8nm, Naval, Power, Protein, Sarcos, Song, Wine, Yacht
- 数据集规模从几百到超过50万样本不等（如 Song: ~515k）

详细统计见附录 Table 3。

### **实验设置**
进行了两种优化配置实验：

#### **Setting 1: Aggressive Optimization (High LR)**
- 学习率：0.1
- Epochs：40
- 架构：两层全连接（50, 50），ReLU激活
- 集成大小 H=5
- 用于模拟轻量级部署场景

#### **Setting 2: Conservative Optimization (Low LR)**
- 学习率：0.001（部分方法更低）
- Epochs：最多500（部分通过验证选择）
- 更长训练周期，更稳定收敛
- 用于测试方法对超参敏感性

所有方法均使用 **20次随机划分**（Protein用5次，Song用1次固定划分），训练/测试比为90%/10%。

### **评估指标**
- **RMSE**（Root Mean Squared Error）：衡量预测准确性  
  $$
  \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}
  $$
- **NLL**（Negative Log-Likelihood）：衡量不确定性校准质量  
  $$
  \text{NLL} = -\frac{1}{N}\sum_{i=1}^N \log p(y_i | x_i)
  $$

### **基线方法对比**
与多种代表性方法比较：
- **GP**: 标准高斯过程（RBF核）
- **DE**: Deep Ensembles（均匀平均）
- **DKL**: Deep Kernel Learning（NN + GP）
- **BLL / LD-BLL**: Bayesian Last Layer 及其导数正则化版本
- **MFVI-BNN**: Mean-Field Variational Inference BNN
- **VBLL**: Variational Bayesian Last Layer

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（摘要）**

#### **Setting 1 (High LR) 结果（Table 1）**
| 方法 | 平均 RMSE 排名 | 平均 NLL 排名 |
|------|----------------|---------------|
| **BDKN** | **1st**（多次最佳） | **1st**（多数领先） |
| DE | 第2–3位 | 第3–4位 |
| GP | 在小数据集上优异 | 在小数据集上优异 |
| BLL/LD-BLL/MFVI/VBLL | 表现不稳定，尤其在Power/Sarcos上退化严重 |

> **亮点**：
> - BDKN 在 **Protein**, **Naval**, **Kin8nm** 等多个数据集上取得最优 RMSE 和 NLL。
> - DE 在 **Power** 和 **Sarcos** 上出现极大方差甚至崩溃（如 Power 上 RMSE 达 3424），而 BDKN 保持稳定。
> - BDKN 的 NLL 显著优于 DE，说明其不确定性更校准。

#### **Setting 2 (Low LR) 结果（Table 2）**
| 方法 | 平均 RMSE 排名 | 平均 NLL 排名 |
|------|----------------|---------------|
| **BDKN** | 前2名 | 前2名 |
| DE | 显著改善，但仍略逊于 BDKN | 改善但仍不如 BDKN |
| GP | 表现强劲但无法运行于大数据集 |
| MFVI-BNN | 表现稳定但计算代价极高 |

> **亮点**：
> - 所有方法稳定性提升，但 BDKN 仍保持竞争力。
> - 在 **Energy**, **Kin8nm**, **Naval** 上 BDKN 继续领先。
> - MFVI-BNN 虽表现好，但训练时间远高于 BDKN（见下表）。

### **与基线方法的对比结果**
- **vs. DE**：BDKN 显著优于 DE，尤其是在高学习率下的不稳定性场景。例如在 **Power** 数据集上，DE 的 RMSE 方差极大，而 BDKN 稳定得多。
- **vs. GP**：在可运行的小数据集上，GP 性能接近或略优，但在 **Protein/Sarcos/Song** 上因内存不足无法运行；BDKN 成功扩展至这些大数据集。
- **vs. BLL 类方法**：BLL/LD-BLL 在某些数据集上表现尚可，但在其他上严重退化（如 Yacht 上 NLL 极差），表明其对单个表示的依赖导致脆弱性。
- **vs. MFVI-BNN**：虽然 MFVI-BNN 在某些指标上接近 BDKN，但其训练时间高出一个数量级以上（见运行时表格）。

### **消融实验与分析（Figure 3）**
研究了**集成大小 H** 对性能的影响（Yacht 数据集）：
- 当 H 从 5 增加到 50 时：
  - RMSE 和 NLL 持续下降
  - 收敛速度加快
  - 最终性能单调提升

> **结论**：更大的集成带来更丰富的特征空间，提升表达能力和不确定性建模能力，且由于并行训练，**增加 H 几乎不增加训练时间**。

### **运行时对比（Table 6 & 7）**
| 方法 | 相对运行时间（Setting 1） | Setting 2 是否可扩展 |
|------|----------------------------|------------------------|
| **BDKN** | ≈ DE（仅多出 $O(H^3)$ BLR） | ✅ 是 |
| DE | ✅ 快速 | ✅ 是 |
| GP | 小数据快，大数据不可行 | ❌ 否 |
| MFVI-BNN / VBLL | ⚠️ 极慢（需多次采样/优化） | ❌ 难以扩展 |

> 示例：在 **Naval** 数据集上，Setting 2 中：
> - BDKN: ~91 分钟
> - MFVI-BNN: **>14600 分钟（约10天）**

---

## **4. 关键结论和发现**

### **主要发现**
1. **BDKN 实现了“三赢”平衡**：在**计算效率、预测性能、不确定性校准**之间取得了优异平衡。
2. **贝叶斯聚合显著优于均匀平均**：不仅能提升性能，还能自动抑制低质量集成成员的影响，增强鲁棒性。
3. **解析推断可行且高效**：通过将深度集成输出作为特征，可在高层实现**闭式贝叶斯推断**，无需近似，保证统计严谨性。
4. **可扩展性强**：得益于并行训练和小规模 BLR，BDKN 可轻松应用于百万级样本的数据集（如 Song）。
5. **对优化策略鲁棒**：无论是在激进还是保守训练设置下，BDKN 均表现出稳定的高性能。

### **方法的局限性**
1. **当前仅适用于回归任务**：分类任务涉及非高斯似然（如 softmax），难以保持解析解。
2. **特征空间受限于集成大小 H**：虽然 H 可增大，但最终仍是有限维表示，可能限制极端复杂函数的拟合能力。
3. **假设同方差先验结构**：目前使用各向同性先验 $A_0 = \sigma_\beta^2 I$，未显式建模成员间相关性。
4. **仍需调参 $\sigma_\beta^2$**：尽管通过 marginal likelihood 自动优化，但初始值和优化过程仍影响结果。

### **未来工作方向**
1. **扩展至分类任务**：探索如何在保持解析性质的同时处理分类中的非共轭似然。
2. **引入结构化先验**：允许协方差矩阵 $A_0$ 学习成员间的依赖关系，进一步提升表达能力。
3. **结合 latent derivative 或 functional regularization**：改进 out-of-distribution 下的不确定性行为。
4. **应用到下游任务**：如 **deep reinforcement learning**, **physics-informed neural networks (PINNs)** 等需要可靠不确定性的系统中。
5. **动态集成裁剪**：利用后验权重识别冗余成员，实现在线压缩与加速。

---

> **总结一句话**：  
> BDKN 提出了一种**高效、可解释、校准良好**的贝叶斯深度集成方法，通过**将独立训练的神经网络作为特征生成器，并在其上构建解析贝叶斯线性模型**，成功弥合了深度集成的经验优势与贝叶斯方法的理论严谨性之间的鸿沟。

</details>

---

### 9. [LEMUR 2: Unlocking Neural Network Diversity for AI](https://arxiv.org/abs/2607.06839)

**Authors**: Tolgay Atinc Uzun, Waleed Khalid, Saif U Din, Sai Revanth Mulukuledu, Akashdeep Singh, Chandini Vysyaraju, Raghuvir Duvvuri, Avi Goyal, Yashkumar Rajeshbhai Lukhi, Muhammad A. Hussain, Krunal Jesani, Usha Shrestha, Yash Mittal, Roman Kochnev, Pritam Kadam, Mohsin Ikram, Harsh R. Moradiya, Alice Arslanian, Dmitry Ignatov, Radu Timofte  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.06839v1  

#### Abstract
Existing NAS benchmarks (e.g., NAS-Bench, NATS-Bench) cover only narrow, task-specific regions of the architectural design space and lack cross-domain or deployment-aware evaluation. LEMUR 2 introduces a large-scale, extensible framework unifying generative, evaluative, and deployment pipelines to u...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LEMUR 2: Unlocking Neural Network Diversity for AI

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有 NAS（Neural Architecture Search）基准（如 NAS-Bench, NATS-Bench）存在以下局限性：
- **设计空间狭窄**：通常基于固定 cell 结构进行穷举搜索，仅覆盖特定任务的小范围架构。
- **缺乏跨域泛化能力**：不支持多模态任务（如图像、文本、语音）之间的架构迁移分析。
- **缺少部署感知评估**：缺乏在真实设备上的延迟、内存等硬件性能元数据，难以指导边缘部署。

### 🚀 提出的新方法与创新思路
LEMUR 2 是一个大规模、可扩展的神经网络多样性框架，其核心创新包括：

#### （1）**统一的生成-评估-部署闭环系统**
- 集成多种自动化架构生成方式，涵盖程序编辑、进化算法、LLM 合成等。
- 构建端到端流程：从模型代码生成 → 训练评估 → 跨平台部署（Android、VR）→ 性能记录。

#### （2）**多样化的架构生成机制**
- **AST-based Mutation**：通过解析 PyTorch 模型源码的 Abstract Syntax Tree 进行通道维度修改，保持结构一致性。
- **Reinforcement Learning (GRPO)**：使用 LLM 作为策略完成“掩码骨架”模型，结合奖励函数优化生成质量。
- **Genetic Algorithm**：对 AlexNet 类型网络进行超参数与结构联合演化。
- **Fractal-Inspired Generation**：构建递归自相似的多列网络（FractalNet 风格），实现深度与宽度平衡增长。
- **Retrieval-Augmented Generation (NN-RAG)**：从开源仓库提取超过 900 个有效 `torch.nn.Module` 模块，形成可复用组件库。
- **LLM-driven Few-Shot Prompting**：利用高分架构示例引导 LLM 生成新模型，提升稳定性和性能。

#### （3）**跨领域任务支持与部署验证**
- 支持多模态任务：
  - 图像分类（CIFAR）
  - 图像描述生成（MS COCO）
  - 文本到图像生成（Diffusion/GAN/CVAE-GAN）
  - 文本到文本生成（WikiText）
  - MoE 架构研究
- 引入两个自动化部署管道：
  - **NN-Lite**：将 PyTorch 模型自动转换为 TFLite 并在 Android 上测试推理延迟。
  - **NN-VR**：集成至 Unity 引擎，评估 VR 场景下的帧时间与显存占用。

### 🔍 相比现有方法的优势
| 维度 | LEMUR 2 | 传统 NAS Benchmarks |
|------|--------|---------------------|
| 架构来源 | 多样化生成（LLM、GA、AST、RAG 等） | 固定搜索空间枚举 |
| 可扩展性 | 开放式代码生成，持续扩展 | 封闭静态集合 |
| 任务覆盖 | 多模态、跨领域 | 单一任务为主 |
| 部署反馈 | 提供真实设备延迟、内存数据 | 无或仅预测值 |
| 数据规模 | >14,000 架构，>750,000 条训练记录 | 数千至数万条 |

> 💡 **优势总结**：LEMUR 2 不只是一个 benchmark，而是一个**数据驱动的 AutoML 生态系统**，推动 LLM-driven AutoML 和跨模态架构泛化。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 任务 | 数据集 |
|------|--------|
| 图像分类 | CIFAR-10, CIFAR-100, ImageNet16-120 |
| 图像描述 | MS COCO |
| 文本生成 | WikiText-2 |
| 文本到图像 | 自定义图文配对数据（基于 CLIP 编码） |
| 语音识别 | TIMIT（间接引用） |
| 图神经网络 | Cora, CiteSeer, ogbn-arXiv 等 |

> 注：部分任务使用标准子集或预处理版本。

### ⚙️ 实验设置
- **训练环境**：NVIDIA RTX 3090/4090 GPU，AI Linux Docker 容器，Kubernetes 集群调度。
- **训练协议**：
  - 使用受限训练周期（bounded training budgets）以控制计算成本。
  - 批量大小、学习率、动量、dropout 等超参数多样化采样。
  - 每次运行记录 accuracy、runtime、参数量、epoch 等元数据。
- **评估方式**：
  - 所有模型均经过编译、前向传播、反向传播验证。
  - 成功模型至少训练一个 mini-epoch 或完整周期（视任务而定）。
  - 报告 **best-per-run accuracy**（各配置下最高准确率），而非最终 epoch 结果。

### 📊 评估指标
| 任务 | 主要指标 |
|------|----------|
| 图像分类 | Top-1 Accuracy |
| 图像描述 | BLEU-4 |
| 文本生成 | Perplexity, BLEU |
| 文本到图像 | CLIP Score（衡量生成图与文本语义匹配度） |
| 部署性能 | 推理延迟（Latency）、帧时间（Frame Time @90Hz）、内存开销 |

### 🆚 基线方法对比
未直接与单一 SOTA 模型比较，而是与以下基准横向对比：
- **NAS-Bench-101/201**, **NATS-Bench**, **TransNAS-Bench-101**, **HW-NAS-Bench**, **JAHS-Bench-201** 等主流 NAS benchmarks（见 Table 1）
- 手工设计 baseline 如 ResNet-LSTM（用于图像描述）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2 和正文）

| 方法 | 前缀 | 模型数量 | 最佳性能（Accuracy / Score） | 成功率 |
|------|------|---------|-------------------------------|--------|
| Genetic Algorithm | `ga-` | 2,000 | **0.8004** (CIFAR-10) | 100% |
| Fractal Networks | `frac-` | 1,258 | **0.8018** (CIFAR-10, 5 epochs) | 97% |
| NN-RAG | `rag-` | 1,289 | **0.9281** (CIFAR-10) | 73.0% |
| Mixture-of-Experts (homogeneous) | `moe-` | 8 | **0.9390** (CIFAR-10) | 100% |
| Mixture-of-Experts (heterogeneous) | `moe-` | 8 | **0.9313** | 100% |
| Image Captioning (best) | `C*C` | 357 | **BLEU-4 = 0.317** | >50% |
| Text-to-Image (CVAE-GAN) | `t2i-` | 3 | **CLIP Score = 0.2751** | 100% |
| Data Augmentation (combinatorial) | — | 6,000 | **Acc = 0.6124** | 100% |
| Data Augmentation (LLM-gen) | — | 280 | **Acc = 0.5728** | 22% |

> ✅ **亮点结果**：
> - MoE 模型表现最优（>93%），且异构 MoE 超过所有单骨干模型。
> - NN-RAG 提取模块中 **73% 可执行成功**，并达到接近 93% 准确率。
> - 组合式数据增强优于 LLM 生成方案（61.24% vs 57.28%）。

### 📊 与基线方法对比
- 在图像分类任务中，**进化类模型（GA, Fractal）中位数准确率显著高于 LLM 生成模型**（Fig. 9）：
  - `ga-`: median acc = **0.6928**
  - `rl-`: **0.5886**
  - `alt-nn1`（few-shot LLM）: **0.3048**
- LLM 生成模型虽峰值性能高（Fig. 8 中 top-10 多为 `alt-` 开头），但**方差大、成功率低**。
- Few-shot prompting 中，**n=3 示例效果最佳**（平均 acc 53.1%），过多示例（n=6）导致上下文溢出，失败率达 99.8%。

### 🔍 消融实验与关键发现
- **AST Mutation 敏感性实验**：局部通道调整对初始模型选择高度敏感，收敛行为差异大。
- **Few-shot Prompting 参数影响**：示例数量严重影响生成稳定性与性能。
- **FractalNet 加速技术有效性**：使用 AMP + Gradient Checkpointing 可扩展至 1,200+ 模型训练。
- **NN-Lite 规模验证**：连续 48 小时处理 **7,512 个模型**，实现全自动转换与延迟测量。
- **NN-VR 兼容性检查**：95% 模型可通过 Barracuda 成功部署至 Unity，支持 VR 实时推理分析。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **多样化生成策略能有效探索更广的架构空间**，超越传统 cell-based 搜索限制。
2. **优化驱动的方法（GA、Fractal）比纯随机或 LLM 生成更具稳定性与性能密度**。
3. **LLM 在 few-shot 设置下可生成高性能模型，但需谨慎控制提示长度与去重机制**。
4. **真实设备部署数据（latency, memory）是模型选型的关键依据**，理论指标无法替代。
5. **跨任务架构具有迁移潜力**：例如 MoE、Transformer 解码器可在不同任务间复用。
6. **NN-RAG 实现了“神经架构挖掘”范式**：将分散代码转化为标准化、可执行模块库。

### ⚠️ 方法的局限性
- **LLM 生成成功率较低**：尤其在复杂任务中，语法错误、依赖缺失等问题频发。
- **训练预算有限**：多数模型未完全收敛，报告的是“计算效率”而非绝对性能上限。
- **硬件覆盖仍有限**：目前主要支持 Android 与 Unity VR，尚未涵盖更多嵌入式平台（如 Raspberry Pi、Jetson）。
- **缺乏理论解释性**：大量生成模型的行为机制尚不明确，黑箱程度较高。

### 🔮 未来工作方向
1. **扩展至更多模态与任务**：如视频理解、音频合成、强化学习策略网络。
2. **引入更强的 LLM 微调机制**：基于 LEMUR 数据集 fine-tune 专用 **NNGPT** 模型，提升生成质量与成功率。
3. **构建动态更新机制**：实现在线收集社区提交的新架构，形成开放协作生态。
4. **加强因果分析与可解释性工具**：识别哪些结构模式真正带来性能增益。
5. **深化硬件协同设计**：结合芯片级特性（如 Tensor Core、NPU 支持）进行定制化架构生成。

---

## 总结

> 🌟 **LEMUR 2 定义了一个新的 AI 设计范式：以大规模、多样性、可部署性的神经网络数据集为基础，打通“生成—训练—评估—部署”全链路，推动 LLM-driven AutoML 向实用化迈进。**

它不仅是 benchmark，更是**下一代 AutoML 的基础设施**，为架构泛化、跨模态迁移、边缘智能提供了坚实的数据支撑。

</details>

---

### 10. [Physical activities enable scalable foundation modelling for broad-spectrum health prediction](https://arxiv.org/abs/2607.06954)

**Authors**: Zhenghuang Wu, Yuyao Zhu, Songlin Xu  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.06954v1  

#### Abstract
Wearable and mobile sensing technologies have demonstrated strong potential for health inference; however, most sensor models are designed for specific disease types, limiting their transferability across different health risks. Wearable foundation models offer a more generalizable approach in diver...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Physical activities enable scalable foundation modelling for broad-spectrum health prediction*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前可穿戴设备健康预测模型存在以下瓶颈：
- **依赖高频率原始传感器数据**（如 IMU、PPG）：引发隐私泄露风险（如通过加速度计识别用户行为）、计算开销大、难以跨设备扩展。
- **专用模型设计**：多数模型针对特定疾病训练，缺乏在多种健康风险任务间的泛化能力。
- **可扩展性差**：高维信号处理限制了其在大规模人群监测中的应用。

本文提出：**能否仅用低维度、广泛可用且隐私友好的步数数据构建一个通用的健康基础模型？**

---

### 🚀 提出的新方法与新思路
作者提出了 **StepFM** —— 一种**完全基于步数计数数据**（step count）的**基础模型**（foundation model），用于广谱健康风险预测。

#### 创新点包括：
1. **首次将 step count 视为统一的基础信号**  
   将传统上用于单一任务的步数数据提升为支持超过 20 种不同健康风险预测的通用表征基础。

2. **设计了面向步数的行为感知预训练框架**（behavior-aware pre-training）
   - 引入 **log-scaled tokenization**：对高度偏态分布的步数进行非线性离散化，增强低中强度活动的分辨率。
   - 加入 **Fourier-based temporal rhythm encoding**：显式编码昼夜节律和周周期模式，使相同步数在不同时段具有不同语义。
   - 构建 **双流架构**（dual-stream architecture）：
     - **Macro Stream**：建模小时级宏观趋势（Mamba backbone）；
     - **Micro Stream**：利用 1D CNN 提取分钟级活动形态（如持续跑步 vs 零星走动），并通过 **FiLM** 动态调制宏观状态。

3. **引入分层表型对齐目标**（hierarchical activity phenotype alignment）
   在预训练阶段联合优化：
   - 自回归下一token预测（next-token prediction）
   - 显式对齐小时、日、周三个尺度上的临床相关活动表型（如久坐负担、节律稳定性等），增强表示的医学可解释性。

4. **强调“低维信号也能支撑强泛化”范式转变**
   挑战了“必须依赖高维生理信号”的主流认知，证明简单行为信号在大规模建模下仍能实现高效、隐私保护且可扩展的健康推断。

---

### 🔍 相比现有方法的优势
| 维度 | StepFM | 传统方法 |
|------|--------|---------|
| **输入数据** | 步数（低维、通用、隐私友好） | 原始 IMU/PPG（高维、敏感、设备绑定） |
| **模型通用性** | 支持 >20 种异构健康任务迁移 | 多为单任务或窄领域专用模型 |
| **计算效率** | 参数仅 3.4M，适合边缘部署 | 高频信号处理资源消耗大 |
| **跨域泛化** | 跨设备、跨地区、跨疾病表现稳健 | 易受传感器差异影响 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

#### （1）预训练数据集
- **NHANES 2011–2014**
  - 来源：美国国家健康与营养调查
  - 数据形式：**分钟级步数序列**（来自 ActiGraph GT3X+ 手腕加速度计）
  - 处理方式：使用 `scrfsteps` 算法从原始加速度信号提取步数
  - 规模：约 **14,000 名参与者**，共 **1.41亿条分钟级观测记录**

#### （2）下游微调与评估数据集
| 数据集 | 特点 |
|-------|------|
| **NHANES 2005–2006** | 腰部佩戴 ActiGraph，验证跨设备泛化能力 |
| **BarKA-MS** | 瑞士多发性硬化患者队列，使用 Fitbit 步数，含新型神经功能障碍标签（disability, fatigue） |
| **RESILIENT** | 英国老年人群，使用 ScanWatch 步数，包含焦虑、抑郁、认知障碍等心理量表衍生标签 |

---

### ⚙️ 实验设置与评估指标

#### 评估协议
- **全样本线性探测**（full-shot linear probing）为主
- 同时测试 **少样本场景**（30% 标注数据）
- 主干网络冻结，仅训练轻量 MLP 头部 → 验证表征质量

#### 评估指标
- **AUROC**（主要指标）：因多数疾病正例稀疏（如 emphysema 占比 <1.5%），更鲁棒
- **F1 Score**：辅助衡量分类平衡性

#### 下游任务数量
- 共 **21 项健康风险预测任务**，涵盖五大类：
  - 心血管循环系统
  - 代谢内分泌肾病
  - 呼吸免疫肿瘤
  - 神经精神感官
  - 新增罕见病与心理量表任务（如 GAD-7 定义的焦虑）

---

### 🆚 基线方法对比

| 类别 | 基线模型 |
|------|--------|
| **通用时间序列模型** | TimeSiam（自监督对比重建） |
| **生理信号基础模型** | NormWear（多模态 PPG/ECG/IMU） |
| **可穿戴运动基础模型** | PAT（NHANES actigraphy transformer） |
| **PPG 基础模型** | Pulse-PPG（开源 PPG foundation model） |
| **步数表示学习模型** | SSCP（对比学习预测抑郁症改善） |
| **传统机器学习基线** | Trad. ML：基于手工特征的逻辑回归（LR），含 circadian rhythm、sedentary pattern 等 99 个特征 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）

| 模型 | 平均 AUROC（21 任务） |
|------|---------------------|
| Pulse-PPG | 0.5947 |
| PAT | 0.6314 |
| SSCP | 0.6566 |
| TimeSiam | 0.6704 |
| NormWear | 0.7079 |
| **Trad. ML** | **0.7065** |
| **StepFM（Ours）** | **0.7318** ✅ |

> 💡 StepFM 在 **20/21 个任务中排名第一**，平均 AUROC 超过最强基线 NormWear 达 **+2.4个百分点**，相对提升约 **3.4%**。

#### 典型优势任务举例：
| 任务 | StepFM AUROC | 最佳基线 | 提升幅度 |
|------|-------------|----------|----------|
| Diabetes | **0.8509** | 0.8072 (NormWear) | +4.37% |
| Heart Failure | **0.8320** | 0.8089 (NormWear) | +2.31% |
| Stroke | **0.8284** | 0.7985 (NormWear) | +2.99% |
| Depression | **0.6642** | 0.6499 (NormWear) | +1.43% |

> 表明 StepFM 对心血管与代谢类疾病的预测尤其出色，符合流行病学证据中步数与这些疾病强相关的结论。

---

### 🔍 消融实验结果（Table 2 & 图分析）

逐步添加模块后的平均 AUROC 变化：

| 模型变体 | 平均 AUROC |
|--------|-----------|
| Vanilla Mamba（仅自回归） | 0.7007 |
| + Tokenization + Temporal Encoding | 0.7095 (+0.88%) |
| + Micro Stream with FiLM | 0.7282 (+1.87%) |
| **完整 StepFM（+ Phenotype Alignment）** | **0.7318 (+0.36%)** |

> 结论：
- **Tokenization 和 temporal encoding** 显著提升低强度活动建模能力；
- **Micro Stream + FiLM** 是最大增益来源，说明捕捉细粒度活动形态至关重要；
- **Phenotype alignment** 进一步稳定长期行为模式建模。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **步数是强大的数字生物标志物（digital biomarker）**
   - 尽管是低维信号，但在大规模建模下仍能有效预测多种健康风险，尤其是心血管与代谢性疾病。
   - AUROC 曲线在所有模型间呈现“平行振荡”，表明**疾病可预测性主要由其与身体活动的内在生物学关联决定**，而非模型结构本身。

2. **StepFM 实现了真正的跨域泛化**
   - **跨设备**：在手腕（预训练）、腰部（NHANES 05-06）、商用手环（Fitbit, ScanWatch）上均表现优异；
   - **跨区域**：在美国、瑞士、英国人群中保持领先性能；
   - **跨疾病**：成功迁移到未见过的 MS-related disability（AUROC=0.8894）和 GAD-7 定义的焦虑（AUROC=0.7890）。

3. **数据效率高，适用于小样本场景**
   - 仅需 **30% 标注数据**即可达到 95% 以上全量性能（图 2a），特别适合罕见病建模。

4. **时间窗口越长，预测越准**
   - 输入从 1 天增至 7 天，平均 AUROC 从 0.6928 提升至 0.7318，说明长期节奏比短期快照更具判别力。

5. **中间层表征最优**
   - 层级探针显示：**第 2 层 Mamba 输出的表征最有利于下游任务**，深层反而退化 → 表明自监督目标与临床任务存在错位，需谨慎选择特征抽取层。

---

### ⚠️ 方法的局限性

1. **对弱相关疾病的预测能力有限**
   - 如贫血（Anemia）、慢性支气管炎等与步数关系较弱的任务，所有模型 AUROC 均偏低（~0.5–0.6），反映信号本身的信噪比限制。

2. **F1 分数普遍较低**
   - 因正例极度稀疏（如 angina 占比 1.4%），即使 AUROC 较高，F1 也受限于假阳性惩罚。

3. **未融合其他模态**
   - 当前仅为纯步数模型，在需要精细生理信息的任务（如房颤检测）上可能不如多模态模型。

---

### 🔮 未来工作方向

1. **与其他轻量模态结合**
   - 探索 screen usage、mobility traces 等被动信号作为补充，构建更丰富的“最小传感集”。

2. **集成到多模态 foundation model 中**
   - 将 StepFM 的输出作为上下文先验，指导 IMU 或 PPG 模型的注意力机制，提升效率与可解释性。

3. **个性化适配机制研究**
   - 开发轻量适配器（adapter）或 prompt tuning 技术，进一步降低下游标注需求。

4. **真实世界部署验证**
   - 在医院康复、慢病管理等实际场景中测试模型的临床效用与用户接受度。

---

## 总结

📌 **StepFM 成功验证了一个核心理念：**
> “**Ubiquitous low-dimensional behavioral signals, when modeled at scale, can serve as a practical and powerful foundation for broad-spectrum health inference.**”

它不仅提供了一种**隐私友好、计算高效、易于推广**的健康建模范式，也为未来移动健康系统的设计提供了新的理论依据和技术路径。

</details>

---

### 11. [Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata](https://arxiv.org/abs/2607.07026)

**Authors**: Meihua Dang, Stefano Ermon  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.07026v1  

#### Abstract
Constrained decoding is essential for serving LLMs, ensuring that generated outputs follow specific structures such as JSON schema-formatted function calls. Existing systems are designed for autoregressive models and assume left-to-right generation, masking out invalid next tokens at each step. Diff...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Constrained Decoding** 技术主要针对 **autoregressive LLMs** 设计，通过在每一步生成中屏蔽违反约束的下一个 token 来保证输出符合特定结构（如 JSON Schema、SQL 语法等）。然而，**diffusion language models (dLLMs)** 在每个去噪步骤中会并行采样多个位置的 token，其生成过程是全局而非左到右的。因此，传统的逐 token 屏蔽策略无法有效防止局部合法但组合后非法的序列产生。

本文旨在解决 **如何在 dLLMs 上实现精确且高效的 constrained decoding**，确保生成的完整序列满足由 **finite automaton (FA)** 定义的硬性结构约束。

### 提出的新方法与创新思路
作者提出了一种 **基于有限自动机（finite automaton）的精确推断算法**，用于在每个扩散步骤中从受约束的均场后验分布（constrained mean-field posterior）中进行采样：

- **将有限自动机视为图模型（graphical model）**：  
  将 FA 转化为一个链式结构的隐马尔可夫模型（HMM），其中隐变量表示自动机的状态转移路径，观测变量对应生成的 token 序列。该模型的支撑集（support）恰好是所有满足约束的序列集合 $ C $。

- **构造受约束的联合分布**：  
  在每个去噪步骤中，不是直接从模型的均场预测 $ p_\theta(x^0|x') $ 中采样，而是从以下受约束的分布中采样：
  $$
  p(x^0|x', C) \propto p_\theta(x^0|x') \cdot \mathbf{1}[x^0 \in C]
  $$
  通过将自动机定义的概率分布 $ p_M(x) $ 与模型预测相乘，得到一个可高效推理的链式结构模型。

- **高效对数深度采样（Log-depth Tree Sampling）**：  
  利用算术电路中的深度约简技术（depth-reduction），将标准前向-后向算法的 $ O(L) $ 串行计算深度降低至 $ O(\log L) $。具体做法是构建一个二叉树状的递归分解结构，在每一层并行处理子区间的中间状态采样，从而极大提升 GPU 并行效率。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **正确性保障** | 保证最终输出 **100% 满足约束**（by construction），而 rejection sampling 或 rejection-then-verify 类方法无法在有限预算内保证成功。 |
| **兼容性广** | 支持 **greedy decoding** 和 **stochastic sampling**，适用于任意 remasking schedule 的 **parallel/block-wise decoding**。 |
| **效率高** | 推理开销极低（<5% wall-clock overhead），尤其在长序列上得益于 $ O(\log L) $ 深度设计，显著优于 naive $ O(L) $ 方法。 |
| **通用性强** | 可处理 **nondeterministic finite automata (NFA)**，支持比 DFA 更复杂的约束表达（如 Spider 的 SQL grammar）。 |

---

## 2. 核心实验方法和设置

### 使用的数据集与任务
实验覆盖五类典型结构化生成任务，编码为 DFA 或 NFA：

| 任务 | 数据集 | 约束类型 | 示例 |
|------|--------|----------|------|
| **Function Calling** | xLAM, BFCL | JSON/Python 函数调用格式 | `{"name": "get_weather", "args": {...}}` |
| **Planning** | Sudoku (4×4), Countdown | 数字谜题格式与规则 | 行列不重复、算术表达式合法性 |
| **Text-to-SQL** | Spider | SQL 语法 + 数据库 schema 合法性 | 使用正确的表名、列名、关键字 |
| **Math Reasoning** | GSM-Symbolic | 符号表达式包裹于 `《》` 内 | 自然语言中嵌套数学公式 |

### 实验设置
- **模型**：  
  - **Dream-7B** 和 **LLaDA-8B**（均为开源 dLLMs）
  - 使用 Instruct 版本用于 function calling、Spider、GSM；Base 版本用于 Sudoku、Countdown
- **解码方式**：
  - Greedy decoding ($T=0$)
  - Stochastic sampling ($T=1$)
- **评估指标**：
  - **Exact Match (EM)**：完全匹配黄金答案
  - **Execution Accuracy**（Spider）：查询执行结果一致
  - **Constraint Satisfaction Rate (CS)**：输出是否符合语法/格式约束
- **基线方法**：
  - **Unconstrained baseline**：原始 dLLM，无任何约束控制
  - **DINGO**：早期基于 FA 的 MAP 推理方法，仅支持贪婪解码
  - **Lookahead-then-Verify** [37]：基于 CFG 的采样验证方法，不保证收敛

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 **BFCL-Live (JSON)** 上的表现（Dream-7B）：
| 方法 | Greedy EM | Sampling EM | CS Rate |
|------|-----------|------------|---------|
| Unconstrained | 63.9% | 22.3% | 31.2% |
| **Ours (Constrained)** | **71.5%** | **69.0%** | **100%** |

> ✅ **改进幅度巨大**：采样模式下准确率从 **22.3% → 69.0%**，接近贪婪解码水平，几乎消除了“采样崩溃”现象。

#### 在其他任务上的提升：
- **Sudoku**：Dream-7B 从 36.4% → **92.4%**
- **Countdown**：Dream-7B 达到 **100%**（原为 35.2%）
- **xLAM (JSON)**：Dream-7B 从 50.5% → **75.7%**
- **Spider**：Dream-7B 从 15.5% → **52.1%**

### 与基线方法的对比
- **远超 unconstrained baseline**：特别是在 stochastic sampling 下，基线严重退化，而本文方法保持稳健。
- **优于 DINGO**：DINGO 仅支持 greedy decoding，且未提供采样能力；本文方法是其严格泛化。
- **优于 rejection-based 方法**：后者接受率极低，尤其在复杂约束下难以生成有效样本。

### 消融实验结果（Table 2 & Table 3）

#### 消融：Remasking Confidence 策略
| 策略 | 描述 | 性能影响 |
|------|------|--------|
| **Mf** | 使用原始模型的 marginal $ p(x_i|x') $ 作为置信度 | 提升有限 |
| **Mar** | 使用 **受约束分布下的 marginal $ p(x_i|x', C) $** 作为置信度 | 显著更优，尤其当 baseline 性能差时 |

> 🔍 发现：仅“过滤”无效 token 不够，必须重新计算受约束下的 token 重要性以引导解码顺序。

#### 运行时开销（Table 3）
| 方法 | Accuracy | Wall-clock Overhead |
|------|----------|---------------------|
| Unconstrained | 67.5% | 0% |
| Ours (chain) | 78.5% | +114% |
| **Ours (tree)** | **79.2%** | **+4%** |

> ⚡️ **关键优势**：采用 tree-structured sampling 后，精度大幅提升的同时，运行时开销仅增加 **4%**，实际可用性强。

---

## 4. 关键结论和发现

### 主要发现
1. **dLLMs 在结构化生成中极易失败**：一旦启用随机采样，unconstrained dLLMs 的 constraint satisfaction rate 断崖式下降，导致性能崩溃。
2. **格式错误是主因而非语义错误**：模型知道“调哪个函数”，但不知道“怎么写成 JSON”。例如 Dream 在 BFCL-Python 上仅得 22.4%，而在约束下可达 69.7%。
3. **我们的方法能恢复采样鲁棒性**：即使在 $ T=1 $ 下也能达到近似 greedy 的性能，真正释放 dLLMs 的多样性潜力。
4. **log-depth sampling 高效实用**：理论上的 $ O(\log L) $ 深度转化为实际的低延迟，GPU 并行利用率高。

### 方法的局限性
- **约束表达能力受限于 finite automata**：无法处理上下文无关文法（CFG）级别的约束（如嵌套括号、递归结构），尽管可通过 NFA 扩展部分覆盖。
- **不解决语义错误**：只能保证语法正确，不能纠正参数值错误（见 Table 9）。例如 location 多写了冗余信息、枚举值选错、遗漏可选参数等仍可能发生。
- **预编译 FA 成本**：对于动态 schema（如每次不同的 JSON Schema），需实时构建 FA，可能引入额外延迟。

### 未来工作方向
- 扩展至 **context-free grammars (CFG)** 或 **tree automata**，以支持更复杂的结构（如嵌套 JSON、完整编程语言）。
- 结合 **semantic verification** 模块，在语法正确基础上进一步筛选语义合理的输出。
- 探索 **adaptive constraint tightening**：根据模型置信度动态调整约束强度。
- 将该框架集成进主流 dLLM inference engine（如 SGLang、vLLM 的扩散分支）。

--- 

> 📌 **一句话总结**：  
> 本文首次实现了在 **diffusion language models** 上的 **精确、高效、通用的 constrained decoding**，通过将 **finite automata 视为图模型** 并结合 **log-depth parallel sampling**，在几乎零额外延迟下大幅提升了结构化任务的准确率与鲁棒性，为 dLLMs 的实际部署扫清了关键障碍。

</details>

---

### 12. [Voltron: Enabling Elastic Multi-Device Execution of LLM Inference for Empowered Edge Intelligence](https://arxiv.org/abs/2607.07046)

**Authors**: Chanwoo Cho, Wooseok Kim, Yonglak Son, Young Seo Lee, Young Geun Kim  
**Category**: cs.DC  
**Published**: 2026-07-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.07046v1  

#### Abstract
Large language models (LLMs) are widely used in intelligent services due to their remarkable capability in generative tasks. Typically, LLM-based services process the inference requests of the users in a centralized data center. Unfortunately, such centralized execution has limitations for end-users...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Voltron: Enabling Elastic Multi-Device Execution of LLM Inference for Empowered Edge Intelligence*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 LLM 推理依赖于**集中式数据中心**执行，存在以下问题：
- **高延迟**：受限于网络通信开销，影响用户体验（QoS）。
- **隐私风险**：用户敏感数据需上传至云端，存在泄露风险。
- **单设备资源限制**：在边缘设备上本地运行 LLM 受限于内存和算力，难以部署大模型。

尽管已有研究通过 **sLLMs**、**quantization** 和 **pruning** 等技术实现 on-device LLM，但这些方法通常导致显著的 **accuracy drop**，且无法突破单设备内存上限。

### 提出了什么新方法或新思路
本文提出 **Voltron** —— 一种全新的、支持弹性多设备协同的 on-device LLM 推理框架。其核心思想是：
> 利用用户身边多个可用的边缘设备（如手机、平板、手表、IoT 设备）组成一个临时的本地计算集群，共同执行 LLM 推理任务。

Voltron 的关键技术创新包括：

#### （1）Layer-wise Hybrid Parallelism (HP)
- 动态为每一层选择最优的并行策略：**Model Parallelism (MP)** 或 **Tensor Parallelism (TP)**。
- 综合考虑设备异构性（compute/memory）、layer 类型（ATTN/FFN）、精度配置等因素，避免“一刀切”策略带来的性能瓶颈。

#### （2）Importance-aware Mixed Precision
- 引入基于通道重要性的离线分析机制，为每层分配不同精度（FP16/INT8/INT4）。
- 对重要层保留高精度以维持 accuracy，对非关键层降低精度以节省 memory 和 latency。

#### （3）Elastic Model Execution
- 在推理过程中持续监控运行时变化（KV cache 增长、无线信号波动），动态调整执行策略：
  - **Computation Scaling**：通过 precision scaling + importance-aware pruning 应对内存不足。
  - **Communication Scaling**：量化 activation 数据以应对弱网络连接。
- 支持断连恢复与精度回滚，提升鲁棒性。

#### （4）Energy Optimization
- 当优先节能时，利用 DVFS 技术降低电压频率，在满足 QoS 的前提下大幅减少能耗。

### 相比现有方法的优势
| 维度 | 现有方法（如 sLLM, quantization） | Voltron |
|------|-------------------------------|--------|
| 准确率 | 显著下降（受限于小模型或低精度） | 更高（可运行更大模型 + 智能混合精度） |
| 内存利用率 | 单设备限制 | 多设备聚合，突破单机瓶颈 |
| 自适应能力 | 静态配置，难适应动态环境 | 实时感知并弹性调整执行策略 |
| 隐私保护 | 云端处理有风险 | 完全本地化执行 |
| 能耗控制 | 通常不优化 | 支持主动节能模式 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **输入负载**：采用 `LMSYS-Chat-1M` 数据集中的对话轨迹，模拟真实场景下的 variable input/output length。
- **准确率评估基准**：
  - **MMLU** (0-shot)
  - **Hellaswag** (0-shot)
  - **GSM8K** (8-shot, CoT)
  - **MATH** (4-shot, CoT)

### 实验设置
#### 硬件平台
使用六种移动设备构建三个典型场景 cluster（见 Table 1 & 2）：

| Cluster | 场景 | 包含设备 |
|--------|------|---------|
| Cluster 1 (Car) | 车载环境 | Galaxy S22 Ultra (Mobile), Pixel 7 (Auto SoC), Pixel 5 (Infotainment) |
| Cluster 2 (Office) | 办公室 | S22 Ultra, Lenovo Legion Y700 (Tablet), Pixel 2 XL (Watch) |
| Cluster 3 (Home) | 家庭 | S22 Ultra, Pixel 2 XL (Watch), Nexus 5X (IoT) |

所有设备间通过 **Wi-Fi Direct** 连接。

#### 模型家族
测试了三大主流开源 LLM 家族共 9 个模型（见 Table 3）：
- **gemma-3** (4B, 12B)
- **Qwen 1.5** (4B, 7B, 14B, MoE-A2.7B)
- **Qwen 2.5** (3B, 7B, 14B)

均支持 **mixed precision**（FP16/INT8/INT4）。

### 评估指标
| 指标 | 描述 |
|------|------|
| **TTFT** | Time to First Token，归一化到 QoS 上限（10s） |
| **TPOT** | Time per Output Token，归一化到 QoS 上限（400ms） |
| **Accuracy** | 四项 benchmark 的平均得分 |
| **Energy Consumption** | 使用外部功耗仪测量 |
| **Overhead** | 执行计划模块、精度调整等引入的时间开销 |

### 基线方法对比
1. **Single-device execution**：仅在集群中最强设备（S22 Ultra）上运行。
2. **Device-heterogeneity-aware MP/TP**：基于设备性能分配层或张量片，但未考虑 layer 架构差异和动态变化。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- Voltron 在所有 cluster 和 model 上均成功满足 **TTFT ≤ 10s** 和 **TPOT ≤ 400ms** 的 QoS 要求。
- 平均 **accuracy 提升达 10.2%**，最高提升 **16.5%**（相比 single-device baseline）。
- 在极端情况下（KV cache 达 50k tokens），baseline 已违反 QoS，而 Voltron 仍可通过 computation scaling 维持服务。

### 与基线方法的对比结果（图 10 & 表 4）
| 方法 | Average Accuracy | 是否满足 QoS |
|------|------------------|-------------|
| Single-device | 0.6232 ~ 0.6903 | 是（但 accuracy 低） |
| MP/TP（异构感知） | 0.7262 ~ 0.7425 | 否（部分场景失败或超时） |
| **Voltron** | **0.7263 ~ 0.7485** | ✅ 全部满足 |

> 示例：在 `Qwen2.5-14B` 上，Voltron 将 MMLU 准确率从 0.7047 提升至 **0.7747**（+6.9%）。

### 消融实验结果
#### （1）Adaptability to Runtime Variance（图 12）
- **计算动态性**（KV cache 增长）：
  - 当 cache size 从 0k 增至 50k，baseline TPOT 超出 QoS。
  - Voltron 通过 precision scaling + pruning 成功维持 TPOT 在阈值内，accuracy 下降 < 2%。
- **网络波动**（RSSI 从 -30dBm → 断连）：
  - MP/TP 在弱信号下严重超时或中断。
  - Voltron 通过 communication scaling 维持低延迟，并在断连后仅牺牲少量 accuracy（via pruning）继续运行。

#### （2）Energy Optimization（图 13）
- 在启用 energy optimization 后，能量消耗最多可降低 **59.0%**。
- 即使压缩率达 30%，accuracy 也仅下降 **1.9%**，远优于 naive 方法。

#### （3）Overhead Analysis（表 5.2.4）
- **Execution Plan Module**：平均耗时 0.1ms（占 TPOT 的 0.03%）
- **Precision Allocation**：平均 0.8ms（得益于 binary search，比 exhaustive search 快 30 倍）
- **Plan Modification Overhead**：通过 I/O overlapping 和 shard-wise allocation，将修改开销从 31.7ms 降至 **4.1ms/token**

✅ 总体 overhead 仅占 TPOT 的 **1.2%**，支持 token-level 动态调整。

---

## 4. 关键结论和发现

### 主要发现
1. **多设备协同是突破 on-device LLM 瓶颈的有效路径**：聚合多个边缘设备资源可运行更大模型，显著提升 accuracy。
2. **静态并行策略不适用于异构边缘环境**：必须结合 device capability、layer architecture、precision 配置进行细粒度决策。
3. **runtime variance 是现实部署的关键挑战**：KV cache 增长和无线信号波动会严重影响性能，需具备弹性适应能力。
4. **Voltron 实现了 accuracy、latency、energy 的良好权衡**：不仅提升性能，还能主动优化能耗。

### 方法的局限性
- **依赖设备间稳定连接**：虽然支持弱网和断连，但频繁切换会影响效率。
- **初始冷启动开销**：首次建立执行计划需要 profiling，可能略增加首 token 延迟。
- **跨设备同步复杂性**：目前假设设备时间同步较好，实际中可能存在 clock skew 问题。
- **未考虑设备电量状态**：若某设备即将关机或低电，可能影响整体稳定性。

### 未来工作方向
- 支持更广泛的设备类型（如 PC、AR glasses）。
- 引入联邦学习思想，实现多用户设备间的协作推理。
- 结合 LLM 编译器进一步优化 shard 分割与调度。
- 探索基于 RL 的自适应策略选择机制。
- 开源 Voltron 框架以推动社区发展（作者已承诺接受后开源）。

---

> ✅ **总结一句话**：  
> Voltron 通过 **layer-wise hybrid parallelism + importance-aware mixed precision + elastic execution**，首次实现了高性能、高准确率、强鲁棒性的多设备协同 LLM 推理，为 empowered edge intelligence 提供了可行的技术路径。

</details>

---

### 13. [Fractal KV-Cache Archives: Lossless Symbolic Storage with In-Place Retrieval for Long-Context LLM Inference](https://arxiv.org/abs/2607.07144)

**Authors**: Vladimir Gusev  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.07144v1  

#### Abstract
The key-value (KV) cache dominates the memory cost of long-context autoregressive inference, and a growing body of work compresses it through quantization, eviction, or offloading. We study a complementary question: once a position's KV state has been quantized to codebook indices, how should the re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Fractal KV-Cache Archives: Lossless Symbolic Storage with In-Place Retrieval for Long-Context LLM Inference*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在长上下文（long-context）的 **autoregressive inference** 中，Transformer 模型的 **Key-Value (KV) cache** 占据了绝大部分内存开销，成为推理时的瓶颈。现有的主流方法（如量化、eviction、offloading）虽然能压缩存储，但通常将 KV cache 视为“惰性数据块”，忽略了其作为可检索记忆的潜力。

本文提出一个更深层次的问题：  
> 当 KV 状态已经被量化为 codebook indices（符号流）后，如何高效地**存储这些符号**？能否让存储层本身支持快速随机访问和检索，而不仅仅是被动存放？

### 提出的新方法与创新思路
作者引入并重构了一类经典的 **contractive iterated-map code**（收缩迭代映射码），将其应用于量化后的 KV cache 存储，形成一种名为 **Fractal KV-Cache Archive** 的新型存档格式。该方法的核心是：

- 将每个 token 的量化索引序列编码为平面上的一个低维实数向量轨迹（即“行走”在分形结构上）。
- 利用这种编码天然具备的数学性质，实现：
  - **Lossless compression**：无损还原原始符号序列。
  - **O(1) 随机访问** 和 **O(1) 摊销追加**：满足动态增长 cache 的操作需求。
  - **In-place substring retrieval**：直接在压缩向量空间中执行近似子串匹配查询，无需解压全文。

这一设计统一了三个功能于一身：**存储器（store）、解码器（decoder）、搜索索引（index）**。

### 相比现有方法的优势
| 维度 | 传统方法（如 byte compressor） | 本文方法（Fractal Archive） |
|------|-------------------------------|-----------------------------|
| 存储效率 | 高压缩率但不可随机访问 | 压缩率相近（≈ bit-packing），但支持随机访问 |
| 访问模式 | 必须解压整个流才能读取中间位置 | 支持 O(1) 随机访问任意位置 |
| 扩展性 | 追加需重写或复杂管理 | 支持 O(1) 摊销时间追加 |
| 功能扩展 | 仅用于存储 | 同时作为检索索引，支持 context-aware 查询 |

> ✅ **核心创新**：不是提出新的量化算法，而是重新定义了量化之后的“存储层”角色——从被动容器变为多功能主动组件。

---

## 2. 核心实验方法和设置

### 数据集
- 主要使用 **Project Gutenberg 上的《The Time Machine》**（公共领域文本）
- 所有实验基于此单一语料进行训练与评估

### 模型
- **GPT-2 (124M)**，共 12 层，每层 12 个 attention head
- 上下文长度固定为 **1024 tokens**

### 实验设置
- **Exact Window**：保留前 4 个 token（attention sinks）和最近 32 个 token 的精确 KV 值
- 其余历史 token 的 KV 向量被量化后送入 **Fractal Archive**
- 量化方式：
  - **Vector Quantization (VQ)** 或 **Residual VQ (RVQ)**
  - 分 **pooled codebooks**（跨 head 共享）与 **per-head codebooks**（每个 head 独立）
- Codebook 使用 **k-means** 在独立文本段落上预训练

### 评估指标
| 指标 | 描述 |
|------|------|
| **Perplexity (PPL)** | 衡量语言建模质量，相对于原始 fp16 cache 的增量 ΔPPL |
| **Storage (B/token)** | 序列化后的索引流大小（经 fractal codec 编码） |
| **Compression Ratio** | 相对于 fp16 KV cache（36,864 B/token）的压缩倍数 |
| **Random Access Latency** | 单次随机读取延迟（μs） |
| **Append Latency** | 单 token 追加延迟（μs） |
| **Retrieval Recall/Precision** | 子串匹配的检索性能 |

### 基线方法对比
- **Pooled VQ/RVQ**：共享 codebook，典型 baseline
- **Exact KV cache (fp16)**：无压缩上限 baseline
- **Bit-packing / general-purpose compressors**：用于比较存储效率
- 隐含对比：FibQuant [5], RetroInfer [6]（同期工作，未直接对比）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & Figure 2）

| 配置 | ΔPPL ↑ | 第二半段 ΔPPL | B/token | 压缩比 (vs. fp16) |
|------|--------|---------------|---------|------------------|
| Pooled VQ, k=256 | +51.0% | +72.6% | 342 | 108× |
| Per-head VQ, k=256 | **+23.7%** | **+32.0%** | 342 | 108× |
| Per-head RVQ, k=256×2 | +15.0% | +22.4% | 684 | 54× |
| **Per-head Hybrid (K×4, V×2)** | **+11.2%** | **+16.6%** | 1025 | **36×** |

> 🔍 注：相同比特预算下，**per-head codebooks 显著优于 pooled**；引入残差结构进一步提升率失真表现。

### 与基线方法的对比结果
- 在相同 bit rate 下，**per-head codebooks 比 pooled 方案降低约 27–36 PPL 点**
- 最优配置（Hybrid K×4/V×2）在 **36× 压缩比** 下仅带来 **+11.2% 的 perplexity 损失**
- 相比之下，pooled 方案在更高压缩比下损失更大，说明参数分配不合理

### 消融实验结果
#### （1）Key / Value 不对称性分析
| 条件 | ΔPPL |
|------|------|
| Only quantize **values** (keys exact) | +4.0% |
| Only quantize **keys** (values exact) | +14.5% |
| Quantize both | +15.0% → 几乎等于独立叠加 |

👉 发现：**key quantization 的损害是 value 的 ~4 倍**  
原因：keys 决定 attention routing（通过 query-key dot product），误差直接影响路由准确性；而 values 是加权平均输入，噪声被平滑。

#### （2）混合位分配策略（Hybrid Scheme）
- 利用上述不对称性，将更多 RVQ stage 分配给 keys（如 K×4, V×2）
- 结果验证：该策略显著优于对称分配，在同等总比特下达到最低 ΔPPL

#### （3）Fractal Codec 性能（Table 1）
| 操作 | 成本 |
|------|------|
| Encode | 0.68 μs/char |
| Random Access (in 1M-char doc) | 311 μs/lookup |
| Append (amortized) | 175 μs/token |

✅ 所有 round-trip 测试均 **lossless**

#### （4）In-place Retrieval 性能（Figure 3）
- 支持基于 **suffix similarity** 的近似子串检索
- **Recall = 1.00**（理论保证）
- Precision 受数值精度影响：
  - `float64`（16 B/char）：precision ≈ 0.89–1.00
  - `float32`：仍高，可用作粗筛
- 查询耗时：**~0.9ms / query over 100K positions**
- 支持从匹配点反向解码出原始上下文，无需外部文本

---

## 4. 关键结论和发现

### 主要发现
1. **Per-head codebooks 显著优于 pooled codebooks**  
   在相同比特预算下，per-head 设计全面帕累托占优（Pareto-dominant），应优先采用。

2. **KV 之间存在强烈不对称性：keys 更敏感**  
   - Key quantization 对 PPL 影响约为 value 的 **4 倍**
   - 机制解释清晰：keys 控制 attention 路由，errors 更致命
   - 可转化为 bit allocation 策略，指导 hybrid quantization 设计

3. **Fractal Archive 是多功能一体化结构**  
   - 不只是存储，更是：
     - 支持 O(1) 随机访问与追加的动态结构
     - 天然的 suffix-based 检索索引
   - 检索可在压缩空间内完成，极大减少 I/O 开销

4. **方法轻量且可复现**  
   - 所有实验可在 **单核 CPU 上几分钟内运行完毕**
   - 完整代码开源：[https://github.com/eighteight/fractal-kv](https://github.com/eighteight/fractal-kv)
   - “Every number reproduces from a single command”

### 方法的局限性
- 实验规模小：仅在 **GPT-2 (124M)** 和单一文本上验证
- 上下文长度固定为 1024，未测试超长 context（如 100K+）
- 使用 **per-corpus trained codebooks**，缺乏通用性，部署需在线学习或跨域适应
- 检索能力限于 **exact/approximate suffix matching**，非语义检索
- 存储压缩比主要来自 VQ，fractal codec 本身不优于通用压缩器（优势在于功能而非体积）

### 未来工作方向
- 将该存储框架扩展到更大模型（如 Llama 系列）和多样化任务（如 QA、摘要）
- 探索 **online codebook training** 或 **universal codebooks**
- 结合语义 embedding，发展支持语义检索的 fractal index
- 与更强量化器（如 FibQuant）结合，构建更高效的端到端系统
- 探索硬件加速（如 GPU 并行化 fractal walk）

---

> 📌 **一句话总结**：  
> 本文提出了 **Fractal KV-Cache Archive** —— 一种将量化后的 KV cache 符号流编码为分形轨迹的新型存储格式，它不仅是高效的 lossless 存储，还天然支持 O(1) 随机访问与 in-place 子串检索，并揭示了 key/value 量化敏感性的 4× 差距，指导出更优的 hybrid 量化策略，在 GPT-2 上实现了 **36× 压缩比 + 仅 +11.2% PPL 损失** 的优异表现。

</details>

---

### 14. [MILES: Modular Instruction Memory with Learnable Selection for Self-Improving LLM Reasoning](https://arxiv.org/abs/2607.06974)

**Authors**: Ruilin Tong, Dong Gong  
**Category**: cs.CL  
**Published**: 2026-07-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.06974v1  

#### Abstract
Large language models (LLMs) increasingly improve their reasoning at test time via additional computation, yet most existing works treat each problem in isolation. When problems arrive sequentially, accumulating reusable experience across them can further improve performance. Existing memory-based m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MILES: Modular Instruction Memory with Learnable Selection for Self-Improving LLM Reasoning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的大语言模型（LLM）推理增强方法在测试时通常面临以下挑战：
- **孤立处理问题**：多数方法将每个问题独立处理，无法跨问题积累可复用的经验。
- **记忆泛化能力差**：基于记忆的方法要么存储完整解题模板（如 Buffer-of-Thought），难以泛化到新结构的问题；要么采用启发式选择机制（如相似度检索），未针对最终答案正确性进行优化。
- **学习策略受限**：需要大规模标注数据或固定动作空间来训练选择策略，不适用于测试时增量扩展的记忆场景。

### 提出的新方法与核心思想
本文提出 **MILES**（Modular Instruction Memory with LEarnable Selection），一种支持**测试时自改进**的框架，其核心创新包括：

#### ✅ **模块化指令记忆结构（Modular Instruction Memory）**
- 将经验分解为 `(sub-goal embedding, sub-instruction text)` 的不对称对。
  - `sub-goal` 是目标语义的嵌入表示，作为检索键；
  - `sub-instruction` 是自然语言指导文本，用于条件生成。
- 支持灵活组合不同子步骤，实现跨问题的知识迁移。

#### ✅ **粗粒度到细粒度的两层选择机制（Coarse-to-Fine Retrieval & Selection）**
- **Layer 1（粗选）**：基于 sub-goal 嵌入的余弦相似性进行快速检索。
- **Layer 2（精选）**：引入轻量级 per-item selection head，通过上下文特征打分重排序候选单元。
- 该机制实现了**记忆扩展**与**学习选择**的协同演进。

#### ✅ **从自信样本中自监督学习选择策略**
- 利用初始多路径响应的一致性判断“自信”与“不确定”样本。
- 在**自信样本上运行 MCTS**，收集高质量轨迹，并从中提取 `(state, unit, correctness)` 元组训练 selection head。
- 实现无需外部标签、参数冻结下的**在线自我提升**。

### 相比现有方法的优势
| 维度 | MILES | 传统方法（如 BoT, DC, ToT） |
|------|-------|--------------------------|
| 可复用性 | ✅ 模块化 step-level 单元 | ❌ 完整方案或模板 |
| 泛化能力 | ✅ 跨问题组合推理模块 | ❌ 高度依赖问题匹配 |
| 学习机制 | ✅ 自监督、无外部标签、增量更新 | ❌ 固定动作空间 / 外部训练数据 |
| 测试时适应性 | ✅ 动态增长记忆 + 学习选择 | ❌ 无学习或需微调 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共六个复杂推理基准，涵盖数学竞赛与研究生级别考试：
- **MATH-500**：高中数学难题集合
- **AIME 2024 / AIME 2025**：美国数学邀请赛真题
- **GPQA-Diamond**：高难度科学问答（博士水平）
- **MMLU-Pro Physics / Engineering**：物理与工程领域的进阶多任务理解

### 实验设置与评估指标
- **主干模型（Backbone LLMs）**：
  - 闭源模型：`GPT-4.1`, `GPT-4.1-mini`
  - 开源模型：`Qwen3-30B-Instruct`, `GPT-OSS-20B`
- **冻结模型参数**：所有实验均保持 LLM 参数不变，仅训练轻量子网络（selection heads）。
- **评估指标**：
  - 主要指标：**Final-answer accuracy (%)**
  - 辅助分析：accuracy-efficiency tradeoff（计算成本 vs 性能）、transferability、robustness

### 基线方法对比
| 类型 | 方法 |
|------|------|
| Prompt-based | Zero-shot CoT, Self-Consistency (SC) |
| Memory-based | Buffer-of-Thoughts (BoT), Dynamic CheatSheet (DC) |
| Test-time Scaling | Tree-of-Thought (ToT), rStar, DORA |
| 其他 | Retrieval-of-Thought, ArcMemo |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
在多个数据集和模型上的最终准确率表现如下（部分代表性结果）：

| Dataset | Model | ZS-CoT | SC | DC | BoT | **MILES (Ours)** |
|--------|--------|--------|-----|-----|-----|------------------|
| MATH-500 | GPT-4.1-mini | 88.00 | 89.40 | 87.00 | 86.80 | **92.60** |
| AIME 2024 | GPT-4.1 | 87.20 | 88.00 | 83.20 | 84.80 | **91.60** |
| AIME 2025 | GPT-OSS-20B | 86.67 | 90.00 | 36.67 | 33.33 | **93.33** |
| GPQA-Diamond | Qwen3-30B-Instruct | 70.20 | 72.23 | 42.42 | 67.68 | **73.74** |
| MMLU-Pro Physics | Qwen3-30B-Instruct | 87.00 | 88.50 | 63.50 | 86.00 | **88.50** |

✅ **结论**：MILES 在所有设置下**一致匹配或超越基线方法**，尤其在开源小模型上提升显著。

### 与基线方法的对比结果
- 在 `GPT-OSS-20B` 上，DC 和 BoT 表现不佳（prompt 过长导致干扰），而 MILES 仍能达到接近饱和性能（如 AIME 2024 达 93.33%）。
- 在 `AIME` 系列上，MILES 显著优于 ToT、rStar、DORA 等搜索方法。
- 图 2 显示，在相同 token 预算下，MILES 的 accuracy/token 曲线持续上升，而其他方法趋于平缓 → **更高效地利用额外计算资源**。

### 消融实验结果（Ablation Study）

#### （1）组件有效性（Table 4 & Figure 3）
| 实验配置 | AIME 2024 | AIME 2025 |
|---------|-----------|-----------|
| 无 Layer 1 & 2（无记忆） | 60.00 | 53.33 |
| 仅有 Layer 1（仅相似性） | 63.33 | 53.33 |
| 仅有 Layer 2（仅分类器） | 63.33 | 56.67 |
| **完整 MILES（Layer 1 + 2）** | **66.67** | **60.00** |

➡️ 结果表明：**两层机制协同作用效果最佳**，且 learned selection head 明显优于 LLM prompt reranking 或 state similarity reranking。

#### （2）MCTS Rollout 数量影响（Table 5）
| Rollouts | Memory Size | AIME 2024 Acc | AIME 2025 Acc |
|----------|-------------|---------------|---------------|
| 20 | 9 | 60.00 | 53.33 |
| 30 | 15 | 66.67 | 56.67 |
| 40 | 20 | 66.67 | 60.00 |

➡️ 更多 rollouts → 更丰富的 memory → 更高性能，验证了记忆覆盖范围的重要性。

#### （3）交叉模型迁移能力（Table 2 & 3）
- 使用 `Qwen3-30B-Instruct` 构建 memory，迁移到 `GPT-4.1-mini` 推理：
  - AIME 2024：从 60.00 → **66.67**
  - AIME 2025：从 50.00 → **60.00**
- 使用历史年份 AIME 数据预构建 memory，在 AIME 2025 不确定样本上提效明显。

➡️ 证明 MILES 支持**跨模型、跨时间的知识迁移**，具备良好 transferability。

---

## 4. 关键结论和发现

### 主要发现
1. **模块化 step-level 记忆是实现测试时自改进的有效途径**  
   MILES 成功展示了如何在不更新 LLM 参数的前提下，通过构建可复用的 `(sub-goal, sub-instruction)` 单元并学习其应用策略，持续提升推理能力。

2. **learnable selection 比 heuristic selection 更优**  
   基于 confidence-driven 自监督训练的 selection head 能有效识别何时应用某个推理模式，显著优于纯相似性或 prompt-based 选择。

3. **记忆与推理形成正向循环**  
   自信样本用于扩充记忆和训练选择器，不确定样本则受益于已学得的记忆指导，形成“越用越强”的良性循环。

4. **具有良好的效率-精度权衡与鲁棒性**  
   - 随着计算预算增加，性能持续提升（图 2）；
   - 对样本顺序敏感性低（表 6），在足够 rollouts 下几乎无关；
   - 超参数设置稳定，无需精细调参。

### 方法的局限性
- **依赖 rollout 数据收集**：MCTS 引入额外计算开销，尤其在低资源环境下可能成为瓶颈。
- **LLM 指令遵循能力要求高**：若 LLM 不能准确执行 sub-instruction，则 memory 效果下降。
- **初期冷启动问题**：早期缺乏记忆时性能接近 baseline，需一定数量的 confident samples 才能起飞。

### 未来工作方向
- 结合 **self-refinement** 方法（如 SELF-REFINE, Reflexion）以提升 sub-instruction 质量。
- 引入 **hierarchical memory**（如 Reasonflux）提高检索与选择效率。
- 探索 **KV-cache 复用** 和 **并行化数据采集** 降低延迟（已在 G.1–G.2 中讨论）。
- 设计机制监控、过滤和修正记忆中的错误或偏见，防止偏差累积。

---

> 📌 **一句话总结**：  
> MILES 提出了一种无需微调、可在测试时动态积累并学会使用模块化推理知识的框架，在多种复杂推理任务上实现了卓越的准确性与效率平衡，为构建可持续进化的 LLM 推理系统提供了新范式。

</details>

---

### 15. [An optimal control approach for neural network architecture adaptation with a posteriori error estimation](https://arxiv.org/abs/2607.07637)

**Authors**: C G Krishnanunni, Thomas Scott, Tan Bui-Thanh  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.07637v1  

#### Abstract
This work presents a novel approach for adapting neural network architecture along the depth based on a posteriori error estimation. By formulating neural network training as a continuous-time optimal control problem, we derive rigorous error estimates that quantify how approximation error distribut...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An optimal control approach for neural network architecture adaptation with a posteriori error estimation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**神经网络架构沿深度方向自适应增长**中的一个核心挑战：如何**系统性地决定在何处插入新层、何时停止增长以及如何初始化新增参数**。现有方法多依赖启发式规则或随机搜索，缺乏理论依据。

### 提出的新方法与新思路
作者提出了一种基于**最优控制理论**（optimal control theory）和**后验误差估计**（a posteriori error estimation）的新型神经网络架构自适应框架，其核心思想如下：

- 将残差网络（ResNet）视为对连续时间动态系统的离散化，并将训练过程建模为一个**连续时间最优控制问题**。
- 引入**有限元方法**（finite element method），将权重 $W(t)$ 和偏置 $b(t)$ 表示为关于深度变量 $t$ 的**分段线性函数**，从而构建更精细的参数表示。
- 利用**对偶加权残差法**（Dual Weighted Residual, DWR）推导出可计算的**功能误差上界**，并将其分解为各层间区间的贡献 $\mathcal{E}_n$。
- 基于该误差分解，设计了**自适应深度增长算法**：在估计误差最大的区间插入新层，并通过插值初始化新层参数。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **理论基础** | 首次将 DWR 方法引入神经网络架构搜索，提供严格的数学误差分析，而非经验性策略。 |
| **决策机制** | 明确回答“在哪里加”、“怎么加”、“何时停”，所有三个关键设计选择均由误差估计驱动。 |
| **效率与效果平衡** | 不需要像 NAS 那样训练大量候选模型；相比宽度增长方法，更适合捕捉深层抽象特征的变化。 |
| **初始化合理性** | 新层由邻近层线性插值得到，在保持功能平滑性的同时打破对称性。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **合成二维回归任务**  
   学习非线性函数：  
   $$
   f(x, y) = e^{-0.1(x+y)} \cdot x \cdot \sin x \cdot \cos y,\quad (x,y)\in[-5,5]^2
   $$  
   - 训练集：1000 个样本  
   - 验证/测试集：各 500 个样本  

2. **Navier-Stokes 方程逆问题**（严重不适定 inverse problem）  
   - 目标：从最终时刻 $t=0.5$ 的稀疏观测点（10个）的涡度（vorticity）测量值，反演初始涡度场 $w_0(x)$。
   - 数据生成：使用谱方法求解 N-S 方程，KL 展开生成初始场，加入 1% 高斯噪声。
   - 输入维度：10（观测值）
   - 输出维度：50（KL 系数）
   - 训练集：700 个样本
   - 验证集：100 个样本
   - 测试集：300 个样本

### 实验设置与评估指标
| 设置项 | 描述 |
|-------|------|
| **优化器** | Adam |
| **激活函数** | 隐藏层使用 `tanh`，输出层为线性 |
| **初始网络** | 3 层隐藏层，每层仅含少量神经元（如 5 或 20） |
| **超参数 K** | 子步长划分参数，用于精确传播状态/伴随变量（取值 4） |
| **终止条件** | 验证损失不再改善或达到最大迭代次数 |

#### 评估指标
- **均方误差**（MSE）：用于合成任务
- **平均相对误差**（Average Relative Error）：
  $$
  \text{Err} = \frac{1}{M}\sum_{i=1}^M \frac{\|\mathbf{w}_{\text{pred}}^{(i)} - \mathbf{w}_{\text{true}}^{(i)}\|_2}{\|\mathbf{w}_{\text{true}}^{(i)}\|_2}
  $$
- **逐点平均相对误差图**（Pointwise Average Relative Error Map）

### 基线方法对比
| 方法 | 简要说明 |
|------|----------|
| **Proposed Approach** | 本文提出的基于 DWR 的误差估计与自适应增长算法（Algorithm 1） |
| **Random layer insertion** | 在误差最小处插入层（作为负面对照） |
| **Net2DeeperNet [17]** | 函数保持变换插入层，位置随机 |
| **Forward Thinking [16]** | 逐层贪婪训练策略 |
| **Baseline network (B)** | 直接训练与最终层数相同的固定结构网络 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 合成二维回归任务（Table 1）
| 方法 | Test Loss (MSE) | Time Taken |
|------|------------------|------------|
| **Proposed approach** | **9.0×10⁻⁶** | 46 min |
| Random layer insertion | 9.41×10⁻⁵ | 5 min |
| Net2DeeperNet [17] | 7.66×10⁻³ | 4.5 min |
| Forward Thinking [16] | 7.1×10⁻⁴ | 2.5 min |
| Baseline network | 3.82×10⁻⁵ | 9 min |

> ✅ **结论**：本文方法取得最低测试误差，显著优于其他方法。

#### Navier-Stokes 逆问题（Table 2）
| 方法 | Test Loss (Avg Rel Err) | Time Taken |
|------|--------------------------|------------|
| **Proposed approach** | **0.161** | 9 min |
| Random layer insertion | 0.170 | 1 min |
| Net2DeeperNet [17] | 0.171 | 2 min |
| Forward Thinking [16] | 0.172 | 30 sec |
| Baseline network | 0.166 | 3 min |

> ✅ **结论**：在更具挑战性的科学计算任务中仍取得最佳泛化性能。

### 与其他方法的对比结果
- 所有基线方法均在添加少数几层后即停止改进，而本文方法能持续提升性能直至收敛。
- 图 7 和图 12 显示，本文方法预测的等高线更接近真实解，空间误差分布最均匀且幅度最小。
- 图 6 和图 10 表明，随着层数增加，验证/测试损失持续下降，说明误差估计有效引导了网络表达能力增强。

### 消融实验与分析（隐含）
虽然未明确列出消融实验表格，但以下分析具有消融意义：
- **K 值研究**（Figure 3 左）：验证了子网格精度对误差估计可靠性的影响，选择 $K=4$ 达到精度与效率平衡。
- **误差估计有效性验证**（Figure 3 右）：显示所提出的上界 $\sum \mathcal{E}_n$ 与真实误差趋势高度一致，证明其可用于指导架构调整。
- **误差分解可视化**（Figure 4 & 9）：展示了每次插入层后对应区间的误差显著降低，验证了“最大误差处插入”的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. **神经网络训练可形式化为连续最优控制问题**，这为引入成熟的数值分析工具（如 DWR）提供了桥梁。
2. **基于 DWR 的后验误差估计能够准确量化各层间区间的近似误差贡献**，并给出可计算的上界。
3. **以最大误差区间为导向的深度自适应策略能高效提升网络表达能力**，尤其适用于小宽度网络场景。
4. **该方法在多个任务上实现了优于主流自适应方法的泛化性能**，特别是在科学机器学习这类复杂非线性映射任务中表现突出。

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **计算成本较高** | 因需计算伴随变量和细粒度状态传播（参数 $K > 1$），前向/反向传播速度较慢。 |
| **依赖局部最优假设** | 误差估计理论要求网络参数处于局部极小值附近，但在实际训练中可能尚未完全收敛。 |
| **目前仅支持深度增长** | 尚未整合宽度自适应机制，无法同时优化宽度与深度。 |
| **实现复杂度高** | 需要实现伴随方程求解与两层级离散化，工程实现门槛高于标准 ResNet。 |

### 未来工作方向
1. **扩展至 CNN、RNN 等架构**，探索卷积核、循环连接的连续建模范式。
2. **联合优化深度与宽度**，发展统一的拓扑自适应框架。
3. **降低计算开销**：探索更高效的伴随求解器或近似估计技术。
4. **应用于更大规模的真实世界任务**，如图像分类、自然语言处理等。
5. **结合 NAS 框架**，将本方法作为微调阶段的精细化结构调整模块。

---

> **总结一句话**：  
> 本文通过将深度神经网络训练重构为连续最优控制问题，首次将**对偶加权残差法**（DWR）引入架构自适应领域，提出了一个**理论严谨、可解释性强、性能优越**的深度自适应框架，在科学机器学习任务中展现出强大潜力。

</details>

---

### 16. [PeTeR: Post-Training Robustification of Probabilistic Circuits](https://arxiv.org/abs/2607.07671)

**Authors**: Adrian Ciotinga, Yeming Dai, YooJung Choi  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.07671v1  

#### Abstract
Probabilistic circuits (PCs) can model complex joint distributions while supporting exact and efficient computation of many inference queries. However, standard likelihood-based PC learning is vulnerable to overfitting and fragile generalization when confronted with data noise, small sample sizes, o...

---

### 17. [When Does In-Context Search Help? A Sampling-Complexity Theory of Reflection-Driven Reasoning](https://arxiv.org/abs/2607.06720)

**Authors**: Yotam Wolf, Noam Wies, Amnon Shashua  
**Category**: cs.AI  
**Published**: 2026-07-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.06720v1  

#### Abstract
Training large language models (LLMs) with extended reasoning has enabled in-context search, in which models iteratively generate, critique, and revise solution attempts. We provide a theoretical analysis of in-context search by modeling it as approximate inference over reasoning traces, where the b...

---

### 18. [Search, Fail, Recover: A Training Framework for Correction-Aware Reasoning](https://arxiv.org/abs/2607.07492)

**Authors**: Dmitry Beresnev, Vladimir Makharev, Roman Khalikov, Ivan Oseledets, Petr Anokhin  
**Category**: cs.AI  
**Published**: 2026-07-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.07492v1  

#### Abstract
Many reasoning tasks are not well described by a single left-to-right chain: a solver may need to pursue a plausible branch, observe delayed failure, and return to the latest prefix that can still be completed. We introduce Pyligent, a training and inference framework inspired by the Diligent Learne...

---

### 19. [Efficient Long-Horizon Learning for Learned Optimization](https://arxiv.org/abs/2607.06772)

**Authors**: Xiaolong Huang, Benjamin Th\'erien, James Harrison, Eugene Belilovsky  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.06772v1  

#### Abstract
Learned optimization aims to improve upon hand-designed optimizers (e.g., Adam and Muon) by meta-learning small neural network optimizers over a distribution of tasks. While recent work has greatly advanced the architectural design and inductive biases of learned optimizers (LOs), current meta-train...

---

### 20. [Generative Diffusion Models of Stochastic Graph Signals](https://arxiv.org/abs/2607.06833)

**Authors**: Yi\u{g}it Berkay Uslu, Samar Hadou, Sergio Rozada, Shirin Saeedi Bidokhti, Alejandro Ribeiro  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.06833v1  

#### Abstract
Sampling stochastic signals supported on a graph underlies many graph machine learning tasks, including recommender systems, forecasting in financial markets, and wireless network optimization. In these settings, the target signals are realizations of unknown conditional distributions. However, prev...

---

### 21. [Mechanistic Interpretability for Neural Networks: Circuits, Sparse Features and Symbolic Reasoning](https://arxiv.org/abs/2607.07316)

**Authors**: Pranav Sawant, Jakub Krej\v{c}\'i  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.07316v1  

#### Abstract
This article offers a comprehensive overview of mechanistic interpretability, an emerging field that seeks to reverse-engineer the internal algorithms of modern neural networks. While traditional explainable AI methods often stop at surface-level input-output correlations, this approach directly add...

---

### 22. [The Key to Going Linear: Analysis-Driven Transformer Linearization](https://arxiv.org/abs/2607.07706)

**Authors**: Anna Kuzina, Paul N. Whatmough, Babak Ehteshami Bejnordi  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.07706v1  

#### Abstract
The quadratic cost of causal self-attention severely bottlenecks long-context transformer inference. While numerous post hoc linearization pipelines exist, it is difficult to identify which components preserve model quality. This work isolates the effect of state update design in a strict frozen-bac...

---

### 23. [LLMs Silently Correct African American English: Auditing and Mitigating Dialect Bias via Activation Steering](https://arxiv.org/abs/2607.06845)

**Authors**: Huan Wu, Ali Emami, Muhammad Furquan Hassan, Osaretin Igbinoba, Osakpolor Idusuyi, Osamede Igbinoba, Faiza Khan Khattak, Laleh Seyyed-Kalantari  
**Category**: cs.CL  
**Published**: 2026-07-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.06845v1  

#### Abstract
African American English (AAE), a rule-governed dialect spoken by over 30 million people, is routinely misinterpreted and "corrected" by large language models (LLMs). Across six instruction-tuned LLMs (14B to 70B), we show that state-of-the-art models systematically prefer Standard American English ...

---

### 24. [From Noisy Traces to Root Causes: Structural Trajectory Analysis and Causal Extraction for Agent Optimization](https://arxiv.org/abs/2607.07702)

**Authors**: Ying Chang, Jiahang Xu, Xuan Feng, Chenyuan Yang, Peng Cheng, Yuqing Yang  
**Category**: cs.CL  
**Published**: 2026-07-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.07702v1  

#### Abstract
The optimization of long-horizon agents increasingly relies on reflection-based mechanisms, where a large language model (LLM) acts as an optimizer to diagnose agent failures and improve agent policies. However, real execution traces are difficult to use directly for optimization: large trace collec...

---

### 25. [UASPL: Uncertainty-Aware Self-Paced Learning with Evidential Neural Networks](https://arxiv.org/abs/2607.06638)

**Authors**: Yifan Zhang, Yuxin Hu, Zhuobin Hao, Xiaozhuan Gao, Lipeng Pan  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.06638v1  

#### Abstract
Self-paced learning (SPL) is an effective learning paradigm that simulates the human learning process by progressing from easy to difficult samples based on the value of the loss function during the learning process. It has shown great potential in improving model performance and training efficiency...

---

### 26. [Converge to Surprise: Evolutionary Self-supervised Image Clustering](https://arxiv.org/abs/2607.06887)

**Authors**: Canlin Zhang, Xiuwen Liu  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.06887v1  

#### Abstract
Most self-supervised image clustering models, actually almost all deep learning approaches, are based on gradient descent: In order to calculate the loss, every optimization step requires a clearly defined target, whether a contrastive split, a masked patch or entity, an EMA-teacher output, a pseudo...

---

### 27. [Robust Federated Learning Under Real-World Client Churn](https://arxiv.org/abs/2607.06979)

**Authors**: Dhruv Garg, Neha Lakhani, Debopam Sanyal, Myungjin Lee, Alexey Tumanov, Ada Gavrilovska  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.06979v1  

#### Abstract
Federated Learning (FL) enables training shared models on private, on-device data, but production deployments remain constrained to slow, multi-day refresh cycles due to the complexity of coordinating massive client populations. For applications such as feed ranking, ad targeting, and personalized r...

---

### 28. [Asymmetric Focal Loss Improves Graph Neural Network Prediction of Drug-Drug Interactions](https://arxiv.org/abs/2607.07611)

**Authors**: Faranak Hatami, Mousa Moradi  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.07611v1  

#### Abstract
Background: Graph neural networks improve computational prediction of polypharmacy side effects, but standard binary cross-entropy training allocates equal capacity to well-classified and difficult examples, potentially missing clinically significant interactions. We evaluated whether an asymmetric ...

---

### 29. [Neural Operator-enabled Topology-informed Evolutionary Strategy for PDE-Constrained Optimization](https://arxiv.org/abs/2607.07682)

**Authors**: Xiangming Huang, Guannan Zhang, Lu Lu, Rapha\"el Pestourie  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.07682v1  

#### Abstract
The inverse design of physical systems governed by partial differential equations is computationally demanding due to the high dimensionality and non-convexity of design spaces. Generative models for inverse design often lack robustness and transferability, whereas evolutionary strategies are robust...

---

### 30. [ECGLight: Compute-Light Framework For Paper ECG Digitization and Myocardial Infarction Screening](https://arxiv.org/abs/2607.07683)

**Authors**: Shreyasvi Natraj, Cyrus Achtari, Felice Gragnano, Andrea Milzi, Marco Valgimigli, Diego Paez-Granados  
**Category**: cs.LG  
**Published**: 2026-07-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.07683v1  

#### Abstract
Electrocardiography (ECG) is one of the most widely used tests for diagnosing cardiovascular disease. Yet several remote clinics still utilize paper ECG printouts for their analysis due to limited connectivity and computational capacity. As a result, vast numbers of physical ECGs obtained in remote ...

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

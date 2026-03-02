# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-02 06:39:50 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Bi-level RL-Heuristic Optimization for Real-world Winter Road Maintenance](https://arxiv.org/abs/2602.24097)

**Authors**: Yue Xie, Zizhen Xu, William Beazley, Fumiya Iida  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.24097v1  

#### Abstract
Winter road maintenance is critical for ensuring public safety and reducing environmental impacts, yet existing methods struggle to manage large-scale routing problems effectively and mostly reply on human decision. This study presents a novel, scalable bi-level optimization framework, validated on ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Bi-level RL-Heuristic Optimization for Real-world Winter Road Maintenance*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该研究针对**大规模冬季道路维护中的车辆路径规划问题**（Winter Road Maintenance Routing Problem），特别是英国国家高速公路网络（如 M25、M6、A1）中盐撒车的调度难题。传统方法依赖人工决策或简单启发式算法，在处理多仓库（multi-depot）、复杂路网拓扑、环境影响和严格操作约束时表现不佳，导致资源分配不均、响应延迟、碳排放高。

### 🚀 提出的新方法与创新点
提出了一种**可扩展的双层优化框架（bi-level optimization framework）**，结合强化学习（Reinforcement Learning, RL）与启发式路径求解器，实现战略级与战术级协同优化：

- **上层（Upper Level）**：使用基于 **Proximal Policy Optimization (PPO)** 的 RL 代理进行**路段到仓库的分配决策**，将整个路网划分为多个逻辑集群（cluster），并从多个 depot 分配资源。
- **下层（Lower Level）**：在每个 cluster 内部采用**带约束检查的最近邻启发式算法**（Constraint-aware Nearest Neighbor Heuristic）解决多目标 VRP，最小化最大行驶时间（makespan）和总碳排放（total carbon emissions）。

该方法首次将 RL 应用于真实世界战略级道路维护的**高层结构划分任务**，而非直接生成完整路径，显著提升了可扩展性和实用性。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 可扩展性 | 难以应对数千条路段的大规模网络 | 成功应用于超 7 万条有向边的真实路网 |
| 决策自动化 | 严重依赖人工经验或静态规则 | 数据驱动 + 迭代优化，减少人为干预 |
| 多目标优化 | 多数仅关注效率或成本 | 同时优化 **makespan** 和 **carbon emissions** |
| 操作可行性 | 忽略实际限制（如转向、车道数、速度） | 显式建模车辆容量、行驶距离、时间窗、单行道等 |
| 集成友好性 | 黑箱模型难部署 | 保留 NN 路由逻辑，易于审计与集成现有 GIS 工具 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **地理范围**：英格兰一个 60km × 60km 区域（British National Grid EPSG:27700），覆盖坐标 X: 444,000–504,000, Y: 224,000–284,000。
- **路网来源**：基于 **OpenStreetMap (OSM)** 构建有向图 $ G=(V,E) $，并通过历史 GPS 轨迹匹配识别需处理路段。
- **关键参数**：
  - 总边数：71,505 条有向边（压缩后）
  - 需处理路段：1515 条（共 543.5 km，折合 1208.7 lane-km）
  - 仓库数量：3 个（Misterton, Pytchley, Rothersthorpe）
  - 车辆类型：统一载盐能力 $ Q_k = 166 $ km·lane，柴油动力，$ e_k = 2.51 $ kgCO₂/liter
- **预处理**：合并 degree-2 节点链以压缩图结构，保持拓扑完整性。

### 🧪 实验设置与评估指标
#### 对比方法
| 方法 | 描述 |
|------|------|
| **KDTree + NN** | 基线方法：使用 KDTree 进行最近仓库分配，再用 NN 启发式生成路径（一次性执行） |
| **KDTree-PPO + NN (10 iterations)** | 所提方法：初始使用 KDTree 暖启动，随后通过 PPO 迭代调整分配策略，每次反馈 lower-level 的 $ Z_1, Z_2 $ |

#### 评估指标
- $ Z_1 $: **Makespan**（最大单辆车完成时间，单位：分钟），目标 ≤ 120 分钟
- $ Z_2 $: **Total Carbon Emissions**（总碳排放量，单位：kg CO₂）
- **NoV**: Number of Vehicles used（使用车辆数）
- 总行驶距离、能耗、约束违反次数、计算时间

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 2）

| Method | $ Z_1 $ (min) | $ Z_2 $ (kg CO₂) | NoV |
|-------|----------------|------------------|-----|
| KDTree + NN | 122.14 | 3,386.63 | 20 |
| KDTree-PPO + NN (10 iter) | **118.81** | **3,220.95** | **28** |

> 注：尽管 NoV 上升至 28，但这是因更细粒度的任务拆分带来的合规性提升（避免超时），并非低效。

### 🔁 迭代过程分析（Figure 3）
- 大部分改进发生在前 4 次迭代；
- Makespan 从 ~120 min 下降至 ~119 min；
- 排放从 ~3700 kg 降至 ~3220 kg；
- 表明 PPO 能快速捕捉最优分配模式，并逐步精细化 depot 边界。

### 🆚 与基线方法对比结果
- **Makespan 减少约 2.7%**（122.14 → 118.81 min），确保所有任务在 **2 小时阈值内完成**；
- **碳排放降低约 4.9%**（3,386.63 → 3,220.95 kg CO₂）；
- **路线更加紧凑、去中心化程度更高**（Figure 4）：
  - 减少跨 depot 的“入侵”行为（cross-depot incursions）
  - 更少的往返 stub 和回溯路径（backtrack/hairpin turns）
  - 提高了各 depot 的服务区域集中度（depot-centric）

这些改进源于 PPO 学会了生成更适合下层 NN 求解器的**高质量初始聚类**，从而减少了无效行驶（deadheading）和协调开销。

### ❌ 消融实验说明
虽然未明确列出消融实验表格，但从设计中可推断以下关键组件作用：
- **KDTree 暖启动**：提供空间连贯且可行的初始解，加速收敛；
- **PPO 反馈机制**：通过 $ r = -(w_1 Z_1 + w_2 Z_2) $ 奖励函数引导策略优化；
- **闭环迭代架构**：允许 assignment 与 routing 动态交互，形成正向反馈。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **双层架构有效解决了大规模冬季维护路径问题**，能够在真实复杂路网中实现高效、环保、合规的调度方案。
2. **PPO 在高层结构决策中优于静态启发式方法**，即使底层仍使用相同 NN 规则，也能通过优化集群结构带来系统性收益。
3. 改进主要体现在：
   - 更短的 makespan（满足 2 小时限要求）
   - 更低的碳排放
   - 更合理的 workload distribution
   - 更清晰的 depot 责任边界
4. 方法具有良好的**可解释性与工程落地潜力**：无需改变现有 dispatch 流程，仅调整 high-level assignment，即可兼容当前 GIS 与调度工具。

### ⚠️ 局限性
- 当前模型假设天气事件为静态场景，未考虑动态 storm progression 或实时交通变化；
- 车队异构性有限（目前只考虑一种车型）；
- 未模拟中途 reload 或 stock management；
- PPO 训练依赖仿真循环，对极端罕见事件泛化能力未知。

### 🔮 未来工作方向
1. 扩展至全国尺度的战略路网优化；
2. 引入 **storm progression modeling** 与 **priority queue** 机制，支持动态优先级调整；
3. 整合 **depot stock/reload logistics**，实现闭环补给规划；
4. 探索 **limited mid-shift replanning** 能力，增强鲁棒性；
5. 将此模块化模板推广至其他 National Highways 区域。

---

## 总结
本研究展示了 **AI-driven bi-level optimization** 在现实交通运输系统中的巨大潜力。通过将 RL 用于高层结构决策、保留经典启发式用于底层执行，实现了**可扩展、可审计、高性能**的冬季道路维护解决方案，为智能交通系统的自动化与绿色化提供了范例。

</details>

---

### 2. [BTTackler: A Diagnosis-based Framework for Efficient Deep Learning Hyperparameter Optimization](https://arxiv.org/abs/2602.23630)

**Authors**: Zhongyi Pei, Zhiyao Cen, Yipeng Huang, Chen Wang, Lin Liu, Philip Yu, Mingsheng Long  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.23630v1  

#### Abstract
Hyperparameter optimization (HPO) is known to be costly in deep learning, especially when leveraging automated approaches. Most of the existing automated HPO methods are accuracy-based, i.e., accuracy metrics are used to guide the trials of different hyperparameter configurations amongst a specific ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：BTTackler: A Diagnosis-based Framework for Efficient Deep Learning Hyperparameter Optimization**

---

## **1. 主要贡献和创新点**

### **解决的问题**
- **传统 HPO 方法效率低下**：现有的自动化超参数优化（HPO）方法多为 **accuracy-based**，即依赖验证集上的准确率等指标来指导搜索过程。
- **坏试验（bad trials）浪费资源**：许多超参数配置会导致严重的训练问题（如梯度爆炸、梯度消失、loss 不下降等），但这些问题在早期阶段难以通过 accuracy 反映出来，导致这些失败的 trial 占用大量计算资源，降低整体优化效率。

### **提出的新方法与新思路**
- 提出 **BTTackler**（Bad Trial Tackler），首个将 **training diagnosis** 引入 HPO 的框架。
- 核心思想是：**在训练过程中实时诊断潜在的训练问题，并对“坏试验”进行早停（early termination）**，从而节省资源用于更有希望的配置探索。
- 定义了 **Quality Indicator**（质量指标）作为量化训练问题的手段，基于文献综述设计了一套适用于 HPO 场景的可编程、低开销的诊断规则。

### **相比现有方法的优势**
| 维度 | 传统方法（如 RS, BO, ETR） | BTTackler |
|------|----------------------------|----------|
| 决策依据 | 准确率/损失值趋势（accuracy-based） | 训练健康状态（diagnosis-based） |
| 早停机制 | 基于性能预测或中位数比较（易误杀好配置） | 基于明确的训练异常检测（更可靠） |
| 效率提升方式 | 更智能地选择配置 | 更快地淘汰无效配置 |
| 自动化程度 | 高 | 更高（无需人工干预即可识别训练故障） |

> ✅ **创新点总结**：
> 1. 首次系统性地将 DNN training diagnosis 应用于 HPO 流程；
> 2. 设计了面向 HPO 的轻量级、并行化的 quality indicators；
> 3. 提出诊断驱动而非精度驱动的 HPO 新范式。

---

## **2. 核心实验方法和设置**

### **使用的数据集与模型架构**
实验覆盖三种典型任务和 DNN 架构：

| 任务 | 数据集 | 模型 | 类型 |
|------|--------|------|------|
| `Cifar10CNN` | CIFAR-10 | CNN（4 Conv + 3 MLP） | 图像分类 |
| `Cifar10LSTM` | CIFAR-10 | LSTM | 图像分类（序列建模） |
| `Ex96Trans` | Exchange-Rate (时间序列) | Transformer | 时间序列预测 |

> 所有任务均设置了较大的超参数搜索空间以增加挑战性。

### **实验设置**
- **硬件环境**：3台服务器，每台含 Intel Xeon 14核CPU、384GB RAM、8块 NVIDIA TITAN X(Pascal) GPU
- **并发数（Concurrency）**：统一设为 8
- **时间预算**：6 小时
- **重复次数**：每个实验重复 3 次取平均值

### **评估指标**
论文提出了两个新的评估维度，超越传统的 accuracy-only 对比：

#### **(1) Top10 Hit Ratio (Top10HR)**  
衡量在相同时间内找到“顶级配置”的能力：
$$
\text{Top10HR} = \frac{\text{方法 i 在总体 Top10 中的数量}}{10} \times 100\%
$$
> 若 >50%，说明该方法优于基线。

#### **(2) Time-Saving for Baseline Accuracy (TSBA)**  
衡量达到基线最佳性能所需的时间节省比例：
$$
\text{TSBA} = \left(1 - \frac{T_i}{T_j}\right) \times 100\%
$$
其中 $T_i$ 是增强方法达到基线最佳 accuracy 所需时间。

### **基线方法对比**
#### **主流 HPO 方法**
- Random Search (RS)
- Gaussian Process (GP)
- Tree-structured Parzen Estimator (TPE)
- SMAC

#### **主流 Early Termination Rules (ETRs)**
- Learning Curve Extrapolation (LCE)
- Median Stop Rule (MSR)

> 所有基线均来自开源框架 NNI，并与其 BTTackler 增强版本进行对比。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 指标 | 平均表现 |
|------|---------|
| **TSBA（时间节省）** | **40.33%** |
| **Top10HR（前10命中率）** | **72.25%**（vs. ETRs 的 52.08%） |
| **额外完成的 top-10 trials 数量** | **+44.5%** |

> 表明 BTTackler 显著提升了 HPO 的效率与成功率。

### **与基线方法的对比结果**

#### ✅ **有效性（RQ1）**
- 在多数情况下，**x-BTTackler 版本显著优于原始方法 x 和 ETR 版本**。
- 示例（Table 2）：
  - **Random-BTTackler** 在 Cifar10CNN 上 Top1 accuracy 达到 **0.8200**，远高于 Random 的 0.7124。
  - **SMAC-BTTackler** 在 Ex96Trans 上 MSE 更低（0.1802 vs. 0.1826），且 Top10HR 达 **94%**。

> ⚠️ 少数例外出现在 Cifar10LSTM 上（因 LSTM 训练本身不稳定），但仍能在部分时间段内反超。

#### ✅ **效率（RQ2）**
- **在相同时间内运行更多 trials**（Table 3）：
  - 例如，在 Cifar10CNN 上，Random-BTTackler 在 6h 内完成 **634 次 trial**，而普通 Random 仅完成 **367 次**（↑72.8%）。
- **TSBA 结果（Table 4）**：
  | 任务 | SMAC-BTTackler 节省时间 |
  |------|------------------------|
  | Cifar10CNN | 16% |
  | Cifar10LSTM | 47% |
  | Ex96Trans | 58% |
  > → 模型越复杂，BTTackler 提升越明显！

#### ✅ **消融实验 / 质量指标分析（RQ3）**
使用模拟器分析各 quality indicator 的触发频率（Table 5）：

| Quality Indicator | 主要作用场景 |
|------------------|-------------|
| **ERG**（Exponentially Reduced Gradients） | Cifar10CNN/LSTM 中最常触发（>300次）→ 检测梯度消失有效 |
| **NMG**（No More Gain） | 所有任务中均有贡献，尤其在 Transformer 中占主导（35次）→ 判断收敛状态良好 |
| **LAR**（Low Activation Ratio） | 在 LSTM 中频繁触发（105次）→ 发现 ReLU 死亡等问题 |
| **AGV/EAG/PLC** | 触发较少但关键，防止极端情况 |

> 🔍 **发现**：不同任务中起主导作用的 indicator 不同，组合使用可实现最大覆盖率。

---

## **4. 关键结论和发现**

### **主要发现**
1. **诊断驱动的 HPO 是可行且高效的**：引入 training diagnosis 可提前识别 bad trials，避免资源浪费。
2. **BTTackler 显著提升 HPO 效率**：
   - 平均减少 **40.33%** 时间开销即可达到基线最佳性能；
   - 在固定时间内可多执行 **44.5%** 的 top-10 trials。
3. **quality indicators 具有通用性和可定制性**：
   - 多个 indicators 协同工作效果最佳；
   - 用户可根据具体任务启用/调整特定 indicator。
4. **overhead 极低**：由于采用并行化设计，额外开销控制在 **<5%**，几乎不影响主训练流程。

### **方法的局限性**
- **empirical thresholds 依赖经验设定**：当前 indicators 使用保守的经验阈值，可能无法适应所有领域。
- **不适用于极短训练任务**：若单次 trial 时间很短，诊断开销可能得不偿失。
- **不能修复问题，只能终止**：BTTackler 不提供修复建议，仅作“筛选器”。

### **未来工作方向**
1. **理论突破**：建立更坚实的 training diagnosis 理论基础（如权重/梯度分布分析）。
2. **自适应 quality indicators**：研究如何自动调参或学习最优诊断策略。
3. **结合搜索空间控制**：利用诊断反馈动态缩小或引导 HPO 搜索空间。
4. **扩展至其他 AutoML 场景**：如 NAS（神经网络架构搜索）、data augmentation tuning 等。

---

> 📦 **开源支持**：作者已发布 [GitHub 开源库](https://github.com/thuml/BTTackler)，支持与主流 HPO 框架（如 NNI）无缝集成，仅需极少代码修改即可启用。

</details>

---

### 3. [GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks](https://arxiv.org/abs/2602.23795)

**Authors**: Wenwu Tang, Dong Wang, Lothar Thiele, Olga Saukh  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.23795v1  

#### Abstract
Structured deep model compression methods are hardware-friendly and substantially reduce memory and inference costs. However, under aggressive compression, the resulting accuracy degradation often necessitates post-compression finetuning, which can be impractical due to missing labeled data or high ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在**结构化深度模型压缩**（如 structured pruning 和 model folding）中，尽管这些方法硬件友好且能显著降低内存和推理开销，但在高比例压缩下会导致严重的**精度下降**（accuracy degradation）。传统解决方案依赖于压缩后的微调（finetuning），但这在实际部署中往往不可行——原因包括：
- 缺少带标签的训练数据；
- 微调成本过高。

现有的一些无需训练的补偿方法（如 data-free folding 或简单的通道选择）通常忽略真实数据分布下的激活模式，导致对下游层输入几何结构的扭曲。

### 提出了什么新方法或新思路
作者提出了一种名为 **GRAIL**（**GRAm-Integrated Linear compensation**）的**后处理块级补偿方法**，其核心思想是：
- 在模型压缩之后，利用一个**小型无标签校准集**（calibration set），通过前向传播收集隐藏层激活值；
- 构建**Gram矩阵**（即二阶统计量 $ G = H^T H $）来捕捉通道间的相关性；
- 使用**岭回归**（ridge regression）学习从压缩后的低维表示 $ h_p \in \mathbb{R}^K $ 到原始高维表示 $ h \in \mathbb{R}^H $ 的线性重建映射 $ B \in \mathbb{R}^{H \times K} $，使得 $ h \approx B h_p $；
- 将该线性映射 $ B $ 合并到下游消费者层（consumer layer）的权重中（即 $ W'_{\text{proj}} = W_{\text{proj}} B $），从而恢复每个模块的输入-输出行为。

这一过程完全**无需反向传播、无需标签、无需额外参数**，是一个**一次性、零微调**（zero-finetuning）的操作。

### 相比现有方法的优势
| 特性 | GRAIL | 现有方法（如 SparseGPT, SlimGPT, FLAP） |
|------|-------|----------------------------------------|
| 是否需要训练/梯度 | ❌ 不需要 | ✅ 多数需要局部优化或微调 |
| 是否通用（agnostic） | ✅ 支持多种压缩器（pruning/folding, Magnitude/Wanda等） | ❌ 通常绑定特定压缩策略 |
| 是否数据感知（data-aware） | ✅ 使用真实激活统计 | ⚠️ 多为 data-free，假设均匀分布 |
| 是否适用于多架构 | ✅ 统一应用于 CNNs、ViTs、LLMs | ❌ 多针对单一架构设计 |
| 是否可插拔 | ✅ 可作为“即插即用”模块附加于任意压缩流程后 | ❌ 通常与压缩耦合 |

此外，当 Gram 矩阵接近单位阵时（表示通道间弱相关），GRAIL 自然退化为经典剪枝/折叠方法，具备良好的理论一致性。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 模型类别 | 数据集 |
|---------|--------|
| **视觉模型** | CIFAR-10, ImageNet-1K |
| **语言模型** | C4, WikiText-2, PTB（用于 perplexity 评估）<br>ARC-C, ARC-E, HellaSwag, PIQA, BoolQ, Winogrande（用于 zero-shot accuracy） |

### 实验设置和评估指标
- **压缩方式**：统一层间压缩比率（layer-wise compression ratio）从 10% 到 90%；
- **压缩方法**：
  - 结构化剪枝：Magnitude pruning (L1/L2), Wanda, Wanda++, SlimGPT, FLAP
  - 模型折叠（Model Folding）
- **校准集大小**：
  - 视觉模型：仅需 **128 张无标签图像**
  - LLMs：**128 条序列**，每条长度 2048 tokens
- **评估指标**：
  - 分类任务：**Top-1 Accuracy (%)**
  - 语言建模：**Perplexity ↓**
  - 零样本能力：**Zero-shot Accuracy (%)**

### 基线方法对比
- **纯压缩方法**：Magnitude pruning, Wanda, SlimGPT, FLAP, model folding
- **增强型压缩方法**：Wanda++（含局部梯度优化）、FLAP（含偏置补偿）
- **其他补偿机制**：REPAIR（BatchNorm 重归一化）
- **微调模型**：作为性能上限参考（fine-tuned for 5 epochs）

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 在 ResNet-18 上的表现（CIFAR-10）
- 在 **75% 稀疏率**下：
  - 基线（无补偿）准确率暴跌至约 **17.6%**
  - 应用 GRAIL 后恢复至 **84.8%**，提升高达 **67.2个百分点**
- 即使在低压缩率（10%-40%）下也能实现近乎无损压缩（within 0.5% of original）

> 图2显示：GRAIL 显著优于 REPAIR，并大幅缩小与微调模型之间的差距。

#### ✅ 在 ViT-B/32 和 CLIP-ViT-B/32 上的表现（ImageNet-1K）
- 对敏感的 Transformer 架构尤其有效；
- 在高稀疏率下防止了灾难性精度崩溃；
- **剪枝 + GRAIL 效果优于 折叠 + GRAIL**，说明通道选择配合重建更优。

> 图3显示：所有压缩比下 GRAIL 均带来增益，尤其在 L1/L2 magnitude pruning 中表现最强。

#### ✅ 在 LLaMA-2-7B 上的语言建模结果（Table 1）
| 方法 | Sparsity | C4 ↓ | PTB ↓ | WikiText-2 ↓ |
|------|----------|------|-------|-------------|
| Wanda | 50% | 155.41 | 423.63 | 272.47 |
| **Wanda + GRAIL** | 50% | **25.97** | **62.86** | **39.59** |
| SlimGPT | 50% | 67.63 | 76.58 | 23.45 |
| **SlimGPT + GRAIL** | 50% | **67.63** | **76.58** | **23.45** |
| FLAP | 60% | 100.53 | 100.01 | 31.80 |
| **FLAP + GRAIL** | 60% | **40.77** | **52.16** | **20.46** |

> **结论**：GRAIL 显著降低 perplexity，尤其对未内置补偿机制的方法（如 Wanda）效果最明显。

#### ✅ 零样本任务表现（Table 2）
| 方法 | Sparsity | Average Zero-Shot Accuracy ↑ |
|------|----------|-------------------------------|
| Wanda | 20% | ~0.64 |
| **Wanda + GRAIL** | 20% | **~0.70** |
| SlimGPT | 50% | 0.52 |
| **SlimGPT + GRAIL** | 50% | **0.58** |

> 表明特征空间重建有助于保留语义表达能力，在跨任务上具有泛化优势。

### 消融实验结果
#### 数据效率分析（Figure 4）
- **准确性提升随校准样本数呈对数增长**；
- 使用 **128 个样本即可达到性能饱和**；
- SlimGPT 需要更多样本稳定，而 Wanda 和 FLAP 更高效。

#### 资源开销（Table 3）
| Model | Calibration Time | Compensation Time | Peak Memory |
|-------|------------------|--------------------|------------|
| ResNet-18 | 0.19s | 0.10s | 162 MB |
| ViT-B/32 | 0.20s | 0.04s | 161 MB |
| CLIP | 0.95s | 0.16s | 300 MB |
| **LLaMA-2-7B** | **58.08s** | **3.16s** | **~3.3 GB** |

> 补偿步骤本身轻量，主要开销来自校准阶段的激活收集，但仍可在单张 A100 GPU 上完成。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **GRAIL 是一种通用、即插即用的后处理补偿框架**，适用于多种结构化压缩方法（pruning/folding）和多种模型架构（CNNs/ViTs/LLMs）。
2. ✅ **仅需极少量无标签数据**（百级图像或万级 token）即可实现显著性能恢复。
3. ✅ **无需任何训练或梯度计算**，完全基于闭式解（closed-form ridge regression）进行线性重建。
4. ✅ **性能显著优于各类 data-free 和部分 data-aware 基线**，在极端压缩下仍能维持可用精度。
5. ✅ **可与已有补偿机制结合使用**（如 REPAIR），进一步提升效果。

### 方法的局限性
1. **依赖完整的前向激活**：必须运行完整模型一次以获取激活值，在边缘设备上可能受限；
2. **内存开销为 $ O(H^2) $**：Gram 矩阵存储需求随隐藏维度平方增长，对超大模型（如 >70B）可能成为瓶颈；
3. **块局部性限制**：仅在单个 producer-consumer 对之间进行补偿，未考虑全局误差传播；
4. **对分布偏移敏感**：若校准集与真实测试分布差异较大，Gram 统计可能失真；
5. **不适用于非结构化压缩**：专为 structured width reduction 设计。

### 未来工作方向
1. **扩展至多层联合补偿**：设计跨层协同重建机制，缓解累积误差；
2. **集成量化与 KV-cache 压缩**：将 GRAIL 思路推广至其他压缩范式；
3. **降低内存占用**：探索近似 Gram 矩阵估计（如随机投影、流式更新）；
4. **任务感知校准**：引入任务相关信号指导校准集采样；
5. **应用于 MoE 架构**：探索专家选择后的特征重建可能性。

---

> 🔗 **代码开源地址**：[https://github.com/TWWinde/GRAIL](https://github.com/TWWinde/GRAIL)

</details>

---

### 4. [RF-Agent: Automated Reward Function Design via Language Agent Tree Search](https://arxiv.org/abs/2602.23876)

**Authors**: Ning Gao, Xiuhui Zhang, Xingyu Jiang, Mukang You, Mohan Zhang, Yue Deng  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.23876v1  

#### Abstract
Designing efficient reward functions for low-level control tasks is a challenging problem. Recent research aims to reduce reliance on expert experience by using Large Language Models (LLMs) with task information to generate dense reward functions. These methods typically rely on training results as ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RF-Agent: Automated Reward Function Design via Language Agent Tree Search

## 1. 论文的主要贡献和创新点

### 解决的问题
在强化学习（Reinforcement Learning, RL）中，**Reward Function 设计**是影响策略性能和训练效率的关键环节，尤其在低层次控制任务（如机器人行走、复杂操作）中至关重要。传统方法依赖专家手动设计稠密奖励函数，耗时且可能次优；而稀疏奖励则难以指导策略学习。

近期研究尝试利用 **Large Language Models (LLMs)** 自动生成奖励函数（如 Eureka、Revolve），但这些方法存在两大瓶颈：
- **搜索效率低下**：采用贪婪（greedy）或进化（evolutionary）算法，难以平衡探索（exploration）与利用（exploitation），易陷入局部最优。
- **历史反馈利用率低**：仅利用最近一次或种群内的局部信息，忽略了跨路径的全局决策线索。

### 提出的新方法：RF-Agent
本文提出 **RF-Agent**，一个将 LLM 视为语言智能体（language agent）、将奖励函数设计建模为**序列化决策过程**的自动化框架。其核心创新在于引入 **Monte Carlo Tree Search (MCTS)** 来管理整个奖励函数的设计与优化流程。

#### 主要创新点：
- **决策视角重构**：将每次 LLM 生成奖励函数视为一次“行动”（action），环境训练反馈作为“奖励”，整个过程被形式化为一个基于树的决策问题。
- **MCTS 驱动搜索**：利用 MCTS 的四阶段（Selection, Expansion, Simulation, Backpropagation）机制，系统性地探索奖励函数空间，通过改进的 **UCT (Upper Confidence Bound for Trees)** 公式平衡探索与利用。
- **多类型启发式动作（Actions）**：在 Expansion 阶段定义五种不同的 action 类型，引导 LLM 生成多样化的奖励函数：
  - `Mutation`（局部修改结构或参数）
  - `Crossover`（融合精英节点的高分组件）
  - `Path Reasoning`（回溯优化路径进行推理）
  - `Different Thought`（生成与现有路径差异大的新思路）
- **自验证（Self-Verify）与思维对齐（Thought Alignment）**：
  - **Self-Verify**：让 LLM 自行评估当前奖励函数接近“专家级策略”的潜力，提供更早、更有价值的选择信号。
  - **Thought Alignment**：修正因 LLM 幻觉导致的“设计思想”与实际代码不一致问题，确保上下文连贯性。

### 相比现有方法的优势
- **更高的搜索效率**：MCTS 能有效避免过早收敛，找到更优的奖励函数。
- **更强的历史信息利用**：不仅利用父节点，还能从全局精英集和祖先路径中提取知识。
- **生成更高质量的奖励函数**：在多个复杂任务上显著超越 Eureka、Revolve 及人类专家水平。

---

## 2. 核心实验方法和设置

### 数据集与环境
实验在两个主流的低层次控制仿真环境中进行，共涵盖 **17 个多样化任务**：
- **IsaacGym**：包含 7 个任务，涉及**运动控制**（如 Ant、Humanoid 行走）和**单臂操作**（如 AllegroHand 旋转物体）。
- **Bi-DexHands**：包含 10 个双手机器人灵巧操作任务，进一步分为两类：
  - **Expert-Easy**：人类奖励函数成功率高（如 GraspAndPlace, BlockStack）。
  - **Expert-Hard**：人类奖励函数表现较差，任务更复杂（如 SwingCup, DoorCloseOutward）。

### 实验设置
- **Policy Training**：所有任务均使用 **PPO** 算法，超参数固定，每个最终奖励函数在 5 个不同随机种子下独立训练并报告平均最大评估分数。
- **LLM 模型**：主实验使用 `GPT-4o-mini` 和 `GPT-4o` 进行公平比较。
- **采样限制**：
  - IsaacGym：最多生成 80 个奖励函数。
  - Bi-DexHands：上限提高至 512，以测试在复杂任务中的搜索能力。

### 评估指标
- **IsaacGym 任务**：根据任务目标设定具体指标，如 Ant 的前进速度、FrankaCabinet 的开门成功次数等。
- **Bi-DexHands 任务**：统一使用 **Success Rate（成功率）** 作为二元评估指标（0 或 1）。
- **归一化得分（Avg norm score）**：用于跨任务比较，计算方式为 `(Method - Sparse) / (Human - Sparse)`。

### 基线方法对比
- **Sparse**：直接使用任务完成信号作为奖励（通常稀疏）。
- **Human**：由任务设计者编写的手工奖励函数，代表专家水平。
- **Eureka**：基于 LLM 的贪婪迭代方法，每轮批量生成并保留最优。
- **Revolve**：基于 LLM 的进化算法，维护种群并进行交叉变异。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### 在 IsaacGym 上的结果（见 Table 1）
- **绝对优势**：RF-Agent 在所有任务上均优于 Eureka 和 Revolve。
- **超越人类专家**：即使使用轻量级模型 `GPT-4o-mini`，RF-Agent 在多数任务上已超过 Human 水平；使用 `GPT-4o` 时，**归一化平均得分高达 2.68**，远超 Eureka (2.00) 和 Revolve (2.03)。
- **稳定性强**：Eureka 在 FrankaCabinet 上出现震荡，Revolve 在 ShadowHand 上无法优化，而 RF-Agent 在所有任务上均稳定提升。

#### 在 Bi-DexHands 上的结果（见 Figure 3 & 4）
- **Expert-Easy 任务**：Eureka 和 Revolve 仅达到人类性能的一半左右，而 **RF-Agent 达到或略微超越人类水平**。
- **Expert-Hard 任务**：所有 LLM 方法表现更好，**RF-Agent 在大多数任务上具有明显领先优势**，例如在 SwingCup 和 DoorCloseOutward 上。
- **训练高效性**：如 Figure 4 所示，RF-Agent 生成的奖励函数能**更快地引导策略收敛到高成功率**，体现出更强的训练效率。

#### 奖励函数优化轨迹（见 Figure 5）
- RF-Agent 在两种难度的任务上都表现出**最高的优化效率**，随着采样次数增加，其最大得分增长最快，证明其强大的持续改进能力。

### 消融实验结果（见 Figure 6 & Table 7/8）

#### 不同搜索方法的影响
- 替换为 DFS、BFS 或 Greedy 后，性能在至少一个任务上显著下降，验证了 **MCTS 在平衡探索与利用上的必要性**。

#### 不同动作类型的作用
- 移除任一 action 类型都会导致性能下降，尤其是移除全部动作后性能严重退化。
- 组合实验（Table 7）表明：**本地操作（Mutation）与全局操作（Crossover, Reasoning）的结合至关重要**，单一类型无法取得最佳效果。

#### 推理机制的有效性
- 移除 **Self-Verify** 或 **Thought Alignment** 均会导致性能下降。
- 特别是在复杂的 AllegroHand 任务上，**移除 Thought Alignment 导致性能下降 35%**，说明其对缓解 LLM 幻觉、保持上下文一致性极为重要。

---

## 4. 关键结论和发现

### 主要发现
1. **将奖励函数设计视为序列决策问题是有效的**：该视角使我们能够利用 LLM 的上下文推理能力，结合历史反馈进行更智能的搜索。
2. **MCTS 显著提升了搜索效率**：相比贪婪或进化方法，MCTS 能更好地探索复杂空间，避免局部最优。
3. **全局历史信息至关重要**：仅依赖局部反馈不足以解决复杂控制任务，跨路径的知识融合（如 Crossover 和 Path Reasoning）是性能突破的关键。
4. **RF-Agent 可生成超越人类专家的奖励函数**：在多个复杂灵巧操作任务上，其自动设计的奖励函数性能优于人工设计。

### 方法的局限性
- **计算成本高**：需要多次完整的 RL 训练循环来获取反馈，时间与资源消耗大。
- **未减少 RL 训练次数**：虽然生成质量更高，但并未降低对策略训练迭代次数的需求。
- **依赖 LLM 的可靠性**：仍受 LLM 幻觉、输出不稳定等问题影响，尽管通过 Thought Alignment 有所缓解。

### 未来工作方向
- **减少 RL 训练循环**：探索如何在不牺牲性能的前提下，降低对完整策略训练的依赖（例如使用代理模型预测奖励函数质量）。
- **扩展到更多任务领域**：验证 RF-Agent 在非控制类任务（如 NLP、规划）中的通用性。
- **优化 MCTS 与 LLM 的协同机制**：进一步设计更高效的提示工程与搜索策略，提升整体框架的鲁棒性与可扩展性。

</details>

---

### 5. [Divide and Conquer: Accelerating Diffusion-Based Large Language Models via Adaptive Parallel Decoding](https://arxiv.org/abs/2602.23792)

**Authors**: Xiangzhong Luo, Yilin An, Zhicheng Yu, Weichen Liu, Xu Yang  
**Category**: cs.CL  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.23792v1  

#### Abstract
Diffusion-based large language models (dLLMs) have shown promising performance across various reasoning tasks, establishing themselves as an alternative to autoregressive large language models (LLMs). Unlike autoregressive LLMs that generate one token per step based on all previous tokens, dLLMs the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Divide and Conquer: Accelerating Diffusion-Based Large Language Models via Adaptive Parallel Decoding*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **diffusion-based large language models (dLLMs)** 虽然在理论上支持每个解码步并行生成多个 token，但在实践中仍普遍采用**逐个 token 生成**（one-token-per-step），以维持生成质量和稳定性。这种做法严重限制了其推理效率，导致理论上的并行潜力无法发挥。

这一现象揭示了一个显著的“**理论并行性与实际性能之间的鸿沟**”（gap between theoretical parallelism and practical performance）。

### 🚀 提出的新方法：DiCo（Divide and Conquer）
作者提出了一种**无需重新训练**（training-free）的自适应并行解码框架 —— **DiCo**，基于“分而治之”（divide-and-conquer）范式，系统性地释放 dLLMs 的内在并行能力。

#### 三大核心阶段：
1. **Divide 阶段（划分）**  
   - 探索输入序列中的 masked tokens，识别出高置信度且空间分离的 **seed tokens**。
   - 利用 Soft-NMS 启发的位置感知函数 $D(i,j)$ 和轨迹引导机制 $W(j,t)$ 来避免种子聚集。
   - 将 seed tokens 双向扩展为局部簇（local clusters），并在满足条件时合并成非重叠的大簇。

2. **Conquer 阶段（征服）**  
   - 在不同 local clusters 上进行**自适应并行解码**。
   - 动态判断每一步应解码多少 token：  
     $$
     S_t = \{i \mid (|S_t| + 1)(1 - c(i)) < 1\}
     $$
   - 实现过程中持续更新 cluster 边界，确保始终追踪高置信区域。

3. **Finalize 阶段（收尾）**  
   - 对剩余少量、依赖性强的 masked tokens 使用**细粒度复合解码策略**（fine-grained compound decoding）。
   - 综合考虑 top-1 confidence 和 logit margin（最高与次高 logit 差值），平衡决策的果断性与稳健性。

### 🔍 相比现有方法的优势
| 方面 | DiCo | 现有方法（如 Fast-dLLM） |
|------|------|--------------------------|
| 是否需再训练 | ❌ 不需要（training-free） | 多数不需要，但部分需额外训练 |
| 并行策略 | ✅ 自适应动态调整 | ⚠️ 固定阈值或块大小（block-wise / semi-AR） |
| 利用上下文结构 | ✅ 显式建模局部簇结构 | ❌ 忽视 token 间依赖演化 |
| 收敛速度 | ✅ 更快完成解码（减少 step 数） | ⚠️ 步骤更多 |
| 性能提升 | ✅ 显著加速 + 更高 accuracy | ⚠️ 加速但 accuracy 下降或持平 |

> 💡 **核心思想突破**：利用 dLLMs 解码早期阶段中 masked tokens 的**局部稀疏依赖特性**，安全地实现跨簇并行；晚期则切换至精细解码，兼顾质量。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **数学推理任务**：
  - **GSM8K**：小学数学应用题，测试逻辑推理能力。
  - **Math-500**：更具挑战性的数学问题集合。
- **代码生成任务**：
  - **HumanEval**：函数级代码生成，zero-shot 设置。
  - **MBPP**：面向编程实践的任务，few-shot 设置。

### ⚙️ 实验设置
- **模型**：
  - **LLaDA-8B-Instruct** (Nie et al., 2025)
  - **Dream-7B-Instruct** (Ye et al., 2025)
- **硬件平台**：NVIDIA RTX 4090 GPU（24GB）
- **生成长度**：256 tokens
- **semi-AR 设置**：block size = 128

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 使用 `lm-eval-harness` 框架评测任务正确率 |
| **Throughput (TPS)** | 每秒生成 token 数（Tokens Per Second），衡量推理效率 |
| **Speedup Ratio** | 相对于 Vanilla 基线的吞吐量倍数提升 |

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Vanilla** | Top-1 confidence-based decoding | 单 token/step，基准线 |
| **Fast-dLLM** (Wu et al., 2025b) | Training-free parallel decoding | 固定置信阈值（0.95），支持并行 |
| **DiCo (Ours)** | Training-free + adaptive + cluster-aware | 分阶段动态控制并行度 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| 模型 | 方法 | Accuracy ↑ | Throughput (TPS) ↑ | Speedup × |
|------|------|-----------|--------------------|------------|
| **LLaDA-8B-Instruct** (non-AR) | Vanilla | 56.33 (GSM8K) | 6.94 | 1.0× |
| | Fast-dLLM | 57.01 | 16.18 | **2.3×** |
| | **DiCo (ours)** | **75.13** (+18.8%) | **23.46** | **3.4×** |
| **Dream-7B-Instruct** (non-AR) | Vanilla | 52.01 (GSM8K) | 8.07 | 1.0× |
| | Fast-dLLM | 50.80 | 17.10 | 2.1× |
| | **DiCo (ours)** | **63.38** (+11.37%) | **31.83** | **3.9×** |

> ✅ 在所有四个数据集上，DiCo 均实现了**更高的 accuracy 和 throughput**。

#### 综合平均表现（Average Across Datasets）：
| 方法 | Avg Accuracy | Avg Throughput | Avg Speedup |
|------|--------------|----------------|-------------|
| Vanilla | 27.23 | 10.58 | 1.0× |
| Fast-dLLM | 27.35 | 32.44 | 3.1× |
| **DiCo (ours)** | **43.10** | **41.56** | **3.9×** |

> 🔥 DiCo 在平均准确率上远超基线（+15.87%），同时保持更高加速比。

---

### 🔍 消融实验结果

#### （1）Seed Token 数量 $N$ 的影响（Figure 5）
- 测试 $N=2$ 到 $10$ 对 HumanEval 表现的影响。
- 结果显示：accuracy 和 throughput **高度稳定**，说明 DiCo 对 $N$ 不敏感，具备良好鲁棒性。

#### （2）是否使用 Trajectory Guidance（Table 2）
| 方法 | Accuracy (GSM8K) | Throughput |
|------|------------------|----------|
| DiCo w/o TG | 49.28 | 34.00 (4.9×) |
| DiCo w/ TG | **75.13** | 23.46 (3.4×) |

> ✅ 轨迹引导虽略微降低吞吐，但大幅提升 accuracy（+25.85%），有效对齐顺序敏感任务的生成路径。

#### （3）是否使用 Fine-grained Compound Decoding（Table 3）
| 方法 | Accuracy | Throughput |
|------|--------|----------|
| DiCo w/o LM | 74.45 | 19.45 (2.8×) |
| DiCo w/ LM | **75.13** | **23.46** (3.4×) |

> ✅ 引入 logit margin 显著提升最终阶段的生成稳定性与质量。

#### （4）解码轨迹可视化（Figure 7）
- **Vanilla**：需 256 步完成。
- **Fast-dLLM**：87 步。
- **DiCo**：仅需 **67 步**即收敛。
> 📉 DiCo 显著减少了总解码步数，验证其高效性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **dLLMs 存在“理论并行 vs 实践串行”的根本矛盾**，根源在于 naive parallel decoding 忽略 token 间的上下文依赖。
2. **dLLMs 解码过程呈现“局部稀疏 → 全局密集”的依赖演化规律**，早期适合并行，晚期需精细处理。
3. **masked tokens 会自然形成“高置信簇”**，这些簇内部强相关、外部弱依赖，是安全并行的理想单元。
4. **DiCo 成功将 divide-and-conquer 范式引入 dLLM 解码**，通过三阶段流程实现了**高质量 + 高效率**的统一。

### ⚠️ 方法的局限性
- 当前方法依赖于 **confidence 和 logit margin 的可靠性**，若模型校准不佳可能影响 cluster 构建。
- 对极端长序列的 scalability 尚未充分验证。
- 当前为 inference-time 方法，未探索与训练端联合优化的空间。

### 🔮 未来工作方向
- 将 DiCo 思想推广至图像、音频等其他模态的 diffusion model。
- 探索与 speculative decoding 或 MoE 架构的结合。
- 设计更智能的 cluster merging/splitting 策略，适应复杂语义结构。
- 进一步压缩 Finalize 阶段开销，实现端到端最优化。

---

## ✅ 总结一句话
> **DiCo 通过“分而治之”的三阶段自适应并行解码框架，在无需重新训练的前提下，成功弥合了 dLLMs 理论并行性与实际性能之间的鸿沟，实现了显著的速度提升与生成质量增强。**

</details>

---

### 6. [Actor-Critic Pretraining for Proximal Policy Optimization](https://arxiv.org/abs/2602.23804)

**Authors**: Andreas Kernbach, Amr Elsheikh, Nicolas Grupp, Ren\'e Nagel, Marco F. Huber  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.23804v1  

#### Abstract
Reinforcement learning (RL) actor-critic algorithms enable autonomous learning but often require a large number of environment interactions, which limits their applicability in robotics. Leveraging expert data can reduce the number of required environment interactions. A common approach is actor pre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Actor-Critic Pretraining for Proximal Policy Optimization

## 1. 论文的主要贡献和创新点

### 解决的问题
- **样本效率低**：标准的 **Reinforcement Learning (RL)** 方法如 **Proximal Policy Optimization (PPO)** 在训练过程中需要大量环境交互，尤其在机器人任务中成本高昂。
- **Critic 初始化被忽视**：尽管 **actor-critic 架构** 中的 **critic** 对策略优化至关重要，但现有预训练方法（如 **Behavioral Cloning, BC**）通常只对 **actor** 进行初始化，而 **critic** 仍随机初始化，导致训练初期不稳定、收敛慢。

### 提出的新方法
本文提出了一种新的 **Actor-Critic Pretraining (ACP)** 方法，用于提升 PPO 的样本效率：
- **Actor 预训练**：通过 **Behavioral Cloning (BC)** 在专家演示数据 $D_{\text{exp}}$ 上预训练 actor 网络，使其初始策略接近专家行为。
- **Critic 预训练**：使用预训练后的 actor 执行 rollouts，生成新的轨迹数据 $D_{\text{rol}}$，并以这些轨迹中的实际回报 $G^{\text{rol}}_t$ 作为目标值，对 critic 网络进行监督学习（最小化 MSE）。

此外，还引入两个增强设计：
- **Extended Step Limit**：为缓解固定步长截断带来的价值估计偏差，延长 rollout 步数以更准确估计长期回报。
- **Residual Model Architecture**：actor 采用骨干网络 + 决策头 + 残差连接结构，在 PPO 微调阶段冻结骨干网络，仅更新决策头，防止灾难性遗忘的同时保留灵活性。

### 相比现有方法的优势
- **相比无预训练 (No Pretraining, NP)**：显著减少达到目标性能所需的环境交互步数。
- **相比仅 actor 预训练 (Actor-only Pretraining, AP)**：进一步提升样本效率，缓解 AP 中常见的灾难性遗忘问题。
- **相比 PIRL 等先进方法**：避免了“先冻 actor 训练 critic”的不稳定性，训练更平滑、收敛更快。

---

## 2. 核心实验方法和设置

### 数据集与环境
- 使用 **15 个模拟机器人任务**，来自 **Gymnasium** 和 **Gymnasium-Robotics** 基准套件，涵盖：
  - **Manipulation 任务**：`FetchReach`, `FetchPush`, `FetchPickAndPlace`, `FetchSlide`, `Pusher`
  - **Locomotion 任务**：`Walker2D`, `Hopper`, `Ant`, `HalfCheetah`, `Swimmer`, `Humanoid`, `HumanoidStandup`
  - **Balance 控制任务**：`Reacher`, `InvertedPendulum`, `InvertedDoublePendulum`

### 实验设置
- **专家策略获取**：从 **RL Baselines3 Zoo** 库中训练得到专家策略，设定其平均回报为 $G_e = 0.65 \times G_{\text{target}}$，确保有提升空间。
- **预训练数据量**：所有环境中 $n_{\text{exp}}$ 固定（见 Table I），rollout 步数 $n_{\text{rol}}$ 作为变量进行分析。
- **PPO 微调预算**：统一设置为 $10^6$ 环境步。
- **评估指标**：
  - 达到目标回报 $G_{\text{target}}$ 所需的总环境步数 $n_{\text{tot}} = n_{\text{exp}} + n_{\text{rol}} + n_{\text{PPO}}$
  - **Sample Reduction (%)**：相对于基线方法的步数节省比例。

### 基线方法对比
| 方法 | 简称 | 描述 |
|------|------|------|
| 无预训练 | NP | 标准 PPO，参数随机初始化 |
| 仅 actor 预训练 | AP | BC 初始化 actor，critic 随机初始化 |
| PIRL | PIRL | 先冻结 actor 训练 critic，再联合优化 |
| 本文方法 | ACP | actor + critic 联合预训练 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table I 综合）
| 指标 | 数值 |
|------|------|
| **相比 NP 的平均样本减少** | **86.1%** |
| **相比 AP 的平均样本减少** | **30.9%** |
| **相比 PIRL 的平均样本减少** | **20.5%** |
| **NP 无法收敛的任务数** | **9 / 15**（60%） |

> 注：当 NP 未达目标时，样本减少定义为 100%

### 典型任务表现（图 2 与表 I 示例）
- **Walker2D**：NP 未能在预算内达标；AP 初期出现严重下降（灾难性遗忘）；ACP 快速稳定上升。
- **Ant**：ACP 仅需约 1.7k rollout + 96k PPO 步即达标，远优于 AP 的 191k 总步数。
- **FetchReach**：ACP 总步数 54.5k vs AP 的 70.5k，提升明显。

### 消融实验结果
#### （1）Rollout 数据的影响（图 3b）
- 在 **12/15** 环境中，加入 rollout 数据可降低总步数。
- 存在**饱和效应**：超过一定 rollout 步数后增益不再增加。
- 不受益于 critic 预训练的 3 个任务（如 `InvertedDoublePendulum`）恰好是那些无需 rollout 即最优的任务。

#### （2）Extended Step Limit 与 Residual Architecture
- 引入 **Extended Step Limit** 平均减少 **10.4%** 环境步。
- 加入 **Residual Architecture** 后，相比无此结构的 ACP，平均减少 **22.1%** 步数。
- 表明这两个设计对提升样本效率具有实证有效性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **同时预训练 actor 和 critic 显著提升样本效率**：ACP 在多数任务上大幅优于仅 actor 预训练或无预训练方法。
2. ✅ **Critic 预训练有助于缓解灾难性遗忘**：由于 critic 初始值更准确，PPO 更新更稳定，避免策略退化。
3. ✅ **Rollout 是必要的**：直接用专家数据训练 critic 效果不佳，必须使用当前 actor 的 rollout 来生成一致的价值标签。
4. ✅ **Extended Step Limit 和 Residual Architecture 有效**：前者改善长期回报估计，后者保护预训练知识。

### 局限性
1. ❗ **依赖专家数据**：若无法获得高质量专家策略或演示，该方法无法应用。
2. ❗ **超参数选择困难**：最佳的 $n_{\text{exp}}$, $n_{\text{rol}}$ 具有环境依赖性，缺乏通用指导原则。
3. ❗ **并非所有环境都受益**：在 3/15 任务中（尤其是高维观测的 Humanoid 家族），ACP 未优于 AP，甚至略有下降（最大 -9.2%）。
4. ❗ **目前聚焦连续动作空间**：虽理论上可扩展至离散空间，但尚未验证。

### 未来工作方向
- 探索 **何时 critic 预训练有效/无效** 的判别条件，建立启发式规则。
- 自动化选择 $n_{\text{exp}}$ 和 $n_{\text{rol}}$ 的机制（例如基于不确定性估计）。
- 将 ACP 扩展至其他 **actor-critic 算法**（如 SAC, TD3）。
- 在 **真实工业场景** 和 **离散动作空间环境** 中进行实证研究。

---

> **总结一句话**：  
> 本论文提出的 **Actor-Critic Pretraining (ACP)** 方法通过联合初始化 actor 与 critic，并辅以 rollout、延长时间窗口和残差架构，显著提升了 PPO 的样本效率，在 15 个机器人任务上平均比无预训练快 **86.1%**，比仅 actor 预训练快 **30.9%**，为现实世界 RL 应用提供了高效可行的预训练范式。

</details>

---

### 7. [Chunk-wise Attention Transducers for Fast and Accurate Streaming Speech-to-Text](https://arxiv.org/abs/2602.24245)

**Authors**: Hainan Xu, Vladimir Bataev, Travis M. Bartley, Jagadeesh Balam  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.24245v1  

#### Abstract
We propose Chunk-wise Attention Transducer (CHAT), a novel extension to RNN-T models that processes audio in fixed-size chunks while employing cross-attention within each chunk. This hybrid approach maintains RNN-T's streaming capability while introducing controlled flexibility for local alignment m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Chunk-wise Attention Transducers for Fast and Accurate Streaming Speech-to-Text

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **RNN-T**（RNN-Transducer）模型虽然广泛用于流式语音处理任务（如 **ASR** 和 **AST**），但由于其严格的帧同步机制和单调对齐特性，在建模复杂任务（尤其是需要灵活对齐的 **speech translation**）时存在建模能力受限的问题。此外，RNN-T 的训练过程依赖于复杂的前向-后向算法，导致计算开销大、内存占用高。

### 提出了什么新方法或新思路
本文提出了一种新型架构 —— **Chunk-wise Attention Transducer (CHAT)**，其核心思想是：
- 将输入音频按固定大小的 **chunk** 划分进行处理；
- 在每个 chunk 内部引入 **cross-attention** 机制，使 joiner 能够在局部范围内灵活地建模声学特征之间的依赖关系；
- 保持 RNN-T 的流式处理能力和实时性约束。

该方法结合了 RNN-T 的低延迟优势与 attention 模型的局部对齐灵活性。

### 相比现有方法的优势
- ✅ **更高的效率**：显著降低训练和推理阶段的计算负担；
- ✅ **更强的建模能力**：通过 chunk 内 attention 改善局部对齐质量，尤其利于翻译等非单调任务；
- ✅ **无需时间戳监督**：不同于某些改进方法，CHAT 不依赖 token 级别的对齐标注；
- ✅ **兼容性强**：可无缝集成到现有的 FastConformer + LSTM Predictor 架构中。

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 语音识别（ASR）
- **English**: Librispeech (`test-clean`, `test-other`)
- **German**: 
  - Common Voice
  - Voxpopuli (**VOX**)
  - Multilingual Librispeech (**MLS**)

#### 语音翻译（AST）
- 英语 → 德语（EN-DE）、中文（EN-ZH）、加泰罗尼亚语（EN-CA）
- 主要基于 **CoVoST** 测试集，训练数据来自多个公开语料库（见 [20]）

### 实验设置
- **模型架构**：
  - Encoder: FastConformerLarge（约 110M 参数，17 层，causal 结构）
  - Predictor: LSTM
  - Chunk size: 12 帧（对应 960ms 音频，因有 8x subsampling）
  - Joiner attention: 4 heads
- **训练配置**：
  - 使用 NeMo 工具包
  - batch=32 进行单轮 mini-epoch 分析
  - checkpoint averaging 基于最多 500k 更新步数的最佳检查点
- **缓存机制**：采用 cache-aware chunk-based streaming encoder [14]，允许当前 chunk 访问前 6 个历史 chunks

### 评估指标
| 任务 | 指标 |
|------|-------|
| ASR | Word Error Rate (**WER%**) |
| AST | BLEU Score |
| 效率 | 推理耗时（秒）、GPU 内存占用、训练速度 |

### 基线方法对比
- **Baseline**: 标准 RNN-T（相同 encoder/predictor 架构）
- 所有比较均控制变量，仅替换 joiner 并调整输入粒度为 chunk

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### 📊 表 1：ASR 性能与推理速度（WER% / 解码时间）

| Model       | English test-clean | English test-other | German VOX | German MLS |
|-------------|--------------------|---------------------|------------|------------|
| RNN-T       | 3.01 / 157         | 11.56 / 140         | 7.23 / 390 | 11.51 / 86 |
| CHAT        | **2.82 / 93**      | **7.45 / 90**       | **7.01 / 238** | **7.01 / 238** |
| rel. WER ↓  | **-6.3%**          | **-2.1%**           | **-3.0%**  | **-3.0%**  |
| rel. speed ↑| **1.69×**          | **1.66×**           | **1.63×**  | **1.64×**  |

> ✅ CHAT 在所有语言和数据集上均优于 RNN-T，且推理速度快 **1.63–1.69×**

#### 📊 表 2：AST BLEU 成绩

| Model   | EN→DE | EN→ZH | EN→CA |
|--------|--------|--------|--------|
| RNN-T  | 29.44  | 34.01  | 18.95  |
| CHAT   | **32.33** | **39.55** | **23.10** |
| rel. ↑ | **+9.8%** | **+16.3%** | **+18.0%** |

> ✅ 在语音翻译任务中提升尤为显著，最高达 **18.0% BLEU 相对增益**

#### 📊 图 1 & 表 4：计算效率表现

| 指标 | RNN-T | CHAT | 提升幅度 |
|------|--------|--------|----------|
| 峰值训练 GPU 内存 | ~40GB | ~21.5GB | **↓46.2%** |
| 单 epoch 训练时间 | 较长 | 更短 | **↑1.36×** |
| 批量推理速度（batch=16） | 84 秒（CoVoST） | **56 秒** | **↑1.5×** |

> ✅ 显著减少内存消耗和训练/推理时间，得益于 joiner 输出维度从 `[B, T, U, V]` 缩减为 `[B, T/C, U, V]`

### 消融实验结果（Ablation Studies）

#### 🔤 不同 chunk size 对性能的影响（EN→DE AST）

| chunk size | 6    | 12   | 24   | 36   |
|-----------|------|------|------|------|
| RNN-T     | 26.63| 29.44| 29.57| 30.60|
| CHAT      | 31.16| 32.33| 33.45| 33.63|

> ✅ CHAT 在各种 chunk 大小下始终优于 RNN-T，表明方法具有良好的鲁棒性和可扩展性

#### ⏱️ 批量推理速度对比（label-looping 优化实现）

| batch size | 2   | 4   | 8   | 16  |
|-----------|-----|-----|-----|-----|
| RNN-T     | 288 | 182 | 115 | 84  |
| CHAT      | 221 | 125 | 77  | **56** |

> ✅ CHAT 在任意 batch size 下都更快，最大提速 **1.5×**

#### ⏳ 延迟测量（Latency Proxy）

| model | clean (ms) | other (ms) |
|-------|------------|------------|
| RNN-T | 6346       | 5712       |
| CHAT  | 6422       | 5779       |

> ✅ 发射时间戳几乎一致（差异 <1%），说明 CHAT **未牺牲实时性**

---

## 4. 关键结论和发现

### 主要发现
1. **效率与精度双赢**：CHAT 在不破坏流式特性的前提下，实现了 **更高准确率** 与 **更低资源消耗** 的统一。
2. **特别适合 AST 任务**：由于 RNN-T 的严格单调对齐限制了翻译性能，而 CHAT 引入的 chunk 内 attention 显著缓解了这一问题，带来高达 **18.0% BLEU 提升**。
3. **训练友好**：joiner 输出序列长度减少 `chunk_size` 倍，大幅降低中间张量存储压力，使更大规模模型训练更可行。
4. **部署实用性强**：在真实 chunked 输入场景下设计，天然适配工业级流式系统。

### 方法的局限性
- ❗ 当前 chunk size 是固定的，可能无法适应不同语速或发音节奏；
- ❗ attention 作用范围局限于 chunk 内部，不能跨 chunk 建模长距离依赖（但仍可通过缓存历史 chunk 缓解）；
- ❗ 对非常短的 utterance 可能引入轻微延迟（chunk boundary 同步发射）。

### 未来工作方向
- 🔮 探索 **adaptive chunk sizing**（动态调整 chunk 长度）以进一步优化延迟与性能平衡；
- 🔮 将 CHAT 扩展至其他 **sequence-to-sequence** 任务，如语音合成、对话系统；
- 🔮 结合 **multi-blank transducer** 或 **TDT** 思路，联合优化 token 和 duration 预测。

---

> ✅ **总结一句话**：  
> **CHAT 是一种高效、准确、实用的流式语音转录架构，在保持 RNN-T 实时性的同时，通过 chunk-wise attention 显著提升了建模能力和运行效率，尤其适用于语音翻译等复杂任务。**

</details>

---

### 8. [The Auton Agentic AI Framework](https://arxiv.org/abs/2602.23720)

**Authors**: Sheng Cao, Zhao Chang, Chang Li, Hannan Li, Liyao Fu, Ji Tang  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.23720v1  

#### Abstract
The field of Artificial Intelligence is undergoing a transition from Generative AI -- probabilistic generation of text and images -- to Agentic AI, in which autonomous systems execute actions within external environments on behalf of users. This transition exposes a fundamental architectural mismatc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《The Auton Agentic AI Framework》总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对当前**Agentic AI**（自主智能体系统）在企业级部署中面临的三大核心挑战提出解决方案：

- **集成悖论（Integration Paradox）**：大型语言模型（LLMs）生成的是**概率性、非结构化输出**，而下游系统（如数据库、API、云服务）要求**确定性、符合Schema的输入**，两者之间存在根本性的架构不匹配。
- **生态系统碎片化（Ecosystem Fragmentation）**：现有框架（如LangChain、AutoGen）将代理定义与运行时逻辑耦合，导致**厂商锁定（vendor lock-in）、不可审计、跨语言移植困难**。
- **状态缺失与效率瓶颈**：LLMs本身是无状态的，缺乏持久记忆机制；同时多步推理流程延迟高，难以满足实时交互需求。

---

### 提出的新方法与新思路

论文提出了 **Auton Agentic AI Framework**，其核心是一个**声明式架构**，通过以下四大支柱实现系统性突破：

#### （1）**AgenticFormat Standard**（声明式代理规范）
- 将代理的“认知蓝图”（Cognitive Blueprint）从运行时引擎中分离。
- 使用**语言无关的声明式格式**（YAML/JSON）定义代理的身份、工具绑定、内存配置、安全约束等。
- 类比于 **Infrastructure-as-Code**（如Kubernetes/Terraform），实现版本控制、可审计、可复用。

> ✅ 创新点：首次将“代理”作为**可移植、可审计的数据工件**而非代码来管理。

#### （2）**Constraint Manifold**（约束流形）—— 安全治理新范式
- 不依赖事后过滤（post-hoc filtering），而是通过**策略投影**（policy projection）在生成过程中强制执行安全策略。
- 在token级别进行logit masking，确保所有输出动作天然落在安全子空间 $ C \subseteq A $ 内。

> ✅ 创新点：从“检测违规”转向“构造即合规”，提升安全性与可靠性。

#### （3）**分层记忆架构 + Reflector-Driven Consolidation Protocol**
- 构建**短期记忆**（事件流）与**长期记忆**（知识库）双层结构。
- 引入类生物启发的**记忆压缩机制**：会话结束后由Reflector Agent自动提炼“洞察”并存入向量存储，支持语义检索。

> ✅ 创新点：解决上下文窗口有限性问题，实现**跨会话经验积累**，避免重复犯错。

#### （4）**三层次自进化框架（Self-Evolution Framework）**
| 层级 | 机制 | 特点 |
|------|------|------|
| Level 1: In-Context Evolution | 失败后生成“Lesson”存入长期记忆，在后续任务中通过检索激活 | 无需训练，快速适应 |
| Level 2: Self-Taught Reasoning (STaR) | 自我蒸馏成功轨迹，进行Supervised Fine-Tuning（SFT） | 将复杂推理内化为直觉 |
| Level 3: Agentic Reinforcement Learning | 使用GRPO/PPO优化多轮POMDP策略 | 探索超越已有数据的新策略 |

> ✅ 创新点：构建**持续学习闭环**，使代理能随时间演进。

#### （5）**运行时效率优化**
- **Cognitive Map-Reduce**：将执行计划解析为DAG，对独立步骤并行执行。
- **Speculative Execution**：预测工具调用结果，提前进行下一步推理，类似CPU分支预测。
- **Dynamic Context Pruning**：基于注意力分数动态清理KV缓存，防止上下文爆炸。

> ✅ 创新点：显著降低端到端延迟，适用于低延迟场景（如广告竞价、事务处理）。

---

### 相比现有方法的优势

| 维度 | 现有方法（如LangChain/AutoGen） | Auton Framework |
|------|-------------------------------|------------------|
| 可移植性 | 锁定Python生态，难迁移到Java/C++环境 | AgenticFormat语言无关，支持`agentic-java`, `agentic-py`等SDK |
| 安全性 | 依赖prompt engineering或后置过滤 | Constraint Manifold实现前向安全保证 |
| 可审计性 | 行为隐含在代码中，难以形式化验证 | 蓝图是结构化数据，支持diff、审查、合规检查 |
| 效率 | 串行执行，延迟累加 | DAG并行 + 推测执行 + 上下文剪枝，关键路径决定总耗时 |
| 持久性 | 会话间无记忆 | 分层记忆 + 自动巩固，支持长期学习 |

---

## 2. 核心实验方法和设置

> ⚠️ 注意：本文是一篇**系统架构论文**，侧重理论建模与设计原则，并未提供传统意义上的“实验结果表格”。文中没有明确列出使用的具体数据集或基准测试集，也未给出与其他框架的量化对比图表。

尽管如此，论文仍描述了若干**评估机制与仿真设定**：

### 实验设置与评估方式

#### （1）**形式化建模与理论分析**
- 将代理建模为增强型**Partial Observable Markov Decision Process (POMDP)**，引入**Latent Reasoning Space (Z)** 区分内部思考与外部行动。
- 形式化定义**Factorized Policy Architecture**：
  - $ z_t \sim \pi_{\text{reason}}(z_t | m_t) $
  - $ a_t \sim \pi_{\text{action}}(a_t | m_t, z_t) $
- 推导出目标函数：
  $$
  J(\theta,\phi) = \mathbb{E}_{\tau \sim (\pi_{\text{reason}}, \pi_{\text{action}})} \left[ \sum_{t=0}^T \gamma^t R(s_t, a_t, z_t) \right]
  $$

#### （2）**运行时性能模拟与估算**
- 对比不同执行模式下的延迟模型：
  - 串行执行：$ L_{\text{total}} = \sum_i (L_{\text{inference},i} + L_{\text{network},i}) $
  - 并行执行（Cognitive Map-Reduce）：$ L_{\text{total}} = \max_{\text{path} \in \text{DAG}} \sum L_{\text{node}} $
- 举例说明推测执行如何隐藏网络延迟（如数据库查询期间预推理）。

#### （3）**自我演化闭环验证机制**
- 引入**Verifier Module** 对最终状态评分（sparse reward）。
- 使用**Process Reward Model (PRM)** 或 Reflector Agent 对中间推理步骤打分（dense reward）。
- 构造复合奖励函数：
  $$
  R(T) = R_{\text{outcome}} + \lambda \sum_t R_{\text{process}}(z_t)
  $$

#### （4）基线方法对比（概念层面）
虽然没有数值对比，但文中多次以以下系统作为对照：
- **LangChain / AutoGen**：代表当前主流但耦合度高的代理框架。
- **硬编码脚本（Hard-coded workflows）**：代表传统自动化方案。
- **纯LLM直接输出**：无结构约束的生成模式。

---

## 3. 主要实验结果和性能指标

> ❗由于本文属于**系统设计与理论框架类论文**，并未报告具体的实验数据（如准确率、响应时间、吞吐量等），也没有消融实验表格。

但可以从文中提取出**定性性能优势与预期收益**：

| 优化项 | 性能提升描述 |
|--------|--------------|
| **Cognitive Map-Reduce** | 对于宽而浅的任务图（如并行查多个股票价格），端到端延迟从“各步之和”降至“最长链长度”，理论上可减少数倍延迟。 |
| **Speculative Execution** | 当工具输出可预测时（如标准API返回格式），推测命中率高，有效隐藏大部分网络等待时间。 |
| **Dynamic Context Pruning** | KV缓存大小保持有界，避免$ O(N^2) $注意力成本随会话增长而失控，支持长周期任务。 |
| **Constraint Manifold** | 安全违规行为在生成阶段即被阻止，无需重试或回滚，提高一次成功率。 |
| **In-Context Evolution** | 同类错误不会重复发生，因失败教训已转化为可检索的“Lesson”。 |

> 💡 文中强调：“这些优化共同作用，使得多步代理工作流的端到端延迟得以控制在可接受范围内，适合实时应用场景。”

---

## 4. 关键结论和发现

### 主要发现

1. **代理应被视为“数据”而非“代码”**  
   通过**AgenticFormat**将代理规格化为声明式配置文件，实现了真正的跨平台可移植性和可审计性。

2. **安全必须前置，不能靠过滤补救**  
   **Constraint Manifold** 提供了一种结构性安全保障，优于脆弱的事后过滤机制。

3. **记忆需要主动压缩而非简单追加**  
   直接拼接历史记录会导致上下文膨胀。**Reflector驱动的记忆巩固协议**模仿人类睡眠中的记忆重组，实现高效长期记忆。

4. **代理可以自我进化**  
   从上下文调整 → 自我蒸馏 → 强化学习，形成一个**渐进式能力跃迁路径**，逐步摆脱对人工干预的依赖。

5. **运行时效率可通过系统级优化大幅提升**  
   即使底层LLM不变，通过**并行化、推测执行、上下文剪枝**等手段也能显著改善用户体验。

---

### 方法的局限性

- **依赖高质量的Schema定义**：若输出合同（output contract）设计不合理，仍可能导致无效输出。
- **Reflector模块自身可能出错**：记忆压缩、Lesson生成等过程仍由LLM完成，存在幻觉风险。
- **初始部署成本较高**：需建立完整的SDK生态、MCP工具连接器体系。
- **未开放完整代码与训练数据**：目前仅宣布开源AgenticFormat标准，实际效果有待社区验证。

---

### 未来工作方向

1. **构建开源生态系统**
   - 推动第三方开发兼容SDK（如Go、Rust版本）。
   - 鼓励共享“Agent Cards”（基于AgenticFormat的代理模板）。

2. **深化与MCP（Model Context Protocol）的集成**
   - 实现更广泛的工具互操作性，打造“即插即用”的工具市场。

3. **发展自动化Reflector能力**
   - 减少对人工标注的依赖，实现完全自动化的经验提炼与策略更新。

4. **探索多代理协作机制**
   - 在单个代理成熟的基础上，研究团队式协同、角色分工、通信协议等问题。

5. **加强形式化验证能力**
   - 结合程序分析技术，对代理行为进行静态推理与边界验证。

---

## 总结

| 维度 | 内容 |
|------|------|
| **论文定位** | 提出一种面向企业级部署的**声明式自主代理系统架构** |
| **核心思想** | **分离关注点**：蓝图 vs 运行时、思考 vs 行动、短期记忆 vs 长期记忆 |
| **关键技术** | AgenticFormat, Constraint Manifold, Hierarchical Memory, Cognitive Map-Reduce, Three-Level Self-Evolution |
| **适用场景** | 数据分析代理、客户支持机器人、自动化运维、金融交易系统等需高可靠、低延迟、可审计的领域 |
| **现实意义** | 为Agentic AI从“玩具”走向“生产系统”提供了工程化路径 |

> 📌 **一句话总结**：  
> Auton Framework 通过**声明式架构 + 结构性安全 + 分层记忆 + 运行时优化 + 自我进化**五大支柱，系统性解决了Agentic AI在企业落地中的可信、可控、可用难题，标志着从“生成式AI”向“自主式AI”的重要演进。

</details>

---

### 9. [Learning Generation Orders for Masked Discrete Diffusion Models via Variational Inference](https://arxiv.org/abs/2602.23968)

**Authors**: David Fox, Sam Bowyer, Song Liu, Laurence Aitchison, Raul Santos-Rodriguez, Mengyue Yang  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.23968v1  

#### Abstract
Masked discrete diffusion models (MDMs) are a promising new approach to generative modelling, offering the ability for parallel token generation and therefore greater efficiency than autoregressive counterparts. However, achieving an optimal balance between parallel generation and sample quality rem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning Generation Orders for Masked Discrete Diffusion Models via Variational Inference*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Masked Discrete Diffusion Models (**MDMs**) 虽然支持并行 token 生成，从而提升推理效率，但在实际应用中面临一个核心挑战：**如何在保持高并行度的同时不损害样本质量**。过度并行可能导致违反 token 间的统计依赖关系，导致生成错误。

现有方法主要依赖**启发式采样策略**（如 Top-k、Top Probability Margin）或通过强化学习训练 unmasking 策略，但这些方法存在以下问题：
- 启发式方法过于刚性，依赖 poorly校准的 logits；
- 学习型方法缺乏从**变分推断**（Variational Inference, VI）角度对生成顺序建模的系统性框架。

### 🚀 提出的新方法与创新
本文提出了一种基于 **Variational Inference** 的新框架，用于**学习 MDMs 中的并行生成顺序**（generation order），其核心创新包括：

1. **概率化建模生成顺序**  
   将 token 的 unmasking 顺序视为隐变量（latent variable），构建一个完整的生成模型与近似后验分布 $ Q $，实现对生成路径的显式建模。

2. **ELBO 目标函数设计与方差降低**  
   推导出带有 **Rao-Blackwellisation** 效果的 ELBO 损失，有效降低了梯度估计的方差，提升了训练稳定性。

3. **高效的可学习 posterior 设计**  
   提出一种参数化的近似后验 $ q_\phi(\mathbf{r}| \mathbf{x}_{t+1}, \mathbf{x}_0) $，满足以下特性：
   - 支持高效采样（计算复杂度不随时间步增长）
   - 允许并行生成
   - 编码生成顺序先验（某些 token 应早于其他生成）
   - 每步至少 unmask 一个 token（避免空操作）

4. **端到端联合训练机制**  
   通过 REINFORCE + **RLOO**（REINFORCE-Leave-One-Out）控制变量进行低方差梯度估计，实现 denoiser 和 unmasking policy 的协同优化。

### 🔍 相比现有方法的优势
| 方面 | 现有方法 | 本文方法 |
|------|--------|---------|
| 建模范式 | 启发式 / 强化学习黑箱优化 | 变分推断，理论清晰 |
| 并行灵活性 | 固定调度或简单规则 | 自适应调整并行程度 |
| 训练稳定性 | 高方差策略梯度 | Rao-Blackwell + RLOO 降方差 |
| 泛化潜力 | 特定任务定制 | 统一概率框架，易于扩展 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **GSM8K**：一个数学应用题问答数据集，用于评估模型在结构化文本生成中的表现。

### ⚙️ 实验设置
- **基础模型**：170M 参数的 MDM，先在 GSM8K 上进行 45,000 步监督微调。
- **后续训练**：在此基础上额外训练 15,000 步，引入本文提出的 unmasking policy 网络 $ p_\theta $ 和 posterior 网络 $ q_\phi $，batch size 为 32，每样本采样 8 次 RLOO。
- **Baseline 对照组**：继续用原 MDM 微调 15,000 步（batch size 256），确保总样本量相近。

### 🎯 评估指标
- **Accuracy (%)**：在 GSM8K 测试集上的最终答案准确率。
- **Average Steps**：平均解码步数（反映并行效率）。
- **Range**：最小到最大解码步数（体现自适应能力）。
- 控制变量：比较时固定平均或最大步数以公平对比。

### 🆚 基线方法
1. **IID**：每个 masked token 以相同概率独立 unmask。
2. **Top Probability**：选择模型预测最自信的 K 个位置 unmask。
3. **Top Probability Margin**：选择最大概率与次大概率差距最大的 K 个位置 unmask，更注重“确定性”。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Budget (T) | Method             | Avg. Steps | Range   | Acc. (%) |
|-----------|--------------------|------------|---------|----------|
| 5         | IID                | 4.0        | [4,4]   | 29.0     |
| 5         | Top Prob           | 4.0        | [4,4]   | 23.7     |
| 5         | Top Prob Marg.     | 4.0        | [4,4]   | 24.0     |
| **5**     | **Ours (Learned)** | **4.01**   | **[2,5]** | **33.1** |
| 10        | Top Prob Marg. @Max| 10.0       | [10,10] | 39.5     |
| **10**    | **Ours (Learned)**  | **9.57**   | **[7,10]**| **37.8** |
| 15        | Top Prob Marg. @Max| 12.0       | [12,12] | 42.3     |
| **15**    | **Ours (Learned)**  | **9.43**   | **[5,12]**| **39.0** |

### 🔍 结果分析
- 在 **极低预算 T=5** 下，本方法仅需约 **4 步**即达到 **33.1% 准确率**，显著优于所有基线（最高为 29.0%），说明其在高度并行下仍能维持高质量生成。
- 即使在更高预算下（T=10,15），虽然绝对准确率略低于最优基线（因使用更少平均步数），但**单位步数效率更高**，且展现出良好的自适应性（步数范围宽）。
- 特别是在 T=10 时，基线需 10 步达 39.5%，而本方法仅用 9.57 步就达到 37.8%，表明其策略更高效。

### ❌ 消融实验（文中未明确提供）
- 文中未报告系统的消融研究（如 ablation on posterior design 或 temperature scaling）。
- 但作者提到尝试了多种 posterior 形式，最终选定 Eq. (14) 的形式，并指出温度系数（temperature scaling）设定在 0.1–0.05 之间效果最佳，推测有助于稳定训练初期的随机性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **变分推断框架适用于学习生成顺序**  
   将 generation order 视为隐变量并通过 VI 进行建模是可行且有效的，能够显式分离 “unmask 哪些位置” 和 “预测什么值” 两个子任务。

2. **所提 posterior 设计兼顾效率与表达力**  
   所设计的 $ q_\phi $ 支持快速采样、并行生成，并能编码合理的生成顺序偏好，同时保证每步至少 unmask 一个 token。

3. **在高并行场景下显著超越启发式方法**  
   在极低解码步数（如 ~4 步）下，本方法大幅领先现有策略，验证了其在防止“过并行化”（over-parallelisation）方面的优势。

4. **自适应解码策略具有实用价值**  
   模型可根据输入动态决定生成节奏，在简单样本上快速完成，在复杂样本上逐步细化，提升整体效率。

### ⚠️ 局限性
- **实验规模有限**：仅在单一数据集（GSM8K）和单一模型大小（170M）上验证，泛化性有待进一步检验。
- **训练成本较高**：由于引入 RLOO 和多次采样，训练效率低于标准 MDM。
- **缺乏深入消融分析**：未充分探讨不同 posterior 结构、温度参数、KL 权重等的影响。
- **未开放代码与超参细节**：限制复现与社区跟进。

### 🔮 未来工作方向
1. **扩展至更多任务与更大模型**  
   如代码生成、生物序列建模、大规模 LLMs 上验证该框架的有效性。
   
2. **探索更强大的 posterior 结构**  
   当前采用 i.i.d. Bernoulli 假设较简单，未来可考虑引入结构性依赖（如 causal masking 或图结构 prior）。

3. **结合 curriculum learning 或 meta-learning**  
   动态调整 unmasking 难度，让模型逐步学会复杂生成模式。

4. **改进梯度估计器**  
   探索更低方差的 policy gradient 方法，如使用 critic-based baselines 或 implicit differentiation。

5. **理论分析生成顺序与任务性能的关系**  
   建立“最优生成顺序”的理论边界，指导模型设计。

---

> **总结一句话**：  
> 本文首次将 **Variational Inference** 系统地应用于 **MDM 的 generation order 学习**，提出了一种高效、可训练的框架，在 **GSM8K 上实现了在极少解码步数下的显著性能提升**，为解决离散扩散模型中“效率 vs 质量”的权衡问题提供了新思路。

</details>

---

### 10. [ODAR: Principled Adaptive Routing for LLM Reasoning via Active Inference](https://arxiv.org/abs/2602.23681)

**Authors**: Siyuan Ma, Bo Gao, Xiaojun Jia, Simeng Qin, Tianlin Li, Ke Ma, Xiaoshuang Jia, Wenqi Ren, Yang Liu  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.23681v1  

#### Abstract
The paradigm of large language model (LLM) reasoning is shifting from parameter scaling to test-time compute scaling, yet many existing approaches still rely on uniform brute-force sampling (for example, fixed best-of-N or self-consistency) that is costly, hard to attribute, and can trigger overthin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ODAR: Principled Adaptive Routing for LLM Reasoning via Active Inference

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前大型语言模型（LLM）的推理范式正从参数扩展转向测试时计算扩展（test-time compute scaling），但主流方法如 **Best-of-N** 或 **Self-Consistency** 依赖于统一、暴力式的采样策略，存在以下问题：
- **成本高昂**：对所有查询都进行大量采样，资源浪费严重。
- **效率低下**：简单任务被过度推理（"overthinking"），复杂任务可能仍不足。
- **融合机制不透明**：多候选答案的投票机制缺乏理论依据，难以归因。

### 提出了什么新方法或新思路
本文提出 **ODAR-Expert**（Open-Domain Adaptive Reasoner），一个基于主动推断（Active Inference）的自适应路由框架，其核心创新包括：

- **自适应路由（Adaptive Routing）**：
  - 引入轻量级 **Difficulty Estimator (DE)**，预测每个查询的难度 `d ∈ [0,1]`。
  - 动态路由到 **Fast Agent**（快速启发式推理）或 **Slow Agent**（深度审慎推理），实现“系统1/系统2”双模式认知架构。

- **基于自由能原理的融合机制（FEP-Based Fusion）**：
  - 将答案选择建模为最小化 **Variational Free Energy** 的过程，公式为：
    ```
    F(y|x) ≈ -Σ log p(y_t | y_<t, x) + λ · Var[-log p(y_t | y_<t, x)]
    ```
    其中第一项是准确率（log-likelihood），第二项是风险惩罚（varentropy，衡量认知不确定性）。
  - 该机制在异构生成器之间提供了一个**原则性的比较标准**，替代了启发式投票。

- **端到端可复现的模块化栈**：
  - 整合了难度感知路由、快慢代理专业化和基于自由能的融合规则，形成一个理论严谨、可复现的推理系统。

### 相比现有方法的优势
- **更高的准确性-效率权衡**：在相同计算预算下显著提升性能，或在达到同等性能时大幅降低成本。
- **理论基础坚实**：将现代LLM编排与主动推断、自由能原理等认知科学和贝叶斯推理理论联系起来。
- **避免过拟合和关键词陷阱**：通过显式结构先验（structural priors）而非黑盒语义嵌入进行难度估计，增强了跨域泛化能力。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
在 **23个基准** 上进行了广泛评估，覆盖 **8个任务类别**：
- **数学**：MATH, GSM8K, **IMO 2025**, MathVista
- **常识**：ARC-Challenge, OpenBookQA, BoolQ, StrategyQA
- **知识问答**：MMLU-Pro, GPQA, HotpotQA
- **多跳推理**：MuSiQue, ScienceQA
- **多模态**：AOK-VQA, MMMU-Pro
- **高级认知**：BBH, BBEH, TruthfulQA, **HLE (Humanity's Last Exam)**
- **编程**：SWE-bench, LIVEBENCH
- **指令遵循**：IFEval
- **抽象推理**：ARC-AGI-2

### 实验设置和评估指标
- **主干模型**：
  - **Fast Agent**: GPT-5.1 (低温度 T=0.2)
  - **Slow Agent**: Claude-4.5 Sonnet (高温度 T=0.3)
- **路由策略**：
  - **Simple Path (d < 0.3)**: 仅调用 Fast Agent (1次调用)
  - **Medium Path (0.3 ≤ d < 0.7)**: Fast生成 + Slow验证 (2次调用)
  - **Hard Path (d ≥ 0.7)**: Fast生成 + Slow进行 Best-of-N (N=5) 采样，共6次调用
- **评估指标**：
  - 主要指标：**平均准确率（Average Accuracy）**
  - 成本指标：**归一化计算成本（Normalized Cost）** 和 **加权平均调用次数**

### 基线方法对比
- **单智能体基线**：GPT-5.1, Claude-4.5
- **多候选策略**：Self-Consistency, Best-of-N (N=5)
- **SOTA方法**：TOPS, Stop Spinning, Simple Ensemble
- **开源复现**：在 **Llama 4 + DeepSeek V3.2** 上实现 Open-ODAR，与 Self-Consistency 对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **总体性能**：ODAR 在 23 个基准上的平均准确率达到 **89.6%**，远超最强基线 Self-Consistency (83.6%)。
- **顶尖任务表现**：
  - **MATH**: **98.2%** (超越 GPT-5.1 的 91.8%)
  - **IMO 2025**: **68.7%** (+20.2% 绝对增益)
  - **HLE**: **54.8%** (+12.0% 绝对增益)
  - **GPQA**: **78.5%** (+10.0% 绝对增益)

### 与基线方法的对比结果
- **性能优势**：在 23 个基准中的 22 个上建立了新的 SOTA 结果。
- **效率优势**：
  - 相比 Self-Consistency，计算成本降低 **1.78倍**（加权平均调用次数 2.55 vs 5.0）。
  - 在开源栈上，Open-ODAR 相比 Self-Consistency 减少 **82%** 的计算成本。
- **SOTA对比**：在 MATH 上超越 TOPS **+2.7%**，同时效率得分（Efficiency Score）高达 **2.8**，是 TOPS (1.2) 的两倍以上。

### 消融实验结果
- **Difficulty Estimator (DE)** 是成本效益的关键驱动因素。移除 DE 会导致成本爆炸（135% 增长）而精度收益微乎其微。
- **Slow Agent 和 FEP 融合** 是处理高风险逻辑任务的基础。移除它们会导致 HLE 上性能下降 **12.0%**。
- **随机路由 vs 学习路由**：在相同计算预算下，ODAR 的智能路由比随机分配高出 **8.3%**，证明“在哪里花费计算”与总预算同等重要。
- **FEP 融合优于启发式**：相比 Max Confidence、Majority Voting 和 Average Log-Prob，FEP 融合在多个数据集上均取得最优结果，平均增益 **+1.2%**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **思考最优扩展（Thinking-Optimal Scaling）** 需要的是**自适应资源分配**，而非简单的增加测试时计算量。ODAR 通过动态路由实现了这一点。
2. **自由能原理（FEP）** 为多智能体融合提供了**原则性且有效的替代方案**，能够通过 varentropy 显式过滤掉高不确定性的“幻觉”推理链。
3. **难度估计器（Difficulty Estimator）** 的设计至关重要。使用显式结构特征（如逻辑连接词、解析树深度）比依赖语义嵌入更鲁棒，能有效避免“关键词陷阱”（keyword trap）。
4. **系统协同效应显著**：路由、快慢代理专业化和 FEP 融合共同作用，形成了一个高效且强大的推理系统。

### 方法的局限性
1. **延迟问题**：Hard Path 的尾部延迟超过 60 秒，限制了实时部署。
2. **受限于基础模型能力**：约 66% 的残余错误源于基础模型的知识缺口或推理缺陷，表明性能瓶颈已从系统架构转移到模型本身。
3. **依赖透明推理访问**：FEP 融合需要访问 token-level log-probabilities，在某些封闭API环境下不可行。

### 未来工作方向
- 探索**轻量级、无需logits的不确定性估计**方法，以解决可移植性问题。
- 设计**更低延迟的硬路径执行策略**，例如更高效的并行化或增量式推理。
- 将框架扩展到**多轮对话和长期规划**场景，实现更复杂的认知控制。

</details>

---

### 11. [ProductResearch: Training E-Commerce Deep Research Agents via Multi-Agent Synthetic Trajectory Distillation](https://arxiv.org/abs/2602.23716)

**Authors**: Jiangyuan Wang, Kejun Xiao, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.23716v1  

#### Abstract
Large Language Model (LLM)-based agents show promise for e-commerce conversational shopping, yet existing implementations lack the interaction depth and contextual breadth required for complex product research. Meanwhile, the Deep Research paradigm, despite advancing information synthesis in web sea...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ProductResearch: Training E-Commerce Deep Research Agents via Multi-Agent Synthetic Trajectory Distillation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **LLM-based shopping agents** 在处理复杂、信息密集型的电商购物决策时存在明显不足：
- **交互深度不足**：传统 ReAct-style 代理仅适用于简单任务（如商品检索），难以支持多轮、多源的深入研究。
- **领域迁移困难**：虽然 **Deep Research 范式** 在开放域信息合成中表现出色（如网页搜索分析），但其工具使用模式（以 web search 为主）无法有效迁移到电商场景。
- **缺乏高质量训练数据**：真实世界中的长周期、高保真度的“产品研究”行为轨迹稀缺，限制了模型学习复杂推理能力。

### **提出的新方法与思路**
作者提出了 **ProductResearch** ——一个基于 **multi-agent synthetic trajectory distillation** 的框架，用于生成高质量、长视野的电商研究轨迹，并用于微调轻量级 LLM 代理。

#### **核心组件**
- **User Agent**  
  基于用户历史行为序列（购买、评论、对话等）推断用户画像（persona）、生成复杂的 **research query** 和定制化的 **evaluation rubric**（RACE 标准）。
  
- **Research Agent**  
  执行 `Plan → Toolcall → Report` 的 ReAct 式循环，利用双环境工具集进行跨源信息整合：
  - **Web Environment**：`web_search`, `web_visit`
  - **E-commerce Environment**：`product_search`, `visit_product`

- **Supervisor Agent**  
  基于三阶段状态机（Check Plan / Check Toolcall / Check Report）提供细粒度监督反馈，防止幻觉、逻辑漂移和证据覆盖不足。

- **Reflective Internalization**  
  将多轮反馈交互（assistant ↔ supervisor）提炼为单一角色的连贯输出，使合成轨迹可直接用于标准的 **supervised fine-tuning**。

### **相比现有方法的优势**
| 维度 | 现有方法 | ProductResearch |
|------|--------|----------------|
| **训练数据来源** | 真实人类轨迹或简单合成 | 多智能体协同生成高保真、长周期轨迹 |
| **评估机制** | 静态评分或二元成功指标 | 动态、查询自适应的 RACE 评价体系 |
| **监督方式** | 无监督或弱监督 | 分阶段、状态感知的强监督 |
| **可扩展性** | 受限于标注成本 | 完全自动化合成，具备良好可扩展性 |

---

## **2. 核心实验方法和设置**

### **数据集构建**
- **原始数据**：从阿里巴巴国际数字商业集团收集的匿名用户交互日志，包括：
  - 购买记录
  - 商品评论
  - 用户与平台/卖家的客服对话
- **构造过程**：
  - 选取 1,000 名代表性用户实例化 User Agent。
  - 自动生成包含 **complex query**, **evaluation rubric**, 和 **agent trajectory** 的合成数据集。
- **划分比例**：训练集 : 验证集 : 测试集 = 8:1:1

### **实验设置**
- **基础模型**：`Qwen3-30B-A3B`（MoE 架构，总参数 30B，每 token 激活 3B）
- **微调配置**：
  - 使用 Megatron-LM 框架
  - 硬件：32×NVIDIA A100 (80GB)
  - 并行策略：CP=4, TP=4, PP=2, EP=1
  - 全局 batch size = 4，训练 3 轮
  - 最大上下文长度变体：32k, 64k, 80k, 128k tokens

### **评估指标**
采用改进版 **RACE metric**（源自 DeepResearch Bench）进行多维评估：

| 指标 | 描述 |
|------|------|
| **RACE Overall** | 综合得分（加权平均） |
| **Comp. (Comprehensiveness)** | 内容覆盖广度 |
| **Depth** | 分析深度与因果推理 |
| **Inst. (Instruction-Following)** | 对用户需求的遵循程度 |
| **Read. (Readability)** | 报告结构清晰度与可用性 |
| **E.Prod (Effective Product Count)** | 成功推荐的有效不同商品数量（衡量产品探索广度） |

> ✅ **参考报告**：由 ProductResearch 合成框架生成的最终报告作为每个查询的 reference report。  
> ✅ **评分机制**：LLM judge 进行 query-adaptive 的 pairwise 比较，计算相对得分（0.5 表示与 reference 持平）。

### **基线方法对比**
分为两类：

#### **(1) Deep Research Agents**
| 模型 | 类型 | 工具情况 |
|------|------|---------|
| Tongyi-DeepResearch | 开源 | 使用相同工具集 $T$ |
| Qwen-DeepResearch | 闭源 | 使用内置原生工具 |
| Gemini-DeepResearch | 闭源 | 使用内置原生工具 |

#### **(2) ReAct Agents**
| 模型 | 类型 | 工具情况 |
|------|------|---------|
| Gemini-3-flash | 前沿 LLM | 接入工具集 $T$ |
| GPT-4.1 | 前沿 LLM | 接入工具集 $T$ |
| Qwen3-max | 前沿 LLM | 接入工具集 $T$ |
| Qwen3-30B-A3B | 本工作的 base model | 接入工具集 $T$ |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**

| Model | RACE Overall | Comp. | Depth | Inst. | Read. | E.Prod |
|-------|--------------|-------|-------|-------|--------|--------|
| Tongyi-DeepResearch | 29.84 | 29.10 | 26.43 | 33.00 | 32.79 | 6.69 |
| Qwen-DeepResearch | 42.76 | 41.70 | 42.87 | 43.45 | 43.15 | 14.4 |
| Gemini-DeepResearch | **45.56** | 45.81 | **47.46** | 45.38 | 42.31 | **25.2** |
| Gemini-3-flash | 32.41 | 30.16 | 29.17 | 38.43 | 33.85 | 6.54 |
| GPT-4.1 | 36.46 | 33.88 | 41.47 | 41.10 | 37.65 | 7.98 |
| Qwen3-max | 36.67 | 35.40 | 33.44 | 41.28 | 38.74 | 6.06 |
| Qwen3-30B-A3B (base) | 31.78 | 29.81 | 28.41 | 36.33 | 35.42 | 3.58 |
| **ProductResearch-SFT-128k** | **45.40** | **45.44** | 43.87 | **46.09** | **47.22** | 12.45 |

> 🔵 **Bold** = 最佳；**Underline** = 第二佳

### **与基线方法的对比结果**
- 相比其 base model (`Qwen3-30B-A3B`)：
  - RACE 总分从 **31.78 → 45.40**（↑ +13.62）
  - E.Prod 从 **3.58 → 12.45**（↑ >3 倍），表明产品探索能力显著增强
- 与最强闭源系统对比：
  - 与 **Gemini-DeepResearch (45.56)** 几乎持平，差距仅 0.16
  - 在 **Instruction-Following** 和 **Readability** 上表现更优（46.09 vs 45.38；47.22 vs 42.31）
- 显著优于所有接入相同工具的 ReAct 代理（最高领先 >8.7 分）

### **消融实验与关键发现**
#### **(1) 上下文长度的影响（Figure 2）**
- 从 32k → 64k：RACE 从 ~37.75 → 44.59（巨大跃升）
- 64k → 128k：持续稳步提升至 45.40
- **结论**：长上下文对建模长周期推理至关重要

#### **(2) 中间报告质量演化（Figure 3）**
- 第一轮平均得分约 0.43，第六轮接近 0.50（与 reference 持平）
- **最大提升发生在第一到第二轮**，说明 Supervisor 的初始反馈最能纠正根本缺陷
- 验证了 **iterative feedback loop** 的有效性

---

## **4. 关键结论和发现**

### **主要发现**
1. **Multi-agent synthetic trajectory 是有效的训练范式**  
   即使是紧凑的 MoE 模型，也能通过高质量合成数据掌握复杂的产品研究能力。

2. **Supervision + Reflection 是关键**  
   - Supervisor Agent 提供的分阶段验证有效抑制了幻觉和逻辑错误。
   - Reflective internalization 成功将多角色交互转化为单角色训练样本，保留纠错信号。

3. **Domain-specific adaptation 至关重要**  
   通用 Deep Research 模型（如 Tongyi-DeepResearch）在电商场景下表现不佳，凸显了领域适配的必要性。

4. **长上下文窗口带来持续收益**  
   更长的 context length 显著提升了模型对复杂推理链的理解与执行能力。

### **局限性**
1. **工具仍有优化空间**  
   当前的 `web_visit` 和 `visit_product` 工具在实现层面和调用策略上尚有改进余地。

2. **仅支持单轮查询**  
   现实中用户意图常随对话演进，当前框架未模拟 multi-turn intent shift。

3. **依赖底层 LLM 和产品目录的质量**  
   若基础模型或 catalog 存在偏见，可能被放大至合成报告中。

### **未来工作方向**
- 扩展 User Agent 以模拟 **multi-turn 对话中的意图演变**
- 优化 tool API 设计与 agent 的 tool-use policy
- 探索将该框架应用于其他垂直领域（如医疗咨询、金融理财）
- 结合 reinforcement learning 进一步提升自主研究能力

--- 

> 📌 **一句话总结**：  
> **ProductResearch 证明了通过 multi-agent 协同合成高保真 long-horizon trajectory，可以有效地将前沿 Deep Research 能力蒸馏到轻量级电商 shopping agent 中，在多个维度上逼近甚至超越闭源系统的表现。**

</details>

---

### 12. [Reasoning-Driven Multimodal LLM for Domain Generalization](https://arxiv.org/abs/2602.23777)

**Authors**: Zhipeng Xu, Zilong Wang, Xinyang Jiang, Dongsheng Li, De Cheng, Nannan Wang  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.23777v1  

#### Abstract
This paper addresses the domain generalization (DG) problem in deep learning. While most DG methods focus on enforcing visual feature invariance, we leverage the reasoning capability of multimodal large language models (MLLMs) and explore the potential of constructing reasoning chains that derives i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Reasoning-Driven Multimodal LLM for Domain Generalization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本文聚焦于 **Domain Generalization (DG)** 中的 **领域偏移（domain shift）** 问题，即模型在训练时仅能访问源域数据，但在测试时需泛化到未见过的目标域。传统 DG 方法大多依赖于学习 **不变特征表示（invariant feature representations）**，例如通过正则化、数据增强或元学习来减少跨域差异。

然而，这些方法通常停留在 **表征层面（representation-level）**，忽略了更高层次的 **推理过程（reasoning process）**，而后者可能更具可解释性和跨域稳定性。

本文提出：  
> **显式建模分类任务中的推理链（reasoning chain）** 可以作为补充信号，提升模型在未知领域的鲁棒性。

### **提出了什么新方法或新思路**
作者提出了 **RD-MLDG（Reasoning-Driven Multimodal LLM for Domain Generalization）**，这是首个将 **class-relevant reasoning chains** 显式整合进 DG 的框架。其核心思想是利用 **Multimodal Large Language Models (MLLMs)** 的强大推理能力，构建结构化的、类别相关的推理路径，并将其用于监督训练。

为解决以下两个关键挑战，RD-MLDG 包含两个核心模块：

- **MTCT (Multi-Task Cross-Training)**  
  同时优化两条路径：  
  - **直接分类路径（direct classification）**：提供稳定、简单的标签监督信号。  
  - **推理增强路径（reasoning-augmented）**：学习生成完整的推理链。  
  该设计使分类目标“引导”推理训练，缓解推理链监督难以优化的问题。

- **SARR (Self-Aligned Reasoning Regularization)**  
  采用 **自标注（self-labeling）** 机制迭代生成推理链：  
  - 初始使用 GPT-4o 生成的高质量推理链进行训练。  
  - 随后让模型自身生成推理链，仅保留最终结论正确的样本作为后续监督信号。  
  此策略缓解了 **外部模型（如 GPT-4o）与目标模型（如 InternVL）之间推理模式不匹配（reasoning-pattern mismatch）** 的问题。

此外，作者构建了 **DomainBed-Reasoning** 数据集，扩展了标准 DG 基准（如 PACS、OfficeHome），为每个样本配对由 GPT-4o 生成的五段式推理链（`<SUMMARY>`, `<CAPTION>`, `<REASONING>`, `<REFLECTION>`, `<CONCLUSION>`），并确保生成时不暴露真实标签，以保证推理基于视觉证据。

### **相比现有方法的优势**
- ✅ **引入过程不变性（process-level invariance）**：不仅追求特征不变，更强调 **推理逻辑的跨域一致性**。
- ✅ **更强的可解释性**：推理链本身是人类可读的决策依据。
- ✅ **更高的泛化潜力**：实验证明，推理链的语义嵌入比视觉特征具有更低的跨域分布差异（MMD 下降 58.6%）。
- ✅ **无需复杂架构修改**：基于 LoRA 的参数高效微调，适用于多种 MLLM。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **主实验数据集**（多源 DG）：
  - **PACS**（4 domains, 7 classes）
  - **VLCS**（4 domains, 5 classes）
  - **OfficeHome**（4 domains, 65 classes）
  - **TerraIncognita**（4 domains, 10 classes）
- 所有数据集均使用作者构建的 **DomainBed-Reasoning** 版本，包含 GPT-4o 生成的推理链。
- **额外实验**还覆盖：
  - **FGVC-Aircraft**（细粒度飞机分类，用于 base-to-new 类别泛化）
  - **VOLDOGER-VQA / VE**（视觉问答与视觉蕴含任务）

### **实验设置和评估指标**
- **评估协议**：标准的 **leave-one-domain-out** 协议，报告所有目标域上的平均准确率（**Avg-acc**）。
- **训练方式**：
  - 使用 **Supervised Fine-Tuning (SFT)**。
  - 采用 **LoRA** 进行参数高效微调（rank=8）。
  - Batch size = 128，learning rate = 5e-4，3 epochs。
  - SARR 设置 $ N=3 $ 轮自标注迭代。
- **基础模型**：
  - 主要使用 **InternVL3-8B** 和 **InternVL3-2B**。
  - 对比模型包括 ResNet-50、ViT-B/16、CLIP 及其变体（如 CoOp、MaPLe）、以及 GPT-4o。

### **基线方法对比**
- **传统 DG 方法**：CORAL, MLDG, MixStyle, SWAD
- **基于 CLIP 的方法**：CoOp, MaPLe, SIMPLE+, DGCLDTP
- **商业 MLLM**：GPT-4o（零样本）
- **消融基线**：
  - `+ CLS only`：仅直接分类
  - `+ Reasoning only`：仅推理链监督
  - `+ MTCT` / `+ SARR`：单独模块

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**
| Method | PACS | VLCS | OfficeHome | TerraInc | **Avg** |
|--------|------|------|------------|----------|--------|
| GPT-4o (zero-shot) | 97.83 | 85.41 | 90.12 | 60.49 | **83.46** |
| DGCLDTP (SOTA CLIP-based) | 97.03 | 84.79 | 87.65 | 63.27 | **83.19** |
| **InternVL3-8B + RD-MLDG (Ours)** | **98.13** | **87.03** | **91.73** | **70.65** | **86.89** |

👉 **RD-MLDG 在所有四个基准上均达到 SOTA 性能，平均准确率领先第二名达 3.7 个百分点**。

### **与基线方法的对比结果**
- 相比最强的 CLIP-based 方法（DGCLDTP），RD-MLDG 在 **TerraInc** 上提升 **+7.38%**，表明其在自然场景变化下优势显著。
- 相比 GPT-4o 零样本表现，RD-MLDG 仍高出 **+3.43%**，说明微调策略有效提升了下游任务性能。
- 在 **PACS** 上接近完美表现（98.13%），远超传统方法（~90%）。

### **消融实验结果（Table 2）**
在 OfficeHome 和 TerraInc 上验证各组件有效性（以 InternVL3-8B 为例）：

| Method | OfficeHome Avg | TerraInc Avg |
|--------|----------------|-------------|
| + Reasoning only | 88.76 | 64.56 |
| + MTCT | 90.58 (**+1.81%**) | 67.19 (**+2.63%**) |
| + SARR | 90.91 (**+2.14%**) | 65.29 (**+0.73%**) |
| **+ MTCT + SARR (Full)** | **91.73** (**+2.97%**) | **70.65** (**+6.09%**) |

✅ **MTCT 和 SARR 均带来持续增益，联合使用效果最佳**。

#### **其他重要发现**
- **SARR 收敛分析**（Fig. 7）：$ N=3 $ 轮后性能趋于稳定，拒绝率从 39.5% 降至 14.8%，说明自生成推理链质量逐步提升。
- **推理链域不变性验证**（Appendix A.1）：  
  推理链文本嵌入的跨域 MMD 平均值为 **0.099**，远低于视觉特征的 **0.239**（↓58.6%），证明其更强的域不变性。
- **跨任务泛化性验证**（Appendix E）：  
  在 **VQA** 和 **VE** 任务中，RD-MLDG 同样有效，InternVL3-8B 分别达到 **77.28%** 和 **72.09%**，超越多数商用 MLLM。

---

## **4. 关键结论和发现**

### **主要发现**
1. 🔍 **推理链是比视觉特征更稳定的跨域信号**：  
   结构化推理过程天然过滤风格、背景等干扰因素，具备更强的语义不变性。
   
2. ⚠️ **直接蒸馏外部推理链存在两大挑战**：
   - **优化困难**：推理链监督需要先拟合中间步骤，收敛慢且不稳定。
   - **推理模式不匹配**：GPT-4o 的丰富描述 vs. InternVL 的简洁预测，导致监督信号“水土不服”。

3. 🛠️ **MTCT + SARR 有效解决上述问题**：
   - MTCT 提供“锚点”，用分类任务稳定训练。
   - SARR 实现“自我对齐”，逐步过渡到模型自身的高效推理风格。

4. 📈 **RD-MLDG 具有广泛适用性**：
   - 不仅适用于分类，还可拓展至 VQA、VE 等多模态任务。
   - 在 base-to-new 类别泛化中也表现出色（FGVC-Aircraft 上 H-mean 从 19.01 → 34.63）。

### **方法的局限性**
- 🔄 **依赖外部 LLM 生成初始推理链**：若 GPT-4o 生成错误或偏见推理，可能影响起点质量。
- 💻 **计算成本较高**：SARR 多轮自标注增加训练时间。
- 🧩 **推理链质量依赖 prompt 设计**：当前格式固定，灵活性有限。

### **未来工作方向**
- 🤖 探索 **完全自举（fully bootstrapped）** 的推理链生成，减少对外部 LLM 的依赖。
- 🔄 将 RD-MLDG 应用于 **continual learning** 或 **open-set recognition** 场景。
- 🧠 研究如何 **动态调整推理深度**，根据不同输入复杂度选择是否“深思熟虑”。
- 🌐 构建更大规模的 **reasoning-enhanced DG benchmark**，推动社区发展。

---

> ✅ **总结一句话**：  
> 本文开创性地将 **MLLM 的推理能力** 引入 **Domain Generalization**，提出 **RD-MLDG** 框架，通过 **MTCT** 和 **SARR** 解决推理监督的优化难题，实现了 **SOTA 性能**，并验证了 **推理链作为过程不变性信号** 的巨大潜力。

</details>

---

### 13. [U-CAN: Utility-Aware Contrastive Attenuation for Efficient Unlearning in Generative Recommendation](https://arxiv.org/abs/2602.23400)

**Authors**: Zezheng Wu, Rui Wang, Xinghe Cheng, Yang Shao, Qing Yang, Jiapu Wang, Jingwei Zhang  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.23400v1  

#### Abstract
Generative Recommendation (GenRec) typically leverages Large Language Models (LLMs) to redefine personalization as an instruction-driven sequence generation task. However, fine-tuning on user logs inadvertently encodes sensitive attributes into model parameters, raising critical privacy concerns. Ex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：U-CAN: Utility-Aware Contrastive Attenuation for Efficient Unlearning in Generative Recommendation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **Generative Recommendation (GenRec)** 系统中，大型语言模型（LLMs）通过微调用户交互日志来提升个性化推荐能力。然而，这一过程会将用户的敏感属性隐式编码到模型参数中，引发严重的**隐私泄露风险**。现有的 **Machine Unlearning (MU)** 方法在处理此类问题时面临“**Polysemy Dilemma**”——即同一个神经元同时承载了敏感信息和通用推理模式，导致传统基于梯度更新或剪枝的方法在遗忘目标数据的同时，严重损害模型的整体推荐性能（**Catastrophic Forgetting** 或 **Structural Damage**）。

### 提出的新方法与创新思路
本文提出 **U-CAN (Utility-aware Contrastive AttenuatioN)**，一种面向 GenRec 场景的高效、精准遗忘框架，其核心创新如下：

- **双筛机制（Synergistic Dual-Screening Mechanism）**：
  - **Contrastive Activation**：通过对比遗忘集 $D_f$ 和保留集 $D_r$ 在各层激活值上的差异，识别对敏感输入响应强烈但对正常输入抑制的“高风险神经元”。
  - **Utility Significance Calibration**：引入效用感知校准模块，结合权重幅值与保留集上的激活范数，为每个维度分配效用得分，防止因过度关注隐私而误删关键推理路径。

- **自适应软衰减（Adaptive Soft Attenuation）**：
  - 不采用破坏性的二值剪枝（hard pruning），而是设计了一个**可微分的衰减函数**，对 LoRA 适配器中的高风险参数进行连续、按维度缩放（down-scale）。
  - 该策略在单次前向传播中完成干预，无需反向传播或再训练，有效**保持推理电路的拓扑连通性**。

### 相比现有方法的优势
| 方法 | 局限性 | U-CAN 的优势 |
|------|--------|-------------|
| **Gradient Ascent (GA)** | 易引起“方向坍塌”（Directional Collapse），扰动扩散至共享表示空间，造成全局性能下降 | 基于局部激活差异定位风险，避免全局梯度扰动 |
| **NPO / Preference-based** | 虽稳定但仍作用于损失层面，难以精确定位参数级记忆 | 参数级干预，实现更精细控制 |
| **LLM-Eraser / Pruning-based** | 二值剪枝破坏网络结构，切断功能通路，导致推理断裂 | 软衰减保留连接，仅削弱敏感路径强度 |

> ✅ **核心优势总结**：U-CAN 实现了 **privacy-forgetting precision** 与 **utility-retention stability** 的更好平衡，且具备**零额外训练开销**（one-shot）、**计算高效**的特点。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ML-100k**：经典的电影推荐数据集，约 10 万条用户-项目交互记录。
- **Pantry**：Amazon Reviews 子集，涵盖杂货与家居用品领域，共 32,992 种商品。

> 数据预处理遵循标准序列推荐流程：按时间排序、5-core 过滤（用户/物品至少有 5 条交互）、去除无标题物品。  
> 隐私遗忘场景模拟：随机选取 25% 用户交互作为 $D_f$（遗忘集），其余 75% 作为 $D_r$（保留集）。

### 实验设置
- **基础模型**：**LlamaRec**（基于 Llama-2-7b 的两阶段生成式推荐系统）
- **适配方式**：使用 **LoRA** 进行参数高效微调，所有遗忘操作仅修改 LoRA 适配器参数，冻结主干网络 $\theta_0$
- **部署范式**：支持 4-bit NF4 量化，符合实际边缘部署需求

### 评估指标
#### （1）遗忘有效性（Unlearning Effectiveness）↓
- **KL Divergence ↑**：衡量遗忘集上输出分布偏离原始模型的程度
- **Prediction Shift (%) ↑**：遗忘集中预测结果发生变化的比例
- **Perplexity (PPL) ↑**：反映模型对遗忘序列的信心降低程度（越高越好）

#### （2）效用保留（Utility Preservation）↑
- **Recall@K**, **MRR@K**, **NDCG@K**（K=5,10）：标准推荐质量指标，在保留集 $D_r$ 上评估
- **Trade-off@10**：综合指标，定义为：
  $$
  \text{Trade-off@10} = \Delta\%\text{Forget@10} - \Delta\%\text{Retain@10}
  $$
  数值越大表示在强遗忘的同时较好地保留了效用。

#### （3）运行效率
- **Execution Time**：端到端执行时间
- **Throughput (samples/sec)**：每秒处理样本数，体现可扩展性

### 基线方法对比
| 方法 | 类型 | 是否需再训练 |
|------|------|--------------|
| **Retraining** | 黄金标准 | 是（全量重训） |
| **GA** | 梯度上升 | 否（优化损失） |
| **NPO** | 负偏好优化 | 否（偏好学习） |
| **LLM-Eraser** | 选择性剪枝 + 蒸馏 | 否（mask-based） |
| **U-CAN (Ours)** | 激活对比 + 软衰减 | ❌（one-shot 前向） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 2）

#### 在 ML-100k 上的表现
| 方法 | Trade-off@10 | R@10 (Forget↓) | R@10 (Retain↑) | KL Divergence ↑ | PPL ↑ |
|------|---------------|------------------|------------------|------------------|--------|
| Retraining | 13.94 | 0.1999 | 0.1131 | — | — |
| GA | -3.82 | 0.2303 | 0.1030 | 0.00 | 18.91 |
| NPO | -4.30 | 0.2248 | 0.1030 | 0.00 | 18.90 |
| LLM-Eraser | 17.49 | 0.1714 | 0.1032 | 0.11 | 21.86 |
| **U-CAN (Ours)** | **29.45** | **0.1435** | **0.1098** | **0.13** | **23.83** |

> 🔹 U-CAN 在 **Trade-off@10** 上显著优于所有基线（接近 Retraining 的 2.1 倍），且在遗忘效果（低 Recall@10）和效用保留之间取得最佳平衡。

#### 在 Pantry 上的表现
| 方法 | Trade-off@10 | R@10 (Forget↓) | R@10 (Retain↑) | KL Divergence ↑ | PPL ↑ |
|------|---------------|------------------|------------------|------------------|--------|
| Retraining | 1.55 | 0.0380 | 0.0318 | — | — |
| GA | -23.74 | 0.0416 | 0.0416 | 0.00 | 17.74 |
| NPO | -23.85 | 0.0416 | 0.0416 | 0.00 | 17.77 |
| LLM-Eraser | -7.13 | 0.0347 | 0.0406 | 0.14 | 18.71 |
| **U-CAN (Ours)** | **14.64** | **0.0356** | **0.0469** | **0.41** | **69.67** |

> 🔹 特别是在 **Pantry** 上，U-CAN 引发了极大的 **PPL 上升（69.67）**，远超其他方法，表明其成功瓦解了对遗忘内容的记忆信心。

### 与基线方法的对比结论
- **GA/NPO**：几乎未改变遗忘集分布（KL≈0, PPL 不变），说明仅靠损失层面优化无法实现真正遗忘。
- **LLM-Eraser**：虽有一定遗忘效果，但在某些场景下效用损失较大（如 Pantry 上 Trade-off 为负）。
- **U-CAN**：
  - 实现最强的遗忘信号（最高 KL、最大 Prediction Shift、最高峰值 PPL）
  - 效用保留优于或接近 Retraining
  - Trade-off 综合得分**全面领先**

### 消融实验结果（Ablation Study, Table 3）
移除 U-CAN 的任一组件均导致性能下降：

| 变体 | KL Divergence (Pantry) | PPL (Pantry) | R@10 (Forget) | R@10 (Retain) |
|------|--------------------------|--------------|----------------|----------------|
| w/o C（无对比筛选） | 0.17 | 18.02 | 0.1502 | 0.0770 |
| w/o F（无效用校准） | 0.16 | 19.97 | 0.2070 | 0.1197 |
| w/o H（硬剪枝替代软衰减） | 0.19 | 23.73 | 0.1536 | 0.0820 |
| **U-CAN (Full)** | **0.42** | **69.67** | **0.1435** | **0.1098** |

> 🔹 所有组件均为必要：
> - **Contrastive Screening (C)** 对精准定位至关重要；
> - **Utility-aware Filtering (F)** 显著提升效用保留；
> - **Soft Attenuation (H)** 是实现高不确定性（高 PPL）的关键。

---

## 4. 关键结论和发现

### 主要发现
1. **GenRec 中存在严重的 Polysemy Dilemma**：敏感信息并非孤立存储，而是与通用语法、叙事逻辑等高度纠缠于同一组神经元中，使得传统遗忘方法极易伤及无辜。
2. **基于激活差异的风险定位优于纯梯度或剪枝方法**：U-CAN 利用 $D_f$ 与 $D_r$ 的激活差距，能更精确地识别出“仅对敏感输入敏感”的神经元。
3. **效用感知校准是维持性能的关键**：单纯追求高遗忘率会导致效用崩溃；融合权重大小与保留集激活强度可有效保护重要推理路径。
4. **软衰减优于硬剪枝**：连续缩放而非置零，能在不破坏网络结构的前提下实现深度遗忘，尤其体现在 PPL 的剧烈上升上。
5. **One-shot 高效可行**：U-CAN 无需任何反向传播或再训练，即可达到接近 Retraining 的遗忘-效用平衡，适合高频删除请求的实际部署。

### 方法的局限性
1. **缺乏形式化隐私保证**：当前评估依赖经验指标（KL、PPL、Prediction Shift），尚未提供如 **differential privacy** 等理论保障。
2. **适用范围受限于 LoRA 架构**：目前仅验证于 LoRA 适配器，是否适用于其他 PEFT 方法（如 Adapter、BitFit）尚待研究。
3. **未测试更强攻击者场景**：实验假设非自适应对手，面对专门设计的记忆提取攻击（memory extraction attack）的鲁棒性未知。
4. **跨域泛化能力未充分验证**：仅在两个推荐数据集上测试，是否适用于对话系统、文本生成等更复杂任务仍需探索。

### 未来工作方向
- 将 U-CAN 扩展至多模态 LLMs（如图文推荐系统）
- 探索与 **Knowledge Editing** 技术的结合，实现动态知识修正
- 设计具备 **formal unlearning guarantees** 的变体
- 应用于联邦学习环境下的分布式遗忘（Federated Unlearning）
- 研究如何自动调节风险阈值 $t_{risk}$ 与融合系数 $\lambda$，提升自动化程度

--- 

> ✅ **总体评价**：U-CAN 是一项针对 GenRec 场景下隐私遗忘难题的重要进展，提出了一个**精准、高效、结构友好的遗忘范式**，在多个维度上超越现有 SOTA 方法，具有较强的实用价值和发展潜力。

</details>

---

### 14. [Normalisation and Initialisation Strategies for Graph Neural Networks in Blockchain Anomaly Detection](https://arxiv.org/abs/2602.23599)

**Authors**: Dang Sy Duy, Nguyen Duy Chien, Kapil Dev, Jeff Nijsse  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.23599v1  

#### Abstract
Graph neural networks (GNNs) offer a principled approach to financial fraud detection by jointly learning from node features and transaction graph topology. However, their effectiveness on real-world anti-money laundering (AML) benchmarks depends critically on training practices such as specifically...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Normalisation and Initialisation Strategies for Graph Neural Networks in Blockchain Anomaly Detection*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**图神经网络（GNN）在区块链反洗钱（AML）任务中的训练稳定性与性能优化问题**，特别是针对现实世界中高度类别不平衡、结构复杂（如异质性、度分布偏斜、时间漂移）的交易图数据。尽管GNN被广泛用于金融欺诈检测，但其训练过程中的 **weight initialisation（权重初始化）** 和 **normalisation（归一化）策略** 往往被忽视，缺乏系统研究。

### 🚀 提出的新方法与新思路
- **系统性消融分析（Systematic Ablation Study）**：首次在 Elliptic 比特币数据集上对三种主流 GNN 架构（GCN、GAT、GraphSAGE）进行了关于 **Xavier 初始化** 和 **GraphNorm 归一化** 的全面组合实验。
- **架构依赖性洞察（Architecture-Dependency Insight）**：提出并验证了一个核心观点——**最优的初始化与归一化策略是架构相关的，而非通用方案**。
- **可复现框架开源**：发布了一个完整的、基于时间划分（temporal split）、固定随机种子（seeded runs）和自动化超参调优（Optuna）的实验框架，提升研究可比性和部署可靠性。

### 🔍 相比现有方法的优势
- 超越以往仅关注模型架构设计或采样效率的研究，深入挖掘了**训练工程细节对最终性能的影响**。
- 强调使用 **AUPRC（Area Under Precision-Recall Curve）作为主评估指标**，更适用于严重类别不平衡场景（仅有2%为非法节点），避免 AUC-ROC 或准确率带来的误导。
- 实验设置更加贴近真实部署环境（如时间顺序划分、全图训练），增强了结果的实际指导意义。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Elliptic Bitcoin Dataset**  
  - 包含 203,769 个节点（交易）和 234,355 条有向边（支付流）
  - 每个节点具有 166 维特征（原始特征 + 一跳聚合统计量）
  - 时间跨度划分为 49 个时间步（每步约两周）
  - 标签分布极不平衡：2% 非法（illicit），21% 合法（licit），77% 未知

### ⚙️ 实验设置
- **图构建**：将有向交易图转换为无向图以支持对称消息传递，并添加自环。
- **时间划分（Temporal Splitting）**：
  - 训练集：前 29 个 time steps
  - 验证集：中间 10 个 time steps
  - 测试集：最后 10 个 time steps  
  → 防止信息泄露，模拟真实预测场景
- **模型架构**：评估以下三种 GNN：
  - **GCN**（Graph Convolutional Network）
  - **GAT**（Graph Attention Network）
  - **GraphSAGE**（SAmple and AGGregatE）
- **正则化与初始化配置对比**：
  - 初始设置（Baseline）：各模型默认 `reset_parameters()`
  - Xavier 初始化：统一使用 Xavier uniform 初始化
  - GraphNorm：应用于每一层 GNN 输出后
  - 组合策略：`Xavier + GraphNorm`

### 📊 评估指标
- **主指标**：
  - **AUPRC**（Precision-Recall 曲线下面积）→ 更适合类别极度不平衡的任务
- **辅助指标**：
  - **AUC-ROC**
  - **F1-score @ high-confidence thresholds**（90th, 99th, 99.9th 百分位阈值）
- **鲁棒性评估**：
  - 对测试集进行 **100 次重复子采样（50% 节点）**，报告均值与标准差
- **超参数优化**：
  - 使用 **Optuna** 进行 100 轮搜索，目标为验证集 AUPRC
  - 参数包括学习率、隐藏维度、dropout rate、层数等

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2）

| Model | Configuration | AUC | **AUPRC** |
|-------|---------------|------|-----------|
| GCN | Baseline | 0.8728 | **0.5993** |
| GCN | Xavier | 0.8740 | 0.5939 |
| GCN | Xavier + GraphNorm | 0.8736 | 0.5442 |
| GAT | Baseline | 0.8585 | 0.6022 |
| GAT | Xavier | 0.8486 | 0.6190 |
| GAT | **Xavier + GraphNorm** | **0.8700** | **0.6568** |
| GraphSAGE | Baseline | 0.8593 | 0.6551 |
| GraphSAGE | **Xavier** | **0.8826** | **0.6678** |
| GraphSAGE | Xavier + GraphNorm | 0.8755 | 0.6651 |

> 💡 最佳 AUPRC：**GraphSAGE + Xavier** 达到 **0.6678**

### 🔁 与基线方法的对比结果
- **GAT**：结合 GraphNorm 与 Xavier 后，AUPRC 提升 **+0.055**，AUC 提升 +0.012，是所有模型中增益最大的。
- **GraphSAGE**：仅用 Xavier 初始化即可带来显著提升（AUPRC +0.013），加入 GraphNorm 反而轻微下降。
- **GCN**：对两种改进均不敏感，甚至 **GraphNorm 导致 AUPRC 明显下降至 0.5442**，说明其可能破坏原有稳定机制。

### 🔍 消融实验结果
- **初始化影响**：
  - Xavier 显著加速 GraphSAGE 收敛且提升性能；
  - 对 GCN 几乎无影响；
  - 在 GAT 中单独使用 Xavier 会降低 AUC。
- **归一化影响**：
  - GraphNorm 对 GAT 有强正向作用，有助于稳定 attention 权重；
  - 对 GraphSAGE 和 GCN 不仅无效，还可能导致过平滑或干扰聚合机制。
- **组合效应非线性**：并非“越多越好”，需考虑架构内在机制。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **初始化与归一化的效果具有强烈的架构依赖性（architecture-dependent）**：
   - **GraphSAGE**：最佳策略是 **Xavier 初始化 alone**，无需额外 GraphNorm。
   - **GAT**：最大收益来自 **GraphNorm + Xavier** 的组合，有效缓解 attention 在高阶邻域下的不稳定问题。
   - **GCN**：默认初始化已足够稳健，引入 GraphNorm 反而有害。

2. **Graph-level normalisation 并非万能解药**：
   - 尽管 GraphNorm 被设计用于缓解 over-smoothing 和方差漂移，但在某些架构（如 GCN、GraphSAGE）中可能干扰正常的特征传播路径。

3. **合理的 weight initialisation 是最一致有效的干预手段**：
   - 特别是对需要归纳学习能力的 GraphSAGE，Xavier 初始化显著提升了训练稳定性和最终性能。

4. **AUPRC 是更可靠的评估标准**：
   - 在极端类别不平衡下，F1-score 和 AUC-ROC 容易产生误导，应优先采用 AUPRC。

### ⚠️ 方法的局限性
- **模型覆盖有限**：仅评估了 GCN、GAT、GraphSAGE，未涵盖现代架构如 Graph Transformers、Temporal GNNs（如 TGN）、Higher-order GNNs。
- **数据简化假设**：Elliptic 数据集虽具代表性，但仍为静态快照，缺少动态节点属性更新机制，难以完全反映真实 AML 系统复杂性。
- **硬件限制**：超参搜索仅限 100 次 trial，空间受限；未评估推理延迟、内存占用等实际部署指标。
- **未探索其他 normalisation 方法**：如 PairNorm、DGN 等图专用归一化未纳入比较。

### 🔮 未来工作方向
1. **标准化评估协议**：
   - 推动社区采用统一的时间划分、评估指标（AUPRC为主）、阈值选择策略，增强不同研究间的可比性。

2. **扩展至更先进模型**：
   - 将本研究的训练策略迁移到 **GATv2、ChebNet、Temporal GNNs** 上，检验其泛化能力。

3. **引入 adaptive sampling 技术**：
   - 如 GLASS、SALIENT++ 等方法可在保持性能的同时提高训练效率，尤其适合大规模图。

4. **迁移至 Elliptic2 数据集**：
   - 新发布的 Elliptic2 提供子图级别标注，更适合研究洗钱团伙模式识别，有望揭示新的训练动态规律。

5. **开发轻量级 Python 兼容系统**：
   - 当前部分高效工具（如 SALIENT）依赖 C++，不利于集成。未来可构建纯 Python 的高性能 GNN 训练流水线。

---

> 🔗 开源代码地址：[https://github.com/RMIT-BDSL/Blockchain-Anomaly-Detection](https://github.com/RMIT-BDSL/Blockchain-Anomaly-Detection)  
> 📌 总结一句话：**没有放之四海皆准的最佳训练策略；应根据 GNN 架构特性“量体裁衣”地选择 initialization 与 normalization 方案。**

</details>

---

### 15. [RUMAD: Reinforcement-Unifying Multi-Agent Debate](https://arxiv.org/abs/2602.23864)

**Authors**: Chao Wang, Han Lin, Huaze Tang, Huijing Lin, Wenbo Ding  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.23864v1  

#### Abstract
Multi-agent debate (MAD) systems leverage collective intelligence to enhance reasoning capabilities, yet existing approaches struggle to simultaneously optimize accuracy, consensus formation, and computational efficiency. Static topology methods lack adaptability to task complexity variations, while...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《RUMAD: Reinforcement-Unifying Multi-Agent Debate》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **Multi-agent Debate (MAD)** 系统在以下三方面存在显著缺陷：
- **适应性差**：多数采用静态通信拓扑（如环形、星形），无法根据任务复杂度动态调整，导致简单任务冗余、复杂任务信息不足。
- **隐私与中立性风险**：部分自适应方法依赖外部更强的 LLM（如 GPT-4）作为“裁判”或“摘要器”，引入外部知识，破坏内部 agent 的独立性和多样性。
- **效率建模缺失**：缺乏对 token 成本的显式控制，难以在真实部署中平衡推理质量与计算开销。

### **提出的新方法与思路**
论文提出了 **RUMAD**（Reinforcement-Unifying Multi-Agent Debate），一种基于强化学习（RL）的动态通信拓扑控制框架，其核心创新包括：

- **将 MAD 拓扑控制建模为 RL 问题**：使用 PPO 算法训练一个中央控制器，动态调整 agent 间的通信边权重。
- **内容无关的观察机制（Content-agnostic Observation）**：控制器仅观察高阶辩论动态（如语义相似性、答案一致性、通信成本），不访问 agent 的原始推理内容，保障中立性。
- **多目标奖励函数**：联合优化准确性（Accuracy）、共识形成（Consensus）和通信效率（Efficiency），并引入预算正则项（`L_budget`）鼓励稀疏通信。
- **双阈值机制实现细粒度控制**：
  - **边可见性控制**：通过量化权重决定信息呈现优先级（Critical / Reference / Background）。
  - **节点激活控制**：若 agent 外部影响低于阈值，则复用旧响应，避免生成新 token。

### **相比现有方法的优势**
| 维度 | 静态拓扑方法（如 S-MAD, GD） | 外部 LLM 协调方法 | RUMAD |
|------|-------------------------------|------------------|--------|
| **适应性** | ❌ 固定结构 | ✅ 动态但依赖外部模型 | ✅ 动态且内生 |
| **中立性** | ✅ 不依赖外部知识 | ❌ 引入外部观点 | ✅ 内容无关控制 |
| **效率控制** | ⚠️ 间接限制轮次 | ❌ 通常更高成本 | ✅ 显式 token 节约 |
| **泛化能力** | ⚠️ 任务特定设计 | ❌ 微调需求高 | ✅ 零样本迁移有效 |

> ✅ RUMAD 在保持甚至提升准确率的同时，实现了超过 **80% 的 token 成本降低**，并具备强大的跨域零样本泛化能力。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MMLU**：涵盖 57 个学科的多选题基准，用于训练与主评估。
- **GSM8K**：小学数学应用题，需链式推理，测试算术能力。
- **GPQA**：研究生级别科学问答，难度极高，检验深度理解。

> 🔹 **训练仅在 MMLU 开发集上进行**，其余均为零样本迁移评估。

### **实验设置**
- **Agent 构成**：6 个 agent，来自三种不同 LLM：
  - LLaMA-3.1-8B-Instruct
  - ChatGLM-4-9B
  - Deepseek-Math-7B-Instruct
  （每种两个实例）
- **量化配置**：所有模型使用 4-bit 量化（Q4_K_M），部署于单张 RTX 3090。
- **嵌入模型**：Nomic-Embed-v1 用于计算推理内容的语义相似性。
- **通信轮数**：最多 6 轮。
- **预算参数 B**：控制每轮最大活跃连接数（如 B=12 表示平均每个 agent 最多接收 2 条消息）。

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Accuracy (ACC)** | 多数投票结果是否正确 |
| **Token Cost (TC)** | 平均每任务消耗的 token 数量（k/token） |
| **Cost Saving** | 相较于全连接 MAD 的 token 节省比例 |
| **Zero-shot Generalization** | 在未训练过的任务上的表现 |

### **基线方法对比**
- **MAD**：全连接通信，无剪枝。
- **S-MAD**：固定稀疏拓扑（星形 `*` 和环形 `o`）。
- **GD (Group Debate)**：分组辩论后聚合。
- **S²-MAD**：两阶段稀疏辩论。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 2）**

| 方法 | MMLU ACC | MMLU TC (k) | 节省 |
|------|----------|-------------|-------|
| MAD (baseline) | 49% | 62.58 | — |
| S-MAD* | 61% | 33.32 | -46.75% |
| GD | 53% | 32.17 | -48.59% |
| **RUMAD (B=12)** | **68%** | **11.43** | **-81.74%** |

| 方法 | GSM8K ACC | GSM8K TC (k) | 节省 |
|------|-----------|--------------|-------|
| MAD | 88% | 76.90 | — |
| GD | 84% | 29.76 | -61.30% |
| **RUMAD (B=12)** | **86%** | **10.46** | **-86.40%** |

| 方法 | GPQA ACC | GPQA TC (k) | 节省 |
|------|----------|-------------|-------|
| MAD | 30% | 57.80 | — |
| GD | 34% | 26.05 | -54.93% |
| **RUMAD (B=18)** | **35%** | **33.39** | **-42.24%** |

> 📈 **结论**：RUMAD 在所有任务上均以极低 token 成本实现最优或接近最优准确率。

### **与基线方法的对比结果**
- **效率优势显著**：token 成本平均下降 **>80%**，最高达 **86.4%**（GSM8K）。
- **精度反超**：在 MMLU 上比全连接 MAD 提升 **19%** 准确率，同时节省 80%+ token。
- **优于所有稀疏基线**：无论是固定结构还是分组策略，RUMAD 均在精度和效率上全面领先。

### **消融实验结果（见 Table 3）**
| 消融模块 | MMLU ACC ↓ | MMLU TC ↑ | 结论 |
|---------|------------|-----------|------|
| 完整 RUMAD | **68.3%** | **11.4k** | — |
| 替换为 β 分布 | 65.1% | 18.7k | 高斯分布更优 |
| 移除 `Rep`（终局奖励） | 66.9% | 15.6k | 损害长期目标 |
| 移除 `Rt`（逐轮奖励） | 67.2% | 22.2k | 效率大幅下降 |
| 移除 agent 激活机制 | 65.4% | **51.9k** | token 成本激增 4.5x |
| 移除 `L_budget` 正则项 | 70.0% | 17.4k | 泛化能力下降（OOD 性能差） |

> 🔍 **关键发现**：
> - **Agent 激活机制是 token 节约的核心**。
> - **双层奖励设计（`Rt + Rep`）对平衡即时与最终目标至关重要**。
> - **`L_budget` 不仅控成本，还提升泛化性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **动态拓扑控制可通过 RL 实现高效且中立的 MAD 管理**。
2. **内容无关的设计保障了系统的公平性与可扩展性**，避免外部知识污染。
3. **RUMAD 具备强大的零样本泛化能力**：在完全未见过的 GSM8K 和 GPQA 上仍表现优异，说明学到的是**通用协作原则**而非过拟合模式。
4. **通信预算 `B` 是可解释、可控的超参**，允许用户在资源约束下灵活部署。
5. **扩大 agent 规模可在固定预算下提升性能**（见 Table 4）：从 6→9 个 agent，MMLU 准确率从 68.5% → 70.3%，token 成本反而下降，体现智能调度能力。

### **方法的局限性**
- **中心化控制器可能面临扩展瓶颈**：当前适用于中小规模（6–9 agent），若扩展至百级 agent 可能需要去中心化或多层级 RL 架构。
- **依赖高质量 embedding 模型**：语义相似性计算受 embedding 模型影响。
- **初始化多样性要求高**：若所有 agent 同构，可能削弱辩论价值。

### **未来工作方向**
- 探索 **Decentralized RL** 或 **Hierarchical MARL** 以支持更大规模 agent 群体。
- 将 RUMAD 扩展至 **非辩论类 multi-agent 协作场景**（如规划、工具调用）。
- 引入 **在线学习机制**，使控制器能在部署中持续优化。
- 研究 **异步通信机制**，进一步逼近真实人类辩论节奏。

---

> ✅ **总体评价**：  
> RUMAD 是首个将 **动态通信拓扑控制** 与 **强化学习**、**内容无关性** 和 **显式效率建模** 结合的 MAD 框架，在准确性、效率、中立性和泛化性之间取得了卓越平衡，为大规模 LLM 多智能体系统的实用化提供了重要路径。

</details>

---

### 16. [CIRCLE: A Framework for Evaluating AI from a Real-World Lens](https://arxiv.org/abs/2602.24055)

**Authors**: Reva Schwartz, Carina Westling, Morgan Briggs, Marzieh Fadaee, Isar Nejadgholi, Matthew Holmes, Fariza Rashid, Maya Carlyle, Afaf Ta\"ik, Kyra Wilson, Peter Douglas, Theodora Skeadas, Gabriella Waters, Rumman Chowdhury, Thiago Lacerda  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.24055v1  

#### Abstract
This paper proposes CIRCLE, a six-stage, lifecycle-based framework to bridge the reality gap between model-centric performance metrics and AI's materialized outcomes in deployment. While existing frameworks like MLOps focus on system stability and benchmarks measure abstract capabilities, decision-m...

---

### 17. [Learning Flexible Job Shop Scheduling under Limited Buffers and Material Kitting Constraints](https://arxiv.org/abs/2602.24180)

**Authors**: Shishun Zhang, Juzhan Xu, Yidan Fan, Chenyang Zhu, Ruizhen Hu, Yongjun Wang, Kai Xu  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.24180v1  

#### Abstract
The Flexible Job Shop Scheduling Problem (FJSP) originates from real production lines, while some practical constraints are often ignored or idealized in current FJSP studies, among which the limited buffer problem has a particular impact on production efficiency. To this end, we study an extended p...

---

### 18. [Truncated Step-Level Sampling with Process Rewards for Retrieval-Augmented Reasoning](https://arxiv.org/abs/2602.23440)

**Authors**: Chris Samarinas, Haw-Shiuan Chang, Hamed Zamani  
**Category**: cs.CL  
**Published**: 2026-03-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.23440v1  

#### Abstract
Training large language models to reason with search engines via reinforcement learning is hindered by a fundamental credit assignment problem: existing methods such as Search-R1 provide only a sparse outcome reward after an entire multi-step trajectory, making it infeasible to attribute success or ...

---

### 19. [pathsig: A GPU-Accelerated Library for Truncated and Projected Path Signatures](https://arxiv.org/abs/2602.24066)

**Authors**: Tobias Nygaard  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.24066v1  

#### Abstract
Path signatures provide a rich representation of sequential data, with strong theoretical guarantees and good performance in a variety of machine-learning tasks. While signatures have progressed from fixed feature extractors to trainable components of machine-learning models, existing libraries ofte...

---

### 20. [LemmaBench: A Live, Research-Level Benchmark to Evaluate LLM Capabilities in Mathematics](https://arxiv.org/abs/2602.24173)

**Authors**: Antoine Peyronnet, Fabian Gloeckle, Amaury Hayat  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.24173v1  

#### Abstract
We present a new approach for benchmarking Large Language Model (LLM) capabilities on research-level mathematics. Existing benchmarks largely rely on static, hand-curated sets of contest or textbook-style problems as proxies for mathematical research. Instead, we establish an updatable benchmark eva...

---

### 21. [Hestia: Hyperthread-Level Scheduling for Cloud Microservices with Interference-Aware Attention](https://arxiv.org/abs/2602.23758)

**Authors**: Dingyu Yang, Fanyong Kong, Jie Dai, Shiyou Qian, Shuangwei Li, Jian Cao, Guangtao Xue, Gang Chen  
**Category**: cs.DC  
**Published**: 2026-03-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.23758v1  

#### Abstract
Modern cloud servers routinely co-locate multiple latency-sensitive microservice instances to improve resource efficiency. However, the diversity of microservice behaviors, coupled with mutual performance interference under simultaneous multithreading (SMT), makes large-scale placement increasingly ...

---

### 22. [On the Convergence of Single-Loop Stochastic Bilevel Optimization with Approximate Implicit Differentiation](https://arxiv.org/abs/2602.23633)

**Authors**: Yubo Zhou, Luo Luo, Guang Dai, Haishan Ye  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.23633v1  

#### Abstract
Stochastic Bilevel Optimization has emerged as a fundamental framework for meta-learning and hyperparameter optimization. Despite the practical prevalence of single-loop algorithms--which update lower and upper variables concurrently--their theoretical understanding, particularly in the stochastic r...

---

### 23. [OPTIAGENT: A Physics-Driven Agentic Framework for Automated Optical Design](https://arxiv.org/abs/2602.23761)

**Authors**: Yuyu Geng, Lei Sun, Yao Gao, Xinxin Hu, Zhonghua Yi, Xiaolong Qian, Weijian Hu, Jian Bai, Kaiwei Wang  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.23761v1  

#### Abstract
Optical design is the process of configuring optical elements to precisely manipulate light for high-fidelity imaging. It is inherently a highly non-convex optimization problem that relies heavily on human heuristic expertise and domain-specific knowledge. While Large Language Models (LLMs) possess ...

---

### 24. [PseudoAct: Leveraging Pseudocode Synthesis for Flexible Planning and Action Control in Large Language Model Agents](https://arxiv.org/abs/2602.23668)

**Authors**: Yihan (Logon),  Wen, Xin Chen  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.23668v1  

#### Abstract
Large language model (LLM) agents typically rely on reactive decision-making paradigms such as ReAct, selecting actions conditioned on growing execution histories. While effective for short tasks, these approaches often lead to redundant tool usage, unstable reasoning, and high token consumption in ...

---

### 25. [Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume](https://arxiv.org/abs/2602.24195)

**Authors**: Gregory Kang Ruey Lau, Hieu Dao, Nicole Kan Hui Lin, Bryan Kian Hsiang Low  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.24195v1  

#### Abstract
Despite their capabilities, Multimodal Large Language Models (MLLMs) may produce plausible but erroneous outputs, hindering reliable deployment. Accurate uncertainty metrics could enable escalation of unreliable queries to human experts or larger models for improved performance. However, existing un...

---

### 26. [EDDA-Coordinata: An Annotated Dataset of Historical Geographic Coordinates](https://arxiv.org/abs/2602.23941)

**Authors**: Ludovic Moncla, Pierre Nugues, Thierry Joliveau, Katherine McDonough  
**Category**: cs.CL  
**Published**: 2026-03-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.23941v1  

#### Abstract
This paper introduces a dataset of enriched geographic coordinates retrieved from Diderot and d'Alembert's eighteenth-century Encyclopedie. Automatically recovering geographic coordinates from historical texts is a complex task, as they are expressed in a variety of ways and with varying levels of p...

---

### 27. [MT-PingEval: Evaluating Multi-Turn Collaboration with Private Information Games](https://arxiv.org/abs/2602.24188)

**Authors**: Jacob Eisenstein, Fantine Huot, Adam Fisch, Jonathan Berant, Mirella Lapata  
**Category**: cs.CL  
**Published**: 2026-03-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.24188v1  

#### Abstract
We present a scalable methodology for evaluating language models in multi-turn interactions, using a suite of collaborative games that require effective communication about private information. This enables an interactive scaling analysis, in which a fixed token budget is divided over a variable num...

---

### 28. [Active Value Querying to Minimize Additive Error in Subadditive Set Function Learning](https://arxiv.org/abs/2602.23529)

**Authors**: Martin \v{C}ern\'y, David Sychrovsk\'y, Filip \'Uradn\'ik, Jakub \v{C}ern\'y  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2602.23529v1  

#### Abstract
Subadditive set functions play a pivotal role in computational economics (especially in combinatorial auctions), combinatorial optimization or artificial intelligence applications such as interpretable machine learning. However, specifying a set function requires assigning values to an exponentially...

---

### 29. [From Flat Logs to Causal Graphs: Hierarchical Failure Attribution for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2602.23701)

**Authors**: Yawen Wang, Wenjie Wu, Junjie Wang, Qing Wang  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2602.23701v1  

#### Abstract
LLM-powered Multi-Agent Systems (MAS) have demonstrated remarkable capabilities in complex domains but suffer from inherent fragility and opaque failure mechanisms. Existing failure attribution methods, whether relying on direct prompting, costly replays, or supervised fine-tuning, typically treat e...

---

### 30. [DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](https://arxiv.org/abs/2602.24288)

**Authors**: Fan Shu, Yite Wang, Ruofan Wu, Boyi Liu, Zhewei Yao, Yuxiong He, Feng Yan  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2602.24288v1  

#### Abstract
The fast-growing demands in using Large Language Models (LLMs) to tackle complex multi-step data science tasks create an emergent need for accurate benchmarking. There are two major gaps in existing benchmarks: (i) the lack of standardized, process-aware evaluation that captures instruction adherenc...

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

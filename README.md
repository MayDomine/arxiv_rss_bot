# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-08-10 07:00:04 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Retrofitting Linear Attention into Diffusion Language Models](https://arxiv.org/abs/2608.06628)

**Authors**: Jinha Kim, Younghun Roh, Jaeyeon Kim  
**Category**: cs.LG  
**Published**: 2026-08-10  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2608.06628v1  

#### Abstract
Diffusion language models (dLLMs) offer a promising alternative to autoregressive models by accelerating inference through parallel decoding. Recent dLLMs commonly use blockwise semi-autoregressive decoding, generating blocks autoregressively while denoising tokens within each active block in parall...

---

### 2. [HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management](https://arxiv.org/abs/2608.07009)

**Authors**: Zhiqiang Xie, Zhangheng Huang, Tingwei Huang, Ziyi Xu, Ruiyang Ma, Christos Kozyrakis  
**Category**: cs.DC  
**Published**: 2026-08-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.07009v1  

#### Abstract
Top-k sparse attention makes long-context LLM decoding cheap to compute: each step reads only a few thousand selected KV entries rather than the full context. Serving systems, however, typically keep the entire KV cache in GPU HBM so that every position stays selectable, so a request's memory bill s...

---

### 3. [Theoretical Foundations of Communication-Efficient, Robust, and Practical Distributed and Federated Optimization](https://arxiv.org/abs/2608.06563)

**Authors**: Grigory Malinovsky  
**Category**: cs.LG  
**Published**: 2026-08-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.06563v1  

#### Abstract
Machine learning and optimization have advanced together, with practical demands motivating new theory and theoretical breakthroughs enabling new applications. Modern large-scale training relies on classical optimization principles, but the constraints of distributed systems require these foundation...

---

### 4. [Synthetic LiDAR Data Generation and Deterministic Downsampling for Point Cloud Classification on the Edge](https://arxiv.org/abs/2608.07106)

**Authors**: Niclas Meyer, Stefan Reitmann  
**Category**: cs.LG  
**Published**: 2026-08-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.07106v1  

#### Abstract
Deploying three-dimensional deep learning frameworks to low-power embedded processors is bottlenecked by the unstructured nature of spatial data and the resource-intensive distance sorting algorithms often used before neural network inference. To address this gap, this paper presents a hardware-cons...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Synthetic LiDAR Data Generation and Deterministic Downsampling for Point Cloud Classification on the Edge*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文旨在解决在**边缘设备**（如 Raspberry Pi 5）上部署 3D 深度学习模型时面临的两大瓶颈：
1. **现实差距（Reality Gap）**：大多数 3D 分类网络在干净的 CAD 数据集（如 ModelNet）上训练，但在真实 LiDAR 数据中表现严重下降，因后者存在噪声、稀疏性和视角依赖等问题。
2. **计算延迟瓶颈**：传统几何预处理方法（如 Farthest Point Sampling, FPS）在边缘 CPU 上计算开销大，成为推理流程中的性能瓶颈。

### ✅ 提出的新方法与创新思路
1. **基于物理仿真的合成 LiDAR 数据生成**  
   利用开源工具 **BLAINDER**（Blender 插件），对 ModelNet40 中的 CAD 模型进行物理真实的 LiDAR 扫描模拟，生成带有**高斯噪声**和**视角依赖稀疏性**的合成 LiDAR 数据，以桥接仿真与现实之间的“传感器现实差距”。

2. **引入并优化 Critical Point Layer (CPL) 作为前端滤波器**  
   将 **CPL** 集成到 PointNet 架构前端，作为一个**可训练、确定性、特征驱动的下采样模块**。该层通过共享 MLP 将点映射至高维特征空间，并利用全局 max-pooling 提取最具分类意义的关键点（Critical Points），实现高效压缩。

3. **端到端训练 + 冻结部署策略**  
   - 先将 CPL 与 PointNet 联合训练，使其学习保留对分类最关键的几何特征；
   - 训练完成后冻结 CPL 参数，将其作为独立的确定性下采样模块部署于边缘设备，避免重复训练开销。

### ✅ 相比现有方法的优势
| 方法 | 缺陷 | 本文改进 |
|------|------|---------|
| **FPS** | 时间复杂度为 $O(N \times M)$，在边缘设备上延迟高 | CPL 运行速度更快（约快 3 倍），且为确定性输出 |
| **Random Sampling (RS)** | 非确定性，可能丢失关键结构 | CPL 是确定性的，且保留语义关键点 |
| **直接使用 ModelNet 训练** | 在真实 LiDAR 数据上准确率暴跌 | 使用 BLAINDER 合成数据后显著提升泛化能力 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
1. **ModelNet40**：标准合成 CAD 数据集，用于构建基线。
2. **Synthetic LiDAR Datasets（自建）**：
   - **Rotational LiDAR Dataset**：模拟 Velodyne UltraPuck 扫描仪，从多个角度旋转扫描对象，包含 clean 和 noisy（加高斯噪声，$\sigma=0.01$）两个版本。
   - **Generic LiDAR Dataset**：固定单视角扫描，同样分为 clean/noisy 子集。
   > 所有数据均公开发布：[DOI:10.5281/zenodo.21835460](https://doi.org/10.5281/zenodo.21835460)

### ⚙️ 实验设置
- **主干网络**：PointNet（PyTorch 实现）
- **训练配置**：
  - 80/20 train-test split
  - Batch size: 24
  - Optimizer: Adam ($lr = 0.001$)
  - Epochs: 200
- **硬件平台**：
  - 训练：工作站 GPU
  - 推理测试：Raspberry Pi 5（ARM Cortex-A76 CPU）
- **评估指标**：
  - Instance Accuracy（实例级准确率）
  - Class Accuracy（类别平均准确率）
  - Inference Latency（推理延迟）
  - Chamfer Distance（与 LIME-3D 解释结果的几何相似性）

### 🔁 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Uniform Sampling (1024 pts)** | Baseline | 标准 ModelNet 设置 |
| **Random Sampling (RS)** | Heuristic | 快速但非确定性 |
| **Farthest Point Sampling (FPS)** | Geometry-driven | 空间覆盖均匀，但慢 |
| **No Preprocessing** | None | 直接输入原始点云 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）**跨数据集泛化能力验证（Cross-Dataset Evaluation）**
> 表明：**仅在 clean CAD 上训练的模型无法适应 LiDAR 数据**

| 训练数据 → 测试数据 | 准确率（Instance Acc.） |
|---------------------|------------------------|
| ModelNet (clean) → Rotational Noisy | **2.28%** |
| Rotational Noisy → ModelNet (clean) | **59.10%** |
| Rotational Noisy → Rotational Clean | **57.35%** |

> 💡 发现：**在含噪数据上训练的模型能较好泛化到干净数据，反之则失败**，说明噪声训练增强了鲁棒性。

#### （2）**PointNet 对点密度的鲁棒性（Table 1）**
即使只保留 **16 个点**，PointNet 仍能达到 **83.01%** 实例准确率，证明其对稀疏输入具有高度容忍性。

| 输入点数 | 实例准确率 |
|----------|------------|
| 1024     | 91.09%     |
| 512      | 90.63%     |
| 128      | 89.86%     |
| **64**   | **89.01%** |
| 16       | 83.01%     |

#### （3）**CPL 下采样性能**
- 成功将 **1024 点** 压缩至 **40–60 个唯一坐标**。
- 在 Raspberry Pi 5 上，CPL 处理 **128 目标点** 平均耗时 **1.96ms**，而 FPS 需 **5.73ms**，速度快近 **3 倍**。

#### （4）**最终系统性能（CPL + PointNet）**
- 下采样至 **64 点** 后，系统在 5000 个测试样本上的准确率为 **88.36%**，接近全量随机采样的 89.01%，几乎无性能损失。
- 整体推理吞吐量达 **~50 FPS**，满足实时边缘应用需求。

#### （5）**与解释性方法的几何一致性分析（Chamfer Distance）**
| 方法 vs LIME-3D | CD 值（越低越好） |
|------------------|------------------|
| Random Sampling | ~0.013–0.017 |
| **CPL**         | **~0.026–0.045** |

> ❗ CPL 选出的点与人类可解释区域（LIME-3D）差异较大，说明其更关注**分类效用**而非**感知合理性**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **存在严重的“传感器现实差距”**  
   在干净 CAD 数据上训练的模型在面对模拟 LiDAR 数据时性能崩溃（<3%），必须使用带噪声的合成数据进行训练才能获得良好泛化。

2. **CPL 是一种高效的边缘友好型下采样方案**  
   - 具备**确定性输出**，适合嵌入式部署；
   - 比 FPS 快 **3 倍以上**，显著降低预处理延迟；
   - 与 PointNet 联合训练后，可在极低点数下维持高分类精度。

3. **边缘设备上的实时 3D 感知是可行的**  
   在 Raspberry Pi 5 上实现了 **50 FPS** 的端到端推理速度，结合 88.36% 的准确率，验证了轻量化 3D 感知系统的实用性。

4. **PointNet 对数据稀疏性高度鲁棒**  
   即使仅用 16 个点也能保持超过 80% 的准确率，为边缘剪枝提供了理论依据。

### ⚠️ 方法的局限性
1. **未包含 intensity 信息**  
   BLAINDER 当前生成的点云缺少真实 LiDAR 中的反射强度（intensity），限制了进一步逼近真实传感器行为的能力。

2. **CPL 输出点数不稳定**  
   因 max-pooling 的特性，不同样本压缩后的唯一点数不一致，需通过复制或截断来标准化输出，影响效率。

3. **静态场景假设**  
   所有实验基于静态物体分类，未涉及动态点云序列或 object detection / semantic segmentation 等更复杂任务。

4. **解释性与功能性脱节**  
   CPL 提取的“关键点”虽有助于分类，但不符合人类直觉（CD 较高），不利于调试与可信 AI 构建。

### 🔮 未来工作方向
1. **扩展至 FPGA/NPU 加速**  
   将 CPL 的 shared MLP 部分卸载至 **FPGA** 或专用 NPU，实现更高并发与更低功耗。

2. **引入可微分 Top-K 操作**  
   使用 **Gumbel-Top-k** 等连续松弛技术替代硬筛选，实现完全可微的软采样机制，避免信息坍塌。

3. **支持天气与动态干扰建模**  
   在 BLAINDER 中加入雨、雾、运动模糊等环境扰动，生成更具挑战性的合成数据集。

4. **探索多视角融合与序列建模**  
   扩展当前单帧分类框架，支持 temporal aggregation 与时序建模，适用于机器人导航等动态场景。

---

> **总结一句话**：  
> 本文提出了一套面向边缘设备的完整 3D 分类流水线，通过**物理仿真生成逼真 LiDAR 数据** + **特征驱动的 CPL 下采样**，成功在 Raspberry Pi 5 上实现了 **50 FPS / 88.36% 准确率** 的实时高性能推理，为资源受限场景下的 3D 感知提供了实用解决方案。

</details>

---

### 5. [An AI4AI Framework for Visual Token Pruning](https://arxiv.org/abs/2608.07193)

**Authors**: Zhen Liu, Wenli Huang, Wei Song, Yuhan Liu, Zhiqin Yang, Jingwen Fu  
**Category**: cs.LG  
**Published**: 2026-08-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.07193v1  

#### Abstract
Visual-token pruning can substantially reduce the inference cost of multimodal large language models (MLLMs), yet existing methods largely rely on fixed, handcrafted heuristics and costly expert trial and error. As pruning objectives, budgets, and model architectures diversify, manually navigating t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《An AI4AI Framework for Visual Token Pruning》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
视觉-语言大模型（MLLMs）在推理过程中会生成大量视觉 token，显著增加计算开销（如 FLOPs、prefill 延迟、KV Cache 占用），限制其部署效率。现有的**视觉 token 剪枝方法**大多依赖人工设计的启发式规则（handcrafted heuristics），例如基于注意力重要性、冗余合并或多样性选择等策略。这些方法需要大量专家调参与试错，难以适应多样化的剪枝目标、token 预算和模型架构。

随着 MLLM 架构和任务需求日益复杂，手动探索庞大的剪枝算法设计空间变得不可持续。

### 提出了什么新方法或新思路
本文提出 **AutoPrune** —— 一个训练免费（training-free）、由 LLM 驱动的 AI4AI 框架，用于自动设计高效的视觉 token 剪枝策略。

其核心思想是将 LLM 的通用算法知识转化为特定于视觉 token 剪枝的有效解决方案，关键在于引入了一种**残差搜索状态表示法**（residual search-state representation）。

为此，作者设计了：
- **Token Pruning Domain-Specific Language (TPDSL)**：一种领域专用语言，包含 **131 个可复用的“原子”操作**，涵盖预算控制（Budget）、token 打分（Scoring）、选择约束（Constraint）和重组（Reassembly）四大模块。
- **残差修改机制**：每个候选剪枝策略不是从零开始构建完整程序，而是作为对一个强基线策略（base policy）的**受限残差修正**。这缩小了搜索空间，并引导 LLM 聚焦于最关键的改进部分。
- **Evaluator-in-the-loop 迭代优化**：通过 LLM 提出 TPDSL 结构 → 实例化为可执行策略 → 安全性验证 → 在目标任务上评估 → 反馈历史记录 → 再次迭代的闭环流程，实现自动化策略搜索。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|--------|
| **自动化程度高** | 不再依赖专家手工设计，实现了剪枝算法的自动生成与优化。 |
| **高效且安全** | 使用 TPDSL 替代自由代码生成，确保语法正确性、预算一致性、索引有效性和数值稳定性。 |
| **跨预算与跨架构迁移性强** | 搜索得到的 TPDSL 结构可在不同 token 数量和不同 MLLM backbone 上直接重实例化，无需重新搜索。 |
| **性能更强** | 在极端剪枝比例下仍保持接近全 token 性能，显著优于各类 handcrafted 方法。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共在 **14 个多模态基准测试集**上进行评估，覆盖多种视觉理解能力：
- **通用问答**：VQAv2, GQA, VizWiz
- **科学推理**：ScienceQA-IMG
- **幻觉检测**：HallBench, POPE
- **综合评测**：MME, MMBench-EN/CN, MM-Vet
- **文本密集型图像理解**：TextVQA, ChartQA, AI2D, OCRBench

所有实验遵循官方划分、提示格式和评估指标。

### 实验设置和评估指标
- **MLLM Backbone**：LLaVA-1.5-7B, LLaVA-NeXT-7B, Qwen2.5-VL-7B
- **剪枝比例**：最高达 **94.4%**（如 576→32 或 2880→160）
- **评估指标**：
  - 各数据集的官方得分（如准确率）
  - **Aggregate Score (Acc.)**：归一化后的平均分数
  - **Relative Performance (Rel.)**：相对于全 token 模型的表现百分比
  - 推理效率指标：FLOPs, Prefill Latency, Decoding Latency, KV Cache Size, GPU Memory

- **搜索配置**：
  - 在 LLaVA-1.5-7B + MME 上执行一次搜索（source setting）
  - 搜索轮数：10 轮，每轮生成 5 个候选 → 共 50 个候选
  - LLM Proposer：默认使用 Qwen-Plus
  - Base Policy：CDPruner
  - 残差交换配额：qe=2, 最少保留基线 token 数：rmin=30

### 基线方法对比
与以下代表性训练免费剪枝方法比较：
- **Importance-based**: FastV, PyramidDrop (PDrop), Sparse-VLM
- **Redundancy-reduction**: PruMerge+, TRIM, VisionZip
- **Diversity-aware**: DART, DivPrune, CDPruner

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 在 **94.4% 剪枝比例**下的总体表现（见 Table 1 和 Figure 1）

| Model | 方法 | #Tokens | Acc. | Rel. |
|-------|------|--------|------|------|
| LLaVA-1.5-7B | Full Model | 576 | 63.40 | 100.0% |
| | **AutoPrune (Ours)** | **32** | **63.20** | **99.7%** |
| | CDPruner | 32 | 60.00 | 94.3% |
| LLaVA-NeXT-7B | Full Model | 2880 | 65.20 | 100.0% |
| | **AutoPrune (Ours)** | **160** | **65.20** | **99.9%** |
| | CDPruner | 160 | 62.80 | 96.0% |

> ✅ **即使移除 94.4% 的视觉 token，AutoPrune 仍能保留超过 99% 的原始性能！**

#### 推理效率提升（Table 2）
在 LLaVA-NeXT-7B 上保留 320 个 token：
- **FLOPs 减少 9.9×**
- **Prefill 延迟降低 6.4×**
- KV Cache 从 1440.0 MB → 160.0 MB
- MME 分数从 1453.0（CDPruner）提升至 **1457.9**

> ⚡ 效率大幅提升的同时，还带来了精度增益！

### 与基线方法的对比结果
- 在 LLaVA-1.5-7B 上，相比最强基线 CDPruner，**绝对 Acc 提升 3.2 点**（63.2 vs 60.0）
- 在 LLaVA-NeXT-7B 上，**提升 2.4 点**（65.2 vs 62.8）
- 图 3 显示，在更激进的剪枝比例下（77.8% → 94.4%），AutoPrune 的优势更加明显。
- 多项消融显示，某些情况下剪枝后模型甚至**超越全 token 模型性能**（Rel. > 100%），表明合理剪枝可减少噪声干扰。

### 消融实验结果（Ablation Studies）

#### （1）TPDSL 组件作用（Table 4）
| 方法 | MME 得分 | ΔBase | ΔFull |
|------|--------|--------|--------|
| CDPruner Base | 1373.00 | +0.00 | -40.46 |
| w/o Reference Anchoring | 1223.29 | -149.71 | -190.17 |
| w/o Multi-Score Fusion | 1391.70 | +18.70 | -21.76 |
| w/o Diversity Selection | 1400.80 | +27.80 | -12.66 |
| **Full TPDSL (Ours)** | **1413.46** | **+40.46** | **+0.00** |

> 🔑 **Reference Anchoring（参考锚定）最为关键**，说明保留基线结构至关重要。

#### （2）残差交换配额 $ q_e $ 影响（Table 6）
| $ q_e $ | MME |
|--------|-----|
| 0 | 1373.00 |
| 1 | 1384.95 |
| **2** | **1413.46** ✅ |
| 4 | 1394.28 |
| 8 | 1362.48 |
| 32 | 1217.88 |

> 🎯 **仅替换最多 2 个 token 效果最佳**，过多替换反而破坏可靠结构。

#### （3）不同 LLM Proposer 的鲁棒性（Table 5）
| LLM Proposer | MME |
|--------------|-----|
| Qwen-Max | 1411.04 |
| Qwen-Plus | 1413.46 |
| DeepSeek-V4-Flash | 1414.34 |

> 🔄 不同 LLM 输出差异极小（< 3.3 分），说明性能提升主要来自 TPDSL 结构而非特定 LLM。

#### （4）跨 Base Policy 泛化性（Table 7）
| Base Strategy | Strategy Only | AutoPrune | Gain |
|-------------|---------------|-----------|------|
| PruMerge+ | 1142.82 | 1288.22 | +145.40 |
| VisionZip | 1229.72 | 1357.52 | +127.80 |
| ... | ... | ... | ... |

> 🔄 AutoPrune 可显著增强多种不同类型的基线剪枝策略，证明其通用性。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 可以有效驱动剪枝算法设计**：通过合适的结构化表示（TPDSL），可以将 LLM 的通用算法知识迁移到高度约束的专业任务中。
2. **残差搜索优于端到端生成**：以强基线为基础进行受限修改，比从头生成完整剪枝程序更稳定、高效且性能更高。
3. **高性能与高效率兼得**：AutoPrune 在大幅压缩视觉 token 的同时，不仅维持了几乎全部性能，还在多个任务上实现反超。
4. **强大的迁移能力**：在一个 backbone 和 budget 上搜索出的 TPDSL 策略，可无缝迁移到其他模型和预算设置，无需额外搜索。

### 方法的局限性
- **依赖 TPDSL 的表达能力**：当前 131 个原子可能无法覆盖所有潜在的有效操作组合。
- **依赖任务评估器质量**：若 evaluator 存在偏差，可能导致搜索偏向局部最优。
- **未探索动态调整策略**：目前策略是静态的，未根据输入内容动态变化。

### 未来工作方向
- 设计更丰富、更具表达力的 TPDSL 原子集合。
- 引入更智能、自适应的任务评估机制（如 curriculum learning）。
- 将框架扩展至视频、音频等多模态序列的 token 剪枝。
- 探索在线自适应剪枝策略生成。

---

> 💡 **一句话总结**：  
> AutoPrune 成功地将“如何设计剪枝算法”这一难题，转化为“如何在结构化空间中进行残差搜索”的可控问题，首次实现了真正意义上的 LLM-driven 自动化视觉 token 剪枝，兼具高性能、高效率与强泛化性。

</details>

---

### 6. [StateFlow: Sequence Pipeline Parallelism for Long-Context Modeling with Linear Recurrence](https://arxiv.org/abs/2608.06838)

**Authors**: Wenxuan Zhao, Yingfa Chen, Xu Han, Wenjing Han, Tianbo Huang, Zhiyu Li, Ao Sun, Jingheng Xu, Lin Gan, Guangwen Yang  
**Category**: cs.DC  
**Published**: 2026-08-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.06838v1  

#### Abstract
Long-context training is increasingly important for large language models, and linear attention and state space models have become popular for improving long-context efficiency. However, efficiently parallelizing long-sequence training for recurrent and hybrid models remains challenging.

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：StateFlow: Sequence Pipeline Parallelism for Long-Context Modeling with Linear Recurrence**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **长上下文训练中的内存瓶颈**：尽管 linear recurrence（如 linear attention 和 SSM）在计算上具有线性时间复杂度和常量空间复杂度，适合长序列建模，但在大规模训练中，**activation memory** 仍是主要瓶颈。
- **现有并行策略的不足**：
  - **Pipeline Parallelism (PP)** 虽然减少模型内存占用，但每个 pipeline 单元仍处理完整序列，导致高激活内存峰值。
  - **Sequence Pipeline Parallelism (SPP)** 已用于 softmax attention 模型，但未有效支持具有严格状态依赖的 linear recurrence 模型。
  - **混合架构**（linear recurrence + softmax attention）中，不同层对 chunk 划分的需求不一致，难以平衡负载。

### **提出的新方法**
- **StateFlow**：一种面向 linear recurrence 和 hybrid 模型的 **sequence pipeline parallelism (SPP)** 系统，其核心机制包括：
  1. **State-aware Chunk Scheduling**：
     - 将序列划分为 chunks，并作为 pipeline 调度单元。
     - 显式管理 chunk 边界处的 **recurrent state** 和 **state gradients**，确保前向和反向传播的正确性。
     - 在 chunk 反向执行完成后立即释放其 activations，显著降低峰值内存。
  2. **Profile-guided Non-uniform Chunking**（针对 hybrid 模型）：
     - softmax attention 的计算成本随上下文增长而增加（因需关注更长前缀），而 linear recurrence 更均匀。
     - StateFlow 通过离线 profiling 搜索最优的 chunk 划分策略（由参数 α 控制），在 equal-length 和 FLOP-balanced 之间权衡，实现运行时负载均衡。
  3. **State Transition Overlap & Grid Optimization**：
     - **Split and Overlap**：将 chunk 内部计算拆分为 `Pre` → `ST` → `Out` 阶段，重叠 state transition (ST) 与邻近的密集计算（如 projection），提升 GPU 利用率。
     - **Grid Optimization**：优化 ST kernel 的 launch grid size（通过调整 tile size $ B_v $），改善小 batch 下的 SM 覆盖率。

### **相比现有方法的优势**
| 方法 | Communication | Activation Memory | Balanced Runtime | Supports Linear Recurrence |
|------|---------------|-------------------|------------------|----------------------------|
| TeraPipe [17] | Low | High | Yes | ❌ |
| Seq1F1B [28] | Low | Low | Yes | ❌ |
| MEPipe [30] | Low | Low | ❌ | ❌ |
| SlimPipe [16] | High | Low | ❌ | ❌ |
| **StateFlow (Ours)** | **Low** | **Low** | **Yes** | ✅ |

- **首次将 SPP 成功扩展到 linear recurrence 和 hybrid 架构**，兼顾低通信开销、低内存、高吞吐和状态一致性。
- 在保持数学精确性的同时，实现高达 **2.22× 吞吐提升** 和 **2.45× 峰值内存降低**。

---

## **2. 核心实验方法和设置**

### **使用的模型与任务**
- **模型类型**：
  - **纯 recurrence 模型**：Gated DeltaNet (GDN) 和 Mamba-3。
  - **混合架构模型**：每 3–7 层 linear recurrence 接 1 层 softmax attention（recurrence-to-attention ratio = 3:1）。
- **模型规模**：3B、15B、32B 参数。
- **上下文长度**：最高达 **256K tokens**。
- **无特定下游任务**，聚焦于 **长上下文预训练效率**。

### **实验设置**
- **硬件平台**：
  - NVIDIA A100-SXM4-80GB GPUs。
  - 使用 8 / 16 / 32 张 GPU 分别训练 3B / 15B / 32B 模型。
- **软件栈**：
  - PyTorch 2.6.0, CUDA 12.4, Triton 3.5.1。
  - 基于 **Megatron Core r0.12.0** 和 **ms-swift 3.6.4** 实现。
- **并行配置**：
  - PP8/TP1（3B）、PP4/TP4（15B）、PP4/TP8（32B）。
  - 所有实验采用 **1F1B pipeline schedule**。
  - Microbatch size (MBS) = 1，Global Batch Size (GBS) ∈ {4, 8, 16}。
- **评估指标**：
  - **End-to-end throughput**（K tokens/s）
  - **Peak memory usage**（GB，PyTorch CUDA 最大分配）
  - 支持的最大 sequence length（是否 OOM）

### **基线方法**
- **Megatron-LM** 和 **Swift** 的原生 pipeline parallelism 实现。
- 对比项均为相同模型、recomputation、并行配置下的 baseline。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **(1) 纯 recurrence 模型（GDN / Mamba-3）**
- **吞吐提升**：**1.02× ~ 2.2×**，平均 **1.37×**。
- **峰值内存降低**：**1.31× ~ 2.45×**，平均 **1.63×**。
- **最大加速**：在 3B GDN 上达到 **2.2× speedup**。
- **内存节省最多**：在 3B Mamba-3 上实现 **2.45× peak memory reduction**。
- **无重计算场景**：仍可达 **1.79× speedup** 和 **2.54× memory reduction**。

#### **(2) 混合模型（hybrid）**
- **吞吐提升**：**1.02× ~ 2.22×**，平均 **1.35×**。
- **峰值内存降低**：**1.56× 平均，最高 2.44×**。
- **可行性扩展**：
  - 原生 pipeline 在 256K 上普遍 OOM。
  - **StateFlow 成功训练所有 3B/15B 模型至 256K，32B 至 128K**。

#### **(3) 消融实验与关键发现**
- **Chunk 数量 $ N $ 的影响**：
  - 吞吐随 $ N $ 先升后降，存在最优值（通常为 8 或 16）。
  - 过细 chunk 导致 kernel 效率下降，抵消 pipeline bubble 减少的好处。
- **Hybrid-aware 分区参数 $ \alpha $**：
  - 最优 $ \alpha $ 多在 **0.5 ~ 0.75** 之间，非简单 equal 或 FLOP-balanced。
  - 相比 equal-length 划分，可带来 **1.20× ~ 1.23× 加速**。
- **State Transition 优化效果**：
  - **GDN**：总加速 **1.34× ~ 1.58×**（随模型增大而提升）。
  - **Mamba-3**：overlap 区域加速 **1.14× ~ 1.51×**，**最多隐藏 84.8% 的 fused kernel 延迟**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **SPP 可有效应用于 linear recurrence 模型**：
   - 通过显式管理边界 state 和 gradient，StateFlow 实现了正确的依赖传递和 early activation release。
2. **chunk 级调度显著降低内存并提升吞吐**：
   - 将 pipeline 单位从 full-sequence microbatch 细化为 chunk，大幅减少 pipeline bubbles 和 peak memory。
3. **hybrid 模型需要定制化的 chunk 划分**：
   - 不能简单套用 uniform 或 FLOP-balanced 策略，必须通过 profiling 找到 runtime-optimal 分区。
4. **state transition 是性能瓶颈**：
   - 其低并行性限制 GPU 利用率，但可通过 **overlap** 和 **grid tuning** 有效缓解。

### **方法的局限性**
- **依赖精细调优**：最优 $ N $ 和 $ \alpha $ 需通过 profiling 确定，增加部署成本。
- **仅适用于支持 chunked execution 的 linear recurrence 实现**：要求模型能以 chunk 为单位进行 state transition。
- **当前实现基于特定框架**（Megatron/Swift），通用性有待验证。

### **未来工作方向**
- **自动化 tuning**：开发轻量级 profiler 或预测模型，避免 exhaustive search。
- **与其他并行策略更深集成**：如结合 context parallelism 或 zero-bubble pipeline。
- **扩展到更多 hybrid 架构**：如 Mamba + MoE、SSM + Routing 等复杂组合。
- **支持 inference 场景**：探索 StateFlow 在长文本生成中的应用。

---

> **总结**：StateFlow 是首个成功将 SPP 应用于 linear recurrence 和 hybrid 模型的系统，通过 state-aware 调度、profile-guided 分区和 kernel-level 优化，在不牺牲精度的前提下，实现了显著的训练加速和内存压缩，推动了超长上下文大模型的可训练性边界。

</details>

---

### 7. [MiCoPro: End-to-End Mixed Precision HW/SW Co-design with HW-aware Proxy Model](https://arxiv.org/abs/2608.06916)

**Authors**: Zijun Jiang, Yangdi Lyu  
**Category**: cs.LG  
**Published**: 2026-08-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.06916v1  

#### Abstract
Quantized Neural Networks~(QNN) with low-bitwidth data have proven promising in efficient storage and computation on edge devices. To mitigate accuracy degradation while maximizing speedup, layer-wise mixed-precision quantization~(MPQ) becomes a popular solution. However, existing algorithms for exp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MiCoPro: End-to-End Mixed Precision HW/SW Co-design with HW-aware Proxy Model

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**边缘设备上神经网络混合精度量化（Mixed Precision Quantization, MPQ）探索与部署中的三大挑战**：

1. **搜索效率低**：传统方法在高维 MPQ 搜索空间中难以高效找到最优配置。
2. **硬件感知不足**：现有框架多依赖理论指标如 Bit Operations (BOPs)，无法准确反映真实硬件上的端到端延迟。
3. **缺乏端到端部署支持**：多数研究止步于模型设计，缺少从 PyTorch 到裸机 C 代码的完整部署流程。

---

### 提出的新方法与创新思路

#### ✅ MiCo 框架（基础）
- **RF+NCS 搜索算法**：
  - 使用 **Random Forest (RF)** 构建准确性预测器，替代耗时的 QAT/PTQ 验证。
  - 引入 **Layer-wise Orthogonal Initial Sampling** 和 **Near-Constraint Sampling (NCS)** 提升采样效率。
- 支持灵活的 bitwidth 组合（如 W1A1~W8A8），适用于 PTQ 与 QAT 场景。

#### ✅ MiCoPro 框架（增强版）
- **Hardware-Aware Proxy (HAP) 模型**：
  - 提出 **Composite Bit Operations (CBOPs)** 特征体系：  
    `BMACs = max(bw, ba) × MACs`，`ALoads`, `WLoads`，更精准建模计算与内存开销。
  - 引入 **Min-Max Calibration** 对不同网络进行校准，提升跨模型泛化能力。
  - 支持 **Inter-Hardware Transfer Learning**，大幅减少新硬件平台的 profiling 时间。
- **端到端部署流水线**：
  - 支持将 PyTorch 模型自动转换为可在 **BitFusion 加速器** 和 **SIMD 扩展 RISC-V CPU** 上运行的裸金属 C 程序。

---

### 相比现有方法的优势

| 方面 | MiCoPro 优势 |
|------|--------------|
| **搜索效率** | RF+NCS 显著优于 RL/BO/NLP 方法，在更少评估次数下达到更高精度 |
| **硬件感知能力** | HAP 模型比 BOPs 更准确预测实际延迟（R² 提升至 >0.8） |
| **部署灵活性** | 支持多平台部署（加速器 + CPU），提供完整软件栈（graph converter + kernel lib + codegen） |
| **可迁移性** | 转移学习使新硬件建模时间从数十小时降至 <2 小时 |

---

## 2. 核心实验方法和设置

### 数据集与模型
使用多样化的模型覆盖多种任务和架构：

| 类别 | 模型 | 数据集 |
|------|------|--------|
| 视觉 | CNN4, LeNet5, VGG7, ResNet-18/34, SqueezeNet, ViT-B-32 | CIFAR-10/100, MNIST, ImageNet |
| 音频 | DS-CNN | Speech Commands V2 |
| 语言 | TinyLLaMa-1M / -7M | TinyStories |

> 共测试 **10 个模型**，层数从 4 到 43 层不等，验证方法通用性。

---

### 实验设置

- **搜索预算控制**：
  - PTQ 实验：48 次评估（16 初始 + 32 搜索）
  - QAT 实验：32 次评估（16 初始 + 16 搜索）
- **约束条件**：
  - 以 INT8 模型为基础，设定 BOPs 约束比例（如 0.6×, 0.5×）
- **评估方式**：
  - 多次随机种子重复实验取平均值（PTQ: 5 seeds, QAT: 3 seeds）

---

### 基线方法对比

| 方法 | 类型 | 来源 |
|------|------|------|
| w-based (Edge-MPQ) | NLP 分析 | [10] |
| HAQ | Reinforcement Learning | [16] |
| BOMP | Bayesian Optimization | [15] |
| HAWQ-V3 | ILP + Hessian 敏感度分析 | [8] |
| Uniform / Empirical Policies | 启发式规则（首尾层保持高精度） | — |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 平台 | 模型 | 性能增益 | 准确率损失 |
|------|------|----------|------------|
| BitFusion | VGG7 | 延迟降低 **40%**（Cycle Ratio 0.60×） | <0.5% ↓ |
| BitFusion | ResNet-18 | 延迟降低 **42%**（Cycle Ratio 0.58×） | <0.2% ↓ |
| VexiiMiCo (High) | CNN4 | 延迟降低 **22%**（Cycle Ratio 0.78×） | 可忽略 |
| VexiiMiCo (High) | LeNet5 | 延迟降低 **17%**（Cycle Ratio 0.83×） | <3% ↓ |

> ✅ 最高达 **40% 延迟降低**，同时保持 **<3% 准确率下降**

---

### 与基线方法对比结果

#### 📊 PTQ 搜索表现（Table IV）
- 在大多数模型上，MiCo 在相同预算下取得 **最高准确率**。
- 在 SqueezeNet 上，MiCo 达到 68.22%，优于 HAQ（67.58%），且收敛速度快 **1.84×**。

#### 📊 QAT 搜索表现（Table V）
- 在低比特（1/2/4/8-bit）空间中，MiCo 表现稳健。
- TinyLLaMa-7M 在 0.6×BOPs 下达到 **59.12% 准确率**，显著优于 w-based（51.51%）和 BO（43.30%）。

#### 📊 HAP 模型预测精度（Table IX–X）
| 特征 | 平均 MAPE | 平均 R² |
|------|---------|--------|
| BOPs | 12.5% | 0.45 |
| CBOPs | 6.5% | 0.80 |
| Full Features + RF/XGB | **3.8%** | **0.87** |

> ✅ HAP 模型显著优于传统 BOPs 指标，尤其在 CPU 平台上避免负相关（BOPs 在某些情况下 R² < 0）

---

### 消融实验结果（Table VII）

在 SqueezeNet 上进行 PTQ 搜索（0.6×BOPs, 32 次预算）：

| 方法 | 准确率 (%) | 提升 |
|------|-----------|------|
| Random Forest Only | 66.95 | — |
| + Orthogonal Initial Sampling | 67.13 | +0.18 |
| + Near-Constraint Sampling | **67.92** | **+0.79** |

> 结果表明：**NCS 是性能提升的关键因素**，有效聚焦于约束边界附近的高质量方案。

---

## 4. 关键结论和发现

### 主要发现

1. **BOPs 不足以指导真实硬件优化**：
   - 图 2 显示 BOPs 与实际延迟相关性弱（R² ≈ 0.35~0.55），尤其在 CPU 上甚至出现负相关。
   - 必须引入硬件感知代理模型（HAP）才能实现真正的加速。

2. **CBOPs 是有效的硬件感知特征**：
   - 分离 `BMACs`, `ALoads`, `WLoads` 能更好捕捉并行性和内存带宽影响。
   - 线性模型 + CBOPs 即可实现良好预测，适合快速搜索阶段。

3. **MiCoPro 实现了高效的端到端优化闭环**：
   - 从模型搜索 → 精度预测 → 硬件部署全流程自动化。
   - 支持多种硬件目标（加速器 vs CPU），具备强移植性。

4. **Transfer Learning 显著降低部署门槛**：
   - 仅需 5% 新硬件 profiling 数据 + transfer learning，即可获得接近全量训练的代理模型性能。
   - 将新平台适配时间从 **27 小时缩短至 <2 小时**。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖一次性 profiling** | 虽然 transfer learning 缓解了问题，但仍需初始 RTL 或仿真环境支持 |
| **未覆盖所有硬件类型** | 当前验证集中在 BitFusion 和 RISC-V SIMD，对 GPU 或 NPU 支持待扩展 |
| **QAT 成本仍较高** | 尽管采用 short-QAT，但对于超大模型仍存在训练瓶颈 |

---

### 未来工作方向

1. **扩展至动态稀疏 + 混合精度联合优化**
2. **支持更多开源硬件平台（如 GAP8, PULP）**
3. **构建公共 HAP 模型仓库，促进社区共享**
4. **集成编译器级优化（如 TVM 耦合）进一步压缩部署开销**

---

> 🔚 **总结一句话**：  
> **MiCoPro 通过“硬件感知代理模型 + 高效搜索策略 + 端到端部署”三位一体设计，实现了在真实边缘硬件上高达 40% 延迟降低的同时几乎无损精度，推动了 TinyML 中 MPQ 技术的实用化进程。**

</details>

---

### 8. [Beyond Post-Hoc Temperature Scaling: Bilevel Optimization for LLM Calibration](https://arxiv.org/abs/2608.07419)

**Authors**: Ruochen Jin, Zhanliang Wang, Zongyu Dai, Jiancong Xiao, Bojian Hou  
**Category**: cs.LG  
**Published**: 2026-08-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.07419v1  

#### Abstract
Preference alignment often makes large language models (LLMs) overconfident and poorly calibrated. Traditional post-hoc temperature scaling is inherently domain-dependent: a temperature fitted on one domain does not generalize across domains. This motivates us to modify model parameters during train...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Post-Hoc Temperature Scaling: Bilevel Optimization for LLM Calibration

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在经过 **preference alignment**（如 RLHF 或 DPO）后常表现出严重的 **overconfidence** 和 **poor calibration**。传统的 **post-hoc Temperature Scaling (TS)** 虽然能改善校准，但其优化的温度参数高度依赖于特定数据集，难以泛化到其他领域（即 **dataset-dependent**），限制了其在多样化任务中的实用性。

### 提出的新方法与新思路
作者提出了一种训练时（training-time）的校准框架——**CALM (Calibration for Large Models via Bilevel Optimization)**，其核心思想是将校准目标融入训练过程本身，而非仅作为推理阶段的后处理步骤。

#### 主要创新点：
- **从 post-hoc 到 training-time 的范式转变**  
  不再依赖固定模型后的温度拟合，而是通过联合优化模型参数和损失超参数，在训练过程中学习一个更具泛化能力的“类温度”调整机制。

- **基于熵最大化的上层目标（Upper-Level Objective）**  
  采用 **entropy maximization** 作为上层优化目标，直接抑制预测分布的过度集中（overconfidence），而无需额外的真实标签（hard labels）。这使得方法适用于开放生成任务。

- **双层优化（Bilevel Optimization）框架设计**  
  - **下层（Lower Level）**：在给定超参数 $\alpha$ 下最小化标准交叉熵损失，确保模型判别能力。
  - **上层（Upper Level）**：在验证输入上最大化预测分布的熵，以提升校准度。
  形式化为：
  $$
  \min_\alpha \sum_{x_i \in S_v} H(f_{\alpha,\theta(\alpha)}(\cdot|x_i)) \quad \text{s.t.} \quad \theta(\alpha) \in \arg\min_\theta \mathcal{L}_{tr}(\theta; \alpha)
  $$

- **高效的首阶近似算法（First-Order Approximation）**  
  针对双层优化中计算超梯度（hypergradient）所需的二阶梯度和 Hessian 逆操作在 LLM 上不可行的问题，借鉴 **BOME (Bilevel Optimization Made Easy)** 方法，使用插件残差（plug-in residual）和联合梯度更新避免显式高阶计算，使该框架可扩展至大规模模型。

### 相比现有方法的优势
| 方法 | 是否训练时 | 泛化性 | 是否需标签 | 是否保留性能 |
|------|------------|--------|-------------|----------------|
| TS (Temperature Scaling) | ❌ 后处理 | 差（OOD 下失效） | ✅ 是 | ✅ 是 |
| Label Smoothing | ✅ 是 | 中等 | ✅ 是 | ⚠️ 可能损害性能 |
| CFT (Calibration-aware FT) | ✅ 是 | 中等 | ✅ 是 | ✅ 是 |
| **CALM (Ours)** | ✅ 是 | ✅ 强（尤其 OOD） | ❌ 否（仅需输入） | ✅ 是 |

> ✅ **优势总结**：CALM 在保持任务性能的同时，显著提升了 **out-of-domain (OOD)** 场景下的校准鲁棒性，且不依赖真实标签进行上层优化，适用范围更广。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Multiple-Choice QA (MCQA)**：
  - 包含 MMLU、MedMCQA、OpenBookQA、ARC-Challenge 四个子集。
  - **In-Domain (ID)**：训练与测试均用 MCQA 数据。
  - **Out-of-Domain (OOD)**：训练用 Alpaca 数据集，测试用 MCQA（零样本迁移）。

- **Open-Ended Generative QA**：
  - 使用 **PopQA** 和 **TriviaQA**。
  - 设置为跨域生成任务：在一个数据集上训练，在另一个上测试，输出格式一致，仅问题分布变化。

### 实验设置
- **模型**：在四个主流开源 LLM 上评估：
  - Llama-3.1-Tulu-8B (DPO)
  - Vicuna-7B (RLHF)
  - OLMo-2-7B (DPO)
  - Mistral-7B (DPO)

- **微调方式**：全部使用 **QLoRA**（rank=64, α=32）进行参数高效微调。

- **训练配置**：
  - Batch size: 16
  - Epochs: 5
  - Optimizer: AdamW
  - Precision: float16

### 评估指标
| 任务类型 | 指标 | 描述 |
|--------|------|------|
| Multiple-Choice | **conf-ECE**, **cw-ECE** | 衡量最大预测概率与实际准确率的一致性；越低越好 |
| Open-Ended Generation | **Sem-ECE** (Wang et al., 2026) | 基于语义聚类的置信度估计，适用于自由文本输出 |
| 所有任务 | **Accuracy** | 任务性能基准，越高越好 |

> 注：所有校准指标均为 **lower-is-better**。

### 基线方法对比
| 类型 | 方法 | 简介 |
|------|------|------|
| 原始模型 | DPO/RLHF baseline | 未经校准的对齐模型 |
| 后处理方法 | **Temperature Scaling (TS)** | 标准 post-hoc 温度缩放 |
| 训练时方法 | **Label Smoothing (LS)** | 使用软标签防止过拟合 |
| | **Calibration-aware Fine-Tuning (CFT)** | Xiao et al. (2025a) 提出的校准感知微调 |
| | **Regularization** | 将熵作为单层正则项（消融） |
| | **Iterate** | 交替更新任务与校准目标，无耦合机制（消融） |

---

## 3. 主要实验结果和性能指标

### 多选问答（MCQA）OOD 设置下的关键结果（Table 1）
| Model | Method | conf-ECE ↓ | Accuracy ↑ |
|-------|--------|------------|-----------|
| Llama-3.1 | DPO/RLHF | 0.1784 | 66.80% |
| | **CALM (Ours)** | **0.1050** | 64.45% |
| Vicuna-7B | DPO/RLHF | 0.0581 | 44.70% |
| | **CALM (Ours)** | **0.0380** | 44.95% |
| Mistral-7B | DPO/RLHF | 0.3351 | 59.80% |
| | **CALM (Ours)** | **0.0822** | 51.60% |

> ✅ **结论**：CALM 在三个模型上取得最佳 conf-ECE，且在 Vicuna 和 Llama 上优于所有基线。

### 开放生成任务（Generative QA）跨域表现（Table 3）
| Method | Llama-3.1 | Vicuna-7B | OLMo-2-7B | Mistral-7B | **Mean Sem-ECE ↓** |
|--------|-----------|-----------|-----------|------------|------------------|
| DPO/RLHF | 0.0512 | 0.1421 | 0.0716 | 0.0736 | 0.0846 |
| CFT (best-α) | 0.0754 | 0.1215 | 0.1597 | 0.0870 | 0.1109 |
| **CALM (Ours)** | **0.0490** | **0.1105** | 0.0724 | **0.0366** | **0.0671** |

> ✅ **结论**：CALM 在平均 Sem-ECE 上大幅领先（0.0671 vs. 0.0846），尤其在 Mistral-7B 上实现近 **50% 的相对改进**。

### 消融实验结果
| 消融变体 | 特点 | 性能表现 |
|---------|------|----------|
| **Regularization** | 熵作为固定权重的单层惩罚项 | 在某些模型上有效，但在 Mistral 上 accuracy 暴跌至 44.2%，说明固定权重无法自适应调节 |
| **Iterate** | 交替更新任务与校准目标，无隐式约束 | 经常导致 accuracy 崩溃（如 Mistral 从 59.8% → 25.8%） |
| **Scalar vs Vector**（附录 D.2） | 将 per-vocabulary 参数退化为全局 scalar | conf-ECE 从 0.0822 升至 0.3140，表明向量参数化对精细控制至关重要 |

> 🔍 **关键发现**：**双层结构中的耦合机制（implicit/residual constraint）是成功的关键**，单纯加熵正则或交替训练都无法稳定兼顾校准与性能。

---

## 4. 关键结论和发现

### 主要发现
1. **传统 TS 的泛化瓶颈被突破**  
   CALM 通过将“温度”推广为可学习的 per-vocabulary 超参数，并在训练中联合优化，实现了更强的 **cross-domain calibration transfer 能力**。

2. **熵最大化 + 双层约束 = 最佳权衡**  
   上层熵最大化有效缓解 overconfidence，而下层任务最优性约束保证了判别性能不崩溃，二者结合实现了 **calibration-utility trade-off 的帕累托前沿**。

3. **无需真实标签即可优化校准**  
   上层目标仅依赖输入 $x$ 的预测分布熵，**不需要 ground truth label**，使其天然适用于开放生成等缺乏明确类别结构的任务。

4. **在 OOD 场景下优势最明显**  
   在 out-of-domain 设置中，CALM 显著优于 TS、CFT 等方法，证明其具备更强的分布外鲁棒性。

### 方法的局限性
- **训练成本较高**：由于双层结构需要维护 surrogate model 并执行 inner loop，训练时间约为单层方法的数倍（见 Table 10：CALM ~2小时 vs Regularization ~35分钟）。
- **对随机种子敏感**：在 Llama 和 Mistral 上存在较大运行间方差（附录 Table 11），提示可能需要更多正则或集成策略增强稳定性。
- **目前仅用于监督微调场景**：尚未整合进 RLHF/DPO 等 alignment 流程内部。

### 未来工作方向
- 将 CALM 整合进 **alignment pipeline**（如 DPO 或 RLHF）中，实现端到端的校准对齐。
- 探索更轻量的双层优化变体，进一步降低训练开销。
- 扩展至多模态模型的不确定性校准。
- 结合模型自身生成的 reasoning trace 进行动态置信度建模。

---

> 📌 **一句话总结**：  
> CALM 提出了一种新颖的 **training-time bilevel optimization 框架**，通过 **entropy maximization** 和 **first-order approximation** 实现了对 LLM 的高效、泛化性强的校准，在多种任务和模型上显著优于 post-hoc 和现有训练时方法，尤其在 **out-of-domain 场景下展现出卓越的鲁棒性**。

</details>

---

### 9. [Fast LapSum: Exact Differentiable Top-k at Million Scale](https://arxiv.org/abs/2608.06912)

**Authors**: {\L}ukasz Struski, Joanna Wojciechowicz, Jakub Antczak, Marcin Mazur, Kamil Ksi\k{a}\.zek, Jacek Tabor  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.06912v1  

#### Abstract
The top-$k$ operation is a fundamental building block of modern sparse computation, enabling token routing, expert activation, memory selection, and attention pruning. Yet standard hard top-$k$ blocks gradients, while existing continuous (soft) relaxations remain too costly for large-scale models. W...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Fast LapSum: Exact Differentiable Top-k at Million Scale 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **hard top-k** 操作在深度学习中广泛用于稀疏计算（如 MoE 路由、注意力剪枝），但它不可微分，导致梯度无法回传，阻碍端到端训练。现有的 **soft top-k** 松弛方法虽然可微，但通常存在以下问题：
- 计算成本高（如 $O(n^2)$ 或迭代求解）
- 无法精确控制激活数量（即 $\sum p_i \neq k$）
- 在百万级规模下不实用

本文旨在解决：**如何实现一个既可微、又精确满足预算约束（$\sum p_i = k$）、且能在百万级输入上高效运行的 soft top-k 操作符**。

---

### 提出了什么新方法或新思路
作者提出了 **Fast LapSum**，一种基于 **LapSum** 算子的高效 GPU 实现，其核心思想是：

- **LapSum 原理**：将每个 score $r_i$ 加上一个 Laplace 分布的噪声 $e_i \sim \text{Laplace}(0, t)$，然后通过一个阈值 $b$ 判断其是否属于 top-k。最终的概率为：
  $$
  p_i = \Phi\left(\frac{r_i - b}{t}\right), \quad \text{其中 } \Phi \text{ 是 Laplace CDF}
  $$
  阈值 $b$ 被调整以确保 $\sum p_i = k$。

- **Fast LapSum 的加速机制**：
  1. **排序后前缀/后缀扫描**：对 scores 排序后，利用指数扫描（exponential scan）构造“预算曲线”，从而将阈值 $b$ 的求解转化为一次索引查找 + 闭式根求解。
  2. **分析式 Vector-Jacobian Product (VJP)**：反向传播无需自动微分框架处理复杂结构，而是直接使用解析梯度公式。
  3. **概率性 bracketing（百万级优化）**：对于超大规模输入（如 $n=10^7$），仅对 scores 的“中间模糊区域”进行排序，其余部分用尾部求和近似，显著降低排序开销。

---

### 相比现有方法的优势
| 方法 | 是否精确 $\sum p_i = k$ | 时间复杂度 | 可扩展性 | 备注 |
|------|--------------------------|------------|-----------|------|
| **Hard top-k** | ✅ | $O(n)$ | 高 | 不可微 |
| **DFTopK [26]** | ❌ | $O(n)$ | 高 | 固定阈值，质量差 |
| **NeuralSort / SoftSort** | ✅ | $O(n^2)$ | 低 | 易 OOM |
| **Sinkhorn-OT** | ✅ | $O(n^2)$ | 低 | 迭代慢 |
| **Fast LapSum (ours)** | ✅ | $O(n)$（期望） | 极高 | **唯一同时满足精确性和高效性的方法** |

> ✅ **Fast LapSum 是首个在保持 $\sum p_i = k$ 精确成立的同时，达到线性时间复杂度并支持百万级输入的可微 top-k 方法**。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **DIV2K**：用于稀疏对抗攻击实验，图像分辨率为原生分辨率（约 2–3 百万像素）。
- **自定义图像编码任务**：在 1536×2048 的图像上构建多尺度字典编码器，涉及超过 400 万个 super-pixel。

---

### 实验设置和评估指标

#### （1）稀疏对抗攻击（Sparse Adversarial Attack）
- **目标**：在不超过 $B$ 个“完全修改像素”等效扰动下，使 ConvNeXt-V2 分类错误。
- **扰动形式**：
  $$
  x' = (1 - \text{LapSum}_k(r)) \odot I + \text{LapSum}_k(r) \odot \sigma(z)
  $$
  其中 $r$ 是选择 logits，$z$ 是颜色场。
- **评估指标**：
  - **总注入变化 $B = \sum_i |x'_i - I_i|$**（单位：fully-modified-pixel equivalents）
  - 成功率（Fool%）
  - 中位数 $B$ 达到成功攻击

#### （2）可微图像编码器（Differentiable Image Coder）
- **架构**：金字塔结构，每层从过完备字典中选择 $k$ 个原子进行重建。
- **损失函数**：重建 MSE + 硬比特预算约束。
- **关键机制**：使用 **straight-through estimator (STE)**，前向用 hard mask，反向用 LapSum 梯度。

---

### 基线方法对比
| 方法 | 类型 | 是否可微 | 是否精确 $k$ |
|------|------|----------|--------------|
| **o-zero [4]** | 可微 | ✅ | ❌ ($\ell_0$ 正则化) |
| **Sparse-PGD [25]** | 投影 GD | ✅ | ✅ |
| **PGD-Lo [5]** | 投影 GD | ✅ | ✅ |
| **SparseFool [13]** | 启发式 | ❌ | ✅ |
| **DFTopK [26]** | 可微 | ✅ | ❌ |
| **Fast LapSum (ours)** | 可微 | ✅ | ✅ |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）运行时效率（GPU: RTX 5060 Laptop）
| 输入大小 $n$ | Fast LapSum 时间（ms） |
|---------------|------------------------|
| $10^6$        | 0.41                   |
| $10^7$        | 1.15                   |
| $10^8$        | 5.23                   |

> ✅ **在 $10^8$ 规模下仍低于 6ms，远优于其他方法**。

#### （2）与 DFTopK 对比（Table 13）
| $n$ | DFTopK (ms) | Fast LapSum (ms) | 加速比 |
|-----|-------------|------------------|-------|
| $10^5$ | 0.713       | 0.700            | ~1×   |
| $10^6$ | 9.307       | 1.941            | **4.8×** |
| $3\times10^6$ | 15.25     | 1.997            | **7.6×** |

> ⚠️ DFTopK 在大 $n$ 下因两次 `k-th` 查询而变慢，Fast LapSum 保持线性增长。

#### （3）预算精度（Table 3）
| 方法 | $\|\sum p_i - k\| / k$ |
|------|------------------------|
| **Fast LapSum** | $<10^{-5}$ |
| **DFTopK** | ~2.46（即实际激活数约为 $3.46k$） |
| **NeuralSort / SoftSort** | 精确但 OOM（$n > 10^5$） |

> ✅ **Fast LapSum 是唯一在百万级上既快又精确的方法**。

#### （4）稀疏对抗攻击效果（Table 4）
| 方法 | 成功率 | 中位数 $B$ |
|------|--------|------------|
| **Fast LapSum (ours)** | 100% | **199.8** |
| o-zero | 98% | 3101.0 |
| Sparse-PGD | 100% | 16134.2 |
| PGD-Lo | 100% | 74553.3 |

> ✅ **我们的方法所需扰动仅为最强 baseline 的 1/15 到 1/275**。

#### （5）消融实验（Table 7）
| 变体 | 时间（ms） | 相对加速 |
|------|-----------|---------|
| Baseline（二分法） | 7.29 | 1.0× |
| 单线程扫描 | 0.442 | 16.5× |
| 并行块扫描 | 0.269 | 27.1× |
| 融合 CDF 发射 | **0.107** | **68.1×** |

> ✅ 所有优化叠加带来近两个数量级的加速。

---

## 4. 关键结论和发现

### 主要发现
1. **可微 top-k 不必牺牲效率或精度**：Fast LapSum 首次证明可以在百万级输入上实现 **精确预算 + 完全可微 + 线性时间** 的 soft top-k。
2. **排序不是瓶颈**：通过前缀扫描和 bracketing，避免了传统方法中反复求和或迭代搜索的开销。
3. **真实任务可用**：在百万像素对抗攻击和图像编码中，Fast LapSum 成为训练循环中的“轻量组件”，而非计算瓶颈。
4. **DFTopK 的 trade-off 是不必要的**：其放松 $\sum p_i = k$ 的做法并非实现线性时间所必需，Fast LapSum 在保持精确性的同时更快。

---

### 方法的局限性
1. **依赖 Laplace 核**：当前构造依赖于 Laplace CDF 的 piecewise-exponential 性质，难以直接推广到其他平滑核（如 Gaussian）。
2. **期望线性时间，非最坏情况**：bracketing 基于概率保证，在极端分布下可能需要多次扩展区间，退化为完整排序。
3. **GPU 实现特定优化**：高度依赖 CUDA 的 block 扫描和内存访问模式，移植到其他硬件需重新工程化。

---

### 未来工作方向
- 将 Fast LapSum 应用于更广泛的稀疏系统，如：
  - **MoE 路由器的端到端训练**
  - **大规模检索系统的可微排序**
  - **动态稀疏神经网络训练**
- 探索其他可微核函数下的高效实现。
- 结合熵编码，构建完整的可微压缩系统。

---

> 🔚 **总结**：Fast LapSum 将可微 top-k 从理论工具转变为可在百万级输入上高效运行的 **GPU 原语（primitive）**，为稀疏模型的端到端优化打开了新大门。

</details>

---

### 10. [PTQ4SNN: Membrane-Aware Post-Training Quantization for Spiking Neural Networks](https://arxiv.org/abs/2608.07066)

**Authors**: Hui Xie, Tong Shi, Haotong Qin, Aishan Liu, Xiaode Liu, Jinyang Guo  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.07066v1  

#### Abstract
Spiking neural networks (SNNs) enable sparse and event-driven computation, but their low-bit deployment remains incomplete because recurrent membrane states are commonly retained in floating point even after weight quantization. Quantizing these states is challenging because their distributions diff...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PTQ4SNN: Membrane-Aware Post-Training Quantization for Spiking Neural Networks**

---

## **1. 主要贡献和创新点**

### **解决的问题**
Spiking Neural Networks（SNNs）虽然具备稀疏性和事件驱动计算的优势，但在低比特部署中仍面临挑战：
- **膜电位（membrane potential）通常以浮点形式保留**，即使权重已量化，导致状态存储和数据移动开销依然巨大。
- 膜电位分布与前层权重差异大，直接复用权重尺度会导致严重截断（clipping）。
- 膜电位对阈值附近的微小扰动敏感，量化误差可能改变放电决策并随时间累积。
- 不同通道的放电活跃度和量化敏感性不同，统一比特分配效率低下。

现有方法如 FlowQ 仅支持层级（layer-wise）的幂次二比例耦合，且采用单一比特宽度，无法适应通道级异质性。

---

### **提出的新方法与创新思路**
本文提出 **PTQ4SNN**，一种**面向膜电位的后训练量化（Post-Training Quantization, PTQ）框架**，无需重新训练即可联合量化权重和递归膜状态。

#### **核心创新点：**

1. ✅ **Channel-wise Unified Scale Bridge（统一尺度桥）**
   - 构建权重与膜电位之间的硬件友好关系：  
     $$
     S_{\text{mem},c} = S_{w,c} \cdot 2^{k_c}
     $$
   - 允许每个通道 $c$ 独立调整指数 $k_c$，使膜量化范围适配其实际分布。
   - 利用 **power-of-two shift** 实现无乘法器的高效转换，保持整数运算兼容性。
   - 解决了权重与膜电位分布不匹配问题，同时避免独立 scale 带来的额外转换开销。

2. ✅ **Mixed-Precision Bit Allocation（混合精度比特分配，MPBA）**
   - 在平均比特预算下（如 ~4-bit），为不同膜通道动态分配 2/4/8-bit 精度。
   - 分配依据两个关键因素：
     - **放电率（firing rate）**：高活跃通道需更高精度。
     - **量化敏感性（quantization sensitivity）**：通过梯度相关性衡量扰动影响。
   - 使用加权组合得分 $a_c = \beta r_c + (1-\beta)g_c$ 进行排序，并基于预算设定阈值进行比特分配。
   - 支持细粒度资源优化，在相同平均比特下提升精度。

3. ✅ **通用性与可扩展性**
   - 框架基于 **projection-LIF pairs** 组织模块，适用于传统卷积 SNN 和 spike-driven Transformers（如 SDT、Meta-SpikeFormer）。
   - 仅需少量校准样本（calibration set），不更新主干参数，真正实现“plug-and-play”式 PTQ。

---

### **相比现有方法的优势**
| 方法 | 是否量化膜？ | 尺度策略 | 比特分配 | 是否需重训练 |
|------|--------------|-----------|------------|----------------|
| BRECQ/GPTQ | ❌ | 复用权重 scale | 固定 M32 | ❌ |
| FlowQ | ✅ | 层级 power-of-two | 统一 M4 | ❌ |
| **PTQ4SNN** | ✅ | **通道级 power-of-two shift** | **混合精度（2/4/8-bit）** | ❌ |

> ✔️ 更精细的尺度控制 + 更高效的比特利用 → 显著减少精度损失。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 任务 | 数据集 | 输入形式 | 时间步 T |
|------|--------|----------|---------|
| 静态图像分类 | ImageNet-1K, CIFAR-10/100 | 图像重复或脉冲编码 | T=4 |
| 事件流分类 | CIFAR10-DVS | 动态事件流 | T=10 |
| 语义分割 | Pascal VOC2012 | 图像序列输入 | T=4 |

---

### **模型架构**
- **Spike-driven Transformers**:  
  - SDT-2-256, SDT-8-768  
  - Meta-SpikeFormer-8-512
- **传统 SNN**:  
  - SEW-ResNet18（用于 CIFAR 和 ImageNet）
- **分割模型**:  
  - 基于 SDT 的 FPN 结构

---

### **评估指标**
| 任务 | 主要指标 |
|------|----------|
| 分类任务 | Acc@1 (%) |
| 语义分割 | mIoU, aAcc, mAcc |
| 性能下降 | 相对于浮点基准的 Δ（百分点） |

---

### **基线方法对比**
| 类型 | 方法 | 描述 |
|------|------|------|
| 权重量化基线 | BRECQ, GPTQ | W4/M32：仅量化权重，膜保持 float32；W4/M4 版本复用权重 scale |
| SNN专用量化 | FlowQ | W4/M4，使用 layer-wise power-of-two scale coupling，固定 M4 |
| 本文方法 | **PTQ4SNN** | W4/M4，channel-wise scale bridge + MPBA，平均 ~4-bit 膜精度 |

> 所有方法均在**相同预训练 checkpoint** 和 **相同校准集** 上测试，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### **ImageNet-1K 分类结果（T=4）**
| Backbone | 方法 | W4/M32 | W4/M4 | Δ from FP |
|---------|-------|--------|--------|-----------|
| SDT-8-768 | Full Precision | 75.90 | 75.90 | — |
|  | BRECQ (reuse) | 74.12 | 64.84 | ↓11.06 |
|  | GPTQ (reuse) | 74.11 | 65.34 | ↓10.56 |
|  | FlowQ | — | 63.74 | ↓12.16 |
|  | **PTQ4SNN** | — | **75.16** | **↓0.74** ✅ |

| Meta-SpikeFormer | Full Precision | 78.88 | 78.88 | — |
|  | BRECQ | 74.76 | 68.66 | ↓10.22 |
|  | GPTQ | 77.66 | 75.31 | ↓3.57 |
|  | FlowQ | — | 63.54 | ↓15.34 |
|  | **PTQ4SNN** | — | **78.50** | **↓0.38** ✅ |

> 💡 **结论**：PTQ4SNN 在 W4/M4 下几乎无损，显著优于其他方法（尤其是 FlowQ 和复用策略）。

---

#### **事件流分类（CIFAR10-DVS, T=10）**
| 方法 | W4/M32 | W4/M4 | Δ from FP |
|------|--------|--------|-----------|
| Full Precision | 71.80 | 71.80 | — |
| BRECQ | 69.50 | 61.70 | ↓10.10 |
| GPTQ | 70.20 | 61.20 | ↓10.60 |
| FlowQ | — | 67.30 | ↓4.50 |
| **PTQ4SNN** | — | **70.80** | **↓1.00** ✅ |

> ⚡️ 即使在高度动态的事件流上，PTQ4SNN 也能有效维持时序状态稳定性。

---

#### **语义分割（Pascal VOC2012）**
| 方法 | mIoU | Δ from FP |
|------|------|-----------|
| Full Precision | 73.41 | — |
| BRECQ (reuse) | 67.63 | ↓5.78 |
| GPTQ (reuse) | 70.12 | ↓3.29 |
| FlowQ | 60.95 | ↓12.46 |
| **PTQ4SNN** | **72.49** | **↓0.92** ✅ |

> 🎯 表明该方法不仅适用于分类头，也成功迁移到密集预测任务。

---

### **消融实验结果**

#### **Ablation 1: Unified Scale Bridge 效果（SEW-ResNet18/CIFAR-100）**
| Scale Construction | Acc@1 | Drop | Saturation (%) |
|--------------------|-------|------|----------------|
| Reuse (直接复用) | 72.53 | ↓2.74 | 56.50 |
| Independent Observer | 74.02 | ↓1.25 | 3.58 |
| **Unified Bridge** | **74.19** | **↓1.08** | **5.31** |

> 🔍 统一尺度桥在避免饱和的同时，保持 shift 兼容性，性能优于独立 observer。

---

#### **Ablation 2: Mixed-Precision Bit Allocation（MPBA）增益**
| Dataset | Acc@1 (w/o MPBA) | Acc@1 (w/ MPBA) | Gain |
|--------|-------------------|------------------|------|
| CIFAR-10 | 92.92 | 93.36 | +0.44 |
| ImageNet-1K | 59.10 | 59.62 | +0.52 |

> 📈 MPBA 在相同平均比特预算下提升了精度，验证了通道级差异化分配的有效性。

---

#### **硬件成本估算（SEW-ResNet18 on 8×16-PE model）**
| 模式 | 平均比特 | 非茎部比特 | 总状态 (MiB) | 能耗比 (vs M32) |
|------|----------|-------------|---------------|----------------|
| M32 | 32.00 | 32.00 | 30.293 | 1.000 |
| Uniform M4 | 4.00 | 4.00 | 0.387 | 0.174 |
| **MPBA** | 4.002 | 4.00 | 0.392 | 0.177 |

> 💾 MPBA 几乎达到 uniform M4 的压缩效果，但精度更高，性价比极佳。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **膜电位是 SNN 量化中的关键瓶颈**，必须作为“一等公民”进行专门设计。
2. ✅ **通道级尺度适配 + 混合精度分配** 是实现高效低比特 SNN 推理的关键。
3. ✅ **Unified Scale Bridge** 成功实现了硬件友好的 scale 转换，兼顾精度与效率。
4. ✅ **MPBA 显著优于统一比特分配**，在相同预算下获得更高准确率。
5. ✅ 方法具有强泛化能力，适用于多种 SNN 架构（CNN、Transformer）、多种任务（分类、分割、事件识别）。

---

### **局限性**
- 当前方法依赖 LIF 神经元模型，对更复杂的神经动力学（如 ALIF、Izhikevich）尚未验证。
- 膜电位量化仍可能导致长期误差积累，尤其在超长序列任务中未充分测试。
- 实际硬件部署需实现 packed integer 运算栈，当前实验为模拟量化。

---

### **未来工作方向**
- 扩展至更多类型的 spiking neuron dynamics。
- 探索训练感知的轻量微调（lightweight fine-tuning）以进一步压缩极限。
- 开发端到端的整数量化推理引擎，结合 PTQ4SNN 输出进行真实能效测量。
- 应用于更大规模的 vision-language 或多模态 spike 模型。

---

> ✅ **总结一句话**：  
> **PTQ4SNN 首次系统性地将膜电位纳入 PTQ 设计，通过 channel-wise Unified Scale Bridge 和 MPBA，在无需重训练的前提下实现了 W4 + ~M4 的高效 SNN 部署，精度损失极小，通用性强，为边缘侧低功耗 spike AI 提供了实用解决方案。**

</details>

---

### 11. [DGEMM with Ozaki Scheme I/II on FP4 Tensor Cores: A Base-13 E2M1 Limb Representation](https://arxiv.org/abs/2608.06812)

**Authors**: Shun-ichiro Hayashi, Daichi Mukunoki, Tetsuya Hoshino, Takahiro Katagiri  
**Category**: cs.DC  
**Published**: 2026-08-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.06812v1  

#### Abstract
This paper proposes a method and its implementation for emulating FP64 matrix multiplication (DGEMM) by constructing, on FP4 (E2M1; 2 exponent bits and 1 mantissa bit) Tensor Cores, Ozaki schemes I and II, which realize high-precision matrix multiplication on low-precision arithmetic units. Prior im...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DGEMM with Ozaki Scheme I/II on FP4 Tensor Cores: A Base-13 E2M1 Limb Representation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着大语言模型（LLMs）的发展，GPU厂商将更多硅片资源分配给低精度计算单元（如 FP4、FP8），而传统科学计算依赖的 FP64 单元被大幅削减（例如 NVIDIA B300 中 FP64 性能下降超 90%）。这导致 **FP64 级别的高性能矩阵乘法（DGEMM）面临性能停滞的风险**。

已有研究通过在 INT8 或 FP8 Tensor Cores 上使用 **Ozaki Scheme** 来模拟 DGEMM，但尚未有效利用更高吞吐量的 **FP4 (E2M1)** Tensor Cores——因为 FP4 数值稀疏且动态范围小，中间求和易产生舍入误差，被认为不适合用于高精度模拟。

---

### 🚀 提出的新方法与创新思路

本论文提出了一种基于 **Base-13 FP4 Limb 表示法** 的新框架，首次成功将 **Ozaki Scheme I 和 II** 移植到 **FP4 Tensor Cores** 上，实现对 DGEMM 的高效模拟。

#### 核心思想：
- **FP4 值的整数性质**：所有 E2M1（FP4）格式的数值在乘以 2 后均为整数。
- **任意整数可表示为 `c + 13n` 形式**，其中 `c ∈ S = {0, ±1, ..., ±12, ±24}` 是一个缩放后的 FP4 值集合。
- 利用该性质，将任意整数分解为 **base-13 的 limb 序列**，每个 limb 可精确存储在一个 FP4 数中。
- 所有中间累加保留在 **FP32 accumulator** 中（而非写回 FP4），从而避免精度损失。

#### 主要贡献：
1. **形式化并证明了 Base-13 FP4 Limb 表示法的完备性和无误差性**，使得任意整数可在不丢失精度的情况下编码为 FP4 序列。
2. 首次构建了 **Ozaki Scheme I 和 II 在 FP4 上的完整实现方案（OzI-FP4 和 OzII-FP4）**。
3. 设计了一组新的模数（moduli）集合 `{169, 115, ..., 53}`，共 19 个，满足每个模下的 residue 均可用两个 FP4 limb 表示。
4. 实现了 **bit-exact 的 INT8×INT8→INT32 GEMM 模拟**，表明 FP4 Tensor Cores 可完全替代 INT8 进行整型计算。
5. 提出多项内核优化技术（kernel fusion），显著提升实际性能。

---

### 🔍 相比现有方法的优势

| 维度 | 优势说明 |
|------|--------|
| **理论性能潜力** | 当 FP4 吞吐是 FP8 的两倍时，OzII-FP4 理论上即可超越 FP8 版本（因 GEMM 数量比约为 1.9:1） |
| **硬件趋势适配性** | 更适应未来 GPU 架构（如 B300、Rubin）中 INT8 单元减少、FP4 成为主力的趋势 |
| **精度控制** | 中间运算全程保持整数精度，最终仅一次 round-to-nearest，误差来源仅为输入量化 |
| **通用性扩展** | 方法同样适用于其他低位宽浮点格式（只要其加倍后为整数） |

---

## 2. 核心实验方法和设置

### 🧪 实验平台
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (sm_120)
- **Tensor Core 峰值性能**：
  - FP4: 2000 TFLOPS
  - FP8: 1000 TFLOPS
  - INT8: 1000 TOPS
  - FP64 (cuBLAS): ~1.85 TFLOPS
- **内存带宽**：1792 GB/s（标称），实测 1490 GB/s
- **软件栈**：CUDA 12.8, PyTorch 2.11.0, Triton 3.6.0
- **对比基线**：GEMMul8-FP8 / GEMMul8-INT8（开源 Ozaki Scheme II 实现）、cuBLAS DGEMM

---

### 📊 实验设置与评估指标

#### 测试任务
- **标准 DGEMM 模拟**：$ C = \text{fl}(A \times B) $，其中 $ A, B \in \mathbb{R}^{N\times N} $，目标是达到 FP64 精度输出。

#### 输入生成
- 元素形式：$ \pm(1+u)\cdot2^e $，其中 $ u\sim U[0,1), e\in\{0,\dots,p\} $
- 控制指数范围 $ p $ 以测试不同动态范围下的精度表现

#### 评估指标
| 指标 | 描述 |
|------|------|
| **等效 DGEMM TFLOPS** | $ 2N^3 / t $，越高越好 |
| **端到端延迟（latency）** | 包括预处理 + GEMM + 后处理 |
| **组件级时间拆解** | 分析 compute stage 与 preprocessing 开销 |
| **相对误差** | 对比 220-bit 高精度参考结果，取最大 componentwise 相对误差 |

#### 对比方法
- **OzI-FP4**：基于位置展开的 base-13 limb 方案（p=q=15）
- **OzII-FP4**：基于 RNS + CRT 的 residue 方案（L=19 moduli）
- **GEMMul8-FP8 / INT8**：当前最先进的 Ozaki Scheme II 实现
- **cuBLAS DGEMM**：原生 FP64 实现，作为性能下限基准

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table I）

| 方法 | N=4096 (Compute) | N=8192 (Compute) | N=16384 (Compute) | N=16384 (End-to-End) |
|------|------------------|------------------|--------------------|-----------------------|
| cuBLAS DGEMM | 1.85 TFLOPS | 1.85 TFLOPS | 1.85 TFLOPS | 1.85 TFLOPS |
| GEMMul8-FP8 | 13.68 | 15.41 | 15.68 | **14.65** |
| **OzII-FP4 (本文)** | **16.28** | **17.11** | **17.31** | **15.26** ✅ |

> ✅ 在 **N=16384** 规模下，OzII-FP4 不仅 compute stage 超越 GEMMul8-FP8（快约 **1.11×**），而且 **端到端也更快（快约 1.04×）**

---

### 🔁 与基线方法对比

| 对比维度 | 结果 |
|--------|------|
| **vs GEMMul8-FP8** | 
| - Compute Stage | 快 **1.10–1.19×** |
| - End-to-End | 在 N=16384 下快 **1.04×**（即耗时更短） |
| - 达到峰值比例 | OzII-FP4 为 61–65%，优于 GEMMul8-FP8 的 53–61% |
| **vs cuBLAS DGEMM** |
| - Compute Speedup | 最高达 **9.4×** |
| - End-to-End Speedup | 最高达 **8.2×**（N=16384） |

---

### 🔍 消融实验结果（Ablation Study）

#### （1）OzI-FP4 优化效果（图4）
- **原始版本**：700+ ms
- **+ fused limb decomposition**：提速 2.0×（end-to-end）
- **+ TN-direct writeout & tiled encoding**：总提速达 **2.9×**
- 内核融合使 pipeline 从 memory-bound 转为 **GEMM-bound**

#### （2）OzII-FP4 优化路径（图5）
- **原始（Garner + 多 launch）**：250 ms
- **→ 改用 Direct CRT (O(L))**：降至 183 ms（提速 1.36×）
- **→ Epilogue Fusion + Kernel Fusion**：进一步降至 **81 ms**（总提速 **3.1×**）
- 显示 **kernel fusion 是性能飞跃的关键**

#### （3）运行时建模验证（图7）
- 提出模型：  
  $$
  t_{\text{total}} \approx \frac{2QN^3}{R} + \frac{D}{\beta}
  $$
  - $ Q $: GEMM 数量（OzII-FP4: 75 vs GEMMul8-FP8: 39）
  - $ R $: 实际 FP4 峰值（1408 TFLOPS）
  - $ \beta $: 实测带宽（1490 GB/s）

- 实测与模型高度吻合，证明性能优势来自：
  1. **理论优势**：FP4:FP8 吞吐比（2×） > GEMM 数量比（~1.9×） ⇒ 净增益 1.04×
  2. **实现优势**：epilogue fusion 提升利用率 ⇒ 再获 1.06–1.14× 加成
  - 二者相乘 ≈ 实测的 1.10–1.19× 加速比

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **FP4 可用于高精度 DGEMM 模拟**：尽管其本身精度极低，但借助 **base-13 limb 表示 + FP32 accumulator**，可以安全地进行误差自由的整数 GEMM。
2. **Ozaki Scheme II 成功迁移至 FP4**：设计出适合 FP4 表示的新模数组合（L=19），并在理论上和实践中均优于 FP8 实现。
3. **性能反超 FP8 实现**：在大问题规模（16384³）下，**OzII-FP4 端到端性能已超过 GEMMul8-FP8**，标志着 FP4 正式成为科学计算可用的底层算力。
4. **kernel fusion 至关重要**：epilogue fusion 和单 kernel 多模数处理极大降低了 launch overhead 和内存访问，是达成高性能的关键。
5. **精度达标甚至更优**：所有方法均优于原生 cuBLAS DGEMM；OzI/OzII-FP4 输出 bit-wise 一致，误差仅来源于共享指数量化。

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **依赖特定 FP4 格式特性** | 方法依赖于 E2M1 加倍后为整数的性质，不直接推广至其他非整数兼容格式 |
| **预处理开销仍存在** | 尽管已优化 5×，但在小规模问题中 preprocessing 仍是瓶颈 |
| **Karatsuba 未启用** | 因 shared memory 压力大，未能启用部分 Karatsuba 优化（未来架构可能受益） |
| **尚未支持多卡扩展** | 当前实现针对单卡，而 GEMMul8 支持多 GPU，成熟度仍有差距 |

---

### 🔮 未来工作方向

1. **扩展至其他低位宽格式**：探索类似方法是否可用于 E3M0 或 E1M2 等新型低精度格式。
2. **自动调参系统集成**：结合 ADP 类机制，动态选择 limb 数或模数组合以保证精度。
3. **支持分布式训练场景**：将 OzII-FP4 扩展至多 GPU 并行环境，提升实用价值。
4. **应用于其他 BLAS 操作**：如 SYRK、TRSM 等，构建完整的 FP64-emulated BLAS 库。
5. **编译器层面自动化**：在 Triton 或 CUTLASS 中内置 limb decomposition pass，降低使用门槛。

---

> 💡 **一句话总结**：  
> 本文突破性地将 **FP4 Tensor Cores** 引入科学计算领域，提出 **base-13 limb 表示法**，实现了 **Ozaki Scheme I/II 在 FP4 上的高性能、高精度 DGEMM 模拟**，并在大尺度下 **性能超越现有 FP8 实现**，为后 FP64 时代的高性能计算提供了新路径。

🔗 **源码公开**：[https://github.com/FP4-is-All-you-Need/Oz-FP4](https://github.com/FP4-is-All-you-Need/Oz-FP4)

</details>

---

### 12. [ELMZip: Onboard Satellite Image Compression via Extreme Learning Machines for Efficient Downlink](https://arxiv.org/abs/2608.06942)

**Authors**: Woojin Cho, Junghwan Park, Sangcheol Sim, Steve Andreas Immanuel, Junhyuk Heo, Darongsae Kwon  
**Category**: cs.LG  
**Published**: 2026-08-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.06942v1  

#### Abstract
The acquisition of multispectral imagery via small satellites (e.g., CubeSats) presents significant data downlink challenges due to high data volumes and restricted communication windows. While onboard image compression is critical to address this bottleneck, traditional methods often struggle to ad...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《ELMZip: Onboard Satellite Image Compression via Extreme Learning Machines for Efficient Downlink》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
小型卫星（如 CubeSats）在获取高分辨率多光谱图像时面临严重的**下行链路带宽瓶颈**。传统压缩方法（如 CCSDS 123.0、JPEG）采用固定变换，难以适应多波段、多尺度数据中复杂的非线性统计特性。而现有的基于 **Implicit Neural Representations (INRs)** 的神经压缩方法虽然重建质量高，但依赖迭代优化（backpropagation），计算开销大、耗时长，不适合资源受限的星载平台。

### ✅ 提出的新方法与创新思路
作者提出 **ELMZip** —— 一种基于 **Extreme Learning Machines (ELM)** 和 **domain decomposition** 的新型星载图像压缩框架，其核心思想包括：

- **将图像拟合建模为凸的最小二乘问题**：利用单层随机特征网络，避免反向传播训练，实现秒级解析求解。
- **引入不对称传输协议（asymmetric transmission protocol）**：仅传输紧凑的输出权重 $ \mathbf{p}_k $，输入层参数 $ \theta_{\text{fix}} $ 在星地端通过确定性初始化共享，无需传输。
- **结合 domain decomposition 策略**：将图像划分为多个重叠子区域，每个区域使用局部 ELM 拟合，提升对复杂结构的表达能力。

### ✅ 相比现有方法的优势
| 维度 | ELMZip | 传统 INR 方法（如 SIREN、WIRE） |
|------|--------|-------------------------------|
| **训练速度** | 极快（秒级完成） | 缓慢（需数分钟至小时级迭代） |
| **能耗** | 极低（无反向传播） | 高（依赖梯度下降） |
| **下行负载** | 极小（只传输出权重） | 大（需传全部网络参数） |
| **部署可行性** | 高（适合 Jetson Nano 等边缘设备） | 低（计算资源要求高） |

> ⚡️ **关键优势总结**：ELMZip 实现了**高效、快速、低功耗的星上压缩 + 快速地面重建预览**，显著提升了小卫星的数据回传效率与实时AI地球观测能力。

---

## 2. 核心实验方法和设置

### 📦 数据集
使用 **Sentinel-2 MSI** 的 Level-0 和 Level-1C 多光谱图像数据，涵盖六种典型地理环境，具体如下：

| 数据 | Level | 地理位置 | 环境类型 |
|------|-------|----------|---------|
| Antuco | LO | 37°35'S, 71°13'W | Volcano |
| Puszta | LO | 47°28'N, 19°53'E | Grassland |
| Andaman | LO | 12°26'N, 93°56'E | Marine |
| Cairo | L1C | 30°01'N, 31°55'E | Desert |
| Merapi | L1C | 07°32'S, 110°26'E | Volcano |
| Seoul | L1C | 37°31'N, 126°55'E | Urban |

> 🔍 特别包含 **Level-0 原始数据**，更贴近真实星载传感器输出，增强实验实用性。

### ⚙️ 实验设置
- **模拟平台**：NVIDIA Jetson Nano（代表典型的低功耗星载边缘计算硬件）
- **图像尺寸**：统一标准化为 1024×1024
- **压缩目标**：所有方法均需实现约 **10× 下行负载压缩率**
- **量化处理**：输出权重进行量化以进一步减小传输体积
- **能量测量**：记录模型拟合过程中的电能消耗

### 📊 评估指标
- **PSNR**（Peak Signal-to-Noise Ratio）：衡量重建保真度
- **SSIM**（Structural Similarity Index Measure）：评估感知质量

### 🔁 基线方法对比
对比了多种主流 INR 架构：
- **MLP**（ReLU 激活）
- **SIREN**（sinusoidal 激活）
- **FFN**（Fourier Feature Network）
- **GaussNet**
- **WIRE**（complex Gabor wavelet 激活）

> ⚠️ 注意：这些基线方法在训练时使用了约 **10× 更高的能量预算**，仍用于公平比较性能上限。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table II）

| Model     | Avg PSNR (dB) ↑ | Avg SSIM ↑ |
|----------|------------------|-------------|
| MLP      | ~16.0            | ~0.45       |
| SIREN    | ~20.8            | ~0.83       |
| FFN      | ~20.7            | ~0.81       |
| GaussNet | ~17.8            | ~0.73       |
| WIRE     | ~16.5            | ~0.66       |
| **ELMZip (Ours)** | **~24.5**        | **~0.93**   |

> ✅ 在所有场景下，**ELMZip 全面超越所有基线方法**，即使后者消耗更多能量。

### 🔍 典型案例表现（如 Cairo 和 Seoul）
- **Cairo (L1C)**：PSNR 达到 **20.96 dB**（比最强基线 SIREN 提升 +16.5%）
- **Seoul (L1C)**：PSNR 达到 **19.59 dB**（相对提升 +21.7%）
- 在城市等纹理复杂区域，ELMZip 明显保留了更清晰的边缘和局部对比度，视觉效果最接近 Ground Truth。

### 🖼️ 定性结果（见 Fig. 4）
- 基线方法普遍存在**过平滑**（over-smoothing）和**细节丢失**现象
- ELMZip 能准确恢复建筑物轮廓、道路结构、火山地形等关键特征
- 尽管训练能量仅为基线的 1/10，重建质量反而更高

### ❌ 消融实验（文中未明确列出消融研究）
> 文中未提供系统的 ablation study（例如 domain decomposition 影响、窗口函数设计、子域数量 K 的选择等）。这是未来可拓展的方向。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **ELM 可有效用于星载图像压缩**：通过将图像建模为神经隐式表示，并利用 ELM 的解析解特性，可在极短时间内完成高质量拟合。
2. **不对称传输极大降低下行负载**：只需传输输出权重，节省带宽的同时保证地面可快速重建预览图。
3. **性能-效率权衡优越**：在仅消耗约 1/10 能量的情况下，ELMZip 实现了优于现有 INR 方法的 PSNR 和 SSIM 表现。
4. **适用于多样化自然环境**：在火山、草原、海洋、沙漠、城市等多种地貌中均表现出色，具备良好的泛化能力。

### ⚠️ 方法的局限性
- **内存占用较高**：由于采用 domain decomposition，需存储多个局部模型的隐藏层激活矩阵 $ \mathbf{H}_k $，对星载有限内存构成挑战。
- **缺乏理论最优性分析**：ELM 的随机特征设计虽实用，但缺乏对逼近误差的严格理论边界分析。
- **未考虑动态场景变化**：当前方法针对静态图像，尚未验证在视频或多时相序列上的扩展能力。
- **依赖预共享参数同步机制**：要求星地两端初始化完全一致，对系统鲁棒性和容错提出一定要求。

### 🔮 未来工作方向
1. 探索更高效的数值求解器（如迭代线性求解器）以减少内存占用
2. 扩展至大规模 benchmark 和多时相数据集
3. 结合轻量化编码策略（如熵编码）进一步提升压缩率
4. 向 **onboard AI workflows** 延伸，支持下游任务如植被监测、urban change detection、few-shot segmentation 等直接基于压缩表示运行

---

## ✅ 总结一句话
> **ELMZip 提出了一种面向星载平台的高效神经压缩新范式——通过 ELM + domain decomposition + 不对称传输，在极低能耗和极短时间下实现高质量图像压缩与快速地面重建，为实时 AI 驱动的地球观测提供了切实可行的技术路径。**

</details>

---

### 13. [Evolving Parallel Algorithm Portfolios via Potential-Aware Instance Generation with LLMs](https://arxiv.org/abs/2608.06808)

**Authors**: Shaofeng Zhang, Shengcai Liu, Zhiyuan Wang, Ke Tang  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.06808v1  

#### Abstract
The Automatic Construction of Portfolios via Large Language Models (LLM-ACP) suffers from poor generalization in practical few-shot scenarios when solving complex combinatorial optimization problems. Instance and algorithm co-evolution frameworks address this by expanding the training dataset with g...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Evolving Parallel Algorithm Portfolios via Potential-Aware Instance Generation with LLMs

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对 **LLM-ACP (Large Language Model-based Automatic Construction of Portfolios)** 在**少样本场景**（few-shot scenarios）下泛化能力差的问题展开研究。具体而言：

- 当前的 LLM-ACP 方法在训练实例有限时容易过拟合，难以泛化到未见的数据分布。
- 传统的 co-evolution 框架（如 CEPS、DACE）通过生成“困难实例”来增强训练集，但存在两个关键限制：
  1. **依赖高质量参考解**（high-quality reference solutions）来评估实例难度（hardness），这在实际问题中往往不可得。
  2. **实例生成模式单一**（single-mode generation patterns），导致生成的实例多样性不足，限制了算法组合的泛化能力。

---

### **提出了什么新方法或新思路**

作者提出了一种名为 **Potential-aware Instance and Algorithm Co-evolution (PIAC)** 的新框架，其核心创新包括：

#### ✅ **1. Potential Gain：无需参考解的实例质量度量**
- 提出 **potential gain** 这一新指标，用于衡量一个生成实例对当前算法组合（PAP）的“潜在改进空间”。
- 具体机制是：对当前算法的启发式组件施加微小扰动（perturbation），观察扰动后算法在该实例上的性能提升。
- 若扰动能显著改善结果，则说明该实例暴露了当前算法的决策边界缺陷，具有高“改进潜力”，应被选为高质量训练实例。
- **优势**：完全摆脱了对 LKH、HGS 等外部强求解器提供的最优解的依赖。

#### ✅ **2. LLM 驱动的多样化实例突变器（Instance Mutator）演化**
- 不再使用固定的几何变换或随机扰动操作符，而是利用 LLM 自动生成并进化一批可执行的 **instance mutator 程序**。
- 每个 mutator 是一段 Python 代码，能将基础实例转换为结构更复杂的新实例。
- 通过进化算法选择能生成高 potential gain 实例的 mutator，从而探索更广阔的 instance space。
- **优势**：极大提升了生成实例的多样性和结构性挑战性。

#### ✅ **3. 双阶段协同进化框架**
- PIAC 交替进行两个阶段：
  1. **LLM-Driven Instance Generation**：用 LLM 生成 mutator 并合成高潜力实例。
  2. **LLM-Driven PAP Construction**：用 LLM 进化新的启发式算法以应对这些挑战性实例。
- 形成“算法越弱 → 实例越难 → 算法越强”的正向循环。

---

### **相比现有方法的优势**

| 维度 | 传统方法（如 CEPS/DACE） | PIAC |
|------|--------------------------|------|
| 是否需要参考解 | 是（依赖 Opt Gap） | 否（使用 Potential Gain） |
| 实例生成方式 | 固定手工设计算子 | LLM 自动生成多样化 mutator |
| 泛化能力 | 易受限于生成模式 | 能探索更广的 instance space |
| 适用性 | 限于已有解的问题 | 可扩展至新问题领域 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### **合成数据集（Synthetic Datasets）**
- **TSP 和 CVRP** 问题，规模均为 $ n = 200 $。
- **训练集**：仅使用 8 个来自 **random uniform Euclidean (Rue)** 分布的实例（模拟少样本场景）。
- **测试集**：共六种不同空间分布，每类 100 个实例：
  - Rue（随机均匀）
  - Explosion（爆炸状）
  - Implosion（内爆状）
  - Expansion（扩张状）
  - Cluster（聚簇）
  - Grid（网格）

#### **公开基准数据集（Public Benchmarks）**
- **TSPLib**：标准 TSP 实例集合。
- **CVRPLib**：包含多个子集（A, B, CMT, F, Golden, Li, M, P, tai, X）的标准 CVRP 实例。

---

### **实验设置和评估指标**

#### **算法骨干（Backbone）**
在三种经典算法框架上进行了实例化验证：
- **Greedy Constructive**
- **Ant Colony Optimization (ACO)**
- **Guided Local Search (GLS)**

#### **评估指标**
- **Optimality Gap (%)**：
  $$
  \text{Gap}(A,x) = \frac{F(A,x) - f_{\text{ref}}(x)}{f_{\text{ref}}(x)} \times 100\%
  $$
  - $ F(A,x) $：算法组合 $ A $ 在实例 $ x $ 上的最佳目标值。
  - $ f_{\text{ref}}(x) $：由 LKH（TSP）或 HGS（CVRP）计算的近似最优解。

#### **超参数设置（见 Table I）**
- 最大迭代次数：4
- 每次迭代新增实例数 $ N_{\text{aug}} $：8
- 每轮 mutator 执行次数 $ N_{\text{exec}} $：4
- 噪声扰动次数 $ N_p $：64
- LLM 模型：默认使用 DeepSeek-V3.2

---

### **基线方法对比**

| 方法 | 类型 | 特点 |
|------|------|------|
| **FunSearch**, **EoH**, **ReEvo**, **MCTS-AHD** | 单一启发式生成 | 利用 LLM 设计单个 heuristics |
| **EoH-S** | 多启发式组合 | 构建互补算法组合，但固定训练集 |
| **PIAC (RND)** | 消融版本 | 使用随机 mutator，保留 potential gain |
| **PIAC (GAP)** | 消融版本 | 使用 LLM mutator，但用 Opt Gap 替代 potential gain |

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Tables II–IV）**

#### 📊 **TSP Greedy Constructive 结果（Table II）**
| 方法 | Avg. Gap (%) |
|------|--------------|
| EoH-S | 14.83% |
| **PIAC** | **11.90%** |
| → **相对提升**：**19.76%** ✅

#### 📊 **CVRP Greedy Constructive 结果**
| 方法 | Avg. Gap (%) |
|------|--------------|
| EoH-S | 20.60% |
| **PIAC** | **18.29%** |
| → **相对提升**：**11.21%** ✅

#### 📊 **TSP ACO 与 GLS 结果（Table III）**
| 方法 | TSP ACO (Avg Gap) | TSP GLS (Avg Gap) |
|------|-------------------|-------------------|
| EoH-S | 10.20% | 0.056% |
| **PIAC** | **8.00%** | **0.050%** |
| → 均实现进一步压缩，尤其在已接近饱和的 GLS 上仍有效。

#### 📊 **公共基准表现（Table IV）**
| 数据集 | EoH-S | PIAC |
|--------|-------|------|
| **TSPLib (TSP)** | 13.67% | **11.79%** |
| **CVRPLib (Avg)** | 29.88% | **24.53%** |
| → 在 10 个 CVRPLib 子集中，PIAC 获得 **8 项第一**，2 项第二。

---

### **消融实验结果**

#### 🔍 **PIAC vs. PIAC (RND)**
- 替换为随机 mutator 后性能下降：
  - TSP Constructive：13.47% → 11.90%
  - CVRP Constructive：19.12% → 18.29%
- **结论**：LLM 演化的 mutator 能生成更具挑战性的实例，显著提升泛化。

#### 🔍 **PIAC vs. PIAC (GAP)**
- 使用 Opt Gap 作为评价指标时性能略逊于 potential gain：
  - 表明 **potential gain 更精准地识别出“可改进”的实例**，而非仅仅是“难解”的实例。

#### ⏱️ **效率对比（Table V）**
| 任务 | Potential Gain (s) | Opt Gap (s) |
|------|--------------------|------------|
| CVRP Constructive | **9.46** | 46.35 |
| TSP ACO | 16.63 | 17.46 |
- **结论**：potential gain 在多数情况下**更快且不依赖外部求解器**，更适合在线 co-evolution。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **PIAC 显著优于现有 LLM-ACP 方法**，在多种 backbone 和数据分布下均取得最佳泛化性能。
2. ✅ **potential gain 是一种高效、无监督的实例质量代理指标**，避免了对参考解的依赖，并能精准定位算法弱点。
3. ✅ **LLM 驱动的 mutator 演化机制极大地增强了实例多样性**，使算法组合能适应更广泛的空间拓扑。
4. ✅ **协同进化动态持续推动性能提升**：随着新实例不断加入，算法组合未出现早停现象，而 EoH-S 很快进入平台期（Figure 4）。
5. ✅ **potential gain 与 optimality gap 正相关**（Pearson: TSP 0.7445, CVRP 0.5230），但更关注“改进潜力”。

---

### **方法的局限性**

1. **假设启发式输出为矩阵形式**：当前 perturbation 机制基于 $ M = H(x,s) $ 的矩阵结构，对于非结构化决策表示（如树形策略）需重新设计。
2. **依赖 LLM 的编程能力**：若 LLM 无法正确生成可运行 mutator 或 heuristic 代码，整个流程会失败。
3. **计算开销较高**：尽管 potential gain 快于 Opt Gap，但整体框架涉及多次 LLM 查询与算法执行，在大规模问题上可能受限。

---

### **未来工作方向**

1. **设计通用扰动机制**：支持任意形式的决策表示（如函数指针、神经网络权重等）。
2. **引入多模态 LLM**：结合图像理解能力，直接分析实例几何特征以指导 mutator 生成。
3. **扩展至其他 COP 问题**：如 Job Shop Scheduling、Knapsack 等。
4. **降低 LLM 成本**：探索小型专家模型替代通用 LLM 进行代码生成。
5. **理论分析 co-evolution 收敛性**：建立 potential gain 与泛化误差之间的理论联系。

---

> **一句话总结**：  
> PIAC 通过 **potential gain** 和 **LLM-driven mutator evolution** 实现了无需参考解、高多样性的实例生成，构建出在少样本下仍具强泛化的并行算法组合，在 TSP/CVRP 上实现了高达 **19.76%** 的相对性能提升，标志着 LLM-ACP 向实用化迈出了关键一步。

</details>

---

### 14. [Not All Problems Are Best Modeled as MILP: A DSL-Centric Framework for Flexible and Accurate Optimization Modeling](https://arxiv.org/abs/2608.07040)

**Authors**: Shaofeng Zhang, Hongyuan Su, Qingwen Peng, Zefang Zong, Shengcai Liu, Ke Tang, Yong Li  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.07040v1  

#### Abstract
Solving combinatorial optimization problems (COPs) requires not only efficient algorithms but also carefully crafted formulations. While recent works have leveraged LLMs to automate optimization modeling, current frameworks predominantly rely on a rigid mixed-integer linear programming (MILP) paradi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Not All Problems Are Best Modeled as MILP: A DSL-Centric Framework for Flexible and Accurate Optimization Modeling*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于 **Mixed-Integer Linear Programming (MILP)** 的自动化优化建模框架存在两大瓶颈：
- **表达能力受限**：将复杂的组合优化问题（COPs）强行映射为线性约束会导致“约束爆炸”（constraint explosion），例如 CVRP 中的子环消除约束数量随规模指数增长，超出 LLM 的推理能力。
- **求解器灵活性差**：MILP 范式强制使用通用求解器（如 Gurobi），无法利用领域专用的高效启发式或学习型求解器（如 NCO 方法），牺牲了计算效率。

### 🚀 提出的新方法：OptiDSL
提出一个以 **Domain-Specific Language (DSL)** 为核心的新型优化建模框架 —— **OptiDSL**，其核心思想是：
> **解耦问题建模与求解过程**，通过 LLM 将自然语言描述转换为标准化的 DSL 格式（如 VRPLib、OR-Library 风格），再交由最适合该领域的专用求解器处理。

#### 创新架构包括三个模块：
1. **DSL-Based Task Formulation**  
   使用 LLM 将自然语言问题映射到预定义的 DSL 模板中，避免生成复杂 MILP 数学公式。采用两阶段流程：
   - **语义路由**（Semantic Routing）：LLM 先识别问题类型并选择对应 DSL 模板。
   - **实例化填充**（Instantiation）：从文本中提取参数并逻辑推导隐含约束（如 `OPEN_ROUTE: TRUE`）。

2. **Adaptive Solver Execution**  
   构建一个多样化的 **Solver Pool**（见 Table 1），涵盖精确求解器（Gurobi）、启发式算法（LKH, PyVRP）和神经组合优化（NCO）模型（RouteFinder, POMO）。  
   引入基于离线性能分析的动态调度机制，根据问题规模和用户偏好（速度 vs. 精度）自动匹配最优求解器。

3. **COPs Benchmark Evaluation (OptiDSLBench)**  
   构建了一个大规模、高质量的基准测试集，覆盖 **44 种 COP 类型**，支持跨域公平评估。

### 🔍 相比现有方法的优势
| 维度 | MILP-Based 方法（如 LLMOPT, CoE） | OptiDSL |
|------|-------------------------------|--------|
| **建模表达力** | 受限于线性形式，难以表示复杂结构 | 利用 DSL 天然支持领域语义 |
| **求解器兼容性** | 仅能对接 MILP 求解器 | 支持多种非学习与学习型求解器 |
| **建模效率** | 需要生成大量数学变量与约束 | 输出简洁 DSL 文件，token 更少 |
| **可扩展性** | 随问题规模增大建模失败率上升 | 在大尺度 CVRP 上仍保持高成功率 |

---

## 2. 核心实验方法和设置

### 📚 数据集
构建了新的综合基准 **OptiDSLBench**，包含：
- **44 种 COP 类型**，分布在五个领域：
  - **VRP**: 24 种变体（如 CVRP, OVRP, VRPTW）
  - **Scheduling (SP)**: 6 种（如 JSSP, FJSSP）
  - **Bin Packing (BPP)**: 8 种（2D/3D, 可旋转等）
  - **Graph Problems (GP)**: 4 种（MIS, MVC, MaxCut, MaxClique）
  - **Knapsack (KP)**: 6 种（0-1KP, MDBKP 等）
- 每类问题包含 **100 个实例**，共 **4,400 个样本**。
- 使用半自动化流程生成多样化自然语言描述，并进行人工校验以减少 LLM 幻觉。

此外，在已有 MILP 基准上也进行了验证：
- **NL4Opt**, **NLP4LP**, **MAMO**, **ComplexOR**

### ⚙️ 实验设置
- **LLM 后端**：
  - OptiDSL & CoE 使用 **DeepSeek-V3.2**
  - ORLM 使用微调版 **Llama3**
  - LLMOPT 使用 **Qwen2.5-14B**
- **统一求解器比较**：在建模质量评估阶段，所有方法均使用 **Gurobi** 作为下游求解器，以排除求解差异带来的偏差。
- **问题规模控制**：为确保 Gurobi 能求得最优解用于评估，设定小规模测试（如 VRP 节点数=5，其余=10）。

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Execution Rate (ER)** | 成功解析并执行的实例比例 → 衡量建模结构正确性 |
| **Optimality Rate (OR)** | 得到全局最优解的比例 → 衡量建模准确性 |
| **Modeling Time (MT)** | 从输入到输出 DSL/MILP 的耗时（秒） |
| **Token Consumption** | 输入/输出 token 数量，反映成本开销 |

### 🆚 基线方法
- **Chain-of-Experts (CoE)**：多智能体协作推理框架
- **ORLM**：基于指令微调的大模型优化建模系统
- **LLMOPT**：端到端学习定义与求解优化问题的方法

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2）

| 方法 | Avg. ER ↑ | Avg. OR ↑ | Avg. MT ↓ |
|------|----------|----------|----------|
| **OptiDSL (Ours)** | **0.97** | **0.88** | **9.89s** |
| CoE | 0.87 | 0.57 | 149.87s |
| ORLM | 0.72 | 0.31 | 69.52s |
| LLMOPT | 0.30 | 0.05 | 127.78s |

> 💡 **核心提升**：
> - **OR 提升 51.66%**（相对提升）
> - **建模时间降低 91.71%**
> - **ER 提升 10.13%**

#### 在 VRP 子类上的显著优势：
| 方法 | ER | OR |
|------|----|----|
| OptiDSL | 0.98 | 0.93 |
| CoE | 0.94 | 0.46 |
| ORLM | 0.77 | 0.28 |
| LLMOPT | 0.30 | 0.00 |

> 在最复杂的 **VRP with Time Windows (VRPTW)** 上，OptiDSL 的 OR 达到 **0.99**，而最强基线仅为 **0.15**。

### 🔁 与其他基准的对比（Table 3 & 4）
| 基准 | 方法 | ER | OR |
|------|------|-----|-----|
| **Existing COP Bench** (Jiang et al. 2025b) | OptiDSL | **0.94** | **0.89** |
| | 最佳基线 | 0.83 | 0.66 |
| > **OR 提升 23.09%** | | | |
| **MILP Benchmarks** (NL4Opt, MAMO, etc.) | OptiDSL | — | **92.3%** |
| | CoE | — | 82.1% |
| > **绝对 OR 高出 10.2~43.6 pp** | | | |

### 🔍 消融实验与扩展分析（Supplementary）

#### ✅ 下游求解器分析（Table 5）
展示了不同规模 CVRP 下各求解器的表现：

| Size | Gurobi (Obj/Time) | PyVRP (Obj/Time) | RouteFinder (Obj/Time) |
|------|-------------------|------------------|------------------------|
| n=5  | 2.44 / 0.22s      | 2.45 / 7.56s     | 2.45 / 0.05s           |
| n=10 | 3.76 / 0.52s      | 3.77 / 10.59s    | 3.89 / 0.07s           |
| n=50 | — / >200s         | 10.12 / 15.57s   | 10.32 / 0.23s          |

> 发现：随着规模增大，Gurobi 不可行，而 **RouteFinder** 在毫秒级时间内提供近优解，体现 OptiDSL 的灵活适配能力。

#### ✅ 不同 LLM 后端的影响（Table 3 & 4）
| LLM | 特点 | 性能趋势 |
|-----|------|---------|
| **Llama3.3-70B** | 轻量快速 | 建模时间最短，适合低延迟场景 |
| **DeepSeek-R1 (660B)** | 推理增强 | OR 和 ER 最高，但耗时长 |
| **Qwen3-235B** | 平衡型 | 综合表现良好 |

> 表明 OptiDSL 具有良好的模块化特性，可根据需求灵活替换 LLM。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **并非所有问题都适合建模为 MILP**：强制线性化会引入不必要的复杂性和错误风险，尤其对图结构、路径依赖等问题。
2. **DSL 是更自然的中间表示**：利用领域公认的数据格式（如 VRPLib）能有效降低 LLM 的建模负担，提高准确率。
3. **建模与求解应解耦**：OptiDSL 成功实现了“一次建模，多求解器适配”，极大提升了系统的灵活性和实用性。
4. **显著优于现有自动化建模方法**：在建模准确性、执行成功率、响应速度和资源消耗方面全面领先。

### ⚠️ 局限性
- **DSL 模板依赖**：需要预先定义完整的 DSL 模板池，对于全新未见过的问题类型可能无法覆盖。
- **泛化边界**：虽然支持 44 类 COP，但在极端复杂的混合约束下仍可能出现建模偏差。
- **依赖 LLM 能力**：最终性能受制于 LLM 的信息抽取和逻辑推理能力，尤其是在模糊描述或多义词情况下。

### 🔮 未来工作方向
- 扩展更多领域的 DSL 模板（如供应链网络设计、鲁棒优化）。
- 引入反馈闭环机制，利用求解结果反哺建模修正（self-correction loop）。
- 探索完全免模板的生成方式，结合 grammar-guided decoding 自动合成合法 DSL。
- 构建开放平台，支持社区共建 DSL 库与求解器插件生态。

---

> 🔗 **代码开源地址**：[https://anonymous.4open.science/r/OptiDSL](https://anonymous.4open.science/r/OptiDSL)

</details>

---

### 15. [Fixed and Adaptive Topological DeepONets: Functional Measurements on Hausdorff Locally Convex Spaces](https://arxiv.org/abs/2608.06428)

**Authors**: Khemraj Shukla, George Em Karniadakis  
**Category**: cs.LG  
**Published**: 2026-08-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.06428v1  

#### Abstract
Deep Operator Networks (DeepONets; arXiv:1910.03193) typically encode an input function through point values on a fixed discretization. Building on the Topological DeepONet framework of Ismailov (arXiv:2603.11972), we replace point samples by continuous linear functionals drawn from the continuous d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Fixed and Adaptive Topological DeepONets**

---

## **1. 主要贡献和创新点**

### **解决的问题**
传统 DeepONet 的 **branch 网络**依赖于在固定传感器位置上的点值采样（pointwise evaluations），这导致以下限制：
- 输入必须对齐到统一网格，无法处理**异构分辨率**或**非均匀采样**的数据；
- 点值表示可能不是最优坐标系统，尤其当算子依赖于输入函数的全局结构（如积分、矩、频谱特征）时；
- 在非范数化（non-normable）函数空间（如分布空间 $ \mathcal{D}'(\Omega) $）中，点值甚至不连续，传统 DeepONet 理论框架失效。

### **提出的新方法与新思路**
本文基于 Ismailov [2] 提出的 **Topological DeepONet** 框架，发展了两种具体实现：

#### ✅ **Fixed Topological DeepONet**
- 使用预设的、全局的连续线性泛函（continuous linear functionals）作为测量，例如：
  - 内积 $ l_j(v) = \langle v, \phi_j \rangle $
  - Lagrange 基投影、谱基展开系数等。
- 这些泛函属于输入空间 $ \mathcal{V} $ 的连续对偶 $ \mathcal{V}' $，兼容其拓扑结构。

#### ✅ **Adaptive Topological DeepONet**
- 测量泛函本身从数据中学习，但仍保持为 $ \mathcal{V}' $ 中的元素。
- 学习过程被约束在一个“可接受的对偶字典”（admissible dual dictionary）的线性组合内，确保泛函的连续性。
- 引入一个**仅训练阶段使用的解码器**（training-only decoder）和软正则化项（soft regularization），防止信息丢失和特征坍缩。

#### ✅ **Two-Step 系数空间训练策略**
- 结合增强版 Two-Step 方法 [3]：
  1. 阶段 I：通过加权 SVD 构建稳定、低维的输出基 $ Q $；
  2. 阶段 II：Branch 网络直接预测对应系数 $ c $。
- 分离了输出基学习与输入表示学习，使不同输入编码方式可在相同输出表示下公平比较。

#### ✅ **理论贡献：误差分解与 Barron 率改进**
- 推导了离散误差界，将总误差分解为三项：
  $$
  \text{Error} \leq \underbrace{L_h \cdot \epsilon_{\text{rec}}(q)}_{\text{测量重建误差}} + \underbrace{\epsilon_{\text{out}}(r, q)}_{\text{输出基截断误差}} + \underbrace{\text{ENN}}_{\text{神经近似误差}}
  $$
- 并给出 **Barron-rate refinement**：$ \text{ENN} = \mathcal{O}(1/\sqrt{N}) $，其中 $ N $ 是网络宽度。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **表示能力** | 支持非范数化空间（如分布空间），突破 Banach 空间限制 |
| **离散化无关性** | 同一组泛函可通过自适应求积规则应用于任意网格，无需插值 |
| **紧凑性与效率** | 用少量泛函（如 128）即可达到高精度，显著降低输入维度（压缩比达 32×） |
| **鲁棒性** | 对噪声、缺失观测更稳健 |
| **可解释性** | 泛函具有明确数学意义（如局部平均、频谱成分） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共五个基准任务，涵盖不同类型算子：

| 数据集 | 描述 | 输入空间性质 |
|--------|------|-------------|
| **Antiderivative Operator** | 函数反导数映射 $ v \mapsto \int_0^x v(s)ds $ | 光滑函数空间，用于验证误差分解 |
| **Heterogeneous Darcy Flow** | 渗透率场 $ a(x) $ 到压力场 $ u(x) $ 的非线性映射 | $ L^\infty(\Omega) $，椭圆型 PDE |
| **Fixed-time Navier-Stokes** | 初始涡量场 $ w(x, t_0) $ 到未来时刻 $ w(x, t_{10}) $ 的快照预测 | $ L^2(\Omega) $，时间演化 PDE |
| **Time-evolving Navier-Stokes** | 十个历史快照 $ \{w(t_0), ..., w(t_9)\} $ 到未来四十步轨迹的预测 | 时间序列函数空间 |
| **Distribution-valued Screened Poisson** | 点源分布（Dirac delta 和）到解场的映射 | 分布空间 $ \mathcal{M}(\Omega) $，**非范数化** |

### **实验设置与评估指标**

#### ✅ **通用设置**
- 所有 Two-Step 模型共享相同的输出基 $ Q $（来自训练集 SVD）；
- 固定与自适应模型使用相同参数预算（除训练期额外模块）；
- 自适应模型最终推理模型不含解码器，仅保留 $ M, B_\theta, Q $。

#### ✅ **评估指标**
| 指标 | 定义 |
|------|------|
| **Mean Relative $ L^2 $ Error** | $ \frac{1}{N}\sum_{i=1}^N \frac{\|u_i - \hat{u}_i\|_{L^2}}{\|u_i\|_{L^2}} $ |
| **Global Relative $ L^2 $ Error** | $ \frac{\|\mathbf{U} - \hat{\mathbf{U}}\|_F}{\|\mathbf{U}\|_F} $，所有样本堆叠后计算 |
| **RMSE / MAE** | 根均方误差 / 平均绝对误差 |
| **Coverage @ Threshold** | 预测误差低于某阈值（如 2%, 5%）的样本比例 |

#### ✅ **基线方法对比**
| 基线 | 说明 |
|------|------|
| **Vanilla DeepONet** | 原始 DeepONet，联合训练 branch/trunk |
| **Two-Step DeepONet** | 使用 SVD 输出基的稳定版本 |
| **Sensor Two-Step** | 使用 $ q $ 个点传感器输入的 Two-Step 版本 |
| **Full-field Two-Step** | 使用完整输入场（无压缩） |
| **FNO (Fourier Neural Operator)** | 强有力的谱方法基线，适用于周期域 |
| **PCA / Random Projection** | 经典降维方法 |
| **DeepSets Source List** | 处理变长点源列表的集合模型 |
| **Rasterized Two-Step** | 将点源栅格化后输入 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 任务 | 最佳方法 | 关键指标 |
|------|---------|----------|
| **Darcy Flow** | Adaptive Topological | Global Rel. $ L^2 $: **7.27%** (vs. Sensor: 10.39%) |
| **Fixed-time NS** | Adaptive Topological | Mean Rel. $ L^2 $: **1.685% ± 0.017%** (3 seeds) |
| | FNO (对比) | Mean Rel. $ L^2 $: **0.832% ± 0.172%** |
| **Time-evolving NS** | Adaptive Topological | Mean Rel. $ L^2 $: **4.57%**, Coverage@5%: **78.4%** |
| **Antiderivative** | Learned Measurement | Mean Rel. $ L^2 $: **2.12×10⁻²** (优于 Two-Step 基线 72% 样本) |
| **Screened Poisson (OOD)** | Adaptive Functional | OOD Mean Rel. $ L^2 $: **39.80%** (远优于 DeepSets/Rasterized) |

### **与基线方法的对比结果**

#### 🔹 **Darcy Flow 实验**
- **Fixed Topological vs Sensor Two-Step**（同输入维度 32）：
  - 错误从 **10.39% → 5.88%**，相对降低 **43.4%**
- **跨分辨率泛化**（测试于未见网格 57×57, 73×73, 97×97）：
  - Functional 模型误差稳定在 **~5.5–5.6%**
  - 插值基线（Interpolated Two-Step）误差上升明显
- **抗噪与抗缺失**：
  - 30% 观测缺失时，Functional 模型误差仅增 **~10%**，而插值基线增 **~75%**

#### 🔹 **Navier-Stokes 实验**
- **Fixed-time**：
  - Adaptive Topological 是所有 **DeepONet 类模型中最准确的**
  - 虽然 FNO 更准（0.832% vs 1.685%），但：
    - 需完整 64×64 输入（4096 维）
    - 训练时间约 **2×**
    - 峰值 GPU 内存消耗 **10.7×**
- **Time-evolving**：
  - Adaptive 模型在 **覆盖率** 上领先：
    - 78.4% 样本误差 < 5%（第二名为 72.4%）
  - 且优于全输入 Two-Step 模型（4.57% vs 4.77%），表明有效压缩去冗余

#### 🔹 **消融实验（Controlled Operator）**
设计了一个已知任务相关子空间 $ \text{span}\{\phi_1,\phi_2,\phi_3\} $ 的可控算子：
- **Fixed Topological**：错误高达 **56.3%**
- **Adaptive Topological**：错误降至 **<1%**
- **关键发现**：
  - 自适应学习能精准恢复任务相关泛函子空间（见图6）
  - “任务相关解码”（task-relevant decoding）优于“全输入重建”
  - 正则化有效控制漂移与条件数

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **功能测量优于点传感器**  
   在相同输入维度下，全局泛函（如内积、多尺度原子）比局部点值提供更丰富、更具判别性的信息。

2. ✅ **自适应测量带来显著增益**  
   学习任务相关的测量泛函可大幅降低误差，且不增加推理成本。

3. ✅ **离散化无关性与鲁棒性**  
   功能坐标天然支持异构网格输入，避免插值误差，在噪声和缺失观测下表现更稳健。

4. ✅ **高效压缩仍能超越全输入模型**  
   以 32× 压缩比（128 vs 4096），自适应 Topological DeepONet 在时间演化 NS 任务上**超过全输入 Two-Step 模型**，证明其能提取动态关键特征。

5. ✅ **适用于非范数化空间**  
   在分布值 Screened Poisson 问题中，功能测量直接作用于点源列表，无需栅格化，**首次展示了在非范数化空间的有效学习**。

6. ✅ **FNO 的优势与代价并存**  
   FNO 在周期性问题上精度最高，但牺牲了：
   - 离散化灵活性
   - 训练稳定性（seed 方差大）
   - 计算资源（内存与时间）

### **局限性**
- 当前自适应学习仍受限于预定义字典的张成空间；
- 字典设计缺乏自动化原则，依赖先验知识；
- 理论误差界较保守，实际性能常优于界估计；
- 对极端高频或奇异结构的捕捉能力有待验证。

### **未来工作方向**
- 开发物理信息驱动的 **Physics-Informed Topological DeepONets**；
- 研究**数据驱动的字典生成方法**（如通过注意力机制）；
- 应用于真实世界**多模态、变分辨率实验数据融合**；
- 将离散误差分析推广至**完全无限维局部凸空间**；
- 探索与其他架构（如 Transformer、Neural Fields）结合的可能性。

--- 

> 📌 **一句话总结**：  
> 本文提出的 **Topological DeepONet** 将 DeepONet 从“点值采样”提升到“功能测量”层面，实现了在**非范数化空间中的算子学习**，并提供了**紧凑、可解释、离散化无关且鲁棒的功能坐标系统**，为科学机器学习中的异构数据建模开辟了新路径。

</details>

---

### 16. [EntropyMoE: Entropy-Aware Sparse Expert Routing for Tokenizer-Free LLMs](https://arxiv.org/abs/2608.06398)

**Authors**: Bo Liu, Muxuab Yu, Yu Zhang, Pengfei Gao, Yongping Zhang  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.06398v1  

#### Abstract
Recent byte-level large language models (LLMs) have made tokenizer-free modeling increasingly competitive by grouping bytes into dynamically sized patches. However, existing byte-patch architectures still apply the same dense feed-forward computation to every patch. This uniform computation cannot a...

---

### 17. [Interpretable Unsupervised Community Detection with LLM-Symbolized Structured Processes](https://arxiv.org/abs/2608.06402)

**Authors**: Aoting Zeng, Kai Wang, Jianwei Wang, Yuxiang Sun, Yizhang He, Wenjie Zhang  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.06402v1  

#### Abstract
Community detection is a fundamental task in graph analytics that aims to identify cohesive groups of entities with similar behaviors or interests. Classic objective-driven methods struggle with complex graph structures, while deep-learning approaches improve performance at the expense of interpreta...

---

### 18. [TaskSense: Focusing on What Matters in World Models](https://arxiv.org/abs/2608.06544)

**Authors**: SM Mazharul Islam, Manfred Huber  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.06544v1  

#### Abstract
World models for visual control typically learn compact latent states by reconstructing observations, implicitly encouraging representations to preserve information across the entire visual input. However, task-relevant content often occupies only a small fraction of the observation, while backgroun...

---

### 19. [An Exploratory Evaluation of LLM-Assisted Rewriting of Moderate-Complexity Financial Sentences for DisCoCat-Based Sentiment Analysis](https://arxiv.org/abs/2608.07439)

**Authors**: Brian Llinas, Nikos Chrisochoides  
**Category**: cs.CL  
**Published**: 2026-08-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.07439v1  

#### Abstract
Quantum natural language processing (QNLP) provides a grammar-aware framework for text modeling, and Distributional Compositional Categorical (DisCoCat) is one of its theoretically grounded formulations. Prior work on financial sentiment analysis has identified practical limitations of DisCoCat, inc...

---

### 20. [FedDOSE: Federated Learning Framework Decomposing Site Effects for Modeling Brain Dynamic Functional Connectivity](https://arxiv.org/abs/2608.07393)

**Authors**: Deepank Girish, Yi Hao Chan, Yubin Zheng, Sukrit Gupta, Jagath C. Rajapakse  
**Category**: cs.LG  
**Published**: 2026-08-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.07393v1  

#### Abstract
Functional Magnetic Resonance Imaging ( fMRI ) data are often pooled into collaborative multi-site consortia, as deep learning models for analyses require large datasets to generalize well. While Federated Learning (FL) offers a privacy-preserving paradigm for collaborative training, standard approa...

---

### 21. [Learning to Predict Middle-Layer Attention in MLLMs for Visual Token Prunin](https://arxiv.org/abs/2608.06411)

**Authors**: Yuyao Sun, Tao Deng, Shuang Li, Deqing Wang, Hao Geng, Minjun Yu  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.06411v1  

#### Abstract
Multimodal large language models (MLLMs) achieve strong performance across diverse vision-language tasks, but their efficiency is limited by the cost of processing numerous visual tokens. Visual token pruning can reduce this cost, but requires accurate token importance estimates. Recent studies have...

---

### 22. [ReQuant: Fixed-Grid Discrete Refinement for Post-Training Quantization](https://arxiv.org/abs/2608.07019)

**Authors**: Yongge Ma, Guoan Wang, Feiyu Wang, Yaoming Li, Qian Zhang, Zihan Yan, Yinjun Han, Tong Yang  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.07019v1  

#### Abstract
Post-training quantization (PTQ) is widely used to reduce the memory and computational cost of large language models. Existing PTQ methods typically obtain an initial quantized model through heuristic rules or greedy optimization, and once quantization is completed the resulting integer assignments ...

---

### 23. [DiDPO: Diff-in-Diff Policy Optimization for Coding Agent Training](https://arxiv.org/abs/2608.07147)

**Authors**: Xucong Wang, Zhe Zhao, Liheng Yu, Di Wu, Xiaofeng Cao, Pengkun Wang  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.07147v1  

#### Abstract
Reinforcement learning with Verifiable Reward (RLVR) has emerged as a powerful paradigm for training coding agents, where the execution feedback from compilation and tests provides objective verification. However, unlike agent tasks, coding agents face a unique and finer-grained credit assignment ch...

---

### 24. [An End-to-End Agent Auditing Engine](https://arxiv.org/abs/2608.07346)

**Authors**: Haoning Wang, Mingxun Zhang, Chenyue Yu, Yingjun Shang, Xia Hu, Guanchu Wang, Na Zou  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.07346v1  

#### Abstract
With the rapid advancement of large language models (LLMs), harnesses have become essential infrastructure for deploying agents across a wide range of domains. The fast-evolving harness ecosystem has also made rigorous capability evaluation increasingly important. However, efficiently building an en...

---

### 25. [A Picture is Worth a Thousand Tokens: How Vision Language Models Cut AI Energy Costs While Improving Accuracy](https://arxiv.org/abs/2608.07427)

**Authors**: Bhavika Jalli, Nikhil Korati Prasanna, Jayanta Choudhury  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.07427v1  

#### Abstract
LLM inference accounts for over 90% of AI operational energy, scaling directly with input token count---a critical inefficiency for telecom network analytics and numerical time-series data analysis (NTSDA), where raw multivariate KPI windows from 4G/5G cell sites expand into thousands of floating-po...

---

### 26. [Recovering Lesion Parameters from Aphasic Picture Naming Error Profiles in Large Language Models](https://arxiv.org/abs/2608.06429)

**Authors**: Yong Yang, Roger Newman-Norlund, Xiang Guan, Saeed Ahmadi, Regan Willis, Nadra Salman, Kalil Warren, Sophie Arheix-Parras, Srihari Nelakuditi, Leonardo Bonilha, Christopher Rorden, Rutvik H. Desai, Julius Fridriksson  
**Category**: cs.CL  
**Published**: 2026-08-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.06429v1  

#### Abstract
Interpretability methods for large language models (LLMs) describe internal state but do not directly test whether that state is causally sufficient to produce the observed behavior. In earlier work, we lesioned LLMs to produce error profiles in picture naming, a central task for assessing aphasia, ...

---

### 27. [Skaling: Chinchilla's Exponents Meet Kaplan's Coupling](https://arxiv.org/abs/2608.07222)

**Authors**: Mathurin Videau, Badr Youbi-Idrissi, David Lopez-Paz, Kartik Ahuja  
**Category**: cs.CL  
**Published**: 2026-08-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.07222v1  

#### Abstract
Neural scaling laws are foundational for language model development, yet standard formulations systematically under- and overestimate loss at data-scarce and overtraining extremes. This failure originates in the underlying assumption that model size and training data impact the loss independently. T...

---

### 28. [The Sparsity Whisperer](https://arxiv.org/abs/2608.06630)

**Authors**: Linghao Kong, Inimai Subramanian, Micah Adler, Dan Alistarh, Dan Gutfreund, Nir Shavit  
**Category**: cs.LG  
**Published**: 2026-08-10  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2608.06630v1  

#### Abstract
Pruning reduces the inference cost of large language models, but existing criteria primarily preserve large activations or reconstruct layer outputs. We argue that this overlooks a key computation performed by particularly sparsity-sensitive neurons in the MLP up and gate projections: separating sim...

---

### 29. [Shape Your Feed: An LLM-based Agentic System for Conversational Recommendation](https://arxiv.org/abs/2608.06632)

**Authors**: Ziyun Xu, Bosen Ding, Yue Zhang, Ji Qi, Qingyuan Song, Jizhou Huang, Liwei Wang, Jefferey Santelli, Yue Weng, Qichao Que, Zhenheng Yang, Junfeng Pan, Linhong Zhu  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.06632v1  

#### Abstract
Industrial recommendation systems predominantly adopt a passive ranking paradigm that infers user preferences from implicit behavioral signals (e.g., clicks, dwell time) rather than explicit, natural language inputs. As a result, users experience a persistent discrepancy between their explicit inter...

---

### 30. [CellWorld: From Gene-Level Reconstruction to Latent Cell Prediction in Spatial Transcriptomics Foundation Models](https://arxiv.org/abs/2608.06659)

**Authors**: Haiping Liu, Qian Zhao, Lijing Lin, Jingyuan Sun, Hongpeng Zhou  
**Category**: cs.AI  
**Published**: 2026-08-10  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2608.06659v1  

#### Abstract
This paper shows that latent-space predictive pretraining can provide a scalable route to foundation models for spatial transcriptomics. Existing spatial transcriptomics foundation models primarily reconstruct masked gene identities or expression values, potentially encouraging the reproduction of a...

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

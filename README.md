# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-15 11:38:58 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Riemannian Metric Matching for Scalable Geometric Modeling of Distributions](https://arxiv.org/abs/2606.14334)

**Authors**: Jacob Bamberger, Adam Gosztolai, Pierre Vandergheynst, Michael Bronstein, Iolo Jones  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.14334v1  

#### Abstract
High-dimensional datasets often concentrate near low-dimensional structures, but estimating their geometry from samples typically relies on graphs and kernels that scale poorly with dataset size and dimension. We propose Riemannian metric matching: a denoising probabilistic framework for learning th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Riemannian Metric Matching for Scalable Geometric Modeling of Distributions*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于图（graph）或核函数（kernel）的方法在估计高维数据流形的**黎曼几何结构**（如切空间、内在维度、梯度等）时面临严重瓶颈：
- **计算复杂度高**：构建 $k$-NN 图或稠密核矩阵的时间和内存开销随样本量呈超线性甚至二次增长。
- **推理不可扩展**：对新样本进行几何推断需重新计算邻域关系，无法实现高效的 **amortized inference**。
- **高维失效**：在高维空间中，欧氏距离失去判别力（curse of dimensionality），导致 $k$-NN 失效。

此外，虽然深度生成模型（如VAE、GAN、diffusion）可隐式学习几何，但其提取的几何缺乏理论保证，且依赖于雅可比（Jacobian）计算，在高维下昂贵且不稳定。

### 提出的新方法与思路
本文提出 **Riemannian Metric Matching** ——一种基于神经网络的、可扩展的黎曼几何建模框架，核心思想如下：

- **目标**：直接学习数据分布上的 **carré du champ (CDC) 算子**，该算子通过扩散几何（diffusion geometry）定义了数据流形上的黎曼度量 $g_p$。
- **关键洞察**：将原本难以处理的边际期望形式的 CDC 表达式转化为一个**条件期望**，从而构造出一个**可微分、可采样、可并行化训练的损失函数**。
- **训练机制**：采用类似去噪扩散模型的训练范式：
  - 输入为加噪样本 $Y \sim \mathcal{N}(X, \epsilon I)$，
  - 网络输出预测的 CDC 矩阵 $\Gamma_\theta(Y)$，
  - 损失函数为：
    $$
    \mathcal{L}_{\text{Riem}} = \mathbb{E}_{X,Y|X} \left[ \left\| \Gamma_\theta(Y) - \frac{(X-Y)(X-Y)^T}{2\epsilon} \right\|^2_F \right]
    $$

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可扩展性** | 支持 mini-batch 并行训练；推理为单次前向传播，与数据集大小无关（amortized inference）。 |
| **高效性** | 不需要显式构造 kernel 或 $k$-NN 图；避免昂贵的 Jacobian 计算。 |
| **理论保障** | 在无限数据极限下，当数据局部位于流形上时，所学 CDC 收敛到真实的黎曼度量（即切空间投影矩阵）。 |
| **高维适用性** | 成功应用于高达 $D=196608$ 维的图像（如 FFHQ），而传统方法在此类场景下完全失效。 |
| **灵活性** | 可结合低秩参数化（low-rank training）进一步提升效率，并支持多尺度几何建模。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型 | 维度 $D$ | 样本数 | 特点 |
|-------|------|--------|--------|------|
| **Synthetic Spheres** | 合成数据 | $D=64$, $d=8$ | $N=8k \sim 8M$ | 控制变量，用于验证几何恢复精度与可扩展性 |
| **MNIST** | 手写数字图像 | $D=784$ | 60k | 黑白图像，类别丰富，常用于几何分析 |
| **CIFAR-10** | 彩色自然图像 | $D=3072$ | 60k | 更复杂的视觉结构 |
| **CelebA** | 人脸图像 | $D=12288$ | 200k | 高维、语义结构强 |
| **FFHQ** | 高清人脸图像 | $D=196608$ | 70k | 极高维，挑战传统方法极限 |

### 实验设置与评估指标
#### 几何估计任务
- **内在维度估计**：从 CDC 矩阵特征值谱中估计局部维度（参考 Jones, 2024b）。
- **切空间恢复**：取 CDC 矩阵前 $d$ 个最大特征向量作为切空间基底。
- **评估指标**：
  - **合成数据**：使用 Frobenius 距离衡量预测切空间与真实切空间之间的误差：$\|UU^\top - \hat{U}\hat{U}^\top\|_F$。
  - **真实图像数据**：引入 **Inception Feature Stability** 作为代理指标：
    - 对输入图像沿预测切方向扰动，
    - 计算预训练 Inception 网络输出特征的变化量，
    - 变化越小说明切空间更“合理”（因为流形内变化应保持语义稳定）。

#### 插值任务
- 使用 Riemannian Optimization 解 ODE：
  $$
  \dot{\gamma}(t) = -\Gamma(\gamma(t)) \nabla f(\gamma(t)), \quad f(x)=\|x-p\|^2
  $$
- 生成从源点到目标点的插值路径。
- 对比方法：线性插值（LERP）、Stein Score、Score Jacobian。

### 基线方法对比
| 方法 | 类型 | 是否可扩展 | 是否有理论收敛保证 |
|------|------|------------|------------------|
| $k$-NN CDC Estimator | 图方法 | ❌（时间/内存超线性增长） | ✅（经典 diffusion maps） |
| Stein Score | 去噪模型衍生 | ✅ | ❌（仅粗略近似） |
| Score Jacobian | 去噪模型雅可比 | ⚠️（Jacobian 计算昂贵） | ✅（Kharitenko et al., 2025） |
| **Metric Matching (Ours)** | 神经网络回归 CDC | ✅✅✅ | ✅ |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）可扩展性测试（Fig. 2）
- **吞吐量（inferences/sec）**：
  - 当 $N=2M$ 时，metric matching 比 raw $k$-NN CDC 快 **82×**；
  - 当 $N=8M$ 时，快 **400×**。
- **切空间预测总耗时**（含特征分解）：
  - 使用 low-rank training 后，速度提升达 **359×**（$N=8M$）。
- 结论：**metric matching 在大规模数据上具有压倒性的效率优势**。

#### （2）几何估计准确性（Fig. 7, Table）
- 在合成球面上，metric matching 的切空间预测误差显著低于 $k$-NN 方法（尤其在中小规模数据集上），最高提升 **46%**。
- 特征值谱显示：metric matching 正确捕获了 $d=8$ 的主导维度，而 $k$-NN 方法误判为 9。
- 结论：**metric matching 泛化能力更强，能更准确地识别真实内在维度**。

#### （3）高维图像几何质量评估（Fig. 4, Table 1）
- **Inception Feature Stability**：
  - Metric matching 显著优于随机方向和 Stein Score；
  - 在 MNIST 和 CIFAR 上优于 Score Jacobian；
  - 在 CelebA 和 FFHQ 上，Score Jacobian 因 OOM 无法运行，而 metric matching 仍有效。
- **运行效率（Table 1）**：
  | 方法 | CIFAR-10 推理时间 | 内存占用 |
  |------|------------------|---------|
  | Score Jacobian | 992.3 ± 1.2 ms | 27.4 GB |
  | Metric Matching | **16.6 ± 3.7 ms** | **8.8 MB** |
  | Stein Score | 6.5 ± 1.8 ms | 31.6 MB |
  - **metric matching 比 Score Jacobian 快约 60×，内存少 3000×**。

#### （4）消融实验（隐含于设计中）
- **低秩训练（Low-rank training）**：
  - 理论证明：只要秩 $r \geq 2d - 1$，即可无损表示任意 $d$ 维流形的 CDC 矩阵（Theorem 4.4）。
  - 实践中使用 $r=100 \ll D=196608$ 实现高效训练。
- **均值中心化（Mean-centered loss）**：
  - 引入额外去噪网络 $(Pf)(Y)$ 替代 $f(Y)$，提高稳定性。
  - 实验中在 CelebA 和 FFHQ 上使用此变体。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **carré du champ 是连接去噪学习与黎曼几何的关键桥梁**：通过条件期望重构 CDC，实现了可微分、可扩展的几何学习。
2. ✅ **metric matching 实现了“一次性训练，永久推理”**：训练后可在常数时间内对任意新样本进行几何推断，真正实现 **amortized inference**。
3. ✅ **在高维图像数据上，metric matching 是唯一可行的几何建模方案**：相比 $k$-NN 和 Score Jacobian，它既高效又稳定。
4. ✅ **学到的几何是有意义的**：可视化切向量（如平移、边缘变化）、高质量流形内插值路径均表明模型捕捉到了数据的本质结构。

### 方法的局限性
- **依赖神经网络归纳偏置**：几何质量受网络架构、训练策略影响，目前尚无完整理论解释为何泛化良好。
- **低秩假设可能限制表达能力**：尽管 $r \geq 2d-1$ 理论上足够，但在实践中若 $r$ 设置过小可能导致欠拟合。
- **带宽选择敏感**：性能依赖于噪声尺度 $\epsilon$ 的选择，虽可通过 grid search 缓解，但仍是一个超参调优负担。

### 未来工作方向
- 🔄 **探索 metric matching 与其他生成模型（如 flow matching）的关系**，统一视角。
- 🔍 **研究神经网络如何编码几何先验**，揭示其泛化机制。
- 🧭 **开发基于 metric matching 的新型 Riemannian Optimization 算法**，用于约束优化、对抗攻击、潜空间编辑等任务。
- 🌐 **拓展至非欧嵌入空间**（如 hyperbolic space），支持层次化数据建模。

---

> **总结一句话**：  
> *Riemannian Metric Matching* 将 diffusion geometry 与 denoising learning 相结合，提供了一种**理论上严谨、实践中高效、可扩展至百万级高维数据**的几何建模新范式，为 deep learning 与 Riemannian geometry 的深度融合开辟了新路径。

</details>

---

### 2. [Orchestra-o1: Omnimodal Agent Orchestration](https://arxiv.org/abs/2606.13707)

**Authors**: Fan Zhang, Vireo Zhang, Shengju Qian, Haoxuan Li, Hao Wu, Jinyang Wu, Donghao Zhou, Zhihong Zhu, Zheng Lian, Xin Wang, Pheng-Ann Heng  
**Category**: cs.AI  
**Published**: 2026-06-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.13707v1  

#### Abstract
The recent success of agent swarms has shifted the paradigm of large language model (LLM)-based agents from single-agent workflows to multi-agent systems, highlighting the importance of agent orchestration for task decomposition and collaboration. However, existing orchestration frameworks are limit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Orchestra-o1: Omnimodal Agent Orchestration 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **LLM-based agent** 系统大多局限于单模态或双模态任务（如纯文本或图文任务），难以应对现实世界中多模态共存且交互复杂的场景（如文本、图像、音频、视频联合推理）。尽管已有 **agent swarm** 和 **orchestration** 架构提升了任务分解与协作能力，但它们在**异构模态处理**、**动态子代理专业化**和**并行执行效率**方面仍存在显著限制。

此外，当前的 **omnimodal agents** 主要分为两类：
- **Native Omnimodal Agents**：依赖单一 OLLM 同时完成感知、推理与工具调用，但在长程推理、跨模态理解等复杂任务上表现有限。
- **Orchestration-based Agents**：主代理协调多个专用子代理，但通常采用线性、串行的工作流，缺乏对模态感知的任务分解与并行调度机制。

### 提出了什么新方法或新思路
本文提出 **Orchestra-o1**，一个支持多模态协同的智能体编排框架，其核心创新包括：

#### （1）**Omnimodal Agent Orchestration Framework**
- **统一编排机制**：将高阶决策（orchestration）与低层感知/动作执行解耦，实现模块化设计。
- **模态感知的任务分解（Modality-aware Task Decomposition）**：主代理能识别输入中的模态成分，并据此生成依赖图，区分独立与依赖子任务。
- **在线子代理专业化（Online Sub-agent Specialization）**：根据子任务需求动态选择最适合的 backend 模型与工具组合。
- **并行子任务执行（Parallel Sub-task Execution）**：独立子任务可异步并发执行，显著降低延迟。

#### （2）**DA-GRPO（Decision-Aligned Group Relative Policy Optimization）**
一种高效的离线强化学习训练算法，用于训练开源主代理（如 `Orchestra-o1-8B`）：
- 不仅关注最终答案正确性，更强调**每一步编排决策的质量对齐**（如任务委派、工具选择、停止判断）。
- 引入多维度评分体系（format, action, tool, decision quality），提供密集反馈信号。
- 避免昂贵的实时执行开销，可在重建状态上进行离线评估。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **架构灵活性** | 支持混合 backend（开源 + 闭源）、灵活扩展新工具与模态 |
| **执行效率** | 并行执行减少轮次延迟，成本更低（见 Figure 5） |
| **泛化能力** | 在 OmniGAIA 上跨类别、跨难度均表现优异 |
| **训练有效性** | DA-GRPO 显著提升小型模型作为 orchestrator 的能力 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **OmniGAIA** [Li et al., 2026]：当前最具挑战性的 **omnimodal agent benchmark**，涵盖文本、图像、音频、视频等多种输入形式。
  - 包含多个主题类别：地理（Geo.）、科技（Tech.）、历史（Hist.）、金融（Fin.）、体育（Sport）、艺术（Art）、电影（Movie）、科学（Sci.）、食物（Food）
  - 分为三个难度等级：Easy / Medium / Hard

### 实验设置和评估指标
- **评估指标**：**Accuracy (%)** 为主要指标，衡量最终答案是否准确。
- **最大编排轮数**：10 轮
- **子代理最大步数**：30 步
- **工具集**：
  - 感知类：Image Analysis, Audio Analysis, Video Analysis
  - 动作类：Web Search, Page Visit, Code Execution
- **上下文长度限制**：prompt 最大 24,576 tokens，response 最大 4,096 tokens

### 基线方法对比
#### 开源模型组（Open-Source Agentic Models）
- Qwen2.5-Omni 系列
- Baichuan-Omni-1.5-8B
- MiniCPM-O-2.6-8B
- Ming-Lite/Flash-Omni
- LongCat-Flash-Omni
- OmniAtlas 系列

#### 闭源模型组（Proprietary Agentic Models）
- Gemini-2.5/3.0 Flash/Lite/Pro 系列
- AOrchestra-GPT-5（最强开源编排基线）

#### 编排框架对比
- **ReAct-style 单代理系统**：直接使用 GPT-5 或 Qwen3-8B 执行所有操作
- **AOrchestra**：当前最先进的开源编排框架，作为主要对比对象

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | 整体 Accuracy (%) |
|------|------------------|
| **Orchestra-o1-GPT-5 (ours)** | **72.8** ✅ |
| Gemini-3-Pro | 65.2 |
| AOrchestra-GPT-5 | 40.0 |
| **Orchestra-o1-8B (ours, open-source)** | **30.0** ✅ |
| OmniAtlas-Qwen3-30B-A3B | 20.8 |

> ⬆️ Orchestra-o1-GPT-5 比第二好的方法 **Gemini-3-Pro 提升 10.3% 绝对精度**  
> ⬆️ Orchestra-o1-8B 比此前最佳开源模型 **提升 9.2%**

### 与基线方法的对比结果
#### 类别级表现（Table 1）
- Orchestra-o1-GPT-5 在几乎所有类别中都取得最优成绩，尤其在：
  - **地理（69.4 → 72.5）**
  - **技术（75.8）**
  - **历史（83.8）**
  - **体育（63.9）**
  - **艺术（69.7）**
  - **电影（73.1）**
  - **科学（83.3）**
- 表明该框架具有广泛适用性，而非特定领域优化。

#### 难度级表现（Figure 4）
| 难度 | Orchestra-o1-GPT-5 | AOrchestra-GPT-5 | 提升 |
|------|--------------------|------------------|------|
| Easy | 80.3% | 45.1% | +35.2% |
| Medium | 75.0% | 40.0% | +35.0% |
| Hard | 56.4% | 32.1% | +24.3% |

> 💡 特别是在 **Hard 任务** 上的显著提升说明：**依赖感知分解 + 迭代证据聚合** 对复杂多跳推理至关重要。

### 消融实验结果（Ablation Studies）

#### Ablation on Agent Harness（Figure 6）
比较 ReAct-GPT-5 与 Orchestra-o1-GPT-5：
- ReAct-GPT-5：53.9%
- Orchestra-o1-GPT-5：72.8%
➡️ **+18.9% 提升来自编排架构本身**，而非仅靠 backend 强大。

#### Ablation on Post-training Recipe（Table 2）
以 Qwen3-8B 为主代理的逐步增强效果：
| 设置 | Accuracy (%) |
|------|--------------|
| ReAct (no orchestration) | 12.5 |
| Orchestra-o1 (no training) | 26.3 |
| + SFT | 28.6 |
| + Vanilla GRPO | 27.7 |
| + **DA-GRPO (ours)** | **30.0** |

> 🔍 结论：**DA-GRPO 是关键**，相比 vanilla GRPO 更有效，说明**细粒度决策对齐优于仅奖励最终结果**。

---

## 4. 关键结论和发现

### 主要发现
1. **分离编排与执行是构建强大 omnimodal agent 的有效范式**：
   - 将感知、工具使用交给专业 sub-agent，主代理专注高层规划，提升可扩展性与鲁棒性。
2. **并行执行带来显著效率增益**：
   - Proposition 1 从理论上证明，在条件独立任务下，并行调度可实现接近 $ K_t $ 倍的速度提升。
3. **DA-GRPO 可有效训练小型开源主代理**：
   - 即使使用 8B 模型，也能通过高质量轨迹监督成为高效 orchestrator。
4. **Orchestra-o1 具备更强的成本效益**（Figure 5）：
   - 总成本 **$341.6** vs AOrchestra 的 **$565.7**
   - 总耗时 **61.73h** vs AOrchestra 的 **85.30h**
   - 更高 accuracy + 更低成本 + 更短时间 → 更优性价比

### 方法的局限性
1. **系统复杂度较高**：
   - 需维护 sub-agent 历史、工具 schema、backend 配置、异步执行逻辑等，工程门槛上升。
2. **训练未端到端联合优化**：
   - 当前 DA-GRPO 仅优化主代理策略，sub-agent backends 固定不变。
3. **依赖高质量合成数据**：
   - 数据构造依赖 GPT-5 生成参考轨迹，可能引入偏差。

### 未来工作方向
1. **端到端联合训练**：同时优化主代理、子代理策略与工具选择行为。
2. **更多实际应用场景拓展**：
   - 如 audio-video 协同 vibe coding
   - voice-guided computer-use tasks
3. **轻量化部署方案**：降低 Orchestra-o1 在边缘设备上的运行开销。
4. **开放生态建设**：推动社区共建工具集、模态插件与 sub-agent 库。

---

> 📚 **代码与模型已开源**：  
> GitHub: [https://github.com/zfkarl/Orchestra-o1](https://github.com/zfkarl/Orchestra-o1)  
> Hugging Face: [https://huggingface.co/Karl28/Orchestra-o1-8B](https://huggingface.co/Karl28/Orchestra-o1-8B)

</details>

---

### 3. [Realizing Native INT8 Compute for Diffusion Transformers on Consumer GPUs: A Fused INT8 GEMM Kernel for Ideogram 4.0](https://arxiv.org/abs/2606.14598)

**Authors**: Ali Asaria, Tony Salomone, Deep Gandhi  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.14598v1  

#### Abstract
Post-training INT8 (W8A8) quantization of diffusion transformers is widely deployed as a speed optimization, yet on consumer Ampere GPUs it is frequently slower than the FP8 and NF4 alternatives it is meant to beat. We trace this to a software artifact: the production "INT8" forward quantizes weight...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Realizing Native INT8 Compute for Diffusion Transformers on Consumer GPUs: A Fused INT8 GEMM Kernel for Ideogram 4.0*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前在消费级 Ampere 架构 GPU（如 RTX 3090）上部署的 **post-training INT8（W8A8）量化**扩散模型（diffusion transformers），虽然名义上是“INT8”计算，但实际上是一种 **fake-quant**（伪量化）策略：
- 权重和激活被量化为 INT8 后，**立即反量化回 bf16**，然后执行标准的 `bf16` GEMM 运算；
- 导致 **未真正利用 GPU 的 INT8 Tensor Cores**，从而丧失了硬件应有的计算优势；
- 结果反而比 FP8 和 NF4 更慢。

该现象导致“INT8”本应加速推理，却成为最慢的变体。

### 🚀 提出的新方法
作者提出一种 **单个融合的 Triton INT8 GEMM 内核**，实现真正的原生 INT8 计算：
- 执行 `int8×int8→int32` 的矩阵乘法，直接运行于 Ampere 的 `mma.s8` Tensor Cores 上；
- 将以下操作全部融合进 GEMM 的 **epilogue** 阶段：
  - per-token 动态 activation dequantization
  - per-channel weight dequantization
  - bias 加法
- 替换原有“量化 → 反量化 → bf16 matmul”的路径，变为 **单一 kernel 调用**；
- 对每个实际出现的 GEMM 形状进行 **autotuning**（共五种形状）以最大化性能。

> 🔧 创新本质：将 LLM 推理中成熟的 **fused dequantization** 思路首次成功应用于 diffusion transformer，并适配 consumer Ampere 的硬件特性。

### ⚖️ 相比现有方法的优势
| 方面 | 传统“INT8”（fake-quant） | 本文方法（fused INT8） |
|------|--------------------------|------------------------|
| 计算路径 | 量化 → 反量化 → bf16 matmul | 原生 int8×int8→int32 在 Tensor Cores 上 |
| 是否使用 INT8 TC | ❌ 否 | ✅ 是 |
| kernel 数量 | 多步（至少两步） | 单一 fused kernel |
| 性能表现 | 比 FP8/NF4 更慢 | 显著更快（端到端 ~9–10% 加速） |
| 显存占用 | 更高（峰值 26.7GB） | 更低（峰值 23.4GB） |
| 硬件利用率 | 低下 | 充分释放 INT8 计算潜力 |

---

## 2. 核心实验方法和设置

### 📚 数据集与 Prompt 设置
- **模型**：Ideogram 4.0 —— 一个 9.3B 参数的 flow-matching diffusion transformer（DiT）；
- **Prompt 集固定不变**，确保可比性：
  - Calibration set: `n = 128` prompts（独立于评测）
  - Quality benchmark: `n = 200` prompts
  - Text-rendering benchmark: `n = 100` prompts（含 63 个 OCR 目标）
- 所有生成任务均采用相同配置：
  - 固定 seed
  - 48 denoising steps
  - 分辨率：768px 和 1024px

### ⚙️ 实验设置
- **主平台**：RTX 3090（consumer Ampere，sm_86）
- **对比平台**：A100（datacenter Ampere）、B200（Blackwell）用于验证硬件特异性
- **GPU 健康检查**：所有测量前进行 bf16 TFLOPS 检测，排除因热节流导致的性能偏差（健康卡 ≈65.5 TFLOPS，劣化卡仅 ~8.1 TFLOPS）

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **性能** | per-GEMM latency（μs）、end-to-end s/image、speedup ratio |
| **资源** | peak VRAM usage、required GPU count |
| **质量** | PickScore、CLIPScore（proxy for image quality & prompt alignment） |
| **正确性** | 与 `torch._int_mm` 的 cosine similarity、NaN 检查 |
| **边界指标** | OCR/NED（text rendering 能力，未重新测量） |

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **FP8** | 使用 dequant-to-bf16 路径，在 Ampere 上无原生 FP8 支持 |
| **NF4** | 4-bit 权重量化（如 bitsandbytes），单卡运行 |
| **Fake-quant INT8** | 本文改进前的旧版：量化后反量化至 bf16，不启用 INT8 TC |
| **Fused INT8 (ours)** | 本文提出的方法：原生 INT8 GEMM + 融合反量化 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）Per-GEMM 层面加速（vs. bf16）
所有五个 DiT GEMM 形状均为 **compute-bound**，因此 INT8 加速有效。

| GEMM Shape | Speedup (vs. bf16) |
|------------|--------------------|
| qkv        | 2.79–3.46×         |
| attn-out   | 2.86–3.76×         |
| ffn-up     | 2.78–4.18×         |
| ffn-down   | 2.94–3.51×         |
| llm-proj   | 2.95–3.17×         |

> 💡 平均约 **3.5× per-GEMM 加速**；相比 unfused int8 路径快 **4–8×**

#### （2）端到端推理速度（768px）
| Variant | s/image | Speedup | PickScore | CLIPScore |
|--------|--------|--------|-----------|-----------|
| Fake-quant INT8 | 107.06 | 1.00× | 20.14 | 21.76 |
| **Fused INT8 (ours)** | **97.79** | **~1.095× (~9.5%)** | **21.22** | **24.35** |

- 图像质量 **持平或略优**（基于 n=4 的 point estimate）
- 实际 GEMM 占前向传播时间约 **12%**，符合 Amdahl 定律预测

#### （3）1024px 生成（核心目标达成）
| Variant | s/image | GPUs | Notes |
|--------|--------|------|-------|
| FP8 (prior) | 172.9 | 2 | dequant-to-bf16 |
| NF4 (prior) | 164.5 | 1 | — |
| Fake-quant INT8 | 184–185 | 2 | 无法单卡运行 |
| **Fused INT8 (ours)** | **156.49** | **1** | **峰值显存 23.40 GB，完整生成 4/4 图像** |

✅ **结论**：
- 成功将 1024px 生成从双卡降至 **单 RTX 3090**
- 比 FP8 快 **~9.5%（~16 秒）**
- 比 NF4 快 **~4.9%（~8 秒）**
- **INT8 从最慢变为最快方案**

#### （4）消融实验（Ablations）
| Configuration | s/image | Why No Gain |
|--------------|--------|------------|
| Fused INT8 (baseline) | 97.79 | — |
| + `torch.compile` | 97.74 | graph-breaks at custom linears |
| + SageAttention INT8 | 97.82 | head dim 256 > 128 limit → fallback to FP16 |

> ❌ 两个自然扩展方向均失败，说明当前优化已接近瓶颈。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **当前“INT8”是伪量化**：主流部署中的 INT8 实际并未使用 INT8 Tensor Cores，仅为内存格式转换，造成额外开销。
2. **真正的原生 INT8 可带来显著收益**：通过 fused Triton kernel 实现 `int8×int8→int32` 计算，可在 consumer Ampere 上实现 **2.8–4.2× per-GEMM 加速**。
3. **端到端性能提升明显**：在 768px 下提速 ~9–10%，在 1024px 下实现 **单卡生成（156.5s）且快于 FP8/NF4**。
4. **显存占用下降使其可行**：峰值 VRAM 从 ~26.7GB（OOM on 24GB）降至 23.4GB，使 1024px 单卡运行成为可能。
5. **硬件特异性极强**：
   - ✅ 在 **consumer Ampere（RTX 3090）** 上显著胜出
   - ❌ 在 **A100 / B200** 上落后（因这些芯片已有高速 bf16/FP8 路径）

### ⚠️ 局限性
1. **收益受限于 GEMM 在整体中的占比**：DiT 中 linear layers 仅占前向约 12%，其余时间消耗在 attention、norm、sampling 等模块。
2. **无法与 `torch.compile` 兼容**：自定义 kernel 引发 graph breaks，阻碍进一步图级融合优化。
3. **attention 未量化**：现有 INT8 attention 库（如 SageAttention）不支持 head dim=256，无法启用。
4. **OCR/text rendering 未重新验证**：最敏感的质量指标 OCR/NED 未在 fused 版本上重测，依赖原始 recipe 的鲁棒性。
5. **结果不具备泛化性**：仅针对 Ideogram 4.0 和 consumer Ampere，不能推广至其他模型或架构。

### 🔮 未来工作方向
1. **开发 graph-safe 的 custom op**：使 fused kernel 能通过 `torch.compile` 编译，避免 graph breaks。
2. **构建支持大 head dimension 的 INT8 attention kernel**：覆盖 256+ head dim，以加速 attention 子系统。
3. **进行直接的双卡对比测试**：消除当前 1-GPU vs 2-GPU 基线间的不对称性。
4. **探索更广泛的 kernel 优化空间**：例如使用 CUTLASS/cuBLASLt 实现更高性能的 INT8 GEMM baseline。
5. **扩展至其他 diffusion 架构和 quantization recipes**：验证方法的通用性。

---

## 总结一句话
> 本文揭示了消费级 GPU 上“INT8 量化”普遍存在的 **fake-quant** 陷阱，并通过一个 **fused Triton INT8 GEMM kernel** 实现了真正的原生 INT8 计算，在 Ideogram 4.0 上将 INT8 从最慢变最快，**首次实现了 1024px 单卡生成**，同时明确了其 **hardware-specific applicability**：仅适用于缺乏高效 bf16/FP8 路径的 consumer Ampere 设备。

</details>

---

### 4. [Abstracting Cross-Domain Action Sequences into Interpretable Workflows](https://arxiv.org/abs/2606.14654)

**Authors**: Gaurav Verma, Scott Counts  
**Category**: cs.AI  
**Published**: 2026-06-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.14654v1  

#### Abstract
Sequential or time-stamped interaction logs provide objective records of digital application usage, yet their granularity and noise often obscure meaningful insights into people's work. Such insights are essential for improving digital products in ways grounded in real-world user interactions. Prior...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Abstracting Cross-Domain Action Sequences into Interpretable Workflows*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **UI interaction logs**（用户界面交互日志）虽然记录了用户在数字应用中的详细操作行为（如点击、滚动、输入等），但这些日志通常具有以下问题：
- **粒度过高且噪声大**：一个高层任务（如“撰写报告”）可能由数百个低级动作组成，难以直接理解用户意图。
- **缺乏语义解释能力**：传统方法（如频繁项集挖掘、LSTM建模）难以泛化到不同领域，且依赖大量标注数据进行 fine-tuning。

因此，如何从原始、嘈杂、跨域的动作序列中自动抽象出可解释的 **high-level activities**（高层活动）和 **workflows**（工作流）是一个关键挑战。

### 提出的新方法：WorkflowView
作者提出 **WorkflowView** —— 一种基于 **Large Language Models (LLMs)** 的分层抽象框架，用于将低层次的 action sequences 转换为高层次、可解释的任务描述与行为模式。

#### 方法核心思想
- **分层推理架构**（Hierarchical Abstraction）
  - **Layer 1**: 将原始 action sequence 转换为详细的自然语言描述（Natural Language Description）。
  - **Layer 2**: 在上一层基础上推断用户的 high-level task 或 activity。
  - **Layer 3 (Optional)**: 对高层活动进行分类（如预测是否辍学、归类文档编辑行为）。
- **无需训练**：完全基于 **zero-shot 或 few-shot prompting**，不依赖任务特定的训练数据。
- **模块化设计**：各层输出可复用，支持多种下游任务（如任务重建、分类、聚类）。

### 相比现有方法的优势
| 维度 | 传统方法 | WorkflowView |
|------|--------|-------------|
| 泛化能力 | 弱，需 domain-specific fine-tuning | 强，zero-shot 跨域适用 |
| 数据需求 | 需要数千条标注样本 | 仅需 0~5 个 in-context 示例 |
| 可解释性 | 黑箱模型，输出难解读 | 输出为自然语言，易于理解和审计 |
| 部署成本 | 高（需训练+部署专用模型） | 低（直接调用通用 LLM API） |

> ✅ **核心创新**：首次系统性地将 LLM 的 zero/few-shot 推理能力应用于跨域、多任务的用户行为日志分析，并验证其有效性与通用性。

---

## 2. 核心实验方法和设置

### 使用的数据集
论文在三个差异显著的领域进行了实验，验证方法的通用性：

| 领域 | 数据集 | 描述 |
|------|-------|------|
| **Web 浏览任务** | Mind2Web (Deng et al., 2023) | 包含 2,022 条浏览器操作序列，每条对应一个自然语言任务描述（如“查找最便宜的本田思域”）。 |
| **MOOC 学习行为** | Feng et al. (2019) | 来自 44,008 名学生、247 门课程的交互日志，共 67,699 条记录，75.8% 为 dropout。目标是预测辍学。 |
| **文档协作行为** | Microsoft Word Telemetry | 收集约 50,000 名美国用户使用 Copilot 后的行为日志（匿名、无文本内容），分析前后 30 分钟内的 action sequences。 |

### 实验设置与评估指标

#### （1）Web Task Reconstruction（零样本任务重建）
- **任务**：给定 action sequence，生成对应的 task description。
- **评估方式**：
  - 使用 `text-embedding-ada-002` 编码生成描述与真实描述，计算 **cosine similarity**。
  - 报告 **MRR** 和 **Recall@K**（K=1,3,5,10），候选集分为全局（global）和网站内（website-specific）两种设定。
- **设置**：Zero-shot，使用 GPT-4o。

#### （2）MOOC Dropout Prediction（少样本辍学预测）
- **任务**：基于 N 小时前的操作序列预测是否 dropout。
- **设置**：
  - 时间窗口：start time ∈ {1,6,12,18,24} 天，end time ∈ {1,6,12,18,24} 小时。
  - Few-shot 示例数：{0,1,3,5,10,20} per class。
- **评估指标**：**Weighted F1**, Precision, Recall。
- **模型**：GPT-4o。

#### （3）Document Workflow Analysis（AI 工具使用场景分析）
- **任务**：发现并分类用户在接受 AI 输出前后的行为模式。
- **方法**：
  - 使用 WorkflowView 生成 activity summary。
  - 结合 **TnT-LLM (Wan et al., 2024)** 自动发现类别标签。
  - 第三层进行 multi-class classification。
- **输出形式**：聚合统计（百分比）、可视化图表。

### 基线方法对比
#### （1）Web Task Reconstruction 基线
| 模型 | 类型 | 是否微调 |
|------|------|---------|
| LSTM seq2seq | 序列到序列模型 | 是（在 Mind2Web 上训练） |
| BERT seq2seq | 预训练编码器 + 解码器 | 是 |

#### （2）MOOC Dropout Prediction 基线
| 模型 | 来源 | 特点 |
|------|------|------|
| CFIN-en | Feng et al. (2019) | 使用上下文特征 + DNN + XGBoost ensemble |
| CNN-LSTM | Yang et al. (2024) | 深度学习模型，基于周级别特征 |
| CNN-LSTM Bi-Att | Yang et al. (2024) | 加入注意力机制 |

---

## 3. 主要实验结果和性能指标

### （1）Web Task Reconstruction 结果
- **平均语义相似度**：**0.911 (±0.042)** → 表明生成的任务描述与真实任务高度一致。
- **检索性能优异**：

| 指标 | Global Setting | Website-Specific Setting |
|------|----------------|----------------------------|
| MRR | 0.90 (±0.08) | 0.94 (±0.06) |
| Recall@1 | 0.86 (±0.13) | 0.92 (±0.09) |
| Recall@5 | 0.96 (±0.05) | 0.99 (±0.03) |

> 🔍 定性示例显示，即使 ground truth 更具体，生成描述仍能准确捕捉核心意图（见 Table 2 & Table 11）。

#### 与基线对比（Table 4）
| Model | MRR | Recall@1 |
|-------|-----|----------|
| LSTM seq2seq | 0.54 | 0.49 |
| BERT seq2seq | 0.68 | 0.65 |
| **WorkflowView** | **0.90** | **0.86** |

✅ **优势明显**：零样本 WorkflowView 显著优于需要全量训练的传统 seq2seq 模型。

---

### （2）MOOC Dropout Prediction 结果
- **最佳性能**（start=6天, end=24小时, 5-shot）：
  - **Weighted F1 = 0.90**
  - Precision = 0.81, Recall = 0.97
- **Few-shot 敏感性分析**（Figure 3）：
  - Zero-shot: F1 = 0.84
  - 5-shot: F1 = 0.90（提升显著）
  - >10-shot 性能下降 → 可能因 context length 过长导致“lost in the middle”。

#### 与监督基线对比（Table 5）
| Model | Weighted F1 |
|-------|-------------|
| CFIN-en (Feng et al.) | 0.90 |
| CNN-LSTM Bi-Att | 0.87 |
| **WorkflowView (5-shot)** | **0.90** |

✅ **关键发现**：仅用 **5 个示例**，WorkflowView 即达到与使用数万标注样本的监督模型相当的性能。

---

### （3）Document Workflow 分析结果（案例研究）
- 发现了 **15 类常见文档编辑行为**（见 Table 3），如：
  - Active editing of content
  - Formatting text and layout
  - Reviewing comments
  - Using AI features
- **关键洞察**（Figure 4）：
  - **接受 AI 输出前后，“主动内容编辑”占比均为最高（15%）**。
  - 接受后，“格式化”和“布局优化”行为比例上升 → 用户倾向于调整 AI 输出以匹配原文风格。
  - “Final edits before closing/printing” 在之后增加 → AI 输出常作为最终润色的一部分。

> 📊 这些发现可用于改进产品设计，例如提供更智能的格式建议或上下文感知的 AI 回应整合工具。

---

## 4. 关键结论和发现

### 主要结论
1. ✅ **LLM 可有效抽象跨域 action sequences**  
   WorkflowView 在三个完全不同领域的任务中均表现出色，证明了 LLM 在理解非语言结构化行为数据方面的强大潜力。

2. ✅ **Zero/Few-shot 设置下性能媲美监督模型**  
   - Web 任务重建：similarity 达 0.91，Recall@1 = 86%
   - MOOC 辍学预测：**仅用 5 个示例即达 F1=0.90**，与数千样本训练的模型持平。

3. ✅ **支持隐私保护与匿名分析**  
   所有分析可在不访问用户文本内容的前提下完成，符合企业级隐私要求。

4. ✅ **模块化与易扩展性强**  
   可灵活适配任务描述、分类、聚类等多种下游任务，且支持结合其他 LLM 工具（如 TnT-LLM）实现自动类别发现。

---

### 局限性（Limitations）
1. **Action Name 必须有意义**  
   若日志中 action 名称为模糊标识（如 `Action1`, `EventID_23`），则无法有效推理。要求 logging infrastructure 具备良好的语义命名规范。

2. **Token 效率有待优化**  
   当前采用直接文本表示，未压缩或 chunking，可能导致高 token 开销。未来可探索基于时间窗口的聚合策略。

3. **依赖外部 LLM API**  
   当前依赖 GPT-4o 等闭源模型；虽测试了 Phi-4 和 gpt-oss-20b（性能接近），但在资源受限场景仍需轻量化方案。

4. **缺乏严格消融实验**  
   未对比单层 prompt vs 多层 hierarchical prompting 的效果差异，尽管引用了 “least-to-most prompting” 理论支持其合理性。

---

### 未来工作方向
1. **Multimodal Extension**  
   结合 UI 截图（screenshots）与 action logs，构建更精准的实时意图识别系统（见 Figure 5）。

2. **嵌入 Logging Infrastructure**  
   将 LLM 推理下沉至底层日志采集系统，实现实时、低延迟的行为洞察。

3. **On-device Inference for Privacy**  
   在设备端运行小型 LLM 进行初步抽象，仅上传安全的 high-level summaries，增强用户信任。

4. **Pre-training on Action Sequences**  
   构建大规模跨应用 action sequence corpus，对 LLM 进行领域适应 pre-training，进一步提升 zero-shot 泛化能力。

---

> 💡 **总体评价**：该论文展示了 **LLM 作为通用行为理解引擎** 的巨大前景，推动了从“数据驱动”向“语义驱动”的产品迭代范式转变，具有重要的理论价值与工业落地意义。

</details>

---

### 5. [Efficient On-Device Diffusion LLM Inference with Mobile NPU](https://arxiv.org/abs/2606.13740)

**Authors**: Tuowei Wang, Yanfan Sun, Ju Ren  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.13740v1  

#### Abstract
Diffusion large language models (dLLMs) accelerate generation by denoising multiple tokens in parallel, making them attractive for latency-sensitive mobile inference. However, repeated denoising introduces substantial computation on smartphones. Mobile neural processing units (NPUs) offer high-throu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient On-Device Diffusion LLM Inference with Mobile NPU**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前智能手机上的大语言模型（LLM）推理主要依赖 **autoregressive decoding**，逐个生成 token，导致输出延迟随长度线性增长，难以满足移动端低延迟需求。虽然 **diffusion LLMs (dLLMs)** 通过并行去噪多个 token 来加速生成，但在移动设备上仍面临以下挑战：
- **重复计算开销大**：每次去噪都需对整个序列进行 Transformer 计算。
- **KV cache 复用困难**：token 状态在迭代中不断变化，缓存易失效。
- **Mobile NPU 资源利用率低**：NPU 擅长密集矩阵运算，但 dLLM 推理动态性强，存在负载不均、内存碎片、频繁 remapping 等问题。

### **提出的新方法：LLADA.CPP**
本文提出了 **LLADA.CPP**，首个面向 Mobile NPU 的 dLLM 推理框架，通过算法与系统协同设计，将 dLLM 的并行特性与 NPU 的计算能力高效结合。

#### **三大核心技术**
1. **Multi-Block Speculative Decoding（多块推测解码）**
   - 在当前 block 后期有效负载下降时，**推测性地引入未来 block 的 token** 进行并行处理，提升 NPU 利用率。
   - 保持原始的 block 提交顺序，确保语义正确性。

2. **Dual-Path Progressive Revision（双路径渐进修订）**
   - 将 token 分为 **invisible (IV)**、**visible (V)** 和 **stable (S)** 三种状态。
   - 可见但不稳定 token 的更新由 **CPU 路径异步处理**，避免中断 NPU 密集计算。
   - 采用延迟合并策略，避免细粒度同步。

3. **Swap-Optimized Memory Runtime（交换优化内存运行时）**
   - 利用计算图信息进行 **图引导的缓冲区映射**，减少不必要的 VA 映射和交换。
   - 采用 **双缓冲流水线机制**，重叠数据传输与 NPU 计算，隐藏数据移动延迟。

### **相比现有方法的优势**
- 首次实现 **dLLM 与 Mobile NPU 的高效协同**，将重复计算转化为可利用的并行负载。
- 不仅提升速度，还 **保持生成质量**，解决了推测与修订带来的潜在准确性下降问题。
- 支持跨设备、跨模型部署，具备良好通用性。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **GSM8K**：小学数学题，测试推理能力。
- **BoolQ**：布尔问答。
- **ARC-C**：科学类选择题。
- **HellaSwag**：常识推理填空。

> 所有任务均使用 200 个样本子集进行评估。

### **实验设置**
- **硬件平台**：
  - OnePlus 12 (SM8650)
  - OnePlus Ace5 Pro (SM8750)
  - OnePlus 15 (SM8850)
- **模型**：
  - 主要模型：**LLaDA-8B-Instruct**（dLLM）
  - 对照模型：**Llama-3-8B-Instruct**（autoregressive）
  - 额外验证：**Dream-7B**（不同架构 dLLM）
- **量化方式**：Q4_0 低比特量化以适配手机内存。

### **评估指标**
- **端到端延迟（end-to-end latency）**：从输入到完整输出的时间。
- **生成质量（accuracy）**：按任务特定评分标准计算。
- **平均功耗与能耗（energy per request）**。
- **消融实验**：逐步启用各组件，分析其独立贡献。

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **CPU baseline** | Vanilla dLLM 解码，全在 CPU 上执行 |
| **CPU + prefix KV cache reuse** | CPU 上启用前缀 KV 缓存复用 |
| **NPU baseline** | 仅将密集计算卸载至 NPU，保留原始去噪调度 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 在 **OnePlus Ace5 Pro (SM8750)** 上：
  - LLADA.CPP 相比 **CPU baseline** 实现 **17×–42× 的端到端延迟降低**。
  - 相比 **CPU + KV cache 复用** 基线，仍达到 **34×–42× 加速**。
- 生成 128 个 token 时，LLADA.CPP 在 SM8750 上仅需约 **16ms**，而 CPU 基线超过 **600ms**。

### **与基线方法对比**
| 方法 | 相对加速（vs CPU） | 是否优于 autoregressive？ |
|------|------------------|------------------------|
| CPU baseline | 1.0× | 否 |
| CPU + KV cache | ~5× | 否 |
| NPU baseline | ~8–9× | 否 |
| **LLADA.CPP** | **17×–42×** | **是（最高达 3.9× 快于 Llama）** |

> 图 1 显示，在 GSM8K 等任务上，LLADA.CPP 相比 Llama-3-8B 最高快 **3.9×**。

### **消融实验结果（Table 6）**
在 GSM8K 数据集上对 128-token 生成进行分解：

| 方法 | 延迟 (ms) | 相对增益 | 总加速倍数 |
|------|----------|---------|-----------|
| CPU | 2996.2 | 1.00× | 1.00× |
| + prefix KV cache | 607.0 | 4.94× | 4.94× |
| + NPU offload | 341.2 | 1.78× | 8.78× |
| + SOMR | 87.0 | 3.92× | 34.44× |
| + MBSD | 71.6 | 1.22× | 41.87× |
| + STS/SLS/SCR | 16.1 | — | **~186×** |

> 可见，**Multi-Block Speculative Decoding** 和 **Staged Token Stabilization** 是最大贡献者。

---

## **4. 关键结论和发现**

### **主要发现**
1. **dLLM 的并行性天然契合 Mobile NPU**，但需系统级优化才能释放潜力。
2. **LLADA.CPP 成功将 dLLM 的“重复计算”劣势转化为“并行负载”优势**，显著降低延迟。
3. **双路径设计有效分离密集与稀疏计算**，兼顾效率与质量。
4. **内存管理是关键瓶颈**，swap-optimized runtime 可支持更长上下文生成。

### **方法的局限性**
- 当前实现基于 **Qualcomm Hexagon NPU**，虽具代表性，但向其他厂商 NPU（如 MediaTek, Samsung）迁移需适配。
- **推测解码可能引入额外错误传播风险**，尽管实验显示质量可控。
- 对 **极短序列（<32 tokens）** 加速效果有限，因初始化开销占比较高。

### **未来工作方向**
- 扩展至更多 dLLM 架构（如 AR-Diffusion hybrid models）。
- 支持动态 block size 与 adaptive denoising step。
- 探索 NPU-CPU-GPU 三端协同推理。
- 结合 **early-exit / early-skipping** 技术进一步减少冗余计算。

---

> **总结**：LLADA.CPP 是首个将 dLLM 与 Mobile NPU 深度协同的推理框架，通过 **multi-block speculation**、**dual-path revision** 和 **swap-optimized memory** 三大技术，实现了高达 **42× 的端到端加速**，同时保持生成质量，为移动端低延迟、隐私保护的 AI 应用铺平道路。

</details>

---

### 6. [Provably Safe, Yet Scalable Reinforcement Learning](https://arxiv.org/abs/2606.14536)

**Authors**: Kai S. Yun, Zeyang Li, Navid Azizan  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.14536v1  

#### Abstract
Safe reinforcement learning (RL) aims to learn policies that optimize rewards while satisfying constraints. Predominant approaches rely on soft-constrained policy optimization, which has achieved empirical success but does not provide formal safety guarantees for the learned policy. In contrast, met...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Provably Safe, Yet Scalable Reinforcement Learning

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **Safe RL** 方法面临两大瓶颈：
- **软约束方法**（如 Lagrangian-based）虽然在高维系统中表现良好，但无法提供**形式化的安全保证**，部署时仍可能发生灾难性失败。
- **基于证书的方法**（如 CBF、Barrier Functions）虽能提供严格的安全性证明，但其核心依赖于显式构造**控制不变集**（control-invariant set），该过程计算复杂度随状态维度指数增长，且通常导致行为过于保守。

### 提出的新方法：PS2-RL 框架
本文提出了 **PS2-RL (Provably Safe, yet Scalable RL)**，一个两阶段框架，旨在实现**可证明安全**与**高可扩展性**的统一。

#### 核心创新点
1. **隐式构建控制不变集 (Implicit Invariant Set)**：
   - 不再直接合成全局不变集，而是通过一个**学习到的备份策略**（learned backup policy）对系统动力学进行前向积分，从而在线生成一个**隐式的控制不变集**（implicit control-invariant set）。
   - 该集合由 **Backup Control Barrier Function (BCBF)** 框架保证为控制不变。

2. **两阶段架构 (Two-Phase Architecture)**：
   - **Phase I: 安全到达策略训练 (Safe-Arrival Policy Training)**
     - 引入**安全到达价值函数**（Safe-Arrival Value Function），这是一个基于指示器（indicator-based）的目标函数。
     - 该函数将“安全到达目标集”这一事件建模为一个首次命中（first-hit）问题，并通过折扣奖励鼓励策略尽快、安全地到达。
     - 优化此函数可得到时间最优的备份策略，从而**最大化隐式不变集的体积**。
   - **Phase II: 通过控制不变层进行端到端训练 (End-to-End Training with CIL)**
     - 将第一阶段学到的备份策略固定，构建一个**控制不变层**（Control-Invariant Layer, CIL）。
     - CIL 是一个**可微分的投影层**，它将任意 RL 策略的输出投影到由 BCBF 框架定义的**安全动作空间**（BCBF-admissible set）内。
     - 通过反向传播，可以在保持安全性的同时，对主策略进行端到端的优化。

3. **通用性与表达能力**：
   - PS2-RL **不依赖特定的 RL 算法**，可以作为插件集成到任何现有的 RL 训练流水线中。
   - 理论上证明了 CIL 层具有**通用近似能力**（universal approximation），即投影操作不会牺牲策略网络的表达能力。

### 相比现有方法的优势
- **可证明的安全性**：继承了 BCBF 框架的形式化安全保证。
- **高可扩展性**：避免了全局不变集的显式计算，计算复杂度仅随状态维度线性增长。
- **低保守性**：通过学习到的备份策略，相比解析方法（如 LQR）能生成大得多的安全区域。
- **高性能**：在保证 100% 安全的前提下，实现了优于所有基线的性能。

---

## 2. 核心实验方法和设置

### 数据集与任务
实验在两个机器人控制任务上进行：
1. **Unicycle Lane-Keeping**：一个低维（3维状态）的单车模型车道保持任务，参考轨迹故意超出安全边界。
2. **Quadrotor Powerloop**：一个高维（10维状态）四旋翼飞行器的特技飞行动作跟踪任务，要求完成一个垂直环并同时执行 360° 翻滚，极具挑战性。

### 实验设置
- **安全集**：分别定义了车道边界和天花板高度等物理安全约束。
- **奖励函数**：均为负加权跟踪误差，旨在最小化与参考轨迹的偏差。
- **训练与评估**：所有方法均使用 10 个随机种子进行训练和评估，每个种子评估 1000 个回合。

### 评估指标
- **Tracking Performance (RMSE)**：位置、速度、姿态等的均方根误差。
- **Safety Performance**：
  - `Total Safety (%)`：安全回合的比例。
  - `Per-seed Safety (%)`：每个种子的安全率（IQM 和 95% CI）。
  - `Worst Viol. (m)`：最严重的单次违反量。

### 基线方法对比
共比较了 7 种基线方法，分为三类：
1. **基于惩罚的 RL (Penalty-based)**：
   - `SAC-Penlow`, `SAC-Penhigh`：使用不同强度的奖励惩罚来处理不安全行为。
2. **基于约束的策略优化 (CMDP)**：
   - `SAC-Lagrangian`, `CPO`：通过拉格朗日乘子或信任域方法优化期望成本约束。
3. **基于验证证书的 Safe RL (Verified Certificate)**：
   - `CBF-RL`：在训练时使用 CBF 进行过滤和奖励塑形，但部署时移除滤波器。
   - `MPS`：模型预测屏蔽，一种运行时切换机制。
   - `PS2ABP`：PS2-RL 的变体，其中 Phase I 使用解析备份策略（如 LQR）而非学习策略，用于隔离学习效果。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### Unicycle Lane-Keeping (Table 1)
| Method | y-RMSE (↓) | v-RMSE (↓) | φ-RMSE (↓) | Total Safety (%) |
| :--- | :--- | :--- | :--- | :--- |
| **PS2sA** | **0.52** | 0.83 | **0.11** | **100%** |
| PS2ABP | 0.97 | 0.43 | 0.15 | 100% |
| SAC-Penhigh | 1.11 | 1.85 | 0.15 | 100% |
| MPS | 0.68 | 0.48 | 0.17 | 97.8% |

- **PS2sA** 在所有 100% 安全的方法中取得了**最佳的跟踪性能**。
- 相比使用解析备份的 `PS2ABP`，`PS2sA` 将横向跟踪误差（y-RMSE）从 0.97m 降低到 **0.52m**，降幅约 46%，证明了学习备份策略的有效性。
- `SAC-Penhigh` 虽然安全，但性能远不如 `PS2sA`，表现出极大的保守性。
- `CBF-RL` 和 `MPS` 在采样数据部署下未能达到 100% 安全。

#### Quadrotor Powerloop (Table 2)
| Method | p-RMSE (↓) | v-RMSE (↓) | φ-RMSE (↓) | Total Safety (%) |
| :--- | :--- | :--- | :--- | :--- |
| **PS2sA** | **0.60** | **0.96** | **0.45** | **100%** |
| PS2ABP | 1.39 | 1.74 | 0.82 | 100% |
| MPS | 2.00 | 4.07 | 1.41 | 100% |
| CBF-RL | 2.29 | 4.36 | 2.06 | 99.8% |

- `PS2sA` 再次在 100% 安全的方法中取得了**压倒性的性能优势**。
- 相比 `PS2ABP`，`PS2sA` 的位置 RMSE 降低了约 **57%**（1.39m -> 0.60m），速度 RMSE 降低了约 **45%**。
- 相比 `MPS`，`PS2sA` 的位置 RMSE 降低了约 **70%**。
- 所有非 PS2-RL 的基线方法要么不安全，要么性能极差。

### 消融实验结果 (Ablation Study)

在 Quadrotor 任务上进行了关键消融实验（Table 12）：

| Method | p-RMSE | Safety (%) |
| :--- | :--- | :--- |
| **PS2sA (Full)** | **0.5275** | **100%** |
| PS2sA w/o CIL | 0.7395 | **0.0%** |
| Vanilla + CIL | 0.9283 | 100% |

- **移除 CIL (w/o CIL)**：一旦移除控制不变层，安全率从 100% **骤降至 0%**，证明了 CIL 是安全保证的关键组件。
- **仅后置 CIL (Vanilla + CIL)**：将 CIL 仅作为后置滤波器应用在一个未经安全训练的策略上，虽然能保证 100% 安全，但跟踪性能比端到端训练的 `PS2sA` 差了约 **43%**。
- **结论**：**端到端的可微分训练**对于在保证安全的前提下实现高性能至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **成功平衡了安全与性能**：PS2-RL 首次在高达 10 维的状态空间中，实现了**100% 的可证明安全**与**卓越的任务性能**，解决了传统方法在安全性和性能之间的根本权衡。
2. **学习备份策略是关键**：相比于解析备份策略（如 LQR），学习到的备份策略能显著扩大安全区域，这是性能提升的主要来源。
3. **可微分投影是核心**：将安全约束嵌入一个可微分的投影层，使得策略能够学习如何在安全边界内高效地优化性能，而不仅仅是被一个非微分的“盾牌”纠正。
4. **框架具有普适性**：PS2-RL 不依赖于特定的 RL 算法，是一个通用的、可插入的框架。

### 方法的局限性
- **依赖系统模型**：PS2-RL 的形式化安全保证需要一个**控制仿射的解析动力学模型**或一个**可微分的仿真器**。在未知或不确定的动力学下，难以提供同等保证。
- **计算开销**：尽管可扩展，但 Phase I 的训练和 CIL 层中的 QP 求解仍带来一定的计算负担。
- **感知约束**：当前框架主要处理状态空间中的硬约束，尚未直接整合来自视觉等感知输入的复杂安全约束。

### 未来工作方向
- 将 PS2-RL 扩展到**不确定性动力学**和**部分可观测环境**。
- 探索更高效的**可微分求解器**以降低 CIL 层的计算延迟。
- 将框架应用于**真实硬件**上的机器人控制任务。
- 整合**感知模块**，实现端到端的、从像素到安全动作的决策。

</details>

---

### 7. [A Temporal Planning Framework for Disruption Aware Dynamic Route Optimization in Heterogeneous Railway Systems](https://arxiv.org/abs/2606.14582)

**Authors**: Pollob Chandra Ray, Sabah Binte Noor, Fazlul Hasan Siddiqui  
**Category**: cs.AI  
**Published**: 2026-06-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.14582v1  

#### Abstract
Efficient route optimization play a vital role in ensuring both safety and punctuality in railway operations. It is very crucial particularly in heterogeneous multi-gauge railway networks with varying train speed, stopping pattern, infrastructure compatibility constraints increase coordination compl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Temporal Planning Framework for Disruption Aware Dynamic Route Optimization in Heterogeneous Railway Systems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
该论文针对**异构多轨距铁路系统**（heterogeneous multi-gauge railway systems）中的动态路径优化与扰动管理问题，特别是以下挑战：
- **轨距兼容性约束**（gauge compatibility constraints）：不同轨距列车只能在匹配的轨道上运行，限制了路径选择。
- **单线铁路系统的资源冲突**：所有列车共享轨道，需协调频繁的道岔切换（turnout switching）。
- **随机扰动事件**：如轨道封锁、列车停运、速度减慢、机车故障等，导致时刻表偏离。
- **现有研究缺乏可执行操作序列**：多数方法仅生成高层时刻表，未提供低层控制指令（如道岔操作），依赖人工决策，增加安全风险。

### 🚀 提出的新方法与新思路
提出了一种基于 **Temporal Planning** 的框架——**DART Framework**（Disruption Aware Railway Temporal Planning），其核心创新包括：

1. **将铁路调度建模为 Temporal Planning 问题**
   - 使用 **PDDL 2.1** 对铁路系统进行形式化建模，显式编码时间、数值变量、持续动作（durative actions）。
   - 支持并发动作、时间窗口、资源占用等复杂时序逻辑。

2. **集成扰动感知的恢复机制**
   - 定义四类典型扰动并设计对应的恢复动作：
     - Blocked Track → `clear-blocked-track`
     - Blocked Train → `resolve-train-blockage`
     - Speed Slowdown → 动态调整行驶时间
     - Engine Failure → 调度辅助机车 + `attach-engine` + `drive-assisted-train`

3. **生成带时间戳的可执行操作序列**
   - 输出不仅是列车时刻表，还包括：
     - 列车移动 (`drive-train`)
     - 道岔切换 (`turnout`)
     - 乘客上下车 (`board-passengers`)
     - 故障恢复动作
   - 实现从“计划”到“执行”的闭环，提升自动化与安全性。

4. **构建首个面向异构铁路系统的标准化基准测试集**
   - 包含 **200 个实例**（100 正常 + 100 扰动）
   - 规模从 **小网络（3列火车）扩展至超大规模（120列火车，1000个轨道点）**
   - 支持可扩展性与鲁棒性分析。

### 🔍 相比现有方法的优势
| 维度 | 现有方法（如 MILP, MaxSAT, Metaheuristics） | 本文方法（DART + Temporal Planning） |
|------|--------------------------------------------|----------------------------------------|
| **建模粒度** | 高层时刻表优化 | 显式建模低层操作（如 turnout 控制） |
| **扰动处理** | 多为事后重调度 | 内嵌扰动恢复策略，支持主动响应 |
| **输出形式** | 时间表（timetable） | 可执行的动作序列（action sequence） |
| **安全性保障** | 依赖后处理验证 | 在规划中强制满足互斥、碰撞避免等约束 |
| **通用性与可复用性** | 模型特定 | 基于标准 PDDL，易于迁移与验证 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **自研 Benchmark Problem Set**：
  - 共 **200 个问题实例**，分为两类：
    - **Nominal-operation instances**（p1–p100）：无扰动的标准调度场景。
    - **Disrupted-operation instances**（p101–p200）：引入多种扰动的真实压力场景。
  - 四级规模递增：Small (S), Medium (M), Large (L), Very Large (VL)
  - 最大实例包含：
    - 120 列列车
    - 1000 个 track points
    - 100 个 junctions
    - 多达 108 个并发扰动（如封锁段、故障机车等）

### ⚙️ 实验设置
- **Planners**：
  - **POPF**：基于前向搜索与 TRPG 启发式的高效 temporal planner。
  - **OPTIC**：POPF 的扩展版本，支持混合整数规划与偏好优化。
- **Plan Validator**：
  - 使用 **VAL** 工具对生成的 plan 进行语义正确性验证（precondition satisfaction, mutex, goal achievement）。
- **硬件环境**：
  - Intel Core i7 第七代，32GB RAM，Ubuntu 24.04 LTS
  - 单实例最大运行时间：30 分钟

### 📈 评估指标
| 类型 | 指标名称 | 描述 |
|------|--------|------|
| **通用规划指标** | Makespan | 总完成时间（最后一项动作结束时间） |
| | Plan Length | 计划中动作总数 |
| | Plan Generation Time | 求解器生成 plan 所需时间 |
| | Success Rate | 成功求解的问题比例 |
| **领域专用延迟指标** | Total Delay | 实际到达时间 vs 理想无扰动时间之差总和 |
| | Slowdown Delay | 因限速造成的额外延迟 |
| | Blockage Delay | 因轨道/列车封锁等待的时间 |
| | Engine Failure Delay | 故障恢复过程中的延误（派遣+连接） |

> 注：所有指标均按 scale 分组报告平均值 ± 标准差（每组 10 个实例）

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Table 4 和 Figures）

#### 📌 性能概览（两 planner 结果一致）
| Scale | Nominal Makespan (min) | Disrupted Makespan (min) | Total Delay (min) | Plan Length |
|-------|-------------------------|----------------------------|--------------------|-------------|
| S1    | 30.53                   | 42.52                      | 28.00              | ~15         |
| M2    | 34.53                   | 49.05                      | 88.48              | ~80         |
| L3    | 34.58                   | 53.01                      | 170.32             | ~150        |
| VL3   | 35.16                   | 57.21                      | **377.41**         | ~250        |

- **成功率**：199/200 实例成功求解（唯一失败为最大规模 VL3 中的 p200，因超时）。
- **验证结果**：所有生成 plan 均通过 VAL 验证，无 precondition violation 或 mutex 冲突。

#### 📊 延迟分解分析（Figure 4）
- **Slowdown Delay**：占总延迟 **60.1%**，是最大来源，尤其在 VL3 达 **69.8%**。
- **Engine Failure Delay**：占比 **30.4%**
- **Blockage Delay**：仅 **9.5%**，说明 planner 能有效绕行或协调，减少堵塞影响。

> 表明：**速度降低是最难规避的扰动类型**，因其广泛存在且难以通过路径替代缓解。

#### ⏱️ 计算效率（Figures 5 & 6）
- **Nominal instances**：
  - 平均耗时从 S1 的 **~0.06 秒** 上升至 VL3 的 **~45 秒**
- **Disrupted instances**：
  - 耗时显著更高，VL3 平均达 **619 秒（约10分钟）**
  - 曲线呈指数增长趋势，接近 30 分钟上限，反映组合复杂度剧增。

#### 🔍 相关性分析（Tables 5 & 6）
- **Spearman 相关系数** 显示：
  - 系统规模（trains, track points, junctions）与计算时间呈 **完美正相关（ρ = 1.000）**
  - 扰动数量与 total delay 也高度相关（ρ ≥ 0.976），证明 benchmark 设计合理且模型响应可预测。

### 🔁 与基线方法的对比结果
- **未直接与其他范式（如 MILP、MaxSAT）比较**，但通过文献综述指出：
  - 现有方法无法输出可执行动作序列；
  - 多数忽略 gauge compatibility 和 turnout dynamics；
  - 缺乏统一扰动建模框架。
- 本方法优势体现在：
  - **语义完整性**：plan 可直接用于控制系统输入。
  - **自动化程度高**：无需人工干预即可完成扰动恢复。
  - **可验证性强**：借助 VAL 实现形式化验证，增强可信度。

### ❌ 消融实验（Ablation Study）
- 文中**未明确开展消融实验**（如移除某类扰动或动作的影响分析）。
- 但通过分组实验间接体现模块作用：
  - Nominal vs Disrupted 对比显示扰动带来的 delay 增加；
  - 不同 scale 下性能变化反映 scalability；
  - 延迟分解揭示各类扰动的实际影响权重。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Temporal Planning 能有效建模复杂铁路系统**
   - 成功将 gauge compatibility、turnout control、passenger service、disruption recovery 等要素统一建模于 PDDL 2.1 框架下。
2. **DART 框架具备强健的扰动应对能力**
   - 在高达 108 个并发扰动下仍能生成合法、可行的操作序列。
3. **生成的 plan 是可执行且安全合规的**
   - 所有 plan 经 VAL 验证无误，确保实际部署可行性。
4. **系统性能随规模和扰动强度单调可预测地增长**
   - 支持用于大规模系统的能力评估与容量规划。
5. **Slowdown 是最主要的延迟源**
   - 提示未来应加强动态限速预测与路径重规划策略。

### ⚠️ 方法的局限性
1. **假设前提较强**：
   - 扰动持续时间已知（a priori knowledge），现实中往往不确定。
   - 忽略加减速过程，采用匀速运动模型。
   - 系统完全可观测、确定性环境。
2. **求解时间随规模快速增长**
   - 最大实例接近 30 分钟极限，难以满足实时应急响应需求。
3. **未考虑更复杂的扰动类型**
   - 如脱轨（derailment）、信号系统故障、天气连锁效应等。
4. **缺乏在线 replanning 机制**
   - 当前为离线 batch 模式，尚未实现 streaming data 下的增量更新。

### 🔮 未来工作方向
1. **支持不确定性下的 Plan Repair**
   - 引入 probabilistic 或 robust planning 方法处理未知扰动。
2. **集成 real-time data streams**
   - 接入 ATS、ATC、IoT sensor 数据实现动态感知与响应。
3. **扩展扰动类型建模**
   - 加入 derailment、火灾、自然灾害等极端事件。
4. **开发轻量化 planner 或 heuristic 加速**
   - 提升在线求解效率，适用于紧急调度场景。
5. **与数字孪生系统结合**
   - 将 DART 集成进铁路运营仿真平台，实现“模拟-决策-执行”闭环。

---

> **总结一句话**：  
> 本文提出的 **DART 框架**首次将 **Temporal Planning** 成功应用于**异构多轨距铁路系统的扰动感知动态路径优化**，不仅生成高质量、可执行的操作计划，还建立了首个公开的大规模 benchmark，为智能铁路调度提供了**可验证、可扩展、安全可靠的新范式**。

</details>

---

### 8. [Towards Direct Latent-Space Synthesis for Parallel Branches in LLM-Agent Workflows](https://arxiv.org/abs/2606.14672)

**Authors**: Shikun Liu, Mufei Li, Dongqi Fu, Haoyu Wang, Yinglong Xia, Hong Li, Hong Yan, Pan Li  
**Category**: cs.AI  
**Published**: 2026-06-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.14672v1  

#### Abstract
Large language models increasingly serve as execution engines for agentic systems, yet they still consume context through a sequential text interface. This creates a mismatch with modern structured agent workflows, in which independent branches explore subtasks, retrieve evidence, or generate candid...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Towards Direct Latent-Space Synthesis for Parallel Branches in LLM-Agent Workflows

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 **LLM-Agent 工作流**（如 multi-agent systems、tree-of-thoughts、tool-using agents）常采用“并行分支 → 合成”（parallel-then-synthesize）的 DAG 结构。然而，当前的 LLM 仍通过**顺序文本接口**（text serialization）处理上下文，即把多个 worker agent 的输出拼接成一个长文本前缀供 synthesizer 处理。

这种做法存在两个核心问题：
- **冗余计算**：worker 的输出已被解码一次，但在合成时又被 synthesizer 重新 prefill，造成重复的 KV cache 构建。
- **结构丢失**：并行分支的独立性在文本拼接中被破坏，导致 synthesizer 难以有效区分和整合不同来源的信息。

### 提出了什么新方法或新思路
本文提出 **Parallel-Synthesis**，一种**即插即用**（plug-and-play）的框架，使 synthesizer 能够**直接消费并行 worker agents 生成的 KV cache**，而非依赖文本拼接。

其核心技术组件包括：
- **Positional Re-encoding**：将各 worker 输出的 RoPE 位置对齐到统一的 post-branch 位置区间，使其在逻辑上表现为从同一节点分叉的并行路径。
- **Cache Mapper**：一个可学习的模块，基于 worker 输出长度和 worker 数量等元信息，对每个 worker 的 KV cache 进行仿射变换（affine mapping），以校准不同上下文下生成的 cache。
- **Synthesizer LoRA Adapter**：一个轻量级的 LoRA 适配器，使 synthesizer 能够理解并推理来自多个非连续 cache 上下文的信息。

### 相比现有方法的优势
- **效率更高**：避免了文本拼接后的重复 prefill，显著降低 **Time-to-First-Token (TTFT)**。
- **结构保留**：显式保留了并行分支的 DAG 结构，有助于 synthesizer 更准确地进行证据比较、冲突解决和聚合判断。
- **即插即用**：不改变 worker 端行为，仅需在 synthesizer 端加载适配器即可使用。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖 **9 个下游数据集**，涵盖四大类任务：
| 类别 | 数据集 |
|------|--------|
| **数学推理** | AIME 2024, AIME 2025, GSM8K |
| **科学问答** | GPQA, MedQA |
| **代码生成** | HumanEval-Plus, MBPP-Plus |
| **工具使用与多智能体诊断** | GAIA (Levels 1–3), MARBLE Database |

> 所有数据集均未出现在训练集中，用于测试**跨域泛化能力**。

### 实验设置和评估指标
- **模型基础**：Qwen3-14B 作为 backbone。
- **并行 worker 数量**：除 MARBLE Database 使用 5 个外，其余均为 3 个。
- **评估指标**：
  - **Accuracy**：标准准确率。
  - **Time-to-First-Token (TTFT)**：衡量 synthesizer 的响应延迟，反映效率提升。
- **训练策略**：采用双轨后训练（post-training）并合并：
  - **Track 1**：通用适应，使用 WildChat、UltraChat、FLAN 等构建并行上下文。
  - **Track 2**：蒸馏自 text-based synthesis 的轨迹（来自 BrowseComp），提升复杂推理能力。
  - 最终通过 **checkpoint merging**（加权平均）融合两个轨道的参数。

### 基线方法对比
| 基线类型 | 方法 |
|---------|------|
| **单路径 & 投票** | `Single`（单 agent）、`Voting`（多数投票） |
| **文本拼接** | `Text-Serialization`（标准文本拼接） |
| **RAG 风格缓存复用** | `APE`, `CacheBlend`, `KVLINK`（adapted for agent synthesis） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 表 1：数学、代码、科学 QA 准确率（Accuracy）
| Method | AIME 2024 | AIME 2025 | GSM8K | HumanEvalPlus | MBPPPlus | GPQA | MedQA |
|--------|-----------|-----------|-------|----------------|----------|------|-------|
| Text-Serialization | 63.33 | 23.33 | 92.80 | 90.24 | 81.75 | 50.00 | 82.72 |
| **Parallel-Synthesis** | **63.33** | **46.67** | **94.69** | **90.85** | **80.42** | **52.02** | **83.58** |

> ✅ 在 7/9 数据集上**持平或超越**文本拼接方法，尤其在 AIME 2025、GPQA 等高难度推理任务上优势明显。

#### 表 2：GAIA 与 MARBLE Database 正确数
| Method | GAIA L1 | GAIA L2 | GAIA L3 | MARBLE DB |
|--------|--------|--------|--------|------------|
| Text-Serialization | 24 | 22 | 2 | 33 |
| **Parallel-Synthesis** | 23 | 19 | 2 | **36** |

> ✅ 在 MARBLE Database 上达到最佳表现（36/100），表明其在**多智能体诊断**场景中具备更强的证据整合能力。

### 与基线方法的对比结果
- **效率对比**（图 3）：
  - Parallel-Synthesis 将 TTFT **降低 2.5×–11×**，相比 Text-Serialization。
  - 即使与高效的 CacheBlend 相比，仍快约 **2×**。
- **质量对比**：
  - 显著优于所有 RAG 风格缓存复用方法（APE, CacheBlend, KVLINK）。
  - 在多数任务上优于 `Voting`，说明其并非简单答案聚合，而是进行**深度推理合成**。

### 消融实验结果

#### 表 3：不同训练策略与模块消融
| 变体 | AIME 2024 | HumanEvalPlus | GPQA | GAIA L1 | MARBLE DB |
|------|-----------|----------------|------|--------|------------|
| No training | 10.00 | 38.41 | 15.15 | 16 | 22 |
| Track 1 only | 46.67 | 91.46 | 55.56 | 21 | 30 |
| Track 2 only | 53.33 | 88.41 | 51.52 | 21 | 44 |
| Sequential tuning | 36.67 | 76.22 | 49.49 | 23 | 25 |
| **Full (merged)** | **63.33** | **90.85** | **52.02** | **23** | **36** |

> 🔍 发现：
> - 仅靠 inference-time calibration（如 APE）效果差，**必须进行 post-training**。
> - **Track 1** 擅长代码类任务，**Track 2** 擅长复杂推理。
> - **顺序微调会遗忘早期能力**，而 **checkpoint merging 能更好保留互补优势**。

#### 表 4：传递给 synthesizer 的信息粒度影响（GAIA L1）
| 信息类型 | 正确数 | TTFT (s) |
|--------|--------|---------|
| Final model output | 23 | 0.3769 |
| Each-turn model output | 18 | 0.3788 |
| Full trajectory | 26 | 0.6847 |

> 🔍 发现：
> - 传递完整轨迹最准但最慢。
> - **仅传递最终输出**在准确率与效率间取得最佳平衡，是默认选择。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **KV cache 可作为更高效的合成接口**：direct latent-space synthesis 不仅可行，且在多个复杂任务上**匹配甚至超越文本拼接**。
2. **合成不仅是聚合，更是推理**：Parallel-Synthesis 能利用 worker 的中间推理链、证据质量、冲突信号等进行综合判断，而非简单投票。
3. **结构化通信优于扁平化通信**：保留并行分支的 DAG 结构有助于 synthesizer 更精准地进行证据溯源与冲突解决。
4. **效率提升显著**：通过避免重复 prefill，TTFT 缩短 **2.5×–11×**，对实时系统意义重大。

### 方法的局限性
- 当前方法假设 worker 输出为**语义完整的单元**（如候选解、摘要），若 worker 输出碎片化，则效果可能下降。
- 对于高度复杂的 DAG 结构（如嵌套分支、动态调度），尚未验证其扩展性。
- 依赖高质量的 worker cache，若 worker 推理错误，错误可能被保留在 cache 中。

### 未来工作方向
- 扩展至更大规模的 agent workflow 数据集进行 post-training。
- 探索更复杂的 cache fusion 机制，支持动态权重分配或注意力路由。
- 将该范式推广至其他模态或多跳推理场景。
- 研究如何在 synthesizer 端实现**主动缓存查询**（active cache probing），而非被动接收。

> 💡 **总体评价**：Parallel-Synthesis 为 LLM-Agent 系统提供了一种**更原生、更高效、更具结构性**的合成范式，是迈向“真正结构化 agent 工作流”的重要一步。

</details>

---

### 9. [Decoupled Mixture-of-Experts for Parametric Knowledge Injection](https://arxiv.org/abs/2606.14243)

**Authors**: Baoqing Yue, Weihang Su, Qingyao Ai, Yichen Tang, Changyue Wang, Jiacheng Kang, Jingtao Zhan, Yiqun Liu  
**Category**: cs.CL  
**Published**: 2026-06-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.14243v1  

#### Abstract
Knowledge injection aims to equip large language models (LLMs) with external, domain-specific, or time-sensitive knowledge. Existing approaches typically face a trade-off between flexibility and integration: retrieval-augmented generation keeps knowledge outside the model but only provides prompt-le...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Decoupled Mixture-of-Experts for Parametric Knowledge Injection**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）在预训练后其参数化的知识是静态的，导致在面对**领域特定**或**时效性强**的知识查询时容易产生幻觉或过时回答。现有的知识注入方法面临以下权衡：
- **Retrieval-Augmented Generation (RAG)**：将外部知识保留在模型之外，仅通过提示（prompt）进行浅层增强，灵活性高但集成深度有限。
- **Post-training-based 方法（如 SFT、LoRA）**：将新知识编码进共享参数中，可能导致**灾难性遗忘**（catastrophic forgetting）、**知识冲突**（knowledge conflict）以及高昂的更新成本。

### **提出的新方法：DMoE**
本文提出了 **Decoupled Mixture-of-Experts (DMoE)**，一种用于参数化知识注入的模块化架构，具有以下核心创新：

#### ✅ **完全解耦的设计**
- **专家（Experts）和路由器（Router）均从基础模型中解耦**，不修改原始模型参数。
- 外部知识被转换为独立可更新的轻量级专家模块（基于 LoRA 等 PEFT 技术），存储于模型外。

#### ✅ **不确定性感知路由（Uncertainty-Aware Routing）**
- 在生成过程中，通过计算 token uncertainty（TU，即预测分布熵）判断是否需要调用专家。
- 当 TU 超过阈值时，才触发路由机制，避免不必要的开销。

#### ✅ **最终层 FFN 附加策略（Final-Layer FFN Attachment）**
- 专家仅附加到 Transformer 的**最后一层前馈网络（FFN）**上。
- 这一设计保证了 **KV-cache 的可重用性**，显著降低自回归推理中的延迟和内存消耗。

### **相比现有方法的优势**
| 维度 | RAG | Post-training (e.g., SFT-LoRA) | DMoE |
|------|-----|-------------------------------|-------|
| **知识集成深度** | 浅层（prompt-level） | 深层（parameter-level） | ✅ 深层 |
| **模块化与可更新性** | 高（无需改参） | 低（共享参数易冲突） | ✅ 高（独立增删改专家） |
| **推理效率** | 受限于重复检索与长上下文处理 | 高（静态模型） | ✅ 支持 KV-cache 重用，高效动态增强 |
| **知识隔离性** | 强（外部存储） | 弱（混合写入） | ✅ 强（每个专家对应独立知识单元） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个典型的 **knowledge-intensive QA 基准**上进行评估：
- **ComplexWebQuestions (CWQ)**：开放域组合推理任务。
- **HotpotQA**：多跳推理，需整合多个文档信息。
- **Quasar-T**：基于大规模语料的事实型问答。
- **StrategyQA**：隐式多跳推理，依赖世界知识推断。

### **实验设置与评估指标**

#### **评估指标**
- **有效性（Effectiveness）**：
  - EM（Exact Match）
  - F1 分数
  - StrategyQA 使用 ACC（Accuracy）
- **效率（Efficiency）**：
  - 平均推理时间（Time/s）
  - GPU 内存占用（GPU Memory/GB）

#### **基线方法对比**
所有方法基于两个基础模型构建：
- `Llama-3.2-1B-Instruct`
- `Qwen2.5-1.5B-Instruct`

对比的基线包括：
| 方法 | 类型 | 特点 |
|------|------|------|
| **Basic-RAG** | Retrieval-based | 单次检索，拼接至 prompt |
| **FLARE** | Dynamic RAG | 解码中动态检索，无法复用 KV-cache |
| **PRAG** | Parametric RAG | 结合检索与参数化适配器（adapter） |
| **SFT-LoRA** | Post-training | 在整个知识库上训练单一 LoRA 适配器 |

#### **实现细节**
- **专家构造**：每篇维基百科段落生成一个 LoRA 专家（rank=4, α=16），训练数据由原文本及其 paraphrase 和 QA 对构成。
- **路由机制**：使用 **BM25** 构建倒排索引，根据当前上下文作为查询检索最相关的 top-k 专家（默认 k=3）。
- **触发条件**：当 token uncertainty > 2.0 时激活路由。
- **硬件环境**：单张 NVIDIA A100 80GB GPU。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| 方法 | CWQ-F1 ↑ | HotpotQA-F1 ↑ | Quasar-T-F1 ↑ | StrategyQA-ACC ↑ |
|------|----------|----------------|----------------|--------------------|
| **DMoE (Llama3.2-1B)** | **0.3479** | **0.2553** | **0.3658** | **0.5667** |
| Basic-RAG | 0.2384 | 0.2463 | 0.3513 | 0.4333 |
| FLARE | 0.3154 | 0.1303 | 0.2190 | 0.5367 |
| PRAG | 0.3284 | 0.1427 | 0.2514 | 0.5600 |
| SFT-LoRA | 0.3092 | 0.1325 | 0.2481 | 0.5533 |

> ✅ **DMoE 在 14 项指标中有 11 项取得最优或并列最佳**，尤其在 F1 指标上全面领先。

### **与基线方法的对比结果**
- **优于 RAG 系列**：DMoE 显著优于 Basic-RAG 和 FLARE，尤其是在多跳推理任务（HotpotQA）上表现突出。
- **优于参数化适配器方法**：相比 PRAG 和 SFT-LoRA，DMoE 实现更深的知识集成且无遗忘风险。
- **效率远超动态 RAG**：相比 FLARE，DMoE 推理速度快约 **3×**，GPU 内存减少 **1.6–1.9×**。

### **消融实验结果**

#### 🔍 **专家放置位置的影响（Table 3）**
- 将专家插入中间层或多层会破坏 KV-cache，导致性能下降。
- **仅在最后一层 FFN 添加专家（ONLY-LAST）效果最好**，兼顾准确率与效率。

#### 🔍 **DMoE vs. 传统 MoE 架构（Table 4）**
| 方法 | CWQ-F1 | Time(s) ↓ | GPU(GB) ↓ |
|------|--------|-----------|------------|
| **DMoE (Llama)** | **0.3479** | **2.67** | **7.24** |
| Coupled MoE (OLMoE) | 0.2638 | 20.02 | 26.08 |

> ❗ DMoE 不仅更有效，而且推理速度提升 **7.5 倍以上**，内存占用仅为 1/3.6。

#### 🔍 **其他消融分析（Appendix）**
- **Top-k 敏感性（Figure 5）**：性能对激活专家数量（k=1~5）鲁棒，说明路由稳定。
- **触发阈值敏感性（Figure 6）**：在较宽范围内性能稳定，允许灵活调整效率/效果平衡。
- **LoRA 参数敏感性（Table 8）**：默认配置 `(r=4, α=16)` 已足够，更大容量未带来一致收益。
- **专家库规模影响（Table 9）**：即使缩减至 1/10 规模，性能仍保持稳健，表明方法具备良好扩展性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **DMoE 实现了参数级知识注入与模块化更新的统一**：
   - 既实现了比 RAG 更深的知识融合，又避免了 post-training 方法的知识干扰问题。
2. **KV-cache 兼容性是高效推理的关键**：
   - 最终层专家附加策略使得缓存得以复用，大幅降低动态增强的成本。
3. **不确定性驱动的稀疏激活机制有效且高效**：
   - 仅在必要时激活专家，避免冗余计算。
4. **解耦设计支持真正的“即插即用”知识管理**：
   - 新知识可通过训练独立专家加入；旧知识可直接删除而不影响主干模型。

### **方法的局限性**
- **依赖高质量的知识分割**：如何将外部知识合理划分为“知识单元”尚无统一标准。
- **路由精度受限于词法匹配（BM25）**：对于语义复杂或抽象概念可能不如 dense retriever 准确（尽管实验显示 BM25 已足够强）。
- **目前仅验证于中小规模模型**：在超大规模模型上的扩展性和稳定性有待进一步验证。

### **未来工作方向**
- 探索更精细的 **知识粒度划分策略**（如句子级、事件级）。
- 设计 **端到端可学习的轻量级神经路由器**，同时保持解耦特性。
- 将 DMoE 应用于 **持续学习**（continual learning）和 **个性化知识定制** 场景。
- 扩展至多模态领域，实现跨模态知识专家的动态调用。

---

> 📌 **总结一句话**：  
> **DMoE 提出了一种解耦、缓存友好、按需激活的 MoE 架构，在不牺牲推理效率的前提下实现了模块化、可更新的参数级知识注入，为 LLM 的可持续知识增强提供了新范式。**

</details>

---

### 10. [VaultxGPU: GPU-Accelerated Blockchain Consensus](https://arxiv.org/abs/2606.14007)

**Authors**: Samuel Taiwo Fatunmbi, Om Amit Gandhi, Luke Logan  
**Category**: cs.DC  
**Published**: 2026-06-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.14007v1  

#### Abstract
Blockchain consensus mechanisms based on Proof-of-Work consume significant energy, with Bitcoin alone estimated at approximately 150 TWh per year. Proof-of-Space reduces this cost by replacing repeated computation with storage, but plot generation remains bottlenecked by CPU hashing throughput. Prio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*VaultxGPU: GPU-Accelerated Blockchain Consensus*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
区块链共识机制中，**Proof-of-Work (PoW)** 虽安全但能耗极高（如比特币年耗电约150 TWh），而 **Proof-of-Space (PoSp)** 通过用存储替代计算来降低运行时能耗。然而，PoSp 的一个关键瓶颈是**plot 文件生成阶段**，该过程依赖大量 Blake3 哈希计算，受限于 CPU 的并行处理能力，导致矿工难以快速部署参与网络。

尽管已有高效 CPU 实现（如 *VaultX*），但在高端硬件上仍无法充分发挥性能潜力。本文旨在解决这一“一次性高成本”问题——**加速 PoSp 的 plot 生成过程**。

---

### 🚀 提出的新方法与创新点

1. **GPU 加速的 PoSp Plotter：VaultxGPU**
   - 将原本基于多线程 CPU 的 *VaultX* 架构扩展至 GPU 平台，首次实现完整的 GPU 端 PoSp 绘图流水线。
   - 完全在 GPU 上执行哈希生成、排序与匹配阶段，最小化主机交互。

2. **双后端支持（CUDA + SYCL）**
   - 实现 **CUDA 后端**（针对 NVIDIA GPU）
   - 实现 **SYCL 后端**（兼容 AMD 和 Intel GPU），提升跨平台可移植性。
   - 共享核心 Blake3 kernel 代码，仅适配内存模型与原子操作语义差异。

3. **关键优化设计**
   - **Blake3 Hashing Kernel 重构**：每个 nonce 分配一个独立 GPU thread，在寄存器内完成完整哈希计算，消除线程间依赖。
   - **Sort + Match 阶段融合**：将插入排序与配对检测合并为单个 kernel，全部运行于 shared/local memory 中，避免中间全局内存访问开销。
   - **全表驻留 VRAM**：Table-1（含 $2^k$ 条目）全程保留在 GPU 显存中，防止 PCIe 数据往返拖累性能。

4. **与现有系统的兼容性**
   - 输出 plot 文件格式与原始 CPU 版本完全兼容，无需转换即可被现有 prover 使用。

---

### ⚖️ 相比现有方法的优势

| 方面 | VaultxGPU | 传统方案（如 Chia/VaultX-CPU） |
|------|-----------|-------------------------------|
| 性能 | 最高达 **59.2×** 单线程 CPU 加速比 | 多线程优化有限，扩展性差 |
| 成本效益 | 使用更低成本 GPU 实现更高吞吐 | 高核数 CPU 成本高昂（> \$10,000） |
| 可扩展性 | 支持 K=31 大规模 plot，且随问题规模增长表现稳定 | 随 K 增大效率下降明显 |
| 跨平台性 | SYCL 支持 AMD/Intel/NVIDIA | 多为专有 CUDA 或纯 CPU 实现 |

---

## 2. 核心实验方法和设置

### 📊 实验设置

#### 测试参数范围：
- **K-values**: 从 `K=27` 到 `K=31`，每步问题规模翻倍。
- 主要关注 `K=31` 场景（典型大规模应用）。

#### 硬件配置：
| 后端 | GPU 设备 | CPU 主机 |
|------|----------|---------|
| **CUDA** | Tesla V100 ×8（16/32 GB VRAM） | 64–192 核服务器级 CPU |
| **SYCL** | AMD Vega 20 / Intel Arc A770（各 16 GB VRAM） | 6-core 桌面级 PC |
| **CPU Baseline** | N/A | 64-core 服务器，测试 1~384 threads |

#### 基线方法对比：
- **Naive Single-threaded CPU**：串行版本，作为基准（runtime = 2688 秒 @ K=31）
- **Optimized Multi-threaded CPU**：使用 384 threads 的 VaultX-CPU 最优配置
- **Chia Reference Plotter**：原始 PoSp 实现，用于上下文参考（非直接对比）

#### 评估指标：
- **总绘图时间（Total Plotting Time）**
- **Speedup 相对于单线程 CPU**
- **Pipeline 阶段时间分解**（Hash Gen / Sort&Match / Disk Write）
- **Scaling Behavior**：$T(k+1)/T(k)$ 比值分析
- **Parallel Efficiency**：实际加速 vs 理想线性加速

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（@ K=31）

| 方法 | 运行时间 | Speedup（vs 单线程 CPU） |
|------|--------|--------------------------|
| **SYCL GPU**（AMD/Intel） | **45.4 秒** | **59.2×** ✅（最佳） |
| **CUDA GPU**（NVIDIA） | 53.8 秒 | 50.0× |
| **Best CPU**（384 threads） | 52.2 秒 | 51.5× |
| **SYCL CPU Fallback** | 99.2 秒 | 27.1× |
| **Single-thread CPU** | 2688 秒（≈44.8 分钟） | 1.0× |

> 🔥 **SYCL GPU 不仅超越所有 CPU 配置，还优于 CUDA 实现**

---

### 📊 Pipeline 阶段分析（以 CUDA 为例）

| 阶段 | 占比（K=27~29） | 占比（K=31） | 趋势说明 |
|------|------------------|-------------|--------|
| **Disk Write** | ~80–82% | 59% | 始终主导，但相对占比下降 |
| **Sort / Table-2** | 9–11% | **34%** | 插入排序成为显著瓶颈（O(n²)） |
| **Table-1 Generation** | 7–9% | 7% | Blake3 GPU kernel 扩展良好 |

> 💡 表明：虽然 GPU 加速了计算部分，但 **I/O 和排序仍是制约因素**

---

### 🔁 Scaling Behavior（问题规模翻倍时的表现）

| 方法 | $T(k+1)/T(k)$ 平均值 | 是否接近理想线性（2.0×）？ |
|------|-----------------------|----------------------------|
| **SYCL GPU** | 1.95–2.03× | ✅ 几乎完美线性扩展 |
| **CUDA GPU** | 1.72→2.40×（波动大） | ❌ 在 K=30→31 达 **2.40×**（超线性退化） |

> 表明：**SYCL 实现在大规模下更具稳定性**；CUDA 因 kernel 管理开销增大而效率下降。

---

### ⚖️ 消融分析（隐含在设计讨论中）

- **Sort & Match 融合 kernel** 是最有效的优化：
  - 消除 global memory round-trip，显著减少延迟。
- **Insertion Sort 单线程 per bucket** 成为瓶颈：
  - 随 K 增加，bucket size 增长 → 排序时间从 11%（K=27）升至 34%（K=31）
- **Disk I/O 是最大瓶颈**：
  - 占据 **59–82%** 总时间，且完全由 CPU 串行执行，限制整体吞吐。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **GPU 加速是 PoSp Plotting 的正确方向**
   - SYCL GPU 实现达到 **59.2× speedup**，远超最优 CPU 配置（51.5×），且硬件成本更低。
   - GPU 在 Blake3 哈希密集型任务中具有压倒性优势。

2. **跨平台编程模型（SYCL）可行且高效**
   - 一套代码可在 AMD 和 Intel GPU 上高效运行，具备良好的生态前景。

3. **算法融合显著提升性能**
   - 将 Sort 与 Match 融合进单一 kernel 并驻留 shared memory，极大减少了内存带宽压力。

4. **CPU 并行效率低下**
   - 即使使用 384 线程，parallel efficiency 仅 **14%**，远低于理想值，表明单纯增加 CPU 核心无法有效扩展。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **Insertion Sort 瓶颈** | 当前每桶单线程排序，复杂度 O(n²)，随 K 增大影响加剧 |
| **Disk Write 完全串行** | 写盘阶段未并行化或异步化，成为最大性能瓶颈（占 >50% 时间） |
| **VRAM 容量限制** | K=31 需 24.7 GB VRAM，逼近高端卡极限，难以支持更大 K |
| **CUDA 管理开销大** | 在高负载下 kernel 调度效率下降，出现超线性缩放行为 |

---

### 🔮 未来工作方向

1. **替换 Insertion Sort**
   - 引入 **radix sort** 或 **merge sort** 等 GPU 并行排序算法，消除 O(n²) 瓶颈。

2. **优化 Disk Write Pipeline**
   - 使用 **asynchronous CUDA streams** 重叠 PCIe 传输与计算。
   - 探索 **direct NVMe write path**，绕过主机 CPU 缓冲区，降低 I/O 延迟。

3. **支持更大 K-values**
   - 采用 **chunked VRAM tiling** 技术，分块处理超出显存容量的数据，突破单卡限制。

4. **进一步统一编程模型**
   - 推动 SYCL 成为跨厂商 GPU 加速 PoSp 的标准开发框架。

---

> **总结一句话**：  
> **VaultxGPU 证明了 GPU 加速是突破 PoSp plot 生成瓶颈的关键路径，其 SYCL 实现在性能、可移植性和扩展性方面均展现出巨大潜力。**

</details>

---

### 11. [CacheRL:Multi-Turn Tool-Calling Agents via Cached Rollouts and Hybrid Reward](https://arxiv.org/abs/2606.14179)

**Authors**: Md Amirul Islam, Sumiran Thakur, Huancheng Chen, Su Min Park, Jiayun Wang, Gyuhak Kim  
**Category**: cs.CL  
**Published**: 2026-06-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.14179v1  

#### Abstract
We present CacheRL, a system for training small agent foundation models that achieves 92 percent process accuracy on multi-step tool-calling tasks, approaching GPT-5's 94 percent while requiring 100 times less compute. Our approach addresses three challenges in practical agent training: transferring...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《CacheRL: Multi-Turn Tool-Calling Agents via Cached Rollouts and Hybrid Reward》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文旨在解决**小规模 Agent 基础模型（small agent foundation models）在多轮工具调用（multi-turn tool-calling）任务中训练困难**的三大挑战：
1. **知识迁移难**：如何从大模型（如 GPT-5）有效迁移工具调用的推理能力到小模型。
2. **强化学习（RL）成本高**：传统 RL 需要实时执行工具调用，带来高昂的延迟、费用和风险。
3. **缓存环境噪声干扰**：使用缓存替代真实执行时，缓存质量不一（exact/fuzzy/best-effort），导致奖励信号失真。

---

### **提出的新方法与创新点**
CacheRL 提出了一套完整的系统级解决方案，包含以下三项核心技术：

#### **(1) Hybrid Thinking Trajectory Pipeline（混合思维轨迹流水线）**
- **方法**：利用 GPT-5 对已有工具调用轨迹进行增强，自动生成结构化 `<think>` 推理链，解释“为何”选择某个工具。
- **优势**：
  - 教会小模型“因果理解”，而不仅是模仿“调用动作”。
  - 保留原始工具调用和返回结果的真实性，避免合成数据失真。
  - 分类已有分析内容并复用，节省 15–20% API 成本。

#### **(2) CacheAgentLoop（缓存代理循环）**
- **方法**：引入一个**三级模糊缓存系统**（three-tier fuzzy cache）替代实时工具执行：
  - **Tier 1 – Exact**：精确匹配（SHA256 哈希）
  - **Tier 2 – Fuzzy**：基于 Jaccard 相似度的近似匹配
  - **Tier 3 – Best-effort**：工具级回退响应
- **关键技术**：
  - **Token-level masking**：仅对模型生成部分计算梯度，缓存注入结果设为 `mask=0`。
  - **Pre-close thinking blocks**：注入 `</think>` 引导模型进入回答阶段，防止无限推理循环。
- **优势**：
  - 将每次 rollout 成本从美元级降至**几分之一美分**，降低 100× 训练成本。
  - 支持长上下文（最长 132K tokens）、多步交互（平均 4.5 次工具调用）。

#### **(3) Cache-Tier-Aware Hybrid Reward（缓存层级感知混合奖励）**
- **方法**：动态调整奖励权重：
  $$
  R = \alpha(T) \cdot R_{\text{answer}} + (1 - \alpha(T)) \cdot R_{\text{process}}
  $$
  其中 $\alpha$ 根据缓存质量动态设定：
  | 缓存层级 | $\alpha$（答案评分权重） | 侧重 |
  |--------|--------------------------|------|
  | Exact  | 0.60                     | 答案质量 |
  | Fuzzy  | 0.30                     | 过程+答案 |
  | Best-effort | 0.10              | 工具调用过程 |

- **优势**：
  - 避免因缓存错误惩罚模型（如模糊匹配返回旧金山天气实为圣何塞）。
  - 在低质量缓存下仍能提供稳定的学习信号。

---

### **相比现有方法的优势**
| 维度 | CacheRL | 现有方法（如 ToolRL、Search-R1） |
|------|--------|-------------------------------|
| 模型大小 | 4B 参数 | 多为 70B+ 大模型 |
| 执行方式 | 完全缓存，无实时调用 | 依赖实时 API 调用，成本高 |
| 奖励机制 | 动态加权，抗噪声 | 固定奖励，易受缓存误差影响 |
| 知识迁移 | 显式推理链注入 | 仅复制工具调用序列 |
| 可扩展性 | 支持 44K 多轮轨迹训练 | 多限于单轮或小规模 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：44,449 条多轮工具调用对话，来源包括：
  - Toucan（84%）
  - Agent data（9%）
  - Toolathlon（3%）
  - 其他专用集合（4%）
- **覆盖领域**：天气查询、代码执行、文档管理、Web 搜索、金融 API、化学数据库等。
- **工具多样性**：共涉及 **1,185 种不同工具**，体现强泛化需求。
- **验证集**：120 条独立样本，用于每 5 步评估一次。

### **实验设置**
- **基础模型**：`Qwen3-4B-Thinking`
- **训练流程**：
  1. **Stage 1**：SFT（Supervised Fine-Tuning）在 44K 增强轨迹上。
  2. **Stage 2**：GRPO（Group Relative Policy Optimization）结合 CacheAgentLoop 进行 RL 微调。
- **关键技术参数**：
  - LoRA rank: 128
  - KL 正则系数 $\beta = 0.15$
  - Batch size: 16 prompts × 16 rollouts
  - 使用 8×H100 GPU，每步耗时约 15–20 分钟。

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Process Accuracy** | 工具调用是否正确（名称、参数、顺序） |
| **End-to-End Accuracy** | 最终答案是否满足用户需求 |
| **Validation Reward** | 混合奖励得分（GPT-5 判分 + 确定性规则） |
| **Tool-call Rate** | 成功发起至少一次工具调用的比例 |
| **Memory Recall** | 对历史工具结果的记忆与引用能力 |

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| GPT-5 | 前沿闭源模型，作为性能上限参考 |
| Qwen3-4B-Instruct | 同规模通用指令模型 |
| Qwen3-4B-Thinking | 加入思考模式但未微调的版本 |
| SFT-only | 仅监督微调，无 RL 优化 |
| v1–v5 版本 | 不同奖励设计与数据配置的消融变体 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | Process Accuracy | End-to-End Accuracy | 模型大小 | 训练成本 |
|------|------------------|---------------------|----------|-----------|
| GPT-5 | 94% | 77.8% | ~175B? | 极高 |
| **CacheRL (Ours)** | **92%** | **70.3%** | **4B** | **0.01× GPT-5** |
| Qwen3-4B-Thinking (SFT only) | 88.8% | — | 4B | — |
| Qwen3-4B-Instruct | 25.1% | — | 4B | — |

> ✅ **结论**：4B 模型达到 GPT-5 92% 的过程准确率，仅需其 **2.4% 的参数量** 和 **1% 的训练成本**。

---

### **与基线方法的对比结果**
- CacheRL 在 **Process Accuracy 上超越所有 4B 基线 60–70 个百分点**。
- 在 Memory Recall 任务中，CacheRL 达到 **34/37 正确率**，接近 GPT-5 的 37/37。
- 推理效率方面，CacheRL 比 GPT-5 快数倍，适合部署。

---

### **消融实验结果（Ablation Study）**
| 配置 | 验证奖励 | 相对下降 |
|------|---------|----------|
| Full CacheRL (v6 + 44K) | 0.779 | — |
| w/o GPT-5 reasoning | 0.458 | ↓41.2% |
| w/o cache-tier-aware reward | 0.645 | ↓17.2% |
| w/o rejection sampling | 0.687 | ↓11.8% |
| w/o thinking mode | 0.412 | ↓47.1% |
| w/o GRPO (SFT only) | 0.756 | ↓3.0% |

> 🔍 **关键发现**：
> - **高质量数据和推理链注入是决定性因素**（损失 41.2%）。
> - **缓存感知奖励至关重要**（损失 17.2%），否则模型被错误惩罚。
> - **RL 提升有限**：GRPO 仅带来 3% 提升，说明**数据质量 > RL 优化**。

---

## **4. 关键结论和发现**

### **主要结论**
1. ✅ **小模型也能成为强大 Agent**：通过系统性知识迁移，4B 模型可逼近 GPT-5 的工具调用能力。
2. ✅ **数据质量远胜算法复杂度**：高质量推理轨迹 + 清洗后的 GT 是性能基石。
3. ✅ **缓存可替代实时执行**：三级缓存 + token masking + 动态奖励使 RL 训练变得可行且高效。
4. ✅ **RL 主要提升稳定性而非性能上限**：在高质量 SFT 基础上，RL 能稳定训练但难以突破瓶颈。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **缓存陈旧性（Staleness）** | API 返回格式可能变化，缓存无法反映最新行为 |
| **依赖教师模型质量** | GPT-5 自身错误（如 meta-reasoning）会被学生继承 |
| **长尾工具泛化差** | 对训练中未见的新工具（尤其是私有 API）表现下降 15–20% |
| **推理延迟增加** | `<think>` 结构使推理变慢 2–3×，不适合低延迟场景 |
| **评估偏理想化** | 当前 benchmark 多为确定性任务，缺乏开放意图处理能力测试 |

---

### **未来工作方向**
1. **Online Learning**：部署中引入真实反馈，实现缓存动态更新。
2. **Multi-Teacher Distillation**：融合多个大模型（如 GPT-5 + Claude + Gemini）减少单一偏见。
3. **Tool Embedding Learning**：基于工具 schema 自动生成嵌入，提升对新工具的零样本适应能力。
4. **Adaptive Thinking**：仅在复杂任务中启用 `<think>` 模式，平衡速度与精度。
5. **Hierarchical Caching**：构建工具类别级缓存策略，提高覆盖率与灵活性。
6. **Replace LLM-as-Judge**：训练专用 Reward Model 替代 GPT-5 判分，降低成本与波动。

---

> 🏁 **最终启示**：  
> “**Building strong small agent models is less about fancy RL algorithms and more about systematic knowledge transfer, smart data engineering, and robust reward design.**”  
> —— CacheRL 的成功表明，通往实用 Agent 的路径在于**工程系统的完整性**，而非单纯追求模型规模或 RL 技巧。

</details>

---

### 12. [AdaSR: Adaptive Streaming Reasoning with Hierarchical Relative Policy Optimization](https://arxiv.org/abs/2606.14694)

**Authors**: Junlong Tong, Wenqi Xu, Yingqi Fan, Anhao Zhao, Xuan Lu, Yang Tan, Xiaoyu Shen  
**Category**: cs.CL  
**Published**: 2026-06-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.14694v1  

#### Abstract
Large reasoning models typically follow a read-then-think paradigm: they observe the complete input, reason over a static context, and then produce the answer. Yet many real-world scenarios are inherently dynamic, such as audio and video stream, where information arrives as a continuous stream and m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AdaSR: Adaptive Streaming Reasoning with Hierarchical Relative Policy Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的大模型推理范式遵循“先读完再思考”（read-then-think）模式，即在完整输入上下文可见后才开始生成推理链（Chain-of-Thought, CoT）。然而，在许多现实场景中（如语音、视频流、实时交互系统），输入是**动态连续到达的流式数据**，等待全部输入完成会引入不必要的延迟。

现有**streaming CoT**方法大多依赖于对预构建的流式推理轨迹进行监督微调（SFT），这存在两个关键问题：
- **标注成本高**：每个部分输入都需要对应局部推理标注；
- **缺乏适应性**：模型只是模仿表面形式，无法自主决定何时需要浅层理解、深层推理或跳过不相关信息。

### 提出的新方法与新思路
本文提出 **AdaSR**（Adaptive Streaming Reasoning），一个基于强化学习（RL）的自适应流式推理框架，其核心创新包括：

#### （1）自适应计算策略学习
AdaSR 允许模型在输入流到来时**边读边想**（think while reading），并学会：
- **When to think**: 是否对当前输入段进行推理；
- **When to skip**: 是否跳过无关内容；
- **How much to compute**: 在流式推理阶段和最终深度推理阶段之间**动态分配计算资源**。

#### （2）分层相对策略优化（HRPO）
为解决标准 GRPO 在流式推理中的**粗粒度信用分配问题**，作者提出 **Hierarchical Relative Policy Optimization (HRPO)**，将策略优化分解为三个层次：
- **Streaming-local advantage**：奖励流式推理阶段的局部决策；
- **Deep-local advantage**：奖励最终整合阶段的全局推理质量；
- **Global-level advantage**：保证整体答案正确性。

通过这种细粒度的优势分配机制，HRPO 能更准确地归因不同阶段的贡献，避免“成功掩盖冗余思考”或“失败惩罚有用早期推理”的信用悖论。

#### （3）多目标自适应奖励设计
引入三项奖励函数联合优化：
- **Format Reward**：确保输出符合 `<EOT>`、`<skip>` 等协议格式；
- **Accuracy Reward**：任务最终准确性；
- **Adaptive Thinking Reward**：以 token 长度为代理，鼓励高效计算，尤其考虑**流式推理可与输入重叠而深度推理不可**的时间不对称性。

---

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **灵活性** | 不依赖人工标注的流式轨迹，通过 RL 自主学习何时思考、何时跳过； |
| **效率更高** | 显著减少总生成长度和响应延迟，尤其降低关键路径上的深度推理负担； |
| **泛化更强** | 在 out-of-domain 和 out-of-task 数据上表现更好，表明具备更强的迁移能力； |
| **结构匹配** | HRPO 的分层优势设计与流式推理的自然阶段结构高度契合，优于扁平化优化。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **In-domain**：
  - `GSM-Symbolic`（P1/P2）：数学推理题，长文本描述可自然切分为句子级流；
  - `MetaMathQA`：数学问答数据集；
  - `PubMedQA`：生物医学领域的上下文问答。
- **Out-of-domain / Out-of-task**：
  - `GSM-Infinite`：测试极长上下文下的推理能力；
  - `LogicNLI`：逻辑推理任务，用于评估跨任务泛化。

所有问题陈述足够长且可按句分割，适合模拟流式输入。

### 实验设置
- **模型基础**：基于 Qwen3 系列模型（1.7B 和 4B 参数规模）；
- **训练流程**：
  - 初始策略来自 StreamingThinker 的 SFT 模型；
  - 使用 vLLM 实现高效的流式 rollout 推理；
  - 采用 group sampling（每组 G=12 条 rollout）进行 GRPO/HRPO 更新；
  - 使用 verl 框架适配 streaming rollout 与 log-prob 计算。
- **推理方式**：
  - 输入按句子逐步喂入；
  - 每句后允许生成一段流式推理（以 `<EOT>` 结束）或 `<skip>`；
  - 最终拼接所有流式推理后执行一次 deep reasoning 得到答案。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (Acc)** | 主要任务性能指标，衡量最终答案正确率； |
| **Streaming/Deep/Total Length (Len.)** | 分别统计流式推理、深度推理及总生成 token 数，反映计算开销； |
| **Latency (s)** | 用户感知延迟，从最后一个输入 token 到第一个输出 token 的时间； |
| **Throughput (token/s)** | 吞吐量，衡量系统处理效率； |
| **Speedup** | 相对于 read-then-think 的延迟加速比。 |

### 基线方法对比
| 方法 | 类型 | 说明 |
|------|------|------|
| **Read-then-think** | 基准 | 完整输入后再推理，无流式处理； |
| **StreamingThinker (SFT)** | 监督基线 | 使用 SFT 训练的流式 CoT 模型； |
| **AdaSR-GRPO** | 强化学习基线 | 使用标准 GRPO 进行 RL 微调； |
| **AdaSR-HRPO (Ours)** | 本文方法 | 提出的分层优势优化框架。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen3-1.7B / 4B）

#### 表格摘要：AdaSR-HRPO vs. 基线（部分代表性结果）

| Model | Benchmark | Acc ↑ | Total Len ↓ | Deep Len ↓ | Latency ↓ |
|-------|-----------|--------|-------------|------------|-----------|
| Read-then-think | GSM-symbolic P2 | 0.424 | 1866.474 | — | 56.250s |
| SFT (StreamingThinker) | GSM-symbolic P2 | 0.642 | 361.906 | 132.616 | 6.080s |
| AdaSR-GRPO | GSM-symbolic P2 | 0.758 (+18.1%) | 384.488 (+6.2%) | 148.704 | 6.710s |
| **AdaSR-HRPO (Ours)** | GSM-symbolic P2 | **0.788 (+22.7%)** | **370.256 (+2.3%)** | **160.210** | **6.520s** |
| **AdaSR-HRPO (Qwen3-4B)** | GSM-symbolic P2 | **0.884 (+8.3%)** | **327.552 (-8.4%)** | 131.162 | — |

> 注：百分比均为相对于 SFT 基线的变化。

### 与基线方法的对比结果
- **相比 SFT 基线**：
  - 在 Qwen3-1.7B 上，HRPO 在 GSM-symbolic P2 上提升 **+22.7% 准确率**，同时仅轻微增加总长度（+2.3%）；
  - 在 Qwen3-4B 上实现**帕累托改进**：准确率提升的同时，总长度下降 **6.9%-13.4%**。
- **相比 GRPO**：
  - HRPO 在几乎所有任务上都实现了更高的准确率和更低的生成长度；
  - 例如在 MetaMathQA 上，HRPO 比 GRPO 多提升 1.7%，且总长度减少 4.8%；
  - 证明**分层优势分配能有效抑制冗余推理**，引导更高效的计算路径。

### 消融实验结果

#### （1）分层优势分配粒度比较（Table 2）
| 方法 | Acc (P2) | Total Len | 说明 |
|------|---------|----------|------|
| GRPO | 0.758 | 384.488 | 扁平化优势分配 |
| HRPO (stage-level) | **0.788** | **370.256** | ✅ 最佳平衡 |
| HRPO-sentence | 0.754 | 381.970 | 句级分配 → 准确率下降 |
| HRPO-token | 0.770 | 358.960 | 仅边界 token 控制 → 效果次优 |

> 发现：**最细粒度 ≠ 最好效果**。阶段级（stage-level）划分最符合流式推理的语义结构。

#### （2）奖励组件消融（Table 3）
| Reward Setting | Acc (P2) | Total Len |
|----------------|----------|-----------|
| Acc only | 0.726 | 397.908 |
| Acc + Format | 0.762 | 479.282 |
| Acc + Format + Length (**HRPO**) | **0.788** | **370.256** |

> 发现：
> - 加入 format reward 提升稳定性但导致长度膨胀；
> - 加入 adaptive thinking reward 后显著压缩长度并进一步提升准确率；
> - 三者协同作用明显，缺一不可。

#### （3）跨域泛化能力（Table 4）
| Method | GSM-Infinite Acc | LogicNLI Acc |
|--------|------------------|---------------|
| SFT-based | 0.479 | 0.474 |
| GRPO | 0.509 | 0.445 |
| **HRPO** | **0.546** | **0.489** |

> HRPO 在未见过的任务和更复杂输入下仍保持领先，显示良好泛化性。

---

## 4. 关键结论和发现

### 主要发现
1. **流式推理不是简单的“提前输出”**，而是一个需要**精细信用分配和计算调度的新型优化问题**；
2. **HRPO 的分层优势机制显著优于传统 GRPO**，解决了流式推理中“局部行为难以被正确归因”的核心挑战；
3. **AdaSR 学会了真正的自适应推理**：在简单段落跳过推理，在关键信息处主动深入分析，最终大幅缩短深度推理负担；
4. **延迟显著降低**：相比 read-then-think，所有流式方法实现 **8–9倍的延迟压缩**，AdaSR-HRPO 在 vLLM 下达到 **8.82× 加速**（Table 5）；
5. **小模型受益更大**：Qwen3-1.7B 上 AdaSR 通过增加少量流式计算换取巨大准确率提升，体现自适应计算的价值。

### 方法的局限性
- 当前工作聚焦于**有明确答案的文本流推理任务**，尚未扩展至音频、视频等多模态流；
- 奖励信号仍依赖自动可验证的答案（verifiable rewards），难以应用于开放对话等主观任务；
- 流式切分目前基于句子，未探索更细粒度（如词、短语）或动态分块策略；
- HRPO 的超参数（如 λ, β, α）需仔细调节，对初学者有一定门槛。

### 未来工作方向
- 扩展至 **audio/video streaming reasoning**，结合感知信号设计更丰富的 reward shaping；
- 构建**多模态 rollout engine**，支持语音输入同步推理与应答；
- 探索**在线调度机制**，动态调整 streaming/deep 阶段的计算预算；
- 将 AdaSR 应用于 **agentic systems**，实现在工具调用、检索、行动中持续推理；
- 研究如何让模型**自我判断是否需要进入 deep thinking**，而非固定阶段划分。

---

> 🔗 **代码已开源**：[EIT-NLP/AdaSR](https://github.com/EIT-NLP/AdaSR)

</details>

---

### 13. [Zeta: Dual Whitening for Matrix Optimization via Coordinate-Adaptive Preconditioning](https://arxiv.org/abs/2606.14187)

**Authors**: Kaiwen Chen, Shuhai Zhang, Qiuwu Chen, Zimo Liu, Linxiao Li, Ying Sun, Yuchen Li, Yifan Zhang, Bo Han, Mingkui Tan  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.14187v1  

#### Abstract
Large-scale neural network training increasingly relies on matrix-aware optimizers that exploit the structure of weight parameters beyond element-wise adaptation. However, existing matrix-aware methods such as Muon have an underappreciated vulnerability: their core operation, Newton-Schulz iteration...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Zeta: Dual Whitening for Matrix Optimization via Coordinate-Adaptive Preconditioning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现有的矩阵感知优化器（如 **Muon**）虽然通过 **spectral whitening**（谱白化）在大规模训练中表现出色，但其核心操作 **Newton-Schulz 迭代** 对输入的条件数（conditioning）高度敏感。然而，在实际训练中，动量矩阵（momentum matrix）存在严重的 **coordinate-wise scale heterogeneity**（坐标尺度异质性），即不同参数坐标的梯度幅值差异巨大。

这种尺度不平衡导致：

- 谱白化的输入条件差；
- Newton-Schulz 迭代收敛不稳定；
- 正交化误差增大，更新方向失真。

因此，尽管 Muon 在几何上设计精巧，但其性能受限于未处理的原始动量矩阵的尺度失衡问题。

---

### 提出了什么新方法或新思路

作者提出 **Zeta**，一种 **dual whitening（双重白化）** 优化器，其核心思想是：**spectral whitening 必须建立在统计稳定性之上**。

Zeta 将优化过程组织为一个 **严格有序的双阶段流水线**：

1. **Coordinate Whitening（坐标白化）**  
   使用类似 Adam 的方式，对每个参数进行基于运行时二阶矩（running second moment）的归一化，消除坐标间的尺度差异。

2. **Spectral Whitening（谱白化）**  
   在坐标白化后的“良好条件”矩阵上执行 Newton-Schulz 迭代，实现稳定的正交化。

> ✅ **关键洞见**：这两个步骤不是可选组合，而是数学依赖关系——坐标白化为谱白化提供了必要的统计各向同性（statistical isotropy）前提。

---

### 相比现有方法的优势

| 方面 | 优势说明 |
|------|----------|
| **理论保障** | 证明了坐标白化等价于对角预条件（diagonal preconditioning），并能严格降低输入矩阵的条件数，从而减少正交化误差（Theorem 1 & 2）。 |
| **结构统一** | 统一了 coordinate-adaptive 和 matrix-structured 两类优化范式，前者为后者服务，形成互补而非竞争。 |
| **效率可控** | 与 AdaMuon 相比，Zeta 不增加渐近时间复杂度（仍为 $O(Kmnr)$），仅增加低阶元素级操作，计算开销小。 |
| **泛化性强** | 在语言模型（dense & MoE）、视觉任务（ViT）上均表现优越，验证了方法的通用性。 |

---

## 2. 核心实验方法和设置

### 使用的数据集

| 模型类型 | 数据集 |
|---------|--------|
| **语言模型（Dense）** | OpenWebText3（用于 GPT2-Large 和 Qwen3-0.6B）<br>DCLM-baseline（用于 Qwen3-1.7B, 8B） |
| **混合专家模型（MoE）** | DCLM-baseline（Qwen3-1.3B-A0.6B） |
| **视觉模型（ViT）** | CIFAR-100 |

---

### 实验设置和评估指标

#### 模型规模
- **语言模型**：0.6B, 1.7B, 8B 参数的 Qwen3 系列；GPT2-Large
- **MoE 架构**：总参数 1.3B，激活参数 0.6B
- **视觉模型**：ViT-Tiny 和 ViT-Base

#### 优化配置
- 学习率调度：cosine decay，warmup 比例 1%
- 批大小：global batch size 256（语言模型）
- 序列长度：4096（除 GPT2-Large 为 1024）
- Newton-Schulz 步数：$K=5$
- Zeta 超参：$\beta_1=0.95$, $\beta_2=0.99$

#### 评估指标
- **训练阶段**：训练损失下降速度（loss vs. iteration/time）
- **下游任务**：12 项 NLP 任务的平均准确率（如 MMLU, GSM8k, C-Eval 等）
- **视觉任务**：CIFAR-100 上的 Top-1 验证准确率

---

### 基线方法对比

| 优化器 | 特点 |
|-------|------|
| **AdamW** | 标准自适应优化器，逐元素缩放 |
| **Muon** | 纯谱白化，无坐标归一化 |
| **AdaMuon** | 结合 sign-based 自适应与谱白化 |
| **Zeta（本文）** | 坐标白化 + 谱白化，顺序不可逆 |

所有实验保持相同模型架构、数据、硬件环境，仅更换优化器以公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📈 语言模型训练效率（迭代速度）

| 模型 | Zeta 相对 AdamW 的加速比（基于迭代） |
|------|-------------------------------|
| Qwen3-1.7B | **1.67×** |
| Qwen3-8B | **1.25×** |
| Qwen3-MoE | **1.47×** |

> 🔹 Muon 加速比分别为 1.52×、1.19×、1.18×，Zeta 明显更优。

#### ⏱️ 单步耗时（接近 AdamW）

| 模型（8B） | 单步 wall-clock 时间 |
|-----------|------------------|
| AdamW | 24.46s |
| Muon | 26.19s |
| **Zeta** | **25.47s** |

> ✅ Zeta 在显著提升收敛速度的同时，几乎不增加每步开销。

---

### 与基线方法的对比结果

#### 下游任务性能（Qwen3-8B）

| 优化器 | 平均准确率（AVG） |
|--------|------------------|
| AdamW | 23.42% |
| Muon | 26.03% |
| AdaMuon | 25.02% |
| **Zeta** | **26.29%** |

> 🔺 超越最强基线 Muon 达 **+0.26%**，且全面优于其他方法。

#### 视觉任务性能（CIFAR-100）

| 模型 | AdamW | Muon | AdaMuon | **Zeta** |
|------|-------|------|---------|--------|
| ViT-Tiny | 57.02% | 64.68% | 63.96% | **64.98%** |
| ViT-Base | 57.98% | 64.53% | 62.03% | **65.34%** |

> ✅ Zeta 在两种规模上均达到最高精度，尤其在 ViT-Base 上领先 Muon **+0.81%**。

---

### 消融实验结果

#### 动量系数鲁棒性（Qwen3-0.6B）

| $(\beta_1, \beta_2)$ | 最终训练损失 |
|---------------------|-------------|
| (0.9, 0.99) | **2.564** |
| (0.95, 0.99) | **2.564** |
| (0.95, 0.95) | 2.565 |
| (0.9, 0.9) | 2.571 |

> 🔍 Zeta 对超参数不敏感，在多个设置下性能稳定，无需精细调参。

#### 学习率敏感性分析（见附录）

- Zeta 在多个学习率下均优于基线；
- 性能峰值明显，易于调优。

---

## 4. 关键结论和发现

### 主要发现

1. **坐标尺度异质性是谱白化失效的根本原因**  
   通过 chi-square uniformity test 验证，Transformer 各层动量矩阵普遍存在严重尺度不平衡，而坐标白化可有效纠正。

2. **双重白化具有数学必要性而非启发式组合**  
   坐标白化改善输入条件数，使 Newton-Schulz 迭代更稳定，该顺序不可颠倒。

3. **Zeta 实现更快收敛与更好泛化**  
   在多种架构（dense、MoE、ViT）和规模（0.6B–8B）上一致超越 AdamW、Muon、AdaMuon。

4. **理论与实践统一**  
   提出的 dual preconditioning 框架有严格理论支撑（Theorem 1–3），解释了为何 Zeta 更优。

---

### 方法的局限性

- **尚未在超大规模模型（>10B）上验证**：当前实验最大为 8B，更大模型的表现有待探索。
- **未考虑动态稀疏性或其他结构先验**：假设所有矩阵参数都适用 dual whitening，可能忽略特定模块的优化需求。
- **存储开销略高于 Muon**：需维护两个动量状态（$M_t, V_t$），存储复杂度从 $O(mn)$ 升至 $O(2mn)$。

---

### 未来工作方向

1. **扩展到更多架构**：如 diffusion models、graph neural networks 等非 Transformer 结构。
2. **结合 curvature-aware 方法**：将 Zeta 与更高阶几何优化（如 Shampoo、SOAP）融合。
3. **系统级优化**：研究 Zeta 在分布式训练中的通信效率与负载均衡。
4. **理论深化**：分析 dual whitening 在非凸优化中的全局收敛性质。

---

> 💡 **一句话总结**：Zeta 揭示了“先坐标归一，再谱正交”的数学必要性，通过 dual whitening 实现了更稳定、高效、通用的矩阵优化，为下一代优化器设计提供了新范式。

</details>

---

### 14. [Can Deep Neural Networks Improve Compression of Very Large Scientific Data?](https://arxiv.org/abs/2606.14353)

**Authors**: Muhannad Alhumaidi, Guozhong Li, Spiros Skiadopoulos, Panos Kalnis  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.14353v1  

#### Abstract
Error-bounded lossy compression is a fundamental technique for managing the rapidly growing volumes of scientific data produced by modern simulations and observational instruments. Most state-of-the-art-compressors follow a prediction-residual paradigm, where compression effectiveness depends on the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Can Deep Neural Networks Improve Compression of Very Large Scientific Data?

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文探讨了一个核心问题：**现代深度学习模型（尤其是预训练的天气预报 Foundation Models）能否作为更优的预测器，提升大规模科学数据的 error-bounded lossy compression（有误差界约束的有损压缩）性能？**

传统压缩器（如 SZ3.1）依赖简单的空间插值预测器，而近年来在气象领域出现的高精度 ML 预测模型（如 GraphCast、Aurora）能够捕捉复杂的物理先验和长程时空依赖关系。这引发思考：这些先进模型是否可以直接用于提升压缩效率？

然而，从头训练一个面向压缩任务的专用 ML 模型成本极高，且难以公平比较。因此，作者提出了一种“代理实验”（proxy experiment）策略。

---

### 提出了什么新方法或新思路

1. **构建了一个统一的 error-bounded 压缩框架**  
   将多种不同类型的 ML 模型（空间重建型、时间序列预测型）集成到标准的 prediction-residual 压缩流水线中，并共享相同的量化（quantization）和熵编码（entropy coding）后端，确保公平比较。

2. **设计了 auto-regressive 压缩框架以支持 temporal forecasters**  
   针对 GraphCast 和 Aurora 这类需要历史状态进行预测的时间模型，提出了一个闭环的 auto-regressive 压缩流程：
   - 使用两个无损存储的初始状态“种子”启动预测；
   - 每一步预测后计算残差并量化；
   - 将“修正后的状态”反馈给下一时刻作为输入；
   - 保证每一步都满足误差边界，避免误差累积。

   > ✅ 创新点：该设计使得原本用于自由推演（free-running）的天气模型可以稳定地用于长期压缩任务，且不违反误差约束。

3. **系统性评估三种代表性 ML 预测器 vs. 经典压缩器 SZ3.1**
   - **CRA5**：基于 VAEformer 的 learned spatial codec，专为 ERA5 设计；
   - **GraphCast**：基于图神经网络的时间预测器；
   - **Aurora**：基于 Vision Transformer 的时间预测器；
   - **SZ3.1**：经典多项式空间预测器，作为 baseline。

---

### 相比现有方法的优势

- **方法论优势**：首次将最先进的天气预报 foundation models 系统性地用作压缩中的预测模块，在相同误差约束下进行横向对比。
- **工程实现优势**：解决了 temporal model 在压缩场景下的反馈稳定性问题，实现了长达近 2000 步的稳定 auto-regressive 压缩。
- **分析深度优势**：不仅报告压缩率，还深入分析了残差结构、误差分布、空间模式等，揭示了“为何更好的预测不一定带来更高压缩比”的根本原因。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

- **ERA5 全球再分析数据集**（European Centre for Medium-Range Weather Forecasts）
  - 时间跨度：提取连续 1,997 个时间步（约 500 天），采样间隔为 6 小时；
  - 空间分辨率：0.25° × 0.25°，即 $721 \times 1440$ 网格；
  - 数据总量：约 **1.7 TB** 的单精度浮点数；
  - 包含变量：共 9 个常见大气变量，分为两类：
    - 单层变量（single-level）：T2m（2米温度）、U10/V10（10米风速分量）、MSL（海平面气压）；
    - 多层变量（multi-level）：T（温度）、Z（位势高度）、U/V（风速分量）、Q（比湿），分别在多个 pressure levels 上。

---

### 实验设置和评估指标

#### 主要评估指标

| 指标 | 描述 |
|------|------|
| **Compression Ratio (CR)** | 原始大小 / 压缩后大小（含所有开销）；也报告排除模型权重后的 $p'$ |
| **Mean Absolute Error (MAE)** | 平均绝对误差，衡量重构保真度 |
| **Root Mean Square Error (RMSE)** | 均方根误差 |
| **CDF of Pointwise Absolute Error** | 逐点误差分布，反映误差集中程度 |
| **Per-pixel MAE Map** | 时空平均误差的空间分布图 |
| **Throughput (MB/s)** | 压缩/解压吞吐量 |

#### 误差约束设置
- 相对误差界 $\epsilon \in \{10^{-2}, 10^{-3}, 10^{-4}\}$，适用于所有方法；
- 所有方法使用相同的 quantization bin 宽度 $\delta_x = \epsilon \cdot r_x$ 和 Huffman + Zstd 编码后端。

#### 硬件配置
- CRA5 & Aurora：NVIDIA A100-SXM4-80GB GPU；
- SZ3.1 & GraphCast：Intel Xeon Gold CPU（因 JAX 实现非确定性，必须运行于 CPU 以保证 bit-reproducibility）。

---

### 基线方法对比

| 方法 | 类型 | 是否时间感知 | 特点 |
|------|------|----------------|--------|
| **SZ3.1** | Classical spatial predictor | ❌ 否 | 使用动态多维样条插值，仅利用空间邻域信息 |
| **CRA5** | Learned spatial codec (VAEformer) | ❌ 否 | 专为 ERA5 设计，独立重建每一帧 |
| **GraphCast** | Temporal GNN forecaster | ✅ 是 | 图神经网络，需前两步状态预测下一步 |
| **Aurora** | Temporal ViT forecaster | ✅ 是 | 基于 Patch 的 3D Swin Transformer，更强表达能力 |

> ⚠️ 所有方法共享同一 quantization 和 entropy coding 后端，仅“预测器”部分不同 → 实现公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）

| Method | $\epsilon=10^{-2}$ ($p$) | $\epsilon=10^{-2}$ ($p'$) | $\epsilon=10^{-3}$ ($p$) | $\epsilon=10^{-4}$ ($p$) |
|--------|----------------------------|------------------------------|----------------------------|----------------------------|
| **SZ3.1** | **198.80** | — | **36.07** | **11.72** |
| CRA5 | 130.54 | 136.68 | 19.83 | 7.05 |
| GraphCast | 59.87 | 60.15 | 13.77 | 5.35 |
| Aurora | 55.87 | **109.72** | 14.35 | 5.20 |

> 注：$p$: 包含模型权重开销；$p'$: 排除模型权重（更合理用于长期复用场景）

---

### 与基线方法的对比结果

#### 📉 整体压缩率：**SZ3.1 显著优于所有 ML 方法**
- 在所有误差界下，**SZ3.1 的 dataset-level compression ratio 最高**；
- 即使排除模型权重（$p'$），Aurora 虽然大幅提升（从 56 → 110），但仍低于 SZ3.1；
- 原因：ML 模型虽然预测更准，但其残差缺乏空间连贯性，不利于熵编码。

#### ✅ 重构质量（Reconstruction Fidelity）：**ML 方法显著更优**
- **CRA5** 在多数变量上 MAE 最低，例如：
  - 对 **Z（位势高度）**，MAE 比 SZ3.1 降低 **91%**；
  - 对 **MSL（海平面气压）**，MAE 降低 **84%**；
- **Aurora** 在光滑场（如 MSL、Z）上表现优异，MAE 比 SZ3.1 低 27–34%；
- **误差分布更平滑**：SZ3.1 的误差呈孤立像素状“噪声”，而 ML 方法的误差集中在可预测性低的区域（如风暴带），更具物理意义。

#### 🔍 分变量压缩率（Per-variable CR）
- 在某些高度可预测的变量上，ML 方法反超 SZ3.1：
  - **Aurora 对 Z@top-of-atmosphere** 达到 **>45,000×** 压缩比（vs. SZ3.1 的 ~670×）；
  - **Aurora 对 MSL、T2m** 在 $\epsilon=10^{-2}$ 下可达 **8,500×~9,600×**，是 SZ3.1 的 **9.6 倍以上**；
- 但在湍流性强的变量（如 Q、U/V）上，SZ3.1 仍占优。

#### ⏱️ 压缩/解压速度（Throughput）
| Method | Compress (MB/s) | Decompress (MB/s) |
|--------|------------------|--------------------|
| **SZ3.1** | **348.9** | **1384.8** |
| CRA5 | 237.8 | 723.2 |
| GraphCast | 30.7 | 33.3 |
| Aurora | 36.1 | 46.0 |

> SZ3.1 快一个数量级以上，尤其 auto-regressive 方法受限于串行推理。

---

### 消融实验结果

#### （1）初始化开销影响（Initialization Overhead）
- GraphCast/Aurora 需存储两个 seed states（约 8.3 GB），在短序列中占比大；
- 当 $\epsilon=10^{-2}$ 时，即使经过 1997 步，seed 开销仍未被摊薄；
- 在长期归档场景中，此开销可忽略 → 支持 $p'$ 更具现实意义。

#### （2）Auto-regressive 稳定性验证
- 图 11 显示，在长达 500 天（~2000 步）的 rollout 中，Normalized MAE 保持平稳，无漂移趋势；
- 表明所提闭环反馈机制有效防止了误差积累。

#### （3）残差结构分析（Residual Structure）
- **SZ3.1** 残差：零星分散，集中在少数孤立点；
- **ML 方法** 残差：成片出现，沿海岸线、锋面等地形/气象特征组织；
- 但由于当前熵编码器为 **order-0 Huffman + 1D Zstd**，无法利用二维结构 → 导致压缩效率未提升。

> 💡 发现：**预测准确性 ≠ 更高压缩比**，关键在于残差的熵 $H(Q)$，而非 MAE。

---

## 4. 关键结论和发现

### 论文的主要发现

1. **核心悖论解释**：  
   > “尽管 ML 预测器能生成更准确的预测并显著提高重构质量（up to 91% MAE reduction），却未能提升整体 dataset-level compression ratio。”

   原因在于：**压缩比由残差的熵决定，而非预测精度本身**。ML 模型虽减小了残差幅值，但其残差具有更强的空间结构性，导致整数量化后的符号分布更均匀，熵更高，反而不利于传统 1D 熵编码。

2. **ML 预测器的价值是情境化的**：
   - 在特定变量（如 Z、MSL）上，ML 可实现高达 **9.6× 更高的压缩比**；
   - 在需要高保真重构的应用中（如气候模拟后处理），ML 方法提供更自然、物理一致的误差分布；
   - 但在整体压缩目标下，简单高效的 SZ3.1 仍是赢家。

3. **Auto-regressive 框架可行且稳定**：  
   所提出的 feedback loop 成功将 temporal forecasters 应用于长期 error-bounded 压缩，且无误差漂移。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **计算与存储开销大** | ML 模型参数庞大（Aurora: 1.3B），部署成本高；需额外存储 seed states |
| **顺序解码瓶颈** | auto-regressive 方法无法并行解码 timestep $k$，限制实时应用 |
| **压缩后端不匹配** | 当前熵编码器无法利用 ML 残差的二维结构，造成潜力浪费 |
| **硬件差异影响性能对比** | GraphCast 因非确定性被迫运行于 CPU，拖慢速度 |

---

### 未来工作方向

1. **Structure-aware Residual Coding**  
   设计能利用残差空间结构的 2D 或 context-adaptive 编码器（如 CNN-based entropy model），释放 ML 预测的优势。

2. **Compression-aware Predictor Training**  
   不再以 MSE 为目标，而是直接优化 **residual entropy** 或 **rate-distortion tradeoff**，使模型适配压缩需求。

3. **扩展至更多 forecasting models**  
   如 Pangu-Weather、FourCastNet、FengWu 等，进一步探索不同架构的影响。

4. **探索 hybrid 架构**  
   结合 spatial 和 temporal 预测优势，或融合物理模型与 ML 模型。

5. **端到端 trainable compression pipeline**  
   构建 joint optimization 框架，联合训练预测器与编码器。

---

> ✅ 总结一句话：  
> **“Foundation Models 能极大提升科学数据压缩的重构质量，但要转化为更高的压缩比，还需新的编码范式来解锁其残差中的结构信息。”**

</details>

---

### 15. [Formalizing Numerical Analysis: An Agent Pipeline and Quality Audit Beyond Kernel Acceptance](https://arxiv.org/abs/2606.14000)

**Authors**: Theodore Meek, Siyuan Ge, Di Qiu Xiang, Simon Chess, Vasily Ilin  
**Category**: cs.AI  
**Published**: 2026-06-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.14000v1  

#### Abstract
Recent work has demonstrated that coding agents can formalize entire advanced mathematics textbooks in Lean 4, yet existing efforts concentrate on branches of mathematics already well-represented in mathlib and measure success solely through kernel acceptance. We address both limitations by applying...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Formalizing Numerical Analysis: An Agent Pipeline and Quality Audit Beyond Kernel Acceptance

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前的 **autoformalization**（自动形式化）研究存在两大局限：

1. **领域覆盖偏差**：现有工作大多集中在 `mathlib` 中已有良好支持的数学分支（如代数、拓扑），而对缺乏现成库支持的领域（如数值分析）缺乏验证。
2. **评估方式单一**：主流评价标准是“代码是否通过编译”（kernel acceptance），即能否在 Lean 中成功构建（build）且无 `sorry`。然而，这只能保证语法正确性，无法确保**语义忠实性**（semantic correctness），即形式化是否真正忠于原始文本。

本文针对这两个问题，提出更严格的形式化质量评估体系，并在一个全新领域进行压力测试。

---

### 提出了什么新方法或新思路

#### （1）**新型 Agent Pipeline**
- 构建了一个基于多角色 LLM 的自动化形式化流水线（pipeline），包含四个角色：
  - **Planner**：规划下一步任务
  - **Worker**：执行编辑与证明
  - **Evaluator**：评估进展并打分
  - **Consultant**：在连续失败时介入（escalation）
- 流水线为**状态无关**（stateless），所有通信通过文件和 GitHub Actions 完成，提升可复现性。
- 支持动态调度，而非固定“先写 statement 再证”的两阶段模式。

#### （2）**三维质量审计框架（Quality Audit Framework）**
超越 kernel acceptance，从三个维度系统评估形式化质量：

| 维度 | 描述 |
|------|------|
| **Semantic Correctness** | 形式化陈述是否在逻辑上忠实于原文？使用 LLM-as-judge 进行双向判断（Lean → NL 和 NL → Lean）。 |
| **Mathlib Reuse** | 是否合理复用 `mathlib`？避免重复造轮子。通过 `exact?+all` 检查定理是否可由 `mathlib` 直接推出。 |
| **Cross-file Reuse** | 是否在项目内部复用已形式化的中间结果？通过提取教科书依赖图并与 Lean 代码引用对比来衡量。 |

该框架首次将**结构性复用行为**纳入评估，适用于大规模形式化项目。

---

### 相比现有方法的优势

| 方面 | 本文优势 |
|------|--------|
| **领域广度** | 首次在 `mathlib` 几乎空白的**数值分析**领域完成全书形式化，验证 agent 在“从零构建理论”场景下的能力。 |
| **评估深度** | 不再仅看“是否能编译”，而是揭示出大量 kernel acceptance 完全无法发现的错误模式。 |
| **可复现性** | 提供完整 pipeline 设计、prompt 模板、评估脚本，支持他人复现和扩展。 |
| **系统性** | 将形式化质量拆解为多个可观测、可量化的维度，推动建立标准化评估基准。 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **主数据集**：Butcher 的《Numerical Methods for Ordinary Differential Equations》（2016 第三版），共 175 个数学实体（定义、定理等）。
- **对比数据集**：
  - **RepoProver** 发布的代数组合学形式化结果（基于 Gloeckle et al., 2026）
  - **M2F** 发布的分析学形式化结果（Wang et al., 2026）

这些系统也采用 agent 范式，适合横向比较。

---

### 实验设置和评估指标

#### Agent Pipeline 设置
- **LLM 模型**：
  - Planner & Worker：Claude Opus 4.6
  - Evaluator：Claude Sonnet
- **工具链**：Lean LSP、Mathlib search（LeanSearch、Loogle）、GitHub Actions 自动化
- **控制机制**：
  - 编译门控（build gate）：修改必须通过编译才能提交
  - `sorry` 数量不得上升（M2F-style verifier-certified refinement）
  - 策略合规性检查（regex 匹配 strategy.md 与实际提交）

#### 三维评估指标

| 维度 | 具体方法 |
|------|--------|
| **Semantic Correctness** | 两种 LLM-as-judge 协议：<br>1. **Direct Judgment**: LLM 判断 Lean 与原文之间的逻辑蕴含关系<br>2. **Round-trip Judgment**: Lean → Back-translate → LLM 判断 back-translation 与原文一致性<br>评分标准见附录 Rubric，输出 JSON 结构化结果 |
| **Mathlib Reuse** | 对每个 proved statement 执行 `exact?+all`，若可在 `mathlib` 下直接证明，则视为非原创贡献 |
| **Cross-file Reuse** | 1. 从教科书中提取依赖边（citation + keyword heuristic）<br>2. 使用 LLM 过滤虚假依赖<br>3. 检查 Lean 代码中是否显式引用对应声明 |

---

### 基线方法对比

| 系统 | 特点 |
|------|------|
| **RepoProver** | 多 agent 并行形式化代数组合学教科书 |
| **M2F** | 形式化数百页数学分析内容，生成 docstring 作为 NL 表示 |
| **本文（OpenMath）** | 本文提出的 agent pipeline，在数值分析领域运行 |

三者均报告低 `sorry` 和 `axiom` 数量，常被视为“高质量”。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）形式化覆盖率（Coverage）

| 指标 | 数据 |
|------|------|
| 总陈述数 | 175 |
| 完全形式化（proof-complete） | 84 (48.0%) |
| 部分形式化（statement 完成但仍有 `sorry`） | 12 (6.9%) |
| **总覆盖率** | **96 (54.9%)** |
| 最难章节（Chapter 3: Runge-Kutta） | 仅 34.8% 覆盖率 |

> 注：Chapter 3 涉及 rooted trees、order conditions、A-stability 等复杂概念。

#### （2）Proof Completeness 与 Axiom 使用（Table 1）

| System | Sorries | Axioms |
|--------|--------|--------|
| RepoProver | 5* | 0 |
| M2F | 12 | 4 |
| **OpenMath** | **0** | **0** |

> *RepoProver 的 5 个 `sorry` 是有意留作练习题。

表明本文系统在表面指标上达到“完美”——无遗留 `sorry` 或用户自定义 `axiom`。

---

### 与基线方法的对比结果

#### （1）Semantic Correctness（Table 2）

| 方法 | Faithful | Stronger | Different | Agreement |
|------|---------|----------|-----------|------------|
| **OpenMath (NL vs Lean)** | 54% | 4% | **42%** | 72.6% |
| M2F (NL vs Lean) | 73% | 2% | **25%** | 74.7% |
| RepoProver (NL vs Lean) | 82% | 1% | **18%** | 78.4% |

> “Different” 表示形式化与原文存在实质性偏离。

尽管 OpenMath 编译成功率为 100%，但高达 **42% 的陈述被判定为不忠实**。

#### （2）Mathlib Reuse（Table 3）

| System | Proofs | Overlap (%) |
|--------|--------|-------------|
| RepoProver | 6,960 | 4% (278) |
| M2F | 4,636 | 5% (230) |
| **OpenMath** | **4,264** | **2% (84)** |

说明三个系统都以**原创性贡献为主**，未过度依赖 `mathlib`。

#### （3）Cross-file Reuse（Table 4）

| Edge Type | OpenMath Reuse Rate | RepoProver Reuse Rate |
|----------|---------------------|------------------------|
| Citation only | 36% (4/11) | 62% (426/679) |
| Keyword only | 54% (12/22) | 38% (719/1852) |
| All edges | **48% (16/33)** | **45% (1164/2557)** |

表明两个系统都只实现了约一半的预期跨文件复用，存在大量冗余证明或绕路现象。

---

### 消融实验结果（Qualitative Analysis）

通过对错误案例的手动分析，发现以下**反复出现的不忠实模式**：

| 类型 | 示例 | 含义 |
|------|------|------|
| **Incomplete Multi-part Statements** | Definition 312A 定义三个权重，Lean 只实现 `derivativeWeight` | 单一定理遗漏合取项，导致语义缺失 |
| **Added Weakening Hypotheses** | Theorem 514A 添加 `h_norm_obligation`；Theorem 520B 添加 `h_inv` 和 `hY_stage` | Lean 版本更弱，无法推出原命题 |
| **Parameter Restriction** | Theorem 550A 应对任意 $ n\times n $ 矩阵成立，Lean 仅证明 $ n=1,\dots,7 $ | 有限情况不能推出通用结论 |
| **Skolemization Misjudgment** | 将存在量化改为参数输入（如 `∃C` → `M:ℝ`）被误判为“添加假设” | 实际是 Mathlib 常规做法，应视为 faithful |

> 这些问题全部被 kernel acceptance **完全掩盖**。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Kernel acceptance 是必要但远不充分的质量指标**  
   即使 `sorry=0` 且 `axiom=0`，仍可能有超过 40% 的形式化存在严重语义偏差。

2. 🔍 **现有 agent 系统在新领域表现脆弱**  
   在 `mathlib` 覆盖差的数值分析中，agent 更容易引入错误假设、参数限制和不完整形式化。

3. 📊 **语义偏离具有结构性模式**  
   不忠实并非随机噪声，而是集中在：
   - 多部分陈述的遗漏
   - 添加不必要的假设（使定理变弱）
   - 参数范围受限（破坏普遍性）

4. 🧩 **复用行为低下**  
   无论是在 `mathlib` 层还是项目内部，agent 都未能有效识别和复用已有成果，导致重复劳动。

5. ⚖️ **LLM-as-judge 具有一定可靠性，但有已知盲区**  
   在 20 个样本的人工验证中，18 个一致；主要错误在于误解 Skolemization 惯例。

---

### 方法的局限性

| 问题 | 说明 |
|------|------|
| **LLM Judge 的偏见** | 对 Skolemization、隐式 unfold、simp set 调用等不敏感，可能导致保守高估 divergence |
| **Cross-file 检测为下界** | 仅检测显式名称引用，忽略通过 `simp`、`rw` 或定义展开的隐式依赖 |
| **M2F 评估近乎自洽性检查** | 因其 NL 来源于自身生成的 docstring，非原始教材，故 round-trip 判断实为 self-consistency |
| **覆盖率不均衡** | Chapter 3 覆盖率极低，反映 agent 在高度结构化理论面前能力不足 |

---

### 未来工作方向

1. **标准化 Skolemization 处理规则**  
   在 rubric 中明确如何处理参数化存在量词，减少 LLM 误判。

2. **增强 premise selection 与 reuse 能力**  
   训练 agent 更好地检索和调用 `mathlib` 或项目内已有定理，避免重复证明。

3. **开发专用 benchmark**  
   建立涵盖不同数学领域的 autoformalization 评测集，尤其关注低 `mathlib` 覆盖区域。

4. **结合形式化与反向 informalization**  
   利用 back-translation 技术持续监控形式化保真度，形成闭环反馈。

5. **改进 agent 的长期记忆与知识管理**  
   当前 pipeline 为 stateless，未来可探索轻量级知识图谱辅助决策。

---

> **总结一句话**：  
> 本文揭示了当前 agentic autoformalization 的“皇帝新衣”——看似完美的编译成功率背后，隐藏着严重的语义失真；并提出首个系统性的三维质量审计框架，呼吁社区超越 kernel acceptance，走向更严谨、可度量的形式化评估时代。

</details>

---

### 16. [The Culture Funnel: You Can't Align What isn't in the Data](https://arxiv.org/abs/2606.13808)

**Authors**: Ananya Sahu, Mehrnaz Mofakhami, Daniel D'Souza, Thomas Euyang, Julia Kreutzer, Marzieh Fadaee  
**Category**: cs.CL  
**Published**: 2026-06-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.13808v1  

#### Abstract
Current cultural alignment approaches focus on inference-time interventions, assuming models already contain sufficient cultural knowledge. We argue modern LLM pipelines suffer from a cultural data funnel. Using a multidimensional tagging framework across pretraining, fine-tuning, alignment, and rea...

---

### 17. [Hybrid Classical-Quantum Variational Autoencoder for Neural Topic Modeling](https://arxiv.org/abs/2606.13852)

**Authors**: Ivan Kankeu  
**Category**: cs.CL  
**Published**: 2026-06-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.13852v1  

#### Abstract
Neural topic models enable scalable semantic discovery, but their integration with quantum hardware remains largely unexplored. We present a proof-of-concept hybrid classical-quantum variational autoencoder (VAE) for topic modeling, embedding parameterized quantum circuits within the VAE inference n...

---

### 18. [Can Post-Training Turn LLMs into Good Medical Coders? An Empirical Study of Generative ICD Coding](https://arxiv.org/abs/2606.13940)

**Authors**: Ziqing Wang, Weihao Li, Shijie Chen, Yuan Luo, Kaize Ding  
**Category**: cs.CL  
**Published**: 2026-06-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.13940v1  

#### Abstract
Automated International Classification of Diseases (ICD) coding is a core medical-coding task for billing, epidemiology, and clinical decision support. Generative large language models (LLMs) are often reported as weak medical coders, but this finding mainly comes from inference-time settings such a...

---

### 19. [Retrospective Progress-Aware Self-Refinement for LLM Agent Training](https://arxiv.org/abs/2606.14302)

**Authors**: Xinbei Ma, Congmin Zheng, Jiyang Qiu, Jiale Hong, Yao Yao, Xiangmou Qu, Jiaxin Yin, Xingyu Lou, Jun Wang, Weiwen Liu, Weinan Zhang, Zhuosheng Zhang, Hai Zhao  
**Category**: cs.CL  
**Published**: 2026-06-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.14302v1  

#### Abstract
LLM-based agents trained with reinforcement learning optimize step-wise action prediction but lack metacognitive awareness of task progress, inducing a gap that hinders long-horizon scaling. A pilot study reveals that online progress prompting hurts performance while retrospective demonstrations hel...

---

### 20. [Smoothing Dark Areas in Molecular Latent Diffusion](https://arxiv.org/abs/2606.13955)

**Authors**: Xi Wang, Jiahan Li, Yuxuan Xia, Yingcheng Wu, Shaoyi Zheng, Shengjie Wang  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.13955v1  

#### Abstract
Latent diffusion is a promising framework for scalable 3D molecular generation, but it requires a latent space that remains smooth, valid, and navigable beyond posterior samples. Existing molecular VAEs, however, are typically learned through reconstruction-based objectives, which do not guarantee s...

---

### 21. [A Deep Reinforcement Learning (DRL)-Based Transformer Method for Solving the Open Shop Scheduling Problem](https://arxiv.org/abs/2606.13682)

**Authors**: Faezeh Ardali, Mwembezi A. Nyelele, Gerald M. Knapp  
**Category**: cs.AI  
**Published**: 2026-06-15  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.13682v1  

#### Abstract
The open shop scheduling problem (OSSP) arises in many industrial and service settings but remains computationally challenging as the number of jobs and machines increases. While exact methods quickly become intractable, classical dispatching rules and metaheuristics may require substantial tuning t...

---

### 22. [Causal Object-Centric Models for Planning with Monte Carlo Tree Search](https://arxiv.org/abs/2606.14418)

**Authors**: Rodion Vakhitov, Leonid Ugadiarov, Alexey Skrynnik, Aleksandr Panov  
**Category**: cs.AI  
**Published**: 2026-06-15  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.14418v1  

#### Abstract
We introduce COMET (Causal Object-centric Model for Efficient Tree search), a model-based reinforcement learning algorithm that performs Monte Carlo Tree Search in a slot-structured latent space. COMET pairs a frozen unsupervised object-centric encoder with a transformer-based world model, in which ...

---

### 23. [MedLatentDx: Latent Multi-Agent Communication for Cross-Hospital Rare-Disease Diagnosis](https://arxiv.org/abs/2606.13945)

**Authors**: Ziqing Wang, Lili Zhao, Kaize Ding  
**Category**: cs.CL  
**Published**: 2026-06-15  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.13945v1  

#### Abstract
Rare diseases affect over $300$ million patients across more than $7{,}000$ conditions, yet no single hospital encounters enough cases of any one condition for reliable diagnosis. Cross-hospital collaboration could help by allowing a diagnosing institution to use distributed, case-specific diagnosti...

---

### 24. [Does the Judge Prefer English? Evaluating Language-Switching Invariance in LLM-as-a-Judge](https://arxiv.org/abs/2606.14278)

**Authors**: Shaojie Yin  
**Category**: cs.CL  
**Published**: 2026-06-15  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.14278v1  

#### Abstract
Large language models (LLMs) are now widely used as automatic judges for open-ended instruction-following evaluation. This practice is convenient, scalable, and often more semantically aware than reference-based metrics, but it also introduces a new reliability question: does a judge evaluate the qu...

---

### 25. [CORA: Analyzing and bridging thinking-answer gap in Multimodal RLVR via Consistency-Oriented Reasoning Alignment](https://arxiv.org/abs/2606.14691)

**Authors**: Jiayue Cao, Zhicong Lu, Xuehan Sun, Wei Jia, Hongling Zheng, Changyuan Tian, Zichuan Lin, Wenqian Lv, Nayu Liu  
**Category**: cs.CL  
**Published**: 2026-06-15  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.14691v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has successfully elicited the reasoning capabilities of large language models, motivating its extension to multimodal scenarios. Existing methods primarily focus on improving the visual coverage of reasoning traces and mitigating visual hallucina...

---

### 26. [Diffusion Policy Optimization without Drifting Apart](https://arxiv.org/abs/2606.13795)

**Authors**: Haozhe Jiang, Haiwen Feng, Pieter Abbeel, Jiantao Jiao, Angjoo Kanazawa, Nika Haghtalab  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.13795v1  

#### Abstract
RL post-training has become increasingly pivotal for improving diffusion policies, but existing diffusion policy-gradient methods are often unstable and cannot achieve reliable policy improvement. We identify the cause as the double-drift phenomenon: optimizing a variational surrogate can let the EL...

---

### 27. [Attention-Based Estimation of the Individual Treatment Benefit Probability under Dose Variation](https://arxiv.org/abs/2606.13821)

**Authors**: Lev V. Utkin, Andrei V. Konstantinov, Stanislav K. Kogan, Natalya M. Verbova, Maksim I. Goriunov  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.13821v1  

#### Abstract
Estimating the probability that a treatment outperforms a control for an individual patient, called the Individual Probability of Treatment Benefit (IPTB), offers a clinically intuitive alternative to population-average metrics. However, existing methods for IPTB estimation are largely confined to b...

---

### 28. [Utility-Constrained Policy Optimization](https://arxiv.org/abs/2606.14029)

**Authors**: Mehrdad Moghimi, Bernardo Avila Pires  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.14029v1  

#### Abstract
Constrained MDPs (CMDPs) are a widely adopted framework for incorporating safety into RL agents; however, the framework does not support risk-sensitive constraints. This can be problematic: For example, CMDPs allow for optimal solutions that, in order to satisfy the risk-neutral constraints, mix inf...

---

### 29. [MUFFLe: Efficient Model Update Compression via Generalized Deduplication for Federated Learning](https://arxiv.org/abs/2606.14354)

**Authors**: Xiaobo Zhao, Daniel E. Lucani  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.14354v1  

#### Abstract
Federated learning is well suited to edge environments but is often limited by the uplink cost of transmitting model updates. This Work-in-Progress paper presents MUFFLe, a communication-efficient update compression scheme that integrates generalized deduplication (GD) into the FedAvg pipeline. MUFF...

---

### 30. [SemPiper: Interactive Code Synthesis for Semantic Operators in Machine Learning Pipelines](https://arxiv.org/abs/2606.14361)

**Authors**: Olga Ovcharenko, Luciano Duarte, Sebastian Schelter  
**Category**: cs.LG  
**Published**: 2026-06-15  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2606.14361v1  

#### Abstract
Machine learning (ML) pipelines require extensive data preparation, feature engineering, and integration across heterogeneous sources, making them tedious and error-prone to develop. While large language models (LLMs) have recently shown promise for assisting programming tasks, chat-based interfaces...

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

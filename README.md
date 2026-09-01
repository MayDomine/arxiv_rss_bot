# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-01 10:30:48 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Ceiling-Clipped Acceptance Histograms Indicate Stranded Speed-up in Block-Diffusion Speculative Decoding](https://arxiv.org/abs/2608.30427)

**Authors**: Ephrem Wu  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2608.30427v1  

#### Abstract
Speculative decoding speeds up generation with an efficient draft model (drafter) that proposes tokens for a target model to verify in one pass, preserving the target's output distribution. High-acceptance block-diffusion drafters such as DFlash and DFlare fill an entire block in one parallel pass. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Ceiling-Clipped Acceptance Histograms Indicate Stranded Speed-up in Block-Diffusion Speculative Decoding*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文揭示并解决了一个在 **block-diffusion speculative decoding** 中被忽视的关键瓶颈——“**stranded speed-up**”（搁浅加速）。  
现有的高接受率 drafters（如 DFlash 和 DFlare）虽然平均 **committed length** 较高，但其性能可能受到预设 **block size** 的限制。当目标模型（target）在多个周期中都接受了整个 block 的所有 token 时，说明 drafter 具备生成更多可接受 token 的潜力，但由于 block 已满而无法展现，这部分潜在的加速机会被“搁浅”。

传统评估仅依赖平均 committed length，无法区分是 **draft-limited**（生成质量差导致拒绝）还是 **block-limited**（块大小不足导致截断）。

### 提出了什么新方法或新思路
作者提出了两个核心创新：

1.  **诊断工具：Acceptance Histogram 的 Ceiling Bin 分析**
    *   提出通过观察每个解码周期的 **acceptance histogram** 来诊断性能瓶颈。
    *   特别关注最右侧的 **ceiling bin**（即 `n = B-1` 的 bin），它表示目标模型完全接受整个 block 的频率。
    *   **ceiling bin 的“尖峰”** 类似于照片的“过曝”，表明存在大量被截断的接受序列，暗示扩大 block size 可以释放额外的加速潜力。

2.  **解决方案：DBloom 方法**
    *   提出 **DBloom**，一种高效地将已训练好的 B16 drafter 扩展到更大 block size（如 B24）的方法。
    *   采用 **curriculum post-training** 策略，使用一个约 30K prompts 的小型扩展语料库，专门针对新暴露的位置进行强化训练。
    *   在损失函数上对新位置（offset 16-23）施加 **3× 的权重提升**，以快速学习这些位置，同时保护已学好的前段位置。

### 相比现有方法的优势
*   **诊断先行**：提供了一种低成本的“预检”手段，在投入昂贵的再训练之前就能判断扩大 block 是否值得。
*   **数据高效**：DBloom 仅需约 1-2% 的原始训练数据量即可完成从 B16 到 B24 的扩展，成本远低于从头训练。
*   **性能显著提升**：成功解锁了被“搁浅”的加速潜力，在高 ceiling 分数的基准测试上实现了显著的 committed length 增长。
*   **竞争力强**：扩展后的线性 B24 drafter 在性能上可与更复杂的 **tree-based** 方法（如 JetSpec）相媲美，甚至超越。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在以下七个基准数据集上进行：
*   **GSM8K**
*   **MATH-500**
*   **AIME21-26** (竞赛数学)
*   **HumanEval**
*   **MBPP**
*   **LiveCodeBench (LCB)**
*   **MT-Bench**

### 实验设置和评估指标
*   **目标模型 (Target)**：Qwen3-8B, Qwen3-4B, 以及用于跨家族验证的 Gemma-4-12B-IT。
*   **drafter 模型**：基于 DFlash 和 DFlare 架构的 B16 原始模型，并应用 DBloom 扩展至 B24。
*   **解码方式**：贪婪解码 (greedy decoding)，温度为 0。
*   **主要评估指标**：
    *   **Committed Length (τ)**：每个解码周期内，目标模型最终提交的 token 数量（包括接受的 draft token 和保证的 bonus token）。
    *   区分两种统计方式：
        *   **Prompt-mean committed length (T)**：按提示词（prompt）平均。
        *   **Cycle-mean committed length (E[n]+1)**：按解码周期（cycle）平均。
*   **对比方式**：所有模型均在统一协议下重新评估，确保比较的公平性。

### 基线方法对比
*   **内部基线**：原始的 B16 drafter（DFlash 和 DFlare）。
*   **外部强基线**：**JetSpec**，一种先进的、高接受率的 **tree-based speculative decoding** 方法。将其作为“设计外检查”（out-of-design check），以验证 DBloom 的通用竞争力。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
1.  **Ceiling Bin 与增益高度相关**：
    *   扩展前的 **ceiling bin 分数** 与扩展后获得的 **committed length 增益** 呈现极高的 **Spearman 秩相关性**（ρ = +0.88 至 +0.93）。
    *   这证明了 ceiling bin 是一个有效的预测指标，能准确指出哪些任务能从 block 扩展中获益最多。

2.  **DBloom 显著提升性能**：
    *   在 **Qwen3-8B/4B** 上，将 DFlare drafter 从 B16 扩展到 B24（Arm-B），在高 ceiling 基准上的 **prompt-mean committed length (T)** 平均提升了 **+0.8 tokens**，最高达 **+1.37 tokens**。
    *   在 **Gemma-4-12B-IT** 上，同样扩展也带来了显著提升（中位数 **+0.41 tokens**）。

3.  **与 JetSpec 的对比结果**：
    *   在与 **JetSpec** 的成对、提示词匹配的比较中，**DBloom-DFlare B24** 在所有基准测试上均表现出色。
    *   对比 JetSpec 最大 256 节点的树预算，DBloom **没有出现任何统计上显著的性能损失**。
    *   在多数基准上，DBloom 表现持平或优于 JetSpec，尤其是在较低的树预算下优势明显。

### 与基线方法的对比结果
*   **vs. Naive Block Expansion**：直接在推理时将 B16 drafter 用于 B24 会严重损害性能（速度下降 4%-52%），因为 bidirectional attention 导致分布偏移。
*   **vs. Original B16**：经过 DBloom 训练的 B24 drafter 在所有高 ceiling 基准上均显著优于原始 B16 drafter。
*   **vs. JetSpec**：如上所述，性能相当甚至更优，证明了简单的线性扩展策略的有效性。

### 消融实验结果
*   **损失函数权重消融**：实验了多种 per-position loss weighting 方案（指数衰减、前沿敏感等），发现简单的 **flat 3× boost** 对新位置的效果最好且最稳定，其他复杂方案并未带来额外收益。
*   **数据 vs. 扩展控制**：实验证明，性能提升主要来自 **block size 的扩大**，而非仅仅是增加了 30K 的训练数据。在 B16 上用相同数据微调仅带来微小增益（+0.05 至 +0.21），远小于扩展到 B24 的增益（+0.26 至 +1.02）。
*   **种子鲁棒性**：在不同随机种子下重复训练，结果高度一致（差异 < 0.03 tokens），证明了方法的稳定性。

---

## 4. 关键结论和发现

### 论文的主要发现
1.  **平均指标具有欺骗性**：仅看平均 **committed length** 会掩盖 **block-limited acceptance** 的问题。**Acceptance histogram 的 ceiling bin 是一个关键的诊断信号**。
2.  **盲目扩展无效**：在推理时简单地增大 block size 而不重新训练会因 **bidirectional attention** 引起的分布偏移而导致性能急剧下降。
3.  **高效扩展可行**：通过 **DBloom** 方法，可以用极少的额外数据（约 1-2%）和计算成本，将已有的 B16 drafter 成功扩展到 B24，从而解锁被“搁浅”的加速潜力。
4.  **线性可与树竞争**：一个经过良好扩展的线性 B24 drafter 在性能上可以与复杂的 256 节点树状 drafter（如 JetSpec）相匹敌，挑战了“必须用树才能获得高性能”的观念。

### 方法的局限性
*   **适用范围有限**：研究主要集中在 Qwen3 和 Gemma 家族，其普适性尚未在更广泛的模型架构上得到充分验证。
*   **天花板效应**：随着 block size 增大，ceiling bin 分数会下降，表明继续扩大的边际效益递减。目前停在 B24，更大的 block 可能需要更长的课程学习。
*   **依赖高质量基线**：DBloom 是一种后训练（post-training）方法，其效果依赖于一个已经训练良好的 B16 drafter 作为起点。

### 未来工作方向
*   **探索更大的 block size**：研究如何通过更长的课程学习或其他技术来有效训练 B32 或更大的 block。
*   **通用化和自动化**：将 DBloom 的流程（诊断 -> 扩展）自动化，并验证其在更多模型家族和任务上的有效性。
*   **结合其他优化**：将 block-horizon 扩展与其他优化策略（如 CaDDTree 的动态树预算优化）相结合，实现端到端的最优吞吐量。
*   **理论分析**：对 block-diffusion 模型在不同 block size 下的分布偏移进行更深入的理论建模和分析。

</details>

---

### 2. [Sensitivity-Constrained Neural Operators for Data-Efficient Forward and Inverse Modeling of Partial Differential Equation Systems](https://arxiv.org/abs/2608.29888)

**Authors**: Abdolmehdi Behroozi, Chaopeng Shen, Daniel Kifer, Kathryn Lawson  
**Category**: cs.LG  
**Published**: 2026-09-01  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2608.29888v1  

#### Abstract
Neural operators provide fast surrogates for partial differential equation (PDE) solvers, but their reliability can degrade for high-dimensional spatial inputs and inverse or repeated inference. State-only training constrains solution values but not the learned input--output response. We study sensi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sensitivity-Constrained Neural Operators for Data-Efficient Forward and Inverse Modeling of Partial Differential Equation Systems

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **Neural Operators** 在处理高维空间场输入（如初始条件、边界、材料参数等）时存在以下关键缺陷：
- **仅依赖状态值监督（state-only training）** 导致模型虽然能较好预测解场，但学习到的输入-输出敏感性（sensitivity）不准确。
- 这种不准确的敏感性在 **inverse problems**（反演问题）、**autoregressive rollout**（自回归推演）和 **out-of-distribution**（OOD）场景中会导致严重错误，例如梯度误导、误差累积和泛化失败。

### 提出的新方法：Sensitivity-Constrained Neural Operators (SC-NO)
提出 **SC-NO** 框架，在标准 Neural Operator 训练基础上，引入 **solver-derived Jacobian supervision**（求解器导出的雅可比矩阵监督）：
- **核心思想**：不仅让模型匹配 PDE 的解场 $ u $，还强制其匹配求解器计算出的 **输入-输出敏感性** $ \frac{\partial u}{\partial p} $，其中 $ p $ 是高维输入场（如初始浓度、涡量、海底变形等）。
- **关键技术**：采用 **sampled Jacobian supervision**（采样雅可比监督），即在每个训练批次中只随机采样部分雅可比项进行监督，避免直接处理完整雅可比矩阵带来的巨大计算开销。

### 相比现有方法的优势
| 方面 | 优势 |
| :--- | :--- |
| **方法论** | 不是提出新架构，而是为现有 NO 架构（FNO, WNO, DeepONet）提供一种通用的、可插拔的训练增强策略。 |
| **效率** | 通过采样机制，实现了 **full-field sensitivity information** 的摊销（amortization），在可控的计算成本下获得大量敏感性信息。 |
| **效果** | 显著提升模型在 **低数据量** 和 **高维输入** 场景下的性能，尤其在 **inverse reconstruction** 中增益远超前向预测。 |

---

## 2. 核心实验方法和设置

### 使用的数据集和基准问题
论文设计了三个层次的实验来验证 SC-NO 的有效性：

1.  **PDE1: Advection-Diffusion Equation**
    - **任务**：给定初始浓度 $ C_0(x) $ 和速度场 $ u(x) $，预测浓度随时间的演化。
    - **输入维度**：高维空间场。
    - **目的**：控制变量，研究前向预测和反演重建。

2.  **PDE2: RANS-Spalart-Allmaras Turbulent Flow**
    - **任务**：给定初始涡量 $ \Omega_0(x) $ 和外力场 $ f(x) $，预测湍流演化。
    - **特殊设置**：进行了 **input-dimensionality scaling** 实验，系统性地改变 $ \Omega_0(x) $ 的内在分辨率（从 4x4 到 64x64），以研究维度扩展对性能的影响。
    - **目的**：评估方法在复杂物理系统和不同输入维度下的表现。

3.  **PDE3: 2D Shallow Water Equations for Tohoku Tsunami**
    - **任务**：一个大规模真实世界应用案例。给定地震引起的海底变形场 $ \Delta z_b(x) $，预测海啸波传播。
    - **反演挑战**：利用稀疏的早期浮标观测（gauge observations）反演未知的 $ \Delta z_b(x) $，并进行后续预报。
    - **目的**：作为“压轴应用”（capstone application），测试方法在时间紧迫、数据稀疏、高维反演中的实用性和鲁棒性。

### 实验设置和评估指标
- **总计算成本会计**（Crucial）：论文明确将 **总计算成本** $ C_{total} = C_{data} + C_J + C_{train} $ 作为比较基准，其中 $ C_J $ 是生成求解器雅可比的成本。这确保了比较的公平性。
- **评估指标**：
    - **前向预测**：Relative L2 Error, MAE。
    - **反演重建**：Relative L2 Error of the inferred input field (e.g., $ C_0 $, $ \Omega_0 $, $ \Delta z_b $)。
    - **长时程推演**：Rollout Horizon 上的累积 Relative L2 Error。
    - **鲁棒性**：在含噪声观测下的反演 R² 分数。

### 基线方法对比
- **Baseline**: 标准的 FNO, WNO, DeepONet。
- **Proposed**: 对应的 SC-FNO, SC-WNO, SC-DeepONet。
- **对比方式**：在相同网络架构、超参数和总计算成本（或训练样本量）下进行对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### 前向预测 (Forward Prediction)
- **PDE1 & PDE2**：SC-NO 在所有样本量下均优于基线，尤其是在 **低数据量**（如 100-200 样本）时优势最明显。
- **PDE3 (Tsunami)**：在 1000 个训练样本下，SC-FNO 的 Relative L2 Error 约为 **0.062**，而 FNO 约为 **0.089**，性能提升显著。

#### 反演重建 (Inverse Reconstruction)
这是 SC-NO 收益最大的领域。
- **PDE1 & PDE2**：SC-NO 的反演误差远低于基线。例如，在 PDE2 中，当使用 1000 个样本时，SC-FNO 的反演 L2 Error 降至约 **0.061**，而 FNO 高达 **0.96**，差距超过一个数量级。
- **PDE3 (Tsunami)**：SC-FNO 能更准确地恢复海底变形的主位置、极性和大尺度几何结构，而 FNO 的估计则较弱且更弥散。

#### 长时程自回归推演 (Autoregressive Rollout)
- **PDE2**：在 25 秒的推演中，标准 FNO 的最终累积误差约为 **0.276**，而 SC-FNO 仅为 **0.086**，实现了 **约 3.2 倍的误差降低**，证明了其在长期稳定性上的巨大优势。

#### 消融实验结果 (Ablation Studies)
1.  **Jacobian Sampling Density**：实验表明，即使只使用少量（如 16x16）的雅可比采样，也能带来显著的性能提升，且随着采样密度增加，性能持续改善直至饱和。这证明了 **采样策略的有效性**。
2.  **State vs. Jacobian Information**：实验分离了状态值监督和雅可比监督的作用。结果显示，两者提供的信息是 **互补的**（complementary）。仅增加状态数据无法达到加入雅可比监督的效果，反之亦然。这强有力地支持了 SC-NO 的核心价值。

---

## 4. 关键结论和发现

### 主要发现
1.  **敏感性监督至关重要**：对于高维 PDE 建模，仅匹配解场是不够的。显式约束模型的 **输入-输出敏感性** 是提升其在反演、OOD 和长时程任务中可靠性的关键。
2.  **SC-NO 显著提升数据效率**：在低数据量场景下，SC-NO 通过从每个模拟轨迹中提取更多信息（敏感性），大幅降低了对海量训练数据的需求。
3.  **反演性能提升最大**：SC-NO 在 **gradient-based inverse problems** 中的收益最为显著，因为它直接提供了更准确的梯度信息，使优化过程更加稳定和高效。
4.  **提升鲁棒性和稳定性**：SC-NO 在 **out-of-distribution** 测试和 **noisy observations** 下表现出更强的鲁棒性，并在 **autoregressive rollout** 中展现出卓越的长期稳定性。
5.  **通用性强**：该方法在 FNO, WNO, DeepONet 等多种 NO 架构上均有效，证明了其作为一种通用训练范式的潜力。

### 局限性
1.  **依赖求解器的敏感性信息**：SC-NO 需要访问能够计算雅可比（通过自动微分或离散伴随法）的 PDE 求解器。这对于许多现有的黑盒求解器是一个障碍。
2.  **预处理开销**：计算和存储求解器的雅可比会带来额外的 $ C_J $ 成本，尽管论文证明了其性价比高，但这仍是需要付出的代价。
3.  **非万能药**：SC-NO 并未解决 PDE 求解的“维度诅咒”（curse of dimensionality），它是在特定计算预算下改善了数据-计算权衡，而非从根本上改变复杂度。
4.  **理想化假设**：实验基于精确的求解器雅可比。在实践中，如果求解器本身有误差或近似，这种监督信号的质量也会下降。

### 未来工作方向
1.  **扩展到更多 PDE 类型**：将 SC-NO 应用于更广泛的科学计算领域，如生物医学、量子力学或多相流。
2.  **探索更高效的敏感性获取方式**：研究如何用更低的成本（如近似伴随法、随机投影）生成有效的敏感性监督信号。
3.  **结合其他物理信息**：将 SC-NO 与 **PDE-residual constraints** 或 **equivariance** 等其他物理引导学习方法相结合，构建更强大的混合模型。
4.  **不确定性量化**：在 SC-NO 框架内发展不确定性量化能力，以更好地评估反演结果的可信度。
5.  **端到端实时系统集成**：将 SC-NO 集成到真实的预警系统（如海啸、洪水预警）中，实现从观测到预报的全流程分钟级响应。

</details>

---

### 3. [A.X K2 Technical Report](https://arxiv.org/abs/2608.30181)

**Authors**: Cheolseung Baek, Dhammiko Arya, Eunki Kim, Gun Song, Gyoungeun Han, Hyunho Yang, Hyunjun Eun, Jin Kim, Junyoung Park, Juyun Wee, Minki Hong, Minkyung Park, Minsang Kim, Minsoo Kang, SaeRom Kim, Sangjin Kim, Sangyeol Lee, Seojin Lee, Seokhwan Jo, Seokyoung Hong, Seongho Choi, Seonghye Cho, Seongmin Ok, Sereimony Sek, Seungmo Cho, Seungsik Kim, Singon Kim, Sohee Park, Sooyeon Park, Subin Yi, Sungbin Yoon, Sungeun Lee, Sung Jun Cheon, Sungwan Kim, Sunwoo Lee, Tae Yoon Kim, Wonbeom Jang, Yohan Ra, Yong-jin Han, Youngjin Kim, Youngrang Kim, Yujin Kang, Yujin Lee  
**Category**: cs.AI  
**Published**: 2026-09-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.30181v1  

#### Abstract
We introduce A.X K2, a 688B-parameter Mixture-of-Experts (MoE) language model trained from scratch as a high-performance foundation for \emph{agentic} applications. Trained on approximately 8.5T tokens---fewer than its predecessor, A.X K1---on a smaller but higher-quality mixture with substantially ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# A.X K2 Technical Report 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
A.X K2 旨在解决**大规模语言模型在实现高智能代理能力（agentic competence）时面临的效率、可控性和部署成本之间的权衡问题**。具体挑战包括：
- 长上下文推理带来的计算开销和延迟；
- 复杂推理模式与简单响应模式难以统一控制；
- 大规模 MoE 模型训练不稳定、低精度部署精度损失大；
- 对韩国语言和文化理解不足，依赖国外闭源系统。

### 提出的新方法与新思路
1. **Sparse Gated Attention (SGA)**  
   - 结合 **sparse attention**（通过轻量级 indexer 选择 top-k tokens）与 **gated attention**（head-specific 输出门控），显著降低长上下文下的注意力计算量。
   - 引入 **sparse indexer warmup** 策略：直接优化 indexer 在其自身的稀疏 top-k 选择上，而非先拟合密集注意力分布，大幅降低适配成本。

2. **Gated Norm (GN)**  
   - 在 RMSNorm 后引入输入相关的门控机制，抑制隐藏状态中的 outlier，提升训练稳定性，并显著改善 FP8 和 NVFP4 等低比特格式下的量化鲁棒性。

3. **Think-Fusion SFT Recipe**  
   - 一种监督微调策略，使用成对的“思考”与“非思考”响应进行训练，使单一模型能通过显式控制 token 切换推理模式，实现用户可控制的推理深度。

4. **Checkpoint Merging with WSM**  
   - 采用 Warmup-Stable-Merge (WSM) 风格的权重平均，融合最后多个检查点，在不增加训练成本的前提下提升泛化能力和鲁棒性。

### 相比现有方法的优势
| 方法 | A.X K2 优势 |
|------|------------|
| Dense Attention | SGA 将每个 query 的 KV 访问从 $O(n)$ 降至固定 2,048 个位置，在 128K 上下文中仅占 1.6%，实现近似线性扩展 |
| Dual Normalization | GN 单一归一化即可稳定训练，简化架构，避免冗余设计 |
| Dual-Track Training | Think-Fusion 统一训练路径，防止因数据长度差异导致模式混淆 |
| Post-hoc Quantization | 支持原生 FP8 训练与发布，无需额外量化步骤；NVFP4 保留 99% FP8 性能 |

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 预训练数据（约 8.5T tokens）
- **Web 文本**：Nemotron-CC v2.1、FineWeb2、自建韩语爬虫数据
- **代码**：Nemotron-Pretraining-Code-v2
- **学术文献**：PDF 解析的 STEM、医学、工程等领域论文
- **合成数据**：基于强开源模型生成的推理轨迹、工具使用数据
- **指令数据**：SFT 格式的任务描述与响应，用于预训练后期强化指令遵循

#### 微调与评估数据
- **SFT 数据**：包含数学、科学知识、编程、代理行为、通用对话、安全对齐等类别的成对“思考/非思考”样本（共 9M+ 样本）
- **RL 数据**：涵盖指令遵循、人类偏好判断、工具调用、安全性等多目标奖励信号
- **评估基准**：
  - 数学：AIME26, Apex, KMO26
  - 韩语理解：KMMLU-Pro, CLIcK, KoBALT
  - 编码：LiveCodeBench v6, SciCode
  - 科学与知识：GPQA Diamond, Humanity's Last Exam
  - 长上下文：RULER, LongBench v2, AA-LCR
  - 代理能力：t²-Bench Telecom, GDPval (Elo), BrowseComp

### 实验设置与评估指标
| 类别 | 设置 |
|------|------|
| 模型参数 | MoE 架构，总参数 688B，激活参数 33B，256 个专家中路由 8 个 |
| 上下文长度 | 原生训练至 128K，通过 YaRN 扩展至 256K |
| 推理模式 | 支持 `<think>...</think>{response}`（思考）与 `</think>{response}`（非思考）两种模式 |
| 评估方式 | 多数任务报告 pass@1 平均值（8 次采样）；部分来自 Artificial Analysis 的第三方评测 |
| 服务配置 | 使用 vLLM + B200 GPU 节点，测试吞吐量、内存占用、延迟等实际部署指标 |

### 基线方法对比
- **A.X K1**：前代模型（519B 参数）
- **Qwen3.5-397B-A17B**
- **DeepSeek-V4 Flash**
- **GLM-5.1**
- **Kimi-K2.6**
- **MiniMax-M2.7**

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | A.X K2 表现 |
|------|-----------|
| **数学能力** | AIME26: **97.1%**, Apex: **45.8%**, KMO26: **92.5%** |
| **韩语能力** | KMMLU-Pro: **80.5%**, CLIcK: **91.6%** |
| **长上下文理解** | RULER Overall: **94.6**（最高达 256K） |
| **代理能力** | t²-Bench Telecom: **98.0%**, GDPval Elo: **1031** |
| **低比特精度保持** | NVFP4 保留 **99.0%** 的 FP8 准确率 |
| **长上下文检索** | 在 512K 下仍实现 **100%** 的 Needle-in-a-Haystack 成功率 |

### 与基线方法的对比结果
| 基准 | A.X K2 vs 最佳基线 |
|------|------------------|
| **AIME26** | 优于所有模型（第二名为 96.7%） |
| **Apex** | **+32.3pp** 超越 Qwen3.5（13.5%） |
| **KMMLU-Pro** | **+1.7pp** 超越 Qwen3.5（78.4%） |
| **CLIcK** | **+2.8pp** 领先 GLM-5.1（88.8%） |
| **t²-Bench Telecom** | 显著领先，达到 **98.0%**（第二为 97.7%） |
| **AA-LCR** | 达到 66.0，较 A.X K1 提升 **+40pp** |

### 消融实验结果
| 实验项 | 发现 |
|-------|------|
| **SGA 对 LongBench 影响** | 引入稀疏注意力后，LongBench v1 分数从 62.80 → 62.99（+0.19），表明质量无损 |
| **GN 对量化影响** | NVFP4 服务时误差小于 1 point，远优于传统方案 |
| **Think-Fusion 控制有效性** | 在非思考模式下仍能正确执行指令，且不会误触发长链推理 |
| **Checkpoint Merging** | 融合模型比任一单独检查点更强，验证 flat minima 更优 |

---

## 4. 关键结论和发现

### 主要发现
1. **Token 效率大幅提升**：尽管训练 token 数量少于 A.X K1（8.5T vs ~10T），但在多数基准上表现更优，尤其在 Apex 上提升超 30 个百分点，证明高质量数据混合与课程学习的有效性。
2. **长上下文可高效支持**：SGA 实现了近乎恒定的注意力开销，使得 A.X K2 在 128K 上下文中每 query 仅读取 2,048 个位置，同时保持 RULER 94.6 的高分。
3. **推理模式可灵活切换**：Think-Fusion 允许用户在同一模型内自由选择是否启用复杂推理，兼顾响应速度与准确性。
4. **低比特部署可行**：得益于 GN 和 SGA 的 outlier 抑制，NVFP4（W4A4）部署几乎无损（99% FP8 精度），极大降低服务成本。
5. **韩国语境理解领先**：在多个韩语专项测试中排名第一，体现其作为国家主权 AI 的战略价值。

### 方法的局限性
- **基础设施限制**：受限于 EP=8 的 intra-node NVLink 设计，未能探索更高程度的专家并行。
- **模态单一**：当前仅为 text-only 模型，缺乏视觉或多模态理解能力。
- **资源约束下的性能折衷**：受固定算力预算限制，未完全发挥更大规模潜力。
- **思考模式的安全风险**：在红队测试中，思考模式下的攻击成功率上升至 33.9%（非思考为 12.1%），说明深层推理可能被恶意利用。

### 未来工作方向
- **扩展至万亿参数级别**：继续推进 MoE 规模化训练，构建更强大的基础模型。
- **引入原生多模态能力**：集成图像、音频等模态，打造真正的 agentic 多模态智能体。
- **增强跨语言一致性**：进一步缩小不同语言间的性能差距，提升全球适用性。
- **优化推理安全性**：开发更精细的思考控制机制，在保证能力的同时防范滥用。
- **推动 Sovereign AI 生态建设**：围绕 A.X K2 构建完整的国产 AI 工具链与应用生态。

--- 

> ✅ **总结一句话**：  
> **A.X K2 是一个兼具高性能、高效率与高可控性的前沿 MoE 模型，它不仅在数学与韩语任务上达到领先水平，还通过 SGA、GN 和 Think-Fusion 等技术创新，解决了长上下文推理、低比特部署与推理模式控制等关键挑战，为下一代 agentic AI 提供了实用且可扩展的基础。**

</details>

---

### 4. [TuringLLM: Efficiently Scaling Foundation Models Toward Physical AI](https://arxiv.org/abs/2608.30567)

**Authors**: Yuheng Zhang, Yizhao Wang, Da Zhu, Hua Zhou, Yue He, Jiahui Hu, Shaman Tang, Hanlin Chen, Yuhua Wei, Anhua Liu, Shuang Su, Rui Xin, MingYuan Wang, MingHao Li, HaoJie Yang, Siqi Liu, Jianlei Zheng, WeiChao Huang, Qiman Wu, Hang Zhang, HongGou Yang, Xianming Liu  
**Category**: cs.AI  
**Published**: 2026-09-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.30567v1  

#### Abstract
We present Turing-20B-A2B, a 20B-parameter Mixture-of-Experts language model that activates approximately 2B parameters per token, designed for long-context and latency-sensitive physical AI applications. The model adopts Quantile Routing in a dynamic top-k configuration, enabling token-adaptive exp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TuringLLM: Efficiently Scaling Foundation Models Toward Physical AI**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在**Physical AI**（如自动驾驶、具身智能）系统中，基础模型需要同时满足三个关键要求：
- **强大的通用能力**（广博的世界知识与推理能力）
- **高效的长上下文建模**（处理长时间观测序列）
- **低延迟推理**（满足闭环交互的实时性约束）

然而，传统大模型在扩展规模时面临计算成本高、推理延迟大、长上下文效率差等问题。现有 MoE 和高效注意力机制虽有改进，但在部署层面仍存在负载不均、执行不规则等挑战。

---

### **提出的新方法与创新思路**

Turing-20B-A2B 是一个 **20B 参数的 Mixture-of-Experts (MoE)** 模型，每 token 激活约 **2B 参数**，专为物理 AI 场景设计。其核心创新包括：

#### ✅ **动态 Top-k Quantile Routing**
- 采用 **Quantile Routing** 动态决定每个 token 激活多少个专家（routed experts），而非固定 k。
- 通过在线跟踪 router score 分布的分位数来设定专家阈值，实现：
  - **平衡的专家利用率**
  - **可控的平均计算预算**
  - **token 自适应的专家分配**（难样本获得更多计算资源）

#### ✅ **混合注意力架构（Hybrid Attention）**
- 大部分层使用 **Lightning Attention**（线性复杂度）进行高效长上下文处理。
- 每隔五层插入一层 **Full Attention** 层（共 4 层），保留全局 token 交互能力。
- 架构比例为：**20 层 Lightning Attention + 4 层 Full Attention**

#### ✅ **容量约束路由（Capacity-Constrained Routing）用于部署**
- 在预训练阶段保持 **dropless routing**（无丢弃），避免因果依赖问题。
- 在 **prompt prefill 部署阶段引入专家容量限制**（capacity factor γ=1.25），提升执行规律性和硬件效率。

#### ✅ **渐进式长上下文训练 + 推理时外推**
- 原生支持 **128K 上下文长度**，通过两个阶段继续预训练完成：
  - 4K → 32K（LC-1）
  - 32K → 128K（LC-2）
- 使用 **YaRN** 在推理时进一步扩展至 **512K**，无需额外参数更新。

#### ✅ **硬件友好设计（Hardware-Model Co-Design）**
- 使用 **Dynamic Tanh (DyT)** 替代 RMSNorm，简化部署。
- 整体结构轻量，适配边缘设备（如自研 Turing 芯片）。

---

### **相比现有方法的优势**

| 维度 | 优势说明 |
|------|----------|
| **效率 vs. 能力平衡** | 激活仅 ~2B 参数/token，性能超越 Qwen3-8B Base，接近 Qwen3.5-9B Base |
| **长上下文扩展性** | 原生存量训练至 128K，优于直接使用 YaRN 扩展的模型（如 Qwen3 系列） |
| **推理延迟控制** | 在 128K 上下文下，prefill latency 比 Qwen3 快 **2.2×**，比 Qwen3.5 更显著 |
| **部署稳定性** | 容量约束使专家负载更规则，适合实际系统调度 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

#### **预训练语料（三阶段课程学习）**
| 阶段 | 数据构成 |
|------|--------|
| **Stage 1: Knowledge Foundation** | DCLM, FineWeb-Edu, FineMath, InfiMM-WebMath, OpenWebMath, The Stack v2, peS2o, Wikipedia, Cosmopedia v2 |
| **Stage 2: Capability Enhancement** | 过滤后的高质量网页/数学/代码数据 + MegaMath + Stack-Edu |
| **Stage 3: Quality Annealing** | 内部构建的高质量合成数据（数学、代码、知识为主）+ 少量书籍/论文 |

> 总训练量约为 **9.5T tokens**，主预训练耗时约 22 天（512 × H800 GPU）

#### **长上下文专项训练数据**
- 保留 45% Stage 3 数据分布
- 合成任务（20%）：needle-in-a-haystack、频率聚合、长问答
- 自然长文本（35%）：ProLong/FineWeb 中的长书、教科书、Scale-SWE-Distilled 代码数据、LongAlign 指令数据

---

### **实验设置与评估指标**

#### **模型配置**
- 总参数：**20B**
- 激活参数/token：**~2B**
- MoE 结构：256 个 routed experts + 1 个 shared expert（dim=2048）
- 平均激活 routed experts 数：**~8**
- 原生上下文长度：**128K**
- 推理扩展长度：**512K（via YaRN）**

#### **评估基准**
| 类别 | 基准 |
|------|------|
| **知识** | MMLU, MMLU-Redux, CMMLU, C-Eval |
| **推理** | MMLU-Pro, BBH, DROP, WinoGrande, HellaSwag |
| **数学与 STEM** | ARC-C, GPQA, GSM8K, MATH |
| **长上下文能力** | RULER（8K–512K） |
| **推理效率** | Prefill Latency（FP16, 单 H800 GPU, batch=1） |

#### **基线对比模型**
- **Qwen3 系列**：Qwen3-1.7B, Qwen3-4B, Qwen3-8B
- **Qwen3.5 系列**：Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B

所有模型使用 **OpenCompass** 统一评估协议（生成式评测，相同 prompt 模板、few-shot 示例、答案提取逻辑）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **总体能力（Base Model 阶段）**
| 模型 | MMLU | CMMLU | BBH | MATH | 综合得分 |
|------|------|-------|-----|------|---------|
| Qwen3-8B Base | 78.92 | 81.28 | 73.67 | 56.12 | — |
| **Turing-20B-A2B** | **79.83** | **84.17** | **73.61** | **62.20** | **超过 Qwen3-8B，接近 Qwen3.5-9B** |

> 尽管激活参数仅为 ~2B/token，综合能力已**超越 Qwen3-8B**，逼近更大的 **Qwen3.5-9B**

---

#### ✅ **长上下文能力（RULER 评估）**

| 模型 \ Context | 8K | 16K | 32K | 64K | 128K | 256K | 512K |
|---------------|----|-----|-----|-----|------|------|------|
| Qwen3.5-9B | 97.72 | 96.97 | 96.05 | 93.01 | 88.89 | 81.81 | — |
| **Turing-20B-A2B** | 93.80 | 93.45 | 93.16 | **90.00** | **84.87** | **81.31** | **77.38** |

> - 在 ≤64K 时略低于 Qwen3.5-9B，但从 **64K 开始反超 Qwen3.5-4B**
> - 在 **256K 时接近 Qwen3.5-9B**
> - **512K 仍保持 77.38**，显示 YaRN 外推稳定

---

#### ✅ **推理效率（Prefill Latency）**

| 输入长度 | 速度提升（vs Qwen3） | 速度提升（vs Qwen3.5） |
|--------|------------------|--------------------|
| 32K    | **1.2×**          | —                  |
| 64K    | **1.6×**          | —                  |
| 128K   | **2.2×**          | 显著优于 Qwen3.5-4B/Qwen3.5-9B |
| 256K   | —                | 优势持续扩大        |

> 得益于 Lightning Attention 占主导地位（20/24 层），二次项增长被有效抑制。

---

### **消融实验结果**

#### 🔍 **5.1 MoE Routing 策略对比（Quantile vs Loss-Free）**
- 使用 10B MoE 模型（~1.6B 激活参数/token），对比两种路由策略：
  - **Loss-Free Routing**：固定 top-8
  - **Quantile Routing**：动态激活数量，平均 ~8

| 指标 | Quantile Routing | Loss-Free Routing |
|------|------------------|-------------------|
| **早期训练性能（MMLU）** | **24.06** | 23.27 |
| **HellaSwag** | **25.18** | 22.81 |
| **MBPP** | **18.78** | 14.81 |
| **最大专家负载偏差（MaxVio）** | **更低且更稳定** | 初期剧烈波动 |

> ✅ **Quantile Routing 不仅性能更好，而且负载更均衡**

#### 🔍 **5.2 专家容量控制对部署的影响**

| 设置 | GSM8K（RL后） | BBH | GPQA | MoE Prefill Speedup |
|------|---------------|-----|------|---------------------|
| Dropless | **90.22** | 65.13 | 31.31 | 基线 |
| CF=1.25（容量约束） | 88.93 | **66.02** | **34.34** | **平均 1.53× 加速** |

> ⚠️ 容量约束带来轻微性能下降（尤其 GSM8K），但：
> - 在多个任务上仍有竞争力
> - **MoE 模块 prefill 延迟降低 53%**
> - 提升部署规律性，利于边缘设备运行

---

## **4. 关键结论和发现**

### **主要发现**

1. **紧凑激活预算下可实现高性能**
   - Turing-20B-A2B 以 **~2B 激活参数/token** 实现了超越 Qwen3-8B、接近 Qwen3.5-9B 的综合能力，验证了 **MoE + 高效架构** 的有效性。

2. **动态路由优于静态 top-k**
   - **Quantile Routing** 可自动平衡专家负载并控制计算预算，同时允许模型根据输入难度动态分配资源。

3. **渐进式长上下文训练优于纯外推**
   - 经过 4K→32K→128K 的逐步训练，模型在长上下文下的表现明显优于仅靠 YaRN 扩展的模型。

4. **容量约束显著提升部署效率**
   - 引入容量因子 γ=1.25 后，MoE 模块 prefill 延迟平均降低 **1.53×**，且未造成全面能力退化。

5. **硬件-模型协同设计至关重要**
   - Lightning Attention + DyT + 简单 MoE 结构，使得模型易于部署到边缘设备（如 Turing 芯片）。

---

### **局限性**

1. **目前仅为 Base Model**
   - 尚未发布 SFT 或 RLHF 版本，对话、工具调用等下游能力待验证。

2. **合成数据依赖较强**
   - 后期训练大量使用内部合成数据，可能影响泛化性或引入偏见。

3. **缺乏多模态能力**
   - 当前为纯语言模型；虽然计划结合 **TuringViT** 构建多模态版本，但尚未实现。

4. **MoE 路由开销未完全消除**
   - 尽管做了优化，但 MoE 的动态 dispatch 仍可能带来不确定性，尤其在分布式场景。

---

### **未来工作方向**

1. **Post-training 探索**
   - 开展 Supervised Fine-Tuning 和 Reinforcement Learning（基于 GSM8K、Agent 任务等）。

2. **多模态扩展**
   - 与 **TuringViT** 结合，构建支持视觉、状态、动作输入的 **Vision-Language-Action (VLA)** 模型，应用于自动驾驶与机器人。

3. **Agent 工作负载评估**
   - 在真实物理 AI 场景中测试模型表现，如：
     - 多轮交互
     - 长文档理解
     - 代码生成与软件工程任务
     - 具身决策代理

4. **更大规模 MoE 探索**
   - 延续当前范式，探索百亿级参数、万专家级别的 MoE 架构。

5. **端到端系统集成**
   - 将模型深度集成进 **XPeng Iron 机器人**、**智能座舱**、**一体化驾泊系统** 等产品线。

---

> 📌 **总结一句话**：  
> **Turing-20B-A2B 成功探索了一条面向 Physical AI 的高效 MoE 扩展路径——在极小激活参数下，通过 Quantile Routing + Hybrid Attention + 渐进训练，实现了强大能力、长上下文鲁棒性与低延迟推理的“不可能三角”平衡。**

</details>

---

### 5. [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/abs/2608.30320)

**Authors**: Zihan Qiu, Zekun Wang, Xiao Li, Yanpeng Li, Yang Xu, Yixuan Wang, Huaqing Zhang, Rui Men, Bochao Mao, Chengruidong Zhang, Fan Zhou, Hao Luo, Haofeng Huang, Haoran Lian, Haoyan Huang, Hongqing Chen, Jianwei Zhang, Jing Xu, Junjie Wang, Langshi Chen, Liangyu Wang, Linlang Jiang, Man Yuan, Minmin Sun, Peng Jin, Siqi Zhang, Siyu Wang, Xingzhang Ren, Yakai Wang, Yi Zhang, Yiming Dong, Yizhong Cao, Yubo Ma, Yunfei Mao, Bo Zheng, Dayiheng Liu  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.30320v1  

#### Abstract
We describe the architecture and ablations of Qwen3.8-Flash-Next, a sparse mixture-of-experts model with 125B parameters, 6B activated per token, and additional 51B parameters of n-gram embedding tables held off the accelerator. On fourteen pre-training benchmarks the model leads the 397B-A17B prede...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心总结：On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在设计一个高效且稳定的大型语言模型架构——**Qwen3.8-Flash-Next**，在显著降低训练和推理成本的同时，保持甚至超越前代旗舰模型（如 397B-A17B）的性能。具体解决以下瓶颈：
- **计算效率低**：传统 full attention 在长序列下呈 $O(n^2)$ 复杂度，难以扩展。
- **训练不稳定**：大规模训练中易出现 loss spike 和梯度爆炸。
- **参数利用率低**：MoE 模型激活参数多，但有效容量未充分挖掘。

### 提出的新方法与创新思路
作者提出了一套联合优化架构、优化器与训练策略的整体方案，四大核心组件如下：

#### （1）**Token Mixing: GDN + QSA 混合注意力机制**
- **Gated DeltaNet (GDN)** 层用于大部分层，以线性复杂度压缩上下文状态，实现高效的局部建模。
- 每四层插入一层 **Qwen Sparse Attention (QSA)**，替代 full attention，保留全局检索能力。
- **QSA 创新点**：
  - 使用 micro-block 粒度进行索引，通过压缩轻量级 indexer 将 indexing 成本从 $O(n^2)$ 降至 $O(n^2/r)$。
  - 支持 continued pretraining 阶段无缝替换 full attention 层，提升长文本推理效率。

#### （2）**Gated Residual (GR) 结构**
- 将 residual stream 扩展为四个并行分支，并引入 elementwise gate 控制读写操作。
- 设计融合了 **GatedNorm** 思想，在 RMSNorm 后加入 sigmoid 自门控，提供动态 rescaling，增强表达力与稳定性。
- 仅对 read 路径增加数据依赖性（elementwise gate），write 使用 per-branch scalar，避免 Hres 混合算子带来的额外开销与不稳定性。

#### （3）**N-gram Embedding Layer**
- 在 Layer 2 添加一个独立的 n-gram embedding 表（共 51B 参数），存储于主机内存（host memory），通过预取机制加载。
- 实现“off-accelerator”容量扩展：几乎无额外 per-token FLOPs 开销，显著提升总参数量。

#### （4）**Muon Optimizer + 新超参缩放律**
- 主干权重采用 **Muon** 优化器（基于 Newton-Schulz 正交化），提升大矩阵更新稳定性。
- 输入嵌入、MoE router、GR 投影等仍用 AdamW。
- 提出新的 **hyperparameter scaling law**，预测更优的学习率与 batch size。
- 发现无需 **batch-size warmup**，直接使用目标 batch size 更高效稳定。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 激活参数仅为前代 1/3，训练 token 数量为 1/3，训练 FLOPs 约为 1/9 |
| **性能** | 在 14 项 pre-training benchmark 中，8 项优于前代 397B 模型，其余差距不超过 2.6 分 |
| **稳定性** | 应力测试下（4×最优学习率）无 loss spike 或梯度裁剪触发，远超前代结构 |
| **推理速度** | 在 1M 上下文长度下，prefill 快 7.6×，decode 快 4.9×（kernel level） |

---

## 2. 核心实验方法和设置

### 数据集
- **Pre-training Benchmarks (14项)**：
  - **General Knowledge**: MMLU, MMLU-Pro, SuperGPQA, MMLU-Redux
  - **Reasoning**: BBH
  - **Math & STEM**: MATH, GSM8K, GPQA
  - **Coding**: EvalPlus, MultiPL-E, SWEBench-Pretrain
  - **Multilingual**: MMMLU, MGSM, INCLUDE
- **Long-context Retrieval**:
  - **RULER**：评估不同长度下的上下文理解能力（4K ~ 1M）
  - **8-needle MRCR**：多针检索任务，测试极端长上下文中的信息定位能力

### 实验设置
- **模型规模**：
  - Qwen3.8-Flash-Next: 125B total params, 6B activated per token, +51B n-gram embedding
  - 对比基线：Qwen3.7-Plus-Base (397B), Qwen3.8-27B-Base (27B)
- **训练阶段**：
  - Continued Pretraining (CPT) 使用 256K 上下文长度
  - QSA 分两阶段训练：dense distillation → sparse training
- **硬件与实现**：
  - 使用 FlashQLA 加速 GDN kernel
  - QSA 实现 fused kernel 减少中间结果存储
  - Muon 优化器使用 Canzona 框架处理分布式正交化

### 评估指标
| 类别 | 指标 |
|------|------|
| **模型能力** | 各 benchmark 的准确率（pass@k, acc 等） |
| **训练效率** | 训练 loss、收敛速度、optimizer steps |
| **推理效率** | Prefill / decode 延迟、kernel speedup（vs dense attention） |
| **训练稳定性** | loss spike 次数、pre-clip gradient norm p99.9、是否触发 clipping |

### 基线方法对比
- **Architecture Baselines**:
  - Full attention Transformer
  - Sliding Window Attention (SWA) hybrid
  - GDN-only / GDN+full attn 对比
- **Residual Path Baselines**:
  - Pre-norm residual
  - Hyper-Connections (HC), mHC
  - Attention Residual (AttnRes)
- **Optimizer Baselines**:
  - AdamW（原版 Qwen3.5 架构）
  - Muon vs AdamW 对比
- **Hyperparameter Strategy**:
  - 是否使用 batch-size warmup
  - 不同 learning rate 设置（×1, ×√2, ÷√2）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table 11）

| Benchmark | Qwen3.8-Flash-Next-Base | Qwen3.7-Plus-Base (397B) | 结果对比 |
|-----------|--------------------------|----------------------------|---------|
| MMLU | 90.36 | 90.43 | ≈ |
| MMLU-Pro | **73.23** | 70.90 | ↑ +2.33 |
| SuperGPQA | **51.36** | 48.42 | ↑ +2.94 |
| BBH | **90.87** | 89.41 | ↑ +1.46 |
| GPQA | 51.42 | **51.52** | ≈ |
| GSM8K | **93.29** | 92.95 | ↑ +0.34 |
| MATH | 72.78 | **74.38** | ↓ -1.6 |
| EvalPlus | **78.76** | 78.06 | ↑ +0.7 |
| MultiPL-E | 79.09 | **81.68** | ↓ -2.59 |
| MMMLU | **84.86** | 84.53 | ↑ +0.33 |
| MGSM | **89.33** | 85.42 | ↑ +3.91 |

> ✅ **结论**：在仅使用约 **1/3 激活参数** 和 **1/9 训练 FLOPs** 的情况下，该模型在 **8/14 项任务上超过更大模型**，整体表现极具竞争力。

---

### 与其他方法对比结果

#### （1）Token Mixing 架构对比（Table 1）
| 方法 | Avg Score |
|------|---------|
| Full attention | 50.91 |
| SWA hybrid | 52.49 |
| **GDN hybrid (ours)** | **54.66** |

> GDN + full attention 混合结构在多数任务上优于 SWA 和纯 full attention。

#### （2）QSA vs Full Attention（Table 2）
| 方法 | Avg Score |
|------|--------|
| Full Attn | 75.9 |
| **w/QSA** | **76.8** |

> QSA 不仅没有性能损失，反而平均提升 0.9 分，尤其在长上下文检索任务中表现更好。

#### （3）Long-context Retrieval（Table 3）
| 方法 | RULER (>512K) | MRCR (1M) |
|------|---------------|----------|
| Full Attn | 90.08 | 20.71 |
| **w/QSA** | **93.00** | **26.44** |

> QSA 在极长上下文中显著优于 dense attention，验证其稀疏索引的有效性。

#### （4）推理效率（Figure 6）
| 场景 | Speedup |
|------|--------|
| Prefill @ 1M | **7.6×** |
| Decode @ 1M | **4.9×** |

> 核心优势之一：极大加速长文本生成。

---

### 消融实验结果

#### （1）Residual 结构消融（Table 5）
| 方法 | Loss ↓ | Avg ↑ |
|------|-------|-----|
| Pre-norm | 1.617 | 50.91 |
| mHC (static) | 1.596 | 52.49 |
| mHC (dynamic) | 1.594 | 54.47 |
| **GR (ours)** | **1.590** | **54.66** |

> GR 在更低 loss 和更高 benchmark 得分之间取得最佳平衡。

#### （2）N-gram Vocabulary Scaling（Table 9）
- **Loss 随 vocab 增大单调下降**
- **下游任务得分先升后饱和**，部分任务（如中文 C-Eval）持续受益
- 发现：**loss 与 downstream accuracy 并不总一致**，需综合评估

#### （3）Batch Size Warmup 消融（Figure 8b）
- 使用 warmup 反而导致最终 loss 更高（差 2.5–3.5×10⁻⁴）
- 多消耗 **18.8% optimizer steps**
- **结论：无需 warmup**

#### （4）Learning Rate Stress Test（Figure 10）
| 条件 | AdamW | Muon | Muon+GR |
|------|-------|------|--------|
| @2× LR | 频繁 spike | 稳定 | 稳定 |
| @4× LR | 完全失控 | 轻微波动 | **完全稳定** |

> GR 是提升训练鲁棒性的关键因素。

---

## 4. 关键结论和发现

### 主要发现
1. **Loss 与 downstream accuracy 并非强相关**  
   - 扩大 n-gram vocab 可持续降低 loss，但 accuracy 会饱和
   - 需同时监控多个指标，防止“过拟合 loss”

2. **Gated Residual (GR) 显著提升训练稳定性**  
   - 引入门控机制后，即使在 4× 学习率下也无 loss spike
   - 替代了显式的 qk-clip 或 SwiGLU-clip 等稳定技术

3. **新架构 + Muon 使超参选择更激进且安全**  
   - 最优 batch size 和 learning rate 显著上升
   - 无需 batch-size warmup，节省训练时间

4. **稀疏注意力可在不牺牲性能前提下大幅提升效率**  
   - QSA 在长文本场景下优于 dense attention
   - indexing 成本随序列增长缓慢，具备良好可扩展性

5. **宽度优于深度表达性**  
   - widening residual stream 本身就能带来显著增益
   - GR 将表达性集中在 read 路径，而非复杂的 branch mixing

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **评估吞吐仍是瓶颈** | 当前 post-training 评估耗时高，限制搜索空间 |
| **n-gram embedding 泛化有限** | 英文代码类任务收益较小，可能因模式差异 |
| **FP8 存储依赖特定硬件支持** | 推理端需兼容低精度格式 |
| **multi-token prediction 复用索引尚未完全探索** | MTP 模块仍有优化潜力 |

---

### 未来工作方向
1. **构建更高效的 mid-scale 评估代理**  
   - 开发能在小模型上可靠预测 post-training 排序的方法，加速架构迭代。

2. **进一步优化 n-gram embedding 的参数分配策略**  
   - 探索非均匀分配、频率感知分区等方式提高效率。

3. **将 GR 与 MoE 路由机制更深耦合**  
   - 利用 residual branch 差异化控制 expert routing 动态。

4. **探索更多 off-accelerator 存储技术**  
   - 如 KV cache 分层存储、embedding streaming 等，突破显存限制。

5. **自动化 hyperparameter tuning pipeline**  
   - 结合 scaling law 与 stress test，实现端到端鲁棒训练配置生成。

---

> 🔚 **总结一句话**：  
> Qwen3.8-Flash-Next 通过 **GDN+QSA 混合注意力、Gated Residual、n-gram embedding 与 Muon 优化器** 的协同设计，在 **1/9 的训练成本** 下实现了与前代超大模型相当甚至更优的性能，同时具备更强的 **训练稳定性与推理效率**，标志着高效大模型设计进入“架构-优化-训练”一体化时代。

</details>

---

### 6. [ERR+: Sequential Entropy Resolution for Efficient and Decisive LLM Reasoning](https://arxiv.org/abs/2608.28771)

**Authors**: Xin Jiang, Minhao Wang, Wen Wu, Zhentao Xie, Shangheng Du, Jinxin Shi, Jiabao Zhao  
**Category**: cs.LG  
**Published**: 2026-09-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.28771v1  

#### Abstract
Large reasoning models achieve strong performance on complex tasks by generating extended chain-of-thought (CoT) traces via reinforcement learning with verifiable rewards (RLVR). While current RLVR methods have achieved strong results with correctness-based reward signals, they provide limited guida...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《ERR+: Sequential Entropy Resolution for Efficient and Decisive LLM Reasoning》总结

---

## 1. 主要贡献和创新点

### ✅ 解决的问题
当前基于 **Reinforcement Learning with Verifiable Rewards (RLVR)** 的大型推理模型（如 DeepSeek-R1、Qwen3）虽然在复杂任务上表现优异，但存在以下关键缺陷：
- **仅依赖最终答案正确性作为奖励信号**（binary correctness reward），导致模型无法区分“正确但冗长”与“简洁且逻辑清晰”的推理路径。
- 现有基于熵控制的方法（如 PEAR）通过**惩罚高熵状态来压缩长度**，但会抑制探索性生成，损害推理质量，造成 **accuracy-conciseness trade-off**。

### 🆕 提出的新方法：ERR+
提出 **ERR+** ——一种两阶段的 RLVR 框架，其核心思想是：  
> **不惩罚高熵，而是奖励“不确定性被解决”的过程（即 token-level entropy drops）**。

#### 两个核心组件：
1. **Phase 1: Entropy Relief Reward (ERR)**
   - 在思考阶段（thinking phase），对每个显著的 **token-level entropy drop** 进行积分，并以 `log(Tk+1)` 归一化。
   - 奖励公式：  
     $$
     \text{ERR}(y) = \frac{\sum_{t=2}^{T_k} \max(H_{t-1} - H_t - \epsilon, 0)}{\log(T_k + 1)}
     $$
   - 优点：鼓励模型做出**决定性的推理步骤**（decisive commitments），同时保留必要的探索空间。

2. **Phase 2: Robust Relative Efficiency Reward (RRER)**
   - 引入一个**难度感知的相对长度奖励机制**：
     - 对每组共生成响应（co-generated group）计算长度的 z-score。
     - 使用 `tanh` 映射进行饱和处理，避免极端值影响。
     - 只有当响应正确时才给予长度奖励，防止错误短答获利。
   - 实现了**自适应压缩**：简单问题自然短，难题允许更长。

### 🔍 相比现有方法的优势
| 方法 | 是否提升 accuracy | 是否缩短 length | 是否避免 trade-off |
|------|-------------------|------------------|--------------------|
| GRPO / DAPO | ✅ 是 | ❌ 否（甚至变长） | ❌ 存在 trade-off |
| PEAR / LASER | ❌ 否（常降准） | ✅ 是 | ❌ 牺牲准确率换简短 |
| **ERR+** | ✅✅ 显著提升 | ✅✅ 显著缩短 | ✅✅ **打破 trade-off** |

此外，作者从理论上证明了 ERR 与 RRER 存在早期梯度冲突（gradient conflict），因此必须采用**顺序训练**而非联合优化（joint training），这是设计上的根本创新。

---

## 2. 核心实验方法和设置

### 📚 数据集
在五个具有代表性的推理基准上进行全面评估：
- **GSM8K**: 小学数学应用题（2–8步算术）
- **MATH-500**: 高中竞赛级数学题
- **AIME24**: 2024年美国数学邀请赛题目（高难度）
- **AMC23**: 美国数学竞赛题
- **MMLU-STEM**: STEM领域的多任务理解子集（用于跨域泛化测试）
- **GPQA Diamond**（附录）: 博士级别科学问答（生物/物理/化学）

> 所有模型均只在 GSM8K 上训练，其余为 zero-shot 测试，验证泛化能力。

### ⚙️ 实验设置
- **模型骨干**（backbone）：
  - DeepSeek-R1-Distill-Qwen-1.5B / 7B
  - Qwen3-4B / 8B
- **训练框架**：开源 verl 框架
- **训练数据**：GSM8K 的 7,473 条训练样本
- **超参数**：
  - Batch size: 128
  - LR: 1e-6
  - Temperature: 0.6, top_p: 0.95
  - Rollouts per prompt: 8
- **默认超参**：ε=0.01, γ=0.5, λ=0.3, α=0.3, R_b=1.0, R_f=0.1, R_max=1.5

### 📊 评估指标
- **Pass@1 Accuracy (Acc)**：最终答案是否正确
- **Average Response Length (Tok)**：平均输出 token 数量
- **消融实验**：分析各模块必要性
- **稳定性分析**：四次独立运行的标准差

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| 零样本基线 | Base (zero-shot) |
| 标准 RLVR | GRPO, DAPO |
| 长度控制方法 | PEAR, DEER, Dynasor, LASER-L2048 |
| 消融变体 | Phase-2 only, Joint training (R1+R2) |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（以 DeepSeek-R1-Distill-Qwen-1.5B 为例）

| Method | Avg Acc (%) | Avg Tok | Δ Acc | Δ Tok |
|--------|-------------|---------|-------|--------|
| Base | 63.0 | 6818 | — | — |
| GRPO | 64.2 | 7240 | +1.2 | +422 |
| PEAR | 63.6 | 6621 | +0.6 | -197 |
| **ERR+** | **68.4** | **5450** | **+5.4** | **-1790** |

> ✅ ERR+ 在所有基准上均取得最高准确率，同时将响应长度减少约 **20–30%**

#### 更细粒度结果（Table 1 & 2）：
- 在 **AIME24** 上，ERR+ 准确率从 30.0%（GRPO）提升至 **33.3%**
- 在 **AMC23** 上，从 70.0% 提升至 **77.5%**
- 在 **MMLU-STEM** 上，从 51.6% 跃升至 **58.0%**（+6.4pp）
- 平均长度从 7240 降至 5450（↓24.7%）

#### 泛化到其他 backbone（Table 2）：
| Backbone | GRPO Acc / Tok | ERR+ Acc / Tok | Gain |
|----------|----------------|----------------|------|
| Qwen3-4B | 81.8 / 5687 | **84.2 / 4752** | +2.4 / ↓935 |
| Qwen3-8B | 81.7 / 6269 | **84.0 / 5451** | +2.3 / ↓818 |
| DS-R1-7B | 79.4 / 5509 | **82.5 / 4495** | +3.1 / ↓1014 |

> 表明 ERR+ 具有良好的**可迁移性和扩展性**

### 🔬 消融实验结果（Table 3）

| Variant | Avg Acc | Avg Tok | 结论 |
|--------|---------|---------|------|
| **Full ERR+** | **68.4** | **5450** | 最优组合 |
| w/o Phase-1 (Phase-2 only) | 62.9 | 5465 | 准确率↓5.5%，说明无稳定结构前压缩有害 |
| w/o Phase-2 (Phase-1 only) | 66.8 | 6625 | 长度↑21.5%，说明需第二阶段压缩 |
| Joint training (R1+R2) | 62.6 | 6459 | 性能最差，验证了梯度冲突理论 |

> ✅ 两阶段顺序训练至关重要，任何简化都会显著劣化性能。

### 🧪 抗奖励欺骗分析（Anti-Reward-Hacking）
- 若模型通过制造“虚假熵下降循环”来刷分，则应出现更多熵上升/方向切换。
- 实际观察（Table 6）：
  - `Up/Drop` 比例始终 ≈1.0（未膨胀）
  - `Sign change` 数量持续下降（从 1013 → 610）
- 表明熵下降是真实决策事件，非人为操纵。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **正确的推理轨迹表现出更频繁、更深的 token-level entropy drops**  
   → 可作为衡量推理质量的内部信号。

2. **奖励“不确定性解除”（entropy relief）优于惩罚“高熵”本身**  
   → 既保持探索自由，又促进果断决策。

3. **长度压缩应在高质量推理结构建立之后进行**  
   → Phase 1 构建“里程碑式”推理节点，Phase 2 安全移除中间冗余。

4. **ERR 与 RRER 存在早期梯度冲突**（Theorem 1）  
   → 必须采用顺序训练，联合优化会导致性能崩溃。

5. **ERR+ 成功打破了 accuracy 与 conciseness 的权衡困境**  
   → 实现“越准越短”的理想目标。

### ⚠️ 局限性
- 当前目标聚焦于 **accuracy 和 response length**，尚未考虑人类偏好因素，如：
  - Step-by-step explainability
  - Pedagogical clarity（教学清晰度）
  - Stylistic consistency
- 依赖于 `<think>...</think>` 标记明确划分思考阶段，在开放格式中可能受限。
- 超参数敏感性虽不高，但在极难任务上仍需调优。

### 🔮 未来工作方向
- 引入 **multi-objective RL** 或 **preference learning**，融合 human-centric qualities。
- 探索无需显式标记的自动 phase detection。
- 将 entropy relief 思想推广至非文本模态或多智能体协作推理。
- 结合 **process-based supervision**（如 step-level verification）进一步增强推理可控性。

---

## ✅ 总结一句话
> **ERR+ 通过“先奖决断、再压长度”的两阶段策略，利用 token-level entropy drops 作为内在质量信号，在理论上解释并实践中解决了 accuracy 与 conciseness 的根本矛盾，实现了高效而果断的大模型推理。**

</details>

---

### 7. [Deploying DeepSeek 175B Locally on a Single Consumer-Grade RTX 4060 Laptop with 32GB RAM for 200k-Scale Protein-Ligand Virtual Screening](https://arxiv.org/abs/2608.30877)

**Authors**: Rui Xiao, Yili Xu  
**Category**: cs.LG  
**Published**: 2026-09-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2608.30877v1  

#### Abstract
Recent advances in large language models (LLMs) have demonstrated exceptional performance in protein-ligand interaction prediction, but state-of-the-art pipelines for large-scale virtual screening almost exclusively rely on high-end GPU clusters with hundreds of gigabytes of memory, creating prohibi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Deploying DeepSeek 175B Locally on a Single Consumer-Grade RTX 4060 Laptop with 32GB RAM for 200k-Scale Protein-Ligand Virtual Screening

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **LLM** 的 **protein-ligand virtual screening** 几乎全部依赖于高成本的 **A100/H100 GPU 集群**，需要数百 GB 内存和数据中心基础设施，严重限制了小型学术团队、高校实验室和独立研究者的访问能力。此外，现有低资源 LLM 优化方法多面向通用 NLP 任务，难以满足生物医药领域对化学精度（chemical accuracy）的严格要求。

本论文旨在解决以下核心痛点：
- 如何在消费级硬件上部署 **trillion-parameter 级别的 LLM（如 DeepSeek 175B）**
- 如何完成工业规模（200k-scale）的虚拟筛选任务
- 在不牺牲预测精度的前提下实现极低硬件门槛的本地化运行

---

### 🚀 提出的新方法与新思路

作者提出了一套**端到端的低资源优化框架**，支持在单台配备 **RTX 4060（8GB VRAM）+ 32GB RAM** 的消费级笔记本上完整运行 **DeepSeek 175B** 模型，并完成大规模虚拟筛选任务。其核心创新包括：

1. **多层级低资源优化框架（Multi-level low-resource optimization framework）**
   - 结合 **structured pruning**、**mixed-precision quantization（4-bit）** 和 **adaptive memory management**
   - 针对 DeepSeek 175B 的 **32K sliding window attention 架构**进行结构适配
   - 将模型总 VRAM 占用压缩至 **<8GB**，完全适配 RTX 4060 显存限制

2. **领域自适应知识迁移（Domain-specific knowledge adaptation）**
   - 使用一个高质量公共数据集（含 2.3M 蛋白质-配体结合亲和力标注）对压缩后模型进行微调
   - 控制精度损失 < 0.2 kcal/mol，确保满足药物发现中的化学精度标准

3. **全本地化、断点可恢复的 end-to-end pipeline**
   - 整个流程（预处理 → 编码 → 预测 → 排序导出）均在本地执行，无需云资源
   - 支持自动断点续跑，保障 72 小时连续稳定推理

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 硬件需求 | 多卡 A100/H100 集群（>640GB 显存） | 单台消费级笔记本（RTX 4060 + 32GB RAM） |
| 成本门槛 | 极高（百万级投入） | 极低（<$2000 设备） |
| 数据隐私 | 存在云端传输风险 | 完全本地处理，无数据外泄风险 |
| 吞吐效率 | 受限于通用调度策略 | 深度优化批处理流程，吞吐提升 100x |
| 预测精度 | SOTA 但依赖闭源模型 | 开源模型 + 化学精度达标（<1.0 kcal/mol） |

> ✅ **首次验证了 trillion-parameter LLM 在边缘设备上执行工业级生物医学计算任务的工程可行性**

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **主数据集**：包含 **20 个不同蛋白靶点**（涵盖 kinase、GPCR、ion channel 等家族）
- **配体数量**：共 **200,000 个化合物（ligands）**
- **训练/适配数据集**：来自公开的 **2.3M annotated protein-ligand binding affinity dataset**（Zhou et al., 2024），用于 domain adaptation 微调

所有数据均为公开可用，未引入私有或封闭数据。

---

### ⚙️ 实验设置

| 项目 | 设置详情 |
|------|----------|
| **测试平台（本工作）** | 消费级笔记本：<br>• GPU: RTX 4060 (8GB VRAM)<br>• CPU/RAM: 32GB system RAM<br>• 模型版本: 4-bit quantized DeepSeek 175B |
| **基线平台** | 工业级 8-card NVIDIA A100 80GB GPU cluster（总计 640GB 显存） |
| **软件环境** | 开源 LLM inference framework（相同于基线） |
| **任务配置** | 相同输入数据、batch size、序列长度等参数保持一致 |

---

### 🎯 评估指标

1. **Throughput Performance**  
   - 总完成任务量 / 时间（72小时内完成的 protein-ligand 对数）

2. **Prediction Accuracy**  
   - 平均 binding affinity 预测误差（单位：kcal/mol）
   - 是否满足 **1.0 kcal/mol 化学精度阈值**

3. **Runtime Bottleneck Analysis**  
   - 运行时开销分解（memory swap vs. GPU compute）

4. **Error Source Decomposition**  
   - 模型优化导致的误差 vs. 数据噪声与泛化误差

---

### ↔️ 基线方法对比

- **Baseline**: 工业标准 8×A100 集群运行相同的 DeepSeek 175B 模型和任务流程
- 对比维度：吞吐量、响应延迟、能耗、部署复杂度、成本

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

| 指标 | 本文方法（RTX 4060 笔记本） | 基线（8×A100 集群） | 提升倍数 |
|------|-----------------------------|---------------------|--------|
| **72小时完成任务量** | 200,000 pairs | 2,000 pairs | **×100** |
| **平均预测误差** | **0.88 kcal/mol** | ~0.85 kcal/mol（略优） | 持平（均 <1.0 kcal/mol） |
| **是否满足化学精度** | ✅ 是（全部 20 个 target 均 ≤0.95） | ✅ 是 | —— |
| **显存占用峰值** | <8GB VRAM | >500GB aggregate VRAM | ↓ 98%+ |

> 💡 **尽管硬件资源相差三个数量级，本文方法实现了 100 倍的吞吐优势**

---

### 🔍 与基线方法的对比结果

- 在相同任务设置下，**8-card A100 集群仅完成了 1% 的总任务量（2k/200k）**
- 本文方法得益于：
  - 更高效的 **continuous batching pipeline**
  - 针对长序列（32K context）优化的任务调度
  - 减少跨节点通信与数据同步开销（分布式系统的固有瓶颈）

> ⚠️ 注意：该“反常”性能反转并非因 A100 性能不足，而是由于**集群系统存在严重的 I/O 和调度延迟**，而本地流程实现了极致轻量化与流水线并行优化

---

### 🔤 消融实验与误差分解（Ablation & Quantitative Analysis）

#### （1）运行时开销分解（Runtime Overhead Breakdown）
| 开销来源 | 占比 |
|--------|-----|
| 异构内存页交换调度（Heterogeneous memory page swap） | **72%** |
| GPU 核心计算（GPU core computation） | 21% |
| 其他（I/O、调度等） | 7% |

> ❗ 表明当前边缘侧 LLM 生物计算的**主要瓶颈是内存管理效率**，而非算力本身

#### （2）预测误差来源分解（Error Source Decomposition）
| 误差来源 | 贡献比例 |
|--------|--------|
| 模型压缩与优化引入的误差 | **<10%** |
| 数据集噪声与模型泛化误差 | **>90%** |

> ✅ 说明模型压缩带来的精度损失极小，**主要误差来源于原始数据质量和任务本质难度**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Trillion-parameter LLM 可在消费级硬件上运行工业级生物计算任务**
   - 成功将 **DeepSeek 175B** 部署于 **RTX 4060 + 32GB RAM 笔记本**
   - 完整执行 **200k-scale virtual screening**，覆盖 20 个独立蛋白靶点

2. **性能超越高端 GPU 集群 100 倍**
   - 吞吐量显著领先，源于更优的本地化批处理设计和零通信开销

3. **预测精度达到工业可用标准**
   - 平均误差 **0.88 kcal/mol**，低于 **1.0 kcal/mol 化学精度阈值**
   - 所有目标均保持高一致性表现（0.5~0.95 kcal/mol）

4. **系统瓶颈明确指向内存调度**
   - 当前 **72% 的时间消耗在异构内存交换**，为后续优化提供清晰路径

---

### ⚠️ 方法的局限性

1. **不支持超长蛋白质序列（>2000 residues）**
   - 当前框架受限于上下文窗口与内存调度机制，无法有效处理极端长度序列

2. **高度依赖预优化的 batching 与 pipeline 设计**
   - 性能优势部分来自于特定任务定制化调度，通用性有待验证

3. **尚未扩展到多模态或多任务联合推理场景**
   - 当前仅聚焦 binding affinity prediction，未涉及 ADMET 或 de novo design

---

### 🔮 未来工作方向

1. **优化内存调度机制**
   - 引入 **hardware-aware prefetching** 和 **operator fusion** 技术
   - 目标：将 200k 任务从 **72h 缩短至 <24h**

2. **支持更长序列输入**
   - 改进 sliding window attention 的缓存策略，支持 >2000aa 蛋白建模

3. **构建开源低资源 AI 药物发现工具链**
   - 推动更多小型团队使用 open-source LLM 进行 early-stage drug discovery

4. **探索更多 edge biomedical computing 应用**
   - 如基因组分析、单细胞注释、抗体设计等场景的本地化部署

---

## ✅ 总结一句话

> 本论文首次证明：通过合理的低资源优化与流程重构，**trillion-parameter LLM 驱动的工业级 protein-ligand virtual screening 完全可以在一台消费级笔记本上高效、精准地完成**，为 AI 药物发现开辟了低成本、高可及性的全新范式。

</details>

---

### 8. [Pro-Router: Token-Aware Progressive Model Routing with Adaptive Edge-Cloud Collaboration for Efficient Multimodal LLM Inference](https://arxiv.org/abs/2608.28726)

**Authors**: Xinyuan Gui, Shaowen Wang, Sheng Sun, Zijian Wang, Zishu Yu, Zheming Yang  
**Category**: cs.AI  
**Published**: 2026-09-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.28726v1  

#### Abstract
The remarkable performance of multimodal large language models (MLLMs) comes at the cost of substantial computational overhead, posing significant challenges to real-time deployment and cost effectiveness. Existing model routing approaches either decide from coarse request-level features alone or sp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Pro-Router: Token-Aware Progressive Model Routing with Adaptive Edge-Cloud Collaboration for Efficient Multimodal LLM Inference*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
多模态大语言模型（**MLLMs**）虽然性能强大，但其推理成本高昂、延迟高，难以在实时服务中高效部署。现有 **model routing** 方法存在以下缺陷：
- **Request-only routers**：仅基于输入请求（prompt）判断难度，在生成前决策，速度快但准确率低，无法感知小模型实际生成能力。
- **Response-based routers**：让小模型先生成答案再评估是否“可信”，虽更准确，但需额外调用一个语言模型（如 DistilBERT）或多次自验证（self-verification），带来显著计算开销，抵消了路由带来的收益。
- **Edge-Cloud 协作效率低**：现有协作框架依赖频繁通信，受网络延迟影响大，容易导致边缘或云端设备空闲，利用率低下。

### 🚀 提出的新方法：**Pro-Router**
一种**token-aware progressive model routing** 框架，结合**自适应边缘-云协同服务管道**，实现高效、准确、鲁棒的多模态 LLM 推理。

#### 主要创新点：
1. **两阶段渐进式路由机制（Progressive Routing）**
   - **第一阶段：Prompt 预评分器（Prompt Pre-scorer）**
     - 在 token 生成前，使用轻量级模块对请求进行快速预筛。
     - 特征：TF-IDF 向量、图像数量、提示长度 —— 均为低成本特征，运行于 CPU。
     - 输出难度分数，引导简单请求优先流向边缘小模型（MSLM）。
   - **第二阶段：Token-aware 验证器（Token-aware Verifier）**
     - 在 MSLM 解码过程中，直接读取每个 token 的采样概率分布（sampling distribution），无需额外模型。
     - 提取四个每 token 不确定性特征：
       - `logp(y_t)`：生成 token 的对数概率
       - `max_p`：最大预测概率
       - `-H(top-K entropy)`：top-K 熵的负值（衡量置信度）
       - `position_fraction`：当前位置占比
     - 将这些特征序列输入一个小型 Transformer 编码器，输出最终 shipping confidence。
     - 决策逻辑：若 confidence ≥ 阈值 T，则接受边缘输出；否则升级至云端大模型（MLLM）重新生成。

2. **自适应边缘-云协作服务管道（Adaptive Edge-Cloud Collaboration Pipeline）**
   - 引入全局调度器（Global Scheduler），基于各设备实测吞吐量动态分配负载。
   - 维护两个缓冲区：
     - **Scored Input Buffer**：按预评分排序，易题从顶部分发给边缘。
     - **Escalated Buffer**：被验证器拒绝的请求进入此缓冲区，优先发送至云端。
   - 调度策略：
     - 边缘设备接收最简单的请求（Top of scored buffer）。
     - 云端设备先处理 escalation 请求，再处理最难请求（Bottom of scored buffer）。
   - 利用指数加权移动平均（EMA）跟踪设备吞吐量，自动调整每次 dispatch 数量，避免因网络延迟造成阻塞。

### 🔍 相比现有方法的优势
| 方面 | Pro-Router 优势 |
|------|----------------|
| **准确性** | 利用 token 级不确定性信号，比仅看 prompt 或后处理 response 更精准 |
| **效率** | 验证器不引入额外 LM，仅使用解码副产物，几乎零开销 |
| **速度** | 路由信号生成速度 >10× 快于基线方法 |
| **系统吞吐** | 自适应调度使边云设备持续满载，端到端吞吐提升 >75% |
| **鲁棒性** | 对高达 1000ms 的网络延迟不敏感，维持 >90% 性能 |
| **可扩展性** | 支持线性扩展至多台边缘与云端设备 |

---

## 2. 核心实验方法和设置

### 📚 数据集
共使用 **15 个 benchmark**，涵盖多种任务类型：
- **单图任务（9个）**：C18, A-OKVQA, MathVerse, MMStar, HallusionBench, RealWorldQA, OCRBench, MM-Vet, MathVision
- **多图任务（2个）**：MileBench, MuirBench
- **纯文本任务（4个）**：MMLU, GSM8K, CoQA, TriviaQA

> 所有答案由强 LLM judge（如 GPT-4）进行语义等价性评分，不确定判为错误。

### ⚙️ 实验设置
- **模型配置**：
  - 小模型（MSLM）：Qwen2.5-VL-7B, LLaVA-OneVision-7B, Pixtral-12B
  - 大模型（MLLM）：Qwen2.5-VL-72B（云端）
- **硬件环境**：
  - 云端：M × p4d 节点（每节点 8×A100 GPU）
  - 边缘：N × g5.12 节点（每节点 4×A10G GPU）
- **实现平台**：基于 **Ray** 和 **vLLM** 构建服务系统，支持异步 RPC 和 KV-cache 管理。
- **网络模拟**：注入 0 / 200 / 1000 ms 单向延迟以测试鲁棒性。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **AUROC** | 衡量路由信号区分正确/错误响应的能力，越高越好（随机=0.5） |
| **AUARC / AUACC** | 分别基于 correctness label 和 pairwise label 的 shipping accuracy 曲线下面积 |
| **PGR (Performance Gap Recovered)** | 路由恢复的大模型质量差距比例：<br>$$ \text{PGR} = \frac{\text{acc}_{\text{pipeline}} - \text{acc}_{\text{small}}}{\text{acc}_{\text{large}} - \text{acc}_{\text{small}}} $$ |
| **End-to-end Throughput (req/s)** | 整体系统稳态下每秒完成请求数 |
| **Latency (ms)** | 端到端响应延迟（含调度、生成、验证） |

### 🔁 基线方法对比
| 基线 | 类型 | 描述 |
|------|------|------|
| **RouteLLM** | Request-only | 使用 278M 参数的专用路由器，仅分析 prompt |
| **FrugalGPT** | Response-based | 使用 DistilBERT-66M 对小模型输出文本打分 |
| **P(True)** | Self-verification | 小模型自问“我的回答是真的吗？”一次 |
| **AutoMix** | Self-verification | 多次采样 few-shot 自验证器（默认8次）估计可信度 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）路由准确性全面领先
| 方法 | 平均 AUROC (Qwen-7B) | AUARC | AUACC | PGR |
|------|------------------------|-------|-------|-----|
| **Pro-Router (Ours)** | **0.805** | **0.85** | **0.70** | **~0.75** |
| RouteLLM | 0.691 | 0.65 | 0.55 | ~0.50 |
| FrugalGPT | 0.701 | 0.68 | 0.58 | ~0.52 |
| P(True) | 0.683 | 0.66 | 0.56 | ~0.48 |
| AutoMix | 0.677 | 0.64 | 0.54 | ~0.45 |

> ✅ **Pro-Router 在所有指标上均取得最高分**，尤其在 AUROC 上领先第二名超 10 个百分点。

#### （2）路由信号生成速度 >10× 加速
| 方法 | 单请求路由延迟 |
|------|----------------|
| **Pro-Router** | **2–3 ms** |
| RouteLLM | 47–119 ms |
| FrugalGPT | 52–103 ms |
| P(True) | ~2s（1次生成） |
| AutoMix | ~57s（8次生成） |

> ✅ **Pro-Router 的验证器几乎无额外开销**，而其他方法需完整前向传播甚至多次生成。

#### （3）端到端吞吐显著提升
- **相比最强基线（P(True)）**：
  - 吞吐提升 **1.16–1.28×**
- **相比现有路由管道（Ray Serve）**：
  - 吞吐提升 **>75%（达 1.77–1.79×）**

#### （4）鲁棒性与可扩展性
| 条件 | 结果 |
|------|------|
| **网络延迟 1000ms** | 吞吐保持原始的 **90–96%** |
| **增加边缘设备 N** | 吞吐随 N **线性增长**，无饱和现象 |
| **增加云端设备 M** | 同样呈现线性扩展趋势 |
| **最大规模（M=3, N=4）** | 吞吐达单云设备的 **5.8–6.9×** |

---

### 🔍 消融实验（Ablation Study）

| 变体 | AUROC (Qwen) | Throughput (req/s) | Latency (ms) |
|------|---------------|--------------------|--------------|
| **Full Pro-Router** | **0.805** | **124** | **754** |
| 替换为 hidden states 输入 | 0.758 | 95 | 932 |
| 使用 quantile+MLP 替代 Transformer | 0.691 | 118 | 783 |
| 移除批量 dispatch | 0.805 | **28**（↓4.4×） | 1038 |
| 关闭 KV-cache 准入门控 | 0.805 | 112 | **1634**（↑2.2×） |

> ✅ 验证：
> - **采样分布特征优于 hidden states**
> - **Transformer 架构优于 MLP**
> - **批调度与 KV-cache 控制至关重要**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Token-level 不确定性是高质量路由信号的关键来源**  
   直接利用 MSLM 解码过程中的采样分布，可以获得比 prompt 或文本后处理更强的置信度指示。

2. **渐进式决策优于单一时刻决策**  
   “先粗筛 + 后精验” 的两阶段设计兼顾效率与精度，显著提高 ship rate 与 routing accuracy。

3. **系统级优化决定实际性能上限**  
   即便有优秀的路由算法，若缺乏合理的调度机制（如忽略网络延迟、负载不均衡），仍会导致边云资源浪费。

4. **轻量化设计才能真正释放边缘潜力**  
   Pro-Router 的 verifier 仅 ~67k 参数且运行于 CPU，完全不影响 MSLM 解码效率，是实现“零代价验证”的关键。

### ⚠️ 局限性
- **阈值依赖人工校准**：shipping threshold $ T $ 需在验证集上通过 coverage-based rule 校准，尚未完全自动化。
- **Verifier 泛化能力假设较强**：使用统一 verifier 处理 15 个不同 benchmark，可能在极端分布偏移下表现下降。
- **未考虑动态负载波动下的长期稳定性**：实验集中在稳态吞吐，未深入研究突发流量下的行为。

### 🔮 未来工作方向
- 开发**在线自适应阈值调节机制**，根据实时反馈动态调整 $ T $。
- 探索**跨模型通用 verifier**，减少训练成本。
- 扩展至**多跳推理场景**，支持复杂 agent 中的步骤级路由。
- 结合 **speculative decoding** 或 **early exiting** 进一步压缩边缘延迟。

---

> 💬 **代码已开源**：https://github.com/xinyuangui2/pro-router

</details>

---

### 9. [Budget-Aware Compression Pipeline for Single-GPU LLM Inference: Methods, Trade-offs, and Coupling Effects](https://arxiv.org/abs/2608.30076)

**Authors**: Hongyu Yu, Yifei Shen  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.30076v1  

#### Abstract
Single-GPU deployment of 70B-parameter language models on an NVIDIA GPU is constrained by device memory, long-context throughput, and engineering integration cost. We cast single-GPU inference as a budget-aware design problem over these three axes and study how pruning, quantization, and KV-cache co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Budget-Aware Compression Pipeline for Single-GPU LLM Inference: Methods, Trade-offs, and Coupling Effects*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于在**单张消费级GPU（如NVIDIA A40，48GB VRAM）上部署70B参数规模的大语言模型（LLM）** 所面临的三大瓶颈：
- **内存瓶颈**：FP16下70B模型权重超过130GB，远超显存容量；
- **长上下文推理瓶颈**：KV-Cache随序列长度线性增长，在10k token输入时成为主导内存开销；
- **工程集成成本高**：多种压缩技术组合时存在兼容性问题，难以直接堆叠。

传统研究多孤立优化某一技术（如仅量化或仅剪枝），缺乏对**多技术耦合效应**和**实际部署预算约束**的系统考量。

---

### 🚀 提出的新方法与创新思路

作者提出了一种**统一的、预算感知的压缩流水线（budget-aware compression pipeline）**，其核心思想是将单GPU推理建模为一个三维联合优化问题：

> **优化目标 = 在满足以下三个预算的前提下最大化模型质量**
> - `B_mem`：内存预算（≤48GB）
> - `B_thr`：吞吐量预算（≥10 tokens/s，支持10k上下文）
> - `B_int`：集成成本预算（避免定制编译器/重写内核）

#### 创新点包括：
1. **模块化设计框架**  
   将压缩流程分解为三个可插拔模块：
   - **Post-Training Quantization (PTQ)**：4-bit AWQ 权重量化
   - **Structured Pruning**：基于层重要性的深度剪枝（ShortGPT风格）
   - **KV-Cache Optimization**：PyramidKV + INT8 KV量化

2. **揭示关键技术间的耦合效应（Coupling Effects）**
   - 正向耦合：先量化后剪枝 → 更稳定的激活分布；PyramidKV + INT8 KV量化 → 协同降低KV内存而不牺牲解码速度
   - 负向耦合：Vector Quantization（如QTIP）与动态KV缓存冲突；Unstructured Sparsity（如Wanda）破坏AWQ的通道保护机制

3. **引入连续性感知的层重排序策略（Continuity-Aware Layer Reordering）**
   - 不再简单按BI系数排序删除最不重要的层
   - 引入局部连续性评分，优先移除**成片低重要性区域**，提升高比例剪枝下的鲁棒性

4. **端到端可复现评估协议**
   同时报告：
   - 内存占用（NVML实测峰值）
   - 端到端吞吐（prefill + decode）
   - 集成难度等级（plug-and-play vs kernel adaptation）
   - 多项任务准确率（ARC, GPQA等）

---

### 🔍 相比现有方法的优势

| 维度 | 本文方案优势 |
|------|---------------|
| **系统性** | 首次将量化、剪枝、KV压缩纳入统一预算框架，而非孤立优化 |
| **实用性** | 所有组件均基于开源实现，可在Transformers中快速集成 |
| **有效性** | 在真实硬件上实现70B模型在单A40上的完整长上下文推理 |
| **可扩展性** | 提供搜索空间模板，支持自动化pipeline探索 |

---

## 2. 核心实验方法和设置

### 📚 数据集使用

| 类型 | 名称 | 用途 |
|------|------|------|
| **校准数据** | WikiText2, C4 | 用于AWQ量化和剪枝前的激活统计 |
| **下游评测** | ARC-Easy, ARC-Challenge, Winogrande, GPQA-Diamond | 衡量常识与推理能力保留情况 |
| **长上下文测试** | LongBench（官方验证集） | 使用11,427-token输入模拟极端场景 |

> 注：校准数据与评测数据严格分离，确保无泄露。

---

### ⚙️ 实验设置

| 项目 | 设置详情 |
|------|----------|
| **硬件平台** | NVIDIA A40 (48GB VRAM), Dual-Xeon CPU Host |
| **软件环境** | CUDA 12.2, PyTorch 2.3, AutoAWQ, HuggingFace Transformers |
| **模型** | DeepSeek-R1-Distill-Llama-70B |
| **输入长度** | 主要测试 10,240 ~ 11,427 tokens |
| **输出长度** | 256 tokens（greedy decoding） |
| **批大小** | Batch size = 1（memory-bound setting） |

---

### 🎯 评估指标

| 指标类别 | 具体指标 |
|--------|---------|
| **内存效率** | Peak VRAM Usage (MB), Model Artifact Size |
| **推理性能** | Prefill Speed (tok/s), Decode Speed (tok/s), Total Throughput |
| **模型质量** | Task Accuracy (ARC-E, GPQA-D等)，Perplexity（WikiText2/C4） |
| **工程成本** | Integration Effort Score：<br>• Plug-and-play<br>• Kernel Adaptation<br>• Compiler Reimplementation |

---

### 🔁 基线方法对比

| 基线方法 | 描述 |
|--------|------|
| **Dense FP16** | 原始全精度模型，无法放入A40 |
| **AWQ Only** | 仅4-bit权重量化，仍超出内存预算 |
| **ShortGPT Only** | 仅层剪枝，未解决KV-Cache膨胀问题 |
| **SliceGPT** | 宽度剪枝，需重写内核，集成困难 |
| **Wanda** | 非结构化剪枝，破坏量化稳定性 |
| **vLLM + AWQ** | 使用PagedAttention的强baseline，但在本设定下仍OOM |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（最终配置）

| 指标 | 数值 |
|------|------|
| **模型大小** | ~33 GB（原始FP16约132GB） |
| **峰值显存占用** | **42,169 MB (~42.2 GB)** <br>✅ 满足48GB限制 |
| **解码速度（decode speed）** | **57.21 tokens/s** <br>✅ 超过目标阈值（>10 tok/s） |
| **总吞吐** | ~221 tokens/s（含prefill） |
| **任务准确率下降** | ≤ **5% absolute** on ARC & GPQA |
| **集成成本** | Kernel Adaptation level（无需重编译） |

> ✅ 成功实现：**Fit + Fast + Faithful** 的单卡70B部署

---

### 🔀 与基线方法对比结果

| 配置 | 显存 | 是否成功运行 | 解码速度 | 准确率损失 |
|------|------|----------------|-----------|-------------|
| Dense FP16 | >130GB | ❌ OOM | — | — |
| AWQ Only | ~38GB + KV >48GB | ❌ OOM | — | — |
| AWQ + Pruning | ~33GB + KV >48GB | ❌ OOM | — | — |
| **AWQ + Pruning + PyramidKV + INT8 KV** | **~42.2GB** | ✅ Yes | **57.21 tok/s** | **<5% ↓** |

> 只有**三者协同**才能同时满足三项预算

---

### 🔍 消融实验结果

#### （1）不同剪枝策略对比（Table 7）

| 方法 | 剪枝层数 | ARC-E | GPQA-D |
|------|--------|-------|--------|
| Dense | 0 | 0.8068 | 0.6313 |
| ShortGPT (原版) | 15 | 0.7534 | 0.4848 |
| **Continuity-Aware (本文)** | 15 | **0.7580** | **0.5303** (+4.55 pts!) |
| ShortGPT | 20 | 0.7012 | 0.2879 |
| **Continuity-Aware** | 20 | **0.7243** | **0.4091** (+12.12 pts!) |

> 连续性感知重排序显著缓解高比例剪枝带来的精度崩溃，尤其在复杂推理任务（GPQA）上效果明显。

#### （2）KV-Cache优化组合（Table 9）

| 方法 | 峰值内存 | 解码速度 |
|------|--------|--------|
| Full KV (FP16) | 40,690 MB | 68.51 tok/s |
| Full KV (INT8) | 39,615 MB | 55.75 tok/s |
| **PyramidKV + INT8** | **39,029 MB** | **57.21 tok/s** |

> PyramidKV在减少内存的同时，**几乎不损失解码速度**，优于纯量化方案。

#### （3）耦合顺序影响（Section 5）

- **先剪枝后量化**：导致激活异常，解码不稳定（perplexity飙升）
- **先量化后剪枝**：稳定且高效，误差传播路径更短
- **Vector Quantization + Dynamic KV**：❌ 编译失败，因静态shape假设冲突

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **“最优单点技术 ≠ 最优组合”**  
   单独最强的技术（如QTIP 2-bit量化）可能因负向耦合而无法集成，**pipeline级优化优于component级优化**。

2. **正向耦合能突破性能边界**  
   - AWQ + ShortGPT：量化稳定 + 计算节省
   - PyramidKV + INT8 KV：内存压缩叠加，互不干扰

3. **顺序至关重要（Order Matters）**  
   “先量化 → 再剪枝” 比反过来更稳定，说明**压缩操作应从对结构扰动最小的方式开始**。

4. **架构感知剪枝优于盲目排序**  
   层重要性分布具有结构性（banding pattern），利用连续性先验可安全扩大剪枝比例。

5. **KV-Cache是长上下文瓶颈的关键开关**  
   即使模型压缩到位，若不处理KV-Cache，仍会OOM；**runtime memory管理决定可行性**。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **适用范围有限** | 当前方案针对 **70B dense模型 + A40 + 英文QA任务** 设计，不保证泛化至MoE、多语言或安全敏感场景 |
| **未覆盖极低端设备** | 无法部署到≤24GB消费卡（如RTX 3090） |
| **集成成本主观性** | `B_int` 依赖当前serve stack，若有原生sparse支持则结论可能变化 |
| **缺乏多任务微调** | 压缩后未进行轻量微调恢复性能 |

---

### 🔮 未来工作方向

1. **自动化Pipeline Search**  
   构建基于强化学习或贝叶斯优化的搜索系统，在给定预算下自动选择最佳组合（bit-width, pruning ratio, KV retention rate）。

2. **跨硬件适配引擎**  
   开发可根据目标GPU（A100 vs RTX 4090 vs H100）自适应调整压缩策略的编译器前端。

3. **联合训练-压缩范式**  
   探索在蒸馏阶段即引入结构稀疏性与量化友好性，进一步提升压缩极限。

4. **动态弹性压缩**  
   根据输入长度、用户QoS需求实时调节压缩强度，实现“按需降级”。

---

## 总结

> 本文标志着LLM压缩从“单项冠军”走向“全能选手”的转变——不再追求极致压缩率，而是强调**在真实部署预算下实现内存、速度、质量、工程成本的均衡**。

通过揭示量化、剪枝、KV压缩之间的**耦合规律**，并提出**continuity-aware pruning** 和 **budget-aware protocol**，该工作为构建下一代**自动化、可复现、生产就绪的LLM压缩系统**奠定了坚实基础。

</details>

---

### 10. [A Smallest-Need-First Job Scheduling Framework with Adaptive Optimization of Idle Node Counts for Energy-Efficient HPC Systems](https://arxiv.org/abs/2608.29656)

**Authors**: Reza Pulungan, Raka Satya Prasasta, Santana Yuda Pradata,  Mursalim, Hiroyuki Takizawa, Muhammad Alfian Amrizal  
**Category**: cs.DC  
**Published**: 2026-09-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.29656v1  

#### Abstract
Power-state management in high-performance computing (HPC) clusters must reduce idle energy without excessive wake-up delays for rigid parallel jobs. This paper presents SNF-ICON, an event-driven controller combining smallest-need-first (SNF) gang scheduling, predictive wake timing, and adaptive war...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Smallest-Need-First Job Scheduling Framework with Adaptive Optimization of Idle Node Counts for Energy-Efficient HPC Systems

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对 **High-Performance Computing (HPC)** 系统中的能效优化问题，特别是**刚性并行作业（rigid parallel jobs）**在调度过程中面临的“等待时间”与“能耗”之间的权衡挑战。核心问题是：

- 如何在减少空闲节点能耗的同时，避免因频繁唤醒节点导致的作业等待时间增加。
- 现有方法（如 FCFS + backfilling）未能最小化平均等待时间，尤其在大规模多服务器作业场景下表现不佳。

### **提出了什么新方法或新思路**

作者提出了一种名为 **SNF-ICON** 的事件驱动型双模式调度框架，其核心创新包括：

- **Smallest-Need-First (SNF) Gang Scheduling**：优先调度请求节点数最少的作业，理论上可逼近最优等待时间下界。
- **Adaptive Optimization of Idle Node Counts (ICON)**：动态决定应保持为“热备”（warm spare）状态的空闲节点数量，以平衡唤醒延迟与能耗。
- **Markovianity-Gated Control**：引入一个基于近期工作负载特征的“马尔可夫性检测器”，仅当数据符合指数分布假设时才启用 ICON 模型；否则退回到更稳健的 **SNF+IPM** 模式。
- **Predictive Wake Timing**：利用历史运行时数据预测作业完成时间，并据此规划节点唤醒时机，减少不必要的空转。

### **相比现有方法的优势**

| 对比维度 | SNF-ICON | FCFS/B+IPM | RL-based 方法 |
|--------|---------|-----------|-------------|
| **等待时间** | 显著降低（所有测试案例） | 较高 | 可极低，但代价高昂 |
| **能耗控制** | 接近最优，无剧烈波动 | 中等 | 极端情况下能耗翻倍 |
| **鲁棒性** | 高（通过 fallback 机制） | 高 | 依赖训练数据，泛化差 |
| **部署复杂度** | 无需离线训练，参数少 | 简单 | 需大量训练与调优 |

> ✅ **优势总结**：SNF-ICON 在不显著增加能耗的前提下，系统性地降低了作业平均等待时间，且具备良好的适应性和稳定性。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **真实工作负载**：
  - **DAS2 FS1–FS4**：来自生产级 HPC 集群的工作负载轨迹，前 3000 个作业用于主实验。
- **合成工作负载**：
  - **Generated Markovian Workload**：服从指数到达与服务时间的泊松过程模型，共 3000 个作业。
- **跨平台验证**：
  - **SDSC Blue**：映射到 AOBA 衍生的 1152 节点模型上进行扩展测试。

### **实验设置**

- **模拟平台**：基于 **SPARS**（Reinforcement Learning-enabled Simulator for Power Management）构建离散事件模拟器。
- **节点模型**：
  - AOBA-derived 64-node 和 1152-node 配置。
  - 包含完整的节点状态转换模型（Computing, Idle, Switching-On, Switching-Off, Sleeping）及相应功耗与时延。
- **默认参数**（见 Table I）：
  - 初始到达率 λ₀ = 1/3600 s⁻¹
  - 屏蔽窗口大小 Tₘ = 2h
  - 最小样本数 n_min = 3
  - 权重 α = 5000, β = 1（等待 vs 能耗）

### **评估指标**

| 指标 | 缩写 | 描述 |
|------|-----|------|
| 平均队列等待时间 | AW / Mean Waiting Time | 主要延迟指标 |
| 总浪费能量 | EW / Total Wasted Energy | 非计算状态下的总能耗 |
| 最大等待时间 | MW | 衡量极端情况 |
| 系统利用率 | SU | 资源使用效率 |
| 能效比 | EE | 综合性能指标 |

### **基线方法对比**

| 方法 | 简称 | 特点 |
|------|------|------|
| FCFS + Backfilling + IPM | FCFS/B+IPM | 当前主流策略，作为主要 baseline |
| SNF + IPM | SNF+IPM | 控制变量，验证 SNF 本身效果 |
| Reinforcement Learning (Curriculum Learning) | RL Budiarjo | 数据驱动型先进方法，用于展示极端 trade-off |
| SNF-ICON 家族变体 | SNF-ICON-NF, NG, NGNF | 消融实验用 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据汇总**

| 工作负载 | 平台 | 相对于 FCFS/B+IPM 的等待时间下降 | 能耗偏离最近基线程度 |
|--------|-------|-------------------------------|------------------|
| DAS2 FS1 | AOBA-64 | ~2.2% ↓ | ≈1% 内 |
| DAS2 FS2 | AOBA-64 | **~44.8% ↓** | ≈1% 内 |
| DAS2 FS3 | AOBA-64 | ~2.4% ↓ | ≈1% 内 |
| DAS2 FS4 | AOBA-64 | ~11.6% ↓ | ≈1% 内 |
| Markovian 3000 | AOBA-64 | ~42.4% ↓ | +6.2%（高于 SNF+IPM） |
| SDSC Blue | AOBA-1152 | ~16.4% ↓ | ≈1% 内 |

> 🔺 注：尽管在生成的 Markovian 工作负载中能耗略高（+6.2%），但其等待时间改善显著，仍具实用价值。

### **与基线方法的对比结果**

- **在所有六种 workload-platform 组合中，SNF-ICON 均优于 FCFS/B+IPM 的平均等待时间**。
- 相较于 **SNF+IPM**，SNF-ICON 进一步提升了五种情况下的性能（最大提升达 26.9%），仅在 SDSC Blue 上略有上升（+0.15%）。
- 与 **RL 方法**相比：
  - RL 可实现更低等待时间（如 DAS2 FS4 下降 76%），但能耗激增 **4.8–5.8 倍**。
  - RL 在节能模式下反而导致等待时间飙升至 **1.9–6.5 倍**。
  - ➠ 表明 RL 政策存在严重过拟合与极端 trade-off 问题。

### **消融实验结果**

#### （1）**Fallback 机制的影响（NF vs 完整 SNF-ICON）**

- 在 DAS2 FS3 上，强制进入 ICON 模式（NF）虽略微降低等待时间，但能耗更高。
- 在大多数情况下，fallback 机制有效防止了模型误用带来的性能劣化。

#### （2）**Arrival-Recency Gate 的影响（NG vs 完整）**

- 关闭门控（NG）对整体性能影响较小，说明该机制主要用于抑制无效预热。
- 与 fallback 相比，gate 的作用更温和。

#### （3）**Markovianity Lookback Horizon 扫描（Tₘ = 1h ~ 24h）**

- 对 DAS2 工作负载影响有限，表明 fallback 已吸收大部分不确定性。
- 对 **Markovian 3000** 敏感：较长窗口（如 24h）可进一步降低等待时间（从 2.9min → 2.0min），但伴随能耗微升。

#### （4）**Reward Weight α 扫描（α = 10 ~ 20,000）**

- α 越大，越偏向减少等待时间，能耗随之上升。
- 存在明显的帕累托前沿（Pareto Frontier），最佳选择取决于具体应用场景偏好。

#### （5）**Cross-Platform Robustness（AOBA vs Taurus）**

- 不同平台模型下绝对数值差异巨大（如 SDSC Blue 在 Taurus 上能耗仅 5.22MWh，在 AOBA 上为 36–39MWh）。
- 但 **SNF 类方法始终处于低等待区域**，表明其优势具有跨平台一致性。
- 最优配置随平台变化，强调需根据目标系统的 power model 和 transition behavior 调参。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **SNF 是降低刚性作业等待时间的有效策略**：即使在非理想条件下，也能稳定优于 FCFS。
2. ✅ **ICON 模型在适配马尔可夫性假设时能显著提升性能**：尤其在合成或接近泊松过程的真实负载中表现突出。
3. ✅ **Fallback 机制至关重要**：它使得系统能在不确定或稀疏数据下保持稳健，避免模型误用。
4. ✅ **没有单一最优参数适用于所有场景**：最佳配置高度依赖 workload 特征和 platform model。
5. ✅ **SNF-ICON 实现了良好的 energy-delay trade-off**：相比 RL 方法，其性能提升更为均衡可靠。

### **方法的局限性**

- ❗ **依赖 workload 的统计特性**：若长期偏离指数假设（如突发性极强的负载），ICON 模式激活频率低，难以发挥优势。
- ❗ **未处理抢占（preemption）和弹性作业（malleable jobs）**：当前仅支持 rigid gang scheduling。
- ❗ **预测精度受限于用户提供的 requested runtime**：虽然引入了 log-ratio correction，但仍受初始估计偏差影响。
- ❗ **warm-spare 决策为单步近似（one-step）**：未考虑多步未来事件序列，可能错过全局最优。

### **未来工作方向**

1. **扩展至支持 preemptive 和 malleable jobs**，增强适用范围。
2. **结合轻量级在线学习机制**，动态调整 ICON 模型参数，提高自适应能力。
3. **探索多层级 spare capacity control**（如分组休眠、部分唤醒）。
4. **集成 renewable energy 或 power capping 约束**，构建更全面的绿色调度框架。
5. **在真实 HPC 系统中部署原型并实测验证**，评估实际开销与收益。

---

> 📌 **总体评价**：  
> SNF-ICON 是一项兼具理论动机与工程可行性的创新工作。它将经典的排队论思想（SNF）、现代预测技术与自适应控制相结合，提供了一个**稳健、高效、易于部署**的能量感知调度方案，为下一代绿色 HPC 系统的设计提供了重要参考。

</details>

---

### 11. [ToxLens: A Reproducible Graph-Learning Framework for Leakage-Aware, Uncertainty-Calibrated Molecular Toxicity Prediction](https://arxiv.org/abs/2608.30472)

**Authors**: Magnus H. Str{\o}mme, Alex G. C. de S\'a, David B. Ascher  
**Category**: cs.LG  
**Published**: 2026-09-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.30472v1  

#### Abstract
Molecular toxicity prediction is increasingly used to prioritise compounds before experimental testing, but conventional benchmark performance can overstate practical utility when structurally related molecules occur across training and test folds. We introduce ToxLens, a reproducible multi-task gra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ToxLens: A Reproducible Graph-Learning Framework for Leakage-Aware, Uncertainty-Calibrated Molecular Toxicity Prediction

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
分子毒性预测模型在传统基准测试中常因训练集与测试集之间存在**结构相似性或近似重复分子**（structural leakage）而高估其实际泛化能力。此外，现有方法普遍存在以下缺陷：
- **缺乏可靠性报告**：未提供不确定性量化（uncertainty quantification）和适用域分析（applicability domain）；
- **解释性不足**：黑箱模型难以提供化学上可解释的毒性子结构依据；
- **强制分类决策**：即使对分布外（OOD）分子也强行给出二元分类，不利于实际化合物优先级排序。

### 🚀 提出的新方法与创新
作者提出 **ToxLens**，一个可复现、多任务的图学习框架，集成多项提升模型可靠性的技术，其核心创新包括：

| 创新技术 | 说明 |
|--------|------|
| **Leakage-Aware 数据划分** | 采用 `UMAP-HDBSCAN` 聚类分割策略，结合 `sphere-exclusion` 过滤，减少训练/测试间结构泄漏，确保更真实的泛化评估。 |
| **Late Concatenation 架构设计** | 并行使用 **Graph Neural Network (GINE)** 和 **Global Molecular Features Encoder**，仅在预测头前进行特征拼接，避免早期融合导致的信息压制。 |
| **不确定性校准与置信预测集** | 引入 **temperature-scaled Monte Carlo dropout** 与 **conformal-style prediction sets**，允许模型输出“不确定”类别（双标签集合），实现校准的 abstention 决策。 |
| **可解释性与反事实验证** | 结合 **GradientSHAP** 归因分析与 **occlusion controls**，并通过 **consensus subgraph mining** 发现潜在 toxicophores，并以反事实标准验证其影响。 |

### 🔍 相比现有方法的优势
- 在相同泄漏控制划分下，**全面超越四种基于 ECFP4 的浅层模型**（Random Forest, XGBoost, MLP, SVM）在全部 11 个毒性终点上的表现；
- 提供完整的“可靠性堆栈”（reliability stack）：从数据划分到不确定性报告再到化学解释，增强模型透明度与实用性；
- 不追求单一指标 state-of-the-art，而是强调**可审计性、可复现性和实用导向的设计原则**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **11 个毒性终点**，涵盖：
  - **Ames**：细菌致突变性
  - **LD50_Zhu**：急性口服毒性
  - **hERG_Karim**：hERG 钾通道抑制
  - **Tox21** 系列：核受体（NR-AhR, NR-ER 等）和应激反应通路（SR-ARE, SR-MMP 等）
- 数据来源：
  - Therapeutics Data Commons (TDC) 中的 Ames, LD50_Zhu, Tox21；
  - 公开的 hERG_Karim 数据集；
  - 外部验证使用 **Tox21 Challenge** 和 **TDC ADMET** 固定划分。

### ⚙️ 实验设置
| 组件 | 设置详情 |
|-----|---------|
| **数据预处理** | RDKit 标准化、去盐、互变异构归一化、生成 canonical SMILES；冲突标签设为缺失；连续值平均后二值化。 |
| **数据划分** | 主实验采用 **UMAP-HDBSCAN 分割**（保留簇完整性），并与其他方式（random, Butina, scaffold）比较；外部基准保留原始划分。 |
| **特征工程** | 
| - 图节点特征（134维） | 元素、电荷、杂化、芳香性、药效团标志等 |
| - 图边特征（35维） | 键序、立体化学、共轭、环成员等 |
| - 全局特征（3,190维） | ECFP4、RDKit 描述符、SMARTS 毒性模式、MolFormer-XL 嵌入、3D 几何描述符等（有效非零维度为 2,990） |
| **模型架构** | 
| - 图主干 | 5 层残差 GINE，带虚拟节点（virtual node）和随机深度（stochastic depth） |
| - 全局路径 | 独立 FFN 编码器 |
| - 融合方式 | Late concatenation（优于 GCMI/FiLM） |
| - 输出 | 多任务二分类，每任务独立阈值（validation MCC 最优） |
| - 集成 | 5 种随机种子的 soft-voting ensemble |

### 📊 评估指标
| 指标 | 用途 |
|------|------|
| **MCC (Matthews Correlation Coefficient)** | 综合衡量分类平衡性能，尤其适用于不平衡数据 |
| **AUROC (Area Under ROC Curve)** | 排序能力评估 |
| **AUPRC (Area Under PR Curve)** | 对阳性样本稀疏的任务更敏感 |
| **Brier Score** | 概率预测质量（校准性） |
| **Conformal Prediction Set Efficiency** | 单例预测比例（singleton rate），反映模型自信程度 |
| **Applicability Domain Analysis** | 按最大训练集相似性分四分位评估性能趋势 |

### 🆚 基线方法对比
- 四种基于 **ECFP4 (1024-bit)** 的浅层模型：
  - Random Forest
  - XGBoost
  - Multilayer Perceptron (MLP)
  - Support Vector Machine (SVM)
- 所有模型使用**相同的训练/验证/测试划分**和**相同的阈值选择协议**（validation MCC 最大化），确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（五种子软投票集成）

| 指标 | 数值 |
|------|------|
| **Macro MCC** | **0.44** |
| **Macro AUROC** | **0.83** |
| **Macro AUPRC** | **0.58** |
| **Brier Score** | 0.11 |

> 在所有 11 个毒性终点上均优于 ECFP4 浅层基线。

### 📊 与基线方法的对比结果
- **全面领先**：ToxLens ensemble 在所有 11 个任务上 MCC 均高于最强浅层基线（通常是 RF 或 XGBoost）；
- **最大优势出现在**：
  - **SR-MMP**: +0.19
  - **SR-HSE**: +0.16
  - **NR-ER**: +0.15
- **最小优势出现在**：
  - **NR-AhR**: +0.03
  - **Ames**: +0.04
  - **hERG_Karim**: +0.07

> 表明 ToxLens 尤其擅长处理某些复杂或稀疏信号较强的终点。

### 🔬 消融实验结果（Ablation Study）

| 变体配置 | Test MCC | 关键发现 |
|--------|----------|---------|
| **Full GCMI** | 0.382 | 性能较差，表明门控融合可能破坏互补信息 |
| **FiLM** | 0.398 | 略好于 GCMI，但仍不如 late concat |
| **Mid-Trunk Concatenation** | 0.403 | 表现中等 |
| **No Mid-Trunk Fusion (Late Concat)** ✅ | **0.438 / 0.425** | **最佳表现**，支持 late concatenation 为最优策略 |
| **Remove Global Pathway (GNN-only)** ❌ | 0.341 | 显著下降，证明全局特征提供重要补充信息 |
| **Linear Head (vs Deep)** | 0.407 | 差异不大，说明 head 深度不是关键因素 |

> **结论**：late concatenation + 全局路径 是本任务面板下的最优组合。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **结构泄漏显著影响评估真实性**：通过 UMAP-HDBSCAN 分割可有效降低近似分子跨集泄露，使模型评估更贴近真实应用场景。
2. **late concatenation 优于模态调制机制**：GCMI 和 FiLM 等早期融合方式未能提升性能，反而可能削弱图与全局特征的互补性。
3. **全局特征至关重要**：移除全局路径导致 MCC 下降约 10%，说明传统描述符仍具不可替代价值。
4. **不确定性与适用域密切相关**：
   - 分布内分子（Q4）平均 AUROC 达 0.86，MCC 0.44；
   - 分布外分子（Q1）降至 AUROC 0.77，MCC 0.28；
   - 支持将相似性作为适用域警告指标。
5. **conformal-style prediction sets 效率高度依赖任务**：
   - **NR-ER-LBD**：100% 单例预测（非常确定）
   - **LD50_Zhu**：仅 20.4% 单例，其余标记为“不确定”，提示需进一步审查
6. **可解释性产出合理假设**：
   - 发现 49 个共识毒性片段（consensus clusters）；
   - 其中 **44 个至少有一个实例通过反事实验证标准**；
   - 包括已知毒理模式如 **phenol (SR-MMP)**、**tertiary aminoalcohol (hERG)**、**aromatic amine (Ames)** 等。

### ⚠️ 方法的局限性
1. **单一分割策略**：尽管 UMAP-HDBSCAN 设计严谨，但结论仍基于单一固定划分，未评估 split variability；
2. **外部基准为重新训练而非零样本迁移**：Tox21 和 TDC 实验是对架构的 transfer 测试，不能代表 11-endpoint checkpoint 的泛化能力；
3. **conformal calibration 存在偏差**：validation fold 同时用于 checkpoint selection、temperature scaling 和 calibration，违反严格 conformal 假设，coverage 仅为目标操作水平（target operating level）；
4. **解释性结果为模型级假设**：SHAP + occlusion 仅验证模型内部一致性，不等于生物学因果关系，需实验验证；
5. **PubChem 生物活性块为空**：200 维 PubChem 特征全为零，无法评估其作用。

### 🔮 未来工作方向
1. **多轮 chemistry-aware splits**：引入时间序列或外部队列进行前瞻性验证；
2. **动态 recalibration**：在部署过程中监测分布漂移并调整 conformal threshold；
3. **实验验证模型推断的 toxicophores**：将计算发现的结构假说送入湿实验验证；
4. **扩展多任务模型至更多 assay metadata**：整合细胞系、暴露时间等条件信息；
5. **multi-seed Tox21 Challenge 评估**：消除与 DeepTox ensemble 的不对称性比较。

---

## 总结
ToxLens 并非追求单一性能突破的“SOTA 模型”，而是一个**面向实用场景的、系统化的毒性预测决策支持框架**。它通过整合泄漏感知划分、不确定性建模、适用域分析与可解释性工具，构建了一个**可审计、可复现、可行动**的 QSAR 工作流范式，推动计算毒理学向更高透明度与可信度发展。

</details>

---

### 12. [Liquid Gated Attention](https://arxiv.org/abs/2608.30695)

**Authors**: Yiheng Jiang, Yuanbo Xu, Yongjian Yang  
**Category**: cs.LG  
**Published**: 2026-09-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.30695v1  

#### Abstract
Real-world time series often exhibit irregular sampling and extended temporal horizons, requiring models to capture continuous-time dynamics across arbitrary intervals without prohibitive scaling costs. Discrete-time methods collapse variable time intervals into static positional steps; solver-depen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Liquid Gated Attention**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现实世界的时间序列通常具有**不规则采样**（irregular sampling）和**长时程依赖**（extended temporal horizons）两大挑战。现有方法存在以下缺陷：

- **Discrete-time models**（如 RNN、Transformer）将时间投影到均匀离散索引上，丢失了观测之间的实际时间间隔，无法建模连续时间动态。
- **Solver-dependent continuous-time models**（如 NODE、NCDE）虽保留时间结构，但依赖数值求解器进行顺序积分，导致计算不可并行化，且在长序列中易累积截断误差。
- **Solver-free models** 虽可并行，但缺乏将**观测时间间隔**与**输入驱动的状态调制**显式耦合的机制，难以区分真实动态与噪声。

### **提出的新方法：Liquid Gated Attention (LGA)**
本文提出 **Liquid Gated Attention (LGA)**，一种**无求解器**（solver-free）、**可并行**的连续时间注意力算子，其核心思想是将**液体时间常数网络**（Liquid Time-Constant, LTC）中的门控机制嵌入到线性注意力框架中。

#### **四大设计步骤**：
1. **Continuous-time Gating**  
   从一维 LTC 方程推导出“液体门”（liquid gate），该门控同时编码**时间衰减**（temporal decay）和**输入自适应调制**（input-driven modulation），为模型引入连续时间归纳偏置。

2. **Computational Efficiency**  
   用**可学习端点插值**（learnable endpoint interpolation）近似积分项，灵感来自梯形法则。固定插值权重为 0.5 可恢复经典梯形规则，而可学习配置则作为可训练的数值代理，实现并行计算。

3. **Expressive Capability**  
   将标量状态提升为**矩阵值关联记忆**（matrix-valued associative memory），采用 fast-weight 编程框架，使液体门成为注意力特征的显式调制系数，支持线性复杂度的并行序列建模。

4. **Numerical Safety**  
   引入**序列级归一化**（sequence-level normalization），约束累积衰减系数，防止长序列优化中的指数下溢和梯度消失，提升训练稳定性。

### **相比现有方法的优势**
- ✅ **保持连续时间建模能力**：通过液体门显式耦合时间间隔与输入调制。
- ✅ **完全并行化**：避免数值求解器，实现线性时间复杂度 $O(n)$。
- ✅ **抗噪鲁棒性**：液体门的凸组合特性（∈(0,1)）天然抑制高频噪声传播。
- ✅ **模块化设计**：基于 LGA 构建的 **LFormer** 是一个通用、可扩展的 backbone。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共涵盖 **6 类任务**、**16 个数据集**，最长序列达 **17,984 步**：

| 任务 | 数据集 | 特点 |
|------|--------|------|
| **LTS-C/R** (长时分类/回归) | EC, EW, HB, MI, SCP1, SCP2, HR, RR, SpO2 | 长序列（最长 17,984），密集信号，预测全局标签或标量 |
| **PTS-C/R** (逐点分类/回归) | HA, PDL | 不规则采样，需精细状态追踪 |
| **TS-I/E** (插值/外推) | 2DS, USH, PHY | 从稀疏、含噪观测中重建轨迹 |

### **实验设置**
- **模型配置**：默认 `L=2` Liquid Mixers, `d=64`, `H=4` heads, `dff=176`
- **优化器**：Adam，学习率 `1e-3`（部分任务 `1e-2`）
- **早停**：验证集指标连续 20 轮未提升即停止
- **硬件**：单卡 RTX 4090，Intel i9-13900K，128GB RAM

### **评估指标**
| 任务类型 | 指标 |
|---------|------|
| 分类（LTS-C, PTS-C） | Top-1 Accuracy |
| 回归（LTS-R） | MAE, RMSE |
| 逐点回归（PTS-R） | MSE |
| 插值/外推（TS-I/E） | MAE, RMSE, MSE |

### **基线方法对比**
涵盖三大范式：
- **Discrete-time**: GRU, Transformer, LRU, Mamba
- **Solver-dependent**: NODE, NCDE, NRDE, ContiFormer, CRU
- **Solver-free**: CfC, RFormer, mTAND, S5, ACSSM

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **长时分类（LTS-C）**
| 方法 | 平均准确率 (%) | 平均排名 |
|------|----------------|----------|
| **LFormer (ours)** | **71.3** | **1.33** |
| RFormer | 64.5 | 4.50 |
| S5 | 61.8 | 5.92 |
| Transformer | 49.2 | 6.50 |

> 在 6 个数据集中，LFormer 在 4 个上排名第一，尤其在超长序列 **EW (17,984步)** 上稳定运行，而 Transformer 内存溢出（OOM）。

#### **长时回归（LTS-R）**
| 数据集 | 指标 | 最佳基线 | **LFormer** | **相对提升** |
|-------|------|-----------|------------|---------------|
| HR | RMSE | 1.52 (CfC) | **1.03** | **+47.72%** |
| RR | RMSE | 0.97 (CfC) | **0.74** | **+20.43%** |
| SpO2 | RMSE | 0.42 (CfC) | **0.29** | **+71.57%** |

> 显著优于所有基线，尤其在高噪声 SpO2 任务上表现突出。

#### **插值与外推（TS-I/E）**
| 任务 | 数据集 | 指标 | 最佳基线 | **LFormer** | **相对提升** |
|------|--------|------|-----------|------------|---------------|
| TS-I | PHY | MSE | 0.116 (ACSSM) | **0.040** | **+65.34%** |
| TS-I | USH | MSE | 0.006 (ACSSM) | **0.001** | **+83.33%** |
| TS-E | USH | MSE | 0.941 (ACSSM) | **0.840** | **+10.73%** |
| TS-E | PHY | MSE | 0.340 (mTAND) | 0.372 | -9.41% |

> 在插值任务上大幅领先；在外推任务上对平稳信号（USH）有效，但对非平稳信号（PHY）略逊于 mTAND。

#### **逐点任务**
| 任务 | 数据集 | 指标 | LFormer | 最佳基线 |
|------|--------|------|---------|----------|
| PTS-C | HA | Accuracy | **92.9%** | 91.3% (ACSSM) |
| PTS-R | PDL | MSE | **2.1‰** | 3.1‰ (ACSSM) |

> 在 PDL 上实现 **31.2%** 的 MSE 降低，显示其在精细状态追踪上的优势。

---

### **消融实验结果（Ablation Study）**

| 组件 | 影响（平均性能下降） |
|------|------------------|
| **w/o MHLGA** | -20.28% (LTS-R), -41.22% (TS-I) |
| **w/o Output Gate** | -14.65% (TS-E), 训练崩溃（PDL） |
| **w/o Decoupled u** | -3.47% (LTS-C), 训练崩溃（PTS-R） |
| **w/o Learnable u** | -2.00% (LTS-C), **+23.90%** (PTS-R) |
| **w/o Input-aware g** | -15.83% (LTS-R), -39.02% (PTS-R) |
| **w/o Normalized g** | -32.88% (TS-I), 多任务训练崩溃 |

> **关键发现**：
> - **MHLGA 和输出门** 对回归和稀疏任务至关重要。
> - **可学习插值** 在多数任务上有益，但在高度稀疏/噪声条件下（如 PDL），固定 `μ=0.5` 更优，因其提供低通平滑先验。
> - **序列级归一化** 是长序列训练稳定的必要条件。

---

## **4. 关键结论和发现**

### **主要发现**
1. **LGA 成功将连续时间动力学嵌入并行架构**：通过液体门机制，实现了时间间隔与输入调制的显式耦合，兼顾建模能力与效率。
2. **LFormer 在长程依赖、精细追踪、轨迹重建三大能力上全面领先**：尤其在稀疏、含噪场景下表现出强鲁棒性。
3. **线性复杂度得到实证验证**：在长达 17,984 步的序列上仍保持高效，内存和延迟均为线性增长。
4. **序列级归一化是训练稳定的关键**：解决了长序列中累积衰减导致的梯度消失问题。

### **方法的局限性**
1. **序列级归一化依赖全序列访问**：当前版本适用于离线任务，**在线流式处理**需前缀归一化变体。
2. **可学习插值可能过拟合**：在极端稀疏/噪声条件下，固定 `μ=0.5` 表现更优，说明需任务自适应策略。
3. **外推非平稳信号能力有限**：在高度非平稳的 PHY 数据上，mTAND 的查询机制更具优势。

### **未来工作方向**
1. 设计**流式兼容的归一化机制**（prefix-normalized variant）。
2. 开发**任务自适应的端点插值策略**，平衡灵活性与稳定性。
3. 探索**LGA 与时间感知查询网络**（time-aware query networks）的融合，结合全局一致性与局部敏感性。

---

> **总结**：  
> **LGA** 通过将液体门控与 fast-weight 注意力结合，首次实现了**并行、无求解器、显式时间耦合**的连续时间建模。  
> **LFormer** 作为其实例化，展现出强大的表达能力、鲁棒性和效率，为时间序列表示学习提供了一个**可扩展、通用的 backbone**。

</details>

---

### 13. [TDDM-Melatt: A Decoupled Memory and Diffusion Framework for Generalizable Encrypted Traffic Classification](https://arxiv.org/abs/2608.30745)

**Authors**: Ze Chen, Qiming Yu, Zijia Song, Guozheng Yang, Wei Yan  
**Category**: cs.LG  
**Published**: 2026-09-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2608.30745v1  

#### Abstract
The widespread adoption of encrypted traffic poses severe challenges to current security situational awareness systems based on network traffic monitoring. In existing dataset-driven training and testing studies, limitations such as shortcut learning induced by spurious feature correlations and samp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《TDDM-Melatt: A Decoupled Memory and Diffusion Framework for Generalizable Encrypted Traffic Classification》核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前基于深度学习的 **Encrypted Traffic Classification**（加密流量分类）面临三大核心挑战：

1. **Shortcut Learning（捷径学习）**：模型过度依赖数据集中非因果的显式标识符（如 IP、Port、Flow ID），这些特征与标签存在虚假相关性（spurious correlation），导致在真实网络环境中泛化能力严重下降。
2. **Class Imbalance（类别不平衡）**：现实网络中流量呈长尾分布，少数类（如恶意流量、小众应用）样本稀缺，导致模型偏向多数类，对关键安全事件识别率低。
3. **模型架构耦合性强**：主流方法将 encoder、pre-training backbone 和分类头紧密耦合，难以跨任务迁移，且全量微调会破坏预训练获得的通用表征。

---

### **提出了什么新方法或新思路**

本文提出 **TDDM-Melatt**，一个解耦的内存与扩散框架，旨在实现可泛化的加密流量分类。其核心由两部分构成：

#### **(1) Melatt：解耦的记忆表示模型**
- **Memory-Decoupled 架构**：引入独立的外部记忆模块（external memory module），将“特征提取”与“知识存储”分离。编码器（CG-LSTM）专注时序建模，记忆模块存储各类流量的行为原型（prototype）。
- **Spurious-Correlation-Free 预训练范式**：
  - 数据预处理阶段严格移除 IP、Port、Flow ID 等拓扑信息，并泛化时间戳。
  - 编码器在下游任务中保持冻结（frozen），仅训练轻量级分类器（如 XGBoost），避免模型重新学习捷径特征。
- **CG-LSTM（Competitive Gating LSTM）**：
  - 将传统 LSTM 的三个门（forget, input, output）置于共享的 softmax 框架下，强制其总和为 1，形成“注意力预算”机制。
  - 增强对突发流量与静默背景的区分能力，并天然提升对少数类样本的学习权重。
- **多头交叉注意力解码器（multi-head cross-attention decoder）**：动态查询记忆原型，增强重构能力。

#### **(2) TDDM：面向流量数据的去噪扩散模型**
- **轻量化扩散架构**：摒弃图像领域复杂的 CNN/Attention 结构，采用基于 MLP 的残差网络作为噪声预测器，降低计算开销。
- **特征选择与填充机制**：
  - 基于 Random Forest 的 Gini Importance 进行特征筛选，去除冗余零值特征，在保留 95% 关键信息的同时降低维度超 30%。
  - 生成后通过 **nearest-neighbor padding** 恢复完整特征维度，保持特征间内在一致性。
- **条件引导的紧凑采样（condition-guided compact noise sampling）**：
  - 在反向去噪过程中引入类别质心（centroid）引导项，使生成样本更紧密地围绕真实数据分布，提升生成质量。

---

### **相比现有方法的优势**

| 维度 | 优势 |
|------|------|
| **泛化能力** | 通过拓扑匿名化 + 冻结编码器，彻底切断模型对捷径特征的依赖，显著提升跨场景部署鲁棒性。 |
| **数据效率** | TDDM 可有效生成高质量少数类样本，缓解长尾问题，尤其在细粒度分类任务中提升显著。 |
| **架构灵活性** | “预训练-解耦-分类”范式支持即插即用，下游可灵活替换不同分类器，适应边缘设备等资源受限场景。 |
| **推理效率** | 固定表征 + 轻量分类器设计，单样本推理延迟 <12ms，远优于主流 SOTA 模型（>50ms）。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

在 **4 个代表性公开基准数据集** 上进行验证：

| 数据集 | 任务类型 | 类别数 | 特点 |
|--------|---------|-------|------|
| **CIC-IDS-2017 (IDS)** | 攻击检测 | 2 / 15 类 | 包含良性与多种攻击流量 |
| **ISCX-VPN-2016 (VPN)** | 加密服务识别 | 2 / 6 / 16 类 | 区分 VPN 与非 VPN 流量 |
| **USTC-TFC2016 (USTC)** | Tor 流量识别 | 2 / 20 类 | 专注于匿名网络流量分类 |
| **CSTNET-TLS1.3 (TLS)** | TLS 1.3 网站识别 | 120 类 | 细粒度网站指纹识别，高度不平衡 |

所有数据集均采用 **严格的流级别划分（flow-level splitting）**，确保同一 flow 的包不同时出现在训练集和测试集，防止数据泄露。

---

### **实验设置和评估指标**

#### **数据预处理**
- 提取 **76 个流级统计特征**（仅基于 IP/TCP/UDP 头部，不访问加密载荷）。
- 移除 IP、Port、Flow ID 等显式标识符。
- 时间戳泛化处理。

#### **评估指标**
- **分类性能**：Accuracy (AC)、Precision (PR)、F1-Score（Macro）
- **生成质量**：Maximum Mean Discrepancy (MMD)、Average Pairwise Distance (APD)

#### **基线方法对比**

| 类型 | 基线模型 |
|------|--------|
| **分类模型** | MLP, 1D-CNN, 2D-CNN, LSTM, GRU, Transformer |
| **SOTA 表示学习模型** | ET-BERT, YaTC, NetMamba, TrafficFormer, netFound, Pcap-Encoder |
| **生成模型** | GAN, WGAN-GP, ACGAN, NetDiffus, LRT-DDPM |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **(1) 分类性能对比（Table 3）**

| 任务 | 最佳 F1（TDDM-Melatt） | 第二名（Pcap-Encoder） | 提升幅度 |
|------|------------------------|--------------------------|----------|
| **VPN-6class** | **0.889** | 0.710 | +25.2% |
| **USTC-20class** | **0.966** | 0.871 | +10.9% |
| **TLS-120class** | **0.681** | 0.637 | +6.9% |

> ✅ TDDM-Melatt 在所有多类任务上均取得 **SOTA 性能**，尤其在高难度、长尾任务（如 TLS-120class）中优势明显。

#### **(2) 泛化鲁棒性测试（Table 4）**

当对 Pcap-Encoder 输入逐步施加匿名化处理：
- 移除载荷 → F1 下降不明显
- 移除 IP/Port → **TLS-120class 的 F1 从 0.637 骤降至 0.130**
- 而 TDDM-Melatt 在同等条件下仍保持 **F1=0.681**，验证其对拓扑匿名化的强鲁棒性。

#### **(3) 数据增强效果（Table 5）**

| 任务 | TDDM-Melatt F1 提升 | 其他生成模型效果 |
|------|--------------------|------------------|
| **TLS-120class** | **+9.98%**（F1 从 0.581 → 0.681） | GAN/DDPM 等基本无提升甚至下降 |
| **VPN-6class** | **+5.56%** | 同上 |

> ✅ TDDM 是唯一在两个任务上带来显著性能增益的生成方法，证明其生成样本具有高保真度与任务相关性。

#### **(4) 生成质量评估（Table 6）**

| 方法 | MMD ↓ | APD ↑ |
|------|--------|--------|
| **TDDM** | **0.2667** | **2.9633** |
| GAN | 0.3930 | 2.4315 |
| SMOTE | 0.0518 | 3.0356 |

> ⚠️ 尽管 SMOTE 在 MMD 上表现更好，但其生成的是线性插值样本，缺乏语义合理性，实际分类性能反而受损。TDDM 在 MMD 与 APD 之间取得更好平衡。

---

### **消融实验结果**

#### **(1) Melatt 组件消融（Figure 9）**
- 在简单任务（USTC-20class）中，各组件影响较小。
- 在复杂任务（TLS-120class）中：
  - 移除 **CG-LSTM** 或 **cross-attention** 导致 F1 显著下降。
  - 移除任一损失函数（compactness, diversity, entropy）均造成性能退化。
> ✅ 所有组件在高难度任务中均不可或缺。

#### **(2) 记忆模型有效性（Figure 8）**
- Melatt 提取的特征在 **6 种不同分类器**（XGBoost, CatBoost, RF 等）上均优于原始特征和 Autoencoder 特征。
> ✅ Melatt 学到了更具判别性的通用表征。

#### **(3) 推理效率（Table 7）**
| 模型 | 推理延迟 (ms) |
|------|---------------|
| **Melatt + XGBoost** | **<12** |
| NetMamba | >15 |
| ET-BERT / TrafficFormer | >50 |
| Pcap-Encoder | >180 |

> ✅ TDDM-Melatt 具备极高的部署效率，适合实时检测场景。

---

## 4. 关键结论和发现

### **主要发现**

1. **捷径学习是当前 SOTA 模型泛化失败的主因**：一旦移除 IP/Port，几乎所有 SOTA 模型性能崩塌，暴露了“benchmark overfitting”问题。
2. **解耦架构是实现泛化的关键路径**：Melatt 通过冻结编码器 + 外部记忆，迫使模型学习流量的本质行为模式而非表面标识。
3. **通用生成模型不能直接用于流量数据**：TDDM 通过特征选择、类内训练、质心引导等定制化设计，显著优于 GAN 和通用 DDPM。
4. **数据增强并非万能**：当多数类性能饱和时，单纯增加少数类样本收益有限；需结合更优的损失函数与训练策略。

---

### **方法的局限性**

1. **TDDM 的兼容性受限**：实验表明 TDDM 对某些模型（如 YaTC、NetMamba）可能引入噪声并导致性能下降（Appendix C），说明其增强效果依赖于基础模型的表征学习范式。
2. **静态记忆假设**：当前记忆模块为静态预训练，无法自适应网络中新出现的流量模式，缺乏增量学习能力。
3. **生成样本真实性边界**：尽管 MMD/APD 表现良好，但仍需更精细的评估手段判断生成流量是否具备真实的网络行为逻辑。

---

### **未来工作方向**

1. **构建动态可更新的记忆系统**，支持在线学习与增量更新。
2. **开发模型感知的数据增强策略**，使 TDDM 能适配更多类型的 backbone。
3. **探索解释性机制**，可视化模型如何匹配记忆原型，提升可解释性。
4. **扩展至未知流量检测（unknown traffic detection）**，利用记忆重建误差识别异常或新型应用。

--- 

> **总结**：TDDM-Melatt 提出了一种全新的“解耦记忆 + 定制扩散”范式，从根本上应对加密流量分类中的泛化瓶颈与数据不平衡问题，为构建真正可落地的网络安全 AI 系统提供了重要技术路径。

</details>

---

### 14. [RouteSparse: Input-Conditional Pattern Routing for Budgeted Long-Context Prefilling](https://arxiv.org/abs/2608.29058)

**Authors**: Chao Zhang, Yifan Ji, Ziyan Zhang, Kai Song, Fei Lin  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2608.29058v1  

#### Abstract
Dynamic sparse attention can reduce the quadratic cost of long-context prefilling without changing model weights. MInference assigns each attention head one pattern offline and estimates that pattern's sparse indices for every prompt. This design is efficient, but it assumes that a head's preferred ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：RouteSparse: Input-Conditional Pattern Routing for Budgeted Long-Context Prefilling**

---

## 1. **论文的主要贡献和创新点**

### **解决的问题**
长上下文语言模型在 `prefilling` 阶段面临自注意力机制的 **quadratic cost**（计算复杂度为 $O(n^2)$），导致首 token 生成延迟显著增加。虽然已有动态稀疏注意力方法（如 MInference）通过预设每个 attention head 的稀疏模式来加速，但其假设——“每个 head 的最优稀疏模式和预算在不同输入下保持稳定”——在实践中可能不成立，尤其在面对代码、检索、多文档问答等结构差异大的输入时。

### **提出的新方法：RoUTESPARSE**
RoUTESPARSE 是一种 **input-conditional pattern routing** 方法，核心思想是：
- 在推理时，**根据当前输入动态选择** 每个 head 和层的稀疏模式（pattern）和稀疏预算（budget）。
- 引入一个 **低开销的共享 attention probe** 来估计不同候选稀疏掩码的质量。
- 设计 **latency-aware router**，在满足硬件延迟约束的前提下，选择风险最小的 pattern + budget 组合。
- 对于不确定性高的情况，自动 fallback 到更密集的掩码（selective dense fallback），保障质量。

### **相比现有方法的优势**
| 方面 | MInference（基线） | RoUTESPARSE（本文） |
|------|---------------------|------------------------|
| **模式选择** | 固定 per-head，离线确定 | 输入条件化，每 prompt 动态路由 |
| **预算控制** | 固定预算 | 可变预算 + fallback 机制 |
| **鲁棒性** | 对输入变化敏感 | 在 domain/length shift 下更稳健 |
| **质量-延迟权衡** | 更快但质量下降明显 | 略慢但质量更接近 dense attention |

> ✅ **核心优势**：在几乎相同的稀疏速度下，显著提升任务质量，尤其在分布偏移场景中表现更优。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **RULER**：评估长上下文下的检索、追踪与聚合能力，支持可控长度测试。
- **InfiniteBench**：涵盖长文档 QA、摘要、代码、合成检索等任务。
- **PG-19**：长文本语言建模，使用 **perplexity (PPL)** 作为指标。
- **LongBench & L-Eval**：辅助评估集，增强语言、领域和任务多样性。

> 📌 报告各任务类别的独立得分，避免平均分数掩盖特定任务崩溃（如 retrieval collapse）。

---

### **实验设置**
- **模型**：Llama 3.1-8B-Instruct，bfloat16 精度
- **上下文长度**：最高 **128K tokens**
- **硬件平台**：单张 A100 80GB GPU
- **prefill 定义**：从输入 token ID 到首个 token logits 输出的时间

### **评估指标**
| 类型 | 指标 |
|------|------|
| **系统性能** | end-to-end prefill latency（median, 95%）、peak memory、kernel time |
| **模型质量** | 各 benchmark 的官方 score；PG-19 使用 PPL；attention-output relative error（$E_{rel}$） |
| **可靠性** | certificate coverage（基于 omitted probability mass） |

### **基线方法对比**
- **Dense FlashAttention-2**：全注意力，质量上限
- **MInference (fixed)**：固定 per-head 模式分配
- **MInference (larger budget)**：扩大预算以匹配 RoUTESPARSE 延迟
- **Global Router**：每层统一选择一个 pattern
- **Static A-shape**：静态局部+全局窗口

所有方法使用相同模型权重、精度和解码设置。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（128K tokens）**

| Method | Prefill (s) | Speedup | RULER | InfiniteBench | PG-19 PPL |
|--------|-------------|---------|-------|---------------|-----------|
| **Dense FA2** | 43.6 | 1.0× | 84.7 | 47.8 | 8.24 |
| **MInference (fixed)** | 6.0 | 7.3× | 83.1 | 46.5 | 8.39 |
| **MInference (larger)** | 7.4 | 5.9× | 84.0 | 47.0 | 8.32 |
| **RoUTESPARSE (Ours)** | **6.7** | **6.5×** | **84.5** | **47.6** | **8.27** |

> ✅ **核心结果**：
> - 较 MInference 固定方案 **仅慢 0.7 秒**，但 **RULER 提升 1.4 分**（从 83.1 → 84.5）
> - 质量接近 dense attention（仅下降 0.2 分），而 MInference 下降 1.6 分
> - 实现 **6.5× 预填充加速**，兼顾速度与质量

---

### **分布偏移下的鲁棒性（图2）**
| 场景 | MInference 质量损失 | RoUTESPARSE 质量损失 |
|------|----------------------|------------------------|
| In-domain | 1.6 pts | 0.2 pts |
| Domain + Length Shift | 6.4 pts | 2.0 pts |

> ✅ RoUTESPARSE 在分布外场景下优势更明显，验证其 **shift robustness**

---

### **消融实验（Ablation Study）**

| Variant | Prefill (s) | RULER | Worst-decile Δ | Fallback (%) |
|--------|------------|--------|----------------|--------------|
| **Full RoUTESPARSE** | 6.7 | 84.5 | -0.9 | 6.8% |
| No input routing | 6.0 | 83.1 | -4.8 | 0.0% |
| No uncertainty score | 6.3 | 83.8 | -3.1 | 2.1% |
| No dense fallback | 5.9 | 83.9 | -3.7 | 0.0% |
| Predicted FLOPs | 7.5 | 84.5 | -0.9 | 6.8% |
| Per-head launches | 9.2 | 84.5 | -0.9 | 6.8% |

> 🔍 **关键发现**：
> - **No input routing**：质量显著下降 → 验证 H1（条件路由有效）
> - **No fallback**：最差 10% prompts 损失剧增 → 支持 H3（fallback 至关重要）
> - **Predicted FLOPs**：虽质量不变，但延迟增加 0.8s → 说明 **hardware profiling 必不可少**
> - **Per-head launches**：延迟大幅上升 → 强调 **kernel grouping 的重要性**

---

## 4. **关键结论和发现**

### **主要结论**
1. **输入条件化路由优于固定分配**：即使微小的动态调整也能显著提升质量，尤其是在复杂或分布外输入上。
2. **latency-aware routing + selective fallback 是高效设计**：在不确定时 fallback 比统一提高预算更高效。
3. **硬件感知至关重要**：理论 FLOPs 不等于实际延迟，必须依赖实测 profiling。
4. **error certificate 具有实用价值**：基于 omitted probability mass 的 bound 可用于指导 fallback 决策。

> ✅ **三大假设均被验证**：
> - **H1 成立**：条件路由在相同延迟下提升质量
> - **H2 成立**：在 domain/length shift 下增益更大
> - **H3 成立**：selective fallback 比 uniform 扩展更优

---

### **局限性（Limitations）**
- **probe 可能遗漏关键交互**：采样 query/key 可能错过罕见但重要的 attention link。
- **error bound 非分布无关保证**：empirical quantile calibration 无法应对任意分布偏移。
- **bound 是局部的**：不能紧致地绑定最终生成质量。
- **实现依赖性强**：性能受硬件、kernel 实现影响大。
- **维护成本高**：多种 kernel variant 增加部署复杂性。
- **校准成本高**：极端长度下的 dense auditing 昂贵，限制研究规模。

---

### **未来工作方向**
- 探索更鲁棒的 probe 设计，减少遗漏风险
- 发展分布无关的 error guarantee 方法
- 将 routing 思想扩展到 decode 阶段
- 支持更多模型架构与 GPU 架构
- 研究 routing 与其他优化技术（如 KV-cache 压缩）的联合优化

---

> 💡 **一句话总结**：  
> **RoUTESPARSE 证明了“动态选择比固定更好”——通过轻量级 probe + latency-aware routing + selective fallback，在 128K 上实现 6.5× 加速且质量几乎无损，是 post-training 长上下文加速的重要进展。**

</details>

---

### 15. [LoGo: Token-Level Dynamic Local-Global Attention](https://arxiv.org/abs/2608.29539)

**Authors**: Yuqi Pan, Zheng Li, Bohao Tang, Zhen Qin, Guoqi Li  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.29539v1  

#### Abstract
As context lengths scale, attention increasingly becomes a primary computational bottleneck in large language models. Standard Transformers remain powerful but computationally inefficient, as they allocate the same attention budget to every token regardless of its contextual demand. Existing local-g...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《LoGo: Token-Level Dynamic Local-Global Attention》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

随着大语言模型（LLMs）上下文长度（context length）不断增长，**标准 Transformer 的注意力机制**（standard attention）因计算复杂度为 $O(T^2)$ 而成为主要瓶颈。尽管已有 **local-global attention 混合架构**（如 inter-layer 或 intra-layer hybrids）通过混合局部与全局注意力来降低计算成本，但这些方法通常采用**静态分配策略**——即所有 token 共享相同的注意力跨度（attention span），无法根据实际需求动态调整。

这导致两个问题：
- **计算浪费**：局部可预测的 token（如固定短语）仍被赋予全局注意力；
- **资源错配**：真正需要长程依赖的 token 可能得不到足够的全局关注。

### ✅ 提出了什么新方法或新思路

本文提出 **LoGo**（Looking far, Glancing only），一种**token-level 动态局部-全局注意力机制**，其核心思想是：

> 将 **attention span 视为 attention budget 的直接代理**，实现按需分配。

#### 主要创新设计包括：

1. **双分支耦合结构**（Coupled Local/Global Branches）  
   - 所有 token 都执行高效的 **local attention**（窗口大小 $w=128$）；
   - 仅对“需要”长程信息的 token 激活 **global attention** 分支；
   - 两分支共享主参数，通过轻量级变换（如 Linear Projection）解耦，增加约 1% 参数。

2. **学习式门控路由机制**（Learned Gate + Threshold Controller）
   - 每个 token 输出一个标量偏好值 $p_t$，表示其对全局注意力的需求；
   - 引入**自适应阈值**（adaptive threshold）将连续偏好转化为二元决策 $z_t \in \{0,1\}$；
   - 控制器动态调整阈值以维持预设的全局激活比例（如 50%），无需辅助损失函数。

3. **渐进掩码训练策略**（Progressive Masking, P-mask）
   - 初始阶段所有 token 均启用全局注意力，确保梯度充分传播；
   - 随着训练进行逐步引入稀疏路由，提升稳定性。

4. **查询稀疏内核实现**（Query-Sparse Triton Kernels）
   - 仅对选中的 query 执行 global attention 计算，key/value 保持稠密；
   - 显著降低 FLOPs 并转化为实际推理加速。

---

### ✅ 相比现有方法的优势

| 维度 | LoGo | 静态混合（如 inter/intra-layer） | 其他动态方法（如 CoLT5, SMA） |
|------|------|-------------------------------|------------------------------|
| **span 分配粒度** | ✅ Token-level | ❌ Layer/Head-level | ⚠️ Token-level but indirect |
| **是否可学习** | ✅ 端到端学习 | ❌ 预定义规则 | ⚠️ 部分依赖手工设计 |
| **预算控制方式** | ✅ 自适应阈值，无辅助损失 | ✅ 固定比例 | ❌ 复杂平衡机制 |
| **实现效率** | ✅ Query-sparse kernel 加速 | ✅ 简单易集成 | ❌ 实现开销高 |
| **兼容性** | ✅ 可替换任意 hybrid 中的 global 层 | ✅ 成熟架构 | ❌ 架构绑定 |

> **核心优势总结**：LoGo 实现了**细粒度、可学习、可控预算、高效实现**的动态注意力机制，在不牺牲性能的前提下显著优化了长上下文下的性能-计算权衡。

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集

- **预训练语料**：内部构建的类 `Team` 数据集（类似 The Pile）；
- **评估任务**：
  - **语言建模**：WikiText、Lambada；
  - **常识推理**（Zero-shot）：PIQA、HellaSwag、WinoGrande、ARC-e/c；
  - **短上下文回忆**：FDA、SWDE、SQuAD、TriviaQA、NQ、DROP；
  - **长程回忆能力测试**：RULER 套件（含 S-NIAH、MK-NIAH 等 needle-in-a-haystack 任务）。

### ✅ 实验设置

- **模型规模**：从 200M 到 3.3B 参数；
- **上下文长度扩展**：
  - 预训练：8k；
  - 两次扩展：32k 和 128k（各训练 10B tokens）；
- **注意力配置**：
  - Local window: $w = 128$；
  - Target global ratio: $p^{(l)} = 0.5$（即平均一半 token 激活 global attention）；
- **训练细节**：
  - 使用 AdamW，cosine 学习率衰减；
  - Tokens-per-parameter 比例遵循 Chinchilla 原则；
  - Batch size: 0.5M–1M tokens。

### ✅ 基线方法对比

| 基线 | 类型 | 特点 |
|------|------|------|
| **Standard Transformer** | Full attention | 全局注意力，FLOPs 最高 |
| **Inter-layer Hybrid** | Static | 交替使用 local/global 层（如 1:1） |
| **Intra-layer Hybrid** | Static | 每层部分 heads 使用 global attention |
| **LoGo** | Dynamic | 本文提出，token-level 动态选择 |

> 所有方法在 **相同参数量、相同 attention FLOPs** 下比较，确保公平。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### 🔹 表 1：Scaling Behavior（跨尺度表现）

| 参数量 | 模型 | Train Loss ↓ | Wiki PPL ↓ | LMB PPL ↓ | CSR-Avg ↑ |
|--------|------|--------------|------------|-----------|-----------|
| 1.5B | Transformer | 2.162 | 18.241 | 12.025 | 52.88 |
| 1.5B | LoGo | **2.156** | **18.052** | **11.941** | **52.68** |

> ✅ LoGo 在所有尺度上均**匹配或优于**标准 Transformer，表明其具备良好的**可扩展性**。

---

#### 🔹 表 2：语言建模与常识推理（8k 上下文）

| 模型 | Attn Budget | Loss ↓ | Wiki PPL ↓ | LMB PPL ↓ | CSR-Avg ↑ |
|------|-------------|--------|------------|-----------|-----------|
| Transformer | 1.0 | 2.119 | 16.12 | 8.36 | 55.70 |
| inter-layer-H | 0.5 | 2.121 | 16.12 | 8.43 | 56.22 |
| intra-layer-H | 0.5 | 2.122 | 17.78 | 8.48 | 55.76 |
| **LoGo** | **0.5** | **2.112** | **16.04** | **7.50** | **56.30** |

> ✅ LoGo 在**更低 FLOPs**（50% global budget）下取得**最佳语言建模性能**，尤其在 LMB 上大幅领先。

---

#### 🔹 表 4：长程回忆能力（RULER 任务）

| 模型 | RULER-32k Avg ↑ | RULER-128k Avg ↑ |
|------|------------------|-------------------|
| Transformer | 78.9 | 58.1 |
| inter-layer-H | 80.2 | 58.5 |
| intra-layer-H | 75.8 | 47.6 |
| **LoGo** | **83.0** | **65.4** |

> ✅ LoGo 在长序列下优势明显，**RULER-128k 提升达 7.3 个百分点**，验证其卓越的**长程依赖捕捉能力**。

---

#### 🔹 图 2：效率分析（Operator-level Speedup）

- 在 64k 序列长度、50% global budget 下：
  - 相比 Triton 实现的 dense FlashAttention，**速度提升达 1.99×**；
  - 接近理论最优（理想为 2×）；
- 即使在 CUDA baseline 下，高稀疏度时仍有优势。

> ✅ 查询稀疏内核实现在**长序列上的线性加速潜力**。

---

### ✅ 消融实验结果（Ablation Study）

| 变体 | Train Loss ↓ | Wiki PPL ↓ | LMB PPL ↓ | Recall-Avg ↑ |
|------|--------------|------------|-----------|---------------|
| LoGo（完整） | 2.112 | 16.04 | 7.50 | 73.74 |
| Shared params | 2.116 | 16.12 | 7.83 | 69.92 |
| Full params | 2.101 | 15.85 | 7.71 | 73.42 |
| w/o ContextNorm | 2.121 | 16.36 | 8.39 | 70.90 |
| Additive fusion | 2.113 | 16.14 | 7.90 | 72.52 |
| Auxiliary loss | 2.118 | 16.18 | 7.87 | 71.73 |
| w/o P-mask | 2.112 | 15.96 | 7.73 | 71.59 |

> ✅ 关键发现：
> - **ContextNorm 对稳定融合至关重要**；
> - **加权融合优于 additive fusion**；
> - **阈值控制器优于带辅助损失的方法**；
> - **P-mask 提升下游任务表现**；
> - **轻量级解耦（LP）是参数与性能的最佳折衷**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Token-level span allocation 是有效的**  
   LoGo 能够学会根据上下文需求动态分配注意力跨度，避免在局部可预测 token 上浪费全局计算。

2. **保留了 Transformer 的 scaling behavior**  
   从 200M 到 3.3B，LoGo 始终能匹配甚至超越 full-attention Transformer 的性能曲线。

3. **显著提升长程建模能力**  
   在 RULER 等 needle-in-a-haystack 任务上，LoGo 明显优于各类静态混合架构，证明其能精准识别并服务长程依赖 token。

4. **实现了真正的性能-计算双赢**  
   在相同参数和 FLOPs 下，LoGo 不仅语言建模更强，还能通过 query-sparse kernel 实现实际加速。

5. **学习到可解释的分配模式**（见 Figure 3 & 5）
   - 句首、句尾、重复内容、上下文切换处等均有规律性的 global attention 激活；
   - 固定短语后期趋于 local；
   - 新语境切换后 global 使用下降，直到触发长程匹配。

---

### ⚠️ 方法的局限性

1. **KV Cache 仍为稠密**  
   LoGo 未减少 key/value 缓存内存占用，**内存复杂度仍为 $O(T)$**，不适合极端内存受限场景。

2. **依赖 gate 学习质量**  
   若 gate 未能准确判断 span 需求，可能导致关键 token 错过全局注意力。

3. **当前 serving kernel 保守**  
   当前 batched decoding 实现仅在全 microbatch 无激活时跳过 global call，仍有优化空间。

4. **未探索更复杂的 span 形式**  
   当前仅为 binary decision（local vs global），未来可拓展至多级 span 或 soft routing。

---

### 🔮 未来工作方向

1. **结合 memory-saving hybrid 架构**  
   如将 LoGo 替换 Mamba、SSM 或 linear attention 中的 global 层，进一步压缩 memory 与 compute。

2. **开发专用 serving kernel**  
   支持动态 compact decode query rows，提升 batched inference 效率。

3. **探索更精细的 span 控制**  
   如基于 token 类型、位置、语义角色进行分层控制。

4. **应用于 encoder-decoder 或 multimodal 场景**  
   验证 LoGo 在非 decoder-only 架构中的通用性。

5. **理论分析动态 span 的归纳偏置**  
   理解为何 token-level allocation 更有利于 long-range retrieval。

---

## 总结

✅ **LoGo 是一项兼具实用性与创新性的注意力机制改进**：

- 它首次实现了 **token-level、end-to-end learnable、budget-controlled** 的 local-global attention 动态路由；
- 在保持 Transformer 可扩展性的同时，显著提升了长上下文建模效率；
- 通过轻量设计与硬件友好实现，将算法优势转化为实际性能增益；
- 实验充分，分析深入，为未来高效 LLM 设计提供了重要范式。

> **一句话总结**：  
> **LoGo 用“智能调度”的方式让每个 token “该看远时才看远”，在不牺牲性能的前提下，把宝贵的全局注意力花在刀刃上。**

</details>

---

### 16. [Evolutionary Soups: Evolving Mixture-of-Experts for Multi-Objective LLM Alignment](https://arxiv.org/abs/2608.29978)

**Authors**: Lingxiao Kong, Steffen Staab, Cong Yang, Oya Beyan, Zeyd Boukhers  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.29978v1  

#### Abstract
Large language models are increasingly required to generate responses that satisfy multiple competing objectives. Since optimal trade-offs depend on both user preferences and input prompts, controllable multi-objective generation must dynamically adapt models at inference time without retraining. To...

---

### 17. [Verification-Aware Training for Speculative Decoding](https://arxiv.org/abs/2608.30135)

**Authors**: Geonmo Gu, Byeongho Heo, HeeJae Jun, Yoohoon Kang, Sangmin Lee, Sangdoo Yun, Dongyoon Han  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2608.30135v1  

#### Abstract
Speculative decoding accelerates large language model inference by using a draft model to generate candidate tokens, which are verified by the target model in a single forward pass. Verification proceeds sequentially and discards every position from the first rejection onward, yet existing draft tra...

---

### 18. [A Generalized Optimization Engine (GOE) for Edge AI Inference Acceleration](https://arxiv.org/abs/2608.28652)

**Authors**: Venkat R. Dasari, Jakob A. Adams, Vinod K. Mishra, Brian Jalaian  
**Category**: cs.AI  
**Published**: 2026-09-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.28652v1  

#### Abstract
Artificial intelligence (AI) models have demonstrated remarkable capabilities across various domains, yet their widespread deployment is impeded by significant computational costs, particularly on resource-constrained devices. This paper explores the theoretical underpinnings of various AI model opt...

---

### 19. [JudgePanel: A Compact Judge with Panel Deliberation via Adaptive Multi-Reward Reinforcement Learning](https://arxiv.org/abs/2608.29168)

**Authors**: Yiyue Qian, Shinan Zhang, Huan Song, Hannah Marlowe  
**Category**: cs.AI  
**Published**: 2026-09-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.29168v1  

#### Abstract
The LLM-as-a-Judge paradigm has emerged as a scalable alternative to human evaluation. However, single-model judges are limited by their inherent model biases, while multi-agent evaluation protocols that mitigate this through diverse deliberation are prohibitively expensive at inference time. To thi...

---

### 20. [SIC-Agents: Benchmarking and Building an Adaptive Simulator for Pediatric Serious Illness Communication Training](https://arxiv.org/abs/2608.29481)

**Authors**: Zihan Wang, Anita Marie Slominska, Rennie Bimman, Elizabeth Di Flumeri, Amanda Mayappo-Neeposh, Conall Francoeur, Tamara Ellen Carver, Xiao-Wen Chang, Doina Precup, Esin Darici Haritaoglu, Ismail Haritaoglu, Akshatha Arodi, Naomi Goloff  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.29481v1  

#### Abstract
Pediatric serious illness communication (SIC) is critically important, yet scalable communication training for clinicians remains limited. Compared with other dialogue simulation settings, pediatric SIC poses additional challenges, including multi-party interactions, response to parental distress an...

---

### 21. [HiVe: Beyond Static Prompts for Multitask Learning via Hierarchy-based Vertical Mixture-of-Experts](https://arxiv.org/abs/2608.29790)

**Authors**: HyeonJik Bae, Minyeol Kim, Susik Yoon  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.29790v1  

#### Abstract
As large language models (LLMs) continue to scale, parameter-efficient fine-tuning (PEFT) has become a practical alternative to full-parameter adaptation. Prompt tuning is effective, but existing approaches either use flat prompt structures or hierarchical structures with fixed prompt composition, l...

---

### 22. [Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data](https://arxiv.org/abs/2608.29001)

**Authors**: Thiago C\'esar Castilho Almeida, Gustavo Rosseto Let\'icio, Lucas Pascotti Valem, Andr\'e Freitas, Daniel Carlos Guimar\~aes Pedronette  
**Category**: cs.LG  
**Published**: 2026-09-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.29001v1  

#### Abstract
In a data-driven world, efficiently organizing and mapping relationships between objects is crucial. Graphs are powerful tools for modeling these connections, being widely used in social networks, telecommunications, and biology. However, graph-based methods often face high computational costs, part...

---

### 23. [RL-FAT: Reinforcement Learning for Fair Adversarial Training](https://arxiv.org/abs/2608.29247)

**Authors**: Tejaswini Medi, Levan Mikeladze, Margret Keuper  
**Category**: cs.LG  
**Published**: 2026-09-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2608.29247v1  

#### Abstract
Deep neural networks remain highly vulnerable to adversarial perturbations, and adversarial training (AT) has become a widely used approach for improving robustness. However, improvements in average robust accuracy often mask substantial class-wise disparities: while some classes become more robust,...

---

### 24. [Machine Learning-Enhanced Tabu Search for Tactical Wireless Network Design](https://arxiv.org/abs/2608.28627)

**Authors**: Wissem Ahmed Zaid, Alain Hertz, Defeng Liu  
**Category**: cs.AI  
**Published**: 2026-09-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.28627v1  

#### Abstract
Designing high-performance tactical wireless networks under realistic operational constraints gives rise to challenging combinatorial optimization problems, where the evaluation of candidate solutions relies on detailed physical and traffic-aware models. Although classical metaheuristics such as Tab...

---

### 25. [PermitGPT: A Unified Generative-AI Pipeline for Construction Hazard Forecasting, Permit Prediction, and Community Impact](https://arxiv.org/abs/2608.28728)

**Authors**: Mohd Ruhul Ameen, Farjana Aktar, Akif Islam, Momen Khandoker Ope, Abu Saleh Musa Miah, Jungpil Shin  
**Category**: cs.AI  
**Published**: 2026-09-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.28728v1  

#### Abstract
Urban construction governance requires early decisions that connect workplace safety, permitting requirements, and community impact, yet the relevant evidence is often scattered across separate municipal and regulatory data sources. This paper presents PermitGPT, a unified generative artificial inte...

---

### 26. [FRAMEWORKERS: A Dynamic Multi-Agent Framework for AI-Generated Video Production](https://arxiv.org/abs/2608.29814)

**Authors**: Zhendong Li, Lei Sun, Letian Shi, Deheng Zhang, Ruibo Ming, Mengshun Hu, Dannong Xu, Jian Wang, Danda Paudel, Luc Van Gool, Jinjin Gu  
**Category**: cs.AI  
**Published**: 2026-09-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.29814v1  

#### Abstract
Modern video generators excel at synthesizing individual clips, but complete video production requires coordinating a long sequence of interdependent creative steps, including scripting, storyboarding, generation, and editing. It further demands persistent asset management and dynamic task orchestra...

---

### 27. [BLOOM-WILT: Logit Tilting for Behaviour Elicitation in Automated LLM Auditing](https://arxiv.org/abs/2608.31105)

**Authors**: Adrians Skapars, Edoardo Manino  
**Category**: cs.AI  
**Published**: 2026-09-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.31105v1  

#### Abstract
Users of a deployed language model routinely encounter behaviours that testing almost never surfaces, since deployment puts the model through orders of magnitude more interactions than any evaluation can simulate. Automated auditors make testing cheap to scale and flexible enough to cover almost any...

---

### 28. [When Does Bigger Help? A Controlled Study of LLM Scale for Ontology Learning](https://arxiv.org/abs/2608.31118)

**Authors**: Hamed Babaei Giglou, S\"oren Auer, Jennifer D'Souza  
**Category**: cs.AI  
**Published**: 2026-09-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.31118v1  

#### Abstract
The effect of Large Language Model (LLM) scale on ontology learning (OL) performance remains insufficiently characterized. We present a controlled evaluation of 13 models spanning dense and Mixture-of-Experts variants from the Qwen3.5 and Qwen3.6 lineages, together with proprietary GPT release varia...

---

### 29. [COGTRL: Training LLMs for Scientific Discovery Assistance using Cognitive Traces via Reinforcement Learning](https://arxiv.org/abs/2608.30109)

**Authors**: Shrinidhi Kumbhar Santosh Mashetty Divij Handa Kevin Coutinho, Siddharth Sambhaji Ghule, Chitta Baral  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.30109v1  

#### Abstract
Large Language Models (LLMs) trained on extensive scientific research are increasingly integrated as assistants for scientific discovery. However, most research papers omit the fine-grained cognitive process of examining constraints, failed alternatives, and iterative decisions required to achieve t...

---

### 30. [CAST: Critique-Aware Supervision for Training Reliable Long-Horizon Tool-Calling Agents](https://arxiv.org/abs/2608.30147)

**Authors**: Amir Saeidi, Zehua Zhang, Rishitosh Singh, Naman Ahuja, Vivek Gupta, Ali Payani, Gaowen Liu, Jayanth Srinivasa, Chitta Baral  
**Category**: cs.CL  
**Published**: 2026-09-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2608.30147v1  

#### Abstract
Large language model (LLM) agents are increasingly deployed in long-horizon, interactive, and stateful environments. In these settings, a single wrong action, such as refunding the wrong purchase, can cause irreversible task failure and must be intercepted before execution. Such failures may not app...

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

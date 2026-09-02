# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-02 09:58:58 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SFAD: Speculative Factuality-Aware Decoding](https://arxiv.org/abs/2609.00796)

**Authors**: Guanqiao Chen, Di Wang, Lijie Hu  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2609.00796v1  

#### Abstract
As one of the most critical challenges in large language models, contextual faithfulness directly determines their reliability in knowledge-intensive applications. This task is particularly challenging as it requires balancing factual consistency with generation efficiency. Contrastive decoding meth...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SFAD: Speculative Factuality-Aware Decoding 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在知识密集型任务中面临严重的**上下文忠实度（contextual faithfulness）**挑战，即模型倾向于依赖其静态的参数化先验知识而非提供的外部上下文，导致“幻觉”（hallucination）。现有方法存在以下瓶颈：
- **Contrastive decoding** 方法需要双次前向传播（with/without context），计算开销翻倍。
- **Post-training alignment** 方法依赖大规模强化学习，训练成本高昂。

因此，如何在不牺牲推理效率的前提下提升生成的事实一致性，是当前的关键难题。

---

### 提出的新方法与核心思想
作者提出 **SFAD (Speculative Factuality-Aware Decoding)** ——首个将**推测性解码（speculative decoding）**与**事实性增强**相结合的框架，实现高效且可靠的生成。

#### 核心创新点：
1. **构建 ConFide 数据集**
   - 通过原子级分解（atomic fact decomposition）和可控扰动机制（controllable perturbation），生成细粒度、多样化的负样本（如实体替换 `Tent`、数值扭曲 `Tnum`、关系反转 `Trel`）。
   - 构建高质量偏好对用于 **Direct Preference Optimization (DPO)**，训练一个轻量级的**上下文忠实 draft model**。

2. **引入 Epistemic Friction 检测机制**
   - 定义一种新的冲突检测指标：  
     $$
     F_t = \text{JS}(P_M \| P_m) \cdot K_t
     $$
     其中 $K_t$ 是 specialist certainty（专家确定性），用于加权分布张力（distributional tension）。
   - 只有当目标模型与 draft 模型意见分歧大 **且** draft 模型高度自信时才触发修正，避免误纠。

3. **设计 Asymmetric Logit Steering 修正机制**
   - 当 $F_t > T$ 时，采用非对称残差注入方式更新 logits：
     $$
     z^* = z_{M,t} + \lambda \cdot \text{ReLU}(z_{m,t} - z_{M,t}) \cdot \mathbb{I}(v \in V_{\text{cpc}})
     $$
   - 引入 **Contextual Plausibility Mask (CPM)** 确保修正后的 token 在语法上合理，防止语言退化。

4. **动态自适应门控策略**
   - 使用软门控 $\lambda = \sigma(\beta(F - T))$ 控制干预强度，在“快速路径”（Fast Path）与“修正路径”（Steering Path）之间平滑切换。

---

### 相比现有方法的优势
| 维度 | 传统方法 | SFAD |
|------|--------|-------|
| 推理速度 | Contrastive decoding 耗时翻倍 | 保留 speculative decoding 加速效果（达 2.48×） |
| 幻觉抑制能力 | 需要额外 RL 微调或复杂架构修改 | 利用 draft model 作为“事实哨兵”，实时检测并纠正 |
| 训练成本 | 高（需 reward modeling 或 RLHF） | 中等（仅需 DPO 微调 draft model） |
| 分布偏移风险 | Subtractively-based 方法易造成零概率陷阱 | Additive + ReLU 设计保障支持集不变（support preservation） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **基础数据来源**：
  - `LLM-AggreFact` 和 `CG2C`：提供富含多跳推理和知识冲突的事实密集样本。
- **偏好数据集构建**：
  - **ConFide**：由上述源数据经原子分解 + 扰动生成，含约 18K 新样本。
  - 结合已有的 `ConFiQA`（18K 样本），共约 36K DPO 训练样本。

### 实验设置
- **模型配置**：
  - **Target Model**: Qwen3-14B
  - **Draft Model**: Qwen3-1.7B（经 DPO 微调）
- **评估任务类别**：
  1. **基础问答（Foundation QA）**：HotpotQA, PopQA, TriviaQA
  2. **摘要生成（Summarization）**：XSum, TofuEval
  3. **长文本问答（Long-Form QA）**：CLAPNQ, ExpertQA, HAGRID
  4. **知识冲突分析**：LLM-AggreFact held-out 测试集
  5. **通用能力测试**：GSM8K, Just-Eval

### 评估指标
| 类别 | 指标 |
|------|------|
| 准确率 | EM (Exact Match), ROUGE-L, BERT-P |
| 忠实度 | FaithScore（基于 MiniCheck）、AlignScore |
| 上下文依赖性 | Context-faithful Frequency ($P_c$), Memory Reliance ($MR$) |
| 效率 | ATGA（Average Token Generation Acceleration）= 无 SFAD 时间 / 有 SFAD 时间 |

### 基线方法对比
- **Decoding-level 方法（应用于 Qwen3-14B）**：
  - Greedy Decoding
  - CAD (Context-Aware Decoding)
  - AdaCAD
  - COIECD
- **前沿大模型参考**：
  - Llama-3.1-70B-Instruct（参数量为 Qwen3-14B 的 5 倍）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 表1：Foundation QA 性能对比（部分）
| Method | TriviaQA (EM) | HotpotQA (EM) | PopQA (EM) | Rel. Latency |
|--------|----------------|----------------|-------------|---------------|
| Vanilla Qwen3-14B | 53.87 | 41.77 | 78.21 | 1.00× |
| CAD | 41.43 | 39.51 | 71.29 | 2.00× |
| AdaCAD | 82.11 | 45.63 | 77.39 | 2.15× |
| COIECD | 83.07 | 45.63 | 76.29 | 2.40× |
| Llama-3.1-70B | **90.20** | **56.11** | **86.11** | 4.85× |
| **SFAD (Ours)** | **85.12** | **52.19** | **86.39** | **0.82×** |

> ✅ SFAD 在三项指标均显著优于所有 decoding-level 方法，并接近 5× 更大的 Llama-70B 模型，同时速度快近 **2.48×**

#### 表2：摘要任务表现（XSum & TofuEval）
| Method | XSum (R-L) | XSum (BERT-P) | TofuEval (AlignScore) | Rel. Latency |
|--------|------------|----------------|------------------------|--------------|
| Vanilla | 13.67 | 91.67 | 59.84 | 1.00× |
| CAD | 14.59 | 93.65 | 83.23 | 2.00× |
| AdaCAD | 14.91 | 94.29 | 85.07 | 2.20× |
| Llama-3.1-70B | 16.35 | 94.12 | 87.31 | 5.10× |
| **SFAD (Ours)** | **16.32** | **93.97** | **87.53** | **0.85×** |

> ✅ 在保持更高效率的同时，达到甚至超越 70B 模型的表现

#### 表3：长文本 QA 表现（FaithScore）
| Method | CLAPNQ (Faith) | ExpertQA (Faith) | HAGRID (Faith) | Rel. Latency |
|--------|------------------|-------------------|------------------|---------------|
| Vanilla | 59.73 | 51.29 | 57.63 | 1.00× |
| COIECD | 61.96 | 56.32 | 59.76 | 2.42× |
| Llama-3.1-70B | 92.45 | 72.40 | 82.20 | 5.30× |
| **SFAD (Ours)** | **90.93** | **71.13** | **81.99** | **0.78×** |

> ✅ 在极端长文本场景下仍保持高忠实度，逼近 70B 模型水平

---

### 消融实验结果

#### (1) ConFide + DPO 的有效性（图3）
- **ConFide + DPO** 显著优于：
  - ConFide + SFT（记忆依赖 MR 高）
  - ConFiQA + DPO（缺乏细粒度扰动信号）
- 说明：**原子级扰动生成 + DPO 优化** 对培养 draft model 的上下文忠诚至关重要。

#### (2) Logit Steering 影响分析（表4）
| Strategy | Prob(%) of Faithful Tokens | Relative Gain |
|---------|----------------------------|---------------|
| Original Target | 18.73% | 1.0× |
| Standard SD | 18.91% | 1.01× |
| **SFAD (Ours)** | **62.45%** | **3.33×** |

> 🔥 证明：**logit-level steering 能从根本上重塑输出分布**，而不仅仅是接受/拒绝 token。

#### (3) Epistemic Friction 分析（图5）
- 单纯使用 JS divergence 会因风格差异频繁误触发。
- 加入 specialist certainty $K_t$ 后有效过滤低置信扰动，仅在真正“自信幻觉”处激活（红色方块位置）。

#### (4) 摩擦阈值 $T$ 敏感性分析（附录 A.2）
| Threshold $T$ | Steering Ratio | Faithfulness | Speedup (ATGA) |
|---------------|----------------|-------------|----------------|
| 0.1 (Aggressive) | 48.2% | 86.7 | 2.12× |
| **0.5 (Default)** | **22.4%** | **85.2** | **2.48×** |
| 0.9 (Conservative) | 4.5% | 41.8 | 2.82× |

> ✅ 默认 $T=0.5$ 实现最佳权衡：以最小干预获得最大忠实度增益。

---

## 4. 关键结论和发现

### 主要发现
1. **SFAD 成功实现了“双赢”**：
   - 在多个知识密集型任务上显著提升 **faithfulness**（最高达 85.2 分）。
   - 同时实现 **2.48× 的推理加速**，优于所有对比的 decoding 方法。

2. **draft model 可作为“事实哨兵”**：
   - 经过 ConFide + DPO 训练的小型 draft model 能有效识别知识冲突。
   - 其预测可安全地用于指导更大 target model 的生成。

3. **Epistemic Friction 是高效的触发器**：
   - 融合分布张力与专家确定性的设计，能精准定位需干预的位置，避免过度修正。

4. **Asymmetric Logit Steering 优于传统融合方式**：
   - 消融显示 Linear Sum、Interpolation、Subtractive Contrast 均会导致 fluency 下降。
   - SFAD 的 ReLU-based 注入机制在提升事实性的同时维持语言流畅。

5. **通用性良好**：
   - 在 Llama-3.1-8B 上复现实验也取得一致收益（见附录 B），表明方法具有跨模型家族泛化能力。

---

### 局限性
1. **依赖高质量的 draft model 训练**：
   - 需预先构建 domain-aligned 的 ConFide 类型数据集，增加前期投入。
2. **超参敏感性**：
   - 摩擦阈值 $T$ 和 sharpening coefficient $\gamma$ 在跨领域应用时可能需要重新调优。
3. **未完全消除幻觉**：
   - 尽管大幅降低，但在极复杂或多模态场景中仍有残留风险。

---

### 未来工作方向
1. **自动化 ConFide 构建流程**：
   - 探索 self-instruct 或 active learning 方式减少人工标注负担。
2. **扩展至多模态 LLMs**：
   - 将 SFAD 应用于视觉-语言模型中的跨模态幻觉检测。
3. **在线自适应调整机制**：
   - 动态估计最优 $T$ 和 $\lambda$，提升跨域鲁棒性。
4. **结合 retrieval-augmented generation (RAG)**：
   - 将 SFAD 与 RAG 系统集成，形成端到端可信生成管道。

---

> 📌 **总结一句话**：  
> **SFAD 是首个将 speculative decoding 用于幻觉缓解的工作，通过“忠实 draft 模型 + epistemic friction 检测 + asymmetric logit steering”的闭环机制，在几乎不增加延迟的情况下大幅提升 LLM 的上下文忠实度，兼具理论严谨性与工程实用性。**

</details>

---

### 2. [IMPACT: Attention Is the Interaction Map for Scalable Interaction-Aware World Model Training](https://arxiv.org/abs/2609.00161)

**Authors**: Rongze Tang, Jianjie Fang, Zhaolu Wang, Ziyou Wang, Xvyuan Liu, Haisheng Su, Xin Zhang, Wei Wu, Chen Gao, Yong Li, Zhibo Chen  
**Category**: cs.AI  
**Published**: 2026-09-02  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.00161v1  

#### Abstract
World models have made remarkable progress in action-conditioned future prediction for embodied agents, yet still struggle to model physically plausible interactions. Existing approaches address this limitation by constraining the generation process with external representations encoding motion, geo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：IMPACT: Attention Is the Interaction Map for Scalable Interaction-Aware World Model Training**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **World Model** 在动作条件下的未来预测中取得了显著进展，但在建模**物理上合理的交互行为**（physically plausible interactions）方面仍存在不足。  
传统方法依赖于外部密集表示（如 **optical flow**, **depth maps**, **3D reconstruction**）来约束生成过程，以增强交互区域的建模能力。然而，这些表示通常需要额外的估计器或人工标注，导致**训练成本高、扩展性差**。

此外，作者指出一个被忽视的关键问题：**监督分配不匹配（supervision-allocation mismatch）**。在标准的全局平均 **MSE** 损失下，静态背景等大面积区域主导了优化信号，而稀疏但关键的动态交互区域（如手-物接触、抓取动作）则被**欠监督**，导致模型无法充分学习交互动力学。

---

### **提出的新方法：IMPACT**
为解决上述问题，作者提出了 **IMPACT**（**I**nteraction-aware **M**odel training framework with **P**rior-guided **A**ttention **C**alibration and **T**argeting），其核心思想是：

> **利用模型自身内部的 cross-attention 作为交互先验，构建“交互图谱”（interaction map），并据此重新加权 denoising 监督信号，使训练更关注关键交互区域。**

#### **核心创新点：**
- **无需外部表示**：完全基于模型前向传播中的 **cross-attention** 机制，提取与操作对象相关的注意力分布，作为交互区域的**内部先验**。
- **Attention Distribution Sampling (ADS)**：
  - 从 manipulated-object 对应的文本 token 的 cross-attention 中聚合出空间分布。
  - 采样多个候选区域，并用**去梯度的局部预测误差**（detached local prediction error）进行校准，形成最终的 **interaction map**。
- **Interaction-Weighted Supervision (IWS)**：
  - 使用 interaction map 对 denoising loss 进行加权，提升交互区域的监督强度。
  - 通过**梯度解耦**（gradient routing）防止 cross-attention 自身因优化目标而坍缩（collapse）——即 cross-attention 参数仍由原始全局 MSE 优化，其余参数由加权后的目标优化。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | IMPACT |
|------|--------|--------|
| **是否需要外部表示** | 是（如光流、深度图） | 否（仅用模型内部 attention） |
| **训练可扩展性** | 受限于外部标注/估计成本 | 高（随数据规模自然扩展） |
| **推理开销** | 可能增加 | 无（仅训练时使用，推理不变） |
| **监督效率** | 均匀分配，静态区域主导 | 动态加权，聚焦交互区域 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **RoboTwin** (~350K 视频片段)
   - 机器人双臂操作任务，包含 RGB 图像、语言指令、14-DoF 动作轨迹。
   - 用于 **robot-arm manipulation** 实验。
2. **EgoDex** (~256K 视频片段)
   - 第一人称视角的人手精细操作数据集，包含手部姿态、语言指令。
   - 用于 **human-hand manipulation** 实验。

---

### **实验设置**
- **模型架构**：基于 **DiT**（Diffusion transformer）的 latent video diffusion model，采用 **flow matching** 训练。
- **输入条件**：语言指令 `y`、参考图像 `x_ref`、控制信号 `a`（如机器人动作或手部姿态）。
- **实现细节**：
  - 分辨率：720p
  - 批大小：global batch size = 32（8 GPU × 1 × 4 step accum）
  - 学习率：2e-5，bf16 混合精度，FSDP 并行
  - 训练周期：1 epoch
  - IMPACT 超参固定：`K=8`, `γ=5`, `β=0.7`, `κ=0.65`

---

### **评估指标**

#### **Robot-arm manipulation (WorldArena)**
- **综合指标**：**EWMScore**（16 项归一化指标的平均值，越高越好）
- **维度分解**：
  - **Visual Quality**（图像质量）
  - **Motion Quality**（运动质量）
  - **Content Consistency**（内容一致性）
  - **Physics Adherence**（物理合理性）
  - **3D Accuracy**（3D 准确性）
  - **Controllability**（可控性）

#### **Human-hand manipulation (EgoDex)**
- **视觉质量**：
  - **FVD**（Fréchet Video Distance，越低越好）
  - **FID**（Fréchet Inception Distance，越低越好）
- **交互质量**：
  - **CLIP-Hand**（手与物体的语义对齐，越高越好）
  - **Hand IoU**（手部位置重叠度，越高越好）

---

### **基线方法对比**
#### **Robot-arm 基线**
- **通用世界模型**：CogVideoX, Veo 3.1, Wan 2.6
- **具身世界模型**：GigaWorld-0, Genie Envisioner, IRASim, CtrlWorld
- **表示引导模型**：TesserAct, RoboMaster, WoW
- **骨干模型对比组**：
  - Wan 2.2-AC（MSE）
  - Cosmos-Predict 2.5 (action)

#### **Human-hand 基线**
- **通用视频生成**：HunyuanVideo-1.5, Cosmos-Predict 2.5
- **姿态控制模型**：MimicMotion, MagicPose, VACE, LOME

---

## **3. 主要实验结果和性能指标**

### **Robot-arm manipulation 结果（Table 1）**
| 模型 | EWMScore | Physics Adherence | 3D Accuracy | Controllability |
|------|----------|-------------------|-------------|-----------------|
| Wan 2.2-AC | 58.65 | 53.41 | 86.16 | 54.40 |
| **+ IMPACT** | **62.46 ↑3.81** | **55.87** | **92.56** | **60.00** |
| Cosmos-Predict 2.5 (action) | 55.91 | 42.23 | 82.53 | 49.51 |
| **+ IMPACT** | **62.53 ↑6.62** | 41.82 | 77.99 | **71.19** |

- **IMPACT 在两个 backbone 上均显著超越基线**，分别提升 **+3.81** 和 **+6.62** EWMScore。
- 在 **Wan 2.2-AC** 上取得最佳 **Physics Adherence (55.87)** 和 **3D Accuracy (92.56)**。
- 在 **Cosmos** 上实现最高 **Controllability (71.19)**。

> ✅ **IMPACT 超过了 Wan 2.6、CtrlWorld 和 WoW 等强基线，验证了其有效性。**

---

### **Human-hand manipulation 结果（Table 2）**
| 模型 | FVD ↓ | FID ↓ | CLIP-Hand ↑ | Hand IoU ↑ |
|------|-------|-------|--------------|------------|
| Wan 2.2-AC | 366.12 | 44.71 | 0.921 | 0.693 |
| **+ IMPACT** | **110.94 ↓70%** | **5.79 ↓87%** | **0.952** | **0.772** |

- **FVD 下降 70%**，**FID 下降 87%**，表明生成视频的**时间连贯性和帧质量大幅提升**。
- **CLIP-Hand 和 Hand IoU 显著提升**，说明手-物交互的**语义和空间定位更准确**。

> ✅ **IMPACT 在人手操作任务中实现了压倒性优势，尤其在视觉保真度方面。**

---

### **消融实验（Ablation Study, Table 3）**
| 方法 | EWMScore | Visual Quality | Motion Quality | Physics Adherence |
|------|---------|----------------|----------------|--------------------|
| Wan 2.2-AC | 58.65 | 56.57 | 48.23 | 53.41 |
| + IWS | 61.54 | 60.44 | 49.16 | 55.16 |
| + IWS + ADS (**IMPACT**) | **62.46** | **60.60** | **53.28** | **55.87** |

- **IWS 单独带来 +2.89 提升**，说明加权监督本身有效。
- **ADS 进一步提升 +0.92**，证明通过预测误差校准 attention prior 能更精准定位交互区域。
- 二者协同作用，共同提升交互建模能力。

---

## **4. 关键结论和发现**

### **主要发现**
1. **监督分配不匹配是制约交互建模的关键瓶颈**：全局 MSE 导致静态区域主导训练，动态交互区域被忽略。
2. **Cross-attention 可作为有效的交互先验**：语言指令中 manipulated-object token 的 attention 分布天然指向可能发生交互的区域。
3. **IMPACT 无需外部表示即可实现更优的交互建模**：通过 ADS + IWS 构建 interaction map 并重加权监督，在多种场景下一致优于基线。
4. **方法具有良好的泛化性**：在 **robot-arm** 和 **human-hand** 两种截然不同的交互模式、不同 backbone（Wan vs Cosmos）、不同控制信号下均表现优异。

---

### **局限性**
- **依赖语言指令的质量**：若指令未明确提及操作对象（如“把它拿起来”），object-token grounding 可能失败。
- **attention prior 可能模糊**：初始 cross-attention 分布可能覆盖过大区域（如整个机械臂），需依赖 ADS 进行校准。
- **目前仅用于训练阶段**：虽不影响推理，但无法在推理时动态调整关注区域。

---

### **未来工作方向**
- 将 IMPACT 思想推广到**无语言指令**的 setting（如纯视觉或动作驱动）。
- 探索将 interaction map 用于**推理时的注意力引导**，实现动态聚焦。
- 扩展至更复杂的多智能体交互或长期任务规划场景。
- 结合强化学习框架，利用 interaction-aware 生成进行更高效的策略学习。

---

> 🔚 **总结**：  
> **IMPACT 提出了一种简洁而强大的训练范式，揭示了 attention 本身即是 interaction map 的潜力。它摆脱了对外部表示的依赖，实现了高效、可扩展的交互感知世界模型训练，在机器人与人机交互两大领域均展现出卓越性能。**

</details>

---

### 3. [PCoMoE: Shifting MoE Inference from Monolithic Expert Selection to Fine-Grained Path Composition](https://arxiv.org/abs/2609.01024)

**Authors**: Ziyan Gan, Fangxin Liu, Chenyang Guan, Junjie Wang, Ning Yang, Haomin Li, Xiang Li, Siran Yang, Jiamang Wang, Lin Qu, Zongwu Wang, Li Jiang, Haibing Guan  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.01024v1  

#### Abstract
Mixture-of-Experts (MoE) architectures scale Large Language Model (LLM) capacity efficiently by activating a sparse subset of experts per token. However, modern MoE inference remains heavily constrained by the rigid, whole-expert abstraction. Existing frameworks manage, schedule, or prune experts as...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PCoMoE: Shifting MoE Inference from Monolithic Expert Selection to Fine-Grained Path Composition

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Mixture-of-Experts (MoE)** 推理框架受限于“**整体专家选择**”（monolithic expert selection）的粗粒度抽象。尽管 MoE 能通过稀疏激活提升模型容量，但当前优化方法将每个 expert 视为不可分割的原子执行单元，导致以下问题：
- 无法挖掘 **intra-expert 内部的计算冗余**；
- 强制在“完全执行”或“完全跳过”之间二选一，造成效率与精度的次优权衡；
- 动态路由带来的不规则计算模式加剧系统开销。

### 提出的新方法：PCoMoE
本文提出 **PCoMoE**（Path-Compositional MoE），一种从“整体专家选择”转向“细粒度路径组合”的新型 MoE 推理执行框架，其核心思想是打破专家边界，进行 **sub-transformation 级别的重组与复用**。

#### 创新点
1. **Path-Level Composition（路径级组合）**  
   将传统 SwiGLU 风格的 expert 分解为两个可独立调度的子模块：
   - **Expansion-side**：包含 `gate` 和 `up` 投影，占总 FLOPs 的约 2/3；
   - **Projection-side**：`down` 投影。
   由此构建 $n \times n$ 的虚拟路径空间（$n$ 为专家数），允许跨专家组合 expansion 与 projection 模块，形成新的执行轨迹。

2. **Compatibility-Aware Gating（兼容性感知门控）**  
   设计了一种带学习偏置的路径评分机制：
   $$
   s_{i,j}(h) = g_\text{base,i}(h) + b_{i,j} + \lambda g_\text{tgt,j}(h)
   $$
   其中 $b_{i,j}$ 是可学习的 **compatibility bias**，用于抑制低价值的 off-diagonal 路径（即跨专家组合），防止组合爆炸并保证表征保真度。

3. **Hardware-Efficient Runtime（硬件友好运行时）**  
   实现 **source-grouped compute reuse**：对共享相同 expansion-side 的路径进行分组，仅执行一次 `gate+up` 计算，实现确定性加速；同时通过离线缓存 active mask 和 dispatch layout 来最小化控制流开销。

### 相比现有方法的优势
| 维度 | 传统方法（如 MoE-Pruner, MoEITS） | PCoMoE |
|------|-------------------------------|--------|
| 优化粒度 | Inter-expert（专家间） | **Intra-expert + Inter-expert**（路径级） |
| 执行单位 | 完整 expert | 可组合的 sub-modules |
| 冗余消除能力 | 跳过整个 expert | 复用高成本模块（expansion side） |
| 系统开销 | 高动态调度复杂度 | 严格限制 overhead，支持 source grouping |
| 精度保持 | 易因跳过专家而下降 | 通过 diagonal fallback 和 compatibility 控制 |

---

## 2. 核心实验方法和设置

### 使用的数据集
下游任务采用五个标准零样本评测基准：
- **BoolQ**：自然语言的是/否问答
- **ARC-Easy (ARC-E)** 和 **ARC-Challenge (ARC-C)**：科学推理挑战
- **HellaSwag**：常识填空
- **WinoGrande**：代词消解

综合指标为五项任务的 **macro-average 准确率**。

### 实验设置
- **模型**：在三种不同路由配置的 MoE 模型上验证：
  - **Qwen1.5-MoE-A2.7B**：top-4，60 个专家
  - **Mixtral-8x7B-v0.1**：top-2，8 个专家
  - **DeepSeek-V2-Lite**：top-6，64 个专家
- **硬件平台**：单张 NVIDIA H20 GPU + Intel Xeon CPU
- **训练细节**：
  - 使用 Alpaca + SQuAD 的混合数据集（25K 样本）
  - 采用 LoRA 微调 gate 层（rank=16, batch_size=32）
  - 冻结所有 expert 参数
  - 学习 compatibility bias 并逐步剪枝低价值路径

### 评估指标
- **准确性**：各任务准确率及平均得分
- **推理速度**：
  - **Token throughput (tokens/s)**：prefill、decode、end-to-end 各阶段吞吐量
  - **Speedup**：相对于 vanilla MoE 的加速比
- **消融研究**：分析各组件贡献

### 基线方法对比
- **Vanilla / Vanilla-FT**：原始 MoE 与微调版本
- **MoE-I²**（Yang et al., 2024）：基于专家间剪枝与低秩分解
- **MoE-Pruner**（Xie et al., 2024）：利用 router hint 进行专家剪枝
- **MoEITS**（Balderas et al., 2026）：绿色 AI 导向的 MoE 简化方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 准确性提升（Zero-shot Accuracy）
| Model | 方法 | Avg. Acc (%) | 相对 Vanilla-FT 提升 |
|-------|------|---------------|---------------------|
| Qwen1.5-MoE | PCoMoE | **73.70** | **+2.87 pts** |
| Mixtral-8x7B | PCoMoE | **79.97** | **+1.61 pts** |
| DeepSeek-V2-Lite | PCoMoE | **72.48** | **+2.14 pts** |

> 在 Qwen 上相比 vanilla 提升高达 **+5.8 pts**，说明 off-diagonal 路径能引入有效且高质量的计算路径。

#### ✅ 推理加速（Throughput & Speedup）
| 模型 | 阶段 | Vanilla (t/s) | PCoMoE (t/s) | 加速比 |
|------|------|-------------|--------------|--------|
| Mixtral-8x7B | Decode | 26.50 | **34.58** | **1.305×** |
| Mixtral-8x7B | End-to-End | 133.43 | **172.72** | **1.31×** |
| Qwen1.5-MoE | End-to-End | — | — | **>1.2×** |
| DeepSeek-V2-Lite | End-to-End | — | — | **稳定增益（top-6 场景下）** |

> 图9显示 PCoMoE 在 decode 阶段带来显著加速，**最高达 1.31× end-to-end 推理加速**。

#### 🔍 消融实验结果

##### （1）设计消融（Table 2）
| 方法 | Avg Acc (%) | Decode Speedup |
|------|-------------|----------------|
| Vanilla | 67.90 | 1.000× |
| Vanilla-FT | 70.83 | 1.009× |
| Router-Only FT | 67.51 | 1.030× |
| Frozen-Bias（无学习兼容性） | 55.50 | 1.286× |
| **PCoMoE（完整）** | **73.70** | **1.238×** |

> 结论：**learned compatibility bias 至关重要**，否则虽快但严重损害精度。

##### （2）执行优化消融（Table 3）
| 配置 | Speedup | 说明 |
|------|---------|------|
| Base PCoMoE | 1.00× | 仅有路径构造 |
| +Routing | 1.15× | 引入路径选择 |
| +Fusion | 1.28× | 融合过滤与 top-k |
| +Dispatch | 1.45× | source-grouped dispatch |
| **All Optimizations** | **1.73×** | 综合优化累计加速 |

> 表明 **hardware-aware 调度设计是实现实际加速的关键**。

---

## 4. 关键结论和发现

### 主要发现
1. **打破 monolithic expert 抽象可释放巨大优化潜力**：intra-expert 的结构性不对称（如 expansion/projection 成本差异）提供了高效的复用机会。
2. **路径组合优于简单跳过**：通过合成 diagonal + off-diagonal 路径，在不增加参数的前提下实现更灵活的效率-质量权衡。
3. **compatibility-aware gating 可控地扩展搜索空间**：避免组合爆炸的同时保留高价值路径。
4. **source-grouped execution 将算法灵活性转化为真实加速**：硬件协同设计至关重要。

### 方法的局限性
1. **依赖 SwiGLU 架构假设**：当前分解基于典型的 gate-up-down 结构，其他 expert 内部结构需重新定义 operator boundaries。
2. **主要优化 autoregressive decoding**：prefill 阶段未受益，因其依赖静态 masking 和 grouped compute。
3. **offline calibration 开销**：虽然推理无额外负担，但 fine-tuning + progressive pruning 增加了部署前准备成本。

### 未来工作方向
1. **扩展至更多 expert 架构**：适配非 SwiGLU 或更复杂的 FFN 结构。
2. **集成分布式与 prefill 优化**：结合 pipeline parallelism、expert offloading 等技术实现全栈加速。
3. **自动化编译流程**：将 compatibility learning 与 pruning 整合进端到端模型编译器，降低部署门槛。
4. **探索动态路径选择机制**：在推理时根据输入自适应调整路径密度。

---

> 💡 **一句话总结**：  
> PCoMoE 通过将 MoE 推理从“选专家”升级为“拼路径”，实现了 **1.31× 推理加速 + 最高 10% 精度提升**，揭示了 **intra-expert 结构复用** 是下一代高效 MoE 系统的关键突破口。  
> 代码开源地址：[https://github.com/gzyyy0/PCoMoE](https://github.com/gzyyy0/PCoMoE)

</details>

---

### 4. [AInfer-PD: Communication-Safe In-Place Prefill-Decode Multiplexing for Distributed MoE Rollouts](https://arxiv.org/abs/2609.00993)

**Authors**: Guowei Wang, Chaokun Yang, Zhenxuan Pan, Yuhong Guo, Minghua Zhu, Zhechuan Zhang, Shuo Wan, Xiaowei Zhu  
**Category**: cs.DC  
**Published**: 2026-09-02  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.00993v1  

#### Abstract
Rollout inference often dominates the wall-clock time of large-scale reinforcement learning (RL). In agentic RL, each trajectory alternates between model generation and environment interaction over multiple turns. Asynchronous trajectories consequently introduce new prefill (P) work while other traj...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AINFER-PD: Communication-Safe In-Place Prefill-Decode Multiplexing for Distributed MoE Rollouts

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在基于 **Reinforcement Learning (RL)** 的 **agentic rollout** 推理中，每个轨迹（trajectory）会交替进行 **prefill (P)** 和 **decode (D)** 阶段。由于不同轨迹的环境响应延迟、生成长度不一致，导致 P 和 D 在时间上持续共存（persistent P/D coexistence）。当两者共享同一组加速器时，长而可变的 P 批次会干扰对延迟敏感的 D 阶段，造成通信竞争和性能下降。

此外，在大规模 **MoE (Mixture-of-Experts)** 模型部署中，常见的并行策略（如 **ADP/ATP** 和 **DeepEP**）存在以下两个关键问题：
- **通信死锁风险**：P 和 D 路径可能使用相交的 collective 通信组（如 AllReduce vs ReduceScatter/AllGather），若跨 rank 的 enqueue 顺序不一致，会导致分布式死锁（progress cycle）。
- **状态共享冲突**：DeepEP 的 normal-P 和 low-latency-D 路径共享 mutable 协议状态，无法安全并发执行。

### 🚀 提出的新方法：AINFER-PD
AINFER-PD 是一种支持 **通信安全的原位 P/D 多路复用（in-place P/D multiplexing）** 的系统设计，专为分布式 MoE rollout 引擎优化。其核心创新包括：

#### （1）跨 rank 的 collective 通信排序机制（Cross-Rank Collective Ordering）
- 引入 **rank-aligned segment turnstile** 同步机制，在每个 turnstile 阶段强制所有 rank 先统一 enqueue D 操作，再允许 P 进入下一个 segment。
- 对于相交的 collective（如 P 的 full-TP AllReduce 与 D 的 DP-attention RS+AG），通过将 P 的 collective 显式置于 D 之后，打破潜在的全局等待环。
- 支持 **fine-grained segmentation**，使 D 可以在长 P 执行过程中多次插入，而非等待整个 P 完成。

#### （2）DeepEP 路径的状态隔离（Phase-Owned Communication State）
- 将 DeepEP 的 mutable 状态（buffers, counters, workspaces, events, QP ranges）按 phase 划分：
  - **P 使用 normal-path state**
  - **D 使用 low-latency-path state**
- 在一个进程内实现两条路径的并发执行，同时保留共享的模型权重、KV Cache 和底层 runtime。
- 在 prefill 中引入 **safe boundary**（如 expert notification 后拆分 dispatch），使得 D 可提前推进而不受阻塞。

#### （3）端到端集成于 MoE rollout 引擎
- 与调度器协同，动态控制 P segment 的 admission 策略。
- 支持多种并行拓扑（ADP/ATP/EP）、精度（BF16/FP8）、MTP 模式等。

### 🔍 相比现有方法的优势
| 方法 | 是否共享设备 | 是否避免 KV Transfer | 是否解决通信死锁 | 是否支持 DeepEP 并发 |
|------|---------------|------------------------|--------------------|------------------------|
| **P/D Disaggregation** [15,16,21] | ❌ 分离池 | ❌ 需要 KV transfer | ✅ 是 | ✅ 是 |
| **传统 In-Place Multiplexing** [1,2,5,10,11] | ✅ 是 | ✅ 是 | ❌ 否 | ❌ 否 |
| **AINFER-PD（本文）** | ✅ 是 | ✅ 是 | ✅ 是 | ✅ 是 |

> ✅ **AINFER-PD 在保持设备、模型、KV 共享的前提下，首次实现了通信安全且支持 DeepEP 并发的 in-place P/D multiplexing。**

---

## 2. 核心实验方法和设置

### 📊 数据集与工作负载
- 使用 **匿名内部 RL 轨迹数据集**，包含 128 条多轮对话轨迹，共 1,265 个请求。
- 每条轨迹具有因果依赖关系，模拟真实 agent 行为：生成 → 等待环境响应 → 续写 prefill → decode。
- 请求分布特征（见 Figure 6）：
  - 新增 prompt tokens：median=292, p99=3,062
  - 生成 tokens：median=88, p99=1,057
  - 上下文深度：median=4,716, p99=12,507
  - 客户端等待时间（client wait）：median=3.1s, p99=15.3s

### ⚙️ 实验设置
- **硬件平台**：
  - 单节点：8× NVIDIA H20-3E GPUs
  - 双节点：16× H20-3E GPUs（跨节点验证）
- **软件栈**：
  - PyTorch 2.8, CUDA 13.0, NCCL 2.27.7
  - DeepEP 1.2.1, SGLang 0.5.15
- **模型配置**：
  - 内部 MoE 模型：42 层，hidden size=2,560，512 个 routed experts，top-8 routing
  - 支持 BF16 和 FP8 精度
- **并行拓扑（见 Table 2）**：
  - H1–H4：非 DeepEP 后端，测试 ADP/ATP 影响
  - Crossed：EP8/ADP2（ATP4），用于触发通信冲突
  - Two-node：EP16/ADP8（ATP2），验证扩展性

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Makespan (E2E Time)** | 固定 workload 下完成全部 1,265 请求的时间（主指标） |
| **Completed Requests/s** | 吞吐率，与 makespan 成倒数关系 |
| **p99 TTFT (Time to First Token)** | 第一个 token 输出延迟的 p99 |
| **Mean D Wait** | decode 请求平均等待调度的时间 |
| **Ov. (%)** | P/D GPU 工作并发比例 |

### 🆚 基线方法
| 基线 | 描述 |
|------|------|
| **AInfer Normal** | 同引擎但禁用 P/D multiplexing，P 有绝对优先级 |
| **SGLang** | 外部端到端推理系统，作为独立 baseline |
| **Global-Complete / Global-Enqueue / Fine-grained** | 消融实验中的不同 ordering 控制 |

---

## 3. 主要实验结果和性能指标

### 📉 性能提升（vs 基线）

#### （1）单节点性能（Figure 7）
| 对比项 | Makespan 减少幅度 |
|--------|------------------|
| vs AInfer Normal（禁用 multiplexing） | **7.1% – 22.5%** |
| vs SGLang | **24.8% – 32.9%** |

> 在高 P 压力场景下，AINFER-PD 显著缩短了 rollout 完成时间。

#### （2）双节点性能（Table 6）
| 对比项 | Makespan 减少幅度 |
|--------|------------------|
| vs AInfer Normal | **18.0% – 35.3%** |
| vs SGLang | **18.3% – 31.8%** |

> 表明 AINFER-PD 在更大规模部署中仍具显著优势。

#### （3）关键延迟指标
- **p99 Request Completion Time**：
  - vs Normal：降低 **21.3% – 37.9%**
  - vs SGLang：降低 **39.3% – 44.0%**
- **Trajectory Completion Time**：
  - vs Normal：降低 **8.2% – 22.7%**
  - vs SGLang：降低 **27.0% – 33.4%**
- **p99 TTFT**：基本持平（±6.3%），说明未牺牲首 token 延迟。

---

### 🔬 消融实验结果（Table 5）

#### 不同 segment ordering 策略对比（BF16, DeepEP, MTP off）

| 策略 | E2E (s) | Req/s | D Wait (ms) | Ov. (%) |
|------|--------|-------|-------------|---------|
| Global-Complete | 179.79 | 7.036 | 138.5 | 89.8 |
| Global-Enqueue | 181.39 | 6.974 | 127.3 | 90.4 |
| **Fine-grained** | **165.83** | **7.628** | **17.5** | **95.0** |

> ✅ **Fine-grained segmentation 比 whole-epoch enqueue 进一步减少 8.6–19.8% 的 completion time**，并将 D 等待时间从 ~127ms 降至 **17.5ms**，证明细粒度边界的重要性。

#### Segment Size 自适应选择（Figure 8b）
- 运行时 selector 在所有测试中均选择 **segment size=2**
- 最优固定 size 可达最佳性能的 99.2%，但错误选择（如 size=1 或 4）最多损失 **6.1%**
- 说明自适应策略可有效平衡同步开销与并行收益。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Persistent P/D coexistence 是 agentic RL rollout 的常态**，必须通过安全并发机制处理。
2. **传统 in-place multiplexing 缺乏通信隔离**，在 MoE + ADP/ATP 场景下易引发分布式死锁。
3. **AINFER-PD 通过 turnstile 同步 + phase-owned state 设计**，首次实现了通信安全的并发 P/D 执行。
4. **细粒度 segmentation 显著优于粗粒度调度**，可在不牺牲吞吐的情况下极大降低 D 等待延迟。
5. **在真实 RL 轨迹上，AINFER-PD 可将 rollout 时间缩短近 1/3**，显著加快训练循环。

### ⚠️ 局限性
- **适用范围有限**：主要针对 in-place 部署场景；若已有足够资源做 P/D disaggregation，则后者更简单。
- **ordering 假设强健性依赖健康网络与 rank 行为一致**，异常情况下的恢复机制未深入探讨。
- **未完全解耦资源竞争**：NIC bandwidth、SM 资源仍共享，需依赖 admission control 和 budgeting。
- **当前仅验证 DeepEP**，其他 expert communication runtime 是否可迁移尚待验证。

### 🔮 未来工作方向
- 开发 **online adaptive segment policy**，根据实时负载动态调整 P admission 粒度。
- 扩展至更多复杂并行拓扑（如 MoE + PP + TP + DP 混合）。
- 探索 **TTFT-aware admission control**，在 completion time 与用户体验间更好权衡。
- 将通信安全机制推广至其他并发推理场景（如 speculative decoding + user request）。

---

> 💡 **总结一句话**：  
> **AINFER-PD 在共享设备上实现了通信安全、状态隔离的 in-place P/D multiplexing，解决了 MoE rollout 中长期存在的并发干扰问题，在真实 RL 负载下将端到端 rollout 时间缩短了 18–35%，是迈向高效 agentic RL 推理的重要一步。**

</details>

---

### 5. [DRLM: Deep Reinforcement Learning-Based LLM Query Orchestration in Edge Environments](https://arxiv.org/abs/2609.00442)

**Authors**: Reza Farahani, Zoha Azimi Ourimi, Mario Colosi, Lauri Loven, Christian Timmerer, Schahram Dustdar  
**Category**: cs.DC  
**Published**: 2026-09-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.00442v1  

#### Abstract
Large language model (LLM) services increasingly process heterogeneous queries with diverse latency, accuracy, and resource requirements. While edge deployment reduces response time, the heterogeneity of devices and the diversity of model families, parameter scales, and quantization levels make effi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DRLM: Deep Reinforcement Learning-Based LLM Query Orchestration in Edge Environments

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **edge computing** 环境中，大型语言模型（LLM）服务面临以下挑战：
- **设备异构性**：边缘节点硬件差异大（如 Raspberry Pi 与 Jetson AGX），导致相同模型部署后延迟和资源消耗差异显著。
- **模型多样性**：不同 **model family**（如 Llama3、Gemma）、参数规模（parameter scale）和量化级别（quantization level）带来复杂的 **latency-quality-resource 权衡**。
- **查询异质性**：用户请求在语义类别、复杂度、token 长度等方面高度多样化。

传统方法通常将 LLM 调度简化为静态的模型选择或设备分配，忽略了 **query semantics、model configuration 和 runtime system state** 的联合动态影响，难以实现细粒度、自适应的 **per-query orchestration**。

---

### 🚀 提出的新方法：DRLM
本文提出 **DRLM** —— 一种基于 **Deep Reinforcement Learning (DRL)** 的 LLM 查询编排框架，其核心创新包括：

#### （1）**联合建模三大要素**
DRLM 将 LLM 编排建模为一个 **Markov Decision Process (MDP)**，综合考虑：
- **Query semantics**：通过轻量级编码器提取语义特征并分类。
- **Predictive performance modeling**：
  - **Class-conditioned quality estimator**：按语义类别聚合历史表现，预测响应质量（避免逐查询精度预测的噪声）。
  - **Feature-driven latency predictor**：使用 LightGBM 回归模型，结合 query 结构、model 配置、device 特征预测推理延迟。
- **Runtime system state**：实时监控设备利用率（CPU/GPU/Memory）和队列状态（queue length & remaining time）。

#### （2）**Factorized Proximal Policy Optimization (PPO) Agent**
采用分解式策略网络：
```math
\pi(a|s) = \pi(f,p|s)\cdot\pi(z|s,f,p)\cdot\pi(e|s,f,p,z)
```
即分阶段决策：先选 model family & scale → 再选 quantization → 最后选执行 device。该设计：
- 显著降低组合动作空间大小；
- 支持动态 masking 不可行选项（如设备不支持某配置）；
- 提高训练效率与稳定性。

#### （3）构建大规模基准数据集
发布了一个包含 **223,835 条测量记录** 的 benchmark dataset，涵盖：
- 1258 个 queries
- 6 个 semantic classes
- 8 个 model families（共 32 个实例）
- 5 种 quantization levels
- 多种 edge devices（RPi, Jetson, VMs）

用于训练分类器、延迟/质量预测器，并支持可复现的数据驱动研究。

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | DRLM 如何改进 |
|------|--------|----------------|
| **RouteLLM** | 仅在强-弱模型对间二分类路由 | 支持多模型、多设备、多量化级别的细粒度决策 |
| **OptLLM** | 基于 Pareto frontier 的静态选择 | 引入运行时状态感知，动态调整策略 |
| **ExeGPT / Bullet** | 优化固定模型下的执行调度 | 同时进行模型选择 + 执行调度 |
| **Edge-LLM / EdgeShard** | 关注协同推理与模型切片 | 更聚焦于 per-query 的模型-设备联合选择 |

> ✅ DRLM 是首个将 **state-aware DRL** 应用于异构边缘环境中 LLM 查询编排的工作。

---

## 2. 核心实验方法和设置

### 📚 数据集与工作负载
- **Queries 来源**：
  - MMLU（生物学、地理、历史等）
  - GSM8K（数学题）
  - CommonsenseQA
  - TruthfulQA
- **预处理**：
  - 分成 **6 个 semantic classes**（非原始数据集标签，而是基于 embedding 聚类分类）
  - 每类约 200 queries，共 1258 个
- **到达模式**：Poisson 过程，λ ∈ {0.5, 1, 2}

### 💻 实验平台
- **Edge Testbed**：64 节点 Kubernetes 集群，包含：
  - Raspberry Pi 3/4（受限设备）
  - Jetson Nano / Orin Nano / Orin AGX（嵌入式 GPU）
  - KVM-based VMs（高性能虚拟机）
- **部署模型**：32 个 LLM 实例，来自 8 个 model families（见 Table I），参数从 0.27B 到 20B，支持多种 quantization（Z2~Z16）

### 🧪 评估指标
| 指标 | 定义 |
|------|------|
| **Response Quality** | 平均正确率（vs ground truth） |
| **Inference Latency** | 模型执行时间（ms） |
| **Waiting Time** | 排队延迟（反映拥塞情况） |
| **End-to-End Latency** | `inference + waiting` |
| **Orchestration Overhead** | 单次决策耗时（ms/query） |

### 🆚 对比基线方法
| 方法 | 描述 |
|------|------|
| **Random** | 随机选择可行 (m,e) 对 |
| **High-Acc** | 选择预测准确率最高的模型（忽略延迟与负载） |
| **Fastest** | 选择预测延迟最低的配置（忽略质量） |
| **RouteLLM-style** | 强模型（GPT-OSS 20B）vs 弱池；根据优势阈值路由 |
| **OptLLM-style** | 在 quality-latency Pareto frontier 上选择最优部署配置 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Fig. 7c 与摘要）

| 指标 | DRLM 提升幅度 | 说明 |
|------|---------------|------|
| **Inference Latency ↓** | **最高减少 51%** | 相比 High-Acc / OptLLM-style |
| **Queuing Delay (Waiting Time) ↓** | **最高减少 67%** | 显著缓解热点设备拥堵 |
| **End-to-End Latency ↓** | 最高减少 ~60% | 在高负载下仍保持低延迟 |
| **Accuracy Loss** | **≤ 8%** | 在大幅提速前提下维持合理质量 |
| **Under Increasing Load** | Latency improvement up to **61.4%** | 表现出良好可扩展性和鲁棒性 |
| **Orchestration Time** | ~35 ms/query | 比 OptLLM-style 快约 60%，适合在线场景 |

---

### 📊 与其他方法对比分析

#### （1）**质量 vs 延迟权衡**
- **High-Acc** 虽然准确率最高（~0.56），但在高负载下因集中使用大模型导致严重排队，end-to-end 延迟极高（尾部超 100s）。
- **Fastest** 延迟低但质量差（~0.4），且过度依赖小模型造成局部过载。
- **DRLM** 在质量（~0.50–0.52）与延迟之间取得最佳平衡，且随负载增加性能波动最小。

#### （2）**调度开销**
- DRLM 决策时间为 **35 ms/query**，远低于 OptLLM-style 和 RouteLLM-style（~80–90 ms），得益于 factorized policy 设计避免穷举搜索。

#### （3）**资源利用分布**
- DRLM 成功实现 **load-aware 分流**：
  - 简单查询 → 小模型 + 受限设备（RP/NJN）
  - 复杂查询 → 大模型 + 高性能设备（JOA/VM）
- 而 High-Acc 导致 JOA 和 VM 成为瓶颈，形成“热点”。

---

### 🔬 消融实验与验证（间接体现）

虽然未明确列出消融表，但通过多个模块的独立评估验证有效性：

#### （1）**Predictor 准确性**
- **Latency Predictor (LightGBM)**：$ R^2 = 0.75 $，log-scale 下预测趋势准确，足以支撑排序决策。
- **Quality Estimator**：基于 class-conditioned 统计，稳定可靠，避免 per-query 波动。
- **Query Classifier (RF)**：混淆矩阵接近对角线，分类一致性高。

#### （2）**训练收敛性**
- PPO 训练在 200 episode 内快速收敛，reward 稳定上升，critic loss 快速下降至接近零，表明价值函数估计准确。

#### （3）**特征重要性分析**
- Latency prediction 中，**prompt length 和结构复杂度**贡献最大，其次是 model 参数规模，device 影响较小（因与 model 部署耦合）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 编排必须是状态感知的**：不能仅依据静态性能表，而需结合当前系统负载、队列状态进行动态决策。
2. **模型性能具有任务依赖性**：没有“全能”模型。例如：
   - Gemma3 在 Mathematics 表现好，Qwen3 在 TruthfulQA 更优。
   - 扩展模型规模不一定提升所有任务表现（scaling benefit 非单调）。
3. **量化效果是双刃剑**：可能提升 accuracy（如 Gemma3 on RP），但也可能显著增加 latency（如 Qwen3 on JOA），需联合建模。
4. **Factorized DRL 策略高效可行**：通过分解动作空间，可在大规模异构环境下实现快速、稳定的策略学习。

---

### ⚠️ 方法局限性
1. **依赖高质量预测器**：若 latency 或 quality 预测偏差较大，会影响策略质量（尽管作者已证明其足够用于相对排序）。
2. **冷启动问题**：新 query 类别或新 model 配置上线初期缺乏数据，class-conditioned estimator 和 predictor 性能可能下降。
3. **未考虑 energy/cost**：目前只优化 latency 与 quality，尚未纳入能耗或经济成本作为目标。
4. **仿真训练 vs 实际部署 gap**：策略训练基于 predictor 输出轨迹，而非真实 LLM 执行，可能存在建模误差。

---

### 🔮 未来工作方向
- 扩展为 **multi-objective optimization**，引入 **cost** 和 **energy consumption** 指标。
- 支持 **online adaptation** 机制，应对模型漂移、设备故障或新 workload。
- 探索 **federated learning** 架构，在保护隐私的前提下跨边缘域共享编排经验。
- 结合 **LLM-as-a-Judge** 技术，自动标注 query 类别与质量，降低人工标注依赖。

---

## 总结

> **DRLM 是首个将 state-aware deep reinforcement learning 应用于异构边缘环境中 LLM 查询编排的框架**。它通过融合 query profiling、lightweight predictors 与 factorized PPO agent，实现了细粒度、动态、高效的模型-设备联合调度。实验证明其在显著降低延迟的同时，仅牺牲有限的质量，展现出强大的实用潜力，为未来智能边缘 AI 服务提供了新的架构范式。

</details>

---

### 6. [Characterizing the Scalability and Performance of Large-Scale AI Training Under Multi-Tenancy](https://arxiv.org/abs/2609.00817)

**Authors**: Jacopo Raffi, Thomas Pasquali, Lorenzo Piarulli, Filippo Spiga, Marco Faltelli, Andreas Herten, Domenico Siracusa, Daniele De Sensi, Flavio Vella  
**Category**: cs.DC  
**Published**: 2026-09-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.00817v1  

#### Abstract
Characterising AI workload performance on modern HPC systems requires understanding both their scalability in isolation and their behaviour under concurrent execution. However, the interplay among parallelisation strategies, network congestion, compute capability, and interconnect technologies remai...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Characterizing the Scalability and Performance of Large-Scale AI Training Under Multi-Tenancy*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统地研究了**大规模分布式AI训练在多租户环境下的可扩展性（scalability）和性能表现**。尽管已有大量工作分析了单个训练任务的通信开销或网络瓶颈，但对以下因素如何**共同作用**的研究仍不充分：
- 不同 **parallelization strategies**（如 DP, FSDP, DP+PP+TP 等）
- 多种 **interconnect 技术与拓扑**（如 Dragonfly, Dragonfly+, NVLink）
- **job placement** 策略
- **multi-tenancy 干扰**（即多个作业并发执行时的网络拥塞）

这些问题在真实超算和AI集群中尤为关键，因为资源是共享的，且训练任务规模巨大（可达数千GPU）。

### 提出了什么新方法或新思路
作者提出了一个名为 **DLNetBench** 的基准测试框架，其核心思想是：
- **解耦计算与通信**：不运行真实的深度学习模型（如 PyTorch），而是通过 **roofline 性能模型** 精确模拟每个计算阶段的时间。
- **直接调用 collective operations**：根据每种 parallelization strategy 自动生成对应的 NCCL/RCCL/MPI 集合通信操作（如 `ALLREDUCE`, `ALLTOALL`, `SENDRECV`），从而精确控制通信模式。
- **支持细粒度并发建模**：可以灵活配置多个并发作业的混合负载（mix）、放置策略（placement）和拓扑感知分配，实现对多租户干扰的可控研究。

### 相比现有方法的优势
| 方面 | 传统方法（如 PyTorch + DeepSpeed） | DLNetBench |
|------|-------------------------------|-----------|
| 控制精度 | 受限于框架内部调度、内存管理等不可控因素 | 完全控制通信顺序、计算延迟、job placement |
| 实验可复现性 | 易受系统噪声、动态调度影响 | 在保留系统底层通信栈的同时提供高可复现性 |
| 多租户建模能力 | 难以构造复杂的并发场景 | 支持从均匀到幂律分布的真实 workload 分布 |
| 跨平台比较 | 模型实现差异大，难以公平对比 | 统一抽象层，可在不同硬件（NVIDIA/AMD）、软件栈上运行 |

> ✅ **创新点总结**：提出了一种**硬件感知、通信驱动、轻量级的仿真框架**，用于在真实HPC/AI基础设施上进行**系统性的、可重复的大规模AI训练行为表征**。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集 / 模型
论文并未使用传统意义上的“数据集”，而是基于代表性 **AI模型架构** 构造通信行为，具体包括：

| Parallelization Strategy | 对应模型 | 特点 |
|--------------------------|--------|------|
| **DP** | ViT-H | 纯数据并行，小模型，高频全局同步 |
| **FSDP** | LLaMA3-8B, Minerva-7B | 参数分片，减少内存占用 |
| **DP+PP** | LLaMA3-8B, Minerva-7B | 流水线并行 + 数据并行 |
| **DP+PP+TP** | LLaMA3-70B | 张量并行增强，通信密集 |
| **DP+PP+EP** | Mixtral-8x7B | MoE 架构，引入 `ALLTOALL` 通信 |

这些模型的选择覆盖了当前主流的大语言模型训练范式。

### 实验设置
#### 系统平台
实验跨越六类系统，涵盖从节点级到超算级的不同规模：
- **Node-scale**: NVIDIA DGX A100, LUMI-G
- **Rack-scale**: NVIDIA NVL72 GB300
- **Supercomputers**: Alps, Leonardo, JUPITER

所有系统均采用现代互连技术（NVLink, InfiniBand NDR/HDR, Slingshot）和 Dragonfly 或 Dragonfly+ 拓扑。

#### 并发配置设计
- **Allocation Patterns**:
  - **Uniform**: 所有 job 大小相同
  - **Tier-sampled**: 模拟真实集群中的幂律分布（75% 小任务, 20% 中等, 5% 大任务）
- **Strategy Assignment**: 使用 **entropy-based sampling** 来生成不同多样性水平的任务组合（低/中/高策略混合度）
- **Job Placement Classes**:
  - INTRA-L1（同一L1交换机内）
  - INTRA-GROUP（同组内跨L1）
  - INTER-GROUP（跨组）
  - SAME-L1-n（拓扑感知绑定）

#### 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **Communication Fraction** | 单次迭代中通信时间占总时间的比例 | 衡量 workload 的网络敏感性 |
| **Parallel Efficiency** | 相对于最小规模的加速比归一化效率 | 衡量可扩展性 |
| **Throughput (samples/sec)** | 每秒处理的样本数 | 衡量绝对性能 |
| **Slowdown** | $ \frac{T_{\text{baseline}}}{T_{\text{concurrent}}} $ | 衡量多租户干扰程度 |

> 🔍 基线定义为：在保留资源上单独运行的结果；并发实验则通过 SLURM 生产队列提交，反映真实用户体验。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### A. 通信开销随规模急剧上升（Fig. 2）
| 系统 | 最大通信占比（典型策略） |
|------|--------------------------|
| DGX A100 (node-scale) | <10% （基本 compute-bound） |
| NVL72 GB300 (rack-scale) | ~40–58% |
| Leonardo / JUPITER / Alps | **30–80%**，即使仅几节点也显著 |

> ⚠️ **发现**：一旦跨出节点边界，scale-out 通信成为主导瓶颈。

#### B. 可扩展性表现差异巨大（Fig. 3）
| Strategy | 并行效率趋势 |
|---------|-------------|
| **FSDP** | >90%，几乎理想扩展（得益于良好重叠） |
| **DP** | 在超算上严重退化，64 GPU 时效率常低于 50% |
| **DP+PP+TP / DP+PP+EP** | 80–95%，稳健但受 placement 影响 |

> 📈 在 DGX A100 上所有策略均接近完美扩展（>99%），说明 NVSwitch 成功消除了通信瓶颈。

#### C. 多租户干扰下的 Slowdown 分布（Figs. 4–8）

| 系统 | 典型 slowdown | 极端情况 |
|------|---------------|----------|
| **DGX A100** | <1.15× | 极少数达 21×（瞬态噪声） |
| **NVL72 GB300** | <1.015× | 几乎无影响 |
| **JUPITER** | 1.1–1.5×（INTRA-L1）<br>**8.6× (intra-group), 23.6× (inter-group)** | 跨组通信极度脆弱 |
| **Leonardo** | 整体较低（A100 GPU 较慢） | 但 1K-GPU DP+PP+EP 达 65% 吞吐损失 |
| **Alps / LUMI** | DP 高方差，部分 run 达 60–100× slowdown | 与 RCCL/NCCL 初始不稳定有关 |

> 💡 **Topology-aware placement 效果显著**：
> - 在 JUPITER 上，将 DP+PP+TP 从普通跨组改为 **4-nodes-per-switch** 配置，平均 slowdown 从 15% 降至 **5%**。
> - 类似优化对 DP+PP 几乎消除 variability。

---

## 4. 关键结论和发现

### 主要发现

#### ✅ **Observation 1: Communication Overhead 是 Scale-Out 的主要瓶颈**
- 在 node/rack 内部（如 DGX A100, NVL72），通信开销极低，系统 compute-bound。
- 一旦跨节点、跨组，通信时间迅速攀升至 **30–80%**，成为性能决定因素。

#### ✅ **Observation 2: Network Placement 对 Aggregate Throughput Scaling 影响有限（在隔离环境下）**
- 在无干扰情况下，不同 placement 的吞吐曲线非常接近。
- 原因是通信饱和了链路带宽，掩盖了路径长度带来的延迟差异。

#### ✅ **Observation 3: Congestion Effects 已在节点级显现，但在超算级才真正严重**
- DGX A100 和 NVL72 几乎不受并发影响（尤其后者完全免疫）。
- 超算系统中，特别是 **inter-group traffic**，会引发剧烈 slowdown（最高达 23.6×）。

#### ✅ **Observation 4: Topology-aware Placement 是有效的缓解手段，但有边界**
- 固定节点绑定（same-L1-n）可大幅降低平均 slowdown 和 variance。
- 但对于具有长依赖链的策略（如 DP+PP），reservation 有时反而更差——因其对单条消息延迟更敏感。

#### ✅ **其他重要发现**
- **Pure DP 最脆弱**：由于频繁的 global `IALLREDUCE`，极易受网络抖动影响，在 Alps/LUMI 上表现出极高 variance。
- **FSDP 最鲁棒**：良好的通信-计算重叠使其即使在高压下也能维持高效。
- **Hybrid Strategies 更稳定**：虽然通信量大，但若合理映射到拓扑（如 TP intra-node），仍可保持高效率。
- **Library Instability 是独立噪声源**：NCCL/RCCL 在启动初期存在收敛过程，导致前几次迭代异常缓慢，这与拥塞无关，也无法通过 placement 缓解。

---

### 方法的局限性
1. **未运行真实模型**：DLNetBench 使用 roofline 模型替代实际 tensor 计算，可能忽略某些框架级优化或 kernel fusion 效果。
2. **固定参数设定**：batch size、microbatch 数量等为预设值，未探索参数空间的影响。
3. **未考虑存储I/O和检查点开销**：仅关注训练主循环中的 compute-communication 交互。
4. **limited model coverage**：仅覆盖五种典型策略，未包含 ZeRO-3、Pipedream 等更复杂方案。

---

### 未来工作方向
1. **扩展更多模型与参数配置**：研究 compute-to-communication ratio 如何影响 scalability。
2. **深入分析 collective library behavior**：区分“真实拥塞”与“library冷启动噪声”。
3. **集成更多通信后端**：支持 OneCCL、Gloo 等，进行跨 vendor 比较。
4. **构建自动化 placement optimizer**：基于 workload 特征推荐最优 topology-aware 分配策略。
5. **在更大规模上验证 worst-case contention**：例如模拟数十个大型 job 同时跨数百个组通信的情景。

---

## 总结
该论文通过对 **2400 GPUs 规模** 下多种 parallelization strategy、interconnect topology 和 multi-tenancy 场景的系统性实证研究，揭示了：
> 🔑 **大规模AI训练的性能不仅取决于算法和硬件，更由通信模式、job placement 与系统噪声之间的复杂交互所决定。**

它强调了在设计下一代 AI 系统（尤其是 “Giga-Factory-scale” infrastructures）时，必须联合考虑：
- **parallelization strategy selection**
- **network topology design**
- **scheduler-aware job placement**
- **robustness to system noise**

而提出的 **DLNetBench** 框架为这一领域的持续研究提供了强有力的工具支撑。

</details>

---

### 7. [Just Talk Once: Communication-Efficient Split Federated LLM Fine-Tuning on Edge Devices](https://arxiv.org/abs/2609.01457)

**Authors**: Jiaxiang Geng, Xianhao Chen, Bing Luo  
**Category**: cs.DC  
**Published**: 2026-09-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.01457v1  

#### Abstract
Large language model (LLM) fine-tuning is increasingly shifting toward data generated on edge devices, where memory, computation, bandwidth, and connectivity constraints make conventional federated learning difficult to sustain. Split federated fine-tuning (SFT) improves client-side efficiency by of...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Just Talk Once: Communication-Efficient Split Federated LLM Fine-Tuning on Edge Devices*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在边缘设备上进行 **Large Language Model (LLM)** 的微调面临诸多挑战，尤其是传统的 **Federated Learning (FL)** 和现有的 **Split Federated Fine-Tuning (SFT)** 方法存在以下瓶颈：

- **通信开销大**：标准 SFT 要求客户端频繁上传激活值（activations）并接收梯度，导致高带宽消耗。
- **双向依赖强**：U-shaped SFT 为保护目标 token 隐私，将预测头和损失计算移回客户端，形成“上传激活 → 下载输出 → 计算损失 → 上传梯度”的**双向闭环**，显著增加每步通信成本，并要求客户端持续在线。
- **客户端资源受限**：边缘设备内存、算力有限，难以支持全模型训练或长时间同步参与。

这些问题使得现有方案在**带宽受限、连接不稳定**的移动环境中难以实用化。

---

### 提出了什么新方法或新思路
本文提出两种新型框架：

#### ✅ **L-shaped SFT**
- **核心思想**：利用现代 LLM 中普遍存在的 **weight tying**（嵌入层与输出头共享权重）特性，将监督信号从客户端转移到服务器端。
- 客户端不再需要接收服务器输出来计算 loss，而是上传 **cut-layer activation** 和对应的 **target embedding**（由本地 `Embed(y)` 生成）。
- 服务器直接在隐藏状态空间中通过 contrastive loss（称为 `LA-Loss`）监督预测状态 $ h_s $ 与目标 embedding $ e_y $ 的对齐，从而实现 loss 的**全服务器侧计算**。
- 消除了 U-shaped SFT 所需的 activation download 与 gradient upload 步骤。

#### ✅ **One-shot SFT**
- 在 L-shaped SFT 基础上的进一步扩展。
- 客户端仅执行一次前向传播，缓存所有 `(activation, target embedding)` 对，并一次性上传至服务器。
- 之后客户端可完全离线，服务器基于缓存的数据进行多轮优化。
- 实现“**只说一次**”（Just Talk Once）的极低通信频率模式。

---

### 相比现有方法的优势
| 维度 | U-shaped SFT | L-shaped SFT | One-shot SFT |
|------|--------------|---------------|----------------|
| **通信步骤** | 4步（上下下上） | 2步（上传 + 梯度下载） | 1次上传 + 后续无交互 |
| **客户端在线时间** | 全程同步 | 每步参与 | 单次短暂上线 |
| **通信开销** | 高 | 显著降低（减少25%-87%） | 极低 |
| **容错性** | 差（断连即失败） | 较好 | 极强（天然抗 dropout） |

> ✅ **优势总结**：
> - 减少 per-step 通信量，提升系统效率；
> - 放松客户端同步需求，适应不稳定的边缘环境；
> - 保持 fine-tuning 效果接近甚至优于基线；
> - 可无缝集成到现有 Split Learning 架构中。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **WikiText-2**：用于文本生成任务，评估 **Perplexity (PPL↓)**。
- **MMLU (Massive Multitask Language Understanding)**：涵盖57个学科领域的多任务理解数据集，评估 **Accuracy (%)↑**。

> 数据划分模拟真实非独立同分布（non-IID）场景：
> - WikiText-2：按文章主题聚类分配给不同客户端；
> - MMLU：使用 Dirichlet 分布（α=0.8）按答案类别划分。

---

### 实验设置和评估指标

#### 模型选择
分两类 fine-tuning 设置：
- **Full Fine-Tuning**：微调全部参数，使用较小模型（<1B 参数），如：
  - GPT-2 Medium (355M), GPT-2 Large (774M), Qwen2.5-0.5B, Gemma3-270M
- **LoRA Fine-Tuning**：低秩适配，适用于更大模型（1B~8B），如：
  - Llama3.2-1B/3B/8B, Qwen3-1.7B/4B, Gemma3-1B/4B

#### 硬件测试平台
- **服务器**：Dell PowerEdge T640 + 2×NVIDIA A800 GPU
- **客户端**：
  - 2×NVIDIA Jetson Orin Nano（4GB RAM）
  - 3×Huawei nova 9 Pro（8GB RAM）
- **网络**：Wi-Fi 6
- **软件栈**：PyTorch + Transformers + Flower（联邦协调）+ 自研 MobileFineTuner（手机端）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Communication Cost** | 激活与梯度传输总量（GB），不含周期性聚合 |
| **Overall Latency** | 总训练耗时（分钟） |
| **Client Active Time** | 客户端实际在线参与时间 |
| **Fine-tuned Performance** | MMLU 准确率 / WikiText-2 困惑度 |
| **Dropout Robustness** | 不同客户端掉线率下的性能稳定性 |

#### 基线方法对比
- **Baseline**: U-shaped SFT（代表工作如 EUSFL, Mob-iLLM, FlexP-SFL）
- **对比重点**：去除各系统的附加优化模块，仅保留其原始 U-shaped 流水线，以公平比较框架本身差异。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

#### Full Fine-Tuning 结果（Qwen2.5-0.5B on MMLU）
| 方法 | Acc (%) | 通信成本 (GB) | 降幅 | 总延迟 (min) | 降幅 |
|------|--------|-------------|-------|------------|-------|
| U-shaped SFT | 45.63 | 1.71 | — | 252.69 | — |
| L-shaped SFT | 45.24 | 1.29 | ↓24.71% | 167.16 | ↓33.85% |

> ⚠️ 性能几乎持平，通信与延迟大幅下降。

#### LoRA Fine-Tuning 结果（Llama3.2-3B on WikiText-2）
| 方法 | PPL | 通信成本 (GB) | 降幅 | 总延迟 (min) | 降幅 |
|------|-----|--------------|-------|------------|-------|
| U-shaped SFT | 15.55 | 8.79 | — | 1297.85 | — |
| L-shaped SFT | 12.57 | 1.09 | ↓87.5% | 141.99 | ↓89.06% |

> ✅ **惊人提升**：通信减少近90%，延迟下降超89%，且效果更优！

---

### One-shot SFT 实验结果（Table 4）

| 方法 | 通信总成本 (GB) | 总延迟 (min) | 客户端活跃时间 (min) | 占比 |
|------|------------------|--------------|------------------------|------|
| U-shaped SFT | 4.79 | 252.69 | 240.66 | 95% |
| L-shaped SFT | 4.37 | 167.16 | 133.78 | 80% |
| **One-shot SFT** | **0.98** | **45.57** | **4.83** | **11%** |

> 💡 **核心突破**：客户端只需上线几分钟完成一次上传，后续全程离线，极大节省能耗与带宽。

---

### 消融实验结果（Ablation Studies）

#### ✅ 温度系数 $ \tau $ 影响（Fig. 7）
- 最佳 $ \tau = 1 $，与理论分析一致；
- $ \tau < 1 $：分布过尖锐，训练不稳定；
- $ \tau > 1 $：区分能力弱，性能下降。

#### ✅ 客户端层数配置影响（Table 5）
- 增加客户端侧 block 数会显著提高内存占用（从1.2GB → 4.4GB）；
- 但通信成本不变（因接口处 activation size 不变）；
- 超过6个 block 导致 OOM；
> 🔍 表明：**不宜过度增加客户端负载**，轻量化更符合边缘部署现实。

#### ✅ 抗客户端掉线能力（Fig. 8）
- **One-shot SFT**：几乎不受 dropout 影响（只要传过一次即可）；
- **L-shaped SFT**：表现稳健，性能缓慢下降；
- **U-shaped SFT**：随 dropout 率上升性能急剧恶化。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Weight tying 是打破 U-shaped SFT 双向瓶颈的关键**：
   - 利用 embedding 作为监督锚点，可在不暴露 token ID 的前提下实现 server-side supervision。
2. **L-shaped SFT 显著降低通信开销**：
   - 移除 activation download 和 gradient upload，使每步通信减半以上。
3. **One-shot SFT 实现“瞬时参与”范式**：
   - 客户端仅需一次上传即可退出训练，特别适合间歇连接设备。
4. **系统效率提升未牺牲模型质量**：
   - 多数情况下 fine-tuning 效果与 U-shaped SFT 相当甚至更好。
5. **真实异构边缘平台上验证有效**：
   - 在智能手机与开发板混合环境中成功部署，具备工程落地潜力。

---

### 方法的局限性
- **未解决 activation 隐私泄露风险**：
  - 上传的 hidden activation 仍可能携带敏感信息（如输入内容特征），需结合 DP 或 HE 进一步防护。
- **依赖 weight tying 假设**：
  - 若模型未采用 weight tying，则需额外设计机制（如上传部分 head vector）。
- **历史 cache 设计影响近似误差**：
  - cache 规模有限时，`LA-Loss` 是 full CE 的近似，极端情况下可能导致梯度偏差。

---

### 未来工作方向
- 将 L-shaped / One-shot SFT 与其他隐私保护技术（如 **Differential Privacy**, **Homomorphic Encryption**）结合，构建端到端安全框架。
- 探索动态 cache 更新策略，提升负样本代表性。
- 扩展至多模态 LLM 微调场景。
- 研究更高效的 activation 压缩与量化方法，进一步降低上传成本。

---

> 📌 **一句话总结**：  
> 本文提出的 **L-shaped SFT** 和 **One-shot SFT** 通过巧妙利用 weight tying 特性，实现了服务器端监督与单向通信，从根本上缓解了 Split Federated Learning 中的通信瓶颈，在保证 fine-tuning 质量的同时，将客户端通信和在线时间压缩到极致，为边缘设备上的高效、鲁棒 LLM 微调提供了全新路径。

</details>

---

### 8. [Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs](https://arxiv.org/abs/2609.00184)

**Authors**: Jonathan Zheng, Zirui Shao, Alan Ritter, Wei Xu  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.00184v1  

#### Abstract
Large language models (LLMs) rely on static pretraining corpora, causing their knowledge to become outdated over time. Existing approaches for evaluating knowledge edits either suffer from rapid contamination or rely on counterfactual edits that conflict with rigid existing knowledge. In this work, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）依赖于静态预训练语料库，导致其知识随时间推移而过时。现有知识更新方法面临两大挑战：
- **基准污染（Benchmark Contamination）**：真实世界中的新事实很快被纳入后续模型的训练数据中，使得评估失去时效性和有效性。
- **反事实编辑的脆弱性（Counterfactual Brittleness）**：通过修改已知常识（如“英国属于大洋洲”）进行测试会导致逻辑冲突，影响推理一致性。

### 提出的新方法与思路
本文提出了一套完整的合成世界框架，用于研究 LLM 中的时间知识插入与更新：

#### （1）PARALLELEVENTS：抗污染的虚构未来事件基准
- 构建一个从2030到2035年的**平行宇宙式知识图谱**，包含自然灾难、体育赛事、选举等复杂事件。
- 所有事件均为**虚构但合理（plausible yet unseen）**，避免与现实世界重叠，防止数据泄露。
- 支持多跳（multi-hop）、因果（causal）和属性推理，强调事件间的动态传播效应。

#### （2）SYNAPSE：基于合成数据的知识更新训练框架
- 利用教师模型（如 GPT-4.1）生成长文本新闻文章和偏好问答对（preference data），模拟真实信息流。
- 结合 **mid-training（next-token prediction）** 和 **instruction tuning（via DPO）** 阶段，实现参数级知识内化。
- 显式建模“应答”与“应 abstain”的行为偏好，减少幻觉和不必要的拒绝回答。

### 相比现有方法的优势
| 维度 | 现有方法（如 MQuAKE-CF, COUNTERFACT） | 本工作（PARALLELEVENTS + SYNAPSE） |
|------|----------------------------------------|-------------------------------------|
| 数据污染风险 | 高（真实实体易进入训练集） | 极低（完全虚构实体） |
| 推理复杂度 | 单一事实或短链 | 多跳、因果链、全局一致 |
| 可扩展性 | 依赖人工标注 | 全流程自动化生成+验证 |
| 行为控制 | 忽视响应风格与 abstention | 显式优化响应质量与 abstention |

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 主要数据集
- **PARALLELEVENTS**：
  - 包含 **41 个事件中心的知识图谱**（如 2034 FIFA World Cup）
  - 总计 **24,530 个实体**，**15,885 条关系**，**47,748 条新事实**
  - 覆盖自然灾害、体育、选举、城市建立、经济危机等主题
  - 支持三种问题类型：
    - **Single-hop**：直接回忆事实
    - **Multi-hop**：跨实体推理（平均 3.06 步）
    - **Causal**：推断事件后果（如“台风如何影响旅游业”）

#### 对比基准
- **MQuAKE-T / MQuAKE-CF**：基于 Wikidata 的时间变化与反事实编辑基准
- **COUNTERFACT**：经典反事实编辑数据集
- **PopQA**, **MMLU-Pro**, **IFEval**, **IFBench**：通用知识与指令遵循能力评估

### 实验设置与评估指标
#### 插入事实规模
在 LLaMA-3.1-8B-Instruct、OLMo-3-7B-Instruct、Gemma-3-4B-Instruct 上分别插入：
- 20、150、542、1,536 条新事实

#### 评估方式
- **Question Answering Accuracy**：二值判断是否正确
  - Multi-hop：仅看最终答案
  - Causal：需识别至少一个相关因果路径及趋势
- 使用 **LLM-as-a-judge**（LLaMA-3.1-70B, GPT-OSS-20B）自动评分，争议由人工仲裁
- 人类标注一致性达 **Cohen’s Kappa = 0.937**

#### 基线方法对比
| 类型 | 方法 |
|------|------|
| **In-Context Learning** | ICE, IKE |
| **Graph-Based Retrieval** | PokeMQA, DeepEdit, GWalk, MeLLo |
| **Fine-tuning** | LoRA, Single-layer FT |
| **Model Editing** | ROME, MEMIT, AlphaEdit, WISE |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（插入 542 条事实后）

| 方法 | Total Acc | Single-hop | Multi-hop | Causal |
|------|-----------|------------|-----------|--------|
| **IKE (best ICL)** | 63.32% | 92.91% | 53.47% | 29.31% |
| **LoRA Fine-tuning** | 12.02% | — | — | — |
| **ROME (editing)** | 7.63% | — | — | — |
| **SYNAPSE (ours)** | **73.67%** | 87.50% | 44.22% | **84.32%** |
| **SYNAPSE + 10 retrieved facts** | **79.83%** | **98.51%** | 54.76% | 76.86% |

> ✅ **SYNAPSE 比最强基线 IKE 提升 14.23%（总准确率）**

### 与其他方法的关键对比发现
- **ICL 方法（如 IKE）严重过 abstain**：
  - 在 PARALLELEVENTS 上 **72.53% 的查询被拒绝回答**
  - 尤其在因果问题上表现差（仅 29.31% 准确率）
- **传统 fine-tuning 和 editing 方法失效**：
  - 因为大量新实体无法通过权重微调有效绑定
  - ROME 等方法在新实体上几乎无效（<10% 准确率）
- **SYNAPSE 显著提升因果推理能力**：
  - Causal 准确率达 **84.32%**，远超 IKE 的 29.31%
  - 表明参数更新能更好建模事件动态演化

### 消融实验结果

#### （1）不同训练组件的作用（150-fact setting）
| 方法 | Total Acc | Causal Acc |
|------|----------|------------|
| SFT Only | 70.54% | 84.00% |
| Preference Only | 70.54% | 85.00% |
| SYNAPSE (full) | **85.55%** | **95.00%** |
| SYNAPSE + TULU-3 | **85.55%** | **95.00%**（同时保持 generalization） |

> ✔️ 合并 **SFT + DPO + general preference data** 效果最佳

#### （2）教师模型敏感性分析
即使使用开源弱教师（OLMo-3-7B），SYNAPSE 仍优于 IKE：
- GPT-4.1 → OLMo-3-7B 教师，性能下降有限
- 表明该框架对教师强度具有鲁棒性

#### （3）abstention 数据比例的影响
- 加入 **40% 的未知事实偏好数据（Dunk）** 可显著提高模型在无解问题上的 abstention 率（从 ~50% → 80%+）
- 验证了显式训练“何时不说”至关重要

---

## 4. 关键结论和发现

### 主要发现
1. **虚构合成世界是理想的时序知识评估环境**  
   PARALLELEVENTS 成功规避了基准污染问题，在 Wikipedia 页面共现统计中显示仅有 **0.39% 的实体出现在 ≥1,000 篇文章中**（对比 MQuAKE-CF 达 38.43%）。

2. **合成数据驱动的参数更新优于检索增强方法**  
   SYNAPSE 在复杂推理任务（尤其是因果链）上显著胜出，说明**将新知识真正“学进去”比“查出来”更可靠**。

3. **知识更新必须兼顾“获取”与“克制”**  
   单纯注入新知识会导致模型在无关问题上过度自信；引入 **preference learning + abstention signal** 是关键设计。

4. **可扩展性强，支持大规模持续学习场景**  
   插入 1,536 条事实时，SYNAPSE 性能下降平缓（仅比 542-fact 下降 1.2%），而基线平均下降 10.08%。

### 方法的局限性
- **计算成本较高**：需要先生成高质量合成数据（约 50 小时生成 10M tokens）
- **依赖强教师模型**：尽管对弱教师有一定鲁棒性，最优性能仍需 GPT-4 级别模型
- **未考虑顺序更新与灾难性遗忘**：当前为批量更新设定，尚未测试多轮增量学习下的记忆保持
- **领域偏倚**：训练分布偏向社会/事件类知识，STEM 领域略有退化（如 Math 下降 >5%）

### 未来工作方向
- 开发在线版本 SYNAPSE，结合 GRPO 或实时 reward modeling 实现动态更新
- 扩展至更多事件类型（如企业并购、科技发布）
- 引入噪声容忍机制以处理矛盾或不确定的事实输入
- 构建面向用户的交互式知识刷新系统，支持个性化事件追踪
- 进一步探索 episodic memory 与参数更新的融合架构（如 PANINI + SYNAPSE）

---

> 📌 **总结一句话**：  
> 本文提出了首个基于**虚构平行世界**的抗污染知识更新评测体系 **PARALLELEVENTS**，并构建了端到端的合成训练框架 **SYNAPSE**，实现了比现有方法高 **14.23%** 的知识插入性能，推动了 LLM 在动态真实世界中的适应能力研究。

</details>

---

### 9. [OUTLETS: Output-Length Prediction from Speculative Decoding Backbones](https://arxiv.org/abs/2609.01068)

**Authors**: Weihuang Wen, Yingying Liu, Yichuan Liu, Wenqi Zeng, Li Zhou, Chumin Sun, Jie Sun, Tianshu Yu  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.01068v1  

#### Abstract
The heavy-tailed distribution of output lengths in Large Language Model (LLM) serving poses major challenges for resource provisioning and cluster scheduling. Although output-length prediction can mitigate these issues, existing approaches have key drawbacks: external proxy models add substantial la...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《OUTLETS: Output-Length Prediction from Speculative Decoding Backbones》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）服务中的输出长度具有**重尾分布**（heavy-tailed distribution），导致资源调度困难。短请求常被长请求阻塞（Head-of-Line Blocking），造成尾延迟（tail latency）显著上升。有效的**输出长度预测**（output-length prediction）是实现高效调度（如 Shortest-Job-First）的关键，但现有方法存在以下问题：

- **代理模型**（proxy-based）：如 BERT 回归器，引入额外延迟和内存开销，且无法访问目标模型内部状态，预测保真度有限。
- **浅层探针**（shallow probing）：在目标模型隐藏状态上附加轻量 MLP，效率高但表达能力弱，难以捕捉生成轨迹的长期信号。

### 提出的新方法与新思路
提出 **OUTLETS**（Output-Length Prediction from Speculative Decoding Backbones），其核心思想是：

> **利用 speculative decoding 中的 draft decoder 所产生的 latent representations 来进行输出长度预测**。

具体设计包括：
- **共享骨干网络**：复用已有的 speculative decoding 骨干（如 EAGLE-3 架构），避免额外部署独立预测模块。
- **双头架构**（dual-head formulation）：
  - **Draft Model Head**：用于 speculative decoding，生成候选 token。
  - **Length Regression Head**：轻量级 MLP，从同一 draft state 中回归剩余生成长度（log-space 回归以稳定训练）。
- **联合训练目标**：同时优化 token drafting 和 length prediction，损失函数为 $ \mathcal{L} = \mathcal{L}_{sp} + \lambda \mathcal{L}_{len} + \|\theta_{\text{MLP}}\|^2 $。

### 相比现有方法的优势
- **更高的预测精度**：speculative draft decoder 显式建模未来 token 轨迹，其表示比目标模型的 next-token hidden states 更适合长度预测。
- **更低的推理开销**：当 speculative decoding 已启用时，仅需增加一个轻量回归头（约 0.7ms/step），边际成本极低。
- **系统级实用性**：预测结果可直接用于调度策略（如 Load Balancing + SJF），显著改善尾延迟和吞吐量。

---

## 2. 核心实验方法和设置

### 数据集
使用三个公开对话数据集，涵盖不同场景：
- **ShareGPT**：真实多轮对话。
- **Alpaca**：合成单轮指令遵循任务。
- **LMSYS-Chat-1M**：大规模开放对话。

对每个目标模型（Llama/Qwen），重新生成响应并提取真实输出长度作为标签。过滤掉超过 2,048 token 的样本，按 4:1 划分训练/测试集。

### 模型
评估三种主流 LLM：
- `Llama-3.2-1B-Instruct`
- `Llama-3.1-8B-Instruct`
- `Qwen3-30B-A3B`（MoE 推理模型）

### 基线方法
| 类型 | 方法 |
|------|------|
| **Proxy-based** | BERT（bigbird-roberta-base）、OPT（opt-125m 分类） |
| **Internal state-based** | MLP（接在目标模型中间层 hidden states 上） |
| **Instruction-based** | PIA-style prompting（要求模型先输出长度估计） |

### 评估指标
- **MAE**（Mean Absolute Error）：
  - **静态预测**（static）：$ \text{MAE} = \frac{1}{N}\sum_i |\hat{L}_i - L_i| $
  - **动态预测**（dynamic）：每步预测剩余长度，取平均 MAE。
- **系统级指标**（disaggregated serving）：
  - **P99 latency**（尤其是短请求 <800 tokens）
  - **Throughput**（tokens/s）
  - **Avg. latency**

### 实验设置
- 在 vLLM 基础上构建解耦式服务系统（1 prefill + 3 decode 实例）。
- 对比两种调度策略：
  - **Baseline**：Round-Robin + FCFS
  - **Ours**：基于 OUTLETS 预测的 **Load Balancing + Shortest-Job-First (SJF)**

---

## 3. 主要实验结果和性能指标

### 预测精度（MAE）对比（见 Table 1）

| Model | Dataset | OUTLETS (MAE) | MLP | BERT | OPT | PIA |
|-------|--------|---------------|-----|------|-----|-----|
| Llama-3.2-1B | ShareGPT (static) | **100.1** | 109.5 | 128.6 | 130.6 | 204.3 |
| Llama-3.1-8B | Alpaca (static) | **46.0** | 46.8 | 102.9 | 112.8 | 285.1 |
| Qwen3-30B-A3B | ShareGPT (static) | **186.7** | 210.8 | 262.2 | 255.2 | 936.2 |

✅ **OUTLETS 在所有模型和数据集上均取得最低 MAE**，显著优于各类基线。

#### 动态预测趋势
- 随着 decoding 进行，OUTLETS 的 MAE 持续下降（如 ShareGPT 上从 80.6 → 67.6），表明 draft state 能有效跟踪生成进度。

### 系统级性能提升（饱和负载下，100 QPS）

| Method | Throughput (tok/s) | Short P99 Latency |
|--------|--------------------|-------------------|
| RR + FCFS (Baseline) | 17,840.4 | 59.8 s |
| LB + SJF (OUTLETS) | **18,434.7** (+3.3%) | **39.0 s** ↓ **34.8%** |

- **短请求 P99 延迟降低 34.8%**
- **整体吞吐提升 3.3%**
- **长任务未出现饥饿现象**（P99 反而略有下降）

### 消融实验（Ablation Studies）

#### (1) 联合训练影响（OUTLETS vs. SD-ONLY）
- 添加 length prediction 目标后，**speculative acceptance rate 几乎不变** → 不损害 drafting 性能。
- 表明两个任务可在共享 backbone 中共存。

#### (2) 是否需要专用预测器？（OUTLETS vs. LP-ONLY）
- **静态预测**：OUTLETS 与 LP-ONLY 相当甚至更优（因 drafting 提供正则化）。
- **动态预测**：LP-ONLY 略优（有更多监督信号），但差距小，且需额外部署模型。

#### (3) 组件重要性分析
移除 draft decoder 导致 MAE 大幅上升，说明：
- 单纯拼接 hidden states 不够；
- **draft decoder 的 temporal modeling 能力至关重要**。

#### (4) 回归 vs. 分类
- **直接回归**（regression）优于 bucket 分类 → 更符合有序数值特性，训练更稳定。

#### (5) 跨域泛化（Zero-shot）
在未见过的 **GSM8K**（数学推理）和 **HumanEval**（代码生成）上测试：
- GSM8K: MAE = 397.9 （avg len=1,561）
- HumanEval: MAE = 506.6 （avg len=2,324）
→ 表现出良好泛化能力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Speculative decoding 的 latent representations 天然适合输出长度预测**，因其显式建模未来生成轨迹。
2. ✅ **OUTLETS 可无缝集成到现有 speculative decoding 流程中**，仅增加轻量回归头即可获得高质量长度信号。
3. ✅ **该预测信号可用于标准调度策略**（如 SJF 和 Load Balancing），在真实 disaggregated serving 系统中显著降低短请求 P99 延迟（↓34.8%）并提升吞吐。
4. ✅ **联合训练不会损害 speculative decoding 性能**，实现了“一石二鸟”的效果。

### 方法的局限性
- **部署成本依赖 speculative decoding 是否已启用**：若仅用于长度预测，则 speculative backbone 的计算开销不可忽略。
- **仅使用静态预测进行调度决策**：动态预测尚未用于在线迁移或重调度。
- **评估范围受限**：
  - 最大输出长度限制为 2,048 tokens；
  - 未验证极端采样温度、自定义停止条件等复杂场景下的表现。

### 未来工作方向
- 将动态预测用于 **runtime rescheduling 或 request migration**。
- 设计 **adaptive runtime**，根据负载自动切换 speculative acceleration 与 saturation-mode scheduling。
- 探索更长生成轨迹（如 agent workflows）下的预测鲁棒性。
- 结合 KV-cache usage 等运行时指标进一步增强预测能力。

---

> **总结一句话**：  
> OUTLETS 揭示了 speculative decoding 与 output-length prediction 的结构性联系，并通过复用 draft decoder 的 lookahead representations，在几乎零边际成本下实现了高精度长度预测，从而赋能高效的 LLM 服务调度。

</details>

---

### 10. [MakoXC: Rearchitecting DFT Exchange-Correlation with Matrix-Aligned and Knowledge-Organized Sparsity](https://arxiv.org/abs/2609.01025)

**Authors**: Haozhi Han, Fusong Ju, Jing Bai, Ruge Zhang, Xiang Zhao, Liang Yuan, Yunquan Zhang, Ting Cao, Liu Yunxin, Yifeng Chen, Kun Li  
**Category**: cs.DC  
**Published**: 2026-09-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.01025v1  

#### Abstract
Density Functional Theory (DFT) is indispensable for materials science and drug discovery, yet the exchange--correlation (XC) evaluation remains a major bottleneck due to its cubic scaling. Although linear-scaling methods exploit electronic nearsightedness to reduce asymptotic complexity, they produ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MakoXC: Rearchitecting DFT Exchange-Correlation with Matrix-Aligned and Knowledge-Organized Sparsity

---

## 1. 论文的主要贡献和创新点

### **解决的问题**

密度泛函理论（**DFT**）在材料科学和药物发现中至关重要，但其 **Exchange-Correlation (XC)** 能量计算存在严重的性能瓶颈。传统方法具有立方复杂度 $O(N^3)$，而现有的线性缩放方法虽然理论上降低了复杂度，但由于电子近邻效应（electronic nearsightedness）导致的**不规则稀疏性**（irregular sparsity），难以高效利用现代 AI 加速器（如 GPU 上的 Tensor Cores）。这些不规则模式导致内存访问碎片化、计算利用率低，无法实现真正的“实用线性缩放”。

### **提出的新方法与核心思想**

MakoXC 提出了一种模块化的 XC 评估引擎，通过算法-硬件协同设计，将物理诱导的稀疏性重构为**规则、加速器友好的矩阵对齐计算结构**。其核心创新基于“**Matrix-Aligned and Knowledge-Organized Sparsity**”的设计原则，具体包含三个关键技术：

#### **(1) Matrix-Aligned Cells (MACs)**  
- 将由电子近邻效应产生的局部密集交互重新组织成固定形状的微矩阵单元（MAC），使其与 Tensor Core 的 MMA tile（如 16×16）对齐。
- 引入自适应机制，根据 basis set 和 grid 分辨率动态选择最优 MAC 大小，并支持稀疏存储（CSR）以处理极端稀疏情况。
- **优势**：将不规则稀疏转化为硬件友好的规则结构，提升并行效率。

#### **(2) Sparsity-Guided Activation (SGA)**  
- 在 SCF 循环前进行一次性的激活筛选，结合领域知识挖掘显式与隐式稀疏性：
  - **Value-Based Filtering**：基于原子轨道（AO）值的大小过滤无效 MAC。
  - **Correspondence Activation**：利用 AO 乘积 $\phi_u(r)\phi_v(r)$ 与密度矩阵元素 $D_{uv}$ 的一一对应关系，进一步激活仅对最终结果有贡献的密度矩阵块。
- **理论保障**：基于 Gaussian Product Theorem 和 Schwarz Bound，证明 shell-pair screening 可作为 AO 乘积显著性的保守代理，确保无误删（no false negatives）。
- **优势**：首次系统性地将隐式组合稀疏性转化为可执行的结构化计算，逼近理论线性复杂度。

#### **(3) Kernel-Fused Pipeline (KFP)**  
- 将原本分散的 GEMM 和 DOT 操作融合为一个统一的片上流水线（on-chip pipeline），最大化 SRAM 数据复用，减少 HBM 流量。
- 集成两种冗余消除机制：
  - **Symmetry Folding**：利用密度矩阵对称性，只计算上三角部分，节省约一半计算量。
  - **Normcache Gating**：基于缓存的范数快速判断三元运算是否显著，提前过滤无效流。
- **优势**：从“内存受限的小核调用”转变为“高吞吐的计算密集型流水线”，充分释放 Tensor Cores 性能。

---

## 2. 核心实验方法和设置

### **数据集**

使用三类可扩展的代表性分子体系，覆盖不同维度和拓扑结构：

| 类别 | 示例 |
|------|------|
| **线性链状系统** | Polyglycine 链（(gly)$_n$） |
| **二维平面材料** | Boron-Nitride 片层（BN$_n$） |
| **三维球状团簇** | Water clusters（(H₂O)$_n$） |

### **实验设置**

- **硬件平台**：
  - 单卡：AMD EPYC 7V13 CPU + NVIDIA A100 GPU（80GB HBM）
  - 多卡集群：每节点 8×A100 + 96 核 CPU，通过 200 GB/s InfiniBand 互联
- **Basis Sets**：def2-SVP 和 def2-TZVP（来自 Basis Set Exchange）
- **XC Functional**：B3LYP
- **Grid Level**：统一设为 5（约 30,019 网格点/原子）
- **阈值**：精度验证用 $10^{-12}$，性能测试用 $10^{-10}$

### **评估指标**

| 指标 | 定义 |
|------|------|
| **Per-iteration XC time** | 每个 SCF 迭代中 XC 计算平均耗时（用于模块级性能评估） |
| **End-to-end SCF iteration time** | 整体 SCF 循环平均时间（含 Fock 构建等） |
| **Speedup** | 相对于基线方法的加速比 |
| **Parallel Efficiency** | 多 GPU 扩展下的强/弱扩展效率 |
| **Numerical Accuracy** | 相对于高精度参考（grid=7）的能量误差（单位：mHartree/atom） |

### **基线方法对比**

| 方法 | 描述 |
|------|------|
| **DenseXC** | 忽略稀疏性的稠密实现，代表传统立方复杂度基线 |
| **GPU4PySCF (v1.4.1)** | 当前最先进的 GPU 加速线性缩放 XC 方法 |
| **GauXC (v1.0)** | 另一种主流开源线性缩放实现，集成于作者框架内公平比较 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 对比项 | 加速比（Speedup） |
|--------|------------------|
| vs. **DenseXC** | **67.8×**（平均） |
| vs. **GPU4PySCF** | **4.7×**（平均） |
| vs. **GauXC** | **5.15×** |

> 注：在 def2-SVP 下最高达 87.1×（vs. DenseXC），def2-TZVP 下仍保持 48.4×。

### **与基线方法的对比结果**

- **实际线性缩放验证**（Fig. 6）：
  - MakoXC 是首个在真实 GPU 执行中展现出接近理想线性缩放趋势的方法。
  - 而 GPU4PySCF 和 GauXC 虽理论上线性，但因不规则负载未能实现真正线性行为。

- **端到端性能提升**（Fig. 11）：
  - 集成至商业级 DFT 包后，在 def2-SVP 下相较 GPU4PySCF 实现 **3.94×** 的整体 SCF 加速。

- **大规模可扩展性**（Fig. 12）：
  - 在 **Ubiquitin（1,231 原子，def2-SVP）** 上扩展至 **64 A100 GPUs（8 节点）**。
  - 维持 **>74% 并行效率**，单点能量计算在 **5 分钟内完成**。

- **数值精度**（Fig. 7）：
  - 所有体系下 XC 能量误差均 < 1 mHartree/atom，满足化学精度标准。
  - 与 PySCF（grid=7）高度一致，验证了 SGA 的数值正确性。

### **消融实验结果**（Fig. 8）

逐步添加各组件带来的性能增益：

| 阶段 | 相对于 DenseXC 的加速比 | 贡献说明 |
|------|--------------------------|--------|
| 仅 MAC + Value-Based Filtering | ~9.0× | 初步利用显式稀疏性 |
| + SGA（含 AO-Density Correspondence） | ~15.4×（再提升 1.71×） | 激活隐式稀疏性 |
| + KFP（完整 MakoXC） | **~67.8×**（再提升 4.1×） | 流水线融合与冗余消除主导性能飞跃 |

此外：
- **SGA 开销极低**：预激活阶段耗时 < 总运行时间的 3%，且随规模增大占比下降（Fig. 9）。
- **KFP 各技术贡献明确**：Symmetry Folding 和 Normcache Gating 的收益随系统增大而提升（Fig. 10）。

---

## 4. 关键结论和发现

### **主要发现**

1. **稀疏性可以成为桥梁而非障碍**：MakoXC 成功将物理诱导的不规则稀疏性重构为**矩阵对齐的规则计算结构**，实现了算法低复杂度与硬件高并行效率的统一。
2. **首次实现“实用线性缩放”**：不仅理论上线性，更在真实 GPU 执行中表现出近乎理想的线性增长趋势。
3. **端到端大分子模拟成为可能**：成功在 64 GPU 上完成 Ubiquitin 的 DFT 计算（<5 分钟），突破了传统方法的规模限制。
4. **模块化设计具备工业部署能力**：MakoXC 已集成进生产级商业软件，验证了其可靠性与广泛兼容性。

### **方法的局限性**

- **依赖预定义 basis set 和 grid**：MAC 大小需在计算初期确定，对动态调整支持有限。
- **当前聚焦 XC 模块**：虽已集成进完整 DFT 流程，但其他模块（如 ERI）尚未采用类似重构策略。
- **对极度稀疏系统优化空间更大**：若 AO 间相互作用极少，MAC 内部填充率可能偏低，影响 Tensor Core 利用率。

### **未来工作方向**

- 将 MAC 和 KFP 设计推广至 **Fock matrix 构建、ERI 积分等其他 DFT 核心模块**。
- 探索 **异构架构适配**（如 AMD CDNA、Apple Silicon）下的通用性。
- 结合 **adaptive gridding** 或 **machine learning prescreening** 进一步增强稀疏性挖掘能力。
- 支持 **hybrid functionals** 和 **meta-GGA** 等更复杂 XC 泛函的高效计算。

---

> ✅ **总结一句话**：  
> MakoXC 通过 **Matrix-Aligned Cells + Sparsity-Guided Activation + Kernel-Fused Pipeline** 三位一体设计，首次实现了 DFT 中 XC 计算的**实用线性缩放与极致加速**，为科学计算在 AI 加速器上的高效执行提供了全新范式。

</details>

---

### 11. [AgentFactory: Towards Automated Agentic System Design and Optimization](https://arxiv.org/abs/2609.01045)

**Authors**: Enci Zhang, Haofeng Wang, Yuesheng Zhu, Xiaole Cui, Guibo Luo  
**Category**: cs.AI  
**Published**: 2026-09-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.01045v1  

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities as powerful components in agentic systems, enabling sophisticated reasoning and complex task execution. However, current approaches to manually designing and optimizing agentic systems heavily rely on manual effort, limiting thei...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《AgentFactory: Towards Automated Agentic System Design and Optimization》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Models (LLMs)** 的 **Agentic System** 设计严重依赖人工设计与调优，存在以下问题：
- 手动设计流程繁琐、经验驱动，难以探索庞大的配置空间；
- 现有自动化方法（如 ADAS、AFlow）仅聚焦于 **workflow 结构优化**，忽略模型能力本身的提升；
- 多数方法只优化单一目标（如准确率），忽视实际部署中的 **cost（推理成本）** 和 **latency（延迟）** 等现实约束。

### 🚀 提出的新方法：AgentFactory
提出 **AgentFactory** —— 一个联合优化 **foundation model** 与 **workflow structure** 的自动化框架，实现多目标、端到端的 agentic system 自动化设计。

#### 创新点：
1. **联合优化空间扩展**  
   首次将 **LLM fine-tuning** 引入自动化 agent 设计流程，在 **model 参数空间（M）** 与 **workflow 表示空间（W）** 的联合空间中进行搜索，突破传统仅优化 workflow 的局限。

2. **多目标优化（Multi-objective Optimization）**  
   同时优化多个指标：**performance（准确性）、cost（推理开销）、efficiency（响应速度）**，更贴近真实应用场景。

3. **三阶段优化流水线（Three-stage Optimization Pipeline）**  
   - **Planning**：由 LLM 作为 optimizer 分析历史轨迹并生成优化策略；
   - **Tuning**：执行模型 fine-tuning（如 LoRA/QLoRA），增强特定领域能力；
   - **Workflow Design**：生成可执行的 code-based workflow，支持复杂控制流与工具集成。

4. **LLM-as-Optimizer 范式**  
   利用强大的 LLM（如 GPT-4o）作为元优化器（meta-optimizer），通过 prompt 引导其在非微分、部分可观测的空间中进行高效搜索。

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 AFlow、ADAS） | AgentFactory |
|------|-----------------------------|------------|
| 优化对象 | 仅 workflow | ✅ 联合优化 model + workflow |
| 模型适应性 | 固定基础模型 | ✅ 支持 fine-tuning 提升领域适配 |
| 优化目标 | 单一指标（如 accuracy） | ✅ 多目标平衡（accuracy/cost/latency） |
| 自动化程度 | 中等（需预设结构） | ✅ 全流程自动演化 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集（Benchmarks）
共涵盖 **5 个领域、8 个公开 benchmark**：
| 领域 | 数据集 | 测试样本量 | 描述 |
|------|--------|-----------|------|
| 通用推理 | MMLU、DROP | 各 1,000 | 多任务知识理解与段落推理 |
| 编程 | HumanEval、MBPP | 完整测试集 | 代码生成能力评估 |
| 数学 | GSM8K、MATH (level 5) | 完整 + 617题 | 数学推理与复杂计算 |
| 医疗 | MedQA | 完整测试集 | 医学考试问答 |
| 金融 | FinEval | 完整测试集 | 中文金融领域知识评测 |

---

### ⚙️ 实验设置与评估指标

#### 基础模型（Foundation Models）
- 主要使用：`Llama-3.1-8B-Instruct` 和 `GPT-4o-mini`
- AgentFactory 可动态选择最优组合以权衡性能与成本

#### 评估指标
| 指标 | 内容 |
|------|------|
| **Performance** | 各 benchmark 上的准确率（Accuracy/F1） |
| **Cost** | 推理总花费（美元），基于 OpenAI / OpenRouter 定价计算 |
| **Latency** | 响应时间（隐含在效率优化中） |

#### 对比方法（Baselines）
| 类别 | 方法 |
|------|------|
| 手动设计方法 | IO、Chain-of-Thought (CoT)、Self-Consistency (CoT-SC)、Reflexion、LLM Debate |
| 自动化 workflow 优化 | ADAS、AFlow |

> 所有方法在同一基础模型下公平比较；AgentFactory 允许自动切换模型并 fine-tune。

#### 实现细节
- **Fine-tuning 数据集**：CodeBagel（编程）、MathInstruct（数学）、IndustryInstruction（医疗/金融）等；
- **后端框架**：OpenRLHF 进行 fine-tuning；
- **硬件资源**：4×A6000 GPU，每轮训练最多 20K 样本、4 小时限；
- **LLM Optimizer 候选**：GPT-4o、GPT-4o-mini、Claude-3.5-sonnet、Gemini Pro 1.5、DeepSeek V3；
- 最大优化步数：20 步。

---

## 3. 主要实验结果和性能指标

### 📊 性能对比（见 Table 1）

| 方法 | 平均性能（Avg Acc） | 平均成本（$） | 相对提升 |
|------|--------------------|---------------|----------|
| 手动方法（如 Reflexion） | ~56.0–77.4 | $0.19–1.62 | — |
| 自动 workflow 方法（AFlow） | 59.8 (Llama) / 78.5 (GPT) | $0.29 / $2.13 | — |
| **AgentFactory (Ours)** | **68.9 (Llama)** / **83.9 (GPT)** | **$0.26 / $0.68** | **↑9.1% avg** |

#### 关键亮点：
- 在所有 8 个 benchmark 上均超越现有方法；
- **平均性能提升 9.1%**，显著优于手动设计与自动化 workflow 方法；
- 在专业领域提升尤为明显：
  - **MedQA ↑19.6%**
  - **FinEval ↑18.7%**

> 💡 原因分析：这些任务需要强领域知识，仅靠 workflow 优化无法弥补模型能力差距，而 fine-tuning 显著增强了语义理解与推理能力。

#### 成本优势
- 使用 `GPT-4o-mini` 时，相比其他自动化方法（如 AFlow），**平均成本降低高达 68%**；
- 多目标优化有效避免“为精度牺牲成本”的现象。

---

### 🔬 消融实验（Ablation Study，见 Figure 5）

| 设置 | 平均性能 | 相对基准提升 |
|------|---------|--------------|
| Baseline (IO) | 52.1 | — |
| w/o Fine-tuning（仅 workflow 优化） | 60.7 | ↑8.6% |
| w/o Workflow Design（仅 fine-tuning） | 57.4 | ↑5.3% |
| **Full AgentFactory** | **68.9** | **↑12.7% vs fine-tuning only** |

#### 发现：
- **fine-tuning 对专业领域（MedQA、FinEval）至关重要**；
- **workflow 设计对通用任务帮助更大**；
- 二者结合产生协同效应，验证了联合优化的有效性。

---

### 🔄 案例研究：FinEval 上的优化过程（Figure 4）
从初始 IO 准确率 62.3% 开始，经历四个阶段：
1. **Workflow 优化** → 引入 Multi-expert Cross-validation → 达 66.7%
2. **引入金融数据 fine-tuning** → 跃升至 72.5%
3. **跨域增强（加入数学数据）** → 发现金融题含数学逻辑 → 提升至 74.6%
4. **多目标优化** → 构建 Multi-Reasoning 流程，最终达 **75.2% 准确率 + $0.58 成本（目标 $0.60）**

> ✅ 展示了 AgentFactory 能自动发现人类可能忽略的优化路径（如数学-金融交叉）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **联合优化 model 与 workflow 显著优于单一维度优化**；
2. **fine-tuning 是提升领域特异性任务性能的关键**，尤其在医学、金融等专业场景；
3. **LLM 可作为有效的元优化器（LLM-as-Optimizer）**，能在复杂非微分空间中导航并生成高质量设计方案；
4. **多目标优化可在保持高性能的同时大幅降低成本**，具备实际部署潜力；
5. **AgentFactory 具备泛化性**：在不同 LLM optimizer（GPT-4o、Claude-3.5）上表现稳定。

---

### ⚠️ 局限性
1. **计算开销较高**：每次 fine-tuning 需数小时，限制了大规模快速迭代；
2. **依赖高质量 fine-tuning 数据集**：若缺乏领域标注数据，效果受限；
3. **LLM optimizer 的稳定性问题**：不同随机种子可能导致优化路径差异；
4. **未考虑实时反馈机制**：目前是离线优化，尚未接入在线学习或 human-in-the-loop。

---

### 🔮 未来工作方向
1. **轻量化 fine-tuning 策略**：探索更高效的参数更新方式（如 adapter fusion）；
2. **在线自适应优化**：让系统在运行过程中持续学习与调整；
3. **引入 human feedback**：构建 human-AI 协同优化闭环；
4. **扩展至多模态 agent system**：支持图像、语音等输入输出形式；
5. **开源 AgentFactory 框架**：推动社区共建自动化 agent 生态。

---

## ✅ 总结
**AgentFactory** 是首个将 **LLM fine-tuning** 与 **workflow 自动生成** 融合的多目标自动化 agent 设计框架。其实验证明：
> “**Better models + Better workflows = Much better agents**”

该工作标志着 agentic system 设计正从“手工工程”迈向“自动化智能演化”，为构建高效、低成本、高适应性的 AI agent 提供了新范式。

</details>

---

### 12. [Neural Symbollic Regression Using Deep Learning and Sparse Modelling](https://arxiv.org/abs/2609.01102)

**Authors**: Ravi Kumar U, Sumitra S  
**Category**: cs.LG  
**Published**: 2026-09-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2609.01102v1  

#### Abstract
Symbolic Regression (SR) seeks to find succinct mathematical expressions that represent the fundamental relationships within data, providing interpretability and scientific understanding that exceeds that of black-box models. Nevertheless, traditional methods like Genetic Programming face challenges...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Neural Symbolic Regression Using Deep Learning and Sparse Modeling 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 **Symbolic Regression (SR)** 方法面临以下挑战：
- **Genetic Programming (GP)**：收敛慢、计算开销大、对噪声敏感，且容易出现“代码膨胀”（code bloat）。
- **SINDy** 等稀疏回归方法：依赖于人工设计的固定特征库（feature library），表达能力受限，难以捕捉复杂的非线性交互关系。
- **神经符号混合方法**（如 DSR、AI Feynman）：将函数逼近与符号搜索耦合，导致训练不稳定、解释性差、计算成本高。

本文旨在解决这些方法在**可扩展性、噪声鲁棒性、符号可解释性**之间的权衡问题。

---

### 提出了什么新方法或新思路
提出了一种名为 **Neural Symbolic Regression (NSR)** 的解耦框架，其核心思想是：

> 将神经网络视为 **functional preconditioner**（功能预处理器），而非直接生成符号表达式的工具。

#### NSR 框架包含四个关键组件：
1. **Decoupled Pipeline**（解耦流程）  
   - 第一阶段：使用 **MLP 神经网络** 在一个非线性特征空间中学习目标函数的平滑、抗噪近似。
   - 第二阶段：在神经网络输出的基础上，应用 **LASSO** 进行稀疏回归，提取简洁、可解释的闭式表达式。

2. **Interaction-Aware Feature Library**（交互感知特征库）  
   - 构建包含基础变换（如 `sin`, `exp`, `x^2`）及其**成对交互项**（pairwise interactions）的特征矩阵 $\Phi_{\text{full}}(X)$。
   - 平衡表达力与可管理性，避免纯多项式展开带来的组合爆炸。

3. **Neural Preconditioning for Sparse Recovery**（神经预处理用于稀疏恢复）  
   - 利用神经网络“去噪”并增强符号结构的可识别性，使 LASSO 更易恢复真实方程。

4. **Integrated Hyperparameter Optimization**（集成超参优化）  
   - 使用 **Ray Tune + ASHA scheduler** 实现分布式超参数调优，自动优化神经网络结构（深度、宽度）、学习率、batch size 和 LASSO 正则化强度 $\lambda$。

---

### 相比现有方法的优势
| 方法 | 可解释性 | 可扩展性 | 噪声鲁棒性 | 符号准确性 |
|------|----------|-----------|-------------|--------------|
| GP | 高 | 低 | 低 | 中等 |
| SINDy | 高 | 中 | 中 | 依赖特征库 |
| DSR / Neural SR | 中 | 高 | 中 | 不稳定 |
| **NSR (Ours)** | **高** | **高** | **高** | **高** |

- **更稳健**：神经网络作为平滑器显著提升对噪声的容忍度。
- **更准确**：通过交互特征和深度神经拟合，能捕获复杂非线性关系。
- **更可解释**：最终输出为 LASSO 提取的稀疏解析表达式。
- **更高效可控**：解耦设计避免了端到端训练的不稳定性。

---

## 2. 核心实验方法和设置

### 使用的数据集
采用标准 **Nguyen Benchmark Suite (Nguyen-1 至 Nguyen-7)**：
- 包含多种函数类型：多项式、三角函数、指数、对数、复合形式。
- 示例：
  - Nguyen-1: $ f(x) = x^5 + x^4 + x^3 + x^2 + x $
  - Nguyen-3: $ f(x_1,x_2) = x_1 x_2 + x_1 + x_2 + x_1^2 + x_2^2 $

每组生成 1000 个样本，输入范围 $[-1, 1]$，加入可选高斯噪声 $\epsilon \sim \mathcal{N}(0, 0.01^2)$。

数据划分：70% 训练，15% 验证，15% 测试。

---

### 实验设置和评估指标

#### 模型架构
- **Feature Library**:  
  基础函数集：`{id, sin, cos, exp, log(1+x), x², x³, tanh}`  
  加入所有成对交互项（$a \neq b$）
- **Neural Approximator**:  
  - MLP，层数 $L \in \{1,2,3\}$，宽度 $H \in \{32,64,128\}$
  - 激活函数：GELU / ReLU
  - 正则化：Dropout, Weight Decay
  - 优化器：AdamW
- **Symbolic Extraction**: LASSO 回归，$\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}\}$
- **Hardware**: 使用 GPU（CUDA 或 Apple MPS）加速训练

#### 超参数优化
- 工具：**Ray Tune** + **ASHA scheduler**
- 搜索空间：学习率、网络深度/宽度、batch size、LASSO $\lambda$
- 目标：最小化验证集 RMSE

---

### 基线方法对比
1. **SINDy**：在同一特征库上进行线性稀疏回归（LASSO）
2. **Non-tuned NSR Baseline**：无超参优化的 NSR 版本
3. **PySR**：基于遗传编程的现代符号回归工具（作为外部参考）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（RMSE 对比）
**表 IV：Nguyen 各任务上的平均 RMSE（越低越好）**

| Method       | Nguyen-1 | Nguyen-2 | Nguyen-3 | Nguyen-4 |
|--------------|----------|----------|----------|----------|
| PySR         | 0.2111   | 0.0802   | 0.0798   | 0.0039   |
| SINDy        | 0.0423   | 0.0425   | 0.0433   | 0.0463   |
| **NSR (Tuned)** | **0.0087** | **0.0099** | **0.0122** | **0.0205** |

> ✅ **NSR 在 Nguyen-1~3 上显著优于所有基线，在 Nguyen-4 接近最优**

---

### 与其他方法的对比结果
- **预测精度**：NSR 在多数任务上达到最低 RMSE，表明其强大的拟合能力。
- **噪声鲁棒性**（图 4）：
  - 随着噪声标准差增加（从 0 到 0.1），NSR 性能下降最缓慢。
  - 显示出神经预处理有效抑制了噪声影响。
- **OOD 泛化能力**（图 6）：
  - 在训练域 $[-1,1]$ 上训练，在 $[-2,2]$ 上测试。
  - NSR 的预测曲线保持良好一致性，而 SINDy 出现明显发散。
  - 表明神经模型学到的是连续泛化映射，而非局部插值。

---

### 消融实验结果（Ablation Study）

**表 V：消融配置下的 RMSE 与选中项数**

| Configuration               | RMSE  | Terms Selected |
|----------------------------|-------|----------------|
| **Full NSR (tuned)**        | 0.004 | 5              |
| No Interaction Terms        | 0.037 | 3              |
| Reduced Feature Library     | 0.142 | 2              |
| Shallow Network ($L=1$)    | 0.081 | 4              |
| No Hyperparameter Tuning   | 0.019 | 7              |

#### 发现：
- 移除任一组件均导致性能下降。
- **交互项**对于多变量函数（如 Nguyen-6/7）至关重要。
- **更深网络**有助于建模复杂非线性。
- **超参调优**极大提升收敛速度与符号准确性。

---

## 4. 关键结论和发现

### 主要发现
1. **神经网络可作为有效的 symbolic preconditioner**  
   - 不必直接生成符号，而是先提供一个干净、平滑的目标供稀疏回归使用。
   - 这种“两步走”策略兼顾了灵活性与可解释性。

2. **解耦设计带来稳定性与鲁棒性**  
   - 分离 approximation 与 symbolic search，避免联合优化中的梯度冲突与训练震荡。

3. **超参数优化对性能至关重要**  
   - Ray Tune + ASHA 显著提升了模型表现，说明自动化调参应成为神经符号系统的标配。

4. **NSR 具备良好的外推能力（OOD Generalization）**  
   - 得益于神经网络学习到的连续潜在表示，适合科学发现场景。

---

### 方法的局限性
1. **特征库的组合爆炸问题**  
   - 交互项数量随维度呈 $O(d^2)$ 增长，限制了在高维系统中的应用。
2. **依赖函数光滑性假设**  
   - 对不连续或分段函数建模困难。
3. **LASSO 的系数偏差**  
   - L1 正则可能导致系数收缩，轻微扭曲真实值。
4. **训练开销较高**  
   - 相比纯稀疏方法（如 SINDy），NSR 运行时间更长（见 Table II：平均 42.7s vs 1.2s）。

---

### 未来工作方向
1. **Adaptive Feature Libraries**  
   - 动态构建或剪枝特征库，减少冗余。
2. **Transformer-Based Symbolic Decoding**  
   - 结合语言模型生成更灵活的符号表达式，超越线性组合限制。
3. **Scaling to High-Dimensional Systems**  
   - 引入注意力机制、低秩分解、层次化特征构造以应对 PDE、生物网络等复杂系统。
4. **Improved Optimization Strategies**  
   - 设计专用于符号任务的损失函数、课程学习策略或元学习迁移超参。
5. **Integration with Scientific Simulators**  
   - 与物理引擎、数值求解器结合，实现从仿真轨迹中自动发现控制方程。
6. **Real-World Applications**  
   - 应用于气候数据、生物信号、工程传感器等真实噪声数据，验证实际价值。

---

> 🔍 **总体评价**：  
> 该论文提出的 **NSR 框架**为 **Scientific Machine Learning** 提供了一个**可扩展、可解释、鲁棒性强**的新范式。它重新定义了神经网络在符号回归中的角色——不是“创造者”，而是“清洁工”和“桥梁”。这一思路有望推动 AI for Science 在物理建模、系统辨识、自动化科学发现等领域的发展。

</details>

---

### 13. [One Policy, Any Budget: Internalizing Budget-Aware Search via Reinforcement Learning](https://arxiv.org/abs/2609.00813)

**Authors**: Xiaowei Sun, Jin Li, Yili Hong, Yikun Fu, Yanghua Xiao  
**Category**: cs.AI  
**Published**: 2026-09-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.00813v1  

#### Abstract
While reinforcement learning has enabled LLM-based search agents to invoke external tools, existing methods train under fixed budgets and cannot adapt when constraints vary at deployment. We propose AnySearch, a framework that enables a single policy to perform budget-aware search under any budget c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：One Policy, Any Budget: Internalizing Budget-Aware Search via Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的基于 **Reinforcement Learning (RL)** 的 LLM 搜索代理（search agents）通常在**固定预算**下训练，无法适应部署时变化的资源约束。例如，在低延迟场景中需要最小化搜索调用，而在深度研究任务中则允许广泛探索。这种“预算刚性”导致模型在面对未见过的预算时表现不佳，要么过度使用资源，要么未能充分利用。

此外，现有方法如 **BATS** 虽引入了预算跟踪机制，但依赖外部追踪器，使得预算感知能力成为外部依赖，而非策略本身的内化能力。

### 提出了什么新方法或新思路
本文提出 **AnySearch**，一个通过渐进式移除训练支架（training scaffold）将**预算感知搜索能力内化到单一策略中**的框架。其核心思想是：

- **两阶段课程学习（Two-Phase Curriculum Learning）**：
  - **Phase I（引导阶段）**：使用显式的 `<budget>` 状态注入和结构化推理提示（structured reasoning prompts），指导代理在预算衰减条件下学习高效的资源分配。
  - **Phase II（自主阶段）**：完全移除训练支架，仅在对话开始时提供总预算 `B`，并通过自适应采样聚焦于表现较差的预算水平，使训练条件与推理一致，消除训练-推理差距。

- **复合奖励设计（Composite Reward）**：
  - 包含绝对效率信号 `R_abs` 和相对效率信号 `R_rel`，结合答案准确性与预算使用效率。
  - 引入**自适应效率权重 `γ_q`**：当查询组准确率高时增强效率信号；当准确率低时减弱效率压力，防止牺牲正确性以追求效率。

### 相比现有方法的优势
| 方面 | 现有方法（如 Search-R1, BATS） | AnySearch |
|------|-------------------------------|---------|
| **预算适应性** | 固定预算训练，无法泛化至新预算 | 单一策略适应任意预算，泛化至训练范围外 |
| **外部依赖** | BATS 需要外部预算追踪器 | 完全内化，推理时无需额外组件 |
| **效率与准确性平衡** | 缺乏动态调节机制 | 自适应权重实现智能权衡 |
| **工具生产力（Tool Productivity）** | 较低，常冗余调用搜索 | 显著更高，更少搜索达成相同精度 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **训练集**：
  - **Natural Questions (NQ)**：通用问答，基于维基百科。
  - **HotpotQA**：多跳问答，含干扰段落，测试复杂推理。
  
- **评测集（共7个）**：
  - **通用 QA**：NQ, TriviaQA*, PopQA*
  - **多跳 QA**：HotpotQA, 2WikiMultiHopQA, MuSiQue**, Bamboogle'
  - （*表示跨域评估，**表示严格多跳要求）

所有实验基于 **2018 Wikipedia dump** 构建检索语料库，使用 **E5** 作为 dense retriever，返回 top-3 文档。

### 实验设置和评估指标
- **骨干模型**：Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Qwen3-4B
- **训练预算上限**：`B_max = 5`
- **RL算法**：GRPO（Generalized Reward Policy Optimization）
- **主要评估指标**：
  - **Exact Match (EM)**：衡量答案准确性。
  - **Tool Productivity (TP)**：每单位搜索调用所能正确回答的问题数，定义为：
    $$
    \text{TP} = \frac{\sum \mathbb{I}\{\text{ans}_i = y_i\}}{\sum c_i}
    $$
    其中 $c_i$ 是第 $i$ 个样本的搜索次数。
  - **总 Token 消耗**：输出 token + 检索文档 token，反映实际成本。

### 基线方法对比
| 类型 | 方法 | 特点 |
|------|------|------|
| **Prompt-based Budget-Aware** | BATS | 注入预算状态但无 RL 优化 |
| **RAG-based** | Search-o1 | 推理时交替思考与搜索，无额外训练 |
| **RL-based** | Search-R1, ZeroSearch, StepSearch | 使用不同 RL 策略优化搜索行为 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 表1：七项 QA 基准上的平均 EM 分数（B=5）

| 方法 | Qwen2.5-7B 平均 | Llama-3.1-8B 平均 | Qwen3-4B 平均 |
|------|------------------|--------------------|---------------|
| BATS | 0.227 | 0.264 | 0.204 |
| Search-o1 | 0.266 | 0.347 | 0.286 |
| Search-R1 | 0.397 | 0.403 | 0.373 |
| StepSearch | 0.403 | 0.410 | 0.383 |
| **AnySearch (Ours)** | **0.431** | **0.448** | **0.407** |

👉 在所有骨干模型上均取得最佳性能，平均提升约 **2.8–3.8个百分点**。

#### 表2：不同设置下的 Tool Productivity 对比

| 设置 | 方法 | Accuracy ↑ | TP ↑ |
|------|------|------------|-------|
| Qwen2.5-7B, B=6 | AnySearch | 0.398 | **0.354** |
| 同上 | Search-R1 | 0.375 | 0.212 |
| 同上 | StepSearch | 0.388 | 0.276 |

👉 AnySearch 实现最高准确率的同时，**TP 提升达 67% 以上**，表明其极高的搜索利用效率。

#### 图3 & 表10：Token 消耗分析
- AnySearch 在所有数据集上实现了**最低的总 token 数量**。
- 检索 token 占总消耗的 **51%-68%**，验证了减少搜索调用是降低开销的关键。

### 与基线方法的对比结果
- **泛化至未见预算（图4）**：
  - 在训练最大预算为 5 的情况下，AnySearch 在 `B=6~8` 上仍持续提升性能，而基线趋于饱和甚至下降。
  - 说明其学会了**按比例调整搜索深度**，而非记忆固定模式。

- **成本-效益帕累托前沿领先**：
  - 在相同预算下，AnySearch 准确率更高；
  - 达到相同准确率所需预算更低（如在 Bamboogle 上，AnySearch @ B=3 ≈ Search-R1 @ B=4）。

### 消融实验结果
#### 表3：工具奖励消融（Qwen3-4B on 2wiki）

| 设置 | Accuracy | TP |
|------|----------|-----|
| Full (Adaptive γ) | **0.375** | **0.342** |
| Fixed γ = max | 0.354 | 0.328 |
| Fixed γ = 0.5max | 0.366 | 0.334 |
| w/o R_tool (γ=0) | 0.348 | 0.226 |
| w/o R_abs | 0.352 | 0.272 |
| w/o R_rel | 0.359 | 0.295 |

✅ 结论：
- 自适应 `γ_q` 至关重要，避免在难问题上过度追求效率。
- `R_abs` 和 `R_rel` 提供互补信号，缺一不可。

#### 图5：模块消融分析
- **支架内化成功**：移除支架后性能不降反升，且重新启用支架无增益 → 表明预算感知已**内化为策略本身**。
- **自适应采样优于均匀采样**：聚焦低预算训练显著提升整体鲁棒性。
- **两阶段课程必要**：仅 Phase I 或 Phase II 均不如完整流程稳定高效。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **预算感知可以被内化**：通过渐进式移除训练支架，AnySearch 成功将预算管理从外部依赖转化为策略内部能力。
2. ✅ **单一策略可适配任意预算**：无需为不同预算重新训练多个模型，极大提升了部署灵活性。
3. ✅ **效率与准确性可协同优化**：自适应奖励机制有效缓解了二者之间的冲突，在高准确区推动效率，在低准确区优先保正确性。
4. ✅ **显著提升工具生产力与降低成本**：相比基线，AnySearch 用更少的搜索调用达到更高的准确率，总 token 开销最低。

### 方法的局限性
1. **预算建模简化**：当前预算为离散的搜索调用次数，未考虑真实世界中的多维成本（延迟、金钱、系统负载等）。
2. **受限于骨干模型能力**：若答案既不在参数知识中也不在检索库中，则无法恢复。
3. **静态知识源**：使用固定的 2018 Wikipedia，未测试时间动态更新环境下的适应能力。

### 未来工作方向
- 将预算扩展为**连续、多目标的成本函数**（如 latency + cost + energy）。
- 探索在**开放网络环境**中进行动态检索与学习。
- 将该框架应用于其他工具调用场景（如代码执行、数据库查询等）。

---

> 🔗 **代码开源地址**：[https://github.com/xwsun01/AnySearch](https://github.com/xwsun01/AnySearch)

</details>

---

### 14. [Deterministic LLM Inference Across GPU Kernels: Power-of-Two INT8 Quantization Scales and the Limits of Tolerance-Based Conformance](https://arxiv.org/abs/2609.00363)

**Authors**: Teng-Ruei Chen  
**Category**: cs.LG  
**Published**: 2026-09-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.00363v1  

#### Abstract
Conformance suites for quantized GEMM kernels ask whether two implementations agree within a tolerance. We measure what such a suite can detect. Injecting nine faults into a reference INT8 pipeline over 8,232 layer--fault--regime cells of Qwen3-1.7B, we find that every one of five epilogue faults --...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Deterministic LLM Inference Across GPU Kernels: Power-of-Two INT8 Quantization Scales and the Limits of Tolerance-Based Conformance*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文聚焦于 **大型语言模型（LLM）在不同 GPU kernel 实现之间进行 INT8 量化推理时缺乏跨核比特级确定性（bitwise determinism）** 的问题。尽管两个实现相同的 scaled INT8 GEMM 接口的 kernel（如 CUTLASS 和 Triton）通常被视为可互换，但先前研究已表明它们在输出上存在差异，影响了推理的可复现性和部署一致性。

此外，现有基于容忍度（tolerance-based）的合规性测试套件（conformance suite）是否真正有效、能否检测出实际缺陷，以及“使用幂次二（power-of-two）量化尺度”作为解决方案是否具备实际部署可行性，均未得到验证。

### 提出了什么新方法或新思路
本论文提出并验证了一种 **通过重新量化（requantization）构建幂次二尺度检查点** 的方法，以实现跨 kernel 的比特级一致推理。其核心思想是：
- 不是简单地重写 scale（scale rewriting），而是从原始浮点权重出发，在约束为幂次二 scale 的条件下重新生成 int8 权重（即 requantization），从而保证 weight-scale 的一致性。
- 利用 **幂次二 scale 下乘法顺序交换不变性**（$ \text{rnd}(x) \cdot 2^k = \text{rnd}(x \cdot 2^k) $），使得 epilogue 阶段的舍入结果唯一，强制要求 bitwise equality 而非容忍区间。

同时，作者设计了一个 **预注册（pre-registered）、两阶段锁定的 fault injection 框架**，首次对一个 conformance suite 的敏感性进行了系统性测量。

### 相比现有方法的优势
- **首次实证验证了 conformance suite 的有效性边界**：揭示了 tolerance-based 测试在检测单精度/舍入类故障上的结构性盲区。
- **纠正误解**：指出此前报告中 +157% 困惑度（perplexity）激增是由于错误构造 probe（仅改 scale 未 requantize 权重）所致，并非幂次二本身的代价。
- **提供可部署方案**：证明 requantized power-of-two checkpoints 可使 CUTLASS 与 Triton 在所有线性层和生成序列上达成字节一致，且困惑度变化极小（-0.28% ~ +0.48%），具备实际部署价值。
- **方法论严谨性**：采用 fault injection + 构造性真值（ground truth by construction）+ 预注册协议，避免了事后拟合偏差。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要基于 **Qwen3 系列模型** 的三个规模：**1.7B、8B 和 14B**。
- 具体使用了从 **Qwen3-1.7B 的 196 个线性层中捕获的实际 int8 激活值（activations）和权重** 进行 fault injection 实验。
- 用于最终端到端生成比较的提示来自 **8 个固定的 pinned prompts**，每条生成 64 个 token。
- 准确率成本评估使用 **WikiText 数据集上的 256 个固定窗口** 进行 perplexity 测量。

### 实验设置和评估指标
#### Fault Injection 设计
- **参考流水线（Reference Pipeline）**：使用 float64 或 INT64 累加器确保 accumulator 结果精确无误，作为 ground truth。
- **注入 9 类故障（fault families）**，覆盖：
  - Epilogue 缺陷（5 类）：如 scale cast 到 bfloat16（F1）、双重舍入（F2）、乘法顺序错误（F3）、截断代替舍入（F4）、融合顺序错误（F5）
  - 前提条件违反（2 类）：INT32 溢出（F6）、超过 float32 精确表示范围（F7）
  - 输入不一致（1 类）：operand mismatch（F9）
  - 控制组：null fault（F8）
- 每类故障在 **两种 scale 制度**（原生 scale vs 幂次二 scale）和 **三种覆盖率**（单元素、1%、全部）下运行，共产生 **8,232 个实验单元（cells）**。

#### Conformance Suite 与预测矩阵
- 使用一个 **7 检查项的 conformance suite**，包括 operand 一致性、overflow 检查、accumulator 一致性、幂次二恒等性、real-scale tolerance（1 ULP）等。
- 构建一个 **77-cell prediction matrix**（最初为 63-cell，经 smoke run 后修正为 77-cell），预先声明每个 check 在每种 fault 下应“触发”、“静默”或“不适用”，并在完整运行前重新锁定。

#### 性能评估
- **比特级一致性**：逐层输出是否完全相同（per-layer bitwise agreement）
- **生成一致性**：greedy token 序列是否字节一致（byte-identical generation）
- **准确率影响**：relative perplexity change（相对困惑度变化）
- **置信区间**：使用 paired cluster bootstrap（256 窗口，10,000 抽样）计算 90% 区间

### 基线方法对比
- **Baseline 1**：原始 checkpoint 下的 CUTLASS vs Triton 推理（非幂次二 scale）
- **Baseline 2**：此前研究中使用 scale rewriting 构造的幂次二 probe（导致 +157% perplexity）
- **本文方法**：requantized power-of-two checkpoints（nearest 和 ceiling 两种策略）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 自有 scale 下 per-layer 一致率 | 幂次二 requantized 下 per-layer 一致率 | 自有 scale 下生成一致率（8 prompts） | 幂次二 requantized 下生成一致率 |
|------|-------------------------------|-------------------------------------|------------------------------------|-------------------------------|
| Qwen3-1.7B | 8 / 196 | **196 / 196** | 0 / 8 | **8 / 8** |
| Qwen3-8B   | 10 / 252 | **252 / 252** | 0 / 8 | **8 / 8** |
| Qwen3-14B  | ——     | ——                                 | 0 / 8 | **8 / 8** |

> 注：14B 因显存限制无法做 per-layer capture，仅有端到端结果。

### 与基线方法的对比结果
- **跨 kernel 一致性**：
  - 原始 checkpoint：几乎无一致性（仅 8/196 层一致）
  - 本文 requantized 幂次二 checkpoint：**所有线性层和生成序列完全一致**
- **准确率损失**：
  - 使用 nearest power-of-two requantization：
    - Qwen3-1.7B: **+0.32%**
    - Qwen3-8B: **-0.28%**（轻微提升，可能为噪声）
    - Qwen3-14B: **+0.48%**
  - 90% 置信区间上限分别为 +0.71%（1.7B）和 +0.76%（14B），仅在 14B 显著偏离零。
- **+157% 困惑度分解**：
  - 实际 requantized 版本仅增加 +0.32%，说明原 +157% 中 **99.8% 是由 weight-scale mismatch 引起的人工 artifact**

### 消融实验结果
- **Fault Injection 结果**（针对 conformance suite）：
  - 对修正后的 77-cell prediction matrix，实现了 **0 false positives 和 0 false negatives**
  - 但 **4/5 的 epilogue faults 完全未被任何 check 检测到**（F1, F2, F3, F5）
  - 唯一被检测到的是 F4（output truncation），且仅在幂次二 scale 下通过 `power-of-two identity` check 触发
  - 所有 precondition faults（F6, F7）和 operand fault（F9）均被可靠捕获
- **Scale 构造方式对比**：
  - `rewriting`（错误方式）→ +157% perplexity
  - `nearest` requantization → +0.32% ~ +0.48%
  - `ceiling` requantization → +0.54%（更高，因放弃剪枝换取分辨率）

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **Tolerance-based conformance 无法检测多数 epilogue 故障**  
   因为单次舍入或精度缺陷最多只会将输出移动 **一个 bfloat16 spacing**，而这是合法浮点差异的最小单位，因此 1-ULP tolerance 天然对其免疫。

2. ✅ **真正的比特级一致性需要移除算术自由度**  
   幂次二 scale 通过数学性质（公式 $ \text{rnd}(x \cdot 2^k) = \text{rnd}(x) \cdot 2^k $）消除了 epilogue 中的乘法顺序自由度，从而使 bitwise equality 成为必要而非期望。

3. ✅ **Requantized 幂次二 checkpoints 是可部署的**  
   它们能在 **极低准确率代价**（< ±0.5% perplexity）下实现 **CUTLASS 与 Triton 的完全比特级一致**，解决了生产环境中 kernel 选择可观测的问题。

4. 🚫 **现有 conformance suite 不应被解释为“kernel interchangeability”的判定工具**  
   它实际只能验证：
   - accumulator 前提条件成立（no overflow, etc.）
   - 两 kernel 接收到相同 operands
   - 输出差异不超过一个 spacing  
   → 因此作者正式撤回（retract）此前关于其决定 kernel 可互换性的说法。

### 方法的局限性
- 实验仅限于 **Qwen3 系列模型** 和 **CUTLASS-Triton 内核对**，其他模型或框架组合尚未验证。
- 未找到满足隔离条件的第二组 kernel pair（经预注册搜索十种引擎后失败）。
- 14B 模型受限于显存（24GB），无法进行 per-layer capture，仅支持端到端验证。
- throughput 仅在 1.7B 上测量，未全面评估性能开销。
- token-level risk check 未被评估（layer-level injection 无法提供 logits margin）。
- fault injection 覆盖范围有限，未包含可能导致 >1 spacing 偏差的新机制。

### 未来工作方向
- 寻找并验证第二组满足条件的 kernel pair（ROCm 的 aiter 路径为候选）。
- 实现在 14B 及以上模型的 per-layer capture，尤其是在 decode 阶段。
- 开展端到端 fault injection 以评估 token-level check 的敏感性。
- 探索超出当前九类故障的新 fault families，特别是那些不修改 operands 却能引起 >1 spacing 偏差的机制。
- 将该方法扩展至其他量化格式（如 FP8）或其他硬件平台。

--- 

> **总结一句话**：  
> 本文证明，**基于 tolerance 的测试无法保障 INT8 推理 kernel 的可互换性**；唯有通过 **requantized power-of-two scales** 消除 epilogue 的算术歧义，才能实现可部署的跨 kernel 比特级确定性，且代价极小。这不仅是技术方案，更是一种对“数值合规性”本质理解的深化。

</details>

---

### 15. [CacheBridge: Efficient Cross-Model KV Cache Transfer](https://arxiv.org/abs/2609.00891)

**Authors**: Xingyu Qu, Siyuan Lu, Zhiyu Chen, Sheng Wang, Tao Lin  
**Category**: cs.AI  
**Published**: 2026-09-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.00891v1  

#### Abstract
Sharing context between LLMs in a multi-model system requires the receiving model to prefill the shared prefix because KV caches are model-specific. Recent closed-form cross-model KV transfer, hereafter Full-Head Mapping, avoids this replay by fitting a training-free affine mapper from source to tar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CacheBridge: Efficient Cross-Model KV Cache Transfer**

---

## 1. **论文的主要贡献和创新点**

### **解决的问题**
在多模型系统中，当控制权从一个大语言模型（LLM）传递到另一个时，接收模型无法直接复用发送方的 **KV Cache**（因其为模型特有），必须重新执行 **prefill** 阶段以重建共享前缀的上下文。这导致显著的计算冗余，尤其在长上下文场景下成为延迟瓶颈。

现有方法 **FULL-HEAD MAPPING**（Heo et al., 2026）提出了一种无需训练的仿射映射器（affine mapper）实现跨模型 KV Cache 转移，避免了目标端的 prefill。然而该方法存在以下问题：
- **模型敏感性**：在不同架构间性能不稳定，某些转移方向准确率大幅下降。
- **高开销**：映射器存储和应用成本随支持层数线性增长。
- **构造低效**：mapper 构造过程中需反复 materialize 中间张量，内存开销大。

---

### **提出的新方法：CACHEBRIDGE**
作者提出 **CACHEBRIDGE**，通过协同设计 **mapper 支持结构、校准目标和构造路径**，在保持闭式（closed-form）仿射接口的前提下，显著提升效率与鲁棒性。其三大核心组件为：

#### **S1: HEAD-LOCAL —— 结构化头局部性**
- 将每个目标 KV 头仅映射到一个**架构索引对齐的源 KV 头**（而非全连接所有源头）。
- 显著减少参数量（系数数量降低 8×），降低 mapper 存储与推理开销。
- 利用 **Grouped-Query Attention** 中的 KV 组先验进行对齐。

#### **S2: ATTN-REPAIR —— 注意力对齐的校准**
- 不再均匀加权 KV 重建误差，而是根据**接收模型注意力机制对查询的敏感度**动态加权残差。
- 引入基于 **Jacobian 块迹（trace）** 的敏感度度量，并结合有效样本大小（effective sample size）防止权重过度集中。
- 更好地匹配实际生成质量（continuation quality），而非单纯优化坐标级 R²。

#### **S3: FUSED-FIT —— 融合式高效构造**
- 设计新的 **融合 GPU 内核（fused GPU kernel）**，在 bounded 内存面板中直接流式构建加权充分统计量（weighted sufficient statistics）。
- 避免中间张量 materialization，极大加速 mapper 构造过程。

---

### **相比现有方法的优势**
| 维度 | FULL-HEAD MAPPING | CACHEBRIDGE |
|------|-------------------|-------------|
| 参数量 | 高（全头连接） | ↓ 8× 减少 |
| 存储 | 高（如 4.296 GB） | ↓ 降至 0.538 GB |
| 应用延迟 | 高，随层线性增长 | ↓ 最快提速 3.0× |
| 构造时间 | 长（92.63 秒） | ↓ 缩短至 8.63 秒（10.7× 加速） |
| 准确率稳定性 | 差（在 Ministral 上崩溃） | ✔️ 完全恢复 |
| 校准目标 | 坐标级 R² | ✔️ 注意力敏感度对齐 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **FineWeb-Edu**：用于 calibration 的 500 条长度为 1,024 的文本序列（每 4 个 token 采样一次，共约 128k 位置）。
- **评估任务子集**：
  - **HellaSwag**
  - **ARC-Challenge**
  - **WinoGrande**
  - **MMLU**
- **长上下文 NLL 测试**：使用 16 条固定长度为 4,096 的文本流，teacher-forced 推理 128 步。

---

### **实验设置**
- **模型对（Transfer Directions）**：
  1. **Ministral 3 3B → 14B**
  2. **Ministral 3 8B → 14B**
  3. **Qwen3 14B → 32B**
- 所有模型均为 **GQA 架构，8 个 KV heads**。
- **选定层数 k**：分别为 20、12、8。
- **Ridge 回归正则化 λ = 0.01**，沿用基线设定。
- **ATTN-REPAIR 边界数**：32 个对数间隔的前缀边界（token 12 ~ 1023）。

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **HellaSwag Accuracy** | 主要任务准确率 |
| **Target Retention** | 转移后准确率 / 原生目标模型准确率（均值） |
| **K/V R²** | KV 缓存重建的决定系数 |
| **NLL@4K** | 在 4K 长度上下文下的负对数似然 |
| **Mapper Storage** | 序列化 mapper 文件大小 |
| **Application Latency** | 映射应用耗时（ms） |
| **Construction Time** | mapper 构造总时间（秒） |

---

### **基线方法对比**
- **FULL-HEAD MAPPING**（Heo et al., 2026）：当前最优闭式跨模型 KV 转移方法。
- **CACHEBRIDGE**：本文方法，包含三部分联合优化。
- 消融实验还包括：
  - **Block-PCA 控制组**（压缩维度但不保证结构对齐）
  - **Cyclic Head Shift**（错位头对应关系）

---

## 3. **主要实验结果和性能指标**

### **关键性能数据汇总**

| 指标 | 结果 |
|------|------|
| **Ministral 3B→14B HellaSwag** | ↑ 从 52.2% → **72.6%**（+20.4 pts） |
| **Ministral 8B→14B HellaSwag** | ↑ 从 44.4% → **76.0%**（+31.6 pts） |
| **Qwen3 14B→32B 平均保留率** | 99.83%（vs. FULL-HEAD 的 99.72%） |
| **Mapper 存储（Qwen3）** | ↓ 从 **4.296 GB → 0.538 GB**（↓8×） |
| **Mapper 应用延迟（1k prefix）** | ↓ 从 65.12 ms → **21.66 ms**（↑3.0×） |
| **Mapper 构造时间（500 seq）** | ↓ 从 92.63 s → **8.63 s**（↑10.7×） |
| **校准数据效率** | 用 **1/10 数据** 即可匹敌 FULL-HEAD 的平均保留率 |

---

### **与基线方法对比结果**

#### ✅ **Ministral 方向完全修复性能崩溃**
- FULL-HEAD 在两个 Ministral 转移方向上严重失效（<50% HellaSwag）。
- CACHEBRIDGE 成功恢复至接近原生模型水平（>72%），验证了 **HEAD-LOCAL 对架构差异的鲁棒性**。

#### ✅ **Qwen3 上保持高保真**
- 在已表现良好的 Qwen3 上，CACHEBRIDGE 不仅未退化，反而略微提升平均保留率（99.83% vs 99.72%）。
- NLL@4K 从 2.446 降至 **2.350**，表明更优的长上下文建模能力。

#### ✅ **效率全面领先**
- **参数量减少 8×** → 存储与通信成本大幅下降。
- **应用速度提升最多达 3.0×**，尤其在长前缀下优势明显。
- **构造时间缩短 10.7×**，适合快速部署与迭代。

---

### **消融实验结果（见 Table 2 & Figure 5）**

| 配置 | 主任务保留率 | Δ vs HEAD-LOCAL | KR² / VR² | NLL@4K |
|------|--------------|------------------|-----------|--------|
| Block-PCA（降维控制） | 81.30% | -13.97 | 0.644 / 0.514 | 2.943 |
| Cyclic Shift（非对齐头） | 79.76% | -15.50 | 0.620 / 0.473 | 3.094 |
| HEAD-LOCAL-only | 95.26% | 0.00 | 0.678 / 0.655 | 2.446 |
| **CACHEBRIDGE（完整）** | **96.51%** | **+1.25** | 0.672 / 0.654 | **2.350** |

> 🔍 发现：
> - **结构对齐 > 单纯降维**：即使压缩到相同参数量，错位或无结构的映射仍严重损害性能。
> - **注意力加权带来额外增益**：尽管 K/V R² 几乎不变，但 **primary retention 提升 1.25 pts，NLL 显著下降**，说明传统 R² 不足以反映生成质量。

---

## 4. **关键结论和发现**

### **主要发现**
1. **全头连接不可靠**：FULL-HEAD MAPPING 的“全连接”假设忽略了模型架构中的结构性对应关系，在跨尺度或残差宽度变化大的情况下会引入噪声方向，导致性能崩溃。
2. **结构先验至关重要**：利用 GQA 中的 **KV head ownership 先验** 进行对齐（HEAD-LOCAL），是实现稳定迁移的关键。
3. **重建误差 ≠ 生成误差**：KV 坐标级拟合（R²）不能代表最终输出漂移；应依据 **attention sensitivity** 动态加权（ATTN-REPAIR）。
4. **静态支持可编译优化**：top-k 层选择虽不规则，但一旦确定即可通过 **fused kernel** 流式处理，避免内存爆炸（FUSED-FIT）。

---

### **方法的局限性**
1. **同家族模型限制**：目前仅在 **同一模型族内**（Ministral/Qwen3）验证，跨家族（如 Llama → Qwen）尚未测试。
2. **注意力机制依赖**：假设所有模型使用 **dense grouped-query attention** 且 KV head 数匹配；对稀疏、滑动窗口、线性注意力等扩展未知。
3. **单轮延续评估**：仅评测 immediate continuation（如多项选择题、teacher-forced NLL），未考察多轮对话中错误累积的影响。

---

### **未来工作方向**
- 扩展至 **跨家族、异构架构** 的 KV 映射。
- 探索 **自适应 head correspondence learning** 机制，应对 head 数不匹配场景。
- 引入 **multi-turn evaluation protocol**，衡量长期生成一致性。
- 将 CACHEBRIDGE 集成进 **multi-agent LLM 系统**（如 AutoGen、RouteLLM）以实测端到端收益。

---

> 📌 **一句话总结**：  
> **CACHEBRIDGE 通过结构对齐、注意力感知校准与融合构造，实现了高效、鲁棒、低成本的跨模型 KV Cache 转移，在修复 FULL-HEAD MAPPING 故障的同时，将存储、延迟、构造时间全面优化一个数量级以上。**

</details>

---

### 16. [Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs](https://arxiv.org/abs/2609.01117)

**Authors**: Zhaoliang Chen, Jie Fu  
**Category**: cs.AI  
**Published**: 2026-09-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.01117v1  

#### Abstract
Chain-of-thought reasoning unfolds in discrete token space: each step is committed as text, errors propagate, and eliciting good traces presupposes traces to imitate. Reasoning instead in a model's continuous representation space - where intermediate states are vectors rather than words - sidesteps ...

---

### 17. [Ready to Speak: Aligning LLMs for TTS-Friendly Text Generation](https://arxiv.org/abs/2609.01246)

**Authors**: Thibaut Thonet, Jos Rozen, Laurent Besacier  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.01246v1  

#### Abstract
Current Large Language Models (LLMs) are primarily optimized for written text, often producing outputs that are grammatically correct and helpful yet poorly suited for spoken delivery via Text-to-Speech (TTS). In this work, we study how to make LLMs natively generate TTS-friendly text, which we fram...

---

### 18. [AdaptNTK: Adaptive Uncertainty Quantification and Active Learning for Neural Network Potentials](https://arxiv.org/abs/2609.00488)

**Authors**: Prajwal Ananth, Shuwen Yue  
**Category**: cs.LG  
**Published**: 2026-09-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.00488v1  

#### Abstract
Machine learning interatomic potentials bridge the gap between quantum chemical precision and classical computational speed, enabling molecular dynamics simulations with first-principles accuracy. Their reliability is often improved through active learning, which iteratively expands the training set...

---

### 19. [SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers](https://arxiv.org/abs/2609.01343)

**Authors**: Shaowen Wang, Ge Zhang, Kairong Luo, Yuhao Wu, Shaofan Liu, Jiaheng Liu, Wenhao Huang, Shen Yan, Jian Li  
**Category**: cs.LG  
**Published**: 2026-09-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.01343v1  

#### Abstract
Looped Transformers increase effective depth by iterating a shared block of layers, but most evaluations compare at fixed model size, conflating architectural advantage with extra FLOPs. We study looping on Mixture-of-Experts Transformers while closely matching per-token FLOPs, total non-embedding p...

---

### 20. [Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems](https://arxiv.org/abs/2609.00237)

**Authors**: Rakibul Hasan Rajib, Mengxing Zheng, Qian Lou  
**Category**: cs.AI  
**Published**: 2026-09-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.00237v1  

#### Abstract
Large language model (LLM)-based multi-agent systems tackle complex reasoning by orchestrating how multiple agents are configured and how they collaborate. A central challenge is to adapt orchestration to the evolving collaboration state. Routing from the query alone cannot adapt to intermediate pro...

---

### 21. [FractalNet-Based Heterogeneous Federated Learning for Orbital Edge Intelligence in Satellite Mega-Constellations: A Wildfire Case Study](https://arxiv.org/abs/2609.00875)

**Authors**: Sai Puppala, Koushik Sinha  
**Category**: cs.AI  
**Published**: 2026-09-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.00875v1  

#### Abstract
Satellite mega-constellations are emerging as large-scale sensing, communication, and computation fabrics, yet their learning architectures remain largely inherited from terrestrial federated learning and ground-centric mission operations--- ill-suited to satellites that differ by orders of magnitud...

---

### 22. [Beneath the Diff: Diagnosing and Mitigating Algorithmic Mode Collapse in Code-Level Autonomous Research Loops](https://arxiv.org/abs/2609.00077)

**Authors**: Bowei He, Weixu Zhang, Yili Jin, Xue Liu  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.00077v1  

#### Abstract
Code-level autonomous research loops (ARLs) have recently emerged as a concrete object of study in automated machine learning research. In such loops, an LLM agent proposes modifications to an experimental training pipeline, executes the modified pipeline, and retains edits that improve a verifiable...

---

### 23. [Subword Segmental BabyLMs: Learning to Tokenise for Sample-Efficient Pretraining](https://arxiv.org/abs/2609.01151)

**Authors**: Francois Meyer  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.01151v1  

#### Abstract
In the standard LM training pipeline, subword tokenisation is applied as a preprocessing step. Subword segmental language modelling is an alternative paradigm in which tokenisation is learned during training, allowing the model to discover subword units that optimise its training objective. In this ...

---

### 24. [CaRL-EM: Cost-Aware Reinforcement Learning for Entity Matching with LLMs](https://arxiv.org/abs/2609.01195)

**Authors**: Chaohui Guo, Michel Klein, Zhisheng Huang  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.01195v1  

#### Abstract
Entity matching (EM) requires fine-grained contextual understanding and domain knowledge. Recent work shows that large language models (LLMs) can serve as strong matchers across domains, but most methods either make independent pairwise decisions or rely on manually designed composite pipelines, thu...

---

### 25. [StudentSim: Training LLM-based Student Simulators](https://arxiv.org/abs/2609.01591)

**Authors**: Ke Yang, Chenglong Wang, Michel Galley, Chandan Singh, Jeevana Priya Inala, ChengXiang Zhai, Jianfeng Gao  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.01591v1  

#### Abstract
AI tutors are most useful when they adapt to each student's strengths, weaknesses, and preferred guidance, but evidence about which guidance works for which student is sparse, slow, and costly to collect from real learners. Student simulators can provide this signal as a proxy, yet existing approach...

---

### 26. [Triple-Bottom-Line Sustainability of Language Models for Edge AI: A Comparison Between SLMs and Quantized LLMs](https://arxiv.org/abs/2609.00665)

**Authors**: Jainil Dharmil Shah  
**Category**: cs.AI  
**Published**: 2026-09-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.00665v1  

#### Abstract
Edge-AI model selection is commonly driven by one isolated metric - accuracy, latency, memory, energy, or safety, even though a deployable language model must balance all five. Our work focuses on answering the question whether na- tively trained small language models (SLMs) or large language models...

---

### 27. [Reinforcement Learning Enhanced LLM Agents for Complex Vehicle Routing Problems](https://arxiv.org/abs/2609.00859)

**Authors**: Yi Chen, Zikang Yu, Jiahai Wang, Jinbiao Chen, Jianpeng Zhou, Zizhen Zhang  
**Category**: cs.AI  
**Published**: 2026-09-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.00859v1  

#### Abstract
Vehicle Routing Problems (VRPs) are fundamental combinatorial optimization problems with widespread applications in various scenarios. The advanced optimization solvers can effectively solve such problems. However, modeling complex VRP variants for solvers often requires substantial domain expertise...

---

### 28. [ARISE-RL: Agentic Rubric-Grounded Iterative Self-Evolution with Reinforcement Learning](https://arxiv.org/abs/2609.01058)

**Authors**: Fanrui Zhang, Ruixue Ding, Qiang Zhang, Xi Chen, Boli Chen, Shihang Wang, Qiuchen Wang, Hongmin Zhan, Jinxin Bian, Li xingchao, Peijin Zheng, Hao cheng, Pengjun Xie, Kaipeng Zhang, Jiawei Liu, Zheng-Jun Zha  
**Category**: cs.AI  
**Published**: 2026-09-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.01058v1  

#### Abstract
Training open-ended agents via reinforcement learning (RL) is hindered by the lack of verifiable gold answers and scalable rubrics. Moreover, even near the model's capability boundary, long-horizon open-ended agentic tasks often yield brittle and unstable rewards, resulting in weak or noisy rollout ...

---

### 29. [Enoki: Efficient Multi-Level Hallucination Detection](https://arxiv.org/abs/2609.00581)

**Authors**: Elisei Rykov, Timur Ionov, Nikolay Ivanov, Maksim Savkin, Maksim Makarenko, Alexander Panchenko, Vasily Konovalov, Julia Belikova  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.00581v1  

#### Abstract
Ensuring factuality remains a critical challenge for deploying LLMs in high-stakes settings. Existing hallucination detectors usually operate at a single level: claim-level methods provide interpretable factual units, while span-level methods localize unsupported text. Bridging these views is costly...

---

### 30. [Quit While You're Ahead: Quit for Efficient Candidate Generation in Machine Translation Reranking](https://arxiv.org/abs/2609.00588)

**Authors**: Guangyu Chen, Boxuan Lyu, Hidetaka Kamigaito, Kotaro Funakoshi, Manabu Okumura  
**Category**: cs.CL  
**Published**: 2026-09-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.00588v1  

#### Abstract
Reranking methods, such as Minimum Bayes Risk (MBR) decoding and Quality Estimation (QE) reranking, are widely used in modern neural machine translation (NMT) to select an output from a set of candidate hypotheses. However, the performance gains come at the cost of high inference latency. Existing a...

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

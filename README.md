# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-01 10:44:55 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Efficient Diffusion LLMs via Temporal-Spatial Parallel Decoding and Confidence Extrapolation](https://arxiv.org/abs/2605.30753)

**Authors**: Zekai Li, Ji Liu, Yiqing Huang, Ziqiong Liu, Dong Li, Emad Barsoum  
**Category**: cs.CL  
**Published**: 2026-06-01  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.30753v1  

#### Abstract
Diffusion-based large language models (dLLMs) support parallel text generation via iterative denoising, yet inference remains latency-heavy because many steps are spent on redundant refinement and repeated remasking of tokens whose final values are already determined. Prior acceleration methods main...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Diffusion LLMs via Temporal-Spatial Parallel Decoding and Confidence Extrapolation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
扩散型大语言模型（dLLMs）虽然支持并行生成，但由于**冗余的去噪迭代**和**重复对已确定 token 的重新掩码（remasking）**，推理延迟仍然很高。现有加速方法依赖于**步级局部置信度启发式规则**或**固定调度策略**，这些方法在不同提示（prompt）和任务下表现不稳定，且忽略了序列内部的**位置效应（positional effects）** 和**时序动态性（temporal dynamics）**。

---

### **提出的新方法与新思路**
本文将扩散解码建模为一个**动态控制问题**，而非一系列独立的阈值判断，并提出了两个核心组件：

#### **(1) Temporal-Spatial Parallel Decoding (TSPD)**
- 引入一个轻量级的**时序-空间控制器（temporal-spatial controller）**。
- 利用每个 token 的**轨迹特征（trace features）**，包括：
  - **置信度（confidence）**
  - **熵（entropy）**
  - **动量（momentum）**
  - **相对位置（relative position）**
- 结合 token 的历史演化路径和其在序列中的位置，判断是否可以安全地“固定”该 token，避免后续不必要的更新。

#### **(2) Confidence Extrapolation (CE)**
- 一种**无需训练、即插即用**的状态空间模块。
- 将置信度演化建模为带噪声的状态转移过程（类似 Kalman Filter），预测未来几步的置信趋势及其不确定性。
- 支持**风险感知的前瞻决策（risk-aware look-ahead）**，提前承诺 token，尤其适用于缓慢上升但尚未达阈值的稳定趋势。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 决策依据 | 单步局部统计量（如当前置信 > 0.9） | 多步轨迹 + 位置上下文 |
| 控制方式 | 被动等待（passive waiting） | 主动预测（proactive forecasting） |
| 鲁棒性 | 对 prompt/位置敏感，易误判 | 更强泛化能力，减少早停或冗余迭代 |
| 兼容性 | 多数需修改训练流程 | 完全训练免费，兼容 KV Cache 等系统优化 |

> ✅ **优势总结**：TSPD 提供更鲁棒的 token 固定策略；CE 实现前瞻性加速；两者协同显著减少无效步骤，提升端到端吞吐量。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个代表性基准上进行评估，覆盖多种任务类型：
- **GSM8K**（5-shot）：数学应用题推理
- **MATH**（4-shot）：复杂数学问题
- **HumanEval**（0-shot）：代码生成
- **MBPP**（3-shot）：Python 编程任务

此外还在 **Dream-7B** 和 **LLaDA-MoE** 架构上验证通用性。

---

### **实验设置**
- **主干模型**：LLaDA-8B-Instruct（代表性 dLLM）
- **生成长度**：256 / 512 / 1024 tokens
- **块大小（block size）**：32
- **最大去噪步数（K）**：256
- **批大小**：1（greedy decoding）
- **硬件平台**：NVIDIA A100 80GB GPU
- **实现框架**：PyTorch + CUDA，FP16 推理

---

### **评估指标**
| 指标 | 含义 |
|------|------|
| **Tokens Per Second (TPS)** | 每秒生成 token 数，衡量效率 |
| **Speedup (×)** | 相对于 vanilla dLLM 的加速比 |
| **Accuracy (Acc.)** | 任务特定准确率，衡量质量保留能力 |

---

### **基线方法对比**
| 方法 | 是否使用 KV Cache | 类型 |
|------|------------------|------|
| **Vanilla** | ❌ | 原始 dLLM |
| **Fast-dLLM** | ✅ | 基于缓存 + 并行解码 |
| **Credit Decoding (CD)** | ✅ | 迹分机制 |
| **Prophet** | ❌ | 启发式阈值 |
| **Learn2PD** | ❌ | 学习型并行解码策略 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（LLaDA-8B-Instruct, 256 tokens）**

| 方法 | TPS | Speedup | Accuracy |
|------|-----|---------|----------|
| Vanilla | 6.9 | 1.0× | 79.3 |
| Fast-dLLM | 53.8 | 7.8× | 78.5 |
| CD | 54.5 | 7.9× | 78.7 |
| Prophet | 11.0 | 1.6× | 79.4 |
| Learn2PD | 29.0 | 4.2× | 79.1 |
| **Ours (w/o KV)** | **34.2** | **5.0×** | **79.4** |
| **Ours (w/ KV)** | **77.3** | **11.2×** | **78.8** |

> 🔥 在 GSM8K 上实现 **11.2× 加速**，同时保持接近原始模型的准确性。

---

### **与其他 dLLM 架构的兼容性测试**

| 模型 | 方法 | TPS | Speedup | Accuracy |
|------|------|-----|---------|----------|
| Dream-7B | Ours (w/ KV) | 69.2 | 7.6× | 75.0 |
| LLaDA-MoE | Ours (w/ KV) | 21.2 | 5.1× | 75.2 |

✅ 表明方法具有良好的架构通用性，适用于 MoE 和其他 dLLM 变体。

---

### **长序列生成性能（Scaling Behavior）**

| 生成长度 | 方法 | TPS | Speedup | Accuracy |
|--------|------|-----|---------|----------|
| 512 | Ours (w/ KV) | 79.0 | 24.7× | 77.6 |
| 1024 | Ours (w/ KV) | **64.1** | **58.3×** | 78.5 |

> 📈 **加速比随长度增加而提升**，说明冗余迭代在长输出中更为严重，本方法收益更大。

---

### **消融实验结果（Ablation Studies）**

#### **TSPD 与 CE 的独立贡献（GSM8K, 256 tokens）**
| 方法 | TPS | Speedup | Accuracy |
|------|-----|---------|----------|
| Ours (full) | 34.2 | 5.0× | 79.4 |
| w/o TSPD | 20.0 | 2.9× | 79.1 |
| w/o CE | 30.3 | 4.4× | 79.5 |

➡️ **TSPD 是主要加速来源**，CE 提供额外增益。

---

#### **TSPD 特征重要性分析**
| 移除特征 | TPS | Speedup | Accuracy |
|--------|-----|---------|----------|
| 原始 TSPD | 30.3 | 4.4× | 79.5 |
| w/o confidence | 19.3 | 2.8× | 78.8 |
| w/o entropy | 28.3 | 4.1× | 79.0 |
| w/o momentum | 29.7 | 4.3× | 79.2 |
| w/o position | 31.1 | 4.5× | 78.5 |

➡️ **置信度最关键**，位置信息虽轻微降低速度但显著提升准确性（防止右端 token 早停）。

---

#### **CE 与不同控制器组合效果**
| 控制器 | +CE？ | TPS | Speedup |
|--------|-------|-----|--------|
| TSPD | ❌ | 30.3 | 4.4× |
| TSPD | ✅ | 34.2 | 5.0× |
| Learn2PD | ✅ | 32.4 | 4.7× |
| Fast-dLLM | ✅ | 64.2 | 9.3× |

✅ **CE 是控制器无关的插件**，可广泛增强各类解码策略。

---

## **4. 关键结论和发现**

### **主要发现**
1. **扩散解码应视为动态控制问题**，仅靠单步置信度不足以可靠判断收敛。
2. **token-wise 轨迹（trace）蕴含丰富稳定性信号**，结合位置信息可大幅提升固定决策的鲁棒性。
3. **未来置信趋势可被有效外推**，通过状态空间模型进行风险可控的前瞻承诺是可行且高效的。
4. **TSPD + CE 显著减少冗余去噪步**，带来高达 **58.3× 的端到端加速**，且几乎无损精度。
5. **方法完全兼容 KV Cache**，二者叠加可进一步放大加速效果（如 5.0× → 11.2×）。

---

### **方法的局限性**
- **依赖轨迹特征的质量**：若基础 dLLM 输出波动剧烈，CE 的预测可能不可靠。
- **CE 超参数需调优**：如 coverage threshold `T` 和最大前瞻步 `H` 影响速度-质量权衡（见 Table 12）。
- **目前基于 greedy decoding**：未探索采样（sampling）场景下的行为稳定性。
- **控制器需少量训练**：尽管参数极少（仅 ~2K），但仍非完全免训练（CE 是免训练的）。

---

### **未来工作方向**
1. 扩展至 **采样式生成（stochastic decoding）** 场景，研究多样性与稳定性平衡。
2. 探索 **自适应超参数调节机制**，使 CE 能自动适配不同任务难度。
3. 将 TSPD 思路应用于 **vision-language 或语音扩散模型**。
4. 设计 **完全免训练的 trace-aware 控制器**，进一步降低部署门槛。
5. 结合 **alignment/watermarking 技术**，确保高速生成仍符合安全规范。

---

> ✅ **总体评价**：本文提出的 **TSPD + CE** 框架为 dLLM 推理加速提供了**新范式**——从“被动响应”转向“主动预测”，兼具高性能、高兼容性和强鲁棒性，是迈向实用化扩散语言模型的重要一步。

</details>

---

### 2. [Speculative Pipeline Decoding: Higher-Accruacy and Zero-Bubble Speculation via Pipeline Parallelism](https://arxiv.org/abs/2605.30852)

**Authors**: Yijiong Yu, Huazheng Wang, Shuai Yuan, Ruilong Ren, Ji Pei  
**Category**: cs.CL  
**Published**: 2026-06-01  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.30852v1  

#### Abstract
Speculative Decoding (SD) accelerates low-concurrency LLM inference by employing a draft-then-verify paradigm. However, mainstream methods typically rely on multi-token prediction, which introduces escalating prediction difficulty and serial drafting latency. To address these, we propose Speculative...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Speculative Pipeline Decoding: Higher-Accuracy and Zero-Bubble Speculation via Pipeline Parallelism*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Speculative Decoding (SD)** 方法（如 EAGLE）虽然能加速大语言模型（LLM）推理，但仍存在两个根本性瓶颈：

1. **Compounding Prediction Difficulty（预测难度累积）**  
   多token预测范式中，越往后预测的token依赖于未验证的浅层隐藏状态，导致分布偏移（OOD），接受率随预测长度增加而急剧下降。

2. **Latency Overhead and Mutual Waiting（延迟开销与相互等待）**  
   串行生成draft token会导致target model空闲，即使并行化（如P-EAGLE）也会引入训练复杂度上升或精度损失。

此外，PPSD虽尝试利用pipeline parallelism，但其speculation模块仅基于第一阶段浅层特征，且串行执行，无法实现高接受率与零延迟。

---

### 提出的新方法：Speculative Pipeline Decoding (SPD)

SPD是一种全新的 **speculative decoding 范式**，通过重构LLM为n-stage pipeline架构，并设计一个并行运行的 **Speculation Module** 来持续填充流水线。

#### 核心创新点：

- ✅ **Multi-Depth Feature Aggregation（多深度特征聚合）**  
  聚合当前在pipeline中各阶段处理中的token的中间隐藏状态（来自不同layer depth），以及已验证token的完整表示。这使得speculation始终基于更丰富、对齐的目标模型上下文，显著提升准确率。

- ✅ **Zero-Bubble Parallel Execution（零气泡并行执行）**  
  将Speculation Module的执行窗口前移，在pipeline输入时刻即开始推测下一个token，使其与target model的pipeline step完全并行。只要speculation latency ≤ 单stage耗时，即可完全掩盖延迟，实现“零等待”。

- ✅ **Constant Prediction Difficulty Bound**  
  最大特征不完整性被限制在pipeline长度 $ n $ 内，避免传统方法中随draft length增长而无限累积误差。

- ✅ **无需选择draft length超参**  
  pipeline机制天然规避了传统SD中敏感的draft length调参问题。

---

### 相比现有方法的优势

| 方面 | SPD | EAGLE-3 | PPSD |
|------|-----|--------|------|
| 预测准确性 | ✅ 高（融合多深度特征） | ⚠️ 中等（依赖长程预测） | ❌ 低（仅用首层浅特征） |
| 并行性 | ✅ 完全并行（zero bubble） | ⚠️ 串行或部分并行 | ❌ 串行执行 |
| 可扩展性 | ✅ 支持更深speculation网络（无延迟惩罚） | ⚠️ 深网络增加延迟 | ❌ 深网络会破坏并行性 |
| 接受率稳定性 | ✅ 不随pipeline加深恶化 | ❌ 随draft length增长迅速下降 | ⚠️ 浅层限制导致低接受率 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MT-Bench**：多轮对话任务，评估开放生成能力。
- **GSM8K**：数学推理任务，测试逻辑连贯性。
- **HumanEval**：代码补全任务，衡量确定性模式下的表现。

训练数据混合自：
- ShareGPT-70k
- UltraChat-200k
- SmolTalk & SmolTalk-Chinese

共约120万样本，最大序列长度2048。

---

### 实验设置

- **目标模型**：Qwen3.5-4B 和 Qwen3.5-9B（均为 L=32 层）
- **Pipeline stages (n)**：4, 8, 16
- **Speculation Module层数 (Ls)**：1, 2, 4（SPD可自由扩展而不影响延迟）
- **采样方式**：
  - Greedy decoding ($ T=0 $)
  - Random sampling ($ T=1, \text{top-k}=50, \text{top-p}=1.0 $)
- **Draft Tree支持**：宽度 $ W=1 $（单路径）和 $ W=4 $（保留Top4分支）

---

### 评估指标

由于当前实现基于原生PyTorch，未进行底层优化（如Triton kernel、vLLM集成），因此不报告端到端wall-clock时间，而是采用理论加速比分析：

#### 🔹 **Equivalent Acceptance Length ($ L_{\text{acc}} $)**  
定义为：
$$
L_{\text{acc}} = \frac{N}{K} \cdot n
$$
其中：
- $ N $：总生成token数
- $ K $：实际pipeline步数（含flush惩罚）
- $ n $：pipeline stage数

该指标直接反映**理论速度提升上限**，且对于SPD有：
> $ S_{\text{SPD}} = L_{\text{acc}} $

而对于其他方法（如EAGLE-3、PPSD），需额外计入speculation计算开销，故其理论速度提升严格低于 $ L_{\text{acc}} $。

---

### 基线方法对比

| 方法 | 类型 | 特点 |
|------|------|------|
| **EAGLE-3** | Multi-token Self-Speculative Decoding | 使用轻量头+多层特征融合预测多个future tokens；但存在串行延迟与compounding error |
| **PPSD** | Pipeline-based Speculation | 将LLM分段，但speculation仅基于第一阶段输出，且串行执行，难以扩展 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见Table 1）

| Model | Method | T=0, W=1 | T=1, W=1 | T=0, W=4 | T=1, W=4 |
|-------|--------|----------|----------|----------|----------|
| Qwen3.5-4B | **Ours (SPD, n=8, Ls=4)** | **2.91** | **3.43** | — | — |
| Qwen3.5-9B | **Ours (SPD, n=16, Ls=2)** | **3.24** | **3.83** | — | — |

> 注：SPD的结果直接以 $ L_{\text{acc}} $ 表示，等于理论speedup。

---

### 与基线方法对比结果

#### ✅ 在大多数配置下，SPD取得最高理论speedup：
- 在 **Qwen3.5-9B + HumanEval + T=1, W=4** 上，SPD达到 **$ L_{\text{acc}} = 5.97 $**，远超EAGLE-3的3.98/2.71（即理论speedup仅为2.71）。
- 在 **GSM8K 和 MT-Bench** 上也普遍优于EAGLE-3和PPSD。

#### ❗例外情况：
- 在 **greedy decoding + draft tree (W=4)** 下，EAGLE-3因原始接受长度更高，在某些场景略胜（如HumanEval上EAGLE-3达7.30/4.97，SPD为5.97）。但这是以更高计算代价换来的。

#### 📉 PPSD表现最差：
- 接受率低（~1.6–2.2），且随着stage增多反而下降（如Qwen3.5-9B上从2.17降至2.10），说明其不可扩展。

---

### 消融实验结果

#### 🔹 **Input States vs. Output States（执行时机消融）**

| Method (n=16, Ls=2) | Raw $ L_{\text{acc}} $ | Theoretical Speedup |
|---------------------|-------------------------|----------------------|
| Using **output states**（串行） | 3.66 → 4.78 | ↓ 1.83 → 2.39 |
| Using **input states**（并行，本文方法） | 3.28 | ↑ **3.83** |

👉 结果表明：尽管使用output states能获得更高的raw接受长度，但由于必须等待target model完成，重新引入了latency bubble，最终理论speedup大幅下降。

✅ **证明了“early-start + input-state speculation”设计的优越性**。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Pipeline Parallelism 是下一代 Speculative Decoding 的理想架构基础**  
   SPD首次真正释放了pipeline并行潜力，实现了高精度与零延迟的统一。

2. ✅ **Multi-Depth Feature Aggregation 显著提升speculation准确性**  
   利用pipeline中各阶段的中间状态，使预测始终“扎根”于目标模型的真实上下文中。

3. ✅ **Zero-Bubble Design 实现完美延迟隐藏**  
   只要speculation module的延迟不超过单stage执行时间，即可完全并行，允许使用更深网络提升精度。

4. ✅ **SPD具有卓越的可扩展性**  
   - 增加pipeline stage数（n） → 理论speedup稳定上升
   - 增加speculation层数（Ls） → 准确率提升且无额外延迟

5. ✅ **在随机采样（T=1）下鲁棒性强于EAGLE-3**  
   表明SPD更好捕捉了teacher model的整体logit分布，而非仅top-k logits。

6. ✅ **任务相关性明显：低熵任务收益更大**  
   接受率排序：**HumanEval > GSM8K > MT-Bench**  
   → 表明SPD特别适合代码生成、结构化推理等确定性强的任务。

---

### 方法的局限性

1. ⚠️ **工程实现尚未优化**  
   当前基于PyTorch原生实现，缺乏：
   - 异步执行
   - 自定义CUDA kernel
   - 连续批处理（continuous batching）支持  
   → 导致无法测量真实端到端wall-clock加速。

2. ⚠️ **异构架构负载不平衡问题**  
   如Qwen3.5-4B中标准注意力与线性注意力交错，若划分成16 stage，会导致某些stage计算量更大，破坏同步性，可能产生latency bubble。

3. ⚠️ **训练效率妥协**  
   为匹配推理时的动态feature depth，采用了$(n+1)$倍扩展输入+非对称attention（仅g⁰作为query），虽节省内存，但可能轻微降低预测精度。

---

### 未来工作方向

1. 🔧 **集成至主流推理引擎**  
   将SPD嵌入 **SGLang** 或 **vLLM**，结合paged attention、fused kernel等技术，实现真实环境下的高速推理。

2. 🔄 **解决异构模型的load balancing问题**  
   设计智能layer分配策略，确保各pipeline stage计算均衡。

3. 🧠 **探索更高效的speculation module结构**  
   如稀疏化、量化、蒸馏等，进一步降低VRAM占用。

4. 🌐 **扩展至分布式多GPU部署场景**  
   研究跨设备通信开销下的最优pipeline调度策略。

---

> 💡 **总结一句话**：  
> **SPD通过将speculative decoding从“多token预测”转向“pipeline并行+单token推测”，从根本上解决了预测误差累积与延迟等待两大难题，是迈向高效、可扩展LLM推理的重要一步。**

</details>

---

### 3. [Mellum2 Technical Report](https://arxiv.org/abs/2605.31268)

**Authors**: Marko Kojic, Ivan Bondyrev, Aral de Moor, Joseph Shtok, Petr Borovlev, Kseniia Lysaniuk, Madeeswaran Kannan, Ivan Dolgov, Nikita Pavlichenko  
**Category**: cs.CL  
**Published**: 2026-06-01  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.31268v1  

#### Abstract
We present Mellum 2, an open-weight 12B-parameter Mixture-of-Experts (MoE) language model with 2.5B active parameters per token. Mellum 2 is a general-purpose language model specialized in software engineering, spanning code generation and editing, debugging, multi-step reasoning, tool use and funct...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MELLUM 2 Technical Report 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前开源大语言模型在**代码工程任务**中面临“质量-成本”权衡困境：
- **Dense 模型**（如 4B–7B）部署成本低，但在复杂编码、数学推理等任务上表现有限；
- **大规模 MoE 模型**虽具备前沿能力，但推理开销高，难以在消费级硬件上高效部署。

MELLUM 2 旨在构建一个**兼具高性能与高效率的通用代码助手模型**，特别适用于集成开发环境（IDE）中的实时辅助场景，实现高质量代码生成、调试、工具调用等功能，同时保持极低的每 token 推理成本。

---

### 提出的新方法与创新点

#### （1）效率优先的 MoE 架构设计
基于 Qwen3-MoE 配方，引入多项优化以提升推理效率：
- **Mixture-of-Experts (MoE)**：总参数 12B，每 token 激活仅 2.5B 参数（64 个专家中激活 8 个），显著降低计算量。
- **Grouped-Query Attention (GQA)**：仅保留 4 个 KV 头，在高并发场景下大幅提升吞吐量。
- **Sliding Window Attention (SWA)**：每 4 层中有 3 层采用 1,024 token 的滑动窗口，减少长序列注意力计算负担。
- **Multi-Token Prediction (MTP) Head**：作为辅助预训练目标，并可作为 speculative decoding 的 draft model，加速推理。

#### （2）三阶段渐进式预训练课程
遵循 “web early, curated late” 范式：
- **Phase 1: Foundation**（6.18T tokens）：以通用网页为主（70%），初步建立语言理解；
- **Phase 2: Quality Uplift**（2.79T tokens）：提升高质量代码与数学数据比例至 42%；
- **Phase 3: Capability Sharpening**（1.69T tokens）：代码占比达 59%，强化专业能力。

该策略使模型逐步从广泛知识过渡到深度技术能力。

#### （3）长期上下文扩展（128K）
通过 **layer-selective YaRN** 方法将原生 8K 上下文扩展至 **131,072 tokens**：
- 仅对使用全注意力的层进行频率重映射，避免扰动滑动窗口层；
- 在 RULER 基准上优于统一 RoPE 扩展方案。

#### （4）双路径后训练输出两种变体
从同一基础模型衍生两个版本：
- **Instruct Model**：直接输出答案，适合快速响应；
- **Thinking Model**：显式输出推理链（reasoning trace），再给出最终答案，增强可解释性。

两者均经过 **Reinforcement Learning with Verifiable Rewards (RLVR)** 优化，无需人工标注奖励。

#### （5）开放发布
所有 checkpoint（base/instruct/thinking）、训练配方和技术报告均以 **Apache 2.0 协议**公开，推动社区复现与迭代。

---

### 相比现有方法的优势

| 维度 | MELLUM 2 优势 |
|------|---------------|
| **推理效率** | 每 token 计算量相当于 2.5B dense 模型，远低于同级 7B–14B 模型 |
| **部署可行性** | 可运行于单张 H100 GPU，适合 IDE 实时服务 |
| **功能完整性** | 支持 agentic coding、tool use、function calling、multi-step reasoning |
| **上下文长度** | 支持 128K 上下文，优于多数同类模型 |
| **开放程度** | 完整开源模型权重与训练细节，透明度高 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### 预训练数据（总计约 10.6T tokens）
| 类别 | 内容 |
|------|------|
| **Source Code** | 公共仓库源码、Common Crawl 中提取的代码、合成代码数据集（含注释、测试生成、跨语言翻译等） |
| **Web & General Knowledge** | Common Crawl 衍生语料、教育网站、PDF 文档、多语言问答、维基百科改写、合成百科文章 |
| **Mathematical Content** | 数学教材、STEM 教学数据、数学指令微调数据 |

#### 后训练数据
- **SFT 数据**：涵盖通用对话、单步/多步编码、agent 编程轨迹、函数调用、安全响应等；
- **RL 数据**（RLVR）：来自公开 RLVR 发布 + 自建任务，覆盖六大领域：
  - Code（~22–28%）
  - Math（~23–28%）
  - Agentic Tool Use
  - Instruction Following
  - Reasoning
  - Knowledge

---

### 实验设置与评估指标

#### 模型架构参数
| 参数 | 值 |
|------|----|
| 总参数 | ~12B |
| 激活参数/Token | ~2.5B |
| 专家数 | 64（Top-8 激活） |
| 注意力头 | 32 Query Heads, 4 KV Heads (GQA) |
| 上下文长度 | 原生 8K → 扩展至 128K |
| Tokenizer | 98,304 词表大小 |
| Position Encoding | RoPE (θ=500,000) |
| MoE 路由 | Global Load Balancing Loss + Z-loss |

#### 训练基础设施
- **硬件**：32 节点 × 8 H200 GPUs
- **并行策略**：Expert Parallelism (8), Tensor/Pipeline Parallelism (1)
- **优化器**：Distributed Muon（Moonlight 配置）
- **精度**：BF16 + FP8 hybrid mixed precision
- **批处理**：Global batch size 最终为 4,096 sequences（~33.6M tokens/step）

#### 评估基准
| 类别 | 基准 |
|------|------|
| **Code Generation** | HumanEval, HumanEval+, MBPP, MBPP+, MultiPL-E, CRUXEval, LiveCodeBench v6 |
| **Math & Reasoning** | GSM8K, MATH, AIME, GPQA, MMLU, MMLU-Pro, BBH |
| **Tool Use** | BFCL v3/v4（支持 web search 和 memory） |
| **Knowledge** | MMLU-Redux, TruthfulQA |
| **Conversational** | IFEval, MixEval, BS-Bench |
| **Safety** | HarmBench（越低越好）, XSTest（越高越好） |
| **Long Context** | RULER @ 16K–128K |

#### 对比基线模型
- Qwen2.5-7B
- Qwen3.5-4B / Qwen3.5-9B
- OLMo-3-7B
- Ministral-3-14B
- Seed-Coder-8B

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Post-Training 结果）

#### ✅ Coding Performance（EvalPlus 平均值）
| Model | EvalPlus (↑) |
|-------|-------------|
| **MELLUM 2-RL (Instruct)** | **78.4%** |
| Qwen3.5-9B-Instruct | 71.8% |
| Seed-Coder-8B | 73.8% |
| OLMo-3-7B-Instruct | 67.3% |

> 📌 **MELLUM 2 在函数级代码生成上超越所有对比模型**

#### 🔁 Thinking Mode 下算法推理能力爆发
| Model | LiveCodeBench v6 (↑) |
|--------|---------------------|
| **MELLUM 2-SFT-Thinking** | **75.1%** |
| Qwen3.5-9B-Thinking | 68.3% |
| MELLUM 2-Instruct | 37.2% |

> 💡 显式思考模式极大释放模型潜力，尤其在需要多步推理的任务中

#### 🧮 数学能力（AIME）
| Model | AIME (↑) |
|--------|---------|
| **MELLUM 2-RL-Thinking** | **58.4%** |
| Qwen3.5-9B-Thinking | 73.4% |
| MELLUM 2-SFT-Thinking | 20.0% |

> ⚠️ SFT 阶段未充分学习数学推理，需 RL 阶段才能激活

#### 🛠️ 工具使用（BFCL v4）
| Model | BFCL v4 (↑) |
|--------|------------|
| **MELLUM 2-RL-Thinking** | **45.6%** |
| Qwen3.5-9B-Thinking | 42.7% |
| Qwen3.5-4B-Thinking | 42.9% |

> ✅ 在 agentic tool calling 上领先，表明其工具调用策略具有泛化能力

#### 🌐 世界知识（MMLU-Redux）
| Model | MMLU-Redux (↑) |
|--------|---------------|
| Qwen3.5-9B-Instruct | 91.1% |
| **MELLUM 2-RL-Instruct** | **78.1%** |

> ❗ 知识类任务是短板，因训练数据偏向代码与工程文档

---

### 与基线对比总结

| 维度 | MELLUM 2 表现 |
|------|----------------|
| **代码生成** | ✔️ 领先，尤其在 EvalPlus 上超越更大模型 |
| **数学推理** | ✔️ RL + Thinking 模式下接近 SOTA |
| **工具使用** | ✔️ 在 BFCL v4 上排名第一 |
| **上下文理解** | ✔️ 成功扩展至 128K，RULER 表现优异 |
| **通用知识** | ❌ 弱于 Qwen3.5 系列，属有意 trade-off |
| **安全性** | ⚠️ SFT 版本最安全（HarmBench 8.4%），但 RL 后上升至 23.1%（alignment tax） |

---

### 消融实验结果

#### （1）MTP Head 消融（14B MoE, 105B tokens）
| Benchmark | Baseline | +MTP | Δ |
|----------|----------|------|---|
| HumanEval | 20.73 | 31.10 | +10.37 |
| MMLU | 37.49 | 41.06 | +3.57 |
| GSM8K | 30.63 | 33.59 | +2.96 |
| BBH | 35.00 | 37.74 | +2.74 |

> ✅ MTP 显著提升各项任务表现，且不增加推理负担（训练后移除）

#### （2）KV Heads 数量影响
- **4 KV Heads**：最佳平衡点；少于 4 导致质量下降，多于 4 则吞吐下降明显。
- 在 throughput mode 下，KV cache 成为主要瓶颈。

#### （3）Sliding Window Size
- **Window Size 1,024** > 512，在质量与延迟之间取得更好平衡。
- 3:1 SWA 模式（3 层滑窗 + 1 层全注意）有效保留长程依赖。

#### （4）Optimizer 对比
| Optimizer | Dense 7B Loss | MoE 14B Loss |
|----------|----------------|--------------|
| AdamW | 1.40 | 1.30 |
| Muon (Megatron) | diverged | 1.27 |
| **Muon (Moonlight)** | **1.37** | **1.28** |

> ✅ Moonlight 配置更稳定，尤其在 dense 模型上避免发散

---

## 4. 关键结论和发现

### 主要发现

1. **MoE 架构可在极低激活参数下媲美甚至超越 7B–14B Dense 模型**
   - 尽管仅激活 2.5B 参数，MELLUM 2 在多个推理密集型任务（如 BBH、MMLU-Pro、GSM8K）上表现优于或匹敌更大模型。

2. **Thinking 模式是解锁复杂推理的关键**
   - 显式推理链能显著提升模型在算法编程、数学解题上的表现，说明“思考预算”对小规模模型至关重要。

3. **RLVR 是高效的后训练范式**
   - 使用程序化验证奖励（而非人类偏好），避免了 Reward Model 噪声问题，更适合代码与数学任务。

4. **layer-selective YaRN 是有效的长上下文扩展策略**
   - 仅修改全局注意力层即可成功扩展至 128K，优于统一调整所有层的方法。

5. **开放性与可复现性并重**
   - 提供完整训练日志、超参配置、数据流程，极大促进社区研究。

---

### 方法的局限性

| 局限 | 描述 |
|------|------|
| **世界知识不足** | 因训练数据聚焦代码与工程，MMLU/GPQA 等常识任务得分偏低 |
| **RL 导致安全行为退化** | RL 阶段 HarmBench 分数从 8.4% 升至 23.1%，存在 alignment tax |
| **过度合规倾向** | BS-Bench 显示模型倾向于接受错误前提完成任务，缺乏质疑精神 |
| **XSTest 过度拒绝** | 安全提示被过度拒绝，反映安全与可用性的权衡尚未最优 |
| **Thinking 模型难部署** | 若无生成上限控制，可能陷入无限推理循环（类似 Qwen3 系列） |

---

### 未来工作方向

1. **深化 SWE Agent 训练**
   - 直接在真实软件工程项目上进行 RL 训练，打造真正可用的小型 SWE Agent。

2. **扩大 RL 基础设施与环境覆盖**
   - 构建更多可验证的交互式任务环境，提升 agent 泛化能力。

3. **优化长上下文中期训练数据混合**
   - 当前 Long-context mix 未能复现 OLMo 3 报道的增益，需进一步探索。

4. **改进 alignment 机制**
   - 减少 RL 对安全性的负面影响，联合优化 HarmBench 与 XSTest。

5. **探索 Hybrid 架构（如 Mamba/MoE）**
   - 当前因短上下文推理性能不佳暂未采用，未来随框架优化有望突破。

6. **推出更大规模的 inference-aware Mellum**
   - 延续“固定推理预算下最大化能力”的设计理念，拓展至更高容量模型。

---

> ✅ **总体评价**：  
> MELLUM 2 是一款极具工程导向的开源 MoE 模型，精准定位“低成本 + 高效能 + 强代码能力”的开发者助手角色。它不仅在性能上达到先进水平，更重要的是提供了完整的训练蓝图，为后续小型 MoE 模型的发展树立了标杆。

</details>

---

### 4. [CoMem: Context Management with A Decoupled Long-Context Model](https://arxiv.org/abs/2605.30842)

**Authors**: Yuwei Zhang, Chengyu Dong, Shuowei Jin, Changlong Yu, Hejie Cui, Hongye Jin, Xinyang Zhang, Hamed Bonab, Colin Lockard, Jianshu Chen, Zhenyu Shi, Jingbo Shang, Xian Li, Bing Yin  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.30842v1  

#### Abstract
Context management enables agentic models to solve long-horizon tasks through iterative summarization of previous interaction histories. However, this process typically incurs substantial decoding overhead for the extra summarization tokens, which significantly affect the end-to-end response latency...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# CoMEM: Context Management with A Decoupled Long-Context Model 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **agentic LLM**（代理型大语言模型）执行长周期任务时，需要维护完整的交互历史以保证推理连贯性。然而，随着上下文长度增长，**KV Cache** 的内存占用和注意力计算开销呈线性甚至超线性增长，导致 **decoding 阶段成为内存瓶颈（memory-bound）**，显著增加端到端响应延迟，影响系统吞吐和用户体验。

现有方法如 **Sliding Window** 或 **RAG** 虽能减少上下文长度，但会丢失远距离依赖；而系统级优化（如 PagedAttention、KV Cache Quantization）虽提升效率，但仍需对完整历史进行重复处理，无法根本解决冗余计算问题。

---

### 提出了什么新方法或新思路
本文提出 **CoMEM**，一种将 **memory management** 与 **agentic reasoning** 解耦的新型框架，核心思想如下：

- **Decoupled Architecture**：引入一个轻量级的专用 **memory model**，负责异步压缩长程历史为紧凑的“摘要状态” $ s_t $，而主 **agent model** 仅基于该摘要和最近几轮原始交互进行决策。
  
- **k-step-off Asynchronous Pipeline**：设计了一种新颖的异步流水线机制，允许 memory model 滞后 $ k $ 步更新摘要。在此期间，agent 利用旧摘要继续生成动作，实现 **context summarization 与 agent inference 的并行化**，有效掩盖摘要延迟。

- **Reward-Driven Alignment Training**：采用基于 **functional equivalence** 的强化学习目标（使用 GRPO），训练 memory model 生成能够引导 agent 做出与全上下文一致决策的“充分统计量”（sufficient statistics），而非追求表面文本质量。

---

### 相比现有方法的优势
| 维度 | CoMEM | 传统方法 |
|------|-------|--------|
| **延迟控制** | 将 context 处理移出关键路径，并行执行，大幅降低 decoding 延迟 | 串行处理，延迟随上下文线性增长 |
| **性能保持** | 通过功能一致性训练保留关键决策信息，性能接近 full-context baseline | 易丢失重要远期依赖，性能下降明显 |
| **可扩展性** | 支持独立优化 memory 和 agent 模块，适合高并发部署 | 架构耦合，难以模块化升级 |
| **资源效率** | 使用小模型处理压缩任务，节省 GPU HBM 占用 | 大模型全程参与，资源消耗高 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 主要评测任务：**SWE-Bench-Verified** —— 一个人工验证过的 GitHub 代码修复基准，用于评估自主软件工程 agent 的能力。
- 泛化性测试：**BrowseComp-EN** —— 一个多跳网页搜索与信息抽取的复杂推理任务，验证方法在非编码场景下的适用性。

---

### 实验设置和评估指标

#### 模型配置
- **Agent Models**：
  - DeepSWE (32B Dense)
  - Qwen3-Coder-Max (480B A35B)
  - GLM-4.7 (355B 32B)
- **Memory Model**：统一使用 **Qwen3-4B**，具备长上下文能力且参数量小。

#### 推理设置
- 使用 **vLLM 0.14.0** 作为推理引擎，启用 prefix caching 和 chunked prefilling。
- 异步 pipeline 中，$ k=2 $（DeepSWE / Qwen3-Coder-Max）或 $ k=4 $（GLM-4.7）。
- 最大 summary 长度固定为 **2048 tokens**。
- 测试硬件：A100 (80GB) 或 H200。

#### 评估指标
| 指标 | 含义 |
|------|------|
| `%Resolved` | 成功解决的 issue 比例（主要效果指标） |
| `#Tool Calls` | 平均调用工具次数（反映决策效率） |
| `Latency (×128/s)` | 处理 128 个 issue 的总推理时间（效率指标） |
| `Speedup` | 相对于 full-context baseline 的加速比 |

---

### 基线方法对比
| 基线名称 | 描述 |
|---------|------|
| **Full-Context** | 标准全流程上下文输入，无压缩 |
| **No Summary** | 移除摘要，仅保留近期交互 |
| **Qwen3-4B (base)** | 使用原始预训练模型做摘要，无微调 |
| **Qwen3-4B (SFT)** | 经过监督微调的摘要模型 |
| **GRPOAC** | 本文提出的奖励驱动训练版本（即 CoMEM 核心） |
| **MemAgent (7B)** | 外部通用记忆模型（Yu et al., 2025），用于横向比较 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 5）

| Agent Backbone | Memory Method | %Resolved | Latency (s) | Speedup |
|----------------|---------------|-----------|-------------|---------|
| GLM-4.7 | Full-Context | 69.0 | 5869.42 | 1.00× |
| GLM-4.7 | CoMEM (GRPOAC) | **62.7** | **2821.38** | **2.08×** |
| Qwen3-Coder-Max | Full-Context | 57.2 | 5129.90 | 1.00× |
| Qwen3-Coder-Max | CoMEM (GRPOAC) | **51.0** | **3594.51** | **1.43×** |
| DeepSWE | Full-Context | 40.4 | 6390.62 | 1.00× |
| DeepSWE | CoMEM (GRPOAC) | **41.0** | **4400.89** | **1.45×** |

> ✅ **关键发现**：CoMEM 在所有 backbone 上实现了 **1.43× ~ 2.08× 的端到端加速**，同时保持了接近甚至略优于 full-context 的性能（如 DeepSWE 上达到 41.0% > 40.4%）。

---

### 与基线方法的对比结果
- **vs. Off-the-shelf/SFT 摘要模型**：未经任务对齐训练的 base/SFT 版本性能显著下降（如 GLM-4.7 上从 69.0% → 58.3%/61.3%），说明通用摘要不足以支撑 agent 决策。
- **vs. MemAgent (7B)**（见 Table 5）：
  - 在 GLM-4.7 上，CoMEM 分辨率高出 **7.9%**（62.7% vs. 54.8%），速度更快（2.08× vs. 1.80×）。
  - 表明 **针对下游 agent 进行 action-consistency 对齐训练** 是关键优势。

---

### 消融实验结果（Ablation Studies）

#### （1）不同 $ k $ 值的影响（Table 4）
| $ k $ | %Resolved | Latency (s) | Speedup |
|-------|-----------|------------|---------|
| 1 | 57.2 | 3843.5 | 1.53× |
| 2 | **64.2** | 2841.4 | 2.07× |
| 4 (default) | 62.7 | **2821.4** | **2.08×** |
| 8 | 62.4 | 2836.0 | 2.07× |
| 16 | 60.2 | 3576.8 | 1.64× |

> 🔍 发现：$ k \in \{2,4,8\} $ 性能稳定，$ k=1 $ 因频繁 uncached prefill 导致开销过大，$ k=16 $ 则因摘要过时而性能下降。

#### （2）泛化性测试：BrowseComp-EN（Table 6）
| Method | Accuracy (%) |
|--------|--------------|
| Full-Context | 28.1 |
| CoMEM ($k=1$) | **32.0** |

> 🚀 **惊人发现**：CoMEM 不仅没有损失性能，反而提升了 **3.9% 绝对准确率**，表明结构化摘要有助于过滤噪声、聚焦关键信息，在某些任务中甚至优于原始长上下文。

#### （3）跨 scaffold 和 backbone 的迁移能力
- **跨 scaffold 迁移**（R2E-Gym → OpenHands）：性能仅下降 **0.3%**，显示对 prompt 模板和工具接口变化具有鲁棒性。
- **跨 backbone 迁移**（Qwen3-Coder-Max → GLM-4.7）：性能差距仅 **0.8%**，说明 memory model 学习的是通用压缩策略。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **解耦架构可行且高效**：将 memory 管理从 agent 推理中分离是可行的，且可通过异步 pipeline 实现显著延迟隐藏。
2. **功能一致性训练至关重要**：直接优化“是否能复现原 policy”的 reward，比文本相似度更能引导 memory model 抽取决策相关的关键信息。
3. **延迟收益随吞吐量放大**：在高并发场景下（batch=256），CoMEM 可达 **2.52× 加速**，峰值单步提速达 **4.95×**（见 Figure 5 和 Table 2）。
4. **摘要不仅是压缩，更是增强**：在 BrowseComp 等信息密集任务中，摘要可主动过滤噪声，提升 agent 表现。

---

### 方法的局限性
- **依赖高质量 reference trajectory**：训练 memory model 需要 full-context agent 产生的黄金轨迹，增加了前期成本。
- **存在摘要滞后风险**：当 $ k $ 过大时，摘要可能无法及时反映最新状态，影响短期决策。
- **需额外部署 memory server**：虽然成本低，但仍引入了系统复杂性和运维负担。

---

### 未来工作方向
- **动态调整 $ k $**：根据任务复杂度或上下文变化频率自适应调节更新频率。
- **多粒度摘要分层**：构建 hierarchical memory，支持不同时间尺度的信息检索。
- **联合训练 memory 与 agent**：探索 joint RL 框架，在线协同优化两者策略。
- **应用于更广泛领域**：如长期对话系统、机器人控制、科学发现等需要持久记忆的任务。

---

> 💡 **总体评价**：CoMEM 是一项兼具理论深度与工程实用性的创新工作，它不仅解决了 long-context agent 的核心性能瓶颈，还为构建模块化、可扩展的智能体系统提供了新的架构范式。其“绿色 AI”潜力（降低能耗）也值得重视。

</details>

---

### 5. [UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-Time Scaling](https://arxiv.org/abs/2605.30898)

**Authors**: Kaiyu Huang, Xingyu Wang, Mingze Kong, Zhubo Shi, Yuqian Hou, Hong Xu, Zhongxiang Dai, Minchen Yu, Qingjiang Shi  
**Category**: cs.AI  
**Published**: 2026-06-01  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.30898v1  

#### Abstract
In real-world deployments of large language models (LLMs), balancing inference quality and computational cost has become a central challenge. Existing approaches tackle this trade-off along two largely independent dimensions: model routing, which switches among models of different scales to match re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**UNISCALE: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-Time Scaling**

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

在大型语言模型（LLM）的实际部署中，如何在推理质量（如准确性）与计算成本之间取得平衡是一个核心挑战。现有方法通常将以下两个维度独立处理：

- **Model Routing**：在不同规模的模型间切换，以匹配请求复杂度。
- **Test-Time Scaling (TTS)**：在固定模型内调整推理时的计算量（如采样次数、搜索深度等），实现细粒度控制。

然而，这种**解耦设计存在固有局限**：
- Model Routing 只能提供**粗粒度、离散的质量-成本变化**（因模型池有限）。
- 单一模型的 TTS 存在**容量上限**，增加计算常出现收益递减。
- 二者分离导致系统在动态环境中适应能力差。

---

### ✅ **提出了什么新方法或新思路**

本文提出 **Unified Inference Scaling (UIS)** 范式，并构建了在线框架 **UNISCALE** 来实现该范式的自适应优化。

#### 🔹 **Unified Inference Scaling (UIS)**
- 将 Model Routing 和 TTS 统一为一个联合决策空间。
- 每次推理由配置 `(M, QP, CP, BS)` 参数化：
  - `M`：基础模型
  - `QP`（Question Parallelism）：并行探索的子树数量
  - `CP`（Candidate Parallelism）：每步生成的候选状态数
  - `BS`（Beam Size）：保留用于扩展的路径数
- 构建了一个**连续且表达力强的质量-成本前沿**（quality-cost frontier），通过 TTS 弥合模型间的性能鸿沟，同时通过路由突破小模型的 TTS 上限。

#### 🔹 **UNISCALE 框架**
- 将 UIS 配置选择建模为 **contextual multi-armed bandit** 问题，使用 **LinUCB** 算法进行在线学习。
- 包含三大核心机制确保高效稳定优化：
  1. **Path-Aware Early Exiting**：基于验证器分数实时判断低潜力路径并提前终止，显著降低冗余计算。
  2. **Dense Verification Feedback**：利用 PRM 输出的连续分值替代稀疏的二元正确性信号，提供更密集的奖励反馈。
  3. **UIS Cost Model**：采用 **equivalent FLOPs (eFLOPs)** 统一度量计算与内存开销，实现跨硬件的成本一致性建模。

---

### ✅ **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **优化粒度** | 实现**细粒度 + 广覆盖**的联合控制，超越单一维度方法的极限 |
| **在线适应性** | 支持环境漂移（如查询分布变化、模型增删、目标切换）下的持续策略更新 |
| **效率与稳定性** | 语义动作表示促进跨配置知识迁移；轻量级线性估计器保证低延迟更新 |
| **通用性** | 框架正交于具体 TTS 技术（如 BoN、Beam Search），可组合使用 |

---

## 2. 核心实验方法和设置

### 📚 **使用的数据集**

- **AIME'24**, **AIME'25**, **MATH-500** 中共抽取 **210 个数学推理实例**。
  - MATH-500 每难度等级随机采样 30 例（Level 1–5）。
- 编码任务额外测试使用 **LiveCodeBench** 数据集（210 个编程题）。

---

### ⚙️ **实验设置**

- **候选模型**：Qwen3 系列（0.6B, 1.7B, 4B, 8B, 14B, 32B）
- **TTS 配置范围**：
  - QP, CP ∈ {1,…,8}, QP×CP ≤ 64
  - BS ∈ {1,2,4}
- **验证器（PRM）**：Skywork-o1-Open-PRM-Qwen-2.5-1.5B / -7B
- **执行平台**：NVIDIA A800 80GB GPU 集群
- **推理引擎**：vLLM（支持 prefix caching/sharing, dynamic batching）

---

### 🎯 **评估指标**

| 类型 | 指标 |
|------|------|
| **静态性能** |  
| - Reward | 复合奖励：$ r_t = w_1 \cdot \text{Correct}(a) + w_2 \cdot \text{Score}(a) + w_3 \cdot (1 - C_{\text{uis}}(a)) $ |
| - Accuracy (%) | 最终答案准确率（百分点） |
| - Cost (Tera-eFLOPs) | 推理总成本（基于 eFLOPs） |
| **动态学习效率** |  
| - Cumulative Regret | 衡量策略收敛速度与最优性的差距 |
| - Reg.@130 / Reg.@210 | 第130/210步时的累积遗憾 |
| - 正确计数 vs. 成本曲线 | 展示性价比演化过程 |

---

### 🔁 **基线方法对比**

| 类别 | 基线方法 |
|------|----------|
| **Multi-armed Bandit** |  
| - Random | 随机选择 |
| - Greedy | 基于当前估计选择最高预期奖励 |
| - Thompson Sampling (TS) | 贝叶斯后验采样 |
| - NeuralUCB | 基于神经网络的非线性上界探索 |
| **Predictive Routing** |  
| - MLP | 在线训练的多层感知机预测器 |
| - k-NN | 基于历史相似上下文的最近邻投票 |
| **Oracle** | 理论最优（已知所有配置真实表现） |
| **对比变体** |  
| - TTS-only / Routing-only | 分别仅优化 TTS 或仅路由 |
| - BEST-Route* | 限制为 Best-of-N 的受限 UIS 空间 |

---

## 3. 主要实验结果和性能指标

### 📊 **关键性能数据（来自 Table 2）**

| 方法 | 场景 | Reward | Accuracy (%) | Cost (Tera-eFLOPs) |
|------|------|--------|---------------|---------------------|
| **UNISCALE (ours)** | **UIS (Cost-Sensitive)** | **0.7079** | **46.88** | **49.4** |
| k-NN | UIS (Cost-Sensitive) | 0.6590 | 41.38 | 326.0 |
| Oracle | UIS (Cost-Sensitive) | 0.8337 | 57.38 | 10.6 |
| **UNISCALE (ours)** | **UIS (Quality-Priority)** | **0.6306** | **57.37** | **1374.7** |
| k-NN | UIS (Quality-Priority) | 0.5807 | 46.75 | 1113.4 |
| Oracle | UIS (Quality-Priority) | 0.7924 | 68.12 | 115.7 |

> ✅ **UNISCALE 在两种模式下均取得最佳综合性能**，尤其在 Cost-Sensitive 模式下实现**极低成本（<50 Tera-eFLOPs）与高准确率（>46%）的结合**。

---

### 🔍 **与基线方法的对比结果**

- **全面优于所有基线**：
  - 在 **UIS 场景**中，UNISCALE 的 Reward 显著高于其他方法（如比 k-NN 提升约 +7.4%）。
  - 在 **Cost-Sensitive 模式**下，其成本仅为 k-NN 的 **15% 左右**，而准确率更高。
  - 在 **Quality-Priority 模式**下，达到 **57.37% 准确率**，领先第二名近 3.4pp。

- **优于 BEST-Route***（见 Table 7）：
  - 在 Cost-Sensitive 模式下，UNISCALE 实现 **+11.25pp 更高准确率**，同时**减少超过 95% 的推理开销**。
  - 证明了**多维 TTS + 模型路由联合空间的有效性远超单维 Best-of-N**。

- **编码任务泛化性验证**（Table 8）：
  - 在 LiveCodeBench 上，UNISCALE 相比 k-NN：
    - 成本下降 **59.97%**
    - Reward 提升 **+0.0182**
    - 准确率仅微降 0.63pp → 显示出强大的**任务无关（task-agnostic）调度能力**

---

### 🔧 **消融实验结果**

#### （1）**Action Semantic Representations**（表3 & 图9）
- 移除语义表示（w/o Sem.）后：
  - 累积遗憾上升（Reg.@210 从 34.43 → 41.56）
  - 成本敏感模式下成本反而更高（629.7 vs 48.2）
- 结论：**语义嵌入极大提升跨配置的知识迁移能力，加速收敛**

#### （2）**Path-Aware Early Exiting**（表4）
- 引入早退机制后：
  - 计算负载 ↓87.26%
  - 内存访问 ↓78.06%
  - 总成本 ↓79.33%
  - 准确率仅下降 0.91pp
- 结论：**有效识别逻辑收敛点，消除“拖尾效应”**

#### （3）**Dense Verification Feedback**（表5）
- 使用稀疏二元反馈（w/o Dense Feedback）：
  - Quality-Priority 模式下准确率 ↓5.62pp，成本 ↑104.7%
  - Cost-Sensitive 模式下准确率 ↓2.5pp
- 结论：**连续验证分数是高质量策略学习的关键监督信号**

#### （4）**Exploration Factor α 敏感性分析**（图11）
- α=1（默认）时性能最优
- α=0（纯贪婪）易陷入局部最优
- α=10（过度探索）导致效率低下
- 结论：**LinUCB 的置信上界机制实现了 exploitation 与 exploration 的良好平衡**

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **Model Routing 与 TTS 应被统一建模**：二者具有互补性——TTS 可平滑模型跳跃，模型路由可突破 TTS 容量瓶颈。
2. **UIS 空间能构建更优的质量-成本前沿**：联合优化显著优于任一单独维度的优化。
3. **UNISCALE 实现高效在线自适应**：
   - 利用 contextual bandit 实现低延迟决策
   - 语义表示 + LinUCB 实现快速跨配置泛化
   - 早退机制与稠密反馈大幅提升实际效率
4. **框架具备强鲁棒性与泛化性**：
   - 对环境漂移（模型增删、目标切换）响应迅速
   - 在数学与编程任务上均表现出色
   - 对 PRM 质量有一定容忍度（见 Table 9）

---

### ⚠️ **局限性**

1. **依赖高质量 PRM**：虽然对弱 PRM 仍有效，但更强的验证器能进一步提升上限。
2. **动作空间需预定义**：目前假设候选模型和 TTS 配置集合固定，未考虑完全开放的动作生成。
3. **离散化带来的搜索限制**：尽管空间丰富，但仍为离散集合，无法实现完全连续控制。
4. **冷启动阶段依赖随机探索**：前 50 步 warm-up 影响初期性能。

---

### 🔮 **未来工作方向**

1. **引入更强的通用验证机制**：探索不依赖特定 PRM 的自洽性评估方法。
2. **扩展至连续动作空间**：结合强化学习或梯度策略实现更精细的参数调节。
3. **支持动态模型池管理**：实现模型即服务（MaaS）场景下的自动注册与淘汰。
4. **联邦式协同推理调度**：结合 federated orchestration 实现隐私保护下的跨组织策略共享。
5. **集成 speculative decoding 等加速技术**：进一步压缩端到端延迟。

---

## 💡 总结

> **UNISCALE 是首个将 Model Routing 与 Test-Time Scaling 统一建模并实现在线联合优化的框架**。它打破了传统方法的技术孤岛，通过构建 **Unified Inference Scaling (UIS)** 决策空间，在保持低延迟的同时实现了**宽范围、细粒度、自适应的质量-成本权衡**。实验证明其在多种任务和环境下均显著优于现有方法，为未来智能、绿色、普惠的大模型推理基础设施提供了重要范式。

</details>

---

### 6. [An Efficient and Scalable Graph Condensation with Structure-Preserving](https://arxiv.org/abs/2605.31016)

**Authors**: Yulin Hu, Fuyan Ou, Ye Yuan  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.31016v1  

#### Abstract
Graph condensation (GC) is pivotal for enabling Graph Neural Networks (GNNs) deployment in resource-constrained scenarios by compressing large-scale graphs into compact synthetic counterparts. Existing GC methods commonly suffer from computational inefficiency due to coupled optimization as well as ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An Efficient and Scalable Graph Condensation with Structure-Preserving (SP-ESGC)

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
图神经网络（GNN）在大规模图上训练成本高昂，限制了其在资源受限场景（如神经架构搜索、持续学习）中的应用。**Graph Condensation (GC)** 被提出用于将大图压缩为小型合成图以降低计算开销，但现有方法存在以下问题：
- **计算效率低**：多数方法采用耦合优化（如双层/三层优化），导致训练缓慢且难以扩展到大规模图。
- **泛化能力差**：依赖特定 GNN 架构作为“中继模型”，导致在不同 GNN 上表现不稳定。
- **结构信息保留不足**：难以有效保持原始图的拓扑结构特征。

---

### 🚀 提出的新方法：SP-ESGC
本文提出 **SP-ESGC (Structure-Preserving Efficient and Scalable Graph Condensation)**，其核心是**解耦设计**（decoupled design），将节点压缩与图结构生成分离：

#### 主要组件：
1. **Heat Kernel Feature Propagation (HKP)**  
   - 基于谱图理论进行特征扩散，通过热核 $ e^{-tL} $ 平滑节点表示。
   - 使用截断泰勒级数近似计算，提升可扩展性。
   - 引入全局结构信息，增强表示稳定性。

2. **Hybrid Clustering Strategy for Node Condensation**
   - 对每个类别内的节点表示执行混合聚类：
     - 先进行 **SVD 降维** 获取低秩子空间；
     - 再使用 **Random Fourier Features (RFF)** 近似 RBF 核映射，增强非线性判别能力；
     - 在谱嵌入空间中聚类，并回投影得到原始空间中的代表性类中心（centroids）。

3. **Pre-trained Edge Predictor for Graph Generation**
   - 预训练一个参数化的边预测器（edge predictor）来学习原始图中节点对之间的连接模式。
   - 在合成节点特征上进行全对推理（all-pairs inference），生成概率邻接矩阵。
   - 通过高分位阈值稀疏化，构建最终的稀疏图结构。

---

### 🔍 相比现有方法的优势
| 维度 | SP-ESGC 的优势 |
|------|----------------|
| **效率** | 避免双层优化和重复 GNN 训练，显著减少时间开销（例如在 Reddit 上比第二快的方法快约 16 倍）。 |
| **可扩展性** | 支持大规模图（如 Ogbn-arxiv 和 Reddit），无 OOM 问题。 |
| **泛化性** | 不依赖具体 GNN 架构，在多种 GNN（GCN, GAT, APPNP 等）上均表现稳定。 |
| **结构保持** | 利用预训练 edge predictor 学习可迁移的拓扑规则，实现更准确的结构重建。 |

---

## 2. 核心实验方法和设置

### 📊 数据集
实验在五个真实世界图数据集上进行，涵盖**归纳式**（inductive）与**直推式**（transductive）任务：

| 数据集 | 类型 | #节点 | #边 | #类别 | 特征维度 |
|--------|------|-------|-----|--------|----------|
| Cora | Transductive | 2,708 | 5,429 | 7 | 1,433 |
| Citeseer | Transductive | 3,327 | 4,732 | 6 | 3,703 |
| Ogbn-arxiv | Transductive | 169,343 | 1,166,243 | 40 | 128 |
| Flickr | Inductive | 89,250 | 899,756 | 7 | 500 |
| Reddit | Inductive | 232,965 | 57,307,946 | 210 | 602 |

> 注：对于归纳式数据集，$N$ 指训练子图节点数；condensation ratio $r = n / N$。

---

### 🧪 实验设置与评估指标

#### 评估任务
- **Node Classification Accuracy**：在合成图上训练 GNN 模型，在原始测试集上评估性能。

#### Condensation Ratios
- 设置多个压缩比例（如 0.05% ~ 5.20%），验证小样本下的有效性。

#### Baseline 方法对比
分为两类：
1. **Core-set 方法**：
   - Random
   - Herding
   - K-Center
2. **Graph Condensation 方法**：
   - GCond (gradient matching)
   - SGDD (structural gradient distillation)
   - SimGC (simple GC)
   - GC-SNTK (kernel-based)
   - SFGC (trajectory matching)

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Test Accuracy %）

见 **Table II** 结果摘要（部分关键项）：

| Dataset | Ratio | Best Method | Accuracy (%) | Whole Graph Performance |
|--------|-------|-------------|--------------|--------------------------|
| Cora | 1.30% | **SP-ESGC** | **82.6±0.6** | 81.2±0.2 |
| Reddit | 0.05% | **SP-ESGC** | **90.7±0.0** | 93.9±0.0 |
| Ogbn-arxiv | 0.50% | **SP-ESGC** | **66.8±0.1** | 71.4±0.1 |
| Flickr | 0.50% | **SP-ESGC** | **47.2±0.3** | 47.2±0.1 |

> ✅ SP-ESGC 在大多数情况下达到 **最优或次优性能**，尤其在极低压缩比下仍能保持高精度。

---

### ⏱️ Condensation Time 对比（秒）

见 **Table III**，在统一压缩率下比较总耗时：

| Method       | Cora | Citeseer | Ogbn-arxiv | Flickr | Reddit     |
|--------------|------|----------|------------|--------|------------|
| GCond        | 653.9 | 940.3    | 13,521.7   | 1,455.6| 20,528.8   |
| SGDD         | 4,848.6 | 3,091.2 | 35,179.7   | 26,767.4| 378,220.9  |
| SimGC        | 289.3 | 495.7    | 476.6      | 744.5  | 2,655.9    |
| GC-SNTK      | 92.1  | 69.7     | 28,897.8   | 889.8  | **OOM**    |
| SFGC         | 5,895.8 | 3,807.2 | 156,975.2  | 46,706.7| 370,089.0  |
| **SP-ESGC**  | **12.4** | **26.4** | **143.4**  | **22.6** | **162.5**  |

> 🔥 **SP-ESGC 是最高效的方法**，在 Reddit 上仅需 **162.5 秒**，约为第二快方法（SimGC）的 **1/16**。

---

### 🔬 消融实验结果（Ablation Study）

见 **Table IV**，验证各模块贡献：

| 方法变体 | Cora | Citeseer | Ogbn-arxiv | Flickr | Reddit |
|--------|------|----------|------------|--------|--------|
| SP-ESGC (full) | **82.7±0.1** | **72.8±0.2** | **66.7±0.1** | **47.2±0.3** | **91.6±0.0** |
| w/o HKP | 78.6±0.3 | 70.4±0.2 | 65.4±0.3 | 46.7±0.4 | 90.5±0.1 |
| w/o EP | 81.9±0.2 | 72.5±0.2 | 66.0±0.2 | 45.8±0.9 | 91.1±0.1 |
| w/ K-means | 81.4±0.2 | 72.0±0.2 | 66.6±0.0 | 46.7±0.2 | 91.4±0.0 |

#### 发现：
1. **移除 HKP 导致最大性能下降** → 表明全局平滑传播对表示质量至关重要。
2. **移除 Edge Predictor (EP)** → 结构建模能力减弱，说明基于特征相似性的简单连接不可靠。
3. **替换为 K-means** → 性能略降，表明提出的谱空间聚类更能捕捉非线性分布。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **解耦设计显著提升效率与稳定性**：SP-ESGC 成功避免复杂的 bi-level optimization，实现快速、可扩展的图压缩。
2. **Heat Kernel Propagation 提供高质量初始表示**：融合局部与全局信息，有助于后续聚类与结构恢复。
3. **Edge Predictor 实现结构可迁移性**：无需显式建模图结构，即可从原始图中学得通用连接规律并应用于合成节点。
4. **强泛化能力**：在多种 GNN 架构（GCN, GAT, APPNP 等）上均表现稳健，优于依赖特定 relay model 的方法（如 GCond 在 GAT 上性能下降明显）。

---

### ⚠️ 局限性
- 当前 edge predictor 假设连接模式可通过节点特征组合预测，可能无法完全捕获复杂依赖（如高阶路径、社区结构）。
- 聚类策略虽有效，但在类别极度不平衡或特征重叠严重时可能失效。
- 所有操作基于静态图，未考虑动态图扩展。

---

### 🔮 未来工作方向
1. 将 SP-ESGC 扩展至 **动态图 condensation** 场景。
2. 探索更强大的 **graph-to-edge mapping functions**，如引入 GNN-based predictor 或注意力机制。
3. 结合 **主动学习或强化学习** 策略优化 condensation 过程。
4. 应用于更大规模工业图（如社交网络、推荐系统）的实际部署验证。

---

## 总结
> **SP-ESGC 是一种高效、可扩展且结构感知的图压缩框架**。它通过**解耦节点压缩与结构生成**，结合 **heat kernel 传播 + 混合聚类 + 可迁移 edge predictor**，实现了在极高压缩比下仍保持高性能的图 condensation。实验证明其在准确性、效率和泛化性方面全面超越现有方法，特别适合大规模图和资源受限应用场景。

</details>

---

### 7. [Eigenvectors of Experts are Training-free Non-collapsing Routers](https://arxiv.org/abs/2605.30992)

**Authors**: Giang Do, Hung Le, Truyen Tran  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.30992v1  

#### Abstract
Sparse Mixture of Experts (SMoE) architectures improve the training efficiency of Large Language Models (LLMs) by routing input tokens to a selected subset of specialized experts. Despite their remarkable success, both training and inference in SMoE models suffer from the expert collapse issue (Chi ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Eigenvectors of Experts are Training-free Non-collapsing Routers*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **Sparse Mixture of Experts (SMoE)** 架构中的**专家坍缩（expert collapse）**问题展开研究。在SMoE模型中，路由机制（router）负责将输入token分配给最相关的专家子网络。然而，实践中多个专家常被分配相似或冗余的表示，导致“表示坍缩”，降低了模型的有效容量和泛化能力。

尽管已有工作尝试通过改进router设计来缓解此问题，但这些方法通常需要从头训练或微调，计算成本高昂，且在先进预训练模型上仍难以彻底解决坍缩现象。

### 提出了什么新方法或新思路
作者提出了一种全新的、无需训练的路由框架——**Singular Value Decomposition SMoE (SSMoE)**，其核心思想是：

> **利用专家权重矩阵的特征向量（eigenvectors）作为语义丰富的路由信号，替代或增强传统可学习的router。**

具体而言：
- 对每个专家的FFN层权重进行SVD分解，提取其Gram矩阵的主特征向量。
- 这些特征向量天然具有良好的正交性，并编码了专家的专业化语义信息。
- 构建基于特征向量的“EV Router”，并与原始router结合形成混合路由策略。

### 相比现有方法的优势
| 维度 | SSMoE优势 |
|------|----------|
| **是否需要训练** | ✅ 完全**training-free**，无需任何参数更新或微调 |
| **计算开销** | ✅ 路由过程轻量，仅涉及矩阵乘法与相似度计算 |
| **抗坍缩能力** | ✅ 特征向量天然近似正交，显著降低专家间相关性 |
| **通用性** | ✅ 可插拔应用于已有的SMoE模型（如GPT-OSS、OLMoE等） |
| **资源效率** | ✅ 支持专家剪枝（expert dropping），减少约23%内存占用 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖三大类任务，验证方法的广泛适用性：

#### （1）大型推理模型（Large Reasoning Models）
- **GPT-OSS-20B / GPT-OSS-120B**：开源MoE语言模型
- **评估基准**：
  - `ARC-C`, `ARC-E`（科学问答）
  - `BoolQ`（二元阅读理解）
  - `GSM8K`（数学应用题）
  - `HellaSwag`, `OBQA`, `PIQA`, `WinoGrande`（常识推理）

#### （2）大型语言模型（LLMs）
- 模型：`OLMoE-7B`, `Qwen-MoE-7B`, `DeepSeekMoE-16B`
- **评估基准**：**Massive Text Embedding Benchmark (MTEB)**
  - 包括分类（Classification）、聚类（Clustering）、句子对分类、重排序、检索、语义文本相似度（STS）、摘要等任务

#### （3）视觉-语言模型（Vision-Language Models）
- 模型：`CLIP-MoE`
- **任务**：
  - 零样本图像-文本检索（Zero-shot Image-Text Retrieval）：`COCO`, `Flickr30k`
  - 零样本图像分类：`CIFAR-10/100`, `STL-10`, `Caltech101`, `ImageNet-1K/O`

此外还测试了**对抗噪声下的鲁棒性**（corrupt setting），例如注入随机token或添加图像噪声。

### 实验设置和评估指标
| 设置 | 描述 |
|------|------|
| **评估模式** | 主要采用 **5-shot 和 10-shot in-context learning**，不进行fine-tuning |
| **内存优化** | 在部分实验中采用 **expert dropping（保留75%专家）** 以节省显存 |
| **关键指标** | Accuracy, Recall@K, Spearman相关系数, V-Measure, Silhouette Score, Perplexity, Answer Flip Rate |

### 基线方法对比
- **Original SMoE**：原始模型，使用标准learned router
- **RandomDrop**：随机移除25%专家
- **Router-only**：仅使用原始router输出
- **MoEE (Li & Zhou, 2025)**：结合router与隐藏状态的嵌入方法
- **PromptEOL**：基于prompt的上下文学习方法
- **Hash-based / Random routing**：非语义驱动的控制基线

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 大型推理任务（GPT-OSS系列）
| 模型 | 方法 | 平均得分 ↑ | 内存 ↓ |
|------|------|-----------|--------|
| GPT-OSS-20B | Original | 48.8 | 38.96 GB |
| GPT-OSS-20B | SSMoE (75%专家) | **54.1** (+10.9%) | **30.05 GB (-22.9%)** |
| GPT-OSS-120B | Original | 52.6 | 217.61 GB |
| GPT-OSS-120B | SSMoE (75%专家) | **54.9** (+4.4%) | **164.20 GB (-24.5%)** |

> 🔥 在`BoolQ`上提升高达 **+24%**，`GSM8K`上 **+17%**

#### ✅ 语言模型嵌入质量（MTEB平均分）
| 模型 | Router | SMoE | MoEE | EV (仅特征向量) | **SSMoE** |
|------|-------|------|------|------------------|------------|
| OLMoE-7B | 34.3 | 29.9 | 35.2 | 40.9 | **39.7** |
| Qwen-MoE-7B | 35.0 | 36.9 | 42.6 | 36.0 | **45.8** |
| DeepSeekMoE-16B | 33.9 | 32.1 | 35.9 | 35.5 | **41.1** |

> 💡 SSMoE在所有模型上均取得最佳表现，尤其在STS和Clustering任务中领先明显。

#### ✅ 视觉-语言任务（COCO零样本检索）
| 模型 | 方法 | I2T R@1 ↑ | T2I R@1 ↑ |
|------|------|----------|----------|
| CLIP-MoE | Baseline | 65.0 | 46.8 |
| SSMoE | Ours | **65.7 (+1.1%)** | **47.1 (+0.6%)** |
| SSMoE | Corrupt Setting | **37.2 (+2.8%)** | **30.0 (+2.0%)** |

> 🛡️ 在噪声环境下增益更大，显示更强鲁棒性。

### 与基线方法的对比结果
- SSMoE相比**hash-based routing**平均提升 **+18.0 pts**
- 相比**random routing**提升 **+9.1 pts**
- 相比**RandomDrop**在多数任务上显著更优（如GSM8K下降而SSMoE上升）
- 在**perplexity**指标上优于原模型（WikiText-2: 82.12 → 73.23），说明未破坏语言建模能力

### 消融实验结果

#### （1）特征向量选择方式
| 方法 | 平均提升 |
|------|--------|
| 仅平均特征向量（无router引导） | +3.9 pts |
| router引导选择top-c特征向量 | **+5.3 pts** |
> 表明router起到“过滤器”作用，而非决定性依赖。

#### （2）平衡因子 α 影响
- 最佳性能出现在 **α ≈ 0.9**，即更偏好特征向量路由信号
- α ∈ [0.5, 0.9] 范围内性能稳定，具备良好调参鲁棒性

#### （3）专家重叠分析
- SSMoE与原始router的专家选择**平均重叠率仅为15.3%**
> 表明SSMoE并非简单复制原有路由行为，而是提供了互补甚至更优的选择路径。

---

## 4. 关键结论和发现

### 主要发现
1. **✅ 特征向量蕴含丰富语义信息**  
   专家权重矩阵的特征向量自然编码了其功能专业化信息，可作为高质量语义表示。

2. **✅ 特征向量路由具有内在正交性**  
   理论证明与实验证实EV Router能有效缓解representation collapse，保持专家多样性。

3. **✅ SSMoE是一种高效、免训练的增强方案**  
   无需额外训练即可提升多个下游任务性能，适用于各类SMoE架构。

4. **✅ 少量专家足以完成复杂推理任务**  
   通过合理路由，即使移除25%专家，性能仍可超越完整模型，支持“稀疏即强”。

5. **✅ 更强的分布外鲁棒性**  
   在噪声输入下表现更稳健，表明其学到的是更具泛化的语义结构。

### 方法的局限性
- 当前方法主要适用于FFN-based专家结构，对其他形式专家（如attention-based）适配尚需探索。
- 特征向量提取引入一定计算开销（虽远小于训练成本），在极端低延迟场景可能受限。
- 对数学类任务（如GSM8K）敏感于专家数量，过度剪枝会导致性能下降。

### 未来工作方向
- 探索如何动态调整 `TOPC`（选取的特征向量数）或 `α` 以适应不同任务。
- 将SSMoE思想扩展至多模态MoE、Adapter MoE等新型稀疏架构。
- 结合SVD分析进行自动专家初始化或模型压缩。
- 研究特征向量空间的可解释性，用于诊断和干预专家行为。

---

> 📌 **一句话总结**：  
> SSMoE揭示了**专家权重的谱特性本身就是一种强大且抗坍缩的路由机制**，为构建高效、免训练、高鲁棒性的MoE系统开辟了新路径。

</details>

---

### 8. [Speculative Decoding Across Languages](https://arxiv.org/abs/2605.30580)

**Authors**: Nirajan Paudel, Michael Ginn, Luc De Nardi, Alexis Palmer  
**Category**: cs.CL  
**Published**: 2026-06-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.30580v1  

#### Abstract
Speculative decoding has become a crucial component of large language model (LLM) inference, enabling faster generation by drafting multiple tokens and verifying them in parallel. However, small draft models tend to suffer from disproportionately poor multilingual capabilities. Thus, when generating...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Speculative Decoding Across Languages》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文聚焦于**多语言场景下 speculative decoding 的低效问题**。  
- 在非英语语言中，由于小规模 draft model 的 multilingual 能力较弱，其生成的 token 与大规模 verifier model 的分布差异较大，导致 **acceptance rate 显著下降**。
- 此外，tokenization 偏差使得低资源语言需要更多 tokens，进一步加剧推理延迟。
- 因此，标准 speculative decoding 在非英语语言上往往无法带来加速，甚至可能变慢。

### **提出了什么新方法或新思路**
作者系统比较了三种提升多语言 speculative decoding 效率的方法，并提出一个反直觉但有效的发现：
1. **Task-specific distillation**：在目标任务（如英→目标语翻译）的平行语料上对 draft model 进行知识蒸馏。
2. **General-domain distillation**：在目标语言的单语语料上进行无监督蒸馏。
3. **N-gram draft models**：训练简单的 n-gram 模型作为 draft model，利用单语语料建模局部 token 序列模式。

> **创新点**：首次系统评估 speculative decoding 在跨语言、跨任务场景下的泛化能力，并揭示 **n-gram 模型尽管 acceptance rate 较低，但由于推理成本极低，反而能实现更高的 speed-up**。

### **相比现有方法的优势**
| 方法 | 优势 | 劣势 |
|------|------|------|
| Task-specific distillation | 在目标任务上显著提升 acceptance rate 和 speed-up | 泛化性差，在新任务（如故事生成）上性能骤降 |
| General-domain distillation | 稍微优于 baseline，具备一定泛化能力 | 提升有限，效果不稳定 |
| **N-gram models** | ✅ 推理速度快（~0.001s/step），cost ratio 极低<br>✅ 跨任务泛化能力强<br>✅ 只需单语语料即可训练 | ❌ acceptance rate 最低（平均 ~0.24） |

> **核心优势**：**n-gram 方法通过“廉价试错”策略实现了最优的实际加速效果**，为低资源语言提供了实用解决方案。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共覆盖 **11 种语言**，涵盖不同资源水平、地理区域和语言类型学特征：

| 语言 | 单语语料（训练/测试 token 数） | 平行语料（训练/测试句数） |
|------|-------------------------------|--------------------------|
| Amharic, Berber, Cherokee, Guarani, Hawaiian, Igbo, Nepali, Occitan, Quechua, Yoruba, Tamazight | 最高达 13M tokens（Nepali） | 最多 4800 句（多数语言） |

- **单语语料来源**：Leipzig Corpus, Tatoeba, Aya Dataset 等（见 Table 4）
- **平行语料来源**：Opus, Tatoeba, ChrEn, MENYO-20k 等（见 Table 5）

### **实验设置**
- **Verifier Model**：Qwen 3.5 9B
- **Draft Models**：
  - Baseline：Qwen 3.5 0.8B（未微调）
  - Distilled (task)：0.8B 在翻译任务上蒸馏
  - Distilled (general)：0.8B 在单语语料上蒸馏
  - N-gram model：基于 Qwen tokenizer 训练的 bigram 模型
- **解码参数**：top-k=100, top-p=0.9, max 128 tokens
- **KV caching** 启用
- **ry 值**（draft length）在 {2,4} 中搜索最优

### **评估指标**
1. **Acceptance Rate (α)**：
   $$
   \alpha = \frac{\text{accepted tokens}}{\text{drafted tokens}}
   $$
   衡量 draft token 被 verifier 接受的概率。

2. **Speed-up Factor (f)**：
   $$
   f = \frac{1 + y}{(1 - \alpha)(y c + 1)},\quad c = \frac{t_{\text{draft}}}{t_{\text{target}}}
   $$
   综合考虑 acceptance rate 和推理时间的成本比。

3. **Throughput (tokens/sec)**：实际生成速度测量。

### **基线方法对比**
- **Baseline**：原始 small model（0.8B）直接用于 speculative decoding
- **Distilled (task)** vs **Distilled (general)**：检验领域适配的影响
- **N-gram**：挑战“必须用 neural model 做 draft”的默认假设

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **翻译任务（In-domain）**
| 方法 | Avg Acceptance Rate (α) | Speed-up Factor (f) |
|------|------------------------|--------------------|
| Baseline | 0.40 | 1.02× |
| Distilled (task) | **0.60** | **1.28×** |
| Distilled (general) | 0.39 | 1.03× |
| N-gram | 0.30 | 1.30× |

> ✅ Task-specific distillation 显著提升 acceptance rate，但 n-gram 凭借极低成本实现更高 speed-up。

#### **故事生成任务（Out-of-domain）**
| 方法 | α | f |
|------|----|----|
| Baseline | 0.46 | 1.09× |
| Distilled (task) | 0.43 | **1.03×**（退化！）|
| Distilled (general) | 0.47 | 1.10× |
| **N-gram** | 0.24 | **1.39×** |

> ⚠️ **Task-specific 模型泛化失败**：在新任务上 performance 下降；
> ✅ **N-gram 模型表现最佳**：speed-up 高达 **1.39×**，远超其他方法。

### **与基线方法的对比结果**
- 所有语言中，**baseline speculative decoding 几乎无效**（f ≈ 1.02×），说明标准方法不适用于多语言场景。
- **Task-specific distillation** 在翻译任务上有效，但在故事生成上不如 baseline。
- **N-gram 模型在 10/11 语言上 beat baseline**，即使 acceptance rate 最低。

### **消融实验结果**
#### **D.3 Draft Model Size（0.8B vs 2B vs 4B）**
- 更大 draft model 通常带来更高 acceptance rate 和 speed-up。
- 但在 Amharic 上，4B 模型出现 speed-up 下降，表明存在收益递减。

#### **D.4 Translation Quality vs Acceptance Rate**
- 测试了 **chrF++ 分数与 acceptance rate 的相关性**。
- 结果：**相关系数 r = 0.170**，几乎无相关性。
> 表明 acceptance rate 不取决于模型翻译质量，而是与 draft/verifier 分布匹配度更相关。

#### **D.5 Distillation Lower Bound**
- 理论推导出 acceptance rate 的下界：
  $$
  \alpha \geq 1 - \sqrt{2 \cdot D_{KL}(P \| Q)}
  $$
- 实验验证所有蒸馏模型均超过该下界，但 KL divergence 与 α 之间无明显负相关。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **标准 speculative decoding 对非英语语言无效**：acceptance rate 普遍低于 0.5，speed-up 接近 1×。
2. ✅ **Task-specific distillation 提升 in-domain performance，但严重过拟合**：在新任务上表现劣于 baseline。
3. ✅ **N-gram draft models 尽管 acceptance rate 低，却因极快推理速度实现最高 speed-up**（最高达 **1.39×**）。
4. ✅ **General-domain distillation 效果有限**：仅轻微优于 baseline。
5. ✅ **Forward pass time 差异是决定性因素**：
   - N-gram: ~0.001s
   - 0.8B LLM: ~0.033s
   > 成本比 $ c = t_{\text{draft}} / t_{\text{target}} $ 极小，使 n-gram 即便低 accept 也能高效。

### **方法的局限性**
- 仅在 **Qwen 3.5 系列模型** 上验证，结果是否适用于其他 model family（如 Llama、Mistral）尚不确定。
- 仅测试 **11 种语言**，且均为有一定数字资源的语言，极端低资源语言（如仅有几百句）未覆盖。
- 仅评估两个任务（MT 和 story generation），复杂任务（如数学推理、代码生成）未涉及。
- N-gram 模型无法捕捉长距离依赖，仅适合局部 coherent 文本生成。

### **未来工作方向**
1. **动态 draft model 切换机制**：根据当前任务和语言自动选择最优 draft model（neural vs n-gram）。
2. **Hybrid draft models**：结合 n-gram 的高速与 small LLM 的语义理解能力。
3. **轻量神经架构探索**：设计专用于 speculative decoding 的 ultra-fast draft model（如 TinyBERT、MobileLLM）。
4. **多语言 tokenization 优化**：缓解 token inflation 问题，提升低资源语言效率。
5. **绿色 AI 视角下的 speculative decoding**：量化 energy saving，推动可持续推理技术。

---

> **一句话总结**：  
> 在多语言 speculative decoding 中，**“快而粗”的 n-gram draft model 比“慢而准”的蒸馏小模型更有效**，揭示了 **cost-aware drafting** 的重要性，为低资源语言推理优化提供了新路径。

</details>

---

### 9. [dMoE: dLLMs with Learnable Block Experts](https://arxiv.org/abs/2605.30876)

**Authors**: Sicheng Feng, Zigeng Chen, Gongfan Fang, Xinyin Ma, Xinchao Wang  
**Category**: cs.CL  
**Published**: 2026-06-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.30876v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have recently emerged as a promising alternative to autoregressive models, offering competitive performance while naturally supporting parallel decoding. However, as dLLMs are increasingly integrated with Mixture-of-Experts (MoE) architectures to scale model c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：dMoE: dLLMs with Learnable Block Experts

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Diffusion Large Language Models (dLLMs)** 中，尽管其天然支持并行解码（parallel decoding），但当与 **Mixture-of-Experts (MoE)** 架构结合时，出现了一个根本性的效率瓶颈：

- **传统 MoE 采用 token-level 路由**：每个 token 独立选择专家。
- **而 dLLMs 在一个 forward pass 中处理整个 block 的 tokens**，具有双向依赖关系。

这种“**块级处理 vs. 令牌级路由**”的不匹配导致：
- 单次推理中激活的 **unique experts 数量急剧增加**；
- 导致 **内存访问成为主要瓶颈**（memory-bound），严重影响推理速度和显存占用。

---

### 🚀 提出的新方法：dMoE
作者提出 **dMoE** —— 一种简单而有效的 **block-level MoE 框架**，核心思想是：

> 将 token-level 的专家分布聚合为 **block-level 的统一专家分布**，并以此指导整个 block 的专家路由。

#### 具体实现机制：
1. **聚合 token-level router scores** 得到 block-level expert scores；
2. 对 block-level scores 应用 **top-p 动态阈值**（而非固定数量）来确定一个共享的 **expert coreset**；
3. 所有 block 内的 token 只能从该 coreset 中选择专家，从而显著减少 unique expert 数量；
4. 训练阶段采用 **self-distillation** 范式，保持训练与推理一致性。

---

### 🔍 相比现有方法的优势
| 方面 | dMoE 的优势 |
|------|-------------|
| **效率提升** | 显著降低 unique expert 数量 → 减少内存带宽压力，缓解 memory-bound 问题 |
| **性能保留** | 几乎无损模型性能（保留原始模型 99.11% 性能） |
| **动态适应性** | 使用 top-p 策略自适应不同 denoising step 和 block 的路由集中度变化 |
| **无需额外训练开销** | 不改变每 token 选中的 expert 数量，仅限制搜索空间 |
| **优于同类方法** | 在同等压缩率下，性能远超 DES 等 baseline |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **MATH500**：数学推理任务
- **GSM8K**：小学级别数学应用题
- **ARC-C**：科学常识推理
- **MMLU**：多领域知识理解（使用其中的 math 子集）

所有数据均通过 **self-distillation** 生成，监督信号来自 base model 自身输出。

---

### ⚙️ 实验设置
| 项目 | 设置说明 |
|------|----------|
| **Base Model** | LLaDA2.0-mini（开源 MoE dLLM） |
| **训练方式** | Full fine-tuning，2 epochs，batch size=4 |
| **学习率** | 2.0×10⁻⁶，cosine schedule |
| **Block Size** | 32（默认） |
| **Top-p Threshold** | 训练时设为 0.6 |
| **硬件平台** | 4×NVIDIA H100 GPU |
| **评估框架** | Transformers + SGLang |

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 各 benchmark 上的任务准确率 |
| **Unique Expert Count** | 每层平均激活的独特专家数（越低越好） |
| **Memory Usage** | MoE 参数的显存占用 |
| **End-to-End Latency** | 完整推理延迟，含 MoE、Attention 等组件 |
| **Speedup** | 相对于原模型的端到端加速比 |

---

### 🆚 基线方法对比
| 方法 | 说明 |
|------|------|
| **Original** | 原始 LLaDA2.0-mini，block diffusion 设置 |
| **Top-4** | 每个 token 仅选前 4 个专家（减少计算） |
| **DES-S** | Sequence-level routing（作者复现） |
| **DES-V** | Vote-based block-level routing（作者复现） |
| **DES-S\*/DES-V\*** | 调整参数以达到更高压缩率的版本 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（平均 across 四个 benchmark）
| 指标 | 原始模型 | dMoE（ours） | 提升/下降 |
|------|--------|------------|-----------|
| **Unique Expert Count** | 69.5 | **14.6** | ↓ **79.04%** |
| **Performance** | 83.95% | **83.2%** | 保留 **99.11%** |
| **Memory Usage** | - | ↓ **76.64% ~ 79.84%** | 显著降低 |
| **Latency Speedup** | - | **1.14× ~ 1.66×** | 端到端加速 |

> 示例：在 MATH500 上，专家数从 70.0 降至 14.1，准确率从 72.0% → 71.0%

---

### 🔁 与基线方法对比（Table 2）
| 方法 | Unique Experts | Performance Retention |
|------|----------------|------------------------|
| **DES-S** | ~41 | ~98.6% |
| **DES-V** | ~37 | ~97.6% |
| **DES-S\*** | ~13 | ↓ 至 ~74%（严重下降） |
| **dMoE (p=0.6)** | **14.1** | **99.11%** |
| **dMoE (p=0.5)** | **11.4** | **97.50%** |

✅ **结论**：dMoE 在相似甚至更低的专家数下，**性能损失极小**，明显优于 DES 系列方法。

---

### 🔍 消融实验结果

#### ✅ Ablation on Cumulative Probability Threshold $ p $
（见 Table 3）

| $ p_{\text{test}} $ | Unique Experts | Accuracy |
|-----------------------|----------------|----------|
| 0.4 | 10.3 | 69.8% |
| 0.5 | 11.5 | 70.0% |
| 0.6 | 14.1 | 71.0% |
| 0.7 | 19.0 | 72.6% |
| 0.8 | 27.1 | 74.2% |

📌 发现：
- $ p $ 越大 → 专家越多 → 性能越高；
- 方法具有良好的 **可调性（tunability）**，可根据硬件资源灵活调节压缩强度。

#### ✅ Ablation on Block Size
（见 Table 4）

| Block Size | 压缩率（Unique Experts ↓） | 性能保留 |
|----------|----------------------------|---------|
| 8 | ↓79.04% | 99.11% |
| 16 | ↓73.82% | 100.63% |
| 24 | ↓66.33% | 99.53% |
| 32 | ↓53.92% | 99.64% |

📌 发现：
- 在不同 block size 下均有效，**压缩效果随 block size 减小而增强**；
- **性能无明显下降**，验证了方法鲁棒性。

---

### 📊 性能-效率权衡分析（Figure 6）
- dMoE 在相同 unique expert 数量下，**性能始终高于 DES-V**；
- 或者说，在相同性能水平下，**dMoE 激活更少的专家**；
- 实现了 **更优的 performance-efficiency trade-off**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MoE dLLMs 的主要瓶颈是 memory-bound**，源于 token-level 路由导致大量 unique experts 被激活。
2. **token-level router scores 是专家重要性的有效指示器**，可用于指导 block-level 路由。
3. **expert concentration 随 denoising step 和 block 动态变化**，需动态控制 coreset 大小。
4. **dMoE 通过 block-level aggregation + top-p coreset selection**，实现了高效且几乎无损的专家压缩。
5. **dMoE 可压缩至接近理论极限（≈8 experts/token）仍保持高性能**。

---

### ⚠️ 方法的局限性
1. **目前仅验证于语言模态**，尚未扩展至 vision/video 等多模态场景；
2. **未进一步优化计算成本**（如减少每 token 的 expert 数量）；
3. **依赖 self-distillation 进行训练**，对高质量合成数据有一定要求；
4. **top-p 阈值需要手动设定**，缺乏完全自动化调节机制。

---

### 🔮 未来工作方向
1. **扩展至多模态 dLLMs**（如 diffusion VLMs），探索跨模态 block routing；
2. **联合优化 computation 与 memory**：不仅限制 unique experts，也尝试减少 per-token expert 数；
3. **极端压缩方向**：鼓励 block 内所有 token 共享同一 expert group；
4. **动态自适应 p-threshold**：根据输入复杂度自动调整压缩强度；
5. **系统级集成**：与 SGLang、Tensor Parallelism 等推理引擎深度整合。

---

## 💡 总结
**dMoE** 是首个针对 **MoE dLLMs** 设计的 **learnable block-level expert routing** 方法。它巧妙地解决了 **parallel decoding 与 token-level MoE routing 的结构性矛盾**，通过聚合 token-level 信息生成 block-level coreset，在几乎不牺牲性能的前提下，将 unique expert 数量降低近 **5倍**，带来高达 **1.66× 的端到端加速** 和 **近 80% 的显存节省**，为大规模 dLLMs 的高效部署提供了强有力的技术路径。

> 🔗 开源地址：[https://github.com/fscdc/dMoE](https://github.com/fscdc/dMoE)

</details>

---

### 10. [Parallel Tempering Initial Sampling in Inference-Time Reward Alignment](https://arxiv.org/abs/2605.30991)

**Authors**: Myeongjun Oh, Gwangho Kim, Sungyoon Lee  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.30991v1  

#### Abstract
Inference-time reward alignment steers pretrained diffusion and flow-based generative models to satisfy user-specified rewards without retraining. Recently, Sequential Monte Carlo (SMC) has emerged as a powerful framework for this task by iteratively filtering and propagating multiple particles. How...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Parallel Tempering Initial Sampling in Inference-Time Reward Alignment**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 **Sequential Monte Carlo (SMC)** 的 **inference-time reward alignment** 方法在复杂奖励场景下表现不佳，主要原因有两个：
- **初始化偏差（Poor Initialization）**：标准 SMC 方法从无奖励感知的先验分布（如 $\mathcal{N}(0, I)$）初始化粒子，而高奖励区域在复杂任务中极其稀疏，导致初始粒子难以覆盖目标区域。
- **模式捕获（Mode Trapping）**：即使像 **V-Sampler** 这样通过 pCNL 算法进行奖励感知初始化的方法，在多模态奖励景观中仍容易陷入局部最优，无法有效探索全局高奖励区域。

### **提出的新方法：PATHS**
作者提出了 **PATHS (PArallel Tempering for High-complexity reward Sampling)**，一种用于推理时奖励对齐的新型初始化框架，其核心思想是：
- 在初始化阶段引入 **Parallel Tempering (PT)**，即构建一个温度阶梯（temperature ladder），运行多个不同温度下的 pCNL 采样链。
- 高温链（hot chains）在平坦化的奖励景观中自由探索，发现潜在的高奖励状态。
- 低温链（cold chain, $T=1$）专注于精细优化，但易陷入局部模式。
- 通过周期性的 **Metropolis swap** 在相邻链之间交换状态，使低温链能继承高温链发现的优质配置，从而突破局部障碍。

### **相比现有方法的优势**
- **更有效的探索机制**：克服了独立 MCMC 链的“模式捕获”问题，实现了跨模式的信息传递。
- **无需额外训练**：作为纯推理时方法，不修改预训练模型，保持灵活性。
- **预算匹配下的性能提升**：在相同 reward-model evaluation 预算下，显著优于 SMC 和 V-Sampler 等方法。
- **特别适用于复杂任务**：在布局生成、数量控制等多模态、稀疏奖励任务中优势明显。

---

## **2. 核心实验方法和设置**

### **使用的数据集与任务**
论文在两个具有复杂奖励景观的任务上进行评估：
- **Layout-to-Image Generation**  
  输入为对象短语和目标边界框，要求生成图像满足空间布局约束。  
  数据来源：`selected_prompts_data.json`，按对象数量分为：
  - **Simple**（≤3 objects, N=30）
  - **Complex**（4 objects, N=20）
- **Quantity-Aware Sampling**  
  要求生成指定数量的目标对象（如“82 blueberries”）。  
  数据来源：`quantity_aware_selected_20.json`，按目标数量分：
  - **Simple**（<25 objects, N=30）
  - **Complex**（≥25 objects, N=30）

### **实验设置**
- **基础模型**：FLUX.1-schnell（flow-based 模型）
- **总预算**：1000 次 reward-model 调用
  - 初始化阶段：500 次
  - SMC 推理阶段：500 次（20 particles × 25 steps）
- **PATHS 设置**：
  - 温度链数 $L=4$
  - 每链粒子数 $C=1$
  - Burn-in: 65，Post-burn-in: 60
  - Swap interval: 每 5 步
  - 温度阶梯：
    - Layout: `{1, 2, 4, 8}`
    - Quantity: `{1, 4, 16, 64}`

### **评估指标**
| 任务 | 指标 | 说明 |
|------|------|------|
| **Layout-to-Image** | `GroundingDINO+` ↑ | 布局对齐得分（用于引导） |
| | `mIoU` ↑ | 使用 Salience DETR 的 hold-out 分割质量 |
| | `ImageReward` ↑ | 图像偏好得分（hold-out） |
| | `VQA Score` ↑ | 文本-图像对齐（hold-out） |
| **Quantity-Aware** | `T2I-Count+ ↓` | 计数误差（越低越好） |
| | `ImageReward` ↑ | 图像质量 |
| | `VQA Score` ↑ | 对齐能力 |

### **基线方法**
- **Prior-initialized SMC**：
  - **TDS** [44]
  - **DAS** [22]
- **Posterior-initialized**：
  - **V-Sampler** [50]：使用 pCNL 从奖励感知后验采样
  - **Best-of-4**：运行 4 条独立冷链，选择最高奖励链

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

#### **Layout-to-Image 任务（Complex 子集）**
| 方法 | GroundingDINO+ | mIoU | ImageReward | VQA Score |
|------|----------------|------|-------------|-----------|
| TDS | 0.261 | 0.287 | 1.034 | 0.618 |
| DAS | 0.226 | 0.265 | 0.861 | 0.608 |
| V-Sampler | 0.239 | 0.266 | 0.625 | 0.580 |
| Best-of-4 | 0.265 | 0.273 | 0.781 | 0.588 |
| **PATHS (Ours)** | **0.266** | **0.274** | **1.411** | **0.706** |

> ✅ **PATHS 在 ImageReward 上提升超过 50%**，VQA 提升显著。

#### **Quantity-Aware 任务（Complex 子集）**
| 方法 | T2I-Count+ ↓ | ImageReward | VQA Score |
|------|--------------|-------------|-----------|
| TDS | 3.250 | -0.300 | 0.546 |
| DAS | 3.108 | -0.325 | 0.555 |
| V-Sampler | 4.181 | 0.104 | 0.616 |
| Best-of-4 | 2.463 | 0.170 | 0.608 |
| **PATHS (Ours)** | **1.725** | **0.202** | **0.657** |

> ✅ **计数误差降低约 30%**，且图像质量和对齐均领先。

### **与基线方法的对比结果**
- **远超 SMC 方法（TDS/DAS）**：在复杂子集上全面领先，尤其在高阶语义对齐（VQA）和图像质量（ImageReward）上。
- **优于 V-Sampler 和 Best-of-4**：
  - 表明 **简单增加链数（brute-force）无法替代 replica exchange 的信息共享机制**。
  - Best-of-4 依赖“幸运初始化”，而 PATHS 主动打破局部陷阱。
- **在简单任务上增益较小**：说明其优势集中在 **multi-modal、compositional** 场景。

### **消融实验与分析（隐含）**
- **温度阶梯设计敏感性分析（Figure 6）**：
  - 不同任务需不同宽度的温度阶梯（layout 用 `{1,2,4,8}`，quantity 用 `{1,4,16,64}`）。
  - 验证了 **exploration-communication trade-off**：太窄则探索不足，太宽则耦合弱。
- **swap 机制可视化（Figure 8–9）**：
  - 显示高温链发现完整布局后，通过 swap 成功传递给低温链，实现“跳跃式优化”。

---

## **4. 关键结论和发现**

### **主要发现**
1. **初始化质量决定推理时对齐上限**：即使有先进的 twisting 或 tempering，若初始粒子未覆盖高奖励区域，后续优化无效。
2. **Parallel Tempering 是解决 mode trapping 的有效机制**：通过跨温度链的状态交换，实现了 **global exploration → local exploitation** 的协同。
3. **PATHS 特别适合复杂、组合性强的任务**：在布局生成和精确计数等任务中表现卓越，验证了其设计动机。

### **方法的局限性**
- **仅在多模态奖励下有效**：在平滑、单峰奖励（如美学评分）任务中增益有限（见 Table 3 和 Figure 7）。
- **依赖 reward landscape 的结构特性**：若奖励信号无法区分不同构型（如仅关注视觉美感而非文本对齐），则优势减弱。
- **温度阶梯需手动调参**：目前采用任务级校准，尚未实现自适应温度调度。

### **未来工作方向**
- **自适应温度调度**：根据 reward landscape 动态调整温度阶梯。
- **per-prompt ladder calibration**：为每个提示词自动选择最优温度配置。
- **扩展到其他生成模型**：如 autoregressive models 或 video generation。
- **结合 fine-tuning 与 inference-time 方法**：探索 hybrid alignment 策略。

---

> **总结**：PATHS 揭示了 **inference-time reward alignment** 中 **初始化策略** 的核心地位，并通过 **Parallel Tempering + replica exchange** 提供了一种系统性解决方案，尤其适用于当前前沿的 **structured generation** 任务（如 layout、counting、spatial reasoning），为未来高精度可控生成提供了新范式。

</details>

---

### 11. [Fixed-Point Masked Generative Modeling](https://arxiv.org/abs/2605.31215)

**Authors**: Andrea Miele, Yiming Qin, Alba Carballo-Castro, Justin Deschenaux, Pascal Frossard  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.31215v1  

#### Abstract
Masked Generative Models (MGMs) enable parallel decoding and achieve strong performance across modalities, but require full-sequence bidirectional transformers at every step, making training costly and degrading quality under low sampling budgets. Existing work improves efficiency via better sampler...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Fixed-Point Masked Generative Modeling**

---

## **1. 主要贡献和创新点**

### **解决的问题**
Masked Generative Models (MGMs)，如 MDLM 和 MaskGIT，在生成建模中表现出色，支持并行解码且跨模态性能强。然而，其训练成本高昂，因为每一步去噪都需要对整个序列进行全双向 Transformer 前向传播。此外，在低采样预算（即有限的 Transformer 块前向传递次数）下，生成质量显著下降。

现有方法通过改进采样器或设计更高效的固定深度架构来提升效率，但仍存在以下局限：
- 每步分配的计算量是固定的，无法根据步骤难度动态调整；
- 架构冗余未被充分利用；
- 缺乏对跨步表示一致性的建模。

### **提出的新方法：FP-MGM 与 CoFRe 框架**
本文提出了 **Fixed-Point Masked Generative Models (FP-MGMs)**，将部分去噪器替换为共享注意力层上的 **fixed-point solver**，从而实现自适应深度和参数减少。

在此基础上，构建了完整的训练到推理框架 **CoFRe**，包含两大核心机制：

#### ✅ **(1) Cross-step Consistency Loss**
- 引入一个跨时间步的一致性损失 $ C_{\text{cons}} $，对齐相邻去噪步骤中的隐藏状态。
- 行为类似于 **cross-time self-distillation**，使模型在噪声较大的学生状态下预测更接近干净教师状态的结果，显著提升低预算下的生成质量。

#### ✅ **(2) Three-State Reuse (3SR)**
- 在推理时 warm-start 固定点求解器，但针对三种 token 类型分别处理：
  - **unchanged visible tokens**：完全复用上一步的隐状态；
  - **still-masked tokens**：部分复用；
  - **newly revealed tokens**：更多依赖当前输入注入。
- 这种 token-aware 初始化策略大幅提升了求解器收敛速度和生成稳定性。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **效率** | 参数减少、训练时间缩短、VRAM 占用降低 |
| **性能** | 显著改善低采样预算下的生成质量（如 Gen. PPL、FID） |
| **灵活性** | 支持自适应计算深度，不同去噪步可动态调整迭代次数 |
| **兼容性** | 可将预训练 MGM 快速转换为 FP-MGM，无需从头训练 |

---

## **2. 核心实验方法和设置**

### **数据集**
- **语言建模**：OpenWebText (OWT)，上下文长度 1024，使用 GPT-2 tokenizer
- **图像生成**：ImageNette（ImageNet 的 10 类子集），分辨率 256×256，使用 VQ-GAN tokenizer 编码为 16×16 latent tokens

### **评估指标**
| 模态 | 指标 | 说明 |
|------|------|------|
| **文本** | Generative Perplexity (Gen. PPL) | 使用 GPT-2 Large 对生成样本打分，越低越好 |
| | Unigram Entropy | 衡量生成多样性，越高越好 |
| | MAUVE | 衡量生成文本与真实分布之间的差异 |
| **图像** | FID (Fréchet Inception Distance) | 越低越好，衡量生成图像质量与真实性 |
| | Inception Score (IS) | 越高越好，衡量多样性和清晰度 |

### **基线方法对比**
- **MDLM**：原始 Masked Diffusion Language Model
- **MDLM + SDTT**：带 Self-Distillation Through Time 的增强版
- **MaskGIT-Large**：大型图像生成基线
- **PGM**, **IDLM-MDLM**：其他加速扩散模型方法

所有比较均基于相同的 **transformer-block forward-pass 预算**（例如 96、192、384、768），确保公平。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### 🔹 **语言建模（OpenWebText）**

| 方法 | 参数量 | 训练时间 | VRAM | Gen. PPL @96 | Gen. PPL @768 |
|------|--------|----------|-------|--------------|----------------|
| MDLM | 170M | ~139h | 112.4 GiB/GPU | 830.8 → 193.1 (with SDTT) | 143.9 → 47.0 (with SDTT) |
| **CoFRe** | **104M (-38.8%)** | **~123h (-11.5%)** | **93.4 GiB/GPU (-16.9%)** | **101.8** | **37.8** |

> ✅ **结论**：CoFRe 在更低资源消耗下，实现了远超 MDLM+SDTT 的生成质量。

#### 🔹 **图像生成（ImageNette）**

| 方法 | 训练时间 | VRAM | FID @48 | FID @384 |
|------|--------|-------|---------|-----------|
| MaskGIT-Large | 17h46m | 72.45 GiB | 174.1 | 30.0 |
| **CoFRe** | **9h08m (-48.6%)** | **35.74 GiB (-50.7%)** | **96.7** | **22.8** |

> ✅ **结论**：CoFRe 不仅训练更快、内存更少，且在所有采样预算下 FID 更优。

---

### **消融实验结果**

#### ✅ **组件有效性分析（Table 9 & F.1.4）**
| 方法 | Gen. PPL @96 | Gen. PPL @768 |
|------|---------------|----------------|
| FP-MDLM (base) | 375.6 | 179.7 |
| + Lcons | 104.2 | 41.7 |
| + Lcons + 3SR | **101.8** | **37.8** |

> 📌 **发现**：`Cross-step consistency` 是最大增益来源；`3SR` 进一步优化 warm-start 效果。

#### ✅ **初始化方式影响（Table 8）**
- 使用预训练 MDLM 权重初始化 FP-MDLM，仅需 **4% 的原始训练步数（40k steps）** 即可超越从零训练的 FP-MDLM。
- 在 budget 96 下，Gen. PPL 从 296.8 降至 **276.0**，验证了高效迁移能力。

#### ✅ **预算分配策略（F.10）**
- 最佳策略为 **decreasing schedule**：早期去噪步分配更多 fixed-point iterations。
- 原因：初始阶段噪声最多，最难恢复，需要更强的去噪能力。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Fixed-point architecture 显著提升效率**  
   - 通过 weight-sharing 减少参数（-38.8%）、训练时间（-11.5%）、VRAM（-16.9%）
   - 支持动态控制有效深度，适应不同去噪难度

2. **Cross-step consistency 是低预算生成的关键**  
   - 提供跨时间步的监督信号，行为类似 self-distillation
   - 大幅降低 Gen. PPL，尤其在 budget ≤ 192 时效果最明显

3. **Three-State Reuse 实现 token-aware warm-start**  
   - 区分三类 token 的复用强度，避免错误传播
   - 结合 consistency 后，reuse 成为稳定增益项

4. **预训练模型可快速迁移到 FP 架构**  
   - 无需从头训练，短 distillation 阶段即可获得高质量 FP-MGM

---

### **局限性**
1. **适用范围限制**  
   - 当前方法假设单调去噪过程（token 一旦揭示不再遮蔽），若使用 remasking 策略需重新设计 reuse 规则。

2. **额外调参负担**  
   - 需要选择 fixed-point 层数、solver iteration 分配、reuse 系数等，目前仍依赖经验或网格搜索。

3. **硬件开销不完全反映**  
   - 虽然 forward-pass 数减少，但 fixed-point solver 带来额外控制流开销，实际 wall-clock 加速可能受限。

4. **大规模扩展尚未验证**  
   - 实验限于 OWT 和 ImageNette，尚不清楚是否能扩展至更大模型或跨模态任务。

---

### **未来工作方向**
- 设计 **adaptive stopping rules** 替代固定 iteration 数
- 开发 **optimized kernel 实现** 以减少控制流延迟
- 探索 **generalized reuse rules** 支持非单调采样路径（如 remasking）
- 研究 FP-MGM 在 **multimodal generation** 中的应用潜力
- 分析效率增益与 **bias/memorization/safety** 的关系

---

> 💡 **总体评价**：  
> CoFRe 提供了一个实用且高效的框架，使得 MGMs 更便宜、更灵活，并在低采样预算下表现更强。它不仅是架构改进，更是迈向 **practical, scalable masked generation** 的重要一步。

</details>

---

### 12. [EHRBench: An Automated and Reliable EHR-based Benchmark for Clinical Decision Making with LLMs](https://arxiv.org/abs/2605.30637)

**Authors**: Yuzhang Xie, Keqi Han, Yunpeng Xiao, Hejie Cui, Guanchen Wu, Ziyang Zhang, Kai Shu, Jiaying Lu, Xiao Hu, Carl Yang  
**Category**: cs.AI  
**Published**: 2026-06-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.30637v1  

#### Abstract
Clinical decision-making (CDM) is central to real-world clinical workflows, where clinicians infer diagnoses, select treatments, or anticipate future health outcomes under incomplete evidence. LLMs are increasingly used to support these decisions due to strong language capabilities, broad biomedical...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# EHRBench: An Automated and Reliable EHR-based Benchmark for Clinical Decision Making with LLMs 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前在评估 **Large Language Models (LLMs)** 在临床决策支持中的可靠性时，存在以下关键挑战：
- **现有医学 QA 基准**（如 MedQA、MedMCQA）多基于教科书、考试题或临床指南，其推理过程显式明确，缺乏对真实世界中隐含逻辑的考验。
- **基于电子健康记录 (EHR)** 的基准往往侧重于信息检索（如“患者本次就诊用了什么药”），而非需要实质性生物医学知识和临床推理的**决策任务**（如“给定诊断，应开什么药”）。
- 手动构建高质量医疗基准成本高昂，难以实现大规模；而完全依赖 LLM 生成则易产生幻觉（hallucination），导致临床关系不准确。

### 提出的新方法与思路
作者提出了 **EHRBench** ——一个**自动化且可靠的大规模 EHR 基准**，用于评估 LLMs 在临床决策中的能力。其核心是 **EHR-LLM-KB 交互流水线**：
1. **EHR → LLM**: 使用专门的 **Medical LLM**（HuatuoGPT-o1-8B）从原始结构化 EHR 轨迹中自动提取潜在的临床关系（如 `Hyperglycemia` → `Treat-with` → `Insulin`）和上下文事件。
2. **LLM → KB**: 利用外部**知识库 (Knowledge Base, KB)**（整合 UMLS、SemMedDB、DrugBank 等）对提取的关系进行系统性验证与丰富：
   - 验证：检查 KB 中是否存在支持该关系的证据（正向验证）。
   - 过滤：排除存在矛盾证据或与上下文冲突的关系。
3. **模板实例化**: 将验证后的结构化模板确定性地实例化为多种类型的 QA 项目（MCQ 和 OEQ）。

### 相比现有方法的优势
- **规模化与可靠性兼顾**：通过 LLM 实现自动化生成，通过 KB 实现可靠性保障，解决了“规模 vs. 质量”的权衡难题。
- **真实世界导向**：直接基于真实患者的结构化 EHR 数据，捕捉个性化、纵向的临床模式，更贴近实际临床工作流。
- **聚焦核心临床推理**：设计了三个需要实质性推理的任务（诊断、治疗、预后），超越了简单的阅读理解或信息检索。
- **开源与可复现**：发布了近百万级（960,067）的 QA 数据集，为社区提供了宝贵的评估资源。

---

## 2. 核心实验方法和设置

### 使用的数据集
EHRBench 的原始 EHR 数据来源于三个真实世界来源：
- **MIMIC-III** (Version 1.4): 包含 38,597 名患者，53,423 次住院记录。
- **MIMIC-IV** (Version 3.1): 更新版本，包含 364,627 名患者，546,028 次住院记录。
- **PROMOTE**: 一个来自埃默里大学医疗系统的私有数据集，包含 18,561 名患者，912,706 条临床记录，用于减少公共数据污染风险。

### 实验设置和评估指标
- **评估任务**：三大核心临床决策任务：
  - **Diagnosis (Dx)**: 给定同一就诊中的其他诊断，推断缺失的共病诊断。
  - **Treatment (Tx)**: 给定诊断，推断最可能开具的治疗（处方或手术）。
  - **Prognosis (Px)**: 给定前次就诊的诊断/治疗，预测下次就诊可能出现的诊断。
- **问题格式**：4/5/6 选项的多项选择题 (MCQ) 和开放式问题 (OEQ)。
- **评估协议**：统一的零样本 (zero-shot) 推理协议。
- **主要指标**：
  - **MCQ**: 准确率 (Accuracy, ACC)。
  - **OEQ**: 覆盖率 (Coverage, RC)、ROUGE-1、ROUGE-L、BERTScore。

### 基线方法对比
研究评估了超过 **30 个代表性 LLMs**，分为三类：
- **开源通用 LLMs**: 如 `glm4`, `llama3`, `qwen`, `mistral` 系列。
- **医学领域 LLMs**: 如 `doctor-r1-8b`, `med42-8b`, `ultramedical-8b`, `m1-7b-23k`, `m1-32b-1k`。
- **HIPAA 合规 API 模型**: 如 `gpt-4.1`, `gpt-5`, `gpt-5.2` 等。

此外，还与非 LLM 的嵌入式检索基线（如 `SapBERT`, `PubMedBERT`）进行了比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **总体表现**：最强模型 `gpt-5.2` 在所有任务上的**总体准确率达到 70.91%**，平均排名为 1.69，表现稳定。
- **任务难度差异**：不同任务间性能差异显著，符合临床直觉：
  - **Treatment (Tx)**: 平均准确率最高 (**69.33%**)，因为药物-适应症关系通常更直接、明确。
  - **Diagnosis (Dx)**: 平均准确率 **55.02%**。
  - **Prognosis (Px)**: 最具挑战性，平均准确率最低 (**46.67%**)，因其需要复杂的纵向推理。
- **模型规模效应**：在同一模型系列内，更大参数的模型普遍表现更好（如 `qwen3-32b` > `qwen3-8b` > `qwen3-4b`）。

### 与基线方法的对比结果
- **API 模型 vs. 开源模型**：最先进的 API 模型（`gpt-5.2`, `gpt-4.1`）表现最佳。领先的开源模型（如 `llama3.3-70b`, `qwen3-32b`）也极具竞争力，与顶级 API 模型差距仅 3-4 个百分点。
- **医学 LLMs vs. 通用 LLMs**：令人意外的是，**医学领域的微调并未带来一致的性能提升**。例如，`m1-32b-1k`（医学版）的准确率为 63.21%，而其基础模型 `qwen2.5-32b` 为 64.97%。这表明当前的医学微调策略在处理真实的 EHR 推理任务上仍有不足。
- **LLMs vs. 嵌入式检索基线**：嵌入式检索方法（如 `PubMedBERT`，总准确率 32.8%）远低于强大的 LLMs（如 `gpt-5.2` 达 66.8%），尤其是在 Treatment 任务上（26.2% vs. 77.3%），证明了简单语义匹配无法解决复杂的临床推理问题。

### 消融实验结果
- **稳健性分析**：通过更换不同的 LLM 作为生成器（`HuatuoGPT-o1-7B`, `m1-7b-23k`）或改变上下文事件数量，发现模型间的相对排名高度稳定（Kendall's W = 0.937），证明了 EHRBench 评估结果的鲁棒性。
- **扩展评估**：在完整数据集和开放性问题上的测试结果与主实验一致，进一步验证了结论的可靠性。

---

## 4. 关键结论和发现

### 主要发现
1. **EHRBench 是一个可靠且有效的基准**：其评估结果与已知的模型能力趋势一致，能够有效区分不同 LLMs 在临床决策任务上的表现。
2. **任务难度层级分明**：`Tx > Dx > Px` 的性能排序反映了临床推理的真实复杂度。
3. **医学微调的局限性**：当前的医学领域适配技术未能显著超越其通用基础模型，说明 EHR 场景下的临床推理需要更复杂的训练信号（如大规模临床案例监督、决策导向目标）。
4. **规模与效率的权衡**：更大的模型虽然性能更强，但推理速度更慢，成本更高。例如，`gpt-5.2` 性能最好但耗时最长。

### 方法的局限性
- **模态单一**：目前仅使用了结构化的诊断、处方和手术，未包含人口统计学、生命体征、实验室检查、影像等重要模态。
- **上下文窗口有限**：为了控制信息泄露和验证可行性，每个模板只使用了少量上下文事件，限制了长程跨就诊推理的能力。
- **知识库覆盖偏差**：KB 验证策略优先保证精度，牺牲了召回率，可能导致一些合理但未被 KB 收录的新兴或机构特异性实践被排除。
- **计算成本**：对全部 960,067 个问题进行全面评估对许多模型来说成本过高，因此采用了固定子集评估。

### 未来工作方向
- **扩展多模态支持**：将实验室结果、影像报告等纳入基准构建流程。
- **增强长程推理能力**：开发更先进的方法来处理跨越多个就诊的复杂时间序列推理。
- **拓宽验证范围**：结合更多样化的知识源（如专家评审、动态文献）以提高对罕见或新兴实践的覆盖率。
- **推动临床可靠 LLM 的发展**：利用 EHRBench 作为基础，指导和加速能够真正辅助临床医生的、安全可靠的 LLM 系统的研发。

</details>

---

### 13. [AdaptR1: Reinforcement Learning Based Adaptive Interleaved Thinking in Multi-hop Question Answering](https://arxiv.org/abs/2605.31062)

**Authors**: Yuxin Wang, Jiahao Lu, Qifeng Wu, Shicheng Fang, Chuanyuan Tan, Yining Zheng, Xuanjing Huang, Xipeng Qiu  
**Category**: cs.CL  
**Published**: 2026-06-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.31062v1  

#### Abstract
Large Language Models (LLMs) have achieved remarkable performance in complex reasoning tasks through Chain-of-Thought (CoT) prompting. However, this approach often leads to ``over-thinking,'' where models generate unnecessarily long reasoning traces for simple queries and incur avoidable inference c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AdaptR1: Reinforcement Learning Based Adaptive Interleaved Thinking in Multi-hop Question Answering

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Chain-of-Thought (CoT)** 的推理方法在多跳问答（multi-hop QA）任务中普遍存在“**over-thinking**”现象，即模型对简单查询也生成冗长的推理链，导致不必要的计算开销和延迟。现有自适应推理方法通常只在**查询级别**做出是否推理的全局决策，忽略了多步任务中不同阶段对显式推理需求的动态变化。

### 提出的新方法与创新
本文提出 **AdaptR1**，一种基于 **Reinforcement Learning (RL)** 的自适应交错思考框架，用于多跳问答中的细粒度推理控制。

#### 主要创新点：
- **Step-wise Adaptive Thinking**：首次引入在每个中间步骤动态决定是否进行显式推理（`Think`）或跳过（`No-Think`），实现“交错式”推理。
- **RL-only Training**：完全依赖强化学习训练，无需监督微调（SFT）作为冷启动，避免了构建高质量跳过标签轨迹的困难。
- **Quality-Gated Efficiency Reward**：设计了一种新的奖励函数，在保证答案质量（如 F1 ≥ 阈值 T）的前提下才给予效率奖励，防止为了节省 token 而牺牲准确性（reward hacking）。

### 相比现有方法的优势
| 维度 | 传统方法 | AdaptR1 |
|------|--------|---------|
| 决策粒度 | Query-level（单次决策） | Step-wise（每步动态决策） |
| 训练方式 | 多需 SFT 初始化 | 纯 RL，端到端学习 |
| 效率机制 | 固定长度惩罚或静态策略 | 动态预算分配，按需推理 |
| 过度推理缓解 | 有限 | 显著减少冗余推理 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在六个标准多跳 QA 数据集上进行实验：
- **2WikiMultiHopQA**
- **HotpotQA**
- **Musique**
- **NQ (Natural Questions)**
- **PopQA**
- **TriviaQA**

所有数据集统一采用 5,120 条训练样本 + 128 条测试样本的划分以确保公平比较。

### 实验设置
- **骨干模型**：`Qwen2.5-7B-Instruct`
- **训练范式**：基于 **Group Relative Policy Optimization (GRPO)** 的 RL 训练
- **最大轮数**：6 轮交互（reason → search → answer）
- **实现变体**：
  - `Search-AdaptR1`：基于 chunk-based retrieval 的版本
  - `Graph-AdaptR1`：基于 graph-based retrieval 的版本

### 评估指标
| 指标 | 含义 |
|------|------|
| **EM (Exact Match)** | 完全匹配准确率 |
| **F1 Score** | 答案 token 级别的 F1 分数 |
| **R-S (Retrieval Similarity)** | 检索内容与黄金上下文的语义相似度 |
| **Think Token Count** | 平均每轮使用的“思考 token”数量，衡量推理成本 |

### 基线方法对比
分为两类：

#### ✅ Training-Free 方法（Prompt Engineering）
- NaiveGeneration
- StandardRAG
- GraphRAG / LightRAG / PathRAG / HippoRAG2 / HyperGraphRAG

#### ✅ Training-Based 方法
- SFT（监督微调）
- R1 / Search-R1 / R1-Searcher / Graph-R1（均为 RL 基线）

> 所有 RL 方法共享相同 backbone 和训练配置，仅奖励函数不同，隔离出 AdaptR1 的增益。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 2）

#### 在 `Graph-R1` 设置下的表现：

| 模型 | Avg F1 | Avg Think Tokens | Reduction |
|------|--------|------------------|----------|
| Graph-R1 | 59.27 | 102.07 | — |
| **Graph-AdaptR1 (Ours)** | **60.74** (+1.47) | **30.92** | **↓69.71%** |

> ⬆️ 性能提升的同时，推理 token 减少近七成！

#### 最大收益出现在 HotpotQA：
- Think Tokens 从 **103.68 → 10.00**
- **降低 90.35%**
- F1 反而从 63.55 提升至 **64.39**

#### Search-R1 对比：
| 模型 | Avg F1 | Think Tokens | Reduction |
|------|--------|----------------|----------|
| Search-R1 | 43.80 | 111.16 | — |
| **Search-AdaptR1** | **50.68** (+6.88) | **76.68** | ↓31.02% |

> 不仅更高效，且性能大幅提升，说明 `No-Think` 策略有助于聚焦关键推理。

---

### 消融实验结果（Ablation Studies）

#### （1）是否需要 RL 训练？（Table 3）
| 状态 | Musique F1 | No-Think Rate |
|------|------------|---------------|
| 训练前（仅加 prompt） | 8.41 | 13.02% |
| RL 训练后 | **53.42** | **50.67%** |

✅ 表明 `no_think` 是通过 RL 学会的智能行为，而非 prompt 触发即可生效。

#### （2）No-Think 时间分布分析（Table 4）
- 模型倾向于在**早期步骤（Step 1–2）跳过推理**，保留给后期综合（Step 3–4）
- 支持 “**retrieve-then-reason**” 策略：先获取信息，再集中推理
- 设置 step-wise 权重 `λ=0.9`（偏好早期跳过）效果最佳

#### （3）奖励函数设计影响（Table 5–7）
| 变体 | Avg F1 | 结论 |
|------|-------|------|
| Absolute Reward (default) | 60.74 | ✅ 最佳 |
| Relative Reward | 58.25 | ❌ 易被 exploited，鼓励过早终止 |
| w/o Top Ceiling | 56.20 | ❌ 效率奖励主导，损害准确率 |
| Threshold T=0.6 | 60.74 | ✅ 最优平衡点 |
| w=0.2 | 60.74 | ✅ 效率奖励应为辅助信号 |

> ✅ 设计要点：**带门控的质量约束 + 绝对奖励 + 上限封顶 + 合理超参**

---

## 4. 关键结论和发现

### 主要发现
1. **Over-thinking 并非均匀分布**：主要集中在多跳推理的**初始规划阶段**，而非最终合成阶段。
2. **Step-wise 自适应优于全局决策**：允许每步独立判断是否推理，显著提升效率而不损性能。
3. **RL 可学会何时“不思考”**：通过 quality-gated reward，模型能自主识别可跳过的冗余推理步骤。
4. **效率与性能可兼得**：在 Graph-R1 下平均减少 **69.71% think tokens**，同时 F1 提升 1.47。

### 方法的局限性
- **对超参数敏感**：阈值 `T`、奖励系数 `w`、step-weight `λ` 需谨慎调节，否则易出现过度剪枝。
- **潜在训练不稳定**：RL 优化过程中可能出现后期性能下降（见 Figure 3），因模型过度追求简洁。
- **适用范围限制**：目前验证于中短程多跳任务，尚未扩展到需长期持续推理的 **DeepResearch** 类任务。

### 未来工作方向
- 将 AdaptR1 扩展至更复杂的 agentic workflows 和 long-horizon planning 场景。
- 探索更鲁棒的 reward shaping 方法，进一步降低对人工设定超参的依赖。
- 结合 interpretability 技术，可视化并解释模型为何选择 `no_think`，增强可信度。

---

> 💡 **一句话总结**：  
> **AdaptR1** 通过 **RL-driven step-wise No-Think 决策**，实现了多跳问答中“该想时想，不该想时不耗”的智能推理调度，在保持甚至提升性能的同时，将推理成本压缩近七成，是迈向高效、绿色 AI 的重要一步。

</details>

---

### 14. [DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers](https://arxiv.org/abs/2605.30456)

**Authors**: Shraman Pal, Can Li  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.30456v1  

#### Abstract
Many learning tasks in science and engineering are characterized by sparse datasets, which limits the effectiveness of purely data-driven approaches. At the same time, these problems are often accompanied by rich domain knowledge derived from physical laws, operational requirements, and expert heuri...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers**

---

## **1. 主要贡献和创新点**

### **解决的问题**
许多科学与工程领域的学习任务面临**小样本数据**和**分布偏移**的挑战，纯数据驱动的深度学习模型容易违反已知的物理规律、安全约束或专家规则。尽管领域知识常以逻辑命题和线性不等式形式存在，但现有神经符号学习（neuro-symbolic learning）方法在处理这些规则时存在以下问题：
- **软约束（Soft penalties）**：无法保证推理时完全满足规则。
- **硬约束架构设计**：如 MultiplexNet，仅支持全局固定规则，无法处理输入依赖（input-dependent）的动态规则。
- **非可微后处理**：在推理阶段通过 ILP 解码强制满足约束，但无法端到端训练。

此外，将混合整数线性规划（MILP）类的逻辑/离散约束嵌入神经网络中，因非凸性和非光滑性导致优化层难以微分。

### **提出的新方法**
本文提出 **DisjunctiveNet**，一个统一的端到端框架，用于在神经网络中**强制执行硬性的、输入依赖的混合整数线性约束**。其核心思想是：
- 将逻辑规则表示为**析取约束（disjunctive constraints）**，即多个线性不等式组的并集（union of polyhedra）。
- 应用**层次化凸松弛（hierarchical convex relaxation）**，特别是基于**基本步层级（basic step hierarchy）** 的方法，将非凸可行域转化为**凸包（convex hull）**。
- 构造一个**可微的凸优化层（differentiable optimization layer）**，该层在前向传播中求解投影问题，在反向传播中通过隐式微分（implicit differentiation）传递梯度。

具体技术路径：
- **DNF 松弛（Disjunctive Normal Form）**：对所有活动规则的析取项进行交叉枚举，形成“交的并”结构，并对其施加凸包松弛。该松弛是**最紧致的凸松弛**，能保证最终解满足原始逻辑约束。
- **CNF 松弛（Conjunctive Normal Form）**：对每条规则独立凸化后再取交集，计算量小但松弛较松。
- 支持**顺序凸化（sequential convexification）**，在 CNF 和 DNF 之间进行权衡。

### **相比现有方法的优势**
| 维度 | DisjunctiveNet | 现有方法（如 Soft Penalty, MultiplexNet, SATNet） |
|------|----------------|---------------------------------------------|
| **约束类型** | 支持逻辑 + 连续变量的混合约束（Mixed） | 多数仅支持连续或特定逻辑形式 |
| **约束强度** | **硬约束（Hard）**，理论保证精确满足 | 软约束，无可行性保证 |
| **输入依赖性** | ✅ 支持输入依赖规则激活 | ❌ 多数为全局固定规则 |
| **可微性** | ✅ 端到端可微，支持训练时约束集成 | ❌ 后处理不可微或近似梯度 |
| **表达能力** | 等价于 MILP 和 QF-LRA，表达能力强 | 表达受限 |

> ✅ **关键创新**：首次实现了**输入依赖的混合整数线性约束**在神经网络中的**端到端硬约束满足**，且通过凸包松弛保证了解的可行性。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **合成冷却控制任务（Synthetic Cooling-Control Problem）**
   - 模拟工业冷却系统，预测风扇速度（fan）、制冷机水平（chiller）、泵功率（pump）。
   - 输入：环境温度 $T_a$、湿度 $H$、负载 $w$、电价 $\tau$、维护标志等。
   - 规则：共 7 条输入依赖的析取规则，例如：
     - 若高温 → 制冷机开或风扇高速
     - 若需求响应事件 → 功耗低于阈值 或 双设备降载
   - 数据划分：In-Distribution (IID) 和 Out-of-Distribution (OOD) 测试集，后者模拟极端工况。

2. **单细胞 RNA 测序分类（scRNA-seq Classification, PBMC3k）**
   - 任务：根据基因表达预测细胞类型（8 类）。
   - 领域知识：**标记基因规则（marker-gene rules）**，形式为：
     $$
     G(x) \geq T_r \Rightarrow \bigvee_{c \in C_r} y_c \geq p_r
     $$
     即当某基因表达高时，必须至少有一个相关细胞类型的预测概率足够高。
   - 数据稀疏、高维，适合验证小样本下规则引导的有效性。

### **实验设置与评估指标**
#### **评估指标**
| 指标 | 说明 |
|------|------|
| **CSAT (Constraint Satisfaction)** | 满足所有激活规则的样本比例（越高越好） |
| **MSE / Macro-F1** | 回归任务用 MSE，分类任务用 Macro-F1（越高越好） |
| **Inf Time / LP Size** | 推理时间与 LP 变量/约束数量（衡量计算开销） |

#### **基线方法对比**
| 方法 | 说明 |
|------|------|
| **Base** | 无约束的普通 NN |
| **Penalty (Pen)** | 在损失函数中加入规则软惩罚项 |
| **Finetuned Penalty (Fine-Pen)** | 先训 Base，再用软惩罚微调 |
| **Rules Only** | 仅靠规则随机选择输出（验证规则本身质量） |
| **CNF / DNF** | 本文提出的不同松弛策略的 DisjunctiveNet |

所有方法使用相同 NN 骨干（MLP），投影类方法从预训练 Base 模型微调，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **合成冷却任务（Fig. 3–4, Table 1）**
| 方法 | IID CSAT | OOD CSAT | IID MSE | OOD MSE |
|------|---------|---------|--------|--------|
| Base | ~0.80 | ~0.75 | ~0.020 | ~0.025 |
| Pen | ~0.80 | ~0.75 | ~0.020 | ~0.025 |
| CNF | ~0.95 | ~0.90 | ~0.010 | ~0.015 |
| **DNF** | **1.00** | **1.00** | **0.004** | **0.005** |

- **DNF 实现 100% 规则满足率**，显著优于所有基线。
- 在 OOD 上，DNF 的 MSE 比 Base 低 **60%+**，显示其强泛化能力。
- **顺序凸化实验**：随着更多规则被纳入 DNF 展开，CSAT 快速逼近 DNF 性能，表明少量关键规则即可捕获大部分逻辑结构。

#### **scRNA-seq 分类任务（Fig. 5, Tables 8–12）**
| 训练样本数 | 方法 | Macro-F1 | CSAT |
|-----------|------|----------|------|
| 12 | Base | 0.296 | 0.242 |
| 12 | CNF | 0.407 | 0.804 |
| 12 | **DNF** | **0.402** | **0.951** |
| 469 | Base | 0.709 | 0.742 |
| 469 | DNF | 0.666 | **0.882** |

- **小样本下优势明显**：当训练数据极少时（n=12），DNF 的 F1 比 Base 高 **35%+**，CSAT 从 24% 提升至 95%。
- **大样本下权衡显现**：当数据充足时，Base 准确率更高，但 CSAT 仍远低于 DNF，表明**规则满足与预测精度存在权衡**。
- **DNF 始终保持最高 CSAT**，接近 100%，而软约束方法即使微调也无法稳定满足规则。

#### **计算开销（Table 1, E.3）**
| 方法 | 平均变量数 | 平均约束数 | 推理时间 |
|------|------------|------------|----------|
| CNF | 37.5 | 137.3 | 25.03 ms |
| DNF | 44.4 | 160.5 | 28.62 ms |

- DNF 开销略高，但仍**在毫秒级可接受范围**。
- 实际规模小于理论指数爆炸，因多数析取组合为空或不可行。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **DisjunctiveNet 能实现 100% 的硬规则满足**，尤其在 DNF 设置下，通过凸包松弛和极点解机制，保证输出严格符合输入依赖的逻辑规则。
2. ✅ **在小样本和 OOD 场景下显著提升性能**：规则作为强归纳偏置（inductive bias），弥补数据不足，提升泛化能力。
3. 🔁 **存在准确率与规则满足的权衡**：当数据充足时，纯数据驱动模型可能更准，但会牺牲规则一致性；本方法适用于**规则必须满足**的关键场景。
4. ⚖️ **CNF 与 DNF 的实用权衡**：CNF 计算高效，DNF 精度最优，可通过**顺序凸化**在两者间取得平衡。

### **局限性**
1. **计算复杂度**：DNF 松弛在活动规则多时可能指数增长，限制其在超大规模规则系统中的应用。
2. **依赖极点解**：精确满足需求解器返回 LP 的极点解（如 simplex），若使用内点法且无 crossover，可能不满足。
3. **冲突规则处理**：当前框架假设规则一致，若出现矛盾前提（contradictory rules），需外部机制跳过投影。
4. **仅支持线性析取**：尚未扩展到非线性凸约束或更复杂的逻辑结构。

### **未来工作方向**
1. **扩展至非线性约束**：结合非线性析取规划（nonlinear disjunctive programming），支持更丰富的领域知识。
2. **自适应顺序凸化**：研究启发式或学习策略，决定哪些规则优先进行 DNF 展开。
3. **支持欧式投影（Euclidean norm）**：当前使用 $l_1$ 投影以保证 LP 形式，未来可探索 $l_2$ 投影与 SOCP 结合。
4. **鲁棒性增强**：引入对噪声规则或不确定前提的容忍机制。
5. **更大规模基准测试**：在真实工业控制系统或生物网络中验证可扩展性。

---

> **总结**：DisjunctiveNet 提供了一个**理论上严谨、实践中有效**的框架，首次实现了**输入依赖混合逻辑约束**在神经网络中的**端到端硬满足**，为科学机器学习（scientific ML）和安全关键系统提供了强有力的工具。

</details>

---

### 15. [AbstainGNN: Teaching Graph Neural Networks to Abstain for Graph Classification](https://arxiv.org/abs/2605.30786)

**Authors**: Xixun Lin, Zhiheng Zhou, Zhengyin Zhang, Yancheng Chen, Shuai Zhang, Ge Zhang, Shichao Zhu, Lixin Zou, Chuan Zhou, Peng Zhang, Shirui Pan, Yanan Cao  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.30786v1  

#### Abstract
Graph classification is a core task in graph data mining with widespread real-world applications. Recent advances in graph neural networks (GNNs) have led to substantial performance improvements for graph classification. However, existing GNNs are typically forced to make predictions even under high...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AbstainGNN: Teaching Graph Neural Networks to Abstain for Graph Classification

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Graph Neural Networks (GNNs)** 在图分类任务中通常被强制对所有输入样本做出预测，即使模型对某些样本的预测置信度很低。这种“无条件预测”机制在高风险场景（如药物毒性预测、金融风控）中可能导致严重错误，限制了 GNN 在安全敏感领域的应用。

本文首次系统地研究了**图分类中的学习拒绝机制（Learning with Abstention for Graph Classification）**，即允许 GNN 在不确定时主动选择“不预测”，从而提升决策系统的可靠性。

---

### 提出了什么新方法或新思路
作者提出了 **AbstainGNN** ——一个理论驱动的、支持拒绝预测的图神经网络框架，其核心思想包括：

- **双函数架构设计**：
  - 显式建模两个独立函数：
    - **预测函数 $f$**：负责常规的图分类。
    - **拒绝函数 $g$**：输出一个置信度分数，决定是否对该图进行预测。
  - 当 $g(A,X) < \epsilon$ 时，模型选择 **abstain（拒绝预测）**。

- **理论驱动的学习目标**：
  - 从 **PAC-Bayesian generalization bounds** 视角出发，理论分析了分类误差与拒绝成本之间的权衡。
  - 推导出统一的学习目标，证明**最小化图表示的类内方差（intra-class variance of graph-level representations）是实现有效拒绝的关键**。

- **高效的两阶段训练策略**：
  1. **Predictive Function Warm-start**：先用标准交叉熵损失预训练 $f$，使其进入梯度较小区域，稳定后续优化。
  2. **Abstention Function Calibration**：基于已学习的 $f$，训练 $g$ 来校准置信度，确保其与实际预测准确性对齐。

- **全局类簇调整（Global Cluster Adjustment）**：
  - 引入滑动平均机制维护全局类中心 $\mu_y$，缓解小批量更新带来的估计偏差，提高正则项有效性。

---

### 相比现有方法的优势
| 维度 | AbstainGNN | 现有方法（如 SR, MC-Dropout, Deep Gamblers） |
|------|-----------|---------------------------------------------|
| **领域适配性** | 显式建模图结构信息用于拒绝决策 | 多为 CV 领域设计，直接迁移至图数据效果有限 |
| **理论基础** | 有 PAC-Bayes 理论支撑，揭示类内方差的重要性 | 多为启发式设计，缺乏理论解释 |
| **训练稳定性** | 两阶段训练 + 全局聚类调整，提升优化过程鲁棒性 | 单阶段端到端训练易受噪声影响 |
| **性能表现** | 在多个数据集上显著优于基线 | 性能波动大，尤其在高覆盖率下退化明显 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在五个广泛使用的图分类基准数据集上进行实验，涵盖化学图和社会图两类：

| Dataset | 图数量 | 类别数 | 平均节点数 | 平均边数 | 类型 |
|--------|-------|--------|------------|----------|------|
| MUTAG | 188 | 2 | 17.9 | 19.8 | 化学图 |
| PROTEINS | 1,113 | 2 | 39.1 | 72.8 | 化学图 |
| NCI1 | 4,110 | 2 | 29.9 | 32.3 | 化学图 |
| IMDB-BINARY | 1,000 | 2 | 19.8 | 96.5 | 社会图 |
| COLLAB | 5,000 | 3 | 74.5 | 2457.8 | 社会图 |

> 数据统计见原文 Table 2。

---

### 实验设置和评估指标

#### 模型架构
- **预测函数 $f$**：3 层 GCNConv + Sum Pooling + 分类头。
- **拒绝函数 $g$**：1 层 GCNConv + Sum Pooling + 单输出 sigmoid（表示置信度）。
- 所有方法共享相同 backbone 以保证公平比较。

#### 超参数调优
通过网格搜索确定：
- 正则系数 $\lambda_r \in \{0.1, 0.3, ..., 0.9\}$
- 预热轮次 $T_{ws} \in \{50, 100, ..., 200\}$
- Batch size $\in \{16, 32, ..., 256\}$

#### 评估协议
遵循标准的 **Selective Classification Evaluation Protocol**：
- **Coverage**：模型做出预测的样本比例（人为设定阈值筛选高置信样本）。
- **Risk**：在被接受的样本上的分类错误率（Error Rate）。
- 测试时按 $g(A,X)$ 对测试样本排序，在不同 **Coverage levels**（10%, 30%, ..., 95%）下报告 Risk。

#### 主要指标
- 不同 coverage 下的风险（Risk ↓）
- 平均相对风险降低（Average Relative Risk Reduction）
- 消融实验与变体对比
- 运行时间对比（Per-epoch training time）

---

### 基线方法对比
共对比 9 种主流 abstention 方法，分为以下几类：

| 类别 | 方法 | 简介 |
|------|------|------|
| **置信度阈值法** | SR | 使用 Softmax 最大值作为置信度 |
| **不确定性估计** | MC-Dropout | 测试时多次采样估计不确定性 |
| **引入拒绝类** | Deep Gamblers, SAT | 将“拒绝”视为额外类别进行训练 |
| **覆盖控制** | CCL-SC | 基于对比学习的覆盖率控制方法 |
| **图领域相关** | NCwR | 最早将拒绝机制引入节点分类的工作 |
| | GraphPPD | 基于后验预测分布建模不确定性 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1 & 3）

| 方法 | MUTAG (90%) | PROTEINS (90%) | NCI1 (90%) | IMDB-BINARY (90%) | COLLAB (90%) |
|------|-------------|----------------|------------|--------------------|--------------|
| SR | 23.5±1.1 | 17.1±4.0 | 29.5±1.3 | 25.0±3.2 | 17.7±0.5 |
| MC-Dropout | 22.1±1.7 | 17.6±4.6 | 29.9±1.6 | 24.3±1.5 | 19.1±0.8 |
| Deep Gamblers | 25.5±2.1 | 17.6±4.6 | 31.4±2.0 | 24.2±2.8 | 17.8±0.7 |
| CCL-SC | 23.4±0.8 | 17.1±3.4 | 29.6±1.3 | 25.1±2.9 | 17.8±0.7 |
| GraphPPD | 24.5±1.1 | 17.6±3.6 | 29.0±1.1 | 24.7±3.5 | 18.2±0.6 |
| **AbstainGNN** | **21.7±1.0** | **13.3±3.2** | **27.6±1.4** | **22.9±2.7** | **17.4±0.5** |

> ✅ **AbstainGNN 在所有数据集的所有 coverage 水平下均取得最优性能**

---

### 与基线方法的对比结果
- **平均相对风险降低**：
  - MUTAG: **16.8%**
  - PROTEINS: **9.8%**
  - NCI1: **4.8%**
  - IMDB-BINARY: **15.8%**
  - COLLAB: **5.9%**
- 在 **高覆盖率（>80%）** 场景下优势尤为明显，说明 AbstainGNN 能更准确地区分“难样本”并合理拒绝。
- 在低 coverage（如10%）时多数方法都能选出高置信样本，差异不大；但在实际应用中，更高的 coverage 更具实用价值。

---

### 消融实验结果（Figure 3 & Table 4）

#### Ablation Study（移除组件的影响）
| 变体 | 描述 | 影响 |
|------|------|------|
| w/o Warm-start | 移除第一阶段预热训练 | 导致优化不稳定，尤其在中高 coverage 下性能下降明显 |
| w/o Regularization ($L_{icv}$) | 移除类内方差正则项 | 表示空间聚集性差，置信度校准失效，风险上升 |

> 结果表明：**warm-start 和 $L_{icv}$ 正则项都至关重要**，共同保障了高质量的表示学习与置信度估计。

#### 模型变体对比（Table 4）
| 变体 | 架构变化 | 性能 |
|------|---------|------|
| AbstainGNN-GAT | $g$ 使用 GAT 替代 GCN | 性能接近但计算开销更高 |
| AbstainGNN-MLP | $g$ 使用 MLP（忽略图结构） | 显著劣于原模型 → 说明图结构对拒绝决策重要 |
| AbstainGNN-BC | 使用 batch-wise 类中心（无全局调整） | 性能下降 → 说明全局聚类更稳定可靠 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **图分类中的拒绝机制必须显式利用图结构信息**，简单迁移图像领域的 heuristics 效果不佳。
2. ✅ **类内方差最小化是实现高效拒绝的核心机制**，该结论由 PAC-Bayes 理论严格推导得出，并得到实验验证（Figure 4 显示训练后期类内方差显著下降）。
3. ✅ **两阶段训练策略有效提升了优化稳定性与最终性能**，warm-start 有助于收敛到更好的局部最优。
4. ✅ **全局类簇调整显著优于批级估计**，减少了因 mini-batch 抽样带来的偏差。

---

### 方法的局限性
1. **依赖均匀类别假设**：理论分析中假设各类先验概率相等，可能在极端不平衡数据上受限。
2. **阈值需手动设定**：虽然 coverage 可控，但具体阈值 $\epsilon$ 仍需根据应用场景调节。
3. **仅适用于 transductive 设置**：当前实现未考虑完全归纳式（inductive）场景下的泛化能力。
4. **拒绝后的处理未定义**：模型只负责“拒绝”，后续如何处理（如人工审核、延迟判断）不在本工作范围内。

---

### 未来工作方向
1. **扩展到其他图任务**：如节点分类、链接预测、图生成等任务中的拒绝机制。
2. **动态阈值调节机制**：结合任务代价自动调整拒绝阈值，实现 cost-aware abstention。
3. **更优的 GNN 架构设计**：探索更适合拒绝函数 $g$ 的轻量级或注意力机制。
4. **与 Out-of-Distribution Detection 结合**：联合建模分布外检测与拒绝决策，增强模型整体可信度。
5. **在线/流式场景下的拒绝机制**：适应动态图或持续学习环境。

---

> 🔗 **代码开源地址**：  
> https://github.com/ZZY565/AbstainGNN  
> https://doi.org/10.5281/zenodo.20422037

</details>

---

### 16. [Bandwidth Allocation with Device Partitioning for Federated Learning over Industrial IoT networks](https://arxiv.org/abs/2605.30892)

**Authors**: Kangmin Kim, Jaeyoung Song  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.30892v1  

#### Abstract
We consider a federated learning (FL) system in which Industrial Internet-of-Things (IIoT) devices collaboratively train a global model over wireless channels without sharing local data. In such systems, communication time is a primary bottleneck that constrains overall training efficiency. Unlike c...

---

### 17. [Towards Efficient LLMs Annealing with Principled Sample Selection](https://arxiv.org/abs/2605.31175)

**Authors**: Yuanjian Xu, Jianing Hao, Wanbo Zhang, Zhong Li, Guang Zhang  
**Category**: cs.CL  
**Published**: 2026-06-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.31175v1  

#### Abstract
The annealing phase is a pivotal convergence stage in LLM pre-training that ultimately determines final model quality. However, effectively selecting training data during this phase remains a key challenge. Current strategies rely on empirical heuristics, such as domain filtering or context extensio...

---

### 18. [Algorithmic Recourse of In-Context Learning for Tabular Data](https://arxiv.org/abs/2605.31272)

**Authors**: Wenshuo Dong, Jiaming Zhang, Shaopneg Fu, Hongbin Lin, Di Wang, Lijie Hu  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.31272v1  

#### Abstract
As predictive models are increasingly deployed in high-stakes settings such as credit approval, there is a growing need for post-hoc methods that provide recourse to affected individuals. Many such models operate on tabular data, where features correspond to real-world attributes. Recently, in-conte...

---

### 19. [Scalable Inference-Time Annealing with Surrogate Likelihood Estimators](https://arxiv.org/abs/2605.31498)

**Authors**: Daniel Pe\~naherrera, Rishal Aggarwal, David Ryan Koes  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.31498v1  

#### Abstract
A long standing challenge in computational chemistry and biophysics is efficiently sampling the Boltzmann distribution of molecules. Advances in generative modeling have been proposed to address the limitations of conventional sampling techniques by eliminating the computational cost of simulation. ...

---

### 20. [Combinatorial Synthesis: Scaling Code RLVR via Atomic Decomposition and Recombination](https://arxiv.org/abs/2605.31058)

**Authors**: Jiasheng Zheng, Boxi Cao, Boxi Yu, Yuzhong Zhang, Jialun Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  
**Category**: cs.CL  
**Published**: 2026-06-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.31058v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as the cornerstone for shaping the remarkable coding abilities of Large Language Models (LLMs). However, the scalability of RLVR is severely constrained by the scarcity of sufficiently challenging verifiable code tasks that t...

---

### 21. [Spectral Anatomy of Quantum Gaussian Process Kernels](https://arxiv.org/abs/2605.30952)

**Authors**: Jian Xu, Chao Li, Guang Lin, Yuning Qiu, Delu Zeng, John Paisley, Qibin Zhao  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.30952v1  

#### Abstract
Two recent results have reshaped quantum Gaussian processes (QGPs). On the one hand, \citet{lowe2025assessing} rule out the exponential speedups claimed by HHL-based QGP regression in the typical, well-conditioned regime; on the other, an independent line of work shows that highly expressive quantum...

---

### 22. [Revisiting Zeroth-Order Hessian Approximation: A Single-Step Policy Optimization Lens](https://arxiv.org/abs/2605.30960)

**Authors**: Junbin Qiu, Zhaowei Hong, Renzhe Xu, Yao Shu  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.30960v1  

#### Abstract
Accurate Zeroth-Order (ZO) Hessian estimation is a cornerstone of derivative-free methods, essential for tasks such as bilevel optimization, Bayesian inference, and uncertainty quantification. However, obtaining a complete suite of low-variance estimators for the Hessian and its inverse in high-dime...

---

### 23. [Best-Arm Identification-Based Trust Region Selection for Bayesian Optimization on Multimodal Functions](https://arxiv.org/abs/2605.31050)

**Authors**: Nobuo Namura, Sho Takemori  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.31050v1  

#### Abstract
Gaussian process-based Bayesian optimization (BO) is a popular approach for expensive black-box optimization, but its performance often degrades on complex multimodal or high-dimensional problems. Trust region-based BO mitigates this issue by focusing on local regions, and recent studies suggest tha...

---

### 24. [Graphical einops: bridging tensor networks and computation graphs](https://arxiv.org/abs/2605.31485)

**Authors**: Vincent Wang-Ma\'scianica, Nikhil Khatri  
**Category**: cs.LG  
**Published**: 2026-06-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.31485v1  

#### Abstract
Architecture diagrams are ubiquitous in deep learning, but they are usually only representational: the tensor-program identities they suggest are still proved by prose and tensor-axis manipulation. We introduce a formal graphical calculus for the structural fragment of tensor programming underlying ...

---

### 25. [Structure-Induced Information for Rerooting Levin Tree Search](https://arxiv.org/abs/2605.30664)

**Authors**: Jake Tuero, Michael Buro, Laurent Orseau, Levi H. S. Lelis  
**Category**: cs.AI  
**Published**: 2026-06-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.30664v1  

#### Abstract
Subgoal-based policy tree search, which uses a policy to guide search, is effective for complex single-agent deterministic problems but often relies on explicit subgoal generation that can incur substantial overhead and hinders scalability. In this paper, we overcome these limitations by using a lea...

---

### 26. [Planner-Centric Reinforcement Learning for Deep Research with Structure-Aware Reward](https://arxiv.org/abs/2605.30824)

**Authors**: Mustafa Anis Hussain, Xinle Wu, Yao Lu  
**Category**: cs.AI  
**Published**: 2026-06-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.30824v1  

#### Abstract
Deep research tasks require LLMs to plan what to investigate, retrieve evidence, and synthesize long-form answers across multiple branches of inquiry. Existing training paradigms either rely on short-form verifiable QA as a proxy or optimize monolithic long trajectories, which makes planning and exe...

---

### 27. [COMPASS: Cognitive MCTS-Guided Process Alignment for Safe Search Agents](https://arxiv.org/abs/2605.30838)

**Authors**: Wenkai Shen, Pengyang Zhou, Jiahe Xu, Jiaming Qian, Haozhe He, Zhihao Huang, Chaochao Chen, Xiaolin Zheng  
**Category**: cs.AI  
**Published**: 2026-06-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.30838v1  

#### Abstract
LLM-powered search agents enable multi-step reasoning and tool use. However, these capabilities introduce retrieval-induced safety degradation, as harmful intents may decompose into seemingly innocuous sub-queries that lead to unsafe outcomes. Existing alignment methods struggle to capture sparse sa...

---

### 28. [Learning to Adapt: Self-Improving Web Agent via Cognitive-Aware Exploration](https://arxiv.org/abs/2605.31365)

**Authors**: Weile Chen, Bingchen Miao, Qifan Yu, Wendong Bu, Guoming Wang, Wenqiao Zhang, Shengyu Zhang, Juncheng Li, Siliang Tang  
**Category**: cs.AI  
**Published**: 2026-06-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.31365v1  

#### Abstract
Recent advances in Multimodal Large Language Models (MLLMs) have led to promising progress in web agents. However, existing web agents often rely on handcrafted execution pipelines or expensive expert trajectories, limiting their adaptability to complex, dynamic environments. To address these challe...

---

### 29. [Knowledge Graph-Enhanced Zero-Shot Topic Classification: A Multi-Strategy Comparative Study](https://arxiv.org/abs/2605.30465)

**Authors**: Shahana Akter, Yatharth Vohra, Ankita Shukla, Souvika Sarkar  
**Category**: cs.CL  
**Published**: 2026-06-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.30465v1  

#### Abstract
Multi-label topic classification without labeled training data is a challenging task, specially when documents contain complex relational information. We present a zero-shot multi-label topic classification framework and systematically investigate how per-article knowledge graph augmentation affects...

---

### 30. [AI for Monitoring and Classifying Data Used in Research Literature](https://arxiv.org/abs/2605.30582)

**Authors**: Rafael Macalaba, Aivin V. Solatorio  
**Category**: cs.CL  
**Published**: 2026-06-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.30582v1  

#### Abstract
While platforms like Google Scholar and Semantic Scholar track citations for academic papers, no comparable infrastructure exists for monitoring dataset usage in research literature, leaving the landscape of data use largely opaque. Addressing this gap is critical for transparency, reproducibility, ...

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

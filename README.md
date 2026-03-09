# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-09 06:45:32 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [EvoESAP: Non-Uniform Expert Pruning for Sparse MoE](https://arxiv.org/abs/2603.06003)

**Authors**: Zongfang Liu, Shengkun Tang, Boyang Sun, Zhiqiang Shen, Xin Yuan  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.06003v1  

#### Abstract
Sparse Mixture-of-Experts (SMoE) language models achieve strong capability at low per-token compute, yet deployment remains memory- and throughput-bound because the full expert pool must be stored and served. Post-training expert pruning reduces this cost, but most methods focus on which experts to ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# EvoESAP: Non-Uniform Expert Pruning for Sparse MoE — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **Sparse Mixture-of-Experts (SMoE)** 模型虽然通过条件计算降低了每 token 的计算成本，但在部署时仍需存储全部专家，导致内存占用高、吞吐受限。
- 现有的 **expert pruning** 方法大多采用**均匀层间稀疏度分配**（uniform layer-wise sparsity），即每一层移除相同比例的专家，忽略了不同层对模型能力贡献的差异。
- 同时，如何高效评估剪枝后模型是否保留原始生成能力（尤其是开放生成任务）缺乏有效手段。

### 🚀 提出的新方法与思路
1. **解耦剪枝决策**：
   - 将 expert pruning 分为两个独立步骤：
     - **层内排序（within-layer selection）**：基于重要性准则（如 Frequency, EAN, REAP 等）确定每层中专家的移除顺序。
     - **层间预算分配（across-layer allocation）**：在固定全局剪枝预算下，搜索最优的非均匀稀疏度分布（即各层保留专家数不同）。

2. **提出 ESAP（Expected Speculative Acceptance Proxy）**：
   - 一种受 speculative decoding 启发的、teacher-forced 的代理指标，用于衡量剪枝模型与原模型在 next-token 分布上的相似性。
   - 公式上等价于 `1 - TV(p, q)`（总变差距离的补），具有**有界性、稳定性、低计算开销**的优点，避免了昂贵的自回归采样。

3. **提出 EvoESAP 框架**：
   - 基于 ESAP 作为 fitness 函数，构建一个**进化搜索框架**，在整数约束下优化非均匀的层间稀疏度分配。
   - 使用 **level-switch mutation** 操作保持全局预算不变，实现高效的组合优化。

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | EvoESAP |
|------|--------|---------|
| 层间稀疏度 | 默认 uniform | 显式优化 non-uniform 分配 |
| 评估方式 | 多依赖 MCQ 或耗时的 speculative decoding | 提出轻量级 ESAP，速度快 ~18× |
| 插件兼容性 | 通常绑定特定剪枝准则 | 支持多种 within-layer 排序标准（Frequency, EAN, SEER, REAP） |
| 性能提升重点 | 多关注 multiple-choice accuracy | 显著提升 open-ended generation 能力 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### （1）校准与搜索数据集（用于计算专家重要性和 ESAP）
- `evol-codealpaca-v1`：用于计算 expert-importance scores（如 REAP, EAN 等）。
- `tulu-3-sft-personas-math`：用于 evolutionary search 中的 ESAP fitness 评估（64 个样本）。

#### （2）主评估数据集
| 类别 | 数据集 |
|------|-------|
| **Multiple Choice (MC)** | ARC-C/E, BoolQ, HellaSwag, MMLU, OpenBookQA, RTE, WinoGrande |
| **Open-ended Generation** | |
| &nbsp;&nbsp;• 编码 | EvalPlus（HumanEval+/MBPP+）、LiveCodeBench |
| &nbsp;&nbsp;• 数学 | GSM8K、MATH-500 |
| &nbsp;&nbsp;• 创意写作 | WildBench（由 `gpt-oss-120b` 打分） |

> 所有任务均为 zero-shot setting。

### ⚙️ 实验设置
- **模型规模**：覆盖 7B–30B 参数的 SMoE LLMs：
  - OLMoE-1B-7B-0125-Instruct (7B)
  - ERNIE-4.5-21B-A3B-PT (21B)
  - Qwen3-30B-A3B-Instruct-2507 (30B)
- **剪枝比例**：25% 和 50% 全局 sparsity（即总共移除 25%/50% 的专家）。
- **within-layer 排序准则**：Frequency, SEER, EAN, REAP。
- **搜索参数**：
  - 种群大小 P=32，精英数量 m=4
  - 最大迁移步长 Δ_max=4，最大复合操作 T_max=3
  - 进化代数：OLMoE 50 代，ERNIE 20 代，Qwen3 10 代

### 📊 评估指标
| 指标类型 | 指标名称 | 说明 |
|--------|--------|------|
| 开放生成 | Code Avg, Math Avg, WildBench | 子任务平均得分 |
| 多选问答 | MC Avg | 多个 MC 任务的平均准确率 |
| 搜索效率 | ESAP vs SPEC-DEC 时间 | 验证代理指标的有效性 |
| 消融分析 | 不同 fitness 函数 / 样本量的影响 | 验证鲁棒性 |

### 🆚 基线方法对比
- **Uniform Pruning (UNI)**：相同 within-layer 排序下，每层按相同比例剪枝。
- **True Speculative Decoding (SPEC-DEC)**：直接运行 speculative decoding 测 acceptance rate，作为 upper bound 参考（但极慢）。
- 其他剪枝准则本身不涉及非均匀分配，因此 EvoESAP 是在其基础上的增强模块。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & 2）

| 模型 | Sparsity | 方法 | Code Avg ↑ | Math Avg ↑ | MC Avg ↔ |
|------|----------|------|------------|-------------|-----------|
| OLMoE | 25% | REAP + UNI | 0.314 | 0.398 | 0.579 |
|       |        | **REAP + ESAP (Ours)** | **0.344 (+2.9%)** | **0.426 (+2.8%)** | **0.581 (+0.2%)** |
| ERNIE | 50% | Frequency + UNI | 0.647 | 0.397 | 0.565 |
|       |        | **Frequency + ESAP (Ours)** | **0.647 → 0.647** | **0.397 → 0.549 (+15.2%)** | **0.565 → 0.557 (+0.8%)** |
|       |        | &nbsp;&nbsp;&nbsp;&nbsp;其中 MATH-500 单项提升 | — | **+19.6%** | — |
| Qwen3 | 50% | Frequency + UNI | 0.700 | 0.374 | 0.455 |
|       |        | **Frequency + ESAP (Ours)** | **0.781 (+8.1%)** | **0.385 (+11%)** | **0.510 (+5.5%)** |

> 💡 注：提升集中在 open-ended generation，MC 基本持平甚至略有波动。

### 🔁 与基线方法对比结果
- 在 **所有模型、所有剪枝准则、两种稀疏度水平下**，EvoESAP 找到的 non-uniform allocation 均优于 uniform baseline。
- 提升幅度随 sparsity 增加而增大，在 **50% 稀疏度下最为显著**。
- 对原本较弱的剪枝准则（如 Frequency）增益更大；对已较强的准则（如 REAP）也可能因过度调整而轻微下降（见 Qwen3 @25%）。

### 🔍 消融实验结果（Table 5）

#### （1）Fitness 函数对比（在 OLMoE 上，REAP 排序）
| Fitness Function | Code Avg | Math Avg | MC Avg |
|------------------|----------|----------|--------|
| KL Divergence | 0.331 | 0.405 | 0.582 |
| NLL | 0.334 | 0.424 | 0.576 |
| SAP (Monte Carlo) | 0.339 | 0.420 | 0.584 |
| **ESAP (Ours)** | **0.344** | **0.426** | **0.581** |

✅ 结论：**ESAP 在生成任务上表现最佳且稳定**。

#### （2）搜索样本量敏感性（64 样本为默认）
| 样本数 | Code Avg | Math Avg | MC Avg |
|--------|----------|----------|--------|
| 8 | 0.320 | 0.427 | 0.576 |
| 32 | 0.339 | **0.433** | 0.582 |
| **64** | **0.344** | 0.426 | 0.581 |
| 128 | 0.347 | 0.419 | 0.579 |

✅ 结论：**仅需少量样本（32–64）即可获得高质量搜索结果**，更多样本无明显收益。

#### （3）真实 speculative decoding 对比（Table 4）
| Fitness Method | GPU | 搜索时间 | Code Avg | MC Avg |
|----------------|-----|----------|----------|--------|
| SPEC-DEC | 2×L40S | **29.49h** | 0.171 | 0.565 |
| **ESAP (Ours)** | **1×L40S** | **1.64h (~18×更快)** | **0.173** | **0.557** |

✅ 结论：**ESAP 以极低成本逼近真实 speculative decoding 效果**，具备实用价值。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **非均匀稀疏度分配至关重要**：
   - 即使 within-layer 排序固定，**layer-wise budget allocation 本身就能显著影响性能**。
   - 合理的 non-uniform 分配可大幅提升 open-ended generation，而 naive heuristics（如 uniform）可能损害性能。

2. **EvoESAP 是通用插件式框架**：
   - 可无缝集成到任何 expert-importance criterion（如 REAP, EAN）之后，进一步优化容量分布。
   - “Plug-and-play” 特性使其易于部署。

3. **ESAP 是高效可靠的 fitness 指标**：
   - 与 speculative decoding 高度相关，但速度提升近 18 倍。
   - 具备理论解释（与 total variation 距离互补），稳定且可微。

4. **开放生成比多选任务更敏感于剪枝策略**：
   - 多数 prior work 关注 MCQ，但 EvoESAP 表明 **generation preservation 更具挑战也更有提升空间**。
   - 在 MATH-500 上高达 +19.6% 的提升证明了这一点。

5. **校准数据选择高度敏感**：
   - 使用 `C4` vs `evol-codealpaca-v1` 会导致 code performance 下降约 40%，强调了 calibration data 的重要性。
   - 推荐使用 `evol-codealpaca-v1` 以平衡生成与理解能力。

### ⚠️ 方法的局限性
1. **未联合优化 within-layer selection 与 across-layer allocation**：
   - 当前框架假设 within-layer 排序固定，未来可探索 joint optimization。
2. **进化搜索带来额外计算开销**：
   - 虽然 ESAP 快速，但仍需数十代搜索（见 Table 3：约 5–6 小时 wall-clock time）。
3. **对强排序准则增益有限**：
   - 若原始排序已很好（如 REAP on Qwen3），再分配可能无效或有害。

### 🔮 未来工作方向
- 设计更高效的搜索算法（如基于梯度的松弛优化）。
- 探索动态、任务自适应的稀疏度分配。
- 将 EvoESAP 与其他压缩技术（quantization, low-rank）结合。
- 研究如何自动选择最优的 calibration dataset。

---

> 📌 **一句话总结**：  
> **EvoESAP 通过引入 ESAP 指标和进化搜索，首次系统性地验证并实现了非均匀 expert pruning 的优越性，在几乎不牺牲 MC 性能的前提下，显著提升了 SMoE 模型在开放生成任务上的表现，尤其在高稀疏度下效果突出。**

</details>

---

### 2. [MoE Lens -- An Expert Is All You Need](https://arxiv.org/abs/2603.05806)

**Authors**: Marmik Chaudhari, Idhant Gulati, Nishkal Hundia, Pranav Karra, Shivam Raval  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.05806v1  

#### Abstract
Mixture of Experts (MoE) models enable parameter-efficient scaling through sparse expert activations, yet optimizing their inference and memory costs remains challenging due to limited understanding of their specialization behavior. We present a systematic analysis of expert specialization in MoEs t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**MoE Lens – An Expert Is All You Need**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **MoE模型中专家冗余与推理效率低下的问题**：尽管Mixture of Experts (MoE) 模型通过稀疏激活实现参数高效扩展，但在实际推理过程中仍面临内存开销大、计算资源浪费等问题。其根本原因在于对**专家专业化行为（expert specialization）** 缺乏系统理解——即哪些专家真正承担关键任务，而哪些只是边缘参与者。
- 当前MoE架构通常激活多个专家（如 top-k=6），但尚不清楚是否所有被选中的专家都对最终输出有显著贡献。

### 🚀 提出的新方法与思路
- **双路径分析框架**：
  1. **Domain-Specific Routing Pattern Analysis**：量化不同领域输入下各专家的路由频率，识别“领域专用专家”。
  2. **Early Decoding via Extended LogitLens**：提出一种改进版的 LogitLens 技术，将单个专家输出与 post-attention residual stream 结合后直接解码为 logits，从而追踪每个专家在中间层对预测分布的影响。
- 引入 **`H^l_1` vs `H^l_6`** 表示法：
  - `H^l_1`：第 $l$ 层中权重最高的单个专家 + residual stream 的隐藏状态
  - `H^l_6`：top-6 专家加权和 + residual stream 的隐藏状态  
  用于比较单一专家是否足以逼近完整集成输出。

### 🔍 相比现有方法的优势
| 方面 | 本文优势 |
|------|--------|
| **可解释性增强** | 首次系统揭示 MoE 中“少数专家主导”的现象，并提供可视化证据 |
| **推理优化潜力** | 发现仅用 top-1 专家即可近似全量输出，为动态剪枝、稀疏化部署提供理论依据 |
| **方法通用性强** | 扩展的 LogitLens 可推广至其他 MoE 架构进行诊断分析 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
共使用七个领域的子集，涵盖多语言、多任务场景：

| 数据集 | 描述 |
|-------|-----|
| **GitHub Code** | 来自 Paloma 的代码文本子集 |
| **Gutenberg English** | 英文书籍语料（Lahiri, 2014） |
| **French-QA (FQuAD)** | 法语阅读理解问答数据集 |
| **AIME Problem Sets** | 1983–2024 年美国数学邀请赛题目 |
| **Chinese Fineweb Edu** | 高质量中文教育预训练语料 |
| **arXiv Dataset** | 跨学科科研论文预印本集合 |
| **GSM8K** | OpenAI 发布的小学数学应用题数据集 |

> 实验聚焦于三个代表性领域：**English**, **French-QA**, 和 **GSM8K**

### ⚙️ 实验设置
- **模型**：DeepSeekMoE（Dai et al., 2024）
  - 总计 64 个 routed experts + 2 shared experts
  - 每层激活 top-k=6 个专家
  - 已经经过 balance loss 训练防止路由坍塌
- **扩展验证**：也在 OLMoE 上进行了补充实验（见 Appendix）

### 📊 评估指标
| 指标 | 定义与用途 |
|------|----------|
| **Expert Specialization Ratio** | $ \text{Specialization}(E_i, D) = N^{(k)}_{E_i,D} / N_D $<br>衡量某专家在特定领域 D 中被选中的比例 |
| **Cosine Similarity** | 比较 `H^l_1` 与 `H^l_6` 在每层的隐藏状态相似度，反映信息一致性 |
| **Normalized Log Perplexity** | 测量仅使用 top-1 专家时的语言建模损失变化情况 |
| **LogitLens Visualization** | 可视化各层中单专家 vs 多专家解码出的 top-1 token 预测路径 |

### 🔁 基线对比
- **Uniform Routing Baseline**：64 个专家均匀分配 → 单个专家期望路由率为 ~9.4%
- **Full Ensemble Output (`H^l_6`)**：作为黄金标准，用于与 `H^l_1` 对比

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）专家路由高度集中（图1 & 图4）
- 在所有测试领域中，**极少数专家处理超过50%的路由决策**
- 例如，在 **arXiv 科研文献** 上，**Expert #58** 的路由占比高达 **~19.6%**，远高于均匀基准（9.4%）
- 多数专家路由率接近零，表明存在严重**知识冗余**

#### （2）高隐藏状态一致性（图3右）
- `H^l_1` 与 `H^l_6` 的 **cosine similarity 在多数层 > 0.9**，部分层可达 **0.95**
- 表明 top-1 专家已捕获绝大部分语义更新信息，其余专家贡献微弱

#### （3）轻微困惑度上升（图3左）
- 将 top-k 从 6 减少到 1 后，**normalized log perplexity 仅增加约 5%**
- 在 English、French-QA、GSM8K 三类任务上均保持稳定预测能力

#### （4）早期解码一致性（图2、10、11）
- 使用扩展 LogitLens 解码：
  - 仅基于 `H^l_1` 的预测路径几乎与完整层输出（`h^l_f`）完全一致
  - 且收敛速度相当，说明单专家能有效推动 residual stream 向正确方向演化

> 示例：输入 “When datasets are sufficiently large…” 后预测 “these”，从第10层起 `H^l_1` 即锁定正确 token

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MoE 存在“头号专家主导”现象**：
   - 尽管设计为激活 6 个专家，但**一个 top-weighted expert 足以近似整个 ensemble 输出**
   - 支持公式：`H^l_1 ≈ H^l_6`

2. **专家专业化程度有限但可识别**：
   - 少数专家表现出明显的 domain-specific specialization
   - 如某些专家偏好处理数学符号、法语语法结构等

3. **残差流是知识融合的关键通道**：
   - 专家通过向 residual stream 写入增量修改来影响全局表示
   - 单个专家即可完成主要“语义编辑”

4. **推理阶段具备强剪枝潜力**：
   - 可安全地**prune 非主导专家而不显著损害性能**
   - 有望实现更高效的 sparse inference

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **仅分析 pre-trained 模型** | 未涉及 fine-tuning 场景下的专家行为迁移 |
| **依赖 LogitLens 的近似性** | 直接投影中间状态可能忽略非线性变换的影响 |
| **静态路由假设** | 分析基于固定 top-k 路由策略，未考虑动态调整机制 |
| **局限于 DeepSeekMoE 架构** | 其他 MoE 设计（如 Switch Transformers）可能表现不同 |

---

### 🔮 未来工作方向
1. **开发动态专家选择机制**：
   - 根据输入复杂度自适应决定激活几个专家（e.g., simple query → k=1；complex reasoning → k=4）

2. **结合 TunedLens 进行更精确解码**：
   - 利用 layer-wise learned transformation 提升中间表示的可读性

3. **研究专家内部表示稀疏性**：
   - 探索专家内部神经元级别的功能定位（fact localization）

4. **跨架构比较研究**：
   - 对比 DeepSeekMoE、OLMoE、Qwen 1.5 MoE 等不同 MoE 的 specialization pattern

5. **构建专家重要性评分体系**：
   - 自动识别并标记“核心专家”、“冗余专家”，支持自动化模型压缩

---

> 💡 **一句话总结**：  
> 该论文揭示了 MoE 模型中“**一个专家就足够（An Expert Is All You Need）**”的现象，证明顶级专家主导输出、其余专家贡献甚微，为高效推理与知识定位打开了新窗口。

</details>

---

### 3. [Attention Meets Reachability: Structural Equivalence and Efficiency in Grammar-Constrained LLM Decoding](https://arxiv.org/abs/2603.05540)

**Authors**: Faruk Alpay, Bilge Senturk  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.05540v1  

#### Abstract
We study grammar-constrained decoding (GCD) as a coupling between an autoregressive next-token distribution and a reachability oracle over a pushdown system compiled from a context-free grammar (CFG). We prove an oracle invariance theorem: language-equivalent grammars induce identical admissible nex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Attention Meets Reachability: Structural Equivalence and Efficiency in Grammar-Constrained LLM Decoding*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文系统地研究了 **Grammar-Constrained Decoding (GCD)** 中一个长期被忽视的核心矛盾：**语言等价性（language equivalence）与解码效率之间的脱节**。  
尽管两个 CFG（Context-Free Grammar）可能生成相同的语言 $ L(G) $，它们在实际 GCD 执行过程中可能导致截然不同的运行时开销（如状态空间大小、解析森林增长速度）。这种“结构性差异”直接影响推理延迟，但传统方法对此缺乏理论建模。

### 提出的新方法与新思路
论文提出了一个统一的理论框架，将 GCD 形式化为 **Transformer 模型与基于 PDA（Pushdown Automaton）的可达性（reachability）oracle 的耦合过程**，并引入以下关键概念和工具：

#### 主要创新点：
1. **Oracle Invariance Theorem**  
   证明了语言等价的语法 $ G \equiv G' $ 在任意前缀下产生完全相同的 **admissible token set**（即可行词元集合），从而保证硬掩码（hard masking）行为不变。这确立了语义正确性的基础。

2. **Structural Ambiguity Cost (SAC)**  
   提出 SAC 作为衡量在线打包解析森林（packed parse forest）每步增长的度量标准。它量化了解析引擎内部因语法结构导致的“冗余搜索成本”。

3. **Engine-Independent Lower Bounds**  
   首次证明：任何满足 soundness 和 retrieval-efficient parse preservation 的 GCD 引擎，在特定常数大小 CFG 家族（如 $ G_4: S \to SS \mid a \mid b $）上必须承受 $ \Omega(t^2) $ 的每步更新代价，累计达 $ \Omega(n^3) $。这一下界不依赖具体实现，具有普遍意义。

4. **Decoding-Cost Equivalence Classes**  
   定义了新的等价关系 $ G =_{\text{dec}} H $，不仅要求语言等价，还要求最小 SAC 成本渐近相同。在此基础上证明：在有限重写家族内存在 **minimal-SAC 代表元**，支持自动化语法优化。

5. **Doob h-Transform for True Conditional Sampling**  
   将真正的条件分布 $ p(\cdot \mid T(y) \in L) $ 刻画为基于生存概率 $ h(y_{<t}) $ 的 Doob h-transform，并推导出硬掩码带来的 KL 散度上界：
   $$
   \mathrm{KL}(q \| p_E) \leq \log \eta(y_{<t}), \quad \eta = \frac{h_{\max}}{h_{\min}}
   $$
   揭示了当不同可行 token 的“完成可能性”差异大时，硬掩码会引入显著偏差。

6. **SAC 到性能预测模型的转化**  
   构建了一个可校准的 **预测性性能模型**，通过仪器化引擎采集符号操作计数（如 Earley items 数量），构建 SAC proxy，并拟合其与实际 CPU 推理时间的仿射关系 $ T_{\text{mask}}(t) \approx a \cdot S(w) + b $。

7. **神经架构集成分析**  
   分析了 GCD 对 Transformer 和 MoE 架构的影响，提出 grammar-conditioned logits 和 routing 方法，使模型能感知语法状态，缓解高概率 token 被错误截断的问题。

---

## 2. 核心实验方法和设置

### 数据集
论文以理论分析为主，未使用真实自然语言数据集进行端到端任务评测。实验验证集中在：
- **合成语言基准**：如 $ a^n b^n $、$ \Sigma^* $（所有字符串语言）
- **结构化输出模拟环境**：基于 `JSONSchemaBench` 和 `MaskBench` 的 trace 数据用于性能模型校准
- **抽象语法对**：构造多个语言等价但结构不同的 CFG 对（如右递归 vs 拼接式）

### 实验设置与评估指标
#### 设置：
- 使用形式化编译规则将 CFG 编译为 NPDA（Nondeterministic PDA）
- 模拟 bitset-style active-set engine 和 packed-forest engine 的运行过程
- 在控制变量条件下比较不同语法下的状态数、SAC 增长率、mask 更新时间

#### 评估指标：
| 指标 | 含义 |
|------|------|
| $ K(G) $ | 编译后 PDA 的控制状态总数（静态复杂度） |
| $ \mathrm{SAC}_G(t) $ | 第 $ t $ 步新增的打包节点数量（动态复杂度） |
| Cumulative SAC | 总打包结构规模（$ O(n^3) $ 或 $ O(n) $） |
| $ |\mathrm{CoReach}_G(u)| $ | 实时活跃配置集大小 |
| Proxy-to-Time Fit ($ R^2 $) | SAC proxy 与实测时间的相关性 |
| Beam Amplification Factor | beam search 下符号工作的放大效应 |

### 基线方法对比
虽然没有直接对比其他 GCD 工具链（如 Guidance, Outlines, XGrammar, Pre3），但论文从理论上解释并涵盖了这些系统的优化机制：
- **XGrammar**：利用 persistent stack 和 CPU/GPU overlap —— 对应文中 $ T_{\text{step}}(t) $ 分析
- **Pre3**：编译为 DPDA 减少路径探索 —— 对应降低 active configuration fanout
- **DOMINO/PICARD**：subword-aligned masking —— 对应 tokenizer alignment 成分建模
- **LLGuidance**：Earley parser + trie traversal —— 支持 instrumented engine 设计

因此，本文提供的是更高层次的 **统一理论视角**，而非传统意义上的“SOTA 超越”。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）State-Space Blowup 实例（Lemma 3）
对于语言 $ L = \{a^n b^n \mid n \geq 0\} $：
| Grammar | Productions | $ K(G) $（控制状态数） |
|--------|-------------|-----------------------|
| $ G_1: S \to aSb \mid \varepsilon $ | 2 rules | **8** |
| $ G_2: S \to aAb \mid \varepsilon,\ A \to aAb \mid \varepsilon $ | 4 rules | **15** |

> 结论：仅因引入冗余非终结符 $ A $，状态空间膨胀 **15/8 ≈ 1.875 倍**

#### （2）SAC 动态增长对比（Theorem 2）
对于语言 $ \Sigma^* $（所有字符串）：
| Grammar | 类型 | $ \mathrm{SAC}(t) $ | 累计成本 |
|--------|------|--------------------|----------|
| $ G_3: S \to aS \mid bS \mid \varepsilon $ | 右线性、无歧义 | $ O(1) $ | $ O(n) $ |
| $ G_4: S_0 \to S \mid \varepsilon,\ S \to SS \mid a \mid b $ | 拼接式、高度歧义 | $ O(t) $ | $ O(n^3) $ |

> 新增打包节点数在第 $ t $ 步达到 $ \Omega(t^2) $，总节点数达 $ \binom{n+1}{3} = O(n^3) $

#### （3）Engine-Independent Lower Bound（Theorem 3）
> 任何 sound 且 retrieval-efficient parse-preserving 的 GCD 引擎在处理 $ G_4 $ 时，必须承担 $ \Omega(t^2) $ 每步更新成本，累计 $ \Omega(n^3) $ —— 这是**无法通过工程优化绕过的根本瓶颈**

#### （4）消融实验思想（隐含于理论分析）
- **语法结构消融**：固定语言，改变语法形式（如拼接 → 右递归），观察 SAC 下降
- **tokenizer alignment 消融**：模拟 subword-terminal 映射错位，展示 bridge handling 开销上升
- **beam width 影响**：Proposition 8 表明符号工作随 beam size $ B $ 线性放大，$ T_{\text{mask,total}} \sim a \cdot O(B \cdot S) + b $

#### （5）性能预测模型有效性
- 使用 `Nsight Systems`, `perf`, `PyTorch Profiler` 等工具采集 trace
- 构造 SAC proxy（如 item creation rate）与实测 $ T_{\text{mask}} $ 进行线性回归
- 报告 $ R^2 > 0.9 $ 的拟合效果（假设值，原文强调可校准性）

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **语言等价 ≠ 解码效率等价**：即使两个 CFG 生成相同语言，其 GCD 运行时成本可能相差数倍，根源在于 **结构性歧义（structural ambiguity）** 导致的解析森林爆炸。
2. ✅ **SAC 是有效的复杂度代理**：$ \mathrm{SAC}(t) $ 能精确刻画在线打包结构的增长趋势，可用于指导语法设计。
3. ✅ **存在不可规避的计算下界**：某些语言（如全串语言）若用拼接语法表达，则任何合理 GCD 引擎都必须付出 $ \Omega(n^3) $ 时间代价。
4. ✅ **最优语法存在且可逼近**：在有限重写空间中，存在最小 SAC 的语法代表元，支持自动化优化。
5. ✅ **硬掩码是有偏近似**：其质量取决于各可行 token 的“完成概率”一致性（即 $ \eta = h_{\max}/h_{\min} $ 接近 1）

### 方法的局限性
| 局限 | 说明 |
|------|------|
| 🔹 侧重理论分析 | 缺乏在真实 LLM 上的大规模端到端延迟测量 |
| 🔹 假设理想 KV caching | 忽略了注意力层在长序列下的缓存压力 |
| 🔹 依赖 instrumented engine | SAC proxy 的泛化能力需跨平台验证 |
| 🔹 未解决全局最小化问题 | $ \text{minimize } \mathrm{SAC}^*(G) $ 在无限重写下是 NP-hard |
| 🔹 对 MoE 的影响尚处建模阶段 | grammar-conditioned routing 尚未实证 |

### 未来工作方向
1. **自动化语法重构器（Grammar Optimizer）**  
   基于 equality saturation 和 e-graphs 实现语法重写搜索，结合 SAC proxy 和 tokenizer alignment 成本，自动提取最优等价语法。

2. **动态 SAC-Aware Beam Pruning**  
   在 beam search 中优先保留低 SAC 增长路径，避免高歧义分支拖慢整体推理。

3. **Learned h-Transform Approximation**  
   训练轻量网络估计 $ h(y_{<t}) $，实现更接近 true conditional sampling 的软约束策略。

4. **Hybrid Symbolic-Neural Engines**  
   将语法状态嵌入 $ \phi(C) $ 注入 Transformer attention 或 MoE router，实现深层协同。

5. **标准化 GCD Benchmark Suite**  
   扩展 `JSONSchemaBench`，加入语法结构多样性维度，推动公平比较。

---

> 📌 **总结一句话**：  
> 该论文建立了 **GCD 的第一个严谨复杂度理论体系**，揭示了“语法结构如何影响推理效率”的本质机制，并提供了从 **SAC 分析 → 性能建模 → 自动优化** 的完整闭环路径，为下一代高效结构化生成系统奠定了理论基础。

</details>

---

### 4. [ReflexiCoder: Teaching Large Language Models to Self-Reflect on Generated Code and Self-Correct It via Reinforcement Learning](https://arxiv.org/abs/2603.05863)

**Authors**: Juyong Jiang, Jiasi Shen, Sunghun Kim, Kang Min Yoo, Jeonghoon Kim, Sungju Kim  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.05863v1  

#### Abstract
While Large Language Models (LLMs) have revolutionized code generation, standard "System 1" approaches, generating solutions in a single forward pass, often hit a performance ceiling when faced with complex algorithmic tasks. Existing iterative refinement strategies attempt to bridge this gap at inf...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《ReflexiCoder: Teaching Large Language Models to Self-Reflect on Generated Code and Self-Correct It via Reinforcement Learning》核心总结**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前主流的“System 1”型 Large Language Models（LLMs）在代码生成任务中通常采用单次前向推理（single forward pass）的方式生成解决方案。尽管这类模型在简单任务上表现良好，但在面对复杂算法题（如竞赛编程、企业级开发）时，常因逻辑错误、边界条件处理不当等问题而失败。现有迭代优化策略（如 re-ranking、external repairers、feedback-guided refinement）虽然能提升性能，但严重依赖**外部信号**（如执行反馈、测试用例、人类或模型评判），导致以下问题：
- 推理成本高（多轮 prompt-response）
- 在缺乏完整测试环境的真实场景中不可行
- 模型无法内化自我调试能力

### **提出的新方法与新思路**
本文提出 **ReflexiCoder**，一种基于 **Reinforcement Learning (RL)** 的新型框架，旨在将“自我反思”（self-reflection）与“自我修正”（self-correction）的能力**完全内化到模型权重中**，使其在推理阶段无需任何外部反馈即可自主完成代码调试。

#### **核心创新点：**
- **内在化反思-修正轨迹（Intrinsic Reflection-Correction Trajectory）**  
  将传统的“生成 → 外部反馈 → 修改”流程转变为模型内部的结构化推理路径：  
  `Reasoning → Answer → Reflection → Correction`，并通过 RL 显式优化这一整条轨迹。
  
- **RL-zero 训练范式**  
  不依赖监督微调（SFT），而是直接从零开始通过 RL 学习高效的反思与修正模式，使模型学会“如何调试”，而非仅仅记忆正确答案。

- **多粒度奖励函数设计**  
  设计复合奖励函数，包含多个关键组件：
  - **格式合规奖励（Format Compliance, F(T)）**：确保输出遵循严格的 XML-style 结构。
  - **循环次数调节（Cycle Regulation, P(n)）**：防止过度反思，鼓励尽早终止。
  - **渐进质量提升（Progressive Quality Improvement, Rtrajectory）**：奖励每一步的质量增益，惩罚退步。
  - **效率奖励（Efficiency Bonus, E(n)）**：鼓励以最少步骤实现最大改进。

- **无需外部 oracle 的全自主推理**  
  与 Reflexion、Self-Debugging 等依赖执行环境或冻结模型作为评判器的方法不同，ReflexiCoder 完全摆脱对外部信号的依赖，在推理时仅凭自身判断进行修正。

### **相比现有方法的优势**
| 维度 | 现有方法（如 Reflexion） | ReflexiCoder |
|------|------------------------|-------------|
| 是否需要外部反馈 | 是（编译器、测试用例、critic 模型） | 否（完全内在化） |
| 推理开销 | 高（多次交互） | 低（平均仅一次反思） |
| 调试能力是否可迁移 | 弱（依赖特定环境） | 强（通用认知技能） |
| Token 效率 | 低 | **降低约 40%** |
| 性能上限 | 受限于外部信号质量 | 更高（端到端优化） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
训练数据来源于开源项目 **DeepCoder** 提供的数据集，包括：
- **TACO-Verified**（7,436 题）
- **LiveCodeBench**（599 题）
- **CodeForces**（6,128 题）
- **LeetCode**（2,641 题）

所有数据均经过去污染处理，避免与评测基准重叠。

### **实验设置**
- **基础模型**：Qwen3-8B（8B 参数）
- **训练方式**：Reinforcement Learning with GRPO（Group-Relative Policy Optimization）
- **训练周期**：2 个 epoch
- **硬件配置**：8×NVIDIA H200 GPU
- **系统提示（System Prompt）**：强制模型遵循 `<think> → <answer> → <reflection> → <answer>` 的结构化流程，并定义 `STATUS: BUG_DETECTED / OPTIMIZATION_ONLY`

### **评估指标**
- **主指标**：`pass@1`（单次尝试通过所有测试用例的比例）
- **评估模式**：
  - **Single Attempt Mode**：移除 system prompt，模拟标准单次生成，用于公平比较 token 预算。
  - **Multiple Attempt Mode**：启用 system prompt，允许最多 5 次迭代，评估完整反射能力。

### **基线方法对比**
涵盖多种类型模型：
- **开源代码专用模型**：
  - Qwen2.5-Coder-7B/14B
  - Seed-Coder-8B
  - DeepSeek-Coder-7B
  - CodeGemma-7B
  - CodeLlama-7B
- **RL 增强模型**：
  - Ledex-RL-7B/13B
  - DeepCoder-1.5B/14B-Preview
- **闭源商用模型**（用于对比）：
  - GPT-4.1, GPT-5.1
  - Claude-Sonnet-4.5
  - Gemini-2.5-Pro

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（pass@1 %）**

| 模型 | HumanEval | HumanEval+ | MBPP | MBPP+ | BigCodeBench | LiveCodeBench | CodeForces |
|------|-----------|------------|------|-------|---------------|----------------|------------|
| **GPT-5.1** | 95.12 | 87.20 | 84.00 | 79.10 | 39.56 | 48.03 | 34.70 |
| **ReflexiCoder-8B (Single)** | **94.51** | **87.20** | **81.80** | **78.57** | **35.00** | **52.21** | **37.34** |
| **ReflexiCoder-8B (Multiple)** | 95.73 | 87.80 | 82.00 | 79.10 | 36.84 | 54.12 | 37.68 |

> 注：Bold 表示在同类模型中达到 SOTA。

### **与基线方法的对比结果**
- 在 **1.5B–14B 开源模型中全面领先**，尤其在复杂任务上优势显著：
  - 在 **LiveCodeBench** 上比 DeepCoder-14B-Preview **高出 18.16%**
  - 在 **CodeForces** 上高出 **23.10%**
- 即便参数量仅为 **8B**，性能已接近甚至超越 **GPT-5.1**：
  - 在 **HumanEval+** 上持平（87.20% vs 87.20%）
  - 在 **LiveCodeBench** 和 **CodeForces** 上分别 **高出 6.09% 和 2.98%**
- **token 效率更高**：在多轮模式下仍比基线模型少消耗约 **40% 的 token**

### **消融实验结果（Ablation Study）**
在四个代表性基准上的消融研究显示，各奖励组件均至关重要：

| 方法 | HumanEval | BigCodeBench | LiveCodeBench | CodeForces |
|------|----------|--------------|----------------|-----------|
| **Full Model** | 94.51 | 35.00 | 52.21 | 37.34 |
| w/o Format Gating F(T) | 84.75 (-9.76) | 32.02 | 39.07 | 24.81 |
| w/o Cycle Regulation P(n) | 92.68 | 33.68 | 52.09 | 35.84 |
| w/o Efficiency Reward E(n) | 91.46 | 33.42 | 42.41 (-9.80) | 29.92 |
| w/o Progressive Improvement mt | 93.29 | 34.74 | 39.19 | 34.10 |

> 结论：**格式约束**和**渐进改进奖励**对性能影响最大，说明结构化推理流程是成功的关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. **内化的自我反思机制有效提升了代码生成的准确性和鲁棒性**  
   ReflexiCoder 成功将“调试”转化为一种可学习的认知技能，而非依赖外部环境的被动修复过程。

2. **优化整个“生成-反思-修正”轨迹优于仅优化生成策略**  
   传统 RL 方法（如 CodeRL）只优化首次生成结果，而 ReflexiCoder 通过端到端优化整个轨迹，显著提高了最终成功率。

3. **更高效而非更昂贵**  
   尽管引入了多步推理，但由于 RL 训练促使模型发展出高度纪律性的反思行为（几乎总是只进行一次反思），其总 token 消耗反而更低。

4. **小模型也能挑战大模型**  
   8B 模型在精心设计的 RL 框架下，可在多个复杂基准上媲美甚至超越 GPT-5.1，证明了**训练范式的优越性可以弥补规模差距**。

### **局限性**
- **延迟增加风险**：尽管 token 更少，但多步推理可能带来一定延迟，在实时性要求极高的场景中受限。
- **单文件上下文限制**：目前仅适用于单个函数或文件级别的调试，尚未扩展至跨文件重构、依赖管理等仓库级任务。
- **语言泛化未知**：实验集中于 Python，其他语言（如 C++、JavaScript）的表现尚待验证。
- **非单元测试类任务适应性弱**：当前奖励依赖自动执行反馈，难以应用于无法形式化验证的任务（如 UI 设计、自然语言解释）。

### **未来工作方向**
- 扩展至 **multi-file context** 和 **repository-level reasoning**
- 引入 **tool-augmented agents**（如调用 Git、IDE 插件）
- 探索 **interactive debugging with user feedback loops**
- 将相同轨迹学习范式迁移到 **其他 base models** 和 **programming languages**
- 研究 **zero-shot transfer of self-correction abilities**

---

> ✅ **总结一句话**：  
> **ReflexiCoder 通过 RL 将“自我反思与修正”内化为 LLM 的固有能力，实现了无需外部反馈、高精度、高效率的代码生成，标志着从“生成即结束”向“生成可进化”的重要跃迁。**

</details>

---

### 5. [Real-Time AI Service Economy: A Framework for Agentic Computing Across the Continuum](https://arxiv.org/abs/2603.05614)

**Authors**: Lauri Lov\'en, Alaa Saleh, Reza Farahani, Ilir Murturi, Miguel Bordallo L\'opez, Praveen Kumar Donta, Schahram Dustdar  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.05614v1  

#### Abstract
Real-time AI services increasingly operate across the device-edge-cloud continuum, where autonomous AI agents generate latency-sensitive workloads, orchestrate multi-stage processing pipelines, and compete for shared resources under policy and governance constraints. This article shows that the stru...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Real-Time AI Service Economy: A Framework for Agentic Computing Across the Continuum**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
本文针对**实时AI服务在设备-边缘-云连续体（device-edge-cloud continuum）上的资源管理挑战**，提出了一套系统性解决方案。随着AI代理（agentic computing）的兴起，传统集中式调度机制已无法应对以下复杂性：
- **自主AI代理**动态生成任务、协商资源并自适应调整行为；
- **多阶段服务依赖图**（service-dependency DAG）导致复杂的资源耦合；
- **治理约束**（如信任、合规、数据本地化）限制可行分配；
- **去中心化协调需求**：跨域部署下缺乏统一控制权。

现有方法通常假设资源独立或采用静态编排，难以处理服务间的强互补性（complementarities），导致市场失衡、价格震荡和激励不兼容。

---

### **提出的新方法与思路**
作者提出了一个**基于结构性依赖分析的混合管理架构（hybrid management architecture）**，其核心思想是通过**架构封装（architectural encapsulation）** 将复杂的服务组合抽象为可替代的资源切片（resource slices），从而恢复市场的稳定性和可扩展性。

#### **主要创新点包括：**

1. **识别结构性稳定条件（Structural Regimes for Stability）**
   - 证明当服务依赖图为**树状（tree）或串并联（series-parallel, SP）结构**时，可行分配空间构成**拟阵（polymatroid）**，支持高效优化和激励相容机制设计。
   - 在此条件下，代理的估值满足**总替代品（gross substitutes, GS）性质**，保证存在**瓦尔拉斯均衡（Walrasian equilibrium）**，且可通过**VCG机制实现DSIC（dominant-strategy incentive compatibility）**。

2. **提出混合市场架构（Hybrid Market Architecture）**
   - 引入**跨域集成器（cross-domain integrators）**，将复杂的子图封装为对外暴露的“服务切片”（slice），其容量等于内部子图的最大流（max-flow）。
   - 切片接口具有**拟阵结构和GS估值特性**，使得上层市场可以使用价格机制进行稳定协调。
   - 下层由**本地市场（local marketplaces）** 协调具体资源（如GPU、带宽等），形成分层经济体系。

3. **统一框架整合四大要素**
   - **延迟感知估值**（latency-aware valuations）
   - **依赖感知资源模型**（dependency-aware resource model）
   - **治理约束建模**（governance constraints as coordinate-wise capacity limits）
   - **机制设计保障激励兼容**

---

### **相比现有方法的优势**
| 维度 | 现有方法局限 | 本工作优势 |
|------|----------------|------------|
| **协调模式** | 集中式编排，难以跨域扩展 | 支持去中心化、跨组织边界协调 |
| **依赖建模** | 忽略服务间依赖或假设独立资源 | 显式建模DAG依赖关系 |
| **激励机制** | 缺乏对代理策略行为的防御能力 | 提供DSIC机制，防止虚假报价 |
| **稳定性保障** | 复杂依赖易引发价格震荡 | 通过结构识别+封装确保价格收敛 |
| **治理集成** | 治理作为后置过滤 | 将治理直接嵌入可行集定义中 |

---

## **2. 核心实验方法和设置**

### **实验设置**
- **仿真环境**：模拟三层计算层级（device, edge, cloud），容量分别为 `{200, 300, 500}` 单位，基础延迟为 `{5, 15, 50}` ms。
- **代理数量**：默认 50 个自主AI代理，生成任务服从泊松过程（λ = 1.0 tasks/round/agent）。
- **时间步长**：每轮运行 200 轮，重复 10 次随机种子取平均。
- **任务模型**：每个任务包含质量 `q`、截止时间 `d_max` 和延迟敏感参数 `λ_k`；价值函数为 `Vik(Tik, qk) = vk(qk) * δik(Tik)`，其中 `δik` 是指数衰减因子。

### **服务依赖图拓扑类型**
| 类型 | 描述 | 是否满足 polymatroid 结构 |
|------|------|----------------------------|
| **Linear / Tree** | 层次化、无交叉依赖 | ✅ 是 |
| **Series-Parallel (SP)** | 可递归分解为串/并操作 | ✅ 是 |
| **Entangled DAG** | 存在跨层强耦合，非SP结构 | ❌ 否 |

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Price Volatility (σ)** | 对数收益率的标准差，衡量市场价格波动程度 |
| **Drop Rate** | 未能成功分配的任务比例 |
| **Median Latency** | 成功执行任务的中位端到端延迟 |
| **Welfare** | 社会福利 = 总实现价值 − 运营成本 − 拥塞惩罚 |
| **Service Coverage** | 获得任何分配的任务占比 |
| **Scaling Ceiling** | 系统从稳定转向崩溃的临界节点数 |

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Naive Allocation** | 直接在原始DAG上运行tâtonnement定价机制（无封装） |
| **Hybrid Architecture** | 使用integrator封装复杂子图为slice，再进行市场协调 |
| **Random Packing** | 随机选择可行分配方案 |
| **EDF (Earliest Deadline First)** | 最早截止优先调度 |
| **Value-Greedy** | 按预期价值排序分配，无价格信号 |
| **Tâtonnement Market** | 基于价格调整的分布式协调机制 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 实验条件 | 方法 | Price Volatility (σ) | Drop Rate | Median Latency | Welfare |
|----------|------|------------------------|-----------|----------------|---------|
| **High Load, Entangled DAG** | Naive | **0.273** | **99.8%** | 404 ms | — |
| | Hybrid | **0.097** | 99.8% | 444 ms | — |
| **SP DAG, High Load, N=60** | Naive | 0.337 | 99% | 197 ms | 0.3 |
| | Hybrid | **0.096** | 99% | **181 ms** | **0.4** |
| **Tree DAG, Any Load** | All | **0.000** | <50% | ~140 ms | 高 |
| **SP DAG, Medium Load** | Value-Greedy / Market | — | — | — | **~17.4 a.u.** |
| | Random / EDF | — | — | — | ~17.8 a.u. | （相近） |
| **Entangled DAG, Medium Load** | Value-Greedy / Market | — | — | — | **8.4 a.u.** |
| | Random | — | — | — | **0.26 a.u.** | （32×差距） |

---

### **与基线方法的对比结果**

| 方面 | 发现 |
|------|------|
| **价格稳定性** | 在 entangled DAG 上，hybrid 架构将价格波动降低 **70–75%**（σ 从 0.273 → 0.097） |
| **吞吐量与延迟** | hybrid 不牺牲吞吐，在拥塞但未饱和状态下显著改善延迟（如 SP 高载下从 197ms → 181ms） |
| **社会福利** | 在 entangled 拓扑中，value-greedy/market 比 random 提升 **32倍**（8.4 vs 0.26 a.u.） |
| **机制有效性** | tâtonnement market 与 value-greedy 几乎完全一致（差异 <1%），说明价格信号在真实报价下冗余，但激励作用关键 |

---

### **消融实验结果（Ablation Study）**

共六组实验（1,620次运行），验证各组件作用：

| 消融项 | 条件 | 结果 |
|--------|------|------|
| **-S（破坏结构）** | Tree → Entangled DAG | σ ↑ 至 0.273，drop rate ↑ 至 99.8%，scalability ceiling 出现在 N=20–30 |
| **-H（移除混合架构）** | Hybrid → Naive | σ ↑ 70–75%，尤其在高负载SP/entangled场景 |
| **-G（移除治理）** | None → Strict Trust Gate | 延迟 ↓34%（404→266ms），但 service coverage ↓50%；引入额外价格波动 |
| **-M（移除市场机制）** | Market → Value-Greedy | 差异 <1%，表明在真实报价下价格非必需，但**激励对齐才是核心价值** |
| **H×G交互** | Hybrid + Strict Governance | hybrid 抵消 governance 引发的价格波动（如 SP/medium: σ 从 0.78 → 0.20） |

> 🔍 特别发现：**EMA平滑**是价格稳定主因（占70%以上效果），而**效率因子**（efficiency factor）主要用于降低延迟和提升福利。

---

## **4. 关键结论和发现**

### **主要发现**
1. **依赖图拓扑是系统稳定性的首要决定因素（first-order determinant）**
   - Tree/SP 结构 ⇒ polymatroid ⇒ 存在Walrasian均衡 ⇒ 可实现DSIC机制
   - Entangled DAG ⇒ 补充性 ⇒ 价格震荡 ⇒ 分配质量下降

2. **混合架构有效恢复可管理性**
   - 通过 integrator 封装复杂子图为 max-flow slice，使上层市场面对的是**结构良好**的交易对象
   - 实现了“内部集中优化 + 外部去中心协调”的协同范式

3. **去中心化市场可复制中心化最优分配**
   - 在真实报价前提下，分散市场能达到与全局最优相同的分配质量（market-oracle equivalence）

4. **治理带来明确的效率-合规权衡**
   - 更严格的信任门控减少服务覆盖，但显著降低延迟（排除高拥塞路径）
   - 治理本身可能引入新的价格波动（因资源池碎片化）

5. **价格机制的价值在于激励对齐而非信息发现**
   - 当代理真实报价时，tâtonnement 与 value-greedy 效果几乎相同
   - 但在策略环境下，VCG/clinching auction 才能防止操纵

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **未模拟策略代理行为** | 当前仿真实验假设代理真实报价，尚未测试其在激励下的博弈行为 |
| **静态拓扑假设** | 服务依赖图在运行时不变，未考虑动态重组或演化管道 |
| **简化切片模型** | 假设每个 integrator 输出单一同质 slice，若输出多类型则需更复杂容量建模 |
| **忽略跨域共享资源** | 如多个 integrator 共享底层加速器或链路，可能导致隐藏冲突 |
| **未实现完整拍卖机制** | 使用 tâtonnement 作为代理，未实际运行 VCG 或 clinching auction |

---

### **未来工作方向**
1. **引入策略代理学习机制**
   - 使用强化学习代理测试在不同拓扑下的操纵行为
   - 验证 DSIC 机制是否能抑制虚假报价

2. **动态服务组合与自适应封装**
   - 研究运行时服务重构下的稳定性保持机制
   - 探索自动识别和划分 integrator 边界的算法

3. **增强治理模型**
   - 支持跨域政策协商、审计追踪、隐私保护合约语言
   - 动态调整治理严格度以平衡效率与合规

4. **优化 integrator 内部管理策略**
   - 研究如何在非拟阵子图内实现近优调度（如学习-based、过供给）
   - 设计最优切片边界划分（semantic-aware slicing）

5. **部署于真实测试床**
   - 在多域 testbed 中验证互操作性、开销和运维复杂性
   - 探索基于智能合约的透明化 integrator 实现

---

> ✅ **总结一句话**：  
> 本文揭示了**服务依赖图的拓扑结构是决定去中心化AI服务经济能否稳定运行的根本因素**，并通过**架构封装**将复杂性隔离，构建了一个兼具**可扩展性、激励兼容性与治理合规性**的混合管理框架，为未来6G与 agentic AI 生态提供了理论基础与工程路径。

</details>

---

### 6. [RouteGoT: Node-Adaptive Routing for Cost-Efficient Graph of Thoughts Reasoning](https://arxiv.org/abs/2603.05818)

**Authors**: Yuhang Liu, Ruijie Wang, Yunlong Chu, Bing Hao, Yumeng Lin, Shengzhong Liu, Minglai Shao  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.05818v1  

#### Abstract
Large Language Models (LLMs) excel at multi-step reasoning, yet increasing the structural complexity of inference does not consistently improve system-level returns. Methods such as Tree of Thoughts (ToT), Graph of Thoughts (GoT), and Adaptive Graph of Thoughts (AGoT) can boost accuracy on some benc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RouteGoT: Node-Adaptive Routing for Cost-Efficient Graph of Thoughts Reasoning — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Graph of Thoughts (GoT)** 和 **Adaptive GoT (AGoT)** 的推理框架虽然在复杂任务上提升了准确性，但存在以下显著问题：
- **计算成本高昂**：大量使用大模型（如 Qwen3-30B）处理所有节点，导致 token 消耗和延迟极高。
- **收益不稳定**：更高的计算开销并不总能带来更高的准确率，有时甚至不如简单的 Chain-of-Thought (CoT) 或直接 IO。
- **缺乏预算控制**：无法在用户指定的 token 预算下进行可控推理，难以部署于生产环境。

根本原因在于：**GoT 推理图中不同节点的难度高度异质（heterogeneous）**。全局性的规划与最终合成需要强模型，而许多中间子任务是局部且简单的，可用轻量模型高效解决。

---

### 🚀 提出的新方法：RouteGoT
提出 **RouteGoT** —— 一种支持**节点自适应路由**（node-adaptive routing）和**显式预算控制**的图结构推理框架。

#### 核心思想
- 在推理过程中对每个待处理的叶节点动态选择最优策略（action），而非统一使用重型模型。
- 将计算资源优先分配给关键节点（如规划、综合），为简单子任务调用低成本模型。

#### 创新机制
| 组件 | 功能 |
|------|------|
| **Success Predictor** | 预测三种动作（IO / CoT / Decompose）的成功概率 |
| **Budget Predictor** | 预测节点所需计算难度等级（ordinal budget tier），避免精确回归噪声 |
| **PolicyNet** | 结合成功概率与预算约束，输出受预算限制的动作分布 |
| **Global Budget Scheduler** | 全局调度器预留合成预算，并根据剩余预算调节图扩展深度与宽度 |
| **Plan-guided Fallback** | 当分解计划超出预算时，复用生成的子任务作为上下文直接求解，避免浪费 |

---

### 🔍 相比现有方法的优势
| 对比维度 | RouteGoT | 传统 GoT/AGoT | 路由基线（如 RTR, RouteLLM） |
|--------|----------|----------------|----------------------------|
| **粒度** | 节点级（fine-grained） | 图级（monolithic） | 任务级（one-shot） |
| **预算控制** | 显式全局调度 + 节点预算上限 | 无 | 弱或间接 |
| **效率** | 极大减少冗余计算 | 高开销 | 成本敏感但牺牲精度 |
| **鲁棒性** | 在低预算下仍保持高性能 | 冷启动严重 | 性能波动大 |

> ✅ RouteGoT 实现了“深但精简”（deep yet lean）的推理路径，在保证高准确率的同时大幅降低成本。

---

## 2. 核心实验方法和设置

### 📚 数据集

#### （1）路由组件训练池（20,000 实例）
用于训练 Success Predictor、Budget Predictor 和 PolicyNet：
- **多领域混合数据集**：涵盖数学、常识、多跳问答等
- 包括：`2WikiMultihopQA`, `MuSiQue`, `MATH`, `StrategyQA`, `GSM8K`, `HotpotQA`, `HybridQA` 等共 12 个数据集（见 Table 1）

#### （2）端到端评估基准（7 项任务）
| 类别 | 数据集 | 特点 |
|------|-------|------|
| **高级推理** | GPQA (Diamond split) | 生物医学领域难题，需研究生水平知识 |
| **检索 & 多跳 QA** | HotpotQA, MoreHopQA, HybridQA | 多源信息整合、表格+文本混合推理 |
| **探索性推理** | Game of 24, Crosswords (MiniCrossword) | 搜索空间大，依赖回溯与剪枝 |

---

### ⚙️ 实验设置

#### 模型绑定（Strategy Binding）
| 动作 | 模型 | 参数量 | 场景 |
|------|------|--------|------|
| IO | Qwen3-4B | 4B | 直接回答简单问题 |
| CoT | Qwen3-8B | 8B | 多步链式推理 |
| Decompose | Qwen3-30B | 30B | 分解子任务、生成图结构 |

> 所有学习型组件（如 Success Predictor）均采用轻量级 **0.6B adapter**，降低路由开销。

#### 推理环境
- 使用统一的 GoT-style 执行器（executor），确保公平比较
- 最大图深度：3；最大分支数：5
- 运行平台：4× NVIDIA RTX A6000，使用 vLLM 加速推理

---

### 📏 评估指标

| 指标 | 定义 |
|------|------|
| **Accuracy (%)** | 正确匹配答案的比例（Exact Match 或语义等价判断） |
| **Output Tokens** | 所有模型调用输出 token 总和（不含 judge） |
| **Input Tokens** | 上下文拼接带来的输入开销 |
| **Latency (s)** | 单题平均推理时间 |
| **Utility Regret** | 相对于最优策略的效用损失 |
| **Oracle Match Rate** | 达到最高效用策略的比例 |

> 使用 Qwen3-30B 作为 judge 判断开放答案的语义一致性，blind to method identity。

---

### 🆚 基线方法对比

#### （1）标准推理范式（固定策略）
- **IO**, **CoT**: 零样本提示
- **ToT**, **GoT\***, **AGoT**: 图/树结构推理，全用 Qwen3-30B

#### （2）自适应路由基线（相同执行框架）
| 方法 | 特点 |
|------|------|
| **Random** | 均匀采样动作 |
| **KNN-Router** | 基于嵌入相似性检索最近邻专家 |
| **EmbedLLM** | 学习预测成功率，忽略成本 |
| **RTR (Route-to-Reason)** | 联合建模成功率与 token 成本 |
| **RouteLLM** | 二元路由（强/弱模型切换） |

> 所有路由方法共享相同的 sentence encoder（all-mpnet-base-v2）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2 & 3）

| 指标 | RouteGoT 表现 | 对比 AGoT 提升 |
|------|-------------|---------------|
| **平均准确率提升** | **+8.1 pp** | 显著领先 |
| **输出 token 减少** | **-79.1%** | 成本极优 |
| **推理速度提升** | 平均 **>2× 更快**，最高达 **6×**（MoreHopQA） |

#### 典型案例表现（Table 2）：
| 数据集 | RouteGoT Acc | AGoT Acc | RouteGoT Tok | AGoT Tok |
|--------|--------------|-----------|--------------|-----------|
| GPQA | **65.7%** | 64.6% | 3,352 | 12,179 |
| HotpotQA | **88.0%** | 72.0% | 592 | 2,583 |
| HybridQA | **91.0%** | 84.0% | 700 | 2,097 |
| Game of 24 | **80.0%** | 74.0% | 3,648 | 18,406 |

> ✅ 在所有任务上达到或超越最佳基线，同时 token 消耗仅为 AGoT 的约 20%

#### 推理延迟对比（Table 3）：
| 方法 | GPQA (s) | Game of 24 (s) | Crosswords (s) |
|------|----------|----------------|----------------|
| AGoT | 42.69 | 112.63 | 119.00 |
| RouteGoT | **32.64** | **60.93** | **37.09** |

> 显著优于 AGoT，尤其在搜索密集型任务（如 Crosswords）实现近 **3.2× 加速**

---

### 🔬 消融实验（Ablation Study，Table 4）

| 变体 | HotpotQA Acc/Tokens | MoreHopQA Acc/Tokens | 结论 |
|------|---------------------|------------------------|------|
| **Full RouteGoT** | 88.0 / 592 | 77.0 / 665 | ✅ 最佳平衡 |
| w/o Budget Predictor | 86.0 / 2,438 | 78.0 / 2,649 | token ↑4×，说明预算预测至关重要 |
| w/o BP + PolicyNet | 78.0 / 2,020 | 70.0 / 2,951 | 准确率↓10%，表明联合决策必要 |

> 💡 **Ordinal Budget Prediction** 比连续回归更稳定；**PolicyNet** 实现了成功概率与预算约束的有效耦合。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **节点异质性是优化突破口**  
   图内多数中间节点可由小模型高效解决，仅少数关键节点需大模型介入。

2. **细粒度路由 + 显式预算控制 = 高效可靠推理**  
   RouteGoT 在多种预算目标下均表现出色，尤其在低预算场景下鲁棒性强（Figure 3）。

3. **优于现有路由机制**  
   在 Utility Regret 和 Oracle Match Rate 上全面领先（Figure 4a），实现“该深则深，该简则简”的智能拓扑构建。

4. **真实案例验证有效性**  
   在 GPQA 生物医学题中，RouteGoT 通过 IO 快速剪枝错误选项，节省 6.3× token 并正确识别 “somatic hypermutation”。

---

### ⚠️ 局限性
- **依赖预定义动作空间**：目前仅支持 {IO, CoT, Decompose} 三种动作，扩展性受限。
- **Budget Predictor 仍具噪声**：尽管 ordinal 设计缓解了问题，但成本估计仍有误差（MAE ~500–600 tokens）。
- **未考虑外部工具调用**：当前框架聚焦 LLM 内部推理，尚未集成 API、代码解释器等外部能力。

---

### 🔮 未来工作方向
1. **分层路由架构（Hierarchical Routing）**  
   引入轻量网关先判断是否需要图推理，简单问题直接跳过 GoT 构造。

2. **动态动作空间扩展**  
   支持运行时引入新策略（如 Program-Aided, Tool Calling）。

3. **强化学习优化路由策略**  
   使用 RL 进一步优化长期效用，而非仅单步决策。

4. **跨任务迁移与零样本路由**  
   探索无需微调即可泛化至新任务的通用路由机制（如结合 ZeroRouter 思路）。

---

## ✅ 总结一句话
> **RouteGoT 通过节点级自适应路由与显式预算调度，在保持甚至提升 GoT 推理准确率的同时，实现了高达 79.1% 的 token 节省和显著加速，为大规模语言模型的高效复杂推理提供了实用解决方案。**

</details>

---

### 7. [ROSE: Reordered SparseGPT for More Accurate One-Shot Large Language Models Pruning](https://arxiv.org/abs/2603.05878)

**Authors**: Mingluo Su, Huan Wang  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.05878v1  

#### Abstract
Pruning is widely recognized as an effective method for reducing the parameters of large language models (LLMs), potentially leading to more efficient deployment and inference. One classic and prominent path of LLM one-shot pruning is to leverage second-order gradients (i.e., Hessian), represented b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ROSE: Reordered SparseGPT for More Accurate One-Shot Large Language Models Pruning 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **问题背景**：在大型语言模型（LLMs）中，One-shot pruning 方法（如 SparseGPT）通过 Hessian 二阶信息进行权重重建，实现高效剪枝。然而，其采用固定的从左到右（left-to-right）的剪枝顺序，在某些具有**列状模式**（columnar pattern）的层中会导致次优性能。
- **核心问题**：当权重分布呈现明显的列聚集特征时，后期被剪枝的高误差块缺乏足够的剩余参数用于补偿，导致最终重建误差增大。

### 提出了什么新方法或新思路
- **提出 ROSE 方法**：一种基于 SparseGPT 的重排序剪枝框架，旨在优化剪枝顺序以提升精度。
- **核心思想**：
  - **预剪枝估计损失**：先执行一次轻量级预剪枝，识别潜在将被移除的候选权重，并计算每个**列**和**块**的剪枝损失。
  - **两级重排序机制**：
    1. **列内重排序**：在每个 block 内部按列损失降序排列；
    2. **块间重排序**：所有 blocks 按总块损失降序排列。
  - **自适应识别柱状层**：引入 **relative range of block loss** 作为指标，自动判断哪些层具备 columnar pattern 并应用重排序。

### 相比现有方法的优势
- **更合理的剪枝顺序**：优先剪枝高风险（高损失）的权重，使其能利用更多未剪枝参数进行误差补偿。
- **兼容性强**：完全兼容 SparseGPT 的补偿机制，无需额外训练。
- **通用扩展性**：可自然推广至 semi-structured pruning（如 2:4、4:8 模式）和其他压缩任务（如量化联合压缩）。
- **效率代价极低**：仅增加少量计算开销（+0.4 分钟 @ LLaMA2-7B），不影响推理加速效果。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准数据集（Calibration Data）**：
  - C4 数据集的第一个 shard，共 128 个样本，每样本 2048 tokens。
- **评估数据集**：
  - **Perplexity 评估**：WikiText-2-raw
  - **零样本任务评估（zero-shot evaluation）**：BoolQ, WinoGrande, PIQA, OpenBookQA, HellaSwag, ARC-Easy, ARC-Challenge
  - 所有任务使用 `lm-eval-harness` 框架统一评测。

### 实验设置和评估指标
- **模型范围**：
  - LLaMA2 系列：7B, 13B, 70B
  - LLaMA3-8B
  - Mistral-7B
- **剪枝率（Sparsity Rate）**：60% ~ 90%
- **评估指标**：
  - 主要指标：**Perplexity ↓**（越低越好）
  - 辅助指标：**Zero-shot Accuracy ↑**（平均准确率）
  - 其他分析指标：Reconstruction Error, Latency, Speedup
- **实现细节**：
  - Block size = 128（unstructured）
  - Semi-structured 设置：2:4（blocksize=4）、4:8（blocksize=8）
  - ROSE 判定阈值：relative range threshold = 0.5

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Magnitude** | Unstructured | 基于权重大小直接剪枝 |
| **Wanda** | Unstructured | 权重 × 激活范数为重要性评分 |
| **DSnoT** | Unstructured | 动态掩码 + 无训练微调 |
| **OATS** | Unstructured | 稀疏 + 低秩分解保留 outlier |
| **SparseGPT** | Unstructured | Hessian 补偿，one-shot 范式标杆 |
| **ROSE (ours)** | Unstructured/Semi-structured | 在 SparseGPT 上改进剪枝顺序 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Perplexity 对比（WikiText，部分代表性结果）

| Model & Sparsity | SparseGPT | ROSE (ours) | 改进幅度 |
|------------------|-----------|-------------|---------|
| LLaMA3-8B @ 80% | 203.45 | **172.14** | ↓ 15.4% |
| Mistral-7B @ 80% | 78.69 | **78.96** | ≈持平（略优） |
| LLaMA2-7B @ 70% | 27.68 | **26.38** | ↓ 4.7% |
| LLaMA2-70B @ 70% | 9.34 | **9.29** | ↓ 0.5% |

> 💡 ROSE 在多数情况下显著优于 SparseGPT，尤其在高压缩率下优势更明显。

#### ✅ Zero-shot 准确率对比（LLaMA2-7B @ 70%）

| Task | SparseGPT | ROSE (ours) | 提升 |
|------|-----------|-------------|------|
| ARC-e | 40.19 | **41.71** | +1.52 |
| ARC-c | 23.46 | **25.26** | +1.80 |
| Avg | 45.43 | **46.43** | +1.00 |

> 📌 ROSE 在多个任务上表现更好，尤其在复杂推理任务（ARC）上有明显增益。

#### ✅ Semi-structured Pruning 结果（2:4 pattern）

| Model | SparseGPT | ROSE (ours) |
|-------|-----------|-------------|
| LLaMA2-7B | 11.00 | **10.73** |
| LLaMA3-8B | 16.33 | **15.84** |

> ✅ ROSE 同样适用于 semi-structured setting，持续取得更低 perplexity。

---

### 与基线方法的对比结果
- **全面超越非 Hessian 方法**：
  - Wanda、Magnitude、DSnoT 在高 sparsity 下 perplexity 急剧上升，性能崩溃。
- **优于 OATS**：
  - 尽管 OATS 设计保留 outlier，但在大多数设置下仍不如 ROSE。
- **优于原始 SparseGPT**：
  - 在几乎所有模型和 sparsity 设置下，ROSE 实现更低的 perplexity 和更高的 zero-shot accuracy。

---

### 消融实验结果（Ablation Study）

#### 🔍 Blocksize 影响（LLaMA2-7B @ 70%）
- ROSE 在不同 blocksize 下均稳定优于 SparseGPT，表明方法鲁棒性强。

#### 🔍 Calibration Data 规模影响
- 随着 calibration samples 数量或序列长度增加，ROSE 始终保持对 SparseGPT 的性能领先。

#### 🔍 重排序有效性验证（Figure 4）
- **Descending Order（ROSE）**：显著降低 reconstruction error。
- **Ascending Order（反向）**：error 明显升高，证明“先剪大错”策略正确。

#### 🔍 运行时间对比（Table 6）
| Method | LLaMA2-7B 时间（分钟） |
|--------|--------------------------|
| SparseGPT | 4.76 |
| ROSE | **5.15** |
| OATS | 572 |

> ⚠️ ROSE 仅比 SparseGPT 多约 0.4 分钟，远快于 OATS，适合大规模部署。

#### 🔍 推理延迟测试（Table 7）
| Method | Latency (ms) | Speedup |
|--------|--------------|---------|
| Dense | 1791 | – |
| SparseGPT | 1458 | 1.23× |
| ROSE | **1450** | **1.24×** |

> ✅ ROSE 不改变稀疏模式，推理速度与 SparseGPT 几乎一致，无额外开销。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **剪枝顺序至关重要**：SparseGPT 固定的 left-to-right 顺序在 columnar 层中并非最优，是影响 one-shot 剪枝精度的关键因素。
2. **columnar pattern 普遍存在**：主流 LLM 中的 `self_attn.o_proj` 层普遍存在列状权重分布，且这些层对剪枝顺序敏感。
3. **提前剪高损失块更优**：让潜在剪枝误差大的权重尽早被处理，可以利用更多的剩余参数进行有效补偿。
4. **ROSE 可自适应识别并优化**：通过 relative range of block loss 自动检测柱状层，并实施两级重排序，显著提升性能。

### 方法的局限性
- **依赖预剪枝估计**：虽然预剪枝简单有效，但仍是启发式方法，不能保证完全精确预测最终剪枝位置。
- **仅适用于 layer-wise 剪枝框架**：目前集成于 SparseGPT 架构，难以直接迁移到其他非 Hessian 补偿方法。
- **对非 columnar 层收益有限**：对于权重分布均匀的层（如 mlp.proj），重排序带来的提升较小。

### 未来工作方向
- **拓展至其他压缩技术**：探索 ROSE 思路在量化（Quantization）、蒸馏（Distillation）中的应用。
- **动态剪枝顺序学习**：设计可学习的剪枝调度器，替代手工定义的损失度量。
- **跨层协同重排序**：考虑层间依赖关系，进行全局而非局部的剪枝顺序优化。
- **支持更多硬件友好模式**：适配 NVIDIA、TPU 等平台特有的稀疏结构（如 1:N sparsity）。

---

> ✅ **总结一句话**：ROSE 揭示了 **pruning order** 是 one-shot LLM pruning 中一个被忽视但至关重要的因素，通过简单的两阶段重排序策略，在几乎不增加成本的前提下，实现了比 SparseGPT 更高的剪枝精度，为后续研究提供了新的视角。

</details>

---

### 8. [Implicit Style Conditioning: A Structured Style-Rewrite Framework for Low-Resource Character Modeling](https://arxiv.org/abs/2603.05933)

**Authors**: Chanhui Zhu  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.05933v1  

#### Abstract
Large Language Models (LLMs) have demonstrated impressive capabilities in role-playing (RP); however, small Language Models (SLMs) with highly stylized personas remains a challenge due to data scarcity and the complexity of style disentanglement. Standard Supervised Fine-Tuning (SFT) often captures ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Implicit Style Conditioning: A Structured Style-Rewrite Framework for Low-Resource Character Modeling*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前在**低资源角色建模**（low-resource character modeling）中存在两大挑战：
- **角色风格高维复杂**：角色风格涉及词汇偏好、句法模式、语用倾向等多维度特征，难以通过单一隐向量有效捕捉。
- **数据稀缺**：大多数虚构角色仅有少量语料（如 N=25），标准监督微调（SFT）容易导致“出戏”（Out-Of-Character, OOC）生成，无法稳定复现角色特有表达。

此外，现有方法（如 prompt-based 或 RAG）在风格控制上不稳定，且依赖大量标注数据或大型模型，难以部署于消费级硬件。

---

### 提出的新方法与思路
作者提出一个 **Structured Style-Rewrite Framework**，其核心创新包括：

#### （1）**结构化风格表示**（Structured Style Representation）
将角色风格解耦为三个可解释维度：
- **Lexical**：基于 TF-PMI 提取角色特有关键词（如“喵”、“沐沐”）；
- **Syntactic**：基于 PCFG 规则统计句法偏好，压缩为 13 维稠密向量；
- **Pragmatic**：通过 Context-Aware Style Refiner 预测多标签语用风格分布（如“cute, energetic, playful”）。

最终构建统一的结构化风格向量 $ S = [L, S, P] $，实现细粒度、可解释的风格控制。

#### （2）**上下文感知的风格精炼器**（Context-Aware Style Refiner）
针对小样本下伪标签噪声问题，设计轻量 MLP 模型，结合聚类原型与上下文嵌入，修正初始风格标签，提升监督信号可靠性。

#### （3）**重写式数据增强**（Rewrite-Based Data Augmentation）
利用上述结构化风格向量，构建合成平行语料对（Neutral, Stylized），用于训练可控生成模型。该策略显著提升了数据效率。

#### （4）**隐式风格条件机制**（Implicit Style Conditioning via CoT Distillation）
引入 Chain-of-Thought（CoT）监督，在训练阶段显式引导模型推理如何应用风格约束。但在推理时移除 CoT 输出，使风格逻辑内化至模型参数中，实现**无显式推理标记的高效推理**。

---

### 相比现有方法的优势
| 方法 | 局限性 | 本文优势 |
|------|--------|----------|
| Vanilla SFT | 忽略风格细节，易产生语义漂移 | 显式建模多维风格，保持语义一致性 |
| Prompting / RAG | 风格不稳定，输出方差大 | 结构化控制，生成更一致 |
| Holistic Embedding | 黑箱表示，不可解释 | 可分解、可干预的风格向量 |
| 大模型部署 | 资源消耗高 | 在 Qwen-1.7B 上即可实现高性能 |

> ✅ **核心优势**：以极小模型（1.7B）实现媲美甚至超越更大模型（如 4B SFT 或 GLM-4.7）的风格一致性与语义保真度，适合低资源场景下的轻量化部署。

---

## 2. 核心实验方法和设置

### 数据集
- **主训练数据**：基于 `ChatHaruhi-Expand-118K` 构造的合成平行语料，共 **5,786 对**（Neutral, Stylized），经重采样后达 **6,997 对**。
- **测试集**（Hybrid Test Set）：
  - **In-domain Daily Chat**：42 条来自 LCCC 的日常对话中性句子；
  - **Cross-domain Stress Test**：108 条来自未见角色（如雷电将军、凉宫春日）的去风格化句子，用于检验泛化能力。
- **零样本案例**：Frieren（N=25）极端冷启动设定。

---

### 实验设置
- **骨干模型**：Qwen3-1.7B + LoRA 微调。
- **训练目标**：
  $$
  \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{lm}} + \lambda_{\text{recon}}\mathcal{L}_{\text{recon}} + \lambda_{\text{style}}\mathcal{L}_{\text{style}}
  $$
  包含语言建模损失、句法重建损失和语用分类损失。
- **推理模式**：评估“推理专用版”（Model v2 Inference-only），即关闭 CoT 输出，验证隐式风格内化效果。

---

### 评估指标
| 指标 | 定义 |
|------|------|
| **Semantic Score** | 生成句与原句的 BGE-large-zh-v1.5 余弦相似度 |
| **Style Score (Raw)** | 与目标角色风格中心的 RoBERTa 判别器得分 |
| **Valid Style Score** | $ S_{\text{raw}} \times \mathbb{I}(Semantic > 0.75) $，惩罚语义失真的风格匹配 |
| **H-Score** | Semantic 和 Style 的调和平均数 |
| **LLM-as-a-Judge** | 使用 DeepSeek-V3 对生成质量进行三维度评分（1–5分）：<br>• Semantic Faithfulness<br>• Style Logic<br>• Idiolect Naturalness |
| **人工评估** | 四位熟悉动漫文化的标注者盲评 40 条输出，使用 Likert 5 分制 |

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Baseline A (RAG+Few-shot)** | 检索相似风格语句作为 few-shot 示例 |
| **Baseline B (Per-Character SFT)** | 每角色单独微调，上限参考 |
| **Baseline C (Vanilla SFT)** | 同数据集但无 CoT 与辅助损失，使用更强的 Qwen3-4B 模型 |
| **Baseline D (Strong LLM Prompting)** | 使用 GLM-4.7 + 2-shot + 显式风格关键词提示 |

---

## 3. 主要实验结果和性能指标

### 自动评估结果（Hybrid Test Set）
| Model | Semantic | Style (Raw) | H-Score | Valid Style |
|-------|----------|-------------|---------|--------------|
| **Model v2 (Inference-only)** | **0.88** | 0.63 | **0.69** | **0.57** |
| Baseline C (4B SFT) | 0.71 | **0.76** | 0.71 | 0.36 |
| Baseline D (Prompting) | 0.77 | 0.87 | 0.79 | 0.52 |
| Baseline A (RAG) | 0.51 | 0.88 | 0.63 | 0.10 |

> 🔍 **关键发现**：
> - 尽管 Baseline A/D 的 Raw Style 更高，但其 Valid Style 极低，说明它们通过复制记忆片段获得表面风格，牺牲了语义一致性。
> - **本文方法在 Semantic 上领先明显（0.88 vs ≤0.77）**，同时维持合理的 Valid Style（0.57），实现了真正的“忠实风格改写”。

---

### LLM-as-a-Judge 评估（1–5 分）
| Model | Semantic | Style Logic | Naturalness |
|-------|----------|-------------|-------------|
| **Model v2 (Inference-only)** | **4.29** | **2.86** | **3.00** |
| Baseline D (GLM-4.7) | 4.40 | 3.89 | 4.03 |
| Baseline C | 3.12 | 2.51 | 2.60 |

> ⚠️ 注意：虽然 Baseline D 在部分维度得分更高，但分析表明这是由于其“创造性偏差”（creativity bias）——评委偏好夸张表达，即使伴随轻微语义幻觉。

---

### 人工评估结果（1–5 分）
| Model | Semantic | Style Intensity | Overall |
|-------|----------|------------------|---------|
| **Model v2 (Inference-only)** | **4.24** | 3.51 | 3.47 |
| Baseline D | 4.03 | **4.40** | **3.88** |
| Baseline C | 3.47 | 3.60 | 3.18 |

> 📌 **解读**：人工评估中 Baseline D 总体得分最高，但本文方法在**语义保真度上最优**，适用于需严格忠于原文的任务。

---

### 消融实验（Ablation Study）
| Model Variant | Semantic | Style | Valid Style |
|---------------|----------|--------|--------------|
| Full Model v2 | 0.8387 | 0.5875 | **0.4385** |
| w/o Lexical | 0.8471 | 0.5312 | 0.4184 |
| w/o Pragmatic | 0.8344 | 0.5569 | 0.4135 |
| w/o Syntactic | 0.8356 | 0.5805 | 0.4253 |

> 🔍 发现：
> - 移除 **Lexical** 导致 Raw Style 下降最多 → 表层风格最依赖关键词；
> - 移除 **Pragmatic** 对 Valid Style 影响最大 → 语用控制是保持整体风格连贯性的关键；
> - 移除 **Syntactic** 影响最小 → 存在“风格自动补全”现象，归因于 CoT 蒸馏带来的参数内化。

---

## 4. 关键结论和发现

### 主要结论
1. ✅ **结构化解耦 + 显式监督 = 更强的风格控制能力**  
   将风格分解为 lexical/syntactic/pragmatic 并分别建模，显著优于整体嵌入或纯 prompt 方法。

2. ✅ **CoT 蒸馏实现隐式风格内化**  
   即使在推理时不输出 CoT，模型仍能高质量生成，证明 CoT 作为一种**归纳偏置**（inductive bias），成功将推理过程压缩进参数空间。

3. ✅ **小模型也能胜过大模型**  
   Qwen-1.7B 在 Valid Style 和 Semantic 上均优于 Qwen-4B Vanilla SFT，验证了本框架的数据效率与控制有效性。

4. ✅ **支持极端冷启动（zero-shot）泛化**  
   在 Frieren（N=25）上成功提取并注入独特风格特征（如“只是…而已”句式、高修饰密度），未出现背景幻觉。

---

### 方法局限性
| 局限 | 说明 |
|------|------|
| **长对话一致性不足** | 当前风格向量静态提取，未考虑对话历史动态演化，影响多轮交互表现。 |
| **深层语用现象捕捉有限** | 如讽刺、双关、文化隐喻等 subtlety 难以通过统计特征完全建模。 |
| **PCFG 维度固定为 13** | 缺乏自适应调整机制，可能不适用于风格差异极大的角色。 |
| **源句非完全中性** | 社交媒体语料自带“网民风格”，影响风格纯净度。 |
| **评估主观性强** | 人类评价一致性仅为 fair level（Krippendorff’s α ≈ 0.33–0.42）。 |

---

### 未来工作方向
- 引入**动态风格更新机制**，结合对话历史实时调整风格向量；
- 探索**自适应 PCFG 特征选择**，根据不同角色分布学习最优维度映射；
- 开发更客观的**风格幻觉检测指标**，减少 LLM judge 的偏见；
- 扩展至**非虚构角色建模**（如历史人物、专业人士），提升领域通用性；
- 研究**显式 vs 隐式推理的权衡边界**，明确何时需要保留 CoT 输出。

---

## 总结
> 💡 **一句话总结**：  
> 本文提出一种**结构化风格重写框架**，通过**解耦风格维度 + 重写增强 + CoT 蒸馏**，在极低资源条件下实现了高保真、高一致性的角色建模，使得小型语言模型也能在消费级硬件上完成高质量的角色扮演任务，为轻量化、可控化对话系统提供了新范式。

</details>

---

### 9. [SPOT: Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning in Large Language Models](https://arxiv.org/abs/2603.06222)

**Authors**: Yunlong Chu, Minglai Shao, Yuhang Liu, Bing Hao, Yumeng Lin, Jialu Wang, Ruijie Wang  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.06222v1  

#### Abstract
Explicit Chain-of-Thought improves the reasoning performance of large language models but often incurs high inference cost due to verbose token-level traces. While recent approaches reduce this overhead via concise prompting or step pruning, they largely truncate what the model says rather than inte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SPOT: Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning in Large Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

当前主流的 **Chain-of-Thought (CoT)** 推理虽然提升了大语言模型（LLM）的推理能力，但其显式的、逐 token 的推理轨迹导致**推理成本高昂**，表现为“过度思考”（overthinking），即生成大量冗余步骤而未显著提升准确率。

现有压缩方法存在两大缺陷：
- **刚性对齐（rigid point-to-point alignment）**：将一个 `<pause>` token 强制对齐到某一步推理的终点表示，无法捕捉整个推理段落（span）的密集语义。
- **缺乏可解释性（lack of interpretability）**：隐状态通过无约束优化得到，难以解码为人类可读的“思想”，不利于审计与调试。

### 🚀 提出的新方法：SPOT（Span-level Pause Of Thought）

SPOT 是一种灵活的框架，将显式 CoT 压缩为紧凑的隐式 `<pause>` token，实现高效且可解释的 **latent reasoning**，核心创新如下：

#### （1）**Span-level Semantic Alignment（跨段语义对齐）**
- 不再采用“点对点”对齐，而是使用 **Sinkhorn 正则化最优传输（Sinkhorn-regularized Optimal Transport, OT）** 目标函数，实现一个 `<pause>` token 与整个推理段落（variable-length reasoning span）之间的软匹配。
- 该机制能更鲁棒地捕捉段落级语义，克服传统方法因固定长度或仅对齐末态而导致的信息丢失。

#### （2）**Frozen-Head Decoding Constraint（冻结头解码约束）**
- 在训练过程中保持预训练的 **Language Modeling (LM) Head 冻结**，并将 `<pause>` 隐状态投影回词表空间。
- 使得 `<pause>` 状态可以直接被解码为高概率 token 分布（如 top-K keywords），从而提供**可读的语义解释**，增强透明性和可审计性。

#### （3）**两阶段训练 + 外部插入控制**
- **Stage I：OT 对齐训练** —— 利用 SpanDrop 数据进行 span-level 对齐学习。
- **Stage II：Rejection-Sampled Fine-Tuning (RFT)** —— 通过筛选正确且更短的输出微调模型，提升对外部 `<pause>` 插入模式的鲁棒性。
- 支持在推理时由用户**外部控制 `<pause>` 的插入频率和位置**，实现准确率与生成长度的灵活权衡。

### 🔍 相比现有方法的优势

| 维度 | SPOT | 传统方法 |
|------|------|--------|
| **对齐方式** | 跨段软对齐（soft span-level） | 点对点硬对齐（point-to-point） |
| **可解释性** | 高（Frozen-Head 解码关键词） | 低（隐向量难解释） |
| **灵活性** | 高（无需固定模板，支持外部控制） | 低（常依赖固定交错模式） |
| **效率 vs 准确率平衡** | 更优（大幅减长同时提准） | 易失衡（剪枝过多导致欠推理） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 类型 | 特点 |
|-------|------|------|
| **GSM8K** | 小学数学应用题 | 数值答案，标准 CoT 测试基准 |
| **MATH500** | 竞赛级数学题 | 更复杂多步推导，挑战更强 |
| **AIME 2024 / 2025** | 美国数学邀请赛题 | 极高难度，整数答案 [0,999] |
| **GPQA-Diamond** | 科学领域问答（OOD） | 研究生级别选择题，检验跨域泛化能力 |

> 所有方法均基于 **DeepSeek-R1-Distill-Qwen-7B** 模型骨架。

### ⚙️ 实验设置与评估指标

| 设置项 | 说明 |
|-------|------|
| **训练数据** | GSM8K 训练集 |
| **推理控制参数 `N`** | 每隔 `N` 个显式推理段落插入一个 `<pause>`（数学任务用 `N=3`，GPQA 用 `N=1`） |
| **解码策略** | 温度 0.6，top-p 0.95，最大生成长度 16,384 tokens |
| **重复次数** | 每样本运行 10 次取平均 |
| **评估指标** | - **Pass@1 Accuracy (Acc)**：最终答案正确率<br>- **#L（Output Length）**：总生成 token 数量 |

### 🆚 基线方法对比

分为两类：

#### （1）**显式推理压缩方法（Explicit Trace Control）**
- **CCoT**, **ConciseHint**, **Step Entropy**, **L1-Max**, **DEER**  
→ 通过提示或熵信号剪枝冗余步骤，但仍受限于 autoregressive 解码长度。

#### （2）**隐式/潜变量推理方法（Implicit/Latent Reasoning）**
- **COCONUT**, **CODI**, **LightThinker**, **Latent-SFT**  
→ 将部分推理移至隐空间，但多采用端点对齐，缺乏可解释性。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 方法 | 平均 Acc ↑ | 平均 #L ↓ | 相对 Vanilla 的长度减少 |
|------|-----------|----------|------------------------|
| **Vanilla** | — | — | — |
| **SPOT-stage2** | **+2.3%** | **-37.5%** | 最佳综合表现 |
| DEER | +0.0% | -24.0% | 压缩有限 |
| ConciseHint | +0.2% | -22.6% | 效果一般 |
| CODI / COCONUT | -25% ~ -29% | -78% ~ -85% | **严重掉点**，不可靠 |

> ❗SPOT 在显著缩短输出的同时，**反而提升准确率**，打破“压缩必降准”的常规认知。

### 🔍 各数据集上的具体表现（SPOT-stage2 vs Vanilla）

| 数据集 | Acc 提升 | 生成长度减少 |
|--------|---------|-------------|
| **GSM8K** | +3.1% | **-52.1%** |
| **MATH500** | +1.4% | -43.0% |
| **AIME2025** | **+3.3%** | -15.8% |
| **GPQA-Diamond** | **+4.5%** | **-49.3%** |

> ✅ 即使在最难的 AIME 和跨域 GPQA 上也稳定增益，表明 SPOT 泛化能力强。

### 🔧 消融实验结果

#### （1）**对齐目标对比（Table 2）**

| 对齐方式 | GSM8K Acc | AIME2025 Acc | #L 变化 |
|--------|----------|------------|--------|
| **Sinkhorn OT（SPOT）** | **92.72** | **39.33** | -52.1% |
| End_KL（只对齐最后 token） | 87.30 | 21.33 | -8.5% |
| MSE（池化平均回归） | 88.93 | 15.67 | +12.6%（更长！） |

> ❌ 替换为简单对齐方式会导致严重性能下降，验证了 **Sinkhorn OT 的必要性**。

#### （2）**压缩粒度（Spans per `<pause>`）**

| G（每 `<pause>` 压缩 G 个段） | Acc 趋势 | #L 趋势 |
|----------------------------|--------|--------|
| G=1 | 最高 | 适中 |
| G=2~3 | 明显下降 | 进一步降低 |

> ✅ **G=1（每个段独立压缩）取得最佳 trade-off**，过高压缩损害准确性。

#### （3）**对齐权重 λ 敏感性分析（Table 4）**

| λ | Acc（GSM8K） | #L |
|----|-------------|-----|
| 0.0 | 86.66 | -6.1% |
| 0.6 | 90.00 | -26.3% |
| **1.0** | **92.72** | **-52.1%** |

> ✅ 增大 λ 持续提升性能，说明 span-level alignment 越强越好。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **SPOT 成功实现了高效且可解释的 latent reasoning**：
   - 通过 **span-level alignment** 和 **frozen-head decoding**，既压缩了推理轨迹，又保留了语义完整性与可读性。

2. **显著优于现有压缩方法**：
   - 相比显式剪枝方法，SPOT 压缩更彻底；
   - 相比隐式方法，SPOT 不牺牲甚至提升准确率。

3. **具备良好的可控性与鲁棒性**：
   - 用户可通过调节 `<pause>` 插入密度，在推理长度与精度间灵活权衡（见 Figure 3）。
   - RFT 阶段增强了模型对不同插入模式的适应能力。

4. **可解释性强**：
   - `<pause>` 状态可解码为关键词（如 `"multiply"`, `"add"`），反映其内部计算意图。
   - LLM-as-a-Judge 评测显示 SPOT 的 `<pause>` 边界具有更高的 **pause_utilization** 和 **Joint@4**（达 83.6% vs Vanilla 的 17.2%），说明其确实承载了有意义的推理跳跃。

### ⚠️ 局限性

1. **依赖段落分割规则**：
   - 当前使用 `\n\n` 作为 span 边界，适用于 DeepSeek-R1 风格输出，但在其他格式下可能需调整。
   - 未来可探索 learnable 或 adaptive 分割策略。

2. **极端密集插入可能导致重复行为**：
   - 如 Appendix A.6 所示，当 `<pause>` 插入过于频繁时，模型可能出现重启或重复推理，影响连贯性。

3. **仍需人工设计插入策略**：
   - 虽然支持外部控制，但最优插入模式尚未自动化，依赖经验设定。

### 🔮 未来工作方向

1. **动态自适应 `<pause>` 插入机制**：
   - 结合 difficulty estimation 或 uncertainty signal 自动决定何时插入 `<pause>`。

2. **可学习的 span 划分模块**：
   - 替代启发式分段，让模型自动识别合理的推理单元边界。

3. **扩展至规划与长程决策任务**：
   - 将 SPOT 应用于复杂任务分解、Agent 行为链等场景，进一步验证其通用性。

4. **多模态 latent reasoning**：
   - 探索图像、代码等非文本中间表示的压缩与解释。

---

> 💡 **总结一句话**：  
> **SPOT 通过 span-level 语义对齐与 frozen-head 解码约束，首次实现了高效、准确、可解释的 latent reasoning，在多个推理任务上实现了“更短却更好”的突破性效果。**

</details>

---

### 10. [Provuse: Platform-Side Function Fusion for Performance and Efficiency in FaaS Environments](https://arxiv.org/abs/2603.06170)

**Authors**: Niklas Kowallik, Natalie Carl, Leon P\"ollinger, Wei Wang, Sharan Santhahanam, David Bermbach  
**Category**: cs.DC  
**Published**: 2026-03-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.06170v1  

#### Abstract
Function-as-a-Service (FaaS) platforms provide scalable and cost-efficient execution but suffer from increased latency and resource overheads in complex applications comprising multiple functions, particularly due to double billing when functions call each other. This paper presents Provuse, a trans...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Provuse: Platform-Side Function Fusion for Performance and Efficiency in FaaS Environments*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
- **FaaS（Function-as-a-Service）环境中函数调用开销大**：在由多个函数组成的复杂应用中，频繁的远程函数调用导致显著的**延迟增加**和**资源浪费**。
- **“Double Billing”问题**：由于每个函数调用都独立计费，即使一个函数同步调用另一个函数，平台也会对两个实例同时计费，造成成本冗余。
- 当前优化多依赖开发者干预（如手动合并函数），缺乏透明、自动化的平台级解决方案。

### 🚀 提出的新方法或新思路
- 提出 **Provuse** —— 一种**平台侧（platform-side）、运行时自动执行的 function fusion 机制**。
- 在不改变用户代码的前提下，通过监控函数间的**同步调用行为**，动态地将频繁相互调用的函数**融合为单个执行单元（merged function instance）**。
- 引入两个核心组件：
  - **Function Handler**：拦截并监控函数出口连接，识别同步调用模式。
  - **Merger**：基于容器运行时，合并多个函数的文件系统，构建新的融合函数镜像并部署。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | Provuse |
|------|--------|--------|
| **控制方** | 应用层 / 开发者主导（如注解、配置） | 平台侧全自动，无需开发者参与 |
| **透明性** | 需修改代码或部署逻辑 | 完全透明，保留原有函数接口 |
| **适用范围** | 多为静态分析或预定义规则 | 动态基于运行时调用模式决策 |
| **兼容性** | 可能依赖特定编程模型 | 支持通用 bring-your-own-function-code 模型 |
| **部署灵活性** | 融合需重新打包发布 | 运行时动态合并，支持热更新 |

> 💡 创新本质：将 function fusion 从“开发辅助工具”升级为“平台基础设施能力”，实现**透明化、自动化、运行时驱动的性能优化**。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集与应用
使用两个典型 FaaS 工作负载进行评估：
1. **TREE 应用**  
   - 结构：简单的二叉树调用图，一侧为同步调用链（A→B→D/E），另一侧为异步分支（A→C→F/G）
   - 特点：用于验证 fusion 对不同调用模式的选择性优化能力。

2. **IOT 应用（Fusionize++ IoT）**  
   - 场景：模拟物联网传感器数据分析流程，输入经 `AnalyzeSensor` 后并行检测温度、空气质量、声音等。
   - 调用特征：存在明显的同步调用序列（如 `CheckWorking` → `StoreEvent`），适合 fusion。
   - 图结构见原文 Figure 3。

> ⚠️ 数据来源：复用先前研究 [Schirmer et al., 2024] 中已公开的工作负载设计，确保可比性。

### ⚙️ 实验设置
- **测试环境**：
  - 两台虚拟机（QEMU/KVM）：一台运行 SUT（System Under Test），一台作为 k6 压测客户端。
  - 配置：4 vCPU + 16GB RAM，网络带宽 10 Gbps。
- **平台实现**：
  - 在两种 FaaS 架构上实现 Provuse：
    - **tinyFaaS**：轻量级边缘 FaaS 平台，低开销。
    - **Kubernetes + Knative/OpenFaaS 类架构**：代表主流云原生容器编排平台。
- **请求负载**：
  - 总请求数：10,000 次 HTTP 请求。
  - 请求速率：恒定 5 req/s（与 prior work 一致）。
  - 工具：使用 [k6](https://k6.io/) 发起压测。

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **End-to-end Latency** | 从客户端发起请求到收到响应的时间（重点关注中位数） |
| **RAM Usage** | 平台整体内存占用情况（反映资源效率） |
| **Developer Transparency** | 是否需要修改用户代码或配置（定性评估） |

### 🔁 基线方法对比
- **Vanilla Deployment**：标准 FaaS 部署方式，无任何 fusion 优化，作为基准。
- **Provuse-enabled Deployment**：启用 function fusion 的版本，其余完全相同。

> ❗ 注意：未直接与其他 fusion 方法（如 Fusionize++）对比，而是强调其“platform-side”属性带来的透明性和普适性优势。

---

## 3. 主要实验结果和性能指标

### 📉 关键性能数据汇总

| 应用 | 平台 | 中位延迟（Vanilla） | 中位延迟（Provuse） | 延迟降低 | RAM 使用减少 |
|------|-------|------------------|------------------|----------|--------------|
| IOT | tinyFaaS | 807 ms | 574 ms | **28.9%** | ~57% |
| TREE | tinyFaaS | 452 ms | 350 ms | **22.6%** | ~50% |
| IOT | Kubernetes | 815 ms | 551 ms | **32.4%** | ~57% |
| TREE | Kubernetes | 456 ms | 358 ms | **21.5%** | ~50% |

> ✅ **总体平均提升**：
> - **中位端到端延迟降低 26.33%**
> - **平均 RAM 占用减少 53.57%**

### 🆚 与基线方法的对比结果
- 所有实验配置下，**Provuse 显著优于 Vanilla 部署**。
- 时间序列显示（Figure 5）：随着 fusion 过程完成（merge event），延迟逐步下降并在稳定后维持低位。
- 内存节省来自于减少了并发运行的 function instance 数量，避免重复加载 runtime 和依赖。

### 🔍 消融实验（Ablation Study）
- 文中虽未明确命名“ablation study”，但隐含进行了以下关键变量分析：
  - **调用模式影响**：仅同步调用被触发 fusion，异步路径不受影响 → 表明 fusion 具有选择性。
  - **平台无关性**：在 tinyFaaS 与 Kubernetes 上均取得相似收益 → 证明方法具有良好的可移植性。
  - **语言限制**：当前原型仅支持 Python 函数 → 被视为未来扩展方向而非消融项。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Platform-side function fusion 是可行且高效的**：
   - 能在不改动用户代码的情况下，显著降低延迟和资源消耗。
2. **自动识别同步调用是有效的优化策略**：
   - 同步调用天然存在阻塞等待，合并后可通过 inlining 消除 IPC 开销。
3. **资源利用率大幅提升**：
   - 减少冷启动、共享 runtime、消除双倍 billing，带来约 **53.6% RAM 节省**。
4. **适用于多种 FaaS 架构**：
   - 成功集成于 tinyFaaS 和 Kubernetes，表明其可推广至边缘与云端场景。

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **信任域假设** | 要求被融合函数属于同一 trust domain，否则会削弱安全隔离性。 |
| **不适用于高度异步应用** | 若应用以非阻塞调用为主（如事件驱动流处理），fusion 效果有限。 |
| **当前仅支持 BYO-function-code 模型** | 不支持 “bring-your-own-container”（BYOC）模式，限制第三方容器使用。 |
| **仅支持 Python** | 原型实现语言受限，尚未覆盖 Node.js、Java 等主流 FaaS 语言。 |
| **融合开销存在** | 容器重建与部署有一定代价，需足够多后续调用来摊销成本。 |

### 🔮 未来工作方向
1. **支持更多语言和运行时**：扩展至 Java、Node.js、Go 等常见 FaaS runtime。
2. **混合部署模型支持**：结合 BYO-function 与 BYOC 模型，允许更灵活的优化。
3. **融合策略增强**：
   - 引入机器学习预测调用模式。
   - 支持分层 fusion（如按热度、频率、拓扑聚类）。
4. **集成其他平台优化技术**：
   - 与 pre-warming、peak shaving、live migration 等结合，形成综合优化框架。
5. **安全性增强机制**：
   - 在融合环境下提供细粒度权限控制与故障隔离机制。

---

## ✅ 总结一句话
> **Provuse 通过平台侧自动化的 function fusion，在无需开发者干预的前提下，实现了平均 26.3% 的延迟降低和 53.6% 的 RAM 节省，展示了 infrastructure-level 透明优化在 FaaS 系统中的巨大潜力。**

</details>

---

### 11. [Parallelization Strategies for Dense LLM Deployment: Navigating Through Application-Specific Tradeoffs and Bottlenecks](https://arxiv.org/abs/2603.05692)

**Authors**: Burak Topcu, Musa Oguzhan Cim, Poovaiah Palangappa, Meena Arunachalam, Mahmut Taylan Kandemir  
**Category**: cs.DC  
**Published**: 2026-03-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.05692v1  

#### Abstract
Breakthroughs in the generative AI domain have fueled an explosion of large language model (LLM)-powered applications, whose workloads fundamentally consist of sequences of inferences through transformer architectures. Within this rapidly expanding ecosystem, dense LLMs--those that activate all mode...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Parallelization Strategies for Dense LLM Deployment**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本文聚焦于**密集型大语言模型（Dense LLM）在推理部署中的并行化策略选择难题**。随着 Llama-3.1-70B 和 Llama-3.1-405B 等超大规模模型的兴起，单设备内存已无法容纳完整模型权重与动态增长的 KV Cache，导致必须依赖多 GPU 部署。然而，不同的 **Model Parallelism**（如 TP、PP 及其混合）在 **Latency**（延迟）与 **Throughput**（吞吐量）之间存在显著权衡，且受输入长度、批处理大小、硬件配置等多重因素影响。

现有研究多集中于训练阶段的并行优化，而对**推理场景下应用特定目标（如低延迟 vs 高吞吐）的并行策略系统性分析仍不足**。本文填补了这一空白。

---

### **提出了什么新方法或新思路**
- 构建了一个高保真的 **in-house simulator**，用于模拟不同并行策略下的 LLM 推理行为，并验证其与真实硅片执行的相关性（>86% 准确率）。
- 对 **Tensor Parallelism (TP)**、**Pipeline Parallelism (PP)** 及其 **Hybrid (TP+PP)** 在多种配置下的性能表现进行了全面量化分析。
- 提出了一种 **“可调平衡”框架**：通过控制 TP 和 PP 的 degree，实现对 **Latency-Throughput Interplay** 的精细调控。

---

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **系统性分析** | 现有工作多关注单一策略（如仅 TP 或 PP），本文首次系统比较了多种并行方案在多样化负载下的综合表现。 |
| **贴近实际应用** | 考虑了真实世界的数据分布（长短序列）、batch size 变化、量化精度（FP8/4bit）等因素，更具实用价值。 |
| **揭示瓶颈机制** | 不仅报告性能趋势，还深入剖析了 all-reduce 通信开销、P2P 延迟、KV Cache 容量限制等根本性瓶颈。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 特点 | 应用模型 |
|--------|------|----------|
| **LongAlpaca** | 长上下文输入（平均 ISL ≈ 9092） | Llama-3.1-70B |
| **MLPerf Inference Dataset** | 长文本摘要任务（ISL ≈ 9428, OSL ≈ 684） | Llama-3.1-405B |
| **BBH + GSM8K + HumanEval** | 短序列任务（ISL ≈ 106） | 两类模型均测试 |

> 所有实验基于平均 ISL/OSL 进行模拟，以代表典型负载特征。

---

### **实验设置和评估指标**

#### **模型**
- **Llama-3.1-70B**: 80 层 Transformer，FP8 量化
- **Llama-3.1-405B**: 126 层 Transformer，4-bit 量化

#### **硬件平台（模拟）**
- 单节点内含 8× AMD Instinct **MI325x** 或 **MI355x** GPU
- 全互联拓扑（all-to-all interconnect）
- 支持不同聚合带宽（256–608 GB/s）进行敏感性分析

#### **并行策略**
| 类型 | Degree 范围 |
|------|-------------|
| **TP (Tensor Parallelism)** | 1–8 |
| **PP (Pipeline Parallelism)** | 1–8 |
| **Hybrid TP+PP** | 组合测试（如 TP4_PP2） |

#### **评估指标**
| 指标 | 含义 |
|------|------|
| **TTFT** | Time to First Token，衡量响应延迟 |
| **TPOT** | Time Per Output Token，衡量生成速度 |
| **TPS** | Total Tokens Per Second，系统吞吐量 |
| **Latency-Throughput Tradeoff** | 多维度权衡分析 |

---

### **基线方法对比**
- **NoPar (No Parallelism)**：无并行，作为基准
- **Pure TP**：仅使用 Tensor Parallelism
- **Pure PP**：仅使用 Pipeline Parallelism
- **Hybrid TP+PP**：组合策略
- 与已有系统（如 DeepSpeed-Inference, TensorRT-LLM）的设计原则进行定性对比

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **Llama-3.1-70B 结果（LongAlpaca, BS=256）**
| 配置 | TTFT 加速比 | TPOT 加速比 |
|-------|--------------|---------------|
| **TP8** | 3.61× | 3.01× |
| **TP4** | 1.87× | 1.67× |
| **PP8** | ~1.0× | ~1.0× |

> ✅ **TP 显著降低延迟，PP 几乎无改善**

#### **Llama-3.1-405B 结果（MLPerf, BS=256）**
| 配置 | TTFT 加速比 | TPOT 加速比 |
|-------|--------------|---------------|
| **TP8** | 3.67× | 2.81× |
| **TP4** | 1.90× | 1.62× |
| **TP4_PP2** | 1.89× | 1.61× |

> ⚠️ **PP 引入额外通信延迟，轻微劣化 TP 性能**

#### **吞吐量（TPS）对比（Llama-3.1-405B, MLPerf）**
| 配置 | TPS 提升倍数（vs NoPar） |
|--------|---------------------------|
| **PP8** | **13.8×** |
| **TP8** | < 1× （下降） |
| **TP4_PP2** | 中等提升 |

> ✅ **PP 显著提升吞吐量，尤其在大 batch 下接近饱和**

---

### **与基线方法的对比结果**
| 维度 | TP | PP | Hybrid |
|------|----|----|--------|
| **Latency (TTFT/TPOT)** | ✅ 最优 | ❌ 无益甚至有害 | ✅ 可控（由 TP 主导） |
| **Throughput (TPS)** | ❌ 劣化 | ✅ 最优 | ✅ 可扩展（由 PP 主导） |
| **KV Cache 利用率** | 中等 | ✅ 高（缓解内存压力） | ✅ 更高 |
| **通信开销** | 高（all-reduce） | 低（P2P） | 中等偏高 |

> 💡 **TP 适合低延迟服务，PP 适合高吞吐批处理**

---

### **消融实验结果**
#### **(1) TP 深度对 all-reduce 开销的影响**
- TP8 相比 TP1，TTFT 降低 68%，但 all-reduce 占比稳定在 ~30%
- 增加链路带宽（256→608 GB/s）使 TTFT 下降 34%，说明 **interconnect 是关键瓶颈**

#### **(2) PP 深度对 P2P 通信的影响**
- PP8 场景下，P2P 通信仅占 TTFT 的 **<0.5%**
- 表明 PP 的通信开销极小，非主要瓶颈

#### **(3) Batch Size 影响**
- 小 batch：TP 占优（计算未饱和）
- 大 batch：PP 占优（KV Cache 容量释放潜力）
- 存在一个“饱和点”，超过后 decode 成为 compute-bound，TPS 增长放缓

---

## **4. 关键结论和发现**

### **主要发现**
1. 🔹 **TP 显著优化 Latency**  
   - 通过将 Attention 和 FFN 层分片到多个 GPU，并行加速前向传播。
   - 特别是在 prefill 阶段（compute-bound），效果最明显。
   - 更深的 TP（如 TP8）带来更强的延迟灵活性。

2. 🔹 **PP 显著提升 Throughput**  
   - 通过将 Transformer blocks 分布到 pipeline stages，允许并发处理多个 micro-batches。
   - 缓解 per-GPU 内存压力，支持更大 KV Cache 和 batch size。
   - 在大 batch 场景下吞吐增益可达 **10× 以上**。

3. 🔹 **Hybrid TP+PP 实现可调平衡**  
   - **TP 控制 Latency**，**PP 控制 Throughput Scaling**
   - 可根据 SLA 要求灵活调整 degree，例如：
     - 交互式对话 → 高 TP / 低 PP
     - 批量文档摘要 → 低 TP / 高 PP

4. 🔹 **All-reduce 是 TP 的主要瓶颈**  
   - 尽管聚合带宽提升有助于缓解，但在 multi-node 场景下可能成为主导延迟源。

5. 🔹 **PP 的通信开销远低于 TP**  
   - 因为 P2P 传输次数少（PP_depth - 1），且每次数据量小。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **仅限 intra-node 分析** | 未考虑跨节点（multi-node）部署中的网络延迟与拓扑影响 |
| **依赖模拟器近似** | 虽然校准到 >86% 精度，但仍无法完全替代真实部署测量 |
| **未涵盖 MoE 模型** | 当前分析针对 dense LLM，MoE 的 expert routing 会引入新的通信模式 |
| **静态 batch 假设** | 实际中 dynamic batching 会影响资源利用率和调度效率 |

---

### **未来工作方向**
1. **Multi-node 扩展分析**  
   - 研究 **inter-node bandwidth**、**NCCL/RoCE 协议**、**Dragonfly 等拓扑** 对 TP/PP 性能的影响。

2. **结合 Dynamic Batching 与 Continuous Batching**  
   - 探索如何在 TP/PP 架构下实现高效的请求调度与资源复用。

3. **面向 MoE 模型的 Expert Parallelism 分析**  
   - 如何将 EP 与 TP/PP 结合，在稀疏激活下进一步优化性能。

4. **软硬协同设计建议**  
   - 基于本研究提出对 GPU interconnect（如 XGMI, NVLink）、内存架构、通信算法的改进需求。

5. **自动化并行策略推荐系统**  
   - 构建一个可根据模型规模、输入特征、SLA 要求自动推荐最优 TP/PP 配置的工具。

---

> 📌 **一句话总结**：  
> **Tensor Parallelism 为 Latency 而生，Pipeline Parallelism 为 Throughput 而战；唯有 Hybrid 并行，方能在现实应用中游刃有余地驾驭 Latency-Throughput 权衡。**

</details>

---

### 12. [Edge Intelligence-Driven LegalEdge Contracts for EV Charging Stations: A Fedrated Learning with Deep Q-Networks Approach](https://arxiv.org/abs/2603.06041)

**Authors**: Rahim Rahmani, Arman Chianeh  
**Category**: cs.DC  
**Published**: 2026-03-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.06041v1  

#### Abstract
We introduce LegalEdge, an edge intelligence-driven framework that integrates Federated Learning (FL) and Deep Q-Networks (DQN) to optimize electric vehicle (EV) charging infrastructure. LegalEdge contracts are novel smart contracts deployed on the blockchain to manage dynamic pricing and incentive ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对当前电动汽车（EV）充电系统中存在的以下关键挑战提出了解决方案：
- **缺乏信任与透明度**：传统充电系统依赖中心化第三方（如 eMSP、Clearing House），存在单点故障风险，且用户隐私易受侵犯。
- **低效的能源调度**：无控充电（Uncontrolled Charging）导致电网峰谷叠加，增加配电网络压力和基础设施投资需求。
- **法律与技术脱节**：智能合约（Smart Contract）虽能实现自动化执行，但缺乏法律可读性和可执行性，难以在现实法律体系中被认可。
- **数据隐私与通信开销**：集中式机器学习需要汇聚原始用户数据，带来隐私泄露风险和高通信成本。

### 提出的新方法与新思路
作者提出了 **LegalEdge Contracts** ——一种融合边缘智能（Edge Intelligence）、联邦学习（Federated Learning, FL）、深度强化学习（Deep Q-Networks, DQN）与区块链的新型框架，其核心创新包括：

#### （1）LegalEdge Contract 架构
- **混合合约模型**：将 **Ricardian Contract (RC)** 的人类可读法律条款与 **Smart Contract (SC)** 的自动执行代码相结合，形成“法码同源”的合同结构。
  - RC 提供法律清晰性（Legal Clarity）
  - SC 实现去信任执行（Trustless Execution）
  - 区块链保障不可篡改性与安全性
- **支持动态合规性**：通过 Oracle 和 AI 驱动的法律分析，合约可响应法规变更并自动更新（Dynamic and Adaptive Compliance）。

#### （2）基于 FL + DQN 的分布式优化机制
- 在多个 **Edge Node**（部署于充电站或车辆端）上训练本地 DQN 智能体，用于实时决策最优充电策略（价格、时间、功率等）。
- 利用 **Federated Learning** 协调器聚合各节点的模型参数更新，实现全局策略优化，而无需共享原始数据，保护用户隐私。
- 引入 **Quantization-Aware Training (QAT)** 支持模型压缩（float32 → int8），降低边缘设备推理延迟与通信开销。

#### （3）DFA 形式化建模
- 使用 **Discrete Finite Automata (DFA)** 对 LegalEdge Contract 的生命周期进行形式化建模，定义状态（Drafted, Active, Disputed 等）、事件（Signing Request, Violation Detected 等）和转移函数，确保合约行为确定、可验证、防异常。

### 相比现有方法的优势
| 维度 | 传统方法 | LegalEdge 方案 |
|------|--------|----------------|
| **信任机制** | 中心化第三方中介 | 去中心化区块链 + 智能合约 |
| **隐私保护** | 数据集中存储，易泄露 | 联邦学习，数据不出本地 |
| **法律效力** | 智能合约无法直接作为法律证据 | Ricardian + Smart Contract 双重保障 |
| **实时性** | 云端处理延迟高 | 边缘智能实现实时决策 |
| **适应能力** | 固定规则调度 | DQN 动态学习最优策略 |

---

## 2. 核心实验方法和设置

### 实验环境配置
| 类别 | 配置详情 |
|------|----------|
| **硬件平台** | 3 × Raspberry Pi 5 / 1 × Jetson Nano（模拟 Edge Nodes）<br>1 × Ubuntu Server（作为协调器与区块链节点） |
| **软件栈** |  
- **FL 框架**：Flower + PyTorch  
- **DQN 实现**：Stable-Baselines3 + Custom DQN  
- **区块链**：Ethereum（Ganache 本地测试网） + Solidity + Web3.py  
- **仿真工具**：SimPy（流程模拟）、GridLAB-D（电网建模）  
- **监控**：Prometheus + Grafana, TensorBoard |

### 实验设置与参数
| 参数 | 设置值 |
|------|-------|
| **Number of Clients (Nodes)** | 3–10（含 EV 与 CS） |
| **FL Rounds** | 50–200 |
| **Local Epochs** | 1–5 |
| **Batch Size** | 32–128 |
| **Learning Rate** | 1e-4 ~ 1e-3 |
| **Optimizer** | Adam |
| **Aggregation Algorithm** | FedAvg（默认），支持 FedProx/FedDyn |
| **Privacy Mechanism** | Differential Privacy（Opacus） |
| **Model Compression** | Quantization + Sparsification |

### 评估指标
| 指标类别 | 具体指标 |
|--------|---------|
| **系统效率** | 能源分配吞吐量 vs 决策成本（Efficiency） |
| **学习性能** | 平均 TD Error 下降趋势（Convergence） |
| **区块链性能** | 智能合约调用延迟（Transaction Speed）<br>合约状态一致性（Contract Integrity） |
| **安全与合规** | 抗 Sybil 攻击能力、惩罚触发准确性、GDPR 合规性 |

### 基线方法对比
- **Baseline 1**：传统集中式 FL + DQN（无区块链）
- **Baseline 2**：标准 FL（FedAvg）+ 固定规则调度
- **Baseline 3**：纯区块链智能合约控制（无 AI 学习能力）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Figure 10）
| 指标 | 结果描述 |
|------|----------|
| **系统效率 (Efficiency)** | 从初始 **55%** 提升至 **>90%**（经过约 20 轮 FL 训练后趋于稳定） |
| **学习收敛性 (Convergence)** | TD Error 呈现**指数级衰减**，在第 20 轮左右基本收敛，噪声极小（得益于优先经验回放 PER 与目标网络） |
| **交易速度 (Transaction Speed)** | 平均智能合约调用延迟为 **0.12 秒**（标准差 ±0.01s），适用于实时闭环控制 |
| **合约完整性 (Contract Integrity)** | 在 20+ 次交互中平均得分为 **0.98**，未出现异常中断或逻辑错误 |

### 与基线方法的对比结果（Table 3）
| 特性 | 传统 FL | LegalEdge-FL |
|------|--------|-------------|
| **FL 算法** | FedAvg | DQN + FL (Flower) |
| **智能合约反馈** | 无 | On-policy reward（策略奖励反馈） |
| **区块链延迟** | N/A | <0.15s（平均） |
| **效率扩展性** | ~65% | >90% |
| **收敛动态** | 线性 | 指数型（DQN 加速学习） |

> ✅ **结论**：LegalEdge 在效率、收敛速度、自动化程度方面显著优于传统方法。

### 消融实验结果（隐含于实验设计中）
尽管文中未明确列出消融表，但从实验设计可推断以下关键组件的影响：
- **Experience Replay + Target Network**：显著降低 TD Error 波动，提升学习稳定性。
- **QAT 支持**：使模型可在边缘设备以 int8 推理，内存占用减少约 75%，推理速度提升 2–3×。
- **区块链激励反馈**：通过 `rewardFunction()` 编码经济激励，引导 DQN 快速收敛到高效策略。

---

## 4. 关键结论和发现

### 主要发现
1. **LegalEdge 实现了“法码合一”**：首次将 Ricardian Contract 与 Smart Contract 深度融合，并通过 DFA 形式化建模，确保合约既具法律效力又可自动化执行。
2. **边缘智能 + FL + DQN 显著提升调度效率**：相比传统方法，系统效率提升超过 **35%**，且具备自适应调节能力。
3. **低延迟区块链集成可行**：在 Ganache 测试网上实现了 **<0.15s** 的平均交易延迟，证明区块链可用于近实时控制系统。
4. **隐私与性能兼顾**：联邦学习避免数据外泄，同时 QAT 技术保证了轻量化模型在边缘设备上的高性能运行。

### 方法的局限性
1. **依赖可信 Oracle**：虽然减少了对第三方的信任，但仍需 Oracle 注入电价、电网负载等真实世界数据，存在潜在攻击面。
2. **DQN 对复杂状态空间敏感**：当环境维度升高（如多类型电价、多服务商竞争）时，可能面临维度灾难。
3. **测试仍处于仿真阶段**：所有实验基于模拟环境（Ganache + SimPy），尚未在真实城市电网中部署验证。
4. **Gas 成本未考虑**：在真实 Ethereum 主网上运行可能导致高昂 Gas 费用，影响经济可行性。

### 未来工作方向
1. **动态联邦策略**：研究跨区域、多主体之间的动态联盟形成机制（Dynamic Federation Strategies）。
2. **增强隐私保护**：引入 **Secure Aggregation** 与 **Homomorphic Encryption** 进一步防止模型反演攻击。
3. **自动化法律推理引擎**：开发基于 NLP 的 AI 模块，自动解析新颁布法规并生成合规性更新建议。
4. **实地试点部署**：计划在斯德哥尔摩大学校园内开展小规模现场试验，评估实际性能与用户体验。
5. **整合 V2G 场景**：扩展框架支持双向能量流动（Vehicle-to-Grid），进一步参与电网调频服务。

---

> 📌 **总结一句话**：  
> **LegalEdge** 是一个开创性的框架，它通过 **Edge Intelligence + FL + DQN + Blockchain + Ricardian Contract** 的深度融合，构建了一个**高效、可信、合法、自主**的下一代 EV 充电管理系统，为“AI 与法律共治”的数字基础设施提供了范式参考。

</details>

---

### 13. [Adapter-Augmented Bandits for Online Multi-Constrained Multi-Modal Inference Scheduling](https://arxiv.org/abs/2603.06403)

**Authors**: Xianzhi Zhang, Yue Xu, Yinlin Zhu, Di Wu, Yipeng Zhou, Miao Hu, Guocong Quan  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.06403v1  

#### Abstract
Multi-modal large language model (MLLM) inference scheduling enables strong response quality under practical and heterogeneous budgets, beyond what a homogeneous single-backend setting can offer. Yet online MLLM task scheduling is nontrivial, as requests vary sharply in modality composition and late...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Adapter-Augmented Bandits for Online Multi-Constrained Multi-Modal Inference Scheduling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **多模态大语言模型**（MLLM）在异构执行后端（如本地设备与云端API）上的**在线推理调度问题**，旨在在满足严格资源预算（如延迟、金钱成本）的前提下，最大化响应质量（reward）。该问题面临两大挑战：
- **任务表示困难**：真实请求在模态组合、输入规模和隐含推理难度上高度异质，且系统抖动和网络波动导致执行成本非平稳。
- **在线决策复杂性**：需在低开销下做出不可逆的实时决策，同时保证长期多维预算不被违反。

传统方法依赖手工特征或全模型微调，难以兼顾准确性与效率；而强化学习策略缺乏理论约束保障。

---

### 🚀 提出的新方法：M2-CMAB
作者提出 **M2-CMAB**（Multi-modal Multi-constraint Contextual Multi-Armed Bandit），一个基于上下文多臂赌博机的在线调度框架，包含三个核心组件：

1. **CLS-attentive Predictor（预测器）**
   - 利用冻结主干的 MLLM（如 Qwen3-VL-2B-Instruct），通过引入 `[CLS]` token 并进行 attention pooling 提取紧凑的任务表示。
   - 仅训练轻量级 **adapters** 进行 reward 和 cost 的 action-specific 预测，实现高效更新与稳定表征。

2. **Primal-Dual Constrainer（约束控制器）**
   - 引入在线拉格朗日乘子（Lagrange multipliers）维护机制，采用 **Online Mirror Descent**（OMD）动态调整 dual variables。
   - 将长期多维预算约束解耦为每轮可优化的目标函数，有效控制不可逆资源消耗。

3. **Two-Phase Scheduler（调度器）**
   - 包含初始探索阶段（initial phase）与探索-利用阶段（exploration-exploitation phase）。
   - 在第二阶段结合预测 reward 与 penalized cost 构建 action score，使用带温度采样的策略平衡探索与利用。

---

### 🔍 相比现有方法的优势
| 方面 | M2-CMAB 的优势 |
|------|----------------|
| **表示能力** | 利用 MLLM 自身生成语义一致的任务 embedding，优于手工统计特征（如 token count）或静态 profile。 |
| **计算效率** | 冻结主干 + 轻量 adapter 更新，避免重复 fine-tuning 开销，适合低延迟调度路径。 |
| **理论保障** | 提供在多维背包约束下的 **regret 上界保证**，确保长期约束满足性。 |
| **适应性** | 动态调整 dual penalties 应对时变成本分布，优于固定阈值或贪婪策略。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
构建了一个混合多模态基准测试集，包含六个数据集：
- **Individual Datasets**:
  - `InfoVQA`: 图像信息问答
  - `GSM8K`: 数学推理
  - `SimpleVQA`: 多模态事实性评估
  - `CoQA`: 对话式问答
  - `AI2D`: 基于图表的视觉推理
- **Composite Dataset (COMPOSITE)**: 将上述五者合并形成大规模异构任务流，用于综合评估。

> 所有数据集均映射到统一的五级奖励尺度 `[1,5]`，提升跨任务可比性。

---

### ⚙️ 实验设置
- **Backends（5个）**:
  - **Local**: Qwen3-VL-2B-Instruct（设备端部署）
  - **Cloud APIs**: GPT-5-nano, Qwen3-VL-32B-Instruct, Qwen3-VL-30B-A3B-Instruct, GLM-4.6V-Thinking
- **Budget Regimes（三种）**:
  - **Restricted**: 总预算 = 最小聚合成本
  - **Normal**: 第二小聚合成本
  - **Generous**: 中位数聚合成本
- **评估指标**:
  - 主要指标：**平均 inference reward**（每完成任务的平均质量得分）
  - 辅助分析：regret bound、ablation study、超参数敏感性

---

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **Random** | 随机选择 backend |
| **Latency-first** | 贪婪选择预测延迟最低的 backend |
| **Money-first** | 贪婪选择预测金钱成本最低的 backend |
| **BGT-planner** | 当前最优的 CMAB-based 预算分配框架 |
| **Threshold-based** | 基于效用/成本比值决策，多成本取平均 |
| **Optimal (Oracle)** | 使用每轮真实 reward/cost 的理想上限（非现实可行） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Figure 3）

在 **COMPOSITE** 数据集上，M2-CMAB 相比次优基线（BGT-planner）的提升如下：

| Budget Regime | Reward Improvement |
|---------------|--------------------|
| Restricted     | **+6.79%**         |
| Normal         | **+13.08%**        |
| Generous       | **+14.18%**        |

> ✅ M2-CMAB 在所有预算条件下均显著优于现有方法，并最接近 Oracle 上限（差距 < 1.2%）。

---

### 🔬 消融实验结果（Ablation Study，Figure 5）
移除任一组件均导致性能下降，验证各模块必要性：
- **w/o Reward Adapter**: 性能下降最严重 → 表明准确 reward 预测是调度核心驱动力。
- **w/o Latency/Money Adapter**: 下降较缓 → 成本维度间存在部分补偿效应，但仍不可或缺。
- 结论：**reward adapter > money adapter ≈ latency adapter** 的重要性排序。

---

### 🎯 超参数敏感性分析（Figure 4）
- 初始阶段占比（Initial phase ratio）从 2.5% 增至 10%，性能略有下降，尤其在受限预算下。
- 原因：过多探索减少可用于 exploitation 的轮次，体现“探索-利用”权衡。
- 推荐设置：适度初始探索（如 2.5%-5%），避免过度牺牲主阶段。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MLLM 自身可作为强大任务编码器**：冻结主干 + adapter 微调能提取既保语义又利于调度的 compact 表示。
2. **Primal-dual 控制机制有效解耦长期约束**：通过 OMD 动态调节 dual variables，实现在线预算管理。
3. **M2-CMAB 实现高性能与强鲁棒性**：在多种预算和任务分布下持续领先，逼近 Oracle 性能。
4. **理论与实践结合良好**：提出的 regret bound 为算法提供了形式化保障，且实验验证其有效性。

---

### ⚠️ 局限性
1. **Dual Radius Estimation 依赖初始探索**：若 To 设置不当，可能影响 dual feasible set 准确性。
2. **Adapters 仍需离线准备**：虽然在线更新轻量，但初始 adapter 训练需历史轨迹支持。
3. **未考虑任务间依赖或批处理优化**：当前为单任务逐轮调度，未拓展至 batch-level 协同优化。
4. **Regret Guarantee 依赖 Estimator Performance**：理论 bound 中包含 Reg’(T)，目前尚无针对 MLLM-based predictor 的 sublinear regret 证明。

---

### 🔮 未来工作方向
1. **建立 MLLM-based Estimator 的在线 regret 理论**：将神经估计器纳入 bandit regret 分析框架。
2. **更细粒度的任务与 backend 表示**：探索 token-level 或 layer-wise 的 cost modeling。
3. **扩展至动态 action space**：支持模型版本更新或新 backend 加入的场景。
4. **隐私保护与联邦调度机制**：在边缘-云协同中引入差分隐私等安全机制。

---

> 💡 **总结一句话**：  
> M2-CMAB 通过 **adapter-augmented representation + primal-dual constraint control + two-phase bandit scheduling**，实现了高效、稳健且具理论保障的 MLLM 在线推理调度，在异构多模态环境下显著超越现有方法。

</details>

---

### 14. [Sparse Crosscoders for diffing MoEs and Dense models](https://arxiv.org/abs/2603.05805)

**Authors**: Marmik Chaudhari, Nishkal Hundia, Idhant Gulati  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.05805v1  

#### Abstract
Mixture of Experts (MoE) achieve parameter-efficient scaling through sparse expert routing, yet their internal representations remain poorly understood compared to dense models. We present a systematic comparison of MoE and dense model internals using crosscoders, a variant of sparse autoencoders, t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Sparse Crosscoders for diffing MoEs and Dense models*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于 **Mixture of Experts (MoE)** 和 **Dense 模型** 在内部表示机制上的差异这一尚未被充分研究的问题。尽管 MoE 因其参数高效扩展能力在大规模语言模型中广泛应用（如 DeepSeek-V3、Switch Transformer），但其内部特征组织方式远不如 Dense 模型清晰。现有可解释性工具（如稀疏自编码器）主要针对 Dense 模型设计，难以直接用于比较架构迥异的 MoE 与 Dense 模型。

### 提出了什么新方法或新思路
作者提出使用 **Crosscoders** ——一种改进版的 **Sparse Autoencoder (SAE)**——来系统性地对比 MoE 与 Dense 模型的激活空间。具体创新包括：
- 采用 **BatchTopK Crosscoder** 变体，通过硬稀疏约束（hard sparsity constraint）提升特征可解释性。
- 引入 **显式共享特征机制**（explicitly designated shared features），将部分特征强制设为两个模型共用，并施加更轻的稀疏惩罚（`λ_s / λ_f ≈ 0.7`），从而缓解标准 Crosscoder 对“共享”结构的高估问题。
- 利用 Crosscoder 学习到的 **decoder weights 范数差异 △norm** 来量化特征是 MoE 特有、Dense 特有还是共享。

### 相比现有方法的优势
- **超越传统 Fine-tuning 分析**：现有 Crosscoder 多用于同一模型及其微调版本之间的比较（base vs. fine-tuned），而本工作首次将其成功应用于 **不同架构**（MoE vs. Dense）的对比分析。
- **更高的重建精度与可解释性**：结合 BatchTopK 和显式共享机制后，Crosscoder 达到了约 **87% 的 fractional variance explained**，显著优于标准 Crosscoder 在此任务上的表现。
- **揭示深层结构差异**：不仅能识别共享/特有特征，还能分析这些特征的密度分布和语义特性，提供对模型内部工作机制的新洞见。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
训练原始 Dense 和 MoE 模型的数据集包含约 **10 亿 tokens**，由三个领域等比例组成：
- **Arxiv**（科学文本）来自 RedPajama 数据集
- **Code** 来自 StarCoder
- **English Stories** 来自 SimpleStories

### 实验设置和评估指标
#### 模型训练
- 训练两个 **5-layer** 模型：一个 Dense 模型和一个 MoE 模型，二者具有相同的 **active parameters** 数量以确保公平比较。
- 均使用标准 **Cross Entropy Loss** 进行训练，MoE 额外加入 **Switch load balancing loss**。
- 各自独立从头训练 **2 epochs**。

#### Crosscoder 训练
- 在两个模型的 **第三层输出激活** 上训练 Crosscoder。
- 使用 **BatchTopK Crosscoder** 架构，显式指定一部分特征为共享（shared features），其余为独占（exclusive）。
- 优化目标为重建损失 + 分级稀疏正则项（`λ_s` for shared, `λ_f` for exclusive）。

#### 评估指标
- **Fractional Variance Explained (FVE)**：衡量 Crosscoder 对原始激活的重建能力。
- **Feature Classification via △norm**：
  $$
  \Delta_{\text{norm}}(i) = \frac{\left| \|W_{\text{dense},i}\|^2 - \|W_{\text{MoE},i}\|^2 \right|}{\max(\|W_{\text{dense},i}\|^2, \|W_{\text{MoE},i}\|^2)}
  $$
  - △norm ∈ [0.7, 1] → Dense-only feature
  - △norm ∈ [0.0, 0.3] → MoE-only feature
  - △norm ∈ [0.3, 0.7] → Shared feature
- **Feature Density**：特征激活频率的统计分布。
- **Decoder Vector Cosine Similarity**：衡量跨模型间相同索引特征的方向一致性。

### 基线方法对比
- **Standard Crosscoder**：未使用显式共享机制的标准版本，在本任务中表现出对“共享”特征的过度估计（即使 decoder 向量余弦相似度接近零也被判为共享）。
- 本文提出的 **Fixed Shared + BatchTopK Crosscoder** 作为主方法，显著优于基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 所提 Crosscoder 实现了高达 **~87% 的 fractional variance explained**，表明其能有效捕捉 MoE 与 Dense 模型第三层激活的主要变化。
- 经过 40K 训练步后收敛良好，验证损失稳定。

### 与基线方法的对比结果
| 方法 | FVE | 问题 |
|------|-----|------|
| Standard Crosscoder | 较低 | 过度识别“共享”特征，decoder 向量无实际相关性（cosine ~ 0） |
| **Proposed (BatchTopK + Fixed Shared)** | **~87%** | 成功区分 MoE-only、Dense-only 和真正共享特征 |

> 注：原文指出先前工作中推荐的 `λ_s / λ_f ≈ 0.1–0.2` 不适用于本场景，最终需调整至 **≈0.7** 才能获得合理结果，说明独立训练的不同架构模型之间激活差异更大。

### 特征分布与性质（Table 1 & Figure 3）
| 类别 | △norm 区间 | 数量 | 特征密度趋势 |
|------|------------|--------|----------------|
| MoE-only features | 0.0–0.3 | 910 | **高于共享特征** |
| Dense-only features | 0.7–1.0 | 3,226 | **低于共享特征** |
| Shared features | 0.3–0.7 | 18,940 | 中等 |

> ⚠️ 与 base/fine-tuned 场景不同：在微调场景中，两类专属特征通常都比共享特征更稀疏；而在本工作中，MoE 专属特征反而更密集。

### 其他观察
- **缺乏三峰结构**：不像 base/fine-tuned 比较中常见的 trimodal △norm 分布，MoE/Dense 的 △norm 分布未呈现明显三峰，说明两者差异更为连续复杂。
- **共享特征方向不一致**：许多被识别为“共享”的特征其 decoder 向量余弦相似度较低，甚至出现反向（cosine ~ -1），提示“功能等价”不一定意味着“方向对齐”。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **MoE 学习更少但更专精的特征**：相比 Dense 模型，MoE 发展出显著更少的独特特征（910 vs. 3,226），支持“稀疏性促进局部专业化”的假设。
2. **信息组织方式根本不同**：
   - MoE 倾向于发展出 **高密度、高度专业化** 的表示（尤其在其独有特征上）。
   - Dense 模型则倾向于将信息分散在大量 **低密度、通用性强** 的特征中。
3. **Crosscoder 可迁移到架构比较**：尽管最初为微调分析设计，Crosscoder 经改进后可用于跨架构模型的 mechanistic interpretability 研究。
4. **共享 ≠ 对齐**：高方差解释并不意味着语义对齐；许多共享特征在方向上并不一致，提示需结合更多语义分析手段。

### 方法的局限性
- 当前 Crosscoder 仍难以完全解耦结构性差异带来的激活偏移，可能低估真实共享结构。
- 仅分析单一层（第3层），未能构建跨层动态图谱。
- 缺乏对发现特征的 **qualitative semantic analysis**（例如通过 prompt 查看其触发行为），无法确认其是否 truly monosemantic。
- 使用的是小型 5-layer 模型，结论在超大规模 MoE 上是否成立有待验证。

### 未来工作方向
- 开展 **定性特征分析**，验证所提取特征的语义可解释性和功能性。
- 将 Crosscoder 应用于更多层或全模型，建立 MoE 与 Dense 的 **cross-layer feature mapping**。
- 探索更适合结构差异大的模型对的新型 Crosscoder 架构（如引入 alignment loss 或非线性映射）。
- 研究专家（expert）级别的特征专门化模式，理解路由机制如何塑造表示空间。
- 将该框架推广至其他稀疏架构（如 Block-Sparse、Conditional Computation）的比较研究。

</details>

---

### 15. [Weak-SIGReg: Covariance Regularization for Stable Deep Learning](https://arxiv.org/abs/2603.05924)

**Authors**: Habibullah Akbar  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.05924v1  

#### Abstract
Modern neural network optimization relies heavily on architectural priorssuch as Batch Normalization and Residual connectionsto stabilize training dynamics. Without these, or in low-data regimes with aggressive augmentation, low-bias architectures like Vision Transformers (ViTs) often suffer from op...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Weak-SIGReg: Covariance Regularization for Stable Deep Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代深度学习模型（尤其是低归纳偏置架构如 Vision Transformers, ViTs）在缺乏 Batch Normalization、Residual Connections 等架构先验时，训练过程容易出现**优化崩溃（optimization collapse）**，表现为表示空间退化为低维流形（即维度坍缩），导致性能急剧下降。尤其在小数据集上使用强数据增强（如 MixUp、CutMix）时，该问题尤为严重。

传统解决方案依赖于复杂的架构设计或精细的超参数调优，而本文旨在提出一种**通用、轻量、无需架构修改的优化稳定机制**。

---

### **提出了什么新方法或新思路**
作者引入并改进了来自自监督学习框架 LeJEPA 的 **Sketched Isotropic Gaussian Regularization (SIGReg)**，将其应用于**监督学习场景**，并提出了简化版本：

#### ✅ **Weak-SIGReg（核心创新）**
- 原始的 **Strong-SIGReg** 通过匹配嵌入表示的经验特征函数（Empirical Characteristic Function, ECF）与各向同性高斯分布的解析特征函数来约束所有矩（moments），理论上更完整但计算开销大。
- **Weak-SIGReg** 则仅聚焦于**第二阶矩（协方差矩阵）的正则化**，利用 Random Sketching 技术将高维嵌入投影到低维空间，在该低维空间中强制协方差接近单位矩阵 $I$。
- 使用 Frobenius 范数最小化 $\|\text{Cov}(ZS^T) - I\|_F$ 作为损失项，实现高效正则化。

> 🔍 **关键洞察**：在监督学习中，防止坍缩的关键是控制表示的协方差结构，而非完全匹配整个分布。

---

### **相比现有方法的优势**
| 维度 | Weak-SIGReg 的优势 |
|------|--------------------|
| **通用性** | 可作为“插件式”正则项，适用于任何网络结构（ViT、MLP、CNN），不依赖特定架构组件（如 BN / ResNet） |
| **效率** | 内存复杂度从 $O(C^2)$ 降至 $O(CK)$（$K \ll C$），适合高维表示（如 $C=1024$） |
| **稳定性** | 显著提升 AdamW 或纯 SGD 下不稳定训练的收敛性，避免坍缩 |
| **替代调参** | 在无需专家级超参数调整的情况下，达到甚至超过手动调优基线的性能 |
| **理论基础** | 基于随机动力学视角（Dean-Kawasaki dynamics），将坍缩视为“随机漂移”，提供数学解释 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 主要实验基于 **CIFAR-100** 数据集进行验证。
- 所有模型均在此数据集上训练与评估。

---

### **实验设置和评估指标**

#### ✅ **评估指标**
- **Top-1 Accuracy** 为主要评价标准。
- 是否发生 **训练崩溃（Collapse）** 也被记录（如准确率停留在 ~20% 视为崩溃）。

#### ✅ **统一设置**
- 所有实验均使用 **gradient clipping (norm=1.0)** 以确保公平比较。
- 使用 **AdamW** 或 **pure SGD（无动量）** 进行优化。
- **Sketch Dimension $K = 64$** 对 Strong 和 Weak SIGReg 均适用。
- 正则强度 $\alpha = 0.1$ 默认值。

---

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Baseline** | 无任何 SIGReg 的标准训练流程 |
| **Expert-Tuned Baseline** | 应用多种 ViT 稳定技巧（weight decay、positional embedding、LR scheduler 等）的手动调优版本 |
| **Strong-SIGReg (LeJEPA)** | 原始全阶矩匹配方法，作为上界参考 |
| **Other Optimizers** | 引入 Muon（一种支持正交更新的新优化器）测试组合效果 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### 📊 **表1：ViT 在 CIFAR-100 上的优化稳定性（强增广 + AdamW）**

| Optimizer | SIGReg | Top-1 Acc | Status |
|----------|--------|-----------|--------|
| AdamW    | None   | 20.73%    | Collapse |
| AdamW    | Strong | 70.20%    | Converged |
| AdamW    | Weak   | **72.02%** | Converged |

> 💡 **结论**：Weak-SIGReg 不仅完全恢复训练，且性能略优于 Strong-SIGReg。

---

#### 📊 **表2：专家调优 vs. Weak-SIGReg**

| Model Setup | SIGReg | Top-1 Acc |
|------------|--------|-----------|
| Expert-Tuned Baseline | None | 70.76% |
| Expert-Tuned Baseline | Strong | 72.71% |
| Expert-Tuned Baseline | Weak | 71.65% |

> ✅ **发现**：Weak-SIGReg 在未经专家干预的默认设置下即可匹敌甚至超越精心调参的模型，说明其可作为鲁棒默认正则器。

---

#### 📊 **表3：Vanilla MLP 压力测试（6层，无 BN / Residuals，纯 SGD）**

| Augmentation | SIGReg | Top-1 Acc |
|--------------|--------|-----------|
| None         | None   | 26.77%    |
| None         | Strong | 35.99%    |
| None         | Weak   | **42.17%** |

> ⚡️ **提升幅度**：+15.4%，表明 Weak-SIGReg 极大改善了深层 MLP 中的梯度传播条件，起到了类似 Soft Batch Normalization 的作用。

---

#### 📊 **附录扩展实验亮点**

##### 🔹 **ResNet18 验证安全性**
- 在已有 BatchNorm 和 Residual 结构的 ResNet18 上，加入 SIGReg 并未降低性能（甚至略有提升），证明其**对已稳定架构无副作用**。

##### 🔹 **与 Muon 优化器结合**
| Model Version | Augmentation | SIGReg | Top-1 Acc | Gain |
|---------------|--------------|--------|-----------|------|
| Standard ViT  | Yes          | None   | 62.44%    | — |
| Standard ViT  | Yes          | Weak   | **74.56%** | +12.12% |
| Fixed ViT     | Yes          | Strong | **76.98%** | 最高精度 |

> 🔗 **结论**：SIGReg 与先进优化器（Muon）互补，联合使用可达最佳性能。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **几何正则化是优化稳定的有力工具**  
   通过约束表示空间的协方差结构（使其趋向 isotropic Gaussian），可以有效抑制由小批量噪声、高学习率和强增广引起的“随机漂移”。

2. ✅ **Weak-SIGReg 是高效实用的稳定器**  
   尽管只约束二阶矩，但在监督学习任务中足以防止坍缩，并在多个架构（ViT、MLP）上显著提升性能。

3. ✅ **可替代复杂调参流程**  
   在无需手动调整 weight decay、初始化、LR schedule 等情况下，Weak-SIGReg 即能达到专家调优水平，具备成为“默认正则项”的潜力。

4. ✅ **与现代技术正交且兼容**  
   与 Muon 等新型优化器、专家架构改进均可叠加使用，带来持续增益。

---

### **方法的局限性**
- ❗ **主要验证于中小规模数据集（CIFAR-100）**，在 ImageNet 等大规模图像任务上的泛化能力尚待验证。
- ❗ **对某些强增广场景可能需更长训练周期**（见 Appendix A.2.4 中 CutMix 下表现略降，推测因 epoch 不足）。
- ❗ **理论假设依赖各向同性高斯先验**，是否最优仍可探讨（例如任务相关结构是否应被保留？）

---

### **未来工作方向**
1. 🔄 探索其他 sketching 方法（如 Count Sketch、Fast Johnson-Lindenstrauss Transform）进一步加速。
2. 🧠 将 Weak-SIGReg 扩展至语言模型、图神经网络等其他模态。
3. 🤖 结合动态调节机制，根据训练阶段自适应调整正则强度 $\alpha$。
4. 🧬 研究如何在保持稳定性的同时允许表示学习更具判别性的几何结构（非完全各向同性）。

---

> 🔗 **代码开源地址**：[github.com/kreasof-ai/sigreg](https://github.com/kreasof-ai/sigreg)

</details>

---

### 16. [Omni-Masked Gradient Descent: Memory-Efficient Optimization via Mask Traversal with Improved Convergence](https://arxiv.org/abs/2603.05960)

**Authors**: Hui Yang, Tao Ren, Jinyang Jiang, Wan Tian, Yijie Peng  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.05960v1  

#### Abstract
Memory-efficient optimization methods have recently gained increasing attention for scaling full-parameter training of large language models under the GPU-memory bottleneck. Existing approaches either lack clear convergence guarantees, or only achieve the standard ${\mathcal{O}}(\epsilon^{-4})$ iter...

---

### 17. [A recipe for scalable attention-based MLIPs: unlocking long-range accuracy with all-to-all node attention](https://arxiv.org/abs/2603.06567)

**Authors**: Eric Qu, Brandon M. Wood, Aditi S. Krishnapriyan, Zachary W. Ulissi  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.06567v1  

#### Abstract
Machine-learning interatomic potentials (MLIPs) have advanced rapidly, with many top models relying on strong physics-based inductive biases. However, as models scale to larger systems like biomolecules and electrolytes, they struggle to accurately capture long-range (LR) interactions, leading curre...

---

### 18. [The World Won't Stay Still: Programmable Evolution for Agent Benchmarks](https://arxiv.org/abs/2603.05910)

**Authors**: Guangrui Li, Yaochen Xie, Yi Liu, Ziwei Dong, Xingyuan Pan, Tianqi Zheng, Jason Choi, Michael J. Morais, Binit Jha, Shaunak Mishra, Bingrou Zhou, Chen Luo, Monica Xiao Cheng, Dawn Song  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.05910v1  

#### Abstract
LLM-powered agents fulfill user requests by interacting with environments, querying data, and invoking tools in a multi-turn process. Yet, most existing benchmarks assume static environments with fixed schemas and toolsets, neglecting the evolutionary nature of the real world and agents' robustness ...

---

### 19. [Agentic LLM Planning via Step-Wise PDDL Simulation: An Empirical Characterisation](https://arxiv.org/abs/2603.06064)

**Authors**: Kai G\"obel, Pierrick Lorang, Patrik Zips, Tobias Gl\"uck  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.06064v1  

#### Abstract
Task planning, the problem of sequencing actions to reach a goal from an initial state, is a core capability requirement for autonomous robotic systems. Whether large language models (LLMs) can serve as viable planners alongside classical symbolic methods remains an open question. We present PyPDDLE...

---

### 20. [SAHOO: Safeguarded Alignment for High-Order Optimization Objectives in Recursive Self-Improvement](https://arxiv.org/abs/2603.06333)

**Authors**: Subramanyam Sahoo, Aman Chadha, Vinija Jain, Divya Chaudhary  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.06333v1  

#### Abstract
Recursive self-improvement is moving from theory to practice: modern systems can critique, revise, and evaluate their own outputs, yet iterative self-modification risks subtle alignment drift. We introduce SAHOO, a practical framework to monitor and control drift through three safeguards: (i) the Go...

---

### 21. [Boosting deep Reinforcement Learning using pretraining with Logical Options](https://arxiv.org/abs/2603.06565)

**Authors**: Zihan Ye, Phil Chau, Raban Emunds, Jannis Bl\"uml, Cedric Derstroff, Quentin Delfosse, Oleg Arenz, Kristian Kersting  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.06565v1  

#### Abstract
Deep reinforcement learning agents are often misaligned, as they over-exploit early reward signals. Recently, several symbolic approaches have addressed these challenges by encoding sparse objectives along with aligned plans. However, purely symbolic architectures are complex to scale and difficult ...

---

### 22. [Why Depth Matters in Parallelizable Sequence Models: A Lie Algebraic View](https://arxiv.org/abs/2603.05573)

**Authors**: Gyuryang Heo, Timothy Ngotiaoco, Kazuki Irie, Samuel J. Gershman, Bernardo Sabatini  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.05573v1  

#### Abstract
Scalable sequence models, such as Transformer variants and structured state-space models, often trade expressivity power for sequence-level parallelism, which enables efficient training. Here we examine the bounds on error and how error scales when models operate outside of their expressivity regime...

---

### 23. [A Novel Hybrid Heuristic-Reinforcement Learning Optimization Approach for a Class of Railcar Shunting Problems](https://arxiv.org/abs/2603.05579)

**Authors**: Ruonan Zhao, Joseph Geunes  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.05579v1  

#### Abstract
Railcar shunting is a core planning task in freight railyards, where yard planners need to disassemble and reassemble groups of railcars to form outbound trains. Classification tracks with access from one side only can be considered as stack structures, where railcars are added and removed from only...

---

### 24. [Bias In, Bias Out? Finding Unbiased Subnetworks in Vanilla Models](https://arxiv.org/abs/2603.05582)

**Authors**: Ivan Luiz De Moura Matos, Abdel Djalil Sad Saoud, Ekaterina Iakovleva, Vito Paolo Pastore, Enzo Tartaglione  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.05582v1  

#### Abstract
The issue of algorithmic biases in deep learning has led to the development of various debiasing techniques, many of which perform complex training procedures or dataset manipulation. However, an intriguing question arises: is it possible to extract fair and bias-agnostic subnetworks from standard v...

---

### 25. [Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments](https://arxiv.org/abs/2603.06009)

**Authors**: Michael Beukman, Khimya Khetarpal, Zeyu Zheng, Will Dabney, Jakob Foerster, Michael Dennis, Clare Lyle  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.06009v1  

#### Abstract
Plateaus, where an agent's performance stagnates at a suboptimal level, are a common problem in deep on-policy RL. Focusing on PPO due to its widespread adoption, we show that plateaus in certain regimes arise not because of known exploration, capacity, or optimization challenges, but because sample...

---

### 26. [RoboLayout: Differentiable 3D Scene Generation for Embodied Agents](https://arxiv.org/abs/2603.05522)

**Authors**: Ali Shamsaddinlou  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.05522v1  

#### Abstract
Recent advances in vision language models (VLMs) have shown strong potential for spatial reasoning and 3D scene layout generation from open-ended language instructions. However, generating layouts that are not only semantically coherent but also feasible for interaction by embodied agents remains ch...

---

### 27. [Structured Multidimensional Representation Learning for Large Language Models](https://arxiv.org/abs/2603.05727)

**Authors**: Alaa El Ichi, Khalide Jbilou, Mohamed El Guide, Franck Dufrenois  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.05727v1  

#### Abstract
Transformer architectures achieve state-of-the-art performance across a wide range of pattern recognition and natural language processing tasks, but their scaling is accompanied by substantial parameter growth and redundancy in the embedding dimension. In this work, we introduce a structured spectra...

---

### 28. [InfoGatherer: Principled Information Seeking via Evidence Retrieval and Strategic Questioning](https://arxiv.org/abs/2603.05909)

**Authors**: Maksym Taranukhin, Shuyue Stella Li, Evangelos Milios, Geoff Pleiss, Yulia Tsvetkov, Vered Shwartz  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.05909v1  

#### Abstract
LLMs are increasingly deployed in high-stakes domains such as medical triage and legal assistance, often as document-grounded QA systems in which a user provides a description, relevant sources are retrieved, and an LLM generates a prediction. In practice, initial user queries are often underspecifi...

---

### 29. [Evaluation of Deontic Conditional Reasoning in Large Language Models: The Case of Wason's Selection Task](https://arxiv.org/abs/2603.06416)

**Authors**: Hirohiko Abe, Kentaro Ozeki, Risako Ando, Takanobu Morishita, Koji Mineshima, Mitsuhiro Okada  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.06416v1  

#### Abstract
As large language models (LLMs) advance in linguistic competence, their reasoning abilities are gaining increasing attention. In humans, reasoning often performs well in domain specific settings, particularly in normative rather than purely formal contexts. Although prior studies have compared LLM a...

---

### 30. [A Lock-Free Work-Stealing Algorithm for Bulk Operations](https://arxiv.org/abs/2603.05766)

**Authors**: Raja Sai Nandhan Yadav Kataru, Danial Davarnia, Ali Jannesari  
**Category**: cs.DC  
**Published**: 2026-03-09  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.05766v1  

#### Abstract
Work-stealing is a widely used technique for balancing irregular parallel workloads, and most modern runtime systems adopt lock-free work-stealing deques to reduce contention and improve scalability. However, existing algorithms are designed for general-purpose parallel runtimes and often incur over...

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

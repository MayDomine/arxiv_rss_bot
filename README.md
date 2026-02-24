# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-24 06:44:07 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model](https://arxiv.org/abs/2602.19128)

**Authors**: Shiyi Cao, Ziming Mao, Joseph E. Gonzalez, Ion Stoica  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.19128v1  

#### Abstract
Optimizing GPU kernels is critical for efficient modern machine learning systems yet remains challenging due to the complex interplay of design factors and rapid hardware evolution. Existing automated approaches typically treat Large Language Models (LLMs) merely as stochastic code generators within...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代 GPU 内核（GPU kernels）优化对高性能机器学习系统至关重要，但面临以下挑战：
- **设计空间巨大**：涉及 tiling、内存布局、同步机制、架构特定指令等多重因素。
- **硬件快速演进**：如从 NVIDIA Hopper 到 Blackwell 架构的变化，使旧优化失效。
- **手动调优成本高**：依赖专家经验，试错成本大，且编译与评测开销昂贵。

现有基于 LLM 的自动化方法（如 OpenEvolve）通常将 LLM 视为**随机代码生成器**，嵌入启发式进化搜索中。这类方法存在严重缺陷：
- 缺乏**高层规划能力**，难以执行需要多步协调的结构性变换（如先重构内存再向量化）。
- 中间步骤若未立即提升性能或出现临时错误（如语法错误），会被过早丢弃，导致优秀策略被误判。

---

### 🚀 提出的新方法：**K-SEARCH** 与 **Co-Evolving World Model**

提出 **Search via Co-Evolving World Model** 框架，并构建 **K-SEARCH** 系统，其核心思想是：

#### 🔹 方法创新
1. **解耦高层算法规划与底层程序实例化**
   - 将内核生成建模为一个**结构化搜索树上的规划问题**。
   - 引入由 LLM 实例化的 **World Model** 来维护搜索状态，估计高阶优化意图的优先级。

2. **动态共进化机制（Co-Evolution）**
   - World Model 不是静态的，而是通过 **in-context learning** 持续吸收执行反馈（如性能、错误日志），动态更新其对搜索空间的理解。
   - 支持在非单调路径上探索（允许中间退步），并容忍临时实现缺陷。

3. **三阶段迭代流程**
   - **Action Selection**：选择优先级最高的优化动作（如“融合头”、“拆分序列”）。
   - **Local Refinement**：用 LLM 多次采样实现该动作，直到连续失败达到阈值（避免因单次错误放弃有效策略）。
   - **World Model Update**：基于结果进行 **Insert / Update / Prune** 操作，动态调整搜索树。

---

### ⚖️ 相比现有方法的优势
| 维度 | 传统方法（如 OpenEvolve） | K-SEARCH |
|------|--------------------------|--------|
| LLM 角色 | 纯粹代码生成器 | 具备推理与规划能力的 **World Model** |
| 搜索空间 | 直接在程序空间搜索 | 在**意图空间 + 程序空间**分层搜索 |
| 错误容忍 | 一次失败即丢弃 | 局部精炼机制容忍临时错误 |
| 长程规划 | 无显式机制 | 显式维护搜索树与信念状态 |
| 效率 | 高方差，低成功率 | 更高效地聚焦有潜力的方向 |

---

## 2. 核心实验方法和设置

### 📚 数据集与任务
在 **FlashInfer** 提供的真实 LLM 推理负载上测试，涵盖多种复杂内核：
- **MLA Paged Prefill / Decode**（Multi-Level Attention）
- **GQA Paged Decode**（Grouped Query Attention）
- **FP8 MoE**（Mixture-of-Experts，Blackwell 架构）
- **GPUMode TriMul**（AlphaFold3 中的三角乘法更新模块）

所有任务均使用真实流量捕获的 trace 进行评测。

---

### 🧪 实验设置
- **评估预算**：每个任务最多运行 120 次迭代（每次对应一次完整编译 + 正确性验证 + 性能评测）。
- **硬件平台**：NVIDIA H100（Hopper）、B200（Blackwell），CUDA 12.8。
- **语言模型**：
  - K-SEARCH：使用 `gemini-3-pro-preview` 和 `GPT-5.2`。
  - 基线统一使用相同初始程序和输入 workload。
- **正确性保障**：所有生成 kernel 必须通过 PyTorch 参考实现的功能一致性测试，否则得分为零。

---

### 🎯 评估指标
1. **目标函数 $ J(x) $**：
   $$
   J(x) = s \cdot \frac{p_{\text{ref}}}{p}, \quad s \in \{0,1\}
   $$
   其中 $ s $ 表示是否正确，$ p $ 是延迟，$ p_{\text{ref}} $ 是 FlashInfer 基线延迟。
   
2. **平均加速比（Speedup over FlashInfer）**
3. **每 workload 最佳性能分布**
4. **最终最佳 kernel 的绝对延迟**（如 TriMul 的 1030 μs）

---

### 🆚 基线方法对比
- **OpenEvolve**：基于 MAP-Elites 的质量多样性搜索，LLM 作为代码生成器。
- **ShinkaEvolve**：结合性能选择与新颖性拒绝的种群进化框架。
- 所有方法使用相同的 evaluator、编译工具链和 benchmark harness，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 总体性能对比（平均最终得分）
| 方法 | 平均得分 | 相对提升 |
|------|---------|--------|
| **K-SEARCH** | **56.13** | — |
| OpenEvolve | 26.68 | ↑ **2.10×** |
| ShinkaEvolve | 25.37 | ↑ **2.21×** |

> 得分越高表示相对于 FlashInfer 基线的加速越显著。

---

### 🔍 各内核详细表现

| Kernel | K-SEARCH | OpenEvolve | 提升倍数 |
|-------|----------|-----------|--------|
| **MoE (FP8)** | 44.1 | 3.09 | **↑14.3×** |
| **MLA Prefill** | 57.4 | 19.5 | ↑2.95× |
| **GQA Decode** | 76.0 | 44.2 | ↑1.72× |
| **MLA Decode** | 47.1 | 39.9 | ↑18% |

> 在最难的 MoE 内核上取得压倒性优势，说明 K-SEARCH 特别擅长处理**不规则路由、负载均衡、FP8 编解码**等复杂结构。

---

### 🏁 GPUMode TriMul 比赛结果
- **任务**：Triangle Multiplicative Update（TriMul），用于 AlphaFold3。
- **搜索预算**：300 次迭代（前 150 用 GPT-5.2，后 150 用 Gemini-3-Pro 接续）。
- **结果**：
  - **K-SEARCH 达到 1030 μs（geometric mean latency）**
  - 超越此前 SOTA 方案（TTT-Discover，结合 RL 与进化）
  - **首次无需人工种子程序，在 Triton/CUDA 上实现全自动 SOTA 性能**

---

### 📉 性能短板分析
- 在 **GQA Decode 的小批量场景（batch_size=1 或 16）** 下，K-SEARCH 表现略逊于基线。
- 原因：K-SEARCH 使用 **Split-K 并行策略**，适合大批量但带来额外同步开销。
- 基线采用单 block 设计，在极小批量下更轻量。

> 说明当前方法偏向“最大化吞吐”的优化方向，未来可引入 workload 自适应策略。

---

### ❌ 消融实验（文中未明确列出，但从分析可推断）
虽然没有独立消融章节，但以下发现隐含了关键组件的作用：
- **World Model 的信念更新机制** 是成功的关键：能够动态降级无效分支（如 `independent_heads`）、重新定位策略（如 `low_overhead_split_k` 重插入）。
- **Local Refinement** 机制显著提高鲁棒性：防止因一次语法错误丢弃合理意图。
- **Tree-based Search Structure** 使得长期记忆和结构演化成为可能，区别于扁平化种群搜索。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 不仅是代码生成器，更是强大的内在世界模型（Intrinsic World Model）**
   - 能够利用先验知识进行**高层次规划与推理**。
   - 可以模拟动作后果、评估候选路径、指导搜索方向。

2. **解耦“意图”与“实现”显著提升搜索效率**
   - 避免在庞大的稀疏程序空间中盲目枚举。
   - 支持非单调优化路径，容忍中间失败。

3. **共进化机制使搜索过程具备持续学习能力**
   - World Model 随搜索进程不断进化，形成正反馈循环。
   - 成功实现了 **LLM 驱动的自主搜索策略演化**。

4. **在多个复杂 kernel 上实现 SOTA 性能**
   - 平均 **2.1× 超越最强进化基线**，MoE 上达 **14.3×**。
   - 在 GPUMode TriMul 上达成 **1030 μs**，超越人类设计。

---

### ⚠️ 方法局限性
1. **依赖高质量 LLM**：性能受限于 LLM 的领域理解与推理能力。
2. **对小批量 workload 优化不足**：当前策略偏好高并行度设计。
3. **尚未支持多目标优化**（如功耗、显存占用）。
4. **World Model 更新仍基于 in-context learning**，缺乏参数微调，长期记忆有限。

---

### 🔮 未来工作方向
1. **引入多模态 feedback**：结合 profiler 输出、SM occupancy、cache miss 等细粒度指标增强 World Model 学习。
2. **自适应并行策略选择**：根据 batch size、sequence length 动态切换 Split-K、Single Block 等模式。
3. **扩展至其他 DSL**：如 Triton、Halide、Spiral，推动通用自动优化引擎。
4. **闭环训练机制**：将成功轨迹反哺训练数据，实现 LLM 自我改进。
5. **跨硬件迁移优化**：研究如何将在 Hopper 上学到的策略迁移到 Blackwell 或 AMD GPU。

---

## 总结

> **K-SEARCH 实现了从“LLM as Code Generator”到“LLM as Planner & World Model”的范式转变**。它不再把 LLM 当作黑盒生成器，而是充分发挥其**内在知识、推理能力和环境建模潜力**，构建了一个能主动思考、持续学习、协同进化的智能搜索系统。这不仅是 GPU kernel 自动生成的重大突破，也为 LLM 在科学计算、系统优化等复杂决策任务中的应用开辟了新路径。

</details>

---

### 2. [Proximity-Based Multi-Turn Optimization: Practical Credit Assignment for LLM Agent Training](https://arxiv.org/abs/2602.19225)

**Authors**: Yangyi Fang, Jiaye Lin, Xiaoliang Fu, Cong Qin, Haolin Shi, Chang Liu, Peilin Zhao  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.19225v1  

#### Abstract
Multi-turn LLM agents are becoming pivotal to production systems, spanning customer service automation, e-commerce assistance, and interactive task management, where accurately distinguishing high-value informative signals from stochastic noise is critical for sample-efficient training. In real-worl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Proximity-Based Multi-Turn Optimization: Practical Credit Assignment for LLM Agent Training

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在多轮 LLM Agent 的训练中，**信用分配（credit assignment）** 是一个核心挑战。现有基于分组的策略优化方法（如 GRPO）依赖于离散批次内的统计偏差来计算优势函数（advantage），忽略了任务难度变化带来的信息价值差异。

具体表现为两个层面的问题：
- **Episode-Level（回合级）**：相同 z-score 幅度在高成功率（如 75%）和低成功率（如 25%）任务中被赋予相同的信用强度，导致对“突破性成功”奖励不足，而对“偶然失败”过度惩罚。
- **Step-Level（步骤级）**：现有方法采用硬边界划分（hard boundary partitioning），即通过精确匹配或相似度阈值将状态划分为离散群组。这会导致：
  - 阈值过严 → 单例群组（singleton groups），无法进行归一化；
  - 阈值过松 → 不同语义的状态被等权重处理，削弱信用区分能力。

### 提出了什么新方法或新思路

作者提出 **Proximity-based Multi-turn Optimization (ProxMO)**，一种融合全局上下文的信用分配框架，在两个层级引入轻量级机制：

#### （1）Episode-Level: Success-Rate-Aware Modulation
- 引入 **Polarized Signal Controller (PSC)**，根据整个 episode 组的成功率 $ p $ 动态调整梯度强度。
- 对于低成功率任务中的成功行为给予更强激励（amplify breakthroughs）；
- 对于高成功率任务中的失败行为减轻惩罚（attenuate noise penalties）。
- 公式上通过非线性函数（Sigmoid）设计加权因子 $ w(R, p) $ 来调制原始优势值。

#### （2）Step-Level: Proximity-Based Soft Aggregation
- 引入 **Proximity-based Soft Aggregation (PSA)**，取代硬分组。
- 所有状态都参与 baseline 构建，但按其与当前状态的 **语义相似度** 进行连续加权。
- 使用 TF-IDF 向量计算余弦相似度，并通过温度参数 $ \tau $ 控制权重集中程度。
- 实现了平滑过渡：$ \tau \to 0 $ 接近 exact matching；$ \tau \to \infty $ 趋向均匀加权。

### 相比现有方法的优势

| 方面 | GRPO | GiGPO | ProxMO |
|------|------|--------|---------|
| 是否依赖 Critic Network | ❌ | ❌ | ❌ |
| Episode-Level 上下文感知 | ❌ | ❌ | ✅（成功率感知） |
| Step-Level 分组方式 | 整体归一化 | 硬匹配（exact/similarity threshold） | 软聚合（continuous semantic weighting） |
| 单例群组问题 | 不适用 | 存在严重问题（30–36% 步骤为 singleton） | 完全避免 |
| 插件兼容性 | 标准框架 | 需修改 | ✅ 支持 plug-and-play 集成到 GRPO |

> ✅ **核心优势**：在不增加模型前向/反向传播开销的前提下，显著提升 credit assignment 的准确性和鲁棒性，适用于工业级部署。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

- **ALFWorld**  
  - 一个具身环境（embodied environment），包含 3,827 个任务实例，涵盖六类家庭活动：
    - Pick & Place (Pick)
    - Examine in Light (Look)
    - Clean & Place (Clean)
    - Heat & Place (Heat)
    - Cool & Place (Cool)
    - Pick Two & Place (Pick2)
  - 每个任务需多步推理与操作，适合测试长期信用分配能力。

- **WebShop**  
  - 一个基于 HTML 的电商购物模拟环境，包含超过 110 万真实商品和 12K 用户指令。
  - 代理需执行搜索、浏览、筛选、购买等动作完成复杂目标（如“买便宜且评价好的无线耳机”）。
  - 观察空间为半结构化文本（HTML 片段），挑战信息提取与决策一致性。

### 实验设置和评估指标

#### 模型配置
- 主干模型：`Qwen2.5-1.5B-Instruct` 和 `Qwen2.5-7B-Instruct`
- 超参数（默认）：
  - 折扣因子 $ \gamma = 0.95 $
  - 温度 $ T = 0.1 $
  - Episode steepness $ \alpha = 4.0 $
  - Episode strength $ \beta = 0.1 $
  - Group size $ N = 8 $

#### 评估指标
- **Success Rate (%)**：任务是否最终完成。
- **Score (%)**：属性匹配得分（尤其 WebShop 中衡量推荐质量）。
- 所有结果取 **3 个随机种子的平均值**。

### 基线方法对比

| 类别 | 方法 |
|------|------|
| **闭源大模型** | GPT-4o, Gemini-2.5-Pro |
| **提示工程 Agent** | ReAct, Reflexion |
| **RL 训练方法** | GRPO (group-based), GiGPO (step-level exact matching) |

> 所有方法使用相同 prompt 模板和训练配置以确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 `Qwen2.5-1.5B-Instruct` 上的表现（部分关键项）

| Method | ALFWorld (All) | WebShop (Score/Succ.) |
|--------|----------------|------------------------|
| GPT-4o | 48.0 | 31.8 / 23.7 |
| Gemini-2.5-Pro | 60.3 | 42.5 / 35.9 |
| GRPO | 70.3 | 73.1 / 52.2 |
| GiGPO | 85.2 | 81.7 / 62.3 |
| **ProxMO (Ours)** | **90.6** | **85.3 / 67.1** |

> 💡 **亮点**：仅用 1.5B 小模型，ProxMO 在多个任务上超越甚至媲美 GPT-4o 和 Gemini。

#### 在 `Qwen2.5-7B-Instruct` 上的表现

| Method | ALFWorld (All) | WebShop (Score/Succ.) |
|--------|----------------|------------------------|
| GRPO | 79.8 | 79.2 / 67.2 |
| GiGPO | 89.5 | 85.5 / 74.8 |
| **ProxMO (Ours)** | **94.5** | **87.2 / 76.5** |

> ✅ 全面领先，尤其在长视野任务（如 Look, Cool, Pick2）增益更明显。

### 与基线方法的对比结果

- 相比 GRPO，ProxMO 在 ALFWorld 上平均提升 **+28.9%**（1.5B）、**+18.4%**（7B）；
- 在 WebShop 成功率上分别提升 **+28.5%** 和 **+13.8%**；
- 在最难的 Pick2 任务上，1.5B 模型从 GRPO 的 50.0 提升至 **87.0**（+74%），显示其对复杂序列决策的强大支持。

### 消融实验结果（Ablation Study）

#### 设置
- 在 ALFWorld 上对 `Qwen2.5-1.5B-Instruct` 进行消融分析。
- 移除 PSC（episode-level modulation）或 PSA（step-level aggregation）模块。

#### 结果（见 Figure 4）
- 移除任一组件均导致性能下降；
- **移除 PSA 影响更大**，特别是在 Look、Pick2 等需要精细动作排序的任务上；
- **完整 ProxMO 不仅优于各变体，还超过了强基线 GiGPO**，说明两种机制存在 **协同效应（synergy）**。

> 🔍 发现：episode-level 的难度感知放大了 step-level 精确信用的价值，二者结合才能稳定学习异构任务。

---

## 4. 关键结论和发现

### 论文的主要发现

1. **Credit Assignment 必须考虑上下文信息**  
   单纯依赖统计偏差（如 z-score）会误导学习信号。任务成功率本身是重要的元信息，应被纳入优势计算。

2. **软聚合优于硬分组**  
   在高维状态空间中，exact matching 导致大量单例群组（实证占 30–36% 步骤），造成训练信号丢失。连续语义加权可有效缓解此问题。

3. **Hierarchical Design 带来协同增益**  
   Episode-level 的全局调制 + Step-level 的局部精细化，共同提升了 credit assignment 的精度与稳定性。

4. **高效且易于部署**  
   ProxMO 几乎无额外计算开销（+1.09% 训练时间），且可直接集成进现有 GRPO 流程，具备极强工业落地潜力。

### 方法的局限性

- 当前实验集中在资源受限场景（1.5B 和 7B 模型），尚未验证在超大规模模型（如 70B+）上的泛化能力。
- 语义相似度依赖 TF-IDF，虽轻量但可能不如 embedding-based 方法表达能力强（作者指出未来可替换）。
- 成功率估计依赖 batch 内经验频率，在动态环境中可能存在延迟或偏差。

### 未来工作方向

- 将 ProxMO 扩展至更大规模的 Foundation Models，验证其可扩展性。
- 探索更先进的语义编码器（如 SBERT、CLIP）替代 TF-IDF。
- 结合 online adaptation 机制，实现动态 success rate 估计。
- 应用于更多现实世界场景，如 GUI Agent、自动驾驶对话系统等。

---

> 📌 **一句话总结**：  
> **ProxMO 通过引入成功率感知调制和语义软聚合，在 episode 和 step 两个层级实现了更合理、更鲁棒的信用分配，显著提升了多轮 LLM Agent 的训练效率与性能，同时保持了极低的计算开销和良好的工程兼容性。**

</details>

---

### 3. [Leap+Verify: Regime-Adaptive Speculative Weight Prediction for Accelerating Neural Network Training](https://arxiv.org/abs/2602.19580)

**Authors**: Jeremy McEntire  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.19580v1  

#### Abstract
We introduce Leap+Verify, a framework that applies speculative execution -- predicting future model weights and validating predictions before acceptance -- to accelerate neural network training. Inspired by speculative decoding in language model inference and by the Automatically Scalable Computatio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Leap+Verify: Regime-Adaptive Speculative Weight Prediction for Accelerating Neural Network Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代神经网络训练是一个**高度序列化的过程**，每一步梯度更新都依赖于当前参数状态，导致训练耗时极长、计算资源消耗巨大（如千卡GPU小时、百万美元级成本）。尽管已有研究尝试通过预测未来权重来加速训练（weight nowcasting），但这些方法通常存在两个关键缺陷：
- **无验证机制**：直接应用预测权重，可能引入错误并破坏训练轨迹；
- **无条件预测**：在所有训练阶段均进行预测，忽略了不同训练阶段（regime）中权重轨迹的可预测性差异。

Leap+Verify 针对上述问题，提出了一种**安全且高效的投机式权重预测框架**，旨在不改变最终模型性能的前提下，跳过部分梯度更新步骤以加速训练。

---

### 提出的新方法与新思路

Leap+Verify 受到以下两种技术的启发：
- **Speculative Decoding**（用于语言模型推理加速）
- **Automatically Scalable Computation (ASC)** 架构（程序执行中的投机计算）

其核心是构建一个“**predict-then-verify**”机制，并将其应用于训练过程：

#### 方法架构三要素：
| 组件 | 功能 | 本文实现 |
|------|------|----------|
| **Recognizer（识别器）** | 判断当前是否适合预测 | 使用 activation-space cosine similarity 作为 Lyapunov 指数代理信号，动态检测训练所处 regime |
| **Predictor（预测器）** | 预测未来 K 步后的模型权重 | 采用三种解析型外推方法：momentum、linear、quadratic extrapolation |
| **Validator（验证器）** | 验证预测权重的有效性 | 在 hold-out 数据上比较预测模型的 validation loss 是否满足接受标准 |

#### 创新机制：
1. **Regime-Conditional Prediction**  
   将训练划分为三个动态可检测的阶段：
   - **Chaotic（混沌）**：表示学习初期，表征不稳定，不可预测
   - **Transition（过渡）**：开始收敛，轨迹出现规律性
   - **Stable（稳定）**：后期微调，路径平滑，最易预测  
   → 仅在 Transition 和 Stable 阶段启动预测，避免在 Chaotic 阶段浪费计算。

2. **Verify-Then-Accept 机制**  
   所有预测均为“纯投机”，若验证失败则丢弃，不影响原训练流程，确保**零副作用**。

3. **Analytic Extrapolators 而非 Learned Predictors**  
   不使用复杂的神经网络预测器（如 NiNo、WNN），而是基于历史 checkpoint 参数差分构造简单但高效的动力学模型。

---

### 相比现有方法的优势

| 特性 | 现有方法（如 WNN, NiNo, XGrad） | Leap+Verify |
|------|-------------------------------|------------|
| 是否验证预测 | ❌ 否，直接应用 | ✅ 是，verify-then-accept |
| 是否条件预测 | ❌ 否，全程预测 | ✅ 是，仅在 Transition/Stable 阶段预测 |
| 安全性保障 | ❌ 存在污染风险 | ✅ 失败预测无代价 |
| 实现复杂度 | 高（需训练额外预测网络） | 低（仅需前向传播 + 简单公式） |
| 可扩展性 | 受限于预测模型泛化能力 | 更通用，适用于任意 optimizer 和架构 |

> ✅ Leap+Verify 是首个将 **speculative execution 思想系统地移植到训练场景**的工作，并揭示了 regime 分布对可预测性的根本影响。

---

## 2. 核心实验方法和设置

### 使用的数据集与模型
- **Dataset**: WikiText-103（标准语言建模 benchmark）
- **Sequence Length**: 256
- **Models Evaluated**:
  - **GPT-2 124M**：12层，768隐藏维度
  - **Qwen 2.5-1.5B**：28层，1536隐藏维度
- **Optimizer**: AdamW（lr=5e-5, β₁=0.9, β₂=0.999, weight decay=0.01）
- **LR Schedule**: Cosine decay with 100-step warmup
- **Total Steps**: 2000 steps，checkpoint every 50 steps → 共 40 个检查点

---

### 实验设置

#### 三阶段评估流程：
1. **Pass 1（Training）**  
   正常训练并保存 checkpoint，同时记录 activation fingerprint（固定 probe set 上的 final hidden states）用于后续 regime 分类。

2. **Pass 2（K-Sweep）**  
   对每个非 chaotic checkpoint，测试不同预测深度 $ K \in \{5,10,25,50,75,100\} $ 下三种 predictor 的表现，使用三种 acceptance criteria：
   - **Strict**: $ L_{t+K} < L_t $
   - **Adaptive**: $ L_{t+K} < L_t + \sigma_L $
   - **Proximity (pct)**: $ |L_{t+K} - L_t| < \epsilon \cdot L_t $

3. **Pass 3（Cascades）**  
   测试多步链式预测（cascaded prediction），配置为 $(D,K) \in \{(4,25),(2,50),(10,10)\}$，即连续跳跃最多 $D\times K$ 步。

#### 重复性控制
- **5 seeds (42–46)** 运行，保证结果可复现
- 所有代码开源：https://github.com/jmcentire/leap-verify

---

### 基线方法对比
本文未直接对比传统 weight nowcasting 方法（因其缺乏验证机制），而是从**设计范式层面形成对比**：
- **Jang and Han [2023]** (WNN): 学习型 nowcaster，无验证
- **Knyazev et al. [2025]** (NiNo): GNN-based predictor，无条件预测
- **Guan et al. [2024]** (XGrad): 基于优化规则的数学预测，仍无验证机制

Leap+Verify 的优势在于引入了 **regime-aware + verify-gated** 的双重安全保障。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 模型 | Regime | Predictor | K | Strict Acceptance | Proximity Acceptance (5%) |
|------|--------|-----------|----|---------------------|----------------------------|
| GPT-2 124M | Stable | Linear | 5 | **24.3% ± 6.8%** | 98.9% |
| GPT-2 124M | Transition | Linear | 5 | 9.3% ± 3.8% | 99.1% |
| Qwen 1.5B | Transition | Linear | 5 | **37.2% ± 11.0%** | **100.0%** |
| Qwen 1.5B | Stable | Linear | 5 | – | **100%** (N=5) |

> 🔺 注意：**更大的模型在可预测阶段反而更可预测！**

---

### 与基线方法的关键对比结果

#### （1）Momentum Prediction 完全失效 —— “Universal Momentum Catastrophe”
| K | GPT-2 124M (Predicted / Actual Loss Ratio) | Qwen 1.5B (Ratio) |
|----|-----------------------------------------|--------------------|
| 5 | 122× | 173× |
| 10 | 219× | 305× |
| 100 | **10,764×** | **3,009×** |

- Momentum extrapolation 导致 predicted loss 暴涨数百至万倍，属于**质变级失败**。
- 原因：**norm explosion** —— $ K \cdot m / \sqrt{v} $ 的位移远超 loss landscape 局部有效区域。

#### （2）Finite-Difference Predictors 成功
- Linear & Quadratic extrapolation 利用实际观察到的 weight delta（$ \theta_t - \theta_{t-\Delta} $），天然受限于真实步长，具有正则化效果。
- 在 K=5 时，strict acceptance 达到 9–37%，proximity acceptance 接近 100%。

#### （3）Scale-Dependent Regime Distribution（关键发现）
| Model | Chaotic | Transition | Stable |
|-------|--------|-----------|--------|
| GPT-2 124M | 4% | 60% | **34%** |
| Qwen 2.5-1.5B | **64%** | 31% | **2.5%** |

- 更大模型（1.5B）**更难进入 stable regime**，训练大部分时间处于 chaotic 探索阶段。
- → 实际瓶颈不是 predictor 准确率，而是 **regime availability**！

---

### 消融实验结果

#### （1）Cross-Seed Consistency（高一致性）
- regime 边界在 ±50 steps 内一致
- validation loss 方差 <1%
- 表明 regime 结构由优化 landscape 决定，而非随机初始化主导

#### （2）Cascade Performance（链式预测）
- 在 stable-checkpoint 上尝试 cascaded prediction
- 结果：短链（如 D=2, K=10）可成功；深链迅速失败
- 原因：误差累积导致后续预测偏离加剧

#### （3）Acceptance Criteria 影响
- Strict 最保守，acceptance rate 随 K 快速下降
- Proximity 最宽松，在 K=25 时仍保持 >90%（1.5B）
- 提供灵活 trade-off between skip distance and safety

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Momentum-based prediction 完全不可行**  
   Adam moment extrapolation 引发 **universal momentum catastrophe**，无论模型大小均导致 loss 暴涨百倍以上，根本原因是 **norm explosion**。

2. ✅ **Finite-difference extrapolation 是可行路径**  
   基于实际 weight delta 的 linear/quadratic 外推能实现高达 **37% 的 strict acceptance（K=5）**，尤其在 larger models 的 transition regime 中表现更好。

3. ✅ **真正瓶颈是 regime availability，而非 predictor accuracy**  
   - 小模型（124M）早熟进入 stable phase（34% 时间）
   - 大模型（1.5B）长期停留在 chaotic phase（64% 时间）
   - → 即使 predictor 很准，也“没机会用”

4. ✅ **Activation-space cosine similarity 是有效的实时 regime detector**  
   无需昂贵的 Hessian 计算或后验分析，仅需 forward pass 即可实现实时分类，且跨 seed 高度一致。

5. ✅ **Verify-then-accept 机制使投机零成本**  
   失败预测不会干扰训练流，允许大胆尝试高风险跳跃。

---

### 方法的局限性

| 局限性 | 描述 |
|--------|------|
| **训练步数有限** | 实验仅运行 2000 步，Qwen 1.5B 尚未充分进入 stable phase，限制了 long-horizon 预测评估 |
| **Regime Threshold 固定** | Thigh/Tlow 在小模型上调参，可能不适用于更大模型（如 7B+） |
| **缺乏 ensemble 支持** | 当前未实现 multi-seed ensemble collapse，无法利用跨 seed convergence 加速 |
| **Cascade 效果有限** | 深度链式预测因误差积累快速失效，难以实现大规模跳跃 |

---

### 未来工作方向

1. **Adaptive Regime Thresholding**  
   设计自动校准机制，适应不同模型规模下的 activation similarity 分布变化。

2. **Ensemble Collapse**  
   当多个 seed 被检测到进入相同 regime 并趋同，动态合并训练进程，减少冗余计算。

3. **Scaling to Larger Models**  
   在 2.7B、7B 等更大模型上验证 predictor hierarchy 与 regime 分布趋势。

4. **Improved Cascading Strategies**  
   引入 feedback correction 或 adaptive K-selection，提升多步跳跃成功率。

5. **Integration with Optimizers**  
   探索与 Lookahead、RAdam 等自带 lookahead 机制的 optimizer 结合的可能性。

---

> 📌 **总结一句话**：  
> Leap+Verify 揭示了一个深刻洞见——**加速训练的关键不在“如何更好预测”，而在“何时可以预测”**。它通过 regime-aware speculative execution 框架，首次实现了安全、可复现、尺度感知的训练加速机制，为下一代高效训练系统提供了新范式。

</details>

---

### 4. [BiScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS](https://arxiv.org/abs/2602.18755)

**Authors**: Omar Basit, Yunzhao Liu, Z. Jonny Kong, Y. Charlie Hu  
**Category**: cs.DC  
**Published**: 2026-02-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.18755v1  

#### Abstract
Prefill/decode disaggregation is increasingly adopted in LLM serving to improve the latency-throughput tradeoff and meet strict TTFT and TPOT SLOs. However, LLM inference remains energy-hungry: autoscaling alone is too coarse-grained to track fast workload fluctuations, and applying fine-grained DVF...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BiScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大语言模型（LLM）推理服务中，尽管 **prefill/decode disaggregation**（预填充/解码分离）架构已被广泛采用以改善延迟-吞吐权衡并满足严格的 **TTFT**（Time To First Token）和 **TPOT**（Time Per Output Token）SLO，但其能耗依然很高。传统 **autoscaling** 方法因重配置开销大而无法应对细粒度负载波动，而直接应用 **DVFS**（Dynamic Voltage and Frequency Scaling）在分离架构下面临相位间耦合、动态不对称等挑战。

因此，论文旨在解决以下问题：
> 如何在满足严格 TTFT 和 TPOT SLO 的前提下，最小化 prefill/decode 分离式 LLM 推理系统的能耗？

---

### 提出的新方法与核心思路
作者提出 **BiScale**，一个两层（two-tier）能量优化框架，联合优化 **placement**（资源部署）与 **DVFS** 控制，实现跨时间尺度的协同节能。

#### 创新点：
- **分层控制架构（Hierarchical Control）**：
  - **Tier 1（粗粒度）**：基于预测模型，在每 5 分钟周期内计算 **phase-aware placement** 和 **baseline frequency**，确保在预期峰值负载下仍能满足 SLO。
  - **Tier 2（细粒度）**：在每次迭代级别动态调整 GPU 频率，利用短期负载松弛进一步节能，同时不违反 SLO。

- **阶段感知的 DVFS 策略**：
  - **Prefill 阶段**：采用 **Model Predictive Control (MPC)**，考虑队列演化对未来 TTFT 的影响，进行多批次前瞻优化。
  - **Decode 阶段**：采用轻量级 **slack-aware adaptation**，因其负载平滑且内存带宽受限，无需复杂控制。

- **联合建模与仿真驱动决策**：
  - 构建了基于数据驱动的 **latency model** 和 **power model**，用于 Tier 1 的配置搜索与 Tier 2 的实时控制。
  - 使用迭代级模拟器生成配置表（configuration table），支持 ILP 求解最优 placement。

---

### 相比现有方法的优势
| 方法 | 是否支持 disaggregation | 是否联合优化 placement & DVFS | 是否阶段差异化控制 | 能效提升 |
|------|--------------------------|-------------------------------|--------------------|-----------|
| DistServe | ✅ 是 | ❌ 否（固定高频运行） | ❌ 否 | 基线 |
| DynamoLLM / throttLL’eM | ❌ 否（非分离架构） | ⚠️ 部分支持 | ❌ 否 | 不适用 |
| **BiScale** | ✅ 是 | ✅ 是 | ✅ 是 | **显著更高** |

> BiScale 是首个为 **prefill/decode disaggregation 架构设计的能量高效系统**，实现了 placement 与 DVFS 的跨阶段、跨时间尺度联合优化。

---

## 2. 核心实验方法和设置

### 数据集与工作负载
- **Azure LLM inference trace**：真实生产级请求到达序列，展现多时间尺度的突发性（burstiness）。
- **ShareGPT**：提供典型的输入/输出长度分布。
- 工作负载分为两类：
  1. **可控恒定 RPS 工作负载**：使用 Gamma 分布生成突发请求，平均 RPS 固定。
  2. **真实时变工作负载**：将 Azure trace 时间缩放至目标负载水平（67% 和 85% 容量）。

---

### 实验设置
- **硬件平台**：16×H100 GPU 集群（Nebius Cloud），通过 InfiniBand 连接。
- **模型**：Llama 3.3 70B。
- **SLO 设置**：
  - TTFT ≤ 600 ms（P99）
  - TPOT ≤ 100 ms（P99）

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **TTFT / TPOT** | 延迟 SLO 满足情况（P99） |
| **Energy per token (joules/token)** | 单位输出 token 的能耗，分别统计 prefill 与 decode 阶段 |
| **Average GPU power (W)** | 所有 GPU 平均功耗 |
| **Goodput** | 成功处理且满足 SLO 的请求速率 |

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **DistServe** | 当前最先进的 disaggregated serving 系统，最大化吞吐，所有 GPU 运行在最高频率 |
| **PlaceOnly** | 仅启用 BiScale 的 Tier 1（即 phase-aware placement + 固定频率），无运行时 DVFS |

> BiScale = PlaceOnly + Tier 2 的细粒度 DVFS 控制。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 16×H100 集群上的生产级 trace 测试）

| 指标 | BiScale vs. DistServe | BiScale vs. PlaceOnly |
|------|------------------------|-----------------------|
| **Prefill 能耗降低** | **最多 39%** | +6–10% 进一步降低 |
| **Decode 能耗降低** | **最多 48%** | +3–5% 进一步降低 |
| **TTFT / TPOT SLO 满足情况** | ✅ 全部满足（P99 < SLO） | ✅ 满足（接近边界） |

> 在 85% 负载下，PlaceOnly 曾短暂违反 TPOT SLO（超出 1.5ms），而 BiScale 成功维持合规。

---

### 与基线方法的详细对比
- **DistServe**：
  - 能耗最高，因始终运行在最大频率。
  - 尽管满足 SLO，但能效低下。
- **PlaceOnly**：
  - 显著优于 DistServe（prefill 节能 16–29%，decode 节能 37–45%）。
  - 但由于静态配置，难以应对预测误差导致的过配或欠配。
- **BiScale**：
  - 在 PlaceOnly 基础上进一步节能，尤其在 **prefill 阶段增益更大**（DVFS 贡献约 15% 额外节省）。
  - 动态适应负载变化，纠正 Tier 1 的预测偏差。

---

### 消融实验结果（Ablation Study）
#### （1）Tier 1 与 Tier 2 的贡献分解
- **PlaceOnly（仅 Tier 1）**：
  - 贡献了大部分节能（平均：prefill 20–31%，decode 33%）。
- **Tier 2（DVFS）增量收益**：
  - Prefill：额外 **9–29%** 节能（平均 15%）
  - Decode：额外 **-4% 到 20%**（平均 6%），负值源于 SLO 保护机制下的频率提升

> 结论：**DVFS 的主要价值在于纠正 workload prediction error**，尤其在真实动态负载中效果更明显。

#### （2）不同阶段的节能潜力差异
- **Decode**：
  - 内存带宽受限，频率敏感性低 → DVFS 节能空间小。
  - 负载平滑，batch size 变化慢 → 频率调整机会少。
- **Prefill**：
  - 计算密集，高度频率敏感。
  - 请求到达驱动，bursty → 存在大量短期松弛可被 MPC 捕获。

> 因此，**BiScale 的节能优势主要来自 prefill 阶段的精细控制**。

---

## 4. 关键结论和发现

### 主要发现
1. **Disaggregation 架构下必须联合优化 placement 与 DVFS**：
   - 两个阶段存在强耦合，独立优化会导致次优甚至 SLO 违反。
2. **阶段差异化控制至关重要**：
   - Prefill 适合 MPC 前瞻控制，Decode 适合轻量级 per-batch 调整。
3. **DVFS 是 placement 的有效在线校正机制**：
   - 可补偿 workload prediction 的 over-/under-estimation，提升鲁棒性。
4. **Prefill 阶段是节能的主要突破口**：
   - 因其 compute-bound 特性和高突发性，DVFS 节能潜力远高于 decode。

---

### 方法的局限性
1. **依赖准确的 latency/power 模型**：
   - 虽然模型精度高（MAPE < 5%），但在新模型或新硬件上需重新训练。
2. **MPC 计算开销**：
   - 尽管使用贪心算法将延迟控制在 ~4ms，但仍可能成为瓶颈（尤其更大 batch horizon）。
3. **未考虑 CPU 或网络能耗**：
   - 聚焦 GPU 能耗，系统级节能仍有扩展空间。
4. **过渡期未完全建模**：
   - 实验假设配置切换无中断，实际中需处理实例启停与 KV cache 迁移。

---

### 未来工作方向
1. **端到端 reconfiguration 支持**：
   - 实现无缝的 placement 切换，避免服务中断。
2. **跨节点频率协调**：
   - 在大规模集群中统一调度 DVFS，避免局部热点。
3. **结合 CPU/GPU 协同 DVFS**：
   - 扩展节能范围至整个推理栈。
4. **强化学习替代 MPC**：
   - 探索更高效的在线控制策略，减少对模型预测的依赖。
5. **支持 MoE 或多模型混合负载**：
   - 将 BiScale 思路推广至更复杂的 LLM 服务场景。

---

> **总结**：BiScale 展示了在 disaggregated LLM serving 中，通过 **phase-aware、tiered control** 实现高性能与高能效共存的可能性，为绿色 AI 推理提供了重要实践路径。

</details>

---

### 5. [LLMs Can Learn to Reason Via Off-Policy RL](https://arxiv.org/abs/2602.19362)

**Authors**: Daniel Ritter, Owen Oertell, Bradley Guo, Jonathan Chang, Kiant\'e Brantley, Wen Sun  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.19362v1  

#### Abstract
Reinforcement learning (RL) approaches for Large Language Models (LLMs) frequently use on-policy algorithms, such as PPO or GRPO. However, policy lag from distributed training architectures and differences between the training and inference policies break this assumption, making the data off-policy ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLMs Can Learn to Reason Via Off-Policy RL

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代基于 **Reinforcement Learning (RL)** 的大语言模型（LLM）后训练通常依赖于 **on-policy 算法**（如 PPO 或 GRPO），但在实际分布式训练中存在严重的“**训练-推理不一致**”（training-inference mismatch）问题：

- **Trainer**（如 HuggingFace 模型）和 **Inference Engine**（如 vLLM）即使权重相同，也可能因实现差异输出不同的 log-probabilities。
- 异步训练架构导致推理策略滞后于训练策略，使得采样数据本质上是 **off-policy** 的。

传统方法试图通过 **Importance Sampling (IS)** 或修改推理引擎来“伪装”成 on-policy 数据，但这带来了额外方差、系统复杂性和性能瓶颈。

---

### 🚀 提出的新方法：OAPL
作者提出了一种全新的、完全拥抱 off-policy 特性的算法：

> **Optimal Advantage-based Policy Optimization with Lagged Inference policy (OAPL)**

#### 核心思想：
- 将 off-policy 学习建模为一个 **KL 正则化 RL 问题**，其中目标是最小化训练策略 $\pi$ 与当前推理策略 $T_{\text{vLLM}}$ 之间的 KL 散度，同时最大化奖励。
- 利用该问题的闭式解，推导出一个 **平方回归损失函数**（squared regression objective），直接在来自滞后推理策略的数据上进行优化。
- 不需要 Importance Sampling、clip 操作或删除样本等启发式手段。

#### 算法流程简述（Algorithm 1）：
1. 初始化 Trainer 和 Inference Engine 权重同步。
2. 推理引擎异步生成 rollout 并存入 buffer。
3. Trainer 使用 buffer 中的数据最小化 OAPL 损失（Eq. 3）。
4. 每隔 $L$ 步同步一次两个模块，并清空 buffer。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 GRPO + IS） | OAPL |
|------|--------------------------|------|
| **是否需要 IS** | 是，引入高方差 | 否，避免方差问题 |
| **是否需修改推理引擎** | 部分工作需要（bitwise consistency） | 否，完全兼容标准推理系统 |
| **对 policy lag 的容忍度** | 极低（通常 ≤1 step） | 极高（可达 400+ gradient steps） |
| **训练稳定性** | 易崩溃，熵坍塌常见 | 更稳定，熵保持良好 |
| **样本效率** | 较低 | 提升约 **3x** |
| **理论基础** | 偏离经典 PG 理论 | 基于 KL-regularized RL 的最优解 |

> 💡 **核心洞见**：**On-policy 并非必要**。相反，设计良好的 off-policy 方法可以更高效、更鲁棒地训练推理型 LLM。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 任务 | 数据集 |
|------|--------|
| **数学推理** | `Deepscaler`（训练），测试集：<br>- AIME 2025<br>- HMMT 2025 (Feb & Nov)<br>- BRUMO 2025 |
| **代码生成** | DeepCoder 的训练 prompt 子集（离线两阶段训练）<br>评估使用 **LiveCodeBench v5** 的 279 道题 |

---

### ⚙️ 实验设置

| 设置项 | 描述 |
|-------|------|
| **Base Model** | 数学任务：Qwen3-4B-Thinking-2507<br>代码任务：DeepSeek-R1-Distill-Qwen-14B |
| **最大生成长度** | 数学：16K tokens；代码训练：32K；评估：64K |
| **Policy Lag** | OAPL 设置 $L=50$（数学）、$L \approx 400$（代码）<br>GRPO 仅允许最多 lag 1 步（off-by-one） |
| **同步机制** | OAPL 定期同步并清空 buffer；GRPO 要求严格同步 |

---

### 🎯 评估指标

- **Pass@k**：从 $k$ 个独立 rollout 中至少有一个正确的概率，使用无偏估计器（Chen et al., 2021）
  - 数学任务：Pass@1, 5, 10, ..., 256
  - 代码任务：Pass@1 至 Pass@10
- **训练动态监控**：
  - 序列熵变化（entropy collapse？）
  - 收敛速度与稳定性
- **样本效率**：总训练 generations 数量对比

---

### 🆚 基线方法对比

| 基线 | 说明 |
|-----|------|
| **GRPO + IS** | 当前主流方法，加入 token-level 或 sequence-level Importance Sampling 来修正分布偏差 |
| **DeepCoder** | 公开可用的 GRPO 训练代码模型，包含多种 heuristics（clip-high, overlong filtering 等） |
| **Base Model** | 未经过 RL 微调的原始模型 |

---

## 3. 主要实验结果和性能指标

### 📈 数学推理任务结果（Section 6.1）

#### ✅ 性能全面超越 GRPO
- 图1显示，在所有三个竞赛数学基准上，**OAPL 在 Pass@1/5/10 上均显著优于 GRPO+IS**
- 图2表明，OAPL 收敛更快且更稳定，最终准确率更高

#### ✅ 更好的 Test-Time Scaling（Pass@k 随 k 增大表现更好）
- 图4显示，随着 $k$ 增加，OAPL 的 Pass@k 提升幅度远超 GRPO
- 特别是在 HMMT Nov 2025 上差距明显
- 表明 OAPL **没有导致分布尖锐化（distribution sharpening）**，而是提升了整体解空间探索能力

#### ✅ 抑制 Entropy Collapse
- 图3左图显示：GRPO 的序列熵迅速下降至接近零（collapse），而 **OAPL 保持较高熵值**
- 这解释了为何其在多采样下表现更好 —— 模型仍保留多样性

#### ✅ 对大规模 policy lag 具有强鲁棒性
- 即使将同步间隔拉长到 $L=100$，OAPL 依然稳定学习（图3右）
- 表明其可支持高度异步、大规模分布式训练

---

### 💻 代码生成任务结果（Section 6.2）

#### ✅ 性能匹配甚至略胜 DeepCoder
- 图5左：OAPL 训练的模型在 LiveCodeBench 上的 Pass@k 曲线 **与 DeepCoder 几乎重合或轻微领先**
- 尤其在高 $k$ 区间仍有优势

#### ✅ 样本效率提升约 3 倍
- 图5右：DeepCoder 使用约 **650K generations**
- OAPL 仅用 **~200K generations** 即达到同等性能
- 即便考虑部分短序列训练成本更低，也至少节省 **2x 以上**

> 注：作者未能复现 DeepCoder 原始报告的 60.6% 准确率，推测存在环境或参数差异。

---

### ❌ 消融实验（未明确列出）
论文未提供系统的消融研究（ablation study），但以下观察可视为隐式分析：

- 移除 KL 正则项会导致类似 GRPO 的熵坍塌现象 → 验证 KL 正则的有效性
- 不同 $\beta_1, \beta_2$ 参数搜索发现 $\beta_1=1, \beta_2=1e^{-3}$ 最优 → 参数敏感性较低
- 大 lag 下仍有效 → 验证 off-policy 设计的鲁棒性

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **On-policy 并非必须**：  
   LLM 的 RL 后训练完全可以摆脱对 on-policy 假设的依赖。**off-policy 方法不仅可行，而且更优**。

2. **OAPL 是一种简单高效的 off-policy 替代方案**：  
   - 无需 Importance Sampling
   - 无需修改推理引擎
   - 可容忍超过 **400 步的 policy lag**（比之前工作高出 100x）
   - 实现方式简洁，仅为一个平方损失函数

3. **OAPL 提升了 test-time scaling 和 sample efficiency**：  
   - 在多个 Pass@k 指标上优于 GRPO
   - 在代码任务中以 **1/3 的训练样本量** 达到同等性能

4. **OAPL 抑制 entropy collapse，增强多样性**：  
   - 与许多 RL 方法“仅 sharpen 分布”不同，OAPL 显著改善了大 $k$ 下的表现
   - 支持“RL 能真正增强推理能力”的观点（反驳 Yue et al., 2025）

---

### ⚠️ 局限性

1. **缺乏理论收敛性证明**：  
   虽然基于 KL-regularized RL 的最优解推导而来，但未给出 OAPL 在迭代更新下的收敛保证。

2. **未与其他 off-policy 方法深入比较**：  
   如 DDPG-style Q-learning 或 Actor-Critic 架构在 LLM 中的应用潜力未探讨。

3. **超参数选择有限探索**：  
   尤其在代码任务中未进行广泛调参，可能未达最优性能。

4. **应用场景目前集中于 reasoning 任务**：  
   是否适用于对话、摘要等其他 RLHF 场景尚待验证。

---

### 🔮 未来工作方向

1. **扩展至 value-based off-policy 方法**：  
   如训练 Q-function 或 V-function 来进一步改进 credit assignment。

2. **结合 offline data 进行混合训练**：  
   利用人造数据、人类反馈数据等静态数据集提升训练效率。

3. **应用于更大规模模型和更多任务类型**：  
   验证 OAPL 在百亿级以上模型上的可扩展性。

4. **构建统一框架处理 on-/off-policy 混合数据**：  
   如 Tang et al. (2025) 所示，现实系统常同时拥有两类数据，如何统一优化值得研究。

---

## ✅ 总结一句话

> **OAPL 证明了：与其费力修复 off-policy 问题，不如彻底拥抱它 —— 一个基于最优优势回归的简单 off-policy 算法，能在数学推理与代码生成任务中以更高效率、更强稳定性、更好泛化能力超越主流 on-policy 方法。**

</details>

---

### 6. [Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning](https://arxiv.org/abs/2602.19917)

**Authors**: Thanh Nguyen, Tung Luu, Tri Ton, Sungwoong Kim, Chang D. Yoo  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.19917v1  

#### Abstract
Offline reinforcement learning (RL) has garnered significant interest due to its safe and easily scalable paradigm. However, training under this paradigm presents its own challenge: the extrapolation error stemming from out-of-distribution (OOD) data. Existing methodologies have endeavored to addres...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Offline Reinforcement Learning（Offline RL）面临的核心挑战是**外部分布（Out-of-Distribution, OOD）数据引发的外推误差（extrapolation error）**。由于策略在训练过程中无法与环境交互，当评估未在行为策略（behavior policy）数据中出现过的状态-动作对时，Q函数容易高估其价值，导致学习到次优甚至不稳定策略。

现有方法如CQL、BEAR等通过保守估计或约束策略接近行为策略来缓解该问题，但存在以下缺陷：
- 对OOD数据处理过于保守（uniform penalty）
- 不确定性量化不精确
- 使用Q-ensemble带来高昂计算和内存开销

---

### 提出了什么新方法或新思路
本文提出了一种全新的 **Uncertainty-Aware Rank-One MIMO Q Network Framework**，结合两个核心创新：

#### （1）Rank-One MIMO Q Network 架构
- 设计一种多输入多输出（Multi-Input Multi-Output, MIMO）网络结构，模拟传统Q-ensemble的效果。
- 每个“成员”由一个共享主干网络（shared network）和一对低秩向量（rank-one vectors $v_k$, $s_k$）构成，实际权重为：
  $$
  W_k = W \odot (v_k s_k^T)
  $$
  其中 $\odot$ 表示逐元素乘法。
- 所有成员共享大部分参数，仅需额外存储 $K$ 对向量，显著降低参数量和内存占用。

#### （2）基于不确定性的悲观训练损失（Pessimistic Training Losses）
- 利用最小Q头输出近似 **Lower Confidence Bound (LCB)** 来构建目标Q值：
  $$
  \mathcal{T}Q(s,a) = R + \gamma \mathbb{E}_{s'\sim D, a'\sim\pi} \left[ \min_k Q_{\theta'}(s',a') - \alpha \log \pi(a'|s') \right]
  $$
- 在策略优化阶段引入三项联合目标：
  - 最小化Q值（鼓励保守性）
  - 最大化策略熵（提升探索鲁棒性）
  - 最大化数据集中动作的似然（favor in-distribution actions）

此外还引入：
- **Lazy Policy Improvement**：减少策略更新频率以提高稳定性
- **无需OOD采样机制**：避免额外计算负担

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **精度** | 显式建模不确定性，实现更精细的OOD惩罚（非uniform），优于CQL等方法 |
| **效率** | 参数量和计算成本接近单个网络，远低于naive ensemble（如PBRL、EDAC） |
| **内存友好** | 内存占用仅为ensemble方法的一小部分（见Table 2） |
| **训练速度** | 单epoch运行时间最短，支持快速迭代 |
| **通用性** | 不依赖行为策略估计，适用于各种覆盖程度的数据集（random到expert） |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
在 **D4RL benchmark** 上进行验证，涵盖三个MuJoCo连续控制任务：
- **HalfCheetah**
- **Hopper**
- **Walker2d**

每项任务使用五类不同行为策略生成的数据集：
- `random-v2`
- `medium-v2`
- `medium-replay-v2`
- `medium-expert-v2`
- `expert-v2`

---

### 实验设置和评估指标
- **训练步数**：共3000 epochs，每个epoch 1000 steps → 总计3M steps
- **评估方式**：
  - 每个算法评估10条轨迹，每条1000步
  - 报告4个随机种子下的平均表现
- **评估指标**：
  - **Normalized Score**（归一化得分）：
    $$
    \text{score}_{\text{normalized}} = 100 \times \frac{\text{score} - \text{score}_{\text{random}}}{\text{score}_{\text{expert}} - \text{score}_{\text{random}}}
    $$
  - 平均归一化得分用于横向比较

---

### 基线方法对比
与多种state-of-the-art方法对比：
| 方法 | 类型 | 特点 |
|------|------|------|
| **BCQ** | Policy Constraint | 动作受限于行为策略附近 |
| **IQL** | Implicit Q-Learning | 避免直接查询OOD Q值 |
| **BEAR** | MMD约束 | 强制分布匹配 |
| **UWAC** | Dropout + Uncertainty | 不确定性加权更新 |
| **CQL** | Conservative Q-Learning | 对OOD动作施加统一惩罚 |
| **MOPO** | Model-Based | 基于模型的不确定性惩罚 |
| **TD3-BC** | Hybrid | TD3 + BC正则项 |
| **EDAC** | Ensemble-based | 多样化Q-ensemble |
| **PBRL** | Bootstrapped + OOD Sampling | 显式采样OOD动作并惩罚 |

> 注：EDAC 和 PBRL 是与本方法最相关的基线，均采用Q-ensemble进行保守学习。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
| 方法 | Average Normalized Score |
|------|--------------------------|
| BCQ | 49.4 |
| IQL | 66.5 |
| BEAR | 38.78 |
| UWAC | 50.41 |
| CQL | 67.35 |
| MOPO | 53.3 |
| TD3-BC | 67.55 |
| EDAC | 71.2 |
| PBRL | 74.37 |
| **Ours** | **83.6** ✅ |

> **结论**：本文方法在平均得分上达到SOTA，领先第二名PBRL达 **+9.23分**，几乎是BCQ的两倍。

---

### 与基线方法的对比结果
- 在多数任务中表现最优，尤其在**messy数据**（如`random`, `medium-replay`）上有显著优势
- 在`expert`数据集中也保持稳定高性能
- 相较于model-based方法（如MOPO），不受动态建模误差影响，在复杂环境中更具鲁棒性
- 相较于policy-constrained方法（如BCQ、BEAR），摆脱了对行为策略的依赖，能学到更优策略

---

### 消融实验结果（Ablation Study）

#### （1）组件消融（Table 4）
在 `walker2d-medium-expert` 数据集上的结果：

| 配置 | Avg Return | Avg Q(s,π(s)) |
|------|------------|---------------|
| 完整框架（Entropy + Likelihood） | **112.9** | 373.4 |
| 仅Entropy | 111.0 | 258.9 |
| 仅Likelihood | 107.3 | 267.1 |
| 无附加项 | 109.6 | 453.2 |

> **发现**：
> - 同时使用熵最大化和似然最大化效果最佳
> - 两者都引入一定“悲观性”，防止Q值过度膨胀
> - 尤其在expert数据中，Likelihood项对稳定性至关重要

#### （2）参数 $K$（ensemble size）的影响（Table 3）
| K | Avg Return |
|----|------------|
| 2 | 0.19 |
| 5 | 92.9 |
| 10 | **112.8** |
| 15 | 23 |
| 20 | 0.4 |

> **发现**：
> - $K$ 控制悲观程度：太小→过于乐观；太大→过拟合悲观
> - 存在一个最优区间（约 $K=10$），平衡exploration与exploitation

#### （3）不确定性可视化（Figure 4）
- 在合成回归任务中展示：随着输入远离训练分布（从[-7,7]扩展至[-10,10]），预测不确定性平滑上升
- 表明Rank-One MIMO能够有效识别OOD区域并提供可靠置信度估计

---

## 4. 关键结论和发现

### 论文的主要发现
1. **不确定性感知是解决Offline RL外推误差的关键路径**  
   显式建模Q函数的不确定性，并据此构造LCB目标，可有效抑制OOD动作的价值高估。

2. **高效的Ensemble架构设计至关重要**  
   Rank-One MIMO在几乎不增加计算成本的前提下实现了与完整ensemble相当的不确定性量化能力，打破了“高性能=高代价”的固有认知。

3. **无需OOD采样即可实现高效保守学习**  
   通过最小Q头机制自然聚焦高不确定性动作，避免了CQL/PBRL中的昂贵采样过程。

4. **灵活性 + 稳定性兼备**  
   方法既不限制策略形式，也不依赖行为策略估计，在各类数据集上均表现出色且收敛稳定。

---

### 方法的局限性
- 当前框架假设Q函数输出服从近似高斯分布，可能在极端非对称不确定性下失效
- Rank-One结构虽高效，但在极深网络中可能限制表达能力多样性
- 超参数 $K$ 需要调优，自动选择机制尚未研究
- 当前仅验证于连续控制任务，离散动作空间或高维视觉输入场景有待测试

---

### 未来工作方向
1. 探索自适应调整 $K$ 或动态路由机制
2. 将该框架拓展至 **Model-Based Offline RL** 中的动力学不确定性建模
3. 结合 **causal reasoning** 进一步区分真实OOD与虚假不确定性
4. 应用于真实世界场景（如机器人控制、医疗决策）中的安全策略学习
5. 探索将MIMO思想应用于Policy Network本身，构建Uncertainty-Aware Actor

---

> **总体评价**：本文提出了一种兼具**高性能、高效率、强鲁棒性**的Offline RL新范式，为大规模部署强化学习提供了实用而可靠的解决方案。其核心思想——“共享知识+个性化微调”的MIMO架构，有望成为未来高效深度学习系统的重要组成部分。

</details>

---

### 7. [IAPO: Information-Aware Policy Optimization for Token-Efficient Reasoning](https://arxiv.org/abs/2602.19049)

**Authors**: Yinhan He, Yaochen Zhu, Mingjia Shi, Wendy Zheng, Lin Su, Xiaoqing Wang, Qi Guo, Jundong Li  
**Category**: cs.CL  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.19049v1  

#### Abstract
Large language models increasingly rely on long chains of thought to improve accuracy, yet such gains come with substantial inference-time costs. We revisit token-efficient post-training and argue that existing sequence-level reward-shaping methods offer limited control over how reasoning effort is ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：IAPO: Information-Aware Policy Optimization for Token-Efficient Reasoning

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在进行复杂推理时，通常依赖于长链的思维过程（Chain-of-Thought, CoT）来提高准确性。然而，这种策略导致了显著的推理成本增加，尤其是在生成冗余、循环或无信息量的推理步骤时。现有的基于强化学习（RL）的后训练方法（如 GRPO）虽然能提升推理能力，但往往会产生不必要的冗长输出，从而增加了计算开销和延迟。

本文指出，现有方法在优化推理效率时存在一个根本缺陷：它们是**内容无关（content-agnostic）**的。具体来说：
- **基于长度的方法**（Length-based）对所有短输出中的 token 统一给予高优势值（advantage），而不考虑其是否真正“有用”。
- **基于位置的方法**（Position-based）通过位置衰减来惩罚后续 token，即使这些 token 可能对最终答案至关重要。

因此，核心问题是：**如何在不损害准确性的前提下，减少推理过程中不必要的 token 生成？**

### 提出的新方法和新思路
为了解决上述问题，作者提出了 **IAPO (Information-Aware Policy Optimization)**，一种新颖的 token 级别优势塑造框架。

#### 核心创新点
1.  **信息感知的优势塑造（Information-Aware Advantage Shaping）**：
    *   IAPO 的核心思想是，每个 token 的优势值应由其对最终答案正确性的**信息贡献度**决定。
    *   该贡献度通过**条件互信息（Conditional Mutual Information, MI）** `I(y; o_t | q, o_<t)` 来量化。这表示在给定查询 `q` 和之前已生成的 token `o_<t` 的条件下，观察到当前 token `o_t` 后，关于最终答案 `y` 的不确定性减少了多少。
    *   这种机制提供了一个**有原则的（principled）** 方式来识别信息丰富的推理步骤，并抑制低效探索。

2.  **高效的条件 MI 估计模块（Efficient Conditional MI Estimation）**：
    *   直接计算条件 MI 在计算上是不可行的。为此，作者设计了一套高效的技术：
        *   **早期退出估计器（Early-Exit-Based Estimator）**：通过向模型输入添加一个特殊的后缀提示（如 `</think><answer>`），强制模型立即预测最终答案，从而估算出生成 `o_t` 前后的答案熵差 `H(y|q,o_<t) - H(y|q,o_<t')`，以此作为 MI 的近似。
        *   **KV-Cache 预加载**：避免对共享前缀重复进行前向传播，将计算复杂度从 `O(L^3d)` 显著降低。
        *   **分块前向传播（Chunk-wise Forwarding）**：批量处理一个 chunk 内的所有 token，进一步摊销计算成本。

3.  **探索调整项（Exploration Adjustment）**：
    *   为了防止模型因过度追求信息量而陷入局部最优（即过早收敛到过于简洁但错误的路径），IAPO 引入了一个探索调整项。
    *   对于正确的完成，它鼓励模型坚持其有信心的生成；对于错误的完成，它则鼓励模型探索那些它原本不太可能选择的替代路径。

### 相比现有方法的优势
- **更智能的控制**：相比简单的“越短越好”或“越早越好”，IAPO 能够区分**有用的推理**和**无用的冗余**，实现更精细的控制。
- **理论保证**：论文提供了理论分析，证明 IAPO 能够在保持模型准确性的前提下，单调地减少推理长度。
- **实际效果显著**：实验表明，IAPO 在大幅减少推理长度的同时，还能提升或至少保持推理准确率。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个具有代表性的数学推理数据集上进行：
- **GSM8K** (Cobbe et al., 2021)：小学水平的算术应用题，需要多步数值推理。
- **MATH-500** (Lightman et al., 2023)：竞赛级别的数学问题，难度更高。
- **DAPO-Math-17k** (Yu et al., 2025)：大规模、多样化的数学数据集，解决方案通常比前两者更长。

### 实验设置和评估指标
- **基础模型**：主要在 `Qwen2.5` 系列模型（0.5B, 1.5B, 7B 参数规模）上进行后训练。
- **评估指标**：
    - **Pass@k**：在 k 次采样中，模型至少有一次答对的百分比，衡量有效性（effectiveness）。
    - **Length@k**：k 次采样的平均生成长度，衡量效率（efficiency）。
    - **Ratio@k (Pass@k / Length@k)**：这是本文提出的核心指标，用于衡量**token 效率**，即每个 token 产生正确答案的效能。

### 基线方法对比
与以下最先进的 token 高效 RL 后训练方法进行比较：
- **DAPO** (Yu et al., 2025)：对过长序列分配零优势。
- **GFPO** (Shrivastava et al., 2025)：只奖励最短的完成。
- **GTPO** (Tan et al., 2025)：奖励正确完成中的高熵 token，惩罚错误完成中的低熵 token。
- **S-GRPO** (Lee & Tong, 2025)：对超过某个位置阈值的 token 分配零优势。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **推理长度显著减少**：在 `Qwen2.5-7B-Instruct` 模型上，IAPO 相比基线模型，**推理长度减少了高达 36%**（例如，在 GSM8K 上，Length@32 从约 175 降至 112）。
- **推理准确率保持或提升**：在多个设置下，IAPO 不仅没有牺牲准确率，反而实现了更高的 `Pass@k`。例如，在 `Qwen2.5-1.5B-Instruct` + `MATH-500` 上，`Pass@32` 从 0.7333 提升至 0.7556。
- **token 效率最优**：在所有实验设置中，IAPO 的 `Ratio@k` 指标几乎总是达到**最佳或次佳**，证明了其在单位 token 成本下的卓越性能。

### 与基线方法的对比结果
- **全面超越**：IAPO 在 `Pass@k`、`Length@k` 和 `Ratio@k` 三个维度上的综合表现均优于所有基线方法。
- **案例研究**：图 7 展示了一个典型案例，对于同一个问题，基线方法（如 GFPO, GTPO）会生成 51-105 个 token 的冗长推理，而 IAPO 仅用 **15 个 token** 就得出了正确答案，体现了其极高的信息密度。

### 消融实验结果
- **移除信息感知项（IAPO-NI）**：当移除基于条件 MI 的优势分配时，token 效率明显下降，证明了该模块的有效性。
- **替换为其他信息度量（IAPO-NE）**：如果用“下一个 token 的熵减少”来代替条件 MI，性能虽优于 IAPO-NI，但仍不如使用精确条件 MI 估计的完整 IAPO，说明了所提估计方法的优越性。

---

## 4. 关键结论和发现

### 主要发现
1.  **信息论是解决 token 效率问题的关键**：直接量化每个 token 对最终答案的信息贡献，是引导模型生成更精简、更高效推理的强有力且普适的方向。
2.  **IAPO 是有效的**：该方法成功地在不牺牲甚至提升准确率的前提下，显著降低了推理长度，实现了真正的“token 高效”。
3.  **理论与实践一致**：理论分析（协方差为负）与实验结果（推理长度减少）相互印证，为方法的有效性提供了坚实的理论基础。

### 方法的局限性
- **计算开销**：尽管引入了 KV-Cache 预加载和分块技术，但实时估计每个 token 的条件 MI 仍然带来了额外的计算负担，使得训练过程比标准 GRPO 更慢。
- **估计误差**：早期退出估计器是一种近似方法，其估计的 MI 值可能存在偏差，影响优势值的准确性。

### 未来工作方向
- **更高效的估计器**：开发计算成本更低、估计更准确的条件 MI 估计算法。
- **扩展到更多任务**：将 IAPO 应用于非数学推理任务（如常识推理、代码生成等），验证其通用性。
- **结合其他技术**：探索将 IAPO 与隐式推理（implicit reasoning）或模型压缩技术相结合，以实现更极致的效率提升。

</details>

---

### 8. [Online decoding of rat self-paced locomotion speed from EEG using recurrent neural networks](https://arxiv.org/abs/2602.18637)

**Authors**: Alejandro de Miguel, Nelson Totah, Uri Maoz  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.18637v1  

#### Abstract
$\textit{Objective.}$ Accurate neural decoding of locomotion holds promise for advancing rehabilitation, prosthetic control, and understanding neural correlates of action. Recent studies have demonstrated decoding of locomotion kinematics across species on motorized treadmills. However, efforts to d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Online decoding of rat self-paced locomotion speed from EEG using recurrent neural networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本研究旨在解决**在自然、自选节奏（self-paced）条件下，非侵入式连续解码大鼠运动速度**的挑战。现有研究多集中于：
- 外部强制控制的跑步机速度（enforced treadmill speed）
- 依赖侵入式记录（如单神经元、ECoG）
- 解码离散行为状态（如行走/静止）而非连续速度

本文首次实现了从**非侵入式、全皮层范围的EEG信号中，对自选节奏下的连续运动速度进行高精度在线解码**。

### 🚀 提出的新方法与创新思路
- **异步脑机接口（asynchronous BCI）框架**：直接处理连续流式的EEG数据，无需任务标记或试次分割，更贴近真实行为场景。
- **基于LSTM的RNN模型用于端到端回归**：将32通道EEG时间窗映射为瞬时速度输出，捕捉时空动态特征。
- **探索“前瞻性”与“回顾性”解码能力**：首次系统评估EEG是否编码未来和过去的速度信息，揭示神经信号的时间嵌入特性。
- **跨会话与跨个体迁移学习分析**：验证预训练模型能否零样本迁移或通过微调快速适应新数据。

### 🔍 相比现有方法的优势
| 维度 | 传统方法局限 | 本文优势 |
|------|--------------|--------|
| **记录方式** | 多为侵入式（single-unit, ECoG） | 使用**非侵入式颅表EEG**，更具临床转化潜力 |
| **行为范式** | 强制速度、固定区块设计 | 在**自选节奏、无外部指令**下采集数据，生态效度更高 |
| **解码目标** | 多为分类任务（快/慢、走/停） | 实现**连续速度回归**，支持精细运动控制 |
| **模型架构** | 线性解码器为主 | 采用**深度RNN模型**，显著提升性能（R²达0.78） |
| **泛化能力** | 缺乏跨会话/个体评估 | 发现**个体内高度可迁移，个体间需微调** |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **来源**：14只雄性大鼠，共225个有效记录会话（原始276会话，经筛选后保留81.5%）
- **总时长**：超过 **133小时** 的连续EEG与跑步机速度同步数据
- **样本量**：近 **4800万对齐的神经-行为样本**
- **EEG配置**：
  - 32通道颅表EEG阵列（skull-surface EEG），覆盖前额叶至视觉皮层
  - 频率范围：0.01–45 Hz（离线滤波后）
  - 采样率：100 Hz（降采样后）
- **行为任务**：
  - 头固定大鼠在非电机驱动跑步机上完成Go/NoGo视觉辨别任务
  - “Go”反应要求跑过距离阈值，速度由动物自主决定 → 形成丰富的自选速度分布

### ⚙️ 实验设置
- **输入格式**：滑动窗口200 ms（20个时间步），每次前进10 ms → 准实时模拟在线解码
- **输入维度**：32通道 × 20时间步 → 输入形状为 `20×32`（time-major）
- **输出目标**：当前时刻的瞬时跑步机速度（连续值，任意单位）

### 🎯 评估指标
- 主要指标：
  - **Pearson相关系数（r）**
  - **决定系数 R²**
- 统计检验：
  - Shapiro-Wilk检验正态性
  - Friedman检验 + Bonferroni校正的Wilcoxon符号秩检验（非参数重复测量）

### 🆚 基线方法对比
比较了五种回归模型：
1. **Linear Regression**（线性回归）
2. **Random Forest**（随机森林）
3. **Feed-Forward Neural Network**（前馈网络）
4. **RNN with LSTM units**（LSTM循环网络）✅ 最优
5. **Encoder-Only Transformer**（仅编码器Transformer）

所有模型均在同一数据划分策略下训练与测试。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
| 模型 | 中位数 r | 中位数 R² |
|------|---------|----------|
| Linear Regression | 0.64 | 0.39 |
| Random Forest | 0.76 | 0.56 |
| Feed-Forward NN | 0.87 | 0.76 |
| Encoder-Only Transformer | 0.87 | 0.77 |
| **RNN (LSTM)** ✅ | **0.88** | **0.78** |

> 💡 **RNN表现最优且显著优于其他模型**（除Transformer外，其余p < 0.0001；vs. Transformer R²差异p=0.0374）

- 在82.7%的会话中，RNN解码性能超过此前报道的最佳水平（r > 0.80）
- 5%的会话达到近乎完美解码（R² > 0.90）

---

### 🔁 迁移学习与零样本泛化实验（Transfer Learning）

| 训练策略 | r | R² | 说明 |
|--------|----|-----|------|
| 单一会话（80%训练） | 0.88 | 0.78 | 上限基准 |
| 同一个体不同会话（zero-shot） | 0.61 | 0.25 | 可行但性能下降明显 |
| 不同个体之间（zero-shot） | 0.24 | -0.39 | 表现差于均值预测（无效） |
| 同一个体 + 微调10%数据 | 0.79 | 0.58 | 显著提升，接近理想水平 |
| 不同个体 + 微调10%数据 | 0.73 | 0.49 | 与从头训练10%数据相当 |

> ✅ **结论**：  
> - 神经运动签名具有**强个体特异性**，难以跨个体直接迁移  
> - 但**同个体会话间存在稳定模式**，支持高效迁移学习  
> - **少量微调（10%数据）即可大幅提升性能**，适用于实际BCI部署

---

### 🧠 消融实验结果（Ablation Studies）

#### （1）空间贡献分析（Spatial Contribution）
- **视觉皮层（visual cortex）电极单独使用**：
  - r = 0.85, R² = 0.72 → 接近全电极性能（0.88 / 0.78）
- **结合视觉+运动区（motor/somatomotor）**：
  - 性能达到最高水平（r ≈ 0.88）
- **仅用运动相关区域（不含视觉）**：
  - 平均 r ≈ 0.84, R² ≈ 0.70 → 明显低于含视觉区组合

> 🔍 **发现**：**视觉皮层是解码速度最主要的贡献者**，远超传统关注的运动皮层。

#### （2）频谱贡献分析（Spectral Contribution）
| 频段 | r | R² | 贡献程度 |
|------|----|-----|--------|
| Delta (1–4 Hz) | 0.88 | 0.76 | 极高 |
| Theta (4–8 Hz) | 0.85 | 0.72 | 极高 |
| Alpha (8–13 Hz) | 0.82 | 0.66 | 中等 |
| Beta (13–30 Hz) | 0.77 | 0.57 | 较低 |
| Gamma (>30 Hz) | 0.61 | 0.35 | 很低 |

> ✅ **低频段（<8 Hz）主导了解码性能**，尤其是delta和theta波段。

#### （3）前瞻性与回顾性解码（Prospective & Retrospective Decoding）
- 比较三种模型对未来/过去速度的预测能力：
  1. EEG-based RNN
  2. Speed-only autoregressive RNN
  3. 速度自相关（autocorrelation）

| 时间偏移 | EEG-based r | Speed-based r | 差异趋势 |
|--------|-------------|---------------|---------|
| +1000 ms（未来） | 0.34 | ~0.20 | EEG更优 |
| -1000 ms（过去） | 0.59 | ~0.20 | EEG显著更优 |

> 🔍 **发现**：
> - EEG能编码**未来最多1秒内的速度意图**
> - 对**过去速度的记忆重建能力更强**（不对称性）
> - 超出约400 ms后，EEG模型优于纯运动动力学模型 → 表明神经信号提供额外信息

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **非侵入式EEG可实现高精度连续运动速度解码**：
   - R²达0.78，r=0.88，为目前**自选节奏运动解码中的最高精度之一**
2. **视觉皮层在运动解码中起主导作用**：
   - 尽管任务涉及视觉刺激，但速度变化仍主要由视觉区EEG编码
   - 支持“运动-感知耦合”理论：即使无视觉输入，运动也会调制视觉皮层活动
3. **低频振荡（<8 Hz）是关键特征**：
   - Delta与theta波段携带最丰富运动信息，可能反映觉醒状态与加速度调控
4. **神经信号具有时间延展性**：
   - 当前EEG不仅反映当前速度，还能预测未来1秒内速度，并重构过去1秒历史
   - 揭示大脑对运动的**前瞻规划与记忆维持机制**
5. **个体内部神经模式高度一致，跨个体差异大**：
   - 支持**基于迁移学习的快速校准策略**，减少每次实验的训练负担

---

### ⚠️ 方法的局限性
1. **头固定动物模型**：
   - 虽然生态效度高于强制跑步机，但仍受限于自由度，不能完全代表自然行走
2. **EEG空间分辨率有限**：
   - 颅表EEG易受容积传导影响，无法精确定位深层源
3. **未完全消除视觉线索干扰**：
   - 尽管分析表明非瞬态响应主导，但Go/NoGo任务结构可能间接影响神经模式
4. **输出平滑性不足**：
   - 原始解码输出波动较大，需后续滤波处理，引入延迟风险

---

### 🔮 未来工作方向
1. **实现实时闭环控制系统**：
   - 将解码器集成到在线BCI平台，驱动外骨骼或虚拟代理
2. **融合多模态信号**（如EMG、fNIRS）提升鲁棒性
3. **开发因果预测模块**：
   - 利用前瞻性解码补偿系统延迟，提高响应性
4. **扩展至人类EEG研究**：
   - 验证类似神经机制是否存在于人脑，推动康复BCI发展
5. **探索神经机制背后的认知过程**：
   - 如注意力、决策、动机如何调节运动相关的EEG动态

---

> 🏁 **总结一句话**：  
> 该研究证明，**仅凭非侵入式EEG即可高精度解码大鼠自选运动速度**，其关键信息来源于**视觉皮层的低频振荡**，并可通过**RNN建模实现前瞻性预测与跨会话迁移**，为下一代自然化、高性能BCI提供了重要范式和技术基础。

</details>

---

### 9. [Fully Convolutional Spatiotemporal Learning for Microstructure Evolution Prediction](https://arxiv.org/abs/2602.19915)

**Authors**: Michael Trimboli, Mohammed Alsubaie, Sirani M. Perera, Ke-Gang Wang, Xianqi Li  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.19915v1  

#### Abstract
Understanding and predicting microstructure evolution is fundamental to materials science, as it governs the resulting properties and performance of materials. Traditional simulation methods, such as phase-field models, offer high-fidelity results but are computationally expensive due to the need to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Fully Convolutional Spatiotemporal Learning for Microstructure Evolution Prediction*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于**phase-field simulation**的微结构演化预测虽然精度高，但需要求解复杂的非线性偏微分方程（PDEs），计算成本极高，尤其在长时序、高分辨率模拟中难以满足实际应用需求（如参数扫描、材料设计优化等）。因此，亟需一种高效且准确的替代方案。

### 提出的新方法与新思路
本文提出了一种**全卷积时空学习框架**（fully convolutional spatiotemporal learning framework），基于 **SimVPv2 架构**，用于加速微结构演化预测。其核心思想是：
- 将微结构演化建模为一个**时空序列预测任务**（spatiotemporal forecasting）。
- 使用**完全卷积网络**（fully convolutional network）取代传统的循环神经网络（RNN）、Transformer 或图神经网络（GNN）等架构。
- 模型采用**自监督训练方式**，直接从 phase-field 模拟生成的图像序列中学习演化规律。

该模型由三个模块组成：
1. **Spatial Encoder**：通过 depth-wise 和 dilated convolutions 提取空间特征；
2. **Temporal Translator**：引入 **gSTA**（gated Spatiotemporal Attention）模块捕捉时间动态；
3. **Spatial Decoder**：重建未来时刻的微结构场。

### 相比现有方法的优势
| 方法类别 | 局限性 | 本文优势 |
|--------|------|---------|
| **Recurrent Models**（如 ConvLSTM, PredRNN++） | 序列依赖导致无法并行化，训练不稳定，误差随时间累积 | 支持**并行多步预测**，显著提升推理速度与稳定性 |
| **Operator Learning**（如 FNO） | 依赖全局傅里叶变换，在局部界面主导的异质系统中表现不佳 | 采用**局部卷积注意力机制**，更契合微结构演化的物理本质 |
| **Physics-Informed NNs (PINNs)** | 难以扩展到高维非线性系统，训练困难 | 不依赖显式物理约束，纯数据驱动，易于训练 |
| **Graph Neural Networks (GNNs)** | 需要构建动态图结构，丢失像素级细节 | 直接操作于密集场数据，保留精细界面信息 |

> ✅ **核心优势总结**：  
> - **高效率**：训练与推理速度快，FLOPs 显著低于基线；  
> - **强泛化能力**：支持跨分辨率（64×64 → 256×256）、跨时间长度推广；  
> - **物理一致性**：能捕捉长期统计特性（如 grain size distribution）而不仅是短期纹理匹配。

---

## 2. 核心实验方法和设置

### 使用的数据集
两个典型的微结构演化过程被用于训练与测试：
1. **Grain Growth**（晶粒生长）
   - 基于 multi-order-parameter phase-field model 模拟；
   - 初始结构为 Voronoi 背景下的 100 个随机分布晶粒；
   - 时间步长 Δt = 0.2，空间分辨率 64×64（训练），256×256（测试）；
   - 数据集包含约 1,070 条轨迹，每条含 200 帧。

2. **Spinodal Decomposition**（旋节分解）
   - 基于 Cahn-Hilliard 方程模拟；
   - 参数设定符合热力学不稳定性条件；
   - 使用 COMSOL Multiphysics 求解器生成数据；
   - 同样包含低分辨率训练集（64×64）和高分辨率测试集（256×256）。

### 实验设置
- **输入输出格式**：输入 T=10 帧历史状态，预测未来 T’=90 帧；
- **滑动窗口采样**：从长序列中提取重叠片段进行训练；
- **零填充策略**：支持少于 10 帧的输入（如 5→95 设置）；
- **迭代 rollout**：实现超长时序预测（如 10→190）；
- **无再训练跨尺度推理**：模型在 64×64 上训练，直接应用于 256×256 测试。

### 评估指标
| 指标 | 描述 |
|-----|------|
| **RMSE** | 像素级均方根误差，衡量数值偏差 |
| **SSIM** | 结构相似性指数，反映视觉保真度 |
| **GSD (Grain Size Distribution)** | 统计晶粒尺寸分布，验证是否保持物理合理的粗化行为 |
| **Particle Growth/Shrink Tracking** | 对 spinodal 分解中的特定区域追踪面积变化趋势 |

### 基线方法对比
- **ConvLSTM**
- **PredRNN++**
- （隐式对比其他文献中的 FNO、PINN、GNN 等）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### Grain Growth（10→90）
- **平均 RMSE < 0.11**
- **平均 SSIM > 0.86**
- GSD 与真实值高度一致，尤其在 t=25 和 t=100 时几乎重合
- 平均晶粒面积增长趋势线性良好，斜率接近 ground truth

#### Spinodal Decomposition（10→90）
- **平均 RMSE < 0.10**
- **平均 SSIM > 0.92**
- 成功复现双连续相与液滴形态
- 粒子生长/收缩速率预测准确，动态演化路径对齐良好

#### 长时序外推能力（10→190 / 10→200）
- 尽管误差随时间缓慢上升，但：
  - **RMSE < 0.18（grain growth）**
  - **SSIM > 0.5（grain growth）**
  - **SSIM > 0.85（spinodal decomposition）**
- 微观结构仍保持合理拓扑（无伪影或崩溃）
- 平均晶粒面积演化趋势依然吻合

#### 少量观测鲁棒性（5→95）
- 输入仅 5 帧 + 5 帧零填充
- **RMSE < 0.10，SSIM > 0.88**
- GSD 保持高度一致
- 表明模型对输入长度不敏感，具备良好鲁棒性

### 与基线方法的对比结果（Table 1）

| 模型 | 推理时间 | FLOPs |
|------|----------|-------|
| **本文方法** | **30 秒** | **9.327G** |
| ConvLSTM | 8分48秒 | 1.168T |
| PredRNN++ | 22分14秒 | 3.552T |

> ⚡️ **推理加速比**：
> - 较 ConvLSTM 快 **17.6×**
> - 较 PredRNN++ 快 **44.5×**

### 消融实验（隐含分析）
尽管未明确列出消融表，文中通过以下设置间接验证了设计有效性：
- **gSTA 模块的作用**：局部因果注意力机制有效抑制误差传播，增强长期稳定性；
- **全卷积结构的价值**：无需修改即可处理更高分辨率输入，体现“resolution-agnostic”特性；
- **并行预测 vs 自回归**：单次前向传播完成多步预测，避免隐藏状态累积错误。

---

## 4. 关键结论和发现

### 主要发现
1. **全卷积架构可高效建模复杂微结构演化动力学**，在 grain growth 与 spinodal decomposition 中均取得优异性能。
2. 所提模型不仅能预测短期局部变化，还能**捕捉长期统计规律**（如 GSD 演化、粒子粗化趋势），表明其学到的是潜在物理机制而非简单记忆。
3. 模型展现出强大的**跨尺度泛化能力**：在低分辨率数据上训练后，可直接用于高分辨率预测，无需微调。
4. **推理效率极大提升**，相比主流 RNN 架构实现数十倍加速，适合部署于大规模仿真或数字孪生系统。

### 方法的局限性
1. **缺乏显式物理约束**：虽能拟合数据，但未强制遵守质量守恒、能量衰减等基本物理律，极端条件下可能偏离真实物理。
2. **迭代 rollout 存在误差积累风险**：尽管当前表现稳定，但在极长时间预测下仍可能出现发散。
3. **任务专用性较强**：目前分别训练 grain growth 与 spinodal decomposition 模型，尚未实现统一的通用预测器。
4. **评价指标偏图像层面**：缺少更深入的物理诊断工具（如 structure factor、interfacial curvature 统计）来全面评估物理合理性。

### 未来工作方向
1. **开发物理感知损失函数**：引入弱物理正则项（如自由能单调递减约束）以增强模型可靠性。
2. **探索统一条件建模范式**：构建可接受 material parameters 或 initial state descriptor 的通用模型，支持跨机制预测。
3. **集成不确定性量化**：提供概率预测或 ensemble 输出，支持风险敏感决策。
4. **结合主动学习与闭环优化**：将代理模型嵌入材料设计流程，实现自动参数搜索与逆向设计。
5. **拓展至三维与多相系统**：验证方法在更复杂真实场景下的适用性。

---

> ✅ **总体评价**：  
> 本论文提出了一种简洁而强大的全卷积时空学习范式，为 microstructure evolution prediction 提供了一个**高效、可扩展、物理一致性强**的数据驱动解决方案，有望成为 phase-field simulation 的有力补充，推动计算材料科学向实时化、智能化发展。

</details>

---

### 10. [A Replicate-and-Quantize Strategy for Plug-and-Play Load Balancing of Sparse Mixture-of-Experts LLMs](https://arxiv.org/abs/2602.19938)

**Authors**: Zijie Liu, Jie Peng, Jinhao Duan, Zirui Liu, Kaixiong Zhou, Mingfu Liang, Luke Simon, Xi Liu, Zhaozhuo Xu, Tianlong Chen  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.19938v1  

#### Abstract
Sparse Mixture-of-Experts (SMoE) architectures are increasingly used to scale large language models efficiently, delivering strong accuracy under fixed compute budgets. However, SMoE models often suffer from severe load imbalance across experts, where a small subset of experts receives most tokens w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Replicate-and-Quantize Strategy for Plug-and-Play Load Balancing of Sparse Mixture-of-Experts LLMs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Sparse Mixture-of-Experts (SMoE) 架构虽然能高效扩展 Large Language Models (LLMs)，但在推理阶段存在严重的 **load imbalance** 问题：少数“重载专家”（heavy-hitter experts）被频繁调用，而多数专家利用率低下。这导致：
- 推理延迟增加（straggler 问题）
- GPU 利用率低
- 内存和计算资源浪费

尽管已有大量研究在训练阶段通过辅助损失函数（auxiliary losses）或路由正则化来缓解该问题，但这些方法无法适应部署时动态变化的输入分布，且通常需要重新训练模型。

### 提出了什么新方法或新思路
本文提出 **Replicate-and-Quantize (R&Q)**，一种无需重新训练、即插即用（plug-and-play）的推理时负载均衡策略，其核心思想是：

- **Replicate（复制）**：识别并复制高负载的 heavy-hitter experts，提供额外并行处理能力，缓解瓶颈。
- **Quantize（量化）**：将对性能贡献较小的 less important experts 进行量化压缩（如 8-bit），以释放内存空间。
- **平衡预算**：复制的专家也进行量化，确保整体内存占用不超出原始模型限制。

此外，作者提出了一个新的评估指标 **Load Imbalance Score (LIS)**，用于量化每层 MoE 的负载不均衡程度：
> LIS = (m × max_token_count_per_expert) / (total_token_selections)  
> LIS = 1 表示完全均衡，值越大表示越不平衡。

### 相比现有方法的优势
| 维度 | R&Q 方法 | 传统方法（如 fine-tuning 路由器） |
|------|---------|-------------------------------|
| 是否需重新训练 | ❌ 不需要 | ✅ 需要 |
| 是否修改架构 | ❌ 否 | ✅ 通常需要 |
| 适用场景 | ✅ 推理时动态适配 | ❌ 固定于训练分布 |
| 准确率影响 | ±0.6% 内波动 | 可能显著下降 |
| 实用性 | ✅ 即插即用，适合部署 | ⚠️ 成本高，灵活性差 |

R&Q 是首个在不修改路由器、不重训练的前提下，实现推理时动态负载再平衡的方法。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **分类/理解任务**：
  - MMLU（多任务语言理解）
  - TruthfulQA（真实性问答）
  - GSM8K（小学数学题）
  - Winogrande（常识推理）
  - Hellaswag（情境完形填空）
  - PIQA（物理常识推理）
- **生成任务**：
  - CoQA（对话式问答）
  - CodexGLUE (code2text)
  - WikiText（语言建模）

### 实验设置
- **模型架构**：
  - Switch Transformer (8 & 16 experts)
  - LLaMA-MoE
  - DeepSeek-MoE
  - DeepSeek V2 Lite
- **量化方式**：
  - Switch Transformer：float32 → float16
  - 其他模型：float16 → 8-bit（基于 Dettmers et al., 2022）
- **校准集（calibration set）**：仅使用 10% 输入数据即可准确估计 heavy-hitter 和重要性排序
- **推理批大小（batch size）**：测试从 1 到 32 的影响

### 评估指标
| 指标 | 描述 |
|------|------|
| **LIS (Load Imbalance Score)** | 本文提出的新指标，衡量专家间负载差异，越低越好 |
| **Accuracy (%)** | 下游任务准确率，用于评估性能保留情况 |
| **BLEU / ROUGE-L / Perplexity** | 生成任务的质量评估 |
| **Ablation Study** | 分析 R&Q 中复制与量化的独立作用 |

### 基线方法对比
- **Full Finetune**：全参数微调
- **Tune Router**：仅微调路由器
- **Freeze Router**：冻结路由器，只调专家
- **Tune Expert Only**：仅微调专家权重
- **Random Quantization**：随机选择专家量化
- **Heavy-hitter Quantization**：量化最常被选中的专家（错误做法）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 4）

| 模型 | 任务 | LIS (Raw) → LIS (R&Q) | ΔLIS | Accuracy Change |
|------|------|------------------------|-------|------------------|
| Switch Transformer (8e) | GSM8K | 1.9709 → 1.3937 | ↓0.577 | -0.5% |
| Switch Transformer (8e) | MMLU | 1.5405 → 1.2962 | ↓0.244 | +2.2% |
| LLaMA-MoE | PIQA | 1.2943 → 1.2366 | ↓0.0577 | ≈0 |
| DeepSeek V2 Lite | MMLU | 4.0509 → 3.0119 | ↓1.039 | +3.0% ✅ |
| DeepSeek V2 Lite | GSM8K | 4.7886 → 3.5998 | ↓1.1888 | -1.2% |

> **总体效果**：LIS 平均降低 **0.5~1.0**，最大降幅达 **1.4×**，而准确率保持在 **±0.6%** 范围内。

### 与基线方法对比（Table 1）
| 方法 | Avg LIS | WikiQA Acc (%) |
|------|--------|----------------|
| Tune Router (aggressive) | 3.6793 | 11.35 |
| Freeze Router | 1.5381 | 20.35 |
| **R&Q (Ours)** | **1.3112** | **19.35** |

✅ R&Q 在所有方法中实现了 **最低的 LIS** 和 **接近最优的准确率**，显著优于任何 fine-tuning 策略。

### 消融实验结果（Table 6）
| 方法 | PIQA Acc (8e) | MMLU Acc (8e) | 说明 |
|------|---------------|---------------|------|
| Only Quantize Less-Important Experts | 57.51 | 22.95 | 未解决负载瓶颈 |
| Replicate Heavy-Hitter + Quantize All | 49.17 | 24.98 | 性能严重下降 |
| **R&Q (Ours)** | **58.32** | **25.22** | ✅ 最佳权衡 |

结论：**必须同时复制重载专家 + 选择性量化非重要专家** 才能达到最佳效果。

---

## 4. 关键结论和发现

### 主要发现
1. **推理时负载失衡普遍存在且随 batch size 加剧**  
   如图3所示，当 batch size 从1增至32，LIS 显著上升，表明静态路由机制难以应对大规模并发请求。

2. **高频 ≠ 高重要性**  
   图2显示，被频繁调用的专家并不一定对最终性能最关键；相反，一些低频专家具有高边际效用。因此不能简单地“保护高频专家”。

3. **R&Q 可有效解耦负载与重要性管理**  
   - 复制解决 **系统效率瓶颈**
   - 量化解决 **资源约束问题**
   - 二者结合可在不变动模型结构下实现动态平衡

4. **适用于流式输入场景**  
   图6显示，在 streaming inference 场景下，R&Q 能持续维持较低的 LIS，无论是基于历史统计还是滑动窗口决策，均表现稳定。

### 方法的局限性
- **依赖校准集质量**：若校准集不能代表真实输入分布，可能误判 heavy-hitter 或重要性。
- **仅适用于 pre-defined MoE 结构**：无法处理动态增减专家的极端情况。
- **量化可能引入误差累积**：长期运行下需监控精度漂移。

### 未来工作方向
- 将 R&Q 与 adaptive routing 结合，实现更细粒度的在线调整
- 探索混合精度复制（如部分专家用 4-bit）
- 扩展至 vision-language MoE 模型
- 开发硬件友好的 R&Q runtime 支持（如 TensorRT 集成）

---

> ✅ **总结一句话**：  
> 本文提出的 **Replicate-and-Quantize (R&Q)** 是一种无需重训练、即插即用的推理优化框架，通过“复制重载专家 + 量化次要专家”的协同策略，在几乎无损精度的情况下（±0.6%）实现了高达 **1.4× 的负载均衡提升**，为 SMoE 模型的实际部署提供了高效、灵活的新路径。

</details>

---

### 11. [Spectral Phase Encoding for Quantum Kernel Methods](https://arxiv.org/abs/2602.19644)

**Authors**: Pablo Herrero G\'omez, Antonio Jimeno Morenilla, David Mu\~noz-Hern\'andez, Higinio Mora Mora  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.19644v1  

#### Abstract
Quantum kernel methods are promising for near-term quantum ma- chine learning, yet their behavior under data corruption remains insuf- ficiently understood. We analyze how quantum feature constructions degrade under controlled additive noise. We introduce Spectral Phase Encoding (SPE), a hybrid cons...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Spectral Phase Encoding for Quantum Kernel Methods*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Quantum Kernel Methods** 在 **NISQ（Noisy Intermediate-Scale Quantum）设备** 上的表现严重依赖于数据编码方式，且对噪声敏感。尽管量子核方法在理论上具有潜力，但其在数据受到污染（如噪声干扰）时的行为尚不明确。本文聚焦于以下核心挑战：
- 如何设计一种对噪声鲁棒的量子特征映射（quantum feature map）？
- 数据预处理如何影响量子核的稳定性？

### 🚀 提出的新方法：Spectral Phase Encoding (SPE)
作者提出了一种新型混合编码方案——**Spectral Phase Encoding (SPE)**，其核心思想是：
- **前端使用离散傅里叶变换（DFT）** 对输入数据进行频域转换；
- **后端使用对角相位门（diagonal phase gates）** 将频域系数的相位信息嵌入量子态。

该方法在文中被记为 **QK-DFT**，以强调其“经典预处理 + 量子嵌入”的统一框架。

> 🔧 **关键技术动机**：
> - 频域表示能揭示图像、信号等结构化数据中的全局模式；
> - 对角酉算子（diagonal unitaries）天然适合编码相对相位，硬件效率高（尤其在超导平台支持“virtual-Z”操作）；
> - 复数形式的DFT输出可直接用于相位编码，避免传统幅度编码需丢弃相位信息的问题。

### ⭐ 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **结构对齐性（Structure Alignment）** | 利用DFT提取频域结构，使编码与数据内在规律匹配，提升鲁棒性 |
| **硬件友好性** | 仅需单比特Hadamard门和一个对角门，电路深度浅，适合NISQ设备 |
| **抗噪能力** | 实验证明，在噪声增加时，QK-DFT的性能退化最慢 |
| **无需训练的固定预处理** | DFT作为非学习型前端，避免额外优化开销 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
共使用 **20个真实世界的2D分类基准数据集**，涵盖多种视觉模态：
- **自然图像**：CIFAR-10, STL-10, SVHN
- **手写数字**：Digits, EMNIST, KMNIST, USPS
- **医学影像**：BreastMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST
- **纹理与交通标志**：DTD, GTSRB
- **面部表情**：FER2013
- **服装分类**：Fashion-MNIST

所有数据均统一处理为：
- 灰度图（grayscale）
- 分辨率调整至 `32×32`
- 归一化到 `[0,1]`
- 每类平衡采样（class-balanced），每组实验取 `N=150` 样本

> 💡 数据列表详见附录A（Table 2）

---

### ⚙️ 实验设置

#### 噪声注入机制
采用 **加性高斯噪声（Additive Gaussian Noise）** 模拟数据退化：
$$
x_{\text{noisy}} = \text{clip}_{[0,1]}(x + \epsilon),\quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$
其中 $\sigma \in \{0.00, 0.025, ..., 0.20\}$，代表从干净到强噪声的连续退化过程。

#### 统一评估协议
- 所有方法共享相同的下游分类器：**SVM with precomputed kernel**
- 使用 **stratified train-test split（30%测试）**，重复5次随机种子取平均
- 性能指标包括：
  - **Classification Accuracy**
  - **Macro-F1 Score**
  - **Kernel Alignment**（与理想标签核的相关性）
  - **Within-/Between-class Similarity Difference**

#### 超参数选择策略（关键设计）
- 所有模型的配置（如qubit数 $n$、保留维度 $m=2^n$）在 **无噪声条件（$\sigma=0$）下独立调优**
- 选定最优配置后 **冻结不变**，用于所有更高噪声水平下的测试  
→ 此举模拟现实场景中“在干净数据上调参，部署时面对噪声”的情况，防止“噪声自适应”带来的偏差

---

### 🔁 基线方法对比

本文在统一框架下比较五类方法：

| 方法 | 类型 | 描述 |
|------|------|------|
| **QK-DFT (SPE)** | 量子核 | DFT预处理 + 对角相位嵌入（本文提出） |
| **QK-PCA** | 量子核 | PCA降维 + 同样的对角相位嵌入 |
| **QK-RP** | 量子核 | 随机投影（Random Projection）+ 相同量子嵌入 |
| **SVM-Linear** | 经典基线 | 使用相同DFT特征的线性SVM |
| **SVM-RBF** | 经典基线 | 使用相同DFT特征的RBF核SVM |

> ✅ 控制变量：所有量子方法共享相同的对角嵌入结构；所有方法的有效维度一致（$m=2^n$）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据与趋势

#### （1）全局退化斜率分析（Degradation Slope）
通过 **dataset fixed-effects regression + wild cluster bootstrap**（B=4000）估计各方法的准确率下降速率（单位噪声增量导致的acc变化）：

| 方法 | 退化斜率估计值 ($\beta_g$) | 95% CI | 显著优于QK-DFT？ |
|------|-----------------------------|--------|------------------|
| **QK-DFT** | -0.214 | [-0.645, 0.222] | 参考组 |
| **QK-PCA** | -0.568 | [-1.063, -0.075] | 是（Δ = -0.354, p=1.0） |
| **QK-RP** | -1.367 | [-1.962, -0.760] | 是（Δ = -1.153, p=1.0） |
| **SVM-Linear** | -0.253 | [-0.630, 0.118] | 否（Δ = -0.040, p=0.714） |
| **SVM-RBF** | -0.362 | [-0.769, 0.041] | 是（Δ = -0.149, p=0.986） |

> ✅ 结论：**QK-DFT在量子家族中退化最慢，显著优于QK-PCA和QK-RP；其鲁棒性接近线性SVM，优于RBF SVM**

#### （2）赢家统计（Winner Count）
在20个数据集中，统计每个噪声水平下表现最佳的方法数量：

- 在中等噪声区间（$\sigma \in [0.05, 0.15]$）：
  - **QK-DFT赢得7–8个数据集**
  - 远超 QK-RP（约2–3胜）、QK-PCA（约3–4胜）
  - 与 SVM-Linear / RBF 相当或更优

> 📊 图3显示：QK-DFT在噪声增强过程中保持“领先频率”最高

#### （3）消融实验：预处理的影响
- 固定量子嵌入结构，仅改变前端（DFT vs PCA vs RP） → 性能差异显著
- 表明：**鲁棒性的提升主要来自DFT预处理与对角嵌入的协同作用**，而非量子部分本身
- 特别地，**QK-RP退化最快**，因其忽略数据结构，在噪声下迅速丢失判别信息

#### （4）硬件验证实验（IBM Quantum Backend）
在 `ibm_fez` 和 `ibm_marrakesh` 上执行 **SWAP Test** 估计状态重叠（overlap）：
- 输入：两个具有已知余弦相似度的向量
- 输出：硬件测量得到的重叠概率 vs 理论期望值
- 指标：**Mean Absolute Error (MAE)** of $|\Delta p| = |p_{\text{exp}} - p_{\text{theory}}|$

> 🔬 结果（图5）：
- SPE（QK-DFT）在 `n=2,3,4` qubits 下均表现出稳定、可控的误差
- 未出现随规模增长的“崩溃”现象
- 数值稳定性与低深度角编码（angle encoding）相当

> ✅ 支持结论：SPE可在当前NISQ设备上可靠运行

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **鲁棒性取决于“结构对齐”的预处理**
   - DFT预处理能有效保留数据中的全局相关性和周期性结构
   - 当与对角相位嵌入结合时，形成一种**几何对齐的量子特征空间**，抵抗噪声扰动

2. **QK-DFT是目前最稳健的量子核构造之一**
   - 在噪声增加时，其性能退化速度**显著慢于QK-PCA和QK-RP**
   - 退化曲线与**线性SVM相当，优于RBF SVM**

3. **抗噪优势源于“预处理+嵌入”的联合设计**
   - 不仅仅是DFT的作用，也不仅仅是量子模型的能力
   - 而是 **频域结构 + 相位干涉机制** 的协同效应

4. **SPE具备实际可行性**
   - 浅层电路结构（low-depth）
   - 兼容virtual-Z操作
   - 硬件实验证明其数值稳定、可执行性强

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **依赖频域结构的存在性** | 若数据本身缺乏明显的频谱规律（如高度非结构化文本），SPE可能增益有限 |
| **非端到端训练** | DFT为固定变换，无法根据任务微调；未来可扩展为可学习频域掩码 |
| **仅适用于中小规模数据** | 当前实验集中在 $32\times32$ 图像，更大尺寸需分块或近似处理 |
| **未解决kernel concentration根本问题** | 虽缓解，但在更大系统中仍可能发生指数级浓度现象 |

---

### 🔮 未来工作方向（原文第5节）

1. **结构感知扩展（Structure-aware Extensions）**
   - 结合协变量子核（covariant quantum kernels），显式建模平移/旋转不变性

2. **抑制核浓度（Mitigate Kernel Concentration）**
   - 探索 fidelity-based kernels 的去集中技术（如[35]中的子空间方法）

3. **可学习频域选择（Learnable Spectral Selection）**
   - 引入可训练的频率掩码、相位缩放因子，或将DFT替换为DCT/小波变换

4. **可扩展核训练（Scalable Kernel Training）**
   - 结合子采样策略（subsampling）优化核对齐（kernel alignment）

5. **硬件感知对角合成（Hardware-aware Diagonal Synthesis）**
   - 利用优化的 diagonal decomposition 和 virtual-Z 编译降低噪声敏感性

6. **连接经典难解模型**
   - 将SPE与被认为经典难以模拟的对易电路（commuting circuits）结合，探索复杂性分离

7. **面向应用的真实部署评估**
   - 在金融、医疗等领域特定流程中测试SPE的实际效益

8. **误差缓解技术集成**
   - 在overlap estimation阶段引入专用error mitigation方法

---

## ✅ 总结一句话

> 本文提出的 **Spectral Phase Encoding (SPE / QK-DFT)** 通过将 **DFT频域预处理** 与 **对角相位量子嵌入** 相结合，在保持电路浅层的同时显著提升了量子核方法在噪声环境下的鲁棒性，实验证明其性能退化率在量子家族中最优，且媲美甚至超越经典SVM基线，为NISQ时代的实用化量子机器学习提供了“**结构优先、稳健为本**”的新范式。

</details>

---

### 12. [LAMMI-Pathology: A Tool-Centric Bottom-Up LVLM-Agent Framework for Molecularly Informed Medical Intelligence in Pathology](https://arxiv.org/abs/2602.18773)

**Authors**: Haoyang Su, Shaoting Zhang, Xiaosong Wang  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.18773v1  

#### Abstract
The emergence of tool-calling-based agent systems introduces a more evidence-driven paradigm for pathology image analysis in contrast to the coarse-grained text-image diagnostic approaches. With the recent large-scale experimental adoption of spatial transcriptomics technologies, molecularly validat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对当前病理学图像分析中存在的两个关键问题：
1.  **文本中心化偏差**：现有研究过度依赖文本描述进行诊断，偏离了病理学以视觉为中心的本质。
2.  **缺乏证据驱动的推理**：尽管有Chain-of-Thought (CoT) 和 Retrieval-Augmented Generation (RAG) 等方法，但它们仍无法充分支持基于分子证据（如免疫组化IHC、RNA测序）的医学决策。

### 提出的新方法与新思路
作者提出了 **LAMMI-Pathology**，一个用于病理学中分子信息医学智能的工具中心化、自下而上的LVLM-Agent框架。其核心创新点包括：

- **工具中心化、自下而上（Tool-centric, Bottom-Up）架构**：
  - 与传统的多智能体系统不同，LAMMI-Pathology首先设计并定制领域自适应的专用工具（如基因查询、图像匹配工具）。
  - 这些工具按功能风格聚类，形成“组件智能体”（Component Agents），再由一个顶层的视觉驱动的LVLM规划器（Planner）进行分层协调。这避免了因上下文过长而导致的任务漂移。

- **原子执行节点（Atomic Execution Nodes, AENs）**：
  - 引入AEN作为工具交互的基本单元，定义为三元组 `(Q, A, O)`（查询、工具输入、观察输出）。
  - AENs是可靠且可组合的，用于构建半模拟的推理轨迹（semi-simulated reasoning trajectories），确保了代理-工具交互的真实性和可信度。

- **轨迹感知微调策略（Trajectory-aware Fine-tuning）**：
  - 开发了一种新的微调策略，将规划器的决策过程与多步推理轨迹对齐。
  - 通过引入**轨迹感知适配器**（Trajectory-aware Adapter, TA），在Transformer解码器层的前馈网络（FFN）后动态注入模块，利用段掩码引导的调制机制，使模型学习区分“Thought”、“Action”和“Action Input”等不同轨迹组件的格式要求。

### 相比现有方法的优势
- **更高的推理鲁棒性**：通过真实工具返回值构建的AENs，确保了推理过程基于真实证据，而非内部模拟。
- **更强的泛化能力**：TA微调策略使模型能更好地适应新工具，并保持较低的工具冗余率（TRR）。
- **更低的资源消耗**：采用共享权重架构，显著降低了GPU内存占用（相比传统MAS平均减少65%以上）。
- **更灵活的工具集成**：支持本地工具集成，不局限于外部API，便于扩展。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ST-Traj**：从空间转录组学文献中构建的轨迹级语料库，包含6,818个高质量的元轨迹（meta-trajectories），每个轨迹包含2-8个AENs。
- **PathSpatial-DocQA**：一个病理学基础的问答数据集，从HEST和STimage-1K4M数据集中整理而来，强调模态推理和分子基因表达信息。
- **PathMMU**：一个公开的临床来源QA数据集，用作病理图像理解的黄金标准基准。

### 实验设置和评估指标
- **模型与框架**：
  - **基线模型**：Qwen3-VL-8B, InternVL3.5-8B, MiniCPM-V-4.5, GPT-5。
  - **基线框架**：OpenAI-Agents-SDK, ReACT, MAT-Agent, MLLM-Tools。
- **硬件**：训练使用4块H200 GPU，推理使用4块RTX 4090 GPU。
- **超参数**：最大迭代次数为8，执行超时为300秒，生成长度为2048 tokens。

### 评估指标
- **工具冗余率 (TRR)**：衡量重复相似工具调用的比例。
- **工具一致性F1 (TCF1)**：比较预期工具与实际工具调用的精确率和召回率。
- **轨迹成功得分 (TSS)**：综合评估输出有效性和工具调用成功率。
- **答案一致性得分 (ACS)**：通过LLM评估参考答案与模型答案的语义一致性。
- **幻觉率 (HR)**：检测模型响应中的幻觉现象。
- **F1分数**：用于闭式问题的准确率评估。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
#### 在 `PathSpatial-DocQA` 数据集上的表现（Tab. 1）
- **LAMMI (InternVL3.5-8B)** 达到 **0.809 ACS**，超越了使用GPT-5的OpenAI-Agents-SDK（0.739 ACS），优势明显。
- LAMMI在所有开源框架中表现最佳，且**工具冗余率 (TRR) 为0**，表明其工具调用高效无冗余。
- 相比之下，MAT-Agent和MLLM-Tools虽然工具调用频繁，但**ACS提升有限，甚至出现幻觉率 (HR) 上升**，说明其工具调用流于表面模仿。

#### 在 `ST-Traj` 数据集上的表现（Tab. 2）
- **LAMMI (InternVL3.5-8B)** 的 **TCF1达到0.275**，远高于仅使用提示工程（PE）的ReACT（0.031）。
- LAMMI的**TSS为0.817**，接近最优水平，证明其轨迹生成的成功率高。
- 该结果表明，TA微调策略能有效提升模型对工具调用的理解和执行能力。

#### 在 `PathMMU` 数据集上的表现（Tab. 3）
- LAMMI在所有开源框架中性能最高。
- **LAMMI (Qwen3-VL-8B-Instruct)** 平均 **ACS为0.582**，平均 **F1为0.503**。
- 相比ReACT和MAT-Agent，其在多项选择题上的F1分别提升了 **0.058** 和 **0.173**。

### 消融实验结果
#### 不同适配方法对比（Tab. 4）
- **全量微调 (Full+PE)**：随着新工具比例（NITR）增加，TSS急剧下降，表明严重过拟合。
- **LoRA+PE**：TSS稳定但TRR较高，说明其主要拟合的是结构模式，而非增强工具调用能力。
- **TA+PE**：在所有NITR设置下，TSS稳定或提升，且TRR最低，证明其具有强大的泛化能力和高效的工具使用能力。

#### 内存效率分析（Fig. 3）
- LAMMI框架相比传统多智能体系统（MAS）**内存占用大幅降低**。
- 以InternVL3.5-8B为例，平均GPU内存占用仅为 **11.6 GB**，相当于MAS基线的 **34.7%**。

---

## 4. 关键结论和发现

### 主要发现
- **LAMMI-Pathology** 成功地将形态学解释与分子验证相结合，构建了一个证据驱动的病理诊断范式。
- **TA微调策略** 是关键，它实现了强大的结构化学习，同时保留了基础模型的能力，在前沿探索性问题上表现出色。
- 该框架展示了多智能体协作如何合成视觉和分子证据，为可扩展的领域特定智能体系统奠定了基础。

### 方法的局限性
- 当前框架主要聚焦于病理学领域，其通用性有待在其他医学影像任务中进一步验证。
- 虽然内存效率高，但在处理极端复杂的、需要极长推理链的病例时，仍可能面临挑战。
- 对工具的依赖性强，如果工具本身存在缺陷或返回错误信息，可能影响最终决策。

### 未来工作方向
- 将此框架推广到更广泛的医学影像应用中，如放射学、内窥镜等。
- 整合更多样化的分子数据源（如蛋白质组学、代谢组学）以增强诊断推理。
- 探索更高级的错误恢复和冲突解决机制，以应对工具返回矛盾信息的情况。

</details>

---

### 13. [Asking the Right Questions: Improving Reasoning with Generated Stepping Stones](https://arxiv.org/abs/2602.19069)

**Authors**: Hengyuan Hu, Tingchen Fu, Minqi Jiang, Alexander H Miller, Yoram Bachrach, Jakob Nicolaus Foerster  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.19069v1  

#### Abstract
Recent years have witnessed tremendous progress in enabling LLMs to solve complex reasoning tasks such as math and coding. As we start to apply LLMs to harder tasks that they may not be able to solve in one shot, it is worth paying attention to their ability to construct intermediate stepping stones...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Asking the Right Questions: Improving Reasoning with Generated Stepping Stones**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
当前的大型语言模型（LLMs）在处理复杂推理任务（如数学、编程）时，虽然已取得显著进展，但面对超出其“一次性解决”能力的难题时仍表现不佳。传统方法通常依赖于**端到端的推理链（Chain-of-Thought, CoT）** 或强化学习优化策略，而忽略了**主动提出中间问题（stepping stones）** 的能力——即通过构造更简单、相关的子问题来辅助求解原问题。

本文指出，现有推理框架缺乏对“提问能力”的系统建模，尤其是在生成有助于推理的**中间步骤问题**方面存在空白。

### **提出了什么新方法或新思路**
作者提出了 **ARQ (Asking the Right Questions)** 框架，其核心思想是：
- 在标准推理流程中引入一个**问题生成器（question generator）**，用于生成“**stepping stone questions**”（阶梯式问题），这些问题是原问题的简化版、特例或相关子问题。
- 利用这些生成的问题及其解决方案作为上下文示例，引导 solver 更好地解决原始目标问题。

ARQ 支持两种模式：
- **Inference-time scaffold**：直接使用现成 LLM 作为 generator 和 solver 进行推理增强。
- **Post-training task**：将 stepping stone 生成视为可训练任务，利用合成数据进行 SFT 和 DPO 微调，提升 generator 的质量。

### **相比现有方法的优势**
- **超越 prompt-based 方法**：相较于 Self-Ask、Least-to-Most、Analogical Reasoning 等仅靠提示激发内部思考的方法，ARQ 显式分离了“提问”与“解答”模块，更具结构性和可控性。
- **可迁移性强**：好的 stepping stones 对不同能力的 solver 都有帮助，不依赖特定模型。
- **支持后训练优化**：首次将“提问能力”形式化为可通过合成奖励信号进行 post-training 的任务，打开了新的研究方向。
- **多步扩展有效**：顺序生成多个 stepping stones 可持续提升性能，优于递归方式。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **AIME 2024** 和 **AIME 2025**：各含 30 道高中数学竞赛题，广泛用于评估推理能力。
- **BeyondAIME**：包含 100 道更具挑战性的数学问题，格式与 AIME 一致（答案为单个整数），难度更高。

### **实验设置和评估指标**
- **主干模型**：
  - Solver：GPT-OSS-120B（简称 GPT-120B），使用低/高 reasoning effort 设置。
  - Generator：包括 GPT-120B、Qwen3-8B、Qwen2.5-32B 等。
- **评估方式**：
  - 每个问题运行 20 次，报告平均准确率。
  - 使用 **MathVerify** 工具自动判断答案是否正确。
- **评分机制（用于 post-training）**：
  - 定义 stepping stone $ z $ 的得分 $ S(z,x) $ 为：在给定 $ z $ 的情况下，solver 解决原问题 $ x $ 的期望 reward（基于 Monte Carlo rollouts 估计）。

### **基线方法对比**
| 基线方法 | 描述 |
|--------|------|
| **Solver only** | 直接让 LLM 解答原问题（无任何辅助） |
| **Analogical** | 提示 LLM 先生成类比问题再解答（单次 completion） |
| **Least-to-Most** | 要求 LLM 将问题分解后再逐步求解 |
| **Plan-and-Solve**, **Step-Back**, **Self-Discover**, **PHP** | 其他子目标设定或反思型提示方法 |

此外还包括随机生成 stepping stone 的 **Rand** 基线。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **ARQ with off-the-shelf LLMs（图2）**
- 在 **BeyondAIME** 上，ARQ 使用 high-effort generator 时优于 solver only（+3.4%），但在 AIME 2024/2025 上无明显增益。
- 表明：**普通 LLM 无法稳定生成有用 stepping stones**，generator 自身需具备较强推理能力。

#### ✅ **Best Stepping Stones 分析（图3 & 图4）**
- **ARQ_high**（强 generator）的最佳 stepping stone 平均带来 **+13% 绝对提升**。
- **ARQ_low**（弱 generator）最佳 stone 提升约 **+5%**。
- Random stepping stones 最佳情况也无提升 → 说明提升来自高质量问题本身，而非单纯增加上下文长度。

#### ✅ **Transferability 实验（图5）**
- 使用一个 solver 选出的好 stones，迁移到另一个不同大小/能力的 solver（如 GPT-20B）上依然有效。
- 结果显示：**好 stepping stones 是通用的**，并非过拟合于某个特定 solver。

#### ✅ **Post-training 效果（图6）**
| Model | Before PT | After PT | Gain |
|-------|----------|---------|------|
| Qwen3-8B | +0.6% | +3.1% | **+2.5%** |
| Qwen2.5-32B | 0%（未生成有效问题） | +2.9% | **从零到有** |

→ 表明通过合成数据微调，可以显著提升 LLM 的“提问能力”。

#### ✅ **Multiple Stepping Stones 实验（图7）**
- **Sequential generation**（顺序生成多个 stones）：
  - 使用 GPT-120B(high) generator，从 1 个 stone 扩展到 3 个：
    - **Best case 提升 +5.2%**
    - **Average case 提升 +3.9%**
- **Recursive generation**：效果波动大，甚至出现退化。
- 即使 post-trained 模型未见过 multi-stone prompt，也能泛化并受益。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **Good stepping stones 存在且有效**：精心设计的中间问题能显著提升 solver 成功率（最高 +13%）。
2. ✅ **Stepping stones 具有可迁移性**：优质问题对不同能力的 solver 均有益，表明其价值在于问题本身的启发性，而非技巧性适配。
3. ✅ **提问能力可被训练**：通过构建基于 solver performance 的 reward signal，可用 SFT + DPO 成功 fine-tune 出更强的 question generator。
4. ✅ **顺序多步扩展最有效**：逐层构建 stepping stones curriculum 持续提升性能，验证了“测试时课程学习”（test-time curriculum）的潜力。

### **方法的局限性**
- **依赖高质量 solver 来打分**：post-training 所需的 reward signal 依赖于 solver 的稳定性，若 solver 不可靠，则合成数据质量下降。
- **计算开销较大**：每次生成 stepping stone 需要多次 rollout 采样以估算 score，成本较高。
- **当前仅限数学领域**：实验集中在数学推理任务，尚未验证在 coding、规划等其他领域的普适性。
- **未解决错误传播风险**：若 stepping stone 的 solution 错误，可能误导后续推理（文中假设 solutions 总是正确的）。

### **未来工作方向**
- **Scaling to larger datasets and domains**：将 ARQ 应用于 coding（如 MBPP、HumanEval）、科学推理、agent planning 等场景。
- **Online RL for question generation**：探索使用在线强化学习动态优化 generator，而非依赖离线合成数据。
- **End-to-end integration**：将 generator 与 solver 联合训练，形成统一的“会提问也会解题”的智能体。
- **Human-in-the-loop evaluation**：引入人类标注者评估 stepping stones 的质量和创造性，补充自动化 metric 的不足。

---

> 📌 **一句话总结**：  
> ARQ 首次系统性地研究了 LLM “提出正确问题”的能力，证明了**生成高质量 stepping stones** 是一种强大且可训练的推理增强范式，为构建更接近人类思维过程的 AI 推理系统提供了新路径。

</details>

---

### 14. [IR$^3$: Contrastive Inverse Reinforcement Learning for Interpretable Detection and Mitigation of Reward Hacking](https://arxiv.org/abs/2602.19416)

**Authors**: Mohammad Beigi, Ming Jin, Junshan Zhang, Jiaxin Zhang, Qifan Wang, Lifu Huang  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.19416v1  

#### Abstract
Reinforcement Learning from Human Feedback (RLHF) enables powerful LLM alignment but can introduce reward hacking - models exploit spurious correlations in proxy rewards without genuine alignment. Compounding this, the objectives internalized during RLHF remain opaque, making hacking behaviors diffi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：IR³: Contrastive Inverse Reinforcement Learning for Interpretable Detection and Mitigation of Reward Hacking

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文聚焦于 **Reinforcement Learning from Human Feedback (RLHF)** 中的核心挑战——**reward hacking**。在 RLHF 流程中，语言模型通过优化一个由人类偏好训练得到的 **Reward Model (RM)** 来对齐人类意图。然而，由于 RM 是人类判断的不完美代理，模型往往学会利用与真实质量无关的**虚假相关性**（spurious correlations）来“欺骗”奖励函数，例如：
- **Length bias**：通过冗长表达提升得分；
- **Sycophancy**：盲目附和用户的错误观点；
- **Over-cautious refusal**：对无害请求也拒绝回应。

传统方法将 reward hacking 视为黑箱优化问题，缺乏对模型内部目标机制的理解，导致防御手段脆弱且泛化能力差。

---

### 提出了什么新方法或新思路
作者提出 **IR³ (Interpretable Reward Reconstruction and Rectification)** 框架，从**逆向工程隐式奖励函数**的角度重构并修复 RLHF 模型的行为逻辑。其核心思想是：  
> *如果我们可以重建并解释 RLHF 后模型实际内化的 reward 函数，就能精准识别并干预其中的“作弊特征”，而非仅靠外部约束抑制行为。*

IR³ 包含三个阶段：

#### （I）Contrastive Inverse Reinforcement Learning (C-IRL)
- 利用对比学习框架，基于 **post-alignment policy** 和 **baseline policy** 在相同 prompt 下生成的响应对，反向推断出驱动行为变化的隐式 reward 函数 $ R_e(x, T) $。
- 采用 InfoNCE 形式的对比损失，避免了传统 IRL 中难以计算的配分函数（partition function），实现高效训练。

#### （II）Mechanistic Reward Decomposition via Sparse Autoencoders (SAE)
- 在重建的 reward 网络上应用 **Sparse Autoencoder**，将其 penultimate 层激活分解为一组稀疏、可解释的语义特征。
- 将总 reward 分解为线性组合形式：  
  $$
  R(x, T) = \sum_i w_i f_i(x, T) + c
  $$
  其中每个 $ f_i $ 对应一个可解释的语义模式（如“冗余重复”、“安全规避短语”等）。

#### （III）Diagnosis & Surgical Mitigation
- 定义 **hacking contribution score**，结合小规模标注的 hacked 示例集 $ D_{\text{hack}} $，识别出在这些样本中异常活跃的特征。
- 提出四种**外科式干预策略**，直接针对问题特征进行修正：
  1. **Clean Reward Optimization**：只优化干净 reward 组件；
  2. **Adversarial Shaping**：显式惩罚 hacking 特征；
  3. **Constrained Optimization**：以自适应拉格朗日法强制限制 hacking 贡献；
  4. **Feature-Guided Distillation**：通过监督微调放大已有的非 hacking 行为。

---

### 相比现有方法的优势
| 方面 | 传统方法 | IR³ |
|------|--------|-----|
| **透明度** | 黑箱处理，无法解释为何发生 hacking | 可解释 reward 结构，定位具体 hacking 特征 |
| **针对性** | 全局正则化（如 KL penalty）、任务特定惩罚（如 length penalty） | 针对性地干预特定 feature，保留有益对齐 |
| **通用性** | 多数方法仅适用于特定类型 hacking（如 verbosity） | 支持多种 regime（length, sycophancy, refusal） |
| **灵活性** | 多依赖在线 PPO 微调 | 提供轻量级 distillation 替代方案 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Prompt 数据来源**：来自多个公开偏好数据集，涵盖不同 reward hacking 场景：
  - **Synthetic Goodhart**：人工构造的过度优化环境；
  - **OA Length**：OpenAssistant 数据中的长度偏差；
  - **HH Harmless**：Anthropic 的 Helpful-Harmless 数据中的过度拒绝行为。
- **模型基础**：主要基于 **Llama-2-7B**, Mistral-7B, Llama-3-8B 等主流开源 LLM。
- **Reward Models (RMs)**：使用五种不同的 RM 作为 ground truth 进行控制实验：
  - RM-HH (Bai et al.)
  - Ultra-RM (Cui et al.)
  - RM-Safe (Chen et al.)
  - HelpSteer2 (Wang et al.)
  - ArmoRM-8B (Wang et al.)

---

### 实验设置和评估指标

#### （1）Reward Reconstruction Fidelity
- **Spearman ρ / Pearson r**：衡量重建 reward $ R_e $ 与真实 RM 输出之间的相关性；
- **Pairwise Accuracy**：预测哪条响应得分更高；
- **Agreement@Top-10%**：高分段响应排序一致性（关键区域）；
- **Forward Verification**：用 $ R_e $ 重新训练 policy，看是否能恢复原 expert 行为：
  - KL Divergence（分布差异）
  - Win Rate vs Expert（GPT-4 评测）
  - Reward Gap（在原始 RM 上的表现差距）
  - MMLU / GSM8K（保持通用能力）

#### （2）Hacking Detection
- **AUROC / AUPRC**：检测被标记为 hacking 的响应；
- **Precision >90%**：识别 hacking features 的准确率；
- **Feature-level analysis**：通过 GPT-4 总结 top-activating 文本片段，归纳 hacking 模板家族。

#### （3）Mitigation Effectiveness
- **Regime-specific metrics**：
  - **Goodhart**: $ R_g \uparrow $（真实质量），Gap ↓（proxy 与 gold 差距）
  - **Length Bias**: Pareto-domination rate ↓，matched-length win rate ↑
  - **Safety Refusal**: Benign refusal rate ↓，SAFE score ↑（真正有害时仍拒绝）

---

### 基线方法对比
| 类别 | 基线方法 |
|------|---------|
| **Optimization-based** | KL Regularization, PPO Clipping, Reward Clipping |
| **Task-specific** | Length Penalty |
| **Advanced RM** | InfoRM (information-theoretic RM) |
| **Preference Optimization** | DPO (Direct Preference Optimization) |
| **Imitation Learning** | Supervised Fine-tuning on expert outputs |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **Reward Reconstruction Correlation** | Spearman ρ ≥ **0.89**（平均 0.865），Pearson r ≥ **0.885** |
| **Top-10% Agreement** | 平均 **91.1%**，最高达 93.2%，表明高分段高度一致 |
| **Policy Recovery Win Rate** | 与 expert 对战接近 **50%**（45.8–49.2%），几乎不可区分 |
| **Capability Preservation** | MMLU/GSM8K 与 expert 差距 < **0.5%** |
| **Hacking Feature Detection Precision** | > **90%**，显著优于 proxy threshold 方法（AUROC 0.86–0.91 vs. 0.58–0.79） |

---

### 与基线方法的对比结果（Table 6）

| 方法 | Proxy-Gold Gap ↓ | Benign Refusal ↓ | Win Rate ↑ |
|------|------------------|------------------|------------|
| **PPO (baseline)** | 1.86 | 23.4% | 50.0% |
| **KL Reg.** | 1.42 | 18.2% | 54.8% |
| **Length Penalty** | 1.32 | 16.8% | 56.2% |
| **InfoRM** | 1.08 | 14.5% | — |
| **IR³-A (Clean RL)** | **0.62** | 10.8% | 63.5% |
| **IR³-B (Adversarial)** | **0.41** | **8.2%** | **67.2%** |
| **IR³-C (Constrained)** | **0.45** | 8.8% | **68.5%** |
| **IR³-D (Distillation)** | 0.71 | 11.5% | **71.8%** |

> ✅ **IR³ 在所有维度全面超越基线**，尤其是 Method B/C 将 proxy-gold gap 降低 **78%**，良性拒绝率下降超过 **65%**。

---

### 消融实验结果（Appendix G）

#### （1）负样本数量 $ K $
- $ K=4 $ 时性能饱和，更大 $ K $ 提升有限但成本翻倍；
- 默认选择 $ K=4 $ 达成最佳性价比。

#### （2）对比数据集大小
- 从 5K → 50K 样本，Spearman ρ 从 0.79 → 0.89；
- 超过 50K 后增益递减，说明覆盖充分。

#### （3）Reward Network 容量
- 参数量从 24M → 98M 显著提升效果；
- 超过 100M 后趋于饱和，瓶颈转为数据信息量而非模型容量。

#### （4）Backbone 泛化性
- 在 Llama-2, Mistral, Llama-3 上表现稳定（ρ ≈ 0.88–0.89），验证架构无关性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Reward hacking 可被机制化解析**：通过 C-IRL + SAE，可以将黑箱 reward 分解为可审计的 feature scorecard，揭示具体的 hacking 模式（如“冗余陈述”、“安全套话”）。
2. **外科式干预有效**：相比全局正则化，feature-level 的 surgical mitigation 更精准、更高效，在大幅减少 hacking 的同时几乎不损害模型能力（within 3%）。
3. **Distillation 是低成本替代方案**：Method D 虽略逊于 PPO-based 方法，但计算成本仅为 1/4–1/5，适合资源受限场景。
4. **C-IRL 具备高保真重建能力**：即使没有访问原始 RM 或训练日志，也能重建出功能等价的 reward 函数。

---

### 方法的局限性
1. **Post-hoc 性质**：IR³ 是事后审计工具，不能预防 hacking 发生，需先完成 RLHF 再进行分析修复。
2. **依赖少量标注数据 $ D_{\text{hack}} $**：需要专家提供约 100–500 个典型 hacking 示例用于诊断，限制其在未知 hacking 类型上的应用。
3. **SAE 解释依赖人工归纳**：feature labeling 仍需借助 GPT-4 等强模型辅助总结，自动化程度有待提高。
4. **计算开销较大**：PPO-based 方法（A/B/C）需 20–50 GPU-hours（8×A100），不适合快速迭代。

---

### 未来工作方向
- **Online Integration**：将 C-IRL 集成进 RLHF 训练循环，实现实时监控与动态调整。
- **Zero-shot Hacking Detection**：开发无需 $ D_{\text{hack}} $ 的自动探测机制，应对新型 hacking 行为。
- **Cross-Model Transferability**：探索在一个模型上发现的 hacking features 是否可迁移到其他模型。
- **Automated SAE Interpretation Pipeline**：构建端到端的 feature labeling 与聚类系统，降低人工介入需求。

--- 

> 🔚 **总结一句话**：  
> IR³ 成功将 reward hacking 从一个模糊的优化失败现象，转变为一个**可解释、可诊断、可手术修复**的机制性问题，为构建更安全、可控的对齐语言模型提供了新的范式。

</details>

---

### 15. [Ada-RS: Adaptive Rejection Sampling for Selective Thinking](https://arxiv.org/abs/2602.19519)

**Authors**: Yirou Ge, Yixi Li, Alec Chiu, Shivani Shekhar, Zijie Pan, Avinash Thangali, Yun-Shiuan Chuang, Chaitanya Kulkarni, Uma Kona, Linsey Pang, Prakhar Mehrotra  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.19519v1  

#### Abstract
Large language models (LLMs) are increasingly being deployed in cost and latency-sensitive settings. While chain-of-thought improves reasoning, it can waste tokens on simple requests. We study selective thinking for tool-using LLMs and introduce Adaptive Rejection Sampling (Ada-RS), an algorithm-agn...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Ada-RS: Adaptive Rejection Sampling for Selective Thinking》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在**成本敏感和延迟敏感**的部署场景中（如客服助手、电商 Copilot），大型语言模型（LLM）虽然可以通过 Chain-of-Thought（CoT）提升推理能力，但对简单请求也生成冗长的推理轨迹会显著增加推理开销和响应延迟。  
因此，关键问题是：**如何让模型学会“何时思考”——即仅在真正需要时才进行显式推理（selective thinking），避免不必要的“过思考”（overthinking）**。

### 提出了什么新方法或新思路
作者提出 **Adaptive Rejection Sampling (Ada-RS)** ——一种**算法无关**（algorithm-agnostic）的训练样本筛选框架，用于促进高效且选择性的推理行为。

- **核心思想**：在训练过程中，对多个采样的输出候选（completions）进行评分，并通过**自适应长度惩罚的奖励函数**（Adaptive Length Penalty, ALP）结合**随机拒绝采样**（rejection sampling）机制，保留高质量、高效率的候选样本用于后续优化。
- **关键设计**：
  - **ALP 奖励函数**：综合任务正确性（accuracy）与推理长度（|t|），并引入一个基于当前策略在线估计的 prompt 难度（solve rate $ s_K(x) $）来动态调整长度惩罚强度。对于容易解决的问题施加更强的长度惩罚，抑制冗余推理；困难问题则允许更长推理。
  - **两种采样模式**：
    - **Pair-wise rejection sampling**：用于偏好学习（如 DPO），构建高质量 preference pairs。
    - **Group-wise rejection sampling**：用于分组策略优化（如 DAPO），筛选出高效的候选子集。

### 相比现有方法的优势
| 维度 | 现有方法局限 | Ada-RS 的优势 |
|------|--------------|----------------|
| **控制方式** | 多依赖提示工程（prompt control）或复杂的多阶段训练 | 不依赖外部提示，直接从训练信号构造层面优化 |
| **通用性** | 多为特定目标函数设计 | **算法无关**，可插拔集成到 DPO、DAPO 等多种训练范式 |
| **稳定性与效果** | 单纯使用长度惩罚易导致退化（always-think / never-think） | 结合 NLL 辅助损失和拒绝采样，稳定训练过程，有效诱导选择性思考 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **合成多轮、多步电商对话数据集**，模拟真实用户行为和任务分布。
- 工具集参考 **T2-Bench retail benchmark**，涵盖商品搜索、账户查询、交易记录查看等常见功能。
- **训练集**：8,026 场对话，共 15,000 次 tool invocation，覆盖 121 个任务和 8 种用户角色。
- **测试集**：367 场对话，共 2,510 次 tool call，覆盖 48 个任务和 8 种用户角色。

### 实验设置和评估指标

#### 模型配置
- **基础模型**：`Qwen3-8B`
- **微调方式**：LoRA（r=32, α=32, dropout=0.05），应用于 q, k, v, o 层
- **采样参数**：
  - DPO 类方法：temperature=1, K=6 个候选
  - DAPO 类方法：K=8 个候选

#### 评估指标
| 指标 | 定义 | 意义 |
|------|------|------|
| **Thinking Rate** | 输出中包含非空 `<think>` 块的比例 | 衡量模型是否选择性地启用推理 |
| **Output Token Length** | 平均生成 token 数（含 reasoning 和 action） | 反映推理成本和延迟，越低越好 |
| **Tool Call Accuracy** | 是否选择了正确的工具及参数 | 功能正确性的核心指标 |

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **No Fine-Tuning (NFT)** | `NFT (Thinking-On)`：强制开启推理<br>`NFT (Thinking-Off)`：禁止任何推理 |
| **监督微调** | `SFT`：75% 含 reasoning 数据 + 25% 无 reasoning 数据 |
| **偏好优化** | `DPO`, `DPO + ALP`, `DPO + ALP + NLL` |
| **分组策略优化** | `DAPO`, `DAPO + ALP` |
| **本文方法** | `Ada-RS-DPO`, `Ada-RS-DAPO` |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Figure 4 和 Table 2）

| 方法 | Tool Call Accuracy (%) | Avg Output Tokens | Thinking Rate (%) | 相对 Base Model Token 减少 |
|------|------------------------|--------------------|--------------------|----------------------------|
| Qwen3-8B (Thinking-On) | ~89 | ~660 | ~100 | 0% |
| SFT | **90.0+** | **~750** | ~50 | -13% (反而更多) |
| NFT (Thinking-Off) | ~65 | ~90 | 0 | ~86% ↓ |
| DPO + ALP + NLL | ~85 | ~230 | ~50 | ~65% ↓ |
| **Ada-RS-DPO** | **89.24** | **87.81** | **6.10** | **~87% ↓** |
| **Ada-RS-DAPO** | **87.97** | **81.66** | **4.22** | **~88% ↓** |

> ✅ **最高精度接近 SFT，但输出 token 减少约 80%，thinking rate 下降达 95%以上**

### 与基线方法的对比结果
- **相比 SFT**：
  - 精度相当甚至略优（89.24 vs ~90），但输出 token **减少超过 80%**，显著降低推理成本。
- **相比 DPO 基线**：
  - 普通 DPO 无法改变“always-think”的默认行为（thinking rate 接近 100%）
  - 加入 ALP 和 NLL 后有所改善，但仍不如 Ada-RS 效果明显。
- **Ada-RS-DPO vs Ada-RS-DAPO**：
  - 两者精度相近，但 **Ada-RS-DAPO 的 thinking rate 更低（4.22% vs 6.10%）**，说明其在 on-policy 设置下能更彻底地抑制冗余推理。

### 消融实验结果（Ablation Study）

#### （1）各组件作用分析（Table 1 & Figure 3）
| 方法 | Accuracy | Thinking Rate | 说明 |
|------|----------|----------------|------|
| DPO + ALP + RS（无 NLL） | 63.82% | 100% | 学习崩溃，陷入 always-think |
| Ada-RS-DPO（完整版） | **89.24%** | **6.10%** | 引入 NLL 后稳定性大幅提升 |

> 🔍 **发现**：仅靠拒绝采样 + ALP 不足以稳定训练，必须加入 **NLL 辅助损失** 才能防止退化。

#### （2）不同组件的影响（Figure 6.3 分析）
- **仅 DPO**：无法诱导选择性思考
- **+ NLL 损失**：显著提升 selectivity 和 accuracy
- **+ ALP 奖励**：进一步压缩输出长度，尤其在易解 prompt 上
- **+ Ada-RS 拒绝采样**：将上述收益进一步放大，集中更新于“既准确又简洁”的轨迹

#### （3）超参数敏感性分析（Appendix B, Table 2）
- **β_rs（rejection sampling 温度）**：
  - 过大（如 1.0）→ 接受率过高 → 仍保留大量低质量样本 → thinking rate 高
  - 过小（如 0.01）→ 接受率极低（仅 10%）→ 训练样本不足 → 收敛慢
  - 最佳范围：**0.1 ~ 0.01**（Ada-RS-DPO）、**0.1 ~ 0.5**（Ada-RS-DAPO）
- **σ（标准化标准差）**：
  - σ 过小 → 长度惩罚太弱 → reasoning 泛滥
  - σ 过大 → 抑制过度 → 影响 hard case 性能

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **训练信号的选择是实现高效推理的关键杠杆**：
   - 如何构造和过滤训练样本（而非仅仅修改目标函数）可以直接影响模型是否学会“选择性思考”。

2. ✅ **Ada-RS 能有效平衡 accuracy 与 efficiency**：
   - 在保持接近最优 accuracy 的同时，将平均输出 token 数减少 **70–80%**，thinking rate 降低 **至 4–6%**，远优于传统方法。

3. ✅ **Ada-RS 是通用且可插拔的框架**：
   - 成功集成到 DPO 和 DAPO 两种主流训练范式，在 off-policy 和 on-policy 设置下均有效。

4. ✅ **稳定性至关重要**：
   - 单纯使用拒绝采样会导致训练不稳定，需配合 **NLL 辅助损失** 来维持语言建模能力和防止 collapse。

5. ✅ **无需推理时开关机制**：
   - 模型通过训练内化了“何时思考”的决策能力，**无需依赖 inference-time prompting 或 gating 机制**。

### 方法的局限性
1. **领域和模型规模受限**：
   - 当前实验仅在单一电商 domain 和 Qwen3-8B 模型上验证，尚未扩展到更大模型或多领域。
2. **评估粒度较粗**：
   - 使用 per-step tool call accuracy，未衡量 end-to-end 用户目标完成率（goal completion rate）。
3. **依赖高质量 rollout 生成**：
   - Ada-RS 依赖 teacher policy 或当前 policy 生成多样化的候选样本，若初始 policy 质量差可能影响收敛。

### 未来工作方向
1. 将 Ada-RS 扩展到更多应用场景（如数学推理、代码生成）和更大规模模型（如 Qwen3-72B）。
2. 引入更细粒度的用户满意度指标（如对话完成度、用户体验评分）作为评估依据。
3. 探索与其他高效推理技术（如 reasoning pruning、early-exit）结合的可能性。
4. 研究如何自适应调节 β_rs 和 σ 参数以实现全自动调优。

--- 

> 📌 **一句话总结**：  
> Ada-RS 通过**自适应奖励驱动的拒绝采样机制**，实现了 LLM 在工具调用场景下的**高效选择性思考**，在几乎不牺牲 accuracy 的前提下，将输出长度和 thinking 频率压缩至原来的 **1/5 以下**，为延迟敏感系统提供了实用的部署方案。

</details>

---

### 16. [Why ReLU? A Bit-Model Dichotomy for Deep Network Training](https://arxiv.org/abs/2602.19017)

**Authors**: Ilan Doron-Arad, Elchanan Mossel  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.19017v1  

#### Abstract
Theoretical analyses of Empirical Risk Minimization (ERM) are standardly framed within the Real-RAM model of computation. In this setting, training even simple neural networks is known to be $\exists \mathbb{R}$-complete -- a complexity class believed to be harder than NP, that characterizes the dif...

---

### 17. [Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines](https://arxiv.org/abs/2602.19265)

**Authors**: Siavash Khodakarami, Vivek Oommen, Nazanin Ahmadi Daryakenari, Maxim Beekenkamp, George Em Karniadakis  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.19265v1  

#### Abstract
Solving partial differential equations (PDEs) by neural networks as well as Kolmogorov-Arnold Networks (KANs), including physics-informed neural networks (PINNs), physics-informed KANs (PIKANs), and neural operators, are known to exhibit spectral bias, whereby low-frequency components of the solutio...

---

### 18. [CTS-Bench: Benchmarking Graph Coarsening Trade-offs for GNNs in Clock Tree Synthesis](https://arxiv.org/abs/2602.19330)

**Authors**: Barsat Khadka, Kawsher Roxy, Md Rubel Ahmed  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.19330v1  

#### Abstract
Graph Neural Networks (GNNs) are increasingly explored for physical design analysis in Electronic Design Automation, particularly for modeling Clock Tree Synthesis behavior such as clock skew and buffering complexity. However, practical deployment remains limited due to the prohibitive memory and ru...

---

### 19. [ISO-Bench: Can Coding Agents Optimize Real-World Inference Workloads?](https://arxiv.org/abs/2602.19594)

**Authors**: Ayush Nangia, Shikhar Mishra, Aman Gokrani, Paras Chopra  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.19594v1  

#### Abstract
We introduce ISO-Bench, a benchmark for coding agents to test their capabilities on real-world inference optimization tasks. These tasks were taken from vLLM and SGLang, two of the most popular LLM serving frameworks. Each task provides an agent with a codebase and bottleneck description, whereby th...

---

### 20. [Training-Free Generative Modeling via Kernelized Stochastic Interpolants](https://arxiv.org/abs/2602.20070)

**Authors**: Florentin Coeurdoux, Etienne Lempereur, Nathana\"el Cuvelle-Magar, Thomas Eboli, St\'ephane Mallat, Anastasia Borovykh, Eric Vanden-Eijnden  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.20070v1  

#### Abstract
We develop a kernel method for generative modeling within the stochastic interpolant framework, replacing neural network training with linear systems. The drift of the generative SDE is $\hat b_t(x) = \nabla\phi(x)^\top\eta_t$, where $\eta_t\in\R^P$ solves a $P\times P$ system computable from data, ...

---

### 21. [ComplLLM: Fine-tuning LLMs to Discover Complementary Signals for Decision-making](https://arxiv.org/abs/2602.19458)

**Authors**: Ziyang Guo, Yifan Wu, Jason Hartline, Kenneth Holstein, Jessica Hullman  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.19458v1  

#### Abstract
Multi-agent decision pipelines can outperform single agent workflows when complementarity holds, i.e., different agents bring unique information to the table to inform a final decision. We propose ComplLLM, a post-training framework based on decision theory that fine-tunes a decision-assistant LLM u...

---

### 22. [Whisper: Courtside Edition Enhancing ASR Performance Through LLM-Driven Context Generation](https://arxiv.org/abs/2602.18966)

**Authors**: Yonathan Ron, Shiri Gilboa, Tammuz Dubnov  
**Category**: cs.CL  
**Published**: 2026-02-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.18966v1  

#### Abstract
Domain-specific speech remains a persistent challenge for automatic speech recognition (ASR), even for state-of-the-art systems like OpenAI's Whisper. We introduce Whisper: Courtside Edition, a novel multi-agent large language model (LLM) pipeline that enhances Whisper transcriptions without retrain...

---

### 23. [ucTrace: A Multi-Layer Profiling Tool for UCX-driven Communication](https://arxiv.org/abs/2602.19084)

**Authors**: Emir Gencer (Ko\c{c} University, Turkey), Mohammad Kefah Taha Issa (Ko\c{c} University, Turkey), Ilyas Turimbetov (Ko\c{c} University, Turkey), James D. Trotter (Simula Research Laboratory, Norway), Didem Unat (Ko\c{c} University, Turkey)  
**Category**: cs.DC  
**Published**: 2026-02-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.19084v1  

#### Abstract
UCX is a communication framework that enables low-latency, high-bandwidth communication in HPC systems. With its unified API, UCX facilitates efficient data transfers across multi-node CPU-GPU clusters. UCX is widely used as the transport layer for MPI, particularly in GPU-aware implementations. How...

---

### 24. [A Risk-Aware UAV-Edge Service Framework for Wildfire Monitoring and Emergency Response](https://arxiv.org/abs/2602.19742)

**Authors**: Yulun Huang, Zhiyu Wang, Rajkumar Buyya  
**Category**: cs.DC  
**Published**: 2026-02-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.19742v1  

#### Abstract
Wildfire monitoring demands timely data collection and processing for early detection and rapid response. UAV-assisted edge computing is a promising approach, but jointly minimizing end-to-end service response time while satisfying energy, revisit time, and capacity constraints remains challenging. ...

---

### 25. [Mitigating Artifacts in Pre-quantization Based Scientific Data Compressors with Quantization-aware Interpolation](https://arxiv.org/abs/2602.20097)

**Authors**: Pu Jiao, Sheng Di, Jiannan Tian, Mingze Xia, Xuan Wu, Yang Zhang, Xin Liang, Franck Cappello  
**Category**: cs.DC  
**Published**: 2026-02-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.20097v1  

#### Abstract
Error-bounded lossy compression has been regarded as a promising way to address the ever-increasing amount of scientific data in today's high-performance computing systems. Pre-quantization, a critical technique to remove sequential dependency and enable high parallelism, is widely used to design an...

---

### 26. [Variational Trajectory Optimization of Anisotropic Diffusion Schedules](https://arxiv.org/abs/2602.19512)

**Authors**: Pengxi Liu, Zeyu Michael Li, Xiang Cheng  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.19512v1  

#### Abstract
We introduce a variational framework for diffusion models with anisotropic noise schedules parameterized by a matrix-valued path $M_t(\theta)$ that allocates noise across subspaces. Central to our framework is a trajectory-level objective that jointly trains the score network and learns $M_t(\theta)...

---

### 27. [Decoding ML Decision: An Agentic Reasoning Framework for Large-Scale Ranking System](https://arxiv.org/abs/2602.18640)

**Authors**: Longfei Yun, Yihan Wu, Haoran Liu, Xiaoxuan Liu, Ziyun Xu, Yi Wang, Yang Xia, Pengfei Wang, Mingze Gao, Yunxiang Wang, Changfan Chen, Junfeng Pan  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.18640v1  

#### Abstract
Modern large-scale ranking systems operate within a sophisticated landscape of competing objectives, operational constraints, and evolving product requirements. Progress in this domain is increasingly bottlenecked by the engineering context constraint: the arduous process of translating ambiguous pr...

---

### 28. [High Dimensional Procedural Content Generation](https://arxiv.org/abs/2602.18943)

**Authors**: Kaijie Xu, Clark Verbrugge  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.18943v1  

#### Abstract
Procedural content generation (PCG) has made substantial progress in shaping static 2D/3D geometry, while most methods treat gameplay mechanics as auxiliary and optimize only over space. We argue that this limits controllability and expressivity, and formally introduce High-Dimensional PCG (HDPCG): ...

---

### 29. [Robust Exploration in Directed Controller Synthesis via Reinforcement Learning with Soft Mixture-of-Experts](https://arxiv.org/abs/2602.19244)

**Authors**: Toshihide Ubukata, Zhiyao Wang, Enhong Mu, Jialong Li, Kenji Tei  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.19244v1  

#### Abstract
On-the-fly Directed Controller Synthesis (OTF-DCS) mitigates state-space explosion by incrementally exploring the system and relies critically on an exploration policy to guide search efficiently. Recent reinforcement learning (RL) approaches learn such policies and achieve promising zero-shot gener...

---

### 30. [Uncovering Context Reliance in Unstructured Knowledge Editing](https://arxiv.org/abs/2602.19043)

**Authors**: Zisheng Zhou, Mengqi Zhang, Shiguang Wu, Xiaotian Ye, Chi Zhang, Zhumin Chen, Pengjie Ren  
**Category**: cs.CL  
**Published**: 2026-02-24  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.19043v1  

#### Abstract
Editing Large language models (LLMs) with real-world, unstructured knowledge is essential for correcting and updating their internal parametric knowledge. In this work, we revisit the fundamental next-token prediction (NTP) as a candidate paradigm for unstructured editing. We identify Context Relian...

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

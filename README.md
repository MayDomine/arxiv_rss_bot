# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-12-02 05:58:31 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SIMPLE: Disaggregating Sampling from GPU Inference into a Decision Plane for Faster Distributed LLM Serving](https://arxiv.org/abs/2512.00719)

**Authors**: Bohan Zhao, Zane Cao, Yongchao He  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2512.00719v1  

#### Abstract
As large language models (LLMs) scale out with tensor parallelism (TP) and pipeline parallelism (PP) and production stacks have aggressively optimized the data plane (attention/GEMM and KV cache), sampling, the decision plane that turns logits into tokens, becomes a new bottleneck. This creates a st...

---

### 2. [RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs](https://arxiv.org/abs/2512.00319)

**Authors**: Ruike Hu, Shulei Wu  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2512.00319v1  

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language generation and reasoning. However, their integration into automated software ecosystems is often hindered by the "Structure Gap" - the inherent tension between the probabilistic nature of token generation and ...

#### AI Summary (by kimi-k2-thinking)
# RL-Struct: 面向LLM可靠结构化输出的轻量级强化学习框架

## 1. 主要贡献和创新点

### 解决的问题
- **Structure Gap**：LLM作为概率性token生成器与结构化数据格式（JSON/XML等）的确定性要求之间的根本矛盾
- SFT的局限性：无法显式惩罚结构违规，导致"近似正确"的幻觉键或格式错误
- 约束解码的缺陷：推理时计算开销大，显著增加延迟，且不改善模型内部表示

### 提出的新方法
- **RL-Struct框架**：轻量级RL框架，通过多维奖励函数将结构化输出任务分解为层次化约束：
  - **结构完整性**（R_struct）：强制必需键的存在
  - **格式正确性**（R_format）：鼓励标准markdown格式
  - **语法有效性**（R_valid）：确保严格语法正确性（可解析的JSON）
  - **语义正确性**（R_correct）：衡量与ground truth的对齐程度
  - **长度约束**（R_length）：防止输出冗长
- **GRPO优化**：采用Gradient Regularized Policy Optimization，通过组内相对奖励估计baseline，**无需critic网络**

### 相比现有方法的优势
- **效率**：相比PPO减少**40% VRAM**（14.2GB vs 22.8GB），训练吞吐量提升60%（42 vs 26 samples/min）
- **性能**：在4B参数模型上实现**89.7%结构准确性**，超越LLaMA-3-8B（78.2%）和Phi-3-mini（74.1%）
- **零推理开销**：训练时干预，推理速度与标准生成相同，而约束解码方法（如Outlines）有**~6倍延迟**
- **样本效率**：仅需**1000个样本**即可达到>80%结构准确性

---

## 2. 核心实验方法和设置

### 数据集
- **主任务**：食谱生成（AkashPS11/recipes_data_food.com），生成包含ingredients、steps、nutritional_info的JSON
- **泛化评估**：
  - GSM8K-JSON：数学推理任务，要求分离推理步骤和最终答案
  - ToolUse：函数调用任务，生成API调用参数

### 实验设置
- **基础模型**：Qwen3-4B-Instruct
- **PEFT配置**：LoRA（rank=32, alpha=32），4-bit量化
- **训练参数**：250 steps，学习率5×10⁻⁶，cosine衰减，batch size=1，group size G=6
- **硬件**：单张NVIDIA RTX 4090（24GB）

### 评估指标
- **结构性指标**：结构准确性、JSON有效性、格式一致性、模式合规性
- **语义指标**：内容准确性（F1-Score + GPT-4-Turbo Judge评分）
- **效率指标**：推理延迟、VRAM占用、训练吞吐量

### 基线方法
- **零样本**：GPT-3.5-Turbo、Mistral-7B-Instruct
- **SFT**：Phi-3-mini-3.8B、LLaMA-3-8B-Instruct、Qwen3-4B
- **约束解码**：Qwen3-4B + Outlines
- **偏好优化**：Qwen3-4B + DPO

---

## 3. 主要实验结果和性能指标

### 核心性能数据
| 方法 | 结构准确性 | JSON有效性 | 内容准确性 |
|------|-----------|-----------|-----------|
| GPT-3.5 (Zero-shot) | 45.5% | 82.1% | 88.0% |
| LLaMA-3-8B (SFT) | 78.2% | 85.4% | 86.0% |
| Qwen3-4B + Outlines | **99.8%** | **100.0%** | 79.5% |
| Qwen3-4B + DPO | 82.5% | 88.4% | 82.0% |
| **RL-Struct (Ours)** | **89.7%** | **92.1%** | **84.5%** |

### 关键对比结果
- **vs SFT**：结构准确性提升**11.5-15.6个百分点**，JSON有效性提升**6.7-20个百分点**
- **vs DPO**：结构准确性提升**7.2个百分点**，验证group-based探索优于pairwise偏好
- **vs Outlines**：内容准确性高5个百分点，且**无推理延迟**（Outlines有~6倍延迟）

### 资源效率
- **VRAM占用**：GRPO 14.2GB vs PPO 22.8GB（**降低37.7%**）
- **训练速度**：42 samples/min vs 26 samples/min（**提升61.5%**）

### 样本效率
- **1000样本**：达到>80%结构准确性
- **SFT对比**：需要显著更多数据才能达到可比水平

### 泛化能力
| 任务 | RL-Struct | SFT | Zero-shot |
|------|-----------|-----|-----------|
| GSM8K-JSON | **85.4%** | 58.2% | 25.5% |
| ToolUse | **91.2%** | 70.1% | 30.0% |

### 消融实验结果
| 配置 | JSON有效性 | 结构准确性 |
|------|-----------|-----------|
| Full Reward | **92.1%** | **89.7%** |
| w/o R_valid | 68.3% (-23.8) | 85.2% |
| w/o R_struct | 90.5% | 45.6% (-44.1) |
| w/o R_format | 88.2% | 87.1% |

---

## 4. 关键结论和发现

### 主要发现
1. **自步课程学习（Emergent Curriculum）**：训练动态呈现两阶段
   - **阶段I（0-100 steps）**：快速学习语法（R_valid主导）
   - **阶段II（100-250 steps）**：优化语义内容（R_correct提升）
   - 无需手动设计课程，GRPO自动实现"先学会如何说，再学会说什么"

2. **梯度主导效应**：当R_valid≈0时，∇(w_valid·R_valid) ≫ ∇(w_correct·R_correct)，确保结构约束优先

3. **错误模式抑制**：几乎消除语法错误，显著减少幻觉键（hallucinated keys）和类型不匹配

4. **帕累托最优**：在推理延迟-结构准确性权衡中位于**帕累托前沿**，实现无延迟下的最佳结构可靠性

### 方法局限性
1. **奖励工程成本**：依赖手动设计的奖励组件，对动态模式需重新训练
2. **对抗鲁棒性**：对抗攻击（如"忽略先前指令"）偶尔可绕过约束，但比SFT更抗"jailbreak"
3. **模式泛化**：对未见模式的零样本泛化能力有限，需微调或适配
4. **序列长度**：为严格满足冗长模式可能生成稍长序列

### 未来工作方向
1. **混合策略**：结合RL微调模型与轻量级引导解码，处理动态模式约束
2. **LLM-as-a-Judge奖励**：用强教师模型提供密集标量反馈，替代脆弱启发式
3. **模式感知奖励学习**：从形式化模式定义（JSON Schema/XSD）自动生成奖励函数
4. **自适应奖励权重**：基于奖励方差实现动态权重调度，自动化课程学习
5. **多轮代理工作流**：扩展到多轮代理交互和更丰富的模式类型

---

## 核心结论
RL-Struct通过**层次化奖励分解**和**高效GRPO优化**，在**4B参数模型**上实现了媲美或超越更大模型的结构化输出能力，**无推理延迟**，**资源消耗降低40%**，揭示了RL在弥合"Structure Gap"中的独特优势，为构建可靠的LLM代理提供了轻量级解决方案。

---

### 3. [SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs](https://arxiv.org/abs/2512.00722)

**Authors**: Jiaming Xu, Jiayi Pan, Hanzhen Wang, Yongkang Zhou, Jiancai Ye, Yu Wang, Guohao Dai  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.00722v1  

#### Abstract
In this paper, we point out that the objective of the retrieval algorithms is to align with the LLM, which is similar to the objective of knowledge distillation in LLMs. We analyze the similarity in information focus between the distilled language model(DLM) and the original LLM from the perspective...

---

### 4. [Efficient and Programmable Exploration of Synthesizable Chemical Space](https://arxiv.org/abs/2512.00384)

**Authors**: Shitong Luo, Connor W. Coley  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.00384v1  

#### Abstract
The constrained nature of synthesizable chemical space poses a significant challenge for sampling molecules that are both synthetically accessible and possess desired properties. In this work, we present PrexSyn, an efficient and programmable model for molecular discovery within synthesizable chemic...

---

### 5. [Financial Text Classification Based On rLoRA Finetuning On Qwen3-8B model](https://arxiv.org/abs/2512.00630)

**Authors**: Zhiming Lian  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.00630v1  

#### Abstract
Financial text classification has increasingly become an important aspect in quantitative trading systems and related tasks, such as financial sentiment analysis and the classification of financial news. In this paper, we assess the performance of the large language model Qwen3-8B on both tasks. Qwe...

---

### 6. [Domain-Decomposed Graph Neural Network Surrogate Modeling for Ice Sheets](https://arxiv.org/abs/2512.01888)

**Authors**: Adrienne M. Propp, Mauro Perego, Eric C. Cyr, Anthony Gruber, Amanda A. Howard, Alexander Heinlein, Panos Stinis, Daniel M. Tartakovsky  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.01888v1  

#### Abstract
Accurate yet efficient surrogate models are essential for large-scale simulations of partial differential equations (PDEs), particularly for uncertainty quantification (UQ) tasks that demand hundreds or thousands of evaluations. We develop a physics-inspired graph neural network (GNN) surrogate that...

#### AI Summary (by kimi-k2-thinking)
# 论文总结：Domain-Decomposed Graph Neural Network Surrogate Modeling for Ice Sheets

## 1. 主要贡献和创新点

### 解决的问题
- **计算成本瓶颈**：冰盖模拟涉及大规模非线性PDE求解，单次高保真模拟需数小时至数天，不确定性量化(UQ)需数千次评估，传统方法计算不可行
- **几何灵活性不足**：现有深度算子网络(DeepONet)等方法绑定固定传感器位置和域几何，网格变化需完全重训练
- **GNN过平滑问题**：标准消息传递GNN在深层传播中丢失高频信息，难以捕捉冰厚、基底摩擦和速度场的尖锐梯度
- **数据稀缺性**：高质量冰盖模拟数据获取成本极高，需要数据高效的学习策略

### 提出的新方法
1. **物理启发的Bracket-based GNN架构**
   - 采用Hamiltonian bracket框架，将消息传递重构为能量守恒动力系统
   - 通过可学习的内积矩阵实现图注意力机制，避免特征同质化
   - 在潜在空间保持全局不变量，防止过平滑

2. **域分解(Domain Decomposition, DD)策略**
   - 将非结构化网格划分为物理一致的子域(subgraph)
   - 在各子域并行训练独立GNN代理模型
   - 推理时聚合子域预测，实现全局场预测

3. **迁移学习流水线**
   - 在子域或低保真数据上预训练
   - 通过参数迁移微调全局模型或其他子域模型
   - 实现"热启动"(warm start)加速收敛

### 相比现有方法的优势
- **几何无关性**：直接作用于非结构化网格，无需重网格化或插值
- **计算可扩展性**：DD降低内存需求，支持并行训练，训练时间显著缩短
- **数据效率**：迁移学习在数据有限时(仅5-10个样本)仍能实现高性能
- **物理一致性**：Bracket架构保证稳定性，避免非物理振荡
- **模型可迁移性**：训练好的模型可跨网格、跨域微调，适应动态冰盖系统

---

## 2. 核心实验方法和设置

### 数据集
- **模拟器**：MPAS-Albany Land Ice (MALI) 冰盖模型
- **研究对象**：格陵兰Humboldt冰川(约200万平方公里)
- **数据生成**：
  - 从贝叶斯后验分布采样基底摩擦场μ(x,y)
  - 模拟2007-2100年冰盖演化，每个模拟93个年度快照
  - 采用Mono-Layer Higher-Order (MOLHO)近似和Budd非线性滑动定律
- **图表示**：使用Voronoi网格的对偶Delaunay三角剖分
  - 节点：18,544个(速度自由度)
  - 边：54,962条(FEM连接性)
- **特征维度**：
  - 输入：7维节点特征(冰厚、床地形、基底摩擦、接地/漂浮冰指示器)
  - 输出：2维速度场(x,y分量)

### 实验设置
- **训练集**：25个基底摩擦场 × 40个时间步 = 1,000个样本
- **验证/测试集**：各20个独立模拟
- **归一化**：节点特征z-score标准化，边距离min-max标准化至[0,1]
- **硬件**：NVIDIA A100 40GB GPU集群

### 评估指标
- **主要指标**：均方误差(MSE)和相对测试误差
- **可视化**：速度场预测、绝对误差分布、 grounding line通量直方图
- **对比策略**：
  1. **Cold start**：从头训练全局模型
  2. **Warm start**：子域预训练→全局微调
  3. **Warm start + DD**：子域预训练→各子域微调→聚合预测

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 训练策略 | 收敛速度 | 终点区域误差 | 数据效率 |
|---------|---------|-------------|---------|
| Cold start (全局) | 基准 | 较高 | 需要25+摩擦场 |
| Warm start (迁移) | **加速2-3倍** | 降低30-50% | 5个摩擦场即可收敛 |
| Warm start + DD | **最快(4-5倍)** | **降低70-90%** | **5个摩擦场达到最优** |

### 与基线对比结果
- **定性对比** (Figure 5)：
  - Cold start：在接地线(grounding line)和快速流动终点区域误差显著(>100 m/yr)
  - Warm start：误差分布更均匀，终点区域误差降至~50 m/yr
  - **Warm start + DD**：终点区域误差几乎消除(<20 m/yr)，全场预测与真值几乎不可区分

- **收敛曲线** (Figure 6-7)：
  - 在相同epoch下，Warm start + DD的相对测试误差比Cold start低一个数量级
  - 预训练在**终点区域**(高变异度)比内陆区域效果更好，收敛更快且最终误差更低

### 消融实验结果
- **分区策略影响**：非重叠分区已足够，未观察到边界不连续或误差累积
- **注意力头数**：2头→4头无明显改善，表明模型对超参数不敏感
- **编码器/解码器宽度**：从32降至16会使误差翻倍，说明容量关键
- **物理正则化**：未添加质量守恒等物理约束，仅靠架构本身已足够稳定

---

## 4. 关键结论和发现

### 主要发现
1. **域分解的双重优势**：
   - **计算层面**：降低单GPU内存占用，支持并行训练
   - **模型层面**：子域成为知识转移的自然单元，缓解神经网络的频谱偏差(spectral bias)，使高频局部特征更易学习

2. **迁移学习的"热启动"效应**：
   - 在数据稀缺时(仅5个摩擦场)，Warm start使模型仍能收敛，而Cold start严重欠拟合
   - 预训练模型提供良好的注意力权重初始化，加速新任务学习

3. **非重叠分区的鲁棒性**：
   - 与传统PDE求解器不同，GNN代理无需重叠区域或粗网格来传递nullspace信息
   - 数据驱动特性+MSE目标函数自动保证边界一致性

4. **基础模型潜力**：
   - 在180个基底摩擦场样本上训练的模型能准确预测**均值** grounding line通量
   - 但分布方差被低估(过窄)，需**针对性微调**才能捕捉不确定性传播

### 方法局限性
- **分区算法**：当前基于谱聚类+尺寸惩罚k-means，最优分区策略仍需系统研究
- **UQ能力**：默认训练目标优化平均性能，需设计专门微调策略才能准确量化不确定性
- **训练成本**：单次训练约30小时，虽比传统模拟快数个量级但仍需优化
- **外推能力**：模型在训练分布外的参数泛化能力未充分验证

### 未来工作方向
1. **UQ专用微调**：设计强调输入变异性的训练目标(如对抗性损失、分布匹配损失)
2. **基础模型范式**：构建跨冰川、跨分辨率的预训练模型，支持快速适配新科学问题
3. **自适应分区**：开发基于物理特征和误差指标的自适应域分解策略
4. **重叠分区探索**：在节点数极少或感受野严重截断的极端情况下测试重叠分区
5. **多物理场扩展**：将方法推广至温度演化、冰架崩解等更复杂的耦合过程

---

## 总结
该工作通过**物理启发的GNN架构**、**域分解**和**迁移学习**的三重结合，为大规模PDE系统代理建模提供了可扩展、高效且准确的解决方案。在冰盖模拟中，该方法将训练时间缩短数倍，终点区域预测精度提升70%以上，并展现出作为**科学基础模型**的潜力。核心洞见在于：利用图结构的灵活性实现计算与模型的双重本地化，再通过知识迁移克服数据稀缺瓶颈，为气候科学中的不确定性量化开辟了新路径。

---

### 7. [FlexiWalker: Extensible GPU Framework for Efficient Dynamic Random Walks with Runtime Adaptation](https://arxiv.org/abs/2512.00705)

**Authors**: Seongyeon Park, Jaeyong Song, Changmin Shin, Sukjin Kim, Junguk Hong, Jinho Lee  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.00705v1  

#### Abstract
Dynamic random walks are fundamental to various graph analysis applications, offering advantages by adapting to evolving graph properties. Their runtime-dependent transition probabilities break down the pre-computation strategy that underpins most existing CPU and GPU static random walk optimization...

#### AI Summary (by kimi-k2-thinking)
# FlexiWalker: 高效动态随机游走的可扩展GPU框架

## 1. 主要贡献和创新点

### 解决的问题
动态随机游走（Dynamic Random Walks）的转移概率依赖于游走历史，只能在运行时动态计算，无法像静态随机游走那样通过预计算和缓存转移概率来优化。这导致现有CPU/GPU静态随机游走优化框架失效，迫使开发者编写手工调优的kernel，且无法适应工作负载多样性。

### 提出的新方法
**FlexiWalker** - 首个专为动态随机游走设计的高效GPU框架，包含三个核心创新：

1. **高性能采样Kernel**：
   - **eRVS**（增强型Reservoir Sampling）：采用统计等价的argmax转换，消除prefix sum需求；引入"jump technique"大幅减少随机数生成次数
   - **eRJS**（增强型Rejection Sampling）：使用理论上限边界替代精确的max reduction，消除全局归约操作

2. **轻量级运行时成本模型**：基于一阶成本估计，在**每个节点、每个采样步骤**动态选择最优采样策略（eRVS vs eRJS）

3. **编译时自动特化**：Flexi-Compiler通过LLVM/Clang静态分析用户提供的游走逻辑，自动生成优化的构建块和辅助函数

### 相比现有方法的优势
- **性能突破**：消除global reductions、冗余内存访问和随机数生成三大瓶颈
- **自适应性**：首次实现细粒度（per-node, per-step）的采样策略动态切换
- **易用性**：用户只需编写轻量级工作负载特定函数，框架自动完成优化
- **通用性**：成功支持Node2Vec、MetaPath、Second-Order PageRank等多种动态随机游走算法

---

## 2. 核心实验方法和设置

### 数据集
使用10个真实世界图数据集，涵盖社交网络、引文网络和网页图：
- **小规模**：com-youtube (1.1M顶点, 6M边), cit-patents (3.8M, 33M)
- **中规模**：Livejournal (4.8M, 86M), Orkut (3.1M, 234M), EU-2015 (11M, 522M)
- **大规模**：Arabic-2005 (23M, 1.1B), UK-2005 (39M, 1.6B), Twitter (42M, 2.4B), SK-2005 (51M, 3.6B), Friendster (66M, 3.6B)

### 实验设置
- **硬件**：AMD EPYC 9124P (16核32线程), 512GB DDR5 ECC内存, 最多4×NVIDIA A6000 GPU (48GB VRAM/卡)
- **软件**：Ubuntu, CUDA 12.1.1, cuRAND 10.3.2.106
- **工作负载**：Node2Vec、MetaPath、Second-Order PageRank（2nd PR），分别测试加权和未加权版本
- **游走长度**：80步（MetaPath为5步）
- **查询规模**：为图中每个节点生成游走查询

### 基线方法对比
6个代表性CPU/GPU基线：
- **CPU**：SOWalker (SOTA out-of-core), ThunderRW (SOTA in-memory)
- **GPU**：C-SAW (ITS-based), NextDoor (RJS-based), Skywalker (ALS-based), FlowWalker (RVS-based, SOTA GPU动态游走框架)

---

## 3. 主要实验结果和性能指标

### 整体性能提升
在均匀权重分布下，FlexiWalker相比**最佳基线**实现：
- **CPU基线**：**73.44×几何平均加速**（最高4246.71×）
- **GPU基线**：**5.91×几何平均加速**（最高1040.54×）

### 关键性能数据
| 场景 | 数据集 | 加速比 (vs CPU) | 加速比 (vs GPU) | 备注 |
|------|--------|----------------|----------------|------|
| 加权Node2Vec | Friendster | 4246.71× | 1040.54× | 基线出现OOT |
| 未加权Node2Vec | SK-2005 | - | 10.79× / 1040.54× | CSAW/Skywalker OOT |
| 加权MetaPath | Orkut | - | 1.12× | 小数据集NextDoor快0.05% |
| 2nd PR | EU-2015 | - | 无OOM/OOT | 其他基线普遍超时 |

### 权重分布鲁棒性测试
- **Power-law分布**（α=1.0-4.0）：相比NextDoor和FlowWalker，**26.60×和4.37×几何平均加速**
- **Degree-based分布**：相比NextDoor和FlowWalker，**最高10.24×和3.29×加速**，且在大图SK上唯一完成计算

### 消融实验结果
1. **Kernel优化效果**：
   - eRVS vs FlowWalker：**1.44-1.82×加速**（均匀分布），**1.47-1.81×加速**（skewed分布）
   - eRJS vs NextDoor：**54.49-1698.35×加速**（均匀分布），**最高7.27×加速**（skewed分布）

2. **运行时组件价值**：
   - 相比纯eRJS：**最高421.56×加速**（高度skewed分布）
   - 相比纯eRVS：**最高3.37×加速**
   - 运行时选择策略比随机选择快**15.86×**，比基于度数的启发式快**2.66×**

3. **多GPU扩展性**：4块GPU时实现**3.23×几何平均加速**（接近线性）

4. **能效**：相比KnightKing CPU框架，**最高10.15×能效提升**（joules/query）

---

## 4. 关键结论和发现

### 主要发现
1. **运行时分布动态性**：在2nd PR工作负载中，大量节点表现出高变异系数（CV），证明转移权重分布在游走过程中显著变化，验证了per-step自适应的必要性
2. **采样策略依赖性**：eRJS性能高度依赖权重分布skewness，而eRVS表现稳定；无单一策略在所有场景最优
3. **编译时分析有效性**：Flexi-Compiler能自动提取max/sum边界，预处理开销仅占总运行时间**0.46%-3.98%**
4. **选择策略智能性**：在高度skewed分布（α=1）下，FlexiWalker自动减少eRJS使用比例（从75%降至25%），避免性能陷阱

### 方法局限性
1. **动态图支持有限**：Flexi-Compiler无法处理运行时图拓扑更新（如边权重动态变化），会损害预处理值的准确性
2. **代码复杂度限制**：对递归调用、数据依赖循环、深层嵌套结构等复杂代码，编译器可能生成错误代码，此时安全回退到eRVS-only模式
3. **小图开销**：在极小数据集和短游走场景（如MetaPath），框架开销可能导致轻微性能下降（<0.05%）

### 未来工作方向
1. **分布式大图支持**：借鉴分布式GNN框架，通过图划分和多GPU通信支持万亿边规模图
2. **动态图扩展**：增加模块以更新预处理值和图拓扑，利用Flexi-Runtime的per-step选择优势
3. **低精度优化**：支持INT8等低精度边权重，初步实验显示仍保持27.59×加速优势
4. **通信优化**：针对I/O密集型特性，优化inter-GPU通信以提升分布式性能

---

**开源**：FlexiWalker已开源在 https://github.com/AIS-SNU/FlexiWalker

---

### 8. [Efficient Training of Diffusion Mixture-of-Experts Models: A Practical Recipe](https://arxiv.org/abs/2512.01252)

**Authors**: Yahui Liu, Yang Yue, Jingyuan Zhang, Chenxi Sun, Yang Zhou, Wencong Zeng, Ruiming Tang, Guorui Zhou  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.01252v1  

#### Abstract
Recent efforts on Diffusion Mixture-of-Experts (MoE) models have primarily focused on developing more sophisticated routing mechanisms. However, we observe that the underlying architectural configuration space remains markedly under-explored. Inspired by the MoE design paradigms established in large...

---

### 9. [KV Pareto: Systems-Level Optimization of KV Cache and Model Compression for Long Context Inference](https://arxiv.org/abs/2512.01953)

**Authors**: Sai Gokhale, Devleena Das, Rajeev Patwari, Ashish Sirasao, Elliott Delaye  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.01953v1  

#### Abstract
Long-context Large Language Models (LLMs) face significant memory bottlenecks during inference due to the linear growth of key-value (KV) cache with sequence length. While individual optimization techniques like KV cache quantization, chunked prefill, and model weight quantization have shown promise...

---

### 10. [Heimdall++: Optimizing GPU Utilization and Pipeline Parallelism for Efficient Single-Pulse Detection](https://arxiv.org/abs/2512.00398)

**Authors**: Bingzheng Xia, Zujie Ren, Kuang Ma, Xiaoqian Li, Wenda Li, Shuibing He  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2512.00398v1  

#### Abstract
With the increasing time and frequency resolution of modern radio telescopes and the exponential growth in observational data volumes, real-time single-pulse detection has become a critical requirement for time-domain radio astronomy. Heimdall, as a representative GPU-accelerated single-pulse search...

#### AI Summary (by kimi-k2-thinking)
# Heimdall++: 高效单脉冲检测的GPU利用率与流水线并行优化

## 1. 主要贡献和创新点

### 解决的问题
Heimdall++针对原始Heimdall单脉冲搜索工具在**GPU利用率低下**和**流水线并行性不足**方面的根本缺陷进行系统性优化：
- **GPU利用率不足**：原始Heimdall平均GPU利用率仅51.2%，大量计算周期被浪费
- **PCIe数据传输瓶颈**：处理1GB文件产生7.85GB主机-设备数据传输，造成严重的GPU stall问题
- **串行执行限制**：DM trials循环顺序执行，无法充分利用GPU大规模并行能力
- **多文件处理效率低**：多进程并发导致CUDA上下文竞争和频繁切换，小文件场景下开销接近计算时间
- **内存分配开销**：单次运行产生近20万次cudaMalloc调用，重复分配/释放造成显著延迟

### 提出的新方法
**Heimdall++采用三项核心创新设计：**

1. **细粒度多流并行架构**
   - 将DM trials循环分解为独立任务，通过OpenMP多线程与CUDA多流（multi-stream）机制实现并发执行
   - 每个线程分配DM trials子集，在独立CUDA流中异步启动Baseline Removal、Normalization、Matched Filtering和Peak Detection等内核
   - 实现Streaming Multiprocessor occupancy最大化，减少内核启动开销

2. **统一内存管理与零拷贝优化**
   - 采用CUDA Unified Memory替代显式cudaMemcpy操作
   - Dedispersion输入驻留设备内存，输出直接写入Unified Memory，由硬件自动管理页面迁移
   - 将数据传输量从7.85GB降至1.17GB（减少6.7倍），消除PCIe往返延迟

3. **多线程共享设备内存分配器**
   - 自定义内存池机制，维护全局已分配块队列和空闲块队列
   - 通过C++读写锁保证线程安全，实现跨DM trials的内存块复用
   - cudaMalloc调用次数减少41.2倍（198,392 → 4,810次）

4. **多文件流水线并行框架**
   - 将工作流解耦为**Pipeline Creation**（CPU端I/O与元数据准备）和**GPU Execution**（计算密集型处理）两个阶段
   - 通过无锁任务队列实现阶段间通信，支持跨文件的I/O与计算重叠
   - 单进程多线程模型避免CUDA上下文竞争，解决多进程并发下的资源争用问题

### 相比现有方法的优势
- **性能提升**：单文件处理速度提升2.66倍，多文件批处理提升2.05倍，GPU利用率从51%提升至92%
- **结果一致性**：保持与原始Heimdall完全一致的搜索结果，确保科学数据可靠性
- **可扩展性**：支持更高并发度（Heimdall最多2进程，Heimdall++稳定支持4线程以上）
- **资源效率**：显著降低PCIe带宽占用和内存分配开销，提升硬件资源利用率
- **适用性**：适用于大DM范围、高分辨率观测数据，避免内存溢出风险

---

## 2. 核心实验方法和设置

### 实验环境
- **硬件**：NVIDIA GeForce RTX 3080Ti GPU + 12th Gen Intel Core i9-12900K CPU
- **软件**：Ubuntu 20.04.6 LTS, NVIDIA CUDA Toolkit
- **基线方法**：原始Heimdall（启用CUDA Multi-Process Service用于多进程对比）

### 数据集
1. **单文件基准测试**：1GB filterbank文件（J0528_2200_arcdrift-M01_0009.fits），来自CRAFTS巡天项目
2. **大文件处理测试**：142GB filterbank文件（M5球状星团NGC 5904的30分钟观测数据，282个FITS文件合并）
3. **多文件批处理测试**：FAST-FREX数据集的FRB20201124子集，125个FITS文件（每个488MB），每文件含1个FRB信号

### 实验参数
- **DM范围**：0–1000 cm⁻³
- **Chunk大小**：256K samples
- **并行度设置**：1, 2, 4, 6, 8（线程/进程数）
- **评估指标**：
  - 端到端处理时间
  - GPU利用率（Nsight Systems profiling）
  - 主机-设备数据传输量
  - cudaMalloc调用次数
  - 各阶段执行时间分解

---

## 3. 主要实验结果和性能指标

### 单文件处理性能
| 并行度 | Heimdall++加速比 | GPU利用率 |
|--------|------------------|-----------|
| 1      | 1.42×            | -         |
| 2      | 2.06×            | -         |
| 4      | 2.84×            | -         |
| 6      | 3.26×            | -         |
| **8**  | **3.40×**        | **92.11%** |

- **GPU利用率对比**：Heimdall平均51.17% → Heimdall++平均92.11%，消除GPU stall间隙
- **数据传输优化**：总数据量从7.85GB降至1.17GB（Host-to-Device: 5.19GB→716.87MB, Device-to-Host: 2.66GB→480.75MB）

### 大文件（142GB）处理性能
- **加速比**：并行度8时达到**2.66×**（相比Heimdall的1.35×-2.66×随并行度提升）
- **GPU利用率**：平均77.73%（持续处理模式，低于单chunk的92%因含I/O开销）
- **双缓冲策略**：异步数据读取与计算重叠，有效掩盖I/O延迟

### 多文件批处理性能
| 并行度 | Heimdall性能 | Heimdall++加速比 |
|--------|--------------|------------------|
| 2      | 基准         | 1.41×            |
| 3      | 不支持       | 1.80×            |
| **4**  | **OOM失败**  | **2.05×**        |

- **可扩展性**：Heimdall在>2进程时因GPU内存不足中断，Heimdall++稳定支持4线程并发
- **资源竞争消除**：单进程多线程模型避免CUDA上下文切换开销，吞吐量提升显著

### 消融实验结果
**内存分配优化效果**：
- cudaMalloc调用次数：**198,392 → 4,810次**（减少41.2倍）
- 直接贡献DM trials循环阶段4.33×-6.05×加速

**各阶段独立加速**（单线程模式）：
- **RFI Mitigation**：3.25×（消除冗余数据传输）
- **Dedispersion**：1.59×（统一内存优化）
- **Candidate Merging & Clustering**：算法重构后内存访问模式优化，全局内存事务减少一个数量级

**DM trials循环内各阶段加速**（并行度8）：
- Normalization：6.05×
- Peak Detection：5.73×
- Baseline Removal：5.42×
- Matched Filtering：4.33×

---

## 4. 关键结论和发现

### 主要发现
1. **并行化是提升GPU利用率的关键**：通过将DM trials循环分解到多线程/多流，Heimdall++成功将GPU利用率提升至92%，证明细粒度并行化能有效解决原始Heimdall的串行瓶颈
2. **数据局部性优化收益巨大**：统一内存策略减少6.7倍PCIe传输量，是消除GPU stall的最有效手段，其开销相对于性能提升可忽略不计
3. **内存分配开销不容忽视**：在数千次DM trials场景下，重复的cudaMalloc/cudaFree成为显著瓶颈，内存池机制带来41倍调用次数减少
4. **多文件场景需架构级重构**：传统多进程方法在小文件场景下因上下文切换开销导致效率崩溃，流水线化单进程多线程模型是根本解决方案

### 方法局限性
- **硬件依赖性**：最优并行度受GPU计算能力限制，超过8线程后收益递减（受限于RTX 3080Ti的SM数量）
- **内存容量约束**：尽管使用Unified Memory，极端大DM范围或超高分辨率数据仍可能触发主机内存压力
- **固定开销影响**：Pipeline creation和预处理阶段的开销不随并行度缩放，在大文件场景下稀释整体加速比
- **线程完成时间不均**：DM trials负载不均衡导致部分线程提前完成，末期GPU利用率出现短暂下降

### 未来工作方向
1. **动态并行度调整**：根据输入数据特性（DM数量、chunk大小）和硬件配置自动选择最优线程数
2. **多GPU扩展**：将架构扩展至多GPU节点，支持分布式DM trials处理
3. **算法级优化**：探索更高效的候选聚类算法（如基于KD-Tree或哈希的空间索引），将O(N²)复杂度降至O(NlogN)
4. **实时处理集成**：与射电望远镜数据流直接集成，实现真正的在线实时单脉冲检测
5. **下一代望远镜适配**：针对SKA、FAST等未来设施的数据速率（数十至数百Tbps）进行前瞻性优化

### 科学意义
Heimdall++为时域射电天文学提供了**高吞吐量、低延迟**的单脉冲搜索解决方案，使实时处理现代望远镜海量数据成为可能，显著降低计算成本，为下一代大巡天项目（如CRAFTS、FAST-FRB）奠定软件基础。

---

### 11. [EDIT: Early Diffusion Inference Termination for dLLMs Based on Dynamics of Training Gradients](https://arxiv.org/abs/2512.00670)

**Authors**: He-Yen Hsieh, Hong Wang, H. T. Kung  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.00670v1  

#### Abstract
Diffusion-based large language models (dLLMs) refine token generations through iterative denoising, but answers often stabilize before all steps complete. We propose EDIT (Early Diffusion Inference Termination), an inference-time criterion that adaptively stops denoising once sufficient reasoning st...

#### AI Summary (by kimi-k2-thinking)
# EDIT: Early Diffusion Inference Termination for dLLMs Based on Dynamics of Training Gradients

## 1. 论文的主要贡献和创新点

### 解决的问题
- **diffusion LLMs的计算效率瓶颈**：dLLMs（diffusion-based large language models）通过迭代denoising生成token，但推理过程计算昂贵，因为使用固定的denoising steps数量，即使高质量输出早已出现
- **训练元数据的浪费**：现代LLM部署范式中，训练期间的优化动态（梯度信息）会产生关于参数重要性的丰富元数据，但这些信息在训练完成后通常被丢弃，造成信息浪费

### 提出的新方法
- **EDIT（Early Diffusion Inference Termination）**：一种利用训练时优化元数据来指导推理时自适应提前终止的方法
- **核心思想**：在SFT（Supervised Fine-Tuning）期间，AdamW优化器的moment estimates记录了哪些LoRA参数在推理任务学习中持续获得强方向性更新。这些"AdamW evolution"模式构成了模型学习到的推理路径图。EDIT将这些信息保存为紧凑的元数据（而非丢弃），在推理时通过比较当前token activations与这些保存的模式来检测推理是否完成

### 相比现有方法的优势
- **无需架构改动**：与现有diffusion language models无缝集成
- **极小存储开销**：仅需存储约1.5-2MB元数据（占8GB模型的0.02%）
- **理论保证**：提供PAC-style bounds和可验证的停止证书（certificates）
- **性能提升**：在减少11.8%-68.3%推理步骤的同时，多数任务上保持或提升准确率

---

## 2. 核心实验方法和设置

### 数据集
在5个推理benchmark上评估：
- **Countdown**：算术推理任务
- **Sudoku**：数独求解
- **MATH500**：数学问题求解
- **GSM8K**：小学数学应用题
- **GPQA**：研究生级别问答

### 实验设置
- **基线模型**：LLaDA-8B
- **训练数据**：s1数据集（Muennighoff et al., 2025）
- **微调方法**：仅更新QKV projections中的LoRA参数（rank=128）
- **序列长度**：128/256/512三种设置
- **硬件**：Intel XPU确保可复现性
- **超参选择**：使用20%训练数据作为验证集调参，避免测试集泄露

### 评估指标
- **准确率**：各benchmark的标准评估指标
- **效率**：平均denoising steps数量及相比full diffusion的减少百分比
- **理论合规性**：满足Corollary 6的提前终止实例比例

### 基线方法
- **LLaDA (No SFT)**：未微调的原始模型
- **LLaDA (SFT)**：完整SFT训练，使用固定64/128/256 steps（对应序列长度128/256/512）

---

## 3. 主要实验结果和性能指标

### 推理步骤减少（表2）
| Dataset (Seq Len) | Baseline Steps | EDIT Steps | Reduction |
|-------------------|----------------|------------|-----------|
| Countdown (128)   | 64             | 40.4       | **36.9%** |
| Countdown (256)   | 128            | 40.6       | **68.3%** |
| Sudoku (128)      | 64             | 38.3       | **40.2%** |
| Sudoku (256)      | 128            | 74.9       | **41.5%** |
| MATH500 (128)     | 64             | 38.1       | **40.5%** |
| MATH500 (256)     | 128            | 81.9       | **36.0%** |
| GSM8K (256)       | 128            | 103.5      | **19.2%** |
| GSM8K (512)       | 256            | 225.8      | **11.8%** |
| GPQA (128)        | 64             | 40.3       | **37.0%** |

**总体**：在所有任务上实现**11.8%至68.3%**的steps减少，短序列上收益更显著

### 准确率对比（表1）
| Dataset (Seq Len) | No SFT | SFT (Full) | EDIT (Ours) | EDIT提升 |
|-------------------|--------|------------|-------------|----------|
| Countdown (256)   | 19.5   | 20.7       | **31.6**    | **+10.9**|
| Sudoku (128)      | 10.4   | 11.4       | **16.1**    | **+4.7** |
| MATH500 (512)     | 36.0   | 35.4       | **36.6**    | **+1.2** |
| GSM8K (256)       | 75.8   | 77.0       | 77.6        | +0.6     |
| GPQA (256)        | 20.5   | 20.5       | **27.7**    | **+7.2** |

**关键发现**：
- 在**Countdown**和**Sudoku**等"crisp tasks"上提升最显著（最高+31.6%和+16.1%）
- 在**GSM8K (512)**上略有下降（81.2%→76.2%），因长推理链常在推理完成前就已稳定
- 整体平均准确率保持竞争力或提升

### 消融实验结果
- **模块选择**：Query projection的LoRA-B矩阵配合row-wise energy reduction效果最佳（KL divergence stability最低0.089）
- **LoRA-B vs LoRA-A**：LoRA-B参数在训练中显示出清晰的激活模式（Figure 4），而LoRA-A变化极小（Figure 5），证实LoRA-B更适合捕获推理路径
- **理论合规性**：使用PAC-style calibration后，**72.3%**的提前终止实例满足Corollary 6的理论安全边界

---

## 4. 关键结论和发现

### 主要发现
1. **训练元数据的价值**：AdamW优化轨迹包含丰富的推理路径信息，利用这些信息可避免"盲目"的推理计算浪费
2. **推理-训练对齐信号**：通过比较推理时的pseudo-gradients与SFT期间的梯度统计量（mean±variance band），可可靠检测推理收敛点（Figure 1）
3. **任务特异性**：EDIT在具有清晰、一致推理路径的任务（如Countdown、Sudoku、Molecular Biology、Astrophysics）上效果最佳，而在需要长链推理的任务（如GSM8K长序列）上需谨慎
4. **早停避免退化**：提前终止可防止后期denoising步骤覆盖正确的中间状态，这是准确率提升的关键机制

### 方法局限性
1. **依赖训练动态**：需要访问SFT期间的AdamW moment estimates，对已发布的模型不可用
2. **超参数敏感性**：需要任务特定的阈值(δ, Ω)，尽管提供了系统化的验证集调参协议
3. **评估范围有限**：仅在LoRA微调上验证，full-parameter训练的扩展是未来工作
4. **长序列挑战**：极长推理链可能在真正完成前就已稳定，导致过早终止

### 未来工作方向
1. **模型发布范式**：建议模型提供方在发布时包含优化元数据，使EDIT类方法可广泛应用
2. **自适应标准**：开发学习式或动态调整的(δ, Ω)选择机制
3. **扩展应用**：
   - Token-wise freezing（附录D）：为每个token提供实例级安全证书
   - Subspace EDIT（附录E）：用低维推理子空间替代单一向量
   - 动态层间计算分配、早期质量预测、识别需要额外推理预算的prompt
4. **系统性效率提升**：揭示训练信息被丢弃的普遍 inefficiency，推动更整体的ML pipeline设计

---

### 核心洞见
EDIT开创了一个新范式：**利用本应被丢弃的训练优化元数据来指导智能推理决策**。这不仅实现了显著的计算节省（最高68.3%），还在多数情况下提升了准确率，证明了"训练即推理"（training informs inference）的理念在diffusion LLMs中的巨大潜力。

---

### 12. [Multi-Path Collaborative Reasoning via Reinforcement Learning](https://arxiv.org/abs/2512.01485)

**Authors**: Jindi Lv, Yuhao Zhou, Zheng Zhu, Xiaofeng Wang, Guan Huang, Jiancheng Lv  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.01485v1  

#### Abstract
Chain-of-Thought (CoT) reasoning has significantly advanced the problem-solving capabilities of Large Language Models (LLMs), yet conventional CoT often exhibits internal determinism during decoding, limiting exploration of plausible alternatives. Recent methods attempt to address this by generating...

#### AI Summary (by kimi-k2-thinking)
# Multi-Path Collaborative Reasoning via Reinforcement Learning 论文总结

## 1. 主要贡献和创新点

### 解决的问题
- **CoT推理的局限性**：标准Chain-of-Thought (CoT)推理在解码过程中存在**内部确定性**，限制了合理替代方案的探索空间。当推理轨迹始于错误前提时，缺乏及时纠正机制会导致错误逻辑无阻碍传播。
- **Soft Thinking的不足**：近期提出的Soft Thinking方法虽通过连续语义空间增强表示能力，但仍受**自回归解码的贪婪性**限制，无法并行探索多样化推理路径。实验表明，Soft Thinking会沿单一主导语义轨迹快速收敛，引入额外推理噪声（如图1c、1d所示）。

### 提出的新方法：M3PO框架
提出**Multi-Path Perception Policy Optimization (M3PO)**，一种新颖的强化学习框架，核心创新包括：
- **多路径协作机制**：将并行策略rollout视为天然多样化的推理源，通过**无参数门控函数**实现跨路径动态交互，在推理阶段（thinking phase）融合集体洞察。
- **混合嵌入表示**：在每个推理步骤，通过凸组合构建混合思考嵌入：$\bar{h}_{i}^{(l)}=(1-\lambda)e_{i}^{(l)}+\lambda c_{i}^{(l)}$，平衡轨迹内在方向与跨路径信息。
- **分布相似性融合**：基于输出分布的余弦相似度计算协作权重，促进分布一致路径间的强交互，同时允许与分歧路径的受控多样性。

### 相比现有方法的优势
- **无需辅助数据**：利用RL训练中的自然rollout多样性，无需策划额外数据集。
- **参数高效**：协作机制完全无参数，不增加模型复杂度，保持与预训练模型的兼容性。
- **推理效率**：测试时采用单路径解码，与标准LLM推理效率一致，但通过训练内化鲁棒推理模式。
- **性能突破**：在知识密集型任务上平均提升**9.5%**，在STEM推理上超越更大规模模型。

---

## 2. 核心实验方法和设置

### 数据集
**知识密集型任务**（5个开放域和多跳QA数据集）：
- Natural Questions (NQ), TriviaQA, HotpotQA, 2WikiMultiHopQA (2WikiMQA), Bamboogle

**推理密集型STEM任务**（5个数学和科学数据集）：
- GSM8k, MATH, MATH500, MMLU-STEM, ARC-Challenge

### 实验设置
- **模型规模**：Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct（主要实验），Qwen2.5-7B-Instruct（参考）
- **训练配置**：单GPU训练，使用LoRA（rank=32, α=64），AdamW 8bit优化器，学习率5e-6，batch size 64
- **M3PO超参**：λ=0.1, T=0.1, group size N=4（知识任务）/8（推理任务）
- **评估指标**：Exact Match (EM) scores, Accuracy

### 基线方法对比
- **监督微调**：SFT
- **RL方法**：PPO, GRPO, RLOO, REINFORCE++
- **检索增强**：RAG, IRCoT, Search-o1
- **连续推理方法**：Soft Thinking, HRPO（混合推理）, Coconut, CODI
- **少样本推理**：DeepSeekMath-7B, Gemma-2-9B, MAmmoTH2-7B/8B

---

## 3. 主要实验结果和性能指标

### 知识基准测试结果（Table 1）
| 模型 | 方法 | NQ | TriviaQA | HotpotQA | 2WikiMQA | Bamboogle | **平均** |
|------|------|----|----------|----------|----------|-----------|---------|
| Qwen2.5-7B | RAG | 34.9 | 58.5 | 29.9 | 23.5 | 20.8 | 33.5 |
| Qwen2.5-1.5B | M3PO | **41.4** | 56.8 | 28.7 | 27.9 | 23.2 | **35.6** (+2.1%) |
| Qwen2.5-3B | M3PO | **44.1** | **61.0** | **33.2** | 31.4 | **31.2** | **40.2** (+6.7%) |

**关键发现**：
- 1.5B模型超越7B RAG基线**2.1%**，3B模型超越**6.7%**
- 在NQ数据集上提升最显著：1.5B模型达41.4% EM（+6.5% vs 7B RAG），3B模型达44.1%（+9.2%）
- 相比GRPO平均提升**9.5%**，验证跨路径协作超越单纯group-relative更新

### STEM基准测试结果（Table 2）
| 模型 | 方法 | GSM8k | MATH | MATH500 | MMLU-ST | ARC-C | **平均** |
|------|------|-------|------|---------|---------|-------|---------|
| Qwen2.5-7B | CoT | 85.4 | 49.8 | 46.4 | 72.3 | 63.7 | 63.5 |
| Qwen2.5-3B | M3PO | **84.8** | **60.7** | **63.0** | **61.6** | **82.6** | **70.5** (+5.3%) |

**关键发现**：
- 3B模型平均准确率**70.5%**，超越最强7B基线（MAmmoTH2-8B: 65.2%）**5.3%**
- MATH数据集达**60.7%**，显著超越最佳7B基线（49.8%）
- 全面超越HRPO、GRPO等RL方法，验证协作机制更有效内化鲁棒推理模式

### 消融实验结果

**连续推理范式对比**（图4）：
- **Hidden States**：零奖励，因隐藏状态与预训练嵌入空间分布不一致
- **Soft Thinking**：收敛慢，最终奖励低，存在噪声累积
- **HRPO**：混合推理有提升，但逊于M3PO
- **M3PO**：最快收敛，最高稳定奖励，验证显式多路径交互优越性

**跨路径融合机制**（图6）：
- **No Cross-Path**（λ=0）：性能最低，退化为标准CoT
- **Peer Mean**（均匀平均）：性能下降，分布分歧引入冲突信号
- **M3PO**（相似性加权）：最高奖励，选择性整合最一致信号

**超参数敏感性**：
- **λ系数**（图5）：λ≥0.5时性能崩溃，λ=0.1最优，需保持原始推理流主导性
- **温度T**（图7）：T=0.1时性能峰值，尖锐注意力权重增强相关信号聚焦

---

## 4. 关键结论和发现

### 主要发现
1. **多路径协作的有效性**：并行rollout作为独立推理源，通过跨路径交互可打破自强化逻辑循环，培养更可靠的多步推理模式。这种机制符合人类认知中并行维护多个假设的特点。

2. **参数效率与性能平衡**：M3PO在**不增加任何可训练参数**的情况下，通过轻量级协作机制实现SOTA性能，在1.5B/3B小模型上解锁了超越7B大模型的能力。

3. **训练稳定性**：相比GRPO、HRPO在训练后期出现性能崩溃（图12-15），M3PO展现出卓越的**训练稳定性**，同时生成最紧凑的推理链，降低计算开销。

4. **定性优势**：Soft Thinking存在格式错误、逻辑不连续和噪声插入（图8、16），而M3PO生成**干净、连贯、格式一致**的推理过程。HRPO出现重复循环行为，M3PO则无退化模式（图17）。

### 方法局限性
- **规模限制**：计算资源限制仅在最多3B参数模型上验证，更大模型的可扩展性待探索
- **协作机制**：当前采用固定超参（λ, T），未来可研究**自适应协作机制**动态调整路径融合强度
- **任务范围**：主要聚焦知识问答和STEM推理，其他复杂任务（如代码生成）的泛化能力需进一步验证

### 未来工作方向
- **模型规模化**：在7B及以上参数规模验证M3PO的有效性，探索协作机制随模型容量的扩展行为
- **自适应机制**：开发动态λ和T调度策略，根据任务难度和推理阶段自动调整跨路径交互强度
- **更广泛评估**：扩展至代码推理、长文本生成等任务，验证通用性
- **理论分析**：深入研究多路径协作如何影响策略优化景观，解释其提升鲁棒性的理论机制

---

**总结**：M3PO通过创新的多路径协作RL框架，在保持参数和推理效率的前提下，显著提升了LLM的推理鲁棒性，为高效推理模型的发展提供了有前景的方向。

---

### 13. [Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling](https://arxiv.org/abs/2512.02010)

**Authors**: Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, Song Han  
**Category**: cs.CL  
**Published**: 2025-12-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.02010v1  

#### Abstract
As large language models have grown larger, low-precision numerical formats such as NVFP4 have become increasingly popular due to the speed and memory benefits they provide. However, to accelerate computation with NVFP4, all matrix multiplication operands--weights and activations in the forward pass...

---

### 14. [A Parallel and Distributed Rust Library for Core Decomposition on Large Graphs](https://arxiv.org/abs/2512.00233)

**Authors**: Davide Rucci, Sebastian Parfeniuc, Matteo Mordacchini, Emanuele Carlini, Alfredo Cuzzocrea, Patrizio Dazzi  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.00233v1  

#### Abstract
In this paper, we investigate the parallelization of $k$-core decomposition, a method used in graph analysis to identify cohesive substructures and assess node centrality. Although efficient sequential algorithms exist for this task, the scale of modern networks requires faster, multicore-ready appr...

#### AI Summary (by kimi-k2-thinking)
# 论文核心结论与实验结果总结

## 1. 主要贡献和创新点

**解决的问题：**
- 大规模图数据的k-core分解计算瓶颈：现有顺序算法（如Batagelj-Zaversnik）无法应对现代图数据（数亿顶点、数十亿边）的规模
- 分布式算法在共享内存架构上的适配难题：同步开销、缓存一致性、共享数据结构竞争等问题

**提出的新方法：**
- **共享内存适配的分布式协议**：将Montresor等人的去中心化消息传递算法从分布式环境迁移到共享内存多核系统
- **三阶段渐进式优化实现**：
  - **SequentialK**：基于消息队列的顺序基线实现，验证协议正确性
  - **ParallelK**：采用多线程消息传递，使用Rayon并行迭代器或原生线程池
  - **FastK**：核心创新版本，采用全局共享状态向量（`est`和`active`）、缓存感知数据结构、选择性消息传播和动态并行度调整

**相比现有方法的优势：**
- **性能突破**：FastK在16线程下实现**最高11倍加速**，比NetworkX快**两个数量级**
- **内存安全**：利用Rust所有权模型消除数据竞争，同时保持底层控制
- **同步优化**：通过选择性消息发送减少90%以上无效通信，动态切换并行/顺序执行模式
- **缓存友好**：排序向量替代HashMap带来**30%性能提升**，利用现代CPU缓存层次结构

---

## 2. 核心实验方法和设置

**数据集（SNAP库）：**
| 图类型 | 数据集 | 规模 | 最大核数k_max |
|--------|--------|------|---------------|
| 道路网络 | roadNet-CA/PA/TX | 100-200万节点，150-280万边 | 3 |
| 网页图 | web-Google/Stanford/BerkStan | 28-68万节点，230-760万边 | 44-201 |
| 社交网络 | soc-Pokec/LiveJournal1 | 160-480万节点，3060-6899万边 | 47-372 |
| 通信网络 | wiki-Talk | 239万节点，502万边 | 131 |

**实验环境：**
- **硬件**：双路AMD EPYC 7551（32核×2，共64核），128GB共享内存
- **软件**：Rust 1.x（`--release`模式 + LTO优化），线程数配置：1, 2, 4, ..., 128
- **评估指标**：运行时间（不含图加载）、收敛速度（中间结果与最终解的L1距离）、活跃节点比例

**基线方法：**
- **NetworkX 2.x**：Python参考实现（顺序算法）
- **SequentialK**：本论文顺序基线
- **ParallelK**：本论文并行基线（三种线程策略）

---

## 3. 主要实验结果和性能指标

**关键性能数据（16线程）：**
| 数据集 | FastK (秒) | NetworkX (秒) | 加速比 | SequentialK (秒) | 并行加速 |
|--------|------------|---------------|--------|------------------|----------|
| soc-LiveJournal1 | **3.87** | 480.92 | **124×** | 78.11 | 20.2× |
| web-BerkStan | **0.59** | 686.63 | **1164×** | 5.19 | 8.8× |
| wiki-Talk | **0.61** | 115.28 | **189×** | 8.76 | 14.4× |
| roadNet-CA | **0.35** | 40.48 | **116×** | 2.45 | 7.0× |

**加速比分析：**
- **线性扩展性**：在16线程前接近理想加速比，之后因同步开销增长而趋于饱和
- **最佳线程数**：社交网络类图在16-32线程达到峰值，道路网络因k_max=3导致并行收益较低

**消融实验结果：**
1. **数据结构选择**：排序向量 vs HashMap
   - **平均提速30%**，在web-Stanford上最高达**40%**
   - 原因：小度分布（平均度<20）下，缓存局部性优于渐近复杂度

2. **并行策略对比**（6线程固定）：
   - **原生线程**：基准性能
   - **Rayon线程池**：开销增加**15-20%**
   - **Rayon并行迭代器**：开销增加**25-30%**
   - **结论**：原生线程+手动batching最优

3. **Batch Size调优**：
   - **最优值**：256节点/线程
   - 过小（64）导致线程管理开销过大，过大（1024）导致负载不均衡

4. **收敛动态**：
   - **快速收敛**：前10-20次迭代完成95%以上节点
   - **长尾效应**：最后5%节点需50-300次迭代微调（±1误差）
   - **动态切换**：当活跃节点数 < batch size时切换至顺序模式，减少**20%**总时间

---

## 4. 关键结论和发现

**主要发现：**
1. **分布式协议可有效适配共享内存**：Montresor的消息传递模型通过**全局共享向量**和**原子操作**重构后，同步开销降低**60%以上**
2. **消息选择性至关重要**：通过`est[u] < est[v]`条件过滤，减少**80-90%**无效消息，显著降低锁竞争
3. **动态并行度调整有效**：根据活跃节点比例自适应切换并行/顺序模式，避免**"过度并行化"**导致的资源浪费
4. **Rust的零成本抽象验证**：在保持内存安全前提下，性能与C++相当，开发效率更高

**方法局限性：**
- **静态图假设**：未处理动态增删边，仅支持批量离线计算
- **单节点限制**：实验仅限单NUMA节点，跨节点通信开销未评估
- **内存限制**：图必须完全加载到内存，不支持外存计算
- **负载均衡**：高度偏斜的度分布（如星型图）可能导致batch间负载不均

**未来工作方向：**
1. **真分布式扩展**：保留共享内存内核，增加跨节点RPC层
2. **动态图维护**：支持增量/减量k-core更新，参考[27]的时序图工作
3. **NUMA感知分区**：针对多路服务器优化内存亲和性，减少跨socket访问
4. **异构加速**：探索GPU辅助的邻居列表处理，利用CUDA Graphs加速批量消息传递
5. **外部内存支持**：开发基于SSD的流式处理，处理万亿边级图

---

**核心结论**：本工作证明了**Rust在高性能图计算中的实用性**，通过**协议重构**和**架构感知优化**，将理论上的分布式算法转化为高效的共享内存实现，为大规模图分析提供了**安全、快速、可扩展**的解决方案。

---

### 15. [Clinical-R1: Empowering Large Language Models for Faithful and Comprehensive Reasoning with Clinical Objective Relative Policy Optimization](https://arxiv.org/abs/2512.00601)

**Authors**: Boyang Gu, Hongjian Zhou, Bradley Max Segal, Jinge Wu, Zeyu Cao, Hantao Zhong, Lei Clifton, Fenglin Liu, David A. Clifton  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.00601v1  

#### Abstract
Recent advances in large language models (LLMs) have shown strong reasoning capabilities through large-scale pretraining and post-training reinforcement learning, demonstrated by DeepSeek-R1. However, current post-training methods, such as Grouped Relative Policy Optimization (GRPO), mainly reward c...

#### AI Summary (by kimi-k2-thinking)
# Clinical-R1 论文核心结论与实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
- **单目标优化的局限性**：现有RL后训练方法（如GRPO）仅优化答案正确性，忽视了医疗等高风险领域对推理过程**忠实性(faithfulness)** 和**全面性(comprehensiveness)** 的严格要求
- **临床推理对齐挑战**：医疗场景需要模型提供可验证、可追溯的逐步推理，而现有方法产生的推理片段化或走捷径，难以获得临床信任

### 提出的新方法
- **Clinical-Objective Relative Policy Optimization (CRPO)**：一种可扩展的多目标可验证强化学习框架
  - 基于GRPO扩展，引入**基于规则的可验证奖励信号**
  - 联合优化三个目标：**准确性(accuracy)**、**忠实性(faithfulness)** 和**全面性(comprehensiveness)**
  - 无需人工标注，完全自动化

### 相比现有方法的优势
- **多目标平衡**：在保持准确性的同时，显著提升推理的临床可信度和完整性
- **可验证性**：强制模型遵循结构化格式（`<dx>`和`<conclusion>`标签），使推理过程可审计
- **计算高效**：在有限计算资源下保持训练稳定性和效率
- **临床对齐**：模拟临床医生的双系统思维（System 1直觉 + System 2分析）

---

## 2. 核心实验方法和设置

### 数据集
| 数据集 | 类型 | 规模 | 特点 |
|--------|------|------|------|
| **MedQA** | 医疗执照考试题 | 12,723题（英文） | USMLE等考试，4选项 |
| **MedMCQA** | 印度医学入学考试 | 194,000+题 | AIIMS/NEET PG，多科目 |
| **MedXpertQA** | 专家级推理基准 | 4,460题 | 17个专科，11个身体系统，文本题 |

**使用策略**：MedQA用于训练和领域内测试，MedMCQA和MedXpertQA用于领域外评估

### 实验设置
- **基础模型**：Qwen2.5-3B-Instruct
- **训练框架**：Volcano Engine RL (verl)
- **硬件**：8×A6000 48GB
- **训练配置**：
  - **Cold Start**：在5,000个MedQA样本上蒸馏DeepSeek-R1，13个epoch
  - **CRPO/GRPO训练**：在剩余5,000个MedQA样本上训练20个epoch
  - **Rollout数量**：G=5
  - **准确性权重系数**：k=10

### 评估指标

#### 认知行为评估（LLM-as-judge）
- **Backtracking**：识别并修正错误推理步骤
- **Answer Verification**：显式检查答案一致性
- **Subgoal Setting**：引入中间目标结构化推理
- **Backward-Chaining**：从候选答案反向推导证据

#### 医学忠实性评估（LLM-as-judge）
- **Faithfulness to Medical Knowledge**：临床相关事实声明与案例对齐程度
- **CECD (Case-grounded Evidence Citation Density)**：推理中明确引用患者具体发现的密度
- **DRC (Distractor Rejection Coverage)**：对错误选项的临床有效反驳覆盖率
- **Hallucination**：检测不支持或虚构的声明

### 基线方法
1. **Baseline**：原始Qwen2.5-3B + CoT prompting
2. **GRPO**：仅优化正确性和通用思维格式
3. **Cold Start + GRPO**：蒸馏后的模型 + GRPO
4. **CRPO**：直接应用CRPO
5. **Cold Start + CRPO (Clinical-R1-3B)**：完整提出的方法

---

## 3. 主要实验结果和性能指标

### 准确性对比
| 方法 | MedQA | MedMCQA | MedXpertQA | 平均提升 |
|------|-------|---------|------------|----------|
| Baseline | 41.95% | 46.78% | 10.51% | - |
| GRPO | 50.35% | 49.87% | 12.64% | +3.8% |
| **CRPO** | **52.41%** | **55.13%** | **14.88%** | **+7.3%** |
| Cold Start + GRPO | 51.07% | 48.86% | 12.64% | +4.0% |
| **Cold Start + CRPO** | **53.07%** | **58.10%** | **16.14%** | **+9.1%** |

**结论**：CRPO在所有数据集上均优于GRPO，Cold Start + CRPO组合效果最佳

### 认知行为提升（MedMCQA示例）
| 指标 | Baseline | GRPO | CRPO | Cold Start + GRPO | Cold Start + CRPO |
|------|----------|------|------|-------------------|-------------------|
| Backtracking | 0.36 | 0.40 | **0.73** | 1.88 | **2.49** |
| Backward-Chaining | 0.19 | 0.25 | **0.62** | 1.00 | **1.06** |
| Subgoal Setting | 2.04 | 1.88 | **2.51** | 3.39 | **4.23** |
| Verification | 0.18 | 0.17 | **0.53** | 0.89 | **1.28** |

**结论**：CRPO显著增强所有认知行为，Cold Start提供更好初始化

### 医学忠实性提升（MedMCQA示例）
| 指标 | Baseline | GRPO | CRPO | Cold Start + GRPO | Cold Start + CRPO |
|------|----------|------|------|-------------------|-------------------|
| Faithfulness | 4.66 | 4.71 | **7.13** | 7.66 | **12.95** |
| CECD | 1.42 | 1.51 | **5.36** | 2.31 | **5.76** |
| DRC | 1.84 | 1.75 | **2.79** | 2.33 | **3.36** |
| Hallucination↓ | 0.40 | 0.85 | **0.64** | 0.81 | **0.66** |

**结论**：CRPO在CECD和DRC上提升最显著，幻觉率更低

### 响应长度分析
- **Cold Start**：产生冗长、枚举式的推理链（约1500 tokens）
- **GRPO**：过度压缩响应（约1100 tokens），可能丢失必要信息
- **CRPO**：平衡长度（约1300 tokens），**在保留临床必要结构的同时剪枝冗余**

---

## 4. 关键结论和发现

### 主要发现
1. **多目标优化的有效性**：CRPO通过可验证的奖励设计，成功将模型能力转化为**可审计的临床工作流程**，使推理忠实于病例、系统性地拒绝干扰项
2. **结构化推理的必要性**：`<dx>`和`<conclusion>`标签强制模型**分离诊断过程与结论**，并要求结论明确引用诊断部分的证据，显著提升忠实性
3. **Cold Start的协同效应**：蒸馏提供领域对齐的初始化，CRPO提供约束规范，二者结合产生最佳效果
4. **认知行为涌现**：CRPO训练出的模型表现出更多**回溯、验证、子目标设定**等高级推理行为，而非简单模式匹配

### 方法局限性
1. **训练不稳定性**：CRPO优化可能不稳定，特别是没有Cold Start时，策略与参考模型间的KL散度增长过快
2. **评估依赖LLM**：认知行为和医学忠实性评估依赖LLM-as-judge（Llama-3.1-8B和GPT-5），可能与人类判断不完全一致
3. **基础模型限制**：实验基于通用模型Qwen2.5-3B，未在医学语料上预训练，领域知识表示深度有限
4. **任务范围**：当前聚焦非影像学的多选题QA，未覆盖更复杂的临床决策场景

### 未来工作方向
1. **算法改进**：探索更稳定的CRPO变体（如自适应或离策略更新）
2. **评估增强**：引入人类在环评估，验证自动评估的可靠性
3. **模型扩展**：在更强的医学专用基础模型上应用CRPO
4. **场景拓展**：扩展到多模态临床数据、开放式问答和实时决策支持
5. **安全性研究**：进一步研究如何减少幻觉和不当推理

### 核心贡献总结
CRPO为医疗AI提供了一条**可扩展、可验证、多目标**的RL后训练路径，使小型模型（3B参数）能够生成**可信、完整、临床对齐**的推理，为构建更安全、更协作的临床AI系统奠定基础。

---

### 16. [ARCADIA: Scalable Causal Discovery for Corporate Bankruptcy Analysis Using Agentic AI](https://arxiv.org/abs/2512.00839)

**Authors**: Fabrizio Maturo, Donato Riccio, Andrea Mazzitelli, Giuseppe Bifulco, Francesco Paolone, Iulia Brezeanu  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.00839v1  

#### Abstract
This paper introduces ARCADIA, an agentic AI framework for causal discovery that integrates large-language-model reasoning with statistical diagnostics to construct valid, temporally coherent causal structures. Unlike traditional algorithms, ARCADIA iteratively refines candidate DAGs through constra...

---

### 17. [Probabilistic Neuro-Symbolic Reasoning for Sparse Historical Data: A Framework Integrating Bayesian Inference, Causal Models, and Game-Theoretic Allocation](https://arxiv.org/abs/2512.01723)

**Authors**: Saba Kublashvili  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.01723v1  

#### Abstract
Modeling historical events poses fundamental challenges for machine learning: extreme data scarcity (N << 100), heterogeneous and noisy measurements, missing counterfactuals, and the requirement for human interpretable explanations. We present HistoricalML, a probabilistic neuro-symbolic framework t...

---

### 18. [Steady and Energy-Efficient Multi-Hop Clustering for Flying Ad-Hoc Networks (FANETs)](https://arxiv.org/abs/2512.00623)

**Authors**: Basilis Mamalis, Marios Perlitis  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.00623v1  

#### Abstract
Flying Ad-hoc Networks (FANETs), formed by Unmanned Aerial Vehicles (UAVs), represent an emerging and promising communication paradigm. These networks face unique challenges due to UAVs high mobility, limited energy resources, and dynamic topology. In this work, we propose a novel multi-hop clusteri...

---

### 19. [Scalable and Interpretable Scientific Discovery via Sparse Variational Gaussian Process Kolmogorov-Arnold Networks (SVGP KAN)](https://arxiv.org/abs/2512.00260)

**Authors**: Y. Sungtaek Ju  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.00260v1  

#### Abstract
Kolmogorov-Arnold Networks (KANs) offer a promising alternative to Multi-Layer Perceptron (MLP) by placing learnable univariate functions on network edges, enhancing interpretability. However, standard KANs lack probabilistic outputs, limiting their utility in applications requiring uncertainty quan...

---

### 20. [Upcycled and Merged MoE Reward Model for Mitigating Reward Hacking](https://arxiv.org/abs/2512.00724)

**Authors**: Lingling Fu  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.00724v1  

#### Abstract
Reward models play a critical role in Reinforcement Learning from Human Feedback (RLHF) by assessing the consistency between generated outputs and human preferences. However, conventional reward models are prone to reward hacking or over-optimization, where the policy exploits shortcut patterns to o...

---

### 21. [When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](https://arxiv.org/abs/2512.00709)

**Authors**: Yifan Xu, Xichen Ye, Yifan Chen, Qiaosheng Zhang  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.00709v1  

#### Abstract
Quality of datasets plays an important role in large language model (LLM) alignment. In collecting human feedback, however, preference flipping is ubiquitous and causes corruption in data annotation; the issue necessitates the alignment algorithms with improved robustness against potential flipped p...

---

### 22. [Conveying Imagistic Thinking in Traditional Chinese Medicine Translation: A Prompt Engineering and LLM-Based Evaluation Framework](https://arxiv.org/abs/2512.01198)

**Authors**: Jiatong Han  
**Category**: cs.CL  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.01198v1  

#### Abstract
Traditional Chinese Medicine (TCM) theory is built on imagistic thinking, in which medical principles and diagnostic and therapeutic logic are structured through metaphor and metonymy. However, existing English translations largely rely on literal rendering, making it difficult for target-language r...

---

### 23. [Elastic Mixture of Rank-Wise Experts for Knowledge Reuse in Federated Fine-Tuning](https://arxiv.org/abs/2512.00902)

**Authors**: Yebo Wu, Jingguang Li, Zhijiang Guo, Li Li  
**Category**: cs.DC  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.00902v1  

#### Abstract
Federated fine-tuning offers a promising solution for adapting Large Language Models (LLMs) to downstream tasks while safeguarding data privacy. However, its high computational and communication demands hinder its deployment on resource-constrained devices. In this paper, we propose SmartFed, a reso...

---

### 24. [Hybrid Context-Fusion Attention (CFA) U-Net and Clustering for Robust Seismic Horizon Interpretation](https://arxiv.org/abs/2512.00191)

**Authors**: Jose Luis Lima de Jesus Silva, Joao Pedro Gomes, Paulo Roberto de Melo Barros Junior, Vitor Hugo Serravalle Reis Rodrigues, Alexsandro Guerra Cerqueira  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.00191v1  

#### Abstract
Interpreting seismic horizons is a critical task for characterizing subsurface structures in hydrocarbon exploration. Recent advances in deep learning, particularly U-Net-based architectures, have significantly improved automated horizon tracking. However, challenges remain in accurately segmenting ...

---

### 25. [Projection-Free CNN Pruning via Frank-Wolfe with Momentum: Sparser Models with Less Pretraining](https://arxiv.org/abs/2512.01147)

**Authors**: Hamza ElMokhtar Shili, Natasha Patnaik, Isabelle Ruble, Kathryn Jarjoura, Daniel Suarez Aguirre  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.01147v1  

#### Abstract
We investigate algorithmic variants of the Frank-Wolfe (FW) optimization method for pruning convolutional neural networks. This is motivated by the "Lottery Ticket Hypothesis", which suggests the existence of smaller sub-networks within larger pre-trained networks that perform comparatively well (if...

---

### 26. [Sum Rate Maximization in STAR-RIS-UAV-Assisted Networks: A CA-DDPG Approach for Joint Optimization](https://arxiv.org/abs/2512.01202)

**Authors**: Yujie Huang, Haibin Wan, Xiangcheng Li, Tuanfa Qin, Yun Li, Jun Li, Wen Chen  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.01202v1  

#### Abstract
With the rapid advances in programmable materials, reconfigurable intelligent surfaces (RIS) have become a pivotal technology for future wireless communications. The simultaneous transmitting and reflecting reconfigurable intelligent surfaces (STAR-RIS) can both transmit and reflect signals, enablin...

---

### 27. [Efficient Hyperparameter Search for Non-Stationary Model Training](https://arxiv.org/abs/2512.01258)

**Authors**: Berivan Isik, Matthew Fahrbach, Dima Kuzmin, Nicolas Mayoraz, Emil Praun, Steffen Rendle, Raghavendra Vasudeva  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.01258v1  

#### Abstract
Online learning is the cornerstone of applications like recommendation and advertising systems, where models continuously adapt to shifting data distributions. Model training for such systems is remarkably expensive, a cost that multiplies during hyperparameter search. We introduce a two-stage parad...

---

### 28. [Forget Less, Retain More: A Lightweight Regularizer for Rehearsal-Based Continual Learning](https://arxiv.org/abs/2512.01818)

**Authors**: Lama Alssum, Hasan Abed Al Kader Hammoud, Motasem Alfarra, Juan C Leon Alcazar, Bernard Ghanem  
**Category**: cs.LG  
**Published**: 2025-12-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.01818v1  

#### Abstract
Deep neural networks suffer from catastrophic forgetting, where performance on previous tasks degrades after training on a new task. This issue arises due to the model's tendency to overwrite previously acquired knowledge with new information. We present a novel approach to address this challenge, f...

---

### 29. [SemAgent: Semantic-Driven Agentic AI Empowered Trajectory Prediction in Vehicular Networks](https://arxiv.org/abs/2512.00834)

**Authors**: Lin Zhu, Kezhi Wang, Luping Xiang, Kun Yang  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.00834v1  

#### Abstract
Efficient information exchange and reliable contextual reasoning are essential for vehicle-to-everything (V2X) networks. Conventional communication schemes often incur significant transmission overhead and latency, while existing trajectory prediction models generally lack environmental perception a...

---

### 30. [CLIP-RL: Aligning Language and Policy Representations for Task Transfer in Reinforcement Learning](https://arxiv.org/abs/2512.01616)

**Authors**: Chainesh Gautam, Raghuram Bharadwaj Diddigi  
**Category**: cs.AI  
**Published**: 2025-12-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.01616v1  

#### Abstract
Recently, there has been an increasing need to develop agents capable of solving multiple tasks within the same environment, especially when these tasks are naturally associated with language. In this work, we propose a novel approach that leverages combinations of pre-trained (language, policy) pai...

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

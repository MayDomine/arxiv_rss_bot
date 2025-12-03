# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-12-03 04:59:41 UTC
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

#### AI Summary (by kimi-k2-thinking)
# SIMPLE论文核心结论与实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
- **采样瓶颈问题**：在分布式LLM推理中，随着tensor parallelism (TP)和pipeline parallelism (PP)扩展以及GPU数据平面优化，sampling（决策平面）成为新的结构性瓶颈。采样在迭代时间中占比可达20-38%，且该比例随TP增加而上升（2→8卡增加约10%）。
- **并行化缺陷**：采样操作无法随TP扩展（vocabulary轴操作需跨rank协调），且在PP中仅执行于最后阶段，导致stage skew和pipeline bubble（22-40%）。
- **性能天花板**：采样作为串行epilogue，其延迟floor限制了per-token latency和pipeline frequency，随着GPU加速而愈发严重（Amdahl定律）。

### 提出的新方法
**SIMPLE**：一个与阶段无关、序列并行、可重叠的决策平面架构，将采样从GPU推理解耦到CPU侧服务。包含三项核心技术：

1. **序列并行采样（Sequence-parallel sampling）**：沿batch维度分片工作，消除vocabulary轴的collective通信，将采样转为独立的per-sequence任务。
2. **CPU优化算法**：采用列式（column-wise）惩罚计算和截断优先（truncation-first）过滤，实现单遍线性时间内核，避免全vocabulary扫描。
3. **推测热词表采样（Speculative Hot-Vocab Sampling, SHVS）**：基于Zipf分布构建小型热词集H，在H上快速采样并通过拒绝采样（rejection sampling）保证分布精确性，接受率达80-95%。

### 相比现有方法的优势
- **无需用户代码更改**：作为vLLM的插件式扩展，对上游透明。
- **与数据平面优化正交**：兼容FlashAttention、PagedAttention等现有优化，收益可叠加。
- **显著性能提升**：吞吐量最高提升96%，P95延迟降低20-65%。
- **资源效率高**：仅使用 modest CPU辅助，主机内存开销<1.3%。

---

## 2. 核心实验方法和设置

### 实验平台
| 平台 | GPU型号 | GPU内存 | CPU | 网络 |
| :--- | :--- | :--- | :--- | :--- |
| L40 | NVIDIA L40 | 48GB | 128× Intel Xeon Platinum 8358 | PCIe 4.0 + 200 Gbps |
| H100 | NVIDIA H100 | 80GB | 192× Intel Xeon Platinum 8468 | NVLink + 8×400 Gbps |
| B200 | NVIDIA B200 | 180GB | 256× Intel Xeon 6767P | NVLink + 8×400 Gbps |

- 每个节点8 GPU + 2TB主机内存
- CUDA版本：L40/H100使用12.6，B200使用12.8

### 模型与部署配置
- **测试模型**：QwQ-32B, Llama-3.1-70B, Qwen-2.5-72B, Qwen3-235B-A22B, DeepSeek V3, Qwen3-Coder-480B-A35B
- **并行配置**：TP4-PP1/2/4（t≤4以保留跨节点扩展效率）
- **Batch size**：默认每GPU B=32（总batch size = 256当p×t=8）

### 数据集与Workload
- 从**ShareGPT**数据集采样固定prompt集
- 禁用early stopping，启用完整生产采样控制：top-p, top-k, min-p, temperature, repetition/presence/frequency penalties

### 基线方法
- **vLLM (0.10.1)**：主流高性能推理引擎
- **SGLang (0.5.2)**：另一优化推理栈

### 评估指标
- **端到端吞吐量**（tokens/s）
- **Time-per-Output-Token (TPOT)**：P50/P95/P99延迟
- **GPU/CPU利用率**（mid-50%区间）
- **主机内存使用率**
- **Total Variation Distance (TVD)**：验证SHVS分布精确性

---

## 3. 主要实验结果和性能指标

### 端到端吞吐量提升
| 平台 | 平均提升 | 最高提升 | 典型模型表现 |
| :--- | :--- | :--- | :--- |
| **L40** | +50% | **+96%** (Qwen3-235B-A22B) | QwQ-32B: +67% |
| **H100** | +50% | **+74%** (Qwen3-235B-A22B) | Llama-3.1-70B: +67% vs SGLang |
| **B200** | +28% | **+36%** (Qwen3-Coder-480B-A35B) | DeepSeek V3: +30% |

**关键观察**：词汇量越大的模型（如Qwen3-235B/480B）收益越显著，符合Zipf驱动的成本模型。

### 延迟降低（TPOT P95）
| 平台 | 平均降低 | 最高降低 | 典型模型表现 |
| :--- | :--- | :--- | :--- |
| **L40** | -39% | **-49%** (Qwen3-235B-A22B) | QwQ-32B: -42% |
| **H100** | -55% | **-65%** (Llama-3.1-70B) | Qwen3-235B-A22B: -58% |
| **B200** | -28% | **-34%** (DeepSeek V3) | Qwen3-Coder-480B-A35B: -30% |

### 负载-延迟权衡（H100, Qwen3-235B-A22B）
- **饱和负载**：P99 TPOT从105ms降至63ms (**-40%**)，吞吐量从5326→9421 tok/s (**+77%**)
- **中等负载**（rate=64）：P99从178ms降至87ms (**-51%**），吞吐量+119%（~2.2×）
- **低负载**（rate=1）：P99从62ms降至36ms (**-42%**）

### 资源利用率
- **GPU利用率**：B200上从75%→96%（**+21%平均**），最高+28%（Qwen3-235B-A22B）
- **CPU利用率**：B200上平均增加17%，L40上增加8%（仍<31%，未饱和）
- **主机内存**：增加≤1.3%（B200: 6.8%→8.1%）

### 消融实验（QwQ-32B, L40）
| 设计变体 | 每sampler吞吐量 | 相对提升 |
| :--- | :--- | :--- |
| vLLM CPU（全V端口） | 1.3 tokens/s | 1× |
| 并行Sampling（GPU驻留） | 6.4 tokens/s | **+4.8×** |
| Offloading（CPU+列式优化） | 53 tokens/s | **+8.4×** |
| **SHVS（热词表推测）** | **300 tokens/s** | **+225×总计** |

**线程扩展性**：32线程时SHVS达393 tokens/s，128线程时降至228 tokens/s（内存控制器竞争）。

### SHVS精确性验证
- **TVD**：在H100上测试DeepSeek V3、Llama-3.1-70B、Qwen3-235B-A22B，累积TVD **<0.1%**（平均0.067%）
- **分布无偏**：拒绝采样机制保证输出分布与baseline完全一致，残差仅来自浮点精度。

### 热词表大小模型验证（QwQ-32B, L40）
- **成本模型**：CPU时间符合仿射模型 $T_{cpu}(H)=cH+c_0$（拟合残差小）
- **最优H选择**：模型预测的$H^\star$与实测吞吐量峰值位置一致，误差在可接受范围内。

---

## 4. 关键结论和发现

### 主要发现
1. **采样是持久性瓶颈**：其成本不随TP扩展，在PP中造成stage skew，且随GPU加速而占比增长（Amdahl定律效应）。
2. **解耦决策平面是关键**：将采样从GPU关键路径移除、序列并行化、与forward pass重叠，可解锁显著性能。
3. **SHVS的有效性**：利用Zipf分布特性，在保持分布精确的前提下，将决策平面工作从$O(V)$降至$O(H)$，实现数量级加速。
4. **资源效率**： modest CPU资源（<31%利用率）即可换取GPU利用率提升至96%，内存开销可忽略。

### 方法局限性
1. **CPU带宽限制**：极高线程数时，per-thread吞吐量因NUMA和内存控制器竞争而下降。
2. **热词表覆盖率敏感**：当领域漂移或解码约束严重时，$\bar{\alpha}(H)$降低，SHVS接受率下降，收益收窄。
3. **GPU计算受限场景**：当数据平面本身是compute-bound时，重叠空间有限。
4. **部署复杂性**：需管理CPU sampler进程和共享内存ring buffer，增加系统组件。

### 未来工作方向
1. **在线自适应控制**：基于QoS和运行时反馈动态调整热词表大小$H$。
2. **拓扑感知放置**：优化CPU sampler的NUMA亲和性，减少跨socket内存流量。
3. **扩展应用**：将SHVS推广到结构化/语法约束解码场景。
4. **异构资源调度**：在CPU和GPU间更细粒度地协同调度决策与数据平面任务。

---

## 总结
SIMPLE通过**架构解耦**和**算法创新**，成功将采样从分布式LLM推理的瓶颈转变为可忽略的背景任务。其核心洞察是：采样本质是**序列并行**而非词汇并行，应利用CPU的灵活性和Zipf分布的偏斜性进行加速。实验表明，该方法在多种硬件和模型上均能实现**吞吐量接近翻倍、延迟减半**的显著收益，且与现有优化正交，为未来GPU世代预留了复合扩展空间。

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

## 1. 论文的主要贡献和创新点

### 解决的问题
- **Structure Gap**：LLMs固有的probabilistic token generation与structured data formats（如JSON、XML）的deterministic requirements之间存在根本性矛盾
- 传统SFT无法有效惩罚structural violations，导致"近似正确"但无法严格解析的输出
- Constrained decoding方法（如PICARD、Outlines）虽然能保证语法正确性，但显著增加inference latency（最高达6倍）

### 提出的新方法
- **RL-Struct框架**：轻量级RL框架，通过dense、rule-based reward signal直接优化模型生成结构化输出的能力
- **Multi-dimensional Reward Function**：将结构化输出任务分解为五个层次化约束：
  - Structural Integrity（强制必需key的存在）
  - Format Compliance（markdown/JSON格式）
  - Validity（严格语法正确性，可解析）
  - Correctness（内容与ground truth对齐）
  - Length（输出长度正则化）
- **GRPO优化**：采用Gradient Regularized Policy Optimization，通过group-based relative rewards估计baseline，无需critic network

### 相比现有方法的优势
- **效率**：相比PPO减少40% VRAM占用（14.2GB vs 22.8GB），训练吞吐量提升62%（42 vs 26 samples/min）
- **无推理开销**：作为training-time intervention，不增加inference latency
- **可靠性**：在4B参数模型上实现89.7% structural accuracy，超越更大的7B/8B模型
- **样本效率**：仅需1000个样本即可达到>80% accuracy，显著优于SFT

---

## 2. 核心实验方法和设置

### 数据集
- **主任务**：Recipe Generation（AkashPS11/recipes_data_food.com），生成包含ingredients、steps、nutritional information的JSON
- **泛化验证**：
  - GSM8K-JSON：数学推理任务，要求输出结构化推理步骤和最终答案
  - ToolUse：函数调用任务，生成符合API schema的JSON参数

### 实验设置
- **Base Model**：Qwen3-4B-Instruct
- **PEFT方法**：LoRA（rank=32, alpha=32），4-bit QLoRA量化
- **训练配置**：250 steps，learning rate=5×10⁻⁶，cosine decay，group size G=6
- **硬件**：单张NVIDIA RTX 4090（24GB）

### 评估指标
- **Structural Accuracy**：整体语法正确性
- **JSON Validity**：严格可解析性（json.loads成功）
- **Format Consistency**：格式风格一致性
- **Schema Compliance**：必需key的recall
- **Content Accuracy**：语义正确性（F1-Score + GPT-4 Judge评分归一化）

### 基线方法
- **Zero-shot**：GPT-3.5-Turbo, Mistral-7B-Instruct
- **SFT**：Phi-3-mini (3.8B), LLaMA-3-8B-Instruct, Qwen3-4B
- **Constrained Decoding**：Qwen3-4B + Outlines
- **Preference Optimization**：Qwen3-4B + DPO（基于valid vs invalid JSON对）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| Method | Structural Acc. | JSON Validity | Content Acc. |
| :--- | :--- | :--- | :--- |
| GPT-3.5 (Zero-shot) | 45.5% | 82.1% | 88.0% |
| Mistral-7B (Zero-shot) | 52.3% | 83.5% | 85.0% |
| Phi-3-mini (SFT) | 74.1% | 84.8% | 81.5% |
| LLaMA-3-8B (SFT) | 78.2% | 85.4% | 86.0% |
| Qwen3-4B (SFT) | 65.4% | 72.1% | 80.0% |
| Qwen3-4B + Outlines | **99.8%** | **100.0%** | 79.5% |
| Qwen3-4B + DPO | 82.5% | 88.4% | 82.0% |
| **RL-Struct (Ours)** | **89.7%** | **92.1%** | **84.5%** |

### 核心发现
1. **模型尺寸非关键**：4B参数的RL-Struct在structural accuracy上超越8B的LLaMA-3-8B (89.7% vs 78.2%)
2. **效率-质量平衡**：Outlines达到近乎完美的语法（99.8%）但inference latency增加6倍，且content accuracy下降；RL-Struct在保持原生推理速度的同时实现高可靠性
3. **GRPO vs DPO**：GRPO的group-based dense reward比DPO的pairwise preference更有效（89.7% vs 82.5%）

### 消融实验结果
| Configuration | JSON Validity | Structural Acc. |
| :--- | :--- | :--- |
| Full Reward | **92.1%** | **89.7%** |
| w/o R_valid | 68.3% (-23.8) | 85.2% |
| w/o R_struct | 90.5% | 45.6% (-44.1) |
| w/o R_format | 88.2% | 87.1% |

- **R_valid至关重要**：移除后JSON validity骤降23.8%，证明语法约束的必要性
- **R_struct决定完整性**：移除后模型生成有效JSON但缺失必需key，structural accuracy下降44.1%

---

## 4. 关键结论和发现

### 主要发现
1. **Emergent Curriculum现象**：训练动态呈现自发的课程学习
   - **Phase 1 (0-100 steps)**：模型优先学习syntax (R_valid快速上升)
   - **Phase 2 (100-250 steps)**：语法稳定后优化semantic content (R_correct逐渐提升)
   - 无需手动设计课程，GRPO的group-based优化自然实现"先学会说，再学会说什么"

2. **Gradient Dominance效应**：通过权重设计（w_valid ≫ w_correct）实现层次化优化，早期训练由结构奖励主导，强制策略投影到有效语法流形上

3. **强泛化能力**：在OOD任务上表现稳健
   - GSM8K-JSON：85.4% structural accuracy（SFT: 58.2%, Zero-shot: 25.5%）
   - ToolUse：91.2% accuracy，证明模型内化了结构化输出原则

4. **错误模式分析**：RL-Struct几乎消除syntax errors，显著减少hallucination（虚构key）和type mismatch，失败时倾向于"可修复"状态（如缺失闭合括号）

### 方法局限性
1. **Reward Engineering成本**：依赖手动设计的reward组件，对高度dynamic schemas（每次请求变化的API）需要重新训练
2. **格式扩展性**：当前主要针对JSON，虽可适配XML/SQL/YAML，但需要格式特定的parser
3. **序列长度**：为严格满足verbose schemas可能生成稍长序列
4. **对抗鲁棒性**：对adversarial attacks（如"jailbreak"提示）仍可能被绕过，但比SFT更稳健

### 未来工作方向
1. **Hybrid Strategy**：结合RL-tuned模型与轻量级guided decoding，处理动态schema约束
2. **LLM-as-a-Judge Rewards**：用更强模型提供dense reward，替代启发式规则
3. **Schema-Aware Reward Learning**：从JSON Schema/XSD等正式定义自动生成reward函数
4. **Adaptive Reward Weighting**：基于reward variance动态调整权重，自动化课程学习
5. **多轮Agent工作流**：扩展到multi-turn agentic workflows和多样化schema类型

---

## 总结
RL-Struct通过**层次化reward设计**和**高效GRPO算法**，在**4B参数模型**上实现了**训练高效、推理无损**的结构化输出能力，超越了更大模型的SFT效果。核心洞见是RL信号作为non-differentiable regularizer，能强制模型学习形式语言的严格约束，而GRPO的group-based机制自然诱导出syntax-first的课程学习。该框架为构建可靠的LLM agents提供了轻量级解决方案，特别适合边缘计算和隐私保护场景。

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

#### AI Summary (by kimi-k2-thinking)
# SpeContext论文核心结论与实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
论文针对**长上下文推理**场景下LLM的KV缓存瓶颈，指出现有KV缓存优化方法直接应用于长上下文推理时存在三大核心挑战：
- **Challenge-1**: 逐层检索操作带来高达60%的延迟开销，因数据依赖导致同步开销随模型深度线性增长
- **Challenge-2**: 必须完全保留新生成的KV缓存，导致预处理开销在推理阶段重复累积
- **Challenge-3**: 序列长度微小增加（如从120K到128K）就引发>80%的性能退化，因静态卸载策略无法适应动态增长的序列长度

### 提出的新方法
**SpeContext**：一种基于**推测性上下文稀疏性**的算法与系统协同设计框架，核心创新包括：

1. **算法层：轻量级检索头设计**
   - 关键洞察：蒸馏语言模型(DLM)与原始LLM在信息焦点上具有高度相似性（从信息论角度通过互信息和数据处理不等式证明）
   - 基于DLM的头级注意力权重进行重要token选择，剪枝冗余操作实现>90%参数减少（从~0.58B降至~0.03B）
   - 支持MHA/GQA/MQA/MLA四种主流注意力机制

2. **系统层：异步预取数据流**
   - 利用检索头在LLM推理前完成全局KV选择，消除逐层数据依赖
   - 设计弹性加载策略：利用相邻token生成间>80%的KV选择重叠率，仅加载差异部分（约20%），大幅减少数据传输

3. **编译层：自适应内存管理系统**
   - 构建理论内存模型，考虑模型架构、硬件规格和工作负载
   - 预计算序列长度阈值，在推理过程中动态逐层卸载KV缓存到CPU，最大化GPU内存利用率

### 相比现有方法的优势
- **性能突破**：相比Huggingface最高24.89×吞吐量提升，相比SOTA FlashInfer最高2.20×提升
- **精度保持**：在LongBench和LongWriter上实现可忽略的精度损失（预算≥1K时精度超过基线）
- **动态适应**：首次解决长上下文推理中序列长度动态增长导致的性能悬崖问题
- **低 overhead**：检索头仅60MB，无需复杂预处理（如ClusterKV的聚类或ShadowKV的量化）

---

## 2. 核心实验方法和设置

### 硬件平台
| 环境 | GPU | CPU | 内存 |
|------|-----|-----|------|
| **云端** | NVIDIA A100-80GB | Intel Xeon Platinum 8358 | 1008GB DRAM |
| **边缘端** | NVIDIA RTX 4060 Laptop (8GB) | Intel i7-13650HX | 24GB DRAM |

### 评估模型
- **云端**：Llama3.1-8B, DeepSeek-R1-Distill-Llama-8B, Qwen3-8B
- **边缘端**：Reasoning-Llama-3.2-1B（因内存限制）

### 数据集与任务
- **长上下文输入场景**：LongBench的4个任务
  - 2WiKiMQA, TriviaQA, HotpotQA, Passage count
- **长上下文推理场景**：LongWriter（评估生成质量）
  - 使用GPT-4o从6个维度评分：相关性、准确性、连贯性、清晰度、广度与深度、阅读体验

### 基线方法
- **全注意力**：Huggingface (Eager), FlashAttention, FlashInfer
- **稀疏注意力**：Quest, ClusterKV, ShadowKV

### 评估指标
- **性能**：吞吐量(tokens/s)、加速比
- **精度**：任务准确率、GPT-4o评分
- **内存**：GPU内存占用、KV缓存预算
- **开销**：检索延迟、数据传输量

---

## 3. 主要实验结果和性能指标

### 云端多请求吞吐量（表3）
| 模型 | 配置[In, Out] | Huggingface | FlashInfer | ShadowKV | **SpeContext** |
|------|---------------|-------------|------------|----------|----------------|
| DeepSeek-Distill | [2k, 32k] | 27.74 (1.00×) | 314.25 (11.32×) | 240.47 (8.67×) | **690.59 (24.89×)** |
| Qwen3-8B | [2k, 32k] | 19.28 (1.00×) | 254.92 (13.22×) | 424.92 (22.03×) | **424.92 (22.03×)** |
| Llama3.1-8B | [2k, 16k] | 45.57 (1.00×) | 490.04 (10.75×) | 366.74 (8.05×) | **824.22 (18.09×)** |

**关键数据**：
- **最高24.89×** 相比Huggingface
- **最高2.20×** 相比FlashInfer（DeepSeek-Distill [2k,32k]）
- 支持**32个并发请求**，而Huggingface仅支持4个

### 边缘端单请求加速（图10b）
| 模型 | 配置 | Huggingface | FlashAttention | ShadowKV | **SpeContext** |
|------|------|-------------|----------------|----------|----------------|
| Reasoning-Llama-3.2-1B | [2k, 16k] | 1.00× | 2.71× | 8.67× | **10.06×** |

- **10.06×** 相比Huggingface
- **1.17×** 相比ShadowKV

### 精度评估结果
**长上下文输入（图8）**：
- KV预算=512时，精度略低于ClusterKV
- **KV预算≥1K时，SpeContext超越所有基线**，接近全注意力精度

**长上下文推理（图9 & 表4）**：
- LongWriter上平均得分：**2.86-2.95**（预算2K-4K），接近全注意力（2.84-3.55）
- 在相关性、准确性等维度上保持竞争力
- **关键发现**：Quest/ClusterKV/ShadowKV因仅预处理输入提示（~100 tokens），不同预算下生成结果相同，无法适应长推理场景

### 消融实验结果（图11）
| 组件 | 加速比贡献 | 关键作用 |
|------|------------|----------|
| **C1: 轻量级检索头** | 基础加速 | 实现稀疏注意力，支持更多并行请求 |
| **C2: 异步预取+弹性加载** | **12.07×→18.09×** | 消除I/O瓶颈，减少数据传输 |
| **C3: 自适应内存管理** | **18.09×→24.89×** | 最大化GPU利用率，避免性能悬崖 |

---

## 4. 关键结论和发现

### 主要发现
1. **DLM作为检索器的有效性**：从信息论角度证明DLM与LLM在上下文信息焦点上的相似性，为使用轻量级模型指导大模型推理提供理论支撑
2. **全局选择优于逐层选择**：SpeContext在推理前一次性选择全局重要KV，避免逐层检索的60%延迟开销，精度反而更高
3. **弹性加载的巨大价值**：相邻生成间>80%的KV重叠率使数据传输量减少80%，这是边缘端实现10×加速的关键
4. **动态内存管理的必要性**：静态策略在长推理场景下会导致性能悬崖，自适应逐层卸载可维持稳定性能

### 方法局限性
- **小预算精度下降**：KV预算<1K时精度略低于SOTA（但仍优于Quest/ShadowKV）
- **模型依赖性**：需要为每个LLM训练对应的DLM（依赖EAGLE-3，需24小时RTX 3090训练）
- **检索头开销**：虽然大幅剪枝，但仍增加~20%计算（通过异步重叠可基本隐藏）

### 未来工作方向
1. **信息论视角扩展**：将DLM-LLM对齐的方法论推广到模型压缩、早期退出等场景
2. **检索头优化**：探索更极端的剪枝或量化，进一步降低检索开销
3. **多模态支持**：将SpeContext扩展到视觉-语言模型等长上下文多模态场景
4. **自适应预算**：根据输入内容动态调整KV预算，实现精度-效率的细粒度权衡

---

**总结**：SpeContext通过**算法-系统-编译**三层协同设计，首次在长上下文推理场景中同时解决了性能、精度和动态适应性问题，将Pareto前沿推向新高度，为资源受限环境下的LLM部署提供了实用解决方案。

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

#### AI Summary (by kimi-k2-thinking)
# Efficient and Programmable Exploration of Synthesizable Chemical Space 论文总结

## 1. 主要贡献和创新点

### 解决的问题
- **可合成性瓶颈**：现有生成模型提出的分子难以实验合成，合成成功率低
- **化学空间覆盖不足**：当前SOTA方法对Enamine REAL空间的重建率仅约70%，限制实用价值
- **两阶段不一致性**：传统"生成-投影"范式导致分子图与可合成类似物在结构和功能上存在差异
- **效率低下**：推理速度慢，采样效率低，难以满足大规模应用需求

### 提出的新方法：PrexSyn框架
- **统一架构**：基于decoder-only transformer的属性条件生成模型，直接在后缀表示法（postfix notation）的合成路径空间生成分子
- **高通量数据引擎**：C++实现的多线程数据生成管道，实现**实时、十亿级**训练数据流生成（>6,000 samples/秒）
- **可编程查询**：支持使用逻辑运算符（AND, NOT, OR）组合多个分子属性，形成复合查询条件
- **查询空间优化**：在**数值化属性空间**而非离散分子空间进行迭代优化，显著提升黑盒优化效率

### 相比现有方法的优势
| 维度 | PrexSyn | 现有SOTA（如ReaSyn, SynFormer） |
|------|---------|-------------------------------|
| **覆盖率** | **94.06%**重建率 | ~75%重建率 |
| **推理速度** | **0.26秒/目标**（256样本） | ~20秒/目标（>60倍加速） |
| **采样效率** | GuacaMol 6/8任务最优 | 普遍较低 |
| **可编程性** | 支持复合逻辑查询 | 仅支持单属性或图条件 |
| **通用性** | 通用分子优化工具 | 任务特定模型为主 |

---

## 2. 核心实验方法和设置

### 数据集
- **Building blocks**：Enamine US库存building blocks（约48万个，2023年10月）
- **反应模板**：115个来自Gao et al. (2025)的模板
- **训练数据**：**13亿**合成路径-属性对（实时生成，非静态数据集）
- **测试集**：
  - **Enamine testset**：1,000个Enamine REAL分子（评估空间覆盖）
  - **ChEMBL testset**：1,000个ChEMBL分子（评估泛化能力）

### 模型配置
- **架构**：12层decoder-only transformer
  - 模型维度：1,024
  - 前馈维度：2,048
  - 注意力头数：16
  - 总参数量：**5.89亿**（含building block嵌入矩阵）
- **训练设置**：
  - 硬件：2× NVIDIA H100 GPU + 32 CPU核心
  - 时长：48小时（640,000步）
  - 批量大小：2,048
  - 优化器：Adam（lr=3×10⁻⁴，cosine退火）
  - 精度：32-bit全精度

### 训练属性
- **结构指纹**：ECFP4, FCFP4
- **子结构**：BRICS分解片段
- **物化描述符**：分子量、CLogP、TPSA、可旋转键数、氢键供体/受体等

### 评估指标
- **重建率**：Tanimoto相似度=1的分子比例
- **相似度**：ECFP4/Morgan指纹Tanimoto相似度
- **推理时间**：单目标平均耗时
- **AUC-Top10**：GuacaMol标准指标（10,000 oracle calls预算）
- **多样性**：1 - 平均成对Tanimoto相似度
- **对接得分**：AutoDock Vina/GPU预测的负结合能

---

## 3. 主要实验结果和性能指标

### 3.1 化学空间投影任务

| 方法 | Enamine重建率 | Enamine相似度 | ChEMBL重建率 | ChEMBL相似度 | 推理时间 |
|------|---------------|---------------|--------------|--------------|----------|
| SynFormer | 66.10% | 0.9137 | 20.67% | 0.6737 | 3.45s |
| ReaSyn | 74.93% | 0.9403 | 22.07% | 0.6740 | 19.71s |
| **PrexSyn (256样本)** | **94.06%** | **0.9859** | **28.32%** | **0.7533** | **0.26s** |

**关键发现**：
- **绝对提升**：相比ReaSyn，重建率提高**19.13个百分点**
- **速度优势**：比最高精度基线快**60-75倍**
- **数据规模效应**：重建率随训练数据量单调递增，13亿样本达到饱和

### 3.2 GuacaMol黑盒优化基准

| 方法 | 可合成性 | Amlodipine | Fexofenadine | Osimertinib | Perindopril | Ranolazine | Sitagliptin | Zaleplon | Celecoxib | 平均 |
|------|----------|------------|--------------|-------------|-------------|------------|-------------|----------|-----------|------|
| REINVENT | ✗ | 0.635 | 0.784 | 0.837 | 0.537 | 0.760 | 0.021 | 0.358 | 0.713 | 0.581 |
| MolGA | ✗ | 0.688 | 0.825 | 0.844 | 0.547 | 0.804 | 0.582 | 0.519 | 0.567 | **0.672** |
| SynFormer | ✓ | 0.696 | 0.786 | 0.816 | 0.530 | 0.751 | 0.338 | 0.478 | 0.559 | 0.619 |
| ReaSyn | ✓ | 0.678 | 0.788 | 0.820 | 0.560 | 0.742 | 0.342 | 0.492 | **0.754** | 0.647 |
| **PrexSyn** | ✓ | **0.781** | **0.837** | **0.855** | **0.714** | **0.807** | **0.471** | **0.504** | 0.801 | **0.721** |

**结果**：PrexSyn在**6/8任务**中取得最高AUC-Top10，平均得分超越所有基线（包括不可合成方法）

### 3.3 复合属性查询生成

| 任务 | 描述 | 查询类型 | Top 5%得分 | Top 10%得分 | 多样性 |
|------|------|----------|------------|-------------|--------|
| Task 1 | Lipinski's Rule of Five | 多属性AND | **1.0000** | **1.0000** | 0.8902 |
| Task 2 | Cobimetinib类似物优化 | 指纹+属性 | 0.8975 | 0.8848 | 0.6017 |
| Task 3 | Osimertinib MPO优化 | 指纹+NOT+属性 | 0.8314 | 0.8068 | 0.7499 |

**亮点**：
- Task 1中**top 5%分子完美满足**所有Lipinski规则
- 在保持高得分的同时，多样性保持在0.6-0.9区间

### 3.4 对接优化任务

**sEH任务**：
- **sEH得分**：1.01 ± 0.00（vs SynFlowNet最佳0.94）
- **QED**：0.80 ± 0.01（vs 0.68）
- **SA**：2.23 ± 0.04（vs 2.67）

**Mpro2任务**：
- 在2,000 oracle calls预算下，生成分子**优于COVID Moonshot基线抑制剂**
- 结合模式可视化显示生成的分子能正确占据关键子口袋

---

## 4. 关键结论和发现

### 主要发现
1. **规模效应显著**：十亿级训练数据使模型能够**近乎完美**地重建可合成化学空间，证明数据规模比复杂架构更重要
2. **属性条件优于图条件**：直接以分子属性为条件生成，避免了图-路径不一致问题，提升采样效率
3. **查询空间优化的优势**：数值化属性空间比离散分子空间或合成树空间**更平滑、更易优化**，即使面对黑盒oracle也表现优异
4. **复合查询的有效性**：逻辑组合能**精确控制**生成分子的多维度特性，在scaffold hopping等任务中显著提升优化效率（AUC提升0.154）

### 方法局限性
1. **属性独立性假设**：复合查询基于属性条件独立的简化假设，对强相关属性（如分子量与重原子数）或矛盾查询可能失效
2. **训练属性覆盖范围**：无法直接处理训练时未见过的属性（如特定靶点活性），需依赖查询空间优化间接逼近
3. **Building block规模**：当前基于分类器的tokenization在百万级building block下可行，但未来超大规模库需采用sampled softmax等技术
4. **反应模板固定性**：依赖预定义的115个反应模板，对模板外反应无法处理

### 未来工作方向
- **扩展化学空间**：整合更多vendor的building blocks和反应模板
- **增强查询语言**：支持更复杂的逻辑表达式和属性关系建模
- **优化算法改进**：探索贝叶斯优化、强化学习等更高效的查询空间搜索策略
- **实验验证**：与自动化合成平台集成，验证生成分子的实际可合成性和生物活性
- **3D结构整合**：将分子构象信息融入生成过程，提升结构基础药物设计能力

---

**代码与资源**：https://github.com/luost26/PrexSyn

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

#### AI Summary (by kimi-k2-thinking)
# 论文总结：基于 rLoRA 微调的 Qwen3-8B 金融文本分类

## 1. 主要贡献和创新点

### 解决的问题
论文针对金融领域中两类关键文本分类任务——**金融情感分析**和**金融新闻主题分类**，探索了如何高效利用大语言模型（LLM）提升分类准确性和训练效率，以满足量化交易系统对实时性和准确性的要求。

### 提出的新方法
1. **Noisy Embedding Instruction Finetuning (NEFTune)**：在监督微调过程中向嵌入层注入受控噪声（α=0.3），增强模型鲁棒性，减少过拟合
2. **Rank-stabilized Low-Rank Adaptation (rLoRA)**：采用改进的 LoRA 技术，通过 √r 缩放因子稳定高秩参数更新，支持更高秩（rank=8）的安全训练
3. **FlashAttention 集成**：结合内存高效的注意力机制，降低 GPU 内存占用并加速训练
4. **双模式推理控制**：利用 Qwen3 的 `\no_think` 标签关闭思考模式，实现低延迟的确定性分类

### 相比现有方法的优势
- **性能更优**：在两项任务上均超越传统 Transformer 和同等规模 LLM 基线
- **训练高效**：仅需 3 个 epoch 即可收敛，而非 LLM 方法需 10+ epoch
- **内存友好**：参数高效微调仅训练低秩矩阵，冻结原模型权重
- **实时适用**：结合 FlashAttention 和 GQA 架构，支持长文本（32K+ tokens）快速推理

---

## 2. 核心实验方法和设置

### 数据集
| 数据集 | 样本量 | 类别分布 | 任务类型 |
|--------|--------|----------|----------|
| **金融情感分类数据集** | 4,845 条 | 中性 2,879 / 正面 1,362 / 负面 604 | 三分类 |
| **Twitter 金融新闻数据集** | 训练集 16,990 / 测试集 4,117 | 20 个新闻主题类别 | 多分类 |

### 实验设置
- **骨干模型**：Qwen3-8B（8.2B 参数，36 层 Transformer，GQA 架构）
- **微调方式**：Supervised instruction-tuning with rLoRA
- **关键超参数**：
  - Batch size: 3（梯度累积 4 步，有效 batch size=12）
  - Learning rate: 5e-5（Adam 优化器）
  - Max token length: 360
  - LoRA rank: 8，dropout: 0.1
  - NEFTune alpha: 0.3
  - Epochs: 3
- **评估指标**：Accuracy（准确率）

### 基线方法对比
- **非 LLM 模型**：RoBERTa、BERT
- **开源 LLM**：Baichuan2-7B、LLaMA-7B、LLaMA2-7B

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 金融情感分类 (ACC) | 金融主题分类 (ACC) | 相对提升 |
|------|-------------------|-------------------|----------|
| **RoBERTa** | 0.7928 | 0.8612 | - |
| **BERT** | 0.7854 | 0.8523 | - |
| **Baichuan2-7B** | 0.8165 | 0.8784 | +2.9% / +2.0% |
| **LLaMA-7B** | 0.8297 | 0.8801 | +4.7% / +2.2% |
| **LLaMA2-7B** | 0.8322 | 0.8877 | +5.0% / +3.1% |
| **Qwen3-8B (本文)** | **0.8415** | **0.9315** | **+6.1% / +8.2%** |

### 训练效率分析
- **收敛速度**：Qwen3-8B 在 **3 个 epoch** 内快速收敛，损失曲线平滑稳定
- **迭代次数**：显著少于非 LLM 方法（通常需 10+ epoch）
- **内存优化**：FlashAttention 和 rLoRA 使长序列训练可行

### 消融实验
论文未明确报告完整的消融实验，但提及了关键组件的协同作用：
- **rLoRA** 相比标准 LoRA 支持更高秩训练，提升稳定性
- **NEFTune** 噪声正则化减少过拟合，增强指令遵循能力
- **GQA 架构** 降低 KV-cache 内存，提升推理效率

---

## 4. 关键结论和发现

### 主要发现
1. **Qwen3-8B 在金融 NLP 任务上表现卓越**：在情感分类（84.15%）和主题分类（93.15%）上均达到 SOTA，显著优于同等规模开源模型
2. **指令微调 + 参数高效训练是有效范式**：结合 NEFTune、rLoRA 和 FlashAttention 的框架在保持低计算成本的同时实现高性能
3. **架构优势转化为实际效益**：Qwen3 的 GQA、RoPE 和长上下文支持（32K+ tokens）使其特别适合处理长金融文档
4. **实时应用可行性**：通过 `\no_think` 模式控制，可在批量分类任务中实现低延迟、高吞吐

### 方法局限性
- **数据集规模**：使用的金融数据集相对较小（<5K 样本），未充分验证大规模数据下的扩展性
- **模型规模**：仅在 8B 参数模型上验证，未探索更大 Qwen3 版本（如 14B、32B）的潜力
- **任务覆盖**：仅测试分类任务，未涉及生成、推理等更复杂金融场景
- **消融分析**：缺乏对各组件（NEFTune vs rLoRA vs FlashAttention）贡献度的定量分析

### 未来工作方向
论文未明确提及，但基于内容可推断：
- 在更大规模金融数据集和更多任务类型上验证框架有效性
- 探索 Qwen3 更大参数版本（如 Qwen3-14B/32B）的性能边界
- 进行详细的消融研究，量化各优化技术的独立贡献
- 将该框架应用于实时量化交易系统，评估实际部署效果

---

**核心总结**：该工作证明了 Qwen3-8B 结合 rLoRA、NEFTune 和 FlashAttention 的微调框架，在金融文本分类任务上实现了**准确性、效率和内存占用**的最佳平衡，为金融 NLP 应用提供了一个可扩展且经济的解决方案。

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
# Domain-Decomposed Graph Neural Network Surrogate Modeling for Ice Sheets 核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
- **计算瓶颈**：冰盖模拟需要求解复杂的非线性PDE系统（Blatter-Pattyn近似），单次高保真模拟耗时数小时至数天，不确定性量化(UQ)需要数千次评估，计算不可行
- **过平滑问题**：标准message-passing GNN在深层网络中会出现特征过度平滑，丢失冰厚、基底摩擦和速度场等关键物理量的尖锐梯度信息
- **可扩展性挑战**：现有方法如DeepONet需要为每个新网格或几何重新训练，缺乏灵活性

### 提出的新方法
1. **物理启发的Bracket-based GNN架构**
   - 采用Hamiltonian bracket-based GNN[14]，将信息传播重构为能量守恒动力学系统
   - 通过skew-adjoint算子保证全局不变量，从根本上避免过平滑
   - 集成graph attention机制，内积矩阵A₀,A₁动态调整节点/边权重

2. **域分解(Domain Decomposition, DD)框架**
   - 将非结构化网格划分为物理一致的子域（subgraphs）
   - 在各子域上并行训练独立的GNN代理模型
   - 推理时聚合预测结果，支持非重叠分区

3. **迁移学习策略**
   - 在子域（如冰川终端区）预训练，将学习到的attention机制、编码器/解码器权重迁移到全模型
   - 在数据有限场景下（5-10个摩擦场样本）实现"warm start"加速收敛

### 相比现有方法的优势
- **精度提升**：在Humboldt冰川测试中，终端区域预测误差降低近一个数量级
- **训练效率**：DD+迁移学习组合策略收敛速度比冷启动快3-5倍
- **内存可扩展**：子域训练内存需求与全局问题规模解耦
- **物理一致性**：Bracket架构隐式满足守恒律，无需显式物理正则化项
- **网格灵活性**：直接在Delaunay三角剖分上操作，无需重网格或插值

---

## 2. 核心实验方法和设置

### 数据集
- **研究对象**：Greenland Humboldt冰川（约200万平方公里）
- **模拟器**：MPAS-Albany Land Ice (MALI)模型，采用Mono-Layer Higher-Order (MOLHO)近似
- **训练数据生成**：
  - 从基底摩擦场μ(x,y)的后验分布采样（Laplace近似，PDE-based先验）
  - 模拟时段：2007-2100年，93个年度快照
  - 20个模拟用于验证，20个用于测试
- **图结构**：18,544节点，54,962边（Delaunay三角剖分）
- **节点特征**：7维向量（冰厚度、床地形、基底摩擦、接地/漂浮冰布尔标志）
- **输出**：速度场u,v分量（2维）

### 实验设置
- **训练配置**：
  - 基准：25个基底摩擦场 × 40个时间步 = 1,000样本
  - 数据受限场景：5场和10场子集
  - 批量梯度下降，Adam优化器，学习率1e-3
  - 损失函数：MSE（速度场预测误差）
- **评估指标**：
  - 相对测试误差（Relative test error）
  - 点wise速度误差分布
  - 接地线冰通量分布（用于UQ评估）
- **基线方法**：
  - **Cold start**：从头训练的全局GNN模型
  - **Warm start**：在子域预训练后全局微调
  - **DD + Warm start**：子域预训练→各子域微调→聚合

### 域分解实现
- **分区算法**：谱聚类+尺寸惩罚k-means
  - 边权重：W_uv = exp(-||Δx||/σ_x²) × exp(-||Δf||/σ_f²) × exp(-||Δy||/σ_y²)
  - 融合空间距离、特征相似性（厚度/摩擦/地形）和目标相似性（速度）
- **分区数**：k=3（终端区、内部区、过渡区）
- **分区类型**：非重叠（non-overlapping），节点唯一归属

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 训练策略 | 收敛速度 | 终端区误差 | 相对测试误差（25场） |
|----------|----------|------------|---------------------|
| Cold start | 基准（~10⁴ epochs） | 高（>100 m/yr） | ~10⁻¹ |
| Warm start | 加速2-3倍 | 中（~50 m/yr） | ~5×10⁻² |
| DD + Warm start | 加速3-5倍 | **极低（<20 m/yr）** | **~2×10⁻²** |

### 与基线对比结果
1. **Cold start基准**：
   - 能捕捉大尺度速度场结构
   - 误差集中区：接地线（grounding line）和快速流动终端区
   - 最大绝对误差出现在速度梯度最陡区域

2. **Warm start改进**：
   - 在5场数据场景下，误差比cold start降低**40-60%**
   - 预训练在终端区（高复杂度）效果优于内部区
   - 收敛曲线显示：相同epoch下误差持续低于cold start（Figure 6）

3. **DD + Warm start最优**：
   - **误差近零**：终端区预测误差几乎消除（Figure 5底行）
   - **并行效率**：子域训练可完全并行，单个子域内存占用减少~60%
   - **鲁棒性**：非重叠分区无边界不连续问题，无需overlap或粗网格校正

### 消融实验结果
- **预训练数据源影响**：终端区预训练 → 内部区预训练，最终误差相差约15%
- **分区数量**：k=3在计算效率和精度间取得最佳平衡（k>5未测试）
- **Attention head数**：2→4头无明显提升，表明模型对超参数不敏感
- **编码器/解码器宽度**：32→16维度使误差翻倍，显示对表示能力敏感

---

## 4. 关键结论和发现

### 主要发现
1. **域分解的双重优势**：
   - **计算层面**：降低内存、支持并行、加速训练
   - **学习层面**：本地化高频函数，缓解神经网络的频谱偏差（spectral bias），使高变异区域（终端区）能被有效学习

2. **迁移学习的样本效率**：
   - 在仅5个基底摩擦场样本下，warm start达到cold start 25场样本的精度水平
   - 预训练模型提供强初始化，减少新任务所需数据量达**80%**

3. **非重叠分区的有效性**：
   - 与传统PDE求解器不同，GNN代理无需overlap传递nullspace信息
   - 数据驱动+MSE目标使模型在边界处自然平滑，无接口误差累积

4. **UQ适配能力**：
   - 基础模型能准确预测均值响应，但低估不确定性分布宽度（Figure 8左）
   - 在UQ专用数据集（单快照/样本）上微调后，**完整恢复接地线冰通量的真实分布**（Figure 8右）
   - 展现"基础模型"潜力：通用预训练→轻量微调适配新科学目标

### 方法局限性
- **分区策略启发式**：当前谱聚类依赖手工调参（σ_x, σ_f, σ_y），最优分区标准未理论化
- **UQ需专门微调**：基础模型不能直接用于不确定性传播，需设计强调输入变异性的微调数据集
- **几何固定**：当前实验未测试动态演化网格或不同冰川的零样本迁移
- **损失函数单一**：仅MSE可能不足以保证长期稳定性，未测试物理守恒硬约束

### 未来工作方向
1. **基础模型开发**：构建跨冰川、跨分辨率的预训练模型，实现"即插即用"式UQ
2. **自适应分区**：研究基于物理特征和误差指标的自适应域分解策略
3. **多保真度融合**：结合低/高保真模拟数据，进一步提升样本效率
4. **长期稳定性**：探索保结构积分器（如symplectic integrator）在GNN-ODE中的应用
5. **气候决策集成**：将代理模型耦合到地球系统模型，支持实时风险评估和政策制定

---

**核心贡献总结**：本工作首次系统性地将域分解与迁移学习结合用于大规模GNN代理建模，在冰盖模拟中实现了**精度、效率、可扩展性**的三重突破，为科学机器学习的基础模型范式提供了可行路径。

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
动态随机游走（Dynamic Random Walks）的转移概率依赖于运行时状态（如游走历史），无法像静态随机游走那样预先计算和缓存概率分布。这导致现有基于预计算策略的CPU/GPU优化框架失效，迫使开发者编写手工调优的内核，且无法适应工作负载多样性。

### 核心创新方法
**FlexiWalker** 提出了首个支持高效动态随机游走的GPU框架，包含三个关键创新：

1. **高性能采样内核**：
   - **eRVS**（增强水库采样）：通过统计等效方法消除前缀和计算，采用"jump technique"减少随机数生成，将内存访问降低约50%
   - **eRJS**（增强拒绝采样）：用理论边界估计替代全局max归约，避免全量权重访问，保持采样正确性的同时消除冗余计算

2. **运行时自适应选择**：轻量级一阶成本模型，**按节点、按步骤**动态选择eRVS或eRJS中更快的方法，适应高度偏斜和时变的边权重分布

3. **编译时自动特化**：Flexi-Compiler基于LLVM/Clang静态分析用户提供的游走逻辑，自动生成优化的参数和代码块，用户只需编写轻量级工作负载特定函数

### 相比现有方法的优势
- **性能突破**：在真实图数据集上，相比最佳CPU/GPU基线分别实现**73.44×**和**5.91×**的几何平均加速
- **通用性**：成功执行了先前系统无法高效支持的工作负载（如加权Node2Vec、Second-Order PageRank）
- **自适应能力**：运行时选择机制使性能对权重分布变化鲁棒，在幂律分布（α=1.0）下仍保持稳定性能

---

## 2. 核心实验方法和设置

### 数据集
使用10个大规模真实世界图（表1），涵盖社交网络、引文网络和网页图：
- **小规模**：com-youtube (1.1M顶点, 6M边), cit-patents (3.8M, 33M)
- **中规模**：Livejournal (4.8M, 86M), Orkut (3.1M, 234M), EU-2015 (11M, 522M)
- **大规模**：Arabic-2005 (23M, 1.1B), UK-2005 (39M, 1.6B), Twitter (42M, 2.4B), SK-2005 (51M, 3.6B), Friendster (66M, 3.6B)

### 实验设置
- **硬件**：AMD EPYC 9124P (16核32线程), 512GB DDR5 ECC内存, 最多4×NVIDIA A6000 GPU (48GB显存)
- **软件**：Ubuntu, CUDA 12.1.1, cuRAND 10.3.2.106
- **游走参数**：步长80（MetaPath为5），为每个顶点生成查询，Node2Vec中a=2.0, b=0.5，2nd PR中γ=0.2

### 基线方法
- **CPU基线**：SOWalker（最新out-of-core框架）、ThunderRW（内存内引擎）
- **GPU基线**：C-SAW（逆变换采样）、NextDoor（拒绝采样）、Skywalker（alias采样）、FlowWalker（水库采样，SOTA动态游走框架）

---

## 3. 主要实验结果和性能指标

### 整体性能对比（均匀权重分布）
| 对比对象 | 几何平均加速比 | 最大加速比 | 关键发现 |
|---------|---------------|-----------|---------|
| **最佳CPU基线** | **73.44×** | 4246.71× | 在Friendster上实现最大加速 |
| **最佳GPU基线** | **5.91×** | 1040.54× | 在加权Node2Vec上优势更显著 |
| **NextDoor** | 26.60× | - | 幂律分布下NextDoor因OOM失败 |
| **FlowWalker** | 4.37× | - | FlexiWalker更鲁棒，不受α值影响 |

### 消融实验结果

**内核优化效果**（图12）：
- **eRVS vs FlowWalker**：EXP优化（减少内存访问）带来**1.30-1.60×**加速，JUMP优化（减少随机数生成）额外提升，总计**1.44-1.82×**（均匀分布）和**1.47-1.81×**（偏斜分布）
- **eRJS vs NextDoor**：边界估计优化在均匀分布下实现**54.49-1698.35×**加速，在高度偏斜分布（α=1）下仍保持**最高7.27×**加速

**运行时组件效果**（图11）：
- 相比**仅eRJS**：最高**3.37×**加速，防止在偏斜分布下性能崩溃
- 相比**仅eRVS**：最高**421.56×**加速，避免在高均匀分布下的次优选择
- 在SK数据集上，运行时选择策略成功避免了eRJS在α=1时的严重 slowdown

### 其他关键指标
- **多GPU扩展性**：4 GPU达到**3.23×**几何平均加速，接近线性扩展
- **能量效率**：相比KnightKing（最佳CPU基线）**最高10.15×**能效提升，比FlowWalker节省**1.18×**功耗
- **开销分析**： profiling + 预处理时间仅占总运行时间的**0.46%-3.98%**

---

## 4. 关键结论和发现

### 主要发现
1. **采样方法适用性**：在GPU大规模并行环境下，**拒绝采样（RJS）和水库采样（RVS）** 比alias采样和逆变换采样更适合动态随机游走，因其无需重复构建辅助数据结构

2. **运行时自适应的必要性**：边权重分布的偏斜度和动态变化（CV值高达200+）显著影响采样性能，静态选择策略会导致**高达2个数量级**的性能波动

3. **编译时分析的价值**：自动边界估计和代码生成使eRJS避免手工实现，同时保持通用性，成功支持Node2Vec、MetaPath、2nd PR等5种工作负载

### 方法局限性
- **Flexi-Compiler约束**：无法处理运行时图拓扑更新（如动态边权重）和复杂控制流（递归调用、数据依赖的循环退出），此时安全回退到eRVS-only模式
- **内存需求**：在最大数据集（SK-2005）上，NextDoor因内部排序导致OOM，而FlexiWalker通过优化内存访问模式避免此问题

### 未来工作方向
1. **分布式扩展**：利用图划分器[19]和跨GPU通信库（NCCL/NVSHMEM）支持万亿边规模的图
2. **动态图支持**：增加模块以更新预处理值和图拓扑，利用Flexi-Runtime的每步选择能力适应图变化
3. **低精度优化**：支持INT8等低精度边权重，初步实验显示相比FlowWalker仍有**27.59×**加速
4. **更广泛工作负载**：扩展对更复杂游走逻辑的支持，提升Flexi-Compiler的代码分析能力

---

**开源代码**：https://github.com/AIS-SNU/FlexiWalker

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

#### AI Summary (by kimi-k2-thinking)
# Efficient Training of Diffusion Mixture-of-Experts Models: A Practical Recipe

## 1. 论文的主要贡献和创新点

### 解决了什么问题
论文指出当前Diffusion MoE研究存在**过度关注路由机制设计**而**忽视底层架构配置空间**的问题。尽管MoE在LLM中取得了巨大成功，但在扩散模型中的应用效果不佳，先前工作将此归因于文本token与视觉token的本质差异，因而专注于设计复杂的路由策略。作者通过系统性研究发现，**性能差距的核心原因在于架构设计空间未被充分探索**，而非路由机制不足。

### 提出了什么新方法或新思路
论文提出了一套**以架构为中心的实用设计范式**，借鉴LLM中验证有效的MoE设计原则，而非创新复杂路由机制。核心创新包括：
- **DeepSeek风格的MoE模块**：采用细粒度专家+隔离共享专家的结构
- **缩小的MLP中间层宽度**：将缩放因子降至4.0以下，显著减少激活参数
- **2D RoPE位置编码**：为注意力层引入显式的二维空间位置信息
- **参数效率优化**：通过增加专家数量（如48个）同时缩小中间维度，实现性能与效率的双重提升

### 相比现有方法的优势
- **性能更优**：在相等或更少的激活参数下，显著超越DiffMoE、ProMoE等强基线
- **训练效率更高**：收敛速度更快，训练损失更低，3B模型仅需700K步达到SOTA
- **参数效率更高**：总参数量减少30-50%的同时保持或提升性能
- **简单实用**：无需复杂路由逻辑，架构改进即可释放MoE潜力

---

## 2. 核心实验方法和设置

### 数据集与任务
- **数据集**：ImageNet 256×256（1,281,167张训练图像）
- **任务**：类别条件图像生成（class-conditional generation）

### 模型框架
- **Latent diffusion**：基于DiT架构的DSMoE系列（S/B/L/3B规模）
- **Pixel-space diffusion**：基于JiT框架的JiTMoE系列（B/L规模）
- **训练范式**：统一采用Rectified Flow / Flow Matching

### 评估指标
- **FID50K**（Fréchet Inception Distance，越低越好）
- **IS**（Inception Score，越高越好）
- 生成50,000张图像进行评估
- CFG（Classifier-Free Guidance）scale分别测试1.0和1.5

### 基线方法
- **DiffMoE**：当前SOTA的扩散MoE模型
- **ProMoE**：基于复杂路由的MoE方法
- **Dense-DiT**：标准密集Transformer基线
- **JiT**：像素空间扩散SOTA模型

### 模型配置
采用"共享专家+路由专家"结构（S1E16A2表示1个共享专家、16个路由专家、每token激活2个）：
- **DSMoE-S/B/L/3B**：分别对应33M/132M/465M/965M激活参数
- **JiTMoE-B/L**：133M/465M激活参数

---

## 3. 主要实验结果和性能指标

### 与DiffMoE对比（Latent Diffusion, 700K步）
| 模型 | 激活参数 | 总参数 | FID↓ (CFG=1.0) | FID↓ (CFG=1.5) | IS↑ (CFG=1.5) |
|------|----------|--------|----------------|----------------|---------------|
| DiffMoE-B-E16 | 130M | 555M | 20.83 | 4.87 | 183.43 |
| **DSMoE-B-E48** | **118M** | **263M** | **19.46** | **4.27** | **191.03** |
| DiffMoE-L-E16 | 458M | 1.982B | 11.16 | 2.84 | 256.57 |
| **DSMoE-L-E48** | **436M** | **1.112B** | **9.19** | **2.55** | **278.35** |
| **DSMoE-3B-E16** | **965M** | **2.958B** | **7.52** | **2.38** | **304.93** |

**关键发现**：DSMoE-3B-E16以**700K训练步数**达到FID 2.38，匹配DiffMoE的2.30（需**7000K步**），训练成本降低**10倍**。

### 与Dense-DiT和ProMoE对比（500K步）
- **DSMoE-B-E48** vs Dense-DiT-B：FID从9.02降至5.14（↓43%），IS从131.13提升至172.80（↑32%）
- **DSMoE-L-E48** vs ProMoE-L：FID从2.79降至2.72，总参数减少44%（1.112B vs 1.063B）

### 消融实验结果

#### 位置编码的影响（DSMoE-S-E16）
| PE方法 | FID↓ (CFG=1.0) | FID↓ (CFG=1.5) | IS↑ (CFG=1.5) |
|--------|----------------|----------------|---------------|
| APE | 45.13 | 18.10 | 82.37 |
| 1D RoPE | 44.75 | 18.12 | 83.13 |
| **2D RoPE** | **39.84** | **14.53** | **97.55** |

**结论**：2D RoPE带来**FID绝对降低3.57**，IS提升15.18，训练收敛更快。

#### 共享专家的作用
- **移除共享专家**（S0A3）：收敛速度明显变慢，训练损失更高
- **作用**：提供跨token/time步的一致表示，降低路由方差，起到正则化效果

#### 专家数量与宽度权衡
- **E48 vs E16**：在相同激活参数下，48专家+窄中间层（hidden×0.3）一致优于16专家+宽层（hidden×4）
- **训练损失**：E48配置在所有规模下均获得更低的MSE损失，收敛更稳定

---

## 4. 关键结论和发现

### 论文的主要发现
1. **架构设计 > 路由创新**：LLM中验证的MoE架构（DeepSeek风格）在扩散模型中同样高度有效，性能提升主要来自架构而非复杂路由
2. **参数重分配策略**：增加专家数量同时缩小中间层宽度，能在**减少总参数30-50%**的前提下提升性能，实现"免费午餐"
3. **2D位置信息至关重要**：显式编码行列结构的2D RoPE对视觉生成任务不可或缺，显著改善空间推理和优化稳定性
4. **共享专家的正则化效应**：共享专家提供稳定的全局表示，防止路由崩溃，是训练稳定性的关键
5. **跨框架通用性**：该设计范式在latent和pixel-space扩散框架中均一致有效

### 方法的局限性
- **评估范围有限**：主要在ImageNet 256×256上验证，尚未充分覆盖文本到图像、视频生成等更复杂任务
- **路由策略简单**：当前采用无辅助损失的Top-K路由，未探索更复杂的时间/空间自适应路由
- **专家专业化分析**：虽观察到专家激活模式差异，但未深入分析其学到的具体视觉概念

### 未来工作方向
1. **架构+路由结合**：将本文的架构改进与更强的路由策略（如时间感知、空间自适应路由）相结合
2. **任务扩展**：应用于文本到图像生成、视频生成等更复杂的生成任务
3. **效率优化**：探索GQA/MLA等注意力压缩技术在扩散MoE中的潜力
4. **强化学习**：研究扩散MoE上的RL微调方法
5. **开源生态**：已开源代码和模型权重，促进社区系统性探索

---

**总结**：该论文通过**架构优先**的实用主义路线，为Diffusion MoE提供了简单高效的训练配方，证明**借鉴LLM成熟设计**比**发明复杂路由**更能释放扩散模型的MoE潜力，在参数效率、训练速度和生成质量上实现三重突破。

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

#### AI Summary (by kimi-k2-thinking)
# KV Pareto: 长上下文推理中KV缓存与模型压缩的系统级优化

## 1. 主要贡献和创新点

### 解决的问题
长上下文LLM推理面临**KV cache内存瓶颈**——KV cache随序列长度线性增长，导致边缘部署时内存消耗过大、延迟增加（TTFT和TPOT）。现有优化技术（KV量化、chunked prefill、权重量化）通常被**孤立评估**，其联合效果与最优配置缺乏系统性研究。

### 提出的新方法
**KV Pareto框架**：首个系统级联合优化框架，同时考虑三种互补技术：
- **KV cache量化**：支持int2/4/8及混合精度（k8v8, k8v4, k8v2, k4v4, k4v2, k2v2），多种粒度（per-token, per-tensor, per-block）
- **Prefill Chunking (PC)**：将输入分块处理以降低峰值内存
- **AWQ权重量化**：4-bit激活感知权重量化

### 相比现有方法的优势
- **联合评估**：首次系统性地量化三种技术的**协同效应**与**权衡关系**，而非孤立分析
- **Pareto最优**：为每个模型识别内存-精度的最优前沿配置，提供可部署的实用方案
- **轻量化**：无需重新训练，即插即用，适用于多种LLM架构
- **系统级视角**：综合考虑峰值内存、KV cache内存和模型内存的总消耗

---

## 2. 核心实验方法和设置

### 评估模型
覆盖多种架构和规模：
- **Qwen2.5**-3b/7b-instruct
- **Llama3.2**-3b-instruct / **Llama3.1**-8b-instruct
- **Mistral**-7b-instruct-v0.3

### 数据集与任务
| 类型 | 数据集 | 评估重点 |
|------|--------|----------|
| 长上下文 | LongBench (HotpotQA, Qasper) | 多文档/单文档QA (F1) |
| 检索能力 | Needle-in-a-Haystack (NIAH) | 最长32k上下文的文本检索 |
| 短上下文 | GSM8k | 数学推理 (生成任务) |
| 短上下文 | MMLU | 多任务理解 (非生成任务) |

### 实验设置
- **Chunk size**：固定为256（消融实验显示64-1024范围内性能无显著差异）
- **KV量化**：采用**round-to-nearest (RTN)**非对称量化，结合**k-smoothing**（沿序列维度均值平滑）改善分布
- **权重量化**：AWQ 4-bit unsigned，group size 128
- **评估指标**：
  - **总内存** = 峰值内存 + KV cache内存 + 模型内存
  - **任务精度**：各数据集的标准指标

### 基线方法
- **w16a16_k16v16**：FP16权重 + FP16 KV cache（无优化）
- 对比PC开启/关闭、AWQ启用/禁用的组合效果

---

## 3. 主要实验结果和性能指标

### Pareto最优配置（10k上下文长度）
| 模型 | 最优配置 | 内存减少 | 长文精度损失 |
|------|----------|----------|--------------|
| Qwen2.5-3B | **w4a16-k4v4** + PC | **73%** | ~1-3% |
| Llama3.2-3B | **w4a16-k4v4** + PC | **76%** | ~1-3% |
| Mistral-7B | **w4a16-k4v4** + PC | **78%** | ~1-3% |
| Qwen2.5-7B | **w4a16-k8v8** + PC | **68%** | ~1-3% |
| Llama3.1-8B | **w4a16-k8v2** + PC | **75%** | ~1-3% |

**注**：w4a16表示4-bit权重，k4v4表示key和value均为4-bit量化

### 关键性能数据
- **内存节省**：总体达到 **68-78%** 总内存减少
- **精度保持**：长上下文任务（HotpotQA/Qasper）仅 **1-3%** 精度下降
- **短任务影响**：GSM8k下降 **1-10%**（AWQ影响较大），MMLU下降 **1-5%**

### NIAH验证结果
- **Mistral-7B w4a16-k4v4**：在**20k tokens内**保持稳定的检索性能（>90%准确率）
- **Qwen-2.5-3B**：w4a16-k4v4在**14k内**性能良好，基线在26k后显著下降

### 消融实验发现
1. **量化粒度**：**per-token** 显著优于per-tensor/per-block
   - 小模型（3B）：group size **32**最优
   - 大模型（7B/8B）：group size **64**最优
2. **K-smoothing**：对**int4**至关重要，无平滑会导致性能崩溃
3. **PC效果**：仅降低峰值内存，**几乎不影响任务精度**
4. **AWQ协同效应**：与k4v4组合时，部分模型在HotpotQA上**精度不降反升**（归因于k-smoothing的正则化效果）

---

## 4. 关键结论和发现

### 主要发现
1. **联合优化必要性**：单一技术无法达到最优权衡，PC+AWQ+KV量化的组合才能实现**78%内存节省**
2. **模型特异性**：最优配置因模型而异，不存在通用最优解（如Qwen-7B需k8v8，而Llama-8B需k8v2）
3. **系统级收益**：在128k上下文下，小chunk size（1k）额外节省**23%**内存，FlashAttention再节省**6%**
4. **精度鲁棒性**：Pareto配置在**20k以内**上下文长度泛化良好，超出后需微调恢复

### 方法局限性
- **固定chunk size**：未探索动态分块策略，128k场景下chunk size选择关键
- **量化算法**：采用简单的RTN量化，未使用Hessian旋转等先进方法（如QuaRot, SpinQuant）
- **延迟未优化**：PC可能增加延迟，当前框架未将**latency**作为优化目标
- **长上下文验证不足**：未全面评估128k以上长度的任务性能退化程度

### 未来工作方向
1. **动态chunk sizing**：根据上下文长度和硬件条件自适应调整
2. **先进量化技术**：引入Hessian矩阵旋转、层自适应混合精度
3. **延迟-内存-精度三目标Pareto**：将推理延迟纳入优化空间
4. **更大规模验证**：在128k+长度上系统评估任务性能
5. **架构扩展**：支持Granite等混合架构模型

---

**核心价值**：KV Pareto为边缘部署长上下文LLM提供了**可操作的配置选择指南**，在极限内存压缩与精度保持间找到了实用平衡点。

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
# Heimdall++: Optimizing GPU Utilization and Pipeline Parallelism for Efficient Single-Pulse Detection

## 1. 论文的主要贡献和创新点

### 解决的问题
- **GPU利用率低下**：原始Heimdall在单脉冲搜索中GPU平均利用率仅51.2%，存在严重的"GPU stall problem"
- **数据传输瓶颈**：处理1GB文件产生7.85GB的PCIe传输量，频繁的主机-设备数据拷贝导致GPU空闲等待
- **串行执行限制**：DM trials循环采用串行执行，无法充分利用GPU大规模并行能力
- **多文件处理效率低**：多进程并发导致CUDA context contention，资源竞争严重，可扩展性差
- **内存分配开销**：Peak Detection阶段重复调用Thrust库，产生198,392次cudaMalloc调用

### 提出的新方法
- **细粒度并行化架构**：将DM trials循环分解为独立任务，通过OpenMP多线程和CUDA多流(multiple streams)实现并发执行
- **统一内存管理**：采用CUDA Unified Memory消除显式主机-设备数据拷贝，实现按需页面迁移
- **多线程共享设备内存分配器**：设计基于双队列(allocated-block/free-block)的内存池机制，支持跨线程内存块复用
- **流水线并行框架**：将CPU-bound的Pipeline Creation与GPU-bound的Execution解耦，通过线程安全任务队列实现异步重叠执行

### 相比现有方法的优势
- **性能提升显著**：单文件处理速度提升2.66倍，多文件批处理提升2.05倍
- **GPU利用率飞跃**：从51%提升至92%，消除GPU空闲等待
- **数据传输量锐减**：PCIe传输量减少6.7倍(7.85GB→1.17GB)
- **内存分配开销降低**：cudaMalloc调用减少41.2倍(198,392→4,810次)
- **结果完全一致**：保持与原始Heimdall完全一致的搜索结果
- **可扩展性增强**：避免多进程资源竞争，支持更高并发度

## 2. 核心实验方法和设置

### 实验环境
- **硬件**：NVIDIA GeForce RTX 3080Ti GPU + 12th Gen Intel Core i9-12900K CPU
- **软件**：Ubuntu 20.04.6 LTS, NVIDIA CUDA Toolkit
- **基线方法**：原始Heimdall实现

### 数据集
| 数据集 | 规模 | 来源 | 用途 |
|--------|------|------|------|
| J0528_2200_arcdrift-M01_0009.fits | 1GB (转换后) | CRAFTS巡天 | 阶段级性能分析 |
| M5球状星团(NGC 5904) | 142GB (30分钟观测) | FAST存档数据 | 大文件处理性能测试 |
| FRB20201124子集 | 125个文件，每个488MB | FAST-FREX数据集 | 多文件批处理测试 |

### 评估指标
- **端到端处理时间**和**加速比(speedup)**
- **GPU利用率**（通过Nsight Systems profiling）
- **主机-设备数据传输量**
- **CUDA内存分配调用次数**
- **各处理阶段执行时间**（RFI Mitigation, Dedispersion, Candidate Merging等）
- **搜索结果一致性验证**

### 实验配置
- **DM范围**：0–1000 cm⁻³
- **Chunk大小**：256K samples
- **并行度设置**：1, 2, 4, 6, 8（线程数/CUDA流数）

## 3. 主要实验结果和性能指标

### 整体性能加速
- **单文件处理**：并行度为8时达到**3.40倍**加速（1GB文件）
- **大文件处理**：142GB文件处理速度提升**2.66倍**（并行度8）
- **多文件批处理**：125个文件批处理速度提升**2.05倍**（并行度4）

### GPU利用率对比
- **Heimdall**：平均利用率仅**51.17%**，存在长时间空闲
- **Heimdall++**：并行度8时平均利用率提升至**92.11%**，消除GPU stall

### 数据传输优化
| 传输方向 | Heimdall | Heimdall++ | 减少倍数 |
|----------|----------|------------|----------|
| Host-to-Device | 5.19 GB | 716.87 MB | 7.2× |
| Device-to-Host | 2.66 GB | 480.75 MB | 5.5× |
| **总计** | **7.85 GB** | **1.17 GB** | **6.7×** |

### 内存分配优化
- **cudaMalloc调用次数**：从198,392次降至4,810次，**减少41.2倍**
- **内存池机制**：通过块复用避免重复分配/释放开销

### 各阶段加速比（并行度8）
| 处理阶段 | 加速比 | 优化技术 |
|----------|--------|----------|
| Baseline Removal | 4.33× | 多流并行 |
| Normalization | **6.05×** | 多流并行 |
| Matched Filtering | 4.33× | 多流并行 |
| Peak Detection | 4.33× | 多流并行+内存池 |
| RFI Mitigation | 3.25× | 统一内存 |
| Dedispersion | 1.59× | 统一内存 |
| Candidate Merging | 显著提升 | Shared memory优化 |

### 可扩展性表现
- **并行度1**：仍比Heimdall快1.42倍（其他优化贡献）
- **并行度2-4**：加速比线性增长，达到2.06×和2.84×
- **并行度6-8**：边际效益递减，受GPU硬件能力限制
- **多文件场景**：支持4线程稳定运行，而Heimdall多进程在2进程时即因内存不足崩溃

## 4. 关键结论和发现

### 主要发现
1. **并行化是提升GPU利用率的关键**：将串行的DM trials循环分解为独立任务，通过多线程+多流实现并发，使GPU occupancy接近饱和
2. **统一内存有效解决容量-带宽权衡**：Unified Memory自动管理页面迁移，在保持大chunk尺寸（高并行度）的同时，避免显式PCIe传输
3. **内存分配开销不容忽视**：在数千次DM trial重复执行中，cudaMalloc/cudaFree累积开销显著，内存池策略带来数量级改善
4. **CPU-GPU流水线重叠至关重要**：解耦Pipeline Creation与Execution，通过双缓冲和任务队列隐藏I/O延迟，是批处理场景的核心优化

### 方法局限性
- **硬件依赖性**：最优并行度需根据GPU计算能力 empirical tuning，过高并行度收益递减
- **内存容量限制**：尽管使用Unified Memory，极端DM范围和大chunk尺寸仍可能超出系统内存
- **多文件扩展性**：超过4线程后可能遇到系统资源限制，需要workload-specific配置
- **算法复杂度**：Candidate Clustering的O(N²)复杂度在大规模候选体场景仍可能成为瓶颈

### 未来工作方向
- **动态并行度调整**：根据实时GPU负载和输入数据特征自动优化线程/流数量
- **多GPU支持**：将架构扩展到多GPU系统，处理超大规模巡天数据
- **算法优化**：进一步改进Candidate Clustering算法，降低复杂度至O(N log N)或更低
- **集成AI加速**：探索将机器学习模型集成到pipeline中，用于RFI抑制或候选体分类
- **异构计算支持**：结合CPU、GPU和其他加速器（如FPGA）实现更高效的异构处理

### 科学意义
Heimdall++为实时处理下一代射电望远镜（如SKA、FAST）产生的TB/s级数据流提供了可行方案，将计算成本降低2-3倍，为大规模时域巡天奠定了软件基础。

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
- **扩散型大语言模型（dLLMs）推理效率瓶颈**：dLLMs通过迭代去噪生成token，但通常需要固定数量的去噪步骤，即使答案在早期就已经稳定，造成计算资源浪费
- **训练元数据浪费问题**：在训练过程中，优化动态（特别是梯度信息）产生了关于参数重要性的丰富元数据，但这些信息在训练完成后通常被丢弃，未被用于指导推理

### 提出的新方法
- **EDIT（Early Diffusion Inference Termination）**：一种基于训练梯度动态的推理时提前终止准则
- **核心思想**：利用AdamW优化器在监督微调（SFT）期间的动量估计来构建"AdamW演化"模式，作为模型学习到的推理路径的紧凑表示
- **工作机制**：在推理时，将当前token激活与保存的AdamW演化向量进行对齐度比较，当对齐度在连续步骤中保持稳定时，提前终止去噪过程

### 相比现有方法的优势
- **无需架构修改**：EDIT是即插即用的方法，不需要改变模型结构
- **极小存储开销**：仅需存储约1.5-2MB的元数据（占8GB模型的0.02%）
- **理论保证**：提供基于KL散度和总变差距离的收敛性证明，确保提前终止的可靠性
- **性能提升**：在减少11.8%-68.3%去噪步骤的同时，多数任务准确率保持或提升

## 2. 核心实验方法和设置

### 数据集
在五个推理基准上评估：
- **Countdown**：算术推理任务
- **Sudoku**：数独求解
- **MATH500**：数学竞赛题
- **GSM8K**：小学数学应用题
- **GPQA**：研究生级别问答

### 实验设置
- **基线模型**：LLaDA-8B
- **微调数据**：s1数据集
- **适配方法**：在QKV投影上应用LoRA（Low-Rank Adaptation）
- **序列长度**：128、256、512三种设置
- **硬件**：Intel XPU

### 评估指标
- **准确率**：各任务上的推理正确率
- **效率**：平均去噪步骤数、步骤减少百分比
- **存储开销**：元数据大小
- **计算开销**：额外操作的时间复杂度

### 基线方法对比
- **LLaDA (No SFT)**：未微调的原始模型
- **LLaDA (SFT)**：监督微调后的模型，使用完整去噪步骤（64/128/256步对应序列长度128/256/512）
- **EDIT (Ours)**：使用AdamW演化元数据的提前终止方法

## 3. 主要实验结果和性能指标

### 关键性能数据

**准确率结果（表1）**：
| 数据集 | 序列长度 | LLaDA (No SFT) | LLaDA (SFT) | EDIT (Ours) | 提升幅度 |
|--------|----------|----------------|-------------|-------------|----------|
| Countdown | 128 | 19.9% | 19.5% | **28.9%** | +9.0% |
| Countdown | 256 | 19.5% | 20.7% | **31.6%** | +10.9% |
| Sudoku | 128 | 10.4% | 11.4% | **16.1%** | +4.7% |
| MATH500 | 256 | 32.4% | 30.4% | **32.8%** | +2.4% |
| GSM8K | 256 | 75.8% | 77.0% | 77.6% | +0.6% |
| GPQA | 256 | 27.9% | 20.5% | **27.7%** | +7.2% |

**效率提升（表2）**：
- **平均步骤减少**：11.8% - 68.3%
- **最佳表现**：Countdown任务在序列长度256时减少68.3%（从128步降至40.6步）
- **典型表现**：多数任务减少35-45%的去噪步骤
- **GSM8K例外**：在序列长度512时准确率从81.2%降至76.2%（长推理链可能过早稳定）

### 消融实验结果

**模块选择分析（表4）**：
- **Query投影的LoRA-B矩阵**配合行方向能量归约表现最佳（平均KL散度0.089）
- 优于Key/Value投影和LoRA-A矩阵

**超参数敏感性**：
- 阈值δ：在{0.025, 0.05, 0.1, 0.25, 0.45, 0.55}中选择
- 稳定跨度Ω：在{6, 8, 10, 12}中选择
- 块温度τblk：固定为1.0

**梯度对齐验证**：
- 在GPQA任务上，伪梯度在第19步收敛到SFT均值附近
- 提前终止（~20步/块）与完整去噪相比保持相当的准确率

## 4. 关键结论和发现

### 主要发现
1. **训练元数据的价值**：AdamW优化器在微调期间的动量估计确实编码了参数重要性信息，这些信息可以作为推理路径的可靠指示器
2. **效率与质量兼得**：在多数推理任务上，EDIT能在显著减少计算步骤的同时保持或提升准确率，特别是在结构化推理任务（Countdown、Sudoku）上效果最明显
3. **任务特异性**：不同任务和子领域（如GPQA中的分子生物学vs天体物理学）激活不同的参数子集，验证了AdamW演化模式能捕捉任务特定的推理路径
4. **理论保证**：基于KL散度的稳定性检测提供了可证明的收敛保证，72.3%的提前终止满足PAC风格的正确性证书

### 方法的局限性
1. **依赖训练动态**：需要访问训练过程中的AdamW状态，对已发布模型（通常只提供权重）不适用
2. **任务特定阈值**：当前方法需要为不同任务调整(δ, Ω)超参数，虽然可通过验证集选择，但增加了使用成本
3. **长序列推理挑战**：在需要长链推理的任务（如GSM8K长序列）上可能过早终止，导致准确率下降
4. **LoRA限制**：当前评估仅限于LoRA微调，全参数微调的扩展尚未验证

### 未来工作方向
1. **全参数扩展**：将EDIT应用于全参数微调场景
2. **自适应阈值**：开发基于学习的或动态调整的终止准则
3. **更广泛的应用**：探索训练元数据在动态计算分配、token级处理、早期质量预测等方面的应用
4. **标准化建议**：呼吁模型提供者在发布模型时包含优化元数据，以支持更高效的推理
5. **系统性改进**：解决当前ML pipeline中训练信息被丢弃的低效问题，推动更整体化的训练和推理设计

---

**核心洞见**：EDIT揭示了一个被忽视的机会——训练过程中产生的优化动态本身就是宝贵的资产，而非副产品。通过智能地保存和利用这些信息，可以在不增加模型复杂度的情况下实现显著的推理效率提升。

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
- **Chain-of-Thought (CoT) 推理的局限性**：标准CoT在解码过程中存在**内部确定性**（internal determinism），限制了模型探索合理替代推理路径的能力
- **Soft Thinking 的缺陷**：虽然通过连续语义空间增强了表示能力，但仍受自回归解码的贪婪性制约，无法并行表示多样化语义轨迹，导致**推理噪声累积**和**单一路径强化**

### 提出的新方法：M3PO框架
**Multi-Path Perception Policy Optimization (M3PO)** 是一种新颖的强化学习框架，核心创新包括：

1. **并行Rollout作为自然多样性来源**：利用RL训练中固有的多个独立策略rollout，无需额外辅助数据集即可生成多样化推理路径
2. **跨路径协作机制**：设计轻量级、**无参数**的gating函数，在推理步骤级别实现轨迹间动态信息交换
3. **策略优化闭环**：将协作精炼的轨迹用于基于组相对优势估计的策略更新，形成"探索-协作-优化"的闭环

### 相比现有方法的优势
- **打破自增强循环**：通过跨路径反馈及时纠正错误前提，防止错误逻辑传播
- **保持模型兼容性**：在输入嵌入空间操作，无需偏离预训练嵌入空间
- **推理效率**：训练时多路径协作，**推理时单路径解码**，不增加部署成本
- **性能提升显著**：在知识密集型任务上平均提升9.5%，超越GRPO达9.5个百分点

---

## 2. 核心实验方法和设置

### 数据集
**知识密集型任务**（5个开放域和多跳QA数据集）：
- Natural Questions (NQ)
- TriviaQA
- HotpotQA
- 2WikiMultiHopQA (2WikiMQA)
- Bamboogle

**推理密集型STEM任务**（5个数学和科学数据集）：
- GSM8k
- MATH
- MATH500
- MMLU-STEM
- ARC-Challenge

### 实验设置
- **模型规模**：Qwen2.5-1.5B-Instruct 和 Qwen2.5-3B-Instruct
- **训练配置**：
  - 使用LoRA（rank=32）进行参数高效微调
  - Group size N：知识任务为4，复杂推理任务为8
  - 学习率：5e-6，KL系数β=0.005
  - 混合精度训练（BF16）
- **评估指标**：Exact Match (EM) for QA，Accuracy for STEM
- **检索设置**：使用E5-base嵌入模型检索Top-3 Wikipedia文档作为上下文

### 基线方法对比
- **监督微调**：SFT
- **强化学习方法**：PPO, GRPO
- **检索增强**：RAG, IRCoT, Search-o1
- **混合推理方法**：HRPO（基于Soft Thinking的RL方法）
- **大模型Few-shot**：DeepSeekMath-7B, Gemma-2-9B, Qwen2.5-7B等

---

## 3. 主要实验结果和性能指标

### 知识密集型任务性能
| 模型 | 方法 | NQ | TriviaQA | HotpotQA | 2WikiMQA | Bamboogle | **平均** |
|------|------|-----|----------|----------|----------|-----------|---------|
| Qwen2.5-7B | RAG | 34.9 | 58.5 | 29.9 | 23.5 | 20.8 | 33.5 |
| Qwen2.5-1.5B | **M3PO** | **41.4** | 56.8 | 28.7 | 27.9 | 23.2 | **35.6** |
| Qwen2.5-3B | **M3PO** | **44.1** | **61.0** | **33.2** | 31.4 | **31.2** | **40.2** |

**关键发现**：
- **1.5B模型**：M3PO平均EM 35.6%，**超越Qwen-7B RAG基线2.1%**
- **3B模型**：M3PO平均EM 40.2%，**超越7B RAG基线6.7%**
- **相对GRPO提升**：在1.5B设置下，M3PO超越GRPO **9.5%**（35.6% vs 26.1%）

### 推理密集型STEM任务性能
| 模型 | 方法 | GSM8k | MATH | MATH500 | MMLU-ST | ARC-C | **平均** |
|------|------|-------|------|---------|---------|-------|---------|
| Qwen2.5-7B | CoT | 85.4 | 49.8 | 46.4 | 72.3 | 63.7 | 63.5 |
| Qwen2.5-3B | **M3PO** | **84.8** | **60.7** | **63.0** | **61.6** | **82.6** | **70.5** |

**关键发现**：
- **3B模型平均准确率70.5%**，**超越最强7B基线5.3%**
- **MATH基准**：3B模型达到60.7%，**超越最佳7B基线10.9个百分点**
- **全面超越HRPO**：在所有数据集上均优于同规模的HRPO方法

### 消融实验结果

#### 不同推理范式对比（Qwen2.5-3B on MATH）
| 方法 | 准确率 | 收敛速度 | 训练稳定性 |
|------|--------|----------|------------|
| Hidden States | 0%（失败） | 未收敛 | 极差 |
| Soft Thinking | ~45% | 慢 | 低 |
| HRPO | 58.6% | 中等 | 中等 |
| **M3PO** | **60.7%** | **最快** | **最高** |

#### 跨路径融合机制消融
- **No Cross-Path**（λ=0）：性能最低，退化为标准CoT
- **Peer Mean**（均匀平均）：性能下降，验证选择性gating的必要性
- **M3PO（相似度加权）**：性能最优，证明分布相似性引导的有效性

#### 超参数敏感性分析
- **λ（混合系数）**：最优值0.1，λ≥0.5时性能崩溃
- **T（温度参数）**：最优值0.1，较高温度降低选择性导致性能下降

---

## 4. 关键结论和发现

### 主要发现
1. **多路径协作的有效性**：并行rollout提供了自然的推理多样性，跨路径交互能显著增强策略学习的鲁棒性，超越单纯的组相对优势估计
2. **轻量级设计的优势**：无参数的gating机制在保持计算效率的同时，实现了与参数化方法相当或更优的性能
3. **推理-效率平衡**：训练时多路径探索，推理时单路径解码，在不增加部署成本的前提下解锁大模型能力
4. **错误纠正机制**：协作机制能及时识别并抑制错误推理路径，防止错误在自回归生成中累积放大

### 方法局限性
- **计算资源限制**：实验最大仅探索到3B参数模型，更大规模的扩展性有待验证
- **协作机制固定**：当前采用静态的相似度加权策略，缺乏自适应的动态调整能力
- **任务范围**：主要聚焦于知识问答和数学推理，其他复杂推理场景（如代码生成）未充分验证

### 未来工作方向
- **规模化研究**：在更大模型（7B+）上验证M3PO的有效性和效率
- **自适应协作**：开发动态调整协作强度和选择性的机制
- **更广泛评估**：扩展到代码推理、多模态推理等更多任务类型
- **理论分析**：深入研究多路径协作对策略优化收敛性的理论保证

---

## 核心贡献总结

M3PO通过**强化学习驱动的多路径协作推理**，成功解决了传统CoT和Soft Thinking的固有缺陷。其**无参数、步骤级**的协作机制在训练时实现跨轨迹知识共享，推理时保持单路径效率，在**1.5B和3B小模型上实现了超越7B大模型的性能**，为高效、鲁棒的LLM推理提供了新范式。

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

#### AI Summary (by kimi-k2-thinking)
# Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling - 论文总结

## 1. 主要贡献和创新点

### 解决的问题
- **NVFP4量化精度瓶颈**：标准NVFP4量化在训练和推理中导致模型性能显著下降，甚至训练发散。根本原因在于FP4格式的非均匀量化步长（0.5/1/2）导致**接近最大值的数值（near-maximal values）**量化误差极大。
- **硬件限制**：NVIDIA Blackwell GPU要求矩阵乘法的所有操作数（权重、激活值、梯度）都必须量化为NVFP4，加剧了精度损失问题。

### 提出的新方法
- **Four Over Six (4/6)**：一种**自适应块缩放**算法，为每个16值块动态选择两个候选缩放因子之一：
  - **Scale=6**：标准NVFP4，使用完整FP4范围[-6, 6]
  - **Scale=4**：缩小动态范围至[-4, 4]，使量化分布更均匀，更好表示75%最大值附近的数值
- **实现方式**：对每个块同时量化两次（scale=4和scale=6），通过比较量化误差（MSE）自适应选择更优方案。

### 相比现有方法的优势
- **精度提升**：显著降低near-maximal值的量化误差，在PTQ中使WikiText-2 perplexity平均提升19.9%（相对于BF16）。
- **训练稳定性**：有效防止NVFP4训练发散，loss曲线接近BF16基线。
- **低开销**：在Blackwell GPU上通过PTX指令高效实现，推理序列长度≤16384时开销<2%，训练时<15%。
- **即插即用**：可无缝集成到现有PTQ方法（GPTQ/AWQ/SmoothQuant）中，无需修改训练流程。

---

## 2. 核心实验方法和设置

### 数据集
- **语言建模**：WikiText-2, C4
- **下游任务**：BoolQ, ARC-Easy, ARC-Challenge, HellaSwag

### 模型架构
- **预训练实验**：自定义Transformer、Hybrid（滑动窗口Attention）、Hybrid with Gated Attention（340M/1.3B/1.4B参数）
- **PTQ实验**：Llama-3系列（1B, 8B, 70B），Qwen-3系列（1.7B, 8B, 32B）

### 评估指标
- **Perplexity**（↓）：WikiText-2, C4词困惑度
- **任务准确率**（↑）：BoolQ, ARC-Easy, ARC-Challenge, HellaSwag（归一化）
- **训练稳定性**：训练loss曲线和发散情况

### 基线方法
- **高精度**：BF16
- **量化格式**：MXFP4, NVFP4 (M=6)
- **PTQ方法**：RTN, GPTQ, AWQ, SmoothQuant

---

## 3. 主要实验结果和性能指标

### 预训练结果（图4）
| 模型架构 | 参数 | BF16 | NVFP4 | NVFP4+4/6 | 效果 |
|---------|------|------|-------|-----------|------|
| Hybrid | 340M | 2.6 | **发散** | 2.7 | 防止发散 |
| Hybrid | 1.3B | 2.4 | **发散** | 2.5 | 防止发散 |
| Transformer | 340M | 2.6 | **发散** | 2.6 | 稳定训练 |
| Hybrid+Gated | 1.4B | 2.4 | **发散** | 2.5 | 防止发散 |

**关键发现**：4/6在所有测试架构中均成功防止NVFP4训练发散，loss接近BF16。

### PTQ结果（表5-7）

**WikiText-2 Perplexity（↓）**
| 方法 | Llama-3-8B | Llama-3-70B | Qwen-3-8B | 平均提升 |
|------|------------|-------------|-----------|----------|
| BF16 | 7.54 | 2.86 | 12.22 | - |
| NVFP4 (RTN) | 8.43 | 4.00 | 12.68 | - |
| **+4/6** | **8.30** | **3.83** | **12.56** | **+19.9%** |
| AWQ | 8.33 | 3.86 | 12.68 | - |
| **AWQ+4/6** | **8.24** | **3.71** | **9.64** | **最佳** |

**下游任务准确率（↑）**
- **Llama-3-8B**：AWQ+4/6平均准确率73.1%（vs 72.2% AWQ）
- **Qwen-3-8B**：RTN+4/6平均准确率72.9%（vs 72.3% RTN）
- **普遍趋势**：4/6在几乎所有任务和方法中提升或保持性能

### 消融实验（表4）
| 块选择策略 | Llama-3-8B | Qwen-3-8B | 结论 |
|-----------|------------|-----------|------|
| MSE | **8.30** | **12.56** | **最优** |
| L1 Norm | 8.33 | 12.63 | 次优 |
| Abs-Max | 8.36 | 12.86 | 较差 |

---

## 4. 关键结论和发现

### 主要发现
1. **误差根源**：NVFP4性能损失主要来自**FP4数值量化误差**（而非scale factor误差），特别是无法精确表示5附近的值。
2. **自适应有效性**：统一使用scale=4反而比scale=6更差（表3），但**自适应选择**能显著提升精度。
3. **硬件友好**：利用Blackwell的PTX指令，4/6可在寄存器中完成双重量化和误差计算，开销可控。

### 方法局限性
- **格式限制**：**不适用于MXFP4**，因其scale factor使用FP8E8M0格式，无法精确表示4和6的50%差异。
- **模型依赖**：对于outlier极少的模型，INT4可能更优。
- **开销**：训练时最大序列长度（131072）下仍有~15%开销，需进一步优化。
- **兼容性**：与QuaRot/SpinQuant等rotation-based方法结合效果不佳。

### 未来工作方向
1. **优化实现**：进一步降低CUDA kernel开销，接近零成本。
2. **大规模训练**：在更大模型（>70B）和更长训练周期验证有效性。
3. **算法融合**：探索4/6与GPTQ优化过程的深度集成（而非仅替换量化层）。
4. **格式扩展**：为未来的块缩放浮点格式设计更通用的自适应缩放框架。

---

**总结**：4/6通过简单的"两次量化选优"策略，精准解决了NVFP4的致命精度问题，在保持硬件效率的同时实现了训练稳定性和推理精度的双重提升，是NVFP4量化走向实用的关键一步。

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
# 论文总结：A Parallel and Distributed Rust Library for Core Decomposition on Large Graphs

## 1. 主要贡献和创新点

### 解决的问题
- **大规模图核分解的计算挑战**：尽管存在线性时间复杂度的顺序算法（Batagelj和Zaversnik），现代图数据（数亿顶点、数十亿边）的规模对单核内存带宽和缓存容量构成压力，需要多核并行化方案
- **分布式算法在共享内存系统的适配难题**：将Montresor等人的分布式消息传递协议适配到共享内存架构时面临同步成本、缓存一致性和共享数据结构竞争等非平凡问题

### 提出的新方法
- **Rust实现的渐进优化框架**：开发了三个版本：
  - **SequentialK**：基于消息传递的基线实现，验证正确性
  - **ParallelK**：引入多线程消息传递，使用Rayon并行迭代器、线程池和原生Rust线程三种策略
  - **FastK**：核心创新版本，采用**全局共享状态**设计，结合缓存感知数据结构、选择性消息传播和动态激活策略

- **关键优化技术**：
  - **全局共享向量**：将`est`（coreness估计值）和`active`（激活状态）向量化，使用raw pointers避免Arc/Mutex开销
  - **选择性消息发送**：仅当`estimate(u) < estimate(v)`时才发送消息，利用coreness值单调不增的特性
  - **动态并行度调整**：在活跃节点数低于batch size时切换到顺序执行（优先队列），避免高频重计算

### 相比现有方法的优势
- **性能突破**：FastK在16线程上实现**高达11×加速**，比NetworkX快**两个数量级**
- **内存安全与性能共存**：Rust的所有权模型消除数据竞争，同时保持底层控制能力
- **同步开销最小化**：通过合并处理阶段、减少barrier数量和锁竞争，显著提升扩展性

---

## 2. 核心实验方法和设置

### 数据集
使用**SNAP真实世界数据集**，涵盖多种类型和规模：
- **道路网络**：roadNet-PA/TX/CA（100万-200万顶点，低度数，kmax=3）
- **Web图**：web-NotreDame/Stanford/Google/BerkStan（28万-68万顶点，度数较高，kmax=44-201）
- **社交网络**：soc-Pokec（160万顶点，3000万边）、soc-LiveJournal（480万顶点，6900万边，kmax=372）
- **通信网络**：wiki-Talk（240万顶点，500万边）

### 实验环境
- **硬件**：双路AMD EPYC 7551（32核/路，共64物理核），128GB共享内存
- **线程数**：1, 2, 4, ..., 128（超线程启用）
- **编译**：Rust `--release`模式 + LTO优化
- **参数**：Batch size=256节点/线程（经调优确定）

### 评估指标与基线
- **运行时间**：平均5次执行（不含图加载时间，含并行准备开销）
- **收敛速度**：每轮迭代中节点估计值与最终coreness的平均偏差
- **基线方法**：
  - NetworkX Python实现（顺序）
  - SequentialK（Rust顺序基线）
  - ParallelK（Rust并行中间版本）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（16线程）
| 图数据集 | FastK (秒) | NetworkX (秒) | 加速比 |
|---------|-----------|--------------|--------|
| roadNet-CA | **0.35** | 40.48 | **116×** |
| soc-LiveJournal1 | **3.87** | 480.92 | **124×** |
| web-BerkStan | **0.59** | 686.63 | **1164×** |
| web-Google | **0.23** | 38.40 | **167×** |
| wiki-Talk | **0.61** | 115.28 | **189×** |

### 与基线对比
- **vs NetworkX**：FastK平均快**100-1000倍**，最大达1164倍（web-BerkStan）
- **vs SequentialK**：FastK快**3-20倍**（如soc-LiveJournal1: 78.11s → 3.87s）
- **vs ParallelK**：FastK持续快**1.5-3倍**（如web-BerkStan: 2.22s → 0.59s）

### 消融实验结果
1. **Batch Size调优**（图1）：
   - 256节点/线程为最优平衡点，过小导致线程管理开销，过大导致负载不均

2. **数据结构选择**（图2）：
   - **排序向量 vs 哈希表**：在真实图（低平均度数）上，排序向量快**30%**，得益于缓存友好性和避免哈希计算

3. **并行策略对比**（图3）：
   - **原生线程 > Rayon线程池 > Rayon并行迭代器**：原生线程管理开销最低，Rayon引入额外抽象层成本

4. **收敛行为分析**（图4-5）：
   - **快速初始收敛**：前10-20轮误差下降90%以上
   - **长尾效应**：后期仅**<5%节点**活跃，但需50-300轮精细调整（误差±1）
   - **动态切换收益**：在活跃节点<batch size时切换到顺序执行，避免无效并行

5. **扩展性**（图7）：
   - **亚线性加速**：超过64线程后收益递减，因同步开销和NUMA效应
   - **最优线程数**：16-32线程达到峰值性能（11×加速）

---

## 4. 关键结论和发现

### 主要发现
1. **分布式协议可高效适配共享内存**：Montresor的消息传递模型在最小化同步点后，在共享内存上表现优异
2. **消息选择性是关键**：通过条件发送（`estimate(u) < estimate(v)`）减少**50%以上**无效消息和锁竞争
3. **动态并行度调整有效**：根据活跃节点比例切换执行模式，平衡并行收益与开销
4. **Rust适合高性能图计算**：内存安全保证不牺牲性能，零成本抽象和细粒度并发原语是核心优势

### 方法局限性
- **静态图假设**：未处理动态增量/减量更新
- **单节点限制**：实验限于单NUMA机器，未跨节点分布式验证
- **图加载时间未计入**：报告时间仅包含计算，忽略I/O开销
- **NUMA效应未优化**：未针对跨socket内存访问做特殊处理

### 未来工作方向
1. **真正分布式部署**：保留优化的共享内存内核，增加节点间通信层
2. **动态核维护**：研究增量/减量coreness更新算法
3. **NUMA感知与外部内存**：针对超大图（>RAM）设计分区和外存算法
4. **异构加速**：探索GPU辅助计算，卸载高密度顶点处理
5. **时态图扩展**：支持时态图的k-core分解（引用作者相关工作[27][32]）

---

**核心贡献总结**：该工作证明了通过**全局共享状态**、**选择性通信**和**动态调度**三者的结合，可将分布式消息传递算法转化为高效的共享内存并行算法，在Rust中实现内存安全与极致性能的统

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
- **单目标优化的局限性**：现有RL后训练方法（如GRPO）仅优化**准确性**，忽视了医疗等高风险领域对**忠实性（faithfulness）**和**全面性（comprehensiveness）**的多维要求
- **推理过程不可验证**：传统方法产生的推理片段化或走捷径，难以被临床医生验证和信任
- **人工标注依赖**：偏好对齐方法需要昂贵的人工反馈，难以规模化

### 提出的新方法
- **Clinical-Objective Relative Policy Optimization (CRPO)**：一种可扩展的、多目标、可验证的RL框架
  - 在GRPO基础上扩展，引入**基于规则的奖励信号**
  - 联合优化三个目标：**准确性 + 忠实性 + 全面性**
  - 强制模型遵循结构化推理格式：`<dx>`（分析过程）和`<conclusion>`（结论）标签分离

### 相比现有方法的优势
- **无需人工标注**：完全依赖可验证的奖励函数
- **临床对齐**：模拟临床医生的双系统思维（System 1直觉 + System 2分析）
- **可解释性**：强制结论必须引用分析部分的证据，生成可审计的推理链
- **训练稳定性**：通过KL散度正则化和一致性奖励抑制离轨探索

---

## 2. 核心实验方法和设置

### 数据集
| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| **MedQA** | 12,723题（英文） | 美国/中国医师执照考试题 | 领域内评估 |
| **MedMCQA** | 194,000+题 | 印度医学院入学考试 | 领域外评估 |
| **MedXpertQA** | 4,460题 | 专家级多专科推理题 | 领域外评估 |

### 实验设置
- **基础模型**：Qwen2.5-3B-Instruct
- **训练框架**：Volcano Engine RL (verl)
- **训练流程**：
  1. **Cold Start**：用DeepSeek-R1蒸馏5,000道MedQA题（13 epochs）
  2. **CRPO/GRPO训练**：在剩余5,000道MedQA题上训练20 epochs
- **采样参数**：Rollout数量 G=5，准确性权重系数 k=10

### 评估指标
**认知行为评估**（LLM-as-judge）：
- Backtracking（回溯纠错）
- Backward-Chaining（反向链推理）
- Subgoal Setting（子目标设定）
- Answer Verification（答案验证）

**医疗忠实性评估**：
- **Faithfulness**：与医学知识的一致性
- **CECD**（Case-grounded Evidence Citation Density）：病例证据引用密度
- **DRC**（Distractor Rejection Coverage）：干扰项拒绝覆盖率
- **Hallucination**：幻觉率（越低越好）

---

## 3. 主要实验结果和性能指标

### 准确率对比（%）
| 方法 | MedQA | MedMCQA | MedXpertQA | 平均提升 |
|------|-------|---------|------------|----------|
| Baseline (CoT) | 41.95 | 46.78 | 10.51 | - |
| GRPO | 50.35 | 49.87 | 12.64 | +4.8 |
| **CRPO** | **52.41** | **55.13** | **14.88** | **+9.0** |
| Cold Start + GRPO | 51.07 | 48.86 | 12.64 | +4.6 |
| **Cold Start + CRPO (Clinical-R1-3B)** | **53.07** | **58.10** | **16.14** | **+11.0** |

### 医疗忠实性关键数据（MedMCQA）
| 指标 | Baseline | GRPO | CRPO | Cold Start+CRPO |
|------|----------|------|------|-----------------|
| Faithfulness | 4.66 | 4.71 | 7.13 | **12.95** |
| CECD | 1.42 | 1.51 | 5.36 | **5.76** |
| DRC | 1.84 | 1.75 | 2.79 | **3.36** |
| Hallucination↓ | 0.40 | 0.85 | 0.64 | **0.66** |

### 认知行为提升（MedQA）
- **Backtracking**：从0.66（Baseline）→ **3.29**（Clinical-R1）
- **Backward-Chaining**：从0.75 → **2.33**
- **Subgoal Setting**：从2.93 → **4.95**
- **Verification**：从0.78 → **2.02**

---

## 4. 关键结论和发现

### 主要发现
1. **多目标优化有效性**：CRPO在保持准确性的同时，忠实性和全面性指标提升**2-3倍**， hallucination率降低
2. **结构化推理的必要性**：强制`<dx>`和`<conclusion>`分离使模型学会区分**临床表现**、**病因暴露**和**干扰项**，避免将疾病标签误认为风险因素
3. **Cold Start的价值**：蒸馏初始化显著增强认知行为能力（如回溯和反向链），为CRPO提供稳定起点
4. **长度模式优化**：CRPO生成响应长度介于Cold Start（过长）和GRPO（过短）之间，实现**简洁且结构完整**的平衡

### 方法局限性
- **训练不稳定性**：无Cold Start时KL散度增长过快，可能导致收敛困难
- **评估偏差**：认知行为和医疗忠实性依赖LLM-as-judge（Llama-3.1-8B/GPT-5），与人类判断可能存在偏差
- **领域知识局限**：基础模型未在医学语料上预训练，知识表示深度有限
- **单模态限制**：当前仅支持文本推理，未扩展到多模态临床数据

### 未来工作方向
1. **算法改进**：探索自适应或离策略CRPO变体提升训练稳定性
2. **人类评估**：引入临床专家进行人工验证
3. **模型增强**：在更强医学专用基础模型（如MEDITRON）上应用CRPO
4. **多模态扩展**：支持临床图像和结构化患者数据
5. **临床部署**：开发可解释AI系统用于实际临床决策支持

---

**核心贡献**：CRPO首次将**可验证的多目标RL**引入医疗推理后训练，为构建**更可信、可协作**的临床AI系统提供了可扩展的技术路径。

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-12-02 06:29:22 UTC
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
# SIMPLE: 解耦采样与GPU推理的决策平面以加速分布式LLM服务

## 1. 主要贡献和创新点

### 解决的问题
- **采样瓶颈**：在分布式LLM推理中，随着GPU计算速度提升和词表规模扩大（从100k增长到200k+），采样（sampling）成为新的性能瓶颈，在迭代时间中占比高达20-38%
- **并行化缺陷**：采样无法随tensor parallelism (TP)扩展，也无法在pipeline parallelism (PP)各阶段间平衡，导致：
  - 在TP下需要跨rank的vocabulary轴collective通信（如AllGather）
  - 在PP下仅在最后阶段执行，造成22-40%的流水线气泡
- **结构性限制**：采样作为串行尾声，其执行时间不受GEMM加速影响，成为Amdahl定律下的性能天花板

### 提出的新方法
**SIMPLE**（Stage-agnostic, Sequence-parallel, Overlappable Decision Plane）将采样从GPU数据平面解耦为独立的CPU端决策服务，包含三项核心技术：

1. **Sequence-parallel sampling**：沿batch维度分片工作负载，每个sampler独立处理部分序列，消除vocabulary轴的collective通信
2. **CPU-based高效算法**：采用column-wise penalties和truncation-first filtering实现单遍线性时间复杂度内核，避免全词表扫描
3. **Speculative Hot-Vocab Sampling (SHVS)**：基于Zipf分布构建小型热词表（hot set），在热词表上执行采样并通过rejection sampling保证分布精确性，将常见情况复杂度从O(V)降至O(H)

### 相比现有方法的优势
- **无需用户代码修改**：作为vLLM的插件式扩展，直接替换决策平面钩子
- **与数据平面优化正交**：可与FlashAttention、PagedAttention等现有优化无缝组合
- **性能可扩展**：随着GPU代际提升（L40→H100→B200），收益持续存在甚至放大
- **分布精确性**：SHVS通过rejection-correctness机制保持与全词表采样完全相同的输出分布

---

## 2. 核心实验方法和设置

### 测试平台
| 平台 | GPU型号 | GPU内存 | 互连 | CPU | 节点配置 |
|------|---------|---------|------|-----|----------|
| L40 | NVIDIA L40 | 48GB | PCIe 4.0 | 128× Intel Xeon Platinum 8358 | 8 GPU + 2TB内存 |
| H100 | NVIDIA H100 | 80GB | NVLink | 256× Intel Xeon 6767P | 8 GPU + 2TB内存 |
| B200 | NVIDIA B200 | 180GB | NVLink | 192× Intel Xeon Platinum 8468 | 8 GPU + 2TB内存 |

### 评估模型与配置
- **模型**：QwQ-32B、Llama-3.1-70B、Qwen-2.5-72B、Qwen3-235B-A22B、DeepSeek V3、Qwen3-Coder-480B-A35B
- **并行度**：TP≤4，PP深度1-4（根据模型大小优化）
- **决策平面**：每个引擎使用16个CPU samplers，每个sampler 4线程

### 数据集与负载
- **数据集**：ShareGPT真实对话数据
- **采样配置**：启用完整生产级控制（temperature, top-k, nucleus top-p, min-p, repetition/presence/frequency penalties）
- **批处理**：默认每GPU batch size B=32（总batch size = 256当p×t=8）

### 评估指标
- **吞吐量**：tokens/s（端到端）
- **延迟**：Time-per-Output-Token (TPOT) P50/P95/P99
- **资源利用率**：GPU/CPU利用率、主机内存占用
- **分布质量**：Total Variation Distance (TVD)
- **消融研究**：各组件对性能的贡献度

### 基线方法
- **vLLM (0.10.1)**：主流GPU驻留采样实现
- **SGLang (0.5.2)**：另一优化推理引擎

---

## 3. 主要实验结果和性能指标

### 端到端吞吐量提升
| 平台 | 平均提升 | 最大提升 | 最佳模型场景 |
|------|----------|----------|--------------|
| L40 | +50% | **+96%** | Qwen3-235B-A22B (TP4-PP4) |
| H100 | +50% | +74% | Qwen3-235B-A22B (TP4-PP4) |
| B200 | +28% | +36% | Qwen3-Coder-480B-A35B (TP4-PP2) |

**vs SGLang**：在H100上Llama-3.1-70B提升达+67%，证明优势不依赖特定基线

### 延迟降低（TPOT P95）
| 平台 | 平均降低 | 最大降低 | 典型场景 |
|------|----------|----------|----------|
| H100 | **-55%** | -65% | Llama-3.1-70B (从~200ms降至~70ms) |
| L40 | -39% | -49% | Qwen3-235B-A22B |
| B200 | -28% | -34% | DeepSeek V3 |

**负载-延迟权衡**：在Qwen3-235B-A22B上，饱和负载下P99从105ms降至63ms（-40%），吞吐量从5326→9421 tok/s（+77%）

### 资源利用率变化
- **GPU利用率**：B200上从75%提升至96%（+21%），L40/H100趋势一致
- **CPU利用率**：适度增加，B200上+17%（均值<31%），L40上+8%，未达饱和
- **主机内存**：增加≤1.3%（Qwen3-235B-A22B在B200上：6.8%→8.1%）

### 消融实验结果
**决策平面吞吐量（per-sampler）**：
- vLLM CPU基线：1.3 tokens/s
- Sequence-parallel (GPU驻留)：6.4 tokens/s (**+4.8×**)
- CPU offloading：11.3 tokens/s (**+8.4×**)
- **SHVS完整方案**：300 tokens/s (**+225×**)

**热词表大小优化**：
- 成本模型预测的H⋆与实际吞吐量峰值位置高度吻合
- 在QwQ-32B上，最优H通常位于25k-50k之间（词表V=152k）

### 分布精确性验证
- **TVD**：所有模型累积TVD < 1%（Llama-3.1-70B平均0.067%）
- **无偏差漂移**：曲线平坦，证明SHVS的rejection-correctness机制保持分布精确

---

## 4. 关键结论和发现

### 核心发现
1. **采样是结构性瓶颈**：其成本不随TP扩展，在PP中造成阶段倾斜，且随着GPU加速和词表增大，瓶颈效应愈发显著（Amdahl定律）
2. **解耦是有效方案**：将采样移至CPU端作为独立服务，可消除流水线气泡，使GPU利用率接近饱和（96%）
3. **投机采样高效且安全**：基于Zipf分布的SHVS在覆盖80-95%概率质量的情况下，将采样速度提升两个数量级，同时通过rejection sampling保持分布精确性
4. **硬件代际兼容性**：性能提升在L40（PCIe）、H100、B200（NVLink）上均有效，且在更快GPU上收益更明显

### 方法局限性
1. **CPU带宽瓶颈**：极高线程数下，内存控制器和LLC竞争会导致per-sampler吞吐量下降（SHVS从32线程的393 tok/s降至128线程的228 tok/s）
2. **热词表覆盖率敏感**：当领域漂移或约束解码导致热词表覆盖质量αb(H)过低时，SHVS接受率下降，收益收窄
3. **计算受限场景**：当GPU数据平面本身为计算瓶颈（非内存瓶颈）时，采样重叠带来的提升有限
4. **部署复杂性**：需要额外的CPU资源规划和共享内存管理，尽管开销很小（<1.3%内存）

### 未来工作方向
1. **在线自适应控制**：基于QoS和运行时统计动态调整H，应对领域漂移和负载变化
2. **拓扑感知部署**：优化NUMA亲和性，减少跨socket内存流量
3. **扩展约束解码**：将SHVS推广到结构化/语法约束解码场景
4. **异构资源调度**：结合prefill-decode分离等disaggregation趋势，实现更细粒度的资源复用

### 实践意义
SIMPLE使采样回归其应有的"不起眼尾声"角色，通过**零用户代码修改**的方式解锁了分布式LLM推理的下一波性能提升，其架构优势将随着未来GPU代际持续放大。

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
- **Structure Gap**：LLM的概率性token生成机制与结构化数据格式（JSON/XML等）的确定性语法要求之间存在根本矛盾
- SFT的局限性：缺乏显式惩罚机制，导致"近似正确"的输出（如缺少括号、幻觉键名）
- 约束解码的缺陷：推理延迟高（最高6倍），且无法改善模型的内部表示

### 核心创新
1. **多维奖励函数（Multi-dimensional Reward Function）**
   - 将结构化输出任务分解为五个层次化约束：
     - `R_struct`：结构完整性（强制必需键）
     - `R_format`：格式合规性（Markdown/JSON风格）
     - `R_valid`：语法有效性（可解析性）
     - `R_correct`：语义正确性（F1分数）
     - `R_length`：长度约束
   - 通过权重分配实现课程学习：优先保证语法正确性，再优化语义内容

2. **GRPO优化算法**
   - 采用**Gradient Regularized Policy Optimization**，无需独立critic网络
   - 通过group sampling估计baseline，显著降低内存占用
   - 结合LoRA实现参数高效微调

### 相比现有方法的优势
- **效率**：相比PPO减少40% VRAM占用（14.2GB vs 22.8GB），训练吞吐量提升62%（42 vs 26 samples/min）
- **零推理延迟**：训练时干预，推理速度等同于标准生成
- **样本效率**：仅需1,000样本即可达到>80%结构准确率
- **可靠性**：在4B参数模型上实现89.7%结构准确率，超越更大模型

---

## 2. 核心实验方法和设置

### 数据集
- **主任务**：食谱生成（AkashPS11/recipes_data_food.com）
  - 输出包含ingredients、steps、nutrition等字段的嵌套JSON
- **泛化验证**：
  - GSM8K-JSON：数学推理任务，要求分离推理步骤和最终答案
  - ToolUse：函数调用任务，生成API参数JSON

### 实验设置
- **基础模型**：Qwen3-4B-Instruct
- **PEFT配置**：LoRA（rank=32, alpha=32），4-bit量化
- **训练参数**：250 steps，lr=5×10⁻⁶，cosine decay，group size G=6
- **硬件**：单张NVIDIA RTX 4090（24GB）

### 评估指标
| 指标 | 定义 |
|------|------|
| **Structural Accuracy** | 整体语法正确率 |
| **JSON Validity** | 严格JSON解析成功率 |
| **Format Consistency** | 风格一致性（Markdown等） |
| **Schema Compliance** | 必需键召回率 |
| **Content Accuracy** | F1分数 + GPT-4 Judge评分的归一化聚合 |

### 基线方法
- **Zero-shot**：GPT-3.5-Turbo, Mistral-7B-Instruct
- **SFT**：Phi-3-mini (3.8B), LLaMA-3-8B, Qwen3-4B
- **约束解码**：Qwen3-4B + Outlines
- **偏好对齐**：Qwen3-4B + DPO（合成valid/invalid JSON对）

---

## 3. 主要实验结果和性能指标

### 核心性能对比（食谱生成任务）

| 方法 | Structural Acc. | JSON Validity | Content Acc. |
|------|-----------------|---------------|--------------|
| GPT-3.5 (Zero-shot) | 45.5% | 82.1% | 88.0% |
| Mistral-7B (Zero-shot) | 52.3% | 83.5% | 85.0% |
| Phi-3-mini (SFT) | 74.1% | 84.8% | 81.5% |
| LLaMA-3-8B (SFT) | 78.2% | 85.4% | 86.0% |
| Qwen3-4B (SFT) | 65.4% | 72.1% | 80.0% |
| **RL-Struct (Ours)** | **89.7%** | **92.1%** | **84.5%** |
| Qwen3-4B + Outlines | 99.8% | 100.0% | 79.5% |
| Qwen3-4B + DPO | 82.5% | 88.4% | 82.0% |

**关键发现**：
- **模型规模非关键**：4B参数的RL-Struct超越8B参数的LLaMA-3 SFT（+11.5%结构准确率）
- **超越偏好学习**：GRPO dense reward比DPO pairwise objective更有效（+7.2%）
- **效率-质量权衡**：Outlines虽达99.8%准确率，但推理延迟高6倍

### 资源效率对比
- **VRAM占用**：GRPO 14.2GB vs PPO 22.8GB（-40%）
- **训练吞吐量**：GRPO 42 samples/min vs PPO 26 samples/min（+62%）
- **样本效率**：1,000样本达到80%+准确率，SFT需数倍数据

### 泛化能力验证
| 任务 | Zero-shot | SFT | RL-Struct (Ours) |
|------|-----------|-----|------------------|
| GSM8K-JSON | 25.5% | 58.2% | **85.4%** |
| ToolUse | 15.2% | 70.1% | **91.2%** |

### 消融实验结果

| 配置 | JSON Validity | Structural Acc. | Content Acc. |
|------|---------------|-----------------|--------------|
| **Full Reward** | **92.1%** | **89.7%** | **84.5%** |
| w/o R_valid | 68.3% (-23.8) | 85.2% | 83.0% |
| w/o R_struct | 90.5% | 45.6% (-44.1) | 84.0% |
| w/o R_format | 88.2% | 87.1% | 84.0% |

**结论**：R_valid对语法正确性至关重要，R_struct确保模式完整性

---

## 4. 关键结论和发现

### 主要发现

1. **自步课程学习（Emergent Curriculum）**
   - **Phase 1**（0-100 steps）：快速习得语法（R_valid迅速上升）
   - **Phase 2**（100-250 steps）：稳定后优化语义（R_correct逐步提升）
   - 无需手动设计课程，GRPO的group-based优化自然实现

2. **梯度主导效应（Gradient Dominance）**
   - 当R_valid≈0时，∇(w_valid·R_valid) ≫ ∇(w_correct·R_correct)
   - 优化轨迹优先投影到有效语法流形，再细化语义

3. **错误模式抑制**
   - 几乎消除语法错误（vs SFT的10-15%语法错误率）
   - 显著减少幻觉键名（Hallucination）和类型不匹配

### 方法局限性

1. **奖励工程成本**：需手动设计reward组件，对动态schema需重新训练
2. **泛化瓶颈**：固定schema训练难以适应任意API调用等高度动态场景
3. **对抗鲁棒性**：对"jailbreak"提示仍偶尔失效（但比SFT更鲁棒）
4. **序列长度**：为严格满足schema可能生成略长的输出

### 未来工作方向

1. **格式无关框架**：扩展至XML/HTML/SQL，开发统一元语法奖励
2. **自动奖励合成**：从JSON Schema/XSD自动生成验证逻辑作为reward
3. **混合策略**：结合轻量级约束束搜索处理动态schema
4. **LLM-as-a-Judge**：用强模型提供dense奖励信号替代启发式规则
5. **自适应权重**：基于奖励方差动态调整w_valid vs w_correct
6. **多轮Agent工作流**：应用于更复杂的agentic场景

### 长期影响

- 弥合概率AI与确定性软件工程之间的鸿沟
- 为边缘计算和隐私保护本地AI应用提供轻量级可靠方案
- 使小型模型具备与大型专有模型竞争的结构化输出能力

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
论文针对**长上下文推理**场景下的KV cache瓶颈问题，指出现有KV cache优化方法存在的三大挑战：
1. **逐层检索开销大**：检索操作引入的同步开销随模型深度线性增长，导致高达60%的延迟开销
2. **新生成KV cache完全保留**：现有方法仅预处理输入提示的KV cache，在解码阶段完全保留新生成的KV对，在长上下文推理中效果不佳
3. **序列长度微小增加导致性能急剧下降**：预设的卸载策略无法适应推理过程中动态增长的序列长度，导致>80%的性能下降

### 提出的新方法
**SpeContext**：基于**Speculative Context Sparsity**的算法与系统协同设计框架，核心创新包括：

- **算法层面（C1）**：提出基于**Distilled Language Model (DLM)**的轻量级检索头
  - 利用DLM与原始LLM在信息焦点上的相似性（通过信息论中的互信息和数据处理不等式分析）
  - 基于头级（head-level）注意力权重进行token选择，剪枝冗余操作，实现>90%参数减少
  - 支持MHA、GQA、MQA、MLA四种主流注意力机制

- **系统层面（C2）**：设计**异步预取数据流**与**弹性加载策略**
  - 消除KV检索与LLM计算之间的数据依赖，通过多CUDA流实现计算与数据传输重叠
  - 利用相邻token生成间>80%的重要token重叠率，仅加载差异部分（约20%更新量）

- **编译层面（C3）**：构建**自适应内存管理系统**
  - 建立理论内存开销模型，考虑LLM架构、硬件规格和工作负载
  - 根据序列长度动态调整KV cache在GPU HBM和CPU DRAM间的分布，最大化GPU内存利用率

### 相比现有方法的优势
- **性能突破**：云端吞吐量提升最高达**24.89×**，边缘端加速达**10.06×**，准确率损失可忽略
- **推理效率**：消除逐层检索的同步开销，避免复杂预处理，支持动态增长的序列长度
- **资源适应性**：在资源受限环境（低显存GPU、多请求并发）下仍保持高效
- **帕累托前沿**：推动长上下文输入与推理场景中准确率-吞吐量的帕累托前沿

---

## 2. 核心实验方法和设置

### 硬件平台
- **云端（高显存多请求）**：NVIDIA A100-80GB GPU + Intel Xeon Platinum 8358 CPU + 1008GB DRAM
- **边缘端（低显存）**：NVIDIA RTX 4060 Laptop GPU (8GB) + Intel i7-13650HX CPU + 24GB DRAM

### 评估模型
- **云端**：Llama3.1-8B、DeepSeek-R1-Distill-Llama-8B、Qwen3-8B
- **边缘端**：Reasoning-Llama-3.2-1B

### 数据集与基准
- **长上下文输入场景**：LongBench的4个任务（2WiKiMQA、TriviaQA、HotpotQA、Passage count）
- **长上下文推理场景**：LongWriter基准（生成10000+词）
- **评估指标**：使用GPT-4o从**相关性、准确性、连贯性、清晰度、广度和深度、阅读体验**六个维度评分

### 基线方法
- **全注意力**：Huggingface (Eager)、FlashInfer
- **稀疏注意力**：Quest、ClusterKV、ShadowKV

### 实验配置
- **KV cache budget**：512、1024、2048、4096
- **输入/输出长度组合**：[2k, 16k]、[2k, 32k]、[16k, 2k]、[32k, 2k]
- **批量大小**：根据显存动态调整（4-64个请求）

---

## 3. 主要实验结果和性能指标

### 准确率表现
**长上下文输入场景（LongBench）**
- **Budget=512**：SpeContext准确率略低于ClusterKV（因仅做全局token选择）
- **Budget≥1024**：SpeContext**超越所有基线**，达到与全注意力相当的准确率
- **Budget=4096**：在HotpotQA上达到95%以上的全注意力准确率

**长上下文推理场景（LongWriter）**
- Quest/ClusterKV/ShadowKV因仅预处理输入，不同budget输出相同，评分接近全注意力但**吞吐量极低**
- SpeContext在**budget=2048**时，平均评分2.86（vs 全注意力2.84），**几乎无损失**
- 在budget=4096时，平均评分达2.95，**超越全注意力**

### 吞吐量与加速比
**云端多请求场景（A100-80GB）**
| 模型 | 配置 | 吞吐量(tokens/s) | 加速比(vs Eager) | 加速比(vs FlashInfer) |
|------|------|------------------|------------------|----------------------|
| DeepSeek-Distill-Llama-8B | [2k, 32k] | 690.59 (32请求) | **24.89×** | 2.20× |
| Llama3.1-8B | [2k, 16k] | 824.22 (32请求) | **18.09×** | 1.68× |
| Qwen3-8B | [2k, 32k] | 424.92 (32请求) | **22.03×** | 1.67× |

**边缘端单请求场景（RTX 4060 8GB，限制4GB显存）**
- DeepSeek-Distill-Llama-8B [2k, 16k]：**10.06×** vs Full Attn (Eager)，**1.17×** vs ShadowKV
- 在[32k, 2k]配置下，全注意力OOM，SpeContext仍保持346.88 tokens/s

### 消融实验结果
**DeepSeek-Distill-Llama-8B [2k, 32k]配置**
- **C1（轻量级检索头）**：12.07×加速（主要收益来自FlashInfer后端和稀疏注意力）
- **C1+C2（+异步预取+弹性加载）**：18.09×加速（减少数据传输瓶颈）
- **C1+C2+C3（+自适应内存管理）**：**24.89×**加速（最大化GPU内存利用率）

---

## 4. 关键结论和发现

### 主要发现
1. **DLM-LLM信息焦点相似性**：通过信息论分析证实，蒸馏模型与原始模型在上下文信息焦点上高度相似，为使用DLM作为检索算法提供理论基础
2. **头级检索优于批级**：头级（head-level）注意力权重在重要token选择上的相似度和命中率显著高于批级（batch-level）
3. **上下文相似性利用**：相邻token生成间的重要token重叠率>80%，弹性加载策略可减少高达80%的数据传输量
4. **动态内存管理必要性**：静态卸载策略无法适应推理中动态增长的序列长度，自适应管理可维持性能稳定

### 方法局限性
- **依赖DLM质量**：检索效果受限于蒸馏模型的对齐程度，需额外训练DLM（EAGLE-3需24小时RTX 3090训练）
- **小budget准确率**：在极小的KV cache budget（如512）下，全局选择策略准确率略低于逐层选择的ClusterKV
- **检索头开销**：轻量级检索头仍引入约20%额外计算开销，尽管通过异步预取被大幅掩盖

### 未来工作方向
- **信息论视角扩展**：论文指出该方法论可推广到考虑信息流的机器学习架构与系统设计研究
- **更激进的稀疏性**：探索更细粒度的稀疏模式（如token级、channel级）
- **多模态应用**：将Speculative Context Sparsity思想扩展到视觉-语言模型
- **硬件协同设计**：针对弹性加载和异步预取优化专用硬件架构

---

**核心总结**：SpeContext通过**算法-系统-编译**三层协同设计，首次将DLM用于KV cache检索，结合异步预取和自适应内存管理，在长上下文推理场景中实现了数量级的性能提升，同时保持准确率几乎无损，为资源受限环境下的LLM部署提供了高效解决方案。

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
- **可合成性瓶颈**：现有生成模型提出的分子难以实验合成，合成可行性低（成功率约85%）
- **化学空间覆盖不足**：SOTA方法对Enamine REAL空间的重建率仅~70%
- **两阶段不一致性**：传统"生成-投影"框架导致分子图与可合成类似物在结构和功能上存在差异
- **效率低下**：推理速度慢，采样效率低，难以满足大规模应用需求

### 提出的新方法：PrexSyn框架
- **统一架构**：基于decoder-only transformer，直接学习**属性→合成路径**的映射，消除两阶段不一致性
- **后缀表示法**：使用合成路径的后缀表示（postfix notation）作为分子表示，天然保证可合成性
- **高通量数据引擎**：C++实现的多线程数据生成管道，支持**十亿级**训练数据实时生成（>6,000样本/秒）
- **可编程查询**：支持通过逻辑运算符（AND, NOT, OR）组合多个属性条件，实现"程序化"分子设计
- **查询空间优化**：将黑盒oracle优化转化为数值属性空间的迭代查询优化，避免在离散稀疏的合成树空间搜索

### 相比现有方法的优势
- **覆盖率SOTA**：Enamine REAL空间重建率达**94.06%**，接近完美覆盖
- **速度提升60倍**：推理时间仅0.26秒/目标，比最高精度基线快60倍以上
- **采样效率更高**：在GuacaMol基准上6/8任务取得最佳AUC-Top10，超越合成无关方法
- **训练成本低**：2×H100 GPU + 32 CPU核心，2天完成13亿样本训练，成本低于同类方法
- **可合成性保证**：所有生成分子均基于vendor-validated building blocks和反应模板

---

## 2. 核心实验方法和设置

### 数据集
- **Building blocks**：Enamine US库存building blocks（~48万）和WuXi（~9万）
- **反应模板**：115个Gao et al. (2025)精选模板
- **测试集**：
  - Enamine testset：1,000个来自Enamine REAL数据库的分子
  - ChEMBL testset：1,000个ChEMBL分子（不一定可合成）
- **基准测试**：GuacaMol的7个多属性目标+1个再发现任务
- **对接任务**：sEH（可溶性环氧化物水解酶）和Mpro2（SARS-CoV-2主蛋白酶）

### 模型配置
- **架构**：12层transformer，d_model=1024，feedforward=2048，16个注意力头
- **参数量**：主干网络1.31亿，总计（含embedding）5.89亿
- **训练**：Adam优化器，lr=3×10⁻⁴，cosine退火，batch size=2048
- **训练数据**：13亿样本，覆盖ECFP4指纹、FCFP4、BRICS子结构、RDKit物化描述符等

### 评估指标
- **重建率**：Tanimoto相似度=1的分子比例
- **相似度**：ECFP4指纹Tanimoto相似度
- **推理时间**：单目标平均生成时间
- **采样效率**：AUC-Top10（10,000 oracle calls预算）
- **药物相似性**：QED（Quantitative Estimate of Drug-likeness）
- **合成可及性**：SA（Synthetic Accessibility）分数
- **多样性**：1 - 平均成对Tanimoto相似度

### 基线方法
- **合成路径搜索**：SynNet, ChemProjector, SynthesisNet, SynLlama, SynFormer, ReaSyn
- **合成无关方法**：REINVENT, GraphGA, MolGA
- **混合方法**：DoG-Gen, SyntheMol, SynFlowNet

---

## 3. 主要实验结果和性能指标

### 3.1 化学空间投影任务

| 方法 | Enamine重建率 | Enamine相似度 | ChEMBL重建率 | ChEMBL相似度 | 推理时间 |
|------|---------------|---------------|--------------|--------------|----------|
| SynFormer | 66.10% | 0.9137 | 20.67% | 0.6737 | 3.45s |
| ReaSyn | 74.93% | 0.9403 | 22.07% | 0.6740 | 19.71s |
| **PrexSyn (256 samples)** | **94.06%** | **0.9859** | **28.32%** | **0.7533** | **0.26s** |

**关键发现**：
- 重建率比SOTA提升**19个百分点**（94.06% vs 74.93%）
- 推理速度比ReaSyn快**76倍**，比SynFormer快**13倍**
- 在ChEMBL上同样达到SOTA，证明对未知化学空间的泛化能力

### 3.2 GuacaMol黑盒优化基准

| 方法 | 可合成性 | 平均AUC-Top10 | 最优任务数 |
|------|----------|---------------|------------|
| MolGA | ✗ | 0.682 | 2/8 |
| SynFormer | ✓ | 0.646 | 1/8 |
| ReaSyn | ✓ | 0.702 | 1/8 |
| **PrexSyn** | **✓** | **0.758** | **6/8** |

**关键发现**：
- 在**Celecoxib再发现**任务中成功重建目标分子并发现高评分类似物
- 在**sEH对接**任务中，PrexSyn得分**1.01**（vs SynFlowNet 0.94），QED达**0.80**
- 查询空间优化比直接搜索合成树空间效率更高，优化景观更平滑

### 3.3 复合属性查询生成

| 任务 | 描述 | 平均得分 | Top 5%得分 | Top 10%多样性 |
|------|------|----------|------------|---------------|
| Task 1 | Lipinski's Rule of Five | 0.9549 | 1.0000 | 0.8902 |
| Task 2 | Cobimetinib类似物优化 | 0.7108 | 0.8975 | 0.6017 |
| Task 3 | Osimertinib MPO优化 | 0.4971 | 0.8314 | 0.7499 |

**关键发现**：
- **100%**的Top 5%分子完美满足Lipinski五规则
- 复合查询能有效缩小搜索空间，在**scaffold hopping**任务中AUC提升**0.154**

### 3.4 消融实验
- **数据规模影响**：重建率随训练数据量单调递增，13亿样本时趋于饱和
- **样本数影响**：从64→256 samples，重建率从92.64%→94.06%，时间仅增加0.16s
- **C++引擎加速**：数据生成吞吐量>6,000样本/秒，内存占用比Python实现显著降低

---

## 4. 关键结论和发现

### 主要发现
1. **规模效应显著**：十亿级训练数据使模型能近乎完美重建可合成化学空间，学习到有意义的building block embedding
2. **表示法优势**：后缀表示法+分类器选择building blocks比指纹检索更灵活，支持上下文感知的动态选择
3. **查询空间优化的威力**：将离散分子优化转化为连续属性空间优化，效率超越合成无关方法
4. **可编程性的价值**：逻辑组合查询支持复杂药物设计场景（如scaffold hopping、多目标优化）

### 方法局限性
- **条件独立性假设**：复合查询基于属性条件独立假设，可能产生矛盾约束（如MW<100且重原子数>50）
- **模板依赖性**：受限于预定义反应模板和building block库，对全新化学反应泛化能力有限
- **属性相关性**：未显式建模属性间相关性，可能影响某些复杂查询的准确性

### 未来工作方向
- **扩展化学空间**：整合更多vendor的building blocks和反应模板
- **增强表示能力**：引入3D结构信息、量子化学属性等更丰富特征
- **自适应学习**：在线学习新反应类型，减少对固定模板的依赖
- **实验验证**：与自动化合成平台集成，验证生成分子的实际可合成性
- **应用拓展**：针对特定疾病靶点（如抗生素、抗癌药物）进行大规模虚拟筛选

---

**代码和模型**：https://github.com/luost26/PrexSyn

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
# 论文总结：基于 rLoRA 微调的 Qwen3-8B 金融文本分类模型

## 1. 主要贡献和创新点

### 解决的问题
论文针对金融领域的两个核心文本分类任务：**金融情感分析**（三分类：正面/负面/中性）和**金融新闻主题分类**（20 类），探索如何高效利用大语言模型（LLM）提升分类准确率和训练效率。

### 提出的新方法
提出了一个**集成化的高效微调框架**，将以下技术有机结合：
- **Qwen3-8B 作为基础模型**：利用其 32K+ 上下文窗口、GQA 架构和 thinking/non-thinking 双模式特性
- **rLoRA（Rank-stabilized Low-Rank Adaptation）**：改进的 LoRA 方法，通过 √r 缩放因子稳定高秩适配器的训练过程
- **NEFTune（Noisy Embedding Instruction Finetuning）**：在嵌入层注入可控噪声（α=0.3）以增强鲁棒性和泛化能力
- **FlashAttention**：加速注意力计算并降低 GPU 内存占用

### 相比现有方法的优势
1. **性能优势**：在金融主题分类任务上准确率显著超越所有基线（93.15% vs 最佳基线 88.77%）
2. **效率优势**：仅需 **3 个 epoch** 即可收敛，而传统非 LLM 方法需 10+ epoch
3. **参数效率**：通过 rLoRA 仅训练低秩适配器，冻结原模型权重，大幅降低计算资源需求
4. **推理优化**：利用 Qwen3 的 `\no_think` 指令禁用思考模式，减少 token 消耗和延迟

---

## 2. 核心实验方法和设置

### 数据集
| 数据集 | 任务类型 | 样本分布 | 规模 |
| :--- | :--- | :--- | :--- |
| 金融情感数据集 | 三分类（正/负/中性） | 2,879 中性 / 1,362 正面 / 604 负面 | 4,845 样本 |
| Twitter 金融新闻数据集 | 20 类主题分类 | 训练集 16,990 / 测试集 4,117 | 21,107 样本 |

### 实验设置
- **训练配置**：
  - Batch size: 3（梯度累积 4 步，等效 batch size 12）
  - Learning rate: 5e-5，Optimizer: Adam
  - Max token length: 360
  - Epochs: 3
  - Precision: bfloat16/float16 混合精度

- **rLoRA 参数**：
  - LoRA rank: 8
  - LoRA dropout: 0.1
  - 缩放因子：√r（秩稳定机制）

- **NEFTune**：噪声尺度 α = 0.3

### 评估指标
- **主要指标**：准确率（Accuracy）
- **对比方法**：
  - **非 LLM 基线**：BERT, RoBERTa
  - **LLM 基线**：LLaMA-7B, LLaMA2-7B, Baichuan2-7B

---

## 3. 主要实验结果和性能指标

### 关键性能数据（表 4）
| 模型 | 模型类型 | 金融情感分类 (ACC) | 金融主题分类 (ACC) |
| :--- | :--- | :--- | :--- |
| **Qwen3-8B (rLoRA)** | LLM | **0.8415** | **0.9315** |
| LLaMA2-7B | LLM | 0.8322 | 0.8877 |
| LLaMA-7B | LLM | 0.8297 | 0.8801 |
| Baichuan2-7B | LLM | 0.8165 | 0.8784 |
| RoBERTa | 非 LLM | 0.7928 | 0.8612 |
| BERT | 非 LLM | 0.7854 | 0.8523 |

### 与基线方法的对比结果
1. **情感分类**：Qwen3-8B 达到 84.15%，比最佳 LLM 基线（LLaMA2-7B）提升 **0.93 个百分点**，比最佳非 LLM 基线（RoBERTa）提升 **4.87 个百分点**
2. **主题分类**：Qwen3-8B 达到 93.15%，比最佳基线（LLaMA2-7B）显著高出 **4.38 个百分点**，展现了在复杂多分类任务上的强大优势
3. **收敛速度**：训练损失曲线（图 5）显示，模型在 **3 个 epoch 内快速收敛**，损失值稳定下降，验证了训练效率

### 消融实验结果
论文未明确报告独立的消融实验，但通过集成方法的有效性可推断：
- **rLoRA** 的秩稳定机制支持使用更高 rank 而不损失稳定性
- **NEFTune** 的噪声注入有效缓解过拟合，提升指令遵循能力
- **FlashAttention** 使长序列（360 tokens）训练可行且高效

---

## 4. 关键结论和发现

### 主要发现
1. **Qwen3-8B 是金融 NLP 的强大基础模型**：在两项任务上均实现 SOTA 性能，尤其在主题分类上优势显著，证明了其长上下文理解和指令遵循能力
2. **高效微调框架的有效性**：rLoRA + NEFTune + FlashAttention 的组合在 **保持参数效率的同时提升了鲁棒性和收敛速度**，适合资源受限场景
3. **指令模板的工程价值**：通过 `\no_think` 标签显式控制模型推理模式，在分类任务中减少 30-50% token 消耗，降低延迟

### 方法局限性
1. **数据集规模**：实验数据集相对较小（情感数据仅 4,845 样本），未充分验证在超大规模数据集上的扩展性
2. **模型规模限制**：仅测试了 8B 参数版本，未探索 Qwen3 系列更大模型（如 32B、72B）的潜在提升
3. **评估维度单一**：主要关注准确率，未深入分析 F1、Precision/Recall 等指标，也未评估推理速度和内存占用的绝对数值

### 未来工作方向
1. **实时系统部署**：探索模型在量化交易系统中的实际部署，优化推理引擎和批处理策略
2. **多模态扩展**：结合金融时序数据、图表等非文本信息，构建多模态分类框架
3. **更大规模验证**：在百万级金融文档数据集上验证方法的可扩展性和稳定性
4. **风险分析应用**：将框架扩展至金融事件检测、风险预警等更复杂的决策支持任务

---

**总结**：该论文通过精巧的技术集成，证明了 Qwen3-8B 在金融文本分类任务上的卓越性能，为构建**高效、经济、可扩展**的金融 NLP 应用提供了实用范式。

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
# 论文核心结论和实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
- **计算成本瓶颈**：冰盖模拟涉及大规模非线性PDE系统，传统数值求解器单次模拟耗时数小时至数天，不确定性量化(UQ)需要数百到数千次评估，计算不可行
- **现有ML代理模型局限性**：DeepONet等算子学习方法需要为每个新网格重新训练，缺乏灵活性和泛化能力；标准GNN存在**oversmoothing**问题，难以捕捉冰厚、基底摩擦和速度场的尖锐梯度

### 提出的新方法
1. **物理启发的Bracket-based GNN架构**：将消息传递重构为Hamiltonian动力系统，通过skew-adjoint算子保证能量守恒，从根本上避免特征过平滑
2. **域分解(Domain Decomposition, DD)框架**：将非结构化网格划分为子域，并行训练局部GNN代理模型，推理时聚合预测
3. **迁移学习策略**：在子域上预训练，将学习到的注意力机制和权重迁移到全域或其他子域，实现"warm start"

### 相比现有方法的优势
- **几何灵活性**：直接在Delaunay三角剖分上操作，无需重网格化或插值，支持动态演化域
- **计算可扩展性**：DD策略降低内存需求，支持并行训练，训练时间比单一全局模型显著减少
- **数据效率**：迁移学习在数据稀缺场景(5-10个样本)下加速收敛并提升精度
- **模型可迁移性**：同一模型可在不同网格、子域间共享参数，避免重复训练

---

## 2. 核心实验方法和设置

### 数据集
- **物理模型**：MPAS-Albany Land Ice (MALI)求解Blatter-Pattyn近似方程
- **研究区域**：格陵兰Humboldt冰川（约200万平方公里）
- **网格规模**：18,544个节点，54,962条边（图1）
- **训练数据生成**：
  - 从基底摩擦场μ的后验分布采样（Laplace近似，PDE先验精度算子）
  - 模拟时段：2007-2100年，每年一个快照
  - 25个基底摩擦场实现 × 40个时间步 = **1,000个训练样本**
  - 验证集：20个实现；测试集：20个实现

### 实验设置
- **节点特征**（7维）：冰厚度、床形地形、基底摩擦、grounded/floating ice布尔指示器
- **输出**：速度场u的x和y分量
- **归一化**：节点特征z-score归一化，边距离min-max归一化
- **训练配置**：
  - 优化器：Adam，学习率1e-3，scheduler gamma=0.95
  - 损失函数：MSE（重点惩罚末端区域大误差）
  - 硬件：NVIDIA A100 40GB GPU集群
  - 训练时间：约30小时（离线）

### 评估指标
- **相对测试误差**：$\|u_{pred} - u_{true}\| / \|u_{true}\|$
- **定性评估**：速度场可视化、末端区域误差分布
- **UQ评估**：grounding line质量通量分布直方图

### 基线方法对比
1. **Cold start**：从头训练单一全局GNN模型
2. **Warm start**：在子图（南部内部或东北部末端）预训练后全局微调
3. **DD + Warm start**：将全域划分为3个子域（图4），分别预训练和微调

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 训练策略 | 收敛速度 | 相对误差 | 末端区域误差 |
|---------|---------|---------|------------|
| Cold start (25 fields) | 基准 | 基准 | 高误差 |
| Warm start (5 fields) | **加速3-5倍** | **降低30-50%** | 显著改善 |
| DD + Warm start (terminus) | **最快** | **最低** | **几乎消除** |

### 与基线方法的对比结果
- **数据稀缺场景**：仅5个摩擦场时，warm start误差比cold start低**约一个数量级**（图6）
- **固定训练轮次**：在相同epoch下，warm start和DD+warm start均显著优于cold start
- **固定数据量**：相同样本数下，迁移学习策略达到更低误差（图6实线vs虚线）
- **预训练区域影响**：在**末端区域**预训练比在内部区域预训练收敛更快、误差更低（图7），因为末端速度变异性最大，传递了更丰富的局部结构

### 消融实验结果
- **分区策略**：k=3的非重叠分区已足够，未观察到边界不连续或误差增长，无需重叠分区或粗水平校正
- **注意力机制**：基于bracket的注意力Laplacian同时编码拓扑和度量信息，比标准GNN更稳定
- **架构鲁棒性**：注意力头数(2→4)和隐藏维度(20→50)变化对结果影响小，但编码器/解码器宽度缩小(32→16)会使误差翻倍

---

## 4. 关键结论和发现

### 主要发现
1. **DD+迁移学习协同效应**：域分解不仅降低计算成本，还为知识迁移提供自然单元。在末端高变异性区域预训练，能将局部结构迁移到全域，实现**计算效率与精度的双重提升**
2. **物理启发架构的有效性**：Hamiltonian bracket机制通过能量守恒约束稳定信息传播，在深层网络中保持空间细节，特别适合冰盖这种具有尖锐梯度的系统
3. **数据驱动模型的鲁棒性**：与经典DD求解器不同，GNN代理无需在子域间传输PDE零空间信息，非重叠分区即可避免边界伪影，简化了实现
4. **基础模型潜力**：预训练模型能准确预测平均行为，但会低估不确定性分布的方差；通过在强调输入变异性的数据集上微调，可恢复完整的后验分布（图8），展现出**foundation model**特性

### 方法的局限性
- **UQ能力有限**：基线模型虽能预测均值，但对基底摩擦不确定性导致的输出分布捕捉不足（图8左）
- **分区策略启发式**：当前采用谱聚类+尺寸惩罚k-means，最优分区理论尚未充分探索
- **物理约束隐式**：未在损失函数中显式加入质量守恒等物理残差项，依赖架构的归纳偏置
- **训练成本**：离线训练仍需约30小时，尽管推理仅需15ms/样本

### 未来工作方向
1. **UQ导向的微调**：设计专门强调输入空间变异性的训练集和目标函数，使代理模型能准确传播不确定性
2. **Foundation model范式**：构建大规模预训练模型，通过轻量级微调适配不同冰盖、网格或科学问题
3. **最优分区理论**：研究基于物理特征和误差分布的自适应分区算法
4. **显式物理约束**：探索将守恒定律嵌入损失函数或架构，提升长期稳定性
5. **多保真度融合**：结合低精度模拟数据进一步加速训练，推动冰盖UQ研究进入实用阶段

---

**核心结论**：该工作证明**图神经网络+域分解+迁移学习**是构建大规模PDE系统代理模型的可扩展、可靠路径，在冰盖模拟中实现了高精度、高效率的预测，为不确定性量化等决策任务奠定了坚实基础。

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
# Efficient Training of Diffusion Mixture-of-Experts Models: A Practical Recipe 论文总结

## 1. 主要贡献和创新点

### 解决的问题
现有Diffusion MoE研究过度关注复杂路由机制的设计，而**MoE架构配置空间未被充分探索**，这是导致扩散模型MoE性能不及LLM MoE的根本原因，而非视觉token与文本token的差异。

### 提出的新方法
系统性地借鉴LLM MoE设计范式，提出一套实用的扩散模型MoE架构设计原则：
- **DeepSeek-style专家模块**：采用细粒度专家+共享专家结构
- **缩小的FFN中间宽度**：将缩放因子从传统的4.0降至0.3-1.0
- **增加专家数量**：从16个增至48个，同时保持激活参数不变
- **2D RoPE位置编码**：替代绝对位置编码，增强空间结构感知

### 相比现有方法的优势
- **性能超越路由创新**：架构优化带来的增益超过单纯改进路由策略
- **参数效率更高**：在相同或更少激活参数下，总参数减少30-50%
- **训练效率提升**：收敛速度更快，训练损失更低
- **通用性强**：同时适用于latent diffusion（DSMoE）和pixel-space diffusion（JiTMoE）框架

---

## 2. 核心实验方法和设置

### 数据集
- **ImageNet** 256×256分辨率，1,281,167张训练图像
- 类别条件生成任务

### 实验设置
- **训练框架**：Rectified Flow（Flow Matching）
- **模型规模**：100M至3B参数（S/B/L/XL/3B）
- **训练步数**：700K steps（latent diffusion），200 epochs（pixel-space）
- **Batch size**：512（latent），1024（pixel-space）
- **优化器**：AdamW，学习率1e-4（latent），2e-4（pixel-space）

### 评估指标
- **FID50K**（Fréchet Inception Distance，越低越好）
- **IS**（Inception Score，越高越好）
- 生成50,000张图像进行评估
- CFG（Classifier-Free Guidance）scale：1.0和1.5

### 基线方法对比
- **DiffMoE**：当前SOTA latent diffusion MoE
- **ProMoE**：基于DiT的MoE方法
- **Dense-DiT**：标准密集DiT模型
- **JiT**：pixel-space密集扩散模型

---

## 3. 主要实验结果和性能指标

### 关键性能数据（latent diffusion）

| 模型 | 激活参数 | 总参数 | FID (CFG=1.0) | FID (CFG=1.5) | IS (CFG=1.5) |
|------|----------|--------|---------------|---------------|--------------|
| **DSMoE-3B-E16** | 965M | 2.958B | **7.52** | **2.38** | **304.93** |
| DSMoE-L-E48 | 436M | 1.112B | 9.19 | 2.55 | 278.35 |
| DiffMoE-L-E16 | 458M | 1.982B | 11.16 | 2.84 | 256.57 |
| DSMoE-B-E48 | 118M | 263M | 19.46 | 4.27 | 191.03 |
| DiffMoE-B-E16 | 130M | 555M | 20.83 | 4.87 | 183.43 |

**核心突破**：DSMoE-3B-E16以**700K训练步数**达到FID 2.38，匹配DiffMoE的2.30（需**7000K步**），训练成本降低**10倍**。

### 与基线对比结果
- **vs DiffMoE**：DSMoE-L-E48在总参数减少44%情况下，FID从11.16降至9.19（CFG=1.0）
- **vs Dense-DiT**：DSMoE-B-E16激活参数相近，FID从30.61降至22.46（CFG=1.0）
- **vs ProMoE**：DSMoE-L-E48在更少参数下，FID从11.61降至9.87（CFG=1.0）

### 消融实验结果

#### 位置编码影响（DSMoE-S-E16）
| 方法 | FID (CFG=1.0) | IS (CFG=1.0) | FID (CFG=1.5) | IS (CFG=1.5) |
|------|---------------|--------------|---------------|--------------|
| APE | 45.13 | 33.27 | 18.10 | 82.37 |
| 1D RoPE | 44.75 | 33.64 | 18.12 | 83.13 |
| **2D RoPE** | **39.84** | **38.63** | **14.53** | **97.55** |

**结论**：2D RoPE带来**11.7% FID提升**，训练收敛更快。

#### 共享专家作用
- **移除共享专家**（S0A3）：收敛速度明显变慢，训练不稳定
- **共享专家提供正则化**：降低路由方差，优化更平滑

#### 专家数量与宽度权衡
- **E48 vs E16**：在相同激活参数下，48专家+窄中间层（hidden×0.3）一致优于16专家+宽层（hidden×4）
- **训练损失更低**：E48配置在所有模型规模下MSE损失均低于E16

---

## 4. 关键结论和发现

### 主要发现
1. **架构设计 > 路由策略**：MoE架构本身的优化比复杂路由机制更关键
2. **LLM MoE经验可迁移**：DeepSeek-V3的MoE块在扩散模型中同样有效
3. **参数重分配策略**：增加专家数量+减小中间宽度是更优的扩展方式
4. **2D结构信息至关重要**：2D RoPE显著提升空间推理能力和训练稳定性
5. **共享专家是稳定器**：提供跨token/time的通用表示，防止路由崩溃

### 方法局限性
- 仅在ImageNet类别条件生成上验证，未测试文本到图像生成
- 路由策略仍采用简单的Top-K，未探索更复杂的路由-架构联合优化
- 专家特化模式分析较初步，缺乏深层语义解释

### 未来工作方向
1. **结合更强路由策略**：将架构优化与先进路由方法结合
2. **扩展到多模态任务**：文本到图像、视频生成等
3. **RL-based优化**：探索强化学习在扩散MoE中的应用
4. **更大规模验证**：在10B+参数规模测试设计原则的有效性
5. **开源生态**：已开源代码和模型权重，促进社区进一步探索

---

**核心价值**：该研究首次系统性地证明了**架构设计是扩散模型MoE性能瓶颈**，而非路由机制，为高效训练大规模扩散模型提供了可复现的实用指南。

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
长上下文LLM推理面临严重的内存瓶颈：KV缓存随序列长度线性增长，导致**time-to-first-token (TTFT)** 和 **time-per-output-token (TPOT)** 显著增加。现有优化技术（KV量化、分块预填充、权重量化）通常被孤立评估，其**联合效果**和**最优配置**在边缘部署场景下缺乏系统性研究。

### 提出的新方法
**KV Pareto框架**：首个系统级评估框架，用于联合优化三种互补技术：
- **KV缓存量化**：支持int2/4/8及混合精度（k8v8, k8v4, k8v2, k4v4, k4v2, k2v2）
- **分块预填充（Prefill Chunking, PC）**：将输入提示分割为小块顺序处理
- **模型权重量化**：采用AWQ实现4-bit权重压缩

### 相比现有方法的优势
1. **联合评估**：首次系统性地研究三种优化技术的交互效应，而非孤立分析
2. **Pareto最优配置**：为不同模型识别内存-准确性的权衡前沿，提供可部署的配置方案
3. **系统级视角**：综合考虑峰值内存、KV缓存内存和模型内存的总消耗
4. **轻量化**：所有优化均为训练后方法，无需微调即可扩展到不同LLM架构

---

## 2. 核心实验方法和设置

### 数据集
- **长上下文任务**：LongBench的HotpotQA（平均9k tokens）和Qasper（平均4k tokens）
- **检索能力验证**：Needle-in-a-Haystack (NIAH) 最高32k上下文
- **短上下文验证**：GSM8k（数学推理）和MMLU（多任务理解）确保非长上下文任务性能

### 实验设置
- **硬件**：AMD MI-210和MI-325 GPU
- **模型**：覆盖多种架构
  - Qwen2.5-3b/7b-instruct
  - Llama3.2-3b/3.1-8b-instruct
  - Mistral-7b-instruct-v0.3
- **评估指标**：
  - **总内存消耗** = 峰值内存 + KV缓存内存 + 模型内存
  - **任务准确性**：F1分数（HotpotQA/Qasper）、检索准确率（NIAH）、任务特定指标（GSM8k/MMLU）

### 基线方法
- **无优化基线**：w16a16_k16v16（FP16权重和KV缓存）
- **消融配置**：单独启用PC、单独AWQ、不同KV量化粒度对比

### 搜索空间
- **KV量化粒度**：per-token, per-tensor, per-block
- **组大小**：{32, 64, 128}
- **分块大小**：{64, 128, 256, 512, 1024}（最终固定为256）
- **K-smoothing**：均值平滑以降低量化误差

---

## 3. 主要实验结果和性能指标

### 关键性能数据（10k上下文长度）
| 模型 | Pareto最优配置 | 内存减少 | 长上下文任务准确率下降 |
|------|----------------|----------|------------------------|
| Qwen2.5-3B | w4a16_k4v4 + PC | **73%** | ~1-3% |
| Llama3.2-3B | w4a16_k4v4 + PC | **76%** | ~1-3% |
| Qwen2.5-7B | w4a16_k8v8 + PC | **68%** | ~1-3% |
| Llama3.1-8B | w4a16_k8v2 + PC | **75%** | ~1-3% |
| Mistral-7B | w4a16_k4v4 + PC | **78%** | ~1-3% |

**总体**：实现**68-78%总内存节省**，仅**1-3%长上下文任务准确率下降**

### 与基线对比结果
1. **分块预填充（PC）**：
   - 峰值内存减少最显著
   - 对任务准确率影响极小（<0.5%）
   - 块大小变化（64-1024）对性能无显著影响

2. **AWQ权重量化**：
   - 进一步减少模型内存占用的15-20%
   - 对GSM8k等推理任务影响较大（最高-7%），但对MMLU影响较小（<2%）
   - 意外发现：w4a16_k4v4在HotpotQA上**优于**k8v4配置

3. **KV量化**：
   - **per-token量化**性能最佳，优于per-block和per-tensor
   - 小模型（3B）最优组大小为32，大模型（7B/8B）为64
   - **K-smoothing**对int4量化至关重要，可提升5-10%准确率

### 消融实验结果
- **粒度消融**：per-token + group size 32/64组合在Qwen3B和Mistral7B上分别比per-tensor提升**11%和10%** F1分数
- **K-smoothing消融**：int4量化下，沿序列维度均值平滑使Qwen3B在HotpotQA上从"gibberish"提升至**41.69%**
- **块大小消融**：256为最佳平衡点，兼顾内存与效率

### 扩展上下文验证（128k）
- **内存优势更显著**：w4a16-k4v4 + PC + FlashAttention在128k时比基线节省**>60%**内存
- **NIAH性能**：Mistral-7B的w4a16-k4v4配置在**20k内保持>90%**检索准确率，Qwen-2.5-3B在**14k内稳定**
- **块大小影响**：1k块大小比4k额外节省**23%**内存

---

## 4. 关键结论和发现

### 主要发现
1. **联合优化必要性**：三种技术组合效果远超单一优化，且存在**协同效应**（如PC+KV量化在Qasper上准确率反超高精度基线）
2. **模型依赖性**：Pareto最优配置因模型而异，无通用配置
   - **小模型**（3B）倾向激进量化（k4v4）
   - **大模型**（7B/8B）倾向保守策略（k8v8或k8v2）
3. **精度分配策略**：权重4-bit + KV混合精度为边缘部署最佳实践
4. **K-smoothing有效性**：显著改善低精度KV量化的分布偏移问题
5. **任务鲁棒性**：长上下文任务（HotpotQA/Qasper）对量化容忍度高于短上下文推理任务（GSM8k）

### 方法局限性
1. **固定块大小**：未探索动态块大小调整，128k上下文显示块大小对性能影响显著
2. **量化算法**：仅使用RTN量化，未采用Hessian-based方法（如QuaRot、SpinQuant）
3. **延迟未优化**：PC引入额外KV缓存写入延迟，未纳入Pareto权衡
4. **上下文长度**：NIAH验证仅到32k，128k+的准确率评估留待未来工作
5. **架构覆盖**：未验证Granite等混合架构的通用性

### 未来工作方向
1. **动态优化**：探索上下文长度感知的动态块大小和自适应层间量化精度
2. **先进量化**：集成Hessian旋转和SpinQuant以提升低精度性能
3. **多目标优化**：将**延迟**作为第三维指标纳入Pareto前沿
4. **超长上下文**：在128k+上下文全面评估任务鲁棒性
5. **硬件协同**：结合FlashAttention等优化内核实现额外6%内存节省
6. **架构扩展**：验证框架在MoE和混合模型上的适用性

---

**核心信息**：KV Pareto为边缘部署长上下文LLM提供了**可实践的Pareto最优配置**，在**不微调**前提下实现**近4倍内存压缩**，为资源受限场景的长上下文推理开辟了新路径。

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
- **GPU利用率低下**：原始Heimdall在单脉冲搜索工作负载中GPU平均利用率仅51.2%，存在严重的硬件资源浪费
- **GPU停滞问题（GPU stall problem）**：PCIe频繁数据传输导致GPU空闲等待，处理1GB文件产生7.85GB数据移动量
- **串行执行瓶颈**：DM trials循环中的Baseline Removal、Normalization、Matched Filtering和Peak Detection等阶段顺序执行，无法充分利用GPU并行能力
- **多文件处理效率低**：多进程并发导致CUDA上下文争用、资源竞争和进程中断问题

### 提出的新方法
- **细粒度并行化策略**：将DM trials循环分解为独立任务，通过OpenMP多线程和CUDA多流（multi-stream）实现并发执行
- **统一内存管理**：采用CUDA Unified Memory消除显式的host-device数据拷贝，实现按需页迁移
- **共享设备内存分配器**：自定义多线程内存池，通过全局队列管理内存块复用，减少cudaMalloc/cudaFree开销
- **双阶段流水线框架**：将CPU绑定的Pipeline Creation与GPU绑定的Execution解耦，通过无锁任务队列实现异步重叠

### 相比现有方法的优势
- **性能提升**：单文件处理速度提升最高2.66倍，多文件批处理提升2.05倍
- **GPU利用率**：从51%提升至92%，消除停滞周期
- **内存效率**：数据传输量减少6.7倍，内存分配调用减少41.2倍
- **结果一致性**：与原始Heimdall搜索结果完全等价
- **可扩展性**：支持更高并发度，避免多进程资源争用

---

## 2. 核心实验方法和设置

### 数据集
- **小规模测试**：1GB PSRFITS文件（J0528_2200_arcdrift-M01_0009.fits），来自CRAFTS巡天
- **大文件场景**：142GB filterbank文件（M5球状星团NGC 5904的30分钟观测数据）
- **多文件场景**：FAST-FREX数据集的FRB20201124子集，125个FITS文件（每个488MB）

### 实验环境
- **GPU**：NVIDIA GeForce RTX 3080Ti
- **CPU**：12th Gen Intel Core i9-12900K
- **系统**：Ubuntu 20.04.6 LTS
- **工具链**：NVIDIA CUDA Toolkit

### 评估指标
- **端到端处理时间**和**加速比**
- **GPU利用率**（通过Nsight Systems profiling）
- **Host-Device数据传输量**
- **CUDA API调用次数**（cudaMalloc）
- **各阶段执行时间**分解（RFI Mitigation、Dedispersion、Candidate Merging等）

### 基线方法
- **原始Heimdall**：GPU加速单脉冲搜索工具，作为性能基准
- **对比配置**：单线程/单流模式（等价于原始Heimdall执行流）

---

## 3. 主要实验结果和性能指标

### 单文件处理性能
| 并行度 | 加速比 | GPU平均利用率 |
|--------|--------|---------------|
| 1      | 1.42×  | -             |
| 2      | 2.06×  | -             |
| 4      | 2.84×  | -             |
| 6      | 3.26×  | -             |
| **8**  | **3.40×** | **92.11%** |

- **关键改进**：并行度为8时，DM trials循环阶段加速4.33×-6.05×
- **Normalization阶段**获益最大（6.05×），Matched Filtering最小（4.33×）

### 大文件处理（142GB M5数据）
- **加速比**：并行度8时达到**2.66×**
- **GPU利用率**：平均77.73%（持续处理模式）
- **双缓冲策略**：有效重叠I/O与计算，缓解GPU停滞

### 多文件批处理性能
| 并行度 | Heimdall性能 | Heimdall++加速比 |
|--------|--------------|------------------|
| 2      | 基准         | 1.41×            |
| 3      | 无法运行     | 1.80×            |
| **4**  | **无法运行** | **2.05×**        |

- **并发能力**：Heimdall++支持4线程稳定运行，而Heimdall在2进程以上因GPU内存不足中断
- **资源效率**：多线程避免CUDA上下文切换开销

### 内存与数据传输优化
- **数据移动总量**：从7.85GB降至1.17GB（**减少85%**）
  - Host-to-Device：5.19GB → 716.87MB
  - Device-to-Host：2.66GB → 480.75MB
- **内存分配**：cudaMalloc调用从198,392次降至4,810次（**减少97.6%**）
- **RFI Mitigation**：加速3.25×
- **Dedispersion**：加速1.59×
- **Candidate Merging**：通过共享内存重构算法，复杂度从O(N²)优化

---

## 4. 关键结论和发现

### 主要发现
1. **并行化有效性**：DM trials的细粒度并行化是提升GPU利用率的关键，使流式多处理器 occupancy显著提高
2. **数据传输是主要瓶颈**：PCIe带宽限制导致的GPU停滞占原始Heimdall运行时间的很大比例，Unified Memory的按需迁移策略有效解决此问题
3. **内存分配开销不可忽视**：在数千次DM trial重复调用中，cudaMalloc/cudaFree累积延迟显著，内存池复用带来数量级改进
4. **流水线重叠至关重要**：CPU阶段的I/O和元数据准备可与GPU计算重叠，掩盖延迟并提升吞吐量
5. **多线程优于多进程**：单进程多线程模型避免CUDA上下文争用，实现更好的可扩展性和资源利用率

### 方法局限性
- **硬件依赖性**：最优并行度受GPU计算能力和内存容量限制，超过阈值后收益递减
- **内存容量约束**：尽管使用Unified Memory，极大DM范围或超高分辨率数据仍可能触发OOM
- **线程不均衡**：DM trials循环末尾存在线程负载不均，导致短暂GPU利用率下降
- **系统资源限制**：并行度超过4时可能遇到系统级资源约束（如主机内存带宽）

### 未来工作方向
- **动态并行度调整**：根据工作负载特征和硬件能力自动优化线程/流数量
- **多GPU扩展**：探索跨多个GPU的分布式处理以支持超大规模巡天数据
- **算法级优化**：进一步改进Candidate Clustering算法，降低全局内存访问
- **异构计算**：结合CPU预处理与GPU加速，实现更细粒度的任务卸载
- **实时处理集成**：将Heimdall++嵌入下一代射电望远镜（如SKA）的实时数据流水线

---

**总结**：Heimdall++通过系统性的端到端优化，在保持科学结果一致性的前提下，将单脉冲搜索效率提升2-3倍，为应对未来TB/s级数据率的时域射电天文观测提供了可行的实时处理方案。

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

## 1. 主要贡献和创新点

### 解决的问题
- **diffusion LLMs (dLLMs) 推理效率瓶颈**：dLLMs通过迭代denoising生成token，但答案往往在固定步数完成前就已稳定，导致计算资源浪费
- **训练元数据丢弃问题**：现代LLM部署范式中，训练阶段的优化动态（梯度信息）包含丰富的参数重要性信号，但这些元数据在训练完成后通常被丢弃，造成信息浪费

### 提出的新方法
- **EDIT (Early Diffusion Inference Termination)**：一种基于训练梯度动态的推理时自适应终止准则
- **核心思想**：利用AdamW优化器在SFT（Supervised Fine-Tuning）过程中产生的moment estimates，构建"reasoning map"来编码学习到的推理路径
  - **训练阶段**：保存LoRA参数的AdamW更新历史（AdamW evolution），形成参数重要性指纹
  - **推理阶段**：通过cosine similarity比较当前token activations与保存的reasoning map，当alignment稳定时提前终止denoising

### 相比现有方法的优势
- **无需架构改动**：直接兼容现有dLLMs，无需重新设计模型结构
- **极小存储开销**：仅存储约1.5-2MB元数据（占8GB模型的0.02%）
- **理论保证**：提供基于KL divergence和total variation distance的收敛性证明，确保终止安全性
- **性能提升**：在保持或提升准确率的同时，显著减少推理计算量

---

## 2. 核心实验方法和设置

### 数据集
在5个推理benchmark上评估：
- **Countdown**（组合数学）
- **Sudoku**（逻辑推理）
- **MATH500**（数学问题）
- **GSM8K**（数学应用题）
- **GPQA**（研究生级别问答）

### 实验设置
- **基线模型**：LLaDA-8B
- **微调数据**：s1数据集
- **适配方法**：LoRA应用于所有32个Transformer块的QKV投影
- **序列长度**：128 / 256 / 512 tokens
- **硬件平台**：Intel XPU（确保可复现性）

### 评估指标
- **准确率**：各任务的标准评估指标
- **效率**：平均denoising steps减少百分比
- **存储开销**：元数据大小（MB和占比）
- **理论验证**：满足PAC-style bound的终止比例

### 基线对比
- **LLaDA (No SFT)**：未微调的原始模型
- **LLaDA (SFT)**：完整SFT + 固定步数denoising（64/128/256 steps）
- **EDIT (Ours)**：SFT + 自适应提前终止

---

## 3. 主要实验结果和性能指标

### 推理效率提升
EDIT显著减少denoising steps，**平均减少11.8%至68.3%**：

| 数据集 | 序列长度 | Baseline步数 | EDIT步数 | 减少率 |
|--------|----------|--------------|----------|--------|
| Countdown | 128 | 64 | 40.4 | **36.9%** |
|          | 256 | 128 | 40.6 | **68.3%** |
|          | 512 | 256 | 133.3 | **47.9%** |
| Sudoku   | 128 | 64 | 38.3 | **40.2%** |
|          | 256 | 128 | 74.9 | **41.5%** |
|          | 512 | 256 | 163.3 | **36.2%** |
| MATH500  | 128 | 64 | 38.1 | **40.5%** |
|          | 256 | 128 | 81.9 | **36.0%** |
|          | 512 | 256 | 197.2 | **23.0%** |
| GSM8K    | 128 | 64 | 42.8 | **33.1%** |
|          | 256 | 128 | 103.5 | **19.2%** |
|          | 512 | 256 | 225.8 | **11.8%** |
| GPQA     | 128 | 64 | 40.3 | **37.0%** |
|          | 256 | 128 | 81.3 | **36.5%** |
|          | 512 | 256 | 194.1 | **24.2%** |

### 准确率表现
- **显著提升**：Countdown（**+31.6%**）、Sudoku（**+16.1%**）
- **保持竞争力**：MATH500、GSM8K、GPQA与baseline相当
- **关键发现**：提前终止避免了late-step degradation，防止正确中间状态被覆盖

### 存储和计算开销
- **存储**：每LoRA模块存储16KB，总计约**1.5MB**（占8GB模型的**0.02%**）
- **计算**：增加cosine similarity和KL divergence计算，复杂度为O(|S_t|·d_out)和O(|I_t|)，相比self-attention的O(L²·d)可忽略不计
- **净效果**：显著加速，无实际性能损失

### 消融实验结果
- **模块选择**：Query projection的LoRA-B矩阵 + row-wise energy reduction效果最佳（KL stability最低0.089）
- **超参数敏感性**：不同任务需要不同的threshold δ和stability span Ω（通过验证集选择）
- **梯度分析**：pseudo-gradients在约19-23步收敛到SFT regime，与EDIT终止点一致

---

## 4. 关键结论和发现

### 主要发现
1. **训练元数据的价值**：AdamW优化动态编码了模型学习推理路径的关键信息，这些信息可用于指导推理决策
2. **推理-训练对齐**：当推理时的token activations与训练时的reasoning map对齐稳定后，继续denoising只会增加计算成本而不会提升质量
3. **任务特异性**：EDIT在结构化推理任务（Countdown、Sudoku）上效果最显著，因这些任务有清晰的推理模式；在长篇推理链任务（GSM8K长序列）上增益较小
4. **理论-实践一致性**：72.3%的提前终止满足PAC-style理论保证，表明EDIT在实践中可靠

### 方法局限性
- **依赖训练动态**：需要访问SFT过程中的AdamW状态，对已发布模型（仅含权重）不适用
- **任务特定阈值**：δ和Ω需针对任务调整，尽管可通过验证集自动选择
- **LoRA限制**：当前仅评估LoRA适配，full-parameter微调扩展待验证
- **长序列挑战**：长篇推理链可能过早稳定但推理未完成，导致准确率轻微下降（如GSM8K 512长度从81.2%降至76.2%）

### 未来工作方向
1. **元数据标准化**：建议模型提供者在发布时包含优化元数据，推动更高效的推理生态
2. **自适应阈值**：开发动态或学习式的(δ, Ω)选择机制
3. **扩展应用**：
   - Token-wise freezing：逐token提前冻结，进一步提升效率
   - Subspace EDIT：使用低维reasoning subspace替代单一向量
   - 动态计算分配：根据prompt难度分配不同计算预算
   - 早期质量预测：在推理早期预测最终输出质量
4. **全参数扩展**：将EDIT应用于full-parameter微调场景
5. **跨模态推广**：探索在diffusion图像/视频模型中的应用

---

## 核心洞见
EDIT揭示了现代ML部署中的系统性低效：**训练阶段产生的丰富优化信息被浪费，而推理阶段却在做无信息指导的盲目计算**。通过保存和利用这些本已存在的元数据，可以在不牺牲质量的前提下实现显著的效率提升，为"全栈式"优化开辟了新研究方向。

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
- **CoT推理的局限性**：标准Chain-of-Thought (CoT)在解码过程中存在**内部确定性**，限制了对合理替代推理路径的探索
- **Soft Thinking的不足**：虽然通过连续语义空间改善了表示能力，但仍受**自回归解码的贪婪性质**限制，无法并行探索多样化推理轨迹，导致错误逻辑无法及时纠正
- **单路径推理的缺陷**：当推理轨迹基于错误前提时，缺乏及时的纠正反馈机制，使错误逻辑在后续步骤中无阻碍传播

### 提出的新方法
**Multi-Path Perception Policy Optimization (M3PO)**：一个新颖的强化学习框架，核心思想包括：
- 将**并行策略rollout**作为自然的多样化推理源，无需额外辅助数据集
- 设计**轻量级协作机制**，通过无参数门控函数实现跨路径交互，在策略更新中整合全局洞察
- 采用**混合思维嵌入**（hybrid thinking embedding）：$\bar{h}_{i}^{(l)}=(1-\lambda)e_{i}^{(l)}+\lambda c_{i}^{(l)}$，平衡内在推理方向与同伴洞察

### 相比现有方法的优势
- **性能提升**：在知识密集型任务上平均提升**9.5%**，在STEM任务上超越强7B基线**5.3%**
- **参数效率**：**无额外可训练参数**，保持与标准LLM相同的推理效率
- **鲁棒性**：通过跨路径反馈打破自增强循环，培养更可靠的多步推理模式
- **兼容性**：仅在训练时激活协作机制，测试时保持单路径解码，与现有LLM架构完全兼容

---

## 2. 核心实验方法和设置

### 数据集
**知识密集型任务**（5个开放域和多跳QA数据集）：
- Natural Questions (NQ), TriviaQA, HotpotQA, 2WikiMultiHopQA (2WikiMQA), Bamboogle

**推理密集型STEM任务**（5个数学和科学基准）：
- GSM8k, MATH, MATH500, MMLU-STEM, ARC-Challenge

### 实验设置
- **模型规模**：Qwen2.5-1.5B-Instruct 和 Qwen2.5-3B-Instruct
- **检索设置**：使用E5-base嵌入模型检索top-3 Wikipedia文档作为上下文
- **训练配置**：
  - 知识任务：group size N=4
  - 复杂推理任务（GSM8k/MATH/MMLU-STEM）：N=8
  - 优化器：AdamW 8bit，学习率5e-6
  - LoRA微调：rank=32，α=64
  - 最大生成长度：1024 tokens

### 评估指标
- **知识任务**：Exact Match (EM) 分数
- **STEM任务**：Accuracy（基于最终答案正确性）
- **推理链质量**：token长度、格式规范性、逻辑连贯性

### 基线方法对比
- **监督微调**：SFT
- **强化学习方法**：PPO, GRPO, RLOO, REINFORCE++
- **检索增强**：RAG, IRCoT, Search-o1
- **混合推理方法**：HRPO (Hybrid Reasoning Policy Optimization)
- **连续推理方法**：Soft Thinking, Coconut, CODI

---

## 3. 主要实验结果和性能指标

### 知识基准测试结果
| 模型 | 方法 | NQ | TriviaQA | HotpotQA | 2WikiMQA | Bamboogle | **Average** |
|------|------|----|----------|----------|----------|-----------|-------------|
| Qwen2.5-7B | RAG | 34.9 | 58.5 | 29.9 | 23.5 | 20.8 | 33.5 |
| Qwen2.5-1.5B | **M3PO** | **41.4** | 56.8 | 28.7 | 27.9 | 23.2 | **35.6** (+2.1%) |
| Qwen2.5-3B | **M3PO** | **44.1** | **61.0** | **33.2** | 31.4 | **31.2** | **40.2** (+6.7%) |

**关键发现**：
- 1.5B模型超越7B RAG基线**2.1%**，3B模型超越**6.7%**
- 在NQ数据集上，3B模型达到**44.1% EM**，比7B RAG高**9.2%**
- 相比GRPO，1.5B模型提升**9.5%**，验证跨路径协作的有效性

### STEM基准测试结果
| 模型 | 方法 | GSM8k | MATH | MATH500 | MMLU-ST | ARC-C | **Average** |
|------|------|-------|------|---------|---------|-------|-------------|
| DeepSeekMath-7B | CoT | 64.2 | 36.2 | 34.6 | 56.5 | 67.8 | 51.9 |
| Qwen2.5-7B | CoT | 85.4 | 49.8 | 46.4 | 72.3 | 63.7 | 63.5 |
| Qwen2.5-3B | **M3PO** | **84.8** | **60.7** | **63.0** | **61.6** | **82.6** | **70.5** (+5.3%) |

**关键发现**：
- 3B模型平均准确率**70.5%**，超越最强7B基线**5.3%**
- 在MATH基准上达到**60.7%**，显著超过最佳7B基线（49.8%）
- 相比HRPO，在所有数据集上均实现提升

### 消融实验结果

**混合推理范式对比**（Qwen2.5-3B on MATH）：
- **Hidden States**：零奖励（与预训练嵌入空间不兼容）
- **Soft Thinking**：收敛慢，最终奖励低
- **HRPO**：性能接近但始终低于M3PO
- **M3PO**：**最快收敛**，**最高稳定奖励**

**跨路径融合机制**：
- **No Cross-Path**（λ=0）：性能最低，退化为标准CoT
- **Peer Mean**（均匀平均）：性能下降，验证选择性门控的必要性
- **M3PO**（相似度加权）：**最高奖励**，有效过滤冲突信号

**超参数敏感性**：
- **λ系数**：λ=0.1最优，λ≥0.5时**性能崩溃**（锚定轨迹失去主导性）
- **温度T**：T=0.1最优，过高温度导致注意力分散，降低推理稳定性

---

## 4. 关键结论和发现

### 主要发现
1. **多路径协作的有效性**：并行rollout提供了自然的推理多样性，跨路径交互能打破自增强循环，使模型从替代路径中学习并纠正错误
2. **参数效率与性能兼得**：M3PO**无需额外参数**，在保持推理效率的同时，通过训练时协作将鲁棒推理模式内化为模型能力
3. **双重控制机制**：λ平衡内在推理与外部洞察，T精细过滤信号质量，二者协同实现稳定的多路径推理
4. **小模型大能力**：1.5B/3B模型通过M3PO训练可**超越7B模型**性能，证明协作学习能有效补偿模型容量限制

### 方法的局限性
- **规模限制**：仅在最多3B参数的模型上验证，更大模型的可扩展性待探索
- **计算开销**：训练时需要生成多个rollout（N=4或8），增加训练时间（但推理时无额外开销）
- **任务范围**：主要评估集中在QA和数学推理，其他复杂推理场景需进一步验证

### 未来工作方向
- **可扩展性研究**：在更大规模模型（7B+）上验证M3PO的有效性
- **自适应协作**：探索动态调整λ和T的机制，实现更智能的路径融合
- **更广泛评估**：扩展至代码生成、逻辑推理等更多任务类型
- **理论分析**：深入研究多路径协作如何影响模型的内部表示和推理动力学

---

## 核心结论
M3PO通过**强化学习驱动的多路径协作机制**，成功将并行推理轨迹的集体洞察整合到策略优化中，在**无额外参数**和**保持推理效率**的前提下，显著提升了LLM的推理鲁棒性。该方法不仅在知识密集型和STEM基准上达到SOTA性能，更重要的是为**小模型实现大模型能力**提供了可行路径，代表了向高效、可解释且稳健的LLM推理系统迈进的重要一步。

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
# Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling 论文总结

## 1. 主要贡献和创新点

### 解决了什么问题
- **NVFP4量化的精度瓶颈**：标准NVFP4将所有量化块(block)缩放到FP4的最大值6，导致接近最大值的数值（特别是5附近）量化误差极大，这是模型性能下降的主要原因
- **训练不稳定**：现有NVFP4训练配方在多种架构（Transformer、Hybrid、Gated Attention）中会出现训练发散问题
- **计算开销与精度权衡**：现有方法需要大量补偿操作（如随机Hadamard变换、随机舍入、高精度"修复"训练），导致实际加速效果有限

### 提出的新方法
**Four Over Six (4/6)**：一种自适应块缩放算法，为每个16值块评估两种缩放因子：
- 标准缩放：$M=6$（使用FP4全范围±{0, 0.5, 1, 1.5, 2, 3, 4, 6}）
- 替代缩放：$M=4$（使用子范围±{0, 0.5, 1, 1.5, 2, 3, 4}）

**核心思想**：通过牺牲极端值(±6)的表示能力，换取对近最大值（如75%最大值）更精确的表示。当块内数值分布集中在75%-100%最大值区间时，缩放至4能显著降低MSE误差。

### 相比现有方法的优势
- **精度提升**：在PTQ中，WikiText-2 perplexity平均降低2.8-4.3%，下游任务准确率平均提升0.5-1.2%
- **训练稳定性**：有效防止NVFP4训练发散，loss曲线接近BF16基线
- **低开销**：在NVIDIA Blackwell GPU上实现，推理开销<2%，训练开销<15%
- **即插即用**：可无缝集成到GPTQ、AWQ、SmoothQuant等现有PTQ方法中
- **硬件兼容**：利用现有NVFP4指令集，无需硬件修改

---

## 2. 核心实验方法和设置

### 数据集
- **语言建模**：WikiText-2, C4
- **下游任务**：BoolQ, ARC-Easy, ARC-Challenge, HellaSwag

### 模型架构
- **预训练实验**：
  - Transformer 340M/1.3B（带QK Norm）
  - Hybrid 340M/1.3B（交替全注意力和滑动窗口注意力）
  - Hybrid with Gated Attention 1.4B
- **PTQ实验**：
  - Llama 3系列：1B, 8B, 70B
  - Qwen 3系列：1.7B, 8B, 32B

### 评估指标
- **困惑度(Perplexity)**：WikiText-2 Word PPL, C4 Word PPL（↓越低越好）
- **下游任务准确率**：BoolQ, ARC-Easy, ARC-Challenge, HellaSwag（↑越高越好）
- **训练稳定性**：训练loss曲线对比

### 基线方法
- **精度基线**：BF16
- **量化格式**：MXFP4, NVFP4(M=6)
- **PTQ方法**：RTN, GPTQ, AWQ, SmoothQuant
- **训练方法**：标准NVFP4训练配方（含随机Hadamard变换、随机舍入）

---

## 3. 主要实验结果和性能指标

### 预训练结果
**训练稳定性显著提升**：
- NVFP4基线在Transformer/Hybrid架构中均出现发散（图4）
- **4/6使所有架构稳定训练**，loss曲线接近BF16
- 在340M Transformer上，2D块量化+4/6达到最佳效果（图5）

### PTQ量化结果（WikiText-2 Perplexity）

| 模型 | BF16 | NVFP4(M=6) | +4/6(MSE) | 改善率 |
|------|------|------------|-----------|--------|
| Llama-3-1B | 11.98 | 14.27 | **13.84** | -3.0% |
| Llama-3-8B | 7.54 | 8.43 | **8.30** | -1.5% |
| Llama-3-70B | 2.86 | 4.00 | **3.83** | -4.3% |
| Qwen-3-1.7B | 21.06 | 23.06 | **23.60** | +2.3%* |
| Qwen-3-8B | 12.22 | 12.68 | **12.56** | -0.9% |

*注：Qwen-3-1.7B在C4上改善明显（65.54→66.32）

### 与现有PTQ方法结合效果
**AWQ+4/6表现最佳**：
- 平均WikiText-2 PPL：11.58（vs 12.22 BF16）
- 平均C4 PPL：32.36
- **相比单独AWQ**：WikiText-2提升0.37-0.77 PPL，C4提升0.71-1.01 PPL

**SmoothQuant+4/6**：
- 所有模型均获得一致改善
- 平均准确率提升0.3-0.5个百分点

**GPTQ+4/6**：
- 效果不一致，部分模型性能轻微下降
- 作者归因于GPTQ优化过程未原生集成4/6

### 下游任务性能（Llama-3-8B示例）
| 方法 | BoolQ | Arc-E | Arc-C | HellaSwag | 平均 |
|------|-------|-------|-------|-----------|------|
| BF16 | 83.2 | 82.5 | 55.0 | 79.3 | 75.0 |
| NVFP4 | 80.2 | 77.5 | 52.5 | 77.7 | 72.0 |
| **+4/6** | **80.9** | **80.2** | **52.6** | **78.0** | **72.2** |
| AWQ+4/6 | 80.4 | 80.2 | 53.6 | 78.2 | 73.1 |

---

## 4. 关键结论和发现

### 主要发现
1. **误差来源定位**：NVFP4性能下降主要源于FP4数值舍入误差，而非FP8缩放因子误差（图2a）
2. **关键数值区间**：接近最大值的数值（特别是缩放后约5.0）是性能瓶颈，这些值在FP4中无精确表示（图2b）
3. **自适应有效性**：全局缩放至M=4反而恶化性能（表3），但**块级自适应选择**能充分利用两种缩放的优势
4. **MSE最优**：基于量化后MSE误差的块选择策略效果最佳（表4）

### 方法局限性
1. **格式限制**：仅适用于NVFP4，**不兼容MXFP4**（MXFP4使用FP8E8M0缩放因子，无法表示4和6之间的50%差异）
2. **开销存在**：尽管已优化，训练时仍有<15%额外计算开销
3. **旋转方法不兼容**：QuaRot、SpinQuant等旋转量化方法与NVFP4（含4/6）结合效果不佳
4. **极端情况**：对于异常值极少的模型，INT4等均匀量化格式可能更合适

### 未来工作方向
1. **更大规模训练**：在更大模型（>70B）和更长训练周期上验证有效性
2. **内核优化**：进一步降低4/6的计算开销
3. **GPTQ集成**：将4/6原生集成到GPTQ优化过程中
4. **格式扩展**：探索其他块缩放格式的自适应缩放策略
5. **理论分析**：深入研究为何旋转方法与NVFP4不兼容

### 实践建议
- **预训练**：在NVFP4训练中必须启用4/6以防止发散
- **PTQ**：推荐与AWQ或SmoothQuant结合使用
- **硬件**：需NVIDIA Blackwell GPU支持NVFP4指令集
- **避免**：与旋转类量化方法同时使用

---

**核心贡献总结**：4/6通过一个简单的"双候选评估"机制，在不改变硬件实现的前提下，显著提升了NVFP4的实用价值，使其在保持加速优势的同时，精度更接近BF16基线。

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

### 解决的问题
- **大规模图k-core分解的并行化挑战**：尽管存在线性时间的顺序算法（如Batagelj和Zaversnik算法），但现代社交网络、Web图等数据集（数亿顶点、数十亿边）对单核内存带宽和缓存容量造成巨大压力，亟需利用多核/众核架构加速
- **分布式算法在共享内存系统的适配难题**：将Montresor等人的分布式消息传递算法适配到共享内存架构时，面临同步开销、缓存一致性、共享数据结构竞争等非平凡问题

### 提出的新方法
开发了**三个渐进优化的Rust实现版本**：
- **SequentialK**：顺序基线版本，验证正确性，采用纯消息传递范式模拟分布式环境
- **ParallelK**：多线程并行版本，基于Rayon库和原生Rust线程，引入MessageMail结构实现邻居级锁粒度
- **FastK**：高度优化的共享内存版本，核心创新包括：
  - **全局共享状态**：将`est`（coreness估计值）和`active`（激活标志）向量全局共享，消除冗余存储
  - **选择性消息传播**：仅当`estimate(u) < estimate(v)`或`old_estimate(u) ≥ estimate(v)`时才发送消息，减少无效通信
  - **动态并行度调整**：根据活跃节点比例自动切换到顺序执行（优先队列），避免对高coreness节点的频繁重计算
  - **缓存感知数据结构**：使用排序向量替代HashMap存储邻居列表，利用小世界特性提升缓存命中率

### 相比现有方法的优势
- **性能突破**：在16线程下实现**最高11倍加速**，比NetworkX快**两个数量级**（如soc-LiveJournal1图：3.87秒 vs 480.92秒）
- **内存安全与性能兼得**：Rust的所有权模型在编译期消除数据竞争，同时零成本抽象和细粒度并发原语保证高性能
- **同步开销最小化**：通过合并收发阶段、减少屏障同步次数、原子操作优化，显著降低并行化开销

---

## 2. 核心实验方法和设置

### 数据集
使用**SNAP真实世界数据集**，涵盖三类典型图结构：
- **道路网络**：roadNet-PA (108万顶点, 154万边), roadNet-TX (138万顶点, 192万边), roadNet-CA (196万顶点, 276万边)
- **Web图**：web-NotreDame (32.5万顶点, 149万边), web-Stanford (28.1万顶点, 231万边), web-Google (87.5万顶点, 510万边), web-BerkStan (68.5万顶点, 760万边)
- **社交网络**：Wiki-Talk (239万顶点, 502万边), soc-Pokec (163万顶点, 3062万边), soc-LiveJournal1 (484万顶点, 6899万边)

**统计特征**：最大coreness值(kmax)从3(道路网)到372(LiveJournal)不等，平均度数普遍较低(2.76-27.32)，符合幂律分布

### 实验设置
- **硬件**：双路AMD EPYC 7551处理器（32核/路，共64核），128GB共享内存
- **线程配置**：测试线程数s = 1, 2, 4, ..., 128
- **编译优化**：Rust使用`--release`标志 + LTO（Link Time Optimization）
- **评估指标**：
  - **运行时间**：排除图加载时间，包含并行任务准备时间
  - **收敛速度**：每轮迭代中节点coreness估计值与最终正确值的平均距离
  - **活跃节点比例**：每轮仍需计算的节点占比

### 基线方法对比
- **NetworkX 2.x**：Python参考实现（顺序）
- **SequentialK**：本论文顺序基线
- **ParallelK**：本论文并行基线（三种线程策略）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（16线程）
| 图数据集 | FastK (秒) | NetworkX (秒) | 加速比 | ParallelK (秒) | SequentialK (秒) |
|----------|------------|---------------|--------|----------------|------------------|
| roadNet-CA | **0.35** | 40.48 | **116×** | 0.82 | 2.45 |
| soc-LiveJournal1 | **3.87** | 480.92 | **124×** | 7.07 | 78.11 |
| web-BerkStan | **0.59** | 686.63 | **1164×** | 2.22 | 5.19 |
| web-Google | **0.23** | 38.40 | **167×** | 0.34 | 3.13 |
| wiki-Talk | **0.61** | 115.28 | **189×** | 0.82 | 8.76 |

**总体表现**：FastK在所有数据集上** consistently 最优**，比ParallelK快1.5-3.8倍，比SequentialK快5-20倍

### 消融实验结果

#### 批大小（Batch Size）影响
- **测试范围**：64, 128, 256, 512, 1024节点/线程
- **最优值**：**256节点/线程**在soc-Pokec和soc-LiveJournal上取得最佳权衡
- **趋势**：过小导致线程管理开销大，过大导致负载不均衡

#### 数据结构选择（HashMap vs Sorted Vector）
- **性能提升**：排序向量比HashMap**快30%**
- **原因**：真实图平均度数低（<30），小向量上的线性扫描/二分查找因缓存友好性优于哈希计算
- **内存局部性**：连续存储减少缓存未命中，节省哈希函数开销

#### 并行策略对比
- **Native Threads vs Rayon**：原生线程** consistently 优于**Rayon库（6线程时快10-20%）
- **结论**：外部库引入管理开销，原生线程配合手动batching更优

#### 收敛行为分析
- **快速收敛**：前10-20轮误差下降4个数量级（从10⁴到10⁰）
- **长尾效应**：后期迭代为"微调"阶段，仅±1误差，活跃节点数降至**<1%**
- **动态切换阈值**：当活跃节点数 < batch size时转为顺序执行，避免过度并行化

---

## 4. 关键结论和发现

### 主要发现
1. **分布式协议可有效适配共享内存**：Montresor等人的消息传递协议在最小化同步点和缓存友好数据结构下，能高效运行于共享内存系统
2. **消息选择性至关重要**：通过条件判断过滤无效消息，减少锁竞争和通信量，是FastK优于ParallelK的关键
3. **激活调度优化必要**：活跃节点数呈指数衰减，动态调整并行度可避免对高coreness节点的不必要重计算
4. **Rust的实用性验证**：内存安全与底层控制可共存，为数据密集型计算提供安全高性能选择

### 方法局限性
- **评估范围局限**：仅在**单台NUMA机器**的内存执行，未测试跨节点分布式场景
- **静态图假设**：仅研究静态图，未评估动态增量/减量core维护
- **加载时间排除**：报告的运行时间不包含图加载和预处理时间
- **硬件依赖性**：性能可能受不同内存层次结构、图划分策略和偏斜度影响

### 未来工作方向
1. **真正分布式部署**：保留优化的共享内存内核，同时实现跨节点通信
2. **动态core维护**：研究增量/减量算法，支持实时图更新
3. **NUMA感知优化**：针对多路服务器优化分区策略，减少跨NUMA节点访问
4. **外部内存变体**：支持超出RAM容量的超大规模图处理
5. **异构加速**：探索GPU辅助过滤等硬件加速技术

---

**核心贡献总结**：本工作首次系统性地将分布式k-core分解协议移植到Rust共享内存环境，通过全局状态共享、选择性通信和动态调度等创新，在保持内存安全的同时实现极致性能，为大规模图分析提供了高效可靠的基础库。

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
现有RL后训练方法（如GRPO）在医疗领域存在根本性局限：**仅优化答案正确性**，忽视了临床推理所需的多维度目标。这导致模型生成碎片化、不可靠的推理过程，难以满足临床安全和监管合规要求。

### 提出的新方法
**Clinical-Objective Relative Policy Optimization (CRPO)**：一种可扩展的多目标、可验证RL框架，特点包括：
- **多目标奖励机制**：同时优化**准确性**、**忠实性**和**全面性**
- **结构化推理格式**：强制使用`<dx>`和`<conclusion>`标签分离分析过程与结论
- **无需人工标注**：完全基于规则的可验证奖励信号
- **临床认知对齐**：模拟临床医生的双系统思维（System 1直觉 + System 2分析）

### 相比现有方法的优势
- **GRPO仅奖励最终答案**，而CRPO奖励推理过程的结构化和可验证性
- 在保持准确性的同时，显著提升**医学忠实性**（+49%）和**推理全面性**
- 生成的推理过程透明、可审计，便于临床医生验证
- 训练稳定且计算高效，适用于资源受限场景

---

## 2. 核心实验方法和设置

### 数据集
| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| **MedQA** | 12,723题 | 美国/中国医师执照考试题 | 领域内评估（in-domain） |
| **MedMCQA** | 194,000题 | 印度医学入学考试题 | 领域外评估（out-of-domain） |
| **MedXpertQA** | 4,460题 | 专家级临床推理题 | 高难度评估 |

### 实验设置
- **基础模型**：Qwen2.5-3B-Instruct
- **训练框架**：Volcano Engine RL (verl)
- **训练流程**：
  1. **Cold Start**：用DeepSeek-R1蒸馏5,000道MedQA题（13 epochs）
  2. **CRPO训练**：在剩余5,000道MedQA题上训练20 epochs（rollout数G=5）
- **关键参数**：准确性奖励系数k=10，最大奖励值k+2=12

### 评估指标
**多维度评估体系**：
1. **准确性**：答案正确率
2. **认知行为评估**（LLM-as-judge）：
   - Backtracking（回溯）
   - Answer Verification（答案验证）
   - Subgoal Setting（子目标设定）
   - Backward-Chaining（反向链推理）
3. **医学忠实性**：
   - Faithfulness to Medical Knowledge（医学知识忠实性）
   - CECD（Case-grounded Evidence Citation Density）
   - DRC（Distractor Rejection Coverage）
   - Hallucination（幻觉率↓）

---

## 3. 主要实验结果和性能指标

### 准确性对比
| 方法 | MedQA | MedMCQA | MedXpertQA | 平均提升 |
|------|-------|---------|------------|----------|
| Baseline (CoT) | 41.95% | 46.78% | 10.51% | - |
| GRPO | 50.35% | 49.87% | 12.64% | +5.2% |
| **CRPO** | **52.41%** | **55.13%** | **14.88%** | **+8.6%** |
| Cold Start + GRPO | 51.07% | 48.86% | 12.64% | +5.4% |
| **Cold Start + CRPO (Clinical-R1-3B)** | **53.07%** | **58.10%** | **16.14%** | **+10.2%** |

### 认知行为提升（MedMCQA数据集）
| 指标 | Baseline | GRPO | CRPO | Cold Start+CRPO | 提升倍数 |
|------|----------|------|------|-----------------|----------|
| Backtracking | 0.36 | 0.40 | 0.73 | **2.49** | **5.9×** |
| Backward-Chaining | 0.19 | 0.25 | 0.62 | **1.06** | **4.6×** |
| Subgoal Setting | 2.04 | 1.88 | 2.51 | **4.23** | **2.1×** |
| Verification | 0.18 | 0.17 | 0.53 | **1.28** | **6.1×** |

### 医学忠实性结果（MedMCQA）
| 指标 | Baseline | GRPO | CRPO | Cold Start+CRPO | 相对提升 |
|------|----------|------|------|-----------------|----------|
| Faithfulness | 4.66 | 4.71 | 7.13 | **12.95** | **+178%** |
| CECD | 1.42 | 1.51 | 5.36 | **5.76** | **+306%** |
| DRC | 1.84 | 1.75 | 2.79 | **3.36** | **+83%** |
| Hallucination↓ | 0.40 | 0.85 | 0.64 | **0.66** | **-35%** |

### 响应长度分析
- **Cold Start**：生成冗长、枚举式推理（~1,500 tokens）
- **GRPO**：过度压缩响应（~1,100 tokens），丢失必要细节
- **CRPO**：平衡长度（~1,300 tokens），**简洁且结构完整**

---

## 4. 关键结论和发现

### 主要发现
1. **多目标优化有效性**：CRPO在保持准确性的同时，使医学忠实性提升近2倍，认知行为指标提升3-6倍
2. **结构化推理的关键作用**：`<dx>`与`<conclusion>`的强制分离促使模型进行**显式证据引用**和**干扰项系统排除**
3. **Cold Start的协同效应**：蒸馏初始化+CRPO优化产生最佳效果，表明**强先验知识**与**过程约束**相辅相成
4. **跨领域泛化能力**：CRPO在MedMCQA和MedXpertQA上均表现优异，证明其**领域外泛化性**

### 典型成功模式（Case Study）
在DLBCL风险因素案例中：
- **GRPO**：混淆"既往乳腺癌"（疾病标签）与"放疗暴露"（真实风险因素）
- **CRPO**：明确区分三类信息：
  - **因果暴露**：放疗（C选项）
  - **疾病标签**：乳腺癌（E选项）
  - **临床表现**：腋窝淋巴结受累（B选项）
  - **无关因素**：非洲旅行、性别（A/D选项）

### 方法局限性
1. **训练不稳定性**：无Cold Start时，策略与参考模型间的KL散度快速增长，导致训练不稳定
2. **评估依赖自动化**：认知行为和医学忠实性评估依赖Llama-3.1-8B和GPT-4作为裁判，可能与人类判断存在偏差
3. **基础模型限制**：Qwen2.5-3B未在医学语料上预训练，限制了领域知识深度
4. **单模态局限**：当前仅支持文本推理，未整合临床影像或结构化数据

### 未来工作方向
- 探索更稳定的CRPO变体（如自适应KL系数、离策略更新）
- 引入**人类在环评估**以增强可信度验证
- 扩展到**医学预训练模型**（如BioMistral、MEDITRON）
- 开发**多模态CRPO**支持影像和实验室数据
- 应用于**临床决策支持系统**的真实部署测试

---

**核心价值**：CRPO为高风险领域LLM后训练提供了可扩展的**多目标对齐范式**，在无需人工标注的情况下，实现了从"答案正确"到"推理可信"的跨越，为医疗AI的安全应用开辟了新路径。

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

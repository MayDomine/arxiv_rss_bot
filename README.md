# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-23 08:06:20 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Odin: Primitive-Level Synchronization for Distributed Point-Based Neural Rendering](https://arxiv.org/abs/2607.19893)

**Authors**: Zhenxiang Ma, Zeyu He, Yuanzhen Zhou, Zhenyu Yang, Yuchang Zhang, Miao Tao, Rong Fu, Jidong Zhai, Hengjie Li  
**Category**: cs.DC  
**Published**: 2026-07-23  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2607.19893v1  

#### Abstract
Point-based neural rendering (PBNR) represents 3D scenes as explicit, trainable primitives and underpins high-quality reconstruction and emerging embodied AI and world-model pipelines. Unlike layer-structured neural networks, PBNR has primitive-indexed dependencies: each view reads and updates only ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Odin: Primitive-Level Synchronization for Distributed Point-Based Neural Rendering

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在分布式 **Point-Based Neural Rendering (PBNR)**（如 3D Gaussian Splatting）训练中，传统方法依赖于**全局任务级或迭代级同步屏障**（global task-/iteration-level barriers），导致大量 GPU 等待时间。尽管每个训练视图只更新场景中的一小部分可见 primitives（点状基元），但同步机制仍强制所有后续视图等待整个全局状态更新完成。

这造成了严重的**系统不匹配**：依赖单元是 primitive index，而同步粒度却是整个任务或迭代。随着渲染优化（如 TamingGS、DashGS）缩短计算时间，通信开销占比上升，使得同步成为性能瓶颈。

---

### 🚀 提出的新方法与核心思想
论文提出 **Odin**，一种基于 **primitive-level synchronization**（基元级同步）的新型分布式 PBNR 训练系统，其核心创新如下：

#### （1）**预测-验证架构（Predict-and-Validate Framework）**
- **静态调度器（Ahead-of-Time Scheduler）** 利用稳定的 **relative locality graph (RLG)** 和 phase task graph 预测低冲突的任务顺序和重叠窗口。
- **运行时验证（Runtime Validation）** 在实际读取前检查 primitive 发布是否安全，仅发布“可能被观察到”的更新。

#### （2）**相对局部性图（Relative Locality Graph, RLG）**
- 构建于 SfM tracks 上的加权无向图，衡量不同 view 之间访问相同 primitives 的可能性。
- 使用 Jaccard 相似度定义边权重：  
  $$
  w_{ij} = \frac{|P_i \cap P_j|}{|P_i \cup P_j|}
  $$
  其中 $P_i$ 是第 $i$ 个 view 观察到的 track 集合。
- RLG 提供比几何粗略估计更精确、比运行时激活更早的依赖关系信号。

#### （3）**Shadow Graph 执行模型**
- 支持多个 task unit 并发执行而不复制完整模型状态。
- 每个 unit 拥有私有的逻辑视图（virtual data buffer + local autograd graph），梯度写入共享双槽环形缓冲区，经验证后提交至全局状态。
- 实现零拷贝下的并发执行与版本控制。

#### （4）**两条执行路径**
- **Quality-First Path**：严格等价于同步训练，仅当 primitive 范围无交集或已发布时才允许重叠，保证质量一致性。
- **Throughput-First Path**：允许有限延迟读取（delayed read），条件为：
  - 延迟范围比例 ≤ 参数 $t$
  - 生产者梯度强度低于活跃集均值的 $t$ 倍  
    → 适用于弱影响交互（如遮挡边缘、透明混合），提升吞吐量而不显著损害质量。

---

### 🔍 相比现有方法的优势
| 维度 | 现有方法（如 Grendel、DDP） | Odin |
|------|-------------------------------|------|
| 同步粒度 | Task/Iteration-level barrier | **Primitive-level publication** |
| 冗余等待 | 存在大量无关 primitive 更新阻塞 | 只等待真正相关的 pending updates |
| 通信隐藏 | 有限，受限于粗粒度屏障 | 显著提升 overlap 机会 |
| 质量保障 | 强一致性 | 质量优先路径保持等效；吞吐路径可控降级 |
| 系统侵入性 | 需修改 renderer 或 optimizer | **无需改动 kernel、optimizer、budget、capacity** |

---

## 2. 核心实验方法和设置

### 📚 数据集
实验覆盖多种场景类型，共 **14 个场景**：
| 数据集 | 场景数 | 类型 | 规模 |
|--------|-------|------|------|
| **MipNeRF360** | 9 | 室内外 360° | 混合 |
| **Tanks & Temples** | 2 | 户外/物体 | 中等 |
| **DeepBlending** | 2 | 室内房间 | 中等 |
| **MatrixCity** | 1 | 航拍城市 | 大规模（MP-only） |

> 注：MatrixCity 用于大规模混合并行（Mixed Parallelism, MP）案例研究。

---

### ⚙️ 实验设置
- **硬件平台**：NVIDIA RTX 4090 GPUs，PCIe 4.0 ×16（节点内），160 Gbps eRDMA（跨节点）
- **GPU 数量**：单节点 8-GPU 至多节点 64-GPU
- **批大小**：每 GPU $B_{\text{gpu}} = 2$，全局 batch size 可变（4–32）

#### ✅ 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput** | $\text{throughput} = \frac{N_{\text{iter}} \cdot B}{T_{\text{wall}}}$（samples/s） |
| **Speedup** | 相对于基线的加速比 |
| **Reconstruction Quality** | PSNR、SSIM、LPIPS，报告相对于基线的归一化 delta |
| **Communication Hiding** | 成功隐藏的 critical-path wait 百分比 |
| **Overhead** | AOT 编译时间、内存开销、fallback rate |

---

### 🆚 基线方法对比
| 场景 | 基线方法 |
|------|---------|
| **Data Parallelism (DP)** | PyTorch DDP（含 PBNR 特定 densify/prune） |
| **Mixed Parallelism (MP)** | Grendel [9]（当前最优分区方案） |
| **其他对照组**：
- No-locality async overlap
- Locality-only ordering（无异步）
- HOGWILD-style partitionless execution
- Without dynamic validation

> 所有比较均保持 renderer kernels、optimizer、training budget、model capacity 不变。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）端到端吞吐提升
- 在 **13 个非城市 8-GPU 场景**上平均提速 **1.22×**
- 在 **MatrixCity 混合并行扩展至 64 GPUs** 时，相比 Grendel 最高提速 **1.89×**

> 加速效果随 batch size 增大而略有下降（因同步频率降低），符合预期行为。

#### （2）通信等待隐藏
- 成功隐藏了 **82% 的 critical-path wait 时间**
- 如图 19 所示，Odin 将暴露的通信开销从 baseline 的 ~200ms 降至 ~35ms

#### （3）质量影响（Throughput-First 路径）
- 在 $K=4, t=0.2$ 设置下，所有场景的 PSNR、SSIM、LPIPS 的 aggregate delta 均在 **±1% 报告带宽内**
- 表明延迟读取对最终重建质量影响极小，尤其在非主导交互区域（如遮挡边界）

#### （4）消融实验（Ablation Study）

| 消融项 | 对比结果 |
|--------|----------|
| **w/o SAS**（无静态异步调度） | 性能大幅回落，接近 baseline → 表明 overlap 是主要收益来源 |
| **w/o SDS**（无静态数据调度） | 显著下降 → 说明低耦合排序至关重要 |
| **w/o DAS/DDS**（无动态验证/调度） | 下降明显 → 动态修正提升了鲁棒性和利用率 |
| **Partitionless HOGWILD 控制组** | 吞吐更低且质量受损 → 单纯去同步不可行 |

> 结论：Odin 是一个协调设计的整体，各组件协同作用。

#### （5）质量优先路径回退行为
- 在稀疏场景（playroom, drjohnson）中，overlap 接受率 >98%，speedup 达 1.23×
- 在密集场景（kitchen）中，fallback rate 高达 98.6%，speedup ≈ 1.00× → 自动退化为同步模式，确保安全性

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **全局同步不是必须的**：PBNR 的稀疏依赖特性使其天然适合细粒度同步。
2. **primitive-level synchronization 可显著提升吞吐**：通过打破粗粒度 barrier，Odin 实现平均 **1.22×** 加速。
3. **利用物理先验（SfM tracks）构建 RLG 是高效可行的**：AOT 编译开销极低（50k images < 10 秒），可忽略不计。
4. **Shadow Graph 实现轻量级并发执行**：相比全状态复制，内存开销近乎恒定，操作虚拟化成本降低最多 **257×**。
5. **Throughput-First 路径安全可用**：在合理参数下（$t=0.2$），质量损失在感知阈值内。

---

### ⚠️ 局限性
1. **依赖 conservative scope 信息**：若无法获取 view 的候选 primitive 集合（如某些黑盒 renderer），则需 fallback 到更保守策略。
2. **结构化变更仍需同步**：densification、pruning、opacity reset 等全局操作仍触发 barrier。
3. **对极端密集场景增益有限**：如 kitchen 等高耦合场景，fallback 率高，加速不明显。
4. **当前假设 primitive ID 稳定**：若存在频繁分裂/合并，需额外版本管理机制。

---

### 🔮 未来工作方向
1. **支持动态 primitive 拓扑变化的版本控制机制**
2. **将 Odin 思想推广至其他显式状态 AI 系统**：
   - Neural Mapping
   - Online Reconstruction
   - Object-/Voxel-/Map-level World Models
3. **结合硬件支持实现更低延迟的 Shadow Graph 执行**
4. **自适应 tuning $t$ 和 $K$ 参数以最大化收益**

---

## 总结一句话
> **Odin 通过 primitive-level synchronization 打破了分布式 PBNR 中的全局同步瓶颈，在不改变任何模型或优化器的前提下，实现了高达 1.89× 的训练加速，同时保持重建质量稳定，为下一代大规模神经渲染系统提供了新的系统范式。**

</details>

---

### 2. [LISA: Linear-Indexed Sparse Attention for Efficient Long-Context Reasoning](https://arxiv.org/abs/2607.19358)

**Authors**: Yu Zhao, Zekun Zhang, Fan Jiang, Bo Zeng, Linlong Xu, Shimin Shan, Yu Liu, Longyue Wang, Weihua Luo  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2607.19358v1  

#### Abstract
Recent advances in long chain-of-thought reasoning models such as DeepSeek-R1 have led to increasingly longer inference context lengths under the test-time scaling paradigm. However, the O(n^2) computational complexity of standard self-attention causes inference costs to grow sharply with long seque...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LISA: Linear-Indexed Sparse Attention for Efficient Long-Context Reasoning —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Chain-of-Thought (CoT)** 推理模型（如 DeepSeek-R1）在测试时通过扩展推理步数来提升性能（test-time scaling），导致上下文长度急剧增长。然而，标准 **Transformer** 架构中的 **self-attention** 模块具有 $O(n^2)$ 的计算复杂度，使得长序列推理的 **计算成本和内存开销过高**，严重限制了其在生产环境中的部署。

### 🚀 提出的新方法：LISA (Linear-Indexed Sparse Attention)
LISA 是一种**即插即用**（plug-and-play）的注意力模块替代方案，无需从头预训练即可集成到现有模型中。其核心设计为一个**双分支混合架构**：

1. **Linear Attention 分支**  
   - 引入一个轻量级的线性注意力模块，以 $O(n)$ 时间复杂度维护全局上下文状态，提供长距离记忆能力。
   - 利用 **Test-Time Training (TTT)** 性质，在推理过程中动态更新状态矩阵。

2. **Sparse Self-Attention 分支 + Lightning Indexer**  
   - 设计一个轻量级 **Indexer** 模块，动态选择上下文中最重要的前 $M$ 个 token，送入固定大小的稀疏自注意力模块。
   - Indexer 通过可学习的 query-key 投影和 ReLU 点积实现高效重要性评分。

3. **门控融合机制**  
   - 使用可学习的门控参数 $g$ 融合两个分支输出：
     $$
     O_{\text{LISA}} = g \cdot O_{\text{LA}} + (1-g) \cdot O_{\text{SA}_M}
     $$

4. **两阶段训练策略**
   - **Stage 1**: 冷启动线性注意力，使用滑动窗口（sliding window）作为初始稀疏注意力，并通过 **cross-entropy loss** 避免 KL 散度过拟合。
   - **Stage 2**: 引入 Indexer 替代静态窗口，采用 **per-head KL divergence loss** 对齐教师模型的注意力分布，确保稳定的重要 token 选择。

### 🔍 相比现有方法的优势
| 方法类型 | 代表工作 | 局限性 | LISA 的优势 |
|--------|--------|------|-----------|
| **CoT 压缩** | LightThinker, INFTYTHINK | 可能丢失关键推理信息 | 不压缩输出 CoT，仅加速内部推理过程 |
| **KV Cache 压缩** | H2O, SnapKV, StreamingLLM | 主要减少内存，未显著降低计算量 | 同时降低 **计算复杂度** 和 **内存消耗** |
| **纯线性注意力** | Zhang et al. [2026] | 上下文索引能力弱，精度下降明显 | 结合稀疏注意力保留精确检索能力 |

> ✅ **核心优势总结**：将推理复杂度从 $O(n^2)$ 降至 $O(nM)$（$M < n$），实现**效率与性能双提升**。

---

## 2. 核心实验方法和设置

### 📚 数据集
在五个数学推理基准上进行评估：
- **GSM8K**：小学数学应用题
- **MATH-500**：高难度数学竞赛题子集
- **AMC23**, **AIME24**, **AIME25**：美国数学竞赛（AMC/AIME）题目

### ⚙️ 实验设置
- **基础模型**：基于 `Qwen2.5-1.5B` 和 `Qwen2.5-7B` 系列模型
- **初始化**：使用 DeepSeek-R1 蒸馏版本权重初始化
- **训练数据**：从 OpenR1-Math-220K 中随机采样 100K 条样本
- **超参数**：
  - $M = 256$（稀疏注意力窗口大小）
  - Stage 1：2 epochs, LR=2e-5, batch=128
  - Stage 2：1 epoch, LR=2e-5, batch=64
- **硬件**：NVIDIA H100 GPU，每阶段约 240 GPU 小时

### 📊 评估指标
| 指标 | 定义 |
|-----|------|
| **Accuracy (Acc)** | 正确答案比例 |
| **Token Number (Tok)** | 平均 CoT 序列长度 |
| **Reasoning Latency (ReL)** | 单样本平均推理时间（A100, BF16, batch=1） |

### 🔁 基线方法对比
| 类型 | 基线方法 |
|------|---------|
| **Efficient Reasoning** | LightThinker, INFTYTHINK, Zhang et al. [2026] |
| **KV-Cache Reduction** | H2O, SapLLM |
| **原始模型** | Qwen2.5-Base（无优化） |

所有基线均使用 LoRA 微调，保持可训练参数量相近。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

#### 在 **Qwen2.5-7B** 上的结果：
| 方法 | AVG Acc (%) | ReL (相对延迟) |
|------|-------------|----------------|
| Base Model | 65.9 | 314.2 |
| Zhang et al. [2026] | 68.3 | 263.3 |
| **LISA (Ours)** | **70.6** | **226.6** |

- ✅ **平均准确率提升 5.6%**（从 65.9 → 70.6）
- ✅ **推理延迟降低 50%**（ReL 从 314.2 → 226.6）
- ✅ 在 **AIME25** 上达到 40.0%，显著优于 Zhang et al. [2026] 的 33.3%

#### 在 **16K 上下文长度下**：
- 实现 **50% 的推理速度提升**
- 同时保持甚至超越完整 self-attention 的推理质量

### 🔁 与基线方法对比
| 对比维度 | LISA 表现 |
|--------|----------|
| **vs. H2O/SapLLM** | 更低延迟 + 更高准确率 |
| **vs. LightThinker/INFTYTHINK** | 不压缩输出，避免信息丢失风险 |
| **vs. Zhang et al. [2026]** | 显著更优的长序列建模能力（尤其在 AIME 等复杂任务上） |

### 🔍 消融实验与分析（Section 4.5）

#### （1）Linear 与 Sparse Attention 的协同效应
- 计算两者输出的 **余弦相似度** 发现始终为负（约 -0.27 ~ -0.3），表明二者表征互补、非冗余。
- 支持“线性分支提供全局修正信号，稀疏分支负责局部精细推理”的假设。

#### （2）Indexer 成功恢复原 attention 动态
- 可视化 Indexer 选出的 top-256 token，发现其集中于：
  - 当前 token 的**局部邻域**（recency bias）
  - **句首/句尾**（attention sink）
  - **标点符号**等语义锚点
- 表明 Indexer 学会了模仿 full attention 的关键模式。

#### （3）MASS 指标验证选择质量
定义：
$$
\text{MASS}(q,M) = \frac{\sum_{k \in \text{top-}M} \text{Attn}(q,k)}{\sum_{k<q} \text{Attn}(q,k)}
$$
- 当 $M=256$ 时，MASS 达到 **80%**，接近理论上限（86%）
- 表明仅用 256 个 token 即可覆盖绝大多数注意力质量

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **LISA 实现了效率与性能的双赢**：
   - 将推理复杂度从 $O(n^2)$ 降为 $O(nM)$
   - 在 16K 上下文下实现 **50% 推理加速**
   - 在多个数学推理任务上 **平均准确率提升 5.6%**

2. **双分支设计有效且必要**：
   - Linear Attention 提供稳定的长程依赖建模
   - Sparse Attention + Indexer 保证关键信息不丢失
   - 门控机制实现动态平衡

3. **无需修改原模型结构或输出格式**：
   - 插件式设计，易于部署
   - 仅新增少量可训练参数（via LoRA）

### ⚠️ 局限性
1. **实验范围有限**：目前仅在数学推理任务上验证，尚未拓展至代码生成、多轮对话等场景。
2. **两阶段训练较复杂**：相比标准 SFT，需 careful tuning 超参数和冻结策略。
3. **主干参数冻结**：未对原始模型参数微调，可能错失进一步优化空间。
4. **仍保留部分原始 attention 层**：每 6 层保留一次 full attention，约占总 FLOPs 的 70%，尚未实现完全替换。

### 🔮 未来工作方向
1. **自适应 token 预算策略**：根据输入复杂度动态调整 $M$
2. **全层 LISA 替换**：探索移除所有原始 attention 层的可能性
3. **扩展至更多任务**：应用于 code generation、agent planning、multi-hop QA 等
4. **联合微调探索**：解冻主干网络，进行端到端 fine-tuning 以进一步提升性能

---

> 💡 **一句话总结**：  
> **LISA 通过“线性记忆 + 稀疏检索”的混合架构，在不牺牲推理能力的前提下，实现了长上下文推理的高效化，是迈向实用化长链推理模型的重要一步。**

</details>

---

### 3. [PyroDash: Cost-Efficient Token-Level Small-Large Language Model Collaborative Inference](https://arxiv.org/abs/2607.20327)

**Authors**: Niqi Lyu, Pengtao Shi, Wei Qiu, Jianlin Zhong, Sicong Xia, Jianyao Ma, Yicheng Ding  
**Category**: cs.CL  
**Published**: 2026-07-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.20327v1  

#### Abstract
Large language models (LLMs) provide strong reasoning capabilities but are expensive to serve at scale, whereas small language models (SLMs) are cheaper but less reliable on difficult problems. We introduce PyroDash, a cost-aware framework for token-level SLM-LLM collaborative inference. During gene...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PyroDash: Cost-Efficient Token-Level Small-Large Language Model Collaborative Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（**LLMs**）虽然推理能力强，但服务成本高昂；小型语言模型（**SLMs**）成本低，但在复杂任务上可靠性不足。如何在保证推理质量的同时显著降低推理成本，是当前部署中的核心挑战。

现有方法如 **request-level routing** 或 **speculative decoding** 存在以下问题：
- **无法动态响应中间推理难度**：决策在生成前做出，不能应对“输入简单但推理过程复杂”的情况。
- **依赖额外组件**：需要独立的路由模型（router）、访问 LLM logits 或权重。
- **未显式优化实际计费成本**：多数方法未将 API 计费模型（如输入/输出 token 分别计价）纳入训练目标。

### 提出的新方法：PyroDash
PyroDash 是一种**成本感知、token-level 的 SLM-LLM 协同推理框架**，其核心思想是：
- 在自回归生成过程中，**SLM 自主决定是否请求 LLM 协助**，通过生成一个特殊控制 token `Toff` 触发协作。
- 一旦检测到 `Toff`，**Collaborate Engine** 将原始查询和 SLM 已生成的部分推理轨迹打包，发送给**冻结的 LLM** 进行一次性补全（single handoff）。
- 整个路由策略内化于 SLM，无需外部 router，也无需 LLM 微调或访问其内部状态。

### 创新点与优势
| 特性 | PyroDash | 传统方法 |
|------|---------|--------|
| **路由粒度** | Token-level | Request-level |
| **路由主体** | 内置于 SLM | 外部 router |
| **LLM 状态** | Frozen，无需微调 | 可能需微调或访问 logits |
| **手递次数** | 最多一次（one-shot） | 可能多次切换 |
| **成本建模** | 显式优化计费成本（prefill + decoding） | 多基于 token 数或延迟 |
| **API 兼容性** | 支持黑盒商业 API | 通常需白盒访问 |

---

## 2. 核心实验方法和设置

### 数据集
- **EasyHard-24k**：用于训练阶段 1 和 2，包含 24,061 条数学推理样本，按 SLM 是否能正确解答分为 **easy** 和 **hard** 子集。
- **DAPO-Math**：用于 Stage 3 的 GRPO 强化学习训练。
- **测试基准**（5 个数学推理数据集）：
  - **GSM8K**：小学数学应用题
  - **Minerva**：技术领域定量推理
  - **OlympiadBench**：奥赛级科学问题
  - **AIME25 & AIME24**：竞赛级数学题

### 实验设置
- **SLM**：Qwen3.5-4B
- **LLM**：GLM-5.2-FP8（冻结）
- **部署架构**：vLLM + 自研 Collaborate Engine
- **控制 token**：`Toff` 被注册为 SLM 采样的停止序列

### 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy (Acc.)** | Pass@1 或 Avg@32 下的答案正确率 |
| **LLM Token Ratio (%)** | $\frac{\text{LLM 输出 token 数}}{\text{SLM + LLM 输出 token 数}}$ |
| **Avg. LLM Calls** | 每个样本平均调用 LLM 次数 |
| **Cost ($)** | 基于公开价格计算的总美元成本：<br> Qwen3.5-4B: \$0.05/M in, \$0.08/M out<br> GLM-5.2-FP8: \$0.90/M in, \$2.86/M out |

### 基线方法对比
| 基线 | 类型 | 说明 |
|------|------|------|
| **Qwen3.5-4B** | SLM-only | 仅用小模型 |
| **GLM-5.2-FP8** | LLM-only | 仅用大模型（成本基准） |
| **SLM + SFT** | 冷启动 | 经过 Stage 1-2 训练，无 GRPO 优化 |
| **RouteLLM** | Request-level | 基于预测难度在解码前路由 |
| **GlimpRouter** | Step-level | 基于首 token 熵值触发 LLM，可多次调用 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1 & 2）

| Method | Avg. Acc. (%) | LLM Token Ratio (%) | Avg. LLM Calls | Cost ($) |
|--------|----------------|----------------------|----------------|----------|
| **GLM-5.2-FP8 (LLM-only)** | 57.68 | 100.00 | 1.000 | **49.36** |
| **PyroDash (λ=0.05)** | **64.04** ↑ | 95.34 | 0.975 | 39.29 ↓ |
| **PyroDash (λ=0.6)** | 54.55 | **1.90** ↓ | **0.012** ↓ | **1.78** ↓ |

#### 核心发现：
- **高精度模式（λ=0.05）**：
  - 平均准确率 **64.04%**，**超过 LLM-only 基线 6.36 个百分点**。
  - 成本从 \$49.36 降至 \$39.29，**节省 20.4%**。
- **低成本模式（λ=0.6）**：
  - 准确率 54.55%，略低于 LLM-only，但**成本骤降 96.4%**（\$49.36 → \$1.78）。
  - **LLM token ratio 仅 1.90%**，平均每样本调用 LLM 0.012 次，几乎不依赖 LLM。

### 与基线方法对比
- **相比 RouteLLM 和 GlimpRouter**：
  - 两者 LLM token ratio >75%，成本分别为 \$44.62 和 \$31.61。
  - PyroDash (λ=0.6) 以更低成本（\$1.78）实现更高准确率（54.55% vs 52.74%/54.20%），且 LLM 使用量减少约 **97.5%**。

### 消融实验结果（Table 3）

| λ | Avg. Acc. (%) | LLM Token Ratio (%) | Cost ($) |
|----|----------------|----------------------|----------|
| 0.05 | 64.04 | 95.34 | 39.29 |
| 0.1 | 55.29 | 8.19 | 4.71 |
| 0.2 | 54.91 | 3.18 | 2.55 |
| 0.6 | 54.55 | 1.90 | 1.78 |

- **λ 控制效果显著**：随着 λ 增大，模型更倾向于节省成本，LLM 使用率急剧下降，而准确率保持在合理区间（54–55%）。
- **最大收益区间**：λ 从 0.05 → 0.1，成本从 \$39.29 → \$4.71，降幅达 88%，是性价比最高的调整。

---

## 4. 关键结论和发现

### 主要结论
1. ✅ **Token-level handoff 有效**：SLM 能从自身生成轨迹中识别能力边界，并在适当时机请求 LLM 协助，实现高质量推理。
2. ✅ **成本-精度权衡可控**：通过调节效率惩罚系数 λ，可在**高精度**与**极低成本**之间灵活切换，满足不同部署需求。
3. ✅ **优于现有路由方法**：在相同或更低成本下，PyroDash 实现更高准确率，且 LLM 使用率远低于 request-level 或 step-level 方法。
4. ✅ **兼容商业 API**：无需 LLM 微调或访问 logits，适用于 GPT-4、Claude 等闭源模型。

### 方法局限性
- **决策透明性不足**：尚不清楚 SLM 是否在“真正困难”时才 handoff，缺乏对触发位置的细粒度分析。
- **奖励归一化影响未验证**：未对比 normalized cost penalty 与 raw token cost penalty 的差异。
- **任务范围有限**：目前仅在数学推理任务上验证，未扩展至代码生成、工具调用或多模态场景。
- **单次 handoff 限制**：不允许 multi-turn 协作，可能限制复杂交互任务的表现。
- **成本为估算值**：基于公开价格计算，未与真实账单对比。

### 未来工作方向
- 分析 `Toff` 触发位置与具体推理失败类型的关联。
- 探索多轮 handoff 机制，支持更复杂的协同推理流程。
- 扩展至其他任务领域（如 code generation、agent planning）。
- 验证不同 SLM-LLM 组合下的泛化能力。
- 将 PyroDash 思想推广为“模型服务网格”（model service mesh），实现异构模型的动态编排。

--- 

> **一句话总结**：  
> PyroDash 通过将 token-level 协同决策内化于 SLM，并结合 GRPO 进行成本感知强化学习，在数学推理任务上实现了 **高达 96.4% 的成本削减**，同时保持甚至超越纯 LLM 的推理性能，为高效、经济的大模型部署提供了新范式。

</details>

---

### 4. [Efficient Clustering with Provable Guardrails for LLM Inference at Scale](https://arxiv.org/abs/2607.19704)

**Authors**: Longshaokan Wang, Wai Tsang Keung, Punit Ghodasara, Roman Wang, Ali Dashti, Francesc Moreno-Noguer  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.19704v1  

#### Abstract
Scaling LLM-based applications to millions of users is bottlenecked by the inference cost and latency of modern foundation models. A natural fix is to cluster the inputs and call the LLM only on cluster representatives, letting other members inherit the output -- but this is only safe if each member...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Clustering with Provable Guardrails for LLM Inference at Scale**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代大型语言模型（LLM）在推理时面临高昂的成本和延迟，尤其是在服务数千万用户规模的电商推荐等场景中。直接为每个用户调用 LLM 进行个性化生成不可持续。

一个自然的优化是**对输入进行聚类，仅对簇代表调用 LLM，其余成员继承输出**。然而，这种近似必须满足严格的质量控制（guardrails），否则可能导致不相关甚至危险的推荐（如向有幼儿的家庭推荐窒息风险玩具）。

现有聚类方法无法同时满足以下四个要求：
1. **样本与其代表之间的语义相似度 ≥ 阈值 α**（质量保障）
2. **簇内所有样本具有相同的分类属性**（如家庭构成、儿童年龄等安全约束）
3. **簇数量远小于原始样本量**（实现显著的数据压缩）
4. **算法可扩展至 n ~ 10⁷ 规模，且运行时间和内存可控**

### **提出的新方法**
作者提出一种**两阶段聚类算法（Two-Stage Clustering Algorithm）**：

- **Stage 1: 初始聚类（Initial Clustering）**
  - 使用 **Mini-batch K-Means** 将嵌入（embeddings）划分为 K 个初始簇 `{Ck}`。
  - 目的是粗略分组，降低后续计算复杂度。

- **Stage 2: 贪心代表选择（Greedy Representative Selection）**
  - 在每个初始簇 `Ck` 内部，基于“匹配关系” `i ~α j ⇔ fsim(Ei, Ej) ≥ α ∧ Ai = Aj` 构建 o-ball（即 α-ball）。
  - 应用 **Johnson-Chvátal 启发式算法** 求解受限于 `Ck` 的 **Set Cover 问题**：每次选择能覆盖最多未被覆盖样本的点作为代表。
  - 可选后处理步骤：**Reassignment**，将每个样本重新分配给与其最相似且满足 guardrail 的代表，提升平均相似度。

### **相比现有方法的优势**
| 维度 | 本文方法 | 现有方法（如 K-Means, Spectral 等） |
|------|--------|-------------------------------|
| **质量保障（Guardrails）** | ✅ 严格保证最小相似度和属性一致性 | ❌ 不支持显式约束，部分样本可能违反阈值 |
| **可扩展性** | ✅ 时间复杂度 O(nd + n²d/K)，当 K=O(n) 时线性于 n；内存 O(nd + n²/K²) | ❌ 多数方法需 O(n²) 或更高内存（如 Spectral 需 O(n³) 时间） |
| **效率与速度** | ✅ 比标准方法快 10–1000×，可在 38M 数据上运行 | ❌ 在 n > 10⁵ 后变得不可行（如 Agglomerative, Spectral） |
| **理论保证** | ✅ 提供 (1 + ln \|Ck\|)-近似于最优 Set Cover 解，并精确满足 guardrail | ❌ 无此类理论边界 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **内部数据集：Shopping Personas（购物画像）**
   - 来自 Amazon 的 38M 用户购物行为构建的 persona。
   - 包含文本描述和分类属性（成人性别/数量、是否有儿童、儿童性别/年龄）。
2. **公开数据集（用于基准测试）**
   - **AG News**：新闻分类数据集，100K 样本。
   - **Cosmopedia**：教育类网页文本数据集，100K 样本。

> 注：由于多数基线方法无法处理大规模数据，主实验在 **100K 子集** 上进行公平比较。

### **实验设置与评估指标**

#### **评估指标**
| 指标 | 定义 | 说明 |
|------|------|------|
| **Avg. Sim.** | 平均样本到其代表的 cosine similarity | 衡量整体语义一致性 |
| **Min. Sim.** | 最差样本到其代表的 similarity | 关键指标，是否满足 α 阈值 |
| **Perc. Below Thresh.** | similarity < α 的样本比例 | 直接反映 guardrail 是否被破坏 |
| **Time (sec.)** | 总运行时间（单机 r7i.12xlarge，无并行化） | 衡量效率 |
| **#Clusters / Data Reduction Fold** | 最终簇数及压缩倍数 | 衡量下游 LLM 调用减少程度 |
| **Cluster-size skew** | 簇大小分布偏度 | 影响尾部裁剪（tail-trimming）潜力 |

#### **基线方法对比**
共对比六种主流聚类方法（均来自 scikit-learn 实现）：
- **Mini-batch K-Means**
- **K-Means**
- **Agglomerative Clustering**（Ward linkage）
- **BIRCH**
- **Spectral Clustering**
- **Gaussian Mixture Model (GMM)**

> 所有基线均调整簇数以匹配本文方法的结果，确保在相同压缩率下比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**

| Method | Dataset | Avg. Sim. | Min. Sim. | % Below α | Time (sec) |
|-------|--------|----------|-----------|------------|-------------|
| **Proposed** | Shopping Personas | 0.802 | **0.750** | **0.0%** | **2.7** |
| Mini-batch K-Means | Shopping Personas | 0.819 | 0.554 | 6.1% | 44 |
| K-Means | Shopping Personas | 0.828 | 0.600 | 2.9% | 154 |
| Agglomerative | Shopping Personas | 0.812 | 0.542 | 10.3% | 1778 |
| **Proposed w/ Reassign.** | Shopping Personas | **0.825** | **0.750** | **0.0%** | **2.7** |

> α 设置为 0.75（Shopping Personas）、0.3（AG News）、0.4（Cosmopedia）

#### **核心发现**
- **所有基线方法都违反了 guardrail**：3–21% 的样本未达到指定相似度阈值。
- **本文方法始终满足 α 要求（0.0% 违规）**，且运行速度快 **10–1000 倍**。
- 加入 **Reassignment 步骤** 后，平均相似度进一步提升（↑0.023），仍保持最小相似度和运行时间不变。

### **与基线方法的对比结果**
- **速度优势明显**：
  - 在 AG News 上，Spectral Clustering 耗时 **6281 秒**，而本文方法仅 **4.2 秒**（>1500× 加速）。
  - 即使轻量级 Mini-batch K-Means 也比本文慢 10–20 倍。
- **可扩展性强**：
  - 本文方法成功应用于 **38M 用户**，而大多数基线在 100K 以上已难以运行。
- **尾部裁剪能力更强**：
  - 本文方法产生的簇大小分布高度右偏（heavy-tailed）。
  - **前 4% 的簇覆盖了 90% 的用户** → 可通过裁剪尾部实现额外 **25× 数据压缩**。
  - 基线方法因追求均衡簇，同等条件下仅能覆盖约 30% 用户。

### **消融实验结果（Appendix F.4）**
- **初始簇数 K 的影响**：
  - K ↑ → Stage 2 计算更快、内存更低（因每簇更小）。
  - 但过大的 K 会引入“分区噪声”，导致相似用户被分到不同初始簇，降低最终压缩效率。
  - 实践建议：选择能满足内存/延迟预算的最小 K。

- **Reassignment 的权衡**：
  - 提升平均相似度（+0.02–0.07），但略微削弱簇大小偏度 → 尾部裁剪效率下降。
  - 是 **average quality vs. compression efficiency** 的典型 trade-off。

---

## **4. 关键结论和发现**

### **主要发现**
1. **聚类 + LLM 推理压缩可行且高效**：
   - 通过对用户 persona 聚类，可在几乎不影响推荐质量的前提下，将 LLM 调用次数减少 **50 倍**。
2. **Set Cover + o-ball 视角有效结合了语义与规则约束**：
   - 将“代表选择”建模为 Set Cover 问题，天然支持最小相似度和属性一致性的硬性约束。
3. **两阶段设计兼顾效率与精度**：
   - 第一阶段粗聚类避免全局 O(n²) 计算；
   - 第二阶段局部贪心搜索确保理论近似比 `(1 + ln |Ck|)` 和 exact guardrail。
4. **簇大小偏度是实用优势**：
   - 自然形成的“幂律分布”允许通过保留头部大簇来进一步压缩数据，适用于 QA 成本高的场景。

### **方法的局限性**
- **依赖高质量 embedding model**：
  - 若 embedding 不能准确反映语义或推荐相关性，则相似度无法作为可靠 proxy。
- **Stage 2 仍是 O(n²d/K)**：
  - 虽优于全量 O(n²)，但在极端高维或极大 K 下仍有挑战。
- **初始聚类可能割裂相似样本**：
  - 若 Mini-batch K-Means 分得太碎，会影响最终压缩上限（见 Remark 7）。
- **静态聚类**：
  - 当前方法为离线批处理，未考虑动态更新或流式数据。

### **未来工作方向**
1. **更优的相似度度量**：
   - 设计能更好预测 LLM 输出一致性的 metric，而非仅依赖 embedding similarity。
2. **学习型代表（Learned Representatives）**：
   - 不再从原始样本中选代表，而是学习虚拟中心点以最小化簇数。
3. **在线/增量聚类机制**：
   - 支持实时新增用户加入已有簇结构。
4. **多目标优化框架**：
   - 联合优化压缩率、公平性、多样性等目标，在 guardrail 下寻找 Pareto 最优解。

---

> ✅ **一句话总结**：  
> 本文提出了一种**可证明保障质量（provable guardrails）的大规模聚类算法**，通过 **Mini-batch K-Means + Set Cover 贪心代表选择** 的两阶段设计，在 **38M 用户场景下实现了 50 倍 LLM 推理成本压缩**，同时保证每个用户与其代表的语义相似度不低于阈值且属性完全一致，**速度比传统方法快 10–1000 倍**，已在 Amazon 生产系统部署验证。

</details>

---

### 5. [How Fast Can Reward Models Score? A Systems Study of C++ and PyTorch Inference Runtimes for RLHF](https://arxiv.org/abs/2607.19712)

**Authors**: Venkata Naga Sai Vishnu Rohit Pulipaka, Anish Katta, Deva Rohit Reddy Peddireddy  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.19712v1  

#### Abstract
In RLHF pipelines, reward scoring blocks policy updates. Slow scoring bottlenecks the entire loop, since no update runs until every rollout gets a score. And yet most setups just default to PyTorch eager mode or torch.compile, no one checks if that's actually fastest. Scoring itself is small. Rollou...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：How Fast Can Reward Models Score? A Systems Study of C++ and PyTorch Inference Runtimes for RLHF

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **RLHF（Reinforcement Learning from Human Feedback）** 训练流程中，**reward model 的推理延迟** 是影响训练效率的关键瓶颈。尽管 reward scoring 是每个训练步的必要环节，且必须等待所有 rollout 被打分后才能进行 policy update，但大多数系统仍默认使用 `PyTorch eager mode` 或 `torch.compile`，缺乏对更高效推理后端的系统性评估。

本文首次从 **系统层面** 对比了不同推理运行时（runtime）在 CPU 和 GPU 上的实际性能表现，揭示了当前实践中存在的严重性能浪费。

### ✅ 提出的新方法与思路
作者构建了一个基于 **ONNX Runtime** 的 **native C++ 推理引擎**，用于执行 reward model 的前向推理。该引擎完全绕过 Python，实现了：
- **原生 C++ Tokenizer**（支持 SentencePiece 和 WordPiece）
- **完整的请求批处理（batching）逻辑**
- **零 Python 依赖的端到端推理**

其核心思想是：**将模型导出为静态图（ONNX），并通过轻量级、低开销的 C++ 运行时执行，以减少框架层开销**。

### ✅ 相比现有方法的优势
- 在 **CPU 上显著优于所有 PyTorch 基线**（包括 `torch.compile`）
- 验证了 **性能提升来自 ONNX Runtime 图执行本身，而非 C++ 语言优势**
- 揭示了 **naive batching（按到达顺序填充）会严重降低吞吐量**
- 发现 **并发请求共享实例无法提升吞吐，独立实例反而导致资源争用和 OOM**

> 🔥 创新点在于：不是简单地“用 C++ 替代 Python”，而是通过严谨的系统实验，识别出真正影响性能的关键因素——**执行模式（graph vs eager）** 和 **批处理策略**。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **Anthropic 的 `hh-rlhf` 数据集** 中的 prompts 和 responses。
- 主要评估集为 **60 行样本**（固定随机种子采样），覆盖多种响应长度。
- 为验证鲁棒性，额外使用两个不同种子的 60 行样本和一个 **150 行样本集** 进行复现测试。

### ⚙️ 实验设置
- **硬件环境**：
  - CPU: AMD Ryzen 7 5800H
  - GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6144 MiB VRAM)
- **软件版本**：
  - ONNX Runtime 1.26.0
  - PyTorch 2.10.0 (+cu126)
  - `torch.compile` 使用默认 Inductor 后端
  - 所有实验均运行在 **fp32 精度**
- **重复性保障**：
  - 每个配置进行 **多次独立进程启动**（5次为主，部分CPU基线为3次）
  - 报告 **均值 ± 95% 置信区间**，避免单次运行噪声误导结论
  - 使用 Welch’s t-test 验证统计显著性（p < .001）

### 📊 评估指标
- **p50 latency**（中位延迟）：每轮 launch 内所有行的中位延迟，再取跨 launch 的均值
- **p95 latency**（尾部延迟）：同上，但使用 95th 百分位数
- **throughput (rows/sec)**：单位时间内处理的样本数
- 所有指标均为 **跨独立进程 launch 的统计汇总**

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **C++ Engine (ONNX Runtime)** | 自研 C++ 引擎，调用 ONNX Runtime 执行 |
| **HF Eager (PyTorch)** | Hugging Face Transformers 默认 eager mode |
| **FastAPI + PyTorch** | 模拟真实服务场景，通过 HTTP 调用 reward model |
| **torch.compile** | PyTorch 原生图编译优化 |

此外还对比了：
- 不同 **batching 策略**：naive padding vs. length-aware bucketing
- 不同 **并发模式**：共享引擎实例 vs. 多实例并行

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table 1 & Figure 2）

| System | CPU p50 (ms) | GPU p50 (ms) |
|--------|---------------|--------------|
| **C++ Engine (ONNX)** | **335.9 [297.7–374.0]** | 27.4 [24.3–30.5] |
| HF Eager (PyTorch) | 602.4 [551.3–653.4] | 57.2 [50.0–64.4] |
| FastAPI | 581.6 [570.4–592.9] | 62.8 [52.8–72.9] |
| **torch.compile** | 628.8 [613.7–644.0] | **19.0 [17.6–20.3]** |

#### ✅ 结果分析：
- **CPU 上**：C++ ONNX 引擎 **快约 1.8x**，且置信区间无重叠，统计显著（p < .001）
- **GPU 上**：`torch.compile` 明显领先，比 C++ 引擎快 **~30%**，尾部延迟也更优

### 📉 尾部延迟表现（Table 2）
| System | GPU p95 (ms) |
|--------|----------------|
| C++ Engine | 116.2 [109.7–122.8] |
| **torch.compile** | **25.6 [18.4–32.9]** |

> ❗ `torch.compile` 在尾部延迟上优势更大，说明其不仅平均更快，且更稳定。

### 🔍 消融实验结果

#### （1）C++ 是否真的带来加速？
- 将同一 ONNX 模型用 **Python 调用 ONNX Runtime** 测试：
  - Python ONNX: ~349 ms
  - C++ ONNX: 335.9 ms
- **两者性能几乎一致** → 加速来自 **ONNX Runtime 的图执行**，而非 C++ 语言本身

#### （2）Tokenizer 性能差异
- C++ 原生 SentencePiece tokenizer: **64.3 μs**
- Python AutoTokenizer: **245.8 μs**
- C++ tokenizer 快约 **3.8x**，但占总延迟比例极小（<1%）

#### （3）Zero-copy / Buffer Preallocation
- 移除内存拷贝或预分配 buffer：
  - **未观察到可测量性能提升**
- 原因：这些操作耗时远低于 Transformer 前向传播（差 3–5 个数量级）

#### （4）Batching 策略影响（Table 4 & 5）
| Batch Size | CPU Throughput (rows/s) – Naive | – Bucketed |
|------------|-------------------------------|-----------|
| 1          | 3.10                          | —         |
| 8          | 0.40                          | 2.43      |

| Batch Size | GPU Throughput (rows/s) – Naive | – Bucketed |
|------------|-------------------------------|-----------|
| 1          | 39.8                          | —         |
| 8          | 10.0                          | 53.7      |

- **Naive batching 导致吞吐下降 5–8x（CPU）、3.5–4x（GPU）**
- **Length-aware bucketing 只在 GPU 上有效**（利用并行计算）
- CPU 上即使 bucketing 也无法超过 batch=1 性能（无 batch 并行）

#### （5）并发请求处理（Table 6 & 7）
- **共享引擎实例**：
  - 吞吐基本不变（仅提升 ~10%）
  - 请求串行化执行，延迟随并发线性增长
- **多实例并发**：
  - CPU：线程池冲突，性能下降
  - GPU：**并发=8 时全部 OOM**（VRAM 不足）
  - 实测内存上限约为 **4–5 个独立 session**

> 🚫 结论：**增加并发无法提升吞吐，唯一有效的扩展方式是 length-aware batching**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **真正的性能差距在于执行模式，而非编程语言**  
   - ONNX Runtime 的图执行显著优于 PyTorch eager mode
   - C++ 本身不提供额外加速，Python 调用 ONNX 同样高效

2. **GPU 上 `torch.compile` 全面胜出**  
   - 即使相比专用 C++ 引擎，`torch.compile` 在 **中位和尾部延迟** 上都更优
   - 对已使用 PyTorch 的团队，直接启用 `torch.compile` 是最优选择

3. **Naive batching 是广泛存在的性能陷阱**  
   - 按到达顺序 padding 会导致吞吐暴跌
   - 正确做法：**先按长度分桶，再组批**

4. **并发不能提升吞吐，反而有害**  
   - 共享实例无法并行
   - 多实例造成资源浪费和 OOM
   - **唯一可行的优化路径是 batching**

5. **测量方法至关重要**  
   - 单次运行极易受系统噪声干扰（如 CPU 频率、内存布局）
   - 必须采用 **多次独立进程启动 + 置信区间分析**

### ⚠️ 局限性
- 实验仅在 **单一开发机** 上完成（1 CPU + 1 GPU），未验证多卡、分布式场景
- 未测试 Triton Inference Server、vLLM 等生产级 serving 框架
- 模型仅限于两个 OpenAssistant reward models（DeBERTa-v3 和 Electra-large）
- 未在完整 RLHF 训练循环中测量端到端训练速度提升

### 🔮 未来工作方向
- 研究 `torch.compile` 在动态输入长度下的 **recompilation 开销与缓存命中率**
- 探索在 **server-grade 多核 CPU 和高显存 GPU** 上的扩展行为
- 构建支持 **自动 length bucketing + 动态 batching** 的 reward model serving 框架
- 将 C++ 引擎集成进主流 RLHF 框架（如 OpenRLHF），进行端到端训练加速实测

---

## 💡 总结一句话
> **Rewards 应该跑得快，但最快的不是 C++，而是正确的执行模式与批处理策略：在 CPU 上用 ONNX Runtime，在 GPU 上用 `torch.compile`，并始终避免 naive batching。**

📌 代码与数据已开源：[GitHub](https://github.com/vishnup22/reward-model-benchmarks)

</details>

---

### 6. [OpenEvoShield: Dual Non-Stationary Continual Defense for Open-World Multi-Agent System Attacks](https://arxiv.org/abs/2607.19351)

**Authors**: Litian Zhang, Chaozhuo Li, Yuting Zhang, Zejian Chen, Bingyu Yan, Qiwei Ye  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.19351v1  

#### Abstract
LLM-based multi-agent systems (LLM-MAS) are increasingly deployed in safety-critical applications, where adversaries inject malicious instructions through inter-agent communication to propagate harmful behaviors. Unlike static threats, these attacks are doubly dynamic: adversaries refine injection s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：OpenEvoShield: Dual Non-Stationary Continual Defense for Open-World Multi-Agent System Attacks**

---

## 1. **论文的主要贡献和创新点**

### **解决的问题**
该论文针对 **LLM-based Multi-Agent Systems (LLM-MAS)** 在开放世界部署中面临的安全挑战，提出了一种新的防御框架。传统防御方法通常假设环境是**封闭且静态的**（closed-world assumption），即攻击模式和正常行为分布固定不变。然而，在真实场景中，系统面临双重非平稳性（**Dual Non-Stationarity**）：

- **攻击策略持续演化**：攻击者会不断调整注入方式以绕过现有检测机制；
- **正常行为动态漂移**：随着系统扩展（如新增代理、工具或任务域），合法行为也会发生变化。

现有方法在面对这两种并发变化时，容易出现**灾难性遗忘**（catastrophic forgetting）、**误报率上升**或**无法识别新型攻击**等问题。

---

### **提出的新方法与创新思路**

作者提出了 **OpenEvoShield**，一个**协同进化式的持续防御框架**，其核心思想是将攻击侧和正常侧行为建模解耦，并赋予不同的学习速率，实现“快攻慢守”的适应策略。

#### 主要模块构成：
| 模块 | 功能 |
|------|------|
| **M1: Asymmetric Co-Evolutionary Rate Controller** | 实时计算双漂移信号（attack-side 和 normal-side），输出不对称的学习率 `α_att`（快）和 `α_norm`（慢），协调两个分支的更新节奏 |
| **M2: Normal-Side Boundary Updater** | 利用生成式奖励模型（GenPRM）和指数移动平均（EMA）缓慢更新正常行为边界，防止因快速适应而误判良性漂移 |
| **M3: Attack-Side Policy Updater** | 采用基于 **Elastic Weight Consolidation (EWC)** 正则化的策略集成，进行快速在线学习，避免遗忘历史攻击模式 |
| **M4: Open-World Multi-Granularity Detector** | 融合节点级、子图级和图级异常证据，并引入**能量函数（energy-based）的OOD检测头**，识别训练分布外的新型攻击 |

---

### **相比现有方法的优势**
| 维度 | OpenEvoShield | 传统方法（如 INFA-GUARD） |
|------|----------------|--------------------------|
| 非平稳性处理 | ✅ 显式建模双漂移并解耦更新 | ❌ 固定分布假设，无法适应变化 |
| 忘记问题 | ✅ EWC 正则化保持记忆稳定性 | ❌ 微调导致灾难性遗忘 |
| 新型攻击检测 | ✅ 能量函数支持 OOD 检测 | ❌ 仅能识别已知类别 |
| 多粒度感知 | ✅ 节点+子图+图级联合判断 | ❌ 多为单一级别检测 |
| 自适应能力 | ✅ 在线持续学习，无需重新训练 | ❌ 静态模型，需离线重训 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
实验在五个标准攻击基准上进行，涵盖三类典型 LLM-MAS 攻击：

| 数据集 | 攻击类型 | 描述 |
|--------|---------|------|
| **PI (CSQA/MMLU/GSM8K)** | Prompt Injection | 通过误导性提示诱导错误答案 |
| **TA (InjecAgent)** | Tool Exploitation | 利用间接提示注入操控工具调用 |
| **MA (PoisonRAG)** | Memory Poisoning | 污染检索记忆模块传播虚假信息 |

---

### **实验设置**
- **部署阶段划分（DNSD Setting）**：共100轮，分为三个阶段：
  1. **Phase I (Warm-up, 1–30)**：稳定初始分布
  2. **Phase II (Co-evolution, 31–70)**：每5轮注入新变种攻击，同时正常行为逐步漂移
  3. **Phase III (Open-world, 71–100)**：引入完全未知的攻击类型（OOD），继续行为漂移

- **通信拓扑结构**：测试四种 MAS 拓扑：链式（chain）、树形（tree）、星型（star）、随机图（random）

- **骨干模型**：Gemini 1.5 Flash-Lite 作为 LLM 代理基础模型

---

### **评估指标**
| 指标 | 含义 |
|------|------|
| **ASR@k ↓** | Attack Success Rate，越低越好 |
| **MDSR@k ↑** | MAS Defense Success Rate，越高越好 |
| **FPR ↓** | False Positive Rate，对良性代理的误报率 |
| **Novel-DR ↑** | Novel Attack Detection Rate，对 OOD 攻击的检出率 |

---

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Node-based** | AgentSafe, Challenger, Inspector |
| **Graph-based** | G-Safeguard, GUARDIAN, INFA-GUARD |
| **Continual Baseline** | Naive-Continual（EWC + INFA-GUARD，统一学习率） |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（来自 Table 1 & 2）**

#### **静态环境下性能（Phase I）**
| 方法 | 平均 ASR@3 ↓ | 平均 MDSR@3 ↑ |
|------|-------------|--------------|
| INFA-GUARD | 13.8% | 89.2% |
| Naive-Continual | 13.5% | 89.7% |
| **OpenEvoShield (Ours)** | **11.7%** | **90.8%** |

✅ 在初始阶段表现最优，说明无适应开销。

---

#### **动态演化环境（Phase II & III）**
| 方法 | Phase II MDSR ↑ | Phase III MDSR ↑ | Phase III Novel-DR ↑ | Phase III FPR ↓ |
|------|------------------|-------------------|------------------------|-----------------|
| INFA-GUARD | 68.5% | 52.8% | 3.1% | 9.2% |
| Naive-Continual | 75.8% | 65.5% | 19.3% | 8.5% |
| **OpenEvoShield (Ours)** | **87.5%** | **83.5%** | **61.8%** | **4.8%** |

📌 **核心优势体现**：
- MDSR 显著领先（+11.7% vs Naive-Continual）
- Novel-DR 提升超 **42.5个百分点**
- FPR 控制极佳，远低于其他方法

---

### **消融实验结果（Ablation Study）**

| 变体 | Phase III MDSR ↓ | Phase III Novel-DR ↓ | FPR ↑ |
|------|------------------|-----------------------|--------|
| Full Model (Ours) | 83.5% | 61.8% | 4.8% |
| w/o M1 (Rate Ctrl) | 71.0% | 49.2% | 9.8% |
| w/o M2 (Boundary) | 74.5% | 53.7% | 14.5% |
| w/o M3 (Policy) | 63.2% | 35.8% | 10.2% |
| w/o M4 (OOD Head) | 74.8% | 0.0% | 8.8% |
| Symmetric Rates | 77.2% | 55.3% | 8.5% |
| w/o EWC | 72.0% | 48.5% | 11.8% |

🔍 **关键发现**：
- **M3（攻击侧策略更新）最重要**：移除后 Novel-DR 下降 26%，MDSR 暴跌近20%
- **M1（速率控制器）控制误报**：移除后 FPR 几乎翻倍
- **M2（正常边界）影响最大误报**：无 M2 时 FPR 达 14.5%
- **M4 的 OOD 头决定泛化能力**：无此模块 Novel-DR 归零
- **对称学习率显著劣化性能**，验证“快攻慢守”必要性

---

## 4. **关键结论和发现**

### **主要发现**
1. **双重非平稳性是 LLM-MAS 安全的核心挑战**：必须同时应对攻击演化与正常行为漂移。
2. **解耦学习速率至关重要**：攻击侧需要快速响应，正常侧行为应缓慢演进以避免误判。
3. **多粒度融合 + 能量函数 OOD 检测** 是识别新型攻击的关键。
4. **协同进化设计优于朴素持续学习**：即使加入 EWC，若缺乏漂移感知与速率调节，仍难以维持长期性能。

---

### **方法的局限性**
1. **漂移建模依赖离散相位转换**：现实中的分布偏移可能更复杂、非单调，当前漂移估计器（M1）未必最优。
2. **GenPRM 推理开销较高**：使用 Qwen2.5-1.5B 作为评分模型，在延迟敏感场景下可能成为瓶颈。
3. **未考虑自适应攻击者**：实验假设攻击者不针对检测机制本身进行反制，未来需研究对抗性更强的威胁模型。

---

### **未来工作方向**
- 构建更具挑战性的 **adaptive adversary benchmark**
- 设计轻量化替代方案（如蒸馏版 GenPRM）提升效率
- 扩展至更多样化的 **real-world LLM-MAS 应用场景**
- 探索 **主动防御机制**，如诱骗攻击者暴露策略

---

> ✅ **总体评价**：  
> OpenEvoShield 是首个系统性解决 LLM-MAS 中 **Dual Non-Stationarity** 问题的持续防御框架。其实验设计严谨，模块分工明确，结果全面超越静态与朴素持续方法，在**检测新型攻击的同时有效控制误报率**，为开放世界下的多智能体安全提供了重要范式。

</details>

---

### 7. [EvoThink: Evolving Thinking in Large Reasoning Models via Self-Pruning and Aha-Moment Preference Optimization](https://arxiv.org/abs/2607.19962)

**Authors**: Xinbang Dai, Zheyu Xin, Huikang Hu, Lin Ren, Rihui Jin, Guohui Xiao, Guilin Qi, Kuicai Dong, Zhaocheng Du, Yuyang Zhang  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2607.19962v1  

#### Abstract
Large Reasoning Models (LRMs) often suffer from overthinking due to redundant verification steps. Existing approaches for mitigating overthinking, such as fast-slow thinking switching and reasoning trajectory compression, fail to make a fine-grained distinction between beneficial and redundant steps...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EvoThink: Evolving Thinking in Large Reasoning Models via Self-Pruning and Aha-Moment Preference Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型推理模型（**Large Reasoning Models, LRMs**）在处理复杂任务（如数学推理和代码生成）时普遍存在“**overthinking**”现象，即模型在推理过程中反复验证相同结论，产生大量冗余的原子推理单元（**Atomic Reasoning Units**），导致：
- 推理效率低下（token 消耗高）
- 推理能力可能因噪声积累而下降
- 现有方法（如 fast-slow 切换、轨迹压缩）无法细粒度区分有益与冗余步骤，常以牺牲推理能力为代价换取效率。

### **提出的新方法与新思路**
本文提出 **EvOTHINK** 框架，包含两个核心组件：

#### **(1) Self-Pruning Training (SPT)**  
- **无监督迭代训练**：将原始推理轨迹分解为 **Atomic Reasoning Units**，识别并剪枝重复的、未推进推理的单元（如反复验证同一中间结论）。
- **自训练机制**：在剪枝后的简洁轨迹上进行自训练，使模型内化高效推理模式。
- **优势**：无需人工标注或强监督信号，保留关键推理路径，显著降低 token 消耗。

#### **(2) Aha-Moment Preference Optimization (AMPO)**  
- **受遗传算法启发**：将失败的推理轨迹视为“种群”，通过基于**多样性**的 fitness 函数筛选出具有探索潜力的错误尝试。
- **构造 aha-moment 数据**：将高潜力错误路径与正确答案连接，合成“从错到对”的过渡样本（e.g., “But is this correct? I think I missed something.” → 正确推导）。
- **使用 DPO 训练**：让模型学习这种“顿悟”模式，提升其从错误中恢复并探索新路径的能力。
- **优势**：相比直接监督学习黄金答案，aha-moment 更符合模型自身分布，学习曲线更平滑，避免过拟合错误路径。

### **相比现有方法的优势**
| 方法类别 | 代表方法 | 局限性 | EvOTHINK 的改进 |
|--------|--------|------|----------------|
| Fast-Slow 切换 | DynaThink, Dualformer | 依赖二分类决策，易误判导致推理失败 | 不依赖预判，动态优化推理过程 |
| 轨迹压缩 | ThinkPrune, DIET | 无差别压缩，可能删去关键推理步骤 | 细粒度识别冗余单元，保留有效推理 |
| 监督学习 | SFT/DPO on gold answers | 黄金答案风格与模型原生推理差异大，泛化差 | 构造符合模型分布的 aha-moment，提升泛化 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **数学推理**：
  - **MATH-500**：标准数学问题基准
  - **AIME24 & AIME25***：更具挑战性的竞赛级数学题
- **代码生成**：
  - **TACO**：算法代码生成数据集

### **实验设置**
- **骨干模型**：
  - `Distill-Qwen-1.5B`
  - `DeepScaleR-1.5B`
  - `QwQ-32B`
- **训练框架**：veRL，使用 8×H100 GPU
- **最大 token 限制**：32,768（超限视为失败）
- **采样策略**：
  - AIME：每题采样 8 次
  - 其他：每题采样 4 次
- **温度与 Top-p**：temperature=0.6, top_p=0.95

### **评估指标**
- **Pass@1 (P@1)**：首次生成即正确的比例
- **#Tok**：平均输出 token 数量（衡量效率）

### **基线方法对比**
| 类别 | 方法 |
|------|------|
| 无压缩 | Vanilla |
| 监督微调 | Kimi 1.5 SFT, Kimi 1.5 DPO |
| 强化学习压缩 | O1-Pruner, ThinkPrune-2k, DIET |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| Backbone | Method | MATH-500 (P@1 / #Tok) | AIME24 (P@1 / #Tok) | TACO (P@1 / #Tok) |
|---------|--------|------------------------|----------------------|-------------------|
| DeepScaleR-1.5B | Vanilla | 80.3 / 3171 | 29.2 / 6436 | 4.2 / 7678 |
| | ThinkPrune-2k | 80.1 / **1838** | 26.7 / 3862 | 3.5 / 6117 |
| | **EvOTHINK SPT** | **80.2 / 1861** | 26.7 / 4321 | **6.4 / 6234** |
| | **EvOTHINK AMPO** | 84.7 / 3441 | 37.5 / 7044 | 7.7 / 8013 |
| | **EvOTHINK SPT+AMPO** | **82.1 / 2146** | **36.7 / 5287** | **7.4 / 6919** |

> ✅ **加粗**表示该列最优或次优；**SPT+AMPO 在多数指标上取得最佳平衡**

### **与基线方法的对比结果**
- **效率方面**：
  - SPT 将 DeepScaleR-1.5B 在 MATH-500 上的 token 数从 3171 降至 **1861**（↓41.3%），优于 ThinkPrune-2k（1838）且保持更高准确率。
- **性能方面**：
  - AMPO 显著提升推理能力：QwQ-32B 在 AIME24 上 Pass@1 从 46.7% 提升至 **55.8%**（↑9.1%）。
- **综合表现**：
  - **SPT+AMPO** 在保持高效（token 数接近压缩方法）的同时，显著优于所有纯压缩方法的准确率。

### **消融实验结果**
- **SPT 单独使用**：
  - 显著降低 token 消耗，推理性能与 Vanilla 相当甚至更好（如 TACO 上从 4.2 → 6.4）。
  - 表明**抑制冗余验证本身即可提升推理质量**。
- **AMPO 单独使用**：
  - 性能大幅提升，但 token 消耗略增（因引入更复杂的修正过程）。
- **SPT+AMPO 联用**：
  - 实现**性能与效率的最佳权衡**，验证两阶段互补性：SPT 提供高效基础，AMPO 在此基础上增强能力。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Overthinking 是推理质量问题而非单纯长度问题**  
   - 冗余的 Atomic Reasoning Units 占据超过 65% 的 token，但未带来性能提升。
   - 简单压缩会破坏有效推理，而**细粒度剪枝**（SPT）可安全去除冗余。

2. **aha-moment 学习优于直接监督**  
   - 图 5 显示，在 vanilla 模型失败的问题上，SFT 和 DPO 对黄金答案的学习效果有限。
   - AMPO 构造的“从错到对”数据显著提升模型在难例上的表现，说明**错误探索过程本身蕴含宝贵学习信号**。

3. **多样性是高质量失败的关键**  
   - 高 fitness（即更多不同局部结论）的失败轨迹更适合作为 mutation 数据源（图 6）。
   - 支持“**探索广度 > 单一路径深度**”的推理优化理念。

4. **方法具备良好泛化性**  
   - 在 MATH 上训练后迁移到 AIME（Table 2）：
     - SFT/DPO 导致严重性能退化（Pass@1 从 29.2% ↓至 4.2%/5.4%）
     - **EvOTHINK AMPO 反而提升性能**（30.0%），表明其学到的是**通用推理策略**而非死记硬背。

### **方法的局限性**
- **依赖模型生成多样失败路径**：若初始模型过于保守或收敛过快，可能难以生成足够多样的错误轨迹。
- **计算开销较高**：AMPO 需生成多个响应并筛选，训练成本高于标准 SFT。
- **fitness 函数较简单**：仅基于局部结论数量，未考虑语义质量或逻辑连贯性。

### **未来工作方向**
- 深入研究 **aha-moment 学习为何有效**：是否模拟了人类“试错-反思-突破”的认知过程？
- 设计更精细的 **fitness 函数**：结合语义多样性、逻辑跳跃性等指标。
- 扩展至 **多模态推理** 和 **Agent 规划任务**，验证其在复杂决策场景中的普适性。

---

> **总结**：EvOTHINK 通过 **SPT** 和 **AMPO** 两阶段设计，首次实现了**在不牺牲甚至提升推理能力的前提下，系统性减少 overthinking**。它不仅是一个压缩工具，更是一种**进化式推理训练范式**，为 LRM 的高效与智能协同优化提供了新方向。

</details>

---

### 8. [Solar Open 2 Technical Report](https://arxiv.org/abs/2607.20062)

**Authors**: Sungrae Park (University of Seoul), Sanghoon Kim (University of Seoul), Gyoungjin Gim (University of Seoul), Jungho Cho (University of Seoul), Hyunwoong Ko (University of Seoul), Minbyul Jeong (University of Seoul), Minjeong Kim (University of Seoul), Keunwoo Choi (University of Seoul), Chaehun Shin (University of Seoul), Chanwoong Yoon (University of Seoul), Dongjun Kim (University of Seoul), Eunwon Kim (University of Seoul), Gyungin Shin (University of Seoul), Hyeonju Lee (University of Seoul), Hyungkyu Kang (University of Seoul), Inseo Song (University of Seoul), Jisu Bae (University of Seoul), Jiyoon Han (University of Seoul), Jiyun Lee (University of Seoul), Joonkee Kim (University of Seoul), Junyeop Lee (University of Seoul), Mikyoung Cha (University of Seoul), Sangwon Yu (University of Seoul), Sehwan Joo (University of Seoul), Seokyoon Kang (University of Seoul), Seonghoon Yang (University of Seoul), Seung Shin (University of Seoul), Seunghyun Lee (University of Seoul), Seungseop Lim (University of Seoul), Seungyoun Shin (University of Seoul), Sukyung Lee (University of Seoul), Taegyeong Eo (University of Seoul), Taehwan Oh (University of Seoul), Taewhoo Lee (University of Seoul), Wonho Song (University of Seoul), Wonjun Oh (University of Seoul), Wonseok Hwang (University of Seoul), Yunsu Kim, Yura Shim, Hwalsuk Lee, Sunghun Kim, Du-Seong Chang, Kyunghyun Cho, Seungju Han, Yejin Choi, Junsuk Choe, Hwaran Lee, Minjeong Ban, Yun Taewon, Hwanjun Song, Jae-Gil Lee, KyungTae Lim, Alice Oh  
**Category**: cs.CL  
**Published**: 2026-07-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.20062v1  

#### Abstract
We present Solar Open 2, a 250B-A15B Mixture-of-Experts language model built for long-horizon agentic tasks, scaled up from Solar Open 1 (Solar Open 100B). To hold entire agent trajectories in a single context, Solar Open 2 reaches a 1M-token window through a hybrid attention stack that interleaves ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Solar Open 2 技术报告核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该研究旨在构建一个**主权AI模型（Sovereign AI）**，以支持韩国在长周期智能体（long-horizon agentic tasks）任务中的战略需求。具体解决了以下三个关键挑战：
- **长上下文能力不足**：前代模型 Solar Open 1 仅支持 128K 上下文，在复杂代理任务中无法容纳完整的交互轨迹。
- **韩语效率低下**：通用 tokenizer 在处理韩语文本时消耗更多 token，导致推理成本高、有效上下文变短。
- **缺乏高质量的代理训练场景**：现有数据集难以覆盖真实办公、编码和工具使用的复杂多轮任务。

### 提出的新方法与创新
#### （1）**Hybrid Attention Stack 架构**
- 采用混合注意力机制：每 4 层中包含 **1 层 Softmax Attention + 3 层 Linear Attention**，形成 [S-L-L-L]×12 的堆叠结构。
- **无位置编码（NoPE）**：完全移除 RoPE 等显式位置编码，依赖 Linear Attention 的递归状态隐式建模顺序。
- **负特征值扩展（Negative Eigenvalues）**：允许 Linear Attention 的状态转移矩阵具有负特征值（范围 [-1,1]），实现状态翻转与主动擦除，增强长期记忆纠错能力。
- **Sigmoid 输出门控**：在 Softmax 层引入逐头 sigmoid 门控，提升非线性表达力并抑制“attention sink”现象。

> ✅ **优势**：将上下文窗口从 128K 扩展到 **1M tokens**，同时内存和计算开销约为全 Softmax 的 **1/4**。

#### （2）**Selective Weight Transfer 初始化**
- 不进行完整蒸馏或随机初始化，而是选择性地将 Solar Open 1 中**架构兼容部分的权重直接迁移**。
- 仅约 **2.3% 参数（5.69B）被复用**，主要包括：
  - Token Embedding / Output Layer（词表一致）
  - Normalization Layers（稳定性）
  - Shared Expert
  - Softmax 层的 q/k/v/o 投影
  - Linear 层的部分 query/output 投影（通过 GQA 映射）

> ✅ **优势**：相比从头训练，达到相同训练损失所需 token 数减少 **~1.7×**，显著节省预训练成本。

#### （3）**Value-per-Token 数据策略**
- 构建 **20T → 10T 的高质量数据池**，通过质量评分、稀有度追踪和去重优化组合比例。
- 最终数据配方强调：
  - Real-to-Synthetic ratio: 4:6
  - Math & Code 各占至少 15%
  - 英语占比 >80%

> ✅ **优势**：在同等 token 预算下，新数据配方在多个基准上优于旧版，尤其在 `en_code` (+0.216) 和 `ko_math` (+0.204) 表现突出。

#### （4）**Multi-teacher On-Policy Distillation (MOPD)**
- 在后训练阶段，先独立训练 **12 个领域专家（specialists）**，再通过 MOPD 将其知识整合为单一模型。
- 学生模型基于自身 rollout 生成轨迹，由对应领域的教师模型提供 full-vocabulary KL 教学信号。
- 使用精确闭式 KL 计算，而非采样估计，避免方差干扰。

> ✅ **优势**：实现高效的能力融合，避免暴露偏差（exposure bias），且无需额外奖励工程，训练更稳定。

---

## 2. 核心实验方法和设置

### 使用的数据集

| 类别 | 主要数据集 |
|------|----------|
| **英文基准** | MMLU-Pro, GPQA-Diamond, HLE, LiveCodeBench v6, ArtifactsBench, HMMT26/AIME26, Multi-Challenge, IFBench, AA-LCR, SWE-Bench Verified, Terminal Bench Hard, APEX-Agents, MCP-Atlas, T3(banking), GDPval-AA v2 |
| **韩文基准** | KMMLU-Pro, CLIcK, HAE-RAE, Ko-AIME25, HRM8K, KBank-MMLU, KBL, KorMedMCQA, **Ko-GDPval**（自研） |

> 🔹 **Ko-GDPval** 是本文提出的关键新基准，用于评估韩语环境下经济价值高的办公任务完成能力，涵盖 11 个行业、58 种职业，要求输出 xlsx/docx/pptx/pdf 等格式交付物。

### 实验设置与评估指标

| 维度 | 设置说明 |
|------|--------|
| **模型规模** | MoE 结构，总参数 **250B**，激活参数 **15B/token**，专家数 320，共享专家 1 |
| **上下文长度** | **1M tokens**（通过 Hybrid Attention 支持） |
| **Tokenizer** | 继承 Solar Open 1 的韩语优化 BPE 分词器（vocab=196,608），在韩语文本上达 **4.41 bytes/token**，领先于所有对比模型 |
| **评估方式** | 所有结果由内部评测系统统一运行，保持 per-benchmark 设置一致；推理 effort 设置在表格中标注 |

### 基线方法对比

| 模型类型 | 对比模型 |
|---------|--------|
| **开源模型 (<320B)** | Solar Open 1 (102B-A12B), Command A+ (218B-A25B), Mistral Medium 3.5 (128B dense) |
| **先进开源模型** | MiMo-V2.5 (310B-A15B), DeepSeek-V4-Flash (284B-A13B) |
| **闭源快速API** | Claude Haiku 4.5, GPT-5.4 mini |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（取代表性指标）

| 基准 | Solar Open 2 | 最强基线 | 差距 |
|------|-------------|----------|-----|
| **MMLU-Pro** | **86.2** | DeepSeek-V4-Flash (85.9) | **+0.3** |
| **LiveCodeBench v6** | **92.4** | DeepSeek-V4-Flash (92.3) | **+0.1** |
| **APEX-Agents** | **16.6** | MiMo-V2.5 (13.4) | **+3.2** |
| **Ko-GDPval** | **86.8** | DeepSeek-V4-Pro (86.9) | **-0.1** |
| **Korean Avg.** | **85.4** | DeepSeek-V4-Flash (84.9) | **+0.5** |

> 📊 在 **12 个韩文基准平均得分**上，Solar Open 2 达到 **85.4**，超越所有开源及闭源快速 API 模型，排名第一。

### 与基线方法的对比结果

- **英文综合表现**：
  - 在 **MMLU-Pro、LiveCodeBench、APEX-Agents** 上领先同类开源模型。
  - 与最强模型 DeepSeek-V4-Flash 和 MiMo-V2.5 **基本持平或小幅领先**。
  - 在 SWE-Bench Verified（70.4 vs 73.8）和 Terminal Bench Hard（28.3 vs 41.7）仍有差距。

- **韩文全面领先**：
  - 平均分 **85.4**，高于 DeepSeek-V4-Flash（84.9）、GPT-5.4 mini（80.8）、Claude Haiku 4.5（69.6）。
  - 在 CLIcK（语言文化）、KBank-MMLU（金融）、KBL（法律）等专业领域均取得第一。
  - **Ko-GDPval 得分 86.8**，几乎追平 **1.6T 参数的 DeepSeek-V4-Pro（86.9）**，仅为其参数量的 **1/6**。

- **盲测偏好测试（835 轮韩语对话）**：
  - 用户偏好 Solar Open 2 的比例为 **44.1%**，远高于 DeepSeek-V4-Flash 的 **26.6%**。
  - “决定性胜利”（decisive win）比例为 **21.2% vs 8.5%**，表明其在事实准确性、流畅性和文化适配方面更具优势。

### 消融实验结果

| 实验项目 | 发现 |
|--------|------|
| **Architecture Ablation (10B proxy)** | Hybrid 架构比纯 GQA 架构达到相同 MMLU 0.55 所需训练 token 减少 **3.2×**（210B vs 671B） |
| **Selective Weight Transfer (200B proxy)** | 使用权重迁移后，达到 CE Loss=1.8 所需 token 减少 **1.7×**（12.6B vs 21.5B） |
| **Data Recipe Ablation (10B MoE)** | 新数据配方在等预算下全面优于旧版，最大增益来自 `en_code` (+0.216) 和 `ko_math` (+0.204) |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **主权AI可行路径验证**：通过韩语专用 tokenizer + 高价值数据 + 代理场景驱动训练，可在有限资源下打造具备国际竞争力的本地化大模型。
2. ✅ **长上下文可通过 Hybrid Attention 高效实现**：结合 Linear Attention 与少量 Softmax 层，可在极低开销下支持百万级上下文。
3. ✅ **Selective Weight Transfer 显著加速训练收敛**：即使架构变化较大，只要保留“共享骨架”，仍能有效利用前代知识。
4. ✅ **MOPD 是有效的多专家融合范式**：无需奖励叠加即可实现能力集成，适合复杂 agent 场景下的 post-training。

### 方法的局限性
- **软件工程能力仍有差距**：在 SWE-Bench 和 Terminal Bench 上落后于 DeepSeek 和 MiMo，反映其在代码仓库理解和终端错误恢复方面有待加强。
- **数值精度敏感任务存在波动**：在需要精细数字计算的办公任务中，偶尔出现单位换算或四舍五入误差。
- **MOPD 依赖高质量教师模型**：若某个专家未充分训练，则会影响最终整合效果。

### 未来工作方向
- 加强 **repository-level coding agent** 和 **terminal agent** 的训练，缩小与顶尖模型的差距。
- 引入更强的 **self-verification 和 self-correction 机制**，特别是在数值推理和合规文档生成任务中。
- 推广当前方法论至其他 **underserved languages**，探索通用的主权AI建设路径：
  - Tokenizer efficiency
  - Cross-generation selective transfer
  - Value-per-token data curation
  - Scenario-driven agent post-training with MOPD

> 💡 **最终结论**：Solar Open 2 成功实现了设计目标——在英语能力上媲美最先进开源模型，在韩语知识工作场景中实现领导地位，尤其是在 **Ko-GDPval** 这一高价值办公代理任务上，以不到六分之一的参数量匹配了 1.6T 模型的表现。

</details>

---

### 9. [SUM: Unified Geometric Surgery on Spatio-Temporal Adaptation Vectors for Federated Class Incremental Learning](https://arxiv.org/abs/2607.19384)

**Authors**: Jaeik Kim, Jaeyoung Do  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.19384v1  

#### Abstract
Real-world intelligent systems often require both distributed collaboration across data-isolated clients and continual adaptation to evolving tasks. This setting naturally gives rise to Federated Class Incremental Learning (FCIL), which combines Federated Learning (FL) and Continual Learning (CL). H...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SUM: Unified Geometric Surgery on Spatio-Temporal Adaptation Vectors for Federated Class Incremental Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对 **Federated Class Incremental Learning (FCIL)** 中存在的双重干扰问题：
- **Spatial Interference**：由于客户端数据非独立同分布（non-IID），导致不同客户端更新方向不一致，产生“客户端漂移”。
- **Temporal Interference**：在任务增量学习中，模型容易发生 **Catastrophic Forgetting**，即新任务的学习破坏旧任务的知识。

这两种干扰相互耦合，形成 **Spatial-Temporal Catastrophic Forgetting (ST-CF)**。现有方法通常分别处理空间和时间干扰，且多依赖于客户端侧的额外计算、通信或存储机制（如 replay buffer、distillation、正则化等），限制了其可扩展性和实用性。

### 提出了什么新方法或新思路
提出 **SURGERY & MERGE (SuM)** —— 一种**纯服务器端**的聚合时优化框架，将 FCIL 重新建模为一个统一的 **Multi-Task Learning (MTL)** 问题，其中：
- 客户端更新和任务更新均表示为共享参数空间中的 **Adaptation Vector**（适应向量）。
- 在服务器聚合阶段，对这些向量进行 **几何手术（Geometric Surgery）**，以消除冗余或冲突的方向分量。

#### 核心组件：
- **Spatial SuM**：在每一轮通信中，对来自多个客户端的更新向量进行去相关处理，缓解空间异构性带来的干扰。
- **Temporal SuM**：跨任务累积地对任务级适应向量进行在线投影（online projection），去除跨任务的冲突方向，防止灾难性遗忘。
- **Inference-ready Module Construction**：构建轻量化的任务专用推理模块，支持高效部署。

### 相比现有方法的优势
| 维度 | SUM 的优势 |
|------|-----------|
| **客户端开销** | ✅ 无任何额外开销（无需 replay、正则项、训练修改） |
| **通信效率** | ✅ 保持标准 FL 的通信模式（仅上传权重） |
| **计算效率** | ✅ 所有操作在服务器端完成，不增加客户端负担 |
| **通用性** | ✅ 可应用于任意基于参数聚合的 FL/CL 流程 |
| **效果提升** | ✅ 显著优于现有 FCIL 方法，甚至超越 centralized Joint Training |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖视觉与语言两大领域，共 **8 个基准数据集**：

#### 视觉领域（Vision Domain）
| 数据集 | 类别数 | 任务划分 | 特点 |
|--------|-------|---------|------|
| CIFAR-100 | 100 | 10 tasks × 10 classes | 小图像，in-domain |
| ImageNet-R | 200 | 10 tasks × 20 classes | 非真实风格图像（cartoons, sketches） |
| ImageNet-A | 200 | 10 tasks × 20 classes | 对抗样本，挑战性强 |
| EuroSAT | 10 | 5 tasks × 2 classes | 卫星遥感图像，out-of-domain |
| Cars-196 | 196 | 10 tasks (~20 classes/task) | 细粒度分类 |
| CUB-200 | 200 | 10 tasks × 20 classes | 鸟类细粒度分类 |

#### 语言领域（Language Domain）
| 数据集 | 类别数 | 任务划分 | 特点 |
|--------|-------|---------|------|
| 20-Newsgroups | 20 | 10 tasks × 2 topics | 新闻组文本分类 |
| CLINC-150 | 150 | 10 tasks × 15 intents | 多域意图识别，含 out-of-scope 查询 |

### 实验设置
- **客户端数量**：10
- **任务序列长度**：10（EuroSAT 为 5）
- **每任务通信轮次**：5 轮（部分为 10 轮）
- **数据划分**：使用 Dirichlet 分布模拟 non-IID 场景（β ∈ {0.05, 0.1, 0.5, 1.0}）
- **Backbone**：
  - Vision: ViT-B/16 (ImageNet-pretrained)
  - Language: T5-Small

### 评估指标
- **Final Averaged Accuracy (FAA)**：所有任务完成后，对所有已见类别进行平均准确率。
- **Forgetting Measure**：各任务在刚学完与最终阶段的性能差值。
- **Sharpness & Class Margin**：分析损失曲面平滑性和特征分离度。
- **Ablation Studies**：验证各模块有效性。
- **Scalability & Robustness**：测试客户端数量、模型大小、恶意客户端场景下的表现。

### 基线方法对比
涵盖四大类代表性方法：
| 类型 | 方法 |
|------|------|
| **FL-based** | FedAvg, CCVR, FedProto |
| **CL-based** | EWC, LwF, L2P, CODA-P |
| **Model Merging** | FisherAvg, RegMean |
| **FCIL-specific** | TARGET, PILoRA, FOT, LoRM |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（FAA 最高提升达 22%）

#### 表格摘要（部分关键结果）

| 方法 | CIFAR-100 (β=0.05) | ImageNet-R (β=0.05) | EuroSAT (β=0.2) | CUB-200 (β=0.2) | CLINC-150 (β=0.2) |
|------|-------------------|--------------------|------------------|------------------|-------------------|
| **Joint** | 92.75 | 84.02 | 98.42 | 86.04 | 95.53 |
| **LoRM (SOTA)** | 82.76 | 66.45 | 81.36 | 60.06 | 77.67 |
| **SUM (Ours)** | **96.32** | **86.81** | **98.57** | **79.43** | **94.11** |

✅ **SUM 在几乎所有数据集上达到 SOTA 性能，且在多个场景下超越 centralized Joint Training**。

### 与基线方法的对比结果
- 平均 FAA 提升高达 **22%**。
- 在极端 non-IID 设置下（如 β=0.05），SUM 仍保持稳定，而多数基线严重退化。
- 在 **fine-grained** 和 **out-of-domain** 数据上优势更明显，说明其对复杂语义结构有更好的建模能力。

### 消融实验结果（Ablation Study）

#### 表：CIFAR-100 上消融实验（β=0.05）

| 设置 | FAA (%) | Δ from Full Model |
|------|--------|------------------|
| 完整 SuM | **96.32** | — |
| 移除 Spatial Surgery | 78.93 | ↓17.39 |
| 移除 Temporal Surgery | 87.67 | ↓8.65 |
| 移除 Z-score Trimming | 85.87 | ↓10.45 |
| 移除 Sparsification | 90.23 | ↓6.09 |
| 移除 Elect-Sign | 94.58 | ↓1.74 |

📌 结论：
- **Spatial 和 Temporal SuM 是核心**，缺一不可。
- **Z-score Trimming** 对稳定性至关重要（防止异常坐标主导内积）。
- **Elect-Sign 和 Sparsification** 对构建高质量推理模块不可或缺。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **FCIL 可统一建模为 MTL 问题**：将客户端和任务更新视为共享参数空间中的 Adaptation Vectors，揭示了其方向交互的本质。
2. ✅ **方向性干扰是 ST-CF 的根源**：未调节的向量叠加会导致冗余或破坏性更新，影响优化路径。
3. ✅ **服务器端几何手术有效且高效**：SuM 在不改变客户端流程的前提下，显著提升了全局模型质量。
4. ✅ **优于 centralized Joint Training**：通过显式去除冲突方向，SuM 收敛到更平坦的最小值（flatter minima），并获得更大的类间间隔（larger class margin），从而泛化更好。
5. ✅ **具备强鲁棒性与可扩展性**：
   - 对客户端数量、模型规模（ViT-T → ViT-L）具有良好扩展性。
   - 在存在 **恶意客户端**（标签篡改）时仍保持高性能（FAA 80.5±9.6 vs LoRM 67.7±8.7）。

### 方法的局限性
| 局限 | 说明 | 缓解策略 |
|------|------|----------|
| **服务器存储开销** | 存储历史任务的 Adaptation Vectors，最坏情况为 O(TD) | 使用 Top-K 压缩、低精度（FP16/INT8）存储 |
| **聚合计算开销增长** | 随着任务增多，投影操作变慢 | 限制投影基数（如仅保留代表性方向） |
| **推理时需任务识别** | 推理模块依赖任务选择机制 | 引入轻量检索模块（如 key-value pool） |

### 未来工作方向
1. **动态压缩机制**：设计自适应的向量压缩算法，在长期持续学习中控制存储成本。
2. **自动化任务识别**：结合 prompt 或 adapter routing 实现免人工标注的任务匹配。
3. **理论深化**：建立更严格的收敛性与泛化误差界，解释为何“去干扰”能带来性能超越。
4. **跨模态 FCIL**：探索 SuM 在 multimodal continual learning 中的应用潜力。

---

> 📌 **总结一句话**：  
> **SuM 提出了一种全新的视角——将 FCIL 视为 spatio-temporal adaptation vectors 的方向调控问题，并通过纯服务器端的几何手术（Surgery & Merge）实现了高效、鲁棒、高性能的联邦类增量学习，是迈向实用化 lifelong federated intelligence 的重要一步。**

</details>

---

### 10. [Statistically Grounded Sparse-Feature Interventions for Activation-Space Control in Large Language Models](https://arxiv.org/abs/2607.19364)

**Authors**: Oshayer Siddique, J. M Areeb Uzair Alam, Md Jobayer Rahman Rafy, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.19364v1  

#### Abstract
Activation steering offers a lightweight alternative to fine-tuning for behavioral control of large language models, but SAE-based steering methods often rely on learned steering objectives or single-criterion feature selection. We introduce a transparent SAE-feature steering pipeline that first app...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Statistically Grounded Sparse-Feature Interventions for Activation-Space Control in Large Language Models*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Sparse Autoencoder (SAE)** 的 **Activation Steering** 方法在控制大语言模型行为时存在以下问题：
- 依赖于学习型目标（如优化器训练）或单一标准的特征选择，缺乏透明度；
- 特征选择过程引入了额外的超参数（如学习率、正则化权重），难以复现和审计；
- 评估往往只报告原始行为偏移（raw shift），而忽略了生成质量的退化，导致“可用的”（usable）控制被高估。

本文旨在提供一个**可审计、无学习优化、统计基础坚实**的 SAE 特征干预流程，以实现更可靠的行为控制。

---

### 提出的新方法或新思路
作者提出了一套**三阶段透明管道**（transparent pipeline）用于 SAE 特征引导：

#### （1）**共识驱动的特征选择（Consensus-ranked feature selection）**
- 对每个 SAE 特征应用**六个条件的质量过滤器**（six-condition quality filter），确保特征具有统计可靠性。
- 在通过筛选的特征上，使用三种互补统计量进行独立排序：
  - **F-test**：检测线性均值差异
  - **KSG Mutual Information (MI)**：捕捉非线性、阈值型依赖关系
  - **Cohen’s d**：标准化效应大小（有符号）
- 使用**未加权的 Borda 计数法**（unweighted Borda consensus）整合三个排名，避免引入可学习的聚合权重。

#### （2）**Fisher-LDA 启发的优化自由方向构建**
- 将最终的 steering vector 构建为 **Cohen’s d 加权的 SAE decoder 行向量组合**。
- 这一设计受 **Fisher Linear Discriminant Analysis (LDA)** 启发，在假设 SAE 特征近似去相关且方差相近时，Cohen’s d 权重近似等价于 Fisher 方向。
- **无需梯度优化**即可获得有效方向，消除了学习率、步数等额外选择。

#### （3）**质量条件下的评估协议（Quality-conditioned evaluation）**
- 引入“**clean success**”作为核心指标：要求同时满足：
  - 主要属性得分提升（△p > 0）
  - 相关性 ≥ 7，丰富性 ≥ 4，连贯性 ≥ 4
- 揭示了“原始偏移”与“可用成功”之间的巨大差距。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **透明性与可审计性** | 无学习优化步骤，所有统计阈值显式声明，特征可逐个审查 |
| **方法论简洁性** | 避免了优化器调参、损失函数设计等问题 |
| **评估严谨性** | 明确区分 raw shift 与 usable control，强调生成质量保留 |
| **跨配置泛化能力** | 广泛扫描 layer 和 α 强度空间，揭示 steering 的高度局部性 |

---

## 2. 核心实验方法和设置

### 使用的数据集
每种行为域构造 **800 个对比对（contrastive pairs）**，共 3,200 对用于特征选择：
- **SENTIMENT**：来自 IMDB，正负情感评论对
- **POLITICS**：来自 Twinviews-13k，左右立场对立文本
- **LOGIC**：来自 LogicBench，正确 vs 错误推理样本
- **MORAL**：来自 ETHICS + LLM 生成反例，伦理 vs 非伦理判断

评估使用 **100 个独立 hold-out prompts** 每域，来源不同以避免泄露。

---

### 实验设置
#### 模型
- **Gemma 2 2B**（26层）
- **Gemma 2 9B**（42层）
- **Gemma 3 4B**（34层）

#### SAE 配置
- 使用 **GemmaScope JumpReLU SAEs**（宽度 16,384）
- 接入 post-MLP residual stream
- 每层 max-pool token-axis 得到稀疏激活矩阵 $ Z \in \mathbb{R}^{1600\times16384} $

#### 注入方式
- 在最后一个 token 位置注入：  
  $ h_{\text{steered}} = h + \alpha \|h\| \cdot \hat{o}_h $
- 扫描 $ \alpha \in \{0.0, 0.1, ..., 2.0\} $
- 多层扩展：按质量正向配置加权分配 $ \alpha_{\text{total}} $

---

### 评估指标
采用 **三 judge 协议 + 人工裁决分歧**：
- Judges: Gemini 2.5 Flash, Gemini 2.5 Pro, GPT-5.4
- 每输出评分五维度（1–10分）：
  - **Primary**（主任务得分）
  - Relevance, Richness, Coherence, Factuality
- 最终裁决由人类评审员综合三方意见决定

#### 报告的关键指标：
| 指标 | 定义 |
|------|------|
| **△p** | 主属性得分变化（steered − baseline） |
| **△q** | 质量得分平均变化（relevance + richness + coherence） |
| **Success Rate (SR)** | △p > 0 的比例 |
| **Clean Success** | 同时满足 △p > 0 且 quality ≥ 阈值的比例 |

---

### 基线方法对比
与四种主流 activation steering 方法比较：
- **CAA**（Contrastive Activation Addition）
- **RePe**（Representation Engineering）
- **Top PC**（主成分分析）
- **ITI**（Inference-Time Intervention）

全部在同一 multi-judge 协议下评估。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（单层最佳配置）

| 模型 | 域 | 最大 △p | Clean Success (%) |
|------|-----|--------|------------------|
| Gemma 2 9B | LOGIC | **+1.160** @ L19, α=0.1 | 24% |
| Gemma 2 9B | MORAL | +0.700 @ L38, α=0.5 | 17% |
| Gemma 3 4B | POLITICS | +0.796 @ L22, α=2.0 | **34.7%** |
| Gemma 2 9B | SENTIMENT | +0.640 @ L26, α=0.3 | 17% |

> ✅ **最高 raw shift**：Gemma 2 9B 在 LOGIC 上达到 **+1.160**，是全扫描中最大值。

---

### 与基线方法的对比结果（Success Rate %）

| Model | Method | SENT | POL | MORAL | LOGIC |
|-------|--------|------|-----|--------|--------|
| Gemma 2 9B | CAA | 42.0 | 40.0 | 33.0 | 27.0 |
| | RePe | 17.0 | 20.0 | 18.0 | 14.0 |
| | ITI | 41.0 | 36.0 | 31.0 | 28.0 |
| | **Ours** | **38.0** | **37.8** | **45.0** | **36.0** |

> 📌 **结论**：
> - 在两个 **improvement-oriented domains**（MORAL, LOGIC）上，本方法全面优于所有 dense-vector 基线。
> - 在 polarity domains（SENTIMENT, POLITICS）上，dense 方法略优，说明这些信号更适合用密集方向捕获。

---

### 消融实验与关键发现

#### （1）多层 steering 结果（Multi-layer）
- 设计为预算受限（cumulative α ≤ 1.0），不追求 raw max。
- 结果显示：
  - **Gemma 2 9B LOGIC**：clean success 从 24% 提升至 **31%**（+7pp），尽管 raw △p 下降。
  - 其他多数情况下 multi-layer 会稀释效果。

> 🔍 **意义**：多层组合可在牺牲少量 raw shift 的前提下显著提升**可靠性**，适用于对质量敏感的场景。

#### （2）Raw vs Clean Success Gap
- 所有域中，“raw success”远高于“clean success”
- 差距范围：**4.1 至 28.0 个百分点**
  - 最大差距出现在 Gemma 2 9B MORAL（28.0pp）

> ⚠️ **警示**：仅看 raw 属性移动会严重高估实际可用的 steering 效果。

#### （3）失败模式分析
| 域 | 主要失败原因 |
|----|-------------|
| MORAL | repetition (57.4%), incoherence (44.1%) |
| LOGIC | incorrect (53.3%), repetition (29.9%) |
| POLITICS | incoherent (29.0%), irrelevant (21.8%) |
| SENTIMENT | repetition (52.3%), off-topic (41.0%) |

> 💡 **洞见**：大多数失败并非因目标属性未改变，而是由于**重复、不连贯、离题**等质量问题。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Steering 是高度局部化的**：
   - 最佳干预层（optimal layer）**既不跨模型统一，也不跨域一致**。
   - 例如：LOGIC 在 Gemma 2 9B 中峰值在 L19，但在 Gemma 3 4B 中为 L17。
   - Spearman 相关性检验显示 layer depth 与 △p 几乎无关（ρ ≈ -0.005）。

2. ✅ **α 强度响应是非单调的**：
   - 最强效果常出现在小 α（如 α=0.1），而非最大强度。
   - 大 α 易导致 out-of-distribution 输出，引发不连贯。

3. ✅ **Cohen’s d 初始化已接近优化上限**：
   - 与 SAE-SSV 的梯度优化路径相比，投影行为相似。
   - 支持“无需额外优化”的设计哲学。

4. ✅ **attribute shift ≠ usable control**：
   - “raw success”普遍高估“clean success”达 **4–28 pp**。
   - 必须联合报告质量指标，否则误导性强。

5. ✅ **多层组合提升 clean reliability**：
   - 在最强 domain（Gemma 2 9B LOGIC）中，multi-layer 将 clean success 提升 **+7pp**。
   - 适合质量优先的应用场景。

6. ✅ **方法家族应依 domain 选择**：
   - **Improvement domains**（MORAL, LOGIC）：sparse-feature 方法更强
   - **Polarity domains**（SENTIMENT, POLITICS）：dense-vector 更优

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **因果验证不足** | 特征选择基于统计关联，未通过 ablation 或 patching 验证因果性 |
| **依赖公开 SAE** | 方法有效性受限于 SAE 的训练质量和可解释性 |
| **英语单语评估** | 未测试跨语言迁移能力 |
| **离散评分限制显著性检验** | Judge 评分离散且高 tie rate（~44%），导致严格多重检验下无配置通过 FDR 标准 |
| **单轴控制** | 不支持 compositional steering（如同时调整 sentiment 和 politics） |

---

### 未来工作方向
1. **Compositional Steering**：研究多个属性间的交互机制，开发 principled 的向量组合策略。
2. **Principled α Budgeting**：在多层或多属性场景中，建立基于分布稳定性保证的扰动预算机制。
3. **细粒度 primary scoring**：改用连续评分（如 0–100）或更多 prompts 以增强统计效力。
4. **跨模型泛化研究**：将该共识框架迁移到 Llama、Mistral、Qwen 等其他架构家族。
5. **动态质量感知 steering**：开发能在线监测并调节生成质量的 adaptive injection 策略。

---

> 🧭 **结语**：  
> 本文提出了一种**统计严谨、无需优化、可审计**的 activation steering 新范式。其核心贡献不仅是技术本身，更是倡导一种新的评估文化——**必须将生成质量纳入 success 判定之中**。正如论文结尾所言：“What we can shift, and what we should, are rarely the same vector.”

</details>

---

### 11. [HyGRL: Adaptive Hybrid Graph Reasoning for Multi-Entity Questions](https://arxiv.org/abs/2607.19398)

**Authors**: Junyi Wang  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.19398v1  

#### Abstract
Multi-entity compositional questions pose significant challenges to existing retrieval-augmented language models. Conventional methods fall into a dilemma: standard RAG lacks dynamic reasoning, traditional Graph-RAG is limited by structural sparsity, and LLM-constructed Graph-RAG incurs prohibitive ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HyGRL: Adaptive Hybrid Graph Reasoning for Multi-Entity Questions

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
多实体组合型问题（multi-entity compositional questions）要求模型在多个实体之间进行多跳推理（multi-hop reasoning），而现有的 **Retrieval-Augmented Generation (RAG)** 方法面临以下瓶颈：
- **Standard RAG**：依赖向量相似度检索文本片段，缺乏动态推理能力，容易“丢失中间信息”（lost in the middle）。
- **传统 Graph-RAG**：受限于知识图谱（KG）的结构稀疏性，许多查询路径因缺失关系而中断。
- **LLM 构建的 Graph-RAG**（如 GraphRAG、LightRAG）：虽能从文本中提取三元组构建图，但消耗大量 LLM token，成本高昂且易丢失深层语义。

### 🚀 提出的新方法：HyGRL
提出 **HyGRL**（Hybrid Graph Reasoning with Reinforcement Learning），一个统一的框架，将非结构化文本与结构化知识图谱融合为异构图（heterogeneous graph），并引入基于强化学习的自适应推理机制。

#### 核心创新点：
1. **异构图构建（Heterogeneous Graph Construction）**
   - 不依赖昂贵的 LLM 三元组抽取，而是直接将原始 **text chunks** 作为节点嵌入图中，并通过实体链接连接到 KG。
   - 文本块作为“语义桥”（semantic bridge），缓解 KG 结构稀疏问题，无需完整图即可实现高连通性。

2. **自适应结构归纳推理（Adaptive Structure Induction）**
   - 将推理建模为 **Markov 决策过程（MDP）**，通过轻量级 MLP 策略网络指导证据检索。
   - 推理过程采用 **两阶段训练范式**：
     - **模仿学习（Imitation Learning）**：从启发式专家策略（如 PPR、Adamic-Adar）中蒸馏先验知识，初始化策略。
     - **偏好强化学习（Preference RL）**：利用 LLM 作为“偏好导师”，提供成对动作比较反馈，优化策略以匹配下游任务表现。

3. **高效低开销推理**
   - 策略网络为轻量 MLP，推理时无需调用 LLM 进行搜索，实现“零 token 检索”（zero-token retrieval）。
   - 支持近实时推理（near real-time inference），显著降低延迟和 token 成本。

#### 相比现有方法的优势：
| 维度 | HyGRL | 传统方法 |
|------|-------|--------|
| 图构建成本 | 极低（仅提取实体） | 高（LLM 抽取三元组） |
| 推理灵活性 | 自适应 beam search | 固定规则（如 PPR）或高成本 LLM agent |
| 多模态融合 | 支持文本与图的跨模态过渡 | 多数方法割裂处理 |
| 推理效率 | 轻量 MLP，毫秒级反馈 | 在线 LLM 调用，高延迟 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在三个主流多跳问答基准上评估：
- **2WikiMultiHopQA (2Wiki)**：复杂逻辑链，含结构与文本证据。
- **HotpotQA**：多样化、可解释的多跳问题。
- **MuSiQue**：由单跳问题合成的多跳问题，避免偏差。

> 所有数据集均经过过滤，保留至少包含两个可链接种子实体的样本，确保聚焦多实体推理。

### ⚙️ 实验设置
- **Knowledge Graph**：使用 **Freebase**（40M 实体）作为结构知识源。
- **文本语料**：各数据集提供的 Wikipedia 段落 dump。
- **索引构建**：离线完成，包括实体提取、链接与最短路径连接。
- **推理模块**：基于 MLP 的策略网络，输入为状态-动作向量（基于语义相似度与 PPR 分数）。
- **答案生成**：最终检索子图被转化为自然语言提示，输入 GPT-4o 生成答案。

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **答案质量** | Exact Match (EM), F1 Score |
| **推理保真度** | SF-F1（文本召回）、E-F1（三元组召回）、UE-F1（混合证据综合得分） |
| **效率** | Token 消耗、推理延迟（latency）、内存占用（RAM/VRAM） |
| **鲁棒性** | 注入噪声下的性能退化测试 |

### 🔁 基线方法对比
涵盖三大类 RAG 范式：
1. **Standard RAG**
   - `GPT-4o (Zero-shot)`
   - `textRAG`（标准 top-K 检索）
   - `TextBeamRAG`（基于 beam search 的纯文本推理）

2. **Graph-based RAG**
   - `Microsoft GraphRAG`
   - `PathRAG`
   - `HippoRAG`, `HippoRAG2-hybrid`
   - `Kg2RAG`

3. **变体对比**
   - `HyGRL-Heuristic`：仅使用启发式评分，无学习策略

---

## 3. 主要实验结果和性能指标

### 📈 整体性能（RQ1）
在 **Table 1** 中，HyGRL 在所有数据集上取得 **SOTA 性能**：

| 方法 | 平均 EM | 平均 F1 |
|------|--------|--------|
| Best Baseline (HippoRAG2-hybrid) | 54.30% | 61.23% |
| **HyGRL (Ours)** | **56.24%** | **63.87%** |

- 在 **2Wiki** 上 EM 达 **79.19%**，F1 为 **79.85%**，显著优于依赖全局摘要的 GraphRAG 和易漂移的 TextBeamRAG。
- 表明 HyGRL 能有效抑制噪声积累，提升推理准确性。

### 🔍 消融实验（RQ2）
| 变体 | 平均 F1 |
|------|--------|
| **HyGRL (Full)** | **63.87%** |
| w/o Text Nodes | 51.90% |
| w/o KG Completion | 45.07% |
| Replace RL with BFS | 60.12% |

- 移除文本节点导致 **↓11.97% F1**，说明文本作为“语义桥”至关重要。
- 移除 KG 补全退化为标准 RAG，性能大幅下降。
- 使用随机 BFS 替代 RL 策略，性能仍低于完整模型，验证了 **学习型策略的有效性**。

### ⏱️ 效率与部署成本（RQ3）
- **Token 成本**：
  - GraphRAG 构建耗超 **20 亿 tokens**（用于三元组抽取与社区摘要）。
  - **HyGRL 构建成本降低 90%**，训练成本可忽略（见 Figure 3）。
- **推理延迟**：
  - 平均增加 **0.67s** 检索开销，总延迟 **1.82s**，接近 real-time。
  - 显著优于需在线 LLM 过滤的 HippoRAG2-hybrid（2.91s）。
- **内存与 VRAM**：
  - 推理 VRAM 仅 **~475MB**，训练约 **2GB**，远低于端到端 LLM 微调方案。
  - 系统 RAM 占用较高（35.95GB），但为换取零磁盘 I/O 延迟的主动设计。

### 🧠 推理保真度评估（RQ4）
在 2Wiki 上评估证据检索质量（Table 4）：

| 方法 | SF-F1（文本） | E-F1（结构） | UE-F1（综合） |
|------|-------------|-------------|--------------|
| textBeamRAG | 19.7 | N/A | 17.0 |
| HippoRAG2-hybrid | 30.4 | 21.9 | 28.6 |
| **HyGRL** | **44.7** | 17.1 | **39.9** |

- **SF-F1 提升 47%**，表明 HyGRL 更擅长精准定位关键文本证据。
- 尽管 E-F1 略低于 HippoRAG2-hybrid，但 **UE-F1 领先 11.3%**，体现其在**跨模态协同**上的优势。

### 🛡️ LLM 鲁棒性测试（RQ5）
注入成对偏好标签噪声（Table 5）：

| 噪声比例 | F1 | 下降 |
|--------|-----|------|
| 0% | 50.34 | — |
| 15% | 49.44 | -0.90 |
| 30% | 48.28 | -2.06 |
| 50% | 46.25 | -4.09 |

- 表现出**优雅退化**（graceful degradation），即使在 30% 噪声下仍保持高稳定性。
- 归功于 **模仿学习 + 自批判基线（self-critical baseline）** 的联合目标，减少对 LLM 奖励的过拟合。

### 🔎 参数敏感性分析（RQ6）
- **Beam Width (K)**：最优值为 3，过大导致噪声引入。
- **Search Depth (Hops)**：峰值在 8 步，更深则性能下降。
- **Max Nodes (L)**：最佳长度为 20，超过 25 后效果恶化。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **异构图是解决多实体推理的关键**：结合文本与 KG 可有效缓解结构稀疏问题，文本块不仅是信息载体，更是“推理桥梁”。
2. **轻量策略学习优于在线 LLM agent**：通过冻结 LLM 作为监督者，训练小型 MLP 策略，可在极低成本下实现高性能自适应推理。
3. **两阶段训练范式有效**：模仿学习提供稳定初始化，偏好 RL 实现细粒度对齐，避免稀疏奖励问题。
4. **HyGRL 实现 SOTA 性能与效率平衡**：在准确率、保真度、速度、成本四方面全面领先。

### ⚠️ 局限性
1. **依赖 LLM 提供偏好信号**：尽管通过过滤机制提升可靠性，但策略仍是 LLM 隐式推理的近似。
2. **内存占用较高**：为实现高速检索，需将整个异构图载入内存（35.95GB RAM），不适合资源极度受限场景。
3. **离线构建依赖高质量实体链接**：若实体识别失败，可能导致图连接断裂。

### 🔮 未来工作方向
- 探索更高效的图压缩技术，降低内存需求。
- 引入动态图更新机制，支持持续学习。
- 将 HyGRL 框架扩展至其他复杂推理任务，如数学推理、法律问答等。
- 研究免 LLM 的奖励建模方式，进一步降低成本与偏见风险。

---

> **代码开源**：https://github.com/wjywjy123/HyGRL

</details>

---

### 12. [Notes to Self: Can LLMs Benefit from Experiential Abstractions?](https://arxiv.org/abs/2607.20372)

**Authors**: Chang Liu, Xinyu Li, Artur Dubrawski  
**Category**: cs.CL  
**Published**: 2026-07-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.20372v1  

#### Abstract
Humans distill experience into reusable abstractions, e.g., strategies and cautionary reminders, and apply them to gradually solve problems more effectively. We study whether Large Language Models (LLMs) can similarly benefit from such experiential abstractions. From LLMs' solution traces on the MAT...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Notes to Self: Can LLMs Benefit from Experiential Abstractions?**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
该论文探讨了一个核心问题：**大型语言模型（LLMs）是否能够像人类一样，从自身的推理经验中提炼出可复用的“经验性抽象”（experiential abstractions），并利用这些抽象来提升在数学和逻辑推理任务上的表现？**

传统方法通常依赖强模型（如 frontier models）作为教师来指导小模型学习，而本文关注的是：**小规模学生模型能否自主提取并有效利用自身训练过程中的经验性抽象**，实现自我改进。

---

### 🚀 提出了什么新方法或新思路
作者提出了一套完整的框架，用于从 LLM 的训练轨迹（solution traces）中构建和使用“经验性抽象库”，主要包括两个创新模式：

#### （1）**经验性抽象库的构建**
- 从目标 LLM 在 MATH 训练集上的 solution traces 中提取自然语言形式的抽象。
- 抽取方式有两种：
  - **Teacher-extracted**：由更强的教师模型（如 DeepSeek-V4-Flash）提取。
  - **Self-extracted**：由目标 LLM 自身完成提取。
- 抽象分为两类：
  - **Strategy**：来自正确解法的关键推理步骤，建议复用。
  - **Caution**：来自错误解法的常见失误模式，提醒避免。
- 使用 `all-MiniLM-L6-v2` 编码器进行向量化，并通过聚类去重，形成紧凑的检索库。

#### （2）两种使用抽象的方式
| 模式 | 描述 |
|------|------|
| **Mode 1: Inference-time retrieval** | 在推理时，根据测试问题检索相关抽象，注入 prompt 中供模型参考。 |
| **Mode 2: RL post-training with abstraction-augmented prompts** | 在强化学习微调阶段（GRPO），将检索到的抽象加入训练 prompt，使策略直接学习如何利用这些抽象。 |

> 🔁 这是一种 **train-to-test knowledge transfer** 范式，不同于 test-time scaling（如 Reflection）。

---

### ⭐ 相比现有方法的优势
| 对比维度 | 本文方法 | 其他方法（如 ExpeL, RLAD, Dynamic Cheatsheet） |
|--------|---------|---------------------------------------------|
| 是否需要强教师监督 | ❌ 不需要（self-extraction 可行） | ✅ 通常需要强模型监督或参与推理 |
| 推理成本 | 低（仅一次前向传播） | 高（多轮反馈/再生） |
| 知识存储形式 | 显式的自然语言抽象库 | 隐式记忆更新或内部状态调整 |
| 可解释性 | 高（人类可读的策略/警告） | 较低 |
| 泛化能力 | 支持跨数据集、跨模型迁移 | 多为特定任务设计 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 类型 | 用途 |
|-------|------|------|
| **MATH** (Hendrycks et al., 2021) | 数学问题（7,500 训练题） | 构建抽象库、训练与微调 |
| **MATH-500** | MATH 的测试子集（500 题） | 主要评估基准 |
| **GSM8K** | 小学数学应用题 | OOD 泛化测试（较简单） |
| **OlympiadBench** | 奥赛级别双语科学题 | OOD 泛化测试（更难） |
| **FOLIO** | 一阶逻辑推理题 | 用于生成逻辑类抽象 |
| **MuSR** | 多步软推理逻辑题 | 测试逻辑推理泛化能力 |

---

### ⚙️ 实验设置与评估指标

#### ✅ 模型配置
- **Target Models（学生模型）**：
  - `Llama-3.2-3B-Instruct`
  - `Qwen-2.5-1.5B-Instruct`
- **Teacher Model（抽取器）**：
  - `DeepSeek-V4-Flash`（通过 API 调用）
- **Embedder**：
  - `all-MiniLM-L6-v2`（384维向量，归一化）

#### ✅ 评估指标
- **pass@1 / pass@8**：在温度 0.6 下采样 8 次，至少有一次输出正确答案的比例。
- 所有结果均为 8 次 rollout 的平均值。

#### ✅ 基线方法对比
| 配置 | 描述 |
|------|------|
| **Baseline** | 原始模型，无任何微调或抽象注入 |
| **Inference_abs** | 仅在推理时注入检索到的抽象 |
| **GRPO** | 标准 GRPO 强化学习微调，无抽象 |
| **GRPO_train** | GRPO 微调时使用带抽象的 prompt |
| **GRPO_train+test** | 训练和推理都使用抽象 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Table 1 & Table 3）

#### 在 **MATH-500** 上的表现（pass@1/pass@8）

| Model | Abstraction Extractor | Baseline | Inference_abs | GRPO | GRPO_train | GRPO_train+test |
|-------|------------------------|----------|---------------|------|------------|------------------|
| Llama-3.2-3B | Teacher | 46.42/72.80 | 49.73/75.03 | 47.65/70.23 | **49.45/71.98** | 47.65/70.23 |
| Llama-3.2-3B | Self | 43.11/70.57 | 46.30/71.66 | 46.89/70.54 | **49.07/71.14** | 47.67/70.69 |
| Qwen-2.5-1.5B | Teacher | 50.23/76.47 | 52.55/75.55 | 53.44/76.34 | **53.77/76.83** | 52.66/75.76 |

> ✅ **GRPO_train** 是最稳定的增益来源，相比 Baseline 最高提升 **+6.34 pass@1（Llama）** 和 **+3.54（Qwen）**。

> ✅ **Self-extracted 抽象效果 ≈ Teacher-extracted** → 表明小模型也能自主提炼高质量经验！

---

#### 消融实验（Ablation Study）——验证抽象的有效性（Table 2）

| Mode | No notes (Baseline) | Blank notes | +Abstractions |
|------|--------------------|-------------|----------------|
| Inference-time | 43.11/70.57 | 46.02/70.00 | **46.42/72.80** |
| RL (GRPO) | 46.89/70.54 | 48.55/70.98 | **49.45/71.98** |

> ✅ 即使控制 prompt 结构一致，“+Abstractions”仍带来显著提升 → **抽象内容本身具有独特价值**。

---

#### 跨任务泛化能力（Table 3）

| Test Set | Method | Result (pass@1/pass@8) |
|---------|--------|------------------------|
| **GSM8K** (easy) | GRPO_train | 80.48/92.95 (**vs. Baseline 54.82**) ❌ *无增益* |
| **OlympiadBench** (hard) | GRPO_train | 16.36/35.91 (**↑ from 13.91**) ✅ *有增益* |
| **MuSR (from MATH abstractions)** | GRPO_train | 46.20/75.93 (**↑ from 40.94**) ✅ |
| **MuSR (from FOLIO abstractions)** | GRPO_train | **45.54/79.37** ✅ *逻辑抽象也有效* |

> 🔍 发现：抽象对 **模型原本不擅长的任务** 更有帮助（如 OlympiadBench），而在已饱和任务（如 GSM8K）上无效甚至有害。

---

#### 跨模型迁移能力（Table 4）

| Target Model | Baseline | Inference_abs |
|--------------|----------|----------------|
| gemma-2-2B-it | 21.77/43.00 | 13.75/33.40 ❌ *下降* |
| Phi-3.5-mini-instruct | 40.62/64.00 | **43.60/65.80** ✅ *提升* |

> ⚠️ 抽象迁移具有模型依赖性：对某些模型（如 Gemma）可能因 prompt 模板不适配而损害性能。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLMs 可以像人类一样从经验中提炼并复用抽象知识**：
   - 成功构建了可检索的经验性抽象库（strategies & cautions）。
   - 小模型能自主提取高质量抽象（self-extraction 效果媲美 teacher-extraction）。

2. **两种使用方式均有效，但 RL 微调更稳定**：
   - 推理时注入有一定帮助，但受格式敏感性和检索质量影响。
   - GRPO 微调结合抽象提示（abstraction-augmented training）带来最一致的性能提升。

3. **抽象有助于解决更难的问题，而非简单任务**：
   - 在 OlympiadBench 和 MuSR 上观察到明显增益。
   - 在已接近上限的任务（如 GSM8K）上无效，说明其作用是“补弱”而非“锦上添花”。

4. **抽象具备一定的跨领域和跨模型迁移潜力**：
   - 数学抽象可用于增强逻辑推理能力。
   - Llama 提炼的抽象可在 Phi 系列模型上生效。

5. **RL 会“锐化”输出分布，导致再注入抽象反而有害**：
   - 已经被 GRPO 训练过的模型趋于确定性输出。
   - 此时额外注入抽象可能扰动正确路径 → 导致 **GRPO_train+test < GRPO_train**。

---

### ⚠️ 局限性
1. **依赖固定设计选择**：未对 sentence encoder、k 值、去重阈值等做敏感性分析。
2. **计算资源有限**：RL 微调仅运行一个 epoch，更大规模训练可能改变结果排序。
3. **迁移能力受限**：抽象的跨模型有效性高度依赖下游模型对 prompt 模板的兼容性（如 Gemma 表现变差）。
4. **潜在风险**：若抽象库中包含错误启发式规则，可能误导模型；虽可通过 outcome reward 缓解，但仍需谨慎管理。

---

### 🔮 未来工作方向
1. **自动化抽象优化机制**：让模型不仅能提取抽象，还能动态评估其有效性并进行淘汰或修正。
2. **多轮迭代自提炼系统**：构建类似“笔记到自我”的闭环系统，持续积累和精炼经验库。
3. **探索更多抽象形式**：除自然语言外，尝试符号化、图结构等形式的抽象表示。
4. **扩展至其他复杂任务**：如代码生成、规划决策、科学发现等领域。
5. **降低对外部教师的依赖**：完全实现“自我教学”（self-teaching）范式。

---

## ✅ 总结一句话
> **本研究表明，LLMs 能够像人类一样从自身经验中提炼出可复用的“经验性抽象”，并通过推理时检索或 RL 微调加以利用，在数学与逻辑推理任务上实现有效的自我增强，且无需强模型监督即可达成接近教师提取的效果。**

</details>

---

### 13. [STN-TGAT: Top-K Portfolio Construction via Prior-Guided Graph Attention with Learnable Soft-Threshold Sparsification](https://arxiv.org/abs/2607.19385)

**Authors**: Haoran Guo, Yutong Lu, Li Zhang  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.19385v1  

#### Abstract
This paper tackles the problem of stock ranking and portfolio construction under realistic investment settings by jointly modeling temporal dynamics and cross-sectional dependencies. We propose the Soft-Threshold NMI-prior Transformer Graph Attention Network (STN-TGAT), which integrates a temporal T...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：STN-TGAT: Top-K Portfolio Construction via Prior-Guided Graph Attention with Learnable Soft-Threshold Sparsification

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于**现实投资场景下的股票排名与组合构建（portfolio construction）问题**。传统模型多关注绝对收益预测精度，而忽视了实际投资中更关键的 **Top-K 资产选择任务**——即在有限预算下选出最具潜力的少数资产，并考虑交易成本、权重分配等实践约束。

此外，现有方法在建模跨资产依赖关系时面临两大挑战：
- **图结构噪声**：基于相关性的图容易引入虚假连接；
- **静态图假设**：无法适应市场动态变化。

---

### 提出的新方法与创新思路
作者提出 **STN-TGAT（Soft-Thresholded NMI-prior Transformer Graph Attention Network）**，其核心创新包括：

#### ✅ 创新点一：决策对齐的 Top-K 排名目标
- 将每日选股建模为 **cross-sectional Top-K ranking problem**；
- 引入 **top-weighted ListNet 目标函数**，通过几何衰减加权机制强调高排名资产的排序准确性，使训练目标与实际投资决策一致。

#### ✅ 创新点二：基于 NMI 的先验图 + 可学习软阈值稀疏化
- 使用 **Normalized Mutual Information (NMI)** 构建非线性依赖先验图，捕捉超越线性相关的共动模式；
- 设计 **learnable soft-threshold sparsification 机制**，通过可学习的 sigmoid 门控平滑抑制弱边，保留强连接，实现自适应图稀疏化；
- 图结构对称归一化并加入对角偏移以稳定训练。

#### ✅ 创新点三：融合 Temporal 与 Spatial 建模的端到端框架
- 结合 **Transformer 编码器**提取长序列时间模式；
- 使用 **GAT（Graph Attention Network）** 在由 NMI 先验引导的动态图上传播信息；
- 注意力机制中显式注入 NMI 先验作为 bias 项，增强结构鲁棒性。

#### ✅ 创新点四：贴近实战的投资模拟评估体系
- 组合构建采用 **Top-5 选股策略**（从 S&P 500 前 50 大公司中选）；
- 权重通过 softmax 分配，结合预测得分体现“信心”；
- 显式调整 **transaction costs**，进行净收益回测（net-of-fee backtesting），提升结果经济意义。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **目标设计** | 优于仅优化 MSE 或普通 ListNet，直接对齐 Top-K 投资目标 |
| **图构建** | 比 Pearson/Glasso 更能捕获非线性依赖；比固定阈值更灵活稳健 |
| **模型架构** | 同时建模时序依赖（Transformer）与动态关系（GAT），优于纯 RNN/GCN |
| **实用性** | 支持分数驱动的加权配置，优于等权组合 |

---

## 2. 核心实验方法和设置

### 数据集
- **样本范围**：S&P 500 中市值最大的前 50 家公司（缓解幸存者偏差）
- **时间跨度**：2022-01-03 至 2024-12-30，共 752 个交易日
- **数据来源**：Yahoo Finance 的日频 OHLCV 数据
- **特征处理**：
  - Winsorize 在 ±3σ
  - 使用训练集统计量标准化（防止信息泄露）
  - 输入形式为 `[N, L, F]` 的滚动窗口张量（N=50, L=lookback length）

---

### 实验设置
- **任务**：每日预测下一交易日收益率 → 排名 → 构建 Top-5 多头组合
- **训练方式**：滚动窗口训练，close-to-close rebalancing
- **损失函数**：复合目标  
  $$
  \mathcal{L} = \mathcal{L}_{\text{list}} + \mathcal{L}_{\text{mse}} + \mathcal{L}_{\text{graph}}
  $$
  - $\mathcal{L}_{\text{list}}$: head-weighted KL 散度（ListNet变体）
  - $\mathcal{L}_{\text{mse}}$: 回归损失保留幅度信息
  - $\mathcal{L}_{\text{graph}}$: 控制图密度与稀疏性的正则项

---

### 评估指标

| 类型 | 指标 | 描述 |
|------|------|------|
| **预测性能** | MRR（Mean Reciprocal Rank） | 衡量真实最优股是否排在前列 |
|              | RBO（Rank-Biased Overlap） | 衡量预测与真实排名列表相似度（重视头部） |
|              | MSE | 预测回报值误差 |
| **投资绩效** | IRR（Internal Rate of Return） | 年化累计收益率（此处为测试期总收益） |
|              | Sharpe Ratio | 年化夏普比率（风险调整后收益） |
|              | MDD（Maximum Drawdown） | 最大回撤（下行风险） |

---

### 基线方法对比
| 类别 | 模型 |
|------|------|
| 统计模型 | ARIMA |
| 序列模型 | GRU, LSTM |
| 图增强模型 | GRU-GCN（NMI图）, GRU-GAT（NMI图） |

所有基线均在同一数据和设定下复现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 性能对比（Table 1）

| Model | MRR ↑ | RBO ↑ | MSE ↓ | IRR ↑ | Sharpe ↑ | MDD ↓ |
|-------|--------|--------|--------|--------|-----------|---------|
| ARIMA | 0.1079 | 0.0316 | 0.0017 | 0.0426 | 1.1145 | 0.0456 |
| GRU | 0.2120 | 0.0580 | 0.5142 | 0.1341 | 2.6553 | 0.0585 |
| LSTM | 0.2222 | 0.0601 | 0.2720 | 0.1143 | 2.1899 | 0.0719 |
| GRU-GCN | 0.0768 | 0.0193 | 1.0727 | -0.0435 | -1.4598 | 0.0799 |
| GRU-GAT | 0.1760 | 0.0468 | 0.5013 | -0.0012 | 0.0035 | 0.0799 |
| **STN-TGAT** | **0.1879** | **0.0509** | **0.3115** | **0.1807** | **2.9940** | **0.0634** |

> ✅ **关键发现**：
> - STN-TGAT 在 **IRR 和 Sharpe 上显著领先**，分别达到 **18.07% 收益率** 和 **2.99 夏普比**；
> - 虽然 LSTM 在 MRR 上略高，但其投资表现远不如 STN-TGAT，说明“预测准 ≠ 投资赚”；
> - 图模型若未合理设计（如 GRU-GCN/GAT），可能因噪声传播导致负收益。

---

### 消融实验结果

#### （1）损失函数消融（Table 2）
| Loss Variant | IRR ↑ | Sharpe ↑ | MDD ↓ |
|--------------|--------|----------|--------|
| MSE-only | 0.0395 | 0.7570 | 0.0779 |
| Spearman+MSE | 0.0945 | 2.3560 | 0.0744 |
| Pairwise+MSE | 0.0646 | 1.4740 | 0.0812 |
| ListNet-only | 0.2161 | 3.3160 | 0.0646 |
| **ListNet+MSE** | **0.2132** | **3.3130** | **0.0618** |

> 🔍 发现：
> - **ListNet 对投资绩效至关重要**，MSE-only 几乎无超额收益；
> - 加入 MSE 辅助回归可略微降低波动，改善 MDD，提供更好的分数校准。

#### （2）结构组件消融（Table 3）
| Model Config | IRR ↑ | Sharpe ↑ | RBO ↑ |
|-------------|--------|----------|--------|
| Transformer-GAT (no prior) | 0.0863 | 1.8932 | 0.0413 |
| Transformer-GAT (no sparsity) | 0.1152 | 2.5884 | 0.0413 |
| Transformer-GAT (Glasso prior) | 0.0381 | 0.7480 | 0.0406 |
| **STN-TGAT (full)** | **0.1807** | **2.9940** | **0.0509** |

> 🔍 发现：
> - 移除 NMI 先验导致 Sharpe 下降超 1 个单位；
> - 使用线性 Glasso 替代 NMI 导致接近零收益，表明**非线性依赖建模至关重要**；
> - 可学习稀疏化有效过滤噪声边，提升 Sharpe 超 0.4。

#### （3）组合内加权策略（Table 4）
| Allocation Strategy | IRR ↑ | Sharpe ↑ | MDD ↓ |
|---------------------|--------|----------|--------|
| TopK-EqualWeight | 0.1521 | 2.5595 | 0.0636 |
| **TopK-ScoreWeight** | **0.1807** | **2.9940** | **0.0634** |

> ✅ 得分加权优于等权，说明模型输出的 score magnitude 包含额外置信信息。

---

## 4. 关键结论和发现

### 主要结论
1. **决策对齐的训练目标是关键**：传统的 pointwise 回归或完整排名优化不足以支撑 Top-K 投资决策，必须强调头部排序精度（head-weighted ranking）。
2. **高质量图结构决定 GNN 成败**：NMI 比线性相关更能反映金融资产间复杂联动；可学习软阈值能自适应地去噪，优于手工阈值或全连接图。
3. **Hybrid Architecture 更有效**：Transformer 擅长时间建模，GAT 擅长关系推理，二者结合并在注意力中融合先验，形成互补优势。
4. **投资绩效需端到端验证**：仅看 MSE 或 MRR 不足以判断模型优劣，必须通过带交易成本的回测来衡量经济价值。

---

### 方法局限性
- **计算开销较大**：每步需重新计算 NMI 图与 GAT 传播，难以扩展至全市场（~5000只股票）；
- **依赖历史窗口稳定性**：NMI 计算易受极端行情扰动（如黑天鹅事件），可能导致图结构突变；
- **未建模宏观因子或文本信号**：当前仅用价格特征，未整合新闻、财报、宏观经济变量；
- **参数敏感性**：soft-threshold 参数 $t_g$ 和 sharpness $\beta$ 需仔细调参。

---

### 未来工作方向
1. 扩展至 multi-granularity modeling（分钟级+日级联合建模）；
2. 引入动态图学习机制替代预计算 NMI 图；
3. 融合 alternative data（如 ESG、供应链网络）丰富节点特征；
4. 探索 reinforcement learning 框架进行端到端 portfolio utility 优化；
5. 将模型应用于 sector rotation 或 factor timing 场景。

---

> 📌 **一句话总结**：  
> **STN-TGAT 通过“NMI先验图 + 可学习稀疏化 + 决策对齐排序损失”的设计，在真实投资约束下实现了更优的风险调整收益，验证了结构化关系建模与任务目标对齐在量化投资中的重要性。**

</details>

---

### 14. [HypEMBER: Hypernetwork-based Ensemble for Robust Policy Learning of Parametrized Dynamical Systems](https://arxiv.org/abs/2607.19628)

**Authors**: Nicol\`o Botteghi, Gabriele Pascali, Urban Fasel, Andrea Manzoni  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.19628v1  

#### Abstract
In this work we investigate reinforcement learning (RL) as a framework for the robust control of parametrized dynamical systems in presence of measurements and model uncertainties. High-dimensional state spaces, expensive numerical solvers, the partial knowledge of the governing equations, and the d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：HypEMBER: Hypernetwork-based Ensemble for Robust Policy Learning of Parametrized Dynamical Systems**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**参数化动力系统（parametrized dynamical systems）中的鲁棒控制问题**，尤其是在存在以下挑战时：
- **高维状态空间**（high-dimensional state spaces）
- **昂贵的数值求解器**（expensive numerical solvers）
- **物理参数不确定性或难以准确估计**
- **测量噪声和模型不匹配**（measurement noise, model misspecification）
- **标准 RL 方法在跨参数泛化和鲁棒性方面的不足**

这些问题导致传统 RL 方法训练不稳定、样本效率低、在真实场景中表现差（sim-to-reality gap）。

---

### **提出的新方法：HypEMBER**
作者提出了 **HypEMBER**，一种结合了 **hypernetworks** 和 **ensemble learning** 的新型 RL 框架，用于提升策略学习的**鲁棒性**和**泛化能力**。

#### **核心创新点：**
1. **Hypernetwork-based 参数条件建模**
   - 使用 **hypernetwork** 动态生成 policy 和 value 网络的权重，输入为 `[state, parameter]`。
   - 使得策略能显式地依赖于系统参数 `u`，实现跨不同动力学模式（regimes）的**参数化泛化**。

2. **Ensemble-based 不确定性量化**
   - 构建多个 policy 和 value 函数的集成（ensemble），用于估计 **epistemic uncertainty**。
   - 利用 critic ensemble 的预测标准差作为置信度，加权 Bellman 更新（uncertainty-weighted critic loss），提高训练稳定性。

3. **不确定性感知的动作选择策略（Uncertainty-Aware Action Selection）**
   - 在推理阶段提出新的动作选择准则：
     $$
     a^* = \arg\max_a \left( Q_{\text{mean}}(s,a) - \lambda Q_{\text{std}}(s,a) \right)
     $$
   - 不仅最大化预期回报，还最小化价值估计的不确定性，从而增强鲁棒性。

---

### **相比现有方法的优势**
| 方法 | 局限性 | HypEMBER 的改进 |
|------|--------|------------------|
| **Standard RL (e.g., TD3)** | 缺乏对参数变化的泛化能力，对噪声敏感 | 引入 hypernetwork 实现参数条件建模 |
| **SUNRISE (ensemble-based)** | 能处理不确定性，但无参数依赖建模 | 结合 hypernetwork，支持跨参数泛化 |
| **HypeRL (hypernetwork-based)** | 提升泛化性，但未显式建模不确定性 | 加入 ensemble，实现不确定性估计与探索优化 |

✅ **综合优势**：
- 更强的**训练稳定性**和**样本效率**
- 显著提升在**测量噪声**、**参数误设**及二者组合下的**鲁棒性**
- 支持从理想仿真环境到现实不确定环境的迁移

---

## **2. 核心实验方法和设置**

### **测试任务（数据集/环境）**
论文在两个具有代表性的参数化控制问题上进行验证：

#### **(i) 一维 Kuramoto-Sivashinsky (KS) 方程稳定化**
- 描述非线性 PDE 控制问题，表现出混沌行为。
- 参数 `u ∈ (-0.25, 0.25)` 影响外力项。
- 观测值：8个等距传感器读数 + 参数 `u`。
- 控制目标：将系统稳定至零平衡态，同时最小化控制能耗。

#### **(ii) 二维双陀流场（Double-Gyre Flow）中的粒子导航**
- 时间依赖的流体场，参数为振幅 `β` 和频率 `ω`。
- Agent 控制粒子速度以到达任意目标位置。
- 观测值：相对目标位置、局部涡量（vorticity）、参数 `[β, ω]`。
- 更具挑战性，需适应动态变化的流场结构。

---

### **实验设置与评估指标**

#### **训练与测试分离设计（模拟到现实差距模拟）**
- 所有 agent 在**无噪声、参数一致的理想环境**中训练。
- 测试阶段引入三种扰动：
  1. **测量噪声（Measurement Noise）**: $ s_t \leftarrow s_t + \mathcal{N}(0, \sigma_M I) $
  2. **模型误设（Model Misspecification）**: $ u \leftarrow u + \mathcal{N}(0, \sigma_P) $
  3. **两者结合**

- 噪声强度从 0% 到 40% 逐步增加，评估性能退化趋势。

#### **评估指标**
- **累计奖励（Cumulative Reward）**：越高越好
- **性能标准差**：反映鲁棒性和一致性
- **消融实验**：分析不同组件（如 UA 动作选择）的影响

---

### **基线方法对比**
共比较五种方法：
1. **TD3**：基础确定性策略梯度算法
2. **PolyL0-TD3**：使用稀疏多项式策略，强调可解释性
3. **HypeRL**：基于 hypernetwork 的 RL 方法
4. **SUNRISE**：基于 ensemble 的不确定性感知 RL
5. **HypEMBER**（本文方法）

所有方法均使用相同超参数配置（见附录表5），确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **(i) Kuramoto-Sivashinsky 控制任务**

#### **训练性能（Table 1）**
| 方法 | 平均奖励 ± 标准差 |
|------|--------------------|
| HypeRL | **-70.4 ± 6.2** |
| HypEMBER | -82.9 ± 8.4 |
| SUNRISE | -143.0 ± 77.9 |
| PolyL0-TD3 | -186.4 ± 66.8 |
| TD3 | -207.9 ± 48.5 |

📌 HypEMBER 虽略低于 HypeRL，但训练更稳定，跨种子方差更小。

#### **鲁棒性测试（Table 2, 图3–5）**
在 `u=0.175` 下测试：

| 场景 | HypEMBER | SUNRISE | HypeRL |
|------|----------|---------|--------|
| 理想情况 | **-53.6** | -59.4 | -52.6 |
| 测量噪声（20%） | **-67.1** | -70.8 | -68.1 |
| 参数误设（30%） | **-61.1** | -73.7 | -66.7 |
| 两者结合 | **-76.7** | -84.7 | -89.1 |

✅ **关键发现**：
- HypEMBER 在大多数扰动下表现最优，尤其在复合扰动中优势明显。
- HypeRL 对轻度噪声表现好，但在高噪声下退化严重。
- SUNRISE 鲁棒性强，但不如 HypEMBER 灵活。
- **不确定性感知动作选择（UA）** 在测量噪声下有效降低奖励方差（见图15–17）。

---

### **(ii) 双陀流场粒子导航任务**

#### **训练性能（Table 3）**
| 方法 | 平均奖励 ± 标准差 |
|------|--------------------|
| **HypEMBER** | **-28.6 ± 5.5** |
| HypeRL | -34.2 ± 1.3 |
| SUNRISE | -49.3 ± 7.4 |
| PolyL0-TD3 | -69.0 ± 2.2 |
| TD3 | -76.5 ± 5.5 |

📌 HypEMBER 在此任务上取得最佳训练成绩。

#### **鲁棒性测试（Table 4, 图10–12）**
在高噪声（40%）下：

| 场景 | HypEMBER | SUNRISE | HypeRL |
|------|----------|---------|--------|
| 理想情况 | **-9.75** | -8.80 | -44.65 |
| 测量噪声 | **-315.9** | -629.8 | -376.59 |
| 参数误设 | **-9.12** | -8.31 | -155.2 |
| 两者结合 | **-307.8** | -631.4 | -368.09 |

✅ **关键发现**：
- HypEMBER 在所有扰动下均显著优于其他方法。
- HypeRL 在理想条件下表现极差，说明其对观测质量高度依赖。
- Ensemble 方法（SUNRISE & HypEMBER）更能抵抗噪声影响。
- UA 动作选择进一步提升了 HypEMBER 的鲁棒性（见图18–20）。

---

### **消融实验（Ablation Study）**
- **λ 参数影响（Appendix D）**：
  - 当 λ = 0.25 时，在高噪声环境下获得最高且最稳定的奖励。
  - 表明主动减少不确定性有助于提升决策鲁棒性。
- **是否启用 UA 动作选择**：
  - 在测量噪声和复合扰动下，UA 明显优于平均动作选择。
  - 在纯参数误设下效果不显著。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **HypEMBER 显著提升了 RL 在参数化动力系统中的鲁棒性和泛化能力**。
2. ✅ **Hypernetwork + Ensemble 的结合是有效的架构设计**：
   - Hypernetwork 提供参数条件泛化
   - Ensemble 提供不确定性估计与探索机制
3. ✅ **不确定性感知动作选择（UA）是一种简单而有效的推理策略**，特别适用于存在测量噪声的场景。
4. ✅ 在两种截然不同的物理系统（PDE 与流体导航）上均验证了方法的有效性，表明其**通用性强**。

---

### **方法的局限性**
1. ❗ **计算开销较高**：维护 N 个 actor/critic 网络及其 hypernetwork，推理成本上升约 N 倍。
2. ❗ **超参数敏感性**：如 ensemble 大小、温度参数 T、λ 等需要调优。
3. ❗ **当前仅限于连续控制任务**，尚未扩展至离散动作空间或多智能体场景（尽管 HypeMARL 已有相关工作）。

---

### **未来工作方向**
1. 🔮 将 HypEMBER 扩展至 **multi-agent RL** 场景（已有 HypeMARL 工作基础）。
2. 🔮 探索 **adaptive λ 调整机制**，根据环境不确定性动态调整探索-利用权衡。
3. 🔮 结合 **offline RL** 或 **model-based RL** 进一步提升样本效率。
4. 🔮 应用于更复杂的实际系统，如湍流控制、机器人集群协同等。

---

## **总结**
> **HypEMBER 是首个将 hypernetwork 与 ensemble learning 深度融合用于参数化动力系统鲁棒控制的 RL 框架**。它不仅在训练效率和稳定性上优于现有方法，更重要的是在面对测量噪声、参数误设等现实挑战时展现出卓越的鲁棒性。其实验设计严谨，验证充分，为科学计算与控制领域的 RL 应用提供了重要范式。

🔗 **代码开源地址**：[github.com/nicob15/Hypernetwork-based-Reinforcement-Learning](https://github.com/nicob15/Hypernetwork-based-Reinforcement-Learning)

</details>

---

### 15. [Logic-Guided Data Extraction with Answer Set Programming and Large Language Models](https://arxiv.org/abs/2607.19365)

**Authors**: Mario Alviano, Lorenzo Grillo, Nicola Leone, Fabrizio Lo Scudo  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.19365v1  

#### Abstract
When Large Language Models (LLMs) are used for semantic data extraction from unstructured text, producing candidate relational facts from natural language, they may remain unreliable for tasks requiring complex combinatorial reasoning and global consistency. This paper proposes a logic-guided data e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Logic-Guided Data Extraction with Answer Set Programming and Large Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 Large Language Models (LLMs) 的语义数据提取方法在处理复杂组合推理任务时存在以下问题：
- **独立提取**：传统方法（如 LLMASP）对每个目标谓词（predicate）独立调用 LLM 进行提取，忽略了谓词之间的逻辑依赖关系。
- **资源浪费**：即使某些谓词在当前上下文中不可能成立（例如未提及分层信息却尝试提取 `layer_size`），仍会发起不必要的 LLM 调用。
- **错误传播**：容易产生**幻觉输出**（spurious/hallucinated facts），且缺乏早期一致性检查机制。

### **提出的新方法与思路**
本文提出了一种 **logic-guided data extraction framework**，其核心思想是：
> 将 **LLM-based extraction** 与 **Answer Set Programming (ASP)** 动态交织，利用逻辑推理主动控制提取过程。

#### **关键技术组件**：
- **Guard-based Admissibility Conditions**：为每个目标谓词定义一组逻辑守卫（guards），只有当这些守卫在当前数据库中被满足时，才允许对该谓词发起 LLM 提取请求。
- **Interleaved Execution**：交替执行“LLM 提取”和“ASP 推理”，实现：
  - 从已有事实中推导出隐含信息（无需额外调用 LLM）
  - 早期检测不一致或缺失信息
  - 动态决定下一步是否需要提取某个谓词
- **Guard Caching Mechanism**：利用单调查询的性质缓存守卫判断结果，减少重复的 ASP solver 调用。

### **相比现有方法的优势**
| 维度 | 传统方法（如 LLMASP） | 本论文方法（LGX） |
|------|------------------------|--------------------|
| **控制流** | 底层 → 上层（先全部提取再推理） | 双向闭环（提取 ↔ 推理） |
| **LLM 调用次数** | 固定，等于目标谓词数 | 动态减少，可跳过无效谓词 |
| **一致性保障** | 后置验证（post-hoc） | 前置控制 + 实时检查 |
| **输出质量** | 易受幻觉影响 | 通过逻辑约束抑制无关输出 |
| **效率优化** | 无 | 支持守卫缓存、增量推理 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验基于两个由 ASP 领域衍生的人工构造基准：

1. **Graph Benchmark (G)**  
   - 场景：从自然语言描述中提取**分层图结构**（layered undirected graph）
   - 包含节点、边、层级分配等信息
   - 示例文本：“Layer zero contains nodes n1, n2, and n3...”

2. **Logic Puzzle Benchmark (LNRS)**  
   - 整合四个经典 ASP 域：Labyrinth, Nomystery, Ricochet Robots, Sokoban
   - 涉及动作规划、状态转移、路径寻找等复杂逻辑结构
   - 引入**层次化守卫**：首先识别谜题类型，后续提取仅限该类型的 schema

### **实验设置**
- **LLM Oracle**：
  - 主要使用 **Meta Llama 3.1** 的三个版本：
    - Small (8B)
    - Medium (70B)
    - Large (120B)
  - 额外测试 **GPT-OSS (120B)** 以验证跨架构鲁棒性
- **ASP Solver**：CLINGO 5.8
- **温度设置**：LLM 温度设为 0，确保确定性输出
- **缓存机制**：启用 oracle 和 guard 结果缓存，排除随机性干扰

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Oracle Calls (θ Calls)** | 调用 LLM 的总次数（衡量通信开销） |
| **F1-Score** | 提取事实的精确率与召回率的调和平均 |
| **Perfect Rate** | 完全正确抽取所有实例的比例 |
| **Solver Invocations** | ASP solver 被调用的次数（反映内部推理成本） |

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **LLMASP (None)** | 不进行任何逻辑过滤，盲目提取所有谓词 |
| **LLMASP (A posteriori)** | 先全部提取，再用 ASP 规则后处理过滤错误结果 |
| **LGX (By design)** | 本文方法，在提取前通过守卫控制流程 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

#### **Graph Benchmark (G) — Medium LLM (70B)**
| 方法 | Oracle Calls | F1-Score | Perfect Rate |
|------|---------------|----------|--------------|
| LLMASP (None) | 240 | 0.870 | 0.417 |
| LLMASP (A posteriori) | 240 | 0.952 | 0.850 |
| **LGX (By design)** | **193** | **0.952** | **0.850** |

✅ **结论**：LGX 在**更少调用（↓19.6%）**的情况下达到与“后处理过滤”相同的准确率。

#### **Logic Puzzle Benchmark (LNRS) — Medium LLM (70B)**
| 方法 | Oracle Calls | F1-Score | Perfect Rate |
|------|---------------|----------|--------------|
| LLMASP (None) | 3072 | 0.572 | 0.000 |
| LLMASP (A posteriori) | 3072 | 0.963 | 0.469 |
| **LGX (By design)** | **864** | **0.963** | **0.469** |

✅ **结论**：LGX 将 LLM 调用减少了 **~72%**，同时保持与最优后处理方案完全一致的结果。

### **消融实验与分析**
虽然未明确列出“消融实验”表格，但论文通过多组配置比较揭示了以下关键发现：
- **守卫设计直接影响效率**：在 LNRS 中引入“谜题类型识别作为高层守卫”显著减少了跨领域噪声查询。
- **缓存机制有效降低 solver 开销**：
  - 在 LNRS 上，守卫缓存避免了 **61.4% ~ 82.6%** 的冗余 ASP solver 调用。
  - 即使在简单场景 G 中，guard check 的延迟也远小于节省的 LLM 交互时间。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **理论等价性成立**：在“guard-respecting oracle”假设下，LGX 与 LLMASP 在最终提取结果上**完全等价**（Theorem 1），但调用次数永不增加，通常更少。
2. ✅ **效率显著提升**：通过逻辑前置控制，可在不影响精度的前提下大幅减少 LLM 调用（最高达 72%）。
3. ✅ **质量由设计保证**：相比“先错后修”的 a posteriori 方法，LGX 在提取阶段就规避了非法谓词，实现了“correct-by-design”。
4. ✅ **非单调逻辑的价值凸显**：ASP 的非单调推理能力（如缺省否定 `not`）使其能表达复杂的条件依赖和异常检测规则。

### **方法的局限性**
- **依赖人工设计守卫**：目前守卫集合（guards）需由专家手动编写，自动化生成仍具挑战。
- **初始 schema 固定**：框架假设目标谓词和逻辑程序已知，难以应对开放域动态扩展。
- **对 oracle 行为有假设**：要求 LLM 在守卫失败时不返回无关事实（guard-respecting），现实中可能存在偏差。
- **轻量级 LLM 表现差**：Small 模型（8B）即使配合 LGX，Perfect Rate 仍接近于零，说明基础生成能力仍是瓶颈。

### **未来工作方向**
1. **自动推断守卫属性**：研究如何自动识别守卫的单调性（monotonicity），以支持更激进的缓存策略。
2. **动态 prompt 构造**：结合 logic-driven templating（如 Mustache）技术，根据推理状态动态调整 prompt 内容。
3. **集成 goal-directed 引擎**：探索将 PROLOG 类的局部依赖追踪引擎与 ASP 的全局求解结合，实现更细粒度的提取控制。
4. **增量式推理优化**：开发支持增量更新的 ASP 推理器，进一步降低中间步骤的计算开销。
5. **扩展至多模态输入**：将图像、表格等非文本信息纳入逻辑引导框架。

---

> 📌 **一句话总结**：  
> 本文提出的 **LGX 框架**通过将 **ASP 的逻辑控制力**嵌入到 LLM 数据提取流程中，实现了“何时提取、提取什么”的智能决策，在保证结果正确性的前提下，显著降低了 LLM 的调用频率并提升了输出质量，为可控的神经符号集成提供了新范式。

</details>

---

### 16. [Spectral-LSH: Sub-Quadratic Prompt Compression via Krylov-Projected Locality-Sensitive Hashing](https://arxiv.org/abs/2607.19368)

**Authors**: Ali Mahdavi, Azaseh Zamanifar, Amirfarhad Farhadi, Omid Kashefi  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.19368v1  

#### Abstract
Long-prompt inference remains expensive because prefill attention scales quadratically with sequence length. We propose Spectral-LSH, a training-free prompt compression method that operates before the prompt enters the language model. Spectral-LSH approximates the dominant components of an implicit ...

---

### 17. [Fully Dynamic Rooted Spanning Tree on GPU](https://arxiv.org/abs/2607.20211)

**Authors**: Abhijeet Sahu, Harmit Singh, Soham Nandy, G. Ramakrishna  
**Category**: cs.DC  
**Published**: 2026-07-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.20211v1  

#### Abstract
Spanning trees are fundamental structures in graph theory, essential for various applications such as network maintenance, routing adjustments, and many more. The dynamic nature of real-world networks requires efficient updates to these structures as the underlying graph evolves. Maintaining rooted ...

---

### 18. [Total Variation Distance Estimation in Autoregressive Models](https://arxiv.org/abs/2607.19510)

**Authors**: Eric Price, Kevin Tian, Zhiyang Xun, Yusong Zhu  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.19510v1  

#### Abstract
Modern LLM deployments use a number of implementation choices and inference optimizations (e.g., batching, custom kernels, and quantization) on top of fixed weights, so two engines serving "the same model" can produce meaningfully different distributions. We study the problem of estimating the total...

---

### 19. [Self-organizing Architecture of Receptron Units: a Hardware-Aware Framework for Edge Intelligence](https://arxiv.org/abs/2607.20162)

**Authors**: Stefano Radice, Ludovico Casaccia, Riccaro Emanuele Beccalli, Bruno Paroli, Paolo Milani  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.20162v1  

#### Abstract
The growing demand for intelligent processing at the edge of IoT networks is constrained by the severe computational and memory limitations of microcontroller units, which render impractical conventional deep learning approaches. We propose a neuromorphicinspired classifier based on the Receptron mo...

---

### 20. [FineServe: A Fine-Grained Dataset and Characterization of Global LLM Serving Workloads](https://arxiv.org/abs/2607.19349)

**Authors**: Tiancheng Zhang, Shaoyuan Huang, Mingyuan Wang, Yunfeng Zhao, Xiaofei Wang, Wenyu Wang  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.19349v1  

#### Abstract
Large language models (LLMs) are increasingly deployed as always-on online services, making efficient LLM serving a critical systems challenge. Achieving low latency and high throughput under volatile demand requires deep understanding of real-world serving workloads, yet existing studies often rely...

---

### 21. [AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally](https://arxiv.org/abs/2607.19363)

**Authors**: Shaowen Wang, Yuke Zheng, Tansheng Zhu, Shuang Chen, Shaofan Liu, Suncong Zheng, Jian Li  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.19363v1  

#### Abstract
Rotary Position Embedding (RoPE) is widely adopted in Transformers to encode positional information, yet standard implementations enforce a uniform frequency schedule and scaling across all attention heads. Using simplified retrieval tasks and length generalization scenarios, we show -- both empiric...

---

### 22. [Geometry-Guided Constraint Learning for LLM Safety Classification](https://arxiv.org/abs/2607.19366)

**Authors**: Fumiaki Uehara, Koo Imai, Masato Tsutsumi, Keigo Kansa, Sora Usui, Yuki Kobiyama  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.19366v1  

#### Abstract
Safety as Polytope (SaP) learns linear half-space constraints in LLM hidden space but requires per-category tuning of the constraint count K. We show that sparse autoencoder (SAE) feature extraction resolves this: K=2 becomes optimal for 12/14 categories on Qwen3.5-9B, achieving 96-99% accuracy per ...

---

### 23. [SenWorld: A Digital-Twin Simulation for Generating Context-Rich Evaluation Data](https://arxiv.org/abs/2607.19949)

**Authors**: Zenghui Zhou, Xiaoyang Li, Xiaoxuan Qiao, Zhilang Wei, Tianming Lei  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.19949v1  

#### Abstract
Smartphone personal assistants reason over longitudinal personal data, yet evaluating them requires context-rich evaluation data whose correct answers are known, and real device traces are too privacy-sensitive to share. To address this challenge, we present SenWorld, a physically grounded, determin...

---

### 24. [The Orthogonalized Read Is a Removable Training Scaffold for Recurrent Memory](https://arxiv.org/abs/2607.19390)

**Authors**: Keston Aquino-Michaels  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.19390v1  

#### Abstract
A recent report finds that orthogonalizing the mLSTM memory matrix at read time (five Newton-Schulz iterations, trained through) substantially improves noisy associative recall. The effect replicates, but it is not a memory improvement. Training on this task is a long chance plateau followed by a sh...

---

### 25. [Reward-Aware Population Scaling of Evolutionary Strategies in LLM Fine-Tuning](https://arxiv.org/abs/2607.19408)

**Authors**: Sung Cho, Gyubin Han  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.19408v1  

#### Abstract
Using Evolutionary Strategies (ES) for fine-tuning large language models is attractive because it is memory-efficient, parallel, and compatible with black-box or discrete rewards. Yet its population-size conclusions conflict sharply: fine-tuning with cross-entropy (CE) reward succeeds with $N=1$, wh...

---

### 26. [Autonomous Collaborative Learning Among an Ensemble of Tsetlin Machines with Consensus-Based Inference](https://arxiv.org/abs/2607.20124)

**Authors**: Yehuda Rudin, Osnat Keren, Michal Yemini, Alexander Fish  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.20124v1  

#### Abstract
Tsetlin Machine (TM) is a rule-based machine-learning algorithm comprising collectives of two-action Tsetlin Automata (TAs) that cooperatively form conjunctive logical clauses from Boolean inputs through stochastic feedback. Although few recent studies have examined TM Federated Learning, the broade...

---

### 27. [Local Stability and Gaussian Smoothing of Quantized Neural Networks](https://arxiv.org/abs/2607.20153)

**Authors**: Sergey Salishev, Anton Makarov, Oleg Granichin  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.20153v1  

#### Abstract
We study Gaussian averaging as a smooth surrogate for quantized neural models. Under bounded local oscillation, we derive a local dimension-dependent bound on |f-g|, linking Gaussian smoothing to the stability analysis of discontinuous networks. We compute closed-form Gaussian averages of the rectif...

---

### 28. [Interval and fuzzy physics-augmented neural networks (iPANN and fPANN) for uncertainty quantification and propagation in constitutive modeling](https://arxiv.org/abs/2607.20339)

**Authors**: Somesh Pratap Singh, Govinda Anantha Padmanabha, Jingye Tan, Steven Yang, Reese E. Jones, D. Thomas Seidl, Nikolaos Bouklas  
**Category**: cs.LG  
**Published**: 2026-07-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.20339v1  

#### Abstract
Constitutive modeling under uncertainty remains a central challenge for reliable mechanics simulations, particularly when the available stress-deformation data are sparse, noisy, or heterogeneous. We propose interval and fuzzy physics-augmented neural networks (iPANNs and fPANNs) for uncertainty-awa...

---

### 29. [GraphContainer: A Unified Platform for Comparing and Debugging Graph RAG Methods](https://arxiv.org/abs/2607.19362)

**Authors**: Seonho An, Chaejeong Hyun, Min-Soo Kim  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.19362v1  

#### Abstract
Graph RAG mitigates hallucinations and stale knowledge in LLMs, particularly for multi-hop question answering. However, existing approaches remain highly fragmented and incompatible. The structural heterogeneity of graph formats across different frameworks and the lack of granular visualization tool...

---

### 30. [Rethinking Uncertainty Evaluation in Large Language Models](https://arxiv.org/abs/2607.19367)

**Authors**: Krish Matta, Atharv Naphade, Andy Zou  
**Category**: cs.AI  
**Published**: 2026-07-23  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.19367v1  

#### Abstract
Calibration is the primary criterion for evaluating LLM confidence, but it is insufficient: it admits trivially incoherent estimators, depends on the evaluation distribution, and does not test the extent to which the estimation can be interpreted as a consistent, underlying probability function. Wha...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-07-02 08:36:28 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Message Passing Enables Efficient Reasoning](https://arxiv.org/abs/2607.01077)

**Authors**: Xuecheng Liu, Daman Arora, Gokul Swamy, Andrea Zanette  
**Category**: cs.CL  
**Published**: 2026-07-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2607.01077v1  

#### Abstract
While inference-time scaling has improved the reasoning abilities of large language models (LLMs), the need to generate long chains-of-thought (CoTs) is a computational bottleneck. Thus, in contrast to sequential scaling methods like CoT, recent parallel scaling techniques instead use fork and join ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Message Passing Enables Efficient Reasoning

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在推理时依赖长链式思维（Chain-of-Thought, CoT），这种**串行生成方式**带来了显著的计算瓶颈：
- **高延迟**：由于自回归生成，必须顺序执行。
- **上下文压力大**：Transformer 架构中注意力机制的二次复杂度限制了可处理的上下文长度。
- **并行效率低**：现有的 fork-join 并行范式虽然能并发执行多个线程，但所有协调都通过中心化的父线程完成，导致通信瓶颈和隐式串行化。

### 提出的新方法：Message Passing Language Models (MPLMs)
作者提出了一种全新的推理框架——**Message Passing Language Models (MPLMs)**，其核心是引入类似 MPI（Message Passing Interface）的轻量级通信原语，使多个 LLM 线程能够直接、点对点地通信。

#### 关键指令（Directives）
- `<spawn>`：创建新的 LLM 线程。
- `<send>`：向指定 ID 的线程发送消息。
- `<recv>`：等待来自指定线程的消息（支持 wait-for-all 或 wait-for-any）。
- `<stop>`：终止当前线程。

#### 创新优势
相比传统方法，MPLMs 具有两大核心优势：
1. **降低通信成本（Reduced Communication Costs）**  
   避免了 fork-join 中将完整上下文反复传递给所有子线程的做法，仅传递必要的局部信息，显著减少冗余上下文共享。

2. **支持抢占式执行（Preemption）**  
   线程可以在获得足够信息后提前终止其他分支（如找到 SAT 解后立即停止搜索树的其余部分），避免不必要的计算。

---

## 2. 核心实验方法和设置

### 数据集
论文在三类任务上验证 MPLM 的有效性：

| 任务 | 描述 |
|------|------|
| **Sudoku** | $N^2 \times N^2$ 数独谜题（4×4 到 25×25），使用“裸单”策略求解。 |
| **3-SAT** | 布尔可满足性问题，变量数从 8 到 20 不等。 |
| **LongBench-v2** | 长文本问答基准，包含 503 个多选题，上下文长度从 8K 到 2M tokens，涵盖多种现实场景（多文档 QA、代码理解等）。 |

### 实验设置与评估指标

#### 模型训练
- 在 **Qwen3-0.6B-Base** 上进行监督微调（SFT），模拟 MPLM 推理轨迹生成训练数据。
- 对于 Sudoku 和 3-SAT，基于 naked-singles 和 DPLL 算法生成 CoT 数据。
- 每次训练耗时小于 48 小时（H100 GPU）。

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Accuracy** | 成功解决的谜题比例 |
| **Latency** | 解决一个谜题所需的平均时间（秒） |
| **Maximum Context** | 单个线程所需的最大上下文长度 |
| **Sequential Tokens** | 必须因果依赖地串行生成的 token 数量（衡量串行瓶颈） |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Serial CoT** | 完全串行的链式思维，无并行化 |
| **Fork-Join (FJ)** | 父线程分叉多个子线程，并行执行后汇总结果（如 Pan et al., 2025） |
| **DeepSeek-R1 / GPT-5 Pro** | 闭源前沿推理模型（禁用工具以隔离纯推理能力） |
| **Recursive Language Models (RLMs)** | 当前主流的 fork-join 类型长文本推理框架（Zhang et al., 2026） |

---

## 3. 主要实验结果和性能指标

### Sudoku 实验结果

#### ✅ 可扩展性显著优于基线
| 方法 | 4×4 | 9×9 | 16×16 | 25×25 |
|------|-----|-----|-------|--------|
| **MPLM** | 2.1s (100%) | 14.9s (100%) | 117.7s (92%) | **1017.3s (72%)** |
| **FJ** | 2.9s (100%) | 59.6s (93%) | ❌（不可行） | ❌ |
| **Serial** | 4.5s (99%) | ❌ | ❌ | ❌ |
| **GPT-5 Pro** | ✅ (100%) | ✅ (100%) | ✅ (45%) | ❌（无法扩展） |

> 🔍 **关键发现**：MPLM 是唯一能成功扩展到 **25×25 数独**的方法，而 FJ 和 Serial 因上下文限制失败，GPT-5 Pro 也无法可靠求解。

#### ✅ 上下文与串行 token 扩展性更优
- **最大上下文需求**：MPLM 的增长指数为 ~1.1，远低于 FJ (~1.8) 和 Serial (~1.8)。
- **串行 token 数量**：MPLM 为 $O(TkM)$，FJ 为 $O(TNM)$，理论节省因子达 $O(N/k)$，尤其当通信稀疏时优势巨大。

---

### 3-SAT 实验结果

#### ✅ 抢占机制提升效率
| 方法 | 平均延迟（n=20） | 最大加速比 |
|------|------------------|------------|
| **MPLM** | **~50s** | **2.57×** |
| **FJ** | ~128s | — |
| **Serial** | 超出上下文限制（n>12） | — |

> 🔍 **关键发现**：在不平衡搜索树中，MPLM 可通过早期终止未完成分支实现大幅加速；而 FJ 必须等待所有子线程完成才能聚合。

---

### LongBench-v2 实验结果（无需微调）

#### ✅ 大型预训练模型可直接遵循 MPLM 指令
| 模型 | 方法 | 准确率 | 延迟 |
|------|------|--------|------|
| **Qwen3-30B-A3B** | **MPLM** | **37.8%** | **61.3s** |
| | RLM (FJ) | 29.7% | 105.7s |
| **Qwen3.6-35B-A3B** | **MPLM** | 46.5% | **102.2s** |
| | RLM (FJ) | 46.7% | 223.5s |

> 🔍 **关键发现**：
- MPLM 在保持准确率的同时，**延迟降低约 1.7–2.2 倍**。
- 支持**迭代证据聚合**：主控线程可反复查询同一 worker 获取细粒度信息，无需重新加载上下文。
- RLM 的 worker 是瞬态的，后续查询需重建上下文，效率低下。

---

## 4. 关键结论和发现

### 主要发现
1. **MPLMs 实现了更高效的推理范式**  
   通过点对点消息传递和持久线程状态，打破了传统 CoT 和 fork-join 的串行瓶颈。

2. **理论与实证双重验证效率优势**  
   - 理论上证明 MPLM 的最大上下文需求仅为 $O(TkM)$，优于 FJ 的 $O(TNM)$。
   - 实验显示其在 **Sudoku、3-SAT、LongBench-v2** 上均显著优于现有方法。

3. **支持复杂控制流：抢占与重生成（respawning）**  
   - **Preemption**：允许早期终止无效分支，提升搜索效率。
   - **Respawning**：线程可通过压缩自身状态重启，防止上下文无限增长，支持无限步推理。

4. **无需训练即可被大模型采纳**  
   强大的预训练模型（如 Qwen3）可直接响应 MPLM 指令，在 LongBench 上即插即用取得高效表现。

---

### 局限性
1. **通信模式依赖提示或训练**  
   如何决定“何时”、“与谁”通信，在开放域任务中仍需精心设计提示或额外训练。

2. **目前验证集中于结构化任务**  
   Sudoku 和 3-SAT 具有天然的并行结构，推广到非结构化任务（如创意写作）尚待探索。

3. **运行时控制器开销虽小但仍存在**  
   虽然实验证明控制器开销 <2%，但在极端高并发场景下可能成为瓶颈。

---

### 未来工作方向
1. **构建更丰富的通信原语**  
   如广播（broadcast）、全收集（all-gather）等，增强表达能力。

2. **自动生成并行 CoT 数据**  
   利用强化学习或自我演化生成适用于 MPLM 的训练轨迹，摆脱人工 scaffolding。

3. **扩展至更广泛的应用场景**  
   如 **Agentic Reasoning**、**Code Generation**、**Theorem Proving** 等需要长期记忆与协作的任务。

4. **优化调度与负载均衡策略**  
   动态调整线程数量与资源分配，进一步提升系统吞吐量。

---

> 📌 **总结一句话**：  
> **MPLMs 将 LLM 推理从“中央集权”的串行模式，转变为“分布式组织”式的并行协作，开启了高效、可扩展、具备抢占能力的新一代推理架构。**

</details>

---

### 2. [Prototype Language Models](https://arxiv.org/abs/2607.00510)

**Authors**: Dan Ley, Giang Nguyen, Himabindu Lakkaraju, Julius Adebayo  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2607.00510v1  

#### Abstract
Knowing which training examples drive outputs is fundamental to auditing, correcting, and understanding language models, yet for modern LLMs this remains expensive, approximate, and largely post-hoc. Standard language models generate tokens through a dense network pathway, causing training data's in...

---

### 3. [SmoothAgent: Efficient Long-Horizon LLM-Based Agent Serving with Lookahead Context Engineering](https://arxiv.org/abs/2607.00151)

**Authors**: Zaifeng Pan, Qianxu Wang, Zhengding Hu, Chang Chen, Yue Guan, Yanbo Zhou, Steven Swanson, Yufei Ding  
**Category**: cs.DC  
**Published**: 2026-07-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.00151v1  

#### Abstract
LLM-based agents execute multi-turn workflows with continuously growing contexts, where LLM calls are interleaved with tool invocations and environment feedback. To maintain model quality, modern agent frameworks rely on context engineering strategies such as offloading, reduction, and isolation to ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SmoothAgent: Efficient Long-Horizon LLM-Based Agent Serving with Lookahead Context Engineering》总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在基于大语言模型（LLM）的智能体（Agent）系统中，随着多轮交互的进行，上下文（context）不断累积，导致“**context rot**”现象——即模型性能随上下文增长而下降。为应对该问题，现代Agent框架广泛采用**context engineering**策略（如offloading、reduction、isolation），以控制上下文长度。

然而，这些策略会引发**context transformation overhead**：每次变换都会使已有的KV cache失效，迫使系统重新执行prefill阶段，从而显著增加**Time-to-First-Token (TTFT)**，尤其是在长尾延迟（tail latency）方面表现严重，影响用户体验和服务等级目标（SLO）。

### **提出了什么新方法或新思路**

本文提出 **SmoothAgent**，一种通过**前瞻式上下文工程（lookahead context engineering）** 来消除转换开销的高效Agent服务系统。其核心思想是：

- 发现常见的context transformation具有**segment-decomposable**性质：即对前缀的变换独立于后续token，因此可以提前计算。
- 基于此，设计了一个**lookahead编程模型**，允许将context transformation表达为异步操作，在后台预先计算并缓存变换后的KV cache。
- 当触发正式变换时，可直接替换上下文前缀，避免阻塞式re-prefill。

此外，还设计了**lookahead-aware调度器**，确保异步的lookahead请求不会干扰主路径上的latency-critical（LC）请求，实现共存且可控的资源利用。

### **相比现有方法的优势**

| 方面 | 传统方法 | SmoothAgent |
|------|--------|------------|
| 执行方式 | 同步执行，阻塞生成流程 | 异步预计算，非阻塞提交 |
| KV Cache 利用 | 每次变换后全部失效 | 预先构建并复用 |
| TTFT 影响 | 显著增加，尤其在变换点出现尖峰 | 几乎无尖峰，TTFT曲线平滑 |
| 资源调度 | 无区分对待，易造成干扰 | 区分LC与BE请求，保障SLO |

> ✅ **优势总结**：有效隐藏context transformation开销，提升响应速度和平滑度，同时不牺牲模型质量或系统稳定性。

---

## 2. 核心实验方法和设置

### **使用的模型与硬件平台**

- **模型**：
  - `Qwen3-8B` 和 `Qwen3-32B`（最大上下文长度32K tokens）
- **硬件**：
  - NVIDIA H100 GPU × 8（80GB HBM，NVLink互联）
  - AMD EPYC CPU 主机
- **部署配置**：
  - **PD co-located**：prefill与decode共享GPU实例（单卡或TP=4）
  - **PD disaggregated**：prefill与decode分离部署，通过NIXL传输KV cache

### **工作负载（Workloads）**

- 多步代码分析任务，运行于 **MiniAgent** 框架之上。
- 每一步调用shell命令读取源码（如`head`, `tail`, `sed`），生成分析回复。
- 每步新增约600–650 tokens，共28步，上下文单调增长。
- 每个agent拥有独立prompt，防止radix cache共享抑制增长。

### **评估指标**

| 指标 | 描述 |
|------|------|
| **TTFT (Time-to-First-Token)** | 关注**transform-point处的尾部TTFT（p99）**，反映最差延迟情况 |
| **TBT (Time-Between-Tokens)** | 解码过程中的逐token延迟 |
| **Throughput** | 单位时间内完成的请求量 |
| **SLO Violation Rate** | 是否违反预设延迟目标（如TTFT ≤ 500ms） |

### **基线方法对比**

- **Baseline**：同步context engineering（sync）
  - 包括：`offloading`、`keep-recent-K`、`summarization`、`sub-agent isolation`
  - 变换在critical path上即时触发并执行
- **SmoothAgent**：启用lookahead机制的相同策略
- 所有方法均基于 **SGLang v0.5.9** 作为底层LLM serving引擎

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ 在 **PD co-located 设置下**（图10–11）：

| 策略 | 模型 | 平均TTFT降低 | 最高加速比 |
|-------|------|---------------|-------------|
| Offloading | Qwen3-8B | ~62% | 2.6× |
| Keep-Recent-K | Qwen3-8B | ~62% | 2.7× |
| Summarization | Qwen3-8B | ~62% | **11.9×** |
| Sub-Agent Isolation | Qwen3-8B | ~62% | 3.8× |

> 🔺 **Summarization收益最大**：因其涉及额外LLM推理（生成摘要）+ 完整re-prefill，开销最高，lookahead能完全将其移出关键路径。

#### ✅ 在 **PD disaggregated 设置下**（图13）：

- 支持高达64个并发agent。
- 平均TTFT降低 **64.5%**。
- 尽管存在KV cache跨实例传输开销，lookahead仍能有效消除变换引起的延迟尖峰。

#### ✅ 在多个主流Agent框架中集成验证（LangChain、LlamaIndex、AutoGen、OpenClaw）：

- 替换原生策略为lookahead版本，无需修改业务逻辑。
- 在所有框架中，**transform-point TTFT降低26.8% ~ 80.5%**，证明通用性强。

### **消融实验与调度器有效性（Section 6.3）**

- **Lookahead-aware scheduler vs Vanilla SGLang**：
  - 当lookahead流量增加时，vanilla调度器导致LC请求的TTFT p99迅速超标。
  - SmoothAgent通过**slack-aware co-batching**和**TBT约束调度**，成功隔离干扰，保持低延迟。
- **性能建模准确性**（图15）：
  - 传统仅基于token数的预算模型误差达 **41.5%**
  - 本文提出的**context-aware模型**（考虑KV cache长度）误差降至 **13.7%**，更可靠地指导调度决策。

---

## 4. 关键结论和发现

### **主要发现**

1. **Context transformation overhead 是Agent系统的重大瓶颈**，尤其在长周期任务中频繁触发变换时严重影响TTFT。
2. 大多数常用context engineering策略（offloading, reduction, isolation）天然具备 **segment-decomposable** 属性，支持异步预处理。
3. **Lookahead编程模型 + lookahead-aware调度器** 可有效解耦变换与主流程，实现“零感知”上下文管理。
4. 实验表明，SmoothAgent可将变换点的TTFT降低 **最多达11.9倍**，显著改善尾延迟和平滑性。
5. 该方法具有良好的**通用性和可集成性**，已在多个主流Agent框架中成功部署。

### **方法的局限性**

| 局限性 | 说明 |
|--------|------|
| **依赖segment-decomposability** | 若某些复杂变换无法分解（如全局重排序），则难以应用lookahead |
| **lookahead请求可能未及时完成** | 若系统负载过高，lookahead可能fallback到同步执行，但仍优于baseline |
| **额外内存消耗** | 需维护lookahead状态和KV cache副本，带来一定内存开销 |
| **不适合极短生命周期任务** | 对话过短时，lookahead来不及生效 |

### **未来工作方向**

1. **扩展至更多非decomposable变换**：探索近似分解或增量更新机制。
2. **动态调整lookahead粒度**：根据上下文变化趋势自适应选择segment边界。
3. **结合RAG与lookahead retrieval**：如TeleRAG思路，提前检索相关内容。
4. **商业化API设计**：提供“best-effort lookahead”服务等级，按利用率计费，激励用户采用。
5. **支持MoE模型下的lookahead attention dispatching**：结合disaggregated expert parallelism进一步优化。

---

> 📌 **总体评价**：  
> SmoothAgent首次系统性识别并量化了context transformation带来的性能代价，并提出了一套实用、高效、可集成的解决方案。它不仅提升了Agent系统的响应效率，也为未来构建**真正可持续运行的长期智能体**提供了重要基础设施支持。

</details>

---

### 4. [Decision-focused Sparse Tangent Portfolio Optimization](https://arxiv.org/abs/2607.00581)

**Authors**: Haeun Jeon, Seunghoon Choi, Hyunglip Bae, Yongjae Lee, Woo Chang Kim  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.00581v1  

#### Abstract
Sparse tangent portfolio optimization aims to learn an interpretable, low-cardinality portfolio in the tangency direction of the mean-variance frontier. However, the associated cardinality-constrained formulation is NP-hard, and standard predict-then-optimize pipelines often misalign forecasting acc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Decision-focused Sparse Tangent Portfolio Optimization**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
传统投资组合优化中的 **predict-then-optimize (PFL)** 范式存在严重缺陷：预测模型通常以最小化预测误差（如 MSE）为目标进行训练，而该目标与最终的投资组合决策质量（如 Sharpe ratio）并不对齐。尤其在**稀疏约束**（cardinality constraint）下，微小的预测误差可能通过离散资产选择阶段被放大，导致次优甚至失效的投资组合。

此外，**稀疏切线投资组合优化**（sparse tangent portfolio optimization）是一个 NP-hard 的组合优化问题，难以直接求解，且现有启发式方法（如 OSCAR）缺乏端到端的学习能力。

### **提出了什么新方法或新思路**
本文提出了一种**端到端的 Decision-focused Learning (DFL) 框架**，用于解决带显式基数约束的稀疏切线投资组合优化问题。其核心创新在于构建了一个**可微分的决策层**（differentiable decision layer），将整个“预测 → 资产选择 → 再优化”流程嵌入神经网络中，实现梯度反向传播。

具体技术贡献包括：
- **DPP-compliant 凸优化层**：将 Sharpe ratio 最大化问题通过**同质化变换**（homogenization）转化为一个符合 **Disciplined Parametrized Programming (DPP)** 规则的凸二次约束二次规划（QCQP），从而可以使用 `CVXPYlayers` 进行可微分求解。
- **可微分 Top-k 操作符**：用一个**平滑的 Top-k 选择器**（soft top-k operator）替代传统的硬阈值选择，该操作符能精确满足 `sum-to-k` 的基数约束，并允许梯度流过资产选择步骤。
- **端到端训练**：预测模型 $f_\theta$ 的参数 $\theta$ 通过最小化下游投资组合的**任务损失**（task loss，如 regret）来更新，而非预测损失，使模型学习直接优化决策质量。

### **相比现有方法的优势**
- **目标对齐**：DFL 直接优化投资组合性能（如 Sharpe ratio），而非预测准确性，解决了 PFL 中的目标错配问题。
- **性能更优**：在多个市场和资产规模下，DFL 在 out-of-sample Sharpe ratio 上显著优于历史均值法和多种 PFL 基线，尤其是在**大规模资产集合**中优势更为明显。
- **结构整合**：首次将 DFL 成功应用于带显式基数约束的稀疏切线投资组合优化，实现了从预测到决策的完整可微分管道。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- 四个主要股票市场指数的日频收盘价数据（来自 Yahoo Finance，2016–2025）：
  - **EuroStoxx50** (N ≈ 47)
  - **FTSE100** (N ≈ 93)
  - **KOSPI200** (N ≈ 162)
  - **Nikkei225** (N ≈ 208)

### **实验设置**
- **输入特征**：过去 100 天的滚动收益率窗口。
- **预测目标**：下一期的资产收益率向量 $\mu$。
- **协方差矩阵**：使用相同滚动窗口估计，固定不变，不参与学习。
- **训练/测试划分**：按时间顺序 80%/20% 划分。
- **模型架构**：两层全连接网络（512 → 256 隐藏单元）。
- **混合损失函数**：
  $$
  \mathcal{L}_{\text{Task}} = \alpha \mathcal{L}_{\text{DFL}} + (1-\alpha) \mathcal{L}_{\text{MSE}}
  $$
  其中 $\alpha = 0.5$，平衡决策质量和预测准确性。

### **评估指标**
- **主指标**：**Out-of-sample Sharpe Ratio**（日频）
  $$
  \text{SR} = \frac{\bar{r}_p}{s_p}
  $$
  其中 $r_{p,t} = w_t^\top r_{t+1}$ 为组合收益。
- **辅助指标**：Maximum Drawdown (MDD)，用于分析风险特征。

### **基线方法对比**
#### **常规学习框架**
1. **Historic**：使用样本内历史均值作为预期收益，直接优化。
2. **PFL**：先训练预测模型，再将其输出用于下游优化。

#### **优化模型（用于 PFL 和 DFL 的下游）**
1. **OSCAR**：三阶段启发式方法（优化 → Cholesky 变换空间选 Top-k → 再优化）。
2. **SD-relaxation**：基于半定规划（SDP）的松弛方法。
3. **mSSRM-PGA**：基于近端梯度算法的稀疏优化方法。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**
| 市场 | 资产数 | 方法 | p=10% | p=15% | p=20% |
|------|--------|------|-------|-------|-------|
| **KOSPI200** | 162 | **DFL (ours)** | **2.030** | **2.098** | **2.098** |
| | | PFL + SD-relaxation | 1.684 | 1.793 | 1.793 |
| | | PFL + mSSRM | 1.491 | 1.586 | 1.586 |
| **Nikkei225** | 208 | **DFL (ours)** | **0.878** | **0.878** | **0.946** |
| | | PFL + SD-relaxation | 0.812 | 0.886 | 0.886 |

> 注：DFL 在所有市场和多数设置下取得最高或接近最高的 Sharpe ratio，尤其在 **KOSPI200** 和 **Nikkei225** 等大规模资产集合中表现远超基线。

### **与基线方法的对比结果**
- **Historic 基线最弱**：表明仅依赖历史均值无法适应非平稳市场。
- **PFL 表现不稳定**：依赖于所选优化器；例如 SD-relaxation 在小市场（EuroStoxx50）表现尚可，但在大市场波动大、稳定性差。
- **DFL 显著胜出**：
  - 在 **KOSPI200** 上，DFL 的 Sharpe ratio 比最佳 PFL 基线高出约 **20%~30%**。
  - 在 **FTSE100** 和 **Nikkei225** 上也持续领先。
  - 方差更小，说明结果更稳定。

### **消融实验结果（见 Table 2, C.1）**
- **$\alpha$ 敏感性分析**（混合权重）：
  - $\alpha = 0$（纯 MSE）：性能最差，验证了仅预测准确不足以保证好决策。
  - $\alpha = 1$（纯 DFL）：平均性能高但方差极大，训练不稳定。
  - $\alpha = 0.5$：**最佳权衡点**，在提升性能的同时保持低方差，是推荐默认设置。
- **结论**：混合损失 $\mathcal{L}_{\text{Task}}$ 是稳定训练的关键，纯 DFL 虽潜力大但易受初始化影响。

---

## 4. **关键结论和发现**

### **主要发现**
1. **DFL 显著提升稀疏投资组合性能**：通过端到端对齐预测与决策目标，DFL 能学习出更有利于下游 Sharpe ratio 最大化的预测，尤其在**大规模资产环境**中效果最为显著。
2. **可微分 Top-k + DPP 优化层可行**：成功构建了完整的可微分稀疏投资组合管道，使梯度能够流经预测、软 Top-k 选择和再优化全过程。
3. **目标错配问题真实存在**：PFL 框架即使预测准确，也可能因未考虑优化结构而导致劣质决策；DFL 有效缓解此问题。
4. **性能增益伴随更高风险**（见 Table 3）：
   - DFL 投资组合常伴随更高的 **Maximum Drawdown (MDD)**，表明其策略更具进攻性（允许卖空）。
   - 更高的 Sharpe ratio 是以增加下行风险为代价的，需结合实际约束使用。

### **方法的局限性**
1. **计算开销大**：可微分优化层和软 Top-k 引入额外计算负担，尤其在资产数量 $N$ 或基数 $k$ 较大时。
2. **允许卖空**：当前框架未限制 long-only 或杠杆，不符合许多实际投资场景。
3. **未考虑交易成本**：忽略 turnover 和 transaction cost，可能导致频繁调仓。
4. **特定于 Sharpe ratio 目标**：框架目前针对稀疏切线组合设计，扩展至其他目标（如 VaR、CVaR）需重新建模。

### **未来工作方向**
- 设计更高效的可微分选择机制（如基于 Gumbel-Softmax 或其他稀疏注意力）以提升可扩展性。
- 引入实际约束：long-only、杠杆限制、turnover penalty、transaction cost 等。
- 探索 DFL 在其他 cardinality-constrained 投资组合问题中的应用，如 risk parity、goal-based investing 等。
- 结合多因子模型或图神经网络提升预测能力。

---

> ✅ **总结一句话**：  
> 本文提出首个用于稀疏切线投资组合优化的 **Decision-focused Learning** 框架，通过构建可微分的 DPP 优化层与软 Top-k 选择器，实现端到端训练，在多个市场（尤其是大规模资产集合）上显著提升了 out-of-sample Sharpe ratio，验证了“以决策为导向”的学习范式在金融投资中的巨大潜力。

</details>

---

### 5. [TiRex-2: Generalizing TiRex to Multivariate Data and Streaming](https://arxiv.org/abs/2607.01204)

**Authors**: Patrick Podest, Marco Pichler, Elias B\"urger, Levente Z\'olyomi, Bernhard Voggenberger, Wilhelm Berghammer, Daniel Klotz, Sebastian B\"ock, G\"unter Klambauer, Sepp Hochreiter  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2607.01204v1  

#### Abstract
We introduce TiRex-2, a recurrent xLSTM-based time series foundation model that generalizes the univariate TiRex to multivariate forecasting with both past and future covariates. Real-world forecasting is inherently sequential: observations arrive continuously, variables evolve jointly, and a subset...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TiRex-2: Generalizing TiRex to Multivariate Data and Streaming**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的时间序列基础模型（TSFMs）在处理**多变量预测**时存在以下关键瓶颈：
- **Transformer-based 模型**（如 Chronos-2、MOIRAI）虽然能建模跨变量依赖，但其注意力机制导致推理复杂度随上下文长度呈**二次增长**，且每次新观测到来时需重新计算整个历史，无法支持高效流式预测。
- 多数模型难以同时有效利用**过去协变量**（past covariates）和**未来已知协变量**（future-known covariates），同时保持对目标变量的**严格因果性**（target causality）。

### **提出的新方法与新思路**
作者提出了 **TiRex-2**，一个基于 **xLSTM** 的**循环式多变量时间序列基础模型**，其核心创新包括：

#### ✅ **1. 循环式架构实现常量成本流式推理**
- 采用 **memory-centric recurrent design**，基于 **xLSTM** 构建时间混合器（Time Mixer），使得每个新时间块的更新成本为**常量**（constant per-patch cost），而非随上下文线性增长。
- 支持真正的**流式预测**（streaming inference），适用于实时系统。

#### ✅ **2. 异步分组注意力变量子混合器（Asymmetric Grouped-Attention Variate Mixer）**
- 在变量子混合器中引入**非对称注意力掩码**：允许目标查询读取协变量键值，但禁止协变量查询读取目标键值。
- 这种设计使得模型可以**双向处理未来已知协变量**（例如日历特征、促销计划），同时保证目标变量的表示不会泄露未来信息，从而维持严格的因果性。

#### ✅ **3. 合成多变量耦合预训练管道（Synthetic Multivariate Coupling Pipeline）**
- 利用大规模单变量语料库，在训练时动态合成具有多样化跨变量依赖的多变量样本。
- 耦合机制包括：线性混合、非线性SCM、协整关系、函数映射等，增强模型对真实世界复杂依赖的泛化能力。

### **相比现有方法的优势**
| 特性 | TiRex-2 | Chronos-2 | MOIRAI | Toto |
|------|--------|----------|--------|-------|
| 多变量支持 | ✅ | ✅ | ✅ | ❌ |
| 过去协变量 | ✅ | ✅ | ✅ | ❌ |
| 未来已知协变量 | ✅ | ✅ | ✅ | ❌ |
| 目标因果性（Target Causality） | ✅ | ❌ | ❌ | ✅ |
| 流式推理（Constant Memory） | ✅ | ❌ | ❌ | ❌ |
| 上下文长度扩展性 | ✅（线性） | ❌（二次） | ❌（二次） | ✅ |

> 🔍 **首次实现**：支持多变量 + 协变量 + 双向利用未来协变量 + 严格目标因果 + 常量流式推理。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **fev-bench** (Shchur et al., 2025)：强调协变量利用的真实世界多变量基准，涵盖多种频率与领域。
- **GIFT-Eval** (Aksu et al., 2024)：通用时间序列预测评估基准，测试跨域、跨频、长视界泛化能力。
- **dysts** (Gilpin, 2023)：混沌动力系统轨迹数据集，用于长视界预测压力测试。
- **合成数据**：用于消融实验与流式行为分析。

### **实验设置**
- **预训练**：70万步，2×H100 GPU，bf16混合精度。
- **后训练（posttraining）**：额外10万步，将上下文从 2k 扩展到 8k，预测视界至 512。
- **输入长度**：最多 8,192 步；输出视界：32–512 步。
- **patch size**：32 步/patch。
- **参数规模**：
  - 单变量模式：38.4M active parameters
  - 多变量模式：额外激活 44.1M 参数（总计约 82.5M）

### **评估指标**
| 基准 | 主要指标 | 说明 |
|------|---------|------|
| fev-bench | **SQL**（Scaled Quantile Loss）、**Pairwise Win Rate**、**Skill Score** | 概率预测质量，scale-free |
| GIFT-Eval | **MASE**（Mean Absolute Scaled Error）、**CRPS**（Continuous Ranked Probability Score） | 点预测与概率预测综合评价 |
| 流式测试 | **MASE vs. Streamed Steps** | 验证模型在超长上下文下的稳定性 |
| 长视界测试 | **MASE vs. Horizon**（log scale） | 在混沌系统上评估长期依赖捕捉能力 |

### **基线方法对比**
#### 在 **fev-bench** 上对比：
- Chronos-2、MOIRAI-2、Toto-1.0、TiRex、TimesFM-2.5、Chronos-Bolt

#### 在 **GIFT-Eval** 上对比：
- Chronos-2-Synth、PatchTST-FM-r1、TiRex、TimesFM-2.5、FlowState-r1.1

> 所有比较均为 **zero-shot** 设置，确保公平性。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | fev-bench MASE | GIFT-Eval MASE |
|------|----------------|----------------|
| **TiRex-2** | **1.527** | **0.690** |
| Chronos-2 | 1.600 | 0.705 |
| MOIRAI-2 | 1.700 | — |
| Toto-1.0 | 1.750 | — |
| TimesFM-2.5 | 1.700 | 0.720 |
| FlowState-r1.1 | — | 0.705 |

> ✅ TiRex-2 在两个基准上均达到 **state-of-the-art zero-shot 性能**。

#### **Pairwise Win Rate (fev-bench)**：
- TiRex-2 对其他模型的胜率达到 **94%–100%**，显著领先。

#### **Skill Score (fev-bench)**：
- 平均相对误差降低超过 **45%**，优于所有基线。

### **与基线方法的对比结果**
- **流式推理稳定性**：TiRex-2 可稳定处理长达 **3200万步** 的连续输入（>4000×训练上下文），MASE 无漂移或饱和现象（见 Fig. 6 左图）。
- **长视界预测**：在 dysts 数据集上，TiRex-2 在中短视界表现最优，尤其在细粒度（fine granularity）任务上明显优于 Chronos-2。
- **协变量偏移鲁棒性**：当协变量相对于目标存在时间偏移（△）时，TiRex-2 能更有效地利用信号，即使在较大滞后下仍优于基线（Fig. 6 右图）。

### **消融实验结果（Ablation Study）**
在 fev-bench 上进行组件消融（Table 1）：

| 配置 | MASE Δ (All) | MASE Δ (Future-Cov Subset) |
|------|-------------|----------------------------|
| 完整模型（Full） | 1.527 ± 0.006 | 0.990 ± 0.007 |
| 移除 Group Attention | +0.220 | +0.278 |
| 移除 Binary-aware Scaler | +0.014 | +0.001 |
| 仅前向处理（Forward-only） | +0.022 | +0.077 |
| 仅 sLSTM（无 mLSTM） | +0.008 | +0.002 |
| 无未来协变量输入 | +0.033 | +0.114 |

> 🔍 **关键发现**：
> - **Group Attention 最重要**：移除后性能大幅下降，验证其对跨变量依赖建模的关键作用。
> - **双向协变量处理至关重要**：尤其在“未来已知协变量”子集上影响显著。
> - **Binary-aware scaler 提升小但稳定**：防止稀疏二元变量标准化失真。

---

## **4. 关键结论和发现**

### **主要发现**
1. **循环架构可媲美甚至超越 Transformer-based TSFMs**：尽管 xLSTM 是递归结构，但在 zero-shot 多变量预测中实现了 SOTA 性能。
2. **异步注意力是实现“双向协变量 + 严格因果”的关键**：通过禁止协变量读取目标，打破了传统注意力的信息泄露路径。
3. **合成多变量耦合策略有效弥补真实多变量数据不足**：无需依赖有限的真实多变量语料，即可训练出强泛化能力的模型。
4. **流式推理天然支持任意长度上下文**：得益于常量状态更新，TiRex-2 可无缝应用于工业级长时间序列监控场景。

### **方法的局限性**
- **未来协变量更新需重计算**：虽然目标和过去协变量支持流式更新，但若未来协变量在线变化（如临时取消活动），则需重新编码其表示。
- **patch granularity 限制响应延迟**：当前以 32 步为单位处理，不适合亚 patch 级别的高频决策。
- **未支持动态变量选择**：所有协变量始终参与计算，缺乏运行时剪枝机制以节省 FLOPs。

### **未来工作方向**
- **动态协变量选择机制**：基于上下文估计协变量相关性，动态跳过无关协变量，提升效率。
- **更灵活的时间对齐机制**：支持非均匀采样或多速率输入。
- **扩展至高维时空图结构**：结合图神经网络处理空间-时间联合建模任务（如交通、气象）。
- **探索 in-context learning for covariate adaptation**：让模型学会根据提示自动识别并利用新型协变量。

---

> 📌 **总结一句话**：  
> **TiRex-2 是首个兼具多变量建模、协变量支持、严格因果性和常量流式推理能力的时间序列基础模型，通过创新的 recurrent + asymmetric attention 架构与合成数据策略，在 zero-shot 场景下实现了全面领先的性能。**

</details>

---

### 6. [Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training](https://arxiv.org/abs/2607.00368)

**Authors**: Xiangchen Song, Zhenhao Chen, Lingjing Kong, Shaoan Xie, Xinshuai Dong, Guangyi Chen, Kun Zhang  
**Category**: cs.CL  
**Published**: 2026-07-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2607.00368v1  

#### Abstract
Large language model test-time training (TTT) is often evaluated through local proxy metrics: models are updated on recent tokens, retrieved context, target-domain data, or verifiable task attempts, and then judged by perplexity, future-token loss, long-context performance, or reward. These metrics ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前 Large Language Model (LLM) 的 **Test-Time Training (TTT)** 研究普遍依赖于局部代理指标（proxy metrics）进行评估，如 **perplexity**、**future-token loss**、**long-context accuracy** 或 **reward**。这些指标虽然能有效衡量模型在流式适应（stream adaptation）、领域适应或上下文压缩方面的进展，但被广泛用于支持更强的部署级主张，例如“模型具备部署后记忆”、“个性化能力”或“稀疏学习能力”。

然而，作者指出，这些代理指标与真正的 **deployment-time behavioral learning**（部署时行为学习）之间存在显著差距。这种将证据从一个评估范畴迁移到另一个更强叙事中的现象被称为 **evidence migration**（证据迁移），导致对模型实际能力的误判。

### 提出的新方法与新思路
为解决此问题，论文提出了一个 **claim-calibrated behavioral evaluation framework**，其核心是两个组成部分：

1. **Claim-Calibrated Evidence Ladder（声明校准的证据阶梯）**  
   将 TTT 相关主张分为三个层级：
   - **S-Level (Stream/Domain)**：仅支持流式或领域适应（如降低 perplexity）。
   - **B-Level (Bridge/Internalization)**：支持内部化机制（如参数化记忆、上下文吸收），但未验证稀疏交互后的持久行为。
   - **D-Level (Deployment-Time/Behavioral)**：要求模型在原始支持上下文移除后，仍能在延迟、转述、冲突等条件下正确调用信息并影响行为。

2. **Evaluation Protocol with Matched Baselines**  
   引入一套标准化评估协议，强调：
   - 必须报告 **behavioral evidence**（如 recall、paraphrase robustness、retention、locality、conflict handling）。
   - 必须与 **matched explicit-memory baselines**（如 exact-context prompting、BM25 retrieval、replacement memory）进行公平比较。
   - 明确区分 **proxy metrics**（如 loss 改善）与 **behavioral outcomes**（如生成回忆）。

### 相比现有方法的优势
- **更严格的 claim calibration**：防止将技术进步误读为部署级能力突破。
- **可操作的评估标准**：提供具体模板（如 Table A3）指导如何设计 D-level 测试。
- **揭示真实能力边界**：通过实验证明，即使 proxy metrics 显著改善，behavioral recall 可能仍为零，暴露当前研究的盲区。

---

## 2. 核心实验方法和设置

### 数据集
实验采用 **controlled diagnostic setting**，构造了一个 **sparse nonce-fact**（稀疏一次性事实）任务，不依赖公开数据集。具体包括：
- 注入虚构的访问码事实，如：“The access code for the zuneth-ledger is nexil-774.”
- 设计三种查询方式：
  - **Direct**：直接提问该码。
  - **Paraphrased**：改写问题形式。
  - **Delayed**：中间插入无关任务后再提问。

### 实验设置
- **模型**：使用 **Qwen3** 系列模型的三个规模：
  - Qwen/Qwen3-1.7B
  - Qwen/Qwen3-4B
  - Qwen/Qwen3-8B
- **更新机制**：采用 **one-step LoRA update** 在测试时对支持句子进行微调。
- **上下文处理**：支持上下文在查询前被显式移除，模拟真实部署场景。
- **解码策略**：固定为 **greedy decoding**，避免采样引入噪声。

### 评估指标
| 类型 | 指标 |
|------|------|
| **Proxy Metrics** | △NLL (NLL_after - NLL_before)，负值表示改善 |
| **Behavioral Metrics** | Generated free-form recall success (%)，基于归一化首行答案匹配 |
| **Robustness Tests** | Paraphrased recall, Delayed recall, Locality preservation, Conflict overwrite |
| **Baselines** | BM25-style retrieval, Replacement memory, Exact-context prompting |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 3）
在所有三个 Qwen3 模型上，**one-step LoRA 更新均显著降低了 loss**，但 **free-form recall 成功率为 0.0%**：

| Model | Support △NLL ↓ | Answer △NLL ↓ | Greedy Recall (%) |
|-------|----------------|---------------|--------------------|
| Qwen3-1.7B | -1.151 ± 0.051 | -0.674 ± 0.085 | 0.0 (Direct/Paraphrased/Delayed) |
| Qwen3-4B | -1.218 ± 0.093 | -0.529 ± 0.081 | 0.0 |
| Qwen3-8B | -0.966 ± 0.026 | -0.504 ± 0.079 | 0.0 |

> ✅ **Proxy improvement**: 所有模型在支持句重建和答案 NLL 上均有显著下降。  
> ❌ **Behavioral failure**: 但在自由生成中，无一例成功回忆注入的事实。

### 与基线方法的对比
- **BM25 Retrieval Baseline**：在相同条件下，BM25 能以 **100% Hit@1** 检索到正确事实，并成功生成答案。
- **Replacement Memory**：在冲突覆盖测试中，replacement memory 在多数情况下能正确覆盖旧信息（corrected-only），而 LoRA 更新则完全失败。

### 消融实验结果
#### （1）更强的更新（16-step LoRA）
- 提升 recall 至：
  - Direct: 72.9% (35/48)
  - Paraphrased: 54.2% (26/48)
  - Delayed: 72.9%
- 但 **locality**（无关行为稳定性）从 97.9% 骤降至 **9.7%**，表明存在严重干扰。

#### （2）冲突覆盖测试（Mutually Exclusive Scoring）
| Model | Method | Corrected-only | Both | Neither |
|-------|--------|--------------|------|---------|
| Qwen3-1.7B | LoRA | 0 | 0 | 24 |
| Qwen3-1.7B | Replacement Memory | 20 | 4 | 0 |

> LoRA 更新在所有冲突中均未能输出任何代码，说明其既未保留旧信息也未激活新信息，属于 **behavioral access failure**。

#### （3）其他任务泛化
在 preference、correction 和 procedure mini-tasks 中，同样观察到：
- Proxy loss 下降
- Free-form behavior 保持 0%
- 表明该现象具有普遍性，不限于 access-code 字符串。

---

## 4. 关键结论和发现

### 主要发现
1. **Proxy Metrics ≠ Deployment Behavior**  
   即使模型在 **teacher-forced loss** 上显著改善，也无法保证其能在开放生成中正确回忆或应用所学信息。

2. **Evidence Migration 是普遍问题**  
   当前大多数 TTT 工作停留在 S- 或 B-Level 证据，却被引用支持 D-Level 主张（如“实现个性化记忆”），缺乏必要的 behavioral validation。

3. **Behavioral Recall 与 Interference 存在权衡**  
   更强的更新虽可提升 recall，但会严重损害 **locality**，即破坏模型原有知识，不符合安全可靠的部署要求。

4. **Explicit Memory 是合理基线**  
   对于部署级任务，**retrieval-augmented** 或 **external memory systems**（如 MemGPT、MemoryBank）应作为默认比较对象，而非仅与 no-update 对比。

### 方法的局限性
- 本框架是一个 **evaluation framework**，而非提出新的 TTT 算法。
- 实验仅基于 **LoRA + Qwen3**，未涵盖其他 TTT 方法（如 fast weights、meta-learning）。
- 控制实验规模较小，尚未扩展至多轮对话、长期记忆或真实用户交互。

### 未来工作方向
- 构建统一的 **Deployment-Memory Benchmark**，集成 recall、paraphrase、delay、conflict、locality 等维度。
- 探索 **hybrid architectures**：结合 parametric TTT 与 explicit memory，在隐私、延迟、压缩等约束下优化 trade-offs。
- 加强 **governance-oriented evaluation**：纳入 deletion/forgetting、cross-user isolation、consent scoping 等伦理维度（见 Table A19）。
- 推动社区实践：要求论文在声称“记忆”或“个性化”时，必须报告对应的 D-Level behavioral evidence。

---

> **最终结论**：  
> “Lower perplexity does not imply better deployment behavior.”  
> 本文呼吁研究界停止将 **proxy improvement** 等同于 **memory capability**，并提供了一套可操作的标准，以确保 TTT 的 claims 与其 evidence 严格对齐。

</details>

---

### 7. [AGI Maze as a Benchmark Framework for World-Modeling Agents](https://arxiv.org/abs/2607.00627)

**Authors**: Alexey Potapov  
**Category**: cs.AI  
**Published**: 2026-07-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.00627v1  

#### Abstract
Large language models (LLMs) are powerful pattern-completion systems, but their default operating mode - predicting the next token from a static context - does not reliably produce persistent, manipulable representations of an external world. Many tasks that look like "reasoning" in text become subs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*AGI Maze as a Benchmark Framework for World-Modeling Agents*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Large Language Models (LLMs)** 虽然在语言任务中表现出强大的模式补全能力，但其默认的“静态上下文下的 next-token 预测”机制，**无法可靠地构建和维护一个持久、可操作的外部世界状态表示**。这导致它们在面对部分可观测（partially observable）、有记忆需求、需要推理隐藏状态的任务时表现不佳。

论文指出，许多看似“推理”的行为在文本任务中被掩盖，但在真实交互环境中暴露了 LLM 的根本缺陷——缺乏真正的 **world modeling** 能力。

### 🚀 提出的新方法与创新点
作者提出了 **AGI Maze** —— 一个轻量级、基于文本接口的基准框架，用于测试 AI Agent 的 **world modeling** 能力：

- **核心设计**：提供一系列基于网格的迷宫任务（grid-based maze），具有干净的 API 和多种难度等级。
- **强调要素**：
  - **部分可观测性**（Partial Observability）
  - **动态不确定性**（如河流强制移动、陷阱循环传送）
  - **状态依赖动作链**（如需先拿钥匙 → 再取宝箱 → 最后才能逃出）
  - **有限步数预算**（Step Budget），模拟现实中的效率压力
- **支持扩展机制**（Extensions）：
  - 新物品（boat, flashlight, grenade）
  - 新地形（river, pit）
  - 新动作（blast, shine）
  - 用以测试 agent 是否能从交互中**泛化理解新规则**，而非依赖硬编码或过拟合。

### 🔍 相比现有方法的优势
| 现有方法 | 局限性 | AGI Maze 的改进 |
|--------|-------|----------------|
| **ARC-AGI-3** | 多为全观测、静态规则推断，不考验持续状态建模 | 引入动态环境、局部观测、长期记忆挑战 |
| **VLA（Vision-Language-Action）模型** | 在视觉环境中仍停留在特征匹配层面，未解决 SLAM（Simultaneous Localization and Mapping） | 抽象掉像素感知，聚焦于**抽象状态表示与推理** |
| **纯 LLM + RAG/Tool Use** | 依赖外部检索或工具输出，本质仍是文本补全 | 要求 agent 主动构建内部状态表征（如地图、信念状态） |

> ✅ **优势总结**：  
> AGI Maze **剥离了低层次感知复杂性**（如图像处理），专注于评估 agent 是否具备构建、更新和查询**演化世界状态**的能力，是首个明确针对 LLM 类 agent “world modeling 缺失”的轻量级测试平台。

---

## 2. 核心实验方法和设置

### 🧪 数据集与任务分组
AGI Maze 将迷宫划分为五个任务组，形成渐进式评估体系：

| 组别 | 描述 | 用途 |
|------|------|------|
| **TUTORIAL** | 地图公开，教学用途 | 教学引导，不计分 |
| **TRAINING** | 小型迷宫 + 宽松步数限制 | 用于调试、RL训练、基线测试 |
| **CLASSIC** | 更大迷宫，仅含基础规则 | 对标人类表现，衡量强解法器能力 |
| **EXTENDED** | 加入新机制（如河流、陷阱） | 测试对新规则的理解与适应 |
| **HIDDEN** | 私有保留集，含未知机制 | 防止过拟合，验证真正泛化能力 |

> 所有任务通过 HTTP API 提供，支持 agent 接入（`/api/start`, `/api/step` 等）。

### 📊 实验设置与评估指标

#### 输入格式
Agent 接收以下信息作为输入：
- 游戏通用描述（via `GET /api/description`）
- 当前迷宫参数（大小、起点、可用动作等）
- 历史动作-观察序列（textual feedback，如 `"You tried to go right, but a monolith blocks the way"`）

#### 输出与交互流程
- Agent 每步选择一个 action（up/down/left/right）
- 系统返回观察结果 + 当前库存 + 结构化状态字段
- 游戏继续直到成功逃出或超出推荐步数

#### 评估指标
- **Success Rate (%)**：在给定步数预算内完成任务的比例
- 使用两个阈值：
  - **Lower Budget**：接近人类水平的合理上限
  - **Upper Budget**：Lower 的两倍，允许更多探索

#### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Random-Walk Agent** | 每步随机选择方向，作为最弱基线 |
| **Vanilla LLM Agent** | 直接将历史消息喂给 LLM，无额外记忆结构 |
| **Planning Agent (w/ Working Memory)** | 允许 LLM 在 prompt 中写笔记（如坐标、已探区域、计划路径），再决定行动 |

测试模型包括：
- GPT-4o Mini
- GPT-5 Mini
- Gemini 3.1 Flash Light
- MiniMax M2.7
- GPT-5.5 / Gemini 3.5 Flash（更强版本用于扩展测试）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Maze | Size | Step Budget | Random | GPT4oM | GPT5m | Gem31fl | Minimax27 |
|------|------|-------------|--------|--------|--------|----------|-----------|
| S0-01 | 3×3 | 70 (35) | 24.3 | 11 | 65 | **79** | 28 |
| S0-03 | 4×4 | 160 (80) | 9.2 | 4 | 8 | **39** | 17 |
| S3-03 | 5×4 | 200 (100) | 3.2 | 0 | 3 | **1** | 0 |

> 注：括号内为 Lower Budget 下的成功率；每个条目为 100 次试验的平均成功率。

#### 观察要点：
- 在简单迷宫（S0，只有 key/chest）中，部分 LLM（如 Gemini 3.1 Flash）优于 random walk。
- 随着迷宫变大（4×4 → 5×4）或引入动态机制（river/pit in S3），所有轻量级 LLM 表现急剧下降。
- **GPT-4o Mini 在某些情况下甚至系统性劣于 random-walk agent**，说明其策略不稳定。

### 🔁 强模型 + Planning Agent 改进效果（S3-03，双倍步数）

| Model | Vanilla LLM | w/ Working Memory | 提升幅度 |
|-------|--------------|--------------------|---------|
| GPT-5.5 | ~30% | → 60% | +30pp |
| Gemini 3.5 Flash | ~40% | → 70% | +30pp |
| 轻量级 LLMs | ~0% | → 0% | 无改善 |

> ✅ 显示：**更强的 LLM 可以利用 prompt 作为 working memory 来提升性能**，但轻量模型即使有此机制也无效。

### 🔍 消融分析（隐含）
虽然没有显式消融实验，但从不同 maze 类型的表现差异可得出：
- **加入 River/Pit 后性能骤降** ⇒ 动态不确定性显著增加建模难度
- **仅靠 message history 不足以维持稳定状态表示** ⇒ LLM 内部 latent space 并未自动构造 map-like 结构
- **notes 显著提升表现** ⇒ 外部化状态（externalization）是关键突破口

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Vanilla LLM Agents 普遍失败于基本迷宫任务**，尤其是在引入动态机制（rivers, pits）或增大规模后，连轻度挑战都无法应对。
2. **LLM 并未在 inference 过程中自动构建稳定的环境状态表示**，其行为更接近“高效预测器”而非“世界模拟器”。
3. **允许使用 prompt 作为 working memory（记笔记）能显著提升强 LLM 的表现**（如 GPT-5.5 和 Gemini 3.5 Flash 成功率翻倍），表明状态外化至关重要。
4. **Memory-as-text 是低效且脆弱的方式**：即便能记笔记，agent 通常也不会主动构建完整地图，记忆管理仍不成熟。
5. **现有 LLM 架构难以处理 belief tracking under uncertainty**，例如当 agent 被 river 强制移动后失去定位时，无法有效维护可能位置的分布。

### ⚠️ 方法的局限性
- **非视觉环境**：虽有意避开 perception 复杂性，但也限制了对多模态 agent 的直接评估。
- **人工设计机制**：rivers、pits 等虽具挑战性，但仍属人为构造，未必覆盖真实世界的复杂 dynamics。
- **API 依赖性**：目前依赖中心化服务，不利于大规模分布式测试。
- **尚未集成高级 memory 架构**（如向量数据库、symbolic memory），当前测试仍偏基础。

### 🔮 未来工作方向
1. **开发具备显式 world model 的 agent**：
   - 如构建可计算的地图（map）、图结构（graph）、约束系统或程序化 simulator 作为外部 artifact。
2. **研究更好的 memory management 机制**：
   - 如如何压缩记忆、选择存储内容、区分 episodic vs semantic knowledge。
3. **结合 RL 进行训练**：
   - 利用 TRAINING 组进行强化学习，优化探索策略与长期规划。
4. **引入不确定性建模**：
   - 使用概率图模型或贝叶斯更新来维护 belief states。
5. **推动 AGI Maze 成为标准 benchmark family**：
   - 鼓励社区提交新 extension，形成持续演进的 test suite。

---

## 总结一句话

> **AGI Maze 揭示了 LLM 在 world modeling 上的根本短板：它们擅长“说”，却不擅长“想”一个持续存在并演化的世界；而真正的 AGI 必须学会在头脑中“画一张地图”。**

</details>

---

### 8. [PedNStream: Scalable Network Flow Simulation for Pedestrian Traffic Management](https://arxiv.org/abs/2607.01021)

**Authors**: Weiming Mai, Dorine Duives, Serge Hoogendoorn  
**Category**: cs.AI  
**Published**: 2026-07-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.01021v1  

#### Abstract
Large-scale crowd management requires pedestrian simulations that are both computationally efficient and compatible with feedback-based control. However, most open-source tools are either microscopic or not designed for network-scale closed-loop evaluation. This paper presents PedNStream (Pedestrian...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PedNStream: Scalable Network Flow Simulation for Pedestrian Traffic Management**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前大规模人群管理缺乏**可扩展且支持闭环控制**的仿真工具。现有工具存在以下局限：
- **微观模型**（如 JuPedSim、Vadere）计算开销大，难以用于城市级网络仿真；
- **宏观模型**（如 LTM）虽高效，但多源自车辆交通建模，未充分考虑行人特有的双向流动、随机行为及活动停留等特性；
- 多数工具不支持**实时反馈控制**（closed-loop control），无法有效评估动态干预策略（如分流、导引、闸控）。

### **提出了什么新方法或新思路**
本文提出 **PedNStream** —— 一个基于 **Link Transmission Model (LTM)** 的开源、Python 原生、面向控制优化的**宏观行人网络流仿真框架**，其核心创新包括：

1. **增强的 LTM 行人动力学建模**：
   - 引入**随机链路动态**（stochastic link dynamics），通过 **diffusion model** 和 **binomial release model** 捕捉自由流下的速度离散性和拥堵时的释放不确定性。
   - 支持**活动诱导的滞留行为**（activity-induced holding），模拟行人短暂停留（如拍照、交谈）对流量的影响。

2. **实用化的 LTM 改进机制**：
   - 使用**实际旅行时间**（actual travel time）替代自由流行程时间，提升拥堵条件下的准确性；
   - 接收容量基于**面积而非长度**，更符合二维行人空间利用；
   - 发送边界采用**混合规则**（blending rule），在轻度拥堵时保留延迟累计计数逻辑，在重度拥堵时平滑过渡到基于占有率的约束。

3. **效用驱动的路径选择模型**：
   - 替代传统 **Dynamic User Equilibrium (DUE)**，采用**时间依赖的 Multinomial Logit (MNL) 模型**进行路径选择；
   - 路径效用综合考虑距离、旅行时间、舒适度，并引入随机扰动项以反映感知波动和个体偏好异质性。

4. **原生支持闭环控制的模块化架构**：
   - 内置控制器接口，支持 **Gater**（调节出入口宽度）和 **Separator**（物理隔离双向流）两类控制器；
   - 提供 **rule-based** 和 **pressure-based** 控制器基线，便于集成强化学习等高级控制算法。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **建模粒度 vs 可扩展性** | 宏观建模实现城市/事件级仿真，远超微观工具的局部场景限制 |
| **控制兼容性** | 原生支持实时状态反馈与控制动作执行，优于需外部封装的间接控制工具（如 SUMO） |
| **行为真实性** | 在保持效率的同时，通过扩散与随机释放机制提升了流量波动与拥堵恢复的逼真度 |
| **开放性与复现性** | 开源 Python 包，配置灵活，支持快速部署与定制 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **合成网络**（Synthetic Networks）：
   - Long Corridor：验证双向干扰；
   - Branching Queues：测试瓶颈引发的排队与溢出；
   - Spike：评估突发需求下的阻塞与恢复；
   - Six Intersections：检验动态路径重选能力。

2. **真实网络**：
   - **Delft 市中心网络**：298 节点，818 链路，模拟 46,501 名行人在 84 分钟内的移动；
   - **Melbourne 城市步行网络**：使用 92 个传感器的行人计数数据进行部分观测验证。

### **实验设置和评估指标**

#### **机制验证实验**
- **目标**：验证队列形成、溢出、拥塞消散、自适应路径选择等现象是否被正确再现。
- **方法**：定性观察仿真快照与时空图（space-time diagram）。

#### **流量模式比较**
- **对比模型**：原始 LTM、Bidirectional LTM、PedNStream。
- **可视化**：绘制 inflow/outflow 时间序列，分析波动性与方向耦合效应。

#### **真实网络评估**
- **评估方式**：
  - 将部分传感器作为输入驱动仿真，其余用于验证；
  - 测试不同观测密度（5、10、20 个传感器）下的重建能力。
- **评估指标**：
  - **GEH < 5 (%)**, **GEH < 10 (%)**：衡量体积一致性（> 更好）；
  - **Volume Ratio**：预测与真实总流量之比（接近 1 更好）；
  - **NRMSE**：归一化均方根误差（↓ 更好）；
  - **NDTW**：归一化动态时间规整距离，衡量时间形状匹配度（↓ 更好）；
  - **KNN** 作为空间插值基线。

#### **闭环控制案例研究**
- 在 Delft 网络中部署 7 个 Gater 控制器；
- 对比无控制、rule-based 控制、pressure-based 控制三种策略；
- 评估指标：平均链路流量、平均步行速度。

#### **运行时分析**
- 固定网络规模与时间步长，变化：
  - 总需求量（Total Demand）
  - OD 对数量
  - 控制器数量
- 报告运行时间（Runtime），分析可扩展性。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 实验 | 结果摘要 |
|------|----------|
| **Delft 大规模仿真** | 298 节点、818 链路，500 步（84 分钟）仅耗时约 **45 秒**（Apple M1 Pro） |
| **Melbourne 验证（单点）** | PedNStream RMSE = **16.6**，优于 Bidirectional LTM（18.2） |
| **Melbourne 网络级验证** | 所有设置下 PedNStream 均显著优于 KNN，尤其在 NRMSE 和 NDTW 上表现更优（见下表） |

#### **Melbourne 部分观测验证结果（Obs=10）**
| Method | GEH<5 (%) | GEH<10 (%) | Vol. Ratio | NRMSE | NDTW |
|--------|-----------|------------|------------|-------|------|
| PedNStream (w/o OD) | 6.00 | 11.56 | 2.33 | **2.51** | **2.05** |
| PedNStream (w/ OD) | 6.22 | 13.00 | 2.31 | 2.56 | 2.13 |
| KNN | 1.67 | 3.22 | 5.92 | 5.35 | 5.23 |

> ✅ PedNStream 在所有指标上全面超越 KNN；加入 OD 信息进一步提升 GEH 表现。

### **与基线方法的对比结果**
- **vs 原始 LTM**：
  - 更真实的双向交互：反向流高时正向流入受限；
  - 更平滑的流量曲线，避免非物理振荡；
  - 更合理的近锁死（near-gridlock）恢复行为（保留少量释放机会）。
- **vs Bidirectional LTM**：
  - 加入扩散与随机释放后，流量更具波动性，更贴近现实；
  - 在 Spike 场景中，PedNStream 允许“挤过缝隙”行为，而 Bi-LTM 几乎完全阻断。

### **消融实验结果**
- **控制器影响实验**（Figure 14）：
  - 运行时间随 **OD 对数量线性增长**，但几乎不受总需求或控制器数量影响；
  - 表明系统瓶颈在于路径选择计算，而非流量传播本身；
  - 即使增加控制器，运行时间仅轻微上升，说明控制接口高效。

- **控制策略对比**（Figure 13）：
  - **Rule-based 控制器**：平均链路流量最高，但波动较大；
  - **Pressure-based 控制器**：平均步行速度更高，行为更平稳，响应更连续；
  - 两者均能有效缓解局部拥堵，证明框架支持有效闭环控制。

---

## **4. 关键结论和发现**

### **主要发现**
1. **PedNStream 成功实现了机制层面的真实性与系统层面的可扩展性统一**：
   - 能准确再现双向干扰、排队溢出、拥塞恢复、动态路径选择等关键行人交通现象；
   - 在城市尺度网络中仍保持秒级仿真速度，适合用于实时控制策略开发与测试。

2. **随机链路动态显著提升流量合理性**：
   - 扩散模型使自由流输出更分散；
   - 随机释放机制让拥堵条件下仍存在“微小通行机会”，避免完全僵死，更符合人类行为。

3. **效用驱动路径选择更适合干预环境**：
   - 相比 DUE，MNL 模型能自然响应临时管制、引导信息等外部干预，支持动态重路由。

4. **原生控制器接口有效支持闭环评估**：
   - Gater 与 Separator 设计贴合物理现实（调节宽度而非直接设定流量）；
   - Rule-based 与 pressure-based 控制器均可改善网络性能，为后续 RL 控制提供良好起点。

### **方法的局限性**
- **宏观建模丢失个体轨迹细节**：无法分析个体避让、群体结构等微观行为；
- **路径选择依赖预生成 k-shortest paths**：可能忽略非常规但可行的绕行路径；
- **参数需校准**：如 `y`, `prel`, `pactivity` 等需依赖实测数据调整，否则泛化能力受限；
- **尚未在极端事件（如恐慌疏散）中验证**：当前侧重日常与节庆场景。

### **未来工作方向**
1. **集成强化学习与优化控制器**：
   - 利用 PedNStream 的高效性训练端到端控制策略（如 DRL）；
   - 探索全局协调控制 vs 局部反馈控制的效果差异。

2. **多模态融合扩展**：
   - 集成自行车、滑板车等慢行交通流，构建综合 urban mobility 平台。

3. **更广泛的实证验证**：
   - 在不同城市、气候、文化背景下收集数据，验证模型普适性；
   - 引入视频追踪数据进行微观行为校准。

4. **在线学习与自适应仿真**：
   - 结合实时传感器数据实现动态参数更新与状态估计（data assimilation）。

---

> 📦 **代码已开源**：[https://github.com/WaimenMak/PedNStream](https://github.com/WaimenMak/PedNStream)  
> PedNStream 为连接**描述性仿真**与**控制导向实验**提供了坚实桥梁，是迈向智能、实时人群管理系统的重要一步。

</details>

---

### 9. [ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving](https://arxiv.org/abs/2607.00466)

**Authors**: Sangjin Choi, Sukmin Cho, Yifan Xiong, Ziyue Yang, Youngjin Kwon, Peng Cheng  
**Category**: cs.DC  
**Published**: 2026-07-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.00466v1  

#### Abstract
In prefill-decode (PD) disaggregated LLM serving, each request is assigned to a decode worker after prefill. Existing decode routers balance only load; for mixture-of-experts (MoE) models this is incomplete: equally loaded workers can differ in latency, since each decode step loads the weights of ev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Prefill-Decode (PD) disaggregated** 架构的 LLM 服务中，现有 decode 路由策略仅关注负载均衡（load balancing），而忽略了 **MoE 模型特有的专家激活模式**。由于 MoE 模型在 decode 阶段是 memory-bandwidth-bound，其延迟主要取决于每个 batch 中被激活的 **distinct expert 数量**，而非请求数量。因此，即使负载均衡，若不同请求激活的专家差异大，仍会导致高延迟。

### 提出了什么新方法或新思路
提出 **ELDR (Expert-Locality-Aware Decode Routing)**，一种面向 PD-disaggregated MoE 服务的 decode 路由机制，核心思想如下：

- **Expert Locality 作为可预测的路由轴**：利用 prefill 阶段的专家激活模式构建 **expert signature**，预测 decode 阶段的专家使用情况。
- **离线聚类 + 在线局部带路由**：
  - **Offline**：使用 balanced K-means 将 signature space 划分为 K 个区域（每个 decode worker 一个）。
  - **Online**：采用 **locality-band routing**，将请求路由到与其 signature 最相似且负载最低的 worker。
- **前缀缓存一致性签名机制**：设计 **block-granular signature cache**，与 KV cache 同步管理，确保在 prefix caching 场景下 signature 完整、准确。

### 相比现有方法的优势
- **更优的 TPOT 性能**：相比纯负载均衡方法（如 RR、JSQ、P2C），显著降低 median 和 tail TPOT。
- **无损输出**：不改变模型结构、gate 决策或 kernel，保持与标准 top-k gating 输出一致。
- **兼容性强**：与 prefix caching、expert parallelism 等现代优化技术正交，可无缝集成。
- **开销极低**：<1% HBM 占用，增加 <1.2% TTFT。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Task Workload**：来自多个领域的混合提示，包括：
  - Code（代码）
  - Math（数学）
  - Medical（医学）
  - Legal（法律）
- **Language Workload**：基于 **WildChat [37]** 的多语言真实用户交互数据，涵盖英语、中文、俄语、法语等，具有明显的流量倾斜（如中英文占 ~75%）。

### 实验设置
- **模型**：
  - Qwen3-30B-A3B（128 experts, top-8）
  - GPT-OSS-120B（128 experts, top-4）
  - Gemma-4-26B-A4B（128 experts, top-8）
  - 扩展至 Qwen3-235B-A22B（TP=4, EP=4）
- **硬件平台**：5 节点集群，每节点 8×AMD MI300X GPU（共 40 GPUs），NDR InfiniBand 互联。
- **部署架构**：`xPyD` 架构（如 8P16D），prefill 与 decode worker 分离。
- **实现框架**：基于 **vLLM** 实现，集成 prefill-time signature capture、offline clustering、routing proxy。

### 评估指标
- **TPOT (Time-per-Output-Token)**：核心指标，衡量 decode 效率。
- **TTFT (Time-to-First-Token)**：用于验证对 prefill 的影响。
- **Active Expert Count**：验证 ELDR 是否真正减少了每步激活的专家数。
- **内存开销**：signature cache 大小。

### 基线方法对比
- **Load-Balancing Baselines**：
  - Random
  - Round-Robin (RR)
  - Join-Shortest-Queue (JSQ)
  - Power-of-Two-Choices (P2C)
- **Locality-Aware Baseline**：
  - **Domain**：基于人工 domain label 的静态分区路由（理想化基线）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 工作负载 | **Median TPOT 降低** | Tail TPOT 降低 |
|------|--------|------------------|--------------|
| Qwen3-30B-A3B | Task | **13.9%** | 6.0% |
| GPT-OSS-120B | Task | **7.0%** | 3.4% |
| Gemma-4-26B-A4B | Task | **13.9%** | 6.0% |
| 平均 | Language | **5.9–10.0%** | ±1.5% |

> ✅ **所有场景下，ELDR 均优于最强的 load-balancing baseline**

### 与基线方法的对比结果
- **vs Load-Balancing (RR/JSQ/P2C)**：
  - 显著降低 median TPOT（5.9–13.9%）。
  - 在 task workload 上优势更大（domain 结构清晰）。
- **vs Domain（静态标签路由）**：
  - 在 task 上仍胜出 **1.4–6.9%**（因 signature 更细粒度）。
  - 在 language 上大幅领先 **5.7–9.1%**（因语言内存在子结构，标签太粗）。
  - Domain 在尾部延迟上表现差（因负载不均）。

### 消融实验结果
#### （1）Expert Signature 设计
- **count·idf > gate-prob/logit**：离散计数 + IDF 加权比连续 gate score 更优，平均再降 **3–14 pp TPOT**。
- **Layer Mask 有效压缩**：通过 greedy selection 选择最具判别性的层，提升 signature 质量。

#### （2）Clustering 方法
- **Hungarian-balanced K-means > Vanilla K-means**：
  - Vanilla K-means 可能导致负载不均，尾部延迟恶化（↑17.4%）。
  - Balanced 版本兼顾 locality 与 load，P50↓12.6%，P99↓6.8%。

#### （3）Locality Band Width (τ)
- **τ = 0.1 为最优**：
  - τ = 0（pure top-1）导致尾部延迟上升。
  - τ = 0.1 在保持 locality 的同时引入 load awareness，P50↓5.2–12.7%。

#### （4）Prefix Cache 兼容性
- 开启 prefix caching 后，ELDR 的 TPOT 优势依然保持（~13% P50 优势）。
- TTFT 因 decode 快速释放 prefill 资源而进一步降低。

#### （5）扩展性测试
- **Decoder Pool Size 增加**（8P8D → 8P24D）：
  - TPOT 改善随 decoder 数增加而单调提升（-8.0% → -10.2%）。
- **Large MoE + Expert Parallelism**（Qwen3-235B-A22B）：
  - 在 40 GPU 部署下，median TPOT 降低 **2.7–4.3%**，tail 降低 0.6–2.0%。

---

## 4. 关键结论和发现

### 主要发现
1. **Expert Locality 是真实且可预测的**：
   - Prefill 与 decode 阶段的专家激活高度相关（correlation 0.70–0.92）。
   - 同 domain 请求共享更多专家（same-domain batch 激活专家数减少 17–21%）。
2. **仅负载均衡不足以优化 MoE decode 性能**：
   - 必须考虑 **active expert union** 这一第一性原理。
3. **ELDR 实现了 locality 与 load 的动态平衡**：
   - 离线聚类捕捉结构，在线 band routing 吸收运行时偏斜。
4. **方法轻量且通用**：
   - 开销 <1% HBM，TTFT 影响可忽略。
   - 支持 prefix caching、expert parallelism、scale-out 部署。

### 方法的局限性
- **依赖 prefill 信号质量**：若 prefill 与 decode 专家分布差异大（如 prompt 不具代表性），效果可能下降。
- **calibration 数据需代表 workload**：若 workload 动态变化剧烈，需重新 offline fit。
- **未处理专家内部负载不均**：虽与 EPLB 正交，但极端情况下仍需结合 intra-worker 负载均衡。

### 未来工作方向
- **动态 re-clustering**：支持 workload drift 时自动更新 centroids。
- **跨实例协同 routing**：在更大规模集群中协调多个 ELDR 实例。
- **结合 expert prefetching**：利用 signature 预加载可能使用的专家权重。
- **扩展至其他稀疏架构**：如 block-sparse 或 conditional compute 模型。

---

> **总结**：ELDR 首次将 **expert locality** 引入 PD-disaggregated MoE 服务的 decode 路由决策，提出了一套高效、无损、低开销的解决方案，在多种模型、工作负载和拓扑下均显著降低了 TPOT，为 MoE 模型的高效 serving 提供了新的范式。

</details>

---

### 10. [TallyTrain: Communication-Efficient Federated Distillation](https://arxiv.org/abs/2607.00173)

**Authors**: Radhakrishna Achanta, Will Reed  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.00173v1  

#### Abstract
Federated learning is bandwidth-bound on two orthogonal axes: model size, which limits how often parameter-averaging methods can afford to merge, and class count, which makes per-probe soft-label distillation prohibitive at large vocabularies. Both ceilings tighten as modern systems scale. We collap...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TallyTrain: Communication-Efficient Federated Distillation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代联邦学习（Federated Learning, FL）面临两大带宽瓶颈：
- **模型大小轴**（model size axis）：参数平均类方法（如 FedAvg）通信开销为 $O(|W|)$，在大模型场景下不可行。
- **类别数量轴**（class count axis）：函数空间方法（如 FedMD、FedDF）依赖软标签（soft labels）进行知识蒸馏，每探针通信开销为 $O(C)$，在大词汇任务（如语言建模，$C \sim 50,000$）中变得 prohibitively 昂贵。

随着模型规模和类别数增长，这两个瓶颈同时收紧，限制了 FL 在边缘设备上的应用。

---

### **提出的新方法与新思路**
作者提出 **TallyTrain** —— 一种通信高效的联邦蒸馏协议，其核心创新在于：

- **硬标签共识机制（hard-label consensus）**：  
  每个客户端仅广播其对公共探针集（public probe set）上每个样本的 `argmax` 预测类别索引（即 top-1 类别），而非完整的 $C$ 维软标签向量。
  - 每个探针仅需传输 $\lceil \log_2 C \rceil$ bits（例如 $C \leq 256$ 时为 1 字节，$C \leq 65,536$ 时为 2 字节）。
  - 所有客户端将收到的 `argmax` 投票汇总成一个投票直方图 $H(x)$，作为全局共识目标用于知识蒸馏。

- **两种操作模式**：
  1. **纯函数空间模式（Pure Function-Space Mode, M=0）**：完全基于硬标签蒸馏，无参数交换。
  2. **带宽桥接变体（Bandwidth-Bridge Variant, TallyTrain+faM）**：将稀疏的 FedAvg 参数合并（每 $M$ 轮一次）与高频的硬标签蒸馏通道结合，实现参数空间与函数空间的协同优化。

- **理论支持**：
  - 函数空间收缩引理（Function-space contraction）
  - Condorcet 投票准确性边界
  - 硬标签梯度方差降低定理

---

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **通信效率** | 每探针通信量从 $4C$ 字节（32位软标签）降至 1–2 字节，压缩比达 $40\times$（CIFAR-10）至 $4096\times$（WikiText-2, $C=2048$）。 |
| **抗噪声能力** | 多数投票可过滤“自信错误”（confidently wrong）的欠训练客户端噪声；而软标签平均会放大此类噪声。 |
| **非IID鲁棒性** | 在非独立同分布（non-IID）数据下表现优于 FedAvg 和 FedDF，尤其在类别不平衡或高熵预测场景。 |
| **跨架构兼容性** | 不依赖模型结构一致性，适用于异构客户端。 |
| **隐私增强** | 每探针信息泄露上限从 $O(C)$ 降至 $O(\log C)$，提升对成员推断攻击的防御力。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **图像分类任务**：
  - **CIFAR-10**（$C=10$）
  - **CIFAR-100**（$C=100$）
  - 非IID 划分方式：Dirichlet 分布 ($\alpha=0.5$)，$N=10$ 客户端
- **语言建模任务**：
  - **WikiText-2**（BPE 词汇表 $V=2048$）
  - 下一个词预测任务，$N=8$ 客户端

### **模型架构**
- 图像任务：ResNet-18
- 语言模型：TinyGPT（6层 Transformer，约 5.84M 参数）

### **评估指标**
- 主要指标：尾部准确率（Tail accuracy，最后若干轮的平均测试准确率）
- 辅助指标：峰值准确率（Peak accuracy）、困惑度（Perplexity）、跨客户端标准差、总通信负载（MB）

### **实验设置**
- 本地训练：每轮 $K=5$ 步 SGD，AdamW 优化器
- 公共探针集大小：
  - CIFAR-10: 2,000
  - CIFAR-100: 10,000
  - WikiText-2: 5% 训练数据作为探针
- 蒸馏采样大小 $B_{\text{sample}} = 32$
- 温暖期（warm-up）：前 $W=300$ 轮不启用共识通道

### **基线方法对比**
| 方法 | 类型 | 特点 |
|------|------|------|
| **FedAvg** | 参数空间 | 标准联邦平均 |
| **FedProx** | 参数空间 | 改进版 FedAvg，加入 proximal term |
| **FedMD** | 函数空间 | 使用软标签蒸馏 |
| **FedDF** | 函数空间 | 服务器端集成蒸馏 |
| **Local-only** | 无通信 | 各自独立训练 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **CIFAR-100 实验结果（Table 1）**
| 方法 | 尾部准确率 (%) | 通信成本（每探针字节） | 压缩比 |
|------|----------------|------------------------|-------|
| **TallyTrain** ($\alpha=0.5$) | **33.16 ± 0.27** | **1** | **400×** |
| FedMD ($\alpha=0.5$) | 31.81 ± 0.10 | 400 | 1× |
| 无共识（$\alpha=1.0$） | 32.03 ± 0.06 | 0 | — |

> ✅ **TallyTrain 比 FedMD 高出 1.35 个百分点，且通信量减少 400 倍！**  
> ❗ 软标签共识甚至略微损害性能（-0.22pp），说明其平均化了噪声。

---

#### **CIFAR-10 实验结果（Table 2）**
| 方法 | 尾部准确率 (%) | 总通信量 (MB) | 跨客户端 std |
|------|----------------|---------------|--------------|
| **TallyTrain+fa200** | **71.92 ± 0.38** | 1,253 | **1.00** |
| FedMD+fa200 | 72.68 ± 0.69 | 1,268 | 0.93 |
| FedAvg-fa500（McMahan 默认） | 50.69 ± 1.85 | 537 | 6.50 |
| FedAvg-fa50 | 60.51 ± 1.52 | 5,367 | 5.56 |

> ✅ **TallyTrain+fa200 比 McMahan 默认配置高出 +21.2pp，且通信量更低。**  
> ✅ 相比 FedAvg-fa200，在相同通信预算下高出 +17.3pp。  
> ✅ 种子间方差最小（0.38pp），训练最稳定。

---

#### **WikiText-2 语言建模实验（Table 3）**
| 方法 | 尾部准确率 (%) | 总通信量 |
|------|----------------|----------|
| **TallyTrain+fa200** | **22.97 ± 0.03** | 2.60 GB |
| FedAvg-fa200 | 21.52 ± 0.03 | 2.45 GB |
| **TallyTrain (pure)** | 15.45 ± 0.03 | **0.31 GB** |
| FedMD (pure) | 15.78 ± 0.02 | **1,268 GB** |

> ✅ **纯 TallyTrain 以 4096× 更低带宽达到与 FedMD 相当的准确率。**  
> ✅ **TallyTrain+fa200 比 FedAvg-fa200 高出 +1.45pp，仅增加 6% 带宽。**

---

### **消融实验结果**

#### **(1) 排名信息增益有限（Appendix E）**
在 CIFAR-10 上比较不同 top-$k$ 传输策略：
| top-$k$ | 尾部准确率 (%) | 每探针字节数 |
|--------|----------------|-------------|
| 1（TallyTrain） | 64.82 | 1 |
| 3 | 65.20 | 15 |
| 5 | 65.29 | 25 |

> 🔍 保留更多排名信息带来的增益不足 0.5pp，远低于通信代价。

#### **(2) 子集采样大小 $B_{\text{sample}}$ 影响（Appendix F）**
最优 per-class coverage rate $p = B_{\text{sample}} / C \in [1, 2]$。过小导致信号稀疏，过大则过度拟合公共集。

#### **(3) 更复杂聚合机制无效（Appendix D）**
尝试 disagreement-weighted 或 asymmetric weighting，均未超越 uniform argmax voting，表明后者已接近函数空间上限。

#### **(4) KL衰减防止跨域崩溃（Appendix G）**
在 CIFAR-10 任务使用 CIFAR-100 探针时，若不加 KL-decay，模型会陷入自强化错误共识并崩溃；加入线性衰减后可稳定训练。

---

## **4. 关键结论和发现**

### **主要发现**
1. **硬标签蒸馏几乎携带全部有效信号**：  
   `argmax` 投票足以构建高质量共识，软标签中的概率分布信息在多数情况下冗余。

2. **多数投票是天然的噪声滤波器**：  
   当客户端“自信地错”时，软标签平均会拉低整体预测质量，而硬标签投票可通过多数原则纠正个体偏差。

3. **TallyTrain+faM 实现帕累托主导（Pareto-dominant）**：  
   结合稀疏参数合并与高频硬标签同步，既保持高准确率又大幅降低漂移，优于所有 FedAvg/FedProx/FedDF 配置。

4. **通信优势随 $C$ 线性增长**：  
   压缩比 $\propto C$，使得 TallyTrain 尤其适合大词汇任务（如 LLMs）。

5. **纯蒸馏无法替代参数合并**：  
   即使极大扩充公共探针集（如 100k OOD probes + anchor），纯 TallyTrain 仍无法逼近参数平均的精度上限（gap >14pp），证明“带宽桥”不是补丁而是必要补充。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **Full-mesh 拓扑限制** | 当前分析基于全连接拓扑，$O(N^2)$ 通信复杂度；尚未验证在 sparse-gossip 中的表现。 |
| **不适用于极小 $N$ 场景** | Condorcet 效应要求 $N$ 足够大（经验上 $N \gtrsim 20$）才能形成可靠共识。 |
| **KL-decay 需手动调参** | 虽然窗口 $T$ 对结果鲁棒，但仍需预设。理想情况应自动检测共识稳定性。 |
| **未覆盖超大规模 LMs** | 实验基于 tiny-GPT，扩展到 billion-parameter 模型仍需验证。 |

---

### **未来工作方向**
1. **稀疏八卦拓扑（Sparse Gossip）扩展**：设计去中心化的 gossip-based TallyTrain，支持 $N \gg 10$ 规模。
2. **自适应共识权重机制**：基于投票 margin 动态调整 KL 权重，统一 labeled/unlabeled/LM 场景。
3. **异构架构下的理论分析**：研究不同模型容量/结构下函数空间收缩行为。
4. **生产级大模型验证**：在 $V=32K–50K$、billion-parameter LM 上验证 TallyTrain+faM 的有效性。
5. **形式化隐私分析**：量化硬标签通道在 membership inference 和 model inversion 中的实际防护能力。

---

> 📌 **一句话总结**：  
> **TallyTrain 通过仅传输 `argmax` 类别索引，实现了高达三个数量级的通信压缩，且在 non-IID 场景下因“多数投票抗噪”特性反而优于软标签蒸馏；结合稀疏参数合并后，其带宽-精度权衡全面超越现有联邦学习范式。**

</details>

---

### 11. [Ghost in the Kernel: In-Context Learning with Efficient Transformers via Domain Generalization](https://arxiv.org/abs/2607.00479)

**Authors**: Peilin Liu, Ding-Xuan Zhou  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.00479v1  

#### Abstract
Transformer-based large models have demonstrated remarkable generalization abilities across different tasks by leveraging a context-aware attention module for in-context learning. With richer context, transformers adapt more effectively to the current use case without any parameter updates. However,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Ghost in the Kernel: In-Context Learning with Efficient Transformers via Domain Generalization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文聚焦于 **Transformer** 模型在 **In-Context Learning (ICL)** 场景下的效率与泛化能力问题。标准的 **softmax attention** 虽然具有强大的上下文建模能力，但其计算和内存复杂度为 $ O(n^2) $，严重限制了长序列建模能力。尽管已有如 **linear attention** 等高效替代方案，但其理论基础（尤其是在 ICL 场景中的泛化机制）尚不清晰。

### **提出了什么新方法或新思路**
论文提出了一种全新的理论框架，将 **In-Context Learning** 视为一种 **Domain Generalization** 任务，并在此基础上构建了一个基于 **two-staged sampling process** 的分析模型。主要创新点如下：

- **理论连接**：首次将 **In-Context Learning** 与 **Domain Generalization** 和 **Operator Learning** 理论联系起来，揭示了 ICL 的本质是学习从上下文分布到响应函数的映射。
- **线性注意力的泛化分析**：在该框架下，对 **Linear Transformers** 进行了严格的逼近与泛化能力分析，证明其能以 **维度无关（dimension-independent）的收敛速率** 实现稳健泛化。
- **Fast Eigendecay 现象的利用**：观察并证明了在预训练大模型（如 Qwen3）的 attention 权重中存在快速特征值衰减（fast eigendecay），这一现象有助于 linear attention 更好地模拟 softmax attention 并缓解分布偏移的影响。
- **新的线性化视角**：提出了一种基于理论分析的 **pretrained softmax LLMs 的线性化转换（linear conversion）** 新视角，指导如何设计激活函数和损失函数来保留原始模型的知识。

### **相比现有方法的优势**
- **理论深度**：提供了比以往更深刻的理论解释，阐明了 linear attention 在 ICL 中为何有效，而不仅仅是经验上的性能匹配。
- **普适性**：所得的泛化界不依赖于输入维度，更具可扩展性。
- **指导性**：不仅解释现象，还为实际的模型设计（如激活函数、归一化策略）提供了明确的理论指导。

---

## **2. 核心实验方法和设置**

### **使用了哪些数据集**
本论文是一篇**理论研究**，**并未进行传统意义上的实验**，因此没有使用具体的数据集（如 GLUE、SQuAD 等）。其“实验”部分主要是**数值验证和理论推导**。

- **数值验证**：使用了开源大语言模型 **Qwen3-8B** 的 attention 权重矩阵，对其奇异值（singular values）进行了统计分析，以验证提出的 **fast eigendecay** 现象。

### **实验设置和评估指标**
- **设置**：提取 Qwen3-8B 模型各层 attention 头的 $ W_q $ 和 $ W_k $ 矩阵，计算 $ W_q W_k^\top $ 的奇异值分解（SVD），并绘制奇异值随索引变化的趋势图。
- **评估指标**：主要通过可视化（折线图）展示奇异值的衰减速度，特别是是否呈现近似指数级的快速衰减。

### **基线方法对比**
由于是理论工作，没有直接的“基线方法”对比。但其理论框架和结论与以下方法形成对比：
- **Softmax Attention**：作为被替代的对象，其高复杂度是本文要解决的问题。
- **传统 Linear Attention**（如 Performer, Linformer）：本文指出这些方法的设计缺乏对 ICL 本质的深刻理解，而本文的框架为其有效性提供了理论支撑。
- **其他高效架构**（如 Mamba, RetNet）：虽然未直接比较，但本文专注于从 attention 机制本身出发的优化路径。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **Fast Eigendecay 验证**：如图1所示，在 Qwen3-8B 模型中，attention 权重矩阵的奇异值表现出**极其快速的衰减**，在前几十个维度后迅速趋近于零。这表明模型的注意力行为可以由一个低秩的子空间很好地近似。
- **理论泛化界**：论文推导出的关键泛化误差上界为：
  $$
  \mathbb{E}\left\{\mathcal{E}(S_M(T_{s,n})) - \mathcal{E}(f_c)\right\} \leq K_3 N^{-\frac{2+\gamma}{(\gamma-1)\delta}} (\log N)^3
  $$
  其中 $ N $ 是元样本数量，$ \gamma > 1 $ 控制分布偏移的程度。这个界是**维度无关**的，且揭示了数据分布正则性与泛化速率之间的权衡。

### **与基线方法的对比结果**
- **理论优势**：相比仅凭经验设计的 linear attention，本文的方法有坚实的理论基础，能够解释为何某些设计（如去除归一化因子、使用 RMSNorm）更稳定。
- **性能潜力**：通过利用 fast eigendecay，linear transformer 可以高效地模仿 softmax attention 的行为，从而在保持高性能的同时大幅降低计算成本。

### **消融实验结果**
论文**没有进行传统的消融实验**。但其理论分析本身就包含了对不同因素的“消融”：
- **假设分析**：通过放松对 likelihood ratio 的有界性要求（允许无界），展示了框架的鲁棒性。
- **组件作用**：分析了截断算子（truncation operator）、RMSNorm 等设计在保证算法稳定性中的作用。

---

## **4. 关键结论和发现**

### **论文的主要发现**
1. **In-Context Learning 即 Domain Generalization**：ICL 的本质是模型学习如何根据不同的上下文分布 $ p $，泛化到该分布下的预测任务，这是一个典型的领域泛化问题。
2. **Linear Attention 的有效性源于低秩结构**：预训练模型中存在的 **fast eigendecay** 现象使得用线性注意力近似 softmax 注意力成为可能，且能有效应对分布偏移。
3. **维度无关的泛化能力**：在提出的理论框架下，linear transformer 能够实现维度无关的泛化，这对于处理高维现实世界数据至关重要。
4. **新的线性化设计范式**：为将预训练的 softmax LLM 转换为线性版本提供了新思路——应关注如何捕捉和保留原始模型中的几何信息（如 $ \Sigma_x $）和分布结构。

### **方法的局限性**
- **纯理论性质**：所有结论均基于数学推导和数值验证，缺乏在真实下游任务上的端到端性能测试。
- **理想化假设**：理论分析依赖于一些理想化假设，如特定的核函数形式、概率测度的正则性条件等，这些在真实场景中可能不完全成立。
- **实现挑战**：虽然提出了线性化的思想，但具体的、高效的转换算法仍需进一步探索。

### **未来工作方向**
- **实证研究**：将理论框架转化为具体的 linear conversion 算法，并在多种长文本任务上进行大规模实证评估。
- **混合架构**：探索结合 linear attention 和 softmax attention 的混合模型，以兼顾效率与表达能力。
- **扩展到其他模型**：将该理论框架应用于分析 Mamba、RetNet 等其他高效序列模型的 ICL 能力。
- **优化理论**：从 Operator Learning 的角度，深入研究 linear transformer 的优化动态和收敛性。

---

</details>

---

### 12. [Beyond Activation Alignment:The Alignment-Diversity Tradeoff in Task-Aware LLM Quantization](https://arxiv.org/abs/2607.00908)

**Authors**: Fei Wang, Chao Xue, Taoran Liu, Li Shen, Ye Liu, ChangXing Ding  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.00908v1  

#### Abstract
Mixed-precision quantization (MPQ) has become a key technique for deploying large language models under stringent memory and compute constraints. We first identify a phenomenon that we term the Perplexity Illusion: layers ranked as important by perplexity-based sensitivity show little rank correlati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Beyond Activation Alignment: The Alignment-Diversity Tradeoff in Task-Aware LLM Quantization》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

该论文针对**大语言模型（LLM）在任务感知量化（Task-Aware Quantization）中的两个关键盲点**提出挑战：

1. **Perplexity Illusion（困惑度幻觉）**：传统混合精度量化（MPQ）依赖基于 **perplexity (PPL)** 的敏感性分析来分配比特位宽，但作者发现，PPL 敏感层与对复杂推理任务（如数学、科学问答）真正关键的层几乎没有相关性（Kendall τ ≈ 0）。这意味着传统方法“保护了错误的参数”。

2. **Alignment-Diversity Tradeoff（对齐-多样性权衡）**：虽然使用目标任务数据进行校准可以提高激活对齐（activation alignment），但**仅使用任务特定数据会损害泛化能力**，导致“表示坍缩”（representation collapse）。最优校准策略应在通用域数据（如 WikiText）和任务特定数据（如 GSM8K）之间找到平衡。

---

### 提出的新方法或新思路

作者提出了 **TASA (Task-Aware Sensitivity Analysis)**，一个两阶段联合优化框架，同时解决校准数据构成和比特分配问题：

#### （1）训练免费的梯度迹对齐自动校准（Gradient Trace Alignment Auto-Calibration）
- **目标**：自动搜索最优的校准数据混合比例（`α_wiki`）。
- **方法**：计算不同数据分布下各层激活能量的**迹向量（trace vector）** `h(D)`，并最大化其与目标任务迹向量的**余弦相似度**。
- **优势**：无需实际量化或反向传播，仅需 `(|A|+1)` 次前向传递即可完成搜索，成本极低。

#### （2）多目标聚合（MOA）敏感性分析与混合精度分配
- **敏感性度量**：引入 **Multi-Objective Aggregation (MOA)** 度量，融合多个下游任务的敏感性信号：
  $$
  S_{\text{MOA}}(l,b) = \beta \cdot \Delta_{\text{math}}(b) + (1-\beta) \cdot \Delta_{\text{PPL}}(b)
  $$
  其中 `β=0.7` 默认偏向任务性能。
- **比特分配**：
  - **层间分配**：将 MOA 敏感性作为目标，通过**动态规划求解整数线性规划（ILP）**，在全局比特预算下实现最优层间比特分配。
  - **层内分配**：在每层内部，基于 **OBS salience** 进行组级（group-wise）混合精度分配，进一步提升压缩效率。

---

### 相比现有方法的优势

| 维度 | 传统方法 | TASA |
|------|--------|------|
| **敏感性度量** | 仅用 PPL 或 Hessian trace | 融合 PPL 和任务特定（如数学）信号 |
| **校准数据** | 固定使用通用数据（如 WikiText） | 自动搜索最优混合比例（通用 + 任务数据） |
| **分配粒度** | 层级或列级 | **层级 + 组级** 双重混合精度 |
| **任务适应性** | 无 | 显式支持任务感知量化 |
| **计算开销** | 通常较低 | 一次性离线开销（~47分钟），可跨比特预算复用 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **通用校准数据**：`WikiText-2`（128样本）
- **任务特定校准数据**：`GSM8K`（数学推理）、`ARC-Challenge`（科学问答）
- **评估基准**（共8个任务）：
  - **推理**：GSM8K（8-shot CoT）、ARC-Challenge（25-shot）、ARC-Easy
  - **常识**：HellaSwag（10-shot）、WinoGrande（5-shot）、PIQA
  - **阅读理解**：BoolQ
  - **语言建模**：WikiText-2 PPL
- **主评估指标**：`Avg.` —— 除 PPL 外7个准确率任务的**未加权平均值**

---

### 实验设置

- **模型**：
  - `LLaMA-3-8B`（32层）
  - `Qwen2.5-7B`（28层）
- **量化后端**：基于 `GPTQ`，组大小 `g=128`
- **TASA 参数**：
  - `β = 0.7`（MOA 权重）
  - 混合校准：`α_wiki = 0.5`（LLaMA-3），`α_wiki = 0.75`（Qwen2.5）
- **比特预算**：从 `b3.0` 到 `b4.0`

---

### 基线方法对比

| 类型 | 方法 |
|------|------|
| **均匀精度** | RTN, GPTQ, AWQ, HQQ, OWQ |
| **混合精度** | SpQR（含元数据开销） |
| **任务感知** | TACQ（需反向传播） |
| **其他** | SliM-LLM（作为混合精度基线） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在 `LLaMA-3-8B` 上的结果（`b3.5` 平均精度）

| 方法 | Avg. | GSM8K | PPL ↓ |
|------|------|-------|-------|
| **TASA (ours)** | **68.9** | **46.2** | 8.72 |
| HQQ W4 | 68.1 | 39.4 | 8.09 |
| RTN W4 | 67.9 | 36.2 | 8.37 |
| AWQ W4 | 69.4 | 43.4 | 7.90 |
| **FP16** | **70.9** | **49.8** | **7.25** |

- **结论**：TASA 在 **仅 3.5-bit** 下，**匹配甚至超越多个 4-bit 均匀基线**，且在 GSM8K 上领先显著（+6.8 vs HQQ）。

#### 在 `Qwen2.5-7B` 上的结果（`b3.75`）

| 方法 | Avg. | GSM8K | PPL ↓ |
|------|------|-------|-------|
| **TASA (ours)** | **75.0** | 78.4 | 9.38 |
| HQQ W4 | 74.7 | 78.8 | 9.23 |
| **FP16** | **75.7** | **83.1** | **8.74** |

- **结论**：TASA 在 `b3.75` 下保留了 **99.1% 的 FP16 性能**，优于所有 4-bit 均匀方法。

---

### 与基线方法的对比结果

- **精度反转（Precision Inversion）**：TASA 在更低平均比特（3.5-bit）下，实现了与 4-bit 均匀量化相当甚至更优的综合性能。
- **推理任务大幅提升**：
  - 在 `LLaMA-3-8B` 上，TASA `b3.5` 比最强的 `W3` 基线（OWQ）在 GSM8K 上高出 **+20.3** 个百分点。
  - 在 `b3.0` 时，TASA 避免了 `HQQ/RTN` 几乎为零的推理崩溃（GSM8K=0.0）。
- **SpQR 对比**：尽管 SpQR 名义上是 3-bit，但因元数据和异常值存储，其有效比特高达 **4.12-bit**。TASA 在真实 3.5-bit 下仍优于 SpQR。

---

### 消融实验结果

#### （1）分配策略消融（Table 2）

| 策略 | Avg. | GSM8K |
|------|------|-------|
| **MOA + mixed** | **68.9** | **46.2** |
| MOA + wiki | 66.8 | 33.7 |
| PPL-topK + wiki | 63.3 | 23.1 |
| Random + wiki | 65.4 | 30.6 |

- **结论**：
  - **MOA 分配策略贡献最大**（+5.6 vs PPL-only）。
  - **混合校准提供额外增益**（+2.1 vs MOA-only）。
  - 随机分配在推理任务上表现极差。

#### （2）校准数据混合比例消融（Table 3）

| `α_wiki` | LLaMA-3 Avg. | Qwen2.5 Avg. |
|----------|--------------|--------------|
| 0.00（纯任务） | 68.3 | 74.1 |
| **0.50（平衡）** | **68.9** | 74.1 |
| **0.75（偏通用）** | 68.6 | **74.3** |
| 1.00（纯通用） | 66.8 | 72.5 |

- **结论**：性能峰值出现在中间混合比例，验证了 **Alignment-Diversity Tradeoff** 的存在。

---

## 4. 关键结论和发现

### 主要发现

1. **Perplexity Illusion 是普遍现象**：PPL 敏感性与推理敏感性几乎无关（τ ≈ 0），传统方法无法识别推理关键层。
2. **存在 Alignment-Diversity Tradeoff**：纯任务校准虽提高对齐度，但损害泛化；最优策略是混合通用与任务数据。
3. **TASA 实现精度反转**：在 3.5-bit 下，TASA 匹配 4-bit 均匀量化性能，并在推理任务上大幅领先。
4. **校准数据构成至关重要**：这一因素在以往工作中被严重低估。

---

### 方法的局限性

- **扩展性**：MOA 敏感性分析的计算开销随层数线性增长，扩展到更大模型（如 70B）需要分布式计算。
- **搜索空间**：当前自动校准在离散网格上搜索，连续松弛可能找到更优解。
- **硬件支持**：异构比特模式尚未被所有推理引擎原生支持（尽管 `MLC-LLM` 和 `vLLM` 已开始支持）。

---

### 未来工作方向

- 将 TASA 扩展到 **MoE 架构** 和 **超大规模模型**。
- 探索 **连续优化** 方法以替代离散网格搜索。
- 设计 **硬件友好的异构比特调度器**，以充分发挥 TASA 的潜力。
- 将框架推广至 **多任务联合优化** 场景。

--- 

> **总结**：TASA 通过揭示 **Perplexity Illusion** 和 **Alignment-Diversity Tradeoff**，从根本上重新思考了 LLM 量化中的校准与分配问题。其实验证明，**合理的比特分配比单纯增加比特更重要**，为高效部署具备强推理能力的 LLM 提供了新范式。

</details>

---

### 13. [Balancing Expressivity and Learnability in Quantum Kernel Bandit Optimization](https://arxiv.org/abs/2607.01080)

**Authors**: Yuqi Huang, Vincent Y. F. Tan, Sharu Theresa Jose  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.01080v1  

#### Abstract
We investigate Gaussian process (GP) bandit optimization with quantum kernels, assuming the mean reward function lies in the reproducing kernel Hilbert space (RKHS) induced by the quantum kernel. This setting is motivated by NISQ-era tasks such as quantum control, state preparation and variational q...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Balancing Expressivity and Learnability in Quantum Kernel Bandit Optimization 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文研究了在 **NISQ-era**（Noisy Intermediate-Scale Quantum）背景下，基于 **量子核（quantum kernel）的高维性与带噪声观测** 所导致的 **学习困难（learnability）问题**。

具体而言，虽然量子核（如 fidelity quantum kernel）因其对物理系统的归纳偏置（inductive bias）而具备强大的表达能力（expressivity），但其特征空间维度随 qubit 数量 $ n $ 呈指数增长（$ \sim 4^n $）。这带来了两个严重挑战：
- **过高的信息增益（information gain）**：导致 GP bandit 算法（如 GP-UCB）的累积遗憾（cumulative regret）过高。
- **核值集中现象（kernel concentration）**：随着 $ n $ 增大，不同输入对应的核值趋于相同，需要指数级测量次数才能准确估计。

因此，直接使用完整量子核会导致 **样本效率低下** 和 **计算不可扩展**。

---

### 提出的新方法与新思路
为解决上述问题，作者提出了一套 **平衡表达力与可学习性的框架**，核心思想是：**使用低维近似核（approximate kernel）来降低模型复杂度，以换取可控的核误设（kernel misspecification）误差**。

具体提出了三种近似核构造方法：

#### （1）**线性投影量子核（Linear Projected Quantum Kernels, LPQK）**
- 将完整的 $ n $-qubit 密度矩阵 $ \rho(x) $ 投影到子系统 $ s \subseteq \{1,\dots,n\} $ 上，得到约化密度矩阵 $ \rho^s(x) = \mathrm{Tr}_{s^c}[\rho(x)] $。
- 定义投影核：$ K_s(x,x') = \mathrm{Tr}(\rho^s(x)\rho^s(x')) $。
- 可进一步求和多个子系统上的投影核（summed LPQK）以保留更多信息。

#### （2）**经典随机傅里叶特征（Random Fourier Features, RFF）**
- 利用量子核的 **Fourier 表示**（适用于 shift-invariant 量子核）。
- 通过采样频率 $ \omega \sim p(\omega) $ 构造低维特征映射 $ \phi_{\text{RFF}}(x) $，从而近似原核：  
  $ K_{\text{RFF}}(x,x') = \phi_{\text{RFF}}(x)^T \phi_{\text{RFF}}(x') \approx K_Q(x,x') $。

#### （3）**P-greedy 核逼近**
- 一种数据驱动的方法，无需显式特征映射。
- 迭代选择使“功率函数”（power function）最大的点，构建由核函数张成的低维子空间 $ V(X_D) = \mathrm{span}\{K_Q(\cdot, s_i)\}_{i=1}^D $。
- 最终形成 **Newton basis**，实现低秩核逼近。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **理论分析** | 首次将 **misspecified kernel bandit** 理论应用于量子核场景，推导出包含 **信息增益 $ \gamma_T $** 与 **误设误差 $ \epsilon $** 的 **regret bound**：$ R_T = \mathcal{O}(\sqrt{\gamma_T T} + \epsilon \sqrt{T \gamma_T}) $，为模型选择提供原则性指导。 |
| **可扩展性** | 时间复杂度从 $ \mathcal{O}(T^3) $（标准 GP）降至 $ \mathcal{O}(TD^2 + D^3) $（$ D \ll T $），显著提升计算效率。 |
| **灵活性** | 提供了 **量子硬件依赖**（LPQK）与 **纯经典**（RFF, P-greedy）两种路径，适应不同部署需求。 |
| **性能** | 实验证明，在多数任务中，**适当降维的近似核优于完整量子核**，实现了更低的 regret 和更快的收敛。 |

---

## 2. 核心实验方法和设置

### 数据集
所有实验均基于 **合成或真实量子任务**，不涉及传统机器学习数据集：

1. **Synthetic Quantum Functions**  
   - 从一个 3-qubit 或 6-qubit 的量子核诱导的 GP 先验中采样真值函数 $ f^* $。
   - 输入域：$ [0, 2\pi]^3 $ 或 $ [0, 2\pi]^6 $。
   - 观测奖励：$ y_t = f^*(x_t) + \eta_t $，其中 $ \eta_t \sim \mathcal{N}(0, 0.01) $。

2. **Phase Classification**  
   - 任务：识别广义簇哈密顿量（generalized cluster Hamiltonian）中的铁磁相（ferromagnetic phase II）。
   - 参数空间：$ (J_1, J_2) \in [-4, 4]^2 $，离散为 $ 20 \times 20 = 400 $ 个动作。
   - 奖励函数：基于归一化总磁化率平方，理想输出为 1（相 II）或 0（非相 II）。

3. **Variational Quantum Eigensolver (VQE)**  
   - 任务：优化 3-qubit “Efficient SU(2)” 电路参数，以最小化 XYZ Heisenberg Hamiltonian 的能量。
   - 使用 **Bayesian Optimization**（而非 bandit regret minimization），目标是最小化能量而非 regret。

---

### 实验设置与评估指标

| 设置项 | 描述 |
|--------|------|
| **算法** | - **EC-GP-UCB**（用于 misspecified GP bandit）<br>- **SquareCB**（用于 RFF-based linear bandit） |
| **基线方法** | - **Full Quantum Kernel**（完整量子核，最高复杂度）<br>- **Classical RBF Kernel**（作为经典核基线） |
| **评估指标** | - **Cumulative Regret** $ R_T = \sum_{t=1}^T (f^*(x^*) - f^*(x_t)) $<br>- **Best Found Energy**（VQE 任务）<br>- **Standard Deviation** over 30 trials |
| **控制变量** | - 调整近似核的维度：LPQK 的子系统数量、RFF 的 $ D $、P-greedy 的基函数数 |
| **实现工具** | PennyLane（量子电路）、NumPy/SciPy（数值计算） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）合成任务（Synthetic Quantum Functions）
- **图 3 & 4** 显示，所有三种近似方法（LPQK, RFF, P-greedy）在 **中等维度** 下均取得最低 regret，呈典型的 **U型曲线**：
  - 维度过低 → 欠拟合（underfitting），regret 高；
  - 维度过高 → 过拟合（overfitting），信息增益过大，regret 升高。
- **最优 RFF 维度**：理论建议 $ D = \sqrt{T} = 10 $（当 $ T=100 $），实验结果完美匹配该预测。
- **性能排序**：**近似核 > Full Quantum Kernel > Classical RBF Kernel**

#### （2）相位分类（Phase Classification）
- **图 6** 显示，使用 LPQK 和 P-greedy 的 EC-GP-UCB 在最优复杂度下，**显著优于完整量子核**。
- 说明：**较低维的核已足以区分相边界**，避免了高维核带来的信息增益惩罚。

#### （3）变分量子本征求解器（VQE）
- **图 7** 显示，在能量最小化任务中：
  - 完整量子核收敛最慢（过于复杂）。
  - **中等维度的 LPQK 和 P-greedy 收敛最快**，达到更低能量。
- 证明：该框架的 **效率优势不仅限于 bandit regret 任务**，也适用于一般优化。

#### （4）更大系统规模（6-qubit）
- **附录图 9 & 10** 验证了在 6-qubit 系统中，**U型趋势依然存在**，表明方法具有良好的可扩展性。

---

### 消融实验结果
- **维度选择消融**：系统地改变近似核维度，验证了 **存在一个最优维度**，支持理论分析。
- **误差 vs. 维度关系**：**图 5** 展示了 LPQK 和 P-greedy 的 **最大范数误差（max-norm error）** 随维度增加而下降，为 regret bound 中的 $ \epsilon $ 提供实证支持。
- **理论最优点验证**：将估计的 $ \epsilon(D) $ 代入 regret bound，计算出的理论最优 $ D $ 与实验峰值高度一致。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **表达力与可学习性之间存在根本权衡**：盲目追求高表达力（如完整量子核）会损害学习效率。
2. ✅ **适度降维的近似核能实现更优性能**：在多个量子任务中，**最佳近似核的 regret 显著低于完整量子核**。
3. ✅ **理论指导实践**：提出的 regret bound 能有效预测最优模型复杂度（如 $ D = \sqrt{T} $ for RFF）。
4. ✅ **量子结构可被利用**：LPQK 对应 Pauli-weight 投影，其性能取决于可观测量的局部性或 Pauli-tail 衰减速度。
5. ✅ **计算效率大幅提升**：从 $ \mathcal{O}(T^3) $ 降至 $ \mathcal{O}(TD^2) $，使大规模应用成为可能。

---

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **Regret bounds 的保守性** | 理论上界可能过于宽松，实际最优维度可能小于理论预测。 |
| **近似误差的不确定性** | RFF 的 $ \epsilon = \mathcal{O}(1/\sqrt{D}) $ 是通用界；若频谱衰减快（如指数衰减），实际误差更小，但需问题特定分析。 |
| **Fourier 表示的适用性限制** | RFF 方法要求量子核具有可处理的 Fourier 分解，某些复杂编码可能不满足。 |
| **忽略 NISQ 噪声建模** | 当前分析假设 i.i.d. 高斯噪声，未考虑读出错误、退相干等结构性硬件噪声。 |
| **量子测量开销未完全建模** | 虽然经典计算加速，但量子端的测量成本（尤其对 LPQK 的多子系统测量）仍可能很高。 |

---

### 未来工作方向
1. **自适应复杂度选择**：设计自动调整近似维度 $ D $ 的算法，而非手动调参。
2. **结合结构性先验**：针对特定物理系统（如局部哈密顿量），设计更具针对性的近似核。
3. **联合优化量子电路与核逼近**：同时优化编码电路结构与近似策略。
4. **更真实的噪声建模**：将设备噪声纳入 regret 分析，发展鲁棒的量子核 bandit 算法。
5. **扩展至其他量子模型**：将该框架应用于 QML 中的其他高维模型，如 quantum neural tangent kernel。

</details>

---

### 14. [GAIA: Geometry-Adaptive Operator Learning for Forward and Inverse Problems](https://arxiv.org/abs/2607.01128)

**Authors**: Meenakshi Krishnan, Pranav Pulijala, Ke Chen, Haizhao Yang, Ramani Duraiswami  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2607.01128v1  

#### Abstract
Operator learning for partial differential equations (PDEs) on arbitrary geometries builds fast neural surrogates for large-scale simulation. Although recent geometry-adaptive neural operators have made substantial progress, they are mainly designed for forward problems in which inputs and outputs s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# GAIA: Geometry-Adaptive Operator Learning for Forward and Inverse Problems 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统神经算子（neural operators）如 **FNO** 和 **DeepONet** 依赖规则网格或固定离散化，难以处理几何形状变化的问题。近期的几何自适应算子（如 **GAOT**, **GINO**, **Transolver**）虽能处理非结构化点云，但主要针对前向问题（forward problems），且通常要求输入与输出共享相同的空间域。

这限制了它们在以下场景的应用：
- **边界值问题 (BVPs)**：输入为边界条件，输出为内部场。
- **逆问题 (Inverse Problems)**：输入为边界测量（如传感器数据），输出为内部物理参数（如电导率、散射系数）。

此外，许多方法需要昂贵的迭代优化或图构建，推理效率低。

### 提出了什么新方法或新思路
本文提出 **GAIA (Geometry-Adaptive Integral Autoencoder)**，一种统一的、无需重新训练即可处理任意几何域上前向与逆问题的神经算子模型。其核心创新包括：

- **双路径几何编码机制 (Dual-pathway tokenization)**：
  - **Boundary Tokenizer**：使用 PointNet 架构从边界点坐标提取全局形状特征。
  - **Slice Tokenizer**：通过软聚类（soft-clustering）将内部场分布与空间坐标联合编码为“状态切片”tokens。
  - 两者结合形成统一的 **geometry tokens**，显式地提供几何上下文。

- **基于 Cross-Attention 的核函数调节机制**：
  - 在积分变换层中，通过 **multi-head cross-attention** 将查询点的局部特征与 geometry tokens 进行交互，使积分核（integral kernel）能够**局部自适应于几何特征**。
  - 这种机制实现了真正的空间自适应（spatially adaptive conditioning），而非仅使用全局标量调节（如 FiLM 或拼接）。

- **支持不同输入/输出域的架构设计**：
  - 采用分离的编码器与解码器查询集，允许输入（如边界测量）与输出（如内部场）位于不同的点集上。
  - 可实现**单次前向传播**求解逆问题和 BVP，无需迭代优化或图构建。

### 相比现有方法的优势
| 特性 | GAIA | GINO | GAOT | Transolver | CORAL | LNO |
|------|------|------|------|------------|--------|-----|
| 支持任意几何 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 支持前向 + 逆问题 | ✅ | ⚠️（可但慢） | ❌ | ❌ | ⚠️（需迭代） | ✅ |
| 输入/输出域可不同 | ✅ | ⚠️（依赖图） | ⚠️（修改后） | ❌ | ✅ | ✅ |
| 无需图构建 | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| 单次推理（amortized） | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 离散化不变性（DI） | ✅ | ⚠️ | ⚠️ | ❌ | ✅ | ✅ |

> ✅ 表示原生支持；⚠️ 表示有限支持或需额外处理；❌ 表示不支持。

---

## 2. 核心实验方法和设置

### 使用的数据集
论文在 **7 个基准任务** 上进行评估，涵盖 2D 和 3D、前向与逆问题，其中 **4 个为新引入或大幅扩展的基准**：

#### 逆问题（Inverse Problems）
| 数据集 | 描述 |
|-------|------|
| **EIT (Electrical Impedance Tomography)** | 从边界电压-电流测量重建内部电导率分布，推广至星形变体几何。 |
| **OT (Optical Tomography)** | 从光源-接收器矩阵重建组织内的散射系数，定义于随机凸五边形域（新数据集）。 |
| **Airfoil Flow Reconstruction** | 从稀疏（10%）、含噪（1% 高斯噪声）的马赫数观测中重建完整流场，几何为变形 NACA 翼型。 |

#### 边界值问题（BVPs）与前向问题
| 数据集 | 描述 |
|-------|------|
| **3D Poisson BVP on MCB** | 在机械零件（齿轮、螺母、接头、螺栓）上求解 Poisson 方程，映射边界 Dirichlet 条件到内部解（新设定）。 |
| **3D Darcy Flow** | 学习渗透率场 $K(x)$ 到压力场 $u(x)$ 的映射，在星形 3D 域上。 |
| **Poisson-Gauss** | 经典 Poisson 问题，源项为多个高斯脉冲，单位正方形域。 |
| **Elasticity** | 超弹性材料应力场预测，带随机孔洞的单位方块。 |

> 所有新数据集已公开：[Google Drive Link](https://drive.google.com/drive/folders/10SrAGvJh14M5APg-oRvTKbOMQYQgS6VR)

### 实验设置和评估指标
- **评估指标**：**median relative L2 error (%)**，对所有任务统一使用。
- **训练平台**：单张 NVIDIA A6000 GPU。
- **优化器**：Adam，余弦退火学习率调度，5% warmup。
- **输入/输出分辨率变化测试**：使用最远点采样（Farthest Point Sampling）生成不同密度的测试点云，验证离散化不变性（Discretization Invariance）。
- **噪声鲁棒性测试**：在测试时添加相对高斯噪声（$ \epsilon \sim \mathcal{N}(0, (n \|x\|_\infty)^2) $），$ n \in \{0\%, 1\%, ..., 10\%\} $。

### 基线方法对比
- **GINO**：图+傅里叶混合架构，需构建半径图（radius graph）。
- **CORAL**：基于隐式神经表示（INR），需每样本拟合 SIREN。
- **Transolver**：基于注意力的物理状态压缩，要求输入输出同网格。
- **GAOT**：图注意力架构，原始版本不支持异构域。
- **LNO**：潜在空间注意力，支持逆问题。
- **NGF**：专用于线性对称 PDE 的前向问题。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1–3）

#### 逆问题性能对比（Median relative L2 error %）
| Dataset | GINO | CORAL | LNO | GAOT | **GAIA** |
|--------|------|-------|-----|------|----------|
| **EIT** | 28.73 | 1.66 | 0.97 | 7.69 | **0.71** |
| **OT** | 1.79 | 2.93 | 1.82 | 5.91 | **1.41** |
| **Airfoil** | 4.50 | 1.60 | 2.26 | 9.95 | **0.58** |

> ✅ **GAIA 在所有逆问题上达到 SOTA**，相比次优方法：
> - EIT 上降低 **27%**（0.71% vs 0.97%）
> - Airfoil 上降低 **64%**（0.58% vs 1.60%）

#### BVP on Mechanical Components（Median relative L2 error %）
| Model | Fitting | Gear | Nut | Screws/Bolts |
|-------|---------|------|-----|-------------|
| CORAL | 18.82 | 13.57 | 23.25 | 20.76 |
| NGF | 11.66 | 5.89 | 5.48 | 6.52 |
| LNO | 14.25 | 6.14 | 6.50 | 5.52 |
| **GAIA** | **10.15** | **3.44** | **5.16** | **3.68** |

> ✅ **GAIA 在所有类别上均最优**，尤其在复杂几何（如齿轮）上优势显著。

#### 前向问题性能对比
| Dataset | GINO | CORAL | Transolver | GAOT | **GAIA** |
|--------|------|-------|-------------|------|----------|
| Poisson-Gauss | 1.16 | 4.38 | 1.46 | 1.23 | **0.71** |
| Elasticity | 1.87 | 2.06 | 0.94 | 0.97 | 1.34 |
| 3D Darcy | 20.04 | 21.44 | 0.73 | 39.88 | 1.11 |

> ✅ GAIA 在 Poisson-Gauss 上表现最佳，在其他任务上具有竞争力。

### 消融实验结果（Ablation Studies）

#### 几何调节机制对比（Table 5）
| Conditioning | Elasticity | Airfoil |
|------------|------------|---------|
| Concatenation | 1.77 | 5.91 |
| FiLM | 1.88 | 0.88 |
| **Cross-attention** | **1.34** | **0.58** |

> ✅ **Cross-attention 显著优于全局拼接或 FiLM**，证明空间自适应调节的重要性。

#### Tokenizer 路径消融
| Variant | Elasticity | Airfoil |
|--------|------------|---------|
| No boundary tokens | 2.28 | 1.06 |
| No slice tokens | 1.90 | 1.03 |
| **Full model** | **1.34** | **0.58** |

> ✅ 两种 tokenizer 路径均贡献显著，表明**边界形状**与**场分布**信息互补。

#### 其他敏感性分析
- **Token 数量与维度**：性能对 token count/dim 不敏感，在合理范围内稳定。
- **块数量（Blocks）**：5 层效果最佳，模型对此选择稳健。

---

## 4. 关键结论和发现

### 主要发现
1. **统一架构可行性**：GAIA 成功实现了**单一架构同时高效处理前向与逆问题**，无需重新训练或迭代优化。
2. **显式几何编码有效**：通过 **boundary + slice tokens** 显式建模几何，显著优于隐式方法（如 CORAL）或图方法（如 GINO）。
3. **Cross-Attention 是关键**：相比全局调节，**cross-attention 实现的空间自适应核调节**是提升精度的核心。
4. **离散化不变性强**：在不同分辨率下测试时，GAIA 性能稳定，而 Transolver/GAOT 在非训练分辨率下严重退化（见 Figure 5a）。
5. **噪声鲁棒性好**：在高达 10% 噪声下仍能“优雅退化”（graceful degradation），无崩溃现象。

### 方法的局限性
1. **仅适用于稳态问题**：当前框架未考虑时间演化，无法直接应用于 time-dependent PDEs。
2. **在固定密集网格上前向任务略逊**：当部署分辨率与训练一致时，Transolver 在某些前向任务上略优，说明 GAIA 在**峰值精度 vs 泛化性**之间做了权衡。
3. **计算复杂度为 $O(NK)$**：对于超大点云（$N > 10^6$），cross-attention 开销可能成为瓶颈，需引入稀疏注意力或潜在压缩。

### 未来工作方向
- 扩展至 **time-dependent PDEs**，结合 RNN 或扩散时间步长。
- 探索 **sparse attention mechanisms** 以应对大规模 3D 点云。
- 应用于更多实际工程场景，如多物理场耦合、不确定性量化等。
- 进一步优化 token 表示，探索更高效的几何抽象方式。

---

> **总结**：GAIA 提出了一种新颖且强大的几何自适应神经算子框架，首次实现了在任意几何上对前向与逆问题的统一、高效、单次求解。其实验全面、结果领先，为科学机器学习中的 PDE 求解提供了重要进展。

</details>

---

### 15. [TRIE: An Evaluation Framework for Stochastic PDE Surrogates](https://arxiv.org/abs/2607.00196)

**Authors**: Bharat Srikishan, Javier E. Santos, Nikhil Muralidhar, Charles D. Young  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.00196v1  

#### Abstract
Many scientific systems exhibit uncertainty from stochastic forcing, unresolved degrees of freedom, or imperfect observations, making reliable surrogate forecasting fundamentally distributional rather than pointwise. For such systems, deterministic neural surrogates fail to capture statistical measu...

---

### 16. [From Pixels to Temporal Correlations: Learning Informative Representations for Reinforcement Learning Pre-training](https://arxiv.org/abs/2607.00811)

**Authors**: Jinwen Wang, Youfang Lin, Xiaobo Hu, Siyu Yang, Sheng Han, Shuo Wang, Kai Lv  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.00811v1  

#### Abstract
Unsupervised pre-training on large-scale datasets has demonstrated significant potential for improving the sample efficiency and performance of Reinforcement Learning (RL). Given the large-scale action-free internet videos, existing methods utilize single-step transition prediction and image reconst...

---

### 17. [GSRQ: Gain-Shape Residual Quantization for Sub-1-bit KV Cache](https://arxiv.org/abs/2607.01065)

**Authors**: Soosung Kim, Minjae Park, Eui-Young Chung, Jaeyong Chung  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2607.01065v1  

#### Abstract
The deployment of Large Language Models (LLMs) with extended context windows is increasingly constrained by the linear growth of Key-Value (KV) cache memory. Vector Quantization (VQ), particularly Residual Quantization (RQ), is a promising approach for pushing KV cache storage toward the sub-1-bit r...

---

### 18. [Agentic generation of verifiable rules for deterministic, self-expanding reaction classification](https://arxiv.org/abs/2607.01061)

**Authors**: Daniel Armstrong, Maarten Dobbelaere, Valentas Olikauskas, Helena Avila, Octavian Susanu, J\'er\^ome Waser, Philippe Schwaller  
**Category**: cs.AI  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.01061v1  

#### Abstract
Computer-assisted synthesis planning breaks target molecules into accessible precursors using large libraries of reaction rules that assign each transformation a deterministic, interpretable label. But chemistry is long-tailed, making manual encoding intractable, and existing tools rely on fixed rul...

---

### 19. [Can Agents Generalize to the Open World? Unveiling the Fragility of Static Training in Tool Use](https://arxiv.org/abs/2607.01084)

**Authors**: Song-Lin Lv, Weiming Wu, Rui Zhu, Zi-Jian Cheng, Lan-Zhe Guo  
**Category**: cs.AI  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.01084v1  

#### Abstract
While Large Language Model (LLM) agents demonstrate proficiency in static benchmarks, their deployment in real-world scenarios is hindered by the dynamic nature of user queries, tool sets, and interaction dynamics. To address this generalization gap, we formalize OpenAgent (Tool-Use Agent in Open-Wo...

---

### 20. [Optimal Resource Utilization for Autonomous Laboratory Orchestrators](https://arxiv.org/abs/2607.01188)

**Authors**: Austin McDannald, Julia Tisaranni, Howie Joress  
**Category**: cs.AI  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.01188v1  

#### Abstract
In autonomous laboratories, AI agents suggest the next batch of experiments to do. However, planning and executing those tasks taking full advantage of the available resources is a completely different question. This can be challenging when dealing with real-world hardware constraints, especially so...

---

### 21. [Benchmarking Frontier LLMs on Arabic Cultural and Sociolinguistic Knowledge: A Cross-Evaluation Framework with Human SME Ground Truth](https://arxiv.org/abs/2607.00139)

**Authors**: Sajjad Abdoli, Ghassan Al-Sumaidaee, Ahmad ElShiekh, Clayton W. Taylor, Ahmed Rashad  
**Category**: cs.CL  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.00139v1  

#### Abstract
The cost of human expert evaluation is a principal bottleneck to deploying language models in specialized, high-stakes domains. This is particularly acute for Arabic sociolinguistic knowledge: credible grading requires not only linguistic fluency but deep cultural familiarity that cannot be approxim...

---

### 22. [ALEE: Any-Language Evaluation of Embeddings via English-Centric Minimal Pairs](https://arxiv.org/abs/2607.00171)

**Authors**: Andrianos Michail, Stylianos Psychias, Michelle Wastl, Simon Clematide, Rico Sennrich, Juri Opitz  
**Category**: cs.CL  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.00171v1  

#### Abstract
Text embeddings are standard for semantic similarity tasks, yet their evaluation remains an open challenge. Current benchmarks are static, cover only a limited set of languages, are often domain-specific, susceptible to overfitting, and poorly representative of low-resource languages. To address the...

---

### 23. [MetaHOPE: A Metaphor-Oriented Evaluation Framework for Analysing MT and LLM Translation Errors](https://arxiv.org/abs/2607.00848)

**Authors**: Jiahui Liang, Lifeng Han  
**Category**: cs.CL  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.00848v1  

#### Abstract
In this opinion paper, we propose MetaHOPE, an error severity-aware annotation framework for evaluating metaphor translations. Metaphors present challenges for machine translation (MT) and natural language understanding and processing (NLU, NLP), because it presents the features of semantic complexi...

---

### 24. [MultiSynt/MT: Trillion-Token Multi-Parallel Pre-Training Data Translated Across 36 Languages](https://arxiv.org/abs/2607.00890)

**Authors**: Maximilian Idahl, J\"org Tiedemann, Sampo Pyysalo, David Salinas, Tomasz Galica, Shenbin Qian, Tudor Nicolae Mateiu, Zihao Li, Anna Lokrantz, Fedor Vitiugin, Andr\'e F. T. Martins, Jenna Kanerva, Filip Ginter, Matthias Lindemann, Tim Isbister, Birger Moell, Jonas Lindh, Jan Haji\v{c}, Jenia Jitsev, Andrey Kutuzov, Stephan Oepen, Gema Ram\'irez-S\'anchez  
**Category**: cs.CL  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.00890v1  

#### Abstract
Open web-scale pre-training corpora remain concentrated in English, limiting multilingual LLM development. We introduce MultiSynt/MT, an open synthetic parallel corpus with approximately 4.8 trillion target-language tokens across 36 European languages, produced by translating 100 billion high-qualit...

---

### 25. [Svarna: An Open Corpus Workbench for Modern Greek](https://arxiv.org/abs/2607.00970)

**Authors**: Stergios Chatzikyriakidis  
**Category**: cs.CL  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.00970v1  

#### Abstract
This paper introduces Svarna, a free, open-source, web-based corpus workbench for modern Greek. Svarna integrates five databases covering various registers, institutional, literary, dialectal, social media, and historical, to provide a total of more than 507 million words and around 29 million sente...

---

### 26. [StateFlow: Dual-State Recurrent Modeling for Long-Horizon Time Series Forecasting](https://arxiv.org/abs/2607.00197)

**Authors**: Haroon Gharwi, Yue Dai, Kai Shu  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.00197v1  

#### Abstract
Long-horizon multivariate time series forecasting (LTSF) remains challenging due to non-stationarity, regime shifts, and error accumulation. The Variability-Aware Recursive Neural Network (VARNN) is designed to track such variability by maintaining a residual-memory state driven by one-step predicti...

---

### 27. [PRISM: Prioritized Channel Importance with Semi-supervised Domain Adaptation for Cross-Subject EEG Emotion Recognition](https://arxiv.org/abs/2607.00358)

**Authors**: Xin Zhou, Xiang Zhang, Hao Deng, Lijun Yin  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.00358v1  

#### Abstract
Electroencephalogram (EEG) captures endogenous brain activity with high temporal fidelity and holds substantial promise for precise emotion decoding. However, channel redundancy and pronounced inter-subject variability remain key obstacles to scalable generalization. To address these limitations, we...

---

### 28. [Measuring Dead Directions: Decomposing and Classifying Singular Structure off Canonical Alignment](https://arxiv.org/abs/2607.00603)

**Authors**: Tejas Pradeep Shirodkar  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.00603v1  

#### Abstract
We give a descent-free, alignment-free measurement of singular structure on trained networks. At a single frozen checkpoint the read recovers the order $k$ of each dead direction from the directional-Fisher rate, the master invariant from which the per-direction learning coefficient $1/(2k)$ follows...

---

### 29. [Explainable AI for Cancer Drug Response Prediction: Beyond Univariate Feature Attributions](https://arxiv.org/abs/2607.00931)

**Authors**: Martino Ciaperoni, Margherita Lalli, Simone Piaggesi, Martina Varisco, Francesco Carli, Riccardo Guidotti, Dino Pedreschi, Francesco Raimondi, Fosca Giannotti  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.00931v1  

#### Abstract
Predicting cancer drug response from transcriptomic profiles is a cornerstone of precision oncology, yet the scientific value of machine learning models hinges not solely on predictive accuracy, but also on their capacity to generate reliable biological insights. Current explainability approaches in...

---

### 30. [Quantum vs. Classical Machine Learning: A Unified Empirical Comparison](https://arxiv.org/abs/2607.01197)

**Authors**: Chuanming Yu, Jiaming Liu, Zihao Ge, Xiongfei Wu, Lulu Zhu, Pengzhan Zhao, Jianjun Zhao  
**Category**: cs.LG  
**Published**: 2026-07-02  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2607.01197v1  

#### Abstract
Quantum computing has emerged as a promising computational paradigm for machine learning (ML), with the potential to offer computational advantages over classical approaches. At this stage, the evidence supporting the performance and advantages of quantum machine learning (QML) models relative to cl...

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

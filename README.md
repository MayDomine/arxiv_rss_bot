# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-15 08:31:05 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [EnergyLens: Predictive Energy-Aware Exploration for Multi-GPU LLM Inference Optimization](https://arxiv.org/abs/2605.14249)

**Authors**: Zhiye Song, Kyungmi Lee, Eun Kyung Lee, Xin Zhang, Tamar Eilam, Anantha P. Chandrakasan  
**Category**: cs.LG  
**Published**: 2026-05-15  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2605.14249v1  

#### Abstract
We present EnergyLens, an end-to-end framework for energy-aware large language model (LLM) inference optimization. As LLMs scale, predicting and reducing their energy footprint has become critical for sustainability and datacenter operations, yet existing approaches either require production-level c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EnergyLens: Predictive Energy-Aware Exploration for Multi-GPU LLM Inference Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
随着大语言模型（LLM）规模不断扩大，其推理过程的能耗已成为数据中心可持续运营的关键挑战。然而，现有的能耗建模方法存在以下不足：
- **依赖实现和硬件**：需要实际部署代码和GPU访问，成本高、周期长。
- **忽略多GPU效应**：无法准确捕捉 **tensor parallelism**、**expert parallelism** 和 **compute-communication overlap** 等分布式执行带来的能耗变化。
- **缺乏细粒度分析**：难以提供能量归因（energy attribution），无法指导优化决策。

因此，开发者在早期设计阶段缺乏有效的工具来探索不同配置下的能效权衡。

### **提出了什么新方法或新思路**
本文提出 **EnergyLens**，一个端到端的、无需执行即可预测多GPU LLM 推理能耗的框架，具有以下核心创新：

- **Einsum-based 高层接口**：通过简洁的 `einsum` 表达式描述 LLM 架构（如融合、并行策略、重叠机制），无需底层实现或 profiling。
- **分布式的能耗建模栈**：
  - **Empirically-driven communication energy model**：基于实测数据建模通信能耗，考虑消息大小和 **SM allocation** 对延迟和能耗的影响。
  - **Overlap-aware aggregation**：支持对 Megatron-style 的计算-通信重叠进行建模。
  - **Load-imbalance-aware MoE modeling**：针对 MoE 模型中的专家负载不均衡问题，引入有效参数 $T$（每专家token数）和 $E$（每GPU激活专家数）进行建模。
- **能量驱动的探索能力**：支持快速生成 **energy-latency Pareto front**，帮助识别最优部署配置。

### **相比现有方法的优势**
| 方法 | 是否需实现 | 支持多GPU | 支持MoE | 支持Overlap | 提供能量分解 |
|------|------------|-----------|---------|-------------|----------------|
| TDP-based estimation | 否 | ❌ | ❌ | ❌ | ❌ |
| LLMCO2 (Fu et al.) | 是 | ❌ | ❌ | ❌ | ✅ |
| EnergAIzer (Lee et al.) | 是 | ❌ | ❌ | ❌ | ✅ |
| **EnergyLens (Ours)** | **否** | ✅ | ✅ | ✅ | ✅ |

> ✅ **优势总结**：EnergyLens 在无需实现、无需GPU访问的前提下，首次实现了对 **multi-GPU + MoE + overlap** 场景下能耗的细粒度预测，填补了早期设计空间探索的空白。

---

## **2. 核心实验方法和设置**

### **使用的模型和数据集**
- **模型**：
  - **Dense Models**: Llama3-8B, Llama3-70B
  - **MoE Model**: Qwen3-30B-A3B（含3B激活参数，A3B表示每个token激活3个专家）
- **数据集**：使用 Hugging Face 的 `wikipedia` 数据集进行 workload profiling。

### **实验设置**
- **硬件平台**：8× A100-SXM4-80GB GPUs
- **推理引擎**：
  - **TensorRT-LLM v0.14/v1.0**：用于 dense 和 MoE 模型的 baseline 测量
  - **Megatron-LM v0.15.3**：用于验证 compute-communication overlap 场景
- **测量方式**：
  - 能耗：通过 **NVML** 获取 GPU 总能耗
  - 延迟：端到端测量 + **Torch Profiler** 验证通信原语
  - 热身时间：20秒，确保稳态行为

### **评估指标**
- **MAPE**（Mean Absolute Percentage Error）：用于衡量 EnergyLens 预测值与真实值之间的误差。
- **Energy-Latency Trade-off** 分析：
  - **ETFT**（Energy to First Token）：类似 TTFT，衡量 prefill 阶段总能耗
  - **EPOT**（Energy Per Output Token）：类似 TPOT，衡量 decode 阶段单位输出token能耗
- **Pareto Frontier Recovery Rate**：评估 EnergyLens 是否能正确识别最优配置集合。

### **基线方法对比**
本文未直接对比其他完整框架（因其不支持相同功能），而是通过以下方式间接比较：
- 展示 **TDP-based 方法** 在 decode 阶段可能高估能耗达 **60%**（图9）
- 引用并集成 **EnergAIzer**, **NeuSight**, **Li et al.** 作为 compute kernel backend，验证其在 decode 小负载下的局限性

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（MAPE）**
| 模型 | 场景 | 能耗 MAPE | 延迟 MAPE |
|------|------|----------|----------|
| Llama3-8B | 单卡 | 11.31% | 11.30% |
| Llama3-70B | Tensor Parallelism (TP) | 12.18% | 12.39% |
| Llama3-70B | TP + Overlap | 12.97% | 11.18% |
| Qwen3-30B-A3B | TP+EP | **9.25%** | 21.85% |
| Qwen3-30B-A3B | Decode | 13.19% | 27.16% |

> ✅ 所有能耗预测 MAPE 均低于 **13.2%**，足以用于设计空间探索。

### **与基线方法的对比结果**
- **vs TDP-based estimation**：
  - 在 decode 阶段，TDP 方法会 **高估能耗最多达60%**（图9），而 EnergyLens 准确反映 decode 功率远低于 peak。
- **vs 直接 exhaustive profiling**：
  - 对 Llama3-70B 进行 7 batch sizes × 32 input lengths 的 fusion 策略比较，需 **12小时工程+GPU资源**；
  - 使用 EnergyLens 仅需 **15分钟** 自动生成结果和可视化。

### **消融实验与关键发现**
#### **(1) Kernel Fusion 影响**
- **Prefill 阶段平均节能 4.37%**
- **Decode 阶段平均节能 8.18%**
- 主因：融合减少内存访问开销，尤其在低算力强度的 decode 阶段更显著（图16）

#### **(2) Compute-Communication Overlap 效果**
- Overlap 并非总是有益：在某些配置下（如 TP2 + 小batch），**关闭 overlap 反而更优**。
- EnergyLens 成功识别出 **Pareto-optimal 配置中多数为 non-overlap 设置**。
- 若采用“最大重叠”启发式（max SM for comm），只能恢复 **20% 的 Pareto frontier**；
- EnergyLens 预测后选择，可恢复 **69% 的 Pareto frontier**（图8）。

#### **(3) Disaggregated Serving 动机**
- **Prefill**：小 batch + 低 TP 更节能（通信开销低）
- **Decode**：大 batch 显著降低 EPOT（提升算力强度）
- 两者目标冲突 → **强烈支持将 prefill 和 decode 拆分优化（disaggregated serving）**

---

## **4. 关键结论和发现**

### **主要发现**
1. **能耗差异巨大**：不同配置下，prefill 和 decode 的能效差异可达 **1.47× 至 52.9×**，凸显优化必要性。
2. **通信不可忽视**：在 Llama3-70B + TP8 下，AllReduce 占总能耗 **23%**。
3. **Overlap 难以直觉判断**：最大化重叠并非最优；SM 分配需权衡 compute 与 comm 资源。
4. **EnergyLens 预测足够准**：尽管 decode 延迟 MAPE 较高（~25%），但 **能耗趋势和相对排序准确**，足以支撑设计决策。
5. **支持早期探索**：开发者可在无实现、无GPU的情况下，快速评估 fusion、parallelism、overlap 等策略的能效影响。

### **方法的局限性**
- **Decode 延迟预测精度较低**（MAPE ~25–27%）：
  - 原因：decode 阶段 GEMM 算子算力强度极低、形状高度偏斜，现有 kernel model 普遍表现不佳（表3）。
- **依赖 empirical communication model**：
  - 当网络拓扑或 NCCL 版本变化时，需重新校准通信能耗曲线。
- **MoE routing 统计假设**：
  - 默认均匀路由，若实际负载极度不均，可能影响精度（但可通过单卡模拟提升）。

### **未来工作方向**
- 扩展支持更多并行范式，如 **pipeline parallelism** 和 **context parallelism**（已初步支持，见附录F）。
- 集成更精确的小核延迟模型，改善 decode 阶段预测。
- 结合碳排放因子，扩展为 **end-to-end carbon footprint predictor**。
- 支持异构 GPU 和 disaggregated infrastructure（如 GPU + NIC 解耦）。

---

> 📌 **总结一句话**：  
> **EnergyLens 提供了一个轻量级、无需实现的 einsum 接口，首次实现了对 multi-GPU + MoE + overlap 场景下 LLM 推理能耗的高精度预测（MAPE < 13.2%），揭示了巨大的能效优化空间，并能可靠地识别 Pareto 最优配置，为绿色 AI 推理系统的设计提供了强大工具。**

</details>

---

### 2. [BEAM: Binary Expert Activation Masking for Dynamic Routing in MoE](https://arxiv.org/abs/2605.14438)

**Authors**: Juntong Wu, Jialiang Cheng, Qishen Yin, Yue Dai, Yuliang Yan, Fuyu Lv, Ou Dan, Li Yuan  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.14438v1  

#### Abstract
Mixture-of-Experts (MoE) architectures enhance the efficiency of large language models by activating only a subset of experts per token. However, standard MoE employs a fixed Top-K routing strategy, leading to redundant computation and suboptimal inference latency. Existing acceleration methods eith...

---

### 3. [An Interpretable Latency Model for Speculative Decoding in LLM Serving](https://arxiv.org/abs/2605.15051)

**Authors**: Linghao Kong, Megan Flynn, Michael Peng, Nir Shavit, Mark Kurtz, Alexandre Marques  
**Category**: cs.LG  
**Published**: 2026-05-15  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.15051v1  

#### Abstract
Speculative decoding (SD) accelerates large language model (LLM) inference by using a smaller draft model to propose multiple tokens that are verified by a larger target model in parallel. While prior work demonstrates substantial speedups in isolated or fixed-batch settings, the behavior of SD in p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An Interpretable Latency Model for Speculative Decoding in LLM Serving

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前大多数关于 **Speculative Decoding (SD)** 的研究集中在理想化、单请求或固定 batch size 的环境下进行评估，忽略了真实生产环境中 **动态负载变化** 和 **连续批处理（continuous batching）** 对性能的影响。在实际的 LLM serving 系统（如 vLLM）中，effective batch size 是由请求速率、调度策略和系统并发共同决定的，而非直接控制参数。

因此，现有工作缺乏一个能够解释 SD 在不同负载条件下如何影响端到端延迟（latency）的**可解释模型**，导致难以在部署时合理配置 SD 参数（如 draft length $k$、acceptance rate $\alpha$）。

### 提出了什么新方法或新思路
本文提出了一种**简单且可解释的 latency model**，用于建模 SD 在真实 LLM serving 场景下的行为，其核心思想包括：

- **基于 Little's Law 推断 effective batch size**：利用请求到达率（RPS）和实测延迟反推出系统的平均并发请求数 $B = \text{RPS} \times L$，从而绕过无法直接观测的 batch size。
- **分解服务成本为 load-independent 与 load-dependent 成分**：
  - $C_1$: 固定开销（如权重加载、初始化）
  - $C_2$: 每单位并发带来的增量开销（如 KV-cache 管理、内存带宽竞争）
- **构建闭式表达式描述延迟随负载的变化趋势**：
  $$
  L = \frac{C_1}{1 - \text{RPS} \cdot C_2}
  $$
- **将 SD 引入该框架，通过 prefill/verify/draft 三阶段成本分解建模其对 $C_1$ 和 $C_2$ 的影响**，并进一步推导出 SD 的 speedup 表达式：
  $$
  \text{Speedup} = \frac{1}{C_{1,R}} \left(1 + (1 - C_{2,R}) \cdot \frac{\text{RPS} \cdot C_{2,D}}{1 - \text{RPS} \cdot C_{2,D}}\right)
  $$
  其中 $C_{1,R} = C_{1,\text{SD}} / C_{1,D}$, $C_{2,R} = C_{2,\text{SD}} / C_{2,D}$。

此外，作者还将该模型扩展至 **Mixture of Experts (MoE)** 架构，引入 expert coverage 因子 $\phi$ 来捕捉稀疏激活带来的非线性负载效应。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可解释性** | 明确揭示了为何 SD 的加速效果会随着负载增加而减弱（即 $C_{2,R} > 1$ 导致 load-dependent 开销上升）。 |
| **实用性** | 不依赖具体实现细节（如 kernel 调度），适用于现代 vLLM 类系统，仅需少量测量即可拟合参数。 |
| **通用性** | 可推广到不同模型大小（8B~235B）、架构（dense/MoE）、硬件平台（A100/H100）。 |
| **指导意义** | 揭示了最优 draft length $k$ 随负载变化的趋势：低负载下偏好大 $k$，高负载下应减小 $k$。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 输入文本来自经典小说 *Pride and Prejudice*（Jane Austen, 1813），模拟真实生成任务。
- 模型前缀长度（prefill length）和解码长度（decode length）均设为 {256, 512, 768, 1024} tokens，共 16 种组合。

### 实验设置和评估指标
- **测试平台**：
  - 主要使用单张 NVIDIA A100 SXM GPU（部分大模型用多卡）
  - 验证阶段也在 H100 上复现实验以检验跨硬件鲁棒性
- **推理引擎**：vLLM v0.13.0 + GuideLLM v0.5.2 控制负载
- **SD 设置**：
  - 默认使用 EAGLE-3 风格的轻量 drafter
  - 支持 vanilla SD（独立 drafter 模型，如 8B draft 70B）
  - 控制变量：draft length $k \in [1,10]$，acceptance rate $\alpha \in [50\%, 100\%]$
- **负载扫描方式**：
  - 从同步执行（batch=1）逐步提升 RPS 至接近饱和点（排除 preemption 区域）
  - 每个配置运行 9 个稳定 RPS 点，测量平均 end-to-end latency

### 基线方法对比
- **Baseline**：vanilla autoregressive decoding（无 SD）
- **对比对象**：
  - 不同 $k$, $\alpha$ 下的 SD 性能
  - 不同 verifier-drafter 组合（如 Llama-70B + 8B drafter）
  - Dense vs MoE 架构差异
- **评估指标**：
  - 平均延迟 $L$
  - 拟合优度 $R^2$（衡量模型预测准确性）
  - speedup = $L_{\text{base}} / L_{\text{SD}}$

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 最大 speedup（低负载） | 高负载下 speedup 衰退程度 | 拟合 $R^2$ |
|------|------------------------|----------------------------|------------|
| Llama-3.1-8B | ~3.5x | 显著下降 | 0.99 |
| Qwen3-32B | ~2.5x | 中等下降 | 0.99 |
| gpt-oss-20b (MoE) | ~2.0x | 较快衰退 | 0.96 |
| Qwen3-235B-A22B (MoE) | ~2.0x | 明显衰退 | 0.96 |

> 图1显示所有模型在归一化坐标 $(\text{RPS} \cdot C_2, L/C_1)$ 下的数据点几乎完美地坍缩到理论曲线 $y = 1/(1-x)$ 上，验证了基础模型的有效性。

### 与基线方法的对比结果
- **低负载时 SD 显著提速**：当 RPS 很小时，speedup 主要由 $1/C_{1,R}$ 决定，由于 SD 减少了 verifier 的调用次数，$C_{1,R} < 1$，带来显著加速（可达 2–3.5x）。
- **高负载时 speedup 普遍衰减甚至反转**：
  - 多数情况下 $C_{2,R} > 1$，说明 SD 增加了每并发请求的资源争用（尤其是 drafting 阶段增加了计算负担）
  - 当 $C_{2,R} > 1$ 时，speedup 随 RPS 增加而下降；只有极高 acceptance rate（>90%）时才可能 $C_{2,R} < 1$，此时 speedup 反而随负载上升
- **MoE 模型表现特殊**：
  - 低负载下因 expert 稀疏激活，延迟低于 dense 模型预测值
  - 随着 RPS 或 $k$ 增加，expert coverage 上升，性能趋近于 dense scaling

### 消融实验结果
#### （1）Draft Length $k$ 的权衡
- **最小化 $C_{1,R}$ 的最佳 $k$**：通常较大（如 7–10），有利于摊销 verifier 开销
- **最小化 $C_{2,R}$ 的最佳 $k$**：通常较小（如 1–3），避免 drafting 引发过多并发压力
- ⇒ 存在 trade-off：batch size=1 下最优的 $k$ 在高吞吐场景下并非最优

#### （2）Acceptance Rate $\alpha$ 的影响
- $\alpha$ 越高 → $C_{1,R}$ 和 $C_{2,R}$ 均降低
- 当 $\alpha > 90\%$ 时，有可能实现 $C_{2,R} < 1$，从而使 speedup 随负载增加而增强

#### （3）模型规模扩展性分析（Fig. 5 & 6）
- 所有系数（$c_{1,p}, c_{1,v}, c_{2,p}, c_{2,v}, c_{2,d}$）大致随 verifier/drafter 参数量线性增长
- $c_{2,v}$ 还受 context length 影响：平均 attention cost ∝ prefill + ½ decode length
- leave-n-out 分析表明模型具有良好的泛化能力（即使未见过的 prefill/decode 配置也能准确预测）

#### （4）MoE-aware 模型改进效果（Fig. 8）
- 加入 expert coverage $\phi$ 后，MoE 模型在低 RPS 下的预测误差大幅减少
- 例如 Qwen3-30B-A3B 的 $R^2$ 从 0.93 提升至 0.97+

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **SD 的加速效果高度依赖负载水平**：在低请求率下表现优异，但在高并发下常因增加 load-dependent 开销而导致 speedup 衰退。
2. ✅ **存在“最优 draft length”的负载敏感性**：低负载宜选大 $k$，高负载宜选小 $k$，传统 batch=1 测试不足以指导线上调参。
3. ✅ **可通过 $C_{1,R}$ 和 $C_{2,R}$ 两个比值判断 SD 是否值得启用**：
   - 若 $C_{1,R} \ll 1$ 且 $C_{2,R} \approx 1$，则 SD 有利；
   - 若 $C_{2,R} \gg 1$，则高负载下可能不如 vanilla decoding。
4. ✅ **MoE 模型在低负载下受益更多，但随负载上升收益迅速收敛**，因其 expert coverage 随并发增加而上升。
5. ✅ **同一套建模范式可统一描述 dense 和 MoE、多种 verifier/drafter 组合、不同硬件平台的行为**，具备强泛化性。

### 方法的局限性
- ❌ **仅适用于 pre-saturation regime**：不建模 preemption 或突发流量下的不稳定状态。
- ❌ **假设恒定 acceptance probability**：未考虑 token-level 动态变化或树状 speculative 结构。
- ❌ **基于平均延迟建模**：虽在 appendix 中验证了 p95/p99 的近似有效性，但仍非完整分布建模。
- ❌ **系数具系统依赖性**：$C_1, C_2$ 需要在目标系统上重新测量，不能跨平台直接迁移。

### 未来工作方向
- 🔄 将模型扩展至 **bursty workloads** 和 **动态负载变化** 场景
- ⚖️ 纳入 **preemption 和 timeout 机制** 的影响
- 📊 发展 **latency distribution 建模能力**（如使用 queueing network 或 diffusion approximations）
- 🔁 支持更复杂的 SD 变体，如 **adaptive drafting**, **tree verification**, **multi-step rollback**
- 🛠️ 构建自动化 tuning 工具，根据实时 RPS 和 observed $\alpha$ 动态推荐最优 $k$

---

> **总结一句话**：  
> 本文提出了首个能解释 **Speculative Decoding 在真实 LLM serving 系统中随负载变化的性能演化规律** 的可解释 latency model，揭示了“为何 SD 加速常随负载上升而失效”的根本原因，并为部署中的参数调优提供了理论依据和实用工具。

</details>

---

### 4. [Performance-Driven Policy Optimization for Speculative Decoding with Adaptive Windowing](https://arxiv.org/abs/2605.14978)

**Authors**: Jie Jiang, Xing Sun  
**Category**: cs.CL  
**Published**: 2026-05-15  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.14978v1  

#### Abstract
Speculative decoding accelerates LLM inference by having a lightweight draft model propose speculative windows of candidate tokens for parallel verification by a larger target model. In practice, speculative efficiency is often bottlenecked by hard-to-draft positions, where an early mismatch truncat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Performance-Driven Policy Optimization for Speculative Decoding with Adaptive Windowing

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Speculative Decoding** 中，一个轻量级的 **drafter** 模型生成候选 token 序列（speculative window），由更大的 **target model** 并行验证。然而，当前大多数 drafter 的训练仍基于 **token-level supervised objectives**（如交叉熵损失），这与推理阶段实际追求的目标——**窗口级别的接受长度（accepted prefix length）和端到端加速比（speedup）**——存在严重不匹配。

具体而言：
- 即使窗口中大部分 token 预测准确，只要早期出现一个错误，就会导致整个后续窗口被截断。
- 因此，提升 token-level 准确率未必能有效提高 speculative efficiency。
- 这种“前缀敏感性”（prefix-sensitive）特性未被传统训练目标所建模。

### 提出了什么新方法或新思路
本文提出 **PPOW**（Performance-Driven Policy Optimization with Adaptive Windowing），一种全新的 **window-level reinforcement learning** 框架来优化 drafter，其核心思想是将训练目标从 token-level imitative learning 转向 **performance-driven window-level optimization**。

#### 主要创新组件：
1. **Cost-Aware Speedup Reward**
   - 奖励函数定义为 $ R_{\text{speedup}} = \frac{k}{k\gamma + 1} $
   - 其中 $ k $ 是接受长度，$ \gamma $ 是 drafter 相对于 target model 的计算成本比。
   - 该奖励直接反映推理效率，并考虑了 drafting 和 verification 的相对开销。

2. **Distribution-Based Proximity Reward**
   - 当验证因早期不匹配而失败时（$ k=0 $），若生成的 speculative window 在 target model 下的整体 log-likelihood 接近最优序列，则仍给予部分奖励。
   - 形式化为 $ R_{\text{dist}} = \eta \cdot \mathbb{1}_{[\Delta < \epsilon]} $，其中 $ \Delta $ 是与 target-preferred window 的累积 log-prob 差异。
   - 提供稀疏反馈下的辅助学习信号。

3. **Adaptive Divergence-Aware Windowing (ADAW)**
   - 不对所有 speculative windows 均匀采样，而是优先选择那些更可能成为“瓶颈”的窗口进行训练。
   - 定义 **token-level criticality score**：  
     $ v_t = C(P_t) \cdot D_{KL}(P_t \| Q_t) $，  
     其中 $ C(P_t) = 1 - \frac{H(P_t)}{\log |\mathcal{V}|} $ 表示 target 分布的置信度（低熵则高置信），$ D_{KL} $ 表示 KL 散度。
   - 对每个窗口内 token 的 criticality 取平均，作为窗口优先级权重。

### 相比现有方法的优势
- **目标对齐**：首次将 drafter 训练目标与 speculative decoding 的实际性能（acceptance length, speedup）直接对齐。
- **信号更密集**：通过 proximity reward 缓解早期截断带来的稀疏奖励问题。
- **训练更高效**：ADAW 聚焦于关键瓶颈位置，避免在已掌握的简单窗口上浪费资源。
- **兼容性强**：可应用于各类 drafter 架构（文中基于 EAGLE-family 实现）。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MT-Bench**：多轮对话任务，评估开放生成场景下的表现。
- **HumanEval**：代码生成任务，测试逻辑结构强的任务。
- **GSM8K**：数学推理任务，要求精确且连贯的推导过程。
- 所有实验采用统一的 decoding 协议（tree decoding, rejection sampling, temperature=0.0 或 1.0）。

### 实验设置和评估指标
| 设置项 | 描述 |
|------|------|
| **模型家族** | LLaMA-3（8B/70B）、Qwen3（8B/32B） |
| **Drafter 架构** | 基于 EAGLE-3 的 feature-based drafter |
| **训练流程** | 两阶段：<br>1. Supervised 初始化<br>2. PPOW 强化学习微调 |
| **Speculative Window Size** | $ K = 10 $ |
| **Rollout Group Size** | $ G_{\text{roll}} = 8 $ |

#### 评估指标
- **Average Acceptance Length ($ \bar{T} $)**：每步 speculative verification 平均接受的 token 数。
- **Speedup Ratio**：相对于 vanilla autoregressive decoding 的端到端加速比。

### 基线方法对比
- **EAGLE-3**：当前主流的 feature-based drafter，使用监督学习训练。
- **GRIFFIN**：强调 token 对齐的先进 drafter。
- **Continued Supervised Training (CST)**：在同一 checkpoint 上继续进行监督训练，用于验证 RL 是否带来额外增益。
- 其他补充基线见附录：OSD、Lookahead、FastDraft 等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
| Model | Method | MT-Bench ($ \bar{T} $) | MT-Bench (Speedup) | HumanEval ($ \bar{T} $) | GSM8K ($ \bar{T} $) | Mean $ \bar{T} $ |
|-------|--------|--------------------------|--------------------|--------------------------|---------------------|------------------|
| L31-8B | EAGLE-3 | 5.53 | 2.91× | 6.63 | 6.12 | 6.09 |
| L31-8B | **PPOW** | **5.47** | **2.72×** | **7.23** | **6.50** | **6.40** |
| L33-70B | EAGLE-3 | 5.12 | 3.63× | 6.78 | 5.93 | 5.94 |
| L33-70B | **PPOW** | **5.45** | **3.73×** | **6.96** | **6.47** | **6.29** |
| Q3-8B | EAGLE-3 | 4.95 | 2.64× | 6.68 | 6.86 | 6.16 |
| Q3-8B | **PPOW** | **5.58** | **3.02×** | **7.01** | **6.97** | **6.52** |

> ✅ **总体趋势**：PPOW 在多个模型和任务上均取得最佳平均接受长度（6.29–6.52）和最高加速比（3.39–4.36×）。

### 与基线方法的对比结果
- **相比 EAGLE-3/GRIFFIN**：
  - 在 **HumanEval** 和 **GSM8K** 上提升显著（+0.5~1.0 $ \bar{T} $），说明在结构化任务中优势明显。
  - 在 MT-Bench 上略有波动，推测因开放对话允许多样续写，降低对 drafter 精准性的依赖。
- **相比 Continued Supervised Training (CST)**：
  - 如 Figure 3 所示，CST 初始阶段略有提升但很快下降；而 PPOW 持续稳定上升。
  - 表明单纯延长监督训练无法持续优化 speculative performance，甚至可能导致过拟合或分布偏移。

### 消融实验结果（Table 4）
| Method | MT-Bench ($ \bar{T} $) | GSM8K ($ \bar{T} $) |
|--------|-------------------------|----------------------|
| w/o Rdist | 5.05 | 6.41 |
| w/o ADAW | 4.82 | 6.35 |
| w/o both | 4.38 | 6.05 |
| **PPOW (Full)** | **5.47** | **6.50** |

> 🔍 发现：
> - 移除任一组件都会导致性能下降，尤其在 MT-Bench 上影响更大。
> - 同时移除两者造成最大衰减（↓1.09 $ \bar{T} $），证明两个模块互补。
> - **Rdist** 提供更密集反馈，**ADAW** 提升样本效率。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Token-level training is insufficient for speculative decoding**：局部模仿无法解决前缀截断问题，必须转向 window-level optimization。
2. **Performance-driven RL works**: 将 drafter 视为策略（policy），以 acceptance length 和 cost-aware speedup 为目标进行强化学习，能显著提升 speculative efficiency。
3. **Auxiliary rewards help**: 即使验证失败，只要生成的 window 与 target 偏好接近，也应给予鼓励（via Rdist）。
4. **Smart sampling matters**: 通过 ADAW 聚焦高 divergence 且高 confidence 的“硬窗口”，可大幅提升训练效率。
5. **PPOW is practical**: 在真实部署条件下（小 candidate group size），PPOW 用更少验证预算达到更高接受率（Table 2）。

### 方法的局限性
- **训练复杂度高**：需要冻结 target model、执行 speculative verification、计算 group-relative advantages，实现门槛高于监督训练。
- **超参数较多**：涉及 $ \gamma, \epsilon, \eta, K, \beta $ 等多个 speculative-specific 参数，需仔细调优。
- **依赖高质量初始化**：仍需先进行 supervised pre-training，不能完全从零开始训练。
- **硬件开销大**：PPOW 训练耗时约 50–200 GPU-hours（H100），不适合轻量级场景。

### 未来工作方向
- 设计 **self-tuning 或 adaptive 版本的 ADAW**，减少人工设定窗口优先级规则。
- 探索 **end-to-end joint optimization** of drafter and target model。
- 将 PPOW 思路扩展至其他推理时决策模块，如 **candidate allocation, scheduling, load balancing**。
- 研究如何在 **low-resource 或 mobile 设备** 上部署轻量化 PPOW drafter。

---

> 📌 **总结一句话**：  
> **PPOW 成功地将 drafter 的训练范式从“像 target 一样预测下一个 token”转变为“最大化 speculative decoding 的实际收益”，并通过 cost-aware reward、proximity signal 和 adaptive windowing 实现了显著的性能突破。**

</details>

---

### 5. [Know When To Fold 'Em: Token-Efficient LLM Synthetic Data Generation via Multi-Stage In-Flight Rejection](https://arxiv.org/abs/2605.14062)

**Authors**: Anjir Ahmed Chowdhury, Syed Zawad, Feng Yan  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.14062v1  

#### Abstract
While synthetic data generation with large language models (LLMs) is widely used in post-training pipelines, existing approaches typically generate full outputs before applying quality filters, leading to substantial token waste on samples that are ultimately discarded. To address this, we propose M...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Know When To Fold 'Em: Token-Efficient LLM Synthetic Data Generation via Multi-Stage In-Flight Rejection**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 LLM 合成数据生成方法通常采用“先生成完整输出，再进行质量过滤”的范式，导致大量 **token 浪费**在最终会被丢弃的低质量样本上。这种低效方式在大规模 post-training（如 SFT、RLHF）中造成显著计算开销。

### **提出的新方法：MSIFR**
作者提出了 **Multi-Stage In-Flight Rejection (MSIFR)**，一种轻量级、无需训练的框架，能够在生成过程中**分阶段地检测并终止低质量的生成轨迹**，从而避免浪费 token。

- **核心思想**：将生成过程分解为多个阶段（如问题生成、中间推理、完整解等），在每个阶段后插入轻量级规则验证器（rule-based validators），一旦发现错误即刻终止生成。
- **验证内容**：包括算术不一致、幻觉模式、格式违规等。

### **相比现有方法的优势**
| 对比维度 | 现有方法（如 LYNX, S-GRPO） | MSIFR |
|--------|--------------------------|------|
| **目标** | 减少冗余推理（early exit） | 阻止错误传播（in-flight rejection） |
| **触发机制** | 基于置信度或奖励信号 | 基于规则的即时验证 |
| **适用场景** | 单次推理任务 | 大规模合成数据构建 |
| **是否需训练** | 多数需要 probe 或 reward model | 完全 training-free |
| **效率增益来源** | 缩短正确但冗长的推理链 | 避免继续生成已知错误的轨迹 |

> ✅ **关键优势**：MSIFR 与 early-exit 方法正交且可组合，能实现**叠加式的效率提升**。

---

## **2. 核心实验方法和设置**

### **使用的模型**
在五个主流 instruction-tuned LLM 上进行评估：
- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct
- DeepSeek-LLM-7B-Chat
- Phi-3-mini-4k-instruct
- Mistral-7B-Instruct-v0.3

### **数据集（7个基准）**
涵盖数学推理与科学知识领域：
- **数学类**：GSM8K, MATH500, SVAMP, MAWPS, MathQA, DeepMind Mathematics Dataset
- **科学类**：MMLU-Chem

### **评估指标**
| 指标 | 描述 |
|-----|------|
| **Total Token ↓** | 总生成 token 数量（越低越好） |
| **Eval Accuracy ↑** | 经 LLM-as-a-judge（Llama-3-13B-Instruct）评分后的准确率（阈值 ≥3/5） |
| **Throughput ↑** | 每小时处理的样本对数量（problem-solution pairs/hour） |

### **基线方法对比**
- **Traditional (Full Generation)**：无 early discard，完整生成所有样本后再过滤。
- **DEER / LYNX**：代表性的 early-exit 方法，基于置信度提前结束推理。
- **MSIFR + LYNX**：组合方法，验证协同效应。

### **实验配置**
- 硬件：双 NVIDIA RTX 4090 GPU
- 推理引擎：vLLM（支持 PagedAttention 和 continuous batching）
- 批大小：64
- 温度：问题生成用 `T=0.7`（鼓励多样性），解生成用 `T=0.0`（确保确定性）
- 随机种子固定为 `seed=42`

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表1：MSIFR 在各模型和基准上的表现（部分摘要）**

| Model | Benchmark | Token Reduction | Accuracy Change |
|-------|-----------|------------------|------------------|
| Llama-3.1-8B | GSM8K | **↓42%** (7.46M vs 13.17M) | **↑+8.6pp** (0.638 vs 0.552) |
| Qwen2.5-7B | MathQA | ↓37% | ↑+18.1pp |
| Average across models/benchmarks | — | **11%–77% token reduction** | 多数情况下 accuracy 不降反升 |

> 🔥 **最高节省**：在 GSM8K 上达到 **77% token 节省**（MSIFR 单独使用）

#### **表2：与 early-exit 方法对比（GSM8K）**

| Method | Total Token ↓ | Throughput ↑ | Eval Accuracy ↑ |
|--------|----------------|---------------|--------------------|
| Traditional | 109.09M | 3,291 | 0.720 |
| LYNX | 25.14M | 8,588 | 0.738 |
| **MSIFR (Ours)** | **24.61M** | **15,823** | **0.734** |
| **MSIFR + LYNX** | **23.73M** (**↓78.2%**) | **16,605 (×5.0)** | **0.735** |

> 📌 **结论**：MSIFR 单独使用即可接近甚至超越 early-exit 方法的效率；与 LYNX 结合后实现 **78.2% token 减少 + 5× 吞吐提升**，且 accuracy 保持稳定。

### **消融实验结果**

#### **(1) 中间阶段截断点选择（Mid-Solution Cutoff）**
- 测试了从 30% 到 80% 的生成进度作为第二阶段检查点。
- 发现 **50% 是最优平衡点**：
  - 准确率保持峰值（0.57）
  - token 节省达 **42%**
  - 吞吐未下降

#### **(2) 错误分析（False Positive/Negative Rate）**
在 1,000 样本子集上进行 oracle 分析：

| 指标 | 平均值 |
|------|--------|
| **False Positive Rate (FPR)** | **3.2%**（极少误删高质量样本） |
| **False Negative Rate (FNR)** | **8.7%**（少量坏样本漏检，但仍比传统方法省 token） |

> 表明 MSIFR 的拒绝策略既**高效又可靠**，不会显著牺牲数据质量。

---

## **4. 关键结论和发现**

### **主要发现**
1. **早期拒绝显著降低 token 开销**：任何非平凡的 discard policy 都能严格减少期望 token 消耗（Proposition 3.1）。
2. **无偏性保证**：条件效用估计构成一个 **martingale**，证明 in-flight rejection 不会引入系统性偏差（Proposition 3.2）。
3. **准确性不降反升**：通过剔除低质量轨迹，保留下来的样本整体质量更高，多数情况下 **accuracy 提升最多达 +8.6 个百分点**。
4. **与 early-exit 正交互补**：MSIFR 作用于“防止错误扩散”，而 early-exit 优化“缩短正确推理”，二者结合可实现**复合增益**。

### **方法的局限性**
- **规则依赖性强**：validator 设计是任务相关的，需人工定义领域约束（如算术一致性、格式规范），难以完全自动化迁移至新任务。
- **阈值敏感性**：mid-solution cutoff 等参数需针对不同任务调优，尤其在极长文本（如形式化证明、代码生成）中可能失效。
- **模型规模限制**：当前实验集中在 7B–8B 参数模型，更大模型的行为尚待验证。

### **未来工作方向**
- 探索 **自动学习 validator 规则** 的方法，提升跨任务泛化能力。
- 将 MSIFR 扩展到 **code generation、multi-modal synthesis** 等更复杂场景。
- 研究 **动态调整 rejection threshold** 的机制，适应不同难度和长度的任务分布。
- 探索在 **online learning 或 active learning pipeline** 中集成 MSIFR，进一步提升数据利用效率。

---

> ✅ **总结一句话**：  
> **MSIFR 提供了一种简单、高效、无需训练的方式，在生成过程中“及时止损”，大幅降低 LLM 合成数据的成本，同时提升数据质量和下游性能，是构建高质量训练数据集的一项实用工具。**

</details>

---

### 6. [A Hardware-Aware, Per-Layer Methodology for Post-Training Quantization of Large Language Models](https://arxiv.org/abs/2605.14929)

**Authors**: Earl Killian  
**Category**: cs.LG  
**Published**: 2026-05-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.14929v1  

#### Abstract
Scaled Outer Product (SOP) is a post-training quantization methodology for large language model weights, designed to deliver near-lossless fidelity at 4.5--6 bits per weight on hardware with per-layer LUT decode. The methodology combines per-layer search of fixed and dynamic codebook pairs selected ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Scaled Outer Product: A Hardware-Aware, Per-Layer Methodology for Post-Training Quantization of Large Language Models》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**大语言模型（LLM）后训练量化（Post-Training Quantization, PTQ）中的精度-存储权衡问题**，旨在在极低比特（4.5–6 bpw）下实现接近无损的模型保真度，同时适配现代硬件架构（如支持每层LUT解码的加速器）。传统方法（如 per-layer POT scaling）在处理权重分布不均、存在离群值（outliers）时表现脆弱，难以兼顾精度与效率。

### 提出的新方法与创新思路
论文提出了 **Scaled Outer Product (SOP)**，一种**硬件感知、逐层优化的PTQ框架**，其核心创新包括：

- **灵活的块缩放（Flexible Block Scaling）**  
  在 $K$ 维度上以块为单位（block size $g \in \{8,16,32,64\}$）共享 scale 因子，结合高精度浮点 scale（如 12-bit S1E5M5），显著降低平均比特位宽（bpw），同时保留动态范围。

- **激活加权余弦相似度（Activation-Weighted Cosine Similarity, ACos）**  
  提出 ACos 作为新的保真度度量标准，优于传统的 MSE 或 SQNR。它通过通道范数（channel norms）对误差进行加权，更准确预测下游任务（如 perplexity）的表现。

- **双码本配对搜索（Per-Layer Pair Search）**  
  每一层独立搜索最优的两个 $n$-bit 码本（codebook）组合（如 NF4-DD4），并使用一个**每块元数据位（metabit）** 动态选择使用哪个码本重构当前块，提升表示能力。

- **多修正机制协同设计**  
  引入两种互补的后量化修正：
  - **Outlier Per-Quantum Extraction (OPQ)**：将高幅值权重提取为稀疏精确存储。
  - **Sparse Residual Correction (Wr)**：对剩余重建误差中激活加权大的部分进行稀疏补偿。

- **多选背包分配器（Multiple-Choice Knapsack Allocator, MCKP）**  
  在全局比特预算约束下，联合决策每一层是否进行格式提升（promotion）、应用何种修正策略，最大化整体 ACos。

- **硬件高效输出格式 HIF7/HIF8**  
  设计专用于 SOP 架构的低精度浮点网格（HIF7: 80 values, HIF8: 96 values），支持 shift-add 运算，降低硬件复杂度。

### 相比现有方法的优势
| 方面 | SOP 优势 |
|------|--------|
| **精度** | 在更低 bpw 下实现优于传统 FP8 的重建质量（甚至更低 MSE） |
| **灵活性** | 每层自适应选择码本、缩放格式、修正策略，无需全局统一规则 |
| **硬件友好性** | 利用 rank-1 outer product + Hadamard 积实现 GEMM，匹配低精度 MAC 阵列；支持 LUT SRAM 存储码本 |
| **可扩展性** | 支持从 4.5 bpw 起的精细精度调节，适用于不同部署场景 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准数据集（Calibration Corpus）**：小型文本语料（如 `c4`），用于计算各层的 **channel norms**（输入通道重要性向量）。
- **评估数据集**：`wikitext2`，用于测量最终模型的 **perplexity (PPL)** 和 **KL 散度**。
- **模型家族**：涵盖六个开源 LLM 家族：
  - Gemma-3-1B
  - SmolLM3-3B
  - Llama-3.2-3B
  - Qwen3.5-4B
  - Mistral-7B-v0.3
  - Qwen3-8B

### 实验设置与评估指标
| 设置项 | 描述 |
|-------|------|
| **块大小（block size）** | 默认 $g=16$ |
| **基础比特宽度** | $n=4$（即 4-bit codebook），目标 bpw ≈ 4.5–6.5 |
| **量化粒度** | 每层独立配置，支持 promotion 至 FP8/FP10/FP12 |
| **评估指标** | 
| • **Weight Reconstruction MSE**（未加权/加权）  
| • **ACos**（Activation-Weighted Cosine Similarity），以 ppm 表示与 1 的差距  
| • **Downstream KL Divergence / Perplexity**（间接验证）  

### 基线方法对比
- **主流 FP8 基线**：`E4M3^0sUE8M0` — 即每层单个 power-of-two scale 的 E4M3 权重，共 8.0 bpw。
- **其他对比**：
  - 不同 scale 格式（如 UE4M3 vs E4M3）
  - 是否启用 OPQ/Wr
  - 不同码本组合（如 NF4-only vs NF4-DD4）
  - 是否使用 MCKP 分配器

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **SOP 推荐操作点超越传统 FP8**
论文推荐的操作点为：  
👉 **`E2M3sUE4M4`（6.5 bpw）**

| 模型 | SOP (`E2M3sUE4M4`, 6.5 bpw) | Baseline (`E4M3^0sUE8M0`, 8.0 bpw) | 结果 |
|------|-------------------------------|------------------------------------|------|
| Gemma-3-1B | 3.41×10⁻¹ | 4.40×10⁻² | **↓ MSE** |
| SmolLM3-3B | 1.81×10⁻⁷ | 2.35×10⁻⁷ | **↓ MSE** |
| Mistral-7B | 5.36×10⁻⁹ | 6.89×10⁻⁹ | **↓ MSE** |

> 📌 **结论**：SOP 在 **低 1.5 bpw 存储成本下，实现了比传统 FP8 更优的权重重建精度**，证明 block-scaled small atoms 可替代常规 FP8。

#### ✅ **Scale Precision 在 12-bit 达到饱和**
- 所有模型在使用 **12-bit scale**（如 S1E5M5）后，MSE 与 16-bit BF16（E8M7）基准几乎一致（三位有效数字相同）。
- 表明进一步增加 scale 精度无法改善重建效果，**12-bit 是性价比最优选择**。

#### ✅ **HIF7 与 E2M3 几乎可互换**
- 在相同 scale 下，HIF7 与 E2M3 的 MSE 差距仅 **3.8–4.1%**。
- 尽管 HIF7 有 80 个值而 E2M3 仅有 63 个，但实际损失远小于理论预测（理论应差 ~61%）。
- 说明 **E2M3 能高效捕获 HIF7 中对 MSE 关键的部分**，适合用于公开研究。

#### ✅ **消融实验关键发现**
| 实验 | 发现 |
|------|------|
| **Metabit vs Sign Bit** | 8-bit scale 时二者互斥；12-bit（如 S1E5M5）可同时拥有 sign + metabit，带来约 25 ppm ACos 提升 |
| **ACos vs MSE 作为指标** | 在 promotion 决策中，ACos 与下游 KL 相关性更强，尤其在高动态范围层中优于 MSE |
| **OPQ + Wr 组合** | 两者正交：OPQ 处理 outliers，Wr 修复中等误差，叠加使用可进一步提升 ACos 5–15 ppm |
| **MCKP Allocator 效果** | 相比固定 promotion 策略，在相同 bpw 预算下能获得更高整体 ACos，实现帕累托最优 |

---

## 4. 关键结论和发现

### 主要发现
1. **block-scaled small atoms + high-precision scale 可击败传统 FP8**  
   推荐配置 `E2M3sUE4M4`（6.5 bpw）在更低存储开销下实现更优重建质量，打破“必须用 FP8”的固有认知。

2. **没有全局最优码本，逐层自适应是关键**  
   最优码本组合（如 NF4-DD4、SH4-DD4 等）因层而异，依赖于权重分布与部署格式，SOP 的 per-layer 搜索机制能充分利用这种多样性。

3. **ACos 是比 MSE 更优的 PTQ 指标**  
   激活加权使误差度量更具语义意义，能更好指导资源分配，尤其是在敏感层的 promotion 决策中。

4. **12-bit scale 是精度瓶颈突破口**  
   scale 精度在 12-bit 即达饱和，且 S1E5M5 提供 sign + metabit 共存能力，成为 SOP 的理想选择。

5. **HIF7 是硬件友好的高效 LUT 格式**  
   支持 shift-add 运算，避免通用浮点单元，面积与功耗更优，且与 E2M3 高度兼容。

### 方法的局限性
- **依赖专用硬件支持**：需具备 per-layer LUT SRAM、metabit 解析、rank-1 outer product 支持的矩阵单元。
- **校准数据需求**：虽无需标签，但仍需少量文本生成 channel norms。
- **实现复杂度较高**：涉及 pair search、MCKP 分配、多种修正路径，工程集成难度大于简单均匀量化。
- **当前聚焦 weight-only quantization**：未深入讨论 activation quantization 的联合优化。

### 未来工作方向
- **探索 $n=3$ bit codebook 的可行性**：目前尚无实用方案，可能需结合旋转预处理（如 Hadamard rotation）。
- **扩展至训练阶段（QAT）**：将 SOP 思路引入量化感知训练，探索更低比特下的训练稳定性。
- **动态元数据位扩展**：研究超过 1-bit 的 per-block metadata（如 via S1E5M4），支持更复杂的切换逻辑。
- **跨模态与多任务泛化**：验证 SOP 在非文本类模型（如视觉、多模态）上的有效性。
- **开源工具链建设**：提供完整的 SOP 编译器与运行时支持，推动落地应用。

--- 

> 🔚 **总结一句话**：  
> **SOP 通过“逐层双码本搜索 + 块缩放 + 激活加权指标 + 硬件定制格式”，在 4.5–6.5 bpw 实现了超越传统 FP8 的量化性能，为 LLM 高效部署提供了新的帕累托前沿方案。**

</details>

---

### 7. [APWA: A Distributed Architecture for Parallelizable Agentic Workflows](https://arxiv.org/abs/2605.15132)

**Authors**: Evan Rose, Tushin Mallick, Matthew D. Laws, Cristina Nita-Rotaru, Alina Oprea  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.15132v1  

#### Abstract
Autonomous multi-agent systems based on large language models (LLMs) have demonstrated remarkable abilities in independently solving complex tasks in a wide breadth of application domains. However, these systems hit critical reasoning, coordination, and computational scaling bottlenecks as the size ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：APWA: A Distributed Architecture for Parallelizable Agentic Workflows

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Models (LLMs)** 的多智能体系统在处理大规模、复杂任务时面临严重瓶颈，主要包括：
- **推理瓶颈**：随着任务规模增大，LLM 的输出质量和推理能力显著下降，尤其当输入数据超出其 context window 时。
- **协调瓶颈**：现有架构依赖中心化的 orchestrator 进行同步消息传递，导致可扩展性差，难以支持大规模并行执行。
- **计算瓶颈**：缺乏对高度并行化任务的有效分解与调度机制，无法充分利用分布式计算资源。

这些问题限制了多智能体系统在需要高吞吐量处理大量非重叠子任务（如大规模数据清洗、结构化提取、分层摘要）场景下的应用。

### 提出的新方法：APWA 架构
本文提出了 **Agent-Parallel Workload Architecture (APWA)** ——一种专为可并行化 **agentic workflows** 设计的分布式多智能体系统架构。其核心创新在于引入了一套面向 LLM 智能体的新型编程抽象，实现高效的任务分解与并行执行。

#### 核心设计思想
- **去中心化协调**：采用 **Manager-Worker-Executor** 分层架构，将全局控制与局部执行解耦。
- **无干扰并行化**：通过动态任务分解，将原任务拆分为多个**非相互依赖的 subtasks**，可在独立资源上并行执行而无需跨智能体通信。
- **数据感知规划**：引入 **data table 抽象** 和配套工具集，使 manager 能够以紧凑元数据形式推理超大规模分布式数据集。

#### 关键组件与抽象
| 组件 | 功能 |
|------|------|
| **Manager** | 负责高层任务规划、自动任务分解、subtask 模板生成、状态跟踪；拥有全局视图 |
| **Worker** | 执行具体 subtask；每个 worker 具有本地视图，高度自治 |
| **Executor** | 基于 **Ray** 实现的分布式执行引擎，负责 subtask 调度、容错重试、资源管理 |

#### 主要优势
相比现有方法（如 Autogen、Magentic-One、MegaAgent），APWA 的优势体现在：
- ✅ 支持**自动化、智能化任务分解**，适应异构数据与动态流程
- ✅ 实现**大规模并行执行**（实验中并发运行 >2.5k agents）
- ✅ 高效协调成百上千个 agents，避免中心化瓶颈
- ✅ 支持复杂、动态、数据依赖型的异构处理模式（data-parallel, task-parallel, replication-parallel）
- ✅ 通用性强，不局限于特定领域或预定义 SOP

---

## 2. 核心实验方法和设置

### 使用的数据集
论文在三个基准任务上进行了评估：

| 数据集 | 任务类型 | 描述 |
|--------|---------|------|
| **PII-300k** | 敏感信息脱敏 | 对 30 万条非结构化文本记录进行 PII 检测与红acting，涵盖教育、医疗等领域，共 27 类 PII |
| **SchemaBench** | 结构化内容提取 | 从多种格式（LaTeX, XML, CSV, HTML）文档中提取符合指定 schema 的 JSON 输出 |
| **SummaryBench** | 层次化摘要生成 | 对文学作品按层级结构（如 Scene → Act → Play）生成多粒度摘要，测试数据包括：<br>- *Romeo and Juliet* (166kB)<br>- *The Dynasts* (942kB)<br>- *Decline and Fall of the Roman Empire* (10.5MB) |

此外还进行了一个 **WebSurfer 实验**，测试 APWA 在真实 agentic 场景下（网页浏览+报告生成）的能力。

### 实验设置
- **硬件环境**：单机配置为 AMD Ryzen Threadripper PRO 5955WX + 252GB RAM + 2×RTX 4090 GPU
- **框架实现**：基于 **Ray** 构建分布式执行后端
- **LLM 后端**：使用 GPT-5.4 系列模型（包括 `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`）

### 评估指标
| 指标类别 | 子指标 | 定义 |
|--------|-------|------|
| **Utility** | Structural Score | 输出是否满足格式要求（如 JSON schema、表结构等） |
|          | Semantic Score | 内容正确性（如 F1、ROUGE-F1、人工评分） |
| **Cost** | Wall-Clock Runtime | 总耗时（秒） |
|         | Token Usage / Monetary Cost | 成本开销 |

### 基线方法对比
| 基线方法 | 特点 |
|--------|------|
| **Direct LLM** | 将全部输入直接提交给 LLM，利用 structured generation 输出结果 |
| **Magentic-One** | 基于 Autogen 的多智能体框架，使用 orchestrator-worker 架构，串行执行 |
| **MegaAgent** | 层级化多智能体系统，支持一定并行性，但仍需昂贵的 agent-modulated 协调 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table 1 & 2）

| 方法 | 输入规模 | Failure Rate ↓ | Runtime (s) ↓ | Structural ↑ | Semantic ↑ |
|------|----------|----------------|---------------|--------------|------------|
| **Direct** | PII-4096 / Roman | 100% | — | — | — |
| **Magentic-One** | 所有大任务 | ≥80% | — | 最高仅 0.179 | 几乎全失败 |
| **MegaAgent** | 中等以上任务 | 40–80% | 数百秒 | 最高 0.375 | 最高 0.023 |
| **APWA** | **PII-4096 / Roman** | **0%** | **221–329s** | **~0.90–1.00** | **~0.54–0.77** |

> 📌 注：APWA 在输入数据增长近两个数量级（166kB → 10,500kB）的情况下，runtime 仅从 157s 增至 329s，表现出良好的**亚线性扩展性**。

### 与基线方法的对比结果
- **成功率方面**：
  - Direct 方法在小任务上有效，但在大数据集上因 context overflow 完全失败。
  - Magentic-One 多数任务失败，主因是“orchestration failure”和“context explosion”。
  - MegaAgent 因协调机制低效且无法有效并行化，也频繁崩溃。
  - **APWA 在所有任务上均成功完成，failure rate 为 0%**。

- **效率方面**：
  - APWA 利用并行化实现了高达 **10–100× 的 subtask 吞吐提升**（得益于轻量级 LLM-only 执行路径）。
  - 在 SummaryBench 上，并发执行超过 **2,500 个 agents**。

- **质量方面**：
  - APWA 在 structural 和 semantic 指标上远超所有基线。
  - 在 SchemaBench 上，APWA 的语义得分与 direct LLM 相当（在可容纳范围内），而在更大数据集上则唯一可行。

### 消融实验结果（Table 3）
比较了不同 LLM 配置下的 APWA 表现：

| 配置 | Structural | Semantic | Runtime | Cost |
|------|-----------|----------|--------|------|
| `gpt-5.4 × gpt-5.4-mini` | 1.000 | 0.528 | 152s | $0.628 |
| `gpt-5.4 × gpt-5.4-nano` | 0.943 | 0.439 | 157s | $0.582 |
| `gpt-5.4-mini × gpt-5.4-mini` | 0.897 | 0.394 | **94s** | **$0.294** |
| `gpt-5.4-mini × gpt-5.4-nano` | 0.916 | 0.408 | **96s** | **$0.225** |

> 🔍 发现：使用更强的 `gpt-5.4` 进行 planning 可显著提升输出质量，而 worker 使用更小模型可在保持性能的同时大幅降低成本。

---

## 4. 关键结论和发现

### 主要发现
1. **APWA 能有效解决大规模 agentic 任务的可扩展性问题**，通过智能任务分解与并行执行，在传统方法完全失效的大数据场景下仍能稳定运行。
2. **去中心化 + 数据抽象的设计至关重要**：data table 抽象使得 manager 能在有限 context 下推理海量数据；executor 的自动重试机制提升了系统的鲁棒性。
3. **并行化显著改善响应时间与成本**：即使数据量激增，APWA 也能通过水平扩展维持合理延迟。
4. **APWA 是通用的 agentic 编程范式**：不仅能处理数据并行任务，还可灵活支持 web browsing、代码生成等多种 agentic 工作流。

### 方法的局限性
- ❌ **不支持 worker 间直接通信**：所有交互必须通过 manager，限制了需要子任务协作的场景。
- ❌ **未验证对其他并行模式的泛化能力**：目前仅测试了几种典型模式，是否适用于任意复杂拓扑尚待研究。
- ❌ **安全与隐私未考虑**：高自主性可能带来 prompt injection、恶意工具调用、数据泄露等风险。
- ❌ **依赖外部服务稳定性**：capability registry、object store 等若不可靠会影响整体可用性。

### 未来工作方向
- 探索支持 **worker-to-worker 通信** 的受限通道，以支持协同类任务。
- 扩展 APWA 以支持更多类型的 **动态并行模式** 和 **反馈驱动的工作流调整**。
- 引入 **安全沙箱机制** 和 **策略管控模块**，防止滥用与攻击。
- 在更大规模集群上部署，进一步测试其极限扩展能力。
- 探索 **human-in-the-loop** 机制，增强对高风险决策的可控性。

> 💡 **总体评价**：APWA 提供了一个类似 MapReduce 之于传统数据处理的“范式转移”，为构建可扩展、高性能的 agentic 应用提供了坚实基础，有望推动 LLM agents 在金融、医疗、科研等重负载领域的落地。

</details>

---

### 8. [Mistletoe: Stealthy Acceleration-Collapse Attacks on Speculative Decoding](https://arxiv.org/abs/2605.14005)

**Authors**: Shuoyang Sun, Chang Da, Hao Fang, Kuofeng Gao, Xinhao Zhong, Yi Sun, Fan Mo, Shu-Tao Xia, Bin Chen  
**Category**: cs.CL  
**Published**: 2026-05-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.14005v1  

#### Abstract
Speculative decoding has become a widely adopted technique for accelerating large language model (LLM) inference by drafting multiple candidate tokens and verifying them with a target model in parallel. Its efficiency, however, critically depends on the average accepted length $\tau$, i.e., how many...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Mistletoe: Stealthy Acceleration-Collapse Attacks on Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文揭示了 **Speculative Decoding**（推测性解码）机制中存在的一个**机制级安全漏洞**。尽管 Speculative Decoding 被广泛用于加速大语言模型（LLM）推理，其效率依赖于“平均接受长度” $ T $——即每次验证步骤中能成功通过目标模型校验的草稿 token 数量。

然而，作者指出：由于 **drafter 模型对 target model 分布的近似必然存在偏差**（drafter-target mismatch），这种微小差异可被恶意利用，构造一种**隐蔽攻击**，在不改变最终输出语义的前提下，显著降低 $ T $，从而导致加速效果“崩溃”（acceleration collapse）。

这一问题不同于传统的输出篡改或安全性攻击（如 jailbreak），而是针对**加速机制本身**的鲁棒性缺陷。

---

### 提出了什么新方法或新思路
作者提出了 **MISTLETOE** —— 一种**隐蔽的加速崩溃攻击方法**，其核心思想是：

- **攻击目标**：不是修改最终生成内容，而是破坏 `draft-then-verify` 流程中的 token 接受率。
- **攻击方式**：通过向输入 prompt 添加一个短的离散后缀（adversarial suffix），诱导 drafter 提出的 token 在 target model 下变得“意外”（high surprisal），从而被频繁拒绝。
- **关键技术**：
  - **Null-Space Projected Optimization**：将“降低接受率”的优化方向限制在不影响目标模型输出分布的局部零空间内，避免语义漂移。
  - **KL-Bounded Target Preservation**：引入 KL 散度约束，确保 adversarial prompt 下的目标模型输出分布与原始分布接近，保持输出质量。
  - **KL-Threshold Filtering**：在候选后缀选择时过滤掉引起过大分布偏移的选项，进一步保证隐蔽性。

---

### 相比现有方法的优势
| 维度 | MISTLETOE 的优势 |
|------|------------------|
| **攻击层面** | 首次提出针对 *acceleration mechanism* 的攻击，而非输出内容或隐私泄露 |
| **隐蔽性** | 输出 perplexity、语义一致性、任务正确性基本不变，难以被察觉 |
| **通用性** | 攻击后缀具有跨 decoding method 的**可迁移性**（transferability） |
| **有效性** | 显著降低 $ T $ 和 speed-up，使 Speculative Decoding 退化为普通自回归解码 |

> ✅ **创新本质**：将“模型近似误差”从效率瓶颈转化为**攻击面**，揭示了高效 LLM 推理系统的新风险维度。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在三个代表性基准上进行，覆盖多种生成场景：

| 数据集 | 类型 | 示例数量 | 特点 |
|-------|------|----------|------|
| **MT-Bench** | 开放式对话 | 80 个问题 | 多轮指令遵循、复杂交互 |
| **HumanEval** | 代码生成 | 随机采样 100 个 | 函数级程序合成 |
| **GSM8K** | 数学推理 | 随机采样 100 个 | 多步数学应用题求解 |

---

### 实验设置和评估指标

#### 模型与 Speculative Decoding 系统
- **Target Models**：Vicuna-7B、Vicuna-13B
- **Decoding 方法**：
  - Medusa
  - Hydra
  - EAGLE / EAGLE-2 / EAGLE-3

> 所有模型参数固定，仅优化附加的 adversarial suffix（长度 $ m=20 $）

#### 评估指标
| 指标 | 含义 | 期望变化（攻击成功） |
|------|------|---------------------|
| **Speed-up** | 相比 vanilla autoregressive decoding 的加速比 | ↓ 越低越好 |
| **Average Accepted Length $ T $** | 每次 target model 前向传播接受的 token 数 | ↓ 越低越好 |
| **Perplexity (PPL)** | 输出流畅性与自然度 | ≈ 不变（隐蔽性要求） |
| **Rep-4** | 连续重复 4-gram 的比例 | ≈ 不变 |
| **KL Drift** | adversarial vs clean 目标分布的 KL 散度 | ≤ 阈值 |

---

### 基线方法对比
本文无传统“基线攻击”，而是采用以下对照配置进行消融分析：

| 配置 | 是否使用 $ L_{\text{rej}} $ | 是否使用 $ L_{\text{sem}} $ | 是否使用投影优化 |
|------|----------------------------|------------------------------|------------------|
| Clean | × | × | × |
| $ L_{\text{rej}} $ only | √ | × | × |
| $ L_{\text{sem}} $ only | × | √ | × |
| Naive Joint | √ | √ | × |
| **Full MISTLETOE** | √ | √ | √ |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
在所有设置下，MISTLETOE 均造成显著加速崩溃：

| 指标 | 平均下降绝对值 | 平均相对降幅 |
|------|---------------|-------------|
| **Speed-up** | ↓1.89×~2.20× | **48.1%~51.7%** |
| **Accepted Length $ T $** | ↓0.99~1.13 | **27.2%~28.8%** |

> 🔺 最严重情况下（如 EAGLE-3 on HumanEval）：
> - Speed-up 从 **6.17× → 2.77×**（下降 55%）
> - $ T $ 从 **7.08 → 3.99**

---

### 与基线方法的对比结果
见 **Table 1 & Figure 3**

- 所有 decoding 方法均受影响，无论 chain-based（Medusa）还是 tree-based（EAGLE）。
- 加速越强的方法（如 EAGLE-3），攻击造成的**绝对损失越大**。
- 即使原本加速较弱的方法（如 Medusa），也出现明显退化（如 Vicuna-13B on MT-Bench: 3.26× → 1.48×）。

---

### 消融实验结果（Table 2）
| 配置 | Speed-up ↓ | $ T $ ↓ | PPL ↑ | Rep-4 ↓ |
|------|------------|---------|--------|--------|
| Clean | 5.47× | 5.95 | 2.5 | 0.1813 |
| $ L_{\text{rej}} $ only | 3.30× | 3.41 | **334.1** ❌ | 0.0844 |
| $ L_{\text{sem}} $ only | 4.47× | 4.73 | 213.2 | 0.0634 |
| Naive Joint | 3.73× | 4.13 | 196.6 | 0.0952 |
| **Full MISTLETOE** | **1.83×** ✅ | **2.79** ✅ | **49.2** ✅ | **0.0111** ✅ |

> ✅ 结论：
> - 仅用 $ L_{\text{rej}} $ 可降速但导致 PPL 暴涨（输出异常）
> - 投影优化 + KL 约束 是实现**高效且隐蔽攻击的关键**

---

### 可迁移性分析（Table 3）
在 EAGLE-3 上训练的 adversarial suffix 可迁移到其他方法：

| Target Method | MT-Bench Speed-up | HumanEval Speed-up |
|---------------|--------------------|---------------------|
| Medusa | 3.68× → **1.03×** |
| Hydra | 4.59× → **2.33×** |
| EAGLE | 3.57× → **1.97×** |
| EAGLE-2 | 4.44× → **2.03×** |

> 🔄 表明 MISTLETOE 利用了不同 decoding 方法之间的**共性脆弱性**：都依赖 drafter-target agreement。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Speculative Decoding 存在机制级安全威胁**：
   - drafter 与 target model 的微小不匹配可被放大为攻击面。
   - 攻击可在**不改变输出语义**的情况下摧毁加速效果。

2. **MISTLETOE 实现了高隐蔽性的加速崩溃**：
   - 通过 null-space projection 和 KL 约束，在降低 $ T $ 的同时维持 PPL 和输出正常性。
   - 攻击后缀短（20 tokens）、易部署。

3. **攻击具有跨方法泛化能力**：
   - adversarial suffix 在不同 decoding 架构间具备良好迁移性，说明问题具有普遍性。

4. **加速机制的鲁棒性亟需重视**：
   - 当前研究多关注输出安全或效率提升，忽视了“加速机制自身是否可信”。

---

### 方法的局限性（来自 Appendix C）
| 局限性 | 说明 |
|--------|------|
| **模型范围有限** | 仅在 Vicuna-7B/13B 上验证，未扩展至更大模型或闭源系统 |
| **白盒假设** | 攻击构造阶段需要访问梯度（white-box），对黑盒 API 场景适用性待研究 |
| **输出评估不够全面** | 使用 PPL 和 Rep-4 作为正常性指标，缺乏人工或 LLM judge 对语义等价性的精细评估 |
| **防御未深入探讨** | 提出监测建议，但未实现具体防御机制 |

---

### 未来工作方向
1. **黑盒攻击变体**：研究无需梯度访问的 query-efficient 或 zero-shot 攻击形式。
2. **防御机制设计**：
   - 动态监控 $ T $ 异常波动
   - 设计更鲁棒的 verification 规则
   - 对抗训练增强 drafter robustness
3. **扩展到其他加速技术**：如 Distillation、KV Cache Compression 等是否存在类似机制漏洞？
4. **真实服务环境测试**：在生产级 LLM serving pipeline 中评估攻击影响。

---

> 💡 **总体评价**：  
> MISTLETOE 是一篇具有高度洞察力的工作，它跳出了传统 LLM 安全研究范式，首次从“性能鲁棒性”角度揭示了现代推理加速系统的潜在风险。其提出的“零空间投影”优化策略也为后续对抗学习提供了新工具。该研究应引起 LLM 推理系统设计者对“加速可信性”（trustworthy acceleration）的高度重视。

</details>

---

### 9. [Precise Verification of Transformers through ReLU-Catalyzed Abstraction Refinement](https://arxiv.org/abs/2605.14294)

**Authors**: Hengjie Liu, Zhenya Zhang, Jianjun Zhao  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.14294v1  

#### Abstract
Formal verification of transformers has become increasingly important due to their widespread deployment in safety-critical applications. Compared to classic neural networks, the inferences of transformers involve highly complex computations, such as dot products in self-attention layers, rendering ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Precise Verification of Transformers through ReLU-Catalyzed Abstraction Refinement*

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Transformer** 形式化验证方法（如 [23]）依赖于对非线性操作（尤其是自注意力层中的 **dot product**）进行**凸松弛（convex relaxation）**，通常采用平面边界（planar bounds）进行过近似（over-approximation）。虽然这种方法效率较高，但由于近似误差较大，容易产生**假阳性告警（false alarms）**，即报告模型不鲁棒，而实际上模型是安全的。这严重损害了验证的**精度（precision）**。

### 提出的新方法和新思路
本文提出了一种名为 **BuFFeT**（**BoUnd Fusing-based reFinement for Transformers**）的抽象精炼框架，其核心创新在于**利用 ReLU 函数来表示和融合 dot product 的多个线性边界**。

- **双边界构造（Dual Bound Construction）**：作者指出，现有方法（如 [23]）仅使用一种平面边界来近似 `z = xy` 这类双线性操作。他们提出了一个**对偶边界（dual bound）**，从超双曲抛物面（hyperbolic paraboloid）的另一侧进行逼近。
- **ReLU 驱动的边界融合（ReLU-Catalyzed Fusion）**：直接取两个平面边界的最小值/最大值（min/max）可以得到更紧的边界，但这会引入非线性，导致计算不可扩展。作者的关键洞察是，这种 min/max 操作可以通过巧妙地引入 **ReLU** 函数来等价表示：
  ```math
  f_{\text{upper}} = f_1 - \text{ReLU}(f_1 - f_2)
  ```
  ```math
  f_{\text{lower}} = f_2 + \text{ReLU}(f_1 - f_2)
  ```
  这样，原本复杂的非线性边界被转化为由 ReLU 控制的表达式。
- **两种精炼策略**：基于上述表示，作者借鉴经典神经网络验证中对 ReLU 松弛的研究，提出了两种实现方式：
  1. **r-BuFFeT**：一种**基于规则（rule-based）**的方法。根据 ReLU 输入范围的统计特性（如正负区域大小），选择最优的固定斜率 `α`（0 或 1）来松弛 ReLU，从而决定使用哪个原始平面边界。
  2. **o-BuFFeT**：一种**基于优化（optimization-based）**的迭代方法。将 ReLU 边界中的斜率 `α` 视为可学习参数，构建一个以验证裕度（Margin）最大化为目标的优化问题，并使用梯度下降（如 Adam）迭代搜索最优的 `α`，直到成功验证。

### 相比现有方法的优势
- **更高的精度**：通过融合多个边界并利用成熟的 ReLU 松弛技术，显著减少了过近似误差，能够认证更大的扰动半径 `ε`，从而大幅降低假阳性率。
- **理论上的连接性**：该方法揭示了 Transformer 验证与经典前馈网络验证之间的深刻联系，特别是通过 ReLU 这一桥梁，使得大量已有的 ReLU 分析技术可以直接应用于 Transformer 验证。
- **灵活性**：o-BuFFeT 提供了一种“按需”精炼的机制，对于难以验证的任务，可以通过增加优化迭代次数来换取更高的精度。

## 2. 核心实验方法和设置

### 使用的数据集
实验在两个广泛使用的**情感分析（sentiment analysis）**数据集上进行：
- **SST (Stanford Sentiment Treebank)**：电影评论数据集，用于二分类（正面/负面）。
- **Yelp Polarity**：Yelp 评论数据集，用于二分类（正面/负面）。

### 实验设置和评估指标
- **模型架构**：
  - **标准 Transformer**：编码器-only 架构，层数 `N ∈ {1, 2, 3, 6}`。
  - **TinyBERT**：一个紧凑型预训练模型，作为实际应用的代表。
- **鲁棒性属性**：验证模型在输入词向量受到 `L1` 范数约束下的单个词扰动时，预测标签是否保持不变。
- **评估指标**：
  1. **Maximal verified ε**：能够被形式化证明为鲁棒的最大扰动半径 `ε`。这是衡量**精度**的核心指标，值越大越好。
  2. **Time costs**：完成一次验证任务所需的平均时间，用于衡量**效率**。
- **搜索方法**：使用**二分查找（binary search）** 来寻找最大的安全半径 `ε`。

### 基线方法对比
- **基线（Baseline）**：选择当前最先进的方法 **CrownBaF [23]** 作为主要对比对象。该方法是 BuFFeT 的基础，也是其直接改进目标。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **精度提升显著**：
  - 在绝大多数验证任务中，**r-BuFFeT** 和 **o-BuFFeT** 都优于基线方法。
  - **o-BuFFeT** 的表现最好，尤其是在深层模型（如 `N=6`）和 TinyBERT 上，其认证的 `ε` 显著高于基线和 r-BuFFeT。例如，在某些任务上，o-BuFFeT 的 `ε` 可以达到基线的 **2.7 倍**。
  - 图 4 和图 5 清晰地展示了这一优势，大部分数据点都位于 `y=x` 线之上。
- **效率对比**：
  - **r-BuFFeT**：效率与基线相当，时间开销略高（约 **1.1 到 1.7 倍**），这是一个可接受的代价。
  - **o-BuFFeT**：由于需要迭代优化，时间开销显著增加，约为基线的 **33.5 到 96.9 倍**。然而，作者指出，对于实际部署的模型（如 6 层 Transformer 和 TinyBERT），验证仍可在**几分钟内完成**，且开销的增长相对稳定，不随模型深度急剧恶化。

### 消融实验结果
- **o-BuFFeT 的有效性**：图 6 展示了 o-BuFFeT 在优化过程中的 **Margin** 变化。结果显示，随着优化的进行，Margin 持续增长，最终超过基线并成功验证了那些基线无法解决的困难任务。这直接证明了**动态调整 `α` 参数**的有效性和必要性。
- **互补性验证**：表 5 进行了一个水平对比实验，将 o-BuFFeT 与另一种针对 **softmax** 的先进松弛方法 **LSE [29]** 结合。结果显示，两者的结合效果优于任何单一方法，证明了 o-BuFFeT（精炼 dot product）与其他方法（精炼 softmax）具有**良好的互补性**。

## 4. 关键结论和发现

### 主要发现
1.  **ReLU 是强大的工具**：利用 ReLU 函数来表示和融合 Transformer 中的复杂边界是一种非常有效且新颖的思路，它极大地提升了验证的精度。
2.  **o-BuFFeT 更优**：在精度上，基于优化的 **o-BuFFeT** 显著优于基于规则的 **r-BuFFeT** 和基线方法，尤其擅长处理复杂和困难的验证任务。
3.  **精度与效率的权衡**：r-BuFFeT 提供了精度和效率的良好平衡；而 o-BuFFeT 以较高的时间成本换取了巨大的精度提升，这对于追求高可靠性的场景是值得的。
4.  **方法具有通用性**：该框架建立在 ReLU 松弛的基础上，因此可以无缝集成其他先进的 ReLU 分析技术，为未来的 Transformer 验证研究开辟了新的道路。

### 方法的局限性
- **计算开销大**：o-BuFFeT 的迭代优化过程带来了显著的时间成本，可能不适合需要快速验证的场景。
- **依赖于松弛质量**：最终的精度仍然受限于 ReLU 松弛本身的质量。如果 ReLU 的线性下界不够紧，那么整个链条的精度也会受限。
- **适用范围**：该方法主要针对由 ReLU 激活函数构成的 Transformer 模型。对于使用其他激活函数（如 GeLU）的模型，需要进一步适配。

### 未来工作方向
1.  **提高效率**：探索更高效的初始化策略（如用基线的结果初始化 `α`）或更智能的优化算法，以减少 o-BuFFeT 的迭代次数。
2.  **组合不同策略**：将 r-BuFFeT 和 o-BuFFeT 结合，例如先用 r-BuFFeT 快速筛选，再对困难任务使用 o-BuFFeT 进行精炼。
3.  **扩展到其他领域**：将 BuFFeT 应用于其他类型的 Transformer 模型，如用于计算机视觉的 **Vision Transformer (ViT)** 或用于控制系统的 Transformer 控制器。

</details>

---

### 10. [Factorization-Error-Free Discrete Diffusion Language Model via Speculative Decoding](https://arxiv.org/abs/2605.14305)

**Authors**: Xun Fang, Yunchen Li, Hang Yuan, Zhou Yu  
**Category**: cs.CL  
**Published**: 2026-05-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.14305v1  

#### Abstract
Discrete diffusion language models improve generation efficiency through parallel token prediction, but standard $X_0$ prediction methods introduce factorization errors by approximating the clean token posterior with independent token-wise distributions. This paper proposes Factorization-Error-Free ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Factorization-Error-Free Discrete Diffusion Language Model via Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Discrete Diffusion Language Models (DLLMs)** 在生成过程中通过并行预测多个 token 来提升效率，但通常采用独立的 token-wise 预测方式（如 Xo-prediction），这会引入**因子化误差（factorization error）**。该误差源于将真实的联合分布 $ p(X_o|X_t) $ 近似为各 token 独立分布的乘积，忽略了 token 之间的依赖关系，从而损害生成质量。

此外，虽然已有方法尝试缓解此问题（如 ReDi、DCD 等），但往往以牺牲推理速度为代价，或破坏 DLLM 的非自回归特性。

---

### 提出的新方法或新思路
本文提出 **Factorization-Error-Free Discrete Diffusion Language Modeling (FeF-DLLM)**，其核心思想是：

- **消除因子化误差**：不再独立预测每个 clean token，而是使用**前缀条件化的 clean-token 预测**（prefix-conditioned prediction），即建模：
  $$
  p_\theta(X_i^o | X_t, X_{<i}^o, t)
  $$
  这种分解方式遵循链式法则，保留了 token 间的左到右依赖结构，理论上可从真实后验分布中采样。

- **加速推理过程**：为了克服前缀条件化带来的串行解码瓶颈，引入 **speculative decoding** 机制：
  - 使用一个快速的 draft model 并行生成候选 token 序列；
  - 再由 prefix-conditioned 的 target model 从左至右验证这些候选；
  - 利用 accept-reject 机制确保最终输出仍服从目标分布；
  - 保持 DLLM 的并行预测与 re-masking 能力。

---

### 相比现有方法的优势
| 方面 | FeF-DLLM 的优势 |
|------|----------------|
| **生成质量** | 消除了因子化误差，显著提升任务准确率（平均 +5.04 pp） |
| **推理效率** | 引入 speculative decoding 后实现高达 **3.86× 的 wall-clock 加速** |
| **理论正确性** | 可证明生成样本来自真实的联合分布（distributionally exact） |
| **兼容性** | 不改变原始 diffusion forward process 和 reverse transition 形式，易于集成 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个标准 benchmark 上进行评估，涵盖数学推理与代码生成任务：
- **GSM8K**：小学数学应用题
- **MATH**：复杂数学问题
- **HumanEval**：Python 函数级代码生成
- **MBPP**（Mostly Basic Python Problems）：基础编程任务

---

### 实验设置和评估指标

#### 模型架构
- **主干模型**：基于 **LLaDA-Instruct** [Nie et al., 2025]，一种典型的 DLLM。
- **训练方式**：对 LLaDA 进行监督微调（supervised finetuning），共 72k prompt-response 对（code/math 各半）。
- **优化器**：AdamW，学习率 $1\times10^{-5}$，bf16 混合精度，全局 batch size 16。
- **硬件平台**：8 × NVIDIA A100 80GB GPU。

#### 推理配置
- **Speculative Decoding 设置**：
  - Draft model 与 Target model 使用相同 fine-tuned 模型；
  - Speculative window size = 16；
  - 每个 denoising step 支持多 token 并行 proposal。
- **Denoising Steps**：测试了 `step=2` 和 `step=4` 两种设定。

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 最终任务完成准确率（pass@1） |
| **Speedup** | 相对于 baseline 的 wall-clock 推理时间加速比（以 LLaDA-Instruct 为 1×） |

---

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **LLaDA** | 原始 DLLM | Xo-prediction，单步单 token 解码 |
| **LLaDA/2** | 修改版 | 单步解码两个 token，用于公平比较并行性影响 |
| **SSD** [Gao et al., 2025] | Speculative decoding | 自回归式 speculative，可能削弱非自回归优势 |
| **DCD** [Liu et al., 2024] | Copula-based correction | 改进采样策略，但速度慢（~0.4×） |
| **DDOSP** [Lavenant and Zanella, 2025] | Schedule optimization | 数据驱动 unmasking schedule，未解决因子化误差 |

> 所有 baseline 均在统一评估框架下复现，保证可比性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）
| 方法 | GSM8K Acc | MATH Acc | HumanEval Acc | MBPP Acc | **Mean Acc** | **Mean Speedup** |
|------|-----------|----------|---------------|----------|--------------|------------------|
| LLaDA | 78.60 | 26.60 | 47.60 | 34.20 | 46.75 | 1.00× |
| LLaDA/2 | 76.42 | 25.46 | 32.32 | 35.00 | 42.30 | 1.98× |
| SSD | 77.10 | 34.94 | 43.09 | 39.20 | 48.58 | 2.09× |
| DCD | 78.24 | 26.36 | 50.00 | 37.60 | 48.05 | 0.43× |
| **FeF-DLLM (step=2)** | **79.38** | **36.40** | **48.78** | **42.60** | **51.79** | **3.86×** |
| **FeF-DLLM (step=4)** | **79.68** | **36.56** | **49.39** | **42.60** | **52.06** | **2.33×** |

> ✅ **平均提升 5.04 个百分点准确率，最高达 3.86× 推理加速**

---

### 与基线方法的对比结果
- **vs. LLaDA/2**：尽管后者也有 ~2× 加速，但在多数任务上性能下降，说明单纯增加并行度会加剧 factorization error；而 FeF-DLLM 在大幅提升速度的同时持续提点。
- **vs. SSD**：FeF-DLLM 在 step=2 下比 SSD 多出 **3.21 pp** 准确率，且速度更快（3.86× vs 2.09×），表明其 speculative 设计更高效。
- **vs. DCD/DDOSP**：虽然后者也试图修正因子化误差，但严重拖慢推理（DCD 仅 0.43×）；FeF-DLLM 在保持高精度的同时实现大幅加速。

---

### 消融实验结果（Ablation Studies）

#### Ablation 1: Finetuning 的作用（Table 2）
- 若不进行 finetuning，直接使用原 LLaDA 模型执行 FeF 推理，性能提升有限；
- 经过 position-conditioned objective 微调后，性能显著增强 → 表明 **finetuning + 新 inference 策略 共同驱动性能提升**。

#### Ablation 2: Speculative Decoding 的加速效果（Table 3）
| 方法 | Mean Speed |
|------|------------|
| FeF-DLLM w/o SD (step=2) | 0.67× |
| FeF-DLLM (step=2) | **3.86×** |
→ speculative decoding 带来 **超过 5 倍的实际加速**，几乎恢复到接近纯并行解码的效率。

#### Ablation 3: Draft Model 选择（Table 4）
- 使用相同的模型作为 draft 和 verify model 时，acceptance rate 更高（如 MBPP 达 92.84%），speedup 更优；
- 不同模型组合（如未训练模型作 draft）acceptance 下降，验证了“draft-target alignment”对效率的重要性。

#### Ablation 4: Speculative Window Size（Table 5）
| Window Size | Mean Speed |
|-------------|------------|
| 4 | 0.78× |
| 8 | 1.42× |
| 16 | **2.33×** |
→ **窗口越大，加速越明显**，且 accuracy 完全不变，说明 speculative decoding 可扩展性强。

---

## 4. 关键结论和发现

### 主要发现
1. **因子化误差是 DLLM 性能瓶颈的关键因素**：简单扩大并行解码规模（如 LLaDA/2）会导致性能下降。
2. **精确的 prefix-conditioned factorization 可完全消除因子化误差**，并通过理论证明生成分布与真实后验一致（Theorem 1）。
3. **Speculative decoding 是高效的解决方案**：可在不牺牲分布正确性的前提下，极大缓解 prefix-conditioning 带来的串行开销。
4. **FeF-DLLM 实现了精度与效率的双重突破**：
   - 平均准确率提升 **+5.04 pp**
   - 推理速度达到 **3.86× 加速**
   - 显著优于各类先进 baseline

---

### 方法的局限性
- **计算资源需求更高**：由于 speculative verification 和 prefix-conditioned 输入构造，每次推理需要更多显存和计算量；
- **依赖高质量 draft model**：若 draft model 与 target 差异过大，acceptance rate 下降，导致加速效果减弱；
- **当前 window size 受限于设备内存**（最大试到 16），更大窗口潜力尚未完全释放。

---

### 未来工作方向
- 设计更轻量化的 draft model 或动态 window size 控制策略；
- 探索 resource-efficient implementation，降低部署成本；
- 将 FeF 框架推广至图像、语音等其他离散 diffusion 模型场景；
- 结合 adaptive schedule（如 DDOSP）进一步优化 denoising 路径。

--- 

> 📌 **一句话总结**：  
> FeF-DLLM 通过 **prefix-conditioned prediction + speculative decoding** 的设计，在理论上消除因子化误差、实践中实现高效推理，首次在 DLLMs 中同时达成“高精度”与“高速度”的统一。

</details>

---

### 11. [GraphBit: A Graph-based Agentic Framework for Non-Linear Agent Orchestration](https://arxiv.org/abs/2605.13848)

**Authors**: Yeahia Sarker, Md Rahmat Ullah, Musa Molla, Shafiq Joty  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.13848v1  

#### Abstract
Agentic LLM frameworks that rely on prompted orchestration, where the model itself determines workflow transitions, often suffer from hallucinated routing, infinite loops, and non-reproducible execution. We introduce GraphBit, an engine-orchestrated framework that defines workflows explicitly and de...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《GraphBit: A Graph-based Agentic Framework for Non-Linear Agent Orchestration》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

现有基于 LLM 的多智能体框架（如 LangChain、AutoGen、CrewAI）普遍采用 **prompted orchestration**（提示驱动编排），即由 LLM 自身决定工作流的执行路径。这种设计存在三大致命缺陷：

- **幻觉路由（Hallucinated Routing）**：LLM 可能调用不存在的 agent 或 tool，导致静默失败。
- **无限循环（Infinite Loops）**：agents 之间反复调用而无法终止。
- **非确定性执行（Non-reproducible Execution）**：相同输入可能产生不同执行轨迹，难以审计。

此外，随着任务复杂度增加，上下文不断累积，造成 **context bloat**，影响推理质量。

---

### 🚀 提出了什么新方法或新思路

提出 **GraphBit** —— 一种 **引擎驱动的图式智能体框架（engine-orchestrated agentic framework）**，其核心思想是：

- 将工作流定义为 **有向无环图（DAG）**，节点为 agent、tool 或 control 节点，边表示数据依赖与控制流。
- 所有路由决策由 **Rust 编写的执行引擎** 决定，而非 LLM。
- 引入 **三层内存架构（three-tier memory architecture）** 隔离上下文，防止污染。

#### 主要创新点：

1. **Graph-Native Execution**  
   工作流以显式 DAG 表达，支持并行分支执行、条件控制流，提升效率与可预测性。

2. **Engine-Governed Orchestration**  
   路由、状态转移、工具调用均由引擎控制，**从根本上消除幻觉与死循环**。

3. **Hierarchical Memory Isolation**  
   - **Tier 1: Ephemeral Scratch**：临时存储单个节点中间结果，完成后释放。
   - **Tier 2: Structured State**：全局结构化状态（KV 存储），原子更新，支持审计。
   - **Tier 3: External Connectors**：管理数据库/API/文件系统连接，结果需显式请求，避免自动注入导致上下文膨胀。

4. **高性能 Rust 执行核心 + Python 绑定**  
   利用 Rust 实现低延迟、高吞吐的调度引擎，同时保留 Python 的生态便利性。

---

### 🔍 相比现有方法的优势

| 优势维度 | GraphBit | Prompted Orchestration（如 LangChain, AutoGen） |
|--------|---------|---------------------------------------------|
| **可靠性** | ✅ 0% 幻觉率，确定性执行 | ❌ 高幻觉率（最高达 69.0%） |
| **效率** | ✅ 11.9ms 处理延迟，5,025 ops/min 吞吐 | ❌ 最高达 70.0ms，吞吐仅 1,599 ops/min |
| **可审计性** | ✅ 显式 DAG + 状态追踪 | ❌ 黑箱式自然语言决策 |
| **内存管理** | ✅ 三层隔离，防 context bloat | ❌ 上下文持续增长，token 消耗高 |
| **扩展性** | ✅ 支持并行、条件分支、错误恢复 | ❌ 多为串行或对话式，易陷入循环 |

---

## 2. 核心实验方法和设置

### 📚 数据集

使用 **GAIA benchmark**（Mialon et al., 2023）的一个精选子集：

- 总计 **68 个任务**（从原始 165 个中筛选）
- 排除所有框架均失败的任务，确保判别力
- 涵盖三种难度等级：
  - Level 1: 29 个简单单步任务
  - Level 2: 36 个中等多步推理任务
  - Level 3: 3 个复杂规划任务（需大量工具调用）

按 **workflow 类型** 分为三类：

| 类型 | 数量 | 描述 |
|------|------|------|
| **Zero-Tool** | 7 | 纯 LLM 推理，无需外部工具 |
| **Document-Augmented** | 19 | 需处理本地文件（PDF、Excel、图像等） |
| **Web-Enabled** | 42 | 需使用 Web Search 获取实时信息 |

---

### ⚙️ 实验设置

- **LLM 模型**：统一使用 **GPT-5.2**（闭源模型），temperature=1.0，max_tokens=2000
- **评估方式**：双验证机制
  - 字符串精确匹配（Exact String Matching）
  - 独立 LLM（GPT-5.2-chat）作为 evaluator 进行评分
- **运行限制**：所有框架最大迭代次数设为 `max_iter=3`

---

### 📊 评估指标

共报告六项指标：

| 指标 | 说明 |
|------|------|
| **Accuracy (%)** | 正确完成任务的比例 |
| **Hallucination Rate (%)** | 因框架错误（如路由错误、死循环）导致失败的比例 |
| **Processing Time (ms)** | 框架开销（不含 LLM API 延迟） |
| **CPU Utilization (%)** | 平均 CPU 占用 |
| **Peak Memory (MB)** | 峰值内存消耗 |
| **Throughput (ops/min)** | 每分钟可处理的操作数 |

---

### 🆚 基线方法对比

共比较 **7 种主流框架**：

1. **LangChain**
2. **LangGraph**（LangChain 的图扩展）
3. **CrewAI**
4. **Microsoft AutoGen**
5. **Pydantic AI**（强调类型安全）
6. **LlamaIndex**（聚焦检索增强生成）
7. **GraphBit**（本文方法）

所有框架配置等效 agent 逻辑，仅编排机制不同。

---

## 3. 主要实验结果和性能指标

### 📈 整体性能对比（Table 1）

| Framework | Acc.(%) | Hall.(%) | Proc.(ms) | Mem.(MB) | Thpt. (ops/min) |
|----------|---------|----------|------------|-----------|------------------|
| LangChain | 38.2 | 41.2 | 36.1 | 234.4 | ~1,660 |
| LangGraph | 36.8 | 47.1 | 31.5 | 208.0 | ~1,900 |
| CrewAI | 44.9 | 14.3 | 31.0 | 202.2 | ~1,930 |
| AutoGen | 35.3 | 33.8 | 70.0 | 274.8 | ~857 |
| Pydantic AI | 52.9 | 0.0 | 18.3 | 166.5 | ~3,270 |
| LlamaIndex | 50.0 | 0.0 | 15.0 | 165.4 | ~4,000 |
| **GraphBit** | **67.6** | **0.0** | **11.9** | **126.1** | **5,025** |

> ✅ GraphBit 在 **准确率、延迟、吞吐、内存、幻觉率** 全面领先

---

### 🔍 按任务类型拆解（Table 2）

| Framework | No-Tool Acc/Hall | Local Acc/Hall | Web Acc/Hall |
|----------|------------------|----------------|--------------|
| LangChain | 57.1 / 0.0 | 57.9 / 15.8 | 26.2 / 59.5 |
| LangGraph | 57.1 / 0.0 | 63.2 / 15.8 | 21.4 / **69.0** |
| Pydantic AI | 28.6 / 0.0 | 57.9 / 0.0 | 54.8 / 0.0 |
| **GraphBit** | **57.1 / 0.0** | **68.4 / 0.0** | **69.0 / 0.0** |

- **在 Web-enabled 任务上优势最显著**：69.0% 准确率 vs 第二名 54.8%
- **LangGraph 幻觉率达 69.0%**，超三分之二失败源于框架自身错误

---

### 🧪 按难度级别表现（Table 4）

| Framework | Level 1 | Level 2 | Level 3 |
|----------|--------|--------|--------|
| LangGraph | 48.3 | 27.8 | 0.0 |
| AutoGen | 58.6 | 38.9 | 0.0 |
| **GraphBit** | **79.3** | **63.9** | **66.7** |

- GraphBit 是唯一在 Level 3 仍保持高成功率的框架
- Prompted 框架随复杂度上升性能急剧下降（Pearson r ≈ -0.26~0.27, p<0.05）

---

### 🔬 消融实验（Ablation Study, Table 3）

验证三层内存架构的必要性：

| 配置 | Acc.(%) | Mem.(MB) | ΔAcc |
|------|--------|----------|-------|
| Full GraphBit | 67.6 | 126.1 | — |
| w/o ephemeral scratch | 64.7 | 189.2 | -2.9 |
| w/o structured state | 57.4 | 138.7 | **-10.2** |
| w/o external connectors | 60.3 | 130.4 | -7.3 |
| Single-tier baseline | 52.9 | 247.8 | -14.7 |

> 结论：
> - **structured state 最关键**（损失 10.2 pts）
> - **内存隔离对性能至关重要**

---

### 💡 其他关键效率指标

- **Token 效率**（Table 8）：
  - GraphBit：**1,916 tokens/task**
  - Pydantic AI：6,276
  - CrewAI：13,638
  - → GraphBit 节省 **3.3~7.1 倍 token**
- **初始化延迟**（Table 10）：
  - GraphBit 导入时间 2,400ms，setup 仅 0.1ms，适合 serverless 部署

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **框架级幻觉是当前多 agent 系统的主要瓶颈**  
   在 Web 任务中，LangGraph 高达 69.0% 的失败源于 **LLM 错误路由**，而非推理能力不足。

2. **确定性编排优于提示驱动编排**  
   GraphBit 通过将编排权交给引擎，实现了 **0% 幻觉 + 最高准确率**，证明无需 LLM 参与流程控制也能实现更强性能。

3. **内存隔离显著提升长期推理稳定性**  
   三层架构有效防止 context bloat，降低 token 消耗，提升推理一致性。

4. **Rust 引擎带来显著性能优势**  
   相比 Python 控制平面，Rust 实现的执行引擎延迟降低 **5.9×**，吞吐提升 **3×**。

5. **GraphBit 特别适用于工具密集型、高可靠性场景**  
   如金融、医疗、企业自动化等需要审计与复现性的领域。

---

### ⚠️ 局限性

1. **需要显式 DAG 定义**  
   不支持完全动态的任务分解，灵活性低于纯 LLM 驱动的方法。

2. **评估集中在单一 benchmark（GAIA）**  
   虽然任务多样，但 Level 3 仅含 3 个任务，统计意义有限。

3. **未测试框架特定优化策略**  
   所有框架使用相同 LLM 设置，可能低估某些框架潜力。

4. **缺乏人类参与的真实场景验证**  
   当前为全自动评估，尚未部署于真实人机协作环境。

---

### 🔮 未来工作方向

1. **探索混合模式（Hybrid Deterministic + LLM Routing）**  
   在部分节点允许 LLM 动态决策，其余保持确定性。

2. **支持动态 DAG 重构**  
   允许运行时根据反馈调整图结构，提升适应性。

3. **扩展至更大规模 benchmark 和真实应用**  
   如软件工程全流程、科研自动化等。

4. **集成 DSPy、LMQL 等 prompt 编译技术**  
   进一步提升 agent 内部可靠性。

5. **支持分布式执行与容错机制**  
   用于超长生命周期 pipeline。

---

> 🔗 **开源代码**：[github.com/InfinitiBit/graphbit](https://github.com/InfinitiBit/graphbit)

</details>

---

### 12. [OmniDrop: Layer-wise Token Pruning for Omni-modal LLMs via Query-Guidance](https://arxiv.org/abs/2605.14458)

**Authors**: Yeo Jeong Park, Hyemi Jang, Minseo Choi, Jongsun Lee, Jooyoung Choi, Yongkweon Jeon  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.14458v1  

#### Abstract
Omni-modal large language models have demonstrated remarkable potential in holistic multimodal understanding; however, the token explosion caused by high-resolution audio and video inputs remains a critical bottleneck for real-time applications and long-form reasoning. Existing omni-modal token comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# OmniDrop: Layer-wise Token Pruning for Omni-modal LLMs via Query-Guidance —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
**Omni-modal LLMs**（如 Qwen2.5-Omni）在处理高分辨率音频和视频输入时面临严重的 **token explosion** 问题。例如，一分钟的视频可生成超过 10k 的 audio 和 video tokens，导致计算和内存开销呈 **quadratic 增长**，严重影响实时性和长序列推理能力。

现有方法（如 OmniZip、DASH）通常在 **input embedding level** 进行 token 压缩，依赖于以下两种假设：
- 音频与视频 token 在嵌入空间中具有高相似性即语义对齐；
- 时间共现（temporal co-occurrence）意味着语义相关。

然而，作者通过实验证明这些假设不可靠：
- 输入层的 audio 和 video tokens 在 embedding 空间中占据不同子空间，跨模态对齐较弱；
- 时间上同时发生的音视频事件可能来自不同源（如背景音乐、画外音），并不一定语义相关。

因此，**基于 input-level 相似性的压缩策略可能导致重要信息丢失**。

---

### ✨ 提出的新方法：OmniDrop

OmniDrop 是一种 **training-free、layer-wise 的 token pruning 框架**，其核心思想是将 token 剪枝从输入层后移至 **LLM decoder 层内部**，并引入文本查询作为指导信号。

#### 主要创新点：

1. **Layer-wise Progressive Pruning（PLP）**
   - 不在输入阶段一次性剪枝，而是在 decoder 各层逐步剪枝。
   - 浅层保留更多 token 以支持多模态融合；
   - 深层根据 query 相关性激进剪枝。
   - 剪枝比例按 **sigmoid 函数调度**：  
     $$
     p_l = p_{\text{init}} + (p_{\text{final}} - p_{\text{init}}) \cdot \sigma(l, t_{\text{mid}}, \beta)
     $$

2. **Query-Guided Token Importance**
   - 使用 **text-to-audiovisual attention scores** 来衡量每个 audio/video token 的重要性。
   - 优势：
     - 支持任务自适应（task-adaptive）剪枝：音频任务保留更多 audio tokens，视觉任务反之；
     - 跨模态无关（modality-agnostic），无需显式定义模态关系。

3. **Temporal Diversity Score (TDS)**
   - 为防止剪枝集中在少数高注意力片段而导致全局上下文丢失，提出 TDS：
     - 识别当前层最重要的 chunk（`c_max`）；
     - 对候选低分 token 引入基于时间距离的奖励分数；
     - 鼓励保留时间分布更均衡的 tokens，避免“过度聚焦”。

4. **Intra-modality Pre-pruning**
   - 在进入 LLM 前先进行单模态压缩：
     - Audio：采用 OmniZip 的方式，保留 audio encoder 最后一层 attention 分数最高的 tokens；
     - Video：采用 Dycoke 的 Temporal Token Merging (TTM)，合并相似帧。

---

### 🔍 相比现有方法的优势

| 维度 | OmniZip / DASH | OmniDrop |
|------|----------------|----------|
| **Pruning Stage** | Input-level | **Layer-wise (decoder internal)** |
| **Guidance Signal** | Audio-video similarity / temporal boundary | **Text query attention** |
| **Task Adaptivity** | 固定模式，无法动态调整 | ✅ 查询驱动，自动适配任务需求 |
| **Temporal Coverage** | 易丢失远端上下文 | ✅ TDS 保障时间多样性 |
| **Training Requirement** | Training-free | ✅ Training-free |

> ✅ **关键突破**：首次实现 **query-aware、layer-adaptive、training-free** 的 omni-modal token pruning。

---

## 2. 核心实验方法和设置

### 📚 数据集

| 数据集 | 任务类型 | 描述 |
|--------|---------|------|
| **VideoMME** | 视频理解综合评测 | 包含多个子领域（Film & TV, Sports 等）的视频问答 |
| **WorldSense** | 多模态问答（需音视频联合理解） | 涵盖 Music, Tech & Sci., Daily Life 等真实场景 |
| **AVUT** | 音频为中心的视频理解 | 强调声音定位、语音识别等，避免文本捷径 |

---

### ⚙️ 实验设置

- **模型**：Qwen2.5-Omni-7B 和 Qwen2.5-Omni-3B（公开可用的 Omni-LLM）
- **硬件**：单张 NVIDIA H100 GPU
- **加速技术**：使用 FlashAttention 减少内存占用
- **token retention ratio**：报告各层平均保留率，便于与固定比率基线比较
- **Pruning Schedule 参数**：
  - $ p_{\text{init}} = 0.0 $, $ p_{\text{final}} = 0.2 $（20% 保留）
  - $ t_{\text{mid}} = 0.5 $, $ \beta = 20 $
  - $ \lambda_{\text{div}} = 0.2 $（TDS 权重）

---

### 🆚 基线方法对比

| 方法 | 类型 | 核心机制 |
|------|------|--------|
| **OmniZip [31]** | Training-free | Audio-guided compression，基于 audio-video cosine similarity |
| **DASH [18]** | Training-free | Structure-aware，利用音频边界分割视频流 |

> 所有方法均在相同预处理流程下测试，确保公平性。

---

## 3. 主要实验结果和性能指标

### 📊 性能汇总（Table 2）

| 方法 | Retained Ratio (%) | VideoMME ↑ | WorldSense ↑ | AVUT ↑ | Prefill Time ↓ | GPU Mem ↓ |
|------|--------------------|------------|--------------|--------|----------------|-----------|
| Full (7B) | 100 | 64.67 | 46.85 | 65.17 | 1.73s | 28.92GB |
| OmniZip | 30 | 65.85 | 45.55 | 61.76 | 1.06s | 25.76GB |
| DASH | 30 | 65.67 | 45.87 | 60.96 | 1.07s | 25.68GB |
| **OmniDrop (Ours)** | **30** | **66.52** | **46.60** | **64.01** | **1.05s** | **25.65GB** |
| **OmniDrop (Ours)** | **20** | **66.44** | **46.50** | **63.67** | **1.04s** | **25.65GB** |

> ✅ **关键结果**：
- 在 **30% 保留率下，全面超越所有 baseline**，甚至优于 full-token 模型（如 VideoMME 上达 66.52 vs 64.67）；
- 在极端 **20% 保留率下仍保持高性能**，AVUT 上比 OmniZip 高 **3.58 pts**（63.67 vs 60.09）；
- **Prefill latency 最多降低 40%**（1.73s → 1.04s）；
- **GPU memory usage 最多减少 14.7%**（28.92GB → 25.65GB）；

---

### 🔬 消融实验（Ablation Study）

#### （1）Progressive Layer-wise Pruning (PLP)

| 方法 | Ratio | WorldSense | AVUT |
|------|-------|------------|------|
| Intra-pruning only | 30 | 44.33 | 59.34 |
| PLP (Sigmoid) | 30 | 46.53 | 63.84 |
| + TDS | 30 | **46.60** | **64.01** |

> ➕ PLP 显著提升性能，说明 **layer-wise 剪枝优于 input-level 压缩**。

#### （2）Temporal Diversity Score (TDS)

| 方法 | Ratio | WorldSense | AVUT |
|------|-------|------------|------|
| PLP-Sig | 20 | 46.25 | 63.32 |
| + TDS | 20 | **46.50** | **63.67** |

> ✅ TDS 在高压缩比下增益更大，证明其对维持全局上下文至关重要。

#### （3）Guidance Type 对比（Table 4）

| 方法 | Ratio | WorldSense | AVUT |
|------|-------|------------|------|
| Audio guidance | 20 | 43.25 | 56.57 |
| **Text guidance (Ours)** | **20** | **46.19** | **60.55** |

> ❌ Audio-guided 注意力效果差，说明 **text query 是更可靠的 relevance 判断依据**。

---

### 🎯 Task-adaptive Token Retention 可视化（Fig. 4）

在不同任务下，OmniDrop 自动调节 audio/video token 保留比例：
- **Audio Recognition (AR)**：保留最多 audio tokens；
- **Scene Recognition (SR)**：保留最多 video tokens；
- **Audio Source Localization (ASL)**：平衡保留两类 tokens。

> ✅ 表明方法具备 **无需任务标签即可实现任务自适应的能力**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Input-level audio-video similarity 是弱信号**，不能可靠反映语义对齐；
2. **Audiovisual tokens 在 decoder layers 中逐渐融合**，中层以后才出现明显的 query-relevant token preference；
3. **Query-guided attention 是有效的 pruning signal**，支持任务自适应压缩；
4. **Temporal diversity 至关重要**，尤其在高压缩比下，TDS 显著缓解上下文丢失；
5. **Layer-wise pruning + query guidance 可解耦压缩与模型架构**，适用于任意 Omni-LLM。

---

### ⚠️ 局限性（Limitations）

1. **依赖 text query**：若仅有 audio-video 输入无文本提示，则无法应用；
2. **超参数经验设定**：pruning schedule 和 $ \lambda_{\text{div}} $ 等需手动调优；
3. **未考虑 encoder-side 动态交互**：目前仅在 decoder 内部剪枝，未来可探索双向优化。

---

### 🔮 未来工作方向

1. **Develop query-free variants**：探索直接建模 audio-video semantic relationship 的无监督方法；
2. **Learnable pruning policies**：使用 calibration data 学习最优 pruning schedule 和 hyperparameters；
3. **Extend to encoder-decoder pruning**：在 encoder 阶段也引入动态剪枝机制；
4. **Apply to streaming scenarios**：结合 chunked prefilling 实现实时流式压缩。

---

## ✅ 总结

OmniDrop 提出了一种新颖的 **training-free、layer-wise、query-guided** 的 Omni-modal token pruning 框架，解决了传统方法因依赖 input-level 相似性而导致的信息丢失问题。通过在 decoder 层内逐步剪枝，并以 text query 为指导信号，实现了 **任务自适应、高效且鲁棒的压缩**。

实验表明，OmniDrop 在多种 benchmark 上 **显著优于现有方法**，即使在仅保留 20% tokens 的极端条件下，仍能 **提升性能最多 3.58 pts**，同时 **降低 prefill latency 达 40%，节省 memory 14.7%**，为构建高效、可扩展的 Omni-LLM 提供了实用解决方案。

</details>

---

### 13. [Orchard: An Open-Source Agentic Modeling Framework](https://arxiv.org/abs/2605.15040)

**Authors**: Baolin Peng, Wenlin Yao, Qianhui Wu, Hao Cheng, Xiao Yu, Rui Yang, Tao Ge, Alessandrio Sordoni, Xingdi Yuan, Yelong Shen, Pengcheng He, Tong Zhang, Zhou Yu, Jianfeng Gao  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.15040v1  

#### Abstract
Agentic modeling aims to transform LLMs into autonomous agents capable of solving complex tasks through planning, reasoning, tool use, and multi-turn interaction with environments. Despite major investment, open research remains constrained by infrastructure and training gaps. Many high-performing s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Orchard: An Open-Source Agentic Modeling Framework

## 1. 论文的主要贡献和创新点

### 解决的问题
当前 **Agentic Modeling**（将大语言模型转变为能通过规划、推理、工具使用和多轮交互解决复杂任务的自主智能体）的研究面临基础设施和训练流程上的瓶颈。许多高性能系统依赖于专有代码库、模型或服务，而现有的开源框架则主要集中在 **agent orchestration**（代理编排）和 **harness design**（代理框架设计）上，缺乏对 LLMs 本身 **agentic capabilities**（代理能力）的改进。

具体而言，环境层（Environment Layer）成为了一个基础性的瓶颈：
- 当环境管理是封闭的或与特定训练栈紧密耦合时，其上的所有组件（如训练配方、评估管道）都会继承这些限制，导致无法独立复现和重用。
- 现有的解决方案各有权衡：托管平台（如 E2B, Daytona）提供了便利但牺牲了控制权；垂直集成的训练栈（如 ProRL Agent, MegaFlow）虽然强大，但将环境管理与推理调度、奖励计算等深度绑定，缺乏灵活性。

### 提出的新方法和新思路
本文提出了 **Orchard**，一个以**薄、开放、与框架无关的环境服务**为核心的开源 **Agentic Modeling** 框架。

其核心创新点在于 **Orchard Env**：
- **薄且独立的服务边界 (Thin, Standalone Service Boundary)**：Orchard Env 是一个独立的 Kubernetes 原生服务，仅提供沙箱生命周期管理、命令执行、文件 I/O 和网络策略等通用原语，并通过 REST API 暴露。它与任何特定的 **agent harness**、模型服务或训练编排器完全解耦。
- **支持异构任务环境 (Heterogeneous Environments)**：通过 **Runtime Agent Injection** 技术，Orchard Env 可以在运行时将一个轻量级的执行代理注入到任意的用户提供的 Docker 镜像中，无需为每个任务镜像重新构建，极大地降低了适配成本。
- **高性价比和可扩展性 (Accessible and Cost-Practical at Scale)**：基于标准的 Kubernetes 构建，可以利用集群自动伸缩（cluster autoscaling）和竞价实例（spot instances）等云原生优化，显著降低大规模训练的成本。

在此之上，Orchard 提供了三个可组合的 **Agentic Modeling** 配方（recipes），分别针对软件工程、GUI 导航和个人助理任务。

### 相比现有方法的优势
| 特性 | Orchard Env | 托管平台 (E2B, Daytona) | 垂直集成栈 (ProRL Agent) | 广泛框架 (ROCK) |
| :--- | :--- | :--- | :--- | :--- |
| **开源自托管** | ✅ | ❌ (主要是托管服务) | ✅ | ✅ |
| **薄环境服务** | ✅ | ✅ | ❌ (与harness耦合) | ❌ (功能更广泛) |
| **与框架无关** | ✅ | ✅ | ❌ | ✅ |
| **低成本** | ✅ (低至0.10×) | ❌ (1.0×) | ⚠️ (估计0.150) | ⚠️ (未公开) |

Orchard 的优势在于它创造了一个**可重用的基础层**，使得轨迹数据、训练配方和评估协议可以在不同领域、不同框架和不同训练阶段之间自由迁移。

## 2. 核心实验方法和设置

### 使用的数据集
Orchard 在三个不同的领域进行了验证，使用了以下数据集：
- **软件工程 (SWE)**：
  - **SWE-rebench** 和 **SWE-rebench V2**：来自真实 GitHub issues 的大规模任务集合。
  - **Scale-SWE**：从 5.2k 个仓库的 GitHub PRs 中构建的 100k 个任务实例。
  - **评估基准**：**SWE-bench Verified** (500个实例)，以及 SWE-bench Multilingual 和 Terminal-Bench 2.0。
- **GUI 导航 (GUI)**：
  - **任务来源**：从 **WebGym** 的 29万多个原始任务中，经过五步过滤（去重、去评测集、保留热门网站等）得到 15,601 个独特的种子任务。
  - **评估基准**：**WebVoyager**, **Online-Mind2Web**, 和 **DeepShop**。
- **个人助理 (Claw)**：
  - **任务来源**：使用 **Claude Opus 4.6** 合成的 192 个任务，来源于 **Claw-Eval** 种子和 **ClawHub** 工作流。
  - **评估基准**：**Claw-Eval**。

### 实验设置和评估指标
- **环境**：所有实验均在 Orchard Env 提供的 Kubernetes 原生沙箱环境中进行。
- **训练流程**：采用两阶段训练：**Supervised Fine-Tuning (SFT)** + **Reinforcement Learning (RL)**。
- **评估指标**：
  - **SWE**：在 SWE-bench Verified 上的 **resolve rate**（解决率）。
  - **GUI**：在各基准上的 **success rate**（成功率）。
  - **Claw**：在 Claw-Eval 上的 **pass@3**（三次尝试内成功）。

### 基线方法对比
论文将 Orchard 的成果与多种基线进行了比较：
- **开源方法**：如 OpenSWE-32B/72B, SWE-Master-32B, CoderForge-32B, MolmoWeb-4B/8B, Fara-7B 等。
- **闭源/专有系统**：如 GPT-4o, GPT-5, Claude Opus, Gemini 等。
- **消融实验**：通过移除或替换关键组件（如 credit-assignment SFT, BAR, 多框架训练）来验证其有效性。

## 3. 主要实验结果和性能指标

### 关键性能数据
| 项目 | 性能指标 | 结果 |
| :--- | :--- | :--- |
| **Orchard-SWE** | SWE-bench Verified (SFT) | **64.3%** |
| | SWE-bench Verified (SFT+RL) | **67.5%** |
| **Orchard-GUI** | WebVoyager 成功率 | **74.1%** |
| | Online-Mind2Web 成功率 | **67.0%** |
| | DeepShop 成功率 | **64.0%** |
| | **平均成功率** | **68.4%** |
| **Orchard-Claw** | Claw-Eval (pass@3) | **59.6%** |
| | Claw-Eval (pass@3, +ZeroClaw) | **73.9%** |

### 与基线方法的对比结果
- **Orchard-SWE**：
  - 在 **SWE-bench Verified** 上，67.5% 的成绩超过了所有同规模（~3B 激活参数）的开源模型，甚至优于一些更大的密集模型（如 OpenSWE-72B 的 66.0%）。
  - 其性能接近于大 10-30 倍的 **MoE** 前沿系统。
- **Orchard-GUI**：
  - 以 **68.4%** 的平均成功率，成为最强的开源 GUI 代理。
  - 其性能不仅远超其他开源模型（如 MolmoWeb-8B 的 51.9%），而且与专有的 **Gemini computer-use-preview (69.3%)** 和 **GPT-5 SoM (65.8%)** 相当。
  - 令人惊讶的是，它在一个 4B 的骨干模型上，仅用 2.6k 训练任务就超越了其 235B 的教师模型。
- **Orchard-Claw**：
  - 在仅使用 0.2k 合成任务的情况下，达到了 59.6% 的 pass@3，显著优于同等规模的开源基线。
  - 当与更强的 **ZeroClaw** 框架结合时，性能进一步提升至 **73.9%**，证明了其跨框架的适应能力。

### 消融实验结果
- **Credit-Assignment SFT**：通过从失败轨迹中提取“上升段”（rise segments）作为监督信号，在控制变量下，相比仅使用成功轨迹的 SFT，带来了 **+1.9%** 的性能提升，验证了该方法的有效性。
- **Balanced Adaptive Rollout (BAR)**：该 RL 算法通过动态调整 rollout 数量来确保每个训练组都有平衡的正负样本，避免了无效计算，提高了梯度的信息密度。
- **跨框架泛化**：实验表明，单一框架训练的模型（如 Scale-SWE, OpenSWE-32B）在切换到未见过的框架时性能急剧下降（甚至格式错误）。而 Orchard-SWE 由于在训练中使用了 **OpenHands** 和 **mini-swe-agent** 两种框架，其性能在不同框架间波动很小，证明了多框架训练对于提升泛化能力至关重要。

## 4. 关键结论和发现

### 主要发现
1.  **环境层是可重用性的基石**：论文的核心论断是，**环境层** 不仅仅是一个基础设施组件，而是决定 **agentic modeling** 成果可重用性的“基底”。一个薄、开放、与框架无关的环境服务（如 Orchard Env）能够使轨迹数据、SFT 配方、RL rollout 和评估协议在不同领域、框架和阶段之间自由迁移。
2.  **Orchard 框架的有效性**：通过在软件工程、GUI 导航和个人助理三个截然不同的领域应用相同的框架，Orchard 展示了其强大的通用性和可扩展性，并取得了 SOTA 或接近 SOTA 的性能。
3.  **多样性带来鲁棒性**：无论是多教师、多任务源还是多框架的训练，数据和训练过程的多样性是模型具备良好跨框架、跨任务泛化能力的关键。

### 方法的局限性
- **依赖高质量的教师模型**：Orchard-SWE 和 Orchard-GUI 的成功很大程度上依赖于强大的教师模型（如 MiniMax-M2.5, Qwen3.5-397B）生成的高质量轨迹。
- **合成任务的成本**：Orchard-Claw 使用了合成任务，这需要调用昂贵的闭源模型（如 Claude Opus），单个任务成本高达 4.9 美元，难以大规模复制。
- **通用性尚未完全证明**：虽然在三个领域成功，但该框架是否能无缝迁移到更多样化的领域（如机器人控制、科学发现）仍有待验证。

### 未来工作方向
- **降低对教师模型的依赖**：探索更高效的自我迭代（self-improvement）或无监督学习方法，减少对强教师模型的依赖。
- **自动化任务合成**：开发更经济、更自动化的任务合成流水线，以扩大个人助理等领域的训练数据规模。
- **社区驱动的发展**：作者已将整个框架（包括环境服务、训练配方和轨迹数据集）开源，旨在推动开源 AI 社区在可扩展的 **Agentic Modeling** 方向上的创新。未来的工作可能由社区共同推动，探索新的领域和更先进的训练技术。

</details>

---

### 14. [Conditional Attribute Estimation with Autoregressive Sequence Models](https://arxiv.org/abs/2605.14004)

**Authors**: Erica Stutz, Giacomo Marino, Daniella Meeker, Qiao Liu, Andrew J. Loza  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.14004v1  

#### Abstract
Generative models are often trained with a next-token prediction objective, yet many downstream applications require the ability to estimate or control sequence-level properties. Next-token prediction can lead to overfitting of local patterns during training, underfitting of global structure, and re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Conditional Attribute Estimation with Autoregressive Sequence Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统生成模型（如LLMs）通常采用**next-token prediction**作为训练目标，这虽然可扩展且高效，但存在以下关键缺陷：
- 过度拟合局部模式，忽视全局序列属性（sequence-level attributes）；
- 难以在推理时控制或预测序列级特征（如情感倾向、医疗风险等）；
- 若需估计属性，常依赖昂贵的**Monte Carlo (MC) sampling**，计算开销大。

许多下游任务（如可控文本生成、临床事件预测）需要对序列属性进行高效估计与引导，而现有方法在效率、灵活性和准确性之间难以平衡。

---

### 提出了什么新方法或新思路
本文提出 **Conditional Attribute Transformers (CAT)**，一种联合建模**next-token概率**与**序列属性条件概率**的新框架。其核心思想是：

> 在每一步 token 预测的同时，估计该选择下整个序列最终属性的条件分布 $P(a|S_a, s_n)$。

#### 创新机制包括：
- **统一架构设计**：共享 Transformer backbone，在末层分支出两个头：
  - **Token Head**：标准语言建模头，用于 $P(s_n | S_a)$；
  - **Attribute Head**：新增模块，输出 $P(a | S_a, s_n)$。
- **单次前向传播实现多能力**：
  1. **Credit Assignment**：识别每个 token 对最终属性的影响；
  2. **Counterfactual Analysis**：量化不同 token 选择下的属性变化；
  3. **Steerable Generation**：结合 $P(s_n)$ 和 $P(a|S_a,s_n)$ 进行可控解码。

---

### 相比现有方法的优势
| 方法类别 | 代表模型 | 局限性 | CAT 的优势 |
|--------|--------|-------|-----------|
| Conditioning-based | CTRL, Quark | 需修改输入prompt；无法动态调整；不提供概率解释 | 不修改输入；每步实时估计属性；支持反事实分析 |
| Auxiliary Model Guidance | PPLM, FUDGE, DExperts, Director | 需额外训练辅助模型；高计算成本；表达能力受限 | 单模型集成；无需额外模型；保留完整Transformer表达力 |
| Sampling-based | MC Rollout | 推理慢，复杂度随长度指数增长 | 属性估计**加速约105倍**于MC方法 |

此外，CAT 可通过微调应用于预训练模型，也可在预训练阶段联合优化，具有高度适应性。

---

## 2. 核心实验方法和设置

### 使用的数据集
论文在三个差异显著的任务上验证 CAT 的通用性：

| 数据集 | 类型 | 序列内容 | 属性类型 | 用途 |
|------|-----|---------|--------|------|
| **Key-to-Door** | 强化学习模拟环境 | 动作轨迹（pick key, move, open door） | 二元胜负结果（win/loss） | 测试稀疏奖励下的长期信用分配 |
| **Amazon Reviews** | 自然语言 | 商品类别 + 标题 + 评论文本 | 多分类评分（1–5星） | 测试大规模动作空间下的语义控制 |
| **PhysioNet Sepsis** | 医疗时间序列 | 患者人口统计 + 每小时生命体征 + 实验室值 | 二元 sepsis 发生 / 回归最大心率 | 测试真实世界高维异构数据中的临床预测 |

---

### 实验设置和评估指标

#### 统一训练目标：
$$
\mathcal{L} = \mathcal{L}_{\text{token}} + \lambda \cdot \mathcal{L}_{\text{attr}}
$$
其中 $\mathcal{L}_{\text{token}}$ 为交叉熵损失，$\mathcal{L}_{\text{attr}}$ 根据属性类型选用：
- 分类任务：Cross-Entropy
- 回归任务：Gaussian Negative Log-Likelihood

#### 评估维度：
1. **Next-token Prediction Performance**：验证 perplexity
2. **Attribute Estimation Accuracy**：
   - 分类：Top-1 准确率、AUC、AP
   - 回归：MAE、RMSE
3. **Guided Decoding Performance**：
   - 控制生成准确率（steering accuracy）
   - 生成流畅性（perplexity）
   - 多样性（Dist-n）
4. **效率对比**：运行时间 vs. MC sampling
5. **Counterfactual Sensitivity**：替换关键词后属性概率的变化是否符合语义直觉

---

### 基线方法对比
| 基线方法 | 类别 | 描述 |
|--------|-----|------|
| **Random Policy / Behavioral Cloning** | RL baseline | 仅基于行为模仿 |
| **Decision Transformers** | RL via seq modeling | 使用 reward-to-go token 条件生成 |
| **CTRL** | Prompt-based control | 添加控制码 |
| **DExperts** | Expert/Anti-expert weighting | 利用两个微调模型重加权 |
| **Director / Director*** | Dual-head model | 类似 CAT 架构，但 attribute head 更简单 |
| **MC Simulation (GPT/CAT)** | Sampling-based | 对剩余序列采样多次估算属性期望 |

---

## 3. 主要实验结果和性能指标

### Key-to-Door: 长期信用分配任务
| 方法 | 胜率（Win Rate） |
|------|----------------|
| Random Policy | 0.031 |
| Behavioral Cloning | 0.016 |
| Percentile BC | 0.951 |
| Conservative Q-Learning | 0.133 |
| Decision Transformers | 0.946 |
| **CAT (Ours)** | **0.999** ✅ |

- CAT 实现接近完美胜率，并在 998/999 场胜利中走出最短路径；
- 同时能稳定估计每一步的获胜概率（variance 更低），优于 DT。

---

### Amazon Reviews: 语言建模任务

#### (1) Next-Token Prediction 性能
- 小模型（7M–270M）：joint training 略有负面影响；
- **1B 参数模型**：**CAT 的 perplexity 显著低于标准 GPT**，表明大模型下任务协同增益；
- 最优 $\lambda=0.15$ 平衡两项任务。

> 📈 图3显示：随着模型增大，CAT 在 token 预测上超越纯 GPT，体现“学习全局结构有助于提升局部预测”。

#### (2) Partial Review Rating Prediction（属性估计）
| 方法 | Top-1 准确率（平均） | 相对速度 |
|------|--------------------|--------|
| MC + Standard GPT | ~0.72 | 1×（基准） |
| MC + CAT Token Head | ~0.76 | 相当 |
| **CAT (direct)** | **~0.79** | **≈105× faster** |
| Director* | ~0.78 | 快 |

✅ CAT 在无需采样的情况下达到最佳性能，且**提速两个数量级**。

#### (3) Counterfactual Estimation（反事实分析）
- 替换 `good` → `amazing/horrible`：
  - 正面词显著↑5星概率，↓1星概率；
  - 负面词显著↑1星概率，↓5星概率；
- 在否定上下文（如 `not good`）中：
  - `not good` ≈ `bad`，提升1星概率；
  - `not bad` → 中性评价（主导为3星），合理反映“还行”含义；
- 大写增强语义强度（如 `HORRIBLE` 比 `horrible` 更强）。

👉 表明 CAT 学到了细粒度语义逻辑。

#### (4) Guided Decoding（可控生成）
从3星评论前半部分引导生成1星或5星完整评论：

| 方法 | 1-star 准确率 | 5-star 准确率 | 完成 perplexity ↓ |
|------|---------------|---------------|------------------|
| Director* | 0.58 | 0.65 | 46.77 / 48.16 |
| **CAT (ours)** | **0.64** | **0.77** | **45.88 / 44.03** ✅ |

✅ CAT 在**更高准确率**的同时保持**更低困惑度（更流畅）**，多样性也相当。

---

### PhysioNet Sepsis: 医疗预测任务

#### (1) Sepsis Occurrence Prediction（二元分类）
| 方法 | ROC-AUC | AP（Average Precision） |
|------|--------|------------------------|
| GPT + MC (n=64) | 0.782 | 0.271 |
| Director | 0.688 | 0.277 |
| **CAT** | **0.757** | **0.448** ✅ |

- 尽管 AUC 略低于 MC-GPT，但 **AP 提升巨大（+65%）**，说明在**高度不平衡场景下更精准捕捉正例**，更具临床意义。

#### (2) Maximum Heart Rate Prediction（回归任务）
- CAT 成功预测未来6小时内的最大心率趋势；
- 动态滚动预测与真实值高度一致（见图5C）；
- 可视化 token-level attribution 发现：低 DBP 后出现高 MAP 会显著↑sepsis 风险，可能反映数据中血压测量不一致性（见图A.4）。

#### (3) Counterfactual Vital Sign Analysis
- 升高体温 → ↑sepsis 概率；
- 老年患者（71–87岁）对高温更敏感，风险增幅更大，符合医学常识（thermal dysregulation vulnerability）。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **联合建模 next-token 与 conditional attribute 可实现多重功能一体化**：
   - credit assignment、counterfactual analysis、steerable generation 均可在一次前向传播中完成。
2. ✅ **大模型下，全局属性学习可反哺局部 token 预测**：
   - 在1B参数级别，CAT 的 perplexity 超过标准 GPT，挑战“多任务损害主任务”的传统认知。
3. ✅ **属性估计效率远超 sampling-based 方法**：
   - 达到与 MC 相当甚至更好的性能，但速度快 **105倍以上**，适用于实时系统（如ICU预警）。
4. ✅ **反事实分析具备语义合理性与临床可解释性**：
   - 支持精细化归因分析，可用于调试模型决策逻辑或辅助医生理解风险因素。

---

### 方法的局限性
1. ❌ **仅适用于离散 action space**：
   - 当前只能处理 discrete token 输出，难以直接扩展到 continuous control（如机器人动作）。
2. ❌ **当前策略为单步贪婪优化**：
   - Eq. (10) 中的 decoding 是 greedy over $P(a|S_a,s_n)$，未考虑长期策略规划，非全局最优。
3. ❌ **attribute head 设计仍较轻量**：
   - 虽优于线性层（如 Director），但仍不如独立大型辅助模型灵活（但在效率上碾压）。

---

### 未来工作方向
1. 🔮 扩展至 **continuous action spaces**；
2. 🔮 开发支持 **global optimal policy search** 的搜索算法（如 beam search over attributes）；
3. 🔮 探索 **multi-step counterfactual planning**；
4. 🔮 应用于更多领域：
   - 生物序列设计（protein/DNA generation）
   - 小分子功能预测
   - 基因调控机制建模

---

## 总结
**CAT 提供了一个强大而高效的统一框架，将生成建模与序列属性推理深度融合**。它不仅提升了生成质量与可控性，还极大增强了模型的可解释性和实用性，尤其适合医疗、金融、安全等需高可信推理的领域。  
其设计理念——“在每一步都思考后果”——或将启发下一代**因果感知、自我反思型生成模型**的发展。

</details>

---

### 15. [SimPersona: Learning Discrete Buyer Personas from Raw Clickstreams for Grounded E-Commerce Agents](https://arxiv.org/abs/2605.14205)

**Authors**: Zahra Zanjani Foumani, Alberto Castelo, Shuang Xie, Ted Chaiwachirasak, Han Li, Lingyun Wang  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.14205v1  

#### Abstract
LLM-based web agents can navigate live storefronts, yet they often collapse to a single "average buyer" policy, failing to capture the heterogeneous and distributional nature of real buyer populations. Existing personalization methods rely on hand-crafted prompt-based personas that are brittle, diff...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SimPersona: Learning Discrete Buyer Personas from Raw Clickstreams for Grounded E-Commerce Agents

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的基于 **LLM** 的电商 Web Agent 在模拟真实买家行为时存在严重局限：
- 多数方法依赖于手工设计的 **prompt-based personas**，这些方法脆弱（brittle）、难以扩展、上下文效率低，且无法准确反映真实买家群体的分布特性。
- 现有 Agent 往往退化为单一的“平均买家”策略，无法捕捉真实买家群体中的异质性和多样性。
- 个性化方法通常只关注个体行为建模，忽略了对整个买家群体分布（population distribution）的重建。

### 提出的新方法：SimPersona
提出了一种名为 **SimPersona** 的新框架，通过从原始点击流（raw clickstreams）中学习离散的买家类型（discrete buyer personas），并将其作为紧凑的 **persona token** 注入到 LLM Agent 中。

#### 核心创新点：
1. **行为感知的 VQ-VAE 模型**  
   - 使用 **behavior-aware Vector-Quantized Variational Autoencoder (VQ-VAE)** 从历史点击流中自动学习 K 个离散的买家类型。
   - 引入对比损失（contrastive loss）和辅助分类头（auxiliary heads）来增强语义一致性，确保每个 codebook entry 对应有意义的行为模式（如高购买意愿、高探索性等）。

2. **两阶段监督微调（Two-stage SFT）实现 persona grounding**  
   - **Stage 1（冻结主干，仅训练 persona token embeddings）**：使用意图中立的目标（intent-neutral goals）让模型学习每个 persona token 的内在含义。
   - **Stage 2（解冻全部参数）**：在明确任务意图下进行端到端微调，使模型学会融合 persona 和 goal 信号。
   - 这种解耦设计防止了模型依赖 prompt 中的表面线索而忽略 persona token。

3. **支持群体级仿真与分布重建**  
   - 利用 VQ-VAE 学习到的 token 分布 $ p_s(k) $ 来采样合成买家，从而保留商家特有的买家群体结构。
   - 支持无需重新训练或商店特定提示工程的大规模买家模拟。

4. **开源数据管道**  
   - 发布了一个可复现的数据处理流程，将原始电商事件日志转换为买家表示和 Agent 训练轨迹。

### 相比现有方法的优势
| 维度 | Prompt-based 方法 | SimPersona |
|------|------------------|-----------|
| 可扩展性 | 差（需人工编写模板） | 高（全自动学习） |
| 表达能力 | 有限（文本模板难覆盖复杂行为） | 强（向量量化编码丰富行为结构） |
| 分布建模 | 不支持（无群体混合比例） | 支持（自然恢复 $ p_s(k) $） |
| 上下文效率 | 低（消耗大量 token） | 高（单个 token 即可表示 persona） |
| 泛化性 | 弱（依赖具体 prompt 设计） | 强（跨店铺迁移无需适配） |

---

## 2. 核心实验方法和设置

### 数据集
- **训练数据**：来自 39 家商店的约 3,600 个会话（共 ~34,000 步 trace），用于 VQ-VAE 和两阶段 SFT。
- **评估数据**：**42 家未见过的线上商店（held-out live storefronts）**，涵盖 **837万真实买家**，完全独立于训练集。
- 原始数据为去标识化的点击流日志，包括：
  - 页面浏览（page views）
  - 加购（add-to-cart）
  - 结算（checkout）
  - 搜索查询（search queries）
  - 商品收藏夹操作等

### 实验设置
- **VQ-VAE 架构**：
  - 输入维度：403 维（16 个行为标量 + 3×128 维商品嵌入 + 3 位掩码）
  - 编码器：`[403→256→128→96]`
  - Codebook size：K=256
- **LLM 基座模型**：Qwen3-14B-Base
- **新增 token**：添加 `<|persona_0|>` 到 `<|persona_255|>` 共 256 个 persona token
- **训练方式**：
  - Stage 1：冻结 backbone，仅更新 persona token embeddings（1 epoch）
  - Stage 2：解冻所有参数，继续 fine-tuning

### 评估指标
| 指标 | 定义 | 说明 |
|------|------|------|
| **ATC Alignment** | $1 - |\text{ATC}_{\text{real}} - \text{ATC}_{\text{agent}}|$ | 加购率对齐度 |
| **Purchase Alignment** | $1 - |\text{PUR}_{\text{real}} - \text{PUR}_{\text{agent}}|$ | 下单转化率对齐度 |
| **ARA (Action Rate Alignment)** | $0.5 \times \text{ATC align} + 0.5 \times \text{PUR align}$ | 综合行为对齐得分 |
| **Stratum Purity** | 聚类内是否属于同一漏斗层级（A/B/C/D/E） | 衡量聚类语义合理性 |
| **Auxiliary-head Coherence** | 是否出现“低/高”行为混在同一 cluster | 检查行为一致性 |
| **Calinski-Harabasz Index** | 聚类间离散度 / 聚类内紧密度 | 数值越高越好 |
| **Instruction Following Accuracy** | 成功完成导航任务的比例 | 衡量 goal-oriented 性能 |

### 基线方法对比
- **MiniBatch k-means (K=256)**：传统聚类方法，用于比较聚类质量
- **GPT-OSS-120B**：一个强大的通用 LLM 基线（参数量是 SimPersona 的 ~8.6 倍）
- **消融实验**：对比 single-stage vs. two-stage SFT

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | SimPersona 表现 |
|------|----------------|
| **总体 ARA（Action Rate Alignment）** | **78.7%**（正确匹配 persona token） |
| **错误匹配下的 ARA（All-Mis）** | 57.6%（差距达 **21.1 pp**） |
| **最高对齐（D 类买家）** | 达 **95.4%**（窗口购物者行为易模仿） |
| **最低对齐（C 类买家）** | 仍达 **72.1%**（需控制加购但不结算） |
| **Calinski-Harabasz Index** | **206.5**（优于 k-means 的 173.8） |
| **Stratum Purity** | **84.5%**（优于 k-means 的 78.9%） |
| **Auxiliary-head Incoherence** | **<0.5%**（k-means 高达 66.7%） |

> ✅ **结论**：SimPersona 显著优于传统聚类方法，在行为一致性和聚类质量上全面领先。

### 与基线方法对比
#### vs. GPT-OSS-120B（8× 更大模型）
| 任务类型 | SimPersona | GPT-OSS | 差距 |
|--------|------------|---------|------|
| **Cart Tasks（加购）** | 91.4% | 81.7% | **+9.7pp** |
| **Checkout Tasks（进入结算）** | 76.9% | 53.1% | **+23.8pp** |
| **Purchase Tasks（完成购买）** | 59.3% | 52.2% | **+7.1pp** |
| 平均每轮步数 | 14.9 | 11.4 | 更深入探索 |

> ✅ **结论**：尽管参数少 8 倍，SimPersona 在复杂任务上表现更优，尤其体现在长流程任务中。

### 消融实验结果

#### （1）Two-stage vs. Single-stage SFT
| 指标 | Two-Stage | Single-Stage |
|------|----------|--------------|
| 平均目标达成率 | 83.5% | 82.8% |
| 最大错误率 | 28.5% | **71.3%** |
| >30% 错误的店铺数 | 0 | 5 |
| >50% 错误的店铺数 | 0 | 2 |
| 其他错误（格式/解析失败） | 2.2% | **4.6%** |
| 输出崩溃率（unparseable） | 最高 10.9% | 最高 **51.1%** |

> 🔍 **发现**：虽然平均性能相近，但 single-stage 出现严重的尾部风险（tail risk），部分店铺失败率超过 70%，而 two-stage 稳定可控。

#### （2）Persona Token 消融实验（neutral intent setting）
| 指标 | 有 Token | 无 Token | 比例 |
|------|--------|--------|-----|
| 模拟崩溃总数 | 2,650 (5.3%) | 7,862 (15.6%) | ↓ **66%** |
| StagehandTargetClosed（浏览器崩溃） | 1,430 | 6,579 | ↓ **78%** |
| Context Length Error（长度超限） | 1,141 | 1,119 | ≈ 相同 |
| 添加至购物车次数 | 16,132 | 13,698 | ↑ 18% |
| 到达结算页面次数 | 7,823 | 6,664 | ↑ 17% |
| 平均动作数（成功会话） | 11.1 | 9.1 | ↑ 23% |

> ✅ **结论**：persona token 显著提升系统稳定性，并促进更深、更有目的性的导航行为。

---

## 4. 关键结论和发现

### 主要发现
1. **离散 persona token 能有效编码真实买家行为差异**  
   - VQ-VAE 学习到的 codebook 具备高度语义一致性，不同 token 对应显著不同的行为模式（如购买强度、参与度）。
   - 统计检验显示 High vs. Low 购买强度 token 的 Cohen’s d 达 **2.80**，p < 1e-14。

2. **SimPersona 实现了高水平的真实行为对齐**  
   - 在 42 家未见商店上达到 **78.7% 的 ARA 对齐率**，远超错误匹配基线（57.6%），证明 persona token 是关键驱动因素。

3. **两阶段训练至关重要**  
   - 解耦“理解 persona 含义”与“执行动作”两个过程，避免优化干扰，显著降低输出崩溃和尾部失败风险。

4. **支持群体级仿真与分布重建**  
   - Jensen-Shannon 散度仅为 **0.054**，表明 SimPersona 能准确恢复各商店的买家漏斗结构（如购买者占比、浏览者比例）。

5. **小模型也能超越大模型**  
   - 尽管 Qwen3-14B 参数仅为对比基线 GPT-OSS-120B 的 **1/8.6**，但在复杂任务上全面胜出，说明结构设计优于单纯扩大模型规模。

### 局限性
1. **探索行为（exploration）建模较弱**  
   - auxiliary head 在 exploration 上未能有效传递至 Agent 行为（Cohen’s d = 0.10，不显著），可能因信号本身微弱或任务设置引入地板效应。

2. **依赖历史点击流数据**  
   - 新开店铺缺乏历史数据时无法构建 persona，限制冷启动场景应用。

3. **DOM 表示忽略视觉信息**  
   - 当前基于文本的 DOM 解析未利用图像、布局等视觉线索，可能影响决策真实性。

4. **未动态更新 persona**  
   - persona 是静态分配的，未考虑用户兴趣随时间演变。

### 未来工作方向
1. 引入 **multimodal perception**（结合图像、UI 布局）提升决策真实性。
2. 探索使用 VQ-VAE 学习到的 embedding 初始化 LLM token，减少第一阶段训练需求。
3. 开发适用于冷启动商店的 zero-shot 或 few-shot persona 推断机制。
4. 将 persona 建模扩展为动态序列建模，支持用户兴趣演化。
5. 应用于 A/B 测试、推荐系统评估、用户体验测试等下游任务。

--- 

> 📌 **一句话总结**：  
> **SimPersona 通过从原始点击流中学习离散 persona token，并结合两阶段 fine-tuning，实现了高效、可扩展、分布保真的电商 Agent 行为模拟，在真实环境中达到 78.7% 的行为对齐率，显著优于更大规模的通用 LLM。**

</details>

---

### 16. [Distribution Corrected Offline Data Distillation for Large Language Models](https://arxiv.org/abs/2605.14071)

**Authors**: Yumeng Zhang, Zhengbang Yang, Yevin Nikhel Goonatilake, Zhuangdi Zhu  
**Category**: cs.CL  
**Published**: 2026-05-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.14071v1  

#### Abstract
Distilling reasoning traces from strong large language models into smaller ones is a promising route to improve intelligence in resource-constrained settings. Existing approaches face a fundamental trade-off: offline distillation from teacher-generated traces provides high-quality, sample-efficient ...

---

### 17. [Uncertainty Quantification for Large Language Diffusion Models](https://arxiv.org/abs/2605.14570)

**Authors**: Artem Vazhentsev, Vladislav Smirnov, David Li, Maxim Panov, Timothy Baldwin, Artem Shelmanov  
**Category**: cs.CL  
**Published**: 2026-05-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.14570v1  

#### Abstract
Large Language Diffusion Models (LLDMs) are emerging as an alternative to autoregressive models, offering faster inference through higher parallelism. Similar to autoregressive LLMs, they remain prone to hallucinations, making reliable uncertainty quantification (UQ) crucial for safe deployment. How...

---

### 18. [Prompting Policies for Multi-step Reasoning and Tool-Use in Black-box LLMs with Iterative Distillation of Experience](https://arxiv.org/abs/2605.14443)

**Authors**: Krishna Sayana, Ketan Todi, Ambarish Jash  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.14443v1  

#### Abstract
The shift toward interacting with frozen, "black-box" Large Language Models (LLMs) has transformed prompt engineering from a heuristic exercise into a critical optimization challenge. We propose a Reinforcement Learning (RL) framework for training learned prompting policies via iterative distillatio...

---

### 19. [BiFedKD: Bidirectional Federated Knowledge Distillation Framework for Non-IID and Long-Tailed ECG Monitoring](https://arxiv.org/abs/2605.14886)

**Authors**: Zixuan Shu, Tiancheng Cao, Hen-Wei Huang  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.14886v1  

#### Abstract
Electrocardiogram (ECG) monitoring in Internet of Medical Things (IoMT) networks is constrained by strict data-sharing regulations and privacy concerns. Federated learning (FL) enables collaborative learning by keeping raw ECG data on devices, but frequent transmissions of high-dimensional model upd...

---

### 20. [Polar probe linearly decodes semantic structures from LLMs](https://arxiv.org/abs/2605.14125)

**Authors**: Pablo J. Diego-Sim\'on, Pierre Orhan, Yair Lakretz, Jean-R\'emi King  
**Category**: cs.CL  
**Published**: 2026-05-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.14125v1  

#### Abstract
How do artificial neural networks bind concepts to form complex semantic structures? Here, we propose a simple neural code, whereby the existence and the type of relations between entities are represented by the distance and the direction between their embeddings, respectively. We test this hypothes...

---

### 21. [Multi-objective application placement in fog computing using graph neural network-based reinforcement learning](https://arxiv.org/abs/2605.14649)

**Authors**: Isaac Lera, Carlos Guerrero  
**Category**: cs.DC  
**Published**: 2026-05-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.14649v1  

#### Abstract
We propose a framework designed to tackle a multi-objective optimization challenge related to the placement of applications in fog computing, employing a deep reinforcement learning (DRL) approach. Unlike other optimization techniques, such as integer linear programming or genetic algorithms, DRL mo...

---

### 22. [GenAI for Energy-Efficient and Interference-Aware Compressed Sensing of GNSS Signals on a Google Edge TPU](https://arxiv.org/abs/2605.14839)

**Authors**: Thorben Wegner, Lucas Heublein, Tobias Feigl, Felix Ott, Christopher Mutschler, Alexander R\"ugamer  
**Category**: cs.LG  
**Published**: 2026-05-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.14839v1  

#### Abstract
Traditional methods for classifying global navigation satellite system (GNSS) jamming signals typically involve post-processing raw or spectral data streams, requiring complex and costly data transmission to cloud-based interference classification systems. In contrast, our proposed approach efficien...

---

### 23. [Fast Adversarial Attacks with Gradient Prediction](https://arxiv.org/abs/2605.14868)

**Authors**: Kamil Ciosek, Aleksandr V. Petrov, Nicol\`o Felicioni, Konstantina Palla  
**Category**: cs.LG  
**Published**: 2026-05-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.14868v1  

#### Abstract
Generating adversarial examples at scale is a core primitive for robustness evaluation, adversarial training, and red-teaming, yet even "fast" attacks such as FGSM remain throughput-limited by the cost of a backward pass. We introduce a family of attacks that eliminates the backward pass by predicti...

---

### 24. [AIMing for Standardised Explainability Evaluation in GNNs: A Framework and Case Study on Graph Kernel Networks](https://arxiv.org/abs/2605.14884)

**Authors**: Magdalena Proszewska, N. Siddharth  
**Category**: cs.LG  
**Published**: 2026-05-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2605.14884v1  

#### Abstract
Graph Neural Networks (GNNs) have advanced significantly in handling graph-structured data, but a comprehensive framework for evaluating explainability remains lacking. Existing evaluation frameworks primarily involve post-hoc explanations, and operate in the setting where multiple methods generate ...

---

### 25. [SkillFlow: Flow-Driven Recursive Skill Evolution for Agentic Orchestration](https://arxiv.org/abs/2605.14089)

**Authors**: Mingda Zhang, Tiesunlong Shen, Haoran Luo, Wenjin Liu, Zikai Xiao, Erik Cambria, Xiaoying Tang  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.14089v1  

#### Abstract
In recent years, a variety of powerful LLM-based agentic systems have been applied to automate complex tasks through task orchestration. However, existing orchestration methods still face key challenges, including strategy collapse under reward maximization, high gradient variance with opaque credit...

---

### 26. [KGPFN: Unlocking the Potential of Knowledge Graph Foundation Model via In-Context Learning](https://arxiv.org/abs/2605.14907)

**Authors**: Yisen Gao, Jiaxin Bai, Haoyu Huang, Zhongwei Xie, Yufei Li, Hong Ting Tsang, Sirui Han, Yangqiu Song  
**Category**: cs.AI  
**Published**: 2026-05-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.14907v1  

#### Abstract
Knowledge graph (KG) foundation models aim to generalize across graphs with unseen entities and relations by learning transferable relational structure. However, most existing methods primarily emphasize relation-level universality, while in-context learning, the other pillar of foundation models re...

---

### 27. [VectraYX-Nano: A 42M-Parameter Spanish Cybersecurity Language Model with Curriculum Learning and Native Tool Use](https://arxiv.org/abs/2605.13989)

**Authors**: Juan S. Santillana  
**Category**: cs.CL  
**Published**: 2026-05-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.13989v1  

#### Abstract
We present VectraYX-Nano, a 41.95M-parameter decoder-only language model trained from scratch in Spanish for cybersecurity, with a Latin-American focus and native tool invocation via the Model Context Protocol (MCP). Four contributions: (i) Corpus: VectraYX-Sec-ES, a 170M-token Spanish corpus from a...

---

### 28. [Measuring and Mitigating Toxicity in Large Language Models: A Comprehensive Replication Study](https://arxiv.org/abs/2605.14087)

**Authors**: Mokshit Surana, Archit Rathod, Akshaj Satishkumar  
**Category**: cs.CL  
**Published**: 2026-05-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.14087v1  

#### Abstract
Large Language Models (LLMs), when trained on web-scale corpora, inherently absorb toxic patterns from their training data. This leads to ``toxic degeneration'' where even innocuous prompts can trigger harmful outputs. This phenomenon poses significant risks for real-world deployments. Thus, necessi...

---

### 29. [Language Generation as Optimal Control: Closed-Loop Diffusion in Latent Control Space](https://arxiv.org/abs/2605.14531)

**Authors**: ZiYi Dong, Yuliang Huang, Weijian Deng, Xiangyang Ji, Liang Lin, Pengxu Wei  
**Category**: cs.CL  
**Published**: 2026-05-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.14531v1  

#### Abstract
This work reformulates language generation as a stochastic optimal control problem, providing a unified theoretical perspective to analyze autoregressive and diffusion models and explain their limitations (Efficiency-Fidelity Paradox, Irreversibility Error Propagation, Optimization Tractability and ...

---

### 30. [Supervised Distributed Computing: Efficiency and Robustness under a Majority of Adversarial Workers](https://arxiv.org/abs/2605.14784)

**Authors**: John Augustine, Henning Hillebrandt, Manish Kumar, Christian Scheideler, Julian Werthmann  
**Category**: cs.DC  
**Published**: 2026-05-15  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2605.14784v1  

#### Abstract
We consider a recently proposed \emph{supervised distributed computing} paradigm \cite{augustine2025supervised} that extends and refines the standard master-worker paradigm for parallel computations. In this paradigm, there is a supervisor, a source, a target, and a collection of workers. The distri...

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

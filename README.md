# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-09-17 10:27:58 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [GeoMesh: Workload-Balanced and Sign-Compressed Geo-Distributed LLM Training](https://arxiv.org/abs/2609.18388)

**Authors**: Changyong Shin, Jaerim Park, Minchul Kang, Younghun Go, Zhixiong Niu, Yongqiang Xiong, Gyeongsik Yang, Chuck Yoo  
**Category**: cs.DC  
**Published**: 2026-09-17  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2609.18388v1  

#### Abstract
Large language models are increasingly trained on GPUs distributed across multiple regions, but geo-distributed training is challenging in practice. Real clusters often contain GPUs with different speeds and memory capacities, and they communicate over slow wide-area networks. Our analysis shows tha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# GeoMesh: Workload-Balanced and Sign-Compressed Geo-Distributed LLM Training 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Geo-Distributed Training (GDT)** 方法在异构 GPU 集群上面临两大挑战：
- **Idlenete**（计算空闲）：同步训练中，高速 GPU 被慢速 GPU 拖累，导致等待时间高达运行时长的 **20.9%**。
- **Idlenet**（通信空闲）：跨广域网（WAN）传输高精度伪梯度（pseudo-gradient），平均占用了 **65.8%** 的运行时间。

此外，异步方法（如 HALoS）虽减少等待，但因 **stale updates** 导致模型准确率下降（验证困惑度增加 11.7%）。

### 提出的新方法：GeoMesh
GeoMesh 是一种**同步式、面向异构环境的 GDT 框架**，通过两个核心机制解决上述问题：

#### ✅ Adaptive Workload Balancing (AWB)
- **动态分配每个 worker 的 batch size 和 inner step 数量**，基于其 GPU 内存容量和实测训练速度。
- 快速 GPU 执行更多 inner steps，慢速 GPU 减少负载，使各 worker 的每轮耗时趋于一致，显著降低 **Idlenete**。

#### ✅ Compressed Sign Synchronization (CSS)
- 将 full-precision 伪梯度压缩为：
  - **1-bit sign tensor**（符号张量）
  - **轻量级 magnitude scalar**（每张量均值）
  - **token count 元数据**
- 通信体积减少近 **32×**，大幅缓解 **Idlenet**。
- 引入 **token-weighted aggregation**，补偿不同 worker 处理 token 数不均的问题，保持更新尺度一致性。

### 相比现有方法的优势
| 维度 | DiLoCo (同步) | HALoS (异步) | GeoMesh |
|------|----------------|---------------|---------|
| 同步性 | ✅ 稳定更新 | ❌ 梯度过期 | ✅ 稳定更新 |
| GPU 利用率 | ❌ 严重浪费 | ✅ 较高 | ✅ 高且均衡 |
| 通信开销 | ❌ 高（FP32） | ❌ 高 | ✅ 极低（1-bit + magnitude） |
| 模型准确性 | ✅ 高 | ❌ 下降明显 | ✅ 接近 DiLoCo |

---

## 2. 核心实验方法和设置

### 数据集
- **C4 dataset**（Colossal Clean Crawled Corpus），常用于 LLM 训练评估。

### 模型规模
- 三种 LLaMA-style 模型：
  - **150M**, **300M**, **500M** 参数
  - Token 预算按 Chinchilla 定律设为参数量的 20 倍（例如 150M → 3B tokens）

### 实验平台
- **四 worker 异构集群**，逻辑分布于 Microsoft Azure 四个区域：
  - Worker 1: NVIDIA H100 (80GB)
  - Worker 2: A100 (80GB)
  - Worker 3: L40S (48GB)
  - Worker 4: RTX 6000 Ada (48GB)
- **WAN 模拟**：使用 `iperf` 测量 Azure 区域间带宽（580–940 Mbps），并通过 Linux Traffic Control 限速模拟真实延迟。

### 评估指标
| 指标 | 描述 |
|------|------|
| **TTP (Time-to-target Perplexity)** | 达到目标困惑度所需的时间（小时） |
| **FP (Final Perplexity)** | 在给定 token 和计算预算下的最终困惑度 |
| **GPU Idle** | 分解为 `Idlenete` 和 `Idlenet` 占比 |
| **Zero-shot Accuracy** | 在 ARC-C, ARC-E, HellaSwag, MMLU, PIQA, WinoGrande 上的零样本准确率 |

### 基线方法
- **DiLoCo**：代表性的同步 GDT 方法（Douillard et al., 2024）
- **HALoS**：代表性的异步 GDT 方法（Kim et al., 2025）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **TTP 改进** | 最多比 DiLoCo 快 **70.2%**，比 HALoS 快 **67.2%** |
| **Idlenete 降低** | 最多减少 **8.0×**（从 20.9% → <2.6%） |
| **Idlenet 降低** | 最多减少 **5.6×**（从 65.8% → 11.8%） |
| **通信压缩比** | 近 **32×**（FP32 → 1-bit sign + magnitude） |
| **零样本准确率差距** | 仅比 DiLoCo 低 **0.8%**，优于 HALoS |

### 详细对比结果（150M 模型）

| 方法 | TTP (h) | FP | Idlenete (H100) | Idlenet (avg) | Zero-shot Avg Acc (%) |
|------|--------|-----|------------------|----------------|------------------------|
| DiLoCo | 3.7 | 23.0 | 20.9% | 65.8% | 37.88 |
| HALoS | 2.0 | 25.7 | ~0% | 10.4% | 37.39 |
| **GeoMesh** | **1.1** | **23.4** | **<2.6%** | **11.8%** | **37.58** |

> 注：TTP 目标为 perplexity < 30；GeoMesh 在 **1.1 小时内达成目标**，而 DiLoCo 需 3.7 小时。

### 更大规模表现（300M / 500M）
- **DiLoCo**：无法在计算预算内完成训练（GPU 空闲过高）
- **HALoS**：虽然更快，但 FP 明显更高（收敛差）
- **GeoMesh**：
  - 成功在预算内达到目标 TTP
  - FP 表现最优（300M: 18.2, 500M: 15.2）
  - 零样本准确率也优于 HALoS（+0.8% @300M, +3.2% @500M）

### 消融实验（Ablation Study）

| 配置 | 训练时间 | FP |
|------|----------|----|
| DiLoCo | 12.0h | 23.0 |
| AWB only | 7.6h | 23.3 |
| CSS only | 6.5h | 23.3 |
| **Full GeoMesh** | **2.8h** | **23.4** |

- **AWB** 单独可减少 36.2% 时间（缓解 Idlenete）
- **CSS** 单独可减少 46.2% 时间（缓解 Idlenet）
- 两者结合实现 **76.5% 的总加速**

#### 不同压缩方式对比
| 压缩方法 | 压缩比 | 训练时间 | FP |
|---------|--------|--------|----|
| FP32 (原生) | 1× | 12.0h | 23.0 |
| FP16 | 2× | 10.0h | 23.0 |
| 8-bit Quant | 4× | 8.0h | 23.2 |
| **CSS (1-bit)** | **32×** | **6.5h** | **23.3** |

→ 显示 **CSS 的高压缩比直接转化为训练加速**，且精度损失极小。

---

## 4. 关键结论和发现

### 主要发现
1. **同步 GDT 可以兼顾效率与准确性**：
   - GeoMesh 证明，在异构环境下，通过 **workload balancing** 和 **gradient compression**，可以在不牺牲模型质量的前提下大幅提升训练效率。
   
2. **AWB 和 CSS 是正交且互补的优化**：
   - AWB 解决 **计算不平衡**，CSS 解决 **通信瓶颈**，二者协同作用带来指数级加速。

3. **Lion 优化器与 CSS 具有结构兼容性**：
   - Lion 的 sign-based 更新特性使其生成的梯度更适合 1-bit 压缩，理论分析表明其重建误差有确定上界（而 AdamW 无有效界）。

4. **GeoMesh 实现了资源利用率最大化**：
   - H100 GPU 的内存利用率从 49.7% 提升至接近饱和（>90%），核心利用率提升达 **7.7×**。

### 局限性（Limitations）
1. **WAN 模拟非真实部署**：
   - 当前实验基于带宽限制模拟，未考虑真实 WAN 中的 **latency、jitter、packet loss** 等因素。
2. **静态 profiling 假设**：
   - AWB 仅在初始化时测量一次 worker 速度，假设其相对性能稳定；若训练过程中出现剧烈波动（如热节流），可能需重新校准。
3. **压缩策略空间有限**：
   - 仅比较了 1-bit sign 压缩与其他量化方法，未探索 top-k、low-rank 等更复杂压缩方案。
4. **缺乏形式化收敛证明**：
   - 当前为经验验证，尚未提供严格的收敛性理论保障。

### 未来工作方向
- 在真实 geo-distributed 集群上验证 GeoMesh 性能。
- 动态调整 AWB workload 以应对运行时性能变化。
- 探索 **CSS 与 communication overlap**（如 Streaming DiLoCo）的联合优化。
- 扩展到更大规模模型（如 10B+）和更多 worker 场景。
- 研究更细粒度的 magnitude 表示（如 per-head 或 per-row）以进一步提升压缩精度。

---

> ✅ **总结一句话**：  
> **GeoMesh 通过自适应负载均衡（AWB）和符号压缩同步（CSS），在保持同步训练准确性的前提下，将异构跨区域大模型训练速度提升最多 70.2%，并显著降低 GPU 空闲，是高效、实用的 GDT 新范式。**

</details>

---

### 2. [The Inference Engineering Pareto Atlas: Which Optimizations Dominate the Cost, Quality, and Latency Frontier?](https://arxiv.org/abs/2609.17863)

**Authors**: Srikanta Datta Tumkur, Jay Iyer, Mehar Simhadri, Sai Pavan Kumar, Sai Kapil Kumar, Ramesh Nampelly  
**Category**: cs.AI  
**Published**: 2026-09-17  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2609.17863v1  

#### Abstract
LLM inference optimizations report speedups on different models, GPUs, prompts, and quality metrics, making them hard to compare or combine. We build a cost, quality, and latency Pareto atlas to identify the best configurations for different deployment constraints. Since exhaustive testing is imprac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*The Inference Engineering Pareto Atlas: Which Optimizations Dominate the Cost, Quality, and Latency Frontier?*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大语言模型（LLM）推理优化领域存在大量孤立的性能提升报告（如量化、KV-cache压缩、推测解码等），但这些优化在**不同模型、GPU、提示词和质量度量下进行测试**，导致其结果无法直接比较或组合。工程师难以判断在给定预算、延迟或质量约束下，哪种优化或组合最优。

该论文系统性地解决了这一**可比性缺失**和**决策支持不足**的问题。

### 🚀 提出的新方法与创新思路
提出构建一个 **“成本-质量-延迟帕累托图谱”（Cost-Quality-Latency Pareto Atlas）**，其核心是：

- 将主流推理优化技术（quantization, KV compression, speculative decoding, batching, sparse attention）及其组合置于**统一的评估轴上**；
- 构建**主导配置地图**（dominance atlas），明确指出在每种部署场景（如低延迟、高吞吐、低成本）下的最优选择；
- 采用 **“测量+模拟”（measure-then-simulate）方法论**：通过实测少量锚点（anchors）校准一个 profiled simulator（基于Vidur框架），再用模拟器填充庞大的配置空间，实现高效且可信的帕累托前沿计算。

### 🔍 相比现有方法的优势
| 方面 | 现有工作 | 本文优势 |
|------|--------|---------|
| 可比性 | 各自为政，指标不一致 | 统一模型、硬件、工作负载、评估维度 |
| 完整性 | 多数只评估单一优化 | 覆盖单个方法及兼容组合堆叠 |
| 决策支持 | 报告“加速X倍”无上下文 | 明确命名每个约束下的“赢家配置” |
| 效率 | 全面实测成本过高不可行 | 测量锚点 + 校准模拟器，兼顾精度与覆盖 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与模型
- **主模型**：`Qwen2.5-7B-Instruct`（由 vLLM 0.12 部署）
- **质量评估任务**：
  - **GSM8K**（200题，5-shot，greedy decoding）用于衡量数学推理能力
  - 其他协议工作负载包括 ShareGPT（Chat）、LongBench/RULER（长上下文）、HumanEval（代码生成），但仅GSM8K用于实际质量打分
- **硬件平台**（RunPod实例）：
  - `L4` ($0.39/hr)
  - `A100 80GB PCIe` ($1.39/hr)
  - `H100 PCIe` ($2.89/hr)

### ⚙️ 实验设置
- **输入输出长度**：512-token 输入，128-token 输出
- **优化方法覆盖**：
  - **Quantization**：AWQ-4bit、online FP8 weights
  - **KV Cache Compression**：FP8 KV cache（naive路径）
  - **Speculative Decoding**：n-gram prompt lookup（draft-model speculation未实现）
  - **Batching**：continuous & SLO-aware batching（via vLLM）
  - **Sparse Attention**：NSA（仅模拟）
- **组合方式**：兼容方法堆叠（如 quant + batch + sparse）

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **质量（Quality）** | GSM8K 准确率（strict/flexible extraction） |
| **延迟（Latency）** | Time To First Token (TTFT)，Time Per Output Token (TPOT @ P50/P99) |
| **成本（Cost）** | \$ per million tokens（基于GPU时价和吞吐） |
| **辅助指标** | Throughput (tokens/sec)，GPU memory usage |

### 🔁 方法流程（Algorithm 1）
```python
for each (model, workload, GPU):
    measure anchor configs → get real latency/cost/quality
    calibrate profiled simulator to match anchors
simulate full config space (methods × combinations × hardware)
compute Pareto frontier over (quality, latency, cost)
label dominant config under:
    - tight latency
    - high throughput
    - low cost
    - quality floor @ min cost
return Pareto Atlas
```

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 配置 | 质量 (GSM8K) | 成本 (\$/Mtok) | TPOT (相对基准) | 所属GPU |
|------|-------------|----------------|------------------|--------|
| **AWQ-4bit** | 0.720 (↓5.9%) | 最低（~\$0.07） | **0.34×** on L4 | L4/A100/H100 |
| **FP8 weights** | 0.760 (~99.4% baseline) | \$0.106 | 0.61–0.65× | 所有GPU |
| **FP8 KV cache (naive)** | **0/200 正确** | 正常吞吐 | 快速但完全错误 | — |
| **Throughput_stack@A100** | 0.742 | **\$0.106** | — | A100 |
| **Low_latency_stack@H100** | 0.746 | \$4.62 | 最优TTFT | H100 |

> 注：FP16 baseline 质量为 0.765（strict）

### 🔀 与基线方法对比结果
- **AWQ-4bit** 加速显著（尤其在低端GPU如L4达0.34× TPOT），但**严格准确率下降5.9%**，低于95%质量门槛（边界情况，在采样误差±3pp内）；
- **FP8 weight quantization** 在几乎无损质量下提供稳定加速（0.61–0.65× TPOT），成为三个胜出配置的一部分；
- **n-gram speculative decoding** 实测仅带来 **0.90–0.98×** 延迟改善，**无实质性增益**；
- **稀疏注意力（sparse attention）** 仅能模拟，但在多个前沿配置中出现；
- **KV-cache压缩（FP8 KV）** 表现反直觉：虽维持正常吞吐，但**所有200道题全错**，凸显仅看速度会误导决策。

### 🧪 消融实验与组合分析（见 Fig. 8）
- 总共36个配置中，**18个位于帕累托前沿**；
- 单一方法：21种中有 **9种进入前沿**（占比43%）；
- 组合堆叠：15种中有 **9种进入前沿**（占比60%），说明**堆叠仍有效**；
- 但加入质量维度后，原本因速度快而入围的KV-cache配置被全部淘汰。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **没有“万能赢家”**：最优配置取决于部署约束：
   - **紧延迟（Tight Latency）** → H100 上的 `spec+quant+sparse`
   - **高吞吐 / 低成本** → A100 上的 `quant+batch+sparse`
   - **质量优先最低成本** → A100 上的 `batched`（非压缩）

2. **FP8 weight quantization 是质量-性能平衡的最佳实践**：
   - 保持 **99.4% 原始准确率**
   - 在所有GPU上均有约 **35–39% 延迟降低**
   - 是四个区域胜者中的**三个组成部分**

3. **AWQ-4bit 虽快但牺牲格式一致性**：
   - 数值正确性尚可（flexible extraction持平FP16）
   - 但输出格式错误导致 strict accuracy 下降明显
   - 在质量敏感场景中可能不适用

4. **纯速度导向会选出灾难性配置**：
   - 如“FP8 KV cache”看似高效，实则答案全错
   - 强调必须联合评估 **cost-quality-latency**

5. **硬件影响巨大**：
   - AWQ在便宜GPU（L4）上收益更大（0.34×），而在H100上仅0.53×
   - A100 成为性价比王者（吞吐与成本双优）
   - L4 无法满足任何严苛延迟要求

6. **组合优于单一优化**：
   - 堆叠配置更频繁达到帕累托前沿（60% vs 43%）
   - 特别是 `quant + batch + sparse` 成为高频胜者

### ⚠️ 局限性
1. **质量评估局限于 GSM8K（n=200）**，结论外推需谨慎；
2. **speculative decoding 仅测试 n-gram lookup**，未使用 draft model（因vLLM V1不支持且无EAGLE head）；
3. **KV-cache 压缩仅测试 naive FP8 路径**，未包含 scale-calibrated 或 H2O/PyramidKV 等先进方案；
4. **Sparse attention 完全依赖模拟**，缺乏真实kernel验证；
5. **价格假设基于 RunPod July 2026 rate**，市场波动会影响绝对排名；
6. 模拟器在校准批次大小上有插值保真度，但泛化能力受限于跨campaign漂移（<1.5%）

### 🔮 未来工作方向
- 扩展至更多模型（如 Llama-3）、更多任务（多模态、Agent workflows）
- 支持动态价格重规划（price-aware atlas update）
- 引入更精细的质量度量（如格式鲁棒性、思维链连贯性）
- 实现并锚定真正的 speculative decoding（如 EAGLE）
- 探索 scale-calibrated KV quantization（如 KVQuant）的实际表现
- 开源 Atlas 工具链，形成社区标准评估范式

---

## 结语
本文提出了一个面向工程实践的 **LLM推理优化决策框架**——通过构建 **Pareto Atlas**，将碎片化的优化成果整合为一张可读的地图，帮助开发者在复杂权衡中做出理性选择。其核心价值不仅是具体结果，更是建立了一套 **“测量→校准→模拟→决策”** 的标准化方法论，有望成为未来推理系统设计的基础设施。

</details>

---

### 3. [Zero-I/O Fault Recovery for Sharded Deep Learning via Dynamic Framework Dependency Rebinding](https://arxiv.org/abs/2609.18178)

**Authors**: Genlang Chen, Junyi Zhu  
**Category**: cs.DC  
**Published**: 2026-09-17  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2609.18178v1  

#### Abstract
Distributed model training at scale is frequently interrupted by transient network failures, conventionally forcing cluster managers to abort all processes and roll back to the latest checkpoint. While periodic checkpointing provides durability, frequent snapshotting introduces severe storage backpr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Zero-I/O Fault Recovery for Sharded Deep Learning via Dynamic Framework Dependency Rebinding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在大规模分布式深度学习训练中，尤其是基于 **Fully Sharded Data Parallel (FSDP)** 的模型并行场景下，**短暂的通信故障**（如 NCCL 超时、网络中断）会导致整个训练进程被强制终止，并回滚到最近的持久化 **checkpoint**。尽管这些故障是瞬态的且设备内存中的计算状态（如模型参数、优化器状态）并未损坏，传统系统仍会丢弃所有中间状态，造成严重的计算资源浪费。

此外，频繁 checkpoint 虽可缩短恢复距离，但会引入显著的 **storage backpressure**（存储反压），包括高 I/O 开销、内存占用增加以及训练吞吐下降。

---

### 提出了什么新方法或新思路
本文提出 **AccelPact** —— 一种支持 **zero-I/O in-memory fault recovery** 的并行运行时系统，其核心思想是：

- **识别“可恢复边界”**：仅在 **committed optimizer step** 边界处发生通信失败时进行内存内恢复。此时，参数更新已完成，梯度已清零，设备流同步完成，状态处于静止（quiescent）且一致的状态。
- **动态框架依赖重绑定（Dynamic Framework Dependency Rebinding）**：
  - 发现 PyTorch FSDP 在模块包装器中缓存了 `L+1` 个对旧 communicator 的引用（如 `_inter_node_pg`）。
  - 当替换失效的 communicator 后，必须显式地将这些内部引用重新绑定至新的 communicator 实例，否则后续 collective 操作将崩溃。
- **非侵入式实现**：
  - 通过 hook 机制直接操作原生 C++ communicator 实例，无需修改用户代码。
  - 支持 `torch.compile` 下的 zero graph break。

---

### 相比现有方法的优势
| 维度 | 现有方法（如 Checkpoint Restart / NVRx） | AccelPact |
|------|----------------------------------------|---------|
| **恢复方式** | 从磁盘加载 checkpoint 并重放历史步骤 | 内存中修复 communicator 和框架引用，零重放 |
| **I/O 开销** | 高（需读写数十 GB 状态） | **Zero-I/O** |
| **恢复延迟** | 数分钟级（取决于 checkpoint age） | **亚秒级（~0.7s）** |
| **吞吐影响** | 显著降低（因 checkpoint I/O） | 几乎无影响（guard overhead < 0.5%） |
| **数值一致性** | 依赖于 checkpoint 正确性 | 所有 rank 参数和 optimizer state **bit-identical** |
| **兼容性** | 可能需要集成特定库（如 NVRx） | **零代码修改，无缝集成标准 FSDP** |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **WikiText-103**：用于训练 Mistral-7B 模型的标准语言建模数据集。

---

### 实验设置
- **硬件平台**：
  - 主测试环境：4 节点 × 4 × **NVIDIA RTX 5880**（共 16 GPUs）
  - 对比平台：NVIDIA A100、Huawei Ascend 910B
- **模型配置**：
  - **Mistral-7B-v0.3**（7.25B 参数），32 层 Transformer
  - 使用 **Hybrid Sharded Data Parallel (HSDP)**：4-way intra-node sharding + 4-way inter-node replication
- **训练参数**：
  - 序列长度：512
  - Microbatch size：1，Accumulation steps：8 → Global batch = 65,536 tokens/update
  - Optimizer：AdamW（BF16），每步耗时约 220 秒
- **故障注入**：
  - 在指定 optimizer step（如第 15 或 20 步）后手动触发跨节点 collective 超时
  - 注入方式：利用 `TORCH_NCCL_BLOCKING_WAIT=1` 触发异常

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **Goodput Improvement** | 完整训练任务时间加速比（含故障恢复） |
| **Fault-to-Frontier Time** | 故障发生后恢复至下一个训练前沿所需时间 |
| **Replay Savings** | 节省的重放步数 × 单步耗时 |
| **Numerical Drift** | 多次恢复后与原始轨迹的数值偏差（bitwise comparison） |
| **Memory & Resource Stability** | GPU 内存、host RSS、文件描述符等是否泄漏 |
| **Guard Overhead** | 正常训练步骤中边界检测带来的额外开销 |

---

### 基线方法对比
- **Cold Restart**：完全重启，从 checkpoint 加载并重放所有未提交步骤
- **NVRx Checkpoint Restoration**：使用 NVIDIA Resiliency Extension 进行快速重启恢复
- **No Recovery / Always L2**：不尝试内存恢复，一律走 checkpoint 流程

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### ✅ 全流程 Goodput 提升（A=5）
| 方法 | 平均总耗时（s） | 相对 Cold Restart 加速比 |
|------|------------------|--------------------------|
| **AccelPact** | ~5750 | **1.197×** |
| Cold Restart | ~6860 | 1.000× |
| NVRx Restore | ~6870 | 1.194× |

> ➤ 节省约 **18.8 分钟**，相当于 **5 个完整训练步** 的计算量。

#### ✅ 不同 Checkpoint Age 下的表现（A ∈ {3,6,12,18}）
| A | AccelPact Fault-to-Frontier (s) | Cold Restart (s) | 加速比 |
|----|-------------------------------|------------------|--------|
| 3 | 1267 | 1959 | 1.55× |
| 6 | 1260 | 2620 | 2.08× |
| 12 | 1261 | 3931 | 3.12× |
| 18 | **1259** | **5269** | **4.18×** |

> ➤ AccelPact 恢复时间基本恒定（~21 分钟），而重启方法呈线性增长  
> ➤ 在 A=18 时，**整体任务提速达 1.698×**

#### ✅ 恢复过程分解（16 GPUs）
| 阶段 | 平均耗时（ms） |
|------|---------------|
| Gloo Admission | 26.82 |
| Communicator Retirement | 542.60 |
| NCCL Rebuild | 118.57 |
| **Reference Rebinding** | **0.518** ✅ |
| 总计 | **~694 ms** |

> ➤ Reference rebinding 时间为常数（仅遍历固定 L+1=33 个引用），**不随规模扩展**

---

### 消融实验结果

#### 🔬 **必须重绑定框架引用（ablation study）**
- 若仅替换 communicator 但跳过 `rebind_fsdp_states()`：
  - 新 communicator 初始化成功并通过 probe 测试
  - 但在下一 backward pass 中立即崩溃：
    ```
    DistBackendError: NCCL communicator was aborted
    ```
- 结论：**communicator 替换 alone is insufficient**；必须完成 **L+1 个内部引用的动态重定向**

#### 🔬 **安全拒绝 in-flight 故障**
- 在 forward 或 backward 中途注入故障：
  - AccelPact 正确识别为 “uncommitted”，执行 **safe-rejection**
  - 所有 worker 退出，交由 supervisor 执行 checkpoint reload
  - 数值验证通过（bitwise match）

#### 🔬 **重复生命周期稳定性**
- 连续注入 **10 次故障**（跨越 25 个 steps）：
  - 每次恢复耗时：0.664–0.870 s
  - GPU memory（allocated/reserved）、host RSS、threads、FDs 均保持稳定（见 Table VI）
  - 最终所有 16 ranks 与无故障运行结果 **bit-identical**

#### 🔬 **fault-free guard overhead**
- 在正常训练中插入边界检查：
  - 16-GPU 场景下平均开销：**+0.200% / -0.412%**（几乎无影响）
  - 单节点测试中最大波动 < 2%

---

## 4. 关键结论和发现

### 主要发现
1. **瞬态通信故障不应导致全量重放**：
   - 在 committed boundary 上，device memory 中的状态是完整且未损坏的。
2. **FSDP 的内部缓存机制阻碍了 communicator 替换**：
   - `L+1` 个硬编码的 process group 引用必须被动态 rebinding。
3. **AccelPact 实现了真正的 zero-I/O 恢复**：
   - 无需任何 checkpoint I/O，即可实现亚秒级恢复。
4. **高频 checkpoint 不可行**：
   - 微基准显示，即使异步 checkpoint 也会带来高达 **+656.7% 的延迟开销** 和 **3.39 TB/hour 的写流量**，无法替代内存状态保留。
5. **AccelPact 显著提升有效吞吐（goodput）**：
   - 在真实 Mistral-7B 训练中，相比冷启动恢复最高提速 **1.698×**。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **仅适用于 committed boundary 故障** | 无法处理 forward/backward 中途失败，此类情况仍需 checkpoint 回滚 |
| **框架特异性强** | 当前实现依赖于 FSDP1 内部属性（如 `_inter_node_pg`），未来若迁移到 FSDP2/DTensor 需适配 |
| **仅应对 transient 故障** | 对 GPU ECC 错误、节点宕机等永久性故障无效 |
| **依赖驱动行为一致性** | 如 Ascend 910B 在某些场景下不允许 communicator 重建，需结合平台策略判断 |

---

### 未来工作方向
1. **扩展至 in-flight recovery**：
   - 探索 mid-step 状态保存与恢复机制（如 gradient checkpointing + selective replay）
2. **通用化 rebinding 机制**：
   - 构建自动探测和修补框架内部引用的元系统，适应不同 DL 框架（如 JAX、TensorFlow）
3. **与编译器深度集成**：
   - 在 `torch.compile` 或 `lazy_tensor` 中内置恢复能力，进一步减少 runtime 干预
4. **跨代际容错架构设计**：
   - 将 AccelPact 作为 fast path，与 NVRx/Gemini 等 checkpoint-based 方法组成混合弹性训练栈

---

> 📌 **一句话总结**：  
> **AccelPact 利用“计算状态未损 + 框架引用可修”的洞察，在 committed boundary 上实现了无需 I/O 的故障自愈，为大规模 sharded 训练提供了高效、精确、低侵入的容错新范式。**

</details>

---

### 4. [Beyond Truncation: Rethinking LLM Decoding as Ensemble Pruning](https://arxiv.org/abs/2609.18723)

**Authors**: Dunyao Xue, Chengshuo Du, Zhengbo Wang, Wenlin Dai, Cheng Meng  
**Category**: cs.AI  
**Published**: 2026-09-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.18723v1  

#### Abstract
We introduce Mahalanobis-Ensemble Decoding (ME-Decoding), a novel Large Language Model (LLM) decoding framework that frames candidate token selection as ensemble pruning. Existing selection strategies rely predominantly on scalar probabilities, ignoring geometric semantic relationships and causing c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Truncation: Rethinking LLM Decoding as Ensemble Pruning

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 LLM 解码方法（如 Top-k、nucleus sampling）主要依赖标量概率进行候选 token 选择，忽略了 token 之间的**语义几何关系**，导致生成路径中存在冗余，影响生成质量和推理稳定性。同时，当前一些引入几何信息的方法（如 Top-W、CraEG）往往需要复杂的优化过程或直接重加权原始概率，带来显著的计算开销或推理不稳定性。

### 提出的新方法：ME-Decoding
本文提出了一种全新的解码框架——**Mahalanobis-Ensemble Decoding (ME-Decoding)**，其核心思想是将 LLM 解码中的候选 token 选择问题重新构想为**集成剪枝（Ensemble Pruning）**问题。

- **新视角**：将每个候选 token 视为一个弱语义预测器（weak semantic predictor），目标是从中选择一个紧凑且互补的子集，以平衡个体置信度（token probability）和集体多样性（embedding-based similarity）。
- **核心机制**：定义了一个基于 **Mahalanobis 距离**的目标函数——**Mahalanobis-Ensemble Score (MES)**，该分数在保留高概率 token 的同时，通过 token 相似性矩阵动态地对冗余生成路径进行折扣。
- **高效算法**：设计了一个具有近线性复杂度的贪心选择算法，并提供了理论上的近似保证。

### 相比现有方法的优势
- **更优的准确性-多样性权衡**：通过显式控制冗余，提升了 token 选择的质量，在保持高置信度的同时增强了语义多样性。
- **低推理开销**：算法复杂度在早停条件下接近候选集大小的线性，推理开销极小，是一个即插即用（plug-and-play）模块。
- **理论保障**：提供了贪心轨迹单峰性（greedy-trajectory unimodality）和全局最优近似保证等理论支持。

---

## 2. 核心实验方法和设置

### 数据集
实验涵盖了两类任务：
- **推理任务（Reasoning Benchmarks）**：
  - **GSM8K**：小学数学应用题数据集，评估模型的数学推理能力。
  - **GPQA**：研究生级别的科学问答数据集，评估模型的深度知识推理能力。
- **开放生成任务（Open-ended Generation）**：
  - **AlpacaEval**：指令跟随基准，通过自动裁判（judge）比较生成回复的质量，报告候选胜率（win-rate）。
  - **MT-Bench**：多轮对话质量评估基准，由自动裁判打分（1-10分），报告平均得分。

### 实验设置和评估指标
- **模型**：在三个主流指令微调模型上进行评估：
  - `Qwen3-4B-Instruct`
  - `Phi-4-mini-Instruct`
  - `Mistral-7B-Instruct`
- **温度设置**：`T ∈ {1.0, 1.5, 2.0}`，用于测试不同随机性下的鲁棒性。
- **评估指标**：
  - 推理任务：答案准确率（accuracy）。
  - 开放生成任务：AlpacaEval 的胜率（%）、MT-Bench 的平均裁判得分。
- **实现细节**：候选池大小 `N=512`，ME-Decoding 默认超参数 `λ=0.9`。

### 基线方法对比
与多种代表性解码方法进行了公平比较：
- **概率截断类**：`Min-p`, `Top-p`, `p-less`
- **熵感知/有界熵类**：`Top-H`
- **几何感知类**：`Top-W`, `CraEG`
- **确定性方法**：`Greedy`

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
#### 推理任务（GSM8K 和 GPQA）
- **GSM8K**（表2）：ME-Decoding 在所有模型和温度下均取得最佳表现，平均准确率达到 **72.66%**，显著优于第二名 Top-W（70.62%）。
- **GPQA**（表3）：ME-Decoding 同样取得最高平均准确率 **32.96%**，优于 Top-W（31.56%）和 p-less（31.08%）。
- **趋势分析**：如图2所示，随着温度升高，传统方法（如 Top-p）性能急剧下降，而 ME-Decoding 表现出更强的稳定性，证明其能有效过滤噪声 token。

#### 开放生成任务（AlpacaEval 和 MT-Bench）
- **综合排名**（图3右）：ME-Decoding 在所有模型、温度和基准上的**平均排名最高**，表明其优势具有普适性。
- **具体指标**：
  - **AlpacaEval**：ME-Decoding 平均胜率为 **14.80%**，优于 Top-H（14.36%）和 p-less（14.33%）。
  - **MT-Bench**：ME-Decoding 平均得分为 **7.17**，优于 Top-W（7.08）和 p-less（6.94）。
- **双裁判验证**（表15）：使用 `DeepSeek-V4-Pro` 和 `GLM-5.2` 两个不同裁判均验证了 ME-Decoding 的优越性。

### 消融实验结果
- **组件消融**（表9）：
  - 移除自适应带宽（adaptive bandwidth）会导致性能下降，尤其在高温下。
  - 将相似性矩阵 `K` 替换为单位矩阵 `I`（即忽略几何信息）会显著降低准确率，证明了嵌入空间几何的重要性。
- **核对齐性验证**（表8）：
  - 使用打乱的相似性矩阵（permuted kernel）或单位矩阵，性能均低于正确对齐的语义核，说明 token 间正确的语义关系建模至关重要。
- **超参数敏感性**（表12）：
  - ME-Decoding 对超参数 `λ` 具有较好的鲁棒性，在 `λ=0.9` 时达到最佳平均性能。

---

## 4. 关键结论和发现

### 主要发现
1. **集成剪枝视角的有效性**：将 LLM 解码视为集成剪枝问题是一种新颖且有效的范式，能够自然地平衡 token 的置信度和多样性。
2. **几何信息的价值**：利用 token embedding 的几何关系可以显著提升解码质量，避免仅依赖概率带来的冗余问题。
3. **ME-Decoding 的优越性**：所提出的 ME-Decoding 框架在多个推理和开放生成任务上一致超越了强基线，实现了更好的准确性-多样性权衡，且推理开销极低。
4. **理论与实践结合**：提出的贪心算法不仅高效，而且具有理论上的近似保证和单峰性，支持早停策略。

### 方法的局限性
1. **相似性矩阵的构建**：当前方法依赖静态的 token embeddings，可能无法捕捉上下文相关的语义差异。未来可探索基于隐藏状态的动态相似性核。
2. **多样性保守性**：尽管在推理任务上表现优异，但其输出多样性并非总是最高，表明当前核设计在促进多样性方面可能偏于保守。
3. **子集大小选择**：最优子集大小仍具挑战性，MES 准则虽提供原则性规则，但未必对所有分布都最优。

### 未来工作方向
- 设计更简单、更有效的基于集成剪枝目标的解码目标。
- 将 ME-Decoding 扩展到长文本生成、多样本推理和验证增强解码等更广泛场景。
- 探索自适应的子集大小规则和替代目标函数。

</details>

---

### 5. [Towards Training Private LLMs: Exploring Fine-Tuning Language Models on Apple Silicon with RDMA over Thunderbolt](https://arxiv.org/abs/2609.18066)

**Authors**: En-Ming Huang, Yao-Ting Hsieh, Hsiang-Yu Tsou, Mu-Chi Chen, Shih-Hao Hung, H. T. Kung  
**Category**: cs.DC  
**Published**: 2026-09-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2609.18066v1  

#### Abstract
Private large language model (LLM) fine-tuning is increasingly important for organizations that need to adapt models using sensitive data, but it often exceeds the memory capacity of commodity datacenter accelerators. Apple Silicon offers a different design point through large unified memory and low...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Towards Training Private LLMs: Exploring Fine-Tuning Language Models on Apple Silicon with RDMA over Thunderbolt

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于**私有大型语言模型（Private LLM）的微调**场景，解决以下关键挑战：
- 私有组织需要在不泄露敏感数据的前提下对 LLM 进行 fine-tuning。
- 传统数据中心加速器（如 NVIDIA H100）内存容量有限（仅 80 GiB），难以支持长上下文（long-context）训练任务。
- 苹果 Apple Silicon 平台虽具备高达 512 GiB 的统一内存（unified memory），但其通过 Thunderbolt 支持的 RDMA 通信性能尚未被系统评估和优化。

### 提出的新方法与创新思路
作者提出并实现了三项关键技术优化，以提升 Apple Silicon 多节点集群上的 LLM 微调效率：

#### （1）**Multi-trunk RDMA Communication**
- 利用 Mac Studio 上多达 6 个 Thunderbolt 5 端口，在两个节点之间建立多条物理 RDMA 链路。
- 将每条链路视为一个独立的 Verbs 接口，并将 collective 操作（如 all-reduce）的消息分块并行发送到多个 trunk 上，实现带宽聚合。
- 引入持久化 worker 线程池（persistent worker threads），避免每次通信都创建线程带来的开销。

#### （2）**CPU-side Gradient Overlap**
- 在 GPU 执行 backward pass 的同时，一旦某一层的梯度计算完成，立即由 CPU 启动该层的 all-reduce 操作。
- 实现了 **layer-wise communication/computation overlap**，有效隐藏通信延迟。

#### （3）**端到端系统集成与评估框架**
- 在 Apple 官方的 MLX 框架基础上扩展 JACCL 通信后端，完整实现上述优化。
- 构建了一个四节点 Mac Studio 集群用于实证研究。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **内存容量** | 单节点 512 GiB unified memory 显著优于 H100 的 80 GiB，可容纳更长序列而无需 context parallelism |
| **成本效益** | 单台 Mac Studio 成本约 \$10,000，远低于 \$30,000 的单张 H100 或 \$300,000 的 8×H100 DGX 工作站 |
| **通信优化空间** | 发现原始 JACCL 带宽利用率极低（<20 Gbps per TB），通过 multi-trunk + thread pool 提升至 **53.6 Gbps**（2 节点） |
| **适用场景匹配** | 特别适合 memory-bound、长上下文、小批量 fine-tuning 场景，满足中小企业或研究机构的私有化部署需求 |

---

## 2. 核心实验方法和设置

### 使用的数据集与模型
- **模型**：Qwen3-9B（9B 参数量）
- **任务类型**：Supervised Fine-Tuning（SFT）
- **优化器**：SGD（无 momentum，减少 optimizer state 开销，便于分析纯梯度同步影响）

### 实验平台配置
| 平台 | 配置 |
|------|------|
| **Apple Silicon Cluster** | 4 台 Mac Studio（M3 Ultra），每台含：<br>• 32 核 CPU / 80 核 GPU<br>• 512 GiB LPDDR5 unified memory<br>• 6× Thunderbolt 5 ports（用于 RDMA）<br>• 使用自定义 multi-trunk JACCL backend |
| **NVIDIA H100 对照组** | 单卡及多卡 H100 SXM 系统，80 GiB HBM，NVLink 连接 |

### 评估指标
- **Throughput**：tokens per second（token/s）
- **Weak Scaling Efficiency**：多节点下总吞吐相对于单节点的比例
- **End-to-end Training Time per Iteration**
- **Memory Footprint**：不同 sequence length 和 batch size 下的显存占用
- **Communication Performance**：
  - Point-to-point send/recv 延迟与带宽
  - All-reduce 算法带宽（algorithm bandwidth）

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Original JACCL (single-trunk)** | Apple 官方提供的默认 RDMA-over-TB 实现，仅使用一条 TB 链路 |
| **JACCL + No Overlap** | 使用 multi-trunk，但不启用 CPU-side gradient overlap |
| **10GbE Backend** | 使用 Mac 内置千兆以太网作为通信后端进行对照 |
| **NVIDIA H100 (multi-node)** | 行业标准高性能平台，用于性能天花板比较 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）通信微基准测试（Micro-benchmarks）
| 指标 | 结果 |
|------|------|
| **单 TB 链路实际带宽** | ~20 Gbps（远低于 TB5 标称的 80 Gbps） |
| **6-trunk send/recv 带宽** | 达到 **~100 Gbps**（raw bandwidth） |
| **All-reduce algorithm bandwidth** | 从单 trunk 的 10.7 Gbps 提升至 **53.6 Gbps**（2 节点） |
| **All-reduce 延迟（small msg）** | ~21.3 μs，约为 10GbE 的 1/10 |

> 📌 **重要发现**：应用可见带宽严重受限于软件并发能力；必须结合 multi-threading 和 persistent thread pool 才能释放硬件潜力。

#### （2）Qwen3-9B 微调性能（sequence length = 17408）
| 节点数 | 配置 | Tokens/s | 相对于单节点效率 |
|--------|------|----------|------------------|
| 1 | baseline | 264 | 100% |
| 2 | 6-trunk, no overlap | 490 | 93% |
| 2 | 6-trunk, w/ overlap | 492 | 93% |
| 4 | 6-trunk, w/ overlap | **936** | **89%** |

- 四节点集群达到 **3.5× 单节点吞吐**
- **CPU-side overlap 在 4 节点上带来 1.6× 加速**（从 122s → 74s 每轮迭代时间）

#### （3）与 H100 的横向对比（sequence length = 2048）
| 平台 | 节点数 | Local Batch Size | Throughput (tokens/s) |
|------|--------|------------------|------------------------|
| H100 | 1 | 1 | 1045 |
| H100 | 4 | 1 | 3830 |
| Mac Studio | 4 | 12 | **1210**（with overlap） |

- 尽管 H100 性能更高，但 **4 节点 Mac Studio 吞吐已超过单卡 H100**
- Mac Studio 可运行更大 batch size（因内存充足），从而提高利用率

### 消融实验结果
| 优化项 | 效果 |
|-------|------|
| **Multi-trunk only** | 最高提升 ~1.4× 吞吐（依赖可用 trunk 数） |
| **Persistent thread pool vs 动态线程** | 多 trunk 下性能提升 >3×，证明线程管理至关重要 |
| **Gradient overlap** | 在 4 节点下进一步降低 iteration time 1.6×，尤其在大模型中收益显著 |
| **Trunk 数量增加** | 性能随 trunk 数单调上升，但在 4 节点拓扑中受限于最多 2 trunk/peer |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Apple Silicon 是私有 LLM 微调的可行且经济的选择**  
   - 凭借 **512 GiB 统一内存**，可在单节点上处理长达 **17408 tokens** 的上下文，无需复杂的 context parallelism。
   
2. ⚠️ **RDMA-over-Thunderbolt 实际带宽远低于理论值**  
   - 单端口仅实现约 20 Gbps，仅为 TB5 标称带宽（80 Gbps）的 25%，表明存在严重的软件栈瓶颈。

3. 🔧 **系统级优化可显著释放通信潜力**  
   - 通过 **multi-trunk + persistent thread pool + CPU-side overlap**，all-reduce 带宽提升达 **5× 以上**，end-to-end throughput 提升 **1.6×**。

4. 💰 **性价比优势明显**  
   - 对于中小规模组织，Apple Silicon 提供了一种低成本、高内存容量的替代方案，尤其适用于长上下文、数据敏感型 fine-tuning。

5. 🔄 **weak scaling 效率高达 89%~93%**  
   - 表明经过优化后的 Apple Silicon 集群具备良好的可扩展性。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **Thunderbolt 拓扑限制** | 全连接 mesh 拓扑下，每个 peer pair 最多只能分配 2–3 条 trunk（受 6 port/node 限制） |
| **缺乏 GPU-direct RDMA** | 当前 all-reduce 在 CPU 上执行，无法绕过主机内存，限制了进一步提速 |
| **仅支持 UC Queue Pairs** | 不支持 Reliable Connection（RC），缺少自动重传、流控等机制，可靠性较低 |
| **生态系统尚不成熟** | MLX 生态相比 PyTorch/CUDA 仍较初级，工具链和调试支持有限 |

### 未来工作方向
1. **探索 GPU-offloaded communication**：将 all-reduce 计算卸载到 GPU，减少 CPU-GPU 数据拷贝。
2. **开发更高效的 collective 算法**：适配 Thunderbolt 的点对点特性，设计基于 tree 或 butterfly 的 reduce-scatter/gather 方案。
3. **支持 fault tolerance 机制**：针对 UC 链接不可靠问题，引入 checksum、重发等容错策略。
4. **异构训练架构探索**：结合 Apple Silicon 的大内存与云端 H100 的高算力，构建 hybrid training pipeline。
5. **自动化 trunk 调度与负载均衡**：动态调整消息分片策略以应对链路波动。

---

> **总结一句话**：  
> 本文证明了 **Apple Silicon + RDMA-over-Thunderbolt** 在经过系统级通信优化后，能够成为一种**高效、低成本、适合私有化部署的 LLM 微调平台**，尤其在 **memory-intensive、long-sequence** 场景下具有独特优势。

</details>

---

### 6. [Beyond Static RAG: An Adaptive, Tri-Metric Routing Framework for Efficient Long-Context Inference on Commodity GPUs](https://arxiv.org/abs/2609.17564)

**Authors**: Saipraveen Vabbilisetty, Ajay Kumar Boddepalli, Deep Narayan Mishra, Shashank Kapadia, Haoan Wang, Anupriya Sharma  
**Category**: cs.LG  
**Published**: 2026-09-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2609.17564v1  

#### Abstract
Deploying retrieval-augmented generation (RAG) on commodity GPUs such as the NVIDIA T4 (16 GB VRAM) exposes a practical failure mode we call the Compression Paradox: neural prompt compression can add key-value (KV) cache contention and preprocessing latency that outweigh generation-time savings, whi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Static RAG: An Adaptive, Tri-Metric Routing Framework for Efficient Long-Context Inference on Commodity GPUs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对在**消费级GPU**（如 NVIDIA T4，16GB VRAM）上部署 **Retrieval-Augmented Generation (RAG)** 系统时面临的“**压缩悖论**”（Compression Paradox）问题。

该悖论表现为：
- **启用神经压缩器**（如 LLMLingua-2）会引入额外的编码延迟和显存竞争，导致总延迟反而高于不压缩；
- **不启用压缩器**则长输入会导致 **Out-of-Memory (OOM)** 崩溃，尤其是在 vLLM 和 PyTorch 共驻时发生 **KV-cache block pool 耗尽** 或 **跨进程 CUDA OOM**。

这一矛盾使得传统静态压缩策略失效。

---

### 提出的新方法与新思路
作者提出 **Tri-Metric Router** —— 一种**确定性、无需训练、无辅助模型**的动态路由框架，根据三个 CPU 可计算的轻量信号，在以下三种 pipeline 中进行选择：

| Pipeline | 描述 |
|--------|------|
| `Raw` | 不压缩，直接传入完整上下文 |
| `Neural` | 使用 LLMLingua-2 进行神经压缩 |
| `Lexical` | 使用 BM25 进行基于关键词的稀疏检索 |

#### 三大路由信号（Tri-Metric）
1. **L（Spatial Complexity）**：输入长度（词数），用于判断是否超过延迟收益拐点 $L^*$。
2. **p_key（Syntactic Density）**：关键词密度，高值表示结构化内容（如代码、公式），避免神经压缩破坏语法。
3. **TTR（Type-Token Ratio）**：词型/词符比，低 TTR 表示语义冗余高，适合压缩；高 TTR 则保留原始内容。

> ✅ **核心创新**：将适应信号从传统的“语义复杂度”转向“硬件物理状态”（VRAM headroom、latency crossover），实现**硬件感知的推理调度**。

---

### 相比现有方法的优势
| 维度 | 本工作 | 现有方法 |
|------|-------|---------|
| **适应机制** | 基于硬件物理约束（VRAM、延迟交叉点） | 基于语义复杂度或固定规则 |
| **训练需求** | 完全无需训练或微调 | 多数需学习路由网络（如 RouteLLM） |
| **显存安全** | 提供确定性 OOM 防护（尤其 Long 分支） | 无显存保障机制 |
| **系统开销** | 零额外 VRAM 占用，纯 CPU 决策 | 引入额外模型增加负担 |
| **部署灵活性** | 参数可迁移至不同硬件重新校准 | 多为特定配置设计 |

此外，提出了 **VRAM Partitioning Framework**，定义了 **Goldilocks Zone**（u ∈ [0.50, 0.80]），确保 vLLM 与 PyTorch 模型共驻时不冲突。

---

## 2. 核心实验方法和设置

### 数据集
- 主要使用 **LongBench** 中的 `qasper` 子集（N=100），包含学术问答文档，按长度分为三段：
  - Short: L < 1,500
  - Medium: 1,500 ≤ L ≤ L\*
  - Long: L > L\*
- Out-of-Distribution (OOD) 测试使用 **multifieldqa_en**（N=50），验证泛化能力。

---

### 实验设置
- **硬件平台**：单张 NVIDIA T4（16GB GDDR6）
- **LLM**：Llama-3-8B-Instruct-AWQ（~9GB 显存占用，含 weights + KV-cache）
- **神经压缩器**：LLMLingua-2（XLM-RoBERTa-Large，~2.5GB）
- **词法压缩器**：BM25（CPU-only，输出截断至 4096 tokens）

#### VRAM 分配策略
- 固定 vLLM 显存利用率为 55%（即 8.8GB），预留 7.2GB 给 PyTorch。
- 通过初始化顺序控制资源竞争：先启动 vLLM，再加载 LLMLingua-2。

---

### 评估指标
| 指标 | 说明 |
|------|------|
| **OOM Rate** | 显存崩溃比例（越低越好） |
| **End-to-End Latency (E2E)** | 总响应时间（越低越好） |
| **Combined F1** | 对所有样本（含崩溃者）赋 F1=0，消除幸存者偏差 |
| **Oracle Alignment** | 路由决策与“后见最优”一致的比例（越高越好） |

---

### 基线方法对比
| Baseline | 描述 |
|--------|------|
| Always-Raw | 不压缩，仅用 vLLM |
| Always-Neural | 始终使用 LLMLingua-2 压缩 |
| Always-Lexical | 始终使用 BM25 截断至 4096 tokens |
| Tri-Metric Router (Ours) | 动态三路路由 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（LongBench qasper, T4）

| System | OOM Rate | E2E (s) | Combined F1 | Oracle Align. |
|--------|----------|---------|-------------|----------------|
| Always-Raw | 65.0% | 9.27° | 15.1±2.1% | – |
| Always-Neural | 18.3% | 8.41 | 34.6±1.8% | 76.4% |
| Always-Lexical | 0.0% | 8.14 | 47.8±1.5% | 83.1% |
| **Tri-Metric Router (Ours)** | **0.0%** | **7.91±0.3** | **51.7±1.2%** | **99.0%** |

> 注：° 表示仅在非崩溃样本上测量，存在幸存者偏差。

---

### 核心优势总结
- **0% OOM**：首次实现完全显存安全的多组件 RAG 推理。
- **F1 提升显著**：相比 Always-Lexical 提升 **+3.9 pts**，相比 Always-Neural 提升 **+17.1 pts**（考虑崩溃惩罚）。
- **接近理论最优**：在分布内达到 **99.0% Oracle Alignment**，表明路由几乎总是做出最佳选择。

---

### 消融实验结果

#### Ablation A：p_key 与 TTR 守护机制的影响（ID, N=6）
- 在 medium-band 中有 6 个文档因高 p_key 或高 TTR 被路由到 `Raw`（主要是数学公式密集文本）。
- 若强制改用 `Neural`，F1 下降 2.1±1.4 pts。
- 结果虽未达统计显著（N 小），但机制层面证实：神经压缩会删除关键符号（如运算符），造成不可逆损坏。

#### Ablation B：仅基于长度的路由器（OOD, N=50）
- 移除 p_key/TTR 判断，所有 medium 输入都走 `Neural`。
- 导致 OOD Combined F1 从 **49.3% → 47.8%**，下降 1.5 pts。
- 说明次级信号对保护结构化内容具有实际价值。

#### 压缩率敏感性分析
- 最优压缩率 r=0.5：
  - r=0.3（更强压缩）→ F1 降至 46.0%（事实被删）
  - r=0.7（更弱压缩）→ F1 降至 48.0%（噪声干扰）
- 支持采用中等压缩强度。

---

## 4. 关键结论和发现

### 主要发现
1. **Compression Paradox 是真实存在的系统瓶颈**：
   - 在资源受限设备上，压缩可能“得不偿失”，甚至引发更严重的 OOM。
2. **硬件物理信号优于语义信号用于调度**：
   - VRAM headroom 和 latency crossover 是决定是否压缩的关键因素。
3. **Tri-Metric Router 实现高效且鲁棒的自适应推理**：
   - 在保持 0% OOM 的同时，获得接近 oracle 的性能表现。
4. **Goldilocks Zone 是多组件共驻的前提**：
   - vLLM 显存利用率应控制在 [0.50, 0.80] 区间以平衡 KV-cache 与外部模型需求。

---

### 局限性
| 方面 | 限制 |
|------|------|
| **硬件依赖性** | 当前参数（如 $L^*=4332$）针对 T4 校准，需重新标定才能迁移到其他 GPU |
| **统计置信度** | $L^*$ 的估计基于 N=21 长文档，bootstrap CI 较窄（±185），但采样方差可能更大（建议 N≥100） |
| **单一模型设定** | 仅测试 Llama-3-8B + LLMLingua-2 组合，未覆盖多语言、代码等场景 |
| **批处理支持缺失** | 所有实验基于 B=1；当 B>4 时，Neural 分支可能消失（显存不足） |
| **未集成 KV-cache 优化技术** | 如 SnapKV、H2O、StreamingLLM 等无法在预调度阶段调用 |

---

### 未来工作方向
1. **跨硬件校准**：在 L4、A10G、RTX 4080 等设备上重新运行 calibration 流程。
2. **联合优化压缩率与路由边界**：构建 $(L^*, r)$ 联合优化表面。
3. **在线自适应阈值**：利用轻量级 VRAM 监控器动态调整路由参数。
4. **扩展 p_key 至多语言与代码领域**：提升对结构化内容的识别能力。
5. **集成 KV-cache eviction 技术**：作为 Long 分支内部机制进一步释放内存压力。
6. **批量推理评估**：研究 $B \in [2,4]$ 下 Neural 分支的生存空间。

---

> 📌 **总结一句话**：  
> 本文揭示了消费级 GPU 上 RAG 推理中的“压缩悖论”，并提出首个基于硬件物理信号的 **Tri-Metric Router**，实现了 **零 OOM、高性能、无需训练** 的自适应长上下文推理，为低成本、高可靠性的 agentic system 部署提供了新范式。

</details>

---

### 7. [WFM: Wiki Foundation Model for Complex Agentic Reasoning](https://arxiv.org/abs/2609.18182)

**Authors**: Junnan Dong, Linhao Luo, Senlei Zhang, Gong Chen, Taian Guo, Yifei Yu, Rong Tao, Tao Guo, Qian-Wen Zhang, Siyu An, Ruizhi Qiao, Xing Sun  
**Category**: cs.AI  
**Published**: 2026-09-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.18182v1  

#### Abstract
Real-world agents fundamentally require persistent non-parametric knowledge for dynamic reasoning, i.e., long-term memory and retrieval-augmented generation. While graphs have shown reliable advantages in providing structured evidence, the sparse graph representations naturally restrict machine read...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：WFM: Wiki Foundation Model for Complex Agentic Reasoning

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

传统 **GraphRAG** 和稀疏知识图谱（KG）在复杂 **Agentic Reasoning**（智能体推理）任务中存在以下根本缺陷：

- **语义密度不足**：将文本压缩为 `(head, relation, tail)` 三元组会丢失丰富的上下文语义和跨文档连续性。
- **注意力坍塌（attention collapse）**：在高密度 LLM Wiki 上应用传统 GNN 会导致多头注意力权重趋于均匀，造成梯度锁死（gradient lock），阻碍表示学习。
- **系统扩展瓶颈**：分布式训练中频繁的 CPU 序列化和内存拷贝导致通信开销巨大，难以在大规模商业场景部署。

### **提出了什么新方法或新思路**

作者提出 **Wiki Foundation Model (WFM)**，一种面向 **agent-native** 的新型基础模型范式，核心创新包括：

#### ✅ **(1) 双层 Wiki Graph Schema**
- 将 LLM Wiki 形式化为混合图结构：节点包含 **实体（entity）** 和 **文本段落（passage）**。
- 引入 **跨层超边（cross-layer hyper-edges）** 连接实体与其相关段落，保留细粒度拓扑结构的同时融合密集文本语义。

#### ✅ **(2) 查询条件化的注意力聚合机制 + 注意力方差正则化（Lvar）**
- 设计 **relation-aware attention**，支持实体-实体、实体-段落之间的统一消息传递。
- 提出 **显式注意力方差正则化损失（Lvar）**，防止 Softmax 输出趋同，数学上消除注意力坍塌。

#### ✅ **(3) NCCL 原生边界交换协议（NCCL-native boundary exchange）**
- 预先离线计算图分区索引，利用固定形状的 GPU-to-GPU 通信（如 `NCCL_AllToAll`），完全绕过 CPU 序列化与主机内存拷贝。
- 实现训练延迟从 **2.40s → 0.23s/step**，端到端加速达 **10.5×**。

#### ✅ **(4) 温启动课程学习（Warm-Start Curriculum）**
- 分两阶段训练：
  1. 冻结 GFM 参数，仅优化文档对齐；
  2. 解冻所有参数进行联合优化。
- 有效避免冷启动时的优化不稳定。

---

### **相比现有方法的优势**

| 维度 | 优势 |
|------|------|
| **表示能力** | 融合稀疏拓扑与密集文本，优于纯三元组或纯向量检索 |
| **推理能力** | 支持多跳、迭代式、自省的 agentic 推理路径 |
| **训练效率** | 10.5× 加速，解决分布式系统瓶颈 |
| **通用性** | 支持 zero-shot 跨域迁移与端到端多跳推理 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### 🔹 **多跳问答（Multi-hop QA）**
- **HotpotQA**（1,000 问题）
- **2WikiMultihopQA**（1,000 问题）
- **MuSiQue**（1,000 问题）
> 所有数据集均标注支持文档，测试跨文档组合推理能力。

#### 🔹 **长周期记忆问答（Long-horizon Memory QA）**
- **RHELM**：百万级上下文，涵盖事实、时间、聚合、误导等七类问题。
- **PersonaMem 1M**：来自 20 个用户的 2,674 个问题，测试个性化偏好演化记忆。

---

### **实验设置和评估指标**

#### 📊 **评估指标**

| 任务类型 | 指标 |
|--------|------|
| **多跳 QA** | - `Recall@k`（检索覆盖率）<br>- `ACC`（LLM 判断准确率）<br>- 区分 **Open Mode**（允许参数知识）与 **Reject Mode**（仅依赖检索证据） |
| **记忆 QA** | - `ACC`（总体准确率）<br>- `Recall@5/10/20`（检索召回率） |

#### ⚙️ **实现细节**
- 回答生成模型：DeepSeek V4 Flash
- 判别模型：DeepSeek V4 Pro
- 嵌入模型：all-MiniLM-L6-v2
- 检索深度：20
- WFM 层数：3
- 自反预算（self-reflection budget）：B = 4
- 方差阈值 ε = 0.05，正则权重 λ₂ = 0.1

---

### **基线方法对比**

#### 🔹 **多跳 QA 基线**
- **Zero-shot LLM**：无检索生成
- **Native RAG**：扁平稠密检索
- **Hierarchical Retriever**：RAPTOR, E2GraphRAG
- **Graph Retriever**：LightRAG, GraphRAG, HippoRAG, Youtu-GraphRAG, GFM-RAG

#### 🔹 **记忆 QA 基线**
- 同上图检索器
- 专用记忆系统：A-mem, MemoryOS, LightMem
- 全上下文零样本基线

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **多跳 QA 结果（Table 2）**

| 方法 | HotpotQA (Open/Reject) | 2Wiki (Open/Reject) | MuSiQue (Open/Reject) |
|------|------------------------|--------------------|-----------------------|
| Youtu-GraphRAG | 86.8 / 80.2 | 87.0 / 77.6 | 65.7 / 47.5 |
| **WFM** | **89.6 / 84.3** | **90.2 / 82.4** | **69.8 / 52.6** |
| ↑ 提升 | +2.8 / +4.1 | +3.2 / +4.8 | +4.1 / +5.1 |

> 在 **Reject 模式下提升更显著**，说明 WFM 更强地依赖检索证据而非幻觉补全。

#### ✅ **检索召回率（Table 1 & 4）**

| 数据集 | 方法 | R@20 |
|-------|------|------|
| HotpotQA | WFM | **93.20**（vs. Youtu-GraphRAG 89.70） |
| 2Wiki | WFM | **90.15**（vs. 88.50） |
| MuSiQue | WFM | 75.24（接近最优 75.90） |
| PersonaMem | WFM | **52.63**（vs. A-mem 38.2） |
| RHELM | WFM | **60.03**（vs. A-mem 53.9） |

> WFM 在所有数据集上取得最高或接近最高的 Recall@20，尤其在 PersonaMem 上领先 **14.43 个百分点**。

---

### **消融实验结果（Ablation Study）**

#### 🔽 移除组件的影响（Figure 5）

| 消融项 | 平均 Recall@20 ↓ | 平均 ACC ↓ | 成本变化 |
|--------|------------------|------------|----------|
| 普通图（无段落节点） | -7.48 pts | -7.68 pts | — |
| DistMult 替代注意力 | -11.89 pts | -12.47 pts | ×2.65 更慢 |
| 移除 Lvar 正则 | 显著下降 | 显著下降 | — |
| 无温启动训练 | 下降 | 下降 | — |
| 移除 self-reflection | -3.82 pts | -3.82 pts | 更便宜但弱 |
| 移除 final-answer flag | 效果恢复 | 成本 ↑1.58× | 浪费计算资源 |

> 结论：**段落节点、注意力机制、Lvar、温启动、自反循环** 均不可或缺。

#### 🔍 参数敏感性分析（Figure 3）

- **最佳层数 L=3**：更深反而性能下降（过度平滑）。
- **自反预算 B=4**：平均执行轮次仅 2.58，因 adaptive stopping 提前终止。
- **方差正则参数 ε=0.05, λ₂=0.1** 为最优配置。

---

## 4. 关键结论和发现

### **主要发现**

1. **LLM Wiki 是下一代 agent-native 知识表示范式**  
   相较于传统稀疏 KG，其融合密集文本与显式拓扑的能力更适合复杂推理。

2. **WFM 实现了表示、推理与系统的垂直统一**  
   数学设计（双空间注意力 + Lvar）与硬件实现（NCCL 协议）协同优化，突破性能瓶颈。

3. **注意力坍塌是高密度图学习的关键障碍**  
   Lvar 正则与温启动课程能有效防止梯度锁死，保障训练稳定性。

4. **迭代式自省（iterative self-reflection）显著提升长程记忆推理能力**  
   特别适用于需多步证据收集的任务（如追踪、修订类问题）。

5. **系统级优化带来数量级加速**  
   NCCL 原生通信使训练延迟降低 **10.5×**，具备工业级可扩展性。

---

### **方法的局限性**

- 当前 WFM 仍基于静态 Wiki 构建，尚未支持实时动态更新。
- 多跳推理依赖预定义图结构，对开放世界未知关系泛化有限。
- 虽然训练加速明显，但推理阶段仍涉及多次检索-生成循环，延迟较高。

---

### **未来工作方向**

- 扩展至 **real-time dynamic tasks**，支持在线增量学习。
- 探索 WFM 在更广泛 **complex agentic reasoning** 场景中的泛化能力（如规划、工具调用）。
- 结合强化学习进一步优化自反策略与停止机制。

---

> 💡 **一句话总结**：  
> WFM 通过构建融合密集文本与显式拓扑的 **Wiki Graph**，结合 **注意力方差正则化** 与 **NCCL 原生通信协议**，首次实现了高效、可扩展、agent-native 的复杂推理基础模型，在多跳问答与长周期记忆任务上全面超越 SOTA，推动 RAG 范式向 LLM Wiki 演进。

</details>

---

### 8. [Where Should Agents Live? Energy-Memory Characterization of Agentic AI for the Edge-Cloud Continuum](https://arxiv.org/abs/2609.18283)

**Authors**: Carolina Fortuna, Vid Han\v{z}el, Tim Strnad, Bla\v{z} Bertalani\v{c}  
**Category**: cs.AI  
**Published**: 2026-09-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.18283v1  

#### Abstract
As telecommunication networks evolve toward autonomous 5G-Advanced and 6G operations, agentic artificial intelligence (AI) workflows, where large language models (LLMs) execute multi-step reasoning, invoke diagnostic tools, retrieve domain knowledge, and coordinate across agent teams, are increasing...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Where Should Agents Live? Energy-Memory Characterization of Agentic AI for the Edge-Cloud Continuum*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
随着电信网络向 5G-Advanced 和 6G 自主化演进，**Agentic AI**（即由多个 LLM 代理组成的多步推理、工具调用、知识检索和团队协作的工作流）被广泛部署在 **edge-cloud continuum** 上。然而，当前缺乏对这类分布式多代理工作流的系统性 **能量-内存** 特征建模，导致运营商无法科学决策代理应部署在边缘还是云端。

现有 AI 生命周期度量（如 eCAL）仅适用于单次模型推理，忽视了多代理间的通信开销、上下文累积和异构组件能耗，因此无法指导实际部署。

### 提出的新方法与新思路
本文提出 **agentic-eCAL** —— 一种面向多代理 AI 工作流的端到端能量度量框架，其核心创新包括：

- **推广 eCAL 至多代理场景**：将原始 eCAL 指标从单次推理扩展为支持 **有向图形式的 multi-agent workflow**，涵盖 LLM 推理、工具执行（Tool）、检索增强生成（RAG）、7层 OSI 数据传输及预训练“具身能量”（embodied energy）的摊销。
  
- **建立两速率单次调用能量模型**：
  - **Prefill 阶段**：计算密集型（compute-bound），能量正比于提示长度 $P_{in}$。
  - **Decode 阶段**：内存带宽受限（memory-bound），能量随批处理大小 $b$ 增大而下降（$\propto 1/b$）。
  - 公式：$E_{call}(P_{in}, P_{out}; b) \approx C_{pre} P_{in} + C_{dec}(b) P_{out}$

- **揭示上下文累积导致的超线性能量增长**：由于历史对话需在每一步重新 Prefill，总提示长度呈二次增长，导致整体能量随推理轮数 $K$ 超线性上升（scaling exponent $a \in [1,2]$）。

- **量化跨层通信能耗占比极低**：通过实测验证，**跨代理文本传输能耗仅占全流程 <0.25%**，远低于传统 offloading 模型假设。

### 相比现有方法的优势
| 维度 | 现有方法（如 eCAL） | 本文 agentic-eCAL |
|------|---------------------|------------------|
| 适用对象 | 单模型单次推理 | 多代理有向图工作流 |
| 能耗建模 | 忽略上下文累积与通信图结构 | 显式建模 Prefill 超线性增长与 KV-cache 开销 |
| 通信建模 | 简化为比特级传输 | 基于 7 层 OSI 协议栈的物理能耗模型 |
| 决策依据 | 计算 vs 通信权衡 | **内存容量与上下文管理为主导因素** |

---

## 2. 核心实验方法和设置

### 实验平台与硬件配置
- **GPU 平台**：NVIDIA A100 (80GB) 和 H100 (NVL) 加速器
- **推理引擎**：vLLM（支持 Continuous Batching 和 PagedAttention）
- **测量工具**：NVML 实时采集 GPU 功耗
- **模型范围**：覆盖 **16 个开源权重 LLMs**，参数量从 3B 到 72B（含 MoE 架构如 GPT-OSS-20B）

### 数据集与任务
- **基准测试任务**：基于 **Kubeply 的 Infra-Bench**，一个面向云原生网络功能（CNF）的基础设施故障诊断与修复任务集。
  - 包含 24 个真实 Kubernetes 集群故障案例（服务路由、权限控制等）
  - 每个任务限时 60 分钟，最多允许 50 步操作
- **代理拓扑结构**：评估 **8 种典型 orchestration 架构**（见 Fig. 6）：
  - Star, Persona-Star, Proposer-Critic, Tournament, Tree, Chain, Cascading-Chain, Diamond

### 评估指标
| 指标 | 定义 |
|------|------|
| `agentic-eCAL` | $\frac{E_w + \gamma_v(E_{emb} + E_{emb,ret})}{B_{useful}}$ [J/bit]，综合生命周期能耗 |
| `E_w` | 工作流运行期能耗（LLM + Tool + RAG + Transmission） |
| `E_{emb}` | 模型预训练“具身能量”摊销（来自 Llama 2/3 等官方报告） |
| `Task Success Rate` | 成功修复的故障比例 |
| `Energy per Solved Task` | 总能耗 / 成功任务数 |

### 基线方法对比
- **Single-Agent Baseline**：单一 ReAct 代理完成全部任务
- **Multi-Agent Topologies**：上述 8 类架构中的 5 种代表：
  - W+C（Worker + Critic）
  - W+V（Worker + Verifier）
  - PS（Parallel Proposers + Synthesizer）
  - PS+V（PS + Verification）
  - P+R+W+V（Planner-Reviewer-Worker-Verifier Pipeline）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）能量模型准确性验证
- 在 Qwen2.5-7B 和 Llama-3.1-8B 上，预测 vs 实测能量相关系数 $R^2 \geq 0.99$，平均绝对误差 MAPE ≈ 10%
- 表 II 显示模型估算值与 ML.ENERGY 基准基本一致（考虑设备空闲功耗后）

#### （2）上下文携带 vs 清除的能耗差异（Fig. 3a）
| 模型 | K=6 回合携带历史 vs 无历史 | 能耗增幅 |
|------|----------------------------|--------|
| Qwen2.5-7B | 89 J vs 67 J | +32.8% |
| Llama-3.1-8B | 185 J vs 152 J | +21.7% |

> 证明 **history-carrying loops 导致显著超线性能量增长**

#### （3）跨代理通信能耗占比（Table III）
| 网络承载层 | 传输能耗占总流程比例 |
|----------|------------------|
| Optical Backbone ($10^{-10}$ J/b) | <0.001% |
| Metro/Fixed ($10^{-9}$ J/b) | <0.001% |
| 5G RAN ($10^{-8}$ J/b) | 0.0001% – 0.002% |
| Loaded Edge Cell ($10^{-7}$ J/b) | 最高仅 **0.02%** |

> **文本级通信能耗可忽略不计**

#### （4）多代理部署能效对比（Fig. 9）
| 模型 | 配置 | 能耗倍增 | 任务成功率提升 |
|------|------|--------|-------------|
| Qwen3.5-9B | P+R+W+V vs Single | ×2.9 | 61.7% → 64.6% (+2.9pp) |
| Qwen2.5-7B | 所有多代理方案 | ×up to 23.9× | 成功率反而下降（9.2% → 最高 4.6%） |

| 模型 | 方法 | Energy per Solved Task |
|------|------|-----------------------|
| Qwen3.5-9B | Single | **17.4 kJ** |
| | Multi-Agent Avg. | 29.0 – 79.2 kJ |
| Qwen2.5-7B | Single | **49.7 kJ** |
| | Multi-Agent Max | up to **5222.3 kJ** |

> 多代理不仅未提升成功率，反而大幅增加单位有效任务能耗。

#### （5）KV Cache 迁移代价极高（Fig. 8a）
- 传输完整 KV Cache：**18.9 Gb/session**
- 传输原始文本：仅 **0.33 Mb/session**
- 前者是后者的 **~57,000 倍**

> **KV-cache 不可跨网络迁移**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **跨代理文本通信能耗极低**（<0.25%），传统“计算 vs 通信”权衡模型不再成立。
2. 🔥 **真正的能耗瓶颈在于上下文处理**：尤其是 Prefill 阶段因历史积累导致的 **超线性能量增长**。
3. 💾 **部署可行性由内存决定**：特别是 KV-cache 容量需求（bytes/token）而非模型大小主导部署密度。
4. ⚠️ **盲目增加代理数量会显著降低能效**：在多数情况下并未提升任务成功率，却使能耗激增（最高达 **23.9×**）。
5. 📉 **KV-cache 无法跨节点共享**：其传输开销远超本地计算成本，在所有蜂窝链路上均不可行。

### 方法的局限性
- 当前模型假设 **无 prefix caching**，若启用 APC 或 RadixAttention 可缓解 Prefill 开销。
- 对 **MoE 架构** 中 active parameters 的建模尚不完善。
- 缺乏对新型硬件（FP8/FP4 Tensor Core, Unified Memory）的适配。
- 具身能量摊销依赖全局调用量估计 $G$，难以精确追踪开放模型的实际部署规模。

### 未来工作方向
- 将 **prefix cache hit rate** 纳入 Prefill 能耗建模
- 支持更多 serving stacks（如 TensorRT-LLM）、加速器（TPU, Groq）
- 引入 **speculative decoding** 等优化技术的能量建模
- 开展 **全栈 agentic workload 的直接能耗测量**
- 探索基于任务收益的 **energy-aware agent selection 机制**

---

> **最终结论**：  
> “Where should agents live?” —— 不应基于通信延迟或带宽，而应基于 **内存容量、KV-cache 管理效率与实际任务增益** 来决策。  
> **Less is more**: 更少但更智能的代理，配合增量式上下文传递（incremental hand-off）和本地状态保留，才是可持续 Agentic AI 的未来。

</details>

---

### 9. [SSD-LLaMA: SSD-Native Inference for Trillion-Parameter MoE at 1+ Token/s on a Consumer PC](https://arxiv.org/abs/2609.18110)

**Authors**: Fangzhou Liang, Yibin Shen, Jianmin Hu, Jiayang Xu, Hanchi Gao, Minxian Xu, Zili Meng  
**Category**: cs.DC  
**Published**: 2026-09-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2609.18110v1  

#### Abstract
Frontier open-weight language models increasingly use Mixture-of-Experts (MoE) architectures to expand model capacity while activating only a small subset of experts per token. Local inference must nevertheless keep the complete expert pool available, which remains far beyond consumer-grade RAM and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SSD-LLaMA: SSD-Native Inference for Trillion-Parameter MoE at 1+ Token/s on a Consumer PC

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前前沿的 **Mixture-of-Experts (MoE)** 大模型虽然通过稀疏激活机制降低单个 token 的计算量，但其完整的专家池（expert pool）仍远超消费级设备的 **RAM 和 VRAM 容量**。即使经过量化，也无法在普通 PC 上实现本地推理。传统方法依赖内存映射（mmap）、预取（prefetching）或 CPU 执行，存在以下问题：
- **SSD I/O 效率低**：专家权重分散存储，导致小而碎片化的读取请求，无法充分利用 SSD 高带宽。
- **存储层级割裂**：缺乏对 SSD、RAM、VRAM 的统一管理，难以协调数据流动与缓存策略。
- **CPU-GPU 负载失衡**：将 RAM 中的专家绑定到 CPU 执行，导致 GPU 空闲，形成瓶颈。

### 提出了什么新方法或新思路
SSD-LLaMA 是一个面向消费级硬件的 **SSD-Native MoE 推理系统**，提出三大核心设计：

#### ✅ **高效的 SSD I/O Pipeline**
- **Expert-Pack Layout**：将每个专家的 `gate`, `up`, `down` 张量打包为连续的 SSD 块，并建立索引表，支持 O(1) 查找和单次大块读取。
- **并发直接读取（Direct I/O）**：使用 `io_uring` 实现异步批量提交 SSD 请求，提升吞吐。
- **异步 H2D 传输**：读取完成后立即开始向 VRAM 传输，无需等待所有请求完成。
- **Lossless GPU Decompression**：采用 rANS 对专家权重进行无损压缩，在 GPU 上用 CUDA 内核直接解压，减少 H2D 流量。

#### ✅ **原生三层次存储架构（SSD-RAM-VRAM Hierarchy）**
- SSD 存储完整专家池；
- RAM 和 VRAM 作为动态缓存层，基于访问频率和时间（LRU-like）保留热专家；
- 缓存决策独立于预取预测，更适应实际路由模式。

#### ✅ **平衡的 CPU-GPU 混合执行**
- **解耦权重驻留与执行位置**：不强制 RAM-resident 专家必须由 CPU 执行；
- 动态选择执行设备：若能避免同步开销，则优先在 GPU 上执行，即使需 H2D 传输；
- CPU 负责 I/O 调度、缓存管理和任务分发，GPU 主导计算。

### 相比现有方法的优势
| 方面 | 传统方法（如 llama.cpp, KTransformers） | SSD-LLaMA |
|------|----------------------------------------|-----------|
| I/O 效率 | mmap 导致页错误、碎片读取 | 单次大块读取，接近 SSD 峰值带宽 |
| 缓存机制 | 固定放置或简单预取 | 动态三级缓存，按热度保留专家 |
| 执行调度 | RAM→CPU 绑定，GPU 利用不足 | 灵活分配，最大化 GPU 利用率 |
| 可扩展性 | 难以运行 >500GB 模型 | 支持 **Trillion-parameter 级模型** |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MMLU**：多任务知识与推理基准，平均提示较长，适合测试 **prefill 阶段并行性**。
- **Alpaca**：指令跟随数据集，用于评估通用对话场景下的推理性能。

### 实验设置
- **硬件平台**（Consumer PC）：
  - CPU: Intel Ultra5 230F
  - GPU: NVIDIA RTX 5090（32GB VRAM）
  - RAM: 16GB DDR5
  - SSD: 4TB NVMe PCIe 5.0（理论读速 9 GiB/s）
  - 互连: PCIe 5.0 x16（64 GB/s）

- **模型**（三个前沿 MoE 家族）：
  | Model | Total Params | Activated per Token | Quantization | #Experts | #Layers |
  |-------|--------------|---------------------|---------------|----------|---------|
  | DeepSeek-V4-Flash | ~145.4 GiB | 111.6 GiB | MXFP4 | 256×43 | 43 |
  | Kimi-K2.7-Code | ~543.6 GiB | 19.7 GiB | Q4_0 | 384×61 | 61 |
  | GLM-5.2 | ~401.0 GiB | 23.4 GiB | Q4_0 | 256×79 | 79 |
  | Kimi-K3 | **2.8T 参数** (~1.4TB @4bit) | —— | Q4_0 | —— | —— |

### 评估指标
- **TTFT (Time to First Token)**：请求到首 token 输出的时间。
- **TPOT (Time Per Output Token)**：解码阶段每 token 平均延迟。
- **Decode Throughput (tokens/s)**：生成速度。
- **SSD Read Bandwidth**：实际达到的 SSD 读取带宽。
- **VRAM Hit Rate**：专家缓存命中率。
- **Per-Token I/O Traffic**：每生成一个 token 从 SSD 读取的数据量。

### 基线方法对比
1. **llama.cpp**：轻量级 CPU-first 框架，广泛用于低资源设备，依赖 mmap 和固定加载。
2. **KTransformers**：专为 MoE 设计的异构框架，利用 AMX 加速 CPU 计算，支持专家延迟执行。
3. **Colibri**：基于 SSD 动态流式加载的系统，支持 INT4 压缩和缓存，强调内存可行性而非速度。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | Prefill Throughput (vs baseline) | Decode Throughput (vs baseline) | 最高 Decode 速率 |
|------|-------------------------------|------------------------------|------------------|
| DeepSeek-V4-Flash | ↑1.52×–4.19× | ↑2.10×–15.58× | —— |
| Kimi-K2.7-Code | ↑1.90×–3.61× | ↑1.35×–3.44× | **1.03 tokens/s** (RTX 5090 + 32GB RAM) |
| GLM-5.2 | ↑1.06×–2.92× | ↑1.57×–15.61× | —— |
| **Kimi-K3 (2.8T)** | **1.217 tokens/s (prefill)** | **0.465 tokens/s (decode)** | 较 llama.cpp 提升 **24.1× / 17.2×** |

> 💡 在 **单张 RTX 5090 + ≤32GB RAM** 上实现了 **>1 token/s 的万亿参数模型推理能力**。

### 与基线方法的对比结果
- **SSD Read Bandwidth**：
  - SSD-LLaMA 达到 **6.49–7.77 GiB/s**，占物理上限（9 GiB/s）的 **77.7%**；
  - 基线最高仅达 **3.89 GiB/s（43.2%）**；
  - 带宽提升达 **2.82× (vs llama.cpp)** 和 **1.93× (vs KTransformers)**。

- **Per-Token I/O Traffic**：
  - 减少高达 **1.8× vs llama.cpp**, **1.5× vs KTransformers**；
  - 得益于 expert-pack 和缓存机制，显著降低冗余读取。

- **端到端性能优势**：
  - 在 decode 阶段，相比 KTransformers 最高提速 **15.61×**；
  - 相比 Colibri，在 GLM-5.2 上仍快 **1.78×–2.09×**，且 I/O 更高效。

### 消融实验结果
#### 🔹 Expert-Pack 有效性（图12）
- 仅加载 `gate` 张量时性能较低；
- 全部三个张量都来自 expert-pack 时，吞吐提升 **2.6×–4.1×**；
- 表明 **连续布局是高性能 I/O 的关键**。

#### 🔹 VRAM Cache 策略影响（图13）
- 不启用缓存（None）时性能最差；
- 仅缓存 `gate` 或 `gate-up` 虽然命中率更高，但因未缓存部分仍需频繁 I/O，整体吞吐反而下降；
- **缓存全部三个张量（all）带来 2.78× 吞吐提升**，证明全专家缓存必要。

#### 🔹 硬件配置影响（图15–17）
- 升级至 **RTX 5090 + 32GB RAM** 可使 Kimi-K2.7-Code 达到 **1.03 tokens/s**；
- 更大 RAM 显著提高缓存命中率，减少 SSD 读取次数；
- SSD 耐久性估算显示：该配置下一块 2400 TBW SSD 可服务约 **458万条 prompt（512 tokens each）**。

---

## 4. 关键结论和发现

### 主要发现
1. **SSD 已成为可行的执行内存后备层**：现代 PCIe 5.0 SSD 的顺序带宽已接近 DDR 内存，结合高效 I/O 设计可支撑 MoE 推理。
2. **Expert-Pack + 异步流水线极大提升了 I/O 效率**：通过物理重组、并发读取、异步传输与 GPU 解压，有效隐藏 SSD 延迟。
3. **动态三级缓存优于静态策略**：基于运行时访问模式保留热专家，比预取或固定加载更能适应 MoE 的稀疏性和动态性。
4. **解耦“驻留”与“执行”可优化负载均衡**：允许 RAM 中的专家在 GPU 上执行，避免 CPU 成为瓶颈。
5. **万亿参数 MoE 模型可在消费级 PC 上实现实用级推理**：首次实现 **>1 token/s 的 decode 速度**。

### 方法的局限性
- **依赖特定硬件特性**：高性能依赖 PCIe 5.0 SSD 和高带宽 CPU-GPU 互联（PCIe x16）。
- **冷启动延迟较高**：首次加载专家仍需较长时间，不适合极短会话。
- **未处理 KV Cache Offloading**：KV cache 仍在 VRAM，限制上下文长度扩展。
- **能量效率未优化**：SSD 频繁读取可能增加能耗（见 Related Work 中 [25]）。

### 未来工作方向
- 结合 **PagedAttention** 等技术实现 **KV Cache 的 SSD 卸载**，进一步突破显存限制。
- 探索 **与 MoE-specific 技术的融合**，如 SMoE 的专家替换、MoE-Infinity 的主动缓存。
- 支持更多 **量化格式与压缩算法**，进一步降低 I/O 开销。
- 适配移动端与边缘设备，推动 **Trillion-parameter 模型的普惠化部署**。

---

> 📌 **一句话总结**：  
> SSD-LLaMA 通过构建 **SSD-native 的三层次专家管道**，首次实现了在消费级 PC 上以超过 1 token/s 的速度运行万亿参数 MoE 模型，标志着大规模语言模型本地推理的重大突破。

</details>

---

### 10. [A Calibrated Instrument for Measuring How Inference Optimizations Affect Output Quality](https://arxiv.org/abs/2609.18005)

**Authors**: Jerry Kaplan  
**Category**: cs.CL  
**Published**: 2026-09-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2609.18005v1  

#### Abstract
Large language model optimization is an active research area, spanning quantization of model weights, early-exit methods for skipping layers, and speculative decoding. Each track uses its own quality measures, typically an idiosyncratic benchmark score. Few approach the measurement precision require...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：A Calibrated Instrument for Measuring How Inference Optimizations Affect Output Quality**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前大语言模型（LLM）推理优化技术（如量化、early exit、speculative decoding）在提升效率的同时可能损害输出质量，但现有评估方法存在严重缺陷：
- **Perplexity** 衡量的是对原始模型 token 的预测能力，而非生成质量；
- **Token-level agreement** 忽略不同 token 错误的语义影响差异；
- **Benchmark accuracy** 掩盖了个别样本从正确到错误的翻转；
- **LLM-as-a-judge** 被广泛使用，但缺乏标准化校准，导致测量偏差。

这些问题使得不同优化技术之间的质量比较不可靠，难以指导实际部署决策。

### **提出了什么新方法或新思路**
本文提出了一种**标准化、可复现的测量仪器（calibrated instrument）**，用于精确评估推理优化对输出质量的影响。其核心创新在于以下五个设计组件：

| 组件 | 描述 |
|------|------|
| **Paired Design with Dual Reference** | 每个 prompt 对原始模型采样两次作为参考，所有优化条件与这两个参考的平均评分进行比较，减少采样噪声。 |
| **Exchangeability Null** | 利用两个参考样本的交换不变性（exchangeability），理论上期望差值为零，用于测量系统噪声。 |
| **Implementation Null** | 使用严格拒绝规则（strict rejection rule）的 speculative decoding 作为“理论上的无损”对照组，若测出非零差异，则说明实现有缺陷。 |
| **Positive Controls** | 引入已知劣化程度的配置（如特定 early exit 设置）作为正向控制，验证仪器能检测到预期损失。 |
| **Pre-specified Equivalence Bound** | 预先设定最小感兴趣效应大小（±0.3 分，7分制），采用双单侧检验（TOST）判断是否“等效”。 |

此外，还引入了 **execution-grounded correctness** 机制，在可验证任务上通过程序执行来客观判断正确性。

### **相比现有方法的优势**
- ✅ **科学严谨性**：借鉴自然科学中的仪器校准思想，确保测量“读数为零时确实为零”。
- ✅ **跨技术可比性**：同一套工具可用于比较不同类别的优化技术（量化 vs. speculative decoding vs. early exit）。
- ✅ **高分辨率与低偏倚**：通过双重参考和多个控制条件，显著降低方差和系统偏差。
- ✅ **领域敏感性揭示**：能够发现优化效果在不同任务领域的巨大差异，而传统代理指标无法捕捉这一点。

---

## **2. 核心实验方法和设置**

### **使用的数据集与领域**
共 **220 个 prompts**，分布在 **5 个领域**：
- **English prose**（40）：解释性文本生成
- **Chinese prose**（40）：中文解释性文本
- **Short Python functions**（60）：代码补全
- **Arithmetic word problems**（40）：算术应用题
- **Hard-verifiable problems**（40）：包含 27 个数学证明/计算题 + 13 个编程任务，具有明确的可执行验证逻辑

> 所有 prompts 和实验材料公开于 GitHub：[https://github.com/jerrykaplan/Calibrated-Instrument](https://github.com/jerrykaplan/Calibrated-Instrument)

### **目标模型**
- 主要目标：**Qwen2.5-7B-Instruct**（Alibaba）
- 复现目标：**Llama-3.1-8B-Instruct**（Meta）

所有模型以 `bf16` 运行作为基准，其他优化版本在同一硬件（RTX 5090）上生成。

### **评估指标**
#### **主指标：LLM Judge 评分**
- 使用 **claude-sonnet-5** 作为 judge，独立打分（blind evaluation）
- 评分标准（7-point Likert scale）：
  - 7: 完整、正确、格式良好
  - 4: 明显事实或逻辑错误、重复、不完整
  - 1: 退化或无意义输出
- 同时记录是否“derailed”（跑题）、是否“recovered”（恢复）

#### **辅助指标**
- **△Rating**：优化臂评分减去双参考均值
- **Confidence Interval**：基于 220 prompts 的抽样误差估计
- **Equivalence Test**：使用 TOST 在 ±0.3 分界内判定是否“无显著差异”
- **Execution-grounded correctness**：对可验证任务自动运行代码或检查最终答案

#### **Judge 可靠性验证**
- 20% 样本由同一 judge 重评 → 86% 完全一致，100% 差异 ≤1 分
- 10% 由 **claude-opus-5** 评审 → Spearman ρ=0.88
- 29% 由 **gemini-3.1-pro** 评审 → ρ=0.80，排序一致性高

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Qwen2.5-7B）**

| Arm | Pooled ΔRating | Verdict (±0.3) | Hard Domain Δ |
|-----|----------------|----------------|---------------|
| **reference sample 2** | -0.05 [-0.18, +0.08] | ✅ equivalent | -0.33 |
| **speculative λ=1.0 (null)** | +0.05 [-0.06, +0.16] | ✅ equivalent | +0.16 |
| **speculative λ=0.2** | -0.02 [-0.14, +0.10] | ✅ equivalent | +0.21 |
| **NF4 4-bit** | +0.01 [-0.10, +0.12] | ✅ equivalent | +0.24 |
| **HQQ 3-bit** | -0.70 [-0.85, -0.55] | ❌ worse | -1.11 |
| **early exit (refill)** | -1.07 [-1.26, -0.87] | ❌ worse | -2.54 |
| **early exit (no refill)** | -1.67 [-1.90, -1.43] | ❌ worse | -3.29 |

> ✅ 表示在 ±0.3 内等效；❌ 表示显著更差

### **与基线方法的对比结果**

#### **(1) 4-bit 量化（NF4）几乎无损**
- 在所有领域（包括中文和多步数学）中，**NF4 4-bit 与 full-precision 输出质量无统计显著差异**（Δ ≈ 0）
- 支持了实践中广泛采用 4-bit 量化的合理性

#### **(2) 3-bit 量化代价高昂且领域依赖性强**
- **HQQ 3-bit** 平均损失 **0.7 分**
- 领域差异极大：
  - Arithmetic: -0.09
  - English prose: -0.46
  - Code: -0.84
  - Chinese: -0.91
  - Hard-verifiable: **-1.11**
- 表明 **3-bit 不适用于复杂推理任务**

#### **(3) Early Exit 对复杂任务破坏严重**
- 即使带有 cache repair 和 periodic refill，在 hard domain 中仍损失 **2.5 分以上**
- 正确率从 19/27 暴跌至 6/27
- 但在普通英文段落中仅损失约 0.7 分 → **适用场景高度受限**

#### **(4) Lenient Speculative Decoding 未造成可观测损失**
- 即使使用宽松接受规则（λ=0.2），也未能检测到显著质量下降
- EAR（Expected Acceptance Rate）高达 0.952，表明兼容性好

#### **(5) 不同模型表现差异显著**
| Model | HQQ 3-bit Pooled Loss |
|-------|------------------------|
| Qwen2.5-7B | -0.70 |
| Llama-3.1-8B | **-1.84** |

> 同样的 3-bit 量化器在一个模型上轻度降质，在另一个上重度降质，说明**不能将优化效果泛化到不同架构**

---

### **消融实验与关键发现**
- **Implementation Null 发现 bug**：在开发过程中，该控制项曾检测出 speculative decoding 实现中的两个缺陷
- **Positive Control 验证灵敏度**：早期测得的 early exit 损失被新协议成功复现（顺序正确，幅度合理）
- **Judge 噪声仅占总方差 ~8–12%**，主要变异来自模型自身采样，强调需足够 prompt 数量（n ≥ 220）

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **4-bit 量化和 lossless speculative decoding 在多种任务下是“感知无损”的**，可安全用于生产环境。
2. ⚠️ **3-bit 量化和 early exit 的质量损失高度依赖于任务领域**：
   - 在简单文本生成中尚可接受
   - 在数学推理、编程等复杂任务中会造成灾难性失败
3. 🔍 **现有代理指标（perplexity, token agreement, EAR）无法准确反映真实质量变化**：
   - EAR 高不代表质量好（如重复文本 EAR 很高）
   - Perplexity 可奖励循环输出
   - Token-level confidence 与语义后果无关（改一个数字 vs 改一个逗号）
4. 🌍 **优化效果具有模型特异性**：相同量化策略在 Qwen 和 Llama 上表现迥异，不能跨模型外推
5. 🧪 **LLM judge 是有效的测量工具，但必须经过校准**：未经校准的 judge 会引入系统偏差，导致错误结论

### **方法的局限性**
- 当前仅使用 **claude-sonnet-5** 作为主 judge，虽经多 judge 验证，但仍可能存在模型偏好偏差
- 测试集中在 **7–8B 参数模型**，更大模型（如 70B）可能容忍更低比特宽度
- **hard-verifiable domain 的 paired-score 方差高达 2.0**，需要更多样本才能获得稳定估计
- Early exit 仅测试单一层数（layer 15）和 refill 间隔（16 tokens），未探索完整 trade-off 曲线
- Llama 上的 speculative null 出现轻微负偏移（-0.14），原因未完全定位（疑似 batched arithmetic numerics）

### **未来工作方向**
- 将该协议扩展至其他影响输出分布的技术：
  - Context compaction / summarization
  - KV-cache eviction
  - Prompt compression
  - Retrieval truncation
  - Safety filtering
  - Watermarking
  - Distillation
  - System prompt 修改
- 开发更高效的 cheap signals 来近似 judge 输出（目前所有代理指标均表现不佳）
- 构建面向特定部署场景的“质量风险画像”（quality risk profile）
- 探索动态适应性优化：根据输入领域自动选择最优 inference strategy

---

> **一句话总结**：  
> 本文建立了一个**科学化、可校准的测量框架**，揭示了推理优化对输出质量的影响远比传统指标所显示的更为复杂和领域依赖，呼吁社区放弃单一代理指标，转向更严谨的实验设计。

</details>

---

### 11. [DANTINOX: A Unified Framework for Multi-Paradigm Language Modeling](https://arxiv.org/abs/2609.17535)

**Authors**: Marco Simoni, Aleksandar Fontana, Giulio Rossolini, Andrea Saracino  
**Category**: cs.CL  
**Published**: 2026-09-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.17535v1  

#### Abstract
Language generation research increasingly spans three paradigms: autoregressive decoding, discrete masked diffusion, and continuous flow-matching. Comparing them is difficult because each lives in a separate codebase, so measured differences often reflect implementation details rather than the parad...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DANTINOX: A Unified Framework for Multi-Paradigm Language Modeling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前语言生成研究涵盖多种范式（paradigm），主要包括：
- **Autoregressive (AR)**：传统的从左到右逐词生成
- **Discrete Masked Diffusion**：基于离散空间的去噪扩散模型
- **Continuous Flow-Matching**：在连续嵌入空间中进行流匹配建模

然而，这些范式通常实现在**不同的代码库**中（如 HuggingFace、MaxText、xLM 等），导致跨范式的比较难以公平进行——观察到的差异可能源于实现细节（如 tokenizer、初始化策略、训练循环）而非算法本身。

> 🔍 **核心挑战**：缺乏一个统一框架来在相同条件下公平地训练、评估和部署不同生成范式。

---

### 🚀 提出的新方法与创新点

作者提出 **DANTINOX** —— 一个基于 **JAX/Flax** 的开源框架，其核心思想是：

#### （1）统一的模块化 Transformer Backbone
- 所有三种生成范式共享同一个可配置的 Transformer 架构。
- 范式切换仅需修改 `config.paradigm` 字段（无需重写模型代码）。
- 支持灵活组合：
  - Attention 变体：**MHA**, **GQA**, **MLA**
  - FFN 类型：MLP、SwiGLU、MoE（Top-k）、LatentMoE
  - Positional Encoding：RoPE、Sinusoidal、Learned
  - Normalization：RMSNorm、LayerNorm
  - LoRA 微调支持

#### （2）端到端生命周期支持
提供统一 API 支持：
- `fit()`：训练
- `stream()`：流式推理
- 内置 benchmarking 工具（延迟、吞吐量、能耗分析）

#### （3）硬件感知性能分析
集成 `dx.count_flops`, `dx.profile`, 和 `BenchmarkSuite`，支持零执行开销的 FLOP 分析与硬件 Roofline 分析。

---

### ⚖️ 相比现有方法的优势

| 特性 | DANTINOX | 其他框架（HuggingFace, MaxText, xLM, dLLM 等） |
|------|----------|---------------------------------------------|
| 多范式支持（AR/Diffusion/Flow） | ✅ 完整支持 | ❌ 通常只支持一种或两种 |
| 统一 Backbone | ✅ 是 | ❌ 各自独立实现 |
| 配置驱动切换范式 | ✅ 仅改 config | ❌ 需重构代码 |
| Attention 变体多样性 | ✅ MHA/GQA/MLA | ⚠️ 多数仅支持 MHA/GQA |
| 内置 Benchmark Suite | ✅ 包含延迟/吞吐/能效分析 | ❌ 通常无或需自行构建 |

> 💡 **一句话总结**：DANTINOX 实现了“一次架构，多范式运行”，极大降低了跨范式研究的技术门槛。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **WikiText-103-raw-v1**：用于小规模生成质量评估
  - 训练集限制为 50M tokens
  - 划分比例：90% 训练 / 10% 验证
  - 总预算：262M tokens
- **SentencePiece Tokenizer**：采用 T5 的词汇表（vocab size = 32,128）
- **字符级 Tokenization**：用于部分推理效率测试（图8）

---

### ⚙️ 实验设置

#### （1）模型规模
- **Small Scale**：dim=512, num_blocks=12, ~65–80M 参数 → 用于生成质量对比
- **Large Scale**：dim=1024, num_blocks=16, ~130M 参数 → 用于推理效率分析（A100 GPU）

#### （2）训练配置
- Optimizer：Muon（带 cosine 学习率调度）
- Batch Size：effective batch = 256 sequences × 512 tokens
- Gradient Accumulation：4 steps
- Precision：bf16 混合精度
- 并行策略：Data Parallelism + Tensor Parallelism（SPMD）

#### （3）评估指标

| 类别 | 指标 |
|------|------|
| **生成质量** | MAUVE（人类相似度）、PPL（困惑度）、Distinct-2 / Rep-4（多样性/重复性）、BLEU-4cond（条件续写能力） |
| **推理效率** | Latency（延迟）、Throughput（吞吐量）、Energy per token（每 token 能耗）、MFU（Model FLOPs Utilization） |
| **系统分析** | Hardware Roofline Analysis（算力/内存边界分析） |

#### （4）基线对比
- 与独立实现的框架进行外部验证：
  - **dLLM**（Zhou et al., 2026）：用于 discrete diffusion 对齐
  - **xLM**（Patel et al., 2026）：用于 AR 对齐
- 控制变量：完全相同的架构、优化器、学习率、warmup、tokenizer、数据划分

---

## 3. 主要实验结果和性能指标

### 📊 （1）外部验证结果（Section 4.1）
- 在 **WikiText-103** 上复现 dLLM 和 xLM 的训练曲线：
  - **Discrete Diffusion vs dLLM**：最终损失相差 <1%
  - **AR vs xLM**：最终损失相差 ~8%，残差主要来自 DANTINOX 默认使用 RMSNorm 和 weight tying，而 xLM 使用 LayerNorm 且未绑定 embedding
  - 若关闭 SwiGLU（改为 GELU），差距进一步缩小 0.14 nats

> ✅ 表明 DANTINOX 的实现是准确可靠的，非 artifacts of codebase

---

### 🧪 （2）生成质量对比（Table 2）

在 **9 种 paradigm × attention 组合**下进行控制变量实验（仅改 config）：

| Paradigm | Attention | MAUVE↓ | PPL↓ | D-2↑ | R-4↓ | B-4↑ |
|---------|----------|--------|------|------|------|------|
| AR      | MHA      | 0.17   | 1216 | 0.697| 0.007| 0.052|
|         | GQA      | 0.18   | 1233 | 0.688| 0.005| 0.050|
|         | MLA      | 0.07   | 1860 | 0.682| 0.003| 0.020|
| Discrete Diffusion | MHA | 0.20 | 1834 | 0.728| 0.019| 0.029|
|         | GQA      | 0.12   | 1777 | 0.733| 0.006| 0.030|
|         | MLA      | 0.15   | 1803 | 0.720| 0.020| 0.033|
| Continuous Flow-Matching | MHA | 0.80 | 234.6| 0.627| 0.007| – |
|         | GQA      | 0.69   | 188.0| 0.562| 0.027| – |
|         | MLA      | 0.78   | 156.3| 0.538| 0.107| – |

#### 关键发现：
- **Flow-Matching** 生成文本最流畅（PPL 最低），尤其适合短文本生成；
- **Discrete Diffusion** 词汇最多样（D-2 最高）；
- **AR** 条件续写最准确（BLEU-4cond 最高），重复最少（R-4 最低）；
- **Attention 影响依赖于范式**：
  - MLA 在 Flow-Matching 中表现最好（但重复严重）
  - MLA 在 AR 中表现最弱

> ✅ 验证了 DANTINOX 可通过纯配置变化完成大规模消融实验（共 9×3=27 次 runs，含 seed variance）

---

### ⏱️ （3）推理效率与 Roofline 分析（Figure 5）

在 A100 GPU 上对 large backbone 进行全面 benchmark：

#### （a）低批量场景（B=1）
- **Discrete Diffusion (S=32)**：单请求延迟 **56ms**
- **AR decoding**：延迟高达 **542ms**（超过 200ms SLO）
- 能耗：diffusion 比 AR 低约 **7×**

> ✅ Diffusion 更适合交互式服务（低延迟、高并发响应）

#### （b）高批量场景（B≥32）
- **AR 开始反超**：因 KV-Cache 可跨序列共享，batching 效益显著
- 在 B=256 时，AR 吞吐最高、能耗最低
- Diffusion 每步均为 compute-bound，增加 batch 收益有限

#### （c）Roofline 分析解释原因：
- **AR decoding**：memory-bound → batch 增加提升 arithmetic intensity
- **Diffusion**：compute-bound（已达 60 TF/s，占 bf16 peak 19%）→ 难以进一步利用算力

> ✅ 得出实用决策规则：
> - 小批量、低延迟需求 → 用 diffusion
> - 大批量、高吞吐需求 → 用 AR
> - 切换无需改代码，只需改 config

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **DANTINOX 成功实现了多范式语言建模的统一平台**：
   - 所有范式共享同一 backbone，确保公平比较
   - 范式、attention、并行策略均可通过 config 切换

2. **不同生成范式各有优劣，适用场景不同**：
   - **AR**：精确续写强、重复少 → 适合任务导向生成
   - **Discrete Diffusion**：多样性高 → 适合创意写作
   - **Flow-Matching**：流畅度高 → 适合短文本生成

3. **Attention 设计的影响高度依赖于生成范式**：
   - MLA 并非在所有情况下都优于 MHA/GQA
   - 必须结合范式一起考虑架构选择

4. **推理效率存在明确的“批大小拐点”**：
   - 小 batch：diffusion 占优
   - 大 batch：AR 占优
   - 可通过 Roofline 分析预测最优部署策略

---

### ⚠️ 局限性

1. **当前未支持 prefix conditioning for Flow-Matching**
   - 导致无法计算 BLEU-4cond 指标
   - 计划在未来版本中补全

2. **实验规模较小**
   - 当前最大 ~130M 参数，尚未扩展至百亿级以上
   - 是否结论可外推仍需更大模型验证

3. **仅支持 JAX/Flax 生态**
   - 不兼容 PyTorch 用户，学习成本较高

4. **缺少对 MoE、长上下文等前沿技术的深度集成测试**

---

### 🔮 未来工作方向

1. **扩展更大规模模型训练与评估**
   - 探索 DANTINOX 在千亿参数下的可扩展性

2. **完善 Flow-Matching 的条件生成接口**
   - 支持 prefix/prompt conditioning

3. **引入更多新型 attention 和 sparse 架构**
   - 如 FlashAttention-2、MQA、Ring Attention 等

4. **增强教育功能**
   - 提供教学 notebook，帮助学生理解多范式差异

5. **推动社区共建**
   - 鼓励第三方贡献新的 paradigm 插件（如 VAE-based generation）

---

## 总结

> 🎯 **DANTINOX 的本质价值在于“控制变量”能力**：它让研究人员可以像做物理实验一样，在其他一切不变的前提下，只改变一个变量（如 generation paradigm 或 attention type），从而真正识别出算法层面的本质差异。

该项目已开源（MIT License），可通过 pip 安装，并附带完整文档、notebooks 和演示视频，有望成为未来多范式语言模型研究的标准基础设施之一。

</details>

---

### 12. [A GAN-Based Framework for Robust DDoS Attack Detection](https://arxiv.org/abs/2609.18281)

**Authors**: Makram Chehayeb, Walid Fahs, Amina Rizk, Rida Khatoun, Omran Berjawi  
**Category**: cs.LG  
**Published**: 2026-09-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.18281v1  

#### Abstract
The availability and consistency of online services remain vulnerable due to Distributed Denial of Service (DDoS) attacks. These attacks are evolving by adopting more complex strategies to evade traditional network security systems. Despite the effectiveness of machine learning models in detecting D...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《A GAN-Based Framework for Robust DDoS Attack Detection》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代 **DDoS攻击** 越来越复杂，传统基于规则或静态机器学习的 **Intrusion Detection Systems (IDS)** 难以应对新型变种攻击。更严重的是，攻击者可通过**对抗性扰动**（adversarial perturbations）修改恶意流量的关键特征，使其被误判为良性流量，从而绕过AI驱动的检测系统。

此外，现有基于 **GAN 的对抗训练方法** 存在三大缺陷：
1. 生成的对抗样本可能违反网络协议逻辑（如包速率与字节数不一致），即 **Feature Realizability 问题**；
2. 多数研究仅在离线静态测试集上评估，缺乏对真实部署中多阶段动态流量（正常 → 攻击峰值 → 恢复期）的验证；
3. 忽视模型推理时延、内存占用等 **Operational Overhead** 指标，难以部署到边缘设备或ISP级系统。

---

### 🚀 提出的新方法与创新思路
本文提出一种结合 **WGAN-GP 与先进机器学习模型** 的鲁棒 DDoS 检测框架，其核心创新包括：

#### （1）**WGAN-GP 用于生成高质量对抗流量**
- 使用 **Wasserstein GAN with Gradient Penalty (WGAN-GP)** 生成逼真的对抗性 DDoS 流量，相比传统 GAN 更稳定、多样性更高。
- 生成的对抗样本用于数据增强，提升模型对未知规避策略的泛化能力。

#### （2）**构建混合训练数据集（Hybrid Dataset）**
- 将原始良性/恶意流量与 WGAN-GP 生成的对抗流量融合，形成 **robust hybrid dataset**，使模型学习更具鲁棒性的决策边界。

#### （3）**多维度综合评估体系**
- **Protocol-Aware Feature Realizability Analysis**：分析扰动后的特征是否符合实际网络行为约束；
- **Multi-Phase Live Deployment Simulation**：模拟“正常 → 攻击 → 恢复”三个阶段，评估模型在动态环境中的表现；
- **System Resource & Operational Profiling**：报告延迟、吞吐量、内存消耗等实用化指标，验证边缘部署可行性。

---

### 🔍 相比现有方法的优势
| 维度 | 本文方法优势 |
|------|--------------|
| **对抗防御能力** | 显著优于仅在干净数据上训练的传统模型，在 Adv-5 和 Adv-9 攻击下仍保持高 Recall |
| **实用性** | 提供 Transformer 在网络边缘部署的实际资源开销数据，支持实时检测 |
| **评估全面性** | 不仅看离线准确率，还涵盖动态场景、操作成本和特征可实现性分析 |
| **模型通用性** | 可集成于企业网关、云IDS、ISP边缘等多种安全架构 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 主要使用 **CICDDoS2019** 数据集：
  - 包含多种 DDoS 攻击类型：UDP Flood, HTTP Flood, SYN Flood, DNS Flood, LDAP, NTP, SSDP 等；
  - 提取超过 80 个 flow-level 特征（via CICFlowMeter）；
  - 数据不平衡（攻击样本多于正常流量）；
  - 经预处理后保留 **Top-20 最具区分性的特征**（ANOVA F-test 选择）。

---

### ⚙️ 实验设置

#### （1）**对抗样本生成**
- 使用 **WGAN-GP** 生成合成对抗流量：
  - Generator 输入：100维噪声向量；
  - 输出：20维特征向量，模拟攻击流；
  - 训练策略：Critic 更新4次 / Generator 更新1次，Adam优化器（lr=2e-4, β1=0.5）；
- 构造两种对抗测试集：
  - **Adv-5**：扰动前5个最重要特征；
  - **Adv-9**：扰动前9个最重要特征。

#### （2）**训练策略**
- **Baseline 阶段**：模型在干净 CICDDoS2019 上训练；
- **增强阶段**：使用 **hybrid dataset**（原始数据 + WGAN-GP 生成的对抗样本，随机修改5个特征）重新训练模型。

#### （3）**检测模型**
| 模型 | 结构说明 |
|------|----------|
| **Random Forest (RF)** | 100棵决策树，类别加权平衡 |
| **Deep Neural Ensemble (DNE)** | 5个相同结构的前馈神经网络集成：<br>Input → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Sigmoid |
| **Transformer (TF)** | 基于注意力机制：<br>Embedding(20→32) → 4头Multi-head Attention → FFN → 分类头 |

#### （4）**评估指标**
- 主要关注 **Recall**（避免漏检攻击）；
- 同时报告 **F1-Score**（平衡 Precision 与 Recall）；
- 在真实流量模拟中评估 **Accuracy** 和 **False Positive Rate**。

#### （5）**基线对比方法**
- 本文未直接与其他文献模型进行端到端比较，而是以内置的 RF、DNE、TF 在“clean-trained” vs “augmented-trained”之间做对照，体现**对抗增强的有效性**。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables II–V）

#### （1）**Baseline 性能（Clean Test Set）**
| Model | Recall | F1-Score |
|-------|--------|-----------|
| RF    | 0.9991 | 0.9995    |
| DNE   | 0.9960 | 0.9980    |
| TF    | ~0.99+ | ~0.99+    |

> 所有模型在干净数据上均表现出色（>99% Recall），表明标准评估具有误导性。

---

#### （2）**Baseline 在对抗测试集上的崩溃（Table III）**
| Model | Recall (Adv-5) | Recall (Adv-9) |
|-------|----------------|----------------|
| RF    | 0.5451         | **0.1036**     |
| DNE   | 0.2910         | **0.0465**     |
| TF   | 0.6504         | **0.2045**     |

> 所有模型性能大幅下降，尤其 Adv-9 下 RF 和 DNE 几乎失效，说明传统训练方式极度脆弱。

---

#### （3）**对抗增强后性能恢复（Table IV）**
| Model | Recall (Adv-5) | Recall (Adv-9) |
|-------|----------------|----------------|
| RF    | **1.0000**     | **0.9689**     |
| DNE   | 0.5848         | 0.1631         |
| TF    | **0.9994**     | **0.8002**     |

> ✅ **对抗增强显著提升了鲁棒性**：
> - RF 和 TF 在 Adv-5 上达到近乎完美检测；
> - TF 在最严苛的 Adv-9 上仍保持 **80% Recall**，远超其他模型；
> - DNE 提升有限，显示其结构不适合此类任务。

---

#### （4）**真实流量模拟测试（Table V）**
| Phase | Metric | RF | DNE | TF |
|-------|--------|----|-----|----|
| Before (Normal) | Accuracy | 1.00 | 0.00 | 1.00 |
| During (Attack) | Recall | 0.01 | 1.00 | **0.69** |
| After (Recovery) | Recall | 0.10 | 1.00 | **0.80** |

> 🔍 关键发现：
> - **RF** 过于保守：无误报，但几乎无法检出攻击（Recall=0.01），**不可用作主检测器**；
> - **DNE** 过度敏感：所有流量标记为恶意，**FPR=100%**，导致“告警疲劳”，完全不可行；
> - **TF** 实现最佳平衡：零误报 + 较高攻击检出率，是唯一适合实战部署的模型。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **仅依赖干净数据训练的模型极易被对抗攻击绕过**，即使在标准数据集上表现优异（>99% Recall），在对抗环境下 Recall 可暴跌至 <10%。
2. **WGAN-GP 生成的对抗样本可用于有效数据增强**，显著提高模型对特征空间扰动的鲁棒性，尤其是对 **Transformer 模型效果最为明显**。
3. **Transformer 架构在鲁棒性和实用性之间实现了最佳平衡**：
   - 对抗增强后在 Adv-9 上 Recall 达 **0.8002**；
   - 在真实流量模拟中实现 **零误报 + 高检出率**；
   - 推理资源可控，具备在网络边缘部署潜力。
4. **模型设计直接影响运营行为**：
   - RF 倾向于“宁可放过也不误杀”；
   - DNE 容易“草木皆兵”；
   - TF 更好地捕捉全局特征依赖关系，适应性强。

---

### ⚠️ 局限性
1. **对抗模拟仍处于特征层面**（feature-space evasion），尚未扩展到真实的 packet-level 攻击注入（如通过 ptf-agent 或物理测试床）；
2. 当前方法依赖离线重训练，缺乏在线自适应更新机制应对 zero-day 攻击；
3. 虽然评估了 Transformer 的资源消耗，但未提供硬件加速或轻量化版本的设计；
4. 实验集中在 CICDDoS2019，跨数据集泛化能力（如 Bot-IoT、ToN-IoT）有待验证。

---

### 🔮 未来工作方向
1. **Problem-Space Adversarial Testing**：
   - 引入 packet-level evasion 工具（如 ptf-agent）在真实网络环境中测试攻击可实现性；
   - 验证生成的对抗特征能否映射为合法协议行为。
   
2. **Cross-Dataset Generalization**：
   - 在更多基准数据集（如 Bot-IoT, ToN-IoT）上验证 WGAN-GP pipeline 的迁移能力；
   - 探索跨域检测的统一表示学习。

3. **Online Learning & Real-Time Adaptation**：
   - 设计支持持续学习（continual learning）的框架，实现在动态攻击模式下的自动模型更新；
   - 结合联邦学习保护隐私的同时共享威胁情报。

4. **轻量化与边缘优化**：
   - 对 Transformer 模型进行剪枝、量化或蒸馏，降低部署门槛；
   - 探索 TinyML 或 FPGA 加速方案以满足低延迟需求。

---

> 💡 **总体评价**：  
> 本论文不仅提出了一个有效的对抗增强框架，更重要的是建立了从“实验室精度”到“实战可用性”的完整评估范式，推动了 AI-driven DDoS 检测从理论走向工程落地。Transformer + WGAN-GP 的组合展现出成为下一代智能防御核心组件的巨大潜力。

</details>

---

### 13. [LightSleepX: A Lightweight, Inception-Based Dual-Modal Network for Sleep Staging](https://arxiv.org/abs/2609.19062)

**Authors**: Yi Wang  
**Category**: cs.LG  
**Published**: 2026-09-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2609.19062v1  

#### Abstract
Automatic sleep staging is fundamental to personal health monitoring, yet many existing approaches are ill-suited for real-world applications. Traditional pipelines often rely on hand-crafted features or shallow machine learning models that struggle to generalize, while state-of-the-art deep learnin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《LightSleepX: A Lightweight, Inception-Based Dual-Modal Network for Sleep Staging》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统自动睡眠分期方法存在以下不足：
- **手工特征依赖** 或 **浅层模型** 泛化能力差；
- 当前主流 **deep learning 模型**（如 DeepSleepNet、SleepTransformer）虽然准确率高，但参数量大、计算开销高，难以部署在资源受限设备上（如可穿戴设备）；
- 隐私敏感场景下需要本地化处理，对模型轻量化提出更高要求。

因此，本文旨在设计一个**兼顾高性能与低计算成本**的自动化睡眠分期框架，适用于真实世界中的长期家庭健康监测。

---

### 🚀 提出的新方法与创新思路

LightSleepX 是一种轻量级、双模态（EEG + EOG）深度学习网络，其核心创新在于两个设计原则：

#### （1）**Efficient Multi-Scale Feature Extraction**  
采用 **Multi-Branch Inception-style 架构** 结合 **Depthwise Separable Convolutions** 进行多尺度特征提取，并引入 **Multi-scale Enhanced Attention (MEA)** 实现跨模态融合。
- 利用不同卷积核大小（3, 5, 7）并行捕获短时事件（如spindle）和长周期慢波活动（如delta波）；
- 使用 depthwise separable convolutions 显著降低参数量和FLOPs；
- MEA模块自适应地增强关键通道响应，提升模态间信息融合效果。

#### （2）**Rule-free Temporal Modeling**  
引入 **Mamba encoder** 替代传统的 RNN/LSTM 或人工规则建模睡眠阶段转移。
- Mamba 是一种基于 **Selective State Space Model (SSM)** 的序列建模架构，支持线性时间复杂度下的长距离依赖建模；
- 双向 Mamba 允许模型同时利用前后上下文信息，无需预定义状态转移规则；
- 相比 Transformer 更高效，相比 LSTM 更适合长序列建模。

---

### 🔍 相比现有方法的优势

| 维度 | LightSleepX 的优势 |
|------|------------------|
| **性能** | 在多个公开数据集上达到 SOTA 级别的 accuracy 和 macro-F1 分数 |
| **效率** | 仅 **0.049M 参数** 和 **195.9 MFLOPs**，远低于主流模型（如 U-Time: 1.1M, SleepTransformer: 3.7M） |
| **实用性** | 支持本地化部署，满足隐私保护需求，适合边缘设备应用 |
| **鲁棒性** | 对难分类类别（如 N1）表现优异，尤其在跨被试（cross-subject）任务中泛化能力强 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 描述 |
|-------|------|
| **Sleep-EDF-20** | 包含 20 名健康受试者，共 42,308 个 epoch；用于小规模验证 |
| **Sleep-EDF-78** | 更大规模且年龄跨度更大（25–101岁），共 195,479 个 epoch；测试泛化能力 |
| **ISRUC-S3** | 跨被试挑战性强，包含 100 名受试者；标准 10-fold cross-validation 设置，检验模型鲁棒性 |

> 所有数据均使用 Fpz-Cz EEG 和 ROC-LOC EOG 通道，采样率为 100Hz，按 AASM 标准分为 5 类：Wake, N1, N2, N3, REM。

---

### ⚙️ 实验设置

- **输入格式**：每段输入为连续 10 个 30 秒 epoch（即 5 分钟），构成 `(B, S=10, 1, T=3000)` 张量；
- **预处理**：z-score 归一化，滑动窗口构造序列；
- **训练配置**：
  - Batch size: 128
  - Optimizer: Adam (lr=0.001, weight_decay=0.001, amsgrad=True)
  - Loss: Dynamic Focal Loss（缓解类别不平衡，特别是 N1/N3）
  - Epochs: 100，早停机制防止过拟合
- **实现平台**：PyTorch + NVIDIA GPU

---

### 🎯 评估指标

| 指标 | 说明 |
|------|------|
| **Accuracy (ACC)** | 整体分类正确率 |
| **Macro-F1 Score** | 各类 F1 的平均值，更关注少数类性能（如 N1） |
| **Cohen’s Kappa (K)** | 衡量分类一致性，考虑随机猜测影响 |
| **参数量 (#Param)** 和 **FLOPs** | 衡量模型效率的关键指标 |

---

### 🆚 基线方法对比

参与比较的 SOTA 方法包括：
- **SleepEEGNet**, **DeepSleepNet**: 早期 CNN/RNN 混合结构
- **AttnSleep**, **SeqSleepNet**: 注意力机制代表
- **U-Time**, **SleepTransformer**: 当前高性能模型
- **TinySleepNet**: 轻量级代表之一

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 数据集 | ACC (%) | Macro-F1 | Kappa | #Param (M) | FLOPs (M) |
|--------|---------|----------|--------|------------|-----------|
| **Sleep-EDF-20** | **85.9** | **0.807** | **0.81** | **0.049** | ~195.9 |
| **Sleep-EDF-78** | 82.6 | **0.787** | 0.77 | **0.049** | ~195.9 |
| **ISRUC-S3** | **81.80** | **0.796** | **0.761** | **0.049** | ~195.9 |

> 💡 LightSleepX 在所有三个数据集上均取得 **最佳或接近最佳的 macro-F1 和 accuracy**，尤其在最具挑战性的 **ISRUC-S3** 上领先第二名超过 **5个百分点**。

---

### 🆚 与基线方法对比结果

#### ✅ Sleep-EDF-20（见 Table 1）
- LightSleepX 以 **85.9% ACC** 和 **0.807 Macro-F1** 超越所有基线；
- 参数仅为 TinySleepNet 的 **1/26**，却性能更高。

#### ✅ Sleep-EDF-78（见 Table 2）
- 尽管参数最少（0.049M vs. SeqSleepNet 0.164M），仍获得最高 **Macro-F1 (0.787)**；
- 显示出极强的类别不平衡处理能力。

#### ✅ ISRUC-S3（见 Table 3）
- **ACC 提升超 5%**（从 ~76% → 81.8%），是目前唯一突破 80% 的轻量模型；
- Kappa 达到 0.761，表明具有高度一致性和临床可用潜力。

---

### 🔬 消融实验结果（Ablation Study）

在 Sleep-EDF-20 上进行消融研究，验证各组件贡献：

| 模型变体 | ACC (%) | Macro-F1 |
|--------|--------|----------|
| **Full Model (LightSleepX)** | **85.9** | **0.803** |
| w/o Multi-Branch Inception (Std. CNN) | 80.5 | 0.750 |
| w/o MEA | 81.8 | 0.755 |
| w/o Mamba (Bi-LSTM) | 81.2 | 0.748 |

> 观察结论：
- **Multi-Branch Inception** 贡献最大（↓5.4% ACC），证明多尺度特征提取至关重要；
- **MEA** 显著提升模态融合质量；
- **Mamba > Bi-LSTM**，说明其在捕捉长程依赖方面更具效率与表达力。

此外，在 **N1 阶段 F1-score** 上，LightSleepX 达到 **63.0%**，显著优于其他模型（如 AttnSleep: 42%, U-Time: 51%），显示其对模糊过渡阶段的强大判别能力。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **轻量化不等于低性能**：通过合理的架构设计（Inception + Depthwise Conv + Mamba），可以在极低参数量下实现 SOTA 性能；
2. **Mamba 在生理信号建模中极具潜力**：相比传统 RNN/Transformer，Mamba 在保持高效的同时更好地建模睡眠阶段动态变化；
3. **多尺度 + 注意力融合有效提升双模态表示能力**：MEA 模块增强了 EEG 与 EOG 的互补信息整合；
4. **卓越的跨被试泛化能力**：在 ISRUC-S3 上的表现证明模型具备良好的鲁棒性，适合实际应用场景。

---

### ⚠️ 方法的局限性

- 当前仅使用 **EEG + EOG** 双通道，未融合更多生理信号（如 EMG、ECG），可能限制某些病理状态识别能力；
- 模型尚未在真实边缘设备（如智能手表）上完成端到端部署测试；
- 所有训练基于标注良好的 PSG 数据，对噪声或伪迹的鲁棒性有待进一步验证。

---

### 🔮 未来工作方向

作者指出后续将聚焦于三个方向：
1. **扩展多模态输入**：集成 EMG、ECG 等信号，提升临床诊断价值；
2. **硬件感知优化**：开展量化（quantization）、剪枝（pruning）等技术，推动在 **edge computing platform** 上的实际部署；
3. **提升泛化能力**：探索半监督学习与 domain adaptation 技术，使模型适应不同人群、设备和环境。

---

## ✅ 总结

LightSleepX 成功构建了一个**高性能、超轻量、易于部署**的自动睡眠分期系统，解决了当前 deep learning 模型“高精度但难落地”的矛盾。它不仅在多个基准上刷新记录，更为未来**个人化、本地化、隐私友好的数字健康系统**提供了可行的技术路径。

</details>

---

### 14. [Contiguity, Not Importance: Budgeted Repair of Stale KV Caches After Document Edits](https://arxiv.org/abs/2609.17983)

**Authors**: Mingyang Mao, Wyatt Mackey, Xiaomin Lin  
**Category**: cs.AI  
**Published**: 2026-09-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.17983v1  

#### Abstract
KV-cache reuse can reduce inference cost in retrieval-augmented generation and agentic systems, but cached contexts may become stale when retrieved knowledge, working memory, or user state is edited. Under causal self-attention, even a local edit can affect downstream KV states. A full re-prefill re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Contiguity, Not Importance: Budgeted Repair of Stale KV Caches After Document Edits**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
在 **Retrieval-Augmented Generation (RAG)** 和基于 LLM 的智能体系统中，为提升推理效率，常复用已缓存的 **Key-Value (KV) states**。然而，当被检索的文档、工作记忆或用户状态发生编辑时，原有的 KV 缓存会变得“陈旧”（stale），导致模型基于过时信息生成错误答案。

传统做法是进行 **full re-prefill**（重新编码整个上下文），代价高昂；而仅刷新编辑部分则可能导致依赖该编辑的下游 KV 状态仍处于 stale 状态。本文研究如何在有限计算预算下高效修复 stale KV 缓存。

---

### ✅ 提出的新方法与新思路  

- **将缓存修复形式化为“预算化重计算”（budgeted recomputation）任务**：给定一个编辑位置和固定计算预算 $ K $（额外可重计算的 token 数量），选择最优的一组下游位置进行重计算。
  
- **提出并验证了 EDITLOCAL 方法**：即在编辑结束位置后连续选取 $ K $ 个 token 进行重计算，形成一个从编辑点延伸出去的 **contiguous window**。

- **揭示了一个关键机制洞见：“连续性”（Contiguity）优于“重要性”（Importance）**：
  - 即使某些位置在注意力或因果影响上看似“更重要”，但如果它们孤立存在且周围仍是 stale 状态，则无法有效恢复正确行为。
  - 只有通过 **连续重建前向依赖链** 才能真正修复语义路径。

- **区分了“状态移植”（transplant）与“实际重计算”（recomputation）的效果差异**：
  - 很多基于重要性的选择策略（如 attention、KV deviation）在“移植干净状态”时表现良好，但在真实“重计算”场景中失败严重，因其忽略了周围 stale 上下文的影响。

---

### ✅ 相比现有方法的优势  

| 方法 | 是否训练 | 是否部署可行 | 性能优势 |
|------|--------|-------------|----------|
| **EDITLOCAL** | ❌ 否 | ✅ 是 | 显著优于所有信号驱动方法，在相邻依赖场景下接近 oracle 效果 |
| Attention-based (e.g., ProphetKV) | ❌ 否 | ✅ 是 | 仅在特定模型（Llama）上有一定效果 |
| KV Deviation (e.g., CacheBlend) | ❌ 否 | ✅ 是 | 表现较差，尤其在 derived case 中 |
| Causal Oracle / CarrierWindow | ✅ 需 oracle | ❌ 否（诊断用） | 揭示理论上限，但仍远低于理想 |

> ✅ **核心优势**：无需训练、无需额外前向传播、实现简单、速度快（13–21× 快于 full re-prefill）、在常见邻接场景下几乎完全恢复性能。

---

## 2. 核心实验方法和设置

### 📚 数据集与构造方式  

- 基于 **HotpotQA fullwiki validation set** 构造约 5000 token 的上下文背景（作为非相关信息填充）。
- 在文档中间插入合成的事实记录块（synthetic record block），模拟企业政策、别名映射等结构化信息。
- 每个样本包含两个版本：
  - **Direct Probe**：问题直接询问被编辑的内容（answer span = edit span）
  - **Derived Probe**：答案位于未编辑但逻辑依赖于编辑内容的下游条目中（例如通过别名查表）

> 示例：
> ```
> [Alias Record] X uses alias B     ← 编辑此处（原为 A）
> [Lookup Table] A has value 4471
>              B has value 8730    ← 答案在此，未被编辑
> Q: What value does X’s alias map to? → 正确答案应为 8730
> ```

此外还构建了 **distance-controlled variant**：将 lookup 表移动到距离编辑点 250 或 1500 tokens 下游，测试依赖链长度对修复效果的影响。

---

### ⚙️ 实验设置  

- **模型**：Llama-3.1-8B-Instruct, Qwen3-8B, Mistral-7B-Instruct-v0.3
- **编辑类型**：单次、连续、长度不变的编辑（length-preserving edit）
- **修复操作**：
  - 固定重算编辑 span
  - 从下游候选池 $ D = [S_{\text{end}}, n-1) $ 中选择 $ K $ 个位置进行重计算
  - 其余 KV 状态保持不变
- **预算设定**：主实验 $ K=32 $，也测试 $ K \in \{8,16,32,64\} $

---

### 📊 评估指标  

1. **Margin Recovery (MR)**：
   $$
   MR = \frac{m_{\text{repaired}} - m_{\text{stale}}}{m_{\text{oracle}} - m_{\text{stale}}}
   $$
   - 衡量修复后 logit 差距恢复程度（0 = 无改善，1 = 完全恢复）
   - 能捕捉决策边界前的部分修复

2. **Flip Rate**：
   $$
   \text{Flip} = \mathbb{I}[y_b \neq a_{\text{old}} \land y_b = a_{\text{new}}]
   $$
   - 衡量生成是否成功切换到新答案

3. **KL Recovery**：衡量输出分布恢复程度（次要指标）

4. **Wall-clock Latency**：真实耗时对比（RTX 5090, bfloat16, batch=1）

---

### 🔁 基线方法对比（Selection Policies）  

| Policy | 规则说明 | 是否可部署 |
|-------|---------|-----------|
| **EDITLOCAL** | 选择紧接在编辑后的 $ K $ 个连续 token | ✅ 是 |
| **STRUCTURAL** | 选择最近的结构标记（如 `[SEP]`, `\n`） | ✅ 是 |
| **ATTENTION** | 基于 stale cache 中 query 对各位置的 attention 权重选 top-$K$ | ✅ 是（需一次前向） |
| **CACHEBLEND** | 基于 layer-2 的 key/value deviation 选 top-$K$ | ✅ 是（需一次 prefill 层） |
| **RANDOM** | 随机采样 $ K $ 个下游位置（5 seeds 平均） | ✅ 是 |
| **CAUSALORACLE** | 基于 oracle cache 测量因果效应排序（transplant 排名） | ❌ 否（仅用于诊断） |
| **CARRIERWINDOW** | 已知答案所在 block，重算整个 block（需先验知识） | ❌ 否（诊断） |

> 所有策略共享相同的编辑 span 和 query 处理流程，仅在选择 $ K $ 个下游位置时不同。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（$ K=32 $，derived condition，held-out set）

| Policy | Llama MR | Qwen MR | Mistral MR | Flip (典型值) |
|--------|----------|---------|------------|----------------|
| **EDITLOCAL** | **0.993** | **1.007** | **0.937** | 0.95–1.00 |
| ATTENTION | 0.533 | 0.015 | 0.019 | ≤0.03 |
| CACHEBLEND | 0.238 | 0.076 | 0.012 | ≤0.03 |
| STRUCTURAL | 0.126 | 0.008 | 0.012 | ≤0.03 |
| RANDOM | 0.078 | 0.004 | 0.004 | 0.00 |
| CAUSALORACLE (transplant rank) | 0.409 | 0.187 | 0.096 | — |

> ✅ **EDITLOCAL 在三个模型上全面胜出**，Margin Recovery 达到 0.94–1.01，Flip Rate 接近 100%

---

### 🔍 与基线方法的对比结果  

- **EDITLOCAL vs 所有其他 deployable 方法**：
  - 在所有 15 对比中均 Holm-corrected 显著 ($ p \leq 3\times10^{-9} $)
  - 最佳竞争者 ATTENTION 仅在 Llama 上达到 0.533 MR，其余模型接近失效
  - STRUCTURAL、CACHEBLEND、RANDOM 几乎无修复能力（MR < 0.1）

- **EDITLOCAL vs 非部署方法（诊断）**：
  - 即使使用因果效应最强的位置集合（CAUSALORACLE），在真实重计算下也只能恢复 0.10–0.41 MR
  - 表明“重要位置”的概念在 stale 上下文中不可靠

---

### 🔬 消融实验与关键变量分析  

#### ✅ 不同 $ K $ 的影响（图3）  

- **EDITLOCAL 呈现阈值效应**：
  - 当 $ K < 32 $：修复不完整（平均覆盖不到依赖路径终点）
  - 当 $ K = 32 $：覆盖大多数 block 内部路径，MR 骤升至 >0.9
  - 当 $ K = 64 $：进一步提升有限（边际收益递减）

> 推导出启发式规则：**EDITLOCAL@0** —— 直接重算到下一个文档边界（无需调参）

#### ✅ 距离效应（distance-controlled variant）  

| Policy | d=0 (adjacent) | d=250 | d=1500 |
|--------|----------------|-------|--------|
| **EDITLOCAL** | 1.00 / 1.00 / 0.94 | 0.09 / 0.01 / 0.01 | — / 0.01 / 0.02 |
| ATTENTION | 0.51 / 0.01 / 0.01 | 0.39 / 0.02 / 0.02 | — / 0.05 / 0.11 |
| CARRIERWINDOW | 0.61 / 0.95 / 0.74 | 0.43 / 0.90 / 0.64 | — / 0.89 / 0.60 |

> 发现：
- **EDITLOCAL 的优势高度依赖“邻接性”**：一旦答案移出当前 block，性能崩溃
- **ATTENTION 仅在 Llama 上保留部分有效性**
- **即使知道 carrier block 位置（CARRIERWINDOW），也无法完全恢复** → 表明搜索不是唯一瓶颈

#### ✅ Transplant vs Recomputation Gap  

- 在 development set 上比较相同位置集在两种操作下的表现：
  - **Transplant**：从 oracle cache 拷贝 KV → 平均恢复 0.93–0.99 MR
  - **Recomputation**：在 stale cache 中本地重算 → 仅恢复 0.03–0.41 MR
- 差距主要出现在 **scattered positions**（如 attention/deviation 选出的稀疏点）
- 结论：**transplant 不能作为 recomputation 的上界**

---

## 4. 关键结论和发现

### ✅ 主要发现  

1. **连续性（Contiguity）是修复 stale KV cache 的关键**：
   - 成功修复依赖于 **重建从前向输入到输出的完整依赖链**
   - 孤立地刷新“高重要性”位置无效，因其读取的是 stale 上下文

2. **EDITLOCAL 是强基线**：
   - 在编辑与答案相邻的场景下，$ K=32 $ 的连续窗口即可恢复 ≥94% 的 margin
   - 实现简单、无需训练、无需额外前向、速度极快（13–21× 快于 full re-prefill）

3. **unconditional repair 是合理策略**：
   - 凡涉及答案链的编辑，stale cache 几乎必然失败（failure rate ≥ 98.8%）
   - 失败严重性难以预测（最佳预测模型 Spearman ρ = 0.17）
   - 修复成本极低 → 支持无条件应用 EDITLOCAL

4. **transplant-based ranking 高估修复潜力**：
   - 很多 cache editing 方法使用“clean state transplant”来评估位置重要性
   - 但这高估了真实 recomputation 的能力，不应作为 deployable 方法的性能上界

---

### ⚠️ 方法的局限性  

- **仅适用于邻接依赖场景**：当答案位于数百 tokens 外时，EDITLOCAL 失效
- **假设单一、长度不变的编辑**：不支持多编辑、增删文本等复杂变更
- **合成数据构造**：所有 edits 均为 answer-relevant，现实中外显相关性更低
- **模型规模限制**：集中在 ~8B dense 模型，未验证在更大或 MoE 模型上的泛化性
- **未考虑 positional correction**：对于长度变化的编辑需额外处理 RoPE 等机制（见 Leyline）

---

### 🔮 未来工作方向  

1. **设计跨 block 的修复机制**：
   - 如两阶段修复：先定位 carrier block，再重算路径
   - 引入轻量级 search + local repair pipeline

2. **探索 adaptive budget allocation**：
   - 动态决定 $ K $ 或修复范围（如直到下一结构边界）

3. **结合 retrieval-aware repair**：
   - 利用文档结构元信息指导修复区域选择

4. **扩展至 multi-edit 和 streaming update 场景**

5. **研究更复杂的答案格式**（非二值问答）

---

## ✅ 总结一句话  

> **在 KV cache 修复中，“连续性”比“重要性”更重要：与其寻找最关键的几个 token，不如简单地重算编辑点之后的一段连续文本——只要依赖关系仍然邻接，这就是最快、最稳、最有效的做法。**

</details>

---

### 15. [A Distributed Computing Framework for Satellite Swarms](https://arxiv.org/abs/2609.18839)

**Authors**: Ezra Fielding (CNES), Clement Demazure (IRIT, Toulouse INP), Guthemberg Silvestre (ENAC), Felipe Alves Suana (CNES), Philippe Qu\'einnec (IRIT-ACADIE, Toulouse INP)  
**Category**: cs.DC  
**Published**: 2026-09-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.18839v1  

#### Abstract
The rise of large satellite constellations and Distributed Space Systems (DSS) demands generalized frameworks that enable fault-tolerant, autonomous distributed space applications. Conventional ground-centric command and control does not scale to systems of tens or hundreds of satellites, motivating...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《A Distributed Computing Framework for Satellite Swarms》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
随着大型卫星星座（satellite constellations）和分布式空间系统（Distributed Space Systems, DSS）的快速发展，传统的“地面中心化”指挥控制架构（ground-centric command and control）已无法有效扩展至数十甚至数百颗卫星的协同运行。这种集中式模式导致：
- 地面通信负担过重（需逐个上行指令）
- 系统容错能力差
- 难以实现自主协同与实时响应

因此，亟需一种**可扩展、容错性强、支持自主操作**的分布式计算框架来支撑卫星群（swarm）级任务。

### 提出的新方法与新思路
本文提出了一种**面向卫星群的分布式计算概念框架**，其核心创新包括：

- **三层抽象架构设计**：
  1. **Distributed State**（分布式状态）：构建共享状态基础
  2. **Command and Control**（命令与控制）：实现对整个 swarm 的统一操控
  3. **Mission / Science**（科学任务层）：支持自主科学观测与动态任务调度

- 在底层实现了基于 **Conflict-free Replicated Data Types (CRDT)** 的强最终一致性（Strongly Eventually Consistent, SEC）分布式状态服务，具体采用 **Last-Write-Wins Register (LWW-Register)** 构建 key-value 存储，用于在卫星间高效同步 Space Situational Awareness (SSA) catalog 数据。

- 利用 **ISL（Inter-Satellite Link）网络进行数据扩散**，减少对地面站（Ground Station, GS）的依赖。

### 相比现有方法的优势
| 维度 | 传统方法（Direct Uplink） | 本论文方法（CRDT + ISL） |
|------|--------------------------|----------------------------|
| **GS 上行消息数** | 每次更新需向所有卫星单独发送（O(N)） | 仅需单次上行（O(1)） |
| **时间开销** | 受限于轨道周期和地面可见窗口，传播慢 | 利用 ISL 快速全网扩散 |
| **容错性** | 单星失效不影响其他，但整体依赖地面 | 支持网络分区恢复，“self-healing”机制 |
| **可扩展性** | 不适用于大规模星座 | 天然适合百颗级以上 swarm |

> ✅ **核心优势**：显著降低地面通信频率与延迟，提升系统自主性与鲁棒性。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **SSA Catalog 数据**：由 CelesTrak 提供的 **General Perturbation (GP) 数据**，包含 5 个轨道目标一年内的 TLE（Two-Line Element）信息。
- 数据以 JSON 格式组织，每个对象通过唯一 ID 索引。

### 实验设置
- **仿真平台**：使用 **GoNetEm** 网络模拟器，结合 Docker 容器模拟卫星节点。
- **星座拓扑**：基于 **Iridium 星座**构建的 66 颗卫星 Walker Star 型星座（6 轨道面 × 11 颗），每颗卫星最多连接 4 个邻接节点。
- **通信参数配置**：
  - ISL 带宽：200 Kbps
  - 延迟：250 ms（含 125 ms 接口延迟）
  - 抖动：25 ms
  - 丢包率：0.001%
  - 所有链路使用 UDP 协议传输 CRDT 更新消息

### 评估指标
- **GS Messages**：从地面站发出的上行消息总数
- **ISL Messages**：卫星间转发的消息总数
- **Total Messages**：总通信量
- **传播时间**：从首次上行到全网收敛所需时间
- **消息增长趋势**（多更新场景下的 scalability 分析）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Direct Uplink (No ISL)** | 类似当前“bent-pipe”架构，地面逐个上行更新，无星间通信 |
| **CRDT + Flooding** | 地面上行一次，其余靠洪泛法（flooding）在 ISL 中扩散 |
| **CRDT + Tree Traversal** | 使用生成树（spanning tree）进行定向广播，减少冗余消息 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（单次更新）

| 测试案例 | GS Messages | ISL Messages | 总消息数 |
|---------|-------------|--------------|----------|
| Direct Uplink (No ISL) | **66** | 0 | **66** |
| CRDT + Flooding | **1** | 177 | **178** |
| CRDT + Tree Traversal | **1** | **65** | **66** |

> 🔍 注：Flooding 理论应发送 `2E - N + 1 = 2×132 - 66 + 1 = 199` 条消息，实际为 177，说明存在优化或部分路径未激活；Tree Traversal 接近理论最优 `N−1=65`。

### 多更新并发场景结果（图 4）
- 当同时注入 **1~20 次并发更新**时：
  - GS 上行消息数 = 并发源数量（如 20 次更新来自 20 颗不同卫星，则为 20）
  - ISL 消息呈现 **sub-linear 增长趋势**
    - 原因：节点若已收到更高版本（timestamp 更大），会直接丢弃旧版本更新，避免无效传播
    - 表现出良好的**抗冗余更新能力**

### 与基线方法对比结果
- **GS 通信减少 98.5%**：从 66 次降至 1 次 per update
- **传播速度大幅提升**：
  - Direct Uplink：按 5 秒间隔依次联系，共需 **330 秒**
  - CRDT + ISL：**< 5 秒内完成全网同步**
- **Tree Traversal 比 Flooding 节省 ~63% ISL 流量**（65 vs 177）

### 消融实验分析（隐含）
虽然未明确标注“ablation study”，但以下对比体现了消融思想：
- **是否启用 ISL** → 影响 GS 消息数量（关键变量）
- **不同扩散策略（Flooding vs Tree）** → 影响 ISL 消息效率
- **单更新 vs 多更新** → 验证 CRDT 的冲突解决与去重能力

---

## 4. 关键结论和发现

### 主要发现
1. **CRDT 是实现卫星群分布式状态管理的有效手段**：
   - 支持异步、无序消息传递
   - 自动解决并发写入冲突（last-write-wins）
   - 无需全局共识（consensus），轻量且高效

2. **分布式状态是构建自主卫星 swarm 的基石**：
   - 实现“single entity”操作的关键前提
   - 为后续 Command & Control 和 Mission Planning 层提供数据支撑

3. **ISL 网络可用于高效数据分发**，大幅降低对地面基础设施的依赖，尤其适合 SSA、遥感等需要频繁更新元数据的任务。

4. **系统具备“自愈”能力（self-healing）**：
   - 即使出现短暂网络分区或节点失效，最新状态仍可通过后续更新覆盖旧状态，保障最终一致性。

### 方法的局限性
- **LWW-Register 对时钟同步有一定要求**：依赖逻辑时钟（如 Lamport Clock）保证因果顺序；若物理时钟偏差过大可能导致错误覆盖。
- **不适用于强一致性需求场景**：如轨道机动控制、编队飞行等需严格顺序执行的操作，仍需引入分布式共识算法（如 Paxos/Raft）。
- **当前验证基于静态拓扑**：低轨星座（LEO）中 ISL 连接具有时变性（time-varying connectivity），动态拓扑下的表现有待进一步测试。
- **未考虑安全机制**：如消息认证、防篡改等，在真实部署中必不可少。

### 未来工作方向
1. **引入分布式共识机制**：针对需要 linearizable consistency 的应用，研究 Raft/Paxos/BFT 在星载环境中的可行性。
2. **扩展高层服务**：
   - 实现 Command & Control 层的 leader election、task allocation
   - 开发 Mission Planning 层的自主决策模块
3. **真实轨道动力学集成测试**：将该框架嵌入 CNES 的 KOSMOS 飞行软件栈，在动态轨道环境中验证性能。
4. **硬件在环（HIL）与在轨验证**：推动在真实 CubeSat 或 nanosat 平台上的部署试验。
5. **与其他 swarm 项目对比评估**：例如 NASA 的 **Starling mission**，量化本框架相对于现有方案的优势。

---

> 📌 **总体评价**：  
> 本文提出了一个前瞻性的 **Distributed Computing Framework for Satellite Swarms**，并通过 CRDT 在 SSA catalog 分发任务中的成功应用，证明了其在降低地面依赖、提高可扩展性和容错性方面的巨大潜力。是迈向**真正自主化、智能化太空 swarm 系统**的重要一步。

</details>

---

### 16. [FedGuide: Diffusion Prior Alignment and Value Baseline Guidance for Heterogeneous Federated Reinforcement Learning](https://arxiv.org/abs/2609.18964)

**Authors**: Zhilin He, Gauri Joshi  
**Category**: cs.LG  
**Published**: 2026-09-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2609.18964v1  

#### Abstract
Federated Reinforcement Learning (FRL) enables collaborative policy learning across distributed agents with heterogeneous environments. While recent methods based on variance reduction, divergence penalization, and momentum optimization improve FRL under heterogeneous settings, they still primarily ...

---

### 17. [The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction](https://arxiv.org/abs/2609.18063)

**Authors**: Yu Lin, Yiming Wang, Runyuan Cai, Hanze Liu, Xiaodong Zeng  
**Category**: cs.AI  
**Published**: 2026-09-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.18063v1  

#### Abstract
Mixture-of-experts (MoE) inference on consumer hardware is bounded by weight memory: a 35B-class model is 19.5GB at 4-bit, and sparsity shrinks the compute per token, not the bytes that must be held. Naive offloading to SSD does not help on its own, because layer N+1's experts must be chosen before ...

---

### 18. [Infinite-Parameter LLMs: Generating and Adapting Weights from Live Data](https://arxiv.org/abs/2609.18842)

**Authors**: Jinli Hu, Ross M. Clarke, Yichuan Zhang, Jos\'e Miguel Hern\'andez-Lobato  
**Category**: cs.AI  
**Published**: 2026-09-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.18842v1  

#### Abstract
The scaling laws hold that a language model grows more capable with more parameters and more training data, and Mixture-of-Experts (MoE) architectures have ridden these laws to remarkable results, activating only a fraction of an enormous stored parameter bank for each token. That success is built o...

---

### 19. [Dependency-Aware Trajectory Refinement for Efficient Multi-Turn Agent Fine-Tuning](https://arxiv.org/abs/2609.18417)

**Authors**: Zhuo Chen, Zhen Zhang, Xinyu Wang, Kewei Tu  
**Category**: cs.CL  
**Published**: 2026-09-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.18417v1  

#### Abstract
Multi-turn agent trajectories often contain redundant rounds (failed tool calls, parallel sub-queries, verification-only steps) that inflate both training and inference cost. We propose viewing each trajectory as a \emph{round-level dependency DAG} that exposes which rounds are globally load-bearing...

---

### 20. [NObSP: Functional Decomposition of Neural Networks via Oblique Subspace Projections](https://arxiv.org/abs/2609.17825)

**Authors**: Alexander Caicedo, V\'ictor De La Hoz, Santiago Alf\'erez  
**Category**: cs.LG  
**Published**: 2026-09-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.17825v1  

#### Abstract
Understanding how deep neural networks make decisions remains a fundamental challenge. We present NObSP (Nonlinear Oblique Subspace Projections), a framework that decomposes predictions into explicit per feature contribution functions and an interaction residual. NObSP exploits the linear final laye...

---

### 21. [A Convergence Framework for Deep $V$-Learning: Error Propagation and Sharp Action-Gap Bounds](https://arxiv.org/abs/2609.18782)

**Authors**: Yury Kolomeytsev  
**Category**: cs.LG  
**Published**: 2026-09-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.18782v1  

#### Abstract
We establish convergence bounds for deep $V$-learning with horizon $H$. The algorithm fits a scalar value function to targets from executed transitions and selects actions using a predictive model and the value function. For current observed-successor targets with fresh true-kernel outcomes, the con...

---

### 22. [Integrated Optimization of Automated Warehouse Operations and Last-Mile Transport for Differentiated On-Demand Delivery](https://arxiv.org/abs/2609.19048)

**Authors**: Xiaozhu Sun, Bilal Farooq  
**Category**: cs.LG  
**Published**: 2026-09-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2609.19048v1  

#### Abstract
In the context of differentiated on-demand goods delivery services, this study proposes an integrated optimization method for automated guided vehicles (AGVs) based smart warehouse operations and the last-mile multi-modal transport. A deep reinforcement learning algorithm for multi-objective joint s...

---

### 23. [Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents](https://arxiv.org/abs/2609.18304)

**Authors**: Yi Yu, Liuyi Yao, Yaliang Li, Enshu Wang, Libing Wu  
**Category**: cs.CL  
**Published**: 2026-09-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.18304v1  

#### Abstract
Large language model (LLM) agents increasingly tackle long-horizon tasks through multi-step environment interaction, yet a single erroneous action can alter subsequent states and observations, causing errors to compound over time. Existing methods either correct the context without repairing altered...

---

### 24. [Align, Integrate, and Fire: Efficient Token-Level Alignment for Zero-Shot SpeechLLMs](https://arxiv.org/abs/2609.18516)

**Authors**: Abderrahmane Issam, Yusuf Can Semerci, Jan Scholtes, Gerasimos Spanakis  
**Category**: cs.CL  
**Published**: 2026-09-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.18516v1  

#### Abstract
While Large Language Models excel in natural language processing, efficiently extending their capabilities to spoken input remains a significant challenge. Existing methods for building SpeechLLMs often rely on computationally expensive full-model fine-tuning, or employ parameter-efficient projector...

---

### 25. [SpecReuse: Spectral Graph Reuse for Efficient Vision GNN Inference on FPGAs](https://arxiv.org/abs/2609.17718)

**Authors**: Isabella Bernhardt Eiliya, Anvitha Ramachandran, Dhruv Parikh, Viktor Prasanna  
**Category**: cs.DC  
**Published**: 2026-09-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.17718v1  

#### Abstract
Dynamic Image Graph Construction (DIGC) is the primary performance bottleneck in FPGA acceleration of Vision Graph Neural Networks (ViGs), reconstructing graph connectivity at every layer through irregular, memory-intensive computation. Existing FPGA accelerators optimize DIGC but still execute it u...

---

### 26. [From Pixels to Semantics: Edge AI for UAV-Based Critical Infrastructure Inspection](https://arxiv.org/abs/2609.18448)

**Authors**: Reza Farahani, Naser Hossein Motlagh, Zoha Azimi, Christian Timmerer, Lorenzo Carnevale, Sasu Tarkoma, Schahram Dustdar  
**Category**: cs.DC  
**Published**: 2026-09-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.18448v1  

#### Abstract
Critical infrastructure assets such as bridges, tunnels, dams, and power line networks require timely and scalable inspection. While conventional manual inspection remains costly and hazardous, unmanned aerial vehicle (UAV)-based inspection has emerged as an efficient alternative for monitoring diff...

---

### 27. [RayOrch: Programming and Executing Lineage-Controlled Multi-Grain Dataflows for Foundation-Model Data Preparation](https://arxiv.org/abs/2609.18703)

**Authors**: Xiaochen Ma, Zimo Meng, Junzhu Liang, Youhe Jiang, Yue Cheng, Hao Liang, Bohan Zeng, Dengchun Li, Lu Ma, Zhengyang Zhao, Zhen Hao Wong, Runming He, Meiyi Qiang, Jiangtao Guan, Binhang Yuan, Wentao Zhang  
**Category**: cs.DC  
**Published**: 2026-09-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.18703v1  

#### Abstract
Preparing high quality training data for foundation models requires scalable pipelines that transform heterogeneous documents and videos into structured records. Such pipelines expand each parent item into an ordered and input dependent sequence of children, whose counts may be long tailed. GPUs sho...

---

### 28. [Regularized Least Squares Training of Quadratic Neural Networks with Applications to System Identification](https://arxiv.org/abs/2609.17654)

**Authors**: Luis Rodrigues, Zachary Yetman Van Egmond, Mohammad R. Amiri Fard  
**Category**: cs.LG  
**Published**: 2026-09-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.17654v1  

#### Abstract
This paper proposes a least squares approach for the training of quadratic neural networks with regularization. The proposed methodology yields a lower bound on the solution of the training optimization problem for the case where the regularization coefficient is positive. Moreover, it yields closed...

---

### 29. [Hybrid coupling with numerics-informed neural networks and the overlapping Schwarz alternating method](https://arxiv.org/abs/2609.17841)

**Authors**: George Chumbipuma, Irina Tezaur, Alejandro Diaz, Beatrice Riviere  
**Category**: cs.LG  
**Published**: 2026-09-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.17841v1  

#### Abstract
We develop a hybrid modeling framework for coupling pre-trained numerics-informed neural networks (NINNs) with classical full order models (FOMs) using the overlapping Schwarz alternating method. We consider the two-dimensional advection-diffusion equation in the advection-dominated, Peclet-number 1...

---

### 30. [Beyond the Previous Layer: Residual Predictive Structure in Sparse MoE Routing](https://arxiv.org/abs/2609.17940)

**Authors**: Hao Li, Yasuyuki Tahara, Yuichi Sei  
**Category**: cs.LG  
**Published**: 2026-09-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2609.17940v1  

#### Abstract
Sparse mixture-of-experts models route each token through a sequence of expert selections. We ask whether the immediately preceding selection adequately summarizes this trajectory for predicting the next router. Using frozen OLMoE and JetMoE models, we measure the held-out predictive gain from earli...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-26 08:58:59 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Bandwidth-Aware LLM Inference on Heterogeneous Many-Core Supercomputers](https://arxiv.org/abs/2605.25655)

**Authors**: Yao Lu, Zhongzhi Luan, Gen Li, Jiaxing Qi, Shiqing Ma, Bin Han, Shizhe Shang, Hailong Yang, Depei Qian  
**Category**: cs.DC  
**Published**: 2026-05-26  
**Score**: 18.0  
**Type**: new  
**ArXiv ID**: 2605.25655v1  

#### Abstract
Large language model (LLM) inference is limited by high computational cost and memory bandwidth demands, making deployment on heterogeneous many-core processors challenging. Taking the MT-3000 processor used in the Tianhe supercomputer as an example, its limited main-memory bandwidth and distributed...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对在**异构众核超级计算机**（如基于MT-3000处理器的天河新一代超算）上部署大规模语言模型（LLM）推理所面临的三大挑战：
- **有限的主存带宽**：MT-3000的内存带宽仅为约120 GB/s，远低于GPU（如V100可达900+ GB/s），导致I/O成为瓶颈。
- **分布式内存层次结构**：片上SRAM容量小，数据重用困难，难以支持大模型的KV Cache存储。
- **可扩展性瓶颈**：数十亿参数模型无法单集群容纳，跨集群通信开销大，传统并行策略难以扩展。

这些问题使得现有的GPU优化框架（如DeepSpeed、vLLM等）无法直接迁移至此类平台。

---

### 提出的新方法与创新思路
作者提出了名为 **THInfer** 的硬件感知LLM推理框架，通过**软硬件协同设计**（hardware-software co-design）实现高效推理。其核心创新分为三个层面：

#### （1）高性能算子库（Operator-Level）
- 针对MT-3000的VLIW-SIMD架构，手工编写汇编级优化的FP16 GEMM内核。
- 利用**数据流分析**进行细粒度调度，结合**三缓冲机制**和**动态分块**（dynamic tiling）来隐藏DMA传输延迟。
- 实现单个计算簇（cluster）达到理论峰值性能的**70%以上**。

#### （2）密度驱动的计算图融合（Algorithm-Level）
- 提出“低密度-高密度-低密度”算子融合策略，将RoPE、Add、Scale等轻量操作嵌入到Linear或Attention等重计算算子中。
- 设计专用的 **MT Attention** 机制，替代Flash Attention，适配MT-3000的AM/SM/GSM三级存储体系。
  - 使用广播（broadcast）而非频繁DMA读写KV矩阵。
  - 分阶段流水处理Attention Score，减少冗余I/O。
- 引入**统一Kernel调度**，避免每个Attention Head单独启动Kernel带来的调度开销。

#### （3）自适应并行调度机制（System-Level）
- 构建 **Prefill-Buffer-Decode (P-B-D)** 三级同步流水线，在微批次粒度上解耦预填充与解码阶段。
- 支持混合并行（Hybrid Parallelism）：
  - **簇内**：使用hthreads实现多核任务调度。
  - **簇间**：基于MPI进行两级通信。
- 引入**有界缓冲区**与**反压控制机制**，防止KV Cache无限增长导致OOM。

---

### 相比现有方法的优势
| 维度 | THInfer优势 |
|------|------------|
| **硬件适配性** | 完全针对MT-3000定制，充分利用其VLIW-SIMD、软件控制内存、广播机制等特性 |
| **内存效率** | 通过算子融合、统一Kernel、广播通信显著降低I/O压力，缓解带宽瓶颈 |
| **可扩展性** | P-B-D流水线+混合并行支持百亿参数模型稳定运行，而GPU方案会OOM |
| **吞吐量** | 在同等算力条件下，吞吐优于或媲美V100/A800上的DeepSpeed |

---

## 2. 核心实验方法和设置

### 使用的数据集
- 使用**合成提示数据集**（synthetic prompt dataset），所有输入序列长度统一为 **512 或 1024 tokens**。
- 每个请求生成 **128个输出token**，模拟典型的文本生成负载。

> 注：未使用真实自然语言数据集（如ShareGPT），但强调技术适用于所有Transformer类LLM。

---

### 实验设置
- **硬件平台**：
  - **MT-3000节点**：国产DSP芯片，每节点含4个加速簇，FP16峰值32.4 TFLOPS，主存带宽120 GB/s。
  - 对比设备：Tesla V100S-PCIE-32GB（1134 GB/s）、NVIDIA A800 80GB PCIe（1935 GB/s）。
- **模型系列**：Llama2系列，涵盖 **7B、13B、30B 和 70B** 参数规模。
- **评估指标**：
  - **端到端吞吐量**（Throughput）：单位为 tokens/s。
  - **延迟**（Latency）：首Token延迟（TTFT）及总响应时间。
  - **资源利用率**：内存占用、带宽使用率、计算效率。

---

### 基线方法对比
- **DeepSpeed-Inference**：主流分布式推理框架，支持Tensor Parallelism。
- **Hugging Face Accelerate**：支持Pipeline Parallelism的基础工具链。
- 所有对比均遵循“**峰值性能对齐**”原则：
  - 8个MT-3000 ≈ 2×V100S
  - 10个MT-3000 ≈ 1×A800
- 另外还设置了“**带宽对齐**”配置（18/16台MT-3000）以验证系统扩展能力。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table V & VI）

#### 吞吐量对比（tokens/s）

| 模型 | 输入长度 | DeepSpeed (A800) | THInfer (Peak-Aligned) | 提升幅度 |
|------|---------|------------------|------------------------|----------|
| Llama-7B | 512 | 584 | **975** | ↑67% |
| Llama-7B | 1024 | 310 | **570** | ↑84% |
| Llama-13B | 512 | 345 | **301** | ↓13% |
| Llama-13B | 1024 | 184 | **211** | ↑15% |
| Llama-30B | 512 | 116 | **105** | ≈持平 |
| Llama-30B | 1024 | 60 | **61** | ≈持平 |
| Llama-70B | 512 | OOM | **64** | ✅成功运行 |
| Llama-70B | 1024 | OOM | **41** | ✅成功运行 |

> 💡 **说明**：虽然在小模型上THInfer已展现明显优势；而在大模型（尤其是70B）上，**GPU基线因OOM失败**，而THInfer仍能提供有效推理服务。

---

### 与V100S对比（Table V）
| 模型 | 输入长度 | DeepSpeed (2×V100S) | THInfer (8×MT-3000) | 提升幅度 |
|------|---------|---------------------|----------------------|----------|
| Llama-7B | 512 | 781 | **1755** | ↑125% |
| Llama-7B | 1024 | 456 | **1026** | ↑125% |
| Llama-70B | 512 | OOM | **64** | ✅唯一可行方案 |

> ⚠️ 注意：此处为**带宽对齐**下的测试结果（18台MT-3000），体现极致扩展潜力。

---

### 消融实验结果（Ablation Study, Table VII）
逐步引入优化模块后，30B模型的吞吐变化如下：

| 阶段 | 优化内容 | 吞吐量 (tokens/s) | 相比前一阶段提升 |
|------|--------|------------------|----------------|
| A0 | 无优化 | 0.07 | — |
| A1 | 算子优化（FP16 + 手写Kernel） | 2.47 | ↑35倍 |
| A2 | 图调度优化（融合+MT Attention） | 3.02 | ↑22% |
| A3-1 | 选择性批处理（Selective Batching + PP） | 14.00 | ↑363% |
| A3-2 | 加入Tensor Parallelism | 27.00 | ↑93% |
| A4 | P-B-D三级流水线 | **34.00** | ↑26% |

> 🔍 发现：随着模型增大，**系统级优化**（如P-B-D流水线）带来的增益更为显著。

---

## 4. 关键结论和发现

### 主要发现
1. **THInfer在异构众核平台上实现了高效的LLM推理**，尤其适合**高计算密度、低带宽**的国产超算环境。
2. **软硬件协同设计至关重要**：仅靠算法优化无法突破硬件瓶颈，必须结合底层指令级优化（如VLIW-SIMD汇编）才能释放全部性能。
3. **MT Attention显著优于Flash Attention** 在MT-3000上的表现：
   - 减少DMA次数，改用高速广播。
   - 表格IX显示，在S=4096时延迟降低**超过50%**。
4. **P-B-D流水线解决了KV Cache膨胀问题**，相比DistServe等完全异步方案更具稳定性。
5. **THInfer具备卓越的可扩展性**：即使面对70B模型也能稳定运行，而GPU方案崩溃。

---

### 方法的局限性
- **高度依赖特定硬件架构**：目前仅针对MT-3000设计，移植到其他非SIMD架构可能需重构算子库。
- **开发复杂度高**：需要手动编写大量汇编代码，对开发者要求极高。
- **缺乏对稀疏化、量化等前沿压缩技术的支持**（文中未涉及Sparsity或MoE）。
- **未测试真实交互场景下的动态批处理性能**，主要面向高吞吐离线生成任务。

---

### 未来工作方向
1. 将THInfer扩展至更多异构架构（如GPU+FPGA协同）。
2. 结合**动态稀疏推理**与**运行时负载感知调度**，进一步提升能效比。
3. 探索**自动Kernel生成器**（类似TVM），降低手写汇编的维护成本。
4. 支持更复杂的推理模式，如**流式生成**（streaming）、**函数调用**（function calling）等。
5. 开源框架部分组件，推动国产超算生态建设。

---

> ✅ **总体评价**：  
> THInfer是面向国产超算平台的一次重要工程实践，证明了在**低带宽、强异构**环境下依然可以实现高效LLM推理。它不仅是一个推理引擎，更是**软硬一体化设计范式的典范**，为未来AI-HPC融合提供了关键技术路径。

</details>

---

### 2. [Bandwidth-Aware and Cost-Efficient Pipeline Parallel Scheduling in Geo-Distributed LLM Training](https://arxiv.org/abs/2605.25375)

**Authors**: Han Zhang, Jianchun Liu, Hongli Xu  
**Category**: cs.DC  
**Published**: 2026-05-26  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2605.25375v1  

#### Abstract
The rapid evolution of large language models (LLMs) has made geographically distributed training necessary due to GPU scarcity within a single cloud region. In such cross-region settings, Pipeline Parallelism (PP) is communication-efficient, yet scheduling PP remains challenging under heterogeneous ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题：** *Bandwidth-Aware and Cost-Efficient Pipeline Parallel Scheduling in Geo-Distributed LLM Training*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模 **Large Language Model (LLM)** 训练中，单个云区域内的 GPU 资源日益稀缺，导致训练任务必须跨地理区域（geo-distributed）进行调度。然而，现有调度器面临以下挑战：
- **带宽异构性**：跨区域网络带宽差异大，通信瓶颈严重降低 Pipeline Parallelism (PP) 效率。
- **电力成本差异**：不同地区的电价差异显著，影响总体训练成本。
- **资源碎片化与 Head-of-Line (HoL) 阻塞**：长耗时、高带宽需求的任务会阻塞后续更高效的小任务，导致平均 Job Completion Time (JCT) 上升。

传统调度策略要么是 **Delay-First**（如 Crux、CASSINI），追求低延迟但忽略电费开销；要么是 **Cost-First**（如 TanGo），优先选择低价区但牺牲性能或缺乏动态适应能力。

---

### 提出的新方法：BACE-Pipe
作者提出 **BACE-Pipe** —— 一种面向跨区域 LLM 训练的 **带宽感知且成本高效的 Pipeline 并行调度框架**，其核心由控制平面（Control Plane）和数据平面（Data Plane）组成。

#### 创新机制
1. **动态任务优先级机制（Dynamic Job Prioritization）**
   - 综合考虑任务的 **计算强度（Computation Intensity）** 和 **带宽敏感度（Bandwidth Sensitivity）**
   - 引入自适应权重因子 α，根据实时网络拥塞程度调整优先级评分：
     $$
     \text{Priority}_j = (1-\alpha)(1-I_j) + \alpha(1-D_j)
     $$
   - 在网络空闲时优先短任务（减少 HoL 阻塞），在网络拥塞时优先低带宽需求任务。

2. **带宽感知路径查找器（Bandwidth-Aware Pathfinder）**
   - 构建满足通信约束的跨区域 Pipeline 路径。
   - 使用类 Prim 算法贪心扩展路径，确保每一步添加的链路不会成为通信瓶颈（即通信时间 ≤ 计算时间）。
   - 支持多区域聚合 GPU，突破单区域容量限制。

3. **成本最小化分配器（Cost-Min Allocator）**
   - 在可行路径基础上，将 GPU 尽可能分配到电价更低的区域。
   - 先保证每个阶段至少一个 GPU（维持流水线连续性），再将剩余 GPU 按电价升序填充。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **性能优化** | 同时优化 JCT 与总电费，避免单一目标带来的次优解 |
| **调度灵活性** | 动态重排序任务队列，打破 FCFS 的僵化顺序 |
| **通信效率保障** | 显式建模通信-计算平衡，防止 pipeline bubbles |
| **经济性提升** | 利用全球电价差异，在不牺牲性能的前提下降低成本 |

---

## 2. 核心实验方法和设置

### 数据集与模型配置
- **训练任务**：共 8 个 LLM 训练任务，参数规模从 **14B 到 101B** 不等，涵盖主流架构如：
  - `Llama-3.1-70B`, `Qwen2.5-14B/32B`, `Gemma-3-27B`, `FLM-101B` 等
- **数据集多样性**：随机分配三种典型数据集以模拟真实负载：
  - **Alpaca-52k**（小规模指令微调）
  - **WikiText-103**（中等语言建模）
  - **OpenWebText**（大规模预训练语料）

---

### 实验环境设置
- **模拟环境**：基于 6 个全球代表性区域构建 geo-distributed 集群：
  - EU-West (Ireland), US-East-2 (NY), EU-Central (Germany), EA-East (Tokyo), SEA-South (Singapore), OC-East (Sydney)
- **资源配置**：
  - GPU 数量：16–128 张 / 区域（NVIDIA A6000）
  - 带宽：25–100 Gbps（内部），跨区域带宽取两端均值
  - 电价：来源于 GlobalPetrolPrices 的真实商业电价（$0.156–$0.295/kWh）

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **Avg. JCT** | 所有任务从提交到完成的平均耗时 |
| **Total Electricity Cost** | 所有任务执行期间消耗的总电费（仅活跃期计费） |

---

### 基线方法对比
| 基线 | 类型 | 简介 |
|------|------|------|
| **LCF** | Cost-First | 单区域部署，选电价最低区域 |
| **LDF** | Delay-First | 单区域部署，选 GPU 最多区域 |
| **CR-LCF** | Cross-Region Cost-First | 多区域聚合，按电价升序选区 |
| **CR-LDF** | Cross-Region Delay-First | 多区域聚合，按带宽优先扩展路径 |

所有方法统一使用最优 GPU 数 $K^*$（通过最小化 `t_iter` 得出），确保公平比较。

---

## 3. 主要实验结果和性能指标

### 总体性能表现（图4）
BACE-Pipe 在两个核心指标上全面超越所有基线：

| 指标 | 提升幅度（vs. 最佳基线） |
|------|--------------------------|
| **Avg. JCT** | ↓ **27.9% – 64.7%** |
| **Total Electricity Cost** | ↓ **12.6% – 30.6%** |

> 示例：在标准设置下，BACE-Pipe 的平均 JCT 仅为 CR-LDF 的 ~57%，成本为 LCF 的 ~70%。

---

### 关键现象：“Cross-Region Paradox”
令人惊讶的是，**跨区域方法（CR-LCF / CR-LDF）反而比单区域方法更差**：
- CR-LDF 的平均 JCT 比 LDF **高出 28.8%**
- CR-LCF 比 LCF **高出 13.1%**

**原因分析**：大型任务早期抢占大量 GPU 和关键跨区域链路，造成严重的 **Head-of-Line Blocking**，使后续小任务长期等待，系统吞吐下降。

✅ BACE-Pipe 通过动态优先级机制有效缓解该问题。

---

### 敏感性分析结果

#### （1）带宽波动测试（×0.3 vs ×1.5）
| 场景 | 表现 |
|------|------|
| **低带宽（0.3×）** | BACE-Pipe 成本仍低 **29.2%–34.9%**，因能更好利用有限路径 |
| **高带宽（1.5×）** | BACE-Pipe JCT 优势扩大至 **42.9%–240.3%**，体现其路径搜索优越性 |

> 结论：带宽越丰富，BACE-Pipe 的 Pathfinder 越能发挥优势。

#### （2）GPU 容量变化（×0.5 vs ×1.25）
| 场景 | 表现 |
|------|------|
| **资源紧张（0.5×）** | BACE-Pipe JCT 降低 **32.2%–69.9%**，凸显其抗 HoL 能力 |
| **资源充足（1.25×）** | 差距缩小（JCT +5.5%~20.7%，成本 +0.2%~9.4%），但仍最优 |

> 结论：BACE-Pipe 在资源受限场景中价值最大。

#### （3）工作负载强度（8 → 24 个任务）
随着并发任务增加，各方法差距略有收敛，但：
- BACE-Pipe 仍保持 **9.7%–23.3%** 的 JCT 优势
- 成本优势在重载下趋近（≈1%），因几乎所有 GPU 都被占用

> 表明 BACE-Pipe 在高压力下依然稳定可靠。

---

### 消融实验（Ablation Study）

| 变体 | Avg. JCT ↑ | Total Cost ↑ | 分析 |
|------|-----------|--------------|------|
| **w/o Priority** | +41.9% | +5.0% | 缺少动态排序导致 HoL 阻塞加剧 |
| **w/o Pathfinder** | +52.5% | +20.5% | 无法构建高效跨区域路径，性能崩溃 |
| **w/o Cost-Min** | +4.6% | +13.9% | 成本上升主因；轻微 JCT 上升源于资源碎片化 |

> ✅ 三者协同作用显著，缺一不可。

---

## 4. 关键结论和发现

### 主要发现
1. **跨区域调度需兼顾通信与成本**：单纯追求低延迟或低成本都会导致整体性能退化。
2. **“Cross-Region Paradox” 存在**：盲目聚合资源可能导致更差的平均 JCT，尤其在多任务环境下。
3. **动态优先级至关重要**：固定 FCFS 或静态策略无法应对复杂资源竞争。
4. **路径构造决定上限**：能否找到高质量跨区域 Pipeline 路径是性能基石。
5. **成本优化间接提升性能**：集中分配可减少资源碎片，利于后续任务调度。

---

### 方法局限性
| 局限 | 说明 |
|------|------|
| **静态 PP 规模** | GPU 数量和 pipeline 结构在运行前确定，不支持 runtime 动态调整 |
| **依赖准确 profiling** | 需预先获取任务的 `t_comp`, `A`（activation size）等元数据 |
| **未考虑故障恢复** | 假设节点稳定，未处理机器宕机或网络中断 |
| **集中式调度瓶颈** | 控制平面为单点，可能在超大规模集群中成为瓶颈 |

---

### 未来工作方向
- 支持 **runtime 自适应 PP 调整**（如弹性扩缩容）
- 引入 **去中心化调度架构** 提升可扩展性
- 结合 **energy-aware cooling models** 进一步优化 PUE
- 探索与 **Data Parallelism / Tensor Parallelism** 的混合并行调度
- 集成 **carbon intensity** 指标实现绿色 AI 训练

---

## 总结
**BACE-Pipe** 是首个在 **geo-distributed LLM 训练场景中联合优化 JCT 与电费** 的调度框架。它通过 **动态优先级 + 带宽感知路径搜索 + 成本最小化分配** 的三层机制，成功解决了跨区域训练中的通信瓶颈、资源争抢与经济性难题。实验证明其在多种负载和资源条件下均显著优于现有方法，尤其在资源紧张和带宽丰富的极端情况下表现突出，具备强鲁棒性和实用前景。

</details>

---

### 3. [Accelerating Long-Tail Generation in Synchronous RLHF Training via Adaptive Tensor Parallelism](https://arxiv.org/abs/2605.23945)

**Authors**: Long Zhao, Qinghe Wang, Jiaan Zhu, Youhui Bai, Zewen Jin, Chaoyi Ruan, Shengnan Wang, Cheng Li  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.23945v1  

#### Abstract
Reinforcement Learning from Human Feedback (RLHF) has become a key post-training paradigm for improving model quality. However, the synchronous three-stage RLHF pipeline is often bottlenecked by the generation stage, where response-length skew causes the effective batch size to shrink rapidly during...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerating Long-Tail Generation in Synchronous RLHF Training via Adaptive Tensor Parallelism

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在同步的 **Reinforcement Learning from Human Feedback (RLHF)** 训练流程中，**generation 阶段** 是主要性能瓶颈。由于响应长度存在显著偏斜（response-length skew），即大多数样本较短而少数样本极长，导致在生成后期进入“长尾阶段”（tail phase）时，有效 batch size 急剧下降，大量 GPU 资源被闲置，利用率严重不足。

主流框架采用静态的 **Tensor Parallelism (TP)** 配置，在整个生成过程中无法适应动态变化的 batch 特征，造成资源浪费。

### 提出了什么新方法或新思路
本文提出 **PAT (Progressive Adaptive Tensor Parallelism)**，一种在 RLHF 生成阶段内动态调整 TP/DP 配置的方法，以匹配不同阶段的负载特征：

- **Aligned Phase（对齐阶段）**：多数样本并行生成，batch size 大 → 使用低 TP / 高 DP 配置，最大化吞吐。
- **Tail Phase（尾部阶段）**：仅剩少量长序列 → 动态切换至高 TP / 低 DP 配置，降低单个样本的 decoding latency，提升 GPU 利用率。

PAT 包含两大核心技术：
1. **Predictor-guided Online Reconfiguration**  
   基于离线 profiling 构建延迟预测模型，在线判断是否触发重配置：只有当预期收益（剩余尾部延迟减少）超过切换开销时才执行切换。
   
2. **Lightweight Online Reconfiguration Mechanism**  
   - **KV Cache Handling**：在 KV cache 迁移与重新计算之间基于成本模型选择更优路径。
   - **Weight Resharding**：通过层粒度的 All-Gather + Slice 实现无需重新加载权重的原地重分片。
   - **Communication Group Reuse**：缓存已创建的通信组，避免重复初始化。

### 相比现有方法的优势
| 方法 | 是否改变 RLHF 语义 | 是否适用于同步训练 | 是否直接加速尾部 |
|------|---------------------|------------------------|--------------------|
| StreamRL [20] | 否（异步） | ❌ 不适用 | ⚠️ 间接隐藏 |
| RLHFuse [21] | 否（跨阶段重叠） | ✅ 有限 | ⚠️ 依赖准备阶段时长 |
| Kimi K2 [16] | ✅（部分 rollout） | ❌ 引入 off-policy 风险 | ✅ 加速但牺牲一致性 |
| **PAT (本工作)** | ✅ 完全保持同步语义 | ✅ 支持 | ✅ 直接优化尾部 decoding |

**优势总结**：
- 在不改变 RLHF 算法语义的前提下，显著提升同步训练效率；
- 自适应决策机制确保切换时机最优；
- 轻量级运行时重构将切换开销降低 90% 以上。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **DeepScaleR [9]**：一个注重逻辑推理能力的 RLHF 数据集，具有明显的响应长度分布偏斜，适合作为压力测试场景。

### 实验设置
- **硬件平台**：
  - A40 集群：单节点 8×A40（PCIe 4.0），双节点共 16 GPU（InfiniBand 100Gbps）
  - H100 单机：8×H100 SXM（NVLink 900 GB/s）
- **模型**：
  - LLaMA3.1-8B
  - Qwen3-14B
- **算法**：GRPO（一种 PPO 变体）
- **Batch Size**：128
- **最大响应长度**：从 6K 到 24K tokens 不等

### 评估指标
- **端到端吞吐量（Throughput）**：每秒处理的 token 数（tokens/s）
- **Generation Latency**：生成阶段耗时
- **End-to-End Iteration Latency**：完整 RLHF 迭代时间
- **Switching Overhead**：TP 切换过程引入的时间与内存开销
- **Prediction Accuracy**：延迟预测器的误差率

### 基线方法对比
- **VeRL [14]**：当前最先进的共址式（colocated）RLHF 框架，作为主要基线。
- 对每个配置组合（TP/DP）进行调优，选取最佳静态配置作为 baseline。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 硬件 | 最大长度 | 方法 | 吞吐量 (tokens/s) | 加速比 |
|-------|--------|----------|--------|----------------------|---------|
| LLaMA3.1-8B | 8×H100 | 24K | VeRL | 2570.6 | 1.0× |
| LLaMA3.1-8B | 8×H100 | 24K | **PAT** | **3510.9** | **1.37×** |
| Qwen3-14B | 8×H100 | 16K | VeRL | 1780.0 | 1.0× |
| Qwen3-14B | 8×H100 | 16K | **PAT** | **2292.6** | **1.29×** |

### 与基线方法的对比结果
- **生成延迟降低最多达 34.6%**（LLaMA3.1-8B @24K）
- **端到端训练迭代延迟降低最多达 27.2%**
- 随着最大响应长度增加，PAT 的优势更加明显（因 tail phase 占比更高）
- 在 A40 和 H100 平台上均取得稳定加速，且在高带宽 NVLink 上表现更优

### 消融实验结果
#### （1）切换开销分析（图 12）
| 开销项 | Naive 方法 | PAT 方法 | 减少比例 |
|--------|------------|----------|-----------|
| 权重重载与分片 | 7.47 s | 1.03 s | ↓ 86.2% |
| KV Cache 处理 | 19.01 s | 2.36 s | ↓ 87.6% |
| CUDA Graph 重建 | 3.59 s | 0.73 s | ↓ 79.6% |
| 总切换时间 | 58.98 s | **5.52 s** | **↓ 90.6%** |

> 切换开销仅占总运行时间约 **1.23%**，远低于其带来的收益。

#### （2）内存开销分析（图 13）
- PAT 在切换期间峰值内存仅增加 **2.5 GB**（vs. 全量迁移需 14 GB，可能 OOM）
- 稳态后新增 5.2 GB 用于支持更大 TP 下的 KV 缓存池，属合理分配

#### （3）预测器准确性验证（图 14）
| 配置 | 平均预测误差 |
|------|----------------|
| TP=2, DP=4 | 4.04% |
| TP=8, DP=1 | 3.07% |

> 表明基于 synthetic profiling 的预测器能准确反映真实 workload 的性能趋势。

---

## 4. 关键结论和发现

### 主要发现
1. **响应长度偏斜是同步 RLHF 中不可忽视的现象**，在 DeepScaleR 上有 ~10.85% 序列超过 8K tokens，平均每 batch 含 2.8 个极端长尾样本。
2. **静态 TP 配置无法兼顾 aligned phase 与 tail phase 的需求**：小 batch 适合高 TP，大 batch 适合低 TP。
3. **自适应 TP 切换可显著缓解长尾瓶颈**，尤其在高带宽互连（如 NVLink）下效果更佳。
4. **轻量级运行时重构机制至关重要**，否则切换开销会完全抵消性能增益。
5. **PAT 完全兼容现有 RLHF 框架**（集成于 VeRL + SGLang），无需修改训练逻辑。

### 方法的局限性
- 当前设计假设 TP 变化发生在单个节点内部（intra-node TP），DP 跨节点（inter-node DP）。若需跨节点 TP，则需额外协调机制。
- 切换策略依赖于较精确的延迟建模，对于高度非线性的 kernel behavior 或新硬件可能存在偏差。
- 对于响应长度非常均匀的任务（如短指令遵循），收益较小。

### 未来工作方向
- 扩展至支持 **Pipeline Parallelism** 的动态调整，实现 TP/PP/DP 联合优化。
- 探索 **多级切换策略**（multiple reconfigurations per generation）。
- 将 PAT 思路推广至其他存在长尾请求的场景，如 **offline RLHF evaluation** 或 **agent-based workflows**。
- 结合更细粒度的 runtime profiling 实现在线自学习预测模型。

---

> ✅ **总结一句话**：  
> PAT 通过**预测引导的自适应 Tensor Parallelism**，在不破坏同步 RLHF 语义的前提下，实现了高达 **34.6% 的生成延迟降低** 和 **27.2% 的端到端训练加速**，为解决 RLHF 中的长尾生成瓶颈提供了高效、实用的新方案。

</details>

---

### 4. [Inference Time Context Sparsity: Illusion or Opportunity?](https://arxiv.org/abs/2605.24168)

**Authors**: Sahil Joshi, Prithvi Dixit, Agniva Chowdhury, Anshumali Shrivastava, Joseph E. Gonzalez, Ion Stoica, Kumar Krishna Agrawal, Aditya Desai  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.24168v1  

#### Abstract
Sparsity has long been a central theme in LLM efficiency, but its role in context processing remains unresolved. As LLM workloads shift toward longer contexts and agentic interactions, the compute and memory bottlenecks of attention become increasingly critical, raising the question of whether these...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Inference Time Context Sparsity: Illusion or Opportunity?*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大语言模型（LLM）在处理**长上下文（long context）**时面临严重的计算与内存瓶颈，尤其是在 **attention 机制**中：
- **Prefill 阶段**：计算复杂度为 $O(N^2)$，随上下文长度 $N$ 二次增长。
- **Decoding 阶段**：KV Cache 内存占用为 $O(N \cdot d)$，成为带宽瓶颈。

传统观点认为这些是“不可避免”的限制。本文挑战这一共识，提出：**这些瓶颈是人为的、不必要的，可以通过极端但有原则的上下文稀疏化（context sparsity）来突破**。

---

### 🚀 提出的新方法与新思路

#### （1）核心主张：**Inference-Time Extreme Context Sparsity 是可行且高效的**
- 在 **推理阶段（inference time）** 强制对 attention 进行稀疏化，即每个 query 只 attend 到上下文中极小比例的 token（如 50× 或 100× 稀疏）。
- 即使模型在训练时未显式学习稀疏 attention，也能保持高质量输出。

#### （2）理论支撑：**Dense Attention 本质上是不可行的**
- 提出“**Embedding Bottleneck**”定理：当隐藏维度 $d < N$ 时，attention 输出无法区分所有可能的 attention 分布。
- 因此，**真正的 dense attention 在长上下文中本就不存在**——信息必然被压缩丢失。
- → 结论：**完全稀疏化不是近似，而是更优的设计目标**。

#### （3）系统实现：**Irregular Sparse Decode Kernels**
- 开发了支持 **per-token, per-query, per-head** 级别的细粒度稀疏 attention 内核。
- 基于 FlashInfer + paged KV Cache 构建，无需块状（block-structured）稀疏即可获得显著加速。
- 支持 GQA（Grouped Query Attention）等现代架构。

---

### 🔍 相比现有方法的优势

| 方面 | 本文方法 | 传统方法 |
|------|---------|----------|
| **稀疏模式** | 支持任意不规则稀疏（irregular sparsity），无需 block structure | 多依赖 block sparsity 才能提速 |
| **训练要求** | **无需重新训练**，直接应用于现成模型 | 多数需训练时引入稀疏（如 MoE、post-train pruning） |
| **硬件对齐** | 显示即使不规则稀疏也能在 H100 上实现高达 76× 加速 | 质疑稀疏无硬件对齐则无效 |
| **适用场景** | 覆盖 retrieval、reasoning、agentic coding 等多种复杂任务 | 多限于简单任务或合成数据 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 任务类型 | 上下文长度 | 特点 |
|-------|--------|-----------|------|
| **RULER-HARD** | 长文本检索 | 最高 32K | 六项子任务，测试模型从长文档中定位相关信息的能力 |
| **LOFT** | 检索问答（QA） | 32K / 128K | 包含 HotpotQA、MuSiQue 等多跳问答任务 |
| **AIME 2025** | 数学推理生成 | ~65K 生成长度 | 测试长程自回归生成中的错误累积影响 |
| **SWE-Bench Django** | 代理式代码修复 | >100K（动态增长） | 实际 GitHub issue + 仓库快照，模拟真实 agentic workflow |

---

### ⚙️ 实验设置与评估指标

#### 模型范围
- **5 个模型家族**：Llama3、Qwen2.5、Qwen3.5、Gemma3、Ministral3
- **参数规模跨度大**：从 0.8B 到 27B
- **架构差异**：标准 Transformer vs. Hybrid（含 SSM 或 linear attention 层）

#### 稀疏策略
- **Oracle Top-K**：精确选择 top-k 最相关 token（消除近似索引误差）
- **vAttention**：随机采样方式实现 stochastic sparsity，用于缓解注意力分散问题

#### 稀疏级别
- 5×（保留 20% tokens）、50×（2%）、100×（1%），最高达 250×

#### 评估指标
| 指标 | 含义 |
|-----|------|
| **Score / Subspan-EM** | 任务正确率或子片段匹配准确率 |
| **Retention Rate** | 稀疏 vs. 密集设置下的相对性能比（sparse / dense） |
| **Resolution Rate** | SWE-Bench 中成功修复 issue 的比例 |
| **Speedup** | 相对于 FlashInfer 的解码内核加速比 |
| **Mean Turns / Tokens** | 代理完成任务所需的步数与总 token 数 |

#### 基线对比
- **Dense Attention**：全 attention softmax（1×）
- **FlashInfer**：当前最先进的 dense & paged attention 推理引擎
- **Linear Attention / SSMs**：作为轻量级替代方案进行讨论，但非直接对比对象

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 任务 | 模型 | 稀疏程度 | 性能保留情况 |
|------|------|----------|-------------|
| RULER-HARD | Qwen3.5-27B | 100× | 几乎无损 |
| LOFT-128K | Qwen3.5-27B | 50× | ≤2% 下降 |
| AIME 2025 | Qwen3.5-27B | 50× | 仅轻微增加生成长度 |
| SWE-Bench | Qwen3.5-27B | 50× | 在有效运行中与 dense 差距 <2pp |

---

### 🔁 与基线方法的对比结果

#### （1）质量方面（vs. Dense）
- **Hybrid 模型（Qwen3.5, Gemma3）**：
  - 在 RULER 和 LOFT 上，**50× 稀疏下性能几乎持平 dense**。
  - Qwen3.5-27B 在 RULER-HARD 上 **仅用 16–32 个 token 即可达到 dense 性能**。
- **Standard 模型（Qwen2.5, Ministral3）**：
  - 小模型在 deterministic top-k 下性能下降明显（如 Qwen2.5-1.5B ↓至 63%）。
  - 但使用 **vAttention（stochastic indexing）后可恢复至接近 dense 水平**。

#### （2）效率方面（vs. FlashInfer）
- 在 H100 上测试 sparse decode kernel 性能（Table 1）：

| Batch Size | 50× Sparsity Speedup | 500× Sparsity Speedup |
|------------|------------------------|-------------------------|
| B=1        | 5.57×                  | 11.14×                  |
| B=8        | 8.88×                  | **76.14×**              |
| B=16       | 10.54×                 | **76.77×**              |

> ✅ **结论：即使是最不规则的稀疏模式，在大 batch 下仍可实现数十倍加速**

#### （3）包含 indexer 成本后的净收益（Table 2）
- 使用 DoubleSparsity 作为 indexer 模拟：
  - **MHA**：100× 稀疏下仍可达 **4.17× 净加速**
  - **GQA**：100× 稀疏下 **2.81× 净加速**
- 表明：**indexer 开销可控，整体仍为正向增益**

---

### 🔍 消融实验结果

| 维度 | 发现 |
|------|------|
| **Model Scale** | 越大的模型越鲁棒；小模型可通过 stochastic 方法弥补 |
| **Architecture** | Hybrid 架构（如 Qwen3.5）显著优于标准 Transformer |
| **Context Length** | 32K vs. 128K 表现一致，说明稀疏性泛化良好 |
| **Algorithm** | deterministic top-k 对小模型失效，但 vAttention 可解决 |
| **Task Complexity** | 从 retrieval 到 multi-hop QA、math reasoning、agentic coding 均稳健 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLM 天然具备对 inference-time context sparsity 的强鲁棒性**
   - 即使未经稀疏训练，主流 LLM 在多种任务上对 50×–100× 稀疏具有惊人容忍度。
   - 表明：**sparsity 不是妥协，而是一种被忽视的机会**。

2. **Dense Attention 是理论上的幻觉（illusion）**
   - 当 $d < N$ 时，attention 输出必然坍缩，无法保留全部 attention 分布信息。
   - → **完全稀疏化是更合理的目标**。

3. **硬件已准备好利用稀疏性**
   - 即使是非结构化、不规则稀疏，也能在现代 GPU（H100/B200）上实现高达 **76× 的 kernel 级加速**。
   - **无需 block structure 即可获益**，打破“稀疏必须规整”的迷思。

4. **Hybrid 架构 + 大模型 是稀疏友好的未来方向**
   - Qwen3.5、Gemma3 等新型混合架构表现最佳。
   - 暗示：**下一代模型应主动设计为稀疏优先（sparsity-first）**。

---

### ⚠️ 方法的局限性

1. **依赖外部 indexer（如 Oracle Top-K）**
   - 当前实验使用理想化的“oracle”选择 top-k token，实际部署需高效且准确的 indexing 机制（如 vAttention、PQCache）。
   - 如何低成本实现高质量 indexing 仍是工程挑战。

2. **Serving Stack 不稳定**
   - 在 SWE-Bench 实验中，部分失败源于 `vLLM` 服务器在稀疏 backend 下崩溃（InternalServerError / Timeout），而非模型能力问题。
   - 需要更稳定的推理框架支持。

3. **尚未探索训练时稀疏的影响**
   - 当前研究聚焦 inference-time sparsity。
   - 若在训练中引入稀疏，效果可能进一步提升，但需重新设计训练流程。

---

### 🔮 未来工作方向

1. **Develop Sparsity-Aware Training Paradigms**
   - 设计专门针对稀疏 attention 的预训练或后训练方法（如 sparse RLHF）。

2. **Build Production-Ready Sparse Inference Systems**
   - 整合高效 indexer（如 HashAttention、PQCache）与 sparse kernels，打造端到端稀疏推理栈。

3. **Hardware Co-Design for Sparse Context Processing**
   - 推动硬件厂商优化对 irregular sparse access 的支持（如 memory controller、cache hierarchy）。

4. **Extend to Other Modalities and Workloads**
   - 将 context sparsity 思想推广至多模态、语音、视频等序列建模范畴。

---

## 💡 总结一句话

> **“Dense attention is broken by design for long context.”**  
> 本文证明：**极端上下文稀疏不仅是可行的，而且是通向高效、可扩展 LLM 推理的必经之路**——它不是 illusion，而是巨大的 opportunity。

</details>

---

### 5. [NeurIPS: Neuro-anatomical Inductive Priors for Sphere-based Brain Decoding](https://arxiv.org/abs/2605.24993)

**Authors**: Sijin Yu, Zijiao Chen, Zhenyu Yang, Zihao Tan, Jiakun Xu, Zhongliang Liu, Shengxian Chen, Wenxuan Wu, Xiangmin Xu, Xin Zhang  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.24993v1  

#### Abstract
Current fMRI decoders face a performance-fidelity trade-off where efficient ID encoders outperform geometrically faithful surface-based models. We argue this is partly driven by inefficient surface tokenization and the failure to use anatomy as a predictive signal. We present NeurIPS, a framework th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NeurIPS: Neuro-anatomical Inductive Priors for Sphere-based Brain Decoding

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 fMRI 的图像解码器面临一个**性能-保真度权衡（performance-fidelity trade-off）**：
- **1D 向量编码器**（如 MindBridge）计算高效，在高语义指标上表现优异，但忽略了大脑皮层的几何拓扑结构，导致空间对齐不准确。
- **基于表面（surface-based）的方法**保留了精确的皮层几何结构，但因 token 数量庞大、训练不稳定且收敛慢（通常需 200–600 epochs），性能落后于 1D 方法。

该论文认为这一权衡并非本质限制，而是由两个架构缺陷造成：
1. **低效的表面 tokenization**：在全脑半球进行球面卷积，产生大量无关 token。
2. **将个体解剖差异视为噪声**：主流方法通过 subject ID 进行条件建模，容易导致模型记忆个体模式而非学习泛化规则。

---

### 提出的新方法与核心创新

NeurIPS 提出一种新的框架，将神经解剖变异从“干扰项”转变为强大的**归纳先验（inductive prior）**，实现高效且高保真的跨被试解码。

#### 创新点一：Selective ROI Spherical Tokenizer (SRST)
- **任务对齐的 tokenization**：仅在预定义的视觉 ROI（Region of Interest）内执行球面卷积，显著减少 token 数量。
- 在 `fsaverage6` 模板上，视觉 ROI 包含约 **9,488 个顶点**（vs 全脑 81,924），token 数减少 **88.4%**。
- 输出两种 token：
  - **Local tokens**：保留精细几何细节；
  - **Global token**：提供整体场景上下文。
- 优势：降低内存与计算开销，提升训练稳定性，同时保持拓扑保真度。

#### 创新点二：Structure-Guided Mixture of Experts (SG-MoE)
- 替代传统的 subject ID 条件路由机制。
- 使用个体的**皮层结构特征**作为 MoE 路由信号：
  - Cortical thickness
  - Curvature
  - Sulcal depth
  - Surface area
- 路由器仅接收结构特征输入，**不接触 subject ID 或功能活动**，迫使模型学习“结构→功能”的泛化映射。
- 设计为稀疏激活 MoE（top-k=6, N=16 experts），保证效率。

---

### 相比现有方法的优势
| 维度 | NeurIPS | 传统方法 |
|------|--------|---------|
| **几何保真度** | ✅ 显式建模皮层拓扑 | ❌ 扁平化处理丢失结构信息 |
| **计算效率** | ✅ 仅处理视觉 ROI，token 减少 88.4% | ❌ 全脑处理负担重 |
| **跨被试泛化** | ✅ 基于解剖结构路由，支持可扩展泛化 | ❌ 依赖 subject ID，易过拟合 |
| **收敛速度** | ✅ 10 epochs 内完成微调 | ❌ 需要数百 epochs 收敛 |
| **新被试适应能力** | ✅ 用 20% 数据即可快速个性化 | ❌ 需大量数据与长时间训练 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Natural Scenes Dataset (NSD)** (Allen et al., 2022)
  - 包含 8 名被试（subj01–08）
  - 每人观看 10,000 张 COCO 图像，重复 3 次
  - 共享测试集：1,000 张所有被试共同观看的图像
  - 训练/验证划分：8,500 / 500
  - fMRI 输入为 GLM 估计的 beta weights

### 实验设置
- **多被试联合训练**：初始在 3–4 名被试上预训练
- **新被试微调**：冻结大部分参数，在新被试（held-out subject）上进行少量 epoch 微调（1–10 epochs）
- **数据比例**：使用 5%–100% 的目标被试数据进行微调
- **硬件**：单张 NVIDIA A800 GPU
- **训练时间**：约 138 秒/epoch；推理约 3.4 秒/样本

### 评估指标
共 8 项标准指标，分为两类：

#### 低级视觉保真度（Low-Level Fidelity）
- **PixCor**：像素相关性
- **SSIM**：结构相似性

#### 高级语义相似性（High-Level Semantics）
- **Alex(2), Alex(5)**：AlexNet 第 2 和第 5 层激活匹配度
- **Incep**：InceptionV3 特征匹配
- **CLIP**：CLIP ViT-L/14 图像嵌入匹配
- **Eff**：EfficientNet-B1 平均距离
- **SwAV**：SwAV-ResNet50 距离

最终图像重建使用 **Versatile Diffusion (VD)** 模型，统一生成流程以消除解码后端偏差。

### 基线方法对比
所有基线均使用相同 VD 后端与超参，确保公平比较：
- **1D-vector 方法**：
  - Mind-Vis, Takagi & Nishimoto (2023), MindEye, MindBridge, UMBRAE, NeuroPictor
- **Sphere-based 方法**：
  - Gu et al. (2023)
  - Yu et al. (2025)
  - SIM (Dahan et al., 2025)

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）
在 NSD 多被试设定下，NeurIPS 取得 **sphere-based 方法中的 SOTA 表现**，并接近最强 1D 方法：

| 方法 | PixCor | SSIM | Alex(5) | CLIP | Eff ↓ | SwAV ↓ |
|------|-------|------|--------|------|--------|--------|
| **NeurIPS (Ours)** | **0.248** | **0.370** | **95.2%** | **93.2%** | **0.663** | **0.404** |
| SIM (2025) | 0.119 | 0.260 | 90.4% | 89.4% | 0.733 | 0.448 |
| Yu et al. (2025) | 0.165 | 0.305 | 89.0% | 88.3% | 0.733 | 0.398 |
| MindBridge (2024) | 0.151 | 0.263 | 95.5% | **94.7%** | 0.712 | 0.418 |

> 💡 **结论**：NeurIPS 在保留皮层几何的前提下，实现了与顶级 1D 方法相当的语义解码能力。

---

### 新被试快速适应能力（Figure 4 & 5）
- **仅用 20% 数据 + 1 epoch 微调**，NeurIPS 即可生成语义连贯的图像。
- **10 epochs 内达到 90% 以上最终性能**，远快于 baseline（需 200–600 epochs）。
- 定量显示：
  - 使用 **5% 数据 + 10 epochs**：仍可达 **86.0% Alex(5)** 和 **80.0% CLIP**（Table 8）

---

### 可扩展性测试（Figure 5 右图）
当训练队列从 4 名被试扩展到 8 名时：
- **SIM 的 CLIP 分数下降 2.0 pts**
- **NeurIPS 仅下降 0.6 pts**

> 表明 NeurIPS 对群体异质性更具鲁棒性，适合真实世界部署。

---

### 消融实验结果（Table 2）

| 设置 | CLIP | Alex(5) | SSIM | 说明 |
|------|------|--------|------|------|
| Full Model | **93.2%** | **95.2%** | **0.370** | — |
| w/o global token | 89.4% | 92.7% | 0.370 | 缺少全局上下文损害语义 |
| subject ID gating | 92.7% | 94.9% | 0.370 | 性能略降，表明 anatomy 更优 |
| full brain | 91.0% | 91.7% | 0.316 | 全脑处理反而更差 |
| no anatomy | 90.2% | 93.4% | 0.361 | 解剖信息确实带来增益 |
| random anatomy | 92.0% | 94.4% | 0.369 | 正确 anatomy 最佳 |
| anatomical swap | 91.9% | 94.4% | 0.366 | 结构一致性重要 |

> ✅ **结论**：性能提升源于 anatomy-conditioned routing，而非单纯增加参数或记忆 ID。

---

## 4. 关键结论和发现

### 主要发现
1. **解剖结构是强大的归纳先验**：利用 cortical thickness、curvature 等作为 MoE 路由信号，可有效建模个体差异，促进泛化。
2. **ROI 选择至关重要**：限制在视觉 ROI 内建模不仅大幅提效，还能提升性能，证明“less is more”。
3. **快速个性化成为可能**：借助 anatomy-guided 架构，新被试只需极少量数据（20%）和极少训练（1–10 epochs）即可完成适配。
4. **打破性能-保真度权衡**：NeurIPS 在保持皮层几何完整性的同时，达到了与 1D 方法相媲美的语义解码性能。
5. **模型行为具有神经科学合理性**：
   - SG-MoE 路由高度依赖脑区位置，而非 subject ID（Figure 6A）
   - 解码性能沿视觉通路梯度上升（V1 → ventral stream）（Figure 6D）
   - 贡献热图集中在视觉皮层（Figure 6E）

---

### 方法的局限性
- **适用范围受限于 ROI 定义**：目前专注于静态视觉任务，若用于多模态或全脑任务（如语言、运动），需扩展至更大网络。
- **依赖 FreeSurfer 注册质量**：表面配准误差会影响 SRST 与 SG-MoE 效果。
- **未完全探索感知路径的几何建模**：当前感知分支仍采用 1D MLP，未来可尝试将其也迁移到球面域。

---

### 未来工作方向
- 将 SG-MoE 扩展至其他模态（如 EEG/fNIRS）或多任务解码。
- 探索动态 ROI 选择机制，适应不同认知范式。
- 发展无需高精度表面重建的轻量化版本，便于临床应用。
- 结合发育或病理数据，研究结构-功能映射的变化规律。

---

> 📌 **总结一句话**：  
> **NeurIPS 通过引入 Selective ROI Spherical Tokenizer 和 Structure-Guided MoE，首次实现了高效、高保真、可扩展的 sphere-based 脑解码框架，证明了解剖结构不仅是噪声，更是通往通用脑解码的关键信号。**

</details>

---

### 6. [A Tabular Schedule Abstraction for Communication-Aware Evaluation of Pipeline-Parallel LLM Training](https://arxiv.org/abs/2605.24006)

**Authors**: Daniel Barley, Jonathan Leis, Benjamin Klenk, Holger Fr\"oning  
**Category**: cs.DC  
**Published**: 2026-05-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.24006v1  

#### Abstract
Pipeline parallelism is a key technique for distributed training of large language models because it reduces per-device parameter and activation memory. However, comparing pipeline schedules is difficult: analytical models expose structural quantities such as bubble ratios, while end-to-end hardware...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Tabular Schedule Abstraction for Communication-Aware Evaluation of Pipeline-Parallel LLM Training

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前对 **pipeline-parallel LLM training** 中的调度策略（scheduling）缺乏系统、可比较的评估框架。传统方法存在两个极端：
- **分析模型（analytical models）** 虽然简洁，但忽略通信开销、依赖关系和计算-通信重叠等现实因素；
- **端到端硬件实验** 成本高、难以复现且受限于特定系统配置。

这导致基于“bubble ratio”等结构性指标得出的调度优劣结论，在实际执行中可能不成立。

### 🚀 提出的新方法与创新
本文提出了一种 **统一的多抽象层次评估框架（unified multi-abstraction methodology）**，其核心是引入一个 **表格化调度抽象（tabular schedule abstraction）**，并构建在 Graphculon 模拟器之上。

#### 主要创新点包括：
1. **Tabular Schedule Abstraction（表格化调度表示）**
   - 将 pipeline schedule 表示为 `worker × time slot` 的二维表，每个单元格指定某 worker 在某个时隙执行的 phase（如 fwd, bwd）或为空闲。
   - 该表示解耦了 **调度策略设计** 与 **执行成本建模**，提升了表达性和可比性。

2. **三层次评估体系**
   - **公式级（Formula-based reasoning）**：使用闭式表达式估算 bubble ratio、utilization 等结构性指标。
   - **理想化调度表（Idealized schedule tables）**：显式展示 microbatch 流动、填充/排空行为、空闲槽位等结构特征。
   - **通信感知模拟（Communication-aware execution simulation）**：将调度表转换为带依赖关系的 execution graph，并结合 Hockney 和 Roofline 模型进行细粒度仿真。

3. **支持跨系统配置的可控比较**
   - 可灵活调整 compute、network bandwidth/latency 参数，研究不同 regime 下调度表现的变化。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 抽象程度 | 单一层级（要么纯公式，要么实测） | 多层级联动分析 |
| 可解释性 | 公式易懂但脱离现实；实测真实但难归因 | 结构可见 + 影响可量化 |
| 泛化能力 | 实验绑定具体硬件 | 支持参数化系统建模 |
| 开发效率 | 新调度需重写逻辑 | 新调度只需填写 schedule table |

---

## 2. 核心实验方法和设置

### 🧪 实验模型与任务
- 使用 **Megatron-style Transformer 模型** 进行模拟：
  - 128 层 Transformer blocks
  - 隐藏维度 $d_{\text{hidden}} = 4096$
  - 注意力头数 $n_{\text{heads}} = 80$
  - 序列长度 $s = 4096$
  - 非线性激活函数：GELU
- 固定全局 minibatch size，通过改变 microbatch 数量 $B$ 控制每阶段计算负载。
- 仅考虑 **pipeline parallelism**，其他形式（data/tensor/expert parallelism）设为 1。

### ⚙️ 模拟平台与系统建模
基于 **Graphculon**（源自 Calculon 的图基建模框架），构建 execution graph 并生成 timeline trace。

#### 系统参数建模：
| 组件 | 模型 | 公式 |
|------|------|-----|
| **通信时间** | Hockney 模型 | $t_{\text{comm}} = \frac{V_{\text{net}}}{\text{BW}_{\text{net}}} + L_{\text{net}}$ |
| **计算时间** | Roofline 模型 | $t_{\text{comp}} = \max\left(\frac{F}{TP \cdot e_c + L_c}, \frac{V_m}{\text{BW}_m \cdot e_m} + L_m\right)$ |

其中 $e_c$, $e_m$ 是经验效率因子。

#### 系统配置网格（3×3）
| Compute / Network | Slow | Medium (Baseline) | Fast |
|------------------|------|-------------------|------|
| **Slow Compute** | slow_cp_slow_nw | ... | slow_cp_fast_nw |
| **Medium (Baseline)** | mid_cp_slow_nw | **DGX H100 类似** | mid_cp_fast_nw |
| **Fast Compute** | fast_cp_slow_nw | ... | fast_cp_fast_nw |

> 示例：baseline 配置 ≈ NVIDIA DGX H100：~1 PFLOPs compute, 34 TB/s memory BW, 50 ns latency, 50 GB/s InfiniBand, 500 ns latency。

### 📊 评估指标
分为两类：

#### （1）运行时与利用率相关
- **Bubble Ratio**：空闲时间占比（公式/表层）
- **Schedule Length (slots)**：总调度槽数
- **Simulated Execution Time ($T_{\text{sim}}$)**：模拟训练一步的时间
- **Idle Time Ratio ($p_{\text{idle}}$)**：模拟中设备空闲比例

#### （2）内存压力相关
- **Peak Activation Memory**：瓶颈 worker 上最大激活值内存占用
- **Persistent Memory**：参数、梯度、optimizer state 等常驻内存
- **Overall Per-device Memory Footprint**

### 🔄 对比的调度算法（Schedules under Study）
| 名称 | 特点 |
|------|------|
| **GPipe** | 填充-排空模式，forward 全部完成后才开始 backward，activation retention 高 |
| **1F1B** | 单向交替执行 fwd/bwd，降低 activation lifetime |
| **Chimera** | 双向流水线，减少 bubble，但需参数复制、通信更多 |
| **Hanayo** | 波浪式同步调度，目标提升利用率，适用于 $S=B$ 场景 |

> Hanayo 仅在 $(S,B)=(8,8)$ 的受限场景下测试。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据与对比结果

#### （1）结构 vs. 实际性能：排名不一致（Abstraction-Invariance Failure）
- **公式与表格层** 显示 Chimera 的 bubble ratio 始终优于 GPipe 和 1F1B。
- **但在通信感知模拟中**，这一优势被逆转：
  - 在 **网络受限系统（slow_nw）** 上，Chimera 因额外通信开销反而更慢；
  - 在 **计算受限系统（fast_nw）** 上，Chimera 才表现出优势（尤其当 $B < 64$）；
  - 当 $B > 64$ 后，各调度趋于收敛。

> 👉 **结论：仅靠 bubble ratio 推断调度优劣不可靠。**

#### （2）GPipe vs. 1F1B
- **运行时间等效**：在所有系统配置下，两者 $T_{\text{sim}}$ 几乎相同。
- **内存差异显著**：
  - GPipe 需保留整个 minibatch 的 activations → peak memory 高；
  - 1F1B 缩短 activation-retention interval → **activation memory peak 更低**。
- ✅ **1F1B 是更强的单向 baseline**。

#### （3）Chimera 性能表现
- 在 **低 microbatch 数（$B \leq 16$）且高速网络** 下表现最佳：
  - 最多比 GPipe 快 ~15%
  - idle time 显著更低
- 但在 **高 microbatch 或慢网络** 下劣势明显：
  - 更多通信导致瓶颈
  - 参数复制增加 persistent memory 开销

#### （4）Hanayo 在其目标场景 $(S,B)=(8,8)$ 的表现
| 系统条件 | 相对于 Chimera 的加速 | Idle Ratio 下降 |
|--------|------------------|-------------|
| fast_nw / mid_nw | ~12–14% 加速 | 从 ~35% → ~25% |
| slow_nw_fast_cp | **慢 12.32%** | idle 更高 |

> 👉 Hanayo 在通信良好时有效，但在通信受限时反不如 Chimera。

#### （5）消融实验：非对称 Chimera 放置（Asymmetric Placement）
- **动机**：尝试缓解首 stage 内存压力，采用 1:2 不均衡块分配（如前半管道少分 block，后半多分）。
- **结果**：
  - ❌ **未降低全局 peak memory**（仍由最差 worker 决定）
  - ✅ 在浅层 pipeline（$S=4$）+ 快速网络下有 **约 5% runtime 提升**
  - ❗ 深层 pipeline（$S=8$）在小 microbatch 下甚至变慢

> 👉 表明直觉驱动的设计不一定奏效，必须依赖显式建模验证。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **调度排名不具备抽象不变性（Schedule Rankings Are Not Abstraction-Invariant）**
   - 分析模型中的“最优”调度，在加入通信与依赖后可能不再是最佳。
   - **bubble ratio ≠ 实际性能**，必须结合通信建模。

2. **没有通用最优调度（No Universally Superior Schedule）**
   - 调度质量高度依赖于系统环境（compute/network balance）、超参（microbatch count）、模型规模等。
   - 例如：
     - 1F1B 在内存上优于 GPipe；
     - Chimera 仅在通信友好+小 batch 下占优；
     - Hanayo 敏感于网络瓶颈。

3. **1F1B 是更优的单向 baseline**
   - 与 GPipe 运行时间相当，但 activation memory 显著更低，更适合资源受限场景。

4. **双向调度（如 Chimera）收益有限**
   - 虽能减少结构性 bubble，但代价是更高通信与参数冗余。
   - 实际增益取决于能否有效隐藏通信延迟。

5. **非对称调度修改未必改善瓶颈**
   - 即使逻辑上“平衡”了内存分布，也可能无法降低全局峰值，甚至影响 runtime。
   - 强调了 **bottleneck-aware evaluation** 的必要性。

### ⚠️ 方法的局限性
- **非硬件校准模型**：Graphculon 是分析模型，虽趋势可信，但绝对数值不能直接用于部署预测。
- **仅限同步调度**：未涵盖异步或近零气泡（zero-bubble）调度（如 Interleaving, Bubble-elimination）。
- **静态假设**：未考虑动态负载变化、故障恢复、压缩技术等复杂因素。
- **简化 memory model**：未深入建模 swap、paging 或 activation compression 的影响。

### 🔮 未来工作方向
1. 扩展至 **communication-aware energy modeling**，评估能耗效率。
2. 在更多硬件平台上 **校准计算模型**（如不同 GPU 架构）。
3. 支持更复杂的调度类别：
   - Interleaved pipeline
   - Zero-bubble 或 near-zero-bubble scheduling
4. 引入 **activation compression** 技术的联合建模，权衡精度与效率。
5. 探索 **自动调度搜索空间**，利用此框架指导 schedule design automation。

---

> 💡 **一句话总结**：  
> 本文揭示了 pipeline schedule 评估不能只看“纸面指标”，而应建立“结构-通信-系统”三位一体的多抽象层次分析框架——**好调度，是在正确环境下运行得好的那个**。

</details>

---

### 7. [Visual-Redundancy-Controlled Parallel Decoding for Diffusion-Based Multimodal Large Language Models](https://arxiv.org/abs/2605.25820)

**Authors**: Yulin Yuan, Hongshuo Zhao, Xiangming Meng  
**Category**: cs.LG  
**Published**: 2026-05-26  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.25820v1  

#### Abstract
Diffusion-based multimodal large language models (dMLLMs) decode by iteratively predicting tokens at multiple masked positions in parallel. This turns each decoding step into a position-selection problem: the model must choose not only which predictions are reliable in isolation, but also which posi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Visual-Redundancy-Controlled Parallel Decoding for Diffusion-Based Multimodal Large Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **diffusion-based multimodal large language models (dMLLMs)** 中，现有的并行解码策略（如基于置信度的 `confidence-based decoding`）通常独立地选择高置信度的 token 进行解码。然而，这些高置信度 token 可能依赖于**重叠的视觉区域（overlapping visual grounding）**，导致“**视觉冗余（visual redundancy）**”——即多个被同时解码的 token 引入的是相似而非互补的视觉信息。

这种冗余会限制后续 token 的生成质量，因为模型缺乏多样化的视觉上下文线索。

### ✅ 提出的新方法与新思路
作者提出了以下三个核心贡献：

#### （1）提出 **Visual Redundancy Index (VRI)**  
- 一种量化指标，用于衡量在同一步中被解码的多个 token 所依赖的视觉注意力是否重叠。
- VRI 越低，表示这些 token 的视觉 grounding 更加互补；反之则存在冗余。

#### （2）提出 **Visual-Redundancy-Controlled Decoding (VRCD)**  
- 一种无需训练的（training-free）、推理阶段可用的轻量级重排序方法。
- 利用 **token-to-image attention** 来估计 token 之间的视觉重叠程度，并据此对高置信度候选 token 进行重新打分。
- 在保持置信度作为基础可靠性信号的同时，降低具有高度视觉重叠的 token 被同时选中的概率。

#### （3）强调“**互补视觉上下文**”的重要性  
- 将解码决策从“单个 token 是否可靠”提升到“一组 token 是否提供互补的视觉信息”，为多模态并行解码提供了新的视角。

### ✅ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **有效性** | 显著提升多步推理任务（如 M3CoT）上的准确率，最高达 **18.8%** 的相对增益。 |
| **效率** | 仅引入约 **1.5% 的运行时开销**，几乎不影响吞吐量（throughput）。 |
| **通用性** | 无需微调或修改模型结构，可直接应用于现有 dMLLMs，是一种即插即用的 inference-time 方法。 |
| **可解释性** | 提供 VRI 指标，可用于分析不同解码策略下的视觉冗余水平。 |

---

## 2. 核心实验方法和设置

### ✅ 数据集
实验覆盖五个主流多模态基准，涵盖多种能力：
| 数据集 | 任务类型 | 主要评估指标 |
|--------|---------|-------------|
| **M3CoT** | 多领域、多步骤多模态推理（multi-step multimodal reasoning） | 准确率（Accuracy） |
| **MMBench** | 通用视觉语言理解（perception & reasoning） | 准确率（Accuracy） |
| **SQA-IMG** | 基于图表的科学问答（science QA with diagrams） | 准确率（Accuracy） |
| **DocVQA** | 文档图像问答（text-rich images） | ANLS |
| **InfoVQA** | 信息图问答（infographics with layout & visuals） | ANLS |

> 所有数据集均使用官方 prompt 格式和评分脚本。

---

### ✅ 实验设置

#### 模型主干（Backbone）
- 默认使用 **lavida-llada-reason**（基于 LLaDA-8B 的多模态扩散语言模型）
- 视觉编码器：**SigLIP-400M**，输出 980 个 visual tokens

#### 解码长度与 Forward Ratio (FR)
- 解码长度 $ L \in \{192, 384\} $
- Forward Ratio $ FR \in \{0.125, 0.25, 0.5\} $，对应每步提交 token 数量分别为 8、4、2
- 通过控制 FR 控制并行度，测试高并行场景下方法的有效性

#### 评估指标
| 指标 | 描述 |
|------|------|
| **任务性能得分** | 各数据集的标准评价分数（如 Accuracy 或 ANLS） |
| **VRI (Visual Redundancy Index)** | 衡量同一步中已提交 token 的视觉注意力重叠程度 |
| **Remaining-position entropy** | 衡量剩余 masked token 的预测不确定性，越低越好 |
| **Throughput (tokens/s)** | 推理速度，衡量实际部署可行性 |

---

### ✅ 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **Confidence** | 基线 | 选择置信度最高的 K 个位置进行解码 |
| **Margin** | 不确定性准则 | 选择最大与次大预测概率差值最大的位置 |
| **Entropy** | 不确定性准则 | 选择预测熵最小的位置 |
| **IG (InfoGain)** | 信息增益 | 使用信息增益指导位置选择 |
| **VRCD (Ours)** | 提出方法 | 基于视觉冗余控制的置信度重加权 |

> 所有方法共享相同的模型权重、prompt、解码调度和评估流程。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（L=192）

| 方法 / 数据集 | M3CoT ↑ | MMBench ↑ | SQA-IMG ↑ | DocVQA ↑ | InfoVQA ↑ |
|-------------|--------|----------|-----------|----------|-----------|
| Confidence | 37.68 | 51.14 | 57.11 | 50.04 | 32.93 |
| Margin     | 36.83 | 51.21 | 54.37 | 46.85 | 32.70 |
| Entropy    | 34.86 | 48.24 | 49.95 | 46.72 | 32.04 |
| IG         | 37.39 | 51.16 | 50.67 | 46.81 | 33.26 |
| **VRCD (Ours)** | **40.38** | **59.37** | **58.64** | **52.30** | **33.06** |

> ✅ 在 M3CoT 和 MMBench 上分别取得 **+2.70** 和 **+8.23** 的绝对提升。

---

### ✅ 长序列解码表现（L=384）

| 方法 / 数据集 | M3CoT ↑ | MMBench ↑ |
|-------------|--------|----------|
| Confidence | 38.35 | 63.04 |
| **VRCD (Ours)** | **41.99** (+9.5%) | **66.73** (+5.8%) |

> ✅ 在更长的生成任务中，VRCD 的优势进一步放大，说明其在复杂推理链中能持续积累高质量上下文。

---

### ✅ 性能增益汇总
- 在 **M3CoT** 上实现高达 **18.8% 的相对准确率提升**
- 在 **MMBench** 上实现 **6.9% 的相对提升**
- 改进效果在需要多步推理的任务上尤为显著（M3CoT、MMBench），而在短答案任务（DocVQA）上增益较小

---

### ✅ 消融实验结果

#### （1）参数 $\alpha$ 影响（控制冗余惩罚强度）
| $\alpha$ | M3CoT | MMBench |
|--------|-------|--------|
| 0.0 (Confidence) | 28.82 | 57.29 |
| 1.5 | **34.24** | 61.24 |
| 2.0 | 33.45 | **62.69** |

> ✅ $\alpha=1.5$ 在多数情况下最优，表明适度的冗余抑制效果最好。

#### （2）冗余分数聚合方式
| 聚合方式 | M3CoT (L=384) | MMBench (L=384) |
|--------|---------------|----------------|
| Average | 32.17 | 60.63 |
| **Weighted (w/ confidence)** | **34.24** | **61.24** |

> ✅ 使用置信度加权的聚合方式更优，说明应优先考虑高置信邻居的影响。

#### （3）视觉显著性提取（VSE）模块作用
| 方法 | M3CoT | MMBench |
|------|--------|--------|
| w/o VSE | 34.13 | 61.00 |
| **with VSE (full VRCD)** | **34.24** | **61.24** |

> ✅ 移除 VSE 导致性能下降，验证了去除均匀注意力噪声的重要性。

---

### ✅ 效率与吞吐量（Throughput）

| 方法 | tokens/s |
|------|----------|
| LaViDa (baseline) | 46.982 |
| LaViDa + IG | 24.884 |
| **LaViDa + VRCD** | **46.264** |

> ✅ VRCD 仅带来 **~1.5% 的额外开销**，远低于 IG 等复杂方法，具备实用价值。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **视觉冗余是影响 dMLLM 并行解码质量的重要因素**  
   即使单个 token 置信度很高，若多个 token 共享相同视觉区域，则无法为后续生成提供多样化上下文。

2. **VRCD 能有效减少 VRI，提高后续预测的确定性**  
   - 实验显示 VRCD 显著降低了各步的 **VRI 值**
   - 同时降低了剩余 token 的 **预测熵（entropy）**，说明上下文更清晰

3. **轻量设计即可获得显著收益**  
   VRCD 仅通过 attention 后处理进行重排序，不增加前向传播次数，却能在多个任务上取得一致提升。

4. **改进集中在早期解码步骤**  
   早期提交更具视觉多样性的 token，有助于引导整个生成过程走向正确路径。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **不改变解码调度本身** | VRCD 仅在给定 schedule 下重排序，不能动态调整每步提交多少 token |
| **依赖 token-to-image attention 质量** | 若 attention 分布不准确（如格式符号 attention 均匀），可能误判冗余 |
| **对短任务增益有限** | 如 DocVQA 等短答案任务，因 masked positions 较少，改进空间小 |
| **未探索联合子集搜索** | 完全最优的子集选择计算代价高，目前采用近似 reranking 策略 |

---

### 🔮 未来工作方向
1. **自适应 commit size selection**  
   动态决定每步应解码多少 token，结合冗余感知与任务复杂度。

2. **跨步冗余建模**  
   当前 VRI 仅关注单步内冗余，未来可研究跨时间步的视觉信息重复问题。

3. **与其他解码目标融合**  
   将 VRI 与信息增益（InfoGain）、不确定性等目标联合优化。

4. **扩展至更高分辨率或多视图输入**  
   探索在 PyramidDrop、Sparse-LaViDa 等压缩框架下如何协同利用视觉冗余信号。

---

## 总结

📌 **一句话总结**：  
本文揭示了 dMLLM 并行解码中存在的“**视觉冗余**”现象，提出了 **VRCD** 方法，通过利用 **token-to-image attention** 实现对高置信但冗余 token 的抑制，在几乎无额外成本的前提下显著提升了多模态推理性能。

🎯 **核心价值**：  
将多模态解码的关注点从“**个体可靠性**”转向“**群体互补性**”，为 diffusion-based MLLMs 提供了一种高效、实用且理论清晰的改进路径。

</details>

---

### 8. [Cross-Platform Fused MoE Dispatch in Triton: Portable Expert Routing Without CUDA](https://arxiv.org/abs/2605.23911)

**Authors**: Subhadip Mitra  
**Category**: cs.DC  
**Published**: 2026-05-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.23911v1  

#### Abstract
Mixture-of-Experts (MoE) architectures power the majority of frontier large language models, but their inference is bottlenecked by irregular memory access patterns and expert routing overhead. Existing optimized MoE kernels (Megablocks, Tutel, FasterMoE) are implemented in CUDA and locked to NVIDIA...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Cross-Platform Fused MoE Dispatch in Triton: Portable Expert Routing Without CUDA*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Mixture-of-Experts (MoE) 架构在前沿大语言模型中广泛应用，但其推理过程面临两大挑战：
- **不规则内存访问模式**：token 需要动态路由到不同专家，导致内存访问非连续。
- **专家调度开销高**：传统实现需要为每个专家启动多个 GEMM 内核（如 gate、up、down 投影），造成大量 kernel launch 开销。
- **硬件锁定**：现有优化方案（如 Megablocks、Tutel、FasterMoE）均基于 CUDA 实现，仅限于 NVIDIA GPU，无法在 AMD 等平台运行。

### 🚀 提出的新方法：TRITONMOE
作者提出 **TRITONMOE**，一个完全用 OpenAI Triton 编写的融合 MoE 调度内核，实现了完整的前向传播流程（从路由打分到输出组合），且**不依赖任何 CUDA 代码**。

#### 核心创新点：
1. **Block-Scheduled Grouped GEMM**
   - 将 Triton 的 program block 映射到 `(expert_id, token_offset)` 对上，实现单次 kernel launch 处理所有专家的变长 batch GEMM。
   - 避免 padding 浪费，支持高效的小批量专家计算。

2. **Fused Gate+Up Projection Kernel**
   - 在一个 kernel 中同时完成 SwiGLU 的 gate 和 up 投影。
   - 利用共享的 L2 缓存输入 tile，并在寄存器中执行 SiLU 激活函数。
   - **减少 35% 的全局内存流量**，显著提升带宽效率。

3. **端到端五阶段流水线设计**
   - Router → Permute → Fused Gate+Up → Down GEMM → Unpermute
   - 将原本 $3E + 4$ 次 kernel launch 减少为固定 5 次，与专家数量无关。

### 🔍 相比现有方法的优势
| 维度 | TRITONMOE | Megablocks / Tutel / FasterMoE |
|------|-----------|-------------------------------|
| **跨平台性** | ✅ 支持 NVIDIA 和 AMD（MI300X） | ❌ 仅支持 NVIDIA（CUDA-only） |
| **内存效率** | ✅ Fused gate+up 减少 35% 内存流量 | ⚠️ 分离式 GEMM 导致重复读写 |
| **调度开销** | ✅ 固定 5 次 kernel launch | ⚠️ $O(E)$ 次 launch，$E$ 大时开销剧增 |
| **部署灵活性** | ✅ 纯 Triton 实现，无需编译扩展 | ⚠️ 依赖定制 CUDA kernel 或特殊库 |

---

## 2. 核心实验方法和设置

### 🧪 实验配置

#### 硬件平台
- **NVIDIA A100-SXM4-80GB**（主测试平台）
- **AMD Instinct MI300X**（用于验证跨平台正确性）

#### 软件环境
- PyTorch 2.4.1
- Triton 3.0.0
- CUDA 12.4（NVIDIA）、ROCm 6.1（AMD）

#### 模型配置（覆盖主流 MoE 架构）
| Model | E (Experts) | k (Top-k) | d (Hidden Dim) | dfn (FFN Dim) |
|-------|-------------|----------|----------------|---------------|
| Mixtral-8x7B | 8 | 2 | 4096 | 14336 |
| Mixtral-8x22B | 8 | 2 | 6144 | 16384 |
| DeepSeek-V3 | 256 | 8 | 7168 | 2048 |
| Qwen2-MoE-57B | 64 | 4 | 3584 | 2560 |

### 📊 评估指标
- **End-to-end latency (ms)**：完整 MoE 层前向延迟
- **Throughput (tokens/s)**：每秒处理 token 数
- **Memory traffic reduction**：通过理论分析估算
- **Arithmetic Intensity (FLOPs/Byte)**：roofline 分析
- **Cross-platform correctness**：是否能在 AMD 上无错误运行

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **PyTorch Reference** | 循环调用每个专家的 cuBLAS GEMM，共 $3E$ 次 launch |
| **Megablocks** | 当前最先进的 CUDA 优化 block-sparse MoE 实现（dMoE 变体） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（A100 平台）

#### 表：Mixtral-8x7B 和 Qwen2-MoE-57B 的端到端延迟（单位：ms）

| Tokens | PyTorch | Megablocks | **Ours (TRITONMOE)** |
|--------|---------|------------|------------------------|
| 32     | 10.44   | 2.78       | **2.13**               |
| 128    | 13.14   | 2.77       | **2.27**               |
| 512    | 25.92   | 3.57       | 3.99                   |
| 2048   | 66.22   | 9.08       | 16.48                  |

> ✅ 在推理典型小批量（≤128 tokens）下，TRITONMOE **快于 Megablocks**，最高达 **1.22× 加速**。

#### 其他模型表现
- **Mixtral-8x22B**：在 512 tokens 下达到 Megablocks 的 89%，且随负载增大差距缩小。
- **DeepSeek-V3 (256 experts)**：
  - 不适用 Megablocks 标准配置（不支持 top-8 + 256E）
  - TRITONMOE 相比 unfused 实现有 **16–27% 提升**

### 🔬 消融实验（Ablation Study）——Mixtral-8x7B @ 512 tokens

| 配置 | Latency (ms) | Speedup |
|------|--------------|--------|
| (a) PyTorch reference (24 cuBLAS launches) | 55.18 | 1.0× |
| (b) Triton unfused (3 grouped GEMMs) | 3.59 | 15.4× |
| (c) Triton fused gate+up | **3.11** | **17.7×** |

> 💡 结论：
> - **Grouped GEMM 是最大加速来源（15.4×）**
> - **Gate+Up 融合额外带来 1.15× 提升**

### 📊 性能瓶颈分析（Roofline & Profiling）

| 阶段 | 占比（Mixtral） | 特性 |
|------|------------------|------|
| Expert FFN | >95% | Compute-bound（高算术强度） |
| Permute / Unpermute | <3% | Memory-bound，但耗时极低 |
| Router | ~0.07ms | 可忽略 |

- **Fused Gate+Up Kernel 达到 42.5% 峰值带宽利用率，34.6% 峰值算力利用率**，表明资源利用高效。

### 🌐 跨平台验证结果
- **全部 162 项正确性测试**在 **AMD MI300X** 上通过
- **零代码修改**即可运行
- 使用相同 `.py` Triton kernel 文件，由 Triton 自动编译至 AMD GCN/CDNA
- ✅ 验证了真正的 **cross-platform portability**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Triton 可以实现接近甚至超越 CUDA 的 MoE 推理性能**
   - 在 **小批量（≤128 tokens）场景下，TRITONMOE 快于 Megablocks**，归因于更低的 kernel launch 开销。
   - 在 **大批量（≥2048 tokens）下，Megablocks 更优**，因其 hand-tuned CUDA kernel 更好地压榨 Tensor Core。

2. **跨平台可移植性已可行**
   - 仅使用标准 Triton primitives（无 inline CUDA、无 vendor intrinsic）即可实现 **NVIDIA + AMD 双平台正确运行**。
   - 为未来异构 AI 部署提供坚实基础。

3. **Fusion 是关键优化手段**
   - **Fused gate+up kernel 减少 35% 内存流量**，对带宽敏感场景至关重要。
   - Grouped GEMM 设计将 kernel launch 数从 $O(E)$ 降至常数 5。

4. **Routing 不均衡性影响调度效率**
   - 在 **Qwen2-MoE (64 experts, top-4)** 上，当路由呈 Zipf 分布（α=2.0）时：
     - TRITONMOE 性能稳定（~3.17ms）
     - Megablocks 加速至 2.22ms（因 block-sparse layout 合并热点专家）
     - 导致 TRITONMOE 相对性能下降（speedup 从 1.03× → 0.70×）

---

### ⚠️ 方法的局限性

| 限制 | 说明 |
|------|------|
| **Fixed BLOCK_M Tile Size** | 编译时决定 tile 边界，无法适应极端负载倾斜；Megablocks 的 block-sparse 更灵活 |
| **Down Projection 未融合** | 当前未与 unpermute 融合，因 Triton 不支持对 2D accumulator 的标量索引 |
| **CPU-side Schedule Generation** | block schedule 在 CPU 上生成，引入 host-device 同步点 |
| **仅支持推理** | 未实现 backward pass，训练暂不支持 |
| **AMD 性能尚未优化** | 仅验证正确性，性能调优留待未来 |
| **单 GPU 设计** | 未考虑 multi-GPU expert parallelism 与 all-to-all 通信 |

---

### 🔮 未来工作方向

1. **Dynamic Block-to-Expert Assignment**
   - 替代固定 BLOCK_M 调度，根据 runtime 负载动态分配 block，应对 routing skew。

2. **Persistent Kernel for Full Fusion**
   - 将 down GEMM 与 unpermute 融入同一 kernel，进一步减少 launch 开销。

3. **Multi-GPU MoE Dispatch**
   - 扩展至分布式场景，结合 all-to-all 通信优化跨设备专家调度。

4. **Training Support**
   - 实现高效的 backward pass kernel，支持 MoE 模型训练。

5. **AMD 性能优化**
   - 针对 MI300X 进行 autotuning 和 memory access 优化，释放其高带宽潜力。

6. **Expert Caching / Weight Prefetching**
   - 应对 DeepSeek-V3 类超多专家模型（256 experts），缓解权重加载瓶颈。

---

## ✅ 总结

TRITONMOE 成功证明：  
> **仅使用 OpenAI Triton 的 portable primitives，即可构建高性能、跨平台兼容的 MoE 推理内核，在典型推理负载下媲美甚至超越 CUDA 专用实现。**

这标志着 **GPU 编程正从 vendor-locking 向 portable abstraction 演进**，为未来开放、灵活、高效的 AI 系统提供了重要范例。

</details>

---

### 9. [TSFLora: Token-Compressed Split Fine-Tuning for Wireless Edge Networks](https://arxiv.org/abs/2605.23988)

**Authors**: Xianke Qiang, Zheng Chang, Li Wang, Ying-Chang Liang  
**Category**: cs.DC  
**Published**: 2026-05-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.23988v1  

#### Abstract
Adapting large AI models (LAMs) to personalized edge data is challenging because wireless devices have limited memory, computation, and uplink capacity. Federated fine-tuning preserves data privacy but still requires each device to host the full model, while split learning reduces device memory at t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TSFLora: Token-Compressed Split Fine-Tuning for Wireless Edge Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在无线边缘网络中，**大型AI模型（LAMs）的个性化微调面临三大挑战**：
- **设备内存受限**：边缘设备无法承载完整的LAM骨干网络；
- **上行通信开销大**：Split Learning（SFL）虽缓解内存压力，但需上传大量中间激活（activations），导致高通信成本；
- **服务器端计算瓶颈**：随着客户端增多，服务器处理长token序列的计算负担加重。

传统方法如 **Federated Learning (FL)** 要求设备存储完整模型，而 **Parameter-Efficient Fine-Tuning (PEFT)** 如 LoRA 减少可训练参数但仍需保留骨干；**Split Learning** 缓解内存压力却引入通信与服务器计算瓶颈。

---

### 🚀 提出的新方法：TSFLora
提出 **TSFLora** —— 一种**面向边缘网络的 token 压缩式分拆微调框架**，结合以下关键技术：
- **Attention-guided token selection**：基于[CLS] token注意力选择最重要的patch tokens；
- **Token merging**：将未选中的tokens聚合为一个“merged token”，保留被丢弃token的信息；
- **Low-bit activation quantization**：对压缩后的token进行低比特量化（如4-bit、8-bit），进一步减少传输量；
- **LoRA-based adaptation**：在设备侧和服务器侧均部署LoRA适配器，仅更新少量参数；
- **Split Federated Learning 架构**：模型在某一层切分，前端在设备执行，后端在服务器完成。

> 🔑 **核心思想**：在split layer处压缩token序列再上传，**同时降低上行通信开销和服务器端计算负载**，且不修改冻结的主干网络。

---

### ⚖️ 相比现有方法的优势
| 方法 | 内存占用 | 通信开销 | 服务器计算 | 准确率保持 |
|------|----------|-----------|-------------|--------------|
| FL + LoRA | 高（需全模型） | 中等（传LoRA更新） | 分布式 | ✅ |
| Split Learning (SFL) | 低 | 高（传完整activation） | 高 | ✅ |
| SFL + Quantization | 低 | 中～低 | 中 | 受量化影响 |
| **TSFLora（本文）** | **低** | **极低** | **低** | **✅（接近SOTA）** |

> ✅ **优势总结**：
> - 实现**通信效率与系统可扩展性的双重提升**；
> - 在资源受限设备上实现LAM微调成为可能；
> - 不牺牲太多精度的前提下，显著降低带宽和延迟。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **CIFAR-10**
- **CIFAR-100**
- **TinyImageNet**

> 所有任务为图像分类，采用预训练的Vision Transformer（ViT）系列模型。

---

### 🧪 实验设置
- **模型架构**：
  - ViT-Small/32, ViT-Base/32, ViT-Large/32（来自 `timm` 库）
- **训练配置**：
  - 通信轮数：50轮
  - 每轮参与设备数：随机选取10个虚拟设备（共50个候选）
  - Batch size：64
  - Learning rate：0.1
  - LoRA rank：32
- **数据分布**：
  - IID（独立同分布）
  - Non-IID（通过Dirichlet分布 α=0.5 生成，模拟现实场景）
- **评估指标**：
  - Top-1 Accuracy（主要）
  - Communication overhead（MB/round）
  - Device peak memory usage（MB）
  - End-to-end execution time（含通信+计算）

---

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **LocalLoRA** | 各设备本地训练，无协作 |
| **FedLoRA** | 标准联邦学习 + LoRA，需设备存储完整模型 |
| **SplitLoRA** | 分拆学习 + LoRA，无压缩 |
| **SFLora** | Split Learning + LoRA + 全token传输 |
| **SFLora (8-bit), SFLora (4-bit)** | 加入激活量化的变体 |
| **TSFLora (8-bit, K tokens)** | 本文方法：token选择 + 低比特量化 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table III & Fig. 4）

#### ✅ 通信开销大幅下降
- **最高达 6.8× 通信压缩比**（vs. Full Precision SFLora）
- 在 **4-bit + 30 tokens** 设置下，通信体积减少 **>80%**
- 示例（ViT-B/16）：
  - 原始activation传输：约 **233.5 MB/轮**
  - TSFLora（8-bit, 40 tokens）：降至 ~34.3 MB
  - TSFLora（4-bit, 30 tokens）：进一步降至 ~12–15 MB

#### ✅ 显著节省内存
- **设备峰值内存降低最多达 41%**
- 即使在8-bit精度下，仍能控制在典型边缘设备（如4GB RAM）预算内

#### ✅ 保持竞争力的准确率
| 场景 | 方法 | Accuracy |
|------|------|---------|
| CIFAR-100 (non-IID, ViT-Base) | SFLora (8-bit) | 89.97% |
| | **TSFLora (8-bit, 40 tokens)** | **89.04%** |
| | TSFLora (8-bit, 30 tokens) | 88.07% |
> ➕ 仅损失约1%精度，换来巨大通信节约

#### ✅ 收敛性分析（Fig. 2）
- TSFLora收敛速度略慢于SFLora（因更强压缩），但最终精度接近；
- 在非IID数据下依然稳定收敛。

---

### 🔍 消融实验结果（Fig. 3）
- **Token数量影响**：
  - 从50→40 tokens：几乎无精度损失；
  - 50→30 tokens：轻微下降，但通信再降~40%
- **量化比特宽度**：
  - 2-bit：明显精度下降（尤其浅层/深层split时）；
  - ≥4-bit：性能趋于饱和，8-bit已足够
- **Split layer位置**：
  - 中间层cut（e.g., 第2块后）表现最稳健；
  - 浅层或深层split对量化更敏感

> ✅ **发现**：token selection 与 quantization 是互补机制——前者减元素数，后者减每个元素比特数。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Token压缩是高效SFL的关键路径**：
   - 在split layer压缩token序列，可**同步优化通信、计算、内存三方面瓶颈**；
2. **注意力引导的选择优于随机采样**：
   - 利用[CLS] token注意力挑选重要patches，能更好保留语义信息；
3. **中等压缩即可获得高性价比**：
   - **40 tokens + 8-bit** 是实用操作点，在精度与效率间取得良好平衡；
4. **系统级收益显著**：
   - 在10 Mbps上行带宽下，TSFLora端到端训练时间远低于其他方法（Fig. 4c）；
   - 即使在低带宽（5 Mbps）也具备可行性。

---

### ⚠️ 局限性
- **极端压缩会损害性能**：
  - 如2-bit + 10 tokens虽极致省通信，但精度显著下降；
- **依赖Transformer结构特性**：
  - 当前设计针对ViT类模型，是否适用于CNN或其他架构有待验证；
- **动态调整策略尚未集成**：
  - K 和 q 当前为固定值，未来可探索自适应压缩策略。

---

### 🔮 未来工作方向
1. **自适应token budget与bit-width控制**：
   - 根据信道状态、设备负载动态调节压缩强度；
2. **跨模态扩展**：
   - 将TSFLora应用于视觉-语言模型（如CLIP）、语音或多模态任务；
3. **硬件协同优化**：
   - 结合边缘硬件特性（如NPU支持INT4）设计专用量化方案；
4. **理论深化**：
   - 进一步建模token selection与quantization联合扰动下的收敛边界。

---

## ✅ 总结
**TSFLora** 是首个将 **token-level压缩** 引入 **Split Federated Learning** 的PEFT框架，通过 **attention-guided token selection + token merging + low-bit quantization + LoRA** 四重机制，在边缘设备上实现了高效、低通信、低内存的大模型微调。实验证明其可在**几乎不失精度的情况下实现高达6.8倍通信压缩和41%内存节省**，为资源受限环境下的边缘智能提供了切实可行的技术路径。

</details>

---

### 10. [Distilling Game Code World Model Generation into Lightweight Large Language Models](https://arxiv.org/abs/2605.24375)

**Authors**: Tyrone Serapio, Arjun Prakash, Haoyang Xu, Kevin Wang, Amy Greenwald  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.24375v1  

#### Abstract
Large Language Models (LLMs) have shown great ability in generating executable code from natural language, opening the possibility of automatically constructing environments for AI agents. Recent work on Code World Models (CWMs) demonstrates that LLMs can translate game rules into Python implementat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Distilling Game Code World Model Generation into Lightweight Large Language Models*

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Models (LLMs)** 自动生成可执行游戏环境（即 **Code World Models, CWMs**）的方法存在两大瓶颈：
- **依赖前沿大模型**（如 GPT-4、Gemini 2.5 Pro），限制了可访问性和部署成本；
- **严重依赖推理时计算**（test-time compute），例如通过迭代自修正（iterative refinement loops）结合单元测试来验证和修复生成代码，导致推理效率低下。

本文聚焦于多智能体博弈场景，提出 **Game Code World Models (GameCWMs)** —— 能够实现游戏规则、合法动作、状态转移、观测和奖励等完整逻辑的可执行 Python 环境，并致力于解决上述可扩展性与资源效率问题。

### 提出的新方法与创新点
论文提出了一个完整的 **后训练（post-training）蒸馏框架**，将 GameCWM 生成能力从大模型“蒸馏”到轻量级开源 LLM 中，具体包括三大核心贡献：

1. **Dataset Curation**  
   构建了一个包含 **30 个游戏** 的高质量数据集，涵盖完美信息（perfect information）与不完美信息（imperfect information）游戏，且包含多个 **Out-of-Distribution (OOD)** 游戏（如 *Converge*, *Quadranto*），用于训练和评估模型泛化能力。

2. **Hierarchical Verification Framework**  
   设计了一个四层自动验证框架，无需依赖已有游戏引擎（如 OpenSpiel）即可评估生成代码的正确性：
   - **Tier 1: Static Analysis** – 检查语法、API 完整性、类型签名、导入等；
   - **Tier 2: Dynamics (Fuzzing)** – 随机生成 100 条轨迹，测试确定性、非终端状态合法性、状态不可变性等；
   - **Tier 3: Semantic Rule Adherence** – 使用 **LLM 生成的场景轨迹（scenario traces）** 进行语义对齐测试；
   - **Tier 4: Information Consistency** – 针对不完美信息游戏，验证 `resample_history` 函数是否保持观测一致性。

3. **Two-Stage Post-Training Pipeline (SFT + RLVR)**  
   提出结合 **监督微调（Supervised Fine-Tuning, SFT）** 和 **基于可验证奖励的强化学习（Reinforcement Learning with Verifiable Rewards, RLVR）** 的训练流程：
   - SFT 阶段：在 prompt-code 对上进行有监督训练，使模型掌握 API 结构；
   - RLVR 阶段：采用 **Group Relative Policy Optimization (GRPO)**，利用验证框架输出的复合奖励信号进一步优化模型，提升其生成符合语义规则的代码的能力。

### 相比现有方法的优势
- **摆脱对前沿模型的依赖**：成功在仅 **3B 参数规模** 的 Qwen2.5-3B-Instruct 上实现有效蒸馏；
- **减少推理时开销**：训练阶段完成复杂逻辑内化，避免测试时反复调用 LLM 自我修正；
- **支持 OOD 泛化**：验证框架不依赖真实轨迹，适用于无标准实现的新游戏；
- **端到端自动化**：整个训练与评估流程高度自动化，具备良好可复现性。

---

## 2. 核心实验方法和设置

### 数据集
- **总数**：30 个游戏，划分为 **23 个训练游戏** 和 **7 个保留测试游戏**；
- **类别覆盖**：
  - **完美信息游戏**：Tic-Tac-Toe, Connect4, Y, Generalized Chess, Converge 等；
  - **不完美信息游戏**：Blackjack, Kuhn Poker, Leduc Poker, Gin Rummy, Quadranto, Hand of War 等；
  - **OOD 游戏**：Generalized Tic-Tac-Toe, Generalized Chess, Converge, Quadranto, Hand of War 等（不在 Qwen 训练数据中）；
- **Prompt-Code Pair 构成**：
  - 输入（prompt）：API 规范 + 自然语言游戏描述 + 允许的动作字符串 + 示例轨迹（由 Gemini 3 Pro 生成）；
  - 输出（code）：人工精炼并通过严格测试的 **ground-truth GameCWM 实现**。

### 实验设置
- **基础模型**：Qwen2.5-3B-Instruct；
- **训练配置**：
  - SFT：LoRA 微调，2 轮训练；
  - GRPO：基于 SFT 模型继续训练，温度 0.6，KL 正则项；
  - SFT+GRPO：联合流程；
- **对比模型**：
  - Base：原始 Qwen2.5-3B-Instruct；
  - SFT：仅监督微调；
  - GRPO：直接在 base 上进行 RLVR；
  - SFT+GRPO：完整流水线；
  - GPT-4o：作为强基线参考。

### 评估指标
- **综合验证得分（Verification Score）**：
  $$
  r = w_1 \cdot S_{\text{static}} + w_2 \cdot S_{\text{dynamics}} + w_3 \cdot S_{\text{semantics}} + w_4 \cdot S_{\text{information}}
  $$
  权重设置为：`[0.15, 0.25, 0.3, 0.3]`，强调语义与信息处理；
- **分层得分**：分别报告四层验证的平均通过率；
- **采样策略**：每游戏生成 25 个样本，取平均得分；
- **温度设置**：评估时设为 0.3，平衡准确性与多样性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

| Model | Mean Score | Static | Fuzz | Semantics | Information |
|-------|------------|--------|------|-----------|-------------|
| **Base** | 47.3% | 77.5% | 70.2% | 8.5% | 3.3% |
| **SFT** | 49.0% | 78.0% | 74.3% | 9.4% | 2.0% |
| **GRPO** | 50.4% | 83.3% | 75.3% | 8.4% | 3.7% |
| **SFT+GRPO** | **53.2%** | **86.3%** | 75.1% | **14.4%** | **5.3%** |
| **GPT-4o** | 66.7% | 98.4% | 84.6% | 33.8% | 14.0% |

> ✅ **SFT+GRPO 在所有后训练模型中表现最佳**，显著优于其他变体。

### 与基线方法的对比结果
- **SFT 显著提升语法与结构正确性**：Static 分数从 77.5% 提升至 86.3%，表明模型更好掌握了 API 格式；
- **RLVR 显著增强语义遵循能力**：SFT+GRPO 的语义得分是 SFT 的 **1.5 倍以上**（14.4% vs 9.4%），说明执行反馈有效引导模型理解深层规则；
- **SFT 是 RLVR 成功的前提**：单独使用 GRPO 效果弱于 SFT+GRPO，验证了“先格式对齐，再语义优化”的两阶段必要性；
- **仍远逊于 GPT-4o**：尤其在语义（14.4% vs 33.8%）和信息处理（5.3% vs 14.0%）方面差距明显，反映小模型能力边界。

### 消融实验结果（见 Appendix E）
- **移除 scenario traces 会导致语义得分下降**：在多个 OOD 游戏中（如 Gen. Tic-Tac-Toe, Y），包含 scenario 的奖励机制带来更高语义通过率；
- 验证了 **LLM 生成的 scenario traces 可作为有效的轻量级语义监督信号**，尤其适用于缺乏真实轨迹的新游戏。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **GameCWM 生成能力可以被蒸馏到小型 LLM 中**：通过 SFT+RLVR 流程，Qwen2.5-3B-Instruct 的 GameCWM 生成质量得到系统性提升；
2. ✅ **SFT 改善语法与结构，RLVR 提升语义与执行一致性**：两者互补，联合使用效果最优；
3. ✅ **验证框架支持 OOD 游戏评估**：基于 LLM 生成的 scenario traces 和 property-based testing 实现了对新游戏的有效验证；
4. ⚠️ **不完美信息游戏仍是巨大挑战**：所有模型（包括 GPT-4o）在 `resample_history` 功能上的表现均极低，表明“心智理论”（Theory of Mind）推理尚未被充分掌握；
5. 💬 **模型表现出“自我意识”**：部分生成代码中出现注释如 “resampling logic is too hard”，暗示模型意识到自身局限。

### 方法的局限性
- **数据集有限**：尽管多样，但仍不足以覆盖复杂博弈逻辑，尤其在不完美信息领域；
- **语义验证依赖 LLM 生成的 scenario**：这些 traces 本身可能错误或不全面，影响奖励信号可靠性；
- **未评估下游任务性能**：仅验证代码正确性，未测试生成的 GameCWM 是否能支持 MCTS/ISMCTS 实现强博弈策略；
- **小模型容量限制**：3B 模型难以承载复杂状态追踪与反事实推理，尤其在 OOD 不完美信息游戏中表现崩溃（如 Gin Rummy 上 SFT 导致性能下降）。

### 未来工作方向
- **改进验证框架**：引入更严格的 **information-set 正确性检查**，防止隐藏信息泄露；
- **扩大数据与模型规模**：尝试更大模型（如 14B）并增加不完美信息游戏数量；
- **端到端 gameplay 评估**：将生成的 GameCWM 接入 MCTS/ISMCTS，评估实际博弈表现；
- **迭代自改进机制**：结合当前训练模型与 unit test refinement loop，借鉴 Lehrach et al. 的优势；
- **探索更高效 RL 算法**：研究如何突破 base model sampling distribution 的限制，真正“发明”新推理路径。

---

> 📌 **总体结论**：本文首次展示了将 GameCWM 生成能力从大模型蒸馏至轻量级 LLM 的可行性，提出了一套完整的 **数据构建、验证与训练框架**，为低成本、高可扩展的自动环境生成提供了新路径。虽然在复杂语义与不完美信息处理上仍有局限，但为未来构建具备战略推理能力的 AI Agent 奠定了重要基础。

</details>

---

### 11. [FrontierOR: Benchmarking LLMs' Capacity for Efficient Algorithm Design in Large-Scale Optimization](https://arxiv.org/abs/2605.25246)

**Authors**: Minwei Kong, Chonghe Jiang, Ao Qu, Wenbin Ouyang, Zhaoming Zeng, Xiaotong Guo, Zhekai Li, Junyi Li, Yi Fan, Xinshou Zheng, Xi Jing, Yikai Zhang, Zhiwei Liang, Seonghoo Kim, Runqing Yang, Zijian Zhou, Sirui Li, Han Zheng, Wangyang Ying, Ou Zheng, Chonghuan Wang, Jinglong Zhao, Hanzhang Qin, Cathy Wu, Paul Pu Liang, Jinhua Zhao, Hai Wang  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.25246v1  

#### Abstract
Large language models (LLMs) are increasingly used for optimization modeling and solver-code generation, yet practical operations research and optimization problems often require a harder capability: designing scalable algorithms that exploit problem structure and outperform direct formulation-and-s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# FrontierOR: Benchmarking LLMs' Capacity for Efficient Algorithm Design in Large-Scale Optimization —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前大多数关于 **Large Language Models (LLMs)** 在 **Operations Research (OR)** 和优化领域的研究，主要集中在 **优化建模能力** 上，即能否将自然语言描述的问题转化为数学规划公式或可执行代码（如 MIP、MINLP）。然而，现实世界中的大规模优化问题往往需要更高级的能力：

> **LLMs 是否能设计出高效的算法（efficient algorithms），而不仅仅是生成一个可运行但低效的“单体求解器调用”（monolithic solver call）？**

现有基准（如 NL4OPT、OptiBench、CO-Bench）存在以下不足：
- 规模太小（通常变量数 < 10^4）
- 缺乏真实复杂结构（如分解性、时序耦合、场景树等）
- 仅评估建模正确性，不评估算法效率（runtime、quality-time tradeoff）

因此，**缺乏一个系统性的、面向大规模、真实 OR 场景的基准来评估 LLMs 的高效算法设计能力**。

---

### ✅ 提出了什么新方法或新思路

作者提出了 **FrontierOR**，这是首个专门用于评估 LLMs 在 **大规模优化问题中设计高效算法能力** 的基准测试平台。

#### 主要创新点：

1. **提出“算法效率”作为核心评估维度**
   - 不再只看是否能写出正确的数学模型（formulation accuracy）
   - 而是看生成的算法是否在 **解的质量（solution quality）** 和 **计算效率（computational efficiency）** 上优于或媲美专家实现的 Gurobi 基线

2. **构建高质量、文献驱动的大规模任务集合**
   - 收录来自顶级 OR 期刊（如 *Operations Research*, *Management Science*）的 **180 个真实优化问题**
   - 每个任务包含：
     - 自然语言描述（natural-language problem description）
     - 大规模实例（up to ~10⁷ 变量/约束）
     - 数学建模（mathematical formulation）
     - 验证过的 Gurobi 实现（expert-verified Gurobi baseline）
     - 独立的可行性检查器（standalone feasibility checker）

3. **引入 Hard 子集以聚焦真正困难的问题**
   - 从 180 个任务中筛选出 50 个“Hard”任务，标准包括：
     - 组合爆炸类问题（如排程、装箱）
     - 强耦合结构（多阶段时间窗、容量紧张）
     - Gurobi 在 1 小时内无法达到最优（solver saturation）

4. **建立端到端、自动化的评估流水线**
   - 所有实验在统一 Docker 环境下运行（单核 CPU，禁用网络）
   - 采用多维指标综合评价算法性能

---

### ✅ 相比现有方法的优势

| 维度 | 传统基准（如 NL4OPT） | FrontierOR |
|------|------------------------|------------|
| 评估目标 | 建模正确性（formulation accuracy） | 算法效率（algorithm efficiency） |
| 实例规模 | 小型（< 10^4 变量） | 大型（median ~40K vars, 最高达 10⁷） |
| 任务来源 | 合成或简化问题 | 文献驱动的真实 OR 问题 |
| 基线对比 | 无或弱基线 | 专家验证的 Gurobi 实现 |
| 评估方式 | 是否可行/目标值 | 是否在质量与时间上同时胜出 |

---

## 2. 核心实验方法和设置

### 📚 使用了哪些数据集

- **FrontierOR 数据集**：包含 180 个任务，源自 1992–2025 年发表于顶级 OR 期刊的研究论文。
- **问题类别覆盖广泛**：
  - Routing & TSP
  - Scheduling
  - Lot Sizing & Inventory
  - Location & Network Design
  - Packing & Knapsack
  - Stochastic / Robust Optimization
- **应用领域多样**：
  - Transportation, Healthcare, Energy, Supply Chain, Manufacturing 等
- **每个任务实例具有挑战性**：
  - 中位数：约 40,000 个变量，18,000 个约束
  - 46% 的大规模实例在 1 小时内 Gurobi 无法达到最优

---

### ⚙️ 实验设置和评估指标

#### 实验环境
- 单核 AMD EPYC 9554X 处理器
- Docker 容器化运行（Python 3.13 + Gurobi 12）
- 禁用网络访问，确保公平性

#### 输入给 LLM 的内容
- 仅提供：
  - 自然语言问题描述（不含数学符号）
  - 输入/输出格式（JSON schema）
- **不提供**：
  - 数学公式
  - 参考代码
  - 算法提示

#### 评估流程（Two-stage）
1. **可行性检查（Correctness）**
   - 使用独立的 `feasibility checker` 验证输出是否满足所有硬约束
   - 若不可行，则目标值无效（no objective credit）

2. **性能评估（Performance）**
   - 对可行解，比较其与 Gurobi 基线的：
     - **解质量（solution quality）**
     - **运行时间（runtime）**

#### 主要评估指标

| 指标 | 定义 |
|------|------|
| **Execution Rate (Exec.)** | 成功执行且无运行错误的比例 |
| **Feasibility** | 在大规模实例上返回可行解的比例 |
| **Solution Quality (Sol. q.)** | 可行解的目标值在 Gurobi 基线 **1% 以内** 的比例 |
| **Quality-Time Efficiency (QTE)** | 同时满足：<br>✅ 目标值 ≤ 1% 差距<br>✅ 运行时间 ≤ Gurobi 时间<br>—— 是核心指标！ |

---

### 🔁 基线方法对比

#### 测试的 LLMs（共 7 个）

| 类型 | 模型 |
|------|------|
| **Frontier Models** | GPT-5.3-Codex, Claude Opus 4.6, Gemini 3.1 Pro Preview |
| **Cost-effective/Open-source** | DeepSeek-R1, Grok-4.20-beta, Qwen3-Coder-Plus, LLaMA-4-Maverick |

#### 推理协议（Protocols）

1. **One-shot Generation**
   - LLM 直接根据描述生成完整程序
   - 允许少量自调试（self-debugging）修复语法错误
   - 通过 “tiny instance” 初步验证功能

2. **Self-Evolving Frameworks**（测试时演化）
   - 使用反馈迭代改进候选程序
   - 包括三种代表性框架：
     - **AlphaEvolve**（基于 MAP-Elites 的进化）
     - **EoH**（Evolution of Heuristics，联合演进代码与思维）
     - **CORAL**（多智能体协作演化）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（One-shot Setting）

| Model | Exec. Rate (Full) | Feasibility (Full) | Sol. Quality (Full) | **QTE (Full)** | QTE (Hard) |
|-------|-------------------|--------------------|---------------------|----------------|------------|
| **Claude Opus 4.6** | 0.93 | 0.62 | 0.48 | **0.31** | **0.32** |
| GPT-5.3-Codex | 0.98 | 0.60 | 0.48 | 0.26 | 0.18 |
| Gemini 3.1 Pro | 0.93 | 0.61 | 0.52 | 0.25 | 0.22 |
| DeepSeek-R1 | 0.74 | 0.42 | 0.31 | 0.17 | 0.11 |
| LLaMA-4-Maverick | 0.47 | 0.18 | 0.13 | 0.06 | 0.02 |

> 💡 **最强 one-shot 模型（Claude Opus 4.6）仅在 31% 的任务上实现了 QTE 胜出**

---

### 🔍 与基线方法的对比结果

#### ✅ 与 Gurobi 基线对比
- 即使是最强 LLM，在 **Hard 子集** 上也仅有：
  - 平均 **32% 的任务** 能同时在质量和时间上击败 Gurobi
  - 表明当前 LLMs **远未达到可靠算法工程师水平**

#### ✅ 不同模型的行为差异
| 模型类型 | 倾向使用的算法类型 | 特点 |
|--------|------------------|------|
| **Frontier Models** | 更多使用 decomposition, local search, matheuristic | 能探索非单体结构 |
| **Weaker Models** | 几乎总是调用 monolithic solver（如 LLaMA-4-Maverick 达 99%） | 缺乏算法设计能力 |

#### ✅ 失败模式分析（Failure Modes）
- **弱模型失败早**：常犯接口错误、约束遗漏、schema violation
- **强模型失败晚**：常因启发式搜索不足导致质量下降（heuristic search failure）
- 表明瓶颈已从“写对代码”转向“设计好算法”

---

### 🔬 消融实验结果（Self-Evolution 效果）

| Method | Feasibility | Sol. Quality | **QTE** |
|--------|-------------|--------------|---------|
| One-shot (baseline) | 0.45 | 0.18 | 0.15 |
| EoH | 0.72 | 0.43 | 0.33 |
| OpenEvolve | 0.92 | 0.61 | **0.49** |
| **CORAL** | **1.00** | **0.67** | **0.50** |

> ✅ 所有 self-evolving 框架显著提升性能  
> ✅ CORAL 表现最佳，尤其在稳定性与最终 QTE 上领先  
> ✅ 说明 **test-time search 是提升 LLM 算法能力的关键路径**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLMs 尚不具备可靠的高效算法设计能力**
   - 当前最强 LLM 在 one-shot 设置下仅能在 **约 1/3 的任务上超越 Gurobi 基线**
   - 多数成功依赖于直接调用求解器，而非结构性优化

2. **模型能力分层明显**
   - Frontier models 明显优于 cost-effective/open-source 模型
   - 强模型更倾向于使用 decomposition、matheuristic 等高级策略

3. **失败模式随模型强度迁移**
   - 弱模型：失败于建模阶段（constraint error, schema violation）
   - 强模型：失败于搜索质量（heuristic inefficiency），即“知道怎么建模，但不会有效求解”

4. **Self-evolution 显著提升性能**
   - 通过 test-time feedback 迭代优化，QTE 可从 15% 提升至 **50%**
   - 表明 **推理时搜索（test-time search）是未来突破方向**

5. **算法多样性决定竞争力**
   - Hybrid、matheuristic、local search 方法通常比纯 solver call 更具效率优势
   - Claude Opus 4.6 因算法选择更均衡而在 QTE 上表现最好

---

### ⚠️ 方法的局限性

1. **任务范围受限于公开文献**
   - 仅能收录有明确数学建模和可复现实例的论文
   - 可能忽略某些依赖私有数据或黑盒系统的工业级问题

2. **评估依赖 Gurobi 基线**
   - 虽然 Gurobi 是行业标准，但仍可能偏向特定类型的建模风格
   - 未来可扩展支持其他求解器（如 CPLEX、SCIP）

3. **人力成本高**
   - 每个任务需经过 15 名 OR 专家三周多轮审核
   - 难以快速扩展到千级任务规模

4. **未涵盖强化学习或神经求解器**
   - 当前评估对象为 LLM 或 LLM-agent，未包含纯 ML-based solver（如 NeuroSAT、Graph Neural Networks）

---

### 🔮 未来工作方向

1. **推动 LLM 向“算法工程师”角色演进**
   - 开发能自主进行 decomposition、relaxation、primal-dual 设计的 agent
   - 结合 symbolic reasoning 与 procedural generation

2. **发展更强大的 test-time evolution 框架**
   - 如 CORAL 所示，multi-agent collaboration 可带来质变
   - 探索 memory-augmented、tool-integrated 的 agentic 架构

3. **构建动态更新的 benchmark**
   - 随着新论文发布持续添加新任务
   - 支持社区提交与 peer review

4. **拓展至多目标、在线、分布式优化场景**
   - 当前主要针对静态、单目标问题
   - 下一步可引入 streaming data、real-time decision-making 等挑战

---

> 🔗 **项目主页**：https://anonymous.4open.science/r/efficient-opt-bench-F03D  
> 📄 **论文链接**：[arXiv:2605.25246](https://arxiv.org/abs/2605.25246)

</details>

---

### 12. [MATO: Multi-objective Personalized Alignment with Test-time Optimization for Large Language Models](https://arxiv.org/abs/2605.25342)

**Authors**: Linhao Luo, Thuy-Trang Vu, Van-Anh Nguyen, Junae Kim, Gholamreza Haffari, Dinh Phung  
**Category**: cs.CL  
**Published**: 2026-05-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.25342v1  

#### Abstract
Aligning large language models (LLMs) with diverse and multifaceted user preferences is a fundamental challenge in personalized AI systems. Existing multi-objective alignment methods either rely on costly training or require pre-trained reward models for each preference, making it difficult for them...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MATO: Multi-objective Personalized Alignment with Test-time Optimization for Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **multi-objective personalized alignment** 方法面临以下挑战：
- **训练成本高**：基于强化学习的方法（如 MORLHF）需要为每种偏好组合单独训练模型，难以扩展。
- **依赖外部 Reward Model**：test-time alignment 方法通常需要预训练的 token-level reward model 来引导解码，但这些 reward model 难以随用户偏好的动态变化而更新。
- **prompting steerability 不足**：虽然 prompt-based 方法是 training-free 的，但在处理多个竞争性目标时，LLM 容易过度强调某些偏好而忽略其他，且无法可靠控制不同目标之间的权衡。

### 提出了什么新方法或新思路
本文提出 **MATO**（Multi-objective Alignment with Test-time Optimization），一种无需训练、不依赖外部 reward model 的个性化对齐框架。其核心思想是将多目标个性化对齐建模为一个 **test-time optimization** 问题，在生成过程中通过可调节的权重动态优化 token 分布。

MATO 包含三个关键模块：
1. **Reward Discovery**  
   利用指令跟随能力，直接从 backbone LLM 中恢复每个目标的偏好奖励，无需外部 reward model。具体做法是通过在 prompt 中加入目标描述 $ c_k $，计算条件分布与原始分布的 log-ratio 作为 token-level reward：
   $$
   R_k(y_i|x,y_{<i}) = \beta \log \frac{\pi_{\text{base}}(y_i|x,y_{<i},c_k)}{\pi_{\text{base}}(y_i|x,y_{<i})}
   $$

2. **Weight Optimization**  
   动态调整各目标的权重，以平衡生成过程中的偏好满足度。基于已生成前缀的累积奖励，采用带正则化的优化子问题求解最优权重：
   $$
   \mathbf{w}^* = \arg\min_{\mathbf{w} \in \Delta^K} \sum_k w_k R_k(x,y_{<i}) + \tau D_{\text{KL}}(\mathbf{w} \| \mathbf{w}_{\text{init}})
   $$
   该问题有闭式解，能快速重加权被忽视的目标。

3. **Online Optimization**  
   使用 **Follow-the-Regularized-Leader (FTRL)** 在线优化策略迭代更新输出分布，使其向更优的多目标对齐策略逼近，提升生成质量。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **Training-free** | 无需任何模型微调或额外训练，适用于任意 backbone LLM。 |
| **Reward-free** | 不依赖预训练 reward model，适应动态演化的新偏好。 |
| **Strong steerability** | 用户可通过初始权重控制目标间权衡，MATO 能有效响应并实现帕累托改进。 |
| **Balanced alignment** | 动态权重机制防止某些偏好被忽略，显著提升最差偏好评分（Worst）。 |
| **Model-agnostic & Scalable** | 可应用于多种 LLM，并支持任意数量和类型的自然语言描述偏好。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **Multifaceted Dataset [12]**  
  包含 945 个查询，每个查询分配 4 个来自 4 个维度（背景知识、无害性、信息量、风格）的偏好，共涉及 107 种独特偏好组合。用于评估多目标个性化对齐性能。
  
- **HH-RLHF Dataset [1]**  
  广泛用于多目标对齐研究。从中选取 **helpfulness** 和 **humor** 两个目标，通过改变二者权重（从 1:0 到 0:1，步长 0.2）来评估 steerability。

### 实验设置和评估指标

#### 多目标对齐评估指标（使用 GPT-4o / claude-sonnet-4-5 作为 judge）
| 指标 | 描述 |
|------|------|
| **AMR (All Preference Match Rate)** | 所有偏好评分均 ≥3 的响应比例，衡量完全满足所有偏好的能力。 |
| **APS (Average Preference Score)** | 各偏好评分的平均值，反映整体对齐质量。 |
| **Worst Preference Score** | 单个响应中最低的偏好得分，反映多目标平衡性。 |

#### Steerability 评估
- 使用两个预训练 reward model（helpfulness 和 humor）给出连续分数。
- 绘制不同权重下的 **empirical Pareto front**，评估方法在目标权衡上的可控性和表现。

### 基线方法对比

#### Training-free Baselines
- **Base LLM**：无任何对齐。
- **Preference Prompt [12]**：将多个偏好拼接进 prompt。
- **Linear Alignment [6]**：基于偏好 prompt 进行 token 分布线性更新。
- **CoS [8]**：通过插值增强偏好影响。
- **Amulet [32]**：单目标在线学习。
- **OPAD [37]**：基于发散性奖励优化分布。

#### Training-based Baselines
- **MORLHF [13]**：多目标强化学习。
- **RiC [31]**：监督微调 + 偏好向量输入。
- **MOD [17]**：多个单目标模型加权合并。

> 注：由于 training-based 方法需针对特定偏好组合训练，实际不可行于 Multifaceted 数据集，仅在 HH-RLHF 上进行 steerability 对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Multifaceted Dataset）

| 方法 | Backbone | AMR ↑ | APS ↑ | Worst ↑ |
|------|----------|--------|-------|---------|
| Base LLM | Qwen3-0.6B | 0.14 | 2.58 | 1.49 |
| Preference Prompt | Qwen3-0.6B | 0.19 | 2.78 | 1.64 |
| OPAD | Qwen3-0.6B | 0.25 | 2.97 | 1.80 |
| **MATO (Ours)** | Qwen3-0.6B | **0.27** | **2.99** | **1.87** |
| Base LLM | Qwen3-8B | 0.43 | 3.66 | 2.34 |
| Preference Prompt | Qwen3-8B | 0.70 | 4.19 | 3.23 |
| OPAD | Qwen3-8B | 0.74 | 4.28 | 3.39 |
| **MATO (Ours)** | Qwen3-8B | **0.75** | **4.30** | **3.42** |

> ✅ **MATO 在所有 backbone 和指标上 consistently 超出所有 baseline**

### 与基线方法的对比结果
- **优于 prompt-based 方法**：  
  MATO 显著优于 Preference Prompt、CoS、Linear Alignment 等，尤其在 **Worst** 指标上提升明显（例如在 Llama-3.2-1B 上比 OPAD +0.37），说明其能更好平衡多个目标。
  
- **优于 training-based 方法的 steerability**：  
  在 HH-RLHF 上绘制的 **Pareto front** 显示：
  - Training-based 方法（MORLHF, RiC, MOD）虽有良好 steerability，但整体 alignment 性能较低。
  - Training-free 方法（如 Preference Prompt, OPAD）alignment 强但 Pareto front 分布差，steerability 弱。
  - **MATO 同时实现了强 alignment 和有效 steerability**，Pareto front 更平滑且覆盖范围广。

### 消融实验结果（Ablation Study）
使用 Qwen3-0.6B 在子集上验证各组件作用：

| 方法 | AMR | APS | Worst |
|------|-----|-----|-------|
| MATO (Full) | **0.27** | **2.99** | **1.87** |
| w/o weight optimization | 0.26 | 2.88 | 1.80 |
| w/o online optimization | 0.24 | 2.83 | 1.83 |
| w/o both | 0.21 | 2.73 | 1.62 |

> 🔍 结论：
> - **Weight Optimization** 对提升 Worst 最重要，证明其有效平衡被忽视的目标。
> - **Online Optimization** 进一步提升整体质量。
> - 两者协同带来最大增益。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Test-time optimization 是实现 scalable、controllable 个性化对齐的有效范式**。  
   MATO 成功地将复杂的多目标优化问题转化为可在推理阶段执行的轻量级优化流程。

2. **Reward Discovery + Dynamic Weighting 实现了真正的多目标平衡**。  
   通过从 LLM 自身提取 reward 并动态调整权重，MATO 能主动补偿未被充分满足的偏好，避免“偏好坍缩”。

3. **Strong steerability 与 high alignment performance 可兼得**。  
   MATO 在保持高质量生成的同时，允许用户通过初始权重精确控制目标间的 trade-off，优于现有方法。

4. **方法具有良好的通用性和可扩展性**。  
   在不同规模的 LLM（0.6B ~ 8B）和不同数量的目标（2~10）下均表现稳定且持续领先。

### 方法的局限性
- 依赖 backbone LLM 的 **instruction-following 能力**。若基础模型无法理解偏好描述，则 reward discovery 效果受限。
- 假设所有目标均可通过 **natural language** 表达，对于隐式或复杂目标可能不适用。
- **Online optimization 引入一定延迟**（约 0.16s/token），虽远低于训练成本，但仍高于标准 decoding。

### 未来工作方向
- 探索更高效的 online optimization 算法以降低延迟。
- 将 MATO 扩展至 vision-language 或 agent 决策等多模态场景。
- 研究如何自动发现潜在冲突目标并提供解释性反馈。
- 结合 user feedback 实现闭环自适应优化。

---

> 📌 **一句话总结**：  
> MATO 提出了一种无需训练、无需外部 reward model 的 test-time optimization 框架，首次实现了 **training-free、reward-free、yet highly steerable and balanced multi-objective alignment**，为个性化大模型对齐提供了新范式。

</details>

---

### 13. [Selective Latent Thinking: Adaptive Compression of LLM Reasoning Chains](https://arxiv.org/abs/2605.25745)

**Authors**: Hui Xie, Jie Liu, Ziyue Qiao, Joaquin Vanschore  
**Category**: cs.CL  
**Published**: 2026-05-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.25745v1  

#### Abstract
Explicit chain-of-thought (CoT) reasoning substantially improves the reasoning ability of large language models (LLMs), but incurs high inference cost due to lengthy autoregressive traces. Existing latent reasoning methods offer a promising alternative, yet they often treat reasoning as uniformly co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Selective Latent Thinking: Adaptive Compression of LLM Reasoning Chains

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在执行复杂推理任务时广泛采用显式的 **Chain-of-Thought (CoT)** 推理方式，虽然有效但生成过程冗长，导致**高推理延迟和内存开销**。现有的**Latent Reasoning**方法试图通过将整个推理链压缩到连续的隐空间（latent space）来提升效率，但通常假设所有中间推理步骤都可均匀压缩，忽略了某些**精度敏感的关键步骤**（如数学计算中的数值操作）必须保留为显式文本以保证正确性。

这一“**均匀可压缩性假设**”在数学等精确推理场景中会导致严重错误，牺牲了准确性。

---

### 提出了什么新方法或新思路
本文提出 **Selective Latent Thinking (SLT)** ——一种**选择性地压缩**推理链的新框架。其核心思想是：

> 在同一条推理轨迹中，**动态交替使用显式 CoT 和隐式 Latent Reasoning**：
> - 将**冗余、可预测的推理片段**（如自然语言连接词、简单逻辑过渡）压缩为紧凑的 **latent 表示**
> - 将**精度关键的推理步骤**（如数字运算、最终答案生成）保留在**显式文本形式**

为此，SLT 引入了一个由三部分组成的动态切换机制：
1. **轻量级 Feature Decoder (D)**：基于当前 LLM 隐藏状态，预测接下来几个 token 的推理内容。
2. **置信门控模块 (Confidence Gate)**：评估预测轨迹的可靠性，决定最长可安全压缩的前缀长度。
3. **Latent Compressor (C)**：将被接受的推理片段编码为一个固定长度的 latent 向量，并注入主 LLM 以跳过多步自回归生成。

训练策略分为三个阶段：
- **Stage 1**: 学习如何将显式推理片段压缩为 latent 表示（Latent Compression Learning）
- **Stage 2**: 联合训练 Feature Decoder 和 Confidence Gate，使其能可靠预测并判断何时可压缩（Reliability-Aware Feature Decoding）
- **Stage 3**: 使用 **Group Relative Policy Optimization (GRPO)** 进行轨迹级强化学习，优化全局准确率与推理成本之间的权衡

---

### 相比现有方法的优势
| 对比维度 | 传统 Latent Reasoning（如 Coconut, CoLaR） | SLT |
|--------|--------------------------------------|-----|
| 压缩策略 | 统一压缩整个推理链 | 动态选择性压缩 |
| 关键步骤处理 | 可能过度压缩导致错误 | 显式保留精度关键步骤 |
| 效率-精度平衡 | 往往牺牲精度换效率 | 实现更优 trade-off |
| 控制能力 | 固定模式 | 可通过阈值调节压缩强度 |

SLT 不再追求“全隐”或“全显”，而是实现了一种**混合推理范式**，兼顾效率与准确性。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **主训练与测试集**：
  - `GSM8k-AugNL`：约 385k 训练样本，自由格式的自然语言 CoT，贴近实际使用场景
- **鲁棒性测试集**：
  - `GSM-Hard`, `SVAMP`, `MultiArith`：更具挑战性的数学推理任务
- **结构化格式对比集**：
  - `GSM8k-Aug`：带有标签的结构化推理路径（如 `<5*2=10>`），用于与 Coconut、CoLaR-2 公平比较
- **跨域泛化测试集**：
  - `Math500`：更难的数学问题，评估迁移能力

---

### 实验设置和评估指标
#### 主干模型（Backbone）
- `Llama-3.2-1B`
- `Qwen3-4B`

#### 评估指标
| 指标 | 含义 |
|------|------|
| `Pass@1` | 第一次生成的答案是否正确（Accuracy %） |
| `#L` | 显式生成的 CoT token 数量（越低越好） |
| `△acc/L` | 准确率增益相对于 SFT-w/o CoT 除以 CoT 长度，衡量推理效率 |

#### 训练流程
1. **Stage 1**：LoRA 微调 Latent Compressor，逐步增加压缩比例
2. **Stage 2**：冻结主 LLM，训练 Feature Decoder 和 Confidence Gate
3. **Stage 3**：解冻所有模块，使用 GRPO 进行轨迹级 RL 优化

---

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| `SFT-w/o CoT` | 基线 | 不生成推理过程，直接输出答案 |
| `SFT-CoT` | 显式 CoT | 完整微调，生成完整推理链 |
| `Coconut`, `CoLaR-2`, `RoT` | 隐式 Latent Reasoning | 将 CoT 蒸馏为 latent tokens |
| `Ours-SFT` | 本文方法（监督训练后） | Stage 2 结束后的模型 |
| `Ours-RL` | 本文方法（最终版） | 经过 RL 优化后的完整 SLT 模型 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### 在 Llama-3.2-1B 上的表现（平均 across 四个数据集）
| 方法 | 平均 Pass@1 (%) | #L (token) | △acc/L |
|------|------------------|------------|--------|
| SFT-CoT | 56.87 | 62.13 | 0.515 |
| CoLaR-2 | 31.40 | 29.84 | 0.126 |
| **Ours-RL** | **54.07** | **25.87** | **1.021** |

> ✅ **结论**：
> - 相比显式 CoT：**减少 58.4% 的推理长度**（62.13 → 25.87），仅损失 **2.8% 准确率**
> - 相比 Latent Baseline（CoLaR-2）：**准确率高出 22.7%**（54.07 vs 31.40），且推理更短

#### 在 Qwen3-4B 上的表现
| 方法 | 平均 Pass@1 (%) | #L (token) | △acc/L |
|------|------------------|------------|--------|
| RoT | 55.40 | 32.00 | 0.022 |
| **Ours-RL** | **72.73** | **33.17** | **0.543** |

> ✅ **结论**：
> - 相比 RoT，在相似压缩比下，**准确率提升 17.33%**

#### 在结构化格式 `GSM8k-Aug` 上的结果（Table 2）
| 方法 | 平均 Pass@1 (%) | #L |
|------|------------------|-----|
| SFT-CoT | 56.98 | 21.35 |
| CoLaR-2 | 48.84 | 10.04 |
| **Ours-RL** | **54.42** | **11.45** |

> ✅ **结论**：
> - 即使在高度结构化的数学表达式中，SLT 仍优于 CoLaR-2（+5.58%）
> - 相比显式 CoT，**推理长度减少 46.3%**，仅损失 **2.56%** 准确率

#### 跨域泛化：Math500（Table 3）
| 方法 | Acc (Llama) | #L (Llama) | Acc (Qwen) | #L (Qwen) |
|------|-------------|------------|------------|-----------|
| SFT-CoT (in-domain) | 14.00 | 205.83 | 53.20 | 248.20 |
| **Ours-RL (gsm-trained)** | **14.60** | **90.02** | **42.60** | **107.77** |

> ✅ **结论**：
> - 从未见过 Math500 数据的情况下，**在 Llama 上反而超过 in-domain SFT-CoT**
> - 推理长度大幅缩短（Llama: ↓56.3%, Qwen: ↓56.7%）

---

### 消融实验结果

#### （1）压缩策略对比（Table 4）
| 方法 | 处理方式 | MultiArith Acc | GSM-Hard Acc |
|------|----------|----------------|--------------|
| SFT-CoT | 不压缩 | 97.20 | 13.40 |
| Compress-Random | 随机压缩任意 2–5 个 token | 98.30 | **10.90** ⬇️ |
| **Compress-SkipNum** | **跳过非数字 token，保留数字** | **97.78** | **13.30** ✅ |

> 🔍 发现：**并非所有推理步骤都适合压缩**，尤其是涉及具体数值的部分。这验证了 SLT 中“选择性”的必要性。

#### （2）置信门控作用（Table 5）
| 设置 | GSM Acc | GSM-Hard Acc |
|------|---------|--------------|
| Ours-SFT（完整联合训练） | 63.84 | 29.72 |
| w/o Joint Gate（无联合训练） | 57.77 | 27.90 |
| **w/o Gate（无门控）** | **6.37** ❌ | **1.67** ❌ |

> 🔍 发现：**confidence gate 至关重要**，缺乏它会导致灾难性失败；联合训练 decoder 与 gate 更能提升可靠性。

#### （3）压缩长度分布（Figure 4）
- 政策倾向于使用较短压缩（c=1~2），避免长跨度压缩
- 经过 RL 优化后，最大长度（c=5）使用频率下降，说明 RL 学会了规避风险

#### （4）门控阈值影响（Figure 5）
- 通过调节 confidence threshold，可在准确率与长度间平滑权衡
- **保守阈值（high threshold）下，SLT 甚至能超越原始 SFT-CoT 的准确率**（见 Appendix B）
  - 原因：压缩“语言噪音”起到了**去噪效果**，防止显式链中出现幻觉传播

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **推理步骤不可统一压缩**：数学等精确任务中存在“精度关键步骤”，必须显式保留。
2. ✅ **选择性压缩优于统一压缩**：SLT 通过动态决策，在效率与准确性之间取得显著更优的平衡。
3. ✅ **latent compression 可作为去噪机制**：合理压缩语言冗余反而能提升最终准确率。
4. ✅ **trajectory-level RL 至关重要**：局部训练不足以建模长期误差传播，RL 成功整合了压缩策略。
5. ✅ **良好的跨域泛化能力**：即使未在目标领域训练，SLT 也能保持竞争力。

---

### 方法的局限性
1. **模型规模受限**：由于训练成本高（需训练辅助模块 + RL），目前仅在 `Llama-3.2-1B` 和 `Qwen3-4B` 上验证，尚未扩展至更大模型（如 70B+）。
2. **推理长度有限**：当前框架适用于中等长度推理，对超长思维链（long-form thinking）的支持尚不充分。
3. **look-ahead 窗口限制**：Feature Decoder 的预测范围有限（默认 K=4），难以捕捉远距离依赖。
4. **单 latent vector 容量瓶颈**：实验表明，单个 latent 向量无法稳定模拟超过 5 个 token 的复杂推理（见 Table 7）。

---

### 未来工作方向
1. **扩展至更大模型和更长推理轨迹**
2. **设计更高效的训练与 RL 优化策略**，降低计算开销
3. **探索多 latent block 压缩机制**，突破单向量容量限制
4. **应用于多模态推理场景**，结合视觉与文本 latent reasoning
5. **研究 human-in-the-loop 的可控压缩接口**，允许用户干预压缩策略

---

> 📌 **一句话总结**：  
> **Selective Latent Thinking (SLT)** 提出了一种“**该显则显，该隐则隐**”的智能推理压缩范式，打破了传统 Latent Reasoning “一刀切”的局限，在多个数学推理基准上实现了**接近显式 CoT 的精度 + 接近隐式方法的效率**，为高效且可靠的 LLM 推理提供了新路径。

</details>

---

### 14. [Polar: Agentic RL on Any Harness at Scale](https://arxiv.org/abs/2605.24220)

**Authors**: Binfeng Xu, Hao Zhang, Shaokun Zhang, Songyang Han, Mingjie Liu, Jian Hu, Shizhe Diao, Zhenghui Jin, Yunheng Zou, Michael Demoret, Jan Kautz, Yi Dong  
**Category**: cs.DC  
**Published**: 2026-05-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.24220v1  

#### Abstract
Reinforcement learning for language agents increasingly depends on custom harnesses that manage long-running context, multi-turn tool use and multi-agent orchestration. However, porting these harnesses into RL environment interfaces remains difficult and often loses important training signals. We br...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Polar: Agentic RL on Any Harness at Scale**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前基于大语言模型（LLM）的 **Agentic RL**（代理强化学习）面临一个核心系统挑战：  
现有的 **agent harness**（代理执行框架，如代码生成、工具调用、多轮交互系统）通常是复杂、异构且独立开发的软件系统，可能使用不同语言、依赖外部工具、甚至为闭源二进制。传统 RL 框架要求将这些 harness “重写”为符合 RL 环境接口（如 Gymnasium）的形式，这一过程：

- 高成本、易出错；
- 可能丢失原始执行路径中的细节（如 token-level 交互）；
- 阻碍了对真实部署环境的端到端训练。

因此，论文提出的关键问题是：  
> **能否在不打开（不修改）agent harness 的情况下对其进行 RL 训练？**

---

### **提出了什么新方法或新思路**

作者提出了 **PoLAR**（**P**roxied **o**bservation for **L**LM **A**gent **R**L），一种全新的 **rollout-as-a-service** 架构，其核心思想是：

> **将 agent harness 视为黑盒，通过监听其与 LLM 推理服务之间的 API 调用流量来构建 RL 训练轨迹（trajectory）。**

#### 主要创新点包括：

1. ✅ **基于 API 代理的 rollout 机制（Proxy-based Rollout）**  
   - 在 agent harness 和 inference server 之间插入一个 **model API proxy**。
   - 该 proxy 透明地转发请求，并记录每个 LLM 调用的完整信息：`prompt`, `sampled tokens`, `logprobs`, `responses` 等。
   - 利用这些原始数据重建 **token-faithful trajectory**，确保训练信号精确对应行为策略采样的 token。

2. ✅ **解耦式异步 rollout 架构（Asynchronous Rollout Staging）**  
   - 将 rollout 流程拆分为多个异步阶段：`INIT`（运行时准备）、`RUN`（执行）、`POSTRUN`（轨迹重建 + 评估）。
   - 支持大规模并行调度，避免长尾任务阻塞 GPU 训练进程。
   - 实现真正的 **rollout-as-a-service**，可被任意 trainer（如 Slime、GRPO）消费。

3. ✅ **高保真轨迹重建（Token-Faithful Trajectory Reconstruction）**  
   - 提供两种策略：
     - `per_request`：每条 API 请求作为一个独立 trace（保守但碎片化）。
     - `prefix_merging`：智能合并具有前缀一致性的连续对话，形成更长、更连贯的训练样本，同时保持 **loss mask** 对非生成 token 的屏蔽，保证训练 fidelity。

4. ✅ **Harness-Agnostic 设计**  
   - 不依赖任何特定 agent 框架内部实现。
   - 支持任意可通过标准 API（OpenAI、Anthropic、Google 等风格）调用 LLM 的 harness。
   - 已集成主流编码 harness：`Codex`, `Claude Code`, `Qwen Code`, `Pi`, `Gemini CLI` 等。

---

### **相比现有方法的优势**

| 特性 | PoLAR | PRoRL AGENT | SkyRL-Agent | Agent Lightning / rLLM |
|------|-------|-------------|-------------|------------------------|
| **Harness 无需修改** | ✅ 黑盒支持 | ❌ 需实现 handler | ❌ 需适配环境接口 | ⚠️ 需接入 SDK/装饰器 |
| **异步 rollout staging** | ✅ 完整支持 | ✅ 类似设计 | ✅ 支持 | ❌ 不支持 |
| **Rollout as Service** | ✅ 异步 API | ✅ | ❌ 内嵌于训练栈 | ❌ |
| **Token fidelity 保障** | ✅ 原生支持 | ✅ | ⚠️ 依赖实现 | ✅ |
| **多 harness 兼容性** | ✅ 广泛支持 | ❌ 有限 | ❌ 专用环境 | ⚠️ 有限 |

> PoLAR 是首个真正实现“**Train Any Agent, Without Opening the Box**”的通用 RL rollout 基础设施。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **SWE-Gym**：用于在线 RL 训练的数据集，包含 293 个软件工程任务（来自 GitHub issue），涵盖代码理解、编辑、测试验证等。
- **SWE-Bench Verified**：用于最终评估的标准 benchmark，共 500 个真实世界 GitHub 问题，评估模型解决实际 bug 的能力（pass@1）。
- **自定义 SFT Corpus**：使用 PoLAR 生成的监督微调数据集，基于 SWE-Gym 的 1,638 个任务实例，由 Qwen3.5-122B-A10B + Pi harness 生成。

---

### **实验设置和评估指标**

#### **在线 RL 实验（GRPO Training）**

- **模型**：Qwen3.5-4B（base checkpoint）
- **算法**：GRPO（Generalized Reward Policy Optimization）
- **Trainer**：Slime（异步 RL 框架）
- **Harnesses**：Codex, Claude Code, Qwen Code, Pi
- **Trajectory Builder**：默认使用 `prefix_merging`
- **评估方式**：
  - 在 SWE-Bench Verified 上进行 **zero-shot evaluation**。
  - 使用对应 harness 执行最终 patch，并运行测试用例。
  - 指标：**pass@1**（是否成功修复所有 FAIL_TO_PASS 测试且不破坏 PASS_TO_PASS 测试）。

#### **离线数据生成实验（Offline SFT Data Generation）**

- **模型**：Qwen3.5-122B-A10B
- **Harness**：pi-coding-agent v0.67.68
- **目标**：生成高质量、多轮交互的 SFT 数据。
- **筛选标准**：仅保留最终 patch 通过 SWE-Bench evaluator 的轨迹。
- **输出格式**：OpenAI-style messages list，含 `tool_calls`、`role`、`content` 等字段。

---

### **基线方法对比**

本文未直接与其他 RL 框架进行端到端性能对比，而是强调 **PoLAR 自身带来的增益**，即：

- 同一模型（Qwen3.5-4B）在不同 harness 下经过 PoLAR 训练后的性能提升。
- 不同 trajectory reconstruction 策略对训练效率的影响（消融实验）。

本质上，PoLAR 是一个 **rollout substrate**（底层支撑），而非替代 trainer 的算法，因此其比较维度在于 **系统兼容性、训练效率、fidelity 保障**。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（SWE-Bench Verified）**

| Harness       | Base Model Pass@1 | PoLAR + GRPO Pass@1 | Gain (pts) |
|---------------|--------------------|----------------------|------------|
| **Codex**     | 3.8%               | **26.4%**            | **+22.6**  |
| **Claude Code** | 29.8%             | **34.6%**            | **+4.8**   |
| **Qwen Code** | 34.6%              | **35.2%**            | **+0.6**   |
| **Pi**        | 34.2%              | **40.4%**            | **+6.2**   |

> ✅ 所有 harness 下均取得正向增益，尤其在 **非原生适配 harness（如 Codex）上提升巨大**，说明 PoLAR 能有效帮助模型适应陌生执行环境。

---

### **与基线方法的对比结果**

虽然没有直接对比其他 rollout 框架的最终 accuracy，但在 **训练效率** 上展示了显著优势：

#### **轨迹重建策略对比（Ablation Study）**

| Strategy         | 更新次数（3 steps） | Wall-clock Time | Rollout GPU Utilization |
|------------------|--------------------|------------------|----------------------------|
| `per_request`    | 1,185              | 189.5 min        | 20.4%                      |
| `prefix_merging` | **218**            | **35.2 min**     | **87.7%**                  |

> 🔥 **5.39x 加速训练，GPU 利用率从 20.4% 提升至 87.7%**！

此外，`per_request` 存在严重的 **reward hacking** 问题（credit assignment noise），而 `prefix_merging` 更自然地保留了会话结构，减少噪声传播。

---

### **离线 SFT 数据生成结果**

| Repository             | Attempts | Accepted | Acceptance Rate |
|------------------------|----------|----------|------------------|
| getmoto/moto           | 343      | 184      | 53.6%            |
| python/mypy            | 257      | 101      | 39.3%            |
| conan-io/conan         | 71       | 27       | 38.0%            |
| pydantic/pydantic      | 81       | 24       | 29.6%            |
| iterative/dvc          | 219      | 45       | 20.5%            |
| pandas-dev/pandas      | 477      | 98       | 19.7%            |
| dask/dask              | 141      | 25       | 17.7%            |
| **Total**              | **1,638**| **504**  | **30.8%**        |

- 平均每 session 包含 **104 条消息**，**51 次 assistant 回复**，最长超过 200 轮。
- 成功构建了一个高质量、长上下文、多工具交互的 **SFT corpus**，已开源发布：
  > [https://huggingface.co/datasets/nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories](https://huggingface.co/datasets/nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories)

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Harness-native RL 是可行且高效的**：  
   无需修改 agent harness 即可进行 RL 训练，极大降低集成成本。

2. ✅ **API proxy 是理想的观测边界**：  
   所有 LLM agent 必须调用 model API，这提供了一个天然、统一、可靠的观测点。

3. ✅ **prefix_merging 显著提升训练效率与稳定性**：  
   相比 per-request，它减少了训练样本数量，提高了 GPU 利用率，并缓解了 reward hacking。

4. ✅ **PoLAR 支持多种下游任务**：  
   不仅可用于 online RL，还可作为 **distributed offline data generation service**，用于 SFT、Preference Modeling、Verifier Training 等。

5. ✅ **在非原生 harness 上收益最大**：  
   如 Qwen 模型在 Codex harness 上从 3.8% → 26.4%，证明 PoLAR 能有效桥接模型与陌生执行环境之间的 gap。

---

### **方法的局限性**

1. ❗ **依赖 API 兼容性**：  
   若 harness 使用私有或加密协议调用模型，则无法被捕获（尽管大多数主流工具已支持 OpenAI-like API）。

2. ❗ **streaming 处理简化**：  
   当前实现将 streaming 请求转为 non-streaming 获取完整响应后再模拟流式返回，可能引入轻微延迟或行为偏差。

3. ❗ **credit assignment 仍具挑战**：  
   当前仅支持 outcome reward 或简单 process reward，复杂任务中仍需更精细的 PRM（Process Reward Model）或 session normalization。

4. ❗ **不处理 agent 内部状态同步问题**：  
   如 context compaction、sub-agent 分支等，虽能正确分割轨迹，但未主动优化其协调机制。

---

### **未来工作方向**

1. 🔄 开发更先进的 **session-aware credit assignment** 机制，支持 process-level reward modeling。
2. 🧩 支持更多 provider API 格式和 streaming 模式，增强兼容性。
3. 📦 探索 **auto-instrumentation** 技术，自动识别和配置未知 harness。
4. 🌐 构建更大规模的 **PoLAR-powered agent training platform**，支持跨组织协作训练。
5. 🤖 将 PoLAR 应用于更复杂的 agentic 场景：GUI 操作、操作系统级代理、多模态任务等。

---

> 💡 **总结一句话**：  
> **PoLAR 重新定义了 agentic RL 的 rollout 范式——不再要求“让 agent 适应 RL”，而是“让 RL 适应 agent”。**  
> 它通过 API 代理实现了真正的 harness-agnostic、scale-out、high-fidelity 强化学习基础设施，为未来大规模代理训练提供了坚实基础。

</details>

---

### 15. [Optimus: Elastic Decoding for Efficient Diffusion LLM Serving](https://arxiv.org/abs/2605.24832)

**Authors**: Chiyue Wei, Cong Guo, Bowen Duan, Junyao Zhang, Haoxuan Shan, Yifei Wang, Yangjie Zhou, Hai "Helen" Li, Danyang Zhuo, Yiran Chen  
**Category**: cs.DC  
**Published**: 2026-05-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.24832v1  

#### Abstract
Large language model (LLM) serving is fundamentally limited by inefficient hardware utilization. Autoregressive (AR) decoding underutilizes GPUs due to its strictly sequential execution, while diffusion LLMs (DLLMs) improve throughput by decoding multiple tokens per iteration. However, fixed block-s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Optimus: Elastic Decoding for Efficient Diffusion LLM Serving》总结**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前 **Large Language Model (LLM)** 服务面临硬件利用率低下的瓶颈，尤其是在 **Autoregressive (AR) decoding** 中，由于其严格的逐 token 生成机制，导致 GPU 资源严重浪费。虽然 **Diffusion LLM (DLLM)** 通过一次解码多个 token（block-wise decoding）提升了吞吐量，但其固定 block size 的设计存在“负载敏感”问题：
- 在低负载时，大 block 可以利用空闲资源；
- 在高负载时，大 block 会过早饱和 GPU，并引入大量冗余计算（token utilization 低），反而性能下降。

因此，**单一固定的 decoding granularity 无法在动态负载下保持高效**。

---

### **提出的新方法与思路**
论文提出了 **Optimus**，一种支持 **弹性解码（elastic decoding）** 的 LLM 服务系统，核心思想是将 **decoding granularity 视为运行时可调的控制变量**，动态适应负载变化。

#### **两大核心技术：**
1. **Chunked Decoding（分块解码）**
   - 在不重新训练模型的前提下，将一个大的 diffusion block 分解为更细粒度的执行单元（chunks）。
   - 支持运行时灵活调整 chunk size，实现从 block-level 到 sub-block-level 的精细控制。
   - 结合 **prefix caching** 和 **suffix chunking** 技术减少冗余计算，同时保证正确性。

2. **Saturation-aware Elastic Scheduling（饱和感知弹性调度）**
   - 构建闭环控制系统，实时监测 GPU 利用率和 token utilization。
   - 动态选择最优 chunk size，使系统始终运行在 **GPU 利用率与 token 效率之间的 Pareto 前沿** 上。
   - 采用 **system-algorithm co-modeling**：离线建模系统延迟，线上估计 token 提交数，联合优化调度决策。

---

### **相比现有方法的优势**
| 维度 | AR Decoding | Fixed-block DLLM | Optimus |
|------|-------------|------------------|---------|
| 并行性 | 仅依赖 inter-request 并行（靠 CB） | 引入 intra-request 并行 | 同时利用两者，且可调节 |
| 负载适应性 | 低负载下严重 underutilized | 高负载下易 overload | 动态适配，稳定高效 |
| 硬件效率 | 差（GEMV 占主导） | 中等（高并行但冗余多） | 高（平衡利用率与效率） |
| 实现成本 | 无需改模型 | 需训练多个 block-size 模型 | 单一模型 + 系统层改造 |

> ✅ **优势总结**：Optimus 实现了“一个模型，多种粒度”，避免了多模型部署开销，同时显著提升端到端服务容量。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖多种典型 LLM 应用场景，包括对话、编程、数学推理等：

| 数据集 | 类型 | 特点 |
|--------|------|------|
| **ShareGPT** | 对话 | 真实用户对话轨迹，用于在线 serving 测试 |
| **LMSYS-Chat-1M** | 大规模聊天 | 来自真实交互，评估通用性 |
| **LongBench** | 长文本理解与生成 | 强调上下文长度 |
| **GSM8K** | 数学推理 | 多步逻辑推导 |
| **HumanEval / MBPP** | 代码生成 | 编程能力测试 |
| **IFEval** | 指令遵循 | 评估语义一致性 |

---

### **实验设置与评估指标**

#### **硬件环境**
- 使用 **NVIDIA A100-SXM4-80GB GPU**，通过 NVLink 互联。
- 支持单卡与多卡（tensor parallelism）配置。

#### **模型**
- **DLLM 模型族**：
  - `SDAR-8B`（dense）
  - `LLaDA2.0-16B`（MoE）
- **对应 AR 基线模型**：
  - `Qwen3-8B`, `Ling2.0-16B`

所有模型使用 FP16 推理。

#### **评估指标**
| 指标 | 定义 | 目标 |
|------|------|------|
| **Throughput (tokens/sec)** | 输出 token 数 / 时间 | 越高越好 |
| **P90 Time-per-Output-Token (TPOT)** | 90% 的 token 生成延迟 | 越低越好 |
| **SLO-compliant Request Rate** | 满足延迟约束的最大请求速率 | 越高越好 |
| **End-to-end Serving Capacity** | 在给定 SLO 下能处理的请求数 | 越高越好 |
| **Model Accuracy** | 在标准 benchmark 上的准确率 | 不应显著下降 |

#### **SLO 设置**
- **对话类任务（ShareGPT, LMSYS）**：TPOT ≤ 50ms
- **长文本任务（LongBench）**：TPOT ≤ 100ms

---

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **LMDeploy-AR** | 基于 LMDeploy 的 AR 解码，启用 PagedAttention 和 continuous batching |
| **LMDeploy-BD32** | 固定 block size=32 的 diffusion 解码 |
| **SGLang-BD32** | 当前最先进的 DLLM 推理框架，支持 Radix Attention 和 block-level batching |

> ⚠️ 注意：Optimus 基于 LMDeploy 实现，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 指标 | Optimus vs AR | Optimus vs BD32 | Optimus vs SGLang |
|------|---------------|------------------|--------------------|
| **最大吞吐提升** | **6.1×** | **4.3×** | — |
| **平均吞吐提升** | 2.1× | 1.3× | 2.55×（最高达 9.69×） |
| **SLO 下服务容量提升** | 最高 **3.5×** | 最高 **2.0×** | 最高 **11.8×**（部分场景达 50.7×） |
| **几何平均吞吐提升** | 2.07× | 1.31× | 2.55× |

---

### **详细对比结果**

#### **(1) 吞吐量随 batch size 变化（Figure 8 & 9）**
- 在低 batch size（如 bs=1）时，Optimus 使用大 chunk（如 32）充分利用 GPU，吞吐达 AR 的 **5.59×**。
- 在高 batch size（如 bs=256）时，自动切换至小 chunk（如 8~16），避免冗余计算，仍优于固定 block 方法。
- 在所有数据集上均形成 **Pareto frontier**，说明其自适应策略有效。

#### **(2) 端到端在线服务性能（Figure 10）**
- 在 ShareGPT 上，满足 50ms TPOT SLO 时：
  - 请求处理能力比 SGLang-BD32 提升 **10.2×**
  - 比 LMDeploy-BD32 提升 **1.95×**
- 在 LMSYS 和 LongBench 上同样表现领先。

#### **(3) 运行时调度行为分析（Figure 11）**
- **低负载（0.5 req/s）**：batch size 小（均值 1.8），chunk size 几乎全为 32 → 充分利用空闲 GPU。
- **高负载（4.9 req/s）**：batch size 显著增大（均值 25.0），chunk size 自动降至平均 20.8，最低至 6 → 控制冗余，维持效率。

> 📌 表明 Optimus 能 **动态追踪最优 operating point**。

---

### **消融实验结果（Ablation Study, Figure 13）**

| 配置 | SLO 下请求率（req/s） | 提升倍数 |
|------|------------------------|----------|
| BD32（baseline） | 2.60 | 1.0× |
| Chunk-8（固定） | 5.54 | 2.13× |
| Optimus（弹性调度） | 5.06 | 1.95× |

- **Chunked Decoding 本身即可大幅提升性能**（因减少冗余）。
- **Elastic Scheduling 进一步增强鲁棒性**，虽略低于最佳固定 chunk，但无需手动调参，在动态负载中更具实用性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Fixed-granularity decoding 是次优的**  
   无论是 AR 还是固定 block DLLM，都无法在不同负载下持续高效。

2. **Decoding granularity 应作为运行时控制变量**  
   通过将 block 分解为 chunks，可在不重训模型的情况下实现细粒度控制。

3. **最优策略位于 GPU Utilization 与 Token Utilization 的权衡前沿**  
   Optimus 的闭环比调度机制能有效逼近该前沿。

4. **系统与算法需协同设计（co-design）**  
   单纯改进算法（如更大 block）或系统（如 continuous batching）都不够，必须联合优化。

5. **模型精度基本不受影响**  
   在 block 内 streaming（in-block）模式下，accuracy 与 BD32 相当，甚至略有提升；out-block streaming 有轻微下降但可控。

---

### **方法的局限性**
- **依赖 diffusion LLM 架构**：目前仅适用于 DLLM，不能直接用于传统 AR 模型。
- **chunk size 搜索空间有限**：当前仅支持预设几个 chunk sizes，未实现完全连续调节。
- **out-block streaming 影响准确性**：跨 block streaming 可能破坏训练时假设的依赖关系。
- **冷启动开销**：需要 warm-up 阶段进行 profiling，可能影响初始响应速度。

---

### **未来工作方向**
1. **支持更多 DLLM 架构**：扩展至其他 diffusion 范式（如 flow matching）。
2. **学习型调度器**：用 RL 或 learning-based controller 替代启发式调度。
3. **编译器级集成**：将 chunked decoding 编译进推理图，进一步降低开销。
4. **训练-推理协同优化**：设计对 chunking 更友好的 diffusion 模型结构。
5. **多目标调度**：结合 energy、cost、SLA 等多维度目标进行弹性控制。

---

> 🔚 **总结一句话**：  
> **Optimus 通过 chunked decoding + saturation-aware scheduling，首次实现了 diffusion LLM 的弹性解码，在动态负载下持续逼近理论最优性能，显著提升了 LLM 服务效率。**

</details>

---

### 16. [Fourier Feature Pyramids for Physics-Informed Neural Networks](https://arxiv.org/abs/2605.24278)

**Authors**: Brandon Zhao, Yixuan Wang, Jonathan T. Barron, Katherine L. Bouman, Dor Verbin, Pratul P. Srinivasan  
**Category**: cs.LG  
**Published**: 2026-05-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.24278v1  

#### Abstract
We present an improved neural field architecture for solving partial differential equations (PDEs). Current physics-informed neural networks (PINNs) provide a flexible framework for solving PDEs, but they struggle to achieve highly accurate solutions and require computation that scales poorly with p...

---

### 17. [CoRe-Code: Collaborative Reinforcement Learning for Code Generation](https://arxiv.org/abs/2605.24812)

**Authors**: Zhihao Dou, Qinjian Zhao, Zhongwei Wan, Xiaoyu Xia, Sumon Biswas  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.24812v1  

#### Abstract
Large language models (LLMs) have achieved strong performance in code generation, but most methods rely on autoregressive decoding without global planning, often leading to locally coherent yet globally suboptimal solutions (e.g., failing test cases or inefficient complexity). While recent approache...

---

### 18. [NITP: Next Implicit Token Prediction for LLM Pre-training](https://arxiv.org/abs/2605.24956)

**Authors**: Xiangdong Zhang, Debing Zhang, Shaofeng Zhang, Xiaohan Qin, Yu Cheng, Junchi Yan  
**Category**: cs.CL  
**Published**: 2026-05-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.24956v1  

#### Abstract
Standard next-token prediction (NTP) supervises language models solely through discrete labels in the output logit space. We argue that this sparse one-hot supervision leaves the latent representation space under-constrained, allowing hidden states to drift into degenerate and anisotropic configurat...

---

### 19. [Reinforcement Learning from Denoising Feedback](https://arxiv.org/abs/2605.25638)

**Authors**: Qi He, Huan Chen, Ya Guo, Huijia Zhu, Yi R. Fung, Baojian Zhou  
**Category**: cs.CL  
**Published**: 2026-05-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.25638v1  

#### Abstract
Policy loss estimation remains a fundamental and long-standing challenge in reinforcement learning (RL) for diffusion language models (dLLMs). We introduce Reinforcement Learning from Denoising Feedback (RLDF), a novel training paradigm that leverages feedback obtained from rollout and training proc...

---

### 20. [PrivFusion: A Privacy-preserving Multi-Agent Framework for Harmonizing Distributed Datasets](https://arxiv.org/abs/2605.24249)

**Authors**: Anisa Halimi, Liubov Nedoshivina, Kieran Fraser, Stefano Braghin  
**Category**: cs.LG  
**Published**: 2026-05-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.24249v1  

#### Abstract
The growing availability of clinical data has increased the use of machine learning, yet centralized data aggregation is often infeasible for sensitive health information. Federated Learning (FL) offers a distributed alternative, but its adoption is limited by substantial heterogeneity across instit...

---

### 21. [Label-NTK Alignments and A Tighter Convergence Bound in the NTK Regime](https://arxiv.org/abs/2605.25275)

**Authors**: Ruchirinkil Marreddy, Chaoyue Liu  
**Category**: cs.LG  
**Published**: 2026-05-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.25275v1  

#### Abstract
The Neural Tangent Kernel (NTK) framework explains optimization in over-parameterized neural networks via approximately linearized dynamics, yielding exponential convergence guarantees. However, existing results are often overly pessimistic and do not match the fast training in practice, as they dep...

---

### 22. [Accelerated Dynamic Importance Weighting with Versatile Divergence-Minimizing Estimators](https://arxiv.org/abs/2605.25499)

**Authors**: Tongtong Fang, Nan Lu, Gang Niu, Kenji Fukumizu, Masashi Sugiyama  
**Category**: cs.LG  
**Published**: 2026-05-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.25499v1  

#### Abstract
Importance weighting (IW) is a golden solver for joint distribution shift, where the joint distributions differ between the training and test data. To solve this problem, IW estimates test-to-training density ratios as importance weights and reweights the training losses accordingly. Recent advances...

---

### 23. [Palette: A Modular, Controllable, and Efficient Framework for On-demand Authorized Safety Alignment Relaxation in LLMs](https://arxiv.org/abs/2605.24154)

**Authors**: Qitao Tan, Xiaoying Song, Arman Akbari, Arash Akbari, Yanzhi Wang, Xiaoming Zhai, Lingzi Hong, Zhen Xiang, Jin Lu, Geng Yuan  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.24154v1  

#### Abstract
Current safety alignment of foundation models largely follows a \emph{one-size-fits-all} paradigm, applying the same refusal policy across users and contexts. As a result, models may refuse requests that are unsafe for general users but legitimate for authorized professionals, limiting helpfulness i...

---

### 24. [JT-SAFE-V2: Safety-by-Design Foundation Model with World-Context Data](https://arxiv.org/abs/2605.24414)

**Authors**: Junlan Feng, Fanyu Meng, Chong Long, Pengyu Cong, Duqing Wang, Yan Zheng, Yuyao Zhang, Xuanchang Gao, Ye Yuan, Yunfei Ma, Zhijie Ren, Fan Yang, Na Wu, Di Jin, Chao Deng  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.24414v1  

#### Abstract
We introduce JT-Safe-V2, a large language model designed to advance the safety and trustworthiness of foundation models, extending our previous JT-Safe model toward a more comprehensive safety-by-design paradigm. JT-Safe-V2 emphasizes the joint optimization of general intelligence and safety-by-desi...

---

### 25. [Kavier: Exploring Performance, Sustainability, and Efficiency of LLM Ecosystems under Inference through Cache-Aware Discrete-Event Simulation](https://arxiv.org/abs/2605.25247)

**Authors**: Radu Nicolae, Alexandru Iosup, Animesh Trivedi, Jesse Donkervliet  
**Category**: cs.DC  
**Published**: 2026-05-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.25247v1  

#### Abstract
Large Language Models (LLMs) are widely used by our increasingly digitalized society, but raise sustainability, performance, and financial concerns, especially as inference workloads grow. To improve the design and operation of LLM ecosystems, we envision simulators and simulation-based digital twin...

---

### 26. [A Unified Python Framework for Direct PPO-based Control of AHUs with Economizer Logic and CO2-Constrained Ventilation](https://arxiv.org/abs/2605.24406)

**Authors**: Erfan Haghighat Damavandi, Davide Papurello, Mahdi Alibeigi, Armin Keshavarz, Simone Canevarolo, Marco Condo  
**Category**: cs.LG  
**Published**: 2026-05-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.24406v1  

#### Abstract
Optimizing HVAC (Heating, Ventilation and Air Conditioning) can enhance a building's energy efficiency while providing comfort levels for its occupants. Using conventional control systems to maintain HVAC functions is often difficult because of the nonlinear characteristics of a building envelope as...

---

### 27. [Reason--Imagine--Act: Closed-Loop LLM Decision Making with World Models for Autonomous Driving](https://arxiv.org/abs/2605.24004)

**Authors**: Zhengqi Sun, Yiwen Sun, Boxuan Liu, Tailai Chen, Tianxu Guo, Jiabin Liu  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.24004v1  

#### Abstract
Large language models (LLMs) are promising for autonomous driving, but semantics-only decision policies can yield physically unsafe behavior in dynamic traffic. Existing methods either perform online language reasoning without explicit dynamics verification or use world models mainly in offline pipe...

---

### 28. [EPPC-OASIS: Ontology-Aware Adaptation and Structured Inference Refinement for Electronic Patient-Provider Communication Mining in Secure Messages](https://arxiv.org/abs/2605.24172)

**Authors**: Samah Fodeh, Sreeraj Ramachandran, Elyas Irankhah, Muhammad Arif, Afshan Khan, Ganesh Puthiaraju, Linhai Ma, Srivani Talakokkul, Jordan Alpert, Sarah Schellhorn  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.24172v1  

#### Abstract
Secure patient-provider messages contain clinically important communication behaviors that are difficult to characterize manually at scale. The Electronic Patient-Provider Communication (EPPC) framework provides an ontology for coding these behaviors, but automated extraction remains challenging bec...

---

### 29. [When Does Multi-Agent RL Improve LLM Workflows? Workflow, Scale, and Policy-Sharing Tradeoffs](https://arxiv.org/abs/2605.24202)

**Authors**: Yifan Zeng, Yiran Wu, Yaolun Zhang, Wentian Zhao, Kun Wan, Qingyun Wu, Huazheng Wang  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.24202v1  

#### Abstract
Multi-agent LLM workflows route inference through specialized roles to lift end-task accuracy, but jointly training those roles with reinforcement learning is unstable in ways that are poorly understood. We study when end-to-end RL training of multi-agent LLM workflows improves over their base model...

---

### 30. [Learning to Reason Efficiently with A* Post-Training](https://arxiv.org/abs/2605.24597)

**Authors**: Andreas Opedal, Francesco Ignazio Re, Abulhair Saparov, Mrinmaya Sachan, Bernhard Sch\"olkopf, Ryan Cotterell  
**Category**: cs.AI  
**Published**: 2026-05-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.24597v1  

#### Abstract
Many applications of large language models (LLMs) require deductive reasoning, yet models frequently produce incorrect or redundant inference steps. We frame natural language inference as a search problem where the final answer is the valid proof itself, requiring a reasoning procedure in which inte...

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

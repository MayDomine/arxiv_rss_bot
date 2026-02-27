# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-27 06:35:45 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference](https://arxiv.org/abs/2602.22868)

**Authors**: Yushi Ye, Feng Hong, Huangjie Zheng, Xu Chen, Zhiyong Chen, Yanfeng Wang, Jiangchao Yao  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2602.22868v1  

#### Abstract
Diffusion Large Language Models (DLLMs) promise fast non-autoregressive inference but suffer a severe quality-speed trade-off in parallel decoding. This stems from the ''combinatorial contradiction'' phenomenon, where parallel tokens form semantically inconsistent combinations. We address this by in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对 **Diffusion Large Language Models (DLLMs)** 在并行解码时面临的严重 **质量-速度权衡**（quality-speed trade-off）问题。其根本原因被归结为 **“组合矛盾”**（combinatorial contradiction），即在同一个解码步骤中，并行生成的多个 token 可能语义上相互冲突（例如，“high pair” 而非 “full house”），导致输出不一致。

传统方法因采用纯离散解码（discrete decoding），无法在 token 之间建立有效的依赖关系，从而加剧了这一问题。

---

### 🚀 提出的新方法：ReMix（Rejection Mixing）
作者提出了一种名为 **ReMix** 的训练免费（training-free）解码框架，通过引入 **连续混合状态**（Continuous Mixing State, State C）作为从 `[MASK]` 状态（State M）到最终语义 token 状态（State T）之间的中间状态，实现以下机制：

- **Continuous Mixing State**：允许未确定的 token 在连续嵌入空间中进行迭代优化，保留跨位置的语义依赖。
- **Mixing Rule (M → C ↔ C)**：将当前模型输出的概率分布与 `[MASK]` 嵌入混合，形成一个可更新的连续表示。
- **Rejection Rule (C → M)**：当某位置的输出分布在连续两步间变化过大（用 JS 散度衡量），则将其重置回 `[MASK]` 状态，防止错误传播。

这种方法实现了在离散扩散解码过程中对语义的 **连续空间精炼**（continuous-space refinement），有效缓解了组合矛盾。

---

### 🔍 相比现有方法的优势
| 特性 | ReMix | 其他方法（如 WINO, APD） |
|------|-------|--------------------------|
| 是否需要额外训练 | ❌ 否（training-free） | ✅ 是（需额外模块或微调） |
| 是否引入额外验证模块 | ❌ 否 | ✅ 是（如 WINO 的验证块） |
| 是否破坏双向注意力 | ❌ 否 | ✅ 是（如 APD 引入左到右顺序） |
| 加速比 | 2–8× | 通常 < 4× |
| 输出质量 | 不降反升 | 多数情况下有损失 |

> ReMix 的核心优势在于：**无需修改模型结构或重新训练，即可显著加速推理并提升生成质量**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### 语言任务（基于 LLaDA-8B-Instruct）：
- **数学推理**：GSM8K, MATH-500
- **代码生成**：HumanEval, MBPP
- **逻辑推理**：Countdown, Sudoku
- **常识推理**：ARC-E, ARC-C

#### 多模态任务（基于 MMaDA-8B-MixCoT）：
- **图像描述**：Flickr30k-lite
- **图表理解**：AI2D-lite
- **数学视觉推理**：MathVision-mini, MathVista-mini
- **多学科理解**：MMMU-val, ScienceQA-IMG

---

### ⚙️ 实验设置与评估指标

| 项目 | 设置说明 |
|------|--------|
| **评估模式** | 零样本（zero-shot），Sudoku 使用 4-shot |
| **评估指标** | - 除 Flickr30k 外均使用 **Accuracy**<br>- Flickr30k 使用 **CIDEr**<br>- 多模态任务使用 GPT-4o-mini 进行答案评判 |
| **生成长度** | 默认 256，部分实验测试 128 和 512 |
| **块长度**（Block Length） | 默认 128，部分实验测试 32、64 |
| **硬件平台** | 8×NVIDIA A100（语言） / 8×RTX 3090（多模态） |
| **超参数** | `T_conf=0.8`, `β∈{0.4,0.5,0.6}`, `T_rej∈[0.1,0.4]` |

---

### 🔁 基线方法对比
- **LLaDA / MMaDA**：标准的 DLLM 解码流程（256 步）
- **Fast-dLLM**：基于 KV Cache 和并行解码的加速方法
- **WINO**：可撤销解码策略
- **Learn2PD**：学习预测 token 收敛的轻量过滤器

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（来自 Table 1 & 2）

| 指标 | 表现 |
|------|------|
| **推理加速比** | **2.4× – 8.5×**（平均约 3–5×） |
| **解码步数减少** | 从 256 步降至 **30–85 步** |
| **延迟降低** | 减少 **7–13 秒**，端到端提速明显 |
| **准确率变化** | **全部任务均有提升**，最高达 +14.05%（ARC-C） |

---

### 🔢 典型任务表现示例

#### ✅ 语言领域（GSM8K）
| 方法 | Accuracy | 解码步数 | 加速比 |
|------|----------|-----------|--------|
| LLaDA | 73.01% | 256 | 1.00× |
| ReMix | **75.66%** (+2.65) | **51.55** | **4.97×** |

> **不仅快了近 5 倍，还提升了准确率！**

#### ✅ 多模态领域（Flickr30k-lite）
| 方法 | CIDEr | 解码步数 | 加速比 |
|------|--------|-----------|--------|
| MMaDA | 57.52 | 256 | 1.00× |
| ReMix | **59.59** (+2.07) | **30.13** | **8.50×** |

> 在图像描述任务中实现 **8.5 倍加速 + 质量提升**。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）不同生成长度的影响（Table 3）
- 生成长度越长，ReMix 的优势越明显。
- 在 **512 长度下**，GSM8K 达到 **6.46× 加速 + 2.95% 提升**。

#### （2）是否启用 Mixing 模块（Figure 5）
- 若关闭 Mixing（即 β=0），退化为普通 confidence-aware 解码。
- 结果显示：**启用 Mixing 显著提升准确率**，证明连续状态的有效性。

#### （3）超参数敏感性分析（Figure 6）
- **Mixing ratio β**：适中值（~0.5）最佳；过高导致不稳定。
- **Rejection threshold T_rej**：适度放宽有助于稳定，但过大会引入噪声。
- 总体对参数鲁棒性强，在多数配置下均优于基线。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **“组合矛盾”是限制 DLLM 并行解码性能的根本瓶颈**，源于离散采样缺乏跨位置协调。
2. **引入连续中间状态（State C）可有效缓解该问题**，使 token 在离散化前完成语义对齐。
3. **ReMix 是一种简单、高效且无需训练的解码增强方法**，在多种任务上实现了 **“既快又好”** 的突破。
4. **拒绝机制（Rejection Rule）至关重要**，它防止了低置信度更新带来的误差累积，保障了解码稳定性。

---

### ⚠️ 方法的局限性
- 当前方法依赖于模型输出的概率分布进行连续嵌入构造，若原始 DLLM 本身建模能力弱，则提升有限。
- 对极端长序列（>1024）的表现尚未验证。
- 虽然计算开销小（仅增加 ~9% 运行时间），但在资源极度受限场景仍需权衡。

---

### 🔮 未来工作方向
1. 将 ReMix 扩展至 **文本到图像生成等其他扩散模型领域**。
2. 探索更复杂的 **连续状态更新机制**（如引入轻量 MLP 或记忆单元）。
3. 结合 **自回归蒸馏思想**，进一步缩小与 AR 模型在复杂推理上的差距。
4. 研究如何动态调整 `β` 和 `T_rej` 以实现 **自适应解码控制**。

---

## 💡 总结一句话
> **ReMix 通过引入“连续混合状态 + 拒绝机制”，首次实现了 DLLM 在不解耦双向注意力、不增加训练成本的前提下，达成 2–8 倍加速且质量不降反升，为高效大模型推理提供了全新范式。**

</details>

---

### 2. [CCCL: Node-Spanning GPU Collectives with CXL Memory Pooling](https://arxiv.org/abs/2602.22457)

**Authors**: Dong Xu (UC Merced), Han Meng (UC Merced), Xinyu Chen (Zhejinag University), Dengcheng Zhu (Bytedance and), Wei Tang (Bytedance and), Fei Liu (Bytedance and), Liguang Xie (Bytedance and), Wu Xiang (Bytedance and), Rui Shi (Bytedance and), Yue Li (Bytedance and), Henry Hu (Bytedance and), Hui Zhang (Bytedance and), Jianping Jiang (Xconn-tech), Dong Li (UC Merced)  
**Category**: cs.DC  
**Published**: 2026-02-27  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.22457v1  

#### Abstract
Large language models (LLMs) training or inference across multiple nodes introduces significant pressure on GPU memory and interconnect bandwidth. The Compute Express Link (CXL) shared memory pool offers a scalable solution by enabling memory sharing across nodes, reducing over-provisioning and impr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CCCL: Node-Spanning GPU Collectives with CXL Memory Pooling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大规模 **Large Language Models (LLMs)** 的训练和推理中，跨节点的 **GPU collective communication** 对内存带宽和互连网络提出了极高要求。传统基于 **RDMA + InfiniBand/NVLink** 的通信方式存在以下问题：
- 高昂的硬件成本（如 200 Gbps InfiniBand 交换机价格昂贵）
- 软件栈复杂（需协调 NCCL、MPI 和框架后端）
- 存在通信瓶颈，限制了扩展性

此外，虽然 **CXL shared memory pool** 已被用于 KV Cache 存储等场景（如 Beluga、TraCT），但尚未被系统性地用于实现高效的 **GPU 跨节点集体通信**。

---

### 🔧 提出的新方法与创新思路
本文提出 **CCCL** —— 一个构建于 **CXL shared memory pool** 上的新型 **collective communication library**，其核心创新包括：

#### （1）首次将 CXL 内存池用于 GPU 跨节点集体通信
> 这是**首个利用 CXL 共享内存池执行跨节点 GPU collective operations** 的工作，开辟了一种全新的高性能通信范式。

#### （2）设计了三项关键技术应对挑战
| 挑战 | CCCL 的解决方案 |
|------|----------------|
| **缺乏细粒度交错访问支持**（No cache-line interleaving across CXL devices） | 提出**软件层 interleaving 技术**，通过公式化地址映射将数据轮询分布到多个 CXL 设备上，最大化带宽利用率 |
| **读写依赖导致串行化**（Sequential write-read dependency） | 引入**细粒度数据分块 + 异步重叠机制**，允许不同 rank 的写入与读取操作并发进行，提升并行性 |
| **跨节点同步困难**（Limited cross-node synchronization） | 设计轻量级 **doorbell 机制**，基于内存池中的预分配信号量实现 producer-consumer 同步，避免高开销元数据管理 |

#### （3）简化通信模型
直接使用 **load/store 语义** 替代复杂的 RDMA 请求队列和 CPU-GPU 协调流程，显著降低软件复杂性和控制平面开销。

---

### 🆚 相比现有方法的优势
| 维度 | 传统 RDMA-based NCCL | CCCL（CXL-based） |
|------|------------------------|--------------------|
| **通信语义** | 消息传递（message-passing） | 共享内存（load/store） |
| **硬件依赖** | 高速网络（InfiniBand/NVLink） | CXL switch + 内存池 |
| **成本** | 高（$16K/switch） | 低（$5.8K/switch） |
| **编程模型** | 复杂（需管理 work request） | 简单（类似本地内存访问） |
| **可扩展性潜力** | 受限于网络拓扑 | 更适合小规模高密度集群共享内存池 |

---

## 2. 核心实验方法和设置

### 🧪 实验平台配置
- **硬件环境**：
  - **3 个节点**，每节点配备：
    - Intel Xeon 6960P CPU（72核）
    - 256 GB DRAM
    - NVIDIA H100 GPU（80 GB HBM）
    - PCIe Gen5×16 连接
  - 所有节点连接至 **TITAN-II CXL switch**
  - 后端内存池由 **6 块 Micron CZ120 CXL Type-3 内存卡**组成（共 768 GB）
- **CXL 架构细节**：
  - 使用 **DevDAX** 模式暴露内存池
  - 内存区域设为 **non-cacheable** 并禁用 DDIO
  - 总带宽可达 2 TB/s，延迟约 658 ns（相比本地 DRAM 的 214 ns）

### 💾 数据集与负载
- **微基准测试**：使用 `nccl-tests v2.17.8` 测试标准 collective primitives
- **真实应用测试**：在 **Llama-3-8B 模型** 上进行 FSDP 分布式训练，数据集为 **Wikipedia**

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **性能** | End-to-end latency、bandwidth (GB/s)、speedup |
| **效率** | 资源利用率、通信-计算重叠程度 |
| **成本** | 硬件采购成本对比（CXL vs InfiniBand） |
| **可扩展性** | 节点数从 3 → 6 → 12 的性能变化趋势 |

### ⚖️ 基线方法对比
- **主基线**：基于 **200 Gbps InfiniBand 的 RDMA 实现**（NCCL 默认路径）
- **内部对照版本**（消融实验）：
  1. **CCCL-Naive**：顺序分配内存，无 interleaving 或异步优化
  2. **CCCL-Aggregate**：粗粒度聚合带宽，无细粒度分块与重叠
  3. **CCCL-ALL**：完整功能版（含所有优化）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（平均加速比 vs InfiniBand）

| Collective Operation | 加速比 (Speedup) |
|-----------------------|------------------|
| **AllGather**         | 1.34×            |
| **Broadcast**         | 1.84×            |
| **Gather**            | 1.94×            |
| **Scatter**           | 1.04×            |
| **AllReduce**         | 1.50×            |
| **ReduceScatter**     | 1.43×            |
| **Reduce**            | 1.70×            |
| **AlltoAll**          | 1.53×            |

> 注：以上为不同消息大小下的**平均性能提升**，最大提速可达 **8×**（如 Gather 场景下相对 CCCL-Naive）

---

### 🔁 消融实验结果（vs 不同 CCCL 版本）

| 对比项 | 性能提升（相对 CCCL-Naive） | 说明 |
|--------|----------------------------|------|
| **CCCL-Aggregate vs Naive** | 0.7×–2.9× 降低延迟 | 显示带宽聚合有效 |
| **CCCL-ALL vs Naive** | 最高达 **8×** 速度提升 | 表明细粒度分块 + interleaving + doorbell 协同增效 |
| **CCCL-ALL vs Aggregate** | 额外获得 1.16×–4.2× 提升 | 证明异步重叠机制至关重要 |

例如：
- 在 **Gather** 中，CCCL-ALL 达到 **4.2×–8.0×** 超越 Naive 版本
- 在 **AllReduce** 中，尽管 Ring-based RDMA 可复用中间结果，CCCL 仍能达到 **1.05×** 平均性能（大消息时持平甚至略优）

---

### 💰 成本效益分析
- **硬件成本节省 2.75×**：
  - InfiniBand switch（200Gbps）单价：**$16,000**
  - CXL switch（TITAN-II）单价：**$5,800**
- 在相同性能下，CXL 方案更具经济优势，尤其适用于云厂商部署高密度训练集群

---

### 🔭 可扩展性评估（模拟扩展至 12 节点）
| Primitive | 3→6 节点 | 3→12 节点 |
|----------|---------|----------|
| **AllReduce** | 延迟 ↑ 2.1×–3.0× | 延迟 ↑ 8.7×–12.2× |
| **Broadcast** | 延迟 ↑ ~1.3× | 延迟 ↑ ~2.5× |
| **AlltoAll** | 延迟 ↑ ~1.5× | 延迟 ↑ ~3.6× |

> 发现：随着节点增加，由于仅 6 个 CXL device 支持，出现设备争用加剧现象，表明当前架构更适合 **small-to-medium scale clusters**（如 3–6 节点 POD）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **CXL shared memory pool 完全可用于高效 GPU collective communication**，且性能优于传统 RDMA 方案。
2. **CCCL 通过软件层 interleaving + doorbell 同步 + 异步重叠**，成功克服了 CXL 缺乏硬件交错和弱同步能力的缺陷。
3. **在典型 LLM 训练场景（FSDP）中，CCCL 实现 1.11× 端到端加速**，同时硬件成本降低 **2.75×**，具备极强实用价值。
4. **CXL 更适合“内存中心化”通信模式**，特别利于 AllGather、Gather、Broadcast 等需要大量数据发布的操作。

---

### ⚠️ 局限性
1. **延迟高于本地 DRAM**（658ns vs 214ns），对极低延迟敏感任务不友好
2. **受限于 CXL device 数量和带宽调度策略**，在大规模节点扩展时易产生热点
3. **当前 CXL 生态尚不成熟**，缺乏操作系统原生 NUMA 支持，依赖 DAX 手动管理
4. **GPU DMA engine 限制**：单向只有一个引擎，无法完全压满 PCIe Gen5×16 链路

---

### 🔮 未来工作方向
1. **结合 CXL.cache 与 CXL.mem 实现混合一致性协议**，进一步减少同步开销
2. **开发更智能的数据放置算法**，动态感知负载均衡与热点迁移
3. **探索 CXL + NVLink hybrid topology**，兼顾低延迟与高带宽需求
4. **推动 OS 层支持多主机 CXL NUMA 模型**，实现透明内存访问
5. **扩展至更多 accelerator 类型**（如 TPU、IPU）以验证通用性

---

## 总结
> **CCCL 开辟了“以内存为中心”的 GPU 集体通信新路径**。它不再依赖昂贵的专用网络，而是将 CXL shared memory pool 视为天然的互联媒介，实现了更高性能、更低延迟、更低成本的跨节点协作。这标志着从 “network-centric” 到 “memory-centric” AI 基础设施的重要演进方向。

</details>

---

### 3. [RLHFless: Serverless Computing for Efficient RLHF](https://arxiv.org/abs/2602.22718)

**Authors**: Rui Wei, Hanfei Yu, Shubham Jain, Yogarajan Sivakumar, Devesh Tiwari, Jian Li, Seung-Jong Park, Hao Wang  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.22718v1  

#### Abstract
Reinforcement Learning from Human Feedback (RLHF) has been widely applied to Large Language Model (LLM) post-training to align model outputs with human preferences. Recent models, such as DeepSeek-R1, have also shown RLHF's potential to improve LLM reasoning on complex tasks. In RL, inference and tr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RLHFless: Serverless Computing for Efficient RLHF

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

Reinforcement Learning from Human Feedback (RLHF) 是当前大语言模型（LLM）对齐人类偏好的核心技术，广泛应用于如 DeepSeek-R1 等先进模型中。然而，RLHF 训练存在以下关键效率瓶颈：

- **资源动态需求**：RLHF 的推理（sampling）与训练（learning）阶段交替进行，导致资源需求高度动态，传统静态分配的 **serverful 架构** 难以高效应对。
- **组件间空闲时间**：在同步 RLHF 中，生成、准备和学习阶段之间存在依赖关系，常导致 GPU 资源空闲，造成浪费。
- **冗余计算**：多个采样 actor 对共享前缀的 prompt 重复执行 **KV cache prefill**，带来不必要的计算开销。
- **长尾响应导致负载不均**：部分 prompt 生成极长响应，拖慢整个 actor 执行周期，造成 GPU 利用率低下。

### **提出了什么新方法或新思路**

本文提出 **RLHFless** —— 首个基于 **serverless computing** 的可扩展同步 RLHF 训练框架，其核心创新包括：

1. **去重预填充机制 (Deduplicated Prefill)**  
   - 在生成阶段前，识别并合并具有相同前缀的 prompts，仅对唯一前缀执行一次 KV cache 计算，并将结果缓存供所有 decode actor 复用，显著减少重复计算。

2. **成本感知的 actor 扩缩容策略 (Cost-aware Actor Scaling)**  
   - 动态预测每步的 workload（基于历史响应长度），并在多个可能的 actor 数量中搜索“甜点”（sweet spot），平衡训练速度与资源成本。
   - 通过建模 `execution time` 和 `GPU×second cost`，选择最优 actor 数量 $N^*$ 以最小化加权目标函数。

3. **基于长度预测的提示词分组与迁移策略 (Prompt Assignment with Cut-and-Migrate)**  
   - 利用历史训练数据中的响应长度，采用 **EWMA（指数加权移动平均）** 预测当前 step 的响应长度。
   - 将相似长度的 prompts 分配到同一 actor，减少内部负载不均衡。
   - 引入 **cut-and-migrate** 机制：当某个 actor 中有长尾响应未完成时，将其迁移到已完成短响应的 actor 上继续处理，进一步释放闲置资源。

4. **局部性感知的 actor 放置 (Locality-aware Actor Placement)**  
   - 将 prefill actor 与 learner 共置，减少模型权重同步延迟。
   - 将处理最长响应的 decode actor 也共置于此节点，使其能立即开始解码，隐藏 KV cache 传输延迟。

### **相比现有方法的优势**

| 维度 | 现有方法（如 VERL） | RLHFless |
|------|----------------------|----------|
| 架构 | Serverful（静态资源） | Serverless（弹性伸缩） |
| 资源利用率 | 低（存在大量 idle time） | 高（按需启动/释放） |
| KV cache 计算 | 重复多次 | 去重后复用 |
| Actor 数量 | 固定 | 动态调整以优化 cost-speed 权衡 |
| 负载均衡 | 无显式优化 | 基于长度预测 + cut-and-migrate |
| 成本控制 | 不敏感 | 显式建模并优化 |

> ✅ **优势总结**：RLHFless 实现了更细粒度的资源调度、更低的成本、更高的训练吞吐量，且适用于大规模同步 RLHF 场景。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **GSM8K**：小学数学应用题，用于测试数学推理能力。
- **GPQA**：研究生级别科学问答，考察深度知识理解。
- **LiveCodeBench**：代码生成任务，评估编程能力。

### **模型与算法组合**

- **模型**：
  - Qwen2.5-3B / Qwen2.5-7B（物理集群）
  - Llama2-70B（大规模仿真）
- **RLHF 算法**：
  - **PPO**（Proximal Policy Optimization）
  - **GRPO**（Group Relative Policy Optimization，更采样密集型）

### **实验平台**

- **物理测试床**：2 台 AWS EC2 `g6e.48xlarge` 实例，每台含 8×NVIDIA L40S GPU（共 16 GPU）。
- **大规模仿真**：使用 **Vidur** 模拟器，在最多 20 节点（每节点 8×H100 GPU）上运行 Llama2-70B 的 RLHF 流程。

### **评估指标**

| 指标 | 定义 | 说明 |
|------|------|------|
| **Per-step execution time** | 每个训练 step 的端到端耗时 | 衡量训练速度 |
| **GPU×second cost** | 所有函数调用中 `(GPU 数量) × (执行时间)` 总和 | 衡量资源消耗与经济成本 |
| **Speedup** | 相对于 baseline 的加速比 | 如 1.35× |
| **Cost reduction** | 相对于 baseline 的成本降低百分比 | 如 44.8% |

### **基线方法对比**

- **VERL**：主流开源 RLHF 框架，作为主要 baseline。
- **RLHFuse**：支持 flex fusion 和全局 cut-and-migrate 的系统，用于比较 prompt assignment 效果。
- **Oracle**：理想情况下的上限（假设完全准确的长度预测）。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 指标 | 结果 |
|------|------|
| 最高加速比（speedup） | **1.35×** |
| 最大成本降低 | **44.8%** |
| 大规模仿真平均加速 | **1.23×** |
| 大规模仿真平均成本降低 | **38.7%** |

> 💡 注：性能提升在 **GRPO** 上更为显著，因其采样更密集，优化空间更大。

### **与基线方法的对比结果**

- 在 **Qwen2.5-3B + GSM8K + GRPO** 设置下：
  - RLHFless 相比 VERL 实现 **1.35× 加速** 与 **44.8% 成本下降**。
  - 成本节约主要来自：
    - 去重 prefill 减少约 22% 的 KV cache 计算。
    - 动态 actor scaling 避免过度资源配置。
    - prompt grouping + cut-and-migrate 缩短尾部延迟。

- 在 **Llama2-70B 大规模仿真** 中：
  - 平均实现 **1.23× 速度提升** 与 **38.7% 成本降低**，验证了可扩展性。

### **消融实验结果**

通过逐步启用各模块，验证每个设计的独立贡献（见 Figure 11）：

| 变体 | 相比原始 VERL 的改进 |
|------|------------------------|
| **w/o DP+AS+PA**（全关闭） | 基线，无改进 |
| **+ Deduplicated Prefill** | 成本略有下降（~5–10%），尤其在多响应场景下更明显 |
| **+ Prompt Assignment** | 显著降低成本（减少 actor 内部空闲），提升 GPU 利用率 |
| **+ Actor Scaling** | 明显缩短 step time，实现最佳 speed-cost 权衡 |
| **Full RLHFless** | 综合效果最优，接近 oracle 上限的 90% 以上 |

> 🔍 发现：三个模块协同作用，缺一不可；其中 **actor scaling** 对速度影响最大，**prompt assignment** 对成本最敏感。

---

## 4. 关键结论和发现

### **主要发现**

1. **Serverless 架构适合 RLHF**  
   - RLHF 的阶段性、动态性特征天然契合 serverless 的弹性伸缩特性，能够有效消除 idle time。

2. **历史长度可用于可靠预测**  
   - LLM 在固定 prompt 集上的响应长度变化趋势稳定（70% 差异 < 50 tokens），无需复杂预测模型即可实现有效 workload 调度。

3. **去重 prefill 具有显著收益**  
   - 在 GRPO 中每 prompt 生成 3 条响应时，理论上可节省 66% 的 prefill 开销，实测节省约 22%，仍有优化空间。

4. **动态扩缩容优于静态配置**  
   - 固定 actor 数量无法适应不同 step 的 workload 波动，而 RLHFless 能自动找到 cost-speed 最优点。

5. **轻量级设计引入的 overhead 可忽略**  
   - 包括长度预测、调度决策等在内的额外开销平均低于 30ms，不影响整体性能。

### **方法的局限性**

- **聚焦于同步 RLHF**：未直接支持异步或 off-policy 方法（尽管 actor scaling 和 prompt assignment 可迁移）。
- **依赖历史数据**：第一轮训练缺乏历史长度信息，需借助 ground truth 或默认值估算。
- **cut-and-migrate 存在状态迁移开销**：若触发频繁或网络带宽不足，可能引入额外延迟。
- **未考虑 reward model 推理成本**：目前主要优化 policy model 的 generation 阶段。

### **未来工作方向**

1. **扩展至异步 RLHF 系统**：将 actor scaling 与 prompt assignment 集成到 AReaL、A-3PO 等异步框架中。
2. **支持多模态 RLHF**：将 serverless 思路推广至图像、音频等模态的对齐训练。
3. **联合优化 reward model serving**：将 RM 推理也纳入 serverless 调度，实现全流程成本控制。
4. **探索更智能的预测器**：结合轻量代理模型（proxy model）进一步提升长度预测精度。
5. **跨任务迁移调度策略**：研究通用的 cost model，使 RLHFless 更易部署于新任务。

---

> ✅ **总结一句话**：  
> **RLHFless 是首个将 serverless 计算引入同步 RLHF 的框架，通过去重 prefill、动态 actor scaling 和长度感知的 prompt 分组，实现了高达 1.35× 的加速和 44.8% 的成本降低，为高效、低成本的大模型对齐训练提供了新范式。**

</details>

---

### 4. [AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning](https://arxiv.org/abs/2602.22268)

**Authors**: Changhai Zhou, Shiyang Zhang, Yuhua Zhou, Qian Qiao, Jun Gao, Cheng Jin, Kaizhou Qin, Weizhong Zhang  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.22268v1  

#### Abstract
Quantization followed by parameter-efficient fine-tuning has emerged as a promising paradigm for downstream adaptation under tight GPU memory constraints. However, this sequential pipeline fails to leverage the intricate interaction between quantization bit-width and LoRA rank. Specifically, a caref...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning  
—— 核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

在 **Large Language Model (LLM)** 下游任务适配中，受限于 GPU 内存，通常采用“先量化后微调”（quantize-then-fine-tune）的两阶段范式：
- 首先对预训练模型进行 **Mixed-Precision Quantization**（如 4-bit），以压缩模型大小；
- 然后通过 **Parameter-Efficient Fine-Tuning (PEFT)** 技术（如 LoRA）训练轻量级适配器。

然而，这种**顺序优化方式存在根本缺陷**：
- 它将 **量化位宽（bit-width）** 和 **LoRA 秩（rank）** 视为独立决策变量；
- 忽略了二者之间的**交互作用**：低精度引入的量化噪声可通过高秩适配器补偿；
- 导致资源分配不均，即使重建误差小，最终微调性能仍可能很差。

> 🔍 **核心问题**：如何在固定内存预算下，联合优化每层的量化位宽和 LoRA 秩，实现最优微调性能？

---

### 🚀 **提出了什么新方法或新思路**

作者提出 **AutoQRA**（Automated Quantization-Rank Allocation），一种**联合优化框架**，同时搜索每个 Transformer 层的最佳 `bit-width` 和 `LoRA rank` 配置。

#### 主要创新点：

1. **联合优化建模**
   - 将 bit-width 与 rank 分配形式化为一个**受内存约束的黑盒优化问题**；
   - 明确指出：**decoupled pipeline（先定 bit 再调 rank）与最终目标（post-finetuning performance）错位**。

2. **粗粒度到细粒度的两阶段搜索策略**
   - **Phase I: Global Multi-Fidelity Evolutionary Search**
     - 使用多保真度进化算法快速筛选候选配置；
     - 初始种群由 layer-wise importance prior 进行 warm-start；
     - 引入 surrogate model 对低保真评估结果进行预测，提升筛选效率。
   - **Phase II: Local Trust-Region Bayesian Optimization (TuRBO)**
     - 在 Phase I 找到的 promising 区域内进行精细化搜索；
     - 使用 Expected Improvement (EI) 准则选择下一个待评估配置；
     - 支持自动终止机制，避免无效探索。

3. **可行性修复机制（REPAIR）**
   - 设计了一个 deterministic projection 算子，将超内存的配置映射回可行集；
   - 优先从敏感度低的层降级 bit 或 rank，保护关键层的可学习性。

4. **主动补偿机制**
   - 发现并利用了 “**低 bit-width + 高 rank**” 的补偿模式：在量化更激进的层分配更高适配能力，从而抵消噪声影响。

---

### ⚖️ **相比现有方法的优势**

| 方面 | 传统方法（如 QLoRA, AdaLoRA） | AutoQRA |
|------|-------------------------------|---------|
| 优化方式 | 串行（先量化再适配） | 联合优化（bit + rank 同步搜索） |
| 搜索空间 | 固定或单轴调整 | 全离散联合空间探索 |
| 性能代理 | 使用静态指标（如 perplexity） | 动态 fine-tuning 反馈驱动 |
| 效率 | 依赖启发式规则 | 多保真 + Surrogate 加速收敛 |
| 表现 | 易出现个别任务崩溃 | 更鲁棒，接近全精度表现 |

> ✅ **优势总结**：AutoQRA 实现了在相同甚至更低内存占用下，达到接近 **full-precision LoRA 微调** 的性能，显著优于 uniform 4-bit 方法。

---

## 2. 核心实验方法和设置

### 📚 **使用的数据集**

- **微调数据集**：
  - Alpaca-52k
  - HC3
- **零样本/少样本评估任务**（共 8 项）：
  - BoolQ, PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA, MMLU

> ✅ 综合评估涵盖常识推理、逻辑判断、多步推理等典型下游能力。

---

### ⚙️ **实验设置与评估指标**

| 项目 | 设置说明 |
|------|----------|
| **Backbone Models** | LLaMA-3.1-8B, LLaMA-3.2-3B, Qwen-2.5-7B, Qwen-2.5-3B |
| **量化范围** | Bit ∈ {2, 4, 8}, 权重仅量化（weight-only） |
| **LoRA Rank 范围** | Rank ∈ {4, 8, 16} |
| **内存预算** | 给定严格上限 $ B_{\text{max}} $，所有方法必须满足 |
| **评估指标** | - 平均准确率（task-average accuracy）<br>- AvgBit（平均位宽）<br>- AvgRank（平均秩）<br>- 总内存占用（Mem, GB） |
| **实现平台** | PyTorch + HuggingFace Transformers / PEFT / BitsAndBytes |
| **硬件环境** | NVIDIA A100 GPUs |

---

### 🔁 **基线方法对比**

| 方法 | 描述 |
|------|------|
| **LoRA (FP16)** | 全精度 LoRA 微调，作为性能上界 |
| **QLoRA (4-bit)** | 4-bit 量化 + 固定 rank LoRA |
| **AdaLoRA (4-bit)** | 自适应 rank 分配，但基于固定 4-bit 量化 |
| **LoftQ / LQ-LoRA** | 尝试联合初始化，但仍为交替优化，非端到端联合搜索 |
| **AMQ+LoRA / AMQ+AdaLoRA** | 先用 AMQ 做混合精度量化，再接 LoRA，代表解耦方案 |

> 💡 特别设置了 **decoupled baseline** 来凸显联合优化的价值。

---

## 3. 主要实验结果和性能指标

### 📊 **关键性能数据（来自 Table 1）**

| 方法 | 模型 | Avg Acc (%) | AvgBit ↓ | AvgRank ↓ | Mem (GB) ↓ |
|------|------|-------------|----------|-----------|------------|
| LoRA (FP16) | LLaMA-3.1-8B | 69.94 | 16.00 | 16.00 | 20.50 |
| QLoRA (4-bit) | LLaMA-3.1-8B | 67.45 | 4.00 | 16.00 | 15.22 |
| **AutoQRA (≤4bit)** | LLaMA-3.1-8B | **69.83** | **3.75** | **10.50** | **13.08** |
| **AutoQRA (Optimal)** | LLaMA-3.1-8B | **70.45** | 5.25 | 12.25 | 17.32 |

> ✅ **关键观察**：
> - AutoQRA (≤4bit) 在 **平均位宽低于 4-bit** 的前提下，**准确率逼近全精度 LoRA（69.83 vs 69.94）**；
> - 内存减少 **12–22%**，且使用更小的 AvgRank（10.5 vs 16），说明资源利用率更高；
> - AutoQRA (Optimal) **全面超越 FP16 LoRA**，验证了联合优化潜力。

---

### 📈 **与其他方法对比结果**

| 比较维度 | 结果 |
|--------|------|
| **vs Uniform 4-bit 方法** | 在所有四个 backbone 上均取得最高准确率，尤其在 GSM8K、WinoGrande 等复杂任务上优势明显 |
| **vs Decoupled Pipelines (AMQ+LoRA)** | 即使使用更强的 AMQ 做量化，后续加 LoRA 也无法匹配 AutoQRA 性能，证明**联合优化不可替代** |
| **vs Alternating Methods (LoftQ)** | LoftQ 依赖局部重建损失，无法捕捉全局训练动态，性能落后 |

> 📌 图表支持：
> - **Figure 5**：AutoQRA 搜索效率远高于随机搜索 —— 达到目标精度仅需 **6 次高保真评估**，而随机搜索需要 **107 次**（**快 18×**）；
> - **Figure 4**：surrogate model 显著提升 top-3 推荐命中率（从 44.7% → 67.3%）；

---

### 🔍 **消融实验结果（Ablation Study）**

| 消融条件 | 影响 |
|---------|------|
| **移除 warm-start / importance prior** | 收敛变慢，初期探索效率下降 |
| **禁用 Phase I（只用 BO）** | 搜索陷入局部最优，难以覆盖 Pareto 前沿 |
| **禁用 Phase II（只用 EA）** | 缺乏精细调优，无法找到最佳操作点 |
| **仅优化 bit 或 rank 单一维度** | 性能大幅下降，证明两者需协同设计 |
| **移除 multi-fidelity / surrogate** | 搜索成本急剧上升，实用性降低 |

> ✅ **结论**：两阶段 coarse-to-fine 架构是高效性的关键保障。

---

## 4. 关键结论和发现

### 🎯 **论文的主要发现**

1. **bit-width 与 LoRA rank 存在强交互关系**
   - 低精度层可通过高秩适配器有效补偿量化噪声；
   - 最优配置呈现“负相关”趋势：**量化越激进的层，往往被分配更高的 rank**（见 Figure 3）。

2. **静态代理指标（如 perplexity）不可靠**
   - 与最终微调性能相关性弱（Pearson ρ ≈ 0.46）；
   - 存在大量“proxy-good, result-poor”或反之的情况（见 Figure 1b）；
   - 必须依赖实际 fine-tuning 反馈进行评估。

3. **AutoQRA 实现了高效的资源再分配**
   - 不是简单地“降低整体 bit”，而是智能地在层间重新分配 bit 与 rank；
   - 在同等内存下，实现了比 uniform 方法更强的表达能力和可学习性。

4. **REPAIR 操作符具有语义意义**
   - 分析显示其倾向于修改对性能影响小的层（p = -0.68），体现了“最小扰动”原则（见 Figure 8）。

---

### ⚠️ **方法的局限性**

| 局限性 | 说明 |
|-------|------|
| **搜索开销仍较高** | 尽管已通过 multi-fidelity 优化，但完整搜索仍需数十次 fine-tuning 迭代，不适合实时部署场景 |
| **依赖高质量 calibration set** | importance prior 的质量会影响 warm-start 效果 |
| **当前仅支持 weight-only quantization** | 未考虑 activation quantization 或 KV-cache 压缩 |
| **扩展性挑战** | 对更大模型（如 70B）或更多候选参数组合，搜索空间呈指数增长 |

---

### 🔮 **未来工作方向**

1. **进一步降低搜索成本**
   - 引入 meta-learning 或 warm-start transfer，在不同模型间迁移搜索经验；
   - 探索 gradient-based relaxation 方法处理离散空间。

2. **扩展至其他压缩技术**
   - 联合优化 pruning + quantization + LoRA；
   - 支持 structured sparsity 与 block-wise quantization。

3. **动态运行时调整**
   - 在推理过程中根据输入动态切换 bit/rank 配置（dynamic inference adaptation）。

4. **理论解释补偿机制**
   - 建立 formal theory 解释为何某些层适合“以 rank 换 bit”。

---

## ✅ 总结

AutoQRA 是首个系统性解决 **mixed-precision quantization 与 LoRA rank 联合优化** 的自动化框架。它揭示了传统 pipeline 中因忽略 bit-rank 交互而导致的资源错配问题，并通过 **multi-fidelity evolutionary search + Bayesian refinement** 的两阶段策略，在极少量高成本评估下找到了 Pareto 最优解。

> 🔑 **一句话总结**：  
> **AutoQRA 通过让适配器“聪明地补偿”量化噪声，在不到 4-bit 的平均精度下，实现了媲美全精度微调的性能，树立了高效 LLM 微调的新标杆。**

</details>

---

### 5. [Accelerating LLM Pre-Training through Flat-Direction Dynamics Enhancement](https://arxiv.org/abs/2602.22681)

**Authors**: Shuchen Zhu, Rizhen Hu, Mingze Wang, Mou Sun, Xue Wang, Kun Yuan, Zaiwen Wen  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.22681v1  

#### Abstract
Pre-training Large Language Models requires immense computational resources, making optimizer efficiency essential. The optimization landscape is highly anisotropic, with loss reduction driven predominantly by progress along flat directions. While matrix-based optimizers such as Muon and SOAP levera...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerating LLM Pre-Training through Flat-Direction Dynamics Enhancement

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）预训练过程计算成本极高，优化器效率成为制约训练速度的关键因素。尽管当前主流的矩阵型优化器（如 **Muon** 和 **SOAP**）通过利用更精细的曲率信息超越了 **AdamW**，但它们在更新过程中存在以下两个关键缺陷：

1. **预条件器导致的各向同性更新幅度**：虽然这些优化器能修正下降方向，但其更新幅度趋于各向同性（即在平坦和尖锐方向上的更新量相近），这在高度病态（ill-conditioned）且各向异性（anisotropic）的损失景观中是次优的——在主导损失下降的**平坦方向**上过于保守，在可能导致不稳定的**尖锐方向**上又可能过于激进。

2. **动量机制对非凸性的适应不足**：现有的动量方案（如 EMA）本质上是线性的，施加的是各向同性的阻尼效应，未能有效利用二阶信息来加速在非凸平坦方向上的收敛。

### 提出的新方法和新思路
为解决上述问题，本文提出了两大核心贡献：

#### （1）统一的黎曼常微分方程（Riemannian ODE）框架
- **创新点**：首次建立了一个统一的 **Riemannian ODE** 框架，将主流自适应优化算法（如 AdamW、Lion、Muon、SOAP 及其 Nesterov 加速变体）纳入一个连续时间流形优化的视角进行分析。
- **核心洞察**：该框架揭示了**预条件器**（preconditioner）和**动量**（momentum）的协同作用机制：
  - **预条件器**诱导了一个**黎曼几何**（Riemannian geometry），通过改变空间度量来缓解损失景观的病态性。
  - **动量**则充当了**黎曼阻尼**（Riemannian damping）项，促进在该新度量下的收敛。

这一理论框架弥合了现有研究中将预条件和动量孤立分析的鸿沟，为设计更优的优化器提供了坚实的理论基础。

#### （2）LITE：一种通用的加速策略
- **创新点**：基于上述理论框架，提出 **LITE**（acceLerating adaptIve opTimizers in LLM prE-training），一种通用的优化器加速策略。
- **核心思想**：LITE 旨在**增强平坦方向上的训练动力学**。它通过在平坦子空间内应用更大的**Hessian 阻尼系数**（Hessian damping coefficients）和**学习率**（learning rates），从而：
  - 在平坦方向上**放大更新幅度**（larger update magnitudes）。
  - **增强动量积累**（enhanced momentum accumulation）。
- **实现方式**：LITE 并非一个独立的优化器，而是一个可插拔的**加速框架**，可以应用于 Muon、SOAP 等现有先进优化器之上，形成 **MUON-LITE** 和 **SOAP-LITE**。

### 相比现有方法的优势
- **理论指导性强**：LITE 的设计有明确的理论依据（Riemannian ODE 框架），而非启发式调整。
- **通用性高**：可无缝集成到多种矩阵型优化器中。
- **效率提升显著**：实验证明能大幅加速训练，尤其在长周期训练中表现出接近 **2× 的速度提升**。
- **稳定性好**：通过分别处理平坦和尖锐方向，避免了在尖锐方向上因过度加速而导致的不稳定。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **C4**：用于中小规模预训练实验。
- **The Pile**：用于大规模或长上下文预训练任务。

### 实验设置和评估指标
- **模型架构**：
  - **Dense Models**：LLaMA 系列（0.13B, 0.25B, 0.5B, 1.3B 参数）。
  - **MoE Models**：QwenMoE（1B 参数）。
- **学习率调度**：
  - `cos`：线性预热后接余弦衰减。
  - `wsd`（warmup-stable-decay）：线性预热，稳定期保持最大学习率，最后线性衰减至零。
- **评估指标**：
  - 主要指标：**训练损失**（training loss）随迭代次数的变化。
  - 下游任务：在多个基准（如 ARC, BoolQ, MMLU 等）上的 **0-shot 性能**。
  - 扩展性分析：不同 token 预算和模型规模下的**缩放定律**（scaling laws）。

### 基线方法对比
- **主要基线**：
  - **Muon**（及其 AdamW 变体）
  - **SOAP**
- **LITE 变体**：
  - **MUON-LITE / SOAP-LITE**：完整版本，同时增加平坦方向的学习率和 Hessian 阻尼。
  - **MUON-LITE-L / SOAP-LITE-L**：仅增加学习率（x > 1）。
  - **MUON-LITE-H / SOAP-LITE-H**：仅增加 Hessian 阻尼系数（β₂ > β₁）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **显著降低终端损失**：在所有测试场景下（不同模型大小、数据集、优化器），**MUON-LITE** 和 **SOAP-LITE** 均一致地达到了比其对应基线更低的终端损失。
- **卓越的扩展性**：
  - **图1** 显示，MUON-LITE 展现出更优的缩放行为。在更大的模型规模和 token 预算下，其性能增益持续存在。
  - **长周期训练中的 2× 速度提升**：在将训练 token 预算从 40× 模型参数量提升至 200× 时，MUON-LITE 在约 100× token 处就达到了 Muon 在 200× token 处的损失水平，实现了近似 **2 倍的加速**。
- **MoE 模型上的有效性**：在 QwenMoE-1B 模型上，MUON-LITE 同样显著优于 Muon 基线，且随着训练进行，性能差距逐渐拉大。

### 消融实验结果
- **LITE vs. LITE-L vs. LITE-H**：
  - **LITE 效果最佳**：同时增加学习率和 Hessian 阻尼的完整版本 **LITE** 性能最好。
  - **LITE-H（仅增加阻尼）**：通常优于或等同于 LITE-L，表明**增强动量机制**（Hessian damping）是 LITE 成功的关键。
  - **LITE-L（仅增加学习率）**：效果相对较弱。
- **错误应用的反例**：如果将本应只用于平坦方向的更大阻尼系数（如 β₁.₂ = 0.5 或 1.0）均匀地应用于所有方向，反而会导致终端损失**劣于**原始的 Muon 基线。这有力地证明了 LITE “选择性加速”策略的必要性和正确性。

---

## 4. 关键结论和发现

### 主要发现
1. **理论层面**：成功建立了统一的 **Riemannian ODE 框架**，阐明了预条件器和动量在自适应优化中的协同作用，为理解现代优化器提供了新的理论视角。
2. **方法层面**：提出的 **LITE** 策略通过**增强平坦方向的动力学**，有效解决了现有优化器在各向异性景观中更新幅度各向同性的问题。
3. **实验层面**：LITE 能够**显著加速** Muon 和 SOAP 等 SOTA 优化器的训练过程，不仅降低了最终损失，还展现出优越的扩展性，在长周期训练中实现了接近 **2× 的速度提升**。

### 方法的局限性
- **超参数调优**：LITE 引入了新的超参数（如平坦子空间维度 `ds`、放大系数 `x`、阻尼系数 `β₂`），需要额外的网格搜索来确定最优值。
- **适用范围**：目前主要针对矩阵型优化器（Muon, SOAP）进行了验证，其在其他类型优化器上的普适性有待进一步探索。
- **计算开销**：虽然作者声称开销极小（吞吐量仅下降约 1%），但在极端追求效率的场景下，估计平坦/尖锐子空间的额外计算仍构成轻微负担。

### 未来工作方向
1. **自适应超参数**：探索动态调整 LITE 中的超参数（如 `x`, `β₂`）的机制，减少人工调参成本。
2. **更广泛的集成**：将 LITE 框架扩展到更多新兴的优化器上。
3. **理论深化**：进一步完善对 LITE 在非凸、随机环境下的收敛性理论分析。

</details>

---

### 6. [Efficient Continual Learning in Language Models via Thalamically Routed Cortical Columns](https://arxiv.org/abs/2602.22479)

**Authors**: Afshin Khadangi  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.22479v1  

#### Abstract
Continual learning is a core requirement for deployed language models, yet standard training and fine-tuning pipelines remain brittle under non-stationary data. Online updates often induce catastrophic forgetting, while methods that improve stability frequently increase latency, memory footprint, or...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Efficient Continual Learning in Language Models via Thalamically Routed Cortical Columns*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大型语言模型（LLMs）在部署后面临**持续学习**（Continual Learning）的挑战：  
- 在非平稳数据流中进行在线更新时，容易发生**灾难性遗忘**（Catastrophic Forgetting）。  
- 现有轻量级微调方法（如 LoRA、Adapter）虽降低计算成本，但连续更新仍导致参数干扰和性能退化。  
- 传统架构缺乏对“稳定性-可塑性”（stability-plasticity）权衡的原生支持。

### 提出的新方法：TRC²（Thalamically Routed Cortical Columns）
TRC² 是一种**专为持续学习设计的 decoder-only 架构**，其核心思想是将学习机制从架构层面解耦：

#### 创新点：
- **稀疏丘脑路由**（Sparse Thalamic Routing）  
  每个 token 通过一个 **top-k 路由器**选择激活一小部分“皮层柱”（cortical columns），实现**稀疏且可控的计算通信**，避免全局扰动。
  
- **双路径可塑性机制**  
  - **慢速稳定路径**：主干皮层参数保持长期稳定，支持抽象表示。  
  - **快速修正路径**（Cerebellar Corrector）：引入低秩（low-rank）**快权重**（fast weights）路径，用于在部署时快速适应新数据而不重写主干参数。

- **生物启发的模块化设计**  
  集成多个神经科学启发的子系统：
  - **预测编码**（Predictive Coding）：上下文预测误差反馈
  - **联想记忆**（Associative Memory）：基于 Modern Hopfield Network 的内容寻址记忆
  - **兴奋-抑制门控**（EI Gating）：模拟 SST/PV/VIP 神经元的局部调控
  - **拓扑感知路由先验**（Topology-aware Prior）：鼓励时间连续性，提升路由稳定性

#### 相比现有方法的优势：
| 维度 | TRC² 优势 |
|------|-----------|
| **持续学习能力** | 显著降低遗忘，支持无回放（replay-free）在线更新 |
| **计算效率** | 块级并行、chunk-parallel 实现，支持高效训练与推理 |
| **架构集成性** | 将适应机制内建于前向计算图，而非外部附加流程（bolt-on procedure） |
| **可解释性与控制性** | 各子系统可独立开关，便于分析与调节 |

---

## 2. 核心实验方法和设置

### 数据集
- **训练数据**：`C4`（streaming 模式），模拟真实部署中的非平稳文本流
- **验证探针**（Validation Probes）：
  - `C4`（validation split）
  - `WikiText-103-v1`：检测过拟合与泛化稳定性
  - `LAMBADA`：测试长程依赖与语境理解能力

> 所有探针被视为**连续任务流**，用于评估模型在动态环境下的表现漂移与遗忘。

---

### 实验设置
- **硬件**：单节点 4×NVIDIA V100（32GB），混合精度训练
- **序列长度**：1024
- **批量大小**：每 GPU 8，梯度累积 4 步 → 全局批大小 128 序列（131,072 tokens/step）
- **优化器**：AdamW，lr=2e-4，cosine decay，1000 步 warmup
- **总训练步数**：22,000 步（约 2.88B tokens）

---

### 评估指标
| 类型 | 指标 |
|------|------|
| **基础性能** | Perplexity（PPL）、BLEU 分数 |
| **效率** | Tokens/s（吞吐量）、Peak Memory（峰值内存） |
| **持续学习能力** | **Proxy Forgetting**：<br> - 对于 PPL：当前值相对于历史最佳值的上升<br> - 对于 BLEU/Accuracy：当前值相对于历史最佳值的下降<br> 报告最后一步的平均遗忘 + 归一化 AUC 忘记面积 |
| **其他** | Token Accuracy、Exact Match、ROUGE、chrF |

---

### 基线方法对比
- **Transformer**：标准 dense attention 架构
- **Mamba**：基于状态空间模型（SSM）的高效序列建模架构
- **TRC²（本文）**：提出的稀疏路由架构

所有模型在相同训练管道下进行参数匹配比较。

---

## 3. 主要实验结果和性能指标

### 表1：基础性能与效率对比

| Model | Params | PPL (C4) ↓ | PPL (Wiki) ↓ | PPL (LAM) ↓ | BLEU (C4) ↑ | BLEU (Wiki) ↑ | BLEU (LAM) ↑ | Tokens/s ↑ | Mem×Hour/GPU ↓ |
|-------|--------|------------|-------------|-------------|--------------|----------------|---------------|-------------|------------------|
| Transformer | 162M | 60.70 | 215.18 | 105.72 | 8.12 | 8.23 | 5.09 | ~127,000 | 118 GB·h |
| Mamba | 176M | 70.45 | 357.67 | 116.73 | 6.90 | 2.87 | 3.97 | ~108,000 | 178 GB·h |
| **TRC² (ours)** | **169M** | **2.00** | **2.56** | **2.02** | **71.66** | **66.57** | **70.07** | **~57,000** | **268 GB·h** |

> ✅ **结论**：TRC² 在语言建模质量上远超基线（PPL 下降两个数量级，BLEU 大幅提升），但以显著更高的内存消耗和更低的吞吐为代价。

---

### 表2：持续学习性能（Streaming Task Suite）

| Model | Average Forgetting (Last Step) ↓ | Average Forgetting (Normalized AUC) ↓ |
|-------|-------------------------------|------------------------------------|
| | PPL | TokenAcc | BLEU | PPL | TokenAcc | BLEU |
| Transformer | 0.0000 | 0.0014 | 0.3757 | 0.0669 | 0.0008 | 0.1684 |
| Mamba | 0.0000 | 0.0006 | 0.0900 | 0.3371 | 0.0011 | 0.1957 |
| **TRC² (ours)** | **0.0110** | **0.0010** | **0.0435** | **0.0018** | **0.0008** | **0.0981** |

> ✅ **关键发现**：
> - TRC² 的 **normalized AUC 忘记面积最小**，表明其在整个训练流中**更一致地保留了早期行为**。
> - 尽管最后一步有轻微遗忘（如 PPL 上升 0.011），但这说明模型仍在学习，而非完全冻结。
> - 在 BLEU 指标上，TRC² 的遗忘仅为 Mamba 的 ~48%，Transformer 的 ~12%。

---

### 消融实验（Ablation Studies）
文中虽未列出完整表格，但在讨论中明确指出以下组件对性能至关重要：
- **拓扑感知路由先验**（Topology-aware prior）：提升路由的时间连续性，减少参数干扰。
- **兴奋-抑制门控**（EI Gating）：抑制不稳定激活传播，增强鲁棒性。
- **快权重修正路径**（Cerebellar Corrector）：提供快速适应通道，是实现低遗忘的关键。
- **路由权重 refinement loop**（Cortico-thalamic feedback）：通过反馈优化路由决策，提升一致性。

> 🔍 控制变量实验证明：关闭任一模块均会导致遗忘增加或适应速度下降。

---

## 4. 关键结论和发现

### 主要结论
1. **持续学习应作为架构属性**  
   TRC² 成功将“稳定性-可塑性”权衡内建于模型结构中，而非依赖外部正则化或训练策略。

2. **稀疏路由 + 快速修正路径有效缓解遗忘**  
   通过将新知识路由至局部子网络，并用低秩快权重进行增量调整，实现了**快速适应而不破坏已有知识**。

3. **生物启发机制具有实际工程价值**  
   如预测编码、联想记忆、双室神经元读出等机制，在大规模语言模型中展现出可扩展性和功能性。

4. **TRC² 在持续学习场景下显著优于主流架构**  
   尽管牺牲了部分推理效率，但在**长期记忆保持**方面取得突破性进展。

---

### 局限性
- **计算开销高**：由于稀疏投影与多次 cortex pass，**吞吐量仅为 Transformer 的 ~45%**，内存占用翻倍。
- **对剧烈分布偏移敏感**：当数据流变化过于频繁或剧烈时，路由器可能变得不稳定。
- **chunk-level 路由损失细粒度信号**：固定 chunk 内共享路由决策，可能忽略 token 级别差异。
- **尚未扩展到更大规模**：当前实验限于 ~170M 参数级别，需验证在百亿级以上是否仍有效。

---

### 未来工作方向
1. **优化实现效率**：通过 kernel fusion、内存布局优化、动态稀疏执行等方式提升吞吐。
2. **增强路由器鲁棒性**：研究在强非平稳流中的自适应路由机制。
3. **结合 test-time learning**：利用快权重路径实现真正的部署时学习（test-time training）。
4. **探索可逆/受限更新**：使模型能识别并拒绝噪声或对抗性输入的影响。
5. **扩展至 encoder-decoder 或多模态架构**。

---

> 📌 **总体评价**：  
> TRC² 提供了一个全新的视角——将**持续学习能力视为神经网络架构的一等公民**。它不仅是一个新模型，更是一种**面向动态世界的建模范式转变**。尽管当前效率尚不理想，但其设计理念为构建真正“活”的语言系统指明了方向。

</details>

---

### 7. [Search-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic RAG Training](https://arxiv.org/abs/2602.22576)

**Authors**: Tianle Xia, Ming Xu, Lingxiang Hu, Yiding Sun, Wenwei Li, Linfang Shang, Liqun Liu, Peng Shu, Huan Yu, Jie Jiang  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.22576v1  

#### Abstract
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, yet traditional single-round retrieval struggles with complex multi-step reasoning. Agentic RAG addresses this by enabling LLMs to dynamically decide when and what to retrieve, but current...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SEARCH-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic RAG Training

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **Retrieval-Augmented Generation (RAG)** 在处理复杂多步推理任务时存在显著局限：
- **单轮检索** 无法满足需要跨多个知识域逐步推理的需求。
- 现有的基于强化学习（RL）的 **Agentic RAG** 方法（如 Search-R1）依赖于稀疏的 **outcome-based reward**（仅根据最终答案是否正确打分），导致：
  - **奖励稀疏性**：忽略中间推理路径的质量信号。
  - **样本效率低**：失败轨迹得不到任何训练信号。
  - **收敛慢**：大多数样本获得相似的二元奖励，梯度信息弱。

### 🚀 提出的新方法：SEARCH-P1
提出一种全新的 **path-centric reward shaping** 框架，核心是将奖励设计从“只看结果”转向“关注全过程”。

#### 主要创新点：
1. **Dual-Track Path Scoring（双轨路径评分）**
   - **Track A: Self-Consistency**  
     评估模型是否忠实执行其自声明的计划（`planner → execution` 匹配度）。
   - **Track B: Reference-Alignment**  
     利用离线生成的 **reference planner**（通过拒绝采样 + LLM 投票构建）作为专家路径参考，衡量模型路径对关键步骤的覆盖情况。
   - 采用 **order-agnostic matching**（顺序无关匹配），允许灵活推理顺序。
   - 最终取两者的最大值 $ R_{\text{path}} = \max(S_{\text{self}}, S_{\text{ref}}) $，避免次优参考路径压制更优策略。

2. **Soft Outcome Scoring（软结果评分）**
   - 即使最终答案错误，也给予部分信用：
     $$
     R_{\text{outcome}} = \alpha \cdot r_{\text{acc}} + (1-\alpha) \cdot r_{\text{reason}}, \quad \alpha=0.8
     $$
   - $ r_{\text{acc}} $ 表示答案部分正确性，$ r_{\text{reason}} $ 表示推理质量独立评分。
   - 将原本零奖励的失败样本转化为有效训练信号，提升样本利用率。

3. **Format Reward（格式奖励）**
   - 引入缓冲机制（soft format），对输出格式合规性提供渐进反馈，而非完全失败惩罚，加速初期训练稳定性和收敛速度。

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 Search-R1） | SEARCH-P1 |
|------|--------------------------|-----------|
| 奖励密度 | 稀疏（binary outcome） | 密集（path-level + soft outcome） |
| 样本效率 | 失败样本无贡献 | 失败样本仍可学习 |
| 收敛速度 | 慢（>150 steps） | 快（~60 steps 达标） |
| 推理稳定性 | 成功/失败路径长度差异大 | 更一致的交互次数 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **公开 QA 基准（7个）**：
  - **General QA**：NQ、TriviaQA、PopQA
  - **Multi-Hop QA**：HotpotQA、2WikiMultiHopQA、Musique、Bamboogle
- **内部工业数据集（1个）**：
  - **AD-QA**：来自真实广告业务场景的匿名化多跳问答数据集（1,000 测试实例），涉及广告投放、受众定向等复杂决策。

> 所有 RL 训练使用 NQ 和 HotpotQA 的合并训练集，测试涵盖 in-domain 与 out-of-domain 泛化能力。

### ⚙️ 实验设置
- **模型**：
  - 主模型：`Qwen2.5-7B-Instruct`, `Qwen2.5-3B-Instruct`
  - 参考规划器（offline）：`HY 2.0-Instruct`（专有）
  - 评估器（training-time）：`HY 2.0-Instruct`, `Qwen3-32B`, `Qwen3-8B`
- **检索系统**：
  - 知识源：2018 Wikipedia dump
  - 检索器：E5，每步返回 top-3 文档
- **训练算法**：GRPO（Group Relative Policy Optimization）
- **超参数**：
  - 默认权重：$ \lambda_f=0.1, \lambda_p=0.3, \lambda_a=0.6 $
  - 动作预算（action budget）：最多 4 轮 search-reason 循环

### 📊 评估指标
- **主指标**：Accuracy (ACC%) —— 是否包含标准答案
- **辅助分析指标**：
  - 训练效率（accuracy vs. training steps）
  - 推理步数（interaction turns）
  - 格式合规率（output compliance）
  - 人类一致性（human agreement on scoring）

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **Non-Retrieval** | Direct Inference, Chain-of-Thought (CoT) |
| **Standard RAG** | Single-round retrieval before generation |
| **Prompt-Based Agentic RAG** | IRCoT, Search-o1 |
| **RL-Based Agentic RAG** | Search-R1, HiPRAG |

> 所有 RL 方法共享相同训练配置，仅 reward 函数不同。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table 1）

| 方法 | 平均 ACC (7B) | AD-QA (7B) | 平均 ACC (3B) | AD-QA (3B) |
|------|---------------|------------|----------------|-------------|
| Search-R1 | 39.6% | 65.6% | 33.6% | 58.3% |
| HiPRAG | 42.9% | 75.6% | 36.6% | 70.2% |
| **SEARCH-P1 (Ours)** | **47.3%** | **86.2%** | **41.5%** | **79.5%** |

> ✅ **平均准确率提升 +7.7 pts（7B）**, 在 AD-QA 上高达 **+20.6 pts**

#### 图表支持：
- **Figure 1**: SEARCH-P1 在所有数据集上全面领先。
- **Figure 4**: Soft outcome scoring 对 multi-hop 和 AD-QA 提升显著（+3.5% ~ +8.8%）。

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）路径奖励组件消融（Table 2 & Table 8）
| 方法 | 平均 ACC (7B) | 相比全模型下降 |
|------|----------------|----------------|
| SEARCH-P1 (Full) | 47.3% | — |
| w/o Reference-Alignment | 42.0% | ↓5.3% |
| w/o Self-Consistency | 44.2% | ↓3.1% |

> 💡 结论：两个轨道互补，**reference-alignment 对 multi-hop 更重要**，**self-consistency 对 general QA 更关键**。

#### （2）Soft Outcome Scoring 消融（Figure 4 & Table 10）
| 场景 | 性能增益 |
|------|---------|
| General QA | +1.1 ~ +1.5% |
| Multi-Hop QA | +3.0 ~ +3.7% |
| AD-QA（工业数据） | **+8.8 ~ +11.0%** |

> ✅ 复杂任务中软评分带来的收益更大，验证了“部分正确也应被鼓励”的有效性。

#### （3）Format Reward 设计影响（Figure 3）
- **Strict Format**（格式错则零分）：早期几乎无奖励，训练停滞。
- **Without Format**：格式错误率高，解析困难。
- **Soft Format（本文）**：提供连续梯度，**显著加快收敛速度和稳定性**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **路径中心奖励（path-centric reward）显著优于结果奖励（outcome-only）**：
   - 提供密集、细粒度监督信号，缓解 RL 中的稀疏奖励问题。
2. **双轨路径评分机制有效结合内外部视角**：
   - `Self-Consistency` 鼓励内部逻辑一致；
   - `Reference-Alignment` 提供外部专家引导；
   - 二者结合实现鲁棒且高效的路径优化。
3. **软评分极大提升样本效率**：
   - 即使最终答案错误，高质量推理过程也能获得正向激励。
4. **在真实工业场景（AD-QA）中优势尤为明显**：
   - +20.6 pts 的巨大提升表明该方法在复杂企业级应用中的实用价值。
5. **方法具有良好的泛化性**：
   - 在不同规模模型（7B/3B）、不同 RL 算法（GRPO/PPO）、不同基础模型（Qwen/Llama）上均保持增益（Table 3 & 9）。

### ⚠️ 局限性
1. **依赖外部 LLM evaluator 进行训练期评分**：
   - 虽然推理时不需调用，但训练成本增加；
   - 对 evaluator 的质量有一定敏感性（Table 13 显示小模型 evaluator 如 Qwen3-8B 会导致性能下降）。
2. **reference planner 为静态离线生成**：
   - 无法动态适应新领域或长尾问题；
   - 存在“参考路径本身非最优”的风险。
3. **目前仅适用于特定格式输出（tagged structure）**：
   - 需要精心设计 prompt 和 parser 来提取 reasoning/action 结构。

### 🔮 未来工作方向
1. **动态参考路径生成**：在线学习或迭代优化 reference planner。
2. **减少对外部 evaluator 的依赖**：探索 self-evaluation 或轻量级 reward model 替代方案。
3. **扩展到更多 Agent Action 类型**：如数据库查询、代码执行等 beyond search。
4. **应用于其他 agentic workflow**：如自动科研、法律咨询、医疗诊断等高阶认知任务。

---

## 总结
SEARCH-P1 通过引入 **path-centric reward shaping**，从根本上改进了 agentic RAG 的训练范式。它不仅提升了性能（平均 +7.7 pts），更重要的是解决了 RL 训练中长期存在的 **奖励稀疏性** 和 **样本效率低下** 问题，在学术基准和真实工业场景中都展现出强大潜力，为下一代智能代理系统的高效训练提供了新思路。

</details>

---

### 8. [Sustainable LLM Inference using Context-Aware Model Switching](https://arxiv.org/abs/2602.22261)

**Authors**: Yuvarani, Akashdeep Singh, Zahra Fathanah, Salsabila Harlen, Syeikha Syafura Al-Zahra binti Zahari, Hema Subramaniam  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.22261v1  

#### Abstract
Large language models have become central to many AI applications, but their growing energy consumption raises serious sustainability concerns. A key limitation in current AI deployments is the reliance on a one-size-fits-all inference strategy where most systems route every request to the same larg...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Sustainable LLM Inference using Context-Aware Model Switching》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大多数 **Large Language Models (LLMs)** 部署采用“一刀切”的推理策略，即所有请求无论复杂度高低都路由到同一个大型模型进行处理。这种做法导致了严重的**能源浪费**，尤其是在处理简单查询（如问候、常识问答）时仍消耗与复杂任务（如代码生成、多步推理）相当的计算资源。

此外，随着 LLMs 在生产环境中的广泛应用，**inference 阶段的能耗**已成为 AI 可持续性（Sustainable AI）的重要瓶颈，而现有研究多聚焦于训练阶段或模型压缩技术，对动态推理优化关注不足。

---

### 🚀 提出的新方法与新思路
本文提出了一种 **Context-Aware Model Switching（上下文感知模型切换）** 架构，用于实现节能高效的 LLM 推理。其核心思想是：  
> **根据 query 的复杂度，动态选择最合适大小的 LLM 进行响应**，从而在保证输出质量的前提下最小化能耗和延迟。

该系统采用**三级混合路由架构（three-level hybrid routing architecture）**：
1. **Level 1 (Cache Layer)**：使用 LRU 缓存 + TTL=300s 快速响应重复查询。
2. **Level 2 (Rule-Based Classifier)**：基于 96 个预编译正则表达式和关键词哈希集进行模式匹配，识别编程语法、数学符号等结构特征。
3. **Level 3 (Semantic ML Classifier)**：利用 `all-MiniLM-L6-v2` 生成句子嵌入，并通过余弦相似度匹配预定义的任务向量，判断语义复杂度。

此外，系统还引入了一个**用户自适应组件（user-adaptive component）**，能够基于会话历史动态调整复杂度阈值，提升个性化路由准确性。

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | 本工作的改进 |
|------|--------|-------------|
| **Cascade-based Routing (e.g., FrugalGPT)** | 顺序执行多个模型，引入额外延迟 | 单次决策路径，避免串行调用 |
| **Learned Routing (e.g., RouteLLM)** | 依赖偏好数据（preference data），跨域泛化能力弱 | 结合规则+语义学习，更具可解释性和鲁棒性 |
| **Model Compression / Distillation** | 需要重新训练或微调，部署成本高 | 完全无需修改模型，仅改变调度逻辑 |
| **Early Exit / Sparse Models** | 修改模型内部结构，硬件兼容性差 | 架构无关，支持任意本地开源模型 |

✅ **优势总结**：
- **无需重训练模型**，适用于任何本地部署的 open-source LLM；
- **低延迟设计**：优先使用轻量级机制（cache → rule → ML）；
- **高可解释性**：规则层提供透明决策依据；
- **模块化架构**：易于扩展至其他任务或分布式场景。

---

## 2. 核心实验方法和设置

### 📚 数据集
构建了一个包含 **150 条 prompt 的标准化评估数据集**，均匀分布于三类复杂度：
- **Simple Queries (50条)**：问候语、单句事实查询、常见知识检索（如“你好吗？”、“水的化学式是什么？”）
- **Medium Queries (50条)**：基础推理、多句解释、事实综合（如“简述光合作用的过程”）
- **Complex Queries (50条)**：多步推理、结构化输出、代码编写（如“写一个Python函数实现快速排序”）

所有 query 经人工校验以确保类别清晰、无歧义。

---

### ⚙️ 实验设置
- **运行平台**：单台本地主机（Windows 11, 64-bit）
- **硬件配置**：
  - CPU: AMD Ryzen 7 5800H (8核16线程)
  - RAM: 32GB DDR4
  - GPU: NVIDIA GeForce GTX 1650 Ti (4GB GDDR6)，启用 CUDA 11.8
- **模型后端**：Ollama API
- **测试模型集合**：
  - Small: Gemma3 1B
  - Medium: Gemma3 4B
  - Large: Qwen3 4B

> 所有模型均为本地加载，GPU 内存共享（unified memory model），且启用 `keep_alive: 0` 策略以释放空闲模型内存。

---

### 📊 评估指标
分为两大类：

#### ✅ 效率指标（Efficiency Metrics）
| 指标 | 测量方式 |
|------|---------|
| **End-to-end Latency** | 从输入到完整响应的时间（秒） |
| **Throughput** | Tokens per second (tps) |
| **Energy Consumption** | 使用 NVML GPU 功耗遥测 + 时间戳计算每 query 能耗（单位：Joules） |
| **Estimated CO₂ Emissions** | 基于全球平均碳强度 475 gCO₂e/kWh 换算 |

#### ✅ 效果指标（Effectiveness Metrics）
| 指标 | 描述 |
|------|------|
| **Routing Accuracy** | 分类正确的比例（按类别统计 Precision/Recall/F1） |
| **Output Quality** | 使用 **BERTScore F1** 衡量自适应系统输出 vs. 大模型基线输出之间的语义相似度（baseline 作为 reference） |

---

### 🔁 基线方法对比
- **Baseline**: 所有 query 均由最大模型（Qwen3 4B）处理，模型常驻 GPU。
- **Proposed Method**: 自适应路由系统（Smart Routing Pipeline）

两者在同一硬件环境下运行相同 query 集合，每条执行 3 次取均值。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 指标 | Baseline（始终用大模型） | Adaptive System（本文方法） | 提升幅度 |
|------|--------------------------|-------------------------------|----------|
| **平均响应延迟** | 13.8 秒 | **3.5 秒** | ↓ **68%** |
| **吞吐量 (tokens/sec)** | 25.4 tps | **61.3 tps** | ↑ **141%** |
| **总能耗 (150 queries)** | 84.2 kJ | **22.0 kJ** | ↓ **67.5%** |
| **估算碳排放** | ~11.1 gCO₂e | **~2.9 gCO₂e** | ↓ **67.5%** |
| **输出质量 (BERTScore F1)** | 100% (reference) | **93.6%** | 保持 >93% 质量 |
| **总体路由准确率** | — | **79.3%** | — |
| **加权 F1 Score** | — | **78.1%** | — |

---

### 🔍 类别级表现分析
| 查询类型 | 路由 Recall | 特点 |
|--------|------------|------|
| **Simple** | 98% | 几乎全部被正确识别并路由至小模型 |
| **Medium** | 平衡 | Precision 与 Recall 均良好 |
| **Complex** | 52% | **Recall 较低但 Precision 高达 96.3%**<br>→ 设计上偏向保守升级，防止降级错误 |

> 💡 这表明系统采取“宁可误判为复杂也不错杀”的策略，保障关键任务质量。

---

### ❌ 消融实验与定性观察（Ablation Insights）
虽然文中未明确列出消融实验表格，但从描述中可推断以下关键发现：
- **缓存层贡献显著**：对于重复查询，L1 缓存将延迟降至 <0.1ms，极大降低平均开销。
- **规则层高效稳定**：96 条规则覆盖大量常见复杂结构（如代码块、公式），分类延迟仅 0.1–1.0ms。
- **语义层补全边界 case**：当规则无法置信判断时，`all-MiniLM-L6-v2` 成功捕捉语义意图。
- **用户自适应组件有效**：在长对话中能逐步学习用户风格，提高 borderline query 的路由精度。

> Stress Test 显示：系统在高负载下仍保持稳定性，cache hit rate 上升进一步优化性能。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **模型切换可大幅节能**：相比始终使用最大模型，本文方法实现 **67.5% 的能耗下降** 和 **68% 的延迟降低**，尤其利好简单/中等查询。
2. **响应质量高度保留**：输出与大模型基准的 **BERTScore F1 达 93.6%**，说明语义一致性良好，用户体验未明显受损。
3. **效率提升源于智能分流**：性能增益主要来自将简单任务交给小型模型处理，而非算法层面的模型加速。
4. **混合路由优于纯学习方法**：结合 deterministic rules 与 semantic embedding，在可解释性、效率和泛化之间取得更好平衡。
5. **可持续 AI 的可行路径**：无需专用硬件或模型重构，即可通过架构创新实现绿色推理。

---

### ⚠️ 方法的局限性
1. **单机部署限制**：实验局限于 single-host 场景，未验证高并发或多节点下的扩展性。
2. **自动化评估依赖 BERTScore**：缺乏 human evaluation，可能忽略细微语义差异或创造性退化。
3. **stress test 结果定性为主**：未提供大规模压力测试的量化指标。
4. **规则需手动维护**：96 条规则虽有效，但在新领域需人工扩展，自动化程度有限。
5. **仅适用于对话型 workload**：未测试非文本生成任务（如 embedding、retrieval）。

---

### 🔮 未来工作方向
1. **引入在线学习机制**：让路由策略随时间自动进化，减少人工干预。
2. **支持更多模型与异构设备**：拓展至边缘设备（edge AI）、移动端部署。
3. **构建开放 benchmark**：推动社区建立标准的 **LLM routing evaluation suite**（类似 RouterBench）。
4. **融合 carbon-aware scheduling**：结合电网碳强度波动，实现“绿色时段优先计算”。
5. **探索 domain-specific 路由器训练**：针对医疗、法律等领域定制分类器。

---

## ✅ 总结一句话
> 本文首次在**完全本地化的开源 LLM 环境下**证明：通过 **context-aware model switching** 架构，可在几乎不损失输出质量（93.6% BERTScore）的前提下，实现高达 **67.5% 的能耗削减** 和 **68% 的延迟下降**，为构建可持续、高效率的 AI 推理系统提供了实用且可扩展的技术路径。

</details>

---

### 9. [Energy Efficient Federated Learning with Hyperdimensional Computing (HDC)](https://arxiv.org/abs/2602.22290)

**Authors**: Yahao Ding, Yinchao Yang, Jiaxiang Wang, Zhonghao Liu, Zhaohui Yang, Mingzhe Chen, Mohammad Shikh-Bahaei  
**Category**: cs.DC  
**Published**: 2026-02-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.22290v1  

#### Abstract
This paper investigates the problem of minimizing total energy consumption for secure federated learning (FL) in wireless edge networks, a key paradigm for decentralized big data analytics. To tackle the high computational cost and privacy challenges of processing large-scale distributed data with c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Energy Efficient Federated Learning with Hyperdimensional Computing (HDC)》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**无线边缘网络中联邦学习（Federated Learning, FL）的高能耗与隐私泄露风险**两大核心挑战，提出了一种联合优化框架。具体而言：
- **计算与通信开销大**：传统基于 Neural Networks (NN) 的 FL 在资源受限的边缘设备上执行时，本地训练和模型传输消耗大量能量。
- **隐私保护不足**：标准 FL 中的模型更新可能被用于梯度反演攻击，暴露用户敏感数据。

### 🚀 提出的新方法与思路
作者提出了一个名为 **FL-HDC-DP** 的新型安全联邦学习框架，其核心创新包括：
- **融合 Hyperdimensional Computing (HDC)**：利用 HDC 的轻量级向量化操作（如 bundling、binding）替代复杂的 NN 训练，显著降低本地计算负担。
- **引入 Differential Privacy (DP)**：在上传前对本地 Associative Memory (AM) 添加高斯噪声，实现零集中差分隐私（zCDP），提供严格的数学隐私保障。
- **联合优化策略**：首次将 **HDC 维度 $d$**、**发射功率 $p_i$** 和 **CPU 频率 $f_i$** 进行统一建模与协同优化，以最小化总能耗。

### 🔍 相比现有方法的优势
| 方面 | 本文方法（FL-HDC-DP） | 现有方法 |
|------|------------------------|---------|
| **能效性** | 显著降低总能耗（最高达 83.3%） | 多数仅优化通信或计算单一维度 |
| **隐私保障** | 引入 zCDP，具备可证明隐私性 | 多为 heuristic 加密或弱 DP 机制 |
| **模型效率** | HDC 单次训练即可达到合理精度，支持快速收敛 | NN 需多次迭代，延迟高 |
| **系统适配性** | 特别适用于电池供电的 IoT/边缘设备 | 对硬件要求较高 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **MNIST** 手写数字识别数据集
- 每个用户持有独立同分布（IID）的子集，共 $U=50$ 用户，每人约 1200 个样本

### ⚙️ 实验设置
| 参数 | 设置值 |
|------|-------|
| 网络拓扑 | 单小区，基站位于中心，半径 500m 圆形区域 |
| 通信方式 | Frequency Division Multiple Access (FDMA) |
| 总带宽 $B$ | 1 MHz |
| 噪声谱密度 $N_0$ | -174 dBm/Hz |
| 最大 CPU 频率 $f_{\text{max}}$ | 2.3 GHz |
| 最大发射功率 $P_{\text{max}}$ | 0.1–1 W |
| 目标准确率 | 88% |
| 隐私预算 $(\epsilon, \delta)$ | (25, $10^{-5}$) |

### 📈 评估指标
- **总能耗 $E_{\text{total}}$**：所有用户的本地计算能耗 + 通信能耗之和
- **完成时间 $T$**：每轮最大执行时间约束
- **收敛轮数 $J_a$**：达到目标准确率所需的全局通信轮次
- **隐私保护水平**：通过 zCDP 保证

### 🔁 基线方法对比
1. **Fixed $f = f_{\text{max}}$**：固定 CPU 频率为最大值，仅优化其他变量
2. **Fixed $p = P_{\text{max}}$**：固定发射功率为最大值
3. **Fixed $d=3000$, $d=5000$**：固定 HDC 维度，不进行维度选择优化

---

## 3. 主要实验结果和性能指标

### 📌 关键性能数据
- **最优 HDC 维度**：$d^* = 4000$
- 在此维度下，收敛所需轮数从 $d=3000$ 时的 39 轮降至 **20 轮**
- 当 $d=10000$ 时，仅需 14 轮，但单轮能耗过高导致总体不经济

### 📊 与基线方法的对比结果（图2 & 图3）
| 对比项 | 能耗降低幅度 |
|--------|--------------|
| vs. Fixed $f = f_{\text{max}}$ | **83.3%** |
| vs. Fixed $p = P_{\text{max}}$ | **31.5%** |
| vs. Fixed $d=3000$ | **54.9%** |
| vs. Fixed $d=5000$ | **50.0%** |

> ✅ 结果表明：**维度选择是影响能耗最关键的变量**，远超功率或频率单独优化的效果。

### 🔍 消融实验分析（隐含于图2与表II）
虽然未明确标注“消融实验”，但从不同维度下的表现可得出以下结论：
- **低维 ($d=3000$)**：虽单轮成本低，但需更多轮次（39轮），累积能耗高
- **中维 ($d=4000$)**：取得最佳平衡——轮次减少至 20，且每轮开销可控
- **高维 ($d>5000$)**：尽管轮次继续下降，但每轮计算与传输负载剧增，反而推高总能耗

👉 表明存在一个 **“能耗拐点”**，并非维度越高越好。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **HDC 是实现节能 FL 的有效路径**：其轻量级运算特性非常适合边缘设备部署。
2. **联合优化 HDC 维度至关重要**：首次将 $d$ 作为优化变量纳入能耗模型，实验证明其对总能耗的影响最大。
3. **非单调关系存在**：总能耗与 HDC 维度之间呈 U 形曲线关系，存在最优维度（本实验中为 4000）。
4. **通信与计算需协同设计**：宽松的时间约束允许降低 $f_i$ 和 $p_i$，从而节省能量；反之则被迫高功耗运行。

### ⚠️ 方法的局限性
- **假设 IID 数据分布**：实际场景中可能存在 Non-IID 数据，影响模型聚合效果。
- **依赖 Monte Carlo 模拟获取 $J_a(d)$**：缺乏解析表达式描述维度与收敛轮数的关系，限制理论推广性。
- **未考虑 HDC 编码偏差问题**：超高维空间中的随机编码可能导致语义漂移。
- **zCDP 分析复杂**：隐私预算的累积分析较为繁琐，在多轮中难以动态调整。

### 🔮 未来工作方向
1. 将框架扩展至 **Non-IID 和异构设备环境**
2. 探索 **自适应维度选择机制**，根据数据特征动态调整 $d$
3. 结合 **split learning 或 model pruning** 进一步压缩通信开销
4. 开发 **专用 HDC 硬件加速器**，提升能效比（参考文献[10]）
5. 探索在 **6G 与大模型融合场景下的应用潜力**（参考文献[12]）

---

## 总结一句话
> 本文提出的 **FL-HDC-DP** 框架通过联合优化 HDC 维度、功率与频率，在保障隐私的前提下实现了高达 **83.3% 的能量节约**，为面向资源受限边缘设备的安全高效联邦学习提供了新范式。

</details>

---

### 10. [Multilingual Safety Alignment Via Sparse Weight Editing](https://arxiv.org/abs/2602.22554)

**Authors**: Jiaming Liang, Zhaoxin Wang, Handing Wang  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.22554v1  

#### Abstract
Large Language Models (LLMs) exhibit significant safety disparities across languages, with low-resource languages (LRLs) often bypassing safety guardrails established for high-resource languages (HRLs) like English. Existing solutions, such as multilingual supervised fine-tuning (SFT) or Reinforceme...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Multilingual Safety Alignment Via Sparse Weight Editing*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

大型语言模型（LLMs）在高资源语言（HRLs，如英语）中通常具备良好的安全对齐能力，但在低资源语言（LRLs）中却容易绕过安全防护机制，导致**跨语言安全不平等**（cross-lingual safety disparity）。  
现有方法如多语言监督微调（SFT）或基于人类反馈的强化学习（RLHF）依赖大量标注的安全数据，且计算成本高昂，难以扩展到多种语言。

### 🚀 提出的新方法与新思路

作者提出了一种**无需训练**（training-free）的跨语言安全对齐框架——**SPARSE WEIGHT EDITING**，其核心思想如下：

- **假设**：LLMs 的安全能力集中在少数“**Safety Neurons**”上，这些神经元在处理有害输入时表现出显著激活差异。
- **方法**：将跨语言安全对齐建模为一个**约束下的线性变换问题**，通过稀疏权重编辑（sparse weight editing），将 LRL 中有害输入的表示映射到 HRL（如英语）的安全子空间中。
- **闭式解**（closed-form solution）：推导出该优化问题的解析解，仅需少量锚点样本即可一次性计算修改矩阵 △W，无需梯度更新。

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **效率** | 无需微调，单次计算完成，节省大量计算资源 |
| **数据效率** | 仅需少量有害/无害锚点样本，不依赖大规模多语言安全数据集 |
| **通用性** | 可作为插件（plug-and-play）集成到不同架构的模型中 |
| **可解释性** | 明确作用于“Safety Neurons”，干预过程透明可控 |
| **兼容性** | 可与现有方法（如 MPO）组合使用，进一步提升安全性 |

---

## 2. 核心实验方法和设置

### 📚 数据集

- **MULTI-STRONGREJECT**：本文构建的多语言安全评测基准，由英文 `walledai/StrongREJECT` 经翻译生成，覆盖 **8 种语言**：
  - 英语（En）、中文（Zh）、越南语（Vi）、日语（Ja）、泰语（Th）、印尼语（Id）、孟加拉语（Bn）、希伯来语（He）
- 所有语言子集各含 **313 条有害查询**，用于零样本（zero-shot）评估。
- 对齐阶段使用的数据与评测数据完全隔离，确保严格零样本设定。

### ⚙️ 实验设置

- **模型家族**：
  - Llama-3.2（1B, 3B）
  - Qwen2 / Qwen2.5（0.5B ~ 7B）
- **目标**：验证方法在不同规模、不同预训练语料下的泛化能力。

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **ASR**（Attack Success Rate） | 主要安全指标，表示生成有害响应的比例（越低越好） |
| **MGSM**（Multilingual Grade School Math） | 多语言数学推理能力，衡量通用能力保留情况（越高越好） |
| **M-MMLU**（Multilingual Massive Multitask Language Understanding） | 多语言常识理解任务，评估知识保留能力 |

### 🔁 基线方法对比

| 方法 | 描述 |
|------|------|
| **None** | 原始未对齐模型 |
| **OUR** | 本文提出的 SPARSE WEIGHT EDITING 方法 |
| **MPO** | 当前先进的多语言安全对齐方法（Multilingual Preference Optimization） |
| **MPO + OUR** | 将本文方法作为插件与 MPO 结合使用 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & D）

- 在 **Qwen2-0.5B** 上：
  - 中文原始 ASR 高达 224/313（约 71.6%），应用 OUR 后降至 176/313（-48），结合 MPO+OUR 后进一步降至 **56/313**（-168），下降超过 **53 个百分点**。
- 在 **Llama-3.2-1B** 上：
  - 越南语原始 ASR 为 61/313，OUR 降低至 27/313（-34），MPO+OUR 达到 **22/313**。
- 平均改进（△Avg）：
  - 多数模型上，OUR 单独带来 **20–45 点 ASR 下降**；
  - MPO+OUR 组合普遍达到 **60–130 点以上下降**，显著优于单一方法。

> ✅ **所有语言中，尤其是 LRLs（如 Bengali, Thai, Hebrew）改善最为明显**

### 🔁 与基线方法对比

| 对比项 | 结果 |
|--------|------|
| **OUR vs None** | 在所有模型和语言上显著降低 ASR，尤其在小模型和 LRLs 上效果突出 |
| **OUR vs MPO** | OUR 在部分语言上略逊于 MPO，但无需训练，效率极高；两者性能接近 |
| **MPO + OUR** | **全面胜出**，几乎在所有设置下取得最低 ASR，说明本文方法是有效的“安全增强插件” |

### 🔍 消融实验结果

#### （1）安全神经元识别方式（Table 2）

- 替换为 NeuroStrike 的探针法选择神经元后，仍能实现显著 ASR 下降（从 28.27 → 14.93），表明本框架对神经元识别策略具有鲁棒性。

#### （2）锚点数据选择（Table 3）

- 必须同时使用 **UtilityAnchor**（无害数据）和 **Regular**（有害数据）才能平衡安全与效用。
- 仅用 UtilityAnchor 导致 MGSM 接近 0，严重损害实用性；
- 仅用 Regular 安全提升有限。
> ✅ **平衡的锚点构造至关重要**

#### （3）低秩参数 $ r $ 的影响（Table 4）

- 即使 $ r = 4 $ 或 $ 8 $，也能达到接近最优的 ASR 表现；
- 增大 $ r $ 对性能几乎没有提升，MGSM 和 M-MMLU 几乎不变。
> ✅ 支持“安全更新存在于低维子空间”的假设，且方法对 $ r $ 不敏感，易于部署。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Safety Neurons 存在且可迁移**：
   - 安全行为确实集中在稀疏神经元中，且英语中的 Safety Neurons 对其他语言有跨语言影响力。

2. **表示错位是 LRL 不安全的根本原因**：
   - 图 2 显示 LRL 与 HRL 的 Safety Neuron 集合重叠度低（Jaccard 相似性弱），导致简单放大无效。

3. **无需训练即可实现有效对齐**：
   - 通过闭式求解稀疏权重编辑，可在一次计算中完成跨语言安全对齐，性能媲美甚至超越训练型方法。

4. **轻量高效且可组合**：
   - 方法作为 post-hoc 插件，可无缝集成进已有对齐流程（如 MPO），提供额外安全保障。

### ⚠️ 局限性

- **依赖高质量翻译**：当前实验基于机器翻译构建多语言数据，若翻译失真可能影响对齐效果。
- **静态编辑**：权重修改是固定的，无法动态适应新型攻击或上下文变化。
- **局限于 MLP 层**：目前仅编辑 MLP 中的 up-proj/gate_proj，未考虑 Attention 或多层协同机制。

### 🔮 未来工作方向

1. **自动超参调节**：开发自适应策略选择 $ r $、正则化系数等。
2. **多层/层级编辑**：扩展至跨层 Safety Subspace 编辑，增强对抗 jailbreak 的鲁棒性。
3. **更强评估体系**：结合更复杂的多语言 evaluator 和多样化安全 taxonomy 进行分析。
4. **动态编辑机制**：探索运行时条件化权重调整，提升灵活性。

---

## 总结

> **SPARSE WEIGHT EDITING 是一种高效、轻量、无需训练的跨语言安全对齐新范式**。它利用“Safety Neurons”的稀疏性和可迁移性，通过闭式求解实现从英语到低资源语言的安全知识转移，在大幅降低 ASR 的同时几乎不影响 MGSM 和 M-MMLU，展现出极强的实用潜力。

</details>

---

### 11. [Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation](https://arxiv.org/abs/2602.23188)

**Authors**: Isma\"el Zighed, Andrea N\'ovoa, Luca Magri, Taraneh Sayadi  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.23188v1  

#### Abstract
We propose an efficient retraining strategy for a parameterized Reduced Order Model (ROM) that attains accuracy comparable to full retraining while requiring only a fraction of the computational time and relying solely on sparse observations of the full system. The architecture employs an encode-pro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**非定常流体系统**（unsteady flows）中的参数化 Reduced Order Models (ROMs) 在**跨参数域外推时预测精度下降**的问题。传统方法在模型泛化能力不足时需要重新训练整个模型，计算成本高昂，且通常依赖于高保真、全状态的数据进行 fine-tuning，这在实际应用中难以满足。

### **提出的新方法与新思路**
作者提出了一种**高效、实时的 ROM 自适应策略**，其核心思想是：
- 利用 **Variational Autoencoder (VAE)** 和 **Transformer** 构建一个**概率性、参数化的 ROM**（Probabilistic Parametric ROM），其中 VAE 负责降维，Transformer 利用 attention 机制捕捉时间依赖性和参数影响。
- 发现模型在 out-of-sample 参数区域（如高 Reynolds 数）性能下降的主要原因是**潜在流形（latent manifold）的几何失配**，而非潜在动力学（latent dynamics）本身的错误。
- 因此，提出只需**重新训练 VAE 编码器-解码器部分**，而**冻结 Transformer 动力学模块**，即可实现接近全模型重训练的性能提升。
- 进一步结合 **Ensemble Kalman Filter (EnKF)** 进行 **Data Assimilation**，利用**极稀疏观测数据**（仅占全状态空间 1%）重构完整状态轨迹，并用于 fine-tuning，显著降低对高保真数据的需求。

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **计算效率** | 从完整的 retraining 所需的约 **2 小时**缩短至 **~15 分钟**，甚至第一矩（均值）收敛可在 **30 秒内完成**，支持近实时自适应。 |
| **数据需求** | 仅需 **64 个传感器**（占总空间自由度 1%）的稀疏观测，无需全状态高保真数据。 |
| **方法通用性** | 结合了机器学习（DL）与数据同化（DA），形成闭环反馈，适用于参数变化下的动态系统建模。 |
| **不确定性量化** | 利用 VAE 的随机采样能力生成 ensemble，提供自然的 uncertainty quantification，支持 EnKF 框架下的统计融合。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 模拟二维不可压缩 Navier-Stokes 方程下绕障碍物（椭圆柱）的非定常流动。
- 使用 **Immersed Boundary Method solver (ibmos)** 生成训练数据。
- 空间网格为 `131 × 100`，每个时刻状态维度为 `13100`（含 u, v 速度分量），经下采样后输入维度为 `6550`。
- 时间序列长度：`T = 3033` 个快照。
- 参数范围：Reynolds 数 $ \text{Re} \in [80, 140] $，训练集中在 Re = 90 和 120，测试涵盖插值与外推场景（80–140）。

### **实验设置**
- **模型架构**：
  - **VAE**：将高维物理场映射到 4 维潜在空间。
  - **Transformer**：在潜在空间中自回归地演化状态，引入 cross-attention 以处理参数（Re）依赖。
- **训练策略**：
  - 初始训练在 Re=90 和 120 上完成。
  - 针对 Re=140（最差表现点）进行 fine-tuning。
- **fine-tuning 策略对比**：
  1. 全模型 retraining（Full retraining）
  2. 仅 retrain VAE（VAE-only retraining）
  3. 使用 EnKF 同化稀疏数据后 retrain VAE（VAE + DA）

### **评估指标**
| 指标 | 描述 |
|------|------|
| **2-Wasserstein Distance (Energy Distance)** | 衡量预测与真实动能信号之间的分布差异，对相位偏移不敏感，优于 MSE。 |
| **Relative L1 / L2 Reconstruction Error** | 评估 VAE 对瞬时流场的重建精度。 |
| **Uncertainty Quantification (UQ)** | 通过 ensemble variance 估计预测不确定性，验证其与实际误差的相关性。 |
| **Sensor Placement** | 使用基于 SVD 的 greedy QR pivoting 方法优化 64 个传感器位置，最大化模态信息捕获。 |

### **基线方法对比**
- **Baseline 1**：原始未 fine-tune 模型（Initial model）
- **Baseline 2**：全模型 retraining（Full model retraining）
- **Proposed**：仅 retrain VAE + 使用 EnKF 同化稀疏数据

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **VAE 重建误差显著下降**
| 方法 | Relative L1 Error | Relative L2 Error |
|------|-------------------|-------------------|
| Pre-retraining | 2.53% | 3.11% |
| Post-full retraining | 0.19% | 0.35% |
| Post-VAE-only retraining | 0.19% | 0.37% |
| Post-VAE + DA | 1.37% | 1.98% |

> ➤ **仅 retrain VAE 即可达到与全模型 retraining 几乎相同的重建精度**，证明潜在动力学稳定，无需更新 Transformer。

#### ✅ **能量距离（Energy Distance）大幅降低**
- 在 Re=140 上：
  - 初始模型：~0.005
  - 全模型 retraining：↓93%
  - VAE + DA：**↓70%**
- 使用仅 **1% 的观测数据**即实现接近最优性能。

#### ✅ **EnKF 显著减少预测误差**
- 对速度场 U 和 V 的 L2 误差分别减少：
  - U: ↓89.4%
  - V: ↓96.1%
- 总体误差下降 **93.8%**
- 相位和频率误差被有效消除。

#### ✅ **实时适应能力**
- 第一矩（均值）收敛时间：< **30 秒**
- 第二矩（方差/不确定性）收敛时间：~15 分钟
- 相比从头训练（约 10 小时），效率提升两个数量级。

#### ✅ **消融实验支持核心假设**
- **图8 & 图13** 显示：VAE retraining 成功修复了 latent manifold 的几何结构（周期轨道半径恢复正确）。
- **图9 & 图14** 显示：仅 retrain VAE 或使用 DA 数据 retrain VAE，均能显著降低 energy distance 和 uncertainty。
- **图15** 定量比较不同 fine-tuning 策略，表明 VAE + DA 在性能、数据量、时间成本之间取得最佳平衡。

---

## **4. 关键结论和发现**

### **主要发现**
1. 🔹 **主导误差源是 latent manifold 的几何失配**，而非 latent dynamics 错误 → 只需调整 VAE。
2. 🔹 **Transformer 学习的动力学具有强泛化性**，在相近参数区间内保持有效，无需频繁更新。
3. 🔹 **EnKF 可高效融合稀疏观测与 ROM ensemble 预测**，生成高质量 full-state 轨迹用于 retraining。
4. 🔹 **VAE 天然支持 Gaussian ensemble 输出**，满足 EnKF 对误差正态性的要求（K-S test 验证）。
5. 🔹 **即使使用 DA 生成的“噪声”数据 retrain，模型仍能保持鲁棒性**，且无需在损失函数中显式建模 second-moment 不确定性。

### **方法的局限性**
| 局限性 | 说明 |
|--------|------|
| **依赖平滑参数变化** | 若发生 bifurcation（如 Re 跨越临界值导致流态突变），latent dynamics 可能不再适用，需重新训练 Transformer。 |
| **潜在空间维度低** | 当前 latent dimension = 4，可能不足以表示更复杂流动结构。 |
| **EnKF 引入噪声** | 同化后的数据存在 noise，可能导致模型 confidence 下降，虽不影响预测均值，但影响 uncertainty calibration。 |
| **黑箱模型限制** | 无法直接嵌入物理约束或使用经典 smoother（如 RTS smoother）进行 post-processing。 |

### **未来工作方向**
1. **扩展至三维湍流或多物理场耦合系统**，验证方法 scalability。
2. **引入 physics-informed loss 或 hybrid modeling**，增强模型在大参数跳跃下的鲁棒性。
3. **设计专用 smoothing filter**，去除 EnKF 引入的 assimilation noise，进一步提升 fine-tuning 效果。
4. **在线 adaptive sensor placement**：根据当前 uncertainty 场动态调整传感器位置，实现主动感知。
5. **探索 diffusion models 或 normalizing flows 替代 VAE**，以获得更灵活的概率建模能力。

---

> 💡 **总结一句话**：  
> 本文提出了一种**基于 Data Assimilation 与 selective retraining 的高效 ROM 自适应框架**，通过识别并修正导致误差的 **latent manifold 偏移**，仅需 **1% 观测数据** 和 **分钟级计算时间** 即可实现接近全模型重训练的精度提升，为**实时流场预测与控制**提供了实用解决方案。

</details>

---

### 12. [Compress the Easy, Explore the Hard: Difficulty-Aware Entropy Regularization for Efficient LLM Reasoning](https://arxiv.org/abs/2602.22642)

**Authors**: Qin-Wen Luo, Sheng Ren, Xiang Chen, Rui Liu, Jun Fang, Naiqiang Tan, Sheng-Jun Huang  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.22642v1  

#### Abstract
Chain-of-Thought (CoT) has substantially empowered Large Language Models (LLMs) to tackle complex reasoning tasks, yet the verbose nature of explicit reasoning steps incurs prohibitive inference latency and computational costs, limiting real-world deployment. While existing compression methods - ran...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Compress the Easy, Explore the Hard: Difficulty-Aware Entropy Regularization for Efficient LLM Reasoning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Chain-of-Thought (CoT)** 的推理方法虽然提升了 LLM 在复杂任务上的表现，但其冗长的中间步骤导致显著的 **推理延迟** 和 **计算开销**。现有的 **reasoning compression** 方法（如 RL 中加入长度惩罚）往往通过牺牲模型探索能力来换取响应长度的缩短，从而引发 **entropy collapse** ——即策略熵快速下降，模型变得过于确定，难以发现正确的推理路径，尤其在困难问题上表现恶化。

### 🚀 提出的新方法：CEEH
作者提出 **CEEH**（Compress Easy, Explore Hard），一种**难度感知的熵正则化框架**，用于实现高效且准确的 LLM 推理压缩。其核心思想是：
> **对简单问题进行压缩，对困难问题保持探索**。

该方法包含两个关键组件：

1. **Difficulty-Aware Entropy Regularization（难度感知熵正则化）**
   - 动态估计每个问题的“难度”（基于历史准确率的非对称 EMA）
   - 对当前模型仍难以解决的问题（高难度）施加更强的熵正则化，鼓励多样化探索
   - 对已掌握的问题（低难度）放松正则化，允许更短、更确定的生成

2. **Dynamic Optimal-Length Penalty（动态最优长度惩罚）**
   - 跟踪每个问题历史上最短的**正确响应长度** $L_x$
   - 仅对正确回答施加长度惩罚：$R_{\text{len}} = \frac{L_x}{L_{x,y}}$，避免错误答案因长度短而获得奖励
   - 惩罚信号随训练动态更新，适应模型能力提升，防止长度膨胀

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | CEEH |
|------|--------|-------|
| **长度 vs 准确率权衡** | 优化长度常导致准确率下降 | 在压缩长度的同时维持甚至提升准确率 |
| **探索控制** | 统一熵正则化或无控制 | 难题保留探索，易题允许收敛 |
| **长度控制机制** | 固定目标长度或组内归一化 | 基于历史最优长度，信号稳定、自适应 |
| **训练稳定性** | 易出现 entropy collapse | 有效缓解熵崩溃，提升 Pass@k 性能 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练集**：从 `DeepMath-103K` 和 `DAPO` 各采样 2500 条，合并为 5000 条数学推理样本
- **测试集**：涵盖六项数学推理基准
  - `GSM8K`
  - `MATH500`
  - `AIME24`, `AIME25`
  - `AMC23`
  - `OlymBench`

### ⚙️ 实验设置
- **基础模型**：`R1-Distill-Qwen2.5-7B`（及 `1.5B` 版本用于对比）
- **训练框架**：`verl` 框架，采用 **Group Relative Policy Optimization (GRPO)** 进行无 critic 的 RL 训练
- **LoRA 微调**：降低显存开销
- **温度设置**：训练时 `temperature=0.6`，验证时相同
- **Rollout 数量**：训练每题 12 次，评估每题 16 次以计算 avg@16 和 Pass@k

### 📊 评估指标
| 指标 | 说明 |
|------|------|
| **ACC** | 平均准确率（avg@16） |
| **LEN** | 平均生成 token 数 |
| **NAG ↓** | Normalized Accuracy Gain，综合衡量 **单位长度缩减所牺牲的准确率**：<br>$ \text{NAG} = \left(1 - \frac{\text{Acc}}{\text{Acc}_b}\right) \times 100 / \sqrt{1 - \frac{L}{L_b}} $<br>越低越好（表示压缩效率高且精度损失小） |
| **Pass@k** | 多次采样中至少一次答对的概率，反映模型潜在推理能力上限 |

### 🆚 基线方法分类对比
| 类型 | 方法 |
|------|------|
| **Prompting-based** | ThinkSwitcher, Dynasor-CoT, DEER |
| **Offline Methods** | Spirit, ConCISE-SimPO, DAST |
| **Online RL Methods** | AutoThink, LC-R1, Length-Penalty* |
| **Instruction Models** | Qwen2.5-7B-Ins, Qwen2.5-Math |

> 注：`CEEH-EA` 使用 entropy-based advantage；`CEEH-ME` 使用 maximum-entropy loss

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & 2）

#### 在 `R1-Distill-Qwen2.5-7B` 上的表现（部分摘录）：

| Method | GSM8K ACC | GSM8K LEN | MATH500 ACC | MATH500 LEN | NAG ↓ |
|--------|-----------|------------|--------------|----------------|--------|
| Base Model | 91.3 | 3701 | 50.6 | 10382 | – |
| Length-Penalty* | 91.4 | 2696 | 50.2 | 9517 | ~0.0 |
| **CEEH-EA** | **91.7** | **2327** | **53.5** | **7543** | **-3.0** |
| **CEEH-ME** | 92.1 | **2170** | 53.8 | **6824** | **-3.7** |

> ✅ CEEH 在多个基准上实现了 **显著长度压缩**（减少 30%~50%），同时 **保持或提升准确率**

#### Pass@k 表现（Table 2）——体现推理潜力
| Model | GSM8K Pass@16 | MATH500 Pass@16 | AIME24 Pass@16 |
|-------|----------------|------------------|----------------|
| Base Model | 97.8 | 97.2 | 80.0 |
| Length-Penalty | 97.6 | 97.0 | 76.7 |
| **CEEH-EA** | **98.1** | 97.2 | 80.0 |
| **CEEH-ME** | **98.3** | 97.2 | **80.0** |

> ✅ CEEH **维持甚至提升 Pass@k**，表明其不仅没有削弱推理能力，反而增强了探索有效性

---

### 🔪 消融实验结果（Ablation Study）

#### （1）熵正则化形式的影响
- **CEEH-EA**（entropy-based advantage）倾向于给难题分配更多生成预算，有时达到最大长度仍未解出
- **CEEH-ME**（maximum-entropy loss）更紧凑，在平均长度上表现更优
- → EA 更鼓励探索，ME 更利于压缩

#### （2）长度惩罚系数的影响（Table 3）
- 增大长度惩罚系数可进一步缩短响应长度
- 即使在强惩罚下，**准确率仍能维持在 base model 水平**
- 表明 CEEH 成功引导模型剪除“冗余 fluff tokens”，保留关键推理步骤

#### （3）训练动态分析（Figure 3–6）
- **Policy Entropy**：CEEH 在训练初期维持更高熵，尤其在难题上，避免早期收敛
- **Length Penalty 影响**：传统方法中，增大长度惩罚会加速 entropy collapse；CEEH 缓解此现象
- **Training Accuracy**：CEEH 达到更高的训练准确率，显示其**数据利用效率更高**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Entropy Collapse 是推理压缩中的核心障碍**  
   单纯优化长度会导致策略熵迅速下降，限制模型对困难问题的探索能力，最终损害准确率。

2. **难度感知的探索策略至关重要**  
   “**Compress the Easy, Explore the Hard**” 的设计使得模型能够在不同难度实例间智能分配资源：易题求快，难题求稳。

3. **动态最优长度惩罚提升训练稳定性**  
   基于历史最短正确响应的长度奖励提供了**自适应、稳定的压缩信号**，即使在熵增加导致长度上升时也能有效引导。

4. **CEEH 实现真正的推理能力增强**  
   不仅压缩了输出，还提升了 Pass@k，说明其改善的是模型的**根本推理潜力**，而非仅仅提高置信度。

---

### ⚠️ 局限性
- 当前方法依赖于**每个问题的历史统计信息**（如 Acc_h, L_x），可能不适用于一次性或少样本场景
- 需要在训练过程中维护额外状态（question-level memory），增加了系统复杂性
- 对非常长的推理链（如 Olympiad 级别）是否依然有效尚需更大规模验证

---

### 🔮 未来工作方向
- 将难度估计机制泛化到未见过的问题，例如通过 prompt-level 或 domain-level 难度预测
- 结合 **test-time scaling** 策略，在推理阶段动态调整探索强度
- 扩展至非数学类推理任务（如规划、代码生成、多模态推理）
- 探索更轻量化的实现方式，便于部署到边缘设备

---

## ✅ 总结
CEEH 提出了一种新颖且有效的 **difficulty-aware** 思路，解决了传统推理压缩方法中“**压缩即退化**”的根本矛盾。通过 **选择性熵正则化 + 动态最优长度惩罚**，它实现了：
> **更短的响应长度 + 更高的准确率 + 更强的潜在推理能力（Pass@k）**

该工作为构建**高效、鲁棒、可扩展**的 LLM 推理系统提供了重要范式。

</details>

---

### 13. [Hypernetwork-based approach for grid-independent functional data clustering](https://arxiv.org/abs/2602.22823)

**Authors**: Anirudh Thatipelli, Ali Siahkoohi  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.22823v1  

#### Abstract
Functional data clustering is concerned with grouping functions that share similar structure, yet most existing methods implicitly operate on sampled grids, causing cluster assignments to depend on resolution, sampling density, or preprocessing choices rather than on the underlying functions themsel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Hypernetwork-based approach for grid-independent functional data clustering*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统**functional data clustering**方法通常依赖于函数在固定网格上的离散采样，导致聚类结果对**采样分辨率、网格密度和预处理方式**敏感。这种依赖性使得聚类分配不再是函数本身内在结构的反映，而是受到测量过程（即离散化）的影响，从而引入不必要的不确定性。

此外，现有方法存在以下限制：
- 多数基于**basis expansion**（如B-splines、functional PCA）的方法要求所有函数在**相同网格上观测**，需进行插值或平滑预处理；
- 非正交基会导致聚类结构失真（Tarpey & Kinateder, 2003）；
- 方法难以扩展到高维输入域（如图像 $\Omega \subset \mathbb{R}^2$ 或体数据 $\Omega \subset \mathbb{R}^3$），因基函数数量随维度指数增长；
- 最近学习型方法如**FAEclust**虽引入端到端训练，但仍受限于单变量输入域（$\Omega \subset \mathbb{R}$）且将表示学习与聚类目标耦合，更换聚类算法需重新训练。

### 提出的新方法与新思路
本文提出一种**基于hypernetwork的框架**，实现**与采样网格无关的功能性数据聚类**。其核心思想是：
- 将每个函数（无论其采样分辨率或网格如何）映射为一个**隐式神经表示**（Implicit Neural Representation, INR）的权重向量；
- 使用**hypernetwork**作为编码器，从任意离散坐标-值对 $(x, u(x))$ 中提取特征，并预测该INR的权重；
- 聚类直接在**INR权重空间**中进行，该空间是低维、连续且与采样无关的。

#### 关键组件设计：
- **Encoder**: 基于 **Bunker et al. [2025]** 的mesh-independent架构，使用per-point MLP + **mean pooling**聚合，保证对输入顺序和采样密度不变性；
- **Decoder**: 使用**SIREN**网络作为INR，因其能高效逼近复杂信号并具备良好的初始化特性；
- **Hypernetwork**: $H_\theta$ 将编码后的特征映射至完整SIREN权重空间，实现跨样本的参数共享与快速推理；
- **训练目标**: 仅使用**重建损失**（MSE），完全解耦表示学习与聚类任务。

### 相比现有方法的优势
| 特性 | 本方法 | 传统方法（如basis expansion） | FAEclust等学习方法 |
|------|--------|-------------------------------|---------------------|
| **网格独立性** | ✅ 支持异构网格、不同分辨率 | ❌ 需统一网格 | ❌ 需共享网格 |
| **输入域维度扩展性** | ✅ 支持 $\mathbb{R}^d \to \mathbb{R}^m$（如图像、体场） | ⚠️ 维度灾难 | ❌ 限于 $\mathbb{R} \to \mathbb{R}^m$（时间序列） |
| **聚类算法灵活性** | ✅ 可自由切换K-means/GMM/谱聚类等 | ✅（系数空间通用） | ❌ 表示与聚类目标绑定 |
| **无需预处理** | ✅ 原始坐标-值对直接输入 | ❌ 需插值/对齐 | ❌ 需统一采样 |
| **零样本泛化能力** | ✅ 在未见分辨率上表现稳健（通过multi-resolution training） | ❌ 性能随分辨率变化显著 | ⚠️ 泛化能力有限 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖多类型功能性数据，涵盖不同输入/输出维度：

| 数据集 | 类型 | 输入域 $\Omega$ | 输出维度 $m$ | 分辨率设置 |
|-------|------|------------------|---------------|------------|
| **MNIST** | 图像（灰度） | $\mathbb{R}^2$ | 1 | $r \in \{7,14,28,56,112\}$ |
| **Kvasir** | 医学图像（RGB） | $\mathbb{R}^2$ | 3 | $r \in \{32,64,128,256,512\}$ |
| **ERing** | 多元时间序列 | $\mathbb{R}^1$ | 4 | $t \in \{16,33,65,130,260\}$ |

> 所有数据均以无序的$(x, u(x))$点集形式表示，模拟真实不规则采样场景。

### 实验设置
- **训练策略**：采用**multi-resolution training**，每轮随机选择训练分辨率 $r \sim \mathcal{U}(R_{\text{train}})$，避免过拟合特定尺度；
- **测试设置**：评估包括**已见分辨率**和**未见分辨率**（held-out），验证泛化能力；
- **模型结构**：
  - Per-point MLP ($h^{(1)}$): 3层，宽64；
  - Weight predictor ($h^{(2)}$): 映射至4层SIREN权重；
  - 坐标使用**Random Fourier Features (RFF)** 编码；
- **优化器**：Adam，学习率从 $3\times10^{-4}$ 衰减至 $1\times10^{-4}$；
- **批次大小与epoch数**：按数据集调整（见附录Table 7）。

### 评估指标
- **Adjusted Mutual Information (AMI)**：衡量预测标签与真实标签的信息共享程度；
- **Adjusted Rand Index (ARI)**：衡量样本对是否被一致分组；
- 两者均经“校正随机”，期望得分为0，完美匹配为1；
- 所有结果报告5次随机种子的均值±标准差。

### 基线方法对比
- **直接聚类像素/时序值**：如对flatten pixel应用K-means（用于MNIST）；
- **FAEclust [Singh et al., 2025]**：最接近的SOTA方法，联合优化重建与聚类目标；
- 注意：由于FAEclust仅支持一维输入域，**只能在ERing数据集上进行公平比较**。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### MNIST 结果（K=10）
| Resolution | AMI | ARI |
|-----------|-----|-----|
| 14        | 0.720±0.014 | 0.604±0.035 |
| 28        | 0.724±0.013 | 0.606±0.034 |
| 56        | 0.723±0.014 | 0.607±0.034 |
| **112 (held-out)** | 0.723±0.014 | 0.607±0.034 |
| **7 (held-out)** | 0.578±0.008 | 0.456±0.034 |

> ✅ 在14–112范围内AMI/ARI高度稳定，表明**分辨率不变性**；  
> ⚠️ 7×7性能下降源于严重信息丢失（数字拓扑模糊），非模型缺陷。

#### Kvasir 结果（K=8）
| Resolution | AMI | ARI |
|-----------|-----|-----|
| 64        | 0.479±0.014 | 0.311±0.010 |
| 128       | 0.481±0.013 | 0.312±0.009 |
| 256       | 0.481±0.015 | 0.313±0.010 |
| **32 (held-out)** | 0.479±0.014 | 0.311±0.010 |
| **512 (held-out)** | 0.481±0.014 | 0.311±0.010 |

> ✅ 所有分辨率下性能几乎恒定，说明即使在含噪声、高变异性的医学图像中仍保持稳健。

#### ERing 结果（K=6）
| Resolution | AMI | ARI |
|-----------|-----|-----|
| 33        | 0.564±0.041 | 0.408±0.049 |
| 65        | 0.559±0.042 | 0.404±0.051 |
| 130       | 0.559±0.042 | 0.403±0.049 |
| **16 (held-out)** | 0.558±0.049 | 0.404±0.056 |
| **260 (held-out)** | 0.556±0.044 | 0.402±0.051 |

> ✅ 时间步长跨度达16倍仍保持稳定，验证**时间分辨率不变性**。

---

### 与基线方法对比

| 方法 | 数据集 | AMI | 备注 |
|------|--------|-----|------|
| **Flatten pixels + K-means** | MNIST | ~0.42 | 远低于本文（>0.72） |
| **FAEclust [2025]** | ERing | **0.664** | 更高，但使用聚类监督信号 |
| **本文（无监督重建）** | ERing | 0.564±0.041 | 未使用任何聚类标签 |

> 🔍 虽然FAEclust性能更高，但其优势来自**联合优化聚类目标**，牺牲了通用性和灵活性；而本文方法仅用重建损失即可达到良好效果，且适用于更广的数据类型（如图像）。

---

### 消融实验与算法兼容性验证（Appendices A–C）

使用**Gaussian Mixture Model (GMM)** 替代K-means，在三个数据集上测试：

| Dataset | Clustering Method | AMI (avg.) | ARI (avg.) |
|--------|--------------------|------------|------------|
| MNIST | K-means | ~0.72 | ~0.60 |
|         | GMM     | ~0.70 | ~0.59 |
| Kvasir | K-means | ~0.48 | ~0.31 |
|         | GMM     | ~0.47 | ~0.30 |
| ERing  | K-means | ~0.56 | ~0.40 |
|         | GMM     | ~0.56 | ~0.37 |

> ✅ GMM性能与K-means相当，证明**学到的weight space具有普适聚类结构**，可被多种算法利用，无需重新训练。

---

## 4. 关键结论和发现

### 主要发现
1. **聚类结果真正实现了网格独立性**：在各种分辨率（包括训练中未见者）下，AMI与ARI保持高度稳定，表明聚类依据的是函数的**本质几何结构**而非采样细节。
2. **INR权重空间是紧凑且信息丰富的表示空间**：少量参数即可编码整个函数，适合下游分析。
3. **multi-resolution training 是实现泛化的关键**：相比“zero-shot super-resolution”假设，显式地在多个分辨率上训练更能缓解分布偏移问题。
4. **表示学习与聚类任务成功解耦**：仅通过重建损失训练的表示，即可支持多种聚类算法，提升实用性与灵活性。

### 方法的局限性
1. **输出维度较高时权重向量膨胀**：当 $m$ 很大（如高光谱图像），最终层权重 $O(h \cdot m)$ 可能使weight space变得稀疏，影响聚类效果；
2. **未显式优化聚类分离度**：由于未引入聚类损失，类间距离可能不如FAEclust紧密；
3. **小样本挑战**：ERing仅有30个训练样本，尽管表现尚可，但在极低数据 regime 下仍有改进空间；
4. **SIREN激活函数的全局性**：正弦激活可能不利于捕捉局部突变（如边缘、瞬态事件）。

### 未来工作方向
1. **研究weight space的几何性质**：探索其曲率、连通性及语义邻近性与度量距离的关系；
2. **引入聚类感知正则项**：在重建损失基础上加入轻量级聚类友好约束，提升类内紧致性而不破坏通用性；
3. **支持在线/流式聚类**：利用hypernetwork的快速推断能力，实现实时嵌入与动态聚类更新；
4. **替换decoder为wavelet-based INR**：如使用**complex wavelets** [Roddenberry et al., 2024]，有望更好捕获局部特征；
5. **拓展至半监督/主动学习场景**：结合少量标签进一步提升性能。

---

> 📌 **总结一句话**：本文提出了一种**基于hypernetwork + INR的新型functional data clustering框架**，首次实现了**完全脱离采样网格依赖、支持高维输入域、且与聚类算法解耦**的聚类流程，在多个真实与合成数据上验证了其鲁棒性与泛化能力，为功能性数据分析提供了新的范式。

</details>

---

### 14. [Agentic AI for Intent-driven Optimization in Cell-free O-RAN](https://arxiv.org/abs/2602.22539)

**Authors**: Mohammad Hossein Shokouhi, Vincent W. S. Wong  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.22539v1  

#### Abstract
Agentic artificial intelligence (AI) is emerging as a key enabler for autonomous radio access networks (RANs), where multiple large language model (LLM)-based agents reason and collaborate to achieve operator-defined intents. The open RAN (O-RAN) architecture enables the deployment and coordination ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Agentic AI for Intent-driven Optimization in Cell-free O-RAN

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **cell-free O-RAN** 中复杂 operator intents（操作意图）难以被有效分解与协同执行的问题。现有研究大多假设 intent 可以分解为互不重叠的子任务，并由独立的 AI agents 处理，缺乏对需要多 agent 协作的复杂 intent 的支持。此外，部署多个独立的 LLM agents 导致内存开销大，限制了系统的可扩展性。

### 提出的新方法与创新思路
作者提出了一种基于 **agentic AI** 的 intent-driven 优化框架，用于 cell-free O-RAN 系统，其核心创新包括：

- **多智能体协作架构**：
  - 部署多个 LLM-based agents 在 **non-RT RIC** 和 **near-RT RIC** 上，分别处理不同时间尺度的任务。
  - **Supervisor Agent**（非实时）负责将自然语言形式的 operator intent 转化为具体的优化目标（如能效最大化）和约束（如最小速率要求）。
  - **User Weighting Agent**、**O-RU Management Agent** 和 **Monitoring Agent**（近实时）协同工作，联合决定 precoding 权重和 O-RU 激活状态。

- **跨 agent 协同机制**：
  - 引入 **monitoring agent** 实时监测用户速率，协调 user weighting 和 O-RU management agents，通过反馈循环确保最小速率约束满足，避免因参数更新不同步导致的震荡。

- **检索增强的经验复用机制**：
  - 设计了一个 **memory module**，利用 autoencoder 将环境特征编码为 embedding，并通过 **cosine similarity** 检索历史最优配置（如 αk, λk），加速新场景下的收敛。

- **轻量化与高可扩展性设计**：
  - 采用 **QLoRA**（Quantized Low-Rank Adaptation）技术，在 near-RT RIC 部署一个共享的轻量级 LLM，为每个 agent 训练独立的低秩适配器（adapter）。
  - 实现了 agent 功能专业化的同时，极大降低了内存占用。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **功能完整性** | 支持复杂 intent（如“节能同时保证特定用户最低速率”），需多 agent 协同完成；而现有工作多限于简单、独立子任务。 |
| **系统效率** | 通过 DRL + 迭代优化算法联合决策 O-RU 激活与 precoding，实现更优资源利用。 |
| **可扩展性** | 使用 QLoRA 共享 LLM 架构，相比部署多个独立 LLM，内存使用减少 **92%**。 |
| **响应速度与稳定性** | 检索增强机制避免重复搜索，加快收敛；monitoring agent 提供闭环控制，提升系统稳定性。 |

---

## 2. 核心实验方法和设置

### 实验环境与仿真设置
- **网络拓扑**：Cell-free O-RAN 系统，包含 **L = 50 个 O-RUs**，分布在 500m × 500m 区域。
- **用户数量**：K = 10 ~ 60 用户。
- **天线配置**：每个 O-RU 配备 $N_t = 4$ 天线，用户设备配备 $N_r = 2$ 天线，数据流数 $N_s = 2$。
- **服务策略**：每个用户最多由 $L_{\text{max}} = 8$ 个信道条件最好的 O-RUs 协同服务。
- **功率限制**：每个 O-RU 最大发射功率 $P_{\text{max}} = 30 \text{dBm}$。

### 评估指标
- **Active O-RU Ratio**：激活的 O-RU 数量占比，衡量节能效果。
- **Memory Usage**：near-RT RIC 中 agents 所需的总内存，评估系统可扩展性。
- **User Data Rate**：特别是关键用户的实际速率是否满足 intent 中的最小速率要求（如 user 3 ≥ 50 Mbps）。

### 基线方法对比
1. **Full-power Mode**：所有 O-RUs 始终激活，无节能机制。
2. **Greedy Algorithm**：
   - 对每个用户，优先激活信道增益最大的 O-RU；
   - 若未满足最小速率，则逐步激活次优 O-RU。
3. **DRL + Gradient Ascent (GA)**：
   - 使用相同 DRL 决策 O-RU 激活；
   - 但 user weighting 与 violation penalty 系数（$\mu_k$, $\lambda_k$）独立更新，**缺乏 monitoring agent 协调**。

### LLM 设置
- **Teacher Model**：GPT-4 via OpenAI API，用于生成训练数据。
- **Student Model**：Qwen 2.5，参数规模分别为 **7B** 和 **14B**。
- **Adapter 技术**：QLoRA，量化方式为 **NF4**（NormalFloat 4-bit），adapter rank 分别为 32/64，scaling 参数为 64/128。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **Active O-RU 减少比例** | 在节能模式下，相比三种基线平均减少 **41.93%** 的激活 O-RUs。 |
| **内存使用降低** | 使用 QLoRA 后，相比部署三个独立的 full-precision LLM，内存使用减少 **约 92%**。 |
| **模型大小影响** | 7B 与 14B Qwen 模型在性能上表现相近，说明小模型也能胜任任务。 |

### 与基线方法对比结果
- **图 3(a)(b)** 显示，在不同用户数和 O-RU 总数下，所提方法始终优于所有基线：
  - 比 **greedy algorithm** 更高效地选择 O-RUs，避免冗余激活。
  - 比 **DRL+GA** 更稳定：后者因缺乏协调，$\lambda_k$ 过快增长，导致过多 O-RUs 被激活以“保险”满足速率，反而降低节能效率。
  - 比 **full-power mode** 显著节能。

- **表 I** 内存对比表明：
  - 即使仅共享一个 FP16 LLM + LoRA adapters，内存也减少 **67%~75%**；
  - 加上 4-bit 量化后（FP4 + QLoRA），进一步降至 **92%** 减少。

### 消融实验与动态行为分析（见 Fig. 4）
- **intent 切换测试**（Fig. 4）验证了闭环控制有效性：
  - 当启用 **energy-saving intent** 时，O-RU management agent 关闭部分 O-RUs → user 3 速率下降；
  - monitoring agent 检测到违反约束 → 触发 user weighting agent 提高该用户优先级 $\alpha_3$；
  - 若仍不足，则触发 O-RU management agent 提高惩罚系数 $\lambda_3$，重新激活附近 O-RUs；
  - 最终稳定在满足约束且尽可能节能的状态。
- **记忆模块作用**：再次应用相同 intent 时，系统可直接检索历史最优参数，跳过探索阶段，显著加快响应。

---

## 4. 关键结论和发现

### 主要发现
1. **Agentic AI 是实现 intent-driven O-RAN 的可行路径**：通过多个 LLM agents 分工协作，能够将自然语言 intent 成功转化为底层无线资源配置动作。
2. **跨 agent 协同至关重要**：对于涉及多个目标（如节能 + QoS 保障）的复杂 intent，必须引入 **monitoring agent** 进行协调，否则易引发参数震荡与性能劣化。
3. **QLoRA 极大提升了部署可行性**：在 near-RT RIC 这类资源受限环境中，共享 LLM + QLoRA adapter 的方案可在保持性能的同时，将内存开销降至传统方案的 **8%**，极具实用价值。
4. **经验记忆机制加速收敛**：通过 embedding + cosine similarity 检索历史策略，实现了“类人类”的经验复用能力。

### 方法的局限性
- **依赖高质量 prompt engineering**：supervisor 与各 agent 的行为高度依赖 prompt 设计，鲁棒性有待在更多真实场景中验证。
- **DRL 训练成本较高**：尽管推理轻量，但 multi-agent DRL 的训练过程仍需大量仿真交互。
- **未考虑信令开销**：agent 间频繁通信（如 E2 接口）可能带来额外延迟，在 ultra-low-latency 场景中需进一步优化。
- **当前仅模拟验证**：尚未在真实 O-RAN 平台部署测试。

### 未来工作方向
- 引入更多类型的 agents，例如用于 **resource block allocation** 和 **channel estimation** 的专用 agent。
- 探索 **multi-modal intents**（如语音 + 文本）输入的支持。
- 研究 **federated learning** 框架下的分布式 agent 训练，保护数据隐私。
- 将框架扩展至支持 **URLLC** 和 **eMBB** 混合业务场景。

</details>

---

### 15. [Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization](https://arxiv.org/abs/2602.22675)

**Authors**: Qianben Chen, Tianrui Qin, King Zhu, Qiexiang Wang, Chengjun Yu, Shu Xu, Jiaqi Wu, Jiayu Zhang, Xinpeng Liu, Xin Gui, Jingyi Cao, Piaohong Wang, Dingfeng Shi, He Zhu, Tiannan Wang, Yuqing Wang, Maojia Song, Tianyu Zheng, Ge Zhang, Jian Yang, Jiaheng Liu, Minghao Liu, Yuchen Eleanor Jiang, Wangchunshu Zhou  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.22675v1  

#### Abstract
Recent deep research agents primarily improve performance by scaling reasoning depth, but this leads to high inference cost and latency in search-intensive scenarios. Moreover, generalization across heterogeneous research settings remains challenging. In this work, we propose \emph{Search More, Thin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前的 deep research agent 主要通过增加 **reasoning depth** 和工具调用次数来提升性能，但这带来了以下问题：
- **高推理成本与延迟**：长序列推理导致 inference latency 高，难以在实际场景中部署。
- **泛化能力差**：大多数 agent 在确定性问答任务（如 BrowseComp、GAIA）上表现良好，但在开放性研究任务（如 DeepResearch Bench）上表现不佳，反之亦然，缺乏跨任务类型的通用性。

### **提出的新方法：SMTL（Search More, Think Less）**
SMTL 是一种面向 **long-horizon agentic search** 的新框架，核心思想是“**多搜索，少思考**”，即减少模型内部的深度推理，转而通过并行化外部证据获取来提高效率和泛化能力。

#### **三大创新点**：
1. **并行化 agent 工作流（Parallel Agentic Workflow）**
   - 替代传统的 **sequential reasoning**，采用 **parallel evidence acquisition**。
   - 将任务分解为多个子任务（subtasks），并行执行 `web_search` 和 `crawl_page`。
   - 引入 **plan-driven context management**，定期进行 plan refinement，动态调整执行路径。
   - 支持在有限 context budget 下高效完成长周期任务。

2. **统一的数据合成管道（Unified Data Synthesis Pipeline）**
   - 构建了一个自动化 pipeline，生成涵盖 **deterministic QA** 和 **open-ended research** 的多样化任务。
   - 基于 **knowledge graph** 进行 subgraph extraction，构造多跳、拓扑丰富的任务结构。
   - 支持任务难度控制（depth/width）、信息密度优化，并引入 LLM-based verification 防止信息泄露。

3. **端到端训练策略**
   - 采用 **Supervised Fine-Tuning (SFT)** + **Reinforcement Learning (RL)** 联合训练。
   - RL 使用改进的 **RLOO (REINFORCE Leave-One-Out)** 算法，结合 token-level loss 和 sequence-level importance sampling，缓解训练-推理不一致问题。
   - 引入轨迹质量过滤机制，排除因环境错误或过长响应导致的负样本。

### **相比现有方法的优势**
| 维度 | SMTL | 传统方法（如 MiroThinker） |
|------|------|--------------------------|
| 推理模式 | 并行执行多个子任务 | 串行推理，单步单工具调用 |
| 效率 | 更少的 reasoning steps，更高的信息密度 | 步骤多，冗余查询频繁 |
| 泛化性 | 同一框架支持 QA 与开放研究 | 通常只擅长某一类任务 |
| 上下文管理 | 结构化 plan + overflow-triggered compression | 易受 context window 限制 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 类型 | 数据集 | 描述 |
|------|--------|------|
| **Deep Search** | BrowseComp, GAIA, XBench-DS, WebWalker-QA, FRAMES, SEAL-0 | 确定性问答任务，强调准确性和多跳检索能力 |
| **Deep Research** | DeepResearch Bench RACE | 开放性研究任务，评估报告的 comprehensiveness、insight、instruction following、readability |

### **实验设置**
- **主干模型**：Qwen3-30B-A3B-Instruct-2507
- **上下文长度**：训练时最大 65,536 tokens；推理时 128K tokens
- **交互步数限制**：默认 100 步（SMTL-100），扩展至 300 步（SMTL-300）
- **工具集**：
  - `web_search(query)`：基于 Serper API（Google 搜索）
  - `crawl_page(url, query)`：基于 Jina Reader API + DeepSeek-V3.2 摘要生成
- **计划刷新频率**：每 5 步进行一次 `plan refinement`

### **评估指标**
- **Deep Search**：`pass@1`（由 LLM-as-a-judge 判断语义等价）
- **Deep Research**：LLM-as-a-judge 多维度评分（满分 10 分制）：
  - Comprehensiveness（覆盖广度）
  - Insight/Depth（分析深度）
  - Instruction Following（指令遵循）
  - Readability（可读性）
- **效率指标**：平均 interaction steps、平均每步 tool calls 数量

### **基线方法对比**
| 类别 | 代表模型 |
|------|--------|
| **基础模型 + 工具** | GPT-5, Claude-4.5-Sonnet, DeepSeek-V3.2 |
| **商业研究系统** | OpenAI DeepResearch, Gemini DeepResearch, Perplexity Deep Research |
| **开源 agent 模型** | MiroThinker-v1.0, WebSailor-32B, WebShaper-32B, AFM-32B-RL, Tongyi-DeepResearch-30B |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1 和摘要）**

| 模型 | BrowseComp | GAIA | XBench-DS | WebWalker-QA | DeepResearch Bench (Overall) |
|------|------------|------|-----------|--------------|-------------------------------|
| **SMTL-100** | **43.6%** | 74.8% | **80.0%** | 74.9% | **45.9%** |
| **SMTL-300** | **48.6%** | **75.7%** | **82.0%** | **76.5%** | — |
| MiroThinker-v1.0 | 41.2% | 73.5% | 70.6% | 61.0% | — |
| Tongyi-DeepResearch | 43.4% | 70.9% | 75.0% | 72.2% | 45.7% |
| WebSailor-32B | 10.5% | 53.2% | 53.3% | 60.5% | 32.4% |

> ✅ SMTL 在多个 benchmark 上达到 **state-of-the-art 或领先水平**，尤其在长周期任务（如 BrowseComp）上优势明显。

### **与基线方法的对比结果**
- **效率显著提升**：
  - 在 BrowseComp 上，SMTL-100 相比 MiroThinker-v1.0 **减少 70.7% 的 reasoning steps**（60.4 vs 206.0），同时 **准确率更高**（44.6% vs 41.2%）。
  - SMTL 平均每步执行 **3.5 个 tool calls**，远高于传统方法的 1.0，实现“高信息密度”交互。
- **泛化能力强**：
  - 同一模型在 **deterministic QA** 和 **open-ended research** 上均表现优异，说明其具备强通用性。
  - 在 DeepResearch Bench 上，SMTL-100 以 **45.9% 总分**超越所有开源模型，甚至略超 Tongyi-DeepResearch（45.7%）。

### **消融实验结果**

#### **(1) 最大交互步数的影响（Figure 4a）**
- 成功案例的 median steps 不随 max steps 增加而增长，说明多数任务能在早期收敛。
- 失败案例的 median steps 与 max steps 基本重合，表明失败主因是 **预算耗尽**而非逻辑错误。
- 增加 max steps 可提升成功率，尤其是在困难任务上提供更多探索机会。

#### **(2) 检索宽度（top-k）的影响（Figure 4b）**
- 当 `top-k` 从 4 增加到 8 时，性能大幅提升（SMTL-300 从 43.8 → 47.0）。
- 继续增加到 20，性能趋于饱和，但仍有边际增益。
- 表明 **扩大检索宽度** 是比加深推理更高效的 scaling 方向。

#### **(3) 并行执行 vs 串行执行（Case Study in Figure 5）**
- SMTL 在 **8 步内定位关键实体**（Queen Marie of Romania），而 MiroThinker-v1.0 需要 **16 步以上**。
- SMTL 通过并行探索多个假设路径（如“blue eyes”、“lost child”、“married abroad”），快速收敛；后者逐条验证，效率低下。

---

## **4. 关键结论和发现**

### **主要发现**
1. **“Search More, Think Less” 是有效的设计范式**：
   - 减少模型内部的复杂推理，转而依赖 **并行化、高密度的外部证据获取**，能显著提升效率和性能。
   - **信息获取效率** 比 **推理深度** 更关键。

2. **并行执行 + 结构化计划管理 是长周期任务的关键**：
   - 通过 `plan → parallel execution → refine` 循环，实现动态适应和高效探索。
   - plan-centric context compression 有效缓解 context window 限制。

3. **统一数据管道支持跨任务泛化**：
   - 基于 knowledge graph 的合成 pipeline 可同时生成 QA 和 open-ended 任务，使单一 agent 具备广泛适用性。

4. **检索宽度比推理长度更具性价比**：
   - 增加 `top-k` 比增加 reasoning steps 更能提升性能，说明 **拓宽搜索视野** 比“想得更深”更重要。

### **方法的局限性**
- **对工具质量和覆盖率敏感**：若 `web_search` 返回结果不相关，即使并行也难弥补。
- **plan decomposition 质量依赖初始提示**：若初始 plan 错误，可能影响后续并行效果。
- **硬件资源要求较高**：并行执行多个 tool calls 对系统吞吐量有更高要求。

### **未来工作方向**
- 探索 **adaptive parallelism**：根据任务复杂度动态调整并行度。
- 引入 **multi-agent collaboration**：不同 agent 分工处理不同类型子任务。
- 构建更大规模的 **open-ended research dataset**，推动 agent 在真实科研场景中的应用。
- 研究 **energy-efficient agentic search**，进一步降低推理成本。

---

> **总结一句话**：  
> SMTL 证明了在 long-horizon agentic search 中，**通过并行化搜索和结构化计划管理，可以实现“更少思考、更多搜索”的高效智能体范式，在保持高性能的同时大幅降低推理开销，并具备出色的跨任务泛化能力**。

</details>

---

### 16. [Exploiting network topology in brain-scale simulations of spiking neural networks](https://arxiv.org/abs/2602.23274)

**Authors**: Melissa Lober, Markus Diesmann, Susanne Kunkel  
**Category**: cs.DC  
**Published**: 2026-02-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.23274v1  

#### Abstract
Simulation code for conventional supercomputers serves as a reference for neuromorphic computing systems. The present bottleneck of distributed large-scale spiking neuronal network simulations is the communication between compute nodes. Communication speed seems limited by the interconnect between t...

---

### 17. [FactGuard: Agentic Video Misinformation Detection via Reinforcement Learning](https://arxiv.org/abs/2602.22963)

**Authors**: Zehao Li, Hongwei Yu, Hao Jiang, Qiang Sheng, Yilong Xu, Baolong Bi, Yang Li, Zhenlong Yuan, Yujun Cai, Zhaoqi Wang  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.22963v1  

#### Abstract
Multimodal large language models (MLLMs) have substantially advanced video misinformation detection through unified multimodal reasoning, but they often rely on fixed-depth inference and place excessive trust in internally generated assumptions, particularly in scenarios where critical evidence is s...

---

### 18. [Effective QA-driven Annotation of Predicate-Argument Relations Across Languages](https://arxiv.org/abs/2602.22865)

**Authors**: Jonathan Davidov, Aviv Slobodkin, Shmuel Tomi Klein, Reut Tsarfaty, Ido Dagan, Ayal Klein  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.22865v1  

#### Abstract
Explicit representations of predicate-argument relations form the basis of interpretable semantic analysis, supporting reasoning, generation, and evaluation. However, attaining such semantic structures requires costly annotation efforts and has remained largely confined to English. We leverage the Q...

---

### 19. [Discourse-Aware Dual-Track Streaming Response for Low-Latency Spoken Dialogue Systems](https://arxiv.org/abs/2602.23266)

**Authors**: Siyuan Liu, Jiahui Xu, Feng Jiang, Kuang Wang, Zefeng Zhao, Chu-Ren Huang, Jinghang Gu, Changqing Yin, Haizhou Li  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.23266v1  

#### Abstract
Achieving human-like responsiveness is a critical yet challenging goal for cascaded spoken dialogue systems. Conventional ASR-LLM-TTS pipelines follow a strictly sequential paradigm, requiring complete transcription and full reasoning before speech synthesis can begin, which results in high response...

---

### 20. [Coarse-to-Fine Learning of Dynamic Causal Structures](https://arxiv.org/abs/2602.22532)

**Authors**: Dezhi Yang, Qiaoyu Tan, Carlotta Domeniconi, Jun Wang, Lizhen Cui, Guoxian Yu  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.22532v1  

#### Abstract
Learning the dynamic causal structure of time series is a challenging problem. Most existing approaches rely on distributional or structural invariance to uncover underlying causal dynamics, assuming stationary or partially stationary causality. However, these assumptions often conflict with the com...

---

### 21. [LUMOS: Democratizing SciML Workflows with L0-Regularized Learning for Unified Feature and Parameter Adaptation](https://arxiv.org/abs/2602.22537)

**Authors**: Shouwei Gao, Xu Zheng, Dongsheng Luo, Sheng Di, Wenqian Dong  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.22537v1  

#### Abstract
The rapid growth of scientific machine learning (SciML) has accelerated discovery across diverse domains, yet designing effective SciML models remains a challenging task. In practice, building such models often requires substantial prior knowledge and manual expertise, particularly in determining wh...

---

### 22. [Switch-Hurdle: A MoE Encoder with AR Hurdle Decoder for Intermittent Demand Forecasting](https://arxiv.org/abs/2602.22685)

**Authors**: Fabian Mu\c{s}at, Simona C\u{a}buz  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.22685v1  

#### Abstract
Intermittent demand, a pattern characterized by long sequences of zero sales punctuated by sporadic, non-zero values, poses a persistent challenge in retail and supply chain forecasting. Both traditional methods, such as ARIMA, exponential smoothing, or Croston variants, as well as modern neural arc...

---

### 23. [Doubly Adaptive Channel and Spatial Attention for Semantic Image Communication by IoT Devices](https://arxiv.org/abs/2602.22794)

**Authors**: Soroosh Miri, Sepehr Abolhasani, Shahrokh Farahmand, S. Mohammad Razavizadeh  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.22794v1  

#### Abstract
Internet of Things (IoT) networks face significant challenges such as limited communication bandwidth, constrained computational and energy resources, and highly dynamic wireless channel conditions. Utilization of deep neural networks (DNNs) combined with semantic communication has emerged as a prom...

---

### 24. [MSINO: Curvature-Aware Sobolev Optimization for Manifold Neural Networks](https://arxiv.org/abs/2602.22937)

**Authors**: Suresan Pareth  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.22937v1  

#### Abstract
We introduce Manifold Sobolev Informed Neural Optimization (MSINO), a curvature aware training framework for neural networks defined on Riemannian manifolds. The method replaces standard Euclidean derivative supervision with a covariant Sobolev loss that aligns gradients using parallel transport and...

---

### 25. [A Dataset is Worth 1 MB](https://arxiv.org/abs/2602.23358)

**Authors**: Elad Kimchi Shoshani, Leeyam Gabay, Yedid Hoshen  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.23358v1  

#### Abstract
A dataset server must often distribute the same large payload to many clients, incurring massive communication costs. Since clients frequently operate on diverse hardware and software frameworks, transmitting a pre-trained model is often infeasible; instead, agents require raw data to train their ow...

---

### 26. [Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search](https://arxiv.org/abs/2602.22983)

**Authors**: Xun Huang, Simeng Qin, Xiaoshuang Jia, Ranjie Duan, Huanqian Yan, Zhitao Zeng, Fei Yang, Yang Liu, Xiaojun Jia  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.22983v1  

#### Abstract
As Large Language Models (LLMs) are increasingly used, their security risks have drawn increasing attention. Existing research reveals that LLMs are highly susceptible to jailbreak attacks, with effectiveness varying across language contexts. This paper investigates the role of classical Chinese in ...

---

### 27. [Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design](https://arxiv.org/abs/2602.23092)

**Authors**: Zhuoliang Xie, Fei Liu, Zhenkun Wang, Qingfu Zhang  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.23092v1  

#### Abstract
The Capacitated Vehicle Routing Problem (CVRP), a fundamental combinatorial optimization challenge, focuses on optimizing fleet operations under vehicle capacity constraints. While extensively studied in operational research, the NP-hard nature of CVRP continues to pose significant computational cha...

---

### 28. [Toward Expert Investment Teams:A Multi-Agent LLM System with Fine-Grained Trading Tasks](https://arxiv.org/abs/2602.23330)

**Authors**: Kunihiro Miyazaki, Takanobu Kawahara, Stephen Roberts, Stefan Zohren  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.23330v1  

#### Abstract
The advancement of large language models (LLMs) has accelerated the development of autonomous financial trading systems. While mainstream approaches deploy multi-agent systems mimicking analyst and manager roles, they often rely on abstract instructions that overlook the intricacies of real-world wo...

---

### 29. [Assessing Deanonymization Risks with Stylometry-Assisted LLM Agent](https://arxiv.org/abs/2602.23079)

**Authors**: Boyang Zhang, Yang Zhang  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.23079v1  

#### Abstract
The rapid advancement of large language models (LLMs) has enabled powerful authorship inference capabilities, raising growing concerns about unintended deanonymization risks in textual data such as news articles. In this work, we introduce an LLM agent designed to evaluate and mitigate such risks th...

---

### 30. [FuxiShuffle: An Adaptive and Resilient Shuffle Service for Distributed Data Processing on Alibaba Cloud](https://arxiv.org/abs/2602.22580)

**Authors**: Yuhao Lin, Zhipeng Tang, Jiayan Tong, Junqing Xiao, Bin Lu, Yuhang Li, Chao Li, Zhiguo Zhang, Junhua Wang, Hao Luo, James Cheng, Chuang Hu, Jiawei Jiang, Xiao Yan  
**Category**: cs.DC  
**Published**: 2026-02-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.22580v1  

#### Abstract
Shuffle exchanges intermediate results between upstream and downstream operators in distributed data processing and is usually the bottleneck due to factors such as small random I/Os and network contention. Several systems have been designed to improve shuffle efficiency, but from our experiences of...

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

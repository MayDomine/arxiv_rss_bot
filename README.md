# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-12-04 08:59:15 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing](https://arxiv.org/abs/2512.03394)

**Authors**: Hamed Poursiami, Shay Snyder, Guojing Cong, Thomas Potok, Maryam Parsa  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2512.03394v1  

#### Abstract
Graph classification is a fundamental task in domains ranging from molecular property prediction to materials design. While graph neural networks (GNNs) achieve strong performance by learning expressive representations via message passing, they incur high computational costs, limiting their scalabil...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing — 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
图分类任务在分子性质预测、材料设计等领域至关重要。尽管 **Graph Neural Networks (GNNs)** 在表达能力和准确性上表现优异，但其依赖梯度反向传播和迭代优化，导致训练成本高、内存消耗大，难以部署在资源受限设备上。另一方面，**Hyperdimensional Computing (HDC)** 虽然计算高效、适合边缘和神经形态硬件，但在标准基准上的分类准确率通常显著低于 GNNs。

本文旨在解决这一 **效率与性能之间的权衡问题**：如何在保持 HDC 高效性的同时，提升其在图分类任务中的表达能力，使其接近甚至超越现代 GNNs 的性能。

### 提出的新方法与创新思路
作者提出 **VS-Graph**，一种基于 **Vector-Symbolic Architecture (VSA)** 的新型图学习框架，结合了 HDC 的轻量级特性与消息传递机制的结构表达力。其核心创新包括：

- **Spike Diffusion（脉冲扩散）**  
  一种拓扑驱动的节点识别机制：每个节点初始化一个单位“脉冲”，通过多跳传播形成扩散响应，据此对节点进行排序并赋予结构角色标识（rank）。该过程无需参数学习，完全由图拓扑决定，解决了不同图中节点匿名性和可比性问题。

- **Associative Message Passing（关联式消息传递）**  
  在超维空间内实现多跳邻域聚合。采用逻辑 OR 操作进行邻居信息聚合（`m^(l) = ∨_{j∈N(i)} h_j^(l)`），并通过残差混合更新节点表示：  
  `h_i^(l+1) = α·h_i^(l) + (1−α)·m_i^(l)`。整个过程在高维二进制空间中完成，无须梯度优化。

- **权重无关（weight-free）架构**  
  整个编码流程不涉及任何可训练参数或反向传播，仅依赖预定义的代数操作（binding, bundling, permutation），极大降低训练开销。

### 相比现有方法的优势
- **性能方面**：显著优于现有的 HDC 方法（如 GraphHD），并在多个数据集上达到或超过主流 GNNs（GCN/GAT/GIN）的精度。
- **效率方面**：训练速度比 GNNs 快达 **450×**，推理延迟极低。
- **鲁棒性方面**：即使将超向量维度从 D=8192 压缩至 D=128，准确率下降不到 1.5%，展现出极强的维度压缩容忍度，利于边缘部署。

---

## 2. 核心实验方法和设置

### 使用的数据集
所有实验基于 **TUDataset** 中的五个经典图分类基准，且仅使用图结构（无节点/边属性）：

| 数据集       | 图数量 | 类别数 | 平均节点数 | 平均边数 | 描述 |
|------------|-------|--------|-----------|---------|------|
| **MUTAG**   | 188   | 2      | 17.93     | 19.79   | 化合物致突变性 |
| **PTC_FM**  | 349   | 2      | 14.11     | 14.48   | 致癌性预测 |
| **PROTEINS**| 1113  | 2      | 39.06     | 72.82   | 蛋白质功能分类 |
| **DD**      | 1178  | 2      | 284.32    | 715.66  | 酶类蛋白结构 |
| **NCI1**    | 4110  | 2      | 29.87     | 32.30   | 抗癌化合物筛选 |

> 所有数据集均以 topology-only 形式处理，确保公平比较。

### 实验设置与评估指标
- **评估协议**：10折交叉验证，重复3次，分层采样（stratified splitting）
- **评估指标**：
  - 分类准确率（Accuracy）
  - 每张图平均训练时间（ms）
  - 每张图平均推理延迟（ms）
- **超向量维度范围**：2⁷ 到 2¹³（即 128 到 8192），HDC 方法统一测试不同维度下的表现
- **硬件环境**：NVIDIA Tesla T4 GPU，Intel Xeon 2.20GHz CPU，52GB RAM
- **实现工具**：PyTorch + DGL

### 基线方法对比
- **HDC 基线**：
  - **GraphHD**：当前最先进的 HDC 图分类方法，使用 PageRank 进行节点排名，并绑定边向量后捆绑成图嵌入。
- **GNN 基线**（均只用连接结构，无属性）：
  - **GCN**：基于归一化邻接矩阵的消息传递
  - **GAT**：引入注意力机制加权邻居
  - **GIN**：具有最强区分能力的 GNN，逼近 Weisfeiler-Lehman 测试

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Fig. 1 和 Tables）

#### ✅ 准确率表现（Accuracy）
| 方法 \ 数据集 | MUTAG | PTC_FM | PROTEINS | DD | NCI1 |
|-------------|-------|--------|----------|----|------|
| **VS-Graph (Ours)** | **90.8%** | 78.2% | **78.1%** | **79.8%** | 80.9% |
| GraphHD       | ~86%  | ~74%   | ~74%     | ~75% | ~79% |
| GCN           | 85.5% | 76.3%  | 74.2%    | 76.1% | 77.6% |
| GAT           | 84.7% | 75.8%  | 73.9%    | 75.3% | 78.4% |
| **GIN**         | 87.3% | **78.5%** | 76.8%    | 77.2% | **81.5%** |

> 注：具体数值未直接列出，但从 Fig. 1 推断得出趋势。

- VS-Graph 在 **MUTAG、PROTEINS、DD** 上 **全面超越所有 GNN 和 HDC 方法**
- 在 **PTC_FM** 上与 GIN 接近，略低于 GIN
- 在 **NCI1** 上稍逊于 GIN，但仍优于其他模型

👉 **相比 GraphHD 提升约 4–5%**

---

#### ⚡ 训练效率（Table II）
| 方法 \ 数据集 | MUTAG | PTC_FM | PROTEINS | DD | NCI1 |
|-------------|-------|--------|----------|----|------|
| **VS-Graph (Ours)** | **0.142 ms** | **0.153 ms** | **0.230 ms** | **2.120 ms** | **0.192 ms** |
| GraphHD       | 0.801 ms | 0.929 ms | 0.904 ms | 1.892 ms | 0.768 ms |
| GIN           | 47.20 ms | 37.39 ms | 49.24 ms | 72.92 ms | **85.84 ms** |

- VS-Graph 训练速度比 **GIN 快 250–450×**（尤其在 NCI1 上达 450×）
- 比 GraphHD 快 **5–6 倍**

---

#### 🕒 推理延迟（Table III）
| 方法 \ 数据集 | MUTAG | PTC_FM | PROTEINS | DD | NCI1 |
|-------------|-------|--------|----------|----|------|
| **VS-Graph (Ours)** | 0.366 ms | 0.368 ms | 0.418 ms | **2.288 ms** | 0.328 ms |
| GraphHD       | 0.980 ms | 1.143 ms | 1.050 ms | 2.028 ms | 0.923 ms |
| GIN           | 0.641 ms | 0.552 ms | 0.405 ms | 0.521 ms | 0.374 ms |

- 在小图上（MUTAG~NCI1）VS-Graph 推理速度优于大多数 GNNs 和 GraphHD
- 在大图 DD 上因多层消息传递带来一定开销（2.288ms），但仍可控

---

#### 🔍 维度压缩鲁棒性（Fig. 2 & Fig. 3）
- 当超向量维度从 **8192 压缩到 128**：
  - VS-Graph 准确率下降 < **1.5%**
  - GraphHD 下降明显（部分数据集降幅 >5%）
- 同时，随着维度降低，VS-Graph 的训练/推理时间进一步减少，在 DD 等大数据集上恢复效率优势

> 表明 VS-Graph 编码更具结构性分离能力，适合低维部署

---

## 4. 关键结论和发现

### 主要发现
1. **VS-Graph 成功弥合了 HDC 与 GNN 在图分类任务上的性能鸿沟**：通过 Spike Diffusion 和 Associative Message Passing，实现了接近甚至超越 GNN 的表达能力。
2. **无需梯度优化即可获得高性能**：整个模型为 weight-free 架构，训练仅为一次前向编码 + 原型构建，避免了复杂的反向传播。
3. **极致高效的训练速度**：相比 GNNs 平均加速 **250× 以上**，在 NCI1 上高达 **450×**，适用于大规模快速建模场景。
4. **卓越的维度压缩鲁棒性**：可在 D=128 下保持高精度，为边缘设备和神经形态芯片部署提供坚实基础。
5. **原型分类机制有效**：非参数化的 prototype matching 在结构信息丰富的情况下足以支持高质量分类决策。

### 方法的局限性
- 当前实验仅使用 **拓扑结构信息**，未利用节点/边特征（如原子类型、边类型等），可能限制其在真实化学或生物应用中的上限。
- 在最大规模图（如 DD）上，由于多跳消息传递层数增加，推理延迟相对较高，需借助维度压缩缓解。
- 对超参数（如扩散步数 K、消息传递层数 L、blend factor α）有一定敏感性，需调优。

### 未来工作方向
- 探索 **neuromorphic co-design**：针对 spiking hardware 或存内计算（in-memory computing）优化 VS-Graph 的执行流程。
- 开发专用 **spiking 和 in-memory 加速器**，充分发挥其脑启发式计算特性。
- 扩展至带属性图（attributed graphs）和异构图（heterogeneous graphs）。
- 将 VS-Graph 应用于更大规模的真实世界图任务（如药物发现、材料设计）。

---

> ✅ **总结一句话**：  
> VS-Graph 是首个在保持 HDC 极致效率的同时，在多个标准图分类任务上媲美甚至超越 GNN 的超维计算框架，为高效、可扩展、低功耗图学习提供了全新路径。

</details>

---

### 2. [OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference](https://arxiv.org/abs/2512.03927)

**Authors**: Liujianfu Wang, Yuyang Du, Yuchen Pan, Soung Chang Liew, Jiacheng Liu, Kexin Chen  
**Category**: cs.DC  
**Published**: 2025-12-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.03927v1  

#### Abstract
Mixture-of-Experts (MoE), while offering significant advantages as a Large Language Model (LLM) architecture, faces substantial challenges when deployed on low-cost edge devices with tight memory constraints. Expert offloading mitigates this issue by storing expert parameters in CPU memory and cachi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Mixture-of-Experts (MoE) 架构虽然在扩展大模型规模方面具有显著优势，但其对 GPU 内存的需求远高于 dense LLMs（约 4–5 倍），这使得在**低功耗边缘设备**上部署 MoE 模型面临巨大挑战。现有方法通过将专家参数卸载到 CPU 内存并缓存部分常用专家来缓解内存压力，但仍存在以下问题：

- **GPU 缓存占用高**：需要额外保留大量 GPU 显存用于专家缓存，限制了小显存设备的应用。
- **I/O 瓶颈严重**：从 CPU 向 GPU 加载专家引入显著延迟，尤其在边缘设备的 PCIe 总线下更为明显。
- **精度损失风险**：量化（quantization）或跳过不缓存的专家会导致模型性能下降。

### 提出的新方法与创新思路
本文提出 **OD-MoE** ——一种无需专家缓存的分布式 MoE 推理框架，实现完全按需加载（on-demand expert loading）。其核心创新包括：

#### （1）Scaled Emulative Prediction (SEP)：超精准多层前瞻预测器
- 利用一个轻量级、低精度的“影子模型”（shadow model，如 INT8/FP16 量化版）与主模型并行运行。
- 影子模型更快完成前向推理，提前展开后续多层的 expert activation 路由结果，作为主模型的预测依据。
- 这种“模拟仿真”式的预测机制相比传统基于当前层状态的启发式预测更准确。

#### （2）分布式节点间的并行加载与计算调度
- 将多个边缘节点划分为若干组，采用 **round-robin 调度策略**：
  - 一组节点执行当前层专家计算时，
  - 其他组节点同时预加载未来几层所需的专家。
- 实现 **expert loading 与 expert computation 的跨设备并行化**，大幅提升整体 CPU-GPU I/O 吞吐。

#### （3）KV Cache 与 Token 对齐机制
- 由于影子模型与主模型精度不同，在自回归生成过程中会产生输出差异，导致预测误差累积。
- 提出周期性地将影子模型的 KV cache 和生成 token 与主模型对齐，防止错误传播，保障长期预测稳定性。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **GPU 显存需求** | 无需专家缓存，worker 节点仅需 <1GB GPU 显存，适合低端 IoT 设备 |
| **推理速度** | 达到全 GPU 缓存方案的 **75% 解码速度**，远超其他 offloading 方法 |
| **模型保真度** | 不依赖量化或专家跳过，保持 full-precision 模型性能 |
| **硬件成本** | 所需总 GPU 显存仅为全缓存方案的 **1/3**，大幅降低部署成本 |

---

## 2. 核心实验方法和设置

### 数据集
- 主要测试数据来自 **Alpaca dataset** 的 60 个高质量样本（继承自 HOBBIT 工作）：
  - 30 个输入长度为 16 tokens
  - 30 个输入长度为 128 tokens
- 额外使用 **LongWriter dataset** 中 100 个 prompt 进行预测准确性评估（最大解码长度 512）

### 实验设置
- **基础模型**：Mixtral-8×7B（top-2 activation, 32 层 MoE）
- **硬件平台**：10 节点测试床
  - 1 个 main node（RTX 3090 + AMD R7-7700）
  - 1 个 shadow node（双 RTX 3090，运行 INT8/FP16 影子模型）
  - 8 个 worker nodes（各配 RTX 3090）
  - 所有节点通过 1Gbps Ethernet LAN 连接
- **评估阶段**：
  - **Prefilling 阶段**：衡量 Time-To-First-Token (TTFT)
  - **Decoding 阶段**：衡量平均解码吞吐量（tokens/s）
  - **整体推理速度**：综合 prefilling 与 decoding 的 output throughput

### 评估指标
| 指标类别 | 具体指标 |
|--------|---------|
| **推理性能** | Decoding Throughput (tokens/s), TTFT (ms), Output Throughput (tokens/s) |
| **预测准确性** | Recall Rate（正确预测的 expert 数 / 实际激活数） |
| **资源消耗** | 总 GPU 显存占用（GB） |
| **答案质量** | 多项基准测试得分（MMLU, GSM8k, BigBenchHard, TruthfulQA 等） |

### 基线方法对比
| 类型 | 基线系统 |
|------|----------|
| **全缓存参考** | HuggingFace Transformers（数据中心级）、llama.cpp（CPU-only） |
| **专家卸载系统** | Mixtral-Offloading, MoE-Infinity, HOBBIT, AdapMoE |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）专家激活预测准确率（Recall Rate）
| 方法 | Recall Rate |
|------|------------|
| **OD-MoE (SEP, FP16)** | **99.94%** ✅ |
| OD-MoE (INT8) | 97.34% |
| OD-MoE (NF4) | 95.67% |
| HOBBIT | 91% |
| AdapMoE | 86% |
| DAOP | 84% |

> ➤ OD-MoE 的 SEP 预测器实现了迄今为止最高的 expert activation 预测精度。

#### （2）推理速度对比（Decoding Throughput）
| 方法 | 平均解码速度 (tokens/s) | 相对于 Transformers 的比例 |
|------|------------------------|----------------------------|
| **Transformers（全 GPU 缓存）** | 4.8900 | 100% |
| **OD-MoE** | **3.6925** | **75.5%** ✅ |
| AdapMoE | 3.1300 | 64.0% |
| Mixtral-Offloading | 2.2375 | 45.8% |
| HOBBIT | 0.7850 | 16.1% |
| MoE-Infinity | 0.6875 | 14.1% |
| llama.cpp | 0.8225 | 16.8% |

> ➤ OD-MoE 在仅使用 1/3 GPU 显存的情况下，达到全缓存方案 75% 的速度，并超越所有 offloading 基线。

#### （3）端到端输出吞吐量（Output Throughput）
| 方法 | 平均 output throughput (tokens/s) |
|------|----------------------------------|
| **OD-MoE** | **3.3700** ✅ |
| Transformers | 4.8425 |
| AdapMoE | 3.0350 |
| Mixtral-Offloading | 2.1725 |
| HOBBIT | 0.7575 |
| MoE-Infinity | 0.6675 |

> ➤ OD-MoE 综合表现最优，比次优的 AdapMoE 快 **1.11×**

#### （4）GPU 显存占用
| 方法 | 总 GPU 显存（GB） | worker 单卡显存 |
|------|--------------------|----------------|
| Transformers | ~112 GB（8×3090） | >24 GB |
| **OD-MoE** | **60 GB**（主节点 7 + 影子节点 45 + 8 worker ×1） | **≤1 GB** ✅ |
| 其他 offloading 方法 | 通常单卡 ≥24 GB | 不支持分布式 |

> ➤ OD-MoE 显存效率极高，worker 节点可使用入门级 GPU 或嵌入式设备。

#### （5）答案质量（Answer Quality）
| 基准 | OD-MoE 表现 | 对比说明 |
|------|-------------|-----------|
| MMLU（通识） | 70.34% | 超过所有 offloading 方法，接近全精度 baseline |
| GSM8k（数学） | 64.14% | 显著优于 AdapMoE（22%）、HOBBIT（35%）等 |
| TruthfulQA（抗幻觉） | 89.00% | 最高分，表明 full-precision 优势 |
| MT-bench-101（指令遵循） | 7.83/10 | 优于所有 offloading 方法 |

> ➤ 因未使用量化或剪枝，OD-MoE 完美保留原始模型性能。

---

### 消融实验结果

#### （1）对齐机制的影响（Case 1–6）
| 设置 | 解码速度 (tokens/s) | 说明 |
|------|---------------------|------|
| Case 1（token + KV 对齐） | 3.616 | 最佳性能 |
| Case 2（仅 token 对齐） | 3.453 | 性能下降，标准差增大 |
| Case 3（仅 KV 对齐） | 2.445 | 效果有限 |
| Case 4（无对齐） | 1.185 | 预测错误累积严重 |
| Case 5（随机预加载） | 1.046 | 几乎无益处 |
| Case 6（无预测，按需加载） | 1.032 | 接近 I/O 瓶颈极限 |

> ➤ 验证了 SEP + 双重对齐机制的有效性。

#### （2）对齐周期优化
- 最优配置：**每轮迭代都进行 token 和 KV cache 对齐**（period=1）
- 更长周期会降低预测准确性，影响整体速度
- 在更换为 RTX 3080 后，最佳 KV 对齐周期变为 4，显示硬件相关性

---

## 4. 关键结论和发现

### 主要发现
1. **SEP 是目前最精确的 expert activation 预测器**，利用影子模型进行“行为模拟”，实现高达 **99.94% 的 recall rate**。
2. **通过分布式并行加载，可彻底消除专家缓存需求**，实现真正的 cacheless MoE 推理。
3. **OD-MoE 在极低 GPU 显存条件下仍能高效运行**，worker 节点仅需 **<1GB 显存**，适用于路由器、摄像头等 IoT 设备。
4. 在保持 full-precision 模型性能的前提下，**解码速度达到全缓存方案的 75%**，显著优于现有 offloading 方法（最高快 5.37×）。
5. **KV cache 与 token 对齐机制至关重要**，能有效抑制影子模型因量化带来的误差累积。

### 方法的局限性
1. **依赖高速局域网通信**：节点间频繁传输 embedding 和 KV cache，对网络带宽有一定要求（文中使用 1Gbps LAN）。
2. **影子节点显存开销较大**：shadow node 需要 45GB GPU 显存运行量化模型，可能成为瓶颈。
3. **对齐操作带来“晚出发”延迟**：每次对齐后影子模型需重新启动，影响初期预测可用性。
4. **prefilling 阶段未使用预测机制**：因 batch 内几乎所有专家都会被激活，故直接并行加载全部专家。

### 未来工作方向
1. **进一步压缩影子模型**：探索更小的 shadow model（如蒸馏模型）以减少 shadow node 资源占用。
2. **动态调整对齐频率**：根据上下文复杂度自适应选择对齐周期，平衡延迟与准确性。
3. **扩展至数据中心场景**：利用 SEP 的预测能力进行专家复制（replication）以缓解负载不均。
4. **支持更多 MoE 架构**：适配不同 top-k 策略或稀疏路由算法的模型。

---

> 🔗 **项目已开源**：https://github.com/Anonymous/DoubleBlind （发表后公开）

</details>

---

### 3. [From monoliths to modules: Decomposing transducers for efficient world modelling](https://arxiv.org/abs/2512.02193)

**Authors**: Alexander Boyd, Franz Nowak, David Hyland, Manuel Baltieri, Fernando E. Rosas  
**Category**: cs.AI  
**Published**: 2025-12-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.02193v1  

#### Abstract
World models have been recently proposed as sandbox environments in which AI agents can be trained and evaluated before deployment. Although realistic world models often have high computational demands, efficient modelling is usually possible by exploiting the fact that real-world scenarios tend to ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From monoliths to modules: Decomposing transducers for efficient world modelling*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文致力于解决**高效且可解释的世界模型（world models）构建**中的核心挑战。具体而言：

- **高维世界模型的计算复杂性**：现实世界环境通常由多个相互作用的子系统构成（如物理对象、社会互动等），若将其建模为单一的、高维的**monolithic transducer**（如POMDP的泛化形式），会导致学习和推理成本高昂，且难以理解其内部机制。
- **缺乏结构性透明度**：传统的端到端训练方式产生的模型往往是“黑箱”，不利于AI安全、可靠性和可审计性。
- **逆向分解问题**：虽然已有大量研究关注如何将简单模块组合成复杂系统（composition），但本文聚焦于**逆向问题**——给定一个看似纠缠的整体世界模型，如何将其**分解**（decompose）为具有明确功能划分的、可并行处理的子模块。

### **提出了什么新方法或新思路**

论文提出了一套基于**计算力学**（computational mechanics）和**信息论**的理论框架，用于对表示世界模型的**transducers**进行分解。其核心创新点包括：

1. **定义了两种信息论诊断工具**：
   - **Intransducibility**：当存在潜在变量（latent variables）时，用于检测一组可观测变量是否可以通过一个因果transducer从其余变量生成。若`f(X→Y|R) = 0`，则表明X可以经由潜态R被转导为Y。
   - **Acausality**：在仅有可观测信号的情况下，用于量化一个接口（interface）偏离前馈transducer实现的程度。若`AC[X→Y] = 0`，则该接口可由transducer实现。

2. **提出了两种分解算法**：
   - **基于Intransducibility的分解**（Algorithm 1）：利用潜变量信息，递归地“剥离”出最小的可被驱动的输出模块，从而将大transducer分解为稀疏网络。
   - **基于Acausality的分解**（Algorithm 2）：仅依赖可观测数据，通过检测非预期性（nonanticipatory）来推断变量间的因果流向，并构建模块化网络。

3. **建立了分解与粗粒化**（coarse-graining）
   - 提出了**多尺度视角**，允许在保留剩余节点接口的前提下，通过**条件化**（conditioning）上游节点或**边缘化**（marginalizing）下游节点来简化网络。
   - 这使得可以在不同抽象层次上操作，适应不同的预测或控制任务。

4. **证明了e-transducers的组合封闭性**（Theorem 2）：
   - 若两个组件transducers均为其对应接口的**最小预测模型**（即e-transducers），则它们的组合也是复合接口的e-transducer。
   - 这保证了模块化设计不会牺牲预测的最小性，为构建高效且最优的分层世界模型提供了理论基础。

### **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **可解释性** | 分解后的模块具有明确的功能语义（如“物体A的运动预测器”），提升了模型的**结构性透明度**，有利于AI安全分析。 |
| **计算效率** | 支持**分布式和并行化推理**（parallelizable inference）。每个子模块可独立学习和更新，避免了全局高维隐状态的联合推断。 |
| **理论严谨性** | 基于transducer和e-machine的成熟理论，提供了严格的数学定义和可证明的性质（如Theorem 1, 2），而非启发式方法。 |
| **通用性** | 框架不依赖于特定神经网络架构，适用于任何能表示为transducer的系统，包括但不限于RL环境、语言模型等。 |
| **与现有工作的关系** | 将automata理论中的**Krohn-Rhodes分解**思想推广到**随机、有输入-输出的动态系统**，并与**因果发现**（causal discovery）和**神经表征解耦**（disentangled representations）等领域建立联系。 |

---

## 2. 核心实验方法和设置

> **注意**：该论文是一篇**理论导向**的研究，**并未包含传统意义上的实验**（如在标准数据集上训练神经网络并报告精度）。其“验证”主要通过**数学证明、算法描述和概念性示例**完成。

### **使用的“数据”与方法**

- **数据形式**：假设可以获得过程的**联合概率分布** `Pr(X(J), R(J'))` 或可观测接口 `Pr(Y|X)`。
- **方法本质**：论文提出的是一种**分析性框架**，而非一个具体的机器学习算法。其“实验”体现在：
  - 定义了可计算的信息论量（Intransducibility 和 Acausality）。
  - 给出了递归分解算法（Algorithm 1 & 2）的伪代码。
  - 通过图示（如Fig. 9）和文字描述说明了算法如何运行。

### **评估指标（理论层面）**

- **分解正确性**：分解得到的子transducers能否精确重构原始系统的输入-输出行为（即保持相同的interface）。
- **模块独立性**：分解后的模块是否满足零Acausality，即可独立推理。
- **最小性**：分解是否达到了“素模块”（prime processes）级别，无法进一步分解。
- **计算可行性**：讨论了在有限时间、平稳性等假设下算法的可计算性。

### **基线方法对比**

论文未与具体基线模型进行数值对比，但在**讨论部分**（Section 7）提到了与以下领域的关联与区别：

- **Causal Discovery**（如PCMCI+）：本文方法不依赖Markov和faithfulness假设，能捕捉非马尔可夫依赖。
- **Neural Disentanglement**：本文提供了一个规范性的（normative）框架，解释为何生物或人工系统可能采用模块化潜空间。
- **Finite-State Transducer Composition**：本文的组合框架是文献中平行/串行组合的泛化。

---

## 3. 主要实验结果和性能指标

由于缺乏实证实验，此处列出的是**理论成果和关键发现**：

### **关键理论结果**

1. **Theorem 1**：一个接口是因果的（非预期的）当且仅当它有一个transducer表示。
2. **Proposition 1 & Corollary 1**：transducer组合对应的算子是Kronecker积，组合满足结合律但不满足交换律。
3. **Lemma 1 & 2**：Intransducibility和Acausality为零是存在有效transducer实现的充要条件。
4. **Theorem 2**：e-transducers在组合下是封闭的，即最小预测模型的组合仍是整体的最小预测模型。

### **与“基线”的对比（概念性）**

- 相比于训练一个单一的、高维的world model，本文方法能：
  - 将其分解为多个低维、功能独立的sub-transducers。
  - 实现**分布式信念更新**（distributed belief updates），每个模块只关心其局部历史。
  - 显著降低推理的计算负担。

### **消融实验**

论文未提供消融实验。但讨论了不同分解路径可能导致不同结果（因分解不一定唯一），以及Intransducibility/Acausality作为“余数”的角色，指导分解过程。

---

## 4. 关键结论和发现

### **主要发现**

1. **复杂世界模型可被结构性分解**：许多看似纠缠的高维transducer，只要其内部存在**条件独立性**或**稀疏因果结构**，就可以被分解为交互式的模块网络。
2. **分解带来双重好处**：
   - **认知益处**：暴露了世界的内在模块化结构，提升可解释性。
   - **计算益处**：支持并行化、分布式推理，提高效率。
3. **最小预测性可保持**：使用e-transducers作为模块，其组合仍是最小且最优的，实现了**效率与最优性的统一**。
4. **为AI安全提供新路径**：通过将世界模型分解为可审计的模块，有助于实现对AI系统的**机械性可解释性**（mechanistic interpretability）和**原则性控制**（principled control）。

### **方法的局限性**

1. **计算复杂性**：计算Intransducibility或Acausality需要长序列的联合分布，在实践中可能不可行，需依赖有限视界或变分近似。
2. **假设限制**：
   - 当前框架局限于**前馈**（feedforward）、**机理平稳**（mechanistically stationary）的transducers。
   - 未处理**反馈环**（feedback loops）、**非平稳性**（non-stationarity）和**自适应系统**。
3. **分解不唯一**：可能存在多种有效的分解方式，选择哪一种取决于具体应用目标。
4. **缺乏实证验证**：所有结论均在理论上成立，尚未在真实世界模型（如训练好的transformer或RL agent）上验证其有效性。

### **未来工作方向**

1. **扩展到反馈系统**：将框架推广至包含agent-environment闭环的感知-行动循环。
2. **开发实用算法**：设计基于采样数据估计Intransducibility/Acausality的方法，并应用于训练中的神经网络。
3. **实证研究**：在标准RL环境或大型语言模型上测试该框架，验证其能否成功分解出有意义的认知模块。
4. **与神经科学结合**：探索大脑是否以类似“可组合的transducers”方式组织其世界模型。
5. **集成到训练流程**：将分解思想作为正则化项或架构先验，引导模型学习内在模块化结构。

--- 

> **总结**：本文提出了一种将复杂世界模型从“单体”（monolith）转化为“模块”（modules）的**理论范式**。它通过引入Intransducibility和Acausality等新工具，为世界模型的**结构性分解**提供了严格的信息论基础，架起了**AI安全性**所需的透明性与**实际部署**所需的计算效率之间的桥梁。尽管尚缺实证，但其理论深度和前瞻性使其成为未来可解释、可信赖AI系统设计的重要基石。

</details>

---

### 4. [FFTrainer: Fast Failover in Large-Language Model Training with Almost-Free State Management](https://arxiv.org/abs/2512.03644)

**Authors**: Bohan Zhao, Yuanhong Wang, Chenglin Liu, Jiagi Pan, Guang Yang, Ruitao Liu, Tingrui Zhang, Kai Luo, Wei Xu  
**Category**: cs.DC  
**Published**: 2025-12-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.03644v1  

#### Abstract
Recent developments in large language models (LLMs) have introduced new requirements for efficient and robust training. As LLM clusters scale, node failures, lengthy recoveries, and bulky checkpoints erode efficiency. Infrequent asynchronous checkpoints trigger costly rollbacks, yet higher frequenci...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FFTrainer: Fast Failover in Large-Language Model Training with Almost-Free State Management

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模语言模型（LLM）训练中，随着集群规模扩大，节点故障频发且恢复时间长，导致训练效率严重下降。传统检查点（Checkpoint, CKPT）机制存在以下瓶颈：
- **高开销**：全量异步检查点占用大量存储、网络带宽和GPU内存拷贝开销；
- **低频率**：由于开销大，CKPT通常每30分钟一次，导致失败后需回滚大量迭代；
- **长恢复时间（MTTR）**：恢复过程涉及容器重建、依赖安装、通信组初始化、状态加载等多个串行步骤，耗时可达上千秒；
- **网络利用率低**：训练期间网络大部分时间空闲，但未被有效利用于状态管理。

这些问题共同导致高达43%的训练进度损失。

---

### 提出的新方法与核心思想
作者提出 **FFTrainer**，一个专为高效容错设计的LLM训练系统，其核心创新包括：

#### （1）**仅备份必要状态（Checkpoint Razor）**
- 利用数据并行中的冗余性（如相同模型权重、优化器状态），只保存每个数据并行组内的唯一状态；
- 将CKPT大小压缩至原来的1/10以下，实现“几乎零成本”的每轮次检查点（instant checkpointing）。

#### （2）**解耦训练角色与NCCL网络秩（Decoupling Roles from Ranks）**
- 在LCCL（Lightweight Collective Communication Library）中将逻辑训练角色 `(data_parallel_id, tensor_parallel_id, pipeline_parallel_id)` 与物理NCCL rank分离；
- 实现训练状态恢复与网络状态恢复的**并行化**，打破原有依赖链。

#### （3）**轻量级集体通信库 LCCL**
- 移除MPI风格的全局通信组管理和锁同步；
- 支持无锁连接建立和group-free通信（仅维护ring所需的两个邻居连接）；
- 加速跨节点重连，显著降低MTTR。

#### （4）利用空闲网络带宽进行状态传输（Free Transfer via Idle Network）
- 使用训练专用网络（而非独立的数据网络）在通信空闲期异步传输CKPT和训练数据；
- 通过优先级调度确保 `TRAIN` 流量（梯度/激活）始终优先，避免影响正常训练性能。

#### （5）快速故障检测与协同恢复机制**
- 引入跨层心跳信号（cross-layer heartbeat），由State Controller监控worker健康状态；
- 故障发生时可在数秒内触发恢复，远快于NCCL默认的10分钟超时机制。

---

### 相比现有方法的优势
| 维度 | 传统方案（如DeepSpeed/Gemini/MegaScale） | FFTrainer |
|------|----------------------------------------|---------|
| CKPT频率 | 每30分钟或更少 | **每轮次（per-iteration）** |
| CKPT开销 | 高（增加23%-110%迭代时间） | **<3%** |
| 恢复时间（MTTR） | 数百至上千秒 | **降至29秒以内（最高提速98%）** |
| 网络使用 | 单独存储网络或阻塞主网 | **复用训练网络，无额外硬件需求** |
| 恢复并行性 | 串行恢复流程 | **网络与训练状态恢复可重叠** |
| GPU利用率损失（MFU loss） | 最高可达19% | **最低至0.03%，接近零损失** |

此外，FFTrainer兼容PyTorch、Megatron、DeepSpeed等主流框架，集成仅需修改少量代码（约几十行），易于部署。

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **Common Crawl** 数据集进行预训练任务。

### 模型配置
测试四种不同规模的LLM：
| 模型 | 参数量 | 并行策略（d,p,t） |
|------|-------|------------------|
| GPT-2 | 2.7B | (16,2,4) |
| LLaMA3-8B | 8B | (4,8,4) |
| LLaMA2-13B | 13B | (4,8,4) |
| LLaMA3-70B | 70B | (2,8,8) |

其中 `d`: data parallel degree, `p`: pipeline parallel, `t`: tensor parallel.

### 实验平台
- **硬件环境**：16台服务器，共128个 NVIDIA RTX 4090 GPU；
- **互联网络**：200 Gbps InfiniBand；
- **软件栈**：CUDA 12.4, PyTorch, DeepSpeed, Megatron-LM。

### 评估指标
- **Per-iteration overhead**：启用CKPT后的单步训练延迟增加；
- **MTTR（Mean Time To Recover）**：从故障到完全恢复的时间；
- **MFU（Model FLOPs Utilization）loss**：因故障、恢复、回滚造成的有效计算利用率损失；
- **Main memory overhead**：用于CKPT的CPU内存消耗；
- **Allreduce延迟**：验证LCCL通信性能；
- **Scalability**：控制器在3万+ GPU下的可扩展性测试。

### 基线方法对比
- **Vanilla Megatron / DeepSpeed**：原生异步CKPT；
- **Gemini**：基于内存的快速CKPT系统，支持分钟级CKPT；
- **MegaScale**：专注于加速恢复流程的工业级系统；
- **TorchSnapshot**：PyTorch官方快照工具。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **CKPT频率** | 实现 **每轮次检查点（instant CKPT）** |
| **CKPT开销** | 所有模型下 **<3% 迭代时间增长** |
| **平均恢复时间（MTTR）** | 从 **~1000秒降至29秒**（减少97%） |
| **MFU损失** | 最低达 **0.03%**，相比Megatron降低近10倍 |
| **主存开销** | 仅为Megatron的38%，Gemini的19%（LLaMA3-8B场景） |
| **Allreduce效率** | LCCL达到NCCL的89%性能（2GB数据） |
| **网络初始化时间** | 128 GPU下LCCL比MegaScale快5.8倍（17%耗时） |

---

### 🔍 与基线方法对比结果

#### （1）CKPT开销对比（图4）
- **DeepSpeed/Megatron**：即使每5轮CKPT一次，仍带来 **23%~110%的训练延时**；
- **Gemini**：虽有优化但仍受限于完整状态复制；
- **FFTrainer**：**CKPT开销稳定在3%以内**，且频率更高（每轮 vs 每分钟）。

#### （2）恢复时间分解（表5，LLaMA2-13B）
| 步骤 | Gemini (128 GPU) | FFTrainer (128 GPU) | 减少比例 |
|------|------------------|--------------------|----------|
| 故障检测 | 15s | 6s | 60% |
| Pod创建 | 392s | 7s | 98% |
| 依赖安装 | 421s | 0s | 100% |
| 状态恢复与加载 | 166s | 16s | 90% |
| **总计** | **994s** | **29s** | **97%↓** |

> 注：FFTrainer通过预拉取镜像、免安装依赖、并行恢复等方式大幅压缩关键路径。

#### （3）MFU损失对比（图5）
- 当MTBF=3小时时：
  - Megatron：MFU损失 **~19%**
  - Gemini：**~6%**
  - MegaScale：**~12%**
  - **FFTrainer：仅0.15%**
- FFTrainer通过高频CKPT + 快速恢复，几乎消除rollback和recovery带来的损失。

#### （4）主存开销（图6）
- FFTrainer内存占用显著低于所有基线：
  - 对LLaMA3-8B：**仅Gemini的19%**，Megatron的38%；
- 得益于Checkpoint Razor对冗余状态的剔除。

#### （5）可扩展性测试（图10）
- 控制器处理32,768个worker的心跳和连接建立：
  - 心跳处理耗时：**19ms CPU时间**
  - 总连接建立时间：**14秒（接近线性扩展）**
- 表明中心化控制器不会成为瓶颈，支持 **30k+ GPU集群**。

---

### 🧪 消融实验分析（隐含在文中）

虽然没有显式列出消融实验表格，但从组件设计可推断各模块贡献：

| 模块 | 贡献 |
|------|------|
| Checkpoint Razor | 减少90%以上CKPT体积，是实现低开销的基础 |
| Neighboring Redundancy | 利用ring结构就近备份，避免磁盘I/O |
| Lazy Backup | 失败时才持久化冗余状态，进一步节省资源 |
| LCCL | 加速网络恢复，使MTTR从分钟级进入秒级 |
| Cross-layer Heartbeat | 将故障检测从10分钟缩短至6秒 |
| Pre-packaged Image | 消除依赖安装开销（421s → 0s） |

---

## 4. 关键结论和发现

### 主要发现
1. **LLM训练的MTTR瓶颈不在硬件，而在软件架构**：传统框架中恢复流程高度串行、依赖复杂，是拖慢恢复的根本原因。
2. **网络资源被严重低估**：现代训练网络利用率普遍低于3%，存在巨大潜力用于状态管理。
3. **冗余状态可安全剔除**：利用3D并行中的自然冗余，能极大压缩CKPT体积而不牺牲可靠性。
4. **高频CKPT + 快速恢复 = 接近零停机训练**：FFTrainer将MFU损失压至0.03%，使得即使每3小时一次故障也几乎不影响整体进度。
5. **集中式控制器仍具可扩展性**：合理设计下，轻量控制平面可支撑数万GPU规模。

---

### 方法的局限性
1. **假设fail-stop模型**：不处理拜占庭错误或静默故障；
2. **依赖稳定的RDMA环境**：当前实现基于InfiniBand/RDMA，可能难以直接迁移到纯TCP/IP环境；
3. **极端多点同时故障风险**：若整个数据并行组全部宕机，则无法从内存CKPT恢复（概率极低，<1.7%）；
4. **尚未支持动态扩缩容弹性训练**：聚焦于固定拓扑下的快速恢复。

---

### 未来工作方向
1. **引入Checkpoint压缩技术**：进一步减少内存和传输开销；
2. **探索更激进的懒加载与预测性恢复机制**；
3. **支持异构设备与混合精度下的自适应CKPT策略**；
4. **开放源码计划**：作者承诺论文发表后开源FFTrainer代码，推动社区应用。

---

## 总结
FFTrainer通过**细粒度状态管理、轻量化通信库、智能网络调度和架构解耦**，实现了LLM训练中近乎免费的状态管理与极速故障恢复。它不是简单优化某个环节，而是重新思考了分布式训练容错的整体范式，在不增加硬件成本的前提下，将恢复时间缩短98%，GPU利用率损失降至接近零，为未来更大规模、更高频率的LLM训练提供了坚实基础。

</details>

---

### 5. [UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs](https://arxiv.org/abs/2512.03383)

**Authors**: Hung-Yueh Chiang, Chi-Chih Chang, Yu-Chen Lu, Chien-Yu Lin, Kai-Chiang Wu, Mohamed S. Abdelfattah, Diana Marculescu  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2512.03383v1  

#### Abstract
Deploying large language model (LLM) models on mobile platforms faces significant challenges due to the limited memory and shared computational resources of the device. Resource availability may be an issue as it is directly impacted by the current device workload, adding to the uncertainty of model...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在边缘设备（如移动平台）上部署大型语言模型（LLMs）面临显著挑战，主要包括：
- **内存受限**：设备内存有限且被共享，难以容纳大规模模型。
- **动态资源竞争**：系统资源（如内存）受当前工作负载影响，可用性不确定。
- **固定压缩模型缺乏适应性**：传统的预压缩或量化模型大小固定，在高负载场景下可能无法运行。

为应对这些问题，现有方案如存储多个不同压缩率的模型副本、弹性训练等存在**时间成本高、存储开销大、依赖训练资源和特定数据集**等问题。

---

### 提出的新方法与创新思路
作者提出 **UniQL** —— 一种统一的**后训练量化（Post-Training Quantization, PTQ）与低秩压缩框架**，支持在设备端进行可配置的剪枝（pruning），实现对边缘LLM的自适应部署。

#### 核心创新点：
1. **统一的量化与低秩压缩框架（Unified Framework）**
   - 首次将量化与结构化剪枝以“一次性”（one-shot）方式结合，适用于多种架构：Transformers（如 Llama3）、State Space Models（SSMs，如 Mamba2）、混合模型（Hybrid Models，如 Nemotron-H 和 Bamba-v2）。
   - 支持在云端完成一次压缩流程，生成可在设备端灵活调整剪枝率的模型。

2. **高效的结构化权重排序算法（Structured Weight Sorting）**
   - 提出无需伪逆（pseudo-inverse-free）的矩阵分解方法，用于 MLP 层的通道重要性排序，避免了传统方法中计算复杂度高达 $O(n^3)$ 的 Moore-Penrose 逆矩阵运算。
   - 在 A6000 上处理 $14336 \times 14336$ 的相关矩阵时，相比 MoDeGPT 加速达 **20.58 分钟 → 0.19 分钟**，提速约 **108×**。

3. **量化感知的奇异值分解（Quantization-aware SVD）**
   - 在 MHSA 的 value-output 路径中引入 QSVD，将特征值 $\Sigma$ 融合到左奇异向量 $U$ 中，形成 $(U\Sigma)V$ 结构。
   - 这种设计使得每个列向量自带缩放因子，减少低比特（如 INT4）量化带来的误差，提升精度恢复能力。

4. **状态感知的权重排序（State-aware Weight Sorting）**
   - 针对 SSM 模块对状态矩阵敏感的特点，提出基于 SSM 内部状态 $H_t$ 的相关性分析来指导剪枝，有效缓解因剪枝导致的状态丢失问题。

5. **融合旋转位置嵌入核（Fused RoPE Kernel）**
   - 由于结构化剪枝会打乱原始位置索引，需重新收集 RoPE 的 sin/cos 索引。
   - 设计了一个融合内核（fused kernel），将索引查找与 RoPE 计算合并，减少内存访问开销，实测带来 **10% 延迟降低（1.1× 加速）**。

6. **掩码LoRA微调（Masked LoRA Fine-tuning）**
   - 在未剪枝的排序模型上进行微调，每步随机采样一个全局剪枝率 $P_t$ 并屏蔽对应通道。
   - 实现“一次微调，多级适配”，使最终模型能支持从 0% 到 35% 的任意剪枝率。

---

### 相比现有方法的优势
| 维度 | UniQL | 传统方法（如 MoDeGPT, SVD-LLM） |
|------|-------|-------------------------------|
| **压缩效率** | 一次压缩支持所有剪枝率（One-pass） | 每个剪枝率需单独压缩 |
| **计算开销** | 无伪逆，速度快（快 22×） | 依赖伪逆或多次SVD，耗时长 |
| **硬件兼容性** | 结构化剪枝 + 标准量化，通用性强 | 非结构化剪枝需专用硬件 |
| **模型泛化性** | 支持 Transformer / SSM / Hybrid | 多数仅针对单一架构 |
| **部署灵活性** | 边缘设备可动态调节剪枝率 | 固定模型大小 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 类型 | 数据集 | 用途 |
|------|--------|------|
| **校准集（Calibration）** | `wikitext2`, `alpaca` | 权重排序、层间剪枝分配、PTQ 校准 |
| **微调数据集** | `Alpaca`（51.8K 样本） | 掩码 LoRA 微调 |
| **下游任务评估** | `HellaSwag`, `PIQA`, `ARC-easy/challenge`, `WinoGrande` | 零样本准确率评估 |
| **扩展评估** | `MMLU`（57 主题）, `MBPP+`（编程任务） | 更广泛的能力测试 |

---

### 实验设置与评估指标

#### 模型范围
- **Transformers**: Llama-2-7B, Llama-3.1-8B, Qwen-2.5-7B
- **Hybrid Models**: Nemotron-H-8B, Bamba-9B-v2
- **SSM Models**: Mamba2-8B

#### 压缩配置
- **量化方式**：Group-wise INT4 对称量化（group size=128）
- **剪枝粒度**：结构化通道剪枝（structured channel pruning）
- **剪枝率范围**：0% ~ 35%
- **部署平台**：
  - 云侧：NVIDIA A6000（48GB）
  - 边缘侧：Jetson Orin Nano 8GB

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 下游任务平均准确率（zero-shot） |
| **Model Size / R.size** | 模型体积缩减倍数（vs FP16） |
| **Latency** | Time-to-last-token (TTLT), Time-per-output-token (TPOT) |
| **Throughput** | Token/s 吞吐量 |
| **Energy Efficiency** | Joules/request（Nano） 或 Tokens/GW·s（A6000） |
| **Compression Time** | 整体压缩流程耗时 |

---

### 基线方法对比
| 基线 | 类型 | 特点 |
|------|------|------|
| **MoDeGPT [Lin et al., 2025]** | 结构化剪枝 | 使用伪逆，不支持量化，单剪枝率 |
| **SVD-LLM [Wang et al., 2025b]** | SVD + 剪枝 | 需要微调，每次剪枝独立运行 |
| **TRT-AWQ [Lin et al., 2024a]** | PTQ（W4A16） | TensorRT 实现，embedding/output 层保留 FP16 |
| **TAO-HQQ [Badri & Shaji, 2023]** | PTQ（W4A16） | TorchAO 实现，同上 |
| **GPTQ [Frantar et al., 2023]** | PTQ | 权重量化基准，本文也作为增强 baseline |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **最大内存压缩比** | **4× – 5.7×**（vs FP16） |
| **吞吐提升** | **2.7× – 3.4×** token throughput |
| **边缘设备延迟降低** | Nano 8G 上 TPOT 最高提速 **2.1×**（vs TAO-HQQ） |
| **精度保持** | 在 15% 剪枝率下，精度损失 < 5%（相对原模型） |
| **压缩时间** | 仅需 **~7 小时** 完成全流程（vs MoDeGPT 7h+, SVD-LLM 16h+） |
| **支持剪枝率** | 0% – 35%，**一次压缩，任意选择** |

---

### 与基线方法的对比结果

#### ✅ 表 4 & 表 2：结构化剪枝 + 量化联合效果（One-pass 自适应剪枝）
| 方法 | 剪枝率 | R.size | Llama-3.1-8B 准确率 |
|------|--------|--------|---------------------|
| FP16 | 0% | 1× | 74.0% |
| SVD-LLM | 25% | 4.7× | 54.2% |
| **UniQL (Ours)** | **25%** | **5.3×** | **67.7%** |
| **UniQL (Ours)** | **35%** | **6.1×** | **62.7%** |

> ✅ 在同等压缩强度下，UniQL 显著优于 SVD-LLM，精度高出 **13.5 pp**

#### ✅ 表 3：纯 PTQ 性能对比（无剪枝）
| 方法 | Llama-3.1-8B 准确率 |
|------|--------------------|
| FP16 | 74.0% |
| TRT-AWQ | 71.9% |
| TAO-HQQ | 72.4% |
| **UniQL (Ours)** | **72.9%** |

> ✅ UniQL 在 PTQ 场景下达到 SOTA 水平，并额外支持剪枝

#### ✅ 表 6：模型大小对比（Llama-3.1-8B）
| 方法 | 模型大小 |
|------|---------|
| FP16 | 16.0 GB |
| TRT-AWQ | 5.8 GB |
| TAO-HQQ | 5.7 GB |
| **UniQL (Ours, 0%)** | **4.1 GB** |
| **UniQL (Ours, 35%)** | **2.8 GB** |

> ✅ 模型更小，因 embedding 和 output 层也量化至 4-bit

#### ✅ 表 7 & 8：延迟实测（A6000 / Nano 8G）
| 平台 | 方法 | TPOT (ms) | 提速比 |
|------|------|-----------|--------|
| A6000 | FP16 | 25.0 | 1.0× |
| A6000 | TAO-HQQ | 10.2 | 2.45× |
| A6000 | **UniQL (Ours)** | **9.0** | **2.78×** |
| Nano 8G | TAO-HQQ | 133.6 | 1.0× |
| Nano 8G | **UniQL (35%)** | **55.3** | **2.41×** |

> ✅ 显著优于主流生产级量化库

---

### 消融实验结果（Ablation Study）

#### 🔹 表 10：各组件贡献（Llama-3.1-8B, 25% 剪枝）
| 配置 | 准确率 |
|------|--------|
| FP16 + 剪枝 | 67.0% |
| + Masked LoRA FT | **69.6%** (+2.6%) |
| INT4 + 剪枝 | 65.0% |
| + Masked LoRA FT | 67.7% (+2.7%) |
| + Quantization-aware SVD | **67.7% → 60.2%**（无 QSVD 时仅 60.2%）→ **+7.5% 提升！** |

> ⚠️ **QSVD 是低比特下维持精度的关键**

#### 🔹 表 9：Fused RoPE 核性能影响
| 设置 | Llama-3.1-8B TPOT (ms) |
|------|------------------------|
| 无融合 | 9.9 |
| **有融合** | **9.0** |

> ✅ 融合 RoPE 内核实测加速 **10%**

---

## 4. 关键结论和发现

### 主要发现
1. **UniQL 实现了真正的“一次压缩，多级适配”**，解决了边缘设备资源动态变化下的模型部署难题。
2. **结构化剪枝 + 量化协同优化** 可显著提升压缩效率与推理速度，同时控制精度损失在 5% 以内。
3. **无需伪逆的高效排序算法** 极大降低了压缩时间，适合快速迭代与部署。
4. **量化感知 SVD 与状态感知排序** 是保证 SSM 和混合模型压缩稳定性的关键技术。
5. 在 **A6000 和 Jetson Nano** 上均验证了 **2.7×–3.4× 吞吐提升** 和 **4×–5.7× 存储节省**。

---

### 方法的局限性
1. **仍为后训练方法（PTQ）**：无法利用训练信号进一步优化表示，极限压缩下精度仍有下降。
2. **依赖校准集质量**：虽然使用 Alpaca 微调提升了鲁棒性，但在领域偏移场景下可能表现不稳定。
3. **未探索更低比特（如 INT3/INT2）**：尽管支持 3-bit（见附录 C.2），但主要用于验证可行性，未深入优化。
4. **仅支持结构化剪枝**：牺牲了一定压缩上限，换取通用性和硬件友好性。

---

### 未来工作方向
1. **扩展至更多架构**：如 RetNet、RWKV 等新型序列建模结构。
2. **支持动态比特宽度（Dynamic Bit-width）**：根据不同层或输入动态选择量化精度。
3. **结合轻量训练策略**：探索极低成本的在线微调机制，进一步提升压缩极限下的性能。
4. **跨设备协同压缩**：在边缘-云协同场景下实现分布式自适应压缩。
5. **安全与伦理考量**：随着边缘LLM普及，需加强对抗攻击防御与内容过滤机制。

---

> 📦 **代码与模型已开源**：  
> GitHub: [https://github.com/enyac-group/UniQL](https://github.com/enyac-group/UniQL)

</details>

---

### 6. [TokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving with Token Velocity](https://arxiv.org/abs/2512.03416)

**Authors**: Ruiqi Lai, Hongrui Liu, Chengzhi Lu, Zonghao Liu, Siyu Cao, Siyang Shao, Yixin Zhang, Luo Mai, Dmitrii Ustiugov  
**Category**: cs.DC  
**Published**: 2025-12-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2512.03416v1  

#### Abstract
The architectural shift to prefill/decode (PD) disaggregation in LLM serving improves resource utilization but struggles with the bursty nature of modern workloads. Existing autoscaling policies, often retrofitted from monolithic systems like those in AIBrix and DistServe, rely on lagging indicators...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving with Token Velocity

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现代大型语言模型（LLM）服务系统广泛采用 **Prefill/Decode (PD) disaggregation** 架构以优化资源利用率。然而，当前的自动扩缩容（autoscaling）策略大多沿用传统微服务或单体系统的滞后指标（如 GPU 利用率、请求速率 RPS），难以应对生产环境中高度突发（bursty）的工作负载。

这导致：
- 扩容反应迟缓，造成 **TTFT**（Time-to-First-Token）和 **TPOT**（Time-Per-Output-Token）SLO 违规；
- 被动扩容机制无法及时响应流量尖峰；
- 为避免违规而过度预置资源，增加成本。

### **提出了什么新方法或新思路**

论文提出 **TokenScale**，一个面向解耦式 LLM 服务的高效自动扩缩容框架，其两大核心创新是：

#### ✅ **1. Token Velocity（令牌速度）**
一种全新的、细粒度的、预测性的扩缩容指标，统一衡量 Prefill、Network 和 Decode 三个阶段的“处理能力”：
- **Prefill Velocity (Vp)**：Prefiller 处理输入 token 的最大速率（受限于 GPU compute）；
- **Network Velocity (VN)**：KV-Cache 在 Prefiller 和 Decoder 间传输的最大速率；
- **Decode Velocity (VD)**：Decoder 释放内存并生成输出 token 的速率（受限于 GPU memory）。

通过监控 **token 到达率** 与各阶段 **Token Velocity** 的比率，TokenScale 可在瓶颈发生前主动扩缩容，实现“前瞻性”调度。

#### ✅ **2. Convertible Decoders（可转换解码器）**
一种快速响应机制：部分 Decoder 实例被设计为可在突发时临时充当 Prefiller，利用其空闲的计算周期处理 Prefill 任务。
- 无需启动新实例，转换延迟 <1ms（仅需更新路由规则）；
- 采用 **SLO-aware restricted chunked-prefill** 策略，确保共存的 Decode 任务不违反 TPOT SLO；
- 形成弹性缓冲区，吸收突发流量，消除 Prefiller 启动延迟。

### **相比现有方法的优势**

| 方面 | 现有方法（AIBrix, DistServe, BlitzScale） | TokenScale |
|------|----------------------------------------|-----------|
| **扩缩容指标** | 请求计数（RPS）、并发数、GPU 利用率（滞后） | Token Velocity（细粒度、预测性） |
| **响应速度** | 滞后，依赖队列积压或利用率上升 | 前瞻性，基于 token 流速实时判断 |
| **突发处理** | 依赖冷启动新实例，延迟高（3–10s） | Convertible Decoders 快速吸收，无启动延迟 |
| **资源效率** | 易过量预置或响应不足 | 更精准匹配需求，降低成本 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **Azure 生产 traces** [35]：包含对话（Conversation）和代码生成（Code）场景的真实 LLM 推理日志；
- **BurstGPT traces** [38]：专为研究突发行为构建的数据集；
- **混合 trace**：将 Azure Conversation、Azure Code 和 BurstGPT 按等请求率混合，用于综合评估。

### **实验设置**

- **硬件平台**：
  - **A100 集群**：4 节点 × 4×A100-40GB，NVLink + InfiniBand；
  - **H100 集群**：2 节点 × 8×H100-80GB，更高带宽互联。
- **软件栈**：
  - 基于 **vLLM** 和 **LMCache** 实现 PD 解耦架构；
  - 控制平面用 Golang 实现，集成 Prometheus 监控。
- **模型**：
  - 小模型：**Llama-3.1-8B**（TP=1）
  - 大模型：**Qwen-2.5-32B**（TP=4）

### **评估指标**

- **SLO 达成率（SLO Attainment Rate）**：
  - **TTFT SLO**：短请求 <250ms，中请求 <400ms，长请求 <2000ms；
  - **TPOT SLO**：固定为 100ms。
- **资源成本**：使用的 GPU 数量；
- **吞吐量与延迟**：TTFT、TPOT、生成吞吐量；
- **Pearson 相关系数**：衡量扩容量与实际需求的相关性。

### **基线方法对比**

| 基线 | 扩缩容策略 |
|------|------------|
| **AIBrix** | Prefiller：基于并发数；Decoder：基于 GPU 内存利用率（70%阈值） |
| **BlitzScale** | Prefiller & Decoder：均基于 RPS，支持 live autoscaling（模拟理想情况） |
| **DistServe** | Prefiller & Decoder：基于 RPS，使用模拟器确定阈值 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 模型 | 基线 SLO 达成率 | TokenScale SLO 达成率 | GPU 成本降低 |
|------|----------------|------------------------|-------------|
| Llama-3.1-8B | 62–88% | **89–96%** | 6–13% |
| Qwen-2.5-32B | 50–87% | **78–91%** | 4–14% |
| H100 集群（Llama） | 43–77% | **85–98%** | 38–47% |

> **注**：SLO 达成率提升显著，尤其在大模型和突发场景下。

### **与基线方法的对比结果**

- **SLO 违规大幅减少**：
  - 在突发流量下（10×RPS 尖峰），TokenScale 的 TTFT 仅短暂上升至 ~50ms，迅速恢复；
  - 基线方法 TTFT 暴增至 1200–2300ms，恢复缓慢。
- **资源利用率更优**：
  - 基线（尤其是 AIBrix 和 BlitzScale）因并发控制反应慢，出现明显的“过扩-收缩”震荡；
  - TokenScale 扩容曲线更贴近真实需求，Pearson 相关系数最高（Prefiller: 0.63, Decoder: 0.44）。
- **H100 上优势更明显**：
  - 更强的 GPU 提供更多空闲算力，Convertible Decoders 可吸收更大突发，SLO 提升达 40% 以上。

### **消融实验结果（Ablation Study）**

逐步添加组件验证贡献：
1. **Baseline (DistServe)**：SLO 达成率 78%
2. **+ TokenScale Prefiller Scaler**：TTFT 达成率从 87% → 91%
3. **+ TokenScale Decoder Scaler**：TPOT 达成率从 80% → 99%，总达成率 → 90%
4. **+ Convertible Decoders**：TTFT 达成率进一步提升至 94%，总体 SLO 达成率最高

✅ 结论：三大组件协同作用，缺一不可。

---

## 4. 关键结论和发现

### **主要发现**

1. **Token Velocity 是有效的预测性指标**：
   - 能统一衡量 Prefill、Network、Decode 三阶段的瓶颈；
   - 相比请求级或利用率指标，能更早、更准地触发扩缩容。

2. **Convertible Decoders 显著提升突发响应能力**：
   - 利用 Decoder 的空闲算力作为“弹性缓冲”，避免冷启动延迟；
   - 通过受限的 chunked-prefill 策略保障 SLO 不被破坏。

3. **TokenScale 实现了“快且准”的扩缩容**：
   - 快：Convertible Decoders 实现毫秒级响应；
   - 准：Token Velocity 实现精准容量规划。

4. **在多种硬件和工作负载下均表现优异**：
   - 在 A100 和 H100 集群上均显著优于基线；
   - 对不同请求模式（长短输入/输出）具有鲁棒性。

### **方法的局限性**

- **依赖输出长度预测**：Decoder 扩容依赖 output predictor，虽然实验显示 85% 准确率已足够稳健，但在极端错误预测下仍可能导致轻微过配；
- **Convertible Decoder 数量需离线配置**：数量基于历史 burst ratio 预设，缺乏完全动态调整能力；
- **未考虑 Prefix Caching**：当前假设无 prefix caching，未来需与多级 KV-Cache 架构协同优化。

### **未来工作方向**

- **与 Hierarchical KV-Cache 架构联合设计**：将 Token Velocity 与多级缓存调度结合，进一步优化资源利用；
- **动态调整 Convertible Decoder 数量**：根据实时流量特征自适应增减；
- **扩展到多租户与多模型场景**：支持更复杂的生产部署环境；
- **探索 Token Velocity 在其他 AI 服务中的应用**：如扩散模型、推荐系统等。

---

> **总结**：TokenScale 通过 **Token Velocity** 和 **Convertible Decoders** 的协同设计，实现了对解耦式 LLM 服务系统的“及时且准确”的自动扩缩容，在 SLO 达成率和成本之间取得了显著优于现有方案的平衡，为高动态 LLM 工作负载下的高效资源管理提供了新范式。

</details>

---

### 7. [InvertiTune: High-Quality Data Synthesis for Cost-Effective Single-Shot Text-to-Knowledge Graph Generation](https://arxiv.org/abs/2512.03197)

**Authors**: Faezeh Faez, Marzieh S. Tahaei, Yaochen Hu, Ali Pourranjbar, Mahdi Biparva, Mark Coates, Yingxue Zhang  
**Category**: cs.CL  
**Published**: 2025-12-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.03197v1  

#### Abstract
Large Language Models (LLMs) have revolutionized the ability to understand and generate text, enabling significant progress in automatic knowledge graph construction from text (Text2KG). Many Text2KG methods, however, rely on iterative LLM prompting, making them computationally expensive and prone t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# InvertiTune: High-Quality Data Synthesis for Cost-Effective Single-Shot Text-to-Knowledge Graph Generation  
**论文核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Text2KG**（文本到知识图谱）方法大多依赖对 **LLMs** 进行迭代式提示（iterative prompting），这种方式存在以下问题：
- **计算成本高**：需要多次调用 LLM。
- **错误传播风险**：早期生成错误可能在后续步骤中被放大。
- **训练数据质量差**：现有数据集中的知识图谱（KG）通常规模小、关系简单，无法反映真实场景的复杂性。

此外，缺乏高质量、大规模的 `(text, KG)` 配对训练数据，限制了监督微调（SFT）等更高效范式的应用。

### 提出的新方法与思路
本文提出 **InvertiTune**，一种结合**可控数据生成管道**与**监督微调（SFT）** 的新型框架，其核心思想是“逆向构建”训练数据：

1. **数据生成管道（Inverted Process）**：
   - 不是从文本提取 KG，而是从大型知识库（如 Wikidata）中提取高质量子图 `g`。
   - 对子图进行噪声过滤（noise filtering）和语义一致性控制。
   - 利用 LLM 将子图 `g` 转换为自然语言描述 `T`，形成 `(T, g)` 训练样本。

2. **模型训练**：
   - 使用生成的数据集对轻量级 LLM（如 Qwen2.5-1.5B Instruct）进行 SFT。
   - 微调后的模型可在单次推理中完成从文本到 KG 的端到端生成，无需迭代提示。

### 相比现有方法的优势
| 维度 | InvertiTune | 传统 LLM 提示方法（如 PiVe, ChatGPT） |
|------|-------------|----------------------------------------|
| **效率** | 高（单次前向推理） | 低（多轮提示 + 验证） |
| **错误传播** | 无 | 存在 |
| **训练数据质量** | 高（结构化子图 + LLM 生成文本） | 低（多步提示合成，易引入噪声） |
| **可扩展性** | 可控生成任意规模数据 | 依赖人工标注或多步提示，成本高 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CE12k（本文提出）**：
  - 规模：12,000 个 `(text, KG)` 样本（训练集 10,800，测试集 1,200）。
  - 特点：平均每个 KG 包含 **24.12 条三元组**，远超现有数据集（见下表）。
  - 生成方式：通过 InvertiTune 管道从 Wikidata 提取子图并由 DeepSeek-V3 生成文本。
  - 公开地址：[https://huggingface.co/datasets/FaezehFaez/CE12k](https://huggingface.co/datasets/FaezehFaez/CE12k)

- **CrossEval-1200（本文提出）**：
  - 用于跨数据集泛化评估。
  - 构成：从 KELM、WebNLG+2020、GenWiki-HIQ 和 CE12k 各取 300 个测试样本，共 1,200。
  - 公开地址：[https://huggingface.co/datasets/FaezehFaez/CrossEval-1200](https://huggingface.co/datasets/FaezehFaez/CrossEval-1200)

- **对比基准数据集**：
  - KELM、WebNLG+2020、GenWiki-HIQ（均为现有主流 Text2KG 数据集）。

### 实验设置
- **模型架构**：
  - 主模型：Qwen2.5-1.5B Instruct（轻量级）。
  - 对比模型：Qwen2.5-32B Instruct（大模型）、ChatGPT。
- **训练方式**：Supervised Fine-Tuning（SFT）。
- **推理模式**：Single-shot（单次生成，不迭代）。

### 评估指标
| 指标 | 描述 |
|------|------|
| **G-BLEU (G-BL)** | 基于 BLEU 的图相似度，衡量预测三元组与真实三元组的 n-gram 匹配。 |
| **G-ROUGE (G-RO)** | 基于 ROUGE 的图相似度，强调召回率。 |
| **G-BERTScore (G-BS)** | 基于 BERTScore 的图对齐评分，语义敏感度更高。 |
| **BERTScore** | 当输出非结构化时，直接比较文本相似度。 |

所有指标均报告 **F1 分数**，并附带 **95% 置信区间** 和 **Wilcoxon signed-rank test p-value** 以验证显著性。

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **Non-LLM** | OpenIE6, DeepEx |
| **LLM-based（迭代式）** | PiVe, GraphRAG, LightRAG, ChatGPT |
| **SFT-based** | AutoRE（基于指令微调的文档级关系抽取） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（CE12k 测试集）
| Method | G-BLEU ↑ | G-ROUGE ↑ | G-BERTScore ↑ |
|--------|----------|-----------|----------------|
| OpenIE6 | 4.02 | 6.51 | 41.33 |
| DeepEx | 6.32 | 12.53 | 52.97 |
| PiVe | 39.04 | 48.34 | 75.06 |
| AutoRE | 26.31 | 30.83 | 67.14 |
| ChatGPT | 31.51 | 40.22 | 71.54 |
| **InvertiTune (Ours)** | **82.02** | **82.67** | **92.58** |

> ✅ 所有指标均**显著领先**，且 p-value 极小（~1e-196），统计显著。

### 参数效率分析（BERTScore）
| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Qwen2.5-1.5B Instruct | 81.98 | 82.92 | 82.43 |
| Qwen2.5-32B Instruct | 82.71 | 84.46 | 83.57 |
| **InvertiTune (1.5B)** | **95.77** | **95.70** | **95.73** |

> ✅ 即使是 1.5B 模型，在高质量数据上微调后，性能远超 32B 大模型。

### 跨数据集泛化能力（CrossEval-1200）
| Training Dataset | G-BLEU | G-ROUGE | G-BERTScore |
|------------------|--------|---------|------------|
| KELM | 48.21 | 52.61 | 79.21 |
| WebNLG+2020 | 38.36 | 43.55 | 79.11 |
| GenWiki-HIQ | 35.80 | 45.31 | 75.88 |
| **CE12k (Ours)** | **52.65** | **58.40** | **85.19** |

> ✅ 在分布外数据上表现最佳，说明 CE12k 数据更具多样性与泛化潜力。

### 数据集规模消融实验
- 实验：在 2K ~ 12K 样本上训练，观察性能变化。
- 发现：
  - 性能在约 **8K–10K 样本**时趋于饱和。
  - 达到接近最优性能所需样本数远少于现有数据集（如 KELM 有 60K 样本但平均仅 3.5 个三元组）。
- 结论：**数据质量比数量更重要**。

---

## 4. 关键结论和发现

### 主要发现
1. **高质量训练数据至关重要**：通过从知识库反向生成文本，能构建更真实、复杂的 `(text, KG)` 数据，显著提升 SFT 效果。
2. **轻量模型 + 高质量数据 > 大模型 + 低质量提示**：Qwen2.5-1.5B 在 CE12k 上微调后，性能远超 32B 模型和 ChatGPT。
3. **单次推理可行且高效**：InvertiTune 实现了高质量的 single-shot Text2KG，避免了迭代提示的成本与错误累积。
4. **更强的跨域泛化能力**：在 CrossEval-1200 上表现最佳，证明生成数据具有更好的分布鲁棒性。
5. **数据质量优于数量**：即使样本数较少，只要质量高，仍能达到优异性能。

### 方法的局限性
1. **模型架构受限**：实验仅使用 Qwen2.5-1.5B Instruct，未探索其他架构的影响。
2. **文本生成依赖单一 LLM**：仅使用 DeepSeek-V3 生成描述文本，不同 LLM 可能影响数据多样性与质量。
3. **领域覆盖有限**：初始实体限定为 `Human` 类别，可能限制知识图谱的多样性。

### 未来工作方向
- 探索更多样化的 LLM 用于文本生成。
- 扩展至其他实体类别（如 Disease、Company）以增强领域覆盖。
- 设计更复杂的图结构生成策略（如社区检测、时间演化）。
- 将该框架应用于其他结构化生成任务（如 Text2SQL、Text2Code）。

---

> **总结**：InvertiTune 通过“逆向生成”高质量训练数据，成功实现了**低成本、高性能、强泛化**的单次 Text2KG 生成，为知识图谱自动构建提供了一条高效且可扩展的新路径。

</details>

---

### 8. [A Preliminary Study on the Promises and Challenges of Native Top-$k$ Sparse Attention](https://arxiv.org/abs/2512.03494)

**Authors**: Di Xiu, Hongyin Tang, Bolin Rong, Lizhi Yan, Jingang Wang, Yifan Lu, Xunliang Cai  
**Category**: cs.CL  
**Published**: 2025-12-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.03494v1  

#### Abstract
Large Language Models (LLMs) are increasingly prevalent in the field of long-context modeling, however, their inference computational costs have become a critical bottleneck hindering the advancement of tasks such as agents and multimodal applications. This report conducts a preliminary investigatio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Preliminary Study on the Promises and Challenges of Native Top-k Sparse Attention

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文聚焦于 **Large Language Models (LLMs)** 在长上下文建模中的推理效率瓶颈问题。随着 agent、多模态等应用对超长上下文（如 512K tokens）的需求日益增长，传统的 **full attention** 推理计算开销巨大，导致延迟高、显存占用大，严重制约实际部署。

因此，论文旨在探索如何在不显著牺牲模型性能的前提下，通过 **Top-k Sparse Attention** 显著降低长上下文下的推理成本。

---

### 🚀 提出的新方法与新思路

1. **Exact Top-k Decoding 的有效性验证**  
   - 在解码阶段仅保留与当前 query 最相关的前 $k$ 个 key tokens 作为 context window，其余忽略。
   - 定义 **Top-k Ratio** $p = \frac{|K_{\text{top}}|}{N}$ 来衡量稀疏程度（$p=1$ 表示 full attention）。
   - 实验表明即使在极低 $p$（如 1%~5%）下，性能仍可媲美甚至超越 full attention。

2. **Native Top-k Attention Training（Top-k SFT）**  
   - 创新地提出应在训练阶段就引入 Top-k attention 操作，以实现训练-推理一致性。
   - 在 Llama-3-8B-ProLong-512k-Base 上进行 Supervised Fine-Tuning (SFT)，集成 Top-k Attention kernel（基于 FLASHATTENTION 修改），得到 **Llama-3-8B-ProLong-Instruct-512K-TopK-SFT** 模型。
   - 这是首次系统研究“原生”Top-k 训练对 Top-k 推理性能的影响。

3. **Approximate Top-k 精度影响分析**
   - 针对 exact Top-k 计算复杂度高的问题，探讨近似算法（如 ANN）的精度对下游任务的影响。
   - 引入 **Retrieval Precision** 指标衡量近似检索结果与真实 Top-k 的重合率。
   - 分析 DeepSeek-V3.2-Exp 中的 Lightning Indexer 的实际精度表现。

4. **从 Entropy 视角解释 Top-k 有效性的理论机制**
   - 提出假设：Top-k Decoding 更适合低熵任务环境。
   - 实验验证 Top-k SFT 模型在下游任务中表现出更低的 attention entropy，支持该假设。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| **训练-推理一致性** | 多数模型用 full attention 训练，却用 sparse 推理 → 不匹配 | 提出 native Top-k SFT，确保训练与推理一致 |
| **加速方式** | KV Cache 压缩、滑动窗口等 heuristic 方法 | 基于语义重要性选择 top-k keys，更具可解释性 |
| **理论支撑** | 缺乏对为何 sparse attention 有效的深入分析 | 从信息论角度（entropy）提供理论依据 |
| **实用性考量** | 忽视 exact Top-k 的高复杂度问题 | 系统评估 approximate Top-k 的精度-性能权衡 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **HELMET**：用于评估长上下文理解能力，包含多个子任务（如 `json_kv`, `niah_mk`, `infbench_qa` 等）
  - 包括 8K 和 128K 版本
- **LongBench v2**：综合性长文本基准，涵盖问答、摘要、推理等任务
- **MATH500**, **GSM8K**, **AIME24**：数学推理任务
- **TREC-Coarse/Fine**, **Banking77**, **HotpotQA**, **TriviaQA**：分类与开放域问答任务

---

### ⚙️ 实验设置

| 组件 | 设置说明 |
|------|----------|
| **基础模型** | Llama-3-8B-ProLong-512k-Instruct, Qwen3-32B |
| **Top-k SFT 模型** | 基于 Llama-3-8B-ProLong-512k-Base，在 SFT 阶段注入 Top-k Attention kernel |
| **Top-k Ratio ($p$)** | 测试范围：1%, 5%, 10%, ..., 100% |
| **Approximate Top-k 设置** | 固定 context window $W=2048$，控制 retrieval precision $p$（即 exact indices 占比） |
| **Prefill 阶段** | 使用标准 FLASHATTENTION（full attention） |
| **Decoding 阶段** | 应用 Top-k 或近似 Top-k attention |
| **评估指标** | 下游任务准确率（accuracy）、Retrieval Precision、Attention Entropy |

---

### 🔁 基线方法对比

- **Vanilla Full Attention**：原始 full attention 模型（如 Llama-3-8B-ProLong-Instruct）
- **Exact Top-k Decoding**：直接在 vanilla 模型上应用 Top-k 解码（未在训练中引入 Top-k）
- **Top-k SFT Model**：本文提出的原生 Top-k 训练模型
- **DeepSeek-V3.2-Exp + Lightning Indexer**：作为 approximate Top-k 的典型案例进行分析

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）Exact Top-k Decoding 性能（图1）

- 在 **Llama-3-8B-ProLong-Instruct-512K** 上：
  - 当 Top-k Ratio = 5% 时，在 HELMET-128K 上的整体性能接近 full attention；
  - 多数子任务（如 `Niah Mk 2`, `Infbench QA`）在 10% 以内即可达到 full attention 的 95%+ 性能。
- 在 **Qwen3-32B** 上（non-thinking mode）：
  - Top-k Ratio = 10% 时，LongBench v2 整体得分达 ~45.0，接近 full attention (~47.5)
  - “Hard” 类任务在低 $p$ 下反而略有提升，可能因噪声过滤效应

> ✅ 结论：**exact Top-k decoding 在极低稀疏比下仍能保持高性能**

---

#### （2）Native Top-k SFT 提升效果（图2）

- 在 HELMET-8K 上比较：
  - Vanilla 模型在 Top-k Ratio=1% 时性能急剧下降；
  - **Top-k SFT 模型在 Top-k Ratio=1% 时仍保持 >80% 准确率**，远优于 vanilla 模型；
  - 在多个 dataset（如 `Alce`, `Banking77`, `JSON-KV`）上均有明显增益。

> ✅ 结论：**训练阶段引入 Top-k attention 可显著增强模型对稀疏推理的适应能力**

---

#### （3）Approximate Top-k 精度影响（图3）

- 控制 retrieval precision $p \in [0,1]$，固定 $W=2048$
- 在 HELMET-128K 上观察到：
  - 随着 precision 提升，下游任务性能单调上升；
  - 当 precision ≥ 0.8 后趋于饱和；
  - precision < 0.6 时性能显著劣化。

> ✅ 结论：**approximate Top-k 的性能与其 retrieval fidelity 正相关**

---

#### （4）Lightning Indexer 精度评估（图4）

- 对 DeepSeek-V3.2-Exp 的 Lightning Indexer 在 HELMET-128K 上测试：
  - 平均 retrieval precision ≈ **60%**
  - 尽管精度不高，但由于其大规模参数优势，端到端性能依然领先
  - 不同 attention head 共享 KV tokens（MQA 架构），但仍能逼近 per-head Top-k 效果

> ✅ 结论：**可通过其他手段（如更大模型规模）补偿近似算法的精度损失**

---

#### （5）消融实验：Entropy 分析（图5）

- 比较 Top-k SFT 与 vanilla 模型的 attention entropy：
  - Top-k SFT 模型在大多数任务中呈现 **显著的 entropy reduction**（平均降低 10%~30%）
  - 特别是在 retrieval-heavy 任务（如 `Ruler-CWE`, `JSON-KV`）中更为明显
  - 支持“Top-k 更适合低熵任务”的假设

> ✅ 结论：**Top-k SFT 能使模型注意力分布更集中，提升决策确定性**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Top-k Decoding 是高效的长上下文加速策略**  
   即使只保留 1%-5% 的关键 keys，也能在多种 long-context benchmark 上达到 full attention 的性能水平。

2. **训练-推理一致性至关重要**  
   在 SFT 阶段引入 Top-k Attention 可显著提升模型在稀疏推理下的鲁棒性和性能，证明了 **native Top-k training 的必要性**。

3. **Approximate Top-k 的性能依赖于 retrieval precision**  
   retrieval precision 与下游任务性能呈正相关关系，建议设计高效且高保真的近似索引算法。

4. **Top-k SFT 导致 attention entropy 下降**  
   从信息论视角验证了 Top-k 机制更适合处理低熵任务，为 sparse attention 的有效性提供了理论支持。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **Exact Top-k 计算开销大** | 需频繁 CPU-GPU 数据搬运，难以与 FLASHATTENTION 完全兼容 |
| **Approximate 方法精度有限** | 如 Lightning Indexer 仅 ~60% 精度，可能导致关键信息遗漏 |
| **目前仅在 SFT 阶段尝试 Top-k training** | 更早阶段（如 continue pretraining）尚未充分探索 |
| **缺乏通用性验证** | 实验集中在 Llama/Qwen 系列，是否适用于所有架构待验证 |

---

### 🔮 未来工作方向

1. **开发 IO-efficient 的 exact Top-k kernel**  
   设计类似 FLASHATTENTION 的 tiled 计算方式，避免 materialize logits。

2. **探索更高效的 approximate Top-k 算法**  
   结合 ANN、LSH、Product Quantization 等技术构建高速高精度索引器。

3. **将 Top-k training 扩展至 pretraining 阶段**  
   在更大规模数据上训练 native sparse attention 模型。

4. **动态自适应 Top-k Ratio**  
   根据输入内容复杂度自动调整 $k$ 值，实现精度与速度的最优平衡。

5. **结合其他压缩技术（如 KV Cache quantization）**  
   与 Top-k 形成组合优化方案，进一步降低内存与计算需求。

---

## 总结

> 💡 本文是一篇关于 **Top-k Sparse Attention** 的前瞻性实证研究，系统验证了其在长上下文 LLM 推理中的巨大潜力，并首次提出 **native Top-k training** 的概念，揭示了训练-推理一致性的重要性。同时，从 **retrieval precision** 和 **entropy** 两个维度提供了深刻的理论洞察，为后续高效稀疏注意力机制的设计奠定了坚实基础。

</details>

---

### 9. [Contrastive Deep Learning for Variant Detection in Wastewater Genomic Sequencing](https://arxiv.org/abs/2512.03158)

**Authors**: Adele Chinda, Richmond Azumah, Hemanth Demakethepalli Venkateswara  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.03158v1  

#### Abstract
Wastewater-based genomic surveillance has emerged as a powerful tool for population-level viral monitoring, offering comprehensive insights into circulating viral variants across entire communities. However, this approach faces significant computational challenges stemming from high sequencing noise...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

**论文标题**: *Contrastive Deep Learning for Variant Detection in Wastewater Genomic Sequencing*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
废水基因组测序（Wastewater Genomic Sequencing）在病毒变异株监测中具有重要价值，但面临以下挑战：
- 高噪声、低病毒覆盖率、读段碎片化（fragmented reads）
- 缺乏标注的变异株标签（unlabeled variant annotations）
- 传统基于参考基因组的变异检测流程（如 BWA + LoFreq/iVar）难以识别新型突变，且计算开销大（每样本需数小时）

### **提出了什么新方法或新思路**
本文提出了一种**无监督的病毒变异株检测框架**，基于 **Vector-Quantized Variational Autoencoder (VQ-VAE)**，其核心创新包括：
1. **离散表示学习**：采用 VQ-VAE 学习基因组序列的离散 codebook 表示，避免连续 VAE 的 posterior collapse 问题，更适配基因组的离散突变特性。
2. **k-mer 分词预处理**：将原始序列转换为 k-mer token 序列（k=6），构建固定长度输入，提升模型对短读段的建模能力。
3. **掩码重建预训练（Masked Reconstruction Pretraining）**：借鉴 BERT 思想，在 20% 的 token 上进行掩码，增强模型对缺失/低质量数据的鲁棒性。
4. **对比微调（Contrastive Fine-tuning）**：通过 SimCLR 风格的对比学习，提升嵌入空间的判别能力，用于变异株聚类。

### **相比现有方法的优势**
- **无需参考基因组或变异标签**：实现真正的 reference-free 变异检测，可发现与已知序列差异大的新型变异株。
- **高效且可扩展**：单样本推理时间约 3 分钟（GPU），远快于 LoFreq（2 小时）或 iVar（1.5 小时）。
- **高重建精度与可解释性**：学习到的 codebook 可解释为“基因组词汇”，支持生物学分析。
- **统一生成与判别目标**：同时支持序列重建与变异聚类，适用于多种下游任务。

---

## 2. 核心实验方法和设置

### **使用的数据集**
- **SARS-CoV-2 废水测序数据**，共约 100,000 条高通量测序 reads。
- 读长范围：36–300 bp，中位长度 150 bp。
- 数据来源：市政污水处理厂。
- 预处理工具：Trimmomatic（参数：LEADING:3, TRAILING:3, SLIDINGWINDOW:4:15, MINLEN:36），FastQC 用于质量评估。

### **实验设置**
- **模型架构**：
  - **Encoder**：Token embedding (dim=128) + 2×Conv1D (kernel=3, hidden=256) + LayerNorm + Dropout(0.1) + Linear → $z_e \in \mathbb{R}^{L\times64}$
  - **Quantizer**：Codebook 大小 $K=512$，维度 $D=64$，使用 EMA 更新机制。
  - **Decoder**：对称结构，输出 vocabulary logits ($V=4097$)。
- **训练配置**：
  - Batch size: 32，Epochs: 50，Optimizer: AdamW，LR: 2e-4。
  - 损失函数：重建损失 + commitment loss + codebook entropy 正则化。
- **对比微调**：
  - 投影头（projection head）为 2 层 MLP，维度分别为 128→64→64 和 128→128→128，输出 64-dim 或 128-dim 归一化嵌入。
  - 增广策略：随机掩码（15%）、token dropout（10%）。
  - InfoNCE 损失，温度 $T=0.5$，训练 10 轮。

### **评估指标**
| 类别 | 指标 |
|------|------|
| **重建质量** | Token-level accuracy, Exact sequence match rate |
| **codebook 利用** | Active codes, Codebook utilization, Perplexity |
| **聚类性能** | Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index |
| **效率与实用性** | Time per sample, Reference-free 支持 |

### **基线方法对比**
| 方法 | Token Acc. (%) | Time/Sample | Reference-Free |
|------|----------------|-------------|----------------|
| Standard VAE | 96.8 | 2–3 min | √ |
| LoFreq [9] | ~95.2 | 2 hours | × |
| iVar [7] | 98.34 | 1.5 hours | × |
| K-mer counting | 96.48 | 1 min | √ |
| **VQ-VAE (Ours)** | **99.52** | **3 min** | **√** |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **(1) 基础 VQ-VAE 重建性能**
| 指标 | 数值 |
|------|------|
| Mean token accuracy | **99.52%** |
| Median token accuracy | 100.00% |
| Std. token accuracy | 0.69% |
| Exact sequence match rate | **56.33%** |
| Total sequences evaluated | 6,400 |

> 注：尽管 exact match 仅为 56.33%，但由于平均准确率极高，错误通常集中在 1–2 个 token，表明模型具备强重建能力。

#### **(2) Codebook 利用情况**
| 指标 | 数值 |
|------|------|
| Total codes | 512 |
| Active codes | **101** |
| Codebook utilization | **19.73%** |
| Perplexity | 52.3 |
| Max usage (single code) | 172,600 |

> 低利用率表明模型实现了高效压缩，仅用约 100 个 code 即可表达复杂变异模式，且 perplexity 显示有效码本大小约为 50。

#### **(3) 掩码重建鲁棒性**
- 在 20% token 被掩码的情况下，模型在**被掩码位置上的重建准确率仍达 ~95%**。
- 即使在 30% 掩码下，准确率仍保持在 88–90%，表现出**优雅降级（graceful degradation）**。

#### **(4) 对比微调对聚类性能的提升**
| 指标 | VQ-VAE | Contr-64 | Contr-128 |
|------|--------|----------|-----------|
| **Silhouette ↑** | 0.31 | **0.42 (+35%)** | **0.44 (+42%)** |
| **Davies-Bouldin ↓** | 1.68 | 1.34 (-20%) | 1.28 (-24%) |
| **Calinski-Harabasz ↑** | 1248 | 1876 (+50%) | 1972 (+58%) |

> 结果显示：**embedding 维度显著影响判别能力**，128-dim 表现最优。

### **与基线方法的对比结果**
- **重建精度最高**：99.52% > iVar (98.34%) > LoFreq (~95.2%) > Standard VAE (96.8%)。
- **速度最快之一**：3 分钟/样本，仅次于 k-mer counting (1 分钟)，远超 LoFreq/iVar。
- **唯一兼具高精度、高速度、reference-free 的方法**。

### **消融实验结果**
- **标准 VAE 出现严重 posterior collapse**，KL 散度趋近于零，导致 latent code 无效。
- **掩码预训练显著提升鲁棒性**，在 20% 掩码下仍保持 95% 准确率。
- **对比学习对所有聚类指标均有显著提升**，且效果随 embedding 维度增加而增强。

---

## 4. 关键结论和发现

### **主要发现**
1. **离散表示优于连续表示**：VQ-VAE 成功避免 posterior collapse，学习到高质量、可解释的基因组 codebook。
2. **codebook 自动发现生物意义结构**：高频 code 对应 GC 富集编码区、AT 富集非编码区、突变热点等，表明模型能无监督地捕捉功能区域。
3. **对比学习极大提升聚类能力**：+35% 至 +42% 的 Silhouette 提升，使得嵌入空间可用于无监督变异株分群。
4. **高维 embedding 更利于细粒度区分**：128-dim 比 64-dim 在所有指标上均更优，说明 representation capacity 至关重要。

### **方法的局限性**
- **codebook 利用率偏低**（仅 19.73%），可能存在容量冗余，未来可探索自适应或分层 codebook。
- 当前为静态建模，未考虑**时间动态演化**（如变异株出现、增长、衰退）。
- 尚未与真实 phylogenetic tree 对齐验证聚类结果的生物学意义。

### **未来工作方向**
1. **Hierarchical VQ-VAE**：学习多尺度表示（从 SNP 到 lineage）。
2. **整合系统发育方法**（phylogenetic methods）验证聚类结果。
3. **多病原体监控**（multi-pathogen surveillance）：扩展至流感、肠道病毒、抗生素抗性基因等。
4. **实时部署优化**：将推理延迟降至 1 分钟以内，用于早期预警。
5. **时间序列建模**：追踪变异株频率变化趋势。
6. **跨地域迁移学习**：提升在资源匮乏地区的泛化能力。

---

> **总结**：本文提出的 **VQ-VAE + 对比学习** 框架为废水基因组监测提供了一个**高效、可扩展、可解释、无需参考基因组**的解决方案，在重建精度、聚类能力和计算效率上全面超越传统方法，具有重要的公共卫生应用前景。

</details>

---

### 10. [Beyond Additivity: Sparse Isotonic Shapley Regression toward Nonlinear Explainability](https://arxiv.org/abs/2512.03112)

**Authors**: Jialai She  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.03112v1  

#### Abstract
Shapley values, a gold standard for feature attribution in Explainable AI, face two primary challenges. First, the canonical Shapley framework assumes that the worth function is additive, yet real-world payoff constructions--driven by non-Gaussian distributions, heavy tails, feature dependence, or d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Beyond Additivity: Sparse Isotonic Shapley Regression toward Nonlinear Explainability  
**论文核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Shapley values** 在可解释人工智能（XAI）中的两个关键局限性提出解决方案：

1. **Additivity 假设不成立**：标准 Shapley 框架假设特征贡献是加性的（即 $ v(A) \approx \sum_{j\in A} \phi_j $），但在实际应用中，由于非高斯分布、重尾、特征依赖或领域特定损失函数等因素，这一假设常被违反，导致归因失真。
2. **高维稀疏性处理不足**：传统方法先计算稠密的 Shapley 值，再通过后处理（如阈值化）实现稀疏解释，效率低且可能引入不一致性。

---

### 提出的新方法：Sparse Isotonic Shapley Regression (SISR)

作者提出了 **Sparse Isotonic Shapley Regression (SISR)**，一种统一的非线性可解释性框架，其核心思想是：

- **联合学习单调变换与稀疏归因**：
  - 引入一个未知的单调变换函数 $ T(\cdot) $，将原始非加性的 payoff $ v_A $ 映射到一个新的空间，在该空间中满足“T-加性”结构：  
    $$
    T(v_A) \approx \sum_{j\in A} T(\phi_j)
    $$
  - 同时对 Shapley 向量施加 $ \ell_0 $ 稀疏约束，直接控制支持集大小。

- **无需预设解析形式**：$ T(\cdot) $ 不需要预先指定（如 log、sqrt 等），而是通过 **Isotonic Regression**（使用 PAVA 算法）从数据中自动学习，增强了模型适应性。

- **优化算法设计**：
  - 利用 **Pool-Adjacent-Violators Algorithm (PAVA)** 高效求解单调回归子问题。
  - 使用 **Normalized Hard-Thresholding** 实现 $ \ell_0 $ 约束下的闭式更新，保证全局收敛性。

---

### 相比现有方法的优势

| 维度 | 传统方法 | SISR |
|------|--------|------|
| **Additivity 假设** | 强制加性，易受非线性影响 | 学习变换恢复加性，更鲁棒 |
| **稀疏性处理** | 后处理阈值或 $ \ell_1 $ 正则，有偏且需调参 | 内生 $ \ell_0 $ 约束，无偏、可控、高效 |
| **变换建模** | 固定变换或忽略 | 数据驱动学习单调变换 |
| **理论保障** | 多数无收敛保证 | 具备全局收敛性证明 |
| **计算效率** | 高维下昂贵 | 利用稀疏性和 PAVA 加速 |

---

## 2. 核心实验方法和设置

### 使用的数据集

论文在多个真实与合成数据集上进行了验证：

| 数据集 | 类型 | 特征数 $ p $ | 任务 |
|-------|------|---------------|------|
| **Prostate Cancer** | 医疗 | 9 | 回归（log cancer volume） |
| **Boston Housing** | 社会经济 | 13 | 回归（房价预测） |
| **South German Credit** | 金融风控 | 20 | 分类（信用风险） |
| **Pima Indians Diabetes** | 医疗 | 8 | 分类（糖尿病诊断） |
| **Synthetic Data** | 模拟生成 | $ p=10,15,20,25 $ | 多种变换场景测试 |

此外还构建了多种模拟场景用于分析：
- 不同变换函数（平方根、五次方根、指数、对数、正切、正态分位数）
- 不同 sparsity 水平（$ s^*=2,8,15 $）
- 不同相关性强度（Toeplitz 协方差矩阵，$ \rho=0,0.5,0.9 $）

---

### 实验设置与评估指标

#### Payoff 构造方式
- **Regression**: 使用子集回归的 $ R^2 $
- **Classification**: 使用 pseudo-$ R^2 $ 或负交叉熵
- **Tree Ensembles**: 使用 SAGE 框架中的 interventionally marginalized loss
- 考虑多种 payoff 形式（如 robust MSE、exponential utility）以检验稳定性

#### 评估指标
1. **Affinity (Affn)**：估计的归因向量 $ \hat{\gamma} $ 与真实 $ \gamma^* $ 的余弦相似度（单位范数下），衡量估计精度。
2. **Support Recovery Rate (Supp)**：正确识别出非零特征的比例。
3. **Transformation Recovery**：可视化估计的 $ T(v) $ 是否接近真实变换 $ T^*(v) $。
4. **Rank/Sign Stability**：比较不同 payoff 下特征重要性排序和符号是否一致。

#### 基线方法对比
- **Standard Shapley / SAGE / SHAP**：直接基于原始 payoff 计算归因
- **$ \ell_1 $-regularized variants**：带 shrinkage 的稀疏化方法
- **Post-hoc thresholding**：先计算全模型 Shapley 值，再截断

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）变换恢复能力（Section 4.1）
- 图1显示，在六种不同真实变换下（包括严重非线性的 tan 和 exp），SISR 所估计的 $ T(v) $ 与真实 $ T^*(v) $ 高度吻合。
- 图2展示了在 “winner-takes-all” 动态（$ v_A = \max_{j\in A} \phi_j $）下，SISR 成功恢复出强非线性变换，并且 $ \hat{\gamma} $ 与 $ T(\phi^*) $ 几乎完全相关（correlation ≈ 1.00）。

#### （2）稀疏性恢复能力（Table 1）
| $ p $ | $ \sigma_0 $ | Affn (%) | Supp (%) |
|--------|--------------|----------|----------|
| 10     | 1e-3         | 99.6     | 100      |
| 15     | 1e-3         | 99.8     | 100      |
| 20     | 5e-3         | 97.8     | 100      |
| 25     | 5e-3         | 74.0     | 100      |
| 25     | 2e-1         | 52.1     | 62.0     |

> **结论**：即使在高噪声和高维下，SISR 仍能 **完美恢复支持集（Supp=100%）**，而 Affinity 随维度和噪声下降但仍保持较高水平。

#### （3）计算效率（Figure 3）
- 随着 sparsity 上限 $ s $ 增大，计算时间显著上升。
- 表明 **稀疏约束有效提升计算效率**，尤其在高维场景。

---

### 与基线方法的对比结果

#### （1）Prostate Cancer（Figure 6）
- 标准 Shapley 将 `svi`（精囊侵犯）列为第三重要特征（>10% 权重），但统计检验（AIC/BIC/LASSO/p-value=0.6）均表明其无关紧要。
- SISR 正确将其归为接近零，与医学证据一致。
- **发现**：标准 Shapley 可能因特征相关性产生虚假重要性。

#### （2）Boston Housing（Figure 7）
- 使用 robust payoff（$ \exp(-c \cdot \text{MSE}) $）时：
  - 标准 Shapley 导致 `DIS` 重要性飙升，`CHAS` 出现负值，排名剧烈变化。
  - SISR 自动学习非线性变换补偿 distortion，保持归因模式稳定。
- **结论**：SISR 对 payoff 构造具有鲁棒性，而标准方法高度敏感。

#### （3）Bank Credit（Figure 8）
- 在 risk-averse exponential payoff 下：
  - 标准 Shapley 中 `Residence Duration` 负归因放大近四倍，不稳定。
  - SISR 输出稳定归因，接近零，符合已有研究结论。
- **说明**：SISR 过滤了由 payoff 非线性引起的虚假信号。

#### （4）Diabetes（Figure 9）
- 使用 likelihood payoff 时：
  - 标准 Shapley 导致 `Pregnancies` 出现负归因，多个特征影响力接近 `Glucose`，不合理。
  - SISR 恢复合理结构，`Glucose` 明显主导。
- **验证**：SISR 能揭示真实的 sparse 解释结构。

---

### 消融实验（隐含于多组实验中）

虽然未明确标注“ablation”，但以下实验构成实质消融：

- **有无 sparsity constraint**：在 Boston 和 Diabetes 实验中比较了 $ s=p $ 与 $ s< p $ 的效果，发现稀疏约束提升估计精度（尤其在高 SNR 下）。
- **不同 payoff 函数下的稳定性**：证明 SISR 的归因不受 payoff 形式的显著影响，而 baseline 方法波动剧烈。
- **相关性与稀疏性的影响（Figure 5）**：
  - 高相关性（$ \rho=0.9 $）导致更强非线性变换。
  - 低 sparsity（$ s^*=2 $）导致 piecewise 曲线，反映非加性。
  - **首次揭示**：仅存在无关特征或相关特征即可破坏 additivity。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **非加性普遍存在**：即使是标准的 $ R^2 $ 或 cross-entropy payoff，在特征相关或存在无关变量时也会导致严重的非加性，必须进行校正。
2. ✅ **SISR 能准确恢复真实变换和支持集**：在多种非线性场景下，SISR 可高保真地学习 $ T(\cdot) $ 并识别关键特征。
3. ✅ **归因稳定性显著优于标准方法**：面对不同的 payoff 构造（如 robust loss、utility function），SISR 输出一致，而 standard Shapley 值出现 rank 和 sign 的严重扭曲。
4. ✅ **稀疏性不仅是加速手段，更是提高准确性的方式**：内生稀疏约束避免了后处理带来的偏差，提升了统计效率。

---

### 方法的局限性

1. **计算复杂度仍为 $ O(2^p) $**：尽管利用稀疏性和 PAVA 加速，但在 $ p > 30 $ 时仍难以枚举所有子集。依赖采样策略（如 Covert & Lee, 2021）进行近似。
2. **单调性假设限制表达力**：强制 $ T(\cdot) $ 单调可能无法捕捉某些复杂的非单调 distortion。
3. **当前聚焦于 global explanation**：主要面向 SAGE 类全局归因，local explanation（如 SHAP）扩展需进一步研究。
4. **初始化敏感性**：虽有收敛保证，但初始值选择可能影响收敛速度和局部最优。

---

### 未来工作方向

1. **扩展至 Generalized Linear Model (GLM) 框架**：将 Gaussian 假设推广到指数族分布，形成 **Sparse Isotonic Shapley GLM**。
2. **结合高阶交互项**：在变换后的空间中引入 Shapley Interaction Indices，统一处理非线性和协同效应。
3. **高效采样与近似算法**：发展 scalable 的 coalition sampling 策略，使 SISR 可应用于超大规模特征空间。
4. **理论分析泛化误差与样本复杂度**：建立 finite-sample 收敛速率理论。
5. **应用于更多领域**：如 genomics、finance、causal inference 等需要高可信归因的场景。

---

## 总结

**SISR 是一项重要的理论与实践进展**，它没有抛弃 Shapley 框架的可解释性，而是通过“学习如何变得可加（learning to be additive）”来修复其在现实世界中的失效。通过联合学习单调变换与稀疏归因，SISR 实现了：

> 🔧 **更鲁棒**（robust to payoff design）  
> 📊 **更准确**（faithful to ground truth）  
> 💡 **更可解释**（sparse and stable attributions）  
> ⚙️ **更高效**（convergent and scalable algorithm）

该工作为下一代非线性可解释 AI 提供了一个坚实、统一且实用的框架。

</details>

---

### 11. [Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs](https://arxiv.org/abs/2512.03324)

**Authors**: Ngoc Bui, Shubham Sharma, Simran Lamba, Saumitra Mishra, Rex Ying  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.03324v1  

#### Abstract
Memory and computation remain core bottlenecks in long-horizon LLM inference due to the quadratic cost of self-attention and the ever-growing key-value (KV) cache. Existing strategies for memory-bounded inference, such as quantization, offloading, or heuristic KV eviction, either incur high orchestr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大型语言模型（LLMs）在处理长上下文时面临严重的内存和计算瓶颈。随着序列长度增加，self-attention 的计算复杂度呈二次方增长，而存储所有历史 token 的 Key-Value (KV) Cache 会迅速耗尽 GPU 内存。现有的 KV 缓存管理策略（如量化、卸载或基于启发式的 KV 蒸发）存在以下问题：
- **量化和压缩**：在预填充阶段有效，但在生成阶段扩展性差。
- **KV 卸载**：引入显著的系统协调开销，影响端到端吞吐量。
- **基于注意力的启发式蒸发**：假设近期被关注的 token 是重要的，但这在长程推理任务中不可靠，可能导致重要 token 因暂时未被关注而被错误地提前移除。

### 提出的新方法：TRIM-KV
本文提出了 **TRIM-KV**（Token RetentIon for Memory-bounded KV Cache），一种新颖的、基于学习的 KV 缓存管理方法，其核心思想是：
- **学习每个 token 的内在重要性**：在 token 创建时，通过一个轻量级的 **Retention Gate** 预测其标量保留分数 `β ∈ [0,1]`。
- **时间衰减机制**：该保留分数会随着时间推移（即新 token 的加入）按指数规律衰减，模拟人类大脑的遗忘过程。高 `β` 的 token 能长期保持高分，低 `β` 的 token 影响力快速消失。
- **基于重要性的淘汰策略**：当 KV 缓存超过预设内存预算 `M` 时，移除当前保留分数最低的 token。这确保了缓存中始终保存着对模型未来预测最“关键”的 token。

### 相比现有方法的优势
- **更可靠的代理**：不依赖于瞬时的 attention 分数作为重要性代理，而是学习 token 的**长期固有重要性**，避免了注意力偏差导致的误判。
- **高效且可训练**：仅需微调轻量级的 Retention Gate，主干模型参数冻结，训练成本低，推理开销极小。
- **性能优越**：在多种基准上显著优于现有的启发式和可学习的基线方法，甚至在某些情况下超越了全缓存（full-cache）模型。
- **提供可解释性**：学习到的保留分数为分析不同层和头的功能角色提供了新的诊断工具。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖了多个长上下文和长生成任务的基准：
- **数学推理**：GSM8K, MATH-500, AIME24。
- **程序化生成**：LongProc（包含从 HTML 提取表格到规划旅行等多步任务）。
- **长记忆对话**：LongMemEval（评估聊天助手的长期交互记忆能力）。
- **长上下文理解**：LongBenchV2 和 SCBench。
- **训练数据**：主要使用 OpenR1-MATH-220k 数据集进行 Retention Gate 的微调。

### 实验设置和评估指标
- **基础模型**：主要基于 Qwen3 系列模型（如 Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B）以及 DeepSeek R1 Distill 变体。
- **内存预算**：在不同实验中设置了不同的 KV 缓存大小 `M`（如 128, 512, 1024, 2048, 4096, 32768 等），以测试方法在不同资源限制下的表现。
- **评估指标**：
  - 数学推理任务使用 `pass@1` 准确率。
  - LongProc 任务使用 F1 分数或准确率。
  - LongMemEval 和 SCBench 使用各项任务的准确率。
- **训练细节**：仅微调 Retention Gate 的参数，使用蒸馏损失（distillation loss）和容量损失（capacity loss）联合优化，目标是让带门控的模型模仿原始模型的输出，同时强制满足内存约束。

### 基线方法对比
与以下主流方法进行了比较：
- **启发式 KV 蒸发**：StreamingLLM, H2O, SnapKV, R-KV。
- **可学习的 KV 检索**：SeerAttn-R（一种先进的检索增强方法）。
- **全缓存模型**（Full KV）：作为性能上限的参考。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **显著超越基线**：TRIM-KV 在所有基准上均一致地大幅领先于各种基线方法，尤其是在**低内存预算**（low-memory regimes）下优势更为明显。
- **超越更强的基线**：TRIM-KV 的性能甚至超过了使用 **4倍 KV 预算**的 R-KV 和 SnapKV 等启发式方法。
- **相比可学习检索的提升**：相比于当时最先进的可学习 KV 检索基线 SeerAttn-R，TRIM-KV 在相同预算下实现了 **58.4% 的 pass@1 增益**。
- **超越全缓存模型**：在多个设置下（例如 Qwen3-4B 模型上的 AIME24 和 LongProc 任务），TRIM-KV 的性能**甚至超过了全缓存模型**。这一惊人结果表明，选择性保留可以作为一种有效的正则化手段，通过抑制无信息 token 的噪声来提升模型性能。

### 消融实验结果
- **损失函数的有效性**：消融研究表明，结合 `distillation loss` (LKL) 和 `next-token prediction loss` (LNTP) 的效果最好。单独使用任一损失也能取得不错的效果，但组合后性能进一步提升。
- **容量损失的必要性**：移除 `capacity loss` (LCap) 会导致性能急剧下降，证明了该损失对于实现有效压缩至关重要。
- **门控架构**：使用单隐藏层的 MLP 作为 Retention Gate 比简单的线性投影能提供更强大的保留估计能力。
- **训练预算的影响**：训练时设定的内存容量 `M` 对性能有影响。过小的 `M` 会导致过度优化稀疏性，而过大的 `M` 则无法施加足够的压缩压力。建议将 `M` 设置为预期部署时的内存预算。

---

## 4. 关键结论和发现

### 主要发现
1.  **内在重要性优于注意力代理**：直接学习 token 的长期内在重要性，比依赖瞬时 attention 分数的启发式方法更可靠、更有效。
2.  **选择性保留即正则化**：TRIM-KV 不仅节省了内存，其选择性保留机制还能过滤掉冗余和噪声 token，从而在某些任务上提升了模型的最终性能，这揭示了一种新的模型正则化视角。
3.  **涌现的人类直觉行为**：学习到的保留策略能够自然地涌现出多种已知的启发式模式，如 **sink tokens**（保留起始 token）、**sliding windows**（滑动窗口）和 **gist compression**（要点压缩），而无需显式编程，证明了方法的强大适应性。
4.  **提供可解释性洞见**：保留分数为研究 LLM 内部工作机制提供了新途径。分析发现，不同的 KV head 会发展出特定的功能角色，例如专门保留数学符号、变量、问题描述或指令等。

### 方法的局限性
- **依赖预训练模型**：当前方法是在冻结的预训练模型上添加 Retention Gate 进行微调。它没有从根本上改变模型的注意力机制。
- **固定缓存大小**：虽然实现了内存有界，但缓存大小 `M` 是一个需要预先设定的超参数。

### 未来工作方向
1.  **联合训练**：将 Retention-Gated Attention 机制集成到模型的预训练或后训练过程中，与 Query、Key、Value 的学习协同优化，使模型天生就是内存有界的。
2.  **自适应预算**：开发能够跨层、跨头、跨任务动态分配内存预算的自适应机制。
3.  **扩展应用**：将保留门控机制应用于多模态输入和工具调用（tool-calling）等更复杂的场景。

</details>

---

### 12. [DVPO: Distributional Value Modeling-based Policy Optimization for LLM Post-Training](https://arxiv.org/abs/2512.03847)

**Authors**: Dingwei Zhu, Zhiheng Xi, Shihan Dou, Yuhui Wang, Sixian Li, Junjie Ye, Honglin Guo, Shichun Liu, Chenhao Huang, Yajie Yang, Junlin Shang, Senjie Jin, Ming Zhang, Jiazheng Zhang, Caishuang Huang, Yunke Zhang, Demei Yan, Yuran Wang, Tao Gui  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.03847v1  

#### Abstract
Reinforcement learning (RL) has shown strong performance in LLM post-training, but real-world deployment often involves noisy or incomplete supervision. In such settings, complex and unreliable supervision signals can destabilize training and harm generalization. While existing approaches such as wo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DVPO: Distributional Value Modeling-based Policy Optimization for LLM Post-Training**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在大语言模型（LLM）的后训练（post-training）过程中，强化学习（RL）常用于优化模型行为。然而，现实场景中的监督信号（如人类反馈或奖励模型输出）往往存在**噪声或不完整性**，导致传统 RL 方法（如 PPO、GRPO）出现以下问题：
- **训练不稳定**：噪声干扰价值估计，引发错误的策略更新。
- **泛化能力差**：在分布外（OOD）任务上表现不佳。
- **过于保守或激进**：例如基于鲁棒 Bellman 算子的方法（Robust Bellman PPO）虽提升稳定性，但因过度悲观而抑制探索，损害泛化。

### **提出的新方法：DVPO**
本文提出 **DVPO**（Distributional Value Modeling with Risk-aware Policy Optimization），一种结合**分布值建模**与**条件风险控制理论**的新型 RL 框架，旨在平衡**鲁棒性**与**泛化能力**。

#### **核心创新点**
1. **Token-Level 分布值建模**  
   不再预测标量价值（scalar value），而是学习每个 token 的**价值分布**（value distribution），提供更细粒度的监督信号，捕捉不确定性与高阶统计特性（如方差、偏度）。

2. **不对称风险正则化（Asymmetric Risk Regularization）**  
   引入条件风险约束，对价值分布的尾部进行差异化调控：
   - **收缩下尾（Lower Tail Contraction）**：抑制由噪声引起的负向偏差，增强鲁棒性。
   - **扩展上尾（Upper Tail Expansion）**：保留高价值信号，鼓励探索多样性，提升泛化能力。

3. **多头分位数集成架构（Multi-Headed Quantile Ensemble）**  
   使用多个独立的分位数头（quantile heads）构建更稳定的价值分布估计，减少单点故障和过拟合风险。

4. **复合损失函数设计**  
   综合多种目标函数，包括：
   - 分位数回归（Quantile Regression）
   - 尾部期望校准（CVaR 和 Gain Objective）
   - 均值偏移惩罚（Mean-Shift Penalization）
   - 尾部形状与曲率正则化（Tail Shape & Curvature）
   - 多头一致性（Multi-Distribution Consistency）

---

### **相比现有方法的优势**
| 方法 | 特点 | 局限性 | DVPO 的改进 |
|------|------|--------|-------------|
| **PPO / GRPO** | 基于均值估计，简单有效 | 易受噪声影响，泛化弱 | 引入分布建模，增强抗噪能力 |
| **Robust Bellman PPO** | 最小化最坏情况，提升稳定性 | 过于保守，抑制探索 | 不对称调节，兼顾鲁棒与探索 |
| **RFQI / CQL** | 最坏情况优化 | 政策过于悲观 | 通过上尾扩展保留高价值路径 |

> ✅ **DVPO 的优势在于实现了“稳健而不保守”的政策优化**，在噪声环境下既保持训练稳定，又具备良好的跨域适应能力。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **对话任务**：自建 **Honor-Dialogue Dataset**
   - 多轮、多领域、任务导向的真实场景对话。
   - 包含生活服务、交通出行、医疗健康、社交娱乐、金融服务等五个领域。
2. **数学推理任务**：Light-R1 数据集
   - 包括 MATH500、AIME24、Minerva-Math、AMC23。
3. **科学问答任务**：SuperGPQA 和 SampleQA
   - 覆盖 285 个研究生学科，挑战模型深度知识理解。
   - Humanity's Last Exam（HLE）作为高难度测试集。

### **实验设置**
- **初始模型**：Qwen3-8B（数学/科学任务）、Qwen3-8B 微调版（对话任务）。
- **奖励来源**：
  - 数学/科学任务：基于规则的奖励（rule-based rewards），通过多数投票生成伪标签。
  - 对话任务：基于模型的奖励（reward model），存在显著噪声（准确率仅 ~71.8%）。
- **训练步数**：500 步（避免部分方法后期崩溃）。
- **硬件配置**：8×NVIDIA A100 80GB GPU。

### **评估指标**
| 任务类型 | 主要指标 |
|---------|----------|
| **数学 & 科学任务** | 准确率（Accuracy） |
| **对话任务** | 三重评估框架：<br>• **TCR**（Task Completion Rate）<br>• **ACR**（Ask Completion Rate）<br>• **GCR**（Goal Completion Rate）<br>• 综合平均得分（Domain-AVG & Overall AVG） |

### **基线方法对比**
- **PPO**
- **GRPO**
- **Reinforce++**
- **Dr.GRPO**
- **Robust Bellman PPO**（本文实现的鲁棒版本）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### **表 1：真实对话任务性能（噪声奖励下）**
| 方法 | Life Services | Transport | Healthcare | Social | Finance | **AVG** |
|------|--------------|-----------|-----------|--------|--------|--------|
| Baseline | 86.73% | 84.50% | 90.23% | 87.13% | 82.70% | 86.26% |
| PPO | 84.20% | 86.07% | 90.87% | 81.00% | 83.87% | 85.20% |
| GRPO | 30.03% | 28.57% | 28.33% | 28.90% | 27.90% | 28.75% |
| **DVPO (Ours)** | **88.13%** | **87.73%** | **87.67%** | **87.67%** | **82.73%** | **86.79%** |

> 🔹 DVPO 在所有领域均优于基线，平均提升 **1.59%**（vs PPO），且在噪声严重的 GRPO 上实现巨大超越。

#### **表 2 & 3：跨域泛化能力（数学 vs 科学互训）**
| 训练域 → 测试域 | ID AVG | OOD AVG | ALL AVG |
|------------------|--------|---------|---------|
| **科学域训练 → 数学测试** | 3.83% | 66.48% | **39.63%** |
| **数学域训练 → 科学测试** | 66.45% | 4.04% | **39.70%** |
| （其他方法最高约 37.7%） | — | — | — |

> 🔹 DVPO 显著优于所有基线，在 ID 和 OOD 上均取得最佳表现，证明其强大的**迁移与泛化能力**。

#### **消融实验（Ablation Study）**
| 模型变体 | ID AVG | OOD AVG | ALL AVG |
|---------|--------|---------|---------|
| Core Quantile Regression | 3.65% | 61.37% | 36.63% |
| + Distribution Consistency | 3.73% | 63.73% | 38.02% |
| + Tail Calibration | 3.74% | 64.42% | 38.42% |
| + Shift Penalization | 3.29% | 65.17% | 38.65% |
| **+ Tail Shape & Curvature (Ours)** | **3.83%** | **66.48%** | **39.63%** |

> 🔹 所有组件均有正向贡献，尤其是 **Tail Shape & Curvature** 对 OOD 性能提升最大，验证了尾部调控的有效性。

#### **不同超参数敏感性分析**
- **区间密度（Interval Density）**：200 最优，过高（500）或过低（50）均下降。
- **风险间隔权重（Risk Interval Weight）**：0.1 最佳，过大导致不稳定，过小无法抑制噪声。

---

## **4. 关键结论和发现**

### **主要发现**
1. **分布式值建模显著提升抗噪能力**  
   相比标量估计，建模完整价值分布能更好地处理不确定性和噪声。

2. **不对称尾部调控是关键创新**  
   下尾收缩防噪声，上尾扩张促探索，打破了“鲁棒性 vs 泛化”的传统权衡。

3. **DVPO 在真实噪声环境中表现卓越**  
   在对话、数学、科学三大类任务中，DVPO 始终优于 PPO、GRPO 及鲁棒 Bellman 方法，尤其在 OOD 场景下优势明显。

4. **可视化验证：关注关键语义词**  
   如图 9 所示，DVPO 的 advantage estimation 能精准聚焦“nucleus”、“quarks”等关键词，而 PPO 和 Robust Bellman 方法几乎无差异。

---

### **局限性**
1. **计算开销较大**  
   分布式建模和多头结构带来额外计算负担，可能限制在大规模或低延迟场景的应用。
2. **超参数依赖性强**  
   区间密度、风险阈值等需根据任务调整，缺乏通用最优配置。
3. **极端噪声仍具挑战**  
   当奖励严重失真或误标时，性能仍会下降。

---

### **未来工作方向**
- 探索更高效的分布表示方式（如隐式分位网络 IQN）以降低计算成本。
- 设计自适应的风险控制机制，动态调整上下尾权重。
- 将 DVPO 应用于更多现实世界代理（autonomous agents）决策任务，如机器人控制、推荐系统等。

---

> ✅ **总结**：DVPO 提出了一种新颖的“分布+风险感知”RL 范式，为 LLM 后训练在噪声环境下的稳定与泛化提供了可扩展、有效的解决方案，具有重要的实践意义。

</details>

---

### 13. [Beyond Playtesting: A Generative Multi-Agent Simulation System for Massively Multiplayer Online Games](https://arxiv.org/abs/2512.02358)

**Authors**: Ran Zhang, Kun Ouyang, Tiancheng Ma, Yida Yang, Dong Fang  
**Category**: cs.AI  
**Published**: 2025-12-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.02358v1  

#### Abstract
Optimizing numerical systems and mechanism design is crucial for enhancing player experience in Massively Multiplayer Online (MMO) games. Traditional optimization approaches rely on large-scale online experiments or parameter tuning over predefined statistical models, which are costly, time-consumin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond Playtesting: A Generative Multi-Agent Simulation System for Massively Multiplayer Online Games*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 MMO 游戏中的数值系统与机制设计优化依赖于大规模线上实验（如 A/B testing）或基于统计模型的参数调优，存在以下问题：
- **高时间成本**：调整后需数周甚至数月观察反馈；
- **高机会成本**：错误调整可能破坏游戏经济系统，导致玩家流失；
- **测试局限性**：重大机制变更无法通过小规模测试验证；
- **黑箱预测**：现有模拟方法缺乏微观层面的可解释性，难以指导精细化设计。

此外，已有基于 LLM 的生成式代理研究多集中于社会模拟或孤立场景（如谈判、P2W 机制），缺乏对完整 MMO 游戏生态系统的高保真建模与实证验证。

---

### 🚀 提出的新方法与创新思路
本文提出一个**基于生成式多智能体（Generative Multi-Agent）的 MMO 游戏仿真系统**，其核心创新包括：

1. **高保真的 LLM 驱动玩家代理（Player Agent）构建框架**  
   - 利用 **Supervised Fine-Tuning (SFT)** 和 **Reinforcement Learning (GRPO)** 在真实玩家行为数据上微调 LLM，使其具备符合特定游戏域的决策能力。
   - 设计三阶段训练流程：词汇扩展 → 动作规划 SFT → 强化学习增强推理能力。

2. **数据驱动的游戏环境重建**
   - 基于真实 gameplay logs 构建动态游戏服务模块（Battle Server, NPC Shop, Black Market），实现对战斗结果、资源流动等关键系统的精准建模。

3. **端到端可干预仿真平台**
   - 支持游戏设计师进行“假设分析”（what-if analysis），例如引入新交易系统，并观测其宏观与微观因果效应。
   - 提供 GUI 监控界面，支持从群体分布到个体行为轨迹的多层次分析。

4. **系统级验证与可解释性**
   - 不仅评估宏观统计一致性，还提供每个 agent 的 reasoning log，突破传统“黑箱”模拟局限。

---

### 🔍 相比现有方法的优势
| 维度 | 传统统计模型 | 简化模拟系统 | 本文方法 |
|------|---------------|----------------|-----------|
| 保真度 | 低（宏观聚合） | 中（规则驱动） | ✅ 高（数据+LLM驱动） |
| 可解释性 | ❌ 黑箱 | ⚠️ 有限 | ✅ 显式推理链 |
| 微观行为建模 | ❌ | ⚠️ 粗粒度 | ✅ 个性化 agent 决策 |
| 干预因果推断 | 不可靠 | 有限 | ✅ 支持反事实分析 |
| 扩展性 | 高 | 中 | ✅ 支持千级 agent 并行 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- 来自某真实 MMO 游戏（类似 *Escape from Tarkov* 类型）的百万级 gameplay logs，涵盖：
  - 登录/登出事件
  - 战斗记录（胜负、收益）
  - 购买/出售行为
  - 社交互动
- 数据用于：
  - 构建玩家画像（player profiling）
  - 训练 Player Agent 的 SFT 与 RL 模块
  - 校准 Battle Server 的预测模型
  - 定义五类典型玩家 cluster（见下文）

---

### ⚙️ 实验设置与评估指标

#### （1）Player Agent 性能评估
- **任务**：下一动作预测（next-step action prediction）
- **动作类别**：`{offline, battle, buy, sell}`
- **输入上下文**：历史行为序列、玩家 profile、当前状态
- **评估指标**：
  - Stepwise Prediction Accuracy
  - 类别分布匹配度（Fig. 4a）

#### （2）Battle Server 验证
- **目标**：预测不同玩家群体在第 N 场比赛中的胜率与收入
- **数据划分**：
  - 训练：2025 S1 赛季数据
  - 测试：2025 S2 赛季数据（严格无泄露）
- **玩家聚类（5 类）**：
  1. Stable Development Players（稳定发展型）
  2. Novice Players（新手）
  3. Wealth-Accumulating Elite Players（财富积累精英）
  4. Casual Players（休闲玩家）
  5. High-skill Players（高技能玩家）
- **评估指标**：
  - 胜率预测 MAE / 相关性
  - 单场收入预测误差

#### （3）干预案例研究（Case Study）
- **干预内容**：在游戏中引入官方 Black Market（替代非正式物品交换）
- **评估方式**：
  - 对比干预前后“in-game trading”行为比例变化
  - 分析 agent 是否合理采纳新机制

#### （4）基线方法对比
- **DeepSeek-V3**（未微调的通用 LLM）
- **PlayerAgent-GRPO**（仅用 RL 微调）
- **PlayerAgent-Profile-GRPO**（加入 profile 信息的完整模型）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）Player Agent 准确率提升（Fig. 4b）
| 模型 | 准确率 | 相对提升 |
|------|--------|----------|
| DeepSeek-V3（Base） | 38.69% | — |
| PlayerAgent-GRPO | 47.03% | +8.34% |
| **PlayerAgent-Profile-GRPO**（本文） | **48.88%** | **+10.19%** |

✅ 结果表明：**结合用户画像的 GRPO 微调显著提升动作预测准确性**。

---

#### （2）Battle Server 预测效果（Fig. 5b & 5c）
- 在五个玩家 cluster 上，胜率与收入预测均与真实数据高度一致（存在固定偏移但趋势吻合）。
- 特别是在 **Wealth-Accumulating Elite** 和 **Stable Development** 玩家中表现最优。
- 新手（Novice）和休闲玩家波动较大，预测误差略高，但仍保持合理趋势。

> 示例：第 25 场比赛中，预测胜率与真实值相关系数 > 0.92（除 Novice 外）

---

#### （3）消融实验与设计有效性
- **三阶段微调必要性验证**：
  - 缺少 Vocabulary Expansion → 无法识别装备名称（如 AWM），导致语义错误；
  - 缺少 Action Planning SFT → 决策脱离实际行为模式；
  - 缺少 RL 增强 → 推理过程简略，缺乏反思（reflection）能力。

- **Profile 信息作用**：
  - 加入 player_type 后，agent 更能区分“激进派”与“休闲派”的行为倾向（如失败后是否继续战斗）。

---

#### （4）干预实验结果（Fig. 4c & 4d）
- 引入 Black Market 后：
  - 非正式 in-game trading 行为占比从 **27.4% 下降至 1.5%**
  - 绝大多数 agent 快速转向使用 Black Market
  - 少量 agent 因习惯延续仍使用旧机制（体现行为惯性）

✅ 表明系统能捕捉真实世界的**因果响应**与**行为迁移路径**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 经过 SFT + RL 微调可在 MMO 游戏中生成高保真玩家行为**，其决策与真实玩家具有强一致性。
2. **数据驱动的环境建模（如 Battle Server）能够复现复杂的游戏动力学**，尤其在主流玩家群体中预测准确。
3. **该系统支持有效的“设计前验证”（pre-deployment evaluation）**，可用于评估机制变更的潜在影响。
4. **系统具备良好的可解释性**：每个 agent 的 reasoning log 提供透明决策依据，便于调试与优化。
5. **多层级监控能力**：既可观测财富分布、活跃度等宏观指标，也可追踪单个 agent 的行为演化。

---

### ⚠️ 局限性
1. **计算开销较高**：LLM 推理带来延迟，限制极端规模扩展（如百万 agent 级别）；
2. **对冷启动玩家建模不足**：缺乏足够历史数据的新玩家难以准确刻画；
3. **Black Market 等复杂市场机制仍为简化建模**，未完全模拟博弈策略；
4. **目前聚焦 extraction shooter 类型 MMO**，泛化至其他类型（如 MMORPG）需进一步验证。

---

### 🔮 未来工作方向
1. **引入更多 agent 间交互机制**：如联盟形成、团队协作、社交网络演化；
2. **支持更复杂的经济系统建模**：引入拍卖、供需动态、价格弹性等机制；
3. **轻量化 agent 架构设计**：探索小型化模型（如 TinyLLM）以降低推理成本；
4. **跨游戏迁移学习**：研究如何将在一款游戏中训练的 agent 迁移到相似类型游戏中；
5. **与强化学习策略 agent 结合**：探索 agent 自主进化策略的可能性。

---

## 总结
本文提出的 **Generative Multi-Agent Simulation System** 是首个将 LLM 驱动 agent 应用于完整 MMO 游戏系统仿真的尝试，实现了从“黑箱统计预测”向“白盒行为模拟”的跃迁。它不仅提升了模拟的真实性与可解释性，也为游戏设计提供了高效、低成本、安全的“数字孪生”试验场，具有重要的工业应用前景。

</details>

---

### 14. [Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective](https://arxiv.org/abs/2512.03759)

**Authors**: Jingyang Ou, Jiaqi Han, Minkai Xu, Shaoxuan Xu, Jianwen Xie, Stefano Ermon, Yi Wu, Chongxuan Li  
**Category**: cs.CL  
**Published**: 2025-12-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.03759v1  

#### Abstract
Reinforcement Learning (RL) has proven highly effective for autoregressive language models, but adapting these methods to diffusion large language models (dLLMs) presents fundamental challenges. The core difficulty lies in likelihood approximation: while autoregressive models naturally provide token...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective*

## 1. 主要贡献和创新点

### 解决的问题
本文针对将 **Reinforcement Learning (RL)** 应用于 **diffusion Large Language Models (dLLMs)** 所面临的核心挑战。传统 RL 方法（如 GRPO）依赖于自回归模型（autoregressive models）的 token-level 条件概率进行策略优化。然而，dLLMs 采用非自回归（non-autoregressive）的迭代去噪过程生成序列，其序列似然无法自然分解为 token-level 条件概率。这种根本性的不匹配导致直接应用 token-level RL 目标函数在 dLLMs 上存在困难。

现有方法尝试通过启发式近似（如 mean-field 近似或 token-level ELBO 分解）来解决此问题，但这些方法要么忽略了上下文信息，要么破坏了 ELBO 作为完整序列下界（lower bound）的数学严谨性，导致训练不稳定和性能不佳。

### 提出的新方法：ESPO
作者提出了 **ELBO-based Sequence-level Policy Optimization (ESPO)**，一种专为 dLLMs 设计的、基于序列级别的原则性 RL 框架。其核心创新点如下：

- **序列级动作空间（Sequence-level Action Space）**：不再将每个 token 视为独立动作，而是将整个序列的生成视为一个单一动作。这从根本上避免了对 dLLMs 强加自回归结构的错误假设。
- **ELBO 作为似然代理**：利用证据下界（Evidence Lower Bound, ELBO）作为不可计算的完整序列对数似然的可计算代理。这是 dLLMs 中广泛使用的标准近似，具有坚实的理论基础。
- **稳定化比率估计**：引入了按序列长度归一化的重要性比率（importance ratio），解决了原始 ELBO 差异随序列长度线性增长而导致指数运算后数值爆炸或消失的问题。
- **鲁棒的 KL 散度估计**：采用 `k2` 估计器而非传统的 `k3` 估计器来计算 KL 散度正则项。`k2` 估计器是二次形式，避免了 `k3` 中的指数项，从而保证了梯度信号的稳定性。

### 相比现有方法的优势
- **原则性强**：ESPO 建立在坚实的变分推断基础上，其目标函数是数学上一致的，而 token-level 方法（如 d1, wd1）依赖于有缺陷的启发式分解。
- **训练更稳定**：通过序列级优化和 `k2` KL 估计器，显著提高了大规模训练的稳定性。
- **性能更优**：在需要全局一致性的任务（如规划）上表现尤为突出，大幅超越现有 token-level 基线。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖了三大类推理任务：
- **数学推理**：GSM8K 和 MATH 数据集。
- **代码生成**：HumanEval 和 MBPP（及其增强版 MBPP+）。
- **规划任务**：Countdown 和 Sudoku（合成数据）。

### 实验设置和评估指标
- **模型**：在两个开源 dLLMs 上进行实验——LLaDA-8B-Instruct 和 Dream-7B-Instruct。
- **训练**：直接在预训练模型上应用 RL，无需额外的任务特定监督微调（SFT）。最大训练序列长度为 256。
- **评估**：在生成长度为 128、256 和 512 的情况下进行评估，以测试长度泛化能力。
- **评估指标**：使用各基准的标准评估脚本，报告准确率（accuracy）或通过率（pass@1）。
- **方差缩减技术**：采用了反向采样（antithetic sampling）和耦合采样（coupled sampling）来稳定 ELBO 估计。

### 基线方法对比
- **d1 (diffu-GRPO)**：使用 mean-field 近似作为 token-level 条件概率的代理。
- **wd1**：一种加权策略优化方法，也属于 token-level 范畴。
- **原始模型**：未经过 RL 微调的基线模型。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
根据 **Table 1** 和 **Table 2** 的结果，ESPO 在所有任务上均显著优于基线方法：

#### 数学与规划任务 (Table 1)
| 任务 | 模型 | Avg. 性能 | ESPO 改进 (△) |
| :--- | :--- | :--- | :--- |
| **GSM8K** | LLaDA + ESPO | **82.0** | +6.1 |
| **MATH** | LLaDA + ESPO | **39.5** | +2.5 |
| **Countdown** | LLaDA + ESPO | **81.0** | **+62.3** |
| **Sudoku** | LLaDA + ESPO | **86.0** | **+70.3** |

- **规划任务提升巨大**：在 Countdown 任务上，ESPO 相比最强基线（wd1）平均提升了 **62.3** 个绝对百分点；在 Sudoku 任务上提升了 **70.3** 个点。这验证了序列级优化对于需要全局逻辑一致性的任务至关重要。
- **数学任务稳步提升**：尽管提升幅度较小（约 2-6 个百分点），但 ESPO 在所有长度上都保持了一致且稳定的增益，并超过了所有 token-level 基线。

#### 代码生成任务 (Table 2)
| 任务 | 模型 | HumanEval (Avg.) | MBPP (Avg.) |
| :--- | :--- | :--- | :--- |
| **HumanEval** | LLaDA + ESPO | **40.1** | — |
| **MBPP** | LLaDA + ESPO | — | **41.0** |
| **HumanEval** | LLaDA-1.5 (私有数据) | 40.3 | — |

- **性能媲美更大规模模型**：ESPO 在代码任务上的表现（40.1%）已经接近甚至超过在更大规模私有数据集上训练的 LLaDA-1.5 模型（40.3%），证明了其高效性。

### 消融实验结果
- **动作空间与似然代理**（图1）：消融实验证明，“序列级 + ELBO”组合是唯一能实现快速、稳定学习并达到最高奖励的方法。“token-level + ELBO”虽然初期有效，但很快崩溃，表明 token 级分解 ELBO 是不稳定的。
- **KL 估计器选择**（图2, 表5）：`k2` 估计器是实现稳定学习的关键。使用 `k3` 或 `k1` 会导致训练停滞或崩溃。即使将 `k2` 应用到 token-level 基线（d1 + k2），性能也无法提升（表5），说明 ESPO 的成功主要源于其序列级框架，而非仅靠更好的 KL 估计器。
- **蒙特卡洛样本数**（图4）：增加 MC 样本数可以提高信号丰富任务（如 Sudoku）的收敛速度，但在稀疏奖励任务（如 Countdown）中效果有限。
- **策略更新次数 (u)**（图5）：方法对 `u` 值表现出很强的鲁棒性，在不同设置下都能收敛到高奖励。

---

## 4. 关键结论和发现

### 主要发现
1. **序列级视角是关键**：将整个序列生成视为单一动作，并使用 ELBO 作为序列似然的代理，是为 dLLMs 设计 RL 算法的原则性路径。强行将 dLLMs 套入 token-level 自回归框架是根本错误的。
2. **ESPO 高效且强大**：提出的 ESPO 框架不仅在理论上更严谨，而且在实践中实现了稳定、高效的训练，并在数学、代码和规划任务上全面超越了现有的 token-level RL 方法。
3. **全局一致性任务受益最大**：ESPO 在 Sudoku 和 Countdown 等需要严格逻辑和全局规划的任务上取得了突破性进展，这凸显了序列级优化捕捉长程依赖的能力。

### 方法的局限性
- **计算成本**：尽管相比轨迹级 RL 更高效，但 dLLMs 的非自回归生成过程本身计算开销较大，尤其是在没有有效 KV cache 利用的情况下。
- **通用性**：目前的评估集中在推理和规划任务上，其在其他下游任务（如对话、摘要）上的普适性有待进一步验证。

### 未来工作方向
- 探索更高效的 ELBO 估计方法以进一步降低训练成本。
- 将 ESPO 框架扩展到多模态 dLLMs。
- 研究如何结合监督微调（SFT）与 ESPO 进行更有效的两阶段训练。
- 开源代码和检查点（作者已承诺在盲审后开源）。

</details>

---

### 15. [Training and Evaluation of Guideline-Based Medical Reasoning in LLMs](https://arxiv.org/abs/2512.03838)

**Authors**: Michael Staniek, Artem Sokolov, Stefan Riezler  
**Category**: cs.CL  
**Published**: 2025-12-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.03838v1  

#### Abstract
Machine learning for early prediction in medicine has recently shown breakthrough performance, however, the focus on improving prediction accuracy has led to a neglect of faithful explanations that are required to gain the trust of medical practitioners. The goal of this paper is to teach LLMs to fo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Training and Evaluation of Guideline-Based Medical Reasoning in LLMs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于机器学习的医学预测模型虽然在**预测准确性**上取得了突破，但其推理过程缺乏可解释性和可信度，难以获得临床医生的信任。尤其在早期诊断（如败血症预测）中，模型往往只输出最终标签，而无法提供符合医学共识指南（consensus guidelines）的逐步推理过程。

此外，现有方法在处理规则例外（exceptions）和时间序列稀疏采样变量预测方面存在瓶颈。

### 提出的新方法与创新思路
本文提出了一种**基于医学共识指南的LLM训练与评估框架**，核心思想是：
- 将复杂的医学共识定义（如Sepsis-3）转化为**可执行的推理链（verbalized rule instantiations）**
- 利用电子健康记录（EHR）数据生成大量“输入-推理链-输出”样本，对LLM进行**监督微调（supervised fine-tuning）**
- 引入**多模态架构**，将时间序列预测模型（TSF）的输出嵌入到LLM中，提升对未来临床变量的预测能力

#### 主要创新点包括：
1. **自动构建“医疗Scratchpad”数据**  
   首次系统地将数学领域中的“scratchpad”训练范式应用于医学LLM，通过模板自动生成符合Sepsis-3等指南的详细推理文本。
   
2. **双维度自动评估机制**  
   提出两个关键评估指标：
   - **Derivation Correctness（推导正确性）**：检查模型是否严格按照医学规则进行逻辑推导（衡量推理忠实性）
   - **Value Correctness（数值正确性）**：比较模型预测值与真实测量值之间的偏差（衡量实用性）

3. **支持规则例外的学习机制**  
   合成带有ICD-10编码指示慢性病（如慢性肾病）的数据，模拟临床实践中需忽略某些器官评分的情况，验证模型能否学会这些“例外规则”。

4. **多模态融合TSF + LLM架构**  
   将专用时间序列预测模型（Transformer-based TSF）的隐状态作为额外输入注入LLM，形成端到端联合优化的多模态系统。

### 相比现有方法的优势
| 方面 | 传统方法（Prompting / General Pretraining） | 本文方法（Fine-tuning on Verbalized Rules） |
|------|--------------------------------------------|---------------------------------------------|
| 推理忠实性 | 差，易出现幻觉或跳过步骤 | 极高，接近完美推导一致性 |
| 规则泛化能力 | 对未见患者泛化差 | 在同领域内近乎完全泛化 |
| 处理例外情况 | 几乎无能力 | 可学习并应用合成的例外规则 |
| 时间序列预测 | 依赖LLM自身零样本能力，效果差 | 融合专用TSF模型，显著提升预测质量 |

---

## 2. 核心实验方法和设置

### 数据集
- **MIMIC-III**：包含44,858条ICU住院记录，筛选出至少24小时监护且年龄≥18岁的患者
- 特征：共131个动态临床变量（见Appendix A.4），加上性别和年龄
- 关键任务相关变量（用于SOFA评分）：GCS、MAP、PaO₂/FiO₂、Platelet、Bilirubin、Creatinine/Urine
- 时间窗口：前24小时为观察期，后24小时为预测期，滑动窗口切分

### 实验设置
- **基础模型**：Llama-3 8B（`meta-llama/Llama-3.1-8B-Instruct`）
- **微调方式**：使用LoRA适配器进行参数高效微调（`lora-r=16`, `alpha=16`）
- **多模态架构**：
  - TSF模型：基于Transformer的编码器-解码器结构（IMS decoder）
  - 连接方式：TSF输出向量 → MLP连接层 → 映射为LLM词嵌入维度 → 拼接到文本嵌入前端
  - 所有组件（TSF、Connector、LLM）共同微调

### 评估指标
| 指标 | 定义 | 测量目标 |
|------|------|----------|
| **Derivation Correctness** | 模型每一步推理是否遵循共识规则（如阈值映射、求和、差值判断） | 推理过程的**逻辑忠实性** |
| **Forced Derivation Correctness** | 给定正确中间结果，模型能否继续正确推导到最后结论 | 排除前期错误传播的影响 |
| **Value Correctness** | 预测数值与真实值偏差是否在±5%以内（部分变量放宽） | 数值预测的**实用精度** |
| **Final Task Metrics** | Accuracy, Specificity, Sensitivity, F1-score for SEPSIS prediction | 最终诊断性能 |

### 基线方法对比
| 模型 | 类型 | 描述 |
|------|------|------|
| **one-shot (8B)** | Prompting | 使用包含Sepsis-3规则的prompt进行推理 |
| **one-shot-70B** | Prompting | 更大模型（Llama-3 70B）+ 相同prompt |
| **me-llama (8B)** | Pretrained | 在医学文献（含指南）上预训练的LLM |
| **deepseek (8B)** | Reasoning-focused | 专为推理设计的蒸馏模型 |
| **pipeline** | Hybrid | 先用TSF预测变量，再拼接至prompt供LLM使用（固定TSF） |
| **multimodal** | End-to-end | TSF + LLM 联合训练 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Tables 1–4）

#### ✅ 推导正确性（Derivation Correctness）
| 模型 | SOFA₁₋₂₄ | SOFA₂₅₋₄₈ | SOFAdiff | SEPSIS |
|------|--------|---------|--------|-------|
| one-shot | 0.543 | 0.518 | 0.845 | 0.859 |
| one-shot-70B | 0.893 | 0.868 | 0.891 | 0.909 |
| me-llama | 0.777 | 0.768 | 0.769 | 0.769 |
| fine-tuned | **1.000** | **1.000** | **1.000** | **1.000** |
| multimodal | 0.999 | 1.000 | 1.000 | 1.000 |

> 🔍 微调模型在所有推理步骤上达到**近乎完美的推导正确性**，远超其他方法。

#### ✅ 数值正确性（Value Correctness）——临床变量预测
以PaO₂/FiO₂为例：
| 模型 | 当前值预测 | 未来值预测 |
|------|-----------|------------|
| one-shot | 0.140 | 0.073 |
| fine-tuned | 0.998 | 0.565 |
| multimodal | 0.997 | **0.596** |

> ⚠️ 即使微调后，**未来24小时变量预测仍大幅下降**，尤其是稀疏实验室指标（如Urine: 0.133 → 0.183）

#### ✅ 最终Sepsis预测性能（F1-score）
| 模型 | Accuracy | Specificity | Sensitivity | **F1** |
|------|----------|-------------|-------------|--------|
| one-shot | 0.834 | 0.876 | 0.331 | 0.231 |
| one-shot-70B | 0.857 | 0.915 | 0.118 | 0.108 |
| fine-tuned | 0.868 | 0.936 | 0.263 | 0.254 |
| **multimodal** | **0.886** | **0.922** | **0.386** | **0.309** |

> 📈 多模态方法在F1上提升明显（+22.6% vs fine-tuned），说明**整合TSF能有效改善长期预测性能**

#### ✅ 规则例外处理能力（Table 2）
| 条件 | % Cases Changed | ID Score | OOD Score |
|------|------------------|---------|----------|
| With Precondition | ~40–45% | 1.000 | 1.000 |

> ✅ 模型能准确识别并应用规则例外（如忽略肾功能评分），即使面对未见过的ICD代码（OOD），也能保持100%推导正确性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **小规模微调模型 > 大模型零样本推理**  
   LLaMA-8B微调后在所有指标上全面超越LLaMA-70B零样本表现，证明**高质量指令微调胜于规模红利**。

2. ✅ **微调可实现近乎完美的规则遵循**  
   在已知医学规则下，微调后的LLM能在新患者上实现**接近100%的推导正确性**，具备高度可信赖的推理能力。

3. ✅ **规则例外可通过合成数据学习**  
   模型能够从带ICD标签的训练数据中学会“何时应忽略某项SOFA评分”，为引入临床专家反馈奠定基础。

4. ❗ **瓶颈不在分布外泛化，而在时间上的预测（forecasting）**  
   论文指出：“the bottleneck is not out-of-distribution generalization, but [...] generalization into the future”。即：
   - 模型对**新患者**的规则应用非常稳健
   - 但对**未来临床变量**的预测仍受限于TSF本身的难度，特别是稀疏、不规则采样的lab值

5. ✅ **多模态融合显著缓解TSF瓶颈**  
   将专用TSF模型的表示融入LLM，可在不影响推理忠实性的前提下，提升未来SOFA计算的数值正确性和最终F1得分。

---

### 方法局限性
| 局限 | 说明 |
|------|------|
| **依赖人工构建规则模板** | 当前SOFA规则需手动编码为可执行逻辑，扩展到更多疾病需工程投入 |
| **例外规则为合成数据** | 当前“忽略肾脏评分”等例外是假设性设定，尚未从真实临床决策中学习 |
| **TSF仍是弱环** | 尽管多模态有所改进，但未来变量预测仍是整个链条中最不可靠的一环 |
| **仅验证Sepsis-3** | 方法虽具通用性，但实证仅在一个复杂指南上完成 |

---

### 未来工作方向
1. **从模拟例外转向真实临床反馈学习**  
   收集医生在实际诊疗中偏离指南的案例，训练模型理解合理例外。

2. **提升TSF能力**  
   开发更强大的临床时间序列预测模型，特别是针对稀疏、异步观测数据。

3. **跨共识定义迁移学习**  
   探索不同医学领域（如精神病学、神经病学）间的规则迁移潜力。

4. **任务关联学习（Task Association Learning）**  
   如论文结尾所述，研究相关共识定义之间的共享结构，实现知识迁移。

5. **开放代码与复现**  
   项目已在GitHub开源：[https://github.com/StatNLP/guideline_based_medical_reasoning_LLM](https://github.com/StatNLP/guideline_based_medical_reasoning_LLM)

---

> 💡 **一句话总结**：该论文展示了如何通过**基于共识指南的精细微调**，让LLM成为忠实、可解释的医疗助手；其真正挑战不是“理解规则”，而是“预测未来”。

</details>

---

### 16. [Physics-informed self-supervised learning for predictive modeling of coronary artery digital twins](https://arxiv.org/abs/2512.03055)

**Authors**: Xiaowu Sun, Thabo Mahendiran, Ortal Senouf, Denise Auberson, Bernard De Bruyne, Stephane Fournier, Olivier Muller, Pascal Frossard, Emmanuel Abbe, Dorina Thanou  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.03055v1  

#### Abstract
Cardiovascular disease is the leading global cause of mortality, with coronary artery disease (CAD) as its most prevalent form, necessitating early risk prediction. While 3D coronary artery digital twins reconstructed from imaging offer detailed anatomy for personalized assessment, their analysis re...

---

### 17. [Real-Time Structural Health Monitoring with Bayesian Neural Networks: Distinguishing Aleatoric and Epistemic Uncertainty for Digital Twin Frameworks](https://arxiv.org/abs/2512.03115)

**Authors**: Hanbin Cho, Jecheon Yu, Hyeonbin Moon, Jiyoung Yoon, Junhyeong Lee, Giyoung Kim, Jinhyoung Park, Seunghwa Ryu  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.03115v1  

#### Abstract
Reliable real-time analysis of sensor data is essential for structural health monitoring (SHM) of high-value assets, yet a major challenge is to obtain spatially resolved full-field aleatoric and epistemic uncertainties for trustworthy decision-making. We present an integrated SHM framework that com...

---

### 18. [Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs](https://arxiv.org/abs/2512.03994)

**Authors**: Oren Rachmil, Roy Betser, Itay Gershon, Omer Hofman, Nitay Yakoby, Yuval Meron, Idan Yankelev, Asaf Shabtai, Yuval Elovici, Roman Vainshtein  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.03994v1  

#### Abstract
Aligning proprietary large language models (LLMs) with internal organizational policies has become an urgent priority as organizations increasingly deploy LLMs in sensitive domains such as legal support, finance, and medical services. Beyond generic safety filters, enterprises require reliable mecha...

---

### 19. [DialogGuard: Multi-Agent Psychosocial Safety Evaluation of Sensitive LLM Responses](https://arxiv.org/abs/2512.02282)

**Authors**: Han Luo, Guy Laban  
**Category**: cs.AI  
**Published**: 2025-12-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.02282v1  

#### Abstract
Large language models (LLMs) now mediate many web-based mental-health, crisis, and other emotionally sensitive services, yet their psychosocial safety in these settings remains poorly understood and weakly evaluated. We present DialogGuard, a multi-agent framework for assessing psychosocial risks in...

---

### 20. [From Hypothesis to Premises: LLM-based Backward Logical Reasoning with Selective Symbolic Translation](https://arxiv.org/abs/2512.03360)

**Authors**: Qingchuan Li, Mingyue Cheng, Zirui Liu, Daoyu Wang, Yuting Zeng, Tongxuan Liu  
**Category**: cs.CL  
**Published**: 2025-12-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.03360v1  

#### Abstract
Logical reasoning is a core challenge in natural language understanding and a fundamental capability of artificial intelligence, underpinning scientific discovery, mathematical theorem proving, and complex decision-making. Despite the remarkable progress of large language models (LLMs), most current...

---

### 21. [AR-Med: Automated Relevance Enhancement in Medical Search via LLM-Driven Information Augmentation](https://arxiv.org/abs/2512.03737)

**Authors**: Chuyue Wang, Jie Feng, Yuxi Wu, Hang Zhang, Zhiguo Fan, Bing Cheng, Wei Lin  
**Category**: cs.CL  
**Published**: 2025-12-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.03737v1  

#### Abstract
Accurate and reliable search on online healthcare platforms is critical for user safety and service efficacy. Traditional methods, however, often fail to comprehend complex and nuanced user queries, limiting their effectiveness. Large language models (LLMs) present a promising solution, offering pow...

---

### 22. [AugServe: Adaptive Request Scheduling for Augmented Large Language Model Inference Serving](https://arxiv.org/abs/2512.04013)

**Authors**: Ying Wang, Zhen Jin, Jiexiong Xu, Wenhai Lin, Yiquan Chen, Wenzhi Chen  
**Category**: cs.CL  
**Published**: 2025-12-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.04013v1  

#### Abstract
As augmented large language models (LLMs) with external tools become increasingly popular in web applications, improving augmented LLM inference serving efficiency and optimizing service-level objectives (SLOs) are critical for enhancing user experience. To achieve this, inference systems must maxim...

---

### 23. [Safe and Sustainable Electric Bus Charging Scheduling with Constrained Hierarchical DRL](https://arxiv.org/abs/2512.03059)

**Authors**: Jiaju Qi, Lei Lei, Thorsteinn Jonsson, Dusit Niyato  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.03059v1  

#### Abstract
The integration of Electric Buses (EBs) with renewable energy sources such as photovoltaic (PV) panels is a promising approach to promote sustainable and low-carbon public transportation. However, optimizing EB charging schedules to minimize operational costs while ensuring safe operation without ba...

---

### 24. [ALARM: Automated MLLM-Based Anomaly Detection in Complex-EnviRonment Monitoring with Uncertainty Quantification](https://arxiv.org/abs/2512.03101)

**Authors**: Congjing Zhang, Feng Lin, Xinyi Zhao, Pei Guo, Wei Li, Lin Chen, Chaoyue Zhao, Shuai Huang  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.03101v1  

#### Abstract
The advance of Large Language Models (LLMs) has greatly stimulated research interest in developing multi-modal LLM (MLLM)-based visual anomaly detection (VAD) algorithms that can be deployed in complex environments. The challenge is that in these complex environments, the anomalies are sometimes hig...

---

### 25. [Single-Round Scalable Analytic Federated Learning](https://arxiv.org/abs/2512.03336)

**Authors**: Alan T. L. Bacellar, Mustafa Munir, Felipe M. G. Fran\c{c}a, Priscila M. V. Lima, Radu Marculescu, Lizy K. John  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.03336v1  

#### Abstract
Federated Learning (FL) is plagued by two key challenges: high communication overhead and performance collapse on heterogeneous (non-IID) data. Analytic FL (AFL) provides a single-round, data distribution invariant solution, but is limited to linear models. Subsequent non-linear approaches, like Dee...

---

### 26. [Physics-Driven Learning Framework for Tomographic Tactile Sensing](https://arxiv.org/abs/2512.03512)

**Authors**: Xuanxuan Yang, Xiuyang Zhang, Haofeng Chen, Gang Ma, Xiaojie Wang  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.03512v1  

#### Abstract
Electrical impedance tomography (EIT) provides an attractive solution for large-area tactile sensing due to its minimal wiring and shape flexibility, but its nonlinear inverse problem often leads to severe artifacts and inaccurate contact reconstruction. This work presents PhyDNN, a physics-driven d...

---

### 27. [Quantum Topological Graph Neural Networks for Detecting Complex Fraud Patterns](https://arxiv.org/abs/2512.03696)

**Authors**: Mohammad Doost, Mohammad Manthouri  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.03696v1  

#### Abstract
We propose a novel QTGNN framework for detecting fraudulent transactions in large-scale financial networks. By integrating quantum embedding, variational graph convolutions, and topological data analysis, QTGNN captures complex transaction dynamics and structural anomalies indicative of fraud. The m...

---

### 28. [Model Recovery at the Edge under Resource Constraints for Physical AI](https://arxiv.org/abs/2512.02283)

**Authors**: Bin Xu, Ayan Banerjee, Sandeep K. S. Gupta  
**Category**: cs.AI  
**Published**: 2025-12-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.02283v1  

#### Abstract
Model Recovery (MR) enables safe, explainable decision making in mission-critical autonomous systems (MCAS) by learning governing dynamical equations, but its deployment on edge devices is hindered by the iterative nature of neural ordinary differential equations (NODEs), which are inefficient on FP...

---

### 29. [Randomized Masked Finetuning: An Efficient Way to Mitigate Memorization of PIIs in LLMs](https://arxiv.org/abs/2512.03310)

**Authors**: Kunj Joshi, David A. Smith  
**Category**: cs.CL  
**Published**: 2025-12-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.03310v1  

#### Abstract
The current literature on memorization in Natural Language Models, especially Large Language Models (LLMs), poses severe security and privacy risks, as models tend to memorize personally identifying information (PIIs) from training data. We introduce Randomized Masked Fine-Tuning (RMFT), a novel pri...

---

### 30. [Mitigating hallucinations and omissions in LLMs for invertible problems: An application to hardware logic design automation](https://arxiv.org/abs/2512.03053)

**Authors**: Andrew S. Cassidy, Guillaume Garreau, Jay Sivagnaname, Mike Grassi, Bernard Brezzo, John V. Arthur, Dharmendra S. Modha  
**Category**: cs.LG  
**Published**: 2025-12-04  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.03053v1  

#### Abstract
We show for invertible problems that transform data from a source domain (for example, Logic Condition Tables (LCTs)) to a destination domain (for example, Hardware Description Language (HDL) code), an approach of using Large Language Models (LLMs) as a lossless encoder from source to destination fo...

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

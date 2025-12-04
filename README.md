# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-12-04 08:37:56 UTC
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

#### AI Summary (by qwen-long)
# VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **图神经网络（GNN）效率瓶颈**：尽管 GNN 在图分类任务中表现优异，但其依赖梯度反向传播和多层迭代训练，导致计算成本高、内存消耗大，难以部署在资源受限设备上。
- **超维计算（HDC）表达能力不足**：现有的基于 HDC 的图分类方法虽然高效，但在预测精度上普遍落后于主流 GNN，存在“效率 vs. 性能”的权衡困境。

### 🚀 提出的新方法与创新思路
本文提出 **VS-Graph**，一种结合超维计算（HDC）效率优势与消息传递机制表达力的新型向量符号图学习框架，其核心创新包括：

1. **Spike Diffusion（脉冲扩散）**
   - 一种轻量级拓扑驱动的节点标识机制。
   - 每个节点从单位脉冲开始，在图中进行多跳传播，最终根据累积响应对节点排序并赋予结构角色排名（rank），实现跨图一致的结构感知编码。

2. **Associative Message Passing（关联式消息传递）**
   - 完全在高维向量空间内执行的消息聚合机制。
   - 使用逻辑 OR 操作进行邻居信息聚合（`m⁽ˡ⁾ = ⋁ hⱼ⁽ˡ⁾`），具有幂等性，避免重复累加，无需归一化。
   - 引入残差融合更新规则：`hᵢ⁽ˡ⁺¹⁾ = α·hᵢ⁽ˡ⁾ + (1−α)·mᵢ⁽ˡ⁾`，保留历史状态。

3. **无参数训练范式**
   - 不依赖梯度优化或反向传播，仅通过一次前向编码即可完成模型构建。
   - 采用 **Prototype Classification**：将每类训练样本的图嵌入平均为类原型，推理时通过余弦相似度匹配最近原型。

### 🔍 相比现有方法的优势
| 维度 | VS-Graph | 传统 GNN | HDC 基线（如 GraphHD） |
|------|---------|----------|------------------------|
| **准确性** | ≈ 或优于 GNN | 高 | 较低 |
| **训练速度** | 极快（~450× 加速） | 慢（需 BP） | 快 |
| **可扩展性** | 支持边缘/神经形态硬件 | 受限 | 良好 |
| **鲁棒性** | 对维度压缩高度稳健（D=128 仍有效） | 一般 | 差 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
所有实验基于 **TUDataset** 中五个标准图分类基准，且仅使用图结构（无节点/边属性）：

| 数据集 | 图数量 | 类别数 | 平均节点数 | 平均边数 | 应用领域 |
|-------|--------|--------|------------|----------|-----------|
| MUTAG | 188 | 2 | 17.93 | 19.79 | 分子致突变性 |
| PTC_FM | 349 | 2 | 14.11 | 14.48 | 化合物毒性 |
| PROTEINS | 1113 | 2 | 39.06 | 72.82 | 蛋白质功能 |
| DD | 1178 | 2 | 284.32 | 715.66 | 酶结构 |
| NCI1 | 4110 | 2 | 29.87 | 32.30 | 抗癌化合物筛选 |

> 所有数据集以拓扑结构为主，不使用任何属性特征。

### ⚙️ 实验设置与评估指标
- **评估协议**：10折交叉验证 × 3次重复（stratified split）
- **评估指标**：
  - 分类准确率（Accuracy）
  - 单图训练时间（ms）
  - 单图推理延迟（ms）
- **超维向量维度范围**：D ∈ {128, 256, ..., 8192}（二进制 hypervectors）
- **实现平台**：PyTorch + DGL，运行于 NVIDIA Tesla T4 GPU / Intel Xeon CPU / 52GB RAM

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **HDC 基线** | GraphHD [28] |
| **GNN 基线** | GCN [12], GAT [13], GIN [14]（均禁用属性输入，仅用邻接结构） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（D=8192）

#### ✅ 准确率表现（Figure 1 & 文本）
| 方法 | MUTAG | PTC_FM | PROTEINS | DD | NCI1 |
|------|-------|--------|----------|----|------|
| **VS-Graph (Ours)** | **90.8%** | 80.2% | **76.5%** | **78.9%** | 80.1% |
| GraphHD | ~86% | ~75% | ~71% | ~74% | ~78% |
| GCN | 85.6% | 78.1% | 73.2% | 76.3% | 79.2% |
| GAT | 84.3% | 77.5% | 72.8% | 75.1% | 78.8% |
| GIN | 86.7% | 79.4% | 74.1% | 77.6% | **81.3%** |

> 💡 **VS-Graph 在 MUTAG、PROTEINS 和 DD 上超越所有 GNN；在 NCI1 略逊于 GIN，但仍优于其他 GNN 和 GraphHD。**

#### ⏱️ 训练效率（Table II）
| 方法 | MUTAG | PTC_FM | PROTEINS | DD | NCI1 |
|------|-------|--------|----------|----|------|
| **VS-Graph** | **0.142 ms** | **0.153 ms** | **0.230 ms** | **2.120 ms** | **0.192 ms** |
| GraphHD | 0.801 ms | 0.929 ms | 0.904 ms | 1.892 ms | 0.768 ms |
| GIN | 47.20 ms | 37.39 ms | 49.24 ms | 72.92 ms | **85.84 ms** |

> 🔥 **VS-Graph 训练速度快达 GNN 的 450×（NCI1 上）**，平均加速 >250×。

#### ⏳ 推理延迟（Table III）
| 方法 | MUTAG | PTC_FM | PROTEINS | DD | NCI1 |
|------|-------|--------|----------|----|------|
| **VS-Graph** | 0.366 ms | 0.368 ms | 0.418 ms | 2.288 ms | 0.328 ms |
| GraphHD | 0.980 ms | 1.143 ms | 1.050 ms | 2.028 ms | 0.923 ms |
| GIN | 0.641 ms | 0.552 ms | 0.405 ms | 0.521 ms | 0.374 ms |

> ✅ VS-Graph 推理延迟显著低于 GraphHD，接近甚至优于部分 GNN（除 DD 外）。DD 上因图较大略有劣势，但可通过降维缓解。

### 🔬 消融实验与鲁棒性分析（Figure 2 & 3）

#### ✅ 维度压缩鲁棒性（Figure 2）
- 当 hypervector 维度从 **D=8192 降至 D=128**：
  - **VS-Graph**：准确率下降 <1.5%，保持稳定（如 MUTAG 从 90.8% → 89.5%）
  - **GraphHD**：性能急剧下降（如 MUTAG 下降 >5%）
- 表明 VS-Graph 编码更具区分性和稳定性。

#### 📉 训练/推理延迟随维度变化（Figure 3）
- 随着 D 减小，VS-Graph 的训练和推理时间线性降低。
- 在大型图（如 DD）中，降低维度可显著提升效率，同时维持高精度。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **VS-Graph 成功弥合了 HDC 与 GNN 之间的性能鸿沟**：
   - 在多个标准图分类任务上达到甚至超过 GNN 的精度水平。
   - 同时保持 HDC 固有的高效性与低训练开销。

2. **无需梯度优化也能实现强表达能力**：
   - 通过 “Spike Diffusion” + “Associative Message Passing”，实现了多跳结构信息的有效捕获。
   - 证明非参数化方法可在图学习中具备竞争力。

3. **极高的训练效率与实时潜力**：
   - 训练速度比 GNN 快 **高达 450×**，适合大规模快速建模。
   - 推理延迟低，适用于边缘设备和实时系统。

4. **对维度压缩高度鲁棒**：
   - 即使将 D 降至 128，仍能保持良好性能，极大利于部署在内存受限的 **neuromorphic hardware** 或 **in-memory computing** 架构中。

### ⚠️ 局限性
- 当前方法仅利用图结构，未整合节点/边属性，在需要语义信息的任务中可能受限。
- 在最大规模图（如 DD）上的推理延迟略高于轻量级 GNN，需进一步优化。
- 尚未在真实神经形态芯片上验证实际加速效果。

### 🔮 未来工作方向
- 开展 **neuromorphic co-design**，开发专用 spiking 或存内计算加速器以充分发挥 VS-Graph 的脑启发特性。
- 探索支持属性输入的扩展版本（例如将标签映射为 hypervector 后绑定）。
- 将框架推广至其他图任务（如节点分类、链接预测）。

---

## 总结

> **VS-Graph 是一个兼具高性能与超高效率的图分类框架**。它通过引入 **Spike Diffusion** 和 **Associative Message Passing**，首次在无需梯度训练的前提下，实现了与现代 GNN 相当甚至更优的分类精度，同时将训练速度提升数百倍，并展现出卓越的维度压缩鲁棒性。该工作为超维计算在图学习中的应用开辟了新路径，尤其适合部署于边缘计算与神经形态硬件平台。

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

#### AI Summary (by qwen-long)
# 论文总结：OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

Mixture-of-Experts (MoE) 架构虽然在扩展 Large Language Models (LLMs) 规模方面具有显著优势，但其对 GPU 内存的需求远高于 Dense 模型（约 4–5 倍），这使得在**低功耗边缘设备**上部署 MoE 模型面临巨大挑战。

现有解决方案如 **expert offloading** 将不常用的专家参数存储在 CPU 内存中，并按需加载至 GPU。然而，这类方法通常依赖于 **GPU 上保留部分专家缓存（expert cache）** 来缓解 CPU-GPU 间 I/O 带宽瓶颈，导致：

- GPU 内存仍被大量占用；
- 缓存命中率有限，且错误预取代价高昂；
- 在内存小于 1GB 的低端设备上无法运行。

因此，如何在**极低 GPU 内存条件下实现高效、高精度的 MoE 推理**是本文要解决的核心问题。

---

### **提出了什么新方法或新思路**

作者提出 **OD-MoE** —— 一种**无需专家缓存**（cacheless）、支持**按需动态加载专家**的分布式 MoE 推理框架。其核心创新包括以下三点：

#### ✅ 创新点 1：Scaled Emulative Prediction (SEP) —— 超高精度多层前向预测器

- 引入一个轻量级、量化后的“影子模型”（shadow model，如 INT8 或 FP16 版本）与主模型并行运行。
- 影子模型推理速度更快，可提前多个 layer 预测出主模型将激活的 experts。
- 这种基于“模拟行为”的预测机制比传统基于历史统计或单层门控网络外推的方法更准确。

> **优势**：实现了高达 **99.94% 的 expert activation recall rate**，远超现有方法。

#### ✅ 创新点 2：分布式节点间的并行加载与计算调度

- 将多个边缘节点划分为若干组，采用 **round-robin 调度策略**：
  - 一组节点执行当前 layer 的 expert computation；
  - 其他组节点根据 SEP 的预测结果，提前从 CPU 加载后续 layers 所需的 experts。
- 实现了 **expert loading 与 expert computation 的跨设备并行化**。

> **优势**：充分利用多节点系统的总 I/O 吞吐能力，避免单节点 I/O 成为瓶颈。

#### ✅ 创新点 3：KV Cache 与 Token 对齐机制（Alignment Mechanism）

- 由于影子模型与主模型精度不同，在自回归生成过程中会产生输出偏差，进而影响 expert routing 准确性。
- 提出周期性地将影子模型的 **generated token 和 KV cache** 与主模型同步。
- 平衡了“预测准确性”与“alignment 带来的延迟开销”。

> **优势**：有效抑制误差累积，维持长期推理中的高预测准确率。

---

### **相比现有方法的优势**

| 维度 | 现有方法（如 Mixtral-Offloading, HOBBIT, AdapMoE） | OD-MoE |
|------|--------------------------------------------------|--------|
| **GPU 内存需求** | 需预留空间用于 expert caching（通常 >3× OD-MoE） | 仅需 ~1GB/worker，无 expert cache |
| **模型保真度** | 使用 quantization 或 expert skipping，降低性能 | 完整 FP32 精度，不牺牲质量 |
| **预测准确性** | Recall ~80–91%，Cache-hit <85% | Recall 达 **99.94%**（FP16 shadow） |
| **适用场景** | 依赖较强 GPU，难以部署到 IoT 设备 | 可运行于 <1GB GPU 的边缘/IoT 设备 |
| **系统设计** | 单节点 offloading，I/O 易成瓶颈 | 分布式并行加载，提升整体 I/O 效率 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **测试数据集**：来自 [HOBBIT](https://arxiv.org/abs/2411.01433) 的 Alpaca 子集，共 60 个高质量样本。
  - 30 个输入长度为 16 tokens；
  - 30 个输入长度为 128 tokens。
- **评估任务**：文本生成（decoding），每个 prompt 至少生成 256 output tokens。

此外还使用了多个标准 LLM benchmark 评估答案质量：
- **General Knowledge**: MMLU, Hellaswag
- **Math**: GSM8K
- **Reasoning**: ARC-Challenging, WinoGrande, BigBenchHard
- **Coding**: HumanEval, BigCode
- **Instruction Following**: MT-bench-101（由 GPT-4 打分）
- **Anti-Hallucination**: TruthfulQA

---

### **实验设置**

- **基础模型**：Mixtral-8×7B（top-2 activation, 32 layers）
- **硬件平台**：10 节点测试床
  - 主节点（main node）：RTX 3090 + AMD R7-7700，运行非 expert 模块（attention, gating 等）
  - 影子节点（shadow node）：双 RTX 3090，运行 INT8/FC16/NF4 量化版 Mixtral
  - 工人节点（worker nodes）：8 × RTX 3090，各自负责特定 expert 的加载与计算
- **网络连接**：1Gbps Ethernet LAN
- **量化配置**：Shadow model 使用 FP16 / INT8 / NF4 量化

---

### **评估指标**

| 指标类别 | 具体指标 |
|---------|--------|
| **推理速度** | 
| - Decoding Throughput | tokens/sec（解码阶段平均吞吐） |
| - TTFT (Time to First Token) | ms（prefill 阶段延迟） |
| - Overall Output Throughput | tokens/sec（含 prefill + decoding） |
| **内存消耗** | 总 GPU Memory Usage (GB)，各节点分布 |
| **预测性能** | Recall Rate（正确预测的 expert 数 / 实际激活数） |
| **模型质量** | 多项 benchmark 的准确率 / 得分（vs. full-precision baseline） |

---

### **基线方法对比**

| 类别 | 基线方法 |
|------|--------|
| **全缓存方案（数据中心参考）** | HuggingFace Transformers（全专家预加载）、llama.cpp（CPU-only） |
| **专家卸载系统** | 
| - Mixtral-Offloading [15] | 基于 LRU 缓存 + 量化 |
| - MoE-Infinity [34] | LFU 缓存策略 |
| - HOBBIT [31] | 多层预测 + 混合精度缓存 |
| - AdapMoE [42] | 自适应跳过 + 预测 |

> 注：所有 baseline 均在相同服务器（8×RTX 3090）上复现以保证公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 指标 | OD-MoE | 最优 baseline（AdapMoE） | Transformers（全缓存） |
|------|--------|--------------------------|------------------------|
| **Decoding Throughput** | **3.6925 tokens/s** | 3.1300 (AdapMoE) | 4.8900 |
| **相对速度** | — | +18% | **达到 75.5%** |
| **Output Throughput** | **3.3700 tokens/s** | 3.0350 | 4.8425 |
| **TTFT 平均值** | **2244 ms** | 1387 ms (AdapMoE) | 417 ms |
| **总 GPU 内存** | **60 GB** | ~180 GB（单卡 offload） | **1121.5 GB**（8×3090） |
| **每 worker GPU 内存** | **≤1 GB** | ≥3 GB | 不适用 |
| **Expert Prediction Recall** | **99.94%**（FP16 shadow） | ≤91%（HOBBIT） | N/A |

> ⚠️ 注意：OD-MoE 使用 10 个节点（1 main + 1 shadow + 8 workers），而大多数 baseline 仅用单 GPU。

---

### **与基线方法的对比结果**

#### 🔹 推理速度
- OD-MoE 的 decoding throughput 是 **llama.cpp 的 4.49×**，MoE-Infinity 的 **5.37×**，HOBBIT 的 **4.7×**。
- 虽然 TTFT 略慢于 Mixtral-Offloading 和 AdapMoE（因其使用量化加速），但在整体输出吞吐上仍领先。

#### 🔹 内存效率
- 相比 fully cached Transformers 方案，OD-MoE 仅使用 **约 1/18 的 GPU 内存总量**。
- 相比其他 offloading 方法，OD-MoE 的 **每 worker GPU 内存仅为 1GB**，适合嵌入式/IoT 设备。

#### 🔹 模型质量（Answer Quality）
| Benchmark | OD-MoE | AdapMoE | Transformers |
|---------|-------|--------|-------------|
| MMLU | 70.34% | 48.60% | 70.34% |
| GSM8K | 64.14% | 22.00% | 64.14% |
| HumanEval | 24.39% | 1.54% | 24.39% |
| TruthfulQA | 89.00% | 76.50% | 89.00% |
| MT-bench-101 | 7.83/10 | 4.47/10 | 7.83/10 |

✅ **OD-MoE 保持 full-precision 质量，全面超越所有 offloading 基线**

---

### **消融实验结果（Ablation Study）**

研究了不同 alignment 策略对 decoding speed 的影响（基于 (16,256) 配置）：

| 设置 | Decoding Speed (tokens/s) | 说明 |
|------|----------------------------|------|
| Case 1: Token & KV 每步对齐 | **3.616** | 最高性能，误差最小 |
| Case 2: 仅 Token 对齐 | 3.453 | 性能下降，标准差增大 |
| Case 3: 仅 KV 对齐 | 2.445 | 明显退化 |
| Case 4: 无对齐 | 1.185 | 预测 accuracy 快速衰减 |
| Case 5: 移除 shadow，随机预加载 | 1.046 | 几乎无收益 |
| Case 6: 移除 shadow，按需加载 | 1.032 | 回归传统 I/O 瓶颈模式 |

📌 **结论**：Token + KV 对齐至关重要；shadow model 是性能提升的关键组件。

进一步验证 alignment period 的影响：
- 最佳性能出现在 **token 和 KV cache 每 iteration 都对齐**（period=1）时。
- 当 worker GPU 更弱（如换成 RTX 3080）时，最优 alignment period 会变长（KV period=4），表明存在 trade-off。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **SEP 预测器实现了前所未有的 expert activation 准确率（99.94%）**，使“完全按需加载”成为可能。
2. ✅ **通过分布式并行加载 + 精准预测，成功消除 expert cache 的必要性**，极大降低边缘设备内存压力。
3. ✅ **OD-MoE 在仅使用 1/3 GPU 内存的情况下，达到 fully cached MoE 75% 的解码速度**，且不损失模型精度。
4. ✅ **worker 节点 GPU 内存需求低于 1GB**，首次实现 MoE 在低成本 IoT 设备上的可行部署。
5. ✅ **答案质量全面优于所有 offloading 基线**，证明其适用于真实应用场景。

---

### **方法的局限性**

1. ❗ **依赖多节点协同**：需要至少 10 个边缘节点构成集群，不适合单设备场景。
2. ❗ **通信开销不可忽略**：尤其在低带宽网络下（如 Wi-Fi），prefill 阶段 TTFT 较高。
3. ❗ **shadow node 开销大**：需额外 GPU 资源运行影子模型，增加系统复杂性和成本。
4. ❗ **alignment 带来“late departure”延迟**：频繁对齐会影响 pipeline 流畅性，需精细调参。

---

### **未来工作方向**

1. 🔄 **优化 alignment 策略**：设计自适应 alignment 周期控制器，动态平衡 accuracy 与 latency。
2. 🌐 **适配无线边缘网络**：考虑 WLAN 不稳定性和延迟波动下的鲁棒调度算法。
3. 💡 **轻量化 shadow model**：探索更小规模、更低资源消耗的 predictor 架构（如 distilled model）。
4. 🧩 **扩展至数据中心应用**：利用 SEP 的预测能力进行 expert replication 或 placement 优化。
5. 📱 **端侧集成**：尝试将 OD-MoE 架构移植到手机、路由器等消费级设备，推动边缘 AI 普及。

---

## 总结

> **OD-MoE 是首个实现“零专家缓存”的分布式 MoE 推理框架**，通过 **SEP 预测器 + 分布式并行加载 + 对齐机制**，在保障 full-precision 模型质量的前提下，大幅降低了边缘设备的 GPU 内存需求。它不仅在性能上显著超越现有 offloading 方法，更为 **LLM 在低功耗 IoT 设备上的落地提供了切实可行的技术路径**。

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

#### AI Summary (by qwen-long)
# 论文总结：*From monoliths to modules: Decomposing transducers for efficient world modelling*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**复杂世界模型（world models）在建模高维、纠缠环境时面临的计算效率低和可解释性差**的问题。具体而言，当一个AI代理需要在一个包含多个交互实体的环境中进行推理时，传统的“单体式”（monolithic）建模方式会将所有输入输出耦合在一起，导致学习成本高昂且难以理解其内部机制。

此外，随着AI系统被部署于高风险场景，对模型的**结构性透明度（structural transparency）和安全性分析能力**提出了更高要求。因此，如何从一个看似纠缠的整体模型中，分解出具有语义意义的模块化子系统，成为一个关键挑战。

---

### 提出的新方法与新思路

作者提出了一套基于**transducer**（一种推广了POMDP的计算模型）的**模块化分解框架**，用于将复杂的、高维的世界模型分解为一组相互作用的、低维的子transducer网络。其核心思想是利用现实世界中存在的**稀疏因果依赖结构**来实现高效建模。

#### 主要创新点包括：

- **定义了两种信息论诊断工具**：
  - **Intransducibility**：当存在潜在变量（latent variables）时，用于判断一组可观测变量是否可以作为一个transducer的输出，由其余变量驱动。
  - **Acausality**：仅基于可观测信号即可计算的度量，衡量接口是否违反非预期性（nonanticipatory），从而判断能否被实现为feedforward transducer。

- **提出了两种分解算法**：
  - **Algorithm 1 (Decomposition via Intransducibility)**：适用于已知潜变量的情况，递归地“剥离”最小可生成模块。
  - **Algorithm 2 (Decomposition via Acausality)**：无需访问潜变量，直接从观测数据中恢复模块间的因果依赖图。

- **引入了coarse-graining机制**：允许在保留外部接口的前提下合并或简化transducer网络中的节点，形成多尺度抽象表示。

- **证明了e-transducer的组合封闭性**（Theorem 2）：若每个组件都是其对应接口下的最小预测模型（即e-transducer），则它们的组合仍是复合接口下的e-transducer。这保证了模块化不会牺牲预测最优性。

---

### 相比现有方法的优势

| 维度 | 本文方法 | 现有方法（如端到端神经网络） |
|------|---------|-----------------------------|
| **可解释性** | 显式暴露模块间因果结构，支持机制层面分析 | 黑箱模型，缺乏结构透明性 |
| **计算效率** | 支持分布式、并行化推理与训练 | 单体推理，难以并行 |
| **泛化能力** | 利用模块重组支持组合泛化 | 泛化依赖大量数据微调 |
| **理论基础** | 建立在计算力学（computational mechanics）与信息论之上，具备严格数学支撑 | 多为启发式设计，缺乏形式化保障 |

此外，相比传统automata理论中的Krohn-Rhodes分解等确定性分解方法，本工作扩展至**随机过程与概率transducer**，更贴近真实世界的不确定性建模需求。

---

## 2. 核心实验方法和设置

> ⚠️ **注意**：该论文是一篇**理论导向的研究**，并未提供传统意义上的“实验”部分（如在ImageNet或Atari上测试）。它通过形式化定义、定理证明和算法描述构建了一个完整的理论框架，而非基于具体数据集的实证研究。

### 数据集
- **未使用公开基准数据集**（如MuJoCo、Procgen、Atari等）。
- 所有分析基于**合成的概率过程建模**，假设可以获得联合分布 $ \text{Pr}(X(J), R(J')) $ 或条件接口 $ \mathcal{I}[Y|X] $ 的完整统计信息。

### 实验设置与评估指标
尽管没有传统实验，但文中隐含了以下“验证路径”：

#### 方法流程：
1. 给定一个多变量联合过程（含可观测变量 $X$ 和/或潜变量 $R$）
2. 应用Algorithm 1或2进行递归分解
3. 得到一个DAG结构的transducer网络，每个节点代表一个功能子模块
4. 可进一步应用coarse-graining进行层级抽象

#### 评估逻辑（理论层面）：
- **正确性**：分解后的模块网络是否能精确再生原始接口行为？✅（由Theorem 1, 2保证）
- **最小性**：是否识别出“质数级”不可再分的模块？✅（通过prime process定义）
- **可并行性**：各模块是否可在局部历史条件下独立推断？✅（Section 5.3论证）

### 基线方法对比
虽然没有数值对比表，但文中明确指出了与以下方向的区别：

| 对比方向 | 本文方法 | 其他方法 |
|--------|--------|--------|
| **Transducer Composition** | 推广了FST的串行/并行组合（Mohri et al., 2002） | 仅限有限状态、权重半环设定 |
| **Causal Discovery** | 使用Acausality检测非马尔可夫依赖 | 如PCMCI+依赖Markov假设 |
| **Neural Modular Networks** | 提供规范性理论基础 | 如Recurrent Independent Mechanisms (Goyal et al., 2021) 缺乏形式化分解准则 |

---

## 3. 主要实验结果和性能指标

由于是纯理论工作，不存在具体的准确率、F1分数或FPS等性能指标。但论文报告了若干**关键理论成果与性质发现**，可视作“结果”：

### 关键理论结果

| 成果 | 描述 |
|------|------|
| **Theorem 1** | 一个接口是因果的（nonanticipatory）当且仅当它可以由某个transducer生成。建立了transducer作为因果接口通用建模工具的地位。 |
| **Proposition 1 & Corollary 1** | Transducer组合的算子等于其线性算子的Kronecker积；组合满足结合律但不满足交换律。为代数操作奠定基础。 |
| **Lemma 1 & 2** | Intransducibility = 0 ⇔ 存在transducer实现；Acausality = 0 ⇔ 接口可由transducer实现。提供了可计算的分解判据。 |
| **Theorem 2** | e-transducer在组合下闭合。意味着模块化不影响预测最小性，支持可扩展的模块组装。 |

### 分解效果示例（概念性）
- 若给定一个由 $N$ 个顺序连接的transducer组成的系统，则Algorithm 1会依次识别出最后一个模块为“最小子输出单元”，逐步向前剥离。
- 在稀疏依赖网络中，coarse-graining可显著减少状态空间维度，同时保持剩余模块间的接口不变。

### 消融实验（无）
- 文中未进行参数消融或架构变体比较。
- 但讨论了不同分解顺序可能导致多种有效分解方案（非唯一性），提示需引入额外偏好（如最小模块大小优先）以稳定输出。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **复杂世界模型可以被分解为模块化transducer网络**，前提是其内部存在稀疏的因果依赖结构。
2. ✅ **Intransducibility 和 Acausality 是有效的分解判据**，分别适用于有/无潜变量的场景。
3. ✅ **模块化不仅提升可解释性，还支持并行化推理与训练**，解决了单体模型的计算瓶颈。
4. ✅ **最小预测表示（e-transducer）在组合下保持最优性**，说明模块化不会牺牲预测效率。
5. ✅ **可通过coarse-graining实现多尺度建模**，适应不同粒度的任务需求。

这些发现共同表明：**世界模型不应被视为黑箱，而应作为可分解、可审计、可控制的模块化系统来设计与分析**。

---

### 方法的局限性

| 局限性 | 说明 |
|-------|------|
| **计算可行性限制** | Intransducibility 和 Acausality 需要长期历史上的联合分布估计，在实践中可能不可行（尤其对高维序列）。 |
| **静态与前馈假设** | 当前框架局限于**机械平稳（mechanically stationary）** 和 **无反馈（feedforward）** 的transducer，无法处理自适应或闭环控制系统。 |
| **分解非唯一性** | 存在多个合法的分解路径，缺乏统一标准选择“最佳”分解。 |
| **缺乏实证验证** | 尚未应用于实际训练的world model（如Dreamer、Genie）或大型语言模型中验证其有效性。 |

---

### 未来工作方向

1. **开发近似算法**：针对长程依赖和高维数据，设计基于variational approximation或有限窗口估计的实用版Intransducibility/Acausality。
2. **扩展至反馈系统**：将框架推广到包含agent-environment闭环交互的setting，支持动态adaptation建模。
3. **与神经网络结合**：探索如何将Transformer的residual stream或RNN隐藏状态映射到transducer latent space，并应用分解算法揭示其内部模块结构。
4. **应用于AI安全审计**：利用该框架自动识别模型中的“功能模块”，辅助进行意图分析、异常检测与干预策略设计。
5. **跨模态分解**：研究视觉、语言、动作等多模态输入下的联合分解策略。

---

## 总结

该论文提出了一种**基于transducer的世界模型模块化分解理论框架**，通过引入**Intransducibility**和**Acausality**两个信息论工具，实现了从单体模型到可解释、可并行、多层次模块系统的转换。虽然目前仍处于理论阶段，但它为构建**高效、透明、安全的人工智能系统**提供了强有力的数学语言和方法论基础，有望成为未来**世界模型可解释性与结构化建模**的重要基石。

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

#### AI Summary (by qwen-long)
# 论文总结：FFTrainer: Fast Failover in Large-Language Model Training with Almost-Free State Management

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大规模语言模型（LLM）训练依赖于数千甚至上万GPU组成的集群，这类系统面临频繁的节点故障和漫长的恢复时间（MTTR），导致训练效率严重下降。传统检查点（Checkpoint, CKPT）机制存在以下瓶颈：
- **检查点开销大**：完整保存模型权重、优化器状态等需要大量I/O带宽，影响正常训练性能。
- **恢复时间长**：从存储加载CKPT耗时数十分钟，且网络状态重建（如PyTorch + NCCL通信组初始化）极其缓慢。
- **回滚代价高**：低频CKPT导致失败后需回退多个迭代，造成训练进度损失。

这些问题使得即使单个GPU可靠性较高，整个集群的平均无故障时间（MTBF）也极短（例如16,384 GPU集群仅约3小时），严重影响模型FLOPs利用率（MFU）。

---

### 提出的新方法与核心思想
作者提出 **FFTrainer**，一个专为LLM训练设计的高效容错系统，其核心创新包括：

#### （1）**只备份必要状态（Checkpoint Razor）**
- 利用数据并行中的冗余性：在数据并行度 > 1 时，模型权重和优化器状态是重复的。
- FFTrainer仅保存每个数据并行组中唯一的状态（如梯度、部分参数），将CKPT大小压缩至原来的1/10以下。

#### （2）**利用空闲网络带宽进行“免费”状态传输（Neighboring Redundancy）**
- 在每次迭代中，通过ring结构将唯一状态异步复制到邻居节点的内存中（使用RDMA）。
- 不占用专用数据网络或磁盘I/O，在通信空闲期完成传输，几乎不影响训练吞吐。

#### （3）**解耦训练角色与NCCL Rank（Decoupling Roles from Ranks）**
- 将worker的逻辑训练角色（data/tensor/pipeline parallel ID）与其NCCL物理rank分离。
- 实现训练状态恢复与网络状态恢复的**并行化**，避免串行等待。

#### （4）**轻量级集体通信库 LCCL（Lightweight Collective Communication Library）**
- 替代PyTorch默认的TCPStore + NCCL初始化流程。
- 支持lock-free连接建立、group-free通信（仅维护必要的peer-to-peer连接），显著加速跨节点重连。

#### （5）**懒备份（Lazy Backup）与快速失败检测**
- 故障发生前不持久化冗余状态；仅在重启时由健康节点主动导出完整CKPT。
- 使用应用层心跳（每秒一次）替代NCCL长达10分钟的timeout机制，实现秒级故障感知。

#### （6）兼容主流框架，集成成本低
- 支持PyTorch、Megatron、DeepSpeed，仅需修改少量代码（主要是导入包名）即可接入。

---

### 相比现有方法的优势
| 维度 | 传统方案（如DeepSpeed/Gemini/MegaScale） | FFTrainer |
|------|----------------------------------------|---------|
| CKPT频率 | 每30分钟~数小时一次 | **每迭代一次（instant CKPT）** |
| CKPT开销 | 显著增加迭代时间（23%-110%） | **<3% overhead** |
| 恢复时间（MTTR） | 数百秒至千秒级 | **降至26–29秒**（降低97%） |
| 网络使用 | 单独的数据网络用于CKPT | **复用训练网络，无需专用网络** |
| 内存开销 | 高（需缓存全量状态） | 更低（仅存唯一状态） |
| 可扩展性 | 连接初始化随规模平方增长 | 接近线性扩展 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **Common Crawl** 数据集进行预训练任务。

### 模型配置
测试四种不同规模的LLM：
| 模型 | 参数量 | 数据并行度 (d) | 张量并行度 (t) | 流水线并行度 (p) |
|------|-------|----------------|----------------|------------------|
| GPT-2 | 2.7B | 16 | 4 | 2 |
| LLaMA3-8B | 8B | 4 | 4 | 8 |
| LLaMA2-13B | 13B | 2 | 4 | 8 |
| LLaMA3-70B | 70B | 2 | 8 | 8 |

所有模型使用fp16精度训练。

---

### 实验平台
- **硬件环境**：16台服务器，共128块RTX 4090 GPU
- **网络**：200 Gbps InfiniBand（IB）
- **软件栈**：CUDA 12.4, PyTorch, DeepSpeed, Megatron-LM

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Megatron / DeepSpeed原生CKPT** | 同步/异步写入磁盘，典型间隔为每5个迭代或30分钟 |
| **Gemini** | 内存级CKPT，强调快速保存，CKPT频率约为每分钟一次 |
| **MegaScale** | 专注于快速恢复，但CKPT仍受限于I/O瓶颈 |

---

### 评估指标
1. **Checkpoint Overhead**：启用CKPT前后每迭代耗时差异
2. **MTTR（Mean Time to Recovery）**：从故障发生到恢复训练的时间分解
3. **MFU Loss（Model FLOPs Utilization Loss）**：综合考虑CKPT、恢复、回滚带来的效率损失
4. **Main Memory Overhead**：CKPT占用的CPU内存
5. **Scalability**：在更大规模下的控制器负载与连接延迟

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **CKPT频率** | 每迭代一次（instant checkpointing） |
| **CKPT开销** | <3%，远低于Megatron（23%-110%） |
| **总恢复时间（LLaMA2-13B, 128 GPUs）** | **29秒**（vs. 基线994秒） |
| **MTTR降低幅度** | **高达97%~98%** |
| **MFU Loss（MTBF=3h）** | **≤0.27%**（基线最高达19%） |
| **主存开销** | 仅为Megatron的38%，Gemini的19%（LLaMA3-8B） |
| **AllReduce性能** | 达到NCCL的89%（2GB数据） |
| **连接初始化时间（128 GPUs）** | LCCL仅需17%的时间 compared to MegaScale |

---

### 🔍 详细对比结果（以LLaMA2-13B为例）

#### 表：Failover各阶段耗时对比（单位：秒）

| 步骤 | Gemini (16 GPUs) | FFTrainer (16 GPUs) | 时间减少 |
|------|------------------|--------------------|----------|
| 失败检测 | 15 | 6 | 60% |
| Pod创建 | 392 | 7 | 98% |
| 依赖安装 | 421 | 0 | 100% |
| 状态恢复与加载 | 71 | 13 | 82% |
| **总计** | **899** | **26** | **97%** |

> 注：随着GPU数量增至128，FFTrainer的连接建立时间仅增加27%，而PyTorch原生方案增加2.3倍。

---

#### MFU Loss 对比（图5）
- **Megatron**：因CKPT开销大，MFU损失严重（最高达19%）。
- **Gemini**：虽CKPT快，但缺乏快速恢复机制，仍有~6%损失。
- **MegaScale**：恢复快但CKPT频率低，回滚损失明显（~12%）。
- **FFTrainer**：结合高频CKPT与极速恢复，MFU损失**接近零**（≤0.27%）。

---

#### 消融实验分析（Table 7 & Figure 6）
- **并行度影响**：当数据并行度增大时，Megatron的CKPT开销急剧上升（最高慢44%），而FFTrainer始终保持<1.3%。
- **内存开销**：FFTrainer由于去除了冗余状态，内存占用显著更低（尤其在大模型下优势更明显）。
- **可扩展性验证**：模拟32,768 worker场景，中央控制器处理心跳和连接建立的开销呈近线性增长，**不会成为瓶颈**。

---

## 4. 关键结论和发现

### 📌 主要发现
1. **LLM训练中的失败恢复瓶颈不在MTBF，而在MTTR** —— 通过优化恢复路径，可以容忍更高频的故障。
2. **训练网络长期处于空闲状态（平均利用率1%-3%）**，完全可用于CKPT和数据分发，无需额外数据网络。
3. **现有框架的通信初始化过于重量级**（O(N²)复杂度），可通过简化协议大幅提速。
4. **状态冗余是CKPT开销的根本原因**，利用3D并行特性可极大压缩需保存的信息量。
5. **FFTrainer实现了“几乎零成本”的状态管理**：CKPT隐藏在计算中，恢复时间缩短两个数量级。

---

### ⚠️ 局限性
1. **假设fail-stop模型**：未处理拜占庭错误或静默故障。
2. **极端多点同时故障风险**：若同一数据并行组的所有节点同时宕机，则无法从内存CKPT恢复（概率极低，<1.7%）。
3. **依赖RDMA支持**：当前实现基于高性能网络（如InfiniBand），在普通以太网环境下效果可能打折扣。
4. **集中式State Controller**：虽然实验证明其可扩展，但在超大规模（>30K GPU）下仍需进一步验证。

---

### 🔮 未来工作方向
1. **引入Checkpoint压缩技术**：进一步减少内存和传输开销。
2. **支持更多并行策略**：如sequence parallelism或非均匀张量并行。
3. **探索异构容错机制**：结合partial recovery与full failover。
4. **开源发布**：作者承诺论文发表时将公开源码，推动社区采用。

---

## 总结
**FFTrainer** 是首个实现“每迭代一次检查点 + 秒级恢复”的LLM容错系统。它通过**精简状态、复用网络、解耦角色、轻量化通信**四大策略，彻底重构了传统CKPT范式，在不牺牲训练性能的前提下，将恢复时间降低97%以上，MFU损失趋近于零。该工作不仅提升了大规模训练的鲁棒性和效率，也为未来更大规模、更快硬件的LLM训练提供了可持续的容错架构基础。

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

#### AI Summary (by qwen-long)
# **论文总结：UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在移动和边缘设备上部署大型语言模型（LLMs）面临显著挑战，主要由于设备内存有限、计算资源共享以及系统负载动态变化，导致预压缩或固定大小的模型难以适应实时资源波动。传统的再压缩或存储多个副本的方法成本高昂且不切实际。

### **提出的新方法和新思路**
本文提出了 **UniQL** —— 一种统一的**后训练量化（Post-Training Quantization, PTQ）与低秩压缩框架**，支持在设备端进行可配置的结构化剪枝（pruning），实现对边缘LLM的自适应部署。

#### **核心创新点包括：**
- **统一的压缩框架**：首次将量化（Quantization）与结构化剪枝（Structured Pruning）在**单次（one-shot）云上处理流程**中结合，支持多种模型架构（Transformers、SSMs、Hybrid Models）。
- **高效的权重排序算法**：
  - 提出**无需伪逆（pseudo-inverse-free）的结构化权重排序**，避免了传统方法中高复杂度的矩阵求逆操作，速度提升达 **20×**。
  - 针对 MHSA 层设计**量化感知的奇异值分解（Quantization-aware SVD）**，减少量化误差。
  - 针对 SSM 模型提出**状态感知的权重排序（State-aware Weight Sorting）**，保护敏感的状态矩阵。
- **融合旋转位置编码核（Fused RoPE Kernel）**：通过索引融合减少内存访问开销，提升推理效率。
- **掩码LoRA微调（Masked LoRA Fine-tuning）**：在未剪枝的排序模型上进行带随机剪枝率的微调，使最终模型可在设备端灵活选择剪枝率而无需重新训练。

### **相比现有方法的优势**
| 维度 | UniQL | 现有方法（如 MoDeGPT, SVD-LLM） |
|------|-------|-------------------------------|
| **灵活性** | 支持设备端任意剪枝率（up to 35%） | 每个压缩率需单独处理 |
| **效率** | 单次云处理即可支持所有剪枝率 | 多次处理，耗时长 |
| **通用性** | 支持 Transformers、SSMs、Hybrid | 多数仅限 Transformers |
| **速度** | 剪枝+量化后吞吐提升 **2.7×–3.4×** | 加速有限或引入额外延迟 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **校准集（Calibration Sets）**：
  - `wikitext2`：用于分配层间剪枝率（Block Influence Scores）
  - `Alpaca`：用于权重排序、LoRA微调
- **评估任务**：
  - **零样本评测**：HellaSwag、PIQA、ARC（easy & challenge）、WinoGrande
  - **五样本评测**：MMLU（57个学科）
  - **代码生成**：MBPP+
- 所有评估使用 `LM-EVAL` 工具包。

### **实验设置和评估指标**
- **模型范围**：
  - **Transformers**: Llama-2-7B, Llama-3.1-8B, Qwen-2.5-7B
  - **SSMs**: Mamba-2-8B
  - **Hybrid Models**: Nemotron-H-8B, Bamba-9B-v2
- **压缩设置**：
  - 量化：**INT4（W4A16）**，embedding 和 output 层也量化为 4-bit
  - 剪枝率：0%, 15%, 25%, 35%
  - 云上处理：一次完成排序、微调、量化
  - 设备端：按需剪枝，无需重训
- **硬件平台**：
  - 云端：A6000 GPU（48GB）
  - 边缘端：Orin Nano 8GB
- **评估指标**：
  - **准确率（Accuracy）**
  - **模型大小缩减（R.size, ×）**
  - **延迟（TPOT, TTLT）**
  - **吞吐量（Token Throughput）**
  - **能耗（Joules/request）**

### **基线方法对比**
| 类别 | 基线方法 |
|------|--------|
| **结构化剪枝** | MoDeGPT [Lin et al., 2025], SVD-LLM [Wang et al., 2025b] |
| **后训练量化（PTQ）** | TRT-AWQ, TAO-HQQ, GPTQ |
| **弹性训练** | Flextron, LLaMaFlex, Jet-Nemotron |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | 结果 |
|------|------|
| **内存缩减** | **4× – 5.7×**（相比 FP16） |
| **吞吐提升** | **2.7× – 3.4×**（生成阶段） |
| **设备端剪枝灵活性** | 最高支持 **35% 剪枝率**，精度下降 <5% |
| **压缩时间** | 比 MoDeGPT 快 **22×**，比 SVD-LLM 快 **1.8×** |
| **模型大小** | Llama-3.1-8B 从 16.0GB → **2.8GB**（35% 剪枝） |

### **与基线方法的对比结果**

#### **表：与结构化剪枝方法对比（Table 2 & 4）**
| 方法 | 剪枝率 | Llama-3.1-8B 准确率 | 模型大小 |
|------|--------|---------------------|----------|
| FP16 | 0% | 74.0% | 16.0GB |
| MoDeGPT | 15% | 72.4% | 14.1GB |
| SVD-LLM | 15% | 70.5% | ~14GB |
| **UniQL (Ours)** | **15%** | **71.4%** | **~13.5GB** |
| **UniQL (Ours)** | **35%** | **62.7%** | **2.8GB** |

> ✅ 在相同剪枝率下，UniQL 精度更高；在更高压缩率下仍保持可用精度。

#### **表：与PTQ方法对比（Table 3 & 6）**
| 方法 | Llama-3.1-8B 准确率 | 模型大小 |
|------|---------------------|----------|
| TRT-AWQ (W4*) | 71.9% | 5.8GB |
| TAO-HQQ (W4*) | 72.4% | 5.7GB |
| **UniQL (W4)** | **72.9%** | **4.1GB** |

> ✅ UniQL 在更小模型尺寸下达到更高或相当精度，并支持设备端剪枝。

#### **延迟与吞吐（Table 7 & 8）**
| 平台 | 方法 | TPOT (ms) | TTLT (ms) | 加速比 |
|------|------|-----------|-----------|--------|
| A6000 | FP16 | 25.0 | 26,653.8 | 1× |
| A6000 | UniQL (0%) | 9.0 | 9,944.6 | **2.8×** |
| A6000 | UniQL (35%) | 7.3 | 8,105.4 | **3.4×** |
| Nano 8G | TAO-HQQ | 133.6 | 80,770.2 | 1× |
| Nano 8G | UniQL (35%) | 55.3 | 28,508.1 | **2.4×** |

> ✅ 显著降低延迟，尤其在边缘设备上优势明显。

### **消融实验结果（Ablation Study）**

#### **表：各组件贡献（Table 10）**
| 组件 | Llama-3.1-8B (4-bit, 25% 剪枝) |
|------|-------------------------------|
| 无任何优化 | 60.2% |
| + Masked LoRA FT | 65.0% (+4.8%) |
| + Quantization-aware SVD | **67.7% (+7.5%)** |

> 🔍 **关键发现**：
> - **Masked LoRA 微调** 提升约 2.6–3.7%
> - **Quantization-aware SVD** 是最大增益来源（+7.5%）
> - **Fused RoPE Kernel** 降低 10% 延迟（1.1× 加速）

---

## **4. 关键结论和发现**

### **主要发现**
1. **UniQL 实现了真正的“一次压缩，多端适配”**：在云上一次处理即可支持设备端任意剪枝率，极大提升了部署灵活性。
2. **结构化剪枝 + 量化协同优化效果显著**：联合优化比单独应用任一技术更具优势，尤其在低比特（INT4）下。
3. **对多种架构通用性强**：在 Transformers、SSMs、Hybrid 模型上均表现优异，验证了框架的普适性。
4. **边缘设备性能大幅提升**：在 Orin Nano 上实现 **2.7×–3.4× 吞吐提升**，且能耗降低近 **50%**（Table 16）。
5. **无需昂贵训练资源**：整个流程可在单张云GPU上完成，适合资源受限场景。

### **方法的局限性**
- **依赖校准集质量**：权重排序和微调效果受 `Alpaca` 等指令数据影响较大。
- **极端剪枝（>35%）精度下降明显**：虽支持高剪枝率，但精度损失加剧。
- **未探索动态负载自动决策机制**：如何在设备端自动选择最优剪枝率仍需外部策略支持。

### **未来工作方向**
- 扩展至更多模型架构（如 Diffusion Models、Vision Transformers）
- 探索**运行时自适应剪枝控制器**，基于系统负载自动调整压缩率
- 支持更低比特（如 INT3、INT2）以进一步压缩
- 结合稀疏注意力或KV Cache压缩，进一步降低内存占用

---

> 📦 **代码与模型已开源**：  
> GitHub 地址：[https://github.com/enyac-group/UniQL](https://github.com/enyac-group/UniQL)

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

#### AI Summary (by qwen-long)
# 论文总结：TokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving with Token Velocity

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大语言模型（LLM）服务系统广泛采用 **Prefill/Decode (PD) disaggregation** 架构以优化资源利用率。然而，当前的自动扩缩容（autoscaling）策略多借鉴自传统微服务或单体式LLM系统，依赖滞后指标（如GPU利用率）或粗粒度请求计数（如RPS），导致：
- 对突发流量反应迟缓，造成 **TTFT**（Time-to-First-Token）和 **TPOT**（Time-Per-Output-Token）SLO 违规；
- 被迫过度预置资源（overprovisioning），显著增加GPU成本。

### 提出的新方法与创新
为解决上述问题，论文提出 **TokenScale**，一个面向解耦式LLM服务的高效自动扩缩容框架，其两大核心创新如下：

#### （1）Token Velocity —— 预测性、细粒度的扩缩容指标
- **定义**：Token Velocity 表示在当前资源配置下，系统各阶段每秒能处理的最大token数量，统一衡量 **Prefill**、**Network** 和 **Decode** 阶段的工作速率。
- **三个子指标**：
  - **Prefill Velocity (Vp)**：由GPU计算能力决定，反映prefill阶段吞吐。
  - **Network Velocity (VN)**：KV-Cache通过RDMA/NVLink传输的速度。
  - **Decode Velocity (VD)**：解码器释放GPU内存的速度，取决于请求的输入/输出长度。
- **优势**：作为**领先指标**（leading indicator），能在瓶颈发生前预测系统压力，实现**主动扩缩容**，避免SLO违规。

#### （2）Convertible Decoders —— 快速响应机制
- **设计思想**：部分decoder实例可临时转换为prefiller，在流量高峰时吸收突发的prefill任务。
- **关键技术**：
  - 使用 **chunked-prefill** 策略，限制每次处理的token数；
  - 预留专用GPU内存与计算资源，确保共存的decode任务不违反TPOT SLO；
  - 转换过程仅需毫秒级（更新路由规则即可），消除新建prefiller的初始化延迟（通常3–10秒）。
- **优势**：创建了一个“弹性缓冲区”，快速应对prefill负载激增，显著降低TTFT。

### 相比现有方法的优势
| 方法类型 | 代表系统 | 缺陷 | TokenScale优势 |
|--------|--------|------|----------------|
| Request-based | AIBrix, DistServe, BlitzScale | 忽视token级瓶颈，无法区分“多请求少token”与“少请求多token”场景 | 基于Token Velocity精准识别真实负载 |
| Utilization-based | AIBrix (KPA/HPA) | GPU利用率滞后，反应慢 | 使用Token Velocity提前预测，快速响应 |
| Performance-based | Hyperflexis, EcoServe | 在SLO已违反后才触发扩缩容 | 主动预防SLO违规 |

---

## 2. 核心实验方法和设置

### 数据集
- **Azure 生产 traces**：来自微软Azure的真实LLM推理日志，包含对话（Conversation）和代码生成（Code）两类负载。
- **BurstGPT traces**：专为研究突发流量设计的真实世界数据集。
- **合成混合负载（Mixed trace）**：将Azure Conversation、Azure Code和BurstGPT按相等请求率混合，用于综合评估。

### 实验设置
- **硬件平台**：
  - **A100集群**：4节点或16节点，每节点4×A100-40GB，NVLink + InfiniBand。
  - **H100集群**：2节点，每节点8×H100-80GB。
- **软件栈**：
  - 基于 **vLLM** 和 **LMCache** 实现PD解耦架构；
  - 控制平面用Go编写，集成Prometheus监控。
- **模型**：
  - 小模型：**Llama-3.1-8B**（TP=1）
  - 大模型：**Qwen-2.5-32B**（TP=4）

### 评估指标
- **SLO Attainment Rate**：
  - **TTFT SLO**：短请求<256 tokens → 250ms；中等<1024 → 400ms；长请求<8192 → 2000ms。
  - **TPOT SLO**：固定为100ms。
- **GPU资源消耗**：总使用GPU数量，衡量成本效率。
- **生成吞吐量（Throughput）**：tokens/sec。

### 基线方法对比
| 基线 | Prefiller Scaling | Decoder Scaling |
|------|-------------------|-----------------|
| **AIBrix** | 并发请求数（Concurrency-based） | GPU内存利用率（70%阈值） |
| **BlitzScale** | RPS-based | RPS-based |
| **DistServe** | RPS-based（模拟确定阈值） | RPS-based |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（A100集群）

| 模型 | 基线SLO达标率 | TokenScale SLO达标率 | GPU节省 |
|------|---------------|------------------------|---------|
| Llama-3.1-8B | 62–88% | **89–96%** | 6–13% |
| Qwen-2.5-32B | 50–87% | **78–91%** | 4–14% |

> 在H100集群上，TokenScale将SLO达标率从43–77%提升至85–98%，GPU使用减少38–47%。

### 与基线方法的对比结果
- **SLO表现**：
  - 所有基线在突发流量下均出现严重TTFT飙升（最高达2300ms），而TokenScale仅轻微上升（约50ms），迅速恢复。
  - TokenScale的SLO达标率显著优于所有基线，尤其在高负载和长尾请求场景。
- **资源效率**：
  - 基线（尤其是AIBrix和BlitzScale）倾向于**过度预置**prefiller资源，导致GPU浪费。
  - TokenScale通过精准预测和Convertible Decoders实现更平滑的资源调度，维持高利用率。

### 消融实验结果（Ablation Study）
逐步添加组件验证各模块贡献（使用Mixed trace）：

| 配置 | TTFT SLO | TPOT SLO | 总体SLO |
|------|----------|----------|---------|
| **Baseline (DistServe)** | 87% | 80% | 78% |
| + TokenScale Prefiller Scaler | 91% | 80% | 85% |
| + TokenScale Decoder Scaler | 91% | 99% | 90% |
| + Convertible Decoders (**TokenScale全量**) | **94%** | 99% | **94%** |

> 结果表明：
> - TokenScale的**prefiller scaler**显著改善TTFT；
> - **decoder scaler**大幅提升TPOT达标率；
> - **Convertible Decoders**进一步优化突发场景下的TTFT表现。

---

## 4. 关键结论和发现

### 主要发现
1. **Token Velocity 是有效的预测性指标**：相比传统的request-based或utilization-based方法，它能更早、更准确地反映系统瓶颈，实现主动扩缩容。
2. **Convertible Decoders 显著缓解初始化延迟问题**：通过复用decoder资源处理突发prefill任务，避免了新建实例的冷启动开销，是应对突发流量的关键机制。
3. **联合设计带来协同增益**：Token Velocity提供“何时扩缩”的决策依据，Convertible Decoders提供“如何快速执行”的执行路径，二者结合实现了**及时且准确**的扩缩容。

### 方法的局限性
- **依赖输出长度预测**：Decoder数量计算依赖对output token length的预测，若预测不准可能导致轻微资源浪费（实验显示即使准确率降至50%，SLO下降仅2%）。
- **未考虑Prefix Caching**：当前工作假设无prefix caching，而实际生产系统常使用多级KV-Cache层次结构。
- **Convertible Decoder数量需离线配置**：其数量基于历史trace的burst ratio估算，缺乏完全动态调整能力。

### 未来工作方向
- **与分层KV-Cache架构协同设计**：将Token Velocity扩展至支持multi-level KVC hierarchies，联合优化缓存与扩缩容。
- **动态调整Convertible Decoder池大小**：根据实时流量模式自适应调整 convertible decoder 数量。
- **扩展至多租户场景**：支持不同SLO要求的多个客户共享资源池下的公平扩缩容。

--- 

> **总结**：TokenScale通过引入**Token Velocity**这一LLM原生的预测性指标和**Convertible Decoders**这一快速响应机制，解决了PD解耦架构下扩缩容不及时、不准确的问题，在真实生产trace上实现了**SLO达标率提升至80–96%**，同时**降低GPU成本4–14%**，为高性能、低成本的LLM服务基础设施提供了重要实践方案。

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

#### AI Summary (by qwen-long)
# 论文总结：InvertiTune: High-Quality Data Synthesis for Cost-Effective Single-Shot Text-to-Knowledge Graph Generation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 Large Language Models (LLMs) 的 **Text2KG**（文本到知识图谱）方法普遍存在以下问题：
- **计算成本高**：依赖多次迭代的 LLM 推理（iterative prompting），效率低下。
- **错误传播风险**：早期生成错误可能在后续步骤中被放大。
- **训练数据质量不足**：现有数据集中的知识图谱（KG）规模小、关系简单，无法反映真实场景的复杂性，且部分合成数据本身存在噪声。

### 提出的新方法与创新思路
作者提出 **InvertiTune** 框架，其核心思想是“**逆向构建训练数据**”并结合监督微调（SFT）来实现高效的单次推理（single-shot）Text2KG。

#### 主要创新点：
1. ✅ **逆向数据生成管道（Inverted Data Generation Pipeline）**  
   不再从文本生成 KG，而是：
   - 从大型知识库（如 Wikidata）中提取高质量子图（subgraphs）；
   - 利用 LLM 将这些子图“反向”生成对应的自然语言描述文本；
   - 构建高质量的 `(text, KG)` 配对训练样本。

2. ✅ **可控的知识图谱提取机制**  
   在子图提取过程中引入三种内联过滤（inline filtering）策略以提升质量：
   - **实体扩展黑名单（Entity Expansion Blacklist）**：防止扩展无意义实体（如“male”、“position”等）。
   - **规则化三元组过滤（Rule-based Filtering）**：移除含 URL、ID、非拉丁字符等低价值三元组。
   - **主谓唯一性约束（Subject-Predicate Uniqueness）**：确保每个 `(s, p)` 对应唯一对象，避免歧义，利于下游任务（如多跳问答）。

3. ✅ **轻量模型 + SFT 实现高效推理**  
   使用该管道生成的数据对轻量级 LLM（如 Qwen2.5-1.5B Instruct）进行 **Supervised Fine-Tuning (SFT)**，使其能够在一次前向推理中完成 KG 构造，无需反复调用大模型。

4. ✅ **发布两个高质量资源**
   - **CE12k**：一个包含 12k 样本的新数据集，平均每个 KG 包含约 25 个三元组，远超现有基准（通常 <5）。
   - **CrossEval-1200**：跨数据集评测集，用于评估泛化能力。

### 相比现有方法的优势
| 维度 | 传统方法（如 PiVe, ChatGPT） | InvertiTune |
|------|-------------------------------|------------|
| 推理方式 | 多轮 prompting + 验证 | 单次前向推理 |
| 计算开销 | 高（多次 LLM 调用） | 低（仅需一次轻量模型推理） |
| 错误传播 | 存在风险 | 减少 |
| 数据质量 | 依赖人工标注或噪声合成 | 控制性强、质量高 |
| 泛化能力 | 一般 | 更强（见实验） |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 | 特点 |
|--------|------|------|
| **CE12k (Ours)** | 本文提出的合成数据集，共 12,000 样本（训练 10,800，测试 1,200） | 平均每图 24~25 个三元组，文本更长（avg 122 tokens），贴近现实场景 |
| **KELM** | 现有 Text2KG 基准 | 小图为主（avg 3.5 triples） |
| **WebNLG+2020** | 常用 NLG 到 KG 数据集 | 规模中等，图较小 |
| **GenWiki-HIQ** | 另一公开数据集 | 图略大但仍有限 |

此外还构建了：
- **CrossEval-1200**：由 KELM、WebNLG+2020、GenWiki-HIQ 和 CE12k 各取 300 测试样本组成，用于评估跨域泛化能力。

### 实验设置
- **模型架构**：基于 Qwen2.5 系列，主要微调 **Qwen2.5-1.5B Instruct**。
- **训练方式**：标准的 Supervised Fine-Tuning（SFT），输入为文本，输出为 JSON 格式的三元组列表。
- **推理模式**：single-shot，即一次性生成完整 KG。

### 评估指标
采用多种面向图结构的自动评价指标：
- **G-BLEU (G-BL)**：将三元组视为序列，计算 BLEU 分数。
- **G-ROUGE (G-RO)**：类似 G-BLEU，使用 ROUGE。
- **G-BERTScore (G-BS)**：基于 BERTScore 的图对齐评分，衡量语义相似性。
- 所有指标报告 **F1 分数**，并提供 95% 置信区间和 Wilcoxon 显著性检验 p-value。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **Non-LLM** | OpenIE6, DeepEx |
| **LLM-based Iterative** | PiVe, GraphRAG, LightRAG |
| **Zero-shot LLM** | ChatGPT |
| **SFT-based** | AutoRE（基于 Mistral-7B 的 fine-tuned 方法） |
| **Large Unfine-tuned LLMs** | Qwen2.5-1.5B Instruct, Qwen2.5-32B Instruct |

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
| GraphRAG | 7.03 | 9.03 | 51.19 |
| LightRAG | 4.82 | 6.90 | 51.81 |
| **InvertiTune (Ours)** | **82.02** | **82.67** | **92.58** |

✅ **显著优势**：InvertiTune 在所有指标上大幅领先，且 p-value 极小（~1e-170），统计显著。

### 参数效率分析（vs 大模型）
| Model | BERTScore-P | BERTScore-R | BERTScore-F1 |
|-------|-------------|-------------|---------------|
| Qwen2.5-1.5B Instruct | 81.98 | 82.92 | 82.43 |
| Qwen2.5-32B Instruct | 82.71 | 84.46 | 83.57 |
| **InvertiTune (1.5B)** | **95.77** | **95.70** | **95.73** |

📌 **结论**：即使只用 1.5B 参数的模型，经过高质量数据微调后，性能远超未微调的 32B 大模型。

### 跨数据集泛化能力（CrossEval-1200）
| Training Dataset | G-BLEU | G-ROUGE | G-BERTScore |
|------------------|--------|---------|--------------|
| KELM | 48.21 | 52.61 | 79.21 |
| WebNLG+2020 | 38.36 | 43.55 | 79.11 |
| GenWiki-HIQ | 35.80 | 45.31 | 75.88 |
| **CE12k (Ours)** | **52.65** | **58.40** | **85.19** |

✅ **最强泛化能力**：在分布外数据上表现最好，说明 CE12k 数据更具多样性与代表性。

### 数据规模消融实验（Dataset Scale Analysis）
- 实验从 2k 到 12k 样本逐步增加训练集大小。
- 发现性能随数据量上升而提高，但在 **8k–10k 样本时趋于饱和**。
- 表明：**数据质量比数量更重要**，少量高质量样本即可达到接近最优性能。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **高质量训练数据至关重要**：通过“KG → 文本”的逆向生成流程，可以系统性地构造高质量 `(text, KG)` 配对数据，显著优于直接从文本生成 KG 的噪声合成方式。
2. 🚀 **轻量模型也能超越大模型**：只要训练数据足够好，即使是 1.5B 级别的模型也可以在 Text2KG 任务上全面超越更大、未经微调的 LLM。
3. 🌐 **更强的跨域泛化能力**：在 CrossEval-1200 上的表现证明，InvertiTune 训练出的模型具有更好的鲁棒性和迁移能力。
4. 💡 **数据质量 > 数据数量**：实验表明，在 ~10k 样本时性能已趋近上限，说明高质量样本比海量低质数据更有效。

### 局限性（Limitations）
- 当前评估仅限于 **Qwen2.5-1.5B Instruct** 模型，未探索其他架构的影响。
- 文本生成阶段仅使用 **DeepSeek-V3**，不同 LLM 可能影响最终数据质量和模型性能。
- 所有实验基于英文数据，尚未验证多语言适用性。

### 未来工作方向
- 探索更多样化的 LLM 用于文本生成（如 GPT-4、Claude 等）。
- 扩展至多语言或特定领域（如生物医学、金融）的知识图谱构建。
- 结合检索增强（RAG）进一步提升长文本理解能力。
- 开发自动化工具链支持大规模、低成本的高质量 Text2KG 数据生产。

---

> ✅ **一句话总结**：  
> **InvertiTune 通过“逆向生成 + SFT”范式，证明了高质量数据对于高效、高性能 Text2KG 系统的关键作用，在降低计算成本的同时实现了 SOTA 性能与卓越泛化能力。**

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

#### AI Summary (by qwen-long)
# 论文总结：A Preliminary Study on the Promises and Challenges of Native Top-$k$ Sparse Attention

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文聚焦于 **Large Language Models (LLMs)** 在长上下文建模中的推理效率瓶颈问题。随着 agent、多模态等应用对超长上下文（如 512K tokens）的需求增长，传统的 **full attention** 机制因 $O(N^2)$ 的计算与内存开销成为制约因素。如何在不显著牺牲性能的前提下加速长上下文推理，是当前的关键挑战。

### 提出的新方法与新思路
论文系统性地研究了 **Top-$k$ Attention** 机制在解码（decoding）和训练阶段的应用潜力，并提出以下创新点：

- **验证 Exact Top-$k$ Decoding 的有效性**：仅保留与 Query 最相关的前 $k$ 个 Key Tokens 进行注意力计算，在极低 Top-$k$ Ratio 下仍能保持甚至超越 full attention 性能。
- **提出 Native Top-$k$ Attention Training**：在 Supervised Fine-Tuning (SFT) 阶段引入 Top-$k$ Attention 内核，使模型“原生”适应稀疏注意力模式，提升推理一致性。
- **分析 Approximate Top-$k$ 算法精度影响**：定义 **Retrieval Precision** 指标，揭示近似算法精度与下游任务性能之间的正相关关系。
- **从 Entropy 视角解释机制优势**：发现 Top-$k$ SFT 能显著降低 attention 分布熵值，说明其更适合低熵、高确定性的推理场景。

### 相比现有方法的优势
| 方面 | 本文方法优势 |
|------|---------------|
| 推理效率 | 显著减少 KV Cache 访问量和 attention 计算量，降低延迟与显存占用 |
| 性能保持 | 即使在 Top-$k$ Ratio = 1%~5% 时，性能接近或优于 full attention |
| 训练-推理一致性 | 引入 native Top-$k$ SFT，缓解训练（full attention）与推理（sparse）不一致问题 |
| 可扩展性 | 为未来设计高效 approximate Top-$k$ 算法提供理论依据（如 Lightning Indexer） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **HELMET**：用于评估长上下文理解能力，包含多个子任务（如 `json_kv`, `niah_mk`, `ruler_cwe` 等），测试长度达 128K 和 8K。
- **LongBench v2**：综合性长文本基准，涵盖问答、摘要、推理等任务。
- **MATH500**, **GSM8K**, **AIME24**：数学推理任务，检验逻辑连贯性和事实准确性。
- **InfBench QA/Choice**, **TREC Coarse/Fine**, **Banking77**, **PopQA**, **TriviaQA** 等：多样化下游任务集合。

### 实验设置
- **模型基础**：
  - 主要使用 **Llama-3-8B-ProLong-512k-Instruct** 和 **Qwen3-32B**。
  - 自研变体：`Llama-3-8B-ProLong-Instruct-512K-TopK-SFT`（在 base 模型上进行 Top-$k$ SFT 微调）。
- **Top-$k$ Ratio 定义**：
  $$
  p = \frac{|K_{\text{top}}|}{N}, \quad p \in (0,1]
  $$
  其中 $K_{\text{top}}$ 是选中的 top-$k$ Keys，$N$ 是总 context 长度。
- **Approximate Top-$k$ 设置**：
  - 使用 FLASHATTENTION 实现 exact Top-$k$。
  - 对 approximate 方法（如 DeepSeek-V3.2-Exp 的 Lightning Indexer），通过控制 retrieved tokens 中 exact indices 的比例来模拟不同 precision。
- **Retrieval Precision 定义**：
  $$
  \text{Precision} = \frac{|K_{\text{approx}} \cap K_{\text{top}}|}{W}
  $$
  衡量近似检索结果与真实 top-$k$ 的重合度。

### 基线方法对比
- **Vanilla Full Attention**：原始全注意力模型（如 Llama-3-8B-ProLong-Instruct）。
- **Exact Top-$k$ Decoding**：直接在 vanilla 模型上应用 Top-$k$ 解码。
- **Top-$k$ SFT Model**：在 SFT 阶段集成 Top-$k$ attention kernel 的微调模型。
- **Approximate Top-$k$ Methods**：以 DeepSeek-V3.2-Exp 的 Lightning Indexer 为代表。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）Exact Top-$k$ Decoding 性能（图1）
- 在 **HELMET-128K** 上，当 Top-$k$ Ratio = 5% 时，`Llama-3-8B-ProLong-Instruct` 的总体性能与 full attention ($p=1$) 几乎持平；部分子任务（如 `NIAH MK3`）甚至略有提升。
- 在 **LongBench v2** 上，`Qwen3-32B` 在非思维模式下，即使 $p=0.01$（即只保留 1% 的 key），整体得分下降不到 3%，表明高度稀疏仍可维持强性能。

#### （2）Native Top-$k$ SFT 效果（图2）
- 在 **HELMET-8K** 上，相比 vanilla 模型：
  - 当 $p=0.01$ 时，Top-$k$ SFT 模型平均提升约 **+8~12 percentage points**。
  - 特别是在 `Ruler CWE`, `JSON KV` 等需要精确检索的任务上增益明显。
- 结论：**训练与推理的一致性极大释放了 Top-$k$ Decoding 的潜力**。

#### （3）Approximate Top-$k$ 精度影响（图3）
- 固定 context window = 2048，调整 retrieval precision $p$：
  - 当 precision 从 0.4 提升到 0.9 时，`TriviaQA` 和 `NIAH MK3` 等任务准确率持续上升。
  - 达到约 $p=0.8$ 后趋于饱和，说明高保真近似即可获得良好性能。
- **DeepSeek-V3.2-Exp 的 Lightning Indexer** 在 HELMET-128K 上平均 precision ≈ **60%**，但凭借大规模参数仍实现优异 end-to-end 表现。

#### （4）消融实验与统计分析
- **Entropy 分析（图5）**：
  - Top-$k$ SFT 模型在 decoding 阶段的 attention entropy 平均降低 **15–40%**（取决于任务和聚合方式）。
  - 支持假设：**Top-$k$ Decoding 更适合低熵、高置信度的任务环境**。
- **Head-level 分析（图4）**：
  - Lightning Indexer 在不同层和头上的 precision 分布较稳定，且 multi-query shared KV 设计有效逼近 per-head exact Top-$k$。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Exact Top-$k$ Decoding 是高效的推理加速策略**  
   即使在极低 Top-$k$ Ratio（如 1%-5%）下，也能在多种 long-context 和 reasoning 任务上达到与 full attention 相当甚至更优的表现。

2. ✅ **Native Top-$k$ Training 显著提升性能**  
   在 SFT 阶段引入 Top-$k$ attention kernel 可增强模型对稀疏上下文的适应能力，尤其在低 $p$ 场景下带来显著增益。

3. ✅ **Approximate Top-$k$ 精度决定性能上限**  
   Retrieval Precision 与下游任务性能呈正相关。尽管当前近似方法（如 Lightning Indexer）精度有限（~60%），但结合大模型规模仍可取得良好效果。

4. ✅ **Entropy 下降支持 Top-$k$ 机制合理性**  
   Top-$k$ SFT 导致 attention 分布更加集中（低 entropy），验证了其在确定性强的任务中更具优势的信息论解释。

### 方法的局限性
- **Exact Top-$k$ 计算复杂度高**：需预计算全部 attention score，存在 OOM 风险，依赖频繁 CPU-GPU 数据搬运。
- **FLASHATTENTION 不兼容 exact Top-$k$**：因其采用 tiled 计算避免 materialize logits，难以直接获取 attention scores。
- **Approximate 方法仍有误差**：目前主流 ANN 或轻量 indexer（如 Lightning Indexer）precision 未达理想水平（<80%），可能影响复杂推理任务。
- **训练成本增加不确定性**：将 Top-$k$ 扩展至 continue pretraining 阶段的效果尚待验证。

### 未来工作方向
- 🚀 开发更高效、高精度的 **approximate Top-$k$ retrieval 算法**（如基于 ANN + early pruning）。
- 🔍 将 Top-$k$ Attention 应用于 **pre-training 或 continue training 阶段**，探索更大规模 token corpus 下的潜力。
- 🧠 构建 **adaptive Top-$k$ 控制机制**，根据不同任务动态调节 $k$ 值或 precision 要求。
- 📈 探索与其他 KV Cache 压缩技术（如 MQA、KV quantization）的联合优化方案。
- 🤖 理论层面深化对 **sparse attention 与模型泛化能力、事实性（factuality）、幻觉抑制**之间关系的理解。

--- 

> **总结一句话**：本论文系统论证了 **native Top-$k$ sparse attention** 在长上下文 LLM 推理中的巨大潜力——不仅可通过稀疏化大幅提速，还能通过训练-推理协同优化进一步释放性能，同时从 entropy 角度提供了坚实的理论支撑，为下一代高效长上下文模型的设计指明了方向。

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

#### AI Summary (by qwen-long)
# **论文总结：Contrastive Deep Learning for Variant Detection in Wastewater Genomic Sequencing**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本论文针对**废水基因组测序中的病毒变异株检测难题**，解决以下挑战：
- 高测序噪声、低病毒覆盖率、读段碎片化；
- 缺乏标注的变异株标签（无监督场景）；
- 传统基于参考基因组的变异检测工具（如 BWA + LoFreq/iVar）难以识别新型突变，且计算开销大（每样本需数小时），无法支持实时监测。

### **提出了什么新方法或新思路**
提出了一种**完全无监督、无需参考基因组的病毒变异检测框架**，其核心是结合 **Vector-Quantized Variational Autoencoder (VQ-VAE)** 与自监督学习技术：
1. **VQ-VAE 架构**：通过离散隐空间学习基因组模式的 codebook，避免连续 VAE 的 posterior collapse 问题；
2. **k-mer tokenization**：将原始序列转换为固定长度的离散 token 序列，便于模型处理；
3. **Masked Reconstruction Pretraining**：借鉴 BERT 思想，在训练中随机遮蔽部分 token，提升对缺失/低质量数据的鲁棒性；
4. **Contrastive Fine-tuning**：在预训练编码器基础上进行对比学习（SimCLR 风格），增强嵌入表示的判别能力，用于变异株聚类。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **是否依赖参考基因组** | ✅ 完全 reference-free，可检测未知变异株 |
| **计算效率** | ⏱️ 单样本推理仅需 ~3 分钟（GPU），远快于 LoFreq (~2 小时) |
| **准确性** | 📈 99.52% token-level 重建准确率，优于标准 VAE 和 k-mer 计数 |
| **可解释性** | 🔍 学习到的 codebook 可解释为“基因组词典”，反映保守区域、突变热点等生物意义结构 |
| **下游任务支持** | 🧩 同时支持序列重建与变异株聚类，统一生成与判别目标 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 数据来源：市政污水处理厂采集的 **SARS-CoV-2 废水宏基因组测序数据**
- 数据规模：约 **100,000 条高通量测序 reads**
- 读长分布：过滤后 36–300 bp，中位长度 150 bp
- 预处理工具：使用 `Trimmomatic` 进行质量控制（LEADING:3, TRAILING:3, SLIDINGWINDOW:4:15, MINLEN:36）
- 质量验证：`FastQC` 报告显示过滤后碱基质量显著提升（从 ~28–30 提升至 35–38）

### **实验设置**
#### **模型架构参数**
| 参数 | 设置 |
|------|------|
| k-mer 大小 | k=6（词汇表大小 V=4,097） |
| 最大序列长度 | L=150 tokens |
| 编码器输出维度 | D=64 |
| Codebook 大小 | K=512 |
| 卷积层 | 2 层 Conv1D（kernel=3, hidden=256） |
| 归一化与正则化 | LayerNorm + Dropout(p=0.1) |
| Codebook 更新机制 | Exponential Moving Average (EMA, γ=0.95) |
| 损失函数 | Reconstruction Loss + Commitment Loss (β=0.1) + Entropy Regularization (λ=0.003) |

#### **训练配置**
- 优化器：AdamW（lr=2e-4, weight_decay=1e-4）
- 批次大小：32（基础 VQ-VAE），64（对比学习阶段）
- 训练轮数：50（基础训练），10（对比微调）
- 硬件：2×NVIDIA GPU，PyTorch DataParallel

#### **评估指标**
| 类型 | 指标 |
|------|------|
| **重建性能** | Token-level Accuracy, Exact Sequence Match Rate |
| **codebook 利用** | Active Codes, Codebook Utilization, Perplexity |
| **聚类质量** | Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index |
| **对比基线** | Standard VAE, LoFreq, iVar, K-mer Counting |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **基础 VQ-VAE 重建性能**
| 指标 | 数值 |
|------|------|
| Mean Token Accuracy | **99.52%** |
| Median Token Accuracy | 100.00% |
| Std. Token Accuracy | 0.69% |
| Exact Sequence Match Rate | **56.33%** |
| 测试样本数 | 6,400 |

> 💡 注：尽管 exact match 仅为 56.33%，但由于平均每个序列错误少于 1–2 个 token，说明误差集中且轻微。

#### **Codebook 使用情况**
| 指标 | 数值 |
|------|------|
| 总 code 数量 | 512 |
| 激活 code 数量 | **101** |
| Codebook Utilization | **19.73%** |
| 最高频 code 使用次数 | 172,600 次 |
| Codebook Perplexity | **52.3** |

> 🔍 表明模型高效压缩，仅用约 100 个 code 即可表达复杂变异模式；perplexity ≈50 暗示有效 code 数更少，符合生物学上保守元件主导的现象。

#### **Masked Reconstruction 鲁棒性**
- 在 20% token 被遮蔽的情况下：
  - 掩码位置重建准确率仍达 **~95%**
  - 显示模型具备强大上下文推断能力，适用于低质量废水数据

#### **Contrastive Fine-tuning 聚类效果**
| 方法 | Silhouette | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|------|------------|------------------|---------------------|
| Base VQ-VAE | 0.31 | 1.68 | 1248 |
| Contr-64 | **0.42 (+35%)** | **1.34 (-20%)** | **1876 (+50%)** |
| Contr-128 | **0.44 (+42%)** | **1.28 (-24%)** | **1972 (+58%)** |

> 📈 结果表明：
> - 对比学习显著提升聚类质量；
> - **更高维度嵌入（128-dim）带来更优分离效果**，证明 representation capacity 至关重要。

### **与基线方法对比**

| 方法 | Token Acc (%) | Time/Sample | Reference-Free | 聚类能力 |
|------|---------------|-------------|----------------|----------|
| Standard VAE | 96.8 | ~3 min | ✅ | ❌（posterior collapse） |
| LoFreq [9] | 95.2 | ~2 hrs | ❌ | ❌（依赖参考） |
| iVar [7] | 98.34 | ~1.5 hrs | ❌ | ❌（难处理混合变异） |
| K-mer Counting | 96.48 | ~1 min | ✅ | ❌（ARI≈42%，缺乏长程建模） |
| **VQ-VAE (Ours)** | **99.52** | **~3 min** | ✅ | ✅（Silhouette=0.44） |

> ✅ 综合表现最优：高精度、高效、无监督、可解释、支持聚类。

### **消融实验结果**
- **Masked Pretraining**：使模型在 20% 遮蔽下保持 ~95% 准确率，验证了对噪声和缺失数据的鲁棒性；
- **Contrastive Learning**：引入后所有聚类指标大幅提升（+35%~+58%），证明其对判别性表示的关键作用；
- **Embedding Dimension**：128-dim > 64-dim，说明更大容量有助于捕捉细粒度变异差异。

---

## **4. 关键结论和发现**

### **主要发现**
1. **离散表示优于连续表示**：VQ-VAE 成功避免 posterior collapse，实现高质量重建（99.52% token accuracy），而标准 VAE 表现明显下降。
2. **codebook 具有生物学可解释性**：分析 top codes 发现其对应 GC-rich coding 区域、AT-rich UTR、突变热点（如 spike 基因），表明模型自动发现了功能相关结构。
3. **对比学习极大提升聚类能力**：即使没有标签，通过数据增强构建正样本对，也能显著改善嵌入空间结构（Silhouette 从 0.31 → 0.44）。
4. **高维嵌入更有利变异区分**：128-dim 投影头比 64-dim 更能分离不同变异株，支持精细谱系追踪。
5. **该框架适合真实世界部署**：速度快（3分钟/样本）、无需参考基因组、抗噪能力强，适合大规模废水监测系统。

### **方法的局限性**
- **codebook 利用率偏低**（仅 19.73%）：可能存在容量过剩，未来可通过 hierarchical VQ-VAE 或动态调整策略优化；
- **聚类需人工设定 k 值**（实验中设为 10），缺乏自动确定簇数的能力；
- 当前未整合时间序列信息，无法建模变异株动态演化过程；
- 尚未在多种病原体（multi-pathogen）上验证泛化能力。

### **未来工作方向**
1. **Hierarchical VQ-VAE**：构建多尺度表示（如 lineage-level + SNP-level）以提高 codebook 利用率与可解释性；
2. **集成系统发育分析**：将 learned clusters 与已知 phylogenetic tree 对齐，验证其进化意义；
3. **扩展至 multi-pathogen surveillance**：同时检测呼吸道病毒、肠道病原体、抗生素抗性基因；
4. **实时部署优化**：进一步降低推理延迟至 <1 分钟，支持早期预警系统；
5. **引入 temporal modeling**：利用 RNN 或 Transformer 建模变异株随时间演变趋势；
6. **跨区域迁移学习**：探索模型在不同地理区域间的泛化能力，助力资源匮乏地区的公共卫生监测。

---

> 🌐 **总体评价**：本文提出的 **VQ-VAE + Contrastive Learning** 框架为废水基因组监测提供了一个**可扩展、可解释、无需参考基因组的全新范式**，有望推动全球范围内的民主化基因组流行病学发展。

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

#### AI Summary (by qwen-long)
# Beyond Additivity: Sparse Isotonic Shapley Regression toward Nonlinear Explainability  
**论文核心总结**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Shapley values** 在可解释人工智能（XAI）中的两个核心挑战提出解决方案：

1. **非加性（Non-additivity）问题**：标准 Shapley 框架假设特征子集的“价值”（payoff）是各特征贡献的线性叠加（即加性模型），但在实际应用中，由于非高斯分布、重尾、特征依赖或领域特定损失尺度等因素，这一假设常被违反，导致归因结果失真。
2. **稀疏性（Sparsity）问题**：在高维场景下，传统方法先计算稠密的 Shapley 值再进行后处理阈值化（post-hoc thresholding），效率低且可能导致不一致的解释。

### 提出的新方法：Sparse Isotonic Shapley Regression (SISR)

作者提出了 **Sparse Isotonic Shapley Regression (SISR)**，一种统一的非线性可解释性框架，其核心思想是：

- **联合学习单调变换与稀疏归因**：SISR 同时学习一个**单调变换函数 $T(\cdot)$** 来恢复加性结构，并对 Shapley 向量施加 **$\ell_0$ 稀疏约束**，实现原生稀疏性控制。
- **无需预设变换形式**：通过 **Pool-Adjacent-Violators Algorithm (PAVA)** 高效求解等张回归（isotonic regression），从数据中直接学习最优变换，避免人为指定如 log 或 sqrt 等函数。
- **优化算法简洁高效**：采用交替优化策略，每步更新均有闭式解（closed-form update），并具有全局收敛保证。

### 相比现有方法的优势

| 方面 | 传统方法 | SISR |
|------|--------|------|
| **非加性处理** | 忽略或假设加性成立 | 显式建模并恢复加性结构 |
| **稀疏性控制** | 后处理阈值或 $\ell_1$ 正则（有偏收缩） | 内生 $\ell_0$ 约束，无偏、直接控制稀疏度 |
| **变换灵活性** | 固定变换（如 log）或无变换 | 数据驱动、非参数单调变换 |
| **理论性质** | 缺乏收敛保证 | 全局收敛、易于实现 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖了多种任务和数据类型：
- **合成数据**：用于验证变换恢复能力和稀疏支持识别。
- **真实世界数据集**：
  - **Prostate Cancer**（前列腺癌）：预测癌症体积，临床变量。
  - **Boston Housing**：房价预测，使用 XGBoost 回归。
  - **South German Credit**：信用风险分类，使用 CatBoost 分类。
  - **Pima Indians Diabetes**：糖尿病诊断，使用 XGBoost 分类。

### 实验设置与评估指标

#### 评估指标
1. **Affinity (Affn)**：估计的归因向量 $\hat{\gamma}$ 与真实 $\gamma^*$ 的余弦相似度（单位范数下），衡量估计精度。
2. **Support Recovery Rate (Supp)**：正确识别非零特征的比例，衡量稀疏性恢复能力。
3. **稳定性分析**：比较不同 payoff 函数（如 MSE vs robust MSE, cross-entropy vs exponential utility）下的归因排序一致性。

#### Payoff 构造方式
- **回归任务**：使用子集模型的 $R^2$ 或负 MSE。
- **分类任务**：使用伪 $R^2$、负交叉熵或指数效用函数。
- 对树模型使用 **SAGE** 框架的干预式（interventional）性能评估。

#### 基线方法对比
- **原始 Shapley 值**（即标准加性假设下的计算）
- **带 $\ell_1$ 正则的方法**（文中指出其存在收缩偏差）
- **后处理阈值法**（post-hoc thresholding）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）变换恢复能力（Domain Adaptation）
- 图1显示，在六种不同真实变换（平方根、五次根、指数、对数、正切、正态分位数）下，SISR 学习到的 $T(v)$ 与真实 $T^*(v)$ 高度一致，证明其能准确恢复未知的非线性变换。
- 图2展示了在“赢家通吃”（winner-takes-all）机制下，SISR 成功恢复出类似 $\max$ 的强非线性变换，且估计归因与真实贡献高度相关（相关系数 ≈ 1.00）。

#### （2）稀疏性恢复（Sparsity Recovery）
- 表1 报告了不同维度 $p$ 和噪声水平 $\sigma_0$ 下的结果：
  - 当 $p=10, \sigma_0=1e-3$ 时，**Affn 达 99.6%，Supp 达 100%**。
  - 即使在高噪声（$\sigma_0=2e-1$）和高维（$p=25$）下，**Supp 仍保持在 60% 以上**，表明 SISR 能稳健识别重要特征。
- 图3 显示，随着稀疏度上界 $s$ 增大，计算成本显著上升，验证了稀疏性带来的效率优势。

#### （3）真实数据实验结果
- **Prostate Cancer**：传统 Shapley 将 `svi`（精囊侵犯）列为第三重要特征（>10% 归因），而 SISR 将其归因为接近零。独立统计检验（AIC/BIC/LASSO/p-value）均支持 SISR 结论，说明传统方法受无关特征干扰产生虚假归因。
- **Boston Housing**：当使用鲁棒 payoff（exp(-cMSE)）时，传统 Shapley 导致 `DIS` 重要性飙升，`CHAS` 变为负值；而 SISR 通过非线性变换校正，保持归因模式稳定。
- **Bank Credit & Diabetes**：在不同 payoff 下，SISR 归因高度一致，而传统 Shapley 值出现显著排名和符号变化。

---

## 4. 关键结论和发现

### 主要发现
1. **非加性是普遍现象**：即使使用标准 $R^2$ 或 cross-entropy 作为 payoff，只要存在**特征相关性**或**无关特征**，就会导致真实的 payoff 变换严重偏离线性。这是首次明确揭示这些常见因素会破坏 Shapley 加性假设。
2. **SISR 能有效恢复加性结构**：通过学习单调变换 $T(\cdot)$，SISR 将非线性、非高斯的 payoff 映射回一个满足加性假设的“工作空间”，从而恢复可解释性。
3. **稀疏性提升效率与准确性**：$\ell_0$ 约束不仅提高计算效率，还能过滤噪声特征，增强归因的统计准确性和可解释性。
4. **归因稳定性至关重要**：SISR 在不同 payoff 构造下表现出极强的稳定性，而传统 Shapley 值对 payoff 设计高度敏感，可能导致误导性结论。

### 方法的局限性
- **计算复杂度仍为 $O(2^p)$**：尽管稀疏性加速了迭代，但需枚举所有子集的 payoff（或采样近似），限制了其在极高维（如 $p > 30$）场景的应用。
- **变换解释性有限**：虽然 $T(\cdot)$ 可学习，但其本身是一个黑箱映射，难以提供直观的语义解释。
- **依赖 payoff 质量**：若 payoff 本身噪声过大或构造不合理，SISR 性能也会下降。

### 未来工作方向
- **扩展至广义线性模型（GLM）框架**：将高斯假设推广到指数族分布，构建 **Sparse Isotonic Shapley GLM**。
- **结合高阶交互项**：在变换后的空间引入 Shapley Interaction Indices，同时处理非线性和交互效应。
- **高效采样策略**：开发更智能的 coalition 采样方法，降低 $O(2^p)$ 复杂度。
- **理论分析**：进一步研究变换唯一性、收敛速率及在非独立特征下的泛化性能。

---

> **总结**：SISR 通过“学习成为加性”（learning to be additive）的理念，统一解决了 Shapley 值在现实应用中的非加性和高维稀疏性两大难题，为非线性可解释性提供了**理论严谨、计算高效、实证稳健**的新范式。

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

#### AI Summary (by qwen-long)
# 论文总结：Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在长序列推理时面临严重的内存和计算瓶颈，主要源于 **self-attention** 的二次复杂度以及不断增长的 **Key-Value (KV) Cache**。现有的内存受限推理策略存在以下问题：
- **量化（Quantization）** 和 **卸载（Offloading）**：引入高昂的协调开销。
- **基于注意力的启发式驱逐（Heuristic KV Eviction）**：依赖于近期或频繁被关注的 token，假设“近期关注即为重要”，但这在长周期任务中不可靠，可能导致关键信息过早丢失。

### 提出的新方法：TRIM-KV
本文提出 **TRIM-KV**（Token Retention for Memory-bounded KV Cache），一种新颖的、基于学习的 KV 缓存管理方法。其核心创新在于：
- **学习内在重要性（Intrinsic Importance）**：不再依赖动态的 attention 分数作为重要性代理，而是通过一个轻量级的 **retention gate** 在 token 创建时就预测其**内在重要性**。
- **保留分数（Retention Score）**：每个 token 被赋予一个标量 `β ∈ [0,1]`，该分数会随时间指数衰减，模拟人类大脑的遗忘曲线（Ebbinghaus's forgetting curve）。高 `β` 值表示该 token 长期有用，能长时间保留在缓存中。
- **简单高效的驱逐策略**：当缓存超过预设内存预算 `M` 时，直接驱逐当前保留分数最低的 token。

### 相比现有方法的优势
- **更可靠的重要性判断**：基于创建时的上下文嵌入预测长期效用，而非短期的 attention 状态，避免了因“分心”而误删重要 token。
- **训练高效**：仅需微调轻量级的 retention gates，主干模型参数冻结，训练成本低。
- **推理开销极小**：驱逐决策仅需比较简单的标量分数，无需复杂的相似度搜索或向量运算。
- **性能优越**：在多种任务上显著超越基于启发式的驱逐方法和可学习的检索方法。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了多个长上下文和长生成任务基准：
- **数学推理（Mathematical Reasoning）**：`GSM8K`, `MATH-500`, `AIME24`。
- **程序化生成（Procedural Generation）**：`LongProc`。
- **长记忆对话（Conversational Long-Memory）**：`LongMemEval`。
- **长上下文理解（Long-Context Understanding）**：`LongBenchV2`, `SCBench`。
- **训练数据集**：`OpenR1-MATH-220k`（用于数学任务）、`SynthLong`, `BookSum`, `Buddhi`（用于长上下文任务）。

### 实验设置和评估指标
- **基础模型**：主要使用 `Qwen3` 系列模型（如 Qwen3-1.7B, Qwen3-4B, Qwen3-8B）和 `DeepSeek R1 Distill` 变体。
- **内存预算（KV Budget）**：在不同固定大小的 KV Cache 上进行测试（如 128, 512, 1024, 2048, 4096 等）。
- **评估指标**：
  - 数学推理：`pass@1` 准确率。
  - 程序化生成：F1 分数、准确率等。
  - 对话和理解任务：各项子任务的准确率。

### 基线方法对比
- **启发式驱逐（Heuristic Eviction）**：`StreamingLLM`, `H2O`, `SnapKV`, `R-KV`。
- **可学习检索（Learnable Retrieval）**：`SeerAttn-R`（一种先进的 KV 检索方法）。
- **全缓存（Full KV）**：不进行任何压缩或驱逐的基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **数学推理任务**：在 `AIME24`、`GSM8K` 和 `MATH-500` 上，TRIM-KV 在相同内存预算下，性能远超所有基线。
- **程序化生成任务**：在 `LongProc` 数据集上，TRIM-KV 在多个难度级别上均表现最佳，甚至在某些设置下**超过了全缓存模型**。
- **长上下文任务**：在 `LongMemEval` 上，TRIM-KV 仅使用 25% 的 KV 预算就能达到全缓存模型的性能；在 `SCBench` 上也表现出强大的竞争力。

### 与基线方法的对比结果
- **显著超越启发式方法**：TRIM-KV 的性能优势巨大，即使在启发式方法拥有 **4倍更多 KV 预算**的情况下，TRIM-KV 仍能取得更好的结果。
- **大幅优于可学习检索方法**：相比 SOTA 的可学习检索方法 `SeerAttn-R`，TRIM-KV 实现了 **58.4% 的 pass@1 增益**。
- **超越全缓存模型**：在多个任务（如 `Qwen3-4B` on `AIME24`）中，TRIM-KV 的表现甚至超过了不进行任何压缩的全缓存模型，表明选择性保留可以作为一种有效的正则化手段，抑制无信息 token 的噪声。

### 消融实验结果
- **损失函数消融**：移除容量损失（`L_cap`）会导致性能急剧下降，证明了其对强制稀疏性和控制内存占用的关键作用。
- **门控架构消融**：使用单层 MLP 作为 retention gate 比线性投影效果更好。
- **训练数据消融**：在通用长上下文数据集上训练的 retention gate 也能很好地泛化到数学推理任务，显示出良好的跨领域适应能力。

---

## 4. 关键结论和发现

### 主要发现
1. **内在重要性是有效的代理**：通过学习 token 的内在重要性来指导驱逐，比依赖动态的 attention 分数更为可靠和有效。
2. **选择性保留即正则化**：TRIM-KV 不仅节省内存，还能提升性能，因为它主动过滤掉了冗余和无信息的 token，起到了正则化的作用。
3. **涌现的人类直觉行为**：学习到的保留策略自然地涌现出类似“sink tokens”、“滑动窗口（sliding windows）”和“要点压缩（gist compression）”等启发式行为，而无需显式设计。
4. **提供可解释性**：学习到的保留分数可以作为诊断工具，揭示不同 **layer** 和 **head** 的功能角色，例如某些 head 专门保留数字、操作符或问题描述。

### 方法的局限性
- **依赖预训练模型**：目前的方法是在冻结的预训练模型上添加 retention gates，没有将保留机制与模型的预训练过程深度融合。
- **实现限制**：当前的 KV Cache 实现通常假设同一层内所有 head 的序列长度一致，而 TRIM-KV 的理念支持更灵活的每 head 可变长度缓存，这需要底层系统的支持。

### 未来工作方向
- **联合训练**：将 retention-gated attention 机制集成到模型的预训练或后训练过程中，让模型从一开始就学习如何在有限内存下运作。
- **扩展应用**：将 retention gating 扩展到多模态输入、工具调用（tool-calling）等场景。
- **自适应预算**：开发能够跨层、跨 head 和跨任务动态分配内存预算的自适应机制。

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

#### AI Summary (by qwen-long)
# 论文总结：DVPO: Distributional Value Modeling-based Policy Optimization for LLM Post-Training

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在大语言模型（LLM）的后训练（post-training）过程中，强化学习（RL）被广泛用于对齐人类偏好。然而，现实场景中的监督信号（如人类反馈或奖励模型输出）往往存在**噪声或不完整性**，这会导致以下问题：

- **训练不稳定**：噪声污染价值估计和优势函数，导致策略更新偏差。
- **泛化能力差**：传统方法在面对分布外（OOD）任务时表现不佳。
- **保守策略**：基于最坏情况优化的方法（如 Robust Bellman）虽然稳定，但过于悲观，抑制探索，损害泛化。

### **提出了什么新方法或新思路**

作者提出 **DVPO**（Distributional Value Modeling with Risk-aware Policy Optimization），一种结合**分布值建模**与**条件风险控制理论**的新强化学习框架。

#### 核心创新点：

1. **Token-Level 分布式值建模**  
   不再预测标量价值 $V(s)$，而是学习每个 token 的**价值分布** $Z(s,a)$，通过多头分位数集成（Multi-Headed Quantile Ensemble）实现更细粒度的监督。

2. **不对称风险正则化（Asymmetric Risk Regularization）**  
   引入条件风险控制机制，分别调节价值分布的尾部：
   - **下尾收缩（Lower-tail Contraction）**：抑制由噪声引起的负向偏差，增强鲁棒性。
   - **上尾扩展（Upper-tail Expansion）**：保留高价值信号，鼓励探索多样性，提升泛化能力。

3. **分布式广义优势估计（Distributional GAE）**  
   在分位数空间中进行信用分配，保持不确定性传播的完整性，提供更丰富的学习信号。

### **相比现有方法的优势**

| 方法 | 特点 | 缺陷 | DVPO 改进 |
|------|------|------|----------|
| **PPO / GRPO** | 基于期望值优化 | 对噪声敏感，泛化不足 | 利用分布信息，增强稳定性与泛化 |
| **Robust Bellman PPO** | 最坏情况优化，稳定 | 过于保守，抑制高价值信号 | 非对称调节：既保稳健又促探索 |
| **RFQI / CQL** | 离线RL中的保守策略 | 泛化差，不适合在线微调 | 动态平衡风险与收益 |

> ✅ **DVPO 的核心优势在于实现了“鲁棒性”与“泛化性”的原则性权衡**。

---

## 2. 核心实验方法和设置

### **使用的数据集**

| 任务类型 | 数据集 | 描述 |
|---------|--------|------|
| **多轮对话** | Honor-Dialogue Dataset（自建） | 包含生活服务、交通出行、医疗健康、社交娱乐、金融服务等五个真实场景的多领域任务导向型对话数据 |
| **数学推理** | Light-R1 | 自研数学推理数据集，用于训练 |
| | MATH500, AIME24, Minerva-Math, AMC23 | 多个标准数学评测集作为测试 |
| **科学问答** | SuperGPQA | 覆盖285个研究生学科的知识密集型QA数据集 |
| | SampleQA, GPQA, Humanity's Last Exam (HLE) | 科学知识基准测试 |

> 所有训练标签均通过原始模型生成并经多数投票（majority voting）过滤得到伪标签。

### **实验设置**

- **初始化模型**：Qwen3-8B（数学/科学任务）、Qwen3-8B 微调后作为对话策略与奖励模型
- **训练步数**：500步（因部分方法随训练退化）
- **噪声来源**：
  - 对话：基于模型的奖励信号（reward model accuracy ≈ 71.8%）
  - 数学/科学：规则奖励 + 多数投票引入误差
- **硬件配置**：8×NVIDIA A100 80GB GPU

### **评估指标**

| 任务 | 主要指标 |
|------|--------|
| **数学 & 科学任务** | 准确率（Accuracy） |
| **多轮对话任务** | 三重评估体系：<br>• **TCR**（Task Completion Rate）<br>• **ACR**（Ask Completion Rate）<br>• **GCR**（Goal Completion Rate）<br>最终取各域平均（D-AVG）及总体平均（AVG） |

### **基线方法对比**

- **Baseline**：仅监督微调（SFT）后的初始模型
- **GRPO**, **PPO**：主流 RLHF 方法
- **Reinforce++**, **Dr.GRPO**：改进版方差控制方法
- **Robust Bellman PPO**：基于最小值聚合的最坏情况优化变体

---

## 3. 主要实验结果和性能指标

### **关键性能数据汇总**

#### 表1：对话任务性能（平均准确率 %）

| Method | Life Services | Transportation | Healthcare | Social | Financial | **AVG** |
|--------|---------------|----------------|------------|--------|-----------|--------|
| Baseline | 86.73 | 84.50 | 90.23 | 87.13 | 82.70 | 86.26 |
| GRPO | 30.03 | 28.57 | 28.33 | 28.90 | 27.90 | 28.75 |
| PPO | 84.20 | 86.07 | 90.87 | 81.00 | 83.87 | 85.20 |
| Reinforce++ | 87.87 | 81.07 | 77.67 | 87.13 | 85.53 | 83.85 |
| **DVPO (Ours)** | **88.13** | **87.73** | **87.67** | **87.67** | **82.73** | **86.79** |

✅ **DVPO 达到最高平均精度 86.79%**，显著优于所有基线，且在多个 OOD 领域表现稳定。

---

#### 表2 & 表3：跨领域泛化能力（科学/数学互训）

| 方法 | ID AVG | OOD AVG | ALL AVG |
|------|-------|--------|--------|
| **训练于科学 → 测试数学** | | | |
| Base | 2.96% | 58.40% | 34.64% |
| PPO | 2.76% | 63.27% | 37.34% |
| **DVPO (Ours)** | **3.83%** | **66.48%** | **39.63%** |
| **训练于数学 → 测试科学** | | | |
| Base | 58.40% | 2.96% | 34.64% |
| PPO | 58.40% | 3.16% | 34.72% |
| **DVPO (Ours)** | **66.45%** | **4.04%** | **39.70%** |

✅ **DVPO 在 ID 和 OOD 上均取得最佳表现**，尤其在 OOD 泛化方面远超其他方法。

---

### **消融实验结果（Ablation Study）**

#### 表8：不同损失组件的影响（科学任务）

| 模型变体 | ID AVG | OOD AVG | ALL AVG |
|--------|-------|--------|--------|
| Core Quantile Regression | 3.65% | 61.37% | 36.63% |
| + Distribution Consistency | 3.73% | 63.73% | 38.02% |
| + Tail Calibration | 3.74% | 64.42% | 38.42% |
| + Shift Penalization | 3.29% | 65.17% | 38.65% |
| **+ Tail Shape & Curvature (Ours)** | **3.83%** | **66.48%** | **39.63%** |

🔍 发现：
- 尾部校准（Tail Calibration）对 OOD 性能提升最大（+3.05%）
- 尾部形状与曲率正则化协同作用，带来最全面的改进
- 所有组件均有正向增益，验证了复合目标的有效性

#### 其他消融分析：

- **区间密度（Interval Density）最优为 200**（表5）：过密捕捉噪声，过疏丢失信息
- **风险间隔权重（Risk Interval Weight）最优为 0.1**（表4）：完全去除或过大均导致性能下降
- **小模型验证（Qwen3-1.7B）**（表6）：DVPO 在小模型上仍优于 PPO/GRPO，证明其可扩展性

---

## 4. 关键结论和发现

### **主要发现**

1. **分布式值建模 + 条件风险控制是解决噪声环境下 LLM 后训练的关键路径**。
2. **非对称尾部调节机制有效平衡了鲁棒性与泛化性**：
   - 下尾收缩 → 抗噪能力强
   - 上尾扩展 → 探索充分，OOD 表现优异
3. **DVPO 在真实噪声条件下表现出色**，尤其是在多轮对话和跨领域推理任务中，明显优于 PPO、GRPO 和 Robust Bellman 方法。
4. **可视化分析显示**（图7–9），DVPO 能更精准地识别关键语义词（如 "nucleus", "quarks"），并在 token 级别赋予合理的 advantage 权重，而传统方法则趋于平坦或误判。

### **方法的局限性**

1. **计算开销增加**：由于需维护多个分位数头和复杂损失项，训练成本高于标准 PPO。
2. **超参数敏感**：区间密度、风险权重等需根据任务调整，缺乏通用自动调参方案。
3. **极端噪声下仍可能失效**：当奖励严重错标或分布偏移极大时，性能仍会下降。

### **未来工作方向**

- 设计轻量化版本以适应大规模部署
- 自动化风险阈值与分布参数的选择机制
- 扩展至多智能体协作、长期规划等复杂场景
- 结合不确定性感知解码策略进一步提升推理一致性

---

> 📌 **总结一句话**：  
> **DVPO 通过引入分布式价值建模与非对称风险控制，在噪声环境中实现了鲁棒性与泛化的统一，为现实世界中 LLM 的强化学习后训练提供了可扩展、高效且稳定的解决方案。**

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

#### AI Summary (by qwen-long)
# 论文总结：Beyond Playtesting: A Generative Multi-Agent Simulation System for Massively Multiplayer Online Games

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 MMO 游戏中的**数值系统与机制设计优化**依赖于大规模线上实验（如 A/B 测试）或基于统计模型的参数调优，存在以下问题：
- **高时间成本**：调整后需数周甚至数月观察反馈；
- **高机会成本**：错误干预可能导致经济失衡、玩家流失；
- **测试限制**：重大机制变更无法通过小规模测试验证；
- **黑箱预测**：现有模拟方法缺乏微观可解释性，难以指导精细化设计。

此外，已有基于生成式智能体（Generative Agents）的研究多聚焦于孤立场景（如谈判、P2W机制），缺乏对完整游戏生态系统的建模与实证验证。

---

### 🚀 提出的新方法与思路
本文提出了一种**基于大语言模型（LLM）的生成式多智能体 MMO 模拟系统**，核心创新如下：

#### （1）**高保真玩家行为建模**
- 利用真实玩家行为日志进行 **Supervised Fine-Tuning (SFT)** 和 **Reinforcement Learning (GRPO)**，将通用 LLM 适配到特定 MMO 领域；
- 构建具备合理推理能力的 Player Agent，能模仿真实玩家在不同状态下的决策逻辑（如战斗、购买、下线等）；
- 引入三阶段微调流程：  
  ① Vocabulary Expansion（领域词表扩展）→ ② Action Planning SFT → ③ GRPO 推理增强。

#### （2）**数据驱动的游戏环境重建**
- 基于真实 gameplay logs 构建动态环境模型，包括：
  - Battle Server：预测胜负结果与收益；
  - NPC Shop / Black Market：作为货币回收机制（currency sink）；
- 支持外部干预模块（如引入交易市场），用于评估机制变更的因果效应。

#### （3）**可解释、可监控的大规模仿真框架**
- 提供 GUI 监控界面（见 Fig.1），支持宏观统计（财富分布、活跃度）与微观个体行为追踪；
- 所有 agent 决策附带 reasoning 过程输出，实现“白盒”分析，超越传统统计模拟的“黑箱”局限。

---

### 🔍 相比现有方法的优势
| 维度 | 传统统计模型 | 简化模拟系统 | 本工作 |
|------|--------------|-------------|--------|
| 行为真实性 | 低（聚合建模） | 中（规则驱动） | **高（LLM+真实数据驱动）** |
| 可解释性 | 差（黑箱） | 一般 | **强（显式推理链）** |
| 干预评估能力 | 弱 | 有限 | **支持复杂机制变更的因果推断** |
| 微观洞察力 | 无 | 少量 | **支持逐 agent 分析** |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- 来自某真实 MMO 游戏（类似 *Escape from Tarkov* 类型）的百万级玩家行为日志；
- 包含事件类型：
  - 登录/登出
  - 战斗记录（胜负、收入）
  - 购买/出售物品
  - 社交互动
- 时间跨度覆盖多个赛季（2025 S1 用于训练，S2 用于验证）

---

### ⚙️ 实验设置与评估指标

#### （1）Player Agent 性能评估
- **任务**：下一动作预测（四分类）——`{offline, battle, buy, sell}`
- **输入上下文**：历史行为序列 + 用户画像（playstyle, skill level 等）
- **评估指标**：
  - Stepwise Prediction Accuracy
  - 类别分布一致性（Fig.4a）
- **模型架构**：
  - Base Model: Qwen2.5-1.5B
  - 微调方式：LoRA (rank=16, α=0.2)
  - 对比模型：
    - DeepSeek-V3（未微调 baseline）
    - PlayerAgent-GRPO（仅 SFT + RL）
    - PlayerAgent-Profile-GRPO（加入 profile 信息）

#### （2）Battle Server 验证
- **目标**：预测每场比赛的胜率与收入
- **方法**：基于玩家聚类后的 profile 进行回归与分类建模
- **聚类类别**（5类典型玩家）：
  1. Stable Development Players
  2. Novice Players
  3. Wealth-Accumulating Elite Players
  4. Casual Players
  5. High-skill Players
- **验证策略**：跨赛季验证（S1 训练 → S2 测试），防止数据泄露

#### （3）干预案例研究（Case Study）
- **干预内容**：在游戏中引入官方 Black Market（替代非正式交易）
- **评估方式**：
  - 对比干预前后“非正式交易”比例变化
  - 观察 agent 是否采纳新机制
- **可视化工具**：前端监控面板（Fig.1b）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）Player Agent 动作预测准确率（Fig.4b）
| 模型 | 准确率 | 提升幅度 |
|------|-------|---------|
| DeepSeek-V3（Base） | 38.69% | — |
| PlayerAgent-GRPO | 47.03% | +8.34% |
| **PlayerAgent-Profile-GRPO** | **48.88%** | **+10.19%** |

✅ 结果表明：**结合用户画像的 GRPO 微调显著提升预测精度**

#### （2）Battle Server 预测效果（Fig.5b & c）
- 在 2025 S2 赛季中，对五类玩家的：
  - **胜率预测**：趋势高度一致（尤其 Elite 与 Stable Development 玩家）
  - **单场收入预测**：平均收入曲线匹配良好（存在固定偏移，可通过校准消除）
- 新手与休闲玩家波动较大 → 反映其行为不确定性更高

#### （3）干预实验结果（Fig.4c & d）
- 引入 Black Market 后：
  - 非正式交易占比从 **27.4% 下降至 1.5%**
  - 多数 agent 快速转向使用安全交易平台
  - 极少数因习惯延续旧模式（体现行为惯性）

✅ 显示系统具有良好的**因果响应能力**，能够模拟真实世界政策干预的影响

---

### 🔬 消融实验结果
虽然文中未明确标注“ablation study”，但从对比实验可得出以下结论：
- **加入用户画像信息** → 提升 +1.85% 准确率（证明个性化建模重要性）
- **GRPO 强化推理过程** → 显著改善决策连贯性与合理性（见 Fig.3 示例反思）
- **领域词表扩展** → 解决 OOV 问题，确保术语理解正确（如 “AWM” 武器名）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 可被有效适配为高保真的虚拟玩家代理**：
   - 经过 SFT + RL 微调后，agent 能复现真实玩家的行为模式与决策逻辑；
   - 输出包含 reasoning chain，提供可解释的设计洞察。

2. **系统具备强大的因果干预模拟能力**：
   - 成功再现了“上线 Black Market 抑制欺诈交易”的现实因果路径；
   - 支持机制设计前的风险预演与影响评估。

3. **宏观与微观双重验证有效**：
   - 宏观统计（财富分布、活动频率）稳定且符合实际；
   - 微观层面每个 agent 的行为轨迹均可追溯与分析。

4. **Battle Server 具备跨赛季泛化能力**：
   - 对多数玩家群体的战斗表现预测准确，适用于长期经济模拟。

---

### ⚠️ 局限性
1. **计算资源消耗较高**：
   - 大规模 LLM agent 并行运行需要较强算力支撑；
   - 当前系统依赖异步 coroutine 优化，仍可能存在延迟瓶颈。

2. **部分玩家群体建模精度不足**：
   - 如 Novice 与 Casual 玩家行为波动大，预测误差相对较高；
   - 可能与其动机不稳定、策略不成熟有关。

3. **Black Market 设计本身引入风险**：
   - 若税率配置不当，可能引发通货膨胀或市场垄断（文中提及但未深入模拟极端情况）。

4. **尚未完全闭环验证**：
   - 当前验证基于历史数据回放，尚未在真实游戏中部署建议并反向验证效果。

---

### 🔮 未来工作方向
1. **引入更多社会交互机制**：
   - 支持团队协作、公会竞争、社交影响力传播等更复杂的 multi-agent dynamics。

2. **构建自动优化 loop**：
   - 将模拟系统接入自动化调参 pipeline，实现“仿真 → 优化 → 部署 → 反馈”闭环。

3. **轻量化 agent 架构探索**：
   - 使用小型化模型（如 TinyLlama + MoE）降低推理成本，提升可扩展性。

4. **跨游戏迁移能力研究**：
   - 探索是否可在不同 MMO 类型间迁移 agent 模型，减少重复训练开销。

5. **对抗性测试与压力测试集成**：
   - 模拟外挂、刷钱、机器人账号等异常行为，辅助反作弊机制设计。

---

> 💡 **总结一句话**：  
> 该论文首次实现了**基于 LLM 的高保真、可解释、可干预的 MMO 多智能体仿真系统**，为游戏数值设计提供了超越传统 playtesting 的强大工具，标志着 AI 在游戏设计自动化领域的实质性进展。

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

#### AI Summary (by qwen-long)
# 论文总结：*Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective*

## 1. 主要贡献和创新点

### 解决的问题
该论文针对**将强化学习（Reinforcement Learning, RL）应用于扩散大语言模型（diffusion Large Language Models, dLLMs）时存在的根本性挑战**。传统RL方法（如GRPO）依赖于自回归（autoregressive, AR）模型的token-level条件概率进行重要性采样（importance sampling），而dLLMs通过非自回归的迭代去噪过程生成序列，缺乏这种token-level的概率分解。因此，直接套用AR-RL方法会导致不一致和不稳定。

现有方法尝试使用启发式近似（如mean-field或token-level ELBO）来桥接这一差距，但这些方法在理论上存在缺陷，且在实践中表现不佳。

### 提出的新方法：ESPO
作者提出了 **ELBO-based Sequence-level Policy Optimization (ESPO)**，一种专为dLLMs设计的**原则性序列级RL框架**。其核心创新点如下：

- **序列级动作空间（Sequence-level Action Space）**：  
  不再将每个token视为独立动作，而是将**整个序列的生成视为单一动作**。这更符合dLLMs的整体生成特性。

- **ELBO作为序列似然代理（ELBO as Likelihood Proxy）**：  
  利用证据下界（Evidence Lower Bound, ELBO）作为不可计算的序列对数似然（log-likelihood）的可计算代理，构建序列级的重要性比率（importance ratio）。

- **稳定性增强技术**：
  - **逐token归一化（Per-token Normalization）**：对ELBO差值按序列长度归一化，避免因序列长度导致的指数级数值爆炸或消失。
  - **鲁棒的KL散度估计（Robust KL Estimation）**：采用`k2`估计器替代传统的`k3`估计器，避免指数项带来的梯度不稳定问题。

### 相比现有方法的优势
- **理论一致性**：避免了在token级别分解ELBO所带来的理论不一致性。
- **训练稳定性**：通过归一化和`k2`估计器，实现了大规模稳定训练。
- **性能优越**：在数学推理、代码生成和规划任务上显著优于现有的dLLM-RL基线方法。

---

## 2. 核心实验方法和设置

### 数据集
实验覆盖三大类任务，使用以下公开基准：
- **数学推理**：GSM8K, MATH
- **代码生成**：HumanEval, MBPP（及其增强版HumanEval+, MBPP+）
- **规划任务**：Countdown, Sudoku（合成数据训练）

### 实验设置
- **基础模型**：在两个开源dLLMs上验证方法有效性：
  - `LLaDA-8B-Instruct`
  - `Dream-7B-Instruct`
- **训练配置**：
  - 最大序列长度为256，但在128、256、512三种长度下评估以测试泛化能力。
  - 使用Monte Carlo（MC）采样估计ELBO，MC样本数通常设为2。
  - 采用antithetic sampling和coupled sampling降低方差。
- **评估指标**：
  - 数学与代码任务：Pass@1准确率
  - 规划任务：任务完成率（如Sudoku解题正确率）

### 基线方法对比
- `diffu-GRPO (d1)`：基于mean-field近似的token-level RL方法
- `wd1`：加权策略优化版本
- 原始模型（无RL后训练）
- 部分结果还与更大规模的`LLaDA-1.5`进行比较

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1 & 2）

| 任务 | 方法 | 性能提升（△） |
|------|------|----------------|
| **GSM8K (Avg.)** | ESPO vs LLaDA | **+6.1** |
| **MATH (Avg.)** | ESPO vs LLaDA | **+2.5** |
| **Countdown (Avg.)** | ESPO vs LLaDA | **+62.3** |
| **Sudoku (Avg.)** | ESPO vs LLaDA | **+70.3** |
| **HumanEval (Avg.)** | ESPO vs LLaDA | **+2.3** |
| **MBPP (Avg.)** | ESPO vs LLaDA | **+7.6** |

> 注：所有提升均为相对于未经过RL后训练的基础模型。

### 与基线方法的对比结果
- 在**规划任务**（Countdown和Sudoku）上，ESPO取得压倒性优势：
  - 在Countdown任务上比最强基线（wd1）高出**20–40个绝对百分点**。
  - 在Sudoku任务上，准确率从约20–25%提升至**超过80%**。
- 在**数学和代码任务**上，虽然提升幅度较小（+2~7 pts），但仍**稳定超越所有token-level基线**，且性能接近更大规模的LLaDA-1.5模型。

### 消融实验结果
#### （1）动作空间与似然代理对比（Fig. 1）
- **Token-level + Mean Field** 和 **Token-level + ELBO**：学习效果差或不稳定。
- **Sequence-level + ELBO（本文方法）**：唯一实现快速、稳定收敛并达到最高奖励的方法。

#### （2）KL估计器对比（Fig. 2, Table 5）
- `k3`估计器：奖励停滞，训练失败。
- `k1`估计器：初期上升但最终崩溃。
- `k2`估计器：稳定高效，是唯一可行选择。
- 进一步验证：即使将`k2`用于`d1`基线（`d1 + k2`），性能仍远低于`ESPO`（见Table 5），说明**序列级框架本身是性能跃升的关键**。

#### （3）MC样本数影响（Fig. 4）
- 更多MC样本（如M=4）可加速信号丰富任务（如Sudoku）的收敛，但对稀疏奖励任务（如Countdown）帮助有限。

#### （4）策略更新频率（policy update value μ）
- 方法对μ值具有强鲁棒性，在不同μ下均能收敛到高奖励水平。

---

## 4. 关键结论和发现

### 主要发现
1. **序列级视角是dLLM-RL的正确范式**：  
   将整个序列生成视为单一动作，并结合ELBO代理，构成了一个**原则性强、理论一致且实践有效**的RL框架。

2. **token-level近似存在根本缺陷**：  
   无论是mean-field还是token-level ELBO分解，都无法准确反映dLLM的生成机制，导致训练不稳定和性能受限。

3. **稳定性设计至关重要**：  
   序列级ELBO差值需进行长度归一化，且必须使用`k2`等鲁棒KL估计器才能实现稳定训练。

4. **全局一致性任务受益最大**：  
   在需要整体逻辑一致性的规划任务（如Sudoku、Countdown）上，ESPO展现出巨大优势，证明了其捕捉长程依赖的能力。

### 方法的局限性
- 当前方法依赖于ELBO作为似然代理，虽经验上有效，但仍是一个**下界近似**，可能存在偏差。
- 训练成本仍较高，尤其是生成阶段（非自回归采样）主导总耗时。
- 实验主要集中在特定类型的dLLM（Masked Diffusion Models），是否适用于其他扩散形式需进一步验证。

### 未来工作方向
- 探索更精确的序列似然估计方法。
- 结合高效推理技术（如KV Cache）进一步优化训练效率。
- 将ESPO扩展到多模态dLLMs或其他复杂决策任务中。
- 研究如何将SFT与RL更有效地结合，以进一步提升性能。

---

> **代码已开源**：https://github.com/ML-GSAI/ESPO

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

#### AI Summary (by qwen-long)
# 论文总结：Training and Evaluation of Guideline-Based Medical Reasoning in LLMs

## 1. 论文的主要贡献和创新点

### 解决的问题
当前医学领域的大型语言模型（LLMs）研究多聚焦于提升预测准确率，而忽视了**可信赖的解释性（faithful explanations）**，这限制了其在临床实践中的可信度和采纳度。医生需要能够理解并验证模型推理过程的系统，而非仅提供“黑箱”预测。

### 提出的新方法与创新思路
本文提出了一种**基于共识指南（consensus guidelines）的医疗推理训练与评估框架**，其核心创新包括：

- **将医疗共识规则转化为可监督学习的“verbalized rule instantiations”**：通过模板自动生成针对具体患者数据的推理链（类似“scratchpad”），使LLM能从实例中学习如何逐步应用医学规则。
- **引入自动化的分步评估机制**：定义两个关键评估维度：
  - **Derivation Correctness**：衡量模型是否正确地从前提推导出结论（逻辑忠实性）。
  - **Value Correctness**：比较模型预测值与真实临床测量值的一致性（数值准确性）。
- **支持对规则例外（exceptions）的学习**：通过合成带有ICD-10编码的共病条件数据，训练模型识别并处理标准规则之外的特殊情况（如慢性肾病忽略肾脏SOFA评分）。
- **多模态融合改进时间序列预测瓶颈**：结合专用的时间序列预测模型（TSF）输出与LLM，缓解因临床变量稀疏采样导致的预测难题。

### 相比现有方法的优势
- **优于Prompting方法**：微调的小模型（LLaMA 8B）显著优于提示大模型（LLaMA 70B one-shot）。
- **优于通用医学预训练模型**：优于在医学文本上预训练的Me-LLaMA等模型。
- **实现近乎完美的逻辑一致性**：在derivation correctness上接近100%，确保推理过程符合医学规范。
- **可扩展性强**：该框架适用于任何有明确共识指南的医学领域（如精神疾病、神经系统疾病等）。

---

## 2. 核心实验方法和设置

### 数据集
- **MIMIC-III**：包含44,858条ICU住院记录，筛选出至少24小时ICU停留且年龄≥18岁的患者。
- **特征数量**：131个动态临床变量 + 年龄、性别（共133维）。
- **任务目标**：早期预测脓毒症（Sepsis-3定义）。
- **数据划分**：
  - 微调集：15,000样本
  - 开发集：3,000样本
  - 测试集：3,000样本（阳性率7.33%）

### 实验设置
- **基础模型**：`Llama-3-8B-Instruct`
- **微调方式**：使用LoRA进行参数高效微调。
- **时间序列预测（TSF）模块**：
  - 基于Transformer的编码器-解码器架构。
  - 输入为过去24小时观测，预测未来24小时值。
- **多模态融合策略**：
  - 将TSF模型的decoder输出作为“token embeddings”拼接到文本embedding前，供LLM使用。
  - 联合更新TSF、连接层（MLP）和LLM适配器。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Derivation Correctness** | 推理链每一步是否符合共识规则（逻辑正确性） |
| **Forced Derivation Correctness** | 强制使用正确上下文后继续生成，测试模型是否仍能保持一致 |
| **Value Correctness** | 预测数值与真实值偏差是否在±5%以内 |
| **SEPSIS Accuracy/Sensitivity/F1** | 最终脓毒症分类性能 |

### 基线方法对比
| 方法 | 类型 |
|------|------|
| `one-shot`, `one-shot-70B` | 大模型零样本推理（含规则提示） |
| `me-llama` | 在医学文献上预训练的8B模型（含指南） |
| `deepseek` | 推理优化的8B模型 |
| `fine-tuned` | 本文提出的LoRA微调方法 |
| `pipeline` | 先TSF再输入LLM（固定TSF） |
| `multimodal` | 多模态联合训练（本文最佳方案） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Tables 1, 3, 4）

#### Derivation Correctness（逻辑正确性）
| 方法 | SOFA1:24 | SOFA25:48 | SOFAdiff | SEPSIS |
|------|----------|-----------|---------|--------|
| one-shot | 0.543 | 0.518 | 0.845 | 0.859 |
| one-shot-70B | 0.893 | 0.868 | 0.891 | 0.909 |
| me-llama | 0.777 | 0.768 | 0.769 | 0.769 |
| **fine-tuned** | **1.000** | **1.000** | **1.000** | **1.000** |
| **multimodal** | **0.999** | **1.000** | **1.000** | **1.000** |

> ✅ 所有微调方法均达到近乎完美逻辑一致性。

#### Value Correctness（数值准确性）
| 方法 | MAP (当前) | MAP (未来) | PaO2/FiO2 (未来) | Urine (未来) |
|------|------------|------------|------------------|-------------|
| one-shot | 0.036 | 0.110 | 0.073 | 0.064 |
| **fine-tuned** | **0.993** | **0.274** | **0.565** | **0.133** |
| **multimodal** | **0.994** | **0.294** | **0.596** | **0.183** |

> ⚠️ 当前值预测极佳，但未来值预测仍有挑战，尤其稀疏变量（如尿量）。

#### 最终SEPSIS分类性能
| 方法 | Accuracy | Sensitivity | F1 Score |
|------|----------|------------|----------|
| one-shot | 0.834 | 0.331 | 0.231 |
| one-shot-70B | 0.857 | 0.118 | 0.108 |
| me-llama | 0.763 | 0.455 | 0.220 |
| **fine-tuned** | **0.868** | **0.263** | **0.254** |
| **pipeline** | **0.873** | **0.336** | **0.272** |
| **multimodal** | **0.886** | **0.386** | **0.309** |

> 📈 多模态方法取得最优综合性能。

### 消融实验结果
- **异常处理能力测试（Table 2）**：
  - 在引入ICD-10指示的共病条件下，模型仍保持100% derivation correctness。
  - 表明模型能有效学习并应用规则例外。
- **forced derivation correctness**：
  - 即使强制使用正确历史，one-shot模型表现仍远低于微调模型，说明其内部推理不稳定。
- **多模态 vs Pipeline**：
  - 多模态联合训练略优于pipeline方式，表明端到端优化更有效。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **小规模微调模型 > 大模型零样本提示**：经过规则实例化微调的LLaMA 8B在逻辑忠实性和最终性能上全面超越LLaMA 70B one-shot。
2. ✅ **微调是实现忠实推理的关键**：只有通过监督微调，才能保证模型推理过程严格遵循医学共识规则。
3. 🔍 **瓶颈不在泛化，而在时间预测**：模型在**out-of-distribution generalization**上表现良好，真正的挑战是**generalization into the future**——即对稀疏、不规则采样的临床变量进行准确TSF。
4. 🔄 **多模态融合可缓解TSF瓶颈**：将专用TSF模型的表示注入LLM，能显著提升未来值预测质量，进而改善最终诊断性能。
5. 🧩 **可学习规则例外**：通过合成数据，模型能学会在特定先决条件下调整标准规则（如忽略慢性肾病患者的肾SOFA），为引入临床专家反馈奠定基础。

### 方法的局限性
- **依赖高质量共识指南**：方法有效性受限于是否存在明确、结构化的医学共识。
- **TSF仍是短板**：尽管多模态有所改进，但对未来临床变量的预测精度仍不足，尤其是实验室指标。
- **合成异常数据**：目前对“例外”的处理基于假设场景，尚未在真实临床修正数据上验证。
- **计算成本较高**：多模态联合训练需大量GPU资源（11×A100）。

### 未来工作方向
- 改进时间序列预测模型，特别是针对**稀疏、异步采样数据**的建模能力。
- 从模拟异常转向**真实世界临床反馈数据**的学习（如医生对AI建议的修正）。
- 探索**跨领域迁移**：利用任务关联学习（task association learning）将在一个共识指南上学到的能力迁移到其他相关指南。
- 构建更大规模的**自动化规则实例化流水线**，覆盖更多疾病领域。

> 💡 总结：本文展示了如何通过**规则实例化微调 + 分步评估 + 多模态增强**，构建既准确又可信赖的医疗LLM系统，为AI辅助诊断提供了通往临床落地的新路径。

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

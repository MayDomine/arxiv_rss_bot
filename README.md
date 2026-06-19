# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-19 10:09:55 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Lagrange: An Open-Vocabulary, Energy-Based Sparse Framework for Generalized End-to-End Driving](https://arxiv.org/abs/2606.20274)

**Authors**: Shihao Ji, HongXi Li, Zihui Song, Mingyu Li  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.20274v1  

#### Abstract
Scaling end-to-end autonomous driving to complex, open-world environments requires perceptual models that generalize to anomalous scenarios and planners that produce kinematically valid trajectories. Existing paradigms face a distinct dichotomy between representational efficiency and generalization ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Lagrange: An Open-Vocabulary, Energy-Based Sparse Framework for Generalized End-to-End Driving

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前端到端（end-to-end, E2E）自动驾驶系统面临两大核心矛盾：
- **密集模型**（如Occupancy Networks、BEV Grids）具有良好的几何建模能力，但计算开销大、语义推理弱，且难以处理异常场景（out-of-distribution, OOD）。
- **稀疏查询模型**（如SparseDrive）效率高，但依赖于闭集分类（closed-set categories），对未见过的对象（如散落货物、异形车辆）存在漏检风险。
- **Vision-Language-Action (VLA)** 模型虽具备开放词汇（open-vocabulary）理解能力，但其自回归、离散token生成机制与车辆动力学所需的连续、高频控制不兼容。

### 🚀 提出的新方法与创新思路
作者提出 **Lagrange** —— 一种结合开放词汇感知与物理约束轨迹规划的稀疏、能量驱动框架，核心创新如下：

#### （1）**Open-Vocabulary Sparse Tokenizer**
- 利用 **Vision-Language Model (VLM)** 将任意视觉区域提案（region proposals）编码为连续的、类别无关的语义向量（semantic visual tokens）。
- 不依赖预定义类别标签，实现对未知对象的泛化检测。

#### （2）**Intent-Driven Masked Latent Fields (MLF) Reasoner**
- 引入基于意图的掩码交叉注意力机制，动态过滤与当前驾驶目标无关的实体。
- 通过 ego-state 和导航路径递归更新 `q_intent` 查询向量，模拟人类驾驶员的选择性注意。

#### （3）**Lagrangian Action Minimization 轨迹优化**
- 将决策过程建模为在隐式连续能量场 $E(x,y)$ 上最小化拉格朗日作用量（action）的问题：
  $$
  T^* = \arg\min_T \int_0^H \left[ \sigma E(p(t)) + \sum_j \lambda_j K_j(t) \right] dt
  $$
- 其中 $E(p)$ 是由 VLM 语义映射而来的势能项，$K_j$ 是动能惩罚项（如速度偏差、加速度、jerk），确保轨迹符合非完整运动学约束。

---

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **语义泛化性** | 支持 open-vocabulary 推理，有效应对 OOD 场景（如未知障碍物） |
| **计算效率** | 基于稀疏 token 表示，避免密集 BEV 网络的高计算成本 |
| **物理可行性** | 显式引入动力学约束，保证轨迹平滑且可执行 |
| **可解释性** | 可视化能量场热力图，直观展示危险区域与安全路径 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 用途 |
|-------|------|
| **nuScenes** | 主要用于评估标准闭集场景下的性能（baseline E2E driving） |
| **CODA** | 针对长尾分布和 corner cases 的挑战性数据集，测试 OOD 鲁棒性 |
| **Waymo Open Dataset** | 用于 zero-shot 跨域迁移测试，验证泛化能力 |

---

### 🧪 实验设置与评估指标

#### 评估任务
- **轨迹预测质量**：L2 位移误差（Avg. L2）
- **安全性**：Collision Rate (CR)，特别是 OOD 场景下的 CRood
- **实时性**：推理帧率（FPS）
- **鲁棒性测试**：
  - 加入高斯噪声（10% Visual Noise）
  - 模拟单摄像头失效（1 Camera Drop）

#### 对比基线方法
| 方法 | 类型 | 特点 |
|------|------|------|
| **UniAD [1]** | Dense, BEV | 统一密集 BEV 架构代表，强感知但计算昂贵 |
| **SparseDrive [2]** | Sparse, Whitelist | 稀疏查询架构，高效但受限于闭集分类 |
| **OpenVLA-Car [3]** | VLA, Direct Action | 直接语言到动作映射，支持 open-vocabulary 但缺乏连续控制 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### 表1：在 **CODA** 数据集上的 OOD 性能比较
| Method | Paradigm | Avg. L2 ↓ | CRood (%) ↓ |
|--------|----------|-----------|--------------|
| UniAD | Dense, BEV | 1.85 | 28.4 |
| SparseDrive | Sparse, Whitelist | 1.62 | 31.2 |
| OpenVLA-Car | VLA, Direct Act | 2.10 | 19.5 |
| **Lagrange (Ours)** | **Sparse, MLF Field** | **1.34** | **8.7** |

> ✅ **Lagrange 在 OOD 碰撞率上显著优于所有基线（降低约 60–70%）**

---

#### 表2：在 **nuScenes** 上的标准闭集性能与效率
| Method | CR (%) ↓ | FPS ↑ | Parameters |
|--------|------------|--------|-------------|
| UniAD | 0.31 | 4.2 | ~120M |
| SparseDrive | 0.28 | 18.5 | ~85M |
| OpenVLA-Car | 0.85 | 1.2 | ~7B |
| **Lagrange (Ours)** | **0.25** | **24.3** | ~150M |

> ✅ **同时实现最低碰撞率与最高推理速度（24.3 FPS），适合车载部署**

---

#### 表3：**Zero-Shot 跨域迁移**（nuScenes训练 → Waymo测试）
| Method | Zero-Shot Avg. L2 ↓ | Zero-Shot CR (%) ↓ |
|--------|------------------------|----------------------|
| UniAD | 2.14 | 1.24 |
| SparseDrive | 1.98 | 1.15 |
| **Lagrange (Ours)** | **1.52** | **0.45** |

> ✅ **无需微调即展现卓越跨域适应能力，碰撞率下降超60%**

---

#### 表4：**传感器扰动鲁棒性测试**（nuScenes）
| Method | Clean Input | 10% Noise | 1 Camera Drop |
|--------|--------------|------------|----------------|
| UniAD | 0.31 | 0.89 | 1.45 |
| SparseDrive | 0.28 | 1.12 | 2.30 |
| **Lagrange (Ours)** | **0.25** | **0.42** | **0.58** |

> ✅ **在相机失效下仍保持 <1% 碰撞率，体现结构冗余与动态注意力调整能力**

---

### 🔍 消融实验结果（Ablation Study）

#### 表5：组件消融对 CRood 与 Jerk 的影响
| VLM Tokenizer | Intent Mask | Kinematics | CRood (%) ↓ | Jerk (m/s³) ↓ |
|---------------|-------------|------------|--------------|----------------|
|               | ✓           |            | 32.1         | 1.2            |
|               |             |            | 14.5         | 1.5            |
|               | ✓           |            | 10.2         | 4.8            |
| **✓**         | **✓**       | **✓**      | **8.7**      | **0.9**        |

> 🔹 移除任一组件均导致性能显著下降：
> - 无 VLM Tokenizer → OOD 检测崩溃（CRood↑至32.1%）
> - 无 Intent Mask → 注意力分散，语义错配
> - 无 Kinematic Regularization → 轨迹抖动严重（jerk↑至4.8）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **开放词汇 + 稀疏表示** 可兼顾语义泛化性与计算效率，突破传统闭集限制。
2. **将 VLM 输出嵌入连续能量场** 是连接高层语义与底层动力学的有效桥梁。
3. **基于拉格朗日作用量最小化的轨迹优化** 天然满足车辆运动学约束，提升安全性与舒适性。
4. **Intent-driven attention 机制增强了上下文聚焦能力**，减少干扰信息的影响。
5. **系统具备高度可解释性**：可通过可视化 $E(x,y)$ 能量场验证模型是否正确识别危险源（见 Figure 2）。

---

### ⚠️ 局限性
1. **依赖几何驱动的 RPN**：对于无明确边界的非结构性危险（如黑冰、积水路面）可能无法生成有效 RoI。
2. **未建模长期语义记忆**：当前为帧间独立处理，缺乏对环境状态的持续跟踪与推理。
3. **边缘设备延迟仍有优化空间**：尽管已轻量化，VLM 对齐模块仍可在 ISP 层进一步压缩。

---

### 🔮 未来工作方向
1. 引入 **free-space segmentation token** 作为“能量地板”，补充离散对象 token，增强对连续危险区域的覆盖。
2. 探索将 VLM 对齐模块 **蒸馏至图像信号处理器（ISP）**，实现更低延迟的前端特征提取。
3. 扩展至多智能体交互建模，利用能量场进行社会力类推演（social force analogy）。
4. 结合世界模型（world model）实现长期规划与反事实推理。

---

## ✅ 总结
**Lagrange** 成功融合了 **Vision-Language Models 的开放词汇泛化能力** 与 **Energy-Based Models 的物理一致性约束**，构建了一个**稀疏、高效、安全、可解释**的 end-to-end 自动驾驶框架。它不仅在标准和长尾场景中全面超越主流方法，还在跨域迁移与抗干扰方面展现出强大鲁棒性，为迈向 L3/L4 级真实世界自动驾驶提供了新的理论与工程范式。

</details>

---

### 2. [SAC: Disaggregated KV Cache System for Sparse Attention LLMs with CXL](https://arxiv.org/abs/2606.19746)

**Authors**: Ruiyang Ma, Teng Ma, Junru Li, Hantian Zha, Xuchun Shang, Qingda Hu, Zheng Liu, Xinjun Yang, Tao Ma, Guojie Luo  
**Category**: cs.DC  
**Published**: 2026-06-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.19746v1  

#### Abstract
The scaling of LLMs toward long-context inference has shifted the primary serving system bottleneck from computation to memory capacity. Traditional solutions for dense attention models rely on RDMA-based disaggregated memory pools, which perform coarse-grained fetching of the entire prefix KV cache...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SAC: Disaggregated KV Cache System for Sparse Attention LLMs with CXL》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

随着大语言模型（LLMs）向**长上下文推理**发展，传统基于 **RDMA** 的**解聚式 KV Cache 系统**在服务**稀疏注意力模型**（如 DeepSeek-V3.2、GLM-5.1）时暴露出两大根本性瓶颈：

- **(P1) 传输瓶颈（Transmission Bottleneck）**：  
  RDMA 系统需在解码前将整个前缀 KV Cache 从远程内存池预取到本地，导致高并发场景下网络带宽饱和，造成显著排队延迟，影响 **TTFT（Time to First Token）** 和吞吐量。

- **(P2) 本地内存浪费（Local Memory Wasting）**：  
  稀疏注意力模型在每一步仅访问少量 top-k KV 条目，但 RDMA 仍需将全部 KV Cache 加载至本地 GPU HBM 或主机 DRAM，造成 TB 级内存资源浪费，限制了高并发能力。

### 提出了什么新方法或新思路

论文提出了 **SAC（Sparse Attention on CXL）** ——首个专为稀疏注意力模型优化的、基于 **CXL** 的解聚式 KV Cache 系统。

其核心思想是：

> 利用 **CXL** 的低延迟、缓存行粒度（cache-line granularity）的 **load/store 语义**，在推理过程中按需实时拉取所需的 top-k KV 条目，而非一次性全量预取。

#### 主要创新点包括：

- **按需细粒度拉取（On-demand Fine-grained Fetching）**：  
  在每个 attention 层动态计算 top-k 索引后，直接通过 CXL 从解聚内存池中读取所需 KV 向量，避免全量传输。

- **统一 CXL 内存资源管理**：  
  将 KV 数据和元数据（metadata）均置于全局可访问的 CXL 共享内存中，实现 **zero-protocol overhead** 的元数据同步，替代传统的 RPC 或 RDMA 调用。

- **与 HiSparse 框架深度集成**：  
  基于 SGLang 中的 HiSparse 架构，扩展其实现 CXL 后端支持，构建软硬件协同设计的稀疏推理系统。

### 相比现有方法的优势

| 维度 | RDMA-based 系统 | SAC (CXL-based) |
|------|------------------|------------------|
| **传输效率** | 高开销，整块预取，小包性能差 | 仅拉取必要数据，细粒度高效 |
| **本地内存占用** | 必须驻留完整 KV Cache | 仅缓存热数据，节省本地内存 |
| **访问延迟** | 微秒级，协议栈复杂 | 接近 DRAM 延迟（1.04–1.64×） |
| **适用性** | 适合稠密注意力 | 专为稀疏注意力优化 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **ShareGPT dataset** [30]：用于生成真实用户对话请求轨迹，模拟实际推理负载。
- 请求输入长度覆盖 **16K 到 128K tokens**，输出固定为 **1K tokens**（主实验），并额外测试 2K、4K、8K 输出长度以验证鲁棒性。

### 实验设置

- **硬件平台**：
  - 服务器：2×Intel Xeon Platinum 8575C CPU，2TB DDR5 DRAM
  - GPU：8×NVIDIA H20（共 768GB HBM）
  - CXL 内存池：2TB，由两个 CXL Type-3 设备组成，通过 **XConn XC50256 CXL Switch** 连接
  - RDMA 对照组：使用 ConnectX-7 100Gbps NICs，采用 loopback 配置模拟理想化 RDMA 性能（保守估计）

- **软件框架**：
  - 基于 **SGLang** 和 **HiSparse** 实现稀疏注意力推理
  - 模型：**DeepSeek-V3.2**（AWQ 4-bit 量化）
  - Tensor Parallelism = 8，Data Parallelism = 8

### 评估指标

- **Output Token Throughput**（输出令牌吞吐量，tokens/s）
- **TTFT**（Time to First Token）
- **TBT**（Time Between Tokens）
- **p99 尾延迟**
- **请求吞吐量**（requests/s）

### 基线方法对比

| 基线 | 描述 |
|------|------|
| **RDMA Memory Pool** | 当前主流解聚方案，全量预取 KV Cache |
| **Local DRAM** | 上限基准，KV 存于本地 2TB DRAM，无网络开销 |
| **GPU Only** | KV 完全驻留 HBM，容量受限 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Round-2，Decoding Stage）

在 **128K 上下文、64 并发** 下，SAC 相比 RDMA 基线取得显著提升：

| 指标 | SAC 表现 | 提升倍数（vs RDMA） |
|------|--------|---------------------|
| **Throughput** | ~1800 tokens/s | **2.1× 更高** |
| **TTFT** | 显著降低 | **9.7× 更低** |
| **TBT** | 显著降低 | **1.8× 更低** |

> SAC 在性能上接近本地 DRAM 上限，仅比其低 **9% 吞吐量**，证明 CXL 可作为“远程 DRAM”使用。

### 与基线方法的对比结果

- **图 10（Round-2）**：  
  SAC 在所有上下文长度下均大幅优于 RDMA，尤其在长上下文（64K/128K）时优势更明显。

- **图 11（吞吐可扩展性）**：  
  SAC 吞吐随并发线性增长，而 RDMA 很快达到瓶颈。在 128K 场景下，SAC 最高可达 **3.1× 更高吞吐**。

- **图 12（非解聚对比）**：  
  SAC 性能接近本地 DRAM 基准，远超 GPU-only 方案（后者因 HBM 容量限制无法扩展）。

### 消融实验结果

#### ✅ CXL 设备交错策略（Device Interleaving）

- **方法**：将 KV Cache 分布在多个 CXL 设备上进行条带化存储。
- **结果**（图 13）：
  - 平均提升 **9.2% 吞吐**
  - 在 128K 上下文下峰值增益达 **14.2%**
- **原因**：缓解链路争用，提升聚合带宽利用率。

#### ✅ GPU 缓冲区大小影响（device_buffer_size）

- **配置**：比较 `4K` vs `6K` 缓冲区
- **结果**（图 14）：
  - `6K` 配置平均吞吐提升 **10.4%**
- **原因**：更大的缓冲区减少缓存未命中率，降低 CXL 传输压力。

---

## 4. 关键结论和发现

### 主要发现

1. **RDMA 不适用于稀疏注意力模型的 KV 管理**：  
   其消息传递语义和粗粒度传输机制导致严重传输瓶颈和内存浪费。

2. **CXL 是稀疏注意力系统的理想基础设施**：  
   凭借其接近 DRAM 的延迟和细粒度 load/store 语义，CXL 支持高效的 on-demand top-k KV 拉取。

3. **SAC 实现了近乎本地内存的性能**：  
   在真实稀疏模型（DeepSeek-V3.2）上，SAC 仅比本地 DRAM 基准慢 9%，远胜 RDMA。

4. **解聚式架构不再牺牲性能**：  
   SAC 证明了解聚内存可以同时实现**高容量共享**与**高性能访问**，打破传统权衡。

### 方法的局限性

- **当前仅验证于 DeepSeek-V3.2**：尚未在其他稀疏模型（如 GLM-5.1、DeepSeek-V4）上全面测试。
- **尾延迟较高**：在高并发下，CXL fabric 内部仲裁可能导致 p99 延迟上升（见图 23–24）。
- **依赖 CXL 硬件生态成熟度**：目前 CXL 3.0/switch 技术仍在演进中，大规模部署尚需时间。

### 未来工作方向

- 扩展支持更多稀疏注意力架构（如 DeepSeek-V4 的混合压缩注意力）。
- 进一步优化内存层次结构，结合 HBM、DRAM、CXL 构建多级缓存体系。
- 探索 **processing-near-memory** 架构，在 CXL 设备内执行部分索引计算。
- 研究跨机柜（rack-scale）CXL 拓扑下的调度与容错机制。

---

> **总结一句话**：  
> SAC 通过引入 **CXL** 替代 **RDMA**，实现了面向稀疏注意力 LLM 的高效解聚式 KV Cache 系统，在保持高并发的同时逼近本地内存性能，为下一代长上下文、高稀疏性 LLM 服务提供了优越的架构基础。

</details>

---

### 3. [Online Dynamic Batching with Formal Guarantees for LLM Training](https://arxiv.org/abs/2606.19989)

**Authors**: Dian Li, Zekun Wang, Yaoru Wang, Jiahong Yan  
**Category**: cs.DC  
**Published**: 2026-06-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.19989v1  

#### Abstract
Modern LLM training breaks a core assumption behind offline batch samplers: the true training cost of a sample is only observable after preprocessing, augmentation, templating, tokenization, and multimodal visual-token expansion. Unless one pays for a preprocessing- and augmentation-dependent length...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Online Dynamic Batching with Formal Guarantees for LLM Training》核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大语言模型（LLM）训练中的**动态批处理（dynamic batching）面临一个核心矛盾**：  
传统离线批处理方法（offline batch samplers）依赖于样本长度的静态假设，而实际训练中，样本的真实成本（如 token 数量）只有在经过预处理（preprocessing）、增强（augmentation）、模板化（templating）、分词（tokenization）和多模态视觉-文本扩展（multimodal visual-token expansion）后才能确定。

这导致：
- **固定批次大小（fixed-batch）** 方法要么因填充（padding）浪费资源，要么因批次过大导致显存溢出（OOM）。
- **离线长度缓存（length cache）** 方法虽然能优化批次，但需要为特定的数据变换策略（transform/template/cutoff policy）预先计算并存储长度信息，一旦策略变更，缓存即失效，维护成本高昂。

### 提出的新方法：Online Dynamic Batching (ODB)

作者提出 **Online Dynamic Batching (ODB)**，一种**纯 DataLoader 层面的“即插即用”（drop-in）系统**，其核心思想是将批次形成（batch formation）推迟到**长度信息可被准确观测之后**。

#### 创新点
1.  **时机创新**：将批处理从数据加载早期移到了数据管道末端，此时样本的真实长度已知，从而实现了基于真实成本的动态批处理。
2.  **同步机制创新**：提出了 **Distributed Group Alignment Problem (DGAP)** 来形式化描述分布式训练中各进程（rank）因独立生成不同数量的组（groups）而导致的同步问题。
3.  **协议创新**：设计了一种 **Max-Based Bidirectional Group Alignment 协议**，该协议保证了：
    - **Deadlock-free bounded termination**：无死锁且有界终止。
    - **默认 join 模式下的严格身份覆盖（strict identity coverage）**：确保每个样本都被处理一次。
    - **可选非 join 模式下的样本配额闭合（sample-quota closure）**：保证总发出的样本数符合预期。
4.  **零侵入性**：ODB **无需修改模型、优化器、注意力内核（attention-kernel）或数据集**，仅通过包装 PyTorch DataLoader 实现。

### 相比现有方法的优势
- **高吞吐量**：相比标准固定批次方法，实现了 1.58-3.78× 的吞吐量提升。
- **高质量**：在显著提升速度的同时，保持了与标准方法相当的验证和基准测试质量。
- **免缓存**：完全避免了离线长度缓存的预计算开销和维护成本。
- **强理论保证**：提供了形式化的同步和终止保证，解决了分布式动态批处理的根本难题。

---

## 2. 核心实验方法和设置

### 数据集
实验在多个公开数据集和一个生产级混合数据集上进行，涵盖了文本和多模态场景，并报告了长度异质性（CV = σ/μ）：
- **UltraChat-200K** (文本，208K 样本，CV=0.48)
- **LLaVA-150K** (多模态，158K 样本，CV=0.29)
- **ShareGPT4o** (多模态，57K 样本，CV=1.00)
- **生产级 MM-Mix** (273K 样本，CV~0.80，包含 OCR、VQA 和字幕任务)

### 实验设置
- **硬件**：主要在单节点 8×H20 GPU 或双节点 16×H20 GPU 上进行。
- **模型**：使用 Qwen3-VL-2B 和 Qwen3-VL-8B 模型，进行全量微调（Full FT）和 LoRA 微调。
- **框架**：DeepSpeed ZeRO-2，bf16 精度，不使用梯度累积（gradient accumulation）。
- **评估指标**：
    - **吞吐量（Throughput）**：`sam/s`（每秒发出的样本数），这是核心指标。
    - **质量**：MMLU（文本）、MMMU-MC（多模态选择似然得分）、验证损失（Val Loss）。
    - **效率**：填充率（Padding Rate）、GPU 计算占比。

### 基线方法对比
- **Standard**：固定批次大小，随机采样。
- **Sorted**：在线按长度分组的固定批次。
- **Packing**：序列打包（HuggingFace 实现，仅限文本）。
- **GMT/BMT-oracle**：基于公平序列（fairseq）风格的全局/桶式最大 token 批处理器，使用**一次性标量长度缓存**进行批构建（但训练时仍执行完整的在线预处理流程）。
- **HFG-oracle**：HuggingFace `group_by_length` 的随机化固定批次版本。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
在所有公共数据集上，ODB 在**无需长度缓存的在线方法中取得了最高的吞吐量**。

| 场景 | 吞吐量提升 (vs. Standard) | 与 Oracle 对比 |
| :--- | :--- | :--- |
| **单节点 Full FT (2B/8B)** | **1.58–2.51×** | 在高异质性数据集（如 ShareGPT4o）上显著优于 GMT/BMT |
| **双节点 Full FT (2B/8B)** | **1.71–3.78×** | 在双节点 ShareGPT4o 上达到 3.78× |
| **生产级 MM-Mix** | **4.43×** | 最大加速比，得益于大量短样本（OCR/VQA）可被高效聚合 |

#### 具体示例（来自 Table 1）
- **ShareGPT4o (8B, Full FT)**: ODB 达到 **5.83 sam/s**，相比 Standard 的 2.37 sam/s，提速 **2.46×**。
- **LLaVA (8B, Full FT)**: ODB 达到 **24.87 sam/s**，相比 Standard 的 14.38 sam/s，提速 **1.73×**。
- **UltraChat (8B, Full FT)**: ODB 达到 **10.23 sam/s**，相比 Standard 的 5.77 sam/s，提速 **1.77×**。

### 消融实验结果
1.  **`Lmax` (每步 token 预算) 的影响 (Table 2)**：
    - 吞吐量随 `Lmax` 增加先上升后下降，存在一个最优值（例如 UltraChat 8B 为 12288）。过大的 `Lmax` 会导致内存压力和步骤时间变长，反而降低吞吐量。

2.  **`D` (待处理深度) 的影响 (Table 3)**：
    - `D` 控制着输入准备与 GPU 计算的重叠程度（pipeline overlap）。
    - 当 `D` 较小时，吞吐量会受限；但当 `D` 超过某个阈值（如 1024-2048）后，继续增加 `D` 带来的收益很小甚至可能下降，因为重叠已饱和。

3.  **缓冲区大小 (Buffer Size) 的影响 (Table 17)**：
    - 更大的缓冲区允许更精细的长度分组，从而减少填充。
    - 在高异质性数据集（如 ShareGPT4o）上，吞吐量在 `buffer=1024` 左右趋于稳定。

---

## 4. 关键结论和发现

### 主要发现
1.  **在线可观测性至关重要**：将批处理决策推迟到长度信息已知的时刻，是实现高吞吐量的关键。这在数据增强策略频繁变更、多模态预处理复杂或样本长度分布长尾（high-CV long tails）的场景下尤其有效。
2.  **ODB 实现了理论与实践的统一**：它成功地在不牺牲 DDP 步骤对齐的前提下，实现了在线动态批处理，并提供了形式化的同步和终止保证（DGAP）。
3.  **高吞吐量与高质量可兼得**：尽管 ODB 显著改变了批次形状（增加每步的有效样本数），但其在 MMLU、MMMU-MC 等基准上的表现与 Standard 方法处于同一水平，证明了其有效性。
4.  **优势源于综合效率提升**：ODB 的加速来自于同时优化了三个效率条件：
    - **空间效率**：极低的填充率（< 2%）。
    - **计算饱和度**：每步处理更多有用 token，充分利用 GPU。
    - **时间效率**：通过控制待处理深度，隐藏了输入延迟。

### 局限性
- **`Lmax` 需要调优**：最佳的 `Lmax` 取决于模型、优化器、精度和注意力堆栈，需要针对具体任务进行搜索。
- **增益具有数据依赖性**：对于长度分布均匀（低 CV）的工作负载，提升有限。
- **未涵盖所有并行策略**：实验范围未包括 ZeRO-3 或 FSDP。
- **非 join 模式的覆盖保证较弱**：虽然保证了总样本数，但不保证每个迭代都覆盖所有身份。

### 未来工作方向
- 探索在更大规模的世界尺寸（world size）或异构节点上的对齐交换机制。
- 将 ODB 与模型侧的打包（packing）或变长注意力（varlen attention）系统结合，以获得进一步的收益。
- 研究如何自动化 `Lmax` 和 `D` 的调优过程。

</details>

---

### 4. [Efficient Neural Network Model Selection for Few-Class Application Datasets](https://arxiv.org/abs/2606.19712)

**Authors**: Bryan Bo Cao, Abhinav Sharma, Lawrence O'Gorman, Michael Coss, Shubham Jain  
**Category**: cs.LG  
**Published**: 2026-06-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.19712v1  

#### Abstract
While much effort has focused on developing and benchmarking high-performance neural networks, less attention has been given to how dataset properties, known to practitioners, can guide efficient model selection. Neural models are typically evaluated on datasets with thousands of classes, yet many r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Efficient Neural Network Model Selection for Few-Class Application Datasets》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前神经网络模型选择通常基于大规模、多类别数据集（如 ImageNet、COCO）上的基准测试，而这些基准对**少类（few-class）应用**（如工业检测、无人机识别、IoT设备等，通常仅含2–10个类别）并不适用。直接将多类模型的性能外推至少类场景会导致**效率低下**，无法选出最轻量且高效的模型。

本文指出，**传统模型选择方法在少类场景下失效**，并提出应从“数据侧属性”（data-side properties）出发，而非依赖通用模型族的默认缩放策略。

---

### 提出了什么新方法或新思路

#### （1）提出“少类独特性”（few-class distinctiveness）
- 发现当类别数 $ N_{cL} < 10 $ 时，分类准确率随类别减少而显著提升，且该趋势**不遵循常规的模型缩放律**（scaling law）。
- 在少类场景中，小模型甚至子模型（sub-models）可达到与大模型相近的精度，但计算量大幅降低。

#### （2）构建基于数据侧属性的分类难度度量（Classification Difficulty Metric）
- 定义一个轻量级、无需训练即可计算的**分类难度指标**，综合考虑：
  - **Number of Classes**（类别数量）
  - **Intra-class Similarity**（类内相似性）
  - **Inter-class Similarity**（类间相似性）
- 提出公式：
  $$
  D_r = \gamma(1.0 - S_R) + (1.0 - \gamma)S_E
  $$
  其中 $ S_R $ 为类内平均余弦相似性，$ S_E $ 为类间平均余弦相似性，$ \gamma \in [0,1] $ 是权重参数。

#### （3）提出“子模型”（sub-models）概念
- 将已有的缩放模型族（如 YOLOv5-nano、EfficientNet-B0）进一步向下缩放（under-scaling），得到更小的“子模型”（如 SY1–SY8）。
- 这些子模型在少类任务上保持高精度的同时，显著减小模型大小和计算开销。

#### （4）建立公开基准：Few-Class Arena
- 提供一个公开的 benchmark（Few-Class Arena），预计算多种组合下的分类难度与实际精度，供开发者快速查询匹配模型。

---

### 相比现有方法的优势

| 方法 | 缺陷 | 本文优势 |
|------|------|---------|
| Off-the-shelf models | 存在特征偏移（feature shift），泛化差 | 针对目标数据定制训练，避免冗余特征 |
| Transfer Learning | 主干网络保留大量无关特征，效率低 | 直接训练最小必要模型，更高效 |
| Scaled Model Families（如 YOLO-nano） | 最小模型仍过大，未针对少类优化 | 可继续缩小至 sub-model，节省高达 72% FLOPs |
| Full Training & Testing | 耗时耗能，难以遍历所有组合 | 使用 difficulty metric 快速预测相对性能，提速 6–29× |

> ✅ **核心优势**：通过数据侧先验知识指导模型选择，实现**高效、低成本、可持续的模型部署**。

---

## 2. 核心实验方法和设置

### 使用的数据集

| 数据集 | 类别数 | 应用背景 |
|--------|-------|----------|
| **ImageNet-1000 / ImageNet-200** | 1000 / 200 | 大规模图像分类基准 |
| **CIFAR-10 / CIFAR-100** | 10 / 100 | 小图像分类 |
| **Food-101** | 101 | 食物识别 |
| **CalTech-101 / CalTech-256** | 101 / 256 | 细粒度物体识别 |
| **COCO minitrain** | 80 → 下采样至 1–80 类 | 目标检测 |
| **Custom Safety Wear Dataset** | 6–7 类 | 工地安全穿戴检测（自建） |

> 所有实验均从完整数据集中随机抽取子集进行测试（如取 2, 3, 4, 6, 8 类），每种子集重复 5–200 次以保证统计意义。

---

### 实验设置和评估指标

#### 模型架构
- **分类器**：MobileNet-V2、ResNet-50、ViT、EfficientNet-B0/B1、VGG-19
- **检测器**：YOLOv5-nano（及其 sub-models SY1–SY8）

#### 训练配置
- 优化器：SGD（lr=0.1, momentum=0.9, weight decay=0.0001）
- 输入分辨率：统一调整为标准尺寸（如 224×224）
- Batch Size：32–64
- Epochs：100–200（视收敛情况）

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Top-1 Accuracy** | 分类任务主指标 |
| **mAP@0.5** | 目标检测平均精度 |
| **GFLOPs / FLOPs** | 模型计算复杂度 |
| **Model Size (MB)** | 参数量大小 |
| **Spearman Correlation** | 难度度量与真实精度的相关性 |
| **Energy Consumption (W)** | 实际平台功耗测量 |

---

### 基线方法对比
- **Baseline 1**: 使用 YOLOv5-nano 或 EfficientNet-B0（当前最小发布模型）
- **Baseline 2**: 使用 full-dataset pretraining + fine-tuning（典型迁移学习）
- **Baseline 3**: 遍历多个模型并逐一训练测试（传统 brute-force 方法）
- **Proposed Method**: 使用 difficulty metric 预筛选 + sub-model 训练

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）分类难度度量的有效性
- 在 CIFAR-100 和 ImageNet-200 上，提出的 $ D_r $ 指标与实际精度的 **Spearman 相关系数最高达 0.9**（优于其他方法 DA–DE）。
- 当 $ \gamma = 0.25 $ 时相关性最强，说明**类内相似性比类间相似性更重要**。

#### （2）子模型效率提升
| 模型 | 类别数 | 准确率下降 | 计算量减少 | 模型大小减少 |
|------|--------|------------|-------------|----------------|
| YOLOv5-nano → SY2 | 10类 | <0.5% | ↓36% GFLOPs | — |
| YOLOv5-nano → SY4 | 1类 | <1% | ↓72% GFLOPs | ↓64% MB |
| EfficientNet-B0 子模型 | 4类 | ~0.5% | ↓42% | ↓42% |

> ⚡️ **最大收益出现在 $ N_{cL} \leq 10 $ 场景**

#### （3）模型选择加速效果
| 方法 | 单次测试耗时（秒/对） | 加速比 |
|------|------------------------|--------|
| VGG-19 测试 | 4.49 | 1× |
| EfficientNet-B0 测试 | 21.82 | 1× |
| Difficulty Metric 计算 | **0.76** | **6–29× 更快** |

> 💡 使用 difficulty metric 可在 **不训练模型的情况下预测相对性能优劣**，极大节省调参时间。

#### （4）嵌入式系统实测效率（移动机器人、无人机、IoT）
| 平台 | 指标 | YOLO-nano | Y-4（sub-model） | 改善 |
|------|------|-----------|---------------|------|
| Drone | Power (W) | 54.2 | 53.9 | -0.6% |
| Drone | Memory (MB) | 6.7 | 2.4 | ↓64% |
| Drone | Inference Time (ms) | 51.2 | 35.2 | ↓31% |
| Robot | Power (W) | 99.7 | 99.3 | -0.4% |

> 🔋 虽然功耗改善有限（因运动能耗主导），但在内存和延迟方面有显著优势，适合边缘部署。

---

### 消融实验结果

#### （1）不同 $ \gamma $ 权重的影响（图4）
- $ \gamma \in [0.0, 0.5] $ 时相关性更高 → 表明**类内相似性更重要**
- 随着 $ N_{cL} $ 增加，各曲线趋近 → 表明“少类特性”在 $ N_{cL} > 10 $ 后消失

#### （2）类内 vs 类间相似性的预测能力（表2）
| $ N_{cL} $ | Intra-class corr. | Inter-class corr. |
|------------|--------------------|-------------------|
| 2 | 0.807 | -0.097 |
| 4 | 0.819 | -0.238 |
| 8 | 0.833 | -0.365 |

> ✅ **类内聚类程度是决定分类难易的关键因素**

#### （3）SimSS 难度指标验证（图12）
- SimSS 与 DCN-Sub（子集精度）的 Pearson 相关系数达 **r = 0.90（CLIP）、0.88（DINOv2）**
- 证明所提指标能有效估计**特定子任务的上限性能**

---

## 4. 关键结论和发现

### 主要发现

1. **存在“少类独特性”现象**：
   - 当类别数 $ < 10 $ 时，准确率随类别减少而急剧上升；
   - 此现象不受传统 scaling law 控制，需专门建模。

2. **数据侧属性主导模型效率**：
   - 类内相似性越高、类间越远、类别越少 → 分类越容易；
   - 利用这些属性可提前判断任务难度，避免盲目训练。

3. **子模型（sub-models）可行且高效**：
   - 可将 YOLO/EfficientNet 等模型进一步缩小；
   - 在少类任务中，精度损失极小，但资源节省巨大（↓72% FLOPs）。

4. **difficulty metric 可替代部分训练过程**：
   - 仅需一次 embedding 提取，即可快速比较数千种组合；
   - 速度比传统方法快 **6–29倍**，适用于资源受限场景。

---

### 方法的局限性

1. **适用范围限于视觉任务**：
   - 当前实验集中在图像分类与检测，未扩展到 NLP 或语音。

2. **依赖预训练 embedding 提取器**：
   - 如 CLIP、DINOv2 等，若其本身存在偏差会影响难度估计。

3. **未深入研究 scale-resolution 属性**：
   - 图像尺度与分辨率的关系尚未量化建模。

4. **部分数据集未公开**：
   - 如工地安全检测数据因商业原因未开源。

5. **结论基于特定模型族**：
   - 主要在 ResNet、YOLO、EfficientNet 上验证，是否普适有待更多架构检验。

---

### 未来工作方向

1. **扩展 Few-Class Arena 至更多领域**：
   - 包括 NLP、音频、医疗影像等应用场景。

2. **自动化 sub-model generation pipeline**：
   - 开发工具链支持一键生成任意缩放级别的子模型。

3. **融合 difficulty metric 到 NAS（Neural Architecture Search）**：
   - 引导搜索空间聚焦于“易分类”结构。

4. **动态 difficulty-aware pruning/inference**：
   - 根据输入样本难度动态调整模型路径或精度。

5. **理论建模 few-class scaling law**：
   - 推导适用于 $ N_{cL} < 10 $ 的新型缩放规律。

---

> 📌 **总结一句话**：  
> 本文揭示了“少类应用”的独特性质，提出利用**数据侧属性构建轻量难度指标**，指导高效模型选择，并通过“子模型”实现极致压缩，在保持精度的同时大幅提升效率，为边缘智能提供了实用的新范式。

</details>

---

### 5. [ENPIRE: Agentic Robot Policy Self-Improvement in the Real World](https://arxiv.org/abs/2606.19980)

**Authors**: Wenli Xiao, Jia Xie, Tonghe Zhang, Haotian Lin, Letian "Max" Fu, Haoru Xue, Jalen Lu, Yi Yang, Cunxi Dai, Zi Wang, Jimmy Wu, Guanzhi Wang, S. Shankar Sastry, Ken Goldberg, Linxi "Jim" Fan, Yuke Zhu, Guanya Shi  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.19980v1  

#### Abstract
Achieving dexterous robotic manipulation in the real world heavily relies on human supervision and algorithm engineering, which becomes a central bottleneck in the pursuit of general physical intelligence. Although emerging coding agents can generate code to automate algorithm search, their successe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ENPIRE: Agentic Robot Policy Self-Improvement in the Real World**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前在真实世界中实现灵巧机器人操作（dexterous robotic manipulation）严重依赖人类监督和算法工程干预，这成为迈向通用物理智能（general physical intelligence）的主要瓶颈。尽管现有的 **coding agents** 在数字环境中展现出自动化算法搜索的能力，但在真实机器人任务中的应用仍受限于缺乏闭环反馈机制。

ENPIRE 正是为了解决这一核心问题而提出：如何让 coding agents 在真实物理世界中自主地进行策略优化（policy improvement），从而减少对人类的依赖，并实现可扩展的“物理自动研究”（physical autoresearch）。

---

### **提出的新方法与新思路**
作者提出了 **ENPIRE** ——一个面向 coding agents 的代理式框架（agentic harness），用于实现在真实世界中的机器人策略自改进。该框架将物理自动研究分解为两个阶段：

#### ✅ **第一阶段：环境构建（Environment Construction, EN）**
- 通过 human-in-the-loop 的方式，由 coding agents 利用 **procedural tool calls** 构建具备以下功能的环境接口：
  - **自动重置（automated reset）**：任务失败或完成后自动恢复初始状态。
  - **自动验证（automated verification）**：基于传感器输入生成二值奖励信号以判断任务成败。
  - **硬安全约束（hard safety constraints）**：防止危险动作，保障长期无人值守运行。
- 这些模块一旦构建完成即作为不可变的 Gym API 被后续流程复用，形成标准化、可重复使用的实验平台。

#### ✅ **第二阶段：策略自改进（Policy Improvement, PIRE）**
- coding agents 完全自主地通过以下流程进行策略优化：
  - 阅读文献 → 提出假设 → 修改训练代码（如 BC、RL 等）→ 执行 rollout → 分析日志 → 改进策略。
- 引入 **Evolution 模块（E）** 实现多 agent 协同进化：
  - 多个 agent 并行测试不同训练配方（training recipes）。
  - 通过 Git 共享成果，自发 cherry-pick 或 merge 成功方案。
  - 形成分布式、去中心化的“思想树”（idea tree）促进知识积累。

#### ✅ 新引入的效率度量指标
为了量化物理资源利用效率，提出两个关键指标：
- **Mean Robot Utilization (MRU)**：机器人处于活跃实验状态的时间占比。
- **Mean Token Utilization (MTU)**：每分钟消耗的平均 token 数量。
- 同时定义 **Token-to-Success Ratio** 衡量 token 使用效率。

---

### **相比现有方法的优势**
| 维度 | 现有方法 | ENPIRE |
|------|--------|-------|
| **迭代媒介** | 仿真环境为主，真实执行仅作最终验证 | 直接在真实硬件上闭环迭代 |
| **反馈来源** | 人工标注、预设奖励函数 | 自动化感知 + 编码 agent 自动生成 reward 函数 |
| **协作模式** | 单 agent 或集中式控制 | 去中心化 multi-agent 团队，基于 Git 协同演化 |
| **资源利用率评估** | 缺乏物理资源维度的基准 | 提出 MRU、MTU 等新指标，支持公平比较 |

> 🔍 **核心突破**：ENPIRE 是首个实现 **完全闭环的真实世界机器人自动研究系统**，使得 coding agents 可以像科学家一样，在真实物理条件下持续自我改进策略。

---

## **2. 核心实验方法和设置**

### **使用的任务与数据集**
ENPIRE 在多个高难度灵巧操作任务上进行了验证：

| 任务 | 描述 |
|------|-----|
| **Push-T** | 使用非抓取式推动将 T 形块推入目标区域（需视觉伺服） |
| **Pin Insertion** | 将插针插入直径仅 4mm 的孔中（高精度装配） |
| **GPU Insertion** | 将 GPU 芯片插入主板插槽（接触力控要求高） |
| **Zip Tie Cutting** | 抓取剪刀并切断扎带尾部（长程任务规划） |

此外还在 **RoboCasa365** 模拟器中进行大规模仿真评估，用于跨方法比较。

---

### **实验设置**
- **硬件平台**：8 台双臂 YAM 机器人组成的 fleet，每台配备独立计算单元（RTX 5090）、摄像头（RealSense D405/D435i）和 FastAPI 控制服务。
- **agent 架构**：
  - 使用三种主流 coding agents：**Codex (GPT-5.5)**、**Claude Code (Opus 4.7)**、**Kimi Code (K2.6)**。
  - 每个 station 配备一个 agent，运行在一个沙盒化的 Git 仓库中。
- **通信机制**：所有 agent 通过 Git 共享分支、提交和合并策略更新，实现去中心化协作。

---

### **评估指标**
| 指标 | 定义 |
|------|-----|
| **Success Rate** | 单次 rollout 中完成任务的概率（允许最多 8 次重试，体现容错能力） |
| **Time to Success** | 达到目标成功率所需的墙钟时间（wall-clock time） |
| **MRU / GPU Utilization** | 机器人/GPU 活跃时间占总研究时间的比例 |
| **MTU (Mean Token Utilization)** | 平均每分钟消耗的 token 数 |
| **Token-to-Success** | 总 token 消耗与成功策略之间的比率 |

---

### **基线方法对比**
- **Human-in-the-loop 方法**：如 [48] 中的人类参与强化学习方法（Residual RL + 数据生成）。
- **Zero-shot coding agents**：无 autoresearch 循环的直接生成策略（如 CaP-X）。
- **End-to-end VLA 模型**：如 GR00T N1.5，在 RoboCasa 上进行端到端推理。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **真实世界任务表现**
| 任务 | 最终成功率 | 收敛时间（vs 基线） |
|------|-----------|------------------|
| **Pin Insertion** | **100%**（连续 50 次成功） | ⏱️ **< 40 分钟**（8-agent fleet），比人类干预方法快数倍 |
| **Push-T** | **~99%** | 从 0 到 1.0 normalized score：<br>• 单 agent：约 5 小时<br>• 8-agent：**仅 2 小时** |
| **Zip Tie Cutting** | 成功实现剪切动作 | 展示了 code-based policy 与 VLA 结合的能力 |

#### ✅ **多 agent 加速效果（Scaling Behavior）**
- **Push-T**：从 1 到 8 个 agent，达到满分时间从 ~5h 缩短至 **2h**（↓60%）
- **Pin Insertion**：从 >1.5h 缩短至 **~40min**（↓73%）
- ➕ 显示出良好的并行加速潜力，得益于分布式假设检验与知识共享。

#### ✅ **RoboCasa365 仿真结果**
| 方法 | 平均成功率 |
|------|----------|
| **GR00T (end-to-end VLA)** | ~60% |
| **CaP-X (zero-shot tool use)** | ~65% |
| **Ours (ENPIRE + autoresearch)** | **~75–80%** ✅ |

> 💡 ENPIRE 在多个任务上显著优于纯 VLA 和零样本工具调用方法。

---

### **消融实验结果**

#### 🧪 **视觉能力影响（C.3）**
| 配置 | 是否成功 | 时间 |
|------|--------|-----|
| **Codex + Native Vision** | ✅ 是 | 最快达成 |
| **Codex + Vision Function Call** | ✅ 是 | 较慢（额外开销） |
| **Codex + No Vision Access** | ✅ 是 | 比 function call 更快？ |

> ❗ 惊人发现：即使没有原生视觉访问，coding agent 也能通过日志信号推断状态；频繁调用视觉函数反而增加延迟。

#### 🧪 **模型与 harness 对比（C.4）**
| 配置 | 时间到成功 |
|------|---------|
| **Codex + GPT-5.5** | 最快 |
| **Claude + Opus 4.7** | 中等 |
| **Codex + Opus 4.7** | 最慢（harness 不匹配导致低效） |

> 表明 agent 性能不仅取决于 LLM 本身，还高度依赖于 **agent harness 的设计质量**。

#### 🧪 **Token 利用分析（C.2）**
- 随着 agent 数量增加，**MTU 呈超线性增长**：
  - 4-agent：接近线性
  - 8-agent：token 消耗急剧上升
- 原因：更多时间花在阅读他人分支、总结、协调而非实际执行。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **ENPIRE 实现了真实世界中 fully autonomous policy improvement**：
   - coding agents 可在无人干预下完成从环境构建到策略优化的全过程。
2. ✅ **multi-agent fleet 显著加速研究进程**：
   - 分布式假设探索 + Git 协同 → 快速收敛至高性能策略。
3. ✅ **code-as-policy + autoresearch 可超越传统 VLA 方法**：
   - 在 RoboCasa 上优于 GR00T 和 CaP-X。
4. ✅ **reward 函数也可由 agent 自动生成**：
   - 如 zip-tie 插入任务中，agent 自主设计双视角几何检测逻辑，延迟 <150ms。
5. ✅ **经验可迁移**：
   - 在 pin insertion 中学到的经验可用于初始化 GPU insertion 任务，加快收敛。

---

### **局限性**
1. ⚠️ **资源未被充分利用**：
   - **MRU 随 fleet size 增大而下降**（Fig. 7a），说明 agent 花太多时间在“思考”而非“行动”。
   - GPU 利用率也不饱和，存在计算空转。
2. ⚠️ **Token 成本呈超线性增长**：
   - 扩展到 8 个 agent 时，token 消耗远高于收益提升，经济性差。
3. ⚠️ **缺乏统一调度机制**：
   - 当前依赖 Git 自发协作，可能出现重复实验或信息过载。
4. ⚠️ **reset 与 verification 仍需一次性人工投入**：
   - 虽然是一次性成本，但限制了系统的零样本泛化能力。

---

### **未来工作方向**
1. 🔮 **优化 agent 资源调度策略**：
   - 引入 centralized coordinator 或 incentive mechanism 提高 MRU 和 MTU 效率。
2. 🔁 **实现真正的 lifelong learning**：
   - 让 agent 在多个任务间持续积累技能库，构建可复用的 robotic tool library。
3. 🤖 **融合 VLA 与 code-based policy 的 hybrid 架构**：
   - 利用 VLA 进行高层语义理解，code-based policy 实现底层精确控制。
4. 🌐 **开放 benchmark 与标准协议**：
   - 推动建立类似 MLE-Bench 的 **Physical Autoresearch Benchmark**，统一评估 MRU、MTU、Token-to-Success 等指标。

---

## ✅ 总结
ENPIRE 开创性地实现了 **真实世界中由 coding agents 主导的机器人策略自改进闭环系统**，展示了 agentic AI 在物理科学研究中的巨大潜力。它不仅是技术框架的创新，更是一种新的科研范式的探索——**让 AI 成为真正的“机器人科学家”**。

> 🏁 **一句话总结**：  
> **ENPIRE 让 coding agents 在真实世界中“动手做科研”，实现了从环境搭建、策略训练到协同进化的全自动闭环，为通向通用物理智能提供了可扩展路径。**

</details>

---

### 6. [Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe](https://arxiv.org/abs/2606.20381)

**Authors**: Qian Zhao, Kunlong Chen, Changxin Tian, Zhonghui Jiang, Haitao Zhang, Chaofan Yu, Peijie Jiang, Mingliang Gong, Jia Liu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.20381v1  

#### Abstract
FP4 training promises substantial reductions in memory and computation cost for LLM pretraining, yet current FP4 hardware paths and recipes, including NVIDIA Blackwell/Rubin-class systems and AMD MI350-series GPUs, remain centered on E2M1 data elements. In this study, we identify a fundamental limit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前主流的 **FP4**（4-bit floating point）训练方案（如 NVIDIA NVFP4、MXFP4）普遍采用 **E2M1** 格式（2 exponent bits, 1 mantissa bit），尽管其具备较宽动态范围以应对 outlier，但在实际 LLM 预训练中仍面临严重的 **训练不稳定性和损失退化**（loss degradation）问题。

本文指出，这一问题的根本原因并非仅来自分布或量化误差，而是源于 **E2M1 非均匀表示格式固有的几何不对称性**，导致系统性的负向舍入偏差 —— **Shrinkage Bias**（收缩偏差）。该偏差在深层网络中 **乘性累积**，并被 **Random Hadamard Transform (RHT)** 进一步放大，从而破坏训练稳定性。

### 提出了什么新方法或新思路

作者提出 **UFP4**（Uniform FP4）—— 一种基于 **均匀量化网格**（uniform grid）的新型 4-bit 训练方案，其核心创新如下：

- **理论洞察**：首次形式化定义并分析了 **Shrinkage Bias** 的几何起源，揭示其源于非均匀格式（如 E2M1）在 Round-to-Nearest-Even (RTNE) 下的 **bin 几何不对称性**。
- **设计原则**：一旦通过 RHT 将张量从“动态范围受限”转变为“局部分辨率受限”，应优先保留局部精度而非极端动态范围，因此 **均匀格式（E1M2/INT4）更优**。
- **新配方 UFP4**：
  - 使用 **E1M2 或 INT4-style 均匀网格** 替代 E2M1；
  - 在所有三个 GEMM 路径（FPROP `fwd_y`、DGRAD `bwd_dx`、WGRAD `bwd_dw`）上应用 **RHT**；
  - 仅在上游梯度 `dY` 上使用 **Stochastic Rounding (SR)**。

### 相比现有方法的优势

| 维度 | E2M1-based 方法（如 NVFP4） | UFP4（本文） |
|------|-------------------------------|-------------|
| **量化格式** | 非均匀（E2M1），存在 Shrinkage Bias | 均匀（E1M2/INT4），无几何偏差 |
| **RHT 应用范围** | 通常仅用于 WGRAD（`bwd_dw`），避免前向路径恶化 | 可安全应用于所有三个 GEMM 路径 |
| **训练稳定性** | RHT 可能加剧量化噪声，尤其在后向传播中 | RHT 显著提升桶利用率且不引入额外偏差 |
| **最终性能** | 存在较大 BF16 损失差距 | 显著缩小 BF16 损失差距 |

> ✅ **核心优势**：UFP4 通过改变底层量化格式，从根本上解决了 E2M1 与 RHT 的不兼容问题，使得 **full-RHT 成为稳定且有益的操作**。

---

## 2. 核心实验方法和设置

### 使用了哪些模型与任务

- **模型架构**：
  - **Dense 1.5B**（稠密模型）
  - **MoE 7.9B**（稀疏专家模型）
  - **MoE 124B**（超大规模稀疏模型）
- **任务**：**长周期语言建模预训练**（long-run pretraining），评估在真实训练流程中的表现。

### 实验设置和评估指标

#### 主要设置
- **量化配置统一控制**：除格式和 RHT 范围外，其余设置保持一致（见 Table 1）：
  - 量化块大小：`1×16`
  - 缩放层级：FP32 单级缩放（single-level scaling）
  - Stochastic Rounding 范围：仅作用于 `dY`
  - 权重是否使用 2D 缩放：否（disabled）

#### 评估指标
- **主指标**：**BF16-relative LM loss error**  
  $$
  \frac{|\mathcal{L}_{\text{FP4}} - \mathcal{L}_{\text{BF16}}|}{\mathcal{L}_{\text{BF16}}}
  $$
  衡量 FP4 训练与 BF16 全精度训练之间的损失差距，越小越好。
- **辅助分析指标**：
  - **SQNR**（Signal-to-Quantization-Noise Ratio）：衡量量化保真度。
  - **Effective Bucket Ratio (Beff)**：基于熵的桶利用率指标，反映 RHT 对 outlier 分散的效果。
  - **Scaling Law Analysis**：拟合不同计算预算下的 loss 曲线，验证性能优势是否随规模扩展而持续。

### 基线方法对比

- **强基线 E2M1 配方**：经过系统调优的 E2M1 配置（通过 controlled ablation search 得出）：
  - 格式：E2M1
  - RHT 仅用于 `bwd_dw`（WGRAD）
  - SR 仅用于 `dY`
  - 最大 FP4 值设为 6（保留完整范围）
- **目标对比**：UFP4（E1M2 + full-RHT + dY-only SR）vs. 上述 E2M1 强基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（图 8）

在多个模型上，UFP4 显著优于 E2M1 基线，**大幅降低 BF16 损失差距**：

| 模型 | E2M1 损失误差 | UFP4 损失误差 | **相对改进** |
|------|----------------|---------------|--------------|
| **Dense 1.5B** | 1.2570% | **0.9673%** | ↓ 23.0% |
| **MoE 7.9B** | 2.3596% | **1.8469%** | ↓ 21.7% |
| **MoE 124B** | 1.7308% | **1.3863%** | ↓ 19.8% |

> 📈 所有模型上，UFP4 均显著更接近 BF16 性能，且差距随训练 token 数增加而稳定维持。

### 与基线方法的对比结果

- **长期训练稳定性更高**：UFP4 在 MoE 124B 上训练超过 800B tokens 后仍未发散，而 E2M1 基线出现明显波动。
- **Scaling Law 验证**（图 9）：
  - 在 10M–324M MoE 模型范围内进行缩放律测试；
  - **E1M2 曲线始终低于 E2M1**，表明其在相同计算预算下能达到更低 loss；
  - 优势在更大模型上依然成立，说明效果具有可扩展性。

### 消融实验结果（Table 2 & 图 10）

#### （1）RHT 范围与 SR 消融（Dense 1.5B）

| 设置 | RHT 路径 | SR on dY | LM Loss | ΔLoss |
|------|--------|----------|--------|-------|
| No RHT | – | × | 1.89202 | 0.00000 |
| Only `bwd_dw` | √ | × | 1.88721 | -0.00481 |
| `fwd_y + bwd_dw` | √ | × | 1.88558 | -0.00644 |
| `bwd_dx + bwd_dw` | √ | × | 1.88912 | -0.00290 |
| **Full RHT + SR (UFP4)** | √ | √ | **1.88079** | **-0.01123** |

✅ **结论**：
- **Full-RHT 是最优选择**，尤其 `fwd_y` 和 `bwd_dx` 上启用 RHT 显著提升性能；
- **SR on dY 提供额外增益**（↓0.00456）；
- 支持核心论点：**均匀格式使 full-RHT 安全且有效**。

#### （2）能否用截断 E2M1 模拟均匀格式？

尝试将 E2M1 的最大值限制为 2.0（仅保留 `{0, 0.5, 1.0, 1.5, 2.0}`，避开高倍率非对称 bin），但仍使用 full-RHT。

❌ **结果失败**（图 10）：
- 所有截断版本（max_fpx=2/3/4）均 **劣于原始 E2M1 基线**；
- 原因：牺牲了动态范围和桶利用率，无法真正模拟均匀格式的优势。

> 🔍 **重要发现**：**必须原生支持 E1M2/INT4 格式**，不能靠限制 E2M1 来近似。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Shrinkage Bias 是 E2M1 的根本缺陷**：
   - 源于 RTNE bin 的几何不对称性，导致系统性负向舍入误差；
   - 该误差在 GEMM 中 **乘性累积**，随层数加深而指数衰减信号。

2. ✅ **RHT 加剧 E2M1 的不稳定性**：
   - RHT 将张量能量分散到中等幅值区域，恰好落入 E2M1 最不对称的 bin 区间；
   - 导致 **SQNR 下降**，反而恶化量化质量。

3. ✅ **均匀格式（E1M2/INT4）是更优选择**：
   - 所有 bin 对称，天然消除 Shrinkage Bias；
   - 更好匹配 RHT 后的分布，将桶利用率提升转化为实际精度增益。

4. ✅ **UFP4 实现稳定高效的 full-RHT 训练**：
   - 在所有三个 GEMM 路径启用 RHT；
   - 仅需在 `dY` 使用 SR；
   - 在 Dense 和 MoE 模型上均显著优于 E2M1 基线。

### 方法的局限性

- **硬件依赖性强**：当前多数加速器（如 NVIDIA Blackwell）原生支持 E2M1/NVFP4，缺乏对 E1M2/INT4 的硬件级支持；
- **未探索混合格式**：虽有工作（如 MixFP4）尝试 per-block 自适应选择 E2M1/E1M2，但本文未将其纳入比较；
- **仅限预训练阶段**：未涉及微调或推理场景的表现。

### 未来工作方向

- **推动硬件支持**：呼吁未来 ML 加速器将 **E1M2/INT4 作为一级训练原语**（first-class training primitives），与 E2M1 并列；
- **融合优化**：进一步优化 **RHT + quantization 的 kernel 融合**，降低延迟（当前 fused overhead ≈ 1.06–1.07×）；
- **扩展至其他低比特格式**：探索 3-bit 或 binary 训练中是否存在类似几何偏差问题；
- **结合其他稳定技术**：将 UFP4 与 FAAR、TetraJet-v2 等自适应舍入或 outlier 控制方法结合，进一步逼近 BF16 性能。

---

> 💡 **最终建议**：  
> “**E2M1 应保留用于原始 outlier-heavy 推理任务，但训练硬件必须原生支持 E1M2/INT4-style 均匀 4-bit 格式。**”

</details>

---

### 7. [ADaPT: Token-Level Decoupling for Efficient Large Reasoning Models](https://arxiv.org/abs/2606.19919)

**Authors**: Tingyun Li, Zishang Jiang, Jinyi Han, Xinyi Wang, Sihang Jiang, Han Xia, Zhaoqian Dai, Shuguang Ma, Fei Yu, Jiaqing Liang, Yanghua Xiao  
**Category**: cs.LG  
**Published**: 2026-06-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.19919v1  

#### Abstract
Large reasoning models rely on long chain-of-thought to achieve strong performance, but applying such reasoning uniformly incurs high computational cost. Existing efficiency-oriented methods attempt to shorten or mix reasoning strategies, yet often degrade reasoning capability. We identify the root ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ADaPT: Token-Level Decoupling for Efficient Large Reasoning Models

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型推理模型（Large Reasoning Models, LRMs）依赖长链式思维（Chain-of-Thought, CoT）来实现强大的推理能力，但这种推理方式在所有输入上统一应用会导致高昂的计算成本。现有的效率优化方法通常通过缩短推理长度或混合推理策略来减少开销，但这往往以牺牲推理能力为代价。

根本问题在于：**效率信号与正确性信号在序列级别耦合（sequence-level coupling）**，导致训练过程中对“长但正确的推理路径”产生隐式惩罚——即模型因推理步骤较长而被判定为低效，即使其答案是正确的。这抑制了深度推理能力的发展。

---

### 提出了什么新方法或新思路
论文提出 **Adaptive Dual-Process Thinking (ADaPT)**，一种基于双过程理论（dual-process theory）的 token-level 解耦框架，核心思想如下：

- **引入 mode-selection token**：使用 `<think>` 和 `<answer>` 两个特殊 token 显式控制“慢速深思”与“快速作答”两种推理模式。
- **Token-level 效率奖励解耦**：将效率相关的奖励仅应用于 mode-selection token（如 `<think>`），而非整个输出序列。这样可以避免因推理长度长而惩罚正确答案。
- **两阶段训练流程**：
  1. **SFT 阶段**：监督微调，让模型学习两种推理格式的行为。
  2. **ADaPT-GRPO 阶段**：基于 Group Relative Policy Optimization 的强化学习，优化推理模式选择策略，其中效率奖励仅作用于第一个 token。

---

### 相比现有方法的优势
| 维度 | 现有方法（如 TLMRE, ARM, R-4B） | ADaPT |
|------|-------------------------------|--------|
| **效率-性能权衡机制** | 序列级耦合，存在训练冲突 | Token-level 解耦，消除冲突 |
| **是否惩罚长而正确的推理** | 是 | 否 |
| **推理深度控制粒度** | 离散切换或固定压缩 | 连续可调（通过调整 `<think>` 生成概率阈值） |
| **单一模型灵活性** | 固定行为 | 单一模型可在 Pareto 前沿连续移动 |

> ✅ **优势总结**：ADaPT 在保持强大推理能力的同时显著降低推理成本，并支持推理效率与性能之间的**精细、连续控制**。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **SFT 阶段**：
  - 来自 `arm-team` 数据集的子集，包含 long CoT 和 short CoT 示例。
- **RL 阶段（ADaPT-GRPO）**：
  - 构建了 8.5k 可验证的 QA 对，来源包括：
    - **CSQA**（commonsense reasoning）
    - **GSM8K**（小学数学题）
    - **MATH**（高等数学问题）

---

### 实验设置和评估指标
- **模型基础**：
  - 主要使用 **Qwen2.5-7B-Base** 和 **Qwen2.5-3B-Base**。
  - 补充实验验证了在 **LLaMA3-8B** 和 **Qwen2.5-14B** 上的有效性和可扩展性。
- **训练细节**：
  - 使用 Verl 框架进行 GRPO 训练。
  - Batch size: 128，Mini-batch: 64。
  - 最大 prompt 长度：2048 tokens；最大响应长度：8192 tokens。
- **评估指标**：
  - **Accuracy (ACC)**：pass@1 或 avg@32（AIME24 使用 avg@32, temp=0.6）。
  - **推理效率**：平均生成长度（Length ↓）。
  - 分类任务按难度分为：
    - **Easy Tasks**：CSQA, GSM8K, ARC
    - **Hard Tasks**：MATH500, MMLU-Pro, Olympiad, AIME24

---

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Base** | 未经微调的基础模型 |
| **SFT** | 仅经过第一阶段监督微调 |
| **SFT+GRPO** | 完整 RL 微调，无效率约束 |
| **TLMRE** | 引入长度惩罚的 RL 方法 |
| **ARM** | 基于 GRPO 的四模式路由方法 |
| **R-4B** | 使用 bi-mode annealing 教授高效思考 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### Qwen2.5-7B 结果概览：
| 方法 | AVG ACC ↑ | Avg Length ↓ |
|------|-----------|-------------|
| SFT+GRPO | 63.3 | 1540 |
| TLMRE | 61.0 | 1299 |
| ARM | 61.7 | 1131 |
| R-4B | 60.9 | 1044 |
| **ADaPT** | **62.7** | **1031** |

✅ **结论**：ADaPT 在比 SFT+GRPO 少约 **500 tokens** 的情况下，准确率仅下降 0.6%，远优于其他效率方法（它们损失更大精度）。

#### Qwen2.5-3B 结果概览：
| 方法 | AVG ACC ↑ | Avg Length ↓ |
|------|-----------|-------------|
| SFT+GRPO | 53.1 | 1539 |
| TLMRE | 51.1 | 1279 |
| ARM | 51.0 | 1101 |
| R-4B | 49.4 | 1014 |
| **ADaPT** | **51.9** | **1013** |

同样表现最优，在大幅减长的同时几乎不损精度。

---

### 与基线方法的对比结果
- **相比 SFT+GRPO**：
  - 推理长度减少 **~33%–40%**（从 ~1.5k → ~1k tokens）。
  - 准确率仅轻微下降（<1%），说明有效去除了冗余推理。
- **相比效率导向方法（TLMRE/ARM/R-4B）**：
  - 在相同或更低 token 开销下，**准确率高出 1–3 个百分点**。
  - 特别是在 Hard Tasks（如 MATH, Olympiad）上优势更明显。
- **ADaPTthink vs ADaPTanswer**：
  - ADaPTthink（强制使用 `<think>`）性能接近甚至略超 SFT+GRPO，证明其保留了强推理能力。
  - ADaPTanswer（只用 `<answer>`）在简单任务上仍具竞争力，体现 fast reasoning 的可靠性。

---

### 消融实验结果（Ablation Study）

#### γ 参数的影响（Figure 5）
- γ 控制 fast reasoning 的质量容忍度：
  - γ 越高 ⇒ 快速推理需更可靠才能被采用 ⇒ 更频繁触发 `<think>`。
- 实验显示：随着 γ 从 0 增加到 1，`<think>` 使用比例单调上升。
- 小模型（3B）对 γ 更敏感，因其 fast reasoning 可靠性较低，需更多 fallback 到 slow reasoning。

> 🔍 **发现**：γ 提供了一种平滑调节推理深度的手段，验证了 ADaPT 的可控性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **效率与性能的 trade-off 源于训练信号耦合**：
   - 序列级效率奖励会系统性压制“长而正确”的推理路径，损害深层推理能力。
2. **Token-level 解耦可打破这一困境**：
   - 将效率奖励集中于 mode-selection token（如 `<think>`），可避免误伤高质量长推理。
3. **ADaPT 实现高效且不失准的推理**：
   - 显著降低推理长度（最高减少 87%），同时保持接近最优的准确率。
4. **支持连续的推理效率控制**：
   - 通过调节 `<think>` 的生成阈值，单个模型即可在 **efficiency-performance Pareto frontier** 上连续移动（见 Figure 3）。
5. **方法具有良好的泛化性和可扩展性**：
   - 在不同 backbone（Qwen, LLaMA）、不同规模（3B, 7B, 14B）上均有效（Tables 2 & 3）。

---

### 方法的局限性
1. **任务覆盖有限**：
   - 当前评估集中在数学与常识推理任务，未涵盖长上下文、交互式或多轮推理场景。
2. **二元推理模式设计**：
   - 当前仅区分 fast/slow 两种模式，未能捕捉更细粒度的推理行为变化（如中等长度推理）。
3. **模型规模限制**：
   - 实验最大做到 14B，尚未验证在更大模型（如 70B+）上的表现。
4. **依赖人工标注的难度标签**：
   - SFT 阶段需要根据模型表现划分难易题，可能引入偏差。

---

### 未来工作方向
- 扩展至 **多模态和交互式推理任务**。
- 设计 **多级或连续推理模式**，超越 binary switching。
- 探索 **完全自监督的难度识别机制**，减少人工干预。
- 研究在 **更大模型和真实应用场景** 中的部署效果。
- 结合 **early stopping** 或 **drafting** 技术进一步提升效率。

---

> 📌 **总体评价**：ADaPT 提出了一种新颖且有效的视角——将效率优化从“压缩推理内容”转向“智能决策何时启用何种推理模式”，并通过 token-level reward 解耦实现了性能与效率的双赢，是高效推理模型领域的重要进展。

</details>

---

### 8. [PaAno+: Multiscale Encoding and Cross-Variable Attention for Time Series Anomaly Detection](https://arxiv.org/abs/2606.20055)

**Authors**: Youji Zhu, Hongbing Wang, Wenchao Liu, Xiaodong Liu, Xiangguang Xiong  
**Category**: cs.LG  
**Published**: 2026-06-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.20055v1  

#### Abstract
Time-series anomaly detection has significant practical value for industrial and medical monitoring, as well as other critical domains. Current Transformer- and large-model-based detection approaches incur excessive computational overhead, while existing lightweight alternatives are constrained by i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PaAno+: Multiscale Encoding and Cross-Variable Attention for Time Series Anomaly Detection**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前时间序列异常检测方法存在以下缺陷：
- **基于 Transformer 和大模型的方法**（如 AnomalyTransformer、MOMENT）虽然性能强，但计算开销巨大，难以部署在边缘设备上。
- **轻量级方法**（如原始 PaAno）受限于单一感受野的卷积核，无法有效捕捉多尺度时序特征；且独立编码各变量通道，忽略了变量间的相关性。
- 预训练任务监督信号不足，仅依赖局部连续性约束，难以建模复杂的时序动态。

### **提出的新方法与创新思路**
本文提出了 **PaAno+**，一种轻量、高效的异常检测框架，其核心创新包括：

- **多尺度时间编码器（Multiscale Temporal Encoder）**  
  采用并行多分支卷积结构（kernel sizes: 3, 7, 15），分别捕获短程、中程和长程时序模式，并通过跨尺度自适应注意力聚合机制融合多粒度特征，结合残差连接稳定训练。

- **跨变量融合注意力模块（Cross-Variable Fusion Attention）**  
  在变量维度引入 multi-head self-attention，显式建模多变量之间的依赖关系，增强对交互型异常的识别能力。

- **基于窗口重排的预训练任务（Temporal Patch-Window Sorting）**  
  设计了一种新的自监督预任务：将连续的时间片段随机打乱顺序，要求模型恢复原始时序。该任务提供更强的全局时序上下文监督信号，提升特征判别力。

- **紧凑的记忆库设计（Compact Memory Bank）**  
  利用 K-means 聚类压缩记忆库，显著降低存储与推理开销，同时保持高精度。

### **相比现有方法的优势**
- **高性能 + 轻量化**：参数量远小于主流 Transformer 模型（仅 1.1M / 1.5M），却实现 SOTA 性能。
- **多尺度感知能力强**：可同时处理点突变异常与持续偏移异常。
- **显式建模变量间耦合**：适用于复杂工业场景中的多变量协同异常。
- **鲁棒性强**：对超参数不敏感，适合实际部署。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **TSB-AD benchmark**：专为解决传统异常检测基准缺陷而构建，分为两个子集：
  - **TSB-AD-U**：单变量时间序列，共 350 条测试序列，平均长度 ~51,886，异常比例 4.5%。
  - **TSB-AD-M**：多变量时间序列，共 180 条测试序列，平均长度 ~108,826，异常比例 5.0%。
- 数据集已排除标注偏差、位置偏置等问题，确保评估公平性。

### **实验设置**
- **滑动窗口大小（patch size）**：统一设为 96。
- **编码器结构**：
  - 多分支卷积核：3, 7, 15
  - 输出嵌入维度：256
  - 投影头（projection head）：两层 MLP
- **训练策略**：
  - 使用 **triplet loss** 和 **temporal ranking loss** 联合优化
  - 排序损失权重从 1 线性衰减至 0（前 20 个 epoch）
  - 优化器：Adam，学习率余弦退火（1e-3 → 1e-4）
  - 批大小：512
- **推理阶段**：
  - 构建压缩记忆库（Top 10% 聚类中心）
  - 对每个时间戳的所有覆盖 patch 取 k=3 最近邻的平均距离作为异常分数

### **评估指标**
采用六项无阈值依赖的综合指标，避免评估偏差：

| 类型 | 指标 | 说明 |
|------|------|------|
| 区间级 | **VUS-PR**, **VUS-ROC** | 综合考虑不同阈值与容忍偏移下的 PR/ROC 曲线下面积 |
| 区间级 | **Range-F1** | 基于预测与真实异常区间的重叠度计算的最大 F1 |
| 点级 | **AUC-PR**, **AUC-ROC** | 点级别 PR 与 ROC 曲线下面积 |
| 点级 | **Point-F1** | 不同分类阈值下最优的点级 F1 分数 |

> ✅ **VUS-PR 被选为主指标**

### **基线方法对比**
涵盖三大类共 20+ 种代表性方法：
- **Stat & ML**：SAND, DLinear, NLinear
- **Neural Network (NN)**：TimesNet, FITS, DADA, KAN-AD, CrossAD
- **Transformer/Foundation Models**：AnomalyTransformer, DCdetector, PatchTST, iTransformer, MOMENT, TimesFM, LagLlama

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **单变量任务（TSB-AD-U）**
| 方法 | VUS-PR ↑ | VUS-ROC ↑ | Range-F1 ↑ | Point-F1 ↑ | #Params ↓ |
|------|----------|-----------|------------|-------------|-----------|
| **PaAno+ (Ours)** | **0.55** | **0.90** | **0.52** | **0.54** | 1.1M |
| PaAno [6] | 0.51 | 0.88 | 0.48 | 0.51 | 0.3M |
| CrossAD | 0.45 | 0.84 | 0.41 | 0.47 | 0.9M |
| KAN-AD | 0.43 | 0.82 | 0.43 | 0.44 | <0.1M |
| AnomalyTransformer | 0.12 | 0.56 | 0.14 | 0.12 | 4.8M |

> 📌 PaAno+ 在所有指标上均取得 **SOTA 表现**，尤其在 VUS-PR 上领先第二名 CrossAD 达 **10 个百分点**。

#### **多变量任务（TSB-AD-M）**
| 方法 | VUS-PR ↑ | VUS-ROC ↑ | AUC-ROC ↑ | Point-F1 ↑ | #Params ↓ |
|------|----------|-----------|-----------|-------------|-----------|
| **PaAno+ (Ours)** | **0.42** | **0.80** | **0.77** | **0.43** | 1.5M |
| PaAno [6] | 0.43 | 0.79 | 0.76 | 0.43 | 0.3M |
| KAN-AD | 0.41 | 0.75 | 0.73 | 0.42 | <0.1M |
| CATCH | 0.30 | 0.73 | 0.67 | 0.30 | 210.8M |
| AnomalyTransformer | 0.12 | 0.57 | 0.52 | 0.12 | 4.8M |

> 📌 尽管 PaAno 参数更少，PaAno+ 仍实现了相当甚至略优的整体性能，且显著优于其他大模型。

### **与基线方法的对比结果**
- **超越轻量级模型**：PaAno+ 明显优于 KAN-AD、CrossAD 等先进轻量模型。
- **碾压大模型**：尽管参数仅为 MOMENT (~100M+) 或 TimesFM (~200M+) 的 **~1%**，PaAno+ 性能全面反超。
- **效率优势明显**：推理时间控制在 10–25 秒内，而大模型常需数百秒。

### **消融实验结果（Ablation Study）**

| 消融变体 | TSB-AD-U (VUS-PR) | TSB-AD-M (VUS-PR) | 结论 |
|---------|-------------------|-------------------|------|
| 完整 PaAno+ | **54.6%** | **42.3%** | — |
| w/o Multiscale | 54.2% | 39.8% (-2.5%) | 多尺度结构对多变量任务至关重要 |
| w/o dual-scale | 54.0% | 41.0% (-1.3%) | 三尺度 > 二尺度，感受野多样性重要 |
| w/o Attention | — | 41.4% (-0.9%) | 跨变量注意力提升耦合异常识别 |
| w/o pretext_loss | 52.8% (-1.8%) | 39.1% (-3.2%) | 窗口排序任务提供更强监督信号 |

> 🔍 发现：**多尺度编码** 和 **排序预任务** 是性能提升的关键驱动力，尤其在多变量场景中作用更为显著。

---

## **4. 关键结论和发现**

### **主要发现**
1. **轻量不代表低性能**：PaAno+ 证明了通过合理的架构设计（多尺度 + 注意力 + 自监督），可以在极低参数量下达到甚至超越大模型的检测精度。
2. **多尺度特征提取至关重要**：固定感受野限制了模型对不同类型异常（突发 vs 持续）的适应能力，多分支卷积有效缓解此问题。
3. **变量间依赖建模不可忽视**：跨变量注意力模块显著提升了对多变量协同异常的识别能力。
4. **高质量自监督任务是关键**：相比简单的相邻 patch 分类，窗口重排序任务提供了更强的全局时序一致性监督，极大增强了特征空间判别性。
5. **高度鲁棒且易于部署**：模型对 patch size、k 值、memory bank 压缩比等超参数不敏感，适合工业落地。

### **方法的局限性**
- 当前方法仍基于 **semi-supervised setting**（仅用正常数据训练），未探索零样本或弱监督场景。
- 记忆库存储机制虽经压缩，但在极端长序列下仍有扩展挑战。
- 多尺度卷积结构固定，缺乏动态调整感受野的能力。

### **未来工作方向**
- 进一步优化网络结构，探索 **adaptive receptive field** 调整机制。
- 构建通用的 **contrastive learning framework**，支持跨领域迁移。
- 在更多真实工业场景（如电力、制造、医疗）中验证泛化能力。
- 探索在线更新机制以应对非平稳时间序列分布漂移。

---

> ✅ **总结一句话**：  
> **PaAno+ 通过“多尺度编码 + 跨变量注意力 + 窗口排序预训练”三重创新，在极低计算成本下实现了 SOTA 异常检测性能，为资源受限环境下的实时监控提供了高效可靠的解决方案。**

</details>

---

### 9. [Characterizing Narrative Content in Web-scale LLM Pretraining Data](https://arxiv.org/abs/2606.19468)

**Authors**: Teagan Johnson, Elliott Ash, Andrew Piper, Maria Antoniak  
**Category**: cs.CL  
**Published**: 2026-06-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.19468v1  

#### Abstract
The narrative composition of web-scale LLM pretraining corpora remains largely unexplored even though narrative is a fundamental mode of human communication. We present the first fine-grained study of narrative features in Dolma, a 3-trillion-token open pretraining corpus. Drawing on narrative theor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Characterizing Narrative Content in Web-scale LLM Pretraining Data**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前对大规模语言模型（LLM）预训练数据的研究多集中于**质量、毒性、去重和主题分布**等维度，而对其中**叙事内容（narrative content）的构成**缺乏系统性探索。然而，叙事是人类交流的核心形式，也是LLM在故事生成等任务中的关键能力来源。本文首次系统地研究了预训练语料中叙事结构的分布情况，填补了这一空白。

### **提出了什么新方法或新思路**
1. **提出细粒度的叙事分析框架（Narrative Annotation Framework）**  
   基于叙事理论（narrative theory），将叙事分解为三个核心元素：
   - **Agency（主体性）**：角色视角、情感、认知、状态变化、冲突
   - **Setting（场景）**：具体性、时间锚定、空间锚定、感官细节
   - **Events（事件）**：时序关系、因果密度、事件密度
   共定义了 **11个可解释的叙事维度**，并采用5点Likert量表进行标注。

2. **构建 NARRABERT 模型**  
   基于 RoBERTA 架构，通过知识蒸馏（knowledge distillation）从 LLM 标签中学习，训练出一个高效的小模型，用于自动预测上述11个叙事维度。

3. **发布 NARRADOLMA 数据集**  
   在 DOLMA（3万亿token开源预训练语料）上应用 NARRABERT，标注了约 **300万段文本**，形成首个大规模、细粒度的叙事特征数据集。

### **相比现有方法的优势**
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **叙事检测** | 二分类（是否为叙事） | 多维连续谱系（narrativity as a spectrum） |
| **标注粒度** | 粗粒度（如 STORYSEEKER、NARRADETECT） | 细粒度（11维度+事件关系） |
| **可扩展性** | 依赖人工或大模型标注，成本高 | 提出 NARRABERT，实现低成本大规模自动化标注 |
| **应用场景** | 仅用于检测 | 可用于分析叙事分布、指导数据混合（data mixing）、理解模型叙事能力来源 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **主语料**：**DOLMA**（Soldaini et al., 2024），一个包含12个子语料库、总计超3万亿token的开放预训练语料。
- **采样子集**：从 DOLMA 中抽取约 **1700万段三句长度的文本片段**，最终构建两个数据集：
  - **Gold Dataset**：400段人工标注文本，用于验证。
  - **NARRADOLMA**：约 **300万段** 自动标注文本，覆盖约78.5万个独立网页文档。

### **实验设置和评估指标**
#### **标注流程**
1. **初始提取**：使用 NLTK 分句，提取三句片段。
2. **叙事评分**：使用 DeBERTa 二分类模型对片段打分（p ∈ [0,1]），筛选高叙事性文本。
3. **主题分类**：对 Common Crawl 文本使用 WEBORGANIZER 进行24类主题标注。
4. **分层抽样**：保持原始语料比例，平衡主题和叙事得分。

#### **模型训练与验证**
- **NARRABERT 架构**：
  - 共享 RoBERTA-BASE 编码器 + 9个回归头（用于 Agency & Setting）
  - 独立编码器 + 二分类头（用于 Event Relations）
- **训练方式**：知识蒸馏，使用 GEMMA-31B 生成的标签作为监督信号。
- **评估指标**：
  - **Agency & Setting**：MAE（平均绝对误差）、Krippendorff’s α
  - **Event Relations**：F1、Cohen’s Kappa
  - **下游分析**：PCA、UMAP、逻辑回归分类测试

#### **基线方法对比**
- **LLM 基线**：CLAUDE SONNET 4.6、QWEN3-235B-A22B、GEMMA-31B
- **人工标注**：由作者主导，辅以第二位作者和外部标注者进行一致性检验

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型/标注者 | 任务 | MAE | α / K | F1 |
|------------|------|-----|-------|----|
| **人类标注者** | Agency & Setting | 0.62 | 0.76 | — |
| | Event Relations | — | 0.68 | 0.91 |
| **GEMMA-31B** | Agency & Setting | 0.53 | 0.71 | — |
| | Event Relations | — | 0.56 | 0.78 |
| **NARRABERT** | Agency & Setting | 0.57 | 0.66 | — |
| | Event Relations | — | — | 0.63 (temporal), 0.68 (causal) |

> ✅ NARRABERT 在 Agency 和 Setting 上表现接近 LLM，具备实用价值；但在 Event Relations 上有所下降，表明该任务更具挑战性。

### **与基线方法的对比结果**
- **NARRABERT 显著优于直接使用 LLM 进行全量标注的成本效益**：
  - LLM 推理成本高，难以扩展到数百万样本；
  - NARRABERT 可在单GPU上快速完成全量标注。
- **相比二分类叙事检测模型**，本文框架能揭示更丰富的叙事结构差异。

### **消融实验与分析结果**
- **PCA 分析揭示三大叙事主成分**（共解释 ~72% 方差）：
  1. **PC1: Interiority（内在性）**：聚焦于 Focalization、Emotion、Cognition
  2. **PC2: Grounded Eventfulness（具身化事件性）**：Change of State、Conflict、Event Density、Spatial/Temporal Grounding
  3. **PC3: Storyworld Texture（世界质感）**：Concreteness、Sensory Detail、Spatial Grounding

- **分类实验验证叙事特征的有效性**：
  - 使用12维叙事特征预测文档类别（28类），**GBT 模型 Macro-F1 达 0.32**（随机基线为0.03）
  - 预测“是否为叙事”二分类任务，**F1 达 0.76**，说明叙事特征具有强判别力。

---

## **4. 关键结论和发现**

### **主要发现**
1. **叙事结构是可测量且连续的多维属性**  
   叙事不是“有或无”的二元属性，而是存在于一个多维连续空间中，不同文本在不同维度上表现出差异化强度。

2. **叙事质量在预训练源之间分布极不均衡**
   - **Reddit 和 Gutenberg**：高 **Interiority**（情感、内心活动丰富）
   - **Crime & Law、Wikipedia**：高 **Grounded Eventfulness**（事件性强、时空锚定明确）
   - **Food & Dining、Fashion & Beauty**：高 **Storyworld Texture**（感官细节丰富）
   - **没有单一来源在所有维度上都占优**，说明简单“增加叙事数据权重”无法全面提升叙事能力。

3. **类别内叙事多样性远高于预期**
   - 平均类别内标准差为 0.87（总标准差为1.0），说明即使在同一主题或来源内部，叙事风格也高度异质。
   - 因此，**基于 source-level 或 topic-level 的数据混合策略过于粗糙**，无法精细控制叙事暴露。

4. **Upweighting “high-narrative” sources does not uniformly increase all narrative qualities**  
   例如，提升 Reddit 数据比例会增强情感表达，但不会改善场景描写或事件因果链。

### **方法的局限性**
1. **NARRADOLMA 是分层抽样子集**，并非完整 DOLMA 的无偏代表，且**有意过采样叙事性文本**，因此绝对频率不可外推。
2. **人工标注仅400条**，相对于网络文本的多样性仍显不足，部分维度（如 Temporal Ordering）一致性较低（K=0.60）。
3. **NARRABERT 在事件关系预测上表现弱于教师模型**（F1下降约15%），可能影响相关分析的信噪比。
4. **仅限英文文本**，未涵盖其他语言或文化传统的叙事模式。
5. **未建立因果联系**：尚未证明特定叙事组成如何直接影响下游模型的叙事生成能力。

### **未来工作方向**
1. **受控数据混合实验**（Controlled Data Mixing）  
   结合 RegMix 等框架，设计不同叙事组成的训练数据，观察其对模型叙事能力的影响。
   
2. **中间检查点分析**（Intermediate Checkpoint Analysis）  
   跟踪模型在预训练过程中何时、如何习得叙事能力。

3. **跨语言与跨文化叙事建模**  
   扩展框架至非英语语料，探索不同叙事传统的结构差异。

4. **安全与伦理导向的应用**  
   利用 NARRADOLMA 识别潜在有害叙事模式（如煽动性、暴力叙述），支持更安全的数据过滤。

5. **将叙事结构纳入数据治理框架**  
   建议将“叙事多样性”作为预训练数据评估的新维度，与 quality、toxicity、topic distribution 并列。

---

> 📌 **总结一句话**：  
> 本文首次实现了对 LLM 预训练数据中叙事内容的**系统性、细粒度刻画**，提出 NARRABERT 与 NARRADOLMA，揭示叙事是**多维、不均衡、内部异质**的结构，呼吁将“叙事构成”作为数据设计的重要考量维度。

</details>

---

### 10. [Adaptive Distance-Aware Trunk Deep Operator Learning for Long-Span Roadway Bridges](https://arxiv.org/abs/2606.20015)

**Authors**: Bilal Ahmed, Diab W. Abueidda, Waleed El-Sekelly, Tarek Abdoun, Mostafa E. Mobasher  
**Category**: cs.LG  
**Published**: 2026-06-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.20015v1  

#### Abstract
Long-span roadway bridges exhibit highly localized structural responses under vehicular loading, making repeated FE analysis computationally expensive for applications such as influence surface generation and structural digital twins. Existing SciML approaches struggle to accurately capture these lo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Adaptive Distance-Aware Trunk Deep Operator Learning for Long-Span Roadway Bridges

---

## 1. 论文的主要贡献和创新点

### 解决的问题
长跨径公路桥梁在车辆荷载作用下表现出**高度局部化的结构响应**，即变形和应力集中在荷载作用点附近的小区域（“影响区”），而大部分结构域响应接近于零。这种特性导致传统有限元分析（FEM）在进行影响面生成、数字孪生等需要重复模拟的应用时计算成本极高。

此外，现有的科学机器学习方法（如标准 DeepONet）难以准确捕捉此类稀疏且空间不均衡的响应，因为它们通常在整个结构域上训练，导致模型被大量近零响应主导，无法有效学习高梯度局部行为。

---

### 提出的新方法与创新思路
本文提出了一种**自适应距离感知主干 Deep Operator Network（AD-DeepONet）**，其核心创新包括：

#### （1）**自适应 Schur 域构建（Adaptive Schur Domain Construction）**
- 引入基于 **KNN（K-Nearest Neighbors）** 的动态领域选择机制。
- 对每个加载情况，仅选取靠近荷载位置的 $ K $ 个最近节点作为学习域（称为“adaptive Schur domain”），使网络聚焦于实际发生显著响应的影响区。
- 领域随荷载移动而动态变化，解决了固定域或全域学习的低效问题。

#### （2）**距离感知特征表示（Distance-Aware Feature Representation）**
- 在 trunk network 输入中引入几何关系增强特征：
  - $ \mathbf{x}_k $：节点全局坐标
  - $ \mathbf{x}_k - \mathbf{x}_f $：相对于荷载的位置向量
  - $ \frac{\mathbf{x}_k - \mathbf{x}_f}{\|\mathbf{x}_k - \mathbf{x}_f\|} $：单位方向向量
  - $ \|\mathbf{x}_k - \mathbf{x}_f\| $：到荷载的距离标量
- 这些特征显式编码了**荷载-结构相互作用的几何信息**，提升了模型对局部响应模式的学习能力。

#### （3）**物理引导的全场重建（Physics-Based Full-Field Reconstruction）**
- 利用 **Schur complement 公式**，将 AD-DeepONet 在自适应域上的预测扩展至整个结构域。
- 结合数据驱动预测与基于刚度矩阵的物理平衡方程，确保全场解满足力学一致性。

#### （4）**可扩展的数据生成策略**
- 采用**等效壳模型（equivalent shell model）** 替代详细的三维实体模型进行 FEM 数据生成，大幅降低计算开销。
- 荷载以**单轮形式建模**，多轴车辆响应通过线性叠加实现，提升泛化性和效率。

---

### 相比现有方法的优势
| 方法 | 缺陷 | AD-DeepONet 改进 |
|------|------|------------------|
| **Vanilla DeepONet（全域学习）** | 输出维度大、训练慢；损失被近零区域主导，精度差 | 仅在活跃区域学习，避免数据不平衡 |
| **Fixed-Schur DeepONet [34]** | 学习域固定，无法跟随移动荷载，预测质量不稳定 | 自适应域动态跟踪荷载，保持高保真度 |
| **传统 FEM 影响面分析** | 千次级仿真耗时数十小时 | 可在分钟内完成全场影响面生成 |

> ✅ **优势总结**：更高精度 + 更快推理速度 + 更强的空间鲁棒性 + 物理一致性保证

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **合成基准桥模型（Benchmark Bridge）**
   - 多跨混凝土桥，总长 105 m，宽度 10 m
   - 包含 14,157 个节点，约 84,942 个自由度
   - 加载方式：随机分布的单轮荷载（1–250 kN）
   - 数据量：16,000 组静态 FEM 模拟

2. **真实世界案例：Mussafah Bridge**
   - 位于阿布扎比的预应力混凝土梁桥系统
   - 含多个跨度、膨胀缝和复杂边界条件
   - 等效壳模型含约 15,050 节点，90,300 自由度
   - 数据量：40,000 组加载场景

所有数据均通过 Abaqus 中的 reduced-order shell 模型生成，兼顾精度与效率。

---

### 实验设置与评估指标

#### 模型配置
- **Branch Network**：编码荷载大小 $ F $ 和位置 $ (\mathbf{x}_f, z_f) $
- **Trunk Network**：处理自适应节点及其距离感知特征
- **输出**：每个节点的六自由度响应（$ U_x, U_y, U_z, R_x, R_y, R_z $）
- **优化器**：Adam ($ lr = 5\times10^{-4} $)，batch size = 32
- **硬件平台**：NYUAD HPC 集群（Nvidia A100 GPU）

#### 评估指标
- **平均相对误差（Mean Relative Error）**：
  $$
  \text{Error} = \frac{1}{N}\sum_{i=1}^N \frac{\|u_{\text{pred}}^{(i)} - u_{\text{FEM}}^{(i)}\|}{\|u_{\text{FEM}}^{(i)}\|}
  $$
- **训练时间**、**单次推理时间**
- **影响面生成总耗时**

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Method A**: Vanilla DeepONet | 全域学习，无域缩减 |
| **Method B**: Fixed-Schur DeepONet [34] | 固定子域（如跨中区域），结合 Schur 补全场重建 |
| **Method C**: **AD-DeepONet（本文提出）** | 自适应 KNN 域 + 距离感知特征 + Schur 重建 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Mussafah Bridge 上的结果）

| 指标 | AD-DeepONet (Method C) | Vanilla DeepONet (A) | Fixed-Schur (B) |
|------|------------------------|-----------------------|------------------|
| **全场平均相对误差** | **< 5%**（多数位移分量 < 2%） | 10% ~ 65% | ~8% ~ 15%（波动大） |
| **单次前向推理时间（不含重建）** | **~0.001 秒**（比 FEM 快 4 个数量级） | ~0.1 秒 | ~0.01 秒 |
| **完整响应评估时间（含 Schur 重建）** | **~0.4 秒** | ~20 秒 | ~18 秒 |
| **训练时间** | 显著低于 Method A | 最高 | 居中 |
| **影响面生成时间** | **0.37 小时** | —— | —— |
| **传统 FEM 影响面生成时间** | **28.6 小时** | —— | —— |

> ⚡️ **加速比**：总体响应评估提速约 **60×**；纯推理阶段达 **10⁴×**

---

### 与基线方法的对比结果
- **Method A（全域 DeepONet）**
  - 错误最高（尤其旋转自由度 Rx/Ry/Rz 达 65%）
  - 训练最慢，因输出维度巨大
- **Method B（Fixed-Schur）**
  - 在荷载位于预设域内时表现尚可，但边缘或非对称加载时误差剧增
  - 空间鲁棒性差，依赖人工设定区域
- **Method C（AD-DeepONet）**
  - 所有位置均保持稳定低误差（<5%）
  - 自适应机制使其适用于任意加载路径
  - 距离感知特征显著提升 trunk 表征能力

---

### 消融实验结果（见 Appendix A.2）
研究了 KNN 中邻居数 $ K $ 的影响（从 10 到 100）：

| $ K $ | 平均误差 | 训练时间 |
|-------|----------|-----------|
| 10    | ~7.6%     | ~5 hrs    |
| 20    | **~6.9%** | **~7 hrs** |
| 50    | ~7.0%     | ~12 hrs   |
| 100   | ~7.3%     | ~19 hrs   |

✅ **结论**：较小的 $ K=20 $ 即可捕获主要局部响应，进一步增加 $ K $ 不显著改善精度，反而大幅增加训练成本。因此选择 $ K=20 $（benchmark）、$ K=25 $（Mussafah）为最优折衷。

---

## 4. 关键结论和发现

### 主要发现
1. **局部响应必须用局部学习来解决**  
   全域学习（Vanilla DeepONet）在高度稀疏响应问题上失效，因其损失函数被无效区域主导。

2. **自适应学习域能显著提升精度与鲁棒性**  
   KNN 动态选择影响区节点，使模型始终关注“有意义”的区域，是应对移动荷载的关键。

3. **几何先验信息极大增强 trunk 表示能力**  
   距离感知特征明确建模了荷载-节点的空间关系，帮助网络理解衰减规律和方向依赖性。

4. **混合建模范式（Hybrid Paradigm）最具前景**  
   “数据驱动预测 + 物理引导外推”（即 Schur complement）既保留了 ML 的高效性，又保障了解的物理合理性。

5. **影响面生成效率实现质的飞跃**  
   从传统 FEM 的 **28.6 小时** 缩短至 **0.37 小时**，为实时桥梁监测与数字孪生提供了实用工具。

---

### 方法的局限性
1. **目前仅限于线弹性静力分析**  
   未考虑动力学、非线性材料行为（如裂缝、塑性）、温度效应或疲劳损伤演化。

2. **自适应域基于几何邻近性（KNN），未融合结构特性**  
   当前 KNN 仅依据欧氏距离，未考虑刚度分布、支座约束或实际传力路径，可能导致次优节点选择。

3. **依赖高质量的 reduced-order 模型进行训练数据生成**  
   若等效壳模型不能准确反映原结构行为，则会影响最终代理模型的泛化能力。

4. **尚未验证跨桥型迁移能力**  
   模型针对特定桥梁设计，不同拓扑结构需重新训练。

---

### 未来工作方向
1. **拓展至动态与时变问题**  
   引入 temporal operator learning 或 recurrent 架构，用于移动车辆的动力响应预测。

2. **开发物理感知的 adaptive domain selection**  
   探索基于图神经网络（GNO）、注意力机制或灵敏度分析的方法，构建更智能的影响区识别策略。

3. **引入 transfer learning 与 meta-learning**  
   实现跨桥梁类型的快速适配，减少重复训练成本。

4. **集成不确定性量化（UQ）模块**  
   如 Bayesian DeepONet，用于评估预测置信度，在健康监测中更具实用性。

5. **向非线性与损伤演化建模延伸**  
   结合 history-dependent 算子学习框架，应用于老化桥梁的安全评估。

---

> 🔚 **总结**：  
> 本文提出的 **AD-DeepONet** 是面向大型公路桥梁局部响应分析的一项突破性工作。它通过“**自适应学习域 + 几何增强 trunk + 物理重建**”三位一体的设计，成功克服了传统 ML 方法在稀疏响应建模中的根本瓶颈，在 **精度、效率与可扩展性** 上全面超越现有方案，为桥梁数字孪生、快速影响面分析和实时状态评估提供了强有力的工具。

</details>

---

### 11. [Multi-Task Bayesian In-Context Learning](https://arxiv.org/abs/2606.20538)

**Authors**: Qingyang Zhu, Eric Karl Oermann, Kyunghyun Cho  
**Category**: cs.LG  
**Published**: 2026-06-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.20538v1  

#### Abstract
Bayesian predictive inference provides a principled framework for uncertainty quantification, data efficiency, and robust generalization. However, exact inference is often intractable, and scalable approximations may remain computationally expensive or require restrictive modeling assumptions that d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Multi-Task Bayesian In-Context Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **amortized in-context learning**（如 PFNs、TabPFN）虽然能高效模拟贝叶斯推理，但存在一个根本缺陷：  
- 它们在训练时隐式地将一个固定的先验（prior）编码进模型权重中，无法在测试时灵活调整先验。
- 当测试任务的先验分布与训练分布不一致（即发生 **distribution shift** 或 **out-of-meta-distribution, OoMD**）时，模型缺乏显式的适应机制，导致鲁棒性差。

### **提出的新方法**
作者提出了 **Multi-Task Bayesian In-Context Learning (Multi-Task ICL)**，其核心思想是：
- 将先验信息显式表示为上下文中的 **前缀数据集（prefix datasets）**。
- 在输入序列中，先拼接若干个来自相同先验的任务数据作为 `prior` 前缀，再拼接目标任务的数据作为 `target` 上下文。
- Transformer 模型通过条件于这些前缀数据，学习如何动态调整其预测行为，从而实现对不同先验的适应。

> ✅ **关键创新**：首次将 **hierarchical Bayesian inference** 的结构嵌入到 ICL 框架中，并通过数据前缀提供可控制的“先验接口”。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **灵活性** | 支持测试时零参数更新地切换先验（zero-shot prior adaptation） |
| **鲁棒性** | 在 OoMD 和重尾先验等挑战场景下仍保持良好泛化能力 |
| **效率** | 推理速度比 MCMC/SVI 快数个数量级 |
| **准确性** | 在多种任务上匹配甚至接近 oracle Bayesian 预测器 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验分为合成数据与真实世界数据两类：

#### **(1) 合成数据集（Synthetic Datasets）**
- **任务类型**：
  - 线性回归（Linear Regression）
  - 逻辑回归（Logistic Regression）
- **先验家族（Prior Families）**：
  - 高斯先验 $ \mathcal{N}(\mu\mathbf{1}, I) $
  - 学生t分布先验 $ \text{StudentT}(\mu\mathbf{1}, I) $（用于测试重尾鲁棒性）
  - 流模型生成的复杂高维先验（Spiral Flow-based priors）
- **元分布（Meta-distribution）**：
  - 超参数 $ \lambda \sim p(\lambda) $ 控制每轮 episode 的先验参数（如均值 $\mu$、自由度 $\nu$）

#### **(2) 真实世界数据集**
- **ERA5 气候数据集**：
  - 预测欧洲地区的地表气温（spatiotemporal temperature prediction）
  - 输入：纬度、经度、时间、海拔
  - 辅助数据：同一空间区域但不同时间段的历史温度数据，用作 prior 前缀

---

### **实验设置与评估指标**

| 设置项 | 描述 |
|--------|------|
| **模型架构** | Decoder-only Transformer（GPT-2 风格），使用 Rotary Position Embedding |
| **上下文构造** | `[prior] (x₁,y₁)...(xₘ,yₘ) [prior] ... [target] (x₁,y₁)...(xₜ₋₁,yₜ₋₁), xₜ` |
| **训练目标** | 最大化目标任务的负对数似然（NLL） |
| **训练方式** | 分层采样：先从 meta-prior 采超参，再生成多个任务数据 |

#### **评估指标**
- **KL 散度**：从模型预测的 PPD 到 oracle PPD 的 KL，主指标
- **Total Variation (TV) divergence**：辅助验证一致性
- **Wall-clock time**：比较推理效率
- **NLL / MSE**：在 ERA5 数据上的实际性能

#### **基线方法对比**
| 方法 | 类型 | 是否知道真生成过程 | 是否渐近无偏 |
|------|------|------------------|-------------|
| **MCMC / SVI (Oracle)** | 条件于真实先验参数 $ \lambda $ | ✅ | MCMC ✅, SVI ❌ |
| **MCMC-HIER / SVI-HIER** | 层次贝叶斯，需从 prior datasets 推断 $ \lambda $ | ✅ | MCMC ✅, SVI ❌ |
| **ICL (no prefix)** | 传统 ICL，无 prior 前缀 | ❌ | ❌ |
| **ICL (with prefix)** | 本文方法 | ❌ | ✅（经验上逼近 oracle） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 在 Meta-Distribution 内部（IMD）表现**
- **线性回归**（Fig. 2）：
  - Multi-Task ICL（带前缀）与 MCMC-HIER 几乎完全重合，KL ≈ 0
  - 显著优于无前缀 ICL
- **逻辑回归**（Fig. 3）：
  - 在低数据 regime（target context length=5），ICL-with-prefix 匹配 MCMC-HIER，而 no-prefix 明显落后
  - 在高数据 regime，likelihood 主导，两者差距缩小 → 符合贝叶斯 posterior concentration 理论

#### **(2) Out-of-Meta-Distribution (OoMD) 泛化**
- **重尾先验测试（StudentT 分布）**（Fig. 5）：
  - 当测试先验进入方差/均值未定义区域（$\log \nu \leq 0$），所有方法性能下降
  - 但 Multi-Task ICL 表现出系统性的阈值效应：只要训练混合中包含足够重尾成分（如 $\log \nu \geq -2$），即可泛化到极端重尾测试分布
  - 性能模式与 MCMC-HIER 高度一致 → 表明学到了真正的层次推理机制
- **流模型先验（Spiral Flow）**（Fig. 6）：
  - MCMC-HIER 收敛慢，单次推理耗时 >1 秒
  - Multi-Task ICL 平均推理时间仅 **0.0065 秒**，KL ≈ 0.12，实现**三个数量级加速**

#### **(3) 真实世界任务（ERA5）**
| 配置 | Val NLL | Test NLL | 2020 Test NLL |
|------|---------|----------|--------------|
| MT, K=0 | -1.72 | -2.02 | -2.00 |
| **MT, K=2** | **-2.29** | **-2.33** | **-2.31** |
| Set-MT, K=2 | -2.16 | -2.18 | -2.17 |

- **结论**：
  - 在 IID 场景下，使用两个 prior dataset 显著提升性能
  - 在 OOD 时间划分下，set-aggregation 更鲁棒（因减少顺序依赖）

---

### **消融实验结果**

#### **(1) 前缀是否被正确解释？**
- **定性检查**（Fig. 4a）：
  - 改变 prior 前缀的均值，模型输出的 logit 分布随之系统性偏移 → 表明前缀确实影响预测
- **排除“证据池化”假说**（Fig. 4b）：
  - 若模型只是把 prior 数据当作 target 数据一起拟合，则应接近 MCMC-HIER_POOL
  - 实际上模型更接近 MCMC-ORACLE → 说明它真正执行了 hierarchical inference

#### **(2) 不同数量的 prior datasets (K) 影响**
- **Table 7 & 9**：
  - 随着 K 增加，模型对 prior 前缀的敏感度显著降低（pairwise KL ↓）
  - 表明模型实现了 **shrinkage effect**：更多 prior 数据 ⇒ 更稳定的先验估计 ⇒ 更集中的预测分布
- **长度外推测试（K=15,20 > max training K=10）**：
  - 性能在 OoMD 下下降，归因于 **sequence length extrapolation challenge**

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Multi-Task ICL 成功实现了 amortized hierarchical Bayesian inference**：
   - 在各种 prior 家族下，定量匹配 oracle Bayesian 预测器
   - 机制上符合贝叶斯推理特性（如 shrinkage、posterior concentration）

2. ✅ **支持测试时零参数更新的 prior adaptation**：
   - 通过更换前缀数据即可改变模型的行为，无需 retraining/fine-tuning

3. ✅ **具备强大的 OoMD 泛化能力**：
   - 即使测试先验超出训练分布支持范围，也能稳健推理
   - 泛化模式与 MCMC-HIER 对齐，表明学到的是通用推理机制而非过拟合

4. ✅ **极高的推理效率**：
   - 相比 MCMC/SVI，速度快 **orders of magnitude**
   - 适用于需要快速决策的应用场景（如临床预测、药物发现）

---

### **局限性**
| 局限 | 说明 |
|------|------|
| **Attention 复杂度** | 序列长度随 K*M 增长，attention cost 为 $ O((KM + T)^2) $，限制扩展性 |
| **非置换不变性** | 当前架构对数据顺序敏感（尽管实证影响小） |
| **长度外推困难** | 当测试时 K 远大于训练最大值时，性能下降明显 |
| **依赖大量模拟训练数据** | 需要从 generative model 中采样百万级 episodes |

---

### **未来工作方向**
1. **设计 permutation-equivariant 架构**：例如使用 Set Transformer 或 Pooling Mechanism 处理 prior datasets
2. **改进长序列建模能力**：引入稀疏注意力、状态空间模型（SSM）等以支持更大 K
3. **应用于更多现实场景**：
   - 个性化医疗：用相似患者历史记录作为 prior context
   - 药物研发：用历史试验数据引导新化合物筛选
4. **探索 online meta-adaptation**：允许在部署后继续积累 prior 数据并动态更新 meta-knowledge

---

> 🔗 **代码开源**：https://github.com/martianmartina/multi-task-bayesian-icl/

</details>

---

### 12. [ITNet: A Learnable Integral Transform That Subsumes Convolution, Attention, and Recurrence](https://arxiv.org/abs/2606.19538)

**Authors**: Ashim Dhor, Rasel Mondal, Pin Yu Chen  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.19538v1  

#### Abstract
Convolutional networks, recurrent networks, and transformers each encode different inductive biases -- locality, sequential memory, and content-dependent pairwise interaction -- and have remained mathematically distinct since their inception. We show that this fragmentation reflects not a fundamenta...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ITNet: A Learnable Integral Transform That Subsumes Convolution, Attention, and Recurrence

## 1. 论文的主要贡献和创新点

### 解决的问题
深度学习领域长期存在三大主流架构：**CNN**（卷积神经网络）、**Transformer** 和 **RNN**（循环神经网络），它们分别针对图像、文本和序列数据设计，各自编码了不同的归纳偏置（inductive bias）——如局部性、全局依赖性和时序记忆。这种架构碎片化导致：
- 需要为不同模态选择特定模型；
- 多模态任务需要复杂的融合模块；
- 缺乏一个统一的数学框架来理解信号处理的本质。

本文提出，这种分裂并非源于根本性的多样性，而是对一个更基础数学对象的不完整认知。

### 提出的新方法和新思路
论文提出了 **Integral Transform Network (ITNet)**，一种基于**可学习积分变换**（learnable integral transform）的统一架构。

- **核心思想**：将卷积、自注意力（self-attention）和递归（recurrence）都视为一个通用的、可学习的积分算子的特例。
- **数学形式**：ITNet 算子定义为：
  $$
  (\mathcal{K}_\theta[u])(x) = \int_\Omega K_\theta(x, y, u(x), u(y)) u(y) d\mu(y) + W_o u(x)
  $$
  其中，核函数 $K_\theta$ 是一个小型 MLP，其输入同时依赖于查询位置 $x$、键位置 $y$ 以及两者的特征 $u(x)$ 和 $u(y)$。
- **创新点**：
  1. **统一框架**：首次严格证明 **CNN、Transformer 和 RNN**（包括 LSTM, GRU, S4, Mamba）都可以作为 ITNet 在特定参数化下的精确特例。
  2. **通用逼近能力**：ITNet 是连续算子的**通用逼近器**（universal approximator），其表达能力严格强于上述所有经典架构。
  3. **自适应归纳偏置**：交互模式不是硬编码的（如 CNN 的局部性或 Transformer 的点积注意力），而是直接从数据中学习得到。

### 相比现有方法的优势
- **统一性**：一个单一的架构可以处理多种模态，无需为不同任务设计专用模型。
- **更强的表达能力**：能够表示 CNN、Attention 或 RNN 单独无法捕捉的复杂算子。
- **端到端学习**：通过学习核函数，模型可以自动适应数据的最佳交互模式，而非受限于预设的架构假设。

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了四大模态，验证了 ITNet 的通用性：
- **视觉 (Vision)**：`ImageNet-1K`（图像分类）
- **语言 (Language)**：`GLUE`（自然语言理解基准）
- **3D 几何 (3D Geometry)**：`ModelNet40`（点云分类）
- **多模态 (Multimodal)**：`VQA v2` 和 `NLVR2`（视觉问答和视觉推理）

### 实验设置和评估指标
- **模型规模**：报告了三种规模的 ITNet 模型：
  - **ITNet-S** (22M 参数)
  - **ITNet-B** (86M 参数)
  - **ITNet-L** (307M 参数)
- **评估指标**：
  - `ImageNet-1K`：Top-1 准确率
  - `GLUE`：各任务平均得分
  - `ModelNet40`：总体准确率 (OA)
  - `VQA v2` / `NLVR2`：测试集准确率
- **实现优化**：为了应对 $O(n^2d^2)$ 的计算开销，论文开发了三种高效策略：
  1. **Tiled Kernel Fusion**：用于短序列，通过融合内核计算和矩阵乘法实现高效的片上内存计算。
  2. **Importance-weighted Monte Carlo Integration**：随机采样键，降低复杂度至 $O(nMd)$。
  3. **Learned Low-Rank Factorization**：将核分解为低秩形式，实现 $O(ndr)$ 的线性时间计算。

### 基线方法对比
与各领域的代表性模型进行了比较：
- **视觉**：ResNet-50, ConvNeXt, DeiT, Swin Transformer
- **语言**：BERT, RoBERTa, DeBERTa
- **点云**：PointNet, PointNet++, DGCNN
- **多模态**：ViLT, UNITER, METER, BLIP

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
在所有基准测试中，单个 ITNet 架构均匹配或超过了专用的基线模型：

| 任务 | 数据集 | ITNet 性能 | 最佳基线性能 | 结果 |
| :--- | :--- | :--- | :--- | :--- |
| **图像分类** | ImageNet-1K | **85.8%** (ITNet-L) | 84.4% (BiFormer-B) | **显著超越** |
| **语言理解** | GLUE | **83.1** (ITNet-L) | 82.3 (RoBERTa-base) | **达到 SOTA 水平** |
| **点云分类** | ModelNet40 | **94.6%** (ITNet-B) | 94.1% (PointMLP) | **超越** |
| **视觉问答** | VQA v2 | **83.6%** (ITNet-L) | 78.3% (BLIP) | **大幅超越** |
| **视觉推理** | NLVR2 | **84.1%** (ITNet-L) | 83.5% (BLIP) | **超越** |

### 消融实验结果
消融实验验证了模型设计的关键组件：

- **核输入的重要性**（见 Table 5）：
  - 移除 **Hadamard 项**（$u(x) \odot u(y)$）导致性能下降，表明逐元素特征交互至关重要。
  - 移除 **相对位置**（$x-y$）或 **距离**（$\|x-y\|^2$）信息会损害性能，强调了几何关系的作用。
  - 仅用 **纯内容** 或 **纯位置** 输入的变体表现最差，证实了联合建模的必要性。

- **高效计算策略的有效性**：
  - **低秩分解**（Low-Rank, r=64）在 `ImageNet-1K` 上仅损失 0.1% 准确率，但吞吐量提升超过 2 倍。
  - **蒙特卡洛采样**（MC, M=128）同样实现了高效率与高性能的平衡。

- **其他消融**：
  - **傅里叶特征**（Fourier Features）对于捕捉高频空间模式至关重要，移除后性能下降明显。
  - **模态平衡**（Balanced Measure）在多模态任务中优于均匀权重，说明合理分配模态重要性是有效的。

## 4. 关键结论和发现

### 主要发现
1. **统一性成立**：卷积、自注意力和递归并非互斥的范式，而是同一个可学习积分变换算子的不同实例。这为深度学习提供了一个更统一、更基础的数学视角。
2. **学习优于预设**：与其为不同数据类型预设固定的交互规则（如局部卷积或全局注意力），不如让模型从数据中直接学习最优的交互模式。ITNet 通过一个共享的、可学习的核函数实现了这一点。
3. **通用性与强大性**：一个单一的 ITNet 架构就能在视觉、语言、3D 和多模态任务上达到或超越专用模型的性能，证明了“通用目的、模态无关”架构的可行性。

### 方法的局限性
- **计算成本**：尽管有优化策略，但完整的 ITNet 计算成本仍高于高度优化的专用模型（如 FlashAttention）。精确版本的内存占用较高。
- **生成任务未验证**：论文主要在理解任务上进行评估，尚未在自回归生成任务（如长文本生成）上充分验证其因果核的有效性。
- **超大规模扩展**：将 ITNet 扩展到数十亿甚至万亿参数级别，可能面临优化稳定性和核评估成本的挑战。

### 未来工作方向
- **扩展到生成模型**：将 ITNet 应用于自回归语言建模等生成任务，以全面检验其作为统一框架的能力。
- **改进训练和扩展性**：开发更高效的核参数化方法和训练策略，以支持更大规模的模型。
- **探索更优的融合机制**：虽然去除了显式的融合模块，但如何进一步优化多模态场景下的训练效率（如模块化或部分解耦训练）是一个开放方向。

</details>

---

### 13. [TelcoAgent: A Scalable 5G Multi-KPM Forecasting With 3GPP-Grounded Explainability](https://arxiv.org/abs/2606.19821)

**Authors**: Geon Kim, Dara Ron, Sukhdeep Singh, Suyog Moogi, Pranshav Gajjar, V V N K Someswara Rao Koduri, Een Kee Hong, Vijay K. Shah  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.19821v1  

#### Abstract
Key Performance Measurement (KPM) forecasting is essential for proactive network management of 5G and next-generation telecom networks. However, existing machine learning (ML) approaches face significant limitations in scalability and explainability, restricting their effectiveness in real-world dep...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《TelcoAgent: A Scalable 5G Multi-KPM Forecasting With 3GPP-Grounded Explainability》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于 **Machine Learning (ML)** 的 **KPM 预测模型**在 5G 网络中面临三大瓶颈：
- **可扩展性差**：需要为每个基站（cell）单独训练模型，计算开销大，难以部署到大规模网络；
- **缺乏可解释性**：仅输出预测值，无法提供根因分析（root cause diagnosis）或操作建议；
- **忽略跨 KPM 因果关系**：未有效建模不同 KPM 之间的非线性依赖和 3GPP 协议定义的因果机制。

### 🚀 提出的新方法与创新思路
作者提出 **TelcoAgent** —— 一个基于 **LLM Agent + Time-Series Foundation Model (TSFM)** 的框架，实现 **零样本（zero-shot）、可扩展、可解释的多 KPM 预测**。其核心由三个模块构成：

1. **Knowledge Graph Construction Pipeline**
   - 利用 **三代理（three-agent）自动化流程**从 3GPP 规范文档中提取并构建结构化知识图谱（Knowledge Graph）；
   - 包含：Extractor（抽取三元组）、Aligner（对齐术语）、Evaluator（置信度评分）；
   - 将 KPM 映射到其定义、公式、物理层约束及因果链，形成领域锚定的知识基础。

2. **Zero-Shot Prediction Pipeline**
   - 基于 **TSFM（如 Chronos-2, Moirai）** 进行跨小区、跨 KPM 的联合预测；
   - 无需针对特定站点微调，支持“开箱即用”的城市级预测。

3. **Explanation & Reasoning Pipeline（ReAct 架构）**
   - 结合 **PAX-TS 敏感性分析** 和 **3GPP Knowledge Graph 检索**，识别驱动异常的关键 KPM 及其协议机制；
   - 输出带有证据链的 **可操作建议（actionable recommendations）**，例如调整 OLLA 参数或 HARQ 重传次数；
   - 引入 **自验证机制（self-verification）** 对数值进行交叉检查，防止 LLM 幻觉。

### 🔍 相比现有方法的优势
| 维度 | 传统 ML 方法 | TelcoAgent |
|------|--------------|-----------|
| **Scalability** | 每站训练，不可扩展 | Zero-shot，全网统一模型 |
| **Explainability** | 黑箱预测，无诊断能力 | 基于 3GPP 的因果推理与推荐 |
| **Cross-KPM Modeling** | 多数独立建模或弱相关 | 联合建模 + 显式因果路径匹配 |
| **Deployment Cost** | 高维护成本 | 低运维负担，支持自动决策 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **真实世界 5G KPM 数据集**，来自美国某运营商；
- 时间跨度：**2025年9月至11月，共3个月**；
- 地理范围：**Texas 州 200 个基站（cells）**；
- 频段：**PCS band (1850–1990 MHz)**；
- 采样粒度：**每小时 1 条记录（1-hour granularity）**；
- 包含 **7 个 KPMs**：
  - RRC Connected Users (RRC_Conn)
  - DL CQI (Channel Quality Indicator)
  - iBLER / rBLER (Initial/Residual Block Error Rate)
  - MAC DL Throughput (MAC_DL_Th)
  - PRB Utilization (PRB_Util)
  - IP Throughput (IP_Th)

### ⚙️ 实验设置
#### 预测任务
- **预测目标**：未来 7 天（168 小时）的 KPM 走势；
- 输入历史窗口长度通过调参优化：
  - Chronos-2 / Moirai：81 天
  - MOMENT：22 天

#### 基线模型对比
| 类型 | 模型 | 说明 |
|------|------|------|
| **Supervised Baselines** | N-BEATS, GRU, MLP | 每站独立训练，使用前 81 天数据 |
| **Zero-Shot TSFMs** | Chronos-2, Moirai-1.1-R-base, MOMENT-1-large | 不需微调，直接预测 |

#### 评估指标
##### 预测性能
- **nRMSE**（normalized RMSE）：衡量误差相对于观测值范围的比例；
- **MASE**（Mean Absolute Scaled Error）：相对于朴素季节基准的相对精度。

##### 解释质量
- **Faithfulness**（保真度）：生成声明符合 3GPP 标准的程度（越高越好）；
- **Answer Relevancy**（答案相关性）：解释是否贴合异常事件，在四个维度上打分：
  - KPM coverage
  - 时间窗口匹配
  - 因果机制正确性
  - 推荐动作合理性

所有解释结果由 **ORAN-Sight**（领域专用 LLM Judge）打分，并映射为 [0,1] 分数。

---

## 3. 主要实验结果和性能指标

### 📈 预测性能（见 Table I）
在 **200 个小区平均**下，**Chronos-2 表现最优**，全面优于其他模型：

| 模型 | 平均 nRMSE | 平均 MASE |
|------|------------|-----------|
| **Chronos-2** | **0.15** | **0.74** |
| Moirai-1.1-R-base | 0.16 | 0.76 |
| MOMENT-1-large | 0.18 | 1.02 |
| N-BEATS | 0.19 | 0.94 |
| GRU | 0.22 | 1.07 |
| MLP | 0.20 | 1.10 |

> ✅ **关键发现**：Chronos-2 在所有 7 个 KPM 上均取得最佳表现，尤其在 **RRC_Conn** 和 **PRB_Util** 上显著领先，表明其能有效捕捉流量负载相关的跨通道依赖。

#### 消融分析：为何 MOMENT 表现较差？
- MOMENT 是 **channel-independent** 设计，无法建模 KPM 间耦合；
- 在 **RRC** 和 **PRB** 等受用户行为影响大的指标上明显落后；
- 说明：**跨 KPM 建模对高负载敏感型 KPM 至关重要**。

### 💡 解释质量（见 Fig. 5）
| 指标 | 使用 KG | 无 KG |
|------|--------|-------|
| **Faithfulness** | 0.615 | 0.643 |
| **Answer Relevancy** | **0.807** | 0.748 |

> ❗ 虽然移除 KG 后 Faithfulness 微升（因减少文本转译偏差），但 **Answer Relevancy 下降 7.4%**，说明 **KG 显著提升了解释的操作价值**。

#### 自验证结果（Numerical Fidelity）
- 共验证 **33 项数值**（包括预测均值、斜率、敏感性分数等）；
- **99.8% 的数值在容差范围内匹配**；
- 表明系统具备高度可靠性，有效抑制 LLM 幻觉。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **TelcoAgent 实现了真正的 zero-shot、可扩展 KPM 预测**：
   - 无需站点微调即可在 200 个异构小区上稳定运行；
   - Chronos-2 在 nRMSE 上比最强监督模型（N-BEATS）低 **21%**。

2. **引入 3GPP Knowledge Graph 显著增强了解释的实用性**：
   - 支持从“预测”到“诊断”再到“行动”的闭环；
   - 输出建议具体到参数级别（如 OLLA target、PF scheduler weight）；

3. **结合 PAX-TS 与 GraphRAG 实现因果感知推理**：
   - 成功区分相关性与因果性，避免误判；
   - 示例中准确识别出 MAC_DL_Eff 下降是 Throughput 异常的主因。

4. **系统具备高保真与抗幻觉能力**：
   - 数值自验证机制确保输出可信；
   - 为实际部署提供了安全保障。

### ⚠️ 局限性
- **Knowledge Graph 构建仍存在语义漂移风险**：尽管 evaluator agent 控制质量，但复杂协议描述可能被简化或误解；
- **当前仅处理单小区上下文**：尚未融合邻近小区的时空上下文（spatio-temporal context）；
- **依赖高质量 TSFM**：若 TSFM 预测失准，后续解释将产生连锁错误（garbage in, garbage out）；
- **未在实时控制环中验证**：目前为离线评估，尚未作为 O-RAN rApp 部署于 closed-loop 控制。

### 🔮 未来工作方向
1. 扩展 Knowledge Graph 覆盖更多 3GPP/O-RAN 规范（如 TS 38.473, O-RAN WGs）；
2. 引入 **spatio-temporal modeling**，利用 neighboring cells 提升预测鲁棒性；
3. 将 TelcoAgent 部署为 **O-RAN rApp**，实现 near-real-time resource orchestration；
4. 探索 **feedback loop**：将操作反馈回模型以持续优化推理逻辑。

---

> 🧩 **一句话总结**：  
> TelcoAgent 首次将 **TSFM 的零样本预测能力** 与 **LLM Agent 的 3GPP 锚定推理能力** 深度融合，实现了 **可扩展、可解释、可执行** 的 5G 网络智能运维新范式。

</details>

---

### 14. [Neural network surrogates with uncertainty quantification for inverse problems in partial differential equations](https://arxiv.org/abs/2606.20417)

**Authors**: Christian Jimenez-Beltran, Aretha L. Teckentrup, Antonio Vergari, Konstantinos C. Zygalakis  
**Category**: cs.LG  
**Published**: 2026-06-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.20417v1  

#### Abstract
Inverse problems for differential equations arise throughout science and engineering, where one seeks to infer unknown model parameters from noisy or incomplete observations. Traditional numerical methods for these problems are often computationally expensive, particularly in Bayesian settings where...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Neural Network Surrogates with Uncertainty Quantification for Inverse Problems in Partial Differential Equations

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **Bayesian inverse problems** 中，求解 **Partial Differential Equations (PDEs)** 的前向模型通常需要昂贵的数值计算（如 FEM、谱方法等），尤其是在高维参数空间下。传统方法（如 MCMC）需反复调用前向模型，导致计算成本极高。此外，神经网络作为 surrogate model 虽然高效，但往往 **缺乏可靠的不确定性量化（Uncertainty Quantification, UQ）**，容易在训练数据稀疏时产生过置信（overconfident）预测。

### 提出的新方法：DeepGaLA
本文提出 **DeepGaLA (Deep Galerkin via Laplace Approximation)**，一种结合 **Deep Galerkin Method (DGM)** 与 **Laplace Approximation** 的神经网络 surrogate 方法，用于解决参数化 PDEs 的逆问题。

#### 创新点：
- **首次将 Laplace Approximation 引入 DGM 框架**，实现对神经网络输出的不确定性估计。
- 将 DGM 的优化目标重新解释为 **Maximum a Posteriori (MAP) 估计**，并基于此构建后验近似。
- 采用 **last-layer Bayesianization** 策略：仅对网络最后一层权重进行随机化建模，显著降低计算复杂度。
- 使用 **Generalized Gauss-Newton (GGN)** 近似 Hessian 矩阵，提升 Laplace Approximation 的可扩展性。

### 相比现有方法的优势
| 特性 | DeepGaLA | Gaussian Process (GP) | PINN / DGM |
|------|----------|------------------------|------------|
| 不确定性量化 | ✅（校准良好） | ✅（天然支持） | ❌（通常缺失） |
| 高维参数扩展性 | ✅（线性依赖于网络结构） | ❌（立方依赖于训练点数 $O(N^3)$） | ✅ |
| 非线性 PDE 支持 | ✅ | ❌（PIGP 限于线性算子） | ✅ |
| 推理效率 | ✅（常数时间在线评估） | ⚠️（随训练点增长而变慢） | ✅ |

> **核心优势**：DeepGaLA 在保持神经网络高效性和非线性建模能力的同时，引入了可扩展的不确定性量化机制，在低数据场景下表现优于确定性 surrogate 和部分 GP 方法。

---

## 2. 核心实验方法和设置

### 实验任务
研究三个典型的 PDE 逆问题：
1. **1D Elliptic Boundary Value Problem**（线性）
2. **2D Elliptic Inverse Problem**（线性）
3. **Navier-Stokes Inverse Problem**（非线性）

目标是从有限、含噪的观测 $ y = \mathcal{G}_x(\theta) + n $ 中推断未知参数 $\theta$，其中 $\mathcal{G}_x$ 是观测算子（即 PDE 解在特定点的取值）。

### 数据集生成方式
- 所有“真实”数据由高精度数值求解器生成：
  - **FEM**（有限元法）用于椭圆型方程
  - **Pseudo-spectral method + Crank-Nicolson** 用于 Navier-Stokes 方程
- 参数 $\theta$ 通过 **Karhunen–Loève Expansion (KLE)** 构造空间随机场
- 观测位置固定（如 6 个采样点），噪声服从 $ \mathcal{N}(0, \sigma^2 I) $

### 实验设置
- **Surrogate 类型**：
  - **DeepGaLA**：使用 modified feedforward NN + Fourier features + Laplace Approximation
  - **PIGP (Physics-Informed Gaussian Process)**：作为主要 baseline
- **MCMC 设置**：
  - 使用 **Random Walk Metropolis-Hastings (RWMH)** 采样
  - 采样 $2.5 \times 10^6$ 步以确保 ESS ≥ 1000
- **训练配置**：
  - DeepGaLA 使用不同大小的训练集 $N$（从几十到上万 collocation points）
  - 优化器：Adam，epoch 数：5000
  - Laplace Approximation 在 MAP 权重处执行

### 评估指标
1. **$Q_{\text{val}}$**：基于 **Delayed Acceptance MCMC (DA-MCMC)** 的诊断指标，衡量 surrogate 后验与“精细模型”后验的一致性。
   - 定义为细粒度模型接受提案的比例（使用 $10^4 \sim 10^5$ 次精细模型评估）
   - $Q_{\text{val}} \to 1$ 表示 surrogate 后验接近真实后验
2. **Error Bounds**：理论误差界（来自 Theorem 1），验证 $Q_{\text{val}}$ 的合理性
3. **Online Evaluation Time**：CPU 上单次 likelihood 评估耗时（ms）
4. **Posterior Visualization**：边缘分布对比图

### 基线方法对比
- **PIGP**：物理信息高斯过程，具备天然 UQ
- **Deterministic DGM / DeepONet**：无 UQ 的神经算子
- **Mean vs Marginal Approximation**：比较是否显式建模 surrogate 不确定性的影响

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（代表性）

#### 📊 表 1：1D BVP ($d_\theta=2$, $N=7000$)
| 方法 | $Q_{\text{val}} (\%)$ | Error Bound | CPU Time (ms) |
|------|------------------------|-------------|---------------|
| DeepGaLA (mean) | 99.01 ± 0.08 | $3.41 \times 10^{-4}$ | 0.22 |
| DeepGaLA (marginal) | 99.01 ± 0.26 | $3.41 \times 10^{-4}$ | 0.22 |
| PIGP | 99.20 ± 0.08 | $3.41 \times 10^{-4}$ | 0.23 |

> ✅ 结果表明 DeepGaLA 在准确性上与 PIGP 相当，且推理更快。

#### 📈 图表趋势总结
- **随着训练数据增加，$Q_{\text{val}}$ 单调上升至接近 100%**，说明 surrogate 后验收敛于真实后验。
- **在低数据 regime 下，marginal approximation 显著优于 mean approximation**：
  - 如在 1D BVP 中，$N=40$ 时：
    - $Q_{\text{val}}^{\text{mean}} = 1.45\%$
    - $Q_{\text{val}}^{\text{marginal}} = 17.68\%$
  - 表明 **纳入 surrogate 不确定性可有效防止过拟合和过置信**

#### ⏱️ 效率对比（Table 1 成本分析）
| 操作 | DeepGaLA | PIGP |
|------|---------|------|
| 均值评估 | $O(L)$ 层深相关 | $O(N)$ 训练点相关 |
| 方差评估 | $O(d_h^2)$ | $O(N^3)$ |

> 🔍 实验显示：当 $d_\theta$ 增大时，PIGP 需更多训练点才能维持精度，导致评估时间急剧上升；而 DeepGaLA 几乎不受影响。

#### 🔍 消融实验（隐含）
- **Fourier Features 的作用**：缓解 spectral bias，提升高频模式捕捉能力
- **Last-layer Laplace vs Full Bayesian NN**：
  - 文中指出 full Bayesian 更准确但不可行（计算代价过高）
  - Last-layer 已能提供有意义的不确定性（见 Fig. 9：误差与 std 地图对齐）
- **Marginal vs Mean Approximation**：
  - 在所有实验中，marginal 在低数据下均取得更高 $Q_{\text{val}}$
  - 验证了 **uncertainty-aware inference 的必要性**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **DeepGaLA 能够在不牺牲效率的前提下提供可靠的不确定性估计**，尤其适用于训练数据稀缺或参数维度高的场景。
2. ✅ **Marginal approximation 比 mean approximation 更鲁棒**，特别是在低数据 regime 下能避免错误集中。
3. ✅ **$Q_{\text{val}}$ 是一个有效的后验质量诊断工具**，即使没有真实后验也可用于评估 surrogate 可靠性。
4. ✅ **DeepGaLA 在线评估时间几乎恒定**，不随训练数据量增长而恶化，适合大规模部署。
5. ✅ **相比 PIGP，DeepGaLA 更适合非线性 PDE 和高维参数空间**，突破了 GP 方法的应用边界。

### 方法的局限性
- ❗ **Laplace Approximation 是局部近似**，假设后验为高斯形，可能无法捕捉多峰分布。
- ❗ **仅对最后一层进行贝叶斯化**，限制了不确定性传播的完整性。
- ❗ **超参数调优敏感**：文中发现联合优化 $\gamma$ 和 $\sigma$ 会导致不确定性快速坍缩，最终选择固定 $\sigma=1$。
- ❗ **训练仍依赖大量 collocation points**，尽管少于标准 PINN/DGM，但仍需高性能计算资源。

### 未来工作方向
1. **探索更强大的 Bayesian NN 近似方法**：如使用 **low-rank structure** 或 **variational inference** 替代 Laplace Approximation。
2. **自适应训练策略**：结合 active learning 或 DA-MCMC 输出动态选择新的 collocation points。
3. **拓展至 time-dependent 和 chaotic systems**：当前实验集中在稳态或规则动力系统。
4. **与其他加速技术结合**：如 multilevel MCMC、surrogate preconditioning。
5. **理论分析深化**：建立 DeepGaLA 的收敛速率理论，目前尚无完整理论支撑。

---

> 💡 **总体评价**：  
> 该论文成功地将 **Bayesian 思想** 与 **深度学习求解 PDE** 相结合，提出了一个实用、高效且具备 UQ 能力的 surrogate 框架 —— **DeepGaLA**。它不仅在性能上媲美经典 GP 方法，还在可扩展性和适用范围上实现了突破，为复杂系统的 Bayesian inverse problems 提供了一条可行路径。

</details>

---

### 15. [Which Pairs to Compare for LLM Post-Training?](https://arxiv.org/abs/2606.19607)

**Authors**: Jiangze Han, Vineet Goyal, Will Ma  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.19607v1  

#### Abstract
Preference-based post-training has become a central paradigm for aligning language models. A common data-collection strategy is to generate a small set of completions for each prompt and label the resulting comparison pairs. However, human preference labels are often much more expensive than generat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Which Pairs to Compare for LLM Post-Training?

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于 **LLM 后训练中的偏好数据选择问题**（comparison curation）。在基于偏好的后训练（如 DPO）中，通常的做法是为每个 prompt 生成少量 completion 并标注所有或部分比较对。然而，**生成文本的成本远低于人工标注成本**。因此，一个更高效的数据收集策略是：先从参考模型（reference policy）生成一个较大的 completion 候选池，然后从中选择最具信息量的 comparison pairs 进行标注。

本文的核心问题是：**在有限的标注预算下，应该选择哪些 completion 对进行比较标注，以最大化最终 post-trained policy 的性能？**

### 提出的新方法与新思路
论文将 comparison selection 问题形式化为一个 **采样设计问题**（sampling-design problem），并提出了一个理论驱动的优化准则：

- **理论框架**：将 comparison curation 视为一个实验设计问题，目标是通过选择 informative 的 comparison pairs 来最小化最终 policy 在 KL-regularized RLHF 目标下的 **optimality gap**。
- **核心洞察**：作者证明了 comparison selection 对下游 policy 性能的影响可以通过一个单一的、依赖于设计的 **information matrix** $\Sigma_D(\theta)$ 来刻画。
- **优化准则**：推导出上界和下界匹配的 finite-sample bound，表明最优的设计应最小化以下 **trace criterion**：
  $$
  \text{tr}\left(I(\theta^*) \Sigma_D(\theta^*)^\dagger\right)
  $$
  其中 $I(\theta^*)$ 是 Fisher information matrix，$\Sigma_D(\theta^*)$ 是由采样设计 $D$ 决定的 design covariance matrix。
- **实用设计**：由于真实参数 $\theta^*$ 未知，提出使用参考模型参数 $\theta_0$ 作为代理，得到可实现的 **plug-in design**：
  $$
  D_{\text{plug}} \in \arg\min_D \text{tr}\left(I(\theta_0) \Sigma_D(\theta_0)^\dagger\right)
  $$

### 相比现有方法的优势
- **理论保证**：首次将 comparison selection 与最终 policy 的 RLHF optimality gap 直接联系起来，并提供了严格的 finite-sample 上下界分析。
- **超越启发式**：相比均匀采样（uniform sampling）或按参考模型概率采样（$\pi_0$-weighted heuristic）等简单策略，该方法有明确的理论依据，能系统性地选择对 policy 优化最有帮助的 comparisons。
- **样本效率更高**：实验证明，在相同标注预算下，该方法训练出的 policy 性能显著优于基线方法。

---

## 2. 核心实验方法和设置

### 数据集
实验涵盖了从合成环境到真实 LLM 微调任务的多种场景：

1. **合成实验**（Synthetic Experiments）
   - **Tabular Setting**：模拟每个 prompt 下有 $d$ 个候选 completion 的离散决策问题。
   - **Linear Contextual Setting**：每个 completion 有特征向量 $\phi(x,y)$，policy 为 softmax 形式 $\pi_\theta(y|x) \propto \exp(\theta^\top \phi(x,y))$。

2. **真实世界 LLM 微调任务**
   - **IMDb Dataset**：使用 GPT-2-large 在 IMDb 电影评论数据上进行 SFT 和 DPO 微调。使用 `siebert/sentiment-roberta-large-english` 作为 reward model。
   - **Anthropic-HH Dataset**：使用 Pythia-2.8B 在 Anthropic 的 Helpful-Harmless 数据集上进行微调，这是更贴近实际对齐任务的 benchmark。

### 实验设置和评估指标
- **标注预算**（Labeling Budget）：固定总比较对数量 $n$，比较不同方法在此预算下的表现。
- **评估指标**：
  - **合成实验**：直接计算 RLHF optimality gap $J(\pi^*) - J(\hat{\pi}_n)$。
  - **IMDb 实验**：绘制 **reward-KL frontier**，即在不同 $\beta$ 值下，policy 的 reward 与相对于 reference policy 的 KL 散度之间的权衡曲线。
  - **Anthropic-HH 实验**：使用 **GPT-4.1 作为自动裁判**（automatic judge），报告训练出的 policy 生成的回复相对于 HH 数据集中“chosen”回复的 **win rate**。

### 基线方法对比
论文对比了四种 comparison selection 策略：
1. **Oracle Design $D^*(\theta^*)$**：使用真实最优参数 $\theta^*$ 计算的理想设计（仅用于理论验证）。
2. **Plug-in Design $D^*(\theta_0)$**：使用参考模型参数 $\theta_0$ 的代理设计（本文提出的方法）。
3. **Uniform Sampling**：在所有可能的 comparison pairs 中均匀随机采样。
4. **$\pi_0$-weighted Heuristic**：根据参考模型 $\pi_0$ 的概率分布来采样 completion pairs。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
#### 合成实验（图1）
- **Tabular Setting**（图1a）：$D^*(\theta^*)$ 和 $D^*(\theta_0)$ 的 performance 几乎完全重合，且显著优于 uniform 和 $\pi_0$-weighted 方法。后者在弱信号（weak signal）环境下表现极差。
- **Linear Contextual Setting**（图1b）：$D^*(\theta_0)$ 紧随 oracle 设计，而 uniform 采样在小预算时效率较低，$\pi_0$-weighted 方法不仅性能差，方差也更大。

#### IMDb 实验（图2）
- **Response Selection Task**（图2a）：在 response 选择任务中，$D^*$-curated 数据在所有 $\beta$ 值下均实现了更好的 reward-KL 权衡。
- **Prompt Selection Task**（图2b）：在 prompt 选择任务中，$D^*$-curated 数据同样显著优于基准方法。

#### Anthropic-HH 实验（图3）
- 在两个不同预算（$N=80,400$ 和 $N=96,480$）下，$D^*$-curated 数据集在所有采样温度（0.25, 0.7, 1.0）下都取得了更高的 **win rate**。
- 例如，在 $N=96,480$ 且温度为 0.7 时，$D^*$ 方法的 win rate 超过基准约 5-10 个百分点。

### 消融实验
虽然论文未明确列出消融实验表格，但其实验设计本身构成了一种消融：
- 通过比较 $D^*(\theta^*)$ 和 $D^*(\theta_0)$，验证了使用 $\theta_0$ 作为代理的有效性。
- 通过对比 uniform 和 $\pi_0$-weighted，说明了简单启发式在非均匀输出分布下的失效。

---

## 4. 关键结论和发现

### 主要发现
1. **Comparison Selection 至关重要**：并非所有 comparison pairs 对 policy 优化的贡献是相等的。精心选择 informative 的 pairs 可以显著提升样本效率。
2. **理论指导实践**：提出的 trace criterion $\text{tr}(I(\theta_0)\Sigma_D(\theta_0)^\dagger)$ 是一个有效的 design objective，其对应的 plug-in design 在实践中表现优异。
3. **$\pi_0$-weighted 不可靠**：高概率的 completion 并不一定是信息量最大的 comparison 对象。盲目依赖参考模型的概率可能导致次优选择。
4. **统一的优化视角**：将 comparison curation 视为实验设计问题，为数据收集提供了一个系统性的、理论驱动的框架。

### 方法的局限性
1. **可实现性假设**（Realizability）：理论分析依赖于最优 policy $\pi^*$ 属于所考虑的 policy class 的假设，这在大型神经网络中可能仅近似成立。
2. **正则性假设**：需要满足 Fisher information 非退化、feature separation 等条件，这些在复杂模型中可能难以严格满足。
3. **离线设计**：当前方法是离线的（offline），不适应在线反馈。无法利用已标注数据动态调整后续的 comparison selection。
4. **计算开销**：计算 $D^*$ 设计需要求解一个优化问题（如 Frank-Wolfe），对于超大规模候选池可能带来额外计算负担。

### 未来工作方向
1. **自适应设计**（Adaptive Design）：开发能够根据已观察到的偏好标签动态调整 comparison selection 策略的在线算法。
2. **扩展到其他目标**：将此框架应用于除 DPO 外的其他 preference-based learning 方法，如 PPO 或其他变体。
3. **处理模型误设**（Misspecification）：研究当 realizability 假设不成立时，该设计准则的鲁棒性。
4. **结合主动学习**：将此信息论准则与 active learning 框架结合，进一步降低数据标注成本。

</details>

---

### 16. [PhysDrift: Bridging the Embodiment Gap in Humanoid Co-Speech Motion Generation](https://arxiv.org/abs/2606.19935)

**Authors**: Zhangzhao Liang, Xiaofen Xing, Mingyue Yang, Wenlve Zhou, Xiangmin Xu  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.19935v1  

#### Abstract
Humanoid robots require co-speech motions that are not only expressive and speech-aligned, but also physically executable under embodiment constraints. Existing co-speech generation pipelines are predominantly human-centric: motions are first generated in human-body representations such as SMPL-X an...

---

### 17. [Multi-Head Attention-Based Feature Extractor Integration with Soft Actor-Critic for Porosity Prediction and Process Parameter Optimization in Additive Manufacturing](https://arxiv.org/abs/2606.20087)

**Authors**: Kianoush Aqabakee, Leonardo Stella  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.20087v1  

#### Abstract
Additive manufacturing process optimization requires precise parameter control to minimize defects such as porosity. Traditional reinforcement learning (RL) approaches using discrete action spaces suffer from slow convergence and susceptibility to local optima, limiting their effectiveness for high-...

---

### 18. [CacheWeaver: Cache-Aware Evidence Ordering for Efficient Grounded RAG Inference](https://arxiv.org/abs/2606.19667)

**Authors**: Kaizhen Tan, Rong Gu, Mingyuan Li  
**Category**: cs.CL  
**Published**: 2026-06-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.19667v1  

#### Abstract
Retrieval-Augmented Generation (RAG) improves factual grounding, but it also lengthens prompts and raises prefill cost. Prefix caching in serving engines such as vLLM reduces this cost only when requests share the same token prefix. In grounded generation, however, adjacent queries may retrieve over...

---

### 19. [When Does Streaming Tool Use Help? Characterizing Tool-Intent Stabilization in Streaming Retrieval-Augmented Generation](https://arxiv.org/abs/2606.20113)

**Authors**: Elroy Galbraith  
**Category**: cs.CL  
**Published**: 2026-06-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.20113v1  

#### Abstract
Streaming Retrieval-Augmented Generation (Streaming RAG) reduces user-perceived latency by issuing tool queries in parallel with ongoing user input, before the utterance is complete. Reported gains are aggregate, yet the mechanism's benefit is fundamentally query-intrinsic: speculation can only help...

---

### 20. [Closing the Social-Semantic Gap: SPSD for Edge-Based Prompt Compression in Cloud LLM Inference](https://arxiv.org/abs/2606.19364)

**Authors**: Abhinit Sen, Ajeet Kumar, Manaranjan Pradhan  
**Category**: cs.LG  
**Published**: 2026-06-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.19364v1  

#### Abstract
The prefill stage of Large Language Model (LLM) inference is a growing contributor to cloud-scale energy cost. Many consumer-support and conversational prompts contain social scaffolding: politeness markers, apologetic preamble, repetition, and rapport-building language that is important for human c...

---

### 21. [Federated Bilevel Performative Prediction](https://arxiv.org/abs/2606.19734)

**Authors**: Liangxin Qian, Chang Liu, Xuanyu Cao, Jun Zhao, Kwok-Yan Lam  
**Category**: cs.LG  
**Published**: 2026-06-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.19734v1  

#### Abstract
Federated bilevel optimization is widely used for nested learning problems across distributed clients, such as federated hyperparameter tuning and meta-learning under privacy and communication constraints. Most existing formulations assume fixed client data distributions, which can be violated by pe...

---

### 22. [Physics-Informed Neural Network with Squeeze-Excitation-like Attention](https://arxiv.org/abs/2606.19853)

**Authors**: Yun-Fei Song, Long-Gang Pang, Fu-Peng Li, Jun-Jie Zhang  
**Category**: cs.LG  
**Published**: 2026-06-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.19853v1  

#### Abstract
We introduce SEA-PINN, a novel architecture that incorporates a Squeeze-Excitation-like attention mechanism into physics-informed neural networks to dynamically recalibrate the importance of neurons across layers. A key feature of SEA-PINN is its highly stable initialization. On 17 out of 20 benchma...

---

### 23. [Beyond Entropy: Learning from Token-Level Distributional Deviations for LLM Reasoning](https://arxiv.org/abs/2606.19771)

**Authors**: Xuanzhi Feng, Zhengyang Li, Zeyu Liu, Haoxi Li, Yuming Jiang, Bing Guo, Jingcai Guo, Jie Zhang, Song Guo  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.19771v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced Large Language Model (LLM) reasoning; however, it faces a fundamental optimization instability: uniform token updates precipitate entropy collapse, leading to premature convergence to suboptimal strategies, whereas exce...

---

### 24. [eCNNTO: A Highly Generalizable ConvNet for Accelerating Topology Optimization](https://arxiv.org/abs/2606.19921)

**Authors**: Shengbiao Lu, Xiaodong Wei  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.19921v1  

#### Abstract
This work proposes an element-based Convolutional Neural Network (CNN) to accelerate density-based Topology Optimization (TO), termed eCNNTO. TO generally undergoes a large number of iterations, where finite element analysis is performed in every iteration, leading to the efficiency bottleneck espec...

---

### 25. [Modularity-Free Conflict-Averse Training for Generalized PINNs](https://arxiv.org/abs/2606.20156)

**Authors**: Heejo Kong, Beomchul Park, Sung-Jin Kim, Seong-Whan Lee  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.20156v1  

#### Abstract
Physics-informed neural networks (PINNs) have become a powerful framework for solving PDEs by embedding physical laws into differentiable objectives. Despite their advances, training PINNs remains fragile: recent conflict-averse optimization schemes alleviate gradient interference between residual a...

---

### 26. [Implicit Semantic-Aware Communication Based on Hypergraph Reasoning](https://arxiv.org/abs/2606.20162)

**Authors**: Yiwei Liao, Shurui Tu, Yong Xiao, Yingyu Li, Guangming Shi  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.20162v1  

#### Abstract
Semantic-aware communication has emerged as a transformative paradigm for next-generation communication systems, shifting the fundamental goal from transmitting bit-level symbols to reliably recovering and understanding the semantic meaning of information. Previous studies have demonstrated that rep...

---

### 27. [Granularity-Regulated Adaptive Computational Efficiency for Optimal Verification in Test-Time Scaling](https://arxiv.org/abs/2606.19354)

**Authors**: Ardit Krasniqi, Luan Vejsiu, Elira Dervishi  
**Category**: cs.CL  
**Published**: 2026-06-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.19354v1  

#### Abstract
Test-time scaling (TTS) has emerged as a powerful paradigm for improving the reasoning performance of large language models (LLMs) by investing additional compute at inference time. A central component of TTS is the \emph{verifier}, which selects or scores candidate solutions to guide the search pro...

---

### 28. [MiqraBERT: Regression-Based Sentence-BERT Finetuning for Biblical Hebrew Parallel Detection](https://arxiv.org/abs/2606.19638)

**Authors**: David M. Smiley  
**Category**: cs.CL  
**Published**: 2026-06-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.19638v1  

#### Abstract
Textual reuse pervades the Hebrew Bible, yet the computational methods used to detect it still rest largely on lexical overlap, and they falter once a parallel involves paraphrase, lexical substitution, or syntactic reworking. This paper introduces MiqraBERT, a Sentence-BERT model finetuned from Ale...

---

### 29. [Spectral DPPs via NEPv: A Scalable Continuous Relaxation of Determinantal MAP for Diversity-Aware Data Selection](https://arxiv.org/abs/2606.19411)

**Authors**: Richard Yi Da Xu  
**Category**: cs.LG  
**Published**: 2026-06-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2606.19411v1  

#### Abstract
Selecting a small, diverse, high-quality subset from a massive pool of candidates is a recurring primitive in modern machine learning -- data curation and coreset selection for training and fine-tuning large models, active-learning batch acquisition, prompt and exemplar selection for in-context lear...

---

### 30. [Navigating Unreliable Parametric and Contextual Knowledge: Explicit Knowledge Conflict Resolution for LLM Inference](https://arxiv.org/abs/2606.20245)

**Authors**: Huang Peng, Jiuyang Tang, Weixin Zeng, Hao Xu, Xiang Zhao  
**Category**: cs.AI  
**Published**: 2026-06-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2606.20245v1  

#### Abstract
Large language models (LLMs) have achieved strong performance across a wide range of language-based tasks by leveraging both extensive parametric knowledge and in-context learning ability, enabling them to incorporate external information provided in the input prompt. However, the integration of ext...

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

# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-16 10:34:28 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Coordinated Scheduling for MoE LLM Serving](https://arxiv.org/abs/2606.15177)

**Authors**: Yifan Sun, Zhexiang Zhang, Jiantong Jiang, Gholamreza Haffari, Minxian Xu, Feng Liu, Rajkumar Buyya, Adel N. Toosi  
**Category**: cs.DC  
**Published**: 2026-06-16  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.15177v1  

#### Abstract
Serving Mixture-of-Experts (MoE) large language models (LLMs) is challenging because dynamic request workloads interact with sparse expert routing, creating both data-parallel (DP) engine imbalance and expert-level hotspots. Existing LLM serving systems typically make these decisions in isolation: f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Coordinated Scheduling for MoE LLM Serving**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在 **MoE LLM（Mixture-of-Experts Large Language Models）** 的服务系统中，传统的调度机制存在以下关键问题：

- **前端调度器**（如 vLLM）仅基于粗粒度的请求计数进行 DP（Data Parallelism）引擎选择，忽略了请求间的巨大差异（如 prompt 长度、KV-cache 占用等），导致 **DP 引擎负载不均衡**。
- **后端专家负载均衡器**（如 EPLB）仅依赖聚合的专家激活次数，缺乏对 **来源感知的通信开销** 和 **动态路由模式** 的建模，无法优化跨 DP 组的 all-to-all 通信。
- 前后端调度决策相互隔离，未能形成闭环反馈，导致 **系统整体效率低下**。

这些问题共同引发：
- 数据并行层的引擎压力失衡
- 专家层面的热点（hotspot）
- 跨设备通信开销增加
- 最终表现为高延迟（尤其是 TTFT 和 TPOT）

---

### ✅ 提出的新方法：**Gimbal**

Gimbal 是一个 **协调式的跨层级调度框架**，通过整合前端请求调度与后端专家负载均衡，实现细粒度、源感知的协同优化。

#### 主要创新点：

1. **细粒度 DP 引擎调度（Fine-grained DP-engine Scheduling）**
   - 利用运行时反馈信号（runtime pressure signals）替代请求计数，包括：
     - 剩余 prefill token 数量
     - 等待队列中的 prefill token
     - KV-cache 使用率
     - 后端 MoE 专家压力（expert pressure）
   - 引入轻量级 **SJF-style + Aging 的队列排序策略**，缓解长请求阻塞问题（head-of-line blocking），无需预测输出长度。

2. **源感知的专家负载均衡（Source-aware Expert Load Balancing）**
   - 在线收集 **source-DP-to-expert 路由统计矩阵**（按 layer、source、expert 维度），捕捉请求来源与专家访问之间的关联性。
   - 设计一个受 **MINLP（Mixed-Integer Nonlinear Program）启发的启发式算法**，联合优化：
     - 专家计算负载均衡（`C_load`）
     - 源相关的通信成本最小化（`C_comm`）
     - 专家迁移开销控制（`C_mig`）

3. **跨层级闭环反馈机制**
   - 将后端 MoE 层的压力反馈至前端调度器，避免将新请求分配给已承受高专家负载的 DP 引擎。
   - 形成“**请求调度 ↔ 专家放置**”的正向协同循环。

---

### ✅ 相比现有方法的优势

| 方面 | 传统方法（如 vLLM） | Gimbal |
|------|------------------------|--------|
| 调度粒度 | 请求级（coarse） | Token级 + 多维压力感知 |
| 专家放置依据 | 聚合激活次数（aggregate count） | 源感知路由 + 动态流量结构 |
| 决策耦合 | 前后端独立优化 | 协同闭环反馈 |
| 性能目标 | 局部最优 | 全局延迟与吞吐联合优化 |

> ✅ **优势总结**：Gimbal 不是简单叠加两个优化模块，而是通过**协调设计**实现了 1+1 > 2 的效果。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

- **BurstGPT**：真实世界 LLM 服务 trace，广泛用于系统研究。
- 构造五种典型的请求长度分布以覆盖不同场景：
  - Random
  - Central
  - Descending
  - Two-end
  - Average  
  （见 Figure 7）

### ⚙️ 实验设置

- **硬件平台**：4× NVIDIA H100 SXM5（NVLink 连接），双 CPU
- **部署配置**：`DP=2`, `TP=2`, `EP=4`
- **模型**：Qwen3-30B-A3B（代表性中等规模 MoE 模型）
- **实现基础**：基于 **vLLM** 改造，新增约 1.7K 行 Python/Triton 代码

### 📈 评估指标

| 指标 | 定义 |
|------|------|
| **TTFT**（Time To First Token） | 请求发出到收到第一个 token 的延迟 |
| **TPOT**（Time Per Output Token） | 每个输出 token 的平均解码延迟（不含首 token） |
| **Throughput** | 每秒完成的请求数（RPS） |
| **P99 TTFT** | 第99百分位的 TTFT，反映尾延迟表现 |

### 🔁 对比的基线方法

1. **vLLM**：主流开源推理引擎，默认请求计数调度 + EPLB 专家均衡
2. **MoE-Tuner**：利用离线专家亲和性（expert affinity）进行静态放置
3. **Sem-MoE**：结合语义路由与 ILP 的静态放置方案（本文实现其 oracle 版本）

---

## 3. 主要实验结果和性能指标

### 📉 关键性能提升（vs. vLLM）

| 指标 | 提升幅度 |
|------|----------|
| **平均 TTFT ↓** | **42.9%** 减少 |
| **平均 TPOT ↓** | **33.3%** 减少 |
| **P99 TTFT ↓** | **44.3%** 减少 |
| **高负载吞吐 ↑** | **+3.0%** 提升（RPS=4 时） |

> ✅ 所有 workload 分布下均显著优于所有 baseline，且随着负载上升增益更大。

---

### 🔍 与其他基线对比

| vs. 方法 | TTFT ↓ | TPOT ↓ |
|--------|--------|--------|
| vs. MoE-Tuner | 47.0% | 36.2% |
| vs. Sem-MoE | 34.7% | 29.5% |

> 表明在线、动态、源感知的方法远胜于依赖离线分析或静态假设的方案。

---

### 🔬 消融实验结果（Ablation Study）

比较以下配置（均基于 vLLM 基线）：

| 配置 | 描述 | TTFT ↓ | TPOT ↓ |
|------|------|--------|--------|
| **Gimbal-DP** | 仅启用细粒度 DP 调度 + 队列排序 | 25.1% | 13.4% |
| **Gimbal-EP** | 仅启用源感知专家均衡 | 26.2% | 22.7% |
| **Gimbal-All (No Collab)** | 同时启用两者但无反馈 | 29.8% | 27.3% |
| **Gimbal-All (Full)** | 完整协同设计（含反馈） | **41.4%** | **32.0%** |

> 💡 **关键发现**：
> - 单独优化已有明显收益；
> - **协同设计额外带来 16.5% TTFT 和 6.5% TPOT 的进一步降低**；
> - 证明“**闭环反馈**”是性能跃升的关键。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **MoE 服务中的瓶颈是跨层级耦合的**：
   - DP 层的请求调度直接影响 MoE 层的通信源；
   - MoE 层的专家压力反向影响 DP 引擎的有效容量；
   - 必须打破前后端调度的割裂状态。

2. **细粒度运行时信号至关重要**：
   - 请求数量 ≠ 实际负载；
   - KV-cache 使用、剩余 prefill 工作、MoE 压力等才是更准确的负载指标。

3. **源感知路由统计可有效指导专家放置**：
   - MoE 路由并非随机，存在明显的 source-to-expert 流量偏斜；
   - 将高频访问的专家靠近其主要请求源，能显著减少跨组通信。

4. **协调优于独立优化**：
   - 单独改进 DP 或 EP 模块只能部分缓解问题；
   - **闭环反馈机制**使得系统能够动态适应变化的工作负载。

---

### ⚠️ 方法的局限性

1. **当前评估限于单节点多卡环境**：
   - 多节点场景下的网络拓扑复杂性和带宽限制尚未验证；
   - 跨节点迁移成本更高，可能影响专家重排频率。

2. **专家迁移仍有一定开销**：
   - 平均每次重排耗时 ~0.72 秒（H100 NVLink）；
   - 需谨慎权衡迁移收益与中断代价。

3. **未修改 MoE Router 本身**：
   - 当前方法被动观察路由行为，而非主动引导；
   - 若能结合 routing-aware scheduling，潜力更大。

---

### 🔮 未来工作方向

1. **扩展至多节点集群部署**：
   - 探索跨节点的协调调度与专家放置策略；
   - 结合 RDMA、NIC offloading 等技术降低通信开销。

2. **更自适应的专家放置策略**：
   - 引入强化学习或在线学习机制，动态调整权重参数（α, β, γ）；
   - 支持 bursty、突变型 workload 的快速响应。

3. **与 disaggregated serving 架构融合**：
   - 如与 Splitwise、Mooncake 等 prefill/decode 拆分架构结合；
   - 将 Gimbal 的压力信号作为拆分决策输入。

4. **探索 routing-coordinated scheduling**：
   - 与语义路由（semantic routing）或 prefix-aware routing 联动；
   - 实现从“被动观察”到“主动引导”的演进。

---

> **总结一句话**：  
> Gimbal 通过构建“**前端调度 ↔ 后端专家放置**”的闭环反馈机制，在 MoE LLM 服务中实现了细粒度、源感知、低开销的协同调度，显著降低了延迟并提升了吞吐，为高效 MoE 推理系统的设计提供了新范式。

</details>

---

### 2. [Mixtures of Subspaces for Bandwidth Efficient Context Parallel Training](https://arxiv.org/abs/2606.16384)

**Authors**: Sameera Ramasinghe, Ajanthan Thalaiyasingam, Hadi Mohaghegh Dolatabadi, Gil Avraham, Violetta Shevchenko, Yan Zuo, Chamin Hewa Koneputugodage, Alexander Long  
**Category**: cs.LG  
**Published**: 2026-06-16  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.16384v1  

#### Abstract
Pretraining language models with extended context windows enhances their ability to leverage rich information during generation. Existing methods split input sequences into chunks, broadcast them across multiple devices, and compute attention block by block which incurs significant communication ove...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Mixtures of Subspaces for Bandwidth Efficient Context Parallel Training**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本文针对**去中心化训练环境下的 Context Parallelism（CP）通信瓶颈**问题。在大规模语言模型预训练中，随着上下文长度（context length）增长至数十万甚至百万 token（如 LLaMA 3 达到 130K），传统的 CP 方法需要在每层前向和反向传播时进行全设备间的 Key 和 Value 激活值广播（all-gather 或 ring communication），导致巨大的通信开销。

在**低带宽网络连接**（如 300 Mbps 的互联网链路）下，这种通信成本成为严重瓶颈，使得去中心化训练几乎不可行。而现有方法多集中于 Data Parallelism（DDP）中的梯度压缩，对 CP 场景缺乏有效解决方案。

---

### **提出了什么新方法或新思路**
作者提出了一种**基于子空间混合的激活压缩机制**，用于实现高效的去中心化 Context Parallel 训练：

- **核心洞察**：Transformer 注意力模块输出的 Query、Key、Value（Q/K/V）具有显著的**低秩结构**（low-rank structure）。图1显示，在 LLaMA-70B 中，Q 和 K 的稳定秩（stable rank）仅为满秩的 ~0.1%，V 为 ~0.5%。
  
- **动态子空间混合（Mixture of Subspaces）**：
  - 将注意力权重 $ W $ 分解为 $ W = B U U^\top $，其中 $ U \in \text{St}(d, r) $ 是一个低维正交子空间基，$ B $ 是低秩参数矩阵。
  - 不采用固定子空间，而是通过**联合优化** $ B $ 和 $ U $ 在黎曼流形 $ \mathbb{R}^{d\times d} \times \text{St}(d, r) $ 上进行训练，保证收敛性。

- **高效重参数化（Efficient Reparameterization）**：
  - 引入旋转矩阵 $ R(\theta) \in O(d) $，令 $ U(\theta) = R(\theta) U_0 $，将正交约束转移到可学习的旋转角度 $ \theta $ 上。
  - 使用 **Lie 代数参数化**：$ R(\theta) = \exp\left(\sum_i \theta_i A_i\right) $，其中 $ A_i $ 是固定的斜对称生成元。
  - 采用**二阶泰勒近似**：$ R(\theta) \approx I + \theta A + \frac{1}{2}\theta^2 A^2 $，避免昂贵的矩阵指数计算（从 $ O(d^3) $ 降至 $ O(d^2) $）。

- **按块自适应旋转（Per-chunk Rotation Prediction）**：
  - 每个节点使用一个小的线性头预测当前输入块的最优旋转参数 $ \theta $，提升表达能力而不增加通信负担。
  - 只需传输压缩后的激活 $ Z_{\text{comp}} \in \mathbb{R}^{n_i \times r} $ 和少量标量 $ \theta $（通常 $ k=1 $ 即可）。

- **训练后可移除组件（Unpluggable Components）**：
  - 投影层和旋转头可在训练后期安全移除，恢复为标准 Transformer 架构，兼容现有推理框架。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **通信效率** | 实现 >95% 的通信压缩率（如从 $ O(nd) $ 降至 $ O(nr + k) $），使 300 Mbps 网络可媲美 100 Gbps 数据中心表现 |
| **无性能损失** | 收敛速度与质量与未压缩 CP 相当，甚至略有提升（见 Table 2） |
| **理论保障** | 提供 Riemannian 优化下的线性收敛证明，并保证重参数化不引入虚假极小值（stationary points preserved） |
| **部署友好** | 训练完成后可“拔掉”额外组件，无缝对接标准 Transformer 推理流程 |
| **通用性强** | 可与其他并行策略（如 Pipeline Parallelism）结合使用 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **FineWeb (FW)**：大规模网页文本语料
- **C4 (Colossal Clean Crawled Corpus)**：清洗过的网页文本
- **BookCorpus (BC)**：书籍文本集合

所有数据集均保留 10% 作为验证集。

---

### **实验设置和评估指标**

#### **模型配置**
- 主要测试模型：
  - **8 层，800M 参数**（embedding dim=2048, heads=8）
  - **32 层，3B 参数**（pipeline parallel + CP）
- 上下文长度：**132K tokens**（部分实验达 256K）
- 并行方式：8 或 32 GPU（A100）分布式 CP
- 压缩比例：K 压缩 98%，V 压缩 95%，总体约 **96.5%**

#### **网络模拟**
- **去中心化场景**：300 Mbps（典型互联网带宽）
- **中心化基线**：100 Gbps（数据中心级互联）

#### **评估指标**
- **主指标**：
  - **Wall-clock time 收敛曲线**
  - **Validation Perplexity ↓**
  - **Throughput (Tokens Per Second, TPS) ↑**
- **消融实验**：不同 warm-up 步数、是否使用 reparameterization、是否使用 warm-start 初始化等

---

### **基线方法对比**
由于目前尚无专门针对 CP 的压缩方法，作者构建了两类人工基线进行比较：

1. **Top-K Sparsification**：
   - 仅传输 K/V 中幅值最大的 10% 元素（即 90% 稀疏化）
   
2. **4-bit Quantization**：
   - 对 K/V 激活进行 4-bit 量化（压缩 75%）

此外还对比了长序列模型：
- **BigBird**（稀疏注意力，最大支持 32K）
- **CosFormer**（改进 Softmax 结构）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **通信效率惊人**
- 在 **300 Mbps** 网络上，传统 CP 比 100 Gbps 中心化系统慢 **20 倍以上**。
- 使用本文方法后，**去中心化系统达到与中心化系统几乎相同的 wall-clock 收敛速度**（图2）。

#### ✅ **吞吐量大幅提升**
| 模型 | TPS | 提升倍数 |
|------|-----|--------|
| Centralized CP (100Gbps) | 56K | — |
| Decentralized CP (300Mbps, ours) | **55K** | **×20** vs. uncompressed |

> 注：原始 decentralized CP 在 300Mbps 下仅能实现 ~2.7K TPS，训练完成需 >150 天，实际不可行。

#### ✅ **性能无损甚至更优**
| Model | FW PPL ↓ | C4 PPL ↓ | BC PPL ↓ |
|-------|----------|----------|----------|
| Centralized CP | 17.18 | 17.51 | 17.88 |
| **Ours (Dec. CP Comp.)** | **17.06** | **17.47** | **17.81** |

👉 **不仅没有性能下降，反而略优于中心化基线！**

---

### **与基线方法的对比结果**
- 图5 显示：
  - **Top-K（90% 压缩）** 和 **Quantization（75% 压缩）** 均导致明显收敛延迟和最终性能下降。
  - 而本文方法在 **更高压缩率（96.5%）下仍保持与未压缩 CP 几乎一致的表现**。
- 与 BigBird/CosFormer 相比：
  - 这些模型受限于单卡内存，最多处理 32K 序列。
  - 即便在此限制下，其收敛速度也远逊于本文方法。

---

### **消融实验结果（Ablation Studies）**

#### 🔹 不同设计选择的影响（Table 1 & Table 4）

| 设置 | Perplexity (FW) | TPS |
|------|------------------|-----|
| **Ours (full)** | **22.64** | **55K** |
| + Fixed U | 26.57 | — |
| + Random R(θ) | 24.93 | — |
| - Warm start | 26.63 | — |
| - Reparameterization | — | 37K |
| - 2nd-order approx | — | 30K |

📌 **关键发现**：
- 固定子空间（Fixed U）或随机旋转严重损害性能。
- **Warm-start 初始化（前500步训练主成分）至关重要**。
- 重参数化和二阶近似带来显著 TPS 提升（+49% 和 +83%）。

#### 🔹 Warm-up 步数敏感性分析（Table 3）
| Warm-up Steps | PPL |
|---------------|-----|
| 0 | 26.63 |
| 300 | 22.66 |
| 500 | 22.64 |
| 1000 | 22.87 |

✅ 表明只需 **300–500 步 warm-up** 即可获得稳定高性能，策略轻量且鲁棒。

#### 🔹 插件移除时机影响（Table 5）
| 移除步骤（总10K） | PPL |
|--------------------|-----|
| 不移除 | 22.64 |
| 7k | 22.64 |
| 9.5k | 22.77 |
| 9.9k | 23.12 |

✅ 支持“最后 30% 阶段移除”的经验法则，早期移除虽有扰动但可快速恢复。

---

## **4. 关键结论和发现**

### **主要发现**
1. **注意力激活天然低秩**：Q/K/V 输出普遍存在于极低维子空间中，是压缩的基础前提。
2. **动态子空间优于静态压缩**：联合学习子空间 + 权重分解，避免表达能力受限。
3. **旋转重参数化高效可行**：用 $ \theta $ 参数化 $ O(d) $ 上的旋转，大幅降低优化复杂度，同时保持几何一致性。
4. **超高压缩率可行**：>95% 通信压缩下仍能保持完美收敛，打破带宽壁垒。
5. **兼容性好**：训练后可还原为标准 Transformer，便于部署。

---

### **方法的局限性**
1. **理论解释不足**：
   - 为何极低维 $ \theta $（如 $ k=1 $）就能维持良好性能？尚未有严格理论解释。
   - 可能与隐式正则化、彩票假设（lottery ticket）现象有关，有待深入研究。

2. **依赖 warm-start 初始子空间**：
   - 若初始主成分估计不准，可能影响后续收敛。

3. **未探索其他 reparameterization 形式**：
   - 当前仅使用简单旋转，未来可尝试更复杂的子空间演化模式。

---

### **未来工作方向**
- 探索更灵活的子空间参数化形式（如 hierarchical 或 hierarchical Lie groups）。
- 理论分析低维 $ \theta $ 成功的原因，建立与 implicit bias 的联系。
- 扩展至 MoE、Diffusion Model 等架构中的 CP 场景。
- 实际部署测试：在真实跨地域、异构设备集群中验证稳定性。

---

> 📌 **一句话总结**：  
> 本文首次实现了**高保真、超高压缩的去中心化 Context Parallel 训练**，让普通互联网连接也能支撑百万 token 上下文的大模型训练，逼近数据中心级性能，兼具理论深度与工程实用性。

</details>

---

### 3. [ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training](https://arxiv.org/abs/2606.15682)

**Authors**: Janghwan Lee, Sihwa Lee, Jinseok Kim, Yongjik Kim, Jieun Lim, Jinwook Oh, Jungwook Choi  
**Category**: cs.LG  
**Published**: 2026-06-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.15682v1  

#### Abstract
Large Reasoning Models (LRMs) achieve strong problem-solving through long chain-of-thought, but their deployment is constrained by the high cost of full-precision inference and growing KV cache footprints. Microscaled FP4 formats enable efficient FP4 deployment; however, fully quantizing weights, ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ReQAT: Achieving Full-Precision Reasoning Accuracy with 4-bit Floating-Point Quantization-Aware Training

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型推理模型（**Large Reasoning Models, LRMs**）在数学、逻辑等复杂任务上表现出色，但其部署面临三大效率瓶颈：
- 高精度（如 **BF16**）推理带来的高内存带宽压力；
- 大量浮点运算（**FLOPs**）开销；
- **Key-Value (KV) Cache** 随序列长度线性增长，占用大量显存。

为解决此问题，业界转向 **4-bit 浮点格式（FP4）**（如 **MXFP4**, **NVFP4**），实现 **W4A4KV4**（权重、激活、KV缓存全4位量化）。然而，现有 **PTQ**（Post-Training Quantization）和 **QAT**（Quantization-Aware Training）方法在该配置下导致严重推理准确率下降，尤其在符号化决策（如数字、运算符）上表现脆弱。

### 提出的新方法与思路
本文提出 **ReQAT**，一个面向推理任务的 **FP4 量化感知训练框架**，其核心洞察是：  
> **FP4 推理失败主要集中在低熵（low-entropy）token 上**，即模型本应高度自信的符号化预测（如“3”、“+”），而量化噪声会显著增加非 top-1 token 的采样概率，引发错误级联。

基于此，ReQAT 包含三个核心组件：

1. **Trace-Aligned QAT (TAQ)**  
   - 两阶段训练：先进行 **BF16 FT**，再在**相同的推理路径（reasoning traces）** 上执行 QAT。
   - 优势：确保量化更新反复作用于相同的关键低熵决策点，强化对量化敏感位置的学习。

2. **Selective Entropy Minimization (SEM)**  
   - 引入辅助损失函数，**仅在低熵 token 位置最小化预测熵**。
   - 使用软加权机制（soft weighting），避免对熵值接近阈值的 token 过度惩罚。
   - 目标：增强模型在关键符号预测上的置信度，降低采样错误。

3. **Q-FIT (Quantization-Friendly Initialization via Transformation)**  
   - 在 QAT 前联合校准 **RoPE-consistent KV Cache 变换**：
     - **Pre-RoPE 通道配对缩放（paired scaling）**
     - **Post-RoPE 通道偏移（shifting）**
   - 自适应选择最优组合以稳定不同层的 KV 分布，减少量化误差。

### 相比现有方法的优势
- **首次实现 W4A4KV4 下超越 BF16 FT 的推理准确率**。
- 在相同训练预算下，显著优于现有 **PTQ**、**QAT** 和 **QAD** 方法。
- 支持端到端高效推理，在真实硬件上实现高达 **3.9× 吞吐提升**。

---

## 2. 核心实验方法和设置

### 数据集
- **主训练数据**：`OpenThought-3` 的数学子集（Math Subset）
- **评估基准**：
  - **AIME-120**（2022–2025 年 AIME 试题，共 120 题）—— 主要指标
  - **MATH-500** —— 数学推理补充基准
  - **GSM8K** —— 小规模数学推理测试
  - **LiveCodeBench** —— 代码生成泛化能力测试
- **校准数据**：Wikitext-2（用于 Q-FIT 初始化）

### 实验设置
- **模型**：
  - 主模型：**R1-Qwen-14B**
  - 对比模型：**R1-Llama-8B**
- **量化配置**：
  - **MXFP4 W4A16**（仅权重量化）
  - **MXFP4 W4A4**（权重+激活）
  - **NVFP4 W4A4KV4**（全量化，含 KV Cache）
- **训练设置**：
  - 总微调 token 数：最高达 350M
  - TAQ 阶段固定使用 70M token 的相同推理路径
  - 使用 **AdamW** 优化器，学习率 `1e-5`，余弦调度
  - 所有结果平均 8 个随机种子

### 评估指标
- **准确率（Accuracy）**：AIME-120、MATH-500、GSM8K
- **吞吐量（Throughput）**：输出 token/s（TPS），相对于 BF16 基线的速度提升
- **消融分析**：分别验证 TAQ、SEM、Q-FIT 的贡献

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **PTQ** | AWQ (W4A16), QuaRot / FlatQuant (W4A4KV4) |
| **FT+PTQ** | 先 BF16 微调，再 PTQ |
| **QAT** | 标准量化感知训练 |
| **QAD** | Quantization-Aware Distillation |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（R1-Qwen-14B, NVFP4 W4A4KV4）

| 方法 | AIME-120 准确率 |
|------|----------------|
| BF16 Baseline | 56.83% |
| BF16 FT | 65.46% |
| PTQ | 50.13% |
| QAT | 58.86% |
| **ReQAT (Full)** | **65.94%** ✅ |

> **ReQAT 不仅恢复，且超越 BF16 FT 准确率（+0.48%）**

### 与其他方法对比（AIME-120）

| 方法 | AIME-120 准确率 |
|------|----------------|
| AWQ (INT4) | 53.02% |
| FT+AWQ (MXFP4) | 60.11% |
| QAD (MXFP4) | 54.29% |
| QAT (MXFP4) | 62.29% |
| **ReQAT (MXFP4)** | **68.02%** ✅ |
| **ReQAT (NVFP4)** | **65.63%** ✅ |

> ReQAT 在所有量化配置下均显著领先。

### 吞吐性能（真实硬件）

| 平台 | 配置 | 相对于 BF16 的吞吐加速 |
|------|------|------------------------|
| **NVIDIA DGX Spark** | NVFP4 W4A4KV4 | **3.9×** |
| **NVIDIA B200** | NVFP4 W4A4KV4 | **3.1×** |

> 加速来自两方面：
> 1. 权重与 KV Cache 显存占用减少 → 更大 batch size
> 2. 4-bit GEMM 计算效率提升

### 消融实验结果

#### (1) ReQAT 组件消融（R1-Qwen-14B, NVFP4 W4A4KV4）

| 方法 | AIME-120 |
|------|---------|
| ReQAT (TAQ) | 63.13% |
| ReQAT (TAQ + Q-FIT) | 65.94% ✅ |
| ReQAT (TAQ + Q-FIT + SEM) | 65.63% |

> Q-FIT 贡献最大增益（+2.81%），表明 KV Cache 校准至关重要。

#### (2) Trace Alignment 消融

| 方法 | AIME-120 (280M tokens) |
|------|------------------------|
| QAT（无对齐） | 61.09% |
| FT+QAT（不同路径） | 62.19% |
| **FT+QAT（相同路径，即 TAQ）** | **65.00%** ✅ |

> 路径对齐带来约 **3%** 的显著提升。

#### (3) SEM 正则化强度（λ）影响（R1-Llama-8B）

| λ | AIME-120 |
|----|---------|
| 0.0 | 40.32% |
| 0.03 | 41.46% |
| **0.1** | **41.85%** ✅ |
| 0.5 | 41.88% |

> SEM 效果稳健，中等强度（λ=0.1）即可取得最佳平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **低熵 token 是 FP4 推理失败的关键瓶颈**：量化噪声虽小，但在确定性符号预测上引发采样错误，导致推理链崩溃。
2. **路径对齐（Trace Alignment）能有效聚焦学习信号**：重复在相同推理路径上训练，使 QAT 能精准修复量化敏感位置。
3. **KV Cache 量化需联合变换校准**：单一的缩放或偏移策略不足，Q-FIT 的自适应联合校准显著提升稳定性。
4. **ReQAT 实现了效率与性能的双赢**：在不增加训练成本的前提下，达到甚至超越 BF16 FT 的准确率，并支持高达 3.9× 的端到端吞吐加速。

### 方法的局限性
- **依赖高质量监督信号**：TAQ 基于 SFT，若原始推理路径本身存在噪声或错误，可能限制性能上限。
- **当前设计针对数学推理优化**：虽然在代码生成上也有效，但未专门适配其他领域（如自然语言对话）。
- **Q-FIT 需额外校准步骤**：尽管轻量，但仍引入少量预处理开销。

### 未来工作方向
- 将 TAQ 机制扩展至 **知识蒸馏（Knowledge Distillation）** 或 **强化学习（RL）** 框架。
- 探索 **动态熵阈值调整** 或 **token-level 自适应量化策略**。
- 研究 ReQAT 在 **多模态推理** 或 **长上下文检索增强** 场景下的适用性。

---

> **总结**：ReQAT 通过“**聚焦低熵决策 + 路径对齐训练 + KV 缓存联合校准**”的系统性设计，成功解决了 W4A4KV4 量化下的推理退化难题，为高性能、低成本 LRM 部署提供了实用化路径。

</details>

---

### 4. [Diffusion Offline Reinforcement Learning for Fair and Energy-Efficient UAV-Assisted Wireless Networks](https://arxiv.org/abs/2606.16331)

**Authors**: Eslam Eldeeb, Hirley Alves  
**Category**: cs.LG  
**Published**: 2026-06-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.16331v1  

#### Abstract
The integration of generative artificial intelligence with wireless communication and signal processing systems has opened new avenues for intelligent, data-driven decision-making in future 6G networks. This work proposes a diffusion soft actor-critic (Diffusion-SAC) approach that leverages offline ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Diffusion Offline Reinforcement Learning for Fair and Energy-Efficient UAV-Assisted Wireless Networks

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**UAV-assisted wireless networks**中的轨迹规划与通信调度联合优化问题，旨在在**离线环境**下实现公平性（fairness）和能效（energy efficiency）的平衡。传统 **offline RL** 方法（如 CQL、IQL）虽然能在无在线交互的情况下学习策略，但在低数据量或动态环境中泛化能力差，且难以建模复杂的混合动作空间（hybrid discrete-continuous action space）。此外，静态数据集中可能包含次优行为，导致策略无法超越行为策略（behavior policy）。

### 提出了什么新方法或新思路
作者提出了一种名为 **Diffusion-SAC** 的新型 **offline RL** 框架，其核心创新在于：
- **将 Denoising Diffusion Probabilistic Models (DDPMs)** 引入到 offline RL 的策略建模中，作为 **soft actor-critic (SAC)** 框架中的 **actor**。
- 利用扩散模型强大的生成能力对复杂、高维的动作分布进行建模，尤其适用于 UAV 控制中的连续移动（如位移）和离散调度（如选择服务设备）的混合决策。
- 在训练过程中结合 **behavior cloning (BC) loss** 和 **Q-guidance loss**，通过超参数 $ \eta $ 平衡模仿学习与探索，避免过度保守或过拟合。
- 发现扩散模型本身可作为**隐式的分布正则化器**（implicit distributional regularizer），在某些配置下甚至可以**省去 CQL 的显式保守惩罚项**，从而简化训练并提升性能。

### 相比现有方法的优势
- **更强的泛化能力**：在小样本或质量不均的数据集上表现更鲁棒。
- **更高的表达能力**：能够捕捉复杂、结构化的动作分布，优于传统神经网络参数化的策略。
- **更高效的数据利用**：相比标准 offline RL 方法，在有限数据下仍能收敛至高性能策略。
- **无需显式保守约束**：在合理设计下，可通过扩散机制自然抑制 OOD（out-of-distribution）动作，减少对 CQL 正则项的依赖。

---

## 2. 核心实验方法和设置

### 数据集
- **未使用公开真实数据集**，而是通过模拟环境构建离线数据集。
- 数据来源于一个预训练的 **online SAC agent** 在环境中交互产生的经验回放缓冲区（replay buffer）。
- 最终离线数据集包含约 **30k 个状态-动作-奖励元组**（transition tuples），涵盖早期探索阶段（低质量）和后期近最优策略（高质量）的行为，确保多样性。

### 实验设置和评估指标

#### 环境设置
- 场景：1000×1000 m² 区域内，**10 个固定地面 IoT 设备**由一架飞行高度为 100 m 的 UAV 服务。
- UAV 动作空间为混合型：
  - 连续动作：二维移动位移 $ (w_x, w_y) $
  - 离散动作：选择服务的设备 $ s \in \{0,1,\dots,10\} $（0 表示空闲）
- 状态空间包含：UAV 位置、各设备的 **Age of Information (AoI)**、累计能耗。
- 奖励函数设计为加权组合：$ r_t = -\left(\lambda \cdot \text{average AoI} + (1-\lambda) \cdot \text{energy consumption}\right) $

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Normalized Return** | 累计折扣奖励，衡量整体策略性能 |
| **Average AoI** | 信息新鲜度，越低越好 |
| **Total Energy Consumption** | 能耗总量，越低越好 |
| **Throughput** | 单episode内传输总数据量（GB） |
| **Transmission Time per Packet** | 单个数据包平均传输时间（秒） |
| **Inference Time** | 决策延迟，影响实时性 |

### 基线方法对比
- **Offline RL Baselines**：
  - **CQL**（Conservative Q-learning）
  - **IQL**（Implicit Q-learning）
  - **BCQ**（Batch-Constrained Deep Q-learning）
- **Heuristic Baselines**：
  - **TDM**（Time-division multiplexing）：轮询调度
  - **Saver**：优先服务最近设备以节能
  - **Random Walk (RW)**：随机动作

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | Throughput (GB/episode) | Energy Consumption (J) | Avg. AoI | Transmission Time (s) |
|------|--------------------------|-------------------------|----------|------------------------|
| **Diffusion-SAC** ($\alpha=0, \eta=0.5$) | **>0.65** | **最低** | **最低** | **~0.15** |
| CQL / IQL | ~0.48 | 较高 | 较高 | ~0.22 |
| BCQ | ~0.45 | 高 | 高 | ~0.25 |

> ✅ **吞吐量提升超过 35%**

### 与基线方法的对比结果
- **在所有核心指标上全面超越 baseline**：
  - **Return 更高**：Fig. 7a 显示 Diffusion-SAC 收敛更快且最终回报最高。
  - **AoI 更低**：Fig. 7b 表明其能更及时更新设备信息，提升服务质量。
  - **能耗更低**：Fig. 7c 显示其在保证通信的同时显著降低能量消耗。
- 在小数据场景下优势更加明显：
  - 当数据量从 30k 减少到 10k 时，**Diffusion-SAC 性能几乎不变**，而 CQL/IQL/BCQ 显著下降（Fig. 8a）。
- 对数据质量鲁棒性强：
  - 在“高质量”数据上，Diffusion-SAC 和 BCQ 表现好；
  - 在“低质量”（含大量随机行为）数据上，**CQL 因保守而受限，BCQ 失败，而 Diffusion-SAC 仍保持稳定性能**（Fig. 8b–c）。

### 消融实验结果
- **不同 $\alpha$（CQL loss 权重）的影响**：
  - $\alpha = 0$（无 CQL 正则） + $\eta = 0.5$ 效果最好 → 说明**扩散模型自身具备正则化能力**。
  - $\alpha = 1$ 导致策略过于保守，性能下降。
- **不同 $\eta$（BC vs Q-guidance 权重）的影响**：
  - $\eta = 1$（仅 BC）：退化为纯模仿学习，无法超越数据。
  - $\eta = 0$（仅 Q-guidance）：虽可学习，但稳定性较差。
  - $\eta = 0.5$：最佳平衡点，兼顾泛化与探索。
- **去噪步数（denoising steps）的影响**：
  - 少于 10 步：性能不佳；
  - 超过 20 步：接近最优性能（Fig. 10a）；
  - 推理时间随步数线性增长（Fig. 10b），但 **12ms 延迟在实际系统中可接受**。

---

## 4. 关键结论和发现

### 主要发现
1. **Diffusion models 可有效用于 offline RL 的策略建模**，尤其适合混合动作空间下的无线控制任务。
2. 扩散模型不仅提供强大表达力，还能**隐式地约束 OOD 动作**，在某些情况下可替代 CQL 的显式保守项。
3. **Diffusion-SAC 在小样本、低质量数据下表现出极强鲁棒性**，适合现实世界中难以获取大规模高质量数据的场景。
4. 该方法在 **UAV trajectory and scheduling** 任务中实现了 **AoI、能耗、吞吐量的多目标优化**，综合性能显著优于现有 offline RL 方法。

### 方法的局限性
- **推理延迟较高**：由于需执行多步去噪采样，**inference time 达 12ms**，高于传统方法（~2ms），限制其在超高实时性场景的应用。
- **计算开销大**：训练过程涉及多个网络（actor、critic、target networks）及扩散过程，资源需求高。
- **对去噪步数敏感**：若步数太少，性能下降；需足够步数才能发挥优势。
- **尚未扩展至 multi-agent 场景**，当前为单 UAV 设置。

### 未来工作方向
- **降低推理延迟**：采用 **DDIM**（Denoising Diffusion Implicit Models）等加速技术减少去噪步数。
- **扩展至 multi-agent 和 distributed RL** 框架，支持多 UAV 协同。
- 探索 **diffusion-based meta-RL** 或 **few-shot adaptation**，进一步提升跨环境泛化能力。
- 将该框架应用于其他 **6G wireless control tasks**，如 RIS 配置、RAN slicing、beamforming 等。

--- 

> 📌 **总结一句话**：  
> 本文提出的 **Diffusion-SAC** 成功将生成式 AI 中的 **diffusion models** 与 **offline RL** 结合，解决了 UAV 无线网络中轨迹与调度联合优化的难题，在**公平性、能效、吞吐量**等方面全面超越现有方法，展示了生成模型在智能无线控制中的巨大潜力。

</details>

---

### 5. [SpecAlign: Efficient Specification-Grounded Alignment of Large Language Models via Synthetic Data](https://arxiv.org/abs/2606.16276)

**Authors**: Wenjie Wang, Yue Huang, Zhengqing Yuan, Han Bao, Shiyi Du, Yuchen Ma, Yue Zhao, Yanfang Ye, Xiangliang Zhang  
**Category**: cs.AI  
**Published**: 2026-06-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.16276v1  

#### Abstract
As large language models (LLMs) are increasingly deployed in real-world applications, alignment is no longer governed by a single universal notion of safety or helpfulness, but instead by provider- or application-specific model specifications. These specifications are typically long, structured, and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《SPECALIGN: Efficient Specification-Grounded Alignment of Large Language Models via Synthetic Data》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 **LLM Alignment** 方法大多基于抽象原则（如 Constitutional AI）或静态安全基准，难以适应现实世界中由模型提供方（如 OpenAI、Anthropic）发布的**长篇、结构化且频繁更新的 Model Specification**。这些规范通常包含复杂的规则交互、优先级冲突和上下文依赖，而现有方法缺乏系统机制将这些文档转化为有效的训练信号。

此外，手动构建对齐数据成本高昂，且无法快速响应规范的迭代更新，导致对齐过程昂贵且不可持续。

### **提出的新方法与新思路**
论文提出了 **Specification-Grounded Alignment** 这一新范式，并实现了框架 **SPECALIGN**，其核心创新如下：

- **以 Model Specification 为第一类对齐目标**：将 provider 编写的规范文档作为对齐的直接依据，而非抽象原则或通用基准。
- **自动化合成对齐数据**：通过多阶段流程从原始规范中自动生成高质量的偏好对（preference pairs），实现“规范即训练信号”。
- **三阶段合成框架**：
  1. **Rule Annotation**：为每条规则添加结构化标签（Direction, Stage, Domain, Family）和优先级。
  2. **Specification Generation**：在结构约束下采样规则子集，生成多样化的具体规范实例。
  3. **Multi-Agent Adversarial Data Synthesis**：利用 Planner、Attacker、Defender 多智能体对抗交互，生成边界感知的合规与违规响应对。

### **相比现有方法的优势**
| 维度 | 现有方法（如 Constitutional AI, SPIN） | SPECALIGN |
|------|----------------------------------------|---------|
| **输入依据** | 抽象原则或短规则集 | 长篇、结构化 Model Spec 文档 |
| **可控性** | 弱，难以精确控制规则组合 | 强，支持细粒度规则采样与优先级排序 |
| **数据多样性** | 易重复、表面化 | 高，通过经验池（Experience Pool）和角色切换避免策略固化 |
| **边界探索能力** | 有限 | 强，通过多智能体对抗主动探测边缘案例 |
| **可扩展性** | 低，依赖人工标注 | 高，完全自动化，支持快速迭代 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **基础训练数据**：
  - **Alpaca dataset**（约 4k 条）用于 Supervised Fine-Tuning（SFT），仅含良性指令数据。
- **SPECALIGN 合成数据**：
  - 所有偏好对均由 SPECALIGN 框架基于 **OpenAI Model Spec (2025)** 自动生成。
  - 构建了 **10 个不同的 Model Specs**，每个对应独立的训练-评估设置。
- **评估数据集**：
  - 使用异构模型池（如 GPT-4o-mini, GLM-4.5-Air, Grok-4.1）生成评估样本，避免评估偏见。

### **实验设置**
- **模型选择**：
  - Llama-3.1-8B-Instruct
  - Qwen3-8B
  - GPT-oss-20B
- **训练协议**：
  - 两阶段训练：先 SFT（Alpaca），再联合 SFT-DPO 训练。
  - SFT:DPO 数据比例固定为 **1:3**（经消融实验验证最优）。
- **评估维度与指标**：
  | 类别 | 指标 | 说明 |
  |------|------|------|
  | **规范遵从性** | **RCS (Rule Compliance Score)** | 主要指标，越高越好 |
  | **安全性鲁棒性** | Beaver-Unsafe ↓, FalseReject ↑, XSTest-ORR ↓ | 衡量安全与拒绝行为 |
  | **通用能力保留** | IFEval ↑, MT-Bench ↑, SimpleQA ↑ | 指令遵循、对话质量、事实问答 |

### **基线方法对比**
分为两类进行比较：
1. **优化算法变体**：
   - **GRPO**, **RLOO**：使用相同 SPECALIGN 数据，仅更换优化器。
2. **自博弈/多智能体构造方法**：
   - **SPIN**, **DebateGPT**：使用相同初始攻击提示池，公平比较数据构造能力。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2）**
在 **10 个不同 Model Specs** 和 **3 个主干模型** 上平均表现：

| 模型 | Base RCS | SPECALIGN RCS | 绝对提升 | 相对提升 |
|------|----------|----------------|-----------|------------|
| Llama-3.1-8B-Instruct | 0.73 ± 0.09 | **0.80 ± 0.07** | +0.07 | +9.6% |
| Qwen3-8B | 0.89 ± 0.08 | **0.93 ± 0.07** | +0.04 | +4.5% |
| GPT-oss-20B | 0.83 ± 0.07 | **0.92 ± 0.07** | +0.09 | +10.8% |

> ✅ **所有设置下均显著提升，无例外**，表明方法具有强泛化性。

### **与基线方法对比（Figure 5）**
- **SPECALIGN 在 RCS 上全面优于所有基线**：
  - 超过 GRPO/RLOO → 说明优势不仅来自优化器。
  - 超过 SPIN/DebateGPT → 说明其数据构造更精准捕捉规则边界与冲突。
- **结论**：SPECALIGN 的优势源于其 **规范感知的数据构造机制**，而非单纯优化技巧或多智能体架构。

### **消融实验结果（Figure 7b）**
移除以下组件均导致性能下降：
- **Multi-Agent Framework**：行为多样性降低。
- **Experience Pool**：学习不稳定，难以积累有效攻击模式。
- **Role Switching**：推理能力退化，易陷入固定策略。

> 🔍 三者协同作用，共同提升边界敏感性和训练稳定性。

### **其他重要发现**
- **安全性与通用能力平衡良好**（Table 3）：
  - Beaver-Unsafe 下降或持平。
  - FalseReject 显著上升（减少过度拒绝）。
  - XSTest-ORR 多数下降 → 表明非靠“全拒”策略提升合规。
  - IFEval、MT-Bench 普遍提升 → 通用能力未受损甚至增强。
- **计算成本极低**：
  - 单条数据平均成本仅 **$0.0382**（基于 OpenRouter 定价）。
  - Token 消耗可控，适合大规模部署。

---

## **4. 关键结论和发现**

### **主要结论**
1. **Specification-Grounded Alignment 是可行且高效的范式**：
   - 将 Model Spec 直接作为训练信号，能实现**快速、精确、可扩展**的行为适配。
2. **SPECALIGN 显著提升规则遵从性**：
   - 在多个模型和规范上一致提升 RCS，同时**保持甚至增强通用能力**。
3. **避免过度保守行为**：
   - 不依赖“全拒”策略，而是通过**精细化边界学习**实现合规。
4. **多智能体对抗 + 经验池 + 角色切换** 是生成高质量边界数据的关键机制。

### **局限性**
1. **依赖规范质量**：
   - 若原始 Model Spec 存在不完整或内部矛盾，对齐效果受限。
2. **采用分治策略（divide-and-conquer）**：
   - 仅采样 8–13 条规则的子集，近似而非完全覆盖完整规范。
   - 未来可探索 Curriculum Learning 或层次化建模来桥接差距。
3. **评估指标局限**：
   - 当前依赖静态基准，可能无法捕捉长期行为漂移或开放交互中的风险。

### **未来工作方向**
- 探索 **Curriculum Learning** 以逐步学习复杂、层级化的规范。
- 开发面向 **真实场景长期部署** 的动态评估机制。
- 引入 **人类反馈闭环** 以验证规范的社会合理性。
- 支持 **跨语言、跨文化规范适配**，提升全球化适用性。

---

> 📌 **总结一句话**：  
> **SPECALIGN 成功将长篇、结构化的 Model Specification 转化为高效、精准的对齐训练信号，实现了 LLM 行为的快速、低成本、可解释的政策适配，为下一代可信赖 AI 系统提供了实用路径。**

</details>

---

### 6. [Progressive Knowledge-Guided Large Language Model Framework for Bearing Fault Diagnosis](https://arxiv.org/abs/2606.16684)

**Authors**: Jinghan Wang, Gaoliang Peng, Yanjun Chen, Wei Zhang, Wentao Wu, Tianchen Liu  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.16684v1  

#### Abstract
Vibration-based bearing fault diagnosis requires resolving three interrelated measurement challenges, including the trade-off between global statistical feature efficiency and local transient signal fidelity, insufficient traceability of measurement features to underlying fault physics, and ineffect...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Progressive Knowledge-Guided Large Language Model Framework for Bearing Fault Diagnosis

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**工业部署中轴承故障诊断面临的三大挑战**：
1. **Scale Conflict（尺度冲突）**：全局统计特征高效但丢失局部瞬态细节，而原始信号处理保留时序保真度却计算成本高昂，难以实时应用。
2. **Knowledge Gap（知识鸿沟）**：纯数据驱动模型（如深度学习、LLM）缺乏对轴承动力学和故障机理的显式编码，导致诊断结果不可解释，无法满足安全关键系统的审计要求。
3. **Multi-scale Integration Insufficiency（多尺度融合不足）**：现有方法各阶段孤立进行，缺乏跨阶段的信息流动，限制了诊断准确性和可解释性。

### 提出的新方法与创新思路
提出了一种**三阶段渐进式物理引导的大语言模型框架（Progressive Physics-Guided LLM Framework）**，将物理先验知识系统地嵌入到整个诊断流程中：

#### 主要贡献：
1. **构建了一个81维的知识增强型测量描述符（Knowledge-Enhanced 81-Dimensional Measurement Descriptor）**
   - 基于**轴承运动学理论**和**特征缺陷频率**（如BPFI、BPFO、BSF），从时域、频域和小波域提取物理可追溯的特征。
   - 特征设计具有**一对一映射关系**，确保每个特征都能回溯至具体的故障源周期性，提升**measurement interpretability**。

2. **引入故障自适应信号分段机制（Fault-Adaptive Signal Segmentation）**
   - 利用Stage 1输出的**故障先验概率 $ p_1 $** 动态调制后续分析注意力。
   - 不同故障类型对应不同的注意力权重模式（如内圈故障集中在中间片段，外圈受载荷区调制呈正弦分布），实现无需人工干预的智能聚焦。

3. **设计了渐进式三阶段诊断流水线（Three-Stage Progressive Pipeline）**
   - **Stage 1**: 全局特征诊断 → 快速筛查（~20ms/sample）
   - **Stage 2**: 局部片段诊断 → 细粒度建模
   - **Stage 3**: 多模态融合 → 最终决策
   - 所有知识在训练时通过LoRA注入，在推理时完全**去依赖外部知识库（KG-free inference）**，支持现场自主运行。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **效率 vs 精度平衡** | 要么牺牲精度（仅用特征），要么牺牲效率（端到端信号处理） | 在保持高精度的同时实现低延迟（Stage 1仅20ms） |
| **可解释性** | 黑箱模型，难审计；或KG仅用于事后注释 | 特征激活与故障力学一致，支持traceability |
| **部署可行性** | 依赖网络访问KG服务，不适用于边缘场景 | 推理无外部依赖，适合工业边缘部署 |
| **泛化能力** | 对工况变化敏感 | 在多种变速、混合负载条件下表现稳健 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个公开基准数据集上验证，覆盖多样化操作条件：

| 数据集 | 类别数 | 采样率 | 主要挑战 |
|-------|--------|--------|----------|
| **CWRU** | 4类（Normal, IR, OR, Ball） | 12 kHz | 内圈与滚动体故障频谱相似 |
| **JNU** | 4类 | 50 kHz | 转速变化（600–1000 rpm），测试跨速度泛化 |
| **PU** | 3类 | 64 kHz | 多种轴承类型、不同径向载荷、复杂工况 |
| **MFPT** | 3类 | 48.8 / 97.6 kHz | 严重类别不平衡 + 混合采样率 |

### 实验设置
- **模型架构**：
  - **Stage 1**: ChatGLM-6B + LoRA（r=8），输入为81维特征文本化描述
  - **Stage 2**: GPT-2-12layer + LoRA，处理8个重叠片段（每段256点）
  - **Stage 3**: GPT-2-6layer + LoRA，融合Feature Token、Patch Token 和隐式KG Token
- **训练配置**：
  - Batch Size: 32（Stage 1&2）、8（PU因样本少需更细优化）
  - Optimizer: AdamW，余弦退火调度
  - Epochs: Stage 1&2训练30轮，Fusion阶段50轮以充分收敛
- **评估方式**：5折交叉验证，报告平均值±标准差

### 评估指标
- **主指标**：Overall Accuracy
- **细粒度指标**：Per-class Precision, Recall, F1-Score
- **辅助分析工具**：
  - Confusion Matrix（混淆矩阵）
  - Token Contribution Analysis（基于梯度归因）
  - Cross-Modal Attention Visualization（跨模态注意力热图）

### 基线方法对比
文中虽未列出具体命名基线，但从上下文可知其对比对象包括：
- **传统方法**：FFT+SVM、Wavelet+RF 等手工特征+浅层分类器
- **深度学习方法**：CNN（处理TFR图像）、RNN/LSTM、Transformer等端到端信号建模
- **KG增强方法**：依赖运行时查询外部知识图谱的方法
- **LLM-based方法**：如BearLLM、FD-LLM等信号转语言范式

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 总体准确率（Table 3）
| Dataset | Stage 1 | Stage 2 | **Fusion** | Δ(1→2) | Δ(2→F) |
|--------|--------|--------|-----------|--------|--------|
| CWRU   | 95.59% | 97.61% | **98.19%** | +2.02% | +0.58% |
| JNU    | 89.69% | 99.77% | **99.27%** | +10.08%| -0.50% |
| PU     | 92.98% | 95.05% | **97.06%** | +2.07% | +2.01% |
| MFPT   | 99.07% | 99.15% | **99.45%** | +0.08% | +0.30% |
| **Avg** | **94.33%** | **97.90%** | **98.49%** | **+3.57%** | **+0.59%** |

> ✅ **最终平均准确率达到98.49%**，相比Stage 1提升4.16个百分点。

#### 每类性能（Fusion阶段，Table 4）
所有数据集加权平均F1-score均超过 **0.970**：
- CWRU: 0.982
- JNU: 0.993
- PU: 0.971
- MFPT: 0.994  
表明方法在各类别上均有稳定表现，尤其在MFPT严重不平衡下仍保持极高性能。

### 与基线方法的对比结果
- **计算效率方面**：
  - Fusion阶段训练时间仅为Stage 1的 **1/12.6**（Fig. 6）
  - Stage 1单样本推理延迟约 **20ms**，满足在线监控需求
  - 相比“signal-level baselines”实现 **12.6倍计算成本降低**

- **性能优势体现**：
  - 在JNU上Stage 2增益达 **+10.08%**，说明知识引导注意力有效应对转速漂移
  - 在PU上Fusion带来最大提升 **+2.01%**，体现多模态融合在异构工况下的价值
  - MFPT接近饱和（>99%），显示框架对采样率差异和样本稀缺鲁棒

### 消融实验与归因分析（Interpretability Analysis）

#### Token贡献分析（Fig. 7）
- **Patch Tokens主导决策**：对外圈和滚动体故障贡献高达71.54%~79.89%，符合其脉冲特性
- **Feature Tokens重要性上升**：外圈故障依赖更高（24.65%），因其能量分布更广
- **KG Tokens非均匀参与**（3.8%~7.5%）：在模糊情况下提供更多先验支持，而非固定加权

#### 跨模态注意力可视化（Fig. 8）
- **KG → Patch 注意力强**（0.106~0.118）：知识主动验证局部波形形态
- **KG → Feature 中等强度**（0.02~0.107）：静态统计特征需结合领域知识
- **Patch → Feature 上下文反馈**（0.156~0.291）：局部片段寻求全局统计背景
- 表明模型实现了**物理一致性的多模态推理机制**

---

## 4. 关键结论和发现

### 主要发现
1. **物理先验可以系统性嵌入LLM框架并显著提升性能与可解释性**  
   将轴承动力学知识转化为特征空间结构和注意力机制，使模型不仅“猜得准”，而且“说得清”。

2. **渐进式架构优于单一端到端模型**  
   分阶段处理兼顾效率与精度，且中间输出可用于人机协同审核，符合工业实践需求。

3. **知识可以在训练中内化，推理时完全脱离外部依赖**  
   通过将KG信息编码进LoRA参数，解决了KG增强方法在边缘部署中的基础设施瓶颈。

4. **多模态融合应是动态、选择性的，而非简单拼接**  
   实验显示不同故障类型激活不同信息流组合，证明了自适应融合的有效性。

### 方法的局限性
- 当前验证局限于**单故障类型**，未考虑复合故障或多部件耦合场景
- 框架仍依赖预定义的81维特征工程，尽管已高度自动化，但仍有一定领域门槛
- 所有实验为**within-dataset evaluation**，尚未测试跨数据集迁移能力
- 模型规模较大（Stage 1使用6B参数LLM），虽经LoRA压缩，但在极低端设备部署仍有挑战

### 未来工作方向
1. **跨数据集迁移学习与零样本诊断能力研究**
2. **扩展至复合故障识别与故障程度估计（severity monitoring）**
3. **开发轻量化变体（lightweight variants）用于边缘设备部署**
4. **探索自适应融合机制以更好处理未知工况与新型故障模式**

--- 

> 🔚 **总结一句话**：  
本论文成功将**物理机理**、**多尺度信号处理**与**大语言模型的强大推理能力**有机结合，提出一个既高效、准确又可解释、可部署的轴承故障诊断新范式，为工业AI迈向可信智能运维提供了重要路径。

</details>

---

### 7. [Solyx AI Grid: Hardware-Telemetry-Aware Routing Across Geographically Distributed GPU Clusters](https://arxiv.org/abs/2606.15050)

**Authors**: Aleks Bernhard, Nithin Katla  
**Category**: cs.DC  
**Published**: 2026-06-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.15050v1  

#### Abstract
As GPU capacity fragments across geographically distributed sites, single-cluster LLM inference routing assumptions break down in measurable ways. We present Solyx AI Grid, a cross-site inference routing control plane that integrates GPU hardware telemetry (DCGM), vLLM application metrics, and real-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Solyx AI Grid: Hardware-Telemetry-Aware Routing Across Geographically Distributed GPU Clusters**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
随着GPU部署日益分散在**地理上分布的多个站点**，传统的单集群LLM推理路由假设已不再成立。现有系统存在以下三大典型失败模式：
- **Capability Mismatch Leakage**：副本通过`/health`检查但仍因`max_model_len`、dtype不匹配等问题导致部分请求失败。
- **Tail Latency Amplification Under Jitter**：跨广域网（WAN）路径的RTT和jitter显著影响TTFT（Time to First Token），而传统负载均衡器对此无感知。
- **Cold-Start Cascade During Failover**：故障转移时，新启动的副本尚未加载完成就被分配流量，引发级联失败。

现有方法如 **NVIDIA Dynamo**、**vLLM Router** 仅限于单数据中心；**SkyWalker** 虽支持跨区域但假设副本同质且无硬件遥测；**Helix** 是离线优化器，无法实现实时逐请求决策。

### **提出了什么新方法或新思路**
提出 **Solyx AI Grid** —— 一种**跨站点推理路由控制平面**，其核心创新在于：
- 构建了一个融合 **10个信号** 的加权压力评分器（weighted pressure scorer），用于实时逐请求的placement决策。
- 首次将 **DCGM硬件遥测**、**vLLM应用层指标** 和 **主动探测的WAN网络信号（RTT、jitter）** 统一整合到生产级路由系统中。
- 设计了基于心跳的**生命周期状态机**（Loading, Ready, Draining, Failed），实现快速故障检测与优雅降级。

### **相比现有方法的优势**
| 特性 | Solyx AI Grid | Round-Robin | Least-Request | SkyWalker |
|------|---------------|-------------|----------------|-----------|
| 跨站点支持 | ✅ | ❌ | ❌ | ✅ |
| 硬件异构性支持 | ✅ | ❌ | ⚠️（依赖错误反馈） | ❌（假设同质） |
| 实时硬件遥测（DCGM） | ✅ | ❌ | ❌ | ❌ |
| 网络质量感知（RTT/jitter） | ✅ | ❌ | ❌ | ❌ |
| 错误率驱动去权重 | ✅ | ❌ | ❌ | ❌ |
| 快速failover（<1.5s p99） | ✅ | ❌ | ⚠️ | 未验证 |

---

## **2. 核心实验方法和设置**

### **实验设置**
进行了两个独立的实证研究（campaign）：

#### **Campaign 1 (Apr 2026)**  
- **硬件配置**：6块GPU分布在3个美国数据中心（H100/H200 SXM）
- **测试场景**：
  - Homogeneous：全为H100
  - Hardware-Heterogeneous：混合H200与受限H100（`max_model_len=1024`）
  - Capability-Mismatch：故意制造能力不匹配
- **模型**：Llama 3.1 70B AWQ INT4
- **路由策略对比**：Round-Robin (RR), Least-Request (LR), Solyx AI Grid（v2, 8-signal）

#### **Campaign 2 (May 2026)**  
- **硬件配置**：9块 **RTX PRO 6000 Blackwell SE**（96GB GDDR7, PCIe Gen5）分布在3个站点
- **网络环境**：所有跨站流量经由公网WAN路径
- **模型**：Llama 3.1 70B AWQ INT4（启用prefix caching和chunked prefill）
- **工作负载类**（8类）：
  - `baseline_mixed`, `rapid_shift`, `code_heavy`, `adversarial_spike`
  - `multiturn`, `rag_multidoc`, `long_context`, `very_long`

### **评估指标**
- **SLO Throughput Matrix**：通过二分法找到满足P95 TTFT ≤ SLO且成功率 > 95% 的最大可持续RPS
- **TTFT P50/P95/P99**
- **End-to-End Latency**
- **Leak Rate**：长提示被错误路由至能力不足节点的比例
- **Failover Reroute Latency (p99)**
- **DCGM Lead Time**：硬件指标预警领先于SLO违规的时间
- **Success Rate during DCGM exporter failure**

### **基线方法对比**
- **Round-Robin (RR)**：无状态轮询，代表默认Envoy/Nginx行为
- **Least-Request (LR)**：选择当前飞行请求数最少的节点，是生产中最常见的动态负载均衡策略
- **Solyx AI Grid**：使用10-signal v3 scorer进行智能加权路由

---

## **3. 主要实验结果和性能指标**

### **关键性能数据与对比**

#### ✅ **尾延迟显著降低**
- 在**同质H100集群**中，Solyx实现 **27.9% 的P99尾延迟下降**（从9,894ms降至7,138ms）
- 在**高jitter环境下**，Solyx带来 **27.8% 的P95 TTFT减少**

#### ✅ **异构集群自动流量集中**
- 在硬件异构环境中，Solyx实现了 **134:1 的流量集中比**，完全由实时遥测驱动，无需人工配置。

#### ✅ **能力错配泄露几乎消除**
| 方法 | Leak Rate | Long-Prompt Success |
|------|-----------|---------------------|
| Solyx | **0.43%** | **99.57%** |
| Least-Request | 28.71% | 71.29% |
| Round-Robin | 32.11% | 67.89% |

> ❗表明：**仅靠“负载感知”不足以解决capability mismatch**，必须引入error-rate信号。

#### ✅ **SLO吞吐量大幅提升**
在Tier-2 SLO下，Solyx相较RR提升 **1.56–1.75× 可持续RPS**，且在所有8个工作负载类别中均胜出：

| Workload Class | Solyx / RR |
|----------------|------------|
| baseline_mixed | 1.67× |
| code_heavy | 1.60× |
| adversarial_spike | 1.65× |
| very_long | 1.75× |

#### ✅ **Failover恢复更快**
| 模式 | Reroute p99 | Post-Kill Success |
|------|-------------|------------------|
| Solyx | **1,247ms** | **99.76%** |
| LR | 2,104ms | 98.81% |
| RR | 4,226ms | 94.12% |

> Solyx比RR快 **3.2倍**，得益于基于heartbeat TTL的快速故障检测机制。

#### ✅ **DCGM硬件遥测具有预测性**
- DCGM硬件信号平均**提前11.2秒**预示应用层SLO违规：
  - Thermal stress：提前12.3秒
  - PCIe contention：提前8.9秒
- 所有9次压力测试中**零SLO违规发生**，因Solyx已在预警窗口内完成流量迁移。

#### ✅ **网络信号带来额外增益**
| 子阶段 | Network Signal Gain (vs GPU-only) |
|--------|-------------------------------|
| saturation | 27.7% ↓ P95 |
| jitter20ms | **27.8% ↓ P95** |
| recovery | 21.2% ↓ P95 |

> 表明：**RTT与jitter作为一级输入对WAN环境至关重要**。

#### ✅ **控制面开销极低**
- 控制面更新延迟仅 **0.2 ms**
- 支持每秒数千次xDS推送（Campaign 2期间共推送超5万次）

#### ✅ **优雅降级能力**
- 当DCGM exporter宕机时，系统自动切换至vLLM-only模式：
  - 平均响应时间 < 1秒（824ms）
  - 成功率仍达 **99.90%**
  - 无流量丢失

---

## **4. 关键结论和发现**

### **主要发现**
1. **硬件遥测具有预测价值**：DCGM指标可提前约 **11.2秒** 预警应用层性能退化，使系统能**主动排水**而非被动响应。
2. **多信号融合优于单一维度**：结合硬件、应用、网络三类信号，可在复杂多变的生产环境中实现鲁棒路由。
3. **Least-Request并非万能**：在capability mismatch场景下表现接近Round-Robin，因其会因“失败返回快”而误判为“轻载”，反而加剧问题。
4. **网络路径质量不可忽略**：在WAN环境下，RTT与jitter直接影响用户体验，必须纳入路由决策。
5. **Solyx适用于绝大多数真实生产场景**：只要存在**非饱和负载、硬件退化、配置漂移或网络波动**，Solyx即可提供显著收益。

### **方法的局限性**
- **未验证大规模扩展性**：当前实验仅覆盖6~9个GPU节点，尚不清楚在数百或数千节点下的控制面开销。
- **未考虑KV-Cache Locality**：跨站点路由可能破坏prefix caching的局部性，在多轮对话场景中可能导致重计算代价。
- **单模型假设**：所有实验均运行同一模型（Llama 3.1 70B），未涉及多模型混合服务。
- **合成工作负载**：虽设计贴近真实分布，但仍非真实用户流量。
- **未评估成本与能耗**：缺乏cost-per-token或energy-per-request等经济性指标。
- **未处理区域性相关故障**：如整个region断电或网络中断的情况未测试。

### **未来工作方向**
1. **构建分层控制架构**（Hierarchical Aggregation Topology）：
   - 区域子控制器汇总本地遥测
   - 全局控制器基于摘要信息跨区调度
   - 解决大规模扩展问题

2. **平衡Pressure Minimization与KV-Cache Locality**：
   - 引入**前缀亲和度项**（prefix-affinity term）
   - 当目标节点压力在容忍范围内时，优先选择已有cache的节点

3. **推广至异构硬件+多模型场景**：
   - 支持不同generation GPU混合部署
   - 实现跨模型的能力感知路由（capability-aware model selection）

4. **量化DCGM lead time在更多硬件平台上的普适性**：
   - 扩展至A100、B200、MI300等其他架构
   - 建立通用的“硬件退化→应用影响”预警模型

5. **探索成本感知路由**（Cost-Aware Routing）：
   - 结合spot instance价格、电力成本等因素进行综合决策

---

> 📌 **总结一句话**：  
> **Solyx AI Grid首次在物理多站点基础设施上实证了“硬件遥测+应用指标+网络信号”三位一体的智能路由方案，不仅显著提升了LLM推理的服务质量，更揭示了DCGM等底层硬件信号在系统稳定性保障中的前瞻性价值。**

</details>

---

### 8. [AI-Driven Framework for Adaptive Water Network Management with Proof-of-Concept Implementation: Addressing Non-Revenue Water in Jordan](https://arxiv.org/abs/2606.15709)

**Authors**: Mohammed Fasha, Nahel Al-Maayta, Bilal Sowan, Mohammad Athamneh, Husam Barham  
**Category**: cs.AI  
**Published**: 2026-06-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.15709v1  

#### Abstract
Jordan faces severe water scarcity with 50\% of water produced is lost to leakage, theft and metering issues also known as non-revenue water (NRW). Traditional reactive approaches have proven insufficient for sustained NRW reduction. This paper proposes an intelligent framework integrating EPANET hy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对约旦严重的水资源短缺问题，特别是高达 **50%** 的非收益水（Non-Revenue Water, NRW）现象展开研究。NRW 主要由管道泄漏、非法取水和计量误差导致。传统方法依赖周期性人工检测和手动决策，反应滞后且难以持续降低 NRW。

### 提出的新方法与新思路
作者提出了一种**基于人工智能驱动的自适应供水网络管理框架**，其核心创新包括：

- **多技术融合架构**：首次将 **EPANET 水力模型**、**数字孪生（Digital Twin）**、**SCADA/IoT 数据流** 和 **大型语言模型（LLM）智能体** 集成于统一系统中，实现连续监控与自主决策。
- **离线运行能力**：通过本地部署开源 LLM（如 `llama3.1:8b` via Ollama），构建完全**离线的 PoC 实现**，避免云 API 成本、延迟及数据隐私风险。
- **LLM 作为决策中枢**：
  - 利用 **Retrieval-Augmented Generation (RAG)** 技术使 LLM 能理解操作规范；
  - 使用 **Function Calling** 机制让 LLM 可调用控制函数（如 `close_valve()`、`adjust_pump_speed()`）进行网络调控；
  - 将计算结果转化为自然语言健康报告，提升可解释性和运维效率。
- **适用于间歇供水场景**：专门设计支持约旦典型的间歇供水模式（intermittent supply），具备时间感知策略以区分计划停水与异常事件。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 响应速度 | 数小时至数天 | 分钟级（端到端 < 22分钟） |
| 决策方式 | 人工分析为主 | 自动化模拟 + AI 推理 |
| 成本结构 | 高人力成本 | 无API费用，低运维开销 |
| 数据安全 | 存在外泄风险 | 完全本地处理，零数据外传 |
| 扩展性 | 依赖高自动化设施 | 支持分阶段部署（从监测到自治） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **真实城市区域网络模型**：来自安曼（Amman）的一个典型供水分区（Distribution Zone, DZ），包含：
  - **1,164 个节点（Junctions）**
  - **1,310 条管道（Pipes）**
  - **2 个水源（Reservoirs）**
  - 海拔范围 864–984 米，管径 100–400 mm
- 该网络为重力供水平衡系统，具有代表性的环状拓扑结构。

### 实验设置
- **三层 PoC 架构**（见图2）：
  1. **Layer 1 - 水力仿真层**：使用 `EPYT`（EPANET-Python Toolkit）自动执行稳态水力模拟，获取压力、流量、液位等数据。
  2. **Layer 2 - 数据处理层**：Python 脚本提取统计指标，识别异常（如压力 <20m 或 >60m），并将结果结构化为 JSON 输入给 LLM。
  3. **Layer 3 - AI 分析层**：本地运行 `Ollama` 搭载 `llama3.1:8b` 模型，接收格式化提示（prompt），生成结构化健康报告。
- **测试场景**：
  - **Scenario 1**：基准状态下的网络行为分析
  - **Scenario 2**：模拟一个 **30.1 L/s 的管道破裂（burst）**，验证异常检测与定位能力

### 评估指标
- **响应时间**：从仿真到生成报告的端到端耗时
- **异常检出率**：能否准确识别低压/高压节点及局部流量异常
- **定位精度**：是否能收敛到破裂发生的节点集群
- **报告质量**：LLM 输出是否逻辑清晰、建议合理、符合工程实践
- **运行成本**：是否实现零 API 调用费用

### 基线方法对比
文中未直接与其他 ML/AI 模型进行量化对比，而是与以下现实情况形成隐含比较：
- 当前主流做法：定期巡检 + 人工数据分析
- 典型智能系统：依赖云端 LLM 或规则引擎的控制系统
- 对比维度强调“**离线可行性”、“响应速度”、“无需高自动化基础设施”**

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| 网络规模 | 1,164 Junctions, 代表性 DZ |
| 单次仿真+分析总耗时 | **< 22 分钟**（平均报告生成时间 15–30 秒） |
| API 成本 | **$0**（全程离线运行） |
| 异常检测粒度 | 支持节点级与管道级分析 |
| 爆管检测灵敏度 | 可识别 **30.1 L/s 泄漏量** |
| 局部流量变化 | 最大增加 **+10.5%**，15 条管道流量偏差 >1 L/s |
| 定位效果 | 成功识别出 **15 节点聚类**，空间上收敛于爆管位置 |
| 平均网络压力 | 保持稳定（40.2 m），体现系统抗干扰能力 |

### 与基线方法的对比结果
- 相较于传统人工分析（通常需数小时甚至数天）：
  - 本系统可在 **22 分钟内完成全流程分析并输出报告**
  - 减少突发泄漏造成的水量损失（例如：15 分钟内响应 vs 4 小时可节省约 **135 m³ 水**）
- 相较于依赖云服务的 AI 方案：
  - 实现 **零数据上传、零订阅成本**
  - 在普通硬件上即可运行（`llama3.1:8b` 仅需约 4.9GB 显存）

### 消融实验（Ablation Study）
虽未明确列出消融实验表格，但通过不同模块的功能剥离体现了关键组件作用：
- **无 RAG 政策检索** → 改为在 prompt 中嵌入政策文本 → 表明当前版本尚未完全实现动态知识查询，是未来改进方向
- **无 Function Calling** → 控制指令未实际发送 → 当前仅为“决策支持”，非闭环控制
- **不同 LLM 模型测试** → 发现 `llama3.1:8b` 在资源消耗与分析能力之间达到良好平衡

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **AI 驱动的水网管理系统在技术上可行**：即使在大规模（>1k 节点）复杂网络中也能实现自动化仿真与分析。
2. ✅ **LLM 可有效充当“虚拟专家”角色**：能够解读水力仿真输出，并生成具备优先级排序的维修建议和应急响应方案。
3. ✅ **本地化 LLM 具备实用价值**：使用 `Ollama + llama3.1:8b` 可实现高质量报告生成，无需连接外部 API。
4. ✅ **爆管检测范式转变**：在重力供给、双水源、环状网络中，**全局压力变化不显著**，必须转向 **局部流量再分布分析** 才能有效定位泄漏。
5. ✅ **分阶段实施路径切实可行**：
   - Phase 1（已验证）：自动监控 + 报告生成
   - Phase 2：AI 辅助决策（需接入 SCADA）
   - Phase 3：自主控制（需完善安全机制）

### 方法的局限性
| 局限性 | 说明 |
|-------|------|
| 尚未实现闭环控制 | 当前仅为“观察-分析-建议”流程，未真正执行 `close_valve()` 等动作 |
| 缺乏实时数据输入 | PoC 使用的是离线仿真数据，尚未集成 SCADA 实时流 |
| RAG 功能未完整实现 | 政策检索仍靠硬编码进 prompt，未建立动态向量数据库 |
| 未考虑水质因素 | 当前聚焦水量与压力，未涉及浊度、余氯等参数 |
| 场景泛化能力待验证 | 仅在一个典型 DZ 上测试，需更多区域验证普适性 |

### 未来工作方向
1. **推进三阶段部署路线图**：
   - 与水务公司合作开展试点，接入真实 SCADA 系统
   - 开发完整的 RAG 系统用于动态政策检索
   - 实现安全约束下的 Function Calling 控制闭环
2. **增强安全性与合规性**：
   - 进行网络安全认证
   - 获取监管机构对自动控制的审批
3. **扩展多智能体协作**：
   - 引入多个专业化 LLM Agent（如泄漏诊断 Agent、调度优化 Agent）协同工作
4. **推广至其他缺水地区**：
   - 将此低成本、离线可用的框架复制到中东及其他发展中国家的老旧管网系统中

---

> 🔗 **代码公开地址**：[https://github.com/msfasha/Research-Resources/tree/main/epanetjordan](https://github.com/msfasha/Research-Resources/tree/main/epanetjordan)  
> 📄 **论文引用**：Fasha, M., Al-Maayta, N., Barham, H., & Athamneh, M. *AI-Driven Framework for Adaptive Water Network Management...*, 2025.

</details>

---

### 9. [Replay What Matters: Off-Policy Replay for Efficient LLM Reinforcement Unlearning](https://arxiv.org/abs/2606.15333)

**Authors**: Zirui Pang, Chenlong Zhang, Haosheng Tan, Zhuoran Jin, Jiaheng Wei, Zixin Zhong  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.15333v1  

#### Abstract
LLM unlearning has emerged as a cost-effective alternative to full retraining for removing hazardous knowledge from pretrained models while preserving general utility. Recent RL-based methods such as RULE reformulate unlearning as learning a refusal behavior, but their on-policy optimization repeate...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Replay What Matters: Off-Policy Replay for Efficient LLM Reinforcement Unlearning**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的基于强化学习（RL）的 LLM unlearning 方法（如 **RULE**）采用 **on-policy** 优化策略，在训练过程中反复对相同的 forget 和 retain/boundary prompts 进行采样。然而，作者观察到存在显著的 **hard-easy disparity**：
- **Easy cases**：快速收敛，后期提供的梯度信号微弱；
- **Hard cases**：位于 forget/retain 语义边界附近，持续产生低奖励的 rollout，但由于是 on-policy 方法，这些 rollout 在单次使用后即被丢弃，造成信息浪费。

这导致训练效率低下：计算资源被浪费在已收敛的简单样本上，而真正需要优化的困难样本却未被充分利用。

---

### **提出了什么新方法或新思路**
作者提出 **ReRULE**（Replay-enhanced Reinforcement UnLEarning），一种 **off-policy replay 增强机制**，用于提升强化学习 unlearning 的效率。

#### **核心思想**：
- 在 **早期 GRPO 训练阶段**，将平均奖励低于阈值 $T$ 的 **low-reward rollout groups** 存入一个 **replay buffer** 中。
- 当 buffer 积累足够多的 hard cases 并完成 warm-up 后，进入 **hybrid training 阶段**：
  - 继续执行标准的 **on-policy GRPO 更新**；
  - 同时引入 **importance-sampled off-policy 更新**，重用 replay buffer 中的 hard cases。

通过这种方式，ReRULE 将梯度计算的注意力从已收敛的 easy cases 转移到仍需学习的 boundary cases 上。

---

### **相比现有方法的优势**
- ✅ **更高的训练效率**：避免重复采样已收敛样本，减少冗余计算。
- ✅ **更好的 hard-case 优化效果**：通过重放 hard-case rollouts，增强对边界区域的学习。
- ✅ **理论支持更强收敛性**：理论上证明 ReRULE 比纯 on-policy RULE 具有更紧的 hard-case 收敛界。
- ✅ **仅增加少量训练开销**：额外训练时间仅增加 **5–11%**，性价比高。
- ✅ **条件有效性**：当 hard/easy 差异显著时（如 MUSE-Books），提升明显；而在较简单的任务中（如 TOFU）则提升有限，符合预期设计。

---

## **2. 核心实验方法和设置**

### **使用了哪些数据集**
论文在三个主流 LLM unlearning benchmark 上进行实验：
- **RWKU**（Real-World Knowledge Unlearning）：针对现实世界名人实体的知识遗忘。
- **MUSE-Books**（Copyrighted Unlearning）：要求模型忘记《哈利·波特》书籍内容。
- **TOFU**（Entity Unlearning）：虚构作者传记问答任务，测试 1% 忘记比例下的表现。

---

### **实验设置和评估指标**

#### **模型架构**
- 基础模型：Meta Llama3-8B-Instruct（RWKU）、Llama2-7B（MUSE & TOFU）
- 使用 **GRPO**（Group-Relative Policy Optimization）作为 RL 优化器。

#### **ReRULE 关键参数**
- **Reward threshold $T$**：决定是否将 rollout group 存入 buffer（MUSE 中设为 0.4）。
- **Warm-up 条件**：buffer 达到最小容量且训练步数超过指定值（如 MUSE 中为 step > 26）。
- **Off-policy 更新方式**：使用 **importance sampling weight** 进行加权更新，防止旧策略偏差。

---

### **评估指标**

| 数据集       | Forget Quality ↓ | Retain Quality ↑ | Forget Naturalness ↑ |
|------------|------------------|------------------|------------------------|
| RWKU       | ROUGE-L on forget probes (FB/QA/AA) | ROUGE-L on retain probes | Readability, Helpfulness, Truthfulness |
| MUSE-Books | VerbMem, KnowMem | Utility (KnowMem on retain) | Readability, Helpfulness, Truthfulness |
| TOFU       | FQ (KS test), F-RL | MU (Model Utility) | — |

> 注：↑ 表示越高越好，↓ 表示越低越好。

---

### **基线方法对比**
- **Fine-tuning-based**: GA, NPO, Sim-NPO, FLAT, SGA
- **Preference-based**: DPO, KTO, PO
- **Other RL-based**: RULE (直接比较对象)
- **Generation-based**: Guard, ECO Prompt（不修改参数）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **MUSE-Books 结果（最显著提升）**
| 方法        | VerbMem ↓ | KnowMem ↓ | **Utility ↑** |
|-----------|----------|----------|--------------|
| RULE      | 0.0      | 1.4      | **46.3**     |
| **ReRULE** | **0.0**  | **1.4**  | **56.2**     |

👉 **Retain Quality 提升 9.9 个百分点**，同时保持同等优秀的遗忘性能。

#### **RWKU 结果**
| 方法           | Forget Quality (All) ↓ | Retain Quality (All) ↑ |
|--------------|-------------------------|------------------------|
| RULE (GRPO)  | 27.7                    | 73.7                   |
| **ReRULE (GRPO)** | **24.2**                | **74.1**               |

👉 在遗忘质量和保留质量上均优于 RULE。

#### **TOFU 结果**
| 方法       | FQ ↑    | MU ↑    | F-RL ↓   |
|----------|--------|--------|---------|
| RULE     | 0.1650 | 0.6151 | 0.1322  |
| ReRULE   | 0.1650 | 0.6076 | 0.1322  |

👉 性能与 RULE 相当，无显著提升，说明在简单任务中 replay 机制增益有限。

---

### **与基线方法的对比结果**
- ReRULE 在 **MUSE-Books** 上取得 **top-2 的 Forget Quality 和 Forget Naturalness**，同时实现 **最高的 Retain Quality（56.2）**，显著优于其他方法（如 NPO/KLR 最高仅 67.3，但 Forget Quality 更差）。
- 在 RWKU 上，ReRULE 在 **Forget Quality 和 Retain Quality 上均优于 RULE**。
- 在 TOFU 上，ReRULE 与 RULE 表现相近，验证其“仅在 hard/easy 差异大时有效”的假设。

---

### **消融实验结果**

#### **(1) Replay Buffer 内容的重要性**
| 方法             | KnowMem ↓ | Utility ↑ |
|----------------|----------|----------|
| RULE           | 1.4      | 46.3     |
| ReRULE         | 1.4      | 56.2     |
| **ReRULE_random** | 1.5      | **46.3** |

👉 若 buffer 随机存储样本（而非 hard cases），性能退化至与 RULE 相当 → 证明 **hard-case selection 是关键**。

#### **(2) Threshold $T$ 的影响**
| 方法         | KnowMem ↓ | Utility ↑ |
|------------|----------|----------|
| ReRULE ($T=0.4$) | 1.4      | **56.2** |
| ReRULE ($T=0.6$) | 1.3      | 52.3     |
| ReRULE ($T=1.0$) | 1.4      | 53.7     |

👉 随着 $T$ 增大，更多“非 hard”样本被存入 buffer，导致重复优化 easy cases，反而降低性能 → 说明 **选择标准需严格**。

#### **(3) 训练成本分析**
| Benchmark | RULE 训练时间 | ReRULE 训练时间 | 增加幅度 |
|---------|---------------|------------------|----------|
| RWKU    | 21.2h         | 22.3h            | ~5.2%    |
| MUSE    | 6.8h          | 7.1h             | ~4.4%    |
| TOFU    | 1.8h          | 2.0h             | ~11.1%   |

👉 **仅增加 5–11% 的训练时间**，即可获得显著性能提升，性价比极高。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Hard-case replay 显著提升 unlearning 效率**：通过 off-policy replay 机制，ReRULE 成功将训练资源聚焦于最难优化的边界样本。
2. ✅ **ReRULE 在 hard/easy disparity 明显的任务中效果最佳**：在 MUSE-Books 上 Retain Quality 提升近 10 分，而在 TOFU 上几乎无提升，符合理论预期。
3. ✅ **理论支持实践**：论文从理论上证明 ReRULE 可以获得比 RULE 更紧的 hard-case generalization bound。
4. ✅ **replay buffer 的构建方式至关重要**：只有存储 hard cases 才能带来收益，随机存储无效。

---

### **方法的局限性**
- ❗ **仅适用于存在明显 hard/easy 分化的场景**：若所有样本难度均匀或普遍较易（如 TOFU），则 replay 增益有限。
- ❗ **依赖于 reward signal 的稳定性**：hard-case 判定基于 rollout 的平均 reward，若 reward 设计不合理，可能导致错误分类。
- ❗ **当前仅在 GRPO 框架下验证**：是否可推广至其他 RL 算法（如 PPO）尚待研究。

---

### **未来工作方向**
- 🔮 探索 **adaptive thresholding** 策略，动态调整 $T$ 以适应不同训练阶段。
- 🔮 将 ReRULE 扩展至 **multi-turn dialogue unlearning** 或 **multimodal unlearning** 场景。
- 🔮 结合 **active learning** 思想，主动识别潜在 hard cases 并优先采样。
- 🔮 研究如何将 replay 机制应用于 **SFT-based unlearning** 方法中，提升其样本利用效率。

--- 

> **一句话总结**：  
> ReRULE 通过引入 **off-policy replay** 机制，高效复用 hard-case rollouts，在几乎不增加训练成本的前提下，显著提升了 LLM 强化学习 unlearning 的性能，尤其在复杂边界任务中表现出色。

</details>

---

### 10. [Rethinking the Role of Efficient Attention in Hybrid Architectures](https://arxiv.org/abs/2606.15378)

**Authors**: Ziqing Qiao, Yinuo Xu, Chaojun Xiao, Zhou Su, Zihan Zhou, Yingfa Chen, Xiaoyue Xu, Xu Han, Zhiyuan Liu  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.15378v1  

#### Abstract
Modern language models increasingly adopt hybrid architectures that combine full attention with efficient attention modules, such as sliding-window attention (SWA) and recurrent sequence mixers. However, how these efficient modules shape model capabilities remains poorly understood. To address this ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Rethinking the Role of Efficient Attention in Hybrid Architectures

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代语言模型广泛采用**Hybrid Architecture**（混合架构），即结合 **Full Attention**（全注意力）与 **Efficient Attention**（高效注意力，如滑动窗口注意力 SWA 或循环序列混合器）。然而，这些高效模块在混合架构中究竟扮演何种角色、如何影响长上下文能力（long-context capability）的形成，目前仍缺乏系统性的理解。

本文旨在回答以下三个核心研究问题：
- **RQ1 - Scaling Behavior**: 混合架构在短/长上下文任务上的扩展规律是怎样的？
- **RQ2 - Mechanism Analysis**: 高效注意力的设计如何影响长上下文性能？
- **RQ3 - Architecture Design**: 如何设计更有效的混合架构？

### 提出了什么新方法或新思路
1. **提出“Large-Window Laziness”现象**  
   发现更大的滑动窗口（如 SWA-2048）反而会**延迟 full attention 中 retrieval head 的形成**，因为局部窗口已能覆盖大部分依赖关系，削弱了 full attention 学习长距离检索的动力。

2. **重新定义高效注意力的角色**  
   高效注意力并非直接承担长距离信息传递，而是作为 **Optimization Prior**（优化先验），通过调节训练动态来间接影响 full attention 学习长距离检索的速度。

3. **提出新的设计原则**  
   混合架构的设计应从“增强高效注意力本身”转向“更好地激活和强化 full attention”，例如：
   - 使用小窗口 SWA（如 SWA-128）
   - 在 full attention 层应用 **NoPE**（移除位置编码）

### 相比现有方法的优势
- **机制解释性强**：首次从 scaling law 和 retrieval head 动态角度揭示了高效注意力的间接作用机制。
- **设计指导明确**：提出了可操作的新设计原则（如 NoPE + 小窗口 SWA），显著提升长上下文性能而不损害短上下文表现。
- **评估更连续可靠**：使用 **log(LongPPL)** 作为连续指标，比传统离散基准（如 RULER）更适合分析训练动态。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
| 数据集 | 用途 |
|-------|------|
| **C4** | 用于预训练和验证 Loss（short-context 建模质量） |
| **GovReport** | 用于计算 **LongPPL**，评估长上下文建模能力 |
| **RULER** 和 **LongBench** | 下游长上下文任务评估（如 NIAH、多文档问答等） |
| **NIAH (Needle-in-a-Haystack)** | 探针任务，用于追踪 retrieval head 的形成过程 |

### 实验设置和评估指标

#### 模型架构对比
比较了七种架构：
- **Full**: 全注意力 Transformer
- **SWA-128 / SWA-512 / SWA-2048**: 不同窗口大小的滑动窗口注意力
- **Lightning / Mamba-2 / GDN**: 循环序列混合器（recurrent sequence mixers）

所有混合模型均采用 **1:1 层交替**（layer-wise alternation）策略。

#### 模型规模与训练预算
- 规模 S1–S5（参数量从 15M 到 477M，不含嵌入层）
- 训练 token 数 $ D \in \{100N, 200N, ..., 1000N\} $
- 上下文长度统一为 **16K**

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Validation Loss** | 衡量短上下文建模能力 |
| **log(LongPPL)** | 连续指标，衡量长上下文建模能力（基于 GovReport 和 Llama-3.1-8B 作为参考模型） |
| **RULER / LongBench 准确率** | 下游任务性能 |
| **Retrieval Head 分析** | 包括 attention entropy 和参数收敛距离，用于追踪 retrieval 能力发展 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Scaling Law 结果（图2）
- **Validation Loss**：所有混合架构曲线几乎重合，说明高效注意力对短上下文能力影响极小。
- **log(LongPPL)**：
  - 初期差异显著：**SWA-2048 性能最差**（gap 最大）
  - 随着训练充分，所有架构最终收敛到相似水平
  - 表明：**高效注意力主要影响长上下文能力的“出现速度”而非“最终上限”**

#### ✅ 机制分析结果
- **Receptive-field constraint 实验**（图3）：
  - 限制 full attention 的可见范围 → log(LongPPL) 显著上升
  - 限制 efficient attention → 几乎无影响
  - ➜ **长距离信息主要由 full attention 承载**

- **Layer-wise probing**（图4）：
  - 信息增益集中在 **full attention 层**（奇数层）
  - efficient attention 层贡献微弱甚至负向
  - ➜ 再次验证 full attention 是长距离建模的核心

- **Retrieval Head Tracing**（图5b）：
  - **SWA-2048 的 retrieval head 收敛最慢**（entropy 更高，参数变化更缓）
  - 支持 “Large-Window Laziness” 假说

#### ✅ 新设计带来的性能提升（表2 & 图8）

| 模型 | ShortAvg | RULER (16K) | LongBench (16K) | RULER (32K) | LongBench (32K) |
|------|----------|-------------|----------------|-------------|----------------|
| **SWA-128** | 41.31 | 46.13 | 17.52 | 41.86 | 18.30 |
| **SWA-128-NoPE** | **41.32** | **52.88** | **19.02** | **46.98** | **19.46** |

- **NoPE 应用于 full attention 层后**：
  - **RULER 提升 +6.75 pts**（S5, 16K）
  - **LongBench 提升 +1.5 pts**
  - **短上下文性能基本不变**
- 消融实验证明该改进来自对 full attention 的强化，而非其他因素

---

## 4. 关键结论和发现

### 论文的主要发现
1. 🔹 **Long-range retrieval 主要由 Full Attention 承担**，而非高效注意力模块（即使是理论上无限感受野的循环混合器）。
2. 🔹 **Efficient Attention 是 Optimization Prior**：它不决定最终能力，但显著影响 full attention 学习长距离检索的**速度和路径**。
3. 🔹 **Large-Window Laziness**：更大的滑动窗口会延迟 retrieval head 的形成，因为它减少了 full attention 必须学习远程依赖的压力。
4. 🔹 **设计启示**：应优先考虑能**激活 full attention 检索能力**的设计，如：
   - 使用较小的 SWA 窗口
   - 在 full attention 层应用 **NoPE**
   - 减少 full attention 层密度（当模型足够大时可行）

### 方法的局限性
- 实验最大模型仅 **~0.66B 参数**，未达到工业级超大规模（如 >10B）。
- 预训练直接使用 16K 上下文，未模拟“先短后长”的主流训练范式。
- 仅测试了代表性高效注意力算子（SWA、Lightning、Mamba-2、GDN），未涵盖 RWKV-7、Kimi-Linear 等变体。
- 设计探索（如 NoPE）主要用于验证机制，非全面架构搜索。

### 未来工作方向
- 在更大规模模型上验证结论是否依然成立。
- 探索更多直接增强 full attention 的方法（如改进 PE、引入显式记忆机制）。
- 研究不同训练调度（如 curriculum learning）对 retrieval head 形成的影响。
- 将本框架应用于其他高效架构（如 MoE、state reuse）的机制分析。

---

> **一句话总结**：  
> 本文颠覆了“高效注意力负责长上下文”的直觉认知，指出其本质是**塑造 full attention 学习过程的优化先验**，并据此提出“用小窗口 + NoPE 强化 full attention”的新设计范式，在不牺牲短上下文性能的前提下显著提升长上下文能力。

</details>

---

### 11. [LESS Is More: Mutual-Stability Sampling for Diffusion Language Models](https://arxiv.org/abs/2606.16908)

**Authors**: Amr Mohamed, Guokan Shang, Michalis Vazirgiannis  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.16908v1  

#### Abstract
Diffusion large language models (dLLMs) offer a promising alternative to autoregressive decoding by iteratively refining masked sequences, enabling parallel token updates and bidirectional conditioning. Their practical efficiency, however, is limited by sampling procedures that execute a fixed numbe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LESS Is More: Mutual-Stability Sampling for Diffusion Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
扩散语言模型（**dLLMs**）虽然在生成效率上优于自回归（AR）模型，但其标准采样过程存在以下瓶颈：
- **固定步数预算**（fixed reverse-step budget）导致计算资源浪费：部分位置在早期已稳定，但仍被重复更新。
- **过早提交不稳定预测**（premature commitment）：一旦错误 token 被写入，后续步骤难以修正。
- 缺乏对“何时停止 refine 并提交 token”的**动态、可靠判断机制**。

### 🚀 提出的新方法：LESS（Learning-free Early Stopping Sampler）
LESS 是一种**无需训练、模型无关**（training-free, model-agnostic）的自适应采样器，将 token 提交视为一个**在线停止问题**（online stopping problem），提出 **mutual-stability sampling** 框架。

#### 核心思想：联合稳定性规则（Joint Stability Rule）
一个 masked position 只有在同时满足以下三个条件时才可被 unmask：
1. **Top-1 高置信度**（High Confidence）  
   $ \text{conf}_t = p_t(\text{argmax}) \geq c $
2. **Top-1 token 持续性**（Persistence）  
   当前 top-1 token 在最近 $P$ 步中保持不变。
3. **Top-K 分布漂移小**（Low Drift）  
   使用 **Top-K Inter-step Jensen-Shannon Divergence (JSD)** 衡量连续两步间预测分布的变化：
   $$
   \text{JSD}_{t,i} = \text{KL}(p_t \| m) + \text{KL}(p_{t+1} \| m),\quad m = \frac{p_t + p_{t+1}}{2}
   $$

> 🔑 **JSD 的优势**：对称、有界（$[0, \log 2]$）、不依赖参考分布，适合衡量双向变化。

### ⚖️ 相比现有方法的优势
| 方法 | 局限性 | LESS 改进 |
|------|--------|----------|
| **Confidence-based** (e.g., Prophet) | 仅看当前置信度，忽略历史变化 | 加入 persistence 和 JSD，防止“刚变就提交” |
| **KL-based divergence** (e.g., KLASS) | KL 不对称、无界，阈值难调 | 使用对称有界的 JSD，更鲁棒 |
| **Fixed schedule** | 浪费计算资源 | 动态终止，平均减少 **72.1%** 反向步数 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
涵盖三大类任务共 **7 个基准**：
| 类别 | 数据集 | 任务类型 |
|------|--------|---------|
| **Math & Reasoning** | GSM8K, MATH | 数学推理 |
| **Code Generation** | HumanEval, MBPP | 代码生成 |
| **General Knowledge** | MMLU, HellaSwag, WinoGrande | 常识与知识问答 |

### ⚙️ 实验设置
- **模型**：
  - **Dream-7B**：全序列扩散采样（full-sequence diffusion）
  - **LLaDA-8B**, **LLaDA-1.5-8B**：半自回归块状采样（semi-autoregressive blockwise）
- **最大反向步数**：$T_{\text{max}} = 256$
- **生成长度**：统一为 256 tokens
- **评估方式**：zero-shot，统一 prompt 和答案提取协议

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 各任务的标准准确率（exact match / pass@1 / accuracy） |
| **Avg. Steps** | 每样本平均执行的反向去噪步数 |
| **FLOPs / Wall-clock Latency** | 推理计算量与实际延迟 |
| **Mean Accuracy** | 对 math、code、general knowledge 三类任务取 macro-average |

### 🆚 基线方法对比
| 基线 | 类型 | 特点 |
|------|------|------|
| **Base** | 固定步数采样器 | 执行全部 256 步 |
| **Prophet** | 自适应采样器 | 基于 top-2 confidence gap 决策 |
| **KLASS** | 自适应采样器 | 结合 confidence 与 token-level KL divergence |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 模型 | 方法 | Mean Acc (%) | Avg. Steps | 步数减少 |
|------|------|--------------|-----------|----------|
| Dream-7B | Base | 63.30 | 256.0 | — |
| | **LESS (ours)** | **65.18** (+1.88) | **64.3** | **-74.9%** |
| LLaDA-8B | Base | 56.58 | 256.0 | — |
| | **LESS (ours)** | **57.41** (+0.83) | **76.5** | **-70.1%** |
| LLaDA-1.5-8B | Base | 57.84 | 256.0 | — |
| | **LESS (ours)** | **57.85** (+0.01) | **73.2** | **-71.4%** |

> ✅ **总体表现**：LESS 在所有三个模型家族上均提升或持平准确率，同时**平均减少 72.1% 的反向步数**。

### 🔍 与基线对比结果
- **相比 Prophet 和 KLASS**：
  - 在所有模型上，LESS 实现更高的 **mean accuracy** 和更低的 **step count**。
  - 尤其在 **math 和 code 任务**上增益最显著（如 GSM8K 上 Dream-7B 提升 +2.28 pts）。
- **速度提升**：
  - 在 Dream-7B + GSM8K 上，wall-clock 延迟从 19.45s 降至 **5.16s**（**3.77× 加速**）。
  - FLOPs 减少达 **5.01×**（在 $T=256$ 设置下）。

### 🔬 消融实验结果（Ablation Studies）

#### （1）单信号 vs 联合规则（Table 4）
| 变体 | 平均准确率（Dream-7B） | 说明 |
|------|------------------------|------|
| Conf-only ($c=0.75$) | 69.4 | 最强单一信号 |
| JSD-only ($d=0.04$) | 25.3 | 单独 JSD 效果差 |
| Persistence-only ($P=2$) | 39.1 | 无法独立作为判据 |
| **Full LESS** | **65.2** | 显著优于任一单独信号 |

> ✅ **结论**：confidence 是主门控，JSD 和 persistence 是互补的安全阀。

#### （2）drop-one 消融（Table 5）
| 移除组件 | Dream-7B 准确率下降 |
|--------|--------------------|
| Remove confidence | -18.7 pts (GSM8K) |
| Remove JSD | -0.2 ~ +1.4 pts |
| Remove persistence | -0.2 ~ -2.4 pts |

> ✅ **结论**：confidence 是核心；JSD 和 persistence 提供稳定性保障。

#### （3）策略分析
- **fallback 触发率低**（<10%），表明大多数决策由稳定性规则主导。
- **frontier-first policy** 比 parallel unmasking 更优，避免同时提交多个 token 导致上下文扰动。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Stability ≠ Confidence**  
   高置信度不代表稳定。LESS 引入 **temporal persistence** 和 **distributional drift** 判断，有效防止过早提交。
2. **JSD 是理想的分布稳定性度量**  
   对称、有界、无需指定参考分布，特别适合 diffusion 这种双向上下文更新场景。
3. **联合规则优于单一判据**  
   confidence + persistence + JSD 构成互补三角，实现高效且鲁棒的 early stopping。
4. **效率提升来自 step reduction，而非 per-step 优化**  
   每步仍需一次 Transformer forward，但总步数大幅下降 → 直接降低 FLOPs 和 latency。

### ⚠️ 局限性
- **显式质量-效率权衡**：通过阈值 $(c, d)$ 控制，保守设置更安全但加速有限，激进设置可能引入错误。
- **全局配置泛化性依赖**：当前使用统一 $(c,d)=(0.75,0.040)$，最优值可能随任务/模型变化。
- **未覆盖所有 dLLM 架构**：目前验证集中于 Dream 和 LLaDA 系列，其他架构需进一步测试。

### 🔮 未来工作方向
- **动态阈值调整**：根据任务难度、输出长度自动调节 $c$ 和 $d$。
- **结合训练时加速方法**：如 flow matching 或 distillation，与 LESS 形成端到端加速 pipeline。
- **扩展至图像/多模态 diffusion 模型**：探索 mutual-stability 在非文本生成中的应用。
- **理论分析 stopping time 的收敛性质**：建立 online stopping 与生成质量之间的形式化关系。

---

> 💡 **一句话总结**：  
> **LESS 通过 confidence、persistence 和 JSD 的联合判断，在 dLLM 中实现了“该停则停”的智能采样，在几乎不损甚至提升准确率的前提下，将反向步数平均减少 72.1%，显著降低了推理成本。**

</details>

---

### 12. [Exploring Extrinsic and Intrinsic Properties for Effective Reasoning with Code Interpreter](https://arxiv.org/abs/2606.16934)

**Authors**: Patomporn Payoungkhamdee, Napat Laosaengpha, Jenta Wonglertsakul, Pittawat Taveekitworachai, Pume Tuchinda, Panjapong Poobanchuen, Ekapol Chuangsuwanich, Can Udomcharoenchaikit, Samuel Cahyawijaya, Peerat Limkonchotiwat, Sarana Nutanong  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.16934v1  

#### Abstract
Reasoning with a Code Interpreter (CI) has emerged as an effective paradigm for enhancing the reasoning capabilities of large language models (LLMs) through executable computation and iterative verification. Despite its growing adoption, the behavioral properties underlying effective code reasoning ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Exploring Extrinsic and Intrinsic Properties for Effective Reasoning with Code Interpreter

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统性地研究了**基于 Code Interpreter (CI) 的推理过程中，哪些关键属性（properties）促成了有效的推理行为**。尽管 CI 已被广泛用于提升 LLM 的数学、逻辑等复杂任务表现，但其背后的行为机制——即模型在生成代码时表现出的“外在”和“内在”特性——尚未被深入探索。

具体而言，论文聚焦于两个核心问题：
- **RQ1**: 在 CI 推理中，关键的外在（extrinsic）和内在（intrinsic）属性如何体现？
- **RQ2**: 这些属性能否在推理时（test time）或训练时（training time）被利用以提升下游性能？

---

### 提出了什么新方法或新思路
论文从两个维度提出新的分析框架，并设计了相应的干预策略：

#### （1）**Extrinsic Property（外在属性）：Crucial Tokens**
- 定义：指在推理过程中频繁出现、具有语义引导作用的关键 token，如 `let`, `check`, `wait`。
- 创新点：首次将自然语言推理中的“crucial token”概念迁移到 CI 场景，验证其对 test-time scaling 的影响。

#### （2）**Intrinsic Property（内在属性）：Code-Specific Cognitive Behaviors (CoBE)**
- 定义：模型在多轮 CI 交互中展现出的认知行为模式，包括：
  - **Verification**（验证中间结果）
  - **Backtracking**（回溯失败路径）
  - **Subgoal Setting**（子目标分解）
  - **Backward Chaining**（逆向推理）
- 创新点：扩展了 Gandhi et al. (2025) 的认知行为分类，适配到 CI 场景，并通过 prompt engineering 和数据合成实现可量化分析。

#### 新干预方法：
- **Test-time intervention**：通过强制追加 crucial tokens（如 `let`, `check`）来延长推理链，观察性能变化。
- **Training-time augmentation**：构建 CoBE-augmented 数据集，在 SFT 和 RL 阶段注入认知行为监督信号。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **理论层面** | 首次系统刻画 CI 推理的有效性源于哪些行为属性，填补了工具增强型推理的机制空白 |
| **实践层面** | 提出可在训练/推理阶段直接应用的轻量级改进手段（token 注入 / 行为蒸馏） |
| **泛化性** | 发现不同模型家族（Qwen vs Llama）对属性响应存在差异，揭示了架构依赖性 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 |
|--------|------|
| **SymBench** (Chen et al., 2025) | 多任务符号推理基准，涵盖：<br>• 数学（Math）<br>• 空间推理（Spatial）<br>• 逻辑（Logical）<br>• 排序（Ordering）<br>• 优化（Optimization）<br>• 搜索（Search） |
| **AIME24, AIME25, MATH500** | 数学推理标准测试集，用于评估训练后性能 |

---

### 实验设置与评估指标

#### 模型选择
- 主要模型族：
  - **Qwen3-8B**, **Qwen2.5-7B-Instruct**
  - **Llama3.1-8B-Instruct**
  - 蒸馏模型：DeepSeek-R1-Distill-Qwen-7B / Llama-8B

#### 测试时干预（Test-Time Scaling）
- 方法：替换终止 token（如 `</THINK>`）为 crucial token（`wait`, `let`, `check`），最多追加 5 次
- 评估方式：比较不同 token 注入下的准确率变化趋势

#### 训练时增强（Training-Time Augmentation）
- 方法：基于 ReTool 框架（Feng et al., 2026），引入 CoBE 数据增强
  - 使用 **Claude Sonnet 4.5** 合成带有显式认知行为的新轨迹
  - 仅保留最终答案正确的样本（共 1,915 条）
- 训练流程：
  1. Supervised Fine-Tuning (SFT)
  2. Reinforcement Learning (RL) with GRPO

#### 评估指标
- **Accuracy (%)**：主指标
- **Response Length (Tokens)**：衡量效率
- **Crucial Token Frequency (normalized)**：外在属性强度
- **Cognitive Behavior Prevalence (%)**：内在属性强度（由 Oracle LLM 判断）

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **No extrapolation** | 不进行任何 token 注入的标准推理 |
| **ReTool-SFT / ReTool-RL** | Feng et al. (2026) 的标准训练流程 |
| **+ CoBE** | 在 SFT 或 RL 中加入 CoBE 增强数据 |
| **2x wait / let / check** | 两次注入对应 crucial token |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table 2）

| 方法 | AIME24 | AIME25 | MATH500 | 平均 |
|------|--------|--------|---------|------|
| Qwen2.5-Coder-7B-Instruct | 7.1 | 2.5 | 58.6 | 22.7 |
| +ReTool-SFT | 15.8 | 12.1 | 73.9 | 33.9 |
| +ReTool-SFT w/ CoBE | **23.3** | 13.8 | 73.2 | **36.8** |
| +ReTool-RL | 40.0 | 29.6 | 84.4 | 51.6 |
| +ReTool-RL w/ CoBE | **42.1** | **31.7** | **84.5** | **52.8** |

> ✅ **CoBE 增强带来持续增益**：在 Qwen2.5 系列上，SFT 阶段平均提升 +2.9%，RL 阶段再提升 +1.2%

---

### 与基线方法的对比结果

#### Test-Time Scaling 结果（Figure 4 & Table 1）
- **`let` token 效果最稳定**：
  - 在数学、排序、优化任务中几乎单调提升
  - Qwen3-8B 上数学准确率从 28.92% → 30.08%（+1.16%）
- **`check` 在空间/逻辑任务更优**
- **`wait` 表现不佳**：多数情况下低于 baseline，甚至导致性能下降
- **小模型（<7B）难以受益**：Qwen3-4B 几乎无增益，部分任务反而退化

#### Training-Time CoBE 增强结果
- 对 **Qwen2.5 系列有效**：
  - SFT 阶段平均提升约 3%
  - RL 阶段进一步提升 1–5%
- 对 **Qwen3-8B-Instruct 反而有害**：
  - 性能全面下降（如 AIME24 从 67.5 → 58.8）
  - 分析表明：CoBE 导致 crucial token 频率降低、响应长度过短

---

### 消融实验结果（Ablation Study）

#### （1）Response Length 影响（Figure 5）
- CoBE 训练后的输出显著变短（尤其在 SFT 阶段）
- 但过短可能丢失关键推理步骤 → 损害高阶模型表现

#### （2）Crucial Token Preservation（Figure 5 右图）
- ReTool-RL 模型保持较高 crucial token 频率
- 加入 CoBE 后频率下降 → 说明 **外在属性也被削弱**

#### （3）Base Model 验证（Table 3）
- 在 **Qwen3-8B-Base** 上应用 CoBE：
  - AIME24 提升 15%
  - AIME25 提升 35.1%
- 说明：**CoBE 对未经充分 post-training 的基础模型有益**

> 🔍 结论：**CoBE 的有效性取决于模型是否已具备成熟推理能力**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **更强的 CI 推理模型天然表现出更多 crucial tokens 和 CoBE 行为**
   - 如 Qwen3-8B 显著高于弱模型
   - 特别是 `verification`, `subgoal setting`, `backward chaining`

2. ✅ **`let` 是最有效的 test-time scaling token**
   - 尤其适用于数学、排序、优化类任务
   - `wait` 并非通用有效，需谨慎使用

3. ✅ **CoBE 增强可提升训练效果（尤其对中等能力模型）**
   - 在 Qwen2.5 上 SFT 和 RL 均获益
   - 改善 token efficiency，减少 overthinking

4. ⚠️ **对高级推理模型（如 Qwen3-8B-Instruct）可能适得其反**
   - 因破坏原有 crucial token 分布和长度分布
   - 表明：**不能简单叠加干预，需考虑已有能力的兼容性**

5. 🔄 **内外属性相互关联**
   - 单独增强内在行为（CoBE）可能导致外在信号（crucial tokens）减弱
   - 成功推理需要两者协同

---

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **依赖 Oracle LLM 判断行为** | 使用 Qwen3-Next-80B 作为 judge，可能存在偏见或不一致 |
| **未统一整合内外属性** | 当前实验分别处理 extrinsic/intrinsic，缺乏 joint framework |
| **局限于 CI 场景** | 是否推广到其他工具（如 web browser, API）尚不清楚 |
| **执行安全风险** | 自动生成并运行代码存在潜在安全隐患（虽在沙箱中） |

---

### 未来工作方向
1. **Develop Unified Framework**  
   设计同时优化 extrinsic 和 intrinsic 属性的联合训练机制。

2. **Cross-Architecture Generalization**  
   研究为何某些模型（如 DeepSeek distill-Qwen）更能从 token 注入中受益。

3. **Dynamic Test-Time Control**  
   根据任务类型自适应选择最佳 crucial token（如数学用 `let`，搜索用 `check`）。

4. **Behavior-Preserving Fine-Tuning**  
   在注入 CoBE 的同时，约束 crucial token 频率和长度分布不变。

5. **Extend to Multi-Tool Agents**  
   将本框架应用于更复杂的 agent 系统（如 WebAgent, AutoCoder）。

--- 

> 💡 **一句话总结**：  
> 本文首次系统揭示了 Code Interpreter 推理背后的“双引擎”驱动机制——**外在的 crucial tokens 引导推理节奏，内在的 cognitive behaviors 构建稳健思维结构**；二者皆可被利用以提升性能，但也需警惕对先进模型的干扰效应。

</details>

---

### 13. [Conflict-Aware Federated Fine-Tuning of Large Language Models with Mixture-of-Experts](https://arxiv.org/abs/2606.15625)

**Authors**: Yijun Lu, Zihan Fang, Pengpeng Qiao, Zheng Lin, Jing Yang, Yuxin Zhang, Por Lip Yee, Zhe Chen, Jun Luo  
**Category**: cs.LG  
**Published**: 2026-06-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.15625v1  

#### Abstract
The continuous scaling of large language models (LLMs) incurs prohibitive computational costs, making Mixture-of-Experts (MoE) a scalable alternative for efficient fine-tuning via sparse activation. While federated learning (FL) emerges as the paradigm for privacy-preserving collaborative optimizati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Conflict-Aware Federated Fine-Tuning of Large Language Models with Mixture-of-Experts

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Federated Learning (FL)** 场景下对基于 **Mixture-of-Experts (MoE)** 架构的大语言模型（LLMs）进行微调时，由于客户端数据具有高度 **non-IID** 特性（即数据分布异构），导致相同索引的专家（same-indexed experts）在不同客户端上优化目标不一致甚至冲突。这种“**专家冲突（expert conflicts）**”引发梯度聚合时的 **destructive interference**，破坏全局优化路径，降低模型收敛稳定性与性能。

### 🚀 提出的新方法：FC-MoE
作者提出 **FC-MoE**（Federated Conflict-aware MoE），一种面向联邦环境下 MoE 模型微调的冲突感知框架，其核心创新包括：

1. **Importance-aware Expert Weighting（重要性感知专家加权）**  
   - 综合考虑专家的 **activation frequency**（激活频率）和 **gradient magnitude**（参数更新范数）来计算每个本地专家更新的重要性得分 $ c_{i,k} $。
   - 更可靠地反映专家在本地数据中的代表性与学习强度，用于指导聚合权重分配。

2. **Gradient Consensus Projection（梯度共识投影）**  
   - 在服务器端构建一个 **global consensus direction**（基于加权平均的主导优化方向）。
   - 对于与共识方向夹角为钝角（cosine similarity < 0）的本地梯度，将其 **投影到共识方向的正交平面上**，抑制冲突分量而不完全丢弃本地更新。

3. **Local Knowledge Retention（本地知识保留机制）**  
   - 将被过滤掉的领域特异性梯度残差（residuals）在本地缓存，并在下一轮训练开始前通过缩放重新锚定（re-anchor）回本地模型。
   - 实现 **global alignment 与 local specialization 的平衡**。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | FC-MoE 的改进 |
|------|--------|---------------|
| **FedAvg**, **FedMoE**, **PFL-MoE** | 默认所有专家更新兼容，采用简单平均，易受冲突干扰 | 显式检测并缓解冲突，提升聚合质量 |
| 传统 MoE 联邦方案 | 忽视非 IID 下专家功能漂移问题 | 引入几何一致性判断与动态投影机制 |
| 多数个性化 FL 方法 | 难以兼顾全局收敛与局部性能 | 通过残差重注入保留客户专长 |

> ✅ 总体优势：**加速收敛、增强全局泛化能力、同时保障各客户端的个性化表现**

---

## 2. 核心实验方法和设置

### 📚 数据集
- **AGNews**：新闻文本分类任务，基础语义理解基准。
- **MMLU**（Massive Multitask Language Understanding）：涵盖57个学科领域的复杂知识推理任务，评估模型多任务泛化能力。

### ⚙️ 实验设置
- **模型架构**：采用 **Switch Transformer** 作为 MoE 主干网络，每层使用 Top-1 路由策略，共 16 个专家。
- **联邦系统配置**：
  - 客户端数量：10 个
  - 中央服务器：1 个
  - 硬件加速：NVIDIA RTX 4090 GPU
- **训练细节**：
  - 每轮通信执行 1 epoch 本地微调
  - 学习率：$1 \times 10^{-4}$
  - 总通信轮数：25
  - 关键超参：知识保留系数 $\lambda = 0.5$
- **Non-IID 设置**：
  - 使用 **Dirichlet 分布** 划分数据标签，浓度参数 $\alpha \in \{0.1, 0.5, 1.0\}$，$\alpha=0.1$ 表示极端标签偏斜，挑战性强。

### 📊 评估指标
- **Global Test Accuracy**：全局模型在测试集上的准确率
- **Local Accuracy**：各客户端本地模型在私有数据上的平均准确率及方差
- **Convergence Speed**：达到稳定性能所需的通信轮数
- **Ablation Study**：验证各模块独立贡献

### 🆚 基线方法对比
| 方法 | 类型 |
|------|------|
| **FedAvg** | 标准联邦平均，全模型更新 |
| **FedMoE** | 针对 MoE 的联邦微调，仅聚合激活专家 |
| **PFL-MoE** | 个性化联邦 MoE，支持客户端定制专家组合 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table I，$\alpha=0.1$ 最具挑战场景）

| 方法 | AGNews 准确率 | MMLU 准确率 |
|------|----------------|-------------|
| FedAvg | 0.7740 | 0.3008 |
| PFL-MoE | 0.8223 | 0.3541 |
| FedMoE | 0.8374 | 0.3699 |
| **FC-MoE（本文）** | **0.8617** | **0.4017** |

> ✅ 在最严苛的 non-IID 条件下（$\alpha=0.1$），FC-MoE 在两个任务上均显著领先：
- AGNews 提升约 **+2.4%** vs FedMoE
- MMLU 提升约 **+3.2%** vs FedMoE

随着数据分布趋于均衡（$\alpha$ 增大），性能差距缩小，说明 FC-MoE 在 **高异构环境** 中更具优势。

### 🔄 收敛速度（见 Fig. 3a）
- FC-MoE 在更少的通信轮次内达到更高精度，表现出 **更快且更稳定的收敛曲线**。
- 归因于 **gradient consensus projection** 抑制了震荡，维持了平滑优化轨迹。

### 👤 本地性能表现（见 Fig. 3b 和 Table III）
| 方法 | AGNews 局部准确率 | MMLU 局部准确率 |
|------|--------------------|------------------|
| FedAvg | 0.5574 | 0.2992 |
| FC-MoE | **0.6732** | **0.3559** |

- FC-MoE 不仅提升了全局性能，也实现了更高的 **平均局部准确率**。
- 客户端间准确率 **方差更低**，表明模型具备更强的鲁棒性和公平性。

### 🔍 消融实验（Ablation Study，见 Table II & Fig. 4）
比较以下变体：
- **(a)** 移除 Gradient Consensus Projection
- **(b)** 移除 Local Knowledge Retention
- **(c)** 移除 Importance-aware Weighting
- **FC-MoE（完整版）**

#### 结果分析：
| 变体 | AGNews 准确率 | MMLU 准确率 | 观察 |
|------|----------------|--------------|-------|
| (a) 无共识投影 | 0.8068 | 0.3478 | 收敛慢、波动大 → **共识机制是稳定性的关键** |
| (b) 无知识保留 | 0.8264 | 0.3616 | 局部性能下降明显 → **保留机制对个性化至关重要** |
| (c) 无重要性加权 | 0.8264 | 0.3616 | 共识方向偏差 → **加权影响聚合可靠性** |
| **FC-MoE** | **0.8359** | **0.3764** | 所有组件协同作用，实现最优平衡 |

> ✅ 三大模块缺一不可，共同支撑 FC-MoE 的优越表现。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **专家冲突是联邦 MoE 微调中的根本性问题**，尤其在 non-IID 场景下会导致严重性能退化。
2. **显式的冲突检测与几何投影机制**（Gradient Consensus Projection）能有效缓解 destructive interference，显著提升全局收敛速度与稳定性。
3. **重要性感知加权**使聚合过程更加智能，优先采纳高质量、高代表性的本地更新。
4. **本地知识保留机制**成功防止了全局聚合对客户特定知识的“清洗”，实现了 **personalization 与 generalization 的双赢**。
5. FC-MoE 在多种任务和异构程度下均优于主流基线，尤其在极端 non-IID 场景中优势最为突出。

### ⚠️ 方法的局限性
- 当前方法依赖于服务器端计算共识方向，可能增加中心节点的计算负担（尽管仍远低于集中训练）。
- 仅考虑 **Top-k 路由机制下的线性投影**，未探索更复杂的非线性协调机制。
- 实验基于模拟的 non-IID 划分，真实世界中专家漂移模式可能更为复杂。

### 🔮 未来工作方向
1. 扩展至 **decentralized FL** 或 **cross-device FL** 场景，减少对中央服务器的依赖。
2. 探索 **dynamic expert reallocation** 机制，在训练过程中根据共识信号调整路由策略。
3. 将 FC-MoE 思路推广至其他稀疏化结构，如 **LoRA-MoE** 或 **Activation Sparsity** 模型。
4. 结合 **federated evaluation** 机制，实现端到端的隐私保护 LLM 协同开发 pipeline。

---

> 💡 **总结一句话**：  
> **FC-MoE 是首个明确提出并系统解决联邦 MoE 中“专家冲突”问题的框架，通过“共识引导 + 冲突抑制 + 知识保留”三位一体设计，在保持高效稀疏微调的同时，实现了更强的鲁棒性、更快的收敛和更好的个性化效果。**

</details>

---

### 14. [Mean-Field Parallel Decoding for Discrete Diffusion Language Models](https://arxiv.org/abs/2606.15805)

**Authors**: Tamim Zoabi, Ameen Ali, Liran Ringel, Lior Wolf  
**Category**: cs.LG  
**Published**: 2026-06-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.15805v1  

#### Abstract
Discrete diffusion language models enable parallel token generation, offering a pathway to low-latency decoding. However, selecting tokens independently by marginal confidence limits effective parallelism: tokens that appear reliable in isolation can form incompatible configurations when several pos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Mean-Field Parallel Decoding for Discrete Diffusion Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
离散扩散语言模型（**Discrete Diffusion Language Models, dLLMs**）虽然支持并行生成，能够显著降低推理延迟，但在实际应用中面临“**联合不一致性（joint inconsistency）**”问题。具体表现为：
- 当前主流方法基于**边际置信度（marginal confidence）**选择要解码的位置（如熵、margin等），忽略了位置间的依赖关系。
- 多个高置信度位置同时解码时，可能产生语法冲突或语义重复（例如重复生成“3”），导致整体序列质量下降。

该问题限制了**有效并行度**，使得dLLMs难以在保持高质量的同时实现高吞吐量。

---

### **提出了什么新方法或新思路**
本文提出了一种**无需训练、单次前向传播即可完成的解码框架**：**Mean-Field Parallel Decoding（均场并行解码）**。

其核心思想是将并行提交（commit）决策建模为一个**结构化推断问题**，通过以下机制协调多个位置的并行更新：

1. **能量函数建模（Energy-Based Formulation）**  
   将每个被掩码位置 $i$ 是否被提交表示为二元变量 $s_i \in \{0,1\}$，构建一个**成对马尔可夫随机场（MRF）**：
   $$
   P(S) \propto \exp\left(\sum_i c_i s_i - \sum_{i<j} D_{ij} s_i s_j\right)
   $$
   - **一元势（Unary Potential）**：$c_i$ 表示局部置信度（如top-1与top-2预测之间的log margin）
   - **成对势（Pairwise Potential）**：$D_{ij}$ 衡量两个位置预测分布的相似性，使用**Jensen-Shannon Divergence (JSD)** 构造：
     $$
     D_{ij} = 1 - \frac{\text{JSD}(\pi_{t,i}, \pi_{t,j})}{\ln 2}
     $$
     若两位置输出分布高度重叠，则 $D_{ij}$ 接近1，抑制两者同时提交。

2. **变分均场优化（Variational Mean-Field Inference）**  
   对上述NP难的MAP问题进行**mean-field近似**，得到一个简单的固定点迭代公式：
   $$
   q_i^{(r+1)} = \sigma\left(c_i - \sum_{j \neq i} D_{ij} q_j^{(r)}\right)
   $$
   其中 $\sigma$ 是sigmoid函数，$q_i$ 表示第$i$个位置的“提交强度”。经过几次迭代后，通过阈值筛选最终的提交集合 $C_t$。

---

### **相比现有方法的优势**
| 特性 | 本文方法（Ours） | 典型基线（如Entropy/KLASS） | DAWN / DEMASK 类方法 |
|------|------------------|-------------------------------|------------------------|
| 是否需要额外模型 | ❌ 否 | ❌ 否 | ✅ 是（需辅助模型或学习模块） |
| 是否需要重新训练 | ❌ 否 | ❌ 否 | ✅ 是（部分方法） |
| 是否捕捉位置间依赖 | ✅ 显式建模 | ❌ 仅看边际置信度 | ✅ 是（但更复杂） |
| 计算开销 | 轻量级，仅增加 $O(m^2|V|)$ 成本（可并行） | 最低 | 高（图构造、验证等） |
| 并行效率提升 | ⬆️ 最高 | 中等 | 中等偏高 |

✅ **优势总结**：
- **Training-free & Gradient-free**：直接利用已有模型输出，无需微调或额外参数。
- **Single-pass & Lightweight**：所有计算在一个forward pass内完成，适合部署。
- **提高安全并行度**：通过抑制竞争性位置，允许更多高置信位置安全提交。
- **改善质量-延迟权衡（quality-latency trade-off）**：在几乎不损失准确率的前提下大幅提升TPS。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
涵盖数学推理与代码生成两大类任务：
- **数学推理**：
  - **GSM8K**：小学级别数学应用题
  - **MATH**：竞赛级数学问题
- **代码生成**：
  - **HumanEval**：Python函数级代码生成
  - **MBPP**：面向初学者的编程任务

---

### **实验设置和评估指标**

#### **模型**
在三个主流dLLM上测试泛化能力：
- **LLaDA-8B-Instruct**
- **LLaDA-1.5**
- **Dream-v0-Instruct-7B**

所有方法使用相同checkpoint，仅改变解码策略。

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **Accuracy** | 任务正确率（pass@k 或 functional correctness） |
| **Tokens Per Second (TPS)** | 每秒生成token数，衡量吞吐量 |
| **Speedup** | 相对于Entropy基线的TPS加速比 |
| **NFE (Number of Function Evaluations)** | 完成一次生成所需的去噪步数，反映效率 |

#### **硬件与实现**
- 使用 **8× NVIDIA H100 GPUs**
- 基于 DAWN 开源代码库实现
- 所有方法统一block size、停止条件、采样协议

---

### **基线方法对比**
| 基线 | 类型 | 特点 |
|------|------|------|
| **Entropy** | 不确定性驱动 | 按预测熵排序，保守但慢 |
| **KLASS** | KL引导选择 | 更强的边际置信标准，优于Entropy |
| **LocalLeap** | 局部确定性传播 | 利用锚点扩展局部提交范围，无依赖建模 |
| **DAWN** | 依赖感知调度 | 显式估计位置依赖关系，指导安全并行 |
| **Ours (Mean-Field)** | 结构化提交选择 | 引入JSD+均场迭代，隐式协调提交 |

> 注：未比较DEMASK因其引入可训练组件，不符合“training-free”目标。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| Model | Method | GSM8K Acc. | Speedup× | MATH Acc. | Speedup× | HumanEval Acc. | Speedup× | MBPP Acc. | Speedup× |
|-------|--------|------------|----------|-----------|----------|----------------|----------|-----------|----------|
| LLaDA-8B | Entropy | 79.15 | 1.00× | 33.36 | 1.00× | 40.24 | 1.00× | 29.41 | 1.00× |
|           | KLASS | 75.05 | 2.27× | 31.80 | 1.79× | 39.63 | 1.83× | 27.23 | 2.28× |
|           | LocalLeap | 77.71 | 4.32× | 32.96 | 3.30× | 40.24 | 4.04× | 30.67 | 4.60× |
|           | DAWN | 77.78 | 4.38× | 32.34 | 3.41× | 40.24 | 4.05× | 28.07 | 4.87× |
|           | **Ours** | **76.34** | **4.93×** | **31.10** | **4.07×** | **40.70** | **4.88×** | **28.08** | **6.10×** |

> ✅ **观察**：
> - 在所有模型和任务上，**Ours取得了最高的Speedup**（平均约 **5.12×** vs Entropy）
> - 准确率略有波动，但仍保持在合理范围内，甚至在HumanEval上略胜其他方法
> - 特别是在MBPP上达到 **6.10× 加速**，远超DAWN（4.87×）

---

### **消融实验结果（见 Table 2）**

| 变体 | Acc. | TPS | NFE |
|------|------|-----|-----|
| Baseline (Entropy) | 75.51 | 11.15 | 256.0 |
| No pairwise interaction | 72.40 | 56.20 | 48.7 |
| Uniform interaction matrix | 71.95 | 51.92 | 52.6 |
| One-shot update (R=1) | 54.51 | 20.11 | 140.0 |
| **Ours (full)** | **73.24** | **49.13** | **54.3** |

> 🔍 **分析**：
> - 移除pairwise interaction → 准确率大幅下降 → 说明**交互建模至关重要**
> - 使用uniform矩阵而非JSD → 性能下降 → 说明**JSD提供的结构信息有效**
> - R=1（只一次更新）→ 准确率暴跌至54.5 → 说明**迭代均场收敛必要**
> - **完整方法在精度与效率之间取得最佳平衡**

---

### **敏感性分析（Table 3 & Figure 2）**
- **Commit Threshold $T$**：
  - $T$ 越高 → 提交越保守 → 准确率↑，TPS↓
  - 存在平滑的质量-延迟权衡曲线，便于调节
- **Mean-field Iterations $R$**：
  - $R=2$ 已足够收敛；继续增加收益递减
- **Block Size**（图2）：
  - 增大block size → 可选位置增多 → TPS持续上升
  - 准确率在中等block size下稳定，过大时轻微下降

> ✅ 表明方法具有良好的**可控性和鲁棒性**

---

## **4. 关键结论和发现**

### **主要发现**
1. **边际置信度不足以支撑高效并行解码**：独立决策会导致“联合不一致”，成为dLLMs的性能瓶颈。
2. **JSD是一种有效的轻量级交互信号**：无需额外模型，仅从模型自身预测分布即可提取位置间竞争/协同关系。
3. **Mean-field迭代能有效协调提交行为**：通过简单固定点更新，实现全局一致的并行解码。
4. **显著改善quality-latency trade-off**：在多个任务和模型上均实现**最高吞吐量**，且准确率接近最优水平。
5. **完全无需训练或修改模型架构**：即插即用，适用于任何现有的dLLM解码流程。

---

### **方法的局限性**
1. **计算复杂度为 $O(m^2 |V|)$**：
   - 在极大规模block（如 $m > 1000$）时可能成为瓶颈
   - 但可通过稀疏化或近似JSD缓解
2. **仅建模成对交互**：
   - 忽略更高阶依赖（如三元组语法约束）
   - JSD是启发式代理，并非真实联合概率
3. **仍存在质量-延迟权衡**：
   - 极端追求速度仍可能导致错误累积
   - 最优配置需根据任务调整（如commit threshold）

---

### **未来工作方向**
- **扩展到高阶交互建模**：结合attention heads或factor graph进行更精细依赖建模
- **动态block划分**：根据语义边界自适应切分block以增强局部一致性
- **与其他加速技术结合**：如speculative decoding、drafting机制融合
- **理论分析收敛性与误差传播**：建立更坚实的理论基础

---

## ✅ 总结
**Mean-Field Parallel Decoding** 提供了一个简洁而强大的视角：将dLLM中的并行解码视为**结构化推断问题**，并通过**变分均场+JSD交互**实现在单次前传中协调多位置提交。它不仅在实践中显著提升了生成效率（**最高达8.62×加速**），而且保持了生成质量，是当前**最高效的training-free并行解码方案之一**，有望广泛应用于各类dLLM系统中。

</details>

---

### 15. [Scalable Pairwise Kernel Learning with Stochastic Vec Trick](https://arxiv.org/abs/2606.16979)

**Authors**: Napsu Karmitsa, Tapio Pahikkala, Antti Airola  
**Category**: cs.LG  
**Published**: 2026-06-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.16979v1  

#### Abstract
Pairwise learning is a specialized form of supervised learning that focuses on predicting outcomes for pairs of objects. In this work, we introduce SPaiK, a new scalable kernel learning method tailored for pairwise settings. Our approach preserves the expressive power of kernel methods while substan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scalable Pairwise Kernel Learning with Stochastic Vec Trick

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **pairwise kernel learning**（如药物-靶标亲和力预测、推荐系统）在处理大规模数据时面临严重的计算瓶颈。尽管 **Generalized Vec Trick (GVT)** 已能将 Kronecker product kernel 的矩阵-向量乘法从 $O(n^2)$ 降低到 $O(nm + nq)$（$n$: 配对数，$m/q$: 独特药物/靶标数），但在超大规模场景下，训练成本仍然过高。

此外，主流方法如 **CGKronRLS** 和 **KronSVM** 虽然有效，但难以扩展到百万级样本，且在 **zero-shot learning (ODOT)** 场景中表现受限。

### 提出的新方法与创新
本文提出 **SPAIK**（Scalable Pairwise Kernel Learning with Stochastic Vec Trick），其核心创新包括：

- **Stochastic Generalized Vec Trick (sGVT)**  
  对 GVT 进行随机化扩展，通过 **target-wise 或 drug-wise 批次采样**，将每次迭代的计算复杂度进一步降至 $O(n_B m + n_B q)$，其中 $n_B \ll n$ 是批次大小。这是首次将 **stochastic optimization** 引入 pairwise kernel learning 的高效 kernel matrix-vector 乘法中。

- **Stochastic Inexact Limited-Memory Bundle Method (StoILMBM)**  
  一种新的随机优化器，专为非光滑目标函数（如 $\epsilon$-insensitive squared loss + $l_1$ 正则）设计，结合 sGVT 实现 mini-batch 训练。

- **信息保持机制（Auxiliary Dual-Drug Matrix M）**  
  在 sGVT 中引入辅助矩阵 $M \in \mathbb{R}^{m \times q}$ 来缓存跨批次的 dual 变量依赖关系，避免因随机采样丢失关键交互信息，显著提升收敛稳定性和预测精度。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可扩展性** | 支持 mini-batch 训练，内存和计算开销大幅下降，适用于更大规模数据 |
| **零样本学习性能** | 在 ODOT 设置下表现优于或媲美 CGKronRLS，尤其在稀疏数据中更具鲁棒性 |
| **灵活性** | sGVT 可独立用于其他基于 Kronecker kernel 的方法（如集成进 CGKronRLS） |
| **效率-精度平衡** | 如 `SPaiK20` 在仅用 20% 的 target 批次下，精度损失极小但速度提升一个数量级 |

---

## 2. 核心实验方法和设置

### 数据集
在 **7 个真实世界 Drug-Target Affinity (DTA)** 数据集上进行实验，涵盖连续型与二分类标签：

| 数据集 | 类型 | 药物数 (m) | 靶标数 (q) | 配对数 (n) | 观测比例 (%) |
|--------|------|------------|------------|-------------|----------------|
| Davis | Continuous | 68 | 442 | 30,056 | 100% |
| Metz | Continuous | 1421 | 156 | 93,356 | 42% |
| KIBA | Continuous | 2111 | 229 | 118,254 | 24% |
| Merget | Continuous | 2967 | 226 | 167,995 | 25% |
| GPCR | Binary | 223 | 95 | 21,185 | 100% |
| Ion Channels | Binary | 210 | 204 | 42,840 | 100% |
| Enzymes | Binary | 445 | 664 | 295,480 | 100% |

### 实验设置
- **五折随机划分**：训练 / 验证 / 测试 = 1:1:1，确保各 OTS 设置下的公平比较
- **四种预测设置（OTS）**：
  - **IDIT**：药物和靶标均见于训练（插值）
  - **IDOT**：新靶标，已知药物
  - **ODIT**：新药物，已知靶标
  - **ODOT**：全新药物-靶标对（zero-shot learning）
- **每种设置重复 5 次**，报告平均结果

### 评估指标
| 指标 | 描述 |
|------|------|
| **C-index** | 衡量预测排序一致性，越高越好（完美=1.0，随机=0.5） |
| **IC-index** | Interaction Concordance Index，衡量模型是否捕捉到真正的 pairwise interaction（而非主效应），完美建模=1.0，线性模型≈0.5 |
| **MSE** | 回归任务误差，越低越好 |
| **CPU Time** | 实际运行时间（秒） |

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **CGKronRLS** | RLS + Conjugate Gradient + GVT | 使用平方损失 + $l_2$ 正则，成熟实现，当前 SOTA 基线 |
| **KronSVM** | SVM + Truncated Newton + GVT | 使用 hinge loss，适合二分类任务 |
| **SPaiK**（本文） | $\epsilon$-insensitive loss + $l_1$ 正则 + sGVT + StoILMBM | 支持多种 batch size（如 `SPaiK100`, `SPaiK20`, `SPaiK10`） |
| **SPaiK-R10** | 消融变体 | 完全随机 target 采样（无 epoch-wise 控制），用于验证 batch selection 策略有效性 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（代表性结果）

#### ✅ 在 Zero-Shot Learning (ODOT) 上的表现（C-index）
| Dataset | CGKronRLS | SPaiK100 | SPaiK20 |
|--------|-----------|---------|--------|
| Davis | 0.614 | **0.634** | **0.638** |
| Metz | 0.601 | 0.603 | **0.608** |
| KIBA | 0.602 | 0.603 | 0.600 |
| Merget | 0.663 | 0.663 | 0.660 |
| GPCR | 0.693 | **0.724** | 0.699 |
| Ion Channels | 0.598 | 0.592 | **0.613** |
| Enzymes | 0.632 | 0.625 | 0.604 |

> 🔍 **观察**：在多数数据集上，尤其是 Davis 和 GPCR，`SPaiK100` 和 `SPaiK20` 显著优于 CGKronRLS，表明所提方法在 zero-shot 场景中更具泛化能力。

#### ⏱️ 效率对比（以 Davis IDIT 为例）
| 方法 | C-index | CPU Time (s) |
|------|--------|--------------|
| CGKronRLS | **0.845** | 10.67 |
| SPaiK100 | 0.822 | 8.40 |
| SPaiK20 | 0.813 | **4.39** |
| SPaiK10 | 0.808 | **2.91** |
| SPaiK1 | 0.763 | **1.64** |

> 📉 **趋势**：随着 batch size 减小，C-index 缓慢下降，但 CPU 时间显著减少。`SPaiK20` 实现了最佳权衡——精度仅降 0.013，时间节省 ~60%。

#### 🔄 消融实验：Epoch-wise vs Fully Random Batch Selection (`SPaiK10` vs `SPaiK-R10`)
- **C-index**：epoch-wise 略优或相当（除个别 ODOT 外）
- **IC-index**：epoch-wise **显著更优**，说明其更好地保留了 pairwise interaction 结构
- **CPU Time**：两者接近，无明显差异

> ✅ **结论**：epoch-wise 批次选择策略有助于维持 interaction 学习质量，是推荐做法。

#### 🧪 不同 batch size 的精度-效率权衡
| Batch Size ($p_B$) | 平均 C-index 下降（vs SPaiK100） | 速度提升倍数（估算） |
|--------------------|-------------------------------|-----------------------|
| 100 (full-batch) | 0.000 | 1x |
| 20 | ~0.003–0.008 | 2–5x |
| 10 | ~0.011–0.018 | 3–7x |
| 5 | ~0.030–0.033 | 5–10x |
| 1 | ~0.059–0.061 | >10x |

> 💡 **建议配置**：`SPaiK20` 是默认推荐设置，在几乎所有任务中提供最优 trade-off。

---

## 4. 关键结论和发现

### 主要发现
1. **sGVT 成功实现了 scalable pairwise kernel learning**  
   将原本受限于全批量计算的方法推广至 mini-batch 范式，使 kernel 方法可应用于更大规模 pairwise 问题。

2. **SPAIK 在 zero-shot learning 中表现卓越**  
   得益于非光滑优化（InexactLMBM）、$\epsilon$-insensitive loss 和 $l_1$ 正则，模型在 ODOT 设置下具有更强鲁棒性，甚至超越 CGKronRLS。

3. **适度 batch size（如 20% targets）即可保持高精度**  
   `SPaiK20` 在几乎不牺牲性能的前提下大幅提升训练速度，是实际应用的理想选择。

4. **epoch-wise batch selection 更有利于 interaction 学习**  
   IC-index 分析显示，系统性覆盖所有 targets 的策略比完全随机采样更能保留 pairwise interaction 信号。

5. **Merget 数据集表现异常**  
   所有方法在该数据上的 IC-index 接近 0.5，提示其可能缺乏强交互结构或存在数据偏差，非算法缺陷。

### 方法局限性
- **依赖 side information**：要求药物和靶标均有 feature/kernels，无法处理纯协同过滤场景。
- **实现复杂度较高**：相比标准 RLS，需维护辅助矩阵 $M$ 和 $G$，增加工程负担。
- **未探索 drug-wise batching**：文中仅验证 target-wise 更优，但未深入分析为何如此，可能存在领域特定偏好。
- **GPU 加速未涉及**：当前实现基于 CPU（Python/Fortran），尚未利用 GPU 并行潜力。

### 未来工作方向
- 将 sGVT 扩展至 **validation/test phase**，实现端到端随机化推理
- 探索 **hybrid batching strategies**（如动态调整 $p_B$）
- 集成 sGVT 到 **CGKronRLS** 等经典框架中，构建“stochastic RLS”
- 开发 **GPU-accelerated version** of sGVT for even larger-scale applications
- 探索 **adaptive auxiliary matrix update** 策略以进一步降低内存占用

---

> ✅ **总体评价**：  
> 本论文提出的 **SPAIK + sGVT** 框架成功解决了 pairwise kernel learning 的可扩展性难题，在保持 kernel 方法强大表达能力的同时，显著提升了训练效率，并在最具挑战性的 zero-shot 场景中展现出优越性能。该工作为将 kernel 方法重新带回大规模 pairwise 学习前沿提供了坚实路径。

</details>

---

### 16. [CONCORD: Asynchronous Sparse Aggregation for Device-Cloud RAG under Document Isolation](https://arxiv.org/abs/2606.15179)

**Authors**: Xuedong Hu, Zhiqing Tang, Zhi Yao, Tian Wang, Weijia Jia  
**Category**: cs.AI  
**Published**: 2026-06-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.15179v1  

#### Abstract
Retrieval-augmented generation (RAG) has emerged as a pivotal technique for improving language models by incorporating external knowledge at inference time. As device-cloud collaborative inference makes it feasible to deploy small language models on edge devices, a new setting arises in which privat...

---

### 17. [Kairos: A Native World Model Stack for Physical AI](https://arxiv.org/abs/2606.16533)

**Authors**: Kairos Team, Fei Wang, Shan You, Qiming Zhang, Tao Huang, Zuoyi Fu, Zhisheng Zheng, Yunlong Xi, Feng Lv, Xiaoming Wu, Zeyu Liu, Cong Wan, Pu Li, Ruiqing Yang, Xiaoou Li, Wei Wang, Kangkang Zhu, Yuwei Zhang, Shi Fu, Xiaoning Wu, Xuzeng Fan, Dacheng Tao, Xiaogang Wang  
**Category**: cs.AI  
**Published**: 2026-06-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.16533v1  

#### Abstract
World models are transitioning from passive visual generators to foundational, operational infrastructure for Physical AI: they must natively acquire world knowledge from heterogeneous experience, maintain persistent states over long horizons, and execute efficiently within real deployment constrain...

---

### 18. [BALTO: Balanced Token-Level Policy Optimization for Hallucination Mitigation](https://arxiv.org/abs/2606.15893)

**Authors**: Ning Li, Zixuan Guo, Yan Xu, Wenbo Fei, Yifan Niu, Chang Luo, Yasheng Wang, Weiwen Liu, Yong Yu, Weinan Zhang  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.15893v1  

#### Abstract
Hallucinations remain a major obstacle to deploying large language models (LLMs) in knowledge-intensive settings, where generated responses must be faithfully grounded in provided evidence. Reinforcement learning (RL) is a promising direction for hallucination mitigation, but response-level faithful...

---

### 19. [A Large-Scale Multi-Dimensional Empirical Study of LLMs for Conversation Summarization](https://arxiv.org/abs/2606.15974)

**Authors**: Weixiao Zhou, Gengyao Li, Xianfu Cheng, Junnan Zhu, Feifei Zhai, Zhoujun Li  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.15974v1  

#### Abstract
Despite the significant advancement of LLMs in conversation summarization, their evaluation remains limited by insufficient scenarios, input lengths, and sample sizes. Furthermore, existing benchmarks often omit frontier reasoning systems and efficient small models, or lack fine-grained, multi-dimen...

---

### 20. [DEEPRUBRIC: Evidence-Tree Rubric Supervision for Efficient Reinforcement Learning of Deep Research Agents](https://arxiv.org/abs/2606.17029)

**Authors**: Minghang Zhu, Chuyang Wei, Junhao Xu, Yilin Cheng, Zhumin Chen, Jiyan He  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.17029v1  

#### Abstract
Deep research agents synthesize long-form reports by searching and reasoning over retrieved evidence. Reinforcement learning with rubric-based rewards improves these agents by optimizing them against checkable criteria that translate report quality into reward signals, but its efficiency depends on ...

---

### 21. [KVEraser: Learning to Steer KV Cache for Efficient Localized Context Erasing](https://arxiv.org/abs/2606.17034)

**Authors**: Mufei Li, Shikun Liu, Dongqi Fu, Haoyu Wang, Yinglong Xia, Hong Li, Hong Yan, Pan Li  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.17034v1  

#### Abstract
Post-hoc context erasing over the KV cache is challenging because a local edit has a global consequence: once a span has been processed, its influence propagates into the cached states of all subsequent tokens. This issue arises naturally in long-context LLM applications, where stale retrieved facts...

---

### 22. [Is RISC-V Ready for Massively Parallel Astrophysical Codes?](https://arxiv.org/abs/2606.15490)

**Authors**: Jenny Lynn Almerol, Nitin Shukla, Federico Ficarelli, Geray S. Karademir, Andrea Bartolini, Emanuele Venieri, Giacomo Madella, Elisabetta Boella  
**Category**: cs.DC  
**Published**: 2026-06-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.15490v1  

#### Abstract
We present a performance and portability evaluation of three well-established astrophysical production codes, namely iPIC3D, PLUTO, and OpenGGCM, on a Sophgo SG2044 RISC-V processor (part of the Monte Cimone cluster), with comparisons to AMD EPYC 9554 (x86) and NVIDIA GH200 Grace (ARM) systems. Thes...

---

### 23. [Tangram: Hiding GPU Heterogeneity for Efficient LLM Parallelization](https://arxiv.org/abs/2606.16907)

**Authors**: Yanda Tao, Pedro F. Silvestre, Marcel Wagenl\"ander, Peter Pietzuch  
**Category**: cs.DC  
**Published**: 2026-06-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.16907v1  

#### Abstract
The scale of LLM training jobs requires parallelization planning over large GPU clusters. Due to different GPU types and interconnects added over time, these GPU clusters are increasingly heterogeneous. Automatic LLM parallelizers can search for parallelization plans but face an exploding search spa...

---

### 24. [Stop the Sampler! Classifier-Based Adaptive Stopping for Sampling Kernels](https://arxiv.org/abs/2606.16073)

**Authors**: Kirill Korolev, Nikita Morozov, Stepan Pavlenko, Esmeralda S. Whitammer, Sergey Samsonov  
**Category**: cs.LG  
**Published**: 2026-06-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.16073v1  

#### Abstract
Sampling from complex, unnormalized probability densities is a fundamental challenge in Bayesian inference and probabilistic modeling. While Markov chain Monte Carlo (MCMC) methods provide asymptotic guarantees, they often suffer from slow mixing and high computational costs due to fixed or manually...

---

### 25. [CogGuard: Cognitive and Operational Profiling for Proactive Warning in Edge Intelligent Services](https://arxiv.org/abs/2606.15199)

**Authors**: Zhi Yao, Weihao Chen, Zhiqing Tang, Hanshuai Cui, Qianli Ma, Weijia Jia, Wei Zhao  
**Category**: cs.AI  
**Published**: 2026-06-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.15199v1  

#### Abstract
Proactive warning is an important capability for edge intelligent services, where the system predicts whether a subject will successfully complete an incoming task under strict latency and privacy constraints. Such prediction depends on both long-term static attributes and short-term dynamic states ...

---

### 26. [ROMPAR: Morphological Completion and Demographic Unlearning for Romanian-Accented Speech Recognition](https://arxiv.org/abs/2606.15984)

**Authors**: Andrei-Marius Avram, Aureliu-Valentin Antonie, \c{S}tefan-Bogdan Badea, Andrei Florea, Robert-Nicolae Zaharoiu, Dumitru-Clementin Cercel  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.15984v1  

#### Abstract
Automated transcription of parliamentary proceedings faces significant hurdles due to demographic bias, dialectal variation, and technical artifacts such as utterance truncation during segmentation. This paper introduces the ROManian PARliamentary Speech Corpus (ROMPAR) dataset, a 17.80-hour corpus ...

---

### 27. [Surpassing Scale by Efficiency: A Compact 135M Parameter Foundational LLM Natively Adapted for the Bangla Language](https://arxiv.org/abs/2606.16383)

**Authors**: Rabindra Nath Nandi  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.16383v1  

#### Abstract
While the NLP landscape is dominated by multi-billion parameter architectures, their deployment in low-resource, non-Latin scripts remains computationally prohibitive for edge configurations, mobile systems, and decentralized local hardware. This paper presents bangla-smollm-135m, a highly compact 1...

---

### 28. [Can LLM Agents Infer World Models? Evidence from Agentic Automata Learning](https://arxiv.org/abs/2606.16576)

**Authors**: Reef Menaged, Gili Lior, Shauli Ravfogel, Roee Aharoni, Gabriel Stanovsky  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.16576v1  

#### Abstract
We propose agentic automata learning to evaluate the extent to which tool-calling LLM agents can uncover hidden environments through interaction. In our setup, an agent should uncover a hidden deterministic finite automaton (DFA) by interacting with an oracle through (1) membership queries ("Does th...

---

### 29. [SING: Synthetic Intention Graph for Scalable Active Tool Discovery in LLM Agents](https://arxiv.org/abs/2606.16591)

**Authors**: Qiao Xiao, Haochen Shi, Yisen Gao, Wenbin Hu, Huihao Jing, Tianshi Zheng, Baixuan Xu, Ziheng Zhang, Weiqi Wang, Haoran Li, Jiaxin Bai, Yangqiu Song  
**Category**: cs.CL  
**Published**: 2026-06-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.16591v1  

#### Abstract
Large language model (LLM) agents increasingly rely on agent harnesses that manage context, tools, and multi-turn execution, making tools a central interface for acting in realistic digital environments. As harness-connected tool ecosystems expand to hundreds or thousands of APIs, services, and task...

---

### 30. [Secure and Low-Latency IoT Analytics Using an Edge-Based Streaming Architecture](https://arxiv.org/abs/2606.14712)

**Authors**: Atul, Varun Shukla, Vivek Shukla, Mehul Kumar Das  
**Category**: cs.DC  
**Published**: 2026-06-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2606.14712v1  

#### Abstract
The rapid growth of Internet of Things (IoT) devices has led to large-scale continuous data streams that require realtime processing. Traditional cloud-centric architectures fail to meet low-latency and bandwidth efficiency requirements due to network delays and high data transmission overhead. This...

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
